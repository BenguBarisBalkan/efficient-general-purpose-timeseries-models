"""
TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis
ICLR 2025  —  Wang et al.

Architecture overview
─────────────────────
Input Projection
  1. Multi-scale downsampling  (Conv1d stride=2, M+1 scales)
  2. Channel mixing            (variate-wise self-attention at coarsest scale)
  3. Embedding                 (TokenEmbedding per scale → d_model)

L × MixerBlock
  a. Multi-Resolution Time Imaging  (MRTI)  — FFT → top-K periods → 2D images
  b. Time Image Decomposition        (TID)   — dual-axis attention → seasonal + trend
  c. Multi-Scale Mixing              (MSM)   — Conv2D bottom-up season / TransConv top-down trend
  d. Multi-Resolution Mixing         (MRM)   — amplitude-weighted sum across K periods

Output Projection
  — one linear head per scale → ensemble (sum)
  — task-adaptive: forecast / imputation / anomaly-detection / classification
"""

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow import of shared layers from the parent TimeMixer directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from layers.StandardNorm import Normalize


# ══════════════════════════════════════════════════════════════════════════════
# Helper: FFT-based period extraction
# ══════════════════════════════════════════════════════════════════════════════

def _sanitize(x: torch.Tensor) -> torch.Tensor:
    """
    Replace NaN/inf with 0 (PyTorch 1.7-compatible; no torch.nan_to_num).
    torch.where routes gradient only to the finite branch, so sanitized
    elements get 0 gradient — this stops a diverged activation from
    backpropagating NaNs into the weights and killing the whole run.
    """
    return torch.where(torch.isfinite(x), x, torch.zeros_like(x))


def _rfft_amplitude(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the one-sided amplitude spectrum along dim=1 (time dimension).
    Compatible with both PyTorch 1.7.x (old API) and ≥1.8 (new API).

    x   : (B, T, d_model)
    out : (B, T//2+1, d_model)
    """
    if hasattr(torch.fft, "rfft"):
        # PyTorch >= 1.8 new API
        return torch.fft.rfft(x, dim=1).abs()
    else:
        # PyTorch 1.7.x: torch.rfft applies over last signal_ndim dims
        # Permute T to last position: (B, d_model, T)
        x_perm = x.permute(0, 2, 1).contiguous()          # (B, d, T)
        out = torch.rfft(x_perm, signal_ndim=1)           # (B, d, T//2+1, 2)
        amp = (out[..., 0] ** 2 + out[..., 1] ** 2).sqrt()  # (B, d, T//2+1)
        return amp.permute(0, 2, 1)                        # (B, T//2+1, d)


def _top_k_periods(x_M: torch.Tensor, top_k: int):
    """
    FFT on the coarsest-scale embedding to find the K most energetic periods.

    x_M : (B, T_M, d_model)
    Returns
        amp_weights : (K,)  — softmax-normalised amplitudes (differentiable)
        periods     : (K,)  — integer period lengths (not differentiable)
        K           : int   — actual number of periods (≤ top_k)
    """
    B, T_M, d = x_M.shape
    amp_spec = _rfft_amplitude(x_M)              # (B, T_M//2+1, d)
    amp = amp_spec.mean(dim=(0, 2))              # (T_M//2+1,)
    # Replace any NaN/Inf (from large embeddings) before ranking.
    # torch.nan_to_num requires PyTorch >= 1.8; use manual masking for 1.7 compat.
    amp = torch.where(torch.isfinite(amp), amp, torch.zeros_like(amp))
    amp[0] = 0.0                                 # suppress DC component

    n_freq = amp.shape[0] - 1                    # usable non-DC bins
    K = min(top_k, max(1, n_freq))
    _, top_idx = torch.topk(amp[1:], K)          # 0-based index into non-DC bins
    freq_idx = top_idx + 1                       # 1-based frequency index

    periods_raw = T_M // freq_idx.clamp(min=1)   # integer period lengths
    periods = periods_raw.clamp(min=2)            # enforce minimum period of 2

    # Clamp before softmax: prevents softmax([inf,inf,…]) = [nan,nan,…].
    amp_raw = amp[freq_idx].clamp(min=0.0)
    amp_weights = torch.softmax(amp_raw, dim=0)  # (K,)
    return amp_weights, periods, K


# ══════════════════════════════════════════════════════════════════════════════
# Helper: 1D ↔ 2D reshape
# ══════════════════════════════════════════════════════════════════════════════

def _to_image(x: torch.Tensor, period: int):
    """
    Fold a 1-D time series into a 2-D image along dominant period.

    x      : (B, T_m, d_model)
    period : int  (number of rows, p_k)
    Returns  (B, p_k, f_k, d_model)  where f_k = ceil(T_m / p_k)
    """
    B, T, d = x.shape
    f = math.ceil(T / period)
    pad = f * period - T
    if pad > 0:
        x = F.pad(x, (0, 0, 0, pad))            # pad time dim at the end
    # (B, f*p, d)  →  (B, f, p, d)  →  (B, p, f, d)
    return x.view(B, f, period, d).permute(0, 2, 1, 3).contiguous()


def _from_image(img: torch.Tensor, orig_len: int):
    """
    Unfold a 2-D image back into a 1-D time series, trimming padding.

    img      : (B, p_k, f_k, d_model)
    orig_len : int
    Returns    (B, orig_len, d_model)
    """
    B, p, f, d = img.shape
    return img.permute(0, 2, 1, 3).contiguous().view(B, p * f, d)[:, :orig_len, :]


# ══════════════════════════════════════════════════════════════════════════════
# 1. Channel Mixer
# ══════════════════════════════════════════════════════════════════════════════

class ChannelMixer(nn.Module):
    """
    Variate-wise self-attention applied at the coarsest temporal scale x_M.
    Captures cross-variate dependencies *before* embedding.

    Input / Output : (B, T_M, C)
    """

    def __init__(self, T_M: int, C: int, d_attn: int, dropout: float = 0.1):
        super().__init__()
        self.scale = math.sqrt(d_attn)
        self.q = nn.Linear(T_M, d_attn)
        self.k = nn.Linear(T_M, d_attn)
        self.v = nn.Linear(T_M, d_attn)
        self.out = nn.Linear(d_attn, T_M)
        self.norm = nn.LayerNorm(T_M)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T_M, C)
        xT = x.permute(0, 2, 1)                         # (B, C, T_M)
        Q, K, V = self.q(xT), self.k(xT), self.v(xT)   # each (B, C, d_attn)
        scores = (Q @ K.transpose(-2, -1) / self.scale).clamp(min=-1e4, max=1e4)
        attn = torch.softmax(scores, dim=-1)              # (B, C, C)
        out = self.dropout(attn) @ V                     # (B, C, d_attn)
        out = self.dropout(self.out(out))                # (B, C, T_M)
        return self.norm(xT + out).permute(0, 2, 1)      # (B, T_M, C)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Time Image Decomposition (TID) — dual-axis attention
# ══════════════════════════════════════════════════════════════════════════════

class _AxisAttention(nn.Module):
    """Single-head dot-product attention for one axis of a time image."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.scale = math.sqrt(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (Batch, SeqLen, d_model)
        residual = x
        Q, K, V = self.qkv(x).chunk(3, dim=-1)
        scores = Q @ K.transpose(-2, -1) / self.scale
        # Clamp scores to prevent softmax(inf) = nan on unstable early training steps.
        scores = scores.clamp(min=-1e4, max=1e4)
        attn = torch.softmax(scores, dim=-1)
        out = self.dropout(attn) @ V
        out = self.dropout(self.out(out))
        return self.norm(residual + out)


class TimeImageDecomposition(nn.Module):
    """
    Dual-axis attention on a 2-D time image.

    • Column-axis attention  (over f_k dimension, period folded into batch)
      → captures seasonal patterns within a period repetition.
    • Row-axis attention     (over p_k dimension, frequency folded into batch)
      → captures trend across period repetitions.

    Input / Output : (B, p_k, f_k, d_model)  →  seasonal image, trend image
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.col_attn = _AxisAttention(d_model, dropout)  # seasonality
        self.row_attn = _AxisAttention(d_model, dropout)  # trend

    def forward(self, z: torch.Tensor):
        B, p, f, d = z.shape

        # ── Column-axis: fold p into batch, attend over f ─────────────────
        s = self.col_attn(z.reshape(B * p, f, d)).reshape(B, p, f, d)

        # ── Row-axis: fold f into batch, attend over p ────────────────────
        t = self.row_attn(
            z.permute(0, 2, 1, 3).reshape(B * f, p, d)
        ).reshape(B, f, p, d).permute(0, 2, 1, 3).contiguous()

        return s, t


# ══════════════════════════════════════════════════════════════════════════════
# 3. Multi-Scale Mixing (MSM) — Conv-based
# ══════════════════════════════════════════════════════════════════════════════

class _SeasonConv(nn.Module):
    """
    Bottom-up seasonal mixing: reduce frequency dimension by ~2×.
    Two Conv1d layers (temporal stride=2 then 1×1), applied along the f axis.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, target_f: int) -> torch.Tensor:
        # x : (B, p, f_in, d)
        B, p, f, d = x.shape
        h = x.reshape(B * p, f, d).permute(0, 2, 1)   # (B*p, d, f)
        h = self.conv(h)                               # (B*p, d, ~f/2)
        if h.shape[-1] != target_f:
            h = F.interpolate(h, size=target_f, mode="linear", align_corners=False)
        return h.permute(0, 2, 1).reshape(B, p, target_f, d)


class _TrendTransConv(nn.Module):
    """
    Top-down trend mixing: expand frequency dimension by ~2×.
    Two ConvTranspose1d layers (temporal stride=2 then 1×1), along the f axis.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose1d(d_model, d_model, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, target_f: int) -> torch.Tensor:
        # x : (B, p, f_in, d)
        B, p, f, d = x.shape
        h = x.reshape(B * p, f, d).permute(0, 2, 1)   # (B*p, d, f)
        h = self.conv(h)                               # (B*p, d, ~2f)
        if h.shape[-1] != target_f:
            h = F.interpolate(h, size=target_f, mode="linear", align_corners=False)
        return h.permute(0, 2, 1).reshape(B, p, target_f, d)


# ══════════════════════════════════════════════════════════════════════════════
# 4. MixerBlock  (MRTI → TID → MSM → MRM)
# ══════════════════════════════════════════════════════════════════════════════

class MixerBlock(nn.Module):
    """
    One complete MixerBlock as described in TimeMixer++ Section 3.2.

    Forward pass
    ────────────
    x_list : list of M+1 tensors (B, T_m, d_model), finest → coarsest

    Returns the same list structure with updated representations.
    """

    def __init__(self, d_model: int, top_k: int, n_scales: int, dropout: float = 0.1):
        super().__init__()
        self.top_k = top_k
        self.M = n_scales - 1          # number of extra (downsampled) scales

        self.tid = TimeImageDecomposition(d_model, dropout)

        # One Conv per scale transition (M transitions for M+1 scales)
        self.season_convs = nn.ModuleList([_SeasonConv(d_model) for _ in range(self.M)])
        self.trend_convs  = nn.ModuleList([_TrendTransConv(d_model) for _ in range(self.M)])

        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_list):
        M = self.M
        T_list = [x.shape[1] for x in x_list]

        # ── MRTI ──────────────────────────────────────────────────────────
        amp_w, periods, K = _top_k_periods(x_list[M], self.top_k)

        # Build images[m][k] : (B, p_k, f_{m,k}, d_model)
        images = []
        for x_m in x_list:
            row = []
            for k in range(K):
                p_k = int(periods[k].item())
                row.append(_to_image(x_m, p_k))
            images.append(row)

        # ── TID ───────────────────────────────────────────────────────────
        s_imgs = [[None] * K for _ in range(M + 1)]
        t_imgs = [[None] * K for _ in range(M + 1)]
        for m in range(M + 1):
            for k in range(K):
                s, t = self.tid(images[m][k])
                s_imgs[m][k] = s
                t_imgs[m][k] = t

        # ── MSM: bottom-up seasonal (fine → coarse) ───────────────────────
        for m in range(1, M + 1):
            conv = self.season_convs[m - 1]
            for k in range(K):
                tgt_f = s_imgs[m][k].shape[2]
                s_imgs[m][k] = s_imgs[m][k] + conv(s_imgs[m - 1][k], tgt_f)

        # ── MSM: top-down trend (coarse → fine) ───────────────────────────
        for m in range(M - 1, -1, -1):
            conv = self.trend_convs[M - 1 - m]
            for k in range(K):
                tgt_f = t_imgs[m][k].shape[2]
                t_imgs[m][k] = t_imgs[m][k] + conv(t_imgs[m + 1][k], tgt_f)

        # ── MRM: amplitude-weighted aggregation across K periods ──────────
        out_list = []
        for m in range(M + 1):
            weighted = None
            for k in range(K):
                combined = s_imgs[m][k] + t_imgs[m][k]
                ts = _from_image(combined, T_list[m])      # (B, T_m, d)
                weighted = ts * amp_w[k] if weighted is None else weighted + ts * amp_w[k]

            # Residual connection + feedforward
            out = self.norm(x_list[m] + self.ff(weighted))
            out_list.append(out)

        return out_list


# ══════════════════════════════════════════════════════════════════════════════
# 5. Full Model
# ══════════════════════════════════════════════════════════════════════════════

class Model(nn.Module):
    """
    TimeMixer++  —  universal time-series pattern machine.

    Supported tasks  (set via configs.task_name):
      long_term_forecast / short_term_forecast
      imputation
      anomaly_detection
      classification
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name   = configs.task_name
        self.seq_len     = configs.seq_len
        self.pred_len    = configs.pred_len
        self.enc_in      = configs.enc_in
        self.c_out       = configs.c_out
        self.d_model     = configs.d_model
        self.M           = configs.down_sampling_layers   # extra downsampled scales
        self.n_scales    = self.M + 1
        self.top_k       = configs.top_k
        self.e_layers    = configs.e_layers
        self.dropout_p   = configs.dropout

        # ── Multi-scale downsampling convolutions ──────────────────────────
        # Depthwise Conv1d (stride=2) keeps channels independent during downsampling.
        self.down_convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.enc_in,
                out_channels=self.enc_in,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=self.enc_in,   # depthwise → no cross-channel mixing here
            )
            for _ in range(self.M)
        ])

        # ── Channel Mixer (coarsest scale, before embedding) ──────────────
        T_M = self._coarsest_len()
        d_attn = max(self.d_model, 8)           # attention dimension for channel mixer
        self.channel_mixer = ChannelMixer(T_M, self.enc_in, d_attn, self.dropout_p)

        # ── Per-scale embedding (TokenEmbedding: Conv1d → d_model) ────────
        from layers.Embed import DataEmbedding_wo_pos
        self.embeddings = nn.ModuleList([
            DataEmbedding_wo_pos(self.enc_in, self.d_model, configs.embed, configs.freq, self.dropout_p)
            for _ in range(self.n_scales)
        ])

        # ── MixerBlocks ────────────────────────────────────────────────────
        self.mixer_blocks = nn.ModuleList([
            MixerBlock(self.d_model, self.top_k, self.n_scales, self.dropout_p)
            for _ in range(self.e_layers)
        ])

        # ── Normalization (RevIN) per scale ────────────────────────────────
        self.normalize_layers = nn.ModuleList([
            Normalize(self.enc_in, affine=True, non_norm=(configs.use_norm == 0))
            for _ in range(self.n_scales)
        ])

        # ── Output heads ───────────────────────────────────────────────────
        if self.task_name in ("long_term_forecast", "short_term_forecast"):
            # One temporal projection per scale  (T_m → pred_len)
            self.predict_layers = nn.ModuleList([
                nn.Linear(self._scale_len(m), self.pred_len)
                for m in range(self.n_scales)
            ])
            # Shared channel projection  (d_model → c_out)
            self.projection = nn.Linear(self.d_model, self.c_out)

        elif self.task_name in ("imputation", "anomaly_detection"):
            self.projection = nn.Linear(self.d_model, self.c_out)

        elif self.task_name == "classification":
            self.act       = nn.GELU()
            self.dropout   = nn.Dropout(self.dropout_p)
            self.projection = nn.Linear(self.d_model * self.seq_len, configs.num_class)

    # ── helpers ─────────────────────────────────────────────────────────────

    def _scale_len(self, m: int) -> int:
        """Return T_m = seq_len // 2^m."""
        return self.seq_len // (2 ** m)

    def _coarsest_len(self) -> int:
        return self._scale_len(self.M)

    def _downsample(self, x_enc):
        """
        Build multi-scale list from x_enc.
        x_enc : (B, T, C)
        Returns list of M+1 tensors finest → coarsest.
        """
        scales = [x_enc]
        cur = x_enc.permute(0, 2, 1)   # (B, C, T)
        for conv in self.down_convs:
            cur = conv(cur)
            scales.append(cur.permute(0, 2, 1))
        return scales

    # ── forward helpers ─────────────────────────────────────────────────────

    def _encode(self, x_enc, x_mark_enc):
        """
        Full encoder: downsample → normalise → channel mix → embed → MixerBlocks.
        Returns enc_out_list (M+1 tensors of shape (B, T_m, d_model)).
        """
        # Downsample
        scale_inputs = self._downsample(x_enc)   # list of (B, T_m, C)

        # Normalise each scale independently
        for m in range(self.n_scales):
            scale_inputs[m] = self.normalize_layers[m](scale_inputs[m], "norm")

        # Channel mixing at coarsest scale
        scale_inputs[self.M] = self.channel_mixer(scale_inputs[self.M])

        # Embed
        # x_mark_enc is either None, a 3-D time-feature tensor (B,T,d_mark),
        # or a 2-D padding mask (B,T) used only in classification — pass None in the latter case.
        use_mark = x_mark_enc is not None and x_mark_enc.dim() == 3
        enc_out_list = []
        for m, x_m in enumerate(scale_inputs):
            x_mark = x_mark_enc[:, :: (2 ** m), :] if use_mark else None
            enc_out_list.append(self.embeddings[m](x_m, x_mark))

        # MixerBlocks
        for block in self.mixer_blocks:
            enc_out_list = block(enc_out_list)

        # Sanitize encoder features: replace any NaN/inf with 0 so a single
        # diverged activation cannot poison the heads or (via backprop) the
        # weights. torch.where gives 0 gradient to the sanitized elements,
        # which breaks the NaN cascade that otherwise kills the whole run.
        enc_out_list = [_sanitize(e) for e in enc_out_list]

        return enc_out_list

    # ── task heads ──────────────────────────────────────────────────────────

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B = x_enc.shape[0]
        enc_out_list = self._encode(x_enc, x_mark_enc)

        # Per-scale prediction  →  sum ensemble
        dec_out = None
        for m, enc_out in enumerate(enc_out_list):
            # enc_out : (B, T_m, d_model)
            pred = self.predict_layers[m](
                enc_out.permute(0, 2, 1)          # (B, d_model, T_m)
            ).permute(0, 2, 1)                    # (B, pred_len, d_model)
            pred = self.projection(pred)           # (B, pred_len, c_out)
            dec_out = pred if dec_out is None else dec_out + pred

        # Denormalise using finest-scale statistics
        dec_out = self.normalize_layers[0](dec_out, "denorm")
        return dec_out

    def imputation(self, x_enc, x_mark_enc, mask):
        # Simple mean/std normalisation for imputation (mask-aware)
        means = (x_enc * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(
            (x_enc * x_enc * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1) + 1e-5
        ).unsqueeze(1).detach()
        x_enc = x_enc / stdev

        enc_out_list = self._encode(x_enc, x_mark_enc)
        dec_out = self.projection(enc_out_list[0])     # finest scale: (B, T, c_out)
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).expand_as(dec_out)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).expand_as(dec_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        B, T, N = x_enc.shape
        enc_out_list = self._encode(x_enc, None)
        dec_out = self.projection(enc_out_list[0])     # (B, T, c_out)
        dec_out = self.normalize_layers[0](dec_out, "denorm")
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        enc_out_list = self._encode(x_enc, x_mark_enc)
        out = enc_out_list[0]                          # finest scale: (B, T, d_model)
        out = self.act(out)
        out = self.dropout(out)
        if x_mark_enc is not None:
            out = out * x_mark_enc.unsqueeze(-1)
        out = out.reshape(out.shape[0], -1)
        return self.projection(out)

    # ── main forward ────────────────────────────────────────────────────────

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ("long_term_forecast", "short_term_forecast"):
            out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        elif self.task_name == "imputation":
            out = self.imputation(x_enc, x_mark_enc, mask)
        elif self.task_name == "anomaly_detection":
            out = self.anomaly_detection(x_enc)
        elif self.task_name == "classification":
            out = self.classification(x_enc, x_mark_enc)
        else:
            raise ValueError(f"Unknown task: {self.task_name}")
        # Final safety net: guarantee the model never returns NaN/inf, so the
        # loss (and therefore backprop) stays finite even if denorm overflowed.
        return _sanitize(out)
