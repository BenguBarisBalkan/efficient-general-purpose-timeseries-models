"""SparseTSFPlus — SparseTSF with cheap units grafted on from TimeMixer / TimeMixer++.

Motivation. SparseTSF matches TimeMixer's long-term forecasting accuracy at ~1/1800th the
parameters, but the 8-task sweep exposed three gaps with three *different* causes:

  * PEMS short-term  (+96% MAE vs TimeMixer) -- partly cross-channel correlation, which this
    model deliberately does NOT address, and partly raw capacity: at seq_len=96, pred_len=12,
    period_len=12 the whole forecast head is nn.Linear(8, 1) -- eight parameters.
  * classification   (-15.0 accuracy points) -- an ad-hoc flatten-linear head, plus a
    padded-mean bug.
  * imputation       (+52% on ETTm1)         -- a phase-blind reconstruction linear that
    cannot distinguish a masked zero from an observed one, hence a gap that grows
    monotonically with mask_rate.

Design invariant. Every unit is toggleable and defaults to off, residual units are
zero-initialised, and **with all units off this model is numerically identical to
SparseTSF** (proved bitwise by benchmarks/tools/assert_sparsetsf_plus_equiv.py). For the
forecasting path with a vanilla config we do not merely reproduce the reference -- we
delegate to the untouched vendored core, so reproduction is true by construction.

The forecasting and reconstruction backbone stays strictly channel-independent: no unit
here introduces a parameter that depends on `enc_in`. That protects the 6/6 zero-shot
transfer wins and keeps checkpoints portable across datasets with different channel counts
(`benchmarks/run_zeroshot_benchmarks.py` literally copies a checkpoint between datasets).
The classification head is necessarily channel-coupled -- it is a readout over all
channels, not backbone mixing, so it is not what "no channel mixing" rules out.

Per-task interface (identical to models/SparseTSF.py, which the exp loops already expect):

    long/short forecast : model(x_enc, x_mark, x_dec, x_mark_dec)   -> (B, pred_len, C)
    imputation          : model(inp, x_mark, None, None, mask)      -> (B, seq_len, C)
    anomaly_detection   : model(x, None, None, None)                -> (B, seq_len, C)
    classification      : model(x, padding_mask, None, None)        -> (B, num_class)
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from layers.Autoformer_EncDec import series_decomp
from layers.StandardNorm import Normalize
from implementations.sparsetsf.model import Model as SparseTSFCore
from implementations.sparsetsf_plus.units import (
    PhaseMix,
    UnitConfig,
    count_trainable,
    fold_periods,
    masked_stats,
    odd,
    unfold_periods,
)

_FORECAST_TASKS = ("long_term_forecast", "short_term_forecast")
_RECON_TASKS = ("imputation", "anomaly_detection")


class Model(nn.Module):
    def __init__(self, configs, force_reimpl=False):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.units = UnitConfig.from_configs(configs)
        self.pad_fold = bool(self.units.pad_fold)

        self._validate_divisibility()

        # Vanilla forecasting delegates to the untouched vendored core, so the reference
        # point is reproduced by construction rather than by argument. `force_reimpl` is
        # used only by the equivalence checker to exercise the re-implemented path.
        self.delegate_core = (
            self.task_name in _FORECAST_TASKS
            and self.units.is_vanilla
            and not force_reimpl
        )
        if self.delegate_core:
            self.core = SparseTSFCore(configs)
            self._announce()
            return

        # Base parameters are constructed in the vendored core's order (conv first, then
        # the linear) so under a fixed seed they draw from the same RNG stream as
        # SparseTSF's. Unit parameters are built afterwards.
        self.conv1d = self._make_conv(self.period_len)

        # Ceiling division: identical to SparseTSF's floor division whenever period_len
        # divides the length (the only case stock SparseTSF allows), but correct for the
        # phase-aligned padded fold, which yields ceil(L / p) segments.
        if self.task_name in _FORECAST_TASKS:
            self.seg_num_x = -(-self.seq_len // self.period_len)
            self.seg_num_y = -(-self.pred_len // self.period_len)
            self.linear = nn.Linear(self.seg_num_x, self.seg_num_y,
                                    bias=bool(self.units.linear_bias))
        elif self.task_name in _RECON_TASKS:
            self.seg_num = -(-self.seq_len // self.period_len)
            self.recon_linear = nn.Linear(self.seg_num, self.seg_num,
                                          bias=bool(self.units.linear_bias))
        elif self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(getattr(configs, "dropout", 0.1))
            self._build_cls_head(configs)
        else:
            raise ValueError(
                "SparseTSFPlus: unsupported task_name '{}'. Supported: {}".format(
                    self.task_name, _FORECAST_TASKS + _RECON_TASKS + ("classification",)
                )
            )

        self._build_units()
        self._announce()

    # ── construction helpers ─────────────────────────────────────────────────────────
    def _make_conv(self, period):
        """SparseTSF's depthwise aggregator: single in/out channel, bias-free, residual."""
        return nn.Conv1d(
            in_channels=1, out_channels=1,
            kernel_size=1 + 2 * (period // 2),
            stride=1, padding=period // 2,
            padding_mode="zeros", bias=False,
        )

    def _build_cls_head(self, configs):
        head = self.units.cls_head
        n_class = configs.num_class
        if head in ("flat", "flat_fixed"):
            # flat        : upstream behaviour, padded-mean bug included (kept for the
            #               ablation baseline row)
            # flat_fixed  : identical head, mask-aware mean -- isolates the bug fix from
            #               the head redesign
            self.projection = nn.Linear(self.enc_in * self.seq_len, n_class)
        elif head == "stats":
            # Statistics pooling over time: mean/std/max per channel. Removes the
            # seq_len scaling and adds translation invariance. On PEMS-SF this is
            # 20,230 params against the flat head's 971,397.
            #
            # MEASURED: this loses accuracy on shape-driven tasks, badly. Handwriting
            # (pen trajectories, 26 classes) drops to 3.41% -- the 3.85% chance floor --
            # because three statistics per channel cannot encode a trajectory. Use
            # `segpool` when temporal ORDER carries the signal.
            self.projection = nn.Linear(3 * self.enc_in, n_class)
        elif head == "segpool":
            # Middle ground between `flat` (no invariance, scales with seq_len) and
            # `stats` (throws temporal order away): mask-aware mean-pool into a fixed
            # number of time segments, so coarse ORDER survives at a cost independent of
            # seq_len. Handwriting: 8 segments -> 650 params vs the flat head's 11,882.
            self.n_segments = max(1, int(self.units.cls_segments))
            self.projection = nn.Linear(self.n_segments * self.enc_in, n_class)
        else:
            raise ValueError(
                "SparseTSFPlus: unknown --stsf_cls_head '{}' "
                "(flat | flat_fixed | stats | segpool)".format(head))

    def _build_units(self):
        u = self.units

        # U1 RevIN. Reuses layers/StandardNorm.py. It is stateful (stashes mean/stdev for
        # denorm), so norm/denorm must pair inside one forward and the instance is never
        # shared -- which is why TimeMixer++ builds one per scale.
        #
        # MEASURED CAVEAT, and it is a property of SparseTSF rather than of this code:
        # `--stsf_revin 1` (mean+std) is a mathematical NO-OP here. SparseTSF's forecast
        # map is linear and entirely bias-free, hence positively homogeneous:
        #     f(x / sigma) * sigma == f(x)
        # so dividing by a per-window, per-channel scalar and multiplying it back cancels
        # exactly. Verified: max|diff| vs mean-only is 1.5e-05 on inputs scaled to ~50,
        # i.e. float32 rounding. It changed nothing on PEMS08 or Weather (identical to 6
        # decimal places on both validation and test).
        #
        # Std scaling can only matter once homogeneity is broken. Live variants:
        #   --stsf_revin 2        subtract_last instead of the mean (max|diff| 1.1e+02)
        #   --stsf_linear_bias 1  a bias breaks homogeneity (max|diff| 8.4e+00)
        #   --stsf_revin_affine 1 learnable affine, but costs 2*enc_in and breaks zero-shot
        # Do not report `--stsf_revin 1` alone as a unit: it cannot do anything.
        self.normalize = None
        if u.revin:
            self.normalize = Normalize(
                self.enc_in,
                affine=bool(u.revin_affine),
                subtract_last=(int(u.revin) == 2),
            )

        # U2 phase-axis mixing (the Kronecker factorisation; see units.PhaseMix).
        self.phase_mix = None
        if u.phase_mix != "off":
            self.phase_mix = PhaseMix(
                self.period_len, mode=u.phase_mix,
                rank=int(u.phase_rank), kernel=int(u.phase_kernel),
            )

        # U3 multi-period ensemble. Branch 0 is the base path above; each extra period
        # gets its own conv + linear behind a zero-initialised gate, so switching the unit
        # on starts from exactly the base model.
        self.extra_convs = nn.ModuleList()
        self.extra_linears = nn.ModuleList()
        self.extra_periods = []
        extra = [p for p in u.extra_periods() if p != self.period_len]
        if extra and self.task_name in _FORECAST_TASKS + _RECON_TASKS:
            for p in extra:
                if not self.pad_fold:
                    if self.seq_len % p != 0:
                        raise ValueError(
                            "SparseTSFPlus: extra period {} must divide seq_len {} "
                            "(or pass --stsf_pad_fold 1)".format(p, self.seq_len))
                    if self.task_name in _FORECAST_TASKS and self.pred_len % p != 0:
                        raise ValueError(
                            "SparseTSFPlus: extra period {} must divide pred_len {} "
                            "(or pass --stsf_pad_fold 1)".format(p, self.pred_len))
                n_in = -(-self.seq_len // p)
                n_out = n_in if self.task_name in _RECON_TASKS else -(-self.pred_len // p)
                self.extra_convs.append(self._make_conv(p))
                self.extra_linears.append(nn.Linear(n_in, n_out, bias=False))
                self.extra_periods.append(p)
            self.extra_gates = nn.Parameter(torch.zeros(len(self.extra_periods)))

        # U4 trend/season split. `free` is parameter-free (trend persists its last value);
        # `linear` gives the trend its own cross-period head.
        self.decomp = None
        self.trend_linear = None
        if u.decomp != "off" and self.task_name in _FORECAST_TASKS:
            self.decomp = series_decomp(odd(u.decomp_kernel))
            if u.decomp == "linear":
                self.trend_linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)
            elif u.decomp != "free":
                raise ValueError(
                    "SparseTSFPlus: unknown --stsf_decomp '{}' (off | free | linear)".format(u.decomp))

        # U7 leave-one-out reconstruction. `recon_linear` plus the conv's residual makes
        # the identity map exactly representable, so MSE reconstruction training drifts
        # toward it and anomalies get reconstructed faithfully -- exactly SMD's
        # recall-limited signature. Zeroing the diagonal costs nothing.
        if u.diag_mask and self.task_name in _RECON_TASKS:
            self.register_buffer(
                "diag_keep", 1.0 - torch.eye(self.seg_num), persistent=False)
        else:
            self.diag_keep = None

    def _announce(self):
        # Label deliberately avoids the substring 'mse' -- every benchmark runner regexes
        # lowercase mse:/mae: and takes the LAST match, so a colliding label would
        # silently corrupt every existing results table.
        print("trainable_params:{}".format(count_trainable(self)))
        print("units:{}".format(self.units.describe()))

    def _validate_divisibility(self):
        p = self.period_len
        if self.pad_fold:
            return  # phase-aligned padding handles any period
        if self.task_name in _FORECAST_TASKS:
            if self.seq_len % p != 0:
                raise ValueError(
                    "SparseTSFPlus: seq_len ({}) must be divisible by period_len ({}); "
                    "pass --stsf_pad_fold 1 to allow padding".format(self.seq_len, p))
            if self.pred_len % p != 0:
                raise ValueError(
                    "SparseTSFPlus: pred_len ({}) must be divisible by period_len ({}); "
                    "pass --stsf_pad_fold 1 to allow padding".format(self.pred_len, p))
        elif self.task_name in _RECON_TASKS:
            if self.seq_len % p != 0:
                raise ValueError(
                    "SparseTSFPlus ({}): seq_len ({}) must be divisible by period_len "
                    "({})".format(self.task_name, self.seq_len, p))
        # classification: no divisibility requirement (the flat head does not reshape).

    # ── shared front-end ─────────────────────────────────────────────────────────────
    def _aggregate(self, x, conv=None, mask=None):
        """(B, C, S) -> depthwise conv over time + residual -> (B, C, S).

        With U5 (`--stsf_mask_conv 1`) the convolution is renormalised by how much of each
        window was actually observed, so a masked zero no longer drags the aggregate toward
        zero. That is the mechanism behind imputation's monotonic degradation with
        mask_rate: at mask 0.5 half the receptive field is literal zeros.
        """
        conv = self.conv1d if conv is None else conv
        b, c, s = x.shape
        flat = x.reshape(-1, 1, s)
        num = conv(flat)
        if mask is not None and self.units.mask_conv:
            m = mask.reshape(-1, 1, s)
            ones = torch.ones_like(conv.weight)
            cnt = F.conv1d(m, ones, padding=conv.padding[0])
            num = num * (conv.weight.numel() / torch.clamp(cnt, min=1.0))
        return num.reshape(b, c, s) + x

    def _cross_period(self, x_bcs, period, conv, linear, out_len, phase_mix=None,
                      weight=None, mask_bcs=None):
        """One fold -> (optional phase mix) -> cross-period linear -> unfold branch."""
        b, c, s = x_bcs.shape
        agg = self._aggregate(x_bcs, conv=conv, mask=mask_bcs)
        img, _ = fold_periods(agg.reshape(-1, s), period, self.pad_fold)
        if phase_mix is not None:
            img = phase_mix(img)
        if weight is not None:
            y = F.linear(img, weight, linear.bias)
        else:
            y = linear(img)
        return unfold_periods(y, out_len).reshape(b, c, out_len)

    # ── normalisation ────────────────────────────────────────────────────────────────
    def _norm(self, x, mask=None):
        """-> (normalised x, denorm closure). Default is SparseTSF's mean-only removal."""
        if self.normalize is not None and mask is None:
            return self.normalize(x, "norm"), lambda y: self.normalize(y, "denorm")
        use_std = bool(self.units.revin)
        mean, std = masked_stats(x, mask, use_std=use_std)
        if use_std:
            return (x - mean) / std, lambda y: y * std + mean
        return x - mean, lambda y: y + mean

    # ── task paths ───────────────────────────────────────────────────────────────────
    def _forecast(self, x_enc):
        x_norm, denorm = self._norm(x_enc)

        trend_out = None
        if self.decomp is not None:
            season, trend = self.decomp(x_norm)
            if self.trend_linear is not None:
                trend_out = self._cross_period(
                    trend.permute(0, 2, 1), self.period_len, self.conv1d,
                    self.trend_linear, self.pred_len)
            else:
                # `free`: persist the trend's last value over the horizon (0 params).
                trend_out = trend[:, -1:, :].permute(0, 2, 1).expand(-1, -1, self.pred_len)
            x_norm = season

        x = x_norm.permute(0, 2, 1)                                   # (B, C, S)
        y = self._cross_period(x, self.period_len, self.conv1d, self.linear,
                               self.pred_len, phase_mix=self.phase_mix)
        for i, p in enumerate(self.extra_periods):
            y = y + self.extra_gates[i] * self._cross_period(
                x, p, self.extra_convs[i], self.extra_linears[i], self.pred_len)
        if trend_out is not None:
            y = y + trend_out
        return denorm(y.permute(0, 2, 1))

    def _reconstruct(self, x_norm, denorm, mask=None):
        x = x_norm.permute(0, 2, 1)                                   # (B, C, S)
        m = mask.permute(0, 2, 1) if mask is not None else None
        weight = None
        if self.diag_keep is not None:
            weight = self.recon_linear.weight * self.diag_keep
        y = self._cross_period(x, self.period_len, self.conv1d, self.recon_linear,
                               self.seq_len, phase_mix=self.phase_mix,
                               weight=weight, mask_bcs=m)
        for i, p in enumerate(self.extra_periods):
            y = y + self.extra_gates[i] * self._cross_period(
                x, p, self.extra_convs[i], self.extra_linears[i], self.seq_len,
                mask_bcs=m)
        return denorm(y.permute(0, 2, 1))

    def _imputation(self, x_enc, mask):
        x_norm, denorm = self._norm(x_enc, mask=mask)
        if mask is not None:
            x_norm = x_norm * mask
        out = self._reconstruct(x_norm, denorm, mask=mask)

        # U5 refinement: fill the holes with pass 1, recompute statistics on the now
        # complete series, and refine with the SAME weights (0 extra parameters). The
        # imputation loss is scored on masked positions only (exp/exp_imputation.py:78),
        # so this targets the metric directly.
        if mask is not None and int(self.units.impute_passes) > 1:
            for _ in range(int(self.units.impute_passes) - 1):
                filled = x_enc * mask + out * (1.0 - mask)
                x2, denorm2 = self._norm(filled)
                out = self._reconstruct(x2, denorm2)
        return out

    def _anomaly(self, x_enc):
        x_norm, denorm = self._norm(x_enc)
        return self._reconstruct(x_norm, denorm)

    def _classification(self, x_enc, padding_mask):
        head = self.units.cls_head
        if head == "flat":
            # Upstream behaviour, padded-mean bug included: the mean is taken over the
            # full zero-padded window and padding_mask is applied only afterwards.
            x = (x_enc - x_enc.mean(dim=1, keepdim=True)).permute(0, 2, 1)
            x = self._aggregate(x)
            if padding_mask is not None:
                x = x * padding_mask.unsqueeze(1)
            x = self.dropout(self.act(x))
            return self.projection(x.reshape(x.shape[0], -1))

        # UEA windows are zero-padded to max_seq_len (1751 on EthanolConcentration), so
        # the unmasked mean is badly biased. Centre on observed steps only.
        m = None if padding_mask is None else padding_mask.unsqueeze(-1)
        mean, _ = masked_stats(x_enc, m)
        x = (x_enc - mean)
        if m is not None:
            x = x * m
        x = self._aggregate(x.permute(0, 2, 1))                       # (B, C, S)
        if padding_mask is not None:
            x = x * padding_mask.unsqueeze(1)
        x = self.dropout(self.act(x))

        if head == "flat_fixed":
            return self.projection(x.reshape(x.shape[0], -1))

        if head == "segpool":
            # Mask-aware segment means: pool signal and mask with the same kernel, then
            # divide, so padded steps contribute nothing. Segments that are entirely
            # padding come out as 0.
            m1 = None if padding_mask is None else padding_mask.unsqueeze(1)
            num = F.adaptive_avg_pool1d(x if m1 is None else x * m1, self.n_segments)
            if m1 is None:
                seg = num
            else:
                den = F.adaptive_avg_pool1d(m1.expand_as(x), self.n_segments)
                seg = num / torch.clamp(den, min=1e-6)
            return self.projection(seg.reshape(seg.shape[0], -1))

        # `stats`: pool over time so the head no longer scales with seq_len.
        if padding_mask is not None:
            denom = torch.clamp(padding_mask.sum(dim=1), min=1.0).unsqueeze(1)
            pooled_mean = x.sum(dim=2) / denom
            var = ((x - pooled_mean.unsqueeze(2)) * padding_mask.unsqueeze(1)) ** 2
            pooled_std = torch.sqrt(var.sum(dim=2) / denom + 1e-5)
            neg_inf = torch.full_like(x, float("-inf"))
            pooled_max = torch.where(padding_mask.unsqueeze(1) > 0, x, neg_inf).max(dim=2)[0]
            pooled_max = torch.where(torch.isfinite(pooled_max), pooled_max,
                                     torch.zeros_like(pooled_max))
        else:
            pooled_mean = x.mean(dim=2)
            pooled_std = torch.sqrt(x.var(dim=2, unbiased=False) + 1e-5)
            pooled_max = x.max(dim=2)[0]
        return self.projection(torch.cat([pooled_mean, pooled_std, pooled_max], dim=1))

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in _FORECAST_TASKS:
            if self.delegate_core:
                return self.core(x_enc)
            return self._forecast(x_enc)
        if self.task_name == "imputation":
            return self._imputation(x_enc, mask)
        if self.task_name == "anomaly_detection":
            return self._anomaly(x_enc)
        if self.task_name == "classification":
            return self._classification(x_enc, x_mark_enc)
        raise ValueError("SparseTSFPlus: unsupported task_name '{}'".format(self.task_name))
