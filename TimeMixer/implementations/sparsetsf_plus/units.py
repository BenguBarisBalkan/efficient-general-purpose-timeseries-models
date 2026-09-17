"""Grafted units and shared plumbing for SparseTSFPlus.

Every unit here is **off by default**, and every default in `UnitConfig` reproduces stock
SparseTSF behaviour. That is the invariant the whole study rests on: with all units off,
SparseTSFPlus must be numerically identical to SparseTSF, so none of the already-published
numbers are invalidated and the ablation gets a real baseline row rather than a
re-implementation that "should" match.

Units are borrowed from this repo's other models wherever one already exists:

  RevIN                 layers/StandardNorm.py                  (0 params when affine=False)
  series_decomp         layers/Autoformer_EncDec.py             (0 params)
  phase-axis mixing     implementations/timemixer_pp/model.py   (TimeImageDecomposition,
                                                                 reduced from attention
                                                                 to a shared linear)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Flag name -> default. Every default is the stock-SparseTSF behaviour.
_UNIT_DEFAULTS = {
    "stsf_revin": 0,            # 0 mean-only (SparseTSF) | 1 mean+std | 2 subtract_last+std
    "stsf_revin_affine": 0,     # 1 -> 2*enc_in params; breaks enc_in-free counts, keep 0
    "stsf_phase_mix": "off",    # off | linear | conv
    "stsf_phase_rank": 0,       # 0 = full period_len^2, >0 = low-rank factorisation
    "stsf_phase_kernel": 3,     # for stsf_phase_mix=conv
    "stsf_periods": "",         # extra comma-separated periods, e.g. "8,24"
    "stsf_decomp": "off",       # off | free | linear
    "stsf_decomp_kernel": 25,   # forced odd (moving_avg pads (k-1)//2 per side)
    "stsf_pad_fold": 0,         # 1 -> allow period_len not dividing seq_len/pred_len
    "stsf_linear_bias": 0,      # bias on the cross-period linear
    "stsf_mask_conv": 0,        # imputation: partial-conv renormalisation + passthrough
    "stsf_diag_mask": 0,        # reconstruction: leave-one-out (zero the diagonal)
    "stsf_impute_passes": 1,    # 2 -> weight-shared refinement pass
    "stsf_cls_head": "flat",    # flat (current) | flat_fixed | stats | segpool
    "stsf_cls_segments": 8,     # segpool: number of time segments to pool into
}


class UnitConfig(object):
    """Which units are enabled, read off the argparse namespace.

    Uses getattr with defaults so the model still constructs when run.py has not been
    given the --stsf_* flags at all (e.g. an older runner, or count_params.py).
    """

    __slots__ = tuple(k[len("stsf_"):] for k in _UNIT_DEFAULTS)

    def __init__(self, **kwargs):
        for key, default in _UNIT_DEFAULTS.items():
            setattr(self, key[len("stsf_"):], kwargs.get(key, default))

    @classmethod
    def from_configs(cls, configs):
        return cls(**{k: getattr(configs, k, v) for k, v in _UNIT_DEFAULTS.items()})

    @property
    def is_vanilla(self):
        """True when every unit is at its stock-SparseTSF default."""
        return all(
            getattr(self, k[len("stsf_"):]) == v for k, v in _UNIT_DEFAULTS.items()
        )

    def extra_periods(self):
        """Parsed --stsf_periods, ignoring blanks."""
        if not self.periods:
            return []
        return [int(tok) for tok in str(self.periods).split(",") if tok.strip()]

    def describe(self):
        """Compact non-default summary, for logs and report rows."""
        if self.is_vanilla:
            return "vanilla"
        parts = []
        for key, default in _UNIT_DEFAULTS.items():
            name = key[len("stsf_"):]
            value = getattr(self, name)
            if value != default:
                parts.append("{}={}".format(name, value))
        return ",".join(parts)


def odd(kernel):
    """moving_avg pads (k-1)//2 on each side, so an even kernel silently drops a sample."""
    kernel = int(kernel)
    return kernel if kernel % 2 == 1 else kernel + 1


def fold_periods(x, period, pad=False):
    """(N, L) -> (N, period, n_seg), matching SparseTSF's phase/period axis order.

    SparseTSF folds with `reshape(-1, seg_num, period).permute(0, 2, 1)`, so axis 1 is the
    within-period phase and axis 2 indexes which period. When `period` divides `L` this
    function is exactly that reshape, which is what keeps the vanilla path bit-identical.

    When it does not divide and `pad=True`, we **front**-pad by `q = (-L) % period`. Front
    padding is the only choice that preserves input/output phase alignment for forecasting:
    original step `t` lands at padded position `t + q`, so the last observed step lands in
    phase `period - 1` and forecast step `h` correctly lands in phase `h % period`. End
    padding would shift the output's phase-0 row to `L % period` and teach the model the
    wrong phase correspondence. It also puts the zeros in the *oldest* period rather than
    the most recent one.
    """
    n, length = x.shape
    q = (-length) % period
    if q:
        if not pad:
            raise ValueError(
                "period_len ({}) must divide length ({}); pass pad=True "
                "(--stsf_pad_fold 1) to allow phase-aligned padding".format(period, length)
            )
        x = F.pad(x, (q, 0))
    n_seg = (length + q) // period
    return x.reshape(n, n_seg, period).permute(0, 2, 1), q


def unfold_periods(img, out_len):
    """(N, period, n_seg) -> (N, out_len), inverse of fold_periods for the output side.

    Forecast step `h` sits at unfolded position `h`, whose phase is `h % period` — already
    the correspondence fold_periods established — so we simply crop the tail.
    """
    n, period, n_seg = img.shape
    flat = img.permute(0, 2, 1).reshape(n, n_seg * period)
    return flat[:, :out_len]


class PhaseMix(nn.Module):
    """Mix along the *phase* axis of the folded image. Residual, zero-initialised.

    SparseTSF's forecast map is, per channel, exactly `I_p (x) W_n`: one `n_x -> n_y` matrix
    applied independently at each of `p` phases, so within-period shape is never modelled.
    Adding a phase-axis mixer makes it `W_p (x) W_n` — a Kronecker-factorised full `S -> P`
    linear map costing `p^2 + n_x*n_y` parameters instead of `S*P` (592 vs 9,216 for ETTh1
    96->96). Unlike SparseTSF's `pdm`-free design this stays `seq_len`-free: the same `p^2`
    at sl=96 and sl=720.

    Zero-init + residual means epoch 0 behaves exactly like SparseTSF, which kills the
    "maybe it just initialised badly" objection and keeps the ablation honest.
    """

    def __init__(self, period_len, mode="linear", rank=0, kernel=3):
        super(PhaseMix, self).__init__()
        self.mode = mode
        self.rank = rank
        if mode == "linear":
            if rank and rank > 0:
                self.down = nn.Linear(period_len, rank, bias=False)
                self.up = nn.Linear(rank, period_len, bias=False)
                nn.init.zeros_(self.up.weight)
            else:
                self.mix = nn.Linear(period_len, period_len, bias=False)
                nn.init.zeros_(self.mix.weight)
        elif mode == "conv":
            k = odd(kernel)
            self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
            nn.init.zeros_(self.conv.weight)
        else:
            raise ValueError("PhaseMix: unknown mode '{}'".format(mode))

    def forward(self, img):
        # img: (N, period, n_seg) -- mix over dim=1 (phase)
        if self.mode == "linear":
            h = img.transpose(1, 2)                      # (N, n_seg, period)
            h = self.up(self.down(h)) if self.rank else self.mix(h)
            return img + h.transpose(1, 2)
        n, period, n_seg = img.shape
        h = img.permute(0, 2, 1).reshape(-1, 1, period)  # (N*n_seg, 1, period)
        h = self.conv(h).reshape(n, n_seg, period).permute(0, 2, 1)
        return img + h


def zero_diagonal_(weight):
    """In-place: drop the self-term of a square reconstruction matrix (leave-one-out).

    `recon_linear = nn.Linear(seg_num, seg_num, bias=False)` with the conv applied as a
    residual means the identity map is exactly representable, so MSE reconstruction training
    on mostly-normal data converges toward it and anomalies get reconstructed faithfully —
    the signature of SMD's recall-limited failure (recall 73.90 vs precision 87.25). Zeroing
    the diagonal forces a genuine leave-one-out prediction for 0 parameters.
    """
    with torch.no_grad():
        weight.fill_diagonal_(0.0)


def masked_stats(x, mask, eps=1e-5, use_std=False):
    """Per-channel mean (and optionally std) over OBSERVED positions only.

    layers/StandardNorm.py's `Normalize` cannot be used for imputation: it computes
    statistics over every position, and masked positions are literal zeros. It is also
    stateful (stashes mean/stdev for `denorm`), so an instance cannot be shared across
    calls. This mirrors the mask-aware statistics in timemixer_pp/model.py:534-541.
    """
    if mask is None:
        mean = x.mean(dim=1, keepdim=True)
        if not use_std:
            return mean, None
        return mean, torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + eps)
    denom = torch.clamp(mask.sum(dim=1, keepdim=True), min=1.0)
    mean = (x * mask).sum(dim=1, keepdim=True) / denom
    if not use_std:
        return mean, None
    var = (((x - mean) * mask) ** 2).sum(dim=1, keepdim=True) / denom
    return mean, torch.sqrt(var + eps)


def sanitize(x):
    """Replace non-finite values with 0, routing zero gradient to the sanitized branch.

    Same guard as timemixer_pp/model.py:40, which is the coordinated system that made
    TimeMixer++ trainable. These units are linear/conv rather than attention, so the risk is
    much lower, but the guard is free.
    """
    return torch.where(torch.isfinite(x), x, torch.zeros_like(x))


def count_trainable(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
