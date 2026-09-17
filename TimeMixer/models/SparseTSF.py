"""
Task-aware SparseTSF for this repo's experiment loops.

Upstream SparseTSF (ICML 2024) is a forecasting-only model: `implementations/sparsetsf/model.py` maps
`(B, seq_len, C) -> (B, pred_len, C)` with a single-argument `forward(self, x)`. This module
extends it to the five Time-Series-Library tasks — long/short-term forecasting, imputation,
anomaly detection, classification — the same set TimeMixer / TimeMixer++ support (see the
`task_name` branch in `models/TimeMixer.py`). The experiment loops (`exp/exp_*.py`) and the
model registry (`exp/exp_basic.py`) are generic and unchanged; only this model gains the extra
task heads.

Design: the **forecasting path delegates to the untouched vendored core** (`SparseTSFCore`), so
existing forecast benchmark results are reproduced bit-for-bit and the vendored file keeps its
Apache-2.0 provenance. The non-forecast heads are built here from SparseTSF's own primitives —
per-channel mean removal, the depthwise 1-D conv "cross-period" aggregator, and period-segment
reshaping + one shared linear — to stay faithful to its sparse, tiny-parameter philosophy.

Per-task interface (matches how each `exp/exp_*.py` calls the model and slices the output):

    long/short forecast : model(x_enc, x_mark, x_dec, x_mark_dec)   -> (B, pred_len, C)
    imputation          : model(inp, x_mark, None, None, mask)      -> (B, seq_len, C)
    anomaly_detection   : model(x, None, None, None)                -> (B, seq_len, C)
    classification      : model(x, padding_mask, None, None)        -> (B, num_class)

Config fields read (all already provided by run.py / the exp classes): task_name, seq_len,
pred_len, enc_in, period_len, model_type, d_model, plus num_class (set by Exp_Classification)
and dropout (classification head only).

`period_len` divisibility is validated per task: forecasting needs it to divide BOTH seq_len and
pred_len; imputation/anomaly need it to divide seq_len; classification imposes no requirement
(UEA `max_seq_len` is arbitrary, so its head avoids period reshaping).
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from implementations.sparsetsf.model import Model as SparseTSFCore

_FORECAST_TASKS = ("long_term_forecast", "short_term_forecast")
_RECON_TASKS = ("imputation", "anomaly_detection")


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len

        self._validate_divisibility()

        if self.task_name in _FORECAST_TASKS:
            # Delegate forecasting to the untouched vendored core (exact reproducibility).
            self.core = SparseTSFCore(configs)
            return

        # Shared SparseTSF front-end for the non-forecast tasks: a depthwise 1-D conv that
        # aggregates each channel over time (identical construction to the core's conv1d).
        self.conv1d = nn.Conv1d(
            in_channels=1, out_channels=1,
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1, padding=self.period_len // 2,
            padding_mode="zeros", bias=False,
        )

        if self.task_name in _RECON_TASKS:
            # Cross-period reconstruction: for each within-period phase, map the seg_num
            # per-period values back onto themselves with one shared linear (the seq_len ->
            # seq_len analogue of the core's seg_num_x -> seg_num_y forecast linear).
            self.seg_num = self.seq_len // self.period_len
            self.recon_linear = nn.Linear(self.seg_num, self.seg_num, bias=False)
        elif self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(getattr(configs, "dropout", 0.1))
            self.projection = nn.Linear(self.enc_in * self.seq_len, configs.num_class)
        else:
            raise ValueError(
                f"SparseTSF: unsupported task_name '{self.task_name}'. Supported: "
                f"{_FORECAST_TASKS + _RECON_TASKS + ('classification',)}"
            )

    def _validate_divisibility(self):
        p = self.period_len
        if self.task_name in _FORECAST_TASKS:
            if self.seq_len % p != 0:
                raise ValueError(f"SparseTSF: seq_len ({self.seq_len}) must be divisible by period_len ({p})")
            if self.pred_len % p != 0:
                raise ValueError(f"SparseTSF: pred_len ({self.pred_len}) must be divisible by period_len ({p})")
        elif self.task_name in _RECON_TASKS:
            if self.seq_len % p != 0:
                raise ValueError(
                    f"SparseTSF ({self.task_name}): seq_len ({self.seq_len}) must be divisible "
                    f"by period_len ({p})"
                )
        # classification: no divisibility requirement (head does not reshape into periods).

    # ── shared front-end ─────────────────────────────────────────────────────────
    def _aggregate(self, x):
        # x: (B, C, S) -> per-channel conv over time + residual -> (B, C, S)
        B, C, S = x.shape
        return self.conv1d(x.reshape(-1, 1, S)).reshape(B, C, S) + x

    def _reconstruct(self, x, seq_mean):
        # x: mean-subtracted (B, S, C); reconstruct the full-length sequence, add mean back.
        x = self._aggregate(x.permute(0, 2, 1))               # (B, C, S)
        B, C, S = x.shape
        # (B, C, S) -> (B*C, seg_num, period_len) -> (B*C, period_len, seg_num)  [core's ordering]
        x = x.reshape(-1, self.seg_num, self.period_len).permute(0, 2, 1)
        y = self.recon_linear(x)                              # (B*C, period_len, seg_num)
        y = y.permute(0, 2, 1).reshape(B, C, S).permute(0, 2, 1)  # (B, S, C)
        return y + seq_mean

    # ── task heads ───────────────────────────────────────────────────────────────
    def _imputation(self, x_enc, mask):
        # Normalise using observed values only (masked positions are 0 in x_enc).
        if mask is not None:
            denom = torch.clamp(mask.sum(dim=1, keepdim=True), min=1.0)
            seq_mean = (x_enc * mask).sum(dim=1, keepdim=True) / denom
            x = (x_enc - seq_mean) * mask
        else:
            seq_mean = x_enc.mean(dim=1, keepdim=True)
            x = x_enc - seq_mean
        return self._reconstruct(x, seq_mean)

    def _anomaly(self, x_enc):
        seq_mean = x_enc.mean(dim=1, keepdim=True)
        return self._reconstruct(x_enc - seq_mean, seq_mean)

    def _classification(self, x_enc, padding_mask):
        x = (x_enc - x_enc.mean(dim=1, keepdim=True)).permute(0, 2, 1)  # (B, C, S)
        x = self._aggregate(x)
        if padding_mask is not None:
            x = x * padding_mask.unsqueeze(1)          # zero out padded time steps
        x = self.dropout(self.act(x))
        return self.projection(x.reshape(x.shape[0], -1))              # (B, num_class)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in _FORECAST_TASKS:
            return self.core(x_enc)                    # (B, pred_len, C)
        if self.task_name == "imputation":
            return self._imputation(x_enc, mask)       # (B, seq_len, C)
        if self.task_name == "anomaly_detection":
            return self._anomaly(x_enc)                # (B, seq_len, C)
        if self.task_name == "classification":
            return self._classification(x_enc, x_mark_enc)  # (B, num_class)
        raise ValueError(f"SparseTSF: unsupported task_name '{self.task_name}'")
