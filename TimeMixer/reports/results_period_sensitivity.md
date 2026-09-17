# Period-length sensitivity of SparseTSF

Diagnostics run **before** any architectural change, to separate "SparseTSF is missing a
component" from "SparseTSF's `period_len` was mis-specified for this dataset". Uses only
existing `run.py` flags — no model code was modified.

Every non-period hyperparameter is copied verbatim from the runner that produced the
published cell (`run_sparsetsf_benchmarks.py` for forecasting, `run_pems_benchmarks.py`
for PEMS), so the **baseline row must reproduce the published number**. A mismatch there
means the probe is mis-wired, not that the period matters.

Regenerate:

```bash
venv/Scripts/python.exe benchmarks/run_period_sensitivity.py
```

<!-- AUTO_PERIOD_PROBE_START -->
### Long-term forecasting (seq_len=96, pred_len=96)

`*` marks the published baseline period; **<-** marks the period with the
lowest **validation** loss, which is the only leakage-free way to select it.
**Params** is the whole model.

| Dataset | period_len | Params | Best vali | MSE | MAE | vs baseline MSE | Runtime (s) | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Weather | 4* **<-** | 581 | 0.4914 | 0.198635 | 0.238264 | — | 77 | OK |
| Weather | 8 | 153 | 0.5006 | 0.200913 | 0.242017 | +1.15% | 62 | OK |
| Weather | 12 | 77 | 0.5079 | 0.204776 | 0.244794 | +3.09% | 54 | OK |
| Weather | 24 | 41 | 0.5257 | 0.211284 | 0.252217 | +6.37% | 102 | OK |
| Weather | 48 | 53 | 0.5497 | 0.213992 | 0.255119 | +7.73% | 59 | OK |
| _Weather published_ | _4_ | _—_ | _—_ | _0.198635_ | _0.238264_ | _REPRODUCED_ | _—_ | _reference_ |
| ETTm1 | 4* **<-** | 581 | 0.4127 | 0.362121 | 0.378584 | — | 32 | OK |
| ETTm1 | 8 | 153 | 0.4198 | 0.369386 | 0.382464 | +2.01% | 28 | OK |
| ETTm1 | 12 | 77 | 0.4303 | 0.378033 | 0.388852 | +4.39% | 24 | OK |
| ETTm1 | 24 | 41 | 0.4470 | 0.386601 | 0.392209 | +6.76% | 20 | OK |
| ETTm1 | 48 | 53 | 0.4563 | 0.387029 | 0.394135 | +6.88% | 32 | OK |
| _ETTm1 published_ | _4_ | _—_ | _—_ | _0.362121_ | _0.378584_ | _REPRODUCED_ | _—_ | _reference_ |
| ETTh1 | 24* | 41 | 0.7482 | 0.396635 | 0.403991 | — | 12 | OK |
| ETTh1 | 12 **<-** | 77 | 0.7291 | 0.383847 | 0.394121 | -3.22% | 8 | OK |
| ETTh1 | 48 | 53 | 0.7687 | 0.384508 | 0.392750 | -3.06% | 8 | OK |
| _ETTh1 published_ | _24_ | _—_ | _—_ | _0.396635_ | _0.403991_ | _REPRODUCED_ | _—_ | _reference_ |

### PEMS multivariate short-term (seq_len=96, pred_len=12)

At the published `period_len=12`, `seg_num_y = 12/12 = 1`, so the forecast
linear is `nn.Linear(8, 1, bias=False)` — **8 parameters**.

| Dataset | period_len | Params | Best vali | MAE | MAPE (%) | RMSE | Runtime (s) | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| PEMS08 | 12* | 21 | 30.754 | 29.25 | 17.14 | 44.61 | 44 | OK |
| PEMS08 | 6 | 39 | 26.676 | 25.55 | 14.83 | 40.20 | 53 | OK |
| PEMS08 | 4 | 77 | 25.203 | 24.33 | 14.17 | 38.34 | 63 | OK |
| PEMS08 | 3 | 131 | 24.602 | 23.68 | 13.76 | 37.33 | 61 | OK |
| PEMS08 | 2 | 291 | 23.727 | 22.92 | 13.28 | 36.31 | 58 | OK |
| PEMS08 | 1 **<-** | 1,153 | 22.917 | 22.11 | 12.78 | 35.15 | 63 | OK |

| Published reference | MAE | MAPE (%) | RMSE |
|---|---:|---:|---:|
| _TimeMixer (paper, 4-dataset avg)_ | _17.41_ | _10.59_ | _28.01_ |
| _TimeMixer++ (paper, 4-dataset avg)_ | _15.91_ | _10.08_ | _27.06_ |

Note: the published references are averages over PEMS03/04/07/08, while this
probe runs PEMS08 only — read the *trend across period_len*, not the absolute gap.
<!-- AUTO_PERIOD_PROBE_END -->
