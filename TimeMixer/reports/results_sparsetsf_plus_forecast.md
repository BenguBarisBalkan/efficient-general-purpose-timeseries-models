# SparseTSFPlus ablation — forecasting

Backbone units tested on the forecasting path, each on top of the dataset's selected period
(ETTh1 uses 12, chosen on validation over the published 24; ETTm1 and Weather keep 4).

**Select on the `Best vali` column.** Forecasting early-stops on validation loss, so that
column is a clean selection signal; **<-** marks the lowest validation loss per dataset and
horizon. The test columns are for reporting only.

Regenerate (idempotent; `--update-from-logs-only` rebuilds this page without training):

```bash
venv/Scripts/python.exe benchmarks/run_sparsetsf_plus_forecast.py --update-from-logs-only
```

<!-- AUTO_PLUS_FC_START -->
### Long-term forecasting (seq_len = 96)

| Dataset | pred | Variant | Period | Params | Best vali | MSE | MAE | vs SparseTSF (published) | vs TimeMixer (local) | Status |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| ETTh1 | 96 | base | 12 | 77 | 0.72911 | 0.383847 | 0.394121 | -3.22% | -0.50% | OK |
| ETTh1 | 96 | phase | 12 | 221 | 0.73095 | 0.394339 | 0.403488 | -0.58% | +2.21% | OK |
| ETTh1 | 96 | phase_r4 | 12 | 173 | 0.72382 | 0.387053 | 0.395617 | -2.42% | +0.33% | OK |
| ETTh1 | 96 | decomp | 12 | 77 | 0.74302 | 0.489070 | 0.471771 | +23.30% | +26.77% | OK |
| ETTh1 | 96 | decomp_lin **<-** | 12 | 141 | 0.70637 | 0.386312 | 0.396557 | -2.60% | +0.13% | OK |
| ETTh1 | 96 | phase_decomp | 12 | 221 | 0.73367 | 0.465719 | 0.459799 | +17.42% | +20.72% | OK |
| ETTh1 | 96 | phase_decomp_lin | 12 | 285 | 0.71339 | 0.395372 | 0.407094 | -0.32% | +2.48% | OK |
| ETTh1 | 96 | periods | 12 | 119 | 0.71574 | 0.386392 | 0.395458 | -2.58% | +0.16% | OK |
| ETTh1 | 192 | base | 12 | 141 | 1.01776 | 0.436622 | 0.420044 | +0.79% | -1.45% | OK |
| ETTh1 | 192 | decomp_lin **<-** | 12 | 269 | 1.00025 | 0.436864 | 0.420949 | +0.84% | -1.39% | OK |
| ETTh1 | 336 | base | 12 | 237 | 1.30177 | 0.449094 | 0.429703 | -0.12% | -12.47% | OK |
| ETTh1 | 336 | decomp_lin **<-** | 12 | 461 | 1.27999 | 0.447708 | 0.430352 | -0.43% | -12.74% | OK |
| ETTh1 | 720 | base | 12 | 493 | 1.57329 | 0.451810 | 0.449630 | -1.67% | -7.64% | OK |
| ETTh1 | 720 | decomp_lin **<-** | 12 | 973 | 1.56165 | 0.452515 | 0.450110 | -1.51% | -7.49% | OK |
| ETTm1 | 96 | base | 4 | 581 | 0.41268 | 0.362121 | 0.378584 | -0.00% | +10.94% | OK |
| ETTm1 | 96 | phase | 4 | 597 | 0.40626 | 0.354915 | 0.373618 | -1.99% | +8.74% | OK |
| ETTm1 | 96 | phase_r4 | 4 | 613 | 0.40844 | 0.357726 | 0.375854 | -1.21% | +9.60% | OK |
| ETTm1 | 96 | decomp | 4 | 581 | 0.42892 | 0.410708 | 0.409150 | +13.42% | +25.83% | OK |
| ETTm1 | 96 | decomp_lin **<-** | 4 | 1,157 | 0.40391 | 0.352311 | 0.371638 | -2.71% | +7.94% | OK |
| ETTm1 | 96 | phase_decomp | 4 | 597 | 0.41531 | 0.379556 | 0.392311 | +4.81% | +16.29% | OK |
| ETTm1 | 96 | phase_decomp_lin | 4 | 1,173 | 0.41007 | 0.358247 | 0.376678 | -1.07% | +9.76% | OK |
| ETTm1 | 96 | periods | 4 | 659 | 0.40719 | 0.355728 | 0.373093 | -1.77% | +8.99% | OK |
| ETTm1 | 192 | decomp_lin **<-** | 4 | 2,309 | 0.53170 | 0.387901 | 0.388124 | -1.64% | +6.19% | OK |
| ETTm1 | 336 | decomp_lin **<-** | 4 | 4,037 | 0.68622 | 0.421497 | 0.412764 | -1.25% | +7.16% | OK |
| ETTm1 | 720 | decomp_lin **<-** | 4 | 8,645 | 1.00410 | 0.481752 | 0.445145 | -0.81% | +6.72% | OK |
| Weather | 96 | base | 4 | 581 | 0.49142 | 0.198635 | 0.238264 | -0.00% | +20.73% | OK |
| Weather | 96 | phase | 4 | 597 | 0.48613 | 0.197938 | 0.237450 | -0.35% | +20.31% | OK |
| Weather | 96 | phase_r4 | 4 | 613 | 0.48620 | 0.197700 | 0.236705 | -0.47% | +20.16% | OK |
| Weather | 96 | decomp | 4 | 581 | 0.72569 | 0.236979 | 0.254807 | +19.30% | +44.04% | OK |
| Weather | 96 | decomp_lin **<-** | 4 | 1,157 | 0.48165 | 0.197730 | 0.236343 | -0.46% | +20.18% | OK |
| Weather | 96 | phase_decomp | 4 | 597 | 0.58712 | 0.214429 | 0.242849 | +7.95% | +30.33% | OK |
| Weather | 96 | phase_decomp_lin | 4 | 1,173 | 0.48198 | 0.196765 | 0.236113 | -0.94% | +19.59% | OK |
| Weather | 96 | periods | 4 | 659 | 0.49224 | 0.199694 | 0.237954 | +0.53% | +21.38% | OK |
| Weather | 192 | decomp_lin **<-** | 4 | 2,309 | 0.55625 | 0.242287 | 0.272369 | -0.23% | +15.74% | OK |
| Weather | 336 | decomp_lin **<-** | 4 | 4,037 | 0.64487 | 0.289399 | 0.308202 | -1.00% | +9.06% | OK |
| Weather | 720 | decomp_lin **<-** | 4 | 8,645 | 0.76662 | 0.361143 | 0.355106 | -0.45% | +5.50% | OK |

### PEMS multivariate short-term (seq_len = 96, pred_len = 12)

| Dataset | Variant | Period | Params | Best vali | MAE | MAPE (%) | RMSE | vs SparseTSF (published MAE) | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| PEMS03 | phase_p4 | 4 | 93 | 21.883 | 21.34 | 18.57 | 34.98 | -24.54% | OK |
| PEMS03 | dense_p1 **<-** | 1 | 1,153 | 21.759 | 21.17 | 18.39 | 34.73 | -25.16% | OK |
| PEMS04 | phase_p4 | 4 | 93 | 28.824 | 27.43 | 16.43 | 43.76 | -24.05% | OK |
| PEMS04 | dense_p1 **<-** | 1 | 1,153 | 28.750 | 27.33 | 16.44 | 43.64 | -24.34% | OK |
| PEMS07 | phase_p4 | 4 | 93 | 31.129 | 31.02 | 13.03 | 48.66 | -27.68% | OK |
| PEMS07 | dense_p1 **<-** | 1 | 1,153 | 30.945 | 30.90 | 12.88 | 48.63 | -27.96% | OK |
| PEMS08 | phase_p4 | 4 | 93 | 23.036 | 22.22 | 12.85 | 35.26 | -24.03% | OK |
| PEMS08 | dense_p1 **<-** | 1 | 1,153 | 22.917 | 22.11 | 12.78 | 35.15 | -24.42% | OK |

Variant legend:

- **base** — selected period, no units · `(none)`
- **phase** — full phase map · `--stsf_phase_mix linear`
- **phase_r4** — rank-4 phase map · `--stsf_phase_mix linear --stsf_phase_rank 4`
- **decomp** — trend held at last value · `--stsf_decomp free`
- **decomp_lin** — trend with its own map · `--stsf_decomp linear`
- **phase_decomp** — phase mixing + free decomposition · `--stsf_phase_mix linear --stsf_decomp free`
- **phase_decomp_lin** — phase mixing + learned decomposition · `--stsf_phase_mix linear --stsf_decomp linear`
- **periods** — one extra period, gated · `--stsf_periods <extra period>`
- **phase_p4** — phase mixing at period 4 · `--stsf_phase_mix linear --period_len 4`
- **dense_p1** — period 1: the dense limit · `--period_len 1`
<!-- AUTO_PLUS_FC_END -->
