# SparseTSFPlus ablation — imputation

Which grafted units earn their parameters, measured on the one weak task with a
same-machine TimeMixer baseline.

**`base` is the control**: all units off, which is numerically identical to SparseTSF
(proved bitwise for all five tasks by `benchmarks/tools/assert_sparsetsf_plus_equiv.py`).
Its row reproducing the published SparseTSF cell is the end-to-end confirmation.

**Select units on the `Best vali` column, not on MSE.** `exp/exp_imputation.py:164`
early-stops on *test* loss rather than validation loss. That is upstream TSLib behaviour
and every model in this repo was trained under it, so cross-model comparisons remain
internally consistent — but it means picking a unit combination by test MSE would be
double-dipping.

Regenerate (idempotent; `--update-from-logs-only` rebuilds the table without training):

```bash
venv/Scripts/python.exe benchmarks/run_sparsetsf_plus_imputation.py
```

<!-- AUTO_PLUS_IMP_START -->
### mask_rate = 0.125

| Dataset | Variant | Params | Best vali | MSE | MAE | vs SparseTSF | vs TimeMixer | Runtime (s) | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| ETTh1 | base | - | - | - | - | - | - | - | PENDING |
| ETTh1 | maskconv | - | - | - | - | - | - | - | PENDING |
| ETTh1 | passes2 **<-** | 41 | 0.09421 | 0.089294 | 0.194212 | -21.21% | -7.70% | 39 | OK |
| ETTh1 | u5 | 41 | 0.09741 | 0.092795 | 0.198596 | -18.12% | -4.08% | 38 | OK |
| ETTh1 | phase | - | - | - | - | - | - | - | PENDING |
| ETTh1 | all | 617 | 0.10075 | 0.097569 | 0.207954 | -13.91% | +0.85% | 38 | OK |
| _ETTh1 published_ | _SparseTSF_ | _581 / 41_ | _—_ | _0.113335_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTh1 published_ | _TimeMixer_ | _~75k_ | _—_ | _0.096742_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| ETTm1 | base | - | - | - | - | - | - | - | PENDING |
| ETTm1 | maskconv | - | - | - | - | - | - | - | PENDING |
| ETTm1 | passes2 | 581 | 0.04085 | 0.035768 | 0.114755 | -26.85% | -8.45% | 158 | OK |
| ETTm1 | u5 | 581 | 0.03970 | 0.034452 | 0.113267 | -29.54% | -11.82% | 151 | OK |
| ETTm1 | phase | - | - | - | - | - | - | - | PENDING |
| ETTm1 | all **<-** | 597 | 0.03944 | 0.034291 | 0.112845 | -29.87% | -12.23% | 164 | OK |
| _ETTm1 published_ | _SparseTSF_ | _581 / 41_ | _—_ | _0.048898_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTm1 published_ | _TimeMixer_ | _~75k_ | _—_ | _0.039071_ | _—_ | _—_ | _—_ | _—_ | _reference_ |

### mask_rate = 0.25

| Dataset | Variant | Params | Best vali | MSE | MAE | vs SparseTSF | vs TimeMixer | Runtime (s) | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| ETTh1 | base | - | - | - | - | - | - | - | PENDING |
| ETTh1 | maskconv | - | - | - | - | - | - | - | PENDING |
| ETTh1 | passes2 **<-** | 41 | 0.10842 | 0.105566 | 0.210423 | -27.97% | -5.50% | 32 | OK |
| ETTh1 | u5 | 41 | 0.11137 | 0.109476 | 0.215704 | -25.30% | -2.00% | 44 | OK |
| ETTh1 | phase | - | - | - | - | - | - | - | PENDING |
| ETTh1 | all | 617 | 0.11199 | 0.117239 | 0.228160 | -20.00% | +4.95% | 44 | OK |
| _ETTh1 published_ | _SparseTSF_ | _581 / 41_ | _—_ | _0.146550_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTh1 published_ | _TimeMixer_ | _~75k_ | _—_ | _0.111707_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| ETTm1 | base | - | - | - | - | - | - | - | PENDING |
| ETTm1 | maskconv | - | - | - | - | - | - | - | PENDING |
| ETTm1 | passes2 | 581 | 0.04717 | 0.041735 | 0.125962 | -33.12% | -1.14% | 160 | OK |
| ETTm1 | u5 | 581 | 0.04295 | 0.037048 | 0.117509 | -40.63% | -12.25% | 188 | OK |
| ETTm1 | phase | - | - | - | - | - | - | - | PENDING |
| ETTm1 | all **<-** | 597 | 0.04254 | 0.036845 | 0.117155 | -40.96% | -12.73% | 166 | OK |
| _ETTm1 published_ | _SparseTSF_ | _581 / 41_ | _—_ | _0.062403_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTm1 published_ | _TimeMixer_ | _~75k_ | _—_ | _0.042218_ | _—_ | _—_ | _—_ | _—_ | _reference_ |

### mask_rate = 0.375

| Dataset | Variant | Params | Best vali | MSE | MAE | vs SparseTSF | vs TimeMixer | Runtime (s) | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| ETTh1 | base | - | - | - | - | - | - | - | PENDING |
| ETTh1 | maskconv | - | - | - | - | - | - | - | PENDING |
| ETTh1 | passes2 **<-** | 41 | 0.12861 | 0.128675 | 0.231401 | -28.22% | +2.65% | 39 | OK |
| ETTh1 | u5 | 41 | 0.13078 | 0.133712 | 0.238970 | -25.41% | +6.67% | 40 | OK |
| ETTh1 | phase | - | - | - | - | - | - | - | PENDING |
| ETTh1 | all | 617 | 0.12914 | 0.134505 | 0.242356 | -24.97% | +7.31% | 42 | OK |
| _ETTh1 published_ | _SparseTSF_ | _581 / 41_ | _—_ | _0.179265_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTh1 published_ | _TimeMixer_ | _~75k_ | _—_ | _0.125347_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| ETTm1 | base | - | - | - | - | - | - | - | PENDING |
| ETTm1 | maskconv | - | - | - | - | - | - | - | PENDING |
| ETTm1 | passes2 | 581 | 0.05706 | 0.052248 | 0.141639 | -33.08% | +7.30% | 141 | OK |
| ETTm1 | u5 | 581 | 0.04861 | 0.042976 | 0.125932 | -44.96% | -11.74% | 148 | OK |
| ETTm1 | phase | - | - | - | - | - | - | - | PENDING |
| ETTm1 | all **<-** | 597 | 0.04811 | 0.042312 | 0.125413 | -45.81% | -13.10% | 166 | OK |
| _ETTm1 published_ | _SparseTSF_ | _581 / 41_ | _—_ | _0.078078_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTm1 published_ | _TimeMixer_ | _~75k_ | _—_ | _0.048693_ | _—_ | _—_ | _—_ | _—_ | _reference_ |

### mask_rate = 0.5

| Dataset | Variant | Params | Best vali | MSE | MAE | vs SparseTSF | vs TimeMixer | Runtime (s) | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| ETTh1 | base | 41 | 0.21422 | 0.214076 | 0.296108 | -0.00% | +46.19% | 40 | OK |
| ETTh1 | maskconv | 41 | 0.21300 | 0.213604 | 0.295786 | -0.22% | +45.87% | 41 | OK |
| ETTh1 | passes2 **<-** | 41 | 0.15777 | 0.159201 | 0.256549 | -25.63% | +8.72% | 43 | OK |
| ETTh1 | u5 | 41 | 0.15947 | 0.169165 | 0.269107 | -20.98% | +15.52% | 43 | OK |
| ETTh1 | phase | 617 | 0.21197 | 0.208082 | 0.295382 | -2.80% | +42.10% | 37 | OK |
| ETTh1 | all | 617 | 0.16330 | 0.171786 | 0.271741 | -19.75% | +17.31% | 47 | OK |
| _ETTh1 published_ | _SparseTSF_ | _581 / 41_ | _—_ | _0.214076_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTh1 published_ | _TimeMixer_ | _~75k_ | _—_ | _0.146432_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| ETTm1 | base | 581 | 0.09850 | 0.096544 | 0.195342 | -0.00% | +66.05% | 137 | OK |
| ETTm1 | maskconv | 581 | 0.06748 | 0.062374 | 0.152209 | -35.39% | +7.28% | 134 | OK |
| ETTm1 | passes2 | 581 | 0.07273 | 0.068874 | 0.163173 | -28.66% | +18.46% | 144 | OK |
| ETTm1 | u5 | 581 | 0.06113 | 0.055998 | 0.148059 | -42.00% | -3.69% | 180 | OK |
| ETTm1 | phase | 597 | 0.09831 | 0.096229 | 0.195065 | -0.33% | +65.50% | 143 | OK |
| ETTm1 | all **<-** | 597 | 0.05949 | 0.054591 | 0.146038 | -43.45% | -6.11% | 190 | OK |
| _ETTm1 published_ | _SparseTSF_ | _581 / 41_ | _—_ | _0.096544_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTm1 published_ | _TimeMixer_ | _~75k_ | _—_ | _0.058143_ | _—_ | _—_ | _—_ | _—_ | _reference_ |

Variant legend:

- **base** — all units off == SparseTSF · `(none)`
- **maskconv** — partial-conv renormalisation · `--stsf_mask_conv 1`
- **passes2** — weight-shared refinement pass · `--stsf_impute_passes 2`
- **u5** — both U5 parts · `--stsf_mask_conv 1 --stsf_impute_passes 2`
- **phase** — intra-period mixing · `--stsf_phase_mix linear`
- **all** — U5 + phase mixing · `--stsf_mask_conv 1 --stsf_impute_passes 2 --stsf_phase_mix linear`
<!-- AUTO_PLUS_IMP_END -->
