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
| ETTh1 | base | - | 0.11947 | 0.113335 | 0.215180 | -0.00% | +17.15% | 40 | OK |
| ETTh1 | passes2 **<-** | 41 | 0.09421 | 0.089294 | 0.194212 | -21.21% | -7.70% | 39 | OK |
| ETTh1 | u5 | 41 | 0.09741 | 0.092795 | 0.198596 | -18.12% | -4.08% | 38 | OK |
| ETTh1 | all | 617 | 0.10075 | 0.097569 | 0.207954 | -13.91% | +0.85% | 38 | OK |
| _ETTh1 local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.113335_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTh1 local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.096742_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| ETTm1 | base | - | 0.05385 | 0.048898 | 0.139033 | +0.00% | +25.15% | 198 | OK |
| ETTm1 | passes2 | 581 | 0.04085 | 0.035768 | 0.114755 | -26.85% | -8.45% | 158 | OK |
| ETTm1 | u5 | 581 | 0.03970 | 0.034452 | 0.113267 | -29.54% | -11.82% | 151 | OK |
| ETTm1 | all **<-** | 597 | 0.03944 | 0.034291 | 0.112845 | -29.87% | -12.23% | 164 | OK |
| _ETTm1 local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.048898_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTm1 local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.039071_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| ETTh2 | base **<-** | 41 | 0.05113 | 0.058744 | 0.151748 | -0.00% | +0.36% | 32 | OK |
| ETTh2 | passes2 | 41 | 0.07846 | 0.099528 | 0.212542 | +69.43% | +70.05% | 27 | OK |
| ETTh2 | u5 | 41 | 0.07880 | 0.099833 | 0.213050 | +69.95% | +70.57% | 29 | OK |
| ETTh2 | all | 617 | 0.07333 | 0.092549 | 0.203623 | +57.55% | +58.12% | 39 | OK |
| _ETTh2 local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.058744_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTh2 local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.058530_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| ETTm2 | base | - | 0.02469 | 0.026657 | 0.094994 | -0.00% | +10.44% | 114 | OK |
| ETTm2 | passes2 | 581 | 0.02213 | 0.023237 | 0.082994 | -12.83% | -3.73% | 153 | OK |
| ETTm2 | u5 **<-** | 581 | 0.02201 | 0.023098 | 0.083057 | -13.35% | -4.30% | 151 | OK |
| ETTm2 | all | 597 | 0.02203 | 0.023099 | 0.083161 | -13.35% | -4.30% | 185 | OK |
| _ETTm2 local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.026657_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTm2 local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.024137_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| Weather | base | - | 0.06490 | 0.029147 | 0.048418 | +0.00% | +7.44% | 142 | OK |
| Weather | passes2 | 581 | 0.05859 | 0.027586 | 0.042255 | -5.36% | +1.68% | 155 | OK |
| Weather | u5 **<-** | 581 | 0.05552 | 0.026989 | 0.040546 | -7.40% | -0.52% | 205 | OK |
| Weather | all | 597 | 0.05692 | 0.026871 | 0.039921 | -7.81% | -0.96% | 212 | OK |
| _Weather local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.029147_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _Weather local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.027130_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| Electricity | base | - | 0.09860 | 0.099290 | 0.223833 | +0.00% | +61.65% | 185 | OK |
| Electricity | passes2 | 41 | 0.08434 | 0.082416 | 0.195618 | -16.99% | +34.18% | 278 | OK |
| Electricity | u5 | 41 | 0.08204 | 0.080205 | 0.193873 | -19.22% | +30.58% | 285 | OK |
| Electricity | all **<-** | 617 | 0.08001 | 0.078058 | 0.192477 | -21.38% | +27.08% | 292 | OK |
| _Electricity local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.099290_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _Electricity local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.061424_ | _—_ | _—_ | _—_ | _—_ | _reference_ |

### mask_rate = 0.25

| Dataset | Variant | Params | Best vali | MSE | MAE | vs SparseTSF | vs TimeMixer | Runtime (s) | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| ETTh1 | base | - | 0.14990 | 0.146550 | 0.245674 | +0.00% | +31.19% | 32 | OK |
| ETTh1 | passes2 **<-** | 41 | 0.10842 | 0.105566 | 0.210423 | -27.97% | -5.50% | 32 | OK |
| ETTh1 | u5 | 41 | 0.11137 | 0.109476 | 0.215704 | -25.30% | -2.00% | 44 | OK |
| ETTh1 | all | 617 | 0.11199 | 0.117239 | 0.228160 | -20.00% | +4.95% | 44 | OK |
| _ETTh1 local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.146550_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTh1 local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.111707_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| ETTm1 | base | - | 0.06719 | 0.062403 | 0.158274 | +0.00% | +47.81% | 127 | OK |
| ETTm1 | passes2 | 581 | 0.04717 | 0.041735 | 0.125962 | -33.12% | -1.14% | 160 | OK |
| ETTm1 | u5 | 581 | 0.04295 | 0.037048 | 0.117509 | -40.63% | -12.25% | 188 | OK |
| ETTm1 | all **<-** | 597 | 0.04254 | 0.036845 | 0.117155 | -40.96% | -12.73% | 166 | OK |
| _ETTm1 local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.062403_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTm1 local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.042218_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| ETTh2 | base **<-** | - | 0.05699 | 0.066180 | 0.163095 | -0.00% | +4.08% | 26 | OK |
| ETTh2 | passes2 | 41 | 0.07887 | 0.099614 | 0.212439 | +50.52% | +56.66% | 27 | OK |
| ETTh2 | u5 | 41 | 0.07954 | 0.099680 | 0.212627 | +50.62% | +56.76% | 26 | OK |
| ETTh2 | all | 617 | 0.07351 | 0.093092 | 0.204663 | +40.67% | +46.40% | 30 | OK |
| _ETTh2 local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.066180_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTh2 local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.063586_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| ETTm2 | base | - | 0.02812 | 0.030221 | 0.104586 | +0.00% | +16.27% | 128 | OK |
| ETTm2 | passes2 | 581 | 0.02443 | 0.025629 | 0.089871 | -15.20% | -1.40% | 159 | OK |
| ETTm2 | u5 **<-** | 581 | 0.02402 | 0.025095 | 0.087698 | -16.96% | -3.45% | 159 | OK |
| ETTm2 | all | 597 | 0.02402 | 0.025097 | 0.087705 | -16.95% | -3.44% | 189 | OK |
| _ETTm2 local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.030221_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTm2 local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.025992_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| Weather | base | - | 0.07714 | 0.032362 | 0.055355 | -0.00% | +7.62% | 149 | OK |
| Weather | passes2 | 581 | 0.06554 | 0.029277 | 0.045821 | -9.53% | -2.64% | 178 | OK |
| Weather | u5 | 581 | 0.06255 | 0.028401 | 0.042738 | -12.24% | -5.55% | 193 | OK |
| Weather | all **<-** | 597 | 0.06109 | 0.028484 | 0.042895 | -11.98% | -5.28% | 227 | OK |
| _Weather local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.032362_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _Weather local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.030070_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| Electricity | base | - | 0.11337 | 0.116498 | 0.245634 | -0.00% | +60.45% | 186 | OK |
| Electricity | passes2 | 41 | 0.09747 | 0.097463 | 0.218777 | -16.34% | +34.23% | 272 | OK |
| Electricity | u5 **<-** | 41 | 0.09360 | 0.093482 | 0.213026 | -19.76% | +28.75% | 260 | OK |
| Electricity | all | 617 | 0.12157 | 0.125131 | 0.241181 | +7.41% | +72.34% | 290 | OK |
| _Electricity local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.116498_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _Electricity local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.072609_ | _—_ | _—_ | _—_ | _—_ | _reference_ |

### mask_rate = 0.375

| Dataset | Variant | Params | Best vali | MSE | MAE | vs SparseTSF | vs TimeMixer | Runtime (s) | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| ETTh1 | base | - | 0.17957 | 0.179265 | 0.271485 | -0.00% | +43.01% | 32 | OK |
| ETTh1 | passes2 **<-** | 41 | 0.12861 | 0.128675 | 0.231401 | -28.22% | +2.65% | 39 | OK |
| ETTh1 | u5 | 41 | 0.13078 | 0.133712 | 0.238970 | -25.41% | +6.67% | 40 | OK |
| ETTh1 | all | 617 | 0.12914 | 0.134505 | 0.242356 | -24.97% | +7.31% | 42 | OK |
| _ETTh1 local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.179265_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTh1 local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.125347_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| ETTm1 | base | - | 0.08149 | 0.078078 | 0.176335 | -0.00% | +60.35% | 129 | OK |
| ETTm1 | passes2 | 581 | 0.05706 | 0.052248 | 0.141639 | -33.08% | +7.30% | 141 | OK |
| ETTm1 | u5 | 581 | 0.04861 | 0.042976 | 0.125932 | -44.96% | -11.74% | 148 | OK |
| ETTm1 | all **<-** | 597 | 0.04811 | 0.042312 | 0.125413 | -45.81% | -13.10% | 166 | OK |
| _ETTm1 local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.078078_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTm1 local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.048693_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| ETTh2 | base **<-** | - | 0.06286 | 0.073772 | 0.173912 | -0.00% | +7.79% | 26 | OK |
| ETTh2 | passes2 | 41 | 0.08036 | 0.100829 | 0.213557 | +36.68% | +47.33% | 26 | OK |
| ETTh2 | u5 | 41 | 0.08107 | 0.100837 | 0.213556 | +36.69% | +47.34% | 27 | OK |
| ETTh2 | all | 617 | 0.07499 | 0.094798 | 0.206233 | +28.50% | +38.51% | 30 | OK |
| _ETTh2 local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.073772_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTh2 local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.068440_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| ETTm2 | base | - | 0.03179 | 0.034499 | 0.113122 | +0.00% | +20.49% | 134 | OK |
| ETTm2 | passes2 **<-** | 581 | 0.02740 | 0.029000 | 0.098329 | -15.94% | +1.28% | 156 | OK |
| ETTm2 | u5 | 581 | 0.02891 | 0.030293 | 0.098381 | -12.19% | +5.80% | 161 | OK |
| ETTm2 | all | 597 | 0.02828 | 0.029384 | 0.096326 | -14.83% | +2.62% | 190 | OK |
| _ETTm2 local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.034499_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTm2 local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.028633_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| Weather | base | - | 0.09066 | 0.035990 | 0.061407 | +0.00% | +10.19% | 152 | OK |
| Weather | passes2 **<-** | 581 | 0.07703 | 0.032149 | 0.051384 | -10.67% | -1.57% | 180 | OK |
| Weather | u5 | 581 | 0.08684 | 0.034414 | 0.051374 | -4.38% | +5.36% | 172 | OK |
| Weather | all | 597 | 0.08398 | 0.033786 | 0.050840 | -6.12% | +3.44% | 241 | OK |
| _Weather local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.035990_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _Weather local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.032663_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| Electricity | base | - | 0.13302 | 0.138658 | 0.269219 | -0.00% | +62.70% | 192 | OK |
| Electricity | passes2 | 41 | 0.11425 | 0.116466 | 0.242612 | -16.01% | +36.66% | 329 | OK |
| Electricity | u5 **<-** | 41 | 0.10976 | 0.111595 | 0.234775 | -19.52% | +30.94% | 297 | OK |
| Electricity | all | 617 | 0.13213 | 0.137145 | 0.254888 | -1.09% | +60.93% | 279 | OK |
| _Electricity local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.138658_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _Electricity local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.085223_ | _—_ | _—_ | _—_ | _—_ | _reference_ |

### mask_rate = 0.5

| Dataset | Variant | Params | Best vali | MSE | MAE | vs SparseTSF | vs TimeMixer | Runtime (s) | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| ETTh1 | base | 41 | 0.21422 | 0.214076 | 0.296108 | -0.00% | +46.19% | 40 | OK |
| ETTh1 | maskconv | 41 | 0.21300 | 0.213604 | 0.295786 | -0.22% | +45.87% | 41 | OK |
| ETTh1 | passes2 **<-** | 41 | 0.15777 | 0.159201 | 0.256549 | -25.63% | +8.72% | 43 | OK |
| ETTh1 | u5 | 41 | 0.15947 | 0.169165 | 0.269107 | -20.98% | +15.52% | 43 | OK |
| ETTh1 | phase | 617 | 0.21197 | 0.208082 | 0.295382 | -2.80% | +42.10% | 37 | OK |
| ETTh1 | all | 617 | 0.16330 | 0.171786 | 0.271741 | -19.75% | +17.31% | 47 | OK |
| _ETTh1 local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.214076_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTh1 local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.146432_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| ETTm1 | base | 581 | 0.09850 | 0.096544 | 0.195342 | -0.00% | +66.05% | 137 | OK |
| ETTm1 | maskconv | 581 | 0.06748 | 0.062374 | 0.152209 | -35.39% | +7.28% | 134 | OK |
| ETTm1 | passes2 | 581 | 0.07273 | 0.068874 | 0.163173 | -28.66% | +18.46% | 144 | OK |
| ETTm1 | u5 | 581 | 0.06113 | 0.055998 | 0.148059 | -42.00% | -3.69% | 180 | OK |
| ETTm1 | phase | 597 | 0.09831 | 0.096229 | 0.195065 | -0.33% | +65.50% | 143 | OK |
| ETTm1 | all **<-** | 597 | 0.05949 | 0.054591 | 0.146038 | -43.45% | -6.11% | 190 | OK |
| _ETTm1 local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.096544_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTm1 local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.058143_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| ETTh2 | base | - | 0.06920 | 0.082488 | 0.185551 | +0.00% | +10.09% | 43 | OK |
| ETTh2 | passes2 | 41 | 0.08205 | 0.102799 | 0.215418 | +24.62% | +37.20% | 25 | OK |
| ETTh2 | u5 **<-** | 41 | 0.06762 | 0.081580 | 0.183226 | -1.10% | +8.88% | 27 | OK |
| ETTh2 | all | 617 | 0.06799 | 0.081958 | 0.184157 | -0.64% | +9.39% | 30 | OK |
| _ETTh2 local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.082488_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTh2 local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.074926_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| ETTm2 | base | - | 0.03597 | 0.039384 | 0.122035 | -0.00% | +21.60% | 131 | OK |
| ETTm2 | passes2 **<-** | 581 | 0.03139 | 0.033597 | 0.108182 | -14.69% | +3.74% | 147 | OK |
| ETTm2 | u5 | 581 | 0.03168 | 0.033516 | 0.107494 | -14.90% | +3.49% | 130 | OK |
| ETTm2 | all | 597 | 0.03157 | 0.033296 | 0.106585 | -15.46% | +2.81% | 195 | OK |
| _ETTm2 local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.039384_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTm2 local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.032387_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| Weather | base | - | 0.10859 | 0.040054 | 0.067667 | -0.00% | +10.27% | 138 | OK |
| Weather | passes2 **<-** | 581 | 0.09848 | 0.036328 | 0.058815 | -9.30% | +0.01% | 159 | OK |
| Weather | u5 | 581 | 0.10423 | 0.038116 | 0.059038 | -4.84% | +4.94% | 187 | OK |
| Weather | all | 597 | 0.10647 | 0.039526 | 0.059573 | -1.32% | +8.82% | 218 | OK |
| _Weather local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.040054_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _Weather local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.036323_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| Electricity | base | - | 0.15891 | 0.167128 | 0.296695 | +0.00% | +63.85% | 199 | OK |
| Electricity | passes2 | 41 | 0.13750 | 0.142230 | 0.270014 | -14.90% | +39.44% | 273 | OK |
| Electricity | u5 **<-** | 41 | 0.13194 | 0.136062 | 0.260992 | -18.59% | +33.40% | 277 | OK |
| Electricity | all | 617 | 0.14881 | 0.155295 | 0.274305 | -7.08% | +52.25% | 284 | OK |
| _Electricity local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.167128_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _Electricity local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.101999_ | _—_ | _—_ | _—_ | _—_ | _reference_ |

Variant legend:

- **base** — all units off == SparseTSF · `(none)`
- **maskconv** — partial-conv renormalisation · `--stsf_mask_conv 1`
- **passes2** — weight-shared refinement pass · `--stsf_impute_passes 2`
- **u5** — both U5 parts · `--stsf_mask_conv 1 --stsf_impute_passes 2`
- **phase** — intra-period mixing · `--stsf_phase_mix linear`
- **all** — U5 + phase mixing · `--stsf_mask_conv 1 --stsf_impute_passes 2 --stsf_phase_mix linear`
<!-- AUTO_PLUS_IMP_END -->
