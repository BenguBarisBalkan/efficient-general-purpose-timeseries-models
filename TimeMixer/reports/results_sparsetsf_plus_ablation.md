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
### Mean over the four mask rates: masked test MSE

| Dataset | base | passes2 | u5 | all | u5g | allg | allrg |
|---|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | 0.1633 | 0.1207 | 0.1263 | 0.1303 | 0.1230 | 0.1217 | 0.1222 |
| ETTm1 | 0.0715 | 0.0497 | 0.0426 | 0.0420 | 0.0420 | 0.0415 | 0.0415 |
| ETTh2 | 0.0703 | 0.1007 | 0.0955 | 0.0906 | 0.0641 | 0.0766 | 0.0770 |
| ETTm2 | 0.0327 | 0.0279 | 0.0280 | 0.0277 | 0.0269 | 0.0269 | 0.0268 |
| Weather | 0.0344 | 0.0313 | 0.0320 | 0.0322 | 0.0299 | 0.0299 | 0.0300 |
| Electricity | 0.1304 | 0.1096 | 0.1053 | 0.1239 | 0.1048 | 0.1159 | 0.1158 |
| **Mean (6 datasets)** | **0.0838** | **0.0733** | **0.0716** | **0.0744** | **0.0651** | **0.0688** | **0.0689** |

### Mean over the four mask rates: best validation loss (select on this)

| Dataset | base | passes2 | u5 | all | u5g | allg | allrg |
|---|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | 0.1658 | 0.1223 | 0.1248 | 0.1263 | 0.1248 | 0.1238 | 0.1239 |
| ETTm1 | 0.0753 | 0.0545 | 0.0481 | 0.0474 | 0.0476 | 0.0470 | 0.0470 |
| ETTh2 | 0.0600 | 0.0799 | 0.0768 | 0.0725 | 0.0567 | 0.0638 | 0.0638 |
| ETTm2 | 0.0301 | 0.0263 | 0.0267 | 0.0265 | 0.0257 | 0.0257 | 0.0257 |
| Weather | 0.0853 | 0.0749 | 0.0773 | 0.0771 | 0.0692 | 0.0686 | 0.0687 |
| Electricity | 0.1260 | 0.1084 | 0.1043 | 0.1206 | 0.1039 | 0.1133 | 0.1132 |
| **Mean (6 datasets)** | **0.0904** | **0.0777** | **0.0763** | **0.0784** | **0.0713** | **0.0737** | **0.0737** |

### Learned gate values (mask 0.125 / 0.25 / 0.375 / 0.5)

Read from each run's tested checkpoint. Every gate starts at 0, the model without the unit. A refinement gate of 1 is ungated refinement (pass 2 replaces pass 1), and 0.5 averages the two passes. The phase gate multiplies a mixer trained together with it, so its values are not on a common scale across runs: read only whether it stayed near 0.

| Dataset | u5g | allg | allrg |
|---|---|---|---|
| ETTh1 | refine 0.58 / 0.63 / 0.66 / 0.70 | phase -0.54 / -0.18 / -0.17 / -0.20; refine 0.49 / 0.62 / 0.68 / 0.72 | refine 0.48 / 0.58 / 0.69 / 0.76 |
| ETTm1 | refine 0.80 / 0.84 / 0.88 / 0.94 | phase 0.08 / 0.21 / 0.59 / 0.79; refine 0.82 / 0.88 / 1.08 / 1.03 | refine 0.84 / 0.93 / 1.01 / 1.05 |
| ETTh2 | refine 0.48 / 0.57 / 0.62 / 0.54 | phase 0.13 / 0.10 / 0.93 / 0.81; refine 0.47 / 0.58 / 1.17 / 0.70 | refine 0.41 / 0.53 / 1.08 / 0.82 |
| ETTm2 | refine 0.81 / 0.85 / 0.89 / 0.92 | phase 0.21 / 0.08 / 0.19 / 0.54; refine 0.38 / 0.86 / 0.94 / 1.15 | refine 0.86 / 0.97 / 1.16 / 1.21 |
| Weather | refine 0.67 / 0.91 / 1.03 / 1.07 | phase 0.11 / 0.06 / 0.05 / 0.07; refine 0.56 / 0.88 / 1.00 / 1.09 | refine 0.54 / 0.84 / 0.92 / 1.00 |
| Electricity | refine 0.96 / 1.02 / 1.03 / 1.05 | phase -0.51 / -0.60 / -0.75 / -0.77; refine 0.91 / 1.10 / 1.12 / 1.16 | refine 0.97 / 1.10 / 1.13 / 1.15 |

### mask_rate = 0.125

| Dataset | Variant | Params | Best vali | MSE | MAE | vs SparseTSF | vs TimeMixer | Runtime (s) | Learned gates | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| ETTh1 | base | - | 0.11947 | 0.113335 | 0.215180 | -0.00% | +17.15% | 40 | - | OK |
| ETTh1 | passes2 **<-** | 41 | 0.09421 | 0.089294 | 0.194212 | -21.21% | -7.70% | 39 | - | OK |
| ETTh1 | u5 | 41 | 0.09741 | 0.092795 | 0.198596 | -18.12% | -4.08% | 38 | - | OK |
| ETTh1 | all | 617 | 0.10075 | 0.097569 | 0.207954 | -13.91% | +0.85% | 38 | - | OK |
| ETTh1 | u5g | 42 | 0.09565 | 0.090447 | 0.195195 | -20.19% | -6.51% | 32 | refine 0.581 | OK |
| ETTh1 | allg | 619 | 0.09788 | 0.092553 | 0.198398 | -18.34% | -4.33% | 41 | phase -0.540, refine 0.491 | OK |
| ETTh1 | allrg | 618 | 0.09721 | 0.091630 | 0.197167 | -19.15% | -5.28% | 39 | refine 0.476 | OK |
| _ETTh1 local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.113335_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTh1 local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.096742_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| ETTm1 | base | - | 0.05385 | 0.048898 | 0.139033 | +0.00% | +25.15% | 198 | - | OK |
| ETTm1 | passes2 | 581 | 0.04085 | 0.035768 | 0.114755 | -26.85% | -8.45% | 158 | - | OK |
| ETTm1 | u5 | 581 | 0.03970 | 0.034452 | 0.113267 | -29.54% | -11.82% | 151 | - | OK |
| ETTm1 | all | 597 | 0.03944 | 0.034291 | 0.112845 | -29.87% | -12.23% | 164 | - | OK |
| ETTm1 | u5g | 582 | 0.03955 | 0.034251 | 0.113124 | -29.95% | -12.34% | 136 | refine 0.796 | OK |
| ETTm1 | allg | 599 | 0.03939 | 0.034178 | 0.112823 | -30.10% | -12.52% | 166 | phase 0.076, refine 0.817 | OK |
| ETTm1 | allrg **<-** | 598 | 0.03932 | 0.034092 | 0.112614 | -30.28% | -12.74% | 153 | refine 0.844 | OK |
| _ETTm1 local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.048898_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTm1 local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.039071_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| ETTh2 | base | 41 | 0.05113 | 0.058744 | 0.151748 | -0.00% | +0.36% | 32 | - | OK |
| ETTh2 | passes2 | 41 | 0.07846 | 0.099528 | 0.212542 | +69.43% | +70.05% | 27 | - | OK |
| ETTh2 | u5 | 41 | 0.07880 | 0.099833 | 0.213050 | +69.95% | +70.57% | 29 | - | OK |
| ETTh2 | all | 617 | 0.07333 | 0.092549 | 0.203623 | +57.55% | +58.12% | 39 | - | OK |
| ETTh2 | u5g | 42 | 0.05061 | 0.056278 | 0.145693 | -4.20% | -3.85% | 34 | refine 0.483 | OK |
| ETTh2 | allg | 619 | 0.05035 | 0.056080 | 0.145814 | -4.53% | -4.19% | 47 | phase 0.128, refine 0.469 | OK |
| ETTh2 | allrg **<-** | 618 | 0.04955 | 0.055451 | 0.145511 | -5.61% | -5.26% | 41 | refine 0.413 | OK |
| _ETTh2 local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.058744_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTh2 local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.058530_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| ETTm2 | base | - | 0.02469 | 0.026657 | 0.094994 | -0.00% | +10.44% | 114 | - | OK |
| ETTm2 | passes2 | 581 | 0.02213 | 0.023237 | 0.082994 | -12.83% | -3.73% | 153 | - | OK |
| ETTm2 | u5 | 581 | 0.02201 | 0.023098 | 0.083057 | -13.35% | -4.30% | 151 | - | OK |
| ETTm2 | all | 597 | 0.02203 | 0.023099 | 0.083161 | -13.35% | -4.30% | 185 | - | OK |
| ETTm2 | u5g **<-** | 582 | 0.02200 | 0.023083 | 0.083120 | -13.41% | -4.37% | 140 | refine 0.805 | OK |
| ETTm2 | allg | 599 | 0.02218 | 0.023265 | 0.083907 | -12.72% | -3.61% | 163 | phase 0.208, refine 0.378 | OK |
| ETTm2 | allrg | 598 | 0.02236 | 0.023232 | 0.083868 | -12.85% | -3.75% | 153 | refine 0.859 | OK |
| _ETTm2 local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.026657_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTm2 local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.024137_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| Weather | base | - | 0.06490 | 0.029147 | 0.048418 | +0.00% | +7.44% | 142 | - | OK |
| Weather | passes2 | 581 | 0.05859 | 0.027586 | 0.042255 | -5.36% | +1.68% | 155 | - | OK |
| Weather | u5 | 581 | 0.05552 | 0.026989 | 0.040546 | -7.40% | -0.52% | 205 | - | OK |
| Weather | all | 597 | 0.05692 | 0.026871 | 0.039921 | -7.81% | -0.96% | 212 | - | OK |
| Weather | u5g **<-** | 582 | 0.05451 | 0.026794 | 0.040511 | -8.07% | -1.24% | 131 | refine 0.669 | OK |
| Weather | allg | 599 | 0.05583 | 0.026517 | 0.039938 | -9.02% | -2.26% | 168 | phase 0.106, refine 0.565 | OK |
| Weather | allrg | 598 | 0.05592 | 0.026549 | 0.039987 | -8.91% | -2.14% | 170 | refine 0.540 | OK |
| _Weather local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.029147_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _Weather local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.027130_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| Electricity | base | - | 0.09860 | 0.099290 | 0.223833 | +0.00% | +61.65% | 185 | - | OK |
| Electricity | passes2 | 41 | 0.08434 | 0.082416 | 0.195618 | -16.99% | +34.18% | 278 | - | OK |
| Electricity | u5 | 41 | 0.08204 | 0.080205 | 0.193873 | -19.22% | +30.58% | 285 | - | OK |
| Electricity | all **<-** | 617 | 0.08001 | 0.078058 | 0.192477 | -21.38% | +27.08% | 292 | - | OK |
| Electricity | u5g | 42 | 0.08289 | 0.080868 | 0.193876 | -18.55% | +31.66% | 228 | refine 0.958 | OK |
| Electricity | allg | 619 | 0.08179 | 0.079859 | 0.194090 | -19.57% | +30.01% | 240 | phase -0.512, refine 0.909 | OK |
| Electricity | allrg | 618 | 0.08004 | 0.077921 | 0.191814 | -21.52% | +26.86% | 236 | refine 0.968 | OK |
| _Electricity local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.099290_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _Electricity local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.061424_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |

### mask_rate = 0.25

| Dataset | Variant | Params | Best vali | MSE | MAE | vs SparseTSF | vs TimeMixer | Runtime (s) | Learned gates | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| ETTh1 | base | - | 0.14990 | 0.146550 | 0.245674 | +0.00% | +31.19% | 32 | - | OK |
| ETTh1 | passes2 **<-** | 41 | 0.10842 | 0.105566 | 0.210423 | -27.97% | -5.50% | 32 | - | OK |
| ETTh1 | u5 | 41 | 0.11137 | 0.109476 | 0.215704 | -25.30% | -2.00% | 44 | - | OK |
| ETTh1 | all | 617 | 0.11199 | 0.117239 | 0.228160 | -20.00% | +4.95% | 44 | - | OK |
| ETTh1 | u5g | 42 | 0.11072 | 0.107708 | 0.212176 | -26.50% | -3.58% | 36 | refine 0.630 | OK |
| ETTh1 | allg | 619 | 0.10946 | 0.106430 | 0.212134 | -27.38% | -4.72% | 39 | phase -0.179, refine 0.618 | OK |
| ETTh1 | allrg | 618 | 0.11073 | 0.110436 | 0.217216 | -24.64% | -1.14% | 40 | refine 0.583 | OK |
| _ETTh1 local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.146550_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTh1 local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.111707_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| ETTm1 | base | - | 0.06719 | 0.062403 | 0.158274 | +0.00% | +47.81% | 127 | - | OK |
| ETTm1 | passes2 | 581 | 0.04717 | 0.041735 | 0.125962 | -33.12% | -1.14% | 160 | - | OK |
| ETTm1 | u5 | 581 | 0.04295 | 0.037048 | 0.117509 | -40.63% | -12.25% | 188 | - | OK |
| ETTm1 | all | 597 | 0.04254 | 0.036845 | 0.117155 | -40.96% | -12.73% | 166 | - | OK |
| ETTm1 | u5g | 582 | 0.04285 | 0.036931 | 0.117362 | -40.82% | -12.52% | 134 | refine 0.841 | OK |
| ETTm1 | allg | 599 | 0.04249 | 0.036781 | 0.117038 | -41.06% | -12.88% | 163 | phase 0.211, refine 0.877 | OK |
| ETTm1 | allrg **<-** | 598 | 0.04249 | 0.036777 | 0.116992 | -41.07% | -12.89% | 152 | refine 0.926 | OK |
| _ETTm1 local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.062403_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTm1 local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.042218_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| ETTh2 | base | - | 0.05699 | 0.066180 | 0.163095 | -0.00% | +4.08% | 26 | - | OK |
| ETTh2 | passes2 | 41 | 0.07887 | 0.099614 | 0.212439 | +50.52% | +56.66% | 27 | - | OK |
| ETTh2 | u5 | 41 | 0.07954 | 0.099680 | 0.212627 | +50.62% | +56.76% | 26 | - | OK |
| ETTh2 | all | 617 | 0.07351 | 0.093092 | 0.204663 | +40.67% | +46.40% | 30 | - | OK |
| ETTh2 | u5g | 42 | 0.05384 | 0.059713 | 0.149591 | -9.77% | -6.09% | 44 | refine 0.574 | OK |
| ETTh2 | allg | 619 | 0.05371 | 0.059621 | 0.149639 | -9.91% | -6.24% | 50 | phase 0.098, refine 0.575 | OK |
| ETTh2 | allrg **<-** | 618 | 0.05362 | 0.059616 | 0.150150 | -9.92% | -6.24% | 40 | refine 0.532 | OK |
| _ETTh2 local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.066180_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTh2 local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.063586_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| ETTm2 | base | - | 0.02812 | 0.030221 | 0.104586 | +0.00% | +16.27% | 128 | - | OK |
| ETTm2 | passes2 | 581 | 0.02443 | 0.025629 | 0.089871 | -15.20% | -1.40% | 159 | - | OK |
| ETTm2 | u5 | 581 | 0.02402 | 0.025095 | 0.087698 | -16.96% | -3.45% | 159 | - | OK |
| ETTm2 | all | 597 | 0.02402 | 0.025097 | 0.087705 | -16.95% | -3.44% | 189 | - | OK |
| ETTm2 | u5g | 582 | 0.02402 | 0.025098 | 0.087718 | -16.95% | -3.44% | 141 | refine 0.853 | OK |
| ETTm2 | allg | 599 | 0.02403 | 0.025100 | 0.087765 | -16.95% | -3.43% | 165 | phase 0.077, refine 0.863 | OK |
| ETTm2 | allrg **<-** | 598 | 0.02402 | 0.025082 | 0.087638 | -17.00% | -3.50% | 154 | refine 0.968 | OK |
| _ETTm2 local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.030221_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTm2 local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.025992_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| Weather | base | - | 0.07714 | 0.032362 | 0.055355 | -0.00% | +7.62% | 149 | - | OK |
| Weather | passes2 | 581 | 0.06554 | 0.029277 | 0.045821 | -9.53% | -2.64% | 178 | - | OK |
| Weather | u5 | 581 | 0.06255 | 0.028401 | 0.042738 | -12.24% | -5.55% | 193 | - | OK |
| Weather | all | 597 | 0.06109 | 0.028484 | 0.042895 | -11.98% | -5.28% | 227 | - | OK |
| Weather | u5g | 582 | 0.06257 | 0.028371 | 0.042737 | -12.33% | -5.65% | 131 | refine 0.906 | OK |
| Weather | allg **<-** | 599 | 0.06025 | 0.028245 | 0.042625 | -12.72% | -6.07% | 168 | phase 0.060, refine 0.880 | OK |
| Weather | allrg | 598 | 0.06037 | 0.028293 | 0.042645 | -12.57% | -5.91% | 169 | refine 0.837 | OK |
| _Weather local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.032362_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _Weather local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.030070_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| Electricity | base | - | 0.11337 | 0.116498 | 0.245634 | -0.00% | +60.45% | 186 | - | OK |
| Electricity | passes2 | 41 | 0.09747 | 0.097463 | 0.218777 | -16.34% | +34.23% | 272 | - | OK |
| Electricity | u5 | 41 | 0.09360 | 0.093482 | 0.213026 | -19.76% | +28.75% | 260 | - | OK |
| Electricity | all | 617 | 0.12157 | 0.125131 | 0.241181 | +7.41% | +72.34% | 290 | - | OK |
| Electricity | u5g | 42 | 0.09335 | 0.093182 | 0.212217 | -20.01% | +28.33% | 243 | refine 1.016 | OK |
| Electricity | allg | 619 | 0.09094 | 0.090458 | 0.209465 | -22.35% | +24.58% | 255 | phase -0.601, refine 1.099 | OK |
| Electricity | allrg **<-** | 618 | 0.08989 | 0.089342 | 0.208263 | -23.31% | +23.05% | 238 | refine 1.098 | OK |
| _Electricity local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.116498_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _Electricity local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.072609_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |

### mask_rate = 0.375

| Dataset | Variant | Params | Best vali | MSE | MAE | vs SparseTSF | vs TimeMixer | Runtime (s) | Learned gates | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| ETTh1 | base | - | 0.17957 | 0.179265 | 0.271485 | -0.00% | +43.01% | 32 | - | OK |
| ETTh1 | passes2 **<-** | 41 | 0.12861 | 0.128675 | 0.231401 | -28.22% | +2.65% | 39 | - | OK |
| ETTh1 | u5 | 41 | 0.13078 | 0.133712 | 0.238970 | -25.41% | +6.67% | 40 | - | OK |
| ETTh1 | all | 617 | 0.12914 | 0.134505 | 0.242356 | -24.97% | +7.31% | 42 | - | OK |
| ETTh1 | u5g | 42 | 0.13146 | 0.131169 | 0.233306 | -26.83% | +4.64% | 35 | refine 0.658 | OK |
| ETTh1 | allg | 619 | 0.12939 | 0.128119 | 0.232117 | -28.53% | +2.21% | 42 | phase -0.170, refine 0.684 | OK |
| ETTh1 | allrg | 618 | 0.12934 | 0.128010 | 0.232161 | -28.59% | +2.12% | 40 | refine 0.694 | OK |
| _ETTh1 local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.179265_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTh1 local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.125347_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| ETTm1 | base | - | 0.08149 | 0.078078 | 0.176335 | -0.00% | +60.35% | 129 | - | OK |
| ETTm1 | passes2 | 581 | 0.05706 | 0.052248 | 0.141639 | -33.08% | +7.30% | 141 | - | OK |
| ETTm1 | u5 | 581 | 0.04861 | 0.042976 | 0.125932 | -44.96% | -11.74% | 148 | - | OK |
| ETTm1 | all **<-** | 597 | 0.04811 | 0.042312 | 0.125413 | -45.81% | -13.10% | 166 | - | OK |
| ETTm1 | u5g | 582 | 0.04857 | 0.042921 | 0.125771 | -45.03% | -11.85% | 138 | refine 0.877 | OK |
| ETTm1 | allg | 599 | 0.04821 | 0.042464 | 0.125877 | -45.61% | -12.79% | 161 | phase 0.594, refine 1.082 | OK |
| ETTm1 | allrg | 598 | 0.04821 | 0.042429 | 0.125600 | -45.66% | -12.87% | 153 | refine 1.013 | OK |
| _ETTm1 local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.078078_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTm1 local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.048693_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| ETTh2 | base | - | 0.06286 | 0.073772 | 0.173912 | -0.00% | +7.79% | 26 | - | OK |
| ETTh2 | passes2 | 41 | 0.08036 | 0.100829 | 0.213557 | +36.68% | +47.33% | 26 | - | OK |
| ETTh2 | u5 | 41 | 0.08107 | 0.100837 | 0.213556 | +36.69% | +47.34% | 27 | - | OK |
| ETTh2 | all | 617 | 0.07499 | 0.094798 | 0.206233 | +28.50% | +38.51% | 30 | - | OK |
| ETTh2 | u5g **<-** | 42 | 0.05762 | 0.065037 | 0.158067 | -11.84% | -4.97% | 44 | refine 0.618 | OK |
| ETTh2 | allg | 619 | 0.07223 | 0.091247 | 0.200922 | +23.69% | +33.32% | 51 | phase 0.928, refine 1.169 | OK |
| ETTh2 | allrg | 618 | 0.07407 | 0.093897 | 0.205109 | +27.28% | +37.20% | 40 | refine 1.079 | OK |
| _ETTh2 local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.073772_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTh2 local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.068440_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| ETTm2 | base | - | 0.03179 | 0.034499 | 0.113122 | +0.00% | +20.49% | 134 | - | OK |
| ETTm2 | passes2 | 581 | 0.02740 | 0.029000 | 0.098329 | -15.94% | +1.28% | 156 | - | OK |
| ETTm2 | u5 | 581 | 0.02891 | 0.030293 | 0.098381 | -12.19% | +5.80% | 161 | - | OK |
| ETTm2 | all | 597 | 0.02828 | 0.029384 | 0.096326 | -14.83% | +2.62% | 190 | - | OK |
| ETTm2 | u5g | 582 | 0.02656 | 0.027805 | 0.093134 | -19.40% | -2.89% | 144 | refine 0.889 | OK |
| ETTm2 | allg | 599 | 0.02660 | 0.027734 | 0.092942 | -19.61% | -3.14% | 162 | phase 0.187, refine 0.943 | OK |
| ETTm2 | allrg **<-** | 598 | 0.02654 | 0.027695 | 0.092995 | -19.72% | -3.28% | 153 | refine 1.159 | OK |
| _ETTm2 local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.034499_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTm2 local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.028633_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| Weather | base | - | 0.09066 | 0.035990 | 0.061407 | +0.00% | +10.19% | 152 | - | OK |
| Weather | passes2 | 581 | 0.07703 | 0.032149 | 0.051384 | -10.67% | -1.57% | 180 | - | OK |
| Weather | u5 | 581 | 0.08684 | 0.034414 | 0.051374 | -4.38% | +5.36% | 172 | - | OK |
| Weather | all | 597 | 0.08398 | 0.033786 | 0.050840 | -6.12% | +3.44% | 241 | - | OK |
| Weather | u5g **<-** | 582 | 0.07162 | 0.030562 | 0.046008 | -15.08% | -6.43% | 137 | refine 1.031 | OK |
| Weather | allg | 599 | 0.07168 | 0.030576 | 0.045580 | -15.04% | -6.39% | 167 | phase 0.053, refine 1.002 | OK |
| Weather | allrg | 598 | 0.07176 | 0.030625 | 0.045519 | -14.91% | -6.24% | 169 | refine 0.921 | OK |
| _Weather local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.035990_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _Weather local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.032663_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| Electricity | base | - | 0.13302 | 0.138658 | 0.269219 | -0.00% | +62.70% | 192 | - | OK |
| Electricity | passes2 | 41 | 0.11425 | 0.116466 | 0.242612 | -16.01% | +36.66% | 329 | - | OK |
| Electricity | u5 | 41 | 0.10976 | 0.111595 | 0.234775 | -19.52% | +30.94% | 297 | - | OK |
| Electricity | all | 617 | 0.13213 | 0.137145 | 0.254888 | -1.09% | +60.93% | 279 | - | OK |
| Electricity | u5g **<-** | 42 | 0.10851 | 0.110202 | 0.232925 | -20.52% | +29.31% | 252 | refine 1.028 | OK |
| Electricity | allg | 619 | 0.13208 | 0.137631 | 0.255592 | -0.74% | +61.50% | 265 | phase -0.755, refine 1.124 | OK |
| Electricity | allrg | 618 | 0.13327 | 0.138896 | 0.257278 | +0.17% | +62.98% | 250 | refine 1.126 | OK |
| _Electricity local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.138658_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _Electricity local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.085223_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |

### mask_rate = 0.5

| Dataset | Variant | Params | Best vali | MSE | MAE | vs SparseTSF | vs TimeMixer | Runtime (s) | Learned gates | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| ETTh1 | base | 41 | 0.21422 | 0.214076 | 0.296108 | -0.00% | +46.19% | 40 | - | OK |
| ETTh1 | maskconv | 41 | 0.21300 | 0.213604 | 0.295786 | -0.22% | +45.87% | 41 | - | OK |
| ETTh1 | passes2 **<-** | 41 | 0.15777 | 0.159201 | 0.256549 | -25.63% | +8.72% | 43 | - | OK |
| ETTh1 | u5 | 41 | 0.15947 | 0.169165 | 0.269107 | -20.98% | +15.52% | 43 | - | OK |
| ETTh1 | phase | 617 | 0.21197 | 0.208082 | 0.295382 | -2.80% | +42.10% | 37 | - | OK |
| ETTh1 | all | 617 | 0.16330 | 0.171786 | 0.271741 | -19.75% | +17.31% | 47 | - | OK |
| ETTh1 | u5g | 42 | 0.16122 | 0.162700 | 0.258683 | -24.00% | +11.11% | 36 | refine 0.696 | OK |
| ETTh1 | allg | 619 | 0.15862 | 0.159736 | 0.257891 | -25.38% | +9.09% | 41 | phase -0.198, refine 0.724 | OK |
| ETTh1 | allrg | 618 | 0.15822 | 0.158673 | 0.257261 | -25.88% | +8.36% | 40 | refine 0.760 | OK |
| _ETTh1 local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.214076_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTh1 local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.146432_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| ETTm1 | base | 581 | 0.09850 | 0.096544 | 0.195342 | -0.00% | +66.05% | 137 | - | OK |
| ETTm1 | maskconv | 581 | 0.06748 | 0.062374 | 0.152209 | -35.39% | +7.28% | 134 | - | OK |
| ETTm1 | passes2 | 581 | 0.07273 | 0.068874 | 0.163173 | -28.66% | +18.46% | 144 | - | OK |
| ETTm1 | u5 | 581 | 0.06113 | 0.055998 | 0.148059 | -42.00% | -3.69% | 180 | - | OK |
| ETTm1 | phase | 597 | 0.09831 | 0.096229 | 0.195065 | -0.33% | +65.50% | 143 | - | OK |
| ETTm1 | all | 597 | 0.05949 | 0.054591 | 0.146038 | -43.45% | -6.11% | 190 | - | OK |
| ETTm1 | u5g | 582 | 0.05949 | 0.054034 | 0.142101 | -44.03% | -7.07% | 139 | refine 0.935 | OK |
| ETTm1 | allg **<-** | 599 | 0.05793 | 0.052645 | 0.140677 | -45.47% | -9.46% | 168 | phase 0.788, refine 1.028 | OK |
| ETTm1 | allrg | 598 | 0.05807 | 0.052758 | 0.140814 | -45.35% | -9.26% | 155 | refine 1.049 | OK |
| _ETTm1 local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.096544_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTm1 local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.058143_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| ETTh2 | base | - | 0.06920 | 0.082488 | 0.185551 | +0.00% | +10.09% | 43 | - | OK |
| ETTh2 | passes2 | 41 | 0.08205 | 0.102799 | 0.215418 | +24.62% | +37.20% | 25 | - | OK |
| ETTh2 | u5 | 41 | 0.06762 | 0.081580 | 0.183226 | -1.10% | +8.88% | 27 | - | OK |
| ETTh2 | all | 617 | 0.06799 | 0.081958 | 0.184157 | -0.64% | +9.39% | 30 | - | OK |
| ETTh2 | u5g **<-** | 42 | 0.06467 | 0.075279 | 0.175343 | -8.74% | +0.47% | 35 | refine 0.538 | OK |
| ETTh2 | allg | 619 | 0.07893 | 0.099440 | 0.211071 | +20.55% | +32.72% | 40 | phase 0.813, refine 0.700 | OK |
| ETTh2 | allrg | 618 | 0.07810 | 0.098849 | 0.210711 | +19.83% | +31.93% | 40 | refine 0.819 | OK |
| _ETTh2 local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.082488_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTh2 local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.074926_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| ETTm2 | base | - | 0.03597 | 0.039384 | 0.122035 | -0.00% | +21.60% | 131 | - | OK |
| ETTm2 | passes2 | 581 | 0.03139 | 0.033597 | 0.108182 | -14.69% | +3.74% | 147 | - | OK |
| ETTm2 | u5 | 581 | 0.03168 | 0.033516 | 0.107494 | -14.90% | +3.49% | 130 | - | OK |
| ETTm2 | all | 597 | 0.03157 | 0.033296 | 0.106585 | -15.46% | +2.81% | 195 | - | OK |
| ETTm2 | u5g | 582 | 0.03010 | 0.031716 | 0.101314 | -19.47% | -2.07% | 145 | refine 0.919 | OK |
| ETTm2 | allg | 599 | 0.02995 | 0.031523 | 0.101210 | -19.96% | -2.67% | 170 | phase 0.537, refine 1.150 | OK |
| ETTm2 | allrg **<-** | 598 | 0.02983 | 0.031390 | 0.100902 | -20.30% | -3.08% | 152 | refine 1.210 | OK |
| _ETTm2 local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.039384_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _ETTm2 local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.032387_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| Weather | base | - | 0.10859 | 0.040054 | 0.067667 | -0.00% | +10.27% | 138 | - | OK |
| Weather | passes2 | 581 | 0.09848 | 0.036328 | 0.058815 | -9.30% | +0.01% | 159 | - | OK |
| Weather | u5 | 581 | 0.10423 | 0.038116 | 0.059038 | -4.84% | +4.94% | 187 | - | OK |
| Weather | all | 597 | 0.10647 | 0.039526 | 0.059573 | -1.32% | +8.82% | 218 | - | OK |
| Weather | u5g | 582 | 0.08802 | 0.033810 | 0.051459 | -15.59% | -6.92% | 141 | refine 1.070 | OK |
| Weather | allg **<-** | 599 | 0.08659 | 0.034310 | 0.051555 | -14.34% | -5.54% | 168 | phase 0.069, refine 1.086 | OK |
| Weather | allrg | 598 | 0.08688 | 0.034369 | 0.051402 | -14.19% | -5.38% | 168 | refine 0.999 | OK |
| _Weather local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.040054_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _Weather local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.036323_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| Electricity | base | - | 0.15891 | 0.167128 | 0.296695 | +0.00% | +63.85% | 199 | - | OK |
| Electricity | passes2 | 41 | 0.13750 | 0.142230 | 0.270014 | -14.90% | +39.44% | 273 | - | OK |
| Electricity | u5 | 41 | 0.13194 | 0.136062 | 0.260992 | -18.59% | +33.40% | 277 | - | OK |
| Electricity | all | 617 | 0.14881 | 0.155295 | 0.274305 | -7.08% | +52.25% | 284 | - | OK |
| Electricity | u5g **<-** | 42 | 0.13085 | 0.134859 | 0.259398 | -19.31% | +32.22% | 257 | refine 1.051 | OK |
| Electricity | allg | 619 | 0.14821 | 0.155619 | 0.274361 | -6.89% | +52.57% | 269 | phase -0.771, refine 1.160 | OK |
| Electricity | allrg | 618 | 0.14962 | 0.157077 | 0.276112 | -6.01% | +54.00% | 257 | refine 1.152 | OK |
| _Electricity local sweep_ | _SparseTSF_ | _—_ | _—_ | _0.167128_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |
| _Electricity local sweep_ | _TimeMixer_ | _~58–60k_ | _—_ | _0.101999_ | _—_ | _—_ | _—_ | _—_ | _—_ | _reference_ |

Variant legend:

- **base** — all units off == SparseTSF · `(none)`
- **maskconv** — partial-conv renormalisation · `--stsf_mask_conv 1`
- **passes2** — weight-shared refinement pass · `--stsf_impute_passes 2`
- **u5** — both U5 parts · `--stsf_mask_conv 1 --stsf_impute_passes 2`
- **phase** — intra-period mixing · `--stsf_phase_mix linear`
- **all** — U5 + phase mixing · `--stsf_mask_conv 1 --stsf_impute_passes 2 --stsf_phase_mix linear`
- **u5g** — U5, refinement gated · `--stsf_mask_conv 1 --stsf_impute_passes 2 --stsf_refine_gate 1`
- **allg** — U5 + phase mixing, both gated · `--stsf_mask_conv 1 --stsf_impute_passes 2 --stsf_refine_gate 1 --stsf_phase_mix linear --stsf_phase_gate 1`
- **allrg** — U5 + phase mixing, refinement gated · `--stsf_mask_conv 1 --stsf_impute_passes 2 --stsf_refine_gate 1 --stsf_phase_mix linear`
<!-- AUTO_PLUS_IMP_END -->
