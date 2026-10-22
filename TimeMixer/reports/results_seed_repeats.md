# Seed repeats

Every earlier result in this study used one seed (2021). Here the headline comparisons are
repeated with seeds 2022 and 2023 by `benchmarks/run_seed_repeats.py`; seed 2021 is read from the
existing logs. Each repeated run is the exact command behind the seed-2021 number with only the
seed changed.

- **Configurations:** SparseTSF; the fixed one-model SparseTSFPlus; gated refinement alone
  (`u5g`, the per-task imputation form, imputation only); and any candidate configuration.
- **Tasks:** long-term forecasting on ETTh1, ETTh2, ETTm1, ETTm2 and Weather, M4, PEMS, few-shot,
  zero-shot, anomaly detection, and imputation on ETTh1 and ETTh2. Solar, Electricity and Traffic
  long-term, the other imputation datasets and every TimeMixer repeat are left out for cost.
- **What counts as a finding:** a difference with the same sign on all three seeds whose mean is
  larger than twice the standard deviation of the per-seed differences.
- Imputation, anomaly detection and classification early-stop on test (upstream behaviour), so
  read those rows as a sensitivity analysis.

```bash
venv/Scripts/python.exe benchmarks/run_seed_repeats.py
```

<!-- AUTO_SEEDS_START -->
### Per task, per seed (mean over the task's cells)

| Task | Metric | Configuration | 2021 | 2022 | 2023 | Mean ± std | Spread (% of mean) |
|---|---|---|---:|---:|---:|---:|---:|
| Long-term (ETT×4, Weather; 4 horizons) | MSE ↓ | SparseTSF | 0.3542 | 0.3542 | 0.3530 | 0.3538 ± 0.0007 | 0.2% |
| Long-term (ETT×4, Weather; 4 horizons) | MSE ↓ | fixed | 0.3505 | 0.3500 | 0.3504 | 0.3503 ± 0.0003 | 0.1% |
| M4 (6 subsets) | OWA ↓ | SparseTSF | 0.972 | 0.970 | 0.978 | 0.973 ± 0.004 | 0.4% |
| M4 (6 subsets) | OWA ↓ | fixed | 0.914 | 0.916 | 0.913 | 0.914 ± 0.002 | 0.2% |
| PEMS (4 datasets) | MAE ↓ | SparseTSF | 34.13 | 34.12 | 34.12 | 34.12 ± 0.01 | 0.0% |
| PEMS (4 datasets) | MAE ↓ | fixed | 25.45 | 25.99 | 25.93 | 25.79 ± 0.30 | 1.1% |
| PEMS (4 datasets) | MAE ↓ | chmean | 24.46 | 25.31 | 25.19 | 24.99 ± 0.46 | 1.8% |
| PEMS (4 datasets) | MAE ↓ | chcorr | 24.20 | 24.57 | 24.54 | 24.44 ± 0.21 | 0.8% |
| Few-shot (6 datasets) | MSE ↓ | SparseTSF | 0.3763 | 0.3740 | 0.3737 | 0.3747 ± 0.0014 | 0.4% |
| Few-shot (6 datasets) | MSE ↓ | fixed | 0.3687 | 0.3693 | 0.3701 | 0.3694 ± 0.0007 | 0.2% |
| Zero-shot (6 pairs) | MSE ↓ | SparseTSF | 0.3929 | 0.3920 | 0.3943 | 0.3931 ± 0.0011 | 0.3% |
| Zero-shot (6 pairs) | MSE ↓ | fixed | 0.3912 | 0.3913 | 0.3917 | 0.3914 ± 0.0003 | 0.1% |
| Anomaly (4 datasets) | F1 ↑ | SparseTSF | 79.96 | 81.45 | 80.00 | 80.47 ± 0.85 | 1.1% |
| Anomaly (4 datasets) | F1 ↑ | fixed | 79.98 | 81.04 | 80.45 | 80.49 ± 0.53 | 0.7% |
| Imputation (ETTh1, ETTh2; 4 mask rates) | MSE ↓ | SparseTSF | 0.1168 | 0.1146 | 0.1139 | 0.1151 ± 0.0015 | 1.3% |
| Imputation (ETTh1, ETTh2; 4 mask rates) | MSE ↓ | fixed | 0.0996 | 0.0926 | 0.0973 | 0.0965 ± 0.0035 | 3.7% |
| Imputation (ETTh1, ETTh2; 4 mask rates) | MSE ↓ | u5g | 0.0935 | 0.0945 | 0.0959 | 0.0947 ± 0.0012 | 1.3% |
| Imputation (ETTh1, ETTh2; 4 mask rates) | MSE ↓ | allrg_pm | 0.0951 | 0.0924 | 0.0953 | 0.0943 ± 0.0016 | 1.7% |
| Imputation (ETTh1, ETTh2; 4 mask rates) | MSE ↓ | chmean | 0.0996 | - | - | - | - |
| Imputation (ETTh1, ETTh2; 4 mask rates) | MSE ↓ | chcorr | 0.0996 | - | - | - | - |
| Imputation (Electricity; 4 mask rates) | MSE ↓ | SparseTSF | 0.1304 | - | - | - | - |
| Imputation (Electricity; 4 mask rates) | MSE ↓ | fixed | 0.1158 | - | - | - | - |
| Imputation (Electricity; 4 mask rates) | MSE ↓ | u5g | 0.1048 | - | - | - | - |
| Imputation (Electricity; 4 mask rates) | MSE ↓ | allrg_pm | 0.1058 | - | - | - | - |
| Imputation (Electricity; 4 mask rates) | MSE ↓ | chmean | 0.1158 | - | - | - | - |
| Imputation (Electricity; 4 mask rates) | MSE ↓ | chcorr | 0.1158 | - | - | - | - |

### Comparisons (relative difference per seed)

Negative is better for error metrics; for anomaly F1 the difference is in points and positive is better.

| Task | Comparison | 2021 | 2022 | 2023 | Mean | Std | Finding? |
|---|---|---:|---:|---:|---:|---:|---|
| Long-term (ETT×4, Weather; 4 horizons) | fixed vs SparseTSF | -1.05% | -1.21% | -0.73% | -1.00% | 0.24% | yes |
| M4 (6 subsets) | fixed vs SparseTSF | -5.97% | -5.57% | -6.65% | -6.06% | 0.55% | yes |
| PEMS (4 datasets) | fixed vs SparseTSF | -25.43% | -23.81% | -24.02% | -24.42% | 0.88% | yes |
| PEMS (4 datasets) | chmean vs fixed | -3.88% | -2.65% | -2.84% | -3.12% | 0.66% | yes |
| PEMS (4 datasets) | chcorr vs fixed | -4.92% | -5.46% | -5.35% | -5.25% | 0.29% | yes |
| Few-shot (6 datasets) | fixed vs SparseTSF | -2.01% | -1.26% | -0.96% | -1.41% | 0.54% | yes |
| Zero-shot (6 pairs) | fixed vs SparseTSF | -0.42% | -0.18% | -0.66% | -0.42% | 0.24% | no |
| Anomaly (4 datasets) | fixed vs SparseTSF | +0.03 pts | -0.41 pts | +0.45 pts | +0.02 pts | 0.43 pts | no |
| Imputation (ETTh1, ETTh2; 4 mask rates) | fixed vs SparseTSF | -14.75% | -19.20% | -14.57% | -16.17% | 2.62% | yes |
| Imputation (ETTh1, ETTh2; 4 mask rates) | u5g vs SparseTSF | -19.91% | -17.58% | -15.74% | -17.74% | 2.09% | yes |
| Imputation (ETTh1, ETTh2; 4 mask rates) | u5g vs fixed | -6.06% | +2.00% | -1.37% | -1.81% | 4.05% | no |
| Imputation (ETTh1, ETTh2; 4 mask rates) | allrg_pm vs fixed | -4.46% | -0.26% | -2.01% | -2.25% | 2.11% | no |
| Imputation (ETTh1, ETTh2; 4 mask rates) | chmean vs fixed | +0.00% | - | - | - | - | incomplete |
| Imputation (ETTh1, ETTh2; 4 mask rates) | chcorr vs fixed | +0.00% | - | - | - | - | incomplete |
| Imputation (ETTh1, ETTh2; 4 mask rates) | allrg_pm vs u5g | +1.69% | -2.22% | -0.66% | -0.39% | 1.97% | no |
| Imputation (Electricity; 4 mask rates) | fixed vs SparseTSF | -11.18% | - | - | - | - | incomplete |
| Imputation (Electricity; 4 mask rates) | u5g vs SparseTSF | -19.64% | - | - | - | - | incomplete |
| Imputation (Electricity; 4 mask rates) | u5g vs fixed | -9.53% | - | - | - | - | incomplete |
| Imputation (Electricity; 4 mask rates) | allrg_pm vs fixed | -8.67% | - | - | - | - | incomplete |
| Imputation (Electricity; 4 mask rates) | chmean vs fixed | +0.00% | - | - | - | - | incomplete |
| Imputation (Electricity; 4 mask rates) | chcorr vs fixed | +0.00% | - | - | - | - | incomplete |
| Imputation (Electricity; 4 mask rates) | allrg_pm vs u5g | +0.95% | - | - | - | - | incomplete |

<!-- AUTO_SEEDS_END -->
