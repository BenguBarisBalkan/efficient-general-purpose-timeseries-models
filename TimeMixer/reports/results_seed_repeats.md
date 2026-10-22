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

## Findings

**The one-model SparseTSFPlus beats SparseTSF on five tasks, over three seeds.** The fixed
configuration's gain is a finding on PEMS (−24.4% ± 0.9%), imputation on ETTh1/ETTh2
(−16.2% ± 2.6%), M4 (−6.1% ± 0.6%), few-shot (−1.4% ± 0.5%) and long-term forecasting
(−1.0% ± 0.2%). It is not a finding on zero-shot (−0.42% ± 0.24%: the same sign on every seed, but
under twice the spread) or anomaly detection (+0.02 F1 points).

**One configuration costs nothing measurable on the forecasting tasks.** Against the per-task
version, every difference is under 0.7%:
- long-term: fixed is better (−0.37% ± 0.13%, a finding);
- M4 and few-shot: no finding (−0.69% ± 0.63%, −0.35% ± 0.51%);
- zero-shot: per task is better by a hair (+0.24% ± 0.07%, a finding).

**Seed 2021 was not representative everywhere.** The fixed configuration's PEMS mean over three
seeds is 25.79, not the 25.45 reported from seed 2021, and its Electricity imputation mean is
0.1067, not 0.1158.

**Imputation differences between SparseTSFPlus variants are mostly run-to-run noise.** The fixed
imputation form varies by 3.7% across seeds on ETTh1/ETTh2 and by 7.4% on Electricity. Against it:

| Variant | ETTh1/ETTh2 | Electricity |
|---|---:|---:|
| Mask-aware phase mixing (`allrg_pm`) | −2.3% ± 2.1% | −6.0% ± 7.4% |
| No phase mixing (`u5g`) | −1.8% ± 4.1% | −5.4% ± 5.0% |

None of these is a finding. Two consequences:
- **Phase mixing's cost in imputation is not established.** Measured against the same model
  without it, it was +20.1% on ETTh2 at seed 2021 but −2.7% and −2.0% at seeds 2022 and 2023. On
  Electricity it was +10.5%, +7.4% and −0.1%. The bad ETTh2 cells come from a training plateau that
  strikes at random (all three variants, including the one without phase mixing, hit it at
  mask 0.25 on seed 2023).
- **Mask-aware phase mixing has one consistent effect:** on Electricity at mask 0.5 it is best on
  all three seeds (0.1108–0.1158 against 0.1293–0.1571 for the others), and at mask 0.375 it is
  worse on all three.

**TimeMixer, repeated on PEMS and anomaly detection.** Its PEMS lead is a finding (the fixed
configuration is +47.7% ± 2.0% behind; TimeMixer 17.46 ± 0.06 MAE). Its anomaly F1 was 75.49,
83.55 and 83.39: the seed-2021 MSL run failed (48.0 against 80.2 and 79.4 on the other seeds). So
the fixed configuration's seed-2021 anomaly win (+4.5 points) does not hold: −2.5 and −2.9 points on
the other seeds, not a finding either way. TimeMixer is ahead on PSM, SMD and SMAP on every seed.

**Channel correlation's PEMS gain is real.** Against the fixed configuration it is −5.25% ± 0.29%
on PEMS (a finding), and channel mean −3.12% ± 0.66% (also a finding). The one-seed screen had
rejected both for missing the 5% validation threshold (channel correlation reached 4.7%).

<!-- AUTO_SEEDS_START -->
### Per task, per seed (mean over the task's cells)

| Task | Metric | Configuration | 2021 | 2022 | 2023 | Mean ± std | Spread (% of mean) |
|---|---|---|---:|---:|---:|---:|---:|
| Long-term (ETT×4, Weather; 4 horizons) | MSE ↓ | SparseTSF | 0.3542 | 0.3542 | 0.3530 | 0.3538 ± 0.0007 | 0.2% |
| Long-term (ETT×4, Weather; 4 horizons) | MSE ↓ | fixed | 0.3505 | 0.3500 | 0.3504 | 0.3503 ± 0.0003 | 0.1% |
| Long-term (ETT×4, Weather; 4 horizons) | MSE ↓ | task | 0.3514 | 0.3517 | 0.3517 | 0.3516 ± 0.0002 | 0.0% |
| M4 (6 subsets) | OWA ↓ | SparseTSF | 0.972 | 0.970 | 0.978 | 0.973 ± 0.004 | 0.4% |
| M4 (6 subsets) | OWA ↓ | fixed | 0.914 | 0.916 | 0.913 | 0.914 ± 0.002 | 0.2% |
| M4 (6 subsets) | OWA ↓ | task | 0.918 | 0.918 | 0.926 | 0.921 ± 0.005 | 0.5% |
| PEMS (4 datasets) | MAE ↓ | SparseTSF | 34.13 | 34.12 | 34.12 | 34.12 ± 0.01 | 0.0% |
| PEMS (4 datasets) | MAE ↓ | fixed | 25.45 | 25.99 | 25.93 | 25.79 ± 0.30 | 1.1% |
| PEMS (4 datasets) | MAE ↓ | chmean | 24.46 | 25.31 | 25.19 | 24.99 ± 0.46 | 1.8% |
| PEMS (4 datasets) | MAE ↓ | chcorr | 24.20 | 24.57 | 24.54 | 24.44 ± 0.21 | 0.8% |
| PEMS (4 datasets) | MAE ↓ | TimeMixer | 17.51 | 17.48 | 17.39 | 17.46 ± 0.06 | 0.3% |
| Few-shot (6 datasets) | MSE ↓ | SparseTSF | 0.3763 | 0.3740 | 0.3737 | 0.3747 ± 0.0014 | 0.4% |
| Few-shot (6 datasets) | MSE ↓ | fixed | 0.3687 | 0.3693 | 0.3701 | 0.3694 ± 0.0007 | 0.2% |
| Few-shot (6 datasets) | MSE ↓ | task | 0.3688 | 0.3728 | 0.3705 | 0.3707 ± 0.0020 | 0.5% |
| Zero-shot (6 pairs) | MSE ↓ | SparseTSF | 0.3929 | 0.3920 | 0.3943 | 0.3931 ± 0.0011 | 0.3% |
| Zero-shot (6 pairs) | MSE ↓ | fixed | 0.3912 | 0.3913 | 0.3917 | 0.3914 ± 0.0003 | 0.1% |
| Zero-shot (6 pairs) | MSE ↓ | task | 0.3905 | 0.3902 | 0.3907 | 0.3905 ± 0.0003 | 0.1% |
| Anomaly (4 datasets) | F1 ↑ | SparseTSF | 79.96 | 81.45 | 80.00 | 80.47 ± 0.85 | 1.1% |
| Anomaly (4 datasets) | F1 ↑ | fixed | 79.98 | 81.04 | 80.45 | 80.49 ± 0.53 | 0.7% |
| Anomaly (4 datasets) | F1 ↑ | TimeMixer | 75.49 | 83.55 | 83.39 | 80.81 ± 4.61 | 5.7% |
| Imputation (ETTh1, ETTh2; 4 mask rates) | MSE ↓ | SparseTSF | 0.1168 | 0.1146 | 0.1139 | 0.1151 ± 0.0015 | 1.3% |
| Imputation (ETTh1, ETTh2; 4 mask rates) | MSE ↓ | fixed | 0.0996 | 0.0926 | 0.0973 | 0.0965 ± 0.0035 | 3.7% |
| Imputation (ETTh1, ETTh2; 4 mask rates) | MSE ↓ | u5g | 0.0935 | 0.0945 | 0.0959 | 0.0947 ± 0.0012 | 1.3% |
| Imputation (ETTh1, ETTh2; 4 mask rates) | MSE ↓ | allrg_pm | 0.0951 | 0.0924 | 0.0953 | 0.0943 ± 0.0016 | 1.7% |
| Imputation (ETTh1, ETTh2; 4 mask rates) | MSE ↓ | chmean | 0.0996 | - | - | - | - |
| Imputation (ETTh1, ETTh2; 4 mask rates) | MSE ↓ | chcorr | 0.0996 | - | - | - | - |
| Imputation (Electricity; 4 mask rates) | MSE ↓ | SparseTSF | 0.1304 | - | - | - | - |
| Imputation (Electricity; 4 mask rates) | MSE ↓ | fixed | 0.1158 | 0.1031 | 0.1012 | 0.1067 ± 0.0079 | 7.4% |
| Imputation (Electricity; 4 mask rates) | MSE ↓ | u5g | 0.1048 | 0.0960 | 0.1014 | 0.1007 ± 0.0044 | 4.4% |
| Imputation (Electricity; 4 mask rates) | MSE ↓ | allrg_pm | 0.1058 | 0.0910 | 0.1036 | 0.1001 ± 0.0079 | 7.9% |
| Imputation (Electricity; 4 mask rates) | MSE ↓ | chmean | 0.1158 | - | - | - | - |
| Imputation (Electricity; 4 mask rates) | MSE ↓ | chcorr | 0.1158 | - | - | - | - |

### Comparisons (relative difference per seed)

Negative is better for error metrics; for anomaly F1 the difference is in points and positive is better.

| Task | Comparison | 2021 | 2022 | 2023 | Mean | Std | Finding? |
|---|---|---:|---:|---:|---:|---:|---|
| Long-term (ETT×4, Weather; 4 horizons) | fixed vs SparseTSF | -1.05% | -1.21% | -0.73% | -1.00% | 0.24% | yes |
| Long-term (ETT×4, Weather; 4 horizons) | fixed vs task | -0.25% | -0.50% | -0.36% | -0.37% | 0.13% | yes |
| M4 (6 subsets) | fixed vs SparseTSF | -5.97% | -5.57% | -6.65% | -6.06% | 0.55% | yes |
| M4 (6 subsets) | fixed vs task | -0.44% | -0.22% | -1.40% | -0.69% | 0.63% | no |
| PEMS (4 datasets) | fixed vs SparseTSF | -25.43% | -23.81% | -24.02% | -24.42% | 0.88% | yes |
| PEMS (4 datasets) | fixed vs TimeMixer | +45.35% | +48.69% | +49.05% | +47.70% | 2.04% | yes |
| PEMS (4 datasets) | SparseTSF vs TimeMixer | +94.92% | +95.16% | +96.17% | +95.42% | 0.66% | yes |
| PEMS (4 datasets) | chmean vs fixed | -3.88% | -2.65% | -2.84% | -3.12% | 0.66% | yes |
| PEMS (4 datasets) | chcorr vs fixed | -4.92% | -5.46% | -5.35% | -5.25% | 0.29% | yes |
| Few-shot (6 datasets) | fixed vs SparseTSF | -2.01% | -1.26% | -0.96% | -1.41% | 0.54% | yes |
| Few-shot (6 datasets) | fixed vs task | -0.01% | -0.94% | -0.11% | -0.35% | 0.51% | no |
| Zero-shot (6 pairs) | fixed vs SparseTSF | -0.42% | -0.18% | -0.66% | -0.42% | 0.24% | no |
| Zero-shot (6 pairs) | fixed vs task | +0.17% | +0.29% | +0.26% | +0.24% | 0.07% | yes |
| Anomaly (4 datasets) | fixed vs SparseTSF | +0.03 pts | -0.41 pts | +0.45 pts | +0.02 pts | 0.43 pts | no |
| Anomaly (4 datasets) | fixed vs TimeMixer | +4.49 pts | -2.52 pts | -2.94 pts | -0.32 pts | 4.17 pts | no |
| Anomaly (4 datasets) | SparseTSF vs TimeMixer | +4.47 pts | -2.11 pts | -3.39 pts | -0.34 pts | 4.22 pts | no |
| Imputation (ETTh1, ETTh2; 4 mask rates) | fixed vs SparseTSF | -14.75% | -19.20% | -14.57% | -16.17% | 2.62% | yes |
| Imputation (ETTh1, ETTh2; 4 mask rates) | u5g vs SparseTSF | -19.91% | -17.58% | -15.74% | -17.74% | 2.09% | yes |
| Imputation (ETTh1, ETTh2; 4 mask rates) | u5g vs fixed | -6.06% | +2.00% | -1.37% | -1.81% | 4.05% | no |
| Imputation (ETTh1, ETTh2; 4 mask rates) | allrg_pm vs fixed | -4.46% | -0.26% | -2.01% | -2.25% | 2.11% | no |
| Imputation (ETTh1, ETTh2; 4 mask rates) | chmean vs fixed | +0.00% | - | - | - | - | incomplete |
| Imputation (ETTh1, ETTh2; 4 mask rates) | chcorr vs fixed | +0.00% | - | - | - | - | incomplete |
| Imputation (ETTh1, ETTh2; 4 mask rates) | allrg_pm vs u5g | +1.69% | -2.22% | -0.66% | -0.39% | 1.97% | no |
| Imputation (Electricity; 4 mask rates) | fixed vs SparseTSF | -11.18% | - | - | - | - | incomplete |
| Imputation (Electricity; 4 mask rates) | u5g vs SparseTSF | -19.64% | - | - | - | - | incomplete |
| Imputation (Electricity; 4 mask rates) | u5g vs fixed | -9.53% | -6.90% | +0.13% | -5.43% | 4.99% | no |
| Imputation (Electricity; 4 mask rates) | allrg_pm vs fixed | -8.67% | -11.69% | +2.29% | -6.02% | 7.36% | no |
| Imputation (Electricity; 4 mask rates) | chmean vs fixed | +0.00% | - | - | - | - | incomplete |
| Imputation (Electricity; 4 mask rates) | chcorr vs fixed | +0.00% | - | - | - | - | incomplete |
| Imputation (Electricity; 4 mask rates) | allrg_pm vs u5g | +0.95% | -5.14% | +2.16% | -0.68% | 3.91% | no |

<!-- AUTO_SEEDS_END -->
