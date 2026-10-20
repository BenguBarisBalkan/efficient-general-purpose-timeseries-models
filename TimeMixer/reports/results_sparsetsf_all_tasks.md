# SparseTSF across the TimeMixer++ task suite — consolidated

One page linking every per-task benchmark run locally for **SparseTSF** (ICML'24, ~1k parameters)
and **TimeMixer** (ICLR'24), against the 8 task categories the TimeMixer++ paper (ICLR'25) evaluates.

## Scope and how to read this

- **SparseTSF and TimeMixer are both run locally on every task**, same machine and same protocol.
  TimeMixer uses its upstream script settings where they exist (long-term, M4, PEMS) and its own
  long-term architecture elsewhere (`benchmarks/tm_config.py`). On anomaly detection and
  classification TimeMixer publishes no settings at all, so those rows are untuned local references.
- Where both a local and a published TimeMixer figure exist, they agree closely (M4 OWA 0.841 vs
  0.840, PEMS MAE 17.51 vs 17.41, zero-shot MSE 0.448 vs 0.467). The exception is few-shot ETT
  (0.513 vs 0.453), where the shared few-shot protocol (batch 32) does not suit TimeMixer's
  full-data learning rate.
- **TimeMixer++** figures are the *published* values from its paper, in *italics*: produced with the
  authors' tuning on their hardware, so they are reference points, not a controlled head-to-head.
  TimeMixer++ is not re-run per task because it is stable only at `lr=5e-5`, which undertrains it.
- Hardware: RTX 3050 Ti Laptop (4 GB). Batch sizes are reduced from published values where needed.
- Every runner is idempotent and regenerates its table between `<!-- AUTO_* -->` markers.

## Task coverage

| # | Task | Report | Metric | Status |
|---|---|---|---|---|
| 1 | Long-term forecasting | [results_comparison_sparsetsf.md](results_comparison_sparsetsf.md) | MSE/MAE | ✅ 8 datasets (5 also at sl=720) |
| 2 | Univariate short-term (M4) | [results_comparison_m4.md](results_comparison_m4.md) | SMAPE/MASE/OWA | ✅ 6 subsets |
| 3 | Multivariate short-term (PEMS) | [results_comparison_pems.md](results_comparison_pems.md) | MAE/MAPE/RMSE | ✅ 4 datasets |
| 4 | Imputation | [results_comparison_imputation.md](results_comparison_imputation.md) | masked MSE/MAE | ✅ 6 datasets × 4 mask rates |
| 5 | Few-shot (10% train) | [results_comparison_fewshot.md](results_comparison_fewshot.md) | MSE/MAE | ✅ 6 datasets × 4 horizons |
| 6 | Zero-shot (transfer) | [results_comparison_zeroshot.md](results_comparison_zeroshot.md) | MSE/MAE | ✅ 6 pairs × 4 horizons |
| 7 | Anomaly detection | [results_comparison_anomaly.md](results_comparison_anomaly.md) | point-adjusted F1 | ⚠️ 4 of 5: PSM, SMD, MSL, SMAP (SWaT gated) |
| 8 | Classification | [results_comparison_classification.md](results_comparison_classification.md) | accuracy | ✅ 10 UEA datasets |

Every dataset in the study — what it is, measured sizes, and why it was or wasn't usable — is
documented in [datasets_reference.md](datasets_reference.md). **How SparseTSF was adapted to each
task** (the new heads, the design decisions, the traps) is in
[methodology_sparsetsf_tasks.md](methodology_sparsetsf_tasks.md).

Energy/carbon for the 3 models lives separately in
[results_carbon_comparison.md](results_carbon_comparison.md); the original 3-model accuracy study is
[results_final_report.md](results_final_report.md).

## Protocol deviations worth stating in a write-up

1. **Imputation input length.** The paper uses `seq_len=1024`; these tables use **96**. SparseTSF
   requires `period_len` to divide `seq_len`, and 1024 admits only powers of two — which cannot
   align to the true 24-hour cycle of the hourly datasets. Running the paper's length would force
   SparseTSF off its own inductive bias; that constraint is itself a finding.
2. **Zero-shot `period_len`.** The source dataset's `period_len` is reused for the target, because
   SparseTSF's linear layer is shaped `(seq_len/period_len → pred_len/period_len)` and the
   checkpoint must load. All horizons used are divisible by both 24 and 4, so every pair is valid.
3. **Anomaly coverage.** SWaT is gated behind the iTrust request form, so the average F1 here is
   over **4 of 5** datasets and is not directly comparable to the paper's 5-dataset average. MSL and
   SMAP are rebuilt from a mirror of NASA's release at exactly the published sizes, with anomaly
   rates slightly below the cited ones (10.53% vs 10.72%, 12.79% vs 13.13%).
4. **Classification head.** SparseTSF has no published classification head; the one used here
   (conv aggregation → flatten → linear) was written for this study. Weak results measure the
   architecture's fit to the task, not a tuned upper bound.

## Headline findings

All comparisons below are against the **local** TimeMixer unless marked published.

**1. SparseTSF's small capacity is a generalization *advantage*, not just an efficiency trade.**
It **beats TimeMixer on all 6 zero-shot transfer pairs** (mean MSE 0.393 vs 0.448), the strongest
claim here and now a same-machine result. On few-shot ETT it is ahead too (0.441 vs 0.513; vs the
published 0.453 the margin is a modest ~3%), while TimeMixer leads on few-shot Weather (0.254 vs
0.271). With ~1k parameters there is very little to overfit to the source dataset, which is the
regime where data is scarce or the target distribution shifts.

**2. It loses where the task is not forecasting-shaped, or where channels interact.**
- **PEMS** (MAE 34.13 vs 17.51): traffic networks carry their signal in *cross-channel* correlations,
  and SparseTSF is deliberately channel-independent. TimeMixer here mixes channels and has
  0.87–1.24 M parameters against 21, yet trains in only about twice the time.
- **M4** (OWA 0.972 vs 0.841): it beats the Naive2 baseline, but M4's short, heterogeneous series
  give its cross-period mechanism little to exploit.
- **Long-term at seq_len 96 on Solar, Electricity and Traffic:** behind at every horizon, with only
  four periods of context in the window.

**3. Classification is weak for both locally run models.** SparseTSF averages 60.90% and an untuned
TimeMixer 62.37%, both far below TimeMixer++'s published 75.9%. SparseTSF is strong where a periodic
signature is enough (SpokenArabicDigits 94.0, PEMS-SF 84.4) and collapses where it is not
(Handwriting 8.6% over 26 classes, as does TimeMixer at 6.4%).

**4. Anomaly detection is mixed.** Over the four obtainable datasets SparseTSF averages F1 79.96
against TimeMixer's 75.49, with 41 parameters against ~62,600–62,800. The average hides that
TimeMixer wins three of the four (PSM, SMD, SMAP), mostly on recall. SparseTSF's lead comes
entirely from MSL, where the untuned TimeMixer catches only a third of the anomalies (F1 48.0
against 77.8). Against TimeMixer++'s published figures on the same four datasets (85.75), SparseTSF
is 5.8 points behind. (SWaT is gated, so this covers 4 of 5 datasets.)

**5. Imputation is a clean efficiency trade:** TimeMixer wins all 24 cells and SparseTSF's masked MSE
is ~35% higher over the six datasets, at about **6.5× less training time**.

<!-- AUTO_SUMMARY_START -->
| Task | Datasets | Metric | SparseTSF (local) | TimeMixer (local) | Reference (paper) |
|---|---|---|---:|---:|---:|
| Univariate short-term | M4 (6 subsets) | OWA (lower=better) | 0.972 | 0.841 | _TimeMixer 0.840_ |
| Multivariate short-term | PEMS03/04/07/08 | MAE (lower) | 34.13 | 17.51 | _TimeMixer 17.41_ |
| Imputation | ETT x4, Weather, ECL | masked MSE (lower) | 0.0838 (6 ds) | 0.0622 (6 ds) | — |
| Few-shot (10% train) | ETT (Avg) | MSE (lower) | 0.441 | 0.513 | _TimeMixer 0.453_ |
| Zero-shot (transfer) | 6 ETT pairs | MSE (lower) | 0.393 | 0.448 | _TimeMixer 0.467 (avg)_ |
| Anomaly detection | PSM, SMD, MSL, SMAP (of 5) | F1 (higher) | 79.96 | 75.49 | _TimeMixer++ 85.75 (same 4 ds)_ |
| Classification | 10 UEA | accuracy (higher) | 60.90 | 62.37 | _TimeMixer++ 75.9_ |
<!-- AUTO_SUMMARY_END -->
