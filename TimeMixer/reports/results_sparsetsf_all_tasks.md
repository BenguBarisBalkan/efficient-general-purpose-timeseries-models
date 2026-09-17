# SparseTSF across the TimeMixer++ task suite — consolidated

One page linking every per-task benchmark run locally for **SparseTSF** (ICML'24, ~1k parameters),
against the 8 task categories the TimeMixer++ paper (ICLR'25) evaluates.

## Scope and how to read this

- **Only SparseTSF was run locally.** Every TimeMixer / TimeMixer++ figure quoted in these tables is
  the **published** value from the TimeMixer++ paper, shown in *italics*. Those were produced with
  the authors' own tuning, on their hardware, and (for imputation) a different input length — so
  they are **reference points, not like-for-like comparisons**. Do not present them as a controlled
  head-to-head.
- Hardware: RTX 3050 Ti Laptop (4 GB). Batch sizes are reduced from published values where needed.
- Every runner is idempotent and regenerates its table between `<!-- AUTO_* -->` markers.

## Task coverage

| # | Task | Report | Metric | Status |
|---|---|---|---|---|
| 1 | Long-term forecasting | [results_comparison_sparsetsf.md](results_comparison_sparsetsf.md) | MSE/MAE | ✅ 5 datasets × 2 lookbacks |
| 2 | Univariate short-term (M4) | [results_comparison_m4.md](results_comparison_m4.md) | SMAPE/MASE/OWA | ✅ 6 subsets |
| 3 | Multivariate short-term (PEMS) | [results_comparison_pems.md](results_comparison_pems.md) | MAE/MAPE/RMSE | ✅ 4 datasets |
| 4 | Imputation | [results_comparison_imputation.md](results_comparison_imputation.md) | masked MSE/MAE | ✅ 6 datasets × 4 mask rates |
| 5 | Few-shot (10% train) | [results_comparison_fewshot.md](results_comparison_fewshot.md) | MSE/MAE | ⚠️ 23 of 24 (ECL@720 exceeds RAM) |
| 6 | Zero-shot (transfer) | [results_comparison_zeroshot.md](results_comparison_zeroshot.md) | MSE/MAE | ✅ 6 pairs × 4 horizons |
| 7 | Anomaly detection | [results_comparison_anomaly.md](results_comparison_anomaly.md) | point-adjusted F1 | ⚠️ PSM + SMD (MSL/SMAP/SWaT unobtainable) |
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
3. **Anomaly coverage.** MSL/SMAP need the TimesNet preprocessed `.npy` bundle (the original
   telemanom S3 object now returns HTTP 403) and SWaT is gated behind the iTrust request form, so
   the average F1 here is over a **subset** and is not comparable to the paper's 5-dataset average.
4. **Classification head.** SparseTSF has no published classification head; the one used here
   (conv aggregation → flatten → linear) was written for this study. Weak results measure the
   architecture's fit to the task, not a tuned upper bound.

## Headline findings

**1. SparseTSF's small capacity is a generalization *advantage*, not just an efficiency trade.**
It edges out the published TimeMixer on **few-shot** ETT (0.441 vs 0.453 — a modest ~3%, and across
protocols, so treat it as "comparable", not a decisive win) and **beats published TimeMixer on all 6
zero-shot transfer pairs** (avg 0.393 vs 0.467, a much clearer margin), even edging TimeMixer++ on 3
of 6. With ~1k parameters there is very little to overfit to the source dataset — exactly the regime
where data is scarce or the target distribution shifts. Zero-shot is the strongest claim here.

**2. It loses where the task is not forecasting-shaped, or where channels interact.**
- **PEMS** (MAE 34.13 vs 17.41): traffic networks carry their signal in *cross-channel* correlations,
  and SparseTSF is deliberately channel-independent.
- **Classification** (60.9% vs 75.9%): strong where a periodic signature is enough
  (SpokenArabicDigits 94.0, PEMS-SF 84.4, SCP1 82.3) and collapses where it is not
  (Handwriting 8.6% over 26 classes). Note the head was written for this study, so this is a floor.
- **M4** (OWA 0.972): beats the Naive2 baseline, but M4's short, heterogeneous series give its
  cross-period mechanism little to exploit.

**3. Anomaly detection is a genuine strength.** PSM F1 95.3 and SMD F1 80.0 with a reconstruction
head of a few dozen parameters. (Average over 2 of 5 datasets — not comparable to the paper's
5-dataset average.)

**4. Imputation is a clean efficiency trade** — ~27% higher masked MSE than TimeMixer on the shared
5 datasets, at roughly **5.7× less training time**.

<!-- AUTO_SUMMARY_START -->
| Task | Datasets | Metric | SparseTSF (ours) | Reference (paper) |
|---|---|---|---:|---:|
| Univariate short-term | M4 (6 subsets) | OWA (lower=better) | 0.972 | _TimeMixer 0.840_ |
| Multivariate short-term | PEMS03/04/07/08 | MAE (lower) | 34.13 | _TimeMixer 17.41_ |
| Imputation | ETT x4, Weather, ECL | masked MSE (lower) | 0.0838 (6 ds) | _TimeMixer 0.0586 (5 ds)_ |
| Few-shot (10% train) | ETT (Avg) | MSE (lower) | 0.441 | _TimeMixer 0.453_ |
| Zero-shot (transfer) | 6 ETT pairs | MSE (lower) | 0.393 | _TimeMixer 0.467 (avg)_ |
| Anomaly detection | PSM, SMD (of 5) | F1 (higher) | 87.64 | _TimeMixer++ 87.47 (5 ds)_ |
| Classification | 10 UEA | accuracy (higher) | 60.90 | _TimeMixer++ 75.9_ |
<!-- AUTO_SUMMARY_END -->
