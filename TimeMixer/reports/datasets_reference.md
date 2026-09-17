# Datasets used in this study

Every dataset touched by the benchmarks in this repository, grouped by the task it serves. All
counts, shapes and rates below were **measured from the local files** in `../dataset/`, not copied
from the papers — where a published figure differs, that is noted.

For deeper split arithmetic and per-model hyperparameters of the 8 long-term forecasting sets, see
the companion [dataset_statistics.md](dataset_statistics.md). This file is the wider map: it also
covers M4, PEMS, the anomaly sets and the UEA classification archives.

**Fetching.** ETT / Weather / Electricity / Traffic / Solar / M4 / PEMS ship with the repo.
PSM, SMD and the 10 UEA archives are downloaded by
[`benchmarks/tools/fetch_task_datasets.py`](../benchmarks/tools/fetch_task_datasets.py).
MSL, SMAP and SWaT could **not** be obtained — see [Unavailable](#unavailable-datasets).

---

## 1. Long-term forecasting (8 datasets)

Input 96 steps → predict {96, 192, 336, 720}. Split: ETT uses a fixed calendar split
(12/4/4 months); the rest use a proportional 70/10/20 split.

| Dataset | Domain | Rows | Channels | Resolution | Loader |
|---|---|---:|---:|---|---|
| **ETTh1 / ETTh2** | Electricity transformer | 17,420 | 7 | 1 hour | `Dataset_ETT_hour` |
| **ETTm1 / ETTm2** | Electricity transformer | 69,681 | 7 | 15 min | `Dataset_ETT_minute` |
| **Weather** | Meteorology | 52,697 | 21 | 10 min | `Dataset_Custom` |
| **Electricity (ECL)** | Household power | 26,305 | 321 | 1 hour | `Dataset_Custom` |
| **Traffic** | Road occupancy | 17,545 | 862 | 1 hour | `Dataset_Custom` |
| **Solar-Energy** | PV generation | 52,560 | 137 | 10 min | `Dataset_Solar` |

- **ETT** (Electricity Transformer Temperature): 7 load signals plus oil temperature (`OT`, the
  target). h1/h2 and m1/m2 are two different transformers at hourly and 15-minute resolution.
  Their **daily cycle (24 h)** is what SparseTSF's `period_len=24` (hourly) / `4` (15-min) locks onto.
- **Weather**: 21 meteorological variables from one German station, 2020.
- **Electricity**: 321 clients' hourly consumption. Wide but strongly periodic.
- **Traffic**: 862 freeway sensors (Caltrans PeMS). The widest forecasting set here; its true cycle
  is **weekly (168 h)**, which a 96-step lookback cannot see — relevant when reading SparseTSF's
  results, since it needs several periods of context.
- **Solar-Energy**: 137 PV plants, headerless `.txt`. Note the loader forces time-feature marks to
  `None` for `Solar` and `PEMS`.

> Scope note: the SparseTSF long-term sweep covers the **5 light datasets** (ETT×4 + Weather) at two
> lookbacks. Electricity / Traffic / Solar were excluded there as too heavy for the 4 GB GPU.

---

## 2. Univariate short-term forecasting — M4 (6 subsets)

100,000 individual series from the M4 competition; each series is forecast from its own history
(`seq_len = 2 × horizon`). Metrics: SMAPE / MASE / **OWA**.

| Subset | Series | Horizon | Seasonality | `period_len` used |
|---|---:|---:|---:|---:|
| Yearly | 23,000 | 6 | 1 | 2 |
| Quarterly | 24,000 | 8 | 4 | 4 |
| Monthly | 48,000 | 18 | 12 | 6 |
| Weekly | 359 | 13 | 1 | 13 |
| Daily | 4,227 | 14 | 1 | 7 |
| Hourly | 414 | 48 | 24 | 24 |

- `period_len` must divide **both** the horizon and `seq_len = 2 × horizon`. Weekly's horizon of 13
  is prime, forcing `period_len = 13`.
- OWA is normalised against a **Naive2** baseline (`dataset/m4/submission-Naive2.csv`, shipped);
  OWA < 1.0 means the model beats that baseline.
- `M4Summary` reports Yearly / Quarterly / Monthly separately and collapses Weekly+Daily+Hourly into
  an **"Others"** group, so per-subset SMAPE is not separable for those three.

---

## 3. Multivariate short-term forecasting — PEMS (4 datasets)

Traffic-sensor networks, 5-minute resolution, input 96 → predict 12. Metrics: MAE / MAPE / RMSE on
the original (inverse-transformed) scale. Split 60/20/20.

| Dataset | Timesteps | Sensors | Raw `.npz` channels |
|---|---:|---:|---|
| PEMS03 | 26,208 | 358 | 1 (flow) |
| PEMS04 | 16,992 | 307 | 3 (flow, occupancy, speed) |
| PEMS07 | 28,224 | 883 | 1 (flow) |
| PEMS08 | 17,856 | 170 | 3 (flow, occupancy, speed) |

- **Only the first channel (traffic flow) is used** — `Dataset_PEMS` slices `data['data'][:, :, 0]`,
  so PEMS04/08's occupancy and speed are discarded. This matches upstream TSLib.
- Run under `task_name=long_term_forecast` with `use_norm=0` (the upstream PEMS scripts do the same;
  the short-term exp class is hard-wired to M4).
- `period_len=12` (one hour) — the only sensible value dividing both 96 and 12.
- These are the datasets where SparseTSF is weakest: the signal lives in **cross-sensor**
  correlations, and SparseTSF is deliberately channel-independent.

---

## 4. Imputation

Reuses the forecasting datasets — **ETT×4, Weather, Electricity** (6 total, matching the paper's
selection). Random masking of {12.5, 25, 37.5, 50}% of entries; only masked positions are scored.

> **Protocol deviation:** the TimeMixer++ paper imputes at `seq_len=1024`; this study uses **96**.
> SparseTSF requires `period_len` to divide `seq_len`, and 1024 admits only powers of two — which
> cannot align to the 24-hour cycle of hourly data. The paper's length would force the model off its
> own inductive bias.

---

## 5. Few-shot & zero-shot forecasting

No new data — both reuse the forecasting sets:

- **Few-shot**: ETT×4 + Weather + Electricity, training on the **first 10%** of the train split
  (`--percent 10`). Validation and test splits are untouched, so results remain comparable to the
  full-data tables. The scaler still fits the full training range (TSLib convention).
- **Zero-shot**: 6 ETT transfer pairs (ETTh1→ETTh2, ETTh1→ETTm2, ETTh2→ETTh1, ETTm1→ETTh2,
  ETTm1→ETTm2, ETTm2→ETTm1). Train on A, evaluate on B with no further training. All ETT sets share
  7 channels, which is what makes the transfer mechanically possible.

---

## 6. Anomaly detection

Sliding window 100; metric = point-adjusted F1. Two of the paper's five datasets are usable here.

| Dataset | Train rows | Test rows | Channels | Anomaly rate | Status |
|---|---:|---:|---:|---:|---|
| **PSM** | 132,481 | 87,841 | 25 | **27.8%** | ✅ fetched |
| **SMD** | 708,405 | 708,420 | 38 | **4.2%** | ✅ assembled |
| MSL | — | — | 55 | — | ❌ unavailable |
| SMAP | — | — | 25 | — | ❌ unavailable |
| SWaT | — | — | 51 | — | ❌ gated |

- **PSM** (Pooled Server Metrics, eBay): server telemetry, from the RANSynCoders repository. Note the
  very high 27.8% anomaly rate — unusually dense, which inflates F1 relative to sparser sets.
- **SMD** (Server Machine Dataset, OmniAnomaly): **assembled locally** by concatenating the 28
  per-machine files in sorted order. The resulting 708,405 training rows split exactly into the
  published 566,724 train + 141,681 validation, which confirms the concatenation matches the
  standard preprocessing.
- The loader derives validation as the last 20% of the training array for SMD, and reuses the test
  split as validation for PSM.

---

## 7. Classification — UEA archive (10 datasets)

The 10 standard multivariate sets used by TSLib / TimesNet / TimeMixer++. Metric: accuracy.
`seq_len`, `enc_in` and `num_class` are set automatically from each dataset by `Exp_Classification`.

| Dataset | Train | Test | Dims | Length | Classes | Domain |
|---|---:|---:|---:|---:|---:|---|
| EthanolConcentration | 261 | 263 | 3 | 1,751 | 4 | Spectroscopy |
| FaceDetection | 5,890 | 3,524 | 144 | 62 | 2 | MEG brain activity |
| Handwriting | 150 | 850 | 3 | 152 | 26 | Accelerometer |
| Heartbeat | 204 | 205 | 61 | 405 | 2 | Heart-sound audio |
| JapaneseVowels | 270 | 370 | 12 | variable (max 26) | 9 | Speech |
| PEMS-SF | 267 | 173 | 963 | 144 | 7 | Traffic occupancy |
| SelfRegulationSCP1 | 268 | 293 | 6 | 896 | 2 | EEG |
| SelfRegulationSCP2 | 200 | 180 | 7 | 1,152 | 2 | EEG |
| SpokenArabicDigits | 6,599 | 2,199 | 13 | variable (max 93) | 10 | Speech (MFCC) |
| UWaveGestureLibrary | 120 | 320 | 3 | 315 | 8 | Gesture accelerometer |

- Variable-length sets are padded/truncated to the dataset maximum by `data_provider/uea.py`'s
  `collate_fn`, which also supplies the `padding_mask` that SparseTSF's head uses to zero padded steps.
- Files are `.ts` format, parsed via `sktime`'s `load_from_tsfile_to_dataframe`.
- **Only the standard `*_TRAIN.ts` / `*_TEST.ts` pair may be present per folder.** `UEAloader` globs
  by a TRAIN/TEST name filter and loads the *first* match, so the `_eq_` (equal-length) variants
  shipped inside the archives are deleted at fetch time.
- Notably these are *tiny* by deep-learning standards — several have only 120–270 training samples,
  which is why accuracy varies so widely (8.6% on Handwriting's 26 classes vs 94% on
  SpokenArabicDigits).

---

## Unavailable datasets

| Dataset | Task | Why not available |
|---|---|---|
| **MSL**, **SMAP** | Anomaly | Need the TimesNet/TSLib *preprocessed* `.npy` bundle. The original telemanom S3 object (`s3-us-west-2.amazonaws.com/telemanom/data.zip`) now returns **HTTP 403**, and the bundle is distributed via Google Drive / Tsinghua Cloud, which is not reliably scriptable. |
| **SWaT** | Anomaly | **Gated** — requires an access request through the iTrust (SUTD) form. |

Both are drop-in once obtained; the exact filenames each loader expects are documented in the header
of `benchmarks/tools/fetch_task_datasets.py`, and the anomaly runner reports missing sets as
`NOT AVAILABLE` rather than silently skipping them.

**Consequence for interpretation:** the anomaly average in this study is over **2 of 5** datasets and
is therefore *not* comparable to the paper's 5-dataset average.

---

## Present but unused

These ship in `dataset/` (from the upstream repo) and are **not** used by any benchmark here:

- `exchange_rate/exchange_rate.csv` — daily exchange rates of 8 countries. Used by the TimeMixer++
  paper's long-term table but outside this study's dataset selection.
- `illness/national_illness.csv` — weekly US influenza-like illness ratios. A short series (~966
  rows) usually run with `seq_len=36`, not the 96 used throughout here.

---

## Provenance and licensing

These are all standard public research benchmarks, redistributed by the upstream Time-Series-Library
ecosystem. The two fetched at build time come from their original public repositories:

- **PSM** — `github.com/eBay/RANSynCoders` (the `data/` directory).
- **SMD** — `github.com/NetManAIOps/OmniAnomaly` (`ServerMachineDataset/`).
- **UEA** — `timeseriesclassification.com` (aeon-toolkit mirror). Note this host rejects `urllib`
  with HTTP 403 regardless of User-Agent but serves `curl` normally, so the fetcher falls back to curl.

Check each source's own terms before redistributing; nothing here is re-licensed by this repository.
