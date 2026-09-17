# SparseTSF Reproduction Results

- **Paper**: SparseTSF: Modeling Long-term Time Series Forecasting with 1k Parameters (ICML 2024, Oral)
- **Upstream**: https://github.com/lss-1138/SparseTSF (Apache-2.0)
- **Vendored model**: `SparseTSF_model/model.py`; adapter for our exp loop: `models/SparseTSF.py`
- **Hardware**: RTX 3050 Ti Laptop GPU (4 GB), Windows
- **Runner**: `run_sparsetsf_benchmarks.py` -> logs in `.run_logs_sparsetsf/`

SparseTSF is the architectural opposite of TimeMixer++: **Cross-Period Sparse Forecasting** with
**under 1,000 trainable parameters**. It subtracts the sequence mean, aggregates with a 1-D conv,
downsamples the series into `period_len` segments, then applies a *single shared linear layer*
across periods.

Measured parameter counts from our integration:

| Config | seg_num_x -> seg_num_y | Trainable params |
|---|---|---:|
| ETTh, seq_len=720, pred_len=720, period_len=24 | 30 -> 30 | **925** |
| ETTh, seq_len=96, pred_len=96, period_len=24 | 4 -> 4 | 41 |
| Weather, seq_len=720, pred_len=720, period_len=4 | 180 -> 180 | 32,405 |

The famous "1k parameters" headline corresponds to **ETTh at seq_len=720, period_len=24** (925).

---

## Setup

SparseTSF's **own published hyperparameters** are used (from `scripts/SparseTSF/linear/*.sh`
upstream), matching how `results_comparison.md` used TimeMixer's own config:

| Setting | Value |
|---|---|
| `learning_rate` | 0.02 |
| `batch_size` | 256 |
| `train_epochs` | 30 |
| `patience` | 5 |
| `lradj` | type3 |
| `model_type` | **linear** (the <1k-param variant) |

**Per-dataset `period_len`** (upstream's choices; it must divide both `seq_len` and `pred_len`):

| Dataset | Sampling | period_len | enc_in |
|---|---|---:|---:|
| ETTh1 / ETTh2 | hourly | 24 | 7 |
| ETTm1 / ETTm2 | 15-min | 4 | 7 |
| Weather | 10-min | 4 | 21 |

**Two lookbacks are swept** so the comparison is honest in both directions:
- `seq_len=96` — matches our TimeMixer / TimeMixer++ protocol, so the numbers are directly comparable
- `seq_len=720` — SparseTSF's published setting; it needs several periods of context to work well

---

## Results

<!-- AUTO_SPARSETSF_RESULTS_START -->
#### seq_len = 96 (comparable to our TimeMixer / TimeMixer++ tables)

| Dataset | period_len | pred_len | MSE | MAE | Runtime (s) | Status |
|---|---:|---:|---:|---:|---:|---|
| ETTh1 | 24 | 96 | 0.396635 | 0.403991 | 9.2 | OK |
| ETTh1 | 24 | 192 | 0.433213 | 0.417648 | 10.2 | OK |
| ETTh1 | 24 | 336 | 0.449656 | 0.433982 | 9.9 | OK |
| ETTh1 | 24 | 720 | 0.459474 | 0.456393 | 7.7 | OK |
| ETTh2 | 24 | 96 | 0.288646 | 0.339110 | 8.8 | OK |
| ETTh2 | 24 | 192 | 0.363235 | 0.385951 | 8.6 | OK |
| ETTh2 | 24 | 336 | 0.365519 | 0.398741 | 8.8 | OK |
| ETTh2 | 24 | 720 | 0.409178 | 0.428967 | 10.6 | OK |
| ETTm1 | 4 | 96 | 0.362121 | 0.378584 | 22.9 | OK |
| ETTm1 | 4 | 192 | 0.394356 | 0.392920 | 17.3 | OK |
| ETTm1 | 4 | 336 | 0.426852 | 0.416844 | 20.5 | OK |
| ETTm1 | 4 | 720 | 0.485662 | 0.446954 | 36.7 | OK |
| ETTm2 | 4 | 96 | 0.185053 | 0.266444 | 22.9 | OK |
| ETTm2 | 4 | 192 | 0.248705 | 0.306049 | 19.1 | OK |
| ETTm2 | 4 | 336 | 0.311663 | 0.345110 | 22.9 | OK |
| ETTm2 | 4 | 720 | 0.408093 | 0.398541 | 24.8 | OK |
| Weather | 4 | 96 | 0.198635 | 0.238264 | 41.4 | OK |
| Weather | 4 | 192 | 0.242842 | 0.273975 | 34.6 | OK |
| Weather | 4 | 336 | 0.292309 | 0.308392 | 50.2 | OK |
| Weather | 4 | 720 | 0.362776 | 0.354812 | 142.4 | OK |

#### seq_len = 720 (SparseTSF's published setting)

| Dataset | period_len | pred_len | MSE | MAE | Runtime (s) | Status |
|---|---:|---:|---:|---:|---:|---|
| ETTh1 | 24 | 96 | 0.353642 | 0.383826 | 23.0 | OK |
| ETTh1 | 24 | 192 | 0.395262 | 0.405433 | 14.9 | OK |
| ETTh1 | 24 | 336 | 0.401939 | 0.417958 | 15.7 | OK |
| ETTh1 | 24 | 720 | 0.419721 | 0.440783 | 16.5 | OK |
| ETTh2 | 24 | 96 | 0.263330 | 0.327415 | 21.0 | OK |
| ETTh2 | 24 | 192 | 0.319403 | 0.364877 | 15.8 | OK |
| ETTh2 | 24 | 336 | 0.310847 | 0.371503 | 21.8 | OK |
| ETTh2 | 24 | 720 | 0.368332 | 0.414595 | 18.8 | OK |
| ETTm1 | 4 | 96 | 0.313554 | 0.355345 | 84.4 | OK |
| ETTm1 | 4 | 192 | 0.342686 | 0.371804 | 78.1 | OK |
| ETTm1 | 4 | 336 | 0.371734 | 0.388843 | 53.8 | OK |
| ETTm1 | 4 | 720 | 0.416650 | 0.413100 | 90.8 | OK |
| ETTm2 | 4 | 96 | 0.163879 | 0.253566 | 77.1 | OK |
| ETTm2 | 4 | 192 | 0.219637 | 0.291708 | 89.0 | OK |
| ETTm2 | 4 | 336 | 0.272018 | 0.327521 | 54.0 | OK |
| ETTm2 | 4 | 720 | 0.350821 | 0.379013 | 78.8 | OK |
| Weather | 4 | 96 | 0.172756 | 0.227416 | 114.2 | OK |
| Weather | 4 | 192 | 0.214261 | 0.260899 | 128.4 | OK |
| Weather | 4 | 336 | 0.259896 | 0.297003 | 105.2 | OK |
| Weather | 4 | 720 | 0.317789 | 0.337996 | 323.2 | OK |

<!-- AUTO_SPARSETSF_RESULTS_END -->

---

## Three-Way Comparison (seq_len = 96)

TimeMixer and TimeMixer++ values are read from `results_comparison.md` and
`results_comparison_pp.md`. All three use `seq_len=96`, so the *task* is directly comparable —
each model uses its own published hyperparameters (best-effort per model, not a matched-config
ablation).

> **Read the TimeMixer++ column with care.** Those numbers come from the full consistent re-run at
> `lr=5e-5`, a deliberately conservative rate needed to stop TimeMixer++ diverging to NaN (see the
> stability changelog in `results_comparison_pp.md`). At that LR it is **undertrained within the
> 20-epoch budget**, so the column understates the architecture's real capability — it is not a
> like-for-like "TimeMixer++ is worse" result. The TimeMixer column, by contrast, is a converged
> run at its own `lr=1e-2`.

<!-- AUTO_SPARSETSF_3WAY_START -->
| Dataset | pred_len | TimeMixer MSE | TimeMixer++ MSE | SparseTSF MSE | Best |
|---|---:|---:|---:|---:|---|
| ETTh1 | 96 | 0.385794 | 0.692902 | 0.396635 | TimeMixer |
| ETTh1 | 192 | 0.443037 | 0.735732 | 0.433213 | SparseTSF |
| ETTh1 | 336 | 0.513090 | 0.795015 | 0.449656 | SparseTSF |
| ETTh1 | 720 | 0.489165 | 0.763298 | 0.459474 | SparseTSF |
| ETTh2 | 96 | 0.301334 | 0.382873 | 0.288646 | SparseTSF |
| ETTh2 | 192 | 0.391096 | 0.458633 | 0.363235 | SparseTSF |
| ETTh2 | 336 | 0.386519 | 0.487012 | 0.365519 | SparseTSF |
| ETTh2 | 720 | 0.412863 | 0.474172 | 0.409178 | SparseTSF |
| ETTm1 | 96 | 0.326398 | 0.419942 | 0.362121 | TimeMixer |
| ETTm1 | 192 | 0.365291 | 0.426157 | 0.394356 | TimeMixer |
| ETTm1 | 336 | 0.393321 | 0.471204 | 0.426852 | TimeMixer |
| ETTm1 | 720 | 0.451405 | 0.514390 | 0.485662 | TimeMixer |
| ETTm2 | 96 | 0.175972 | 0.198934 | 0.185053 | TimeMixer |
| ETTm2 | 192 | 0.238108 | 0.262988 | 0.248705 | TimeMixer |
| ETTm2 | 336 | 0.298672 | 0.344020 | 0.311663 | TimeMixer |
| ETTm2 | 720 | 0.406894 | 0.444232 | 0.408093 | TimeMixer |
| Weather | 96 | 0.164526 | 0.199892 | 0.198635 | TimeMixer |
| Weather | 192 | 0.209339 | 0.244207 | 0.242842 | TimeMixer |
| Weather | 336 | 0.265349 | 0.294700 | 0.292309 | TimeMixer |
| Weather | 720 | 0.342314 | 0.363639 | 0.362776 | TimeMixer |
<!-- AUTO_SPARSETSF_3WAY_END -->

---

## Caveats

- **No "Paper MSE" column.** SparseTSF's published per-prediction-length numbers live in *images*
  in the upstream README (`Table2.png` etc.), not machine-readable text, so they are deliberately
  not reproduced here rather than risk transcription errors. See the upstream
  [Full Results](https://github.com/lss-1138/SparseTSF#full-results) section to compare by hand.
- **`seq_len=96` is outside SparseTSF's published setting** (720) and will underperform its paper
  numbers — at `period_len=24` a 96-step window is only 4 periods of context. Both lookbacks are
  reported so this is visible rather than hidden.
- **`model_type=linear`** is used; upstream's argparse defaults to `mlp`, but `linear` is the
  <1k-parameter configuration the paper headlines and the one relevant to the energy study.
- The shared stability guards added for TimeMixer++ (gradient clipping, non-finite-gradient step
  skipping, EarlyStopping NaN guard) apply to SparseTSF too, since they live in
  `exp/exp_long_term_forecasting.py` and `utils/tools.py`.
- For **energy/carbon** numbers, see `results_carbon_comparison.md` — that uses a separate
  matched-config protocol and is not comparable to the accuracy numbers here.
