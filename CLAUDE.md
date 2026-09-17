# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local **benchmark of three long-term time-series forecasting models** — **TimeMixer** (ICLR'24,
upstream), **TimeMixer++** (ICLR'25, reimplemented from scratch), and **SparseTSF** (ICML'24,
vendored) — compared on **accuracy** and **energy/carbon** on a Windows laptop with a 4 GB GPU.
The consolidated write-up is `TimeMixer/reports/results_final_report.md`; `README.md` (workspace
root) is the navigation map.

All code lives under **`TimeMixer/`**; the workspace root also holds `dataset/` (shared data).
The workspace root **is the git repository** (branch `main`) — there is no nested repo inside
`TimeMixer/` any more, and one root `.gitignore` governs everything. Fork provenance vs upstream
TimeMixer is in `TimeMixer/UPSTREAM_FORK.md`.

## Environment & commands

- Interpreter is the in-repo venv: **`TimeMixer/venv/Scripts/python.exe`** (Python 3.8.10,
  `torch==1.7.1+cu110`, CUDA enabled). Always use it, not system Python.
- **`run.py` must be run with the current directory = `TimeMixer/`** — it writes to relative
  `./checkpoints/<setting>/` and `./test_results/<setting>/`. The benchmark runners enforce this by
  passing `cwd=REPO_ROOT` to the subprocess.

Run the benchmarks (from `TimeMixer/`; all runners are **idempotent** — a config whose log already
holds a valid MSE/MAE is skipped, so re-launching after an interruption is safe):

```bash
venv/Scripts/python.exe benchmarks/run_missing.py              # TimeMixer v1 accuracy
venv/Scripts/python.exe benchmarks/run_missing_pp.py           # TimeMixer++ accuracy
venv/Scripts/python.exe benchmarks/run_sparsetsf_benchmarks.py # SparseTSF accuracy (sl=96 & 720)
venv/Scripts/python.exe benchmarks/run_carbon_benchmarks.py    # 3-model CodeCarbon sweep
```

Per-task SparseTSF benchmarks (the 8 task categories the TimeMixer++ paper evaluates). Each writes
`reports/results_comparison_<task>.md` and `logs/run_logs_<task>/`, and is idempotent:

```bash
venv/Scripts/python.exe benchmarks/tools/fetch_task_datasets.py       # PSM, 10x UEA, SMD
venv/Scripts/python.exe benchmarks/run_m4_benchmarks.py               # univariate short-term (SMAPE/MASE/OWA)
venv/Scripts/python.exe benchmarks/run_pems_benchmarks.py             # multivariate short-term (MAE/MAPE/RMSE)
venv/Scripts/python.exe benchmarks/run_imputation_benchmarks.py       # imputation (masked MSE/MAE)
venv/Scripts/python.exe benchmarks/run_fewshot_benchmarks.py          # few-shot (10% train)
venv/Scripts/python.exe benchmarks/run_zeroshot_benchmarks.py         # zero-shot transfer
venv/Scripts/python.exe benchmarks/run_anomaly_benchmarks.py          # anomaly (point-adjusted F1)
venv/Scripts/python.exe benchmarks/run_classification_benchmarks.py   # classification (accuracy)
```

```bash
# Rebuild the v1 results table from existing logs WITHOUT training (fast smoke test of paths):
venv/Scripts/python.exe benchmarks/run_all_long_term_benchmarks.py --update-from-logs-only
```

Run one experiment directly (this is the analog of "a single test"; args mirror one runner config):

```bash
venv/Scripts/python.exe run.py --task_name long_term_forecast --is_training 1 \
  --model_id ETTh1_96_96 --model TimeMixer --data ETTh1 \
  --root_path ../dataset/ETT-small --data_path ETTh1.csv --features M \
  --seq_len 96 --label_len 0 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 \
  --e_layers 2 --d_model 16 --d_ff 32 --channel_independence 1 \
  --down_sampling_layers 3 --down_sampling_window 2 --down_sampling_method avg \
  --learning_rate 0.01 --train_epochs 10 --batch_size 128 --num_workers 0
```

There is no lint/test suite. To sanity-check runner edits without training, byte-compile them:
`venv/Scripts/python.exe -m py_compile benchmarks/*.py`.

## Architecture — the big picture

This is a Time-Series-Library (TSLib) style scaffold. Understanding it requires seeing how four
layers connect:

**1. Experiment pipeline (upstream scaffold).** `run.py` parses ~60 argparse flags, builds a
`setting` string (used as both the checkpoint directory name and the CodeCarbon join key), and
dispatches to a task class in `exp/` (one per TSLib task). `exp/exp_long_term_forecasting.py` is the
main one — it owns the train/val/test loop, and its `test()` prints the line **`mse:{}, mae:{}`** that
every benchmark runner regex-parses out of the logs. The other four (`exp_short_term_forecasting`,
`exp_imputation`, `exp_anomaly_detection`, `exp_classification`) are now also exercised, since all
three models — TimeMixer, TimeMixer++, and (as of the multi-task extension) SparseTSF — support the
full task set. Models are looked up by name in `exp/exp_basic.py`'s `model_dict`.

**2. The three models, wired via a shim pattern.** `model_dict` maps `TimeMixer`, `TimeMixerPP`,
`SparseTSF` → modules in `models/`, which is a **registry**: one file per registered model name.
The substantial code lives in `implementations/`, one package per model — `timemixer_pp/` (ours)
and `sparsetsf/` (vendored, beside its own Apache-2.0 licence). Two registry entries are
deliberately *thin*:
- `models/TimeMixer.py` — self-contained (uses `layers/`); multi-scale downsample →
  `PastDecomposableMixing` (season mixed bottom-up, trend top-down) → `FutureMultipredictorMixing`.
- `models/TimeMixerPP.py` — 5-line wrapper re-exporting `implementations/timemixer_pp/model.py` (the from-scratch
  ICLR'25 reimplementation: FFT top-K "time imaging" MRTI, dual-axis attention TID, conv multi-scale
  mixing).
- `models/SparseTSF.py` — a **task-aware** model. The `long_term_forecast`/`short_term_forecast`
  path **delegates to the untouched vendored core** `implementations/sparsetsf/model.py` (Apache-2.0), so
  forecast results are reproduced bit-for-bit; the `imputation`/`anomaly_detection` (cross-period
  reconstruction) and `classification` (conv-aggregation → flatten → linear) heads are built here
  from SparseTSF's own primitives. It branches on `configs.task_name` and returns the per-task shape
  each `exp/exp_*.py` expects. To add a model: put the implementation in `implementations/<name>/`
  (with an `__init__.py`), add a thin `models/<Name>.py` entry point that re-exports its `Model`,
  and register that name in `exp/exp_basic.py`. `models/TimeMixer.py` is the exception and stays
  where upstream put it, so upstream changes keep merging cleanly.

**3. Forward-call & data conventions** (needed to modify any model or the loop). Every model is called
as `model(batch_x, batch_x_mark, dec_inp, batch_y_mark)`. `data_provider/` yields
`(batch_x, batch_y, batch_x_mark, batch_y_mark)`. Two branch conditions recur throughout the loop:
`down_sampling_layers == 0` builds a real decoder input, otherwise `dec_inp = None` (the multi-scale
path); and for `data in {PEMS, Solar}` the time-feature marks are forced to `None`.

**4. Benchmark harness (added on top of upstream), organized into sibling folders:**
- `benchmarks/` — one runner per model/experiment. Each sets `REPO_ROOT =
  Path(__file__).resolve().parent.parent` (the scripts live one level down in `benchmarks/`), then
  shells out to `run.py` once per (dataset × pred_len) config via `sys.executable`, captures stdout
  to a log, parses `mse:`/`mae:` back out, and rewrites a report between `<!-- AUTO_*_START -->` /
  `<!-- AUTO_*_END -->` markers. `run_carbon_benchmarks.py` additionally reconstructs the `setting`
  string to join each run to its CodeCarbon row.
- `reports/` — the human-readable `results_*.md` tables the runners regenerate. This is the
  deliverable; it is **separate from** upstream's `results/`, which the exp loop fills with
  auto-generated `pred.npy`/`true.npy`/`metrics.npy` per run (≈7 GB, gitignored). Do not put
  hand-written reports back into `results/`.
- `logs/` — `run_logs*/` (raw per-run logs, one per config; the **skip/idempotency** decision reads
  these), `emissions_timemixer.csv` (CodeCarbon output; opt-in via `run.py --track_emissions`), and
  `console/` (captured runner stdout/stderr).

If you relocate any of these, the `REPO_ROOT`-relative path constants at the top of each runner
(`MD_PATH`, `LOGS_DIR`, `EMISSIONS_PATH`, `V1_MD`/`PP_MD`) must move in lockstep.

## Non-obvious constraints — do not regress

- **TimeMixer++ NaN-stability is a coordinated system, not one fix.** In
  `exp/exp_long_term_forecasting.py`: skip a batch whose loss is non-finite, and — crucially — skip
  the optimizer step when `clip_grad_norm_` returns a non-finite total norm (grad NaNs can be hidden
  from the loss by output sanitization). In `utils/tools.py`: `EarlyStopping` treats a non-finite
  val-loss as non-improving so it never overwrites a good checkpoint. In `implementations/timemixer_pp/model.py`:
  FFT-amplitude `isfinite`/clamp before softmax, attention-score clamps, output `_sanitize`. PP is
  run at a conservative `lr=5e-5` + `lradj type1`, and Solar needs `use_norm=1`.
- **PP is undertrained by design.** That `lr=5e-5` is a stability requirement, so PP's *accuracy*
  numbers understate the architecture — do not read the PP column as "PP is worse". Its *energy*
  numbers are fair.
- **Two protocols that must never be cross-compared.** Accuracy tables use each model's own published
  hyperparameters *with* early stopping; the carbon sweep uses a *matched* config, fixed 5 epochs,
  early stopping off (`patience=999`). The MSE column in the carbon table is therefore **not** real
  accuracy — it only exists to confirm runs converged.
- **SparseTSF `--period_len` divisibility is task-aware** (asserted up front in `models/SparseTSF.py`):
  forecasting needs `period_len` to divide BOTH `seq_len` and `pred_len`; imputation/anomaly need it
  to divide `seq_len`; classification imposes no requirement (its head avoids period reshaping, since
  UEA `max_seq_len` is arbitrary). Upstream per-dataset values: hourly ETTh = 24, ETTm/Weather = 4.
  SparseTSF's classification head scales as `enc_in·seq_len × num_class`, so only that task breaks its
  "~1k params" claim (inherent to classification).
- **Task datasets: what's local vs. gated.** `benchmarks/tools/fetch_task_datasets.py` fetches
  **PSM** + **SMD** (anomaly) and the **10 standard UEA** classification sets; ETT/Weather/ECL/M4/PEMS
  ship with the repo. Still **not** obtainable: **MSL/SMAP** (need the TimesNet/TSLib preprocessed
  `.npy` bundle — the telemanom S3 source now 403s) and **SWaT** (gated, iTrust request form). Exact
  target filenames are in that script's header; the anomaly runner marks missing sets `NOT AVAILABLE`.
  Note: `timeseriesclassification.com` rejects urllib with HTTP 403 but serves **curl**, so the
  fetcher falls back to curl.
- **`--percent` (few-shot) is opt-in and only forwarded when < 100.** `data_provider/data_factory.py`
  passes it *conditionally*, so loaders that don't accept it (M4/PEMS/Solar) are untouched on the
  normal path. It truncates the TRAIN split only; the scaler still fits the full train range (TSLib
  convention) and val/test are unchanged.
- **`exp_long_term_forecasting.test()` prints `RMSE:`/`MAPE:` in UPPERCASE on purpose.** Every runner
  regexes lowercase `mse:`/`mae:` and takes the LAST match — a lowercase `rmse:` contains `mse:` as a
  substring and would silently corrupt every existing results table. Keep new metric labels uppercase.
- **Classification uses `optim.RAdam`, absent in torch 1.7.1** — `exp/exp_classification.py` falls back
  to Adam when RAdam is missing (model-agnostic; keeps RAdam on newer torch).
- **4 GB GPU / Windows limits.** Batch sizes are reduced from the papers' values. Solar-Energy,
  Electricity, and Traffic are excluded from the carbon sweep as too heavy. Energy: GPU power is real
  (NVML) but CPU/RAM are TDP estimates (no RAPL on Windows) — treat relative ratios as sound and
  absolute kg CO₂ as an estimate.
- **`--track_emissions` is opt-in** (default off) so ordinary accuracy runs carry no CodeCarbon
  overhead or dependency. `run.py` imports `codecarbon` lazily only when the flag is set.
