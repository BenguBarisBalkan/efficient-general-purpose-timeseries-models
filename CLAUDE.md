# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local **benchmark of four time-series models** — **TimeMixer** (ICLR'24, upstream),
**TimeMixer++** (ICLR'25, reimplemented from scratch), **SparseTSF** (ICML'24, vendored), and
**SparseTSFPlus** (this study: SparseTSF with cheap units grafted on from the other two) —
compared on **accuracy** and **energy/carbon** on a Windows laptop with a 4 GB GPU.
The consolidated write-up is `TimeMixer/reports/results_final_report.md`; `README.md` (workspace
root) is the navigation map.

All code lives under **`TimeMixer/`**; the workspace root also holds `dataset/` (shared data).
The workspace root **is the git repository** (branch `main`) — there is no nested repo inside
`TimeMixer/` any more, and one root `.gitignore` governs everything. Fork provenance vs upstream
TimeMixer is in `TimeMixer/UPSTREAM_FORK.md`.

## Thesis context — applies to every step

This repository is the experimental work of a thesis on **general-purpose time-series models**:
one architecture that handles many task families, as TimeMixer++ does (its paper calls it a
"general time series pattern machine"). Such models are trained per task and dataset. They are
**not** pretrained foundation models (TimesFM, Chronos, Moirai, MOMENT), although the zero-shot
task probes the same property. Write "general-purpose model", not "foundation model".

The thesis question is **how efficient these models are, and whether one can be built that is
both efficient and capable**. SparseTSFPlus is that attempt: start from the most efficient model
(SparseTSF, ~1k parameters) and add only what measurably earns its parameters.

**The goal is one model, not a unit per task.** The target is one fixed backbone configuration
plus thin task interfaces, run unchanged across all eight task families. Choosing units per
dataset or per task is an ablation that shows what each unit does, never the headline result. A
unit that helps one task but hurts another needs a safe default (for example a gate that can
learn to switch it off) before it joins the fixed configuration.

Check every step against these three requirements before running it and again before reporting:

1. **Explainable.** Before running, write down the idea and where it comes from (the paper
   section, or the model and file). Also record the mechanism, the prediction, and what result
   would refute it. Report ideas and evidence rather than code. The audience is the thesis
   supervisor, so a unit's reasoning matters more than its implementation.
2. **Comparable.**
   - Follow the TimeMixer++ paper's task protocol (datasets, horizons, metrics) unless that is
     impossible here; state every deviation and why.
   - Copy hyperparameters from the runner that produced the baseline cell, and check the control
     row reproduces the published number before reading any other row.
   - Report accuracy together with cost: trainable parameters, runtime, and energy where the
     matched carbon protocol was used. Never mix accuracy-protocol and carbon-protocol numbers.
   - Label every number as local (this machine, this protocol) or published (paper value, other
     hardware and tuning). Never put them in one comparison without saying which is which.
   - Select on validation, never on test. Imputation, anomaly and classification early-stop on
     test, so call their results a sensitivity analysis.
   - Runs default to one seed (`run.py` fixes 2021; `--seed` repeats a run with another). Do not
     call a gain a finding until it holds across seeds: `benchmarks/run_seed_repeats.py` reports
     seeds 2021–2023 and calls a difference a finding only if it has the same sign on every seed
     and exceeds twice the spread. ETTh2 imputation is the cautionary case: its runs sometimes stay
     on a training plateau at random, so one-seed ETTh2 differences can be pure noise.
3. **Recorded.** Runners write the results into `reports/`; never type numbers in by hand. After
   each completed step, add an entry to the progress log (next section).

### Progress reporting — every two weeks

The supervisor reviews progress every two weeks, and `TimeMixer/reports/progress_log.md` is the
record those meetings are built from. After every completed step (an experiment batch, a new
unit, a report revision), add a dated entry, newest first:

- **Planned:** what and why
- **Done:** runs, datasets, settings
- **Found:** numbers, and the baseline each is compared with
- **Reports changed:** which files
- **Next:** the next step and any open questions

Write entries for the supervisor: ideas and evidence, not implementation. Record negative results,
since a rejected unit is a finding. Leave out process mistakes and their fixes unless they changed
a result. When asked for the biweekly summary, build it from the entries since the last meeting.

### Ideas already tested — read before proposing a new one

The full record is **`TimeMixer/reports/thesis_progress_report.md`** (Section 3: why each idea
was tried, its source, mechanism, result and lesson; tables generated by
`benchmarks/tools/build_progress_document.py --pdf`, which also builds the PDF shown to the
supervisor). Do not re-run an idea below without a new reason, and state that reason in the plan.

- **Adopted in the fixed configuration:** phase mixing (TimeMixer++ dual-axis; PEMS −25%, overfits
  small sets); iterative refinement (largest gain, 0 params) with its ReZero gate; mask-aware
  aggregation (partial conv); learned-trend decomposition (flat trend is harmful).
- **Rejected:** multi-period ensemble (inconsistent); leave-one-out reconstruction (SMD +0.5 F1, PSM
  −2.0); pooled classification readouts (average 60.9 → 36.7; only Heartbeat gains); RevIN (provably
  a no-op on SparseTSF); low-rank phase mixing (never beats full); gate on phase mixing (no effect);
  phase mixing on complete inputs only (+23–42% on Electricity); FFT period rule (picks 48,
  already measured worse).
- **Candidates, unconfirmed or not yet run in full:** mask-aware phase mixing (within seed noise);
  channel correlation for PEMS (−5.25% ± 0.29 over seeds; never run on the full suite).
- **Known levers not in the fixed configuration:** period selection per dataset (ETTh1 period 12
  −3.2%, PEMS08 period 1 −24%), which is tuning, not one model.
- **Established over three seeds:** fixed beats SparseTSF on PEMS, imputation, M4, few-shot and
  long-term; fixed vs per task is under 0.7% on forecasting. Imputation-variant differences and
  single-seed anomaly comparisons are within noise; the TimeMixer seed-2021 MSL run failed.

### Completed plan — 24-hour unattended backbone work (2026-10-21/22)

Full plan, decision rules and a status checklist: Claude Doc "SparseTSFPlus — 24-hour backbone
plan (2026-10-21)", https://claude.ai/artifact/Kg1R4XDoXQSMBaWBkBD5Fb (doc id
`9736c231-3e6d-46ee-85a2-8c18c821723e`). Tick its Status list and this one as steps finish.

Agreed with the user: scope below is approved; a unit that passes the rules is reported as a
**candidate** configuration, never swapped into the headline fixed one; PRs and thesis-document
updates are allowed without asking; if disk runs short, **move** (never delete) `results/` folders
from these runs to a folder on D:/; keep-awake is on.

- [x] **Step 0** (done, PR #16) — `--seed` flag in `run.py` (default 2021, existing runs unchanged); implement the
  step 1–2 units behind default-off flags; equivalence gate, gate-at-zero check, 1-epoch smoke runs.
- [x] **Step 1** (done, PR #17: `allrg_pm` is the candidate, 0.0657; `allrg_pc` rejected) — phase mixing on inputs with missing values: `complete` (skip phase mixing in the
  first imputation pass, apply it in the refinement pass) and `mask-aware` (mix observed positions
  only, renormalised). 48 imputation runs on top of `allrg`. Predict the cost at masks 0.375/0.5 on
  ETTh2 and Electricity < 5% and fixed imputation ≤ 0.0655; refuted if that cost stays > 15%.
- [x] **Step 2** (done, screen only: PEMS −3.9% mean / −4.9% corr, rule not met, no full run; both go to PEMS seed repeats) — channel context for PEMS, gated, no parameter depending on channel count:
  `mean` (SOFTS-style shared channel summary) and `corr` (parameter-free similarity weighting,
  iTransformer-style). Screen PEMS03/04/07/08 + ETTh1/ETTm1/Weather at H=96. Predict PEMS MAE −10%
  or better with ETT/Weather ≤ +1%; refuted if PEMS improves < 3%. Full suite + same-session energy
  only if PEMS validation improves ≥ 5% and no check is > 1% worse; otherwise step 4.
- [x] **Step 3** (done, PRs #19–#20; `results_seed_repeats.md`) — `benchmarks/run_seed_repeats.py`; seeds 2022 and 2023 for SparseTSF, fixed SparseTSFPlus (and the candidates, `--configs`):
  long-term on ETT×4 + Weather, M4, PEMS, few-shot, zero-shot, anomaly, imputation on ETTh1/ETTh2
  (+ `u5g` on ETTh2). New `results_seed_repeats.md`; a finding needs the same sign on all three
  seeds and a difference above twice the seed spread.
- [x] **Step 4 (stretch)** (closed without training: FFT picks 24 on hourly ETT, 48 elsewhere, already measured worse; `results_fft_periods.md`) — automatic period choice: strongest FFT period of the training split,
  restricted to valid divisors; long-term ETT/Weather and PEMS against published periods.
- [x] **Reporting** (done; handoff entry at the top of the progress log) — after every step: progress-log entry (plan with prediction BEFORE the runs),
  runner tables, SparseTSFPlus report, thesis-doc section (layout of Section 17), one merged PR.
  Hard stop for new runs at hour 21; finish with a handoff entry written as the biweekly summary.

## Environment & commands

- Interpreter is the in-repo venv: **`TimeMixer/venv/Scripts/python.exe`** (Python 3.8.10,
  `torch==1.7.1+cu110`, CUDA enabled). Always use it, not system Python.
- **`run.py` must be run with the current directory = `TimeMixer/`** — it writes to relative
  `./checkpoints/<setting>/`, `./test_results/<setting>/`, `./results/<setting>/` (imputation and
  classification only) and `./logs/result_*.txt`. The benchmark runners enforce this by passing
  `cwd=REPO_ROOT` to the subprocess.
- A **fresh clone cannot run anything**: `venv/` and `dataset/` are both gitignored. Rebuild the venv
  from `TimeMixer/requirements.txt` (Python 3.8) and fetch the data — ETT/Weather/ECL/M4/PEMS from
  upstream, PSM/SMD/UEA via `benchmarks/tools/fetch_task_datasets.py`.

Run the benchmarks (from `TimeMixer/`; all runners are **idempotent** — a config whose log already
holds a valid MSE/MAE is skipped, so re-launching after an interruption is safe):

```bash
venv/Scripts/python.exe benchmarks/run_missing.py              # TimeMixer v1 accuracy
venv/Scripts/python.exe benchmarks/run_missing_pp.py           # TimeMixer++ accuracy
venv/Scripts/python.exe benchmarks/run_sparsetsf_benchmarks.py # SparseTSF accuracy (sl=96 & 720)
venv/Scripts/python.exe benchmarks/run_carbon_benchmarks.py    # 3-model CodeCarbon sweep
```

Per-task benchmarks (the 8 task categories the TimeMixer++ paper evaluates). Each trains SparseTSF
**and TimeMixer** locally, writes `reports/results_comparison_<task>.md` and `logs/run_logs_<task>/`,
and is idempotent. TimeMixer uses its upstream script settings where they exist (long-term, M4,
PEMS); elsewhere it uses its own long-term architecture from `benchmarks/tm_config.py`, the single
source of truth that reads them from `run_all_long_term_benchmarks.py`. TimeMixer++ is not re-run
per task (it is only stable at `lr=5e-5`, which undertrains it), so its published figures remain
the reference there.

```bash
venv/Scripts/python.exe benchmarks/tools/fetch_task_datasets.py       # PSM, 10x UEA, SMD
venv/Scripts/python.exe benchmarks/run_m4_benchmarks.py               # univariate short-term (SMAPE/MASE/OWA)
venv/Scripts/python.exe benchmarks/run_pems_benchmarks.py             # multivariate short-term (MAE/MAPE/RMSE)
venv/Scripts/python.exe benchmarks/run_imputation_benchmarks.py       # imputation (masked MSE/MAE)
venv/Scripts/python.exe benchmarks/run_fewshot_benchmarks.py          # few-shot (10% train)
venv/Scripts/python.exe benchmarks/run_zeroshot_benchmarks.py         # zero-shot transfer
venv/Scripts/python.exe benchmarks/run_anomaly_benchmarks.py          # anomaly (point-adjusted F1)
venv/Scripts/python.exe benchmarks/run_classification_benchmarks.py   # classification (accuracy)
venv/Scripts/python.exe benchmarks/tools/count_params_all.py          # params, every model/task/dataset run
venv/Scripts/python.exe benchmarks/tools/build_task_summary.py        # cross-task table in results_sparsetsf_all_tasks.md
venv/Scripts/python.exe benchmarks/tools/build_run_inventory.py       # runs-and-coverage tables, same report
```

SparseTSFPlus ablations (which grafted units earn their parameters). All idempotent:

```bash
venv/Scripts/python.exe benchmarks/run_period_sensitivity.py              # period_len diagnostics, no model code
venv/Scripts/python.exe benchmarks/run_sparsetsf_plus_imputation.py       # unit ablation, imputation
venv/Scripts/python.exe benchmarks/run_sparsetsf_plus_classification.py   # unit ablation, 10 UEA sets
venv/Scripts/python.exe benchmarks/run_sparsetsf_plus_forecast.py         # forecasting units: screen at H=96
venv/Scripts/python.exe benchmarks/run_sparsetsf_plus_forecast.py --full-best   # validation winner at every horizon
venv/Scripts/python.exe benchmarks/run_sparsetsf_plus_forecast.py --group pems  # PEMS03/04/07/08
venv/Scripts/python.exe benchmarks/run_sparsetsf_plus_anomaly.py          # leave-one-out on PSM, SMD
venv/Scripts/python.exe benchmarks/run_sparsetsf_plus_suite.py            # both versions on all 8 tasks + energy
venv/Scripts/python.exe benchmarks/run_sparsetsf_plus_suite.py --only pems,longterm --versions chcorr --keys PEMS03,ETTh1_pred96  # a candidate, some cells
venv/Scripts/python.exe benchmarks/run_seed_repeats.py                    # seeds 2022/2023 of the headline comparisons
venv/Scripts/python.exe benchmarks/tools/fft_periods.py                   # which period an FFT rule would pick (no training)
venv/Scripts/python.exe benchmarks/tools/assert_sparsetsf_plus_equiv.py   # THE GATE (see below)
venv/Scripts/python.exe benchmarks/tools/scrub_log_paths.py --apply       # BEFORE committing new logs
```

The imputation runner reads the published SparseTSF logs as its `base` rows when no SparseTSFPlus
base run exists (valid because base is bitwise SparseTSF), so plain SparseTSF stays a candidate in
the validation selection and a dataset can be assigned "no units". It never writes there. Runs longer
than about two hours should be split (`--datasets`, `--masks`): background jobs in this environment
are stopped at a time limit, and the runners resume cleanly only because they are idempotent.

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

There is no lint/test suite. The working substitute, in increasing strength — run all four after
touching paths, imports or the exp loop:

```bash
venv/Scripts/python.exe -m py_compile benchmarks/*.py benchmarks/tools/*.py models/*.py exp/*.py run.py
venv/Scripts/python.exe -c "from models import TimeMixer, TimeMixerPP, SparseTSF, SparseTSFPlus"
venv/Scripts/python.exe benchmarks/tools/assert_sparsetsf_plus_equiv.py
venv/Scripts/python.exe benchmarks/run_all_long_term_benchmarks.py --update-from-logs-only
venv/Scripts/python.exe benchmarks/tools/build_task_summary.py
```

Byte-compiling is **not** enough for import changes — a wrong module path only fails at runtime,
which is why the registry-import line is separate. For model-wiring changes, also do a real 1-epoch
run (`--train_epochs 1`, a throwaway `--model_id`) and then delete its `checkpoints/`, `results/`
and `test_results/` dirs.

## Repository, git & publishing

The workspace root is the git repo (branch `main`), pushed to the **public** GitHub repo
`BenguBarisBalkan/efficient-general-purpose-timeseries-models`. Consequences worth holding onto:

- **One root `.gitignore` is the single source of truth.** Upstream's nested `TimeMixer/.gitignore`
  was folded into it and deleted — do not reintroduce a nested one.
- **Ignored** (large and reproducible): `TimeMixer/venv/`, `dataset/`, `TimeMixer/results/`,
  `checkpoints/`, `test_results/`, `m4_results/`, `logs/result_*.txt`. Tracked is ~404 files / 8 MB.
- **`TimeMixer/logs/run_logs*/` IS tracked on purpose** — the 268 per-run logs there are the evidence behind every
  table *and* the input to the idempotency/skip decision. Do not add them to `.gitignore` to "clean
  up" the repo.
- **Fork provenance** is in `TimeMixer/UPSTREAM_FORK.md` + `upstream-fork.patch` (base commit
  `e246105` of github.com/kwuking/TimeMixer, 10 modified files). The original clone's `.git` was
  moved to `Masaüstü/timemixer-upstream-git-backup/` — outside any repo, not deleted.
- `.gitattributes` pins `* text=auto`; without it every file on this Windows box shows as rewritten.
- The root `LICENSE` is **MIT** and covers this study's own contributions (harness, TimeMixer++
  reimplementation, SparseTSFPlus, registry shims, reports). Keep the text pristine so GitHub's
  detector still recognises it, and keep the scope note in `README.md` rather than in the file.
  The vendored code keeps its own **Apache-2.0**: `TimeMixer/LICENSE` and
  `implementations/sparsetsf/LICENSE-SparseTSF`. Do not relicense either.

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

**2. The four models, wired via a shim pattern.** `model_dict` maps `TimeMixer`, `TimeMixerPP`,
`SparseTSF`, `SparseTSFPlus` → modules in `models/`, which is a **registry**: one file per
registered model name.
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
- `models/SparseTSFPlus.py` — SparseTSF plus **individually-toggleable units** grafted from the
  other two models (`implementations/sparsetsf_plus/`: `units.py` holds the units, `model.py` the
  task branching). Controlled by the additive `--stsf_*` flags in `run.py`, **every one of which
  defaults to stock-SparseTSF behaviour**. Residual units are zero-initialised, so with all units
  off the model is *numerically identical* to `SparseTSF`, and vanilla forecasting delegates
  straight to the vendored core. Measured results live in `reports/results_sparsetsf_plus_*.md`.

### Adding a unit to SparseTSFPlus

Add the flag to `_UNIT_DEFAULTS` in `implementations/sparsetsf_plus/units.py` **with the default
equal to current behaviour**, add the matching `parser.add_argument` in `run.py`, build it in
`Model._build_units`, and use it in the task path. Then re-run the equivalence gate — a unit whose
default changes behaviour breaks the ablation's baseline row and silently invalidates every
published SparseTSF number.

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
  ship with the repo. **MSL/SMAP** are rebuilt by `build_msl_smap()` in that script from a
  hand-downloaded Kaggle mirror of NASA's release (the telemanom S3 source now 403s); it writes
  nothing unless the rebuilt sizes equal the published ones. **SWaT** is still not obtainable
  (gated, iTrust request form); a Kaggle re-upload exists but is deliberately not used (no
  redistribution right, and it appears to lack the standard attack-period test split). Exact target
  filenames are in that script's header; the anomaly runner marks missing sets `NOT AVAILABLE`.
  Note: `timeseriesclassification.com` rejects urllib with HTTP 403 but serves **curl**, so the
  fetcher falls back to curl.
- **`--percent` (few-shot) is opt-in and only forwarded when < 100.** `data_provider/data_factory.py`
  passes it *conditionally*, so loaders that don't accept it (M4/PEMS/Solar) are untouched on the
  normal path. It truncates the TRAIN split only; the scaler still fits the full train range (TSLib
  convention) and val/test are unchanged.
- **`--stream_metrics 1` is opt-in and exists only for RAM.** `exp_long_term_forecasting.test()`
  normally holds every prediction and target in memory, which exceeds 16 GB for Electricity at
  pred_len 720 and Traffic at 336/720. The flag accumulates the same metrics batch by batch in
  float64; checked against the default path on ETTh1 and PEMS08, it agrees to 6–7 significant
  digits (the default sums in float32). Runners pass it only for the configs that need it, so every
  existing number was produced without it.
- **TimeMixer's classification path needs `--channel_independence 0`.** `TimeMixer.classification`
  embeds all channels jointly and has no per-channel reshape, so the long-term value of 1 crashes
  in the embedding conv. `tm_config.arch(..., channel_independence=0)` handles it for that runner.
- **`exp_long_term_forecasting.test()` prints `RMSE:`/`MAPE:` in UPPERCASE on purpose.** Every runner
  regexes lowercase `mse:`/`mae:` and takes the LAST match — a lowercase `rmse:` contains `mse:` as a
  substring and would silently corrupt every existing results table. Keep new metric labels uppercase.
- **Classification uses `optim.RAdam`, absent in torch 1.7.1** — `exp/exp_classification.py` falls back
  to Adam when RAdam is missing (model-agnostic; keeps RAdam on newer torch).
- **4 GB GPU / Windows limits.** Batch sizes are reduced from the papers' values. Solar-Energy,
  Electricity, and Traffic are excluded from the carbon sweep as too heavy. Energy: GPU power is real
  (NVML) but CPU/RAM are TDP estimates (no RAPL on Windows) — treat relative ratios as sound and
  absolute kg CO₂ as an estimate.
- **The repo is public, and fresh run logs contain local absolute paths.** `run.py` echoes the
  argparse `Namespace`, so every new log embeds `C:\Users\<you>\...\timemixer\...`.
  Tracked logs were scrubbed to `<workspace>` once; **re-scrub before committing new ones** with
  `benchmarks/tools/scrub_log_paths.py --apply` (dry run without the flag). The trap it handles: the
  paths appear in *two* forms — raw in tracebacks, and with **every backslash doubled** in the
  `Namespace` `repr()`. A pattern matching only single backslashes silently finds ~2% of them (5
  files instead of 267). The tool derives the path at runtime rather than hardcoding it (a hardcoded
  path in a committed file would leak it), and refuses to write any file whose metrics would change.
  Run without path arguments, it also blanks the **IP-geolocated position** (`longitude`, `latitude`,
  `region`) that CodeCarbon writes into every `logs/emissions_*.csv` row; no report uses those
  columns, the country is kept for the CO₂ factor, and it refuses if any energy value would change.
  Rows committed before 2026-10-21 still carry coordinates in git history.
- **`.gitignore` does not support same-line comments.** `foo/    # 7 GB` is parsed as a pattern
  *including* the comment, so it matches nothing and the directory is silently committed. Every
  note in that file must sit on its own line. This bit once already (`m4_results/` leaked into
  staging).
- **The all-units-off equivalence gate is non-negotiable.**
  `benchmarks/tools/assert_sparsetsf_plus_equiv.py` asserts that `SparseTSFPlus` with every unit at
  its default is **bitwise identical** to `SparseTSF` on all five tasks (currently
  `max|diff| = 0.0`). Run it after touching anything under `implementations/sparsetsf_plus/`. If it
  fails, the ablation's baseline row is no longer SparseTSF and every published SparseTSF number
  becomes unsafe to compare against.
- **RevIN cannot help SparseTSF, and `--stsf_revin 1` is a mathematical no-op.** The forecast map is
  linear and entirely bias-free, hence positively homogeneous: `f(x/σ)·σ == f(x)`, so per-window
  per-channel scaling cancels exactly (measured `max|diff|` 1.5e-05, i.e. float32 rounding; it
  changed nothing on PEMS08 or Weather). Std scaling only becomes live once something breaks
  homogeneity — `--stsf_revin 2` (subtract_last), `--stsf_linear_bias 1`, or `--stsf_revin_affine 1`
  (which costs `2·enc_in` and breaks zero-shot checkpoint portability). Do not report it as a unit.
- **Three exp loops early-stop on the TEST split**, so they offer no clean validation signal:
  `exp_imputation.py:164` and `exp_anomaly_detection.py:129` pass `test_loss` to `early_stopping`,
  and `exp_classification.py:88-89` uses `flag='TEST'` for *both* the "vali" and "test" loaders.
  Only `exp_long_term_forecasting.py:222` uses `vali_loss`. This is upstream TSLib behaviour and
  every model here was trained under it, so cross-model comparisons stay internally consistent —
  **do not "fix" it**, that would invalidate every existing table. Instead select hyperparameters
  and units on the reported best-validation column where one exists, and label classification
  results a sensitivity analysis.
- **`period_len` is load-bearing far beyond forecasting, including where a comment says it is not.**
  `run_classification_benchmarks.py:78` passes `--period_len 1` as "unused by the classification
  head", but it sets the aggregating conv's kernel to `1 + 2*(period_len//2)` — so `1` means a
  kernel of size **one**, i.e. no receptive field at all. Measured sensitivity is large: ETTh1
  forecasting prefers `12` over the published `24` (−3.22% MSE), and PEMS08 prefers `1` over `12`
  (−24% MAE, because the period fold is actively harmful there). See
  `reports/results_period_sensitivity.md`.
- **`--track_emissions` is opt-in** (default off) so ordinary accuracy runs carry no CodeCarbon
  overhead or dependency. `run.py` imports `codecarbon` lazily only when the flag is set.
