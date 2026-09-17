# Efficient General-Purpose Time-Series Models

A local benchmark of **three long-term time-series forecasting models** on two axes that are
usually reported separately — **accuracy** and **energy/carbon cost** — plus an eight-task
generality study. Everything was run on one consumer laptop: **RTX 3050 Ti Laptop GPU, 4 GB VRAM,
Windows**.

| Model | Paper | Params (ETTh) | Role here |
|---|---|---:|---|
| **TimeMixer** | ICLR 2024 | 75k → 4.0M | Baseline / reference (upstream repo, forked) |
| **TimeMixer++** | ICLR 2025 | 113k → 1.1M | Reimplemented from scratch for this study |
| **SparseTSF** | ICML 2024 (Oral) | **41 → 925** | Vendored, then extended to all five TSLib tasks |

Parameter counts grow with the input/horizon, so each cell spans `sl=96/pred=96` →
`sl=720/pred=720`; they are measured, not estimated — regenerate with
[`benchmarks/tools/count_params.py`](TimeMixer/benchmarks/tools/count_params.py).

## Headline result

**SparseTSF matches TimeMixer's accuracy using ~9% of the energy and three orders of magnitude
fewer parameters — and at its own native lookback it is the most accurate model of the three.**

| Model | Accuracy (sl=96, mean MSE over 20 configs) | Energy (total kWh) | vs TimeMixer |
|---|---:|---:|---:|
| **SparseTSF** | 0.354 (≈ tied) | **7.77e-03** | **0.09× — 11× less** |
| **TimeMixer** | **0.348 (best)** | 8.93e-02 | 1.00× |
| **TimeMixer++** | 0.449 (undertrained — see below) | 1.49e-01 | 1.67× — most |

Three things fall out of this:

1. **Accuracy is a near tie at matched lookback.** 0.348 vs 0.354 across 20 configs, split by data
   frequency: SparseTSF sweeps the hourly ETTh sets (its `period_len=24` locks onto the daily
   cycle), TimeMixer takes the 15-min ETTm and 10-min Weather sets.
2. **Given its native `seq_len=720`, SparseTSF wins outright** — mean MSE **0.312**, beating
   TimeMixer's 0.348, with 925 parameters and 15–90 s per run. (A longer input is a different task,
   so this is reported separately rather than merged into the table above.)
3. **Added architectural complexity did not pay for itself here.** TimeMixer++ cost ~1.67× the
   energy of TimeMixer (2.07–2.10× on ETT, 1.19× on Weather) without buying accuracy in our runs.

→ Full write-up with per-dataset tables: **[`TimeMixer/reports/results_final_report.md`](TimeMixer/reports/results_final_report.md)**

### How to read these numbers honestly

- **TimeMixer++ is undertrained by design.** It needs `lr=5e-5` to avoid NaN divergence, and at that
  rate it does not converge within 20 epochs. Its accuracy column **understates** the architecture;
  this is not a clean "TimeMixer++ is worse" finding. Its *energy* number, measured under a matched
  protocol, is fair.
- **Two protocols that must never be cross-compared.** Accuracy tables use each model's own
  published hyperparameters with early stopping. The carbon sweep uses a *matched* config at a fixed
  5 epochs with early stopping off, to isolate energy per unit of training work — so the MSE column
  in the carbon table is **not** accuracy.
- **Energy caveat.** GPU power is real (NVML); Windows has no RAPL, so CPU/RAM are TDP estimates.
  Relative ratios are sound on one machine; treat absolute kg CO₂ as an estimate.
- **Scope.** The three-way study covers 5 datasets (ETTh1/2, ETTm1/2, Weather). Solar-Energy,
  Electricity and Traffic were too heavy for 4 GB to include in the carbon sweep.

## Generality: SparseTSF across all eight paper tasks

SparseTSF ships as forecasting-only. It was extended here to the full TSLib task set — new
cross-period reconstruction heads for imputation and anomaly detection, and a conv-aggregation head
for classification — then run against the eight task categories the TimeMixer++ paper evaluates.

| Task | Metric | SparseTSF (ours) | Reference (published) |
|---|---|---:|---:|
| Long-term forecasting | mean MSE | 0.354 (sl=96) / 0.312 (sl=720) | *TimeMixer 0.348* |
| Univariate short-term (M4) | OWA ↓ | 0.972 | *TimeMixer 0.840* |
| Multivariate short-term (PEMS) | MAE ↓ | 34.13 | *TimeMixer 17.41* |
| Imputation | masked MSE ↓ | 0.0838 (6 ds) | *TimeMixer 0.0586 (5 ds)* |
| Few-shot (10% train) | MSE ↓ | **0.441** | *TimeMixer 0.453* |
| Zero-shot transfer | MSE ↓ | **0.393** | *TimeMixer 0.467* |
| Anomaly detection | F1 ↑ | **87.64** (2 of 5 ds) | *TimeMixer++ 87.47 (5 ds)* |
| Classification | accuracy ↑ | 60.90 | *TimeMixer++ 75.9* |

**Only SparseTSF was run locally; every italic figure is the published value from the TimeMixer++
paper** — different hardware, the authors' own tuning, and for imputation a different input length.
They are reference points, **not** a controlled head-to-head.

What it suggests: a ~1k-parameter model has very little to overfit, which shows up as a
**generalization advantage** — it beats published TimeMixer on all 6 zero-shot transfer pairs and
edges it few-shot. It loses where the task is not forecasting-shaped or where channels interact
(PEMS traffic networks, classification), which is the expected cost of being channel-independent
with a periodic inductive bias.

→ Per-task detail: **[`TimeMixer/reports/results_sparsetsf_all_tasks.md`](TimeMixer/reports/results_sparsetsf_all_tasks.md)**
· How each head was built: **[`methodology_sparsetsf_tasks.md`](TimeMixer/reports/methodology_sparsetsf_tasks.md)**

---

## Where do I find…?

| I'm looking for… | It's here |
|---|---|
| **The final write-up (all 3 models, accuracy + carbon)** | [`TimeMixer/reports/results_final_report.md`](TimeMixer/reports/results_final_report.md) |
| Per-model accuracy tables | `TimeMixer/reports/results_comparison*.md` |
| **Per-task benchmarks** (M4, PEMS, imputation, few/zero-shot, anomaly, classification) | `TimeMixer/reports/results_comparison_{m4,pems,imputation,fewshot,zeroshot,anomaly,classification}.md` |
| Energy / CO₂ tables | [`results_carbon_comparison.md`](TimeMixer/reports/results_carbon_comparison.md) |
| **What every dataset is** (all tasks, measured stats) | [`datasets_reference.md`](TimeMixer/reports/datasets_reference.md) |
| The scripts that run the benchmarks | [`TimeMixer/benchmarks/`](TimeMixer/benchmarks) (+ its own README) |
| Raw run logs / emissions data | `TimeMixer/logs/` |
| The 3 model definitions | `TimeMixer/models/` (registry) + `TimeMixer/implementations/` (the code) |
| Relation to upstream TimeMixer | [`TimeMixer/UPSTREAM_FORK.md`](TimeMixer/UPSTREAM_FORK.md) |

## Setup

A fresh clone **cannot run anything yet** — the environment and the data are both deliberately
untracked (see [what is not in the repository](#what-is-not-in-the-repository)).

```bash
# 1. environment (Python 3.8, torch 1.7.1+cu110)
python -m venv TimeMixer/venv
TimeMixer/venv/Scripts/pip install -r TimeMixer/requirements.txt

# 2. data — ETT/Weather/ECL/M4/PEMS come from upstream TimeMixer into ./dataset/;
#    PSM, SMD and the 10 UEA classification sets are fetched for you:
cd TimeMixer && venv/Scripts/python.exe benchmarks/tools/fetch_task_datasets.py
```

## Running a benchmark

Always use the venv's Python, from inside `TimeMixer/` (the runners write to relative paths):

```bash
venv/Scripts/python.exe benchmarks/run_missing.py              # TimeMixer accuracy
venv/Scripts/python.exe benchmarks/run_missing_pp.py           # TimeMixer++ accuracy
venv/Scripts/python.exe benchmarks/run_sparsetsf_benchmarks.py # SparseTSF (sl=96 and 720)
venv/Scripts/python.exe benchmarks/run_carbon_benchmarks.py    # energy/CO2 for all 3
```

Every runner is **idempotent**: a config whose log already holds a valid result is skipped, so
re-launching after an interruption is safe. Each rewrites its own report between `<!-- AUTO_* -->`
markers. To rebuild a table from existing logs without training anything:

```bash
venv/Scripts/python.exe benchmarks/run_all_long_term_benchmarks.py --update-from-logs-only
```

## How the models plug in

One pipeline, three models — selected by `--model`:

```
run.py  --model {TimeMixer|TimeMixerPP|SparseTSF}  --data … --seq_len 96 --pred_len 96 …
   └─ exp/exp_long_term_forecasting.py       training / testing loop
        └─ exp/exp_basic.py  model_dict      registry: name → module
             └─ models/<Name>.py             entry point → implementations/<name>/
```

`models/` is a registry — one file per registered model name — and the substantial code lives in
`implementations/`, one package per model. Vendored SparseTSF sits next to its own licence, keeping
the "whose code is this" boundary visible in the layout. `models/TimeMixer.py` stays where upstream
put it, so upstream changes still merge cleanly.

## Layout

```
.
├── TimeMixer/              the study (fork of github.com/kwuking/TimeMixer)
│   │
│   │   ── model code ──────────────────────────────────────────────
│   ├── models/             Registry entry points (imported by exp/exp_basic.py):
│   │   ├── TimeMixer.py        self-contained TimeMixer (ICLR'24), upstream file
│   │   ├── TimeMixerPP.py      5-line shim → implementations/timemixer_pp/
│   │   └── SparseTSF.py        adapter + the non-forecast task heads
│   ├── implementations/    the substantial model code, one package each:
│   │   ├── timemixer_pp/       TimeMixer++ (ICLR'25), written from scratch here
│   │   └── sparsetsf/          SparseTSF (ICML'24) vendored + its Apache-2.0 LICENSE
│   │
│   │   ── shared TSLib scaffold (upstream, mostly untouched) ───────
│   ├── run.py              CLI entry point (argparse → exp loop)
│   ├── exp/                train/val/test loops, one per task
│   ├── layers/             building blocks (decomp, embed, attention…)
│   ├── data_provider/      dataset loaders
│   ├── utils/              metrics, early stopping, LR schedules
│   ├── scripts/            upstream shell scripts (reference)
│   │
│   │   ── benchmark harness (added on top of upstream) ─────────────
│   ├── benchmarks/         all run_*.py runners  (see benchmarks/README.md)
│   ├── reports/            all results_*.md write-ups  ← read these
│   ├── results/            auto-generated .npy per run (pred/true/metrics) — gitignored
│   ├── logs/               run_logs*/ (raw logs), emissions_timemixer.csv,
│   │                       console/ (captured stdout/stderr)
│   ├── checkpoints/        saved weights — gitignored
│   └── test_results/       per-run test output — gitignored
│
├── dataset/                ETT-small, electricity, traffic, weather, solar, PEMS, m4… — gitignored
├── LICENSE                 MIT, for this project's own contributions
└── .gitignore              one source of truth for the whole workspace
```

Further studies get their own top-level folder alongside `TimeMixer/`, sharing `dataset/`.

## What is not in the repository

Ignored because it is large and reproducible, not because it is unimportant:

| Path | Size | How to get it back |
|---|---:|---|
| `TimeMixer/venv/` | 4.7 GB | See [Setup](#setup) |
| `dataset/` | 2.7 GB | See [Setup](#setup) |
| `TimeMixer/results/` | 7.4 GB | Re-run; it is `pred.npy`/`true.npy`/`metrics.npy` per run setting |
| `TimeMixer/checkpoints/` | 270 MB | Re-run (saved weights) |
| `TimeMixer/test_results/`, `m4_results/` | 52 MB | Re-run |

`TimeMixer/logs/` **is** tracked — the 268 per-run logs (11 `run_logs*/` dirs) are the evidence behind every table in
`reports/`, and the runners read them to decide what to skip.

## License

This project's own contributions are **MIT** licensed — see [`LICENSE`](LICENSE). That covers the
benchmark harness (`TimeMixer/benchmarks/`), the TimeMixer++ reimplementation
(`TimeMixer/implementations/timemixer_pp/`), the registry shims and SparseTSF task heads
(`TimeMixer/models/TimeMixerPP.py`, `TimeMixer/models/SparseTSF.py`), and the reports and docs.

Third-party components keep their own licences and are **not** relicensed:

| Component | Licence |
|---|---|
| Upstream TimeMixer / TSLib scaffold — `run.py`, `exp/`, `layers/`, `data_provider/`, `utils/`, `scripts/`, `models/TimeMixer.py` | Apache-2.0 — [`TimeMixer/LICENSE`](TimeMixer/LICENSE) |
| Vendored SparseTSF core — `implementations/sparsetsf/model.py` | Apache-2.0 — [`LICENSE-SparseTSF`](TimeMixer/implementations/sparsetsf/LICENSE-SparseTSF) |

Modifications to the upstream Apache-2.0 files are itemized in
[`TimeMixer/UPSTREAM_FORK.md`](TimeMixer/UPSTREAM_FORK.md) (base commit `e246105`, 10 files,
+147/−19) and captured verbatim in `TimeMixer/upstream-fork.patch`.

## Citing the original work

This is a reproduction and comparison study; the architectures are the original authors'. Please
cite [TimeMixer](https://github.com/kwuking/TimeMixer) (Wang et al., ICLR 2024), TimeMixer++
(Wang et al., ICLR 2025), and [SparseTSF](https://github.com/lss-1138/SparseTSF)
(Lin et al., ICML 2024) as appropriate.
