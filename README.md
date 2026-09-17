# Time-Series Forecasting Benchmark Workspace

Benchmarks **three long-term forecasting models** on both **accuracy** and **energy/carbon**,
on a Windows laptop (RTX 3050 Ti, 4 GB VRAM):

| Model | Paper | Role here |
|---|---|---|
| **TimeMixer** | ICLR 2024 | The baseline / reference model (upstream repo). |
| **TimeMixer++** | ICLR 2025 | From-scratch reimplementation, benchmarked against v1. |
| **SparseTSF** | ICML 2024 (Oral) | ~1k-parameter efficiency model, vendored + extended to all tasks. |

The workspace is a **git repository** (`main`). Study code lives in **`TimeMixer/`**; `dataset/`
holds the shared data and is gitignored, as are the venv and all machine-generated run output —
see [`.gitignore`](.gitignore) for what is and isn't tracked. Further studies get their own
top-level folder alongside `TimeMixer/`, sharing `dataset/`.

**Tasks.** All three models support the five Time-Series-Library tasks — long/short-term
forecasting, imputation, anomaly detection, classification. SparseTSF was extended to the full set
(`models/SparseTSF.py`); forecasting still delegates to the untouched vendored core. Datasets not
shipped with the repo (PSM for anomaly, a UEA set for classification) are fetched by
`TimeMixer/benchmarks/tools/fetch_task_datasets.py`; MSL/SMAP/SMD need the TimesNet `.npy` bundle and
SWaT is gated (see that script's header).

---

## Where do I find…?

| I'm looking for… | It's here |
|---|---|
| **The final write-up (all 3 models, accuracy + carbon)** | `TimeMixer/reports/results_final_report.md` |
| Per-model accuracy tables | `TimeMixer/reports/results_comparison*.md` |
| **Per-task SparseTSF benchmarks** (M4, PEMS, imputation, few/zero-shot, anomaly, classification) | `TimeMixer/reports/results_comparison_{m4,pems,imputation,fewshot,zeroshot,anomaly,classification}.md` |
| Energy / CO₂ tables | `TimeMixer/reports/results_carbon_comparison.md` |
| The scripts that run the benchmarks | `TimeMixer/benchmarks/` |
| Raw run logs / emissions data | `TimeMixer/logs/` |
| The 3 model definitions | `TimeMixer/models/` (registry) + `TimeMixer/implementations/` (the code) |
| The Python environment | `TimeMixer/venv/` (Python 3.8, PyTorch CUDA) |
| The datasets (files) | `dataset/` |
| **What every dataset is** (all tasks, measured stats) | `TimeMixer/reports/datasets_reference.md` |

---

## Layout

```
timemixer/
├── TimeMixer/              ← THE PROJECT (git → github.com/kwuking/TimeMixer, forked)
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
│   ├── exp/                train/val/test loops per task
│   ├── layers/             building blocks (decomp, embed, attention…)
│   ├── data_provider/      dataset loaders
│   ├── utils/              metrics, early stopping, LR schedules
│   ├── scripts/            upstream shell scripts (reference)
│   │
│   │   ── YOUR benchmark harness (added on top) ────────────────────
│   ├── benchmarks/         all run_*.py runners  (see benchmarks/README.md)
│   ├── reports/            all results_*.md write-ups  ← read these (hand-written)
│   ├── results/            auto-generated .npy per run (pred/true/metrics) — gitignored
│   ├── logs/               run_logs*/ (raw logs), emissions_timemixer.csv,
│   │                       console/ (captured stdout/stderr)
│   ├── checkpoints/        saved model weights
│   └── test_results/       per-run test outputs
│
├── dataset/               ETT-small, electricity, traffic, weather, solar, PEMS, m4…
│
└── .gitignore             one source of truth: ignores venv, dataset, run output
```

## How the models plug in

One pipeline, three models — selected by `--model`:

```
run.py  --model {TimeMixer|TimeMixerPP|SparseTSF}  --data … --seq_len 96 --pred_len 96 …
   └─ exp/exp_long_term_forecasting.py       (training / testing loop)
        └─ exp/exp_basic.py  model_dict      (registry mapping name → module)
             └─ models/<name>.py             (thin entry point → real implementation)
```

`TimeMixerPP.py` and `SparseTSF.py` are deliberately thin: they keep the substantial code in
`implementations/`, one package per model, so `models/` stays a registry and the exp loop needs
no changes. Vendored SparseTSF sits next to its own licence, which keeps the "whose code is this"
boundary visible. `TimeMixer.py` stays put as an upstream file, so upstream changes still merge.

## Running a benchmark

Always use the venv's Python. From `TimeMixer/`:

```bash
venv/Scripts/python.exe benchmarks/run_missing.py            # TimeMixer v1 (skips completed)
venv/Scripts/python.exe benchmarks/run_missing_pp.py         # TimeMixer++
venv/Scripts/python.exe benchmarks/run_sparsetsf_benchmarks.py
venv/Scripts/python.exe benchmarks/run_carbon_benchmarks.py  # energy/CO2 for all 3
```

Runners are **idempotent**: a config whose log already holds a valid result is skipped, so
re-launching after an interruption is safe. See `TimeMixer/benchmarks/README.md` for details.

## What is not in the repository

Ignored because it is large and reproducible, not because it is unimportant:

| Path | Size | How to get it back |
|---|---:|---|
| `TimeMixer/venv/` | 4.7 GB | `python -m venv TimeMixer/venv` + `pip install -r TimeMixer/requirements.txt` (Python 3.8) |
| `dataset/` | 2.7 GB | ETT/Weather/ECL/M4/PEMS from upstream; PSM/SMD/UEA via `TimeMixer/benchmarks/tools/fetch_task_datasets.py` |
| `TimeMixer/results/` | 7.4 GB | Re-run; it is `pred.npy`/`true.npy`/`metrics.npy` per run setting |
| `TimeMixer/checkpoints/` | 270 MB | Re-run (saved weights) |
| `TimeMixer/test_results/`, `m4_results/` | 52 MB | Re-run |

`TimeMixer/logs/` **is** tracked — 290 per-run logs are the evidence behind every table in
`reports/`, and the runners read them to decide what to skip.

Relation to upstream TimeMixer (base commit, the 10 modified files, how to diff) is recorded in
[`TimeMixer/UPSTREAM_FORK.md`](TimeMixer/UPSTREAM_FORK.md).
