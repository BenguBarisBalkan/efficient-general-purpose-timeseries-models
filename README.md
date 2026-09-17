# Time-Series Forecasting Benchmark Workspace

Benchmarks **three long-term forecasting models** on both **accuracy** and **energy/carbon**,
on a Windows laptop (RTX 3050 Ti, 4 GB VRAM):

| Model | Paper | Role here |
|---|---|---|
| **TimeMixer** | ICLR 2024 | The baseline / reference model (upstream repo). |
| **TimeMixer++** | ICLR 2025 | From-scratch reimplementation, benchmarked against v1. |
| **SparseTSF** | ICML 2024 (Oral) | ~1k-parameter efficiency model, vendored + extended to all tasks. |

All work lives in **`TimeMixer/`**. The other two top-level folders are `dataset/` (shared data)
and `_archive/` (old clutter — safe to delete, see bottom).

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
| The 3 model definitions | `TimeMixer/models/` (+ `TimeMixer_plus/`, `SparseTSF_model/`) |
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
│   ├── models/             Entry points registered in exp/exp_basic.py:
│   │   ├── TimeMixer.py        self-contained TimeMixer (ICLR'24)
│   │   ├── TimeMixerPP.py      5-line wrapper → TimeMixer_plus/model.py
│   │   └── SparseTSF.py        adapter → SparseTSF_model/model.py
│   ├── TimeMixer_plus/     TimeMixer++ from-scratch implementation (model.py)
│   ├── SparseTSF_model/    SparseTSF vendored from upstream (Apache-2.0)
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
└── _archive/              old clutter, safe to delete (see below)
```

## How the models plug in

One pipeline, three models — selected by `--model`:

```
run.py  --model {TimeMixer|TimeMixerPP|SparseTSF}  --data … --seq_len 96 --pred_len 96 …
   └─ exp/exp_long_term_forecasting.py       (training / testing loop)
        └─ exp/exp_basic.py  model_dict      (registry mapping name → module)
             └─ models/<name>.py             (thin entry point → real implementation)
```

`TimeMixerPP.py` and `SparseTSF.py` are deliberately thin shims: they keep the real
implementation in a sibling package (`TimeMixer_plus/`, `SparseTSF_model/`) so the upstream
`models/` folder stays clean and the exp loop needs no changes.

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

## `_archive/` (safe to delete)

Moved here during cleanup — nothing in the project depends on it:

- `python-installers/` — Python 3.8 installers (the venv is already built).
- `codecarbon-scratch/` — early standalone CodeCarbon test (iris), pre-integration.
- `TimeMixer _plus (dead clone)/` — an abandoned early clone of upstream TimeMixer. Its
  `models/` had only the stock `TimeMixer.py` (no ++/SparseTSF) and its run-logs were exact
  duplicates of `TimeMixer/logs/run_logs/`. **Not** the TimeMixer++ code — that lives in
  `TimeMixer/TimeMixer_plus/`.

Once you've confirmed you don't need any of it, delete the whole `_archive/` folder.
