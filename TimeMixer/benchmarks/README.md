# benchmarks/

Runner scripts for the accuracy and energy benchmarks. Each derives all paths from
`REPO_ROOT = Path(__file__).resolve().parent.parent` (i.e. the `TimeMixer/` folder), then
shells out to `run.py` once per (dataset × pred_len) config, capturing each run's stdout into
a log under `../logs/` and parsing MSE/MAE back out to update a report in `../reports/`.

Run from the `TimeMixer/` directory with the venv Python, e.g.:

```bash
venv/Scripts/python.exe benchmarks/run_missing.py
```

| Script | Model(s) | Writes report | Writes logs to | Notes |
|---|---|---|---|---|
| `run_all_long_term_benchmarks.py` | TimeMixer | `reports/results_comparison.md` | `logs/run_logs/` | Full 8-dataset × 4-pred_len sweep. `--update-from-logs-only` rebuilds the table from existing logs without training. |
| `run_missing.py` | TimeMixer | ″ | ″ | Same as above but **skips** configs that already have a valid result. Preferred for resuming. |
| `run_all_pp_benchmarks.py` | TimeMixer++ | `reports/results_comparison_pp.md` | `logs/run_logs_pp/` | Runs every config unconditionally (no skip). |
| `run_missing_pp.py` | TimeMixer++ | ″ | ″ | Skip-completed version of the above. Preferred. |
| `run_sparsetsf_benchmarks.py` | SparseTSF | `reports/results_comparison_sparsetsf.md` | `logs/run_logs_sparsetsf/` | Sweeps seq_len ∈ {96, 720}. Idempotent. Also builds a 3-way comparison from the v1/PP tables. |
| `run_carbon_benchmarks.py` | all 3 | `reports/results_carbon_comparison.md` | `logs/run_logs_carbon/` | Matched config, fixed epochs, early-stopping off. Uses CodeCarbon (`--track_emissions`) → `logs/emissions_timemixer.csv`. Idempotent. |
| `monitor_electricity_then_stop.py` | — | — | — | One-off watcher (stale hard-coded PID). Kept for reference only. |

**Idempotency:** except for `run_all_pp_benchmarks.py`, a config whose log already contains a
valid MSE/MAE is skipped, so re-launching after an interruption is safe.
