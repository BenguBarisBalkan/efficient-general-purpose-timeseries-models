"""Run only missing/incomplete long-term benchmark experiments.

Checks existing logs first — only re-runs a config if it has no valid MSE/MAE result.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_all_long_term_benchmarks import (
    DATASETS,
    PRED_LENS,
    PAPER,
    REPO_ROOT,
    run_one,
    parse_log_file,
    update_md,
)

rows = []

for ds in DATASETS:
    for pred_len in PRED_LENS:
        paper_mse, paper_mae = PAPER[ds.name][pred_len]
        log_path = REPO_ROOT / "logs" / "run_logs" / f"{ds.name}_pred{pred_len}.log"
        mse, mae, runtime_s, status = parse_log_file(log_path)

        newly_run = False
        if mse is None or mae is None:
            print(f"\n>>> Running {ds.name} pred_len={pred_len} <<<", flush=True)
            mse, mae, runtime_s, status = run_one(ds, pred_len)
            newly_run = True
        else:
            print(f"[SKIP] {ds.name} pred_len={pred_len} — already complete (MSE={mse:.6f})", flush=True)

        rows.append(
            {
                "dataset": ds.name,
                "pred_len": str(pred_len),
                "paper_mse": f"{paper_mse:.3f}",
                "paper_mae": f"{paper_mae:.3f}",
                "local_mse": "-" if mse is None else f"{mse:.6f}",
                "local_mae": "-" if mae is None else f"{mae:.6f}",
                "runtime_s": "-" if runtime_s is None else f"{runtime_s:.1f}",
                "status": status,
            }
        )

        if newly_run:
            update_md(rows)
            print(f"[MD] results_comparison.md updated after {ds.name} pred_len={pred_len}.", flush=True)

update_md(rows)
print("\n[DONE] results_comparison.md updated with all results.", flush=True)
