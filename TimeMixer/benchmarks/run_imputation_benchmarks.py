"""
Imputation benchmark: SparseTSF vs TimeMixer (random-mask reconstruction).

Task added to SparseTSF in models/SparseTSF.py. This sweeps mask_rate ∈ {0.125,0.25,0.375,0.5}
× datasets {ETTh1,ETTh2,ETTm1,ETTm2,Weather} at seq_len=96, pred_len=0, and records the masked-
position MSE/MAE each run prints. Mirrors benchmarks/run_sparsetsf_benchmarks.py:
  • REPO_ROOT = <TimeMixer/>, shells run.py via sys.executable with cwd=REPO_ROOT
  • logs  -> logs/run_logs_imputation/<model>_<dataset>_mask<rate>.log
  • report -> reports/results_comparison_imputation.md (between AUTO markers)
Idempotent: a config whose log already holds a valid MSE/MAE is skipped, so re-launching is safe.

Quick harness check:  venv/Scripts/python.exe benchmarks/run_imputation_benchmarks.py --smoke
Full sweep:           venv/Scripts/python.exe benchmarks/run_imputation_benchmarks.py
"""

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT.parent / "dataset"
LOGS_DIR = REPO_ROOT / "logs" / "run_logs_imputation"
MD_PATH = REPO_ROOT / "reports" / "results_comparison_imputation.md"

# MODELS = what the report displays (TimeMixer cells from the earlier full sweep are kept).
# RUN_MODELS = what this runner will actually train now — SparseTSF only (current scope).
MODELS = ["SparseTSF", "TimeMixer"]
RUN_MODELS = ["SparseTSF"]
MASK_RATES = [0.125, 0.25, 0.375, 0.5]
SEQ_LEN = 96
TRAIN_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.001
DES = "Imp"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    data: str
    data_path: str
    root_path: Path
    enc_in: int
    period_len: int          # SparseTSF; must divide SEQ_LEN=96
    d_model: int             # TimeMixer
    e_layers: int            # TimeMixer
    d_ff: int                # TimeMixer
    down_sampling_layers: int  # TimeMixer
    down_sampling_window: int  # TimeMixer


# period_len: hourly ETTh -> 24, 15-min ETTm / 10-min Weather -> 4 (all divide 96).
DATASETS: List[DatasetConfig] = [
    DatasetConfig("ETTh1", "ETTh1", "ETTh1.csv", DATA_ROOT / "ETT-small", 7, 24, 16, 2, 32, 3, 2),
    DatasetConfig("ETTh2", "ETTh2", "ETTh2.csv", DATA_ROOT / "ETT-small", 7, 24, 16, 2, 32, 3, 2),
    DatasetConfig("ETTm1", "ETTm1", "ETTm1.csv", DATA_ROOT / "ETT-small", 7, 4, 16, 2, 32, 3, 2),
    DatasetConfig("ETTm2", "ETTm2", "ETTm2.csv", DATA_ROOT / "ETT-small", 7, 4, 32, 2, 32, 3, 2),
    DatasetConfig("Weather", "custom", "weather.csv", DATA_ROOT / "weather", 21, 4, 16, 2, 32, 3, 2),
    # Electricity — the 6th dataset in the paper's imputation benchmark (hourly -> period_len 24).
    DatasetConfig("Electricity", "custom", "electricity.csv", DATA_ROOT / "electricity", 321, 24, 16, 2, 32, 3, 2),
]


# ── Log parsing ──────────────────────────────────────────────────────────────
MSE_PAT = re.compile(r"mse:([0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?)")
MAE_PAT = re.compile(r"mae:([0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?)")
EPOCH_PAT = re.compile(r"Epoch:\s+\d+\s+cost time:\s+([0-9]+(?:\.[0-9]+)?)")


def _last(pat: re.Pattern, text: str) -> Optional[float]:
    m = pat.findall(text)
    return float(m[-1]) if m else None


def log_path_for(model: str, ds: DatasetConfig, mask_rate: float) -> Path:
    return LOGS_DIR / f"{model}_{ds.name}_mask{mask_rate}.log"


def parse_log(path: Path) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    if not path.exists():
        return None, None, None, "PENDING"
    text = path.read_text(encoding="utf-8", errors="ignore")
    mse, mae = _last(MSE_PAT, text), _last(MAE_PAT, text)
    if mse is None or mae is None:
        return None, None, None, "RUNNING/INCOMPLETE"
    epochs = [float(x) for x in EPOCH_PAT.findall(text)]
    return mse, mae, (sum(epochs) if epochs else None), "OK"


# ── Runner ───────────────────────────────────────────────────────────────────
def run_one(model: str, ds: DatasetConfig, mask_rate: float, train_epochs: int) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = log_path_for(model, ds, mask_rate)
    model_id = f"IMP_{model}_{ds.name}_mask{mask_rate}"

    cmd = [
        sys.executable, "-u", str(REPO_ROOT / "run.py"),
        "--task_name", "imputation",
        "--is_training", "1",
        "--model_id", model_id,
        "--model", model,
        "--data", ds.data,
        "--root_path", str(ds.root_path),
        "--data_path", ds.data_path,
        "--features", "M",
        "--seq_len", str(SEQ_LEN),
        "--label_len", "0",
        "--pred_len", "0",
        "--enc_in", str(ds.enc_in),
        "--dec_in", str(ds.enc_in),
        "--c_out", str(ds.enc_in),
        "--mask_rate", str(mask_rate),
        "--learning_rate", str(LEARNING_RATE),
        "--lradj", "TST",   # OneCycle (validated: ETTh1 mask0.125 -> mse~0.13 in a few epochs)
        "--train_epochs", str(train_epochs),
        "--batch_size", str(BATCH_SIZE),
        "--num_workers", "0",
        "--des", DES,
        "--itr", "1",
        # SparseTSF params (ignored by TimeMixer)
        "--period_len", str(ds.period_len),
        "--model_type", "linear",
        # TimeMixer params (ignored by SparseTSF)
        "--d_model", str(ds.d_model),
        "--d_ff", str(ds.d_ff),
        "--e_layers", str(ds.e_layers),
        "--down_sampling_layers", str(ds.down_sampling_layers),
        "--down_sampling_window", str(ds.down_sampling_window),
        "--down_sampling_method", "avg",
        "--channel_independence", "1",
    ]

    print(f"\n[RUN] {model} {ds.name} mask_rate={mask_rate}", flush=True)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as f:
        subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
    print(f"      finished in {time.perf_counter() - start:.1f}s", flush=True)


# ── Report ───────────────────────────────────────────────────────────────────
RESULTS_START = "<!-- AUTO_IMPUTATION_RESULTS_START -->"
RESULTS_END = "<!-- AUTO_IMPUTATION_RESULTS_END -->"


def _fmt(v: Optional[float], spec: str) -> str:
    return "-" if v is None else format(v, spec)


def update_md() -> None:
    rows = [
        "| Dataset | Model | mask_rate | MSE | MAE | Runtime (s) | Status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for ds in DATASETS:
        for mask_rate in MASK_RATES:
            for model in MODELS:
                mse, mae, rt, status = parse_log(log_path_for(model, ds, mask_rate))
                rows.append(
                    f"| {ds.name} | {model} | {mask_rate} | {_fmt(mse, '.6f')} | "
                    f"{_fmt(mae, '.6f')} | {_fmt(rt, '.1f')} | {status} |"
                )

    text = MD_PATH.read_text(encoding="utf-8")
    si, ei = text.find(RESULTS_START), text.find(RESULTS_END)
    if si < 0 or ei < 0 or ei <= si:
        raise RuntimeError(f"Markers not found in {MD_PATH.name}")
    text = text[: si + len(RESULTS_START)] + "\n" + "\n".join(rows) + "\n" + text[ei:]
    MD_PATH.write_text(text, encoding="utf-8")


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Run only the first config for 1 epoch to validate the harness.")
    args = parser.parse_args()

    if args.smoke:
        run_one(MODELS[0], DATASETS[0], MASK_RATES[0], train_epochs=1)
        update_md()
        print("\n[SMOKE] harness OK; results_comparison_imputation.md updated.", flush=True)
        return

    total = len(DATASETS) * len(MASK_RATES) * len(RUN_MODELS)
    done = 0
    for ds in DATASETS:
        for mask_rate in MASK_RATES:
            for model in RUN_MODELS:
                done += 1
                mse, _, _, status = parse_log(log_path_for(model, ds, mask_rate))
                if status == "OK":
                    print(f"[SKIP {done}/{total}] {model} {ds.name} mask={mask_rate} "
                          f"(done, MSE={mse:.6f})", flush=True)
                    continue
                print(f"\n>>> [{done}/{total}] {model} {ds.name} mask_rate={mask_rate} <<<", flush=True)
                run_one(model, ds, mask_rate, TRAIN_EPOCHS)
                update_md()
                print("[MD] results_comparison_imputation.md updated.", flush=True)

    update_md()
    print("\n[DONE] Imputation benchmark complete.", flush=True)


if __name__ == "__main__":
    main()
