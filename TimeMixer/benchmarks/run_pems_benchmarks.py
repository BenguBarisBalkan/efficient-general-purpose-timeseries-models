"""
Multivariate short-term forecasting benchmark on PEMS (SparseTSF).

Matches the paper's PEMS setup (TimeMixer++ ICLR'25, Table 3): PEMS03/04/07/08, input 96 -> predict
12, metrics **MAE / MAPE / RMSE** averaged across the four datasets.

Runs under `--task_name long_term_forecast` (the upstream PEMS scripts do the same: the short-term
exp class is M4-specific). `--use_norm 0` follows upstream. Metrics are read from the run log:
`mae:` plus the `RMSE:` / `MAPE:` line added to exp/exp_long_term_forecasting.py — note those two are
UPPERCASE on purpose so they can't collide with the lowercase `mse:`/`mae:` patterns other runners use.

Idempotent: a config whose log already holds a valid MAE is skipped.

    venv/Scripts/python.exe benchmarks/run_pems_benchmarks.py
"""

import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT.parent / "dataset"
LOGS_DIR = REPO_ROOT / "logs" / "run_logs_pems"
MD_PATH = REPO_ROOT / "reports" / "results_comparison_pems.md"

MODELS = ["SparseTSF"]          # local runs — SparseTSF only

SEQ_LEN = 96
PRED_LEN = 12
PERIOD_LEN = 12   # must divide seq_len(96) AND pred_len(12); 12 x 5min = 1 hour
LEARNING_RATE = 0.01
TRAIN_EPOCHS = 10
PATIENCE = 5
BATCH_SIZE = 16
DES = "PEMS"

# Published reference (TimeMixer++ paper, Table 3 — averaged over the 4 PEMS datasets).
PAPER_REF = {
    "TimeMixer (paper)":   {"MAE": 17.41, "MAPE": 10.59, "RMSE": 28.01},
    "TimeMixer++ (paper)": {"MAE": 15.91, "MAPE": 10.08, "RMSE": 27.06},
}


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    data_path: str
    enc_in: int


DATASETS: List[DatasetConfig] = [
    DatasetConfig("PEMS03", "PEMS03.npz", 358),
    DatasetConfig("PEMS04", "PEMS04.npz", 307),
    DatasetConfig("PEMS07", "PEMS07.npz", 883),
    DatasetConfig("PEMS08", "PEMS08.npz", 170),
]

NUM = r"([0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?)"
MAE_PAT = re.compile(r"mae:" + NUM)
RMSE_PAT = re.compile(r"RMSE:" + NUM)
MAPE_PAT = re.compile(r"MAPE:" + NUM)
EPOCH_PAT = re.compile(r"Epoch:\s+\d+\s+cost time:\s+([0-9]+(?:\.[0-9]+)?)")


def _last(pat: re.Pattern, text: str) -> Optional[float]:
    m = pat.findall(text)
    return float(m[-1]) if m else None


def log_path_for(model: str, ds: DatasetConfig) -> Path:
    return LOGS_DIR / f"{model}_{ds.name}.log"


def parse_log(path: Path) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], str]:
    """-> (mae, mape_pct, rmse, runtime_s, status)"""
    if not path.exists():
        return None, None, None, None, "PENDING"
    text = path.read_text(encoding="utf-8", errors="ignore")
    mae, rmse, mape = _last(MAE_PAT, text), _last(RMSE_PAT, text), _last(MAPE_PAT, text)
    if mae is None or rmse is None or mape is None:
        return None, None, None, None, "RUNNING/INCOMPLETE"
    epochs = [float(x) for x in EPOCH_PAT.findall(text)]
    # utils.metrics.MAPE returns a fraction -> report as a percentage like the paper.
    return mae, mape * 100.0, rmse, (sum(epochs) if epochs else None), "OK"


def run_one(model: str, ds: DatasetConfig) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = log_path_for(model, ds)
    cmd = [
        sys.executable, "-u", str(REPO_ROOT / "run.py"),
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--model_id", f"PEMS_{model}_{ds.name}",
        "--model", model,
        "--data", "PEMS",
        "--root_path", str(DATA_ROOT / "PEMS"),
        "--data_path", ds.data_path,
        "--features", "M",
        "--seq_len", str(SEQ_LEN),
        "--label_len", "0",
        "--pred_len", str(PRED_LEN),
        "--enc_in", str(ds.enc_in),
        "--dec_in", str(ds.enc_in),
        "--c_out", str(ds.enc_in),
        "--period_len", str(PERIOD_LEN),
        "--model_type", "linear",
        "--use_norm", "0",              # follows the upstream PEMS scripts
        "--learning_rate", str(LEARNING_RATE),
        "--train_epochs", str(TRAIN_EPOCHS),
        "--patience", str(PATIENCE),
        "--batch_size", str(BATCH_SIZE),
        "--num_workers", "0",
        "--des", DES,
        "--itr", "1",
    ]
    print(f"\n[RUN] {model} {ds.name} (enc_in={ds.enc_in})", flush=True)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as f:
        subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
    print(f"      finished in {time.perf_counter() - start:.1f}s", flush=True)


RESULTS_START = "<!-- AUTO_PEMS_RESULTS_START -->"
RESULTS_END = "<!-- AUTO_PEMS_RESULTS_END -->"


def _fmt(v: Optional[float], spec: str = ".2f") -> str:
    return "-" if v is None else format(v, spec)


def update_md() -> None:
    rows = [
        "| Dataset | Model | MAE | MAPE (%) | RMSE | Runtime (s) | Status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    sums: Dict[str, List[float]] = {m: [0.0, 0.0, 0.0, 0.0] for m in MODELS}
    for ds in DATASETS:
        for model in MODELS:
            mae, mape, rmse, rt, status = parse_log(log_path_for(model, ds))
            rows.append(f"| {ds.name} | {model} | {_fmt(mae)} | {_fmt(mape)} | {_fmt(rmse)} | "
                        f"{_fmt(rt, '.1f')} | {status} |")
            if status == "OK":
                acc = sums[model]
                acc[0] += mae; acc[1] += mape; acc[2] += rmse; acc[3] += 1

    avg = ["", "**Average across PEMS datasets**", "",
           "| Model | MAE | MAPE (%) | RMSE |", "|---|---:|---:|---:|"]
    for model in MODELS:
        a = sums[model]
        n = a[3]
        if n:
            avg.append(f"| **{model} (ours)** | {a[0]/n:.2f} | {a[1]/n:.2f} | {a[2]/n:.2f} | ")
        else:
            avg.append(f"| **{model} (ours)** | - | - | - |")
    for name, r in PAPER_REF.items():
        avg.append(f"| _{name}_ | _{r['MAE']:.2f}_ | _{r['MAPE']:.2f}_ | _{r['RMSE']:.2f}_ |")

    body = "\n".join(rows + avg) + "\n"
    text = MD_PATH.read_text(encoding="utf-8")
    si, ei = text.find(RESULTS_START), text.find(RESULTS_END)
    if si < 0 or ei < 0 or ei <= si:
        raise RuntimeError(f"Markers not found in {MD_PATH.name}")
    MD_PATH.write_text(text[: si + len(RESULTS_START)] + "\n" + body + text[ei:], encoding="utf-8")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    total = len(DATASETS) * len(MODELS)
    done = 0
    for ds in DATASETS:
        for model in MODELS:
            done += 1
            mae, _, _, _, status = parse_log(log_path_for(model, ds))
            if status == "OK":
                print(f"[SKIP {done}/{total}] {model} {ds.name} (done, MAE={mae:.2f})", flush=True)
                continue
            print(f"\n>>> [{done}/{total}] {model} {ds.name} <<<", flush=True)
            run_one(model, ds)
            update_md()
            print("[MD] results_comparison_pems.md updated.", flush=True)

    update_md()
    print("\n[DONE] PEMS benchmark complete.", flush=True)


if __name__ == "__main__":
    main()
