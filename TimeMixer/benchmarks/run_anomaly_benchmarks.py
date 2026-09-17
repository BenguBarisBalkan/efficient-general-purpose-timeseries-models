"""
Anomaly detection benchmark — SparseTSF.

Paper setup (TimeMixer++ ICLR'25, §4.1.7): SMD / MSL / SMAP / SWaT / PSM, sliding window 100,
metric = **point-adjusted F1** (the exp loop applies `utils.tools.adjustment` before scoring).

Only datasets actually present under dataset/ are run; the rest are reported as NOT AVAILABLE:
  • PSM  — fetched by benchmarks/tools/fetch_task_datasets.py (open)
  • SMD  — assembled from the OmniAnomaly per-machine files by that same script
  • MSL / SMAP — need the TimesNet/TSLib preprocessed .npy bundle (the telemanom S3 source is dead)
  • SWaT — gated behind the iTrust request form

SparseTSF reconstructs the window with its cross-period head, so `period_len` must divide
`seq_len=100` (25 is used: 100 = 4 x 25).

Idempotent: a dataset whose log already has an F-score is skipped.

    venv/Scripts/python.exe benchmarks/run_anomaly_benchmarks.py
"""

import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT.parent / "dataset"
LOGS_DIR = REPO_ROOT / "logs" / "run_logs_anomaly"
MD_PATH = REPO_ROOT / "reports" / "results_comparison_anomaly.md"

MODEL = "SparseTSF"
SEQ_LEN = 100
PERIOD_LEN = 25          # must divide SEQ_LEN
LEARNING_RATE = 0.0001
TRAIN_EPOCHS = 3
BATCH_SIZE = 128
DES = "Anom"

# Published reference F1 (TimeMixer++ paper, Table 12 / Fig. 3).
PAPER_REF_F1 = {"SMD": 86.50, "MSL": 85.82, "SMAP": 73.10, "SWaT": 94.64, "PSM": 97.60}
PAPER_AVG = {"TimeMixer++ (paper)": 87.47, "TimesNet (paper)": 84.88}


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    folder: str
    enc_in: int
    anomaly_ratio: float
    # files that must exist for the loader to work
    required: Tuple[str, ...]


DATASETS: List[DatasetConfig] = [
    DatasetConfig("PSM", "PSM", 25, 1.0, ("train.csv", "test.csv", "test_label.csv")),
    DatasetConfig("SMD", "SMD", 38, 0.5, ("SMD_train.npy", "SMD_test.npy", "SMD_test_label.npy")),
    DatasetConfig("MSL", "MSL", 55, 1.0, ("MSL_train.npy", "MSL_test.npy", "MSL_test_label.npy")),
    DatasetConfig("SMAP", "SMAP", 25, 1.0, ("SMAP_train.npy", "SMAP_test.npy", "SMAP_test_label.npy")),
    DatasetConfig("SWaT", "SWaT", 51, 1.0, ("swat_train2.csv", "swat2.csv")),
]

F1_PAT = re.compile(r"F-score\s*:\s*([0-9]+(?:\.[0-9]+)?)")
PREC_PAT = re.compile(r"Precision\s*:\s*([0-9]+(?:\.[0-9]+)?)")
REC_PAT = re.compile(r"Recall\s*:\s*([0-9]+(?:\.[0-9]+)?)")


def available(ds: DatasetConfig) -> bool:
    d = DATA_ROOT / ds.folder
    return all((d / f).exists() for f in ds.required)


def log_path_for(ds: DatasetConfig) -> Path:
    return LOGS_DIR / f"{MODEL}_{ds.name}.log"


def parse_log(ds: DatasetConfig) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    """-> (precision%, recall%, f1%, status)"""
    path = log_path_for(ds)
    if not available(ds):
        return None, None, None, "NOT AVAILABLE"
    if not path.exists():
        return None, None, None, "PENDING"
    text = path.read_text(encoding="utf-8", errors="ignore")
    f1 = F1_PAT.findall(text)
    if not f1:
        return None, None, None, "RUNNING/INCOMPLETE"
    p = PREC_PAT.findall(text)
    r = REC_PAT.findall(text)
    return (float(p[-1]) * 100 if p else None,
            float(r[-1]) * 100 if r else None,
            float(f1[-1]) * 100, "OK")


def run_one(ds: DatasetConfig) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u", str(REPO_ROOT / "run.py"),
        "--task_name", "anomaly_detection",
        "--is_training", "1",
        "--model_id", f"ANOM_{ds.name}",
        "--model", MODEL,
        "--data", ds.name,
        "--root_path", str(DATA_ROOT / ds.folder),
        "--features", "M",
        "--seq_len", str(SEQ_LEN),
        "--pred_len", "0",
        "--enc_in", str(ds.enc_in), "--dec_in", str(ds.enc_in), "--c_out", str(ds.enc_in),
        "--period_len", str(PERIOD_LEN),
        "--model_type", "linear",
        "--anomaly_ratio", str(ds.anomaly_ratio),
        "--learning_rate", str(LEARNING_RATE),
        "--train_epochs", str(TRAIN_EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--num_workers", "0",
        "--des", DES,
        "--itr", "1",
    ]
    print(f"\n[RUN] {MODEL} {ds.name} (enc_in={ds.enc_in}, ratio={ds.anomaly_ratio})", flush=True)
    start = time.perf_counter()
    with log_path_for(ds).open("w", encoding="utf-8") as f:
        subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
    print(f"      finished in {time.perf_counter() - start:.1f}s", flush=True)


RESULTS_START = "<!-- AUTO_ANOMALY_RESULTS_START -->"
RESULTS_END = "<!-- AUTO_ANOMALY_RESULTS_END -->"


def _fmt(v: Optional[float], spec: str = ".2f") -> str:
    return "-" if v is None else format(v, spec)


def update_md() -> None:
    rows = ["| Dataset | Precision (%) | Recall (%) | F1 (%) | TimeMixer++ F1 (paper) | Status |",
            "|---|---:|---:|---:|---:|---|"]
    f1s: List[float] = []
    for ds in DATASETS:
        p, r, f1, status = parse_log(ds)
        rows.append(f"| {ds.name} | {_fmt(p)} | {_fmt(r)} | {_fmt(f1)} | "
                    f"_{PAPER_REF_F1[ds.name]:.2f}_ | {status} |")
        if status == "OK":
            f1s.append(f1)

    summary = ["", "**Average F1 over the datasets that ran**", "",
               "| Model | Avg F1 (%) | Datasets |", "|---|---:|---:|"]
    if f1s:
        summary.append(f"| **SparseTSF (ours)** | {sum(f1s)/len(f1s):.2f} | {len(f1s)}/{len(DATASETS)} |")
    else:
        summary.append("| **SparseTSF (ours)** | - | 0 |")
    for name, v in PAPER_AVG.items():
        summary.append(f"| _{name}_ | _{v:.2f}_ | _5_ |")

    body = "\n".join(rows + summary) + "\n"
    text = MD_PATH.read_text(encoding="utf-8")
    si, ei = text.find(RESULTS_START), text.find(RESULTS_END)
    if si < 0 or ei < 0 or ei <= si:
        raise RuntimeError(f"Markers not found in {MD_PATH.name}")
    MD_PATH.write_text(text[: si + len(RESULTS_START)] + "\n" + body + text[ei:], encoding="utf-8")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    for i, ds in enumerate(DATASETS, 1):
        p, r, f1, status = parse_log(ds)
        if status == "NOT AVAILABLE":
            print(f"[MISS {i}/{len(DATASETS)}] {ds.name} — data not present, skipping", flush=True)
            continue
        if status == "OK":
            print(f"[SKIP {i}/{len(DATASETS)}] {ds.name} (done, F1={f1:.2f}%)", flush=True)
            continue
        print(f"\n>>> [{i}/{len(DATASETS)}] {ds.name} <<<", flush=True)
        run_one(ds)
        update_md()
    update_md()
    print("\n[DONE] Anomaly benchmark complete.", flush=True)


if __name__ == "__main__":
    main()
