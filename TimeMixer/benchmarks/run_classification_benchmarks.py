"""
Classification benchmark on the 10 standard UEA datasets — SparseTSF.

Matches the paper's setup (TimeMixer++ ICLR'25, §4.1.7): 10 multivariate UEA archives, metric =
**accuracy**. seq_len / enc_in / num_class are set automatically by Exp_Classification from the
data, so they are not passed here.

SparseTSF's classification head is a conv-aggregation -> flatten -> linear projection (see
models/SparseTSF.py). It imposes no period-divisibility requirement, so `period_len` is irrelevant
for this task; a nominal value is passed only to satisfy the arg parser.

Idempotent: a dataset whose log already has an accuracy is skipped.

    venv/Scripts/python.exe benchmarks/run_classification_benchmarks.py
"""

import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT.parent / "dataset"
LOGS_DIR = REPO_ROOT / "logs" / "run_logs_classification"
MD_PATH = REPO_ROOT / "reports" / "results_comparison_classification.md"

MODEL = "SparseTSF"
DATASETS = [
    "EthanolConcentration", "FaceDetection", "Handwriting", "Heartbeat", "JapaneseVowels",
    "PEMS-SF", "SelfRegulationSCP1", "SelfRegulationSCP2", "SpokenArabicDigits",
    "UWaveGestureLibrary",
]

LEARNING_RATE = 0.001
TRAIN_EPOCHS = 30
PATIENCE = 10
BATCH_SIZE = 16
DES = "Cls"

# Published reference: average accuracy over the same 10 UEA datasets (paper §4.1.7 / Fig. 3).
PAPER_REF = {"TimeMixer++ (paper)": 75.9, "TimesNet (paper)": 73.6}


def log_path_for(ds: str) -> Path:
    return LOGS_DIR / f"{MODEL}_{ds}.log"


ACC_PAT = re.compile(r"accuracy:([0-9]+(?:\.[0-9]+)?)")


def parse_log(path: Path) -> Tuple[Optional[float], Optional[float], str]:
    """-> (accuracy_pct, runtime_s, status)"""
    if not path.exists():
        return None, None, "PENDING"
    text = path.read_text(encoding="utf-8", errors="ignore")
    m = ACC_PAT.findall(text)
    if not m:
        return None, None, "RUNNING/INCOMPLETE"
    ep = re.findall(r"Epoch:\s+\d+\s+cost time:\s+([0-9]+(?:\.[0-9]+)?)", text)
    return float(m[-1]) * 100.0, (sum(float(x) for x in ep) if ep else None), "OK"


def run_one(ds: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u", str(REPO_ROOT / "run.py"),
        "--task_name", "classification",
        "--is_training", "1",
        "--model_id", f"CLS_{ds}",
        "--model", MODEL,
        "--data", "UEA",
        "--root_path", str(DATA_ROOT / ds),
        "--features", "M",
        # seq_len/enc_in/num_class are overwritten by Exp_Classification from the dataset;
        # period_len is unused by the classification head but must satisfy the parser.
        "--seq_len", "96", "--pred_len", "0", "--enc_in", "1", "--period_len", "1",
        "--model_type", "linear",
        "--learning_rate", str(LEARNING_RATE),
        "--train_epochs", str(TRAIN_EPOCHS),
        "--patience", str(PATIENCE),
        "--batch_size", str(BATCH_SIZE),
        "--num_workers", "0",
        "--des", DES,
        "--itr", "1",
    ]
    print(f"\n[RUN] {MODEL} {ds}", flush=True)
    start = time.perf_counter()
    with log_path_for(ds).open("w", encoding="utf-8") as f:
        subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
    print(f"      finished in {time.perf_counter() - start:.1f}s", flush=True)


RESULTS_START = "<!-- AUTO_CLASSIFICATION_RESULTS_START -->"
RESULTS_END = "<!-- AUTO_CLASSIFICATION_RESULTS_END -->"


def _fmt(v: Optional[float], spec: str = ".2f") -> str:
    return "-" if v is None else format(v, spec)


def update_md() -> None:
    rows = ["| Dataset | Accuracy (%) | Runtime (s) | Status |", "|---|---:|---:|---|"]
    accs: List[float] = []
    for ds in DATASETS:
        acc, rt, status = parse_log(log_path_for(ds))
        rows.append(f"| {ds} | {_fmt(acc)} | {_fmt(rt, '.1f')} | {status} |")
        if status == "OK":
            accs.append(acc)

    summary = ["", "**Average accuracy**", "", "| Model | Avg accuracy (%) | Datasets |",
               "|---|---:|---:|"]
    if accs:
        summary.append(f"| **SparseTSF (ours)** | {sum(accs)/len(accs):.2f} | {len(accs)}/{len(DATASETS)} |")
    else:
        summary.append("| **SparseTSF (ours)** | - | 0 |")
    for name, v in PAPER_REF.items():
        summary.append(f"| _{name}_ | _{v:.1f}_ | _10_ |")

    body = "\n".join(rows + summary) + "\n"
    text = MD_PATH.read_text(encoding="utf-8")
    si, ei = text.find(RESULTS_START), text.find(RESULTS_END)
    if si < 0 or ei < 0 or ei <= si:
        raise RuntimeError(f"Markers not found in {MD_PATH.name}")
    MD_PATH.write_text(text[: si + len(RESULTS_START)] + "\n" + body + text[ei:], encoding="utf-8")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    for i, ds in enumerate(DATASETS, 1):
        acc, _, status = parse_log(log_path_for(ds))
        if status == "OK":
            print(f"[SKIP {i}/{len(DATASETS)}] {ds} (done, acc={acc:.2f}%)", flush=True)
            continue
        print(f"\n>>> [{i}/{len(DATASETS)}] {ds} <<<", flush=True)
        run_one(ds)
        update_md()
    update_md()
    print("\n[DONE] Classification benchmark complete.", flush=True)


if __name__ == "__main__":
    main()
