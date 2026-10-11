"""SparseTSFPlus anomaly ablation: does leave-one-out reconstruction fix SMD's low recall?

SparseTSF's reconstruction head can represent the identity exactly (R = I with a zero
convolution), so training on mostly normal data drifts toward copying the input, and a model
that copies its input reconstructs anomalies faithfully too. That predicts high precision and
low recall, which is SparseTSF's published profile on SMD (precision 87.25, recall 73.90).

`--stsf_diag_mask 1` zeroes the diagonal of the reconstruction map, so each point is rebuilt
only from the same phase in the OTHER periods and can never see its own value.

Training settings are copied from run_anomaly_benchmarks.py, so `base` must reproduce the
published SparseTSF row. The anomaly score and threshold in exp_anomaly_detection.py are
shared with TimeMixer and TimeMixer++ and are deliberately left unchanged.

CAVEAT: exp_anomaly_detection.py early-stops on TEST loss (upstream TSLib behaviour, shared by
every model here). With two variants there is no selection to contaminate, but read the F1
difference as a sensitivity result.

    venv/Scripts/python.exe benchmarks/run_sparsetsf_plus_anomaly.py
"""

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT.parent / "dataset"
LOGS_DIR = REPO_ROOT / "logs" / "run_logs_plus_anom"
MD_PATH = REPO_ROOT / "reports" / "results_sparsetsf_plus_anomaly.md"

MODEL = "SparseTSFPlus"
SEQ_LEN = 100
PERIOD_LEN = 25
LEARNING_RATE = 0.0001
TRAIN_EPOCHS = 3
BATCH_SIZE = 128


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    enc_in: int
    anomaly_ratio: float


DATASETS: List[DatasetConfig] = [
    DatasetConfig("PSM", 25, 1.0),
    DatasetConfig("SMD", 38, 0.5),
]


@dataclass(frozen=True)
class Variant:
    name: str
    flags: Tuple[str, ...]
    note: str


VARIANTS: List[Variant] = [
    Variant("base", (), "all units off == SparseTSF"),
    Variant("loo", ("--stsf_diag_mask", "1"), "leave-one-out reconstruction"),
]

# Published SparseTSF (precision, recall, F1) and TimeMixer++ paper F1.
PUBLISHED: Dict[str, Tuple[float, float, float]] = {
    "PSM": (98.82, 91.96, 95.26),
    "SMD": (87.25, 73.90, 80.02),
}
TMPP_F1 = {"PSM": 97.60, "SMD": 86.50}

PAT = re.compile(r"Precision : ([0-9.]+), Recall : ([0-9.]+), F-score : ([0-9.]+)")
PARAM_PAT = re.compile(r"trainable_params:([0-9]+)")


def log_path_for(ds: DatasetConfig, v: Variant) -> Path:
    return LOGS_DIR / "{}_{}.log".format(ds.name, v.name)


def parse_log(path: Path):
    out = {"p": None, "r": None, "f": None, "params": None, "status": "PENDING"}
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8", errors="ignore")
    m = PAT.findall(text)
    q = PARAM_PAT.findall(text)
    out["params"] = int(q[0]) if q else None
    if not m:
        out["status"] = "RUNNING/INCOMPLETE"
        return out
    p, r, f = m[-1]
    out.update(p=float(p) * 100, r=float(r) * 100, f=float(f) * 100, status="OK")
    return out


def run_one(ds: DatasetConfig, v: Variant) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u", str(REPO_ROOT / "run.py"),
        "--task_name", "anomaly_detection",
        "--is_training", "1",
        "--model_id", "ANOMP_{}_{}".format(ds.name, v.name),
        "--model", MODEL,
        "--data", ds.name,
        "--root_path", str(DATA_ROOT / ds.name),
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
        "--des", "AnomP",
        "--itr", "1",
    ] + list(v.flags)
    print("[RUN] {} {}".format(ds.name, v.name), flush=True)
    t0 = time.time()
    with log_path_for(ds, v).open("w", encoding="utf-8") as f:
        subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
    print("      done in {:.0f}s".format(time.time() - t0), flush=True)


START = "<!-- AUTO_PLUS_ANOM_START -->"
END = "<!-- AUTO_PLUS_ANOM_END -->"

HEADER = """# SparseTSFPlus ablation — anomaly detection

Tests whether leave-one-out reconstruction fixes SparseTSF's recall-limited profile. `base`
is all units off and must reproduce the published SparseTSF row. Point-adjusted F1, as in the
TimeMixer++ paper. The scoring and threshold are unchanged and shared with every model.

```bash
venv/Scripts/python.exe benchmarks/run_sparsetsf_plus_anomaly.py
```
"""


def _fmt(v):
    return "-" if v is None else "{:.2f}".format(v)


def update_md() -> None:
    lines = ["| Dataset | Variant | Params | Precision | Recall | F1 | F1 vs SparseTSF (published) | Status |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    for ds in DATASETS:
        for v in VARIANTS:
            r = parse_log(log_path_for(ds, v))
            pub = PUBLISHED[ds.name]
            d = "-" if r["f"] is None else "{:+.2f}".format(r["f"] - pub[2])
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                ds.name, v.name, "-" if r["params"] is None else "{:,}".format(r["params"]),
                _fmt(r["p"]), _fmt(r["r"]), _fmt(r["f"]), d, r["status"]))
        pub = PUBLISHED[ds.name]
        lines.append("| _{} published_ | _SparseTSF_ | _—_ | _{:.2f}_ | _{:.2f}_ | _{:.2f}_ | _—_ | _reference_ |".format(
            ds.name, pub[0], pub[1], pub[2]))
        lines.append("| _{} published_ | _TimeMixer++_ | _—_ | _—_ | _—_ | _{:.2f}_ | _—_ | _reference_ |".format(
            ds.name, TMPP_F1[ds.name]))
    body = "\n".join(lines)
    MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MD_PATH.exists():
        MD_PATH.write_text(HEADER + "\n" + START + "\n" + END + "\n", encoding="utf-8")
    text = MD_PATH.read_text(encoding="utf-8")
    si, ei = text.find(START), text.find(END)
    if si < 0 or ei < 0 or ei <= si:
        raise RuntimeError("Markers not found in " + MD_PATH.name)
    MD_PATH.write_text(text[: si + len(START)] + "\n" + body + "\n" + text[ei:], encoding="utf-8")
    print("[OK] wrote " + str(MD_PATH.relative_to(REPO_ROOT)), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-from-logs-only", action="store_true")
    args = ap.parse_args()
    if args.update_from_logs_only:
        update_md()
        return
    jobs = [(ds, v) for ds in DATASETS for v in VARIANTS]
    for i, (ds, v) in enumerate(jobs, 1):
        if parse_log(log_path_for(ds, v))["status"] == "OK":
            print("[SKIP {}/{}] {} {}".format(i, len(jobs), ds.name, v.name), flush=True)
            continue
        print("[{}/{}]".format(i, len(jobs)), end=" ", flush=True)
        run_one(ds, v)
        update_md()
    update_md()
    print("[DONE] anomaly ablation complete", flush=True)


if __name__ == "__main__":
    main()
