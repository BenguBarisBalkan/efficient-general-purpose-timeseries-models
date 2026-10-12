"""
Few-shot forecasting benchmark (SparseTSF and TimeMixer): train on 10% of the training timesteps.

Matches the paper's few-shot setup (TimeMixer++ ICLR'25, Table 5): ETT x4 + Weather + Electricity,
seq_len=96, pred_len {96,192,336,720}, MSE/MAE — but each model sees only the first 10% of its
training split (`run.py --percent 10`). Validation and test splits are untouched, so numbers stay
comparable to the full-data forecasting tables.

TimeMixer uses its own long-term configuration per dataset (tm_config.py) and the same few-shot
protocol as SparseTSF: batch size 32 (see below) and 10% of the training split.

Idempotent: a config whose log already has a valid MSE/MAE is skipped.

    venv/Scripts/python.exe benchmarks/run_fewshot_benchmarks.py
"""

import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tm_config

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT.parent / "dataset"
LOGS_DIR = REPO_ROOT / "logs" / "run_logs_fewshot"
MD_PATH = REPO_ROOT / "reports" / "results_comparison_fewshot.md"

MODELS = ["SparseTSF", "TimeMixer"]   # both run locally
PERCENT = 10
SEQ_LEN = 96
PRED_LENS = [96, 192, 336, 720]
LEARNING_RATE = 0.02
# SparseTSF's published batch is 256, but that is unusable here: with only 10% of the training
# split AND drop_last=True, the long-horizon ETTh configs yield ~135 windows -> 0 batches -> the
# run dies with "Expected positive integer steps_per_epoch, but got 0". 32 keeps every config
# non-empty. Applied to ALL configs so the table stays one consistent protocol.
BATCH_SIZE = 32
TRAIN_EPOCHS = 30

# Electricity @ pred_len 720: not the model but the EVALUATION blows up — test() materialises
# 4541 x 720 x 321 float32 (~4.2 GB) for preds and trues, ~12.6 GB on a 16 GB machine. These configs
# use --stream_metrics 1, which accumulates the same metrics batch by batch.
STREAM = {("Electricity", 720)}
PATIENCE = 5
LRADJ = "type3"
DES = "FewShot"

# Published reference (TimeMixer++ paper, Table 5; avg over the 4 pred_lens). MSE/MAE.
PAPER_REF = {
    "ETT (Avg)":   {"TimeMixer (paper)": (0.453, 0.445), "TimeMixer++ (paper)": (0.396, 0.421)},
    "Weather":     {"TimeMixer (paper)": (0.242, 0.281), "TimeMixer++ (paper)": (0.241, 0.271)},
    "Electricity": {"TimeMixer (paper)": (0.187, 0.277), "TimeMixer++ (paper)": (0.168, 0.271)},
}


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    data: str
    data_path: str
    root_path: Path
    enc_in: int
    period_len: int


DATASETS: List[DatasetConfig] = [
    DatasetConfig("ETTh1", "ETTh1", "ETTh1.csv", DATA_ROOT / "ETT-small", 7, 24),
    DatasetConfig("ETTh2", "ETTh2", "ETTh2.csv", DATA_ROOT / "ETT-small", 7, 24),
    DatasetConfig("ETTm1", "ETTm1", "ETTm1.csv", DATA_ROOT / "ETT-small", 7, 4),
    DatasetConfig("ETTm2", "ETTm2", "ETTm2.csv", DATA_ROOT / "ETT-small", 7, 4),
    DatasetConfig("Weather", "custom", "weather.csv", DATA_ROOT / "weather", 21, 4),
    DatasetConfig("Electricity", "custom", "electricity.csv", DATA_ROOT / "electricity", 321, 24),
]

NUM = r"([0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?)"
MSE_PAT = re.compile(r"mse:" + NUM)
MAE_PAT = re.compile(r"mae:" + NUM)
EPOCH_PAT = re.compile(r"Epoch:\s+\d+\s+cost time:\s+([0-9]+(?:\.[0-9]+)?)")


def _last(pat: re.Pattern, text: str) -> Optional[float]:
    m = pat.findall(text)
    return float(m[-1]) if m else None


def log_path_for(ds: DatasetConfig, pred_len: int, model: str = "SparseTSF") -> Path:
    return LOGS_DIR / f"{model}_{ds.name}_pred{pred_len}.log"


def parse_log(path: Path) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    if not path.exists():
        return None, None, None, "PENDING"
    text = path.read_text(encoding="utf-8", errors="ignore")
    mse, mae = _last(MSE_PAT, text), _last(MAE_PAT, text)
    if mse is None or mae is None:
        return None, None, None, "RUNNING/INCOMPLETE"
    epochs = [float(x) for x in EPOCH_PAT.findall(text)]
    return mse, mae, (sum(epochs) if epochs else None), "OK"


def _model_args(model: str, ds: DatasetConfig) -> List[str]:
    if model == "SparseTSF":
        return [
            "--period_len", str(ds.period_len),
            "--model_type", "linear",
            "--percent", str(PERCENT),          # <-- the few-shot setting
            "--learning_rate", str(LEARNING_RATE),
            "--lradj", LRADJ,
            "--train_epochs", str(TRAIN_EPOCHS),
            "--patience", str(PATIENCE),
            "--batch_size", str(BATCH_SIZE),
        ]
    t = tm_config.training(ds.name)
    return tm_config.arch(ds.name) + [
        "--percent", str(PERCENT),
        "--learning_rate", t["learning_rate"],
        "--train_epochs", t["train_epochs"],
        "--patience", t["patience"],
        "--batch_size", str(BATCH_SIZE),        # the few-shot protocol, as for SparseTSF
    ]


def run_one(ds: DatasetConfig, pred_len: int, model: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = log_path_for(ds, pred_len, model)
    prefix = "FEWSHOT" if model == "SparseTSF" else "FEWSHOT_TM"
    cmd = [
        sys.executable, "-u", str(REPO_ROOT / "run.py"),
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--model_id", f"{prefix}_{ds.name}_{SEQ_LEN}_{pred_len}",
        "--model", model,
        "--data", ds.data,
        "--root_path", str(ds.root_path),
        "--data_path", ds.data_path,
        "--features", "M",
        "--seq_len", str(SEQ_LEN),
        "--label_len", "0",
        "--pred_len", str(pred_len),
        "--enc_in", str(ds.enc_in),
        "--dec_in", str(ds.enc_in),
        "--c_out", str(ds.enc_in),
    ] + _model_args(model, ds) + [
        "--num_workers", "0",
        "--des", DES,
        "--itr", "1",
    ] + (["--stream_metrics", "1"] if (ds.name, pred_len) in STREAM else [])
    print(f"\n[RUN] {model} {ds.name} pred_len={pred_len} (percent={PERCENT})", flush=True)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as f:
        subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
    print(f"      finished in {time.perf_counter() - start:.1f}s", flush=True)


RESULTS_START = "<!-- AUTO_FEWSHOT_RESULTS_START -->"
RESULTS_END = "<!-- AUTO_FEWSHOT_RESULTS_END -->"


def _fmt(v: Optional[float], spec: str = ".6f") -> str:
    return "-" if v is None else format(v, spec)


def update_md() -> None:
    rows = ["| Dataset | pred_len | Model | MSE | MAE | Runtime (s) | Status |",
            "|---|---:|---|---:|---:|---:|---|"]
    per_ds: Dict[Tuple[str, str], List[List[float]]] = {}
    for ds in DATASETS:
        for pred_len in PRED_LENS:
            for model in MODELS:
                mse, mae, rt, status = parse_log(log_path_for(ds, pred_len, model))
                rows.append(f"| {ds.name} | {pred_len} | {model} | {_fmt(mse)} | {_fmt(mae)} | "
                            f"{_fmt(rt, '.1f')} | {status} |")
                if status == "OK":
                    per_ds.setdefault((model, ds.name), []).append([mse, mae])

    def avg(model: str, names: List[str]) -> str:
        """Mean over the group, shown only once every cell of the group is measured."""
        vals = [v for n in names for v in per_ds.get((model, n), [])]
        if len(vals) < len(names) * len(PRED_LENS):
            return "-"
        return (f"{sum(v[0] for v in vals) / len(vals):.3f} / "
                f"{sum(v[1] for v in vals) / len(vals):.3f}")

    summary = ["", "**Averages over pred_len (MSE / MAE)**", "",
               "| Group | SparseTSF (local) | TimeMixer (local) | TimeMixer (paper) | TimeMixer++ (paper) |",
               "|---|---:|---:|---:|---:|"]
    groups = {"ETT (Avg)": ["ETTh1", "ETTh2", "ETTm1", "ETTm2"],
              "Weather": ["Weather"], "Electricity": ["Electricity"]}
    for label, names in groups.items():
        ref = PAPER_REF[label]
        tm, pp = ref["TimeMixer (paper)"], ref["TimeMixer++ (paper)"]
        summary.append(f"| {label} | {avg('SparseTSF', names)} | {avg('TimeMixer', names)} | "
                       f"_{tm[0]:.3f} / {tm[1]:.3f}_ | _{pp[0]:.3f} / {pp[1]:.3f}_ |")

    body = "\n".join(rows + summary) + "\n"
    text = MD_PATH.read_text(encoding="utf-8")
    si, ei = text.find(RESULTS_START), text.find(RESULTS_END)
    if si < 0 or ei < 0 or ei <= si:
        raise RuntimeError(f"Markers not found in {MD_PATH.name}")
    MD_PATH.write_text(text[: si + len(RESULTS_START)] + "\n" + body + text[ei:], encoding="utf-8")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    # Electricity last: it is by far the slowest, so the cheap cells finish first.
    order = ([d for d in DATASETS if d.name != "Electricity"]
             + [d for d in DATASETS if d.name == "Electricity"])
    jobs = [(m, d, p) for m in MODELS for d in order for p in PRED_LENS]
    for i, (model, ds, pred_len) in enumerate(jobs, 1):
        mse, _, _, status = parse_log(log_path_for(ds, pred_len, model))
        if status == "OK":
            print(f"[SKIP {i}/{len(jobs)}] {model} {ds.name} pred={pred_len} (done, MSE={mse:.6f})",
                  flush=True)
            continue
        print(f"\n>>> [{i}/{len(jobs)}] {model} {ds.name} pred_len={pred_len} <<<", flush=True)
        run_one(ds, pred_len, model)
        update_md()
    update_md()
    print("\n[DONE] Few-shot benchmark complete.", flush=True)


if __name__ == "__main__":
    main()
