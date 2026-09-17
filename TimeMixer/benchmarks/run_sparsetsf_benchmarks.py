"""
Accuracy benchmark for SparseTSF (ICML 2024) on the light datasets.

Uses SparseTSF's OWN published hyperparameters (lr=0.02, bs=256, 30 epochs, patience=5,
lradj=type3, model_type=linear) — the right choice for an accuracy table, matching how
results_comparison.md used TimeMixer's own published config.

Two lookbacks are swept:
  seq_len=96  -> directly comparable to our TimeMixer / TimeMixer++ tables
  seq_len=720 -> SparseTSF's published setting (it needs several periods of context)

Matrix: 5 datasets x 4 pred_lens x 2 lookbacks = 40 runs. Idempotent/resumable.
"""

import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# This script lives in TimeMixer/benchmarks/, so REPO_ROOT is one level up.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT.parent / "dataset"
LOGS_DIR = REPO_ROOT / "logs" / "run_logs_sparsetsf"
MD_PATH = REPO_ROOT / "reports" / "results_comparison_sparsetsf.md"

# Existing tables we cross-reference for the 3-way comparison
V1_MD = REPO_ROOT / "reports" / "results_comparison.md"
PP_MD = REPO_ROOT / "reports" / "results_comparison_pp.md"

PRED_LENS = [96, 192, 336, 720]
SEQ_LENS = [96, 720]

# SparseTSF's published training setup
LEARNING_RATE = 0.02
BATCH_SIZE = 256
TRAIN_EPOCHS = 30
PATIENCE = 5
LRADJ = "type3"
MODEL_TYPE = "linear"   # the <1k-parameter headline variant
D_MODEL = 128           # only used by the 'mlp' variant
DES = "Sparse"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    data: str
    data_path: str
    root_path: Path
    enc_in: int
    period_len: int   # must divide BOTH seq_len and pred_len
    factor: int


# period_len values are upstream's per-dataset choices (scripts/SparseTSF/linear/*.sh):
# hourly ETTh -> 24; 15-min ETTm and 10-min Weather -> 4 (their "2-6 for large periods" rule)
DATASETS: List[DatasetConfig] = [
    DatasetConfig("ETTh1", "ETTh1", "ETTh1.csv", DATA_ROOT / "ETT-small", 7, 24, 1),
    DatasetConfig("ETTh2", "ETTh2", "ETTh2.csv", DATA_ROOT / "ETT-small", 7, 24, 1),
    DatasetConfig("ETTm1", "ETTm1", "ETTm1.csv", DATA_ROOT / "ETT-small", 7, 4, 1),
    DatasetConfig("ETTm2", "ETTm2", "ETTm2.csv", DATA_ROOT / "ETT-small", 7, 4, 1),
    DatasetConfig("Weather", "custom", "weather.csv", DATA_ROOT / "weather", 21, 4, 3),
]


# ── Log parsing ──────────────────────────────────────────────────────────────

MSE_PAT = re.compile(r"mse:([0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?)")
MAE_PAT = re.compile(r"mae:([0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?)")
EPOCH_PAT = re.compile(r"Epoch:\s+\d+\s+cost time:\s+([0-9]+(?:\.[0-9]+)?)")


def _last(pat: re.Pattern, text: str) -> Optional[float]:
    m = pat.findall(text)
    return float(m[-1]) if m else None


def log_path_for(ds: DatasetConfig, seq_len: int, pred_len: int) -> Path:
    return LOGS_DIR / f"{ds.name}_sl{seq_len}_pred{pred_len}.log"


def parse_log(path: Path) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    if not path.exists():
        return None, None, None, "PENDING"
    text = path.read_text(encoding="utf-8", errors="ignore")
    mse, mae = _last(MSE_PAT, text), _last(MAE_PAT, text)
    if mse is None or mae is None:
        return None, None, None, "RUNNING/INCOMPLETE"
    epochs = [float(x) for x in EPOCH_PAT.findall(text)]
    return mse, mae, (sum(epochs) if epochs else None), "OK"


def parse_existing_table(md: Path) -> Dict[Tuple[str, int], float]:
    """Pull {(dataset, pred_len): local_mse} out of one of the existing result tables."""
    out: Dict[Tuple[str, int], float] = {}
    if not md.exists():
        return out
    for line in md.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")]
        # | Dataset | pred_len | Paper MSE | Paper MAE | Local MSE | ...
        if len(cells) < 6:
            continue
        try:
            out[(cells[1], int(cells[2]))] = float(cells[5])
        except ValueError:
            continue
    return out


# ── Runner ───────────────────────────────────────────────────────────────────

def run_one(ds: DatasetConfig, seq_len: int, pred_len: int) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = log_path_for(ds, seq_len, pred_len)
    model_id = f"SPARSE_{ds.name}_{seq_len}_{pred_len}"

    cmd = [
        sys.executable, "-u", str(REPO_ROOT / "run.py"),
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--model_id", model_id,
        "--model", "SparseTSF",
        "--data", ds.data,
        "--root_path", str(ds.root_path),
        "--data_path", ds.data_path,
        "--features", "M",
        "--seq_len", str(seq_len),
        "--label_len", "0",
        "--pred_len", str(pred_len),
        "--enc_in", str(ds.enc_in),
        "--dec_in", str(ds.enc_in),
        "--c_out", str(ds.enc_in),
        "--period_len", str(ds.period_len),
        "--model_type", MODEL_TYPE,
        "--d_model", str(D_MODEL),
        "--factor", str(ds.factor),
        "--learning_rate", str(LEARNING_RATE),
        "--lradj", LRADJ,
        "--train_epochs", str(TRAIN_EPOCHS),
        "--patience", str(PATIENCE),
        "--batch_size", str(BATCH_SIZE),
        "--num_workers", "0",
        "--des", DES,
        "--itr", "1",
    ]

    print(f"\n[RUN] SparseTSF {ds.name} seq_len={seq_len} pred_len={pred_len} "
          f"(period_len={ds.period_len})", flush=True)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as f:
        subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
    print(f"      finished in {time.perf_counter() - start:.1f}s", flush=True)


def validate_configs() -> None:
    """Fail fast if any config violates SparseTSF's divisibility requirement."""
    bad = []
    for ds in DATASETS:
        for seq_len in SEQ_LENS:
            for pred_len in PRED_LENS:
                if seq_len % ds.period_len or pred_len % ds.period_len:
                    bad.append(f"{ds.name} sl={seq_len} pl={pred_len} per={ds.period_len}")
    if bad:
        raise SystemExit("Invalid (period_len must divide seq_len and pred_len):\n  "
                         + "\n  ".join(bad))
    print(f"[OK] all {len(DATASETS) * len(SEQ_LENS) * len(PRED_LENS)} configs satisfy "
          f"the period_len divisibility constraint", flush=True)


# ── Report ───────────────────────────────────────────────────────────────────

RESULTS_START = "<!-- AUTO_SPARSETSF_RESULTS_START -->"
RESULTS_END = "<!-- AUTO_SPARSETSF_RESULTS_END -->"
COMPARE_START = "<!-- AUTO_SPARSETSF_3WAY_START -->"
COMPARE_END = "<!-- AUTO_SPARSETSF_3WAY_END -->"


def _fmt(v: Optional[float], spec: str) -> str:
    return "-" if v is None else format(v, spec)


def _replace_block(text: str, start: str, end: str, body: str) -> str:
    si, ei = text.find(start), text.find(end)
    if si < 0 or ei < 0 or ei <= si:
        raise RuntimeError(f"Markers {start}/{end} not found in {MD_PATH.name}")
    return text[: si + len(start)] + "\n" + body + text[ei:]


def update_md() -> None:
    # ── per-lookback result tables ──────────────────────────────────────────
    blocks: List[str] = []
    for seq_len in SEQ_LENS:
        label = ("comparable to our TimeMixer / TimeMixer++ tables"
                 if seq_len == 96 else "SparseTSF's published setting")
        blocks.append(f"#### seq_len = {seq_len} ({label})\n")
        blocks.append("| Dataset | period_len | pred_len | MSE | MAE | Runtime (s) | Status |")
        blocks.append("|---|---:|---:|---:|---:|---:|---|")
        for ds in DATASETS:
            for pred_len in PRED_LENS:
                mse, mae, rt, status = parse_log(log_path_for(ds, seq_len, pred_len))
                blocks.append(
                    f"| {ds.name} | {ds.period_len} | {pred_len} | {_fmt(mse, '.6f')} | "
                    f"{_fmt(mae, '.6f')} | {_fmt(rt, '.1f')} | {status} |"
                )
        blocks.append("")

    # ── 3-way comparison at seq_len=96 ──────────────────────────────────────
    v1 = parse_existing_table(V1_MD)
    pp = parse_existing_table(PP_MD)
    cmp_rows = [
        "| Dataset | pred_len | TimeMixer MSE | TimeMixer++ MSE | SparseTSF MSE | Best |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for ds in DATASETS:
        for pred_len in PRED_LENS:
            s_mse, _, _, _ = parse_log(log_path_for(ds, 96, pred_len))
            a, b = v1.get((ds.name, pred_len)), pp.get((ds.name, pred_len))
            cands = {"TimeMixer": a, "TimeMixer++": b, "SparseTSF": s_mse}
            avail = {k: v for k, v in cands.items() if v is not None}
            best = min(avail, key=avail.get) if avail else "-"
            cmp_rows.append(
                f"| {ds.name} | {pred_len} | {_fmt(a, '.6f')} | {_fmt(b, '.6f')} | "
                f"{_fmt(s_mse, '.6f')} | {best} |"
            )

    text = MD_PATH.read_text(encoding="utf-8")
    text = _replace_block(text, RESULTS_START, RESULTS_END, "\n".join(blocks) + "\n")
    text = _replace_block(text, COMPARE_START, COMPARE_END, "\n".join(cmp_rows) + "\n")
    MD_PATH.write_text(text, encoding="utf-8")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    validate_configs()

    total = len(DATASETS) * len(SEQ_LENS) * len(PRED_LENS)
    done = 0
    for seq_len in SEQ_LENS:
        for ds in DATASETS:
            for pred_len in PRED_LENS:
                done += 1
                mse, _, _, status = parse_log(log_path_for(ds, seq_len, pred_len))
                if status == "OK":
                    print(f"[SKIP {done}/{total}] {ds.name} sl={seq_len} pred={pred_len} "
                          f"(done, MSE={mse:.6f})", flush=True)
                    continue
                print(f"\n>>> [{done}/{total}] SparseTSF {ds.name} sl={seq_len} "
                      f"pred_len={pred_len} <<<", flush=True)
                run_one(ds, seq_len, pred_len)
                update_md()
                print("[MD] results_comparison_sparsetsf.md updated.", flush=True)

    update_md()
    print("\n[DONE] SparseTSF benchmark complete.", flush=True)


if __name__ == "__main__":
    main()
