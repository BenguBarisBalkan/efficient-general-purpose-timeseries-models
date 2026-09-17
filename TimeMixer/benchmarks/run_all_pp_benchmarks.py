"""
Runner for TimeMixer++ long-term forecasting benchmarks.

Paper: TimeMixer++: A General Time Series Pattern Machine (ICLR 2025)
Local hardware: RTX 3050 Ti Laptop GPU (4 GB VRAM).

Differences from the v1 runner
  • model = TimeMixerPP
  • logs written to .run_logs_pp/
  • MD target is results_comparison_pp.md
  • Batch sizes reduced from paper's 512 to fit 4 GB VRAM
  • n_heads, top_k added as per TimeMixer++ design
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
MD_PATH  = REPO_ROOT / "reports" / "results_comparison_pp.md"
DATA_ROOT = REPO_ROOT.parent / "dataset"
LOGS_DIR  = REPO_ROOT / "logs" / "run_logs_pp"

PRED_LENS = [96, 192, 336, 720]

# Paper results — Table 16 (Appendix H), full per-pred-len, TimeMixer++ (Ours).
PAPER: Dict[str, Dict[int, Tuple[float, float]]] = {
    "ETTh1": {
        96:  (0.361, 0.403),
        192: (0.416, 0.441),
        336: (0.430, 0.434),
        720: (0.467, 0.451),
    },
    "ETTh2": {
        96:  (0.276, 0.328),
        192: (0.342, 0.379),
        336: (0.346, 0.398),
        720: (0.392, 0.415),
    },
    "ETTm1": {
        96:  (0.310, 0.334),
        192: (0.348, 0.362),
        336: (0.376, 0.391),
        720: (0.440, 0.423),
    },
    "ETTm2": {
        96:  (0.170, 0.245),
        192: (0.229, 0.291),
        336: (0.303, 0.343),
        720: (0.373, 0.399),
    },
    "Weather": {
        96:  (0.155, 0.205),
        192: (0.201, 0.245),
        336: (0.237, 0.265),
        720: (0.312, 0.334),
    },
    "Solar-Energy": {
        96:  (0.171, 0.231),
        192: (0.218, 0.263),
        336: (0.212, 0.269),
        720: (0.212, 0.270),
    },
    "Electricity": {
        96:  (0.135, 0.222),
        192: (0.147, 0.235),
        336: (0.164, 0.245),
        720: (0.212, 0.310),
    },
    "Traffic": {
        96:  (0.392, 0.253),
        192: (0.402, 0.258),
        336: (0.428, 0.263),
        720: (0.441, 0.282),
    },
}


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    data: str
    data_path: str
    root_path: Path
    enc_in: int
    dec_in: int
    c_out: int
    # TimeMixer++ hypers
    d_model: int
    d_ff: int
    n_heads: int
    e_layers: int          # L  MixerBlocks
    top_k: int             # K  dominant periods
    down_sampling_layers: int   # M  extra scales (total M+1)
    learning_rate: float
    batch_size: int
    train_epochs: int
    patience: int
    factor: int
    use_norm: Optional[int]


DATASETS: List[DatasetConfig] = [
    # ── ETT family ──────────────────────────────────────────────────────────
    DatasetConfig(
        name="ETTh1", data="ETTh1", data_path="ETTh1.csv",
        root_path=DATA_ROOT / "ETT-small",
        enc_in=7, dec_in=7, c_out=7,
        d_model=32, d_ff=64, n_heads=4, e_layers=2,
        top_k=3, down_sampling_layers=3,
        learning_rate=0.00005, batch_size=64, train_epochs=20, patience=10,
        factor=1, use_norm=None,
    ),
    DatasetConfig(
        name="ETTh2", data="ETTh2", data_path="ETTh2.csv",
        root_path=DATA_ROOT / "ETT-small",
        enc_in=7, dec_in=7, c_out=7,
        d_model=32, d_ff=64, n_heads=4, e_layers=2,
        top_k=3, down_sampling_layers=3,
        learning_rate=0.00005, batch_size=64, train_epochs=20, patience=10,
        factor=1, use_norm=None,
    ),
    DatasetConfig(
        name="ETTm1", data="ETTm1", data_path="ETTm1.csv",
        root_path=DATA_ROOT / "ETT-small",
        enc_in=7, dec_in=7, c_out=7,
        d_model=32, d_ff=64, n_heads=4, e_layers=2,
        top_k=3, down_sampling_layers=3,
        learning_rate=0.00005, batch_size=64, train_epochs=20, patience=10,
        factor=1, use_norm=None,
    ),
    DatasetConfig(
        name="ETTm2", data="ETTm2", data_path="ETTm2.csv",
        root_path=DATA_ROOT / "ETT-small",
        enc_in=7, dec_in=7, c_out=7,
        d_model=32, d_ff=64, n_heads=4, e_layers=2,
        top_k=3, down_sampling_layers=3,
        learning_rate=0.00005, batch_size=64, train_epochs=20, patience=10,
        factor=1, use_norm=None,
    ),
    # ── Weather ─────────────────────────────────────────────────────────────
    DatasetConfig(
        name="Weather", data="custom", data_path="weather.csv",
        root_path=DATA_ROOT / "weather",
        enc_in=21, dec_in=21, c_out=21,
        d_model=32, d_ff=64, n_heads=4, e_layers=2,
        top_k=3, down_sampling_layers=3,
        learning_rate=0.00005, batch_size=64, train_epochs=20, patience=10,
        factor=3, use_norm=None,
    ),
    # ── Solar-Energy ─────────────────────────────────────────────────────────
    DatasetConfig(
        name="Solar-Energy", data="Solar", data_path="solar_AL.txt",
        root_path=DATA_ROOT / "solar",
        enc_in=137, dec_in=137, c_out=137,
        d_model=32, d_ff=64, n_heads=4, e_layers=2,
        top_k=3, down_sampling_layers=3,
        learning_rate=0.00005, batch_size=16, train_epochs=20, patience=10,
        factor=3, use_norm=1,   # was 0: PP model needs RevIN to bound FFT amplitudes (NaN'd at epoch 1 without it)
    ),
    # ── Electricity ──────────────────────────────────────────────────────────
    DatasetConfig(
        name="Electricity", data="custom", data_path="electricity.csv",
        root_path=DATA_ROOT / "electricity",
        enc_in=321, dec_in=321, c_out=321,
        d_model=32, d_ff=64, n_heads=4, e_layers=2,
        top_k=3, down_sampling_layers=3,
        learning_rate=0.00005, batch_size=16, train_epochs=20, patience=10,
        factor=3, use_norm=None,
    ),
    # ── Traffic ──────────────────────────────────────────────────────────────
    DatasetConfig(
        name="Traffic", data="custom", data_path="traffic.csv",
        root_path=DATA_ROOT / "traffic",
        enc_in=862, dec_in=862, c_out=862,
        d_model=32, d_ff=64, n_heads=4, e_layers=2,
        top_k=3, down_sampling_layers=3,
        learning_rate=0.00005, batch_size=4, train_epochs=20, patience=10,
        factor=3, use_norm=None,
    ),
]


# ── Log parsing ──────────────────────────────────────────────────────────────

MSE_PAT      = re.compile(r"mse:([0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?)")
MAE_PAT      = re.compile(r"mae:([0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?)")
ELAPSED_PAT  = re.compile(r"elapsed_ms:\s*([0-9]+)")
EPOCH_PAT    = re.compile(r"Epoch:\s+\d+\s+cost time:\s+([0-9]+(?:\.[0-9]+)?)")


def _last(pat, text) -> Optional[float]:
    m = pat.findall(text)
    return float(m[-1]) if m else None


def parse_log_file(path: Path):
    if not path.exists():
        return None, None, None, "PENDING"
    text = path.read_text(encoding="utf-8", errors="ignore")
    mse, mae = _last(MSE_PAT, text), _last(MAE_PAT, text)
    if mse is None or mae is None:
        return None, None, None, "RUNNING/INCOMPLETE"
    elapsed_ms = _last(ELAPSED_PAT, text)
    epoch_times = [float(x) for x in EPOCH_PAT.findall(text)]
    runtime_s = (elapsed_ms / 1000.0) if elapsed_ms is not None else (sum(epoch_times) if epoch_times else None)
    return mse, mae, runtime_s, "OK"


# ── Subprocess runner ────────────────────────────────────────────────────────

def run_one(ds: DatasetConfig, pred_len: int):
    model_id = f"PP_{ds.name}_96_{pred_len}"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{ds.name}_pred{pred_len}.log"

    cmd = [
        sys.executable, "-u", str(REPO_ROOT / "run.py"),
        "--task_name",   "long_term_forecast",
        "--is_training", "1",
        "--model_id",    model_id,
        "--model",       "TimeMixerPP",
        "--data",        ds.data,
        "--root_path",   str(ds.root_path),
        "--data_path",   ds.data_path,
        "--features",    "M",
        "--seq_len",     "96",
        "--label_len",   "0",
        "--pred_len",    str(pred_len),
        "--enc_in",      str(ds.enc_in),
        "--dec_in",      str(ds.dec_in),
        "--c_out",       str(ds.c_out),
        "--d_model",     str(ds.d_model),
        "--d_ff",        str(ds.d_ff),
        "--n_heads",     str(ds.n_heads),
        "--e_layers",    str(ds.e_layers),
        "--top_k",       str(ds.top_k),
        "--down_sampling_layers", str(ds.down_sampling_layers),
        "--down_sampling_method", "avg",   # unused by PP model, satisfies run.py arg parser
        "--down_sampling_window", "2",
        "--factor",      str(ds.factor),
        "--learning_rate", str(ds.learning_rate),
        "--lradj",       "type1",   # monotonic decay (no OneCycle ramp-up) for numerical stability
        "--train_epochs", str(ds.train_epochs),
        "--patience",    str(ds.patience),
        "--batch_size",  str(ds.batch_size),
        "--num_workers", "0",
        "--des",         "Exp",
        "--itr",         "1",
    ]
    if ds.use_norm is not None:
        cmd.extend(["--use_norm", str(ds.use_norm)])

    print(f"\n[RUN] {ds.name} pred_len={pred_len} model_id={model_id}", flush=True)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
    wall_s = time.perf_counter() - start

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    mse  = _last(MSE_PAT, text)
    mae  = _last(MAE_PAT, text)
    elapsed_ms = _last(ELAPSED_PAT, text)
    runtime_s = (elapsed_ms / 1000.0) if elapsed_ms is not None else wall_s
    # Detect NaN divergence: numeric mse/mae won't be parsed (regex skips "nan").
    if proc.returncode != 0:
        status = f"ERROR(rc={proc.returncode})"
    elif mse is None or mae is None:
        status = "NAN/DIVERGED"
    else:
        status = "OK"
    return mse, mae, runtime_s, status


# ── Markdown updater ─────────────────────────────────────────────────────────

START_TOKEN = "<!-- AUTO_PP_RESULTS_START -->"
END_TOKEN   = "<!-- AUTO_PP_RESULTS_END -->"


def update_md(rows: List[dict]) -> None:
    text = MD_PATH.read_text(encoding="utf-8")
    si = text.find(START_TOKEN)
    ei = text.find(END_TOKEN)
    if si < 0 or ei < 0:
        raise RuntimeError("Could not find auto markers in results_comparison_pp.md")

    lines = [
        "| Dataset | pred_len | Paper MSE | Paper MAE | Local MSE | Local MAE | Runtime (s) | Runtime (h) | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        rs = r["runtime_s"]
        rh = "-" if rs == "-" else f"{float(rs) / 3600:.2f}"
        lines.append(
            f"| {r['dataset']} | {r['pred_len']} | {r['paper_mse']} | {r['paper_mae']}"
            f" | {r['local_mse']} | {r['local_mae']} | {rs} | {rh} | {r['status']} |"
        )

    new_block = "\n".join(lines) + "\n"
    updated = text[: si + len(START_TOKEN)] + "\n" + new_block + text[ei:]
    MD_PATH.write_text(updated, encoding="utf-8")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    sys.stdout.reconfigure(line_buffering=True)
    rows = []
    for ds in DATASETS:
        for pred_len in PRED_LENS:
            paper_mse, paper_mae = PAPER[ds.name][pred_len]
            mse, mae, runtime_s, status = run_one(ds, pred_len)
            rows.append({
                "dataset":   ds.name,
                "pred_len":  str(pred_len),
                "paper_mse": f"{paper_mse:.3f}",
                "paper_mae": f"{paper_mae:.3f}",
                "local_mse": "-" if mse is None else f"{mse:.6f}",
                "local_mae": "-" if mae is None else f"{mae:.6f}",
                "runtime_s": "-" if runtime_s is None else f"{runtime_s:.1f}",
                "status":    status,
            })
        update_md(rows)
        print(f"[MD] results_comparison_pp.md updated after {ds.name}.", flush=True)

    update_md(rows)
    print("\n[DONE] results_comparison_pp.md fully updated.")


if __name__ == "__main__":
    main()
