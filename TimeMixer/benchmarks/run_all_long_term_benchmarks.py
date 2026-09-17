import re
import subprocess
import time
import sys
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# This script lives in TimeMixer/benchmarks/, so REPO_ROOT is one level up.
REPO_ROOT = Path(__file__).resolve().parent.parent
MD_PATH = REPO_ROOT / "reports" / "results_comparison.md"
DATA_ROOT = REPO_ROOT.parent / "dataset"


PRED_LENS = [96, 192, 336, 720]


# Paper values (Table 13, unified hyperparameter results) for TimeMixer (MSE/MAE).
PAPER: Dict[str, Dict[int, Tuple[float, float]]] = {
    "ETTh1": {96: (0.375, 0.400), 192: (0.429, 0.421), 336: (0.484, 0.458), 720: (0.498, 0.482)},
    "ETTh2": {96: (0.289, 0.341), 192: (0.372, 0.392), 336: (0.386, 0.414), 720: (0.412, 0.434)},
    "ETTm1": {96: (0.320, 0.357), 192: (0.361, 0.381), 336: (0.390, 0.404), 720: (0.454, 0.441)},
    "ETTm2": {96: (0.175, 0.258), 192: (0.237, 0.299), 336: (0.298, 0.340), 720: (0.391, 0.396)},
    "Weather": {96: (0.163, 0.209), 192: (0.208, 0.250), 336: (0.251, 0.287), 720: (0.339, 0.341)},
    "Solar-Energy": {96: (0.189, 0.259), 192: (0.222, 0.283), 336: (0.231, 0.292), 720: (0.223, 0.285)},
    "Electricity": {96: (0.153, 0.247), 192: (0.166, 0.256), 336: (0.185, 0.277), 720: (0.225, 0.310)},
    "Traffic": {96: (0.462, 0.285), 192: (0.473, 0.296), 336: (0.498, 0.296), 720: (0.506, 0.313)},
}


@dataclass(frozen=True)
class DatasetConfig:
    # Paper dataset name used as key in PAPER map.
    name: str
    # run.py args for long-term.
    data: str
    data_path: str
    root_path: Path  # absolute
    enc_in: int
    dec_in: int
    c_out: int
    d_model: int
    e_layers: int
    d_ff: int
    factor: int
    use_norm: Optional[int]  # None means don't override
    channel_independence: int
    down_sampling_layers: int
    down_sampling_window: int
    learning_rate: float
    batch_size: int
    train_epochs: int
    patience: int


DATASETS: List[DatasetConfig] = [
    # ETT datasets (M=3 scales, L=2 PDM blocks).
    DatasetConfig(
        name="ETTh1",
        data="ETTh1",
        data_path="ETTh1.csv",
        root_path=DATA_ROOT / "ETT-small",
        enc_in=7,
        dec_in=7,
        c_out=7,
        d_model=16,
        e_layers=2,
        d_ff=32,
        factor=1,
        use_norm=None,
        channel_independence=1,
        down_sampling_layers=3,
        down_sampling_window=2,
        learning_rate=0.01,
        batch_size=128,
        train_epochs=10,
        patience=10,
    ),
    DatasetConfig(
        name="ETTh2",
        data="ETTh2",
        data_path="ETTh2.csv",
        root_path=DATA_ROOT / "ETT-small",
        enc_in=7,
        dec_in=7,
        c_out=7,
        d_model=16,
        e_layers=2,
        d_ff=32,
        factor=1,
        use_norm=None,
        channel_independence=1,
        down_sampling_layers=3,
        down_sampling_window=2,
        learning_rate=0.01,
        batch_size=128,
        train_epochs=10,
        patience=10,
    ),
    DatasetConfig(
        name="ETTm1",
        data="ETTm1",
        data_path="ETTm1.csv",
        root_path=DATA_ROOT / "ETT-small",
        enc_in=7,
        dec_in=7,
        c_out=7,
        d_model=16,
        e_layers=2,
        d_ff=32,
        factor=1,
        use_norm=None,
        channel_independence=1,
        down_sampling_layers=3,
        down_sampling_window=2,
        learning_rate=0.01,
        batch_size=128,
        train_epochs=10,
        patience=10,
    ),
    DatasetConfig(
        name="ETTm2",
        data="ETTm2",
        data_path="ETTm2.csv",
        root_path=DATA_ROOT / "ETT-small",
        enc_in=7,
        dec_in=7,
        c_out=7,
        d_model=32,
        e_layers=2,
        d_ff=32,
        factor=1,
        use_norm=None,
        channel_independence=1,
        down_sampling_layers=3,
        down_sampling_window=2,
        learning_rate=0.01,
        batch_size=128,
        train_epochs=10,
        patience=10,
    ),
    # Custom datasets
    DatasetConfig(
        name="Weather",
        data="custom",
        data_path="weather.csv",
        root_path=DATA_ROOT / "weather",
        enc_in=21,
        dec_in=21,
        c_out=21,
        d_model=16,
        e_layers=2,
        d_ff=32,
        factor=3,
        use_norm=None,  # keep code default
        channel_independence=1,
        down_sampling_layers=3,
        down_sampling_window=2,
        learning_rate=0.01,
        batch_size=128,
        train_epochs=20,
        patience=10,
    ),
    DatasetConfig(
        name="Solar-Energy",
        data="Solar",
        data_path="solar_AL.txt",
        root_path=DATA_ROOT / "solar",
        enc_in=137,
        dec_in=137,
        c_out=137,
        d_model=128,
        e_layers=2,
        d_ff=2048,
        factor=3,
        use_norm=0,
        channel_independence=0,
        down_sampling_layers=3,
        down_sampling_window=2,
        learning_rate=0.01,
        batch_size=32,
        train_epochs=20,
        patience=10,
    ),
    DatasetConfig(
        name="Electricity",
        data="custom",
        data_path="electricity.csv",
        root_path=DATA_ROOT / "electricity",
        enc_in=321,
        dec_in=321,
        c_out=321,
        d_model=16,
        e_layers=2,
        d_ff=32,
        factor=3,
        use_norm=None,
        channel_independence=1,
        down_sampling_layers=3,
        down_sampling_window=2,
        learning_rate=0.01,
        batch_size=32,
        train_epochs=20,
        patience=10,
    ),
    DatasetConfig(
        name="Traffic",
        data="custom",
        data_path="traffic.csv",
        root_path=DATA_ROOT / "traffic",
        enc_in=862,
        dec_in=862,
        c_out=862,
        d_model=32,
        e_layers=2,
        d_ff=64,
        factor=3,
        use_norm=None,
        channel_independence=1,
        down_sampling_layers=3,
        down_sampling_window=2,
        learning_rate=0.001,
        batch_size=8,
        train_epochs=20,
        patience=10,
    ),
]


MSE_PAT = re.compile(r"mse:([0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?)")
MAE_PAT = re.compile(r"mae:([0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?)")
ELAPSED_PAT = re.compile(r"elapsed_ms:\s*([0-9]+)")
EPOCH_TIME_PAT = re.compile(r"Epoch:\s+\d+\s+cost time:\s+([0-9]+(?:\.[0-9]+)?)")


def _parse_last_number(pat: re.Pattern, text: str) -> Optional[float]:
    matches = pat.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def parse_log_file(log_path: Path) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    if not log_path.exists():
        return None, None, None, "PENDING"
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    mse = _parse_last_number(MSE_PAT, text)
    mae = _parse_last_number(MAE_PAT, text)
    if mse is None or mae is None:
        return None, None, None, "RUNNING/INCOMPLETE"
    epoch_times = [float(x) for x in EPOCH_TIME_PAT.findall(text)]
    runtime_s = sum(epoch_times) if epoch_times else None
    return mse, mae, runtime_s, "OK"


def run_one(ds: DatasetConfig, pred_len: int) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    model_id = f"{ds.name}_{96}_{pred_len}"

    cmd = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "run.py"),
        "--task_name",
        "long_term_forecast",
        "--is_training",
        "1",
        "--model_id",
        model_id,
        "--model",
        "TimeMixer",
        "--data",
        ds.data,
        "--root_path",
        str(ds.root_path),
        "--data_path",
        ds.data_path,
        "--features",
        "M",
        "--seq_len",
        "96",
        "--label_len",
        "0",
        "--pred_len",
        str(pred_len),
        "--e_layers",
        str(ds.e_layers),
        "--d_model",
        str(ds.d_model),
        "--d_ff",
        str(ds.d_ff),
        "--learning_rate",
        str(ds.learning_rate),
        "--train_epochs",
        str(ds.train_epochs),
        "--patience",
        str(ds.patience),
        "--batch_size",
        str(ds.batch_size),
        "--down_sampling_layers",
        str(ds.down_sampling_layers),
        "--down_sampling_window",
        str(ds.down_sampling_window),
        "--down_sampling_method",
        "avg",
        "--num_workers",
        "0",
        "--enc_in",
        str(ds.enc_in),
        "--dec_in",
        str(ds.dec_in),
        "--c_out",
        str(ds.c_out),
        "--factor",
        str(ds.factor),
        "--channel_independence",
        str(ds.channel_independence),
    ]

    if ds.use_norm is not None:
        cmd.extend(["--use_norm", str(ds.use_norm)])

    print(f"\n[RUN] {ds.name} pred_len={pred_len} model_id={model_id}", flush=True)
    start = time.perf_counter()
    logs_dir = REPO_ROOT / "logs" / "run_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{ds.name}_pred{pred_len}.log"

    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
    wall_s = time.perf_counter() - start

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    mse = _parse_last_number(MSE_PAT, text)
    mae = _parse_last_number(MAE_PAT, text)

    # Prefer printed elapsed_ms if available, else use wall time.
    elapsed_ms = _parse_last_number(ELAPSED_PAT, text)
    runtime_s = (elapsed_ms / 1000.0) if elapsed_ms is not None else wall_s

    status = "OK" if proc.returncode == 0 else f"ERROR(rc={proc.returncode})"
    return mse, mae, runtime_s, status


def update_md(rows: List[Dict[str, str]]) -> None:
    text = MD_PATH.read_text(encoding="utf-8")
    start_token = "<!-- AUTO_LONG_TERM_RESULTS_START -->"
    end_token = "<!-- AUTO_LONG_TERM_RESULTS_END -->"

    start_idx = text.find(start_token)
    end_idx = text.find(end_token)
    if start_idx < 0 or end_idx < 0 or end_idx <= start_idx:
        raise RuntimeError("Could not find auto markers in results_comparison.md")

    table = []
    table.append("| Dataset | pred_len | Paper MSE | Paper MAE | Local MSE | Local MAE | Runtime (s) | Runtime (h) | Status |")
    table.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")

    for r in rows:
        rs = r['runtime_s']
        runtime_h = "-" if rs == "-" else f"{float(rs) / 3600:.2f}"
        table.append(
            f"| {r['dataset']} | {r['pred_len']} | {r['paper_mse']} | {r['paper_mae']} | {r['local_mse']} | {r['local_mae']} | {rs} | {runtime_h} | {r['status']} |"
        )

    new_block = "\n".join(table) + "\n"
    updated = text[: start_idx + len(start_token)] + "\n" + new_block + text[end_idx:]
    MD_PATH.write_text(updated, encoding="utf-8")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update-from-logs-only",
        action="store_true",
        help="Do not run new experiments; only parse existing .run_logs and update markdown.",
    )
    args = parser.parse_args()

    rows: List[Dict[str, str]] = []

    for ds in DATASETS:
        for pred_len in PRED_LENS:
            paper_mse, paper_mae = PAPER[ds.name][pred_len]
            if args.update_from_logs_only:
                log_path = REPO_ROOT / "logs" / "run_logs" / f"{ds.name}_pred{pred_len}.log"
                mse, mae, runtime_s, status = parse_log_file(log_path)
            else:
                mse, mae, runtime_s, status = run_one(ds, pred_len)
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

    update_md(rows)
    print("\n[DONE] Updated results_comparison.md with long-term run log.")


if __name__ == "__main__":
    main()

