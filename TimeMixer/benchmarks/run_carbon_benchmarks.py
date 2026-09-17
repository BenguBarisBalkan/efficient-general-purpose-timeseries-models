"""
Energy / carbon benchmark: TimeMixer (v1) vs TimeMixer++ , measured with CodeCarbon.

Protocol: MATCHED config + FIXED epochs (early stopping disabled) for both models, so the
measured difference reflects the *architecture*, not differing hyperparameters. Only the light
datasets are used (ETT x4 + Weather); Solar/Electricity/Traffic are deliberately excluded.

Run matrix: {TimeMixer, TimeMixerPP} x {ETTh1,ETTh2,ETTm1,ETTm2,Weather} x {96,192,336,720} = 40 runs.

Idempotent: a config whose log already holds a valid result is skipped, so this can be
re-launched safely after an interruption.
"""

import csv
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
LOGS_DIR = REPO_ROOT / "logs" / "run_logs_carbon"
MD_PATH = REPO_ROOT / "reports" / "results_carbon_comparison.md"
EMISSIONS_FILE = "emissions_timemixer.csv"
EMISSIONS_PATH = REPO_ROOT / "logs" / EMISSIONS_FILE

PRED_LENS = [96, 192, 336, 720]
MODELS = ["TimeMixer", "TimeMixerPP", "SparseTSF"]
BASELINE_MODEL = "TimeMixer"   # 1.0x reference for the energy-ratio column

# ── Matched configuration (identical for both models) ────────────────────────
D_MODEL = 32
D_FF = 64
N_HEADS = 4
E_LAYERS = 2
DOWN_SAMPLING_LAYERS = 3
BATCH_SIZE = 64
TRAIN_EPOCHS = 5        # fixed budget
PATIENCE = 999          # effectively disables early stopping
LEARNING_RATE = 0.00005
LRADJ = "type1"
SEQ_LEN = 96
TOP_K = 3
DES = "Carbon"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    data: str
    data_path: str
    root_path: Path
    enc_in: int
    factor: int
    period_len: int   # SparseTSF-only; must divide seq_len(=96) and every pred_len. Ignored by other models.


# period_len follows SparseTSF's per-dataset choice (24 for hourly ETTh, 4 for ETTm/Weather).
# Both divide seq_len=96 and all pred_lens, so the matched-protocol carbon runs are valid.
DATASETS: List[DatasetConfig] = [
    DatasetConfig("ETTh1", "ETTh1", "ETTh1.csv", DATA_ROOT / "ETT-small", 7, 1, 24),
    DatasetConfig("ETTh2", "ETTh2", "ETTh2.csv", DATA_ROOT / "ETT-small", 7, 1, 24),
    DatasetConfig("ETTm1", "ETTm1", "ETTm1.csv", DATA_ROOT / "ETT-small", 7, 1, 4),
    DatasetConfig("ETTm2", "ETTm2", "ETTm2.csv", DATA_ROOT / "ETT-small", 7, 1, 4),
    DatasetConfig("Weather", "custom", "weather.csv", DATA_ROOT / "weather", 21, 3, 4),
]


# ── Log parsing ──────────────────────────────────────────────────────────────

MSE_PAT = re.compile(r"mse:([0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?)")
MAE_PAT = re.compile(r"mae:([0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?)")


def _last(pat: re.Pattern, text: str) -> Optional[float]:
    m = pat.findall(text)
    return float(m[-1]) if m else None


def build_setting(ds: DatasetConfig, model: str, pred_len: int) -> str:
    """Reproduce the `setting` string run.py builds, used as the emissions join key."""
    model_id = f"CARBON_{model}_{ds.name}_{SEQ_LEN}_{pred_len}"
    return (
        f"long_term_forecast_{model_id}_none_{model}_{ds.data}"
        f"_sl{SEQ_LEN}_pl{pred_len}_dm{D_MODEL}_nh{N_HEADS}_el{E_LAYERS}_dl1"
        f"_df{D_FF}_fc{ds.factor}_ebtimeF_dtTrue_{DES}_0"
    )


def log_path_for(ds: DatasetConfig, model: str, pred_len: int) -> Path:
    return LOGS_DIR / f"{model}_{ds.name}_pred{pred_len}.log"


def parse_log(path: Path) -> Tuple[Optional[float], Optional[float]]:
    if not path.exists():
        return None, None
    text = path.read_text(encoding="utf-8", errors="ignore")
    return _last(MSE_PAT, text), _last(MAE_PAT, text)


def read_emissions() -> Dict[str, dict]:
    """Map project_name -> latest emissions row."""
    if not EMISSIONS_PATH.exists():
        return {}
    out: Dict[str, dict] = {}
    with EMISSIONS_PATH.open(encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f):
            name = row.get("project_name")
            if name:
                out[name] = row  # later rows win
    return out


def _f(row: Optional[dict], key: str) -> Optional[float]:
    if not row:
        return None
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return None


# ── Runner ───────────────────────────────────────────────────────────────────

def run_one(ds: DatasetConfig, model: str, pred_len: int) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = log_path_for(ds, model, pred_len)
    model_id = f"CARBON_{model}_{ds.name}_{SEQ_LEN}_{pred_len}"

    cmd = [
        sys.executable, "-u", str(REPO_ROOT / "run.py"),
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--model_id", model_id,
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
        "--d_model", str(D_MODEL),
        "--d_ff", str(D_FF),
        "--n_heads", str(N_HEADS),
        "--e_layers", str(E_LAYERS),
        "--down_sampling_layers", str(DOWN_SAMPLING_LAYERS),
        "--down_sampling_method", "avg",
        "--down_sampling_window", "2",
        "--factor", str(ds.factor),
        "--learning_rate", str(LEARNING_RATE),
        "--lradj", LRADJ,
        "--train_epochs", str(TRAIN_EPOCHS),
        "--patience", str(PATIENCE),
        "--batch_size", str(BATCH_SIZE),
        "--top_k", str(TOP_K),
        "--num_workers", "0",
        "--des", DES,
        "--itr", "1",
        # SparseTSF params (ignored by TimeMixer/TimeMixerPP). period_len must divide
        # seq_len and pred_len; model_type=linear is the <1k-param variant.
        "--period_len", str(ds.period_len),
        "--model_type", "linear",
        # energy tracking
        "--track_emissions", "1",
        "--emissions_output_dir", str(REPO_ROOT / "logs"),
        "--emissions_file", EMISSIONS_FILE,
    ]

    print(f"\n[RUN] {model} {ds.name} pred_len={pred_len}", flush=True)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as f:
        subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
    print(f"      finished in {time.perf_counter() - start:.1f}s", flush=True)


# ── Report ───────────────────────────────────────────────────────────────────

RESULTS_START = "<!-- AUTO_CARBON_RESULTS_START -->"
RESULTS_END = "<!-- AUTO_CARBON_RESULTS_END -->"
SUMMARY_START = "<!-- AUTO_CARBON_SUMMARY_START -->"
SUMMARY_END = "<!-- AUTO_CARBON_SUMMARY_END -->"


def _fmt(v: Optional[float], spec: str) -> str:
    return "-" if v is None else format(v, spec)


def _replace_block(text: str, start: str, end: str, body: str) -> str:
    si, ei = text.find(start), text.find(end)
    if si < 0 or ei < 0 or ei <= si:
        raise RuntimeError(f"Markers {start}/{end} not found in {MD_PATH.name}")
    return text[: si + len(start)] + "\n" + body + text[ei:]


def update_md() -> None:
    emissions = read_emissions()

    # ── per-run rows ────────────────────────────────────────────────────────
    rows = [
        "| Dataset | pred_len | Model | Duration (s) | GPU kWh | CPU kWh | RAM kWh | Total kWh | CO2 (kg) | MSE | MAE |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    # totals[ds][model] = [energy_kwh, co2_kg, duration_s, n_runs]
    totals: Dict[str, Dict[str, List[float]]] = {}

    for ds in DATASETS:
        for pred_len in PRED_LENS:
            for model in MODELS:
                mse, mae = parse_log(log_path_for(ds, model, pred_len))
                row = emissions.get(build_setting(ds, model, pred_len))
                dur = _f(row, "duration")
                gpu = _f(row, "gpu_energy")
                cpu = _f(row, "cpu_energy")
                ram = _f(row, "ram_energy")
                tot = _f(row, "energy_consumed")
                co2 = _f(row, "emissions")

                rows.append(
                    f"| {ds.name} | {pred_len} | {model} | {_fmt(dur, '.1f')} | "
                    f"{_fmt(gpu, '.3e')} | {_fmt(cpu, '.3e')} | {_fmt(ram, '.3e')} | "
                    f"{_fmt(tot, '.3e')} | {_fmt(co2, '.3e')} | "
                    f"{_fmt(mse, '.6f')} | {_fmt(mae, '.6f')} |"
                )

                if tot is not None:
                    acc = totals.setdefault(ds.name, {}).setdefault(model, [0.0, 0.0, 0.0, 0.0])
                    acc[0] += tot
                    acc[1] += co2 or 0.0
                    acc[2] += dur or 0.0
                    acc[3] += 1

    # ── head-to-head summary (N models, energy in kWh; ratio vs baseline) ────
    # Total kWh per model, plus a ratio column normalised so the baseline model = 1.0x.
    kwh_cols = " | ".join(f"{m} kWh" for m in MODELS)
    ratio_cols = " | ".join(f"{m} /{BASELINE_MODEL}" for m in MODELS if m != BASELINE_MODEL)
    summary = [
        f"| Dataset | Runs | {kwh_cols} | {ratio_cols} |",
        "|---|---:|" + "---:|" * len(MODELS) + "---:|" * (len(MODELS) - 1),
    ]

    grand = {m: [0.0, 0.0] for m in MODELS}  # model -> [kwh, co2]

    def _row(label: str, per: Dict[str, List[float]], n: int, bold: bool) -> str:
        kwh = {m: (per[m][0] if m in per else None) for m in MODELS}
        base = kwh.get(BASELINE_MODEL)
        wrap = (lambda s: f"**{s}**") if bold else (lambda s: s)
        kwh_str = " | ".join(wrap(_fmt(kwh[m], '.3e')) for m in MODELS)
        ratio_str = " | ".join(
            wrap('-' if (base is None or kwh[m] is None or not base) else f"{kwh[m] / base:.2f}x")
            for m in MODELS if m != BASELINE_MODEL
        )
        return f"| {label} | {n if n else ''} | {kwh_str} | {ratio_str} |"

    for ds in DATASETS:
        per = totals.get(ds.name, {})
        if not per:
            continue
        n = int(max((per[m][3] for m in per), default=0))
        summary.append(_row(ds.name, per, n, bold=False))
        for m in MODELS:
            if m in per:
                grand[m][0] += per[m][0]
                grand[m][1] += per[m][1]

    if any(grand[m][0] for m in MODELS):
        grand_per = {m: [grand[m][0], grand[m][1], 0.0, 0.0] for m in MODELS if grand[m][0]}
        summary.append(_row("**TOTAL**", grand_per, 0, bold=True))
        # a compact CO2 line so carbon isn't buried
        co2_str = " | ".join(f"{m}: {_fmt(grand[m][1] if grand[m][1] else None, '.3e')} kg"
                             for m in MODELS)
        summary.append("")
        summary.append(f"_Total CO2 — {co2_str}_")

    text = MD_PATH.read_text(encoding="utf-8")
    text = _replace_block(text, RESULTS_START, RESULTS_END, "\n".join(rows) + "\n")
    text = _replace_block(text, SUMMARY_START, SUMMARY_END, "\n".join(summary) + "\n")
    MD_PATH.write_text(text, encoding="utf-8")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    total = len(DATASETS) * len(PRED_LENS) * len(MODELS)
    done = 0

    for ds in DATASETS:
        for pred_len in PRED_LENS:
            for model in MODELS:
                done += 1
                mse, _ = parse_log(log_path_for(ds, model, pred_len))
                if mse is not None:
                    print(f"[SKIP {done}/{total}] {model} {ds.name} pred={pred_len} "
                          f"(done, MSE={mse:.6f})", flush=True)
                    continue

                print(f"\n>>> [{done}/{total}] {model} {ds.name} pred_len={pred_len} <<<", flush=True)
                run_one(ds, model, pred_len)
                update_md()
                print(f"[MD] results_carbon_comparison.md updated.", flush=True)

    update_md()
    print("\n[DONE] Carbon benchmark complete; results_carbon_comparison.md updated.", flush=True)


if __name__ == "__main__":
    main()
