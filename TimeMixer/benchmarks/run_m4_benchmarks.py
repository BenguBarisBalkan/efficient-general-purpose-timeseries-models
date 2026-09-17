"""
Univariate short-term forecasting benchmark on M4 (SparseTSF vs TimeMixer).

Matches the paper's M4 setup: 6 subsets (Yearly/Quarterly/Monthly/Weekly/Daily/Hourly),
input length = 2×horizon, metrics **SMAPE / MASE / OWA** averaged across subsets (Table 2).

Per model it runs all 6 subsets (run.py writes ./m4_results/<model>/<Subset>_forecast.csv), then
computes the aggregate metrics directly with utils.m4_summary.M4Summary — no log scraping, and OWA
uses dataset/m4/submission-Naive2.csv. Idempotent: a subset whose forecast CSV already exists is
skipped, so re-launching is safe.

    venv/Scripts/python.exe benchmarks/run_m4_benchmarks.py
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT.parent / "dataset"
M4_ROOT = DATA_ROOT / "m4"
M4_RESULTS = REPO_ROOT / "m4_results"          # run.py's short-term test() writes here
MD_PATH = REPO_ROOT / "reports" / "results_comparison_m4.md"

sys.path.insert(0, str(REPO_ROOT))             # so we can import utils.m4_summary

MODELS = ["SparseTSF"]          # local runs — SparseTSF only (compute focus)
SUBSETS = ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"]

# Published reference numbers (not re-run locally) for context — TimeMixer++ paper, Table 2 (M4 avg).
PAPER_REF = {
    "TimeMixer (paper)":   {"SMAPE": 11.723, "MASE": 1.559, "OWA": 0.840},
    "TimeMixer++ (paper)": {"SMAPE": 11.448, "MASE": 1.487, "OWA": 0.821},
}

# SparseTSF period_len per subset — must divide BOTH horizon and seq_len(=2×horizon);
# chosen near each frequency's natural cycle.  (h, sl): Y(6,12) Q(8,16) M(18,36) W(13,26) D(14,28) H(48,96)
SPARSE_PERIOD = {"Yearly": 2, "Quarterly": 4, "Monthly": 6, "Weekly": 13, "Daily": 7, "Hourly": 24}

# Shared / per-model training config (each model's own reasonable hypers; documented in the report).
COMMON = dict(train_epochs="10", patience="5", batch_size="16", num_workers="0", loss="SMAPE")
MODEL_CFG = {
    "SparseTSF": dict(learning_rate="0.01", extra=["--model_type", "linear"]),
    # NOTE: if TimeMixer is ever added back to MODELS, it needs >=2 scales
    # (--down_sampling_layers 1 --down_sampling_window 2); 0 crashes its season mixing.
    "TimeMixer": dict(learning_rate="0.001",
                      extra=["--d_model", "16", "--d_ff", "32", "--e_layers", "2",
                             "--down_sampling_layers", "1", "--down_sampling_window", "2",
                             "--down_sampling_method", "avg", "--channel_independence", "1"]),
}


def forecast_csv(model: str, subset: str) -> Path:
    return M4_RESULTS / model / f"{subset}_forecast.csv"


def run_one(model: str, subset: str) -> None:
    log_dir = REPO_ROOT / "logs" / "run_logs_m4"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{model}_{subset}.log"
    cfg = MODEL_CFG[model]

    cmd = [
        sys.executable, "-u", str(REPO_ROOT / "run.py"),
        "--task_name", "short_term_forecast",
        "--is_training", "1",
        "--model_id", f"M4_{model}_{subset}",
        "--model", model,
        "--data", "m4",
        "--root_path", str(M4_ROOT),
        "--seasonal_patterns", subset,
        "--features", "M",
        "--enc_in", "1", "--dec_in", "1", "--c_out", "1",
        "--period_len", str(SPARSE_PERIOD[subset]),   # SparseTSF only; ignored by TimeMixer
        "--learning_rate", cfg["learning_rate"],
        "--train_epochs", COMMON["train_epochs"],
        "--patience", COMMON["patience"],
        "--batch_size", COMMON["batch_size"],
        "--num_workers", COMMON["num_workers"],
        "--loss", COMMON["loss"],
        "--des", "M4",
    ] + cfg["extra"]

    print(f"\n[RUN] {model} {subset} (period_len={SPARSE_PERIOD[subset]})", flush=True)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as f:
        subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
    print(f"      finished in {time.perf_counter() - start:.1f}s", flush=True)


def compute_metrics(model: str) -> Optional[Dict[str, Dict[str, float]]]:
    """Aggregate SMAPE/MASE/OWA/MAPE via M4Summary once all 6 subset forecasts exist."""
    if not all(forecast_csv(model, s).exists() for s in SUBSETS):
        return None
    from utils.m4_summary import M4Summary
    # M4Summary concatenates file_path + '<Subset>_forecast.csv', so it needs a trailing sep.
    fp = str(M4_RESULTS / model) + "/"
    smape, owa, mape, mase = M4Summary(fp, str(M4_ROOT)).evaluate()
    return {"SMAPE": smape, "MASE": mase, "OWA": owa, "MAPE": mape}


RESULTS_START = "<!-- AUTO_M4_RESULTS_START -->"
RESULTS_END = "<!-- AUTO_M4_RESULTS_END -->"


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.3f}"


def update_md(metrics_by_model: Dict[str, Optional[dict]]) -> None:
    rows = [
        "| Model | SMAPE | MASE | OWA |",
        "|---|---:|---:|---:|",
    ]
    for model in MODELS:
        m = metrics_by_model.get(model)
        if m is None:
            rows.append(f"| {model} (ours) | - | - | - |")
        else:
            rows.append(f"| {model} (ours) | {_fmt(m['SMAPE'].get('Average'))} | "
                        f"{_fmt(m['MASE'].get('Average'))} | {_fmt(m['OWA'].get('Average'))} |")
    # published reference rows (cited, not re-run here)
    for name, r in PAPER_REF.items():
        rows.append(f"| _{name}_ | _{r['SMAPE']:.3f}_ | _{r['MASE']:.3f}_ | _{r['OWA']:.3f}_ |")
    # per-subset SMAPE detail
    detail = ["", "**Per-subset SMAPE**", "",
              "| Model | " + " | ".join(SUBSETS) + " | Average |",
              "|---|" + "---:|" * (len(SUBSETS) + 1)]
    for model in MODELS:
        m = metrics_by_model.get(model)
        if m is None:
            detail.append(f"| {model} |" + " - |" * (len(SUBSETS) + 1))
        else:
            s = m["SMAPE"]
            # M4Summary groups Weekly/Daily/Hourly under 'Others'
            vals = [s.get("Yearly"), s.get("Quarterly"), s.get("Monthly"),
                    s.get("Others"), s.get("Others"), s.get("Others"), s.get("Average")]
            detail.append(f"| {model} | " + " | ".join(_fmt(v) for v in vals) + " |")

    body = "\n".join(rows + detail) + "\n"
    text = MD_PATH.read_text(encoding="utf-8")
    si, ei = text.find(RESULTS_START), text.find(RESULTS_END)
    if si < 0 or ei < 0 or ei <= si:
        raise RuntimeError(f"Markers not found in {MD_PATH.name}")
    MD_PATH.write_text(text[: si + len(RESULTS_START)] + "\n" + body + text[ei:], encoding="utf-8")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    metrics: Dict[str, Optional[dict]] = {}
    for model in MODELS:
        for subset in SUBSETS:
            if forecast_csv(model, subset).exists():
                print(f"[SKIP] {model} {subset} (forecast exists)", flush=True)
                continue
            run_one(model, subset)
        metrics[model] = compute_metrics(model)
        m = metrics[model]
        if m:
            print(f"[{model}] SMAPE={m['SMAPE'].get('Average')} "
                  f"MASE={m['MASE'].get('Average')} OWA={m['OWA'].get('Average')}", flush=True)
        update_md(metrics)

    update_md(metrics)
    print("\n[DONE] M4 benchmark complete; results_comparison_m4.md updated.", flush=True)


if __name__ == "__main__":
    main()
