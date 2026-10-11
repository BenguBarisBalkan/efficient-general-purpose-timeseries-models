"""SparseTSFPlus forecasting ablation: long-term (ETTh1, ETTm1, Weather) and PEMS short-term.

Tests the backbone units on the forecasting path, on top of each dataset's selected period:

  phase         phase-axis mixing, full p x p map
  phase_r4      phase-axis mixing, rank-4 factorisation (the small-data candidate)
  decomp        trend-season split, trend held at its last value (0 params)
  decomp_lin    trend-season split, trend gets its own period map
  phase_decomp  phase mixing + free decomposition together
  periods       multi-period ensemble: one extra period behind a zero-initialised gate

Every training setting is copied from the runner that produced the published SparseTSF cell
(run_sparsetsf_benchmarks.py for long-term, run_pems_benchmarks.py for PEMS), so the `base`
row is directly comparable to the published numbers.

SELECTION. exp_long_term_forecasting.py early-stops on VALIDATION loss, so the best-validation
column here is a clean selection signal. The protocol is: screen every variant at
pred_len=96, pick the best per dataset on validation, then run only that pick (plus base) at
the other horizons with --full. Never pick on the test columns.

Idempotent: a config whose log already holds valid metrics is skipped.

    venv/Scripts/python.exe benchmarks/run_sparsetsf_plus_forecast.py                 # screen
    venv/Scripts/python.exe benchmarks/run_sparsetsf_plus_forecast.py --group pems
    venv/Scripts/python.exe benchmarks/run_sparsetsf_plus_forecast.py --full --variants base,phase_r4
    venv/Scripts/python.exe benchmarks/run_sparsetsf_plus_forecast.py --update-from-logs-only
"""

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT.parent / "dataset"
LOGS_DIR = REPO_ROOT / "logs" / "run_logs_plus_fc"
MD_PATH = REPO_ROOT / "reports" / "results_sparsetsf_plus_forecast.md"

MODEL = "SparseTSFPlus"
SEQ_LEN = 96


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    group: str               # "long" or "pems"
    data: str
    data_path: str
    root_path: Path
    enc_in: int
    factor: int
    period: int              # the selected period (validation-chosen in the period probe)
    extra_period: int        # the second period used by the multi-period variant
    pred_lens: Tuple[int, ...]
    lr: float
    batch: int
    epochs: int
    patience: int
    lradj: Optional[str]     # None -> run.py default (what the PEMS runner uses)
    use_norm: Optional[int]  # None -> not passed


LONG = dict(group="long", lr=0.02, batch=256, epochs=30, patience=5, lradj="type3",
            use_norm=None, pred_lens=(96, 192, 336, 720))
PEMS = dict(group="pems", data="PEMS", root_path=DATA_ROOT / "PEMS", factor=1, lr=0.01,
            batch=16, epochs=10, patience=5, lradj=None, use_norm=0, pred_lens=(12,))

# ETTh1 uses period 12, which beat the published 24 on validation in the period probe.
DATASETS: List[DatasetConfig] = [
    DatasetConfig("ETTh1", data="ETTh1", data_path="ETTh1.csv", root_path=DATA_ROOT / "ETT-small",
                  enc_in=7, factor=1, period=12, extra_period=24, **LONG),
    DatasetConfig("ETTm1", data="ETTm1", data_path="ETTm1.csv", root_path=DATA_ROOT / "ETT-small",
                  enc_in=7, factor=1, period=4, extra_period=12, **LONG),
    DatasetConfig("Weather", data="custom", data_path="weather.csv", root_path=DATA_ROOT / "weather",
                  enc_in=21, factor=3, period=4, extra_period=12, **LONG),
    DatasetConfig("PEMS03", data_path="PEMS03.npz", enc_in=358, period=12, extra_period=4, **PEMS),
    DatasetConfig("PEMS04", data_path="PEMS04.npz", enc_in=307, period=12, extra_period=4, **PEMS),
    DatasetConfig("PEMS07", data_path="PEMS07.npz", enc_in=883, period=12, extra_period=4, **PEMS),
    DatasetConfig("PEMS08", data_path="PEMS08.npz", enc_in=170, period=12, extra_period=4, **PEMS),
]


@dataclass(frozen=True)
class Variant:
    name: str
    flags: Tuple[str, ...]
    groups: Tuple[str, ...]
    period: Optional[int] = None    # overrides the dataset's period when set
    uses_extra: bool = False        # appends --stsf_periods <extra_period>
    note: str = ""


VARIANTS: List[Variant] = [
    Variant("base", (), ("long",), note="selected period, no units"),
    Variant("phase", ("--stsf_phase_mix", "linear"), ("long",), note="full phase map"),
    Variant("phase_r4", ("--stsf_phase_mix", "linear", "--stsf_phase_rank", "4"), ("long",),
            note="rank-4 phase map"),
    Variant("decomp", ("--stsf_decomp", "free"), ("long",), note="trend held at last value"),
    Variant("decomp_lin", ("--stsf_decomp", "linear"), ("long",), note="trend with its own map"),
    Variant("phase_decomp", ("--stsf_phase_mix", "linear", "--stsf_decomp", "free"), ("long",),
            note="phase mixing + free decomposition"),
    Variant("phase_decomp_lin", ("--stsf_phase_mix", "linear", "--stsf_decomp", "linear"), ("long",),
            note="phase mixing + learned decomposition"),
    Variant("periods", (), ("long",), uses_extra=True, note="one extra period, gated"),
    Variant("phase_p4", ("--stsf_phase_mix", "linear"), ("pems",), period=4,
            note="phase mixing at period 4"),
    Variant("dense_p1", (), ("pems",), period=1, note="period 1: the dense limit"),
]

SCREEN_PRED = {"long": 96, "pems": 12}

# The period each published SparseTSF result used. Where the selected period equals it, the
# `base` variant IS published SparseTSF (bitwise, by the equivalence gate), so --full-best does
# not re-run it at the other horizons.
PUBLISHED_PERIOD = {"ETTh1": 24, "ETTm1": 4, "Weather": 4,
                    "PEMS03": 12, "PEMS04": 12, "PEMS07": 12, "PEMS08": 12}

# Published SparseTSF (its own published period) and local TimeMixer, same protocol.
SPARSETSF_PUBLISHED: Dict[Tuple[str, int], float] = {
    ("ETTh1", 96): 0.396635, ("ETTh1", 192): 0.433213, ("ETTh1", 336): 0.449656, ("ETTh1", 720): 0.459474,
    ("ETTm1", 96): 0.362121, ("ETTm1", 192): 0.394356, ("ETTm1", 336): 0.426852, ("ETTm1", 720): 0.485662,
    ("Weather", 96): 0.198635, ("Weather", 192): 0.242842, ("Weather", 336): 0.292309, ("Weather", 720): 0.362776,
    ("PEMS03", 12): 28.28, ("PEMS04", 12): 36.12, ("PEMS07", 12): 42.89, ("PEMS08", 12): 29.25,
}
TIMEMIXER_LOCAL: Dict[Tuple[str, int], float] = {
    ("ETTh1", 96): 0.385794, ("ETTh1", 192): 0.443037, ("ETTh1", 336): 0.513090, ("ETTh1", 720): 0.489165,
    ("ETTm1", 96): 0.326398, ("ETTm1", 192): 0.365291, ("ETTm1", 336): 0.393321, ("ETTm1", 720): 0.451405,
    ("Weather", 96): 0.164526, ("Weather", 192): 0.209339, ("Weather", 336): 0.265349, ("Weather", 720): 0.342314,
}

NUM = r"([0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?)"
MSE_PAT = re.compile(r"mse:" + NUM)
MAE_PAT = re.compile(r"mae:" + NUM)
RMSE_PAT = re.compile(r"RMSE:" + NUM)
MAPE_PAT = re.compile(r"MAPE:" + NUM)
VALI_PAT = re.compile(r"Vali Loss: ([0-9]+(?:\.[0-9]+)?)")
PARAM_PAT = re.compile(r"trainable_params:([0-9]+)")
EPOCH_PAT = re.compile(r"Epoch:\s+\d+\s+cost time:\s+([0-9]+(?:\.[0-9]+)?)")


def _last(pat, text):
    m = pat.findall(text)
    return float(m[-1]) if m else None


def period_for(ds: DatasetConfig, v: Variant) -> int:
    return v.period if v.period is not None else ds.period


def log_path_for(ds: DatasetConfig, v: Variant, pred: int) -> Path:
    return LOGS_DIR / "{}_p{}_pl{}_{}.log".format(ds.name, period_for(ds, v), pred, v.name)


def parse_log(path: Path):
    out = {"mse": None, "mae": None, "rmse": None, "mape": None, "vali": None,
           "params": None, "runtime": None, "status": "PENDING"}
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8", errors="ignore")
    out["mse"], out["mae"] = _last(MSE_PAT, text), _last(MAE_PAT, text)
    out["rmse"], out["mape"] = _last(RMSE_PAT, text), _last(MAPE_PAT, text)
    v = [float(x) for x in VALI_PAT.findall(text)]
    out["vali"] = min(v) if v else None
    p = PARAM_PAT.findall(text)
    out["params"] = int(p[0]) if p else None
    ep = [float(x) for x in EPOCH_PAT.findall(text)]
    out["runtime"] = sum(ep) if ep else None
    out["status"] = "OK" if out["mae"] is not None and out["mse"] is not None else "RUNNING/INCOMPLETE"
    return out


def run_one(ds: DatasetConfig, v: Variant, pred: int) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    period = period_for(ds, v)
    cmd = [
        sys.executable, "-u", str(REPO_ROOT / "run.py"),
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--model_id", "FCP_{}_p{}_{}_{}".format(ds.name, period, pred, v.name),
        "--model", MODEL,
        "--data", ds.data,
        "--root_path", str(ds.root_path),
        "--data_path", ds.data_path,
        "--features", "M",
        "--seq_len", str(SEQ_LEN),
        "--label_len", "0",
        "--pred_len", str(pred),
        "--enc_in", str(ds.enc_in),
        "--dec_in", str(ds.enc_in),
        "--c_out", str(ds.enc_in),
        "--period_len", str(period),
        "--model_type", "linear",
        "--factor", str(ds.factor),
        "--learning_rate", str(ds.lr),
        "--train_epochs", str(ds.epochs),
        "--patience", str(ds.patience),
        "--batch_size", str(ds.batch),
        "--num_workers", "0",
        "--des", "FcP",
        "--itr", "1",
    ]
    if ds.lradj is not None:
        cmd += ["--lradj", ds.lradj]
    if ds.use_norm is not None:
        cmd += ["--use_norm", str(ds.use_norm)]
    cmd += list(v.flags)
    if v.uses_extra:
        cmd += ["--stsf_periods", str(ds.extra_period)]

    print("[RUN] {} pred={} {}".format(ds.name, pred, v.name), flush=True)
    t0 = time.time()
    with log_path_for(ds, v, pred).open("w", encoding="utf-8") as f:
        subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
    print("      done in {:.0f}s".format(time.time() - t0), flush=True)


START = "<!-- AUTO_PLUS_FC_START -->"
END = "<!-- AUTO_PLUS_FC_END -->"

HEADER = """# SparseTSFPlus ablation — forecasting

Backbone units tested on the forecasting path, each on top of the dataset's selected period
(ETTh1 uses 12, chosen on validation over the published 24; ETTm1 and Weather keep 4).

**Select on the `Best vali` column.** Forecasting early-stops on validation loss, so that
column is a clean selection signal; **<-** marks the lowest validation loss per dataset and
horizon. The test columns are for reporting only.

Regenerate (idempotent; `--update-from-logs-only` rebuilds this page without training):

```bash
venv/Scripts/python.exe benchmarks/run_sparsetsf_plus_forecast.py --update-from-logs-only
```
"""


def _fmt(v, nd=6):
    return "-" if v is None else "{:.{}f}".format(v, nd)


def _pct(a, b):
    return "-" if a is None or b is None else "{:+.2f}%".format(100.0 * (a - b) / b)


def update_md() -> None:
    lines: List[str] = []
    for group, title in (("long", "Long-term forecasting (seq_len = 96)"),
                         ("pems", "PEMS multivariate short-term (seq_len = 96, pred_len = 12)")):
        dss = [d for d in DATASETS if d.group == group]
        vs = [v for v in VARIANTS if group in v.groups]
        preds = sorted({p for d in dss for p in d.pred_lens})
        lines.append("### " + title)
        lines.append("")
        if group == "long":
            lines.append("| Dataset | pred | Variant | Period | Params | Best vali | MSE | MAE | vs SparseTSF (published) | vs TimeMixer (local) | Status |")
            lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|")
        else:
            lines.append("| Dataset | Variant | Period | Params | Best vali | MAE | MAPE (%) | RMSE | vs SparseTSF (published MAE) | Status |")
            lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
        any_row = False
        for ds in dss:
            for pred in preds:
                rows = [(v, parse_log(log_path_for(ds, v, pred))) for v in vs]
                rows = [(v, r) for v, r in rows if r["status"] != "PENDING"]
                if not rows:
                    continue
                any_row = True
                done = [(v, r) for v, r in rows if r["vali"] is not None and r["status"] == "OK"]
                best = min(done, key=lambda t: t[1]["vali"])[0].name if done else None
                for v, r in rows:
                    mark = " **<-**" if v.name == best else ""
                    params = "-" if r["params"] is None else "{:,}".format(r["params"])
                    if group == "long":
                        lines.append("| {} | {} | {}{} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                            ds.name, pred, v.name, mark, period_for(ds, v), params,
                            _fmt(r["vali"], 5), _fmt(r["mse"]), _fmt(r["mae"]),
                            _pct(r["mse"], SPARSETSF_PUBLISHED.get((ds.name, pred))),
                            _pct(r["mse"], TIMEMIXER_LOCAL.get((ds.name, pred))), r["status"]))
                    else:
                        mape = None if r["mape"] is None else r["mape"] * 100.0
                        lines.append("| {} | {}{} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                            ds.name, v.name, mark, period_for(ds, v), params,
                            _fmt(r["vali"], 3), _fmt(r["mae"], 2), _fmt(mape, 2), _fmt(r["rmse"], 2),
                            _pct(r["mae"], SPARSETSF_PUBLISHED.get((ds.name, pred))), r["status"]))
        if not any_row:
            lines.append("| _no runs yet_ |")
        lines.append("")
    lines.append("Variant legend:")
    lines.append("")
    for v in VARIANTS:
        flags = " ".join(v.flags) if v.flags else ""
        if v.uses_extra:
            flags = (flags + " --stsf_periods <extra period>").strip()
        if v.period is not None:
            flags = (flags + " --period_len {}".format(v.period)).strip()
        lines.append("- **{}** — {} · `{}`".format(v.name, v.note, flags or "(none)"))

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
    ap.add_argument("--group", choices=("long", "pems", "all"), default="long")
    ap.add_argument("--full", action="store_true", help="every horizon, not just the screen horizon")
    ap.add_argument("--full-best", action="store_true",
                    help="per dataset, run the variant with the lowest VALIDATION loss at the "
                         "screen horizon (plus base, when base differs from published SparseTSF) "
                         "at every horizon")
    ap.add_argument("--variants", type=str, default="")
    ap.add_argument("--datasets", type=str, default="")
    args = ap.parse_args()

    if args.update_from_logs_only:
        update_md()
        return

    groups = ("long", "pems") if args.group == "all" else (args.group,)
    jobs = []
    if args.full_best:
        base = next(v for v in VARIANTS if v.name == "base")
        for ds in DATASETS:
            if ds.group not in groups or (args.datasets and ds.name not in args.datasets.split(",")):
                continue
            sp = SCREEN_PRED[ds.group]
            cands = [(v, parse_log(log_path_for(ds, v, sp))) for v in VARIANTS if ds.group in v.groups]
            done = [(v, r) for v, r in cands if r["status"] == "OK" and r["vali"] is not None]
            if not done:
                print("[WARN] {}: no screened variants yet, skipping".format(ds.name), flush=True)
                continue
            best = min(done, key=lambda t: t[1]["vali"])[0]
            picks = [best]
            if best.name != "base" and base.groups and ds.group in base.groups \
                    and ds.period != PUBLISHED_PERIOD.get(ds.name):
                picks.insert(0, base)
            print("[SELECT] {}: best on validation at pred={} is '{}' -> running {}".format(
                ds.name, sp, best.name, [v.name for v in picks]), flush=True)
            for pred in ds.pred_lens:
                for v in picks:
                    jobs.append((ds, v, pred))
    for ds in ([] if args.full_best else DATASETS):
        if ds.group not in groups or (args.datasets and ds.name not in args.datasets.split(",")):
            continue
        preds = ds.pred_lens if args.full else (SCREEN_PRED[ds.group],)
        for pred in preds:
            for v in VARIANTS:
                if ds.group not in v.groups:
                    continue
                if args.variants and v.name not in args.variants.split(","):
                    continue
                jobs.append((ds, v, pred))

    total = len(jobs)
    for i, (ds, v, pred) in enumerate(jobs, 1):
        if parse_log(log_path_for(ds, v, pred))["status"] == "OK":
            print("[SKIP {}/{}] {} pred={} {}".format(i, total, ds.name, pred, v.name), flush=True)
            continue
        print("[{}/{}]".format(i, total), end=" ", flush=True)
        run_one(ds, v, pred)
        update_md()
    update_md()
    print("[DONE] forecasting ablation complete", flush=True)


if __name__ == "__main__":
    main()
