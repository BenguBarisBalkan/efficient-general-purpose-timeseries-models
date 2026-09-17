"""Period-length sensitivity probe for SparseTSF — diagnostics, no new model code.

Several of SparseTSF's task gaps may be a `period_len` mis-specification rather than a
missing architectural component:

  * Weather uses period_len=4, i.e. 40 minutes at 10-min resolution — not a physical cycle.
  * PEMS uses period_len=12 with pred_len=12, so seg_num_y = 1 and the forecast linear is
    nn.Linear(8, 1, bias=False): EIGHT parameters for the whole model.

Before attributing either gap to a new unit, establish how much a plain hyperparameter
sweep recovers. This probe uses ONLY existing run.py flags, so it needs no model changes.
Every other hyperparameter is copied verbatim from the runner that produced the published
cell, which means the baseline period_len row must REPRODUCE that cell — a built-in check
that the probe is wired correctly.

Idempotent: a config whose log already holds valid metrics is skipped.

    venv/Scripts/python.exe benchmarks/run_period_sensitivity.py
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
LOGS_DIR = REPO_ROOT / "logs" / "run_logs_period_probe"
MD_PATH = REPO_ROOT / "reports" / "results_period_sensitivity.md"

MODEL = "SparseTSF"

# ── forecasting probe: hyperparameters copied from run_sparsetsf_benchmarks.py ────────
FC_LR = 0.02
FC_BATCH = 256
FC_EPOCHS = 30
FC_PATIENCE = 5
FC_LRADJ = "type3"
FC_SEQ_LEN = 96
FC_PRED_LEN = 96

# ── PEMS probe: hyperparameters copied from run_pems_benchmarks.py ───────────────────
PEMS_LR = 0.01
PEMS_BATCH = 16
PEMS_EPOCHS = 10
PEMS_PATIENCE = 5
PEMS_SEQ_LEN = 96
PEMS_PRED_LEN = 12


@dataclass(frozen=True)
class FcProbe:
    name: str
    data: str
    data_path: str
    root_path: Path
    enc_in: int
    factor: int
    baseline_period: int
    periods: Tuple[int, ...]
    # the published cell this probe's baseline row must reproduce (MSE, MAE)
    published: Tuple[float, float]


# period_len must divide BOTH seq_len(96) and pred_len(96); all of 4/8/12/24/48 do.
# ETTh1 is the GUARD row: SparseTSF already beats TimeMixer there, so it must not move.
FC_PROBES: List[FcProbe] = [
    FcProbe("Weather", "custom", "weather.csv", DATA_ROOT / "weather", 21, 3,
            4, (4, 8, 12, 24, 48), (0.198635, 0.238264)),
    FcProbe("ETTm1", "ETTm1", "ETTm1.csv", DATA_ROOT / "ETT-small", 7, 1,
            4, (4, 8, 12, 24, 48), (0.362121, 0.378584)),
    FcProbe("ETTh1", "ETTh1", "ETTh1.csv", DATA_ROOT / "ETT-small", 7, 1,
            24, (24, 12, 48), (0.396635, 0.403991)),
]


@dataclass(frozen=True)
class PemsProbe:
    name: str
    data_path: str
    enc_in: int
    baseline_period: int
    periods: Tuple[int, ...]


# period_len must divide gcd(96, 12) = 12.  PEMS08 is the smallest (C=170) hence fastest.
PEMS_PROBES: List[PemsProbe] = [
    PemsProbe("PEMS08", "PEMS08.npz", 170, 12, (12, 6, 4, 3, 2, 1)),
]

NUM = r"([0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?)"
MSE_PAT = re.compile(r"mse:" + NUM)
MAE_PAT = re.compile(r"mae:" + NUM)
RMSE_PAT = re.compile(r"RMSE:" + NUM)
MAPE_PAT = re.compile(r"MAPE:" + NUM)
EPOCH_PAT = re.compile(r"Epoch:\s+\d+\s+cost time:\s+([0-9]+(?:\.[0-9]+)?)")
VALI_PAT = re.compile(r"Vali Loss: ([0-9]+(?:\.[0-9]+)?)")


def _last(pat, text):
    m = pat.findall(text)
    return float(m[-1]) if m else None


def _best_vali(text):
    """Lowest validation loss seen. period_len must be selected on THIS, not on test.

    exp_long_term_forecasting.py:222 early-stops on vali_loss, so the per-epoch
    validation loss in the log is a legitimate, leakage-free selection signal.
    """
    v = [float(x) for x in VALI_PAT.findall(text)]
    return min(v) if v else None


def _runtime(text):
    ep = [float(x) for x in EPOCH_PAT.findall(text)]
    return sum(ep) if ep else None


def fc_log_path(name: str, period: int) -> Path:
    return LOGS_DIR / "FC_{}_p{}.log".format(name, period)


def pems_log_path(name: str, period: int) -> Path:
    return LOGS_DIR / "PEMS_{}_p{}.log".format(name, period)


def parse_fc(path: Path):
    """-> (mse, mae, best_vali, runtime_s, status)"""
    if not path.exists():
        return None, None, None, None, "PENDING"
    text = path.read_text(encoding="utf-8", errors="ignore")
    mse, mae = _last(MSE_PAT, text), _last(MAE_PAT, text)
    if mse is None or mae is None:
        return None, None, None, None, "RUNNING/INCOMPLETE"
    return mse, mae, _best_vali(text), _runtime(text), "OK"


def parse_pems(path: Path):
    """-> (mae, mape_pct, rmse, best_vali, runtime_s, status)"""
    if not path.exists():
        return None, None, None, None, None, "PENDING"
    text = path.read_text(encoding="utf-8", errors="ignore")
    mae, rmse, mape = _last(MAE_PAT, text), _last(RMSE_PAT, text), _last(MAPE_PAT, text)
    if mae is None or rmse is None or mape is None:
        return None, None, None, None, None, "RUNNING/INCOMPLETE"
    # utils.metrics.MAPE returns a fraction -> report as a percentage like the paper.
    return mae, mape * 100.0, rmse, _best_vali(text), _runtime(text), "OK"


def _run(cmd, log_path: Path, label: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    print("[RUN] " + label, flush=True)
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as f:
        subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
    print("      done in {:.0f}s".format(time.time() - t0), flush=True)


def run_fc(p: FcProbe, period: int) -> None:
    _run([
        sys.executable, "-u", str(REPO_ROOT / "run.py"),
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--model_id", "PROBE_{}_p{}".format(p.name, period),
        "--model", MODEL,
        "--data", p.data,
        "--root_path", str(p.root_path),
        "--data_path", p.data_path,
        "--features", "M",
        "--seq_len", str(FC_SEQ_LEN),
        "--label_len", "0",
        "--pred_len", str(FC_PRED_LEN),
        "--enc_in", str(p.enc_in),
        "--dec_in", str(p.enc_in),
        "--c_out", str(p.enc_in),
        "--period_len", str(period),
        "--model_type", "linear",
        "--factor", str(p.factor),
        "--learning_rate", str(FC_LR),
        "--lradj", FC_LRADJ,
        "--train_epochs", str(FC_EPOCHS),
        "--patience", str(FC_PATIENCE),
        "--batch_size", str(FC_BATCH),
        "--num_workers", "0",
        "--des", "PeriodProbe",
        "--itr", "1",
    ], fc_log_path(p.name, period), "{} period_len={}".format(p.name, period))


def run_pems(p: PemsProbe, period: int) -> None:
    _run([
        sys.executable, "-u", str(REPO_ROOT / "run.py"),
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--model_id", "PROBE_{}_p{}".format(p.name, period),
        "--model", MODEL,
        "--data", "PEMS",
        "--root_path", str(DATA_ROOT / "PEMS"),
        "--data_path", p.data_path,
        "--features", "M",
        "--seq_len", str(PEMS_SEQ_LEN),
        "--label_len", "0",
        "--pred_len", str(PEMS_PRED_LEN),
        "--enc_in", str(p.enc_in),
        "--dec_in", str(p.enc_in),
        "--c_out", str(p.enc_in),
        "--period_len", str(period),
        "--model_type", "linear",
        "--use_norm", "0",
        "--learning_rate", str(PEMS_LR),
        "--train_epochs", str(PEMS_EPOCHS),
        "--patience", str(PEMS_PATIENCE),
        "--batch_size", str(PEMS_BATCH),
        "--num_workers", "0",
        "--des", "PeriodProbe",
        "--itr", "1",
    ], pems_log_path(p.name, period), "{} period_len={}".format(p.name, period))


def n_params_fc(seq_len: int, pred_len: int, period: int) -> int:
    """SparseTSF 'linear': conv taps + one bias-free (seq/p -> pred/p) matrix."""
    return (1 + 2 * (period // 2)) + (seq_len // period) * (pred_len // period)


START = "<!-- AUTO_PERIOD_PROBE_START -->"
END = "<!-- AUTO_PERIOD_PROBE_END -->"

HEADER = """# Period-length sensitivity of SparseTSF

Diagnostics run **before** any architectural change, to separate "SparseTSF is missing a
component" from "SparseTSF's `period_len` was mis-specified for this dataset". Uses only
existing `run.py` flags — no model code was modified.

Every non-period hyperparameter is copied verbatim from the runner that produced the
published cell (`run_sparsetsf_benchmarks.py` for forecasting, `run_pems_benchmarks.py`
for PEMS), so the **baseline row must reproduce the published number**. A mismatch there
means the probe is mis-wired, not that the period matters.

Regenerate:

```bash
venv/Scripts/python.exe benchmarks/run_period_sensitivity.py
```
"""


def _fmt(v, nd=6):
    return "-" if v is None else "{:.{}f}".format(v, nd)


def update_md() -> None:
    lines = []

    lines.append("### Long-term forecasting (seq_len=96, pred_len=96)")
    lines.append("")
    lines.append("`*` marks the published baseline period; **<-** marks the period with the")
    lines.append("lowest **validation** loss, which is the only leakage-free way to select it.")
    lines.append("**Params** is the whole model.")
    lines.append("")
    lines.append("| Dataset | period_len | Params | Best vali | MSE | MAE | vs baseline MSE | Runtime (s) | Status |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for p in FC_PROBES:
        base_mse = parse_fc(fc_log_path(p.name, p.baseline_period))[0]
        valis = {q: parse_fc(fc_log_path(p.name, q))[2] for q in p.periods}
        done = {q: v for q, v in valis.items() if v is not None}
        chosen = min(done, key=done.get) if done else None
        for period in p.periods:
            mse, mae, vali, rt, status = parse_fc(fc_log_path(p.name, period))
            mark = "*" if period == p.baseline_period else ""
            if period == chosen:
                mark += " **<-**"
            delta = "-"
            if period == p.baseline_period:
                delta = "—"
            elif mse is not None and base_mse:
                delta = "{:+.2f}%".format(100.0 * (mse - base_mse) / base_mse)
            lines.append(
                "| {} | {}{} | {:,} | {} | {} | {} | {} | {} | {} |".format(
                    p.name, period, mark,
                    n_params_fc(FC_SEQ_LEN, FC_PRED_LEN, period),
                    _fmt(vali, 4), _fmt(mse), _fmt(mae), delta, _fmt(rt, 0), status))
        if base_mse is not None:
            ok = abs(base_mse - p.published[0]) < 5e-4
            lines.append(
                "| _{} published_ | _{}_ | _—_ | _—_ | _{:.6f}_ | _{:.6f}_ | _{}_ | _—_ | _reference_ |".format(
                    p.name, p.baseline_period, p.published[0], p.published[1],
                    "REPRODUCED" if ok else "MISMATCH"))
    lines.append("")

    lines.append("### PEMS multivariate short-term (seq_len=96, pred_len=12)")
    lines.append("")
    lines.append("At the published `period_len=12`, `seg_num_y = 12/12 = 1`, so the forecast")
    lines.append("linear is `nn.Linear(8, 1, bias=False)` — **8 parameters**.")
    lines.append("")
    lines.append("| Dataset | period_len | Params | Best vali | MAE | MAPE (%) | RMSE | Runtime (s) | Status |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for p in PEMS_PROBES:
        valis = {q: parse_pems(pems_log_path(p.name, q))[3] for q in p.periods}
        done = {q: v for q, v in valis.items() if v is not None}
        chosen = min(done, key=done.get) if done else None
        for period in p.periods:
            mae, mape, rmse, vali, rt, status = parse_pems(pems_log_path(p.name, period))
            mark = "*" if period == p.baseline_period else ""
            if period == chosen:
                mark += " **<-**"
            lines.append(
                "| {} | {}{} | {:,} | {} | {} | {} | {} | {} | {} |".format(
                    p.name, period, mark,
                    n_params_fc(PEMS_SEQ_LEN, PEMS_PRED_LEN, period),
                    _fmt(vali, 3), _fmt(mae, 2), _fmt(mape, 2), _fmt(rmse, 2),
                    _fmt(rt, 0), status))
    lines.append("")
    lines.append("| Published reference | MAE | MAPE (%) | RMSE |")
    lines.append("|---|---:|---:|---:|")
    lines.append("| _TimeMixer (paper, 4-dataset avg)_ | _17.41_ | _10.59_ | _28.01_ |")
    lines.append("| _TimeMixer++ (paper, 4-dataset avg)_ | _15.91_ | _10.08_ | _27.06_ |")
    lines.append("")
    lines.append("Note: the published references are averages over PEMS03/04/07/08, while this")
    lines.append("probe runs PEMS08 only — read the *trend across period_len*, not the absolute gap.")

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
    jobs = []
    for p in FC_PROBES:
        for period in p.periods:
            jobs.append(("fc", p, period))
    for p in PEMS_PROBES:
        for period in p.periods:
            jobs.append(("pems", p, period))

    total = len(jobs)
    for i, (kind, probe, period) in enumerate(jobs, 1):
        if kind == "fc":
            status = parse_fc(fc_log_path(probe.name, period))[-1]
        else:
            status = parse_pems(pems_log_path(probe.name, period))[-1]
        if status == "OK":
            print("[SKIP {}/{}] {} p={} (done)".format(i, total, probe.name, period), flush=True)
            continue
        print("[{}/{}]".format(i, total), end=" ", flush=True)
        if kind == "fc":
            run_fc(probe, period)
        else:
            run_pems(probe, period)
        update_md()

    update_md()
    print("[DONE] period sensitivity probe complete", flush=True)


if __name__ == "__main__":
    main()
