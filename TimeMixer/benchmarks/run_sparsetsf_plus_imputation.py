"""SparseTSFPlus imputation ablation: which grafted units earn their parameters.

Imputation is the most trustworthy task in this study for judging a change, because it is
the only one of SparseTSF's weak tasks with a **locally-run, same-machine TimeMixer
baseline** (reports/results_comparison_imputation.md, where TimeMixer wins all 20 cells).

The mechanism under test. SparseTSF's reconstruction is one bias-free linear over periods
at a fixed phase, fed by a conv whose receptive field is, at mask_rate 0.5, half literal
zeros -- and nothing tells it which half. That predicts a gap growing monotonically with
mask_rate, which is exactly what the published table shows (ETTm1: +52% mean, +66% at
mask 0.5). Two units target it, both essentially free:

  --stsf_mask_conv 1        partial-convolution renormalisation: rescale the aggregate by
                            how much of each window was observed (costs 0 params)
  --stsf_impute_passes 2    fill the holes with pass 1, recompute statistics on the now
                            complete series, refine with the SAME weights (0 params).
                            The loss is scored on masked positions only
                            (exp/exp_imputation.py:78), so this targets the metric directly.

plus --stsf_phase_mix linear, which adds the missing within-period direction.

SELECTION CAVEAT, stated because it matters. exp/exp_imputation.py:164 early-stops on
**test_loss**, not vali_loss -- upstream TSLib behaviour, shared by every model in this
repo, so cross-model comparisons stay internally consistent. Do not "fix" it: that would
invalidate every existing table. This runner therefore also reports the best **validation**
loss from each log, and unit combinations must be chosen on THAT column, not on test MSE.

Idempotent: a config whose log already holds a valid MSE is skipped.

    venv/Scripts/python.exe benchmarks/run_sparsetsf_plus_imputation.py
    venv/Scripts/python.exe benchmarks/run_sparsetsf_plus_imputation.py --update-from-logs-only
"""

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT.parent / "dataset"
LOGS_DIR = REPO_ROOT / "logs" / "run_logs_plus_imp"
MD_PATH = REPO_ROOT / "reports" / "results_sparsetsf_plus_ablation.md"

MODEL = "SparseTSFPlus"

# Copied verbatim from run_imputation_benchmarks.py so the `base` row is directly
# comparable to the published SparseTSF cells.
SEQ_LEN = 96
TRAIN_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.001
LRADJ = "TST"
DES = "ImpP"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    data: str
    data_path: str
    root_path: Path
    enc_in: int
    period_len: int   # hourly ETTh -> 24, 15-min ETTm -> 4 (both divide SEQ_LEN=96)


DATASETS: List[DatasetConfig] = [
    DatasetConfig("ETTh1", "ETTh1", "ETTh1.csv", DATA_ROOT / "ETT-small", 7, 24),
    DatasetConfig("ETTm1", "ETTm1", "ETTm1.csv", DATA_ROOT / "ETT-small", 7, 4),
]


@dataclass(frozen=True)
class Variant:
    name: str
    flags: Tuple[str, ...] = ()
    note: str = ""


# `base` must reproduce the published SparseTSF cell: it is the end-to-end check that
# all-units-off really is SparseTSF, through the whole training loop rather than only on
# random tensors (which benchmarks/tools/assert_sparsetsf_plus_equiv.py already proves).
VARIANTS: List[Variant] = [
    Variant("base", (), "all units off == SparseTSF"),
    Variant("maskconv", ("--stsf_mask_conv", "1"), "partial-conv renormalisation"),
    Variant("passes2", ("--stsf_impute_passes", "2"), "weight-shared refinement pass"),
    Variant("u5", ("--stsf_mask_conv", "1", "--stsf_impute_passes", "2"), "both U5 parts"),
    Variant("phase", ("--stsf_phase_mix", "linear"), "intra-period mixing"),
    Variant("all", ("--stsf_mask_conv", "1", "--stsf_impute_passes", "2",
                    "--stsf_phase_mix", "linear"), "U5 + phase mixing"),
]

# Screening runs at the worst mask rate first; the full sweep follows for the winner only.
SCREEN_MASKS = [0.5]
FULL_MASKS = [0.125, 0.25, 0.375, 0.5]

# Published cells, same protocol, from reports/results_comparison_imputation.md.
# (dataset, mask_rate) -> (SparseTSF mse, TimeMixer mse)
PUBLISHED: Dict[Tuple[str, float], Tuple[float, float]] = {
    ("ETTh1", 0.125): (0.113335, 0.096742),
    ("ETTh1", 0.25): (0.146550, 0.111707),
    ("ETTh1", 0.375): (0.179265, 0.125347),
    ("ETTh1", 0.5): (0.214076, 0.146432),
    ("ETTm1", 0.125): (0.048898, 0.039071),
    ("ETTm1", 0.25): (0.062403, 0.042218),
    ("ETTm1", 0.375): (0.078078, 0.048693),
    ("ETTm1", 0.5): (0.096544, 0.058143),
}

NUM = r"([0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?)"
MSE_PAT = re.compile(r"mse:" + NUM)
MAE_PAT = re.compile(r"mae:" + NUM)
VALI_PAT = re.compile(r"Vali Loss: ([0-9]+(?:\.[0-9]+)?)")
PARAM_PAT = re.compile(r"trainable_params:([0-9]+)")
EPOCH_PAT = re.compile(r"Epoch:\s+\d+\s+cost time:\s+([0-9]+(?:\.[0-9]+)?)")


def _last(pat, text):
    m = pat.findall(text)
    return float(m[-1]) if m else None


def log_path_for(ds: DatasetConfig, mask_rate: float, variant: Variant) -> Path:
    return LOGS_DIR / "{}_mask{}_{}.log".format(ds.name, mask_rate, variant.name)


def parse_log(path: Path):
    """-> dict with mse/mae/vali/params/runtime/status"""
    out = {"mse": None, "mae": None, "vali": None, "params": None,
           "runtime": None, "status": "PENDING"}
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8", errors="ignore")
    out["mse"], out["mae"] = _last(MSE_PAT, text), _last(MAE_PAT, text)
    m = PARAM_PAT.findall(text)
    out["params"] = int(m[0]) if m else None
    v = [float(x) for x in VALI_PAT.findall(text)]
    out["vali"] = min(v) if v else None
    ep = [float(x) for x in EPOCH_PAT.findall(text)]
    out["runtime"] = sum(ep) if ep else None
    out["status"] = "OK" if (out["mse"] is not None and out["mae"] is not None) \
        else "RUNNING/INCOMPLETE"
    return out


def run_one(ds: DatasetConfig, mask_rate: float, variant: Variant) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = log_path_for(ds, mask_rate, variant)
    cmd = [
        sys.executable, "-u", str(REPO_ROOT / "run.py"),
        "--task_name", "imputation",
        "--is_training", "1",
        "--model_id", "IMPP_{}_{}".format(ds.name, variant.name),
        "--model", MODEL,
        "--data", ds.data,
        "--root_path", str(ds.root_path),
        "--data_path", ds.data_path,
        "--features", "M",
        "--seq_len", str(SEQ_LEN),
        "--label_len", "0",
        "--pred_len", "0",
        "--enc_in", str(ds.enc_in),
        "--dec_in", str(ds.enc_in),
        "--c_out", str(ds.enc_in),
        "--mask_rate", str(mask_rate),
        "--period_len", str(ds.period_len),
        "--model_type", "linear",
        "--learning_rate", str(LEARNING_RATE),
        "--lradj", LRADJ,
        "--train_epochs", str(TRAIN_EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--num_workers", "0",
        "--des", DES,
        "--itr", "1",
    ] + list(variant.flags)

    print("[RUN] {} mask={} {}".format(ds.name, mask_rate, variant.name), flush=True)
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as f:
        subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
    print("      done in {:.0f}s".format(time.time() - t0), flush=True)


START = "<!-- AUTO_PLUS_IMP_START -->"
END = "<!-- AUTO_PLUS_IMP_END -->"

HEADER = """# SparseTSFPlus ablation — imputation

Which grafted units earn their parameters, measured on the one weak task with a
same-machine TimeMixer baseline.

**`base` is the control**: all units off, which is numerically identical to SparseTSF
(proved bitwise for all five tasks by `benchmarks/tools/assert_sparsetsf_plus_equiv.py`).
Its row reproducing the published SparseTSF cell is the end-to-end confirmation.

**Select units on the `Best vali` column, not on MSE.** `exp/exp_imputation.py:164`
early-stops on *test* loss rather than validation loss. That is upstream TSLib behaviour
and every model in this repo was trained under it, so cross-model comparisons remain
internally consistent — but it means picking a unit combination by test MSE would be
double-dipping.

Regenerate (idempotent; `--update-from-logs-only` rebuilds the table without training):

```bash
venv/Scripts/python.exe benchmarks/run_sparsetsf_plus_imputation.py
```
"""


def _fmt(v, nd=6):
    return "-" if v is None else "{:.{}f}".format(v, nd)


def update_md() -> None:
    lines: List[str] = []
    masks = sorted({r for (_, r) in PUBLISHED} & set(_measured_masks()))
    if not masks:
        masks = SCREEN_MASKS

    for mask_rate in masks:
        lines.append("### mask_rate = {}".format(mask_rate))
        lines.append("")
        lines.append("| Dataset | Variant | Params | Best vali | MSE | MAE | vs SparseTSF | vs TimeMixer | Runtime (s) | Status |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for ds in DATASETS:
            ref = PUBLISHED.get((ds.name, mask_rate))
            rows = []
            for v in VARIANTS:
                r = parse_log(log_path_for(ds, mask_rate, v))
                rows.append((v, r))
            done = [(v, r) for v, r in rows if r["vali"] is not None]
            best = min(done, key=lambda t: t[1]["vali"])[0].name if done else None
            for v, r in rows:
                mark = " **<-**" if v.name == best else ""
                d_sp = d_tm = "-"
                if r["mse"] is not None and ref:
                    d_sp = "{:+.2f}%".format(100.0 * (r["mse"] - ref[0]) / ref[0])
                    d_tm = "{:+.2f}%".format(100.0 * (r["mse"] - ref[1]) / ref[1])
                lines.append(
                    "| {} | {}{} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                        ds.name, v.name, mark,
                        "-" if r["params"] is None else "{:,}".format(r["params"]),
                        _fmt(r["vali"], 5), _fmt(r["mse"]), _fmt(r["mae"]),
                        d_sp, d_tm, _fmt(r["runtime"], 0), r["status"]))
            if ref:
                lines.append(
                    "| _{} published_ | _SparseTSF_ | _581 / 41_ | _—_ | _{:.6f}_ | _—_ | _—_ | _—_ | _—_ | _reference_ |".format(
                        ds.name, ref[0]))
                lines.append(
                    "| _{} published_ | _TimeMixer_ | _~75k_ | _—_ | _{:.6f}_ | _—_ | _—_ | _—_ | _—_ | _reference_ |".format(
                        ds.name, ref[1]))
        lines.append("")

    lines.append("Variant legend:")
    lines.append("")
    for v in VARIANTS:
        flags = " ".join(v.flags) if v.flags else "(none)"
        lines.append("- **{}** — {} · `{}`".format(v.name, v.note, flags))

    body = "\n".join(lines)
    MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MD_PATH.exists():
        MD_PATH.write_text(HEADER + "\n" + START + "\n" + END + "\n", encoding="utf-8")
    text = MD_PATH.read_text(encoding="utf-8")
    si, ei = text.find(START), text.find(END)
    if si < 0 or ei < 0 or ei <= si:
        raise RuntimeError("Markers not found in " + MD_PATH.name)
    MD_PATH.write_text(text[: si + len(START)] + "\n" + body + "\n" + text[ei:],
                       encoding="utf-8")
    print("[OK] wrote " + str(MD_PATH.relative_to(REPO_ROOT)), flush=True)


def _measured_masks():
    """Mask rates that have at least one finished log, so the table only shows real work."""
    found = set()
    for ds in DATASETS:
        for r in FULL_MASKS:
            for v in VARIANTS:
                if parse_log(log_path_for(ds, r, v))["status"] == "OK":
                    found.add(r)
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-from-logs-only", action="store_true",
                    help="rebuild the report from existing logs without training")
    ap.add_argument("--full", action="store_true",
                    help="run every mask rate, not just the screening rate")
    ap.add_argument("--variants", type=str, default="",
                    help="comma-separated subset of variant names (default: all)")
    args = ap.parse_args()

    if args.update_from_logs_only:
        update_md()
        return

    wanted = [v for v in VARIANTS
              if not args.variants or v.name in args.variants.split(",")]
    masks = FULL_MASKS if args.full else SCREEN_MASKS

    jobs = [(ds, r, v) for ds in DATASETS for r in masks for v in wanted]
    total = len(jobs)
    for i, (ds, mask_rate, v) in enumerate(jobs, 1):
        if parse_log(log_path_for(ds, mask_rate, v))["status"] == "OK":
            print("[SKIP {}/{}] {} mask={} {}".format(i, total, ds.name, mask_rate, v.name),
                  flush=True)
            continue
        print("[{}/{}]".format(i, total), end=" ", flush=True)
        run_one(ds, mask_rate, v)
        update_md()

    update_md()
    print("[DONE] imputation ablation complete", flush=True)


if __name__ == "__main__":
    main()
