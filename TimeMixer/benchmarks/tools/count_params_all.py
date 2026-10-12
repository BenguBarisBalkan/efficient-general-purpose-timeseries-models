"""Trainable parameters of every model on every task and dataset that has been run.

Each count rebuilds the model from the argument Namespace that its own run log records, so the
figure belongs to exactly the configuration that produced the result. Where the task code changes
arguments after that line is printed, the same change is applied here: classification takes
seq_len / enc_in / num_class from the dataset (exp_classification.py), and M4 takes seq_len /
pred_len from the subset's horizon (exp_short_term_forecasting.py).

Forecasting counts depend on pred_len, and imputation counts can depend on the dataset, so a cell
shows the range over the runs it covers. Few-shot and zero-shot reuse the long-term architectures
unchanged and are not listed separately. SparseTSFPlus prints its own count into every log; its
figures are in the results_sparsetsf_plus_*.md reports.

    venv/Scripts/python.exe benchmarks/tools/count_params_all.py   # writes reports/results_param_counts.md
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from data_provider.m4 import M4Meta  # noqa: E402
from models import SparseTSF, TimeMixer, TimeMixerPP  # noqa: E402

LOGS = REPO_ROOT / "logs"
WORKSPACE = REPO_ROOT.parent
MD_PATH = REPO_ROOT / "reports" / "results_param_counts.md"
MODULES = {"TimeMixer": TimeMixer, "TimeMixerPP": TimeMixerPP, "SparseTSF": SparseTSF}
MODEL_ORDER = ["SparseTSF", "TimeMixer", "TimeMixerPP"]
LABEL = {"SparseTSF": "SparseTSF", "TimeMixer": "TimeMixer", "TimeMixerPP": "TimeMixer++"}

# (section title, log directory, filename regex whose first group is the dataset, row filter)
SOURCES = [
    ("Long-term forecasting (seq_len 96)", "run_logs", r"^(.+)_pred\d+\.log$", None),
    ("Long-term forecasting (seq_len 96)", "run_logs_pp", r"^(.+)_pred\d+\.log$", None),
    ("Long-term forecasting (seq_len 96)", "run_logs_sparsetsf", r"^(.+)_sl96_pred\d+\.log$", None),
    ("Short-term forecasting, M4", "run_logs_m4", r"^[A-Za-z]+_(.+)\.log$", None),
    ("Short-term forecasting, PEMS", "run_logs_pems", r"^[A-Za-z]+_(.+)\.log$", None),
    ("Imputation", "run_logs_imputation", r"^[A-Za-z]+_(.+)_mask[0-9.]+\.log$", None),
    ("Anomaly detection", "run_logs_anomaly", r"^[A-Za-z]+_(.+)\.log$", None),
    ("Classification", "run_logs_classification", r"^[A-Za-z]+_(.+)\.log$", None),
]

NAMESPACE = re.compile(r"^Namespace\(.*\)\s*$", re.M)


def load_args(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    m = NAMESPACE.search(text)
    if not m:
        return None
    return eval(m.group(0), {"Namespace": argparse.Namespace})  # our own log, written by run.py


def real_path(p):
    """The dataset folder, rebuilt from its last component: logs record the path either scrubbed
    to `<workspace>` or in the console's code page, which can mangle non-ASCII folder names."""
    return str(WORKSPACE / "dataset" / re.split(r"[\\/]+", p.rstrip("\\/"))[-1])


_UEA_CACHE = {}


def apply_task_changes(args):
    if args.task_name == "classification":
        key = real_path(args.root_path)
        if key not in _UEA_CACHE:
            from data_provider.data_factory import data_provider
            args.root_path = key
            train, _ = data_provider(args, "TRAIN")
            test, _ = data_provider(args, "TEST")
            _UEA_CACHE[key] = (max(train.max_seq_len, test.max_seq_len),
                               train.feature_df.shape[1], len(train.class_names))
        args.seq_len, args.enc_in, args.num_class = _UEA_CACHE[key]
        args.pred_len = 0
    if args.task_name == "short_term_forecast" and args.data == "m4":
        args.pred_len = M4Meta.horizons_map[args.seasonal_patterns]
        args.seq_len = 2 * args.pred_len
        args.label_len = args.pred_len
        args.frequency_map = M4Meta.frequency_map[args.seasonal_patterns]
    return args


def count(args):
    model = MODULES[args.model].Model(args)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fmt(vals):
    if not vals:
        return "—"
    lo, hi = min(vals), max(vals)
    return "{:,}".format(lo) if lo == hi else "{:,}–{:,}".format(lo, hi)


def main():
    table = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # section -> ds -> model
    order = defaultdict(list)
    unrecorded = []
    for section, folder, pattern, _ in SOURCES:
        for path in sorted((LOGS / folder).glob("*.log")):
            m = re.match(pattern, path.name)
            if not m:
                continue
            args = load_args(path)
            if args is None:
                unrecorded.append("{}/{}".format(folder, path.name))
                continue
            if args.model not in MODULES:
                continue
            ds = m.group(1)
            n = count(apply_task_changes(args))
            table[section][ds][args.model].append(n)
            if ds not in order[section]:
                order[section].append(ds)
            print("{:<38} {:<22} {:<12} {:>12,}".format(section[:38], ds, args.model, n), flush=True)

    out = [
        "# Trainable parameters per task",
        "",
        "Every count is rebuilt from the argument list in the run's own log "
        "(`benchmarks/tools/count_params_all.py`), so it matches the configuration that produced the "
        "result. A range covers the horizons (forecasting) or mask rates (imputation) of that dataset. "
        "Few-shot and zero-shot reuse the long-term architectures unchanged. SparseTSFPlus counts are "
        "in its own reports. TimeMixer on anomaly detection and classification has no published "
        "configuration; it uses its ETTh1 long-term architecture (classification additionally needs "
        "channel mixing, which TimeMixer's classification code requires).",
        "",
        "```bash",
        "venv/Scripts/python.exe benchmarks/tools/count_params_all.py",
        "```",
        "",
    ]
    for section in dict.fromkeys(s for s, *_ in SOURCES):
        models = [mm for mm in MODEL_ORDER if any(table[section][d][mm] for d in order[section])]
        if not models:
            continue
        out += ["## " + section, "",
                "| Dataset | " + " | ".join(LABEL[mm] for mm in models) + " |",
                "|---|" + "---:|" * len(models)]
        for ds in order[section]:
            out.append("| {} | {} |".format(ds, " | ".join(fmt(table[section][ds][mm]) for mm in models)))
        out.append("")
    if unrecorded:
        out += ["Logs that record no argument list, so they are not in the ranges above: "
                + ", ".join("`{}`".format(u) for u in unrecorded) + ".", ""]
    MD_PATH.write_text("\n".join(out), encoding="utf-8")
    print("[OK] wrote " + str(MD_PATH.relative_to(REPO_ROOT)))


if __name__ == "__main__":
    main()
