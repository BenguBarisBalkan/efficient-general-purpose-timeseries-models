"""
Regenerate the headline table in reports/results_sparsetsf_all_tasks.md from the per-task reports.

Reads each results_comparison_<task>.md, pulls the summary figure(s) already computed there, and
writes a single cross-task overview between the AUTO_SUMMARY markers. Pure aggregation — it does not
recompute anything except the imputation means, which it averages from that report's per-cell rows.
It can be re-run any time after a sweep finishes.

A model's cell stays "-" until its task average is complete in the source report, so a partly
finished sweep is never summarised as if it were whole.

    venv/Scripts/python.exe benchmarks/tools/build_task_summary.py
"""

import re
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "reports"
TARGET = RESULTS / "results_sparsetsf_all_tasks.md"
START, END = "<!-- AUTO_SUMMARY_START -->", "<!-- AUTO_SUMMARY_END -->"

NUM = r"-?[0-9]+(?:\.[0-9]+)?"


def read(name):
    p = RESULTS / name
    return p.read_text(encoding="utf-8") if p.exists() else ""


def grab(pattern, text):
    m = re.search(pattern, text)
    return m.group(1) if m else "-"


def bold_avg(model_label, text):
    """`| **<label>** | <avg> | ...` rows written by the PEMS, anomaly and classification runners."""
    return grab(r"\| \*\*" + re.escape(model_label) + r"[^*]*\*\* \| (" + NUM + r")", text)


def imputation_means(text):
    """Mean masked MSE per model over the OK cells, and the number of datasets it covers."""
    cells = re.findall(r"^\| (\S+) \| (\w+) \| [0-9.]+ \| (" + NUM + r") \| " + NUM
                       + r" \| [^|]* \| OK \|$", text, flags=re.M)
    out = {}
    for model in ("SparseTSF", "TimeMixer"):
        vals = [float(m) for ds, mod, m in cells if mod == model]
        dsets = {ds for ds, mod, _ in cells if mod == model}
        out[model] = "%.4f (%d ds)" % (sum(vals) / len(vals), len(dsets)) if vals else "-"
    return out


def zeroshot_avg(text, column):
    """Mean of one model's per-pair averages, only when all six pairs are filled."""
    rows = re.findall(r"^\| (ETT\w+->ETT\w+) \| ([^|]+) \| ([^|]+) \| _", text, flags=re.M)
    vals = []
    for _, sp, tm in rows:
        m = re.match(r"\s*(" + NUM + r") /", (sp, tm)[column])
        if m:
            vals.append(float(m.group(1)))
    return "%.3f" % (sum(vals) / len(vals)) if len(vals) == 6 else "-"


def main():
    rows = [
        "| Task | Datasets | Metric | SparseTSF (local) | TimeMixer (local) | Reference (paper) |",
        "|---|---|---|---:|---:|---:|",
    ]

    m4 = read("results_comparison_m4.md")
    owa = r" \(local\) \| " + NUM + r" \| " + NUM + r" \| (" + NUM + r")"
    rows.append("| Univariate short-term | M4 (6 subsets) | OWA (lower=better) | "
                + grab(r"SparseTSF" + owa, m4) + " | " + grab(r"TimeMixer" + owa, m4)
                + " | _TimeMixer 0.840_ |")

    pems = read("results_comparison_pems.md")
    rows.append("| Multivariate short-term | PEMS03/04/07/08 | MAE (lower) | "
                + bold_avg("SparseTSF (local)", pems) + " | " + bold_avg("TimeMixer (local)", pems)
                + " | _TimeMixer 17.41_ |")

    imp = imputation_means(read("results_comparison_imputation.md"))
    rows.append("| Imputation | ETT x4, Weather, ECL | masked MSE (lower) | "
                + imp["SparseTSF"] + " | " + imp["TimeMixer"] + " | — |")

    fs = read("results_comparison_fewshot.md")
    fs_row = r"\| ETT \(Avg\) \| "
    rows.append("| Few-shot (10% train) | ETT (Avg) | MSE (lower) | "
                + grab(fs_row + r"(" + NUM + r") /", fs) + " | "
                + grab(fs_row + r"[^|]+\| (" + NUM + r") /", fs) + " | _TimeMixer 0.453_ |")

    zs = read("results_comparison_zeroshot.md")
    rows.append("| Zero-shot (transfer) | 6 ETT pairs | MSE (lower) | "
                + zeroshot_avg(zs, 0) + " | " + zeroshot_avg(zs, 1) + " | _TimeMixer 0.467 (avg)_ |")

    an = read("results_comparison_anomaly.md")
    rows.append("| Anomaly detection | PSM, SMD (of 5) | F1 (higher) | "
                + bold_avg("SparseTSF (local)", an) + " | " + bold_avg("TimeMixer (local", an)
                + " | _TimeMixer++ 87.47 (5 ds)_ |")

    cl = read("results_comparison_classification.md")
    rows.append("| Classification | 10 UEA | accuracy (higher) | "
                + bold_avg("SparseTSF (local)", cl) + " | " + bold_avg("TimeMixer (local", cl)
                + " | _TimeMixer++ 75.9_ |")

    body = "\n".join(rows) + "\n"
    text = TARGET.read_text(encoding="utf-8")
    si, ei = text.find(START), text.find(END)
    if si < 0 or ei < 0:
        raise RuntimeError("summary markers missing")
    TARGET.write_text(text[: si + len(START)] + "\n" + body + text[ei:], encoding="utf-8")
    print(body)


if __name__ == "__main__":
    main()
