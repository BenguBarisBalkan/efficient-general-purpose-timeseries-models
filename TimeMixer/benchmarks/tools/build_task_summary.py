"""
Regenerate the headline table in reports/results_sparsetsf_all_tasks.md from the per-task reports.

Reads each results_comparison_<task>.md, pulls the summary figure(s) already computed there, and
writes a single cross-task overview between the AUTO_SUMMARY markers. Pure aggregation — it does not
recompute anything, so it can be re-run any time after a sweep finishes.

    venv/Scripts/python.exe benchmarks/tools/build_task_summary.py
"""

import re
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "reports"
TARGET = RESULTS / "results_sparsetsf_all_tasks.md"
START, END = "<!-- AUTO_SUMMARY_START -->", "<!-- AUTO_SUMMARY_END -->"

NUM = r"-?[0-9]+(?:\.[0-9]+)?"
OURS = r"SparseTSF \(ours\)\*\* \| (" + NUM + r")"


def read(name):
    p = RESULTS / name
    return p.read_text(encoding="utf-8") if p.exists() else ""


def grab(pattern, text):
    m = re.search(pattern, text)
    return m.group(1) if m else "-"


def main():
    rows = [
        "| Task | Datasets | Metric | SparseTSF (ours) | Reference (paper) |",
        "|---|---|---|---:|---:|",
    ]

    m4 = grab(r"SparseTSF \(ours\) \| " + NUM + r" \| " + NUM + r" \| (" + NUM + r")",
              read("results_comparison_m4.md"))
    rows.append("| Univariate short-term | M4 (6 subsets) | OWA (lower=better) | "
                + m4 + " | _TimeMixer 0.840_ |")

    pems = grab(OURS, read("results_comparison_pems.md"))
    rows.append("| Multivariate short-term | PEMS03/04/07/08 | MAE (lower) | "
                + pems + " | _TimeMixer 17.41_ |")

    rows.append("| Imputation | ETT x4, Weather, ECL | masked MSE (lower) | "
                "0.0838 (6 ds) | _TimeMixer 0.0586 (5 ds)_ |")

    fs = grab(r"\| ETT \(Avg\) \| (" + NUM + r")", read("results_comparison_fewshot.md"))
    rows.append("| Few-shot (10% train) | ETT (Avg) | MSE (lower) | "
                + fs + " | _TimeMixer 0.453_ |")

    zs_txt = read("results_comparison_zeroshot.md")
    zs = re.findall(r"\| ETT\w+->ETT\w+ \| (" + NUM + r") / " + NUM + r" \|", zs_txt)
    zs_avg = "%.3f" % (sum(map(float, zs)) / len(zs)) if zs else "-"
    rows.append("| Zero-shot (transfer) | 6 ETT pairs | MSE (lower) | "
                + zs_avg + " | _TimeMixer 0.467 (avg)_ |")

    an = grab(OURS, read("results_comparison_anomaly.md"))
    rows.append("| Anomaly detection | PSM, SMD (of 5) | F1 (higher) | "
                + an + " | _TimeMixer++ 87.47 (5 ds)_ |")

    cl = grab(OURS, read("results_comparison_classification.md"))
    rows.append("| Classification | 10 UEA | accuracy (higher) | "
                + cl + " | _TimeMixer++ 75.9_ |")

    body = "\n".join(rows) + "\n"
    text = TARGET.read_text(encoding="utf-8")
    si, ei = text.find(START), text.find(END)
    if si < 0 or ei < 0:
        raise RuntimeError("summary markers missing")
    TARGET.write_text(text[: si + len(START)] + "\n" + body + text[ei:], encoding="utf-8")
    print(body)


if __name__ == "__main__":
    main()
