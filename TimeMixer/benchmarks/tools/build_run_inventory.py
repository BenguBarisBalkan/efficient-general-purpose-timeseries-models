"""
Count every run in logs/run_logs*/ and write the "Runs and coverage" tables into
reports/results_sparsetsf_all_tasks.md (between the AUTO_RUNS markers).

A run counts as finished when its log holds the line the task prints at the end of testing:
`mse:` (forecasting, imputation, energy), two `mse:` lines for zero-shot (source test, then the
transferred target test), `F-score` (anomaly), `accuracy:` (classification) and `test shape:` (M4,
printed just before the forecast is saved). The model is read from the argument list the log
records. Pure log aggregation: nothing is trained, so it can be re-run at any time.

    venv/Scripts/python.exe benchmarks/tools/build_run_inventory.py
"""
import re
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOGS = REPO_ROOT / "logs"
TARGET = REPO_ROOT / "reports" / "results_sparsetsf_all_tasks.md"
START, END = "<!-- AUTO_RUNS_START -->", "<!-- AUTO_RUNS_END -->"
SECTION = "## Runs and coverage"
INSERT_BEFORE = "## Protocol deviations worth stating in a write-up"

MODELS = ["SparseTSF", "TimeMixer", "TimeMixerPP", "SparseTSFPlus"]
LABEL = {"SparseTSF": "SparseTSF", "TimeMixer": "TimeMixer", "TimeMixerPP": "TimeMixer++",
         "SparseTSFPlus": "SparseTSFPlus"}
# Logs written before run.py echoed its arguments fall back to the directory's only model.
DEFAULT_MODEL = {"run_logs": "TimeMixer", "run_logs_pp": "TimeMixerPP"}
MODEL_RE = re.compile(r"model='([A-Za-z+]+)'")

# (task, paper protocol, [(log directory, filename filter or None)])
TASKS = [
    ("Long-term forecasting", "8 datasets × 4 horizons = 32",
     [("run_logs", None), ("run_logs_pp", None), ("run_logs_sparsetsf", r"_sl96_"),
      ("run_logs_plus_fc", r"^(?!PEMS)"), ("run_logs_plus_suite", r"^longterm_")]),
    ("M4 short-term", "6 subsets", [("run_logs_m4", None), ("run_logs_plus_suite", r"^m4_")]),
    ("PEMS short-term", "4 datasets", [("run_logs_pems", None), ("run_logs_plus_fc", r"^PEMS"),
                                       ("run_logs_plus_suite", r"^pems_")]),
    ("Imputation", "6 datasets × 4 mask rates = 24",
     [("run_logs_imputation", None), ("run_logs_plus_imp", None)]),
    ("Few-shot (10% train)", "6 datasets × 4 horizons = 24",
     [("run_logs_fewshot", None), ("run_logs_plus_suite", r"^fewshot_")]),
    ("Zero-shot", "6 pairs × 4 horizons = 24",
     [("run_logs_zeroshot", None), ("run_logs_plus_suite", r"^zeroshot_")]),
    ("Anomaly detection", "5 datasets (4 obtainable)",
     [("run_logs_anomaly", None), ("run_logs_plus_anom", None), ("run_logs_plus_suite", r"^anomaly_")]),
    ("Classification", "10 UEA datasets",
     [("run_logs_classification", None), ("run_logs_plus_cls", None)]),
]
OTHER = [
    ("Energy study (fixed settings for every model, 5 datasets × 4 horizons; SparseTSF and TimeMixer measured twice, July and 2026-10-21)",
     [("run_logs_carbon", None), ("run_logs_plus_suite", r"^energy")]),
    ("Period-length tests", [("run_logs_period_probe", None)]),
    ("First SparseTSFPlus unit tests", [("run_logs_plus_probe", None)]),
]


def finished(folder, text, name=""):
    if folder == "run_logs_plus_suite":   # one folder for every task: dispatch on the file prefix
        folder = {"m4": "run_logs_m4", "zeroshot": "run_logs_zeroshot",
                  "anomaly": "run_logs_anomaly"}.get(name.split("_")[0], folder)
    if folder == "run_logs_m4":
        return "test shape:" in text
    if folder in ("run_logs_anomaly", "run_logs_plus_anom"):
        return "F-score" in text
    if folder in ("run_logs_classification", "run_logs_plus_cls"):
        return "accuracy:" in text
    if folder == "run_logs_zeroshot":
        return len(re.findall(r"^mse:", text, re.M)) >= 2
    return re.search(r"mse:[0-9]", text) is not None


def scan():
    """-> {(folder, model): [(filename, finished), ...]}"""
    out = defaultdict(list)
    for d in sorted(p for p in LOGS.iterdir() if p.is_dir() and p.name.startswith("run_logs")):
        for f in sorted(d.glob("*.log")):
            text = f.read_text(encoding="utf-8", errors="ignore")
            m = MODEL_RE.search(text)
            model = m.group(1) if m else DEFAULT_MODEL.get(d.name, "?")
            out[(d.name, model)].append((f.name, finished(d.name, text, f.name)))
    return out


def count(runs, folder, model, name_filter=None):
    rows = [ok for name, ok in runs.get((folder, model), [])
            if name_filter is None or re.search(name_filter, name)]
    return sum(rows), len(rows) - sum(rows)


def cell(done, bad):
    return "{}{}".format(done, " (+{} unfinished)".format(bad) if bad else "")


def build():
    runs = scan()
    lines = ["**Accuracy runs, by task** (the TimeMixer++ paper's protocol)", "",
             "| Task | Paper protocol | " + " | ".join(LABEL[m] for m in MODELS) + " |",
             "|---|---|" + "---:|" * len(MODELS)]
    for task, protocol, sources in TASKS:
        cells = []
        for model in MODELS:
            done = bad = 0
            for folder, flt in sources:
                d, b = count(runs, folder, model, flt)
                done, bad = done + d, bad + b
            if done + bad == 0:
                cells.append("paper only" if model == "TimeMixerPP" else "—")
                continue
            text = cell(done, bad)
            if task.startswith("Long-term") and model == "SparseTSF":
                d720, _ = count(runs, "run_logs_sparsetsf", model, r"_sl720_")
                text += " (+{} at lookback 720)".format(d720)
            cells.append(text)
        lines.append("| {} | {} | {} |".format(task, protocol, " | ".join(cells)))

    lines += ["", "**Other experiments**", "",
              "| Experiment | " + " | ".join(LABEL[m] for m in MODELS) + " |",
              "|---|" + "---:|" * len(MODELS)]
    for title, sources in OTHER:
        cells = []
        for model in MODELS:
            d = b = 0
            for folder, flt in sources:
                dd, bb = count(runs, folder, model, flt)
                d, b = d + dd, b + bb
            cells.append(cell(d, b) if d + b else "")
        lines.append("| {} | {} |".format(title, " | ".join(cells)))

    lines += ["", "**Totals**", "", "| Model | Finished | Unfinished |", "|---|---:|---|"]
    all_done = all_bad = 0
    for model in MODELS:
        logs = [(f, n, ok) for (f, m), rs in runs.items() if m == model for n, ok in rs]
        done = sum(ok for _, _, ok in logs)
        bad = ["{}/{}".format(f, n) for f, n, ok in logs if not ok]
        all_done, all_bad = all_done + done, all_bad + len(bad)
        lines.append("| {} | {} | {} |".format(
            LABEL[model], done, "{}: {}".format(len(bad), ", ".join("`{}`".format(b) for b in bad))
            if bad else "0"))
    lines.append("| **All** | **{}** | **{}** |".format(all_done, all_bad))
    return "\n".join(lines) + "\n"


INTRO = """Every run in `logs/run_logs*/`, counted from the logs by
`benchmarks/tools/build_run_inventory.py`. The SparseTSFPlus column counts its ablation runs
(unit variants) together with the full-suite runs of its two versions, one fixed configuration
and the best set per task (`results_sparsetsf_plus_suite.md`).
"paper only" means TimeMixer++ is not re-run on that task: it trains stably only at a learning
rate that leaves it undertrained, so its published figures are the fairer reference. "—" means not
run yet. The unfinished runs are ruled out on this laptop (TimeMixer Traffic 336/720, where Traffic
192 alone took 25 h) or never completed (TimeMixer++ Solar 192).
"""


def main():
    body = build()
    text = TARGET.read_text(encoding="utf-8")
    if START not in text:
        i = text.index(INSERT_BEFORE)
        text = text[:i] + SECTION + "\n\n" + INTRO + "\n" + START + "\n" + END + "\n\n" + text[i:]
    si, ei = text.index(START), text.index(END)
    TARGET.write_text(text[: si + len(START)] + "\n" + body + text[ei:], encoding="utf-8")
    print(body)


if __name__ == "__main__":
    main()
