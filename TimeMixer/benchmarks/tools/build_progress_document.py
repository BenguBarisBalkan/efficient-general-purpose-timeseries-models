"""
Build the tables of reports/thesis_progress_report.md from the run logs, and optionally its PDF.

The report's prose is written by hand; every number in its tables comes from here:

  Table 1  the three-model comparison as it stood on 11 October 2026: each log is read from git at
           commit SNAPSHOT (`git show`), so a run counts only if its log existed then
  Table 2  the same comparison from today's logs
  Table 3  the same, plus SparseTSFPlus (fixed configuration and per task) from the suite runner
  Table 4  three-seed means of the headline comparisons (run_seed_repeats.py)
  Table 5  run coverage, then and now; Table 6 trainable parameters; Table 7 energy

Local values (this machine, this protocol) and published values (the TimeMixer and TimeMixer++
papers) sit in separate columns and are never mixed in one cell. Published values are italic.

    venv/Scripts/python.exe benchmarks/tools/build_progress_document.py          # tables only
    venv/Scripts/python.exe benchmarks/tools/build_progress_document.py --pdf    # + PDF (pandoc, Chrome)
"""

import argparse
import functools
import re
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Callable, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
sys.path.insert(0, str(REPO_ROOT))

import run_all_long_term_benchmarks as TMLT  # noqa: E402
import run_all_pp_benchmarks as PP  # noqa: E402
import run_anomaly_benchmarks as AN  # noqa: E402
import run_classification_benchmarks as CLS  # noqa: E402
import run_fewshot_benchmarks as FS  # noqa: E402
import run_imputation_benchmarks as IMP  # noqa: E402
import run_m4_benchmarks as M4  # noqa: E402
import run_pems_benchmarks as PEMS  # noqa: E402
import run_seed_repeats as SEEDS  # noqa: E402
import run_sparsetsf_benchmarks as LT  # noqa: E402
import run_sparsetsf_plus_suite as SUITE  # noqa: E402
import run_zeroshot_benchmarks as ZS  # noqa: E402

MD_PATH = REPO_ROOT / "reports" / "thesis_progress_report.md"
PDF_PATH = REPO_ROOT / "reports" / "thesis_progress_report.pdf"
SNAPSHOT = "72040e6"          # 11 October 2026: start of the current two-week reporting period
SNAPSHOT_DATE = "11 October 2026"

NUM = r"([0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?)"
MSE_PAT = re.compile(r"mse:" + NUM)
MAE_PAT = re.compile(r"mae:" + NUM)
F1_PAT = re.compile(r"F-score\s*:\s*" + NUM)
ACC_PAT = re.compile(r"accuracy:" + NUM)
MODEL_PAT = re.compile(r"model='([A-Za-z+]+)'")


# ── log sources: the working tree, or git at the snapshot commit ─────────────────────────────

class Source:
    def __init__(self, commit: Optional[str] = None):
        self.commit = commit
        self.tree = None
        self.cache: Dict[str, str] = {}
        if commit:
            out = subprocess.run(["git", "ls-tree", "-r", "--name-only", commit, "TimeMixer/logs"],
                                 cwd=str(WORKSPACE), capture_output=True, text=True, check=True)
            self.tree = set(out.stdout.split())

    def read(self, path: Path) -> str:
        key = str(path)
        if key not in self.cache:
            self.cache[key] = self._read(path)
        return self.cache[key]

    def _read(self, path: Path) -> str:
        if self.commit is None:
            return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
        rel = path.resolve().relative_to(WORKSPACE.resolve()).as_posix()
        if rel not in self.tree:
            return ""
        out = subprocess.run(["git", "show", "{}:{}".format(self.commit, rel)], cwd=str(WORKSPACE),
                             capture_output=True, check=True)
        return out.stdout.decode("utf-8", errors="ignore")

    def exists(self, path: Path) -> bool:
        return bool(self.read(path))


def _last(pat, text) -> Optional[float]:
    m = pat.findall(text)
    return float(m[-1]) if m else None


def _mean_all(vals: List[Optional[float]]) -> Optional[float]:
    return None if not vals or any(v is None for v in vals) else sum(vals) / len(vals)


def _partial(vals: List[Optional[float]]) -> Optional[str]:
    """'3/4' when some but not all cells of a dataset exist (its mean is then not shown)."""
    done = sum(v is not None for v in vals)
    return "{}/{}".format(done, len(vals)) if 0 < done < len(vals) else None


@functools.lru_cache(maxsize=None)
def m4_owa(folder: str) -> Optional[float]:
    return SUITE._m4_owa(SUITE.M4_RESULTS / folder)


@functools.lru_cache(maxsize=None)
def suite_tables():
    """The suite runner's per-dataset tables, built once (M4 scoring is slow)."""
    return {"Long-term": SUITE.table_longterm(), "M4": SUITE.table_m4(), "PEMS": SUITE.table_pems(),
            "Imputation": SUITE.table_imputation(), "Few-shot": SUITE.table_fewshot(),
            "Zero-shot": SUITE.table_zeroshot(), "Anomaly": SUITE.table_anomaly(),
            "Classification": SUITE.table_classification()}


# ── per-task, per-dataset local values ───────────────────────────────────────────────────────

LT_LOGS = {"TimeMixer": lambda ds, p: REPO_ROOT / "logs" / "run_logs" / "{}_pred{}.log".format(ds, p),
           "TimeMixer++": lambda ds, p: REPO_ROOT / "logs" / "run_logs_pp" / "{}_pred{}.log".format(ds, p),
           "SparseTSF": lambda ds, p: REPO_ROOT / "logs" / "run_logs_sparsetsf" / "{}_sl96_pred{}.log".format(ds, p)}
MODELS = ["TimeMixer", "TimeMixer++", "SparseTSF"]


class Row:
    """One dataset row: local values per model, published values per model, a note, direction."""

    def __init__(self, name, local, published=None, partial=None, higher_better=False, spec=".4f"):
        self.name, self.local, self.published = name, local, published or {}
        self.partial = partial or {}          # model -> "2/4" when only some cells exist
        self.hb, self.spec = higher_better, spec


def longterm_rows(src: Source) -> List[Row]:
    rows = []
    for ds in TMLT.DATASETS:
        local, partial = {}, {}
        for m in MODELS:
            vals = [_last(MSE_PAT, src.read(LT_LOGS[m](ds.name, p))) for p in LT.PRED_LENS]
            local[m] = _mean_all(vals)
            if _partial(vals):
                partial[m] = _partial(vals)
        pub = {"TimeMixer": mean(TMLT.PAPER[ds.name][p][0] for p in LT.PRED_LENS),
               "TimeMixer++": mean(PP.PAPER[ds.name][p][0] for p in LT.PRED_LENS)}
        rows.append(Row(ds.name, local, pub, partial))
    return rows


def m4_rows(src: Source) -> List[Row]:
    local = {}
    for m, folder in (("TimeMixer", "TimeMixer"), ("SparseTSF", "SparseTSF")):
        logs = [REPO_ROOT / "logs" / "run_logs_m4" / "{}_{}.log".format(m, s) for s in M4.SUBSETS]
        ran = all("test shape:" in src.read(p) for p in logs)
        local[m] = m4_owa(folder) if ran else None
    local["TimeMixer++"] = None
    pub = {k.split(" ")[0]: v["OWA"] for k, v in M4.PAPER_REF.items()}
    return [Row("M4, 6 subsets (OWA)", local, pub, spec=".3f")]


def pems_rows(src: Source) -> List[Row]:
    rows = []
    for ds in PEMS.DATASETS:
        local = {}
        for m in ("TimeMixer", "SparseTSF"):
            t = src.read(PEMS.log_path_for(m, ds))
            local[m] = _last(MAE_PAT, t) if "RMSE:" in t else None
        local["TimeMixer++"] = None
        rows.append(Row(ds.name, local, spec=".2f"))
    avg = {m: _mean_all([r.local[m] for r in rows]) for m in MODELS}
    pub = {k.split(" ")[0]: v["MAE"] for k, v in PEMS.PAPER_REF.items()}
    rows.append(Row("Average", avg, pub, spec=".2f"))
    return rows


def imputation_rows(src: Source) -> List[Row]:
    rows = []
    for ds in IMP.DATASETS:
        local, partial = {"TimeMixer++": None}, {}
        for m in ("TimeMixer", "SparseTSF"):
            vals = [_last(MSE_PAT, src.read(IMP.log_path_for(m, ds, r))) for r in IMP.MASK_RATES]
            local[m] = _mean_all(vals)
            if _partial(vals):
                partial[m] = _partial(vals)
        rows.append(Row(ds.name, local, partial=partial))
    return rows


def fewshot_rows(src: Source) -> List[Row]:
    rows = []
    for ds in FS.DATASETS:
        local, partial = {"TimeMixer++": None}, {}
        for m in ("TimeMixer", "SparseTSF"):
            vals = [_last(MSE_PAT, src.read(FS.log_path_for(ds, p, m))) for p in FS.PRED_LENS]
            local[m] = _mean_all(vals)
            if _partial(vals):
                partial[m] = _partial(vals)
        pub = {}
        if ds.name in FS.PAPER_REF:
            pub = {k.split(" ")[0]: v[0] for k, v in FS.PAPER_REF[ds.name].items()}
        rows.append(Row(ds.name, local, pub, partial))
    ett = [r for r in rows if r.name.startswith("ETT")]
    avg = {m: _mean_all([r.local[m] for r in ett]) for m in MODELS}
    pub = {k.split(" ")[0]: v[0] for k, v in FS.PAPER_REF["ETT (Avg)"].items()}
    rows.insert(len(ett), Row("ETT average", avg, pub))
    return rows


def zeroshot_rows(src: Source) -> List[Row]:
    rows = []
    for a, b in ZS.PAIRS:
        local, partial = {}, {}
        for m in ("TimeMixer", "SparseTSF"):
            vals = []
            for p in ZS.PRED_LENS:
                t = src.read(ZS.log_path_for(a, b, p, m))
                vals.append(_last(MSE_PAT, t) if len(re.findall(r"^mse:", t, re.M)) >= 2 else None)
            local[m] = _mean_all(vals)
            if _partial(vals):
                partial[m] = _partial(vals)
        local["TimeMixer++"] = None
        ref = ZS.PAPER_REF["{}->{}".format(a, b)]
        rows.append(Row("{} → {}".format(a, b), local, {k: v[0] for k, v in ref.items()}, partial))
    return rows


def anomaly_rows(src: Source) -> List[Row]:
    rows = []
    for ds in AN.DATASETS:
        local = {}
        for m in ("TimeMixer", "SparseTSF"):
            f1 = _last(F1_PAT, src.read(AN.log_path_for(ds, m)))
            local[m] = None if f1 is None else f1 * 100
        local["TimeMixer++"] = None
        name = ds.name if AN.available(ds) else ds.name + " (not obtainable)"
        rows.append(Row(name, local, {"TimeMixer++": AN.PAPER_REF_F1[ds.name]}, higher_better=True,
                        spec=".2f"))
    return rows


def classification_rows(src: Source) -> List[Row]:
    rows = []
    for ds in CLS.DATASETS:
        local = {}
        for m in ("TimeMixer", "SparseTSF"):
            acc = _last(ACC_PAT, src.read(CLS.log_path_for(ds, m)))
            local[m] = None if acc is None else acc * 100
        local["TimeMixer++"] = None
        rows.append(Row(ds, local, higher_better=True, spec=".2f"))
    avg = {m: _mean_all([r.local[m] for r in rows]) for m in MODELS}
    rows.append(Row("Average", avg, {"TimeMixer++": CLS.PAPER_REF["TimeMixer++ (paper)"]},
                    higher_better=True, spec=".2f"))
    return rows


# Group rows show a short label; the full metric definitions are in the report's Section 2.
SHORT = {}

TASKS = [
    ("Long-term forecasting — MSE ↓, mean of 4 horizons (96–720), input 96", longterm_rows),
    ("Univariate short-term forecasting (M4) — OWA ↓", m4_rows),
    ("Multivariate short-term forecasting (PEMS) — MAE ↓, input 96 → 12", pems_rows),
    ("Imputation — masked MSE ↓, mean of 4 mask rates, input 96", imputation_rows),
    ("Few-shot forecasting (10% of training data) — MSE ↓, mean of 4 horizons", fewshot_rows),
    ("Zero-shot transfer (train A, test B) — MSE ↓, mean of 4 horizons", zeroshot_rows),
    ("Anomaly detection — point-adjusted F1 (%) ↑", anomaly_rows),
    ("Classification (UEA) — accuracy (%) ↑", classification_rows),
]
SHORT.update(zip([t for t, _ in TASKS],
                 ["Long-term (MSE ↓)", "M4 (OWA ↓)", "PEMS (MAE ↓)", "Imputation (MSE ↓)",
                  "Few-shot (MSE ↓)", "Zero-shot (MSE ↓)", "Anomaly (F1 % ↑)",
                  "Classification (accuracy % ↑)"]))


# ── tables ───────────────────────────────────────────────────────────────────────────────────

def _cell(v, spec, bold=False, italic=False, partial=None):
    if partial:
        return "partial ({})".format(partial)
    if v is None:
        return "—"
    s = format(v, spec)
    if bold:
        return "**{}**".format(s)
    if italic:
        return "*{}*".format(s)
    return s


def _best(row: Row, cols: List[str]) -> Optional[str]:
    vals = {c: row.local.get(c) for c in cols if row.local.get(c) is not None and c not in row.partial}
    if len(vals) < 2:
        return None
    return (max if row.hb else min)(vals, key=vals.get)


def comparison_table(src: Source, extra: Optional[Dict[str, Dict[str, Callable]]] = None) -> List[str]:
    """Rows grouped by task. `extra` adds SparseTSFPlus columns: {label: {task title: fn(rows)}}."""
    extra = extra or {}
    local_cols = MODELS + list(extra)
    head = ["Dataset"] + local_cols + ["TimeMixer (paper)", "TimeMixer++ (paper)"]
    out = ["| " + " | ".join(head) + " |", "|---|" + "---:|" * (len(head) - 1)]
    for title, fn in TASKS:
        rows = fn(src)
        for label, per_task in extra.items():
            values = per_task.get(title, lambda rs: {})(rows)
            for r in rows:
                r.local[label] = values.get(r.name)
        out.append("| **{}** |".format(SHORT[title]) + " |" * (len(head) - 1))
        for r in rows:
            best = _best(r, local_cols)
            cells = [_cell(r.local.get(c), r.spec, bold=(c == best), partial=r.partial.get(c))
                     for c in local_cols]
            cells += [_cell(r.published.get(m), r.spec, italic=True) for m in ("TimeMixer", "TimeMixer++")]
            out.append("| {} | {} |".format(r.name, " | ".join(cells)))
    return out


def plus_columns() -> Dict[str, Dict[str, Callable]]:
    """SparseTSFPlus values per dataset from the suite runner's own tables (seed 2021)."""
    suite = suite_tables()

    def by_name(key, version):
        rows = dict(suite[key].rows)

        def fn(task_rows):
            out = {}
            for r in task_rows:
                name = r.name.replace(" (not obtainable)", "")
                if key == "M4":
                    out[r.name] = rows["M4 (6 subsets)"].get(version)
                elif name in rows:
                    out[r.name] = rows[name].get(version)
            if key in ("PEMS", "Classification"):
                vals = [out.get(r.name) for r in task_rows if r.name != "Average"]
                out["Average"] = _mean_all(vals)
            if key == "Few-shot":
                vals = [out.get(r.name) for r in task_rows if r.name.startswith("ETT") and r.name != "ETT average"]
                out["ETT average"] = _mean_all(vals)
            return out
        return fn

    titles = {t: t.split(" (")[0].split(" —")[0] for t, _ in TASKS}
    key_of = {"Long-term forecasting": "Long-term", "Univariate short-term forecasting": "M4",
              "Multivariate short-term forecasting": "PEMS", "Imputation": "Imputation",
              "Few-shot forecasting": "Few-shot", "Zero-shot transfer": "Zero-shot",
              "Anomaly detection": "Anomaly", "Classification": "Classification"}
    cols = {}
    for label, version in (("SparseTSFPlus fixed", "fixed"), ("SparseTSFPlus per task", "task")):
        cols[label] = {t: by_name(key_of[titles[t]], version) for t, _ in TASKS}
    return cols


def seeds_table() -> List[str]:
    configs = [("SparseTSF", "SparseTSF"), ("fixed", "SparseTSFPlus fixed"),
               ("task", "SparseTSFPlus per task"), ("TimeMixer", "TimeMixer")]
    seeds = [SEEDS.BASE_SEED] + SEEDS.SEEDS
    out = ["| Task | " + " | ".join(lbl for _, lbl in configs) + " |",
           "|---|" + "---:|" * len(configs)]
    for task in ("longterm", "m4", "pems", "fewshot", "zeroshot", "anomaly", "imputation",
                 "imputation_ecl"):
        metric, hb = SEEDS.METRIC[task]
        spec = ".3f" if task == "m4" else (".2f" if task in ("pems", "anomaly") else ".4f")
        cells = []
        for c, _ in configs:
            if c == "task" and task.startswith("imputation"):
                c = "u5g"          # the per-task imputation form is gated refinement
            vals = [SEEDS.task_value(task, c, s) for s in seeds]
            if any(v is None for v in vals):
                cells.append("—")
            else:
                cells.append("{} ± {}".format(format(mean(vals), spec), format(stdev(vals), spec)))
        out.append("| {} ({} {}) | {} |".format(SEEDS.TITLE[task], metric, "↑" if hb else "↓",
                                                " | ".join(cells)))
    return out



# ── compact tables in the layout of the TimeMixer++ paper's main results ──────────────────────

import run_sparsetsf_plus_classification as PCLS  # noqa: E402
import run_sparsetsf_plus_imputation as PIMP  # noqa: E402

RMSE_PAT = re.compile(r"RMSE:" + NUM)
MAPE_PAT = re.compile(r"MAPE:" + NUM)
ETT = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]
PARTIAL = "partial"


def _pair(src: Source, paths: List[Path], zs: bool = False):
    """Mean (MSE, MAE) over a dataset's cells; PARTIAL if only some finished; None if none did."""
    mses, maes = [], []
    for path in paths:
        t = src.read(path)
        ok = len(re.findall(r"^mse:", t, re.M)) >= 2 if zs else MSE_PAT.search(t) is not None
        mses.append(_last(MSE_PAT, t) if ok else None)
        maes.append(_last(MAE_PAT, t) if ok else None)
    if all(v is None for v in mses):
        return None
    if any(v is None for v in mses):
        return PARTIAL
    return mean(mses), mean(maes)


def _avg_pairs(pairs):
    if all(x is None for x in pairs):
        return None
    if any(x is None or x == PARTIAL for x in pairs):
        return PARTIAL
    return mean(x[0] for x in pairs), mean(x[1] for x in pairs)


def _suite(name: str) -> Path:
    return SUITE.LOGS_DIR / name


def _imp_plus_path(ds_name: str, rate: float) -> Path:
    pds = next(d for d in PIMP.DATASETS if d.name == ds_name)
    pv = {v.name: v for v in PIMP.VARIANTS}[SUITE.REUSE_IMPUTATION["fixed"]]
    return PIMP.report_path_for(pds, rate, pv)


def lt_paths(model: str, ds: str) -> List[Path]:
    if model == "SparseTSFPlus":
        return [_suite("longterm_fixed_{}_pred{}.log".format(ds, h)) for h in LT.PRED_LENS]
    return [LT_LOGS[model](ds, h) for h in LT.PRED_LENS]


def imp_paths(model: str, ds: str) -> List[Path]:
    if model == "SparseTSFPlus":
        return [_imp_plus_path(ds, r) for r in IMP.MASK_RATES]
    d = next(x for x in IMP.DATASETS if x.name == ds)
    return [IMP.log_path_for(model, d, r) for r in IMP.MASK_RATES]


def fs_paths(model: str, ds: str) -> List[Path]:
    if model == "SparseTSFPlus":
        return [_suite("fewshot_fixed_{}_pred{}.log".format(ds, h)) for h in FS.PRED_LENS]
    d = next(x for x in FS.DATASETS if x.name == ds)
    return [FS.log_path_for(d, h, model) for h in FS.PRED_LENS]


def zs_paths(model: str, a: str, b: str) -> List[Path]:
    if model == "SparseTSFPlus":
        return [_suite("zeroshot_fixed_{}_to_{}_pred{}.log".format(a, b, h)) for h in ZS.PRED_LENS]
    return [ZS.log_path_for(a, b, h, model) for h in ZS.PRED_LENS]


@functools.lru_cache(maxsize=None)
def m4_metrics(folder: str) -> Optional[Dict[str, float]]:
    path = SUITE.M4_RESULTS / folder
    if not all((path / "{}_forecast.csv".format(x)).exists() for x in M4.SUBSETS):
        return None
    from utils.m4_summary import M4Summary
    smape, owa, _, mase = M4Summary(str(path) + "/", str(M4.M4_ROOT)).evaluate()
    return {"SMAPE": smape["Average"], "MASE": mase["Average"], "OWA": owa["Average"]}


def m4_local(src: Source, model: str) -> Optional[Dict[str, float]]:
    if model == "SparseTSFPlus":
        logs = [_suite("m4_fixed_{}.log".format(x)) for x in M4.SUBSETS]
        folder = "SparseTSFPlus_fixed"
    elif model in ("TimeMixer", "SparseTSF"):
        logs = [REPO_ROOT / "logs" / "run_logs_m4" / "{}_{}.log".format(model, x) for x in M4.SUBSETS]
        folder = model
    else:
        return None
    if not all("test shape:" in src.read(lp) for lp in logs):
        return None
    return m4_metrics(folder)


def pems_local(src: Source, model: str) -> Optional[Dict[str, float]]:
    if model == "TimeMixer++":
        return None
    vals = []
    for d in PEMS.DATASETS:
        path = _suite("pems_fixed_{}.log".format(d.name)) if model == "SparseTSFPlus" \
            else PEMS.log_path_for(model, d)
        t = src.read(path)
        if "RMSE:" not in t:
            return None
        vals.append((_last(MAE_PAT, t), _last(MAPE_PAT, t) * 100.0, _last(RMSE_PAT, t)))
    return {"MAE": mean(v[0] for v in vals), "MAPE": mean(v[1] for v in vals),
            "RMSE": mean(v[2] for v in vals)}


def anomaly_local(src: Source, model: str):
    if model == "TimeMixer++":
        return None
    f1s = []
    for d in AN.DATASETS:
        if not AN.available(d):
            continue
        path = _suite("anomaly_fixed_{}.log".format(d.name)) if model == "SparseTSFPlus" \
            else AN.log_path_for(d, model)
        f1s.append(_last(F1_PAT, src.read(path)))
    if all(v is None for v in f1s):
        return None
    return PARTIAL if any(v is None for v in f1s) else 100 * mean(f1s)


def classification_local(src: Source, model: str):
    if model == "TimeMixer++":
        return None
    flat = {v.name: v for v in PCLS.VARIANTS}[SUITE.REUSE_CLASSIFICATION]
    accs = []
    for d in CLS.DATASETS:
        path = PCLS.log_path_for(d, flat) if model == "SparseTSFPlus" else CLS.log_path_for(d, model)
        accs.append(_last(ACC_PAT, src.read(path)))
    if all(v is None for v in accs):
        return None
    return PARTIAL if any(v is None for v in accs) else 100 * mean(accs)


AVAILABLE_ANOMALY = [d.name for d in AN.DATASETS if AN.available(d)]


def paper_rows(src: Source, local: List[str]):
    """-> list of ("group", title) or (label, kind, {column: value}, higher_better, spec).
    kind "pair" = MSE/MAE; "single" = one metric spanning both sub-columns."""
    rows = []

    def pub_lt(ds_list):
        out = {}
        for key, table in (("pub:TimeMixer", TMLT.PAPER), ("pub:TimeMixer++", PP.PAPER)):
            out[key] = (mean(mean(table[d][h][0] for h in LT.PRED_LENS) for d in ds_list),
                        mean(mean(table[d][h][1] for h in LT.PRED_LENS) for d in ds_list))
        return out

    rows.append(("group", "Long-term forecasting (MSE / MAE, mean of 4 horizons, input 96)"))
    vals = {m: _avg_pairs([_pair(src, lt_paths(m, d)) for d in ETT]) for m in local}
    vals.update(pub_lt(ETT))
    rows.append(("ETT (Avg)", "pair", vals, False, ".3f"))
    for d in ("Weather", "Solar-Energy", "Electricity", "Traffic"):
        vals = {m: _pair(src, lt_paths(m, d)) for m in local}
        vals.update(pub_lt([d]))
        rows.append((d, "pair", vals, False, ".3f"))

    rows.append(("group", "Univariate short-term forecasting, M4 (average of 6 subsets)"))
    m4 = {m: m4_local(src, m) for m in local}
    for metric in ("SMAPE", "MASE", "OWA"):
        vals = {m: (None if m4[m] is None else m4[m][metric]) for m in local}
        vals.update({"pub:" + k.split(" ")[0]: v[metric] for k, v in M4.PAPER_REF.items()})
        rows.append((metric, "single", vals, False, ".3f"))

    rows.append(("group", "Multivariate short-term forecasting, PEMS (average of 4 datasets, 96 → 12)"))
    pe = {m: pems_local(src, m) for m in local}
    for metric in ("MAE", "MAPE", "RMSE"):
        vals = {m: (None if pe[m] is None else pe[m][metric]) for m in local}
        vals.update({"pub:" + k.split(" ")[0]: v[metric] for k, v in PEMS.PAPER_REF.items()})
        rows.append((metric + (" (%)" if metric == "MAPE" else ""), "single", vals, False, ".2f"))

    rows.append(("group", "Imputation (MSE / MAE, mean of 4 mask rates, input 96)"))
    vals = {m: (_avg_pairs([_pair(src, imp_paths(m, d)) for d in ETT]) if m != "TimeMixer++" else None)
            for m in local}
    vals.update({"pub:TimeMixer": "n/c", "pub:TimeMixer++": "n/c"})
    rows.append(("ETT (Avg)", "pair", vals, False, ".3f"))
    for d in ("Electricity", "Weather"):
        vals = {m: (_pair(src, imp_paths(m, d)) if m != "TimeMixer++" else None) for m in local}
        vals.update({"pub:TimeMixer": "n/c", "pub:TimeMixer++": "n/c"})
        rows.append((d, "pair", vals, False, ".3f"))

    rows.append(("group", "Few-shot forecasting, 10% of training data (MSE / MAE, mean of 4 horizons)"))
    for label, ds_list in (("ETT (Avg)", ETT), ("Weather", ["Weather"]), ("Electricity", ["Electricity"])):
        vals = {m: (_avg_pairs([_pair(src, fs_paths(m, d)) for d in ds_list]) if m != "TimeMixer++"
                    else None) for m in local}
        ref = FS.PAPER_REF["ETT (Avg)" if label == "ETT (Avg)" else label]
        vals.update({"pub:" + k.split(" ")[0]: v for k, v in ref.items()})
        rows.append((label, "pair", vals, False, ".3f"))

    rows.append(("group", "Zero-shot transfer, train → test (MSE / MAE, mean of 4 horizons)"))
    for a, b in ZS.PAIRS:
        vals = {m: (_pair(src, zs_paths(m, a, b), zs=True) if m != "TimeMixer++" else None) for m in local}
        ref = ZS.PAPER_REF["{}->{}".format(a, b)]
        vals.update({"pub:" + k: v for k, v in ref.items()})
        rows.append(("{} → {}".format(a, b), "pair", vals, False, ".3f"))

    rows.append(("group", "Anomaly detection and classification (averages)"))
    vals = {m: anomaly_local(src, m) for m in local}
    vals.update({"pub:TimeMixer": None,
                 "pub:TimeMixer++": mean(AN.PAPER_REF_F1[d] for d in AVAILABLE_ANOMALY)})
    rows.append(("Anomaly F1 (%), {} datasets".format(len(AVAILABLE_ANOMALY)), "single", vals, True, ".2f"))
    vals = {m: classification_local(src, m) for m in local}
    vals.update({"pub:TimeMixer": None, "pub:TimeMixer++": CLS.PAPER_REF["TimeMixer++ (paper)"]})
    rows.append(("Classification accuracy (%), 10 UEA datasets", "single", vals, True, ".2f"))
    return rows


LOCAL_LABEL = {"TimeMixer": "TimeMixer", "TimeMixer++": "TimeMixer++", "SparseTSF": "SparseTSF",
               "SparseTSFPlus": "SparseTSFPlus"}


def _fmt_cell(v, spec, rank=None, italic=False):
    if v is None:
        return "—"
    if v == PARTIAL:
        return "partial"
    if v == "n/c":
        return "n/c"
    s = format(v, spec)
    if italic:
        return "<i>{}</i>".format(s)
    if rank == 1:
        return "<b>{}</b>".format(s)
    if rank == 2:
        return "<u>{}</u>".format(s)
    return s


def _ranks(values: Dict[str, Optional[float]], higher_better: bool) -> Dict[str, int]:
    nums = {k: v for k, v in values.items() if isinstance(v, float)}
    order = sorted(nums, key=nums.get, reverse=higher_better)
    if len(order) < 2:
        return {}
    # the second best is marked only when there are at least three values to rank
    return {k: i + 1 for i, k in enumerate(order[:2 if len(order) >= 3 else 1])}


def paper_table(src: Source, local: List[str], published: List[str]) -> List[str]:
    cols = local + ["pub:" + m for m in published]
    n = 1 + 2 * len(cols)
    head1 = "".join('<th colspan="2">{}</th>'.format(LOCAL_LABEL[c]) for c in local) + \
        "".join('<th colspan="2" class="pub">{}<br>(paper)</th>'.format(c[4:]) for c in cols[len(local):])
    head2 = "".join("<th>MSE</th><th>MAE</th>" for _ in cols)
    out = ['<table class="paper">',
           '<thead><tr><th rowspan="2">Benchmark</th>{}</tr><tr>{}</tr></thead>'.format(head1, head2),
           "<tbody>"]
    for row in paper_rows(src, local):
        if row[0] == "group":
            out.append('<tr class="group"><td colspan="{}">{}</td></tr>'.format(n, row[1]))
            continue
        label, kind, vals, hb, spec = row
        cells = []
        if kind == "pair":
            r_mse = _ranks({m: (vals.get(m)[0] if isinstance(vals.get(m), tuple) else None)
                            for m in local}, hb)
            r_mae = _ranks({m: (vals.get(m)[1] if isinstance(vals.get(m), tuple) else None)
                            for m in local}, hb)
            for c in cols:
                v = vals.get(c)
                pub = c.startswith("pub:")
                if isinstance(v, tuple):
                    cells.append("<td>{}</td><td>{}</td>".format(
                        _fmt_cell(v[0], spec, r_mse.get(c), pub), _fmt_cell(v[1], spec, r_mae.get(c), pub)))
                else:
                    cells.append('<td colspan="2">{}</td>'.format(_fmt_cell(v, spec)))
        else:
            r = _ranks({m: vals.get(m) for m in local}, hb)
            for c in cols:
                cells.append('<td colspan="2">{}</td>'.format(
                    _fmt_cell(vals.get(c), spec, r.get(c), c.startswith("pub:"))))
        out.append("<tr><td>{}</td>{}</tr>".format(label, "".join(cells)))
    out += ["</tbody>", "</table>"]
    return out


# ── coverage, parameters, energy ─────────────────────────────────────────────────────────────

COVERAGE = [  # (task, folder, completion test)
    ("Long-term forecasting", "run_logs", "mse"), ("Long-term forecasting", "run_logs_pp", "mse"),
    ("Long-term forecasting", "run_logs_sparsetsf", "mse"),
    ("M4", "run_logs_m4", "m4"), ("PEMS", "run_logs_pems", "pems"),
    ("Imputation", "run_logs_imputation", "mse"), ("Few-shot", "run_logs_fewshot", "mse"),
    ("Zero-shot", "run_logs_zeroshot", "zs"), ("Anomaly detection", "run_logs_anomaly", "f1"),
    ("Classification", "run_logs_classification", "acc"),
]
DIR_MODEL = {"run_logs": "TimeMixer", "run_logs_pp": "TimeMixer++"}


def _finished(kind, text):
    if kind == "m4":
        return "test shape:" in text
    if kind == "pems":
        return "RMSE:" in text
    if kind == "zs":
        return len(re.findall(r"^mse:", text, re.M)) >= 2
    if kind == "f1":
        return F1_PAT.search(text) is not None
    if kind == "acc":
        return ACC_PAT.search(text) is not None
    return MSE_PAT.search(text) is not None


def coverage_counts(src: Source) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {}
    for task, folder, kind in COVERAGE:
        d = REPO_ROOT / "logs" / folder
        paths = sorted(d.glob("*.log")) if src.commit is None else \
            [WORKSPACE / f for f in src.tree if f.startswith("TimeMixer/logs/{}/".format(folder))
             and f.endswith(".log")]
        for p in paths:
            text = src.read(p)
            if not _finished(kind, text):
                continue
            if folder in DIR_MODEL:
                model = DIR_MODEL[folder]
            else:
                m = MODEL_PAT.search(text)
                model = {"TimeMixerPP": "TimeMixer++"}.get(m.group(1), m.group(1)) if m else "?"
            if folder == "run_logs_sparsetsf" and "_sl96_" not in p.name:
                continue
            counts.setdefault(task, {}).setdefault(model, 0)
            counts[task][model] += 1
    return counts


PROTOCOL = {"Long-term forecasting": "8 datasets × 4 horizons = 32", "M4": "6 subsets",
            "PEMS": "4 datasets", "Imputation": "6 datasets × 4 mask rates = 24",
            "Few-shot": "6 datasets × 4 horizons = 24", "Zero-shot": "6 pairs × 4 horizons = 24",
            "Anomaly detection": "5 datasets", "Classification": "10 datasets"}


def coverage_table(then: Source, now: Source) -> List[str]:
    a, b = coverage_counts(then), coverage_counts(now)
    out = ["| Task | Protocol | TimeMixer: then → now | TimeMixer++: then → now | SparseTSF: then → now |",
           "|---|---|---:|---:|---:|"]
    for task in PROTOCOL:
        cells = []
        for m in MODELS:
            x, y = a.get(task, {}).get(m, 0), b.get(task, {}).get(m, 0)
            cells.append("{} → {}".format(x, y) if (x or y) else "published only")
        out.append("| {} | {} | {} |".format(task, PROTOCOL[task], " | ".join(cells)))
    return out


def params_table() -> List[str]:
    """Min–max trainable parameters per model and task, from reports/results_param_counts.md."""
    text = (REPO_ROOT / "reports" / "results_param_counts.md").read_text(encoding="utf-8")
    out = ["| Task | SparseTSF | SparseTSFPlus (fixed) | TimeMixer | TimeMixer++ |",
           "|---|---:|---:|---:|---:|"]
    st = suite_tables()
    plus = {"Long-term forecasting": st["Long-term"].params.get("fixed", "—"),
            "Short-term forecasting, M4": st["M4"].params.get("fixed", "—"),
            "Short-term forecasting, PEMS": st["PEMS"].params.get("fixed", "—"),
            "Imputation": st["Imputation"].params.get("fixed", "—"),
            "Anomaly detection": st["Anomaly"].params.get("fixed", "—"),
            "Classification": st["Classification"].params.get("fixed", "—")}
    for sec in re.split(r"^## ", text, flags=re.M)[1:]:
        title = sec.splitlines()[0].strip()
        lines = [l for l in sec.splitlines() if l.startswith("|") and not l.startswith("|---")]
        if not lines:
            continue
        head = [h.strip() for h in lines[0].strip("|").split("|")]
        nums = {h: [] for h in head[1:]}
        for l in lines[1:]:
            cells = [c.strip() for c in l.strip("|").split("|")]
            for h, c in zip(head[1:], cells[1:]):
                for tok in re.findall(r"[0-9][0-9,]*", c):
                    nums[h].append(int(tok.replace(",", "")))

        def rng(h):
            v = nums.get(h) or []
            return "—" if not v else ("{:,}".format(min(v)) if min(v) == max(v)
                                      else "{:,}–{:,}".format(min(v), max(v)))
        key = title.split(" (")[0]
        out.append("| {} | {} | {} | {} | {} |".format(
            title, rng("SparseTSF"), plus.get(key, "—"), rng("TimeMixer"), rng("TimeMixer++")))
    return out


def energy_table() -> List[str]:
    return [l for l in SUITE.energy_md() if l.startswith("|")]


# ── assembly ─────────────────────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=None)
def source(commit: Optional[str] = None) -> Source:
    return Source(commit)


THREE = (["TimeMixer", "TimeMixer++", "SparseTSF"], ["TimeMixer", "TimeMixer++"])
WITH_PLUS = (["TimeMixer", "SparseTSF", "SparseTSFPlus"], ["TimeMixer", "TimeMixer++"])

BLOCKS = {
    "T1": lambda: paper_table(source(SNAPSHOT), *THREE),
    "T2": lambda: paper_table(source(), *THREE),
    "COVERAGE": lambda: coverage_table(source(SNAPSHOT), source()),
    "T5": lambda: paper_table(source(), *WITH_PLUS),
    "SEEDS": seeds_table,
    "PARAMS": params_table,
    "ENERGY": energy_table,
    "A1": lambda: comparison_table(source(), plus_columns()),
    "A2": lambda: comparison_table(source(SNAPSHOT)),
}


def update_md() -> None:
    text = MD_PATH.read_text(encoding="utf-8")
    for key, fn in BLOCKS.items():
        start, end = "<!-- AUTO_{}_START -->".format(key), "<!-- AUTO_{}_END -->".format(key)
        si, ei = text.find(start), text.find(end)
        if si < 0 or ei < si:
            raise RuntimeError("markers for {} not found".format(key))
        body = "\n".join(fn())
        text = text[: si + len(start)] + "\n\n" + body + "\n\n" + text[ei:]
        print("[OK] {}".format(key), flush=True)
    MD_PATH.write_text(text, encoding="utf-8")


CSS = """
@page { size: A4; margin: 18mm 15mm 18mm 15mm; }
body { font-family: 'Cambria', 'Georgia', serif; font-size: 10.5pt; line-height: 1.42; color: #111;
       max-width: none !important; margin: 0 !important; padding: 0 !important; }
h1 { font-size: 19pt; margin: 0 0 4pt 0; }
h2 { font-size: 14pt; margin-top: 18pt; border-bottom: 1px solid #999; padding-bottom: 2pt; }
h3 { font-size: 11.5pt; margin-top: 12pt; }
p, li { text-align: justify; }
table { border-collapse: collapse; width: 100%; font-size: 7.6pt; margin: 6pt 0 10pt 0;
        page-break-inside: auto; }
th { font-size: 7.4pt; }
table.paper { font-size: 7.4pt; }
table.paper th, table.paper td { text-align: center; padding: 2pt 3pt; }
table.paper td:first-child { text-align: left; white-space: nowrap; }
table.paper tr.group td { text-align: left; font-weight: bold; background: #f4f4f4; }
table.paper th.pub { font-style: italic; }
tr { page-break-inside: avoid; }
th, td { border-top: 1px solid #ccc; border-bottom: 1px solid #ccc; padding: 2.5pt 4pt; }
th { background: #eee; font-weight: bold; }
td:first-child, th:first-child { text-align: left; }
code { font-family: Consolas, monospace; font-size: 9pt; }
pre { font-size: 8.5pt; background: #f6f6f6; padding: 6pt; white-space: pre-wrap; }
blockquote { margin: 6pt 0 6pt 12pt; color: #333; }
blockquote p { text-align: left; }
.subtitle { color: #444; font-size: 11pt; }
"""


def build_pdf() -> None:
    pandoc = shutil.which("pandoc")
    chrome = next((p for p in (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                               r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
                   if Path(p).exists()), None)
    if not pandoc or not chrome:
        raise SystemExit("PDF needs pandoc and Chrome or Edge")
    tmp = REPO_ROOT / "reports" / "_thesis_progress_report.html"
    css = REPO_ROOT / "reports" / "_thesis_progress_report.css"
    css.write_text(CSS, encoding="utf-8")
    subprocess.run([pandoc, str(MD_PATH), "-f", "gfm", "-t", "html5", "--standalone",
                    "--metadata", "title=Efficient general-purpose time-series models",
                    "--css", css.name, "-o", str(tmp)], cwd=str(tmp.parent), check=True)
    html = tmp.read_text(encoding="utf-8")
    html = html.replace('<link rel="stylesheet" href="{}" />'.format(css.name),
                        "<style>{}</style>".format(CSS))
    html = re.sub(r'<header id="title-block-header">.*?</header>', "", html, flags=re.S)
    tmp.write_text(html, encoding="utf-8")
    subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    "--print-to-pdf={}".format(PDF_PATH), tmp.resolve().as_uri()], check=True,
                   capture_output=True)
    tmp.unlink()
    css.unlink()
    print("[OK] wrote " + str(PDF_PATH.relative_to(REPO_ROOT)), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true", help="also build the PDF (pandoc + Chrome/Edge)")
    args = ap.parse_args()
    update_md()
    if args.pdf:
        build_pdf()


if __name__ == "__main__":
    main()
