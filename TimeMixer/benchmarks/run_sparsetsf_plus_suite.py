"""
SparseTSFPlus across the full 8-task suite, in two versions, beside SparseTSF and TimeMixer.

  fixed  ONE configuration for every task and dataset, the thesis's target form:
         learned-trend decomposition (forecasting) and phase mixing (every task that folds the
         series into periods) in the backbone; gated refinement and mask-aware aggregation whenever
         the input has missing values (imputation); default heads otherwise.
  task   the unit set that won each task in the ablations, applied unchanged to all of that task's
         datasets: learned-trend decomposition for forecasting, phase mixing at period 4 for PEMS,
         gated refinement + mask-aware aggregation for imputation, no units for anomaly detection and
         classification. Better numbers, but not one model, so never the headline.

Both use SparseTSF's own published period for every dataset (no per-dataset tuning), except the
per-task PEMS cell, which is the ablation's phase mixing at period 4. Every training setting is
imported from the runner that produced the SparseTSF cell, so each row differs from SparseTSF
only in the units.

Cells that are identical to an existing run are reused rather than re-run:
  imputation      fixed = ablation variant `allrg`, task = ablation variant `u5g` (all 24 cells exist;
                  before the refinement gate these were `all` and `u5`)
  classification  both  = ablation variant `flat` (no unit applies to the classification head)
  PEMS, task      ablation variant `phase_p4`
  anomaly, task   no units, which is SparseTSF itself (bitwise, by the equivalence gate)

Energy uses the matched carbon protocol of run_carbon_benchmarks.py (fixed 5 epochs, no early
stopping) on its 5 datasets x 4 horizons, so it is comparable with that table only.

Idempotent: a run whose log already holds its final metric is skipped.

    venv/Scripts/python.exe benchmarks/run_sparsetsf_plus_suite.py                 # everything
    venv/Scripts/python.exe benchmarks/run_sparsetsf_plus_suite.py --only m4,pems  # some tasks
    venv/Scripts/python.exe benchmarks/run_sparsetsf_plus_suite.py --update-from-logs-only
"""

import argparse
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import run_all_long_term_benchmarks as TMLT
import run_anomaly_benchmarks as AN
import run_carbon_benchmarks as CARBON
import run_classification_benchmarks as CLS
import run_fewshot_benchmarks as FS
import run_imputation_benchmarks as IMP
import run_m4_benchmarks as M4
import run_pems_benchmarks as PEMS
import run_sparsetsf_benchmarks as LT
import run_sparsetsf_plus_classification as PCLS
import run_sparsetsf_plus_forecast as PFC
import run_sparsetsf_plus_imputation as PIMP
import run_zeroshot_benchmarks as ZS

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT.parent / "dataset"
LOGS_DIR = REPO_ROOT / "logs" / "run_logs_plus_suite"
CKPT_DIR = REPO_ROOT / "checkpoints"
M4_RESULTS = REPO_ROOT / "m4_results"
MD_PATH = REPO_ROOT / "reports" / "results_sparsetsf_plus_suite.md"

MODEL = "SparseTSFPlus"
VERSIONS = ["fixed", "task"]
LABEL = {"fixed": "SparseTSFPlus (fixed)", "task": "SparseTSFPlus (per task)"}

FORECAST_FLAGS = {"fixed": ["--stsf_decomp", "linear", "--stsf_phase_mix", "linear"],
                  "task": ["--stsf_decomp", "linear"]}
ANOMALY_FLAGS = {"fixed": ["--stsf_phase_mix", "linear"]}
REUSE_IMPUTATION = {"fixed": "allrg", "task": "u5g"}
REUSE_CLASSIFICATION = "flat"
REUSE_PEMS_TASK = "phase_p4"

NUM = r"([0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?)"
MSE_PAT = re.compile(r"mse:" + NUM)
MAE_PAT = re.compile(r"mae:" + NUM)
F1_PAT = re.compile(r"F-score\s*:\s*" + NUM)
PARAM_PAT = re.compile(r"trainable_params:([0-9]+)")
PY = [sys.executable, "-u", str(REPO_ROOT / "run.py")]


@dataclass
class Job:
    task: str
    version: str
    key: str                       # unique within (task, version); used in the log name
    cmd: List[str]
    done: Callable[[str], bool]    # log text -> finished?
    after: Optional[Callable[[], None]] = None
    zero_shot: Optional[Tuple[List[str], str, str]] = None   # (target cmd, src setting, tgt setting)

    @property
    def log(self) -> Path:
        return LOGS_DIR / "{}_{}_{}.log".format(self.task, self.version, self.key)


def _has_mse(text: str) -> bool:
    return MSE_PAT.search(text) is not None


def _replace_model(cmd: List[str]) -> List[str]:
    out = list(cmd)
    out[out.index("--model") + 1] = MODEL
    return out


# ── job builders (each mirrors the command of the runner that produced the SparseTSF cell) ──

def jobs_longterm(v: str) -> List[Job]:
    jobs = []
    for ds in LT.DATASETS:
        for p in LT.PRED_LENS:
            cmd = PY + [
                "--task_name", "long_term_forecast", "--is_training", "1",
                "--model_id", "PLUS_{}_{}_96_{}".format(v, ds.name, p), "--model", MODEL,
                "--data", ds.data, "--root_path", str(ds.root_path), "--data_path", ds.data_path,
                "--features", "M", "--seq_len", "96", "--label_len", "0", "--pred_len", str(p),
                "--enc_in", str(ds.enc_in), "--dec_in", str(ds.enc_in), "--c_out", str(ds.enc_in),
                "--period_len", str(ds.period_len), "--model_type", LT.MODEL_TYPE,
                "--d_model", str(LT.D_MODEL), "--factor", str(ds.factor),
                "--learning_rate", str(ds.learning_rate), "--lradj", LT.LRADJ,
                "--train_epochs", str(LT.TRAIN_EPOCHS), "--patience", str(LT.PATIENCE),
                "--batch_size", str(ds.batch_size), "--num_workers", "0",
                "--des", "PlusSuite", "--itr", "1",
            ] + (["--stream_metrics", "1"] if ds.stream_metrics else []) + FORECAST_FLAGS[v]
            jobs.append(Job("longterm", v, "{}_pred{}".format(ds.name, p), cmd, _has_mse))
    return jobs


def _m4_csv(v: str, subset: str) -> Path:
    return M4_RESULTS / "{}_{}".format(MODEL, v) / "{}_forecast.csv".format(subset)


def jobs_m4(v: str) -> List[Job]:
    cfg = M4.MODEL_CFG["SparseTSF"]
    jobs = []
    for subset in M4.SUBSETS:
        cmd = PY + [
            "--task_name", "short_term_forecast", "--is_training", "1",
            "--model_id", "M4_PLUS_{}_{}".format(v, subset), "--model", MODEL,
            "--data", "m4", "--root_path", str(M4.M4_ROOT), "--seasonal_patterns", subset,
            "--features", "M", "--enc_in", "1", "--dec_in", "1", "--c_out", "1",
            "--period_len", str(M4.SPARSE_PERIOD[subset]),
            "--learning_rate", cfg["learning_rate"], "--train_epochs", cfg["train_epochs"],
            "--patience", cfg["patience"], "--batch_size", cfg["batch_size"],
            "--num_workers", M4.COMMON["num_workers"], "--loss", M4.COMMON["loss"], "--des", "M4",
        ] + cfg["extra"] + FORECAST_FLAGS[v]

        def move(subset=subset, v=v):
            # run.py writes every SparseTSFPlus forecast to m4_results/SparseTSFPlus/; keep each
            # version's forecasts apart so the two cannot overwrite each other.
            src = M4_RESULTS / MODEL / "{}_forecast.csv".format(subset)
            if src.exists():
                _m4_csv(v, subset).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(_m4_csv(v, subset)))

        jobs.append(Job("m4", v, subset, cmd,
                        lambda t, s=subset, v=v: "test shape:" in t and _m4_csv(v, s).exists(),
                        after=move))
    return jobs


def jobs_pems(v: str) -> List[Job]:
    if v != "fixed":
        return []          # per task: the ablation's phase_p4 runs are reused
    cfg = PEMS.MODEL_CFG["SparseTSF"]
    jobs = []
    for ds in PEMS.DATASETS:
        cmd = PY + [
            "--task_name", "long_term_forecast", "--is_training", "1",
            "--model_id", "PEMS_PLUS_{}_{}".format(v, ds.name), "--model", MODEL,
            "--data", "PEMS", "--root_path", str(DATA_ROOT / "PEMS"), "--data_path", ds.data_path,
            "--features", "M", "--seq_len", str(PEMS.SEQ_LEN), "--label_len", "0",
            "--pred_len", str(PEMS.PRED_LEN),
            "--enc_in", str(ds.enc_in), "--dec_in", str(ds.enc_in), "--c_out", str(ds.enc_in),
            "--use_norm", "0", "--learning_rate", cfg["learning_rate"],
            "--train_epochs", cfg["train_epochs"], "--patience", cfg["patience"],
            "--batch_size", cfg["batch_size"], "--num_workers", "0", "--des", PEMS.DES, "--itr", "1",
        ] + cfg["extra"] + FORECAST_FLAGS[v]
        jobs.append(Job("pems", v, ds.name, cmd, lambda t: "RMSE:" in t))
    return jobs


def jobs_fewshot(v: str) -> List[Job]:
    jobs = []
    for ds in FS.DATASETS:
        for p in FS.PRED_LENS:
            cmd = PY + [
                "--task_name", "long_term_forecast", "--is_training", "1",
                "--model_id", "FEWSHOT_PLUS_{}_{}_96_{}".format(v, ds.name, p), "--model", MODEL,
                "--data", ds.data, "--root_path", str(ds.root_path), "--data_path", ds.data_path,
                "--features", "M", "--seq_len", str(FS.SEQ_LEN), "--label_len", "0",
                "--pred_len", str(p),
                "--enc_in", str(ds.enc_in), "--dec_in", str(ds.enc_in), "--c_out", str(ds.enc_in),
            ] + FS._model_args("SparseTSF", ds) + [
                "--num_workers", "0", "--des", FS.DES, "--itr", "1",
            ] + (["--stream_metrics", "1"] if (ds.name, p) in FS.STREAM else []) + FORECAST_FLAGS[v]
            jobs.append(Job("fewshot", v, "{}_pred{}".format(ds.name, p), cmd, _has_mse))
    return jobs


def _zs_setting(src: str, model_id: str, data: str, p: int) -> str:
    return ZS.build_setting("SparseTSF", src, model_id, data, p).replace(
        "_none_SparseTSF_", "_none_{}_".format(MODEL))


def jobs_zeroshot(v: str) -> List[Job]:
    jobs = []
    for src, tgt in ZS.PAIRS:
        for p in ZS.PRED_LENS:
            mid = "ZS_PLUS_{}_{}_to_{}_{}".format(v, src, tgt, p)
            train = _replace_model(ZS._base_args("SparseTSF", src, src, mid, p)) + FORECAST_FLAGS[v]
            test = _replace_model(ZS._base_args("SparseTSF", tgt, src, mid, p)) + FORECAST_FLAGS[v]
            jobs.append(Job("zeroshot", v, "{}_to_{}_pred{}".format(src, tgt, p),
                            train + ["--is_training", "1"],
                            lambda t: len(re.findall(r"^mse:", t, re.M)) >= 2,
                            zero_shot=(test + ["--is_training", "0"],
                                       _zs_setting(src, mid, ZS.DS[src][0], p),
                                       _zs_setting(src, mid, ZS.DS[tgt][0], p))))
    return jobs


def jobs_anomaly(v: str) -> List[Job]:
    if v not in ANOMALY_FLAGS:
        return []          # per task: no units, which is SparseTSF itself
    jobs = []
    for ds in AN.DATASETS:
        if not AN.available(ds):
            continue
        cmd = PY + [
            "--task_name", "anomaly_detection", "--is_training", "1",
            "--model_id", "ANOM_PLUS_{}_{}".format(v, ds.name), "--model", MODEL,
            "--data", ds.name, "--root_path", str(DATA_ROOT / ds.folder), "--features", "M",
            "--seq_len", str(AN.SEQ_LEN), "--pred_len", "0",
            "--enc_in", str(ds.enc_in), "--dec_in", str(ds.enc_in), "--c_out", str(ds.enc_in),
        ] + AN._model_args("SparseTSF") + [
            "--anomaly_ratio", str(ds.anomaly_ratio), "--learning_rate", str(AN.LEARNING_RATE),
            "--train_epochs", str(AN.TRAIN_EPOCHS), "--batch_size", str(AN.BATCH_SIZE),
            "--num_workers", "0", "--des", AN.DES, "--itr", "1",
        ] + ANOMALY_FLAGS[v]
        jobs.append(Job("anomaly", v, ds.name, cmd, lambda t: "F-score" in t))
    return jobs


def _carbon_id(v: str, ds, p: int) -> str:
    return "CARBON_{}-{}_{}_{}_{}".format(MODEL, v, ds.name, CARBON.SEQ_LEN, p)


def carbon_setting(v: str, ds, p: int) -> str:
    return ("long_term_forecast_{}_none_{}_{}_sl{}_pl{}_dm{}_nh{}_el{}_dl1_df{}_fc{}_ebtimeF_dtTrue_{}_0"
            .format(_carbon_id(v, ds, p), MODEL, ds.data, CARBON.SEQ_LEN, p, CARBON.D_MODEL,
                    CARBON.N_HEADS, CARBON.E_LAYERS, CARBON.D_FF, ds.factor, CARBON.DES))


def jobs_energy(v: str) -> List[Job]:
    jobs = []
    for ds in CARBON.DATASETS:
        for p in CARBON.PRED_LENS:
            cmd = PY + [
                "--task_name", "long_term_forecast", "--is_training", "1",
                "--model_id", _carbon_id(v, ds, p), "--model", MODEL,
                "--data", ds.data, "--root_path", str(ds.root_path), "--data_path", ds.data_path,
                "--features", "M", "--seq_len", str(CARBON.SEQ_LEN), "--label_len", "0",
                "--pred_len", str(p),
                "--enc_in", str(ds.enc_in), "--dec_in", str(ds.enc_in), "--c_out", str(ds.enc_in),
                "--d_model", str(CARBON.D_MODEL), "--d_ff", str(CARBON.D_FF),
                "--n_heads", str(CARBON.N_HEADS), "--e_layers", str(CARBON.E_LAYERS),
                "--down_sampling_layers", str(CARBON.DOWN_SAMPLING_LAYERS),
                "--down_sampling_method", "avg", "--down_sampling_window", "2",
                "--factor", str(ds.factor), "--learning_rate", str(CARBON.LEARNING_RATE),
                "--lradj", CARBON.LRADJ, "--train_epochs", str(CARBON.TRAIN_EPOCHS),
                "--patience", str(CARBON.PATIENCE), "--batch_size", str(CARBON.BATCH_SIZE),
                "--top_k", str(CARBON.TOP_K), "--num_workers", "0", "--des", CARBON.DES,
                "--itr", "1", "--period_len", str(ds.period_len), "--model_type", "linear",
                "--track_emissions", "1", "--emissions_output_dir", str(REPO_ROOT / "logs"),
                "--emissions_file", CARBON.EMISSIONS_FILE,
            ] + FORECAST_FLAGS[v]
            jobs.append(Job("energy", v, "{}_pred{}".format(ds.name, p), cmd, _has_mse))
    return jobs


# Energy depends on the machine's state (power mode, background load), which differed between the
# original July carbon sweep and now. SparseTSF and TimeMixer are therefore re-measured in the same
# session as SparseTSFPlus, under their own run names so the July rows stay untouched.
REF_MODELS = ["SparseTSF", "TimeMixer"]


def _carbon_ref_id(model: str, ds, p: int) -> str:
    return "CARBON_{}-ref_{}_{}_{}".format(model, ds.name, CARBON.SEQ_LEN, p)


def carbon_ref_setting(model: str, ds, p: int) -> str:
    return ("long_term_forecast_{}_none_{}_{}_sl{}_pl{}_dm{}_nh{}_el{}_dl1_df{}_fc{}_ebtimeF_dtTrue_{}_0"
            .format(_carbon_ref_id(model, ds, p), model, ds.data, CARBON.SEQ_LEN, p, CARBON.D_MODEL,
                    CARBON.N_HEADS, CARBON.E_LAYERS, CARBON.D_FF, ds.factor, CARBON.DES))


def jobs_energy_ref() -> List[Job]:
    jobs = []
    for model in REF_MODELS:
        for job in jobs_energy("fixed"):
            flags = FORECAST_FLAGS["fixed"]
            assert job.cmd[-len(flags):] == flags
            cmd = job.cmd[:-len(flags)]          # the unit flags are always appended last
            cmd[cmd.index("--model") + 1] = model
            ds = next(d for d in CARBON.DATASETS if job.key.startswith(d.name + "_"))
            p = int(job.key.rsplit("pred", 1)[1])
            cmd[cmd.index("--model_id") + 1] = _carbon_ref_id(model, ds, p)
            jobs.append(Job("energyref", model, job.key, cmd, _has_mse))
    return jobs


BUILDERS = {"anomaly": jobs_anomaly, "m4": jobs_m4, "pems": jobs_pems, "zeroshot": jobs_zeroshot,
            "fewshot": jobs_fewshot, "longterm": jobs_longterm, "energy": jobs_energy}
ORDER = ["anomaly", "m4", "pems", "zeroshot", "fewshot", "longterm", "energy"]


def all_jobs(only=None) -> List[Job]:
    jobs = [j for v in VERSIONS for t in ORDER if not only or t in only for j in BUILDERS[t](v)]
    if not only or "energyref" in only:
        jobs += jobs_energy_ref()
    return jobs


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def run(job: Job) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with job.log.open("w", encoding="utf-8") as f:
        if job.zero_shot:
            f.write("===== TRAIN on source =====\n"); f.flush()
        subprocess.run(job.cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
        if job.zero_shot:
            test_cmd, src_setting, tgt_setting = job.zero_shot
            src = CKPT_DIR / src_setting / "checkpoint.pth"
            if not src.exists():
                f.write("\n!! source checkpoint missing: {}\n".format(src))
                return
            dst = CKPT_DIR / tgt_setting / "checkpoint.pth"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            f.write("\n===== TEST on target (zero-shot) =====\n"); f.flush()
            subprocess.run(test_cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
    if job.after:
        job.after()
    print("      finished in {:.1f}s".format(time.perf_counter() - start), flush=True)


# ── report ──────────────────────────────────────────────────────────────────

def _last(pat, text) -> Optional[float]:
    m = pat.findall(text)
    return float(m[-1]) if m else None


def _mean(vals: List[Optional[float]]) -> Optional[float]:
    vals = [x for x in vals if x is not None]
    return sum(vals) / len(vals) if vals else None


def _fmt(x: Optional[float], spec: str) -> str:
    return "-" if x is None else format(x, spec)


def _params(texts: List[str]) -> str:
    ps = sorted({int(m) for t in texts for m in PARAM_PAT.findall(t)})
    if not ps:
        return "-"
    return "{:,}".format(ps[0]) if ps[0] == ps[-1] else "{:,}–{:,}".format(ps[0], ps[-1])


def suite_log(task: str, v: str, key: str) -> str:
    return read(LOGS_DIR / "{}_{}_{}.log".format(task, v, key))


class Table:
    """Rows of {column: value}; the mean row covers only rows where every column has a value."""

    def __init__(self, title: str, metric: str, spec: str, higher_better: bool):
        self.title, self.metric, self.spec, self.hb = title, metric, spec, higher_better
        self.rows: List[Tuple[str, Dict[str, Optional[float]]]] = []
        self.params: Dict[str, str] = {}

    def add(self, name: str, **vals):
        self.rows.append((name, vals))

    def columns(self) -> List[str]:
        return ["SparseTSF", "fixed", "task", "TimeMixer"]

    def means(self) -> Tuple[Dict[str, Optional[float]], int]:
        full = [r for _, r in self.rows if all(r.get(c) is not None for c in self.columns())]
        return {c: _mean([r[c] for r in full]) for c in self.columns()}, len(full)

    def markdown(self) -> List[str]:
        head = {"SparseTSF": "SparseTSF", "fixed": LABEL["fixed"], "task": LABEL["task"],
                "TimeMixer": "TimeMixer"}
        out = ["### " + self.title, "",
               "| {} | ".format("Dataset") + " | ".join(head[c] for c in self.columns()) + " |",
               "|---|" + "---:|" * len(self.columns())]
        for name, r in self.rows:
            out.append("| {} | ".format(name) + " | ".join(_fmt(r.get(c), self.spec)
                                                          for c in self.columns()) + " |")
        m, n = self.means()
        out.append("| **Mean ({} of {} rows)** | ".format(n, len(self.rows))
                   + " | ".join("**{}**".format(_fmt(m[c], self.spec)) for c in self.columns()) + " |")
        if self.params:
            out.append("| _SparseTSFPlus parameters_ | | _{}_ | _{}_ | |".format(
                self.params.get("fixed", "-"), self.params.get("task", "-")))
        out.append("")
        return out


def table_longterm() -> Table:
    t = Table("Long-term forecasting (seq_len 96, mean MSE over 4 horizons)", "MSE", ".4f", False)
    texts = {v: [] for v in VERSIONS}
    for ds in LT.DATASETS:
        cells = {c: [] for c in ("SparseTSF", "fixed", "task", "TimeMixer")}
        for p in LT.PRED_LENS:
            cells["SparseTSF"].append(LT.parse_log(LT.log_path_for(ds, 96, p))[0])
            cells["TimeMixer"].append(TMLT.parse_log_file(
                REPO_ROOT / "logs" / "run_logs" / "{}_pred{}.log".format(ds.name, p))[0])
            for v in VERSIONS:
                txt = suite_log("longterm", v, "{}_pred{}".format(ds.name, p))
                texts[v].append(txt)
                cells[v].append(_last(MSE_PAT, txt))
        # a dataset's mean needs all four horizons, so missing cells cannot flatter a model
        t.add(ds.name, **{c: (_mean(vs) if all(x is not None for x in vs) else None)
                          for c, vs in cells.items()})
    t.params = {v: _params(texts[v]) for v in VERSIONS}
    return t


def _m4_owa(folder: Path) -> Optional[float]:
    if not all((folder / "{}_forecast.csv".format(s)).exists() for s in M4.SUBSETS):
        return None
    from utils.m4_summary import M4Summary
    smape, owa, mape, mase = M4Summary(str(folder) + "/", str(M4.M4_ROOT)).evaluate()
    return owa.get("Average")


def table_m4() -> Table:
    t = Table("M4 short-term forecasting (OWA, 6 subsets)", "OWA", ".3f", False)
    t.add("M4 (6 subsets)", SparseTSF=_m4_owa(M4_RESULTS / "SparseTSF"),
          fixed=_m4_owa(M4_RESULTS / "{}_fixed".format(MODEL)),
          task=_m4_owa(M4_RESULTS / "{}_task".format(MODEL)),
          TimeMixer=_m4_owa(M4_RESULTS / "TimeMixer"))
    t.params = {v: _params([suite_log("m4", v, s) for s in M4.SUBSETS]) for v in VERSIONS}
    return t


def table_pems() -> Table:
    t = Table("PEMS short-term forecasting (MAE, input 96 → 12)", "MAE", ".2f", False)
    pfv = {x.name: x for x in PFC.VARIANTS}
    pds = {d.name: d for d in PFC.DATASETS}
    task_texts = []
    for ds in PEMS.DATASETS:
        fixed_txt = suite_log("pems", "fixed", ds.name)
        task_path = PFC.log_path_for(pds[ds.name], pfv[REUSE_PEMS_TASK], 12)
        task_texts.append(read(task_path))
        t.add(ds.name, SparseTSF=PEMS.parse_log(PEMS.log_path_for("SparseTSF", ds))[0],
              fixed=_last(MAE_PAT, fixed_txt) if "RMSE:" in fixed_txt else None,
              task=PFC.parse_log(task_path)["mae"],
              TimeMixer=PEMS.parse_log(PEMS.log_path_for("TimeMixer", ds))[0])
    t.params = {"fixed": _params([suite_log("pems", "fixed", d.name) for d in PEMS.DATASETS]),
                "task": _params(task_texts)}
    return t


def table_imputation() -> Table:
    t = Table("Imputation (masked MSE, mean over 4 mask rates)", "MSE", ".4f", False)
    pv = {x.name: x for x in PIMP.VARIANTS}
    texts = {v: [] for v in VERSIONS}
    for ds in IMP.DATASETS:
        pds = next(d for d in PIMP.DATASETS if d.name == ds.name)
        cells = {c: [] for c in ("SparseTSF", "fixed", "task", "TimeMixer")}
        for m in IMP.MASK_RATES:
            cells["SparseTSF"].append(IMP.parse_log(IMP.log_path_for("SparseTSF", ds, m))[0])
            cells["TimeMixer"].append(IMP.parse_log(IMP.log_path_for("TimeMixer", ds, m))[0])
            for v in VERSIONS:
                path = PIMP.report_path_for(pds, m, pv[REUSE_IMPUTATION[v]])
                texts[v].append(read(path))
                cells[v].append(PIMP.parse_log(path)["mse"])
        t.add(ds.name, **{c: (_mean(vs) if all(x is not None for x in vs) else None)
                          for c, vs in cells.items()})
    t.params = {v: _params(texts[v]) for v in VERSIONS}
    return t


def table_fewshot() -> Table:
    t = Table("Few-shot forecasting, 10% of training data (mean MSE over 4 horizons)", "MSE", ".4f",
              False)
    texts = {v: [] for v in VERSIONS}
    for ds in FS.DATASETS:
        cells = {c: [] for c in ("SparseTSF", "fixed", "task", "TimeMixer")}
        for p in FS.PRED_LENS:
            cells["SparseTSF"].append(FS.parse_log(FS.log_path_for(ds, p, "SparseTSF"))[0])
            cells["TimeMixer"].append(FS.parse_log(FS.log_path_for(ds, p, "TimeMixer"))[0])
            for v in VERSIONS:
                txt = suite_log("fewshot", v, "{}_pred{}".format(ds.name, p))
                texts[v].append(txt)
                cells[v].append(_last(MSE_PAT, txt))
        t.add(ds.name, **{c: (_mean(vs) if all(x is not None for x in vs) else None)
                          for c, vs in cells.items()})
    t.params = {v: _params(texts[v]) for v in VERSIONS}
    return t


def _zs_mse(text: str) -> Optional[float]:
    return _last(MSE_PAT, text) if len(re.findall(r"^mse:", text, re.M)) >= 2 else None


def table_zeroshot() -> Table:
    t = Table("Zero-shot transfer (mean MSE over 4 horizons)", "MSE", ".4f", False)
    texts = {v: [] for v in VERSIONS}
    for src, tgt in ZS.PAIRS:
        cells = {c: [] for c in ("SparseTSF", "fixed", "task", "TimeMixer")}
        for p in ZS.PRED_LENS:
            for model in ("SparseTSF", "TimeMixer"):
                cells[model].append(_zs_mse(read(ZS.log_path_for(src, tgt, p, model))))
            for v in VERSIONS:
                txt = suite_log("zeroshot", v, "{}_to_{}_pred{}".format(src, tgt, p))
                texts[v].append(txt)
                cells[v].append(_zs_mse(txt))
        t.add("{} → {}".format(src, tgt), **{c: (_mean(vs) if all(x is not None for x in vs) else None)
                                              for c, vs in cells.items()})
    t.params = {v: _params(texts[v]) for v in VERSIONS}
    return t


def table_anomaly() -> Table:
    t = Table("Anomaly detection (point-adjusted F1, %)", "F1", ".2f", True)
    for ds in AN.DATASETS:
        if not AN.available(ds):
            continue
        txt = suite_log("anomaly", "fixed", ds.name)
        f1 = _last(F1_PAT, txt)
        sp = AN.parse_log(ds, "SparseTSF")[2]
        t.add(ds.name, SparseTSF=sp, fixed=None if f1 is None else f1 * 100,
              task=sp, TimeMixer=AN.parse_log(ds, "TimeMixer")[2])
    t.params = {"fixed": _params([suite_log("anomaly", "fixed", d.name) for d in AN.DATASETS]),
                "task": "= SparseTSF"}
    return t


def table_classification() -> Table:
    t = Table("Classification (accuracy, %)", "accuracy", ".2f", True)
    flat = {x.name: x for x in PCLS.VARIANTS}[REUSE_CLASSIFICATION]
    texts = []
    for ds in CLS.DATASETS:
        path = PCLS.log_path_for(ds, flat)
        texts.append(read(path))
        acc = PCLS.parse_log(path)["acc"]
        t.add(ds, SparseTSF=CLS.parse_log(CLS.log_path_for(ds, "SparseTSF"))[0], fixed=acc, task=acc,
              TimeMixer=CLS.parse_log(CLS.log_path_for(ds, "TimeMixer"))[0])
    p = _params(texts)
    t.params = {"fixed": p, "task": p}
    return t


def energy_md() -> List[str]:
    em = CARBON.read_emissions()
    cols = [("SparseTSF", "SparseTSF"), ("fixed", LABEL["fixed"]), ("task", LABEL["task"]),
            ("TimeMixer", "TimeMixer"), ("TimeMixerPP", "TimeMixer++")]

    def key(k, ds, p):
        if k in VERSIONS:
            return carbon_setting(k, ds, p)
        if k in REF_MODELS:          # re-measured in the same session as SparseTSFPlus
            return carbon_ref_setting(k, ds, p)
        return CARBON.build_setting(ds, k, p)   # TimeMixer++: original July session only

    out = ["### Energy (matched carbon protocol: fixed 5 epochs, no early stopping)", "",
           "Total kWh over the 4 horizons of each dataset, one seed. Energy depends on the machine's",
           "state, so SparseTSF and TimeMixer were re-measured in the same session as SparseTSFPlus",
           "(the `Measured` row). TimeMixer++ comes from the original session and is not strictly",
           "comparable. None of this is comparable with the accuracy tables above.", "",
           "| Dataset | " + " | ".join(lbl for _, lbl in cols) + " |",
           "|---|" + "---:|" * len(cols)]
    totals = {k: 0.0 for k, _ in cols}
    complete = {k: True for k, _ in cols}
    dates = {k: set() for k, _ in cols}
    for ds in CARBON.DATASETS:
        row = {}
        for k, _ in cols:
            vals = []
            for p in CARBON.PRED_LENS:
                r = em.get(key(k, ds, p))
                vals.append(CARBON._f(r, "energy_consumed"))
                if r:
                    dates[k].add(r.get("timestamp", "")[:10])
            if all(x is not None for x in vals):
                row[k] = sum(vals)
                totals[k] += row[k]
            else:
                row[k] = None
                complete[k] = False
        out.append("| {} | {} |".format(ds.name, " | ".join(_fmt(row[k], ".5f") for k, _ in cols)))
    tot = {k: (totals[k] if complete[k] else None) for k, _ in cols}
    out.append("| **Total** | {} |".format(" | ".join("**{}**".format(_fmt(tot[k], ".5f")) for k, _ in cols)))
    base = tot["SparseTSF"]
    out.append("| _× SparseTSF_ | {} |".format(" | ".join(
        "_{}_".format("-" if base is None or tot[k] is None else "{:.2f}".format(tot[k] / base))
        for k, _ in cols)))
    out.append("| _Measured_ | {} |".format(" | ".join(
        "_{}_".format(", ".join(sorted(dates[k])) or "-") for k, _ in cols)))
    out.append("")
    return out


HEADER = """# SparseTSFPlus across the 8-task suite

SparseTSFPlus in two versions beside the locally run SparseTSF and TimeMixer, on every task and
dataset of the TimeMixer++ protocol that is obtainable here. Generated by
`benchmarks/run_sparsetsf_plus_suite.py` (idempotent).

- **SparseTSFPlus (fixed)** is **one configuration for everything**, the thesis's target form:
  learned-trend decomposition (forecasting) and phase mixing (every task that folds the series into
  periods) in the backbone; gated refinement and mask-aware aggregation whenever the input has
  missing values; default heads otherwise. This is the headline version.
- **SparseTSFPlus (per task)** uses the unit set that won each task in the ablations, unchanged across
  that task's datasets. It shows what per-task choice adds, so the gap between the two columns is
  the current cost of generality.
- Both use SparseTSF's own published period for every dataset, except the per-task PEMS cell (the
  ablation's phase mixing at period 4). All other training settings are SparseTSF's, so each row
  differs from SparseTSF only in the units.
- **Reused cells** (identical runs already existed): imputation, both versions, are the ablation's
  `allrg` and `u5g` variants (gated refinement); classification, both versions, is the ablation's
  default head (no unit applies to it), so it equals SparseTSF up to run-to-run noise; per-task
  PEMS is the ablation's `phase_p4`; per-task anomaly detection has no units and is SparseTSF
  itself.
- A dataset's mean needs all of its horizons or mask rates, and a task's mean covers only the rows
  where every column has a value. Lower is better except for F1 and accuracy.
- Imputation, anomaly detection and classification early-stop on test (upstream behaviour shared
  by every model), so read those rows as a sensitivity analysis.

```bash
venv/Scripts/python.exe benchmarks/run_sparsetsf_plus_suite.py
```
"""
START, END = "<!-- AUTO_PLUS_SUITE_START -->", "<!-- AUTO_PLUS_SUITE_END -->"


def summary_rows(tables: List[Table]) -> List[str]:
    out = ["### Summary", "",
           "| Task | Metric | SparseTSF | {} | {} | TimeMixer |".format(LABEL["fixed"], LABEL["task"]),
           "|---|---|---:|---:|---:|---:|"]
    for t in tables:
        m, n = t.means()
        out.append("| {} | {} {} | {} |".format(
            t.title.split(" (")[0], t.metric, "↑" if t.hb else "↓",
            " | ".join(_fmt(m[c], t.spec) for c in t.columns())))
    out.append("")
    return out


def update_md() -> None:
    tables = [table_longterm(), table_m4(), table_pems(), table_imputation(), table_fewshot(),
              table_zeroshot(), table_anomaly(), table_classification()]
    body = summary_rows(tables)
    for t in tables:
        body += t.markdown()
    body += energy_md()
    MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MD_PATH.exists():
        MD_PATH.write_text(HEADER + "\n" + START + "\n" + END + "\n", encoding="utf-8")
    text = MD_PATH.read_text(encoding="utf-8")
    si, ei = text.find(START), text.find(END)
    if si < 0 or ei < 0 or ei <= si:
        raise RuntimeError("Markers not found in " + MD_PATH.name)
    MD_PATH.write_text(text[: si + len(START)] + "\n" + "\n".join(body) + text[ei:], encoding="utf-8")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-from-logs-only", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated tasks: " + ",".join(ORDER))
    ap.add_argument("--versions", default="", help="comma-separated: fixed,task")
    args = ap.parse_args()
    if args.update_from_logs_only:
        update_md()
        print("[OK] wrote " + str(MD_PATH.relative_to(REPO_ROOT)))
        return
    only = set(filter(None, args.only.split(",")))
    versions = set(filter(None, args.versions.split(","))) or set(VERSIONS)
    jobs = [j for j in all_jobs(only) if j.task == "energyref" or j.version in versions]
    for i, job in enumerate(jobs, 1):
        if job.done(read(job.log)):
            print("[SKIP {}/{}] {} {} {}".format(i, len(jobs), job.task, job.version, job.key), flush=True)
            continue
        print("\n>>> [{}/{}] {} {} {} <<<".format(i, len(jobs), job.task, job.version, job.key), flush=True)
        run(job)
        if job.task in ("m4", "energy") or i % 8 == 0:
            update_md()
    update_md()
    print("\n[DONE] SparseTSFPlus suite complete.", flush=True)


if __name__ == "__main__":
    main()
