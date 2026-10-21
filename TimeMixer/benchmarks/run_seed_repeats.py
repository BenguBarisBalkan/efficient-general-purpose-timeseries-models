"""
Seed repeats: are the SparseTSFPlus gains larger than the run-to-run spread?

Every result before this used one seed (run.py fixes 2021). This runner repeats the headline
comparisons with seeds 2022 and 2023, so each number below is reported over three seeds:

  SparseTSF   the baseline
  fixed       the one-model SparseTSFPlus configuration of run_sparsetsf_plus_suite.py
  u5g         gated refinement alone, the per-task imputation form (imputation only)
  candidates  any configuration passed with --configs (see CANDIDATES below)

Tasks: long-term forecasting on ETTh1, ETTh2, ETTm1, ETTm2 and Weather (4 horizons), M4, PEMS,
few-shot, zero-shot, anomaly detection, and imputation on ETTh1 and ETTh2 (4 mask rates). Left out
for cost: Solar, Electricity and Traffic long-term, imputation on the 15-minute and Electricity
sets, and every TimeMixer repeat (its imputation alone takes about 10 hours per seed).

Seed 2021 is read from the existing logs, never re-run. Every repeated run is built from the
suite's own job for that cell (the command that produced the seed-2021 number), with the seed
added; SparseTSF's runs are the fixed configuration's commands with the unit flags removed, which
is SparseTSF's own command (the suite mirrors it by construction). So each seed differs from the
2021 run only in the seed.

A difference counts as a finding only if it has the same sign on all three seeds and its mean is
larger than twice the standard deviation of the per-seed differences.

Idempotent: a run whose log already holds its final metric is skipped.

    venv/Scripts/python.exe benchmarks/run_seed_repeats.py
    venv/Scripts/python.exe benchmarks/run_seed_repeats.py --only longterm,m4 --seeds 2022
    venv/Scripts/python.exe benchmarks/run_seed_repeats.py --update-from-logs-only
"""

import argparse
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Callable, Dict, List, Optional, Tuple

import run_anomaly_benchmarks as AN
import run_fewshot_benchmarks as FS
import run_imputation_benchmarks as IMP
import run_pems_benchmarks as PEMS
import run_sparsetsf_benchmarks as LT
import run_sparsetsf_plus_imputation as PIMP
import run_sparsetsf_plus_suite as SUITE
import run_zeroshot_benchmarks as ZS

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "logs" / "run_logs_seeds"
CKPT_DIR = REPO_ROOT / "checkpoints"
M4_RESULTS = REPO_ROOT / "m4_results"
M4_SEEDS = M4_RESULTS / "seeds"
MD_PATH = REPO_ROOT / "reports" / "results_seed_repeats.md"

BASE_SEED = 2021
SEEDS = [2022, 2023]
LT_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"]
IMP_DATASETS = ["ETTh1", "ETTh2"]

# configuration -> (suite version whose commands it uses, imputation ablation variant)
CONFIGS: Dict[str, Tuple[Optional[str], Optional[str]]] = {
    "SparseTSF": ("fixed", "base"),
    "fixed": ("fixed", "allrg"),
    "u5g": (None, "u5g"),
}
# Candidates from the 24-hour plan, repeated only when passed with --configs.
CANDIDATES: Dict[str, Tuple[Optional[str], Optional[str]]] = {
    "chmean": ("chmean", "allrg"),
    "chcorr": ("chcorr", "allrg"),
    "allrg_pc": ("fixed", "allrg_pc"),
    "allrg_pm": ("fixed", "allrg_pm"),
}
ALL_CONFIGS = dict(CONFIGS, **CANDIDATES)
TASKS = ["longterm", "m4", "pems", "fewshot", "zeroshot", "anomaly", "imputation"]


@dataclass
class SeedJob:
    task: str
    config: str
    key: str
    seed: int
    cmd: List[str]
    done: Callable[[str], bool]
    m4_subset: Optional[str] = None
    zero_shot: Optional[Tuple[List[str], str, str]] = None

    @property
    def log(self) -> Path:
        return LOGS_DIR / "{}_{}_{}_s{}.log".format(self.task, self.config, self.key, self.seed)


def _strip_units(cmd: List[str], flags: List[str]) -> List[str]:
    """Remove the suite's unit flags (appended as one block; zero-shot adds --is_training after
    it) and switch the model to SparseTSF."""
    n = len(flags)
    starts = [i for i in range(len(cmd) - n + 1) if cmd[i:i + n] == flags]
    assert starts, "unit flags not found in the command"
    out = cmd[:starts[-1]] + cmd[starts[-1] + n:]
    out[out.index("--model") + 1] = "SparseTSF"
    assert not any(a.startswith("--stsf_") for a in out), "a unit flag survived"
    return out


def _seeded(cmd: List[str], seed: int, config: str, flags: List[str]) -> Tuple[List[str], str, str]:
    """-> (command, old model_id, new model_id) for one seed."""
    out = _strip_units(cmd, flags) if config == "SparseTSF" else list(cmd)
    i = out.index("--model_id") + 1
    old = out[i]
    out[i] = "{}_{}_s{}".format(old, config, seed)
    return out + ["--seed", str(seed)], old, out[i]


def _unit_flags(task: str, version: str) -> List[str]:
    return SUITE.ANOMALY_FLAGS.get(version, []) if task == "anomaly" else SUITE.FORECAST_FLAGS[version]


def _m4_seed_csv(config: str, seed: int, subset: str) -> Path:
    return M4_SEEDS / "{}_s{}".format(config, seed) / "{}_forecast.csv".format(subset)


def suite_jobs_for(task: str, version: str):
    builder = {"longterm": SUITE.jobs_longterm, "m4": SUITE.jobs_m4, "pems": SUITE.jobs_pems,
               "fewshot": SUITE.jobs_fewshot, "zeroshot": SUITE.jobs_zeroshot,
               "anomaly": SUITE.jobs_anomaly}[task]
    jobs = builder(version)
    if task == "longterm":
        jobs = [j for j in jobs if j.key.rsplit("_pred", 1)[0] in LT_DATASETS]
    return jobs


def imputation_cmd(ds, mask_rate: float, flags, model: str, model_id: str, seed: int) -> List[str]:
    """The imputation ablation's command (PIMP.run_one), with a model, run name and seed."""
    return [
        sys.executable, "-u", str(REPO_ROOT / "run.py"),
        "--task_name", "imputation", "--is_training", "1",
        "--model_id", model_id, "--model", model,
        "--data", ds.data, "--root_path", str(ds.root_path), "--data_path", ds.data_path,
        "--features", "M", "--seq_len", str(PIMP.SEQ_LEN), "--label_len", "0", "--pred_len", "0",
        "--enc_in", str(ds.enc_in), "--dec_in", str(ds.enc_in), "--c_out", str(ds.enc_in),
        "--mask_rate", str(mask_rate), "--period_len", str(ds.period_len), "--model_type", "linear",
        "--learning_rate", str(PIMP.LEARNING_RATE), "--lradj", PIMP.LRADJ,
        "--train_epochs", str(PIMP.TRAIN_EPOCHS), "--batch_size", str(PIMP.BATCH_SIZE),
        "--num_workers", "0", "--des", PIMP.DES, "--itr", "1",
    ] + list(flags) + ["--seed", str(seed)]


def jobs_for(task: str, config: str, seed: int) -> List[SeedJob]:
    version, imp_variant = ALL_CONFIGS[config]
    if task == "imputation":
        if imp_variant is None:
            return []
        pv = {v.name: v for v in PIMP.VARIANTS}[imp_variant]
        model = "SparseTSF" if config == "SparseTSF" else PIMP.MODEL
        out = []
        for ds in PIMP.DATASETS:
            if ds.name not in IMP_DATASETS:
                continue
            for m in PIMP.FULL_MASKS:
                mid = "SEED_IMP_{}_{}_m{}_s{}".format(ds.name, config, m, seed)
                out.append(SeedJob(task, config, "{}_mask{}".format(ds.name, m), seed,
                                   imputation_cmd(ds, m, pv.flags, model, mid, seed),
                                   lambda t: PIMP.MSE_PAT.search(t) is not None))
        return out
    if version is None:
        return []
    out = []
    for j in suite_jobs_for(task, version):
        flags = _unit_flags(task, version)
        if config == "SparseTSF" and not flags:
            continue            # nothing to strip: the job is not a unit run
        cmd, old, new = _seeded(j.cmd, seed, config, flags)
        sj = SeedJob(task, config, j.key, seed, cmd, j.done)
        if task == "m4":
            subset = j.key
            sj.m4_subset = subset
            sj.done = (lambda t, c=config, s=seed, sub=subset:
                       "test shape:" in t and _m4_seed_csv(c, s, sub).exists())
        if j.zero_shot:
            test_cmd, src_set, tgt_set = j.zero_shot
            tcmd, _, _ = _seeded(test_cmd, seed, config, flags)
            src_set, tgt_set = src_set.replace(old, new), tgt_set.replace(old, new)
            if config == "SparseTSF":
                src_set = src_set.replace("_none_SparseTSFPlus_", "_none_SparseTSF_")
                tgt_set = tgt_set.replace("_none_SparseTSFPlus_", "_none_SparseTSF_")
            sj.zero_shot = (tcmd, src_set, tgt_set)
        out.append(sj)
    return out


def m4_model_dir(config: str) -> Path:
    return M4_RESULTS / ("SparseTSF" if config == "SparseTSF" else SUITE.MODEL)


def protect_m4_originals() -> None:
    """run.py writes M4 forecasts to m4_results/<model>/, where the seed-2021 SparseTSF forecasts
    live. Keep a copy of the originals once, so every seed job can put them back."""
    src = M4_RESULTS / "SparseTSF"
    dst = M4_SEEDS / "SparseTSF_s{}".format(BASE_SEED)
    if src.exists() and not dst.exists():
        shutil.copytree(str(src), str(dst))


def run(job: SeedJob) -> None:
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
            shutil.copyfile(str(src), str(dst))
            f.write("\n===== TEST on target (zero-shot) =====\n"); f.flush()
            subprocess.run(test_cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
    if job.m4_subset:
        written = m4_model_dir(job.config) / "{}_forecast.csv".format(job.m4_subset)
        if written.exists():
            target = _m4_seed_csv(job.config, job.seed, job.m4_subset)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(written), str(target))
        if job.config == "SparseTSF":
            original = _m4_seed_csv("SparseTSF", BASE_SEED, job.m4_subset)
            if original.exists():
                shutil.copyfile(str(original), str(written))
    print("      finished in {:.1f}s".format(time.perf_counter() - start), flush=True)


# ── values: seed 2021 from the existing logs, other seeds from this runner's logs ─────────────

def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def _seed_log(task: str, config: str, key: str, seed: int) -> str:
    return _read(LOGS_DIR / "{}_{}_{}_s{}.log".format(task, config, key, seed))


def _mean_or_none(vals: List[Optional[float]]) -> Optional[float]:
    return None if not vals or any(v is None for v in vals) else sum(vals) / len(vals)


def cell(task: str, config: str, key: str, seed: int) -> Optional[float]:
    """One dataset-level value (a horizon, subset, pair or mask cell)."""
    version, imp_variant = ALL_CONFIGS[config]
    if task == "imputation":
        if seed == BASE_SEED:
            name, m = key.split("_mask")
            pds = next(d for d in PIMP.DATASETS if d.name == name)
            if config == "SparseTSF":
                ids = next(d for d in IMP.DATASETS if d.name == name)
                return IMP.parse_log(IMP.log_path_for("SparseTSF", ids, float(m)))[0]
            pv = {v.name: v for v in PIMP.VARIANTS}[imp_variant]
            return PIMP.parse_log(PIMP.report_path_for(pds, float(m), pv))["mse"]
        return SUITE._last(SUITE.MSE_PAT, _seed_log(task, config, key, seed))
    if seed == BASE_SEED:
        if config == "SparseTSF":
            return sparsetsf_2021(task, key)
        text = SUITE.suite_log(task, version, key)
    else:
        text = _seed_log(task, config, key, seed)
    if task == "pems":
        return SUITE._last(SUITE.MAE_PAT, text) if "RMSE:" in text else None
    if task == "zeroshot":
        return SUITE._zs_mse(text)
    if task == "anomaly":
        f1 = SUITE._last(SUITE.F1_PAT, text)
        return None if f1 is None else f1 * 100
    return SUITE._last(SUITE.MSE_PAT, text)


def sparsetsf_2021(task: str, key: str) -> Optional[float]:
    if task == "longterm":
        name, p = key.rsplit("_pred", 1)
        ds = next(d for d in LT.DATASETS if d.name == name)
        return LT.parse_log(LT.log_path_for(ds, 96, int(p)))[0]
    if task == "pems":
        ds = next(d for d in PEMS.DATASETS if d.name == key)
        return PEMS.parse_log(PEMS.log_path_for("SparseTSF", ds))[0]
    if task == "fewshot":
        name, p = key.rsplit("_pred", 1)
        ds = next(d for d in FS.DATASETS if d.name == name)
        return FS.parse_log(FS.log_path_for(ds, int(p), "SparseTSF"))[0]
    if task == "zeroshot":
        pair, p = key.rsplit("_pred", 1)
        src, tgt = pair.split("_to_")
        return SUITE._zs_mse(_read(ZS.log_path_for(src, tgt, int(p), "SparseTSF")))
    if task == "anomaly":
        ds = next(d for d in AN.DATASETS if d.name == key)
        return AN.parse_log(ds, "SparseTSF")[2]
    raise ValueError(task)


def m4_owa(config: str, seed: int) -> Optional[float]:
    if seed == BASE_SEED:
        folder = M4_RESULTS / ("SparseTSF" if config == "SparseTSF"
                               else "{}_{}".format(SUITE.MODEL, ALL_CONFIGS[config][0]))
    else:
        folder = M4_SEEDS / "{}_s{}".format(config, seed)
    return SUITE._m4_owa(folder)


def task_keys(task: str) -> List[str]:
    if task == "imputation":
        return ["{}_mask{}".format(d, m) for d in IMP_DATASETS for m in PIMP.FULL_MASKS]
    return [j.key for j in suite_jobs_for(task, "fixed")]


def task_value(task: str, config: str, seed: int) -> Optional[float]:
    """The task's headline number for one configuration and seed: the mean over its cells."""
    version, imp_variant = ALL_CONFIGS[config]
    if (task == "imputation" and imp_variant is None) or (task != "imputation" and version is None):
        return None
    if task == "m4":
        return m4_owa(config, seed)
    return _mean_or_none([cell(task, config, k, seed) for k in task_keys(task)])


# ── report ───────────────────────────────────────────────────────────────────

METRIC = {"longterm": ("MSE", False), "m4": ("OWA", False), "pems": ("MAE", False),
          "fewshot": ("MSE", False), "zeroshot": ("MSE", False), "anomaly": ("F1", True),
          "imputation": ("MSE", False)}
TITLE = {"longterm": "Long-term (ETT×4, Weather; 4 horizons)", "m4": "M4 (6 subsets)",
         "pems": "PEMS (4 datasets)", "fewshot": "Few-shot (6 datasets)",
         "zeroshot": "Zero-shot (6 pairs)", "anomaly": "Anomaly (4 datasets)",
         "imputation": "Imputation (ETTh1, ETTh2; 4 mask rates)"}


def _fmt(x: Optional[float], spec: str = ".4f") -> str:
    return "-" if x is None else format(x, spec)


def comparisons(configs: List[str]) -> List[Tuple[str, str]]:
    pairs = [("fixed", "SparseTSF"), ("u5g", "SparseTSF"), ("u5g", "fixed")]
    pairs += [(c, "fixed") for c in configs if c in CANDIDATES]
    return pairs


HEADER = """# Seed repeats

Every earlier result in this study used one seed (2021). Here the headline comparisons are
repeated with seeds 2022 and 2023 by `benchmarks/run_seed_repeats.py`; seed 2021 is read from the
existing logs. Each repeated run is the exact command behind the seed-2021 number with only the
seed changed.

- **Configurations:** SparseTSF; the fixed one-model SparseTSFPlus; gated refinement alone
  (`u5g`, the per-task imputation form, imputation only); and any candidate configuration.
- **Tasks:** long-term forecasting on ETTh1, ETTh2, ETTm1, ETTm2 and Weather, M4, PEMS, few-shot,
  zero-shot, anomaly detection, and imputation on ETTh1 and ETTh2. Solar, Electricity and Traffic
  long-term, the other imputation datasets and every TimeMixer repeat are left out for cost.
- **What counts as a finding:** a difference with the same sign on all three seeds whose mean is
  larger than twice the standard deviation of the per-seed differences.
- Imputation, anomaly detection and classification early-stop on test (upstream behaviour), so
  read those rows as a sensitivity analysis.

```bash
venv/Scripts/python.exe benchmarks/run_seed_repeats.py
```
"""
START, END = "<!-- AUTO_SEEDS_START -->", "<!-- AUTO_SEEDS_END -->"


def update_md(configs: List[str]) -> None:
    seeds = [BASE_SEED] + SEEDS
    lines = ["### Per task, per seed (mean over the task's cells)", "",
             "| Task | Metric | Configuration | " + " | ".join(str(s) for s in seeds)
             + " | Mean ± std | Spread (% of mean) |",
             "|---|---|---|" + "---:|" * (len(seeds) + 2)]
    values: Dict[Tuple[str, str], List[Optional[float]]] = {}
    for task in TASKS:
        metric, hb = METRIC[task]
        spec = ".3f" if task == "m4" else (".2f" if task in ("pems", "anomaly") else ".4f")
        for c in configs:
            vals = [task_value(task, c, s) for s in seeds]
            values[(task, c)] = vals
            if all(v is None for v in vals):
                continue
            done = [v for v in vals if v is not None]
            ms = "-"
            spread = "-"
            if len(done) == len(seeds):
                ms = "{} ± {}".format(_fmt(mean(done), spec), _fmt(stdev(done), spec))
                spread = "{:.1f}%".format(100 * stdev(done) / abs(mean(done)))
            lines.append("| {} | {} {} | {} | {} | {} | {} |".format(
                TITLE[task], metric, "↑" if hb else "↓", c,
                " | ".join(_fmt(v, spec) for v in vals), ms, spread))
    lines.append("")
    lines += ["### Comparisons (relative difference per seed)", "",
              "Negative is better for error metrics; for anomaly F1 the difference is in points and "
              "positive is better.", "",
              "| Task | Comparison | " + " | ".join(str(s) for s in seeds)
              + " | Mean | Std | Finding? |",
              "|---|---|" + "---:|" * (len(seeds) + 2) + "---|"]
    for task in TASKS:
        for a, b in comparisons(configs):
            if a not in configs or b not in configs:
                continue
            va, vb = values.get((task, a)), values.get((task, b))
            if not va or not vb:
                continue
            diffs = []
            for x, y in zip(va, vb):
                if x is None or y is None:
                    diffs.append(None)
                elif task == "anomaly":
                    diffs.append(x - y)
                else:
                    diffs.append(100 * (x - y) / y)
            if all(d is None for d in diffs):
                continue
            unit = " pts" if task == "anomaly" else "%"
            cells = " | ".join("-" if d is None else "{:+.2f}{}".format(d, unit) for d in diffs)
            if all(d is not None for d in diffs):
                m, sd = mean(diffs), stdev(diffs)
                same = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)
                finding = "yes" if same and abs(m) > 2 * sd else "no"
                lines.append("| {} | {} vs {} | {} | {:+.2f}{} | {:.2f}{} | {} |".format(
                    TITLE[task], a, b, cells, m, unit, sd, unit, finding))
            else:
                lines.append("| {} | {} vs {} | {} | - | - | incomplete |".format(
                    TITLE[task], a, b, cells))
    lines.append("")
    MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MD_PATH.exists():
        MD_PATH.write_text(HEADER + "\n" + START + "\n" + END + "\n", encoding="utf-8")
    text = MD_PATH.read_text(encoding="utf-8")
    si, ei = text.find(START), text.find(END)
    if si < 0 or ei < 0 or ei <= si:
        raise RuntimeError("Markers not found in " + MD_PATH.name)
    MD_PATH.write_text(text[: si + len(START)] + "\n" + "\n".join(lines) + "\n" + text[ei:],
                       encoding="utf-8")
    print("[OK] wrote " + str(MD_PATH.relative_to(REPO_ROOT)), flush=True)


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-from-logs-only", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated tasks: " + ",".join(TASKS))
    ap.add_argument("--seeds", default="", help="comma-separated seeds (default: 2022,2023)")
    ap.add_argument("--configs", default="",
                    help="comma-separated configurations (default: {}; candidates: {})".format(
                        ",".join(CONFIGS), ",".join(CANDIDATES)))
    args = ap.parse_args()
    configs = [c for c in args.configs.split(",") if c] or list(CONFIGS)
    for c in configs:
        if c not in ALL_CONFIGS:
            raise SystemExit("unknown configuration: " + c)
    if args.update_from_logs_only:
        update_md(configs)
        return
    seeds = [int(s) for s in args.seeds.split(",") if s] or SEEDS
    only = [t for t in args.only.split(",") if t] or TASKS
    protect_m4_originals()
    jobs = [j for s in seeds for t in only for c in configs for j in jobs_for(t, c, s)]
    for i, job in enumerate(jobs, 1):
        if job.done(_read(job.log)):
            print("[SKIP {}/{}] {} {} {} s{}".format(i, len(jobs), job.task, job.config, job.key, job.seed))
            continue
        print("\n>>> [{}/{}] {} {} {} s{} <<<".format(i, len(jobs), job.task, job.config, job.key, job.seed))
        run(job)
        if i % 10 == 0:
            update_md(configs)
    update_md(configs)
    print("\n[DONE] seed repeats complete.")


if __name__ == "__main__":
    main()
