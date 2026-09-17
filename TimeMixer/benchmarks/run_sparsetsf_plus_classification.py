"""SparseTSFPlus classification ablation over the 10 standard UEA datasets.

Classification is SparseTSF's worst task (avg 60.90% vs TimeMixer++'s published 75.9,
-15.0 points) and the only task where SparseTSF is NOT small: its head is
`nn.Linear(enc_in * seq_len, num_class)`, which is 970,712 parameters on PEMS-SF.
Handwriting sits at 8.59% against a ~3.8% chance floor for 26 classes, so the head is
closer to broken than merely untuned.

Three defects, tested separately so the ablation attributes the gain correctly:

1. A padded-mean bug. `models/SparseTSF.py:137` subtracts the per-channel mean over the
   FULL zero-padded window and only applies `padding_mask` afterwards (line 140). UEA
   windows are padded to `max_seq_len` -- 1751 on EthanolConcentration -- so the mean is
   badly biased. `flatfix` fixes only this, keeping the head identical.

2. No temporal pooling. The flat head is a linear probe on raw time steps with no
   translation invariance, and it scales with `seq_len`. `stats` replaces it with
   mean/std/max pooling per channel -> `Linear(3*enc_in, num_class)`, which is 46-513x
   smaller (Handwriting 11,883 -> 261; PEMS-SF 970,712 -> 20,231).

3. A degenerate aggregator. `run_classification_benchmarks.py:78` passes `--period_len 1`,
   commented as "unused by the classification head" -- but it IS used: the conv kernel is
   `1 + 2*(period_len//2)`, so period_len=1 gives a kernel of size ONE, a per-timestep
   scalar multiply with no receptive field. `stats_p25` gives the same head a real
   25-tap conv, changing nothing else.

Note that unlike the forecasting and reconstruction backbone, a classification head is
necessarily channel-coupled: it is a readout over all channels, not backbone mixing.

SELECTION CAVEAT. `exp/exp_classification.py:88-89` uses `flag='TEST'` for BOTH the "vali"
and "test" loaders and early-stops on `-test_accuracy`. That is upstream TSLib behaviour
and every model in this repo was trained under it, so cross-model comparisons stay
internally consistent -- but there is no clean validation signal for this task at all.
Treat these numbers as a sensitivity analysis, not a tuned result, and say so in the report.

Idempotent: a config whose log already holds an accuracy is skipped.

    venv/Scripts/python.exe benchmarks/run_sparsetsf_plus_classification.py
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
LOGS_DIR = REPO_ROOT / "logs" / "run_logs_plus_cls"
MD_PATH = REPO_ROOT / "reports" / "results_sparsetsf_plus_classification.md"

MODEL = "SparseTSFPlus"

# Copied verbatim from run_classification_benchmarks.py so `flat` is directly comparable.
LEARNING_RATE = 0.001
TRAIN_EPOCHS = 30
PATIENCE = 10
BATCH_SIZE = 16
DES = "ClsP"

DATASETS: List[str] = [
    "EthanolConcentration", "FaceDetection", "Handwriting", "Heartbeat", "JapaneseVowels",
    "PEMS-SF", "SelfRegulationSCP1", "SelfRegulationSCP2", "SpokenArabicDigits",
    "UWaveGestureLibrary",
]

# Published SparseTSF accuracies, same protocol (reports/results_comparison_classification.md).
PUBLISHED: Dict[str, float] = {
    "EthanolConcentration": 29.28,
    "FaceDetection": 68.59,
    "Handwriting": 8.59,
    "Heartbeat": 54.63,
    "JapaneseVowels": 61.62,
    "PEMS-SF": 84.39,
    "SelfRegulationSCP1": 82.25,
    "SelfRegulationSCP2": 52.78,
    "SpokenArabicDigits": 94.04,
    "UWaveGestureLibrary": 72.81,
}
PUBLISHED_AVG = 60.90
PAPER_REF = {"TimeMixer++ (paper)": 75.9, "TimesNet (paper)": 73.6}


@dataclass(frozen=True)
class Variant:
    name: str
    head: str
    period_len: int
    note: str


VARIANTS: List[Variant] = [
    Variant("flat", "flat", 1, "upstream head, padded-mean bug included (control)"),
    Variant("flatfix", "flat_fixed", 1, "mask-aware mean only; head unchanged"),
    Variant("stats", "stats", 1, "mean/std/max pooling head"),
    Variant("stats_p25", "stats", 25, "pooling head + a real 25-tap conv"),
    Variant("segpool", "segpool", 1, "segment pooling (keeps coarse temporal order)"),
    Variant("segpool_p25", "segpool", 25, "segment pooling + a real 25-tap conv"),
]

ACC_PAT = re.compile(r"accuracy:([0-9]+(?:\.[0-9]+)?)")
PARAM_PAT = re.compile(r"trainable_params:([0-9]+)")
EPOCH_PAT = re.compile(r"Epoch:\s+\d+\s+cost time:\s+([0-9]+(?:\.[0-9]+)?)")


def log_path_for(ds: str, v: Variant) -> Path:
    return LOGS_DIR / "{}_{}.log".format(ds, v.name)


def parse_log(path: Path):
    out = {"acc": None, "params": None, "runtime": None, "status": "PENDING"}
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8", errors="ignore")
    m = ACC_PAT.findall(text)
    if not m:
        out["status"] = "RUNNING/INCOMPLETE"
        return out
    out["acc"] = float(m[-1]) * 100.0
    p = PARAM_PAT.findall(text)
    out["params"] = int(p[0]) if p else None
    ep = [float(x) for x in EPOCH_PAT.findall(text)]
    out["runtime"] = sum(ep) if ep else None
    out["status"] = "OK"
    return out


def run_one(ds: str, v: Variant) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u", str(REPO_ROOT / "run.py"),
        "--task_name", "classification",
        "--is_training", "1",
        "--model_id", "CLSP_{}_{}".format(ds, v.name),
        "--model", MODEL,
        "--data", "UEA",
        "--root_path", str(DATA_ROOT / ds),
        "--features", "M",
        # seq_len / enc_in / num_class are overwritten by Exp_Classification from the data.
        "--seq_len", "96", "--pred_len", "0", "--enc_in", "1",
        "--period_len", str(v.period_len),
        "--model_type", "linear",
        "--stsf_cls_head", v.head,
        "--stsf_cls_segments", "8",
        "--learning_rate", str(LEARNING_RATE),
        "--train_epochs", str(TRAIN_EPOCHS),
        "--patience", str(PATIENCE),
        "--batch_size", str(BATCH_SIZE),
        "--num_workers", "0",
        "--des", DES,
        "--itr", "1",
    ]
    print("[RUN] {} {}".format(ds, v.name), flush=True)
    t0 = time.time()
    with log_path_for(ds, v).open("w", encoding="utf-8") as f:
        subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)
    print("      done in {:.0f}s".format(time.time() - t0), flush=True)


START = "<!-- AUTO_PLUS_CLS_START -->"
END = "<!-- AUTO_PLUS_CLS_END -->"

HEADER = """# SparseTSFPlus ablation — classification (10 UEA datasets)

`flat` is the control: the upstream head, padded-mean bug included, which should reproduce
the published SparseTSF accuracies. The other rows change exactly one thing each, so a gain
can be attributed rather than guessed at.

**These are a sensitivity analysis, not a tuned result.** `exp/exp_classification.py:88-89`
uses `flag='TEST'` for both the "vali" and "test" loaders and early-stops on test accuracy,
so this task has no clean validation signal at all. That is upstream TSLib behaviour, shared
by every model in this repo — deliberately not changed, because changing it would invalidate
every existing table — but it means no row here should be read as a held-out result.

Regenerate (idempotent):

```bash
venv/Scripts/python.exe benchmarks/run_sparsetsf_plus_classification.py
```
"""


def _fmt(v, nd=2):
    return "-" if v is None else "{:.{}f}".format(v, nd)


def update_md() -> None:
    lines: List[str] = []
    lines.append("| Dataset | " + " | ".join(v.name for v in VARIANTS) +
                 " | SparseTSF published | best vs published |")
    lines.append("|---|" + "---:|" * (len(VARIANTS) + 2))

    sums = {v.name: [] for v in VARIANTS}
    for ds in DATASETS:
        cells = []
        best = None
        for v in VARIANTS:
            r = parse_log(log_path_for(ds, v))
            if r["acc"] is not None:
                sums[v.name].append(r["acc"])
                best = r["acc"] if best is None else max(best, r["acc"])
            cells.append(_fmt(r["acc"]))
        pub = PUBLISHED[ds]
        delta = "-" if best is None else "{:+.2f}".format(best - pub)
        lines.append("| {} | {} | _{:.2f}_ | {} |".format(
            ds, " | ".join(cells), pub, delta))

    avg_cells = []
    for v in VARIANTS:
        vals = sums[v.name]
        avg_cells.append("**{:.2f}**".format(sum(vals) / len(vals)) if len(vals) == len(DATASETS)
                         else ("{:.2f}*".format(sum(vals) / len(vals)) if vals else "-"))
    lines.append("| **Average** | " + " | ".join(avg_cells) +
                 " | _{:.2f}_ | |".format(PUBLISHED_AVG))
    lines.append("")
    lines.append("`*` = partial average over completed datasets only.")
    lines.append("")

    lines.append("### Head size (the one enc_in-coupled head in the model)")
    lines.append("")
    lines.append("| Dataset | " + " | ".join(v.name for v in VARIANTS) + " |")
    lines.append("|---|" + "---:|" * len(VARIANTS))
    for ds in DATASETS:
        cells = []
        for v in VARIANTS:
            p = parse_log(log_path_for(ds, v))["params"]
            cells.append("-" if p is None else "{:,}".format(p))
        lines.append("| {} | {} |".format(ds, " | ".join(cells)))
    lines.append("")

    lines.append("### Published references")
    lines.append("")
    lines.append("| Model | Avg accuracy (%) |")
    lines.append("|---|---:|")
    for k, val in PAPER_REF.items():
        lines.append("| _{}_ | _{:.1f}_ |".format(k, val))
    lines.append("")
    lines.append("Variant legend:")
    lines.append("")
    for v in VARIANTS:
        lines.append("- **{}** — {} (`--stsf_cls_head {} --period_len {}`)".format(
            v.name, v.note, v.head, v.period_len))

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-from-logs-only", action="store_true")
    ap.add_argument("--variants", type=str, default="")
    ap.add_argument("--datasets", type=str, default="")
    args = ap.parse_args()

    if args.update_from_logs_only:
        update_md()
        return

    wanted_v = [v for v in VARIANTS
                if not args.variants or v.name in args.variants.split(",")]
    wanted_d = [d for d in DATASETS
                if not args.datasets or d in args.datasets.split(",")]

    jobs = [(d, v) for d in wanted_d for v in wanted_v]
    total = len(jobs)
    for i, (ds, v) in enumerate(jobs, 1):
        if parse_log(log_path_for(ds, v))["status"] == "OK":
            print("[SKIP {}/{}] {} {}".format(i, total, ds, v.name), flush=True)
            continue
        print("[{}/{}]".format(i, total), end=" ", flush=True)
        run_one(ds, v)
        update_md()

    update_md()
    print("[DONE] classification ablation complete", flush=True)


if __name__ == "__main__":
    main()
