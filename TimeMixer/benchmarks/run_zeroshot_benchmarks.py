"""
Zero-shot (cross-dataset transfer) forecasting benchmark — SparseTSF.

Matches the paper's setup (TimeMixer++ ICLR'25, Table 6): train on dataset A, evaluate on an unseen
dataset B with no further training, over the 6 ETT transfer pairs, seq_len=96,
pred_len {96,192,336,720}, MSE/MAE.

Mechanism (no changes to run.py needed): run.py derives a `setting` string from the args and
  • `--is_training 1` trains and saves  ./checkpoints/<setting>/checkpoint.pth
  • `--is_training 0` tests  by LOADING ./checkpoints/<setting>/checkpoint.pth
Since the two settings differ only in the `--data` field, we train on A, copy A's checkpoint into
B's setting directory, then run B with `--is_training 0`. The model never sees B's training data.

Architecture note: SparseTSF's linear layer is (seq_len/period_len -> pred_len/period_len), so the
SOURCE dataset's `period_len` must also be used for the target, otherwise the checkpoint shapes
would not match. All pred_lens here are divisible by both 24 and 4, so this is always valid.

Idempotent: a pair whose log already holds a valid MSE/MAE is skipped.

    venv/Scripts/python.exe benchmarks/run_zeroshot_benchmarks.py
"""

import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT.parent / "dataset"
LOGS_DIR = REPO_ROOT / "logs" / "run_logs_zeroshot"
CKPT_DIR = REPO_ROOT / "checkpoints"
MD_PATH = REPO_ROOT / "reports" / "results_comparison_zeroshot.md"

MODEL = "SparseTSF"
SEQ_LEN = 96
PRED_LENS = [96, 192, 336, 720]
LEARNING_RATE = 0.02
BATCH_SIZE = 256
TRAIN_EPOCHS = 30
PATIENCE = 5
LRADJ = "type3"
DES = "ZeroShot"

# dataset name -> (data arg, csv, root, enc_in, natural period_len)
DS = {
    "ETTh1": ("ETTh1", "ETTh1.csv", DATA_ROOT / "ETT-small", 7, 24),
    "ETTh2": ("ETTh2", "ETTh2.csv", DATA_ROOT / "ETT-small", 7, 24),
    "ETTm1": ("ETTm1", "ETTm1.csv", DATA_ROOT / "ETT-small", 7, 4),
    "ETTm2": ("ETTm2", "ETTm2.csv", DATA_ROOT / "ETT-small", 7, 4),
}

PAIRS = [("ETTh1", "ETTh2"), ("ETTh1", "ETTm2"), ("ETTh2", "ETTh1"),
         ("ETTm1", "ETTh2"), ("ETTm1", "ETTm2"), ("ETTm2", "ETTm1")]

# Published reference (TimeMixer++ paper, Table 6; avg over 4 pred_lens) MSE/MAE.
PAPER_REF = {
    "ETTh1->ETTh2": {"TimeMixer": (0.427, 0.424), "TimeMixer++": (0.367, 0.391)},
    "ETTh1->ETTm2": {"TimeMixer": (0.361, 0.397), "TimeMixer++": (0.301, 0.357)},
    "ETTh2->ETTh1": {"TimeMixer": (0.679, 0.577), "TimeMixer++": (0.511, 0.498)},
    "ETTm1->ETTh2": {"TimeMixer": (0.452, 0.441), "TimeMixer++": (0.417, 0.422)},
    "ETTm1->ETTm2": {"TimeMixer": (0.329, 0.357), "TimeMixer++": (0.291, 0.331)},
    "ETTm2->ETTm1": {"TimeMixer": (0.554, 0.478), "TimeMixer++": (0.427, 0.448)},
}

NUM = r"([0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?)"
MSE_PAT = re.compile(r"mse:" + NUM)
MAE_PAT = re.compile(r"mae:" + NUM)


def _last(pat: re.Pattern, text: str) -> Optional[float]:
    m = pat.findall(text)
    return float(m[-1]) if m else None


def build_setting(model_id: str, data: str, pred_len: int) -> str:
    """Reproduce run.py's `setting` string (defaults: dm16 nh8 el2 dl1 df32 fc1 ebtimeF dtTrue)."""
    return (f"long_term_forecast_{model_id}_none_{MODEL}_{data}"
            f"_sl{SEQ_LEN}_pl{pred_len}_dm16_nh8_el2_dl1_df32_fc1_ebtimeF_dtTrue_{DES}_0")


def log_path_for(src: str, tgt: str, pred_len: int) -> Path:
    return LOGS_DIR / f"{src}_to_{tgt}_pred{pred_len}.log"


def parse_log(path: Path) -> Tuple[Optional[float], Optional[float], str]:
    if not path.exists():
        return None, None, "PENDING"
    text = path.read_text(encoding="utf-8", errors="ignore")
    mse, mae = _last(MSE_PAT, text), _last(MAE_PAT, text)
    if mse is None or mae is None:
        return None, None, "RUNNING/INCOMPLETE"
    return mse, mae, "OK"


def _base_args(name: str, model_id: str, pred_len: int, period_len: int) -> List[str]:
    data, csv, root, enc_in, _ = DS[name]
    return [
        sys.executable, "-u", str(REPO_ROOT / "run.py"),
        "--task_name", "long_term_forecast",
        "--model_id", model_id,
        "--model", MODEL,
        "--data", data,
        "--root_path", str(root),
        "--data_path", csv,
        "--features", "M",
        "--seq_len", str(SEQ_LEN),
        "--label_len", "0",
        "--pred_len", str(pred_len),
        "--enc_in", str(enc_in), "--dec_in", str(enc_in), "--c_out", str(enc_in),
        "--period_len", str(period_len),
        "--model_type", "linear",
        "--learning_rate", str(LEARNING_RATE),
        "--lradj", LRADJ,
        "--train_epochs", str(TRAIN_EPOCHS),
        "--patience", str(PATIENCE),
        "--batch_size", str(BATCH_SIZE),
        "--num_workers", "0",
        "--des", DES,
        "--itr", "1",
    ]


def run_pair(src: str, tgt: str, pred_len: int) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = log_path_for(src, tgt, pred_len)
    period_len = DS[src][4]                      # source's period_len (keeps shapes compatible)
    model_id = f"ZS_{src}_to_{tgt}_{pred_len}"

    src_setting = build_setting(model_id, DS[src][0], pred_len)
    tgt_setting = build_setting(model_id, DS[tgt][0], pred_len)

    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as f:
        # 1) train on the SOURCE dataset
        f.write(f"===== TRAIN on {src} =====\n"); f.flush()
        subprocess.run(_base_args(src, model_id, pred_len, period_len) + ["--is_training", "1"],
                       cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)

        # 2) copy the source checkpoint into the target's setting dir so the test run loads it
        src_ckpt = CKPT_DIR / src_setting / "checkpoint.pth"
        tgt_ckpt = CKPT_DIR / tgt_setting / "checkpoint.pth"
        if not src_ckpt.exists():
            f.write(f"\n!! source checkpoint missing: {src_ckpt}\n")
            return
        tgt_ckpt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_ckpt, tgt_ckpt)
        f.write(f"\n===== TRANSFER {src_ckpt.name}: {src_setting} -> {tgt_setting} =====\n"); f.flush()

        # 3) evaluate on the TARGET dataset without any training
        f.write(f"===== TEST on {tgt} (zero-shot) =====\n"); f.flush()
        subprocess.run(_base_args(tgt, model_id, pred_len, period_len) + ["--is_training", "0"],
                       cwd=str(REPO_ROOT), stdout=f, stderr=subprocess.STDOUT, text=True)

    print(f"      {src}->{tgt} pred={pred_len} done in {time.perf_counter() - start:.1f}s", flush=True)


RESULTS_START = "<!-- AUTO_ZEROSHOT_RESULTS_START -->"
RESULTS_END = "<!-- AUTO_ZEROSHOT_RESULTS_END -->"


def _fmt(v: Optional[float], spec: str = ".6f") -> str:
    return "-" if v is None else format(v, spec)


def update_md() -> None:
    rows = ["| Transfer | pred_len | MSE | MAE | Status |", "|---|---:|---:|---:|---|"]
    per_pair: Dict[str, List[List[float]]] = {}
    for src, tgt in PAIRS:
        key = f"{src}->{tgt}"
        for pred_len in PRED_LENS:
            mse, mae, status = parse_log(log_path_for(src, tgt, pred_len))
            rows.append(f"| {key} | {pred_len} | {_fmt(mse)} | {_fmt(mae)} | {status} |")
            if status == "OK":
                per_pair.setdefault(key, []).append([mse, mae])

    summary = ["", "**Averaged over pred_len (vs. published reference)**", "",
               "| Transfer | SparseTSF MSE / MAE (ours) | TimeMixer (paper) | TimeMixer++ (paper) |",
               "|---|---:|---:|---:|"]
    for src, tgt in PAIRS:
        key = f"{src}->{tgt}"
        vals = per_pair.get(key, [])
        if vals:
            m = sum(v[0] for v in vals) / len(vals)
            a = sum(v[1] for v in vals) / len(vals)
            ours = f"{m:.3f} / {a:.3f}"
        else:
            ours = "-"
        ref = PAPER_REF[key]
        summary.append(f"| {key} | {ours} | _{ref['TimeMixer'][0]:.3f} / {ref['TimeMixer'][1]:.3f}_ "
                       f"| _{ref['TimeMixer++'][0]:.3f} / {ref['TimeMixer++'][1]:.3f}_ |")

    body = "\n".join(rows + summary) + "\n"
    text = MD_PATH.read_text(encoding="utf-8")
    si, ei = text.find(RESULTS_START), text.find(RESULTS_END)
    if si < 0 or ei < 0 or ei <= si:
        raise RuntimeError(f"Markers not found in {MD_PATH.name}")
    MD_PATH.write_text(text[: si + len(RESULTS_START)] + "\n" + body + text[ei:], encoding="utf-8")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    total = len(PAIRS) * len(PRED_LENS)
    done = 0
    for src, tgt in PAIRS:
        for pred_len in PRED_LENS:
            done += 1
            mse, _, status = parse_log(log_path_for(src, tgt, pred_len))
            if status == "OK":
                print(f"[SKIP {done}/{total}] {src}->{tgt} pred={pred_len} (done, MSE={mse:.6f})", flush=True)
                continue
            print(f"\n>>> [{done}/{total}] {src} -> {tgt} pred_len={pred_len} <<<", flush=True)
            run_pair(src, tgt, pred_len)
            update_md()
    update_md()
    print("\n[DONE] Zero-shot benchmark complete.", flush=True)


if __name__ == "__main__":
    main()
