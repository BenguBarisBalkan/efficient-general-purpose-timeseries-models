"""Count trainable parameters for all three models, at the configs the study uses.

Nothing in the pipeline logs a parameter count, so the figures quoted in the reports
were previously unverifiable. Run this to regenerate them:

    cd TimeMixer && venv/Scripts/python.exe benchmarks/tools/count_params.py

Parameter counts here are strongly config-dependent: TimeMixer's `pdm_blocks` scale with
`seq_len` and its `predict_layers` with `seq_len x pred_len`, while SparseTSF's single
linear layer is shaped `(seq_len/period_len -> pred_len/period_len)`. So a ratio between
models is only meaningful with the configuration stated alongside it.

Reuses run.py's argparse defaults, overriding only what each runner sets for ETTh.
"""
import copy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import run as run_module  # noqa: E402  (needs REPO_ROOT on sys.path first)
from models import SparseTSF, TimeMixer, TimeMixerPP  # noqa: E402

BASE = run_module.parser.parse_args(
    ["--task_name", "long_term_forecast", "--is_training", "0",
     "--model_id", "param_count", "--model", "TimeMixer", "--data", "ETTh1"]
)

# Shared across every config below; ETTh has 7 channels.
COMMON = dict(
    task_name="long_term_forecast", enc_in=7, dec_in=7, c_out=7,
    label_len=0, features="M", e_layers=2, factor=1, use_norm=1,
    down_sampling_layers=3, down_sampling_window=2, down_sampling_method="avg",
)

# Per-model hyperparameters, matching benchmarks/run_all_*.py for ETTh.
TIMEMIXER = dict(d_model=16, d_ff=32, channel_independence=1)
TIMEMIXERPP = dict(d_model=32, d_ff=64, n_heads=4, top_k=3)
SPARSETSF = dict(d_model=16, period_len=24, model_type="linear")


def cfg(**overrides):
    args = copy.deepcopy(BASE)
    for key, value in {**COMMON, **overrides}.items():
        setattr(args, key, value)
    return args


def count(module, args):
    model = module.Model(args)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def report(seq_len, pred_len):
    tm = count(TimeMixer, cfg(seq_len=seq_len, pred_len=pred_len, **TIMEMIXER))
    pp = count(TimeMixerPP, cfg(seq_len=seq_len, pred_len=pred_len, **TIMEMIXERPP))
    sp = count(SparseTSF, cfg(seq_len=seq_len, pred_len=pred_len, **SPARSETSF))
    print(f"ETTh, seq_len={seq_len}, pred_len={pred_len}")
    print(f"  TimeMixer     {tm:>12,}")
    print(f"  TimeMixer++   {pp:>12,}")
    print(f"  SparseTSF     {sp:>12,}")
    print(f"  TimeMixer / SparseTSF   = {tm / sp:>8,.0f}x")
    print(f"  TimeMixer++ / SparseTSF = {pp / sp:>8,.0f}x")
    print()
    return tm, pp, sp


if __name__ == "__main__":
    # The accuracy protocol's matched setting, and SparseTSF's native lookback.
    report(96, 96)
    report(720, 720)

    # Where TimeMixer's parameters actually sit, at the larger config.
    args = cfg(seq_len=720, pred_len=720, **TIMEMIXER)
    model = TimeMixer.Model(args)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("TimeMixer breakdown at seq_len=720, pred_len=720")
    children = [
        (sum(p.numel() for p in mod.parameters() if p.requires_grad), name)
        for name, mod in model.named_children()
    ]
    for n, name in sorted(children, reverse=True):
        if n:
            print(f"  {name:<22} {n:>12,}  ({100 * n / total:4.1f}%)")
