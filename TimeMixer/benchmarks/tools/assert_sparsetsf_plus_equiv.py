"""Assert SparseTSFPlus with all units off is numerically identical to SparseTSF.

This is the gate the whole ablation rests on. If it ever fails, the new model is no longer
a strict generalisation of SparseTSF, the baseline row of the ablation table is a lie, and
every published SparseTSF number in reports/ becomes unsafe to compare against.

Two things are checked for each of the five tasks:

  delegated    the default path, where vanilla forecasting delegates to the untouched
               vendored core (reproduction by construction)
  reimpl       the re-implemented backbone, forced via force_reimpl=True, which is the
               path that actually runs once any unit is switched on

Weights are copied across in parameter order (shapes must line up exactly, which is itself
a structural check), then both models are evaluated on the same random input.

    cd TimeMixer && venv/Scripts/python.exe benchmarks/tools/assert_sparsetsf_plus_equiv.py
"""

import copy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

import run as run_module  # noqa: E402
from implementations.sparsetsf_plus.model import Model as PlusModel  # noqa: E402
from models import SparseTSF  # noqa: E402

BASE = run_module.parser.parse_args(
    ["--task_name", "long_term_forecast", "--is_training", "0",
     "--model_id", "equiv_check", "--model", "SparseTSFPlus", "--data", "ETTh1"]
)


def cfg(**overrides):
    args = copy.deepcopy(BASE)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


# (label, config, input shape builder, extra forward kwargs)
CASES = [
    ("long_term_forecast",
     dict(task_name="long_term_forecast", seq_len=96, pred_len=96, period_len=24,
          enc_in=7, dec_in=7, c_out=7, label_len=0)),
    ("short_term_forecast",
     dict(task_name="short_term_forecast", seq_len=96, pred_len=48, period_len=24,
          enc_in=7, dec_in=7, c_out=7, label_len=0)),
    ("imputation",
     dict(task_name="imputation", seq_len=96, pred_len=0, period_len=4,
          enc_in=7, dec_in=7, c_out=7, label_len=0)),
    ("anomaly_detection",
     dict(task_name="anomaly_detection", seq_len=100, pred_len=0, period_len=25,
          enc_in=38, dec_in=38, c_out=38, label_len=0)),
    ("classification",
     dict(task_name="classification", seq_len=152, pred_len=0, period_len=1,
          enc_in=3, dec_in=3, c_out=3, label_len=0, num_class=26)),
]

BATCH = 4


def copy_weights(src, dst):
    """Copy parameters positionally; shapes must match exactly."""
    sp = [p for p in src.parameters() if p.requires_grad]
    dp = [p for p in dst.parameters() if p.requires_grad]
    if len(sp) != len(dp):
        raise AssertionError(
            "parameter count differs: SparseTSF has {} tensors, SparseTSFPlus has {}".format(
                len(sp), len(dp)))
    with torch.no_grad():
        for a, b in zip(sp, dp):
            if a.shape != b.shape:
                raise AssertionError(
                    "shape mismatch: {} vs {}".format(tuple(a.shape), tuple(b.shape)))
            b.copy_(a)


def make_inputs(conf):
    torch.manual_seed(1234)
    x = torch.randn(BATCH, conf["seq_len"], conf["enc_in"])
    kwargs = {}
    if conf["task_name"] == "imputation":
        mask = (torch.rand(BATCH, conf["seq_len"], conf["enc_in"]) > 0.5).float()
        kwargs["mask"] = mask
        x = x.masked_fill(mask == 0, 0)
    if conf["task_name"] == "classification":
        # exp_classification passes the padding mask through x_mark_enc
        kwargs["x_mark_enc"] = torch.ones(BATCH, conf["seq_len"])
    return x, kwargs


def run_case(label, conf, force_reimpl):
    args = cfg(**conf)

    torch.manual_seed(0)
    ref = SparseTSF.Model(args).float().eval()
    torch.manual_seed(0)
    new = PlusModel(args, force_reimpl=force_reimpl).float().eval()

    copy_weights(ref, new)

    n_ref = sum(p.numel() for p in ref.parameters() if p.requires_grad)
    n_new = sum(p.numel() for p in new.parameters() if p.requires_grad)
    if n_ref != n_new:
        raise AssertionError("param count {} vs {}".format(n_ref, n_new))

    x, kwargs = make_inputs(conf)
    with torch.no_grad():
        a = ref(x, kwargs.get("x_mark_enc"), None, None, kwargs.get("mask"))
        b = new(x, kwargs.get("x_mark_enc"), None, None, kwargs.get("mask"))

    if a.shape != b.shape:
        raise AssertionError("output shape {} vs {}".format(tuple(a.shape), tuple(b.shape)))
    diff = (a - b).abs().max().item()
    mode = "reimpl" if force_reimpl else "delegated"
    status = "EXACT" if diff == 0.0 else ("close({:.2e})".format(diff) if diff < 1e-6 else "FAIL")
    print("  {:<20} {:<10} params={:<8,} max|diff|={:.3e}  {}".format(
        label, mode, n_new, diff, status))
    return diff


def main():
    print("SparseTSFPlus (all units off)  vs  SparseTSF")
    print("=" * 74)
    failures = []
    for label, conf in CASES:
        for force in (False, True):
            forecast = conf["task_name"].endswith("forecast")
            if not forecast and force:
                continue  # non-forecast tasks have only the one implementation
            try:
                diff = run_case(label, conf, force)
                if diff > 1e-6:
                    failures.append((label, force, "diff={:.3e}".format(diff)))
            except AssertionError as exc:
                print("  {:<20} {:<10} ASSERTION: {}".format(
                    label, "reimpl" if force else "delegated", exc))
                failures.append((label, force, str(exc)))
    print("=" * 74)
    if failures:
        print("FAILED {} case(s):".format(len(failures)))
        for label, force, why in failures:
            print("  - {} ({}): {}".format(label, "reimpl" if force else "delegated", why))
        return 1
    print("ALL EQUIVALENT -- all-units-off reproduces SparseTSF on every task.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
