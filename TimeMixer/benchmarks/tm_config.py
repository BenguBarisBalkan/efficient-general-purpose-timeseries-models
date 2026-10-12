"""TimeMixer's per-dataset settings, shared by every per-task runner that trains TimeMixer.

One source of truth: the values are read from run_all_long_term_benchmarks.py, the runner that
produced the local TimeMixer long-term table (itself copied from upstream
scripts/long_term_forecast). Zero-shot, few-shot and the reconstruction/classification tasks reuse
TimeMixer's own long-term architecture, the same rule the local TimeMixer imputation sweep used.
TimeMixer publishes no settings for anomaly detection or classification (neither its own repository
nor the Time-Series-Library ships them), so those results are labelled "local, no published config".

Training settings that are not passed explicitly fall back to run.py's defaults, exactly as in the
long-term runner (notably lradj='TST').
"""

from typing import Dict, List, Optional

from run_all_long_term_benchmarks import DATASETS as _LONG_TERM

LONG_TERM = {d.name: d for d in _LONG_TERM}


def arch(name: str, down_sampling_layers: Optional[int] = None,
         channel_independence: Optional[int] = None) -> List[str]:
    """TimeMixer's architecture flags for dataset `name`. The overrides exist for tasks whose code
    path cannot use the long-term value: `down_sampling_layers` where the input cannot be halved
    that many times, `channel_independence` where the task embeds all channels jointly
    (TimeMixer.classification has no per-channel path, so it needs 0)."""
    d = LONG_TERM[name]
    ci = d.channel_independence if channel_independence is None else channel_independence
    out = [
        "--e_layers", str(d.e_layers),
        "--d_model", str(d.d_model),
        "--d_ff", str(d.d_ff),
        "--factor", str(d.factor),
        "--channel_independence", str(ci),
        "--down_sampling_layers", str(d.down_sampling_layers if down_sampling_layers is None
                                      else down_sampling_layers),
        "--down_sampling_window", str(d.down_sampling_window),
        "--down_sampling_method", "avg",
    ]
    if d.use_norm is not None:
        out += ["--use_norm", str(d.use_norm)]
    return out


def training(name: str) -> Dict[str, str]:
    """TimeMixer's long-term training settings for dataset `name`."""
    d = LONG_TERM[name]
    return {"learning_rate": str(d.learning_rate), "batch_size": str(d.batch_size),
            "train_epochs": str(d.train_epochs), "patience": str(d.patience)}


def setting_dims(name: str) -> str:
    """The `dm.._nh.._el.._dl.._df.._fc..` part of run.py's setting string."""
    d = LONG_TERM[name]
    return f"dm{d.d_model}_nh8_el{d.e_layers}_dl1_df{d.d_ff}_fc{d.factor}"
