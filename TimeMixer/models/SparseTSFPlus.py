"""Registry entry point for SparseTSFPlus.

Thin by design: `models/` is the registry that `exp/exp_basic.py` imports, and the
implementation lives in `implementations/sparsetsf_plus/`. Same pattern as
`models/TimeMixerPP.py`.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from implementations.sparsetsf_plus.model import Model  # noqa: F401,E402
