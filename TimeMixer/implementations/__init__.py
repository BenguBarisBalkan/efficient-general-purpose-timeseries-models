"""Model implementations, one package per model.

`models/<Name>.py` stays the registry entry point that `exp/exp_basic.py` imports;
the substantial code lives here:

  timemixer_pp/  TimeMixer++ (ICLR'25), reimplemented from scratch for this study
  sparsetsf/     SparseTSF (ICML'24), vendored unmodified under Apache-2.0
                 (see sparsetsf/LICENSE-SparseTSF)

TimeMixer v1 is deliberately NOT here: it remains the upstream file
`models/TimeMixer.py`, so upstream changes still merge cleanly.
"""
