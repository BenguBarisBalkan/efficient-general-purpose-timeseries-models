# SparseTSFPlus ablation — anomaly detection

Tests whether leave-one-out reconstruction fixes SparseTSF's recall-limited profile. `base`
is all units off and must reproduce the published SparseTSF row. Point-adjusted F1, as in the
TimeMixer++ paper. The scoring and threshold are unchanged and shared with every model.

```bash
venv/Scripts/python.exe benchmarks/run_sparsetsf_plus_anomaly.py
```

<!-- AUTO_PLUS_ANOM_START -->
| Dataset | Variant | Params | Precision | Recall | F1 | F1 vs SparseTSF (published) | Status |
|---|---|---:|---:|---:|---:|---:|---|
| PSM | base | 41 | 98.82 | 91.96 | 95.26 | +0.00 | OK |
| PSM | loo | 41 | 98.84 | 88.22 | 93.23 | -2.03 | OK |
| _PSM published_ | _SparseTSF_ | _—_ | _98.82_ | _91.96_ | _95.26_ | _—_ | _reference_ |
| _PSM published_ | _TimeMixer++_ | _—_ | _—_ | _—_ | _97.60_ | _—_ | _reference_ |
| SMD | base | 41 | 87.25 | 73.90 | 80.02 | +0.00 | OK |
| SMD | loo | 41 | 86.46 | 75.39 | 80.55 | +0.53 | OK |
| _SMD published_ | _SparseTSF_ | _—_ | _87.25_ | _73.90_ | _80.02_ | _—_ | _reference_ |
| _SMD published_ | _TimeMixer++_ | _—_ | _—_ | _—_ | _86.50_ | _—_ | _reference_ |
<!-- AUTO_PLUS_ANOM_END -->
