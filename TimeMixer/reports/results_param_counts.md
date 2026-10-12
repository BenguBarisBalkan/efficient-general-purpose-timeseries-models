# Trainable parameters per task

Every count is rebuilt from the argument list in the run's own log (`benchmarks/tools/count_params_all.py`), so it matches the configuration that produced the result. A range covers the horizons (forecasting) or mask rates (imputation) of that dataset. Few-shot and zero-shot reuse the long-term architectures unchanged. SparseTSFPlus counts are in its own reports. TimeMixer on anomaly detection and classification has no published configuration; it uses its ETTh1 long-term architecture (classification additionally needs channel mixing, which TimeMixer's classification code requires).

```bash
venv/Scripts/python.exe benchmarks/tools/count_params_all.py
```

## Long-term forecasting (seq_len 96)

| Dataset | SparseTSF | TimeMixer | TimeMixer++ |
|---|---:|---:|---:|
| Electricity | 41–145 | 95,673–192,825 | — |
| ETTh1 | 41–145 | 75,497–190,313 | 112,951–227,767 |
| ETTh2 | 41–145 | 75,497–190,313 | 112,951–227,767 |
| ETTm1 | 581–4,325 | 75,497–190,313 | 112,951–227,767 |
| ETTm2 | 581–4,325 | 77,769–192,585 | 112,951–227,767 |
| Solar-Energy | 581–4,325 | 2,281,445–2,511,077 | 169,761–187,425 |
| Traffic | 41–145 | 88,769–132,929 | — |
| Weather | 581–4,325 | 75,609–190,425 | 119,069–233,885 |

## Short-term forecasting, M4

| Dataset | SparseTSF | TimeMixer |
|---|---:|---:|
| Daily | 15 | 16,973 |
| Hourly | 33 | 100,069 |
| Monthly | 25 | 22,069 |
| Quarterly | 13 | 11,669 |
| Weekly | 15 | 15,894 |
| Yearly | 21 | 10,525 |

## Short-term forecasting, PEMS

| Dataset | SparseTSF | TimeMixer |
|---|---:|---:|
| PEMS03 | 21 | 966,366 |
| PEMS04 | 21 | 939,999 |
| PEMS07 | 21 | 1,237,791 |
| PEMS08 | 21 | 869,170 |

## Imputation

| Dataset | SparseTSF | TimeMixer |
|---|---:|---:|
| Electricity | 41 | 60,345 |
| ETTh1 | 41 | 57,833 |
| ETTh2 | 41 | 57,833 |
| ETTm1 | 581 | 57,833 |
| ETTm2 | 581 | 60,105 |
| Weather | 581 | 57,945 |

## Anomaly detection

| Dataset | SparseTSF | TimeMixer |
|---|---:|---:|
| PSM | 41 | 62,573 |
| SMD | 41 | 62,677 |

## Classification

| Dataset | SparseTSF | TimeMixer |
|---|---:|---:|
| EthanolConcentration | 21,017 | 18,227,514 |
| FaceDetection | 17,859 | 37,608 |
| Handwriting | 11,883 | 205,896 |
| Heartbeat | 49,413 | 991,156 |
| JapaneseVowels | 3,142 | 14,349 |
| PEMS-SF | 970,712 | 198,463 |
| SelfRegulationSCP1 | 10,755 | 4,784,466 |
| SelfRegulationSCP2 | 16,131 | 7,891,978 |
| SpokenArabicDigits | 12,101 | 71,478 |
| UWaveGestureLibrary | 7,569 | 631,604 |

Logs that record no argument list, so they are not in the ranges above: `run_logs/Electricity_pred96.log`, `run_logs/Traffic_pred720.log`.
