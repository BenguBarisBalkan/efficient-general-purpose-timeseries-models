# SparseTSFPlus ablation — classification (10 UEA datasets)

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

<!-- AUTO_PLUS_CLS_START -->
| Dataset | flat | flatfix | stats | stats_p25 | segpool | segpool_p25 | SparseTSF published | best vs published |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EthanolConcentration | 29.28 | 29.28 | 24.71 | 27.38 | - | - | _29.28_ | -0.00 |
| FaceDetection | 68.59 | 68.59 | 51.79 | 50.20 | - | - | _68.59_ | -0.00 |
| Handwriting | 8.59 | 8.59 | 3.41 | 3.88 | 4.71 | 4.94 | _8.59_ | -0.00 |
| Heartbeat | 54.63 | 54.63 | 71.71 | 63.90 | 71.71 | 70.24 | _54.63_ | +17.08 |
| JapaneseVowels | 61.62 | 45.41 | 15.95 | 7.57 | 12.16 | 18.92 | _61.62_ | +0.00 |
| PEMS-SF | 84.39 | 84.39 | 45.66 | 64.16 | - | - | _84.39_ | +0.00 |
| SelfRegulationSCP1 | 82.25 | 82.25 | 50.85 | 50.85 | - | - | _82.25_ | +0.00 |
| SelfRegulationSCP2 | 52.78 | 52.78 | 50.56 | 50.00 | - | - | _52.78_ | -0.00 |
| SpokenArabicDigits | 94.04 | 93.00 | 42.88 | 62.57 | - | - | _94.04_ | +0.00 |
| UWaveGestureLibrary | 72.81 | 72.81 | 9.38 | 12.50 | 12.19 | 10.62 | _72.81_ | +0.00 |
| **Average** | **60.90** | **59.17** | **36.69** | **39.30** | 25.19* | 26.18* | _60.90_ | |

`*` = partial average over completed datasets only.

### Head size (the one enc_in-coupled head in the model)

| Dataset | flat | flatfix | stats | stats_p25 | segpool | segpool_p25 |
|---|---:|---:|---:|---:|---:|---:|
| EthanolConcentration | 21,017 | 21,017 | 41 | 65 | - | - |
| FaceDetection | 17,859 | 17,859 | 867 | 891 | - | - |
| Handwriting | 11,883 | 11,883 | 261 | 285 | 651 | 675 |
| Heartbeat | 49,413 | 49,413 | 369 | 393 | 979 | 1,003 |
| JapaneseVowels | 3,142 | 3,142 | 334 | 358 | 874 | 898 |
| PEMS-SF | 970,712 | 970,712 | 20,231 | 20,255 | - | - |
| SelfRegulationSCP1 | 10,755 | 10,755 | 39 | 63 | - | - |
| SelfRegulationSCP2 | 16,131 | 16,131 | 45 | 69 | - | - |
| SpokenArabicDigits | 12,101 | 12,101 | 401 | 425 | - | - |
| UWaveGestureLibrary | 7,569 | 7,569 | 81 | 105 | 201 | 225 |

### Published references

| Model | Avg accuracy (%) |
|---|---:|
| _TimeMixer++ (paper)_ | _75.9_ |
| _TimesNet (paper)_ | _73.6_ |

Variant legend:

- **flat** — upstream head, padded-mean bug included (control) (`--stsf_cls_head flat --period_len 1`)
- **flatfix** — mask-aware mean only; head unchanged (`--stsf_cls_head flat_fixed --period_len 1`)
- **stats** — mean/std/max pooling head (`--stsf_cls_head stats --period_len 1`)
- **stats_p25** — pooling head + a real 25-tap conv (`--stsf_cls_head stats --period_len 25`)
- **segpool** — segment pooling (keeps coarse temporal order) (`--stsf_cls_head segpool --period_len 1`)
- **segpool_p25** — segment pooling + a real 25-tap conv (`--stsf_cls_head segpool --period_len 25`)
<!-- AUTO_PLUS_CLS_END -->
