## TimeMixer Reproduction Results vs Paper

- **Paper**: TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting (ICLR 2024).
- **Codebase**: `TimeMixer` (this repo).
- **Hardware**: Local Windows machine, RTX 3050 Ti Laptop GPU (4 GB VRAM).
- **Setup**: unified hyperparameters from Table 7/13, `seq_len=96`, `num_workers=0`, single run per config.

Updated automatically by `run_missing.py` after each completed run.

<!-- AUTO_LONG_TERM_RESULTS_START -->
| Dataset | pred_len | Paper MSE | Paper MAE | Local MSE | Local MAE | Runtime (s) | Runtime (h) | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ETTh1 | 96 | 0.375 | 0.400 | 0.385794 | 0.402099 | 47.6 | 0.01 | OK |
| ETTh1 | 192 | 0.429 | 0.421 | 0.443037 | 0.429825 | 49.3 | 0.01 | OK |
| ETTh1 | 336 | 0.484 | 0.458 | 0.513090 | 0.470159 | 52.8 | 0.01 | OK |
| ETTh1 | 720 | 0.498 | 0.482 | 0.489165 | 0.472423 | 67.6 | 0.02 | OK |
| ETTh2 | 96 | 0.289 | 0.341 | 0.301334 | 0.351145 | 52.8 | 0.01 | OK |
| ETTh2 | 192 | 0.372 | 0.392 | 0.391096 | 0.408280 | 50.9 | 0.01 | OK |
| ETTh2 | 336 | 0.386 | 0.414 | 0.386519 | 0.414844 | 57.2 | 0.02 | OK |
| ETTh2 | 720 | 0.412 | 0.434 | 0.412863 | 0.434229 | 67.9 | 0.02 | OK |
| ETTm1 | 96 | 0.320 | 0.357 | 0.326398 | 0.362354 | 210.7 | 0.06 | OK |
| ETTm1 | 192 | 0.361 | 0.381 | 0.365291 | 0.385513 | 217.0 | 0.06 | OK |
| ETTm1 | 336 | 0.390 | 0.404 | 0.393321 | 0.406302 | 240.4 | 0.07 | OK |
| ETTm1 | 720 | 0.454 | 0.441 | 0.451405 | 0.442712 | 304.0 | 0.08 | OK |
| ETTm2 | 96 | 0.175 | 0.258 | 0.175972 | 0.257030 | 348.6 | 0.10 | OK |
| ETTm2 | 192 | 0.237 | 0.299 | 0.238108 | 0.299325 | 397.0 | 0.11 | OK |
| ETTm2 | 336 | 0.298 | 0.340 | 0.298672 | 0.339644 | 428.7 | 0.12 | OK |
| ETTm2 | 720 | 0.391 | 0.396 | 0.406894 | 0.407958 | 552.4 | 0.15 | OK |
| Weather | 96 | 0.163 | 0.209 | 0.164526 | 0.211217 | 1272.4 | 0.35 | OK |
| Weather | 192 | 0.208 | 0.250 | 0.209339 | 0.252745 | 988.9 | 0.27 | OK |
| Weather | 336 | 0.251 | 0.287 | 0.265349 | 0.293051 | 1092.0 | 0.30 | OK |
| Weather | 720 | 0.339 | 0.341 | 0.342314 | 0.345848 | 1576.0 | 0.44 | OK |
| Solar-Energy | 96 | 0.189 | 0.259 | 0.208248 | 0.275516 | 1361.5 | 0.38 | OK |
| Solar-Energy | 192 | 0.222 | 0.283 | 0.238817 | 0.307026 | 1446.6 | 0.40 | OK |
| Solar-Energy | 336 | 0.231 | 0.292 | 0.232882 | 0.300262 | 1548.9 | 0.43 | OK |
| Solar-Energy | 720 | 0.223 | 0.285 | 0.247561 | 0.307404 | 2068.8 | 0.57 | OK |
| Electricity | 96 | 0.153 | 0.247 | 0.154200 | 0.245481 | 596.8 | 0.17 | OK |
| Electricity | 192 | 0.166 | 0.256 | 0.167991 | 0.257820 | 5489.1 | 1.52 | OK |
| Electricity | 336 | 0.185 | 0.277 | 0.185280 | 0.275914 | 16686.9 | 4.64 | OK |
| Electricity | 720 | 0.225 | 0.310 | 0.224360 | 0.309029 | 99590.2 | 27.66 | OK |
| Traffic | 96 | 0.462 | 0.285 | 0.481342 | 0.307599 | 58317.7 | 16.20 | OK |
| Traffic | 192 | 0.473 | 0.296 | 0.485919 | 0.304961 | 90104.7 | 25.03 | OK |
| Traffic | 336 | 0.498 | 0.296 | - | - | - | - | RUNNING/INCOMPLETE |
| Traffic | 720 | 0.506 | 0.313 | - | - | - | - | RUNNING/INCOMPLETE |
<!-- AUTO_LONG_TERM_RESULTS_END -->

