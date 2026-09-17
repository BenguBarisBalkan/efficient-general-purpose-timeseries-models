# Final Report — Accuracy & Carbon: TimeMixer vs TimeMixer++ vs SparseTSF

Consolidated results for three long-term forecasting models benchmarked locally on an
**RTX 3050 Ti Laptop GPU (4 GB), Windows**. This report ties together the four detailed tables:
`results_comparison.md` (TimeMixer), `results_comparison_pp.md` (TimeMixer++),
`results_comparison_sparsetsf.md` (SparseTSF), and `results_carbon_comparison.md` (energy/CO2).

| Model | Design | Params @ sl=96 / pred=96 | Params @ sl=720 / pred=720 |
|---|---|---:|---:|
| **TimeMixer** (ICLR 2024) | MLP-based multiscale mixing (PDM + FMM) | 75,497 | 4,046,633 |
| **TimeMixer++** (ICLR 2025) | FFT time-imaging + dual-axis attention + conv mixing | 112,951 | 1,080,385 |
| **SparseTSF** (ICML 2024) | Cross-period sparse forecasting, one shared linear layer | **41** | **925** |

Parameter counts are **measured**, reproducible via `benchmarks/tools/count_params.py` (ETTh, 7
channels, each model's own hyperparameters). Two things they show that a single number hides:

- **They are strongly config-dependent.** TimeMixer's `pdm_blocks` scale with `seq_len` (75.9% of
  its parameters at sl=720) and its `predict_layers` with `seq_len × pred_len` (24.1%); SparseTSF's
  single linear layer is shaped `(seq_len/period_len → pred_len/period_len)`. Quoting a ratio
  without the configuration is meaningless.
- **TimeMixer++ is not uniformly smaller.** It is *larger* than TimeMixer at sl=96 (112,951 vs
  75,497) and smaller at sl=720 (1.08M vs 4.05M), because it does not carry
  sequence-length-scaled mixing MLPs.

*Correction: earlier revisions of this report quoted "~100k+" for TimeMixer and "~40k" for
TimeMixer++. Those were unverified estimates — nothing in the pipeline logs a parameter count — and
were low by roughly 40× and 27× respectively at sl=720. SparseTSF's 925 was correct.*

---

## Executive Summary

1. **Accuracy (matched seq_len=96 task):** TimeMixer and SparseTSF are **essentially tied**
   (mean MSE 0.348 vs 0.354 across 20 configs), with SparseTSF winning all of ETTh2 and most of
   ETTh1 while TimeMixer wins the ETTm/Weather datasets. TimeMixer++ trails (0.449) — but see the
   caveat: it is deliberately undertrained at `lr=5e-5` for numerical stability.

2. **Accuracy (SparseTSF's native seq_len=720):** SparseTSF's mean MSE drops to **0.312, beating
   TimeMixer's 0.348** — a **925-parameter** model outperforming a 75,497-parameter one when given
   its proper lookback.

3. **Energy:** For identical training work (matched 5-epoch protocol), **TimeMixer++ costs ~1.67×
   the energy of TimeMixer** (2.07–2.10× on ETT, 1.19× on Weather), while **SparseTSF costs just
   0.09× — roughly 11× less than TimeMixer and ~19× less than TimeMixer++.**

4. **The intersection:** SparseTSF matches TimeMixer's accuracy at **~1/11th the energy**, and
   *exceeds* it at its native lookback while still using a tiny fraction of the power. TimeMixer++'s
   heavier architecture did **not** buy better accuracy in our runs — it cost ~2× the energy for
   worse (undertrained) results.

---

## 1. Accuracy — Three-Way at seq_len = 96

All three models use `seq_len=96` with each model's own published hyperparameters (best-effort per
model). MSE; lower is better. **Bold** = best of the three.

| Dataset | pred_len | TimeMixer | TimeMixer++ | SparseTSF |
|---|---:|---:|---:|---:|
| ETTh1 | 96 | **0.386** | 0.693 | 0.397 |
| ETTh1 | 192 | 0.443 | 0.736 | **0.433** |
| ETTh1 | 336 | 0.513 | 0.795 | **0.450** |
| ETTh1 | 720 | 0.489 | 0.763 | **0.459** |
| ETTh2 | 96 | 0.301 | 0.383 | **0.289** |
| ETTh2 | 192 | 0.391 | 0.459 | **0.363** |
| ETTh2 | 336 | 0.387 | 0.487 | **0.366** |
| ETTh2 | 720 | 0.413 | 0.474 | **0.409** |
| ETTm1 | 96 | **0.326** | 0.420 | 0.362 |
| ETTm1 | 192 | **0.365** | 0.426 | 0.394 |
| ETTm1 | 336 | **0.393** | 0.471 | 0.427 |
| ETTm1 | 720 | **0.451** | 0.514 | 0.486 |
| ETTm2 | 96 | **0.176** | 0.199 | 0.185 |
| ETTm2 | 192 | **0.238** | 0.263 | 0.249 |
| ETTm2 | 336 | **0.299** | 0.344 | 0.312 |
| ETTm2 | 720 | **0.407** | 0.444 | 0.408 |
| Weather | 96 | **0.165** | 0.200 | 0.199 |
| Weather | 192 | **0.209** | 0.244 | 0.243 |
| Weather | 336 | **0.265** | 0.295 | 0.292 |
| Weather | 720 | **0.342** | 0.364 | 0.363 |
| **Mean MSE** | | **0.348** | 0.449 | **0.354** |
| **Wins (of 20)** | | **13** | 0 | **7** |

**Read:** SparseTSF sweeps the **hourly ETTh** datasets (where its `period_len=24` locks onto the
daily cycle); TimeMixer takes the **15-min ETTm and 10-min Weather** datasets (where SparseTSF's
`period_len=4` gives it less structure to exploit). TimeMixer++ wins nothing — undertrained (below).

---

## 2. Accuracy — SparseTSF at its native seq_len = 720

SparseTSF is designed for a long lookback. At `seq_len=720` it improves sharply and **beats
TimeMixer's seq_len=96 mean** (0.312 vs 0.348) — with 925 parameters and ~15-90 s per run.

| Dataset | pred=96 | pred=192 | pred=336 | pred=720 | Mean |
|---|---:|---:|---:|---:|---:|
| ETTh1 | 0.354 | 0.395 | 0.402 | 0.420 | 0.393 |
| ETTh2 | 0.263 | 0.319 | 0.311 | 0.368 | 0.315 |
| ETTm1 | 0.314 | 0.343 | 0.372 | 0.417 | 0.361 |
| ETTm2 | 0.164 | 0.220 | 0.272 | 0.351 | 0.252 |
| Weather | 0.173 | 0.214 | 0.260 | 0.318 | 0.241 |
| **All** | | | | | **0.312** |

(Not directly comparable to the seq_len=96 table — a longer input is a different task — but it is
SparseTSF's honest published capability, so it is reported here.)

---

## 3. Energy & Carbon

Measured with **CodeCarbon 3.2.3** under a *matched* protocol (all three models: `seq_len=96`,
`d_model=32`, `batch=64`, **5 fixed epochs, early stopping disabled**, `lr=5e-5`), summed over the
4 prediction lengths per dataset. This isolates **energy per unit of training work** by architecture.

| Dataset | TimeMixer kWh | TimeMixer++ kWh | SparseTSF kWh | PP / TM | Sparse / TM |
|---|---:|---:|---:|---:|---:|
| ETTh1 | 5.67e-03 | 1.18e-02 | 1.06e-03 | 2.07× | **0.19×** |
| ETTh2 | 4.87e-03 | 1.02e-02 | 5.37e-04 | 2.10× | **0.11×** |
| ETTm1 | 1.94e-02 | 4.03e-02 | 1.91e-03 | 2.08× | **0.10×** |
| ETTm2 | 1.85e-02 | 3.83e-02 | 1.53e-03 | 2.07× | **0.08×** |
| Weather | 4.10e-02 | 4.89e-02 | 2.74e-03 | 1.19× | **0.07×** |
| **TOTAL** | **8.93e-02** | **1.49e-01** | **7.77e-03** | **1.67×** | **0.09×** |
| **CO2 (kg)** | 4.15e-02 | 6.94e-02 | 3.61e-03 | 1.67× | 0.09× |

**Read:**
- **TimeMixer++ ≈ 2× the energy of TimeMixer on ETT**, driven by ~3× longer wall-clock per run
  (its FFT/attention/conv stack). The gap narrows to 1.19× on **Weather** — because Weather has 21
  channels and TimeMixer is *channel-independent*, so its own cost rises on wide data, closing the gap.
- **SparseTSF ≈ 1/11th of TimeMixer and ~1/19th of TimeMixer++** — a single shared linear layer over
  925 parameters barely touches the GPU.

---

## 4. The Point: Accuracy vs Energy Together

| Model | Accuracy (sl=96 mean MSE) | Energy (total kWh) | Energy vs TimeMixer |
|---|---:|---:|---:|
| **SparseTSF** | 0.354 (≈ tied) | **7.77e-03** | **0.09× (11× less)** |
| **TimeMixer** | **0.348 (best)** | 8.93e-02 | 1.00× |
| **TimeMixer++** | 0.449 (undertrained) | 1.49e-01 | 1.67× (most) |

- **SparseTSF is the efficiency winner by a wide margin**: it ties TimeMixer's accuracy while using
  ~9% of its energy, and at its native lookback it is the *most accurate* model overall — with
  **~1,800× fewer parameters** at the sl=96 comparison (41 vs 75,497) and **~4,400× fewer** at
  sl=720/pred=720 (925 vs 4.05M).
- **TimeMixer** is the accuracy leader at the common protocol and a reasonable energy cost.
- **TimeMixer++'s** heavier architecture bought neither accuracy (in our runs) nor efficiency here —
  though its accuracy figure is depressed by the stability-driven low LR (see caveats).

**Bottom line:** on these datasets, added architectural complexity (TimeMixer++) did not pay off in
either accuracy or energy, whereas radical sparsity (SparseTSF) delivered competitive accuracy at a
fraction of the carbon cost.

---

## Methodology & Caveats

- **Two different protocols, do not cross-compare.** *Accuracy* tables use each model's own
  published hyperparameters with full early stopping (best-effort accuracy). *Carbon* tables use a
  matched 5-epoch, no-early-stopping protocol (energy per unit work). The **MSE shown in the carbon
  table is NOT real accuracy** — e.g. SparseTSF's carbon-protocol MSE ~1.0 vs its accuracy-protocol
  0.397 — because the matched config cripples every model equally. Only the accuracy tables report
  meaningful accuracy.
- **TimeMixer++ is undertrained.** It requires `lr=5e-5` to avoid NaN divergence (see the stability
  changelog in `results_comparison_pp.md`); at that rate it does not converge within 20 epochs. Its
  accuracy column therefore *understates* the architecture's real capability — this is not a clean
  "TimeMixer++ is less accurate" result. Its **energy** result, by contrast, is a fair measurement.
- **Windows energy limits.** CodeCarbon reads *real* GPU power via NVML, but Windows has no RAPL, so
  **CPU and RAM energy are TDP-based estimates**. GPU dominates and all models are measured
  identically on one machine, so the *relative* ratios are sound; treat absolute kg CO2 as an
  estimate. Regional carbon intensity (Türkiye) is from CodeCarbon's dataset, not live grid data.
- **SparseTSF@720 was not carbon-profiled** (the carbon protocol fixes seq_len=96 for cross-model
  matching). Its accuracy-run wall-clock (~15-90 s, up to 320 s for Weather-720) shows it remains
  very cheap, but its 720-lookback energy is not directly in the energy table.
- **Scope**: 5 light datasets (ETTh1/2, ETTm1/2, Weather). Solar-Energy, Electricity, Traffic were
  excluded as too heavy for the 4 GB GPU.
- **SparseTSF** is vendored under Apache-2.0 (`implementations/sparsetsf/`) with a 1-arg→4-arg adapter
  (`models/SparseTSF.py`); `model_type=linear` (the <1k-param variant) is used throughout.
