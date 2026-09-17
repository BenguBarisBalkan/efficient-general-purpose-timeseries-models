# Session Summary — Three-Model Forecasting Benchmark (Accuracy + Carbon)

_Local reproduction & comparison of three time-series forecasting models on an **RTX 3050 Ti Laptop
GPU (4 GB), Windows**, on both **accuracy** and **energy/carbon** dimensions._

All working code and detailed tables live in the `TimeMixer/` subfolder. This file is a high-level
recap of what was accomplished across the project.

---

## What we set out to do

Benchmark and compare three long-term forecasting models — one already public, one implemented
from scratch, one vendored — on the same local hardware, measuring **both** predictive accuracy and
**energy/CO₂ cost**, then write a consolidated report.

| Model | Design | Params (ETTh, sl=720) |
|---|---|---:|
| **TimeMixer** (ICLR 2024) | MLP-based multiscale mixing (PDM season/trend + FMM), channel-independent | ~100k+ |
| **TimeMixer++** (ICLR 2025) | FFT time-imaging (MRTI) + dual-axis attention (TID) + conv multi-scale mixing | ~40k (our impl) |
| **SparseTSF** (ICML 2024 Oral) | Cross-Period Sparse Forecasting — one shared linear layer | **925** |

---

## What we did, in order

1. **Completed the TimeMixer v1 accuracy table** (`results_comparison.md`) — ran only the missing
   experiments, updated the file as each run finished, and cleaned up redundant tables at the top.

2. **Implemented TimeMixer++ from scratch** in `TimeMixer/TimeMixer_plus/model.py` (there is no
   public implementation). Built the full architecture: learnable Conv1d downsampling, ChannelMixer
   (variate self-attention at the coarsest scale), MRTI (FFT → top-K periods → 2-D "time images"),
   dual-axis attention TID (season via column-attention / trend via row-attention), Conv2D +
   TransConv multi-scale mixing, and amplitude-weighted multi-resolution mixing. Wrapped it as
   `models/TimeMixerPP.py` and tracked results in `results_comparison_pp.md`.

3. **Fought and won a multi-day NaN-stability battle** on TimeMixer++ (see below).

4. **Integrated SparseTSF** — vendored upstream under Apache-2.0 (`SparseTSF_model/`), wrote a
   1-arg→4-arg adapter (`models/SparseTSF.py`), added `--period_len` / `--model_type` to `run.py`,
   and registered it in `exp/exp_basic.py`. Accuracy tracked in `results_comparison_sparsetsf.md`.

5. **Ran the accuracy sweeps** — SparseTSF at both `seq_len=96` (comparable to our tables) and
   `seq_len=720` (its native setting), 40 runs total, on the 5 light datasets.

6. **Ran a matched-config carbon sweep** across all three models with **CodeCarbon** (60 runs) —
   `seq_len=96`, `d_model=32`, `batch=64`, **5 fixed epochs, early stopping off**, isolating
   *energy per unit of training work* by architecture.

7. **Wrote the consolidated report** — `results_final_report.md` (accuracy + carbon, 3 models).

8. **Conceptual Q&A** — explained the model differences; why TimeMixer++ and SparseTSF score
   similarly on Weather; which datasets suit SparseTSF (and why Electricity/Traffic would need long
   lookback because of their weekly 168 period); what `seq_len=96` means; how SparseTSF's
   `period_len` works; and whether `period_len` can be **auto-detected** (yes — TimeMixer++'s
   `_top_k_periods()` FFT machinery is exactly such a detector).

---

## Headline findings

**Accuracy (matched `seq_len=96`, mean MSE over 20 configs):**
- **TimeMixer 0.348** (best, 13 wins) ≈ **SparseTSF 0.354** (7 wins — sweeps the hourly ETTh
  datasets where its `period_len=24` locks onto the daily cycle)
- **TimeMixer++ 0.449** (0 wins) — **undertrained** at the stability-driven `lr=5e-5`, so this
  understates the architecture; it is *not* a clean "PP is worse" result.

**Accuracy (SparseTSF's native `seq_len=720`):** mean MSE drops to **0.312 — beating TimeMixer's
0.348**, with a **925-parameter** model.

**Energy (total kWh, matched protocol):**
- SparseTSF **7.77e-03 (0.09×)** ≪ TimeMixer **8.93e-02 (1.0×)** < TimeMixer++ **1.49e-01 (1.67×)**
- SparseTSF ≈ **11× less** energy than TimeMixer, ≈ **19× less** than TimeMixer++.
- TimeMixer++ ≈ **2.1× TimeMixer on ETT** (longer wall-clock from its FFT/attention/conv stack),
  narrowing to **1.19× on Weather** (21 channels raise TimeMixer's own channel-independent cost).

**Takeaway:** SparseTSF is the **efficiency winner** — it ties TimeMixer's accuracy at ~1/11 the
energy and ~1000× fewer parameters, and is the *most accurate* overall at its native lookback.
TimeMixer++'s added complexity bought **neither** accuracy nor efficiency in these runs.

---

## TimeMixer++ NaN-stability fix (hard-won — do not regress)

**Definitive fix:** skip the optimizer step when `clip_grad_norm_` returns a non-finite norm
(`exp/exp_long_term_forecasting.py`). Supporting layers: FFT-amplitude `isfinite`/clamp before
softmax, attention-score clamps, output `_sanitize` (torch.where isfinite), an EarlyStopping
NaN-checkpoint guard (`utils/tools.py`), Solar `use_norm=1`, and a conservative `lr=5e-5` + `lradj
type1`. The low LR is *why* PP is undertrained in the epoch budget.

---

## Key files (under `TimeMixer/`)

| File | Purpose |
|---|---|
| `TimeMixer_plus/model.py` | From-scratch TimeMixer++ implementation |
| `models/TimeMixerPP.py`, `models/SparseTSF.py` | Thin wrappers / adapters |
| `SparseTSF_model/model.py` | Vendored SparseTSF (Apache-2.0) |
| `run_all_pp_benchmarks.py`, `run_missing_pp.py` | TimeMixer++ accuracy runners |
| `run_sparsetsf_benchmarks.py` | SparseTSF accuracy runner (40 runs) |
| `run_carbon_benchmarks.py` | 3-model CodeCarbon sweep (60 runs) |
| `results_comparison.md` | TimeMixer v1 accuracy |
| `results_comparison_pp.md` | TimeMixer++ accuracy |
| `results_comparison_sparsetsf.md` | SparseTSF accuracy (both lookbacks + 3-way) |
| `results_carbon_comparison.md` | Energy / CO₂ |
| **`results_final_report.md`** | **Consolidated accuracy + carbon report** |

---

## Methodology caveats

- **Two protocols, do not cross-compare.** Accuracy tables use each model's own published
  hyperparameters with early stopping; carbon tables use a matched 5-epoch, no-early-stopping
  protocol (so the MSE in the carbon table is *not* real accuracy).
- **TimeMixer++ is undertrained** at `lr=5e-5` (stability requirement) — its accuracy understates
  the architecture; its energy measurement is fair.
- **Windows energy limits.** GPU power is real (NVML); CPU/RAM are TDP estimates (no RAPL). Relative
  ratios are sound; treat absolute kg CO₂ as an estimate (region: Türkiye).
- **Scope:** 5 light datasets (ETTh1/2, ETTm1/2, Weather). Solar-Energy, Electricity, Traffic were
  excluded as too heavy for the 4 GB GPU.
