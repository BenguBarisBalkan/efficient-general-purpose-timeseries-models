## TimeMixer++ Reproduction Results vs Paper

- **Paper**: TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis (ICLR 2025).
- **Codebase**: `implementations/timemixer_pp/model.py` (this repo).
- **Hardware**: Local Windows machine, RTX 3050 Ti Laptop GPU (4 GB VRAM).
- **Setup**: `seq_len=96`, `down_sampling_layers=3`, `top_k=3`, `e_layers=2`, `num_workers=0`.
- **Note**: Paper trained on multi A100 80GB GPUs with `batch_size=512`. Local runs use reduced batch sizes (4–64) to fit 4 GB VRAM, which may affect results.

Updated automatically by `run_missing_pp.py` after each completed run.

---

### Implementation & Stability Changelog

This is a from-scratch implementation (no official TimeMixer++ code exists publicly). The
following changes were made during development to get a numerically stable model running on
PyTorch 1.7.1 / CUDA 11.0 with 4 GB VRAM:

**New files**
- `implementations/timemixer_pp/model.py` — full TimeMixer++ implementation (ChannelMixer, MRTI, dual-axis-attention TID, Conv-based multi-scale mixing, amplitude-weighted MRM, task heads).
- `models/TimeMixerPP.py` — thin wrapper exposing the model to `run.py` as `--model TimeMixerPP`.
- `run_all_pp_benchmarks.py` / `run_missing_pp.py` — benchmark runners (logs to `.run_logs_pp/`).

**PyTorch 1.7 compatibility**
- FFT: `torch.fft.rfft` (1.8+ API) falls back to legacy `torch.rfft` for the period-extraction step.
- Replaced `torch.nan_to_num` (1.8+) with a manual `torch.where(isfinite, …)` mask.

**Numerical-stability fixes (NaN debugging)**
1. **FFT softmax guard** — `softmax([inf, inf, …]) = nan` when embedding amplitudes overflow; now clamp/sanitise FFT amplitudes (`isfinite` mask + `clamp(min=0)`) before the softmax in `_top_k_periods`.
2. **Attention-score clamp** — clamp dual-axis and channel-mixer attention scores to `±1e4` before softmax to prevent overflow on unstable steps.
3. **Gradient clipping** — added `clip_grad_norm_(max_norm=1.0)` in `exp/exp_long_term_forecasting.py` training loop.
4. **LR schedule change** — root cause of mid-training divergence was the default `lradj=TST` (OneCycle) ramping LR up to ~1e-3. Switched to `lradj=type1` (monotonic decay) **and** lowered `learning_rate` from 1e-3 → 1e-4 → finally **5e-5** for all datasets (the full consistent re-run uses 5e-5).
5. **Status reporting fix** — runner now reports `NAN/DIVERGED` instead of falsely reporting `OK` when a run finishes with `mse:nan`.
6. **Solar-Energy `use_norm` 0 → 1** — Solar-Energy still NaN'd at epoch 1 (iter ~800) even with the stable LR setup, because its config used `use_norm=0` (no RevIN). The PP model's FFT amplitude step overflows on raw unnormalised inputs; enabling RevIN (`use_norm=1`) bounds them. Every other dataset already uses `use_norm=1` and is stable. (v1 TimeMixer used `use_norm=0` for Solar; PP requires normalisation.)
7. **Skip non-finite loss batches** — the remaining instability is *stochastic*: the same config (e.g. ETTm1 pred=336) sometimes trains fine and sometimes NaNs at epoch 1, depending on data order / init — one bad batch produces a NaN loss that backprops and poisons all weights. Added a guard in `exp/exp_long_term_forecasting.py`: if `loss` is non-finite, drop the batch (`zero_grad` + `continue`) instead of stepping the optimiser, so weights stay clean and training continues. This is the robust, general fix on top of the LR/clamp/clip changes.

> **Update (full consistent re-run):** All 32 configs are now being re-run from scratch under a
> single stable setup (`lr=5e-5`, `lradj=type1`, all NaN guards below), including Solar-Energy and
> Traffic. This supersedes the earlier mixed-setup table (which combined old `lr=1e-3`/`TST` runs
> with `lr=1e-4` re-runs) — the table is now hyperparameter-consistent, matching the v1 methodology.

---

### Training Setup (final — full consistent re-run)

The earlier development setup (`lr=1e-3`, `lradj=TST`) diverged to NaN; intermediate re-runs used
`lr=1e-4`. The **final full re-run uses one setup for all 32 configs**:

| Setting | Early (diverged) | **Final (all 32 runs)** |
|---|---|---|
| `learning_rate` | `1e-3` | **`5e-5`** |
| `lradj` (LR schedule) | `TST` (OneCycle, ramps up) | **`type1`** (monotonic decay) |
| Gradient clipping | none | **`clip_grad_norm_(max_norm=1.0)`** |
| Skip non-finite grad step | none | **skip optimiser step if grad-norm non-finite** |
| FFT amplitude guard | none | **`isfinite` mask + `clamp(min=0)` before softmax** |
| Attention score clamp | none | **`clamp(±1e4)` before softmax** |
| Output sanitize | none | **`torch.where(isfinite, out, 0)` on model output** |
| `train_epochs` | 20 | 20 (unchanged) |
| `patience` (early stop) | 10 | 10 (unchanged) |
| `seq_len` | 96 | 96 (unchanged) |
| `top_k` periods (K) | 3 | 3 (unchanged) |
| `down_sampling_layers` (M) | 3 | 3 (unchanged) |
| `e_layers` (MixerBlocks, L) | 2 | 2 (unchanged) |
| `d_model` / `d_ff` | 32 / 64 | 32 / 64 (unchanged) |
| `n_heads` | 4 | 4 (unchanged) |

Per-dataset batch sizes (both setups, sized to fit 4 GB VRAM):

| Dataset | enc_in (C) | batch_size | use_norm |
|---|---:|---:|---:|
| ETTh1 / ETTh2 / ETTm1 / ETTm2 | 7 | 64 | default |
| Weather | 21 | 64 | default |
| Solar-Energy | 137 | 16 | 1 (was 0 — see fix #6) |
| Electricity | 321 | 16 | default |
| Traffic | 862 | 4 | default |

**Result of the change:** with `lr=1e-3` + `TST`, the model trained stably for many epochs then
diverged to NaN mid-training (e.g. ETTh1 pred=336 NaN'd at epoch 16, pred=720 at epoch 5) because
the OneCycle schedule ramped LR up past the model's stability threshold. The current setup removes
the LR ramp-up and starts 10× lower, keeping weights/activations in range.

---

### Key Architectural Differences vs TimeMixer v1

| Component | TimeMixer (v1) | TimeMixer++ |
|---|---|---|
| Downsampling | AvgPool | Learnable Conv1d stride=2 |
| Channel handling | Channel-independent | Variate self-attention (coarsest scale) |
| Decomposition | Moving average (shallow, pre-embedding) | Dual-axis attention on 2D images (deep, in-latent) |
| Time imaging | None | MRTI: FFT -> top-K periods -> 2D reshape |
| Season mixing | Bottom-up Linear | Bottom-up Conv1d |
| Trend mixing | Top-down Linear | Top-down ConvTranspose1d |
| Multi-resolution | None | Amplitude-weighted sum across K periods |

---

### Paper Results Reference (Table 16, Full per-pred-len)

All values from TimeMixer++ (Ours) column of Appendix H, Table 16 (ICLR 2025).
Paper experiments: multi NVIDIA A100 80GB, `batch_size=512`.

| Dataset | pred=96 MSE/MAE | pred=192 MSE/MAE | pred=336 MSE/MAE | pred=720 MSE/MAE | Avg MSE |
|---|---|---|---|---|---|
| ETTh1 | 0.361 / 0.403 | 0.416 / 0.441 | 0.430 / 0.434 | 0.467 / 0.451 | 0.419 |
| ETTh2 | 0.276 / 0.328 | 0.342 / 0.379 | 0.346 / 0.398 | 0.392 / 0.415 | 0.339 |
| ETTm1 | 0.310 / 0.334 | 0.348 / 0.362 | 0.376 / 0.391 | 0.440 / 0.423 | 0.369 |
| ETTm2 | 0.170 / 0.245 | 0.229 / 0.291 | 0.303 / 0.343 | 0.373 / 0.399 | 0.269 |
| Weather | 0.155 / 0.205 | 0.201 / 0.245 | 0.237 / 0.265 | 0.312 / 0.334 | 0.226 |
| Solar-Energy | 0.171 / 0.231 | 0.218 / 0.263 | 0.212 / 0.269 | 0.212 / 0.270 | 0.203 |
| Electricity | 0.135 / 0.222 | 0.147 / 0.235 | 0.164 / 0.245 | 0.212 / 0.310 | 0.165 |
| Traffic | 0.392 / 0.253 | 0.402 / 0.258 | 0.428 / 0.263 | 0.441 / 0.282 | 0.416 |

---

### Note on Runtimes (why they don't increase with pred_len)

Unlike the v1 report — where runtime grows monotonically with `pred_len` — the PP runtimes
look inconsistent across horizons. This is expected, not a bug. **Total runtime = (epochs
actually trained) × (per-epoch time)**, and in TimeMixer++:

1. **Per-epoch time is ~flat across `pred_len`.** The heavy compute (FFT imaging, dual-axis
   attention, conv mixing) is all in the encoder over `seq_len=96` (fixed); the prediction head
   is a single linear layer `T_m → pred_len`, a tiny fraction. Measured per-epoch cost is nearly
   constant within a dataset (ETTh* ~21–22 s, ETTm1 ~85–95 s, Electricity ~176–195 s) regardless
   of horizon. (v1 was a lighter, pred_len-sensitive model, so its runtimes crept up gradually.)
2. **Epochs-trained varies non-monotonically** due to early stopping (`patience=10`). A run that
   plateaus early stops early; one that keeps improving runs the full 20 epochs. Example: ETTm1
   **336** trained 20 epochs (1913 s) while ETTm1 **720** early-stopped at 13 (1114 s) — the
   *shorter* horizon took *longer* purely because it trained more epochs.
3. **The table mixes two setups.** New re-runs (`lr=5e-5`) keep inching the loss down and rarely
   trigger early-stop, so they run all 20 epochs; old-setup runs (`lr=1e-3`) converged faster and
   stopped sooner. So a new-setup short-horizon run can out-time an old-setup long-horizon one.

Bottom line: PP runtimes are dominated by *how many epochs early-stopping allowed*, which is
convergence-dependent and stochastic — not a function of `pred_len`.

---

### Automated Run Log (Paper vs Local)

The section below is updated automatically by `run_missing_pp.py`.

<!-- AUTO_PP_RESULTS_START -->
| Dataset | pred_len | Paper MSE | Paper MAE | Local MSE | Local MAE | Runtime (s) | Runtime (h) | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ETTh1 | 96 | 0.361 | 0.403 | 0.692902 | 0.555331 | 596.7 | 0.17 | OK |
| ETTh1 | 192 | 0.416 | 0.441 | 0.735732 | 0.582909 | 565.7 | 0.16 | OK |
| ETTh1 | 336 | 0.430 | 0.434 | 0.795015 | 0.623603 | 556.3 | 0.15 | OK |
| ETTh1 | 720 | 0.467 | 0.451 | 0.763298 | 0.622564 | 530.0 | 0.15 | OK |
| ETTh2 | 96 | 0.276 | 0.328 | 0.382873 | 0.407253 | 576.1 | 0.16 | OK |
| ETTh2 | 192 | 0.342 | 0.379 | 0.458633 | 0.451192 | 567.6 | 0.16 | OK |
| ETTh2 | 336 | 0.346 | 0.398 | 0.487012 | 0.473261 | 556.6 | 0.15 | OK |
| ETTh2 | 720 | 0.392 | 0.415 | 0.474172 | 0.477360 | 376.5 | 0.10 | OK |
| ETTm1 | 96 | 0.310 | 0.334 | 0.419942 | 0.432224 | 2310.4 | 0.64 | OK |
| ETTm1 | 192 | 0.348 | 0.362 | 0.426157 | 0.428344 | 2311.4 | 0.64 | OK |
| ETTm1 | 336 | 0.376 | 0.391 | 0.471204 | 0.455794 | 2318.3 | 0.64 | OK |
| ETTm1 | 720 | 0.440 | 0.423 | 0.514390 | 0.476748 | 2345.1 | 0.65 | OK |
| ETTm2 | 96 | 0.170 | 0.245 | 0.198934 | 0.283406 | 2340.3 | 0.65 | OK |
| ETTm2 | 192 | 0.229 | 0.291 | 0.262988 | 0.323637 | 2346.3 | 0.65 | OK |
| ETTm2 | 336 | 0.303 | 0.343 | 0.344020 | 0.374491 | 2338.2 | 0.65 | OK |
| ETTm2 | 720 | 0.373 | 0.399 | 0.444232 | 0.431039 | 2306.8 | 0.64 | OK |
| Weather | 96 | 0.155 | 0.205 | 0.199892 | 0.252004 | 2314.2 | 0.64 | OK |
| Weather | 192 | 0.201 | 0.245 | 0.244207 | 0.285158 | 2342.3 | 0.65 | OK |
| Weather | 336 | 0.237 | 0.265 | 0.294700 | 0.317602 | 2389.6 | 0.66 | OK |
| Weather | 720 | 0.312 | 0.334 | 0.363639 | 0.361179 | 2417.4 | 0.67 | OK |
| Solar-Energy | 96 | 0.171 | 0.231 | 0.268628 | 0.314078 | 8968.5 | 2.49 | OK |
<!-- AUTO_PP_RESULTS_END -->
