# Energy & Carbon: TimeMixer vs TimeMixer++ (CodeCarbon)

Measures the **energy and CO2 cost** of training TimeMixer v1 vs our TimeMixer++ implementation,
to see whether TimeMixer++'s heavier architecture (FFT time-imaging + dual-axis attention +
conv mixing) carries a meaningful energy premium over v1's MLP-based design.

- **Tool**: [CodeCarbon](https://codecarbon.io) 3.2.3, via the opt-in `--track_emissions` flag in `run.py`
- **Hardware**: RTX 3050 Ti Laptop GPU (4 GB), Windows
- **Runner**: `run_carbon_benchmarks.py` -> logs in `.run_logs_carbon/`, raw data in `emissions_timemixer.csv`

---

## Methodology

**Matched configuration + fixed epochs.** Both models get *identical* hyperparameters and a fixed
training budget with early stopping disabled, so the measured difference reflects the architecture
rather than differing settings. (The accuracy benchmarks use each model's own published config —
v1 `d_model=16/bs=128/10ep/lr=1e-2` vs PP `d_model=32/bs=64/20ep/lr=5e-5` — which would otherwise
dominate any energy comparison.)

| Setting | Value (both models) |
|---|---|
| `d_model` / `d_ff` | 32 / 64 |
| `n_heads` | 4 |
| `e_layers` | 2 |
| `down_sampling_layers` | 3 |
| `batch_size` | 64 |
| `train_epochs` | **5 (fixed)** |
| `patience` | **999 (early stopping disabled)** |
| `learning_rate` | 5e-5 |
| `lradj` | type1 |
| `seq_len` | 96 |
| `top_k` | 3 |

Fixed epochs matter: with early stopping active the two models would train for *different* numbers
of epochs, so the comparison would measure convergence luck rather than architectural cost.
`lr=5e-5` + `type1` is used for both because TimeMixer++ diverges to NaN at higher learning rates
(see the stability changelog in `results_comparison_pp.md`). **Accuracy is secondary here** —
5 epochs is a measurement budget, not a converged model — the metric of interest is energy per
unit of training work.

**Datasets**: only the light ones — ETTh1, ETTh2, ETTm1, ETTm2 (7 variables) and Weather (21).
Solar-Energy, Electricity and Traffic are excluded as too heavy (2.5 h - 25 h per run).

**Run matrix**: 2 models x 5 datasets x 4 prediction lengths = **40 runs**.

---

## Head-to-Head Summary

Energy totalled across all 4 prediction lengths per dataset. The ratio is the headline number:
how many times more energy TimeMixer++ consumes for the same training work.

<!-- AUTO_CARBON_SUMMARY_START -->
| Dataset | Runs | TimeMixer kWh | TimeMixerPP kWh | SparseTSF kWh | TimeMixerPP /TimeMixer | SparseTSF /TimeMixer |
|---|---:|---:|---:|---:|---:|---:|
| ETTh1 | 4 | 5.671e-03 | 1.176e-02 | 1.061e-03 | 2.07x | 0.19x |
| ETTh2 | 4 | 4.865e-03 | 1.024e-02 | 5.373e-04 | 2.10x | 0.11x |
| ETTm1 | 4 | 1.936e-02 | 4.026e-02 | 1.906e-03 | 2.08x | 0.10x |
| ETTm2 | 4 | 1.845e-02 | 3.825e-02 | 1.526e-03 | 2.07x | 0.08x |
| Weather | 4 | 4.100e-02 | 4.891e-02 | 2.743e-03 | 1.19x | 0.07x |
| **TOTAL** |  | **8.934e-02** | **1.494e-01** | **7.773e-03** | **1.67x** | **0.09x** |

_Total CO2 — TimeMixer: 4.151e-02 kg | TimeMixerPP: 6.941e-02 kg | SparseTSF: 3.611e-03 kg_
<!-- AUTO_CARBON_SUMMARY_END -->

---

## Per-Run Results

MSE/MAE are included for context only (5-epoch budget, so these are **not** converged accuracy
numbers — for those see `results_comparison.md` and `results_comparison_pp.md`).

<!-- AUTO_CARBON_RESULTS_START -->
| Dataset | pred_len | Model | Duration (s) | GPU kWh | CPU kWh | RAM kWh | Total kWh | CO2 (kg) | MSE | MAE |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | 96 | TimeMixer | 68.1 | 9.517e-04 | 4.743e-04 | 1.822e-04 | 1.608e-03 | 7.471e-04 | 0.540181 | 0.486696 |
| ETTh1 | 96 | TimeMixerPP | 207.8 | 1.810e-03 | 1.120e-03 | 5.572e-04 | 3.488e-03 | 1.620e-03 | 0.704551 | 0.560394 |
| ETTh1 | 96 | SparseTSF | 16.5 | 1.417e-04 | 1.011e-04 | 4.427e-05 | 2.871e-04 | 1.334e-04 | 1.042396 | 0.669582 |
| ETTh1 | 192 | TimeMixer | 60.8 | 8.762e-04 | 3.284e-04 | 1.631e-04 | 1.368e-03 | 6.354e-04 | 0.583259 | 0.513193 |
| ETTh1 | 192 | TimeMixerPP | 200.0 | 1.668e-03 | 9.379e-04 | 5.355e-04 | 3.141e-03 | 1.459e-03 | 0.746751 | 0.587964 |
| ETTh1 | 192 | SparseTSF | 16.6 | 1.003e-04 | 9.247e-05 | 8.914e-05 | 2.819e-04 | 1.310e-04 | 0.878431 | 0.620341 |
| ETTh1 | 336 | TimeMixer | 60.6 | 9.166e-04 | 3.001e-04 | 1.626e-04 | 1.379e-03 | 6.408e-04 | 0.662450 | 0.557910 |
| ETTh1 | 336 | TimeMixerPP | 187.2 | 1.504e-03 | 8.875e-04 | 5.016e-04 | 2.893e-03 | 1.344e-03 | 0.809826 | 0.629891 |
| ETTh1 | 336 | SparseTSF | 19.2 | 1.481e-04 | 1.207e-04 | 5.041e-05 | 3.192e-04 | 1.483e-04 | 0.967761 | 0.661521 |
| ETTh1 | 720 | TimeMixer | 56.0 | 9.400e-04 | 2.263e-04 | 1.499e-04 | 1.316e-03 | 6.115e-04 | 0.701401 | 0.595732 |
| ETTh1 | 720 | TimeMixerPP | 157.0 | 1.183e-03 | 6.297e-04 | 4.206e-04 | 2.233e-03 | 1.038e-03 | 0.778335 | 0.628671 |
| ETTh1 | 720 | SparseTSF | 15.6 | 7.095e-05 | 6.007e-05 | 4.184e-05 | 1.729e-04 | 8.031e-05 | 0.934525 | 0.669013 |
| ETTh2 | 96 | TimeMixer | 51.6 | 7.800e-04 | 2.135e-04 | 1.377e-04 | 1.131e-03 | 5.256e-04 | 0.341510 | 0.381162 |
| ETTh2 | 96 | TimeMixerPP | 169.7 | 1.212e-03 | 6.948e-04 | 4.543e-04 | 2.361e-03 | 1.097e-03 | 0.386083 | 0.409338 |
| ETTh2 | 96 | SparseTSF | 11.6 | 4.931e-05 | 3.899e-05 | 3.083e-05 | 1.191e-04 | 5.535e-05 | 0.442176 | 0.441742 |
| ETTh2 | 192 | TimeMixer | 55.0 | 8.338e-04 | 2.471e-04 | 1.471e-04 | 1.228e-03 | 5.705e-04 | 0.416016 | 0.422016 |
| ETTh2 | 192 | TimeMixerPP | 184.2 | 2.081e-03 | 7.644e-04 | 4.933e-04 | 3.339e-03 | 1.551e-03 | 0.459964 | 0.452052 |
| ETTh2 | 192 | SparseTSF | 12.0 | 5.719e-05 | 4.574e-05 | 3.200e-05 | 1.349e-04 | 6.268e-05 | 0.494032 | 0.467142 |
| ETTh2 | 336 | TimeMixer | 55.3 | 8.713e-04 | 2.404e-04 | 1.480e-04 | 1.260e-03 | 5.853e-04 | 0.442438 | 0.451251 |
| ETTh2 | 336 | TimeMixerPP | 157.1 | 1.335e-03 | 5.704e-04 | 4.207e-04 | 2.327e-03 | 1.081e-03 | 0.489858 | 0.474745 |
| ETTh2 | 336 | SparseTSF | 8.8 | 5.392e-05 | 2.019e-05 | 2.308e-05 | 9.719e-05 | 4.515e-05 | 0.514094 | 0.487789 |
| ETTh2 | 720 | TimeMixer | 53.6 | 9.145e-04 | 1.885e-04 | 1.433e-04 | 1.246e-03 | 5.790e-04 | 0.455553 | 0.463259 |
| ETTh2 | 720 | TimeMixerPP | 147.5 | 1.317e-03 | 4.960e-04 | 3.955e-04 | 2.208e-03 | 1.026e-03 | 0.474172 | 0.477360 |
| ETTh2 | 720 | SparseTSF | 15.6 | 8.755e-05 | 5.666e-05 | 4.185e-05 | 1.861e-04 | 8.644e-05 | 0.507548 | 0.491021 |
| ETTm1 | 96 | TimeMixer | 192.5 | 3.155e-03 | 7.035e-04 | 5.164e-04 | 4.375e-03 | 2.033e-03 | 0.361940 | 0.382999 |
| ETTm1 | 96 | TimeMixerPP | 643.9 | 6.435e-03 | 2.250e-03 | 1.728e-03 | 1.041e-02 | 4.838e-03 | 0.427006 | 0.436269 |
| ETTm1 | 96 | SparseTSF | 28.6 | 1.745e-04 | 1.135e-04 | 7.644e-05 | 3.644e-04 | 1.693e-04 | 0.651327 | 0.526147 |
| ETTm1 | 192 | TimeMixer | 193.8 | 3.248e-03 | 6.676e-04 | 5.200e-04 | 4.435e-03 | 2.061e-03 | 0.406811 | 0.404799 |
| ETTm1 | 192 | TimeMixerPP | 607.1 | 5.007e-03 | 2.063e-03 | 1.629e-03 | 8.699e-03 | 4.041e-03 | 0.429471 | 0.430489 |
| ETTm1 | 192 | SparseTSF | 32.2 | 2.008e-04 | 1.234e-04 | 8.529e-05 | 4.095e-04 | 1.902e-04 | 0.650931 | 0.527806 |
| ETTm1 | 336 | TimeMixer | 206.9 | 3.575e-03 | 7.417e-04 | 5.549e-04 | 4.872e-03 | 2.263e-03 | 0.440777 | 0.428966 |
| ETTm1 | 336 | TimeMixerPP | 653.1 | 7.606e-03 | 2.195e-03 | 1.752e-03 | 1.155e-02 | 5.367e-03 | 0.479843 | 0.460570 |
| ETTm1 | 336 | SparseTSF | 40.4 | 2.045e-04 | 1.500e-04 | 1.079e-04 | 4.623e-04 | 2.148e-04 | 0.668779 | 0.540170 |
| ETTm1 | 720 | TimeMixer | 233.4 | 4.209e-03 | 8.408e-04 | 6.257e-04 | 5.676e-03 | 2.637e-03 | 0.504058 | 0.462860 |
| ETTm1 | 720 | TimeMixerPP | 626.0 | 5.874e-03 | 2.037e-03 | 1.680e-03 | 9.591e-03 | 4.456e-03 | 0.518629 | 0.479064 |
| ETTm1 | 720 | SparseTSF | 55.2 | 3.255e-04 | 1.964e-04 | 1.476e-04 | 6.695e-04 | 3.110e-04 | 0.700791 | 0.558462 |
| ETTm2 | 96 | TimeMixer | 183.2 | 3.024e-03 | 6.183e-04 | 4.906e-04 | 4.133e-03 | 1.920e-03 | 0.190065 | 0.275710 |
| ETTm2 | 96 | TimeMixerPP | 611.2 | 5.177e-03 | 2.007e-03 | 1.640e-03 | 8.824e-03 | 4.100e-03 | 0.200966 | 0.285273 |
| ETTm2 | 96 | SparseTSF | 28.7 | 1.416e-04 | 1.072e-04 | 7.690e-05 | 3.257e-04 | 1.513e-04 | 0.239887 | 0.318128 |
| ETTm2 | 192 | TimeMixer | 188.6 | 3.192e-03 | 6.183e-04 | 5.055e-04 | 4.316e-03 | 2.005e-03 | 0.253502 | 0.312958 |
| ETTm2 | 192 | TimeMixerPP | 616.7 | 5.267e-03 | 2.119e-03 | 1.655e-03 | 9.042e-03 | 4.201e-03 | 0.263987 | 0.324196 |
| ETTm2 | 192 | SparseTSF | 31.5 | 1.581e-04 | 1.075e-04 | 8.476e-05 | 3.504e-04 | 1.628e-04 | 0.296118 | 0.348572 |
| ETTm2 | 336 | TimeMixer | 199.2 | 3.454e-03 | 6.487e-04 | 5.337e-04 | 4.637e-03 | 2.154e-03 | 0.315517 | 0.353349 |
| ETTm2 | 336 | TimeMixerPP | 614.4 | 5.096e-03 | 2.009e-03 | 1.649e-03 | 8.755e-03 | 4.067e-03 | 0.341936 | 0.373362 |
| ETTm2 | 336 | SparseTSF | 31.6 | 1.884e-04 | 1.108e-04 | 8.483e-05 | 3.840e-04 | 1.784e-04 | 0.353933 | 0.381429 |
| ETTm2 | 720 | TimeMixer | 224.0 | 4.054e-03 | 7.091e-04 | 6.009e-04 | 5.364e-03 | 2.492e-03 | 0.414277 | 0.406885 |
| ETTm2 | 720 | TimeMixerPP | 656.9 | 7.492e-03 | 2.372e-03 | 1.762e-03 | 1.163e-02 | 5.401e-03 | 0.442128 | 0.429478 |
| ETTm2 | 720 | SparseTSF | 37.7 | 2.493e-04 | 1.158e-04 | 1.006e-04 | 4.656e-04 | 2.163e-04 | 0.447812 | 0.429947 |
| Weather | 96 | TimeMixer | 342.5 | 6.689e-03 | 1.135e-03 | 9.186e-04 | 8.743e-03 | 4.062e-03 | 0.197428 | 0.239963 |
| Weather | 96 | TimeMixerPP | 665.6 | 7.797e-03 | 2.390e-03 | 1.785e-03 | 1.197e-02 | 5.562e-03 | 0.200655 | 0.252812 |
| Weather | 96 | SparseTSF | 25.5 | 1.545e-04 | 8.036e-05 | 6.793e-05 | 3.028e-04 | 1.407e-04 | 0.231809 | 0.269980 |
| Weather | 192 | TimeMixer | 365.9 | 7.124e-03 | 1.286e-03 | 9.810e-04 | 9.391e-03 | 4.363e-03 | 0.241950 | 0.275476 |
| Weather | 192 | TimeMixerPP | 671.3 | 7.934e-03 | 2.437e-03 | 1.801e-03 | 1.217e-02 | 5.655e-03 | 0.244863 | 0.285754 |
| Weather | 192 | SparseTSF | 45.3 | 2.744e-04 | 2.426e-04 | 1.215e-04 | 6.385e-04 | 2.967e-04 | 0.278813 | 0.304826 |
| Weather | 336 | TimeMixer | 404.3 | 7.916e-03 | 1.325e-03 | 1.085e-03 | 1.033e-02 | 4.797e-03 | 0.292198 | 0.310621 |
| Weather | 336 | TimeMixerPP | 682.0 | 8.079e-03 | 2.427e-03 | 1.829e-03 | 1.233e-02 | 5.731e-03 | 0.295136 | 0.317941 |
| Weather | 336 | SparseTSF | 56.0 | 3.068e-04 | 2.527e-04 | 1.498e-04 | 7.093e-04 | 3.295e-04 | 0.327650 | 0.336269 |
| Weather | 720 | TimeMixer | 494.6 | 9.625e-03 | 1.585e-03 | 1.327e-03 | 1.254e-02 | 5.824e-03 | 0.363954 | 0.355832 |
| Weather | 720 | TimeMixerPP | 685.6 | 8.316e-03 | 2.279e-03 | 1.839e-03 | 1.243e-02 | 5.777e-03 | 0.363980 | 0.361476 |
| Weather | 720 | SparseTSF | 83.9 | 4.818e-04 | 3.863e-04 | 2.245e-04 | 1.093e-03 | 5.076e-04 | 0.395773 | 0.378712 |
<!-- AUTO_CARBON_RESULTS_END -->

---

## Caveats

- **Windows measurement limits.** CodeCarbon reads *real* GPU power via NVML, but Windows exposes
  no RAPL interface, so **CPU and RAM energy are TDP-based estimates**, not measurements. GPU
  dominates training energy and both models are measured identically on the same machine, so the
  *relative* comparison holds; treat absolute kg CO2 as an estimate.
- **Carbon intensity** comes from CodeCarbon's regional dataset (detected region: Türkiye), not
  live grid data.
- **Shared stability guards** — gradient clipping, non-finite-gradient step skipping, and the
  EarlyStopping NaN guard in `exp/exp_long_term_forecasting.py` / `utils/tools.py` — apply to
  *both* models, so the small overhead they add is equal on both sides.
- **Not comparable to the accuracy tables**, which used each model's own hyperparameters and full
  early stopping.
- Energy includes the full `exp.train()` + `exp.test()` window; data loading and Python startup
  happen outside the tracker.
