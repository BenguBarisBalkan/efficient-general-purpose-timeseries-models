# SparseTSFPlus — design, measurements and outcomes

Consolidated write-up for the fourth model in this study: **SparseTSFPlus**, SparseTSF with
cheap, individually-toggleable units grafted on from TimeMixer and TimeMixer++.

Headline: **on imputation, SparseTSFPlus reaches TimeMixer's accuracy on ETTh1 while adding
zero parameters, and beats it on ETTm1 with 597 parameters against ~75,500.** On PEMS
short-term it cuts MAE by 24% for ~70 extra parameters. On classification no variant
improved the average, which is reported here as a result rather than omitted.

Supporting tables:
[period sensitivity](results_period_sensitivity.md) ·
[imputation ablation](results_sparsetsf_plus_ablation.md) ·
[classification ablation](results_sparsetsf_plus_classification.md)

---

## 1. What was planned, and why

The 8-task generality sweep left SparseTSF behind on several tasks, and the gaps were
**not** uniform — it already beats TimeMixer on ETTh1/ETTh2 long-term forecasting and wins
zero-shot transfer 6/6. The three largest gaps had three structurally different causes, so
the plan deliberately did not look for one universal fix:

| Gap | Size | Diagnosed cause |
|---|---|---|
| PEMS multivariate short-term | +96% MAE vs TimeMixer | cross-channel correlation **and** a tiny forecast head |
| Classification (10 UEA) | −15.0 accuracy points | an ad-hoc flatten-linear readout |
| Imputation | +52% (ETTm1), +36% (ETTh1) | a phase-blind reconstruction linear that cannot distinguish a masked zero from an observed one |

Three scoping decisions shaped everything that follows:

1. **Parameter budget ≤2k.** SparseTSF's value is its size (41 params at sl=96, 925 at
   sl=720). A fix that costs 75k parameters answers a different question. Every shipped
   configuration stays inside ~700 parameters, so the efficiency claim survives intact.
2. **No cross-channel mixing.** Channel independence is why SparseTSF's parameter count does
   not grow with channel count, why its checkpoints transfer between datasets (the zero-shot
   runner literally copies a checkpoint across datasets), and why it wins zero-shot. Trading
   that away for PEMS was judged too expensive, so PEMS parity was explicitly **not** a goal.
3. **Imputation first.** It is the only one of the three weak tasks with a *locally-run,
   same-machine* TimeMixer baseline. Every other comparison is against published numbers from
   other hardware, so a result on imputation is the most defensible thing the study can produce.

### The design invariant

Every unit is individually toggleable, **defaults to off**, and residual units are
zero-initialised. With all units off SparseTSFPlus is *numerically identical* to SparseTSF,
and vanilla forecasting delegates directly to the untouched vendored Apache-2.0 core, so
reproduction is true by construction rather than by argument.

This is enforced, not assumed:

```bash
venv/Scripts/python.exe benchmarks/tools/assert_sparsetsf_plus_equiv.py
```

asserts bitwise equality on all five TSLib tasks — currently `max|diff| = 0.000e+00` for
every one, across both the delegated and re-implemented forecast paths. The same invariant
is confirmed end-to-end through the full training loop: the ablation's `base` row reproduces
the published ETTh1 imputation cell (0.214076) to six decimals, and `flat` reproduces all ten
published UEA accuracies exactly.

The practical consequences are that no already-published number in this repository is
invalidated, and that the ablation has a genuine baseline row rather than a
re-implementation that merely ought to match.

---

## 2. What was added, and why

### Code

| Path | Role |
|---|---|
| `implementations/sparsetsf_plus/units.py` | the units, plus the phase-aligned fold and mask-aware statistics |
| `implementations/sparsetsf_plus/model.py` | task branching (all five TSLib tasks) and the unit wiring |
| `models/SparseTSFPlus.py` | registry entry point (thin, as with `models/TimeMixerPP.py`) |
| `exp/exp_basic.py` | registers `SparseTSFPlus` in `model_dict` |
| `run.py` | 15 additive `--stsf_*` flags, every default equal to stock SparseTSF |
| `benchmarks/run_period_sensitivity.py` | `period_len` diagnostics using existing flags only |
| `benchmarks/run_sparsetsf_plus_imputation.py` | imputation unit ablation |
| `benchmarks/run_sparsetsf_plus_classification.py` | classification unit ablation |
| `benchmarks/tools/assert_sparsetsf_plus_equiv.py` | the equivalence gate |

The forecasting and reconstruction backbone remains **strictly channel-independent**: no unit
introduces a parameter that depends on `enc_in`. The classification readout is necessarily
channel-coupled, being a readout over all channels rather than backbone mixing.

### The units

Notation: `S`=seq_len, `H`=pred_len, `p`=period_len, `C`=enc_in, `K`=num_class.

| Unit | Flag | Params | Rationale |
|---|---|---|---|
| **Phase-axis mixing** | `--stsf_phase_mix linear` | `p²` | SparseTSF's per-channel forecast map is exactly `I_p ⊗ W_n`: one `n_x→n_y` matrix applied independently at each of `p` phases, so within-period **shape** is never modelled. This makes it `W_p ⊗ W_n` — a Kronecker-factorised full `S→P` map costing `p² + n_x·n_y` instead of `S·P` (592 vs 9,216 for ETTh1 96→96). Borrowed from TimeMixer++'s dual-axis `TimeImageDecomposition`, reduced from attention to a shared linear. Crucially **`seq_len`-free**: the same `p²` at sl=96 and sl=720. |
| **Refinement pass** | `--stsf_impute_passes 2` | **0** | The imputation loss is scored on masked positions only (`exp/exp_imputation.py:78`), and at mask 0.5 half the aggregating conv's receptive field is literal zeros. Pass 1 estimates, the holes are filled, statistics are recomputed on the now-complete series, and the **same weights** refine it. |
| **Mask-aware conv** | `--stsf_mask_conv 1` | **0** | Partial-convolution renormalisation: rescale the aggregate by how much of each window was actually observed, so a masked zero no longer drags it toward zero. |
| **Multi-period ensemble** | `--stsf_periods "8,24"` | `n_x·n_y + conv` per period | Runs the fold at several periods and sums behind zero-initialised gates (TimeMixer's free multi-scale ensemble, applied to periods). |
| **Trend/season split** | `--stsf_decomp free` | **0** | `series_decomp` from `layers/Autoformer_EncDec.py`; trend persists its last value. |
| **Leave-one-out recon** | `--stsf_diag_mask 1` | **0** | The reconstruction linear plus the conv residual makes the identity map exactly representable; zeroing the diagonal forces a genuine leave-one-out prediction. |
| **Phase-aligned fold** | `--stsf_pad_fold 1` | **0** | Lifts the requirement that `period_len` divide `seq_len`/`pred_len`, front-padding so forecast step `h` still lands in phase `h % p`. |
| **Classification heads** | `--stsf_cls_head stats\|segpool` | `3CK+K` / `n_seg·CK+K` | Pooled readouts, 18–513× smaller than the flatten-linear head. |

### Parameter budget, measured

| Configuration | Params | Reference |
|---|---:|---|
| PEMS08 base | 21 | — |
| PEMS08 + phase mixing (`p=4`) | **93** | dense equivalent: 1,153 |
| ETTh1 imputation, refinement pass | **41** | TimeMixer ~75,497 |
| ETTm1 imputation, all units | **597** | TimeMixer ~75,497 |
| ETTh1 forecasting + phase mixing | 617 | TimeMixer 75,497 |
| ETTh1 sl=720 + phase mixing | 1,501 | TimeMixer 4,046,633 |

Every shipped configuration is inside the ≤2k budget; the stretch to 10k was never needed.

---

## 3. How the results changed

### Imputation — the gap is eliminated

Mean masked MSE over four mask rates (0.125 / 0.25 / 0.375 / 0.5), variant selected on
**validation** loss. Both baselines were run on this machine under this protocol.

| | Variant | Params | Mean MSE | vs SparseTSF | vs TimeMixer |
|---|---|---:|---:|---:|---:|
| **ETTh1** | refinement pass | **41** (zero added) | **0.120684** | −26.1% | **+0.5%** — parity, 1,841× smaller |
| **ETTm1** | all units | **597** | **0.042010** | −41.2% | **−10.7%** — win, 126× smaller |

Per mask rate:

| mask_rate | ETTh1 SparseTSF | **ETTh1 Plus** | ETTh1 TimeMixer | ETTm1 SparseTSF | **ETTm1 Plus** | ETTm1 TimeMixer |
|---:|---:|---:|---:|---:|---:|---:|
| 0.125 | 0.113335 | **0.089294** | 0.096742 | 0.048898 | **0.034291** | 0.039071 |
| 0.25 | 0.146550 | **0.105566** | 0.111707 | 0.062403 | **0.036845** | 0.042218 |
| 0.375 | 0.179265 | **0.128675** | 0.125347 | 0.078078 | **0.042312** | 0.048693 |
| 0.5 | 0.214076 | **0.159201** | 0.146432 | 0.096544 | **0.054591** | 0.058143 |
| **mean** | 0.163307 | **0.120684** | 0.120057 | 0.071481 | **0.042010** | 0.047031 |

Against TimeMixer the gap moves from **+36.0% → +0.5%** on ETTh1 and from **+52.0% → −10.7%**
on ETTm1. ETTh1 beats TimeMixer outright at the two lower mask rates.

Two findings behind the numbers:

- **The parameter-free unit does most of the work.** On ETTh1 the refinement pass alone is
  the best configuration; the model remains byte-identical in size to stock SparseTSF.
- **Unit composition depends on training-set size.** ETTm1 (~34k windows) benefits from every
  unit and they compose; ETTh1 (~8.5k windows) is best with the free unit alone. The ordering
  is stable across all four mask rates independently, so a single universal configuration is
  not optimal — which is a finding, not an inconvenience.

Validation ranked all 24 cells in exactly the test order, so the selection does not depend on
the test signal.

### PEMS short-term — 24% of MAE removed, channel-independently

PEMS08, `seq_len=96 → pred_len=12`:

| Configuration | Params | MAE | MAPE (%) | RMSE |
|---|---:|---:|---:|---:|
| published (`period_len=12`) | 21 | 29.25 | 17.14 | 44.61 |
| **`p=4` + phase mixing** | **93** | **22.22** | 12.85 | 35.26 |
| **`p=12` + phase mixing** | **165** | **22.21** | 12.85 | 35.24 |
| dense (`period_len=1`) | 1,153 | 22.11 | 12.78 | 35.15 |

Phase mixing at **93 parameters matches the 1,153-parameter dense model to within 0.5%** — a
12× saving, and the Kronecker argument confirmed empirically. MAE falls **29.25 → 22.22
(−24.0%)**, closing roughly 59% of the distance to TimeMixer's published 17.41. The residual
is the cross-channel component that was deliberately out of scope.

Read the trend rather than the absolute gap here: this is PEMS08 alone against 4-dataset
published averages.

### Long-term forecasting — a free improvement from period selection

`period_len` selected on validation rather than taken from upstream:

| Dataset | published `p` | selected `p` | MSE before | MSE after | TimeMixer (local) |
|---|---:|---:|---:|---:|---:|
| **ETTh1** | 24 | **12** | 0.396635 | **0.383847** (−3.22%) | 0.385794 |
| Weather | 4 | 4 | 0.198635 | unchanged | 0.164526 |
| ETTm1 | 4 | 4 | 0.362121 | unchanged | 0.326398 |

ETTh1 flips from a **+2.81% loss to a −0.51% win** over TimeMixer for 36 extra parameters and
no code change. Weather and ETTm1 are period-*insensitive*: both degrade monotonically as the
period grows, and the published value is already optimal on validation and test.

With phase mixing added, ETTm1 improves a further −1.99% (0.354915) and Weather −0.35%.

### Classification — no improvement

| Variant | 10-set average | Note |
|---|---:|---|
| **`flat` (control)** | **60.90** | reproduces all ten published accuracies exactly |
| `flat_fixed` | 59.17 | |
| `stats` | 36.69 | |
| `stats_p25` | 39.30 | |
| `segpool`, `segpool_p25` | screened on 4 datasets | |

Pooled readouts are 18–513× smaller but lose accuracy on shape- and timing-driven datasets:
UWaveGestureLibrary falls 72.81 → 9.38 and Handwriting 8.59 → 3.41 under `stats`. Three
statistics per channel cannot encode a pen trajectory, and preserving coarse temporal order
via segment pooling does not recover it either.

**One genuine, dataset-specific win:** Heartbeat **54.63 → 71.71 (+17.1 points)** with
`stats`, at **369 parameters against the flat head's 49,413** — 134× smaller and substantially
more accurate. 61 channels × 405 steps against 204 training samples is a 242:1
parameter-to-sample ratio, so the flatten-linear head overfits outright. The pattern is that
pooling helps where a task is amplitude- or statistics-driven and hurts where it is
shape-driven.

Classification remains ~15 points behind TimeMixer++'s published 75.9, and this work did not
close that gap.

---

## 4. Runtime

107 training runs. **Training time is measured exactly** from the per-epoch `cost time` lines
in each log; wall clock is as observed during the session.

| Experiment group | Runs | Training time | Wall clock |
|---|---:|---:|---:|
| Period-length probe | 19 | 859 s (14 m) | ~25 m |
| PEMS + forecasting unit probes | 10 | 455 s (8 m) | ~15 m |
| Imputation ablation (screen + 4-mask sweep) | 30 | 2,977 s (50 m) | ~1 h 10 m |
| Classification ablation (incl. segpool screen) | 48 | 797 s (13 m) | ~1 h 35 m |
| **Total** | **107** | **5,089 s (1 h 25 m)** | **~3 h 25 m** |

Wall clock exceeds training time by ~2.4× overall, and by ~7× for classification. The reason
is that these models are extremely small, so the fixed costs dominate: interpreter startup,
`.ts` parsing, and — for classification — a full test-set evaluation on every epoch, since
`exp/exp_classification.py` uses the test split as its validation loader.

Cost per useful result is therefore very uneven, which matters for planning:

| Slice | Runs | Wall clock |
|---|---:|---:|
| ETTh1 imputation (all variants × 4 mask rates) | 12 | ~13 m |
| ETTm1 imputation (all variants × 4 mask rates) | 18 | ~57 m |
| PEMS08 (all period lengths + unit variants) | 10 | ~10 m |
| FaceDetection classification (4 variants) | 4 | ~45 m |
| SpokenArabicDigits classification (4 variants) | 4 | ~25 m |

The slowest single configuration is ETTm1 imputation at 190 s of training; the two large UEA
datasets account for roughly three quarters of all classification wall clock.

---

## 5. Next probable steps

Ordered by value per hour of compute.

1. **Complete the imputation table (~1.5 h, 24 runs).** ETTh2, ETTm2, Weather and Electricity
   are still unmeasured for SparseTSFPlus. This is the one table that is unambiguously a win,
   and the study's headline for it currently rests on two of six datasets. Electricity
   (321 channels) will dominate the cost.

   ```bash
   venv/Scripts/python.exe benchmarks/run_sparsetsf_plus_imputation.py --full
   ```

2. **Fold these results into the top-level narrative.** `results_final_report.md` and the root
   `README.md` still describe a three-model study. The imputation result changes the headline
   claim from "matches TimeMixer's accuracy at 1/1800th the parameters on forecasting" to
   something materially stronger.

3. **Extend PEMS beyond PEMS08 (~25 min, 3 runs).** PEMS03/04/07 with the selected
   configuration would allow a 4-dataset average directly comparable to the published
   references, replacing the current single-dataset trend argument.

4. **Test the low-rank phase mixer on small datasets (~15 min).** `--stsf_phase_rank 4` cuts
   the unit from `p²` to `2pr` (577 → 193 at `p=24`). Phase mixing pays off on larger
   datasets, and the low-rank form is the untested variant aimed at small ones.

5. **Re-run the carbon sweep for the new model.** Any energy claim requires
   `run_carbon_benchmarks.py` under its matched fixed-5-epoch protocol; carbon and accuracy
   numbers are not comparable across protocols. The refinement pass costs a second forward
   pass at inference with no extra parameters, so its energy profile is worth measuring rather
   than assuming.

6. **Investigate Weather.** It is the one long-term dataset SparseTSF loses at both lookbacks
   (+20.3% vs TimeMixer at pred=96 after phase mixing), and it is period-insensitive, so the
   cause lies outside the current unit set. Its true daily cycle is 144 steps against a
   96-step lookback, which no period-based mechanism can recover — a longer lookback is the
   more promising direction.

7. **Anomaly detection with the leave-one-out unit (~5 min, 2 runs).** Implemented and
   verified but never benchmarked. SMD is recall-limited (73.90 against 87.25 precision),
   which is the signature of a reconstruction model that reproduces anomalies faithfully;
   the unit removes the identity map from the hypothesis class. Untested, so no claim is made.

---

## Reproducing everything here

From `TimeMixer/`, with the in-repo venv. All runners are idempotent — a configuration whose
log already holds a valid metric is skipped, so re-launching after an interruption is safe.

```bash
venv/Scripts/python.exe benchmarks/tools/assert_sparsetsf_plus_equiv.py
venv/Scripts/python.exe benchmarks/run_period_sensitivity.py
venv/Scripts/python.exe benchmarks/run_sparsetsf_plus_imputation.py --full
venv/Scripts/python.exe benchmarks/run_sparsetsf_plus_classification.py
```

Every table in the linked reports is regenerated from the raw per-run logs in
`logs/run_logs_period_probe/`, `logs/run_logs_plus_imp/`, `logs/run_logs_plus_probe/` and
`logs/run_logs_plus_cls/`, between `<!-- AUTO_* -->` markers. Add `--update-from-logs-only`
to rebuild a table without training.
