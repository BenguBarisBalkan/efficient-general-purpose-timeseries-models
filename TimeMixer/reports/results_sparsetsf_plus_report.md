# SparseTSFPlus — design rationale, units and outcomes

Consolidated write-up for the fourth model in this study: **SparseTSFPlus**, SparseTSF extended
with cheap, individually-toggleable *units* borrowed from TimeMixer and TimeMixer++.

**Headline.** On imputation, SparseTSFPlus reaches TimeMixer's accuracy on ETTh1 while adding
**zero parameters**, and beats it on ETTm1 with 597 parameters against ~75,500. On PEMS
multivariate short-term it removes 24% of the MAE for ~70 extra parameters. On classification no
variant improved the average, which is reported here as a finding rather than omitted.

Supporting tables: [period sensitivity](results_period_sensitivity.md) ·
[imputation ablation](results_sparsetsf_plus_ablation.md) ·
[classification ablation](results_sparsetsf_plus_classification.md) ·
[8-task suite](results_sparsetsf_all_tasks.md)

---

## 1. Starting point: SparseTSF across the full 8-task suite

The TimeMixer++ paper evaluates eight task categories. SparseTSF was extended to all of them in
the previous phase of this study, giving the baseline picture below. This table is what the choice
of target tasks was made from.

| # | Task | Metric | SparseTSF | TimeMixer | TimeMixer++ | Gap (SparseTSF vs best) | Baselines |
|---|---|---|---:|---:|---:|---:|---|
| 1 | Long-term forecasting (sl=96, 5 datasets × 4 horizons) | mean MSE ↓ | 0.354 | **0.348** | 0.449 † | **+1.8%** | **local, all three** |
| 2 | Univariate short-term (M4, 6 subsets) | OWA ↓ | 0.972 | 0.840 | **0.821** | **+18.4%** | published |
| 3 | Multivariate short-term (PEMS03/04/07/08) | MAE ↓ | 34.13 | 17.41 | **15.91** | **+114.5%** | published |
| 4 | Imputation (6 datasets × 4 mask rates) | masked MSE ↓ | 0.0838 § | **0.0586** § | — | **+43.0%** | **local (TimeMixer)** |
| 5 | Few-shot, 10% train (ETT avg) | MSE ↓ | 0.441 | 0.453 | **0.396** | **+11.4%** | published |
| 6 | Zero-shot transfer (6 ETT pairs) | mean MSE ↓ | **0.393** | 0.467 | — | **wins −15.8%** | published |
| 7 | Anomaly detection (PSM, SMD) | F1 ↑ | **87.64** | — | 87.47 | comparable ‡ | published |
| 8 | Classification (10 UEA) | accuracy ↑ | 60.90 | — | **75.9** | **−15.0 pts** | published |

† **TimeMixer++ is undertrained by design.** It requires `lr=5e-5` for NaN stability, so its
*accuracy* column understates the architecture. Its *energy* numbers are fair. Do not read the
TimeMixer++ column as "TimeMixer++ is worse".

‡ Not comparable: SparseTSF covers 2 of 5 anomaly datasets (MSL/SMAP/SWaT are unobtainable), so
the F1 averages are over different dataset sets.

§ The two imputation means are over **different dataset counts** — SparseTSF over 6, TimeMixer over
5 (Electricity was not run for TimeMixer). The per-dataset comparison used in §5 is like-for-like
on ETTh1 and ETTm1, and TimeMixer wins all 20 of those cells, so the direction of the gap is not in
doubt; only the exact headline percentage is affected.

**Which baselines are local.** Long-term forecasting has full local sweeps for **both** TimeMixer
(32 runs) and TimeMixer++ (22 runs); imputation has a local TimeMixer sweep (20 runs); and the
energy study ran all three models under a matched protocol (40 TimeMixer/TimeMixer++ runs) —
**114 local TimeMixer/TimeMixer++ runs in total**, all on the same RTX 3050 Ti (4 GB). The other
five per-task benchmarks (M4, PEMS, few-shot, zero-shot, anomaly, classification) were run locally
for SparseTSF only, against *published* TimeMixer/TimeMixer++ figures produced with the authors'
tuning on their hardware. Those five are reference points, not controlled head-to-heads.

### What this table says

SparseTSF is **not** uniformly behind. It is essentially at parity on long-term forecasting
(+1.8%, and it *wins* 7 of 20 configurations, sweeping the hourly ETTh datasets), and at its native
`seq_len=720` its mean MSE of 0.312 **beats** TimeMixer's 0.348 outright. It also wins zero-shot
transfer decisively and is competitive few-shot. The deficits are concentrated and structural.

---

## 2. Why PEMS, classification and imputation

Ranked by gap, the candidates were PEMS (+114.5%), classification (−15.0 points), imputation
(+43.0%), M4 (+18.4%) and few-shot (+11.4%). The three chosen were selected on three criteria:
**size of the gap**, **whether the cause was diagnosable**, and **whether a result there would be
defensible**.

**PEMS — chosen because it is the largest gap and the cause was visible in the configuration.**
More than doubling TimeMixer's error is the single biggest deficit in the study. Two causes were
identifiable. The first is cross-channel correlation: PEMS is 170–883 traffic sensors on a road
network, and TimeMixer's own paper enables channel mixing for it. The second is far simpler — at
`seq_len=96, pred_len=12, period_len=12`, SparseTSF's forecast head is a single `8 → 1` linear map,
i.e. **eight parameters** for the whole model, predicting all twelve future steps as one 8-tap
combination per phase. The second cause is addressable without touching channel independence, which
made PEMS worth attacking even with parity off the table.

**Imputation — chosen because it has a local baseline and a mechanically diagnosable cause.**
At +43.0% it is the largest gap among tasks with a *same-machine* TimeMixer sweep, and TimeMixer
wins all 20 cells there, so the deficit is not noise. More importantly the gap **grows
monotonically with mask rate** (+17% at mask 0.125 rising to +46% at mask 0.5 on ETTh1), which
points directly at the mechanism: SparseTSF's reconstruction is one bias-free linear over periods
at a fixed phase, fed by a convolution whose receptive field is half literal zeros at mask 0.5 —
and nothing in the model distinguishes a masked zero from an observed one. A monotone trend in a
controlled comparison is the most tractable kind of gap.

**Classification — chosen because it is the one task where SparseTSF is not small.**
The −15.0 point deficit is the largest in its own metric, and the readout is
`Linear(enc_in × seq_len, num_class)`, which on PEMS-SF is **970,712 parameters** — three orders of
magnitude above the model's headline size, and the only place the efficiency claim breaks. A
successful fix would have improved accuracy *and* restored the claim. Handwriting sitting at 8.59%
against a ~3.8% chance floor for 26 classes suggested the head was closer to broken than untuned.

**What was not chosen, and why.** Long-term forecasting was already at parity and wins at the
native lookback, so there was little to gain. Zero-shot transfer SparseTSF already wins 6/6.
Few-shot beats TimeMixer and trails only TimeMixer++, and its worst cases look like undertraining
on a truncated training split rather than an architectural deficit. M4's gap is real (+18.4%) but
its cause — short, heterogeneous, largely aperiodic series starving a period-based model — is
intrinsic to the design rather than a fixable omission. Anomaly detection could only be measured on
2 of 5 datasets, too weak a basis for a claim.

---

## 3. Design constraints, and the reasoning behind them

Three decisions shaped every unit that follows.

**A parameter budget of ≤2k.** SparseTSF's entire contribution is that it achieves competitive
forecasting accuracy with 41 parameters at `seq_len=96` and 925 at `seq_len=720`. A fix costing
75,000 parameters would answer a different research question — it would simply be a small
TimeMixer. Holding the budget near SparseTSF's own published headline keeps the comparison
meaningful. Every shipped configuration came in under ~700 parameters, and the fallback to 10k was
never needed.

**No cross-channel mixing.** This was the hardest trade. Cross-channel coupling is the only route
to PEMS parity, but strict channel independence buys three things: the parameter count does not grow
with channel count (the same weights serve 7 channels or 883), checkpoints transfer between datasets
with different channel counts — which is *why* zero-shot transfer works at all, and that is
SparseTSF's best task — and the energy profile stays flat in channel count. Spending all of that to
improve one task was judged a bad exchange, so PEMS parity was explicitly **not** a goal and the
residual gap is reported as a documented limitation rather than a failure.

**All-units-off must equal SparseTSF exactly.** Every unit defaults to off and residual units are
zero-initialised, so with everything disabled the model is *numerically identical* to SparseTSF —
verified as bitwise equality on all five task types, and confirmed end-to-end through the full
training loop by the ablation's control row reproducing published cells to six decimals. Three
things follow: no previously published number in this study is invalidated; the ablation has a
genuine baseline rather than a re-implementation that merely ought to match; and a unit can never be
credited with a gain that actually came from an incidental change elsewhere. This property is what
makes the rest of the report interpretable.

---

## 4. The units

Each unit targets one diagnosed mechanism. Parameter costs are exact, with `S`=seq_len,
`H`=pred_len, `p`=period_len, `C`=channels, `K`=classes.

### Phase-axis mixing — the flagship, and the one idea with theory behind it

*Borrowed from: TimeMixer++'s dual-axis `TimeImageDecomposition`, reduced from attention to a
shared linear map.* **Cost: `p²`.**

SparseTSF folds each channel's window into a `period × period-count` grid and applies **one**
linear map across periods, independently at each within-period phase. Written out, its per-channel
forecast map is exactly a Kronecker product with an identity factor:

> `I_p ⊗ W_n` — one `n_x → n_y` matrix reused at every one of the `p` phases.

The consequence is that **within-period shape is never modelled**. Only the period-to-period
evolution at each fixed phase is. Adding a second linear map along the *phase* axis turns this into

> `W_p ⊗ W_n` — a Kronecker-**factorised** full `S → H` linear map.

This is a strict generalisation of SparseTSF, and the parameter accounting is the interesting part:
a dense `S → H` map costs `S·H` (9,216 for ETTh1 96→96), while the factorised form costs
`p² + n_x·n_y` (592). Crucially it is **`seq_len`-free** — the same `p²` at `seq_len=96` and
`seq_len=720` — which is precisely the architectural lesson of TimeMixer++, whose mixer stack is
`O(d_model²)` and stays at 23k parameters at both lookbacks while TimeMixer's grows quadratically
to 4.05M.

The empirical test of the theory: `period_len=1` collapses the fold entirely and *is* the dense
limit. On PEMS08 the factorised form at 93 parameters matched the 1,153-parameter dense model to
within 0.5% — a 12× saving for the same accuracy.

### Iterative refinement — parameter-free, and the largest single win

*Derived from the loss definition rather than borrowed.* **Cost: 0.**

The imputation objective is scored on masked positions only. The model therefore never receives
gradient on the positions it *can* see, and at mask 0.5 its aggregating convolution reads a window
that is half zeros. The unit exploits this asymmetry: a first pass produces an estimate, the holes
are filled with it, the normalisation statistics are recomputed on the now-complete series, and the
**same weights** refine the result. No new parameters, one extra forward pass at inference.

This turned out to be the strongest unit in the study, and it is the one that costs nothing.

### Mask-aware aggregation

*Partial-convolution renormalisation.* **Cost: 0.**

Rescales the convolution output by how much of each window was actually observed, so a masked zero
no longer drags the aggregate toward zero. Complements refinement rather than replacing it: on
ETTm1 the two together are better than either alone.

### Multi-period ensembling

*Borrowed from: TimeMixer's multi-scale sum-ensemble, applied to periods instead of resolutions.*
**Cost: one small matrix per extra period.**

Runs the fold at several period lengths simultaneously and sums the branches behind
zero-initialised gates, so enabling it starts from exactly the base model and training decides
whether the extra periods help. Motivated by the observation that a single hand-chosen `period_len`
is a strong assumption about which cycle dominates.

### Trend/season decomposition

*Borrowed from: Autoformer's `series_decomp`, already vendored in this repository.* **Cost: 0.**

Splits the signal with a moving average, routes the seasonal part through the period machinery and
lets the trend persist its final value. The zero-parameter version of TimeMixer's central idea.

### Leave-one-out reconstruction

*Diagnosis-driven.* **Cost: 0.**

For reconstruction tasks, the linear map plus the convolution's residual connection makes the
**identity** exactly representable. Training a reconstruction objective on mostly-normal data
therefore drifts toward it, and a model that reproduces its input faithfully reconstructs anomalies
faithfully too — which is the signature of SMD's recall-limited failure (recall 73.90 against
precision 87.25). Removing the self-term forces a genuine leave-one-out prediction.

### Phase-aligned folding

*Borrowed from: TimeMixer++'s pad-and-fold, corrected for forecast phase alignment.* **Cost: 0.**

Lifts SparseTSF's requirement that `period_len` divide both `seq_len` and `pred_len`, which
otherwise restricts the searchable periods to divisors. Padding is applied so that forecast step
`h` still lands in phase `h mod p`, preserving the input/output phase correspondence the
architecture depends on.

### Pooled classification readouts

*Statistics pooling and segment pooling.* **Cost: `3CK` / `n_seg·CK`.**

Replace the flatten-linear readout, whose size scales with `seq_len` and which has no translation
invariance, with per-channel summaries over time — 18× to 513× smaller. Two variants were tested:
global statistics (mean/std/max), and segment-wise means that retain coarse temporal ordering.

### What the units cost in practice

| Configuration | Parameters | Comparable TimeMixer |
|---|---:|---:|
| PEMS08, phase mixing | **93** | dense equivalent 1,153 |
| ETTh1 imputation, refinement | **41** — zero added | ~75,497 |
| ETTm1 imputation, all units | **597** | ~75,497 |
| ETTh1 forecasting, phase mixing | 617 | 75,497 |
| ETTh1 `sl=720`, phase mixing | 1,501 | 4,046,633 |

### One unit that provably cannot work

RevIN — per-window mean *and* variance normalisation, standard in modern forecasting — was
implemented and then found to be a **mathematical no-op** on this architecture. SparseTSF's forecast
map is linear and entirely bias-free, hence positively homogeneous: `f(x/σ)·σ = f(x)`. Dividing by
a per-window, per-channel scalar and multiplying it back therefore cancels *exactly*, which
measurement confirmed (difference of 1.5 × 10⁻⁵ on inputs scaled to ~50, i.e. floating-point
rounding). It became effective only when a bias or a last-value offset broke the homogeneity. This
is a property of SparseTSF worth stating: **any scale-equivariant normalisation is inert on it**.

---

## 5. How the results changed

### Imputation — the gap is eliminated

Mean masked MSE over four mask rates, variant selected on **validation** loss. Both baselines were
run on this machine under this protocol.

| | Selected units | Parameters | Mean MSE | vs SparseTSF | vs TimeMixer |
|---|---|---:|---:|---:|---:|
| **ETTh1** | refinement only | **41** — zero added | **0.120684** | −26.1% | **+0.5%** — parity, 1,841× smaller |
| **ETTm1** | all units | **597** | **0.042010** | −41.2% | **−10.7%** — win, 126× smaller |

Per mask rate:

| mask | ETTh1 SparseTSF | **ETTh1 Plus** | ETTh1 TimeMixer | ETTm1 SparseTSF | **ETTm1 Plus** | ETTm1 TimeMixer |
|---:|---:|---:|---:|---:|---:|---:|
| 0.125 | 0.113335 | **0.089294** | 0.096742 | 0.048898 | **0.034291** | 0.039071 |
| 0.25 | 0.146550 | **0.105566** | 0.111707 | 0.062403 | **0.036845** | 0.042218 |
| 0.375 | 0.179265 | **0.128675** | 0.125347 | 0.078078 | **0.042312** | 0.048693 |
| 0.5 | 0.214076 | **0.159201** | 0.146432 | 0.096544 | **0.054591** | 0.058143 |
| **mean** | 0.163307 | **0.120684** | 0.120057 | 0.071481 | **0.042010** | 0.047031 |

The gap moves from **+36.0% → +0.5%** on ETTh1 and **+52.0% → −10.7%** on ETTm1. ETTh1 beats
TimeMixer outright at the two lower mask rates.

Two observations worth discussing:

- **The parameter-free unit does most of the work.** On ETTh1 refinement alone is the best
  configuration, so the improved model is byte-identical in size to stock SparseTSF.
- **Unit composition depends on training-set size.** ETTm1 (~34k windows) benefits from every unit
  and they compose; ETTh1 (~8.5k windows) is best with the free unit alone, and adding capacity
  hurts. The ordering is stable across all four mask rates independently, so this is a real
  data-size effect and not an artefact — and it means no single universal configuration is optimal.

Validation ranked all 24 cells in exactly the test order, so the unit selection does not depend on
the test signal.

### PEMS multivariate short-term — 24% of the MAE removed, channel-independently

PEMS08, `seq_len=96 → pred_len=12`:

| Configuration | Parameters | MAE | MAPE (%) | RMSE |
|---|---:|---:|---:|---:|
| published (`period_len=12`) | 21 | 29.25 | 17.14 | 44.61 |
| **`p=4` + phase mixing** | **93** | **22.22** | 12.85 | 35.26 |
| **`p=12` + phase mixing** | **165** | **22.21** | 12.85 | 35.24 |
| dense (`period_len=1`) | 1,153 | 22.11 | 12.78 | 35.15 |

MAE falls **29.25 → 22.22 (−24.0%)**, closing roughly 59% of the distance to TimeMixer's published
17.41, for about 70 extra parameters. The residual is the cross-channel component that was
deliberately out of scope.

A finding worth raising in discussion: that `period_len=1` — which removes the period fold
altogether — performs *best* means SparseTSF's cross-period sparse structure is **actively harmful**
on PEMS. That is a sharper explanation of the PEMS failure than channel independence alone, and it
suggests the inductive bias, not just capacity, is mismatched to traffic data.

Read the trend rather than the absolute gap here: this is PEMS08 alone against 4-dataset published
averages.

### Long-term forecasting — a free improvement from period selection

`period_len` chosen on validation loss rather than taken from the original paper:

| Dataset | published `p` | selected `p` | MSE before | MSE after | TimeMixer (local) |
|---|---:|---:|---:|---:|---:|
| **ETTh1** | 24 | **12** | 0.396635 | **0.383847** (−3.22%) | 0.385794 |
| Weather | 4 | 4 | 0.198635 | unchanged | 0.164526 |
| ETTm1 | 4 | 4 | 0.362121 | unchanged | 0.326398 |

ETTh1 flips from a **+2.81% loss to a −0.51% win** over TimeMixer for 36 extra parameters and no
architectural change. Weather and ETTm1 are period-*insensitive*: both degrade monotonically as the
period grows, and the published value is already optimal on both validation and test. With phase
mixing added, ETTm1 improves a further −1.99% and Weather −0.35%.

### Classification — no improvement

| Variant | 10-dataset average |
|---|---:|
| flatten-linear readout (control) | **60.90** |
| + mask-aware centring | 59.17 |
| global statistics pooling | 36.69 |
| statistics pooling + wider aggregator | 39.30 |
| segment pooling | screened on 4 datasets |

Pooled readouts are 18–513× smaller but lose accuracy on shape- and timing-driven datasets:
UWaveGestureLibrary falls 72.81 → 9.38 and Handwriting 8.59 → 3.41 under global pooling. Three
statistics per channel cannot encode a pen trajectory, and retaining coarse temporal order through
segment pooling does not recover it either.

**One genuine, dataset-specific win:** Heartbeat **54.63 → 71.71 (+17.1 points)** with statistics
pooling, at **369 parameters against the flatten head's 49,413** — 134× smaller and substantially
more accurate. With 61 channels × 405 steps against 204 training samples, the flatten readout carries
a 242:1 parameter-to-sample ratio and simply overfits. The pattern across the ten datasets is that
pooling helps where a task is amplitude- or statistics-driven and hurts where it is shape-driven.

Classification remains ~15 points behind TimeMixer++'s published 75.9, and this work did not close
that gap.

---

## 6. Runtime

107 training runs for this phase. Training time is measured exactly from the per-epoch timings in
each log; wall clock is as observed.

| Experiment group | Runs | Training time | Wall clock |
|---|---:|---:|---:|
| Period-length diagnostics | 19 | 859 s (14 m) | ~25 m |
| PEMS and forecasting unit probes | 10 | 455 s (8 m) | ~15 m |
| Imputation ablation (screen + 4-mask sweep) | 30 | 2,977 s (50 m) | ~1 h 10 m |
| Classification ablation | 48 | 797 s (13 m) | ~1 h 35 m |
| **Total** | **107** | **5,089 s (1 h 25 m)** | **~3 h 25 m** |

Wall clock exceeds training time by ~2.4× overall and ~7× for classification. The reason is
informative for planning: these models are so small that fixed costs dominate — interpreter
startup, dataset parsing, and for classification a full test-set evaluation on every epoch. Cost per
result is therefore very uneven: all four mask rates of ETTh1 imputation take ~13 minutes for 12
runs, while four classification variants on FaceDetection alone take ~45 minutes.

For context, the model itself trains in seconds. The binding constraint on this study is data
loading and evaluation overhead, not the models under test.

---

## 7. Next steps

Ordered by value per hour of compute.

1. **Complete the imputation table (~1.5 h, 24 runs).** ETTh2, ETTm2, Weather and Electricity are
   still unmeasured for SparseTSFPlus. This is the one unambiguous win and its headline currently
   rests on two of six datasets. Electricity (321 channels) will dominate the cost.

2. **Fold these results into the top-level narrative.** `results_final_report.md` and the root
   `README.md` still describe a three-model study. The imputation result strengthens the headline
   claim materially: not merely competitive accuracy at a thousandth of the parameters, but
   *better* accuracy than a model 126× larger on a task where SparseTSF was previously 52% behind.

3. **Extend PEMS beyond PEMS08 (~25 min, 3 runs).** PEMS03/04/07 under the selected configuration
   would give a 4-dataset average directly comparable to the published references, replacing the
   present single-dataset trend argument.

4. **Test the low-rank phase mixer (~15 min).** Constraining the phase map to low rank cuts the
   unit from `p²` to `2pr`. Phase mixing pays off on larger datasets and overfits small ones, so
   the low-rank form is the untested variant aimed squarely at the small-data regime — and would
   test whether the ETTh1 regression is capacity or optimisation.

5. **Re-run the energy sweep for the new model.** Any energy claim requires the matched
   fixed-epoch carbon protocol; carbon and accuracy numbers are not comparable across protocols.
   The refinement unit is interesting here because it costs a second forward pass at inference while
   adding no parameters — so it trades a little energy for accuracy, which is exactly the axis this
   study measures, and its cost should be quantified rather than assumed negligible.

6. **Investigate Weather.** The one long-term dataset SparseTSF loses at both lookbacks (+20.3% vs
   TimeMixer at `pred=96` after phase mixing), and it is period-insensitive, so the cause lies
   outside the current unit set. Its true daily cycle is 144 steps against a 96-step lookback, which
   no period-based mechanism can recover — a longer lookback is the more promising direction.

7. **Benchmark the leave-one-out unit on anomaly detection (~5 min, 2 runs).** Implemented and
   verified but never measured. SMD is recall-limited, the signature the unit was designed for.
   Untested, so no claim is made.

---

## Reproducing this

All runners are idempotent — a configuration whose log already holds a valid metric is skipped — and
every table is regenerated from the raw per-run logs rather than maintained by hand. The equivalence
check, the period diagnostics, and the two ablations are each a single command from `TimeMixer/`;
see `CLAUDE.md` for the exact invocations and the verification recipe.
