# SparseTSFPlus — design rationale, units and outcomes

Consolidated write-up for the fourth model in this study: **SparseTSFPlus**, SparseTSF extended
with cheap, individually-toggleable *units* borrowed from TimeMixer and TimeMixer++.

**Headline.** Run as **one fixed configuration** on all eight tasks (Section 6), SparseTSFPlus
improves on or matches SparseTSF on every task, with the largest gains on PEMS (−25.4%),
imputation (−17.8%) and M4 (−6.0%). It gives up almost nothing against choosing units per task:
within 0.4% everywhere except imputation (+5.8%, all of it from phase mixing). It beats the locally
run TimeMixer on few-shot, zero-shot and anomaly detection with 37 to 8,661 parameters, and still
trails it on long-term forecasting, M4, PEMS and imputation.

On imputation, a learned gate on the refinement step (Section 4) made one configuration almost as
good as choosing per dataset (0.0651 against 0.0646). That one configuration beats TimeMixer on four
of the six datasets and on the mean of the five other than Electricity (0.0572 against 0.0586), with
42 to 582 parameters against TimeMixer's ~58,000–60,000; it stays 30% behind on Electricity. Chosen
per dataset, SparseTSFPlus cuts SparseTSF's imputation error by 22.8% and closes 88% of its gap to
TimeMixer. On PEMS multivariate short-term it removes 25% of the MAE on all four datasets for ~70
extra parameters. Forecasting gains are small (about 1%), and on classification no variant
improved the average — both reported as findings rather than omitted. A gate on phase mixing did
not work, and phase mixing in imputation at high mask rates is the main open problem.

Supporting tables: [period sensitivity](results_period_sensitivity.md) ·
[imputation ablation](results_sparsetsf_plus_ablation.md) ·
[forecasting ablation](results_sparsetsf_plus_forecast.md) ·
[anomaly ablation](results_sparsetsf_plus_anomaly.md) ·
[classification ablation](results_sparsetsf_plus_classification.md) ·
[**one model on all 8 tasks**](results_sparsetsf_plus_suite.md) ·
[8-task suite](results_sparsetsf_all_tasks.md)

---

## 1. Starting point: SparseTSF across the full 8-task suite

The TimeMixer++ paper evaluates eight task categories. SparseTSF was extended to all of them in
the previous phase of this study, giving the baseline picture below. The choice of target tasks was
made from the published TimeMixer figures; TimeMixer was later run locally on every task
(2026-10-12), and those same-machine figures are what the table now shows. They confirm the published
picture. Anomaly detection, extended to four datasets on 2026-10-21, is mixed: TimeMixer leads on
three of the four, SparseTSF on average.

| # | Task | Metric | SparseTSF | TimeMixer (local) | TimeMixer++ (published) | Gap (SparseTSF vs best comparable) |
|---|---|---|---:|---:|---:|---:|
| 1 | Long-term forecasting (sl=96, 5 datasets × 4 horizons) | mean MSE ↓ | 0.354 | 0.348 | **0.324** † | **+9.1%** |
| 2 | Univariate short-term (M4, 6 subsets) | OWA ↓ | 0.972 | 0.841 | **0.821** | **+18.4%** |
| 3 | Multivariate short-term (PEMS03/04/07/08) | MAE ↓ | 34.13 | 17.51 | **15.91** | **+114.5%** |
| 4 | Imputation (6 datasets × 4 mask rates) | masked MSE ↓ | 0.0838 | **0.0622** | — | **+34.7%** |
| 5 | Few-shot, 10% train (ETT avg) | MSE ↓ | 0.441 | 0.513 § | **0.396** | **+11.4%** |
| 6 | Zero-shot transfer (6 ETT pairs) | mean MSE ↓ | 0.393 | 0.448 | **0.386** | **+1.9%** |
| 7 | Anomaly detection (PSM, SMD, MSL, SMAP) | F1 ↑ | 79.96 | 75.49 ¶ | **85.75** ‡ | **−5.8 pts** |
| 8 | Classification (10 UEA) | accuracy ↑ | 60.90 | 62.37 ¶ | **75.9** | **−15.0 pts** |

† **TimeMixer++ was also run locally for forecasting (mean 0.449), but it is undertrained by
design:** it requires `lr=5e-5` for NaN stability, so its local accuracy understates the
architecture. Its published figure (0.324) is therefore used as the fairer reference here. Its
local *energy* numbers are fair. Against the locally run TimeMixer the forecasting gap is +1.8%.

‡ TimeMixer++'s published per-dataset F1 averaged over the same four datasets. Its 5-dataset
average (87.47) includes SWaT, which is gated. MSL and SMAP are rebuilt from a mirror of NASA's
release at exactly the published sizes, with anomaly rates slightly below the cited ones. The local
TimeMixer wins on PSM, SMD and SMAP; SparseTSF's higher average comes entirely from MSL, where the
untuned TimeMixer collapses (F1 48.0 against 77.8).

§ The one task where the local TimeMixer falls clearly short of its published figure (0.453). The
shared few-shot protocol trains at batch 32, which likely does not suit TimeMixer's full-data
learning rate.

¶ TimeMixer publishes no anomaly or classification settings, so these are untuned local runs with
its own small long-term architecture.

**Which baselines are local.** TimeMixer is now run locally on all eight tasks. Its settings come
from its upstream scripts where they exist and from its own long-term configuration elsewhere.
Where a published TimeMixer figure exists, the local one matches it closely: M4 OWA 0.841 vs 0.840,
PEMS MAE 17.51 vs 17.41, zero-shot 0.448 vs 0.467. TimeMixer++ is run locally only for long-term
forecasting (21 of 32 configurations) and the energy study, because it is stable only at a learning
rate that undertrains it. Its accuracy reference is the published one. All runs are on the same
RTX 3050 Ti (4 GB).

### What this table says

SparseTSF is **not** uniformly behind. On long-term forecasting it is within 1.8% of the locally
run TimeMixer and wins 7 of 20 configurations, sweeping the hourly ETTh datasets; at its native
`seq_len=720` its mean MSE of 0.312 beats both baselines' 96-step figures. On zero-shot transfer it
beats the local TimeMixer on all six pairs (0.393 vs 0.448) and is within 1.9% of the published
TimeMixer++, and it is competitive few-shot. The deficits are concentrated and structural.

---

## 2. Why PEMS, classification and imputation

Ranked by gap, the candidates were PEMS (+114.5%), classification (−15.0 points), imputation
(+34.7%), M4 (+18.4%) and few-shot (+11.4%). The three chosen were selected on three criteria:
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
At +34.7% it was the largest gap among tasks with a *same-machine* TimeMixer sweep when the targets
were chosen, and TimeMixer wins all 24 cells there, so the deficit is not noise. More importantly the gap **grows
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

**What was not chosen, and why.** Long-term forecasting was within 1.8% of the local TimeMixer
and ahead of both baselines at the native lookback, so there was little to gain. On zero-shot
transfer SparseTSF beats TimeMixer on all 6 pairs and is within 1.9% of TimeMixer++.
Few-shot beats TimeMixer and trails only TimeMixer++, and its worst cases look like undertraining
on a truncated training split rather than an architectural deficit. M4's gap is real (+18.4%) but
its cause — short, heterogeneous, largely aperiodic series starving a period-based model — is
intrinsic to the design rather than a fixable omission. Anomaly detection could only be measured on
2 of 5 datasets at the time (4 of 5 since 2026-10-21), too weak a basis for a claim.

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
limit. On all four PEMS datasets the factorised form at 93 parameters lands within 0.6% of the
1,153-parameter dense model — a 12× saving for the same accuracy.

Its weakness is small datasets: the full `p × p` map overfits ETTh1 (+2.7%, worse on validation
too). A rank-4 factorisation removes most of that overfit (ETTh1 0.394 → 0.387) but still does not
beat the plain model there, and at `p = 4` it is pointless, because rank 4 then costs more numbers
(2pr = 32) than the full map (p² = 16).

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

### Gates — a safe default for refinement and phase mixing

*Borrowed from: ReZero (Bachlechner et al., UAI 2021), the same zero-initialised scalar this model
already uses for its multi-period branches.* **Cost: 1 per gated unit.**

**The problem.** Run as one fixed configuration (Section 6), SparseTSFPlus lost to a tuned one in
two places: refinement on ETTh2 imputation and phase mixing on Electricity imputation. In both, the
unit also raised the *training* loss, so the harm was visible to training rather than being
overfitting. On ETTh2 at mask 0.125, refinement's training loss was 0.114 against 0.058 without it;
on Electricity at mask 0.25, phase mixing's was 0.129 against 0.095.

**The idea.** Multiply the unit's contribution by one learned number that starts at 0. A gated unit
then starts out as the model without it, and training decides how much of the unit to use,
including none.
- **Refinement:** the output becomes pass 1 + g·(pass 2 − pass 1). Without the gate, pass 2
  *replaces* pass 1, so once the unit is on, "no refinement" cannot even be represented.
- **Phase mixing:** the phase image becomes img + g·mix(img). The mixer starts at its default
  initialisation, since a zero gate in front of zero weights would never receive gradient.

Before any run it was checked that each gate at 0 gives exactly (bitwise) the output of the model
without that unit. Both gated forms ran on all six imputation datasets at all four mask rates
([imputation ablation](results_sparsetsf_plus_ablation.md), variants `u5g`, `allg` and `allrg`).

**Refinement gate: works, better than predicted.**
- **The prediction was wrong in an instructive way.** It was that the gate would switch refinement
  off on ETTh2. Instead it settled at about 0.5, averaging the two passes. That beat both no
  refinement (0.0641 against 0.0703) and full refinement (0.0955), at every mask rate.
- **It is better on all six datasets**, not only ETTh2. Gated refinement with mask-aware
  aggregation averages 0.0651 against 0.0716 ungated (−9.1%), and it is also lowest on validation.
- **The learned values are readable.** They rise with the mask rate on five of six datasets: the
  more holes, the more weight goes to the refinement pass. They are lowest on the two hourly ETT
  datasets (0.48–0.70) and near 1 elsewhere (mostly 0.8–1.07).

**Phase-mixing gate: does not work.**
- **Where phase mixing helps, it keeps helping.** Under gated refinement, phase mixing helps or is
  neutral at masks 0.125 and 0.25 on every dataset (−4.1% to +2.5%).
- **The harm is only at high mask rates, and the gate does not remove it.** At masks 0.375 and 0.5
  phase mixing costs +31% to +44% on ETTh2 and +16% to +26% on Electricity, with or without the
  gate. There the gate moves well away from 0, and training ends at the same worse result.
- **On average the gate changes nothing:** 0.0688 with it against 0.0689 without. So the harm is
  not something one scalar can switch off; training moves into it.

**What follows.** Gated refinement replaces refinement in both versions of the model. Phase mixing
stays ungated, because PEMS needs it (Section 6), and it is the one remaining source of the
imputation gap. Its harm appearing only when many points are missing suggests a mechanism: mixing
along the phase axis spreads the holes. That is the next thing to test (Section 8).

### Phase mixing on inputs with missing values — a candidate

*Borrowed from: partial convolutions (Liu et al., 2018), the rule mask-aware aggregation already
applies to the time convolution.* **Cost: 0.**

**The problem.** After the refinement gate, phase mixing is the only source of the imputation gap
between the fixed and per-task configurations. It helps or is neutral at mask rates 0.125 and 0.25,
but at 0.375 and 0.5 it costs 31–44% on ETTh2 and 16–26% on Electricity. Harm only when many points
are missing suggests the holes: mixing along the phase axis spreads positions that hold no
observation into the observed hours of each period.

**Two forms were tested** on all six datasets at all four mask rates, on top of the fixed imputation
form (`allrg_pc`, `allrg_pm` in the [imputation ablation](results_sparsetsf_plus_ablation.md)):
- **Complete inputs only:** skip phase mixing in the first pass, which still has the holes, and keep
  it in the refinement pass, whose input has been filled.
- **Mask-aware:** only observed positions feed the mixing, rescaled by the share of the period that
  was observed. With every point observed it equals plain phase mixing exactly (checked).

**Results** (mean masked MSE over six datasets; validation in brackets):

| Without phase mixing | Plain phase mixing (fixed) | Complete inputs only | **Mask-aware** |
|---:|---:|---:|---:|
| 0.0651 (0.0713) | 0.0689 (0.0737) | 0.0728 (0.0781) | **0.0657 (0.0717)** |

- **Mask-aware phase mixing is a candidate.** It brings the fixed imputation form within 1.0% of the
  form without phase mixing, and is better than plain phase mixing on validation. It removes the
  high-mask harm in two of the four bad cells (ETTh2 at 0.5: +31% → −4.8%; Electricity at 0.5:
  +16.5% → −14.1%) and halves it in a third (Electricity at 0.375: +26% → +13%).
- **ETTh2 at mask 0.375 is unchanged** (+47%). That is the dataset whose runs sometimes stay on a
  training plateau, so the seed repeats have to show whether this cell is the unit or chance.
- **Complete inputs only fails.** Skipping phase mixing in the first pass costs 23–42% on
  Electricity and 29–39% on ETTh1 even at low mask rates. The first pass needs phase mixing; what
  helps is mixing only the observed points.

Reported as a candidate beside the fixed configuration, not adopted into it. It changes nothing
outside imputation, because every other task's input is complete.

### Channel context — helps PEMS a little, everywhere else nothing

*Borrowed from: the aggregate-and-redistribute core of SOFTS (Han et al., 2024), and attention across
variates in iTransformer (Liu et al., 2024) with its projections removed.* **Cost: 2–9 parameters.**

**The problem.** PEMS is the largest remaining gap to TimeMixer (+45% MAE). The backbone treats every
channel independently, so it cannot use the fact that neighbouring sensors move together, and
TimeMixer runs PEMS with its channels embedded jointly.

**Two forms, both behind a ReZero gate and with no parameter that depends on the channel count**
(so checkpoints still transfer between datasets):
- **Channel mean:** the mean of all channels in the window gets its own period map, and each channel
  adds it scaled by its own least-squares loading on that mean (computed from the window).
- **Channel correlation:** each channel adds a softmax-weighted mix of the other channels' forecasts,
  weighted by input-window correlation, with a learned temperature.

**Screen** on PEMS03/04/07/08 and, as a guard, ETTh1, ETTm1 and Weather at horizon 96, each run the
fixed configuration's command plus the unit (generated table in
[results_sparsetsf_plus_suite.md](results_sparsetsf_plus_suite.md)):

| | PEMS validation | PEMS test MAE | Long-term checks (test MSE) | Parameters on PEMS |
|---|---:|---:|---|---:|
| fixed | — | 25.45 | — | 173 |
| + channel mean | −3.7% | 24.46 (−3.9%) | −1.5% to +0.3% | 182 |
| + channel correlation | −4.7% | 24.20 (−4.9%) | −0.7% to 0.0% | 175 |

- **Both help PEMS on all four datasets and hurt nothing.** Channel correlation is the stronger:
  −4.3% to −5.5% on every PEMS dataset for 2 parameters.
- **Neither passed the pre-registered rule** for a full suite run (PEMS validation at least 5%
  better); correlation missed at 4.7%. The predicted 10% was too optimistic.
- **The PEMS gap stays large** (+38% against TimeMixer with channel correlation). A channel mean, or
  a correlation-weighted mix of other channels' forecasts, carries only a small part of what
  TimeMixer's joint channel embedding uses.

### Multi-period ensembling

*Borrowed from: TimeMixer's multi-scale sum-ensemble, applied to periods instead of resolutions.*
**Cost: one small matrix per extra period.**

Runs the fold at several period lengths simultaneously and sums the branches behind
zero-initialised gates, so enabling it starts from exactly the base model and training decides
whether the extra periods help. Motivated by the observation that a single hand-chosen `period_len`
is a strong assumption about which cycle dominates.

Measured at H = 96 it is small and inconsistent: −1.8% on ETTm1, +0.7% on ETTh1, +0.5% on Weather.
It never had the lowest validation loss. Kept switched off.

### Trend/season decomposition

*Borrowed from: Autoformer's `series_decomp`, already vendored in this repository.* **Cost: 0
(trend held flat) or n_x·n_y (trend given its own period map).**

Splits the signal with a moving average and routes the seasonal part through the period
machinery. Two treatments of the trend were tested, and the difference between them is the finding:

- **Trend held at its last value: harmful on every dataset** (+13% to +27% MSE at H = 96).
  Decomposition removes the drift from what the period map sees, and a flat trend cannot
  extrapolate it — so this form takes away something SparseTSF could already do.
- **Trend given its own period map: the best forecasting unit.** It had the lowest validation loss
  on all three datasets. Chosen on validation at H = 96 and then run at every horizon, it improves
  ETTm1 at all four horizons (−0.8% to −2.7%) and Weather at all four (−0.2% to −1.0%). On ETTh1 the
  result is mixed (−2.6%, +0.8%, −0.4%, −1.5%), and at H = 96 validation and test disagree there.
  It doubles the period map: 581 → 1,157 parameters for ETTm1 and Weather, 77 → 141 for ETTh1.

Combining it with phase mixing works without problems but adds nothing: on validation the
combination never beat the learned trend alone.

### Leave-one-out reconstruction

*Diagnosis-driven.* **Cost: 0.**

For reconstruction tasks, the linear map plus the convolution's residual connection makes the
**identity** exactly representable. Training a reconstruction objective on mostly-normal data
therefore drifts toward it, and a model that reproduces its input faithfully reconstructs anomalies
faithfully too — which is the signature of SMD's recall-limited failure (recall 73.90 against
precision 87.25). Removing the self-term forces a genuine leave-one-out prediction.

Measured, it does not work. The predicted recall gain appears on SMD but is small (73.90 → 75.39,
F1 80.02 → 80.55), and PSM gets worse (F1 95.26 → 93.23, recall 91.96 → 88.22). Copying the input
explains little of SMD's gap. Rejected as a default.

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

### Imputation — ahead of TimeMixer on four of six datasets

Mean masked MSE over four mask rates. For each dataset the configuration is chosen on
**validation** loss among all seven variants of the ablation, and plain SparseTSF is one of the
candidates, so validation can choose "no units". Both baselines were run on this machine under this
protocol.

| Dataset | Chosen on validation | Params | SparseTSF | **SparseTSFPlus** | vs SparseTSF | TimeMixer | vs TimeMixer |
|---|---|---:|---:|---:|---:|---:|---:|
| ETTh1 | refinement | 41 | 0.163307 | **0.120684** | −26.1% | 0.120057 | +0.5% |
| ETTm1 | gated refinement, mask-aware, gated phase mixing | 599 | 0.071481 | **0.041517** | −41.9% | 0.047031 | **−11.7%** |
| ETTh2 | gated refinement, mask-aware | 42 | 0.070296 | **0.064077** | −8.8% | 0.066370 | **−3.5%** |
| ETTm2 | gated refinement, mask-aware | 582 | 0.032690 | **0.026926** | −17.6% | 0.027787 | **−3.1%** |
| Weather | gated refinement, mask-aware, gated phase mixing | 599 | 0.034388 | **0.029912** | −13.0% | 0.031546 | **−5.2%** |
| Electricity | gated refinement, mask-aware | 42 | 0.130393 | **0.104778** | −19.6% | 0.080314 | +30.5% |
| **Mean, all 6** | | | 0.083759 | **0.064649** | **−22.8%** | 0.062184 | **+4.0%** |
| Mean, the 5 other than Electricity | | | 0.074432 | **0.056623** | **−23.9%** | 0.058558 | **−3.3%** |

Over all six datasets the gap to TimeMixer goes from **+34.7% to +4.0%**, closing 88% of it. On the
five other than Electricity SparseTSFPlus is now 3.3% ahead (from +27.1% behind). Validation picked
the configuration with the lowest test error, or one within 0.3% of it, on every dataset.

**One configuration is now almost as good as choosing per dataset.** Before the refinement gate, the
best single configuration was 8% behind the per-dataset choice (0.0716 against 0.0663). With it,
gated refinement plus mask-aware aggregation on every dataset averages **0.0651**, within 0.7% of
the per-dataset choice (0.0646). It beats TimeMixer on four of the six datasets (ETTh2, ETTm1, ETTm2,
Weather) and on the mean of the five other than Electricity (0.0572 against 0.0586), with 42 to 582
parameters. This is the gate's main result (Section 4).

**Electricity is where TimeMixer stays clearly ahead.** TimeMixer's Electricity cells were measured
last, after this ablation. Gated refinement with mask-aware aggregation cuts SparseTSF's error there
by 19.6%, but TimeMixer is 30.5% better still.

**ETTh2 is no longer a failure.** Every ungated unit variant was worse than plain SparseTSF on ETTh2,
by up to 70% at low mask rates, and its training loss settled at about twice SparseTSF's. Validation
used to reject every unit there. With the gate, refinement settles at about half strength and beats
plain SparseTSF at every mask rate (−8.8% on average), and TimeMixer as well (−3.5%).

ETTh1 and ETTm1 per mask rate:

| mask | ETTh1 SparseTSF | **ETTh1 Plus** | ETTh1 TimeMixer | ETTm1 SparseTSF | **ETTm1 Plus** | ETTm1 TimeMixer |
|---:|---:|---:|---:|---:|---:|---:|
| 0.125 | 0.113335 | **0.089294** | 0.096742 | 0.048898 | **0.034178** | 0.039071 |
| 0.25 | 0.146550 | **0.105566** | 0.111707 | 0.062403 | **0.036781** | 0.042218 |
| 0.375 | 0.179265 | **0.128675** | 0.125347 | 0.078078 | **0.042464** | 0.048693 |
| 0.5 | 0.214076 | **0.159201** | 0.146432 | 0.096544 | **0.052645** | 0.058143 |
| **mean** | 0.163307 | **0.120684** | 0.120057 | 0.071481 | **0.041517** | 0.047031 |

The gap moves from **+36.0% → +0.5%** on ETTh1 and **+52.0% → −11.7%** on ETTm1. ETTh1 beats
TimeMixer outright at the two lower mask rates.

Two observations worth discussing:

- **The parameter-free unit does most of the work.** Refinement, gated or not, is part of the chosen
  configuration on all six datasets. It adds no parameters, and its gate adds one; on ETTh1, ETTh2
  and Electricity the improved model has 41 or 42 parameters, against stock SparseTSF's 41.
- **Unit composition depends on the dataset, but much less than it did.** ETTm1 (~34k windows)
  benefits from every unit and they compose; ETTh1 (~8.5k windows) is best with ungated refinement
  alone, and adding capacity hurts. Gated refinement narrows the spread: on every dataset it is
  within 2% of the validation choice (on ETTh1 it is 1.9% behind).

### PEMS multivariate short-term — 24% of the MAE removed, channel-independently

All four PEMS datasets, `seq_len=96 → pred_len=12`, MAE:

| Dataset | Published SparseTSF (21 params) | **Phase mixing, p = 4 (93)** | Dense, p = 1 (1,153) |
|---|---:|---:|---:|
| PEMS03 | 28.28 | **21.34** | 21.17 |
| PEMS04 | 36.12 | **27.43** | 27.33 |
| PEMS07 | 42.89 | **31.02** | 30.90 |
| PEMS08 | 29.25 | **22.22** | 22.11 |
| **Mean** | **34.13** | **25.50** | **25.38** |

The improvement is consistent: 24% to 28% on every dataset. The 4-dataset mean falls **34.13 →
25.50 (−25.3%)**, with MAPE 20.81% → 15.22% and RMSE 52.10 → 40.67, for about 70 extra parameters.
That closes **52%** of the distance to the locally run TimeMixer's 17.51 (published 17.41; the gap
goes from +95% to +46%). The residual is the cross-channel component that was deliberately out of scope. The dense
model is marginally better on validation on all four datasets, by about 0.5%; phase mixing is the
efficient choice, not the strictly best one.

A finding worth raising in discussion: that `period_len=1` — which removes the period fold
altogether — performs *best* means SparseTSF's cross-period sparse structure is **actively harmful**
on PEMS. That is a sharper explanation of the PEMS failure than channel independence alone, and it
suggests the inductive bias, not just capacity, is mismatched to traffic data.

The TimeMixer and TimeMixer++ figures are published 4-dataset averages from the authors' hardware;
the SparseTSF figures are this study's own runs.

### Long-term forecasting — a free improvement from period selection

`period_len` chosen on validation loss rather than taken from the original paper:

| Dataset | published `p` | selected `p` | MSE before | MSE after | TimeMixer (local) |
|---|---:|---:|---:|---:|---:|
| **ETTh1** | 24 | **12** | 0.396635 | **0.383847** (−3.22%) | 0.385794 |
| Weather | 4 | 4 | 0.198635 | unchanged | 0.164526 |
| ETTm1 | 4 | 4 | 0.362121 | unchanged | 0.326398 |

At H = 96, ETTh1 flips from a **+2.81% loss to a −0.51% win** over TimeMixer for 36 extra
parameters and no architectural change. At the longer horizons period 12 is roughly neutral
(+0.8%, −0.1%, −1.7% at 192, 336 and 720), where SparseTSF already beat TimeMixer. Weather and ETTm1
are period-*insensitive*: both degrade monotonically as the period grows, and the published value
is already optimal on both validation and test.

**Best forecasting configuration at every horizon.** Every unit was screened at H = 96. The one
with the lowest validation loss — the learned-trend decomposition, on all three datasets — was then
run at every horizon:

| Dataset | H = 96 | 192 | 336 | 720 | 4-horizon mean MSE vs TimeMixer |
|---|---:|---:|---:|---:|---:|
| ETTh1 | −2.6% | +0.8% | −0.4% | −1.5% | **−5.9%** |
| ETTm1 | −2.7% | −1.6% | −1.3% | −0.8% | +7.0% |
| Weather | −0.5% | −0.2% | −1.0% | −0.5% | +11.1% |

Horizon columns are the change against published SparseTSF. Over all 12 cells the mean MSE falls
1.0% (0.3754 → 0.3715). The forecasting gains are real and consistent on ETTm1 and Weather, but
small; the gap to TimeMixer on those two remains.

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

## 6. One model on all eight tasks

The ablations above chose units per dataset, which is an ablation, not a model. To test the
thesis's actual target, SparseTSFPlus was run on every task and dataset of the protocol in two
versions ([results_sparsetsf_plus_suite.md](results_sparsetsf_plus_suite.md)):

- **Fixed**: one configuration for everything. Learned-trend decomposition and phase mixing in
  the backbone; gated refinement and mask-aware aggregation whenever the input has missing values;
  default heads otherwise. (Refinement was ungated in the first version of this run; the gate,
  Section 4, changes only the imputation rows.)
- **Per task**: the unit set that won each task in the ablations, applied unchanged to all of
  that task's datasets.

Both use SparseTSF's own published period for every dataset, so nothing is tuned per dataset.

**The cost of generality is small outside imputation.** The fixed version is within 0.4% of the
per-task version on every other task, and slightly ahead on long-term forecasting, M4 and PEMS. In
imputation it is 5.8% behind (0.0689 against 0.0651). That gap has a single source: phase mixing,
which the per-task imputation configuration leaves out. It costs 20.1% on ETTh2 and 10.5% on
Electricity and is within ±1.2% on the other four datasets.

**Against SparseTSF it improves or matches every task:** PEMS −25.4%, imputation −17.8%, M4 OWA
−6.0%, few-shot −2.0%, long-term −1.8%, zero-shot −0.4%. Anomaly detection and classification do
not move. Phase mixing changes anomaly F1 by 0.02 points while taking the model from 41 to 666
parameters, and no unit applies to the classification head.

**Against TimeMixer (local)** it wins few-shot (−12.1%), zero-shot (−12.7%) and anomaly detection
(+4.5 F1 points, carried by MSL). It loses long-term forecasting (+8.4%; it wins both hourly ETT
datasets and trails most on Solar), M4 (+8.7%), PEMS (+45.3%), imputation (+10.8%) and
classification (−1.5 points).

**Where the one configuration still hurts.** The first version of this run found two units that
were not safe by default: refinement on ETTh2 imputation, and phase mixing on Electricity
imputation. A learned gate fixed the first: gated refinement now beats ungated refinement on all
six imputation datasets (Section 4). It did not fix the second. Phase mixing still costs 31–44%
on ETTh2 and 16–26% on Electricity at mask rates 0.375 and 0.5, while helping slightly at lower
rates, and it is the next step's target (Section 8). The remaining gaps to TimeMixer are the
structural ones identified earlier:
- cross-channel structure (PEMS)
- a 96-step lookback shorter than the 144-step daily cycle (Solar, as for Weather)
- the classification head

**Energy**, measured in the matched fixed-epoch protocol with SparseTSF and TimeMixer re-run in
the same session:
- The fixed SparseTSFPlus uses 1.35× SparseTSF's energy, so its units cost about a third more.
- TimeMixer uses 18.9× SparseTSF's energy, which is 14× SparseTSFPlus's.

Runs this short vary between sessions. SparseTSF's July total was 42% higher than today's, while
TimeMixer's moved 5%, so only same-session ratios should be quoted. The July carbon table's
SparseTSF/TimeMixer ratio of 8.7% is therefore an overestimate; the same-session figure is 5.3%.

All of this is one seed. The differences between the two versions are mostly under 1%, too small
to rank them without repeats.

---

## 7. Runtime

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

**Completing the tests.** A second round ran every unit that had been implemented but not measured,
and the four imputation datasets still missing:

| Experiment group | Runs | Training time |
|---|---:|---:|
| Forecasting units, screen at H = 96 and best variant at every horizon | 36 | 1,052 s |
| PEMS03/04/07/08, phase mixing and dense | 8 | 1,014 s |
| Anomaly, leave-one-out (PSM, SMD) | 4 | 58 s |
| Imputation on ETTh2, ETTm2, Weather, Electricity (+1 reproduction check) | 49 | 8,094 s |
| **Total** | **97** | **10,218 s (2 h 50 m)** |

Wall clock was about 4.5 hours. Electricity dominated: about 7 minutes per run against about 1
minute for ETTh2. One batch was stopped by the session's two-hour background limit partway through
Electricity; because every runner skips configurations whose log already holds a result, the
remainder was resumed in two shorter jobs and nothing was lost. Across both rounds, 204 runs took
4 h 15 m of training.

**One model on all eight tasks (Section 6).** 220 runs for the two versions plus 40 same-session
energy references:

| Experiment group | Runs | Training time |
|---|---:|---:|
| Long-term forecasting, 8 datasets × 4 horizons × 2 versions | 64 | 15,210 s |
| Zero-shot, 6 pairs × 4 horizons × 2 | 48 | 1,039 s |
| Few-shot, 6 datasets × 4 horizons × 2 | 48 | 776 s |
| PEMS, fixed version | 4 | 794 s |
| M4, 6 subsets × 2 | 12 | 521 s |
| Anomaly, fixed version | 4 | 96 s |
| Energy, SparseTSFPlus (both versions) | 40 | 667 s |
| Energy, same-session SparseTSF and TimeMixer references | 40 | 3,186 s |
| **Total** | **260** | **22,289 s (6 h 11 m)** |

Wall clock was 9 h 9 m: 7 h 56 m for the suite and 1 h 13 m for the references. Long-term
forecasting on Solar, Electricity and Traffic took most of it. There SparseTSFPlus trains for more
epochs than SparseTSF before early stopping, and each epoch costs more, up to 4× SparseTSF's time
on Traffic.

**Gates (Section 4).** Three imputation variants on 6 datasets × 4 mask rates:

| Experiment group | Runs | Training time |
|---|---:|---:|
| Gated refinement (`u5g`) | 24 | 2,932 s |
| Gated refinement and gated phase mixing (`allg`) | 24 | 3,371 s |
| Gated refinement, ungated phase mixing (`allrg`) | 24 | 3,202 s |
| **Total** | **72** | **9,506 s (2 h 38 m)** |

Wall clock was 3 h 30 m, about a third of it on Electricity.

---

## 8. Next steps

Ordered by value for the goal of one model that serves every task.

1. **Confirm mask-aware phase mixing over seeds.** Of the two fixes tested for phase mixing on
   inputs with missing values (Section 4), mask-aware mixing brought the fixed imputation form within
   1.0% of the form without phase mixing; applying it to complete inputs only failed. It is a
   candidate, not yet part of the fixed configuration. The one cell it did not fix (ETTh2, mask
   0.375) is on the dataset with the training plateau, so seed repeats decide.

2. **Cross-channel structure needs more than a context term.** Both channel-context units improved
   PEMS by 4–5% without harm elsewhere but missed the 5% screening rule (Section 4). A larger PEMS
   gain probably needs channel interaction inside the backbone rather than a term added to the
   forecast, which conflicts with checkpoint transfer unless its parameters stay independent of the
   channel count.

3. **Repeat the headline results over more seeds.** Every run so far uses one seed. The fixed and
   per-task versions differ by under 1% on most tasks, which one seed cannot resolve.

4. **Fold these results into the top-level narrative.** `results_final_report.md` and the root
   `README.md` still describe a three-model study. One SparseTSFPlus configuration that improves
   on SparseTSF on every task and beats TimeMixer on few-shot, zero-shot and anomaly detection, at
   37 to 8,661 parameters, changes the headline claim.

5. **Measure energy beyond long-term forecasting.** The matched energy protocol covers only
   long-term forecasting on five datasets. An efficiency claim across tasks needs the same
   measurement per task.

6. **Close the classification gap in the backbone, not the head.** TimeMixer++ classifies with the
   same kind of flatten-linear readout as SparseTSF, so its 15-point lead comes from what its
   backbone feeds that readout. Every readout change tested here failed.

7. **Investigate Weather.** It is period-insensitive, and the best unit improves it by under 1%. Its
   daily cycle is 144 steps against a 96-step lookback, which no period-based mechanism can
   recover; a longer lookback is the more promising direction.

Measured and closed since the first version of this report: learned gates for refinement (adopted)
and phase mixing (rejected) (Section 4), one fixed configuration on all eight tasks and its energy
cost (Section 6), the four remaining imputation datasets, PEMS03/04/07, low-rank phase mixing, both forms of decomposition, the multi-period
ensemble, phase mixing combined with decomposition, and leave-one-out reconstruction for anomaly
detection.

---

## Reproducing this

All runners are idempotent — a configuration whose log already holds a valid metric is skipped — and
every table is regenerated from the raw per-run logs rather than maintained by hand. The equivalence
check, the period diagnostics, and the two ablations are each a single command from `TimeMixer/`;
see `CLAUDE.md` for the exact invocations and the verification recipe.
