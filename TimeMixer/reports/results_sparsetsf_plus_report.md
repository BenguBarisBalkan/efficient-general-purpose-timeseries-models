# SparseTSFPlus — design rationale, units and outcomes

Consolidated write-up for the fourth model in this study: **SparseTSFPlus**, SparseTSF extended
with cheap, individually-toggleable *units* borrowed from TimeMixer and TimeMixer++.

**Headline.** On imputation, SparseTSFPlus cuts SparseTSF's error by 20.9% across all six
datasets and **matches TimeMixer on the five where TimeMixer was run** (mean 0.0584 against
0.0586), with 41 to 597 parameters against ~75,500. On PEMS multivariate short-term it removes 25%
of the MAE on all four datasets for ~70 extra parameters. Forecasting gains are small (about 1%),
and on classification no variant improved the average — both reported as findings rather than
omitted. One caveat runs through all of it: the best configuration differs by dataset, and on ETTh2
validation chose no units at all.

Supporting tables: [period sensitivity](results_period_sensitivity.md) ·
[imputation ablation](results_sparsetsf_plus_ablation.md) ·
[forecasting ablation](results_sparsetsf_plus_forecast.md) ·
[anomaly ablation](results_sparsetsf_plus_anomaly.md) ·
[classification ablation](results_sparsetsf_plus_classification.md) ·
[8-task suite](results_sparsetsf_all_tasks.md)

---

## 1. Starting point: SparseTSF across the full 8-task suite

The TimeMixer++ paper evaluates eight task categories. SparseTSF was extended to all of them in
the previous phase of this study, giving the baseline picture below. This table is what the choice
of target tasks was made from.

| # | Task | Metric | SparseTSF | TimeMixer | TimeMixer++ | Gap (SparseTSF vs best) | Baselines |
|---|---|---|---:|---:|---:|---:|---|
| 1 | Long-term forecasting (sl=96, 5 datasets × 4 horizons) | mean MSE ↓ | 0.354 | 0.348 | **0.324** † | **+9.1%** | **TimeMixer local**; TimeMixer++ published † |
| 2 | Univariate short-term (M4, 6 subsets) | OWA ↓ | 0.972 | 0.840 | **0.821** | **+18.4%** | published |
| 3 | Multivariate short-term (PEMS03/04/07/08) | MAE ↓ | 34.13 | 17.41 | **15.91** | **+114.5%** | published |
| 4 | Imputation (6 datasets × 4 mask rates) | masked MSE ↓ | 0.0838 § | **0.0586** § | — | **+43.0%** | **local (TimeMixer)** |
| 5 | Few-shot, 10% train (ETT avg) | MSE ↓ | 0.441 | 0.453 | **0.396** | **+11.4%** | published |
| 6 | Zero-shot transfer (6 ETT pairs) | mean MSE ↓ | 0.393 | 0.467 | **0.386** | **+1.9%** | published |
| 7 | Anomaly detection (PSM, SMD) | F1 ↑ | **87.64** | — | 87.47 | comparable ‡ | published |
| 8 | Classification (10 UEA) | accuracy ↑ | 60.90 | — | **75.9** | **−15.0 pts** | published |

† **TimeMixer++ was also run locally for forecasting (mean 0.449), but it is undertrained by
design:** it requires `lr=5e-5` for NaN stability, so its local accuracy understates the
architecture. Its published figure (0.324) is therefore used as the fairer reference here. Its
local *energy* numbers are fair. Against the locally run TimeMixer the forecasting gap is +1.8%.

‡ Not comparable: SparseTSF covers 2 of 5 anomaly datasets (MSL/SMAP/SWaT are unobtainable), so
the F1 averages are over different dataset sets.

§ The two imputation means are over **different dataset counts** — SparseTSF over 6, TimeMixer over
5 (Electricity was not run for TimeMixer). The per-dataset comparison used in §5 is like-for-like
on ETTh1 and ETTm1, and TimeMixer wins all 20 of those cells, so the direction of the gap is not in
doubt; only the exact headline percentage is affected.

**Which baselines are local.** Long-term forecasting has local sweeps for **both** TimeMixer
(30 of 32 configurations; Traffic 336/720 are beyond this laptop) and TimeMixer++ (21 of 32:
ETT ×4, Weather, Solar 96); imputation has a local TimeMixer sweep (20 runs); and the energy study
ran all three models under a matched protocol (40 TimeMixer/TimeMixer++ runs) — **111 completed
local TimeMixer/TimeMixer++ runs in total**, all on the same RTX 3050 Ti (4 GB). The other six
per-task benchmarks (M4, PEMS, few-shot, zero-shot, anomaly, classification) were run locally for
SparseTSF only, against *published* TimeMixer/TimeMixer++ figures produced with the authors'
tuning on their hardware. Those six are reference points, not controlled head-to-heads.

### What this table says

SparseTSF is **not** uniformly behind. On long-term forecasting it is within 1.8% of the locally
run TimeMixer and wins 7 of 20 configurations, sweeping the hourly ETTh datasets; at its native
`seq_len=720` its mean MSE of 0.312 beats both baselines' 96-step figures. On zero-shot transfer it
is 15.8% better than TimeMixer and within 1.9% of TimeMixer++, winning 3 of 6 pairs, and it is
competitive few-shot. The deficits are concentrated and structural.

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

**What was not chosen, and why.** Long-term forecasting was within 1.8% of the local TimeMixer
and ahead of both baselines at the native lookback, so there was little to gain. On zero-shot
transfer SparseTSF beats TimeMixer on all 6 pairs and is within 1.9% of TimeMixer++.
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

### Imputation — parity with TimeMixer across five datasets

Mean masked MSE over four mask rates. For each dataset the configuration is chosen on
**validation** loss, and plain SparseTSF is one of the candidates, so validation can choose "no
units". Both baselines were run on this machine under this protocol.

| Dataset | Chosen on validation | Params | SparseTSF | **SparseTSFPlus** | vs SparseTSF | TimeMixer | vs TimeMixer |
|---|---|---:|---:|---:|---:|---:|---:|
| ETTh1 | refinement | 41 | 0.163307 | **0.120684** | −26.1% | 0.120057 | +0.5% |
| ETTm1 | refinement, mask-aware, phase mixing | 597 | 0.071481 | **0.042010** | −41.2% | 0.047031 | **−10.7%** |
| ETTh2 | none — plain SparseTSF | 41 | 0.070296 | 0.070296 | 0.0% | 0.066370 | +5.9% |
| ETTm2 | refinement | 581 | 0.032690 | **0.027865** | −14.8% | 0.027787 | +0.3% |
| Weather | refinement | 581 | 0.034388 | **0.031335** | −8.9% | 0.031546 | **−0.7%** |
| Electricity | refinement, mask-aware | 41 | 0.130393 | **0.105336** | −19.2% | not run | — |
| **Mean, the 5 with TimeMixer** | | | 0.074432 | **0.058438** | **−21.5%** | 0.058558 | **−0.2%** |
| **Mean, all 6** | | | 0.083759 | **0.066254** | **−20.9%** | | |

On the five datasets where both models were run, the gap to TimeMixer goes from **+27.1% to
−0.2%**. Validation picked the configuration with the lowest test error on 5 of 6 datasets; on ETTm2
it picked refinement over all units, which cost 0.5%.

**ETTh2 is where the units fail.** Every unit variant is worse than plain SparseTSF there — up to 70%
worse at low mask rates — and even the training loss settles at about twice SparseTSF's, so the
shared-weight two-pass setup lands on a worse solution on this dataset. A fresh run of plain
SparseTSF reproduced the published result exactly (0.058744, identical validation loss), so the
setup is not at fault. Validation correctly rejects every unit on ETTh2.

**The parity depends on choosing per dataset.** With one fixed configuration everywhere, the best is
refinement plus mask-aware aggregation: 14.5% better than SparseTSF over six datasets, but 10.8%
behind TimeMixer over five, with ETTh2 the main cost. A refinement step that can switch itself off
where it hurts would close most of that difference, and is the clearest next step toward one
configuration for every dataset.

ETTh1 and ETTm1 per mask rate:

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

- **The parameter-free unit does most of the work.** Refinement is part of the chosen configuration
  on five of six datasets and adds no parameters; on ETTh1 and Electricity the improved model is the
  same 41 parameters as stock SparseTSF.
- **Unit composition depends on the dataset.** ETTm1 (~34k windows) benefits from every unit and
  they compose; ETTh1 (~8.5k windows) is best with the free unit alone, and adding capacity hurts.
  The ordering is stable across all four mask rates independently, so this is a real effect and not
  an artefact — and with ETTh2 it means no single fixed configuration is optimal.

On ETTh1 and ETTm1, validation ranked all 24 cells in exactly the test order, so the unit selection
does not depend on the test signal.

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
That closes **52%** of the distance to TimeMixer's published 17.41 (the gap goes from +96% to
+46%). The residual is the cross-channel component that was deliberately out of scope. The dense
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

---

## 7. Next steps

Ordered by value for the goal of one model that serves every task.

1. **Make refinement safe by default.** It is the strongest unit and is chosen on five of six
   imputation datasets, but on ETTh2 it does real harm, and that one dataset is what separates
   per-dataset parity from a single fixed configuration (−0.2% against +10.8% versus TimeMixer).
   A second pass behind a learned gate initialised to pass-through would let training switch it off
   where it hurts. Success means recovering ETTh2 while keeping the gains elsewhere.

2. **Run one frozen configuration on every task.** Learned-trend decomposition for forecasting,
   phase mixing where the fold constrains the model, and the masked-input rule for missing values,
   all fixed across datasets. Report the gain over SparseTSF and the cost of generality — how much
   of the per-dataset gains a single configuration gives up.

3. **Fold these results into the top-level narrative.** `results_final_report.md` and the root
   `README.md` still describe a three-model study. Imputation parity with TimeMixer across five
   datasets, at 41 to 597 parameters, changes the headline claim.

4. **Re-run the energy sweep for the new model.** Any energy claim requires the matched fixed-epoch
   carbon protocol. Refinement adds no parameters but runs the reconstruction head twice, so its
   energy cost should be measured rather than assumed negligible.

5. **Close the classification gap in the backbone, not the head.** TimeMixer++ classifies with the
   same kind of flatten-linear readout as SparseTSF, so its 15-point lead comes from what its
   backbone feeds that readout. Every readout change tested here failed.

6. **Investigate Weather.** It is period-insensitive, and the best unit improves it by under 1%. Its
   daily cycle is 144 steps against a 96-step lookback, which no period-based mechanism can
   recover; a longer lookback is the more promising direction.

Measured and closed since the first version of this report: the four remaining imputation
datasets, PEMS03/04/07, low-rank phase mixing, both forms of decomposition, the multi-period
ensemble, phase mixing combined with decomposition, and leave-one-out reconstruction for anomaly
detection.

---

## Reproducing this

All runners are idempotent — a configuration whose log already holds a valid metric is skipped — and
every table is regenerated from the raw per-run logs rather than maintained by hand. The equivalence
check, the period diagnostics, and the two ablations are each a single command from `TimeMixer/`;
see `CLAUDE.md` for the exact invocations and the verification recipe.
