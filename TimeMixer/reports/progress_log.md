# Progress log

Running record for the biweekly supervisor meetings, newest entry first. Each entry covers what
was planned and why, what was done, what was found (with the baseline each number is compared
against), which reports changed, and what comes next. It records ideas and evidence, not
implementation. The rules for keeping it are in the root `CLAUDE.md` ("Thesis context").

Numbers are **local** (this machine, this protocol) unless marked **published**.

---

## 2026-10-21 — 24-hour backbone plan, step 2: channel context for PEMS

**Planned.**
- **The problem.** PEMS is SparseTSFPlus's largest remaining gap to the local TimeMixer (25.45
  against 17.51 MAE, +45%). The backbone treats every channel independently, so it cannot use
  the fact that neighbouring traffic sensors move together. TimeMixer runs PEMS with its channels
  embedded jointly (`channel_independence 0`), which points at cross-channel information as the
  missing ingredient.
- **Two candidates,** each behind a ReZero gate (starts as the model without the unit), applied
  to the forecast, and with no parameter that depends on the channel count, so checkpoints still
  transfer between datasets:
  - `mean`: the channel mean of the window gets its own cross-period map; each channel adds it,
    scaled by its own least-squares loading on that mean, computed from the window rather than
    learned. Source: the aggregate-and-redistribute core of SOFTS (Han et al., 2024). Cost: one
    small period map plus the gate (9 parameters on PEMS, 17 on ETTh1).
  - `corr`: each channel adds a softmax-weighted mix of the other channels' forecasts, weighted
    by how strongly their input windows correlate with its own, with a learned temperature.
    Source: attention across variates in iTransformer (Liu et al., 2024), with the learned
    projections removed. Cost: 2 parameters.
- **Checked before running:** both gates at zero reproduce the fixed configuration exactly;
  gradients reach both gates and the temperature is finite (an early version produced a NaN
  gradient for the temperature and was fixed before any run).
- **Screen.** Both on PEMS03/04/07/08 and, as a guard for datasets that prefer independent
  channels, on ETTh1, ETTm1 and Weather at horizon 96 (14 runs). Each run is the fixed
  configuration's own command plus the unit flag, so the fixed cells are the control.
- **Prediction.** PEMS MAE falls by at least 10%, while ETTh1, ETTm1 and Weather are no more than
  1% worse. **Refuted** if PEMS improves by less than 3%.
- **Decision rule (validation).** A candidate goes to the full suite if its PEMS validation loss
  is at least 5% lower on average and no long-term check is more than 1% worse. The report
  prints this decision from the logs.

---

## 2026-10-21 — 24-hour backbone plan, step 1: phase mixing on inputs with missing values

The plan for the next 24 hours (agreed before an unattended run) is in `CLAUDE.md` ("Active plan")
and in the plan document linked there: phase mixing on inputs with missing values (this entry),
channel context for PEMS, seed repeats, and, if time allows, an automatic period choice. A unit
that passes its decision rule is reported as a **candidate** configuration for review; the
headline fixed configuration does not change.

**Planned.**
- **The problem.** With gated refinement in place, phase mixing is the only source of the
  imputation gap between the fixed configuration (0.0689) and the per-task one (0.0651). It
  helps or is neutral at masks 0.125 and 0.25 on every dataset (−4.1% to +2.5%), but at 0.375 and
  0.5 it costs 31–44% on ETTh2 and 16–26% on Electricity.
- **Hypothesis.** The harm appears only when many points are missing, so the holes are the likely
  cause: mixing along the phase axis spreads the positions that hold no observation into the
  observed hours of each period.
- **Two candidates, both free of parameters:**
  - `allrg_pc`: phase mixing on complete inputs only. The first imputation pass, which still has
    the holes, skips it; the refinement pass, whose input has been filled, keeps it. Forecasting
    and anomaly detection always have complete inputs, so they are bitwise unchanged.
  - `allrg_pm`: mask-aware phase mixing. Only observed positions feed the mixing, rescaled by the
    share of the period that was observed. This is the partial-convolution rule (Liu et al.,
    2018) that mask-aware aggregation already applies to the time convolution.
- **Checked before running:** with every point observed, mask-aware mixing equals plain phase
  mixing exactly; on anomaly inputs (always complete) the complete-only form equals plain phase
  mixing exactly; all units off is still bitwise SparseTSF.
- **Runs.** Both on all 6 datasets × 4 mask rates (48 runs, about 2.5 hours), on top of `allrg`
  (gated refinement, mask-aware aggregation, ungated phase mixing).
- **Prediction.** The cost of phase mixing at masks 0.375 and 0.5 on ETTh2 and Electricity falls
  below 5% (against `u5g`, the same model without phase mixing), the low-mask gains stay, and the
  six-dataset mean falls from 0.0689 to 0.0655 or lower. **Refuted** if that cost stays above 15%:
  then the holes are not the mechanism.
- **Decision rule.** The variant with the lowest mean validation loss becomes a candidate if it
  also beats `allrg` on validation.
- **Also added:** a `--seed` flag for the seed repeats of step 3. Its default is the seed of every
  existing run; a 1-epoch check reproduced the default run to 7 significant digits.

**Done.** 48 imputation runs (21:49–00:14).

**Found.** Six-dataset means of masked MSE:

| | without phase mixing (`u5g`) | plain phase mixing (`allrg`, fixed) | complete inputs only (`allrg_pc`) | **mask-aware (`allrg_pm`)** |
|---|---:|---:|---:|---:|
| test | 0.0651 | 0.0689 | 0.0728 | **0.0657** |
| validation | 0.0713 | 0.0737 | 0.0781 | **0.0717** |

- **Mask-aware phase mixing is the candidate** (lowest validation of the two, and better than
  `allrg` on validation). It brings the fixed imputation form from 0.0689 to 0.0657 (−4.6%), within
  1.0% of the form without phase mixing (0.0651). Against SparseTSF that is −21.5%; against the local
  TimeMixer, +5.7% (the fixed form was +10.8%).
- **The prediction held only partly.** At masks 0.375 and 0.5 on ETTh2 and Electricity, phase
  mixing's cost (against the same model without it) went:

  | Cell | Plain phase mixing | Mask-aware |
  |---|---:|---:|
  | ETTh2, mask 0.5 | +31% | −4.8% |
  | Electricity, mask 0.5 | +16.5% | −14.1% |
  | Electricity, mask 0.375 | +26% | +13% |
  | ETTh2, mask 0.375 | +44% | +47% |

  So "below 5% in every high-mask cell" was not met, and the mean (0.0657) narrowly missed the
  predicted 0.0655 or lower. ETTh2 at 0.375 is the dataset with the training plateau, so the seed
  repeats decide whether that cell is the unit or chance. One new cost: Electricity at mask 0.25 is
  +8.5% where plain phase mixing was −4.1%.
- **Complete inputs only is rejected.** Skipping phase mixing in the first pass costs 23–42% on
  Electricity and 29–39% on ETTh1 at low mask rates. So the first pass needs phase mixing; what
  helps is mixing only the observed points, not removing the mixing. The holes are part of the
  mechanism, but not in the simple form tested by `allrg_pc`.
- **Step 4 closed without training.** On ETTh1 and ETTh2 the FFT rule (as in TimesNet) picks the
  published period, 24. On every other dataset it picks 48, the edge of the allowed band, because
  these spectra are dominated by slow variation. The period-sensitivity probe had already measured
  48 as worse: +6.9% MSE on ETTm1 and +7.7% on Weather. On PEMS08 the probe found the error rising
  with the period (period 3: 23.68 MAE; period 12: 29.25). For SparseTSF the dominant cycle is the
  wrong choice; it benefits from short periods with more segments per window.
  (`results_fft_periods.md`)

**Reports changed.** `results_sparsetsf_plus_ablation.md` (two variants); new
`results_fft_periods.md`; `results_sparsetsf_plus_report.md` (new subsection on phase mixing with
missing values); the thesis document.

**Next.** Step 2 screen (running). The seed repeats include the mask-aware candidate on ETTh1 and
ETTh2.

---

## 2026-10-21 — Gates for refinement and phase mixing

**Planned.** The fixed SparseTSFPlus loses to a tuned one in two places, both in imputation:
refinement on ETTh2 (0.0955 against 0.0703 without it) and phase mixing on Electricity (0.1239
against 0.1053). The thesis rule is that a unit which helps one dataset and hurts another needs a
safe default before it joins the fixed configuration.

- **Evidence that a learned switch can work.** In both cases the unit also raises the *training*
  loss: on ETTh2 at mask 0.125, refinement gives 0.114 against 0.058, and on Electricity at
  mask 0.25 phase mixing gives 0.129 against 0.095. So the harm is visible to the training
  signal; it is not overfitting.
- **Idea.** A ReZero gate (Bachlechner et al., UAI 2021), the same zero-initialised scalar this
  model already uses for its multi-period branches. Each unit's contribution is multiplied by one
  learned number that starts at 0, so a gated unit starts as the model without it, and training
  decides how much of it to use. Cost: 1 parameter per unit.
- **Refinement gate.** The output becomes pass 1 + g·(pass 2 − pass 1). Without the gate, pass 2
  replaces pass 1, so "no refinement" cannot even be represented once the unit is on, and the
  shared weights must serve both the input with holes and the filled one.
- **Phase-mixing gate.** img + g·mix(img). The ungated mixer also starts at zero, yet training
  carries it somewhere worse on Electricity; the gate puts the unit's whole contribution behind
  one number.
- **Checked before running.** With each gate at its starting zero, the gated model's output is
  bitwise identical to the model without that unit (forecasting, imputation, anomaly detection),
  and all units off is still bitwise SparseTSF.
- **Runs.** Two imputation variants on all 6 datasets × 4 mask rates (48 runs, about 3 hours):
  `u5g` (refinement gated) and `allg` (refinement and phase mixing both gated, the candidate
  imputation form of the fixed configuration). Each run's learned gate values go into its log.
- **Predictions, and what would refute them.**
  1. ETTh2: `u5g` comes back to within 5% of no refinement (≤ 0.074), with a small refinement
     gate. Refuted if ETTh2 stays above 0.085.
  2. Where refinement helps (ETTh1, ETTm1, ETTm2, Weather, Electricity), `u5g` stays within 3% of
     ungated `u5`. If the gate stays near zero there too, it only switches refinement off
     everywhere: safe, but useless.
  3. Electricity: `allg` comes back to within 2% of `u5` (≤ 0.1075). Refuted if it stays above
     0.115.
  4. The six-dataset mean of `allg` is at or below the per-task result (0.0716), which would
     close the 3.9% cost of generality in imputation.
- **Caveats.** Imputation early-stops on the test split (upstream behaviour), so these results
  are a sensitivity analysis, and every run uses one seed.

**Done.** 72 imputation runs (3 h 30 m): the two planned variants, plus `allrg` (refinement gated,
phase mixing ungated). `allrg` was added once ETTh2 and Electricity showed that only the refinement
gate works, because it is the imputation form the fixed configuration then takes. Learned gate
values were read from every tested checkpoint.

**Found.** Six-dataset means of masked MSE (lower is better):

| | no units | refinement | + phase mixing (old fixed) | **gated refinement** | both gated | gated refinement + phase mixing (new fixed) |
|---|---:|---:|---:|---:|---:|---:|
| test | 0.0838 | 0.0716 | 0.0744 | **0.0651** | 0.0688 | 0.0689 |
| validation | 0.0904 | 0.0763 | 0.0784 | **0.0713** | 0.0737 | 0.0737 |

- **Prediction 1 was wrong, and the result was better than predicted.** On ETTh2 the refinement gate
  did not switch refinement off; it settled at about 0.5, averaging the two passes. That beat both no
  refinement (0.0641 against 0.0703) and full refinement (0.0955) at every mask rate.
- **Prediction 2 held, and more.** Gated refinement is better than ungated on all six datasets,
  −9.1% on average, and lowest on validation. Its gate rises with the mask rate on five of six
  datasets (more holes, more weight on the refinement pass), and is lowest on the hourly ETT sets
  (0.48–0.70) and near 1 elsewhere.
- **One imputation configuration is now within 0.7% of choosing per dataset** (0.0651 against
  0.0646; before the gate, 8% behind). It beats the local TimeMixer on four of six datasets (ETTh2,
  ETTm1, ETTm2, Weather) and on the mean of the five other than Electricity (0.0572 against 0.0586),
  with 42 to 582 parameters. Electricity stays 30.5% behind.
- **Prediction 3 was refuted: the phase-mixing gate does not work.** Electricity reached 0.1159
  against the 0.115 refutation line; on average the gate changes nothing (0.0688 against 0.0689).
  Phase mixing helps or is neutral at masks 0.125 and 0.25 on every dataset (−4.1% to +2.5%), but
  at 0.375 and 0.5 it costs +31% to +44% on ETTh2 and +16% to +26% on Electricity, gated or not.
  There the gate moves well away from 0 and training ends at the same worse result.
- **Prediction 4 held as stated, but the gap it was meant to close moved.** Both gated (0.0688) is
  below the old per-task 0.0716. But the per-task form improved too, to 0.0651, so the fixed
  configuration (0.0744 → 0.0689, −7.4%) is now 5.8% behind it, all of it from phase mixing.
- **Fixed configuration against the baselines, imputation:** −17.8% against SparseTSF (was −11.2%),
  +10.8% against TimeMixer (was +19.6%). No other task changes, since the refinement gate acts only
  on imputation.
- **A side observation about ETTh2.** Some runs sit on a training-loss plateau around 0.11–0.12
  and either escape late in training or never do. That is half the loss of predicting each window's
  mean (0.24), so it is a partial solution rather than the trivial one. Which side a run lands on
  is partly chance, one more reason for seed repeats.

**Reports changed.** `results_sparsetsf_plus_ablation.md` (three variants, a mean table on test and
validation, learned gate values); `results_sparsetsf_plus_suite.md` (both versions now use gated
refinement; findings); `results_sparsetsf_plus_report.md` (headline, new Section 4 subsection on
gates, Section 5 imputation re-selected on validation, Sections 6–8); `results_sparsetsf_all_tasks.md`
(run counts); the thesis document (new Section 17 on gates; Summary, Sections 7, 14, 15 and 16 updated
to the gated numbers, with the first-run values kept where they explain the history).

**Next.** Make phase mixing safe on inputs with missing values. The harm appearing only when many
points are missing suggests that mixing along the phase axis spreads the holes. Candidates: apply
phase mixing only to complete inputs (in imputation, the refinement pass), or make it mask-aware.
If neither removes the high-mask harm, the holes are not the mechanism. Then seed repeats.

---

## 2026-10-21 — SparseTSFPlus as one model on all eight tasks

**Planned.** Until now every SparseTSFPlus result chose its units per dataset, which is an
ablation, not a model. The thesis's target is one fixed configuration. So SparseTSFPlus was run on
every task and dataset the other models have, in two versions:
- **fixed**: one configuration for everything;
- **per task**: each task's winning unit set, applied to all its datasets.

The gap between the two is the current cost of generality.

**Done.** 220 new runs overnight (about 8 hours) plus 40 same-session energy reference runs.
- **Fixed version:** learned-trend decomposition and phase mixing in the backbone; refinement and
  mask-aware aggregation when inputs have missing values.
- **Both versions:** SparseTSF's published periods and training settings, so each row differs from
  SparseTSF only in the units.
- **Reused, not re-run:** imputation, classification, per-task PEMS and per-task anomaly, where
  identical runs already existed.

**Found.**
- **One configuration costs almost nothing.** The fixed version is within 0.4% of the per-task one
  on every task except imputation (+3.9%). That gap comes from phase mixing on Electricity
  imputation (+17.7%).
- **Against SparseTSF it improves or matches every task:** PEMS −25.4%, imputation −11.2%, M4 OWA
  −6.0%, few-shot −2.0%, long-term −1.8%, zero-shot −0.4%. Anomaly detection and classification
  are unchanged.
- **Against TimeMixer (local)** it wins few-shot (−12.1%), zero-shot (−12.7%) and anomaly detection
  (+4.5 F1). It loses long-term (+8.4%), M4 (+8.7%), PEMS (+45.3%), imputation (+19.6%) and
  classification (−1.5 points).
- **Not yet safe by default:** refinement on ETTh2 imputation and phase mixing on Electricity
  imputation, the two places a fixed SparseTSFPlus still loses to a tuned one.
- **Energy:** the fixed version uses 1.35× SparseTSF's energy and TimeMixer 18.9×
  (14× SparseTSFPlus), with all three measured in the same session. Today's SparseTSF total is 42%
  below July's, while TimeMixer's moved 5%, so only same-session ratios are reliable. The July
  SparseTSF/TimeMixer ratio of 8.7% overstates SparseTSF's cost; same-session it is 5.3%.

**Reports changed.** New `results_sparsetsf_plus_suite.md`; the SparseTSFPlus report (new
headline, Section 6, runtime, next steps); the thesis document (new Section 16 and Summary
bullet); `results_sparsetsf_all_tasks.md` (pointer and coverage
tables).

**Next.** Gates that let refinement and phase mixing switch themselves off, measured with the same
suite runner; then seed repeats.

---

## 2026-10-21 — Runs-and-coverage tables

**Planned.** One place that shows every run in the study, per model and task, against the paper's
protocol, so coverage and gaps are visible at a glance.

**Done.** A "Runs and coverage" section in `results_sparsetsf_all_tasks.md`, counted from the logs
by `benchmarks/tools/build_run_inventory.py`, so it stays current after every new run.

**Found.** 559 finished runs: SparseTSF 187, SparseTSFPlus 185, TimeMixer 146, TimeMixer++ 41.
Three are unfinished and ruled out on this laptop (TimeMixer Traffic 336/720, TimeMixer++ Solar
192). SparseTSF and TimeMixer cover every obtainable cell of the paper's protocol; TimeMixer++ is
paper-only outside long-term forecasting. SparseTSFPlus has unit-variant runs only.

**Reports changed.** `results_sparsetsf_all_tasks.md`.

**Next.** The single-configuration SparseTSFPlus sweep, which is the column still missing.

---

## 2026-10-21 — Anomaly detection extended to MSL and SMAP (4 of 5 datasets)

**Planned.** Anomaly detection covered only 2 of the paper's 5 datasets. MSL and SMAP had been
unobtainable because NASA's original download now returns HTTP 403. A Kaggle mirror of that release
makes them available again.

**Done.** Rebuilt MSL and SMAP from the mirror with the standard (OmniAnomaly) preprocessing. The
rebuilt train and test sizes and channel counts equal the published ones exactly (MSL 58,317 /
73,729 steps × 55; SMAP 135,183 / 427,617 × 25). Ran SparseTSF and TimeMixer on both (4 runs, same
protocol as PSM and SMD). SWaT stays out: it is gated by iTrust, and a Kaggle re-upload was rejected
because it has no redistribution right and appears to lack the standard attack-period test split.

**Found.**
- **Mixed result.** TimeMixer has the higher F1 on three of four datasets (PSM, SMD, SMAP 73.05
  against 66.76), but SparseTSF has the higher average (79.96 against 75.49). That comes entirely from
  MSL, where the untuned TimeMixer catches only a third of the anomalies (F1 48.02 against 77.80).
- **Against the published TimeMixer++ on the same four datasets** (85.75), SparseTSF is 5.8 points
  behind.
- **Recall is the weak side for both models on the NASA data:** SparseTSF recall 69.8 (MSL) and
  53.2 (SMAP) at precision ~88–89.
- **Caveat:** the rebuilt anomaly rates (10.53% MSL, 12.79% SMAP) are slightly below the commonly
  cited 10.72% and 13.13%, probably a revision of NASA's label file, so F1 on these two is close to
  but not strictly identical with the papers' data.

**Reports changed.** `results_comparison_anomaly.md`, `datasets_reference.md`, the cross-task
summary, the SparseTSFPlus report's 8-task table, `results_param_counts.md`, and the thesis
document.

**Next.** Request SWaT from iTrust to complete the five. Otherwise unchanged: gated refinement, then
one frozen SparseTSFPlus configuration across all eight tasks.

---

## 2026-10-12 — Same-machine baselines on every task (second half)

**Planned.** Finish the queue from the first half: TimeMixer's missing imputation cells
(Electricity), TimeMixer classification, and few-shot for TimeMixer, plus SparseTSF's
Electricity-720 few-shot cell that previously ran out of RAM.

**Done.** 39 more runs, one at a time: TimeMixer imputation on Electricity (4 mask rates), TimeMixer
classification on the 10 UEA datasets, TimeMixer few-shot on 6 datasets × 4 horizons, and SparseTSF
few-shot on Electricity at horizon 720 (batch-by-batch metrics). TimeMixer is now run locally on
all eight tasks. Classification needed channel mixing on, because TimeMixer's classification code
has no per-channel path.

**Found.**
- **Imputation, all six datasets:** TimeMixer wins all 24 cells (mean masked MSE 0.0622 against
  SparseTSF's 0.0838) and trains about 6.5× slower. This changes a SparseTSFPlus claim: it matches
  TimeMixer on five of six datasets, but on Electricity it stays 31% behind (0.105 against 0.080).
  Over all six it is 6.5% behind, still closing 81% of SparseTSF's gap.
- **Classification:** an untuned TimeMixer averages 62.37% against SparseTSF's 60.90%. Both are
  far below TimeMixer++'s published 75.9%. TimeMixer wins Heartbeat (75.6 against 54.6) and
  SelfRegulationSCP1; SparseTSF wins JapaneseVowels and PEMS-SF. Part of the published gap is
  likely per-dataset tuning, but an untuned TimeMixer cannot settle that.
- **Few-shot:** SparseTSF leads on ETT (0.441 against 0.513), TimeMixer on Weather (0.254 against
  0.271) and Electricity (0.209 against 0.224), though SparseTSF wins Electricity at horizon 720.
  The local TimeMixer ETT figure is 13% worse than its published 0.453, the one task where the
  local run falls short of the paper. The shared batch size of 32 likely does not suit its
  full-data learning rate.

**Reports changed.** `results_comparison_{imputation,classification,fewshot}.md`, the cross-task
summary in `results_sparsetsf_all_tasks.md`, the SparseTSFPlus report (Section 1 now shows local
TimeMixer on every task; the imputation headline is corrected to five of six datasets),
`results_sparsetsf_plus_ablation.md`, `results_param_counts.md`, and the thesis document.

**Next.** All comparability runs that do not depend on SparseTSFPlus are done. What remains needs
a frozen SparseTSFPlus configuration first: the gated refinement, then one configuration across all
eight tasks, the matched energy sweep, and seed repeats.

---

## 2026-10-12 — Same-machine baselines on every task (first half)

**Planned.** Close the comparability gaps from 2026-10-11 that do not depend on freezing a
SparseTSFPlus configuration. On six of the eight tasks SparseTSF was compared only with published
TimeMixer figures (other hardware, the authors' tuning), so TimeMixer is now run locally on each.
SparseTSF also gets the three long-term datasets it was missing.

**Done.** 48 runs so far, one at a time so the reported runtimes stay clean:
- TimeMixer locally on M4 (6 subsets) and PEMS (4 datasets) with its upstream script settings.
- TimeMixer locally on zero-shot (24 transfers) with the source dataset's long-term configuration.
- TimeMixer locally on anomaly detection (PSM, SMD). TimeMixer publishes no anomaly settings, so it
  uses its own ETTh1 architecture with the task's standard training protocol, the same rule as the
  earlier local TimeMixer imputation runs.
- SparseTSF on Solar-Energy, Electricity and Traffic at seq_len 96 (12 runs), with its upstream
  settings.
- An opt-in setting that computes test metrics batch by batch, because Electricity and Traffic at
  long horizons exceed 16 GB of RAM. It agrees with the default path to 6–7 significant digits.
- Parameter counts for every model, task and dataset that has run, rebuilt from each run's own log.

**Found.**
- **Local TimeMixer reproduces its published numbers:** M4 OWA 0.841 against 0.840, PEMS MAE 17.51
  against 17.41, zero-shot mean MSE 0.448 against 0.467. The earlier comparisons against published
  TimeMixer figures therefore hold on this machine.
- **Zero-shot is now a same-machine win:** SparseTSF beats local TimeMixer on all six transfer
  pairs (mean MSE 0.393 against 0.448). This is the strongest generalisation result in the study.
- **Anomaly:** TimeMixer F1 90.45 against SparseTSF 87.64 over PSM and SMD. Most of the gap is SMD
  recall (81.6 against 73.9), the identity-map problem studied in the SparseTSFPlus ablation.
- **M4:** SparseTSF is 15.6% behind on OWA (0.972 against 0.841), with 13–33 parameters against
  10,525–100,069.
- **PEMS:** SparseTSF's MAE is 95% higher (34.13 against 17.51). TimeMixer has 0.87–1.24 M
  parameters against SparseTSF's 21, yet trains in only about twice the time (1,404 s against
  666 s). A parameter ratio does not translate into a runtime or energy ratio.
- **Solar, Electricity, Traffic:** at seq_len 96 SparseTSF trails TimeMixer at every horizon, which
  fits its having only four 24-hour periods of context.

**Reports changed.** `results_comparison_{m4,pems,zeroshot,anomaly,sparsetsf}.md` and the new
`results_param_counts.md`.

**Next.** The queue continues with TimeMixer imputation on Electricity, TimeMixer classification,
and few-shot (TimeMixer, plus SparseTSF's missing Electricity-720 cell). After that, the gated
refinement and a single frozen SparseTSFPlus configuration.

---

## 2026-10-11 — Unit ablations finished on every target task; log started

**Planned.** Measure every unit grafted onto SparseTSF so far, on the tasks it was built for, and
check each against plain SparseTSF under identical settings. The aim was to know which units earn
their parameters before designing a single fixed configuration.

**Done.** 97 runs since the last commit (2026-09-18), bringing the SparseTSFPlus study to 204
runs (about 4 h 15 min of training):
- forecasting ablation on ETTh1, ETTm1 and Weather, with a horizon-96 screen of 8 variants and the
  validation winner at all four horizons (36 runs)
- PEMS03/04/07/08 short-term (8 runs)
- leave-one-out reconstruction for anomaly detection on PSM and SMD (4 runs)
- imputation extended from 2 to 6 datasets: ETTh2, ETTm2, Weather and Electricity added (49 runs)

Every control row reproduced the published SparseTSF number exactly.

**Found.**
- **Imputation:** 20.9% lower error than SparseTSF across six datasets. On the five datasets where
  TimeMixer was run locally, the mean is 0.0584 against TimeMixer's 0.0586, i.e. parity at 41–597
  parameters against ~75,500. This needs the configuration chosen per dataset, and on ETTh2
  validation chose no units at all. With one configuration fixed everywhere, SparseTSFPlus is
  10–13% behind TimeMixer instead.
- **PEMS:** phase-axis mixing at period 4 lowers mean MAE from 34.13 to 25.50 (−25%) at 93
  parameters. That closes 52% of the gap to TimeMixer's published 17.41, and comes within 0.6% of
  a dense 1,153-parameter model.
- **Long-term forecasting:** the learned-trend decomposition was best on validation for all three
  datasets. Over 12 cells, mean MSE goes from 0.3754 to 0.3715 (−1.0%), which is within the range
  one seed cannot settle. Holding the trend flat made every dataset worse (+13% to +27%), and
  combining phase mixing with decomposition added nothing.
- **Anomaly:** leave-one-out reconstruction is rejected. F1 changed from 95.26 to 93.23 on PSM and
  from 80.02 to 80.55 on SMD.

**Reports changed.** `results_sparsetsf_plus_report.md`, `results_sparsetsf_plus_ablation.md`,
new `results_sparsetsf_plus_forecast.md` and `results_sparsetsf_plus_anomaly.md`, and the thesis
document.

**Next.**
1. Make the refinement pass safe by default, with a gate that can learn to switch it off. ETTh2,
   where refinement hurts, is the main reason one fixed configuration is 10–13% behind TimeMixer
   on imputation.
2. Freeze one SparseTSFPlus configuration and run it over the full 8-task suite.
3. Close the comparability gaps below.

**Comparability gaps (the runs still missing).**

| Gap | Runs | Why it matters |
|---|---:|---|
| SparseTSFPlus, frozen configuration, full 8-task suite | ~113 | Without it, SparseTSFPlus has no one-model result to compare |
| SparseTSFPlus in the matched carbon sweep | 20 | No energy figure for the new model yet |
| Parameter counts for every model on every task | 0 (no training) | `count_params.py` covers long-term ETTh only |
| Repeat headline cells over 2 more seeds | ~40–60 | Every run so far uses one seed |
| SparseTSF long-term on Solar, Electricity, Traffic | 12 | Paper protocol has 8 datasets; SparseTSF has 5 |
| TimeMixer local on M4, PEMS, few-shot, zero-shot, anomaly, classification | ~70 | Those six tasks compare against published numbers only, so there is no same-machine accuracy or energy |
| TimeMixer imputation on Electricity | 4 | Imputation mean is over 5 datasets for TimeMixer, 6 for SparseTSF |

Not feasible here; recorded as limitations:
- TimeMixer Traffic 336/720: Traffic 192 alone took 25 h.
- TimeMixer++ Electricity, Traffic and Solar 192–720.
- MSL, SMAP and SWaT: the data is unobtainable.
- Few-shot Electricity at horizon 720: exceeds RAM.
