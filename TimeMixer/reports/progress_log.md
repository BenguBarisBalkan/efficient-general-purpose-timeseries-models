# Progress log

Running record for the biweekly supervisor meetings, newest entry first. Each entry covers what
was planned and why, what was done, what was found (with the baseline each number is compared
against), which reports changed, and what comes next. It records ideas and evidence, not
implementation. The rules for keeping it are in the root `CLAUDE.md` ("Thesis context").

Numbers are **local** (this machine, this protocol) unless marked **published**.

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
