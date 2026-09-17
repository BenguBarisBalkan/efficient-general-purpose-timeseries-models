# How SparseTSF was adapted to every task

SparseTSF (ICML'24) is published as a **forecasting-only** model. This document explains how it was
extended to cover all 8 task categories the TimeMixer++ paper evaluates — what had to change in the
model, what was solved purely in the harness, and why each choice was made.

The whole model-side change lives in one file: [`models/SparseTSF.py`](../models/SparseTSF.py).

---

## 1. What SparseTSF natively does

Its mechanism is **Cross-Period Sparse Forecasting** — three steps, and everything we built reuses them:

1. **Per-channel mean removal.** Subtract each channel's mean over time (a lightweight instance norm).
2. **Conv aggregation.** A depthwise 1-D convolution with kernel `1 + 2·⌊period_len/2⌋` smooths each
   channel along time, plus a residual connection. This is the only place neighbouring timesteps
   interact.
3. **Period folding + one shared linear.** Reshape the length-`S` series into
   `seg_num = S / period_len` segments, transpose so that the *same phase within each period* lines
   up, then apply **one shared `Linear(seg_num_x → seg_num_y)`** across all phases and all channels.

The key structural facts that drive every decision below:

- It is **channel-independent** — channels are folded into the batch dimension and never interact.
- `period_len` **must divide** the sequence length, because of the reshape in step 3.
- It has **no decoder** and ignores time-feature marks entirely.
- Upstream's signature is `forward(self, x)` — one argument.

### 1.1 Worked example: the period fold, step by step

Everything in this document follows from this one transformation. Take ETTh1 with `seq_len=96`
(4 days of hourly data), `period_len=24`, 7 channels:

```
x                     (B, 96, 7)     96 hourly steps, 7 channels
 └ subtract mean      (B, 96, 7)     per-channel, over time
 └ permute            (B,  7, 96)    channels first
 └ conv1d + residual  (B,  7, 96)    smooth along time (the only local mixing)
 └ reshape        (B*7,  4, 24)      ← 4 days × 24 hours
 └ permute        (B*7, 24,  4)      ← 24 hours × 4 days
                        ↑    ↑
                    phase    which
                 in period   period
```

That final permute is the entire idea. Afterwards the **last dimension indexes periods (days)**, and
each of the 24 rows is one *hour-of-day*.

Now apply `Linear(4 → 4)` on the last dimension. PyTorch applies a linear layer across the last axis,
so for **hour 9** it receives

```
[ day1@09:00, day2@09:00, day3@09:00, day4@09:00 ]
```

and maps it to four new values. The **same 16 weights** are reused for all 24 hours and all 7
channels — that is where the parameter count collapses.

This is what "**cross-period sparse**" means: *a time point is predicted from the same phase in other
periods.* 09:00 is predicted from other 09:00s, never from 15:00. That inductive bias is the whole
model, and it is why ~925 parameters can be competitive on periodic data — and equally why the model
struggles when a dataset has no clean period to fold along.

**Consequence:** the fold is a `reshape`, so `period_len` must divide the sequence length exactly.
Every divisibility rule in this document traces back to this single line:

```python
x = x.reshape(-1, self.seg_num, self.period_len).permute(0, 2, 1)
```

---

## 2. The integration problem

This repo is a Time-Series-Library (TSLib) scaffold. Each task has its own experiment loop in
`exp/`, and each calls the model with a **different signature** and expects a **different output
shape**. That contract is what the adaptation had to satisfy:

| Task | Loop calls the model as | Must return |
|---|---|---|
| long / short forecast | `model(x_enc, x_mark, x_dec, x_mark_dec)` | `(B, pred_len, C)` |
| imputation | `model(inp, x_mark, None, None, mask)` | `(B, seq_len, C)` |
| anomaly detection | `model(x, None, None, None)` | `(B, seq_len, C)` |
| classification | `model(x, padding_mask, None, None)` | `(B, num_class)` |

**Design decision: adapt the model, not the loops.** `exp/exp_*.py` and the registry in
`exp/exp_basic.py` are shared by all three models in this study, so changing them to suit SparseTSF
would risk the existing TimeMixer / TimeMixer++ results. Instead `models/SparseTSF.py` became a
**task-aware adapter** that branches on `configs.task_name`, absorbing the differences internally.
The loops never learned that SparseTSF is unusual.

**Design decision: never touch the vendored core.** `SparseTSF_model/model.py` is upstream code
(Apache-2.0). The forecasting path simply delegates to it:

```python
if self.task_name in _FORECAST_TASKS:
    self.core = SparseTSFCore(configs)      # __init__
...
return self.core(x_enc)                      # forward
```

This guarantees forecasting results are reproduced **bit-for-bit** (verified: identical parameter
count, and outputs `torch.allclose` to calling the bare core) and keeps the licence provenance clean.

---

## 3. Task by task

### 3.1 Long-term forecasting — *no change needed*

Already the model's native job. The adapter only bridges the 1-arg → 4-arg signature and drops the
unused `x_mark` / `x_dec`.

`period_len` must divide **both** `seq_len` and `pred_len`. Per-dataset values follow upstream:
**24** for hourly data (ETTh, Electricity — the daily cycle) and **4** for 15-min ETTm / 10-min Weather.

### 3.2 Univariate short-term forecasting (M4) — *no model change, a config problem*

Structurally identical to long-term, so it reuses the same delegated path. The real work was choosing
a per-subset `period_len` that divides both the horizon **and** `seq_len = 2 × horizon`:

| Subset | Horizon | `period_len` |
|---|---:|---:|
| Yearly | 6 | 2 |
| Quarterly | 8 | 4 |
| Monthly | 18 | 6 |
| Weekly | 13 | **13** |
| Daily | 14 | 7 |
| Hourly | 48 | 24 |

Weekly's horizon of 13 is **prime**, so the only non-trivial divisor is 13 itself — meaning that
subset gets exactly one segment per series and the "cross-period" mechanism degenerates. A real
structural limitation, not a tuning choice.

### 3.3 Multivariate short-term forecasting (PEMS) — *no model change, harness plumbing*

Also delegated. Two harness facts had to be discovered rather than assumed:

- PEMS runs under **`task_name=long_term_forecast`**, not the short-term class — `Exp_Short_Term_Forecast`
  is hard-wired to M4 (it calls `last_insample_window()`, which `Dataset_PEMS` doesn't have). The
  upstream PEMS scripts do the same.
- The task is scored with **MAE / MAPE / RMSE**, but the loop only printed `mse:` / `mae:`. The values
  were already computed inside `metric()` — just never printed. Adding them required care:

> **The uppercase trap.** Every benchmark runner regexes lowercase `mse:` and takes the *last* match.
> A lowercase `rmse:` **contains `mse:` as a substring**, so printing it would have silently
> overwritten MSE in every existing results table. The new line is therefore
> `print('RMSE:{}, MAPE:{}, MSPE:{}')` — **uppercase**, invisible to those parsers. Verified with a
> regex test before running anything.

### 3.4 Imputation — *new head: cross-period reconstruction*

The first genuinely new head. Forecasting maps `seg_num_x → seg_num_y` (past periods → future
periods). Reconstruction is the natural analogue: map the sequence **onto itself**,
`seg_num → seg_num`, so each within-period phase is rebuilt from the same phase in every other period.

```python
self.seg_num = self.seq_len // self.period_len
self.recon_linear = nn.Linear(self.seg_num, self.seg_num, bias=False)
```

Concretely, with ETTh1 at `seq_len=96` / `period_len=24`, the fold gives `(B*7, 24, 4)` exactly as in
§1.1, and `Linear(4 → 4)` rebuilds each day's 09:00 value from all four days' 09:00 values.

#### Why the mean has to be mask-aware

This is the one genuinely subtle part of the head. The imputation loop hands the model a tensor in
which **masked entries are literal zeros** (`inp = batch_x.masked_fill(mask == 0, 0)`). A plain mean
would therefore be wrong in two separate ways:

```python
seq_mean = x_enc.mean(dim=1)      # WRONG
```

1. **It is biased toward zero.** At a 50% mask rate roughly half the summands are artificial zeros,
   so the computed mean is roughly half the true mean.
2. **It leaks the mask.** The size of that bias depends on *how many* points were masked, so the
   statistic itself carries information about the mask — a subtle form of leakage that also makes the
   model's behaviour depend on the mask rate it happened to be trained at.

The head therefore averages over **observed positions only**:

```python
denom    = torch.clamp(mask.sum(dim=1, keepdim=True), min=1.0)   # how many were observed
seq_mean = (x_enc * mask).sum(dim=1, keepdim=True) / denom        # mean of observed values only
x        = (x_enc - seq_mean) * mask                              # re-zero after centring
```

Each line earns its place:

- `x_enc * mask` — zeroes masked slots before summing, so only real observations contribute.
- `mask.sum(dim=1)` — the true count of observations, per channel, per sample.
- `clamp(min=1.0)` — guards the degenerate case where a channel is **fully masked**; without it the
  division is 0/0 and produces NaN that would propagate through the whole batch.
- **The trailing `* mask` is essential.** After centring, a masked slot holds `0 - seq_mean`, which is
  non-zero — i.e. it would enter the convolution as a *fabricated observation* with value `-seq_mean`.
  Re-zeroing keeps masked positions neutral.

Finally the mean is added back at the end of `_reconstruct()`, so the model only ever has to learn
the mean-removed structure.

`period_len` now only has to divide `seq_len` (there is no `pred_len`).

### 3.5 Anomaly detection — *same head, simpler normalisation*

Anomaly detection here is reconstruction-based: rebuild the window, score by reconstruction error,
threshold at a percentile, then apply the standard point-adjustment before computing F1. The model
side is therefore **the same reconstruction path** as imputation, with a plain (non-masked) mean,
since nothing is hidden:

```python
def _anomaly(self, x_enc):
    seq_mean = x_enc.mean(dim=1, keepdim=True)
    return self._reconstruct(x_enc - seq_mean, seq_mean)
```

With window 100, `period_len=25` divides it cleanly (4 segments of 25).

#### How a reconstruction becomes a detection

All of this lives in the **unmodified** `exp_anomaly_detection.py`; the model only supplies the
reconstruction. It matters for interpreting the F1 numbers, so it is worth spelling out:

1. **Score.** Per time point, the anomaly score is the reconstruction error averaged over channels:
   `score = mean(MSE(x, x̂), dim=-1)`.
2. **Threshold.** Scores from the **train and test sets are pooled**, and the cut is taken at a
   percentile set by `--anomaly_ratio` (a *percentage*):

   ```python
   combined_energy = np.concatenate([train_energy, test_energy])
   threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
   pred = (test_energy > threshold).astype(int)
   ```

   So `anomaly_ratio=1.0` flags the top **1%** of scores; `0.5` flags the top 0.5%. This is a
   *prior* on how many anomalies exist, not something the model learns — and note it is calibrated
   using test scores, which is the standard protocol in this literature but is worth stating plainly.

   Observed thresholds in our runs: **PSM 0.669**, **SMD 8.834** (the scales differ because each
   dataset is standardised independently).

3. **Point adjustment.** Before scoring, `utils/tools.py::adjustment` expands any *correctly detected*
   point to cover its **entire** ground-truth anomaly segment: if the model catches even one point of
   a contiguous anomaly, the whole segment counts as detected.

   This is the community-standard protocol (used by Anomaly Transformer, TimesNet, TimeMixer++, and
   therefore required for comparability), but it **inflates F1 substantially** relative to raw
   point-wise scoring. All F1 numbers in this study — and all published numbers quoted alongside
   them — use it.

Per-dataset `anomaly_ratio` follows the standard values: **1.0** for PSM/MSL/SMAP/SWaT, **0.5** for SMD.

### 3.6 Classification — *the head that had to break the pattern*

This is the one place the period mechanism had to be **abandoned**, for a concrete reason: UEA
sequence lengths are arbitrary (26, 62, 144, 1751…) and often prime or awkward, so requiring
`period_len` to divide `seq_len` would make most datasets unrunnable. The head therefore keeps the
conv aggregation but skips the period reshape entirely:

```python
x = (x_enc - x_enc.mean(dim=1, keepdim=True)).permute(0, 2, 1)
x = self._aggregate(x)                      # conv + residual
if padding_mask is not None:
    x = x * padding_mask.unsqueeze(1)       # zero padded steps (variable-length UEA)
x = self.dropout(self.act(x))               # GELU + dropout
return self.projection(x.reshape(x.shape[0], -1))   # flatten -> Linear(enc_in*seq_len, num_class)
```

Two consequences, both stated openly in the results:

1. **Divisibility is waived** for this task alone.
2. **The parameter budget explodes** — `enc_in · seq_len × num_class` (see §5). This is inherent to
   flattening for classification, not a flaw in SparseTSF, but it does mean the "~1k parameters"
   headline does not hold for this task.

Since SparseTSF has no published classification head, this one was written for this study — so its
results are a **floor**, not a tuned upper bound.

### 3.7 Few-shot forecasting — *solved entirely in the data layer*

No model change. A `--percent` flag truncates the **training split only**:

```python
if self.set_type == 0 and getattr(self, 'percent', 100) < 100:
    border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len
```

Deliberate choices: validation/test are untouched (so numbers stay comparable to the full-data
tables), the scaler still fits the full training range (the TSLib convention), and `data_factory`
forwards `percent` **only when < 100**, so loaders that don't accept it (M4/PEMS/Solar) are
completely unaffected on the normal path.

#### Where the 10% comes from

**It was not our choice — it is the benchmark's definition.** The TimeMixer++ paper fixes it in
§4.1.5:

> "we test across 6 diverse datasets, training each model on only **10% of available timesteps**"

and Table 5 is titled *"Few-shot learning on 10% training data"*. The whole point of the task is to
compare against those published numbers, so the ratio is a **protocol constant to be matched, not a
hyperparameter to tune**. Choosing our own value (5%, 20%) would have produced numbers that could
not be placed next to the paper's column at all.

#### Where the convention itself comes from

The TimeMixer++ paper does not *justify* 10% — it inherits an established protocol. Tracing it back:

**Canonical citation:** *"One Fits All: Power General Time Series Analysis by Pretrained LM"*
(GPT4TS, Zhou et al., NeurIPS 2023, [arXiv:2302.11939](https://arxiv.org/abs/2302.11939)) states it
as using **"only a certain percentage (10%, 5%) timesteps of training data"**, with validation and
test portions left unchanged. That is exactly the definition implemented here — including that the
percentage is over **timesteps**, not samples.

**Attribution caveat — do not write "introduced by".** GPT4TS neither claims novelty for this
setting nor cites a source for it; it prefixes the description with *"Similar to traditional
experimental settings"*, i.e. presents it as already-conventional. It is the earliest paper we could
trace using this protocol on the standard LTSF benchmarks, and it is what every later paper follows,
but that makes it the **paper that established/popularised the protocol**, not a demonstrable
inventor. The safe phrasing is *"following the few-shot protocol popularised by GPT4TS (Zhou et al.,
2023) and adopted by Time-LLM and TimeMixer++"*.

**Two different things are called "few-shot" in this literature.** Keep them apart in a lit review:

| Sense | "A shot" is | Typical setup | Lineage |
|---|---|---|---|
| **(A) Cross-series / meta-learning** | a *series* or *task* | train on many source series, adapt to an unseen one with few examples | predates 2023 — meta-learning and transfer-learning work (e.g. N-BEATS-style zero-shot transfer, GP/latent-variable few-shot forecasting) |
| **(B) Data-limited fine-tuning** ← *used here* | a *timestep* | same dataset, same splits, but only X% of the training range | GPT4TS (2023) → Time-LLM → LLM4TS → TimeMixer++ |

Sense (A) is an older and broader research problem; sense (B) is a *benchmark protocol* for testing
data efficiency. This study uses **(B)** exclusively. Note that our zero-shot task is closer in
spirit to (A), but is likewise a protocol (train on A, test on B) rather than meta-learning.

**Adoption:** the same setup was carried forward by Time-LLM (ICLR 2024), LLM4TS, and the
LLM-for-time-series line generally, and then by TimeMixer++. Prediction lengths {96, 192, 336, 720}
averaged into one figure is part of the same convention. So it is a genuine community standard, but
its authority is convention and lineage, not an ablation showing 10% is the right number.

**Where this study deviates from the full standard — state these plainly:**

| Aspect | Community standard | This study |
|---|---|---|
| Ratios reported | **both 5% and 10%** | 10% only |
| Repeats | GPT4TS repeats **3×**, reports the mean | **single run** (fixed seed 2021) |
| Prediction lengths | {96, 192, 336, 720} | same ✅ |
| Val/test splits | untouched | same ✅ |
| Percentage applied to | timesteps | same ✅ |

Neither deviation invalidates the comparison — TimeMixer++'s own Table 5 is 10%-only, which is what
we compare against — but a single run means we have no variance estimate, and no 5% column means no
sensitivity analysis. Both are cheap to add later: few-shot runs take seconds to a few minutes each.

**Precedent for the caveat below.** GPT4TS *excluded the ILI dataset entirely* from its few-shot
experiments because of its "limited quantity that is hard to follow the definition of few-shot" —
i.e. the originators of the protocol already recognised that it degenerates on small datasets. The
window-count analysis in the next subsection is the same phenomenon, quantified for ETTh at long
horizons.

#### "10% of timesteps" is not "10% of training examples"

This is the subtlety that matters most, and it is easy to miss. The truncation is applied to the
**timestep range**, but a training example consumes `seq_len + pred_len` consecutive timesteps. So
the number of usable *windows* shrinks far faster than the timestep count, and the effect is worst on
short datasets at long horizons. Measured on the actual loaders:

| Dataset | pred_len | Full windows | Few-shot windows | Effective ratio |
|---|---:|---:|---:|---:|
| ETTh1 | 96 | 8,449 | 759 | 8.98% |
| **ETTh1** | **720** | 7,825 | **135** | **1.73%** |
| ETTm1 | 96 | 34,369 | 3,351 | 9.75% |
| ETTm1 | 720 | 33,745 | 2,727 | 8.08% |
| Weather | 96 | 36,696 | 3,584 | 9.77% |
| Weather | 720 | 36,072 | 2,960 | 8.21% |

ETTh1's training split is only 8,640 timesteps; 10% of it is 950, of which `96 + 720 = 816` are
consumed by a single window — leaving **135 examples, i.e. 1.7% of the full set**, not 10%. On the
larger ETTm1/Weather splits the ratio stays near 9-10% because the consumed prefix is negligible
relative to the split size.

Three consequences:

1. **"Few-shot" is much harsher than 10% suggests** on ETTh at long horizons — closer to a 2% regime.
2. It explains the results: ETTh1@720 is by far the worst few-shot cell (MSE 0.751).
3. It is the direct cause of the `steps_per_epoch = 0` bug below — 135 windows simply cannot fill a
   256-sample batch.

The formula follows the Time-Series-Library convention, and the `+ self.seq_len` term means the kept
range is slightly more than 10% of raw timesteps (950 of 8,640 = 11.0% for ETTh1). We kept TSLib's
formula rather than "fixing" it, precisely because matching the reference implementation is the point.

> **A bug worth recording.** With 10% of the data *and* `drop_last=True`, the long-horizon ETTh
> configs produce ~135 training windows — which is **0 batches** at SparseTSF's published
> `batch_size=256`. Those runs died with `steps_per_epoch = 0`. Because the 720-horizon configs are
> the *hardest*, losing them silently biased the ETT average **downward** (0.438 vs the correct
> 0.441). Fixed by using `batch_size=32` for every few-shot config. The general lesson: **always
> check for incomplete rows before quoting an average** — a partial average is a wrong average.

### 3.8 Zero-shot transfer — *solved with a checkpoint copy, no code change at all*

`run.py` derives a `setting` string from the arguments; `--is_training 1` **saves** to
`checkpoints/<setting>/checkpoint.pth` and `--is_training 0` **loads** from it. Since the source and
target settings differ only in the `--data` field, the whole task reduces to:

1. Train on dataset **A**.
2. Copy `checkpoints/<setting_A>/checkpoint.pth` → `checkpoints/<setting_B>/`.
3. Run **B** with `--is_training 0` — test only, so the model never sees B's training data.

The one constraint: SparseTSF's linear layer is shaped `(seq_len/period_len → pred_len/period_len)`,
so the **source's `period_len` must be reused for the target**, or the checkpoint won't load. Every
horizon used is divisible by both 24 and 4, so all six pairs are valid. (The reconstructed `setting`
string was verified against real checkpoint directory names before running 24 configs.)

---

## 4. One supporting fix outside the model

`exp/exp_classification.py` used `optim.RAdam`, which **does not exist in torch 1.7.1** — the pinned
version here. Classification would have crashed for *any* model. Fixed with a version-guarded
fallback that keeps RAdam on newer torch:

```python
model_optim = optim.RAdam(...) if hasattr(optim, 'RAdam') else optim.Adam(...)
```

---

## 5. Exact settings used, per task

Every value below is the one actually configured in the corresponding `benchmarks/run_*.py`. All runs
use `model_type=linear` (the <1k-parameter variant), `num_workers=0`, `itr=1`, and Adam.

| Task | seq_len | pred / target | lr | batch | epochs | patience | LR schedule | `period_len` |
|---|---:|---|---:|---:|---:|---:|---|---|
| Long-term forecast | 96 & 720 | {96,192,336,720} | 0.02 | 256 | 30 | 5 | `type3` | 24 (h) / 4 (m,W) |
| Short-term M4 | 2×horizon | horizon | 0.01 | 16 | 10 | 5 | `TST` | per subset (below) |
| Short-term PEMS | 96 | 12 | 0.01 | 16 | 10 | 5 | `TST` | 12 |
| Imputation | 96 | 96 (recon) | 0.001 | 16 | 10 | default | `TST` | 24 (h) / 4 (m,W) |
| Few-shot | 96 | {96,192,336,720} | 0.02 | **32** | 30 | 5 | `type3` | 24 (h) / 4 (m,W) |
| Zero-shot | 96 | {96,192,336,720} | 0.02 | 256 | 30 | 5 | `type3` | **source's** |
| Anomaly | 100 | 100 (recon) | 0.0001 | 128 | 3 | default | `TST` | 25 |
| Classification | dataset max | `num_class` | 0.001 | 16 | 30 | 10 | every 5 epochs | unused |

**Schedules.** `TST` is PyTorch `OneCycleLR` (warm-up then anneal, `pct_start=0.2`) and is `run.py`'s
default; `type3` holds the LR flat for 2 epochs then decays ×0.9 per epoch — SparseTSF's published
choice for forecasting. Classification uses its own rule in `exp_classification.py` (adjust every 5
epochs) and additionally clips gradients at `max_norm=4.0`.

**Task-specific thresholds and rates**

| Setting | Value | Where it applies |
|---|---|---|
| `mask_rate` | 0.125, 0.25, 0.375, 0.5 | Imputation — fraction of entries hidden |
| `anomaly_ratio` | 1.0 (PSM), 0.5 (SMD) | Anomaly — top-*x*% of scores flagged |
| Observed threshold | PSM 0.669, SMD 8.834 | Anomaly — the resulting score cut |
| `percent` | 10 | Few-shot — % of the train split kept |
| M4 `period_len` | Y 2, Q 4, M 6, W 13, D 7, H 24 | Must divide horizon *and* 2×horizon |

**Batch-size exceptions, and why.** Two tasks deviate from SparseTSF's published `batch_size=256`:

- **Few-shot → 32.** At 256, the long-horizon ETTh configs have ~135 windows and `drop_last=True`
  discards all of them → 0 batches → crash (see §3.7).
- **Imputation/M4/PEMS/classification → 16, anomaly → 128.** Chosen to fit the 4 GB GPU on the wide
  datasets (Electricity 321ch, PEMS-SF 963ch) while keeping enough steps per epoch on the small ones.

## 6. What this costs in parameters

Measured directly from the built models:

| Task | Config | Parameters | Breakdown |
|---|---|---:|---|
| Forecast | ETTh 720→720, `period_len=24` | **925** | conv 25 + linear 900 |
| Forecast | ETTh 96→96, `period_len=24` | **41** | conv 25 + linear 16 |
| Imputation | ETTh `seq_len=96` | **41** | conv 25 + recon-linear 16 |
| Anomaly | PSM `seq_len=100`, `period_len=25` | **41** | conv 25 + recon-linear 16 |
| Classification | JapaneseVowels (12ch × 26 × 9cls) | **2,818** | projection 2,817 |
| Classification | PEMS-SF (963ch × 144 × 7cls) | **970,712** | projection 970,711 |

The 925 figure reproduces the paper's "1k parameters" claim exactly. **Imputation and anomaly
detection inherit that frugality** — 41 parameters to reconstruct a window. Only classification
breaks it, and only because flattening `enc_in × seq_len` into a class projection is unavoidable.

---

## 7. Summary of the approach

| Task | What changed | Where |
|---|---|---|
| Long-term forecast | nothing (delegated to vendored core) | — |
| Short-term M4 | per-subset `period_len` only | runner |
| Short-term PEMS | extra uppercase metric print | `exp_long_term_forecasting.py` |
| Imputation | **new head** — mask-aware mean + `seg_num→seg_num` linear | `models/SparseTSF.py` |
| Anomaly | **new head** — same reconstruction, plain mean | `models/SparseTSF.py` |
| Classification | **new head** — conv → mask → flatten → linear (no period reshape) | `models/SparseTSF.py` |
| Few-shot | `--percent` train-split truncation | `data_loader.py`, `data_factory.py`, `run.py` |
| Zero-shot | checkpoint copy between settings | runner only |

The guiding principle throughout: **push the adaptation into the model or the runner, never into the
shared experiment loops** — so that the three-model comparison stays valid and the vendored
forecasting core stays byte-identical.

See [datasets_reference.md](datasets_reference.md) for what each dataset is, and
[results_sparsetsf_all_tasks.md](results_sparsetsf_all_tasks.md) for the numbers.
