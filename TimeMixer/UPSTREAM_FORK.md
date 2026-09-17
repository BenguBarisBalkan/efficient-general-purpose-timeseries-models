# Provenance — relation to upstream TimeMixer

This directory started as a clone of the official TimeMixer repository and was then extended
into a three-model benchmark. The original clone's `.git` directory has been moved out of the
tree (see *Backup* below) so that the whole workspace can live in a single repository of its own;
this file records everything that history was carrying.

| | |
|---|---|
| **Upstream remote** | `https://github.com/kwuking/TimeMixer.git` |
| **Base commit** | `e24610583b36fdd8c76cc17a8df4e65759a5f460` (`e246105`, 2025-10-05, "Merge pull request #187 from JasonAlexTan/main") |
| **Local delta to upstream files** | 10 files, +147 / −19 lines — captured in `upstream-fork.patch` |

## Modified upstream files

| File | Why |
|---|---|
| `run.py` | `--track_emissions` (lazy CodeCarbon import), `--percent`, extra model wiring |
| `exp/exp_basic.py` | registers `TimeMixerPP` and `SparseTSF` in `model_dict` |
| `exp/exp_long_term_forecasting.py` | TimeMixer++ NaN stability: skip non-finite loss batches, skip the optimizer step on a non-finite grad norm |
| `exp/exp_classification.py` | `optim.RAdam` fallback to Adam (absent in torch 1.7.1) |
| `exp/exp_imputation.py`, `exp/exp_anomaly_detection.py` | write their `result_*.txt` dumps into `logs/` instead of the repo root |
| `data_provider/data_factory.py`, `data_provider/data_loader.py` | `--percent` few-shot truncation, forwarded only when < 100 |
| `utils/tools.py` | `EarlyStopping` treats a non-finite val loss as non-improving |
| `requirements.txt` | pinned to the local venv (torch 1.7.1+cu110, CodeCarbon) |

## Added on top (not upstream)

`implementations/timemixer_pp/` (TimeMixer++ reimplementation), `implementations/sparsetsf/` (vendored, Apache-2.0),
`models/TimeMixerPP.py` + `models/SparseTSF.py` (entry-point shims), `benchmarks/` (the runner
harness), `reports/` (the write-ups), `logs/`, and the `scripts/**/*.bat` Windows ports of
upstream's shell scripts.

## How to diff against upstream again

```bash
git remote add upstream https://github.com/kwuking/TimeMixer.git
git fetch upstream e24610583b36fdd8c76cc17a8df4e65759a5f460
git diff FETCH_HEAD -- run.py exp/ data_provider/ utils/ requirements.txt
```

## Backup

The original clone's git history (5.6 MB, upstream commits only — it contained none of the work
above, all of which was uncommitted) was moved to
`../../timemixer-upstream-git-backup/` — a sibling of the workspace folder, deliberately outside
any repository. It is re-obtainable at any time by cloning the upstream remote, so it can be
deleted once you are comfortable.
