# Robust Adaptive Thresholding for LSTM Autoencoder-Based Anomaly Detection in Streaming Time Series

**Zain Hassan · MS Data Science · FAST-NUCES Islamabad · Reg. 26I-8015**

## Overview

An LSTM Autoencoder learns normal behaviour in a time series and scores every time step by how badly it rebuilds the most recent window: the *reconstruction error*. A **threshold rule** then turns those scores into alarms. This project tests whether a **moving median + λ·MAD threshold** (the proposed method, P1) works better than fixed thresholds (B1, B2) and the base paper's mean + 3σ rule (B3, B4).

The experiment is controlled: every rule gets the same data, model and reconstruction errors, so only the threshold rule changes. It uses the complete **Numenta Anomaly Benchmark (NAB)**: 58 time series with 116 labelled anomaly windows. All figures, tables and dashboards in this repository are produced by the pipeline code from the experiment outputs.

## Main findings

On the primary full-corpus protocol (52 files with 107 anomaly windows, plus 6 anomaly-free files):

- **The proposed threshold does not beat the baselines.** P1 has composite F1 0.433 and NAB score 30.3. The fixed baselines reach 0.473 / 39.0 (B1) and 0.468 / 43.5 (B2). H1 and H3 are not supported.
- **P1 finds more anomaly windows but raises more false alarms.** Its event recall is 0.815 against 0.717 and 0.692. Its high-variance false-positive rate is 0.255 against 0.172 and 0.134, significantly worse than B2 (p = 0.008). H2 does not hold on average.
- **The model is the bottleneck.** The LSTM-AE score separates anomalies only weakly (median ROC-AUC 0.61), so the differences between threshold rules are small and mostly not significant.
- **Why median/MAD struggles.** The MAD of training errors collapses to zero on 4 of 52 files and is a median 2.2× smaller than the σ-based spread elsewhere, so one λ does not transfer between files.
- **The best rule overall is a rolling mean + k·σ** (ablation A3: composite F1 0.514, NAB 47.1). P1 does relatively well on AWS CloudWatch and Traffic, and poorly on Twitter and the artificial series.

The proposal's 60/20/20 split (18 files, 25 anomaly windows) reaches the same overall conclusion.

## Results summary

The tables below are written automatically by `scripts/make_report.py`.

<!-- RESULTS_START -->
**Full-corpus protocol (primary)** — 52 files, 107 anomaly windows, W* = 30

| Method | Composite F1 ↑ | Point F1 ↑ | Precision ↑ | Recall ↑ | Event recall ↑ | FPR ↓ | High-var FPR ↓ | Alerts ↓ | NAB std ↑ | NAB low-FP ↑ | NAB low-FN ↑ | Delay ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 Fixed percentile | 0.473 | 0.226 | 0.398 | 0.231 | 0.717 | 0.078 | 0.172 | 11.6 | 39.0 | 13.1 | 48.9 | 0.397 |
| B2 Fixed mean + k·σ | 0.468 | 0.222 | 0.398 | 0.217 | 0.692 | 0.069 | 0.134 | 11.0 | 43.5 | 22.3 | 52.1 | 0.429 |
| B3 Rolling mean + 3σ (base paper) | 0.482 | 0.207 | 0.393 | 0.151 | 0.827 | 0.031 | 0.141 | 17.9 | 36.6 | -4.370 | 51.7 | 0.367 |
| B4 Global mean + 3σ (base paper, untuned) | 0.458 | 0.237 | 0.366 | 0.246 | 0.783 | 0.087 | 0.200 | 17.3 | 32.7 | -4.814 | 46.6 | 0.383 |
| **P1 Moving median + λ·MAD (proposed)** | 0.433 | 0.221 | 0.376 | 0.219 | 0.815 | 0.098 | 0.255 | 17.3 | 30.3 | -12.7 | 46.2 | 0.354 |

H1: Not supported · H2: Partially supported · H3: Not supported

Figures: [`figures/full/`](figures/full/)

**Split protocol (proposal, 60/20/20)** — 18 files, 25 anomaly windows, W* = 30

| Method | Composite F1 ↑ | Point F1 ↑ | Precision ↑ | Recall ↑ | Event recall ↑ | FPR ↓ | High-var FPR ↓ | Alerts ↓ | NAB std ↑ | NAB low-FP ↑ | NAB low-FN ↑ | Delay ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 Fixed percentile | 0.707 | 0.370 | 0.613 | 0.370 | 0.972 | 0.196 | 0.238 | 9.056 | 64.0 | 44.4 | 72.0 | 0.366 |
| B2 Fixed mean + k·σ | 0.699 | 0.393 | 0.597 | 0.397 | 0.972 | 0.207 | 0.272 | 9.907 | 60.7 | 37.3 | 69.8 | 0.328 |
| B3 Rolling mean + 3σ (base paper) | 0.741 | 0.255 | 0.714 | 0.169 | 0.910 | 0.024 | 0.090 | 4.741 | 66.6 | 55.9 | 72.0 | 0.396 |
| B4 Global mean + 3σ (base paper, untuned) | 0.743 | 0.373 | 0.657 | 0.373 | 0.972 | 0.194 | 0.245 | 5.611 | 73.3 | 64.1 | 78.2 | 0.380 |
| **P1 Moving median + λ·MAD (proposed)** | 0.547 | 0.179 | 0.489 | 0.138 | 0.793 | 0.037 | 0.090 | 5.759 | 54.7 | 39.7 | 61.8 | 0.399 |

H1: Not supported · H2: Supported descriptively (not significant) · H3: Not supported

Figures: [`figures/split/`](figures/split/)

Full tables, statistics and every figure: [docs/RESULTS.md](docs/RESULTS.md)
<!-- RESULTS_END -->

---

## Where to find everything

### Evaluation: how the methods are scored

| What | Where |
|---|---|
| Experiment design and the decision rules for H1–H3, fixed before training | [`PROTOCOL.md`](PROTOCOL.md) |
| Formula for every metric (precision, recall, F1, composite F1, FPR, NAB score, ROC/PR-AUC, detection delay) | [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) |
| Metric code (point, event and composite metrics, confusion counts) | [`src/atd/metrics.py`](src/atd/metrics.py) |
| NAB scoring code (port of Numenta's official scorer, verified against it) | [`src/atd/nab_score.py`](src/atd/nab_score.py) |
| Threshold rules B1–B4, P1 and ablations A1–A4 | [`src/atd/thresholds.py`](src/atd/thresholds.py) |
| Parameter selection without test labels (validation tuning, leave-one-file-out) | [`src/atd/pipeline.py`](src/atd/pipeline.py) |
| Statistics: Wilcoxon tests, Holm correction, effect sizes, hypothesis decisions | [`src/atd/stats.py`](src/atd/stats.py) |
| Evaluation scripts: 60/20/20 split / full corpus | [`scripts/evaluate.py`](scripts/evaluate.py) / [`scripts/evaluate_full.py`](scripts/evaluate_full.py) |
| Unit tests (metrics, NAB scorer, thresholds, leakage checks) | [`tests/test_core.py`](tests/test_core.py) |

### Metric results: the numbers

Each protocol has its own folder: [`results/full/`](results/full/) (primary, all 58 files) and [`results/split/`](results/split/) (60/20/20 proposal split). Both contain the same files:

| File | Contents |
|---|---|
| `summary_by_method.csv` | mean of every metric for each method, plus corpus NAB scores |
| `per_file_metrics.csv` | every metric per file and method (averaged over 3 seeds) |
| `test_metrics.csv` | every metric per file, method, seed and window size, including TP/FP/FN/TN counts |
| `corpus_nab.csv` | NAB scores (standard, reward-low-FP, reward-low-FN, every-step alarms) |
| `comparisons.csv` | P1 versus every other method: mean difference, 95 % CI, wins/losses, p-value, Holm p-value, effect size |
| `hypotheses.json` | H1–H3 decisions and the comparison against the base-paper rules |
| `clean_stream_per_file.csv` | false alarms on anomaly-free streams |
| `selected_params.csv`, `tuning_summary.csv`, `tuning_candidates.csv` | chosen threshold parameters and every candidate tried |
| `lofo_selected_params.csv` (full only) | the leave-one-file-out parameters used for each file |
| `window_selection.csv` | how the LSTM window size W\* was chosen |
| `scores/` | saved reconstruction errors of every trained model (reproduce results without retraining) |
| `traces/` | threshold values over time for each file |

**The readable report** is [`docs/RESULTS.md`](docs/RESULTS.md). It holds every table, every statistical test and the hypothesis decisions with their direction and p-values.

### Figures

All figures are PNG (200 dpi). Vector PDF copies for the paper are in each `pdf/` subfolder.

| Folder | Contents |
|---|---|
| [`figures/eda/`](figures/eda/) | exploratory data analysis: corpus overview, preprocessing steps, split overview, variability drift, anomaly-window statistics, autocorrelation, domains and sizes |
| [`figures/eda/series/`](figures/eda/series/) | one EDA figure per NAB file: series, rolling statistics, high-variance zones, value distribution |
| [`figures/method/`](figures/method/) | pipeline diagram and a synthetic illustration of how each threshold rule behaves |
| [`figures/full/`](figures/full/) | all result figures for the primary full-corpus protocol |
| [`figures/split/`](figures/split/) | the same result figures for the 60/20/20 split |
| `figures/*/traces/` | for every file: reconstruction error, each rule's threshold over time, and where each rule raises alarms |

Result figures in `figures/full/` and `figures/split/`:

| Topic | File |
|---|---|
| Training and validation loss | `training_curves.png`, `loss_curves_examples.png` |
| Headline metrics of all main methods | `headline_metrics.png` |
| Precision, recall and F1 (point and event level) | `precision_recall_f1.png` |
| **Confusion matrices** | `confusion_matrices.png` |
| True / false / missed alarm shares | `alarm_outcomes.png` |
| Hypothesis decisions | `hypothesis_summary.png` |
| Proposed vs baselines, per file | `paired_scatter.png`, `heatmap_composite_f1.png`, `heatmap_nab_standard.png`, `heatmap_hv_fpr.png`, `average_rank.png`, `radar_profile.png` |
| Model quality (ROC-AUC, PR-AUC) and operating points | `model_quality_auc.png`, `roc_operating_points.png` |
| Anomaly separability | `error_distributions.png` |
| MAD versus σ scale (why P1 struggles) | `scale_diagnostic.png` |
| Anomaly score and thresholds over time | `walkthrough_signal_score_threshold.png` |
| Ablation study and λ sensitivity | `ablation.png`, `sensitivity_lambda_validation.png` |
| Results by domain (RQ4) | `domain_comparison.png` |
| False alarms on anomaly-free streams | `clean_stream_false_alarms.png` |
| Detection delay and alert counts | `delay_and_alerts.png` |
| Window-size effect (split only) | `window_size_effect.png` |

### Interactive dashboards

[`dashboard/index.html`](dashboard/index.html) (full corpus) and [`dashboard/split.html`](dashboard/split.html) (60/20/20 split). Download either file and open it in a browser; internet is needed for the chart library. Tabs: overview, file explorer, comparisons, domains, ablation and sensitivity, data (EDA), method and metrics.

### Data, configuration and logs

| What | Where |
|---|---|
| NAB data (58 files), official labels and license | [`data/nab/`](data/nab/) |
| Checksums of every data file | [`data/nab/MANIFEST.csv`](data/nab/MANIFEST.csv) |
| Experiment configuration (model, threshold grids, metrics) | [`configs/experiment.json`](configs/experiment.json) |
| Which files are used in each protocol | [`configs/datasets.json`](configs/datasets.json) |
| Data summary tables | [`results/eda_summary.csv`](results/eda_summary.csv), [`results/corpus_overview.csv`](results/corpus_overview.csv) |
| Runtime estimate and the 15-hour decision | [`results/runtime_estimate.json`](results/runtime_estimate.json) |
| Log of every pipeline step | [`logs/`](logs/) |
| Changes from earlier project versions, with reasons | [`docs/CHANGES.md`](docs/CHANGES.md) |

### Code

| Path | Purpose |
|---|---|
| [`run_pipeline.py`](run_pipeline.py) | runs everything in order: download → select → EDA → runtime estimate → train → evaluate → figures → dashboards → report |
| [`src/atd/data.py`](src/atd/data.py) | loading, cleaning, chronological splits, scaling, sliding windows, normal-only training windows |
| [`src/atd/model.py`](src/atd/model.py) | LSTM Autoencoder (PyTorch, GPU if available) |
| [`scripts/`](scripts/) | one script per pipeline step |
| [`setup_wsl.sh`](setup_wsl.sh), [`requirements.txt`](requirements.txt) | environment setup |
| [`paper/`](paper/) | IEEE template for the research paper |

## Methods compared

| ID | Threshold rule | Role |
|---|---|---|
| B1 | fixed p-th percentile of training errors | fixed baseline (proposal) |
| B2 | fixed mean + k·σ of training errors | fixed baseline (proposal) |
| B3 | rolling mean + 3σ of the last w errors | base paper |
| B4 | global mean + 3σ of training errors | base paper, untuned |
| **P1** | **moving median + λ·MAD of the last w errors** | **proposed** |
| A1–A4 | {median/MAD, mean/σ} × {plain, gated memory} | ablation |

## Evaluation protocols

| | Full-corpus protocol (primary) | Split protocol (proposal) |
|---|---|---|
| Files | all 58 | 18 evaluation + 30 tuning |
| Model trained on | normal windows in the first 15 % | normal windows in the first 60 % |
| Scored | remaining 85 %: 107 anomaly windows + 6 anomaly-free files | last 20 %: 25 anomaly windows |
| Threshold parameters chosen on | the other 57 files (leave-one-file-out) | validation 20 % of 30 files |

Test labels are never used to choose a threshold.

## Reproduce

Ubuntu or WSL with an NVIDIA GPU (full guide: [`docs/RUN_ON_WSL.md`](docs/RUN_ON_WSL.md)):

```bash
bash setup_wsl.sh
source .venv/bin/activate
python run_pipeline.py                  # about 30 minutes on an RTX 4050
python run_pipeline.py --from figures_full   # redraw figures and report from saved results only
```

Data: Numenta Anomaly Benchmark v1.1 (Lavin & Ahmad, 2015).
