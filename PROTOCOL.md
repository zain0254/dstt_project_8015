# Experiment Protocol (fixed before the training run)

This file records every design decision **before** the LSTM Autoencoder was trained and before any test result existed. The git history shows it was committed before the first score file. Anything not listed here is reported as exploratory.

## 1. Research questions and hypotheses (verbatim from the proposal)

- **RQ1** How does an LSTM Autoencoder with fixed percentile and fixed mean-plus-standard-deviation thresholds perform on selected NAB streaming time-series files?
- **RQ2** Does the proposed moving Median + MAD threshold improve precision, recall, F1-score, PR-AUC, or NAB score compared with fixed threshold baselines?
- **RQ3** Does the proposed threshold reduce false-positive anomaly alerts during periods of changing normal data variability?
- **RQ4** Which NAB categories or anomaly types benefit most from robust adaptive thresholding?
- **H1** The proposed moving Median + MAD threshold will achieve a higher average F1-score than at least one fixed-threshold baseline.
- **H2** The proposed method will reduce false-positive rates on some time series with changing normal variability.
- **H3** The proposed method will maintain or improve NAB score by detecting anomalies with fewer false alerts or earlier detection.

## 2. Data

- NAB v1.1 (numenta/NAB, `data/` and `labels/combined_windows.json`), real-world files only.
- Each file is sorted by timestamp, duplicate timestamps removed, missing values forward-filled.
- Chronological split per file: train = first 60 %, validation = next 20 %, test = last 20 %.
- **Selection rule (uses label positions only, never model output):**
  - *Evaluation set*: every real file with ≥ 1 labelled anomaly window overlapping the test partition → **18 files, 5 domains**.
  - *Tuning set*: every real file with ≥ 1 window overlapping the validation partition → **25 files**.
  - Files in neither set have no labelled anomaly in validation or test and cannot measure detection; they are excluded. The resulting list is in `configs/datasets.json`.
- Min–max scaling fitted on the training partition only.

## 3. Model (identical for every threshold method)

- Univariate LSTM Autoencoder: encoder LSTM(64) → linear latent(32, tanh) → repeated → decoder LSTM(64) → linear output.
- Trained **without labels** on windows that end inside the training partition (proposal §5: labels are not used to train). At most 2,500 evenly-spaced training windows per fit; the chronologically last 10 % of them is the early-stopping hold-out (Adam, lr 1e-3, batch 256, ≤ 25 epochs, patience 3).
- Window sizes W ∈ {30, 50, 100}; seeds {15, 42, 123} → 36 files × 3 × 3 = **324 model fits**.
- Anomaly score at time t = mean squared reconstruction error of the window ending at t (causal).

## 4. Threshold methods (the only experimental variable)

| ID | Rule | Tuned on validation |
|---|---|---|
| B1 | Fixed p-th percentile of training errors | p |
| B2 | Fixed mean + k·std of training errors | k |
| B3 | Rolling mean + 3·std of previous w errors (2026 base paper, k fixed at 3) | w |
| **P1** | **Moving median + λ·raw MAD of previous w errors (proposal Step 6)** | λ, w, memory ∈ {plain, gated} |
| A1 | P1 with plain memory only (ablation) | λ, w |
| A2 | P1 with gated memory only (ablation) | λ, w |
| A3 | Rolling mean + k·std, plain memory (ablation) | k, w |
| A4 | Rolling mean + k·std, gated memory (ablation) | k, w |

*Plain memory*: every error enters the moving window. *Gated (anomaly-aware) memory*: an error that exceeded its threshold is not added to the moving window, so an anomaly cannot inflate the statistics it is judged by and the window always holds real normal errors. If alarms persist for more than w consecutive steps the change is accepted as the new normal and raw values enter again. Rolling rules are causal and run continuously from the start of validation to the end of test, seeded with the last w training errors. Grids are in `configs/experiment.json`.

## 5. Tuning and window selection

- One parameter setting per (method, W), chosen to maximise **mean validation composite F1** over all tuning files and seeds; ties broken by higher validation NAB score, then lower alarm rate. Test labels are never used.
- W\* (primary window size, same for all methods) = W with the highest mean validation ROC-AUC of the raw score over tuning files — threshold-free, so it cannot favour any method. Results for all W are also reported.

## 6. Metrics (test partition of the 18 evaluation files)

- Point: precision, recall, F1, FPR, accuracy (points inside NAB windows are positives).
- Event: event recall (windows with ≥ 1 alarm), event precision (alarm segments touching a window), event F1, detection delay.
- **Composite F1** = harmonic mean of point precision and event recall (Garg et al., 2021).
- **High-variance FPR**: FPR on normal test points whose causal rolling std (50 steps) exceeds the 75th percentile of the training rolling std.
- **NAB score**: official NAB scoring (verified to 1e-14 against numenta `sweeper.py`), standard / reward-low-FP / reward-low-FN profiles, normalised 0 = null detector, 100 = perfect, test windows clipped to the test partition, no probation period. **Alarms are converted to alerts at their onset** (one alert when the stream enters an alarm state), identically for every method; the every-time-step variant is reported as `nab_*_pointwise`.
- ROC-AUC and PR-AUC of the raw score (model quality; identical for all threshold methods — they answer RQ1, not RQ2).

## 7. Statistics and decision rules

Unit of analysis = file (metrics averaged over the 3 seeds at W\*). Paired two-sided Wilcoxon signed-rank tests, Holm correction within each hypothesis family, rank-biserial effect size, bootstrap 95 % CI of the mean difference, win/tie/loss counts. α = 0.05.

- **H1** (primary metric composite F1; point F1 reported as secondary): *supported* if mean P1 > mean of at least one of B1/B2; *statistically supported* if that difference is also Holm-significant.
- **H2** (high-variance FPR): *supported* if P1's mean is lower than both B1 and B2 and P1 is lower on at least half of the evaluable files; *statistically supported* if both differences are Holm-significant; *partially supported* if P1 is lower on some files only.
- **H3** (corpus NAB, standard profile): *improves* if P1 ≥ both B1 and B2; *maintains* if within 5 points of both; otherwise not supported.
- **RQ4**: domain-level means and P1-minus-baseline differences (descriptive; 1–5 files per domain is too few for per-domain tests).
