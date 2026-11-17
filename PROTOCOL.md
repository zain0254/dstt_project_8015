# Experiment Protocol (fixed before the LSTM training run)

This file fixes every design decision **before** any LSTM-AE result exists. Pipeline tests used a stand-in model (PCA). Its outputs were deleted and never reported, and the evaluation scripts refuse to report non-LSTM scores. Anything not listed here is exploratory. Changes from the October 2026 proposal, and the reasons for them, are in [docs/CHANGES.md](docs/CHANGES.md).

## 1. Research questions and hypotheses (proposal wording)

- **RQ1** How does an LSTM Autoencoder with fixed percentile and fixed mean-plus-standard-deviation thresholds perform on NAB streaming time series?
- **RQ2** Does the proposed moving Median + MAD threshold improve precision, recall, F1-score, PR-AUC or NAB score compared with fixed threshold baselines?
- **RQ3** Does the proposed threshold reduce false-positive alerts during periods of changing normal data variability?
- **RQ4** Which NAB categories benefit most from robust adaptive thresholding?
- **H1** P1 achieves a higher average F1-score than at least one fixed-threshold baseline.
- **H2** P1 reduces false-positive rates on some time series with changing normal variability.
- **H3** P1 maintains or improves the NAB score.

In addition, P1 is compared with the base paper's rule (mean + 3σ). This comparison is descriptive and is not a proposal hypothesis.

## 2. Data

NAB v1.1, commit `ea702d7`. The full corpus has **58 files and 116 labelled anomaly windows**. Every file is sorted by timestamp, duplicate timestamps are removed (first kept), and missing values are forward-filled. Min–max scaling is fitted on the training segment of each file only. `data/nab/MANIFEST.csv` stores a sha256 checksum per file.

Two protocols share the model, the methods and the metrics. The selection rules use label positions only, never model output.

| | **Full-corpus protocol (primary)** | **Split protocol (proposal)** |
|---|---|---|
| Files | all 58 | 18 evaluation + 30 tuning (41 distinct) |
| Training segment | first 15 % of each file (NAB probationary period) | first 60 % |
| Scored segment | remaining 85 %: 52 files with **107 anomaly windows** + 6 anomaly-free files | last 20 % of files with a test anomaly: **25 windows**; anomaly-free test parts of 23 tuning-only files |
| Threshold selection | **leave-one-file-out** over the scored segments of the other 57 files | validation 20 % of the 30 tuning files, pooled |
| Window size | W\* from the split protocol | W ∈ {30, 50, 100}; W\* = best mean validation ROC-AUC |

The full-corpus protocol is primary because it uses 4× more anomaly events. That is how NAB itself is scored, with the first 15 % as a probationary period. The split protocol follows the proposal and is reported alongside. If the two disagree, both results are reported.

**Runtime rule (decided before running):** `scripts/estimate_runtime.py` times real fits on the machine that runs the experiment and projects the total. If the projection exceeds 15 h, the full-corpus protocol is restricted to the five real-data categories. Files are then kept in decreasing order of scored anomaly windows (then normal points) until the budget is met. The decision is written to `results/runtime_estimate.json` and reported.

## 3. Model (identical for every threshold method)

- Univariate LSTM Autoencoder: encoder LSTM(64) → linear latent (32, tanh) → repeated → decoder LSTM(64) → linear output.
- **Normal-only training.** Training uses windows that end in the training segment and contain no point of a labelled anomaly window. Only labels inside the training segment are used. If fewer than 32 clean windows remain, all training windows are used, but no file reaches that limit. At most 2,500 evenly spaced windows are used. The chronologically last 10 % are the early-stopping hold-out.
- Adam, learning rate 1e-3. Split protocol: batch 64, ≤ 40 epochs, patience 5. Full-corpus protocol (short training segments): batch 32, ≤ 60 epochs, patience 6. The best hold-out epoch is restored.
- Seeds {15, 42, 123}. Fits: split 41 files × 3 W × 3 seeds = 369; full 58 × 1 × 3 = 174.
- Anomaly score at time t = mean squared reconstruction error of the window ending at t (causal).

## 4. Threshold methods (the only experimental variable)

"Training errors" means the reconstruction errors of normal training points (same rule as §3). Rolling rules are causal: the threshold at time t uses errors before t only. They run continuously over the scored stream and are seeded with the last w training errors.

| ID | Rule | Parameters chosen without test labels |
|---|---|---|
| B1 | fixed p-th percentile of training errors | p ∈ {90, 95, 97, 98, 99, 99.5, 99.9} |
| B2 | fixed mean + k·σ of training errors | k ∈ {1, …, 8} |
| B3 | **rolling mean + 3σ** of the previous w errors (base paper, k = 3 fixed) | w ∈ {50, 100, 200, 400} |
| B4 | **global mean + 3σ** of training errors (base paper, untuned) | none |
| **P1** | **moving median + λ·MAD** of the previous w errors (proposal) | λ ∈ {1.5, …, 50}, w, memory ∈ {plain, gated} |
| A1 / A2 | median + λ·MAD, plain / gated memory only | λ, w |
| A3 / A4 | rolling mean + k·σ, plain / gated memory only | k, w |

*Plain memory*: every error enters the moving window. *Gated memory*: an error that triggered an alarm does not enter the window, so a long anomaly cannot raise the threshold it is judged by. After more than w consecutive alarms, the change is accepted as the new normal and raw values enter again. The ablations A1–A4 separate the effect of the statistic (median/MAD vs mean/σ) from the effect of the memory.

## 5. Parameter selection ("threshold rewarding")

The selection objective is the **composite F1**, the harmonic mean of point precision and event recall (§6). Ties are broken by the higher NAB score, then the lower alarm rate. The same objective, procedure and data apply to every method.

- Split protocol: maximise the mean over tuning files and seeds of the validation-segment objective.
- Full-corpus protocol: for each file f, maximise the mean objective over the other 57 files. File f is then scored with those parameters, so **no file is ever scored with parameters fitted on its own labels**.

## 6. Metrics (scored segment)

Exact definitions are in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

- Point level: precision, recall, F1, FPR and accuracy, where every point inside a NAB window is positive. Confusion matrices are pooled over files.
- Event level: event recall, event precision, event F1, and detection delay (fraction of the window elapsed before the first alarm).
- **Composite F1** (Garg et al., 2021).
- **High-variance FPR**: FPR on normal points whose causal 50-step rolling std exceeds the 75th percentile of the training segment. **Clean-stream FPR**: FPR on anomaly-free scored segments, where every alarm is false.
- **NAB score**: Numenta's scoring function (port of `nab/sweeper.py`, verified to 1e-14 against the official code), standard / reward-low-FP / reward-low-FN profiles, normalised so 0 = never alarming and 100 = perfect, summed over the corpus. Alarms are reported to the scorer as **alert onsets** (first step of each run of alarms), identically for all methods. The every-step version is reported as `*_pointwise`. This is the NAB scoring function applied to our thresholded detectors. It is **not a NAB leaderboard score**, because NAB's leaderboard optimises one threshold per detector on all labels.
- ROC-AUC and PR-AUC of the raw score. These measure model quality and are identical for all threshold methods (RQ1).

## 7. Statistics and decision rules

The unit of analysis is the file (metrics averaged over the 3 seeds). Tests are paired two-sided Wilcoxon signed-rank tests, Holm-corrected within each family, with rank-biserial effect size, bootstrap 95 % CI of the mean difference, and win/tie/loss counts. α = 0.05.

- **H1** (primary: composite F1; secondary: point F1): *supported* if mean P1 > mean of B1 or B2; *statistically supported* if that difference is Holm-significant.
- **H2** (high-variance FPR; secondary: clean-stream FPR): *supported* if P1 is lower than both B1 and B2 on average and on at least half of the files; *statistically supported* if both differences are Holm-significant; *partially supported* if P1 is lower on some files only.
- **H3** (corpus NAB, standard profile): *improves* if P1 ≥ both B1 and B2; *maintains* if within 5 points of both; otherwise not supported.
- **Base paper** (B3, B4): the same tests, reported descriptively.
- **RQ4**: per-domain means (descriptive; per-domain tests would be underpowered).
