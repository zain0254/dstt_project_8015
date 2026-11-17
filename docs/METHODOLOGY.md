# Methodology: definitions of every score

Notation: the scored segment has points t = 1…T. y_t = 1 if t lies inside a labelled NAB anomaly window, else 0. The detector raises alarm a_t ∈ {0, 1}. The labelled windows in the segment are E_1…E_K, each an index interval.

## Anomaly score

e_t = (1/W) Σ_{i=t−W+1..t} (x_i − x̂_i)², the mean squared reconstruction error of the window ending at t. Only past and current values are used, so the score is causal.

## Threshold rules

| Rule | Threshold τ_t | Alarm |
|---|---|---|
| B1 | percentile_p(training errors) | a_t = [e_t > τ] |
| B2 | mean(training errors) + k·std(training errors) | same |
| B3 | mean(e_{t−w..t−1}) + 3·std(e_{t−w..t−1}) | same |
| B4 | mean(training errors) + 3·std(training errors) | same |
| P1 | median(M_t) + λ·MAD(M_t), where M_t = the previous w values in memory | same |

MAD(M) = median(|m − median(M)|). To avoid a zero threshold on flat streams, the scale is floored at 1e-12 + 1e-3·|location|. With *gated* memory, e_t enters M only if a_t = 0, unless more than w consecutive alarms have occurred (the change is then accepted as the new normal).

## Point metrics

TP = Σ a_t y_t, FP = Σ a_t (1−y_t), FN = Σ (1−a_t) y_t, TN = Σ (1−a_t)(1−y_t).
Precision = TP/(TP+FP) (0 if no alarm). Recall = TP/(TP+FN). F1 = 2PR/(P+R). FPR = FP/(FP+TN).

## Event metrics

- Event recall = |{k : ∃ t ∈ E_k with a_t = 1}| / K.
- Event precision = (alarm segments that overlap some E_k) / (all alarm segments). An alarm segment is a maximal run of consecutive alarms.
- Event F1 = harmonic mean of event precision and event recall.
- **Composite F1** = 2·P_point·R_event / (P_point + R_event) (Garg et al., *IEEE TNNLS*, 2021). NAB labels are long windows, so point recall punishes a detector that fires once at the start of an anomaly. Event recall counts that anomaly as found, while point precision still charges every false alarm point. Unlike "point-adjust" F1, it does not inflate scores.
- Detection delay = (index of the first alarm inside E_k − start of E_k) / |E_k|, averaged over detected windows.

## NAB score

This is a port of `numenta/NAB/nab/sweeper.py`, tested against the official code on 900 random cases (largest difference 4·10⁻¹⁵).

- σ(y) = 2/(1+e^{5y}) − 1, the scaled sigmoid. σ(y) = −1 for y > 3.
- Each window E_k contributes once. If it contains alarms, the earliest one at relative position y = −(end − t + 1)/|E_k| earns A_TP·σ(y)/σ(−1): +A_TP at the left edge, about 0 at the right edge. If it contains none, it costs −A_FN.
- Each alarm outside every window costs A_FP·σ(d), where d is its distance past the previous window's right edge divided by (that window's width − 1). An alarm before any window, or far away, costs −A_FP.
- Profiles (A_TP, A_FP, A_FN): standard (1, 0.11, 1), reward-low-FP (1, 0.22, 1), reward-low-FN (1, 0.11, 2).
- Corpus score = 100·(Σraw − Σnull)/(Σperfect − Σnull), where null = −A_FN·K (a detector that never alarms) and perfect = +A_TP·K. Sums run over files. This is computed per seed and averaged over seeds.
- **Alert onsets.** The scorer receives one detection at the first step of each alarm run. Without this, a 200-step alarm run would count as 200 false positives. The every-step version is reported as `nab_*_pointwise` for transparency.
- **How this differs from the NAB leaderboard.** NAB's leaderboard picks one threshold per detector using all labels of the corpus. Here, thresholds are picked without the scored file's labels (validation data or leave-one-file-out). Our NAB numbers are therefore lower than, and not comparable with, leaderboard numbers.

## High-variance and clean-stream FPR

- High-variance points are normal points whose causal 50-step rolling std of the scaled value exceeds the 75th percentile of that rolling std inside the training segment. High-variance FPR is the FPR restricted to these points (RQ3/H2).
- Clean-stream FPR is the FPR on scored segments that contain no labelled window, where every alarm is false.

## Model quality (threshold-free)

ROC-AUC and PR-AUC (average precision) of e_t against y_t on the scored segment. They are the same for every threshold rule and describe how separable the anomalies are for the model (RQ1).

## Statistics

The unit is the file (seed-averaged). The paired Wilcoxon signed-rank test (two-sided, zero differences dropped) is Holm-corrected within each family. Effect size is the matched-pairs rank-biserial r. The 95 % CI of the mean difference comes from 5,000 bootstrap resamples of files.
