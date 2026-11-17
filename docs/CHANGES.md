# What was wrong in the earlier versions, and what changed

## Problems found in the earlier repository (V1, V2, Final46)

| # | Problem | Effect |
|---|---|---|
| 1 | 28 of the 46 Final46 files had **no anomaly in the test part**. V1, V2 and Final46 also used 10, 23 and 46 files. | F1/NAB rested on 17–18 files; the versions can't be compared |
| 2 | λ was tuned **separately on each file**, on validation parts that often held 0–1 anomalies, with a fallback to a 5 % FPR target | Close to guessing; every method was pushed toward the same alarm rate |
| 3 | The rolling history stored **every** error, including errors already flagged as anomalous | During long NAB windows the threshold rose and P1 stopped detecting partway through |
| 4 | The rolling window length was tied to the LSTM input window (30/50/100) | The threshold followed short-term noise |
| 5 | The "base paper μ + 3σ" baseline actually **tuned k** | The base-paper comparison was not the base paper's rule |
| 6 | NAB used a **custom, unnormalised** score (values such as −417) | Not interpretable or comparable |
| 7 | Only point-level F1 | Systematically favoured rules that alarm continuously over long windows |
| 8 | Labels were used to filter training windows, while the proposal said labels would not be used | An undocumented deviation from the proposal |
| 9 | H1–H3 were worded differently in the slides, the alignment note and the proposal | Unclear what was being tested |
| 10 | No protocol written before results; methods changed between versions after results were seen | Risk of selective reporting |

## Changes in this version, with reasons

1. **Full NAB corpus as the primary protocol.** All 58 files and 107 scored anomaly windows (instead of 25), with NAB's own 15 % probationary period. The proposal's 60/20/20 split is kept as a second protocol. *Reason:* 25 events cannot support three hypotheses; the full corpus is how NAB is meant to be used.
2. **Leave-one-file-out threshold selection** (full corpus) and pooled validation tuning (split). *Reason:* one setting per method from many files instead of per-file tuning on 0–1 events; no file is scored with parameters fitted on its own labels.
3. **Normal-only training, now stated openly** (a change from the proposal). Training windows that overlap a labelled anomaly *in the training segment* are excluded, and fixed thresholds are calibrated on normal training errors. *Reason:* a reconstruction-based detector assumes it learned normal behaviour. Anomalies in the training data teach it to reconstruct anomalies and blur the score. Training-segment labels are legitimately available, and validation/test labels are never used. In the full-corpus protocol only 6 of 58 files have an anomaly in their first 15 %, so this mostly affects the split protocol.
4. **Gated memory** as a P1 option, chosen on validation (fixes problem 3). Ablations A1–A4 show whether the statistic or the memory matters.
5. **Window length w separated from W** and chosen from {50, 100, 200, 400}.
6. **Base paper implemented exactly in both readings:** rolling mean + 3σ (B3) and global mean + 3σ (B4), with k = 3 fixed. The base-paper PDF was not available to us. If its window or calibration differ, B3/B4 must be adjusted.
7. **Official NAB scoring function**, verified against Numenta's code, normalised to 0–100, with all three profiles and clearly labelled as alert-onset scoring (not a leaderboard score).
8. **Composite F1 and event metrics** added to point F1; all formulas in `docs/METHODOLOGY.md`.
9. **H2 secondary evidence:** false-alarm rate on anomaly-free streams.
10. **Statistics:** file-level paired Wilcoxon tests, Holm correction, effect sizes, bootstrap CIs, win/loss counts.
11. **Protocol fixed before training** (`PROTOCOL.md`); stand-in pipeline tests are blocked from being reported.
12. **Runtime check:** `scripts/estimate_runtime.py` times real fits on your machine and applies the 15-hour rule automatically.
13. **One-command pipeline** from download to report, with every figure generated from the results (≈ 30 figure types, PNG and vector PDF) and an interactive dashboard.

## Known limitations (to state in the paper)

- NAB labels are windows placed around anomalies by annotators. Point metrics therefore penalise any rule that does not alarm across the whole window.
- Results hold for a small univariate LSTM-AE. Another model could change the ranking of threshold rules.
- The base paper's exact rolling-window length is unknown (see change 6).
- Per-domain results (RQ4) rest on 6–16 files per domain and are descriptive.
