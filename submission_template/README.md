# Robust Adaptive Thresholding for LSTM Autoencoder-Based Anomaly Detection in Streaming Time Series

**Zain Hassan · MS Data Science · FAST-NUCES Islamabad · Reg. 26I-8015**

This repository holds the **research paper and every figure and result table produced by the experiment pipeline**. All figures were generated automatically by code from the experiment outputs, not drawn by hand. The implementation lives in a separate code repository.

- 📄 **Paper (IEEE conference format):** [`paper/`](paper/)
- 📊 **Interactive dashboard:** [`dashboard/index.html`](dashboard/index.html). Download and open it in a browser.
- 🧪 **Protocol fixed before training:** [`PROTOCOL.md`](PROTOCOL.md)
- 📁 **All numbers:** [`results/`](results/)

## Research question

Does a **moving Median + λ·MAD threshold** on LSTM Autoencoder reconstruction errors detect anomalies in NAB streaming time series better than fixed thresholds? The data, split, model and reconstruction errors are kept identical, so **only the threshold rule changes**.

![pipeline](figures/method/pipeline.png)

![threshold behaviour](figures/method/threshold_behaviour_synthetic.png)

## Results (test partitions of the 18 evaluation files)

{{RESULTS}}

![headline](figures/results/headline_metrics.png)

## Figures

| Figure | What it shows |
|---|---|
| `figures/eda/split_overview.png` | where every NAB anomaly window falls in the 60/20/20 split |
| `figures/eda/variability_drift.png` | normal variability drifts over time (why adaptive thresholds) |
| `figures/eda/domains_and_sizes.png` | evaluation files per domain and series length |
| `figures/eda/series/*.png` | EDA of every file: series, rolling stats, high-variance zones, distribution |
| `figures/method/pipeline.png` | experimental pipeline |
| `figures/method/threshold_behaviour_synthetic.png` | how each threshold rule reacts to variance shifts, spikes and long anomalies |
| `figures/results/headline_metrics.png` | main metrics, four primary methods |
| `figures/results/heatmap_*.png` | file × method results |
| `figures/results/paired_scatter.png` | proposed vs each baseline, file by file (H1–H3) |
| `figures/results/domain_comparison.png` | results by NAB domain (RQ4) |
| `figures/results/ablation.png` | robust statistic vs memory type |
| `figures/results/sensitivity_lambda_validation.png` | sensitivity to λ and window length |
| `figures/results/window_size_effect.png` | robustness to the LSTM window size |
| `figures/results/model_quality_auc.png` | LSTM-AE ROC-AUC / PR-AUC per file (RQ1) |
| `figures/results/delay_and_alerts.png` | detection delay and alert counts |
| `figures/results/traces/*.png` | per file: reconstruction error, every threshold line, and alarms |

## Data

Numenta Anomaly Benchmark v1.1 (Lavin & Ahmad, IEEE ICMLA 2015), downloaded from the pinned commit listed in `data/NAB_MANIFEST.csv` with SHA-256 checksums.
