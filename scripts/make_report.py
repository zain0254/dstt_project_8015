"""Step 12 - write docs/RESULTS.md from the result tables (both protocols).

Every number in the report is read from results/*.csv and results/*.json;
nothing is typed by hand. Also fills the results block of README.md.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from atd import ROOT, load_config, protocol, load_datasets
from atd.viz import LABELS, SHORT, PRIMARY, ABLATION

cfg = load_config()
ds = load_datasets()
P = cfg["proposed_method"]
MET = [("composite_f1", "Composite F1 ↑"), ("f1", "Point F1 ↑"), ("precision", "Precision ↑"), ("recall", "Recall ↑"),
       ("event_recall", "Event recall ↑"), ("fpr", "FPR ↓"), ("hv_fpr", "High-var FPR ↓"), ("n_alerts", "Alerts ↓"),
       ("corpus_nab_standard", "NAB std ↑"), ("corpus_nab_reward_low_fp", "NAB low-FP ↑"),
       ("corpus_nab_reward_low_fn", "NAB low-FN ↑"), ("detection_delay_rel", "Delay ↓")]


def f(x, k=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.{k}f}" if abs(x) < 10 else f"{x:.1f}"


def table(summ, methods):
    cols = [c for c, _ in MET if c in summ]
    out = ["| Method | " + " | ".join(l for c, l in MET if c in summ) + " |", "|---|" + "---:|" * len(cols)]
    for m in methods:
        r = summ[summ.method == m]
        if len(r):
            r = r.iloc[0]
            name = f"**{LABELS[m]}**" if m == P else LABELS[m]
            out.append(f"| {name} | " + " | ".join(f(r[c]) for c in cols) + " |")
    return "\n".join(out)


def stat_table(comp, metrics):
    out = ["| Metric | vs | n | P1 mean | other mean | Δ (95 % CI) | P1 better on | p (Holm) | effect r |", "|---|---|---:|---:|---:|---|---:|---:|---:|"]
    for m in metrics:
        d = comp[comp.metric == m]
        lower = m in ("fpr", "hv_fpr", "calm_fpr", "n_alerts", "alerts_per_1000", "detection_delay_rel", "alarm_rate", "clean_fpr", "clean_alerts_per_1000")
        for r in d.itertuples():
            better = r.losses if lower else r.wins
            out.append(f"| {m} | {SHORT[r.b]} | {r.n_files} | {f(r.mean_a)} | {f(r.mean_b)} | {r.mean_diff:+.3f} ({f(r.ci95_low)}, {f(r.ci95_high)}) | "
                       f"{better}/{r.n_files} | {r.p_holm:.3f}{'*' if r.p_holm < 0.05 else ''} | {f(r.rank_biserial, 2)} |")
    return "\n".join(out)


def section(name):
    PR = protocol(cfg, name)
    R = PR["results"]
    if not (R / "summary_by_method.csv").exists():
        return f"## {PR['label']}\n\n*Not run yet.*\n", None
    summ = pd.read_csv(R / "summary_by_method.csv")
    comp = pd.read_csv(R / "comparisons.csv")
    hyp = json.loads((R / "hypotheses.json").read_text())
    pf = pd.read_csv(R / "per_file_metrics.csv")
    sel = pd.read_csv(R / "selected_params.csv")
    W = hyp["W_star"]
    nfiles = pf.file.nunique()
    nev = int(pf[pf.method == P].n_events.sum())
    figdir = "../" + str(Path(PR["figures"]).relative_to(ROOT))   # links are relative to docs/
    s = [f"## {PR['label']}\n"]
    if name == "split":
        s.append(f"Chronological 60/20/20 split per file. {nfiles} files have a labelled anomaly in their test partition "
                 f"({nev} anomaly windows); thresholds were tuned on the validation partitions of {len(ds['tuning_files'])} files. "
                 f"Window size W* = {W} (highest validation ROC-AUC). Metrics are seed-averaged per file, then averaged over files; "
                 f"NAB scores are corpus-level.\n")
    else:
        s.append(f"{len(ds['full_corpus_files'])} NAB files{' (five-category fallback plan)' if ds.get('plan') == 'five_categories' else ''}. "
                 f"The LSTM-AE learned from normal windows in the first 15 % of each file (NAB probationary period); the remaining 85 % is scored: {nfiles} files with {nev} anomaly windows, plus "
                 f"{len(ds['full_clean_files'])} anomaly-free files where every alarm is false. Threshold parameters were chosen "
                 f"leave-one-file-out, so no file is scored with parameters fitted on its own labels. W* = {W} (selected in the split protocol).\n")
    s.append("### Hypothesis decisions\n")
    s.append("| Hypothesis | Metric | Verdict (rule in PROTOCOL.md) | P1 − baseline: mean Δ, p (Holm), files where P1 is better |\n|---|---|---|---|")
    for h in hyp["hypotheses"]:
        lower = h["metric"] in ("hv_fpr", "fpr")
        det = "; ".join(f"{SHORT[d['b']]}: {d['mean_diff']:+.3f}, p={d['p_holm']:.3f}, "
                        f"{d['losses'] if lower else d['wins']}/{d['n_files']}" for d in h.get("detail", []))
        s.append(f"| {h['hypothesis']} ({h['role']}) | {h['metric']} | **{h['verdict']}** | {det} |")
    s.append("\nFor FPR metrics a negative Δ favours P1; for all others a positive Δ does. "
             "The verdict applies the pre-registered rule; the last column shows direction and size.")
    s.append(f"\n![hypotheses]({figdir}/hypothesis_summary.png)\n")
    s.append("### Main comparison\n")
    s.append(table(summ, PRIMARY))
    s.append(f"\n![headline]({figdir}/headline_metrics.png)\n")
    s.append("### Selected parameters (chosen without test labels)\n")
    s.append("| Method | Parameters |\n|---|---|")
    for r in sel[sel.W == W].itertuples():
        extra = f" (most frequent leave-one-out choice, {r.n_files} files)" if "n_files" in sel else ""
        s.append(f"| {LABELS[r.method]} | `{r.params}`{extra} |")
    s.append("\n### Statistical tests: P1 versus the baselines\n")
    s.append("Paired Wilcoxon signed-rank tests across files, Holm-adjusted within each metric; * = p < 0.05. "
             "Δ = P1 − other; for ↓ metrics a negative Δ favours P1.\n")
    s.append(stat_table(comp[comp.b.isin([m for m in PRIMARY if m != P])],
                        ["composite_f1", "f1", "event_recall", "fpr", "hv_fpr", "nab_standard"]))
    s.append(f"\n![paired]({figdir}/paired_scatter.png)\n")
    bp = pd.DataFrame(hyp.get("base_paper", []))
    if len(bp):
        s.append("### P1 versus the base-paper rules (rolling and global mean + 3σ)\n")
        s.append(stat_table(bp.assign(b=bp.b), list(dict.fromkeys(bp.metric))))
        s.append("")
    s.append("### Ablation: statistic × memory\n")
    s.append(table(summ, [P] + ABLATION))
    s.append(f"\n![ablation]({figdir}/ablation.png)\n")
    s.append("### By domain (RQ4)\n")
    dom = pf[pf.method.isin(PRIMARY)].groupby(["domain", "method"])[["composite_f1", "nab_standard", "hv_fpr"]].mean().reset_index()
    s.append("| Domain | n | " + " | ".join(SHORT[k] + " comp.F1" for k in PRIMARY) + " | " + " | ".join(SHORT[k] + " NAB" for k in PRIMARY) + " |")
    s.append("|---|---:|" + "---:|" * (2 * len(PRIMARY)))
    for d, g in dom.groupby("domain"):
        n = pf[(pf.domain == d) & (pf.method == P)].file.nunique()
        s.append(f"| {d} | {n} | " + " | ".join(f(g[g.method == k].composite_f1.mean()) for k in PRIMARY) + " | " +
                 " | ".join(f(g[g.method == k].nab_standard.mean()) for k in PRIMARY) + " |")
    s.append(f"\n![domains]({figdir}/domain_comparison.png)\n")
    s.append("### Other figures\n")
    for fig, cap in [("precision_recall_f1", "Precision, recall and F1"), ("confusion_matrices", "Confusion matrices"),
                     ("walkthrough_signal_score_threshold", "Signal, reconstruction, score and thresholds on one file"),
                     ("loss_curves_examples", "Training vs hold-out loss"), ("heatmap_composite_f1", "Composite F1 per file and method"), ("heatmap_nab_standard", "NAB score per file"),
                     ("heatmap_hv_fpr", "High-variance FPR per file"), ("roc_operating_points", "Each rule's operating point on the model's ROC curve"),
                     ("error_distributions", "Reconstruction-error distributions, normal vs anomalous"),
                     ("scale_diagnostic", "MAD vs std scale of training errors"), ("alarm_outcomes", "TP / FP / FN shares"),
                     ("radar_profile", "Overall profile"), ("average_rank", "Average rank"), ("delay_and_alerts", "Detection delay and alert counts"),
                     ("clean_stream_false_alarms", "False alarms on anomaly-free streams"), ("sensitivity_lambda_validation", "λ sensitivity"),
                     ("window_size_effect", "Window-size effect"), ("training_curves", "LSTM-AE training curves"), ("model_quality_auc", "Model ROC/PR-AUC per file")]:
        if (ROOT / figdir[3:] / f"{fig}.png").exists():
            s.append(f"- [{cap}]({figdir}/{fig}.png)")
    s.append(f"- Per-file threshold traces: `{figdir}/traces/` · vector PDFs for the paper: `{figdir}/pdf/`\n")
    return "\n".join(s), (summ, hyp, nfiles, nev)


parts, heads = [], {}
for name in ("full", "split"):
    txt, info = section(name)
    parts.append(txt); heads[name] = info

doc = ["# Results\n", "All numbers below are read from `results/` by `scripts/make_report.py`; all figures are generated by the pipeline. "
       "Decision rules were fixed in [PROTOCOL.md](../PROTOCOL.md) before training.\n"]
doc += parts
doc.append("## Data and model\n")
doc.append("- [Corpus overview](../figures/eda/corpus_overview.png) · [split overview](../figures/eda/split_overview.png) · "
           "[preprocessing example](../figures/eda/preprocessing_example.png) · [variability drift](../figures/eda/variability_drift.png) · "
           "[anomaly window statistics](../figures/eda/anomaly_windows_stats.png) · [autocorrelation](../figures/eda/autocorrelation_by_domain.png) · "
           "per-file EDA in `figures/eda/series/`\n")
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs/RESULTS.md").write_text("\n".join(doc))

# README block
blk = []
for name in ("full", "split"):
    info = heads[name]
    if info is None:
        continue
    summ, hyp, nfiles, nev = info
    PR = protocol(cfg, name)
    blk.append(f"**{PR['label']}** — {nfiles} files, {nev} anomaly windows, W* = {hyp['W_star']}\n")
    blk.append(table(summ, PRIMARY))
    blk.append("\n" + " · ".join(f"{h['hypothesis']}: {h['verdict']}" for h in hyp["hypotheses"] if h["role"] == "primary") + "\n")
    blk.append(f"Figures: [`{Path(PR['figures']).relative_to(ROOT)}/`]({Path(PR['figures']).relative_to(ROOT)}/)\n")
if blk:
    rd = (ROOT / "README.md").read_text()
    a, b = rd.index("<!-- RESULTS_START -->"), rd.index("<!-- RESULTS_END -->")
    rd = rd[:a] + "<!-- RESULTS_START -->\n" + "\n".join(blk) + "\nFull tables, statistics and every figure: [docs/RESULTS.md](docs/RESULTS.md)\n" + rd[b:]
    (ROOT / "README.md").write_text(rd)
print("docs/RESULTS.md written; README results block updated")
