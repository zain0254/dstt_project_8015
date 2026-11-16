"""Result figures (reads results/*.csv written by evaluate.py).

    python scripts/make_figures.py [--results results] [--scores results/scores]
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyBboxPatch
from atd import ROOT, load_config
from atd.data import load_labels, load_series, file_id
from atd.pipeline import load_errors
from atd.thresholds import rolling_plain_stats, rolling_gated
from atd import viz
from atd.viz import COLORS, LABELS, SHORT, PRIMARY, ABLATION, INK, INK2

ap = argparse.ArgumentParser()
ap.add_argument("--results", default=str(ROOT / "results"))
ap.add_argument("--scores", default=str(ROOT / "results/scores"))
ap.add_argument("--out", default=str(ROOT / "figures"))
args = ap.parse_args()
R, OUT = Path(args.results), Path(args.out)
viz.setup()
cfg = load_config()
labels = load_labels(cfg["labels_file"])
ds = json.loads((ROOT / "configs/datasets.json").read_text())
for d in ("method", "results", "results/traces"):
    (OUT / d).mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)


# ================================================================ method figures
def pipeline_diagram():
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.1); ax.axis("off")

    def box(x, y, w, h, text, fc="#ffffff", ec="#52514e", bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12", fc=fc, ec=ec, lw=1.3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, color=INK,
                fontweight="bold" if bold else "normal", wrap=True)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.3))

    box(0.1, 1.9, 1.7, 0.9, "NAB v1.1\nreal files\n(18 eval + tuning)")
    box(2.2, 1.9, 1.9, 0.9, "Sort · dedupe · ffill\n60/20/20 chronological\nmin-max fit on train")
    box(4.5, 1.9, 1.6, 0.9, "Sliding windows\nW = 30 / 50 / 100")
    box(6.5, 1.9, 1.8, 0.9, "LSTM Autoencoder\n64 → 32 → 64\n(trained without labels)", fc="#e8f0fb", bold=True)
    box(8.7, 1.9, 1.6, 0.9, "Reconstruction\nerror e(t)\n(same for all)")
    for (x0, x1) in ((1.8, 2.2), (4.1, 4.5), (6.1, 6.5), (8.3, 8.7)):
        arrow(x0, 2.35, x1, 2.35)
    meths = [("B1 fixed percentile", COLORS["B1_fixed_percentile"]),
             ("B2 fixed mean + kσ", COLORS["B2_fixed_mean_ksigma"]),
             ("B3 rolling mean + 3σ", COLORS["B3_rolling_mean_3sigma"]),
             ("P1 moving median + λ·MAD", COLORS["P1_moving_median_mad"])]
    for i, (t, c) in enumerate(meths):
        y = 3.75 - i * 1.0
        box(10.75, y, 2.1, 0.62, t, fc="#ffffff", ec=c, bold=t.startswith("P1"))
        arrow(10.3, 2.35, 10.75, y + 0.31)
    ax.text(11.8, 0.25, "tuned on validation (composite F1)\nscored on test: F1 · events · FPR · NAB",
            ha="center", fontsize=8.5, color=INK2)
    ax.text(6.5, 4.85, "Controlled design: data, split, model and errors are identical — only the threshold rule changes",
            ha="center", fontsize=10.5, color=INK, fontweight="bold")
    save(fig, "method/pipeline.png")


def masking_illustration():
    """Synthetic stream: variance shift (normal) + a long anomaly."""
    rng = np.random.default_rng(7)
    n = 900
    e = np.abs(rng.normal(0, 1, n)) * 0.2 + 0.3
    e[300:900] = np.abs(rng.normal(0, 1, 600)) * 0.45 + 0.35   # normal variability increases
    e[620:720] += np.linspace(1.6, 2.2, 100)                    # long anomaly
    e[500] += 4                                                 # isolated spike
    seed = np.abs(rng.normal(0, 1, 400)) * 0.2 + 0.3
    th_fix = np.percentile(seed, 99)
    loc, sc = rolling_plain_stats(seed, e, 100, "mean_std"); th_ms = loc + 3 * sc
    loc, sc = rolling_plain_stats(seed, e, 100, "median_mad"); th_mp = loc + 5 * sc
    _, th_mw = rolling_gated(seed, e, 100, 5, "median_mad")
    fig, ax = plt.subplots(figsize=(13, 4.3))
    ax.axvspan(620, 719, color=viz.ANOM, alpha=0.15, lw=0, label="true anomaly")
    ax.axvspan(300, 899, ymax=0.04, color="#6250d6", alpha=0.35, lw=0, label="normal variability higher")
    ax.plot(e, color="#9a9994", lw=0.7, label="reconstruction error")
    ax.plot(np.full(n, th_fix), color=COLORS["B1_fixed_percentile"], lw=2, label="fixed percentile (B1)")
    ax.plot(th_ms, color=COLORS["B3_rolling_mean_3sigma"], lw=2, label="rolling mean + 3σ (B3)")
    ax.plot(th_mp, color=COLORS["A1_median_mad_plain"], lw=2, ls="--", label="median + λ·MAD, plain memory")
    ax.plot(th_mw, color=COLORS["P1_moving_median_mad"], lw=2.2, label="median + λ·MAD, gated memory (P1 option)")
    ax.set_ylim(0, 4.6); ax.set_xlabel("time step"); ax.set_ylabel("error / threshold")
    ax.set_title("Why the rule matters: a fixed threshold floods alarms after the variance shift; mean + 3σ jumps after the "
                 "spike;\nplain rolling memory absorbs a long anomaly; gated memory keeps the anomaly visible",
                 fontsize=10.5)
    ax.legend(ncol=4, fontsize=8, loc="upper left")
    save(fig, "method/threshold_behaviour_synthetic.png")


pipeline_diagram()
masking_illustration()

# ================================================================ results
summ = pd.read_csv(R / "summary_by_method.csv").set_index("method")
pf = pd.read_csv(R / "per_file_metrics.csv")
tm = pd.read_csv(R / "test_metrics.csv")
hyp = json.loads((R / "hypotheses.json").read_text())
W_STAR = hyp["W_star"]
P = cfg["proposed_method"]
pf["name"] = pf.file.str.split("/").str[-1].str.replace(".csv", "", regex=False)

# 1. headline metrics, primary methods
mets = [("composite_f1", "Composite F1 ↑"), ("f1", "Point F1 ↑"), ("event_recall", "Event recall ↑"),
        ("precision", "Point precision ↑"), ("fpr", "False-positive rate ↓"), ("hv_fpr", "High-variance FPR ↓"),
        ("corpus_nab_standard", "NAB score (standard) ↑"), ("corpus_nab_reward_low_fp", "NAB (reward low FP) ↑")]
fig, axs = plt.subplots(2, 4, figsize=(15, 7))
for ax, (m, t) in zip(axs.flat, mets):
    vals = [summ.loc[k, m] for k in PRIMARY]
    bars = ax.bar(range(4), vals, color=[COLORS[k] for k in PRIMARY], width=0.62, edgecolor="white", lw=2)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}" if abs(v) < 2 else f"{v:.1f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8.5, color=INK)
    ax.set_xticks(range(4)); ax.set_xticklabels([SHORT[k] for k in PRIMARY])
    ax.set_title(t, fontsize=10.5)
    ax.axhline(0, color=INK2, lw=0.8)
fig.suptitle(f"Test results — mean over 18 evaluation files (W* = {W_STAR}, 3 seeds)", fontweight="bold", y=1.0)
fig.legend(handles=[Patch(color=COLORS[k], label=LABELS[k]) for k in PRIMARY], loc="lower center", ncol=4,
           bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(); save(fig, "results/headline_metrics.png")

# 2. heatmaps file x method
for m, t, cmap in (("composite_f1", "Composite F1 (higher is better)", "Blues"),
                   ("nab_standard", "NAB standard score per file (higher is better)", "Blues"),
                   ("hv_fpr", "High-variance FPR (lower is better)", "Oranges")):
    tab = pf.pivot_table(index="name", columns="method", values=m)[PRIMARY + ABLATION]
    tab = tab.reindex(pf.drop_duplicates("name").sort_values(["domain", "name"]).name)
    fig, ax = plt.subplots(figsize=(12, 7.5))
    v = tab.to_numpy(dtype=float)
    lo, hi = np.nanpercentile(v, 2), np.nanpercentile(v, 98)
    im = ax.imshow(v, aspect="auto", cmap=cmap, vmin=lo, vmax=hi)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if np.isfinite(v[i, j]):
                frac = (v[i, j] - lo) / (hi - lo + 1e-12)
                ax.text(j, i, f"{v[i, j]:.2f}" if abs(v[i, j]) < 10 else f"{v[i, j]:.0f}", ha="center", va="center",
                        fontsize=7.5, color="white" if frac > 0.6 else INK)
    ax.set_xticks(range(v.shape[1])); ax.set_xticklabels([SHORT[k] for k in tab.columns])
    ax.set_yticks(range(v.shape[0]))
    dom = pf.drop_duplicates("name").set_index("name").domain
    ax.set_yticklabels([f"{n}  [{dom[n]}]" for n in tab.index], fontsize=8)
    ax.axvline(3.5, color="white", lw=3); ax.grid(False)
    ax.set_title(f"{t} — B1–P1 = main comparison, A1–A4 = ablations")
    fig.colorbar(im, ax=ax, shrink=0.6)
    save(fig, f"results/heatmap_{m}.png")

# 3. paired file-level comparison P1 vs each baseline
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for ax, m, t in zip(axs, ("composite_f1", "hv_fpr", "nab_standard"),
                    ("Composite F1 (H1)", "High-variance FPR (H2)", "NAB standard (H3)")):
    for k in ("B1_fixed_percentile", "B2_fixed_mean_ksigma", "B3_rolling_mean_3sigma"):
        a = pf[pf.method == P].set_index("name")[m]
        b = pf[pf.method == k].set_index("name")[m]
        ax.scatter(b, a.reindex(b.index), s=46, color=COLORS[k], edgecolor="white", lw=1.2, label=f"vs {SHORT[k]}", zorder=3)
    lim = np.nanpercentile(pf[pf.method.isin(PRIMARY)][m], [0, 100])
    pad = 0.05 * (lim[1] - lim[0] + 1e-9)
    ax.plot([lim[0] - pad, lim[1] + pad], [lim[0] - pad, lim[1] + pad], color=INK2, lw=1, ls="--")
    ax.set_xlabel("baseline (per file)"); ax.set_ylabel("P1 proposed (per file)")
    better = "above" if m != "hv_fpr" else "below"
    ax.set_title(f"{t}\npoints {better} the diagonal = P1 better")
    ax.legend(fontsize=8)
fig.tight_layout(); save(fig, "results/paired_scatter.png")

# 4. domain view (RQ4)
dm = pf[pf.method.isin(PRIMARY)].groupby(["domain", "method"])[["composite_f1", "nab_standard", "hv_fpr"]].mean().reset_index()
doms = sorted(dm.domain.unique())
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, m, t in zip(axs, ("composite_f1", "nab_standard", "hv_fpr"), ("Composite F1 ↑", "NAB standard ↑", "High-variance FPR ↓")):
    w = 0.2
    for j, k in enumerate(PRIMARY):
        vals = [dm[(dm.domain == d) & (dm.method == k)][m].mean() for d in doms]
        ax.bar(np.arange(len(doms)) + (j - 1.5) * w, vals, width=w - 0.02, color=COLORS[k], label=SHORT[k])
    nf = pf[pf.method == P].groupby("domain").size()
    ax.set_xticks(range(len(doms))); ax.set_xticklabels([f"{d}\n(n={nf[d]})" for d in doms], fontsize=8)
    ax.set_title(t); ax.axhline(0, color=INK2, lw=0.8)
axs[0].legend(fontsize=8, ncol=2)
fig.suptitle("RQ4 — results by NAB domain (descriptive; n = files per domain)", fontweight="bold")
fig.tight_layout(); save(fig, "results/domain_comparison.png")

# 5. ablation: statistic x memory
ab = summ.loc[ABLATION + [P, "B3_rolling_mean_3sigma"]]
fig, axs = plt.subplots(1, 4, figsize=(15, 4))
for ax, (m, t) in zip(axs, (("composite_f1", "Composite F1 ↑"), ("event_recall", "Event recall ↑"),
                            ("hv_fpr", "High-variance FPR ↓"), ("corpus_nab_standard", "NAB standard ↑"))):
    order = ABLATION + [P]
    vals = [ab.loc[k, m] for k in order]
    ax.barh(range(len(order)), vals, color=[COLORS[k] for k in order], height=0.6)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.3f}" if abs(v) < 2 else f" {v:.1f}", va="center", fontsize=8.5)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([LABELS[k] if ax is axs[0] else "" for k in order], fontsize=8)
    ax.set_title(t); ax.invert_yaxis()
fig.suptitle("Ablation — which part matters: robust statistic (median/MAD vs mean/σ) or memory (plain vs gated)?",
             fontweight="bold")
fig.tight_layout(); save(fig, "results/ablation.png")

# 6. lambda sensitivity on validation (P1)
tsum = pd.read_csv(R / "tuning_summary.csv")
t1 = tsum[(tsum.method == P) & (tsum.W == W_STAR)].copy()
pp = t1.params.apply(json.loads)
t1["lam"] = pp.str["lam"]; t1["w"] = pp.str["w"]; t1["memory"] = pp.str["memory"]
fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
ramp = ["#a9c8f0", "#6ea2e6", "#2a78d6", "#14498c"]
for ax, mem in zip(axs, ("plain", "gated")):
    for c, w in zip(ramp, sorted(t1.w.unique())):
        d = t1[(t1.memory == mem) & (t1.w == w)].sort_values("lam")
        ax.plot(d.lam, d.val_composite_f1, marker="o", ms=4, lw=2, color=c, label=f"w = {w}")
    ax.set_xscale("log"); ax.set_xlabel("λ (MAD multiplier, log scale)")
    ax.set_title(f"{mem} memory")
axs[0].set_ylabel("mean validation composite F1"); axs[0].legend(fontsize=8)
sel = pd.read_csv(R / "selected_params.csv")
chosen = sel[(sel.method == P) & (sel.W == W_STAR)].params.iloc[0]
fig.suptitle(f"P1 sensitivity on validation data (W* = {W_STAR}); selected: {chosen}", fontweight="bold")
fig.tight_layout(); save(fig, "results/sensitivity_lambda_validation.png")

# 7. window size effect
wm = tm[tm.method.isin(PRIMARY)].groupby(["W", "method"])[["composite_f1", "nab_standard"]].mean().reset_index()
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for ax, m in zip(axs, ("composite_f1", "nab_standard")):
    for k in PRIMARY:
        d = wm[wm.method == k]
        ax.plot(d.W, d[m], marker="o", ms=8, lw=2, color=COLORS[k], label=SHORT[k])
    ax.set_xticks(cfg["window_sizes"]); ax.set_xlabel("window size W"); ax.set_title(m)
axs[0].legend(fontsize=8)
fig.suptitle("Robustness to the LSTM window size (mean over files and seeds)", fontweight="bold")
fig.tight_layout(); save(fig, "results/window_size_effect.png")

# 8. model quality (RQ1, threshold-free)
mq = pf[pf.method == P].sort_values("roc_auc")
fig, ax = plt.subplots(figsize=(11, 5.5))
y = np.arange(len(mq))
ax.barh(y - 0.2, mq.roc_auc, height=0.38, color="#2a78d6", label="ROC-AUC")
ax.barh(y + 0.2, mq.pr_auc, height=0.38, color="#1baf7a", label="PR-AUC")
ax.axvline(0.5, color=INK2, ls="--", lw=1)
ax.set_yticks(y); ax.set_yticklabels(mq.name, fontsize=8)
ax.set_title("RQ1 — LSTM-AE score quality on test (threshold-free; identical for every threshold rule)")
ax.legend(); save(fig, "results/model_quality_auc.png")

# 9. detection delay + alerts
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
data = [pf[pf.method == k].detection_delay_rel.dropna() for k in PRIMARY]
bp = axs[0].boxplot(data, patch_artist=True, widths=0.5, medianprops=dict(color=INK))
for b, k in zip(bp["boxes"], PRIMARY):
    b.set_facecolor(COLORS[k]); b.set_alpha(0.75)
axs[0].set_xticks(range(1, 5)); axs[0].set_xticklabels([SHORT[k] for k in PRIMARY])
axs[0].set_title("Detection delay (fraction of window elapsed) ↓")
vals = [summ.loc[k, "n_alerts"] for k in PRIMARY]
axs[1].bar(range(4), vals, color=[COLORS[k] for k in PRIMARY], width=0.6)
for i, v in enumerate(vals):
    axs[1].text(i, v, f"{v:.1f}", ha="center", va="bottom")
axs[1].set_xticks(range(4)); axs[1].set_xticklabels([SHORT[k] for k in PRIMARY])
axs[1].set_title("Alerts raised per test segment (mean) ↓")
fig.tight_layout(); save(fig, "results/delay_and_alerts.png")

# 10. threshold traces (the "why" figure) for every evaluation file, seed 42, W*
for f in ds["evaluation_files"]:
    s = load_series(f, cfg["nab_dir"], labels)
    trp = R / "traces" / f"{file_id(f)}__s42.npz"
    if not trp.exists():
        continue
    tr = np.load(trp)
    err, _ = load_errors(f, W_STAR, 42, args.scores)
    e = err[s.b:]
    ts = s.timestamps[s.b:]
    fig, axs = plt.subplots(2, 1, figsize=(13, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    ax = axs[0]
    for w0, w1 in s.windows_in(s.b, s.n):
        ax.axvspan(s.timestamps[w0], s.timestamps[w1], color=viz.ANOM, alpha=0.15, lw=0)
        axs[1].axvspan(s.timestamps[w0], s.timestamps[w1], color=viz.ANOM, alpha=0.15, lw=0)
    ax.plot(ts, e, color="#9a9994", lw=0.7, label="reconstruction error")
    for k in PRIMARY:
        ax.plot(ts, tr[k], color=COLORS[k], lw=1.8 if k == P else 1.3, label=LABELS[k])
    top = np.nanpercentile(e, 99.5) * 1.6
    ax.set_ylim(0, max(top, np.nanmax([np.nanmedian(tr[k]) for k in PRIMARY]) * 1.3))
    ax.set_title(f"{s.name} ({s.domain}) — test partition, W={W_STAR}, seed 42; red = NAB anomaly window")
    ax.legend(fontsize=7.5, ncol=3, loc="upper left")
    for j, k in enumerate(PRIMARY):
        al = e > tr[k]
        axs[1].scatter(ts[al], np.full(al.sum(), j), s=6, marker="|", color=COLORS[k])
    axs[1].set_yticks(range(4)); axs[1].set_yticklabels([SHORT[k] for k in PRIMARY]); axs[1].set_ylim(-0.6, 3.6)
    axs[1].set_title("alarms raised by each rule", fontsize=9)
    fig.tight_layout(); save(fig, f"results/traces/{file_id(f)}.png")
print("figures written to", OUT)
