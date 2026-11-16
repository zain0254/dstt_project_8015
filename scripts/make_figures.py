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
for d in ("method", "eda", "results", "results/traces"):
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


# ================================================================ extra figures
from sklearn.metrics import roc_curve

# 11. training curves / early stopping (from the saved training history of every fit)
metas = []
for f in Path(args.scores).glob("*.npz"):
    m = json.loads(str(np.load(f)["meta"]))
    metas.append(m)
mdf = pd.DataFrame(metas)
if "loss_history" in mdf:
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    ramp = {30: "#a9c8f0", 50: "#2a78d6", 100: "#14498c"}
    for Wv, d in mdf.groupby("W"):
        H = [np.array(h)[:, 1] / np.array(h)[0, 1] for h in d.loss_history if h]
        L = max(len(h) for h in H)
        M = np.full((len(H), L), np.nan)
        for i, h in enumerate(H):
            M[i, :len(h)] = h
        med = np.nanmedian(M, axis=0)
        q1, q3 = np.nanpercentile(M, 25, axis=0), np.nanpercentile(M, 75, axis=0)
        x = np.arange(1, L + 1)
        axs[0].plot(x, med, color=ramp.get(Wv, "#2a78d6"), lw=2, label=f"W = {Wv} (n={len(H)} fits)")
        axs[0].fill_between(x, q1, q3, color=ramp.get(Wv, "#2a78d6"), alpha=0.15, lw=0)
    axs[0].set_yscale("log"); axs[0].set_xlabel("epoch"); axs[0].set_ylabel("hold-out MSE ÷ epoch-1 MSE")
    axs[0].set_title("LSTM-AE training: hold-out loss (median, IQR)"); axs[0].legend(fontsize=8)
    axs[1].hist(mdf.epochs_run, bins=np.arange(0.5, mdf.epochs_run.max() + 1.5), color="#2a78d6", edgecolor="white")
    axs[1].set_xlabel("epochs until early stopping"); axs[1].set_ylabel("fits")
    axs[1].set_title(f"Early stopping (patience {cfg['model']['early_stopping_patience']}, max {cfg['model']['max_epochs']})")
    if "train_seconds" in mdf:
        for Wv, d in mdf.groupby("W"):
            axs[2].scatter(d.n_train_windows, d.train_seconds, s=14, color=ramp.get(Wv), label=f"W = {Wv}", alpha=0.8)
        dev = mdf["gpu"].dropna().iloc[0] if "gpu" in mdf and mdf["gpu"].notna().any() else "CPU"
        axs[2].set_xlabel("training windows"); axs[2].set_ylabel("seconds")
        axs[2].set_title(f"Training time per fit ({dev})"); axs[2].legend(fontsize=8)
    fig.tight_layout(); save(fig, "results/training_curves.png")

# 12. reconstruction-error distributions: normal vs anomaly-window points (test, W*, seed 42)
rows = []
for f in ds["evaluation_files"]:
    s = load_series(f, cfg["nab_dir"], labels)
    err, _ = load_errors(f, W_STAR, 42, args.scores)
    tr_err = err[W_STAR - 1: s.a]
    z = (err[s.b:] - np.median(tr_err)) / (np.median(np.abs(tr_err - np.median(tr_err))) + 1e-12)
    rows.append((s.name, z[s.labels[s.b:] == 0], z[s.labels[s.b:] == 1]))
fig, ax = plt.subplots(figsize=(13, 5.5))
for i, (nm, zn, za) in enumerate(rows):
    for zz, off, c in ((zn, -0.18, "#2a78d6"), (za, 0.18, viz.ANOM)):
        if len(zz):
            v = np.clip(zz, -5, np.percentile(zz, 99.5))
            bp = ax.boxplot(v, positions=[i + off], widths=0.32, patch_artist=True, showfliers=False,
                            medianprops=dict(color=INK))
            bp["boxes"][0].set_facecolor(c); bp["boxes"][0].set_alpha(0.6)
ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[0] for r in rows], rotation=60, ha="right", fontsize=8)
ax.set_yscale("symlog", linthresh=5); ax.set_ylabel("error, robust z (train median / MAD)")
ax.legend(handles=[Patch(color="#2a78d6", alpha=0.6, label="normal test points"),
                   Patch(color=viz.ANOM, alpha=0.6, label="points inside NAB anomaly windows")], loc="upper left")
ax.set_title(f"How separable are anomalies? Test reconstruction errors (W*={W_STAR}, seed 42)")
save(fig, "results/error_distributions.png")

# 13. ROC curve of the model + each threshold rule's operating point (small multiples)
ev = ds["evaluation_files"]
nc = 6; nr = int(np.ceil(len(ev) / nc))
fig, axs = plt.subplots(nr, nc, figsize=(16, 2.9 * nr), sharex=True, sharey=True)
tmW = tm[(tm.W == W_STAR) & (tm.seed == 42)]
for ax, f in zip(axs.flat, ev):
    s = load_series(f, cfg["nab_dir"], labels)
    err, _ = load_errors(f, W_STAR, 42, args.scores)
    y = s.labels[s.b:]
    if 0 < y.sum() < len(y):
        fpr_c, tpr_c, _ = roc_curve(y, err[s.b:])
        ax.plot(fpr_c, tpr_c, color="#9a9994", lw=1.2)
    for k in PRIMARY:
        r = tmW[(tmW.file == f) & (tmW.method == k)]
        if len(r):
            ax.scatter(r.fpr, r.recall, s=40, color=COLORS[k], edgecolor="white", lw=1, zorder=3)
    ax.plot([0, 1], [0, 1], ls=":", color=viz.GRID, lw=1)
    ax.set_title(s.name[:26], fontsize=8)
for ax in axs.flat[len(ev):]:
    ax.axis("off")
fig.supxlabel("false-positive rate"); fig.supylabel("point recall")
fig.legend(handles=[plt.Line2D([], [], color="#9a9994", label="LSTM-AE ROC curve (all thresholds)")] +
           [plt.Line2D([], [], ls="", marker="o", color=COLORS[k], label=LABELS[k]) for k in PRIMARY],
           loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02), fontsize=8)
fig.suptitle("Same model, different decisions: each threshold rule picks a point on (or near) the same ROC curve",
             fontweight="bold")
fig.tight_layout(rect=(0, 0.03, 1, 1)); save(fig, "results/roc_operating_points.png")

# 14. alarm outcome breakdown (pooled over files, W*, seeds averaged)
g = pf[pf.method.isin(PRIMARY + ABLATION)].groupby("method")[["tp", "fp", "fn"]].sum()
g = g.loc[PRIMARY + ABLATION]
fig, ax = plt.subplots(figsize=(12, 4.2))
left = np.zeros(len(g))
for col, c, lab in (("tp", "#1baf7a", "true alarms (TP)"), ("fp", "#eb6834", "false alarms (FP)"),
                    ("fn", "#9a9994", "missed anomaly points (FN)")):
    share = g[col] / g.sum(axis=1)
    ax.barh(range(len(g)), share, left=left, color=c, label=lab, height=0.62, edgecolor="white", lw=2)
    for i, (l, v) in enumerate(zip(left, share)):
        if v > 0.06:
            ax.text(l + v / 2, i, f"{v:.0%}", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
    left += share
ax.set_yticks(range(len(g))); ax.set_yticklabels([LABELS[k] for k in g.index], fontsize=8.5); ax.invert_yaxis()
ax.set_xlim(0, 1); ax.set_xlabel("share of (TP + FP + FN) points, pooled over evaluation files")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8.5)
ax.set_title("Where do the alarms go?", pad=26)
save(fig, "results/alarm_outcomes.png")

# 15. dataset composition (donut) + anomaly share
eda = pd.read_csv(ROOT / "results/eda_summary.csv")
eva = eda[eda.evaluation]
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
dom_n = eva.groupby("domain")["n"].sum().sort_values(ascending=False)
pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
w, _ = axs[0].pie(dom_n, colors=pal[:len(dom_n)], startangle=90, counterclock=False,
                  wedgeprops=dict(width=0.38, edgecolor="white", linewidth=2))
axs[0].legend(w, [f"{d}: {v:,} pts ({v / dom_n.sum():.0%}), {int((eva.domain == d).sum())} files"
                  for d, v in dom_n.items()], loc="upper center", bbox_to_anchor=(0.5, 0.02), fontsize=8.5, ncol=1)
axs[0].text(0, 0, f"{dom_n.sum():,}\npoints", ha="center", va="center", fontsize=11, fontweight="bold")
axs[0].set_title("Evaluation data by domain")
e2 = eva.sort_values("anomaly_point_share")
axs[1].barh(e2["name"], e2.anomaly_point_share * 100, color=viz.ANOM, height=0.6)
axs[1].set_xlabel("% of points inside NAB anomaly windows"); axs[1].tick_params(axis="y", labelsize=7)
axs[1].set_title("Class imbalance per file")
fig.tight_layout(); save(fig, "eda/dataset_composition.png")

# 16. radar chart of normalised metrics
cats = [("composite_f1", "Composite F1", False), ("event_recall", "Event recall", False),
        ("precision", "Point precision", False), ("fpr", "1 − FPR", True), ("hv_fpr", "1 − HV-FPR", True),
        ("corpus_nab_standard", "NAB / 100", None)]
ang = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist(); ang += ang[:1]
fig = plt.figure(figsize=(7.5, 7)); ax = fig.add_subplot(polar=True)
for k in PRIMARY:
    v = []
    for col, _, inv in cats:
        x = summ.loc[k, col]
        v.append(x / 100 if inv is None else (1 - x if inv else x))
    v += v[:1]
    ax.plot(ang, v, color=COLORS[k], lw=2.2 if k == P else 1.6, label=LABELS[k])
    ax.fill(ang, v, color=COLORS[k], alpha=0.06)
ax.set_xticks(ang[:-1]); ax.set_xticklabels([c[1] for c in cats], fontsize=9); ax.set_ylim(0, 1)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), fontsize=8, ncol=2)
ax.set_title("Overall profile (outer = better)", pad=18)
save(fig, "results/radar_profile.png")

# 17. average rank across files (lower = better)
fig, axs = plt.subplots(1, 3, figsize=(15, 3.6))
for ax, (m, t, asc) in zip(axs, (("composite_f1", "Composite F1", False), ("nab_standard", "NAB standard", False),
                                 ("hv_fpr", "High-variance FPR", True))):
    tab = pf[pf.method.isin(PRIMARY)].pivot_table(index="file", columns="method", values=m)
    rk = tab.rank(axis=1, ascending=asc).mean().sort_values()
    ax.scatter(rk.values, np.zeros(len(rk)), s=160, c=[COLORS[k] for k in rk.index], zorder=3, edgecolor="white", lw=1.5)
    for i, (k, v) in enumerate(rk.items()):
        ax.annotate(f"{SHORT[k]}  {v:.2f}", (v, 0), xytext=(0, 14 if i % 2 == 0 else -20), textcoords="offset points",
                    ha="center", fontsize=9)
    ax.set_xlim(0.8, 4.2); ax.set_yticks([]); ax.set_xlabel("average rank across files (1 = best)")
    ax.set_title(t); ax.grid(axis="y", visible=False)
fig.suptitle("Average rank of the four main rules over the 18 evaluation files", fontweight="bold")
fig.tight_layout(); save(fig, "results/average_rank.png")

# 18. false alarms on anomaly-free test streams (H2 secondary)
if "clean_alerts_per_1000" in summ:
    cp = pd.read_csv(R / "clean_stream_per_file.csv")
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
    order = PRIMARY + ABLATION
    for ax, (m, t) in zip(axs, (("fpr", "False-positive rate on anomaly-free test streams ↓"),
                                ("alerts_per_1000", "False alerts per 1,000 points ↓"))):
        data = [cp[cp.method == k][m].dropna() for k in order]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False, medianprops=dict(color=INK))
        for b, k in zip(bp["boxes"], order):
            b.set_facecolor(COLORS[k]); b.set_alpha(0.75)
        for i, d in enumerate(data, 1):
            ax.scatter(np.random.default_rng(i).normal(i, 0.06, len(d)), d, s=8, color=INK, alpha=0.4, zorder=3)
        ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels([SHORT[k] for k in order]); ax.set_title(t)
    fig.suptitle(f"H2 secondary evidence: {cp.file.nunique()} test partitions with no labelled anomaly "
                 "(every alarm is false)", fontweight="bold")
    fig.tight_layout(); save(fig, "results/clean_stream_false_alarms.png")

# 19. hypothesis summary card (for the paper / slides)
fig, ax = plt.subplots(figsize=(13, 0.9 + 0.75 * len(hyp["hypotheses"]))); ax.axis("off")
colmap = lambda v: ("#e7f5ec", "#0f7a3d") if v.startswith("Supported (") and "descript" not in v else \
    (("#fbe9e9", "#b42323") if v.startswith("Not") else ("#fdf3dc", "#9a6400"))
for i, h in enumerate(hyp["hypotheses"]):
    y = len(hyp["hypotheses"]) - i - 0.5
    bg, fg = colmap(h["verdict"])
    ax.add_patch(FancyBboxPatch((0.0, y - 0.38), 1.0, 0.76, boxstyle="round,pad=0.0,rounding_size=0.02",
                                fc=bg, ec="none", transform=ax.transData))
    det = "; ".join(f"vs {SHORT[d['b']]}: Δ={d['mean_diff']:+.3f}, p_Holm={d['p_holm']:.3f}" for d in h.get("detail", []))
    ax.text(0.01, y + 0.1, f"{h['hypothesis']} ({h['role']}) — {h['metric']}", fontsize=10, fontweight="bold", color=INK, va="center")
    ax.text(0.99, y + 0.1, h["verdict"], fontsize=10, fontweight="bold", color=fg, va="center", ha="right")
    ax.text(0.01, y - 0.18, det, fontsize=8.5, color=INK2, va="center")
ax.set_xlim(0, 1); ax.set_ylim(0, len(hyp["hypotheses"]))
ax.set_title(f"Hypothesis decisions (rules fixed in PROTOCOL.md; P1 vs fixed baselines; W* = {W_STAR})", fontweight="bold")
save(fig, "results/hypothesis_summary.png")
print("extra figures written")
