"""EDA figures for all selected NAB files (proposal Step 1).

Writes figures/eda/*.png and results/eda_summary.csv
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from atd import ROOT, load_config, load_datasets
from atd.data import load_labels, load_series, scaled_series, high_variance_mask
from atd import viz
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

viz.setup()
cfg = load_config()
ds = load_datasets()
labels = load_labels(cfg["labels_file"])
out = ROOT / "figures/eda"; out.mkdir(parents=True, exist_ok=True)
(out / "series").mkdir(exist_ok=True)

rows = []
series = {f: load_series(f, cfg["nab_dir"], labels) for f in ds["all_files"]}
for f, s in series.items():
    v = s.values
    rows.append({
        "file": f, "name": s.name, "domain": s.domain, "n": s.n,
        "start": s.timestamps[0], "end": s.timestamps[-1],
        "median_step_min": float(np.median(np.diff(s.timestamps.asi8)) / 6e10),
        "missing_filled": s.n_missing_filled, "duplicates_removed": s.n_duplicates_removed,
        "mean": v.mean(), "median": np.median(v), "std": v.std(), "min": v.min(), "max": v.max(),
        "skew": pd.Series(v).skew(),
        "windows_total": len(s.windows),
        "windows_train": len(s.windows_in(0, s.a)), "windows_val": len(s.windows_in(s.a, s.b)),
        "windows_test": len(s.windows_in(s.b, s.n)),
        "anomaly_point_share": s.labels.mean(),
        "evaluation": f in ds["evaluation_files"], "tuning": f in ds["tuning_files"],
    })
eda = pd.DataFrame(rows)
eda.to_csv(ROOT / "results/eda_summary.csv", index=False)

# ---- 1. per-file series figure: value, rolling stats, high-variance zones, histogram
for f, s in series.items():
    x = scaled_series(s)
    hv = high_variance_mask(s, x, cfg["high_variance"]["rolling_window"], cfg["high_variance"]["percentile_of_train"])
    rm = pd.Series(s.values).rolling(50, min_periods=1).mean()
    rs = pd.Series(s.values).rolling(50, min_periods=10).std()
    fig = plt.figure(figsize=(13, 6.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[2.2, 1], hspace=0.35, wspace=0.3)
    ax = fig.add_subplot(gs[0, :3])
    viz.shade_split(ax, s)
    ax.plot(s.timestamps, s.values, color=viz.INK, lw=0.6, label="value")
    ax.plot(s.timestamps, rm, color="#2a78d6", lw=1.4, label="rolling mean (50)")
    ax.set_title(f"{s.name}  ·  {s.domain}  ·  n={s.n:,}  ·  {len(s.windows)} anomaly windows")
    h, l = ax.get_legend_handles_labels()
    h += [Patch(color=viz.SPLIT_COLORS[k], label=k) for k in ("train", "validation", "test")]
    ax.legend(handles=h, loc="upper left", ncol=6, fontsize=8)
    ax2 = fig.add_subplot(gs[1, :3], sharex=ax)
    viz.shade_split(ax2, s)
    ax2.plot(s.timestamps, rs, color="#6250d6", lw=1, label="rolling std (50)")
    ax2.fill_between(s.timestamps, 0, np.nanmax(rs.to_numpy()) if rs.notna().any() else 1, where=hv,
                     color="#6250d6", alpha=0.12, lw=0, label="high-variance zone (> train P75)")
    ax2.legend(loc="upper left", fontsize=8, frameon=True, facecolor="white", framealpha=0.85, edgecolor="none")
    ax2.set_ylabel("rolling std")
    ax3 = fig.add_subplot(gs[:, 3])
    ax3.hist(s.values[s.labels == 0], bins=50, color="#2a78d6", alpha=0.8, label="normal points", orientation="horizontal")
    if s.labels.sum():
        ax3.hist(s.values[s.labels == 1], bins=50, color=viz.ANOM, alpha=0.7, label="inside anomaly window", orientation="horizontal")
    ax3.set_title("Value distribution")
    ax3.legend(fontsize=8, loc="lower right")
    fig.savefig(out / "series" / f"{s.fid}.png")
    plt.close(fig)

# ---- 2. dataset overview: where anomaly windows fall in the 60/20/20 split
d = eda.sort_values(["evaluation", "domain", "name"], ascending=[False, True, True]).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(12, 0.32 * len(d) + 1.6))
for i, r in d.iterrows():
    s = series[r.file]
    for k, (lo, hi) in zip(("train", "validation", "test"), ((0, s.a), (s.a, s.b), (s.b, s.n))):
        ax.barh(i, (hi - lo) / s.n, left=lo / s.n, color=viz.SPLIT_COLORS[k], edgecolor="white", height=0.8)
    for w0, w1 in s.windows:
        ax.barh(i, (w1 - w0 + 1) / s.n, left=w0 / s.n, color=viz.ANOM, alpha=0.75, height=0.55)
ax.set_yticks(range(len(d)))
ax.set_yticklabels([f"{'● ' if r.evaluation else '○ '}{r['name']}  ({r.domain})" for _, r in d.iterrows()], fontsize=8)
ax.invert_yaxis(); ax.set_xlim(0, 1); ax.grid(False)
ax.set_xlabel("position in the series (fraction of length)")
ax.set_title("Where the NAB anomaly windows fall in the chronological 60 / 20 / 20 split\n"
             "● = evaluation file (anomaly in test)   ○ = tuning-only file (anomaly in validation)")
ax.legend(handles=[Patch(color=viz.SPLIT_COLORS[k], label=k) for k in ("train", "validation", "test")] +
          [Patch(color=viz.ANOM, alpha=0.75, label="anomaly window")], loc="lower right", ncol=4, fontsize=8,
          bbox_to_anchor=(1, -0.12 - 2 / len(d)))
fig.savefig(out / "split_overview.png"); plt.close(fig)

# ---- 3. domain composition and size
ev = eda[eda.evaluation]
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
cnt = ev.groupby("domain").size().sort_values()
axs[0].barh(cnt.index, cnt.values, color="#2a78d6", height=0.6)
for i, v in enumerate(cnt.values):
    axs[0].text(v + 0.05, i, str(v), va="center", color=viz.INK)
axs[0].set_title(f"Evaluation files per domain (total {len(ev)})"); axs[0].set_xlabel("files")
e2 = ev.sort_values("n")
axs[1].barh(e2["name"], e2["n"], color="#1baf7a", height=0.6)
axs[1].set_title("Series length (points)"); axs[1].tick_params(axis="y", labelsize=7)
fig.tight_layout(); fig.savefig(out / "domains_and_sizes.png"); plt.close(fig)

# ---- 4. variability: rolling-std profile normalised per file (why adaptive thresholds)
fig, ax = plt.subplots(figsize=(12, 4))
for f in ds["evaluation_files"]:
    s = series[f]
    rs = pd.Series(scaled_series(s)).rolling(50, min_periods=50).std()
    ax.plot(np.linspace(0, 1, s.n), rs / np.nanmedian(rs[: s.a]), lw=0.7, alpha=0.6)
ax.axhline(1, color=viz.INK2, lw=1, ls="--")
ax.axvline(0.6, color=viz.INK2, lw=0.8); ax.axvline(0.8, color=viz.INK2, lw=0.8)
ax.text(0.3, 0.97, "train", transform=ax.get_xaxis_transform(), ha="center", va="top", color=viz.INK2)
ax.text(0.7, 0.97, "validation", transform=ax.get_xaxis_transform(), ha="center", va="top", color=viz.INK2)
ax.text(0.9, 0.97, "test", transform=ax.get_xaxis_transform(), ha="center", va="top", color=viz.INK2)
ax.set_yscale("log"); ax.set_xlabel("position in series"); ax.set_ylabel("rolling std ÷ median train rolling std")
ax.set_title("Normal variability is not constant: local volatility drifts by an order of magnitude (18 evaluation files)")
fig.savefig(out / "variability_drift.png"); plt.close(fig)
print(f"EDA: {len(series)} series figures + 3 overview figures -> {out}")
print(eda[eda.evaluation][["name", "domain", "n", "windows_test", "missing_filled", "duplicates_removed"]].to_string(index=False))


# ---- 5. corpus overview: all 58 files, both protocols
ds_all = ds.get("full_corpus_files", ds["all_files"])
allser = {f: (series[f] if f in series else load_series(f, cfg["nab_dir"], labels)) for f in ds_all}
rows_all = []
for f, s in allser.items():
    rows_all.append({"file": f, "name": s.name, "domain": s.domain, "n": s.n, "windows": len(s.windows),
                     "anomaly_points": int(s.labels.sum()),
                     "protocol_A": "evaluation" if f in ds["evaluation_files"] else ("tuning" if f in ds["tuning_files"] else "unused"),
                     "protocol_B": "scored (has anomaly)" if f in ds.get("full_eval_files", []) else "scored (anomaly-free)"})
corpus = pd.DataFrame(rows_all)
corpus.to_csv(ROOT / "results/corpus_overview.csv", index=False)
fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
dom_files = corpus.groupby("domain").size().sort_values()
dom_win = corpus.groupby("domain")["windows"].sum().reindex(dom_files.index)
axs[0].barh(dom_files.index, dom_files.values, color="#2a78d6", height=0.6, label="files")
for i, (v, w) in enumerate(zip(dom_files.values, dom_win.values)):
    axs[0].text(v + 0.2, i, f"{v} files · {w} windows", va="center", fontsize=8.5)
axs[0].set_xlim(0, dom_files.max() * 1.6); axs[0].set_title(f"NAB corpus: {len(corpus)} files, {corpus.windows.sum()} anomaly windows")
useA = corpus.protocol_A.value_counts().reindex(["evaluation", "tuning", "unused"]).fillna(0)
useB = corpus.protocol_B.value_counts()
axs[1].bar(["evaluation\n(test anomaly)", "tuning only\n(val anomaly)", "unused"], useA.values,
           color=["#2a78d6", "#a9c8f0", "#d5d4cf"], width=0.6)
for i, v in enumerate(useA.values):
    axs[1].text(i, v, int(v), ha="center", va="bottom")
axs[1].set_title("Split protocol (60/20/20): how files are used")
axs[2].bar(["scored,\nhas anomaly", "scored,\nanomaly-free"], [useB.get("scored (has anomaly)", 0), useB.get("scored (anomaly-free)", 0)],
           color=["#2a78d6", "#a9c8f0"], width=0.5)
for i, v in enumerate([useB.get("scored (has anomaly)", 0), useB.get("scored (anomaly-free)", 0)]):
    axs[2].text(i, v, int(v), ha="center", va="bottom")
axs[2].set_title("Full-corpus protocol (primary): every file scored")
fig.tight_layout(); fig.savefig(out / "corpus_overview.png"); plt.close(fig)

# ---- 6. preprocessing: raw -> cleaned -> scaled -> windows (one worked example)
ex = series["realKnownCause/machine_temperature_system_failure.csv"] if "realKnownCause/machine_temperature_system_failure.csv" in series else next(iter(series.values()))
raw = pd.read_csv(ROOT / cfg["nab_dir"] / ex.nab_file)
raw["timestamp"] = pd.to_datetime(raw["timestamp"])
x = scaled_series(ex)
Wd = 50
fig = plt.figure(figsize=(13, 8))
gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.25)
ax = fig.add_subplot(gs[0, :])
ax.plot(raw["timestamp"], raw["value"], color=viz.INK, lw=0.5)
ax.set_title(f"1. Raw file: {ex.name} — {len(raw):,} rows, {raw['timestamp'].duplicated().sum()} duplicate timestamps, "
             f"{int(raw['value'].isna().sum())} missing values")
ax = fig.add_subplot(gs[1, :])
viz.shade_split(ax, ex)
ax.plot(ex.timestamps, x, color="#2a78d6", lw=0.6)
ax.set_ylim(-0.05, 1.3)
ax.set_title(f"2. Cleaned (sorted, de-duplicated, forward-filled) and min-max scaled with the training partition only → "
             f"{ex.n:,} points; values above 1 are possible in validation/test (scaler never sees them)")
ax.legend(handles=[Patch(color=viz.SPLIT_COLORS[k], label=k) for k in ("train", "validation", "test")] +
          [Patch(color=viz.ANOM, alpha=0.3, label="anomaly window")], loc="upper left", ncol=4, fontsize=8)
ax = fig.add_subplot(gs[2, 0])
t0 = ex.a - 200
for k, c in zip(range(0, 150, 50), ["#2a78d6", "#1baf7a", "#eb6834"]):
    ax.plot(range(t0 + k, t0 + k + Wd), x[t0 + k: t0 + k + Wd] + k / 400, color=c, lw=1.6, label=f"window ending at t={t0 + k + Wd - 1}")
ax.set_title(f"3. Sliding windows of W = {Wd} points (stride 1), offset for clarity"); ax.legend(fontsize=7.5); ax.set_xlabel("index")
ax = fig.add_subplot(gs[2, 1])
nw = [max(0, ex.a - W + 1) for W in cfg["window_sizes"]]
ax.bar([f"W={W}" for W in cfg["window_sizes"]], nw, color="#2a78d6", width=0.5)
ax.axhline(cfg["model"]["max_train_windows"], color=viz.ANOM, ls="--", label=f"cap used for training ({cfg['model']['max_train_windows']})")
for i, v in enumerate(nw):
    ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8.5)
ax.set_title("4. Training windows available in the 60 % partition"); ax.legend(fontsize=8)
fig.suptitle("Preprocessing pipeline on one file", fontweight="bold", y=0.995)
fig.savefig(out / "preprocessing_example.png"); plt.close(fig)

# ---- 7. seasonality / autocorrelation per domain (why sequence models)
fig, axs = plt.subplots(1, 5, figsize=(16, 3.2), sharey=True)
doms = sorted(set(s.domain for s in series.values() if s.domain != "Artificial"))
for ax, dom in zip(axs, doms):
    for f, s in series.items():
        if s.domain != dom:
            continue
        v = pd.Series(s.values[: s.a])
        lags = np.arange(1, 301)
        ac = [v.autocorr(l) for l in lags]
        ax.plot(lags, ac, lw=0.9, alpha=0.7)
    ax.axhline(0, color=viz.INK2, lw=0.6); ax.set_title(dom); ax.set_xlabel("lag (points)")
axs[0].set_ylabel("autocorrelation (train part)")
fig.suptitle("Temporal dependence by domain: strong periodic autocorrelation motivates a sequence model", fontweight="bold")
fig.tight_layout(); fig.savefig(out / "autocorrelation_by_domain.png"); plt.close(fig)

# ---- 8. anomaly window lengths and positions
lens = [(s.domain, (e - b + 1), (e - b + 1) / s.n) for s in allser.values() for b, e in s.windows]
ld = pd.DataFrame(lens, columns=["domain", "points", "share"])
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
for dom, d in ld.groupby("domain"):
    axs[0].hist(d.points, bins=np.logspace(1, 4, 25), alpha=0.6, label=f"{dom} ({len(d)})")
axs[0].set_xscale("log"); axs[0].set_xlabel("window length (points)"); axs[0].set_ylabel("windows"); axs[0].legend(fontsize=7.5)
axs[0].set_title(f"NAB anomaly windows are long: median {int(ld.points.median())} points")
pos = [(b + e) / 2 / s.n for s in allser.values() for b, e in s.windows]
axs[1].hist(pos, bins=20, color="#2a78d6", edgecolor="white")
axs[1].axvline(0.6, color=viz.INK2, ls="--"); axs[1].axvline(0.8, color=viz.INK2, ls="--"); axs[1].axvline(0.15, color=viz.ANOM, ls=":")
axs[1].text(0.9, axs[1].get_ylim()[1] * 0.9, "test (A)", ha="center", color=viz.INK2)
axs[1].text(0.17, axs[1].get_ylim()[1] * 0.9, "probation end (B)", ha="left", color=viz.ANOM, fontsize=8)
axs[1].set_xlabel("window centre (fraction of series)"); axs[1].set_title("Where anomalies occur within the series")
fig.tight_layout(); fig.savefig(out / "anomaly_windows_stats.png"); plt.close(fig)
print("extra EDA figures written")
