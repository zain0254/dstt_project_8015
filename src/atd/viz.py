"""Shared figure style (validated categorical palette, fixed per method)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "P1_moving_median_mad": "#2a78d6",
    "B1_fixed_percentile": "#eb6834",
    "B2_fixed_mean_ksigma": "#1baf7a",
    "B3_rolling_mean_3sigma": "#eda100",
    "A1_median_mad_plain": "#e87ba4",
    "A2_median_mad_gated": "#008300",
    "A3_mean_ksigma_plain": "#6250d6",
    "A4_mean_ksigma_gated": "#e34948",
}
LABELS = {
    "P1_moving_median_mad": "P1 Moving median + λ·MAD (proposed)",
    "B1_fixed_percentile": "B1 Fixed percentile",
    "B2_fixed_mean_ksigma": "B2 Fixed mean + k·σ",
    "B3_rolling_mean_3sigma": "B3 Rolling mean + 3σ (base paper)",
    "A1_median_mad_plain": "A1 Median+MAD, plain memory",
    "A2_median_mad_gated": "A2 Median+MAD, gated memory",
    "A3_mean_ksigma_plain": "A3 Rolling mean+kσ, plain memory",
    "A4_mean_ksigma_gated": "A4 Rolling mean+kσ, gated memory",
}
SHORT = {k: v.split(" ")[0] for k, v in LABELS.items()}
PRIMARY = ["B1_fixed_percentile", "B2_fixed_mean_ksigma", "B3_rolling_mean_3sigma", "P1_moving_median_mad"]
ABLATION = ["A1_median_mad_plain", "A2_median_mad_gated", "A3_mean_ksigma_plain", "A4_mean_ksigma_gated"]
INK, INK2, GRID, ANOM = "#0b0b0b", "#52514e", "#e6e5e0", "#e34948"
SPLIT_COLORS = {"train": "#f2f1ec", "validation": "#e8f0fb", "test": "#fdf0e9"}


def setup():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "axes.axisbelow": True,
    })


def shade_split(ax, s, x=None):
    """Shade train / validation / test partitions and anomaly windows."""
    ts = s.timestamps if x is None else x
    ax.axvspan(ts[0], ts[s.a - 1], color=SPLIT_COLORS["train"], zorder=0, lw=0)
    ax.axvspan(ts[s.a], ts[s.b - 1], color=SPLIT_COLORS["validation"], zorder=0, lw=0)
    ax.axvspan(ts[s.b], ts[-1], color=SPLIT_COLORS["test"], zorder=0, lw=0)
    for i, (w0, w1) in enumerate(s.windows):
        ax.axvspan(ts[w0], ts[w1], color=ANOM, alpha=0.18, lw=0, zorder=1,
                   label="NAB anomaly window" if i == 0 else None)
