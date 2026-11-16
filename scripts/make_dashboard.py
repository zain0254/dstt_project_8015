"""Build dashboard/index.html - one self-contained interactive page (Plotly).

    python scripts/make_dashboard.py [--results results] [--scores results/scores]
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from atd import ROOT, load_config
from atd.data import load_labels, load_series, file_id
from atd.pipeline import load_errors
from atd.viz import COLORS, LABELS, SHORT, PRIMARY, ABLATION

ap = argparse.ArgumentParser()
ap.add_argument("--results", default=str(ROOT / "results"))
ap.add_argument("--scores", default=str(ROOT / "results/scores"))
ap.add_argument("--out", default=str(ROOT / "dashboard/index.html"))
ap.add_argument("--title", default="Robust Adaptive Thresholding for LSTM-AE Anomaly Detection — Results")
args = ap.parse_args()
R = Path(args.results)
cfg = load_config()
labels = load_labels(cfg["labels_file"])
ds = json.loads((ROOT / "configs/datasets.json").read_text())
hyp = json.loads((R / "hypotheses.json").read_text())
W = hyp["W_star"]
summ = pd.read_csv(R / "summary_by_method.csv")
pf = pd.read_csv(R / "per_file_metrics.csv")
pf["name"] = pf.file.str.split("/").str[-1].str.replace(".csv", "", regex=False)
comp = pd.read_csv(R / "comparisons.csv")
tsum = pd.read_csv(R / "tuning_summary.csv")
sel = pd.read_csv(R / "selected_params.csv")
wsel = pd.read_csv(R / "window_selection.csv")
eda = pd.read_csv(R / "eda_summary.csv") if (R / "eda_summary.csv").exists() else pd.read_csv(ROOT / "results/eda_summary.csv")
tm = pd.read_csv(R / "test_metrics.csv")


def rnd(x, k=4):
    if isinstance(x, float):
        return None if not np.isfinite(x) else round(x, k)
    return x


def records(df):
    return [{k: rnd(v) if isinstance(v, float) else (v.item() if hasattr(v, "item") else v) for k, v in r.items()}
            for r in df.to_dict("records")]


def bucket(arr, n_out, how):
    arr = np.asarray(arr, dtype=float)
    if len(arr) <= n_out:
        return arr
    edges = np.linspace(0, len(arr), n_out + 1).astype(int)
    f = {"max": np.nanmax, "mean": np.nanmean, "first": lambda a: a[0]}[how]
    return np.array([f(arr[a:b]) for a, b in zip(edges[:-1], edges[1:])])


files = {}
for f in ds["evaluation_files"]:
    s = load_series(f, cfg["nab_dir"], labels)
    N = 1400
    idx = np.linspace(0, s.n - 1, min(N, s.n)).astype(int)
    rec = {"name": s.name, "domain": s.domain, "n": s.n,
           "series": {"t": [str(s.timestamps[i])[:16] for i in idx],
                      "v": [round(float(x), 4) for x in bucket(s.values, len(idx), "mean")]},
           "a": str(s.timestamps[s.a])[:16], "b": str(s.timestamps[s.b])[:16],
           "windows": [[str(s.timestamps[x])[:16], str(s.timestamps[y])[:16]] for x, y in s.windows]}
    trp = R / "traces" / f"{file_id(f)}__s42.npz"
    if trp.exists():
        tr = np.load(trp)
        err, _ = load_errors(f, W, 42, args.scores)
        e = err[s.b:]
        m = min(1500, len(e))
        edges = np.linspace(0, len(e), m + 1).astype(int)
        t_idx = edges[:-1] + s.b
        rec["trace"] = {"t": [str(s.timestamps[i])[:16] for i in t_idx],
                        "err": [round(float(x), 6) for x in bucket(e, m, "max")],
                        "thr": {k: [round(float(x), 6) for x in bucket(tr[k], m, "mean")] for k in tr.files},
                        "alarms": {k: [int(x) for x in bucket((e > tr[k]).astype(float), m, "max")] for k in tr.files}}
    files[f] = rec

tsum_p = tsum[tsum.method.isin(["P1_moving_median_mad"])].copy()
data = {
    "title": args.title, "W": W, "proposed": cfg["proposed_method"],
    "primary": PRIMARY, "ablation": ABLATION, "colors": COLORS, "labels": LABELS, "short": SHORT,
    "hypotheses": hyp["hypotheses"], "summary": records(summ), "per_file": records(pf),
    "comparisons": records(comp), "selected": records(sel[sel.W == W]), "wsel": records(wsel),
    "tuning_p1": records(tsum_p[tsum_p.W == W]),
    "by_window": records(tm.groupby(["W", "method"])[["composite_f1", "f1", "event_recall", "fpr", "hv_fpr", "nab_standard"]]
                         .mean().reset_index()),
    "eda": records(eda[[c for c in ["name", "domain", "n", "start", "end", "median_step_min", "missing_filled",
                                    "duplicates_removed", "mean", "std", "min", "max", "skew", "windows_total",
                                    "windows_train", "windows_val", "windows_test", "evaluation", "tuning"] if c in eda]]),
    "files": files, "n_eval": len(ds["evaluation_files"]), "n_tune": len(ds["tuning_files"]),
}
tpl = (ROOT / "scripts/dashboard_template.html").read_text()
html = tpl.replace("__TITLE__", args.title).replace("__DATA__", json.dumps(data, separators=(",", ":"), default=str))
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(html)
print(f"dashboard -> {args.out}  ({len(html)/1e6:.2f} MB)")
