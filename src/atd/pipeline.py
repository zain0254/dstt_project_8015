"""Experiment pipeline: jobs, threshold tuning on validation, test evaluation."""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import ROOT, load_config
from .data import high_variance_mask, load_labels, load_series, scaled_series, file_id
from .metrics import evaluate
from .nab_score import nab_raw
from .metrics import alert_onsets
from .thresholds import PlainCache, apply_method, param_grid

SCORES_DIR = ROOT / "results" / "scores"


# ------------------------------------------------------------------ jobs
def all_jobs(cfg: dict, datasets: dict) -> list[dict]:
    labels = load_labels(cfg["labels_file"])
    jobs = []
    for f in datasets["all_files"]:
        n = len(load_series(f, cfg["nab_dir"], labels).values)
        for W in cfg["window_sizes"]:
            for seed in cfg["seeds"]:
                jobs.append({"file": f, "W": W, "seed": seed, "cost": n * W})
    return jobs


def shard_jobs(jobs: list[dict], shard: int, n_shards: int) -> list[dict]:
    """Longest-processing-time-first assignment: balances shards by cost."""
    loads = [0] * n_shards
    owner = {}
    for j in sorted(jobs, key=lambda j: (-j["cost"], j["file"], j["W"], j["seed"])):
        k = int(np.argmin(loads))
        loads[k] += j["cost"]
        owner[(j["file"], j["W"], j["seed"])] = k
    return [j for j in jobs if owner[(j["file"], j["W"], j["seed"])] == shard]


def score_path(f: str, W: int, seed: int, scores_dir: Path = SCORES_DIR) -> Path:
    return Path(scores_dir) / f"{file_id(f)}__W{W}__s{seed}.npz"


def load_errors(f, W, seed, scores_dir=SCORES_DIR):
    z = np.load(score_path(f, W, seed, scores_dir), allow_pickle=False)
    return z["errors"].astype(float), json.loads(str(z["meta"]))


# --------------------------------------------------------------- context
class Ctx:
    """Everything about one (file, W, seed) needed by threshold methods."""

    def __init__(self, s, errors, W, hv):
        self.s, self.W = s, W
        self.err = errors
        self.train_err = errors[W - 1: s.a]
        self.stream = errors[s.a:]            # validation + test, contiguous
        self.val_len = s.b - s.a
        self.hv = hv


def _quick_objective(alarms, labels, windows, nab_w):
    """Composite F1 (+ tie-breakers) for tuning."""
    tp = int(((alarms == 1) & (labels == 1)).sum())
    fp = int(((alarms == 1) & (labels == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    det = sum(1 for s, e in windows if alarms[s:e + 1].any())
    r = det / len(windows)
    cf1 = 2 * p * r / (p + r) if p + r else 0.0
    nr = nab_raw(alert_onsets(alarms), windows, nab_w["tp"], nab_w["fp"], nab_w["fn"])
    nab = 100 * (nr["raw"] - nr["null"]) / (nr["perfect"] - nr["null"])
    return cf1, nab, float(alarms.mean())


def _tune_one(args):
    """Validation objective of every candidate, for one (file, W, seed)."""
    f, W, seed, cfg_path, scores_dir = args
    cfg = load_config(cfg_path)
    labels = load_labels(cfg["labels_file"])
    s = load_series(f, cfg["nab_dir"], labels)
    err, _ = load_errors(f, W, seed, scores_dir)
    ctx = Ctx(s, err, W, None)
    val_stream = ctx.stream[: ctx.val_len]
    val_lab = s.labels[s.a: s.b]
    val_win = [(x - s.a, y - s.a) for x, y in s.windows_in(s.a, s.b)]
    cache = PlainCache(ctx.train_err, val_stream)
    rows = []
    mcfg = cfg["thresholds"]
    for m in mcfg:
        for prm in param_grid(mcfg, m):
            al, _ = apply_method(mcfg, m, ctx.train_err, val_stream, prm, cache)
            cf1, nab, rate = _quick_objective(al, val_lab, val_win, cfg["nab_profiles"]["standard"])
            rows.append({"file": f, "W": W, "seed": seed, "method": m,
                         "params": json.dumps(prm, sort_keys=True),
                         "val_composite_f1": cf1, "val_nab_standard": nab, "val_alarm_rate": rate})
    # threshold-free model quality on validation (for W selection)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(val_lab, val_stream) if 0 < val_lab.sum() < len(val_lab) else np.nan
    return rows, {"file": f, "W": W, "seed": seed, "val_roc_auc": auc}


def tune(cfg_path="configs/experiment.json", datasets=None, scores_dir=SCORES_DIR, workers=2):
    cfg = load_config(cfg_path)
    args = [(f, W, sd, cfg_path, str(scores_dir)) for f in datasets["tuning_files"]
            for W in cfg["window_sizes"] for sd in cfg["seeds"]]
    rows, aucs = [], []
    with ProcessPoolExecutor(workers) as ex:
        for r, a in ex.map(_tune_one, args, chunksize=1):
            rows += r
            aucs.append(a)
    cand = pd.DataFrame(rows)
    aucs = pd.DataFrame(aucs)
    # aggregate: mean over tuning files and seeds, per (method, W, params)
    agg = (cand.groupby(["method", "W", "params"], as_index=False)
           [["val_composite_f1", "val_nab_standard", "val_alarm_rate"]].mean())
    agg = agg.sort_values(["method", "W", "val_composite_f1", "val_nab_standard", "val_alarm_rate"],
                          ascending=[True, True, False, False, True])
    selected = agg.groupby(["method", "W"], as_index=False).head(1).reset_index(drop=True)
    wsel = aucs.groupby("W", as_index=False)["val_roc_auc"].mean()
    w_star = int(wsel.sort_values(["val_roc_auc", "W"], ascending=[False, True]).iloc[0]["W"])
    return cand, agg, selected, wsel, w_star


# ----------------------------------------------------------------- test
def _test_one(args):
    f, W, seed, cfg_path, scores_dir, selected = args
    cfg = load_config(cfg_path)
    labels = load_labels(cfg["labels_file"])
    s = load_series(f, cfg["nab_dir"], labels)
    err, meta = load_errors(f, W, seed, scores_dir)
    x = scaled_series(s)
    hv = high_variance_mask(s, x, cfg["high_variance"]["rolling_window"],
                            cfg["high_variance"]["percentile_of_train"])
    ctx = Ctx(s, err, W, hv)
    test_win = [(x0 - s.b, y0 - s.b) for x0, y0 in s.windows_in(s.b, s.n)]
    test_lab = s.labels[s.b:]
    test_scores = err[s.b:]
    cache = PlainCache(ctx.train_err, ctx.stream)
    rows, traces = [], {}
    for m, prm in selected.items():
        al, th = apply_method(cfg["thresholds"], m, ctx.train_err, ctx.stream, prm, cache)
        al_t = al[ctx.val_len:]
        met = evaluate(al_t, test_lab, test_win, test_scores, hv[s.b:], cfg["nab_profiles"])
        rows.append({"file": f, "domain": s.domain, "W": W, "seed": seed, "method": m,
                     "params": json.dumps(prm, sort_keys=True), **met})
        traces[m] = th[ctx.val_len:].astype(np.float32)
    return rows, (f, W, seed, traces)


def test(selected_df: pd.DataFrame, cfg_path="configs/experiment.json", datasets=None,
         scores_dir=SCORES_DIR, workers=2, keep_traces_for_W=None):
    cfg = load_config(cfg_path)
    args = []
    for W in cfg["window_sizes"]:
        sel = {r.method: json.loads(r.params) for r in selected_df[selected_df.W == W].itertuples()}
        for f in datasets["evaluation_files"]:
            for sd in cfg["seeds"]:
                args.append((f, W, sd, cfg_path, str(scores_dir), sel))
    rows, traces = [], {}
    with ProcessPoolExecutor(workers) as ex:
        for r, (f, W, sd, tr) in ex.map(_test_one, args, chunksize=1):
            rows += r
            if keep_traces_for_W is not None and W == keep_traces_for_W:
                traces[(f, sd)] = tr
    return pd.DataFrame(rows), traces
