"""Experiment pipeline: jobs, threshold tuning on validation, test evaluation."""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import ROOT, load_config
from .data import high_variance_mask, load_labels, load_series, scaled_series, file_id, normal_train_errors
from .metrics import evaluate
from .nab_score import nab_raw
from .metrics import alert_onsets
from .thresholds import PlainCache, apply_method, param_grid

SCORES_DIR = ROOT / "results" / "split" / "scores"


# ------------------------------------------------------------------ jobs
def all_jobs(cfg: dict, datasets: dict, files_key: str = "all_files", windows=None) -> list[dict]:
    labels = load_labels(cfg["labels_file"])
    jobs = []
    for f in datasets[files_key]:
        n = len(load_series(f, cfg["nab_dir"], labels).values)
        for W in (windows or cfg["window_sizes"]):
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
        self.train_err = normal_train_errors(errors, s, W)
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
    memo = {}
    for m in mcfg:
        for prm in param_grid(mcfg, m):
            al, _ = apply_method(mcfg, m, ctx.train_err, val_stream, prm, cache, memo)
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
    f, W, seed, cfg_path, scores_dir, selected, subset = args[:7]
    split = args[7] if len(args) > 7 else (0.6, 0.2, 0.2)
    cfg = load_config(cfg_path)
    labels = load_labels(cfg["labels_file"])
    s = load_series(f, cfg["nab_dir"], labels, split)
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
    memo = {}
    for m, prm in selected.items():
        al, th = apply_method(cfg["thresholds"], m, ctx.train_err, ctx.stream, prm, cache, memo)
        al_t = al[ctx.val_len:]
        met = evaluate(al_t, test_lab, test_win, test_scores, hv[s.b:], cfg["nab_profiles"])
        met["alerts_per_1000"] = 1000 * met["n_alerts"] / len(al_t)
        rows.append({"set": subset, "file": f, "domain": s.domain, "W": W, "seed": seed, "method": m,
                     "params": json.dumps(prm, sort_keys=True), **met})
        traces[m] = th[ctx.val_len:].astype(np.float32)
    return rows, (f, W, seed, traces)


def test(selected_df: pd.DataFrame, cfg_path="configs/experiment.json", datasets=None,
         scores_dir=SCORES_DIR, workers=2, keep_traces_for_W=None):
    cfg = load_config(cfg_path)
    args = []
    clean = [f for f in datasets["tuning_files"] if f not in datasets["evaluation_files"]]
    for W in cfg["window_sizes"]:
        sel = {r.method: json.loads(r.params) for r in selected_df[selected_df.W == W].itertuples()}
        for subset, files in (("evaluation", datasets["evaluation_files"]), ("clean", clean)):
            for f in files:
                for sd in cfg["seeds"]:
                    args.append((f, W, sd, cfg_path, str(scores_dir), sel, subset))
    rows, traces = [], {}
    with ProcessPoolExecutor(workers) as ex:
        for r, (f, W, sd, tr) in ex.map(_test_one, args, chunksize=1):
            rows += r
            if keep_traces_for_W is not None and W == keep_traces_for_W and f in datasets["evaluation_files"]:
                traces[(f, sd)] = tr
    return pd.DataFrame(rows), traces


# ====================================================== full-corpus protocol (primary)
def _full_candidates(args):
    """Objective of every candidate on the scored part of one (file, seed)."""
    f, W, seed, cfg_path, scores_dir, split = args
    cfg = load_config(cfg_path)
    labels = load_labels(cfg["labels_file"])
    s = load_series(f, cfg["nab_dir"], labels, split)
    err, _ = load_errors(f, W, seed, scores_dir)
    train_err, stream = normal_train_errors(err, s, W), err[s.a:]
    lab = s.labels[s.a:]
    win = [(x - s.a, y - s.a) for x, y in s.windows_in(s.a, s.n)]
    nabw = cfg["nab_profiles"]["standard"]
    cache, memo, rows = PlainCache(train_err, stream), {}, []
    mcfg = cfg["thresholds"]
    for m in mcfg:
        for prm in param_grid(mcfg, m):
            al, _ = apply_method(mcfg, m, train_err, stream, prm, cache, memo)
            nr = nab_raw(alert_onsets(al), win, nabw["tp"], nabw["fp"], nabw["fn"])
            cf1 = _quick_objective(al, lab, win, nabw)[0] if win else np.nan
            rows.append({"file": f, "seed": seed, "method": m, "params": json.dumps(prm, sort_keys=True),
                         "composite_f1": cf1, "nab_raw": nr["raw"], "nab_null": nr["null"],
                         "nab_perfect": nr["perfect"], "alarm_rate": float(al.mean())})
    return rows


def lofo_select(cand: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-file-out selection.

    For every file, each method's parameters are those with the highest mean
    composite F1 over all OTHER files (ties: higher corpus NAB score of the
    other files, then lower alarm rate). A file's own labels never influence
    the parameters it is scored with."""
    g = cand.groupby(["method", "params", "file"], as_index=False)[
        ["composite_f1", "nab_raw", "nab_null", "nab_perfect", "alarm_rate"]].mean()
    files = sorted(g.file.unique())
    chosen, pooled = [], []
    for m, gm in g.groupby("method"):
        P = {c: gm.pivot(index="params", columns="file", values=c).reindex(columns=files)
             for c in ["composite_f1", "nab_raw", "nab_null", "nab_perfect", "alarm_rate"]}
        cf = P["composite_f1"]
        tot_all = pd.DataFrame({"val_composite_f1": cf.mean(axis=1, skipna=True),
                                "val_nab_standard": 100 * (P["nab_raw"].sum(axis=1) - P["nab_null"].sum(axis=1)) /
                                (P["nab_perfect"].sum(axis=1) - P["nab_null"].sum(axis=1)),
                                "val_alarm_rate": P["alarm_rate"].mean(axis=1)})
        pooled.append(tot_all.reset_index().assign(method=m))
        for f in files:
            others = [x for x in files if x != f]
            obj = cf[others].mean(axis=1, skipna=True)
            nab = (100 * (P["nab_raw"][others].sum(axis=1) - P["nab_null"][others].sum(axis=1)) /
                   (P["nab_perfect"][others].sum(axis=1) - P["nab_null"][others].sum(axis=1)))
            rate = P["alarm_rate"][others].mean(axis=1)
            t = pd.DataFrame({"obj": obj.round(10), "nab": nab.round(8), "rate": rate}).reset_index()
            best = t.sort_values(["obj", "nab", "rate", "params"], ascending=[False, False, True, True]).iloc[0]
            chosen.append({"file": f, "method": m, "params": best.params, "lofo_composite_f1": best.obj})
    return pd.DataFrame(chosen), pd.concat(pooled, ignore_index=True)


def evaluate_full(W: int, split, scores_dir, cfg_path="configs/experiment.json", datasets=None, workers=2):
    cfg = load_config(cfg_path)
    files = datasets["full_corpus_files"]
    args = [(f, W, sd, cfg_path, str(scores_dir), split) for f in files for sd in cfg["seeds"]]
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(_full_candidates, args, chunksize=1):
            rows += r
    cand = pd.DataFrame(rows)
    chosen, pooled = lofo_select(cand)
    sel = {f: {r.method: json.loads(r.params) for r in d.itertuples()} for f, d in chosen.groupby("file")}
    targs = [(f, W, sd, cfg_path, str(scores_dir), sel[f],
              "evaluation" if f in datasets["full_eval_files"] else "clean", split)
             for f in files for sd in cfg["seeds"]]
    out, traces = [], {}
    with ProcessPoolExecutor(workers) as ex:
        for r, (f, _, sd, tr) in ex.map(_test_one, targs, chunksize=1):
            out += r
            traces[(f, sd)] = tr
    return cand, chosen, pooled, pd.DataFrame(out), traces
