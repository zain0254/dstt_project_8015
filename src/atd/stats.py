"""Dataset-level statistics and the pre-declared hypothesis decisions.

Unit of analysis = one evaluation file. For each file and method, metrics
are averaged over the three seeds (at the selected window size W*), so every
file counts once. Comparisons are paired across files.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj.tolist()


def per_file(test_df: pd.DataFrame, W: int) -> pd.DataFrame:
    d = test_df[test_df.W == W]
    num = d.select_dtypes("number").columns.drop(["W", "seed"], errors="ignore")
    return d.groupby(["file", "domain", "method"], as_index=False)[list(num)].mean()


def corpus_nab(test_df: pd.DataFrame, W: int, profile="standard") -> pd.DataFrame:
    """Official-style corpus score: sum raw / null / perfect over files, then
    normalise; computed per seed and then averaged over seeds."""
    d = test_df[test_df.W == W]
    g = d.groupby(["method", "seed"])[[f"nab_{profile}_raw", f"nab_{profile}_null", f"nab_{profile}_perfect"]].sum()
    g["score"] = 100 * (g.iloc[:, 0] - g.iloc[:, 1]) / (g.iloc[:, 2] - g.iloc[:, 1])
    out = g.groupby("method")["score"].agg(["mean", "std"]).reset_index()
    out.columns = ["method", f"corpus_nab_{profile}", f"corpus_nab_{profile}_sd"]
    return out


def paired(pf: pd.DataFrame, metric: str, a: str, b: str, alternative="two-sided", seed=0) -> dict:
    """Compare method a vs b on `metric` across files (a - b)."""
    wa = pf[pf.method == a].set_index("file")[metric]
    wb = pf[pf.method == b].set_index("file")[metric]
    idx = wa.dropna().index.intersection(wb.dropna().index)
    diff = (wa[idx] - wb[idx]).to_numpy()
    res = {"metric": metric, "a": a, "b": b, "n_files": len(idx),
           "mean_a": float(wa[idx].mean()) if len(idx) else np.nan,
           "mean_b": float(wb[idx].mean()) if len(idx) else np.nan,
           "mean_diff": float(diff.mean()) if len(diff) else np.nan,
           "median_diff": float(np.median(diff)) if len(diff) else np.nan,
           "wins": int((diff > 1e-12).sum()), "ties": int((np.abs(diff) <= 1e-12).sum()),
           "losses": int((diff < -1e-12).sum())}
    nz = diff[np.abs(diff) > 1e-12]
    if len(nz) >= 1:
        try:
            res["p_value"] = float(wilcoxon(diff, zero_method="wilcox", alternative=alternative).pvalue)
        except ValueError:
            res["p_value"] = np.nan
        # matched-pairs rank-biserial correlation
        r = pd.Series(np.abs(nz)).rank().to_numpy()
        res["rank_biserial"] = float((r[nz > 0].sum() - r[nz < 0].sum()) / r.sum())
    else:
        res["p_value"], res["rank_biserial"] = 1.0, 0.0
    rng = np.random.default_rng(seed)
    if len(diff) > 1:
        boots = [rng.choice(diff, len(diff)).mean() for _ in range(5000)]
        res["ci95_low"], res["ci95_high"] = np.percentile(boots, [2.5, 97.5]).tolist()
    else:
        res["ci95_low"] = res["ci95_high"] = np.nan
    return res


def comparison_table(pf, proposed, others, metrics):
    rows = []
    for m in metrics:
        fam = [paired(pf, m, proposed, o) for o in others]
        adj = holm([r["p_value"] for r in fam])
        for r, h in zip(fam, adj):
            r["p_holm"] = h
            rows.append(r)
    return pd.DataFrame(rows)


def decide_hypotheses(pf, corpus, cfg) -> list[dict]:
    """Decision rules fixed in PROTOCOL.md before the run."""
    P = cfg["proposed_method"]
    B = cfg["fixed_baselines"]
    alpha = cfg["alpha"]
    out = []

    # H1: higher average F1 than at least one fixed baseline (primary = composite F1)
    for metric, role in (("composite_f1", "primary"), ("f1", "secondary (point F1)")):
        fam = [paired(pf, metric, P, b) for b in B]
        adj = holm([r["p_value"] for r in fam])
        better = [r for r in fam if r["mean_diff"] > 0]
        sig = [r for r, h in zip(fam, adj) if r["mean_diff"] > 0 and h < alpha]
        verdict = ("Supported (statistically significant)" if sig else
                   "Supported descriptively (not significant)" if better else "Not supported")
        out.append({"hypothesis": "H1", "role": role, "metric": metric, "verdict": verdict,
                    "detail": [{**r, "p_holm": h} for r, h in zip(fam, adj)]})

    # H2: lower FPR during high-variance normal periods on some series
    fam = [paired(pf, "hv_fpr", P, b) for b in B]
    adj = holm([r["p_value"] for r in fam])
    n = fam[0]["n_files"]
    lower_both = all(r["mean_diff"] < 0 for r in fam)
    majority = all(r["losses"] >= n / 2 for r in fam)  # losses of (P - B) = P lower
    sig = all(r["mean_diff"] < 0 and h < alpha for r, h in zip(fam, adj))
    verdict = ("Supported (statistically significant)" if lower_both and sig else
               "Supported descriptively (not significant)" if lower_both and majority else
               "Partially supported" if any(r["losses"] > 0 for r in fam) else "Not supported")
    out.append({"hypothesis": "H2", "role": "primary", "metric": "hv_fpr", "verdict": verdict,
                "detail": [{**r, "p_holm": h} for r, h in zip(fam, adj)]})

    # H3: maintain or improve NAB score (corpus, standard profile) vs fixed baselines
    c = corpus.set_index("method")["corpus_nab_standard"]
    diffs = {b: float(c[P] - c[b]) for b in B}
    fam = [paired(pf, "nab_standard", P, b) for b in B]
    adj = holm([r["p_value"] for r in fam])
    if all(d >= 0 for d in diffs.values()):
        verdict = "Supported (improves NAB score)"
    elif all(d >= -5 for d in diffs.values()):
        verdict = "Supported (maintains NAB score within 5 points)"
    else:
        verdict = "Not supported"
    out.append({"hypothesis": "H3", "role": "primary", "metric": "corpus nab_standard", "verdict": verdict,
                "corpus_diffs": diffs, "detail": [{**r, "p_holm": h} for r, h in zip(fam, adj)]})
    return out
