"""Threshold rules that turn reconstruction errors into anomaly alarms.

All rules see exactly the same error stream. Rolling rules are causal: the
threshold at time t uses only errors from times < t.

Memory types for rolling rules
------------------------------
plain       Every new error enters the moving window unchanged (textbook
            rolling statistics; used by the 2026 base paper's rolling mu+3sigma).
gated       Anomaly-aware memory. An error that exceeded its threshold is
            NOT added to the window, so an anomaly cannot inflate the
            statistics it is judged by (the "masking" failure described in
            the proposal) and the window always holds real, normal errors.
            If alarms persist for more than w consecutive steps, the change
            is accepted as the new normal and raw values enter again.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

EPS_ABS = 1e-12
EPS_REL = 1e-3   # scale floor relative to the location (avoids MAD = 0 on flat streams)


# ---------------------------------------------------------------- fixed rules
def fixed_percentile(train_err: np.ndarray, p: float) -> float:
    """B1: threshold = p-th percentile of training reconstruction errors."""
    return float(np.percentile(train_err, p))


def fixed_mean_ksigma(train_err: np.ndarray, k: float) -> float:
    """B2: threshold = mean + k * std of training reconstruction errors."""
    return float(np.mean(train_err) + k * np.std(train_err))


# -------------------------------------------------------------- rolling rules
def _floor(loc, scale):
    return np.maximum(scale, EPS_ABS + EPS_REL * np.abs(loc))


def rolling_plain_stats(seed: np.ndarray, stream: np.ndarray, w: int, kind: str):
    """Vectorised rolling (location, scale) with plain memory.

    Returns arrays aligned with `stream`: stats for time t use the w values
    immediately before t (taken from seed + stream)."""
    seed = seed[-w:]
    full = np.concatenate([seed, stream])
    win = sliding_window_view(full, w)[: len(stream)]  # row i = w values before stream[i]
    if kind == "median_mad":
        loc = np.median(win, axis=1)
        scale = np.median(np.abs(win - loc[:, None]), axis=1)
    elif kind == "mean_std":
        loc = win.mean(axis=1)
        scale = win.std(axis=1)
    else:
        raise ValueError(kind)
    return loc, _floor(loc, scale)


def rolling_gated(seed: np.ndarray, stream: np.ndarray, w: int, mult: float, kind: str):
    """Sequential rolling threshold with anomaly-aware (gated) memory.

    Returns (alarms, thresholds) aligned with `stream`."""
    buf = np.array(seed[-w:], dtype=float)
    if len(buf) < w:  # pad if the seed is short
        buf = np.concatenate([np.full(w - len(buf), buf[0] if len(buf) else 0.0), buf])
    pos = 0
    n = len(stream)
    thr = np.empty(n)
    alarm = np.zeros(n, dtype=np.int8)
    if kind == "mean_std":
        s1, s2 = buf.sum(), (buf ** 2).sum()
    run = 0  # consecutive alarms
    for t in range(n):
        if kind == "median_mad":
            loc = np.median(buf)
            scale = np.median(np.abs(buf - loc))
        else:
            loc = s1 / w
            scale = np.sqrt(max(s2 / w - loc * loc, 0.0))
        scale = max(scale, EPS_ABS + EPS_REL * abs(loc))
        th = loc + mult * scale
        e = stream[t]
        thr[t] = th
        if e > th:
            alarm[t] = 1
            run += 1
            if run <= w:
                continue          # flagged: memory unchanged
        else:
            run = 0
        store = e                 # normal value, or persistent change accepted as new normal
        if kind == "mean_std":
            old = buf[pos]
            s1 += store - old
            s2 += store * store - old * old
        buf[pos] = store
        pos = (pos + 1) % w
    return alarm, thr


class PlainCache:
    """Caches plain-memory rolling stats per (w, kind) so that every
    multiplier in the grid reuses them."""

    def __init__(self, seed, stream):
        self.seed, self.stream, self._c = seed, stream, {}

    def get(self, w, kind):
        key = (w, kind)
        if key not in self._c:
            self._c[key] = rolling_plain_stats(self.seed, self.stream, w, kind)
        return self._c[key]


def apply_method(method_cfg: dict, method: str, train_err: np.ndarray, stream: np.ndarray,
                 params: dict, cache: PlainCache | None = None, memo: dict | None = None):
    """Return (alarms, thresholds) over `stream` for one method + params.

    `stream` is the error sequence from the start of validation to the end of
    test; rolling rules are seeded with the last w training errors."""
    if method.startswith("B1"):
        th = fixed_percentile(train_err, params["p"])
        return (stream > th).astype(np.int8), np.full(len(stream), th)
    if method.startswith("B2"):
        th = fixed_mean_ksigma(train_err, params["k"])
        return (stream > th).astype(np.int8), np.full(len(stream), th)

    kind = "median_mad" if "median_mad" in method else "mean_std"
    mult = params.get("lam", params.get("k"))
    w = int(params["w"])
    memory = params.get("memory", method_cfg[method].get("memory"))
    key = (kind, memory, w, float(mult))
    if memo is not None and key in memo:
        return memo[key]
    res = _rolling(kind, memory, w, mult, train_err, stream, cache)
    if memo is not None:
        memo[key] = res
    return res


def _rolling(kind, memory, w, mult, train_err, stream, cache):
    if memory == "plain":
        cache = cache or PlainCache(train_err, stream)
        loc, scale = cache.get(w, kind)
        th = loc + mult * scale
        return (stream > th).astype(np.int8), th
    return rolling_gated(train_err, stream, w, mult, kind)


def param_grid(method_cfg: dict, method: str) -> list[dict]:
    cfg = method_cfg[method]
    grid = [{}]
    for k, vals in cfg.items():
        if k == "memory" and isinstance(vals, str):
            continue  # fixed memory type, not tuned
        grid = [dict(g, **{k: v}) for g in grid for v in vals]
    return grid
