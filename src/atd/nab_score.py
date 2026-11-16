"""NAB scoring for binary alarms, ported from numenta/NAB nab/sweeper.py.

Rules (identical to the official scorer):
* Each labelled window contributes once. If at least one alarm falls inside
  it, it scores the best (earliest) alarm via the scaled sigmoid: an alarm at
  the window's left edge earns +tp, one at the right edge earns ~0.
  A window with no alarm costs -fn.
* Every alarm outside windows is a false positive weighted by fp times the
  scaled sigmoid of its distance past the previous window (-1 x fp when far
  away or before any window; close behind a window it costs less).
* Normalised score = 100 * (raw - null) / (perfect - null), where null is a
  detector that never alarms (-fn per window) and perfect detects every
  window at its left edge (+tp per window). 0 = as good as doing nothing,
  100 = perfect, negative = worse than doing nothing.

Here the scored segment is the test partition, windows are clipped to it and
there is no probationary period (the model was already trained on the
training partition).
"""
from __future__ import annotations

import math

import numpy as np


def scaled_sigmoid(pos: float) -> float:
    if pos > 3.0:
        return -1.0
    return 2.0 / (1.0 + math.exp(5.0 * pos)) - 1.0


MAX_TP = scaled_sigmoid(-1.0)


def nab_raw(alarms: np.ndarray, windows: list[tuple[int, int]], tp=1.0, fp=0.11, fn=1.0) -> dict:
    """alarms: 0/1 array for the segment; windows: inclusive index pairs
    relative to the segment, sorted, non-overlapping."""
    alarms = np.asarray(alarms)
    n = len(alarms)
    windows = sorted(windows)
    win_best = {}
    fp_total = 0.0
    n_fp = 0
    wi = 0
    prev_right, prev_width = None, None
    cur = None
    for i in range(n):
        if cur is None and wi < len(windows) and i == windows[wi][0]:
            cur = windows[wi]
            wi += 1
        if alarms[i]:
            if cur is not None:
                s, e = cur
                width = e - s + 1
                pos = -(e - i + 1) / width
                sc = scaled_sigmoid(pos) * tp / MAX_TP
                win_best[cur] = max(win_best.get(cur, -np.inf), sc)
            else:
                if prev_right is None:
                    sc = -1.0
                else:
                    denom = max(prev_width - 1, 1)
                    sc = scaled_sigmoid(abs(prev_right - i) / denom)
                fp_total += sc * fp
                n_fp += 1
        if cur is not None and i == cur[1]:
            prev_right, prev_width = i, cur[1] - cur[0] + 1
            cur = None
    tp_part = sum(win_best.get(w, -fn) for w in windows)
    raw = tp_part + fp_total
    k = len(windows)
    return {"raw": raw, "null": -fn * k, "perfect": tp * k, "n_windows": k,
            "n_detected": len(win_best), "n_fp_alarms": n_fp}


def normalise(raw: float, null: float, perfect: float) -> float:
    if perfect == null:
        return float("nan")
    return 100.0 * (raw - null) / (perfect - null)
