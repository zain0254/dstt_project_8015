"""Evaluation metrics for one alarm sequence on one segment.

Point metrics treat every time step inside a NAB window as positive.
Event metrics treat each NAB window as one event (NAB labels are windows,
not points). Composite F1 (Garg et al., 2021, IEEE TNNLS) is the harmonic
mean of point precision and event recall: it rewards catching each event
while still penalising every false alarm point, and unlike "point-adjust"
it does not inflate scores.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .nab_score import nab_raw


def _safe_div(a, b):
    return float(a) / b if b else float("nan")


def alarm_segments(alarms: np.ndarray) -> list[tuple[int, int]]:
    a = np.concatenate([[0], np.asarray(alarms, dtype=np.int8), [0]])
    d = np.diff(a)
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1
    return list(zip(starts, ends))


def alert_onsets(alarms: np.ndarray) -> np.ndarray:
    """1 at the first step of every alarm segment, else 0."""
    a = np.asarray(alarms, dtype=np.int8)
    prev = np.concatenate([[0], a[:-1]])
    return ((a == 1) & (prev == 0)).astype(np.int8)


def evaluate(alarms: np.ndarray, labels: np.ndarray, windows: list[tuple[int, int]],
             scores: np.ndarray | None = None, hv_mask: np.ndarray | None = None,
             nab_profiles: dict | None = None) -> dict:
    """All metrics for one segment. `windows` are inclusive index pairs
    relative to the segment."""
    alarms = np.asarray(alarms, dtype=np.int8)
    labels = np.asarray(labels, dtype=np.int8)
    tp = int(((alarms == 1) & (labels == 1)).sum())
    fp = int(((alarms == 1) & (labels == 0)).sum())
    fn = int(((alarms == 0) & (labels == 1)).sum())
    tn = int(((alarms == 0) & (labels == 0)).sum())
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    out = {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": prec if tp + fp else 0.0,
        "recall": rec,
        "fpr": _safe_div(fp, fp + tn),
        "accuracy": _safe_div(tp + tn, len(labels)),
        "alarm_rate": float(alarms.mean()) if len(alarms) else float("nan"),
    }
    p = out["precision"]
    out["f1"] = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0

    # events
    detected, delays, rel_delays = 0, [], []
    for s, e in windows:
        hit = np.where(alarms[s:e + 1] == 1)[0]
        if len(hit):
            detected += 1
            delays.append(int(hit[0]))
            rel_delays.append(hit[0] / (e - s + 1))
    k = len(windows)
    out["n_events"] = k
    out["events_detected"] = detected
    out["event_recall"] = _safe_div(detected, k)
    segs = alarm_segments(alarms)
    true_segs = sum(1 for s, e in segs if any(s <= we and e >= ws for ws, we in windows))
    out["n_alarm_segments"] = len(segs)
    out["event_precision"] = _safe_div(true_segs, len(segs)) if segs else 0.0
    ep, er = out["event_precision"], out["event_recall"]
    out["event_f1"] = 2 * ep * er / (ep + er) if k and (ep + er) > 0 else (0.0 if k else float("nan"))
    out["composite_f1"] = 2 * p * er / (p + er) if k and (p + er) > 0 else (0.0 if k else float("nan"))
    out["detection_delay_points"] = float(np.mean(delays)) if delays else float("nan")
    out["detection_delay_rel"] = float(np.mean(rel_delays)) if rel_delays else float("nan")

    # false alarms during high-variance normal periods
    if hv_mask is not None:
        m = hv_mask & (labels == 0)
        out["hv_normal_points"] = int(m.sum())
        out["hv_fpr"] = _safe_div(int((alarms[m] == 1).sum()), int(m.sum()))
        m2 = (~hv_mask) & (labels == 0)
        out["calm_fpr"] = _safe_div(int((alarms[m2] == 1).sum()), int(m2.sum()))

    # threshold-free model quality (identical for every threshold method)
    if scores is not None and 0 < labels.sum() < len(labels):
        out["roc_auc"] = float(roc_auc_score(labels, scores))
        out["pr_auc"] = float(average_precision_score(labels, scores))
    else:
        out["roc_auc"] = out["pr_auc"] = float("nan")

    # NAB: primary version scores alert onsets (one alert when the stream
    # enters an alarm state); the "_pointwise" version scores every alarm step.
    onsets = alert_onsets(alarms)
    out["n_alerts"] = int(onsets.sum())
    for name, w in (nab_profiles or {}).items():
        for suffix, a in (("", onsets), ("_pointwise", alarms)):
            r = nab_raw(a, windows, w["tp"], w["fp"], w["fn"])
            key = f"nab_{name}{suffix}"
            out[f"{key}_raw"] = r["raw"]
            out[f"{key}_null"] = r["null"]
            out[f"{key}_perfect"] = r["perfect"]
            out[key] = 100 * (r["raw"] - r["null"]) / (r["perfect"] - r["null"]) if k else float("nan")
    return out
