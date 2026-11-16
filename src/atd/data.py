"""NAB loading, chronological splitting, scaling and sliding windows.

Conventions
-----------
* A series has n points, indices 0..n-1.
* Split boundaries: a = floor(0.6 n), b = floor(0.8 n).
  train = [0, a), validation = [a, b), test = [b, n).
* The window ending at t is x[t-W+1 : t+1]. Its reconstruction error is the
  anomaly score of time t, so scores exist for t >= W-1 and are causal.
* Point labels: 1 inside a NAB anomaly window (inclusive), else 0.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import ROOT

DOMAINS = {
    "realAWSCloudwatch": "AWS CloudWatch",
    "realAdExchange": "Ad Exchange",
    "realKnownCause": "Known Cause",
    "realTraffic": "Traffic",
    "realTweets": "Twitter",
}


def file_id(nab_file: str) -> str:
    """'realTweets/Twitter_volume_KO.csv' -> 'realTweets__Twitter_volume_KO'"""
    return nab_file.replace(".csv", "").replace("/", "__")


def short_name(nab_file: str) -> str:
    return nab_file.split("/")[-1].replace(".csv", "")


@dataclass
class Series:
    nab_file: str
    timestamps: pd.DatetimeIndex
    values: np.ndarray            # raw values (float)
    labels: np.ndarray            # 0/1 per point
    windows: list[tuple[int, int]]  # inclusive (start, end) indices
    a: int
    b: int
    n_missing_filled: int = 0
    n_duplicates_removed: int = 0
    domain: str = field(init=False)

    def __post_init__(self):
        self.domain = DOMAINS.get(self.nab_file.split("/")[0], "Other")

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def fid(self) -> str:
        return file_id(self.nab_file)

    @property
    def name(self) -> str:
        return short_name(self.nab_file)

    def windows_in(self, lo: int, hi: int) -> list[tuple[int, int]]:
        """Anomaly windows overlapping [lo, hi), clipped to that range."""
        out = []
        for s, e in self.windows:
            if s < hi and e >= lo:
                out.append((max(s, lo), min(e, hi - 1)))
        return out


def load_labels(labels_file: str | Path | None = None) -> dict:
    p = Path(labels_file) if labels_file else ROOT / "data/nab/labels/combined_windows.json"
    if not p.is_absolute():
        p = ROOT / p
    with open(p) as f:
        return json.load(f)


def load_series(nab_file: str, nab_dir: str | Path = "data/nab", labels: dict | None = None,
                split=(0.6, 0.2, 0.2)) -> Series:
    nab_dir = Path(nab_dir)
    if not nab_dir.is_absolute():
        nab_dir = ROOT / nab_dir
    labels = labels if labels is not None else load_labels()
    df = pd.read_csv(nab_dir / nab_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp", kind="stable")
    n_before = len(df)
    df = df.drop_duplicates("timestamp", keep="first")
    n_dup = n_before - len(df)
    n_missing = int(df["value"].isna().sum())
    df["value"] = df["value"].ffill().bfill()
    df = df.reset_index(drop=True)

    ts = pd.DatetimeIndex(df["timestamp"])
    n = len(df)
    lab = np.zeros(n, dtype=np.int8)
    wins = []
    for s, e in labels.get(nab_file, []):
        i = int(ts.searchsorted(pd.Timestamp(s), side="left"))
        j = int(ts.searchsorted(pd.Timestamp(e), side="right")) - 1
        i, j = max(i, 0), min(j, n - 1)
        if j >= i:
            lab[i:j + 1] = 1
            wins.append((i, j))
    a = int(np.floor(split[0] * n))
    b = int(np.floor((split[0] + split[1]) * n))
    return Series(nab_file, ts, df["value"].to_numpy(dtype=float), lab, wins, a, b, n_missing, n_dup)


def select_datasets(labels: dict, nab_dir="data/nab", split=(0.6, 0.2, 0.2)) -> dict:
    """Label-location rule from PROTOCOL.md (uses no model output)."""
    ev, tune = [], []
    for f in sorted(labels):
        if not f.startswith("real"):
            continue
        if not (Path(ROOT / nab_dir) / f).exists():
            continue
        s = load_series(f, nab_dir, labels, split)
        if s.windows_in(s.b, s.n):
            ev.append(f)
        if s.windows_in(s.a, s.b):
            tune.append(f)
    return {"evaluation_files": ev, "tuning_files": tune,
            "all_files": sorted(set(ev) | set(tune))}


class MinMax:
    """Min-max scaler fitted on the training partition only."""

    def fit(self, x: np.ndarray):
        self.lo = float(np.min(x))
        self.hi = float(np.max(x))
        self.span = self.hi - self.lo if self.hi > self.lo else 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.lo) / self.span


def scaled_series(s: Series) -> np.ndarray:
    return MinMax().fit(s.values[: s.a]).transform(s.values)


def make_windows(x: np.ndarray, W: int) -> np.ndarray:
    """All windows; row i is the window ending at t = i + W - 1."""
    from numpy.lib.stride_tricks import sliding_window_view
    return sliding_window_view(x, W).astype(np.float32)


def training_windows(x: np.ndarray, W: int, a: int, max_windows: int) -> np.ndarray:
    """Windows entirely inside the training partition (end index < a).

    Labels are NOT used: the autoencoder is trained unsupervised, as in the
    proposal. If there are more than max_windows, an evenly strided subset is
    taken (keeps chronological coverage, bounds compute)."""
    allw = make_windows(x[:a], W)
    if len(allw) > max_windows:
        idx = np.linspace(0, len(allw) - 1, max_windows).round().astype(int)
        allw = allw[idx]
    return allw


def high_variance_mask(s: Series, x_scaled: np.ndarray, roll: int = 50, pct: float = 75) -> np.ndarray:
    """Points whose causal rolling std exceeds the pct-th percentile of the
    rolling std inside the training partition ("changing normal variability")."""
    rs = pd.Series(x_scaled).rolling(roll, min_periods=roll).std().to_numpy()
    ref = rs[: s.a]
    ref = ref[np.isfinite(ref)]
    cut = np.percentile(ref, pct) if len(ref) else np.inf
    return np.nan_to_num(rs, nan=-np.inf) > cut
