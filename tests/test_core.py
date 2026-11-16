"""Unit tests: run with  python -m pytest -q  (or python tests/test_core.py)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from atd.metrics import evaluate, alert_onsets
from atd.nab_score import nab_raw
from atd.thresholds import rolling_plain_stats, rolling_winsorized, fixed_percentile, fixed_mean_ksigma


def test_fixed_thresholds():
    e = np.arange(101, dtype=float)
    assert fixed_percentile(e, 90) == 90.0
    assert np.isclose(fixed_mean_ksigma(e, 2), e.mean() + 2 * e.std())


def test_rolling_is_causal():
    seed = np.ones(10)
    stream = np.array([1, 1, 100, 1, 1], float)
    loc, sc = rolling_plain_stats(seed, stream, 5, "median_mad")
    assert loc[2] == 1.0          # spike not in its own history
    a, th = rolling_winsorized(seed, stream, 5, 3, "median_mad")
    assert a[2] == 1 and a[3] == 0


def test_winsorized_memory_resists_masking():
    rng = np.random.default_rng(0)
    seed = rng.normal(1, 0.1, 200)
    s = rng.normal(1, 0.1, 400); s[100:200] += 5   # long anomaly
    loc, sc = rolling_plain_stats(seed, s, 50, "median_mad")
    plain = (s > loc + 6 * sc)[100:200].sum()
    wins = rolling_winsorized(seed, s, 50, 6, "median_mad")[0][100:200].sum()
    assert wins > plain           # plain memory adapts to the anomaly and stops alarming


def test_nab_perfect_and_null():
    wins = [(10, 19), (50, 59)]
    a = np.zeros(100, int); a[10] = 1; a[50] = 1
    assert np.isclose(nab_raw(a, wins)["raw"], 2.0)
    assert np.isclose(nab_raw(np.zeros(100, int), wins)["raw"], -2.0)


def test_event_and_composite():
    lab = np.zeros(100, int); lab[20:30] = 1; lab[60:70] = 1
    al = np.zeros(100, int); al[25:27] = 1; al[90] = 1
    m = evaluate(al, lab, [(20, 29), (60, 69)])
    assert m["event_recall"] == 0.5 and np.isclose(m["precision"], 2 / 3)
    assert np.isclose(m["composite_f1"], 2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))
    assert alert_onsets(al).sum() == 2


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
