"""Unit tests: run with  python -m pytest -q  (or python tests/test_core.py)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from atd.metrics import evaluate, alert_onsets
from atd.nab_score import nab_raw
from atd.thresholds import rolling_plain_stats, rolling_gated, fixed_percentile, fixed_mean_ksigma


def test_fixed_thresholds():
    e = np.arange(101, dtype=float)
    assert fixed_percentile(e, 90) == 90.0
    assert np.isclose(fixed_mean_ksigma(e, 2), e.mean() + 2 * e.std())


def test_rolling_is_causal():
    seed = np.ones(10)
    stream = np.array([1, 1, 100, 1, 1], float)
    loc, sc = rolling_plain_stats(seed, stream, 5, "median_mad")
    assert loc[2] == 1.0          # spike not in its own history
    a, th = rolling_gated(seed, stream, 5, 3, "median_mad")
    assert a[2] == 1 and a[3] == 0


def test_gated_memory_resists_masking():
    rng = np.random.default_rng(0)
    seed = rng.normal(1, 0.1, 200)
    s = rng.normal(1, 0.1, 400); s[100:200] += 5   # long anomaly
    loc, sc = rolling_plain_stats(seed, s, 50, "median_mad")
    plain = (s > loc + 6 * sc)[100:200].sum()
    wins = rolling_gated(seed, s, 50, 6, "median_mad")[0][100:200].sum()
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


def test_normal_only_training_windows():
    from atd.data import training_windows
    x = np.arange(200, dtype=float)
    lab = np.zeros(200, int); lab[50:60] = 1
    w = training_windows(x, 10, 150, 10_000, lab)
    # no training window may contain an index in 50..59, and none may end at/after a=150
    assert not any(((row >= 50) & (row <= 59)).any() for row in w)
    assert w[:, -1].max() == 149
    assert len(training_windows(x, 10, 150, 10_000, None)) == 141


def test_exact_three_sigma_rules():
    from atd.thresholds import apply_method
    cfg = {"B4_fixed_mean_3sigma": {"k": [3]}, "B3_rolling_mean_3sigma": {"k": [3], "w": [5], "memory": "plain"}}
    tr = np.array([1., 2, 3, 4, 5])
    _, th = apply_method(cfg, "B4_fixed_mean_3sigma", tr, np.zeros(3), {"k": 3})
    assert np.allclose(th, tr.mean() + 3 * tr.std())
    _, th = apply_method(cfg, "B3_rolling_mean_3sigma", tr, np.array([10., 0]), {"k": 3, "w": 5})
    assert np.isclose(th[0], tr.mean() + 3 * tr.std())          # uses the 5 previous values only


def test_lofo_never_uses_own_file():
    import pandas as pd
    from atd.pipeline import lofo_select
    rows = []
    # file "a" is best with params X, every other file is best with params Y
    for f in "abcd":
        for prm, good in (('{"x": 1}', f == "a"), ('{"x": 2}', f != "a")):
            rows.append({"file": f, "seed": 1, "method": "M", "params": prm, "composite_f1": 0.9 if good else 0.1,
                         "nab_raw": 0, "nab_null": -1, "nab_perfect": 1, "alarm_rate": 0.01})
    chosen, _ = lofo_select(pd.DataFrame(rows))
    assert chosen.set_index("file").loc["a", "params"] == '{"x": 2}'   # a's own optimum is not visible to a


def test_nab_matches_hand_computation():
    # one window [10, 19]; alarm at its left edge scores +1; one far false alarm costs -0.11
    a = np.zeros(100, int); a[10] = 1; a[80] = 1
    r = nab_raw(a, [(10, 19)])
    assert np.isclose(r["raw"], 1.0 + 0.11 * (2 / (1 + np.exp(5 * (61 / 9))) - 1))


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
