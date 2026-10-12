import numpy as np
import pytest

from fast_vus import vus_pr, vus_pr_auto


def labels(n, spans):
    y = np.zeros(n, int)
    for a, b in spans:
        y[a:b] = 1
    return y


def random_case(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(200, 3000))
    y = np.zeros(n, int)
    for _ in range(rng.integers(1, 15)):
        a = rng.integers(0, n - 1)
        y[a:a + rng.integers(1, 60)] = 1
    score = rng.normal(size=n) + 2 * y * rng.random()
    return y, score, int(rng.integers(0, 120))


n, rng = 1000, np.random.default_rng(1)
noise = rng.random(n)
CASES = {
    "single_point": (labels(n, [(500, 501)]), noise, 50),
    "window_0": (labels(n, [(300, 320)]), noise, 0),
    "window_1": (labels(n, [(300, 320)]), noise, 1),
    "odd_window": (labels(n, [(300, 320), (700, 740)]), noise, 37),
    "edges": (labels(n, [(0, 10), (990, 1000)]), noise, 60),
    "overlapping_buffers": (labels(n, [(400, 410), (430, 440), (460, 470)]), noise, 80),
    "window_larger_than_series": (labels(100, [(40, 50)]), noise[:100], 150),
    "ties": (labels(n, [(200, 260)]), np.round(noise * 5) / 5, 40),
    "constant_score": (labels(n, [(200, 260)]), np.full(n, 0.5), 40),
    "perfect_score": (labels(n, [(200, 260)]), labels(n, [(200, 260)]).astype(float), 40),
    **{f"random_{s}": random_case(s) for s in range(20)},
}


@pytest.mark.parametrize("name", CASES)
def test_equals_tsb_ad(name):
    generate_curve = pytest.importorskip("TSB_AD.evaluation.basic_metrics").generate_curve
    y, score, w = CASES[name]
    ref = float(generate_curve(y, score, w, "opt", 250)[-1])
    assert vus_pr(y, score, w) == pytest.approx(ref, abs=1e-12)


def test_auto_window_equals_tsb_ad():
    pytest.importorskip("TSB_AD")
    from TSB_AD.evaluation.basic_metrics import generate_curve
    from TSB_AD.utils.slidingWindows import find_length_rank

    t = np.arange(4000)
    series = np.sin(2 * np.pi * t / 48) + 0.1 * np.random.default_rng(0).normal(size=len(t))
    y = labels(len(t), [(1000, 1030), (3000, 3010)])
    score = np.abs(series - np.sin(2 * np.pi * t / 48)) + y
    w = find_length_rank(series, rank=1)
    assert vus_pr_auto(y, score, series) == pytest.approx(float(generate_curve(y, score, w, "opt", 250)[-1]), abs=1e-12)
    assert vus_pr_auto(y, score, series[:, None]) == vus_pr_auto(y, score, series)


def test_all_one_label_is_one():
    # TSB-AD raises IndexError here; precision is 1 at every threshold, so VUS-PR is 1
    assert vus_pr(np.ones(500, int), noise[:500], 20) == 1.0


def test_no_anomaly_raises():
    with pytest.raises(ValueError, match="no anomaly"):
        vus_pr(np.zeros(500, int), noise[:500], 20)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_score_raises(bad):
    score = noise.copy()
    score[10] = bad
    with pytest.raises(ValueError, match="NaN or inf"):
        vus_pr(labels(n, [(300, 320)]), score, 20)


@pytest.mark.parametrize("label, score, window, msg", [
    (np.zeros(5), np.zeros(6), 2, "length"),
    (np.zeros((5, 2)), np.zeros((5, 2)), 2, "1-D"),
    ([], [], 2, "empty"),
    ([0, 2, 1], [0.1, 0.2, 0.3], 2, "only 0 and 1"),
    ([0, 1, 0], [0.1, 0.2, 0.3], -1, ">= 0"),
])
def test_bad_input_raises(label, score, window, msg):
    with pytest.raises(ValueError, match=msg):
        vus_pr(label, score, window)


def test_float_window_raises():
    with pytest.raises(TypeError):
        vus_pr([0, 1, 0], [0.1, 0.2, 0.3], 2.5)


def test_label_dtypes_agree():
    y = labels(n, [(300, 320)])
    ref = vus_pr(y, noise, 30)
    assert vus_pr(y.astype(bool), noise, 30) == ref
    assert vus_pr(y.astype(float), noise, 30) == ref
    assert vus_pr(y, noise, np.int64(30)) == ref
