"""Exact vectorised VUS-PR, equal to TSB-AD's `generate_curve(label, score, window, 'opt', 250)[-1]`."""
import operator

import numpy as np


def _segments(y):
    d = np.diff(np.r_[0, y, 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1


def _merge(s, e, h, n):
    # widen segments by h, merge overlaps
    gap = e[:-1] + h < s[1:] - h
    return np.maximum(s[np.r_[True, gap]] - h, 0), np.minimum(e[np.r_[gap, True]] + h, n - 1)


def _cover(s, e, n):
    # mark every point inside [s, e]: +1 at each start, -1 after each end, cumsum > 0
    c = np.zeros(n + 1, int)
    c[s] += 1
    c[e + 1] -= 1
    return np.cumsum(c[:n]) > 0


def _ramp(s, e, l, n):
    # soft labels sqrt(1 - d/l) on both sides of each segment
    d = np.arange(1, l // 2 + 1)
    i = np.r_[(e[:, None] + d).ravel(), (s[:, None] - d).ravel()]
    v = np.tile(np.sqrt(1 - d / max(l, 1)), 2 * len(s))
    ok = (i >= 0) & (i < n)
    return i[ok], v[ok]


def _check(label, score, window):
    y = np.asarray(label)
    x = np.asarray(score, dtype=float)
    if y.ndim != 1 or x.ndim != 1:
        raise ValueError(f"label and score must be 1-D, got shapes {y.shape} and {x.shape}")
    if len(y) != len(x):
        raise ValueError(f"label and score differ in length: {len(y)} vs {len(x)}")
    if len(y) == 0:
        raise ValueError("label and score are empty")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("label must contain only 0 and 1")
    if not y.any():
        raise ValueError("label has no anomaly; VUS-PR is undefined")
    if not np.isfinite(x).all():
        raise ValueError("score contains NaN or inf")
    window = operator.index(window)
    if window < 0:
        raise ValueError(f"window must be >= 0, got {window}")
    return y.astype(np.int8), x, window


def vus_pr(label, score, window, thre=250):
    """VUS-PR of `score` against the binary `label`, over buffer lengths 0..`window`.

    Argument order follows TSB-AD's `generate_curve(label, score, ...)`, the reverse of its
    `get_metrics(score, labels)`. `window` is usually the period of the series; see `vus_pr_auto`.
    """
    y, x, window = _check(label, score, window)
    n = len(x)
    s, e = _segments(y)
    P = y.sum()

    # thresholds and predicted positives, as in TSB-AD
    asc = np.sort(x)
    th = asc[::-1][np.linspace(0, n - 1, thre).astype(int)]
    n_pred = n - np.searchsorted(asc, th)

    # only points near an anomaly carry label weight
    region = np.flatnonzero(_cover(*_merge(s, e, window // 2, n), n))
    pos = np.full(n, -1)
    pos[region] = np.arange(len(region))
    xr, yr = x[region], y[region].astype(float)

    # 1. sort: highest score first, so the points above any threshold are a prefix
    order = np.argsort(-xr, kind="stable")

    # 3. read: prefix length per threshold; the sum of the first cnt points sits at cnt - 1
    cnt = len(xr) - np.searchsorted(np.sort(xr), th)
    at = np.maximum(cnt - 1, 0)

    # 2. cumsum: sum of point weights above every threshold at once
    def above(w):
        return np.where(cnt > 0, np.cumsum(w[order])[at], 0.0)
    
    # labelled anomaly points above each threshold; same for every buffer length
    B = above(yr)
    ap = np.empty(window + 1)
    for l in range(window + 1):
        # soft-label tp outside the anomalies
        i, v = _ramp(s, e, l, n)
        soft = np.zeros(len(region))
        np.add.at(soft, pos[i], v)
        A = above(np.minimum(soft, 1) * (yr == 0))

        # existence: merged segments whose max score clears the threshold
        ms, me = _merge(s, e, l // 2, n)
        seg_max = np.maximum.reduceat(np.where(_cover(ms, me, n), x, -np.inf), ms)

        # same counting as cnt, over segment maxima instead of points (no cumsum needed)
        hit = len(ms) - np.searchsorted(np.sort(seg_max), th)

        # range pr curve and its AP
        tp = B + A
        
        # range recall as in TSB-AD: P_new = (P + N_labels) / 2 = P + A/2
        tpr = np.minimum(tp / (P + A / 2), 1) * hit / len(ms)
        ap[l] = np.diff(tpr, prepend=0) @ (tp / n_pred)
    return float(ap.mean())


def vus_pr_auto(label, score, series, thre=250):
    """VUS-PR with the window TSB-AD uses: `find_length_rank` on the (first channel of the) series.

    Needs TSB-AD (`pip install fast-vus[tsb]`).
    """
    try:
        from TSB_AD.utils.slidingWindows import find_length_rank
    except ImportError as e:
        raise ImportError("vus_pr_auto needs TSB-AD: pip install fast-vus[tsb]") from e
    x = np.asarray(series, dtype=float)
    if x.ndim == 2:
        x = x[:, 0]
    return vus_pr(label, score, int(find_length_rank(x, rank=1)), thre)
