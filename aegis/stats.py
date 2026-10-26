"""Statistics used in the paper: bootstrap intervals, paired tests, multiplicity
correction, effect sizes, calibration error and selective-prediction curves."""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats as st


def boot_ci(x: Sequence[float], n_boot: int = 10000, seed: int = 0, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Mean and percentile bootstrap interval over records."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def boot_ratio_ci(num: Sequence[float], den: Sequence[float], n_boot: int = 5000, seed: int = 0):
    """Ratio of sums (e.g. pooled PPV) with a record-level bootstrap."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    r = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
    return float(num.sum() / max(den.sum(), 1e-12)), float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))


def paired_permutation(a: Sequence[float], b: Sequence[float], n_perm: int = 10000, seed: int = 0) -> float:
    """Two-sided sign-flip permutation test on paired differences."""
    d = np.asarray(a, float) - np.asarray(b, float)
    obs = abs(d.mean())
    if obs == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    return float((1 + np.sum(null >= obs - 1e-15)) / (n_perm + 1))


def mcnemar(a: Sequence[bool], b: Sequence[bool]) -> Tuple[float, int, int]:
    """Exact McNemar test for paired binary outcomes. Returns (p, n10, n01)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b))
    n = n10 + n01
    if n == 0:
        return 1.0, n10, n01
    p = 2 * st.binom.cdf(min(n10, n01), n, 0.5)
    return float(min(1.0, p)), n10, n01


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def cohens_dz(a, b) -> float:
    d = np.asarray(a, float) - np.asarray(b, float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("inf") if d.mean() != 0 else 0.0


def cliffs_delta(a, b, max_n: int = 4000, seed: int = 0) -> float:
    """Cliff's delta between two samples (subsampled for speed when large)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    if len(a) > max_n:
        a = rng.choice(a, max_n, replace=False)
    if len(b) > max_n:
        b = rng.choice(b, max_n, replace=False)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


def ece(conf, correct, n_bins: int = 15) -> Dict[str, float]:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    e, m = 0.0, 0.0
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        acc, cf = correct[mask].mean(), conf[mask].mean()
        gap = abs(acc - cf)
        e += mask.mean() * gap
        m = max(m, gap)
        rows.append((float(cf), float(acc), int(mask.sum())))
    brier = float(np.mean((conf - correct) ** 2))
    return dict(ece=float(e), mce=float(m), brier=brier, bins=rows)


def risk_coverage(conf, correct) -> Dict:
    """Risk (1 - accuracy among accepted) as a function of coverage; AURC."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    order = np.argsort(-conf, kind="mergesort")
    c = correct[order]
    n = len(c)
    cum = np.cumsum(c)
    k = np.arange(1, n + 1)
    risk = 1 - cum / k
    cov = k / n
    aurc = float(np.mean(risk))
    opt = np.sort(correct)[::-1]
    opt_risk = 1 - np.cumsum(opt) / k
    return dict(coverage=cov, risk=risk, aurc=aurc, eaurc=aurc - float(np.mean(opt_risk)))


def at_coverage(conf, correct, target: float) -> Dict[str, float]:
    """Accept the top ``target`` share by confidence; return selective accuracy etc."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    n = len(conf)
    k = max(1, int(round(target * n)))
    order = np.argsort(-conf, kind="mergesort")
    acc_mask = np.zeros(n, bool)
    acc_mask[order[:k]] = True
    sel = correct[acc_mask].mean()
    return dict(coverage=k / n, selective_acc=float(sel), silent_error=float(((1 - correct) * acc_mask).sum() / n),
                yield_=float((correct * acc_mask).sum() / n))


def mcc(tp, fp, tn, fn) -> float:
    den = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1e-12))
    return float((tp * tn - fp * fn) / den)
