"""Per-agent calibration and joint-confidence aggregation.

Each agent's raw evidence (its report features) is mapped to a calibrated
probability that the agent's own axis is correct, using logistic (Platt-type)
calibration fitted on the validation split:

    S: the emitted bundle is FHIR-valid
    R: no semantic critical error (hallucinated or missed condition/drug/allergy,
       medication status error, high-alert dose error, unknown code)
    T: no fabricated or wrong timestamp
    P: no missed gold pharmacovigilance signal

The joint confidence estimates P(bundle acceptable). Six aggregators are
compared. The default, a weighted noisy-AND, is

    sigma_joint = prod_a [1 - w_a (1 - sigma_a)],   w_a in [0, 1],

which is bounded above by every factor and therefore by the least confident
agent with w_a = 1 (weakest-link behaviour). The noisy-OR used in the original
submission, 1 - prod_a (1 - w_a sigma_a), is bounded *below* by the most
confident agent and cannot express that a single failing axis invalidates the
bundle.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

AGENTS = ["S", "R", "T", "P"]
AXIS = {"S": "schema_ok", "R": "semantic_ok", "T": "temporal_ok", "P": "pv_ok"}
EPS = 1e-4


class ConstantModel:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        X = np.asarray(X)
        return np.column_stack([np.full(len(X), 1 - self.p), np.full(len(X), self.p)])


def fit_platt(X, y, C: float = 1.0):
    X, y = np.asarray(X, float), np.asarray(y, int)
    if len(set(y.tolist())) < 2:
        return ConstantModel((y.sum() + 1) / (len(y) + 2))
    mu, sd = X.mean(0), X.std(0) + 1e-9
    m = LogisticRegression(C=C, max_iter=2000)
    m.fit((X - mu) / sd, y)
    m._mu, m._sd = mu, sd
    return m


def predict_platt(m, X) -> np.ndarray:
    X = np.asarray(X, float)
    if isinstance(m, ConstantModel):
        return m.predict_proba(X)[:, 1]
    return m.predict_proba((X - m._mu) / m._sd)[:, 1]


def fit_agents(feats: Dict[str, List[List[float]]], labels: Dict[str, List[bool]]):
    return {a: fit_platt(feats[a], labels[AXIS[a]]) for a in AGENTS if a in feats}


def predict_agents(models, feats) -> Dict[str, np.ndarray]:
    return {a: np.clip(predict_platt(models[a], feats[a]), EPS, 1 - EPS) for a in models}


def raw_agents(feats) -> Dict[str, np.ndarray]:
    """Uncalibrated per-agent scores (the first feature is each agent's raw confidence)."""
    return {a: np.clip(np.asarray([v[0] for v in feats[a]], float), EPS, 1 - EPS) for a in feats}


def _stack(sig):
    return np.column_stack([sig[a] for a in AGENTS if a in sig])


def noisy_or(sig, w):
    S = _stack(sig)
    return 1 - np.prod(1 - S * w, axis=1)


def noisy_and(sig, w):
    S = _stack(sig)
    return np.prod(1 - w * (1 - S), axis=1)


def product(sig, w=None):
    return np.prod(_stack(sig), axis=1)


def minimum(sig, w=None):
    return np.min(_stack(sig), axis=1)


def mean(sig, w=None):
    return np.mean(_stack(sig), axis=1)


def _fit_w(fn, sig, y):
    k = _stack(sig).shape[1]
    y = np.asarray(y, float)
    obj = lambda w: float(np.mean((fn(sig, np.asarray(w)) - y) ** 2))
    r = minimize(obj, x0=np.full(k, 0.8), bounds=[(0.0, 1.0)] * k, method="L-BFGS-B")
    return r.x


class Aggregator:
    def __init__(self, kind: str):
        self.kind = kind
        self.w = None
        self.model = None

    def fit(self, sig, y):
        if self.kind in ("noisy_or", "noisy_and"):
            self.w = _fit_w(noisy_or if self.kind == "noisy_or" else noisy_and, sig, y)
        elif self.kind == "logistic":
            Z = np.log(_stack(sig) / (1 - _stack(sig)))
            self.model = fit_platt(Z, y, C=1.0)
        return self

    def __call__(self, sig):
        if self.kind == "noisy_or":
            return noisy_or(sig, self.w)
        if self.kind == "noisy_and":
            return noisy_and(sig, self.w)
        if self.kind == "product":
            return product(sig)
        if self.kind == "min":
            return minimum(sig)
        if self.kind == "mean":
            return mean(sig)
        if self.kind == "logistic":
            Z = np.log(_stack(sig) / (1 - _stack(sig)))
            return predict_platt(self.model, Z)
        raise ValueError(self.kind)


AGGREGATORS = ["noisy_and", "noisy_or", "product", "min", "mean", "logistic"]


def decide(sigma, valid, tau_auto: float = 0.70, tau_review: float = 0.50, psi_alarm=None):
    """AUTO (emitted without review), REVIEW (emitted with warnings to a human),
    ABSTAIN (no bundle). Invalid bundles are always ABSTAIN (schema gate)."""
    sigma, valid = np.asarray(sigma), np.asarray(valid, bool)
    out = np.where(~valid, "ABSTAIN", np.where(sigma >= tau_auto, "AUTO", np.where(sigma >= tau_review, "REVIEW", "ABSTAIN")))
    if psi_alarm is not None:
        out = np.where((out == "AUTO") & np.asarray(psi_alarm, bool), "REVIEW", out)
    return out
