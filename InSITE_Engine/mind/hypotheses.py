"""Confidence maths for the hypothesis board.

The model never asserts a probability. It reports the *direction* of a piece of
evidence (supports / contradicts) and its *strength* (weak / moderate / strong);
the harness applies a Bayesian log-odds update with fixed likelihood ratios.
"""
from __future__ import annotations

import math

LIKELIHOOD_RATIOS = {"weak": 1.5, "moderate": 3.0, "strong": 8.0}
DIRECTIONS = ("supports", "contradicts")
STRENGTHS = tuple(LIKELIHOOD_RATIOS)

PRIOR = 0.5
SUPPORTED_AT = 0.85
REFUTED_AT = 0.10
P_MIN, P_MAX = 0.01, 0.99  # keep hypotheses revisable: never certain either way


def logit(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    e = math.exp(x)
    return e / (1 + e)


def log_odds_delta(direction: str, strength: str) -> float:
    """Signed change in log-odds for one piece of evidence."""
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}")
    if strength not in LIKELIHOOD_RATIOS:
        raise ValueError(f"strength must be one of {STRENGTHS}")
    delta = math.log(LIKELIHOOD_RATIOS[strength])
    return delta if direction == "supports" else -delta


def apply_update(probability: float, direction: str, strength: str, weight: float = 1.0) -> float:
    """Posterior after one update; `weight` scales the log-odds change (0.5 = counts half)."""
    posterior = sigmoid(logit(probability) + weight * log_odds_delta(direction, strength))
    return round(min(max(posterior, P_MIN), P_MAX), 4)


def status_for(probability: float) -> str:
    if probability >= SUPPORTED_AT:
        return "supported"
    if probability <= REFUTED_AT:
        return "refuted"
    return "open"


def uncertainty(probability: float) -> float:
    """p(1-p): how much a decisive test would teach us."""
    return probability * (1 - probability)
