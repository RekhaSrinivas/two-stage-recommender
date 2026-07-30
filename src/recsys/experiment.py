"""Simulated online A/B test with a real statistical readout.

Offline metrics (Phases 1-4) tell you which model ranks better on held-out data. They do NOT
tell you whether a change is worth shipping — that needs an experiment with a randomization
unit, a business metric, and a significance test. This module simulates one.

Click model (how offline data becomes a simulated online metric):
  Each user has one known-relevant item (their held-out positive). A policy shows a ranked
  top-K list. The user *examines* higher positions more (position bias) and clicks the
  relevant item only if they examine its slot:
      P(click) = examine(rank_of_relevant_item)   if it is in the top-K, else 0
      examine(r) = 1 / log2(r + 2)                (r is 0-indexed)
  So a policy earns clicks by (a) retrieving the item at all and (b) ranking it high — exactly
  what a real recommender is rewarded for. Clicks are Bernoulli draws, giving realistic noise.

Design choices you must be able to defend:
  - Randomization unit = user (each user sees ONE policy, as in a real A/B — no within-user
    leakage).
  - Primary metric = CTR (clicked / shown), a proportion.
  - Significance = two-proportion z-test; uncertainty = bootstrap 95% CI on the lift.
  - Power = the sample size needed to detect a given effect; guards against reading noise as
    signal (or a real effect as "no difference" when simply underpowered).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def examine(rank: int, model: str = "log") -> float:
    """Probability the user examines position `rank` (0-indexed)."""
    return 1.0 / math.log2(rank + 2) if model == "log" else 1.0 / (rank + 1)


def simulate_clicks(recs, ground_truth, rng, model="log") -> dict[int, int]:
    """Bernoulli click per user: examine(rank of relevant item) if it's in the shown list."""
    clicks = {}
    for u, ranked in recs.items():
        rel = ground_truth.get(u, set())
        p = 0.0
        for i, it in enumerate(ranked):
            if it in rel:
                p = examine(i, model)
                break
        clicks[u] = 1 if rng.random() < p else 0
    return clicks


def two_proportion_ztest(x1: int, n1: int, x2: int, n2: int):
    """Pooled two-proportion z-test. Returns (p_control, p_treat, z, p_value)."""
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se if se > 0 else 0.0
    return p1, p2, z, 2 * stats.norm.sf(abs(z))


def bootstrap_diff_ci(a, b, rng, n_boot=5000, alpha=0.05):
    """Percentile bootstrap CI for mean(b) - mean(a) (treatment - control)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diffs = np.array([rng.choice(b, b.size, replace=True).mean()
                      - rng.choice(a, a.size, replace=True).mean() for _ in range(n_boot)])
    return tuple(np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def required_n_per_arm(p_control: float, mde_abs: float, alpha=0.05, power=0.8) -> int:
    """Sample size per arm to detect an absolute lift `mde_abs` at given alpha & power."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    p1, p2 = p_control, p_control + mde_abs
    pbar = (p1 + p2) / 2
    n = (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(n / mde_abs ** 2)
