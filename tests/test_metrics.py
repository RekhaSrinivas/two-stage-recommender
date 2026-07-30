"""Hand-computed checks for the ranking metrics. If these pass, the harness is trustworthy.

Worked example used throughout: ranked = [A, B, C, D], relevant = {B, D}.
  hits land at positions 2 (B) and 4 (D).
"""
import math

import numpy as np
import pytest

from recsys import metrics as M

RANKED = ["A", "B", "C", "D"]
RELEVANT = {"B", "D"}


def test_recall():
    assert M.recall_at_k(RANKED, RELEVANT, 2) == pytest.approx(0.5)   # 1 of 2 relevant in top-2
    assert M.recall_at_k(RANKED, RELEVANT, 4) == pytest.approx(1.0)


def test_precision():
    assert M.precision_at_k(RANKED, RELEVANT, 2) == pytest.approx(0.5)  # 1 hit / 2
    assert M.precision_at_k(RANKED, RELEVANT, 4) == pytest.approx(0.5)  # 2 hits / 4


def test_hit_rate():
    assert M.hit_rate_at_k(RANKED, RELEVANT, 1) == 0.0   # A is not relevant
    assert M.hit_rate_at_k(RANKED, RELEVANT, 2) == 1.0   # B is


def test_ndcg():
    dcg = 1 / math.log2(2 + 1) + 1 / math.log2(4 + 1)    # B at pos2, D at pos4
    idcg = 1 / math.log2(1 + 1) + 1 / math.log2(2 + 1)   # 2 relevant, ideal top-2
    assert M.ndcg_at_k(RANKED, RELEVANT, 4) == pytest.approx(dcg / idcg)


def test_average_precision():
    # precision at B (rank2)=1/2, at D (rank4)=2/4; AP = (0.5+0.5)/min(2,4)
    assert M.average_precision_at_k(RANKED, RELEVANT, 4) == pytest.approx(0.5)


def test_reciprocal_rank():
    assert M.reciprocal_rank(RANKED, RELEVANT) == pytest.approx(1 / 2)  # first hit at rank 2


def test_perfect_and_empty():
    assert M.ndcg_at_k(["B", "D", "A", "C"], RELEVANT, 4) == pytest.approx(1.0)
    assert M.recall_at_k(["A", "C"], RELEVANT, 2) == 0.0
    assert M.reciprocal_rank(["A", "C"], RELEVANT) == 0.0


def test_evaluate_aggregates():
    recs = {0: ["A", "B", "C", "D"], 1: ["B", "D", "A", "C"]}
    truth = {0: {"B", "D"}, 1: {"B", "D"}}
    # item_popularity indexed by int; here items are strings so pass a dummy for coverage.
    # Use integer items instead for the evaluate-level test:
    recs = {0: [1, 0, 2, 3], 1: [0, 3, 1, 2]}
    truth = {0: {0, 3}, 1: {0, 3}}
    pop = np.array([10.0, 5.0, 2.0, 1.0])
    out = M.evaluate(recs, truth, ks=[2, 4], n_items=4, item_popularity=pop)
    assert 0.0 <= out["ndcg@4"] <= 1.0
    assert out["n_eval_users"] == 2.0
    assert 0.0 < out["coverage@4"] <= 1.0
