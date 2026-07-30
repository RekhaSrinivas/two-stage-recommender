"""ANN sanity checks: the exact FAISS index must agree with brute force, and the recall
metric must behave. IVF (approximate) is allowed to miss some, so we only assert it's in
the valid [0, 1] range and improves with more probes on average.
"""
import numpy as np
import pytest

faiss = pytest.importorskip("faiss")  # skip cleanly if faiss isn't installed

from recsys.ann import ANNIndex, exact_topk, recall_at_k


def _vecs(n, d, seed):
    return np.random.default_rng(seed).standard_normal((n, d)).astype("float32")


def test_exact_topk_matches_manual():
    items = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.9]], dtype="float32")
    users = np.array([[1.0, 0.0]], dtype="float32")  # closest is item 0, then item 2
    top = exact_topk(users, items, k=2)
    assert top[0].tolist() == [0, 2]


def test_flat_index_is_exact():
    items, users = _vecs(500, 32, 1), _vecs(200, 32, 2)
    exact = exact_topk(users, items, k=10)
    flat = ANNIndex(items, kind="flat").search(users, k=10)
    assert recall_at_k(flat, exact) == pytest.approx(1.0)


def test_recall_metric_bounds_and_ivf():
    items, users = _vecs(2000, 32, 3), _vecs(300, 32, 4)
    exact = exact_topk(users, items, k=10)
    assert recall_at_k(exact, exact) == pytest.approx(1.0)          # identical -> perfect
    low = recall_at_k(ANNIndex(items, kind="ivf", nprobe=1).search(users, 10), exact)
    high = recall_at_k(ANNIndex(items, kind="ivf", nprobe=16).search(users, 10), exact)
    assert 0.0 <= low <= 1.0 and 0.0 <= high <= 1.0
    assert high >= low                                              # more probes -> >= recall
