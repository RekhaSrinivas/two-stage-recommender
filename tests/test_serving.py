"""Serving contract: recommendations are well-formed and never include already-seen items.

Skips cleanly if the ml-100k artifacts (two-tower vectors, graph embeddings) aren't built.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_VEC = ROOT / "results" / "vectors_ml-100k" / "item_vecs.npy"
_EMB = ROOT / "results" / "graph_emb_ml-100k" / "item_graph_emb.npy"
pytestmark = pytest.mark.skipif(
    not (_VEC.exists() and _EMB.exists()), reason="ml-100k artifacts not built"
)

from recsys.serving import Recommender  # noqa: E402


@pytest.fixture(scope="module")
def rec():
    return Recommender(dataset="ml-100k", root=ROOT)


def test_recommend_shape_and_fields(rec):
    recs = rec.recommend(rec.valid_user_ids[0], n=5)
    assert len(recs) == 5
    assert all({"rank", "item_id", "title", "score"} <= set(r) for r in recs)
    assert [r["rank"] for r in recs] == [1, 2, 3, 4, 5]


def test_recommend_excludes_seen_items(rec):
    uid = rec.valid_user_ids[0]
    u = rec.ds.user_id_map[uid]
    seen = {rec._inv_item[it] for it in rec.ds.train_items_by_user.get(u, set())}
    recommended = {r["item_id"] for r in rec.recommend(uid, n=10)}
    assert not (recommended & seen)


def test_unknown_user_raises(rec):
    with pytest.raises(KeyError):
        rec.recommend(10_000_000)
