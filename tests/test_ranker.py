"""Ranker feature-engineering and candidate-generation checks."""
import numpy as np
import scipy.sparse as sp

from recsys.ranker import FEATURE_NAMES, RankContext, compute_features, generate_candidates


def _ctx():
    tt_user = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)      # 2 users, d=2
    tt_item = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)  # 3 items
    gcn_user, gcn_item = tt_user.copy(), tt_item.copy()
    train = sp.csr_matrix(np.array([[1, 1, 0], [0, 0, 1]], dtype=float))  # who saw what
    genres = sp.csr_matrix(np.array([[1, 0], [0, 1], [1, 1]], dtype=float))
    return RankContext.build(tt_user, tt_item, gcn_user, gcn_item, train, genres)


def test_feature_shape_and_tt_score():
    ctx = _ctx()
    users = np.array([0, 0])
    items = np.array([1, 2])
    feats = compute_features(users, items, ctx)
    assert feats.shape == (2, len(FEATURE_NAMES))
    # tt_score for (user0, item1) = [1,0]·[0,1] = 0 ; (user0, item2) = [1,0]·[1,1] = 1
    tt_col = FEATURE_NAMES.index("tt_score")
    assert feats[0, tt_col] == 0.0
    assert feats[1, tt_col] == 1.0


def test_generate_candidates_excludes_seen_and_ranks():
    ctx = _ctx()
    # user 0 vector [1,0]; item scores: i0=1, i1=0, i2=1. Exclude i0 -> candidates from {i1,i2}.
    cand = generate_candidates(ctx.tt_user, ctx.tt_item, [0], exclude={0: {0}}, n=2)
    assert 0 not in cand[0].tolist()            # excluded seen item
    assert cand[0].tolist()[0] == 2             # highest remaining score first (i2=1 > i1=0)
    assert len(cand[0]) == 2


def test_user_genre_profile_is_averaged():
    ctx = _ctx()
    # user 0 saw items 0 (genre [1,0]) and 1 (genre [0,1]) -> avg profile [0.5, 0.5]
    assert np.allclose(ctx.user_pref_genre[0], [0.5, 0.5])
