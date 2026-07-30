"""Stage 2: ranking. Re-score the retriever's candidates with richer features.

The two-stage pattern: retrieval (Phase 2 two-tower) narrows 3,525 items to a top-N
shortlist cheaply; the ranker then spends real modelling effort ordering just that
shortlist, using features a dot-product retriever can't: cross-signals, popularity, genre
match, and — the point of the graph work — the LightGCN score.

Features per (user, candidate item). Every feature must VARY across a user's candidates —
LambdaMART ranks *within* a user, so a feature that's constant per user (e.g. raw user
activity) carries zero ranking signal and only invites overfitting. So all features here are
item-level or user×item:
- tt_score      : two-tower dot product (the retriever's own opinion)
- gcn_score     : LightGCN dot product  (**the graph feature — toggled in the ablation**)
- item_pop_log  : log popularity (rankers must learn to *discount*, not just chase, this)
- genre_overlap : match between the user's genre profile and the item's genres (user×item)
- item_ngenres  : breadth of the item

The ranker is LightGBM's LambdaMART (`objective="lambdarank"`), the industry-standard
learning-to-rank model — it optimizes a ranking metric (NDCG) directly, per user "query".
"""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import scipy.sparse as sp

FEATURE_NAMES = ["tt_score", "gcn_score", "item_pop_log", "genre_overlap", "item_ngenres"]
GRAPH_FEATURES = ["gcn_score"]  # dropped in the "no graph" ablation


@dataclass
class RankContext:
    """Precomputed arrays the feature builder reads from (all in reindexed id space)."""

    tt_user: np.ndarray            # (n_users x d) two-tower user vectors
    tt_item: np.ndarray            # (n_items x d)
    gcn_user: np.ndarray           # (n_users x d) LightGCN user embeddings
    gcn_item: np.ndarray           # (n_items x d)
    item_pop: np.ndarray           # (n_items,)
    user_activity: np.ndarray      # (n_users,)
    item_genre: np.ndarray         # (n_items x g) dense binary
    user_pref_genre: np.ndarray    # (n_users x g) user's normalized genre profile

    @classmethod
    def build(cls, tt_user, tt_item, gcn_user, gcn_item, train_matrix, item_genres):
        item_pop = np.asarray(train_matrix.sum(axis=0)).ravel()
        user_activity = np.asarray(train_matrix.sum(axis=1)).ravel()
        g = item_genres.toarray().astype(np.float32) if item_genres is not None else \
            np.zeros((tt_item.shape[0], 1), dtype=np.float32)
        # User genre profile = average genre vector of the items they interacted with.
        pref = (train_matrix @ g)
        denom = np.clip(user_activity, 1.0, None)[:, None]
        user_pref_genre = np.asarray(pref) / denom
        return cls(tt_user, tt_item, gcn_user, gcn_item, item_pop, user_activity, g, user_pref_genre)


def compute_features(users: np.ndarray, items: np.ndarray, ctx: RankContext) -> np.ndarray:
    """Vectorized feature matrix for aligned (user, item) arrays -> (len, n_features)."""
    tt = np.sum(ctx.tt_user[users] * ctx.tt_item[items], axis=1)
    gcn = np.sum(ctx.gcn_user[users] * ctx.gcn_item[items], axis=1)
    pop = np.log1p(ctx.item_pop[items])
    genre_ov = np.sum(ctx.user_pref_genre[users] * ctx.item_genre[items], axis=1)
    ngen = ctx.item_genre[items].sum(axis=1)
    return np.column_stack([tt, gcn, pop, genre_ov, ngen]).astype(np.float32)


def generate_candidates(
    tt_user: np.ndarray, tt_item: np.ndarray, users: list[int],
    exclude: dict[int, set], n: int, batch: int = 512,
) -> dict[int, np.ndarray]:
    """Two-tower top-N candidate item ids per user, excluding `exclude[u]` items."""
    out: dict[int, np.ndarray] = {}
    users = list(users)
    for start in range(0, len(users), batch):
        chunk = users[start : start + batch]
        scores = tt_user[chunk] @ tt_item.T                 # (b x n_items)
        for r, u in enumerate(chunk):
            ex = exclude.get(int(u))
            if ex:
                scores[r, list(ex)] = -np.inf
            top = np.argpartition(-scores[r], n - 1)[:n]
            out[int(u)] = top[np.argsort(-scores[r][top])]
    return out


def union_seen(*dicts) -> dict[int, set]:
    """Merge several {user: set-of-items} dicts (e.g. train ∪ val seen)."""
    out: dict[int, set] = {}
    for d in dicts:
        for u, s in d.items():
            out.setdefault(u, set()).update(s)
    return out


def build_training_data(users, cand_by_user, pos_by_user, ctx):
    """(X, y, group_sizes) for LambdaMART. Only users whose positive was genuinely retrieved.

    Injecting positives the retriever missed teaches the ranker that low retriever-scores can
    be positive, poisoning its strongest feature. So we train on the same distribution we
    serve: re-ranking what retrieval actually surfaced.
    """
    X_parts, y_parts, groups = [], [], []
    for u in users:
        pos = pos_by_user[u]
        cand = cand_by_user[u]
        if not (pos & set(cand.tolist())):
            continue
        us = np.full(len(cand), u)
        X_parts.append(compute_features(us, cand, ctx))
        y_parts.append(np.array([1 if it in pos else 0 for it in cand]))
        groups.append(len(cand))
    return np.vstack(X_parts), np.concatenate(y_parts), np.array(groups)


def train_ranker(X, y, groups, cols, cfg, seed):
    """LambdaMART with an early-stopping split by user (never split within a query group)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(groups))
    n_valid = max(1, int(cfg["valid_frac"] * len(groups)))
    valid_g = set(perm[:n_valid].tolist())

    bounds = np.concatenate([[0], np.cumsum(groups)])
    tr_idx, va_idx, tr_g, va_g = [], [], [], []
    for gi in range(len(groups)):
        idx = range(bounds[gi], bounds[gi + 1])
        (va_idx if gi in valid_g else tr_idx).extend(idx)
        (va_g if gi in valid_g else tr_g).append(groups[gi])

    Xc = X[:, cols]
    model = lgb.LGBMRanker(
        objective="lambdarank", n_estimators=cfg["n_estimators"],
        learning_rate=cfg["learning_rate"], num_leaves=cfg["num_leaves"],
        min_child_samples=cfg["min_child_samples"], reg_lambda=cfg["reg_lambda"],
        subsample=cfg["subsample"], subsample_freq=1, colsample_bytree=cfg["colsample_bytree"],
        random_state=seed, n_jobs=-1, verbose=-1,
    )
    model.fit(
        Xc[tr_idx], y[tr_idx], group=tr_g,
        eval_set=[(Xc[va_idx], y[va_idx])], eval_group=[va_g], eval_at=[20],
        callbacks=[lgb.early_stopping(cfg["early_stopping_rounds"], verbose=False)],
    )
    return model


def rerank(cand_by_user, ctx, score_fn, max_k):
    """Order each user's candidates by score_fn(features, items) and keep the top max_k."""
    recs = {}
    for u, cand in cand_by_user.items():
        us = np.full(len(cand), u)
        feats = compute_features(us, cand, ctx)
        order = np.argsort(-score_fn(feats, cand))
        recs[u] = cand[order][:max_k].tolist()
    return recs
