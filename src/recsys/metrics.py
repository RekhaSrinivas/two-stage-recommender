"""Ranking evaluation for top-N recommendation.

All metrics use **binary relevance**: an item is relevant to a user iff it is in that
user's held-out set. Each function takes a `ranked` list (recommendations, best first)
and a `relevant` set, so they are trivial to unit-test against hand-computed values.

Why these metrics (interview-ready):
- Recall@K / HitRate@K: did we surface the thing the user actually wanted?
- Precision@K: how much of the shortlist was useful.
- NDCG@K: rank-sensitive — a hit at position 1 is worth more than at position 10.
- MAP@K (mean of AP@K): rewards putting *all* relevant items high, not just one.
- MRR: focuses purely on the rank of the FIRST hit.
Accuracy/RMSE are deliberately NOT used: we never predict a rating, we produce a ranked
shortlist, and top-N quality is what a user experiences.

Beyond-accuracy metrics (a recommender that only chases NDCG collapses to popularity):
- Coverage: fraction of the catalogue that ever gets recommended.
- Novelty: mean self-information -log2(p(item)); high = recommends non-obvious items.
- Intra-list diversity: 1 - mean pairwise genre-cosine within a user's list.
"""
from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp


def recall_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for it in ranked[:k] if it in relevant)
    return hits / len(relevant)


def precision_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for it in ranked[:k] if it in relevant)
    return hits / k


def hit_rate_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    return 1.0 if any(it in relevant for it in ranked[:k]) else 0.0


def ndcg_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, it in enumerate(ranked[:k]) if it in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits, score = 0, 0.0
    for i, it in enumerate(ranked[:k]):
        if it in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(relevant), k)


def reciprocal_rank(ranked: list[int], relevant: set[int]) -> float:
    for i, it in enumerate(ranked):
        if it in relevant:
            return 1.0 / (i + 1)
    return 0.0


def _intra_list_diversity(ranked: list[int], item_genres: sp.csr_matrix, k: int) -> float | None:
    """1 - average pairwise cosine similarity of item genre vectors within the list."""
    items = [it for it in ranked[:k]]
    if len(items) < 2:
        return None
    vecs = item_genres[items].toarray().astype(float)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = vecs / norms
    sim = unit @ unit.T
    n = len(items)
    off_diag = (sim.sum() - np.trace(sim)) / (n * (n - 1))
    return 1.0 - off_diag


def evaluate(
    recommendations: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    ks: list[int],
    n_items: int,
    item_popularity: np.ndarray,
    item_genres: sp.csr_matrix | None = None,
) -> dict[str, float]:
    """Aggregate all metrics over users that have a non-empty ground-truth set.

    `recommendations[u]` must be at least max(ks) long, best-first, already excluding
    items the user saw in training.
    """
    ks = sorted(ks)
    max_k = max(ks)
    users = [u for u in ground_truth if ground_truth[u]]

    accum = {f"{m}@{k}": [] for k in ks for m in ("recall", "precision", "ndcg", "map", "hit")}
    accum["mrr"] = []
    accum.update({f"diversity@{k}": [] for k in ks} if item_genres is not None else {})

    total_interactions = float(item_popularity.sum())
    pop_prob = np.where(item_popularity > 0, item_popularity / max(total_interactions, 1.0), 1e-12)
    novelty = {k: [] for k in ks}
    recommended_items = {k: set() for k in ks}

    for u in users:
        rel = ground_truth[u]
        ranked = recommendations.get(u, [])
        for k in ks:
            accum[f"recall@{k}"].append(recall_at_k(ranked, rel, k))
            accum[f"precision@{k}"].append(precision_at_k(ranked, rel, k))
            accum[f"ndcg@{k}"].append(ndcg_at_k(ranked, rel, k))
            accum[f"map@{k}"].append(average_precision_at_k(ranked, rel, k))
            accum[f"hit@{k}"].append(hit_rate_at_k(ranked, rel, k))
            topk = ranked[:k]
            recommended_items[k].update(topk)
            if topk:
                novelty[k].append(float(np.mean([-math.log2(pop_prob[it]) for it in topk])))
            if item_genres is not None:
                d = _intra_list_diversity(ranked, item_genres, k)
                if d is not None:
                    accum[f"diversity@{k}"].append(d)
        accum["mrr"].append(reciprocal_rank(ranked, rel))

    out = {name: float(np.mean(vals)) if vals else 0.0 for name, vals in accum.items()}
    for k in ks:
        out[f"coverage@{k}"] = len(recommended_items[k]) / n_items
        out[f"novelty@{k}"] = float(np.mean(novelty[k])) if novelty[k] else 0.0
    out["n_eval_users"] = float(len(users))
    return out
