"""Baselines you must beat before any deep model earns its complexity.

- MostPopular: non-personalised. If your fancy model can't beat this, it's broken.
- ItemItemCF: classic neighbourhood collaborative filtering (cosine item-item).
- ALS (implicit MF): Hu-Koren-Volinsky implemented from scratch in NumPy so the
  update rule is fully transparent and defensible.

Every model implements `.fit(dataset)` and `.recommend(user_indices, k) -> {u: [items]}`,
where recommendations already exclude items the user interacted with in training.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from .data import RecDataset

NEG_INF = -np.inf


def _topk_from_scores(
    scores: np.ndarray, user_indices: np.ndarray, seen: dict[int, set], k: int
) -> dict[int, list[int]]:
    """Mask training items, then return the top-k item indices per row (sorted)."""
    results: dict[int, list[int]] = {}
    for row, u in enumerate(user_indices):
        s = scores[row].copy()
        seen_u = seen.get(int(u))
        if seen_u:
            s[list(seen_u)] = NEG_INF
        # argpartition for the top-k, then sort just those k by score (descending).
        kk = min(k, s.shape[0])
        cand = np.argpartition(-s, kk - 1)[:kk]
        cand = cand[np.argsort(-s[cand])]
        results[int(u)] = cand.tolist()
    return results


class MostPopular:
    """Recommend the globally most-popular items the user hasn't seen."""

    name = "MostPopular"

    def fit(self, dataset: RecDataset) -> "MostPopular":
        self.order = np.argsort(-dataset.item_popularity)  # item idx, most popular first
        self.train_items_by_user = dataset.train_items_by_user
        return self

    def recommend(self, user_indices, k: int) -> dict[int, list[int]]:
        results = {}
        for u in user_indices:
            seen = self.train_items_by_user.get(int(u), set())
            rec, i = [], 0
            while len(rec) < k and i < len(self.order):
                it = int(self.order[i])
                if it not in seen:
                    rec.append(it)
                i += 1
            results[int(u)] = rec
        return results


class ItemItemCF:
    """Item-item collaborative filtering with cosine similarity.

    score(u, i) = sum over items j the user interacted with of cosine(i, j).
    Optionally keep only the top-N neighbours per item to control memory/noise.
    """

    name = "ItemItemCF"

    def __init__(self, topn_neighbors: int = 200):
        self.topn_neighbors = topn_neighbors

    def fit(self, dataset: RecDataset) -> "ItemItemCF":
        X = (dataset.train_matrix > 0).astype(np.float32)      # users x items, binary
        co = (X.T @ X).toarray()                               # items x items co-occurrence
        norms = np.sqrt(np.clip(np.diag(co), 1e-8, None))
        sim = co / np.outer(norms, norms)
        np.fill_diagonal(sim, 0.0)
        if self.topn_neighbors and self.topn_neighbors < sim.shape[0]:
            # Zero out all but the strongest N neighbours per item (row-wise).
            keep = np.argpartition(-sim, self.topn_neighbors, axis=1)[:, : self.topn_neighbors]
            masked = np.zeros_like(sim)
            rows = np.arange(sim.shape[0])[:, None]
            masked[rows, keep] = sim[rows, keep]
            sim = masked
        self.sim = sim.astype(np.float32)
        self.X = X
        self.train_items_by_user = dataset.train_items_by_user
        return self

    def recommend(self, user_indices, k: int, batch: int = 1024) -> dict[int, list[int]]:
        user_indices = np.asarray(list(user_indices))
        out: dict[int, list[int]] = {}
        for start in range(0, len(user_indices), batch):
            b = user_indices[start : start + batch]
            scores = (self.X[b] @ self.sim)                    # (b x n_items)
            scores = np.asarray(scores)
            out.update(_topk_from_scores(scores, b, self.train_items_by_user, k))
        return out


class ALS:
    """Implicit-feedback matrix factorization (Hu, Koren & Volinsky 2008), from scratch.

    Confidence c_ui = 1 + alpha * (feedback). Preference p_ui in {0,1}. We alternate:
        x_u = (Y'Y + Y'(C^u - I)Y + reg*I)^-1 Y' C^u p(u)
    Because p is binary and C^u diagonal, this simplifies (see `_als_step`) to a small
    factors x factors solve per user/item, which is why ALS scales to implicit data.
    """

    name = "ALS"

    def __init__(self, factors=64, regularization=0.01, alpha=40.0, iterations=15, seed=42):
        self.factors = factors
        self.reg = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed

    @staticmethod
    def _als_step(R: sp.csr_matrix, Y: np.ndarray, reg: float, alpha: float) -> np.ndarray:
        """Solve for the factor matrix of the rows of R, holding Y fixed — fully vectorized.

        For row r the system is  A_r x_r = b_r  with
            A_r = Y'Y + reg*I + alpha * sum_{c in R_r} y_c y_c'
            b_r = (1 + alpha) * sum_{c in R_r} y_c            (preference is binary)
        The per-row Gram term is the same for every row up to *which* outer products are
        summed, so stacking outer(y_c, y_c) into a (cols, f*f) matrix turns the whole
        thing into two sparse matmuls: R @ outer_products and R @ Y. No Python loop.
        Empty rows get A_r = Y'Y + reg*I and b_r = 0, so they solve to the zero vector.
        """
        n, f = R.shape[0], Y.shape[1]
        base = (Y.T @ Y) + reg * np.eye(f, dtype=np.float64)          # (f x f), shared
        outer = (Y[:, :, None] * Y[:, None, :]).reshape(Y.shape[0], f * f)  # (cols, f*f)
        gram = np.asarray(R @ outer).reshape(n, f, f)                 # sum of outer prods
        A = base[None] + alpha * gram
        b = (1.0 + alpha) * np.asarray(R @ Y)                         # (n x f)
        return np.linalg.solve(A, b)

    def fit(self, dataset: RecDataset) -> "ALS":
        rng = np.random.default_rng(self.seed)
        R_ui = (dataset.train_matrix > 0).astype(np.float64).tocsr()
        R_iu = R_ui.T.tocsr()
        n_users, n_items = R_ui.shape
        self.U = 0.01 * rng.standard_normal((n_users, self.factors))
        self.V = 0.01 * rng.standard_normal((n_items, self.factors))
        # Pin BLAS to 1 thread: our systems are tiny (factors x factors), and OpenBLAS's
        # per-call thread-pool spin-up dwarfs the actual work (~150x slower otherwise).
        with threadpool_limits(limits=1, user_api="blas"):
            for _ in tqdm(range(self.iterations), desc="ALS", leave=False):
                self.U = self._als_step(R_ui, self.V, self.reg, self.alpha)
                self.V = self._als_step(R_iu, self.U, self.reg, self.alpha)
        self.train_items_by_user = dataset.train_items_by_user
        return self

    def recommend(self, user_indices, k: int, batch: int = 1024) -> dict[int, list[int]]:
        user_indices = np.asarray(list(user_indices))
        out: dict[int, list[int]] = {}
        for start in range(0, len(user_indices), batch):
            b = user_indices[start : start + batch]
            scores = self.U[b] @ self.V.T                      # (b x n_items)
            out.update(_topk_from_scores(scores, b, self.train_items_by_user, k))
        return out
