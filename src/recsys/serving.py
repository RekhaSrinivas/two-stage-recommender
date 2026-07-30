"""Serving: the whole two-stage pipeline behind one `.recommend(user_id, n)` call.

At serve time the retriever is just its precomputed vectors (NumPy) and the ranker is a
LightGBM model — no torch/faiss needed, so the serving image stays small. On startup we load
the saved vectors + graph embeddings, fit the LambdaMART ranker once, and cache everything.
Each request runs retrieval (two-tower top-N) → ranking (LambdaMART) → titles.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .data import load_dataset
from .ranker import (FEATURE_NAMES, RankContext, build_training_data, compute_features,
                     generate_candidates, train_ranker)


class Recommender:
    def __init__(self, dataset: str | None = None, root: str | Path | None = None):
        root = Path(root) if root else Path(__file__).resolve().parents[2]
        cfg = yaml.safe_load(open(root / "configs" / "default.yaml"))
        d = cfg["data"]
        name = dataset or d["dataset"]
        self.ds = load_dataset(
            dataset=name, data_dir=str(root / d["data_dir"]),
            min_rating_positive=d["min_rating_positive"], min_user_interactions=d["min_user_interactions"],
            test_holdout=d["test_holdout"], val_holdout=d["val_holdout"], seed=d["seed"],
        )
        vec = root / "results" / f"vectors_{self.ds.name}"
        emb = root / "results" / f"graph_emb_{self.ds.name}"
        if not (vec / "item_vecs.npy").exists() or not (emb / "item_graph_emb.npy").exists():
            raise FileNotFoundError(
                f"Missing artifacts for {self.ds.name}. Run run_phase2.py and "
                f"run_phase3.py --only k3 (or the Airflow DAG) first."
            )
        self.ctx = RankContext.build(
            np.load(vec / "user_vecs.npy"), np.load(vec / "item_vecs.npy"),
            np.load(emb / "user_graph_emb.npy"), np.load(emb / "item_graph_emb.npy"),
            self.ds.train_matrix, self.ds.item_genres,
        )
        self.cols = list(range(len(FEATURE_NAMES)))
        self.n_candidates = cfg["ranker"]["n_candidates"]

        # Fit the ranker once at startup (fast; on ml-100k a few seconds).
        val_users = [u for u in self.ds.val_items_by_user if self.ds.val_items_by_user[u]]
        cand = generate_candidates(self.ctx.tt_user, self.ctx.tt_item, val_users,
                                   self.ds.train_items_by_user, self.n_candidates)
        X, y, groups = build_training_data(val_users, cand, self.ds.val_items_by_user, self.ctx)
        self.model = train_ranker(X, y, groups, self.cols, cfg["ranker"], cfg["ranker"]["seed"])

        self._inv_item = {v: k for k, v in self.ds.item_id_map.items()}

    @property
    def dataset_name(self) -> str:
        return self.ds.name

    @property
    def valid_user_ids(self) -> list[int]:
        return sorted(self.ds.user_id_map)

    def _title(self, idx: int) -> str:
        return self.ds.item_titles.get(idx, f"item {self._inv_item[idx]}")

    def recommend(self, user_id: int, n: int = 10) -> list[dict]:
        """Top-n recommendations (retrieval -> ranking) for an original MovieLens user id."""
        if user_id not in self.ds.user_id_map:
            raise KeyError(user_id)
        u = self.ds.user_id_map[user_id]
        cand = generate_candidates(self.ctx.tt_user, self.ctx.tt_item, [u],
                                   self.ds.train_items_by_user, self.n_candidates)[u]
        feats = compute_features(np.full(len(cand), u), cand, self.ctx)
        scores = self.model.predict(feats[:, self.cols])
        order = np.argsort(-scores)[:n]
        return [{"rank": i + 1, "item_id": int(self._inv_item[cand[j]]),
                 "title": self._title(int(cand[j])), "score": float(scores[j])}
                for i, j in enumerate(order)]

    def history(self, user_id: int, k: int = 10) -> list[dict]:
        """A few of the user's training interactions, for UI context."""
        if user_id not in self.ds.user_id_map:
            raise KeyError(user_id)
        u = self.ds.user_id_map[user_id]
        items = sorted(self.ds.train_items_by_user.get(u, set()),
                       key=lambda it: -self.ds.item_popularity[it])[:k]
        return [{"item_id": int(self._inv_item[it]), "title": self._title(it)} for it in items]
