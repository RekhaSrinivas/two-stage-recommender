"""Approximate nearest-neighbour (ANN) candidate serving with FAISS.

This is the *serving* half of a two-tower recommender. At training time the item tower
produces one vector per item; at serving time we index those vectors once and answer
"top-K items for this user vector" with a nearest-neighbour lookup instead of scoring the
whole catalogue.

Honest scale note: MovieLens has ~3.5k items, so exact search is already sub-millisecond
and ANN is *not* needed here. ANN earns its keep at 10^5–10^8 items (real catalogues). What
this module demonstrates — and measures — is the mechanism and the accuracy/latency
trade-off you'd deploy at scale: an IVF index trades a little recall for a large speedup,
tuned by `nprobe`.

Maximum inner-product search (MIPS): our relevance is a dot product, so indices use
`METRIC_INNER_PRODUCT` rather than L2.
"""
from __future__ import annotations

import numpy as np

try:
    import faiss

    HAVE_FAISS = True
except ImportError:  # pragma: no cover - environment dependent
    HAVE_FAISS = False


class ANNIndex:
    """Wraps a FAISS index; `kind='flat'` is exact MIPS, `kind='ivf'` is approximate."""

    def __init__(self, item_vecs: np.ndarray, kind: str = "ivf", nlist: int = 128, nprobe: int = 8):
        if not HAVE_FAISS:
            raise ImportError("faiss is required for ANNIndex; pip install faiss-cpu")
        self.item_vecs = np.ascontiguousarray(item_vecs, dtype="float32")
        self.n_items, self.dim = self.item_vecs.shape
        self.kind = kind
        if kind == "flat":
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(self.item_vecs)
        elif kind == "ivf":
            # FAISS wants a healthy number of points per Voronoi cell (~>=39).
            nlist = max(1, min(nlist, self.n_items // 39))
            quantizer = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(self.item_vecs)
            self.index.add(self.item_vecs)
            self.index.nprobe = nprobe
            self.nlist = nlist
        else:
            raise ValueError(f"unknown index kind {kind!r}")

    def search(self, user_vecs: np.ndarray, k: int) -> np.ndarray:
        """Return (n_queries x k) item indices, best first."""
        q = np.ascontiguousarray(user_vecs, dtype="float32")
        _, idx = self.index.search(q, k)
        return idx


def exact_topk(user_vecs: np.ndarray, item_vecs: np.ndarray, k: int) -> np.ndarray:
    """Brute-force top-k by dot product — the ground truth ANN is measured against."""
    scores = user_vecs @ item_vecs.T
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
    return np.take_along_axis(part, order, axis=1)


def recall_at_k(approx: np.ndarray, exact: np.ndarray) -> float:
    """Fraction of each query's exact top-k that the ANN also returned, averaged."""
    hits = [len(set(a.tolist()) & set(e.tolist())) / e.shape[0] for a, e in zip(approx, exact)]
    return float(np.mean(hits))
