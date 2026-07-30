"""Two-tower retrieval model in PyTorch.

The two-tower (a.k.a. dual-encoder) is the standard *candidate generation* model in
industrial recommenders (YouTube, Google Play, Pinterest). Two independent networks:

    user tower:  user_id            -> user vector  (dim d)
    item tower: [item_id, genres]   -> item vector  (dim d)

Relevance is the dot product of the two vectors. Because the towers are independent, at
serving time we precompute *every* item vector once, build an ANN index over them, and a
single forward pass of the user tower + a nearest-neighbour lookup returns candidates in
milliseconds — that separability is the whole point of the architecture.

Training objective: **in-batch sampled softmax** (Yi et al. 2019). For a batch of
(user, positive-item) pairs we score every user against every item in the batch; the other
items act as negatives. Two production details are implemented and worth defending:

- **logQ correction:** in-batch negatives are sampled proportional to popularity, which
  biases the softmax toward popular items. Subtracting log(sampling prob) from each column
  removes that bias.
- **Accidental-hit masking:** if the same item appears as a positive for two users in the
  batch, it must not be treated as a negative for the other — we mask those off-diagonal
  collisions to -inf.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def _tower(in_dim: int, hidden: list[int], out_dim: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers += [nn.Linear(prev, out_dim)]
    return nn.Sequential(*layers)


class TwoTower(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        hidden: list[int] | None = None,
        out_dim: int = 64,
        dropout: float = 0.1,
        item_genres: sp.csr_matrix | None = None,
    ):
        super().__init__()
        hidden = hidden or [128]
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

        item_feat_dim = 0
        if item_genres is not None:
            feats = torch.tensor(item_genres.toarray(), dtype=torch.float32)
            self.register_buffer("item_features", feats)  # (n_items x n_genres), non-trainable
            item_feat_dim = feats.shape[1]
        else:
            self.item_features = None

        self.user_tower = _tower(embedding_dim, hidden, out_dim, dropout)
        self.item_tower = _tower(embedding_dim + item_feat_dim, hidden, out_dim, dropout)

    def user_forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        return self.user_tower(self.user_emb(user_ids))

    def item_forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        x = self.item_emb(item_ids)
        if self.item_features is not None:
            x = torch.cat([x, self.item_features[item_ids]], dim=-1)
        return self.item_tower(x)


def in_batch_softmax_loss(
    user_vec: torch.Tensor,
    item_vec: torch.Tensor,
    item_ids: torch.Tensor,
    log_q: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Cross-entropy where the correct item for row i is the item in column i (the diagonal)."""
    logits = (user_vec @ item_vec.t()) / temperature          # (B, B)
    if log_q is not None:
        logits = logits - log_q[item_ids].unsqueeze(0)        # popularity (logQ) correction
    # Accidental-hit masking: same item id in different rows must not count as a negative.
    same_item = item_ids.unsqueeze(0) == item_ids.unsqueeze(1)  # (B, B)
    off_diagonal = ~torch.eye(len(item_ids), dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(same_item & off_diagonal, float("-inf"))
    labels = torch.arange(len(item_ids), device=logits.device)
    return F.cross_entropy(logits, labels)


def build_log_q(item_popularity: np.ndarray) -> torch.Tensor:
    """log of the empirical sampling probability of each item (for the logQ correction)."""
    total = max(float(item_popularity.sum()), 1.0)
    prob = np.clip(item_popularity / total, 1e-12, None)
    return torch.tensor(np.log(prob), dtype=torch.float32)
