"""PyTorch dataset + retrieval helpers for the two-tower model.

Kept separate from the model so the model file stays pure architecture/loss.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class InteractionDataset(Dataset):
    """One (user, positive-item) pair per training interaction."""

    def __init__(self, train: pd.DataFrame):
        self.users = torch.tensor(train["user"].values, dtype=torch.long)
        self.items = torch.tensor(train["item"].values, dtype=torch.long)

    def __len__(self) -> int:
        return self.users.shape[0]

    def __getitem__(self, i: int):
        return self.users[i], self.items[i]


@torch.no_grad()
def build_item_matrix(model, n_items: int, device: str, batch: int = 4096) -> torch.Tensor:
    """Precompute every item vector once (the reusable half of a two-tower)."""
    model.eval()
    chunks = []
    for start in range(0, n_items, batch):
        ids = torch.arange(start, min(start + batch, n_items), device=device)
        chunks.append(model.item_forward(ids))
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def recommend(
    model, item_matrix: torch.Tensor, user_indices, train_items_by_user: dict[int, set],
    k: int, device: str, batch: int = 1024,
) -> dict[int, list[int]]:
    """Exact top-k retrieval by dot product, excluding items seen in training."""
    model.eval()
    user_indices = list(user_indices)
    out: dict[int, list[int]] = {}
    for start in range(0, len(user_indices), batch):
        uids = user_indices[start : start + batch]
        ut = torch.tensor(uids, dtype=torch.long, device=device)
        scores = model.user_forward(ut) @ item_matrix.t()      # (b, n_items)
        for r, u in enumerate(uids):
            seen = train_items_by_user.get(int(u))
            if seen:
                scores[r, list(seen)] = float("-inf")
        top = torch.topk(scores, k, dim=1).indices.cpu().numpy()
        for r, u in enumerate(uids):
            out[int(u)] = top[r].tolist()
    return out


@torch.no_grad()
def recommend_from_embeddings(
    user_emb: torch.Tensor, item_emb: torch.Tensor, user_indices,
    train_items_by_user: dict[int, set], k: int, batch: int = 1024,
) -> dict[int, list[int]]:
    """Top-k by dot product of precomputed embeddings (e.g. LightGCN), excluding seen items."""
    user_indices = list(user_indices)
    out: dict[int, list[int]] = {}
    for start in range(0, len(user_indices), batch):
        uids = user_indices[start : start + batch]
        scores = user_emb[torch.tensor(uids, dtype=torch.long)] @ item_emb.t()
        for r, u in enumerate(uids):
            seen = train_items_by_user.get(int(u))
            if seen:
                scores[r, list(seen)] = float("-inf")
        top = torch.topk(scores, k, dim=1).indices.cpu().numpy()
        for r, u in enumerate(uids):
            out[int(u)] = top[r].tolist()
    return out
