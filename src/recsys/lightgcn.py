"""LightGCN: graph convolution on the user-item bipartite graph (He et al. 2020).

The idea in one breath: put users and items on a graph (an edge = an interaction), then
smooth each node's embedding over its neighbours, repeatedly. A user ends up represented by
the items they touched, blended with the *other* users who touched those items, and so on —
collaborative signal propagated along graph paths, which plain matrix factorization can't
see beyond first-order.

Why "Light": vanilla GCN does  E' = σ(A_hat · E · W)  — a learned weight matrix W and a
nonlinearity σ at each layer. LightGCN removes BOTH. He et al. showed they hurt
recommendation, because there are no node *features* to transform here — only IDs. So a
layer is just:

    E^(k+1) = A_hat · E^(k)          (A_hat = D^-1/2 A D^-1/2, symmetric normalized adjacency)

and the final embedding is the mean over layers  E = mean(E^0, E^1, ..., E^K). That's the
whole model: the only learned parameters are the layer-0 embeddings E^0.

Trained with **BPR** (Bayesian Personalized Ranking): for a (user, positive, negative)
triple, push the positive's score above the negative's — a pairwise ranking objective,
exactly what a recommender is judged on.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_norm_adj(train_matrix: sp.csr_matrix, n_users: int, n_items: int) -> torch.Tensor:
    """Symmetric normalized adjacency  A_hat = D^-1/2 A D^-1/2  of the bipartite graph.

    Node ids: users are 0..n_users-1, items are n_users..n_users+n_items-1. The adjacency is
    [[0, R], [R^T, 0]]; A_hat[u, i] = 1 / sqrt(deg(u) * deg(i)) for every edge (u, i).
    Returned as a coalesced sparse COO tensor of shape (N, N), N = n_users + n_items.
    """
    R = train_matrix.tocoo()
    N = n_users + n_items
    # Each interaction becomes two directed edges (u->item_node and item_node->u).
    rows = np.concatenate([R.row, R.col + n_users])
    cols = np.concatenate([R.col + n_users, R.row])

    user_deg = np.asarray(train_matrix.sum(axis=1)).ravel()
    item_deg = np.asarray(train_matrix.sum(axis=0)).ravel()
    deg = np.concatenate([user_deg, item_deg])
    deg[deg == 0] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(deg)

    vals = d_inv_sqrt[rows] * d_inv_sqrt[cols]
    idx = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    coo = torch.sparse_coo_tensor(idx, torch.tensor(vals, dtype=torch.float32), (N, N)).coalesce()
    # CSR layout: torch.sparse.mm on CSR is ~40x faster on CPU than COO, which dominates
    # LightGCN training time (one full-graph propagation per step).
    return coo.to_sparse_csr()


class LightGCN(nn.Module):
    def __init__(self, n_users: int, n_items: int, norm_adj: torch.Tensor,
                 dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.n_users, self.n_items, self.n_layers = n_users, n_items, n_layers
        self.emb = nn.Embedding(n_users + n_items, dim)     # layer-0 embeddings E^0
        nn.init.normal_(self.emb.weight, std=0.1)
        self.register_buffer("norm_adj", norm_adj)          # sparse A_hat, moves with .to(device)

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Run K propagation layers and average -> (user_embeddings, item_embeddings)."""
        e = self.emb.weight
        stacked = [e]
        for _ in range(self.n_layers):
            e = torch.sparse.mm(self.norm_adj, e)           # E^(k+1) = A_hat E^(k)
            stacked.append(e)
        final = torch.stack(stacked, dim=0).mean(dim=0)     # layer combination (mean)
        return final[: self.n_users], final[self.n_users :]

    def bpr_loss(self, users, pos_items, neg_items, reg: float):
        user_e, item_e = self.propagate()
        u, p, n = user_e[users], item_e[pos_items], item_e[neg_items]
        pos_scores = (u * p).sum(dim=-1)
        neg_scores = (u * n).sum(dim=-1)
        rank_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        # L2 on the layer-0 embeddings of the nodes in this batch (standard LightGCN reg).
        e0 = self.emb.weight
        reg_loss = reg * 0.5 * (
            e0[users].pow(2).sum(dim=1).mean()
            + e0[self.n_users + pos_items].pow(2).sum(dim=1).mean()
            + e0[self.n_users + neg_items].pow(2).sum(dim=1).mean()
        )
        return rank_loss + reg_loss

    @torch.no_grad()
    def embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        return self.propagate()
