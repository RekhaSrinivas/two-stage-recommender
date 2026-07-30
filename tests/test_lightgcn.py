"""LightGCN correctness: the normalized adjacency is the piece an interviewer will probe,
so we check its symmetry and exact 1/sqrt(deg_u * deg_i) entries against a hand example.

Graph: users {0,1}, items {0,1}; interactions (u0,i0), (u1,i0), (u1,i1).
Degrees: u0=1, u1=2, i0=2, i1=1. Node ids: users 0,1 ; items 2,3.
"""
import math

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from recsys.lightgcn import LightGCN, build_norm_adj

R = sp.csr_matrix(np.array([[1, 0], [1, 1]], dtype=float))  # users x items


def test_norm_adj_symmetric_and_normalized():
    A = build_norm_adj(R, n_users=2, n_items=2).to_dense()
    assert torch.allclose(A, A.t())                              # undirected -> symmetric
    assert A[0, 2].item() == pytest.approx(1 / math.sqrt(1 * 2))  # (u0,i0): deg 1,2
    assert A[1, 2].item() == pytest.approx(1 / math.sqrt(2 * 2))  # (u1,i0): deg 2,2
    assert A[1, 3].item() == pytest.approx(1 / math.sqrt(2 * 1))  # (u1,i1): deg 2,1


def test_norm_adj_has_no_within_side_edges():
    A = build_norm_adj(R, n_users=2, n_items=2).to_dense()
    assert A[0, 1].item() == 0.0    # user-user
    assert A[2, 3].item() == 0.0    # item-item
    assert A[0, 0].item() == 0.0    # no self-loop (LightGCN keeps it out of A_hat)


def test_propagate_shapes():
    model = LightGCN(2, 2, build_norm_adj(R, 2, 2), dim=8, n_layers=3)
    ue, ie = model.propagate()
    assert ue.shape == (2, 8) and ie.shape == (2, 8)


def test_bpr_loss_zero_margin():
    # If the negative item IS the positive item, scores are equal -> loss = -log sigmoid(0).
    model = LightGCN(2, 2, build_norm_adj(R, 2, 2), dim=8, n_layers=2)
    u = torch.tensor([0, 1])
    same = torch.tensor([0, 1])
    loss = model.bpr_loss(u, same, same, reg=0.0)
    assert loss.item() == pytest.approx(math.log(2), abs=1e-5)
