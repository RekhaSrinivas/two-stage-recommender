"""Behavioural tests for the two-tower model and its in-batch softmax loss.

The accidental-hit masking test is the important one: it proves that when the same item
shows up as a positive for two rows in a batch, we don't wrongly penalise it as a negative.
"""
import math

import numpy as np
import torch

from recsys.two_tower import TwoTower, build_log_q, in_batch_softmax_loss


def test_tower_output_shapes():
    model = TwoTower(n_users=10, n_items=8, embedding_dim=16, hidden=[32], out_dim=16)
    uv = model.user_forward(torch.tensor([0, 1, 2]))
    iv = model.item_forward(torch.tensor([0, 1, 2]))
    assert uv.shape == (3, 16) and iv.shape == (3, 16)


def test_loss_rewards_diagonal():
    # Identity-like: user i matches item i strongly, others weakly -> low loss.
    uv = torch.eye(3) * 10.0
    iv = torch.eye(3) * 1.0
    ids = torch.tensor([0, 1, 2])
    loss = in_batch_softmax_loss(uv, iv, ids)
    assert loss.item() < 0.05


def test_accidental_hit_masking():
    # Two rows, identical vectors, SAME item id. All pairwise scores are equal.
    uv = torch.ones(2, 4)
    iv = torch.ones(2, 4)

    # Distinct ids: the off-diagonal is a legitimate negative -> uniform softmax -> log(2).
    loss_distinct = in_batch_softmax_loss(uv, iv, torch.tensor([5, 6]))
    assert abs(loss_distinct.item() - math.log(2)) < 1e-4

    # Same id: the off-diagonal is the SAME item, must be masked -> loss collapses to ~0.
    loss_same = in_batch_softmax_loss(uv, iv, torch.tensor([5, 5]))
    assert loss_same.item() < 1e-4


def test_logq_correction_changes_logits():
    torch.manual_seed(0)
    uv = torch.randn(4, 8)
    iv = torch.randn(4, 8)
    ids = torch.tensor([0, 1, 2, 3])
    pop = np.array([100.0, 50.0, 10.0, 1.0, 1.0])
    log_q = build_log_q(pop)
    a = in_batch_softmax_loss(uv, iv, ids, log_q=None)
    b = in_batch_softmax_loss(uv, iv, ids, log_q=log_q)
    assert not math.isclose(a.item(), b.item())  # correction actually does something
