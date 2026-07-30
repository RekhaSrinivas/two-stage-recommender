"""A/B statistics + click-simulation checks."""
import math

import numpy as np
import pytest

from recsys.experiment import (examine, required_n_per_arm, simulate_clicks,
                               two_proportion_ztest)


def test_examine_position_bias():
    assert examine(0, "log") == pytest.approx(1.0)              # 1/log2(2)
    assert examine(1, "log") == pytest.approx(1 / math.log2(3))
    assert examine(0) > examine(5)                             # higher positions examined more


def test_simulate_clicks_deterministic_edges():
    rng = np.random.default_rng(0)
    # user 1: relevant item at rank 0 -> examine=1.0 -> always clicks.
    # user 2: relevant item not shown -> never clicks.
    recs = {1: [99, 5, 6], 2: [5, 6, 7]}
    truth = {1: {99}, 2: {42}}
    clicks = simulate_clicks(recs, truth, rng)
    assert clicks[1] == 1
    assert clicks[2] == 0


def test_two_proportion_ztest_detects_lift():
    p1, p2, z, pval = two_proportion_ztest(50, 1000, 100, 1000)  # 5% vs 10%
    assert p1 == pytest.approx(0.05) and p2 == pytest.approx(0.10)
    assert z > 0 and pval < 0.001


def test_required_n_grows_as_effect_shrinks():
    big = required_n_per_arm(0.05, 0.02)     # detect a 2pp lift
    small = required_n_per_arm(0.05, 0.005)  # detect a 0.5pp lift -> needs far more users
    assert small > big > 0
