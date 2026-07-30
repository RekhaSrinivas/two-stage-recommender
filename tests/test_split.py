"""Guardrails on the temporal split: no leakage, correct ordering, right sizes.

Uses a tiny in-memory frame so it runs without downloading MovieLens.
"""
import pandas as pd

from recsys.data import _temporal_leave_last_out


def _toy():
    # user 1: 5 interactions at ts 1..5 ; user 2: 4 interactions at ts 10..13
    rows = [(1, 101, 5, 1), (1, 102, 4, 2), (1, 103, 5, 3), (1, 104, 4, 4), (1, 105, 5, 5),
            (2, 201, 5, 10), (2, 202, 4, 11), (2, 203, 5, 12), (2, 204, 4, 13)]
    return pd.DataFrame(rows, columns=["user", "item", "rating", "ts"])


def test_holdout_picks_most_recent():
    train, val, test = _temporal_leave_last_out(_toy(), test_holdout=1, val_holdout=1)
    # Most-recent interaction per user goes to TEST.
    assert set(test[test.user == 1]["item"]) == {105}
    assert set(test[test.user == 2]["item"]) == {204}
    # Second most-recent goes to VAL.
    assert set(val[val.user == 1]["item"]) == {104}
    assert set(val[val.user == 2]["item"]) == {203}


def test_no_leakage_across_splits():
    train, val, test = _temporal_leave_last_out(_toy(), test_holdout=1, val_holdout=1)
    for u in [1, 2]:
        tr_ts = train[train.user == u]["ts"]
        va_ts = val[val.user == u]["ts"]
        te_ts = test[test.user == u]["ts"]
        # Every train interaction is strictly older than val, which is older than test.
        assert tr_ts.max() < va_ts.min()
        assert va_ts.max() < te_ts.min()


def test_partition_is_complete():
    df = _toy()
    train, val, test = _temporal_leave_last_out(df, test_holdout=1, val_holdout=1)
    assert len(train) + len(val) + len(test) == len(df)
    # No (user,item) row appears in more than one split.
    all_rows = pd.concat([train, val, test])
    assert len(all_rows.drop_duplicates(["user", "item"])) == len(df)
