"""Export step: materialize top-N recommendations per user from the full two-stage pipeline.

This is the DAG's final task — the artifact a downstream service or batch job would consume.
Writes original MovieLens ids (not the reindexed internal ids) so the output is portable.

Usage:  python scripts/export_topn.py --dataset ml-100k --topn 10
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recsys.data import load_dataset  # noqa: E402
from recsys.ranker import (FEATURE_NAMES, RankContext, build_training_data,  # noqa: E402
                           generate_candidates, rerank, train_ranker)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--topn", type=int, default=10)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs" / "default.yaml"))
    name = args.dataset or cfg["data"]["dataset"]

    ds = load_dataset(
        dataset=name, data_dir=str(ROOT / cfg["data"]["data_dir"]),
        min_rating_positive=cfg["data"]["min_rating_positive"],
        min_user_interactions=cfg["data"]["min_user_interactions"],
        test_holdout=cfg["data"]["test_holdout"], val_holdout=cfg["data"]["val_holdout"],
        seed=cfg["data"]["seed"],
    )
    vec = ROOT / "results" / f"vectors_{ds.name}"
    emb = ROOT / "results" / f"graph_emb_{ds.name}"
    if not (vec / "item_vecs.npy").exists() or not (emb / "item_graph_emb.npy").exists():
        sys.exit("Missing vectors/embeddings; run the retrieval & graph tasks first.")
    ctx = RankContext.build(
        np.load(vec / "user_vecs.npy"), np.load(vec / "item_vecs.npy"),
        np.load(emb / "user_graph_emb.npy"), np.load(emb / "item_graph_emb.npy"),
        ds.train_matrix, ds.item_genres,
    )

    N = cfg["ranker"]["n_candidates"]
    val_users = [u for u in ds.val_items_by_user if ds.val_items_by_user[u]]
    cand_train = generate_candidates(ctx.tt_user, ctx.tt_item, val_users, ds.train_items_by_user, N)
    X, y, groups = build_training_data(val_users, cand_train, ds.val_items_by_user, ctx)
    cols = list(range(len(FEATURE_NAMES)))
    model = train_ranker(X, y, groups, cols, cfg["ranker"], cfg["ranker"]["seed"])

    all_users = list(range(ds.n_users))
    cand = generate_candidates(ctx.tt_user, ctx.tt_item, all_users, ds.train_items_by_user, N)
    recs = rerank(cand, ctx, lambda f, c: model.predict(f[:, cols]), args.topn)

    inv_user = {v: k for k, v in ds.user_id_map.items()}
    inv_item = {v: k for k, v in ds.item_id_map.items()}
    out = ROOT / "results" / f"topn_{ds.name}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "rank", "item_id"])
        for u in all_users:
            for rank, it in enumerate(recs[u], start=1):
                w.writerow([inv_user[u], rank, inv_item[it]])
    print(f"Exported top-{args.topn} for {len(all_users):,} users -> {out}")


if __name__ == "__main__":
    main()
