"""Phase 4 driver: the full two-stage pipeline.

  retrieval (two-tower top-N)  ->  ranking (LambdaMART re-orders the shortlist)

Reuses the vectors saved by Phase 2 (two-tower) and Phase 3 (LightGCN) — no retraining.
Reports two ablations that together justify the whole architecture:
  1. retrieval-only vs retrieval+ranking   -> does the ranking stage help?
  2. ranker WITH vs WITHOUT the graph feature -> do the graph embeddings pay off end-to-end?

Usage:
    python scripts/run_phase2.py         # first, to save two-tower vectors
    python scripts/run_phase3.py --only k3   # first, to save graph embeddings
    python scripts/run_phase4.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recsys.data import load_dataset  # noqa: E402
from recsys.metrics import evaluate  # noqa: E402
from recsys.ranker import (FEATURE_NAMES, GRAPH_FEATURES, RankContext,  # noqa: E402
                           build_training_data, generate_candidates, rerank,
                           train_ranker, union_seen)


def fmt_table(rows, ks):
    metrics = [f"recall@{max(ks)}", f"ndcg@{max(ks)}", f"map@{max(ks)}",
               f"hit@{max(ks)}", f"coverage@{max(ks)}"]
    lines = ["| Model | " + " | ".join(metrics) + " |", "|" + "---|" * (len(metrics) + 1)]
    for r in rows:
        lines.append("| " + " | ".join([r["model"]] + [f"{r[m]:.4f}" for m in metrics]) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--dataset", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.dataset:
        cfg["data"]["dataset"] = args.dataset
    ks = cfg["eval"]["ks"]
    max_k = max(ks)
    rcfg = cfg["ranker"]

    ds = load_dataset(
        dataset=cfg["data"]["dataset"], data_dir=str(ROOT / cfg["data"]["data_dir"]),
        min_rating_positive=cfg["data"]["min_rating_positive"],
        min_user_interactions=cfg["data"]["min_user_interactions"],
        test_holdout=cfg["data"]["test_holdout"], val_holdout=cfg["data"]["val_holdout"],
        seed=cfg["data"]["seed"],
    )
    print(ds.summary())

    vec = ROOT / "results" / f"vectors_{ds.name}"
    emb = ROOT / "results" / f"graph_emb_{ds.name}"
    if not (vec / "item_vecs.npy").exists() or not (emb / "item_graph_emb.npy").exists():
        sys.exit("Missing saved vectors/embeddings. Run run_phase2.py and run_phase3.py --only k3 first.")
    ctx = RankContext.build(
        np.load(vec / "user_vecs.npy"), np.load(vec / "item_vecs.npy"),
        np.load(emb / "user_graph_emb.npy"), np.load(emb / "item_graph_emb.npy"),
        ds.train_matrix, ds.item_genres,
    )

    N = rcfg["n_candidates"]
    val_users = [u for u in ds.val_items_by_user if ds.val_items_by_user[u]]
    test_users = [u for u in ds.test_items_by_user if ds.test_items_by_user[u]]
    print(f"Generating top-{N} candidates ...")
    cand_train = generate_candidates(ctx.tt_user, ctx.tt_item, val_users, ds.train_items_by_user, N)
    seen_te = union_seen(ds.train_items_by_user, ds.val_items_by_user)
    cand_test = generate_candidates(ctx.tt_user, ctx.tt_item, test_users, seen_te, N)

    # Honest ceiling: the ranker can only rank what retrieval surfaced.
    hit_in_N = np.mean([any(p in set(cand_test[u].tolist()) for p in ds.test_items_by_user[u])
                        for u in test_users])
    print(f"Retriever recall@{N} (ranking ceiling): {hit_in_N:.4f}")

    print("Training LambdaMART rankers (with / without graph feature) ...")
    X, y, groups = build_training_data(val_users, cand_train, ds.val_items_by_user, ctx)
    all_cols = list(range(len(FEATURE_NAMES)))
    nograph_cols = [i for i, f in enumerate(FEATURE_NAMES) if f not in GRAPH_FEATURES]
    m_full = train_ranker(X, y, groups, all_cols, rcfg, rcfg["seed"])
    m_nograph = train_ranker(X, y, groups, nograph_cols, rcfg, rcfg["seed"])

    tt_col = FEATURE_NAMES.index("tt_score")
    variants = [
        ("Retrieval only (two-tower)", lambda f, c: f[:, tt_col]),
        ("+ Rank (LambdaMART, no graph)", lambda f, c: m_nograph.predict(f[:, nograph_cols])),
        ("+ Rank (LambdaMART, with graph)", lambda f, c: m_full.predict(f[:, all_cols])),
    ]
    rows = []
    for name, fn in variants:
        recs = rerank(cand_test, ctx, fn, max_k)
        m = evaluate(recs, ds.test_items_by_user, ks, ds.n_items, ds.item_popularity, ds.item_genres)
        m["model"] = name
        rows.append(m)
        print(f"  {name:38s} ndcg@{max_k}={m[f'ndcg@{max_k}']:.4f} recall@{max_k}={m[f'recall@{max_k}']:.4f}")

    print("\nFeature importances (with-graph ranker):")
    for f, imp in sorted(zip(FEATURE_NAMES, m_full.feature_importances_), key=lambda x: -x[1]):
        print(f"  {f:14s} {imp}")

    table = fmt_table(rows, ks)
    print("\n=== Phase 4 results (" + ds.name + ", test set) ===\n" + table)
    out = ROOT / "results" / f"phase4_{ds.name}.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# Phase 4 — two-stage retrieval->ranking — {ds.name}\n\n{ds.summary()}\n\n"
                f"Retriever recall@{N} (ceiling): {hit_in_N:.4f}\n\n{table}\n")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
