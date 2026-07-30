"""Phase 2 driver: train the two-tower retriever and compare it to the Phase 1 baselines
on the exact same test set and metric harness.

Usage:
    python scripts/run_phase2.py                    # ml-1m
    python scripts/run_phase2.py --dataset ml-100k  # fast smoke test
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recsys.baselines import ALS, ItemItemCF, MostPopular  # noqa: E402
from recsys.data import load_dataset  # noqa: E402
from recsys.metrics import evaluate  # noqa: E402
from recsys.torch_data import InteractionDataset, build_item_matrix, recommend  # noqa: E402
from recsys.two_tower import TwoTower, build_log_q, in_batch_softmax_loss  # noqa: E402


def set_seed(s: int) -> None:
    np.random.seed(s)
    torch.manual_seed(s)


def train_two_tower(ds, cfg, ks, device):
    tt = cfg["two_tower"]
    set_seed(tt["seed"])
    max_k = max(ks)
    model = TwoTower(
        ds.n_users, ds.n_items, embedding_dim=tt["embedding_dim"], hidden=tt["hidden"],
        out_dim=tt["out_dim"], dropout=tt["dropout"],
        item_genres=ds.item_genres if tt["use_item_genres"] else None,
    ).to(device)
    log_q = build_log_q(ds.item_popularity).to(device) if tt["logq_correction"] else None

    loader = DataLoader(
        InteractionDataset(ds.train), batch_size=tt["batch_size"], shuffle=True, drop_last=True,
        generator=torch.Generator().manual_seed(tt["seed"]),
    )
    opt = torch.optim.Adam(model.parameters(), lr=tt["lr"], weight_decay=tt["weight_decay"])
    val_users = list(ds.val_items_by_user.keys())
    best_score, best_state, bad = -1.0, None, 0

    for epoch in range(1, tt["epochs"] + 1):
        model.train()
        total, nb = 0.0, 0
        for users, items in loader:
            users, items = users.to(device), items.to(device)
            uv, iv = model.user_forward(users), model.item_forward(items)
            loss = in_batch_softmax_loss(uv, iv, items, log_q, tt["temperature"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1

        item_matrix = build_item_matrix(model, ds.n_items, device)
        recs = recommend(model, item_matrix, val_users, ds.train_items_by_user, max_k, device)
        vm = evaluate(recs, ds.val_items_by_user, ks, ds.n_items, ds.item_popularity)
        score = vm[f"ndcg@{max_k}"]
        print(f"  epoch {epoch:2d} | loss={total / nb:.4f} | "
              f"val ndcg@{max_k}={score:.4f} recall@{max_k}={vm[f'recall@{max_k}']:.4f}")
        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= tt["patience"]:
                print(f"  early stop (no val gain for {tt['patience']} epochs)")
                break

    model.load_state_dict(best_state)
    return model


def fmt_table(rows, ks) -> str:
    metrics = [f"recall@{max(ks)}", f"ndcg@{max(ks)}", f"map@{max(ks)}", "mrr",
               f"hit@{max(ks)}", f"coverage@{max(ks)}", f"novelty@{max(ks)}"]
    header = "| Model | " + " | ".join(metrics) + " |"
    sep = "|" + "---|" * (len(metrics) + 1)
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join([r["model"]] + [f"{r[m]:.4f}" for m in metrics]) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--dataset", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.dataset:
        cfg["data"]["dataset"] = args.dataset
    dcfg, ks = cfg["data"], cfg["eval"]["ks"]
    max_k = max(ks)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ds = load_dataset(
        dataset=dcfg["dataset"], data_dir=str(ROOT / dcfg["data_dir"]),
        min_rating_positive=dcfg["min_rating_positive"],
        min_user_interactions=dcfg["min_user_interactions"],
        test_holdout=dcfg["test_holdout"], val_holdout=dcfg["val_holdout"], seed=dcfg["seed"],
    )
    print(ds.summary())
    test_users = list(ds.test_items_by_user.keys())

    rows = []
    # Baselines (fast) so the comparison lives in one reproducible table.
    for model in [MostPopular(), ItemItemCF(cfg["baselines"]["itemcf"]["topn_neighbors"]),
                  ALS(**cfg["baselines"]["als"])]:
        model.fit(ds)
        recs = model.recommend(test_users, max_k)
        m = evaluate(recs, ds.test_items_by_user, ks, ds.n_items, ds.item_popularity, ds.item_genres)
        m["model"] = model.name
        rows.append(m)

    print("Training two-tower ...")
    t0 = time.time()
    model = train_two_tower(ds, cfg, ks, device)
    item_matrix = build_item_matrix(model, ds.n_items, device)
    recs = recommend(model, item_matrix, test_users, ds.train_items_by_user, max_k, device)
    m = evaluate(recs, ds.test_items_by_user, ks, ds.n_items, ds.item_popularity, ds.item_genres)
    m["model"] = "TwoTower"
    rows.append(m)
    print(f"  two-tower total {time.time() - t0:.1f}s")

    # Persist the tower outputs so the ANN serving demo (scripts/ann_benchmark.py) can
    # index items and query users without retraining — this is exactly what a serving
    # layer consumes: precomputed vectors, not the model graph.
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    vec_dir = out_dir / f"vectors_{ds.name}"
    vec_dir.mkdir(exist_ok=True)
    with torch.no_grad():
        user_vecs = model.user_forward(torch.arange(ds.n_users, device=device)).cpu().numpy()
    np.save(vec_dir / "user_vecs.npy", user_vecs)
    np.save(vec_dir / "item_vecs.npy", item_matrix.cpu().numpy())
    print(f"  saved tower vectors -> {vec_dir}")

    table = fmt_table(rows, ks)
    print("\n=== Phase 2 results (" + ds.name + ", test set) ===\n" + table)

    with open(out_dir / f"phase2_{ds.name}.md", "w", encoding="utf-8") as f:
        f.write(f"# Phase 2 — two-tower vs baselines — {ds.name}\n\n{ds.summary()}\n\n{table}\n")
    print(f"\nSaved -> {out_dir / f'phase2_{ds.name}.md'}")


if __name__ == "__main__":
    main()
