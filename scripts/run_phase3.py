"""Phase 3 driver: train LightGCN on the user-item graph, compare to the baselines, and
save the learned graph embeddings so the Phase 4 ranker can use them as features.

Usage:
    python scripts/run_phase3.py                    # ml-1m
    python scripts/run_phase3.py --dataset ml-100k  # fast smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recsys.baselines import ALS, ItemItemCF, MostPopular  # noqa: E402
from recsys.data import load_dataset  # noqa: E402
from recsys.lightgcn import LightGCN, build_norm_adj  # noqa: E402
from recsys.metrics import evaluate  # noqa: E402
from recsys.torch_data import recommend_from_embeddings  # noqa: E402


def set_seed(s: int) -> None:
    np.random.seed(s)
    torch.manual_seed(s)


def train_lightgcn(ds, cfg, ks, device, n_layers=None):
    lg = cfg["lightgcn"]
    n_layers = lg["n_layers"] if n_layers is None else n_layers
    set_seed(lg["seed"])
    max_k = max(ks)
    norm_adj = build_norm_adj(ds.train_matrix, ds.n_users, ds.n_items).to(device)
    model = LightGCN(ds.n_users, ds.n_items, norm_adj, dim=lg["dim"], n_layers=n_layers).to(device)

    users = torch.tensor(ds.train["user"].values, dtype=torch.long)
    pos = torch.tensor(ds.train["item"].values, dtype=torch.long)
    loader = DataLoader(TensorDataset(users, pos), batch_size=lg["batch_size"], shuffle=True,
                        drop_last=True, generator=torch.Generator().manual_seed(lg["seed"]))
    opt = torch.optim.Adam(model.parameters(), lr=lg["lr"])
    val_users = list(ds.val_items_by_user.keys())
    best_score, best_state, bad = -1.0, None, 0
    rng = torch.Generator().manual_seed(lg["seed"])

    for epoch in range(1, lg["epochs"] + 1):
        model.train()
        total, nb = 0.0, 0
        for bu, bp in loader:
            bu, bp = bu.to(device), bp.to(device)
            # Uniform random negative items (BPR); rare collisions with positives are tolerated.
            bn = torch.randint(0, ds.n_items, (bu.shape[0],), generator=rng).to(device)
            loss = model.bpr_loss(bu, bp, bn, lg["reg"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1

        ue, ie = model.embeddings()
        recs = recommend_from_embeddings(ue.cpu(), ie.cpu(), val_users, ds.train_items_by_user, max_k)
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
            if bad >= lg["patience"]:
                print(f"  early stop (no val gain for {lg['patience']} epochs)")
                break

    model.load_state_dict(best_state)
    return model


def fmt_table(rows, ks) -> str:
    metrics = [f"recall@{max(ks)}", f"ndcg@{max(ks)}", f"map@{max(ks)}", "mrr",
               f"hit@{max(ks)}", f"coverage@{max(ks)}", f"novelty@{max(ks)}"]
    lines = ["| Model | " + " | ".join(metrics) + " |", "|" + "---|" * (len(metrics) + 1)]
    for r in rows:
        lines.append("| " + " | ".join([r["model"]] + [f"{r[m]:.4f}" for m in metrics]) + " |")
    return "\n".join(lines)


# K=0 == BPR-MF (no graph propagation); K=<config> == full LightGCN. The two are the
# "does the graph help?" ablation. Each is run as its OWN short job (this environment
# reaps background tasks after ~7 min), writing a JSON row that --assemble collects.
VARIANTS = {"k0": (0, "BPR-MF (K=0, no graph)"), "k3": (None, "LightGCN")}


def _load_ds(cfg):
    d = cfg["data"]
    return load_dataset(
        dataset=d["dataset"], data_dir=str(ROOT / d["data_dir"]),
        min_rating_positive=d["min_rating_positive"], min_user_interactions=d["min_user_interactions"],
        test_holdout=d["test_holdout"], val_holdout=d["val_holdout"], seed=d["seed"],
    )


def run_variant(only, cfg, ds, ks, device, rows_dir):
    n_layers, label = VARIANTS[only]
    n_layers = cfg["lightgcn"]["n_layers"] if n_layers is None else n_layers
    max_k = max(ks)
    test_users = list(ds.test_items_by_user.keys())
    print(f"Training {label} (K={n_layers}) ...")
    t0 = time.time()
    model = train_lightgcn(ds, cfg, ks, device, n_layers=n_layers)
    ue, ie = model.embeddings()
    recs = recommend_from_embeddings(ue.cpu(), ie.cpu(), test_users, ds.train_items_by_user, max_k)
    m = evaluate(recs, ds.test_items_by_user, ks, ds.n_items, ds.item_popularity, ds.item_genres)
    m["model"] = label
    print(f"  {label} total {time.time() - t0:.1f}s")
    rows_dir.mkdir(parents=True, exist_ok=True)
    json.dump(m, open(rows_dir / f"{only}.json", "w"))
    if only == "k3":  # save graph embeddings for the Phase 4 ranker
        emb_dir = ROOT / "results" / f"graph_emb_{ds.name}"
        emb_dir.mkdir(parents=True, exist_ok=True)
        np.save(emb_dir / "user_graph_emb.npy", ue.cpu().numpy())
        np.save(emb_dir / "item_graph_emb.npy", ie.cpu().numpy())
        print(f"  saved graph embeddings -> {emb_dir}")


def assemble(cfg, ds, ks, rows_dir, out_dir):
    max_k = max(ks)
    test_users = list(ds.test_items_by_user.keys())
    rows = []
    for model in [MostPopular(), ItemItemCF(cfg["baselines"]["itemcf"]["topn_neighbors"]),
                  ALS(**cfg["baselines"]["als"])]:
        model.fit(ds)
        m = evaluate(model.recommend(test_users, max_k), ds.test_items_by_user, ks,
                     ds.n_items, ds.item_popularity, ds.item_genres)
        m["model"] = model.name
        rows.append(m)
    for only in ("k0", "k3"):
        p = rows_dir / f"{only}.json"
        if p.exists():
            rows.append(json.load(open(p)))
        else:
            print(f"  (missing {p.name}; run --only {only} first)")
    table = fmt_table(rows, ks)
    print("\n=== Phase 3 results (" + ds.name + ", test set) ===\n" + table)
    with open(out_dir / f"phase3_{ds.name}.md", "w", encoding="utf-8") as f:
        f.write(f"# Phase 3 — LightGCN vs baselines — {ds.name}\n\n{ds.summary()}\n\n{table}\n")
    print(f"\nSaved -> {out_dir / f'phase3_{ds.name}.md'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--only", choices=["k0", "k3"], help="train just one variant, save a JSON row")
    ap.add_argument("--assemble", action="store_true", help="baselines + collect rows -> final table")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.dataset:
        cfg["data"]["dataset"] = args.dataset
    ks = cfg["eval"]["ks"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ds = _load_ds(cfg)
    print(ds.summary())
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    rows_dir = out_dir / f"phase3_rows_{ds.name}"

    if args.only:
        run_variant(args.only, cfg, ds, ks, device, rows_dir)
    elif args.assemble:
        assemble(cfg, ds, ks, rows_dir, out_dir)
    else:  # combined path (fine for small datasets that finish quickly)
        run_variant("k0", cfg, ds, ks, device, rows_dir)
        run_variant("k3", cfg, ds, ks, device, rows_dir)
        assemble(cfg, ds, ks, rows_dir, out_dir)


if __name__ == "__main__":
    main()
