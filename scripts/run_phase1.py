"""Phase 1 driver: load data, run baselines, evaluate, print + save a comparison table.

Usage:
    python scripts/run_phase1.py                      # uses configs/default.yaml
    python scripts/run_phase1.py --dataset ml-100k    # quick smoke test
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recsys.baselines import ALS, ItemItemCF, MostPopular  # noqa: E402
from recsys.data import load_dataset  # noqa: E402
from recsys.metrics import evaluate  # noqa: E402


def _fmt_table(rows: list[dict], ks: list[int]) -> str:
    metrics = [f"recall@{max(ks)}", f"ndcg@{max(ks)}", f"map@{max(ks)}", "mrr",
               f"hit@{max(ks)}", f"coverage@{max(ks)}", f"novelty@{max(ks)}"]
    header = "| Model | " + " | ".join(metrics) + " |"
    sep = "|" + "---|" * (len(metrics) + 1)
    lines = [header, sep]
    for r in rows:
        cells = [r["model"]] + [f"{r[m]:.4f}" for m in metrics]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--dataset", default=None, help="override config dataset")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.dataset:
        cfg["data"]["dataset"] = args.dataset
    dcfg, ks = cfg["data"], cfg["eval"]["ks"]
    max_k = max(ks)

    print("Loading dataset ...")
    ds = load_dataset(
        dataset=dcfg["dataset"], data_dir=str(ROOT / dcfg["data_dir"]),
        min_rating_positive=dcfg["min_rating_positive"],
        min_user_interactions=dcfg["min_user_interactions"],
        test_holdout=dcfg["test_holdout"], val_holdout=dcfg["val_holdout"], seed=dcfg["seed"],
    )
    print(ds.summary())

    models = [
        MostPopular(),
        ItemItemCF(topn_neighbors=cfg["baselines"]["itemcf"]["topn_neighbors"]),
        ALS(**cfg["baselines"]["als"]),
    ]

    test_users = list(ds.test_items_by_user.keys())
    rows = []
    for model in models:
        t0 = time.time()
        model.fit(ds)
        recs = model.recommend(test_users, max_k)
        metrics = evaluate(recs, ds.test_items_by_user, ks, ds.n_items,
                           ds.item_popularity, ds.item_genres)
        metrics["model"] = model.name
        rows.append(metrics)
        print(f"  {model.name:14s} done in {time.time() - t0:5.1f}s | "
              f"ndcg@{max_k}={metrics[f'ndcg@{max_k}']:.4f} "
              f"recall@{max_k}={metrics[f'recall@{max_k}']:.4f}")

    table = _fmt_table(rows, ks)
    print("\n=== Phase 1 results (" + ds.name + ", test set) ===\n" + table)

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / f"phase1_{ds.name}.md", "w", encoding="utf-8") as f:
        f.write(f"# Phase 1 baselines — {ds.name}\n\n{ds.summary()}\n\n{table}\n")
        f.write("\n\n## Full metric dump\n\n")
        for r in rows:
            f.write(f"### {r['model']}\n\n")
            for key in sorted(k for k in r if k != "model"):
                f.write(f"- {key}: {r[key]:.4f}\n")
            f.write("\n")
    print(f"\nSaved -> {out_dir / f'phase1_{ds.name}.md'}")


if __name__ == "__main__":
    main()
