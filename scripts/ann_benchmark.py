"""ANN serving demo: measure the accuracy/latency trade-off of approximate retrieval
against exact search, using the two-tower vectors saved by run_phase2.py.

Usage:
    python scripts/run_phase2.py            # first, to produce results/vectors_ml-1m/
    python scripts/ann_benchmark.py         # then this
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recsys.ann import ANNIndex, exact_topk, recall_at_k  # noqa: E402


def _time_search(fn, repeats: int = 3) -> float:
    fn()  # warm up
    t0 = time.time()
    for _ in range(repeats):
        fn()
    return (time.time() - t0) / repeats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ml-1m")
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()

    vec_dir = ROOT / "results" / f"vectors_{args.dataset}"
    user_vecs = np.load(vec_dir / "user_vecs.npy")
    item_vecs = np.load(vec_dir / "item_vecs.npy")
    print(f"users {user_vecs.shape} | items {item_vecs.shape} | k={args.k}\n")

    exact = exact_topk(user_vecs, item_vecs, args.k)
    exact_t = _time_search(lambda: exact_topk(user_vecs, item_vecs, args.k))

    flat = ANNIndex(item_vecs, kind="flat")
    flat_idx = flat.search(user_vecs, args.k)
    flat_t = _time_search(lambda: flat.search(user_vecs, args.k))

    rows = [("exact (numpy)", 1.0, exact_t),
            ("faiss flat (exact MIPS)", recall_at_k(flat_idx, exact), flat_t)]
    for nprobe in (1, 4, 8, 16):
        ivf = ANNIndex(item_vecs, kind="ivf", nprobe=nprobe)
        idx = ivf.search(user_vecs, args.k)
        t = _time_search(lambda: ivf.search(user_vecs, args.k))
        rows.append((f"faiss ivf (nprobe={nprobe})", recall_at_k(idx, exact), t))

    print(f"| Method | Recall@{args.k} vs exact | Latency (all users) | Speedup |")
    print("|---|---|---|---|")
    for name, rec, t in rows:
        print(f"| {name} | {rec:.4f} | {t * 1000:.2f} ms | {exact_t / t:.1f}x |")
    print("\n(Recall here = agreement with exact top-K, not the ranking metric from eval.)")


if __name__ == "__main__":
    main()
