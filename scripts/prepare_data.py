"""Ingest / preprocess steps for the Airflow DAG (kept tiny and idempotent).

    --step ingest      : download the raw MovieLens zip (cached; no-op if present)
    --step preprocess  : build the temporal split + matrices and print a summary
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recsys.data import download_movielens, load_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["ingest", "preprocess"], required=True)
    ap.add_argument("--dataset", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs" / "default.yaml"))
    d = cfg["data"]
    name = args.dataset or d["dataset"]
    data_dir = str(ROOT / d["data_dir"])

    if args.step == "ingest":
        path = download_movielens(name, data_dir)
        print(f"Ingested {name} -> {path}")
    else:
        ds = load_dataset(
            dataset=name, data_dir=data_dir, min_rating_positive=d["min_rating_positive"],
            min_user_interactions=d["min_user_interactions"], test_holdout=d["test_holdout"],
            val_holdout=d["val_holdout"], seed=d["seed"],
        )
        print("Preprocessed: " + ds.summary())


if __name__ == "__main__":
    main()
