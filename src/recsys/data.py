"""Data loading, implicit-feedback framing, and a leakage-free temporal split.

Design decisions you should be able to defend in an interview:

1. **Implicit feedback.** We keep only ratings >= `min_rating_positive` (default 4)
   and treat each as a single positive interaction. Real systems almost never see
   1-5 stars; they see clicks / orders / plays. Framing MovieLens this way makes the
   modelling honest about the production setting.

2. **Temporal, per-user leave-last-out split.** For each user we sort interactions by
   timestamp and hold out the most-recent `test_holdout` for TEST and the ones just
   before for VALIDATION; everything earlier is TRAIN. This mimics deployment: you
   train on the past and predict the future. A random split would leak future
   information into training and inflate every metric.

3. **Catalogue = items seen in TRAIN.** Val/test interactions pointing at items that
   never appear in training are dropped (and counted) — a model cannot rank an item it
   has never seen. This is the standard protocol; we log how many rows it removes so
   nothing is hidden.
"""
from __future__ import annotations

import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

MOVIELENS_URLS = {
    "ml-100k": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "ml-25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
}


@dataclass
class RecDataset:
    """Everything downstream models need, in reindexed (0..n-1) id space."""

    name: str
    train: pd.DataFrame                      # cols: user, item, rating, ts (reindexed ids)
    val: pd.DataFrame
    test: pd.DataFrame
    n_users: int
    n_items: int
    train_matrix: sp.csr_matrix             # (n_users x n_items) binary implicit feedback
    train_items_by_user: dict[int, set]     # items to EXCLUDE at recommendation time
    val_items_by_user: dict[int, set]       # ground truth for validation
    test_items_by_user: dict[int, set]      # ground truth for test
    item_popularity: np.ndarray             # train interaction count per item idx
    item_genres: sp.csr_matrix | None       # (n_items x n_genres) binary, for diversity
    genre_names: list[str]
    user_id_map: dict[int, int]             # original -> reindexed
    item_id_map: dict[int, int]
    item_titles: dict[int, str]             # reindexed item idx -> human title (for serving/UI)

    def summary(self) -> str:
        density = self.train_matrix.nnz / (self.n_users * self.n_items)
        return (
            f"{self.name}: {self.n_users:,} users x {self.n_items:,} items | "
            f"train={len(self.train):,} val={len(self.val):,} test={len(self.test):,} | "
            f"density={density:.4%}"
        )


def download_movielens(dataset: str, data_dir: str | Path) -> Path:
    """Download+unzip a MovieLens release if not already present. Returns extract dir."""
    if dataset not in MOVIELENS_URLS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {list(MOVIELENS_URLS)}")
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = data_dir / dataset
    if extract_dir.exists():
        return extract_dir

    zip_path = data_dir / f"{dataset}.zip"
    if not zip_path.exists():
        print(f"Downloading {dataset} from {MOVIELENS_URLS[dataset]} ...")
        urllib.request.urlretrieve(MOVIELENS_URLS[dataset], zip_path)
    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    return extract_dir


def _load_raw(dataset: str, root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (ratings[user,item,rating,ts], movies[item,title,genres])."""
    if dataset == "ml-100k":
        ratings = pd.read_csv(
            root / "u.data", sep="\t", names=["user", "item", "rating", "ts"], engine="python"
        )
        movies = pd.read_csv(
            root / "u.item", sep="|", encoding="latin-1", header=None, engine="python",
            usecols=[0, 1] + list(range(5, 24)),
        )
        genre_cols = [
            "unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
            "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
            "Romance", "Sci-Fi", "Thriller", "War", "Western",
        ]
        movies.columns = ["item", "title"] + genre_cols
        movies["genres"] = movies[genre_cols].apply(
            lambda r: "|".join(g for g, v in zip(genre_cols, r) if v == 1), axis=1
        )
        movies = movies[["item", "title", "genres"]]
    elif dataset == "ml-1m":
        ratings = pd.read_csv(
            root / "ratings.dat", sep="::", names=["user", "item", "rating", "ts"],
            engine="python", encoding="latin-1",
        )
        movies = pd.read_csv(
            root / "movies.dat", sep="::", names=["item", "title", "genres"],
            engine="python", encoding="latin-1",
        )
    elif dataset == "ml-25m":
        ratings = pd.read_csv(root / "ratings.csv")
        ratings.columns = ["user", "item", "rating", "ts"]
        movies = pd.read_csv(root / "movies.csv")
        movies.columns = ["item", "title", "genres"]
    else:
        raise ValueError(dataset)
    return ratings, movies


def _temporal_leave_last_out(
    df: pd.DataFrame, test_holdout: int, val_holdout: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-user: most-recent `test_holdout` -> test, next `val_holdout` -> val, rest -> train.

    Ties on timestamp are broken deterministically by item id so the split is
    fully reproducible.
    """
    df = df.sort_values(["user", "ts", "item"], kind="mergesort").reset_index(drop=True)
    # 0 == most recent interaction for the user.
    rank_from_end = df.groupby("user").cumcount(ascending=False)
    test_mask = rank_from_end < test_holdout
    val_mask = (rank_from_end >= test_holdout) & (rank_from_end < test_holdout + val_holdout)
    train_mask = rank_from_end >= test_holdout + val_holdout
    return df[train_mask].copy(), df[val_mask].copy(), df[test_mask].copy()


def _build_genre_matrix(
    movies: pd.DataFrame, item_id_map: dict[int, int], n_items: int
) -> tuple[sp.csr_matrix | None, list[str]]:
    movies = movies[movies["item"].isin(item_id_map)].copy()
    if movies.empty or "genres" not in movies:
        return None, []
    all_genres = sorted({g for gs in movies["genres"] for g in str(gs).split("|") if g and g != "(no genres listed)"})
    if not all_genres:
        return None, []
    genre_idx = {g: j for j, g in enumerate(all_genres)}
    rows, cols = [], []
    for _, r in movies.iterrows():
        i = item_id_map[r["item"]]
        for g in str(r["genres"]).split("|"):
            if g in genre_idx:
                rows.append(i)
                cols.append(genre_idx[g])
    mat = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_items, len(all_genres)))
    return mat, all_genres


def load_dataset(
    dataset: str = "ml-1m",
    data_dir: str | Path = "data",
    min_rating_positive: float = 4.0,
    min_user_interactions: int = 5,
    test_holdout: int = 1,
    val_holdout: int = 1,
    seed: int = 42,
) -> RecDataset:
    """Full pipeline: download -> filter -> split -> reindex -> build matrices."""
    root = download_movielens(dataset, data_dir)
    ratings, movies = _load_raw(dataset, root)

    # 1) Implicit feedback: keep positives only.
    pos = ratings[ratings["rating"] >= min_rating_positive].copy()

    # 2) Drop users with too few positives to support a train+val+test split.
    counts = pos.groupby("user")["item"].transform("size")
    pos = pos[counts >= max(min_user_interactions, test_holdout + val_holdout + 1)]

    # 3) Temporal split BEFORE reindexing (so we can define the catalogue from train).
    train, val, test = _temporal_leave_last_out(pos, test_holdout, val_holdout)

    # 4) Catalogue = users & items present in TRAIN. Drop val/test rows outside it.
    user_ids = np.sort(train["user"].unique())
    item_ids = np.sort(train["item"].unique())
    user_id_map = {u: i for i, u in enumerate(user_ids)}
    item_id_map = {it: i for i, it in enumerate(item_ids)}

    def _reindex(frame: pd.DataFrame) -> pd.DataFrame:
        f = frame[frame["user"].isin(user_id_map) & frame["item"].isin(item_id_map)].copy()
        f["user"] = f["user"].map(user_id_map)
        f["item"] = f["item"].map(item_id_map)
        return f

    n_before = len(val) + len(test)
    train, val, test = _reindex(train), _reindex(val), _reindex(test)
    dropped = n_before - (len(val) + len(test))
    if dropped:
        print(f"Dropped {dropped:,} val/test interactions on cold (unseen-in-train) items.")

    n_users, n_items = len(user_ids), len(item_ids)

    # 5) Sparse binary train matrix + per-user ground-truth sets.
    train_matrix = sp.csr_matrix(
        (np.ones(len(train), dtype=np.float32), (train["user"].values, train["item"].values)),
        shape=(n_users, n_items),
    )
    train_matrix.data[:] = 1.0  # collapse any duplicate (user,item) to a single 1

    def _by_user(frame: pd.DataFrame) -> dict[int, set]:
        return {u: set(g) for u, g in frame.groupby("user")["item"]}

    item_popularity = np.asarray(train_matrix.sum(axis=0)).ravel()
    item_genres, genre_names = _build_genre_matrix(movies, item_id_map, n_items)
    item_titles = {item_id_map[it]: str(t) for it, t in zip(movies["item"], movies["title"])
                   if it in item_id_map} if "title" in movies else {}

    return RecDataset(
        name=dataset,
        train=train, val=val, test=test,
        n_users=n_users, n_items=n_items,
        train_matrix=train_matrix,
        train_items_by_user=_by_user(train),
        val_items_by_user=_by_user(val),
        test_items_by_user=_by_user(test),
        item_popularity=item_popularity,
        item_genres=item_genres, genre_names=genre_names,
        user_id_map=user_id_map, item_id_map=item_id_map, item_titles=item_titles,
    )
