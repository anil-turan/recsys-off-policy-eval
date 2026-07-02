"""Load MovieLens-1M and turn it into an implicit-feedback recommendation
problem with a leakage-free temporal split.

MovieLens ratings are explicit (1-5 stars). We binarise to *implicit*
feedback — a rating >= 4 is a positive interaction ("the user liked it") —
because implicit feedback is what production recommenders actually see and it
is the setting the ranking metrics and off-policy estimators assume.

The split is **temporal**, not random: we sort every user's interactions by
time and hold out their most recent ones for the test set. A random split
would leak the future into training and inflate every metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "ratings.dat"
LIKE_THRESHOLD = 4  # rating >= 4 counts as a positive implicit interaction


def load_ratings(path: Path | str = RAW_PATH) -> pd.DataFrame:
    """Load the raw ratings and reindex users/items to contiguous 0..n ids."""
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user", "item", "rating", "timestamp"],
    )
    df["liked"] = (df["rating"] >= LIKE_THRESHOLD).astype(int)
    # contiguous ids make the sparse-matrix construction trivial
    df["user_idx"] = df["user"].astype("category").cat.codes
    df["item_idx"] = df["item"].astype("category").cat.codes
    return df


@dataclass
class Split:
    train: pd.DataFrame
    test: pd.DataFrame
    n_users: int
    n_items: int

    def train_matrix(self, positives_only: bool = True) -> csr_matrix:
        """User x item sparse interaction matrix from the training split."""
        d = self.train[self.train["liked"] == 1] if positives_only else self.train
        return csr_matrix(
            (np.ones(len(d)), (d["user_idx"], d["item_idx"])),
            shape=(self.n_users, self.n_items),
        )

    def test_positives(self) -> dict[int, set[int]]:
        """Map each user to the set of items they liked in the test period —
        the ground-truth relevant set the ranking metrics score against."""
        pos = self.test[self.test["liked"] == 1]
        return pos.groupby("user_idx")["item_idx"].apply(set).to_dict()


def temporal_split(df: pd.DataFrame, test_frac: float = 0.2, min_interactions: int = 10) -> Split:
    """Hold out each user's most recent `test_frac` interactions for test.

    Users with fewer than `min_interactions` are dropped so every user has a
    usable history in both splits.
    """
    counts = df.groupby("user_idx").size()
    keep = counts[counts >= min_interactions].index
    df = df[df["user_idx"].isin(keep)].copy()

    df = df.sort_values(["user_idx", "timestamp"])
    df["rank"] = df.groupby("user_idx").cumcount()
    df["n"] = df.groupby("user_idx")["item_idx"].transform("size")
    cutoff = (df["n"] * (1 - test_frac)).astype(int)
    is_test = df["rank"] >= cutoff

    train, test = df[~is_test], df[is_test]
    return Split(
        train=train.reset_index(drop=True),
        test=test.reset_index(drop=True),
        n_users=int(df["user_idx"].max() + 1),
        n_items=int(df["item_idx"].max() + 1),
    )
