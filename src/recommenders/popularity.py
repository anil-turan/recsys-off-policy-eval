"""Popularity baseline — recommend the globally most-interacted items.

Non-personalised, but a genuinely strong baseline that most naive models fail
to beat. Any personalised model has to justify its complexity against this.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


class PopularityRecommender:
    def __init__(self):
        self.item_scores_: np.ndarray | None = None

    def fit(self, interactions: csr_matrix) -> PopularityRecommender:
        # score = number of users who interacted with each item
        self.item_scores_ = np.asarray(interactions.sum(axis=0)).ravel()
        return self

    def recommend(self, user_ids, interactions: csr_matrix, k: int = 10) -> np.ndarray:
        """Top-k items per user, excluding items already seen in training."""
        assert self.item_scores_ is not None, "call fit() first"
        recs = np.empty((len(user_ids), k), dtype=int)
        for row, u in enumerate(user_ids):
            scores = self.item_scores_.copy()
            seen = interactions[u].indices
            scores[seen] = -np.inf
            top = np.argpartition(-scores, k)[:k]
            recs[row] = top[np.argsort(-scores[top])]
        return recs
