"""Implicit Alternating Least Squares (iALS) matrix factorisation, from scratch.

Implements Hu, Koren & Volinsky (2008), "Collaborative Filtering for Implicit
Feedback Datasets". Each observed interaction gets a preference p=1 and a
confidence c = 1 + alpha, so the model is pulled hardest toward the pairs we
actually observed while still treating the vast unobserved majority as weak
negatives.

The efficiency trick that makes this tractable: for each user the normal
equation is

    x_u = (YtY + Y^T (C^u - I) Y + reg*I)^{-1} Y^T C^u p_u

and because C^u - I is zero except on the user's interacted items, the
per-user solve only touches those few items on top of a single precomputed
YtY (and symmetrically for items).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


class IALS:
    def __init__(self, factors: int = 64, reg: float = 0.01, alpha: float = 10.0,
                 iterations: int = 15, seed: int = 42):
        self.factors = factors
        self.reg = reg
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed
        self.user_factors_: np.ndarray | None = None
        self.item_factors_: np.ndarray | None = None

    def fit(self, interactions: csr_matrix) -> IALS:
        rng = np.random.default_rng(self.seed)
        n_users, n_items = interactions.shape
        # small random init
        self.user_factors_ = 0.01 * rng.standard_normal((n_users, self.factors))
        self.item_factors_ = 0.01 * rng.standard_normal((n_items, self.factors))

        Cui = interactions.tocsr()
        Ciu = interactions.T.tocsr()
        for _ in range(self.iterations):
            self._als_step(Cui, self.user_factors_, self.item_factors_)
            self._als_step(Ciu, self.item_factors_, self.user_factors_)
        return self

    def _als_step(self, C: csr_matrix, X: np.ndarray, Y: np.ndarray) -> None:
        """Solve for factor matrix X given fixed Y (in place)."""
        f = self.factors
        YtY = Y.T @ Y + self.reg * np.eye(f)
        for u in range(X.shape[0]):
            start, end = C.indptr[u], C.indptr[u + 1]
            idx = C.indices[start:end]
            if len(idx) == 0:
                X[u] = 0.0
                continue
            Yi = Y[idx]                       # (nnz, f)
            cu = 1.0 + self.alpha * C.data[start:end]   # confidence for these items
            # A = YtY + Y^T (C^u - I) Y ; b = Y^T C^u p_u  (p_u = 1 on idx)
            A = YtY + (Yi.T * (cu - 1.0)) @ Yi
            b = (Yi.T * cu).sum(axis=1)
            X[u] = np.linalg.solve(A, b)

    def recommend(self, user_ids, interactions: csr_matrix, k: int = 10) -> np.ndarray:
        assert self.user_factors_ is not None, "call fit() first"
        recs = np.empty((len(user_ids), k), dtype=int)
        for row, u in enumerate(user_ids):
            scores = self.item_factors_ @ self.user_factors_[u]
            seen = interactions[u].indices
            scores[seen] = -np.inf
            top = np.argpartition(-scores, k)[:k]
            recs[row] = top[np.argsort(-scores[top])]
        return recs

    def score_users(self, user_ids: np.ndarray) -> np.ndarray:
        """Full item-score matrix for the given users (for OPE policies)."""
        return self.user_factors_[user_ids] @ self.item_factors_.T
