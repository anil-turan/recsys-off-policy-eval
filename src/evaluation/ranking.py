"""Top-k ranking metrics for implicit-feedback recommendation, from scratch.

All metrics take a matrix of recommended item ids (one row per user, ranked)
and a dict mapping each user to their set of relevant (liked) test items.

    Recall@k   : fraction of a user's relevant items that appear in the top-k
    Precision@k: fraction of the top-k that are relevant
    MAP@k      : mean average precision — rewards putting hits near the top
    NDCG@k     : discounted cumulative gain, the standard ranking-quality metric
    Coverage   : fraction of the catalogue that ever gets recommended
                 (guards against a model that only pushes head items)
"""

from __future__ import annotations

import numpy as np


def _dcg(hits: np.ndarray) -> float:
    # hits is a 0/1 array in ranked order; discount by log2(position+1)
    return np.sum(hits / np.log2(np.arange(2, len(hits) + 2)))


def evaluate(recs: np.ndarray, relevant: dict[int, set[int]], user_ids: np.ndarray,
             k: int = 10) -> dict:
    """Compute all ranking metrics averaged over users who have >=1 relevant
    test item."""
    recalls, precisions, aps, ndcgs = [], [], [], []
    recommended_items = set()

    for row, u in enumerate(user_ids):
        rel = relevant.get(int(u))
        recommended_items.update(recs[row, :k].tolist())
        if not rel:
            continue
        topk = recs[row, :k]
        hits = np.array([1 if item in rel else 0 for item in topk])

        n_hits = hits.sum()
        recalls.append(n_hits / min(len(rel), k))
        precisions.append(n_hits / k)

        # average precision @ k
        if n_hits > 0:
            cum_hits = np.cumsum(hits)
            precision_at_i = cum_hits / (np.arange(k) + 1)
            aps.append(np.sum(precision_at_i * hits) / min(len(rel), k))
        else:
            aps.append(0.0)

        # ndcg @ k : dcg normalised by the ideal dcg
        ideal = _dcg(np.ones(min(len(rel), k)))
        ndcgs.append(_dcg(hits) / ideal if ideal > 0 else 0.0)

    n_items = max(max(s) for s in relevant.values() if s) + 1 if relevant else 1
    return {
        f"recall@{k}": float(np.mean(recalls)),
        f"precision@{k}": float(np.mean(precisions)),
        f"map@{k}": float(np.mean(aps)),
        f"ndcg@{k}": float(np.mean(ndcgs)),
        "coverage": len(recommended_items) / n_items,
        "n_users_scored": len(recalls),
    }
