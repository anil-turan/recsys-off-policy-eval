"""Sanity tests for the recommenders on a tiny synthetic interaction matrix."""

import numpy as np
from scipy.sparse import csr_matrix

from src.recommenders.ials import IALS
from src.recommenders.popularity import PopularityRecommender


def _block_matrix():
    """Two user groups with disjoint item tastes — a recommender should learn
    to recommend a group's items to that group."""
    rows, cols = [], []
    for u in range(10):            # group A likes items 0-4
        for i in range(5):
            rows.append(u)
            cols.append(i)
    for u in range(10, 20):        # group B likes items 5-9
        for i in range(5, 10):
            rows.append(u)
            cols.append(i)
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(20, 10))


def test_popularity_recommends_and_excludes_seen():
    M = _block_matrix()
    rec = PopularityRecommender().fit(M)
    recs = rec.recommend(np.array([0]), M, k=3)
    # user 0 has already seen items 0-4, so none should be recommended
    assert not set(recs[0]) & set(range(5))


def test_ials_learns_group_structure():
    M = _block_matrix()
    ials = IALS(factors=8, reg=0.01, alpha=10, iterations=20, seed=0).fit(M)
    # for a group-B user, held-out group-B items should score above group-A items
    scores = ials.item_factors_ @ ials.user_factors_[15]
    assert scores[5:10].mean() > scores[0:5].mean()


def test_ials_recommend_shape_and_exclusion():
    M = _block_matrix()
    ials = IALS(factors=8, iterations=10, seed=0).fit(M)
    recs = ials.recommend(np.array([0, 15]), M, k=3)
    assert recs.shape == (2, 3)
    assert not set(recs[0]) & set(range(5))     # user 0 saw items 0-4
