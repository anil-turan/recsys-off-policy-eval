"""Tests for the ranking metrics — checked against hand-computed values."""

import numpy as np

from src.evaluation.ranking import evaluate


def test_perfect_ranking_scores_one():
    # user 0 likes items 1 and 2; recommend them first
    recs = np.array([[1, 2, 9, 8, 7]])
    relevant = {0: {1, 2}}
    m = evaluate(recs, relevant, np.array([0]), k=5)
    assert m["recall@5"] == 1.0
    assert m["map@5"] == 1.0
    assert m["ndcg@5"] == 1.0


def test_no_hits_scores_zero():
    recs = np.array([[5, 6, 7, 8, 9]])
    relevant = {0: {1, 2}}
    m = evaluate(recs, relevant, np.array([0]), k=5)
    assert m["recall@5"] == 0.0
    assert m["map@5"] == 0.0
    assert m["ndcg@5"] == 0.0


def test_ndcg_rewards_higher_ranks():
    """A relevant item at rank 1 should give higher NDCG than at rank 3."""
    relevant = {0: {1}}
    top = evaluate(np.array([[1, 8, 9]]), relevant, np.array([0]), k=3)["ndcg@3"]
    low = evaluate(np.array([[8, 9, 1]]), relevant, np.array([0]), k=3)["ndcg@3"]
    assert top > low
    assert top == 1.0  # single relevant item at top => perfect


def test_precision_and_recall_partial():
    # 1 of 2 relevant items retrieved in top-2
    recs = np.array([[1, 9]])
    relevant = {0: {1, 2}}
    m = evaluate(recs, relevant, np.array([0]), k=2)
    assert m["precision@2"] == 0.5   # 1 hit / 2 recommended
    assert m["recall@2"] == 0.5      # 1 hit / min(2 relevant, 2)


def test_coverage_counts_distinct_recommended_items():
    recs = np.array([[0, 1], [0, 1]])  # only 2 distinct items ever recommended
    relevant = {0: {0}, 1: {1}}        # catalogue implied size = 2
    m = evaluate(recs, relevant, np.array([0, 1]), k=2)
    assert m["coverage"] == 1.0
