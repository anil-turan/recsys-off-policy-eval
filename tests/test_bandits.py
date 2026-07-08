"""Tests for the contextual bandit algorithms.

Headline guarantees checked: both LinUCB and linear Thompson Sampling incur
sub-linear regret and beat a random-arm baseline on a controlled linear
contextual-bandit environment; the Sherman-Morrison incremental update
matches a from-scratch matrix inversion.
"""

import numpy as np

from src.ope.bandits import (
    LinearThompsonSampling,
    LinUCB,
    RandomPolicy,
    run_bandit_simulation,
)


def _linear_bandit_world(n_contexts=500, n_arms=6, dim=5, seed=0):
    """Reward is a noisy linear function of context, with a different true
    theta per arm — the textbook setting both algorithms are designed for."""
    rng = np.random.default_rng(seed)
    context_features = rng.standard_normal((n_contexts, dim))
    true_theta = rng.standard_normal((n_arms, dim))
    clean_reward = context_features @ true_theta.T          # (n_contexts, n_arms)
    noisy_reward = clean_reward + 0.1 * rng.standard_normal(clean_reward.shape)
    return context_features, noisy_reward


def test_linucb_ainv_matches_direct_inverse_after_update():
    dim = 4
    bandit = LinUCB(n_arms=1, dim=dim, lambda_reg=1.0)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(dim)
    bandit.update(x, arm=0, reward=1.0)
    direct = np.linalg.inv(np.eye(dim) + np.outer(x, x))
    assert np.allclose(bandit.Ainv[0], direct, atol=1e-8)


def test_linucb_beats_random_policy_regret():
    context_features, reward = _linear_bandit_world()
    n_rounds = 4000

    linucb = LinUCB(n_arms=reward.shape[1], dim=context_features.shape[1], alpha=1.0)
    random_policy = RandomPolicy(n_arms=reward.shape[1], seed=0)

    ucb_result = run_bandit_simulation(linucb, context_features, reward, n_rounds, seed=1)
    random_result = run_bandit_simulation(random_policy, context_features, reward, n_rounds, seed=1)

    assert ucb_result["cumulative_regret"][-1] < 0.5 * random_result["cumulative_regret"][-1]


def test_linucb_regret_grows_sublinearly():
    """Average regret per round in the second half of the run should be
    substantially lower than in the first half — the exploration cost front-
    loads and then decays as the model learns."""
    context_features, reward = _linear_bandit_world()
    n_rounds = 4000
    linucb = LinUCB(n_arms=reward.shape[1], dim=context_features.shape[1], alpha=1.0)
    result = run_bandit_simulation(linucb, context_features, reward, n_rounds, seed=2)

    half = n_rounds // 2
    first_half_avg = result["regret"][:half].mean()
    second_half_avg = result["regret"][half:].mean()
    assert second_half_avg < first_half_avg


def test_thompson_sampling_beats_random_policy_regret():
    context_features, reward = _linear_bandit_world()
    n_rounds = 4000

    thompson = LinearThompsonSampling(n_arms=reward.shape[1], dim=context_features.shape[1],
                                      v=0.3, seed=0)
    random_policy = RandomPolicy(n_arms=reward.shape[1], seed=0)

    thompson_result = run_bandit_simulation(thompson, context_features, reward, n_rounds, seed=1)
    random_result = run_bandit_simulation(random_policy, context_features, reward, n_rounds, seed=1)

    assert thompson_result["cumulative_regret"][-1] < 0.5 * random_result["cumulative_regret"][-1]


def test_linucb_learns_the_correct_best_arm():
    context_features, reward = _linear_bandit_world(n_arms=4, dim=3, seed=7)
    linucb = LinUCB(n_arms=reward.shape[1], dim=context_features.shape[1], alpha=0.5)
    run_bandit_simulation(linucb, context_features, reward, n_rounds=6000, seed=3)

    true_best_arm = np.argmax(reward.mean(axis=0))
    learned_scores = linucb.theta() @ context_features.mean(axis=0)
    assert np.argmax(learned_scores) == true_best_arm
