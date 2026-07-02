"""Tests for the off-policy estimators.

The headline guarantee we check: on a controlled simulation where the true
policy value is known, IPS is unbiased and DR is close to the truth.
"""

import numpy as np

from src.ope.estimators import (
    direct_method,
    doubly_robust,
    effective_sample_size,
    ips,
    simulate_bandit_logs,
    snips,
    softmax_policy,
    true_policy_value,
)


def _toy_world(seed=0):
    rng = np.random.default_rng(seed)
    n_contexts, n_actions = 3000, 5
    scores = rng.normal(size=(n_contexts, n_actions))
    reward_true = (rng.random((n_contexts, n_actions)) < softmax_policy(scores)).astype(float)
    return scores, reward_true


def test_softmax_policy_is_a_distribution():
    scores = np.random.default_rng(0).normal(size=(100, 5))
    p = softmax_policy(scores, temperature=0.7, epsilon=0.1)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p > 0).all()  # epsilon mixing keeps full support


def test_ips_is_unbiased_over_many_logs():
    scores, reward_true = _toy_world()
    pi0 = softmax_policy(scores, temperature=1.0, epsilon=0.2)   # logging
    pi1 = softmax_policy(scores, temperature=0.5, epsilon=0.1)   # target
    truth = true_policy_value(pi1, reward_true)

    estimates = []
    for seed in range(200):
        log = simulate_bandit_logs(pi0, reward_true, seed=seed)
        u, a, r, ps = log["context"], log["action"], log["reward"], log["pscore"]
        estimates.append(ips(r, ps, pi1[u, a]))
    # mean of many IPS estimates should sit right on the truth
    assert abs(np.mean(estimates) - truth) < 0.01


def test_snips_has_lower_variance_than_ips():
    scores, reward_true = _toy_world()
    pi0 = softmax_policy(scores, temperature=1.0, epsilon=0.2)
    pi1 = softmax_policy(scores, temperature=0.5, epsilon=0.1)
    ips_est, snips_est = [], []
    for seed in range(200):
        log = simulate_bandit_logs(pi0, reward_true, seed=seed)
        u, a, r, ps = log["context"], log["action"], log["reward"], log["pscore"]
        tps = pi1[u, a]
        ips_est.append(ips(r, ps, tps))
        snips_est.append(snips(r, ps, tps))
    assert np.std(snips_est) <= np.std(ips_est)


def test_doubly_robust_recovers_truth_with_perfect_reward_model():
    """If the reward model equals the true reward, DR should be essentially
    exact regardless of the propensities."""
    scores, reward_true = _toy_world()
    pi0 = softmax_policy(scores, temperature=1.0, epsilon=0.3)
    pi1 = softmax_policy(scores, temperature=0.4, epsilon=0.1)
    truth = true_policy_value(pi1, reward_true)

    log = simulate_bandit_logs(pi0, reward_true, seed=1)
    u, a, r, ps = log["context"], log["action"], log["reward"], log["pscore"]
    q_all = reward_true[u]                       # oracle reward model
    q_logged = q_all[np.arange(len(u)), a]
    v_target = np.sum(pi1[u] * q_all, axis=1)
    dr = doubly_robust(r, ps, pi1[u, a], q_logged, v_target)
    dm = direct_method(v_target)
    assert abs(dr - truth) < 0.02
    assert abs(dm - truth) < 0.02


def test_ess_drops_when_policies_diverge():
    scores, reward_true = _toy_world()
    log = simulate_bandit_logs(softmax_policy(scores, epsilon=0.3), reward_true, seed=0)
    u, a, ps = log["context"], log["action"], log["pscore"]
    close = softmax_policy(scores, temperature=1.0, epsilon=0.3)[u, a]
    far = softmax_policy(scores, temperature=0.1, epsilon=0.01)[u, a]
    assert effective_sample_size(ps, close) > effective_sample_size(ps, far)
