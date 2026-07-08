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
    switch_dr,
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


def test_switch_dr_equals_direct_method_at_tau_zero():
    """No importance weight is ever <= 0, so tau=0 switches the IPS
    correction off everywhere and Switch-DR collapses to the DM term."""
    scores, reward_true = _toy_world()
    pi0 = softmax_policy(scores, temperature=1.0, epsilon=0.3)
    pi1 = softmax_policy(scores, temperature=0.4, epsilon=0.1)

    log = simulate_bandit_logs(pi0, reward_true, seed=1)
    u, a, r, ps = log["context"], log["action"], log["reward"], log["pscore"]
    q_all = reward_true[u]
    q_logged = q_all[np.arange(len(u)), a]
    v_target = np.sum(pi1[u] * q_all, axis=1)

    sw = switch_dr(r, ps, pi1[u, a], q_logged, v_target, tau=0.0)
    dm = direct_method(v_target)
    assert sw == dm


def test_switch_dr_equals_doubly_robust_at_tau_infinity():
    """A threshold no weight can exceed means every sample keeps its IPS
    correction, so Switch-DR collapses to plain DR."""
    scores, reward_true = _toy_world()
    pi0 = softmax_policy(scores, temperature=1.0, epsilon=0.3)
    pi1 = softmax_policy(scores, temperature=0.4, epsilon=0.1)

    log = simulate_bandit_logs(pi0, reward_true, seed=1)
    u, a, r, ps = log["context"], log["action"], log["reward"], log["pscore"]
    q_all = reward_true[u]
    q_logged = q_all[np.arange(len(u)), a]
    v_target = np.sum(pi1[u] * q_all, axis=1)

    sw = switch_dr(r, ps, pi1[u, a], q_logged, v_target, tau=np.inf)
    dr = doubly_robust(r, ps, pi1[u, a], q_logged, v_target)
    assert abs(sw - dr) < 1e-10


def test_switch_dr_recovers_truth_with_perfect_reward_model():
    scores, reward_true = _toy_world()
    pi0 = softmax_policy(scores, temperature=1.0, epsilon=0.3)
    pi1 = softmax_policy(scores, temperature=0.4, epsilon=0.1)
    truth = true_policy_value(pi1, reward_true)

    log = simulate_bandit_logs(pi0, reward_true, seed=1)
    u, a, r, ps = log["context"], log["action"], log["reward"], log["pscore"]
    q_all = reward_true[u]
    q_logged = q_all[np.arange(len(u)), a]
    v_target = np.sum(pi1[u] * q_all, axis=1)

    sw = switch_dr(r, ps, pi1[u, a], q_logged, v_target, tau=2.0)
    assert abs(sw - truth) < 0.05


def test_switch_dr_variance_between_dm_and_dr_at_moderate_tau():
    """A moderate tau should sit between the DM (zero variance from
    weighting) and DR (full IPS correction) variance across repeated logs."""
    scores, reward_true = _toy_world()
    pi0 = softmax_policy(scores, temperature=1.0, epsilon=0.05)   # sparse logging
    pi1 = softmax_policy(scores, temperature=0.2, epsilon=0.01)   # sharp target -> heavy tails

    dm_est, dr_est, switch_est = [], [], []
    for seed in range(100):
        log = simulate_bandit_logs(pi0, reward_true, seed=seed)
        u, a, r, ps = log["context"], log["action"], log["reward"], log["pscore"]
        q_all = reward_true[u]
        q_logged = q_all[np.arange(len(u)), a]
        v_target = np.sum(pi1[u] * q_all, axis=1)
        tps = pi1[u, a]
        dm_est.append(direct_method(v_target))
        dr_est.append(doubly_robust(r, ps, tps, q_logged, v_target))
        switch_est.append(switch_dr(r, ps, tps, q_logged, v_target, tau=5.0))

    assert np.std(dm_est) <= np.std(switch_est) <= np.std(dr_est)


def test_ess_drops_when_policies_diverge():
    scores, reward_true = _toy_world()
    log = simulate_bandit_logs(softmax_policy(scores, epsilon=0.3), reward_true, seed=0)
    u, a, ps = log["context"], log["action"], log["pscore"]
    close = softmax_policy(scores, temperature=1.0, epsilon=0.3)[u, a]
    far = softmax_policy(scores, temperature=0.1, epsilon=0.01)[u, a]
    assert effective_sample_size(ps, close) > effective_sample_size(ps, far)
