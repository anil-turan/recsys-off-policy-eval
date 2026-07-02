"""Off-policy evaluation (OPE): estimate the value of a *target* policy from
data logged under a different *logging* policy — without deploying it.

The core problem: a new recommender (target policy) is expensive and risky to
A/B test on live traffic. OPE estimates how it *would* perform using only the
logs the current system already produced. This is the counterfactual analogue
of the causal work in the sibling uplift project.

Estimators implemented from scratch (all take numpy arrays):

    IPS  — Inverse Propensity Scoring: unbiased, but variance explodes when the
           target policy diverges from the logging policy.
    SNIPS— Self-Normalised IPS: divides by the sum of weights; small bias, much
           lower variance. The usual default.
    DM   — Direct Method: fit a reward model and average its predictions under
           the target policy; low variance but biased if the model is wrong.
    DR   — Doubly Robust: DM baseline + IPS correction on the residual;
           unbiased if *either* the propensities or the reward model are right.

Convention per logged sample i:
    reward[i]        : observed reward of the logged action
    pscore[i]        : logging-policy prob of the logged action  pi_0(a_i | x_i)
    target_pscore[i] : target-policy prob of the logged action   pi_1(a_i | x_i)
    q_hat_logged[i]  : reward-model estimate for the logged (x_i, a_i)
    v_hat_target[i]  : E_{a~pi_1(.|x_i)}[ q_hat(x_i, a) ]  (reward model under target)
"""

from __future__ import annotations

import numpy as np


def softmax_policy(scores: np.ndarray, temperature: float = 1.0,
                   epsilon: float = 0.0) -> np.ndarray:
    """Turn per-action scores (n_contexts x n_actions) into a stochastic policy.

    `epsilon` mixes in a uniform policy so every action keeps positive
    probability — this guarantees the overlap/common-support that unbiased
    IPS requires.
    """
    z = scores / temperature
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    if epsilon > 0:
        n_actions = scores.shape[1]
        p = (1 - epsilon) * p + epsilon / n_actions
    return p


def simulate_bandit_logs(logging_policy: np.ndarray, reward_true: np.ndarray,
                         seed: int = 0) -> dict:
    """Simulate one logged interaction per context under the logging policy.

    logging_policy : (n_contexts x n_actions) probabilities
    reward_true    : (n_contexts x n_actions) ground-truth reward (0/1 here)
    Returns the logged actions, observed rewards and logging propensities.
    """
    rng = np.random.default_rng(seed)
    n = logging_policy.shape[0]
    actions = np.array([rng.choice(logging_policy.shape[1], p=logging_policy[i]) for i in range(n)])
    rewards = reward_true[np.arange(n), actions]
    pscore = logging_policy[np.arange(n), actions]
    return {"context": np.arange(n), "action": actions, "reward": rewards, "pscore": pscore}


def true_policy_value(policy: np.ndarray, reward_true: np.ndarray) -> float:
    """Ground-truth value of a policy: expected reward if we actually deployed
    it. Only computable here because this is a controlled simulation — it is
    what the OPE estimates are validated against."""
    return float(np.mean(np.sum(policy * reward_true, axis=1)))


def ips(reward: np.ndarray, pscore: np.ndarray, target_pscore: np.ndarray) -> float:
    weights = target_pscore / pscore
    return float(np.mean(weights * reward))


def snips(reward: np.ndarray, pscore: np.ndarray, target_pscore: np.ndarray) -> float:
    weights = target_pscore / pscore
    return float(np.sum(weights * reward) / np.sum(weights))


def direct_method(v_hat_target: np.ndarray) -> float:
    return float(np.mean(v_hat_target))


def doubly_robust(reward: np.ndarray, pscore: np.ndarray, target_pscore: np.ndarray,
                  q_hat_logged: np.ndarray, v_hat_target: np.ndarray) -> float:
    weights = target_pscore / pscore
    return float(np.mean(v_hat_target + weights * (reward - q_hat_logged)))


def effective_sample_size(pscore: np.ndarray, target_pscore: np.ndarray) -> float:
    """ESS of the importance weights — a diagnostic for how much the IPS
    estimate can be trusted. Low ESS => high-variance, fragile estimate."""
    w = target_pscore / pscore
    return float(w.sum() ** 2 / np.sum(w ** 2))
