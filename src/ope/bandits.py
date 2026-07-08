"""Contextual bandits for online recommendation, from scratch.

OPE (`estimators.py`) answers "how good would this policy have been?" from
fixed logs. This module answers the complementary question: how does a
recommender *learn* a good policy online, while paying as little exploration
cost as possible? Regret is that exploration cost, made precise:

    regret_t = max_a E[reward | context_t, a] - E[reward | context_t, a_t]

Both algorithms share the same linear-in-context model that the recommender
itself is built on: a per-arm (per-item) reward is modelled as
`theta_a . x`, where `x` is a user context vector (here, the iALS user
factors) and `theta_a` is learned online, one interaction at a time — no
batch refit required.

    LinUCB                  — Li et al. (2010), "A Contextual-Bandit Approach
                               to Personalized News Article Recommendation".
                               Deterministic: picks the arm with the highest
                               upper confidence bound on predicted reward.
    LinearThompsonSampling  — Agrawal & Goyal (2013), "Thompson Sampling for
                               Contextual Bandits with Linear Payoffs".
                               Stochastic: samples theta_a from its Bayesian
                               posterior each round and picks the argmax.

Both maintain, per arm, a ridge-regression posterior (A_a, b_a) and use the
Sherman-Morrison identity to update A_a^{-1} in O(d^2) instead of re-inverting
a (d x d) matrix from scratch every round.
"""

from __future__ import annotations

import numpy as np


class LinUCB:
    """Disjoint LinUCB: one ridge-regression head per arm, upper-confidence
    exploration. `alpha` controls the exploration width — alpha=0 degenerates
    to greedy exploitation of the current point estimate."""

    def __init__(self, n_arms: int, dim: int, alpha: float = 1.0, lambda_reg: float = 1.0):
        self.alpha = alpha
        self.Ainv = np.tile(np.eye(dim) / lambda_reg, (n_arms, 1, 1))
        self.b = np.zeros((n_arms, dim))

    def select_arm(self, x: np.ndarray, rng: np.random.Generator | None = None) -> int:
        theta = np.einsum("aij,aj->ai", self.Ainv, self.b)
        Ainv_x = np.einsum("aij,j->ai", self.Ainv, x)
        mean = theta @ x
        variance = Ainv_x @ x
        ucb = mean + self.alpha * np.sqrt(np.clip(variance, 0.0, None))
        return int(np.argmax(ucb))

    def update(self, x: np.ndarray, arm: int, reward: float) -> None:
        Ainv_x = self.Ainv[arm] @ x
        denom = 1.0 + x @ Ainv_x
        self.Ainv[arm] -= np.outer(Ainv_x, Ainv_x) / denom
        self.b[arm] += reward * x

    def theta(self) -> np.ndarray:
        """Current per-arm point estimate, for turning the learned bandit
        into a scoreable policy after the simulation ends."""
        return np.einsum("aij,aj->ai", self.Ainv, self.b)


class LinearThompsonSampling:
    """Bayesian linear bandit: theta_a ~ N(mu_a, v^2 * A_a^{-1}) posterior,
    sampled fresh each round. `v` scales the posterior spread — larger v
    explores more. Caches a Cholesky factor per arm and only refactorises the
    one arm that was just updated, so a full round costs O(n_arms * d^2)
    rather than O(n_arms * d^3)."""

    def __init__(self, n_arms: int, dim: int, v: float = 1.0, lambda_reg: float = 1.0,
                 seed: int = 0):
        self.v = v
        self.dim = dim
        self.Ainv = np.tile(np.eye(dim) / lambda_reg, (n_arms, 1, 1))
        self.b = np.zeros((n_arms, dim))
        self._chol = np.tile(v * np.eye(dim) / np.sqrt(lambda_reg), (n_arms, 1, 1))
        self._rng = np.random.default_rng(seed)

    def select_arm(self, x: np.ndarray, rng: np.random.Generator | None = None) -> int:
        rng = rng or self._rng
        mean = np.einsum("aij,aj->ai", self.Ainv, self.b)
        z = rng.standard_normal((mean.shape[0], self.dim))
        theta_sample = mean + np.einsum("aij,aj->ai", self._chol, z)
        scores = theta_sample @ x
        return int(np.argmax(scores))

    def update(self, x: np.ndarray, arm: int, reward: float) -> None:
        Ainv_x = self.Ainv[arm] @ x
        denom = 1.0 + x @ Ainv_x
        self.Ainv[arm] -= np.outer(Ainv_x, Ainv_x) / denom
        self.b[arm] += reward * x
        self._chol[arm] = np.linalg.cholesky(self.v ** 2 * self.Ainv[arm])

    def theta(self) -> np.ndarray:
        return np.einsum("aij,aj->ai", self.Ainv, self.b)


class RandomPolicy:
    """Uniform-random arm choice — the regret floor every real policy must
    beat, and a sanity check that the simulation harness itself is correct."""

    def __init__(self, n_arms: int, seed: int = 0):
        self.n_arms = n_arms
        self._rng = np.random.default_rng(seed)

    def select_arm(self, x: np.ndarray, rng: np.random.Generator | None = None) -> int:
        rng = rng or self._rng
        return int(rng.integers(self.n_arms))

    def update(self, x: np.ndarray, arm: int, reward: float) -> None:
        pass


def run_bandit_simulation(policy, context_features: np.ndarray, reward_matrix: np.ndarray,
                          n_rounds: int, seed: int = 0) -> dict:
    """Online loop: each round draws a random context (user), lets `policy`
    pick an arm, reveals the reward, and lets the policy update.

    context_features : (n_contexts, dim) — one feature vector per context.
    reward_matrix     : (n_contexts, n_arms) ground-truth reward, used only to
                         reveal the pulled arm's reward and to compute the
                         regret against the best arm for that context.
    """
    rng = np.random.default_rng(seed)
    n_contexts = context_features.shape[0]
    contexts = rng.integers(0, n_contexts, size=n_rounds)

    rewards = np.empty(n_rounds)
    regrets = np.empty(n_rounds)
    arms = np.empty(n_rounds, dtype=int)

    for t, c in enumerate(contexts):
        x = context_features[c]
        a = policy.select_arm(x, rng)
        r = reward_matrix[c, a]
        policy.update(x, a, r)
        rewards[t] = r
        regrets[t] = reward_matrix[c].max() - r
        arms[t] = a

    return {
        "context": contexts,
        "arm": arms,
        "reward": rewards,
        "regret": regrets,
        "cumulative_regret": np.cumsum(regrets),
        "cumulative_reward": np.cumsum(rewards),
    }
