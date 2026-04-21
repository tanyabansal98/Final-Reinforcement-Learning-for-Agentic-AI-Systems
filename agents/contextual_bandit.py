"""
agents/contextual_bandit.py
METHOD 4 — Exploration Strategies
Implements LinUCB (contextual UCB) and Thompson Sampling.
These handle macro-level source category selection based on query context.
"""

import numpy as np
from typing import List, Dict, Optional


class LinUCB:
    """
    Linear Upper Confidence Bound (LinUCB) contextual bandit.

    For each arm (source) a, maintains a ridge regression model:
        reward(a, x) ≈ θ_a^T x
    UCB score: θ_a^T x + α * sqrt(x^T A_a^{-1} x)

    The exploration bonus sqrt(x^T A_a^{-1} x) grows for contexts
    where arm a has not been well-explored yet.

    Reference: Li et al. (2010) "A Contextual-Bandit Approach to
    Personalized News Article Recommendation"
    """

    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        alpha: float = 1.0,
        lambda_reg: float = 1.0,
    ):
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.alpha = alpha          # exploration strength
        self.lambda_reg = lambda_reg

        # Per-arm parameters
        # A_a: (d x d) regularized design matrix, initialized to λI
        # b_a: (d,) reward-weighted feature sum
        self.A = [np.eye(context_dim) * lambda_reg for _ in range(n_arms)]
        self.b = [np.zeros(context_dim) for _ in range(n_arms)]

        # Stats for analysis
        self.pull_counts = np.zeros(n_arms, dtype=int)
        self.cumulative_rewards = np.zeros(n_arms)
        self.ucb_history: List[np.ndarray] = []

    def select_action(self, context: np.ndarray, exclude: Optional[List[int]] = None) -> int:
        """
        Select best arm using UCB scores given current context.
        exclude: list of arm indices already queried this session.
        """
        x = context.astype(np.float64)
        scores = np.full(self.n_arms, -np.inf)

        for a in range(self.n_arms):
            if exclude and a in exclude:
                continue
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            exploitation = theta @ x
            exploration = self.alpha * np.sqrt(x @ A_inv @ x)
            scores[a] = exploitation + exploration

        self.ucb_history.append(scores.copy())
        return int(np.argmax(scores))

    def update(self, arm: int, context: np.ndarray, reward: float):
        """Update arm model with observed (context, reward) pair."""
        x = context.astype(np.float64)
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
        self.pull_counts[arm] += 1
        self.cumulative_rewards[arm] += reward

    def get_theta(self, arm: int) -> np.ndarray:
        """Return estimated parameter vector for arm (for analysis)."""
        A_inv = np.linalg.inv(self.A[arm])
        return A_inv @ self.b[arm]

    def get_all_thetas(self) -> np.ndarray:
        """Return (n_arms, context_dim) matrix of all θ estimates."""
        return np.stack([self.get_theta(a) for a in range(self.n_arms)])

    def estimated_rewards(self, context: np.ndarray) -> np.ndarray:
        """Return estimated reward for each arm at given context."""
        x = context.astype(np.float64)
        return np.array([self.get_theta(a) @ x for a in range(self.n_arms)])


class ThompsonSampling:
    """
    Thompson Sampling contextual bandit.
    Models each arm's reward as Gaussian with unknown mean,
    using a normal-inverse-gamma conjugate prior.

    At decision time: sample θ_a from posterior for each arm,
    select arm with highest expected reward under sampled θ.

    Tends to explore more smoothly than UCB — useful for comparison.
    """

    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        v: float = 1.0,          # prior variance scale
        lambda_reg: float = 1.0,
    ):
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.v = v
        self.lambda_reg = lambda_reg

        # Posterior parameters per arm (Gaussian approximation)
        self.mu = [np.zeros(context_dim) for _ in range(n_arms)]
        self.B = [np.eye(context_dim) * lambda_reg for _ in range(n_arms)]
        self.f = [np.zeros(context_dim) for _ in range(n_arms)]

        self.pull_counts = np.zeros(n_arms, dtype=int)
        self.cumulative_rewards = np.zeros(n_arms)

    def select_action(self, context: np.ndarray, exclude: Optional[List[int]] = None) -> int:
        """Sample θ from posterior for each arm, pick highest expected reward."""
        x = context.astype(np.float64)
        samples = np.full(self.n_arms, -np.inf)

        for a in range(self.n_arms):
            if exclude and a in exclude:
                continue
            B_inv = np.linalg.inv(self.B[a])
            # Sample θ from N(mu_a, v^2 * B_a^{-1})
            try:
                theta_sample = np.random.multivariate_normal(
                    self.mu[a], self.v ** 2 * B_inv
                )
            except np.linalg.LinAlgError:
                theta_sample = self.mu[a]
            samples[a] = theta_sample @ x

        return int(np.argmax(samples))

    def update(self, arm: int, context: np.ndarray, reward: float):
        """Update posterior for arm with new (context, reward) observation."""
        x = context.astype(np.float64)
        self.B[arm] += np.outer(x, x)
        self.f[arm] += reward * x
        self.mu[arm] = np.linalg.inv(self.B[arm]) @ self.f[arm]
        self.pull_counts[arm] += 1
        self.cumulative_rewards[arm] += reward

    def estimated_rewards(self, context: np.ndarray) -> np.ndarray:
        x = context.astype(np.float64)
        return np.array([self.mu[a] @ x for a in range(self.n_arms)])


class ContextualBanditAgent:
    """
    Wraps LinUCB and Thompson Sampling for easy use in training loop.
    Runs both in parallel so learning curves can be compared directly.
    """

    def __init__(
        self,
        n_sources: int,
        context_dim: int,
        alpha: float = 1.0,
        use_thompson: bool = True,
    ):
        self.ucb = LinUCB(n_sources, context_dim, alpha=alpha)
        self.ts = ThompsonSampling(n_sources, context_dim) if use_thompson else None
        self.n_sources = n_sources
        self._active = "ucb"   # which one drives actual decisions

    def select_action(
        self,
        context: np.ndarray,
        exclude: Optional[List[int]] = None,
        use: str = "ucb",
    ) -> int:
        """Select source to query. use='ucb' or 'ts'."""
        if use == "ts" and self.ts is not None:
            return self.ts.select_action(context, exclude=exclude)
        return self.ucb.select_action(context, exclude=exclude)

    def update(self, arm: int, context: np.ndarray, reward: float):
        """Update both models with new observation."""
        self.ucb.update(arm, context, reward)
        if self.ts is not None:
            self.ts.update(arm, context, reward)

    def get_learned_preferences(self, context: np.ndarray) -> Dict[str, np.ndarray]:
        """Return estimated source values at given context (for heatmap plotting)."""
        return {
            "ucb": self.ucb.estimated_rewards(context),
            "ts": self.ts.estimated_rewards(context) if self.ts else None,
        }
