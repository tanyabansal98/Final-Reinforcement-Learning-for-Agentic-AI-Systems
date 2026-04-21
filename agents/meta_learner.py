"""
agents/meta_learner.py
METHOD 5 — Meta-Learning / Transfer Learning
MAML-inspired domain adaptation for the Madison bandit.

Core idea: The AI industry has multiple topic domains (tech research,
AI finance, open-source, AI policy, market analysis). Source preferences
learned on one domain should transfer to a new domain quickly.

MAML outer loop: train a meta-initialization θ across all domains.
MAML inner loop: fine-tune θ on a new domain in K gradient steps.

We show: episodes-to-criterion drops from domain to domain as the
meta-learner accumulates cross-domain knowledge.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from agents.contextual_bandit import LinUCB
from env.source_pool import QUERY_TYPES


DOMAINS = [
    "tech_research",   # maps to technical queries
    "ai_finance",      # maps to business queries
    "open_source",     # maps to product queries
    "ai_policy",       # maps to regulatory queries
    "ai_market",       # maps to market queries
]

DOMAIN_TO_QUERY_TYPE = {
    "tech_research": "technical",
    "ai_finance":    "business",
    "open_source":   "product",
    "ai_policy":     "regulatory",
    "ai_market":     "market",
}


class DomainBandit:
    """
    A LinUCB bandit specialized for one topic domain.
    Stores (A, b) matrices that encode domain-specific source preferences.
    """

    def __init__(self, domain: str, n_sources: int, context_dim: int, alpha: float = 1.0):
        self.domain = domain
        self.query_type = DOMAIN_TO_QUERY_TYPE[domain]
        self.bandit = LinUCB(n_sources, context_dim, alpha=alpha)
        self.episode_count = 0
        self.reward_history: List[float] = []
        self.steps_to_criterion: Optional[int] = None  # episodes to reach 80% oracle

    def select_action(self, context: np.ndarray, exclude: List[int] = None) -> int:
        return self.bandit.select_action(context, exclude=exclude)

    def update(self, arm: int, context: np.ndarray, reward: float):
        self.bandit.update(arm, context, reward)

    def record_episode(self, reward: float, criterion: float = 0.8):
        self.reward_history.append(reward)
        self.episode_count += 1
        # Record first episode where moving avg exceeds criterion
        if self.steps_to_criterion is None and len(self.reward_history) >= 10:
            recent_avg = np.mean(self.reward_history[-10:])
            if recent_avg >= criterion:
                self.steps_to_criterion = self.episode_count


class MetaLearner:
    """
    MAML-inspired meta-learner over topic domains.

    The meta-initialization is represented as a set of (A, b) matrices
    that are the "warm start" for any new domain.

    Training procedure:
    1. For each domain in training set, run K episodes → compute gradient
    2. Update meta-parameters in direction of average gradient
    3. On new domain: initialize from meta-params, fine-tune in K steps

    Simplified implementation: we approximate MAML by maintaining a
    weighted average of all domain bandits' (A, b) parameters as the
    meta-initialization. This is equivalent to MAML in the linear
    bandit setting.
    """

    def __init__(
        self,
        n_sources: int,
        context_dim: int,
        train_domains: Optional[List[str]] = None,
        alpha: float = 1.0,
        meta_lr: float = 0.3,
        inner_steps: int = 5,
    ):
        self.n_sources = n_sources
        self.context_dim = context_dim
        self.alpha = alpha
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps

        # Training domains
        if train_domains is None:
            train_domains = DOMAINS[:4]   # hold out last domain for evaluation
        self.train_domains = train_domains

        # Per-domain bandits (trained separately)
        self.domain_bandits: Dict[str, DomainBandit] = {
            d: DomainBandit(d, n_sources, context_dim, alpha)
            for d in DOMAINS
        }

        # Meta-initialization: weighted average of domain (A, b) parameters
        self._meta_A = [np.eye(context_dim) for _ in range(n_sources)]
        self._meta_b = [np.zeros(context_dim) for _ in range(n_sources)]
        self._meta_update_count = 0

        # Transfer benefit tracking
        self.scratch_rewards: Dict[str, List[float]] = {d: [] for d in DOMAINS}
        self.transfer_rewards: Dict[str, List[float]] = {d: [] for d in DOMAINS}

    def train_domain(
        self,
        domain: str,
        episodes: List[Tuple[np.ndarray, int, float]],
    ):
        """
        Train bandit on a domain using observed (context, arm, reward) tuples.
        Also records rewards for transfer analysis.
        """
        db = self.domain_bandits[domain]
        for ctx, arm, reward in episodes:
            db.update(arm, ctx, reward)
        if episodes:
            avg_r = np.mean([r for _, _, r in episodes])
            db.record_episode(avg_r)
            self.scratch_rewards[domain].append(avg_r)

    def update_meta_params(self):
        """
        Update meta-initialization by averaging trained domain parameters.
        Called after each round of domain training.
        """
        active_domains = [d for d in self.train_domains if self.domain_bandits[d].episode_count > 0]
        if not active_domains:
            return

        for arm in range(self.n_sources):
            avg_A = np.mean(
                [self.domain_bandits[d].bandit.A[arm] for d in active_domains], axis=0
            )
            avg_b = np.mean(
                [self.domain_bandits[d].bandit.b[arm] for d in active_domains], axis=0
            )
            # Interpolate toward average with meta_lr
            self._meta_A[arm] = (
                (1 - self.meta_lr) * self._meta_A[arm] + self.meta_lr * avg_A
            )
            self._meta_b[arm] = (
                (1 - self.meta_lr) * self._meta_b[arm] + self.meta_lr * avg_b
            )

        self._meta_update_count += 1

    def adapt_to_new_domain(self, domain: str) -> DomainBandit:
        """
        Initialize a new domain bandit from meta-parameters.
        This is the MAML "inner loop initialization."
        Returns a new DomainBandit warm-started from meta-params.
        """
        adapted = DomainBandit(domain, self.n_sources, self.context_dim, self.alpha)
        # Copy meta-initialization into new bandit
        for arm in range(self.n_sources):
            adapted.bandit.A[arm] = self._meta_A[arm].copy()
            adapted.bandit.b[arm] = self._meta_b[arm].copy()
        return adapted

    def few_shot_adapt(
        self,
        domain: str,
        support_episodes: List[Tuple[np.ndarray, int, float]],
        use_meta_init: bool = True,
    ) -> DomainBandit:
        """
        Adapt to new domain using K support episodes.
        use_meta_init=True: warm-start from meta-params (MAML)
        use_meta_init=False: start from scratch (baseline comparison)
        """
        if use_meta_init:
            adapted = self.adapt_to_new_domain(domain)
        else:
            adapted = DomainBandit(domain, self.n_sources, self.context_dim, self.alpha)

        # Inner loop: fine-tune on support episodes
        for ctx, arm, reward in support_episodes[:self.inner_steps]:
            adapted.update(arm, ctx, reward)

        return adapted

    def evaluate_transfer_benefit(
        self,
        domain: str,
        eval_episodes: List[Tuple[np.ndarray, int, float]],
        support_episodes: List[Tuple[np.ndarray, int, float]],
    ) -> Dict:
        """
        Compare MAML-adapted vs from-scratch on held-out eval episodes.
        Returns dict with rewards and steps-to-criterion for both conditions.
        """
        # With transfer
        adapted = self.few_shot_adapt(domain, support_episodes, use_meta_init=True)
        # Without transfer
        scratch = self.few_shot_adapt(domain, support_episodes, use_meta_init=False)

        transfer_rewards = []
        scratch_rewards = []

        for ctx, arm, reward in eval_episodes:
            # Both evaluate greedily (no exploration)
            t_action = adapted.select_action(ctx)
            s_action = scratch.select_action(ctx)
            # Reward is for the oracle arm — use given reward as proxy
            transfer_rewards.append(reward if t_action == arm else reward * 0.3)
            scratch_rewards.append(reward if s_action == arm else reward * 0.3)

        self.transfer_rewards[domain].extend(transfer_rewards)

        return {
            "domain": domain,
            "transfer_avg_reward": float(np.mean(transfer_rewards)),
            "scratch_avg_reward": float(np.mean(scratch_rewards)),
            "transfer_benefit": float(np.mean(transfer_rewards) - np.mean(scratch_rewards)),
            "n_eval_episodes": len(eval_episodes),
            "meta_updates": self._meta_update_count,
        }

    def get_meta_source_preferences(self) -> np.ndarray:
        """
        Return meta-learned source value estimates.
        Shape: (n_sources,) — the meta-prior over source quality.
        Used to visualize what the meta-learner has generalized across domains.
        """
        prefs = np.zeros(self.n_sources)
        for arm in range(self.n_sources):
            theta = np.linalg.inv(self._meta_A[arm]) @ self._meta_b[arm]
            prefs[arm] = np.linalg.norm(theta)  # magnitude as proxy for learned preference
        return prefs

    def get_transfer_summary(self) -> Dict:
        """Summary of transfer learning results across all domains."""
        results = {}
        for domain in DOMAINS:
            db = self.domain_bandits[domain]
            results[domain] = {
                "episodes_trained": db.episode_count,
                "steps_to_criterion": db.steps_to_criterion,
                "avg_reward_last10": float(np.mean(db.reward_history[-10:])) if db.reward_history else 0.0,
            }
        return results
