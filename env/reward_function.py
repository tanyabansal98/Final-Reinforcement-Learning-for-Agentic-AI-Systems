"""
env/reward_function.py
Computes rewards for the Madison RL agent after each source query
and at the end of a session.

Reward design philosophy:
- Step reward:    small signal after each individual source query
- Session reward: rich signal at the end of the episode based on
                  the quality of the full collected set

Both rewards are used: step rewards stabilize Q-Learning and bandit
updates; session rewards train the PPO synthesis agent.
"""

import numpy as np
from typing import Dict, List


# Reward weights — tune these for ablation experiments
WEIGHTS = {
    "relevance":   1.0,   # raw quality of retrieved information
    "novelty":     0.4,   # bonus for new information not already found
    "latency":     0.1,   # penalty for slow sources
    "coverage":    0.3,   # bonus for covering multiple angles
    "efficiency":  0.2,   # bonus for finding good info with fewer queries
    "conflict":   -0.2,   # penalty for unresolved conflicting signals
}

MAX_LATENCY = 10.0  # seconds — above this, full latency penalty applied


def step_reward(result: Dict, session_results: List[Dict]) -> float:
    """
    Reward computed immediately after a single source query.
    Used to update contextual bandit and Q-Learning after each step.

    Args:
        result:          the just-received query result
        session_results: all results so far this session (including current)

    Returns:
        scalar reward in roughly [-0.5, 1.5]
    """
    if not result["available"]:
        return -0.1  # small penalty for unavailable source

    r = 0.0

    # Relevance: core signal
    r += WEIGHTS["relevance"] * result["relevance"]

    # Novelty: did this source add something new?
    prior_relevances = [x["relevance"] for x in session_results[:-1] if x["available"]]
    if prior_relevances:
        max_prior = max(prior_relevances)
        novelty = max(0.0, result["relevance"] - max_prior * 0.8)
    else:
        novelty = result["relevance"]  # first query always novel
    r += WEIGHTS["novelty"] * novelty

    # Latency penalty: normalize to [0, 1] then invert
    latency_penalty = min(result["latency"] / MAX_LATENCY, 1.0)
    r -= WEIGHTS["latency"] * latency_penalty

    return float(r)


def session_reward(session_summary: Dict, collected_results: List[Dict]) -> float:
    """
    Rich reward computed at the end of a full research session.
    Used to train the PPO synthesis agent and as the meta-learning signal.

    Args:
        session_summary:  from SessionState.summary()
        collected_results: full list of result dicts from the session

    Returns:
        scalar reward in roughly [-1.0, 3.0]
    """
    r = 0.0
    available = [x for x in collected_results if x["available"]]

    if not available:
        return -0.5  # nothing collected

    # Core quality: best relevance achieved
    r += WEIGHTS["relevance"] * session_summary["max_relevance"]

    # Average quality bonus
    r += 0.3 * session_summary["avg_relevance"]

    # Coverage: did we query diverse source categories?
    categories = set(x["category"] for x in available)
    coverage_score = len(categories) / 5.0  # 5 categories total
    r += WEIGHTS["coverage"] * coverage_score

    # Efficiency: reward finding high quality with fewer queries
    efficiency = session_summary["max_relevance"] / max(session_summary["n_queried"], 1)
    r += WEIGHTS["efficiency"] * efficiency

    # Conflict penalty
    if session_summary["has_conflict"]:
        r += WEIGHTS["conflict"]

    # Urgency adjustment: high urgency queries penalize latency more
    urgency_latency_multiplier = {"low": 0.5, "medium": 1.0, "high": 2.0}
    urgency = session_summary.get("urgency", "medium")
    latency_penalty = (session_summary["avg_latency"] / MAX_LATENCY) * urgency_latency_multiplier[urgency]
    r -= WEIGHTS["latency"] * latency_penalty

    return float(r)


def regret(optimal_reward: float, achieved_reward: float) -> float:
    """
    Instantaneous regret for a single episode.
    Cumulative regret = sum of this over all episodes.
    """
    return max(0.0, optimal_reward - achieved_reward)


def compute_oracle_reward(source_pool, query_type: str, budget: int) -> float:
    """
    Compute the reward an oracle agent would achieve — knows the true
    quality matrix and always queries the best sources first.
    Used as the upper bound for regret calculation.
    """
    matrix = source_pool.get_ground_truth_matrix()
    from env.source_pool import QUERY_TYPES
    qt_idx = QUERY_TYPES.index(query_type)
    quality_col = matrix[:, qt_idx]

    # Oracle picks top-budget sources
    top_indices = np.argsort(quality_col)[::-1][:budget]
    top_qualities = quality_col[top_indices]

    # Simulate oracle session reward
    mock_summary = {
        "max_relevance": float(top_qualities[0]) if len(top_qualities) > 0 else 0.0,
        "avg_relevance": float(np.mean(top_qualities)),
        "n_queried": budget,
        "avg_latency": 1.0,
        "has_conflict": False,
        "urgency": "medium",
    }
    mock_results = [{"available": True, "relevance": q, "category": "research", "latency": 1.0}
                    for q in top_qualities]
    return session_reward(mock_summary, mock_results)


class RewardTracker:
    """Tracks reward statistics across episodes for plotting."""
    def __init__(self):
        self.episode_rewards: List[float] = []
        self.episode_regrets: List[float] = []
        self.step_rewards: List[float] = []

    def record_episode(self, reward: float, regret_val: float = 0.0):
        self.episode_rewards.append(reward)
        self.episode_regrets.append(regret_val)

    def record_step(self, reward: float):
        self.step_rewards.append(reward)

    def moving_average(self, window: int = 20) -> np.ndarray:
        r = np.array(self.episode_rewards)
        if len(r) < window:
            return r
        return np.convolve(r, np.ones(window) / window, mode="valid")

    def cumulative_regret(self) -> np.ndarray:
        return np.cumsum(self.episode_regrets)
