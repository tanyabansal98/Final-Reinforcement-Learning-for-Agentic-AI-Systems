"""
env/madison_env.py
Main Gym-style environment for the Madison RL agent.
Wraps source pool, query engine, and reward function into a
clean step/reset interface compatible with all five RL agents.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from env.source_pool import SourcePool, QUERY_TYPES
from env.query_engine import QueryEngine, SessionState, CONTEXT_DIM
from env.reward_function import (
    RewardTracker, step_reward, session_reward,
    compute_oracle_reward, regret,
)


class MadisonEnv:
    """
    Single-agent Madison environment.
    One episode = one research query session.
    At each step, the agent selects a source index to query.
    Episode ends when budget is exhausted.

    Observation space: CONTEXT_DIM-dimensional float32 vector
    Action space:      discrete, size = n_sources (12)
    """

    def __init__(
        self,
        seed: int = 42,
        query_type: Optional[str] = None,   # fix type for curriculum training
        domain: str = "general",
    ):
        self.source_pool = SourcePool(seed=seed)
        self.query_engine = QueryEngine(seed=seed)
        self.reward_tracker = RewardTracker()

        self.n_sources = self.source_pool.n_sources
        self.context_dim = CONTEXT_DIM
        self.n_actions = self.n_sources

        self._fixed_query_type = query_type
        self._domain = domain
        self._session: Optional[SessionState] = None
        self._episode_count = 0

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self, query_type: Optional[str] = None) -> Tuple[np.ndarray, Dict]:
        """Start a new episode. Returns (observation, info)."""
        qt = query_type or self._fixed_query_type
        query = self.query_engine.sample_query(query_type=qt, domain=self._domain)
        self._session = SessionState(query=query)
        self._episode_count += 1
        obs = self._session.to_context_vector()
        info = {
            "query_text": query.text,
            "query_type": query.query_type,
            "urgency": query.urgency,
            "budget": query.budget,
            "episode": self._episode_count,
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take one step: query source[action].
        Returns (obs, reward, terminated, truncated, info)
        """
        assert self._session is not None, "Call reset() before step()"
        assert 0 <= action < self.n_sources, f"Invalid action {action}"

        # Query the chosen source
        result = self.source_pool.query(action, self._session.query.query_type)
        self._session.add_result(action, result)

        # Step reward (used by bandit + Q-Learning)
        r_step = step_reward(result, self._session.results)
        self.reward_tracker.record_step(r_step)

        terminated = self._session.is_done
        truncated = False

        # On terminal step: also compute session-level reward
        r_session = 0.0
        if terminated:
            summary = self._session.summary()
            r_session = session_reward(summary, self._session.results)
            oracle = compute_oracle_reward(
                self.source_pool,
                self._session.query.query_type,
                self._session.query.budget,
            )
            r_reg = regret(oracle, r_session)
            self.reward_tracker.record_episode(r_session, r_reg)

        obs = self._session.to_context_vector()
        info = {
            "result": result,
            "step_reward": r_step,
            "session_reward": r_session if terminated else None,
            "budget_remaining": self._session.budget_remaining,
            "sources_queried": list(self._session.sources_queried),
            "summary": self._session.summary() if terminated else None,
        }

        # Combined reward returned at each step
        # Step reward for dense feedback; session reward bonus on terminal
        reward = r_step + (r_session if terminated else 0.0)
        return obs, reward, terminated, truncated, info

    def get_session_state(self) -> Optional[SessionState]:
        return self._session

    def get_source_names(self) -> List[str]:
        return self.source_pool.source_names

    def get_ground_truth_matrix(self) -> np.ndarray:
        """Only for analysis — never pass to agents during training."""
        return self.source_pool.get_ground_truth_matrix()

    def reset_stats(self):
        self.source_pool.reset_counts()
        self.reward_tracker = RewardTracker()


class MultiAgentMadisonEnv:
    """
    Multi-agent wrapper: N parallel agents each receive a sub-query
    and must coordinate to avoid redundant source querying.
    Used by marl_coordinator.py (Method 3).
    """

    def __init__(self, n_agents: int = 3, seed: int = 42):
        self.n_agents = n_agents
        # Each agent gets its own env but shares the source pool reference
        self.envs = [MadisonEnv(seed=seed + i) for i in range(n_agents)]
        self.n_sources = self.envs[0].n_sources
        self.context_dim = CONTEXT_DIM
        # Track which sources have been queried across ALL agents this episode
        self._global_queried: set = set()

    def reset(self, query_type: Optional[str] = None) -> List[Tuple[np.ndarray, Dict]]:
        """Reset all agents. Returns list of (obs, info) per agent."""
        self._global_queried = set()
        return [env.reset(query_type=query_type) for env in self.envs]

    def step(
        self, actions: List[int]
    ) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        """
        Each agent takes one action simultaneously.
        Shared reward penalizes duplicate source queries across agents.

        Returns: (obs_list, reward_list, done_list, info)
        """
        results = []
        dones = []
        obs_list = []
        individual_rewards = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, terminated, _, info = env.step(action)
            results.append(info.get("result", {}))
            obs_list.append(obs)
            dones.append(terminated)
            individual_rewards.append(reward)

        # Shared coordination reward component
        # Penalize when multiple agents query the same source this step
        action_counts = {}
        for a in actions:
            action_counts[a] = action_counts.get(a, 0) + 1
        duplication_penalty = sum(
            (count - 1) * 0.3
            for count in action_counts.values()
            if count > 1
        )

        # Track global coverage for novelty bonus
        new_sources = set(actions) - self._global_queried
        coverage_bonus = len(new_sources) * 0.1
        self._global_queried.update(actions)

        # Apply shared adjustment to all agents
        adjusted_rewards = [
            r - duplication_penalty + coverage_bonus
            for r in individual_rewards
        ]

        info = {
            "results": results,
            "duplication_penalty": duplication_penalty,
            "coverage_bonus": coverage_bonus,
            "global_queried": list(self._global_queried),
            "dones": dones,
        }
        return obs_list, adjusted_rewards, dones, info

    def get_agent_messages(self) -> List[np.ndarray]:
        """
        Each agent broadcasts a summary of what it found this session.
        Used as the communication signal between agents.
        Returns a list of (n_sources,) relevance vectors.
        """
        messages = []
        for env in self.envs:
            session = env.get_session_state()
            if session is None:
                messages.append(np.zeros(self.n_sources, dtype=np.float32))
                continue
            vec = np.zeros(self.n_sources, dtype=np.float32)
            for r in session.results:
                name = r["source"]
                if name in env.get_source_names():
                    idx = env.get_source_names().index(name)
                    vec[idx] = max(vec[idx], r["relevance"])
            messages.append(vec)
        return messages
