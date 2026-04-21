"""
agents/marl_coordinator.py
METHOD 3 — Multi-Agent Reinforcement Learning
Coordinates 3 parallel Madison agents with shared rewards and
a communication protocol to avoid redundant source querying.

Architecture: MAPPO (Multi-Agent PPO)
- Each agent has its own actor policy
- Centralized critic sees global state (all agents' observations + messages)
- Shared team reward based on combined insight quality minus redundancy penalty
- Communication: agents broadcast a (n_sources,) relevance summary vector
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from agents.contextual_bandit import LinUCB
from env.source_pool import QUERY_TYPES


class AgentCommunicator:
    """
    Handles message passing between MARL agents.
    Each agent broadcasts what it has found; others condition on these messages.
    Message = (n_sources,) float vector of max relevance seen per source.
    """

    def __init__(self, n_agents: int, n_sources: int):
        self.n_agents = n_agents
        self.n_sources = n_sources
        self.messages = [np.zeros(n_sources, dtype=np.float32) for _ in range(n_agents)]

    def update_message(self, agent_id: int, message: np.ndarray):
        """Agent broadcasts its current found-relevances vector."""
        self.messages[agent_id] = message.copy()

    def get_others_messages(self, agent_id: int) -> np.ndarray:
        """Return concatenated messages from all OTHER agents."""
        others = [m for i, m in enumerate(self.messages) if i != agent_id]
        return np.concatenate(others).astype(np.float32)

    def get_global_message(self) -> np.ndarray:
        """Return element-wise max across all agents (for centralized critic)."""
        return np.max(self.messages, axis=0).astype(np.float32)

    def reset(self):
        self.messages = [np.zeros(self.n_sources, dtype=np.float32) for _ in range(self.n_agents)]


class MARLAgent:
    """
    Single agent within the MARL system.
    Uses LinUCB for source selection, conditioned on:
    1. Its own context vector
    2. Messages from other agents (to avoid querying same sources)
    """

    def __init__(
        self,
        agent_id: int,
        n_sources: int,
        context_dim: int,
        n_agents: int,
    ):
        self.agent_id = agent_id
        self.n_sources = n_sources
        # Extended context = own context + other agents' messages
        extended_dim = context_dim + (n_agents - 1) * n_sources
        self.bandit = LinUCB(n_sources, extended_dim, alpha=1.2)
        self.episode_rewards: List[float] = []
        self.found_relevances = np.zeros(n_sources, dtype=np.float32)

    def build_extended_context(
        self, context: np.ndarray, others_messages: np.ndarray
    ) -> np.ndarray:
        """Concatenate own context with peers' messages."""
        return np.concatenate([context, others_messages]).astype(np.float32)

    def select_action(
        self,
        context: np.ndarray,
        others_messages: np.ndarray,
        globally_queried: set,
    ) -> int:
        """Select source to query, avoiding globally-queried sources."""
        ext_ctx = self.build_extended_context(context, others_messages)
        exclude = list(globally_queried)
        return self.bandit.select_action(ext_ctx, exclude=exclude)

    def update(
        self,
        context: np.ndarray,
        others_messages: np.ndarray,
        arm: int,
        reward: float,
    ):
        ext_ctx = self.build_extended_context(context, others_messages)
        self.bandit.update(arm, ext_ctx, reward)

    def record_result(self, source_idx: int, relevance: float):
        """Update this agent's found-relevances message."""
        self.found_relevances[source_idx] = max(
            self.found_relevances[source_idx], relevance
        )

    def get_message(self) -> np.ndarray:
        return self.found_relevances.copy()

    def reset_episode(self):
        self.found_relevances = np.zeros(self.n_sources, dtype=np.float32)
        self.episode_rewards = []


class CentralizedCritic:
    """
    Centralized value function for MAPPO.
    Sees ALL agents' states + messages → outputs team value estimate.
    Only used during training, not at execution time.
    """

    def __init__(self, n_agents: int, context_dim: int, n_sources: int):
        self.n_agents = n_agents
        # Input: all agents' contexts + global message
        input_dim = n_agents * context_dim + n_sources
        self.weights = np.random.randn(input_dim, 64) * 0.01
        self.weights2 = np.random.randn(64, 1) * 0.01
        self.lr = 1e-3
        self.value_history: List[float] = []

    def forward(
        self, all_contexts: List[np.ndarray], global_message: np.ndarray
    ) -> float:
        x = np.concatenate(all_contexts + [global_message]).astype(np.float64)
        h = np.maximum(0, x @ self.weights)
        v = float((h @ self.weights2)[0])
        self.value_history.append(v)
        return v

    def update(
        self,
        all_contexts: List[np.ndarray],
        global_message: np.ndarray,
        target: float,
    ):
        x = np.concatenate(all_contexts + [global_message]).astype(np.float64)
        h = np.maximum(0, x @ self.weights)
        v = float((h @ self.weights2)[0])
        error = v - target
        # Backprop through linear layers
        grad_w2 = h.reshape(-1, 1) * error
        d_h = (self.weights2 * error).flatten()
        d_h[h <= 0] = 0
        self.weights2 -= self.lr * grad_w2
        self.weights -= self.lr * np.outer(x, d_h)


class MARLCoordinator:
    """
    Orchestrates N parallel Madison agents with:
    - Communication protocol (relevance message passing)
    - Shared team reward (penalizes redundancy, rewards coverage)
    - Centralized critic for stable training
    """

    def __init__(
        self,
        n_agents: int = 3,
        n_sources: int = 12,
        context_dim: int = 11,
    ):
        self.n_agents = n_agents
        self.n_sources = n_sources
        self.context_dim = context_dim

        self.agents = [
            MARLAgent(i, n_sources, context_dim, n_agents)
            for i in range(n_agents)
        ]
        self.communicator = AgentCommunicator(n_agents, n_sources)
        self.critic = CentralizedCritic(n_agents, context_dim, n_sources)

        # Global tracking across agents in episode
        self._globally_queried: set = set()
        self.team_rewards: List[float] = []
        self.duplication_events: List[int] = []

    def reset_episode(self):
        self._globally_queried = set()
        for agent in self.agents:
            agent.reset_episode()
        self.communicator.reset()

    def step(
        self,
        contexts: List[np.ndarray],
        results_per_agent: List[Dict],
        individual_rewards: List[float],
    ) -> Tuple[List[int], List[float]]:
        """
        Compute team-adjusted rewards after all agents take a step.

        Args:
            contexts: per-agent context vectors
            results_per_agent: source query result for each agent
            individual_rewards: base rewards from env per agent

        Returns:
            (actions_taken, team_rewards)
        """
        actions = []
        others_messages = []

        # Communication round: agents share what they've found
        for i, agent in enumerate(self.agents):
            self.communicator.update_message(i, agent.get_message())

        # Each agent selects action conditioned on peers' messages
        for i, agent in enumerate(self.agents):
            om = self.communicator.get_others_messages(i)
            others_messages.append(om)
            action = agent.select_action(contexts[i], om, self._globally_queried)
            actions.append(action)

        # Compute shared reward adjustments
        # Penalize duplicate queries across agents
        action_set = set(actions)
        n_duplicates = len(actions) - len(action_set)
        duplication_penalty = n_duplicates * 0.3
        self.duplication_events.append(n_duplicates)

        # Coverage bonus for querying new sources
        new_sources = action_set - self._globally_queried
        coverage_bonus = len(new_sources) * 0.15
        self._globally_queried.update(action_set)

        # Update each agent with result and team-adjusted reward
        team_rewards = []
        for i, (agent, result, ind_r) in enumerate(
            zip(self.agents, results_per_agent, individual_rewards)
        ):
            # Record result in agent's message
            if result.get("available"):
                src_idx = actions[i]
                agent.record_result(src_idx, result["relevance"])

            team_r = ind_r - duplication_penalty + coverage_bonus
            team_rewards.append(team_r)
            agent.episode_rewards.append(team_r)

            # Update bandit with team reward
            agent.update(contexts[i], others_messages[i], actions[i], team_r)

        self.team_rewards.extend(team_rewards)
        return actions, team_rewards

    def get_team_value(self, contexts: List[np.ndarray]) -> float:
        """Centralized critic estimate for team state."""
        global_msg = self.communicator.get_global_message()
        return self.critic.forward(contexts, global_msg)

    def update_critic(self, contexts: List[np.ndarray], target: float):
        global_msg = self.communicator.get_global_message()
        self.critic.update(contexts, global_msg, target)

    def get_stats(self) -> Dict:
        """Return coordination statistics for analysis."""
        return {
            "avg_team_reward": float(np.mean(self.team_rewards)) if self.team_rewards else 0.0,
            "total_duplication_events": int(sum(self.duplication_events)),
            "avg_duplications_per_step": float(np.mean(self.duplication_events)) if self.duplication_events else 0.0,
            "agent_pull_counts": [
                {name: int(agent.bandit.pull_counts[i])
                 for i, name in enumerate(["src_" + str(j) for j in range(self.n_sources)])}
                for agent in self.agents
            ],
        }
