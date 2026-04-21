"""
agents/q_learning.py
METHOD 1 — Value-Based Learning
Implements tabular Q-Learning and SARSA for session-level source sequencing.

While the contextual bandit decides WHICH category of source to query,
Q-Learning decides the SEQUENCE: given what has already been collected
this session, which specific source maximizes expected future reward?

State space: discretized session state (query type × budget_bucket × coverage_bucket)
Action space: source index (0..11)
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from env.source_pool import QUERY_TYPES


# State discretization
N_BUDGET_BUCKETS = 4     # [full, 2/3, 1/3, almost_done]
N_COVERAGE_BUCKETS = 3   # [low, medium, high] — fraction of sources queried
N_QUERY_TYPES = len(QUERY_TYPES)
STATE_DIM = N_QUERY_TYPES * N_BUDGET_BUCKETS * N_COVERAGE_BUCKETS


def encode_state(
    query_type: str,
    budget_remaining: int,
    budget_total: int,
    n_queried: int,
    n_sources: int = 12,
) -> int:
    """
    Map session state to a single integer index for the Q-table.
    """
    qt_idx = QUERY_TYPES.index(query_type)

    budget_ratio = budget_remaining / max(budget_total, 1)
    if budget_ratio > 0.66:
        budget_bucket = 0
    elif budget_ratio > 0.33:
        budget_bucket = 1
    elif budget_ratio > 0.1:
        budget_bucket = 2
    else:
        budget_bucket = 3

    coverage_ratio = n_queried / n_sources
    if coverage_ratio < 0.25:
        coverage_bucket = 0
    elif coverage_ratio < 0.6:
        coverage_bucket = 1
    else:
        coverage_bucket = 2

    state = (
        qt_idx * (N_BUDGET_BUCKETS * N_COVERAGE_BUCKETS)
        + budget_bucket * N_COVERAGE_BUCKETS
        + coverage_bucket
    )
    return state


class QLearning:
    """
    Tabular Q-Learning agent.

    Update rule (off-policy):
    Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]

    Off-policy: learns the greedy policy regardless of what action
    was actually taken (safer for exploration with ε-greedy).
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: (n_states, n_actions) initialized optimistically
        self.Q = np.ones((n_states, n_actions)) * 0.5

        # Tracking
        self.update_count = 0
        self.td_errors: List[float] = []

    def select_action(
        self,
        state: int,
        exclude: Optional[List[int]] = None,
        greedy: bool = False,
    ) -> int:
        """ε-greedy action selection. Exclude already-queried sources."""
        available = [a for a in range(self.n_actions) if not (exclude and a in exclude)]
        if not available:
            return 0

        if not greedy and np.random.random() < self.epsilon:
            return int(np.random.choice(available))

        q_vals = self.Q[state].copy()
        if exclude:
            for a in exclude:
                q_vals[a] = -np.inf
        return int(np.argmax(q_vals))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        exclude_next: Optional[List[int]] = None,
    ):
        """Q-Learning (off-policy) Bellman update."""
        if done:
            target = reward
        else:
            next_q = self.Q[next_state].copy()
            if exclude_next:
                for a in exclude_next:
                    next_q[a] = -np.inf
            target = reward + self.gamma * np.max(next_q)

        td_error = target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

        self.td_errors.append(abs(td_error))
        self.update_count += 1
        self._decay_epsilon()

    def _decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self) -> np.ndarray:
        """Return greedy action for each state. Shape: (n_states,)"""
        return np.argmax(self.Q, axis=1)


class SARSA:
    """
    SARSA (State-Action-Reward-State-Action) — on-policy variant.

    Update rule:
    Q(s,a) ← Q(s,a) + α [r + γ Q(s', a') - Q(s,a)]

    Where a' is the ACTUAL next action (not the greedy max).
    More conservative than Q-Learning — tends to learn safer policies
    when exploration is part of the evaluation.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.Q = np.ones((n_states, n_actions)) * 0.5
        self.update_count = 0
        self.td_errors: List[float] = []

        # SARSA needs to track the next action ahead of time
        self._next_action: Optional[int] = None

    def select_action(
        self,
        state: int,
        exclude: Optional[List[int]] = None,
        greedy: bool = False,
    ) -> int:
        available = [a for a in range(self.n_actions) if not (exclude and a in exclude)]
        if not available:
            return 0
        if not greedy and np.random.random() < self.epsilon:
            action = int(np.random.choice(available))
        else:
            q_vals = self.Q[state].copy()
            if exclude:
                for a in exclude:
                    q_vals[a] = -np.inf
            action = int(np.argmax(q_vals))
        self._next_action = action
        return action

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: int,
        done: bool,
    ):
        """SARSA on-policy update using actual next action."""
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.Q[next_state, next_action]

        td_error = target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

        self.td_errors.append(abs(td_error))
        self.update_count += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * 0.995)

    def get_policy(self) -> np.ndarray:
        return np.argmax(self.Q, axis=1)


class ValueBasedAgent:
    """
    Wrapper running both Q-Learning and SARSA in parallel.
    Q-Learning drives actual decisions; SARSA runs shadow updates
    for comparison in analysis.
    """

    def __init__(self, n_sources: int):
        self.n_sources = n_sources
        self.n_states = STATE_DIM

        self.q_learning = QLearning(
            n_states=self.n_states,
            n_actions=n_sources,
            alpha=0.1, gamma=0.95,
            epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995,
        )
        self.sarsa = SARSA(
            n_states=self.n_states,
            n_actions=n_sources,
            alpha=0.1, gamma=0.95,
            epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995,
        )

    def encode(
        self,
        query_type: str,
        budget_remaining: int,
        budget_total: int,
        n_queried: int,
    ) -> int:
        return encode_state(query_type, budget_remaining, budget_total, n_queried)

    def select_action(
        self,
        state: int,
        exclude: Optional[List[int]] = None,
        use: str = "q_learning",
    ) -> int:
        if use == "sarsa":
            return self.sarsa.select_action(state, exclude=exclude)
        return self.q_learning.select_action(state, exclude=exclude)

    def update_q(
        self, state, action, reward, next_state, done, exclude_next=None
    ):
        self.q_learning.update(state, action, reward, next_state, done, exclude_next)

    def update_sarsa(
        self, state, action, reward, next_state, next_action, done
    ):
        self.sarsa.update(state, action, reward, next_state, next_action, done)

    def get_q_table(self) -> np.ndarray:
        return self.q_learning.Q

    def get_td_errors(self) -> Dict[str, List[float]]:
        return {
            "q_learning": self.q_learning.td_errors,
            "sarsa": self.sarsa.td_errors,
        }
