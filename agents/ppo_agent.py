"""
agents/ppo_agent.py
METHOD 2 — Policy Gradient Methods
PPO for synthesis with analytical backpropagation (no PyTorch).
"""

import numpy as np
from typing import List, Tuple, Dict


class AnalyticalMLP:
    """Numpy MLP with stored activations for analytical backprop."""

    def __init__(self, layer_sizes: List[int], seed: int = 42):
        rng = np.random.default_rng(seed)
        self.weights, self.biases = [], []
        self.n_layers = len(layer_sizes) - 1
        for i in range(self.n_layers):
            fi, fo = layer_sizes[i], layer_sizes[i + 1]
            self.weights.append(rng.standard_normal((fi, fo)) * np.sqrt(2.0 / fi))
            self.biases.append(np.zeros(fo))
        self._pre, self._act = [], []

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = x.astype(np.float64).copy()
        self._act = [h.copy()]
        self._pre = []
        for i in range(self.n_layers):
            pre = h @ self.weights[i] + self.biases[i]
            self._pre.append(pre.copy())
            h = np.maximum(0, pre) if i < self.n_layers - 1 else pre
            self._act.append(h.copy())
        return h

    def backward(self, d_output: np.ndarray, lr: float):
        d = np.clip(d_output.copy(), -5.0, 5.0)
        for i in reversed(range(self.n_layers)):
            if i < self.n_layers - 1:
                d = d * (self._pre[i] > 0).astype(np.float64)
            dw = np.clip(np.outer(self._act[i], d), -5.0, 5.0)
            db = np.clip(d, -5.0, 5.0)
            self.weights[i] -= lr * dw
            self.biases[i] -= lr * db
            if i > 0:
                d = d @ self.weights[i].T


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / (np.sum(e) + 1e-8)


class PPOActor:
    def __init__(self, input_dim: int, n_sources: int, hidden_dim: int = 64):
        self.n_sources = n_sources
        self.net = AnalyticalMLP([input_dim, hidden_dim, hidden_dim, n_sources], seed=1)
        self.lr = 3e-4

    def forward(self, obs: np.ndarray) -> np.ndarray:
        logits = self.net.forward(obs)
        return softmax(logits)

    def sample_action(self, obs: np.ndarray) -> Tuple[np.ndarray, float]:
        probs = self.forward(obs)
        noise = np.random.dirichlet(np.ones(self.n_sources) * 0.3)
        weights = 0.85 * probs + 0.15 * noise
        weights /= weights.sum()
        log_prob = float(np.sum(weights * np.log(probs + 1e-8)))
        return weights, log_prob

    def update_step(self, obs, weights, advantage, old_log_prob):
        probs = self.forward(obs)
        log_prob = float(np.sum(weights * np.log(probs + 1e-8)))
        ratio = np.exp(log_prob - old_log_prob)
        clipped = np.clip(ratio, 0.8, 1.2)
        eff = ratio if ratio * advantage < clipped * advantage else clipped
        d_logits = weights - probs
        self.net.backward(-advantage * eff * d_logits, self.lr)


class PPOCritic:
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        self.net = AnalyticalMLP([input_dim, hidden_dim, hidden_dim, 1], seed=2)
        self.lr = 1e-3

    def forward(self, obs: np.ndarray) -> float:
        return float(self.net.forward(obs)[0])

    def update_step(self, obs, target):
        value = self.forward(obs)
        self.net.backward(np.array([2.0 * (value - target)]), self.lr)


class PPOBuffer:
    def __init__(self):
        self.observations, self.actions, self.rewards = [], [], []
        self.log_probs, self.values = [], []

    def add(self, obs, action, reward, log_prob, value):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_returns_and_advantages(self, gamma=0.99, lam=0.95):
        T = len(self.rewards)
        if T == 0:
            return np.array([]), np.array([])
        advantages = np.zeros(T)
        gae = 0.0
        for t in reversed(range(T)):
            nv = self.values[t + 1] if t + 1 < T else 0.0
            delta = self.rewards[t] + gamma * nv - self.values[t]
            gae = delta + gamma * lam * gae
            advantages[t] = gae
        returns = advantages + np.array(self.values)
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)


class PPOAgent:
    """
    Full PPO agent for synthesis decisions.
    Obs: concat(context_vector[11], source_relevances[12]) = 23-dim
    Action: synthesis weight vector over 12 sources
    """

    def __init__(self, context_dim=11, n_sources=12, hidden_dim=64, gamma=0.99, lam=0.95):
        obs_dim = context_dim + n_sources
        self.actor = PPOActor(obs_dim, n_sources, hidden_dim)
        self.critic = PPOCritic(obs_dim, hidden_dim)
        self.buffer = PPOBuffer()
        self.gamma, self.lam = gamma, lam
        self.n_sources, self.context_dim = n_sources, context_dim
        self.update_count = 0
        self.loss_history: List[float] = []

    def build_obs(self, context_vector, source_relevances):
        return np.concatenate([context_vector, source_relevances]).astype(np.float32)

    def select_action(self, context_vector, source_relevances):
        obs = self.build_obs(context_vector, source_relevances)
        weights, log_prob = self.actor.sample_action(obs)
        value = self.critic.forward(obs)
        return weights, log_prob, value

    def store_transition(self, context_vector, source_relevances, synthesis_weights, reward, log_prob, value):
        obs = self.build_obs(context_vector, source_relevances)
        self.buffer.add(obs, synthesis_weights, reward, log_prob, value)

    def update(self, n_epochs=4):
        if len(self.buffer) == 0:
            return
        returns, advantages = self.buffer.compute_returns_and_advantages(self.gamma, self.lam)
        if len(returns) == 0:
            self.buffer.clear()
            return
        for _ in range(n_epochs):
            for i in range(len(self.buffer)):
                self.actor.update_step(
                    self.buffer.observations[i], self.buffer.actions[i],
                    float(advantages[i]), float(self.buffer.log_probs[i])
                )
                self.critic.update_step(self.buffer.observations[i], float(returns[i]))
        self.loss_history.append(float(np.mean(advantages)))
        self.update_count += 1
        self.buffer.clear()

    def synthesize(self, results, source_names, context_vector):
        relevances = np.zeros(len(source_names), dtype=np.float32)
        for r in results:
            if r["available"] and r["source"] in source_names:
                idx = source_names.index(r["source"])
                relevances[idx] = max(relevances[idx], r["relevance"])
        weights, _, _ = self.select_action(context_vector, relevances)
        weighted_score = float(np.dot(weights, relevances))
        attribution = {
            name: float(w) for name, w in zip(source_names, weights)
            if relevances[source_names.index(name)] > 0.01
        }
        return {
            "weighted_insight_score": weighted_score,
            "synthesis_weights": weights,
            "source_attribution": attribution,
            "raw_relevances": relevances,
        }
