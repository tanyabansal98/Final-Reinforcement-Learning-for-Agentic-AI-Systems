"""
env/query_engine.py
Handles query generation and context vector encoding.
The context vector is the input to all RL agents — it encodes
what kind of research task the agent is currently working on.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from env.source_pool import QUERY_TYPES


# Context vector dimension breakdown:
# [0:5]   query type one-hot (technical, business, product, regulatory, market)
# [5:8]   urgency encoding (low, medium, high) — one-hot
# [6]     budget remaining (normalized 0-1)  -- reuse index carefully
# Context vector total dim = 5 + 3 + 1 + 1 + 1 = 11
# [0:5]   query type one-hot
# [5:8]   urgency one-hot
# [8]     budget_remaining (0-1)
# [9]     sources_queried_ratio (0-1)
# [10]    has_conflict (0 or 1) — did two sources return contradictory signals?
CONTEXT_DIM = 11

URGENCY_LEVELS = ["low", "medium", "high"]

EXAMPLE_QUERIES = {
    "technical": [
        "What are state-of-the-art coding benchmark results for LLMs?",
        "How does mixture-of-experts architecture improve efficiency?",
        "What are the latest findings on LLM reasoning capabilities?",
        "Compare transformer vs SSM architectures for long context.",
    ],
    "business": [
        "What is Anthropic's current funding status and valuation?",
        "Who are the major investors in AI infrastructure companies?",
        "What is OpenAI's revenue trajectory and business model?",
        "Which AI startups raised Series B or later in the last quarter?",
    ],
    "product": [
        "What models has Mistral released in the last three months?",
        "What are the key features of the latest Claude release?",
        "Has Google updated Gemini's context window recently?",
        "What open-source models have been released this week?",
    ],
    "regulatory": [
        "What does the EU AI Act require for foundation model providers?",
        "What are NIST AI RMF guidelines for high-risk AI systems?",
        "Has the US government issued new AI executive orders recently?",
        "What are current export controls on AI chips?",
    ],
    "market": [
        "How is GPU demand trending across hyperscalers?",
        "What is the competitive landscape for AI coding assistants?",
        "How does Microsoft's AI infrastructure spending compare to Google?",
        "Which companies are gaining or losing AI market share?",
    ],
}


@dataclass
class Query:
    """Represents a single research query for the Madison agent."""
    query_id: int
    text: str
    query_type: str
    urgency: str
    budget: int               # max number of source queries allowed
    domain: str = "general"   # used by meta-learner for domain identification

    def to_context_vector(
        self,
        budget_remaining: int,
        sources_queried: int,
        has_conflict: bool,
    ) -> np.ndarray:
        """
        Encode current query state into a fixed-length context vector.
        This is the observation passed to all RL agents.
        """
        ctx = np.zeros(CONTEXT_DIM, dtype=np.float32)

        # Query type one-hot [0:5]
        qt_idx = QUERY_TYPES.index(self.query_type)
        ctx[qt_idx] = 1.0

        # Urgency one-hot [5:8]
        urg_idx = URGENCY_LEVELS.index(self.urgency)
        ctx[5 + urg_idx] = 1.0

        # Continuous features [8:11]
        ctx[8] = budget_remaining / max(self.budget, 1)
        ctx[9] = min(sources_queried / 12.0, 1.0)  # normalize by pool size
        ctx[10] = float(has_conflict)

        return ctx


class QueryEngine:
    """
    Generates research queries and tracks session state.
    Each episode = one research session = one query with a budget of K source queries.
    """
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self._query_counter = 0

    def sample_query(
        self,
        query_type: Optional[str] = None,
        urgency: Optional[str] = None,
        domain: str = "general",
    ) -> Query:
        """Sample a new research query, optionally fixing type/urgency."""
        if query_type is None:
            query_type = self.rng.choice(QUERY_TYPES)
        if urgency is None:
            urgency = self.rng.choice(URGENCY_LEVELS, p=[0.3, 0.5, 0.2])

        text = self.rng.choice(EXAMPLE_QUERIES[query_type])
        # Budget scales with urgency: urgent queries get fewer tries
        budget_map = {"low": 8, "medium": 5, "high": 3}
        budget = budget_map[urgency]

        self._query_counter += 1
        return Query(
            query_id=self._query_counter,
            text=text,
            query_type=query_type,
            urgency=urgency,
            budget=budget,
            domain=domain,
        )

    def sample_domain_batch(self, domain: str, n: int = 10) -> List[Query]:
        """Sample a batch of queries from a specific domain (used by meta-learner)."""
        domain_type_map = {
            "tech_research": "technical",
            "ai_finance":    "business",
            "open_source":   "product",
            "ai_policy":     "regulatory",
            "ai_market":     "market",
        }
        qt = domain_type_map.get(domain, None)
        return [self.sample_query(query_type=qt, domain=domain) for _ in range(n)]


@dataclass
class SessionState:
    """
    Tracks the state of a single research session (episode).
    Updated after each source query. Passed to Q-Learning agent.
    """
    query: Query
    results: List[Dict] = field(default_factory=list)
    sources_queried: List[int] = field(default_factory=list)
    budget_used: int = 0
    has_conflict: bool = False

    @property
    def budget_remaining(self) -> int:
        return self.query.budget - self.budget_used

    @property
    def is_done(self) -> bool:
        return self.budget_remaining <= 0

    def add_result(self, source_idx: int, result: Dict):
        """Record a query result and update session state."""
        self.results.append(result)
        self.sources_queried.append(source_idx)
        self.budget_used += 1
        self._check_conflict()

    def _check_conflict(self):
        """Flag if two sources returned significantly different relevance scores."""
        if len(self.results) < 2:
            return
        scores = [r["relevance"] for r in self.results if r["available"]]
        if len(scores) >= 2 and (max(scores) - min(scores)) > 0.5:
            self.has_conflict = True

    def to_context_vector(self) -> np.ndarray:
        return self.query.to_context_vector(
            budget_remaining=self.budget_remaining,
            sources_queried=len(self.sources_queried),
            has_conflict=self.has_conflict,
        )

    def get_collected_relevances(self) -> np.ndarray:
        """Per-source relevance collected so far (for PPO synthesis agent)."""
        from env.source_pool import SourcePool
        scores = np.zeros(12, dtype=np.float32)
        for r in self.results:
            # find source index by name
            pass  # filled in madison_env.py with pool reference
        return scores

    def summary(self) -> Dict:
        """Summary stats for reward computation."""
        available = [r for r in self.results if r["available"]]
        return {
            "n_queried": len(self.results),
            "n_available": len(available),
            "avg_relevance": np.mean([r["relevance"] for r in available]) if available else 0.0,
            "max_relevance": max([r["relevance"] for r in available], default=0.0),
            "avg_latency": np.mean([r["latency"] for r in self.results]) if self.results else 0.0,
            "has_conflict": self.has_conflict,
            "budget_used": self.budget_used,
            "query_type": self.query.query_type,
            "urgency": self.query.urgency,
        }
