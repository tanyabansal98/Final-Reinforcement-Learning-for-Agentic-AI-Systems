"""
env/source_pool.py
Simulates 12 AI industry information sources.
Each source has a quality distribution that varies by query type.
Agents must learn these distributions purely from interaction.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


QUERY_TYPES = [
    "technical",    # architecture, benchmarks, capabilities
    "business",     # funding, valuation, corporate news
    "product",      # releases, launches, updates
    "regulatory",   # policy, compliance, law
    "market",       # competitive landscape, trends
]

@dataclass
class Source:
    """
    A simulated information source.
    quality_matrix[query_type] = (mean, std) of reward distribution.
    latency: seconds to respond (used in reward penalty).
    """
    name: str
    category: str
    quality_matrix: Dict[str, Tuple[float, float]]
    latency_mean: float
    latency_std: float
    query_count: int = 0

    def query(self, query_type: str, rng: np.random.Generator) -> Dict:
        """Simulate querying this source. Returns result dict."""
        self.query_count += 1
        mean, std = self.quality_matrix.get(query_type, (0.1, 0.05))
        relevance = float(np.clip(rng.normal(mean, std), 0.0, 1.0))
        latency = float(np.clip(rng.normal(self.latency_mean, self.latency_std), 0.1, 30.0))
        # Simulate occasionally returning nothing (source outage / no results)
        available = rng.random() > 0.05
        return {
            "source": self.name,
            "category": self.category,
            "query_type": query_type,
            "relevance": relevance if available else 0.0,
            "latency": latency,
            "available": available,
        }


def build_source_pool() -> List[Source]:
    """
    Returns the 12 simulated AI industry sources.
    Quality values are ground truth — agents never see these directly.
    """
    return [
        Source(
            name="arxiv",
            category="research",
            quality_matrix={
                "technical":   (0.92, 0.06),
                "business":    (0.05, 0.04),
                "product":     (0.30, 0.10),
                "regulatory":  (0.08, 0.05),
                "market":      (0.10, 0.06),
            },
            latency_mean=2.0, latency_std=0.5,
        ),
        Source(
            name="semantic_scholar",
            category="research",
            quality_matrix={
                "technical":   (0.85, 0.07),
                "business":    (0.04, 0.03),
                "product":     (0.25, 0.09),
                "regulatory":  (0.06, 0.04),
                "market":      (0.08, 0.05),
            },
            latency_mean=1.5, latency_std=0.4,
        ),
        Source(
            name="sec_edgar",
            category="financial",
            quality_matrix={
                "technical":   (0.03, 0.02),
                "business":    (0.95, 0.04),
                "product":     (0.10, 0.06),
                "regulatory":  (0.60, 0.10),
                "market":      (0.80, 0.08),
            },
            latency_mean=3.0, latency_std=0.8,
        ),
        Source(
            name="conference_proceedings",
            category="research",
            quality_matrix={
                "technical":   (0.90, 0.05),
                "business":    (0.03, 0.02),
                "product":     (0.20, 0.08),
                "regulatory":  (0.04, 0.03),
                "market":      (0.07, 0.04),
            },
            latency_mean=4.0, latency_std=1.0,
        ),
        Source(
            name="tech_news",
            category="media",
            quality_matrix={
                "technical":   (0.45, 0.12),
                "business":    (0.75, 0.10),
                "product":     (0.88, 0.07),
                "regulatory":  (0.50, 0.12),
                "market":      (0.70, 0.10),
            },
            latency_mean=0.5, latency_std=0.2,
        ),
        Source(
            name="github_feeds",
            category="code",
            quality_matrix={
                "technical":   (0.78, 0.09),
                "business":    (0.10, 0.06),
                "product":     (0.82, 0.08),
                "regulatory":  (0.02, 0.02),
                "market":      (0.20, 0.08),
            },
            latency_mean=0.8, latency_std=0.3,
        ),
        Source(
            name="crunchbase",
            category="investment",
            quality_matrix={
                "technical":   (0.04, 0.03),
                "business":    (0.85, 0.08),
                "product":     (0.30, 0.10),
                "regulatory":  (0.15, 0.07),
                "market":      (0.75, 0.09),
            },
            latency_mean=1.2, latency_std=0.4,
        ),
        Source(
            name="social_media",
            category="social",
            quality_matrix={
                "technical":   (0.25, 0.18),
                "business":    (0.40, 0.20),
                "product":     (0.70, 0.18),
                "regulatory":  (0.15, 0.14),
                "market":      (0.35, 0.18),
            },
            latency_mean=0.1, latency_std=0.05,
        ),
        Source(
            name="ai_newsletters",
            category="commentary",
            quality_matrix={
                "technical":   (0.55, 0.14),
                "business":    (0.45, 0.14),
                "product":     (0.65, 0.12),
                "regulatory":  (0.35, 0.13),
                "market":      (0.55, 0.13),
            },
            latency_mean=1.0, latency_std=0.3,
        ),
        Source(
            name="government_policy",
            category="regulatory",
            quality_matrix={
                "technical":   (0.05, 0.03),
                "business":    (0.20, 0.08),
                "product":     (0.08, 0.05),
                "regulatory":  (0.96, 0.03),
                "market":      (0.30, 0.10),
            },
            latency_mean=5.0, latency_std=1.5,
        ),
        Source(
            name="patent_databases",
            category="ip",
            quality_matrix={
                "technical":   (0.70, 0.10),
                "business":    (0.25, 0.09),
                "product":     (0.40, 0.11),
                "regulatory":  (0.30, 0.09),
                "market":      (0.45, 0.10),
            },
            latency_mean=3.5, latency_std=1.0,
        ),
        Source(
            name="analyst_reports",
            category="financial",
            quality_matrix={
                "technical":   (0.40, 0.12),
                "business":    (0.80, 0.09),
                "product":     (0.50, 0.12),
                "regulatory":  (0.45, 0.11),
                "market":      (0.88, 0.07),
            },
            latency_mean=2.5, latency_std=0.7,
        ),
    ]


class SourcePool:
    """
    Manages the full pool of sources.
    Provides batch querying and statistics tracking.
    """
    def __init__(self, seed: int = 42):
        self.sources = build_source_pool()
        self.source_names = [s.name for s in self.sources]
        self.n_sources = len(self.sources)
        self.rng = np.random.default_rng(seed)
        self._query_history: List[Dict] = []

    def query(self, source_idx: int, query_type: str) -> Dict:
        """Query a single source by index."""
        result = self.sources[source_idx].query(query_type, self.rng)
        self._query_history.append(result)
        return result

    def query_batch(self, source_indices: List[int], query_type: str) -> List[Dict]:
        """Query multiple sources (used by MARL agents)."""
        return [self.query(i, query_type) for i in source_indices]

    def get_ground_truth_matrix(self) -> np.ndarray:
        """
        Returns true mean quality for each (source, query_type) pair.
        Only used for analysis/plotting — never passed to agents during training.
        Shape: (n_sources, n_query_types)
        """
        matrix = np.zeros((self.n_sources, len(QUERY_TYPES)))
        for i, source in enumerate(self.sources):
            for j, qt in enumerate(QUERY_TYPES):
                mean, _ = source.quality_matrix.get(qt, (0.0, 0.0))
                matrix[i, j] = mean
        return matrix

    def reset_counts(self):
        """Reset query counters (call between training runs)."""
        for s in self.sources:
            s.query_count = 0
        self._query_history = []

    def get_stats(self) -> Dict:
        """Return per-source query counts."""
        return {s.name: s.query_count for s in self.sources}
