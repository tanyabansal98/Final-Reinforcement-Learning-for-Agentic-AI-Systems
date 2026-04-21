"""
tools/source_credibility_scorer.py
CUSTOM TOOL — Source Credibility Scorer

A dynamic credibility scoring tool that maintains a trust model of each
information source based on historical accuracy, consistency, and timeliness.
Integrates with the Madison RL pipeline to provide agents with an additional
signal beyond raw relevance.

This tool solves a real-world problem: not all high-relevance results are
trustworthy. A clickbait article may seem relevant but be low-credibility,
while a slow-updating government database is highly credible but often stale.

The credibility score is used to:
1. Adjust the reward function (credibility-weighted relevance)
2. Filter out unreliable sources dynamically
3. Provide explainable trust metrics for transparency

Run standalone demo:
    python tools/source_credibility_scorer.py
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class SourceCredibilityProfile:
    """Tracks credibility metrics for a single source."""
    name: str
    # Accuracy: how often does this source's relevance match ground truth expectations?
    accuracy_history: List[float] = field(default_factory=list)
    # Consistency: how stable is the source's quality over time?
    consistency_history: List[float] = field(default_factory=list)
    # Timeliness: does the source respond quickly?
    latency_history: List[float] = field(default_factory=list)
    # Availability: how often is the source actually available?
    availability_history: List[bool] = field(default_factory=list)
    # Conflict rate: how often does this source contradict others?
    conflict_count: int = 0
    total_queries: int = 0

    @property
    def accuracy_score(self) -> float:
        """Mean relevance quality, decayed toward recent observations."""
        if not self.accuracy_history:
            return 0.5  # prior
        # Exponential recency weighting
        n = len(self.accuracy_history)
        weights = np.exp(np.linspace(-2, 0, n))
        weights /= weights.sum()
        return float(np.dot(weights, self.accuracy_history))

    @property
    def consistency_score(self) -> float:
        """Inverse of relevance variance — consistent sources score higher."""
        if len(self.accuracy_history) < 3:
            return 0.5
        recent = self.accuracy_history[-20:]
        variance = np.var(recent)
        # Map variance to [0, 1]: low variance → high consistency
        return float(np.clip(1.0 - variance * 4, 0.0, 1.0))

    @property
    def timeliness_score(self) -> float:
        """Score based on response latency — fast sources score higher."""
        if not self.latency_history:
            return 0.5
        recent = self.latency_history[-20:]
        mean_latency = np.mean(recent)
        # Map latency to [0, 1]: <1s → 1.0, >10s → 0.0
        return float(np.clip(1.0 - mean_latency / 10.0, 0.0, 1.0))

    @property
    def availability_score(self) -> float:
        """Fraction of time the source was available."""
        if not self.availability_history:
            return 0.95
        recent = self.availability_history[-50:]
        return float(np.mean(recent))

    @property
    def conflict_rate(self) -> float:
        """Fraction of queries where this source contradicted peers."""
        if self.total_queries == 0:
            return 0.0
        return self.conflict_count / self.total_queries

    def overall_credibility(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Weighted combination of all credibility factors.
        Returns score in [0, 1].
        """
        if weights is None:
            weights = {
                "accuracy":      0.40,
                "consistency":   0.25,
                "timeliness":    0.15,
                "availability":  0.15,
                "conflict":      0.05,
            }
        score = (
            weights["accuracy"]     * self.accuracy_score
            + weights["consistency"]  * self.consistency_score
            + weights["timeliness"]   * self.timeliness_score
            + weights["availability"] * self.availability_score
            - weights["conflict"]     * self.conflict_rate
        )
        return float(np.clip(score, 0.0, 1.0))


class SourceCredibilityScorer:
    """
    Main tool: maintains credibility profiles for all sources
    and provides trust-adjusted scoring to the RL pipeline.

    Integration points:
    1. Called after each source query to update profiles
    2. Queried before action selection to adjust reward expectations
    3. Provides credibility-weighted relevance for PPO synthesis

    Usage:
        scorer = SourceCredibilityScorer(source_names)
        scorer.record_query(source_name, result_dict, session_results)
        credibility = scorer.get_credibility(source_name)
        adjusted_reward = scorer.credibility_weighted_reward(result_dict)
        report = scorer.generate_trust_report()
    """

    def __init__(self, source_names: List[str]):
        self.source_names = source_names
        self.profiles: Dict[str, SourceCredibilityProfile] = {
            name: SourceCredibilityProfile(name=name) for name in source_names
        }
        self._global_query_count = 0

    # ── Core API ───────────────────────────────────────────────────────────

    def record_query(
        self,
        source_name: str,
        result: Dict,
        session_results: Optional[List[Dict]] = None,
    ):
        """
        Update credibility profile after a query result is received.
        Called by the training loop after each env.step().
        """
        if source_name not in self.profiles:
            return

        profile = self.profiles[source_name]
        profile.total_queries += 1
        self._global_query_count += 1

        # Record accuracy (relevance quality)
        if result["available"]:
            profile.accuracy_history.append(result["relevance"])
        else:
            profile.accuracy_history.append(0.0)

        # Record latency
        profile.latency_history.append(result["latency"])

        # Record availability
        profile.availability_history.append(result["available"])

        # Check for conflict with other sources this session
        if session_results and len(session_results) >= 2 and result["available"]:
            other_scores = [
                r["relevance"] for r in session_results[:-1]
                if r["available"] and r["source"] != source_name
            ]
            if other_scores:
                # Conflict = large disagreement in relevance
                max_diff = max(abs(result["relevance"] - s) for s in other_scores)
                if max_diff > 0.5:
                    profile.conflict_count += 1

    def get_credibility(self, source_name: str) -> float:
        """Get overall credibility score for a source (0 to 1)."""
        if source_name not in self.profiles:
            return 0.5
        return self.profiles[source_name].overall_credibility()

    def get_all_credibilities(self) -> Dict[str, float]:
        """Get credibility scores for all sources."""
        return {name: p.overall_credibility() for name, p in self.profiles.items()}

    def credibility_weighted_reward(self, result: Dict) -> float:
        """
        Compute credibility-adjusted reward.
        High-relevance from a low-credibility source is discounted.
        Used as an alternative reward signal for the bandit/Q-Learning.
        """
        if not result["available"]:
            return -0.1

        source = result["source"]
        cred = self.get_credibility(source)
        raw_relevance = result["relevance"]

        # Credibility-weighted relevance:
        # If credibility is high (>0.7), boost reward slightly
        # If credibility is low (<0.3), discount reward significantly
        adjusted = raw_relevance * (0.5 + 0.5 * cred)
        return float(adjusted)

    def get_credibility_vector(self) -> np.ndarray:
        """
        Return (n_sources,) vector of credibility scores.
        Can be appended to context vector for credibility-aware action selection.
        """
        return np.array(
            [self.profiles[name].overall_credibility() for name in self.source_names],
            dtype=np.float32,
        )

    def get_trust_adjusted_mask(self, threshold: float = 0.2) -> List[int]:
        """
        Return list of source indices that fall below credibility threshold.
        Agents can exclude these from their action space.
        """
        return [
            i for i, name in enumerate(self.source_names)
            if self.profiles[name].overall_credibility() < threshold
        ]

    # ── Reporting ──────────────────────────────────────────────────────────

    def generate_trust_report(self) -> Dict:
        """
        Generate a comprehensive trust report for all sources.
        Used in the analysis section and demo video.
        """
        report = {}
        for name, profile in self.profiles.items():
            report[name] = {
                "overall_credibility": round(profile.overall_credibility(), 4),
                "accuracy":            round(profile.accuracy_score, 4),
                "consistency":         round(profile.consistency_score, 4),
                "timeliness":          round(profile.timeliness_score, 4),
                "availability":        round(profile.availability_score, 4),
                "conflict_rate":       round(profile.conflict_rate, 4),
                "total_queries":       profile.total_queries,
            }
        return report

    def print_trust_report(self):
        """Pretty-print the trust report."""
        report = self.generate_trust_report()

        print(f"\n{'='*85}")
        print(f"  SOURCE CREDIBILITY REPORT  ({self._global_query_count} total queries)")
        print(f"{'='*85}")
        print(f"{'Source':<24} {'Credibility':>12} {'Accuracy':>10} {'Consistency':>12} "
              f"{'Timeliness':>11} {'Avail':>7} {'Conflicts':>10}")
        print(f"{'─'*85}")

        # Sort by credibility
        sorted_sources = sorted(report.items(), key=lambda x: x[1]["overall_credibility"], reverse=True)
        for name, metrics in sorted_sources:
            cred = metrics["overall_credibility"]
            trust_bar = "█" * int(cred * 10) + "░" * (10 - int(cred * 10))
            print(f"{name:<24} {trust_bar} {cred:>.3f}  {metrics['accuracy']:>8.3f}  "
                  f"{metrics['consistency']:>10.3f}  {metrics['timeliness']:>9.3f}  "
                  f"{metrics['availability']:>5.2f}  {metrics['conflict_rate']:>8.3f}")

        print(f"{'='*85}")

    def plot_credibility_dashboard(self, save_path: str = None):
        """Generate a visual credibility dashboard."""
        import matplotlib.pyplot as plt

        report = self.generate_trust_report()
        names = list(report.keys())
        metrics = ["accuracy", "consistency", "timeliness", "availability"]
        colors = ["#185FA5", "#1D9E75", "#BA7517", "#534AB7"]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Left: Overall credibility bar chart
        ax1 = axes[0]
        creds = [report[n]["overall_credibility"] for n in names]
        bar_colors = ["#1D9E75" if c > 0.6 else "#BA7517" if c > 0.4 else "#D85A30" for c in creds]
        bars = ax1.barh(names, creds, color=bar_colors, edgecolor="white")
        ax1.set_xlabel("Overall Credibility Score")
        ax1.set_title("Source Credibility Ranking")
        ax1.set_xlim(0, 1)
        ax1.axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="Neutral threshold")
        ax1.legend(fontsize=8)
        ax1.invert_yaxis()

        # Right: Stacked breakdown by metric
        ax2 = axes[1]
        x = np.arange(len(names))
        width = 0.2
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            vals = [report[n][metric] for n in names]
            ax2.barh(x + i * width - 0.3, vals, width, label=metric.capitalize(), color=color, alpha=0.8)
        ax2.set_yticks(x)
        ax2.set_yticklabels(names, fontsize=8)
        ax2.set_xlabel("Score")
        ax2.set_title("Credibility Breakdown by Metric")
        ax2.set_xlim(0, 1)
        ax2.legend(fontsize=8, loc="lower right")
        ax2.invert_yaxis()

        plt.suptitle("Source Credibility Dashboard", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Dashboard saved to: {save_path}")
            plt.close()
        else:
            plt.show()


# ── Standalone demo ────────────────────────────────────────────────────────────

def demo():
    """Run a standalone demo to verify the tool works."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from env.source_pool import SourcePool, QUERY_TYPES

    print("Source Credibility Scorer — Standalone Demo")
    print("=" * 50)

    pool = SourcePool(seed=42)
    scorer = SourceCredibilityScorer(pool.source_names)

    # Simulate 500 queries across all source/type combinations
    rng = np.random.default_rng(42)
    session_results = []

    for _ in range(500):
        source_idx = rng.integers(0, pool.n_sources)
        query_type = rng.choice(QUERY_TYPES)
        result = pool.query(source_idx, query_type)

        session_results.append(result)
        scorer.record_query(result["source"], result, session_results[-5:])

    # Print the trust report
    scorer.print_trust_report()

    # Show credibility-weighted reward example
    print("\nCredibility-Weighted Reward Examples:")
    for name in ["arxiv", "social_media", "government_policy"]:
        cred = scorer.get_credibility(name)
        dummy_result = {"source": name, "available": True, "relevance": 0.8, "latency": 1.0}
        adj_reward = scorer.credibility_weighted_reward(dummy_result)
        print(f"  {name:<24} credibility={cred:.3f}  raw_reward=0.800  adjusted={adj_reward:.3f}")

    # Save dashboard plot
    save_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
    os.makedirs(save_dir, exist_ok=True)
    scorer.plot_credibility_dashboard(os.path.join(save_dir, "credibility_dashboard.png"))

    print("\nDemo complete.")


if __name__ == "__main__":
    demo()
