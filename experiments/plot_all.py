"""
experiments/plot_all.py
Generates all plots for the Madison RL report:
1. Learning curves (reward over episodes)
2. Cumulative regret (UCB vs random vs greedy)
3. Source selection heatmap (learned source preferences by query type)
4. Meta-learning transfer benefit
5. Q-Learning vs SARSA TD-error convergence
6. MARL duplication reduction over time
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.train import train
from env.source_pool import QUERY_TYPES


def moving_avg(arr, window=50):
    arr = np.array(arr)
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def plot_learning_curves(results, save_dir):
    """Plot 1: Reward over episodes with moving average."""
    fig, ax = plt.subplots(figsize=(10, 5))
    rewards = results["episode_rewards"]
    ax.plot(rewards, alpha=0.15, color="#378ADD", label="Raw reward")
    ma = moving_avg(rewards, 50)
    ax.plot(range(len(rewards) - len(ma), len(rewards)), ma,
            color="#185FA5", linewidth=2, label="Moving avg (50)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward")
    ax.set_title("Madison RL — Learning curve (all 5 methods)")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "learning_curve.png"), dpi=150)
    plt.close()


def plot_regret(results, save_dir):
    """Plot 2: Cumulative regret over episodes."""
    fig, ax = plt.subplots(figsize=(10, 5))
    regret = results["regret_curve"]
    ax.plot(regret, color="#D85A30", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative regret")
    ax.set_title("Cumulative regret vs oracle policy")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "regret_curve.png"), dpi=150)
    plt.close()


def plot_source_heatmap(results, save_dir):
    """Plot 3: Learned source preferences by query type."""
    bandit = results["bandit_agent"]
    source_names = results["source_names"]
    n_sources = len(source_names)

    # Build context vectors for each query type
    heatmap = np.zeros((n_sources, len(QUERY_TYPES)))
    for j, qt in enumerate(QUERY_TYPES):
        ctx = np.zeros(11, dtype=np.float32)
        ctx[j] = 1.0
        ctx[5] = 1.0  # medium urgency
        ctx[8] = 1.0  # full budget
        prefs = bandit.get_learned_preferences(ctx)
        heatmap[:, j] = prefs["ucb"]

    # Normalize columns for readability
    for j in range(heatmap.shape[1]):
        col_range = heatmap[:, j].max() - heatmap[:, j].min()
        if col_range > 1e-6:
            heatmap[:, j] = (heatmap[:, j] - heatmap[:, j].min()) / col_range

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(heatmap, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(QUERY_TYPES)))
    ax.set_xticklabels(QUERY_TYPES, rotation=30, ha="right")
    ax.set_yticks(range(n_sources))
    ax.set_yticklabels(source_names)
    ax.set_xlabel("Query type")
    ax.set_ylabel("Source")
    ax.set_title("Learned source preferences (LinUCB, normalized)")
    fig.colorbar(im, label="Normalized preference score")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "source_heatmap.png"), dpi=150)
    plt.close()

    # Also plot ground truth for comparison
    env = results["env"]
    gt = env.get_ground_truth_matrix()
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    im2 = ax2.imshow(gt, cmap="YlOrRd", aspect="auto")
    ax2.set_xticks(range(len(QUERY_TYPES)))
    ax2.set_xticklabels(QUERY_TYPES, rotation=30, ha="right")
    ax2.set_yticks(range(n_sources))
    ax2.set_yticklabels(source_names)
    ax2.set_xlabel("Query type")
    ax2.set_ylabel("Source")
    ax2.set_title("Ground truth source quality matrix")
    fig2.colorbar(im2, label="Mean reward")
    fig2.tight_layout()
    fig2.savefig(os.path.join(save_dir, "ground_truth_heatmap.png"), dpi=150)
    plt.close()


def plot_td_errors(results, save_dir):
    """Plot 5: Q-Learning vs SARSA TD-error convergence."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ql = moving_avg(results["q_learning_td_errors"], 30)
    sa = moving_avg(results["sarsa_td_errors"], 30)
    ax.plot(ql, color="#185FA5", linewidth=1.5, label="Q-Learning", alpha=0.8)
    ax.plot(sa, color="#1D9E75", linewidth=1.5, label="SARSA", alpha=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean TD error (smoothed)")
    ax.set_title("Value-based learning — TD error convergence")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "td_errors.png"), dpi=150)
    plt.close()


def plot_marl_stats(results, save_dir):
    """Plot 6: MARL duplication reduction over time."""
    team_r = results["marl_team_rewards"]
    if not team_r:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ma = moving_avg(team_r, 20)
    ax.plot(team_r, alpha=0.15, color="#7F77DD")
    ax.plot(range(len(team_r) - len(ma), len(team_r)), ma,
            color="#534AB7", linewidth=2, label="Moving avg (20)")
    ax.set_xlabel("MARL episode")
    ax.set_ylabel("Team reward")
    ax.set_title("Multi-agent coordination — team reward over time")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "marl_team_reward.png"), dpi=150)
    plt.close()


def plot_transfer(results, save_dir):
    """Plot 4: Meta-learning transfer benefit."""
    transfers = results["meta_transfer_results"]
    if not transfers:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(transfers))
    t_rewards = [t["transfer_avg_reward"] for t in transfers]
    s_rewards = [t["scratch_avg_reward"] for t in transfers]
    ax.plot(x, t_rewards, color="#1D9E75", linewidth=2, marker="o", label="With meta-init (MAML)")
    ax.plot(x, s_rewards, color="#D85A30", linewidth=2, marker="s", label="From scratch")
    ax.set_xlabel("Meta-learning evaluation round")
    ax.set_ylabel("Average reward on held-out domain")
    ax.set_title("Meta-learning transfer benefit")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "transfer_benefit.png"), dpi=150)
    plt.close()


def plot_source_usage(results, save_dir):
    """Bonus: bar chart of source usage frequency by query type."""
    log = results["source_selection_log"]
    source_names = results["source_names"]

    usage = {qt: np.zeros(len(source_names)) for qt in QUERY_TYPES}
    for entry in log:
        qt = entry["query_type"]
        src = entry["source"]
        if src in source_names:
            usage[qt][source_names.index(src)] += 1

    fig, axes = plt.subplots(1, len(QUERY_TYPES), figsize=(18, 5), sharey=True)
    colors = ["#378ADD", "#1D9E75", "#D85A30", "#534AB7", "#BA7517"]
    for ax, (qt, color) in zip(axes, zip(QUERY_TYPES, colors)):
        counts = usage[qt]
        ax.barh(range(len(source_names)), counts, color=color, alpha=0.7)
        ax.set_yticks(range(len(source_names)))
        if ax == axes[0]:
            ax.set_yticklabels(source_names, fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.set_title(qt, fontsize=10)
        ax.set_xlabel("Queries", fontsize=8)
    fig.suptitle("Source usage by query type", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "source_usage.png"), dpi=150, bbox_inches="tight")
    plt.close()


def run_all_plots(n_episodes=1000):
    """Train and generate all plots."""
    save_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
    os.makedirs(save_dir, exist_ok=True)

    print("Training model for analysis...")
    results = train(n_episodes=n_episodes, seed=42, verbose=True)

    print("\nGenerating plots...")
    plot_learning_curves(results, save_dir)
    print("  - learning_curve.png")
    plot_regret(results, save_dir)
    print("  - regret_curve.png")
    plot_source_heatmap(results, save_dir)
    print("  - source_heatmap.png + ground_truth_heatmap.png")
    plot_td_errors(results, save_dir)
    print("  - td_errors.png")
    plot_marl_stats(results, save_dir)
    print("  - marl_team_reward.png")
    plot_transfer(results, save_dir)
    print("  - transfer_benefit.png")
    plot_source_usage(results, save_dir)
    print("  - source_usage.png")
    print(f"\nAll plots saved to {save_dir}/")
    return results


if __name__ == "__main__":
    run_all_plots()
