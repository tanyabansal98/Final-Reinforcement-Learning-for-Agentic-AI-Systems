"""
experiments/statistical_validation.py
Run training across multiple seeds and compute:
- Mean ± std learning curves
- 95% confidence intervals
- Paired t-tests between RL methods and baselines
- Effect sizes (Cohen's d)

Run from Madison_r1 root:
    python experiments/statistical_validation.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from training.train import train


# ── Configuration ──────────────────────────────────────────────────────────────
N_SEEDS = 5
SEEDS = [42, 123, 256, 789, 1024]
N_EPISODES = 1000
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")


def cohens_d(group1, group2):
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / (pooled_std + 1e-8)


def run_baseline(seed, n_episodes=200):
    """Run random baseline for comparison."""
    from env.madison_env import MadisonEnv
    np.random.seed(seed)
    env = MadisonEnv(seed=seed)
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total, done, steps = 0.0, False, 0
        while not done and steps < 10:
            action = np.random.randint(env.n_sources)
            obs, r, done, _, _ = env.step(action)
            total += r; steps += 1
        rewards.append(total)
    return rewards


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 60)
    print("  STATISTICAL VALIDATION — Multi-Seed Analysis")
    print(f"  Seeds: {SEEDS}  |  Episodes per seed: {N_EPISODES}")
    print("=" * 60)

    # ── Run training across seeds ──────────────────────────────────────────
    all_rewards = []          # shape: (n_seeds, n_episodes)
    all_regrets = []
    all_ql_td = []
    all_sarsa_td = []
    all_final_rewards = []    # last 100 episodes per seed
    all_transfer_benefits = []
    all_marl_dup_rates = []   # real MARL duplication rates per seed
    all_meta_transfer = []    # real meta transfer results per seed
    seed_times = []

    for i, seed in enumerate(SEEDS):
        print(f"\n{'─'*50}")
        print(f"  Run {i+1}/{N_SEEDS} — seed={seed}")
        print(f"{'─'*50}")
        t0 = time.time()

        results = train(n_episodes=N_EPISODES, n_agents=3, seed=seed, verbose=False)

        elapsed = time.time() - t0
        seed_times.append(elapsed)

        rewards = results["episode_rewards"]
        all_rewards.append(rewards)
        all_regrets.append(results["regret_curve"])
        all_ql_td.append(results["q_learning_td_errors"])
        all_sarsa_td.append(results["sarsa_td_errors"])
        all_final_rewards.append(np.mean(rewards[-100:]))

        if results["meta_transfer_results"]:
            tb = np.mean([t["transfer_benefit"] for t in results["meta_transfer_results"]])
            all_transfer_benefits.append(tb)
            all_meta_transfer.append(results["meta_transfer_results"])

        if results.get("marl_duplication_rates"):
            all_marl_dup_rates.append(results["marl_duplication_rates"])

        print(f"  Mean reward: {np.mean(rewards):.4f}  |  "
              f"Last 100: {np.mean(rewards[-100:]):.4f}  |  "
              f"Time: {elapsed:.1f}s")

    all_rewards = np.array(all_rewards)   # (5, 1000)
    all_regrets = np.array(all_regrets)
    all_ql_td = np.array(all_ql_td)
    all_sarsa_td = np.array(all_sarsa_td)

    # ── Run baselines across same seeds ────────────────────────────────────
    print("\nRunning random baselines across seeds...")
    all_baseline_rewards = []
    for seed in SEEDS:
        baseline_r = run_baseline(seed, n_episodes=200)
        all_baseline_rewards.append(np.mean(baseline_r))
    all_baseline_rewards = np.array(all_baseline_rewards)

    # ══════════════════════════════════════════════════════════════════════
    # STATISTICAL TESTS
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STATISTICAL TEST RESULTS")
    print("=" * 60)

    # 1. RL vs Random baseline (paired t-test)
    rl_means = np.array(all_final_rewards)
    t_stat, p_val = stats.ttest_rel(rl_means, all_baseline_rewards)
    d = cohens_d(rl_means, all_baseline_rewards)
    print(f"\n1. RL Agent vs Random Baseline (paired t-test)")
    print(f"   RL mean (last 100):      {np.mean(rl_means):.4f} ± {np.std(rl_means):.4f}")
    print(f"   Random mean:             {np.mean(all_baseline_rewards):.4f} ± {np.std(all_baseline_rewards):.4f}")
    print(f"   t-statistic:             {t_stat:.4f}")
    print(f"   p-value:                 {p_val:.6f}")
    print(f"   Cohen's d:               {d:.4f}")
    print(f"   Significant (p<0.05):    {'YES ✓' if p_val < 0.05 else 'NO'}")

    # 2. Q-Learning vs SARSA (paired t-test on mean TD errors)
    ql_final_td = np.array([np.mean(td[-100:]) for td in all_ql_td])
    sarsa_final_td = np.array([np.mean(td[-100:]) for td in all_sarsa_td])
    t_stat2, p_val2 = stats.ttest_rel(ql_final_td, sarsa_final_td)
    print(f"\n2. Q-Learning vs SARSA — Final TD Error (paired t-test)")
    print(f"   Q-Learning mean TD:      {np.mean(ql_final_td):.4f} ± {np.std(ql_final_td):.4f}")
    print(f"   SARSA mean TD:           {np.mean(sarsa_final_td):.4f} ± {np.std(sarsa_final_td):.4f}")
    print(f"   t-statistic:             {t_stat2:.4f}")
    print(f"   p-value:                 {p_val2:.6f}")

    # 3. Early vs Late performance (paired t-test — learning actually occurred)
    early_means = np.array([np.mean(r[:100]) for r in all_rewards])
    late_means = np.array([np.mean(r[-100:]) for r in all_rewards])
    t_stat3, p_val3 = stats.ttest_rel(late_means, early_means)
    d3 = cohens_d(late_means, early_means)
    print(f"\n3. Early (eps 0-100) vs Late (eps 900-1000) Performance")
    print(f"   Early mean reward:       {np.mean(early_means):.4f} ± {np.std(early_means):.4f}")
    print(f"   Late mean reward:        {np.mean(late_means):.4f} ± {np.std(late_means):.4f}")
    print(f"   Improvement:             {np.mean(late_means) - np.mean(early_means):+.4f}")
    print(f"   t-statistic:             {t_stat3:.4f}")
    print(f"   p-value:                 {p_val3:.6f}")
    print(f"   Cohen's d:               {d3:.4f}")
    print(f"   Significant (p<0.05):    {'YES ✓' if p_val3 < 0.05 else 'NO'}")

    # 4. Meta-learning transfer benefit (one-sample t-test > 0)
    if all_transfer_benefits:
        tb_arr = np.array(all_transfer_benefits)
        t_stat4, p_val4 = stats.ttest_1samp(tb_arr, 0.0)
        print(f"\n4. Meta-Learning Transfer Benefit > 0 (one-sample t-test)")
        print(f"   Mean transfer benefit:   {np.mean(tb_arr):.4f} ± {np.std(tb_arr):.4f}")
        print(f"   t-statistic:             {t_stat4:.4f}")
        print(f"   p-value:                 {p_val4:.6f}")
        print(f"   Significant (p<0.05):    {'YES ✓' if p_val4 < 0.05 else 'NO'}")

    # 5. 95% confidence intervals
    print(f"\n5. 95% Confidence Intervals (across {N_SEEDS} seeds)")
    overall_means = np.array([np.mean(r) for r in all_rewards])
    ci_low, ci_high = stats.t.interval(
        0.95, df=N_SEEDS - 1, loc=np.mean(overall_means), scale=stats.sem(overall_means)
    )
    print(f"   Overall reward:          [{ci_low:.4f}, {ci_high:.4f}]")
    ci_reg = stats.t.interval(
        0.95, df=N_SEEDS - 1,
        loc=np.mean([r[-1] for r in all_regrets]),
        scale=stats.sem([r[-1] for r in all_regrets]),
    )
    print(f"   Final regret:            [{ci_reg[0]:.2f}, {ci_reg[1]:.2f}]")

    # ══════════════════════════════════════════════════════════════════════
    # PLOTS WITH ERROR BANDS
    # ══════════════════════════════════════════════════════════════════════

    # Plot A: Learning curve with ±1 std shading
    mean_curve = np.mean(all_rewards, axis=0)
    std_curve = np.std(all_rewards, axis=0)

    # Smooth for readability
    window = 50
    def smooth(arr):
        return np.convolve(arr, np.ones(window) / window, mode="valid")

    sm_mean = smooth(mean_curve)
    sm_std = smooth(std_curve)
    x = np.arange(len(sm_mean))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, sm_mean, color="#185FA5", linewidth=2, label=f"Mean across {N_SEEDS} seeds")
    ax.fill_between(x, sm_mean - sm_std, sm_mean + sm_std, alpha=0.2, color="#378ADD",
                     label="±1 std deviation")
    ax.fill_between(x, sm_mean - 1.96 * sm_std / np.sqrt(N_SEEDS),
                     sm_mean + 1.96 * sm_std / np.sqrt(N_SEEDS),
                     alpha=0.35, color="#185FA5", label="95% CI")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title(f"Learning Curve with Statistical Bounds ({N_SEEDS} seeds, smoothed window={window})")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "learning_curve_multiseed.png"), dpi=150)
    plt.close()
    print(f"\nPlot saved: learning_curve_multiseed.png")

    # Plot B: Regret curve with ±1 std shading
    mean_regret = np.mean(all_regrets, axis=0)
    std_regret = np.std(all_regrets, axis=0)

    fig2, ax2 = plt.subplots(figsize=(11, 5))
    ax2.plot(mean_regret, color="#D85A30", linewidth=2, label=f"Mean regret ({N_SEEDS} seeds)")
    ax2.fill_between(range(len(mean_regret)),
                      mean_regret - std_regret, mean_regret + std_regret,
                      alpha=0.2, color="#D85A30", label="±1 std deviation")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Cumulative Regret")
    ax2.set_title(f"Cumulative Regret with Error Bounds ({N_SEEDS} seeds)")
    ax2.legend()
    ax2.grid(alpha=0.2)
    fig2.tight_layout()
    fig2.savefig(os.path.join(SAVE_DIR, "regret_multiseed.png"), dpi=150)
    plt.close()
    print(f"Plot saved: regret_multiseed.png")

    # Plot C: TD error comparison with error bands
    mean_ql = np.mean(all_ql_td, axis=0)
    mean_sa = np.mean(all_sarsa_td, axis=0)
    std_ql = np.std(all_ql_td, axis=0)
    std_sa = np.std(all_sarsa_td, axis=0)

    sm_ql, sm_sa = smooth(mean_ql), smooth(mean_sa)
    sm_ql_s, sm_sa_s = smooth(std_ql), smooth(std_sa)
    x_td = np.arange(len(sm_ql))

    fig3, ax3 = plt.subplots(figsize=(11, 5))
    ax3.plot(x_td, sm_ql, color="#185FA5", linewidth=1.5, label="Q-Learning")
    ax3.fill_between(x_td, sm_ql - sm_ql_s, sm_ql + sm_ql_s, alpha=0.15, color="#185FA5")
    ax3.plot(x_td, sm_sa, color="#1D9E75", linewidth=1.5, label="SARSA")
    ax3.fill_between(x_td, sm_sa - sm_sa_s, sm_sa + sm_sa_s, alpha=0.15, color="#1D9E75")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Mean TD Error")
    ax3.set_title(f"Q-Learning vs SARSA — TD Error Convergence ({N_SEEDS} seeds)")
    ax3.legend()
    ax3.grid(alpha=0.2)
    fig3.tight_layout()
    fig3.savefig(os.path.join(SAVE_DIR, "td_errors_multiseed.png"), dpi=150)
    plt.close()
    print(f"Plot saved: td_errors_multiseed.png")

    # Plot D: Per-seed final reward bar chart
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(N_SEEDS)
    width = 0.35
    ax4.bar(x_pos - width / 2, all_final_rewards, width, color="#185FA5", label="Trained RL")
    ax4.bar(x_pos + width / 2, all_baseline_rewards, width, color="#aaaaaa", label="Random")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f"Seed {s}" for s in SEEDS])
    ax4.set_ylabel("Mean Episode Reward (last 100)")
    ax4.set_title("Per-Seed Performance: RL Agent vs Random Baseline")
    ax4.legend()
    ax4.grid(alpha=0.2, axis="y")
    fig4.tight_layout()
    fig4.savefig(os.path.join(SAVE_DIR, "per_seed_comparison.png"), dpi=150)
    plt.close()
    print(f"Plot saved: per_seed_comparison.png")

    # [EXPERIMENT B] Plot E: Multi-Agent Coordination Benefit (REAL DATA)
    if all_marl_dup_rates:
        # Pad arrays to same length and compute stats across seeds
        min_len = min(len(d) for d in all_marl_dup_rates)
        trimmed = np.array([d[:min_len] for d in all_marl_dup_rates])
        m_mean = smooth(np.mean(trimmed, axis=0))
        m_std = smooth(np.std(trimmed, axis=0))
        trim_len = min(len(m_mean), len(m_std))
        m_mean, m_std = m_mean[:trim_len], m_std[:trim_len]
        x_marl = np.arange(trim_len) * 5

        fig5, ax5 = plt.subplots(figsize=(11, 5))
        ax5.plot(x_marl, m_mean, color="#7B1FA2", linewidth=2, label=f"MARL Coordination ({N_SEEDS} seeds)")
        ax5.fill_between(x_marl, m_mean - m_std, m_mean + m_std, alpha=0.2, color="#7B1FA2")
        # Show the theoretical baseline for independent (non-communicating) agents
        ax5.axhline(y=(1.0 - 1.0/3.0), color="gray", linestyle="--", alpha=0.6,
                    label="Expected dup. rate (random, 3 agents)")
        ax5.set_xlabel("Episode")
        ax5.set_ylabel("Duplicate Query Rate")
        ax5.set_title("Experiment B: Multi-Agent Coordination — Redundancy Reduction")
        ax5.legend()
        ax5.grid(alpha=0.2)
        fig5.tight_layout()
        fig5.savefig(os.path.join(SAVE_DIR, "marl_coordination_benefit.png"), dpi=150)
        plt.close()
        print(f"Plot saved: marl_coordination_benefit.png")
    else:
        print("Warning: No MARL duplication data collected. Skipping Plot E.")

    # [EXPERIMENT C] Plot F: Meta-Learning Transfer Benefit (REAL DATA)
    if all_meta_transfer:
        fig6, ax6 = plt.subplots(figsize=(11, 5))
        # Collect per-evaluation-round transfer vs scratch rewards across seeds
        max_rounds = max(len(mt) for mt in all_meta_transfer)
        transfer_means, scratch_means = [], []
        for r in range(max_rounds):
            t_vals = [mt[r]["transfer_avg_reward"] for mt in all_meta_transfer if r < len(mt)]
            s_vals = [mt[r]["scratch_avg_reward"] for mt in all_meta_transfer if r < len(mt)]
            transfer_means.append(np.mean(t_vals))
            scratch_means.append(np.mean(s_vals))

        x_meta = np.arange(len(transfer_means))
        ax6.plot(x_meta, transfer_means, color="#C2185B", linewidth=2, marker="o",
                 markersize=5, label="With Meta-Init (MAML)")
        ax6.plot(x_meta, scratch_means, color="#757575", linewidth=2, marker="s",
                 markersize=5, linestyle="--", label="From Scratch")
        ax6.set_xlabel("Meta-Learning Evaluation Round")
        ax6.set_ylabel("Average Reward on Held-Out Domain")
        ax6.set_title(f"Experiment C: Meta-Learning Transfer Benefit ({N_SEEDS} seeds)")
        ax6.legend()
        ax6.grid(alpha=0.2)
        fig6.tight_layout()
        fig6.savefig(os.path.join(SAVE_DIR, "meta_transfer_comparison.png"), dpi=150)
        plt.close()
        print(f"Plot saved: meta_transfer_comparison.png")
    else:
        print("Warning: No meta-transfer data collected. Skipping Plot F.")

    print(f"\nTotal time: {sum(seed_times):.1f}s ({np.mean(seed_times):.1f}s per seed)")
    print("Statistical validation complete.")


if __name__ == "__main__":
    main()
