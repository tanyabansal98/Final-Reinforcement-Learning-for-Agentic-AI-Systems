"""
main.py — Run the full Madison RL pipeline
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from training.train import train


def main():
    print("=" * 60)
    print("Madison RL — AI Industry Intelligence Agent")
    print("5 RL Methods: Bandits | Q-Learning | PPO | MARL | MAML")
    print("=" * 60 + "\n")

    results = train(
        n_episodes=1000,
        n_agents=3,
        seed=42,
        verbose=True,
    )

    print("\n--- Results Summary ---")
    rewards = results["episode_rewards"]
    print(f"Episodes run:       {len(rewards)}")
    print(f"Mean reward (all):  {np.mean(rewards):.4f}")
    print(f"Mean reward (last 100): {np.mean(rewards[-100:]):.4f}")
    print(f"Final cumulative regret: {results['regret_curve'][-1]:.2f}")

    if results["meta_transfer_results"]:
        last = results["meta_transfer_results"][-1]
        print(f"\nMeta-learning transfer benefit: {last['transfer_benefit']:+.4f}")
        print(f"  Transfer avg reward:  {last['transfer_avg_reward']:.4f}")
        print(f"  Scratch avg reward:   {last['scratch_avg_reward']:.4f}")

    print(f"\nMARL stats: {results['marl_stats']}")
    print(f"\nSimulation data manifest available at: ./env/simulation_data.json")
    print("\nDone. Run analysis scripts or run_demo.py to see results.")


if __name__ == "__main__":
    main()
