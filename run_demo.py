"""
run_demo.py
LIVE SESSION DEMO — Madison RL Agent in Action

This script demonstrates a research session using TRAINED RL agents.
It first runs a short training pass, then uses the learned policies
to process queries through the controller.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.train import train
from madison_integration import MadisonRLController, MADISON_AGENTS
from env.source_pool import QUERY_TYPES

def print_header(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")

def main():
    print_header("MADISON RL — LIVE RESEARCH SESSION DEMO")
    print("Framework: Humanitarians.AI")
    print("Goal: Gather and synthesize AI Industry Intelligence\n")

    # 1. Show the agent team
    print("[STEP 1] Agent Architecture:")
    for agent in MADISON_AGENTS:
        print(f"  • {agent.name:<18} | Method: {agent.rl_method:<40}")

    # 2. Train (short run for demo speed)
    print_header("TRAINING PHASE (500 episodes)")
    results = train(n_episodes=500, n_agents=3, seed=42, verbose=True)

    # 3. Load trained models into the controller
    controller = MadisonRLController()
    controller.load_trained_agents(results)

    # 4. Run inference with trained agents
    for query_type in ["technical", "business", "regulatory"]:
        print_header(f"INFERENCE — Query Type: {query_type.upper()}")

        result = controller.process_query(query_type, budget=4)

        for step in result["pipeline"]:
            time.sleep(0.3)
            print(f"\n  ▶ [{step['agent']:<18}] {step['action']}")
            print(f"    → {step['result']}")

        print(f"\n  Status: {result['status']}  |  Sources: {result['sources_used']}")

    # 5. Show credibility tool output
    print_header("CUSTOM TOOL: Source Credibility Report")
    scorer = results["credibility_scorer"]
    scorer.print_trust_report()

    # 6. Show memory state
    print_header("PERSISTENT MEMORY STATE")
    mem_stats = controller.memory.summary()
    print(f"  Episodic entries:    {mem_stats['episodic_entries']}")
    print(f"  Sources tracked:     {mem_stats['sources_tracked']}")
    print(f"  Total observations:  {mem_stats['total_observations']}")

    # 7. Show communication log
    print_header("INTER-AGENT COMMUNICATION (Message Bus)")
    stats = controller.message_bus.stats()
    print(f"  Total messages: {stats['total_messages']}")
    for msg_type, count in stats['by_type'].items():
        print(f"    • {msg_type:<25}: {count}")

    print_header("DEMO COMPLETE")
    print("All results generated from trained RL policies.")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
