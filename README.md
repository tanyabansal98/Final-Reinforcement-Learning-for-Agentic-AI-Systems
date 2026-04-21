# Madison RL — AI Industry Intelligence Agent

> Reinforcement Learning for Agentic AI Systems  
> Take-Home Final — Humanitarians.AI Madison Framework Enhancement

## Overview

This project integrates **all five reinforcement learning approaches** into the Madison intelligence-gathering framework, creating an AI agent that learns optimal source selection strategies for AI industry monitoring.

The agent learns which information sources (arXiv, tech news, SEC filings, GitHub, social media, etc.) are most valuable for different types of research queries, and improves its intelligence-gathering strategy through experience.

## Architecture

The system operates as a **five-layer RL pipeline**, where each method handles a distinct decision level:

| Layer | Method | RL Approach | Decision |
|-------|--------|-------------|----------|
| 1 | Meta-Learning (MAML) | Transfer Learning | Warm-start source priors for new topic domains |
| 2 | Contextual Bandits (LinUCB + Thompson) | Exploration Strategies | Which source category fits this query context? |
| 3 | Q-Learning + SARSA | Value-Based Learning | Given what's been found, what source is next? |
| 4 | PPO (Actor-Critic) | Policy Gradient | How to weight and synthesize collected info? |
| 5 | MAPPO Coordinator | Multi-Agent RL | How do parallel agents avoid redundancy? |

## RL Methods Implemented

### Method 1 — Value-Based Learning (`agents/q_learning.py`)
- **Q-Learning**: Off-policy Bellman updates with ε-greedy exploration
- **SARSA**: On-policy variant for conservative policy learning
- State discretization: query_type × budget_bucket × coverage_bucket
- Both run in parallel for comparative analysis

### Method 2 — Policy Gradient (`agents/ppo_agent.py`)
- **PPO** with clipped surrogate objective
- Generalized Advantage Estimation (GAE, λ=0.95)
- Analytical backpropagation (no PyTorch dependency)
- Actor-critic architecture for synthesis weight optimization

### Method 3 — Multi-Agent RL (`agents/marl_coordinator.py`)
- **MAPPO** with centralized critic, decentralized actors
- Communication protocol: agents broadcast relevance vectors
- Shared team reward with duplication penalty
- 3 parallel agents coordinating source queries

### Method 4 — Exploration Strategies (`agents/contextual_bandit.py`)
- **LinUCB**: Linear contextual bandits with UCB exploration bonus
- **Thompson Sampling**: Bayesian posterior sampling for smooth exploration
- Both run in parallel for learning curve comparison

### Method 5 — Meta-Learning (`agents/meta_learner.py`)
- **MAML-inspired** domain adaptation across 5 AI topic domains
- Meta-initialization learned from cross-domain experience
- Few-shot adaptation to held-out domains
- Transfer benefit measured as episodes-to-criterion

## Simulation Environment

12 simulated AI industry sources with topic-conditional quality distributions:

| Source | Category | Best For |
|--------|----------|----------|
| arXiv | Research | Technical capability queries |
| SEC EDGAR | Financial | Business and funding queries |
| Tech News | Media | Product launches and announcements |
| GitHub Feeds | Code | Open-source model releases |
| Social Media | Social | Breaking news (noisy) |
| Government Policy | Regulatory | Compliance and policy queries |
| ... | ... | ... |

5 query types: `technical`, `business`, `product`, `regulatory`, `market`

## Setup and Running

```bash
pip install -r requirements.txt

# Full training run
python main.py

# Baseline comparison
python training/evaluate.py

# Generate all analysis plots
python analysis/plot_all.py
```

## Results

### Learning Performance
- Agent converges to near-oracle performance (~98% of optimal) within 300 episodes
- Sublinear cumulative regret confirms genuine learning
- Clear improvement over random (+50%), round-robin (+58%), and greedy (+24%) baselines

### Key Findings
1. **LinUCB discovers context-source mapping**: The agent learns that arXiv is best for technical queries, SEC filings for business queries — without being told
2. **Q-Learning vs SARSA**: Q-Learning converges faster; SARSA produces more stable policies
3. **MARL reduces redundancy**: Duplication penalty drives agents to diversify source coverage
4. **Meta-learning accelerates adaptation**: 15-25% fewer episodes needed on new domains

## Project Structure

```
madison_rl/
├── env/
│   ├── source_pool.py       # 12 simulated sources
│   ├── query_engine.py       # Query generation, context vectors
│   ├── reward_function.py    # Relevance, novelty, latency rewards
│   └── madison_env.py        # Gym-style environment
├── agents/
│   ├── contextual_bandit.py  # LinUCB + Thompson Sampling
│   ├── q_learning.py         # Q-Learning + SARSA
│   ├── ppo_agent.py          # PPO synthesis agent
│   ├── marl_coordinator.py   # Multi-agent coordination
│   └── meta_learner.py       # MAML domain adaptation
├── training/
│   ├── train.py              # Main training loop
│   └── evaluate.py           # Baseline comparisons
├── analysis/
│   └── plot_all.py           # All visualizations
├── plots/                    # Generated analysis plots
├── main.py                   # Entry point
├── requirements.txt
└── README.md
```

## Ethical Considerations

- Source reliability is simulated, not measured from real sources — production deployment would require careful calibration and auditing of source quality assessments
- Automated intelligence gathering should complement, not replace, human judgment
- Exploration strategies must be bounded to prevent excessive querying of real-world APIs
- Multi-agent coordination must not amplify biases present in individual sources
