# Technical Report: Reinforcement Learning for Agentic AI Systems
**Project:** Madison RL — AI Industry Intelligence Agent  
**Framework:** Humanitarians.AI Madison Integration  
**Author:** [Your Name]  
**Date:** April 2026

---

## 1. Executive Summary

This project integrates five reinforcement learning methods into the Humanitarians.AI **Madison** intelligence framework. The system enables an autonomous agent to learn optimal information-gathering strategies across 12 simulated AI industry sources (arXiv, SEC EDGAR, GitHub, etc.) and 5 query types (technical, business, product, regulatory, market). Through 2,000 episodes of interaction, the agent discovers context-dependent source preferences purely from reward feedback, without access to the ground-truth quality matrix.

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────┐
│              MadisonRLController                     │
│         (Orchestration & Error Handling)              │
├──────────────────────────────────────────────────────┤
│                                                      │
│  Layer 1: DomainAdapter (Meta-Learning / MAML)       │
│      → Warm-start source priors for new domains      │
│                    ↓                                 │
│  Layer 2: SourceSelector (Contextual Bandits/LinUCB) │
│      → Pick highest-value source for this context    │
│                    ↓                                 │
│  Layer 3: SessionPlanner (Q-Learning + SARSA)        │
│      → Optimize multi-step query sequence            │
│                    ↓                                 │
│  Layer 4: CoordinationAgent (MAPPO)                  │
│      → 3 parallel agents avoid redundant queries     │
│                    ↓                                 │
│  Layer 5: SynthesisAgent (PPO)                       │
│      → Weight and combine results                    │
│                                                      │
├──────────────────────────────────────────────────────┤
│  Custom Tool: SourceCredibilityScorer                │
│  Memory: MadisonMemory (Episodic + Semantic)         │
│  Communication: MadisonMessageBus                    │
└──────────────────────────────────────────────────────┘
         ↕                    ↕
┌────────────────┐  ┌─────────────────────┐
│  MadisonEnv    │  │  MultiAgentEnv      │
│  (12 sources)  │  │  (3 parallel agents)│
└────────────────┘  └─────────────────────┘
```

### Component Summary

| Layer | RL Method | Algorithm | File |
|:------|:----------|:----------|:-----|
| 1 | Meta-Learning | MAML-inspired | `agents/meta_learner.py` |
| 2 | Exploration | LinUCB + Thompson Sampling | `agents/contextual_bandit.py` |
| 3 | Value-Based | Q-Learning + SARSA | `agents/q_learning.py` |
| 4 | Multi-Agent | MAPPO (Centralized Critic) | `agents/marl_coordinator.py` |
| 5 | Policy Gradient | PPO (Analytical Backprop) | `agents/ppo_agent.py` |

---

## 3. Mathematical Formulations

### 3.1 Value-Based Learning (Q-Learning)

The agent maintains a tabular Q-function Q(s, a) over discretized states.

**State space:** s = (query_type, budget_bucket, coverage_bucket)  
- `query_type` ∈ {technical, business, product, regulatory, market} (5 values)  
- `budget_bucket` ∈ {full, 2/3, 1/3, depleted} (4 values)  
- `coverage_bucket` ∈ {low, medium, high} (3 values)  
- Total: |S| = 5 × 4 × 3 = 60 states

**Action space:** a ∈ {0, 1, ..., 11} (12 sources)

**Q-Learning update (off-policy):**

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

**SARSA update (on-policy):**

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]$$

where $a'$ is the actual next action taken under the current policy.

**Exploration:** ε-greedy with exponential decay: $\epsilon_{t+1} = \max(\epsilon_{min}, \epsilon_t \cdot 0.995)$

### 3.2 Contextual Bandits (LinUCB)

For each arm (source) $a$, we maintain a ridge regression model:

$$\hat{r}(a, x) = \theta_a^T x, \quad \theta_a = A_a^{-1} b_a$$

where $A_a = \lambda I + \sum_{t: a_t=a} x_t x_t^T$ and $b_a = \sum_{t: a_t=a} r_t x_t$.

**UCB selection rule:**

$$a^* = \arg\max_a \left[ \theta_a^T x + \alpha \sqrt{x^T A_a^{-1} x} \right]$$

The exploration bonus $\sqrt{x^T A_a^{-1} x}$ is large for contexts where arm $a$ has been rarely observed.

### 3.3 Policy Gradient (PPO)

PPO optimizes synthesis weights using a clipped surrogate objective:

$$L^{CLIP}(\theta) = \mathbb{E} \left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$ is the probability ratio.

**Advantage estimation (GAE-λ):**

$$\hat{A}_t = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

### 3.4 Multi-Agent RL (MAPPO)

**Team reward function:**

$$R_{team} = \sum_{i=1}^{N} R_i^{indiv} - 0.3 \cdot n_{duplicates} + 0.15 \cdot n_{new\_sources}$$

**Centralized critic:** $V(s^{global}) = f(x_1, x_2, ..., x_N, m_{global})$  
where $m_{global} = \max_i m_i$ is the element-wise max of all agents' message vectors.

### 3.5 Meta-Learning (MAML-inspired)

Meta-initialization is a weighted average of domain-specific parameters:

$$\theta^{meta}_a = (1 - \beta) \theta^{meta}_a + \beta \cdot \frac{1}{|D_{train}|} \sum_{d \in D_{train}} \theta^d_a$$

**Few-shot adaptation:** Initialize new domain from $\theta^{meta}$, fine-tune with K support examples.

---

## 4. Reward Engineering

The reward function has two components:

**Step reward** (after each source query):
$$r_{step} = w_1 \cdot relevance + w_2 \cdot novelty - w_3 \cdot \frac{latency}{10}$$

**Session reward** (end of episode):
$$r_{session} = w_1 \cdot max\_relevance + 0.3 \cdot avg\_relevance + w_4 \cdot coverage - w_5 \cdot conflict$$

**Credibility-adjusted reward** (via custom tool):
$$r_{adjusted} = relevance \cdot (0.5 + 0.5 \cdot credibility)$$

---

## 5. Experimental Results

### Experiment A: RL vs. Random Baseline
The trained agent significantly outperforms a random policy across all 5 query types, validated with a paired t-test ($p < 0.05$) and Cohen's d effect size.

### Experiment B: Multi-Agent Coordination
The MARL coordinator reduces duplicate source queries compared to independent agents, as measured by the duplication rate metric tracked during training.

### Experiment C: Meta-Learning Transfer
Agents initialized with meta-learned parameters adapt faster to held-out domains (e.g., "ai_market") compared to training from scratch, as measured by average reward on evaluation episodes.

Refer to `plots/` directory for all generated visualizations.

---

## 6. Challenges and Solutions

| Challenge | Solution |
|:----------|:---------|
| **State space explosion** | Discretized into 60 states using budget/coverage buckets instead of raw values. |
| **Noisy rewards** | Used credibility-weighted rewards to discount unreliable source signals. |
| **MARL credit assignment** | Used centralized critic with decentralized actors (CTDE paradigm). |
| **Meta-learning overfitting** | Held out 1 of 5 domains for transfer evaluation; used meta_lr=0.3 for slow interpolation. |
| **PPO without PyTorch** | Implemented analytical backpropagation through a custom NumPy MLP with gradient clipping. |

---

## 7. Limitations

1. **Simulation fidelity:** Source quality distributions are synthetic. Real-world sources have non-stationary quality and adversarial dynamics not captured here.
2. **Scalability:** The tabular Q-Learning approach (60 states) does not scale to continuous or high-dimensional state spaces. A neural function approximator would be needed for production.
3. **MARL simplification:** Agents share the same LinUCB architecture. True heterogeneous agent policies could improve coordination.
4. **No online human evaluation:** The "relevance" metric is model-defined, not validated by human judges.
5. **Static source pool:** The set of 12 sources is fixed. A production system would need to handle source addition/removal dynamically.

---

## 8. Ethical Considerations

### 8.1 Bias Amplification
The RL agent will converge toward sources that maximize its reward function. If the reward function encodes implicit biases (e.g., favoring English-language sources), the agent will amplify these biases over time. The coverage bonus in the MARL reward partially mitigates this by incentivizing diverse source selection.

### 8.2 Trust Calibration
The `SourceCredibilityScorer` assigns numeric trust scores to sources. In a production deployment, these scores must be transparent and auditable. A source marked as "low trust" could be unfairly excluded from consideration, especially if the trust model has not been validated on diverse content.

### 8.3 Automation Risk
Fully autonomous intelligence gathering could produce reports without human review. We recommend that the agent's outputs always be presented as recommendations, with a human analyst making final judgments.

### 8.4 Exploration Safety
During training, the exploration strategy (ε-greedy, UCB) causes the agent to query sources it hasn't tried before. In a production environment with rate-limited APIs, this exploration must be bounded to prevent excessive API calls or terms-of-service violations.

---

## 9. Future Improvements

1. **Deep RL:** Replace tabular Q-Learning with DQN or a neural bandit for continuous state spaces.
2. **Real data integration:** Connect to actual APIs (arXiv, SEC EDGAR) with rate limiting and caching.
3. **Human-in-the-loop:** Add a feedback mechanism where analysts rate the agent's recommendations, creating a human reward signal.
4. **Curriculum learning:** Automatically adjust the difficulty of queries during training based on the agent's current performance.
5. **Adversarial robustness:** Test the agent's behavior when sources deliberately provide misleading information.

---

## Appendix: Reproducibility

```bash
# Install dependencies
pip install -r requirements.txt

# Full training run (1000 episodes)
python main.py

# Generate all analysis plots
python analysis/plot_all.py

# Multi-seed statistical validation
python analysis/statistical_validation.py

# Live demo with trained agents
python run_demo.py
```
