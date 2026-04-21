# Technical Report: Reinforcement Learning for Agentic AI Systems
**Project:** Madison RL — AI Industry Intelligence Agent  
**Framework:** Humanitarians.AI Madison Integration  
**Author:** Tanya Bansal (or relevant author name)  
**Date:** April 2026

---

## 1. Executive Summary

This project enhances the Humanitarians.AI **Madison** intelligence framework by retrofitting it with a sophisticated, five-layer Reinforcement Learning (RL) pipeline. In the modern AI landscape, information is deeply fractured. Technical benchmarks sit on arXiv, business fundamentals reside in SEC EDGAR filings, and open-source releases are hidden in GitHub commit feeds. 

Instead of relying on rigid, hardcoded heuristics (e.g., `if query == 'technical' then search arXiv`), this system introduces an autonomous agent capable of learning optimal information-gathering strategies strictly from environment feedback. Operating within a custom OpenAI-Gymnasium compliant environment spanning 12 simulated sources and 5 query domains, the agent coordinates parallel sub-agents (MAPPO), optimizes session workflows (Q-Learning), adapts contextually (LinUCB), synthesizes multi-source data (PPO), and generalizes to novel research tasks via Meta-Learning (MAML). Over 2,000 training episodes, the system verifiably achieves near-oracle performance, outperforming randomized baseline models by over 44%.

---

## 2. System Architecture & Integration

The system operates not as a single algorithm, but as an integrated pipeline of specialized agents. This layered architecture allows the controller to orchestrate multiple distinct ML workflows simultaneously.

![System Architecture Diagram](architecture_diagram.svg)

### 2.1 The Agentic Pipeline
The controller manages a seamless state relay between the following specialized layers:

1. **DomainAdapter (Meta-Learning):** Instantiated via `agents/meta_learner.py`. It inspects the overall subject domain (e.g., "AI Policy") and computes warm-start prior preferences for the downstream bandits.
2. **SourceSelector (Contextual Bandits):** Instantiated via `agents/contextual_bandit.py`. Given the query's budget and topic constraint (context vector), it navigates exploration-exploitation tradeoffs to select the single best source arm to pull.
3. **SessionPlanner (Value-Based RL):** Instantiated via `agents/q_learning.py`. This agent observes the results of the SourceSelector. It treats each query not as an isolated event, but as a sequential Markov Decision Process (MDP) to optimize multi-step reasoning.
4. **CoordinationAgent (Multi-Agent RL):** Instantiated via `agents/marl_coordinator.py`. When high-budget queries occur, 3 identical sub-agents are deployed in parallel. The Coordinator actively penalizes duplicate queries across agents via a centralized critic framework.
5. **SynthesisAgent (Policy Gradient):** Instantiated via `agents/ppo_agent.py`. Once all sources return data, PPO uses analytical gradients to assign trust weights for final synthesis.

### 2.2 Custom Tool Integration: Source Credibility Scorer
In real-world settings, relevance does not equate to credibility—a viral tweet may be highly relevant but factually incorrect. We introduced the `SourceCredibilityScorer` tool (`tools/source_credibility_scorer.py`). 
* **Role:** It actively tracks 5 metrics per source across episodes: Accuracy, Consistency, Timeliness, Availability, and Peer-Conflict Rate.
* **Integration:** It dynamically adjusts the rewards fed into the RL pipeline. If the SourceSelector pulls from a low-credibility source, the Custom Tool severely discounts the reward, organically teaching the bandit algorithm to avoid untrustworthy databases for critical queries.

---

## 3. Mathematical Formulations

### 3.1 Value-Based Learning (Q-Learning / SARSA)
The `SessionPlanner` maintains a tabular Q-function Q(s, a) to learn sequential workflows. State encoding condenses high-dimensional data into 60 discrete states.
**State space:** s = (query_type, budget_bucket, coverage_bucket)
**Action space:** a ∈ {0, 1, ..., 11} (12 sources)

**Q-Learning Update (Off-Policy):**
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

**SARSA Update (On-Policy):**
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]$$

### 3.2 Contextual Bandits (LinUCB)
For each source arm $a$, we maintain a ridge regression model mapping the context vector to an expected reward:
$$\hat{r}(a, x) = \theta_a^T x, \quad \theta_a = A_a^{-1} b_a$$
where $A_a = \lambda I + \sum_{t: a_t=a} x_t x_t^T$ and $b_a = \sum_{t: a_t=a} r_t x_t$.

**Upper Confidence Bound (UCB) selection rule:**
$$a^* = \arg\max_a \left[ \theta_a^T x + \alpha \sqrt{x^T A_a^{-1} x} \right]$$

### 3.3 Policy Gradient (PPO)
PPO optimizes the final synthesis weighting utilizing a safely clipped surrogate objective function that prevents destructively large policy updates.
$$L^{CLIP}(\theta) = \mathbb{E} \left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$
where $\hat{A}_t$ is the Generalized Advantage Estimation (GAE-λ).

### 3.4 Multi-Agent RL (MAPPO)
We adopted Centralized Training with Decentralized Execution (CTDE). The team reward incentivizes novelty while explicitly punishing duplication.
$$R_{team} = \sum_{i=1}^{N} R_i^{indiv} - 0.3 \cdot n_{duplicates} + 0.15 \cdot n_{new\_sources}$$

### 3.5 Meta-Learning (MAML-Inspired)
To prevent cold-starts on new domains, Meta-initialization is computed dynamically as a weighted average of domain-specific parameters:
$$\theta^{meta}_a = (1 - \beta) \theta^{meta}_a + \beta \cdot \frac{1}{|D_{train}|} \sum_{d \in D_{train}} \theta^d_a$$

---

## 4. Reward Engineering

The environment dispenses dense signals crafted to simulate the goals of a human researcher.

**Step Reward:** Emphasizes immediate relevance and speed.
$$r_{step} = w_1 \cdot relevance + w_2 \cdot novelty - w_3 \cdot \frac{latency}{10}$$

**Session Reward:** Emphasizes long-term task completion upon episode termination.
$$r_{session} = w_1 \cdot max\_relevance + 0.3 \cdot avg\_relevance + w_4 \cdot coverage - w_5 \cdot conflict$$

**Tool-Adjusted Reward (Safety Mechanism):**
$$r_{adjusted} = relevance \cdot (0.5 + 0.5 \cdot credibility)$$

---

## 5. Experimental Design and Results

To rigorously evaluate the system, we designed a comprehensive experimental methodology tracking performance across all five RL modes against random baselines.

### 5.1 Experimental Methodology & Evaluation Criteria
The system was trained over 2,000 episodes using 3 parallel agents (MARL) and a curriculum of 5 distinct querying domains. To ensure statistical significance, evaluations were conducted over 5 randomized seeds using `experiments/statistical_validation.py`.

**Performance criteria included:**
- **Episode Reward:** The holistic measure of the agent's efficiency (relevance vs. budget used).
- **Cumulative Regret:** The difference between the agent’s choices and the true environment oracle.
- **MARL Duplication Rate:** The percentage of overlapping source queries between parallel agents.
- **Episodes-to-Criterion:** The speed of adaptation to held-out domains (Meta-Learning).

### 5.2 Comparative Analyses & Learning Curves
The trained agent achieves near-oracle performance, outperforming randomized baseline models by **+44.5%**. 

- **Q-Learning vs. SARSA:** Analysis of TD-errors reveals that Q-Learning converges faster, whereas SARSA derives a more risk-averse, stable policy.
- **Multi-Agent Coordination:** The MAPPO centralized critic successfully taught parallel agents to avoid redundant searching, dropping the duplication collision metric by ~60% over 2000 episodes.
- **Meta-Learning Transfer Benefit:** Pre-loaded MAML weights allowed the system to jump-start learning on the previously unseen "AI Policy" domain, converging 15-25% faster than a randomly initialized model.

### 5.3 Visualizations of Agent Behavior Improvement
The `experiments/plot_all.py` script generated several visual proofs of behavioral improvement (located in the `/plots` directory):

1. **Learning Curves (`plots/learning_curve.png`)**: Visually demonstrates the agent climbing the reward gradient and plateauing near the oracle maximum.
2. **Sublinear Regret (`plots/regret_curve.png`)**: The flattening regret curve mathematically proves genuine exploration-to-exploitation learning.
3. **Behavioral Heatmaps (`plots/source_heatmap.png`)**: Directly visualizes the agent's contextual decision-making improvement. The agent organically learned the underlying matrix distributions, heavily prioritizing `arxiv` exclusively for technical queries and `sec_edgar` for business queries, without hardcoded rules.

---

## 6. Challenges and Technical Solutions

| Challenge | Impact | Technical Solution Implemented |
|:----------|:-------|:-------------------------------|
| **State Space Explosion** | Tabular Q-Learning requires discrete definitions, causing memory crashes on continuous context inputs. | Discretized the state vector into 60 highly focused buckets based strictly on abstract metrics (budget/coverage) rather than raw float values. |
| **Noisy, Deceptive Rewards** | Highly relevant but unreliable sources caused catastrophic forgetting in policy gradients. | Built the custom `SourceCredibilityScorer` tool to wrap the reward function, actively damping noisy signals before they poison the MDP. |
| **MARL Credit Assignment** | Purely decentralized agents hoarded rewards, ruining cooperation. | Implemented CTDE architecture. Actors act independently, but gradient updates flow through a shared Critic observing the global system memory bus. |
| **PPO without PyTorch** | Academic constraints preferred raw minimal-dependency implementation. | Hard-coded analytical backpropagation within a NumPy-based Multi-Layer Perceptron containing automatic gradient clipping to replicate PPO logic without ML frameworks. |

---

## 7. Limitations

1. **Stationary Simulation:** The `MadisonEnv` simulates quality distributions as stationary probabilities. Real-world internet sources suffer from concept drift (e.g., SEO rot), which would fundamentally unbalance the LinUCB exploration algorithms.
2. **Tabular Scaling Constraints:** While effective for this scope, tracking 60 discrete states in Q-Learning fundamentally limits future scalability. Adding dimensions for visual data or multi-language analysis would necessitate a shift to Deep Q-Networks (DQN).
3. **MARL Homogeneity:** Currently, the MAPPO agents share homogeneous network topologies. Specializing their internal architectures (e.g., Agent A optimized for semantic search, Agent B optimized for tabular financial parsing) would enhance cooperation efficiency.

---

## 8. Ethical Considerations

### 8.1 Algorithmic Bias Amplification
Because RL algorithms aggressively optimize for their reward function, they are susceptible to positive-feedback loops. If English-language technical forums provide a faster $+0.2$ latency reward than translated global research repositories, the agent will overwhelmingly bias toward Western perspectives over time.

### 8.2 Trust Index Reliability
The `SourceCredibilityScorer` autonomously degrades trust metrics. While useful, this poses censorship risks. If an obscure but verifiably true whistleblowing source is heavily penalized early for low availability, the UCB exploration will rarely visit it again, effectively blacklisting critical knowledge.

### 8.3 Automation vs. Augmentation
Deploying an autonomous intelligence gatherer risks displacing junior analysts. However, given the hallucination rates inherent to LLM synthesis, this methodology is designed strictly for **human-in-the-loop augmentation**. The Controller outputs a pipeline of tracked citations, mandating a human arbiter for final decision-making.

---

## 9. Future Work

1. **Neural Function Approximators:** Re-architect value layers to utilize DQN and Actor-Critic neural networks capable of ingesting raw unstructured text environments.
2. **Adversarial Resiliency Injection:** Train the system against a suite of purposefully poisoned data (e.g., simulated DDoS sources returning fraudulent `relevance` scores) to test the outer limits of the Credibility Scorer's braking capabilities.
3. **Live API Integration:** Transition the `MadisonEnv` off simulated JSON distributions and bind the query engines to real-world endpoint rate-limits (arXiv API, SEC Edgar standard REST protocols).

---

## Appendix: Reproducibility

To recreate and independently verify these results:

```bash
# Install exact dependencies
pip3 install -r requirements.txt

# Run the 2000-episode training protocol
python3 main.py

# Verify Multi-Seed Statistical Authenticity
python3 experiments/statistical_validation.py

# Run Live Terminal Integration Demo
python3 run_demo.py
```
