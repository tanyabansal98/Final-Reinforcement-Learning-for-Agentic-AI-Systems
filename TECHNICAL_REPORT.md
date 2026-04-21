# Technical Report: Reinforcement Learning for Agentic AI Systems
**Project:** Madison RL — Autonomous AI Industry Intelligence Orchestrator  
**Framework:** Humanitarians.AI Madison Integration  
**Author:** Tanya Bansal  
**Date:** April 2026

---

## 1. Executive Summary

This architecture enhances the Humanitarians.AI **Madison** intelligence framework by retrofitting it with a sophisticated, five-layer Reinforcement Learning (RL) pipeline. In the modern AI landscape, mission-critical information is deeply fractured and highly volatile. Technical benchmarks sit on arXiv, business fundamentals reside in SEC EDGAR filings, and open-source releases are obfuscated within GitHub commit feeds. 

Instead of relying on rigid, deterministic heuristics (e.g., `if query == 'technical' then route to arXiv`), this framework introduces a fully autonomous agent capable of discovering optimal information-gathering topologies strictly from environmental feedback. Operating within a custom OpenAI-Gymnasium compliant environment spanning 12 simulated sources and 5 query domains, the agent coordinates parallel sub-agents via MAPPO, optimizes sequential workflows using Q-Learning, adapts contextually via LinUCB, synthesizes multi-source vectors with PPO, and generalizes to novel research paradigms using Meta-Learning (MAML). Over 2,000 training episodes, the system verifiably approaches near-oracle performance under simulated conditions, yielding a **+44.5% efficiency improvement** over stochastic baseline routing.

---

## 2. System Architecture & Integration

The system operates not as a monolithic algorithm, but as a modular pipeline of specialized, cooperating agents. This layered topography allows the central orchestrator to manage multiple distinct ML workflows simultaneously.

![System Architecture Diagram](architecture_diagram.svg)

### 2.1 The Agentic Pipeline
The controller manages a seamless state relay between the following specialized layers:

1. **DomainAdapter (Meta-Learning):** Instantiated via `agents/meta_learner.py`. It evaluates the macro subject domain (e.g., "AI Policy") and computes warm-start prior distributions for downstream stochastic bandits.
2. **SourceSelector (Contextual Bandits):** Instantiated via `agents/contextual_bandit.py`. Given the query's budget and topic constraint (context vector), it navigates exploration-exploitation tradeoffs to select the highest-expected-value source arm.
3. **SessionPlanner (Value-Based RL):** Instantiated via `agents/q_learning.py`. This agent observes the SourceSelector's output state. It treats the intelligence gathering process as a sequential Markov Decision Process (MDP), optimizing multi-step reasoning.
4. **CoordinationAgent (Multi-Agent RL):** Instantiated via `agents/marl_coordinator.py`. For high-budget queries, three identical parallel agents deploy. The Coordinator actively penalizes duplicate API calls across agents via a Centralized Critic framework.
5. **SynthesisAgent (Policy Gradient):** Instantiated via `agents/ppo_agent.py`. Once all sources return vectors, PPO employs analytical gradients to assign probabilistic confidence weights for precise final synthesis.

### 2.2 Custom Tool Integration: Epistemic Safety & Credibility Scoring
In real-world data environments, relevance does not equate to truth—a viral tweet may be highly relevant to a query but entirely hallucinatory. To counteract reward hacking, we developed the `SourceCredibilityScorer` tool (`tools/source_credibility_scorer.py`). 

* **Mechanism:** It operates as an active safety/alignment mechanism, continuously monitoring five dynamic parameters per source: *Accuracy, Consistency, Timeliness, Availability, and Peer-Conflict Rate*.
* **Targeted Reward Interception:** Instead of allowing the orchestrator to ingest raw environment rewards blindly, the Scorer dynamically intercepts the signal. The scorer directly modifies the agent's action space via trust-based masking and reshapes the reward signal, ensuring both safe exploration and reliable exploitation. This acts as an autonomous negative reinforcement loop, organically teaching the bandit algorithms to avoid untrustworthy databases for critical queries without explicit human programming.

---

## 3. Mathematical Formulations & Explainability

### 3.1 Value-Based Learning (Q-Learning / SARSA)
The `SessionPlanner` maintains a tabular Q-function Q(s, a) to map sequential workflows. High-dimensional data is cleanly discretized into 60 semantic states.
**State space:** s = (query_type, budget_bucket, coverage_bucket)
**Action space:** a ∈ {0, 1, ..., 11} (12 sources)

**Q-Learning Update (Off-Policy):**
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

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

### 3.5 System Explainability 
Unlike black-box LLM deployments, this RL pipeline is fundamentally interpretable.
1. **Feature Attribution:** The LinUCB Ridge Regression matrix ($\theta_a$) allows analysts to extract exactly *why* an agent selected an API based purely on contextual feature weights.
2. **Q-Table Transparency:** The MDP state space is explicitly printed at episode termination, providing a human-readable trace of the agent's sequential logic and demonstrating exactly when it chose to shift from exploration to pure exploitation.

```markdown
Example Decision Trace:

Query: "Latest AI benchmarks"

Chosen Source: arXiv
Expected Reward: 0.91
Credibility: 0.88
Reason: High historical relevance + high trust + not yet queried in session
```

---

## 4. Reward Engineering

The environment dispenses dense signals crafted to simulate the objectives of an elite human researcher.

**Step Reward:** Emphasizes immediate relevance and speed.
$$r_{step} = w_1 \cdot relevance + w_2 \cdot novelty - w_3 \cdot \frac{latency}{10}$$

**Session Reward:** Emphasizes comprehensive task completion upon episode termination.
$$r_{session} = w_1 \cdot max\_relevance + 0.3 \cdot avg\_relevance + w_4 \cdot coverage - w_5 \cdot conflict$$

**Tool-Adjusted Reward (The Safety Mechanism):**
$$r_{adjusted} = relevance \cdot (0.5 + 0.5 \cdot credibility)$$

---

## 5. Experimental Design and Baseline Comparisons

To rigorously evaluate the system, we executed a comprehensive methodology tracking performance across all five RL topologies against randomized controls.

### 5.1 The "Before & After" Baseline Comparison
To prove learning efficacy, we forced a completely untrained, randomized baseline orchestrator to compete side-by-side against our fully converged RL pipeline over thousands of evaluative runs (`experiments/compare_baselines.py`). 

| Metric | "Before" (Untrained Baseline) | "After" (Fully Trained RL Pipeline) | Result Impact |
| :--- | :--- | :--- | :--- |
| **Reward Yield** | 3.444 | **4.976** | **+44.5% Efficiency Jump** |
| **Duplication Collisions** | High | Near-Zero | **60% Reduction in Redundancy** |
| **State Knowledge** | Zero (Amnesiac) | Oracle Matched | Optimal Exploitation achieved |

### 5.2 Multi-Seed Analytical Insights
The framework underwent rigorous multi-seed statistical validation (5 seeds) to derive 95% confidence intervals, proving mathematically significant behavioral milestones:

- **MAPPO Synergy:** The centralized critic successfully taught parallel agents to avoid redundant searching, dropping the duplication collision metric by ~60% over 2000 episodes.
- **Q-Learning vs. SARSA:** Analysis of TD-errors reveals that Q-Learning converges upon an optimal policy much faster, whereas SARSA derives a more risk-averse, highly stable sequence matrix.
- **Meta-Learning Transfer Speed:** Pre-loaded MAML weights allowed the system to jump-start adapting to the previously unseen "AI Policy" domain, converging 15-25% faster than uniformly initialized peers.

### 5.3 Visualizations of Policy Convergence
The `experiments/plot_all.py` script generated several visual proofs of behavioral improvement (located in the `/plots` directory):
1. **Sublinear Regret (`plots/regret_curve.png`)**: The flattening regret curve mathematically proves genuine exploration-to-exploitation transition.
2. **Behavioral Heatmaps (`plots/source_heatmap.png`)**: Directly visualizes the contextual decision-making improvement. The agent organically learned the underlying matrix distributions, heavily prioritizing `arxiv` exclusively for technical queries and `sec_edgar` for business queries, entirely without hardcoded deterministic rules.

---

## 6. Challenges and Technical Solutions

| Challenge | Impact | Technical Solution Implemented |
|:----------|:-------|:-------------------------------|
| **State Space Exhaustion** | Tabular Q-Learning requires discrete definitions, causing memory permutation crashes on continuous inputs. | Discretized the state vector into 60 minimal, semantic abstraction buckets (budget/coverage constraints). |
| **Corrupted Signal Loops** | Highly relevant but untrustworthy sources caused catastrophic forgetting in policy gradients due to reward hacking. | Engineered the custom `SourceCredibilityScorer` tool to wrap the reward function, aggressively clipping noisy signals before they poisoned the MDP. |
| **Credit Assignment Failures** | Purely decentralized agents hoarded rewards, ruining parallel cooperation. | Implemented CTDE architecture. Actors act independently, but gradient updates flow through a shared Critic observing the global system memory bus. |
| **PPO PyTorch Bloat** | Academic constraints preferred raw minimal-dependency implementation for fast-edge deployment. | Hard-coded analytical backpropagation within a NumPy-based Multi-Layer Perceptron containing automatic gradient clipping to replicate exact PPO boundaries. |

---

## 7. Limitations

1. **Stationary Simulation Bias:** `MadisonEnv` simulates quality distributions as stationary probabilities. Real-world internet topologies suffer from persistent concept drift (e.g., SEO rot), which could eventually unbalance the LinUCB exploitation vectors if left un-updated.
2. **Tabular Constraints:** While highly effective for this scope, tracking 60 discrete states in Q-Learning limits multi-dimensional scalability. Handling unstructured video or multilingual analysis would require upgrading the orchestration layer to Deep Q-Networks (DQN).
3. **MARL Homogeneity:** Currently, the MAPPO agents share homogeneous network topologies. Specializing their internal model architectures based on modality (e.g., semantic search vs. financial data parsing) would yield compounding cooperative efficiencies.

---

## 8. Ethical Considerations

### 8.1 Algorithmic Bias Amplification
Because RL algorithms aggressively optimize for their reward function, they are deeply susceptible to positive-feedback loops. If English-language technical forums provide a faster $+0.2$ latency reward than translated global research repositories, the agent will overwhelmingly bias toward Western perspectives over time.

### 8.2 Epistemic Trust Index Reliability
The `SourceCredibilityScorer` autonomously degrades trust metrics. While useful for alignment, this poses downstream censorship risks. If an obscure but verifiably true whistle-blowing API is heavily penalized early for low availability, the UCB exploration parameter will rarely visit it again, effectively blacklisting critical intelligence.

### 8.3 Automation vs. Augmented Control
Autonomous intelligence gathering risks displacing junior human analysts. However, given the hallucination rates inherent to LLM synthesis layers, this methodology is designed strictly for **human-in-the-loop augmentation**. The Controller outputs a traceable, probabilistic pipeline of citations, mandating a human arbiter for final decision-making.

---

## 9. Future Work

1. **Neural Function Approximators:** Re-architect value layers to utilize DQN and scalable Actor-Critic neural networks capable of ingesting raw, unstructured linguistic vectors from the live internet.
2. **Adversarial Resiliency Injection:** Validate the system against a suite of purposefully poisoned data pipelines (e.g., simulated DDoS sources returning fraudulent `relevance` scores) to test the outer boundaries of the Credibility Scorer's braking algorithm.
3. **Live API Integration:** Transition the `MadisonEnv` off simulated JSON distributions and natively bind the query engines to real-world rate-limited endpoints (arXiv API, SEC Edgar REST protocols).

---

## Appendix: Reproducibility

To authentically recreate and independently execute these experiments:

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
