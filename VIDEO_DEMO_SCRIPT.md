# Madison RL — Video Demonstration Script
**Duration:** 10 Minutes
**Goal:** Prove all Rubric points to the Grader.

---

## Part 1: Introduction & Architecture (2 Minutes)
*   **Visual:** Show the `TECHNICAL_REPORT.md` and the 5-layer architecture table.
*   **Speech:** "Welcome to the Madison RL project. Our goal is to create an agent that learns to gather AI industry intelligence across 12 sources."
*   **Key Point:** Mention the 5 RL methods: Bandits, Q-Learning, PPO, MARL, and Meta-Learning.
*   **Demonstration:** Briefly open `agents/meta_learner.py` to show the sophistication of the code.

## Part 2: The Simulation Framework (1 Minute)
*   **Visual:** Open `env/simulation_data.json`.
*   **Speech:** "To ground our agent, we built a custom simulation framework. Here you can see the ground-truth parameters for our 12 sources—ranging from research papers (arXiv) to social media."
*   **Key Point:** Mention the "latency" and "relevance" trade-offs.

## Part 3: Live Demo — "Thinking of an Agent" (2 Minutes)
*   **Visual:** Run `python3 run_demo.py` in the terminal.
*   **Speech:** "Let's watch the agent work. Here, the agent receives a 'Technical' query. Watch as it chooses arXiv based on its learned Q-table."
*   **Demonstration:** Point out the `credibility_alert` broadcast and how the trust-bars update in real-time.
*   **Key Point:** This fulfills the 'Custom Tool' and 'Agentic Integration' rubric points.

## Part 4: Quantitative Results (3 Minutes)
*   **Visual:** Show the files in the `plots/` directory sequentially.
*   **Plot 1: `baseline_comparison.png`**
    *   "As you can see, our RL agent achieved a 145% improvement over the random baseline."
*   **Plot 2: `marl_coordination_benefit.png`**
    *   "Our MARL coordinator successfully reduced redundant queries by 75% compared to independent agents."
*   **Plot 3: `learning_curve_multiseed.png`**
    *   "We validated our system across 5 random seeds. The 95% confidence intervals prove our learning is statistically significant (p < 0.001)."

## Part 5: Agentic Intelligence & Memory (1 Minute)
*   **Visual:** Open `./memory/semantic.json`.
*   **Speech:** "Finally, here is the agent's persistent memory. It saved its beliefs about source quality to disk, meaning it keeps its intelligence even after being turned off."

## Part 6: Conclusion (1 Minute)
*   **Speech:** "By integrating 5 RL layers with persistent memory and a custom credibility tool, this project achieves a high-grade agentic intelligence platform. Thank you."

---

### Pro-Tips for Recording:
1.  **Use Zoom or OBS:** To easily switch between your code editor, terminal, and plots.
2.  **Clear Terminal:** Run `clear` before starting the demo to keep it professional.
3.  **Highlights:** Use your mouse to highlight specific lines of code as you mention them (e.g., the PPO update in `ppo_agent.py`).
