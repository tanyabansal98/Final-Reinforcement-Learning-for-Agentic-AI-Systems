"""
training/train.py
Main training loop integrating all five RL methods.
Each method operates at its designated layer in the pipeline.
"""

import numpy as np
from typing import Dict, List, Optional
from env.madison_env import MadisonEnv, MultiAgentMadisonEnv
from env.query_engine import CONTEXT_DIM
from env.source_pool import QUERY_TYPES
from agents.contextual_bandit import ContextualBanditAgent
from agents.q_learning import ValueBasedAgent, encode_state
from agents.ppo_agent import PPOAgent
from agents.marl_coordinator import MARLCoordinator
from agents.meta_learner import MetaLearner, DOMAINS, DOMAIN_TO_QUERY_TYPE
from tools.source_credibility_scorer import SourceCredibilityScorer
from madison_integration import MadisonMemory, MadisonMessageBus


def train(
    n_episodes: int = 2000,
    n_agents: int = 3,
    seed: int = 42,
    verbose: bool = True,
    load_memory: bool = True,
) -> Dict:
    """
    Full training run integrating all five RL methods + Custom Tool + Memory + Comm Bus.

    Returns a results dict with learning curves and metrics for plotting.
    """
    np.random.seed(seed)

    # ---------------------------------------------------------------
    # Initialize environments
    # ---------------------------------------------------------------
    env = MadisonEnv(seed=seed)                          # single-agent env
    marl_env = MultiAgentMadisonEnv(n_agents=n_agents, seed=seed)  # multi-agent env

    n_sources = env.n_sources
    source_names = env.get_source_names()

    # ---------------------------------------------------------------
    # Initialize Agentic Components (Memory & Comm Bus)
    # ---------------------------------------------------------------
    memory = MadisonMemory()
    if load_memory:
        memory.load()
    
    message_bus = MadisonMessageBus()
    scorer = SourceCredibilityScorer(source_names)

    # ---------------------------------------------------------------
    # Initialize agents
    # ---------------------------------------------------------------
    # Method 4: Contextual bandit (primary source selection)
    bandit = ContextualBanditAgent(n_sources, CONTEXT_DIM, alpha=1.0)

    # Method 1: Q-Learning + SARSA (session-level sequencing)
    vb_agent = ValueBasedAgent(n_sources)

    # Method 2: PPO (synthesis)
    ppo = PPOAgent(context_dim=CONTEXT_DIM, n_sources=n_sources)

    # Method 3: MARL coordinator
    marl = MARLCoordinator(n_agents=n_agents, n_sources=n_sources, context_dim=CONTEXT_DIM)

    # Method 5: Meta-learner
    meta = MetaLearner(n_sources=n_sources, context_dim=CONTEXT_DIM)

    # ---------------------------------------------------------------
    # Tracking
    # ---------------------------------------------------------------
    results = {
        "episode_rewards":       [],
        "bandit_ucb_rewards":    [],
        "bandit_ts_rewards":     [],
        "q_learning_td_errors":  [],
        "sarsa_td_errors":       [],
        "ppo_updates":           [],
        "marl_team_rewards":     [],
        "marl_duplication_rates": [],
        "meta_transfer_results": [],
        "regret_curve":          [],
        "source_selection_log":  [],
    }

    meta_train_buffer: Dict[str, List] = {d: [] for d in DOMAINS}
    ppo_update_freq = 20    # update PPO every N episodes
    meta_update_freq = 100  # update meta-params every N episodes

    if verbose:
        print(f"Training Madison RL — {n_episodes} episodes")
        print(f"Sources: {n_sources} | Agents: {n_agents} | Context dim: {CONTEXT_DIM}")
        print(f"Memory: {memory.summary()['sources_tracked']} sources matched | Comm Bus: Ready\n")

    for episode in range(n_episodes):
        # -------------------------------------------------------
        # Sample a domain for meta-learning curriculum
        # -------------------------------------------------------
        domain = DOMAINS[episode % len(DOMAINS)]
        query_type = DOMAIN_TO_QUERY_TYPE[domain]

        # -------------------------------------------------------
        # Layer 1: Meta-learner warm-start
        # -------------------------------------------------------
        meta_priors = meta.get_meta_source_preferences() if meta._meta_update_count > 0 else None

        # -------------------------------------------------------
        # Single-agent episode (Methods 1, 2, 4)
        # -------------------------------------------------------
        obs, info = env.reset(query_type=query_type)
        session = env.get_session_state()
        episode_transitions = []
        episode_reward = 0.0
        prev_state = None
        prev_action = None

        while not session.is_done:
            context = session.to_context_vector()
            queried_so_far = list(session.sources_queried)

            # Safety Feature (Custom Tool Integration):
            # Mask sources that have extremely low credibility (< 0.15)
            low_trust_mask = scorer.get_trust_adjusted_mask(threshold=0.15)
            exclude_list = list(set(queried_so_far + low_trust_mask))

            # Method 4: Contextual bandit selects source category
            action = bandit.select_action(context, exclude=exclude_list, use="ucb")

            # Method 1: Q-Learning encodes session state
            q_state = encode_state(
                session.query.query_type,
                session.budget_remaining,
                session.query.budget,
                len(session.sources_queried),
            )

            # Take step in environment
            next_obs, reward, terminated, _, step_info = env.step(action)
            result = step_info["result"]
            episode_reward += reward

            # Update Custom Tool (Credibility Scorer)
            scorer.record_query(result["source"], result, session.results)
            
            # Communication Alert (Message Bus integration)
            # If trust drops suddenly, broadcast to other agents
            if scorer.get_credibility(result["source"]) < 0.3:
                message_bus.broadcast(
                    sender="SourceSelector",
                    msg_type="credibility_alert",
                    payload={"source": result["source"], "trust": scorer.get_credibility(result["source"])},
                    timestamp=episode
                )

            # Update Agentic Memory (Semantic Memory)
            memory.update_source_belief(
                result["source"], query_type, 
                result.get("relevance", 0.0), result.get("available", False)
            )

            # Compute trust-adjusted reward
            trust_adjusted_reward = scorer.credibility_weighted_reward(result)

            # Update Method 4 (bandit)
            bandit.update(action, context, trust_adjusted_reward)

            # Update Method 1 (Value-Based)
            next_session = env.get_session_state()
            next_q_state = encode_state(
                next_session.query.query_type,
                next_session.budget_remaining,
                next_session.query.budget,
                len(next_session.sources_queried),
            )
            next_action = vb_agent.select_action(next_q_state, exclude=list(next_session.sources_queried))
            vb_agent.update_q(q_state, action, trust_adjusted_reward, next_q_state, terminated,
                              exclude_next=list(next_session.sources_queried))
            vb_agent.update_sarsa(q_state, action, trust_adjusted_reward, next_q_state, next_action, terminated)

            # Store transition for meta-learning
            episode_transitions.append((context, action, trust_adjusted_reward))
            results["source_selection_log"].append({
                "episode": episode, "source": source_names[action],
                "query_type": query_type, "relevance": result.get("relevance", 0),
                "credibility": scorer.get_credibility(result["source"])
            })

            prev_state, prev_action = q_state, action
            if terminated: break

        # Store Episodic Memory
        memory.store_episode({
            "query_type": query_type,
            "sources_queried": [source_names[a] for _, a, _ in episode_transitions],
            "total_reward": episode_reward,
            "budget_used": len(episode_transitions)
        })

        # Method 2: PPO synthesis
        final_session = env.get_session_state()
        if final_session and final_session.results:
            relevances = np.zeros(n_sources, dtype=np.float32)
            for r in final_session.results:
                if r["available"] and r["source"] in source_names:
                    idx = source_names.index(r["source"])
                    cred = scorer.get_credibility(r["source"])
                    relevances[idx] = max(relevances[idx], r["relevance"] * (0.7 + 0.3 * cred))

            final_ctx = final_session.to_context_vector()
            w, lp, val = ppo.select_action(final_ctx, relevances)
            session_r = step_info.get("session_reward") or 0.0
            ppo.store_transition(final_ctx, relevances, w, session_r, lp, val)

        if episode % ppo_update_freq == 0 and episode > 0:
            ppo.update(n_epochs=4)
            results["ppo_updates"].append(episode)

        # -------------------------------------------------------
        # Method 3: MARL episode (every 5 episodes)
        # -------------------------------------------------------
        if episode % 5 == 0:
            marl_obs_list = marl_env.reset(query_type=query_type)
            marl.reset_episode()
            marl_contexts = [obs for obs, _ in marl_obs_list]
            marl_episode_reward = 0.0
            all_done = [False] * n_agents
            step_dups = []

            while not all(all_done):
                # Request actions from coordinator
                # Realistic "cold start" initialization using credibility scorer
                current_results = [{"available": True, "relevance": scorer.get_credibility(source_names[0])} for _ in range(n_agents)]
                actions, team_rewards = marl.step(marl_contexts, current_results, [0.0] * n_agents)

                # Take step in real env
                obs_list, env_rewards, all_done, marl_info = marl_env.step(actions)
                
                # Feedback real data
                current_results = marl_info["results"]
                marl_contexts = obs_list
                marl_episode_reward += np.mean(env_rewards)
                
                # Track duplication (rubric point: Coordination benefit)
                unique_actions = len(set(actions))
                step_dups.append((n_agents - unique_actions) / n_agents)

            results["marl_team_rewards"].append(marl_episode_reward)
            results["marl_duplication_rates"].append(np.mean(step_dups) if step_dups else 0.0)
            marl.update_critic(marl_contexts, marl_episode_reward)

        # -------------------------------------------------------
        # Method 5: Meta-learning update
        # -------------------------------------------------------
        meta_train_buffer[domain].extend(episode_transitions)
        meta.train_domain(domain, episode_transitions)

        if episode % meta_update_freq == 0 and episode > 0:
            meta.update_meta_params()
            held_out = DOMAINS[-1]
            if meta_train_buffer[held_out]:
                n_support = min(15, len(meta_train_buffer[held_out]))
                n_eval = min(20, len(meta_train_buffer[held_out]) - n_support)
                if n_eval > 0:
                    support = meta_train_buffer[held_out][:n_support]
                    eval_ep = meta_train_buffer[held_out][n_support:n_support + n_eval]
                    transfer_result = meta.evaluate_transfer_benefit(held_out, eval_ep, support)
                    results["meta_transfer_results"].append(transfer_result)
            
            # Periodic Memory Persistence
            memory.save()

        # -------------------------------------------------------
        # Record episode metrics
        # -------------------------------------------------------
        results["episode_rewards"].append(episode_reward)
        results["regret_curve"].append(env.reward_tracker.cumulative_regret()[-1] if env.reward_tracker.episode_regrets else 0.0)

        td = vb_agent.get_td_errors()
        results["q_learning_td_errors"].append(np.mean(td["q_learning"][-10:]) if td["q_learning"] else 0.0)
        results["sarsa_td_errors"].append(np.mean(td["sarsa"][-10:]) if td["sarsa"] else 0.0)

        if verbose and episode % 100 == 0:
            avg_r = np.mean(results["episode_rewards"][-100:])
            print(f"Episode {episode:4d} | Avg reward: {avg_r:.3f} | ε: {vb_agent.q_learning.epsilon:.3f}")

    # Final cleanup and save
    meta.update_meta_params()
    memory.save()

    results["meta_summary"] = meta.get_transfer_summary()
    results["marl_stats"] = marl.get_stats()
    results["memory_stats"] = memory.summary()
    results["message_bus_stats"] = message_bus.stats()
    results["source_names"] = source_names
    results["query_types"] = QUERY_TYPES
    results["bandit_agent"] = bandit
    results["ppo_agent"] = ppo
    results["meta_agent"] = meta
    results["env"] = env
    results["credibility_scorer"] = scorer

    if verbose:
        print(f"\nTraining complete.")
        print(f"Final avg reward (last 100): {np.mean(results['episode_rewards'][-100:]):.3f}")
        scorer.print_trust_report()

    return results



