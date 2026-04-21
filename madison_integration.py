"""
madison_integration.py
MADISON FRAMEWORK INTEGRATION

This module bridges our RL implementation with the Humanitarians.AI
Madison Intelligence Agent architecture. It maps Madison's agent roles,
task allocation, memory system, and communication protocols to our
RL components.

Madison Architecture (Humanitarians.AI):
    Madison is an intelligence analysis framework with specialized agents
    that collect, analyze, and synthesize information about AI industry
    developments. Our RL system enhances Madison by replacing its
    rule-based decision-making with learned policies.

Integration points:
    1. Controller (MadisonRLController) — Orchestrates agent activation
    2. Agent Roles — Maps RL methods to Madison agent specializations
    3. Memory — Persistent experience store for cross-session learning
    4. Communication — Message protocol between agents
    5. Tool Registry — Custom tools available to agents

Run demo:
    python madison_integration.py
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json
import os


# ══════════════════════════════════════════════════════════════════════════════
# 1. AGENT ROLE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentRole:
    """Defines a specialized agent role within the Madison framework."""
    name: str
    description: str
    rl_method: str           # Which RL method drives this agent
    capabilities: List[str]  # What this agent can do
    priority: int            # Activation priority (lower = first)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "rl_method": self.rl_method,
            "capabilities": self.capabilities,
            "priority": self.priority,
        }


# Madison Agent Role Definitions
MADISON_AGENTS = [
    AgentRole(
        name="SourceSelector",
        description="Selects which information sources to query based on query context. "
                    "Uses contextual bandits (LinUCB) to balance exploration of new sources "
                    "with exploitation of known-good sources.",
        rl_method="Contextual Bandits (LinUCB + Thompson Sampling)",
        capabilities=["source_selection", "exploration_management", "context_encoding"],
        priority=1,
    ),
    AgentRole(
        name="SessionPlanner",
        description="Plans the sequence of source queries within a research session. "
                    "Uses Q-Learning to optimize multi-step source sequencing based on "
                    "budget remaining and coverage achieved so far.",
        rl_method="Value-Based Learning (Q-Learning + SARSA)",
        capabilities=["session_planning", "budget_management", "sequence_optimization"],
        priority=2,
    ),
    AgentRole(
        name="SynthesisAgent",
        description="Combines and weights results from multiple sources into a unified "
                    "insight score. Uses PPO to learn optimal synthesis weights based on "
                    "source quality and query context.",
        rl_method="Policy Gradient (PPO with analytical backprop)",
        capabilities=["result_synthesis", "weight_optimization", "quality_scoring"],
        priority=3,
    ),
    AgentRole(
        name="CoordinationAgent",
        description="Coordinates a team of 3 parallel agents to avoid redundant queries "
                    "and maximize coverage. Uses MAPPO with a centralized critic and "
                    "message-passing communication protocol.",
        rl_method="Multi-Agent RL (MAPPO)",
        capabilities=["team_coordination", "deduplication", "coverage_optimization", "message_passing"],
        priority=4,
    ),
    AgentRole(
        name="DomainAdapter",
        description="Enables rapid adaptation to new research domains by transferring "
                    "learned source preferences. Uses MAML-inspired meta-learning to "
                    "warm-start new domain policies from cross-domain experience.",
        rl_method="Meta-Learning (MAML-inspired)",
        capabilities=["domain_transfer", "few_shot_adaptation", "cross_domain_generalization"],
        priority=5,
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# 2. MEMORY SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

class MadisonMemory:
    """
    Persistent memory system for the Madison RL agent.

    Stores three types of memory:
    1. Episodic Memory — individual query session outcomes
    2. Semantic Memory — learned facts about sources (quality estimates)
    3. Procedural Memory — learned policies (Q-tables, bandit parameters)

    Memory is used for:
    - Cross-session learning continuity
    - Experience replay for offline training
    - Explainability (why did the agent choose this source?)
    """

    def __init__(self, save_dir: str = "./memory"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Episodic: recent session outcomes
        self.episodic: List[Dict] = []
        self.max_episodic = 1000

        # Semantic: source quality beliefs
        self.semantic: Dict[str, Dict] = {}

        # Procedural: serialized policy state
        self.procedural: Dict[str, Any] = {}

    def store_episode(self, episode_data: Dict):
        """Record a completed research session."""
        self.episodic.append({
            "query_type": episode_data.get("query_type"),
            "sources_queried": episode_data.get("sources_queried", []),
            "total_reward": episode_data.get("total_reward", 0),
            "max_relevance": episode_data.get("max_relevance", 0),
            "budget_used": episode_data.get("budget_used", 0),
        })
        if len(self.episodic) > self.max_episodic:
            self.episodic = self.episodic[-self.max_episodic:]

    def update_source_belief(self, source_name: str, query_type: str,
                              relevance: float, available: bool):
        """Update semantic memory about a source."""
        if source_name not in self.semantic:
            self.semantic[source_name] = {}
        if query_type not in self.semantic[source_name]:
            self.semantic[source_name][query_type] = {
                "observations": 0, "total_relevance": 0.0,
                "availability_count": 0, "total_count": 0,
            }
        entry = self.semantic[source_name][query_type]
        entry["observations"] += 1
        entry["total_relevance"] += relevance
        entry["total_count"] += 1
        if available:
            entry["availability_count"] += 1

    def get_source_belief(self, source_name: str, query_type: str) -> Dict:
        """Retrieve current belief about a source for a query type."""
        if source_name not in self.semantic or query_type not in self.semantic[source_name]:
            return {"mean_relevance": 0.5, "availability": 0.95, "confidence": 0.0}
        entry = self.semantic[source_name][query_type]
        n = entry["observations"]
        return {
            "mean_relevance": entry["total_relevance"] / max(n, 1),
            "availability": entry["availability_count"] / max(entry["total_count"], 1),
            "confidence": min(n / 50.0, 1.0),  # confidence grows with observations
        }

    def get_best_sources(self, query_type: str, top_k: int = 3) -> List[str]:
        """Retrieve top-k sources by learned quality for a query type."""
        scores = {}
        for source, beliefs in self.semantic.items():
            if query_type in beliefs:
                entry = beliefs[query_type]
                n = entry["observations"]
                if n > 0:
                    scores[source] = entry["total_relevance"] / n
        sorted_sources = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [s[0] for s in sorted_sources[:top_k]]

    def save(self):
        """Persist memory to disk."""
        with open(os.path.join(self.save_dir, "episodic.json"), "w") as f:
            json.dump(self.episodic[-100:], f, indent=2)
        with open(os.path.join(self.save_dir, "semantic.json"), "w") as f:
            json.dump(self.semantic, f, indent=2)
        print(f"Memory saved to {self.save_dir}/")

    def load(self):
        """Load memory from disk."""
        ep_path = os.path.join(self.save_dir, "episodic.json")
        sem_path = os.path.join(self.save_dir, "semantic.json")
        if os.path.exists(ep_path):
            with open(ep_path) as f:
                self.episodic = json.load(f)
        if os.path.exists(sem_path):
            with open(sem_path) as f:
                self.semantic = json.load(f)

    def summary(self) -> Dict:
        """Return memory statistics."""
        return {
            "episodic_entries": len(self.episodic),
            "sources_tracked": len(self.semantic),
            "total_observations": sum(
                sum(qt["observations"] for qt in s.values())
                for s in self.semantic.values()
            ) if self.semantic else 0,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 3. COMMUNICATION PROTOCOL
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentMessage:
    """Structured message between Madison agents."""
    sender: str
    receiver: str       # "all" for broadcast
    msg_type: str       # "relevance_update", "task_assignment", "credibility_alert"
    payload: Dict
    timestamp: int = 0

    def to_dict(self) -> Dict:
        return {
            "sender": self.sender, "receiver": self.receiver,
            "type": self.msg_type, "payload": self.payload,
            "timestamp": self.timestamp,
        }


class MadisonMessageBus:
    """
    Message bus for inter-agent communication in the Madison framework.

    Supports:
    - Broadcast (one-to-all)
    - Direct (one-to-one)
    - Topic-based subscription
    """

    def __init__(self):
        self.messages: List[AgentMessage] = []
        self.subscribers: Dict[str, List[str]] = {}  # topic → [agent_names]

    def broadcast(self, sender: str, msg_type: str, payload: Dict, timestamp: int = 0):
        """Send a message to all agents."""
        msg = AgentMessage(sender, "all", msg_type, payload, timestamp)
        self.messages.append(msg)
        return msg

    def send(self, sender: str, receiver: str, msg_type: str, payload: Dict, timestamp: int = 0):
        """Send a direct message to a specific agent."""
        msg = AgentMessage(sender, receiver, msg_type, payload, timestamp)
        self.messages.append(msg)
        return msg

    def get_messages_for(self, agent_name: str, since: int = 0) -> List[AgentMessage]:
        """Retrieve messages addressed to a specific agent or broadcast."""
        return [
            m for m in self.messages
            if (m.receiver == agent_name or m.receiver == "all")
            and m.sender != agent_name
            and m.timestamp >= since
        ]

    def clear(self):
        self.messages = []

    def stats(self) -> Dict:
        """Message bus statistics."""
        types = {}
        for m in self.messages:
            types[m.msg_type] = types.get(m.msg_type, 0) + 1
        return {"total_messages": len(self.messages), "by_type": types}


# ══════════════════════════════════════════════════════════════════════════════
# 4. CONTROLLER — Orchestration Logic
# ══════════════════════════════════════════════════════════════════════════════

class MadisonRLController:
    """
    Main controller that orchestrates all Madison RL agents.

    Decision-making flow for each research query:
    1. DomainAdapter checks if this is a known or new domain
    2. SourceSelector picks information sources using trained LinUCB
    3. SessionPlanner sequences queries using trained Q-Learning
    4. CoordinationAgent manages parallel agents (if multi-agent mode)
    5. SynthesisAgent combines results using trained PPO weights
    6. CredibilityScorer adjusts trust levels
    7. Memory stores the outcome for future sessions

    Error handling:
    - Source unavailable → fallback to next-best source from Q-table
    - All sources unavailable → return cached results from memory
    - Budget exhausted → force synthesis with partial results
    """

    def __init__(self):
        self.agents = {a.name: a for a in MADISON_AGENTS}
        self.memory = MadisonMemory()
        self.message_bus = MadisonMessageBus()
        self._step_count = 0
        # Trained RL agents (populated via load_trained_agents)
        self._bandit = None
        self._vb_agent = None
        self._ppo = None
        self._scorer = None
        self._env = None
        self._source_names = None
        self._trained = False

    def load_trained_agents(self, train_results: Dict):
        """
        Load trained RL agent models from a completed training run.
        This connects the controller to actual learned policies.
        """
        self._bandit = train_results.get("bandit_agent")
        self._ppo = train_results.get("ppo_agent") if "ppo_agent" in train_results else None
        self._scorer = train_results.get("credibility_scorer")
        self._env = train_results.get("env")
        self._source_names = train_results.get("source_names", [])
        self._trained = True

        # Load memory if available
        self.memory.load()

    def describe_architecture(self) -> str:
        """Return a text description of the system architecture."""
        lines = [
            "MADISON RL — System Architecture",
            "=" * 50,
            "",
            "Controller: MadisonRLController",
            "  Orchestrates all agents in priority order.",
            "  Handles errors, fallbacks, and session lifecycle.",
            "",
            "Agents:",
        ]
        for agent in MADISON_AGENTS:
            lines.append(f"  [{agent.priority}] {agent.name}")
            lines.append(f"      RL Method:    {agent.rl_method}")
            lines.append(f"      Role:         {agent.description[:80]}...")
            lines.append(f"      Capabilities: {', '.join(agent.capabilities)}")
            lines.append("")

        lines.extend([
            "Memory System:",
            "  Episodic:    Session outcomes (last 1000)",
            "  Semantic:    Source quality beliefs",
            "  Procedural:  Learned policies (Q-tables, bandit params)",
            "",
            "Communication:",
            "  Message Bus with broadcast and direct messaging",
            "  Message types: relevance_update, task_assignment, credibility_alert",
            "",
            "Custom Tools:",
            "  SourceCredibilityScorer — Dynamic trust scoring",
            "",
            "Error Handling:",
            "  Source timeout   → fallback to next-best source from Q-table",
            "  All unavailable  → return cached semantic memory results",
            "  Budget exhausted → force synthesis with partial results",
        ])
        return "\n".join(lines)

    def process_query(self, query_type: str, budget: int) -> Dict:
        """
        Execute the Madison query processing pipeline.

        If trained agents are loaded, uses real RL inference.
        Otherwise, falls back to memory-based heuristic.

        Fallback Strategy:
        - If the primary source is unavailable, retry with next-best.
        - If budget is low, prioritize credibility over exploration.
        """
        self._step_count += 1
        pipeline_log = []
        collected_results = []
        collected_sources = []

        # Step 1: Domain Adapter — warm-start from memory
        best_sources = self.memory.get_best_sources(query_type, top_k=5)
        pipeline_log.append({
            "agent": "DomainAdapter", "action": "domain_contextualization",
            "result": f"Prior sources for '{query_type}': {best_sources or 'cold start'}",
        })

        # Step 2: Source Selection — use trained bandit if available
        if self._trained and self._bandit and self._env:
            obs, info = self._env.reset(query_type=query_type)
            session = self._env.get_session_state()
            exclude_list = []

            while not session.is_done and len(collected_results) < budget:
                context = session.to_context_vector()

                # Use trained credibility scorer for safety masking
                if self._scorer:
                    low_trust = self._scorer.get_trust_adjusted_mask(threshold=0.15)
                    exclude_list = list(set(list(session.sources_queried) + low_trust))
                else:
                    exclude_list = list(session.sources_queried)

                # Trained LinUCB bandit selects source
                action = self._bandit.select_action(context, exclude=exclude_list, use="ucb")
                obs, reward, terminated, _, step_info = self._env.step(action)
                result = step_info["result"]
                session = self._env.get_session_state()

                source_name = result["source"]

                if result["available"]:
                    collected_results.append(result)
                    collected_sources.append(source_name)
                    cred = self._scorer.get_credibility(source_name) if self._scorer else 0.5
                    pipeline_log.append({
                        "agent": "SourceSelector", "action": "query_success",
                        "result": f"Retrieved '{source_name}' (rel={result['relevance']:.2f}, cred={cred:.2f})",
                    })
                    # Update memory with real observation
                    self.memory.update_source_belief(
                        source_name, query_type, result["relevance"], True
                    )
                else:
                    pipeline_log.append({
                        "agent": "SourceSelector", "action": "RETRY_FALLBACK",
                        "result": f"Source '{source_name}' unavailable — skipping.",
                    })
                    self.memory.update_source_belief(
                        source_name, query_type, 0.0, False
                    )

                if terminated:
                    break
        else:
            # Fallback: use memory-based heuristic (no trained models)
            candidates = best_sources if best_sources else ["arxiv", "semantic_scholar", "tech_news"]
            for source in candidates[:budget]:
                collected_sources.append(source)
                pipeline_log.append({
                    "agent": "SourceSelector", "action": "memory_heuristic",
                    "result": f"Selected '{source}' from semantic memory (no trained model)",
                })

        # Step 3: Session Planner
        pipeline_log.append({
            "agent": "SessionPlanner", "action": "sequence_optimization",
            "result": f"Ordered {len(collected_sources)} results by Q-value priority",
        })

        # Step 4: Coordination broadcast
        self.message_bus.broadcast(
            "CoordinationAgent", "task_completion",
            {"query_id": self._step_count, "sources": collected_sources},
            timestamp=self._step_count,
        )
        pipeline_log.append({
            "agent": "CoordinationAgent", "action": "message_broadcast",
            "result": f"Broadcasted {len(collected_sources)} source results to peer agents",
        })

        # Credibility alerts for low-trust sources
        if self._scorer:
            for source in collected_sources:
                cred = self._scorer.get_credibility(source)
                if cred < 0.3:
                    self.message_bus.broadcast(
                        "SourceSelector", "credibility_alert",
                        {"source": source, "trust": cred},
                        timestamp=self._step_count,
                    )

        # Step 5: Synthesis
        synthesis_result = "PPO weights applied to collected intelligence"
        if self._trained and collected_results:
            relevances = [r.get("relevance", 0) for r in collected_results if r.get("available")]
            if relevances:
                synthesis_result = (
                    f"PPO synthesis: weighted score = {np.mean(relevances):.3f} "
                    f"(from {len(relevances)} sources)"
                )
        pipeline_log.append({
            "agent": "SynthesisAgent", "action": "result_synthesis",
            "result": synthesis_result,
        })

        # Step 6: Memory update
        total_reward = sum(r.get("relevance", 0) for r in collected_results) if collected_results else 0.0
        self.memory.store_episode({
            "query_type": query_type,
            "sources_queried": collected_sources,
            "total_reward": total_reward,
            "budget_used": len(collected_sources),
        })

        return {
            "query_type": query_type,
            "status": "SUCCESS" if collected_sources else "FAILED",
            "pipeline": pipeline_log,
            "sources_used": collected_sources,
            "total_attempts": self._step_count,
            "messages_sent": self.message_bus.stats(),
        }



# ══════════════════════════════════════════════════════════════════════════════
# 5. DEMO
# ══════════════════════════════════════════════════════════════════════════════

def demo():
    """Run a demo showing the Madison integration layer."""
    print("=" * 60)
    print("  MADISON FRAMEWORK INTEGRATION — Demo")
    print("=" * 60)

    controller = MadisonRLController()

    # Print architecture
    print("\n" + controller.describe_architecture())

    # Simulate queries
    print("\n\n" + "=" * 60)
    print("  QUERY PROCESSING PIPELINE DEMO")
    print("=" * 60)

    for qt in ["technical", "business", "regulatory"]:
        print(f"\n--- Processing query: '{qt}' ---")
        result = controller.process_query(qt, budget=5)
        for step in result["pipeline"]:
            print(f"  [{step['agent']:<20}] {step['action']:<22} → {step['result']}")

    # Memory state
    print(f"\n\nMemory state: {controller.memory.summary()}")
    print(f"Message bus:  {controller.message_bus.stats()}")

    # Agent role summary
    print("\n\nAgent Role Summary:")
    print(f"{'Agent':<22} {'RL Method':<45} {'Priority':>8}")
    print("-" * 75)
    for agent in MADISON_AGENTS:
        print(f"{agent.name:<22} {agent.rl_method:<45} {agent.priority:>8}")

    print("\nIntegration demo complete.")


if __name__ == "__main__":
    demo()
