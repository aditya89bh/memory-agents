Project 7 – Experience Memory for Agents
State → Action → Outcome Learning Layer

This project implements Experience Memory: a world-aware memory system that allows an agent to learn from outcomes in specific contexts, not just conversations or facts.

Unlike traditional chatbot memory, this system remembers what situation occurred, what action or strategy was chosen, and what actually happened.

Over time, this enables agents to avoid repeating failures and prefer strategies that worked before, without retraining.

Why this project exists

Most agents today remember conversations, retrieve knowledge, and plan actions.

But they still repeat the same mistakes, ignore past failures in similar contexts, and feel stateless across environments.

This happens because they lack experience memory.

Most chatbots remember conversations.
This agent remembers outcomes.

Core idea

Every experience is stored as a simple triple:

STATE → ACTION → OUTCOME

Examples include:
In this context, strategy B worked better.
Last time this state occurred, action A failed.
This task behaves differently under these constraints.

This is experiential memory, not knowledge memory.

What this project does

The system stores experiences as structured records.
Each experience contains a state, an action, an outcome, and metadata.

The system retrieves relevant past experiences using a hard task match and optional environment or phase filters.

The system computes similarity using transparent feature overlap.
No embeddings are used in the MVP.

The system ranks experiences by usefulness using similarity, outcome quality, and recency.

The system aggregates experience by action or strategy, summarizing trials, success rates, average quality, and common failures.

The system returns recommendations, not decisions.

Experience Memory advises.
The planner decides.

What this project does not do

This project does not perform planning or decision-making.
It does not store user preferences or identity.
It does not store factual knowledge.
It does not remember conversations.
It does not build spatial maps or world models.

The scope is intentionally tight and debuggable.

Data model

Each experience record contains the following parts.

State example:
{
  "task": "summarize_document",
  "env": "web_chat",
  "constraints": ["concise", "time_limited"],
  "signals": { "domain": "technical", "length": "long" },
  "tags": ["nlp"]
}

Action example:
{
  "strategy": "hierarchical_summary",
  "skill": "summarizer_v2",
  "parameters": { "verbosity": "low" }
}

Outcome example:
{
  "success": true,
  "score": 0.82,
  "latency_ms": 1200
}

Retrieval and ranking logic

The task field must match exactly.

State similarity is computed using weighted overlap across environment, phase, constraints, signals, and tags.

Each experience record is ranked using:
rank_score = 0.55 * similarity + 0.25 * outcome_quality + 0.20 * recency

Experiences are then grouped by action signature.

Actions are ranked using:
action_score = 0.60 * success_rate + 0.25 * avg_quality + 0.15 * recency_of_last_success

All weights are explicit and tunable.

Public API

add_experience(state, action, outcome, salience=0.5, episode_id=None)
query(state, k=10, min_similarity=0.0, filters=None)
recommend(state, k_actions=5, k_records=25)
export_json()
import_json(json_str)

The planner consumes recommend().
It never parses raw experience logs.

Integration in the memory stack

The planner builds the current state.
Experience Memory is queried.
Ranked recommendations are returned.
The planner selects the final action.

There is no coupling to identity, skills, or LLM internals.

Design principles

Code-first.
Colab-friendly.
No frameworks.
No embeddings in the MVP.
Transparent logic.
Explainable behavior.
Serializable state.

This is product-grade boring on purpose.

Roadmap position

This is Project 7 in a multi-project memory-agent roadmap:

Project 1: Short-Term Memory
Project 2: Summary Memory
Project 3: Long-Term Memory
Project 4: Unified Memory Stack
Project 5: Memory-Influenced Planning
Project 6: Identity and Personality Memory
Project 7: Experience Memory

Status

Implemented.
PRD-driven.
Runnable demo included.
JSON persistence supported.
Ready for integration.
