# Project 7 – Experience Memory for Agents  
_State → Action → Outcome Learning Layer_

This project implements Experience Memory: a state–action–outcome learning layer that allows an agent to learn from outcomes in specific contexts, not just conversations or facts.

Unlike traditional chatbot memory, this system remembers what situation occurred, what action or strategy was chosen, and what actually happened. Over time, this enables agents to avoid repeating failures and prefer strategies that worked before, without retraining.

Most agents today can remember conversations, retrieve knowledge, and plan actions, but they still repeat the same mistakes, ignore past failures in similar contexts, and feel stateless across environments. This happens because they lack experience memory. Most chatbots remember conversations. This agent remembers outcomes.

The core idea is simple. Every experience is stored as a structured triple:
STATE → ACTION → OUTCOME

Examples include:
In this context, strategy B worked better.
Last time this state occurred, action A failed.
This task behaves differently under these constraints.

This is experiential memory, not knowledge memory.

The system provides the following capabilities:
It stores experiences as structured records.
It retrieves past experiences relevant to a given context.
It ranks experiences by usefulness.
It aggregates experience by action or strategy.
It advises the planner on what tends to work.

The system never makes decisions on its own.

This project explicitly does not perform planning or decision-making, does not store user preferences or identity, does not store factual knowledge, does not remember conversations, and does not build spatial maps or world models. The scope is intentionally narrow and debuggable.

Each experience record contains four parts: state, action, outcome, and metadata.

State is a structured snapshot of the context. The field task is required. All other fields are optional.

Example state:
{
  "task": "summarize_document",
  "env": "web_chat",
  "constraints": ["concise", "time_limited"],
  "signals": { "domain": "technical", "length": "long" },
  "tags": ["nlp"]
}
Action captures what the agent chose to do.

Example action:
{
  "strategy": "hierarchical_summary",
  "skill": "summarizer_v2",
  "parameters": { "verbosity": "low" }
}

Outcome captures what happened as a result.

Example outcome:
{
  "success": true,
  "score": 0.82,
  "latency_ms": 1200
}

The field success is required in the outcome.

Retrieval follows simple and transparent rules. The task must match exactly. Optional filters such as environment or phase may be applied. Similarity is computed using weighted overlap across environment, phase, constraints, signals, and tags. No embeddings are used in the MVP.

Each experience record is ranked using:
rank_score = 0.55 * similarity + 0.25 * outcome_quality + 0.20 * recency

Experiences are then aggregated by action signature:
strategy | skill | parameters

For each action, the system computes the number of trials, success rate, average quality, common failure patterns, and recency of last success. Actions are ranked using:
action_score = 0.60 * success_rate + 0.25 * avg_quality + 0.15 * recency_of_last_success

The output is advisory, not prescriptive.

The public interface exposed to planners includes:
add_experience(state, action, outcome, salience=0.5, episode_id=None)
query(state, k=10, min_similarity=0.0, filters=None)
recommend(state, k_actions=5, k_records=25)
export_json()
import_json(json_str)

The planner consumes recommendations and remains responsible for final decisions.

Typical integration flow:
The planner constructs the current state.
Experience Memory is queried.
Ranked recommendations are returned.
The planner selects the final action.

Experience Memory advises. The planner decides.

This project is part of a larger memory-agent roadmap:
1. Short-Term Memory
2. Summary Memory
3. Long-Term Memory
4. Unified Memory Stack
5. Memory-Influenced Planning
6. Identity and Personality Memory
7. Experience Memory (this project)

Status:
Implemented.
Runnable demo included.
PRD-driven design.
JSON persistence supported.
Ready for integration.
