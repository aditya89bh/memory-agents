# Project 7 – Experience Memory for Agents  
_State → Action → Outcome Learning Layer_

This project implements **Experience Memory**: a world-aware memory system that allows an agent to learn from **outcomes in specific contexts**, not just conversations or facts.

Unlike traditional chatbot memory, this system remembers:
- what situation occurred
- what action or strategy was chosen
- what actually happened

Over time, this enables agents to **avoid repeating failures** and **prefer strategies that worked before**, without retraining.

---

## Why This Project Exists

Most agents today:
- remember conversations
- retrieve knowledge
- plan actions

But still:
- repeat the same mistakes
- ignore past failures in similar contexts
- feel stateless across environments

This happens because they lack **experience memory**.

> Most chatbots remember conversations.  
> This agent remembers outcomes.

---

## Core Idea

Every experience is stored as:
STATE → ACTION → OUTCOME


Examples:
- “In this context, strategy B worked better.”
- “Last time this state occurred, action A failed.”
- “This task behaves differently under these constraints.”

This is **experiential memory**, not knowledge memory.

---

## What This Project Does

### ✅ Stores Experiences
Each experience records:
- **State**: structured context snapshot
- **Action**: strategy or skill chosen
- **Outcome**: success, score, failure type
- **Metadata**: timestamp, salience, source

### ✅ Retrieves Relevant Past Experiences
- Hard filter: task must match
- Optional filters: environment, phase
- Similarity via transparent feature overlap (no embeddings)

### ✅ Ranks Experiences by Usefulness
Ranking combines:
- state similarity
- outcome quality
- recency (time decay)

### ✅ Aggregates by Action
The system summarizes:
- how many times an action was tried
- success rate
- average quality
- common failure patterns

### ✅ Advises the Planner
Returns ranked **recommendations**, not decisions.

> Experience Memory advises.  
> The planner decides.

---

## What This Project Does *Not* Do

This project explicitly does **not**:
- perform planning or decision-making
- store user preferences or identity (Project 6)
- store factual knowledge (Project 3)
- remember conversations (Projects 1–2)
- build spatial maps or world models

This keeps the scope tight and debuggable.

---

## Data Model (MVP)

### Experience Record

**State**
```json
{
  "task": "summarize_document",
  "env": "web_chat",
  "constraints": ["concise", "time_limited"],
  "signals": { "domain": "technical", "length": "long" },
  "tags": ["nlp"]
}
Action
{
  "strategy": "hierarchical_summary",
  "skill": "summarizer_v2",
  "parameters": { "verbosity": "low" }
}
Outcome
{
  "success": true,
  "score": 0.82,
  "latency_ms": 1200
}
Retrieval & Ranking Logic (MVP)
Hard Filter

task must match exactly

State Similarity

Weighted overlap across:

environment

phase

constraints

signals

tags

Record Ranking
rank_score =
  0.55 * similarity +
  0.25 * outcome_quality +
  0.20 * recency
Action Recommendation Score
action_score =
  0.60 * success_rate +
  0.25 * avg_quality +
  0.15 * recency_of_last_success
All weights are explicit and tunable.
Public API (Project Boundary)
add_experience(state, action, outcome, salience=0.5, episode_id=None)

query(state, k=10, min_similarity=0.0, filters=None)

recommend(state, k_actions=5, k_records=25)

export_json()
import_json(json_str)
The planner consumes recommend().
It never parses raw experience logs.

Integration in the Memory Stack

This project plugs in after planning but before action selection:

Planner builds current state

Experience Memory is queried

Recommendations are returned

Planner chooses final action

No coupling to identity, skills, or LLM internals.

Design Principles

Code-first, Colab-friendly

No frameworks

No embeddings (MVP)

Transparent logic

Explainable behavior

Serializable state

This is product-grade boring on purpose.

Roadmap Position

This is Project 7 in a multi-project memory-agent roadmap:

Short-Term Memory

Summary Memory

Long-Term Memory

Unified Memory Stack

Memory-Influenced Planning

Identity & Personality Memory

Experience Memory (this project)

Status

✅ PRD-driven
✅ Implemented
✅ Runnable demo
✅ JSON persistence
✅ Ready for integration

---

## Final checklist before GitHub upload

Do this and you’re golden:

- [ ] File name: `README.md`
- [ ] Code file: `project_7_experience_memory.py`
- [ ] Demo runs clean
- [ ] Repo description:
  **“Experience Memory for agents using state–action–outcome learning.”**

If you want next, we can:
- add a **PRD section inside the README**
- write a **Project 7A vs future 7B roadmap**
- or integrate this into your Project 5 planner with ~5 lines of code

You’ve built something *legit* here.

