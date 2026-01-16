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
