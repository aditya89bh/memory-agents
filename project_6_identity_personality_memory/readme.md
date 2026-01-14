# Project 6 – Identity & Personality Memory

This project implements a persistent **Identity & Personality Memory** layer for a memory-agent system.

The goal is to make the agent feel **stable, consistent, and coherent over long time horizons** (weeks or months), rather than behaving like a stateless or reset chatbot across interactions.

This layer governs *who the agent is* and *how it behaves*, separate from task memory, planning memory, or factual recall.

---

## Why This Project Exists

Most agents:
- Remember facts
- Retrieve documents
- Plan actions

But still:
- Change tone randomly
- Forget user preferences
- Drift in personality
- Overwrite long-term traits too easily

**Identity & Personality Memory** solves this by introducing:
- A protected identity core
- Explicit separation between user data and agent traits
- Conflict-aware updates instead of blind overwrites

---

## Core Concepts

### 1. Separation of Concerns

This project enforces a strict boundary between:

| Category | Description |
|------|-----------|
| User Facts | Stable information about the user (location, role, etc.) |
| User Preferences | How the user prefers the agent to behave |
| Agent Traits | The agent’s own personality and response style |

This prevents:
- User preferences leaking into agent identity
- Agent traits being overwritten by temporary user behavior

---

### 2. Identity Items

Each identity memory is stored as an `IdentityItem` with:

- `key`, `value`
- `category` (user_fact, user_pref, agent_trait)
- `confidence` (how reliable it is)
- `salience` (how important it is)
- `protection` (how hard it is to overwrite)
- `confirmation_count`
- `change history`

This makes identity updates **auditable and explainable**.

---

### 3. Protected Identity Core

Some traits are intentionally harder to overwrite.

Examples:
- Agent emotionality
- Default analytical tone
- Long-standing user preferences

Protected items:
- Require stronger evidence to change
- Resist weak or implicit signals
- Can still be overridden by explicit user intent

---
### 4. Conflict-Aware Updates

When a new identity signal arrives:

- The system **scores** old vs new information
- Factors considered:
  - Confidence
  - Salience
  - Protection level
  - Source (explicit > implicit > inferred)

Possible outcomes:
- Confirm existing value
- Accept new value
- Reject update and log conflict

All conflicts are stored for inspection.

---
## What This Project Implements

### Identity Abstractions
- `IdentityItem`
- `IdentityProfile`
- `IdentityMemory` (project boundary object)

### Update Logic
- Explicit vs implicit updates
- Confirmation boosts
- Auto-protection after repeated confirmations
- Flip-flop penalties

### Integration Surface
- Generates compact **identity directives**
- Designed to plug into an existing Unified Memory Stack
- Zero framework dependencies
- Fully serializable (JSON export/import)

---

## Example Identity Directives

These are consumed downstream by planners or response generators:

AGENT_TRAIT::tone=analytical
AGENT_TRAIT::verbosity=concise
USER_PREFERENCE::answer_length=short
USER_FACT::location=Athens


This allows identity to shape behavior **without leaking raw memory objects**.

---

## Integration with Unified Memory Stack

This project does **not** replace existing memory systems.

It plugs in alongside:
- Short-term memory
- Summary memory
- Long-term vector memory
- Planning memory
- Skill & task memory

Typical usage:
1. Unified stack builds context
2. Identity directives are injected
3. Planner and response layers adapt behavior accordingly

---

## Design Constraints

This project intentionally follows:

- Code-first design
- Colab-friendly execution
- No external frameworks
- Transparent logic
- Clear project boundaries
- Deterministic behavior

Everything runs as a single Python file.

---

## What This Project Does *Not* Do

- It does not extract preferences automatically from language
- It does not generate responses
- It does not perform planning
- It does not store episodic or world state memory

Those belong to other projects in the roadmap.

---

## Roadmap Position

This is **Project 6** in the memory-agent roadmap.

Previous projects handled:
- Short-term memory
- Summary memory
- Long-term memory
- Unified memory stack
- Memory-influenced planning
- Skill & task memory

The next step is **Project 7 – Embodied / World Memory**, which moves memory from conversations into environments, states, and outcomes.

---

## Status

✅ Implemented  
✅ Runnable  
✅ Serializable  
✅ Ready for integration  

