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
