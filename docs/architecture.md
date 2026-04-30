# Memory Agents Architecture

This document explains the architecture behind the Memory Agents learning path.

The repo moves from simple continuity to cognition by adding one memory capability at a time.

---

## Full lifecycle

```text
Input
  -> Memory Gate
  -> Short-Term Memory
  -> Summary Memory
  -> Long-Term Retrieval
  -> Context Assembly
  -> Planning
  -> Action / Response
  -> Outcome Evaluation
  -> Memory Write
```

The important idea is that memory is not only storage. Memory becomes useful when it changes what context is assembled, what plan is selected, and what the agent does next.

---

## Core components

| Component | Role |
|---|---|
| Memory Gate | Decides what should be stored, ignored, or routed |
| Short-Term Memory | Keeps recent turns available for continuity |
| Summary Memory | Compresses older context into a durable narrative |
| Long-Term Memory | Stores searchable facts, preferences, goals, and experiences |
| Metadata Memory | Adds type, source, timestamp, and tags to memories |
| Salience Layer | Decides what deserves long-term storage |
| Context Assembly | Builds the actual context the agent reasons over |
| Planning Layer | Chooses a strategy before action |
| Outcome Evaluation | Records whether the chosen action worked |
| Skill Memory | Compresses repeated task attempts into reusable competence |
| Identity Memory | Preserves stable preferences and behavioral consistency |
| World Memory | Connects memory to environments, states, and actions |

---

## Project progression

| Project | Architectural layer added |
|---|---|
| Project 1 | Short-term memory |
| Project 1B | Summary memory |
| Project 2 | Long-term retrieval |
| Project 3 | Unified memory manager |
| Project 4 | Planning and outcome memory |
| Project 5 | Skill abstraction |
| Project 6 | Identity memory |
| Project 7 | Embodied/world memory |

---

## Why context assembly matters

Memory does not help an agent if it is stored but never used.

A memory system needs an explicit context assembly step that decides what the agent sees before it reasons or acts.

A typical order is:

1. Stable identity and preference memories
2. Relevant long-term recalls
3. Summary memory
4. Recent short-term turns
5. Current user input

This prevents memory from becoming a junk drawer.

---

## Why planning matters

Before planning, memory only improves answers.

After planning, memory can improve decisions.

That is the key transition in Project 4:

```text
Read -> Assemble Context -> Plan -> Act -> Evaluate -> Write
```

When an agent remembers that a strategy failed and chooses differently next time, memory has become cognitive.

---

## Final architecture idea

A memory agent is not a chatbot with a database.

A memory agent is a system where past experience shapes future behavior.