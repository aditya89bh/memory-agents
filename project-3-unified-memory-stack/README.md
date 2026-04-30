# Project 3 — Unified Memory Stack

Project 3 integrates short-term memory, summary memory, and long-term retrieval into one memory stack.

Up to this point, each memory type has been explored separately. This project combines them into a single disciplined lifecycle.

---

## Goal

Make memory feel like one system instead of scattered modules.

---

## What this project proves

An agent can combine recent context, compressed history, and retrieved long-term memories into a single assembled context before reasoning.

---

## Core idea

Memory becomes useful only when it is assembled into the agent's working context at the right time.

The unified lifecycle is:

```text
Input
  -> Memory Gate
  -> Short-Term Memory
  -> Summary Memory
  -> Long-Term Retrieval
  -> Context Assembly
  -> Agent Reasoning
  -> Memory Write
```

---

## Components

| Component | Role |
|---|---|
| Short-Term Memory | Keeps recent turns available |
| Summary Memory | Compresses older context |
| Long-Term Memory | Retrieves relevant durable memories |
| Memory Gate | Decides what should be stored or ignored |
| Context Assembly | Builds the final context used by the agent |
| Memory Manager | Coordinates reads, writes, retrieval, and assembly |

---

## Example trace

```text
User input:
Can you help me continue my memory agents project?

Short-term memory:
Recent conversation mentions Project 2 long-term memory.

Summary memory:
User is building a progressive memory-agents learning repo.

Long-term retrieval:
Relevant memory: user wants minimal frameworks and clear architecture.

Assembled context:
- Current request
- Recent Project 2 context
- Repo goal from summary
- Retrieved preference for clarity

Agent response:
Suggests the next project step without restarting from scratch.

Memory write:
Stores that Project 3 integration was started.
```

---

## Why this matters

Without a unified memory stack, memory becomes fragmented.

The agent may have short-term context, summaries, and retrieval, but still fail to use them coherently.

Project 3 introduces the idea that memory needs an operating layer.

---

## Minimum implementation

A minimal implementation should include:

- a `MemoryManager`
- a short-term memory buffer
- a summary memory layer
- a long-term retrieval layer
- a memory gate
- a context assembly method
- a simple agent loop that uses the assembled context

---

## Key insight

A real memory agent needs a memory operating system, not scattered memory functions.

---

## Status

Complete as a conceptual implementation layer.

This project prepares the repo for Project 4, where memory begins to influence planning and action.