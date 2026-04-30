# Project 1B — Summary Memory

Project 1B extends short-term memory with a compression layer.

Instead of keeping every older turn, the agent maintains a recent buffer and a running summary.

---

## Goal

Prevent context explosion while preserving meaning.

---

## What this project proves

An agent can compress older conversation history into a useful summary while keeping recent turns available in detail.

---

## Core idea

Chronology does not scale.

A useful memory system needs to compress older context instead of carrying every turn forever.

```text
Recent Turns
  -> Short-Term Buffer
  -> Older Turns
  -> Summary Memory
  -> Context Assembly
  -> Agent Response
```

---

## What it shows

- two-tier memory
- recent buffer
- running summary
- explicit forgetting and compression
- prevention of recursive memory growth
- Colab-safe agent loop without frameworks

---

## Example trace

```text
Recent buffer:
Turn 7: User asks about long-term memory.
Turn 8: Agent explains retrieval.
Turn 9: User asks for examples.

Summary memory:
User is building a memory-agents repo and is learning how memory evolves from continuity to retrieval.

Assembled context:
- Summary memory
- Recent buffer
- Current user message

Agent behavior:
Continues the conversation without needing every previous turn.
```

---

## Why this matters

Short-term memory alone eventually runs out of space.

Summary memory allows the agent to preserve the meaning of older context without carrying the full transcript.

---

## Minimum implementation

A minimal version should include:

- a recent-turn buffer
- a running summary string
- a rule for when older turns are summarized
- a method to assemble summary plus recent turns
- a guard against summary recursively summarizing itself

---

## Note

The summarizer can be heuristic for learning purposes.

A future version can replace it with an LLM-based summarization call.

---

## Key insight

Compression is intelligence.

---

## Status

Complete.

This project prepares the repo for Project 2, where memory becomes searchable and durable.