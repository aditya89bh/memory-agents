# Project 2 — Long-Term Memory

Project 2 moves the agent from recent context to searchable experience.

Unlike short-term or summary memory, long-term memory is durable, searchable, and independent of the current conversation window.

---

## Goal

Give the agent the ability to retrieve relevant past information.

---

## What this project proves

An agent can recall memories based on relevance instead of recency.

---

## Core idea

Instead of remembering only what happened last, the agent learns to retrieve what matters right now.

```text
Memory Store
  -> Vectorization / Embedding
  -> Similarity Search
  -> Top-K Retrieval
  -> Context Assembly
  -> Agent Response
```

---

## Structure

Project 2 is intentionally iterative.

| Part | Focus | What it proves |
|---|---|---|
| 2A | Basic vector recall | Relevant memories can be retrieved using TF-IDF and cosine similarity |
| 2B | Metadata-aware memory | Retrieval improves when memories include type, source, timestamp, and tags |
| 2C | Salience and memory gating | Not every interaction deserves long-term storage |
| 2D | Neural embeddings | Recall becomes more semantic and less keyword-bound |

---

## What it shows

- persistent memory records
- TF-IDF vector recall
- cosine similarity search
- top-k retrieval
- metadata-aware filtering
- salience-based storage decisions
- neural embedding upgrade path

---

## Example trace

```text
Stored memories:
1. User prefers concise explanations.
2. User is building a memory-agents repo.
3. User wants to study AGI through projects.
4. User likes Python-first examples.

Query:
How should I explain this repo to the user?

Retrieved memories:
1. User is building a memory-agents repo.
2. User prefers concise explanations.
3. User wants to study AGI through projects.

Agent behavior:
Gives a concise explanation connected to the user's project-based AGI learning path.
```

---

## Why this matters

Short-term memory preserves continuity.

Long-term memory creates reusable experience.

This is the foundation of retrieval-augmented agents and memory-driven systems.

---

## Minimum implementation

A minimal version should include:

- a memory record structure
- a memory store
- a vectorization method
- similarity scoring
- top-k retrieval
- context injection
- optional metadata filters

---

## Dependencies

Project 2A can be implemented with lightweight TF-IDF tools.

Project 2D can use Sentence Transformers:

```bash
pip install sentence-transformers
```

---

## Design principles

- memory is explicit, not implicit
- retrieval happens before reasoning
- memory storage and memory usage are separate concerns
- salience matters more than volume
- forgetting is a feature

---

## Key insight

The agent should remember what matters right now, not only what happened last.

---

## Status

Complete.

This project prepares the repo for Project 3, where memory layers are unified into one stack.