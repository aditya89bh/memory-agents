# Project 3 – Unified Memory Stack

This project integrates all previous memory systems into a **single, explicit memory architecture** for an AI agent.

By this stage, the agent no longer has “multiple memory modules” scattered across the codebase.
Instead, it has **one MemoryManager** that enforces how memory is read, assembled, and written.

---

## Why this project exists

Most agent implementations treat memory as:
- an append-only log, or
- an implicit side-effect of prompts

This project makes memory:
- **explicit**
- **typed**
- **gated**
- **ordered in time**

Most importantly:

> **Memory retrieval happens before reasoning. Always.**

---

## Architecture Overview

The unified memory stack consists of four layers:

### 1. Short-Term Memory
- Rolling buffer of recent turns
- Explicit forgetting via max window size
- Never written directly by the agent

### 2. Summary Memory
- Compressed continuity across turns
- Updated deterministically from recent context
- No recursive summarization

### 3. Long-Term Memory
- Vector-based recall (TF-IDF / embeddings)
- Metadata-aware filtering
- Salience-gated writes
- Supports pinned memories

### 4. MemoryManager (new in this project)
The single orchestration point that enforces:

1. **Read phase**  
   Relevant memory is retrieved *before* response generation.

2. **Context assembly**  
   Memory is converted into a structured context for the agent.

3. **Write phase**  
   Only salient information is persisted.

---

## Core Design Principles

- Retrieval-before-reasoning is enforced by design
- Memory writes are gated, not automatic
- Identity, preferences, and goals are first-class memory types
- Pinned memory cannot be accidentally forgotten
- No heavy frameworks or hidden abstractions
- Fully Colab-friendly

---

## Memory Flow (per turn)

```text
User Input
   ↓
MemoryManager.read()
   ↓
ContextAssembler.build()
   ↓
Agent / LLM
   ↓
MemoryManager.write()  (gated)
