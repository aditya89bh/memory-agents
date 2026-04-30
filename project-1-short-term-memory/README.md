# Project 1 — Short-Term Memory

Project 1 demonstrates the simplest useful memory system for an AI agent: short-term continuity.

The agent keeps a bounded rolling window of recent turns and deliberately forgets older context.

---

## Goal

Give the agent continuity across turns.

---

## What this project proves

An agent can preserve recent interaction state without storing everything forever.

---

## Core idea

Short-term memory is a working buffer.

It gives the agent access to recent context while keeping memory deterministic and bounded.

```text
User Input
  -> Short-Term Buffer
  -> Context Assembly
  -> Agent Response
  -> Memory Write
  -> Forget Oldest Turn if Limit is Exceeded
```

---

## What it shows

- explicit memory storage
- deterministic forgetting
- rolling window memory
- clear separation between memory writing and reading
- a simple agent loop without external frameworks

---

## What it does not try to do

- long-term memory
- vector search
- summarization
- planning
- skill learning

---

## Example trace

```text
Turn 1:
User: My name is Aditya.
Memory write: user said name = Aditya

Turn 2:
User: I am building memory agents.
Memory write: user is building memory agents

Turn 3:
User: What am I building?
Memory read: user is building memory agents
Agent: You are building memory agents.

After memory limit is exceeded:
Oldest turn is removed from the buffer.
```

---

## Why this matters

Without short-term memory, every turn is isolated.

The agent may answer one message correctly but fail to maintain continuity across a conversation.

---

## Minimum implementation

A minimal version should include:

- a list or queue for recent turns
- a fixed memory limit
- a method to write new turns
- a method to read recent context
- automatic removal of old turns

---

## Key insight

Memory is not what you store. Memory is what you choose to forget.

---

## Status

Complete.

This project prepares the repo for Project 1B, where older context is compressed instead of simply discarded.