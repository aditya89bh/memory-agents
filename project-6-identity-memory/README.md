# Project 6 — Identity & Personality Memory

Project 6 is planned as the identity layer of the memory agent stack.

The goal is to make an agent consistent across time without making it rigid.

---

## Goal

Give the agent stable memory of preferences, traits, goals, and behavioral constraints.

---

## What this project should prove

An agent can remember stable user or system-level identity information and use it to resolve future responses.

Example:

```text
Stable memory:
User prefers concise, direct explanations.

New request:
Explain memory retrieval.

Agent behavior:
Responds with a concise explanation instead of a long tutorial.
```

---

## Why identity memory matters

Without identity memory, agents become inconsistent.

They may change tone, forget preferences, repeat old mistakes, or behave like a new system every time.

Identity memory helps agents preserve continuity across weeks and months.

---

## Memory types

| Memory type | Example |
|---|---|
| Preference | User prefers concise answers |
| Goal | User is building memory-agent expertise |
| Trait | Agent should be analytical and direct |
| Boundary | Do not invent unsupported claims |
| Style | Prefer Python-first explanations |
| Long-term context | User is studying memory as a design material |

---

## Architecture idea

```text
User Input
  -> Identity Memory Retrieval
  -> Preference / Goal / Trait Resolution
  -> Context Assembly
  -> Agent Response
  -> Memory Update
```

Identity memories should be retrieved early because they influence how the rest of the context is interpreted.

---

## Key design problem

Identity memory needs conflict resolution.

Example:

```text
Old memory:
User prefers short answers.

Current request:
Explain this deeply.

Correct behavior:
Follow the current request, but preserve the stable preference for future default behavior.
```

Stable memory should guide behavior, not override explicit current intent.

---

## Minimum implementation

A minimal version should include:

- an identity memory store
- preference memories
- goal memories
- trait memories
- simple conflict resolution logic
- context assembly that injects identity memories before response generation

---

## Minimum demo

```text
Turn 1:
User: I prefer concise answers.
Memory write: preference = concise answers

Turn 2:
User: Explain vector recall.
Memory read: preference = concise answers
Agent: Gives a short explanation.

Turn 3:
User: Now explain it deeply.
Agent: Gives a deeper answer because current instruction overrides the default preference.
```

---

## Key insight

Identity is memory stabilized into behavior.

---

## Status

Planned.

This project should be built after Projects 1–5 are fully polished.