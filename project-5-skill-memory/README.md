# Project 5 — Skill & Task Memory

Project 5 introduces skill memory: the ability to turn repeated task attempts into reusable competence.

Up to Project 4, the agent can remember outcomes and choose better strategies. Project 5 asks whether those repeated experiences can become skills.

---

## Goal

Turn repetition into reusable competence.

---

## What this project proves

An agent can remember task attempts, abstract patterns from them, and reuse those patterns when a similar task appears again.

---

## Core idea

Skill is memory compressed into action.

Instead of remembering every attempt forever, the agent should extract reusable patterns.

```text
Repeated task attempts
  -> Outcome memory
  -> Pattern extraction
  -> Skill abstraction
  -> Skill retrieval
  -> Better future action
```

---

## Components

| Component | Role |
|---|---|
| Task Attempt Memory | Stores what the agent tried |
| Outcome Memory | Stores whether the attempt worked |
| Skill Abstraction | Compresses repeated successful patterns |
| Task-to-Skill Mapping | Links new tasks to known skills |
| Skill Retrieval | Finds the right skill for the current task |
| Skill Use | Applies the skill to improve future action |

---

## Example trace

```text
Attempt 1:
Task: Summarize a research paper
Strategy: Read everything line by line
Outcome: Too slow

Attempt 2:
Task: Summarize another research paper
Strategy: Read abstract, method, results, limitations
Outcome: Successful and faster

Skill abstraction:
For research paper summaries, first extract:
- problem
- method
- result
- limitation
- implication

New task:
Summarize a third research paper

Skill retrieval:
Research paper summary skill found

Agent behavior:
Uses the learned structure instead of starting from scratch.
```

---

## Why this matters

Memory becomes more powerful when it stops being a list of past events and starts becoming reusable capability.

This is the bridge from remembering to learning.

---

## Minimum implementation

A minimal implementation should include:

- task attempt records
- outcome records
- task similarity matching
- skill abstraction from successful attempts
- skill retrieval for new tasks
- agent behavior influenced by retrieved skill

---

## Memory schema idea

```json
{
  "memory_type": "skill",
  "skill_name": "research_paper_summary",
  "trigger": "summarize research paper",
  "steps": [
    "identify problem",
    "extract method",
    "capture result",
    "note limitation",
    "write implication"
  ],
  "source_experiences": ["attempt_001", "attempt_002"],
  "confidence": 0.78
}
```

---

## Key insight

Skill memory is how experience becomes reusable action.

---

## Status

Complete as a conceptual implementation layer.

This project prepares the repo for Project 6, where memory stabilizes into identity and personality.