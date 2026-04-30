# Project 4 — Memory + Planning

Project 4 extends the memory-agent architecture by introducing planning, outcomes, and experience-based decision making.

Up to Project 3, memory provides continuity and context. From Project 4 onward, memory influences what the agent decides to do.

---

## Goal

Make memory influence decisions, not just answers.

---

## What this project proves

If the same user input produces different behavior over time because of memory, the project succeeds.

---

## Core idea

The agent should remember what worked, what failed, and choose differently next time.

```text
Read
  -> Assemble Context
  -> Plan
  -> Act
  -> Evaluate
  -> Write Experience Memory
```

---

## Lifecycle shift

| Before Project 4 | After Project 4 |
|---|---|
| Read | Read |
| Assemble context | Assemble context |
| Respond | Plan |
| Write | Act |
|  | Evaluate |
|  | Write outcome memory |

This is where memory begins to close the learning loop.

---

## Components

| Component | Role |
|---|---|
| MemoryItem | Atomic long-term memory record |
| Turn | One interaction step |
| Plan | Chosen strategy and rationale |
| Outcome | Success or failure signal |
| Experience Memory | Stored record of strategy and outcome |
| Planner | Selects a strategy before action |
| Evaluator | Determines whether the action worked |
| Memory Manager | Enforces the full lifecycle |

---

## Example trace

```text
Turn 1:
User: Can you book a flight for me?

Memory read:
No prior experience found.

Plan:
direct_action

Action:
Attempt to book flight.

Outcome:
Failure. Missing destination and travel date.

Memory write:
For flight booking requests, direct_action fails when required details are missing.

Turn 2:
User: Can you book a flight for me?

Memory read:
Previous direct_action failed because destination and date were missing.

Plan:
ask_clarifying_questions

Action:
Ask for destination and travel date.

Outcome:
Success. User provides missing details.
```

---

## Why this matters

Memory is not truly useful if it only improves recall.

The stronger test is whether memory improves action selection.

Project 4 is the turning point where memory becomes cognitive.

---

## Minimum implementation

A minimal version should include:

- typed memory records
- context assembly
- a deterministic planner
- action execution simulation
- outcome evaluation
- experience memory writes
- retrieval of past failures or successes before future planning

---

## How to run

Open the Project 4 notebook or script and run all cells/steps.

The demo should show:

- planning before action
- outcome evaluation
- experience storage
- memory-driven behavior change

---

## Key insight

Memory becomes intelligence when it changes future action.

---

## Status

Complete.

This project prepares the repo for Project 5, where repeated outcomes become reusable skills.