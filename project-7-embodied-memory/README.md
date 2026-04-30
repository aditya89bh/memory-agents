# Project 7 — Embodied / World Memory

Project 7 is planned as the embodied memory layer of the memory agent stack.

The goal is to connect memory to environments, states, actions, and physical context.

---

## Goal

Give the agent memory of the world it acts inside.

This can include simulated environments, robotics tasks, spatial states, object interactions, and action outcomes.

---

## What this project should prove

An agent can remember that a previous action worked or failed in a specific environment and use that memory to choose differently next time.

Example:

```text
Past experience:
Path B was safer in this environment.

New task:
Move through the same environment again.

Agent behavior:
Chooses Path B instead of retrying the failed route.
```

---

## Why world memory matters

Text memory is not enough for embodied agents.

Robots, simulated agents, and physical systems need memory that is attached to:

- places
- objects
- states
- actions
- failures
- constraints
- outcomes

Without world memory, an embodied agent repeats physical mistakes.

---

## Memory types

| Memory type | Example |
|---|---|
| Spatial memory | Object was last seen near the left table |
| Object memory | This cylinder slips under low grip force |
| Environment memory | Path B was safer than Path A |
| State memory | Door was open during the previous attempt |
| Action memory | Side grasp worked better than top grasp |
| Failure memory | Smooth surface caused slip |
| Recovery memory | Increasing grip force solved the failure |

---

## Architecture idea

```text
Environment State
  -> Perception / State Encoder
  -> World Memory Retrieval
  -> Action Planner
  -> Action Execution
  -> Outcome Evaluation
  -> World Memory Write
```

The key difference from text memory is that retrieval must be grounded in state, not just language.

---

## Minimum implementation

A minimal version should include:

- a simple environment state representation
- memory records tied to states or objects
- action attempts
- success/failure outcomes
- retrieval based on current state similarity
- action selection influenced by prior outcomes

---

## Minimum demo

```text
Episode 1:
State: object = smooth cylinder, task = pick
Action: default grip
Outcome: slip failure
Memory write: smooth cylinder + default grip -> failure

Episode 2:
State: object = smooth cylinder, task = pick
Memory read: default grip failed before
Action: increase grip force
Outcome: success
```

---

## Connection to robotics

This project is the conceptual bridge from general memory agents to robotics memory systems.

It can later connect to:

- RoboGPT memory agents
- failure recovery memory
- deployment memory
- process memory
- plant context memory
- embodied world models

---

## Key insight

Embodied intelligence needs memory of consequences in the world.

---

## Status

Planned.

This project should be built after the core text-based memory stack is complete.