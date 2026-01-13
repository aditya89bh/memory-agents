
**New step introduced:**
- **Skill Match** happens before planning

If a valid skill exists, the agent skips deliberation and executes the skill directly.

---

## Key Concepts

### Task
A task represents *what the user is trying to do*.

Examples:
- book_flight
- schedule_meeting
- summarize_text

A task defines:
- required slots
- missing information
- problem shape

---

### Experience
An experience is a record of a past decision and its outcome.

Each experience includes:
- task name
- strategy used
- success or failure
- reason

Experiences are stored as long-term memory but remain **episodic**.

---

### Skill
A skill is a **reusable procedure** derived from repeated successful experiences.

A skill contains:
- associated task
- applicability condition
- strategy to execute
- confidence score (derived from evidence)

Skills are **not manually defined**.  
They are automatically promoted from experience.

---

### Procedural Memory
Skills form procedural memory.

Procedural memory answers:
> “How do I usually do this?”

This is different from factual memory and outcome memory.

---

## Skill Promotion Rules

A skill is created only when:
- the same task + strategy appears multiple times
- the strategy succeeds consistently
- a minimum confidence threshold is reached

This prevents premature or brittle abstraction.

---

## Skill Matching

When a new user input arrives:
1. The task is detected
2. The skill registry is queried
3. Skills are filtered by applicability
4. The most confident skill is selected

If a skill matches:
- the agent uses `use_skill`
- planning is bypassed

---

## Planning Behavior

Decision priority order:

1. **Skill**
2. **Experience-based planning** (Project 4)
3. **Default strategy**

This ensures that proven behavior dominates exploratory behavior.

---

## Outcome Evaluation

After execution:
- the outcome is evaluated (success/failure)
- skill evidence is updated
- experience traces are recorded

Skills gain or lose confidence over time.

---

## Behavior Change Example

**Before skills**
- Agent repeatedly reasons about how to respond
- Same task is re-planned every turn

**After skills**
- Agent immediately applies the learned procedure
- Behavior becomes faster and more consistent

Same input. Same task.  
Different internal behavior due to procedural memory.

---

## What This Project Is Not

Project 5 does not involve:
- reinforcement learning
- neural fine-tuning
- policy gradients
- LLM-based planning trees

This is **symbolic, transparent, and deterministic learning**.

---

## Why This Matters

Project 5 demonstrates that:
- learning does not require model retraining
- competence can emerge from structure + memory
- agents can become less deliberative and more capable over time

---

## Status

- Code complete
- Tested in Colab
- Skills auto-promote from experience
- Skills override planning when applicable

---

## What Comes Next

With Project 5 complete, the agent now has:
- continuity (Project 3)
- experience (Project 4)
- skills (Project 5)

Future extensions may include:
- skill persistence across sessions
- skill composition
- task hierarchies
- skill decay and refinement
