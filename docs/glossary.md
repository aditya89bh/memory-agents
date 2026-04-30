# Memory Agents Glossary

A concise glossary for the concepts used across this repo.

---

## Memory

Stored information that can influence future behavior.

In this repo, memory is not just storage. A memory is useful only if it can shape context, planning, action, or learning.

---

## Context

The information available to the agent at the moment it reasons or acts.

Context may include recent turns, summaries, retrieved memories, goals, constraints, and the current input.

---

## Short-Term Memory

A small rolling buffer of recent interactions.

It preserves continuity across turns but deliberately forgets older information to keep context bounded.

---

## Summary Memory

A compressed representation of older context.

It prevents context explosion while preserving the meaningful storyline of past interactions.

---

## Long-Term Memory

Durable memory that survives beyond the recent conversation window.

It is usually searched or retrieved based on relevance.

---

## Retrieval

The process of finding relevant memories for the current situation.

Retrieval should happen before reasoning, otherwise the agent cannot use the memory effectively.

---

## Salience

The importance of a memory.

Salience helps decide whether something should be stored, discarded, reinforced, or pinned.

---

## Memory Gate

A decision layer that controls what enters memory.

It prevents the system from storing every detail blindly.

---

## Metadata

Structured information attached to a memory.

Examples include memory type, source, timestamp, tags, confidence, and importance score.

---

## Context Assembly

The process of building the final context given to the agent.

This is where short-term memory, summary memory, long-term retrieval, and current input are combined.

---

## Planning

The step where the agent chooses a strategy before acting.

Planning allows memory to influence decisions, not just responses.

---

## Outcome

A result signal from an action.

Outcomes can be success, failure, partial success, or unknown. They become learning signals for future behavior.

---

## Experience Memory

A memory of an attempted action and its outcome.

Example: A strategy failed because required information was missing.

---

## Skill Memory

A reusable pattern learned from repeated task attempts.

Skill memory turns experience into competence.

---

## Identity Memory

Stable memories about an agent or user that preserve consistency over time.

Examples include preferences, traits, long-term goals, and communication style.

---

## World Memory

Memory connected to environments, states, physical spaces, or action outcomes.

World memory is especially important for robotics and embodied agents.

---

## Forgetting

The deliberate removal, compression, or deprioritization of memory.

Forgetting is a feature. Without forgetting, memory becomes noise.

---

## Core principle

Memory becomes intelligence when it changes future action.