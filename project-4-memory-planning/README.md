# Project 4 – Memory + Planning

Project 4 extends the memory-agent architecture by introducing **planning, outcomes, and experience-based decision making**.

Up to Project 3, memory provided continuity and context.  
From Project 4 onward, memory influences **what the agent decides to do**.

---

## Core Idea

The agent should remember **what worked**, **what failed**, and **choose differently next time**.

If the same user input produces different behavior over time **because of memory**, this project succeeds.

---

## Agent Lifecycle

Project 3 lifecycle:

Read → Assemble Context → Respond → Write

Project 4 lifecycle:

Read → Assemble Context → Plan → Act → Evaluate → Write

Two new phases are introduced:
- **Plan**: select a strategy before acting
- **Evaluate**: determine success or failure after acting

Memory now closes the learning loop.

---

## System Components

### Data Types
The system reasons over explicit, typed structures:
- MemoryItem – atomic unit of long-term memory
- Turn – one interaction step
- Plan – chosen strategy and rationale
- Outcome – success or failure signal

Memory is typed, inspectable, and queryable.

---

### Short-Term Memory (Working Memory)
- Rolling window of recent turns
- Automatically forgets older interactions
- Holds immediate conversational state

This functions as working memory (RAM).

---

### Summary Memory (Compression Layer)
- Compresses recent turns deterministically
- Preserves continuity without context bloat

Prevents context window explosion while maintaining narrative flow.

---

### Long-Term Memory (Context + Experience)
Stores retrievable memory using similarity search.

Memory types include:
- identity
- preferences
- goals
- facts
- experience

Long-term memory does not answer queries.  
It conditions planning and reasoning.

---

### Adapter Wrapper
Wraps the long-term memory backend.

Decouples the MemoryManager from storage implementation, allowing TF-IDF, embeddings, hybrid stores, or knowledge graphs to be swapped without rewriting system logic.

---

### Context Assembly
Determines what the agent actually sees.

Context is assembled in the following order:
1. Pinned identity, preferences, and goals
2. Relevant long-term recalls
3. Summary memory
4. Recent short-term turns
5. Current user message

Without explicit context assembly, memory does not reliably influence behavior.

---

### Planning
A lightweight, deterministic planner selects a strategy before the agent responds.

Example strategies:
- direct_answer
- ask_clarify

The planner:
- detects intent
- checks for missing information
- consults past experience memories
- avoids repeating known failures

---

### Outcome Evaluation
After acting, the system evaluates the result.

Each outcome includes:
- success or failure
- reason
- optional score

Outcomes are explicit learning signals.

---

### Experience Memory
Outcomes are stored as long-term memory of type:

experience

Each experience records:
- task or intent
- strategy used
- success or failure
- reason

This enables the agent to learn from its own behavior.

---

### Memory Manager (Core)
The MemoryManager enforces the full lifecycle:

Read → Assemble Context → Plan → Act → Evaluate → Write

The agent never manipulates memory directly.  
This guarantees discipline and consistency.

---

### Agent
The agent:
- follows the enforced lifecycle
- executes the selected plan
- does not control memory behavior

Intelligence comes later.  
Discipline is established here.

---

## Example Behavior

Turn 1:
- User asks: “Can you book a flight for me?”
- Planner selects a strategy
- Action fails due to missing information
- Failure is stored as experience memory

Turn 2:
- Same user request
- Planner retrieves past failure
- Chooses a different strategy
- Asks for clarification instead

Behavior changes **because of memory**.

---

## How to Run

Open the Project 4 Colab notebook and run all cells.

The notebook demonstrates:
- planning before action
- outcome evaluation
- experience storage
- memory-driven behavior change

---

## Status

Complete  
Tested in Colab  
Demonstrates memory-influenced decision making

---

## What Comes Next

With Project 4 complete, the agent now has:
- continuity (Project 3)
- experience (Project 4)

Project 5 will introduce:
- reusable skills
- task abstraction
- procedural memory
- competence over time
