# 🧠 Memory Agents – From Continuity to Cognition

This repository is a **progressive exploration of memory systems for AI agents**.

The goal is not to build a single “chatbot with memory”, but to **systematically design, implement, and understand memory as a cognitive system**—the same way humans use memory for continuity, planning, learning, and identity.

Each project builds on the previous one. Nothing is skipped. Nothing is hidden behind frameworks.

---

## Why Memory Agents?

Most AI systems today are:
- stateless
- reactive
- short-lived
- context-fragile

Real intelligence requires **memory**:
- memory of the past
- memory of what matters
- memory that shapes future decisions

This repository treats memory as a **first-class design problem**, not a feature toggle.

---

## Project Overview

Project 1 → continuity
Project 2 → retrieval & judgment
Project 3 → integration
Project 4 → planning
Project 5 → learning
Project 6 → identity
Project 7 → embodiment


---

## 📘 Project 1 – Short-Term Memory (Continuity)

**Goal:**  
Give the agent continuity across turns.

### What’s built
- Rolling window memory
- Explicit forgetting
- Deterministic context size

### Key insight
Memory is not “what you store”.  
Memory is **what you choose to forget**.

📁 Folder:


---

## 📘 Project 1B – Summary Memory (Compression)

**Goal:**  
Prevent context explosion while preserving meaning.

### What’s built
- Two-tier memory:
  - recent raw buffer
  - running summary
- No recursive memory bloat

### Key insight
Chronology doesn’t scale.  
**Compression is intelligence.**

📁 Folder:
project-1b-summary-memory/
