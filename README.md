# 🧠 Memory Agents – From Continuity to Cognition

This repository is a **progressive exploration of memory systems for AI agents**.

The goal is not to build a single “chatbot with memory”, but to **systematically design, implement, and understand memory as a cognitive system**—the same way humans use memory for continuity, planning, learning, and identity.

Each project builds on the previous one. Nothing is skipped. Nothing is hidden behind frameworks.

---

## Why Memory Agents?

Most AI systems today are stateless, reactive, short-lived, and context-fragile.

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

**What’s built:**
- Rolling window memory  
- Explicit forgetting  
- Deterministic context size  

**Key insight:**  
Memory is not what you store.  
Memory is **what you choose to forget**.

📁 `project-1-short-term-memory/`

---
## 📘 Project 1B – Summary Memory (Compression)

**Goal:**  
Prevent context explosion while preserving meaning.

**What’s built:**
- Two-tier memory (recent buffer + running summary)  
- No recursive memory bloat  

**Key insight:**  
Chronology does not scale.  
**Compression is intelligence.**

📁 `project-1b-summary-memory/`

---

## 📘 Project 2 – Long-Term Memory (Retrieval)

**Goal:**  
Move from recent context to **searchable experience**.

Project 2 is broken into focused sub-projects.

**2A – Vector Recall**
- TF-IDF embeddings  
- Cosine similarity  
- Top-k semantic recall  

**Question answered:**  
Can the agent recall relevant past information at all?

**2B – Metadata-Aware Memory**
- Memory types (identity, preference, goal, fact)  
- Tags, sources, filters  

**Question answered:**  
Which memories should be considered right now?

**2C – Salience & Memory Gating**
- Importance scoring  
- Store vs discard decisions  
- Pinned memories  

**Question answered:**  
What is worth remembering long-term?

**2D – Neural Embeddings**
- Sentence Transformers  
- Paraphrase-robust recall  

**Question answered:**  
Can recall feel semantic instead of keyword-based?

📁 `project-2-long-term-memory/`

---

## 📘 Project 3 – Unified Memory Stack (Integration)

**Goal:**  
Make memory feel like **one brain**, not multiple modules.

**What’s built:**
- Unified `MemoryManager`  
- Clear read/write phases  
- Single context assembly pipeline  

**Architecture flow:**

Input → Memory Gate → Short-Term Buffer → Summary Memory →  
Long-Term Retrieval → Context Assembly → Agent Reasoning

**Why this matters:**  
This is the minimum viable real agent architecture.

📁 `project-3-unified-memory-stack/`

---

## 📘 Project 4 – Memory + Planning (Cognition)

**Goal:**  
Make memory influence **decisions**, not just answers.

**What’s built:**
- Action history memory  
- Outcome memory (success / failure)  
- Planner that consults past experience  

**Example:**  
“Last time this failed, try a different strategy.”

📁 `project-4-memory-planning/`

---

## 📘 Project 5 – Skill & Task Memory (Learning)

**Goal:**  
Turn repetition into reusable competence.

**What’s built:**
- Task attempt memory  
- Skill abstraction  
- Task-to-skill mapping  

**Example:**  
“This looks like a task I’ve done before.”

📁 `project-5-skill-memory/`

---

## 📘 Project 6 – Identity & Personality Memory

**Goal:**  
Make the agent consistent across weeks and months.

**What will be built:**
- Stable identity memory  
- Long-term preferences  
- Trait resolution logic  

**Example:**  
“This user prefers concise answers and Python-first solutions.”

📁 `project-6-identity-memory/`

---

## 📘 Project 7 – Embodied / World Memory

**Goal:**  
Tie memory to environments, not just text.

**Possible directions:**
- Robotics world memory  
- Simulated environments  
- State- or spatial-aware memory  

**Example:**  
“In this environment, path B was safer last time.”

📁 `project-7-embodied-memory/`

---

## Design Principles

- Memory is explicit, never implicit  
- Retrieval happens before reasoning  
- Forgetting is a feature  
- Salience beats volume  
- Minimal frameworks, maximum clarity  
- Colab-first, GitHub-second  

---

## Status

- Projects 1–5: ✅ Completed  
- Projects 6–7: 🧭 Planned  

---

If someone reads just this README, they should understand **how memory evolves into intelligence**.
