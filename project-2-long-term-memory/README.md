# Project 2 – Long-Term Memory (Vector-Based Recall)

This project explores **long-term memory for agents**, where past information is
stored persistently and retrieved based on relevance rather than recency.

Unlike short-term or summary memory, this memory is:
- searchable
- durable
- independent of the recent conversation window

This is the foundation of Retrieval-Augmented Generation (RAG) and memory-driven agents.

---

## Core Idea

Instead of remembering:
> “what happened last”

the agent learns to remember:
> “what matters right now”

This is achieved by:
- converting memories into vectors
- retrieving the most relevant ones at query time
- injecting them into the agent’s context

---

## Structure of Project 2

Project 2 is intentionally **iterative**. Each step builds on the previous one.

### 2A – Basic Vector Recall (current)
**Goal:**  
Prove that semantic recall works at all.

**Features:**
- Text-based memories
- TF-IDF vectorization
- Cosine similarity search
- Top-k memory retrieval

**What it answers:**  
Can the agent recall relevant past facts when asked?

---

### 2B – Metadata-Aware Memory (next)
**Goal:**  
Make recall controllable and context-aware.

**Adds:**
- Metadata (type, source, timestamp, tags)
- Filtered retrieval (e.g. identity vs preferences)
- Better memory organization

**What it answers:**  
Which memories should be considered *in this situation*?

---

### 2C – Salience & Memory Gating (planned)
**Goal:**  
Prevent memory bloat and junk storage.

**Adds:**
- Importance scoring
- Store-or-discard decisions
- Optional decay or reinforcement

**What it answers:**  
What is worth remembering long-term?

---

### 2D – Improved Embeddings (planned)
**Goal:**  
Improve recall quality.

**Changes:**
- TF-IDF → neural embeddings (Sentence Transformers)
- Higher semantic understanding
- Better recall across paraphrases

**What it answers:**  
Can memory recall feel more human and less keyword-based?

## Dependencies (2D)
Project 2D uses Sentence Transformers:

pip install sentence-transformers


---

## Design Principles

- Memory is **explicit**, not implicit
- Forgetting is a feature, not a bug
- Retrieval happens **before reasoning**
- Memory storage and memory usage are separate concerns

This project avoids agent frameworks to keep the mechanics transparent.

---

## Relation to Earlier Projects

- **Project 1:** Short-term memory (rolling window)
- **Project 1B:** Summary memory (compression over time)
- **Project 2:** Long-term memory (semantic retrieval)

Together, these form a complete memory stack.

---

## Notes

- The current implementation is Colab-friendly and lightweight.
- Production systems would replace TF-IDF with neural embeddings
  and use persistent vector databases.

This project focuses on understanding the architecture first.
