Project 7 – Experience Memory for Agents
State → Action → Outcome Learning Layer

This project implements Experience Memory: a state–action–outcome learning layer that allows an agent to learn from outcomes in specific contexts, not just conversations or facts.

Unlike traditional chatbot memory, this system remembers what situation occurred, what action or strategy was chosen, and what actually happened. Over time, this enables agents to avoid repeating failures and prefer strategies that worked before, without retraining.

Most agents today can remember conversations, retrieve knowledge, and plan actions, but they still repeat the same mistakes, ignore past failures in similar contexts, and feel stateless across environments. This happens because they lack experience memory. Most chatbots remember conversations. This agent remembers outcomes.

The core idea is simple. Every experience is stored as a structured triple:
STATE → ACTION → OUTCOME

Examples include:
In this context, strategy B worked better.
Last time this state occurred, action A failed.
This task behaves differently under these constraints.

This is experiential memory, not knowledge memory.

The system provides the following capabilities:
It stores experiences as structured records.
It retrieves past experiences relevant to a given context.
It ranks experiences by usefulness.
It aggregates experience by action or strategy.
It advises the planner on what tends to work.

The system never makes decisions on its own.

This project explicitly does not perform planning or decision-making, does not store user preferences or identity, does not store factual knowledge, does not remember conversations, and does not build spatial maps or world models. The scope is intentionally narrow and debuggable.

Each experience record contains four parts: state, action, outcome, and metadata.

State is a structured snapshot of the context. The field task is required. All other fields are optional.

Example state:
{
  "task": "summarize_document",
  "env": "web_chat",
  "constraints": ["concise", "time_limited"],
  "signals": { "domain": "technical", "length": "long" },
  "tags": ["nlp"]
}
