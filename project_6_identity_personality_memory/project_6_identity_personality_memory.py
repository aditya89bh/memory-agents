"""
Project 6 — Identity & Personality Memory
========================================

Purpose
-------
Adds long-horizon consistency to a memory-agent by storing and applying:
- User facts (declarative, about the user)
- User preferences (how the user wants the agent to behave)
- Agent traits (the agent's own personality defaults)

This module prevents "personality drift" by introducing:
- Salience (importance)
- Confidence (belief strength)
- Protection (overwrite resistance)
- Conflict-aware updates (no blind overwrites)
- Audit history (change tracking)

Design Goals
------------
- Code-first and Colab-friendly
- Minimal frameworks (standard library only)
- Transparent logic (explicit scoring + thresholds)
- Clear project boundary: IdentityMemory is the integration surface
- JSON serialization for persistence

How It Integrates
-----------------
Your unified memory stack should treat this module as a side-channel that produces
identity directives for downstream systems (planner / response generator).

Typical flow:
1) IdentityMemory receives updates (explicit or implicit signals)
2) UnifiedMemoryStack calls IdentityMemory.directives()
3) Planner/response layer reads directives to shape behavior

Example directives:
- AGENT_TRAIT::tone=analytical
- USER_PREFERENCE::answer_length=short
- USER_FACT::location=Athens

Non-Goals
---------
- Automatic preference extraction from text (handled elsewhere)
- World / episodic memory (Project 7)
- Planning algorithms (Project 4/5)
"""
