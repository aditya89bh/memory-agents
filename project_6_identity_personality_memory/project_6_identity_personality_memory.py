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

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import time
import json


# ----------------------------
# Utility
# ----------------------------

def now_ts() -> float:
    """Return current UNIX timestamp as float."""
    return time.time()

def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp x to the range [lo, hi]."""
    return max(lo, min(hi, x))


# ----------------------------
# Identity primitives
# ----------------------------

@dataclass
class IdentityItem:
    """
    A single identity-related memory item.

    Categories:
      - user_fact: stable factual statement about user (e.g., "location=Athens")
      - user_pref: preference (e.g., "answer_length=short")
      - agent_trait: agent behavior/style trait (e.g., "tone=analytical")

    Fields:
      - salience (0..1): importance for long-term consistency
      - protection (0..1): overwrite resistance
      - confidence (0..1): belief strength
      - confirmation_count: number of confirmations seen
      - history: audit trail of creates, confirmations, changes, rejections
    """
    key: str
    value: Any
    category: str

    salience: float = 0.7
    protection: float = 0.0
    confidence: float = 0.7

    confirmation_count: int = 0

    created_at: float = field(default_factory=now_ts)
    updated_at: float = field(default_factory=now_ts)

    history: List[Dict[str, Any]] = field(default_factory=list)

    def directive(self) -> str:
        """
        Return a compact directive string that downstream systems can consume verbatim.
        """
        if self.category == "user_pref":
            return f"USER_PREFERENCE::{self.key}={self.value}"
        if self.category == "agent_trait":
            return f"AGENT_TRAIT::{self.key}={self.value}"
        if self.category == "user_fact":
            return f"USER_FACT::{self.key}={self.value}"
        return f"IDENTITY::{self.category}::{self.key}={self.value}"


@dataclass
class Conflict:
    """
    Represents a conflict between an existing identity value and a proposed update.
    """
    key: str
    category: str
    old_value: Any
    new_value: Any
    resolution: str
    score_old: float
    score_new: float
    timestamp: float = field(default_factory=now_ts)


# ----------------------------
# IdentityProfile
# ----------------------------

@dataclass
class IdentityProfile:
    """
    Stores identity memories with explicit boundaries and conflict-aware update logic.

    Buckets:
      - user_facts: stable or semi-stable truths about the user
      - user_prefs: user preferences that override system defaults
      - agent_traits: stable personality knobs for the agent

    This object is the "engine". IdentityMemory is the boundary wrapper.
    """
    user_facts: Dict[str, IdentityItem] = field(default_factory=dict)
    user_prefs: Dict[str, IdentityItem] = field(default_factory=dict)
    agent_traits: Dict[str, IdentityItem] = field(default_factory=dict)

    conflicts: List[Conflict] = field(default_factory=list)

    # Defaults (generic baseline)
    default_agent_traits: Dict[str, Any] = field(default_factory=lambda: {
        "tone": "neutral",
        "verbosity": "medium",
        "style": "clear",
        "emotionality": "low",
    })
    default_user_prefs: Dict[str, Any] = field(default_factory=dict)

    # Protected keys: items that should be harder to overwrite
    protected_keys: Dict[str, List[str]] = field(default_factory=lambda: {
        "user_prefs": [],
        "agent_traits": ["emotionality"],
        "user_facts": [],
    })

    # Update policy parameters
    params: Dict[str, Any] = field(default_factory=lambda: {
        "confirm_boost": 0.08,
        "salience_confirm_boost": 0.03,
        "protection_confirm_boost": 0.04,

        "change_penalty": 0.06,
        "min_confidence": 0.2,
        "max_confidence": 0.98,

        "protected_override_margin": 0.12,

        "auto_protect_after_confirmations": 3,
        "auto_protect_to": 0.6,

        # Confidence threshold for including user facts into directives
        "user_fact_emit_threshold": 0.55,
    })

    def _bucket(self, category: str) -> Dict[str, IdentityItem]:
        if category == "user_fact":
            return self.user_facts
        if category == "user_pref":
            return self.user_prefs
        if category == "agent_trait":
            return self.agent_traits
        raise ValueError(f"Unknown category: {category}")

    def _is_protected_key(self, category: str, key: str) -> bool:
        if category == "user_fact":
            return key in self.protected_keys.get("user_facts", [])
        if category == "user_pref":
            return key in self.protected_keys.get("user_prefs", [])
        if category == "agent_trait":
            return key in self.protected_keys.get("agent_traits", [])
        return False

    def get(self, category: str, key: str) -> Optional[IdentityItem]:
        """Get an IdentityItem by category + key."""
        return self._bucket(category).get(key)

    def upsert(
        self,
        category: str,
        key: str,
        value: Any,
        *,
        salience: float = 0.7,
        confidence: float = 0.7,
        source: str = "implicit",
        protect: Optional[float] = None,
        reason: str = ""
    ) -> Tuple[IdentityItem, Optional[Conflict]]:
        """
        Insert or update an identity item with conflict-aware resolution.

        Parameters
        ----------
        category : str
            One of: "user_fact", "user_pref", "agent_trait"
        key : str
            Stable identifier (e.g., "answer_length", "tone", "location")
        value : Any
            Proposed value for the key
        salience : float (0..1)
            Importance for long-term consistency. Higher means more resistant to drift.
        confidence : float (0..1)
            Belief strength. Higher means more trusted.
        source : str
            "explicit" (direct user statement) > "system" > "implicit" (inferred)
        protect : Optional[float]
            If set, increases overwrite resistance (0..1)
        reason : str
            Human-readable audit note explaining why this update happened

        Returns
        -------
        (IdentityItem, Optional[Conflict])

        Behavior
        --------
        - If key does not exist: create item
        - If value matches existing: treat as confirmation
        - If value conflicts:
            - compute old_score vs new_score using confidence/salience/protection/source
            - protected items require a margin to override
            - accept or reject update; always log conflict + history
        """
        bucket = self._bucket(category)
        existing = bucket.get(key)

        def score(item_conf: float, item_prot: float, item_sal: float, src: str) -> float:
            # Transparent scoring: src acts like evidence quality
            src_bonus = 0.0
            if src == "explicit":
                src_bonus = 0.10
            elif src == "system":
                src_bonus = 0.07
            elif src == "implicit":
                src_bonus = 0.02
            return clamp(item_conf + 0.35 * item_prot + 0.25 * item_sal + src_bonus, 0.0, 2.0)

        # Create new item if none exists
        if existing is None:
            item = IdentityItem(
                key=key,
                value=value,
                category=category,
                salience=clamp(salience, 0.0, 1.0),
                confidence=clamp(confidence, 0.0, 1.0),
                protection=clamp(protect if protect is not None else 0.0, 0.0, 1.0),
            )
            item.history.append({
                "ts": now_ts(),
                "event": "create",
                "value": value,
                "source": source,
                "reason": reason
            })

            # Enforce extra stability if key is in the protected list
            if self._is_protected_key(category, key):
                item.protection = max(item.protection, 0.75)
                item.salience = max(item.salience, 0.85)

            bucket[key] = item
            return item, None

        # Confirmation path: same value
        if existing.value == value:
            existing.confirmation_count += 1
            existing.confidence = clamp(existing.confidence + self.params["confirm_boost"], 0.0, self.params["max_confidence"])
            existing.salience = clamp(existing.salience + self.params["salience_confirm_boost"], 0.0, 1.0)

            # Optional: protection can increase slightly on confirmations
            existing.protection = clamp(existing.protection + self.params["protection_confirm_boost"], 0.0, 1.0)

            existing.updated_at = now_ts()
            existing.history.append({
                "ts": now_ts(),
                "event": "confirm",
                "value": value,
                "source": source,
                "reason": reason
            })

            # Auto-protect after repeated confirmations
            if existing.confirmation_count >= self.params["auto_protect_after_confirmations"]:
                existing.protection = max(existing.protection, self.params["auto_protect_to"])

            return existing, None

        # Conflict path: different value proposed
        old_score = score(existing.confidence, existing.protection, existing.salience, "implicit")
        new_prot = clamp(protect if protect is not None else 0.0, 0.0, 1.0)
        new_score = score(clamp(confidence, 0.0, 1.0), new_prot, clamp(salience, 0.0, 1.0), source)

        is_key_protected = self._is_protected_key(category, key) or existing.protection >= 0.7
        margin = self.params["protected_override_margin"] if is_key_protected else 0.0

        # Accept update only if new evidence is stronger than old, plus margin if protected
        if new_score > (old_score + margin):
            old_value = existing.value
            existing.value = value
            existing.updated_at = now_ts()

            # Confidence update: explicit changes are trusted more
            if source == "explicit":
                existing.confidence = clamp(max(existing.confidence, confidence), 0.0, self.params["max_confidence"])
            else:
                existing.confidence = clamp((existing.confidence + confidence) / 2.0, self.params["min_confidence"], self.params["max_confidence"])

            # Protection: only bump if requested
            if protect is not None:
                existing.protection = max(existing.protection, new_prot)

            # Penalize flip-flops a bit
            existing.confidence = clamp(existing.confidence - self.params["change_penalty"], self.params["min_confidence"], self.params["max_confidence"])

            existing.history.append({
                "ts": now_ts(),
                "event": "change",
                "from": old_value,
                "to": value,
                "source": source,
                "reason": reason
            })

            conflict = Conflict(
                key=key,
                category=category,
                old_value=old_value,
                new_value=value,
                resolution="accepted_new_value",
                score_old=old_score,
                score_new=new_score
            )
            self.conflicts.append(conflict)
            return existing, conflict

        # Reject update, keep old
        conflict = Conflict(
            key=key,
            category=category,
            old_value=existing.value,
            new_value=value,
            resolution="kept_existing_value",
            score_old=old_score,
            score_new=new_score
        )
        self.conflicts.append(conflict)

        existing.history.append({
            "ts": now_ts(),
            "event": "reject_change",
            "attempted": value,
            "source": source,
            "reason": reason,
            "old_score": old_score,
            "new_score": new_score,
            "protected": is_key_protected
        })

        return existing, conflict

    def identity_directives(self) -> List[str]:
        """
        Build compact directive strings for downstream systems.

        Output is intentionally flat and human-readable to keep integration simple.

        Priority:
          1) Agent traits (merged with defaults)
          2) User preferences (merged with defaults)
          3) User facts (only if confidence >= threshold)
        """
        directives: List[str] = []

        # 1) Agent traits (defaults merged with stored)
        merged_traits = dict(self.default_agent_traits)
        for k, item in self.agent_traits.items():
            merged_traits[k] = item.value
        for k, v in merged_traits.items():
            directives.append(f"AGENT_TRAIT::{k}={v}")

        # 2) User preferences (defaults merged with stored)
        merged_prefs = dict(self.default_user_prefs)
        for k, item in self.user_prefs.items():
            merged_prefs[k] = item.value
        for k, v in merged_prefs.items():
            directives.append(f"USER_PREFERENCE::{k}={v}")

        # 3) User facts (only high-confidence)
        thr = float(self.params.get("user_fact_emit_threshold", 0.55))
        for item in self.user_facts.values():
            if item.confidence >= thr:
                directives.append(item.directive())

        return directives

    def to_json(self) -> str:
        """Serialize full IdentityProfile to JSON (including audit history and conflicts)."""
        def item_to_dict(it: IdentityItem) -> Dict[str, Any]:
            return {
                "key": it.key,
                "value": it.value,
                "category": it.category,
                "salience": it.salience,
                "protection": it.protection,
                "confidence": it.confidence,
                "confirmation_count": it.confirmation_count,
                "created_at": it.created_at,
                "updated_at": it.updated_at,
                "history": it.history,
            }

        payload = {
            "user_facts": {k: item_to_dict(v) for k, v in self.user_facts.items()},
            "user_prefs": {k: item_to_dict(v) for k, v in self.user_prefs.items()},
            "agent_traits": {k: item_to_dict(v) for k, v in self.agent_traits.items()},
            "conflicts": [c.__dict__ for c in self.conflicts],
            "default_agent_traits": self.default_agent_traits,
            "default_user_prefs": self.default_user_prefs,
            "protected_keys": self.protected_keys,
            "params": self.params,
        }
        return json.dumps(payload, indent=2)

    @staticmethod
    def from_json(s: str) -> "IdentityProfile":
        """Load IdentityProfile from JSON created by to_json()."""
        raw = json.loads(s)

        def dict_to_item(d: Dict[str, Any]) -> IdentityItem:
            return IdentityItem(
                key=d["key"],
                value=d["value"],
                category=d["category"],
                salience=d["salience"],
                protection=d["protection"],
                confidence=d["confidence"],
                confirmation_count=d.get("confirmation_count", 0),
                created_at=d["created_at"],
                updated_at=d["updated_at"],
                history=d.get("history", []),
            )

        prof = IdentityProfile()
        prof.user_facts = {k: dict_to_item(v) for k, v in raw.get("user_facts", {}).items()}
        prof.user_prefs = {k: dict_to_item(v) for k, v in raw.get("user_prefs", {}).items()}
        prof.agent_traits = {k: dict_to_item(v) for k, v in raw.get("agent_traits", {}).items()}

        prof.conflicts = [Conflict(**c) for c in raw.get("conflicts", [])]
        prof.default_agent_traits = raw.get("default_agent_traits", prof.default_agent_traits)
        prof.default_user_prefs = raw.get("default_user_prefs", prof.default_user_prefs)
        prof.protected_keys = raw.get("protected_keys", prof.protected_keys)
        prof.params = raw.get("params", prof.params)
        return prof


# ----------------------------
# IdentityMemory wrapper (Project boundary)
# ----------------------------

class IdentityMemory:
    """
    IdentityMemory (Project Boundary Object)
    ----------------------------------------

    This is the only object the rest of the system needs to talk to.

    Public methods:
    - update_user_preference(key, value, source="explicit", confidence=..., salience=..., reason="")
    - update_user_fact(key, value, source="explicit", confidence=..., salience=..., reason="")
    - set_agent_trait(key, value, source="system", confidence=..., salience=..., protect=..., reason="")
    - directives() -> List[str]
    - export_json() -> str
    - import_json(json_str) -> IdentityMemory

    Notes:
    - Protected items resist overwrite unless new evidence is stronger.
    - Conflicts are logged and items keep an audit history.
    """
    def __init__(self, profile: Optional[IdentityProfile] = None):
        self.profile = profile or IdentityProfile()

    def update_user_preference(
        self,
        key: str,
        value: Any,
        *,
        source: str = "explicit",
        confidence: float = 0.8,
        salience: float = 0.8,
        reason: str = ""
    ) -> Tuple[IdentityItem, Optional[Conflict]]:
        """Upsert a user preference (how the user wants the agent to behave)."""
        return self.profile.upsert(
            "user_pref",
            key,
            value,
            source=source,
            confidence=confidence,
            salience=salience,
            reason=reason
        )

    def update_user_fact(
        self,
        key: str,
        value: Any,
        *,
        source: str = "explicit",
        confidence: float = 0.75,
        salience: float = 0.65,
        reason: str = ""
    ) -> Tuple[IdentityItem, Optional[Conflict]]:
        """Upsert a user fact (declarative truth about the user)."""
        return self.profile.upsert(
            "user_fact",
            key,
            value,
            source=source,
            confidence=confidence,
            salience=salience,
            reason=reason
        )

    def set_agent_trait(
        self,
        key: str,
        value: Any,
        *,
        source: str = "system",
        confidence: float = 0.85,
        salience: float = 0.9,
        protect: Optional[float] = None,
        reason: str = ""
    ) -> Tuple[IdentityItem, Optional[Conflict]]:
        """Set an agent trait (stable personality knob)."""
        return self.profile.upsert(
            "agent_trait",
            key,
            value,
            source=source,
            confidence=confidence,
            salience=salience,
            protect=protect,
            reason=reason
        )

    def directives(self) -> List[str]:
        """Return directives to inject into planner/response context."""
        return self.profile.identity_directives()

    def export_json(self) -> str:
        """Serialize profile to JSON for persistence."""
        return self.profile.to_json()

    @staticmethod
    def import_json(s: str) -> "IdentityMemory":
        """Deserialize IdentityMemory from JSON."""
        return IdentityMemory(profile=IdentityProfile.from_json(s))


# ----------------------------
# Minimal Unified Memory Stack Integration Stub
# ----------------------------

class UnifiedMemoryStack:
    """
    Integration stub showing how IdentityMemory plugs into your UnifiedMemoryStack.
    Replace this with your real stack (Projects 1–5), and keep the identity interface.
    """
    def __init__(self, identity_memory: IdentityMemory):
        self.identity_memory = identity_memory

    def build_context(self, user_message: str) -> Dict[str, Any]:
        """
        Build a context object for planners/response generators.

        Identity directives are added as a side-channel so downstream systems
        can shape behavior without handling raw IdentityItems.
        """
        return {
            "user_message": user_message,
            "identity_directives": self.identity_memory.directives(),
        }


# ----------------------------
# Demo + sanity checks
# ----------------------------

# Quickstart (manual)
# -------------------
# ident = IdentityMemory()
# ident.set_agent_trait("tone", "analytical", protect=0.7)
# ident.update_user_preference("answer_length", "short", source="explicit")
# print(ident.directives())

def _demo():
    ident = IdentityMemory()

    # Set stable agent traits
    ident.set_agent_trait(
        "tone",
        "analytical",
        protect=0.7,
        reason="Project 6 baseline: analytical, not emotional"
    )
    ident.set_agent_trait(
        "verbosity",
        "concise",
        protect=0.6,
        reason="Prefer concise by default for this user"
    )

    # Add user preference explicitly
    ident.update_user_preference(
        "answer_length",
        "short",
        source="explicit",
        reason="User said they prefer concise answers"
    )

    # Confirm the same preference (should boost confidence + protection)
    ident.update_user_preference(
        "answer_length",
        "short",
        source="implicit",
        reason="User keeps asking for shorter outputs"
    )
    ident.update_user_preference(
        "answer_length",
        "short",
        source="implicit",
        reason="User edits prompts to keep it tight"
    )

    # Attempt a conflicting update (weak, implicit) likely rejected
    _, conflict_weak = ident.update_user_preference(
        "answer_length",
        "long",
        source="implicit",
        confidence=0.45,
        salience=0.5,
        reason="One interaction asked for more detail"
    )

    # Now explicit override (should be accepted)
    _, conflict_strong = ident.update_user_preference(
        "answer_length",
        "long",
        source="explicit",
        confidence=0.85,
        salience=0.8,
        reason="User explicitly changed preference"
    )

    # Protected trait conflict: emotionality is protected by default protected_keys
    ident.set_agent_trait(
        "emotionality",
        "low",
        protect=0.8,
        reason="Keep agent analytical"
    )
    _, conflict_trait = ident.set_agent_trait(
        "emotionality",
        "high",
        source="implicit",
        confidence=0.6,
        salience=0.6,
        reason="Inferred from user vibe (should be resisted)"
    )

    # Add a user fact
    ident.update_user_fact(
        "location",
        "Athens",
        source="explicit",
        confidence=0.8,
        reason="User stated location"
    )

    stack = UnifiedMemoryStack(identity_memory=ident)
    ctx = stack.build_context("Explain the architecture please.")

    print("=== Identity Directives ===")
    for d in ctx["identity_directives"]:
        print(" -", d)

    print("\n=== Conflicts Logged (latest) ===")
    for c in ident.profile.conflicts[-5:]:
        print(f" - key={c.key} category={c.category} resolution={c.resolution} old={c.old_value} new={c.new_value}")

    if conflict_weak:
        print("\nWeak conflict example (expected to be kept existing):", conflict_weak.resolution)
    if conflict_strong:
        print("Strong conflict example (expected accepted):", conflict_strong.resolution)
    if conflict_trait:
        print("Trait conflict example (expected resisted):", conflict_trait.resolution)

    # Export/import roundtrip
    saved = ident.export_json()
    ident2 = IdentityMemory.import_json(saved)
    assert ident2.directives() == ident.directives(), "Import/export mismatch"

    print("\n✅ Demo complete. IdentityMemory is runnable and serializable.")


if __name__ == "__main__":
    _demo()
