"""
Project 7A — Experience Memory (World / Embodied Memory)
=======================================================

Purpose
-------
Store and retrieve experiential memory as State → Action → Outcome records.
This enables an agent to learn from outcomes in specific contexts without retraining.

PRD Locked Behavior (MVP)
-------------------------
- Hard filter: task must match exactly
- Optional filters: env/phase match if provided
- Similarity: transparent weighted overlap (no embeddings)
- Record ranking: 0.55 * similarity + 0.25 * outcome_quality + 0.20 * recency
- Recommendations: aggregate by action signature; advise planner, never decide
- Persistence: JSON export/import

Non-goals
---------
- Planning logic (Project 5 decides)
- Identity/preferences (Project 6)
- Long-term knowledge retrieval (Project 3)
- Spatial mapping / SLAM (future 7B/7C)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import time
import json
import math
import uuid
from collections import defaultdict, Counter


# ----------------------------
# Utilities
# ----------------------------

def now_ts() -> float:
    return time.time()

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def days_ago(ts: float) -> float:
    return (now_ts() - ts) / 86400.0

def stable_json(obj: Any) -> str:
    """Deterministic JSON string (for signatures/debug)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


# ----------------------------
# Data model
# ----------------------------

@dataclass
class ExperienceRecord:
    """
    Atomic experience unit: State → Action → Outcome + metadata.

    Invariants:
    - state must include 'task'
    - outcome must include 'success' (bool)
    """
    record_id: str
    state: Dict[str, Any]
    action: Dict[str, Any]
    outcome: Dict[str, Any]

    ts: float = field(default_factory=now_ts)
    episode_id: Optional[str] = None
    salience: float = 0.5
    source: str = "planner"  # planner/sim/robot/etc.

    # Filled at query-time for explainability
    similarity: float = 0.0
    outcome_quality: float = 0.0
    recency: float = 0.0
    rank_score: float = 0.0


# ----------------------------
# Experience Memory (Project Boundary)
# ----------------------------

class ExperienceMemory:
    """
    ExperienceMemory (Project 7 boundary object)

    Public API:
    - add_experience(state, action, outcome, salience=0.5, episode_id=None, source="planner") -> record_id
    - query(query_state, k=10, min_similarity=0.0, filters=None) -> List[ExperienceRecord]
    - recommend(query_state, k_actions=5, k_records=25, min_similarity=0.0, filters=None)
        -> Dict with:
           - recommendations: List[dict]
           - supporting_records: List[ExperienceRecord]
    - export_json() / import_json()
    """

    def __init__(self):
        self._records: List[ExperienceRecord] = []

        # Similarity weights (PRD Step 5)
        self.sim_weights = {
            "env": 0.20,
            "phase": 0.10,
            "constraints": 0.25,
            "signals": 0.35,
            "tags": 0.10,
        }

        # Record ranking weights (PRD Step 5)
        self.rank_weights = {
            "similarity": 0.55,
            "outcome_quality": 0.25,
            "recency": 0.20,
        }

        # Recency half-life (days) (PRD Step 5 default)
        self.half_life_days = 14.0

        # Recommendation aggregation weights (PRD Step 5)
        self.action_score_weights = {
            "success_rate": 0.60,
            "avg_quality": 0.25,
            "recency_last_success": 0.15,
        }

    # ----------------------------
    # Validation
    # ----------------------------

    @staticmethod
    def _validate_state(state: Dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise ValueError("state must be a dict")
        if "task" not in state or not isinstance(state["task"], str) or not state["task"].strip():
            raise ValueError("state must include a non-empty string field 'task'")

    @staticmethod
    def _validate_action(action: Dict[str, Any]) -> None:
        if not isinstance(action, dict):
            raise ValueError("action must be a dict")
        # soft requirement: either strategy or skill should exist
        if "strategy" not in action and "skill" not in action:
            raise ValueError("action should include at least 'strategy' or 'skill'")

    @staticmethod
    def _validate_outcome(outcome: Dict[str, Any]) -> None:
        if not isinstance(outcome, dict):
            raise ValueError("outcome must be a dict")
        if "success" not in outcome or not isinstance(outcome["success"], bool):
            raise ValueError("outcome must include boolean field 'success'")

    # ----------------------------
    # Add
    # ----------------------------

    def add_experience(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        outcome: Dict[str, Any],
        *,
        salience: float = 0.5,
        episode_id: Optional[str] = None,
        source: str = "planner",
    ) -> str:
        """
        Store a new experience record and return its record_id.
        """
        self._validate_state(state)
        self._validate_action(action)
        self._validate_outcome(outcome)

        rec = ExperienceRecord(
            record_id=str(uuid.uuid4()),
            state=state,
            action=action,
            outcome=outcome,
            salience=clamp(float(salience), 0.0, 1.0),
            episode_id=episode_id,
            source=source,
        )
        self._records.append(rec)
        return rec.record_id

    # ----------------------------
    # Similarity + Ranking
    # ----------------------------

    @staticmethod
    def _jaccard(a: List[str], b: List[str]) -> float:
        sa, sb = set(a), set(b)
        if not sa and not sb:
            return 1.0
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    @staticmethod
    def _safe_list(x: Any) -> List[str]:
        if x is None:
            return []
        if isinstance(x, list):
            return [str(v) for v in x]
        return [str(x)]

    @staticmethod
    def _safe_dict(x: Any) -> Dict[str, Any]:
        if x is None:
            return {}
        if isinstance(x, dict):
            return x
        return {}

    def _signals_similarity(self, qs: Dict[str, Any], rs: Dict[str, Any]) -> float:
        """
        Signals similarity rewards:
        - key overlap
        - exact value matches on overlapping keys
        Missing keys are not heavily penalized (MVP: ignore absent keys).
        """
        q = self._safe_dict(qs)
        r = self._safe_dict(rs)
        if not q and not r:
            return 1.0
        if not q or not r:
            return 0.0

        q_keys = set(q.keys())
        r_keys = set(r.keys())
        overlap = q_keys & r_keys
        if not overlap:
            return 0.0

        # key overlap portion
        key_overlap = len(overlap) / max(len(q_keys), len(r_keys))

        # value match portion
        matches = 0
        for k in overlap:
            if str(q.get(k)) == str(r.get(k)):
                matches += 1
        value_match = matches / len(overlap)

        # combine (transparent)
        return clamp(0.45 * key_overlap + 0.55 * value_match, 0.0, 1.0)

    def state_similarity(self, query_state: Dict[str, Any], record_state: Dict[str, Any]) -> float:
        """
        Weighted overlap similarity (PRD Step 5).
        Hard filter on 'task' is applied elsewhere.
        """
        q_env = str(query_state.get("env", "")).strip()
        r_env = str(record_state.get("env", "")).strip()
        env_sim = 1.0 if (q_env and r_env and q_env == r_env) else (1.0 if (not q_env and not r_env) else 0.0)

        q_phase = str(query_state.get("phase", "")).strip()
        r_phase = str(record_state.get("phase", "")).strip()
        phase_sim = 1.0 if (q_phase and r_phase and q_phase == r_phase) else (1.0 if (not q_phase and not r_phase) else 0.0)

        q_constraints = self._safe_list(query_state.get("constraints"))
        r_constraints = self._safe_list(record_state.get("constraints"))
        constraints_sim = self._jaccard(q_constraints, r_constraints)

        q_tags = self._safe_list(query_state.get("tags"))
        r_tags = self._safe_list(record_state.get("tags"))
        tags_sim = self._jaccard(q_tags, r_tags)

        signals_sim = self._signals_similarity(query_state.get("signals"), record_state.get("signals"))

        w = self.sim_weights
        sim = (
            w["env"] * env_sim +
            w["phase"] * phase_sim +
            w["constraints"] * constraints_sim +
            w["signals"] * signals_sim +
            w["tags"] * tags_sim
        )
        return clamp(sim, 0.0, 1.0)

    def _outcome_quality(self, outcome: Dict[str, Any]) -> float:
        """
        Outcome quality (PRD Step 5):
        - if 'score' exists, use it (0..1)
        - else 1.0 for success, 0.0 for failure
        """
        if "score" in outcome and outcome["score"] is not None:
            try:
                return clamp(float(outcome["score"]), 0.0, 1.0)
            except Exception:
                pass
        return 1.0 if bool(outcome.get("success", False)) else 0.0

    def _recency_score(self, ts: float) -> float:
        """
        Recency score using exponential decay with half-life in days (PRD Step 5).
        recency = exp(-age_days / half_life_days)
        """
        age = max(0.0, days_ago(ts))
        hl = max(0.001, float(self.half_life_days))
        return float(math.exp(-age / hl))

    def _rank_record(self, similarity: float, outcome_quality: float, recency: float) -> float:
        w = self.rank_weights
        score = (
            w["similarity"] * similarity +
            w["outcome_quality"] * outcome_quality +
            w["recency"] * recency
        )
        return float(score)

    # ----------------------------
    # Query
    # ----------------------------

    def query(
        self,
        query_state: Dict[str, Any],
        *,
        k: int = 10,
        min_similarity: float = 0.0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ExperienceRecord]:
        """
        Retrieve top-k experiences relevant to query_state.

        Hard filter (PRD):
        - task must match exactly

        Optional filters:
        - if filters contains env/phase, require match
        """
        self._validate_state(query_state)
        filters = filters or {}

        q_task = query_state["task"]

        # Optional filters: if provided, enforce
        f_env = filters.get("env")
        f_phase = filters.get("phase")

        candidates: List[ExperienceRecord] = []

        for rec in self._records:
            r_task = rec.state.get("task")
            if r_task != q_task:
                continue  # hard filter

            if f_env is not None:
                if rec.state.get("env") != f_env:
                    continue
            if f_phase is not None:
                if rec.state.get("phase") != f_phase:
                    continue

            sim = self.state_similarity(query_state, rec.state)
            if sim < float(min_similarity):
                continue

            oq = self._outcome_quality(rec.outcome)
            rc = self._recency_score(rec.ts)
            rs = self._rank_record(sim, oq, rc)

            # copy record with query-time scores (we keep object but set fields)
            rec.similarity = sim
            rec.outcome_quality = oq
            rec.recency = rc
            rec.rank_score = rs

            candidates.append(rec)

        candidates.sort(key=lambda r: r.rank_score, reverse=True)
        return candidates[: max(0, int(k))]

    # ----------------------------
    # Recommendations
    # ----------------------------

    @staticmethod
    def action_signature(action: Dict[str, Any]) -> str:
        """
        Deterministic action signature for aggregation:
        <strategy>|<skill>|<sorted_parameters>
        """
        strategy = str(action.get("strategy", "")).strip()
        skill = str(action.get("skill", "")).strip()
        params = action.get("parameters") if isinstance(action.get("parameters"), dict) else {}
        params_items = sorted((str(k), str(v)) for k, v in params.items())
        params_str = ",".join([f"{k}={v}" for k, v in params_items])
        return f"{strategy}|{skill}|{params_str}"

    def recommend(
        self,
        query_state: Dict[str, Any],
        *,
        k_actions: int = 5,
        k_records: int = 25,
        min_similarity: float = 0.0,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Return planner-facing recommendations, plus supporting records for explainability.

        Output:
        {
          "recommendations": [
            {
              "action_signature": str,
              "action_prototype": dict,
              "trials": int,
              "success_rate": float,
              "avg_quality": float,
              "common_failures": [{"failure_type": str, "count": int}, ...],
              "last_seen_ts": float,
              "last_success_ts": Optional[float],
              "action_score": float,
            }, ...
          ],
          "supporting_records": [ExperienceRecord, ...]
        }
        """
        records = self.query(
            query_state,
            k=k_records,
            min_similarity=min_similarity,
            filters=filters
        )

        grouped: Dict[str, List[ExperienceRecord]] = defaultdict(list)
        action_proto: Dict[str, Dict[str, Any]] = {}

        for rec in records:
            sig = self.action_signature(rec.action)
            grouped[sig].append(rec)
            # keep a prototype action (first seen)
            if sig not in action_proto:
                action_proto[sig] = rec.action

        recs_out: List[Dict[str, Any]] = []

        for sig, recs in grouped.items():
            trials = len(recs)
            successes = sum(1 for r in recs if bool(r.outcome.get("success", False)))
            success_rate = successes / trials if trials else 0.0
            avg_quality = sum(r.outcome_quality for r in recs) / trials if trials else 0.0

            # failures
            ft = [str(r.outcome.get("failure_type", "")).strip() for r in recs if not bool(r.outcome.get("success", False))]
            ft = [x for x in ft if x]
            common = [{"failure_type": k, "count": v} for k, v in Counter(ft).most_common(3)]

            last_seen_ts = max(r.ts for r in recs) if recs else 0.0
            last_success_ts = max((r.ts for r in recs if bool(r.outcome.get("success", False))), default=None)

            # recency_of_last_success: 0 if never succeeded
            if last_success_ts is None:
                recency_last_success = 0.0
            else:
                recency_last_success = self._recency_score(last_success_ts)

            w = self.action_score_weights
            action_score = (
                w["success_rate"] * success_rate +
                w["avg_quality"] * avg_quality +
                w["recency_last_success"] * recency_last_success
            )

            recs_out.append({
                "action_signature": sig,
                "action_prototype": action_proto.get(sig, {}),
                "trials": trials,
                "success_rate": float(success_rate),
                "avg_quality": float(avg_quality),
                "common_failures": common,
                "last_seen_ts": float(last_seen_ts),
                "last_success_ts": float(last_success_ts) if last_success_ts is not None else None,
                "action_score": float(action_score),
            })

        recs_out.sort(key=lambda x: x["action_score"], reverse=True)

        return {
            "recommendations": recs_out[: max(0, int(k_actions))],
            "supporting_records": records,  # already ranked
        }

    # ----------------------------
    # Persistence
    # ----------------------------

    def export_json(self) -> str:
        """
        Export all records to JSON.
        """
        payload = []
        for r in self._records:
            payload.append({
                "record_id": r.record_id,
                "state": r.state,
                "action": r.action,
                "outcome": r.outcome,
                "ts": r.ts,
                "episode_id": r.episode_id,
                "salience": r.salience,
                "source": r.source,
            })
        meta = {
            "schema": "experience_memory_v1",
            "created_at": now_ts(),
            "half_life_days": self.half_life_days,
            "sim_weights": self.sim_weights,
            "rank_weights": self.rank_weights,
            "action_score_weights": self.action_score_weights,
            "records": payload,
        }
        return json.dumps(meta, indent=2)

    @staticmethod
    def import_json(s: str) -> "ExperienceMemory":
        """
        Import records from JSON created by export_json().
        """
        raw = json.loads(s)
        mem = ExperienceMemory()

        mem.half_life_days = float(raw.get("half_life_days", mem.half_life_days))
        mem.sim_weights.update(raw.get("sim_weights", {}))
        mem.rank_weights.update(raw.get("rank_weights", {}))
        mem.action_score_weights.update(raw.get("action_score_weights", {}))

        for d in raw.get("records", []):
            rec = ExperienceRecord(
                record_id=d["record_id"],
                state=d["state"],
                action=d["action"],
                outcome=d["outcome"],
                ts=float(d.get("ts", now_ts())),
                episode_id=d.get("episode_id"),
                salience=float(d.get("salience", 0.5)),
                source=str(d.get("source", "planner")),
            )
            mem._records.append(rec)

        return mem


# ----------------------------
# Demo (Colab runnable)
# ----------------------------

def _demo():
    mem = ExperienceMemory()

    # Create a small dataset of experiences for one task
    base_task = "summarize_document"

    def add(state_overrides, action, outcome, days_back=0):
        state = {
            "task": base_task,
            "env": "web_chat",
            "phase": "response",
            "constraints": ["concise"],
            "signals": {"domain": "technical", "length": "long"},
            "tags": ["nlp"],
        }
        state.update(state_overrides or {})
        rid = mem.add_experience(state, action, outcome, salience=0.6, source="planner")
        # manually adjust timestamp for demo
        mem._records[-1].ts = now_ts() - days_back * 86400.0
        return rid

    # Actions
    A = {"strategy": "hierarchical_summary", "skill": "summarizer_v2", "parameters": {"verbosity": "low"}}
    B = {"strategy": "bullet_summary", "skill": "summarizer_v1", "parameters": {"verbosity": "low"}}
    C = {"strategy": "full_rewrite", "skill": "rewriter_v1", "parameters": {"verbosity": "high"}}

    # Experiences: some successes, some failures
    add({}, A, {"success": True, "score": 0.85}, days_back=2)
    add({}, A, {"success": True, "score": 0.82}, days_back=10)
    add({}, B, {"success": True, "score": 0.78}, days_back=4)
    add({}, B, {"success": False, "failure_type": "missed_points", "score": 0.35}, days_back=1)
    add({"constraints": ["concise", "time_limited"]}, A, {"success": True, "score": 0.80}, days_back=3)
    add({"signals": {"domain": "legal", "length": "long"}}, A, {"success": False, "failure_type": "hallucination", "score": 0.20}, days_back=5)
    add({"signals": {"domain": "technical", "length": "short"}}, B, {"success": True, "score": 0.81}, days_back=6)
    add({"signals": {"domain": "technical", "length": "long"}}, C, {"success": False, "failure_type": "too_long", "score": 0.10}, days_back=2)

    # Query state
    query_state = {
        "task": "summarize_document",
        "env": "web_chat",
        "phase": "response",
        "constraints": ["concise", "time_limited"],
        "signals": {"domain": "technical", "length": "long"},
        "tags": ["nlp"],
    }

    print("\n=== Query: top records ===")
    top_records = mem.query(query_state, k=5, min_similarity=0.10)
    for r in top_records:
        sig = mem.action_signature(r.action)
        print(f"- id={r.record_id[:8]} sig={sig}")
        print(f"  sim={r.similarity:.3f} oq={r.outcome_quality:.3f} rec={r.recency:.3f} rank={r.rank_score:.3f} success={r.outcome.get('success')}")
        if not r.outcome.get("success", False):
            print(f"  failure_type={r.outcome.get('failure_type')} score={r.outcome.get('score')}")

    print("\n=== Recommendations (planner-facing) ===")
    recs = mem.recommend(query_state, k_actions=3, k_records=25, min_similarity=0.10)
    for i, a in enumerate(recs["recommendations"], start=1):
        print(f"{i}. {a['action_signature']}")
        print(f"   trials={a['trials']} success_rate={a['success_rate']:.2f} avg_quality={a['avg_quality']:.2f} action_score={a['action_score']:.3f}")
        if a["common_failures"]:
            print(f"   common_failures={a['common_failures']}")

    # Test persistence
    saved = mem.export_json()
    mem2 = ExperienceMemory.import_json(saved)
    recs2 = mem2.recommend(query_state, k_actions=3, k_records=25, min_similarity=0.10)

    assert [x["action_signature"] for x in recs2["recommendations"]] == [x["action_signature"] for x in recs["recommendations"]], \
        "Persistence roundtrip changed recommendation ordering"

    print("\n✅ Demo complete. ExperienceMemory is runnable + persistence roundtrip OK.")


if __name__ == "__main__":
    _demo()
