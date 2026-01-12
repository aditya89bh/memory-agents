"""
Project 4 — Memory + Planning (Single-file version)

Copy-paste into:
  project-4-memory-planning/project4_memory_planning.py

Run:
  pip install scikit-learn
  python project4_memory_planning.py

What it demonstrates:
- Unified memory stack (STM + Summary + LTM)
- Planner consults EXPERIENCE memory before acting
- Act -> Evaluate -> Write closes the learning loop
- Same input can lead to different behavior over time because of memory
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import deque

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------
# Data Types
# ----------------------------

@dataclass
class MemoryItem:
    id: str
    text: str
    mtype: str  # identity | preference | goal | fact | event | experience | note
    tags: List[str] = field(default_factory=list)
    source: str = "chat"
    created_at: float = field(default_factory=lambda: time.time())
    score: float = 0.0
    pinned: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def new(text: str, mtype: str, **kwargs) -> "MemoryItem":
        return MemoryItem(id=str(uuid.uuid4()), text=text, mtype=mtype, **kwargs)


@dataclass
class Plan:
    strategy: str  # direct_answer | ask_clarify
    rationale: str = ""
    slots_needed: List[str] = field(default_factory=list)


@dataclass
class Outcome:
    status: str  # success | failure
    reason: str = ""
    score: float = 0.0


@dataclass
class Turn:
    user: str
    assistant: str
    plan: Optional[Dict[str, Any]] = None
    outcome: Optional[Dict[str, Any]] = None


@dataclass
class MemoryReadResult:
    pinned: List[MemoryItem]
    recalls: List[MemoryItem]


# ----------------------------
# Short-Term Memory (Working Memory)
# ----------------------------

class ShortTermMemory:
    def __init__(self, max_turns: int = 12):
        self.buffer = deque(maxlen=max_turns)

    def add_turn(self, turn: Turn) -> None:
        self.buffer.append(turn)

    def get_recent(self) -> List[Turn]:
        return list(self.buffer)

    def forget_all(self) -> None:
        self.buffer.clear()


# ----------------------------
# Summary Memory (Deterministic)
# ----------------------------

class SummaryMemory:
    """
    Deterministic summary to keep Project 4 inspectable.
    Swap update() with an LLM later if desired.
    """
    def __init__(self, max_chars: int = 2000):
        self.summary = ""
        self.max_chars = max_chars

    def update(self, recent_turns: List[Turn]) -> None:
        if not recent_turns:
            return
        lines = []
        for t in recent_turns[-6:]:
            lines.append(f"- U: {t.user}")
            lines.append(f"  A: {t.assistant}")
        self.summary = "\n".join(lines)[-self.max_chars:]

    def get(self) -> str:
        return self.summary


# ----------------------------
# Long-Term Memory Backend (TF-IDF)
# ----------------------------

class SimpleTfidfLTM:
    """
    Minimal LTM backend:
    - stores MemoryItems in-memory
    - TF-IDF vectorization
    - cosine similarity search
    - supports pinned + basic filters
    """
    def __init__(self):
        self.items: List[MemoryItem] = []
        self.vectorizer = TfidfVectorizer()
        self._tfidf_matrix = None

    def _rebuild_index(self) -> None:
        corpus = [it.text for it in self.items] if self.items else [""]
        self._tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def add(self, item: MemoryItem) -> None:
        self.items.append(item)
        self._rebuild_index()

    def get_pinned(self, mtypes: Optional[List[str]] = None) -> List[MemoryItem]:
        out = [it for it in self.items if it.pinned]
        if mtypes:
            out = [it for it in out if it.mtype in mtypes]
        return out

    def search(self, query: str, k: int = 6, filters: Optional[Dict] = None) -> List[MemoryItem]:
        if not self.items:
            return []

        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self._tfidf_matrix).flatten()

        results: List[MemoryItem] = []
        for it, s in zip(self.items, sims):
            if filters:
                if "mtype" in filters and it.mtype != filters["mtype"]:
                    continue
                if "tags_any" in filters and not any(t in it.tags for t in filters["tags_any"]):
                    continue

            clone = MemoryItem(**{**it.__dict__})
            clone.score = float(s)
            results.append(clone)

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]


class LongTermMemory:
    """
    Adapter wrapper: lets you swap in your Project 2 backend later.
    """
    def __init__(self, backend):
        self.backend = backend

    def add(self, item: MemoryItem) -> None:
        self.backend.add(item)

    def get_pinned(self, mtypes: Optional[List[str]] = None) -> List[MemoryItem]:
        return self.backend.get_pinned(mtypes=mtypes)

    def search(self, query: str, k: int = 6, filters: Optional[Dict] = None) -> List[MemoryItem]:
        return self.backend.search(query=query, k=k, filters=filters)


# ----------------------------
# Context Assembler
# ----------------------------

class ContextAssembler:
    def build(
        self,
        user_msg: str,
        pinned: List[MemoryItem],
        recalls: List[MemoryItem],
        recent: List[Turn],
        summary: str,
        plan: Optional[Plan] = None,
        token_budget_chars: int = 9000,
    ) -> str:

        def fmt_items(title: str, items: List[MemoryItem], limit: int = 8) -> str:
            if not items:
                return f"{title}:\n- (none)"
            lines = [f"{title}:"]
            for it in items[:limit]:
                pin = " (pinned)" if it.pinned else ""
                score = f" score={it.score:.3f}" if it.score else ""
                tags = f" tags={it.tags}" if it.tags else ""
                lines.append(f"- [{it.mtype}]{pin}{score}: {it.text}{tags}")
            return "\n".join(lines)

        def fmt_recent(turns: List[Turn], limit: int = 6) -> str:
            if not turns:
                return "Recent:\n- (none)"
            lines = ["Recent:"]
            for t in turns[-limit:]:
                p = f" plan={t.plan}" if t.plan else ""
                o = f" outcome={t.outcome}" if t.outcome else ""
                lines.append(f"- U: {t.user}")
                lines.append(f"  A: {t.assistant}{p}{o}")
            return "\n".join(lines)

        plan_block = "Plan:\n- (none)"
        if plan:
            plan_block = (
                "Plan:\n"
                f"- strategy: {plan.strategy}\n"
                f"- rationale: {plan.rationale}\n"
                f"- slots_needed: {plan.slots_needed}"
            )

        parts = [
            "=== MEMORY CONTEXT ===",
            fmt_items("Pinned (identity/preferences/goals)", pinned, limit=10),
            fmt_items("Relevant long-term recalls", recalls, limit=8),
            "Summary:\n" + (summary if summary else "- (none)"),
            plan_block,
            fmt_recent(recent, limit=6),
            "=== CURRENT USER MESSAGE ===\n" + user_msg,
        ]

        ctx = "\n\n".join(parts)
        return ctx[-token_budget_chars:]


# ----------------------------
# Intent + Slot Detector (Demo)
# ----------------------------

def detect_intent_and_slots(user_msg: str) -> Dict[str, Any]:
    """
    Minimal demo intent/slot detector.
    Task: booking a flight requires from/to/date.
    """
    text = user_msg.lower()
    if "book" in text and "flight" in text:
        slots = {"from": None, "to": None, "date": None}
        for token in ["from", "to", "date"]:
            if f"{token} " in text:
                slots[token] = "present"
        missing = [k for k, v in slots.items() if v is None]
        return {"intent": "book_flight", "missing_slots": missing}
    return {"intent": "general", "missing_slots": []}


# ----------------------------
# Planner (Memory-Aware)
# ----------------------------

class Planner:
    def __init__(self, ltm: LongTermMemory):
        self.ltm = ltm

    def plan(self, user_msg: str) -> Plan:
        info = detect_intent_and_slots(user_msg)
        intent = info["intent"]
        missing = info["missing_slots"]

        # Consult experience memory
        experiences = self.ltm.search(query=user_msg, k=6, filters={"mtype": "experience"})

        # If past similar attempt failed with direct_answer -> ask_clarify
        for ex in experiences:
            if ex.metadata.get("strategy") == "direct_answer" and ex.metadata.get("status") == "failure":
                return Plan(
                    strategy="ask_clarify",
                    rationale="Past similar attempt failed with direct_answer. Asking clarification.",
                    slots_needed=missing,
                )

        # Slot-based rule: if missing required info -> ask_clarify
        if intent == "book_flight" and missing:
            return Plan(
                strategy="ask_clarify",
                rationale="Slot-based task with missing required info.",
                slots_needed=missing,
            )

        return Plan(strategy="direct_answer", rationale="No missing slots and no prior failures found.")


# ----------------------------
# Outcome Evaluator
# ----------------------------

class OutcomeEvaluator:
    def evaluate(self, user_msg: str, assistant_msg: str, plan: Plan) -> Outcome:
        info = detect_intent_and_slots(user_msg)

        # If it's a flight task with missing slots:
        if info["intent"] == "book_flight" and info["missing_slots"]:
            if plan.strategy == "direct_answer":
                return Outcome(status="failure", reason="Tried direct_answer with missing slots.", score=0.0)
            if plan.strategy == "ask_clarify":
                return Outcome(status="success", reason="Asked for missing slots.", score=1.0)

        return Outcome(status="success", reason="Default success.", score=1.0)


# ----------------------------
# Memory Gating (Stores Experience)
# ----------------------------

def gated_write(turn: Turn) -> List[MemoryItem]:
    """
    Project 4 gating:
    - identity/preferences/goals (pinned)
    - experience memories from plan+outcome (non-pinned)
    """
    items: List[MemoryItem] = []
    u = turn.user.strip()
    ul = u.lower()

    if "my name is" in ul or ul.startswith("i am "):
        items.append(MemoryItem.new(u, "identity", tags=["self_report"], pinned=True))
    if "i prefer" in ul or "i like" in ul:
        items.append(MemoryItem.new(u, "preference", tags=["self_report"], pinned=True))
    if "my goal" in ul or "i want to" in ul:
        items.append(MemoryItem.new(u, "goal", tags=["self_report"], pinned=True))

    if turn.plan and turn.outcome:
        info = detect_intent_and_slots(turn.user)
        ex_text = (
            f"Task={info['intent']}. Strategy={turn.plan.get('strategy')}. "
            f"Outcome={turn.outcome.get('status')}. Reason={turn.outcome.get('reason')}."
        )
        items.append(
            MemoryItem.new(
                ex_text,
                "experience",
                tags=["experience", info["intent"]],
                metadata={
                    "intent": info["intent"],
                    "strategy": turn.plan.get("strategy"),
                    "status": turn.outcome.get("status"),
                    "reason": turn.outcome.get("reason"),
                },
            )
        )

    return items


# ----------------------------
# Memory Manager (Read -> Context -> Write)
# ----------------------------

class MemoryManager:
    def __init__(self, stm, summary_mem, ltm_backend, assembler, gate_fn):
        self.stm = stm
        self.summary = summary_mem
        self.ltm = LongTermMemory(ltm_backend)
        self.assembler = assembler
        self.gate_fn = gate_fn

    def read(self, query: str, k: int = 6, filters: Optional[Dict] = None) -> MemoryReadResult:
        pinned = self.ltm.get_pinned(mtypes=["identity", "preference", "goal"])
        recalls = self.ltm.search(query=query, k=k, filters=filters)
        return MemoryReadResult(pinned=pinned, recalls=recalls)

    def build_context(self, user_msg: str, read_result: MemoryReadResult, plan: Optional[Plan]) -> str:
        return self.assembler.build(
            user_msg=user_msg,
            pinned=read_result.pinned,
            recalls=read_result.recalls,
            recent=self.stm.get_recent(),
            summary=self.summary.get(),
            plan=plan,
        )

    def write(self, turn: Turn) -> None:
        self.stm.add_turn(turn)
        self.summary.update(self.stm.get_recent())

        candidates = self.gate_fn(turn)
        for item in candidates:
            if item.mtype in ["identity", "preference", "goal"]:
                item.pinned = True
            self.ltm.add(item)


# ----------------------------
# Actor (Executes Plan)
# ----------------------------

def toy_actor(user_msg: str, plan: Plan) -> str:
    if plan.strategy == "ask_clarify":
        missing = plan.slots_needed
        if missing:
            return f"I can help. Quick clarifications needed: {', '.join(missing)}?"
        return "I need a bit more info. What should I clarify?"
    return "Sure. I can proceed with that. (Demo direct answer.)"


# ----------------------------
# Agent Loop (Read -> Plan -> Act -> Evaluate -> Write)
# ----------------------------

class Agent:
    def __init__(self, mm: MemoryManager, planner: Planner, evaluator: OutcomeEvaluator):
        self.mm = mm
        self.planner = planner
        self.evaluator = evaluator

    def step(self, user_msg: str, debug: bool = False) -> str:
        read_res = self.mm.read(query=user_msg, k=6)
        plan = self.planner.plan(user_msg)
        context = self.mm.build_context(user_msg, read_res, plan)

        assistant_msg = toy_actor(user_msg, plan)
        outcome = self.evaluator.evaluate(user_msg, assistant_msg, plan)

        turn = Turn(
            user=user_msg,
            assistant=assistant_msg,
            plan={"strategy": plan.strategy, "rationale": plan.rationale, "slots_needed": plan.slots_needed},
            outcome={"status": outcome.status, "reason": outcome.reason, "score": outcome.score},
        )
        self.mm.write(turn)

        if debug:
            print(context[:2800])
            print("\n---\n")
            print("PLAN:", plan)
            print("OUTCOME:", outcome)
            print("\n=========================\n")

        return assistant_msg


# ----------------------------
# Demo
# ----------------------------

def demo():
    stm = ShortTermMemory(max_turns=8)
    summ = SummaryMemory(max_chars=2000)
    ltm_backend = SimpleTfidfLTM()
    assembler = ContextAssembler()

    mm = MemoryManager(stm, summ, ltm_backend, assembler, gated_write)
    planner = Planner(mm.ltm)
    evaluator = OutcomeEvaluator()
    agent = Agent(mm, planner, evaluator)

    print("\n--- PROJECT 4 DEMO START ---\n")

    msg = "Can you book a flight for me?"
    print("USER:", msg)
    print("ASSISTANT:", agent.step(msg, debug=True))

    print("USER:", msg)
    print("ASSISTANT:", agent.step(msg, debug=True))

    print("USER:", "Book a flight from Athens to Berlin date tomorrow")
    print("ASSISTANT:", agent.step("Book a flight from Athens to Berlin date tomorrow", debug=True))

    print("\nStored EXPERIENCE memories:")
    exps = mm.ltm.search(msg, k=10, filters={"mtype": "experience"})
    for e in exps:
        print("-", e.text, "|", e.metadata)

    print("\n--- PROJECT 4 DEMO END ---\n")


if __name__ == "__main__":
    demo()
