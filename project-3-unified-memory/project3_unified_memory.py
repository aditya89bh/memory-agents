"""
Project 3 — Unified Memory Stack (Single-file version)

Copy-paste into:
  project-3-unified-memory/project3_unified_memory.py

Run:
  pip install scikit-learn
  python project3_unified_memory.py

What it demonstrates:
- Unified MemoryManager orchestrating read -> context -> write
- Retrieval-before-reasoning enforced
- Short-term + Summary + Long-term integrated
- Gated memory writes
- Pinned identity/preferences/goals
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
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
    mtype: str  # identity | preference | goal | fact | event | note
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
class Turn:
    user: str
    assistant: str
    outcome: Optional[Dict[str, Any]] = None


@dataclass
class MemoryReadResult:
    pinned: List[MemoryItem]
    recalls: List[MemoryItem]


@dataclass
class MemoryWriteResult:
    stored: List[MemoryItem]
    discarded: List[MemoryItem]


# ----------------------------
# Short-Term Memory
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
# Summary Memory (deterministic)
# ----------------------------

class SummaryMemory:
    """
    Deterministic summary to keep Project 3 transparent.
    Later: swap update() with an LLM summarizer without changing interface.
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
    Minimal LTM backend for Project 3 integration proof.
    Stores MemoryItems, supports pinned retrieval + TF-IDF recall + basic filters.
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
    Adapter wrapper (so you can swap this backend with your Project 2 store).
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
        token_budget_chars: int = 8000,
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
                lines.append(f"- U: {t.user}")
                lines.append(f"  A: {t.assistant}")
            return "\n".join(lines)

        parts = [
            "=== MEMORY CONTEXT ===",
            fmt_items("Pinned (identity/preferences/goals)", pinned, limit=10),
            fmt_items("Relevant long-term recalls", recalls, limit=8),
            "Summary:\n" + (summary if summary else "- (none)"),
            fmt_recent(recent, limit=6),
            "=== CURRENT USER MESSAGE ===\n" + user_msg,
        ]
        ctx = "\n\n".join(parts)
        return ctx[-token_budget_chars:]


# ----------------------------
# Gating Function (transparent)
# ----------------------------

def simple_gate(turn: Turn) -> List[MemoryItem]:
    """
    Transparent heuristic gate for Project 3.
    Later: replace with your salience scorer from Project 2C.
    """
    u = turn.user.strip()
    ul = u.lower()

    items: List[MemoryItem] = []

    if "my name is" in ul or ul.startswith("i am "):
        items.append(MemoryItem.new(u, "identity", tags=["self_report"]))

    if "i prefer" in ul or "i like" in ul:
        items.append(MemoryItem.new(u, "preference", tags=["self_report"]))

    if "my goal" in ul or "i want to" in ul:
        items.append(MemoryItem.new(u, "goal", tags=["self_report"]))

    if any(w in ul for w in ["deadline", "shipped", "launched", "failed", "fixed", "bug"]):
        items.append(MemoryItem.new(u, "event", tags=["high_signal"]))

    return items


# ----------------------------
# MemoryManager (the orchestrator)
# ----------------------------

class MemoryManager:
    """
    Enforces read -> assemble context -> write (gated).
    """
    def __init__(
        self,
        stm: ShortTermMemory,
        summary_mem: SummaryMemory,
        ltm_backend,
        assembler: ContextAssembler,
        gate_fn: Callable[[Turn], List[MemoryItem]],
    ):
        self.stm = stm
        self.summary = summary_mem
        self.ltm = LongTermMemory(ltm_backend)
        self.assembler = assembler
        self.gate_fn = gate_fn

    # Phase 1: Read (retrieve before reasoning)
    def read(self, query: str, k: int = 6, filters: Optional[Dict] = None) -> MemoryReadResult:
        pinned = self.ltm.get_pinned(mtypes=["identity", "preference", "goal"])
        recalls = self.ltm.search(query=query, k=k, filters=filters)
        return MemoryReadResult(pinned=pinned, recalls=recalls)

    # Phase 2: Context assembly
    def build_context(self, user_msg: str, read_result: MemoryReadResult) -> str:
        return self.assembler.build(
            user_msg=user_msg,
            pinned=read_result.pinned,
            recalls=read_result.recalls,
            recent=self.stm.get_recent(),
            summary=self.summary.get(),
        )

    # Phase 3: Write (gated)
    def write(self, turn: Turn) -> MemoryWriteResult:
        self.stm.add_turn(turn)
        self.summary.update(self.stm.get_recent())

        candidates = self.gate_fn(turn)
        stored: List[MemoryItem] = []
        discarded: List[MemoryItem] = []

        for item in candidates:
            if item.mtype in ["identity", "preference", "goal"]:
                item.pinned = True
            self.ltm.add(item)
            stored.append(item)

        return MemoryWriteResult(stored=stored, discarded=discarded)


# ----------------------------
# Toy LLM + Agent Loop
# ----------------------------

def toy_llm(prompt: str) -> str:
    """
    Fake model used to prove retrieval-before-reasoning.
    Reads the memory context and emits a response that confirms it saw it.
    """
    lower = prompt.lower()
    signals = []

    if "=== memory context ===" in lower:
        signals.append("Checked memory before replying.")
    if "pinned (identity/preferences/goals)" in lower:
        signals.append("I see pinned identity/preferences/goals in context.")

    return " ".join(signals) if signals else "OK."


class Agent:
    def __init__(self, mm: MemoryManager):
        self.mm = mm

    def step(self, user_msg: str, debug: bool = False) -> str:
        read_res = self.mm.read(query=user_msg, k=6)
        context = self.mm.build_context(user_msg, read_res)
        assistant_msg = toy_llm(context)

        if debug:
            print(context[:2500])
            print("\n---\n")

        self.mm.write(Turn(user=user_msg, assistant=assistant_msg))
        return assistant_msg


# ----------------------------
# Demo
# ----------------------------

def demo():
    stm = ShortTermMemory(max_turns=8)
    summ = SummaryMemory(max_chars=2000)
    ltm = SimpleTfidfLTM()
    assembler = ContextAssembler()

    mm = MemoryManager(
        stm=stm,
        summary_mem=summ,
        ltm_backend=ltm,
        assembler=assembler,
        gate_fn=simple_gate,
    )

    agent = Agent(mm)

    print("\n--- PROJECT 3 DEMO START ---\n")

    print("USER:", "My name is Aditya.")
    print("ASSISTANT:", agent.step("My name is Aditya.", debug=True))

    print("USER:", "I prefer short answers.")
    print("ASSISTANT:", agent.step("I prefer short answers.", debug=True))

    print("USER:", "What do you remember about me?")
    print("ASSISTANT:", agent.step("What do you remember about me?", debug=True))

    print("\n--- PROJECT 3 DEMO END ---\n")


if __name__ == "__main__":
    demo()
