"""


Install:
  pip install scikit-learn

Run:
  python project5_skill_task_memory.py

What this demonstrates:
- Task detection (minimal)
- Experience recording (success/failure)
- Skill promotion from repeated successful experience
- Skill matching BEFORE planning
- use_skill plan path
- Skill evidence updates + confidence scoring
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import deque

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# Utilities
# ============================================================

def new_id() -> str:
    return str(uuid.uuid4())


# ============================================================
# Core Data Models (Project 5 Step 1)
# ============================================================

@dataclass
class Task:
    """
    A normalized representation of "what the user is trying to do".
    """
    name: str
    slots_required: List[str] = field(default_factory=list)
    slots_filled: Dict[str, Any] = field(default_factory=dict)
    missing_slots: List[str] = field(default_factory=list)

    def is_complete(self) -> bool:
        return len(self.missing_slots) == 0


@dataclass
class Plan:
    """
    Strategy decision.
    strategy: direct_answer | ask_clarify | use_skill
    """
    strategy: str
    rationale: str = ""
    slots_needed: List[str] = field(default_factory=list)
    skill_id: Optional[str] = None


@dataclass
class Outcome:
    """
    Evaluation signal.
    """
    status: str  # success | failure
    reason: str = ""
    score: float = 0.0


@dataclass
class Skill:
    """
    Reusable procedure derived from repeated successful experience.
    """
    id: str
    name: str
    task_name: str
    condition: str            # always | missing_slots  (simple for now)
    strategy: str             # ask_clarify | direct_answer
    created_at: float = field(default_factory=lambda: time.time())

    success_count: int = 0
    failure_count: int = 0
    last_used_at: float = 0.0

    tags: List[str] = field(default_factory=list)
    notes: str = ""

    def total(self) -> int:
        return self.success_count + self.failure_count

    def confidence(self) -> float:
        # simple smoothing
        return (self.success_count + 1) / (self.total() + 2)


@dataclass
class ExperienceTrace:
    """
    Normalized record used for promotion and analysis.
    """
    id: str
    task_name: str
    strategy: str
    outcome_status: str      # success | failure
    reason: str = ""
    created_at: float = field(default_factory=lambda: time.time())
    meta: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Skill Promotion (Project 5 Step 1)
# ============================================================

@dataclass
class SkillPromotionConfig:
    min_successes: int = 2
    min_total: int = 3
    min_success_rate: float = 0.70
    cooldown_seconds: int = 0


def should_promote_skill(
    traces: List[ExperienceTrace],
    task_name: str,
    strategy: str,
    cfg: SkillPromotionConfig
) -> bool:
    relevant = [t for t in traces if t.task_name == task_name and t.strategy == strategy]
    if len(relevant) < cfg.min_total:
        return False

    successes = sum(1 for t in relevant if t.outcome_status == "success")
    failures = sum(1 for t in relevant if t.outcome_status == "failure")
    total = successes + failures

    if successes < cfg.min_successes:
        return False

    success_rate = successes / total if total else 0.0
    return success_rate >= cfg.min_success_rate


class SkillRegistry:
    """
    Procedural memory store (in-memory).
    """
    def __init__(self):
        self.skills: Dict[str, Skill] = {}

    def add_skill(self, skill: Skill) -> None:
        self.skills[skill.id] = skill

    def list_skills(self) -> List[Skill]:
        return list(self.skills.values())

    def find_by_task(self, task_name: str) -> List[Skill]:
        return [s for s in self.skills.values() if s.task_name == task_name]

    def update_evidence(self, skill_id: str, outcome_status: str) -> None:
        s = self.skills.get(skill_id)
        if not s:
            return
        if outcome_status == "success":
            s.success_count += 1
        elif outcome_status == "failure":
            s.failure_count += 1
        s.last_used_at = time.time()


def promote_skills_from_traces(
    traces: List[ExperienceTrace],
    registry: SkillRegistry,
    cfg: SkillPromotionConfig
) -> List[Skill]:
    """
    Promote eligible (task, strategy) patterns into skills.
    """
    new_skills: List[Skill] = []
    pairs = set((t.task_name, t.strategy) for t in traces)

    for task_name, strategy in pairs:
        exists = any(s.task_name == task_name and s.strategy == strategy for s in registry.list_skills())
        if exists:
            continue

        if should_promote_skill(traces, task_name, strategy, cfg):
            relevant = [t for t in traces if t.task_name == task_name and t.strategy == strategy]
            skill = Skill(
                id=new_id(),
                name=f"{task_name}:{strategy}",
                task_name=task_name,
                condition="always",
                strategy=strategy,
                tags=["auto_promoted"],
                notes="Auto-promoted from repeated successful experience."
            )
            skill.success_count = sum(1 for t in relevant if t.outcome_status == "success")
            skill.failure_count = sum(1 for t in relevant if t.outcome_status == "failure")

            registry.add_skill(skill)
            new_skills.append(skill)

    return new_skills


# ============================================================
# Memory (minimal LTM + experience writer)
# ============================================================

@dataclass
class MemoryItem:
    id: str
    text: str
    mtype: str
    tags: List[str] = field(default_factory=list)
    source: str = "chat"
    created_at: float = field(default_factory=lambda: time.time())
    score: float = 0.0
    pinned: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def new(text: str, mtype: str, **kwargs) -> "MemoryItem":
        return MemoryItem(id=new_id(), text=text, mtype=mtype, **kwargs)


@dataclass
class MemoryReadResult:
    pinned: List["MemoryItem"]
    recalls: List["MemoryItem"]


class SimpleTfidfLTM:
    def __init__(self):
        self.items: List["MemoryItem"] = []
        self.vectorizer = TfidfVectorizer()
        self._tfidf_matrix = None

    def _rebuild_index(self) -> None:
        corpus = [it.text for it in self.items] if self.items else [""]
        self._tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def add(self, item: "MemoryItem") -> None:
        self.items.append(item)
        self._rebuild_index()

    def get_pinned(self, mtypes: Optional[List[str]] = None) -> List["MemoryItem"]:
        out = [it for it in self.items if it.pinned]
        if mtypes:
            out = [it for it in out if it.mtype in mtypes]
        return out

    def search(self, query: str, k: int = 6, filters: Optional[Dict] = None) -> List["MemoryItem"]:
        if not self.items:
            return []

        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self._tfidf_matrix).flatten()

        results: List["MemoryItem"] = []
        for it, s in zip(self.items, sims):
            if filters:
                if "mtype" in filters and it.mtype != filters["mtype"]:
                    continue
            clone = MemoryItem(**{**it.__dict__})
            clone.score = float(s)
            results.append(clone)

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]


class LongTermMemory:
    def __init__(self, backend):
        self.backend = backend

    def add(self, item: "MemoryItem") -> None:
        self.backend.add(item)

    def get_pinned(self, mtypes: Optional[List[str]] = None) -> List["MemoryItem"]:
        return self.backend.get_pinned(mtypes=mtypes)

    def search(self, query: str, k: int = 6, filters: Optional[Dict] = None) -> List["MemoryItem"]:
        return self.backend.search(query=query, k=k, filters=filters)


class ShortTermMemory:
    def __init__(self, max_turns: int = 10):
        self.buffer = deque(maxlen=max_turns)

    def add_turn(self, turn: Dict[str, str]) -> None:
        self.buffer.append(turn)

    def get_recent(self) -> List[Dict[str, str]]:
        return list(self.buffer)


class SummaryMemory:
    def __init__(self, max_chars: int = 2000):
        self.summary = ""
        self.max_chars = max_chars

    def update(self, recent_turns: List[Dict[str, str]]) -> None:
        lines = []
        for t in recent_turns[-6:]:
            lines.append(f"- U: {t['user']}")
            lines.append(f"  A: {t['assistant']}")
        self.summary = "\n".join(lines)[-self.max_chars:]

    def get(self) -> str:
        return self.summary


class MemoryManager:
    """
    Minimal read/write used in Project 5 demo.
    Stores experience memories in LTM.
    """
    def __init__(self):
        self.stm = ShortTermMemory()
        self.summary = SummaryMemory()
        self.ltm = LongTermMemory(SimpleTfidfLTM())

    def read(self, query: str) -> MemoryReadResult:
        pinned = self.ltm.get_pinned(mtypes=["identity", "preference", "goal"])
        recalls = self.ltm.search(query=query, k=6)
        return MemoryReadResult(pinned=pinned, recalls=recalls)

    def write_experience(self, user_msg: str, plan: Plan, outcome: Outcome, task: Task) -> None:
        ex_text = (
            f"Task={task.name}. Strategy={plan.strategy}. Outcome={outcome.status}. "
            f"Reason={outcome.reason}. Skill={plan.skill_id}."
        )

        item = MemoryItem.new(
            ex_text,
            "experience",
            tags=["experience", task.name],
            metadata={
                "intent": task.name,
                "strategy": plan.strategy,
                "status": outcome.status,
                "reason": outcome.reason,
                "skill_id": plan.skill_id,
                "missing_slots": task.missing_slots,
            },
        )
        self.ltm.add(item)

        self.stm.add_turn({"user": user_msg, "assistant": ""})
        self.summary.update(self.stm.get_recent())


# ============================================================
# Task detection (minimal)
# ============================================================

def detect_task(user_msg: str) -> Task:
    text = user_msg.lower()

    if "book" in text and "flight" in text:
        required = ["from", "to", "date"]
        slots_filled = {}
        for k in required:
            if f"{k} " in text:
                slots_filled[k] = "present"
        missing = [k for k in required if k not in slots_filled]
        return Task(
            name="book_flight",
            slots_required=required,
            slots_filled=slots_filled,
            missing_slots=missing
        )

    return Task(name="general", slots_required=[], slots_filled={}, missing_slots=[])


# ============================================================
# Skill matching + planning (Project 5 Step 2)
# ============================================================

def skill_condition_holds(skill: Skill, task: Task) -> bool:
    if skill.condition == "always":
        return True
    if skill.condition == "missing_slots":
        return len(task.missing_slots) > 0
    return False


@dataclass
class SkillMatch:
    skill_id: str
    confidence: float
    reason: str


class SkillMatcher:
    def __init__(self, registry: SkillRegistry, min_confidence: float = 0.70):
        self.registry = registry
        self.min_confidence = min_confidence

    def match(self, task: Task) -> Optional[SkillMatch]:
        candidates = self.registry.find_by_task(task.name)
        if not candidates:
            return None

        usable = [s for s in candidates if skill_condition_holds(s, task)]
        if not usable:
            return None

        usable.sort(key=lambda s: s.confidence(), reverse=True)
        best = usable[0]
        conf = best.confidence()

        if conf < self.min_confidence:
            return None

        return SkillMatch(
            skill_id=best.id,
            confidence=conf,
            reason=f"Matched skill '{best.name}' with confidence={conf:.3f}"
        )


def execute_strategy(user_msg: str, strategy: str, slots_needed: List[str]) -> str:
    if strategy == "ask_clarify":
        if slots_needed:
            return f"I can help. Quick clarifications needed: {', '.join(slots_needed)}?"
        return "I need a bit more info. What should I clarify?"
    if strategy == "direct_answer":
        return "Sure. I can proceed with that. (Demo direct answer.)"
    return "OK."


def actor(user_msg: str, plan: Plan, registry: SkillRegistry) -> str:
    if plan.strategy == "use_skill":
        skill = registry.skills.get(plan.skill_id) if plan.skill_id else None
        if not skill:
            return execute_strategy(user_msg, "ask_clarify", plan.slots_needed)
        return execute_strategy(user_msg, skill.strategy, plan.slots_needed)

    return execute_strategy(user_msg, plan.strategy, plan.slots_needed)


class PlannerV2:
    """
    Skill-first planner:
    - if a skill matches, use_skill
    - else fallback to experience-aware planning (Project 4 style)
    """
    def __init__(self, ltm: LongTermMemory, matcher: SkillMatcher):
        self.ltm = ltm
        self.matcher = matcher

    def plan(self, user_msg: str) -> Plan:
        task = detect_task(user_msg)

        # 1) Skill-first
        m = self.matcher.match(task)
        if m:
            return Plan(
                strategy="use_skill",
                rationale=m.reason,
                slots_needed=task.missing_slots,
                skill_id=m.skill_id
            )

        # 2) Experience-aware fallback (avoid known bad move)
        experiences = self.ltm.search(query=user_msg, k=6, filters={"mtype": "experience"})
        for ex in experiences:
            if ex.metadata.get("strategy") == "direct_answer" and ex.metadata.get("status") == "failure":
                return Plan(
                    strategy="ask_clarify",
                    rationale="Past similar attempt failed with direct_answer. Asking clarification.",
                    slots_needed=task.missing_slots
                )

        # 3) Slot rule
        if task.name == "book_flight" and task.missing_slots:
            return Plan(
                strategy="ask_clarify",
                rationale="Slot-based task with missing required info.",
                slots_needed=task.missing_slots
            )

        return Plan(strategy="direct_answer", rationale="No skill match and no prior failures found.")


class OutcomeEvaluatorV2:
    def evaluate(self, user_msg: str, plan: Plan, registry: SkillRegistry) -> Outcome:
        task = detect_task(user_msg)

        # Effective strategy if plan uses skill
        effective = plan.strategy
        if plan.strategy == "use_skill" and plan.skill_id and plan.skill_id in registry.skills:
            effective = registry.skills[plan.skill_id].strategy

        if task.name == "book_flight" and task.missing_slots:
            if effective == "direct_answer":
                return Outcome(status="failure", reason="Tried direct_answer with missing slots.", score=0.0)
            if effective == "ask_clarify":
                return Outcome(status="success", reason="Asked for missing slots.", score=1.0)

        return Outcome(status="success", reason="Default success.", score=1.0)


def trace_from_turn(user_msg: str, plan: Plan, outcome: Outcome) -> ExperienceTrace:
    task = detect_task(user_msg)
    return ExperienceTrace(
        id=new_id(),
        task_name=task.name,
        strategy=plan.strategy,
        outcome_status=outcome.status,
        reason=outcome.reason,
        meta={"missing_slots": task.missing_slots, "skill_id": plan.skill_id}
    )


def maybe_update_skill_evidence(plan: Plan, outcome: Outcome, registry: SkillRegistry) -> None:
    if plan.strategy != "use_skill" or not plan.skill_id:
        return
    registry.update_evidence(plan.skill_id, outcome.status)


# ============================================================
# Project 5 Agent
# ============================================================

class Project5Agent:
    def __init__(
        self,
        mm: MemoryManager,
        registry: SkillRegistry,
        matcher: SkillMatcher,
        cfg: SkillPromotionConfig
    ):
        self.mm = mm
        self.registry = registry
        self.matcher = matcher
        self.planner = PlannerV2(self.mm.ltm, self.matcher)
        self.evaluator = OutcomeEvaluatorV2()
        self.cfg = cfg
        self.traces: List[ExperienceTrace] = []

    def step(self, user_msg: str, debug: bool = False) -> str:
        # READ
        _ = self.mm.read(user_msg)

        # TASK
        task = detect_task(user_msg)

        # PLAN (skill-first)
        plan = self.planner.plan(user_msg)

        # ACT
        assistant_msg = actor(user_msg, plan, self.registry)

        # EVALUATE
        outcome = self.evaluator.evaluate(user_msg, plan, self.registry)

        # WRITE experience memory
        self.mm.write_experience(user_msg, plan, outcome, task)

        # TRACE + evidence update
        self.traces.append(trace_from_turn(user_msg, plan, outcome))
        maybe_update_skill_evidence(plan, outcome, self.registry)

        # PROMOTE skills
        newly = promote_skills_from_traces(self.traces, self.registry, self.cfg)

        if debug:
            print("USER:", user_msg)
            print("TASK:", task)
            print("PLAN:", plan)
            print("ASSISTANT:", assistant_msg)
            print("OUTCOME:", outcome)
            if newly:
                print("NEW SKILLS:", [s.name for s in newly])
            print("SKILLS:", [(s.name, round(s.confidence(), 3), s.success_count, s.total()) for s in self.registry.list_skills()])
            print("----")

        return assistant_msg


# ============================================================
# Demo
# ============================================================

def demo():
    cfg = SkillPromotionConfig(min_successes=2, min_total=3, min_success_rate=0.66)

    mm = MemoryManager()
    registry = SkillRegistry()
    matcher = SkillMatcher(registry, min_confidence=0.70)

    agent = Project5Agent(mm, registry, matcher, cfg)

    msg = "Can you book a flight for me?"

    print("\n--- PROJECT 5 DEMO START ---\n")
    print("TURN 1")
    agent.step(msg, debug=True)

    print("TURN 2")
    agent.step(msg, debug=True)

    print("TURN 3")
    agent.step(msg, debug=True)

    print("TURN 4 (should use skill if promoted + confident)")
    agent.step(msg, debug=True)

    print("\nStored EXPERIENCE memories (top 5):")
    exps = mm.ltm.search(msg, k=5, filters={"mtype": "experience"})
    for e in exps:
        print("-", e.text)

    print("\n--- PROJECT 5 DEMO END ---\n")


if __name__ == "__main__":
    demo()
