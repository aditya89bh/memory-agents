import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class MemoryItem:
    """
    A single memory entry with metadata.
    """
    text: str
    source: str = "user"      # user | assistant | system
    mtype: str = "note"       # identity | preference | goal | skill | fact | task | note
    tags: List[str] = field(default_factory=list)
    ts: float = field(default_factory=time.time)


@dataclass
class VectorMemoryMeta:
    """
    Long-term memory with semantic retrieval + metadata filters.
    """
    items: List[MemoryItem] = field(default_factory=list)

    vectorizer: TfidfVectorizer = field(default_factory=lambda: TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=7000
    ))
    _matrix = None

    def _rebuild_index(self) -> None:
        texts = [it.text for it in self.items]
        self._matrix = self.vectorizer.fit_transform(texts) if texts else None

    def add(self, item: MemoryItem) -> None:
        self.items.append(item)
        self._rebuild_index()

    def search(
        self,
        query: str,
        top_k: int = 3,
        type_filter: Optional[List[str]] = None,
        source_filter: Optional[List[str]] = None,
        tag_filter_any: Optional[List[str]] = None
    ) -> List[Tuple[float, MemoryItem]]:
        """
        Semantic search with optional metadata filters.
        """
        if not self.items or self._matrix is None:
            return []

        candidate_idxs = []
        for idx, it in enumerate(self.items):
            if type_filter and it.mtype not in type_filter:
                continue
            if source_filter and it.source not in source_filter:
                continue
            if tag_filter_any and not any(t in it.tags for t in tag_filter_any):
                continue
            candidate_idxs.append(idx)

        if not candidate_idxs:
            return []

        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix)[0]

        scored = [(float(sims[i]), i) for i in candidate_idxs]
        scored.sort(reverse=True, key=lambda x: x[0])

        return [(score, self.items[i]) for score, i in scored[:top_k]]


def remember(
    memory: VectorMemoryMeta,
    text: str,
    mtype: str,
    source: str = "user",
    tags: Optional[List[str]] = None
) -> None:
    """
    Helper to store structured memories.
    """
    memory.add(MemoryItem(
        text=text,
        source=source,
        mtype=mtype,
        tags=tags or []
    ))


def agent_step(
    user_text: str,
    memory: VectorMemoryMeta,
    top_k: int = 3,
    retrieve_types: Optional[List[str]] = None
) -> str:
    """
    Agent step using metadata-aware retrieval.
    """
    retrieved = memory.search(
        user_text,
        top_k=top_k,
        type_filter=retrieve_types
    )

    # Store raw input as a generic note (will be gated in 2C)
    remember(memory, user_text, mtype="note", source="user", tags=["raw"])

    lines = []
    lines.append("RETRIEVED MEMORIES:")
    if not retrieved:
        lines.append("- (none)")
    else:
        for score, item in retrieved:
            lines.append(
                f"- score={score:.3f} | type={item.mtype} | source={item.source} | tags={item.tags} | text={item.text}"
            )

    lines.append("\nCURRENT INPUT:")
    lines.append(f"User: {user_text}")
    return "\n".join(lines)


if __name__ == "__main__":
    mem = VectorMemoryMeta()

    remember(mem, "My name is Aditya", mtype="identity", tags=["name"])
    remember(mem, "I am building memory agents for robotics", mtype="goal", tags=["robotics"])
    remember(mem, "I prefer Python first and Rust later", mtype="preference", tags=["language"])

    print(agent_step("Do you remember my name?", mem, retrieve_types=["identity"]))
    print("-" * 60)
    print(agent_step("What am I building these agents for?", mem, retrieve_types=["goal"]))
    print("-" * 60)
    print(agent_step("Which languages am I using?", mem, retrieve_types=["preference"]))
