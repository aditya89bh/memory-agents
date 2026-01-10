import time
import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class MemoryItem:
    """
    A single memory with salience and lifecycle control.
    """
    text: str
    source: str = "user"
    mtype: str = "note"            # identity | preference | goal | fact | note
    tags: List[str] = field(default_factory=list)
    ts: float = field(default_factory=time.time)
    salience: float = 0.5          # 0..1 importance score
    pinned: bool = False           # pinned memories never decay


@dataclass
class VectorMemory:
    """
    Vector-based long-term memory with salience gating.
    """
    items: List[MemoryItem] = field(default_factory=list)

    vectorizer: TfidfVectorizer = field(default_factory=lambda: TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=9000
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
        type_filter: Optional[List[str]] = None
    ) -> List[Tuple[float, MemoryItem]]:
        if not self.items or self._matrix is None:
            return []

        candidate_idxs = []
        for idx, it in enumerate(self.items):
            if type_filter and it.mtype not in type_filter:
                continue
            candidate_idxs.append(idx)

        if not candidate_idxs:
            return []

        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix)[0]

        scored = [(float(sims[i]), i) for i in candidate_idxs]
        scored.sort(reverse=True, key=lambda x: x[0])

        return [(score, self.items[i]) for score, i in scored[:top_k]]


# ---------- Salience & Gating Logic ----------

TYPE_PATTERNS = [
    ("identity",  [r"\bmy name is\b"]),
    ("preference",[r"\bi prefer\b", r"\bi like\b", r"\bi love\b", r"\bi hate\b"]),
    ("goal",      [r"\bi want to\b", r"\bi am building\b", r"\bi am working on\b"]),
    ("fact",      [r"\bremember that\b", r"\bnote that\b"]),
]

def infer_type(text: str) -> str:
    t = text.lower()
    for mtype, pats in TYPE_PATTERNS:
        if any(re.search(p, t) for p in pats):
