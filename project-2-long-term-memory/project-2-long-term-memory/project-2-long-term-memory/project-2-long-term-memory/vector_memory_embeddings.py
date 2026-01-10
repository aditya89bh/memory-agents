import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer


@dataclass
class MemoryItem:
    """
    A single memory entry with metadata.
    """
    text: str
    source: str = "user"          # user | assistant | system
    mtype: str = "note"           # identity | preference | goal | fact | note
    tags: List[str] = field(default_factory=list)
    ts: float = field(default_factory=time.time)
    salience: float = 0.5
    pinned: bool = False


@dataclass
class NeuralVectorMemory:
    """
    Long-term memory using neural embeddings (Sentence Transformers).

    - add(item): stores item and its embedding
    - search(query): retrieves top-k by cosine similarity (normalized dot product)
    - supports type/tag filtering before ranking
    """
    items: List[MemoryItem] = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None  # shape: (N, D)

    model_name: str = "all-MiniLM-L6-v2"
    _model: SentenceTransformer = field(default=None, init=False, repr=False)

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def add(self, item: MemoryItem) -> None:
        model = self._get_model()
        vec = model.encode([item.text], normalize_embeddings=True)  # (1, D)
        vec = np.asarray(vec)

        self.items.append(item)
        if self.embeddings is None:
            self.embeddings = vec
        else:
            self.embeddings = np.vstack([self.embeddings, vec])

    def search(
        self,
        query: str,
        top_k: int = 3,
        type_filter: Optional[List[str]] = None,
        tag_filter_any: Optional[List[str]] = None
    ) -> List[Tuple[float, MemoryItem]]:
        if not self.items or self.embeddings is None:
            return []

        # Candidate selection via metadata filters
        candidate_idxs = []
        for idx, it in enumerate(self.items):
            if type_filter and it.mtype not in type_filter:
                continue
            if tag_filter_any and not any(t in it.tags for t in tag_filter_any):
                continue
            candidate_idxs.append(idx)

        if not candidate_idxs:
            return []

        model = self._get_model()
        q = model.encode([query], normalize_embeddings=True)
        q = np.asarray(q)[0]  # (D,)

        # Cosine similarity because embeddings are normalized
        sims = self.embeddings @ q  # (N,)

        scored = [(float(sims[i]), i) for i in candidate_idxs]
        scored.sort(reverse=True, key=lambda x: x[0])

        return [(score, self.items[i]) for score, i in scored[:top_k]]


if __name__ == "__main__":
    mem = NeuralVectorMemory()

    # Seed some structured memories
    mem.add(MemoryItem("My name is Aditya", mtype="identity", tags=["name"], salience=0.95, pinned=True))
    mem.add(MemoryItem("I am building memory agents for robotics", mtype="goal", tags=["robotics", "agents"], salience=0.9))
    mem.add(MemoryItem("I prefer Python first and Rust later", mtype="preference", tags=["language"], salience=0.85))
    mem.add(MemoryItem("Project 2 is about long-term memory retrieval", mtype="fact", tags=["project"], salience=0.7))

    # Tests
    print("TEST 1: Identity retrieval")
    for score, item in mem.search("What's my name again?", top_k=3, type_filter=["identity"]):
        print(score, item.mtype, item.text)

    print("\nTEST 2: Paraphrase goal retrieval")
    for score, item in mem.search("Why am I making these memory agents?", top_k=3, type_filter=["goal"]):
        print(score, item.mtype, item.text)

    print("\nTEST 3: Tag filter retrieval")
    for score, item in mem.search("robots", top_k=5, tag_filter_any=["robotics"]):
        print(score, item.tags, item.text)
