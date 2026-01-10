import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MessageMeta = Dict[str, str]


@dataclass
class VectorMemory:
    """
    Long-term memory using TF-IDF vectors and cosine similarity.

    Stores text memories and retrieves the most relevant ones
    based on semantic similarity.
    """
    texts: List[str] = field(default_factory=list)
    metas: List[MessageMeta] = field(default_factory=list)

    vectorizer: TfidfVectorizer = field(default_factory=lambda: TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000
    ))
    _matrix = None

    def add(self, text: str, meta: MessageMeta = None) -> None:
        self.texts.append(text)
        self.metas.append(meta or {})
        # Re-fit on each add (fine for small-scale learning)
        self._matrix = self.vectorizer.fit_transform(self.texts)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[float, str, MessageMeta]]:
        if not self.texts:
            return []

        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix)[0]

        idxs = np.argsort(-sims)[:top_k]

        results = []
        for i in idxs:
            results.append((float(sims[i]), self.texts[i], self.metas[i]))
        return results


def agent_step(user_text: str, memory: VectorMemory, top_k: int = 3) -> str:
    """
    One agent step:
    - retrieve relevant memories
    - store the current input
    - return the context that would be fed to an LLM
    """
    retrieved = memory.search(user_text, top_k=top_k)

    # Store after retrieval to avoid self-recall
    memory.add(user_text, meta={"source": "user"})

    lines = []
    lines.append("RETRIEVED MEMORIES:")
    if not retrieved:
        lines.append("- (none yet)")
    else:
        for score, text, meta in retrieved:
            lines.append(f"- score={score:.3f} | {text}")

    lines.append("\nCURRENT INPUT:")
    lines.append(f"User: {user_text}")

    return "\n".join(lines)


if __name__ == "__main__":
    memory = VectorMemory()

    print(agent_step("Hi, my name is Aditya", memory))
    print("-" * 60)
    print(agent_step("I am building memory agents for robotics", memory))
    print("-" * 60)
    print(agent_step("I want the agent to remember important facts about me", memory))
    print("-" * 60)
    print(agent_step("Do you remember my name?", memory))
    print("-" * 60)
    print(agent_step("What am I building memory agents for?", memory))
