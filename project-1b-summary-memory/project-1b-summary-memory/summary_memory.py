from dataclasses import dataclass, field
from typing import List, Dict

Message = Dict[str, str]  # {"role": "user" | "assistant", "content": str}


def simple_summarizer(prev_summary: str, chunk: List[Message]) -> str:
    """
    Heuristic summarizer (no LLM).
    Compresses older user intent into a 'story so far'.
    """
    user_lines = [m["content"] for m in chunk if m["role"] == "user"]
    highlights = user_lines[-4:]

    if not highlights:
        return prev_summary

    bullets = ["User context so far:"]
    for line in highlights:
        bullets.append(f"- {line[:150]}")

    new_summary = "\n".join(bullets)

    if prev_summary.strip():
        return prev_summary.strip() + "\n\n" + new_summary
    return new_summary


@dataclass
class SummaryMemory:
    """
    Two-tier memory:
    - Recent rolling buffer
    - Running summary for older context
    """
    window_size: int = 6
    summarize_trigger: int = 10
    summarize_keep_last: int = 4
    messages: List[Message] = field(default_factory=list)
    summary: str = ""

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

        if len(self.messages) > self.summarize_trigger:
            cut = len(self.messages) - self.summarize_keep_last
            old_messages = self.messages[:cut]
            self.summary = simple_summarizer(self.summary, old_messages)
            self.messages = self.messages[cut:]

        if len(self.messages) > self.window_size:
            self.messages = self.messages[-self.window_size:]

    def get_context(self) -> str:
        parts = []
        if self.summary:
            parts.append("SUMMARY:")
            parts.append(self.summary)

        parts.append("RECENT:")
        for m in self.messages:
            speaker = "User" if m["role"] == "user" else "Assistant"
            parts.append(f"{speaker}: {m['content']}")

        return "\n".join(parts)


def agent_step(user_text: str, memory: SummaryMemory) -> str:
    """
    One agent step (Colab-safe).
    """
    memory.add("user", user_text)
    context = memory.get_context()
    memory.add("assistant", "OK")  # avoid recursion bloat
    return context


if __name__ == "__main__":
    mem = SummaryMemory()

    print(agent_step("Hi, my name is Aditya", mem))
    print(agent_step("I am learning about memory agents", mem))
    print(agent_step("I want agents that remember important things", mem))
    print(agent_step("Now testing summary memory", mem))
    print(agent_step("Message 1", mem))
    print(agent_step("Message 2", mem))
    print(agent_step("Can you recall the conversation?", mem))
