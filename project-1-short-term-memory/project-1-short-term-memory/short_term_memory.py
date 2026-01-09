from dataclasses import dataclass, field
from typing import List, Dict

Message = Dict[str, str]  # {"role": "user" | "assistant", "content": str}


@dataclass
class RollingWindowMemory:
    """
    Simple short-term memory with a fixed-size rolling window.
    Older messages are forgotten deterministically.
    """
    window_size: int = 3
    messages: List[Message] = field(default_factory=list)

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.window_size:
            self.messages = self.messages[-self.window_size:]

    def get_context(self) -> str:
        lines = []
        for msg in self.messages:
            speaker = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{speaker}: {msg['content']}")
        return "\n".join(lines)


def agent_step(user_text: str, memory: RollingWindowMemory) -> str:
    """
    One agent step:
    - store user input
    - read memory
    - produce a response (placeholder)
    """
    memory.add("user", user_text)

    response = "Current memory:\n\n" + memory.get_context()

    # NOTE: we intentionally do NOT store the full assistant response
    # to avoid recursive memory explosion.
    memory.add("assistant", "Acknowledged.")

    return response


if __name__ == "__main__":
    memory = RollingWindowMemory(window_size=3)

    print(agent_step("Hi, my name is Aditya", memory))
    print(agent_step("I am learning about memory agents", memory))
    print(agent_step("This is my third message", memory))
    print(agent_step("Do you remember my name?", memory))
