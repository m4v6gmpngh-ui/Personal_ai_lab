from __future__ import annotations

import os
from pathlib import Path

from personal_ai.live_session import LiveSession
from personal_ai.llm.openai_client import OpenAIAdapter


def main() -> None:
    memory_path = Path(os.getenv("MATRIX_BLOOM_MEMORY", "data/live_memories.json"))
    debug = os.getenv("MATRIX_BLOOM_DEBUG", "0") == "1"

    llm = OpenAIAdapter()
    session = LiveSession.from_disk(memory_path, llm)

    print("Matrix Bloom live AI test")
    print("Type /quit to exit. Memories are stored locally in", memory_path)
    print()

    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not text:
            continue
        if text.lower() in {"/quit", "/exit"}:
            break

        turn = session.handle(text)

        if debug:
            print("\n[Matrix Bloom context]")
            print(turn.packet.to_prompt_context())
            print()

        print(f"AI: {turn.answer}\n")


if __name__ == "__main__":
    main()
