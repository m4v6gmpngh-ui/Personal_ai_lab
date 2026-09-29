from __future__ import annotations

import os
from pathlib import Path

from personal_ai.live_session import LiveSession
from personal_ai.llm.openai_client import OpenAIAdapter
from personal_ai.llm.openclaw_workspace import OpenClawWorkspaceAdapter


def build_provider():
    provider = os.getenv("AI_PROVIDER", "openai").strip().lower()
    if provider == "openai":
        return provider, OpenAIAdapter()
    if provider == "openclaw":
        return provider, OpenClawWorkspaceAdapter()
    raise ValueError(
        f"Unsupported AI_PROVIDER={provider!r}. Use 'openai' or 'openclaw'."
    )


def main() -> None:
    memory_path = Path(os.getenv("MATRIX_BLOOM_MEMORY", "data/live_memories.json"))
    debug = os.getenv("MATRIX_BLOOM_DEBUG", "0") == "1"

    provider, llm = build_provider()
    session = LiveSession.from_disk(memory_path, llm)

    print(f"Matrix Bloom live test ({provider})")
    print("Type /quit to exit. Memories are stored locally in", memory_path)
    if provider == "openclaw":
        print("OpenClaw workspace:", llm.workspace_path)
        print(
            "This mode writes the current compact Bloom into OpenClaw's memory "
            "folder. It does not require OpenAI API credits."
        )
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
