"""Interactive ORION-01 console; uses local Ollama and isolated Matrix Bloom state."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from personal_ai.live_session import LiveSession
from personal_ai.llm.ollama_local import OllamaLocalAdapter


def main() -> None:
    root = Path("data/orion_sandbox")
    model = os.environ.get("ORION_MODEL", "qwen3.5:4b").strip()
    if not model:
        raise ValueError("ORION_MODEL must not be empty")
    session = LiveSession.from_disk(
        root / "memories.json",
        OllamaLocalAdapter(model=model),
        state_path=root / "bloom_state.json",
        capture_mode="normal",
    )
    print(f"Orion Chat v1 (local experiment, model={model}). Commands: /trace, /quit")
    print("No REA actions, autonomous skill execution, or skill consolidation.")
    last = None
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in ("/quit", "/exit"):
            break
        if question == "/trace":
            print(json.dumps(last or {"status": "no turns yet"}, indent=2))
            continue
        if not question:
            continue
        started = perf_counter()
        trace = {
            "run_id": str(uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_sha256": sha256(question.encode("utf-8")).hexdigest(),
            "status": "started",
            "tool_calls": [],
            "validation": "not_run",
            "skill_consolidation": "disabled",
        }
        try:
            turn = session.handle(question)
            trace.update(
                status="answered",
                answer_sha256=sha256(turn.answer.encode("utf-8")).hexdigest(),
                recalled_memory_ids=[memory.memory_id for memory in turn.packet.memories],
                stored_memory_id=turn.stored_memory_id,
            )
            print("Orion:", turn.answer)
        except Exception as exc:
            trace.update(status="error", error_type=type(exc).__name__)
            print("Orion is unavailable. Check that local Ollama and its model are running.")
        trace["elapsed_ms"] = round((perf_counter() - started) * 1000)
        root.mkdir(parents=True, exist_ok=True)
        with (root / "execution.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(trace) + "\n")
        last = trace


if __name__ == "__main__":
    main()
