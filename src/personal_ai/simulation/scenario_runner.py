from __future__ import annotations

from personal_ai.context_engine import ContextEngine
from personal_ai.core.event import Event
from personal_ai.matrix_bloom.graph import MemoryGraph
from personal_ai.matrix_bloom.linker import auto_link
from personal_ai.memory.node import MemoryNode
from personal_ai.memory.store import MemoryStore


def build_demo() -> tuple[MemoryStore, MemoryGraph]:
    store = MemoryStore()

    memories = [
        Event(
            "Created the Personal_ai_lab GitHub repository for the personal AI project.",
            entities={"github", "personal_ai_lab"},
            topics={"personal ai", "development", "remote work"},
            goals={"build personal ai"},
            importance=0.85,
        ),
        Event(
            "The Matrix Bloom design uses weighted links between memories so recall can spread through related context.",
            entities={"matrix bloom"},
            topics={"personal ai", "memory", "development"},
            goals={"build personal ai"},
            importance=0.95,
        ),
        Event(
            "Remote development should let the project continue without manually copying code between devices.",
            entities={"github"},
            topics={"development", "remote work"},
            goals={"build personal ai"},
            importance=0.75,
        ),
        Event(
            "The Circular Economics game also uses simulated agents and memory concepts, but it is a separate project.",
            entities={"circular economics"},
            topics={"simulation", "game development"},
            goals={"build ce game"},
            importance=0.55,
        ),
    ]

    for index, event in enumerate(memories, start=1):
        store.add(MemoryNode.from_event(event, memory_id=f"m{index}"))

    graph = MemoryGraph()
    auto_link(store, graph)
    return store, graph


def main() -> None:
    store, graph = build_demo()
    engine = ContextEngine(store, graph)

    current = Event(
        "I want to keep working on the personal AI from another device.",
        entities={"github"},
        topics={"personal ai", "remote work", "development"},
        goals={"build personal ai"},
        emotions={"focused"},
        importance=0.8,
    )

    packet = engine.build_packet(current, limit=4)
    print(packet.to_prompt_context())


if __name__ == "__main__":
    main()
