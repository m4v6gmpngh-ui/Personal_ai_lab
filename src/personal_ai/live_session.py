from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from personal_ai.context_engine import ContextEngine, ContextPacket
from personal_ai.matrix_bloom.graph import MemoryGraph
from personal_ai.matrix_bloom.linker import auto_link, link_new_memory
from personal_ai.memory.node import MemoryNode
from personal_ai.memory.store import MemoryStore


@dataclass(slots=True)
class LiveTurn:
    answer: str
    packet: ContextPacket
    stored_memory_id: str


class LiveSession:
    """One live Matrix Bloom -> LLM conversation session."""

    def __init__(
        self,
        store: MemoryStore,
        graph: MemoryGraph,
        llm: object,
        *,
        memory_path: str | Path | None = None,
    ) -> None:
        self.store = store
        self.graph = graph
        self.llm = llm
        self.memory_path = Path(memory_path) if memory_path else None
        self.context_engine = ContextEngine(store, graph)

    @classmethod
    def from_disk(
        cls,
        memory_path: str | Path,
        llm: object,
    ) -> "LiveSession":
        path = Path(memory_path)
        store = MemoryStore.load_json(path)
        graph = MemoryGraph()
        auto_link(store, graph)
        return cls(store, graph, llm, memory_path=path)

    def handle(self, text: str, *, context_limit: int = 5) -> LiveTurn:
        event = self.llm.extract_event(text)
        packet = self.context_engine.build_packet(
            event,
            limit=context_limit,
            rehearse=True,
        )
        answer = self.llm.answer(text, packet)

        node = MemoryNode.from_event(event)
        existing_nodes = self.store.all()
        self.store.add(node)
        link_new_memory(node, existing_nodes, self.graph)
        self.context_engine.secondary.index_memory(node)

        if self.memory_path is not None:
            self.store.save_json(self.memory_path)

        return LiveTurn(
            answer=answer,
            packet=packet,
            stored_memory_id=node.id,
        )
