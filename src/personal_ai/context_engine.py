from __future__ import annotations

from dataclasses import dataclass, asdict

from personal_ai.core.event import Event
from personal_ai.matrix_bloom.graph import MemoryGraph
from personal_ai.matrix_bloom.secondary import SecondaryMatrices
from personal_ai.memory.store import MemoryStore
from personal_ai.memory.tracer import MemoryTracer


@dataclass(slots=True)
class ContextMemory:
    rank: int
    memory_id: str
    text: str
    score: float
    confidence: float
    importance: float
    path: list[str]
    reason: str


@dataclass(slots=True)
class ContextPacket:
    current_event: str
    activated_matrices: dict[str, list[str]]
    memories: list[ContextMemory]
    suppressed_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "current_event": self.current_event,
            "activated_matrices": self.activated_matrices,
            "memories": [asdict(memory) for memory in self.memories],
            "suppressed_count": self.suppressed_count,
        }

    def to_prompt_context(self) -> str:
        """Compact text packet suitable for handing to an LLM."""

        lines = [
            f"CURRENT EVENT: {self.current_event}",
            "ACTIVE MATRICES:",
        ]
        if self.activated_matrices:
            for family, keys in self.activated_matrices.items():
                lines.append(f"- {family}: {', '.join(keys)}")
        else:
            lines.append("- none")

        lines.append("RELEVANT MEMORIES:")
        for memory in self.memories:
            lines.append(
                f"{memory.rank}. {memory.text} "
                f"[score={memory.score:.3f}; confidence={memory.confidence:.3f}]"
            )
            lines.append(f"   why: {memory.reason}")

        lines.append(f"SUPPRESSED MEMORIES: {self.suppressed_count}")
        return "\n".join(lines)


class ContextEngine:
    """Middleware between a current event and the downstream language model."""

    def __init__(
        self,
        store: MemoryStore,
        graph: MemoryGraph,
        *,
        secondary: SecondaryMatrices | None = None,
    ) -> None:
        self.store = store
        self.graph = graph
        self.secondary = secondary or SecondaryMatrices()
        self.tracer = MemoryTracer(
            store,
            graph,
            secondary=self.secondary,
        )

    def build_packet(
        self,
        event: Event,
        *,
        limit: int = 5,
        threshold: float = 0.08,
        rehearse: bool = True,
    ) -> ContextPacket:
        traces = self.tracer.recall(
            event,
            limit=limit,
            threshold=threshold,
            rehearse=rehearse,
        )

        memories: list[ContextMemory] = []
        for rank, trace in enumerate(traces, start=1):
            node = self.store.get(trace.memory_id)
            memories.append(
                ContextMemory(
                    rank=rank,
                    memory_id=node.id,
                    text=node.text,
                    score=round(trace.score, 4),
                    confidence=round(node.confidence, 4),
                    importance=round(node.importance, 4),
                    path=list(trace.path),
                    reason=trace.reason,
                )
            )

        return ContextPacket(
            current_event=event.text,
            activated_matrices=self.secondary.active_facets(event),
            memories=memories,
            suppressed_count=max(0, len(self.store) - len(memories)),
        )
