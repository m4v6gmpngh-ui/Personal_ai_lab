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
    base_score: float
    direct_score: float
    matrix_score: float
    confidence: float
    importance: float
    source: str
    timestamp: str
    recall_count: int
    accessibility: float
    matched_facets: dict[str, list[str]]
    reinforced_facets: dict[str, list[str]]
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
                f"[score={memory.score:.3f}; confidence={memory.confidence:.3f}; "
                f"source={memory.source}]"
            )
            lines.append(f"   why: {memory.reason}")

        lines.append(f"SUPPRESSED MEMORIES: {self.suppressed_count}")
        return "\n".join(lines)

    def to_debug_context(self) -> str:
        """Detailed provenance view for inspecting why each memory was recalled."""

        lines = [
            f"CURRENT EVENT: {self.current_event}",
            "",
            "ACTIVE CUES:",
        ]
        if self.activated_matrices:
            for family, keys in self.activated_matrices.items():
                lines.append(f"- {family}: {', '.join(keys)}")
        else:
            lines.append("- none")

        lines.extend(["", "RECALL TRACE:"])
        if not self.memories:
            lines.append("- no memories crossed the retrieval threshold")

        for memory in self.memories:
            lines.extend(
                [
                    f"{memory.rank}. MEMORY {memory.memory_id}",
                    f"   text: {memory.text}",
                    f"   source: {memory.source}",
                    f"   stored: {memory.timestamp}",
                    (
                        "   scores: "
                        f"final={memory.score:.4f}; "
                        f"content={memory.base_score:.4f}; "
                        f"secondary={memory.matrix_score:.4f}; "
                        f"blended_seed={memory.direct_score:.4f}; "
                        f"confidence={memory.confidence:.4f}; "
                        f"importance={memory.importance:.4f}"
                    ),
                    (
                        "   recall history: "
                        f"count={memory.recall_count}; "
                        f"accessibility={memory.accessibility:.4f}"
                    ),
                    (
                        "   matched cues: "
                        + (
                            "; ".join(
                                f"{family}=[{', '.join(values)}]"
                                for family, values in memory.matched_facets.items()
                            )
                            if memory.matched_facets
                            else "none (graph propagation may have activated it)"
                        )
                    ),
                    (
                        "   reinforced route: "
                        + (
                            "; ".join(
                                f"{family}=[{', '.join(values)}]"
                                for family, values in memory.reinforced_facets.items()
                            )
                            if memory.reinforced_facets
                            else "none"
                        )
                    ),
                    f"   graph path: {' -> '.join(memory.path)}",
                    f"   why: {memory.reason}",
                    "",
                ]
            )

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

        event_facets = self.secondary.event_facets(event)
        memories: list[ContextMemory] = []
        for rank, trace in enumerate(traces, start=1):
            node = self.store.get(trace.memory_id)
            memory_facets = self.secondary.memory_facets(node)
            matched_facets = {
                family: sorted(event_facets[family] & memory_facets[family])
                for family in event_facets
                if event_facets[family] & memory_facets[family]
            }
            recall_state = self.secondary.recall_state(node.id)

            memories.append(
                ContextMemory(
                    rank=rank,
                    memory_id=node.id,
                    text=node.text,
                    score=round(trace.score, 4),
                    base_score=round(trace.base_score, 4),
                    direct_score=round(trace.direct_score, 4),
                    matrix_score=round(trace.matrix_score, 4),
                    confidence=round(node.confidence, 4),
                    importance=round(node.importance, 4),
                    source=node.source,
                    timestamp=node.timestamp,
                    recall_count=recall_state.count,
                    accessibility=round(recall_state.accessibility, 4),
                    matched_facets=matched_facets,
                    reinforced_facets=dict(trace.reinforced_facets),
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
