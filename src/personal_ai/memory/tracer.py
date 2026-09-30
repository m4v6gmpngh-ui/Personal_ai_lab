from __future__ import annotations

from dataclasses import dataclass, field
import heapq

from personal_ai.core.event import Event
from personal_ai.matrix_bloom.graph import MemoryGraph
from personal_ai.matrix_bloom.secondary import SecondaryMatrices
from personal_ai.memory.node import MemoryNode
from personal_ai.memory.scoring import similarity_score
from personal_ai.memory.store import MemoryStore


@dataclass(slots=True)
class RecallTrace:
    memory_id: str
    score: float
    base_score: float
    direct_score: float
    matrix_score: float = 0.0
    path: list[str] = field(default_factory=list)
    reason: str = ""


class MemoryTracer:
    """Recall engine combining relevance, secondary matrices, and graph spread."""

    def __init__(
        self,
        store: MemoryStore,
        graph: MemoryGraph,
        *,
        secondary: SecondaryMatrices | None = None,
        spread_decay: float = 0.68,
        max_depth: int = 2,
        matrix_blend: float = 0.20,
    ) -> None:
        self.store = store
        self.graph = graph
        self.secondary = secondary
        self.spread_decay = spread_decay
        self.max_depth = max_depth
        self.matrix_blend = max(0.0, min(1.0, matrix_blend))

        if self.secondary is not None:
            self.secondary.index_memories(self.store.all())

    def recall(
        self,
        event: Event,
        *,
        limit: int = 5,
        threshold: float = 0.08,
        rehearse: bool = True,
    ) -> list[RecallTrace]:
        if limit <= 0:
            return []

        base = {node.id: similarity_score(event, node) for node in self.store.all()}
        matrix = {
            node.id: self.secondary.score(event, node.id) if self.secondary else 0.0
            for node in self.store.all()
        }
        direct = {
            memory_id: (
                (1.0 - self.matrix_blend) * base[memory_id]
                + self.matrix_blend * matrix[memory_id]
            )
            for memory_id in base
        }

        best_scores = dict(direct)
        best_paths = {memory_id: [memory_id] for memory_id in direct}
        origins = {memory_id: memory_id for memory_id in direct}

        heap: list[tuple[float, int, str, str, tuple[str, ...]]] = []
        for memory_id, score in direct.items():
            if score >= threshold:
                heapq.heappush(heap, (-score, 0, memory_id, memory_id, (memory_id,)))

        while heap:
            neg_score, depth, current, origin, path = heapq.heappop(heap)
            current_score = -neg_score
            if depth >= self.max_depth:
                continue

            for neighbor, edge_weight in self.graph.neighbors(current).items():
                if neighbor in path:
                    continue

                propagated = current_score * edge_weight * self.spread_decay
                if propagated <= best_scores.get(neighbor, 0.0):
                    continue

                best_scores[neighbor] = propagated
                best_paths[neighbor] = [*path, neighbor]
                origins[neighbor] = origin
                heapq.heappush(
                    heap,
                    (-propagated, depth + 1, neighbor, origin, (*path, neighbor)),
                )

        traces: list[RecallTrace] = []
        for memory_id, score in best_scores.items():
            if score < threshold:
                continue

            origin = origins[memory_id]
            path = best_paths[memory_id]
            base_score = base.get(memory_id, 0.0)
            direct_score = direct.get(memory_id, 0.0)
            matrix_score = matrix.get(memory_id, 0.0)

            if len(path) == 1:
                reason = (
                    f"content relevance={base_score:.3f}; "
                    f"secondary matrices={matrix_score:.3f}; "
                    f"blended seed={direct_score:.3f}"
                )
            else:
                reason = (
                    f"activated from {origin} through {len(path) - 1} weighted link(s); "
                    f"content relevance={base_score:.3f}; "
                    f"secondary matrices={matrix_score:.3f}; "
                    f"blended seed={direct_score:.3f}"
                )

            traces.append(
                RecallTrace(
                    memory_id=memory_id,
                    score=score,
                    base_score=base_score,
                    direct_score=direct_score,
                    matrix_score=matrix_score,
                    path=path,
                    reason=reason,
                )
            )

        traces.sort(key=lambda item: (-item.score, item.memory_id))
        traces = traces[:limit]

        if rehearse and self.secondary is not None:
            for trace in traces:
                memory = self.store.get(trace.memory_id)
                self.secondary.rehearse(
                    memory,
                    recall_strength=trace.score,
                )

        return traces

    def explain(self, trace: RecallTrace) -> dict[str, object]:
        memory: MemoryNode = self.store.get(trace.memory_id)
        return {
            "memory": memory.text,
            "score": round(trace.score, 4),
            "base_score": round(trace.base_score, 4),
            "direct_score": round(trace.direct_score, 4),
            "matrix_score": round(trace.matrix_score, 4),
            "path": trace.path,
            "reason": trace.reason,
        }
