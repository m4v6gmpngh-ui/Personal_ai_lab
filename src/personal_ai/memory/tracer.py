from __future__ import annotations

from dataclasses import dataclass, field
import heapq

from personal_ai.core.event import Event
from personal_ai.matrix_bloom.graph import MemoryGraph
from personal_ai.memory.node import MemoryNode
from personal_ai.memory.scoring import similarity_score
from personal_ai.memory.store import MemoryStore


@dataclass(slots=True)
class RecallTrace:
    memory_id: str
    score: float
    direct_score: float
    path: list[str] = field(default_factory=list)
    reason: str = ""


class MemoryTracer:
    """Recall engine that combines direct relevance with graph activation."""

    def __init__(
        self,
        store: MemoryStore,
        graph: MemoryGraph,
        *,
        spread_decay: float = 0.68,
        max_depth: int = 2,
    ) -> None:
        self.store = store
        self.graph = graph
        self.spread_decay = spread_decay
        self.max_depth = max_depth

    def recall(self, event: Event, *, limit: int = 5, threshold: float = 0.08) -> list[RecallTrace]:
        if limit <= 0:
            return []

        direct = {node.id: similarity_score(event, node) for node in self.store.all()}
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
            direct_score = direct.get(memory_id, 0.0)
            if len(path) == 1:
                reason = f"direct relevance={direct_score:.3f}"
            else:
                reason = (
                    f"activated from {origin} through {len(path) - 1} weighted link(s); "
                    f"direct relevance={direct_score:.3f}"
                )
            traces.append(
                RecallTrace(
                    memory_id=memory_id,
                    score=score,
                    direct_score=direct_score,
                    path=path,
                    reason=reason,
                )
            )

        traces.sort(key=lambda item: (-item.score, item.memory_id))
        return traces[:limit]

    def explain(self, trace: RecallTrace) -> dict[str, object]:
        memory: MemoryNode = self.store.get(trace.memory_id)
        return {
            "memory": memory.text,
            "score": round(trace.score, 4),
            "direct_score": round(trace.direct_score, 4),
            "path": trace.path,
            "reason": trace.reason,
        }
