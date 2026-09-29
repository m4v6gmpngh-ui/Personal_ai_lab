from __future__ import annotations

from collections import defaultdict


class MemoryGraph:
    """Weighted graph used for Matrix Bloom spreading activation."""

    def __init__(self) -> None:
        self._edges: dict[str, dict[str, float]] = defaultdict(dict)

    def connect(self, left: str, right: str, weight: float, *, bidirectional: bool = True) -> None:
        if left == right:
            return
        weight = max(0.0, min(1.0, float(weight)))
        self._edges[left][right] = weight
        if bidirectional:
            self._edges[right][left] = weight

    def neighbors(self, memory_id: str) -> dict[str, float]:
        return dict(self._edges.get(memory_id, {}))

    def edge_weight(self, left: str, right: str) -> float:
        return self._edges.get(left, {}).get(right, 0.0)
