from __future__ import annotations

from collections import defaultdict


class MemoryGraph:
    """Weighted graph used for Matrix Bloom spreading activation."""

    def __init__(self) -> None:
        self._edges: dict[str, dict[str, float]] = defaultdict(dict)

    def connect(self, left: str, right: str, weight: float, *, bidirectional: bool = True) -> None:
        if left == right:
            return
        weight = self._clamp(weight)
        self._edges[left][right] = weight
        if bidirectional:
            self._edges[right][left] = weight

    def adjust_weight(
        self,
        left: str,
        right: str,
        delta: float,
        *,
        bidirectional: bool = True,
        remove_below: float = 0.0,
    ) -> float:
        """Adjust an edge and return its new weight.

        A weight at or below ``remove_below`` is removed so stale relationships can
        disappear instead of accumulating forever.
        """

        if left == right:
            return 0.0

        updated = self._clamp(self.edge_weight(left, right) + float(delta))
        if updated <= remove_below:
            self._edges.get(left, {}).pop(right, None)
            if bidirectional:
                self._edges.get(right, {}).pop(left, None)
            return 0.0

        self.connect(left, right, updated, bidirectional=bidirectional)
        return updated

    def decay(self, factor: float, *, minimum_weight: float = 0.01) -> None:
        """Decay all unique bidirectional edges once.

        ``factor`` must be between 0 and 1. Edges that fall below
        ``minimum_weight`` are removed.
        """

        factor = self._clamp(factor)
        for left, right, weight in self.edges():
            updated = weight * factor
            if updated < minimum_weight:
                self.adjust_weight(
                    left,
                    right,
                    -weight,
                    remove_below=minimum_weight,
                )
            else:
                self.connect(left, right, updated)

    def neighbors(self, memory_id: str) -> dict[str, float]:
        return dict(self._edges.get(memory_id, {}))

    def edge_weight(self, left: str, right: str) -> float:
        return self._edges.get(left, {}).get(right, 0.0)

    def edges(self) -> list[tuple[str, str, float]]:
        """Return each undirected edge once in deterministic order."""

        unique: list[tuple[str, str, float]] = []
        seen: set[tuple[str, str]] = set()
        for left in sorted(self._edges):
            for right in sorted(self._edges[left]):
                pair = tuple(sorted((left, right)))
                if pair in seen:
                    continue
                seen.add(pair)
                unique.append((pair[0], pair[1], self._edges[left][right]))
        return unique

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))
