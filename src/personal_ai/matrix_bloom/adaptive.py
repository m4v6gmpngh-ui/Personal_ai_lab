from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from personal_ai.matrix_bloom.graph import MemoryGraph
from personal_ai.memory.tracer import RecallTrace


@dataclass(slots=True)
class LearningUpdate:
    left: str
    right: str
    before: float
    after: float
    reason: str


class AdaptiveBloom:
    """Small, explicit learning layer for Matrix Bloom relationships.

    v0.1 intentionally learns only from explicit signals. Merely recalling a
    memory does not automatically strengthen it; that avoids runaway feedback
    loops where frequently recalled mistakes become increasingly dominant.
    """

    def __init__(self, graph: MemoryGraph) -> None:
        self.graph = graph

    def reinforce_trace(
        self,
        trace: RecallTrace,
        *,
        reward: float = 1.0,
        learning_rate: float = 0.08,
    ) -> list[LearningUpdate]:
        """Strengthen the links that produced a useful recall."""

        reward = self._clamp(reward)
        learning_rate = self._clamp(learning_rate)
        updates: list[LearningUpdate] = []

        for left, right in zip(trace.path, trace.path[1:]):
            before = self.graph.edge_weight(left, right)
            delta = learning_rate * reward * (1.0 - before)
            after = self.graph.adjust_weight(left, right, delta)
            updates.append(
                LearningUpdate(left, right, before, after, "useful recall")
            )

        return updates

    def penalize_trace(
        self,
        trace: RecallTrace,
        *,
        penalty: float = 1.0,
        learning_rate: float = 0.06,
        minimum_weight: float = 0.01,
    ) -> list[LearningUpdate]:
        """Weaken links that contributed to an explicitly unhelpful recall."""

        penalty = self._clamp(penalty)
        learning_rate = self._clamp(learning_rate)
        updates: list[LearningUpdate] = []

        for left, right in zip(trace.path, trace.path[1:]):
            before = self.graph.edge_weight(left, right)
            delta = -(learning_rate * penalty * before)
            after = self.graph.adjust_weight(
                left,
                right,
                delta,
                remove_below=minimum_weight,
            )
            updates.append(
                LearningUpdate(left, right, before, after, "unhelpful recall")
            )

        return updates

    def coactivate(
        self,
        memory_ids: list[str],
        *,
        strength: float = 0.03,
    ) -> list[LearningUpdate]:
        """Associate memories that are explicitly judged relevant together."""

        strength = self._clamp(strength)
        updates: list[LearningUpdate] = []
        unique_ids = list(dict.fromkeys(memory_ids))

        for left, right in combinations(unique_ids, 2):
            before = self.graph.edge_weight(left, right)
            delta = strength * (1.0 - before)
            after = self.graph.adjust_weight(left, right, delta)
            updates.append(
                LearningUpdate(left, right, before, after, "co-activation")
            )

        return updates

    def decay(self, *, rate: float = 0.002, minimum_weight: float = 0.01) -> None:
        """Apply slow forgetting to relationship weights, not memory contents."""

        rate = self._clamp(rate)
        self.graph.decay(1.0 - rate, minimum_weight=minimum_weight)

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))
