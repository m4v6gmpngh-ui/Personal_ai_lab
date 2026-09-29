from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

from personal_ai.core.event import Event
from personal_ai.memory.node import MemoryNode

MATRIX_FAMILIES = ("concept", "person", "emotion", "goal")


@dataclass(slots=True)
class RecallState:
    count: int = 0
    accessibility: float = 0.0
    last_recalled_at: str | None = None


class SecondaryMatrices:
    """Recall-sensitive indexes that sit beside the primary relationship graph.

    These matrices make a memory easier to reach when it has been recalled in
    matching contexts. They do not change factual confidence. That separation
    lets rehearsal shape accessibility without turning repetition into truth.
    """

    def __init__(self, *, seed_weight: float = 0.35) -> None:
        self.seed_weight = self._clamp(seed_weight)
        self._weights: dict[str, dict[str, dict[str, float]]] = {
            family: defaultdict(dict) for family in MATRIX_FAMILIES
        }
        self._recall: dict[str, RecallState] = defaultdict(RecallState)

    def index_memory(self, memory: MemoryNode) -> None:
        for family, keys in self.memory_facets(memory).items():
            for key in keys:
                current = self._weights[family][key].get(memory.id, 0.0)
                self._weights[family][key][memory.id] = max(current, self.seed_weight)

    def index_memories(self, memories: list[MemoryNode]) -> None:
        for memory in memories:
            self.index_memory(memory)

    def score(self, event: Event, memory_id: str) -> float:
        """Return matrix activation for a memory in the current event."""

        family_scores: list[float] = []
        for family, keys in self.event_facets(event).items():
            if not keys:
                continue
            matched = [
                self._weights[family].get(key, {}).get(memory_id, 0.0)
                for key in keys
            ]
            family_scores.append(max(matched, default=0.0))

        context_score = (
            sum(family_scores) / len(family_scores) if family_scores else 0.0
        )
        accessibility = self._recall[memory_id].accessibility

        # Context remains dominant. Accessibility is a small recall-history bias.
        return self._clamp(0.85 * context_score + 0.15 * accessibility)

    def rehearse(
        self,
        memory: MemoryNode,
        *,
        recall_strength: float,
        learning_rate: float = 0.025,
        accessibility_rate: float = 0.035,
    ) -> None:
        """Strengthen secondary matrix membership after a recall.

        Reinforcement saturates as weights approach 1.0. Confidence is untouched.
        """

        recall_strength = self._clamp(recall_strength)
        learning_rate = self._clamp(learning_rate)
        accessibility_rate = self._clamp(accessibility_rate)

        self.index_memory(memory)

        for family, keys in self.memory_facets(memory).items():
            for key in keys:
                before = self._weights[family][key].get(memory.id, self.seed_weight)
                delta = learning_rate * recall_strength * (1.0 - before)
                self._weights[family][key][memory.id] = self._clamp(before + delta)

        state = self._recall[memory.id]
        state.count += 1
        state.accessibility = self._clamp(
            state.accessibility
            + accessibility_rate * recall_strength * (1.0 - state.accessibility)
        )
        state.last_recalled_at = datetime.now(timezone.utc).isoformat()

    def decay(self, *, factor: float = 0.999, minimum_weight: float = 0.05) -> None:
        """Slowly fade learned accessibility while preserving memory contents."""

        factor = self._clamp(factor)

        for family in MATRIX_FAMILIES:
            for key in list(self._weights[family]):
                bucket = self._weights[family][key]
                for memory_id in list(bucket):
                    updated = bucket[memory_id] * factor
                    if updated < minimum_weight:
                        del bucket[memory_id]
                    else:
                        bucket[memory_id] = updated
                if not bucket:
                    del self._weights[family][key]

        for state in self._recall.values():
            state.accessibility *= factor

    def recall_state(self, memory_id: str) -> RecallState:
        state = self._recall[memory_id]
        return RecallState(
            count=state.count,
            accessibility=state.accessibility,
            last_recalled_at=state.last_recalled_at,
        )

    def weight(self, family: str, key: str, memory_id: str) -> float:
        return self._weights.get(family, {}).get(key.lower(), {}).get(memory_id, 0.0)

    @staticmethod
    def event_facets(event: Event) -> dict[str, set[str]]:
        return {
            "concept": set(event.topics) | set(event.entities),
            "person": set(event.people),
            "emotion": set(event.emotions),
            "goal": set(event.goals),
        }

    @staticmethod
    def memory_facets(memory: MemoryNode) -> dict[str, set[str]]:
        return {
            "concept": set(memory.topics) | set(memory.entities),
            "person": set(memory.people),
            "emotion": set(memory.emotions),
            "goal": set(memory.goals),
        }

    @classmethod
    def active_facets(cls, event: Event) -> dict[str, list[str]]:
        return {
            family: sorted(keys)
            for family, keys in cls.event_facets(event).items()
            if keys
        }

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))
