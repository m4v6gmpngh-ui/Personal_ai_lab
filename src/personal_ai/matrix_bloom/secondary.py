from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

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
        event: Event | None = None,
        recall_strength: float,
        learning_rate: float = 0.025,
        accessibility_rate: float = 0.035,
    ) -> dict[str, list[str]]:
        """Strengthen the route actually used to recall a memory.

        When an event is supplied, only memory facets that are also active in the
        current event are reinforced. This makes learning path-dependent: recalling
        a memory through fear strengthens the fear route without equally boosting
        unrelated calm/person/goal routes. A small global accessibility trace still
        records that the memory was successfully reached at all.

        If no event is supplied, all facets are reinforced for backwards-compatible
        direct use. Confidence is never changed.
        """

        recall_strength = self._clamp(recall_strength)
        learning_rate = self._clamp(learning_rate)
        accessibility_rate = self._clamp(accessibility_rate)

        self.index_memory(memory)

        memory_facets = self.memory_facets(memory)
        if event is None:
            routed_facets = {
                family: set(keys)
                for family, keys in memory_facets.items()
                if keys
            }
        else:
            event_facets = self.event_facets(event)
            routed_facets = {
                family: memory_facets[family] & event_facets[family]
                for family in MATRIX_FAMILIES
                if memory_facets[family] & event_facets[family]
            }

        for family, keys in routed_facets.items():
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

        return {
            family: sorted(keys)
            for family, keys in routed_facets.items()
            if keys
        }

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

    def to_dict(self) -> dict[str, Any]:
        """Serialize learned routing/accessibility state without changing confidence."""

        weights: dict[str, dict[str, dict[str, float]]] = {}
        for family in MATRIX_FAMILIES:
            weights[family] = {
                key: dict(bucket)
                for key, bucket in self._weights[family].items()
            }

        recall = {
            memory_id: {
                "count": state.count,
                "accessibility": state.accessibility,
                "last_recalled_at": state.last_recalled_at,
            }
            for memory_id, state in self._recall.items()
        }

        return {
            "seed_weight": self.seed_weight,
            "weights": weights,
            "recall": recall,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SecondaryMatrices":
        if not data:
            return cls()

        secondary = cls(seed_weight=float(data.get("seed_weight", 0.35)))

        raw_weights = data.get("weights", {})
        if isinstance(raw_weights, dict):
            for family, buckets in raw_weights.items():
                if family not in MATRIX_FAMILIES or not isinstance(buckets, dict):
                    continue
                for key, memory_weights in buckets.items():
                    if not isinstance(memory_weights, dict):
                        continue
                    secondary._weights[family][str(key)] = {
                        str(memory_id): secondary._clamp(float(weight))
                        for memory_id, weight in memory_weights.items()
                    }

        raw_recall = data.get("recall", {})
        if isinstance(raw_recall, dict):
            for memory_id, item in raw_recall.items():
                if not isinstance(item, dict):
                    continue
                secondary._recall[str(memory_id)] = RecallState(
                    count=max(0, int(item.get("count", 0))),
                    accessibility=secondary._clamp(
                        float(item.get("accessibility", 0.0))
                    ),
                    last_recalled_at=item.get("last_recalled_at"),
                )

        return secondary

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
