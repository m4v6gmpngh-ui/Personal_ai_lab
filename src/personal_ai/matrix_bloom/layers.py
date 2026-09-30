from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable
from uuid import uuid4

from personal_ai.core.event import Event
from personal_ai.matrix_bloom.graph import MemoryGraph
from personal_ai.matrix_bloom.linker import shared_context_weight
from personal_ai.matrix_bloom.secondary import SecondaryMatrices
from personal_ai.memory.node import MemoryNode


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class PathwayEvent:
    """Append-only evidence that one contextual pathway changed.

    The logical id makes the learning update idempotent. Replaying the same
    memory-formation operation must not strengthen a pathway a second time.
    """

    event_id: str
    logical_id: str
    created_at: str
    new_memory_id: str
    related_memory_id: str
    before: float
    after: float
    recall_strength: float
    matched_facets: dict[str, tuple[str, ...]] = field(default_factory=dict)
    reason: str = ""


class PathwayLedger:
    """Small append-only learning ledger for pathway reinforcement."""

    def __init__(self) -> None:
        self._events: list[PathwayEvent] = []
        self._logical_ids: set[str] = set()

    def append(self, event: PathwayEvent) -> bool:
        if event.logical_id in self._logical_ids:
            return False
        self._logical_ids.add(event.logical_id)
        self._events.append(event)
        return True

    def all(self) -> list[PathwayEvent]:
        return list(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def to_dict(self) -> dict[str, Any]:
        return {"events": [asdict(event) for event in self._events]}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PathwayLedger":
        ledger = cls()
        if not data:
            return ledger

        for item in data.get("events", []):
            if not isinstance(item, dict):
                continue
            raw_facets = item.get("matched_facets", {})
            if not isinstance(raw_facets, dict):
                raw_facets = {}
            facets = {
                str(family): tuple(str(value) for value in values)
                for family, values in raw_facets.items()
                if isinstance(values, (list, tuple))
            }
            event = PathwayEvent(
                event_id=str(item.get("event_id", uuid4())),
                logical_id=str(item.get("logical_id", "")),
                created_at=str(item.get("created_at", _utc_now_iso())),
                new_memory_id=str(item.get("new_memory_id", "")),
                related_memory_id=str(item.get("related_memory_id", "")),
                before=float(item.get("before", 0.0)),
                after=float(item.get("after", 0.0)),
                recall_strength=float(item.get("recall_strength", 0.0)),
                matched_facets=facets,
                reason=str(item.get("reason", "")),
            )
            if event.logical_id and event.new_memory_id and event.related_memory_id:
                ledger.append(event)

        return ledger


@dataclass(frozen=True, slots=True)
class EncodingResult:
    memory_id: str
    formed_links: tuple[PathwayEvent, ...] = ()
    reinforced_links: tuple[PathwayEvent, ...] = ()

    @property
    def pathway_event_ids(self) -> tuple[str, ...]:
        return tuple(
            event.event_id
            for event in (*self.formed_links, *self.reinforced_links)
        )


class LayeredMemoryEncoder:
    """Place a new episode into the layered Matrix Bloom memory model.

    Current v0.1 mapping:
    - episode layer: MemoryNode (raw durable event)
    - feature layer: SecondaryMatrices facets
    - pathway layer: MemoryGraph plus append-only PathwayLedger
    - working layer: existing WorkingState / ContextPacket

    Semantic abstraction and consolidation are intentionally not promoted
    automatically yet. This encoder changes accessibility/relationships only;
    it never increases factual confidence.
    """

    def __init__(
        self,
        graph: MemoryGraph,
        secondary: SecondaryMatrices,
        *,
        ledger: PathwayLedger | None = None,
        minimum_link_weight: float = 0.15,
        pathway_learning_rate: float = 0.04,
    ) -> None:
        self.graph = graph
        self.secondary = secondary
        self.ledger = ledger if ledger is not None else PathwayLedger()
        self.minimum_link_weight = self._clamp(minimum_link_weight)
        self.pathway_learning_rate = self._clamp(pathway_learning_rate)

    def encode(
        self,
        node: MemoryNode,
        *,
        event: Event,
        existing_nodes: list[MemoryNode],
        recall_signals: Iterable[tuple[str, float]] = (),
    ) -> EncodingResult:
        """Encode a durable episode and strengthen contextual recall paths.

        recall_signals should contain memories admitted to the current context
        packet. A recalled memory is reinforced only when it also shares an
        active facet with the new event. This keeps graph-only activation from
        creating arbitrary new associations.
        """

        node.metadata.setdefault("memory_layer", "episode")
        self.secondary.index_memory(node)

        existing_by_id = {memory.id: memory for memory in existing_nodes}
        formed: list[PathwayEvent] = []
        reinforced: list[PathwayEvent] = []

        # Place the episode in the primary graph using explainable facet overlap.
        # Existing learned weights are never reset.
        for other in existing_nodes:
            weight = shared_context_weight(node, other)
            if weight < self.minimum_link_weight:
                continue
            if self.graph.edge_weight(node.id, other.id) > 0.0:
                continue

            event_record = PathwayEvent(
                event_id=str(uuid4()),
                logical_id=f"form:{node.id}:{other.id}",
                created_at=_utc_now_iso(),
                new_memory_id=node.id,
                related_memory_id=other.id,
                before=0.0,
                after=weight,
                recall_strength=0.0,
                matched_facets=self._matched_facets(event, other),
                reason="initial shared-context link",
            )
            if self.ledger.append(event_record):
                self.graph.connect(node.id, other.id, weight)
                formed.append(event_record)

        # Strengthen routes that were actually useful in understanding the
        # current event. Repetition changes accessibility, not factual truth.
        seen_recall_ids: set[str] = set()
        for memory_id, raw_strength in recall_signals:
            if memory_id in seen_recall_ids:
                continue
            seen_recall_ids.add(memory_id)

            related = existing_by_id.get(memory_id)
            if related is None:
                continue

            matched = self._matched_facets(event, related)
            if not matched:
                continue

            strength = self._clamp(raw_strength)
            before = self.graph.edge_weight(node.id, memory_id)
            delta = self.pathway_learning_rate * strength * (1.0 - before)
            after = self._clamp(before + delta)

            event_record = PathwayEvent(
                event_id=str(uuid4()),
                logical_id=f"reinforce:{node.id}:{memory_id}",
                created_at=_utc_now_iso(),
                new_memory_id=node.id,
                related_memory_id=memory_id,
                before=before,
                after=after,
                recall_strength=strength,
                matched_facets=matched,
                reason="contextual pathway reinforced during memory formation",
            )
            if self.ledger.append(event_record):
                self.graph.connect(node.id, memory_id, after)
                reinforced.append(event_record)

        return EncodingResult(
            memory_id=node.id,
            formed_links=tuple(formed),
            reinforced_links=tuple(reinforced),
        )

    @staticmethod
    def _matched_facets(
        event: Event,
        memory: MemoryNode,
    ) -> dict[str, tuple[str, ...]]:
        event_facets = SecondaryMatrices.event_facets(event)
        memory_facets = SecondaryMatrices.memory_facets(memory)
        return {
            family: tuple(sorted(event_facets[family] & memory_facets[family]))
            for family in event_facets
            if event_facets[family] & memory_facets[family]
        }

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))
