from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4

from personal_ai.core.event import Event


@dataclass(slots=True)
class MemoryNode:
    """A stored memory with explicit salience and contextual facets."""

    id: str
    text: str
    timestamp: str
    entities: set[str] = field(default_factory=set)
    topics: set[str] = field(default_factory=set)
    people: set[str] = field(default_factory=set)
    emotions: set[str] = field(default_factory=set)
    goals: set[str] = field(default_factory=set)
    importance: float = 0.5
    emotional_weight: float = 0.0
    confidence: float = 1.0
    source: str = "conversation"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_event(cls, event: Event, *, memory_id: str | None = None) -> "MemoryNode":
        return cls(
            id=memory_id or str(uuid4()),
            text=event.text,
            timestamp=event.timestamp,
            entities=set(event.entities),
            topics=set(event.topics),
            people=set(event.people),
            emotions=set(event.emotions),
            goals=set(event.goals),
            importance=event.importance,
            emotional_weight=event.emotional_weight,
            confidence=event.confidence,
            source=event.source,
            metadata=dict(event.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for field_name in ("entities", "topics", "people", "emotions", "goals"):
            data[field_name] = sorted(getattr(self, field_name))
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryNode":
        copy = dict(data)
        for field_name in ("entities", "topics", "people", "emotions", "goals"):
            copy[field_name] = set(copy.get(field_name, []))
        return cls(**copy)
