from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4

from personal_ai.core.event import Event


@dataclass(slots=True)
class MemoryNode:
    """A stored memory with explicit salience and semantic labels."""

    id: str
    text: str
    timestamp: str
    entities: set[str] = field(default_factory=set)
    topics: set[str] = field(default_factory=set)
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
            importance=event.importance,
            emotional_weight=event.emotional_weight,
            confidence=event.confidence,
            source=event.source,
            metadata=dict(event.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["entities"] = sorted(self.entities)
        data["topics"] = sorted(self.topics)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryNode":
        copy = dict(data)
        copy["entities"] = set(copy.get("entities", []))
        copy["topics"] = set(copy.get("topics", []))
        return cls(**copy)
