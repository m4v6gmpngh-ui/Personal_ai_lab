from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class Event:
    """A normalized unit of experience entering the personal AI."""

    text: str
    timestamp: str = field(default_factory=utc_now_iso)
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

    def __post_init__(self) -> None:
        self.text = self.text.strip()
        if not self.text:
            raise ValueError("Event text cannot be empty")

        self.importance = _clamp01(self.importance)
        self.emotional_weight = _clamp01(self.emotional_weight)
        self.confidence = _clamp01(self.confidence)

        self.entities = _normalize(self.entities)
        self.topics = _normalize(self.topics)
        self.people = _normalize(self.people)
        self.emotions = _normalize(self.emotions)
        self.goals = _normalize(self.goals)


def _normalize(values: set[str]) -> set[str]:
    return {item.strip().lower() for item in values if item.strip()}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
