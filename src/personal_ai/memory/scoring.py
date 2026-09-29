from __future__ import annotations

import re

from personal_ai.core.event import Event
from personal_ai.memory.node import MemoryNode

_TOKEN_RE = re.compile(r"[a-z0-9_'-]+")


def tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def similarity_score(event: Event, memory: MemoryNode) -> float:
    """Transparent baseline relevance score in the range [0, 1]."""

    event_tokens = tokenize(event.text)
    memory_tokens = tokenize(memory.text)

    union = event_tokens | memory_tokens
    lexical = len(event_tokens & memory_tokens) / len(union) if union else 0.0

    topic_union = event.topics | memory.topics
    topic = len(event.topics & memory.topics) / len(topic_union) if topic_union else 0.0

    entity_union = event.entities | memory.entities
    entity = len(event.entities & memory.entities) / len(entity_union) if entity_union else 0.0

    salience = (
        0.55 * memory.importance
        + 0.25 * memory.emotional_weight
        + 0.20 * memory.confidence
    )

    score = 0.45 * lexical + 0.25 * topic + 0.20 * entity + 0.10 * salience
    return max(0.0, min(1.0, score))
