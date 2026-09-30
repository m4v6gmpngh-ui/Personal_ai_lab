from __future__ import annotations

import re

from personal_ai.core.event import Event
from personal_ai.memory.node import MemoryNode

_TOKEN_RE = re.compile(r"[a-z0-9_'-]+")

_RETRIEVAL_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can",
    "could", "did", "do", "does", "for", "from", "had", "has", "have", "he",
    "her", "hers", "him", "his", "i", "if", "in", "into", "is", "it", "its",
    "me", "my", "of", "on", "or", "our", "ours", "she", "so", "that", "the",
    "their", "theirs", "them", "they", "this", "to", "us", "was", "we", "were",
    "what", "when", "where", "which", "who", "why", "will", "with", "would",
    "you", "your", "yours",
    # Command/conversational glue should not make unrelated memories look similar.
    "also", "always", "around", "feel", "like", "made", "make", "okay", "pretty",
    "remember", "remind", "reminded", "reminds",
}


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(text.lower())
        if token not in _RETRIEVAL_STOPWORDS
    }


def _overlap(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def similarity_score(event: Event, memory: MemoryNode) -> float:
    """Transparent baseline content relevance score in the range [0, 1]."""

    lexical = _overlap(tokenize(event.text), tokenize(memory.text))
    topic = _overlap(event.topics, memory.topics)
    entity = _overlap(event.entities, memory.entities)
    people = _overlap(event.people, memory.people)
    emotions = _overlap(event.emotions, memory.emotions)
    goals = _overlap(event.goals, memory.goals)

    salience = (
        0.55 * memory.importance
        + 0.25 * memory.emotional_weight
        + 0.20 * memory.confidence
    )

    score = (
        0.30 * lexical
        + 0.18 * topic
        + 0.10 * entity
        + 0.14 * people
        + 0.08 * emotions
        + 0.15 * goals
        + 0.05 * salience
    )
    return max(0.0, min(1.0, score))
