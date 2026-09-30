from __future__ import annotations

import re

from personal_ai.core.event import Event

_TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can",
    "could", "did", "do", "does", "for", "from", "had", "has", "have", "he",
    "her", "hers", "him", "his", "i", "if", "in", "into", "is", "it", "its",
    "me", "my", "of", "on", "or", "our", "ours", "she", "so", "that", "the",
    "their", "theirs", "them", "they", "this", "to", "us", "was", "we", "were",
    "what", "when", "where", "which", "who", "why", "will", "with", "would",
    "you", "your", "yours",
    # Conversational/control words are useful for capture policy but poor routing cues.
    "also", "always", "around", "feel", "like", "made", "make", "okay", "pretty",
    "remember", "remind", "reminded", "reminds",
}

_KINSHIP = {
    "mom", "mother", "dad", "father", "wife", "husband", "spouse", "daughter",
    "son", "sister", "brother", "grandma", "grandmother", "grandpa",
    "grandfather", "friend", "boss", "coworker", "co-worker",
}

_EMOTION_WORDS = {
    "happy": "happy",
    "happier": "happy",
    "sad": "sad",
    "angry": "angry",
    "mad": "angry",
    "afraid": "fear",
    "fear": "fear",
    "scared": "fear",
    "worried": "worry",
    "anxious": "anxiety",
    "excited": "excited",
    "frustrated": "frustrated",
    "annoyed": "frustrated",
    "love": "affection",
    "loved": "affection",
    "grief": "grief",
    "miss": "longing",
    "nostalgic": "nostalgia",
    "comfort": "comfort",
    "safe": "safety",
}

_GOAL_MARKERS = {
    "want", "need", "plan", "planning", "trying", "build", "building",
    "create", "finish", "fix", "learn", "test", "improve",
}

_SALIENCE_MARKERS = {
    "always", "never", "important", "remember", "favorite", "tradition",
    "birthday", "thanksgiving", "christmas", "wedding", "funeral", "died",
    "death", "first", "last",
}

_SENSORY_MARKERS = {
    "smell", "smells", "scent", "taste", "tastes", "sound", "sounds", "music",
    "touch", "feels", "feeling", "color", "colors",
}


class LocalEventExtractor:
    """Small offline perception layer for testing Matrix Bloom without an API.

    It intentionally favors transparent lexical cues over pretending to perform
    deep semantic understanding. A stronger local model can replace it later.
    """

    def extract_event(self, text: str) -> Event:
        raw_tokens = _TOKEN_RE.findall(text)
        tokens = [self._normalize_token(token) for token in raw_tokens]
        content_tokens = [
            token for token in tokens
            if len(token) > 2 and token not in _STOPWORDS
        ]

        topics = set(content_tokens[:24])
        topics.update(self._bigrams(content_tokens, limit=16))

        entities = {
            self._normalize_token(token)
            for token in raw_tokens
            if token[:1].isupper()
            and self._normalize_token(token) not in _STOPWORDS
        }

        people = {token for token in tokens if token in _KINSHIP}
        emotions = {
            _EMOTION_WORDS[token]
            for token in tokens
            if token in _EMOTION_WORDS
        }

        goals: set[str] = set()
        if any(token in _GOAL_MARKERS for token in tokens):
            goals.add("active goal")

        marker_count = sum(token in _SALIENCE_MARKERS for token in tokens)
        sensory_count = sum(token in _SENSORY_MARKERS for token in tokens)

        importance = min(0.9, 0.5 + 0.08 * marker_count)
        emotional_weight = min(
            0.85,
            0.08 * sensory_count + 0.15 * len(emotions) + 0.05 * bool(people),
        )

        return Event(
            text=text,
            entities=entities,
            topics=topics,
            people=people,
            emotions=emotions,
            goals=goals,
            importance=importance,
            emotional_weight=emotional_weight,
            confidence=0.9,
            source="local_chat",
            metadata={"extractor": "local_keywords_v1"},
        )

    @staticmethod
    def _normalize_token(token: str) -> str:
        lowered = token.lower()
        if lowered.endswith("'s"):
            lowered = lowered[:-2]
        return lowered

    @staticmethod
    def _bigrams(tokens: list[str], *, limit: int) -> set[str]:
        phrases: set[str] = set()
        for left, right in zip(tokens, tokens[1:]):
            if left == right:
                continue
            phrases.add(f"{left} {right}")
            if len(phrases) >= limit:
                break
        return phrases
