from __future__ import annotations

from dataclasses import dataclass
import re

from personal_ai.core.event import Event


_RETRIEVAL_INTENT_RE = re.compile(
    r"\b(remember|recall|remind|memory|memories|associate|association)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class RecallGuardDecision:
    allow: bool
    reason: str | None = None


class MemoryRecallGuard:
    """Prevent passive emotion from becoming an autobiographical recall loop.

    Emotion is allowed to bias recall when it is accompanied by a concrete
    anchor (concept, entity, person, or goal), and explicit retrieval requests
    may intentionally search by emotion. A bare emotional-state declaration
    updates working state but does not, by itself, pull matching memories into
    the active Bloom.
    """

    def decide(self, event: Event) -> RecallGuardDecision:
        if not event.emotions:
            return RecallGuardDecision(True)

        anchors = (
            set(event.topics)
            | set(event.entities)
            | set(event.people)
            | set(event.goals)
        )
        if anchors:
            return RecallGuardDecision(True)

        text = event.text.strip()
        explicit_retrieval = (
            text.endswith("?")
            or bool(_RETRIEVAL_INTENT_RE.search(text))
        )
        if explicit_retrieval:
            return RecallGuardDecision(True)

        return RecallGuardDecision(
            False,
            (
                "passive emotional state has no non-emotional anchor; "
                "working state may update, but autobiographical recall is suppressed"
            ),
        )
