from __future__ import annotations

from dataclasses import dataclass

from personal_ai.core.event import Event


@dataclass(slots=True)
class WorkingCue:
    value: str
    turns_remaining: int


class WorkingState:
    """Short-lived conversational state used to shape recall across turns.

    This is deliberately separate from durable memory. For the first slice we
    carry emotional state only. Explicit emotion in the current turn overrides
    carried emotion; otherwise recent emotion can influence retrieval for a
    bounded number of following turns.
    """

    def __init__(self, *, emotion_ttl: int = 3) -> None:
        self.emotion_ttl = max(0, int(emotion_ttl))
        self._emotions: dict[str, WorkingCue] = {}

    def enrich(self, event: Event) -> tuple[Event, dict[str, list[str]]]:
        """Return a recall event plus the prior working state actually applied."""

        # Explicit current emotion wins over prior state instead of mixing two
        # potentially incompatible "right now" states.
        carried_emotions = set()
        if not event.emotions:
            carried_emotions = {
                cue.value
                for cue in self._emotions.values()
                if cue.turns_remaining > 0
            }

        carried: dict[str, list[str]] = {}
        if carried_emotions:
            carried["emotion"] = sorted(carried_emotions)

        recall_event = Event(
            text=event.text,
            timestamp=event.timestamp,
            entities=set(event.entities),
            topics=set(event.topics),
            people=set(event.people),
            emotions=set(event.emotions) | carried_emotions,
            goals=set(event.goals),
            importance=event.importance,
            emotional_weight=event.emotional_weight,
            confidence=event.confidence,
            source=event.source,
            metadata={
                **event.metadata,
                "working_state": carried,
            },
        )
        return recall_event, carried

    def advance(self, event: Event) -> None:
        """Update working state after a turn has been processed."""

        if event.emotions:
            # A stated current emotion replaces the prior current-emotion set.
            self._emotions = {
                emotion: WorkingCue(
                    value=emotion,
                    turns_remaining=self.emotion_ttl,
                )
                for emotion in event.emotions
                if self.emotion_ttl > 0
            }
            return

        expired: list[str] = []
        for key, cue in self._emotions.items():
            cue.turns_remaining -= 1
            if cue.turns_remaining <= 0:
                expired.append(key)

        for key in expired:
            del self._emotions[key]

    def snapshot(self) -> dict[str, list[str]]:
        emotions = sorted(
            cue.value
            for cue in self._emotions.values()
            if cue.turns_remaining > 0
        )
        return {"emotion": emotions} if emotions else {}
