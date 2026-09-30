from __future__ import annotations

from dataclasses import dataclass
import re

from personal_ai.core.event import Event


@dataclass(frozen=True, slots=True)
class CaptureDecision:
    durable: bool
    category: str
    score: float
    reasons: tuple[str, ...]

    def summary(self) -> str:
        action = "STORE" if self.durable else "EPHEMERAL"
        reason_text = "; ".join(self.reasons) if self.reasons else "no durable signal"
        return (
            f"{action} category={self.category} score={self.score:.2f} "
            f"because {reason_text}"
        )


class MemoryCapturePolicy:
    """Transparent gate deciding whether a turn becomes durable memory.

    Training mode stores every turn so retrieval behavior can be inspected.
    Normal mode stores only explicit memories, durable personal facts/preferences,
    commitments/decisions, recurring patterns, or sufficiently salient events.
    Ordinary questions and conversational glue remain ephemeral.
    """

    MODES = {"training", "normal"}

    _EXPLICIT = re.compile(
        r"^(?:please\s+)?remember\b|\bkeep (?:this|that) in mind\b|"
        r"\bdon't forget\b|\bdo not forget\b|^note that\b",
        re.IGNORECASE,
    )
    _PREFERENCE = re.compile(
        r"\b(?:i prefer|i like|i love|i dislike|i hate|my favorite)\b",
        re.IGNORECASE,
    )
    _RELATION_FACT = re.compile(
        r"\bmy\s+(?:daughter|son|wife|husband|spouse|mom|mother|dad|father|"
        r"sister|brother|friend|boss|coworker|co-worker)\b.*"
        r"\b(?:is|are|was|were|named|likes|loves|works|has|have|uses|prefers)\b",
        re.IGNORECASE,
    )
    _COMMITMENT = re.compile(
        r"\b(?:i need to|i have to|i will|i'll|we need to|we have to|we will|"
        r"we'll|plan to|planning to|deadline|due date|appointment|promised?|"
        r"we decided|we agreed|the plan is)\b",
        re.IGNORECASE,
    )
    _RECURRING = re.compile(
        r"\b(?:always|usually|often|every|each|normally|typically|recurring)\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        *,
        mode: str = "training",
        importance_threshold: float = 0.74,
        emotional_threshold: float = 0.45,
    ) -> None:
        normalized = mode.strip().lower()
        if normalized not in self.MODES:
            raise ValueError(
                f"Unsupported capture mode {mode!r}. "
                f"Use one of: {', '.join(sorted(self.MODES))}."
            )
        self.mode = normalized
        self.importance_threshold = float(importance_threshold)
        self.emotional_threshold = float(emotional_threshold)

    def decide(self, event: Event) -> CaptureDecision:
        if self.mode == "training":
            return CaptureDecision(
                durable=True,
                category="training_all",
                score=1.0,
                reasons=("training mode stores every turn",),
            )

        text = event.text.strip()
        is_question = text.endswith("?")

        if self._EXPLICIT.search(text):
            return CaptureDecision(
                durable=True,
                category="explicit_memory",
                score=1.0,
                reasons=("explicit remember/note instruction",),
            )

        # Questions should generally query memory, not become autobiography.
        if is_question:
            return CaptureDecision(
                durable=False,
                category="question",
                score=0.0,
                reasons=("ordinary question remains ephemeral",),
            )

        if self._COMMITMENT.search(text):
            return CaptureDecision(
                durable=True,
                category="commitment_or_decision",
                score=0.92,
                reasons=("commitment, plan, deadline, or explicit decision language",),
            )

        if self._RELATION_FACT.search(text):
            return CaptureDecision(
                durable=True,
                category="relationship_fact",
                score=0.90,
                reasons=("durable fact about a close relationship",),
            )

        if self._PREFERENCE.search(text):
            return CaptureDecision(
                durable=True,
                category="preference",
                score=0.88,
                reasons=("stable preference or aversion language",),
            )

        if self._RECURRING.search(text):
            return CaptureDecision(
                durable=True,
                category="recurring_pattern",
                score=0.84,
                reasons=("recurrence language suggests a persistent pattern",),
            )

        if event.importance >= self.importance_threshold:
            return CaptureDecision(
                durable=True,
                category="salient_event",
                score=min(1.0, event.importance),
                reasons=(
                    f"importance {event.importance:.2f} >= "
                    f"{self.importance_threshold:.2f}",
                ),
            )

        if event.emotional_weight >= self.emotional_threshold:
            return CaptureDecision(
                durable=True,
                category="emotionally_salient",
                score=min(1.0, event.emotional_weight),
                reasons=(
                    f"emotional weight {event.emotional_weight:.2f} >= "
                    f"{self.emotional_threshold:.2f}",
                ),
            )

        return CaptureDecision(
            durable=False,
            category="conversational",
            score=0.0,
            reasons=("no durable-memory signal crossed the capture gate",),
        )
