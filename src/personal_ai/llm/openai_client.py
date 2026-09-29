from __future__ import annotations

import json
import os
from typing import Any

from personal_ai.context_engine import ContextPacket
from personal_ai.core.event import Event


EVENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entities": {"type": "array", "items": {"type": "string"}},
        "topics": {"type": "array", "items": {"type": "string"}},
        "people": {"type": "array", "items": {"type": "string"}},
        "emotions": {"type": "array", "items": {"type": "string"}},
        "goals": {"type": "array", "items": {"type": "string"}},
        "importance": {"type": "number", "minimum": 0, "maximum": 1},
        "emotional_weight": {"type": "number", "minimum": 0, "maximum": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "entities",
        "topics",
        "people",
        "emotions",
        "goals",
        "importance",
        "emotional_weight",
        "confidence",
    ],
    "additionalProperties": False,
}


class OpenAIAdapter:
    """Small adapter that places Matrix Bloom in front of an OpenAI model."""

    def __init__(self, *, client: Any | None = None, model: str | None = None) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5")
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "Install the OpenAI SDK first: pip install -r requirements.txt"
                ) from exc
            client = OpenAI()
        self.client = client

    def extract_event(self, text: str) -> Event:
        """Use a model as a perception layer for Matrix Bloom routing facets."""

        response = self.client.responses.create(
            model=self.model,
            instructions=(
                "Extract compact memory-routing facets from the user's current "
                "message. Use only information present or strongly implied in the "
                "message. People are named or clearly referenced people. Emotions "
                "are the emotional state expressed in the message, not a diagnosis. "
                "Goals are active intentions. Keep labels short and concrete."
            ),
            input=text,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "matrix_bloom_event",
                    "schema": EVENT_SCHEMA,
                    "strict": True,
                }
            },
        )
        data = json.loads(response.output_text)
        return Event(
            text=text,
            entities=set(data["entities"]),
            topics=set(data["topics"]),
            people=set(data["people"]),
            emotions=set(data["emotions"]),
            goals=set(data["goals"]),
            importance=float(data["importance"]),
            emotional_weight=float(data["emotional_weight"]),
            confidence=float(data["confidence"]),
            source="live_chat",
        )

    def answer(self, user_text: str, packet: ContextPacket) -> str:
        """Generate an answer using only the compact memory packet as continuity."""

        context = packet.to_prompt_context()
        response = self.client.responses.create(
            model=self.model,
            instructions=(
                "You are the response model behind a personal memory middleware. "
                "The MEMORY CONTEXT below contains retrieved recollections, not "
                "guaranteed facts. Use it when it genuinely improves continuity. "
                "Respect confidence values, do not invent missing details, and do "
                "not mention the retrieval machinery unless the user asks about it."
            ),
            input=(
                "MEMORY CONTEXT\n"
                "--------------\n"
                f"{context}\n\n"
                "USER MESSAGE\n"
                "------------\n"
                f"{user_text}"
            ),
        )
        return response.output_text
