import json
import unittest

from personal_ai.context_engine import ContextPacket
from personal_ai.llm.openai_client import OpenAIAdapter


class _Response:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text


class _Responses:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("text"):
            return _Response(
                json.dumps(
                    {
                        "entities": ["github"],
                        "topics": ["personal ai", "development"],
                        "people": ["amanik"],
                        "emotions": ["focused"],
                        "goals": ["build personal ai"],
                        "importance": 0.8,
                        "emotional_weight": 0.4,
                        "confidence": 0.9,
                    }
                )
            )
        return _Response("Model answer using retrieved context.")


class _Client:
    def __init__(self) -> None:
        self.responses = _Responses()


class OpenAIAdapterTests(unittest.TestCase):
    def test_extracts_event_facets(self) -> None:
        client = _Client()
        adapter = OpenAIAdapter(client=client, model="test-model")

        event = adapter.extract_event("Keep building the personal AI in GitHub.")

        self.assertIn("github", event.entities)
        self.assertIn("personal ai", event.topics)
        self.assertIn("build personal ai", event.goals)
        self.assertEqual(event.source, "live_chat")

    def test_answers_with_context_packet(self) -> None:
        client = _Client()
        adapter = OpenAIAdapter(client=client, model="test-model")
        packet = ContextPacket(
            current_event="Hello",
            activated_matrices={"concept": ["memory"]},
            memories=[],
            suppressed_count=0,
        )

        answer = adapter.answer("Hello", packet)

        self.assertEqual(answer, "Model answer using retrieved context.")
        self.assertIn("MEMORY CONTEXT", client.responses.calls[-1]["input"])


if __name__ == "__main__":
    unittest.main()
