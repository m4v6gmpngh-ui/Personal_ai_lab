import tempfile
import unittest
from pathlib import Path

from personal_ai.core.event import Event
from personal_ai.live_session import LiveSession


class _FakeLLM:
    def extract_event(self, text: str) -> Event:
        return Event(
            text,
            topics={"personal ai"},
            goals={"build personal ai"},
            importance=0.8,
        )

    def answer(self, user_text, packet) -> str:
        return f"answer with {len(packet.memories)} recalled memories"


class _NormalFakeLLM:
    def extract_event(self, text: str) -> Event:
        return Event(
            text,
            topics={"thanksgiving", "family"},
            people={"mom"},
            importance=0.5,
        )

    def answer(self, user_text, packet) -> str:
        return "ok"


class LiveSessionTests(unittest.TestCase):
    def test_turn_is_stored_and_available_to_next_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memories.json"
            session = LiveSession.from_disk(path, _FakeLLM())

            first = session.handle("We are designing Matrix Bloom.")
            second = session.handle("Continue the personal AI.")

            self.assertTrue(first.stored_memory_id)
            self.assertEqual(len(session.store), 2)
            self.assertGreaterEqual(len(second.packet.memories), 1)
            self.assertTrue(path.exists())

    def test_normal_mode_queries_recall_without_becoming_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memories.json"
            session = LiveSession.from_disk(
                path,
                _NormalFakeLLM(),
                capture_mode="normal",
            )

            remembered = session.handle(
                "Remember that Thanksgiving at my mom's house smells like apple pie."
            )
            before_query = len(session.store)

            query = session.handle(
                "What does Thanksgiving at my mom's house remind me of?"
            )

            self.assertTrue(remembered.stored_memory_id)
            self.assertGreaterEqual(len(query.packet.memories), 1)
            self.assertFalse(query.capture.durable)
            self.assertIsNone(query.stored_memory_id)
            self.assertEqual(len(session.store), before_query)

    def test_repeated_ephemeral_queries_rehearse_accessibility_not_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memories.json"
            session = LiveSession.from_disk(
                path,
                _NormalFakeLLM(),
                capture_mode="normal",
            )

            remembered = session.handle(
                "Remember that Thanksgiving at my mom's house smells like apple pie."
            )
            memory_id = remembered.stored_memory_id
            self.assertIsNotNone(memory_id)
            memory = session.store.get(memory_id)
            confidence_before = memory.confidence
            stored_before = len(session.store)

            first_query = session.handle(
                "What does Thanksgiving at my mom's house remind me of?"
            )
            first_state = session.secondary.recall_state(memory_id)

            second_query = session.handle(
                "What does Thanksgiving at my mom's house remind me of?"
            )
            second_state = session.secondary.recall_state(memory_id)

            self.assertFalse(first_query.capture.durable)
            self.assertFalse(second_query.capture.durable)
            self.assertIsNone(first_query.stored_memory_id)
            self.assertIsNone(second_query.stored_memory_id)
            self.assertEqual(len(session.store), stored_before)
            self.assertEqual(first_state.count, 1)
            self.assertEqual(second_state.count, 2)
            self.assertGreater(
                second_state.accessibility,
                first_state.accessibility,
            )
            self.assertEqual(memory.confidence, confidence_before)


if __name__ == "__main__":
    unittest.main()
