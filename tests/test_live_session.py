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


if __name__ == "__main__":
    unittest.main()
