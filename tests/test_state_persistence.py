import tempfile
import unittest
from pathlib import Path

from personal_ai.core.event import Event
from personal_ai.live_session import LiveSession


class _FakeLLM:
    def extract_event(self, text: str) -> Event:
        return Event(
            text,
            topics={"personal ai", "matrix bloom"},
            goals={"build personal ai"},
            importance=0.8,
        )

    def answer(self, user_text, packet) -> str:
        return "ok"


class StatePersistenceTests(unittest.TestCase):
    def test_recall_accessibility_and_graph_survive_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_path = Path(tmp) / "memories.json"

            first_session = LiveSession.from_disk(memory_path, _FakeLLM())
            first_turn = first_session.handle("We are building Matrix Bloom.")
            first_session.handle("Continue building the personal AI.")

            recall_before = first_session.secondary.recall_state(
                first_turn.stored_memory_id
            )
            self.assertGreaterEqual(recall_before.count, 1)
            self.assertGreater(recall_before.accessibility, 0.0)
            self.assertTrue(first_session.state_path.exists())

            edge_count_before = len(first_session.graph.edges())
            self.assertGreaterEqual(edge_count_before, 1)

            restarted = LiveSession.from_disk(memory_path, _FakeLLM())
            recall_after = restarted.secondary.recall_state(
                first_turn.stored_memory_id
            )

            self.assertEqual(recall_after.count, recall_before.count)
            self.assertAlmostEqual(
                recall_after.accessibility,
                recall_before.accessibility,
            )
            self.assertEqual(len(restarted.graph.edges()), edge_count_before)


if __name__ == "__main__":
    unittest.main()
