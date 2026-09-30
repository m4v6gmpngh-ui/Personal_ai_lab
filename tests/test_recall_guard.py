import unittest

from personal_ai.core.event import Event
from personal_ai.memory.recall_guard import MemoryRecallGuard


class RecallGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.guard = MemoryRecallGuard()

    def test_passive_emotion_without_anchor_is_suppressed(self) -> None:
        decision = self.guard.decide(
            Event(
                "I'm feeling afraid right now.",
                emotions={"fear"},
            )
        )
        self.assertFalse(decision.allow)
        self.assertIn("no non-emotional anchor", decision.reason)

    def test_emotion_with_concrete_anchor_is_allowed(self) -> None:
        decision = self.guard.decide(
            Event(
                "I'm afraid of the blue lake cabin.",
                topics={"blue lake cabin"},
                emotions={"fear"},
            )
        )
        self.assertTrue(decision.allow)
        self.assertIsNone(decision.reason)

    def test_explicit_emotional_retrieval_is_allowed(self) -> None:
        decision = self.guard.decide(
            Event(
                "What memories do I associate with fear?",
                emotions={"fear"},
            )
        )
        self.assertTrue(decision.allow)
        self.assertIsNone(decision.reason)


if __name__ == "__main__":
    unittest.main()
