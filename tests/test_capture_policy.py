import unittest

from personal_ai.core.event import Event
from personal_ai.memory.capture import MemoryCapturePolicy


class MemoryCapturePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.normal = MemoryCapturePolicy(mode="normal")

    def test_training_mode_stores_every_turn(self) -> None:
        policy = MemoryCapturePolicy(mode="training")
        decision = policy.decide(Event("What time is it?"))

        self.assertTrue(decision.durable)
        self.assertEqual(decision.category, "training_all")

    def test_normal_question_is_ephemeral(self) -> None:
        decision = self.normal.decide(
            Event("What does Thanksgiving at my mom's house remind me of?")
        )

        self.assertFalse(decision.durable)
        self.assertEqual(decision.category, "question")

    def test_explicit_remember_is_durable(self) -> None:
        decision = self.normal.decide(
            Event("Remember that Thanksgiving smells like apple pie.")
        )

        self.assertTrue(decision.durable)
        self.assertEqual(decision.category, "explicit_memory")

    def test_relationship_fact_is_durable(self) -> None:
        decision = self.normal.decide(
            Event("My daughter named the AI Corium.")
        )

        self.assertTrue(decision.durable)
        self.assertEqual(decision.category, "relationship_fact")

    def test_preference_is_durable(self) -> None:
        decision = self.normal.decide(
            Event("I prefer working on Matrix Bloom at night.")
        )

        self.assertTrue(decision.durable)
        self.assertEqual(decision.category, "preference")

    def test_recurring_pattern_is_durable(self) -> None:
        decision = self.normal.decide(
            Event("The football game is always on during Thanksgiving.")
        )

        self.assertTrue(decision.durable)
        self.assertEqual(decision.category, "recurring_pattern")

    def test_conversational_glue_is_ephemeral(self) -> None:
        decision = self.normal.decide(
            Event("Okay, that makes sense.")
        )

        self.assertFalse(decision.durable)
        self.assertEqual(decision.category, "conversational")


if __name__ == "__main__":
    unittest.main()
