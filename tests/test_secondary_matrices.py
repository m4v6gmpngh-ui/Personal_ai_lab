import unittest

from personal_ai.core.event import Event
from personal_ai.matrix_bloom.graph import MemoryGraph
from personal_ai.matrix_bloom.secondary import SecondaryMatrices
from personal_ai.memory.node import MemoryNode
from personal_ai.memory.store import MemoryStore
from personal_ai.memory.tracer import MemoryTracer


class SecondaryMatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = MemoryStore()
        self.memory = MemoryNode.from_event(
            Event(
                "Worked on the personal AI memory engine with GitHub.",
                entities={"github"},
                topics={"personal ai", "memory"},
                people={"amanik"},
                emotions={"focused"},
                goals={"build personal ai"},
                confidence=0.42,
                importance=0.9,
            ),
            memory_id="m1",
        )
        self.store.add(self.memory)
        self.secondary = SecondaryMatrices(seed_weight=0.35)
        self.tracer = MemoryTracer(
            self.store,
            MemoryGraph(),
            secondary=self.secondary,
        )

    def test_recall_strengthens_secondary_matrix_not_confidence(self) -> None:
        query = Event(
            "Continue the personal AI work.",
            topics={"personal ai"},
            people={"amanik"},
            goals={"build personal ai"},
        )

        before_weight = self.secondary.weight("concept", "personal ai", "m1")
        before_confidence = self.memory.confidence

        traces = self.tracer.recall(query, rehearse=True, threshold=0.01)

        after_weight = self.secondary.weight("concept", "personal ai", "m1")
        state = self.secondary.recall_state("m1")

        self.assertEqual(len(traces), 1)
        self.assertGreater(after_weight, before_weight)
        self.assertEqual(state.count, 1)
        self.assertGreater(state.accessibility, 0.0)
        self.assertEqual(self.memory.confidence, before_confidence)

    def test_evaluation_recall_can_disable_rehearsal(self) -> None:
        query = Event(
            "Continue the personal AI work.",
            topics={"personal ai"},
            goals={"build personal ai"},
        )

        before = self.secondary.weight("concept", "personal ai", "m1")
        self.tracer.recall(query, rehearse=False, threshold=0.01)
        after = self.secondary.weight("concept", "personal ai", "m1")

        self.assertEqual(before, after)
        self.assertEqual(self.secondary.recall_state("m1").count, 0)


if __name__ == "__main__":
    unittest.main()
