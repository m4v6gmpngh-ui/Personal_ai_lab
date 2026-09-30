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
        before_unmatched_emotion = self.secondary.weight("emotion", "focused", "m1")
        before_confidence = self.memory.confidence

        traces = self.tracer.recall(query, rehearse=True, threshold=0.01)

        after_weight = self.secondary.weight("concept", "personal ai", "m1")
        after_unmatched_emotion = self.secondary.weight("emotion", "focused", "m1")
        state = self.secondary.recall_state("m1")

        self.assertEqual(len(traces), 1)
        self.assertGreater(after_weight, before_weight)
        self.assertEqual(after_unmatched_emotion, before_unmatched_emotion)
        self.assertIn("personal ai", traces[0].reinforced_facets["concept"])
        self.assertNotIn("emotion", traces[0].reinforced_facets)
        self.assertEqual(state.count, 1)
        self.assertGreater(state.accessibility, 0.0)
        self.assertEqual(self.memory.confidence, before_confidence)


    def test_recall_reinforces_matching_emotion_not_opposite_route(self) -> None:
        memory = MemoryNode.from_event(
            Event(
                "The cabin carries both calm and fear associations.",
                topics={"blue lake cabin"},
                emotions={"calm", "fear"},
                confidence=0.7,
            ),
            memory_id="mixed",
        )
        store = MemoryStore()
        store.add(memory)
        secondary = SecondaryMatrices(seed_weight=0.35)
        tracer = MemoryTracer(
            store,
            MemoryGraph(),
            secondary=secondary,
        )

        calm_before = secondary.weight("emotion", "calm", "mixed")
        fear_before = secondary.weight("emotion", "fear", "mixed")

        traces = tracer.recall(
            Event(
                "The blue lake cabin feels frightening.",
                topics={"blue lake cabin"},
                emotions={"fear"},
            ),
            rehearse=True,
            threshold=0.01,
        )

        calm_after = secondary.weight("emotion", "calm", "mixed")
        fear_after = secondary.weight("emotion", "fear", "mixed")

        self.assertEqual(len(traces), 1)
        self.assertEqual(calm_after, calm_before)
        self.assertGreater(fear_after, fear_before)
        self.assertEqual(traces[0].reinforced_facets["emotion"], ["fear"])

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
