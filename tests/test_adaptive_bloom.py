import unittest

from personal_ai.matrix_bloom.adaptive import AdaptiveBloom
from personal_ai.matrix_bloom.graph import MemoryGraph
from personal_ai.memory.tracer import RecallTrace


class AdaptiveBloomTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = MemoryGraph()
        self.graph.connect("a", "b", 0.40)
        self.bloom = AdaptiveBloom(self.graph)
        self.trace = RecallTrace(
            memory_id="b",
            score=0.5,
            base_score=0.1,
            direct_score=0.1,
            path=["a", "b"],
        )

    def test_useful_recall_strengthens_path(self) -> None:
        updates = self.bloom.reinforce_trace(self.trace, reward=1.0, learning_rate=0.10)

        self.assertEqual(len(updates), 1)
        self.assertAlmostEqual(updates[0].before, 0.40)
        self.assertGreater(updates[0].after, 0.40)
        self.assertLess(updates[0].after, 1.0)

    def test_unhelpful_recall_weakens_path(self) -> None:
        updates = self.bloom.penalize_trace(self.trace, penalty=1.0, learning_rate=0.10)

        self.assertEqual(len(updates), 1)
        self.assertLess(updates[0].after, 0.40)

    def test_coactivation_creates_relationship(self) -> None:
        self.assertEqual(self.graph.edge_weight("a", "c"), 0.0)
        self.bloom.coactivate(["a", "c"], strength=0.20)
        self.assertAlmostEqual(self.graph.edge_weight("a", "c"), 0.20)

    def test_decay_reduces_old_relationships(self) -> None:
        self.bloom.decay(rate=0.10)
        self.assertAlmostEqual(self.graph.edge_weight("a", "b"), 0.36)


if __name__ == "__main__":
    unittest.main()
