import unittest

from personal_ai.core.event import Event
from personal_ai.matrix_bloom.graph import MemoryGraph
from personal_ai.memory.node import MemoryNode
from personal_ai.memory.store import MemoryStore
from personal_ai.memory.tracer import MemoryTracer


class TracingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = MemoryStore()
        self.store.add(
            MemoryNode.from_event(
                Event(
                    "GitHub repo supports remote development",
                    entities={"github"},
                    topics={"development", "remote work"},
                    importance=0.9,
                ),
                memory_id="github",
            )
        )
        self.store.add(
            MemoryNode.from_event(
                Event(
                    "Matrix Bloom spreads activation through memory links",
                    entities={"matrix bloom"},
                    topics={"memory"},
                    importance=0.9,
                ),
                memory_id="matrix",
            )
        )
        self.store.add(
            MemoryNode.from_event(
                Event(
                    "Unrelated grocery reminder",
                    topics={"errands"},
                    importance=0.3,
                ),
                memory_id="groceries",
            )
        )

        self.graph = MemoryGraph()
        self.graph.connect("github", "matrix", 0.85)

    def test_graph_can_activate_related_memory(self) -> None:
        tracer = MemoryTracer(self.store, self.graph, spread_decay=0.8, max_depth=2)
        query = Event(
            "Continue development remotely with GitHub",
            entities={"github"},
            topics={"development", "remote work"},
        )

        traces = tracer.recall(query, limit=3, threshold=0.05)
        ids = [trace.memory_id for trace in traces]

        self.assertIn("github", ids)
        self.assertIn("matrix", ids)

        matrix_trace = next(trace for trace in traces if trace.memory_id == "matrix")
        self.assertEqual(matrix_trace.path, ["github", "matrix"])
        self.assertGreater(matrix_trace.score, matrix_trace.direct_score)

    def test_unrelated_memory_is_suppressed(self) -> None:
        tracer = MemoryTracer(self.store, self.graph)
        query = Event(
            "Continue development remotely with GitHub",
            entities={"github"},
            topics={"development", "remote work"},
        )

        traces = tracer.recall(query, limit=5, threshold=0.15)
        ids = [trace.memory_id for trace in traces]
        self.assertNotIn("groceries", ids)


if __name__ == "__main__":
    unittest.main()
