import unittest

from personal_ai.context_engine import ContextEngine
from personal_ai.core.event import Event
from personal_ai.matrix_bloom.graph import MemoryGraph
from personal_ai.memory.node import MemoryNode
from personal_ai.memory.store import MemoryStore


class ContextEngineTests(unittest.TestCase):
    def test_builds_ranked_compact_context_packet(self) -> None:
        store = MemoryStore()
        store.add(
            MemoryNode.from_event(
                Event(
                    "Built the Personal AI memory engine.",
                    topics={"personal ai", "memory"},
                    people={"amanik"},
                    goals={"build personal ai"},
                    importance=0.95,
                ),
                memory_id="ai",
            )
        )
        store.add(
            MemoryNode.from_event(
                Event(
                    "GitHub is the remote development workspace.",
                    entities={"github"},
                    topics={"development"},
                    goals={"build personal ai"},
                    importance=0.8,
                ),
                memory_id="github",
            )
        )
        store.add(
            MemoryNode.from_event(
                Event(
                    "Buy groceries after work.",
                    topics={"errands"},
                    importance=0.3,
                ),
                memory_id="groceries",
            )
        )

        engine = ContextEngine(store, MemoryGraph())
        event = Event(
            "Continue building the personal AI.",
            topics={"personal ai", "development"},
            people={"amanik"},
            goals={"build personal ai"},
        )

        packet = engine.build_packet(
            event,
            limit=2,
            threshold=0.05,
            rehearse=False,
        )

        self.assertEqual(len(packet.memories), 2)
        self.assertEqual(packet.suppressed_count, 1)
        self.assertIn("concept", packet.activated_matrices)
        self.assertIn("person", packet.activated_matrices)
        self.assertIn("goal", packet.activated_matrices)

        prompt = packet.to_prompt_context()
        self.assertIn("CURRENT EVENT:", prompt)
        self.assertIn("RELEVANT MEMORIES:", prompt)
        self.assertIn("SUPPRESSED MEMORIES: 1", prompt)


if __name__ == "__main__":
    unittest.main()
