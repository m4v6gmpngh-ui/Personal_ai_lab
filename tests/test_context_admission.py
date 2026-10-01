import unittest

from personal_ai.context_engine import ContextEngine
from personal_ai.core.event import Event
from personal_ai.matrix_bloom.graph import MemoryGraph
from personal_ai.matrix_bloom.secondary import SecondaryMatrices
from personal_ai.memory.node import MemoryNode
from personal_ai.memory.store import MemoryStore


class ContextAdmissionTests(unittest.TestCase):
    def _build_engine(self):
        store = MemoryStore()
        safe_text = "The cedar smell in the old red workshop makes me feel safe again."
        safe_ids = ["safe-1", "safe-2", "safe-3", "safe-4"]
        for memory_id in safe_ids:
            store.add(
                MemoryNode.from_event(
                    Event(
                        safe_text,
                        entities={"old red workshop", "cedar"},
                        topics={"workshop"},
                        emotions={"safety"},
                        importance=0.8,
                    ),
                    memory_id=memory_id,
                )
            )
        store.add(
            MemoryNode.from_event(
                Event(
                    "Thinking about the accident in the old red workshop makes me afraid.",
                    entities={"old red workshop", "accident"},
                    topics={"workshop"},
                    emotions={"fear"},
                    importance=0.8,
                ),
                memory_id="fear-1",
            )
        )
        secondary = SecondaryMatrices()
        graph = MemoryGraph()
        return store, secondary, ContextEngine(store, graph, secondary=secondary)

    def test_duplicate_candidates_do_not_crowd_context_packet(self) -> None:
        store, secondary, engine = self._build_engine()

        packet = engine.build_packet(
            Event(
                "The old red workshop reminds me of cedar and the accident.",
                entities={"old red workshop", "cedar", "accident"},
                topics={"workshop"},
                emotions={"safety", "fear"},
            ),
            limit=5,
            threshold=0.01,
            rehearse=True,
        )

        texts = [memory.text for memory in packet.memories]

        self.assertEqual(len(texts), 2)
        self.assertEqual(
            len(
                [
                    text
                    for text in texts
                    if text
                    == "The cedar smell in the old red workshop makes me feel safe again."
                ]
            ),
            1,
        )
        self.assertIn(
            "Thinking about the accident in the old red workshop makes me afraid.",
            texts,
        )
        self.assertGreaterEqual(packet.admission_duplicate_suppressed, 3)
        self.assertGreaterEqual(packet.candidate_count, 5)

    def test_only_admitted_duplicate_is_rehearsed(self) -> None:
        store, secondary, engine = self._build_engine()

        packet = engine.build_packet(
            Event(
                "The cedar smell in the old red workshop feels safe.",
                entities={"old red workshop", "cedar"},
                topics={"workshop"},
                emotions={"safety"},
            ),
            limit=5,
            threshold=0.01,
            rehearse=True,
        )

        admitted_ids = {memory.memory_id for memory in packet.memories}
        duplicate_ids = {"safe-1", "safe-2", "safe-3", "safe-4"} - admitted_ids

        admitted_safe_ids = admitted_ids & {"safe-1", "safe-2", "safe-3", "safe-4"}
        self.assertEqual(len(admitted_safe_ids), 1)

        admitted_safe_id = next(iter(admitted_safe_ids))
        self.assertEqual(secondary.recall_state(admitted_safe_id).count, 1)

        for memory_id in duplicate_ids:
            self.assertEqual(secondary.recall_state(memory_id).count, 0)

    def test_duplicate_history_remains_in_store(self) -> None:
        store, _, engine = self._build_engine()

        before = len(store)
        packet = engine.build_packet(
            Event(
                "The cedar smell in the old red workshop feels safe.",
                entities={"old red workshop", "cedar"},
                topics={"workshop"},
                emotions={"safety"},
            ),
            limit=5,
            threshold=0.01,
        )

        self.assertEqual(len(store), before)
        self.assertLess(len(packet.memories), len(store))


if __name__ == "__main__":
    unittest.main()
