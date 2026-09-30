import json
import tempfile
import unittest
from pathlib import Path

from personal_ai.core.event import Event
from personal_ai.live_session import LiveSession
from personal_ai.matrix_bloom.graph import MemoryGraph
from personal_ai.matrix_bloom.layers import (
    LayeredMemoryEncoder,
    PathwayEvent,
    PathwayLedger,
)
from personal_ai.matrix_bloom.linker import shared_context_weight
from personal_ai.matrix_bloom.secondary import SecondaryMatrices
from personal_ai.memory.node import MemoryNode


class _LayeredFakeLLM:
    def extract_event(self, text: str) -> Event:
        return Event(
            text,
            topics={"workshop", "troubleshooting"},
            emotions={"focused"},
            goals={"solve problem"},
            importance=0.9,
            confidence=0.63,
        )

    def answer(self, user_text, packet) -> str:
        return f"recalled={len(packet.memories)}"


class LayeredMemoryTests(unittest.TestCase):
    def test_new_episode_strengthens_recalled_context_path_without_changing_truth(self) -> None:
        old = MemoryNode.from_event(
            Event(
                "The old workshop was where I learned to diagnose problems.",
                topics={"workshop", "troubleshooting"},
                emotions={"focused"},
                goals={"solve problem"},
                confidence=0.41,
            ),
            memory_id="old",
        )
        new_event = Event(
            "I diagnosed another difficult problem in the workshop.",
            topics={"workshop", "troubleshooting"},
            emotions={"focused"},
            goals={"solve problem"},
            confidence=0.72,
        )
        new = MemoryNode.from_event(new_event, memory_id="new")

        graph = MemoryGraph()
        secondary = SecondaryMatrices()
        ledger = PathwayLedger()
        encoder = LayeredMemoryEncoder(graph, secondary, ledger=ledger)

        old_confidence = old.confidence
        new_confidence = new.confidence
        initial_weight = shared_context_weight(new, old)

        result = encoder.encode(
            new,
            event=new_event,
            existing_nodes=[old],
            recall_signals=[("old", 0.8)],
        )

        self.assertEqual(new.metadata["memory_layer"], "episode")
        self.assertGreater(graph.edge_weight("new", "old"), initial_weight)
        self.assertEqual(old.confidence, old_confidence)
        self.assertEqual(new.confidence, new_confidence)
        self.assertEqual(len(result.formed_links), 1)
        self.assertEqual(len(result.reinforced_links), 1)
        self.assertEqual(
            result.reinforced_links[0].matched_facets["emotion"],
            ("focused",),
        )
        self.assertEqual(len(ledger), 2)

    def test_graph_only_recall_without_shared_active_facet_does_not_form_path(self) -> None:
        old = MemoryNode.from_event(
            Event("Unrelated grocery reminder.", topics={"errands"}),
            memory_id="old",
        )
        new_event = Event(
            "I am working on Matrix Bloom.",
            topics={"personal ai"},
            goals={"build personal ai"},
        )
        new = MemoryNode.from_event(new_event, memory_id="new")

        graph = MemoryGraph()
        ledger = PathwayLedger()
        encoder = LayeredMemoryEncoder(
            graph,
            SecondaryMatrices(),
            ledger=ledger,
        )

        result = encoder.encode(
            new,
            event=new_event,
            existing_nodes=[old],
            recall_signals=[("old", 0.99)],
        )

        self.assertEqual(graph.edge_weight("new", "old"), 0.0)
        self.assertEqual(result.formed_links, ())
        self.assertEqual(result.reinforced_links, ())
        self.assertEqual(len(ledger), 0)

    def test_replaying_same_encoding_is_idempotent(self) -> None:
        old = MemoryNode.from_event(
            Event("Workshop troubleshooting.", topics={"workshop"}),
            memory_id="old",
        )
        event = Event("Another workshop problem.", topics={"workshop"})
        new = MemoryNode.from_event(event, memory_id="new")

        graph = MemoryGraph()
        ledger = PathwayLedger()
        encoder = LayeredMemoryEncoder(
            graph,
            SecondaryMatrices(),
            ledger=ledger,
        )

        encoder.encode(
            new,
            event=event,
            existing_nodes=[old],
            recall_signals=[("old", 0.7)],
        )
        weight_after_first = graph.edge_weight("new", "old")
        events_after_first = len(ledger)

        encoder.encode(
            new,
            event=event,
            existing_nodes=[old],
            recall_signals=[("old", 0.7)],
        )

        self.assertEqual(graph.edge_weight("new", "old"), weight_after_first)
        self.assertEqual(len(ledger), events_after_first)

    def test_pathway_ledger_round_trip_preserves_logical_identity(self) -> None:
        ledger = PathwayLedger()
        event = PathwayEvent(
            event_id="event-1",
            logical_id="reinforce:new:old",
            created_at="2026-09-30T00:00:00+00:00",
            new_memory_id="new",
            related_memory_id="old",
            before=0.2,
            after=0.25,
            recall_strength=0.8,
            matched_facets={"concept": ("workshop",)},
            reason="test",
        )
        self.assertTrue(ledger.append(event))
        self.assertFalse(ledger.append(event))

        restored = PathwayLedger.from_dict(ledger.to_dict())

        self.assertEqual(len(restored), 1)
        self.assertEqual(restored.all()[0].logical_id, event.logical_id)
        self.assertEqual(
            restored.all()[0].matched_facets["concept"],
            ("workshop",),
        )
        self.assertFalse(restored.append(event))

    def test_live_session_persists_layered_pathway_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_path = Path(tmp) / "memories.json"
            session = LiveSession.from_disk(memory_path, _LayeredFakeLLM())

            first = session.handle("First workshop problem.")
            second = session.handle("Second workshop problem.")

            self.assertIsNotNone(first.stored_memory_id)
            self.assertIsNotNone(second.stored_memory_id)
            self.assertGreater(len(session.pathway_ledger), 0)

            second_node = session.store.get(second.stored_memory_id)
            self.assertEqual(second_node.metadata["memory_layer"], "episode")
            self.assertTrue(second_node.metadata["pathway_event_ids"])

            state_path = memory_path.with_name(
                f"{memory_path.stem}_bloom_state.json"
            )
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["version"], 2)
            self.assertTrue(payload["pathways"]["events"])

            restored = LiveSession.from_disk(memory_path, _LayeredFakeLLM())
            self.assertEqual(
                len(restored.pathway_ledger),
                len(session.pathway_ledger),
            )


if __name__ == "__main__":
    unittest.main()
