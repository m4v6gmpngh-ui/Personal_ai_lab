import unittest

from personal_ai.core.event import Event
from personal_ai.live_session import LiveSession
from personal_ai.matrix_bloom.graph import MemoryGraph
from personal_ai.matrix_bloom.linker import auto_link
from personal_ai.memory.node import MemoryNode
from personal_ai.memory.store import MemoryStore


class _WorkshopLLM:
    def extract_event(self, text: str) -> Event:
        lowered = text.lower()
        emotions = set()
        entities = {"old red workshop"}

        if "safe" in lowered or "cedar" in lowered:
            emotions.add("safe")
            entities.add("cedar")
        if "afraid" in lowered or "fear" in lowered or "accident" in lowered:
            emotions.add("fear")
            entities.add("accident")

        return Event(
            text,
            entities=entities,
            topics={"workshop"},
            emotions=emotions,
            importance=0.9,
            emotional_weight=0.8 if emotions else 0.2,
            confidence=0.64,
        )

    def answer(self, user_text, packet) -> str:
        return "ok"


class LayeredBloomBehaviorTests(unittest.TestCase):
    def _session(self) -> tuple[LiveSession, MemoryNode, MemoryNode]:
        store = MemoryStore()
        safe = MemoryNode.from_event(
            Event(
                "The old red workshop smelled like cedar and made me feel safe.",
                entities={"old red workshop", "cedar"},
                topics={"workshop"},
                emotions={"safe"},
                importance=0.95,
                emotional_weight=0.8,
                confidence=0.73,
            ),
            memory_id="workshop-safe",
        )
        fear = MemoryNode.from_event(
            Event(
                "The old red workshop is also where a frightening accident happened.",
                entities={"old red workshop", "accident"},
                topics={"workshop"},
                emotions={"fear"},
                importance=0.95,
                emotional_weight=0.9,
                confidence=0.71,
            ),
            memory_id="workshop-fear",
        )
        store.add(safe)
        store.add(fear)

        graph = MemoryGraph()
        auto_link(store, graph)
        session = LiveSession(store, graph, _WorkshopLLM())
        return session, safe, fear

    def test_calm_new_episode_prefers_safe_route_without_erasing_fear_route(self) -> None:
        session, safe, fear = self._session()
        safe_confidence = safe.confidence
        fear_confidence = fear.confidence

        turn = session.handle(
            "The cedar smell in the old red workshop made me feel safe again."
        )
        new_id = turn.stored_memory_id
        self.assertIsNotNone(new_id)

        safe_weight = session.graph.edge_weight(new_id, safe.id)
        fear_weight = session.graph.edge_weight(new_id, fear.id)

        self.assertGreater(safe_weight, fear_weight)
        self.assertGreater(fear_weight, 0.0)
        self.assertEqual(safe.confidence, safe_confidence)
        self.assertEqual(fear.confidence, fear_confidence)

        reinforcement = {
            event.related_memory_id: event
            for event in session.pathway_ledger.all()
            if event.new_memory_id == new_id
            and event.reason == "contextual pathway reinforced during memory formation"
        }
        self.assertIn(safe.id, reinforcement)
        self.assertIn("emotion", reinforcement[safe.id].matched_facets)
        self.assertEqual(
            reinforcement[safe.id].matched_facets["emotion"],
            ("safe",),
        )

        if fear.id in reinforcement:
            self.assertNotIn("emotion", reinforcement[fear.id].matched_facets)

    def test_fear_new_episode_prefers_accident_route_without_erasing_safe_route(self) -> None:
        session, safe, fear = self._session()

        turn = session.handle(
            "Thinking about the accident in the old red workshop makes me afraid."
        )
        new_id = turn.stored_memory_id
        self.assertIsNotNone(new_id)

        safe_weight = session.graph.edge_weight(new_id, safe.id)
        fear_weight = session.graph.edge_weight(new_id, fear.id)

        self.assertGreater(fear_weight, safe_weight)
        self.assertGreater(safe_weight, 0.0)

        reinforcement = {
            event.related_memory_id: event
            for event in session.pathway_ledger.all()
            if event.new_memory_id == new_id
            and event.reason == "contextual pathway reinforced during memory formation"
        }
        self.assertIn(fear.id, reinforcement)
        self.assertEqual(
            reinforcement[fear.id].matched_facets["emotion"],
            ("fear",),
        )

        if safe.id in reinforcement:
            self.assertNotIn("emotion", reinforcement[safe.id].matched_facets)

    def test_opposite_emotional_routes_remain_distinct_after_both_new_episodes(self) -> None:
        session, safe, fear = self._session()

        calm = session.handle(
            "The cedar smell in the old red workshop made me feel safe again."
        )
        scared = session.handle(
            "Thinking about the accident in the old red workshop makes me afraid."
        )

        self.assertGreater(
            session.graph.edge_weight(calm.stored_memory_id, safe.id),
            session.graph.edge_weight(calm.stored_memory_id, fear.id),
        )
        self.assertGreater(
            session.graph.edge_weight(scared.stored_memory_id, fear.id),
            session.graph.edge_weight(scared.stored_memory_id, safe.id),
        )

        calm_memory = session.store.get(calm.stored_memory_id)
        fear_memory = session.store.get(scared.stored_memory_id)
        self.assertEqual(calm_memory.emotions, {"safe"})
        self.assertEqual(fear_memory.emotions, {"fear"})


if __name__ == "__main__":
    unittest.main()
