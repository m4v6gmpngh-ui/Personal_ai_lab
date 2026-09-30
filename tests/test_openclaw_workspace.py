import json
import tempfile
import unittest
from pathlib import Path

from personal_ai.live_session import LiveSession
from personal_ai.llm.openclaw_workspace import OpenClawWorkspaceAdapter


class OpenClawWorkspaceTests(unittest.TestCase):
    def test_bridge_exports_compact_bloom_and_recalls_prior_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "openclaw" / "workspace"
            memory_path = root / "matrix" / "memories.json"

            adapter = OpenClawWorkspaceAdapter(workspace_path=workspace)
            session = LiveSession.from_disk(memory_path, adapter)

            first = session.handle(
                "Remember that Thanksgiving at my mom's house always smells "
                "like apple pie and the football game is always on."
            )
            second = session.handle(
                "What does Thanksgiving at my mom's house remind me of?"
            )

            self.assertTrue(first.stored_memory_id)
            self.assertGreaterEqual(len(second.packet.memories), 1)
            recalled_text = "\n".join(
                memory.text for memory in second.packet.memories
            )
            self.assertIn("apple pie", recalled_text.lower())

            self.assertTrue(adapter.context_markdown_path.exists())
            self.assertTrue(adapter.context_json_path.exists())
            self.assertTrue(adapter.trace_markdown_path.exists())
            self.assertTrue(adapter.instructions_path.exists())

            payload = json.loads(
                adapter.context_json_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                payload["current_event"],
                "What does Thanksgiving at my mom's house remind me of?",
            )
            self.assertGreaterEqual(len(payload["memories"]), 1)
            self.assertIn("matched_facets", payload["memories"][0])

            trace_text = adapter.trace_markdown_path.read_text(encoding="utf-8")
            self.assertIn("RECALL TRACE:", trace_text)
            self.assertIn("thanksgiving", trace_text.lower())

    def test_local_extractor_captures_salient_cues(self) -> None:
        adapter = OpenClawWorkspaceAdapter(workspace_path="unused")
        event = adapter.extract_event(
            "Remember Thanksgiving at my mom's house smells like apple pie."
        )

        self.assertIn("thanksgiving", event.topics)
        self.assertIn("apple pie", event.topics)
        self.assertIn("mom", event.people)
        self.assertNotIn("remember", event.topics)
        self.assertNotIn("remember", event.goals)
        self.assertGreater(event.importance, 0.5)

    def test_local_extractor_recognizes_calm_state(self) -> None:
        adapter = OpenClawWorkspaceAdapter(workspace_path="unused")
        event = adapter.extract_event(
            "The blue lake cabin makes me feel calm and relaxed."
        )

        self.assertIn("calm", event.emotions)

    def test_emotional_context_changes_which_conflicting_memory_leads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "openclaw" / "workspace"
            memory_path = root / "matrix" / "memories.json"

            adapter = OpenClawWorkspaceAdapter(workspace_path=workspace)
            session = LiveSession.from_disk(
                memory_path,
                adapter,
                capture_mode="normal",
            )

            calm = session.handle(
                "Remember that the blue lake cabin smells like pine "
                "and makes me feel calm."
            )
            fear = session.handle(
                "Remember that the blue lake cabin reminds me of a bad "
                "storm and makes me afraid."
            )

            afraid_query = session.handle(
                "When I feel afraid, what does the blue lake cabin remind me of?"
            )
            calm_query = session.handle(
                "When I feel calm, what does the blue lake cabin remind me of?"
            )

            self.assertEqual(
                afraid_query.packet.memories[0].memory_id,
                fear.stored_memory_id,
            )
            self.assertEqual(
                calm_query.packet.memories[0].memory_id,
                calm.stored_memory_id,
            )


    def test_contradictory_workshop_memories_beat_unrelated_remember_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "openclaw" / "workspace"
            memory_path = root / "matrix" / "memories.json"

            adapter = OpenClawWorkspaceAdapter(workspace_path=workspace)
            session = LiveSession.from_disk(
                memory_path,
                adapter,
                capture_mode="normal",
            )

            thanksgiving = session.handle(
                "Remember that Thanksgiving at my mom's house always smells "
                "like apple pie and the football game is always on."
            )
            safe = session.handle(
                "Remember that the old red workshop smelled like cedar "
                "and always made me feel safe."
            )
            anxious = session.handle(
                "Remember that the old red workshop also reminds me of a "
                "frightening accident and can make me anxious."
            )

            query = session.handle(
                "What does the old red workshop remind me of?"
            )

            ids = [memory.memory_id for memory in query.packet.memories]
            self.assertIn(safe.stored_memory_id, ids)
            self.assertIn(anxious.stored_memory_id, ids)
            self.assertNotIn(thanksgiving.stored_memory_id, ids)


if __name__ == "__main__":
    unittest.main()
