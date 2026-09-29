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
            self.assertTrue(adapter.instructions_path.exists())

            payload = json.loads(
                adapter.context_json_path.read_text(encoding="utf-8")
            )
            self.assertEqual(
                payload["current_event"],
                "What does Thanksgiving at my mom's house remind me of?",
            )
            self.assertGreaterEqual(len(payload["memories"]), 1)

    def test_local_extractor_captures_salient_cues(self) -> None:
        adapter = OpenClawWorkspaceAdapter(workspace_path="unused")
        event = adapter.extract_event(
            "Remember Thanksgiving at my mom's house smells like apple pie."
        )

        self.assertIn("thanksgiving", event.topics)
        self.assertIn("apple pie", event.topics)
        self.assertIn("mom", event.people)
        self.assertIn("remember", event.goals)
        self.assertGreater(event.importance, 0.5)


if __name__ == "__main__":
    unittest.main()
