import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from personal_ai.live_session import LiveSession
from personal_ai.llm.ollama_local import OllamaLocalAdapter


class _FakeHttpResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class OllamaLocalAdapterTests(unittest.TestCase):
    def test_live_session_returns_answer_and_isolated_state(self):
        adapter = OllamaLocalAdapter(model="test-local-model")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memories.json"
            with patch(
                "personal_ai.llm.ollama_local.urlopen",
                return_value=_FakeHttpResponse(
                    {"message": {"content": "I can help with that."}}
                ),
            ) as send:
                session = LiveSession.from_disk(
                    path, adapter, capture_mode="normal"
                )
                result = session.handle(
                    "Remember that the blue workbench is for model experiments."
                )
                self.assertEqual(result.answer, "I can help with that.")
                self.assertTrue(path.exists())
                self.assertIsNotNone(result.stored_memory_id)
                request = send.call_args.args[0]
                payload = json.loads(request.data.decode("utf-8"))
                self.assertEqual(payload["model"], "test-local-model")
                self.assertFalse(payload["stream"])
                self.assertIn(
                    "MATRIX BLOOM CONTEXT",
                    payload["messages"][1]["content"],
                )

    def test_missing_ollama_is_an_explicit_error_not_cloud_fallback(self):
        adapter = OllamaLocalAdapter()
        from personal_ai.context_engine import ContextPacket
        packet = ContextPacket(
            current_event="Hello",
            activated_matrices={},
            memories=[],
            suppressed_count=0,
        )
        with patch(
            "personal_ai.llm.ollama_local.urlopen",
            side_effect=URLError("not running"),
        ):
            with self.assertRaisesRegex(RuntimeError, "local Ollama"):
                adapter.answer("Hello", packet)


if __name__ == "__main__":
    unittest.main()
