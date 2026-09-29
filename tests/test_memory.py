import tempfile
import unittest
from pathlib import Path

from personal_ai.core.event import Event
from personal_ai.memory.node import MemoryNode
from personal_ai.memory.store import MemoryStore


class MemoryStoreTests(unittest.TestCase):
    def test_event_becomes_memory(self) -> None:
        event = Event(
            "Repository connection fixed",
            entities={"GitHub"},
            topics={"Development"},
            importance=1.2,
        )
        node = MemoryNode.from_event(event, memory_id="m1")

        self.assertEqual(node.id, "m1")
        self.assertEqual(node.entities, {"github"})
        self.assertEqual(node.topics, {"development"})
        self.assertEqual(node.importance, 1.0)

    def test_store_round_trip_json(self) -> None:
        store = MemoryStore()
        store.add(
            MemoryNode.from_event(
                Event("Remember this", topics={"memory"}),
                memory_id="m1",
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memories.json"
            store.save_json(path)
            restored = MemoryStore.load_json(path)

        self.assertEqual(len(restored), 1)
        self.assertEqual(restored.get("m1").text, "Remember this")
        self.assertEqual(restored.get("m1").topics, {"memory"})


if __name__ == "__main__":
    unittest.main()
