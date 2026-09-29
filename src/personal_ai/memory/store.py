from __future__ import annotations

import json
from pathlib import Path

from personal_ai.memory.node import MemoryNode


class MemoryStore:
    """Small deterministic local memory store used by the v0.1 engine."""

    def __init__(self) -> None:
        self._nodes: dict[str, MemoryNode] = {}

    def add(self, node: MemoryNode) -> None:
        if node.id in self._nodes:
            raise ValueError(f"Memory already exists: {node.id}")
        self._nodes[node.id] = node

    def upsert(self, node: MemoryNode) -> None:
        self._nodes[node.id] = node

    def get(self, memory_id: str) -> MemoryNode:
        return self._nodes[memory_id]

    def all(self) -> list[MemoryNode]:
        return list(self._nodes.values())

    def __len__(self) -> int:
        return len(self._nodes)

    def save_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = [node.to_dict() for node in self.all()]
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "MemoryStore":
        store = cls()
        source = Path(path)
        if not source.exists():
            return store
        payload = json.loads(source.read_text(encoding="utf-8"))
        for item in payload:
            store.add(MemoryNode.from_dict(item))
        return store
