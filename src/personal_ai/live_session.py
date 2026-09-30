from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from personal_ai.context_engine import ContextEngine, ContextPacket
from personal_ai.core.working_state import WorkingState
from personal_ai.matrix_bloom.graph import MemoryGraph
from personal_ai.matrix_bloom.linker import auto_link, link_new_memory
from personal_ai.matrix_bloom.secondary import SecondaryMatrices
from personal_ai.memory.capture import CaptureDecision, MemoryCapturePolicy
from personal_ai.memory.node import MemoryNode
from personal_ai.memory.store import MemoryStore


@dataclass(slots=True)
class LiveTurn:
    answer: str
    packet: ContextPacket
    stored_memory_id: str | None
    capture: CaptureDecision


class LiveSession:
    """One live Matrix Bloom -> model/agent conversation session."""

    STATE_VERSION = 1

    def __init__(
        self,
        store: MemoryStore,
        graph: MemoryGraph,
        llm: object,
        *,
        memory_path: str | Path | None = None,
        state_path: str | Path | None = None,
        secondary: SecondaryMatrices | None = None,
        capture_policy: MemoryCapturePolicy | None = None,
        working_state: WorkingState | None = None,
    ) -> None:
        self.store = store
        self.graph = graph
        self.llm = llm
        self.memory_path = Path(memory_path) if memory_path else None
        self.state_path = Path(state_path) if state_path else None
        self.secondary = secondary or SecondaryMatrices()
        self.capture_policy = capture_policy or MemoryCapturePolicy(mode="training")
        self.working_state = working_state or WorkingState()
        self.context_engine = ContextEngine(
            store,
            graph,
            secondary=self.secondary,
        )

    @classmethod
    def from_disk(
        cls,
        memory_path: str | Path,
        llm: object,
        *,
        state_path: str | Path | None = None,
        capture_mode: str = "training",
    ) -> "LiveSession":
        path = Path(memory_path)
        resolved_state_path = (
            Path(state_path)
            if state_path is not None
            else path.with_name(f"{path.stem}_bloom_state.json")
        )

        store = MemoryStore.load_json(path)
        graph = MemoryGraph()
        secondary = SecondaryMatrices()

        if resolved_state_path.exists():
            payload = json.loads(resolved_state_path.read_text(encoding="utf-8"))
            graph = MemoryGraph.from_dict(payload.get("graph"))
            secondary = SecondaryMatrices.from_dict(payload.get("secondary"))
        else:
            auto_link(store, graph)

        # Re-indexing only seeds missing memberships. Existing learned weights win.
        secondary.index_memories(store.all())

        return cls(
            store,
            graph,
            llm,
            memory_path=path,
            state_path=resolved_state_path,
            secondary=secondary,
            capture_policy=MemoryCapturePolicy(mode=capture_mode),
        )

    def handle(self, text: str, *, context_limit: int = 5) -> LiveTurn:
        event = self.llm.extract_event(text)
        recall_event, carried_state = self.working_state.enrich(event)
        packet = self.context_engine.build_packet(
            recall_event,
            limit=context_limit,
            rehearse=True,
            working_state=carried_state,
        )
        answer = self.llm.answer(text, packet)

        # Capture/storage uses only the current turn. Carried working state shapes
        # recall but must not leak into durable autobiographical memory.
        capture = self.capture_policy.decide(event)
        stored_memory_id: str | None = None

        if capture.durable:
            node = MemoryNode.from_event(event)
            node.metadata["capture_category"] = capture.category
            node.metadata["capture_score"] = capture.score
            node.metadata["capture_reasons"] = list(capture.reasons)

            existing_nodes = self.store.all()
            self.store.add(node)
            link_new_memory(node, existing_nodes, self.graph)
            self.context_engine.secondary.index_memory(node)
            stored_memory_id = node.id

        # Update short-lived state only after this turn's recall/storage decisions.
        self.working_state.advance(event)

        # Persist even when the current turn stays ephemeral because recall itself
        # may have strengthened secondary accessibility during this turn. Working
        # state is intentionally not persisted across process restarts.
        self._persist()

        return LiveTurn(
            answer=answer,
            packet=packet,
            stored_memory_id=stored_memory_id,
            capture=capture,
        )

    def _persist(self) -> None:
        if self.memory_path is not None:
            self.store.save_json(self.memory_path)

        if self.state_path is not None:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": self.STATE_VERSION,
                "graph": self.graph.to_dict(),
                "secondary": self.secondary.to_dict(),
            }
            self.state_path.write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
