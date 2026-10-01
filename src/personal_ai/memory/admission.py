from __future__ import annotations

from dataclasses import dataclass, field
import re

from personal_ai.memory.store import MemoryStore
from personal_ai.memory.tracer import RecallTrace


_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    admitted: tuple[RecallTrace, ...]
    duplicate_suppressed_ids: tuple[str, ...] = ()
    budget_suppressed_ids: tuple[str, ...] = ()

    @property
    def suppressed_count(self) -> int:
        return len(self.duplicate_suppressed_ids) + len(self.budget_suppressed_ids)


class ContextAdmissionGate:
    """Second-stage gate between candidate retrieval and LLM context.

    Candidate retrieval is intentionally permissive. Admission is stricter:
    exact/normalized duplicate memories are collapsed so repeated episodes do not
    consume the entire context budget. The underlying memories remain stored and
    available for provenance/history.

    This v0.1 gate deliberately does *not* perform semantic deduplication.
    """

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def admit(
        self,
        traces: list[RecallTrace],
        *,
        limit: int,
    ) -> AdmissionResult:
        if limit <= 0:
            return AdmissionResult(
                admitted=(),
                budget_suppressed_ids=tuple(trace.memory_id for trace in traces),
            )

        seen_fingerprints: set[str] = set()
        admitted: list[RecallTrace] = []
        duplicate_ids: list[str] = []
        budget_ids: list[str] = []

        for trace in traces:
            node = self.store.get(trace.memory_id)
            fingerprint = self._fingerprint(node.text)

            if fingerprint in seen_fingerprints:
                duplicate_ids.append(trace.memory_id)
                continue

            seen_fingerprints.add(fingerprint)

            if len(admitted) < limit:
                admitted.append(trace)
            else:
                budget_ids.append(trace.memory_id)

        return AdmissionResult(
            admitted=tuple(admitted),
            duplicate_suppressed_ids=tuple(duplicate_ids),
            budget_suppressed_ids=tuple(budget_ids),
        )

    @staticmethod
    def _fingerprint(text: str) -> str:
        return _SPACE_RE.sub(" ", text.strip().casefold())
