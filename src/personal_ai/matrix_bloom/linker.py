from __future__ import annotations

from personal_ai.matrix_bloom.graph import MemoryGraph
from personal_ai.memory.node import MemoryNode
from personal_ai.memory.store import MemoryStore


def _overlap(left: set[str], right: set[str]) -> float:
    space = left | right
    return len(left & right) / len(space) if space else 0.0


def shared_context_weight(left: MemoryNode, right: MemoryNode) -> float:
    """Explainable initial relationship weight from shared memory facets."""

    scores = {
        "topics": _overlap(left.topics, right.topics),
        "entities": _overlap(left.entities, right.entities),
        "people": _overlap(left.people, right.people),
        "emotions": _overlap(left.emotions, right.emotions),
        "goals": _overlap(left.goals, right.goals),
    }

    weight = (
        0.30 * scores["topics"]
        + 0.15 * scores["entities"]
        + 0.20 * scores["people"]
        + 0.10 * scores["emotions"]
        + 0.25 * scores["goals"]
    )
    return max(0.0, min(1.0, weight))


def link_new_memory(
    node: MemoryNode,
    existing_nodes: list[MemoryNode],
    graph: MemoryGraph,
    *,
    minimum_weight: float = 0.15,
) -> None:
    """Link one newly stored memory without resetting learned graph weights."""

    for other in existing_nodes:
        weight = shared_context_weight(node, other)
        if weight < minimum_weight:
            continue

        existing = graph.edge_weight(node.id, other.id)
        if existing == 0.0:
            graph.connect(node.id, other.id, weight)


def auto_link(store: MemoryStore, graph: MemoryGraph, *, minimum_weight: float = 0.15) -> None:
    nodes = store.all()
    for index, left in enumerate(nodes):
        for right in nodes[index + 1 :]:
            weight = shared_context_weight(left, right)
            if weight >= minimum_weight:
                graph.connect(left.id, right.id, weight)
