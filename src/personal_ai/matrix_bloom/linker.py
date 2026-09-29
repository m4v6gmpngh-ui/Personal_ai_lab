from __future__ import annotations

from personal_ai.matrix_bloom.graph import MemoryGraph
from personal_ai.memory.node import MemoryNode
from personal_ai.memory.store import MemoryStore


def shared_context_weight(left: MemoryNode, right: MemoryNode) -> float:
    """Create a simple explainable link weight from shared topics/entities."""

    shared_topics = left.topics & right.topics
    shared_entities = left.entities & right.entities

    topic_space = left.topics | right.topics
    entity_space = left.entities | right.entities

    topic_score = len(shared_topics) / len(topic_space) if topic_space else 0.0
    entity_score = len(shared_entities) / len(entity_space) if entity_space else 0.0

    weight = 0.65 * topic_score + 0.35 * entity_score
    return max(0.0, min(1.0, weight))


def auto_link(store: MemoryStore, graph: MemoryGraph, *, minimum_weight: float = 0.15) -> None:
    nodes = store.all()
    for index, left in enumerate(nodes):
        for right in nodes[index + 1 :]:
            weight = shared_context_weight(left, right)
            if weight >= minimum_weight:
                graph.connect(left.id, right.id, weight)
