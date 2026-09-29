# Personal AI Lab

A local-first experimental personal AI system focused on explainable memory, context, and recall.

## Current milestone

The first working spine is:

> Current event → detect contextual facets → activate secondary matrices → trace weighted memory relationships → expand relevant blooms → suppress irrelevant memories → return a compact ranked context packet → LLM.

The memory engine does not need to generate the answer itself. Its job is to prepare the smallest useful continuity packet for the model that does.

## Architecture

- `src/personal_ai/core/` — incoming events and contextual facets
- `src/personal_ai/memory/` — memory nodes, storage, scoring, and recall
- `src/personal_ai/matrix_bloom/` — weighted relationships, spreading activation, adaptive learning, and secondary matrices
- `src/personal_ai/context_engine.py` — middleware that produces ranked context packets for an LLM
- `src/personal_ai/simulation/` — repeatable scenarios for testing behavior
- `tests/` — deterministic tests for memory, tracing, adaptation, and context assembly

## Primary graph vs. secondary matrices

The primary graph stores weighted relationships between memories. It changes from explicit feedback such as useful/unhelpful recall or deliberate co-activation.

The secondary matrices index memories by four context families:

- **concept** — topics and entities
- **person** — people involved
- **emotion** — emotional context
- **goal** — active goals

A recalled memory is rehearsed in these secondary matrices. Repeated recall makes the memory easier to reach in matching future contexts, but does **not** increase its factual confidence. This keeps accessibility and truth separate.

Secondary reinforcement is bounded and saturating, and it can decay slowly over time.

## Context packet

A context packet contains:

1. the current event,
2. activated matrix keys,
3. a ranked set of relevant memories,
4. each memory's score, confidence, and recall path,
5. the number of memories suppressed from the packet.

That packet is intentionally small so a downstream LLM receives continuity without being flooded by the entire memory store.

## Design principles

1. Local-first.
2. Explainable recall.
3. Small testable components.
4. No opaque "magic" state.
5. Memory importance and relationships can change over time.
6. Build the trunk before sensors and interface layers.
7. Recall may strengthen accessibility, but repetition must not silently become factual confidence.
8. Retrieval must suppress irrelevant context instead of merely collecting more context.

## Run the demo

```bash
PYTHONPATH=src python -m personal_ai.simulation.scenario_runner
```

## Run tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Status

The v0.1 memory spine now includes structured event facets, weighted relationship tracing, adaptive primary links, recall-sensitive secondary matrices, suppression, and compact context-packet generation.
