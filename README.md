# Personal AI Lab

A local-first experimental personal AI system focused on explainable memory, context, and recall.

## v0.1 milestone

The first milestone is intentionally narrow:

> Receive an event, store it as structured memory, connect it to related memories, retrieve relevant memories later, and explain the recall path.

This repository will grow in layers. Audio, vision, wearable data, larger-model integration, and richer behavior come after the memory loop is testable.

## Initial architecture

- `src/personal_ai/core/` — incoming events and context
- `src/personal_ai/memory/` — memory nodes, storage, scoring, and recall
- `src/personal_ai/matrix_bloom/` — weighted relationships and spreading activation
- `src/personal_ai/simulation/` — repeatable scenarios for testing behavior
- `tests/` — deterministic tests for memory and tracing

## Design principles

1. Local-first.
2. Explainable recall.
3. Small testable components.
4. No opaque "magic" state.
5. Memory importance and relationships can change over time.\n7. Adaptive learning only strengthens or weakens links from explicit signals; recall alone does not self-reinforce.
6. Build the trunk before sensors and interface layers.

## Run the demo

```bash
PYTHONPATH=src python -m personal_ai.simulation.scenario_runner
```

## Run tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Status

v0.1 foundation in progress. Adaptive Matrix Bloom learning is now implemented with reinforcement, penalty, co-activation, decay, and tests.
