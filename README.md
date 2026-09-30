# Personal AI Lab

A local-first experimental personal AI system focused on explainable memory, context, and recall.

## Current milestone

The working spine is:

> Current event → detect contextual facets → activate secondary matrices → trace weighted memory relationships → expand relevant Blooms → suppress irrelevant memories → return a compact ranked context packet → model/agent.

The memory engine does not need to generate the answer itself. Its job is to prepare the smallest useful continuity packet for whichever model or agent sits behind it.

## Architecture

- `src/personal_ai/core/` — incoming events and contextual facets
- `src/personal_ai/memory/` — memory nodes, storage, scoring, and recall
- `src/personal_ai/matrix_bloom/` — weighted relationships, spreading activation, adaptive learning, and secondary matrices
- `src/personal_ai/context_engine.py` — middleware that produces ranked context packets
- `src/personal_ai/llm/` — replaceable adapters/extractors for model or agent backends
- `src/personal_ai/simulation/` — repeatable scenarios for testing behavior
- `tests/` — deterministic tests for memory, tracing, adaptation, context assembly, provider bridges, and persistence

## Primary graph vs. secondary matrices

The primary graph stores weighted relationships between memories. It can change from explicit feedback such as useful/unhelpful recall or deliberate co-activation.

The secondary matrices index memories by four context families:

- **concept** — topics and entities
- **person** — people involved
- **emotion** — emotional context
- **goal** — active goals

A recalled memory is rehearsed in these secondary matrices. Repeated recall makes the memory easier to reach in matching future contexts, but does **not** increase its factual confidence. Accessibility and truth remain separate.

Secondary reinforcement is bounded and saturating, and it can decay slowly over time.

## Transient working state

Matrix Bloom now has a short-lived working-state layer separate from durable memory. The first implementation carries **emotional state** across a few following turns so recall can remain context-sensitive even when the next sentence does not repeat the emotion.

Example:

```text
I'm feeling afraid right now.
What does the blue lake cabin remind me of?
```

The second turn can inherit `fear` as working context and route toward the fear-linked cabin memory. Explicit new emotion replaces older working emotion, and the carried state expires after a bounded number of turns.

Working state affects retrieval and route reinforcement, but it is **not copied into durable memory** and is intentionally not persisted across process restarts.

### Emotional recall guardrail

A bare emotional-state update does not automatically retrieve autobiographical memories. For example, `I'm feeling afraid right now.` may update working state to `fear`, but Matrix Bloom suppresses memory recall unless there is also a non-emotional anchor such as a person, place, concept, or goal. The next anchored turn can still use the carried emotion to bias recall.

Explicit memory-search questions remain allowed, so the guardrail blocks passive spirals without preventing intentional reflection. Emotional words are also kept out of the concept channel when the local extractor already recognized them as emotion, avoiding double-counting the same affective cue.

## Persistent learning

Two local files are used by the live session:

- `data/live_memories.json` — stored memory nodes
- `data/live_memories_bloom_state.json` — learned graph and secondary-matrix accessibility state

Both are ignored by Git so personal memories and learned state are not committed to the public repository.\n\n## Memory capture behaviors\n\nMatrix Bloom now separates **retrieval** from **durable capture**. A turn can use recalled memories without automatically becoming a new long-term memory.\n\n- `MATRIX_BLOOM_MODE=training` stores every turn. Use this while inspecting retrieval and reinforcement behavior.\n- `MATRIX_BLOOM_MODE=normal` uses a transparent capture gate. Ordinary questions and conversational glue stay ephemeral; explicit `remember` instructions, durable relationship facts, preferences, commitments/decisions, recurring patterns, and sufficiently salient events can be stored.\n\nEvery capture decision is visible in debug mode, and stored memories include capture category/reason metadata. This is deliberately rule-based for now so behavior remains inspectable before a learned capture policy is introduced.

## Context packet

A context packet contains:

1. the current event,
2. activated matrix keys,
3. a ranked set of relevant memories,
4. each memory's score, confidence, and recall path,
5. the number of memories suppressed from the packet.

That packet is intentionally small so a downstream AI receives continuity without being flooded by the entire memory store.

## Design principles

1. Local-first.
2. Model-agnostic memory core.
3. Explainable recall.
4. Small testable components.
5. No opaque "magic" state.
6. Memory importance and relationships can change over time.
7. Recall may strengthen accessibility, but repetition must not silently become factual confidence.
8. Retrieval must suppress irrelevant context instead of merely collecting more context.
9. Contradictory memories may coexist; recall should reveal conflict rather than flatten it.
10. The active Bloom is transient; the learned graph and routing state persist.

## Run tests

Linux/macOS:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Windows PowerShell:

```powershell
$env:PYTHONPATH="src"
py -m unittest discover -s tests -v
```

## Live provider modes

The same live test can use different backends through `AI_PROVIDER`.

### OpenClaw workspace mode

This mode requires no OpenAI API credits. Matrix Bloom stays local, performs transparent offline cue extraction, and writes only the current compact Bloom into OpenClaw's existing workspace memory directory.

Windows PowerShell:

```powershell
$env:PYTHONPATH="src"
$env:AI_PROVIDER="openclaw"
$env:MATRIX_BLOOM_DEBUG="1"
py scripts\live_ai_test.py
```

By default the adapter uses:

```text
%USERPROFILE%\.openclaw\workspace
```

Override it if needed:

```powershell
$env:OPENCLAW_WORKSPACE="C:\Users\yourname\.openclaw\workspace"
```

The bridge writes:

```text
memory\matrix_bloom_context.md
memory\matrix_bloom_context.json
memory\MATRIX_BLOOM_README.md
```

`matrix_bloom_trace.md` is the human-readable provenance report: source memory ID, timestamp, matched cues, graph path, direct score, secondary-matrix score, recall history, confidence, and accessibility.\n\nOpenClaw can read those files without needing direct access to Matrix Bloom's internal memory store. The current bridge is intentionally file-based and does not assume an undocumented OpenClaw CLI or HTTP API. A direct transport can be layered on later without changing the memory engine.

### OpenAI test adapter

The OpenAI path remains available as an optional test adapter.

```powershell
$env:PYTHONPATH="src"
$env:AI_PROVIDER="openai"
$env:OPENAI_API_KEY="your-key"
$env:MATRIX_BLOOM_DEBUG="1"
py scripts\live_ai_test.py
```

API billing is separate from ChatGPT subscriptions.

## What the live test is trying to prove

The immediate experiment is not "can we make another chatbot?" It is:

- does the right past information become salient under the current context?
- do irrelevant memories stay suppressed?
- does repeated relevant recall improve accessibility without increasing factual confidence?
- does learned accessibility survive a restart?
- can the same memory core feed different model/agent backends?

The next deeper layer is to test contradictory traces, sensory/context cues, reinforcement loops, and explicit useful/unhelpful feedback before expanding into sensors or large-scale personal telemetry.
