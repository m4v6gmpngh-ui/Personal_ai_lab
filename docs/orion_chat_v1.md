# ORION-01 — Local conversation proof of concept

Status: **experimental; not independently run on the user's Windows machine**.

This branch adds a minimal terminal conversation with the existing Matrix Bloom
`LiveSession`. It deliberately leaves the existing Pocket Bloom deployment,
MainFrame production records, and Orion benchmark checkpoints untouched.

## What it does

- Accepts a user message and displays an actual locally generated model response.
- Uses Matrix Bloom's existing retrieval, capture policy, and state persistence.
- Uses a local Ollama model over `127.0.0.1:11434` only.
- Keeps new memories in `data/orion_sandbox/`, separate from live memories.
- Writes an append-only JSONL execution record with hashes, selected memory IDs,
  elapsed time, and status. **Conversation text is not copied into the execution log.**
- Supports `/trace` to view the last run and `/quit` to exit.

It **does not** execute REA tools, autonomously select skills, validate answers,
or consolidate Skill Blooms yet. The trace records those features as absent.

## Windows test (PowerShell)

1. Install and run Ollama locally if it is not already running.
2. The default chat model is `qwen3.5:4b`, already present on the development PC.
   The separate `qwen3-embedding:0.6b` model is an embedding model, not used for chat.
   To choose another installed chat model, set `$env:ORION_MODEL="other-model-name"`.
3. Switch to the **experiment/orion-chat-v1** branch of Personal_ai_lab.
4. From the repository root:

```powershell
$env:PYTHONPATH="src"
py scripts\orion_chat.py
```

No model download is triggered automatically. If Ollama isn't available, the console reports a failed turn without falling back to a billed cloud model.

Enter a test message. For example:

```text
You: Remember that the red toolbox contains measuring instruments.
You: What did I tell you about the red toolbox?
You: /trace
You: /quit
```

## Testing

Run from the repository root:

```powershell
$env:PYTHONPATH="src"
py -m unittest discover -s tests -v
```

The new test uses a mocked Ollama endpoint. It does not need the model or network.

## Data and safety

Use synthetic messages for the initial experiment. Normal capture mode does
persist qualifying user messages in `data/orion_sandbox/memories.json`;
those memory files are intentionally gitignored. Don't treat memory recall as
independent evidence that the content is true.

The experimental console is local to the Windows computer; it isn't an
iPhone-accessible chat app yet. We should validate the local turn/output path
before adding an authenticated mobile interface.

## Next experiment

Add a structured read-only REA tool-selection step behind a permission gate,
plus independent checks, only after confirming basic conversation works.
