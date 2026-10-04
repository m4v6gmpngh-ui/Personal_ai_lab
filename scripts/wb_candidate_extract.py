#!/usr/bin/env python3
"""Optional local candidate extractor for WB learning packets.

Uses local Ollama/Qwen to propose grounded craft observations.
Never writes to Notion, never changes technique status, and never writes a project Bloom.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path

DEFAULT_OLLAMA = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3.5:4b"
CLASSES = ["rhythm", "dialogue", "description", "structure", "language", "meaning", "mechanics"]

PROMPT = """You are studying writing craft for a cross-author learning system.
Analyze mechanisms, not plot and not imitation. Do not reproduce the author's style.

Return JSON only:
{
  "summary": "one sentence about what the passage does as writing",
  "techniques": [
    {
      "label": "2-6 word mechanism name",
      "class": "rhythm|dialogue|description|structure|language|meaning|mechanics",
      "evidence": "an EXACT quote from the packet, at most two sentences",
      "effect": "plain statement of what the move does for the reader here",
      "mechanism": "how the move works, generalized beyond this author"
    }
  ]
}

Rules:
- Exact evidence is mandatory. If you cannot quote it exactly, omit the technique.
- Never give more than two sentences of evidence for a technique.
- Never treat one author's habit as a universal rule.
- Do not imitate, continue, or transform the passage.
- Prefer syntax, punctuation, paragraphing, format, structure, rhythm, dialogue,
  description, language, meaning, and mechanics observations that can be tested across authors.
"""

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")).strip().lower()

def sentence_count(s: str) -> int:
    return len([x for x in re.split(r"(?<=[.!?])\s+", s.strip()) if x.strip()])

def grounded_quote(evidence: str, excerpt: str) -> bool:
    q = norm(evidence)
    return len(q.split()) >= 3 and q in norm(excerpt) and sentence_count(evidence) <= 2

def ollama_chat(excerpt: str, *, model: str, ollama: str) -> dict:
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": excerpt},
        ],
    }
    req = urllib.request.Request(
        ollama.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        outer = json.loads(resp.read().decode("utf-8"))
    return json.loads(outer["message"]["content"])

def process_packet(packet_path: Path, *, model: str, ollama: str) -> dict:
    excerpt = packet_path.read_text(encoding="utf-8").strip()
    result = ollama_chat(excerpt, model=model, ollama=ollama)
    kept, dropped = [], []
    for raw in result.get("techniques", []):
        label = str(raw.get("label", "")).strip().lower()
        cls = str(raw.get("class", "")).strip().lower()
        evidence = str(raw.get("evidence", "")).strip()
        if not label or cls not in CLASSES or not grounded_quote(evidence, excerpt):
            dropped.append({"label": label or "(blank)", "reason": "invalid class or ungrounded/too-long quote"})
            continue
        kept.append({
            "label": label,
            "class": cls,
            "evidence": evidence,
            "effect": str(raw.get("effect", "")).strip(),
            "mechanism": str(raw.get("mechanism", "")).strip(),
        })
    return {
        "packet": packet_path.name,
        "packet_sha256": hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
        "summary": str(result.get("summary", "")).strip(),
        "techniques": kept,
        "dropped": dropped,
        "authority": "candidate_only",
    }

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("packet_or_dir")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--ollama", default=DEFAULT_OLLAMA)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    source = Path(args.packet_or_dir)
    files = sorted(source.glob("*.txt")) if source.is_dir() else [source]
    rows = [process_packet(path, model=args.model, ollama=args.ollama) for path in files]
    Path(args.output).write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} candidate packet result(s) to {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
