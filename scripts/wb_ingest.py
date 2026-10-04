#!/usr/bin/env python3
"""Prepare writing sources for the PAIOS Writing Bloom (WB-00).

Deterministic acquisition-side work only:
- rights/storage policy checks
- WB-01-compatible measurements
- bounded learning packets
- provenance/hashes and Notion-ready handoff data

Never writes to Notion and never writes to a project Bloom, personal/global Bloom, or any other domain Bloom.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics as st
import sys
from pathlib import Path
from typing import Iterable

PUBLIC_RIGHTS = {"public_domain", "public_domain_us"}
LOCAL_ONLY_RIGHTS = {"owned", "licensed", "excerpt"}
ALL_RIGHTS = PUBLIC_RIGHTS | LOCAL_ONLY_RIGHTS
TRACKED_OUTPUT_ROOT = Path("data/wb_ingest")
LOCAL_OUTPUT_ROOT = Path(".local_wb")
QUOTE_RE = re.compile(r'“([^”]*)”|"([^"]*)"')
WORD_RE = re.compile(r"\b[\w’'-]+\b", re.UNICODE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;:—])\s+|—")

def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()

def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def strip_heading(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].lstrip().startswith("#"):
        lines = lines[1:]
    return "\n".join(lines).strip()

def paragraphs(text: str) -> list[str]:
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    return [b for b in blocks if not b.lstrip().startswith("#")]

def dialogue_word_count(text: str) -> int:
    total = 0
    for m in QUOTE_RE.finditer(text):
        q = m.group(1) if m.group(1) is not None else m.group(2)
        total += word_count(q or "")
    return total

def remove_dialogue(text: str) -> str:
    return QUOTE_RE.sub("", text)

def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]

def profile(text: str) -> dict:
    """WB-01-compatible source profile."""
    paras = paragraphs(text)
    narr_paras = []
    for p in paras:
        n = normalize_space(remove_dialogue(p))
        if word_count(n):
            narr_paras.append(n)
    sents = [s for p in narr_paras for s in split_sentences(p)]
    lens = [word_count(s) for s in sents if word_count(s)]
    total_words = word_count(text)
    narr_words = sum(word_count(p) for p in narr_paras)
    dialog_words = dialogue_word_count(text)

    def pct(n, d):
        return round(100 * n / d, 1) if d else 0.0

    return {
        "one_sentence_para": pct(sum(len(split_sentences(p)) == 1 for p in narr_paras), len(narr_paras)),
        "short_sent": pct(sum(length <= 6 for length in lens), len(lens)),
        "mean": round(st.mean(lens), 1) if lens else 0.0,
        "sd": round(st.pstdev(lens), 1) if len(lens) > 1 else 0.0,
        "dialogue_share": pct(dialog_words, total_words),
        "semicolons_1k": round(1000 * sum(p.count(";") for p in narr_paras) / narr_words, 2) if narr_words else 0.0,
        "emdash_1k": round(1000 * sum(p.count("—") for p in narr_paras) / narr_words, 2) if narr_words else 0.0,
        "words": total_words,
        "narration_words": narr_words,
        "dialogue_words": dialog_words,
        "narration_paragraphs": len(narr_paras),
        "narration_sentences": len(lens),
    }

def safe_slug(value: str) -> str:
    value = normalize_space(value).lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value).strip("-")
    return value or "source"

def infer_rights(metadata: dict) -> str:
    explicit = normalize_space(str(metadata.get("rights", ""))).lower()
    if explicit:
        if explicit not in ALL_RIGHTS:
            raise ValueError(f"unsupported rights value: {explicit}")
        return explicit
    if metadata.get("source_type") == "standard_ebooks":
        return "public_domain_us"
    raise ValueError("rights must be explicit for non-Standard-Ebooks sources")

def assert_storage_policy(rights: str, storage: str) -> None:
    if rights not in ALL_RIGHTS:
        raise ValueError(f"unsupported rights value: {rights}")
    storage = normalize_space(storage).lower()
    if storage not in {"tracked", "local_private"}:
        raise ValueError("storage must be tracked or local_private")
    if rights in LOCAL_ONLY_RIGHTS and storage == "tracked":
        raise ValueError(
            f"{rights} material may not be stored in the public tracked corpus; "
            "use local_private analysis instead"
        )

def packetize_paragraphs(paras: list[str], *, min_words: int = 700,
                         target_words: int = 1000, max_words: int = 1500) -> list[list[str]]:
    if not (0 < min_words <= target_words <= max_words):
        raise ValueError("packet limits must satisfy 0 < min <= target <= max")
    packets, current, current_words = [], [], 0
    for p in paras:
        pw = word_count(p)
        if not pw:
            continue
        if current and current_words >= min_words and current_words + pw > target_words:
            packets.append(current)
            current, current_words = [], 0
        if current and current_words + pw > max_words:
            packets.append(current)
            current, current_words = [], 0
        current.append(p)
        current_words += pw
        if current_words >= max_words:
            packets.append(current)
            current, current_words = [], 0
    if current:
        if packets and current_words < min_words:
            merged_words = sum(word_count(x) for x in packets[-1]) + current_words
            if merged_words <= max_words:
                packets[-1].extend(current)
            else:
                packets.append(current)
        else:
            packets.append(current)
    return packets

def load_book(book_dir: Path) -> tuple[dict, list[dict], str]:
    meta_path = book_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing metadata.json in {book_dir}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    request_path = book_dir / "request.json"
    if request_path.exists():
        request_data = json.loads(request_path.read_text(encoding="utf-8"))
        metadata = {**request_data, **metadata}
    chapter_dir = book_dir / "chapters"
    chapter_paths = sorted(chapter_dir.glob("*.md"))
    if not chapter_paths:
        raise FileNotFoundError(f"no chapter markdown files in {chapter_dir}")
    chapters, all_parts = [], []
    for i, path in enumerate(chapter_paths, 1):
        raw = path.read_text(encoding="utf-8")
        body = strip_heading(raw)
        title = raw.splitlines()[0].lstrip("# ").strip() if raw.splitlines() else f"Chapter {i}"
        chapters.append({"index": i, "title": title, "path": path, "text": body})
        all_parts.append(body)
    return metadata, chapters, "\n\n".join(all_parts)

def build_public_handoff(book_dir: Path, out_root: Path) -> Path:
    metadata, chapters, full_text = load_book(book_dir)
    rights = infer_rights(metadata)
    assert_storage_policy(rights, "tracked")
    title = metadata.get("title") or book_dir.name
    author = metadata.get("author") or "unknown"
    genre = metadata.get("genre") or []
    if isinstance(genre, str):
        genre = [genre]
    era = metadata.get("era") or ""
    pov = metadata.get("pov") or "other"
    output_slug = metadata.get("output_slug") or "/".join(book_dir.parts[-2:])
    out_dir = out_root.joinpath(*[safe_slug(p) for p in str(output_slug).split("/") if p])
    packets_dir = out_dir / "packets"
    packets_dir.mkdir(parents=True, exist_ok=True)
    source_profile = profile(full_text)
    packet_rows, packet_number = [], 0
    for chapter in chapters:
        for local_index, pset in enumerate(packetize_paragraphs(paragraphs(chapter["text"])), 1):
            packet_number += 1
            packet_text = "\n\n".join(pset).strip()
            packet_id = f"P{packet_number:04d}"
            packet_path = packets_dir / f"{packet_id}.txt"
            packet_path.write_text(packet_text + "\n", encoding="utf-8")
            packet_rows.append({
                "packet_id": packet_id,
                "chapter_index": chapter["index"],
                "chapter_title": chapter["title"],
                "chapter_packet": local_index,
                "words": word_count(packet_text),
                "sha256": sha256_text(packet_text),
                "path": str(packet_path.as_posix()),
            })
    handoff = {
        "schema": "wb-ingest-1",
        "authority": {
            "target": "WB-00 Writing Bloom",
            "writing_bloom_target": True,
            "project_bloom_write": False,
            "personal_global_bloom_write": False,
            "other_domain_bloom_write": False,
            "notion_write": False,
            "purpose": "measurement and evidence preparation only",
        },
        "rights": {
            "classification": rights,
            "storage": "tracked",
            "full_text_retained": True,
            "rule": "tracked full text is permitted only for configured public-domain sources",
        },
        "source": {
            "title": title, "author": author, "genre": genre, "era": era, "pov": pov,
            "obtained": "public domain", "source_url": metadata.get("source_url", ""),
            "source_sha256": metadata.get("source_sha256", ""),
            "source_type": metadata.get("source_type", ""),
            "library_path": str(book_dir.as_posix()),
        },
        "profile": source_profile,
        "notion_profile_fields": {
            "Profile": f"{title} — source profile", "Scope": "source", "Genre": genre,
            "Sample Words": source_profile["words"],
            "Dialogue Share %": str(source_profile["dialogue_share"]),
            "Mean Sentence Length": str(source_profile["mean"]),
            "Sentence Length Spread (SD)": str(source_profile["sd"]),
            "Short Sentence % (<=6 words, narration)": str(source_profile["short_sent"]),
            "One-Sentence Paragraph % (narration)": str(source_profile["one_sentence_para"]),
            "Semicolons per 1k": str(source_profile["semicolons_1k"]),
            "Em-dashes per 1k": str(source_profile["emdash_1k"]),
            "Notes": "Deterministic WB-01-compatible measurement; source relation added during WB learning import.",
        },
        "packets": packet_rows,
        "packet_policy": {
            "min_words": 700, "target_words": 1000, "max_words": 1500,
            "use": "candidate extraction only; Bloom excerpts remain <=2 sentences per technique per source",
        },
    }
    (out_dir / "handoff.json").write_text(json.dumps(handoff, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_dir

def prepare_local_file(input_path: Path, *, title: str, author: str, rights: str,
                       genre: list[str], era: str, pov: str, out_root: Path) -> Path:
    assert_storage_policy(rights, "local_private")
    text = input_path.read_text(encoding="utf-8")
    out_dir = out_root / safe_slug(author) / safe_slug(title)
    packets_dir = out_dir / "packets"
    packets_dir.mkdir(parents=True, exist_ok=True)
    prof, packet_rows = profile(text), []
    for i, pset in enumerate(packetize_paragraphs(paragraphs(text)), 1):
        packet_text = "\n\n".join(pset).strip()
        pid = f"P{i:04d}"
        path = packets_dir / f"{pid}.txt"
        path.write_text(packet_text + "\n", encoding="utf-8")
        packet_rows.append({"packet_id": pid, "words": word_count(packet_text),
                            "sha256": sha256_text(packet_text), "path": str(path.as_posix())})
    handoff = {
        "schema": "wb-ingest-1",
        "authority": {"target": "WB-00 Writing Bloom", "writing_bloom_target": True,
                      "project_bloom_write": False, "personal_global_bloom_write": False,
                      "other_domain_bloom_write": False, "notion_write": False,
                      "purpose": "local/private measurement and candidate preparation only"},
        "rights": {"classification": rights, "storage": "local_private",
                   "full_text_retained_in_git": False,
                   "rule": "owned/licensed/excerpt material must stay outside tracked public corpus"},
        "source": {"title": title, "author": author, "genre": genre, "era": era, "pov": pov,
                   "obtained": "owned" if rights in {"owned", "licensed"} else "excerpt",
                   "local_source_sha256": sha256_text(text)},
        "profile": prof, "packets": packet_rows,
        "packet_policy": {"min_words": 700, "target_words": 1000, "max_words": 1500,
                          "use": "private candidate extraction only; only brief grounded evidence may later enter WB-00"},
    }
    (out_dir / "handoff.json").write_text(json.dumps(handoff, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_dir

def discover_books(library_root: Path) -> Iterable[Path]:
    for meta in sorted(library_root.rglob("metadata.json")):
        yield meta.parent

def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare sources for WB-00 learning.")
    sub = parser.add_subparsers(dest="command", required=True)
    pub = sub.add_parser("public-library")
    pub.add_argument("--library-root", default="data/library")
    pub.add_argument("--output", default=str(TRACKED_OUTPUT_ROOT))
    local = sub.add_parser("local-file")
    local.add_argument("input")
    local.add_argument("--title", required=True)
    local.add_argument("--author", required=True)
    local.add_argument("--rights", choices=sorted(LOCAL_ONLY_RIGHTS), required=True)
    local.add_argument("--genre", action="append", default=[])
    local.add_argument("--era", default="")
    local.add_argument("--pov", default="other")
    local.add_argument("--output", default=str(LOCAL_OUTPUT_ROOT))
    args = parser.parse_args()
    try:
        if args.command == "public-library":
            count = 0
            for book_dir in discover_books(Path(args.library_root)):
                out = build_public_handoff(book_dir, Path(args.output))
                print(f"WB prepared: {book_dir} -> {out}")
                count += 1
            print(f"Prepared {count} public-domain source(s).")
            return 0
        out = prepare_local_file(Path(args.input), title=args.title, author=args.author,
                                 rights=args.rights, genre=args.genre, era=args.era,
                                 pov=args.pov, out_root=Path(args.output))
        print(f"WB local/private prepared: {out}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
