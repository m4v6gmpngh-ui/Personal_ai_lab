#!/usr/bin/env python3
"""Fetch allowlisted public XHTML/HTML sources and normalize them into a local library."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

MAX_BYTES = 20 * 1024 * 1024
ALLOWED_HOSTS = {"standardebooks.org", "www.standardebooks.org"}
EPUB_NS = "http://www.idpf.org/2007/ops"


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def text_content(element: ET.Element) -> str:
    return normalize_space("".join(element.itertext()))


def slugify(value: str) -> str:
    value = normalize_space(value).lower()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value or "section"


def safe_output_path(root: Path, output_slug: str) -> Path:
    parts = [p for p in output_slug.replace("\\", "/").split("/") if p]
    if not parts or any(p in {".", ".."} for p in parts):
        raise ValueError("output_slug must be a safe relative path")
    for part in parts:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", part):
            raise ValueError(f"unsafe output_slug segment: {part!r}")
    out = root.joinpath(*parts).resolve()
    root_resolved = root.resolve()
    if root_resolved not in out.parents and out != root_resolved:
        raise ValueError("output path escaped library root")
    return out


def fetch_url(url: str) -> tuple[bytes, str]:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError("only https URLs are allowed")
    host = (parsed.hostname or "").lower()
    if host not in ALLOWED_HOSTS:
        raise ValueError(f"host is not allowlisted: {host}")

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "PersonalAI-Lab-WebIngest/0.1",
            "Accept": "application/xhtml+xml,text/html;q=0.9,*/*;q=0.1",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        content_type = response.headers.get_content_type()
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > MAX_BYTES:
            raise ValueError("source exceeds 20 MiB limit")
        payload = response.read(MAX_BYTES + 1)
        if len(payload) > MAX_BYTES:
            raise ValueError("source exceeds 20 MiB limit")
        return payload, content_type


def document_title(root: ET.Element) -> str | None:
    for element in root.iter():
        if local_name(element.tag) == "title":
            title = text_content(element)
            if title:
                return re.sub(r"\s*\|\s*Standard Ebooks\s*$", "", title)
    return None


def document_author(root: ET.Element) -> str | None:
    for element in root.iter():
        if local_name(element.tag) != "meta":
            continue
        if element.attrib.get("name", "").lower() == "author":
            value = normalize_space(element.attrib.get("content", ""))
            if value:
                return value
    return None


def section_heading(section: ET.Element) -> str | None:
    for element in section.iter():
        if local_name(element.tag) in {"h1", "h2", "h3", "h4"}:
            heading = text_content(element)
            if heading:
                return heading
    return None


def render_section(section: ET.Element) -> str:
    paragraphs: list[str] = []
    for element in section.iter():
        if local_name(element.tag) == "p":
            paragraph = text_content(element)
            if paragraph:
                paragraphs.append(paragraph)
    return "\n\n".join(paragraphs)


def parse_xhtml(payload: bytes) -> tuple[ET.Element, list[tuple[str, str]]]:
    root = ET.fromstring(payload)
    chapters: list[tuple[str, str]] = []
    for section in root.iter():
        if local_name(section.tag) != "section":
            continue
        epub_type = section.attrib.get(f"{{{EPUB_NS}}}type", "")
        section_id = section.attrib.get("id", "")
        is_chapter = "chapter" in epub_type.split() or section_id.startswith("chapter-")
        if not is_chapter:
            continue
        body = render_section(section)
        if not body:
            continue
        title = section_heading(section) or section_id.replace("-", " ").title() or "Chapter"
        chapters.append((title, body))

    if not chapters:
        body = next((e for e in root.iter() if local_name(e.tag) == "body"), root)
        fallback = render_section(body)
        if fallback:
            chapters.append(("Full Text", fallback))

    return root, chapters


def write_book(request_data: dict, payload: bytes, content_type: str, output_root: Path) -> Path:
    root, chapters = parse_xhtml(payload)
    request_id = normalize_space(str(request_data.get("id", "")))
    output_slug = normalize_space(str(request_data.get("output_slug", request_id)))
    if not request_id or not output_slug:
        raise ValueError("request requires id and output_slug")

    title = normalize_space(str(request_data.get("title", ""))) or document_title(root) or request_id
    author = normalize_space(str(request_data.get("author", ""))) or document_author(root)

    out_dir = safe_output_path(output_root, output_slug)
    chapters_dir = out_dir / "chapters"
    chapters_dir.mkdir(parents=True, exist_ok=True)

    chapter_files: list[str] = []
    full_parts = [f"# {title}"]
    if author:
        full_parts.append(f"**Author:** {author}")

    for index, (chapter_title, chapter_body) in enumerate(chapters, 1):
        filename = f"{index:03d}-{slugify(chapter_title)[:80]}.md"
        rel = f"chapters/{filename}"
        chapter_files.append(rel)
        chapter_markdown = f"# {chapter_title}\n\n{chapter_body}\n"
        (chapters_dir / filename).write_text(chapter_markdown, encoding="utf-8")
        full_parts.append(f"# {chapter_title}\n\n{chapter_body}")

    (out_dir / "full-text.md").write_text("\n\n---\n\n".join(full_parts) + "\n", encoding="utf-8")

    metadata = {
        "request_id": request_id,
        "title": title,
        "author": author,
        "source_url": request_data["url"],
        "source_type": request_data.get("source_type", "standard_ebooks"),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "source_content_type": content_type,
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "chapter_count": len(chapters),
        "chapter_files": chapter_files,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "request.json").write_text(
        json.dumps(request_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out_dir


def process_request_file(request_file: Path, output_root: Path) -> Path:
    request_data = json.loads(request_file.read_text(encoding="utf-8"))
    url = normalize_space(str(request_data.get("url", "")))
    source_type = normalize_space(str(request_data.get("source_type", "standard_ebooks")))
    if source_type != "standard_ebooks":
        raise ValueError(f"unsupported source_type in v1: {source_type}")
    if not url:
        raise ValueError("request requires url")
    payload, content_type = fetch_url(url)
    return write_book(request_data, payload, content_type, output_root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", default="ingest_requests")
    parser.add_argument("--output", default="data/library")
    args = parser.parse_args()

    requests_dir = Path(args.requests)
    output_root = Path(args.output)
    request_files = sorted(requests_dir.glob("*.json")) if requests_dir.exists() else []
    if not request_files:
        print("No ingest request files found.")
        return 0

    failures = 0
    for request_file in request_files:
        try:
            out_dir = process_request_file(request_file, output_root)
            print(f"Ingested {request_file} -> {out_dir}")
        except Exception as exc:
            failures += 1
            print(f"ERROR {request_file}: {exc}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
