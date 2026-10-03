# Web ingestion v1

This is a narrow public-source ingestion path for PAIOS experiments.

## What it does

1. A JSON request is added under `ingest_requests/`.
2. GitHub Actions runs `.github/workflows/web-ingest.yml` on an internet-connected runner.
3. `scripts/web_ingest.py` fetches the allowlisted source.
4. XHTML chapters are normalized into Markdown.
5. The Action commits the normalized book back under `data/library/`.

The first version only permits HTTPS pages on Standard Ebooks. It does not bypass authentication, paywalls, robots controls, or anti-bot systems.

## Request format

```json
{
  "id": "standard-ebooks-circular-staircase",
  "url": "https://standardebooks.org/ebooks/mary-roberts-rinehart/the-circular-staircase/text/single-page",
  "source_type": "standard_ebooks",
  "title": "The Circular Staircase",
  "author": "Mary Roberts Rinehart",
  "output_slug": "mary-roberts-rinehart/the-circular-staircase"
}
```

## Output

```text
data/library/<author>/<book>/
  metadata.json
  request.json
  full-text.md
  chapters/
    001-....md
    002-....md
```

`metadata.json` records the source URL, retrieval timestamp, content type, SHA-256 hash, and chapter file list.

## ChatGPT workflow

Once this branch is merged, a future chat can create a new JSON request in `ingest_requests/`. That push triggers the Action. After GitHub commits the result, ChatGPT can read the normalized files through the GitHub connector without carrying the full book in conversation context.
