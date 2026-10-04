# Writing Bloom ingest v1

This is the acquisition and measurement layer for **WB-00 — Writing Bloom**.

It is intentionally separate from every project Bloom.

## Authority boundary

GitHub may:
- fetch approved public-domain sources;
- normalize and hash source text;
- compute WB-01-compatible measurements;
- split material into bounded learning packets;
- optionally run a local candidate extractor.

GitHub may **not**:
- write or update a project Bloom;
- promote a Writing Bloom technique;
- change Author Count, Status, or Success Weight;
- decide that a project lesson is promoted;
- treat one author's voice as a general writing rule.

Notion WB-00 remains the authoritative learned-skill memory.

## Rights and storage policy

The repository is public, so tracked storage is deliberately conservative.

| Rights | Full text in tracked GitHub corpus | Learning path |
| --- | --- | --- |
| public_domain / public_domain_us | allowed | GitHub Action + WB packets |
| owned | refused | local/private only |
| licensed | refused unless license explicitly permits redistribution | local/private by default |
| excerpt | refused as tracked corpus | local/private analysis; Bloom evidence remains brief |

The system does not bypass paywalls, authentication, DRM, anti-bot systems, or access controls.

For Standard Ebooks, the current adapter is limited to HTTPS pages on `standardebooks.org`. Standard Ebooks states that its ebook files are believed to be free of U.S. copyright restrictions and that its own edition work is dedicated to the public domain. The pipeline records this as `public_domain_us`, not as a worldwide copyright determination.

## Public-domain pipeline

```text
approved source URL
  -> scripts/web_ingest.py
  -> data/library/<author>/<book>/
  -> scripts/wb_ingest.py public-library
  -> data/wb_ingest/<author>/<book>/
       handoff.json
       packets/P0001.txt ...
  -> WB learning lane
  -> WB — Sources / Profiles / Observations / Techniques
```

`handoff.json` includes:
- source metadata and provenance;
- source SHA-256;
- WB-01-compatible profile measurements;
- packet IDs, locations, word counts, and hashes;
- explicit authority flags showing that no Notion or project Bloom write occurred.

## Measurements

The profiler follows WB-01 definitions:
- one-sentence narration paragraph percentage;
- narration sentences of six words or fewer;
- mean narration sentence length;
- sentence-length population SD;
- dialogue word share;
- semicolons per 1,000 narration words;
- em dashes per 1,000 narration words;
- sample word count.

Semicolons, colons, and em dashes count as sentence breaks.

## Learning packets

Packets are grouped on paragraph boundaries:
- minimum target: 700 words;
- normal target: about 1,000 words;
- maximum target: 1,500 words.

Packets are **working evidence windows**, not Bloom entries. WB-01 still limits stored Bloom evidence to at most two sentences, one excerpt per technique per source.

## Optional local candidate extraction

`scripts/wb_candidate_extract.py` can send packets to a local Ollama model (default `qwen3.5:4b`).

It returns candidate mechanisms with:
- label;
- class;
- exact evidence;
- effect;
- generalized mechanism.

Every evidence quote is checked against the packet. Ungrounded evidence and evidence longer than two sentences is dropped.

Candidate output has no authority. The WB learning lane still decides whether an observation:
- confirms;
- varies;
- contradicts;
- or proposes a new technique.

The three-author rule remains in WB-00.

## Local/private owned material

Do not add the source to `data/library`.

Example:

```bash
python scripts/wb_ingest.py local-file /private/path/book.txt \
  --title "Book Title" \
  --author "Author Name" \
  --rights owned \
  --genre literary \
  --pov "close third"
```

Output defaults to `.local_wb/`, which is gitignored.

This allows local measurement and candidate extraction without publishing or committing the copyrighted source.

## Design target

The reusable pattern is:

```text
source
 -> acquisition
 -> deterministic measurement
 -> bounded evidence packets
 -> grounded candidate extraction
 -> domain-Bloom consolidation
 -> real-world/project application
 -> reviewed outcome
 -> stronger or weaker domain knowledge
```

Writing is the first domain. If this proves reliable, the same outer pattern can later support other domain Blooms while replacing the domain-specific measurement and extraction logic.
