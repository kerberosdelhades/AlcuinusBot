# AlcuinusBot — Roadmap

## Current state

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | **Ingestion** — extract messages from source Telegram channel | ✅ Done |
| 1 | **Anchor detection** — identify messages containing links | ✅ Done |
| 2 | **Association** — link subsequent opinions/reactions to each anchor | Pending |
| 3 | **Metadata** — fetch title + description per link | Pending |
| 4 | **Bundle clustering** — BERTopic over bundles (link + opinions) | Pending |
| 5 | **Output** — publish summaries to docs channel | Pending |

## Phase 0 — Ingestion ✅

- **Module**: `src/alcuinus/extraction.py`
- **Data**: 252 messages extracted to `data/channel_messages.json`
- **Date range**: 2022-09-15 → 2026-05-25
- **Key stats**: 71 messages with URLs, 207 forwarded, 43 with reply chains, 0 with extracted reactions (field present but null)
- **Tech**: pytopicgram crawler (vendored in `vendor/pytopicgram/`) over Telethon
- **Config**: `config/.env` (api_id, api_hash, source_channel, docs_channel)

### Extraction module API

```
run_extraction(days_back=0, output_dir="data") → path_to_json
```

`days_back=0` extracts all messages (from 2020-01-01 to now). Pass a positive int to limit.

### Known issues / future work

- **Reactions not populated**: pytopicgram's Telethon crawler may not request `MessageReactions`. Need to verify the Telethon client flags or add a post-processing step that fetches reactions separately.
- **Single channel only**: `run_extraction` hardcodes one channel. Multi-channel ingestion needs extending.

---

## Phase 1 — Anchor detection

**Goal**: Given `data/channel_messages.json`, identify every message that contains at least one URL. These are the "anchors" — the messages around which discussion clusters form.

**Input**: `data/channel_messages.json` (list of Telethon Message dicts)

**Output**: list of anchor records, each with:
- Message ID (anchor)
- Timestamp
- Sender ID / name
- List of extracted URLs (there may be multiple per message)
- Raw message text (for later context)

**Implementation**: `src/alcuinus/anchor_detection.py`
- `extract_urls(text)` → list of http/https URLs via `urlextract`
- `build_anchor(message_dict)` → anchor record or None
- `detect_anchors(messages)` → all anchors sorted by msg_id
- `run_anchor_detection()` → load JSON, detect, write `data/anchors.json`
- Uses `urlextract` (transitive dep from pytopicgram, zero new deps)

**Result**: 71 anchors found, 76 total URLs, spanning 2022-09-15 → 2026-05-25

**Tests**: `tests/test_anchor_detection.py` — 15 tests covering URL extraction, anchor building, empty input, real data round-trip

---

## Phase 2 — Association

**Goal**: For each anchor, determine which subsequent messages are reactions/opinions about that anchor.

**Problem**: Reactions aren't just the next message — they span dozens of messages and interleave with reactions to other links.

**Approach** (per SPECIFICATIONS.md):
- **Window**: messages after the anchor until the next anchor (or gap > 2h, or max 50 messages)
- **Signals**: reply chains (`reply_to`), mentions, keyword patterns
- **Output**: bundles — each bundle = (anchor, list of associated message IDs, metadata)

**Key decisions**:
1. Window strategy: try "until next anchor" first (simplest), add time-gap fallback if results are noisy
2. Overlap handling: a reply that quotes or mentions an earlier anchor explicitly should be assigned to that earlier anchor, even if a new anchor appeared in between
3. Reactions field (currently null): if we fix reaction extraction, emoji reactions become a strong association signal

---

## Phase 3 — Metadata

**Goal**: For each unique URL in the anchors, fetch the page title and meta description.

**Approach**: HTTP GET + BeautifulSoup (already in deps). Lightweight — no full content fetch, no JS rendering.

**Concerns**:
- Rate limits on some domains → add delays / caching
- Paywalled content (Medium, academic journals) → graceful degradation
- Non-HTML links (PDFs, images, GitHub repos) → special-case handling

**Output**: `data/link_metadata.json` mapping URL → {title, description, fetched_at, status}

---

## Phase 4 — Bundle clustering

**Goal**: Cluster the bundles (anchor metadata + associated opinions) to discover discussion topics.

**Tech**: BERTopic / HDBSCAN with multilingual embeddings (`intfloat/multilingual-e5-large`)

**Input per bundle**:
- Anchor link titles + descriptions
- Associated opinion message texts

**Output**:
- Cluster labels (topics)
- Per-cluster: bundles, keywords, size

**Key decision**: clustering is over *bundles*, not raw messages. This groups by "topics that generated discussion" rather than "mentioned in passing."

---

## Phase 5 — Output

**Goal**: Generate and post a structured summary to the docs channel.

**Summary format** (per SPECIFICATIONS.md):
- Top 5 topics (cluster labels + message counts)
- 3 emerging themes (new clusters)
- 5 most influential links (by semantic centrality within each cluster)
- 1 "connection" insight (e.g. "the MoE discussion intersected with NVIDIA's new expert routing paper")

**Tech**: Pyrogram or raw Bot API. The bot writes to the docs channel, never to the source channel.

**Config**: `docs_channel` from `config/.env`.

---

## Future (post-MVP)

Per SPECIFICATIONS.md §"Estrategias de análisis":
1. Coreference resolution
2. Temporal topic trajectories
3. Link-conversation bridge analysis
4. Citation & influence graph
5. Contradiction & consensus mapping
6. Personalized "you might have missed" digests

---

## Immediate next steps

| Priority | Task | Effort | Blocks |
|----------|------|--------|--------|
| P0 | Phase 1 — Anchor detection | Small | Phase 2 |
| P1 | Fix reactions extraction (verify Telethon flags) | Small | Phase 2 quality |
| P2 | Phase 2 — Association (basic window) | Medium | Phase 4 |
| P3 | Phase 3 — Link metadata | Small | Phase 4 |
| P4 | Phase 4 — Bundle clustering | Medium | Phase 5 |
| P5 | Phase 5 — Output | Medium | — |
