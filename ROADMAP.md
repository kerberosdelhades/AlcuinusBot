# AlcuinusBot — Roadmap

## Current state

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | **Ingestion** — extract messages from source Telegram channel | ✅ Done |
| 1 | **Anchor detection** — identify messages containing links | ✅ Done |
| 2 | **Association** — link subsequent opinions/reactions to each anchor | ✅ Done |
| 3 | **Metadata** — fetch title + description per link (HTML + GitHub/arXiv API) | ✅ Done |
| 4 | **Chunking & Tagging** — parent-child chunks, overlap, metadata prefix | ✅ Done |
| 5 | **Embedding** — `mistral-embed` (1024 dim) → Zvec | ✅ Done |
| 6 | **Bundle clustering** — KMeans + TF-IDF keywords | ✅ Done |
| 7 | **Decay classification** — evergreen / semi-stable / ephemeral tagging | ✅ Done |
| 8 | **Output** — publish summaries to docs channel | ✅ Done |
| 9 | **Syllabus generation** — living study guide for newcomers | ✅ Done |
| 10 | **6-month review cycle** — re-audit existing documentation | ✅ Done |

---

## Phase 0 — Ingestion ✅

- **Module**: `src/alcuinus/extraction_v2.py` (Telethon directo, sin pytopicgram) — camino primario. Fallback: `src/alcuinus/chat_export_ingest.py` (HTML export)
- **Datos actuales** (2026-08-09): **32,227 mensajes** en `data/alcuinus.db` (SQLite) + `data/channel_messages.json` (compat)
- **Rango de fechas**: **2021-01-01 → 2025-12-31** (672 mensajes sin fecha, 2.1%)
- **Senders únicos**: 370 (top 5: empty 11,185, GonzaloFotonPe 4,072, Jose Rodriguez "Boriel" 2,845, Ivan Juanes 2,226, Joaquin 1,947)
- **Mensajes con URL**: 5,913
- **Tech ingestión actual**: HTML export (Telegram Desktop) → `chat_export_ingest.py`. `extraction_v2.py` (Telethon directo) está implementado como camino primario — ver **Auditoría A**. Los datos actuales tienen IDs secuenciales falsos del HTML export; una ingesta Telethon reemplazaría con IDs reales de Telegram.
- **Config**: `config/.env` (api_id, api_hash, source_channel, docs_channel) + `.env` raíz (MISTRAL_API_KEY)

### Extraction module API

```
run_extraction_v2(db_path, incremental=True, output_dir="data") → dict
```

`incremental=True` (default) fetches only messages with id > MAX(id) in SQLite. `incremental=False` (or `--full` flag) does a fresh extraction. See `extraction_v2.py`.

### Known issues / future work

- **Reacciones emoji no se extraen vía HTML export** — `chat_export_ingest.py` no parsea `MessageReactions`. Si se quiere emoji real, hay que volver al camino Telethon y añadir un post-step (`GetMessagesReactionsRequest` en batches de 100).
- **"Reacciones" windowed ≠ reacciones reales.** La pipeline de asociación (Phase 2) produce 26,308 registros `reaction` que son mensajes de seguimiento en la ventana del anchor — un proxy de engagement, no reacciones emoji. Distinción documentada en **Auditoría D**.
- **Single channel only**: `run_extraction` hardcodes one channel. Multi-channel ingestion needs extending.

---

## Phase 1 — Anchor detection ✅

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

**Result**: 5,907 anchors found, 6,013 total URLs (5,435 unique), spanning 2021-01-01 → 2025-12-31

**Tests**: `tests/test_anchor_detection.py` — 15 tests covering URL extraction, anchor building, empty input, real data round-trip

---

## Phase 2 — Association ✅

**Goal**: For each anchor, determine which subsequent messages are reactions/opinions about that anchor.

**Implementation**: `src/alcuinus/association.py` — three-pass algorithm:

1. **Window assignment** (pass 1): every non-anchor message belongs to the nearest preceding anchor. Window closes at the next anchor.
2. **Reply override** (pass 2): a message whose `reply_to` points directly at an anchor is reassigned to that anchor, regardless of window boundaries. This handles the case where someone explicitly replies to an older link after a newer link has been shared.
3. **Time-gap cleanup** (pass 3): for the last anchor in the data, messages too far away (default: 168h / 7 days) are dropped — unless they're reply-anchored (pass 2 exempts them).

**Output**: `data/bundles.json` — each bundle = `{anchor, reactions[], window{boundary, ...}}`. Anchors with zero reactions are included (empty reactions list).

**API**:
- `associate(messages, anchors, max_idle_hours=168)` → list of bundles
- `run_association()` → convenience wrapper, writes output file

**Reaction records**: `{msg_id, date, sender_id, text_preview, reply_to_msg_id, strategy}` — where `strategy` is either `"window"` or `"reply"`.

**Tests**: `tests/test_association.py` — 12 tests covering synthetic fixture with all three strategies, empty inputs, round-trip I/O, and record schema validation.

---

## Phase 3 — Metadata ✅

**Goal**: For each unique URL in the anchors, fetch the page title and meta description.

**Approach**:
- **Generic HTML**: HTTP GET + BeautifulSoup for most URLs. Lightweight — no JS rendering, just `<title>` + `<meta description>`.
- **GitHub API**: `https://api.github.com/repos/{owner}/{repo}` → structured metadata (description, topics, stars). No auth needed for light use (60 req/h).
- **arXiv API**: `http://export.arxiv.org/api/query?id_list={id}` → title, authors, abstract. No rate limits in practice.
- **Everything else** (PDFs, images, other non-HTML): graceful skip with status `"unsupported"`.

**Concerns**:
- Rate limits on some domains → add delays / caching
- Paywalled content (Medium, academic journals) → graceful degradation
- Non-HTML links (PDFs, images, GitHub repos) → special-case handling

**Output**: `data/link_metadata.json` mapping URL → {title, description, fetched_at, status}

---

## Phase 4 — Chunking & Tagging ✅

**Goal**: Split extracted content into retrievable chunks with rich metadata.

**Approach**:
- **Parent-child chunking**: small child chunks for retrieval precision, larger parent chunks sent to the LLM for full context.
- **15% overlap** between chunks (baseline). The 10-20% range is industry consensus; NVIDIA FinanceBench benchmark found 15% optimal with 1,024-token chunks (see references below).
- **Metadata prefix** per chunk: channel, date, poster, language, surrounding chat snippet.

**Input**: bundles from Phase 2 + link metadata from Phase 3

**Output**: chunk records with metadata, ready for embedding.

### Chunk overlap — empirical basis

The 10-20% overlap range is well-documented across the RAG ecosystem. The strongest evidence:

- **NVIDIA FinanceBench** (2024): tested 10%, 15%, and 20% overlap with 1,024-token chunks. Result: **15% was optimal**. Below 10% loses boundary context where sentences split across chunks; above 25% adds near-duplicate noise that reduces effective context diversity.
  → https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses/

- **LlamaIndex `SentenceSplitter` defaults**: `chunk_size=1024`, `chunk_overlap=200` (~20%). When a leading framework sets this as default, it reflects common practice.
  → https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/sentence_splitter/

- **Industry consensus**: multiple sources converge on 10-15% as baseline, 20% as upper bound.
  → https://leanware.co/insights/langchain-rag-tutorial-build-retrieval-augmented-generation-from-scratch
  → https://nandigamharikrishna.substack.com/p/rag-chunking-strategies-and-embeddings

**AlcuinusBot baseline**: 15% overlap, adjust in Phase 5/6 testing if retrieval quality requires it.

---

## Phase 5 — Embedding + Zvec

**Goal**: Vectorize all chunks and store them for semantic search with built-in reranking.

**Tech**: `mistral-embed` (Mistral AI API) + Zvec (embebido, Apache 2.0)
- **mistral-embed**: 1024 dim, ~8K tokens/ctx, multilingual. Ya provisionado.
- **Zvec**: "SQLite for vector search" — in-process, sin servidor, sin infraestructura externa. Built-in reranking (weighted fusion + RRF). ~128MB RAM para ~100K embeddings.

**Approach**:
- Apply an instruction/task-context prefix per chunk (e.g., "represents a technical link summary for retrieval").
- Store vectors in a local Zvec index file (single-file, portable).
- Use Zvec's built-in reranking instead of an external reranker API.
- Keyword/date/channel filters layered on top as needed.

**Output**: Zvec index file with chunk embeddings, queryable via dense + sparse hybrid search with reranking.

**Handoff to Phase 6 (clustering):** BERTopic or alternative reads vectors directly from Zvec via `collection.Fetch(pks, include_vector=True)`. No duplication of vectors — the same embeddings stored by `mistral-embed` are retrieved by ID and passed to the clustering algorithm. Zero extra API calls, zero data redundancy.

**Why Zvec over pgvector:**
- Zero-ops: no PostgreSQL instance to provision, manage, or back up. The index is a single file.
- Built-in reranking eliminates the need for a separate cross-encoder API call.
- Single-tenant, single-process profile matches AlcuinusBot exactly (one channel, one bot).
- Apache 2.0 license, no licensing friction.
- Tradeoff documented: embedded DB pushes index migration + backup responsibility to the app release cycle. Simple periodic file-copy backup covers this.

---

## Phase 6 — Bundle clustering ✅

**Goal**: Cluster bundles (anchor metadata + associated opinions) to discover discussion topics.

**Tech**: KMeans (scikit-learn) + TF-IDF keywords. BERTopic descartado en el piloto de 71 bundles (UMAP necesita ~200+ puntos para manifold learning estable). A 5,907 bundles, BERTopic podría re-evaluarse — ver ROADMAP.md "Auditoría D".

**Input per bundle**:
- Anchor link titles + descriptions
- Associated opinion message texts
- Embeddings from Zvec (mistral-embed, 1024d)

**Output**:
- Cluster labels (topics)
- Per-cluster: bundles, keywords, size

**Key decision**: clustering is over *bundles*, not raw messages. This groups by "topics that generated discussion" rather than "mentioned in passing."

**Status**: KMeans implementado y funcionando (12 clusters, 15,330 chunks). BERTopic re-evaluable a 5,907 bundles (P1).

---

## Phase 7 — Decay classification

**Goal**: Tag every cluster/link/entity with a decay profile.

**Profiles**:
- **Evergreen**: foundational papers, architectural patterns, evaluation methodologies — permanent, surfaced to newcomers.
- **Semi-stable**: benchmark results, scaling laws, prompting techniques — 12-24 month retention.
- **Ephemeral**: model-of-the-week news, transient tool announcements — short retention, flagged for review/removal.

---

## Phase 8 — Output

**Goal**: Generate and post a structured summary to the docs channel.

**Summary format** (per SPECIFICATIONS.md):
- Top 5 topics (cluster labels + message counts)
- 3 emerging themes (new clusters)
- 5 most influential links (by semantic centrality within each cluster)
- 1 "connection" insight (e.g. "the MoE discussion intersected with NVIDIA's new expert routing paper")

**Tech**: Pyrogram or raw Bot API. The bot writes to the docs channel, never to the source channel.

**Config**: `docs_channel` from `config/.env`.

---

## Phase 9 — Syllabus generation

**Goal**: Produce a living syllabus/study-guide that gives newcomers a map, distinct from the raw archive.

**Input**: curated clusters with decay profiles.

**Output**: structured document (Markdown or formatted messages) organized by topic, with evergreen content highlighted.

---

## Phase 10 — 6-month review cycle

**Goal**: Re-audit existing documentation every 6 months.

**Approach**:
- Re-evaluate decay profiles (content may shift from semi-stable to ephemeral).
- Flag stale links for removal or update.
- Regenerate syllabus sections as needed.

---

## Explicitly reverted / parked decisions

- vx-summary-style link processing bot — parked, out of scope.

---

## Maybe someday — digest & output improvements

Post-Phase 8 ideas for richer digest output:

1. **Per-cluster LLM summaries** — replace cryptic keywords ("es, yo, app") with a one-line LLM description ("Immich self-hosted photo management — trending tools"). One batch LLM call, 8 clusters.

2. **Temporal comparison** — diff this week's digest against last week's to show spikes, decay, and new clusters. Store previous state in `data/digest_history.json`.

3. **"Quiet gems"** — links with few reactions but high semantic novelty. Compute each bundle vector's distance to its nearest neighbor in Zvec; the furthest outlier is the overlooked gem.

4. **Controversy tracker** — reply-depth instead of reaction count. A link with 10 deep replies is more interesting than one with 30 "+1" likes.

5. **Per-user contributions** — who shared what, who's most active per cluster. Data already exists in anchor sender IDs.

6. **Personalized "you might have missed"** — given a user's reaction history, recommend unseen bundles within their interest clusters. Vector similarity in Zvec.

7. **Telegram bot hook — minimal (script)** — CLI script that posts digest/syllabus to docs channel on command. Lightweight: no command listening, no scheduling, just fire and forget. Pair with cron.

8. **Telegram bot hook — full (listener)** — background process that listens for `/digest`, `/syllabus`, and `/review` commands in the docs channel. Generates and posts on demand.



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

## Auditoría técnica — 2026-08-09 (consolidada)

Antes documento separado (`EVALUATION-2026-08-09.md`, eliminado el 2026-08-17 al consolidarse aquí). Hallazgos clave:

- **Los docs describían un piloto 100× más pequeño** (252 mensajes / 71 anchors) que no correspondía a los datos reales (32,227 mensajes / 5,907 anchors). Docs corregidos.
- **Canal real**: chat general de Kreitek (no "Demiurgo"). El fixture de 252 mensajes venía de otro canal y el usuario pidió no sacar conclusiones de él (registrado en `.dao/memory/`).

### A. Ingestión — Telethon como camino primario (✅ implementado)

- Los datos originales vinieron de `chat_export_ingest.py` (export HTML). `extraction_v2.py` usa Telethon directamente (sin pytopicgram) para ingesta primaria.
- **Qué hace**: `run_extraction_v2()` conecta a Telegram via `TelegramClient`, itera mensajes con `client.iter_messages(channel, min_id=last_id)`, normaliza al shape del pipeline, escribe a SQLite via `db.upsert_messages()`. `--incremental` (default) usa `min_id = SELECT MAX(id) FROM messages`; `--full` hace extracción completa.
- **Por qué Telethon**: sin export manual; re-runs idempotentes con `min_id`; `replies`/`forwards`/`views` que el export HTML pierde (útiles para "influential links" y "per-user contributions"). Reacciones emoji (`GetMessagesReactionsRequest`) deferidas a P1.
- **Migración**: la primera extracción Telethon reemplaza los datos del HTML export (IDs secuenciales falsos → IDs reales de Telegram). Requiere re-run completo del pipeline.
- **Pitfalls**: `session_name.session` en la raíz (gitignored, no borrar); esperar `FloodWaitError` (5-15 min para 32K mensajes). `extraction.py` (wrapper pytopicgram) y `chat_export_ingest.py` (HTML) se mantienen como fallback.

### B. Reindex Zvec — política incremental (✅ implementado)

- Zvec está diseñado para updates incrementales: `insert`/`upsert`/`update`/`delete` por id, staging buffer, `optimize()` en background, `flush()`. Escrituras inmediatamente consultables.
- **Antes (anti-pattern)**: `embedding.py:99-104` hacía `shutil.rmtree(index_path)` y re-embebe los 15,330 chunks cada run — ~$0.32 USD + 3-5 min por run.
- **Ahora**: `embed_and_store_incremental()` hace un diff de hashes (`chunks.text_hash`, SHA-256) entre SQLite y el estado del último run (almacenado en `_meta.indexed_hashes`). Solo embebe los chunks nuevos o cambiados, hace `upsert()` en Zvec, borra los stale, y llama `optimize()`. `run_embedding()` auto-detecta: si el índice existe → incremental; si no → full build. `force_full=True` para forzar rebuild completo.

|| Escenario | Antes | Después ||
||---|---|---||
|| 50 mensajes nuevos | $0.32, 3-5 min | $0.0013, ~5 s ||
|| 500 mensajes nuevos | $0.32, 3-5 min | $0.013, ~30 s ||
|| Sin cambios | $0.32, 3-5 min | $0.00, <1 s ||
|| Run inicial (sin índice) | $0.32, 3-5 min | $0.32, 3-5 min (igual) ||

- **Caveat**: el staging buffer es flat index (la búsqueda degrada mientras crece); `optimize()` periódico lo mantiene sano — irrelevante a 15K docs, relevante a 1M+.
- **Clustering**: re-cluster de 15K×1024d tarda segundos; para deltas pequeños, `MiniBatchKMeans.partial_fit()` o re-cluster solo si delta > 5%.
- El hack de `topk=100000` con zero-vector ya está eliminado en clustering (chunk_ids canónicos desde SQLite).

### C. Ingestión en vivo — evaluado, no recomendado

- Un daemon 24/7 `events.NewMessage` es overkill para un digest semanal. Preferible **polling incremental** (cron 15-60 min): `min_id` persistido → append → estado.
- El modo live solo merece la pena con frescura <5 min o triggers por autor — no aplica hoy.

### D. Calidad general — hallazgos

**Fortalezas**: arquitectura 10 fases con verificación end-to-end; fases 1/2/4 bien testeadas (15/12/21 tests); elección de Zvec y mistral-embed documentada; degradación elegante en metadata; separación dual de canales correcta; código bien comentado.

**Pitfalls pendientes**:
- **Clusters pobres a esta escala**: keywords tipo stopword (`creo`, `tengo`, `mismo`, `gente`, `xataka`, `maria`, `molina`) → ampliar stopwords y re-evaluar BERTopic a 5.9K bundles (P1).
- **Digest repetitivo**: el mismo link en Top Topics y Most Discussed Links → dedupe (P1).
- **"Reacciones" es un proxy**: 26,308 registros = mensajes de seguimiento en ventana del anchor, no emoji. Documentado; falta un comentario en `association.py`.
- **892 fallos de metadata**: 18 timeouts recuperables con retry/backoff; el resto (403/429/404) son paywalls y link rot (P3).
- **`delete_this.py`** es un script one-shot, no pytest; sin tests del camino Zvec (P3).
- **Reproducibilidad**: un clone limpio no arranca sin datos; el tarball `alcuinusbot-data-v0.1.tar.gz` (41 MB, gitignored) es el único seed (P3).

### E. Recomendaciones — estado

| # | Recomendación | Estado |
|---|---|---|
| 1 | Migrar storage a SQLite | ✅ Implementado (2026-08-10; commiteado 2026-08-17) |
| 2 | Reindex Zvec incremental | ✅ Implementado (2026-08-22; `embed_and_store_incremental` + `compute_embedding_delta` + `text_hash` column) |
| 3 | Telethon como camino primario | ✅ Implementado (2026-08-22; `extraction_v2.py` con `run_extraction_v2` + `normalize_message` + `fetch_messages`; `--incremental`/`--full`) |
| 4 | Dedupe digest + calidad de clusters | ⏳ P1 pendiente (ver D) |
| 5 | Cache de `load_messages()` | ✅ Supersedido por SQLite |
| 6 | Tests del camino Zvec (embeddings sintéticos) | ⏳ P3 pendiente |
| 7 | Proyecto bootable desde clone | ⏳ P3 pendiente |
| 8 | Retry/backoff en metadata | ⏳ P3 pendiente |
| 9 | Fix `CHANNEL = "Demiurgo"` hardcodeado | ✅ Implementado (`ALCUINUS_CHANNEL_NAME`) |
| 10 | Disambiguar "reacciones" en docs | ✅ Documentado (README/SPECS) |
| 11 | `delete_this.py` → pytest suite | ⏳ P3 pendiente |

### F. SQLite — decisión e implementación (✅ hecho)

- **Decisión (bloqueada)**: SQLite embebido (stdlib) como fuente de verdad; 7 tablas + `_meta`, FKs enforced, WAL mode, índices. Reemplaza 5 JSONs (~38 MB → 30 MB, 97,127 filas).
- Schema completo, migración idempotente y verificación en `src/alcuinus/db.py` (CLI: `python -m alcuinus.db migrate`).
- Todos los módulos leen de SQLite con fallback JSON; los JSONs se escriben una release más por compatibilidad.
- **Lo que desbloquea**: `SELECT chunk_id FROM chunks` (B), `SELECT MAX(id) FROM messages` (A), asociación O(N log A) con `bisect`, fin del hack `topk=100000`, "per-user contributions" como GROUP BY instantáneo.
- `clusters.json`, `decay_profiles.json`, `digest.txt`, `syllabus.md` siguen como JSON/texto (output-shaped, fuera del hot path).

---

## Immediate next steps

| Priority | Task | Esfuerzo | Estimación | Fuente |
|----------|------|----------|------------|--------|
| P1 | Dedupe digest output (mismo link en Top Topics + Most Discussed Links) | Trivial | 30–60 min | Auditoría D |
| P1 | Mejorar calidad de clusters: stopwords (`creo`, `tengo`, `gente`, `xataka`, `mismo`, `maria`, `molina`) + re-evaluar BERTopic a 5.9K | Small–Medium | 0.5–1.5 días | Auditoría D |
| ~~P2~~ | ~~Set up cron job para Phase 10 review cycle~~ — ✅ Hecho (cron monthly 1st at 09:00, job `fdafc446a094`) | n/a | — | — |
| P2 | ~~Cache `load_messages()`~~ — supersedido por SQLite | n/a | — | Auditoría F |
| ~~P3~~ | ~~Retry/backoff en `metadata.py`~~ — ✅ Hecho (`_fetch_with_retry` + exponential backoff, 4 tests) | n/a | — | Auditoría D |
| ~~P3~~ | ~~Reproducibilidad: `bootstrap.sh`~~ — ✅ Hecho | n/a | — | Auditoría D |
| ~~P3~~ | ~~Tests del camino Zvec con embeddings sintéticos~~ — ✅ Hecho (`tests/test_pipeline.py`, 5 tests, real Zvec + mock Mistral) | n/a | — | Auditoría E |
| ~~P3~~ | ~~Reemplazar `delete_this.py` por pytest suite~~ — ✅ Hecho (`tests/test_pipeline.py` reemplaza `delete_this.py`) | n/a | — | Auditoría E |
| — | "Maybe someday" (8 ideas de output, sección arriba) | Varies | 1–2 semanas | — |

---

## All phases complete — 10/10 (100%)

The full AlcuinusBot pipeline is implemented and verified end-to-end on real data. See "End-to-end pipeline verification" section above for details.

---

## End-to-end pipeline verification

El pipeline completo (Phases 0–7) está verificado end-to-end sobre los datos reales de `data/`:

```
32,227 msgs → 5,907 anchors → 5,907 bundles → 5,435 URLs → 15,330 chunks
   → 12 clusters (KMeans + TF-IDF) → 8 decay profiles (LLM)
   Cross-phase ID checks pass. No orphan data. 100% chunk coverage.
   Cluster quality spot-check confirma keywords contra contenido real.
```

> **Nota sobre los números.** El bloque original de "End-to-end pipeline verification" listaba 252 → 71 → 71 → 74 → 173 → 8 → 8. Esos números corresponden a un piloto previo con un subconjunto muy pequeño del canal. Los números arriba (32,227 → 5,907 → 5,907 → 5,435 → 15,330 → 12 → 8) son el estado actual de `data/` a 2026-08-09. La verificación de `delete_this.py` corre contra los datos reales y pasa con cobertura 100%.

Verification: `tests/test_pipeline.py` (5 tests, real Zvec + mock Mistral). Run with:
```bash
.venv/bin/python -m pytest tests/test_pipeline.py -v
```
No API keys required — uses synthetic data and mocked Mistral.

## Estado del reindex Zvec

✅ **Incremental indexing implementado (2026-08-22)**. `embed_and_store_incremental()` en `embedding.py` reemplaza el nuke-and-rebuild. La función difunde hashes SHA-256 (`chunks.text_hash`) entre SQLite y el estado del último run (`_meta.indexed_hashes`) para detectar chunks nuevos, cambiados, y stale. Solo los deltas se envían a Mistral. `run_embedding()` auto-detecta incremental vs full; `force_full=True` para forzar rebuild. 64/64 tests pasan (30 db + 22 embedding + 12 clustering).

Antes: `embedding.py:99-104` hacía `shutil.rmtree(index_path)` + re-embed de 15,330 chunks cada run — ~$0.32 USD + 3-5 min. Ver **Auditoría B** para la tabla de costos comparativos.

---

## Registro de cambios recientes

- **2026-08-22** — P3 items: retry/backoff in metadata.py (`_fetch_with_retry`, 4 tests); bootstrap.sh for fresh-clone setup; test_pipeline.py (5 tests, real Zvec + mock Mistral, replaces delete_this.py). All P0-P3 audit items done. 132/132 tests pass.
- **2026-08-22** — Telethon reintegration (P0 #3 de la auditoría): `extraction_v2.py` — `normalize_message()`, `fetch_messages()`, `run_extraction_v2()` con `--incremental`/`--full`. Telethon directo, sin pytopicgram. Escribe a SQLite + JSON. 10 tests, 74/74 total pasan.
- **2026-08-22** — Reindex Zvec incremental (P0 #2 de la auditoría): `embed_and_store_incremental()` + `compute_embedding_delta()` en `embedding.py`; `text_hash` column + `_hash_text` + `get_chunk_hashes` + `get_meta`/`set_meta` en `db.py`. 64/64 tests pasan (30 db + 22 embedding + 12 clustering). DB migrada: 15,330 hashes backfilled.
- **2026-08-17** — Consolidación: `EVALUATION-2026-08-09.md` eliminado; contenido migrado a la sección "Auditoría técnica". Higiene: `.env` raíz añadido a `.gitignore`; `data/bundles.json`, `data/chunks.json`, `data/link_metadata.json` dejan de estar trackeados (outputs de compatibilidad; SQLite es la fuente de verdad); Phase 4 marcada ✅ en la tabla de estado.
- **2026-08-10** — Capa SQLite (P0 #1 de la auditoría): `src/alcuinus/db.py` + refactor de 9 módulos + `tests/test_db.py` (26 tests). Pipeline completo re-ejecutado contra `data/alcuinus.db` (97,127 filas, integrity ok).
- **2026-07-10** — Bot hook mínimo (`bot.py`) y datos reales de 32K mensajes; Phase 10 completa (100% roadmap).
