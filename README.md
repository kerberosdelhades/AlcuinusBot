# AlcuinusBot

Bot del grupo de IA de Kreitek. Lee mensajes de un canal de Telegram, analiza contenido con embeddings y clustering, y publica resúmenes estructurados en un canal de documentación separado.

## Documentación

- **[SPECIFICATIONS.md](SPECIFICATIONS.md)** — Arquitectura, MVP, estrategias de análisis, despliegue
- **[ROADMAP.md](ROADMAP.md)** — Estado de las fases, decisiones revertidas, ideas futuras
- **Auditoría técnica (2026-08-09)** — consolidada en [ROADMAP.md](ROADMAP.md), sección "Auditoría técnica"
- **[assets/PERSONALITY.md](assets/PERSONALITY.md)** — Personalidad e identidad del bot

## Estado actual

Pipeline completo de 10 fases implementado y verificado end-to-end sobre datos reales.

| Métrica | Valor |
|---|---|
| Fuente | Kreitek — chat general (vía exportación HTML o Telethon) |
| Mensajes totales | 32,227 (rango 2021-01-01 → 2025-12-31) |
| Anchors (mensajes con URL) | 5,907 |
| URLs únicas en anchors | 5,435 |
| Bundles (anchor + reacciones windowed) | 5,907 |
| Chunks (parent + child) | 15,330 |
| Clusters (KMeans, k=12) | 12 |
| Decay profiles | 8 ephemeral / 3 semi-stable / 1 evergreen |
| Índice Zvec en disco | 73.8 MB |
| Storage layer | **SQLite** (`data/alcuinus.db`, 30 MB) — 7 tablas con FKs, índices, WAL mode. Reemplaza 5 JSONs de 38 MB. Schema completo en `src/alcuinus/db.py` (`SCHEMA`). |

> Nota: la ingestión actual de los datos en `data/channel_messages.json` se hizo vía `chat_export_ingest.py` (export HTML de Telegram Desktop). `extraction_v2.py` (Telethon directo, sin pytopicgram) está implementado como camino primario — ver ROADMAP.md, sección "Auditoría técnica — A. Telethon".

## Dependencias principales

- **[Telethon](https://github.com/LonamiWebs/Telethon)** — Cliente MTProto para ingesta de mensajes (directo, sin pytopicgram). Usado en `extraction_v2.py`.
- **[Zvec](https://zvec.org/)** — Vector store embebido in-process, Apache 2.0. Indexado incremental: `upsert`/`delete` por id + `optimize()` en background.
- **[Mistral `mistral-embed`](https://docs.mistral.ai/)** — Embeddings multilingüe (es/en), 1024 dim, ~8K tokens/ctx.
- **scikit-learn** — KMeans + TF-IDF para clustering y extracción de keywords.
- **BeautifulSoup + requests** — Metadata de enlaces (HTML genérico + GitHub API + arXiv API).
- **[pytopicgram](https://github.com/ugr-sail/pytopicgram)** — Solo como dependencia transitoria para `urlextract` (usado en `anchor_detection.py`). El módulo `crawler.py` ya no se usa.

## Stack

- **Ingesta**: Telethon directo (`extraction_v2.py`) o exportación HTML (fallback)
- **Análisis**: KMeans sobre embeddings de Zvec + TF-IDF para keywords
- **Metadata**: HTTP fetch ligero (solo título + descripción)
- **Salida**: Telethon al canal de documentación (`bot.py`)
- **Frontend**: [`../AlcuinusBotTelegram/`](../AlcuinusBotTelegram/) — `python-telegram-bot` echo bot (referencia, no se analiza aquí)

## Cómo correr

```bash
# Setup
bash bootstrap.sh         # crea venv, instala deps, copia config/.env
# o manual:
uv sync --extra dev       # --extra dev incluye pytest
cp config/.env.example config/.env  # rellenar con api_id, api_hash, source_channel

# Tests (132 tests; sin red — Mistral mockeado)
.venv/bin/python -m pytest tests/

# Verificación end-to-end (sin API keys)
.venv/bin/python -m pytest tests/test_pipeline.py -v

# Ingesta (Telethon — recomendado)
.venv/bin/python -m alcuinus.extraction_v2 --full          # fresh extraction
.venv/bin/python -m alcuinus.extraction_v2 --incremental    # solo nuevos mensajes (default)

# Ingesta desde export HTML (fallback)
.venv/bin/python -m alcuinus.chat_export_ingest /ruta/al/export/

# Publicar digest / syllabus al canal de documentación
.venv/bin/python -m alcuinus.bot --digest
.venv/bin/python -m alcuinus.bot --syllabus
.venv/bin/python -m alcuinus.bot --all
.venv/bin/python -m alcuinus.bot --dry-run --all
```

## Avisos importantes

- **No borrar los archivos `*.session`** en la raíz del repo (`bot_session.session`, `session_name.session`). Contienen los tokens de autenticación de Telegram. Si se pierden, re-correr `extraction_v2.py` para regenerar.
- **`data/` está gitignored.** Para reproducir el estado actual desde un clone vacío:
  ```bash
  # 1. Ingestar (Telethon o HTML export) → data/alcuinus.db
  .venv/bin/python -m alcuinus.extraction_v2 --full     # Telethon
  # o: .venv/bin/python -m alcuinus.chat_export_ingest /ruta/al/export/
  # 2. Migrar JSONs → SQLite (si usaste HTML export)
  .venv/bin/python -m alcuinus.db migrate
  # 3. Re-correr el pipeline
  .venv/bin/python -m alcuinus.anchor_detection
  .venv/bin/python -m alcuinus.association
  .venv/bin/python -m alcuinus.chunking
  .venv/bin/python -m alcuinus.metadata --only-missing
  .venv/bin/python -m alcuinus.embedding          # incremental (auto-detects)
  .venv/bin/python -m alcuinus.clustering
  .venv/bin/python -m alcuinus.decay
  .venv/bin/python -m alcuinus.output
  .venv/bin/python -m alcuinus.syllabus
  ```
- **El storage layer es SQLite** (`data/alcuinus.db`). Schema completo y API en `src/alcuinus/db.py` (CLI: `python -m alcuinus.db migrate`). Los JSONs se siguen escribiendo para compatibilidad.
- **El índice Zvec es incremental** — `embed_and_store_incremental()` hace diff de hashes (`chunks.text_hash`, SHA-256) entre SQLite y el estado del último run. Solo embebe los chunks nuevos/cambiados. Ver ROADMAP.md, sección "Auditoría técnica — B. Reindex Zvec".
- **`.env` raíz (MISTRAL_API_KEY) está gitignored** — no commitear credenciales.
- **Si `.venv` está roto** (p. ej. symlinks apuntando a otra máquina tras mover el repo): `rm -rf .venv && uv sync`.
