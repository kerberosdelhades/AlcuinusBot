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

> Nota: la ingestión actual de los datos en `data/channel_messages.json` se hizo vía `chat_export_ingest.py` (export HTML de Telegram Desktop). El camino de Telethon (`extraction.py` + `pytopicgram`) existe, está parcheado localmente, y es el camino recomendado para producción. Ver ROADMAP.md, sección "Auditoría técnica — A. Telethon".

## Dependencias principales

- **[pytopicgram](https://github.com/ugr-sail/pytopicgram)** — Crawler Telethon para ingesta de mensajes (Universidad de Granada, SoftwareX 2025). Vendored en `vendor/pytopicgram/`.
- **[Zvec](https://zvec.org/)** — Vector store embebido in-process, Apache 2.0. Soporta `insert`/`upsert`/`update`/`delete` por id con staging buffer + `optimize()` en background.
- **[Mistral `mistral-embed`](https://docs.mistral.ai/)** — Embeddings multilingüe (es/en), 1024 dim, ~8K tokens/ctx.
- **scikit-learn** — KMeans + TF-IDF para clustering y extracción de keywords.
- **BeautifulSoup + requests** — Metadata de enlaces (HTML genérico + GitHub API + arXiv API).
- **Telethon** — Cliente MTProto (vía pytopicgram).

## Stack

- **Ingesta**: Telethon (vía pytopicgram) o exportación HTML (fallback)
- **Análisis**: KMeans sobre embeddings de Zvec + TF-IDF para keywords
- **Metadata**: HTTP fetch ligero (solo título + descripción)
- **Salida**: Telethon al canal de documentación (`bot.py`)
- **Frontend**: [`../AlcuinusBotTelegram/`](../AlcuinusBotTelegram/) — `python-telegram-bot` echo bot (referencia, no se analiza aquí)

## Cómo correr

```bash
# Setup
uv sync --extra dev  # --extra dev incluye pytest
cp config/.env.example config/.env  # rellenar con TELEGRAM_API_ID, TELEGRAM_API_HASH, MISTRAL_API_KEY

# Tests (175 tests; sin red — Mistral mockeado)
.venv/bin/python -m pytest tests/

# Verificación end-to-end
uv run python delete_this.py

# Ingesta manual (Telethon — recomendado)
uv run python -m alcuinus.extraction

# Ingesta desde export HTML (fallback)
uv run python -m alcuinus.chat_export_ingest /ruta/al/export/

# Publicar digest / syllabus al canal de documentación
uv run python -m alcuinus.bot --digest
uv run python -m alcuinus.bot --syllabus
uv run python -m alcuinus.bot --all
uv run python -m alcuinus.bot --dry-run --all
```

## Avisos importantes

- **No borrar los archivos `*.session`** en la raíz del repo (`bot_session.session`, `session_name.session`). Contienen los tokens de autenticación de Telegram. Si se pierden, re-correr `extraction.py` para regenerar.
- **`data/` está gitignored.** Para reproducir el estado actual desde un clone vacío:
  ```bash
  # 1. Ingestar (HTML export) o (Telethon) → data/channel_messages.json
  uv run python -m alcuinus.chat_export_ingest /ruta/al/export/
  # 2. Migrar JSONs → SQLite
  uv run python -m alcuinus.db migrate
  # 3. Re-correr el pipeline (anchor → association → chunking → metadata → clustering → decay → output)
  uv run python -m alcuinus.association
  uv run python -m alcuinus.chunking
  uv run python -m alcuinus.metadata --only-missing  # solo fetcha URLs nuevas
  uv run python -m alcuinus.clustering
  uv run python -m alcuinus.decay
  uv run python -m alcuinus.output
  uv run python -m alcuinus.syllabus
  ```
- **El storage layer es SQLite** (`data/alcuinus.db`). Migrado desde 5 JSONs planos de 38 MB. Schema completo y API en `src/alcuinus/db.py` (CLI: `uv run python -m alcuinus.db migrate`). Los JSONs se siguen escribiendo para compatibilidad durante una release.
- **El índice Zvec se reconstruye desde cero en cada run** actualmente — esto es ineficiente. La política de reindex incremental está documentada en ROADMAP.md, sección "Auditoría técnica — B. Reindex Zvec" (P0 pendiente).
- **`.env` raíz (MISTRAL_API_KEY / MISTRAL_API_ENDPOINT) está gitignored** desde 2026-08-17 — no commitear credenciales.
- **Si `.venv` está roto** (p. ej. symlinks apuntando a otra máquina tras mover el repo): `rm -rf .venv && uv sync`.
