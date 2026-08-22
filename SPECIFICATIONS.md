# AlcuinusBot — Specifications

Bot del grupo de IA de Kreitek. Lee mensajes de un canal de Telegram, analiza contenido, y publica resúmenes estructurados en un canal de documentación separado.

## Arquitectura

Dos canales, separación limpia:

- **Canal fuente** (read-only) — El grupo de IA donde se comparten links, papers, discusiones. El bot lee, nunca escribe.
- **Canal de documentación** (write-only) — Donde el bot publica resúmenes y análisis.

### Componentes

| Componente | Propósito | Tecnología | Estado |
|------------|-----------|------------|--------|
| **Ingesta** | Leer mensajes del canal fuente | `extraction_v2.py` (Telethon directo, `--incremental`/`--full`). Fallback: `chat_export_ingest.py` (HTML export) | ✅ |
| **Storage** | Persistir mensajes, anchors, bundles, chunks, metadata | **SQLite** in-process (`data/alcuinus.db`, 7 tablas, FKs, WAL mode) | ✅ |
| **Detección de anchors** | Identificar mensajes con enlaces | urlextract | ✅ |
| **Asociación de opiniones** | Vincular reacciones a su enlace | Three-pass algorithm (window + reply + gap). Usa `bisect` en lista de anchors desde SQLite (O(N log A)) | ✅ |
| **Metadata de enlaces** | Título, descripción de cada enlace | HTTP fetch + HTML parse (genérico) + GitHub/arXiv API calls. Soporta `--only-missing` para re-runs incrementales | ✅ |
| **Chunking & Tagging** | Partir contenido en chunks recuperables con metadata | Parent-child chunks, 15% overlap (baseline). Channel name configurable via `ALCUINUS_CHANNEL_NAME` | ✅ |
| **Embedding** | Vectorizar chunks para búsqueda semántica | `mistral-embed` (Mistral AI API, 1024 dim) | ✅ |
| **Almacenamiento vectorial** | Persistir y consultar vectores | Zvec (embebido, in-process, Apache 2.0). Indexado incremental: `upsert` por id + `optimize()` en background. `text_hash` (SHA-256) en SQLite para diff. `delete` por id para stale chunks | ✅ |
| **Clustering de bundles** | Agrupar enlaces+opiniones por tema | KMeans (scikit-learn) + TF-IDF keywords. BERTopic descartado en el piloto de 71 bundles (UMAP necesita ~200+ puntos); pendiente re-evaluar a 5,907 bundles — ver ROADMAP.md "Auditoría D". Lee chunk_ids de SQLite (no más hack de `topk=100000` con zero-vector) | ✅ |
| **Curación (decay profiles)** | Clasificar contenido por vida útil | Evergreen / semi-stable / ephemeral. LLM-based (Mistral) + heuristic fallback | ✅ |
| **Guía de estudio** | Syllabus: mapa vivo para newcomers, organizado por cluster y decay profile. | Markdown estructurado → `data/syllabus.md`. Tiered: evergreen → semi-stable → ephemeral | ✅ |
| **Salida** | Publicar resúmenes al canal de doc | Markdown formateado → `data/digest.txt`. Top 5 topics, emerging themes, influential links, LLM connection insight | ✅ |

### Dependencia: pytopicgram

Usamos [pytopicgram](https://github.com/ugr-sail/pytopicgram) (Universidad de Granada, SoftwareX 2025) **solo como crawler** — su módulo `crawler.py` para la ingesta de mensajes vía Telethon. No usamos su pipeline de preprocesamiento, métricas, NLP, ni topic modeling.

Como alternativa, `chat_export_ingest.py` parsea exports HTML de Telegram Desktop y produce el mismo formato JSON. Útil cuando solo se tiene un export puntual o se quiere evitar autenticación Telethon. La ingestión actual de los datos en `data/` vino de este camino.

### Notas de integración con pytopicgram

- **Vendored**: pytopicgram se clona en `vendor/pytopicgram` (no pip install — sus dependencias ML son incompatibles con Python 3.14)
- **Patches locales**: `crawler.py` tiene un fix para que `channel_url` use la entidad resuelta cuando `by_url=False`
- **PeerChannel**: los IDs numéricos de canal se pasan como `PeerChannel` para que Telethon use `GetChannelsRequest` en vez de `GetChatsRequest`

### Dependencias técnicas clave

| Dependencia | Tipo | Justificación |
|-------------|------|---------------|
| **`SQLite`** (stdlib) | Embedded DB | Almacenamiento principal. 7 tablas, FKs enforced, WAL mode, índices. Reemplaza 5 JSONs. Schema completo en `src/alcuinus/db.py` (CLI: `python -m alcuinus.db migrate`). |
| **`mistral-embed`** | API externa (Mistral AI) | Embedding multilingual (es/en), 1024 dim, ~8K tokens/ctx. Ya provisionado — sin trabajo extra de integración. |
| **Zvec** | Biblioteca embebida (Apache 2.0) | "SQLite for vector search" — in-process, sin servidor, reranking integrado (weighted fusion + RRF). ~128MB RAM para ~100K embeddings. API `Fetch(pks)` permite recuperar vectores por ID para clustering. Sustituye pgvector y elimina la dependencia de PostgreSQL. |
| **GitHub API** | API externa (sin auth para uso ligero) | Metadata estructurada de repositorios (descripción, topics, estrellas). 60 req/h sin auth. |
| **arXiv API** | API externa (pública) | Metadata de papers (título, autores, abstract). Sin rate limits prácticos. |
| **BERTopic** | Python lib (clustering) | Opción principal para clustering temático sobre vectores de Zvec. Puede operar con embeddings externos. Alternativas por evaluar (KMeans, HDBSCAN standalone). |
| **Telethon** | Python lib (Telegram MTProto) | Crawling del canal fuente. Ya integrado via pytopicgram. |

### Decisiones bloqueadas

- **Storage**: SQLite (embebido, in-process, stdlib). Reemplaza 5 JSONs planos. Schema en `src/alcuinus/db.py`. Migración idempotente via `python -m alcuinus.db migrate`.
- **Embedding**: `mistral-embed` es la decisión final. No se evaluarán alternativas salvo que la calidad de retrieval sea insuficiente en testing.
- **Almacenamiento vectorial**: Zvec (embebido, in-process, Apache 2.0). Sustituye pgvector/PostgreSQL. Elimina la dependencia de un servidor externo. Reranking integrado (weighted fusion + RRF) — no se necesita un reranker externo salvo que la precisión sea insuficiente en testing.
- **Canales**: Dos canales separados (fuente read-only, documentación write-only). Esta decisión es estable.
- **Metadata de enlaces**: HTTP GET + BeautifulSoup para la mayoría de URLs. GitHub y arXiv se resuelven vía sus APIs respectivas (sin rate limits para uso ligero, devuelven datos estructurados).

### Flujo del pipeline

```
Canal fuente (read-only, Kreitek general chat)
    → Telethon (`extraction_v2.py`, `--incremental`): 32,227 mensajes → SQLite + data/channel_messages.json
    → Migración: JSONs → SQLite (data/alcuinus.db, 7 tablas, WAL mode, FKs)
    → Anchor detection: urlextract → 5,907 anchors (5,435 URLs únicas)
    → Association: three-pass con bisect (O(N log A)) → 5,907 bundles (anchor + windowed reactions)
    → Metadata: fetch título + descripción (HTML genérico + GitHub/arXiv API) → 5,435 URLs, 3,851 ok
    → Chunking: parent-child chunks, 15% overlap, metadata prefix → 15,330 chunks
    → Embedding: mistral-embed (1024 dim) → vectores → Zvec (in-process, indexado incremental via upsert + optimize)
    → Clustering: KMeans (scikit-learn) + TF-IDF keywords → 12 clusters. Lee chunk_ids de SQLite
    → Curación: decay profiles (evergreen/semi-stable/ephemeral) → 8/3/1
    → Guía de estudio: mapa vivo de contenidos por cluster
    → Salida: digest + syllabus → canal de documentación
```

**Storage:** SQLite es la fuente de verdad. Los JSONs se siguen escribiendo por compatibilidad (una release), pero todas las operaciones de lectura prefieren SQLite. Ver `src/alcuinus/db.py` y ROADMAP.md "Auditoría F".

**Costo por run:** Run inicial ~$0.32 USD (Mistral) + 3-5 min. Runs incrementales: $0.00 si sin cambios, ~$0.0013 para 50 mensajes nuevos. Ver `ROADMAP.md` "Auditoría B" para la tabla completa de costos comparativos.

### Algoritmo de asociación (Phase 2)

Three-pass algorithm para vincular mensajes posteriores a su anchor:

1. **Window assignment** — cada mensaje pertenece al anchor anterior más cercano. La ventana cierra en el siguiente anchor.
2. **Reply override** — un mensaje con `reply_to` apuntando a un anchor se asigna a ese anchor, sin importar la ventana.
3. **Time-gap cleanup** — para el último anchor, mensajes a más de 168h (7 días) se descartan, salvo los reply-anchored.

## MVP

1. **Ingesta**: Telethon (vía pytopicgram) o export HTML → JSON ✅
2. **Detección de anchors**: mensajes con enlaces ✅
3. **Asociación**: three-pass (window + reply override + time-gap) → bundles ✅
4. **Metadata**: fetch título + descripción de cada enlace (HTML genérico + GitHub API + arXiv API) ✅
5. **Chunking & Tagging**: parent-child chunks, 15% overlap, metadata prefix ✅
6. **Embedding**: `mistral-embed` → vectores 1024d ✅
7. **Storage**: Zvec (in-process, upsert/update/delete por id, optimize() en background) ✅
8. **Clustering**: KMeans (scikit-learn) + TF-IDF keywords ✅
9. **Curación**: clasificación LLM (Mistral) con fallback heurístico → evergreen/semi-stable/ephemeral ✅
10. **Guía de estudio**: mapa vivo de contenidos, organizado por cluster y decay profile ✅
11. **Salida**: digest semanal + syllabus persistente al canal de documentación ✅

## Datos extraídos

Estado a 2026-08-09 (ver `data/`):

- **32,227 mensajes** (2021-01-01 → 2025-12-31)
- **5,907 anchors** con 5,435 URLs únicas
- **5,907 bundles** (3,398 con ≥1 reacción windowed, 26,308 reacciones totales)
- **15,330 chunks** (6,166 parent + 9,164 child)
- **12 clusters** (KMeans, k=12), **8 decay profiles** ephemeral / 3 semi-stable / 1 evergreen
- **Canal**: chat general de Kreitek (no el subgrupo "Demiurgo" que el SPEC original mencionaba — esa descripción corresponde a un piloto previo de 252 mensajes que ya no representa el estado del dato)
- **Ingestión usada**: `chat_export_ingest.py` (export HTML). El camino Telethon existe y es el recomendado para producción — ver ROADMAP.md "Auditoría A".

> **Nota sobre "reacciones":** el SPEC original menciona "0 reacciones extraídas". Esto se refiere a reacciones emoji de Telegram. La pipeline de asociación (Phase 2) produce 26,308 registros `reaction` que son **mensajes de seguimiento dentro de la ventana del anchor** — un proxy de engagement, no reacciones reales. La distinción está documentada en ROADMAP.md "Auditoría D".

## Estrategias de análisis (futuro)

1. Resolución de correferencias
2. Trayectorias temporales de temas
3. Análisis puente enlace-conversación
4. Grafo de citación e influencia
5. Mapeo de contradicciones y consenso
6. "Quizás te perdiste" personalizado

## Referencias

- **pytopicgram**: Gómez-Romero et al. *pytopicgram: A library for data extraction and topic modeling from Telegram channels*. SoftwareX 30, 102141 (May 2025). DOI:10.1016/j.softx.2025.102141
- **Chunk overlap (15% baseline)**: NVIDIA, *Finding the Best Chunking Strategy for Accurate AI Responses*. FinanceBench benchmark: 15% overlap óptimo con chunks de 1,024 tokens (rango probado: 10%, 15%, 20%). https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses/
- **LlamaIndex SentenceSplitter defaults**: `chunk_size=1024`, `chunk_overlap=200` (~20%). https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/sentence_splitter/
