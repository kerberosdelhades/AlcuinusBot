# AlcuinusBot — Specifications

Bot del grupo de IA de Kreitek. Lee mensajes de un canal de Telegram, analiza contenido, y publica resúmenes estructurados en un canal de documentación separado.

## Arquitectura

Dos canales, separación limpia:

- **Canal fuente** (read-only) — El grupo de IA donde se comparten links, papers, discusiones. El bot lee, nunca escribe.
- **Canal de documentación** (write-only) — Donde el bot publica resúmenes y análisis.

### Componentes

| Componente | Propósito | Tecnología | Estado |
|------------|-----------|------------|--------|
| **Ingesta** | Leer mensajes del canal fuente | pytopicgram crawler (Telethon) → JSON | ✅ |
| **Detección de anchors** | Identificar mensajes con enlaces | urlextract | ✅ |
| **Asociación de opiniones** | Vincular reacciones a su enlace | Three-pass algorithm (window + reply + gap) | ✅ |
| **Metadata de enlaces** | Título, descripción de cada enlace | HTTP fetch + HTML parse (genérico) + GitHub/arXiv API calls | ✅ |
| **Chunking & Tagging** | Partir contenido en chunks recuperables con metadata | Parent-child chunks, 15% overlap (baseline) | Pendiente |
| **Embedding** | Vectorizar chunks para búsqueda semántica | `mistral-embed` (Mistral AI API, 1024 dim) | Pendiente |
| **Almacenamiento vectorial** | Persistir y consultar vectores | pgvector sobre PostgreSQL | Pendiente |
| **Clustering de bundles** | Agrupar enlaces+opiniones por tema | Por decidir (BERTopic como opción principal) | Pendiente |
| **Curación (decay profiles)** | Clasificar contenido por vida útil | Evergreen / semi-stable / ephemeral | Pendiente |
| **Guía de estudio** | Mapa vivo de contenidos para newcomers, organizado por cluster temático y curado por decay profile. No es el digest (Salida), sino la tabla de contenidos persistente. | Markdown o mensajes formateados en canal de doc | Pendiente |
| **Salida** | Publicar resúmenes al canal de doc | Pyrogram / Bot API | Pendiente |

### Dependencia: pytopicgram

Usamos [pytopicgram](https://github.com/ugr-sail/pytopicgram) (Universidad de Granada, SoftwareX 2025) **solo como crawler** — su módulo `crawler.py` para la ingesta de mensajes vía Telethon. No usamos su pipeline de preprocesamiento, métricas, NLP, ni topic modeling.

### Notas de integración con pytopicgram

- **Vendored**: pytopicgram se clona en `vendor/pytopicgram` (no pip install — sus dependencias ML son incompatibles con Python 3.14)
- **Patches locales**: `crawler.py` tiene un fix para que `channel_url` use la entidad resuelta cuando `by_url=False`
- **PeerChannel**: los IDs numéricos de canal se pasan como `PeerChannel` para que Telethon use `GetChannelsRequest` en vez de `GetChatsRequest`

### Dependencias técnicas clave

| Dependencia | Tipo | Justificación |
|-------------|------|---------------|
| **`mistral-embed`** | API externa (Mistral AI) | Embedding multilingual (es/en), 1024 dim, ~8K tokens/ctx. Ya provisionado — sin trabajo extra de integración. |
| **pgvector** | Extensión PostgreSQL | Almacén vectorial sobre Postgres gestionado. Evita provisioning de Qdrant hasta que el corpus lo justifique. |
| **GitHub API** | API externa (sin auth para uso ligero) | Metadata estructurada de repositorios (descripción, topics, estrellas). 60 req/h sin auth. |
| **arXiv API** | API externa (pública) | Metadata de papers (título, autores, abstract). Sin rate limits prácticos. |
| **BERTopic** | Python lib (clustering) | Opción principal para clustering temático sobre vectores de pgvector. Puede operar con embeddings externos. Alternativas por evaluar (KMeans, HDBSCAN standalone). |
| **Telethon** | Python lib (Telegram MTProto) | Crawling del canal fuente. Ya integrado via pytopicgram. |

### Decisiones bloqueadas

- **Embedding**: `mistral-embed` es la decisión final. No se evaluarán alternativas salvo que la calidad de retrieval sea insuficiente en testing.
- **Almacenamiento**: pgvector sobre PostgreSQL existente. Qdrant queda descartado hasta que el tamaño del corpus o necesidades de filtrado lo justifiquen.
- **Canales**: Dos canales separados (fuente read-only, documentación write-only). Esta decisión es estable.
- **Metadata de enlaces**: HTTP GET + BeautifulSoup para la mayoría de URLs. GitHub y arXiv se resuelven vía sus APIs respectivas (sin rate limits para uso ligero, devuelven datos estructurados).

### Flujo del pipeline

```
Canal fuente (read-only)
    → pytopicgram crawler: Telethon → JSON (252 mensajes)
    → Anchor detection: urlextract → 71 anchors (76 URLs)
    → Association: three-pass → bundles (anchor + reactions + window)
    → Metadata: fetch título + descripción (HTML genérico + GitHub/arXiv API) ✅
    → [Pendiente] Chunking: parent-child chunks, 15% overlap, metadata prefix
    → [Pendiente] Embedding: mistral-embed (1024 dim) → vectores
    → [Pendiente] Storage: pgvector (PostgreSQL)
    → [Pendiente] Clustering: por decidir (BERTopic como opción)
    → [Pendiente] Curación: decay profiles (evergreen/semi-stable/ephemeral)
    → [Pendiente] Guía de estudio: mapa vivo de contenidos por cluster
    → [Pendiente] Salida: formatear resumen → canal de documentación
```

### Algoritmo de asociación (Phase 2)

Three-pass algorithm para vincular mensajes posteriores a su anchor:

1. **Window assignment** — cada mensaje pertenece al anchor anterior más cercano. La ventana cierra en el siguiente anchor.
2. **Reply override** — un mensaje con `reply_to` apuntando a un anchor se asigna a ese anchor, sin importar la ventana.
3. **Time-gap cleanup** — para el último anchor, mensajes a más de 168h (7 días) se descartan, salvo los reply-anchored.

## MVP

1. **Ingesta**: pytopicgram crawler lee todo el historial → JSON ✅
2. **Detección de anchors**: mensajes con enlaces ✅
3. **Asociación**: ventana de mensajes → bundles ✅
4. **Metadata**: fetch título + descripción de cada enlace ✅
5. **Chunking & Tagging**: parent-child chunks, 15% overlap, metadata prefix (pendiente)
6. **Embedding**: `mistral-embed` → vectores 1024d (pendiente)
7. **Storage**: pgvector en PostgreSQL (pendiente)
8. **Clustering**: por decidir (BERTopic como opción) (pendiente)
9. **Curación**: clasificación evergreen/semi-stable/ephemeral (pendiente)
10. **Guía de estudio**: mapa vivo de contenidos, organizado por cluster y decay profile. La tabla de contenidos persistente del canal — distinta del digest periódico (pendiente)
11. **Salida**: resumen en canal de documentación (pendiente)

## Datos extraídos

- **252 mensajes** (Sep 2022 → May 2026)
- **71 anchors** con 76 URLs
- Canal: Demiurgo group (`@demiurgo_group`), 2 subscribers, megagroup

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
