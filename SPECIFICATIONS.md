# AlcuinusBot — Specifications

Bot del grupo de IA de Kreitek. Lee mensajes de un canal de Telegram, analiza contenido, y publica resúmenes estructurados en un canal de documentación separado.

## Arquitectura

Dos canales, separación limpia:

- **Canal fuente** (read-only) — El grupo de IA donde se comparten links, papers, discusiones. El userbot lee, nunca escribe.
- **Canal de documentación** (write-only) — Donde el bot publica resúmenes y análisis. El userbot escribe.

### Componentes

| Componente | Propósito | Tecnología |
|------------|-----------|------------|
| **Ingesta** | Leer mensajes del canal fuente | Pyrogram (userbot, only first extraction) |
| **Almacenamiento** | Persistir mensajes y metadatos | SQLite |
| **Contenido** | Resolver enlaces a texto completo | `trafilatura` |
| **Embeddings** | Representación semántica de textos | `sentence-transformers` |
| **Clustering** | Agrupación temática | BERTopic / HDBSCAN |
| **Salida** | Publicar resúmenes al canal de doc | Pyrogram (mismo userbot) |

### Flujo del pipeline

```
Canal fuente (read-only)
    → Userbot lee mensajes → SQLite
    → Extraer enlaces → Fetch contenido (trafilatura)
    → Embed (sentence-transformers)
    → BERTopic clustering → LLM etiqueta clusters
    → Formatear resumen
    → Canal de documentación (write-only)
```

### Decisión de diseño: un solo userbot

El mismo cliente Pyrogram sirve para leer y escribir. No necesitamos un bot separado con Bot API. Esto simplifica la autenticación a una sola cuenta de Telegram.

## MVP (proyecto de fin de semana)

1. **Ingesta**: Userbot lee últimos 7 días de mensajes del canal fuente → SQLite
2. **Contenido**: Para cada mensaje con enlace: fetch + extraer texto
3. **Análisis**: Embed → BERTopic clustering → LLM etiqueta clusters
4. **Salida**: Escribir "Resumen Semanal" en el canal de documentación:
   - Top 5 temas (etiquetas de cluster + conteo de mensajes)
   - 3 temas emergentes (clusters nuevos esta semana)
   - 5 enlaces más influyentes (por centralidad semántica)
   - 1 "conexión" (ej: "la discusión sobre MoE se mezcló con el paper nuevo de NVIDIA sobre expert routing")

## Estrategias de análisis (futuro)

Ideas para expandir después del MVP. Implementación incremental.

1. **Clusters semánticos con etiquetas legibles** — Agrupar mensajes por tema, que un LLM asigne nombres descriptivos.
2. **Resolución de correferencias** — "el paper de Google DeepMind" y "el reporte técnico de Gemini 2.5" → misma entidad.
3. **Trayectorias temporales de temas** — Cómo evolucionan los temas: se dividen, fusionan, desaparecen, aparecen nuevos.
4. **Análisis puente enlace-conversación** — ¿Un enlace ilustró lo que se discutía o introdujo un tema nuevo?
5. **Grafo de citación e influencia** — Trazar cómo una idea evoluciona de lunes a miércoles.
6. **Mapeo de contradicciones y consenso** — Zonas de acuerdo vs. desacuerdo, con evidencia.
7. **"Quizás te perdiste" personalizado** — Digests dirigidos por usuario basados en sus intereses.

## Despliegue

| Desafío | Solución |
|---------|----------|
| Bot no lee canales | Userbot (Pyrogram) para todo: lectura y escritura |
| Rate limits en fetch | Cache de contenido; priorizar enlaces de mensajes con más engagement |
| Ruido (memes, off-topic) | Filtrar por longitud de mensaje, presencia de enlaces, o señales de engagement |
| Contenido multilingual | Embeddings multilingües (`intfloat/multilingual-e5-large`) |

## Esqueleto de implementación

```python
from pyrogram import Client
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import trafilatura

# Configuración
SOURCE_CHANNEL = -100XXXXXXXXXX    # canal fuente (read-only)
DOCS_CHANNEL = -100YYYYYYYYYY      # canal de documentación (write-only)

# 1. Ingesta: userbot lee mensajes del canal fuente → SQLite
# 2. Para cada mensaje con enlace: fetch + extraer texto (trafilatura)
# 3. Embed: model = SentenceTransformer("all-MiniLM-L6-v2")
# 4. Cluster: topic_model = BERTopic(hdbscan_model=HDBSCAN(min_cluster_size=5))
# 5. Topics dinámicos: topics_over_time = topic_model.topics_over_time(docs, timestamps)
# 6. Formatear y enviar:
app.send_message(
    chat_id=DOCS_CHANNEL,
    text=formatted_report
)
```

## Personalidad del bot

Ver `assets/PERSONALITY.md` — "El Arquitecto de la Claridad", un sintetizador y tutor que destila documentación de IA en conocimiento accionable.
