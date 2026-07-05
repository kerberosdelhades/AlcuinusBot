# AlcuinusBot — Specifications

Bot del grupo de IA de Kreitek. Lee mensajes del canal, analiza contenido y escribe resúmenes estructurados en un sub-hilo de documentación.

## Arquitectura

| Componente | Propósito | Tecnología |
|------------|-----------|------------|
| **Ingesta** | Leer mensajes del canal + extraer enlaces | Pyrogram / Telethon (userbot) |
| **Contenido** | Resolver enlaces a texto completo | `trafilatura`, `newspaper3k` |
| **Embeddings** | Agrupación semántica | `sentence-transformers` + BERTopic / HDBSCAN |
| **Grafo de conocimiento** | Extracción de entidades y relaciones | spaCy NER + LLM |
| **Análisis temporal** | Evolución de temas en el tiempo | BERTopic `topics_over_time()` |
| **Salida** | Resúmenes estructurados en sub-hilo | `message_thread_id` via Bot API |

### Flujo del pipeline

```
Canal Telegram → Userbot (lee mensajes) → SQLite
    → Extraer enlaces → Fetch + Resumir (LLM) → Embed
    → BERTopic clustering → LLM etiqueta clusters
    → Escribir resumen al sub-hilo
```

### Nota: Userbot vs Bot API

El Bot API **no puede leer mensajes de canales** solo participar en grupos. Para leer el canal se necesita un **userbot** (Telethon/Pyrogram con tu cuenta de Telegram). El bot normal se usa solo para escribir al sub-hilo.

## MVP (proyecto de fin de semana)

1. **Userbot** lee últimos 7 días de mensajes → SQLite
2. **Extraer enlaces** → fetch + resumir (LLM) → embed
3. **BERTopic** clustering → LLM etiqueta los clusters
4. **Escribir "Resumen Semanal"** en el sub-hilo de documentación:
   - Top 5 temas (etiquetas de cluster + conteo de mensajes)
   - 3 temas emergentes (clusters nuevos esta semana)
   - 5 enlaces más influyentes (por centralidad semántica)
   - 1 "conexión" (ej: "la discusión sobre MoE se mezcló con el paper nuevo de NVIDIA sobre expert routing")

## Estrategias de análisis (futuro)

Estas son ideas para expandir el bot después del MVP. Se implementan de forma incremental.

1. **Clusters semánticos con etiquetas legibles** — Agrupar mensajes por tema, que un LLM asigne nombres descriptivos a cada grupo.
2. **Resolución de correferencias** — Detectar que "el paper de Google DeepMind" y "el reporte técnico de Gemini 2.5" son la misma entidad.
3. **Trayectorias temporales de temas** — Cómo evolucionan los temas día a día: se dividen, fusionan, desaparecen, aparecen nuevos.
4. **Análisis puente enlace-conversación** — ¿Un enlace ilustró lo que se discutía o introdujo un tema nuevo?
5. **Grafo de citación e influencia** — Trazar cómo una idea evoluciona desde que se introduce un lunes hasta el debate del miércoles.
6. **Mapeo de contradicciones y consenso** — Zonas de acuerdo vs. desacuerdo en el canal, con evidencia.
7. **"Quizás te perdiste" personalizado** — Digests dirigidos por usuario basados en sus intereses históricos.

## Despliegue

| Desafío | Solución |
|---------|----------|
| Bot no lee canales | Userbot (Telethon/Pyrogram) para leer, bot para escribir |
| Rate limits en fetch | Cache de contenido; priorizar enlaces de mensajes con más engagement |
| Ventana de contexto limitada | Resumen map-reduce (LangChain) para artículos largos |
| Ruido (memes, off-topic) | Filtrar por longitud de mensaje, presencia de enlaces, o señales de engagement |
| Contenido multilingual | Embeddings multilingües (`intfloat/multilingual-e5-large`) |

## Esqueleto de implementación

```python
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import trafilatura
from telegram import Bot

# 1. Ingesta: userbot lee mensajes → SQLite
# 2. Para cada mensaje con enlace: fetch + extraer texto (trafilatura)
# 3. Embed: model = SentenceTransformer("all-MiniLM-L6-v2")
# 4. Cluster: topic_model = BERTopic(hdbscan_model=HDBSCAN(min_cluster_size=5))
# 5. Topics dinámicos: topics_over_time = topic_model.topics_over_time(docs, timestamps)
# 6. Extracción de entidades: spaCy NER + LLM → grafo de conocimiento
# 7. Escribir al sub-hilo:
bot.send_message(
    chat_id=CHANNEL_ID,
    message_thread_id=DOC_SUBTHREAD_ID,
    text=formatted_report,
    parse_mode="MarkdownV2"
)
```

## Personalidad del bot

Ver `assets/PERSONALITY.md` — "El Arquitecto de la Claridad", un sintetizador y tutor que destila documentación de IA en conocimiento accionable.
