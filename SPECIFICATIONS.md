# AlcuinusBot — Specifications

Bot del grupo de IA de Kreitek. Lee mensajes de un canal de Telegram, analiza contenido, y publica resúmenes estructurados en un canal de documentación separado.

## Arquitectura

Dos canales, separación limpia:

- **Canal fuente** (read-only) — El grupo de IA donde se comparten links, papers, discusiones. El bot lee, nunca escribe.
- **Canal de documentación** (write-only) — Donde el bot publica resúmenes y análisis.

### Componentes

| Componente | Propósito | Tecnología | Estado |
|------------|-----------|------------|--------|
| **Ingesta** | Leer mensajes del canal fuente | pytopicgram crawler (Telethon) → JSON | ✅ Hecho |
| **Detección de anchors** | Identificar mensajes con enlaces | Custom (sobre JSON de mensajes) | Pendiente |
| **Metadata de enlaces** | Título, descripción de cada enlace | HTTP fetch + HTML parse | Pendiente |
| **Asociación de opiniones** | Vincular reacciones a su enlace | Custom (ventana temporal + señales) | Pendiente |
| **Clustering de bundles** | Agrupar enlaces+opiniones por tema | BERTopic (propio, sobre bundles) | Pendiente |
| **Salida** | Publicar resúmenes al canal de doc | Pyrogram / Bot API | Pendiente |

### Dependencia: pytopicgram

Usamos [pytopicgram](https://github.com/ugr-sail/pytopicgram) (Universidad de Granada, SoftwareX 2025) **solo como crawler** — su módulo `crawler.py` para la ingesta de mensajes vía Telethon. No usamos su pipeline de preprocesamiento, métricas, NLP, ni topic modeling.

Lo que tomamos de pytopicgram:
- Conexión a Telegram API vía Telethon
- Batch fetching de mensajes con metadata completa (reacciones, views, forwards)
- Extracción de URLs de mensajes

Lo que construimos nosotros:
- Detección de anchors (mensajes con enlaces)
- Asociación de opiniones posteriores a cada enlace
- Metadata ligera de enlaces (título, descripción)
- Clustering sobre bundles (enlace + opiniones) en vez de mensajes crudos
- Generación de resúmenes para el canal de documentación

### Notas de integración con pytopicgram

- **Vendored**: pytopicgram se clona en `vendor/pytopicgram` (no pip install — sus dependencias ML son incompatibles con Python 3.14)
- **Patches locales**: `crawler.py` tiene un fix para que `channel_url` use la entidad resuelta cuando `by_url=False`
- **PeerChannel**: los IDs numéricos de canal se pasan como `PeerChannel` para que Telethon use `GetChannelsRequest` en vez de `GetChatsRequest`

### Flujo del pipeline

```
Canal fuente (read-only)
    → pytopicgram crawler: Telethon → JSON (mensajes con metadata)
    → AlcuinusBot:
        → Identificar anchors (mensajes con URLs)
        → Para cada anchor: ventana de mensajes posteriores
        → Asociar opiniones/reacciones al enlace más cercano
        → Fetch metadata de cada enlace (título, descripción)
        → Embed del bundle (metadata + opiniones)
        → BERTopic clustering sobre bundles
        → Formatear resumen
    → Canal de documentación (write-only)
```

### Asociación de opiniones a enlaces

La parte más interesante del pipeline. Cuando alguien comparte un enlace, las reacciones no están en el mensaje siguiente — se extienden decenas de mensajes después, y pueden mezclarse con reacciones a otros enlaces.

**Problema**: dado un mensaje con enlace (anchor), ¿qué mensajes posteriores son reacciones a ese enlace?

**Criterios de ventana** (a definir con datos reales):
- Mensajes hasta el próximo enlace compartido
- Gap temporal (ej: >2h sin mensajes = cierre de ventana)
- Límite fijo (ej: máximo 50 mensajes posteriores)
- Señales explícitas: reply chains, menciones, palabras clave

**Clustering de bundles**: cada bundle (enlace + metadata + opiniones asociadas) se embeddea y clusteriza. Esto agrupa por *temas que generan discusión*, no por contenido mencionado de pasada.

## MVP

1. **Ingesta**: pytopicgram crawler lee todo el historial del canal fuente → JSON ✅
2. **Detección de anchors**: identificar mensajes con enlaces en el JSON
3. **Asociación**: ventana de mensajes posteriores → vincular opiniones a anchors
4. **Metadata**: fetch título + descripción de cada enlace
5. **Clustering de bundles**: BERTopic sobre los bundles (enlace + opiniones)
6. **Salida**: Escribir resumen en el canal de documentación:
   - Top 5 temas (etiquetas de cluster + conteo de mensajes)
   - 3 temas emergentes (clusters nuevos)
   - 5 enlaces más influyentes (por centralidad semántica)
   - 1 "conexión" (ej: "la discusión sobre MoE se mezcló con el paper nuevo de NVIDIA sobre expert routing")

## Estrategias de análisis (futuro)

Ideas para expandir después del MVP. Implementación incremental.

1. **Resolución de correferencias** — "el paper de Google DeepMind" y "el reporte técnico de Gemini 2.5" → misma entidad.
2. **Trayectorias temporales de temas** — Cómo evolucionan los temas: se dividen, fusionan, desaparecen, aparecen nuevos.
3. **Análisis puente enlace-conversación** — ¿Un enlace ilustró lo que se discutía o introdujo un tema nuevo?
4. **Grafo de citación e influencia** — Trazar cómo una idea evoluciona de lunes a miércoles.
5. **Mapeo de contradicciones y consenso** — Zonas de acuerdo vs. desacuerdo, con evidencia.
6. **"Quizás te perdiste" personalizado** — Digests dirigidos por usuario basados en sus intereses.

## Despliegue

| Desafío | Solución |
|---------|----------|
| Bot no lee canales | pytopicgram (Telethon) para lectura |
| Rate limits en fetch | Solo metadata (título, descripción), no contenido completo |
| Ruido (memes, off-topic) | Filtrar por presencia de enlaces |
| Contenido multilingual | Embeddings multilingües (`intfloat/multilingual-e5-large`) |

## Personalidad del bot

Ver `assets/PERSONALITY.md` — "El Arquitecto de la Claridad", un sintetizador y tutor que destila documentación de IA en conocimiento accionable.

## Referencias

- **pytopicgram**: Gómez-Romero et al. *pytopicgram: A library for data extraction and topic modeling from Telegram channels*. SoftwareX 30, 102141 (May 2025). DOI:10.1016/j.softx.2025.102141. GitHub: [ugr-sail/pytopicgram](https://github.com/ugr-sail/pytopicgram)
