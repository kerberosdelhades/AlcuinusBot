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
| **Metadata de enlaces** | Título, descripción de cada enlace | HTTP fetch + HTML parse | Pendiente |
| **Clustering de bundles** | Agrupar enlaces+opiniones por tema | BERTopic (sobre bundles) | Pendiente |
| **Salida** | Publicar resúmenes al canal de doc | Pyrogram / Bot API | Pendiente |

### Dependencia: pytopicgram

Usamos [pytopicgram](https://github.com/ugr-sail/pytopicgram) (Universidad de Granada, SoftwareX 2025) **solo como crawler** — su módulo `crawler.py` para la ingesta de mensajes vía Telethon. No usamos su pipeline de preprocesamiento, métricas, NLP, ni topic modeling.

### Notas de integración con pytopicgram

- **Vendored**: pytopicgram se clona en `vendor/pytopicgram` (no pip install — sus dependencias ML son incompatibles con Python 3.14)
- **Patches locales**: `crawler.py` tiene un fix para que `channel_url` use la entidad resuelta cuando `by_url=False`
- **PeerChannel**: los IDs numéricos de canal se pasan como `PeerChannel` para que Telethon use `GetChannelsRequest` en vez de `GetChatsRequest`

### Flujo del pipeline

```
Canal fuente (read-only)
    → pytopicgram crawler: Telethon → JSON (252 mensajes)
    → Anchor detection: urlextract → 71 anchors (76 URLs)
    → Association: three-pass → bundles (anchor + reactions + window)
    → [Pendiente] Metadata: fetch título + descripción de cada URL
    → [Pendiente] Clustering: BERTopic sobre bundles
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
4. **Metadata**: fetch título + descripción de cada enlace (pendiente)
5. **Clustering de bundles**: BERTopic sobre bundles (pendiente)
6. **Salida**: resumen en canal de documentación (pendiente)

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
