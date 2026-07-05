# AlcuinusBot

Bot del grupo de IA de Kreitek. Lee mensajes del canal de Telegram, analiza contenido con embeddings y clustering, y publica resúmenes estructurados en un canal de documentación separado.

## Documentación

- **[SPECIFICATIONS.md](SPECIFICATIONS.md)** — Arquitectura, MVP, estrategias de análisis, despliegue
- **[assets/PERSONALITY.md](assets/PERSONALITY.md)** — Personalidad e identidad del bot

## Estado actual

Proyecto en fase de diseño. No hay código implementado aún.

## Dependencias principales

- **[pytopicgram](https://github.com/ugr-sail/pytopicgram)** — Ingesta de mensajes de Telegram + topic modeling con BERTopic (Universidad de Granada, SoftwareX 2025)
- **sentence-transformers** — Embeddings semánticos
- **BERTopic / HDBSCAN** — Clustering de bundles (enlace + opiniones)
- **requests + BeautifulSoup** — Metadata de enlaces (título, descripción)

## Stack

- **Ingesta**: pytopicgram (Telethon + SQLite)
- **Análisis**: BERTopic sobre bundles (enlaces + opiniones asociadas)
- **Metadata**: HTTP fetch ligero (solo título + descripción)
- **Salida**: Pyrogram / Bot API al canal de documentación
