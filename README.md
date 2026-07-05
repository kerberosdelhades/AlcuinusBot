# AlcuinusBot

Bot del grupo de IA de Kreitek. Lee mensajes del canal de Telegram, analiza contenido con embeddings y clustering, y publica resúmenes estructurados en un sub-hilo de documentación.

## Documentación

- **[SPECIFICATIONS.md](SPECIFICATIONS.md)** — Arquitectura, MVP, estrategias de análisis, despliegue
- **[assets/PERSONALITY.md](assets/PERSONALITY.md)** — Personalidad e identidad del bot

## Estado actual

Proyecto en fase de diseño. No hay código implementado aún.

## Stack (planificado)

- **Userbot**: Pyrogram / Telethon (lectura del canal)
- **Bot API**: python-telegram-bot (escritura al sub-hilo)
- **NLP**: sentence-transformers, BERTopic, HDBSCAN
- **Contenido**: trafilatura (extracción de texto de enlaces)
- **LLM**: Etiquetado de clusters y resúmenes
- **Storage**: SQLite
