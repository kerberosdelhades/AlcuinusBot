## Core Architecture

| Component | Purpose | Tech Options |
|-----------|---------|--------------|
| **Ingestion** | Read channel messages + extract links | `python-telegram-bot` / Pyrogram / Telethon (userbot for channels) |
| **Content Fetching** | Resolve links to full text | `trafilatura`, `newspaper3k`, `readability-lxml` |
| **Embedding & Clustering** | Semantic grouping of messages/links | `sentence-transformers` + `BERTopic` / `HDBSCAN` + `UMAP` |
| **Knowledge Graph** | Entity/relation extraction across messages | `iText2KG`, `LangGraph`, custom NER + RE pipeline |
| **Temporal Analysis** | Track topic evolution over time | `BERTopic.topics_over_time()`, dynamic topic modeling |
| **Output** | Write structured summaries to sub-thread | `message_thread_id` via Bot API |