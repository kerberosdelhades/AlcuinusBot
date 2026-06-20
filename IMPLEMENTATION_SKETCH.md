### Implementation Sketch (Python)

```python
# Core pipeline (simplified)
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import trafilatura
from telegram import Bot

# 1. Ingest messages (with message_thread_id for sub-thread writes)
# 2. For each message with link: fetch + extract text (trafilatura)
# 3. Embed: model = SentenceTransformer("EmbeddingGemma")
# 4. Cluster: topic_model = BERTopic(hdbscan_model=HDBSCAN(min_cluster_size=5))
# 5. Dynamic topics: topics_over_time = topic_model.topics_over_time(docs, timestamps)
# 6. Entity extraction: spaCy NER + LLM RE → knowledge graph
# 7. Write to sub-thread:
bot.send_message(
    chat_id=CHANNEL_ID,
    message_thread_id=DOC_SUBTHREAD_ID,
    text=formatted_report,
    parse_mode="MarkdownV2"
)
```