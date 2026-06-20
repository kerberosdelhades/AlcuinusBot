### First MVP Scope (Weekend Project)

1. **Userbot** reads last 7 days of messages → SQLite
2. **Extract links** → fetch + summarize (LLM) → embed
3. **BERTopic** clustering → LLM labels clusters
4. **Write one "Weekly Summary"** to your documentation sub-thread with:
   - Top 5 themes (cluster labels + message counts)
   - 3 emerging topics (new clusters this week)
   - 5 most influential links (by semantic centrality in graph)
   - 1 "connection insight" (e.g., "Discussion on MoE architectures merged with the new NVIDIA paper on expert routing")