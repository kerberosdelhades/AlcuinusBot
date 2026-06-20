### Practical Deployment Notes

| Challenge | Solution |
|-----------|----------|
| **Bot can't read channel messages** | Use a **userbot** (Telethon/Pyrogram with your account) to read, then pass to bot for writing  [stackoverflow](https://stackoverflow.com/questions/68709527/how-to-read-receive-telegram-channel-messages-in-my-telegram-bot) |
| **Rate limits on link fetching** | Cache fetched content; prioritize links from high-engagement messages |
| **Context window limits** | Use **map-reduce summarization** (LangChain) for long articles  [medium](https://medium.com/@ankita.bagaria8/theme-detection-and-paragraph-summarization-with-langchain-and-llm-f3db3c202615) |
| **Noise (memes, off-topic)** | Filter by message length, link presence, or engagement signals before clustering |
| **Multilingual content** | Use multilingual embeddings (`intfloat/multilingual-e5-large`, `EmbeddingGemma`) |