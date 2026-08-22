"""
Extraction v2 — direct Telethon integration (no pytopicgram dependency).

Uses Telethon's client.iter_messages() to fetch messages from the source
channel, normalizes them to the pipeline's expected JSON shape, and writes
them to both SQLite (primary) and JSON (backwards compat).
"""

import asyncio
import json
import os

from telethon import TelegramClient
from telethon.tl.types import PeerChannel

from alcuinus import db


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load Telegram credentials and channel IDs from config/.env.

    Same pattern as extraction.py's load_config, kept as a separate
    function so tests can monkeypatch it.
    """
    config = {}
    env_path = os.path.join(os.getcwd(), "config", ".env")
    if not os.path.exists(env_path):
        raise FileNotFoundError(
            f"Config not found: {env_path}\n"
            "Copy config/.env.example to config/.env and fill in your credentials."
        )
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                config[key.strip()] = value.strip()
    return config


# ---------------------------------------------------------------------------
# Task 1: normalize_message
# ---------------------------------------------------------------------------

def normalize_message(msg: dict) -> dict:
    """Normalize a Telethon msg.to_dict() output to the pipeline's JSON shape.

    - from_id: {"_": "PeerUser", "user_id": 12345} -> {"user_id": 12345}
    - from_id: None -> {} (empty dict)
    - fwd_from: keep as-is ({"from_name": "..."} or None)
    - reply_to: keep as-is ({"reply_to_msg_id": 42} or None)
    - media: any non-None -> has_media=True; None -> has_media=False
    - id, date, message: pass through
    """
    from_id_raw = msg.get("from_id")
    if from_id_raw is not None:
        from_id = {"user_id": from_id_raw.get("user_id")}
    else:
        from_id = {}

    return {
        "id": msg.get("id"),
        "date": msg.get("date"),
        "from_id": from_id,
        "message": msg.get("message"),
        "fwd_from": msg.get("fwd_from"),
        "reply_to": msg.get("reply_to"),
        "has_media": msg.get("media") is not None,
    }


# ---------------------------------------------------------------------------
# Task 2: fetch_messages
# ---------------------------------------------------------------------------

async def fetch_messages(
    client,
    channel_id: int,
    min_id: int = 0,
    limit: int | None = None,
) -> list[dict]:
    """Fetch messages from a Telegram channel via Telethon.

    Args:
        client: A Telethon TelegramClient instance.
        channel_id: Positive channel ID (already abs()'d).
        min_id: Only fetch messages with id > min_id (for incremental).
        limit: Max messages to fetch (None = all).

    Returns:
        List of normalized message dicts.
    """
    channel = PeerChannel(channel_id=channel_id)
    result = []
    async for msg in client.iter_messages(channel, min_id=min_id, limit=limit):
        result.append(normalize_message(msg.to_dict()))
    return result


# ---------------------------------------------------------------------------
# Task 3: run_extraction_v2
# ---------------------------------------------------------------------------

def run_extraction_v2(
    db_path: str = db.DEFAULT_DB_PATH,
    incremental: bool = True,
    output_dir: str = "data",
) -> dict:
    """Main entry point for extraction v2.

    Args:
        db_path: Path to the SQLite database file.
        incremental: If True, only fetch messages newer than the max id in DB.
        output_dir: Directory for backwards-compat JSON output.

    Returns:
        Summary dict with keys: messages_fetched, min_id, incremental.
    """
    config = _load_config()

    api_id = int(config["api_id"])
    api_hash = config["api_hash"]
    raw_id = int(config["source_channel"])
    channel_id = abs(raw_id)

    # Determine min_id for incremental extraction
    if incremental and db.db_exists(db_path):
        min_id = db.get_max_message_id(db_path) or 0
    else:
        min_id = 0

    async def _run():
        client = TelegramClient("session_name", api_id, api_hash)
        await client.start()
        try:
            messages = await fetch_messages(client, channel_id, min_id=min_id)
        finally:
            await client.disconnect()
        return messages

    messages = asyncio.run(_run())

    if not messages:
        print("No new messages")
        return {"messages_fetched": 0, "min_id": min_id, "incremental": incremental}

    # Write to SQLite
    db.init_db(db_path)
    with db.connect(db_path) as conn:
        count = db.upsert_messages(conn, messages)

    # Write JSON for backwards compat
    json_path = os.path.join(output_dir, "channel_messages.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2, ensure_ascii=False)

    print(f"  {count} messages stored.")
    return {"messages_fetched": count, "min_id": min_id, "incremental": incremental}


# ---------------------------------------------------------------------------
# Task 4: main() CLI
# ---------------------------------------------------------------------------

def main():
    """CLI entry point. --incremental is default; --full overrides."""
    import sys
    incremental = "--full" not in sys.argv
    result = run_extraction_v2(incremental=incremental)
    if result["messages_fetched"] > 0:
        print(f"  {result['messages_fetched']} messages stored.")
    else:
        print("  No new messages.")


if __name__ == "__main__":
    main()
