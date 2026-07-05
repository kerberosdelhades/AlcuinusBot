"""
Extraction module — wraps pytopicgram's crawler to ingest messages
from the source Telegram channel.

Uses pytopicgram's Telethon-based crawler for the heavy lifting
(API connection, batch fetching, metadata extraction). We only
need the crawler stage — the rest (metrics, NLP, BERTopic) we
build ourselves with our association logic.
"""

import datetime
import os
import sys
import asyncio

# Add vendored pytopicgram to path
_VENDOR = os.path.join(os.path.dirname(__file__), "..", "..", "vendor", "pytopicgram")
if os.path.isdir(_VENDOR):
    sys.path.insert(0, _VENDOR)

from pytopicgram.crawler import process_channels  # noqa: E402
import pandas as pd  # noqa: E402


def load_config() -> dict:
    """Load Telegram credentials and channel IDs from config/.env."""
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


def build_channels_df(channel_name: str, channel_id: int) -> pd.DataFrame:
    """Build a pytopicgram-compatible channels DataFrame for a single channel."""
    return pd.DataFrame([
        {
            "channel_name": channel_name,
            "url": str(channel_id),
            "user": "",
            "cluster": "source",
            "id": channel_id,
        }
    ])


async def extract_messages(
    channel_name: str,
    channel_id: int,
    api_id: int,
    api_hash: str,
    start_date: datetime.datetime,
    end_date: datetime.datetime,
    output_path: str,
) -> str:
    """
    Extract messages from a Telegram channel using pytopicgram's crawler.

    Args:
        channel_name: Human-readable channel name (for logging).
        channel_id: Telegram channel numeric ID (negative for channels).
        api_id: Telegram API ID.
        api_hash: Telegram API hash.
        start_date: Start of extraction window.
        end_date: End of extraction window.
        output_path: Path for the output JSON file.

    Returns:
        Path to the output JSON file containing extracted messages.
    """
    channels_df = build_channels_df(channel_name, channel_id)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    await process_channels(
        channels=channels_df,
        start_date=start_date,
        end_date=end_date,
        api_id=api_id,
        api_hash=api_hash,
        output_file_name=output_path,
        by_url=False,
        photos=False,
    )

    return output_path


def run_extraction(
    days_back: int = 7,
    output_dir: str = "data",
) -> str:
    """
    Convenience wrapper: load config, extract last N days of messages.

    Args:
        days_back: How many days of history to extract.
        output_dir: Directory for output files.

    Returns:
        Path to the output JSON file.
    """
    config = load_config()

    api_id = int(config["api_id"])
    api_hash = config["api_hash"]
    channel_name = config.get("source_channel_name", "kreitek-ia")
    channel_id = int(config["source_channel"])

    end_date = datetime.datetime.now(datetime.timezone.utc)
    start_date = end_date - datetime.timedelta(days=days_back)

    output_path = os.path.join(output_dir, "channel_messages.json")

    asyncio.run(extract_messages(
        channel_name=channel_name,
        channel_id=channel_id,
        api_id=api_id,
        api_hash=api_hash,
        start_date=start_date,
        end_date=end_date,
        output_path=output_path,
    ))

    return output_path


if __name__ == "__main__":
    output = run_extraction()
    print(f"Messages extracted to: {output}")
