"""
Minimal bot hook — post digest and/or syllabus to the docs Telegram channel.

Usage (one-shot):
    uv run python3 -m alcuinus.bot --digest     # post digest only
    uv run python3 -m alcuinus.bot --syllabus   # post syllabus only
    uv run python3 -m alcuinus.bot --all         # post both
    uv run python3 -m alcuinus.bot --dry-run     # preview only, don't send

Config reads api_id, api_hash, and docs_channel from config/.env.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_env() -> dict[str, str]:
    """Load config/.env and return key-value dict."""
    env_path = Path(__file__).resolve().parent.parent.parent / "config" / ".env"
    if not env_path.exists():
        print(f"ERROR: config/.env not found at {env_path}")
        sys.exit(1)

    config = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                config[key.strip()] = value.strip()
    return config


def _get_docs_channel(config: dict) -> int:
    raw = config.get("docs_channel", "")
    try:
        return int(raw)
    except (ValueError, TypeError):
        print(f"ERROR: docs_channel must be an integer, got: {raw!r}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Posting
# ---------------------------------------------------------------------------


def _split_for_telegram(text: str, max_chars: int = 4000) -> list[str]:
    """Split long text into chunks that fit Telegram message limits."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    while len(text) > max_chars:
        # Find a good split point (paragraph break or newline)
        split_at = text.rfind("\n\n", 0, max_chars)
        if split_at == -1:
            split_at = text.rfind("\n", 0, max_chars)
        if split_at == -1:
            split_at = max_chars

        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()

    if text.strip():
        chunks.append(text.strip())
    return chunks


async def _post_message(client, channel_id: int, text: str) -> None:
    """Post a single message, splitting if too long."""
    parts = _split_for_telegram(text)
    for i, part in enumerate(parts):
        # Add continuation marker for long messages
        if len(parts) > 1:
            if i == 0:
                prefix = "📊 _Part 1/" + str(len(parts)) + "_\n\n"
                part = prefix + part
            elif i < len(parts) - 1:
                prefix = "_Continued (part " + str(i + 1) + "/" + str(len(parts)) + ")_\n\n"
                part = prefix + part

        await client.send_message(channel_id, part, parse_mode="Markdown")


async def _run_bot(
    config: dict,
    send_digest: bool = False,
    send_syllabus: bool = False,
    dry_run: bool = False,
) -> None:
    """Main bot logic."""
    from telethon import TelegramClient

    bot_token = config.get("BOT_TOKEN", "").strip('"')
    docs_channel = _get_docs_channel(config)

    if bot_token:
        # Bot API token mode
        client = TelegramClient("bot_session", int(config.get("api_id", 0) or 0), config.get("api_hash", ""))
    else:
        # User API mode (api_id/api_hash required)
        api_id = int(config["api_id"])
        api_hash = config["api_hash"]
        client = TelegramClient("session_name", api_id, api_hash)

    if dry_run:
        # Skip Telethon connection — just print content
        pass
    else:
        client.start(bot_token=bot_token) if bot_token else client.start()

    # Post digest
    if send_digest:
        digest_path = Path("data") / "digest.txt"
        if digest_path.exists():
            text = digest_path.read_text(encoding="utf-8")
            if dry_run:
                print("=== DIGEST (dry-run, not sending) ===")
                print(text)
                print()
            else:
                await _post_message(client, docs_channel, text)
                print("✅ Digest posted to docs channel")
        else:
            print("⚠️  No digest found — run Phase 8 first")

    # Post syllabus
    if send_syllabus:
        syllabus_path = Path("data") / "syllabus.md"
        if syllabus_path.exists():
            text = syllabus_path.read_text(encoding="utf-8")
            if dry_run:
                print("=== SYLLABUS (dry-run, not sending) ===")
                print(text[:500] + "..." if len(text) > 500 else text)
                print()
            else:
                await _post_message(client, docs_channel, text)
                print("✅ Syllabus posted to docs channel")
        else:
            print("⚠️  No syllabus found — run Phase 9 first")

    if not send_digest and not send_syllabus:
        print("Nothing selected. Use --digest, --syllabus, or --all")

    await client.disconnect()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    """CLI entry point — parse args and run bot."""
    args = sys.argv[1:]

    dry_run = "--dry-run" in args or "-n" in args
    send_digest = "--digest" in args or "--all" in args
    send_syllabus = "--syllabus" in args or "--all" in args

    # Remove flags so we don't parse them as positional args
    args = [a for a in args if not a.startswith("-")]

    config = _load_env()

    if dry_run:
        print("🔍 DRY RUN MODE — messages will be printed, not sent\n")

    asyncio.run(_run_bot(config, send_digest, send_syllabus, dry_run))


if __name__ == "__main__":
    main()
