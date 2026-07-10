"""
Phase 0 — Chat Export Ingest (HTML → JSON)

Parses Telegram chat export HTML files into our standard message JSON format.
Replaces the Telethon-based extraction.py when working with exported data.

Usage:
    uv run python -m alcuinus.chat_export_ingest /tmp/ChatExport_2026-07-10
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

DEFAULT_OUTPUT = "data/channel_messages.json"


def parse_date(date_div) -> str | None:
    """Extract ISO-ish date from a date div's title attribute."""
    title = date_div.get("title", "") if date_div else ""
    if title:
        return title.strip()
    return None


def parse_message_body(body_div) -> dict | None:
    """Parse a single message body div into our JSON record.

    Returns None for system messages (class 'body details').
    """
    classes = body_div.get("class", [])
    if "details" in classes:
        return None  # system message or date separator

    # Date
    date_div = body_div.find("div", class_="date")
    date = parse_date(date_div)

    # Sender
    from_div = body_div.find("div", class_="from_name")
    sender = from_div.text.strip() if from_div else ""

    # Text
    text_div = body_div.find("div", class_="text")
    text = ""
    if text_div:
        # Get clean text (strip HTML but keep URLs)
        text = text_div.get_text(separator=" ", strip=True)

    # Forwarded info
    fwd_div = body_div.find("div", class_="forwarded")
    forwarded_from = None
    if fwd_div:
        fwd_name = fwd_div.find("div", class_="from_name")
        forwarded_from = fwd_name.text.strip() if fwd_name else None

    # Reply to
    reply_div = body_div.find("div", class_="reply_to")
    reply_to = None
    if reply_div:
        details = reply_div.find("div", class_="details")
        if details:
            reply_text = details.get_text(strip=True)
            # Try to extract message ID if present
            reply_to = reply_text[:50] if reply_text else None

    # Media
    media_div = body_div.find("div", class_="media_wrap")
    has_media = media_div is not None

    # Build standard record (Telethon-compatible format)
    # Phase 1 expects: id, date, from_id.user_id, fwd_from.from_name, message
    record = {
        "id": None,  # will be assigned sequentially
        "date": date,
        "from_id": {"user_id": sender},  # Telethon-compatible: string ID
        "message": text,
        "fwd_from": {"from_name": forwarded_from} if forwarded_from else None,
        "reply_to": reply_to,
        "has_media": has_media,
    }

    return record


def parse_html_file(filepath: str) -> list[dict]:
    """Parse all messages from a single HTML file."""
    with open(filepath, encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    bodies = soup.find_all("div", class_="body")
    messages = []
    for body in bodies:
        record = parse_message_body(body)
        if record is not None and record["message"]:
            messages.append(record)

    return messages


def ingest_chat_export(
    export_dir: str,
    output_path: str = DEFAULT_OUTPUT,
) -> str:
    """Parse all HTML files in a Telegram chat export directory.

    Returns path to the output JSON file.
    """
    export_path = Path(export_dir)
    if not export_path.exists():
        raise FileNotFoundError(f"Export directory not found: {export_dir}")

    all_messages = []
    html_files = sorted(export_path.glob("messages*.html"))

    if not html_files:
        raise FileNotFoundError(f"No messages*.html files found in {export_dir}")

    for html_file in html_files:
        records = parse_html_file(str(html_file))
        all_messages.extend(records)

    # Assign sequential IDs
    for i, msg in enumerate(all_messages):
        msg["id"] = i + 1

    # Sort by date
    all_messages.sort(key=lambda m: m.get("date") or "")

    # Write output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_messages, f, indent=2, ensure_ascii=False)

    return output_path


if __name__ == "__main__":
    if len(sys.argv) > 1:
        export_dir = sys.argv[1]
    else:
        print("Usage: python -m alcuinus.chat_export_ingest <export_dir>")
        sys.exit(1)

    output = ingest_chat_export(export_dir)
    print(f"Messages written to: {output}")

    with open(output) as f:
        data = json.load(f)
    print(f"  Total messages: {len(data)}")
    if data:
        dates = [d for d in (m.get("date") for m in data) if d]
        if dates:
            print(f"  Date range: {dates[0]} → {dates[-1]}")
    url_count = sum(1 for m in data if "http" in (m.get("message") or ""))
    print(f"  Messages with URLs: {url_count}")
