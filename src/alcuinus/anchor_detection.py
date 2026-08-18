"""
Anchor detection — identify messages containing one or more URLs.

These "anchor" messages are the pivot points around which discussion
clusters form. The URLs they carry are what the subsequent conversation
reacts to, debates, or builds upon.

Storage: SQLite is the source of truth. JSON is the legacy/exchange format
(during one-release migration window).
"""

from __future__ import annotations

import json
import os
from urlextract import URLExtract

from alcuinus import db


_HTTPISH = ("http://", "https://")


def extract_urls(text: str) -> list[str]:
    """Extract URLs from message text. Returns only http/https URLs."""
    extractor = URLExtract()
    raw = extractor.find_urls(text)
    return [u for u in raw if u.startswith(_HTTPISH)]


def build_anchor(message: dict) -> dict | None:
    """Transform a Telethon message dict into an anchor record, or None.

    Returns None when the message contains no http/https URLs.
    """
    text = message.get("message", "") or ""
    urls = extract_urls(text)
    if not urls:
        return None

    fwd = message.get("fwd_from") or {}

    return {
        "msg_id": message["id"],
        "date": message["date"],
        "sender_id": (message.get("from_id") or {}).get("user_id"),
        "forwarded_from": fwd.get("from_name"),
        "urls": urls,
        "text_preview": text[:300],
    }


def detect_anchors(messages: list[dict]) -> list[dict]:
    """Find all anchor messages in a list of Telethon message dicts.

    Returns a list of anchor records, ordered by message ID ascending.
    """
    anchors = []
    for m in messages:
        anchor = build_anchor(m)
        if anchor is not None:
            anchors.append(anchor)
    anchors.sort(key=lambda a: a["msg_id"])
    return anchors


def run_anchor_detection(
    db_path: str = db.DEFAULT_DB_PATH,
    messages_path: str = "data/channel_messages.json",
    output_path: str = "data/anchors.json",
) -> str:
    """Convenience wrapper: load messages, detect anchors, write to SQLite + JSON.

    SQLite is the source of truth (writes to anchors + anchor_urls tables).
    JSON is written for backwards-compat with the rest of the pipeline that's
    being migrated.
    """
    # Load messages — prefer SQLite, fall back to JSON
    if db.db_exists(db_path):
        messages = db.load_messages(db_path)
    else:
        with open(messages_path, encoding="utf-8") as f:
            messages = json.load(f)

    anchors = detect_anchors(messages)

    # Write to SQLite (source of truth)
    with db.connect(db_path) as conn:
        db.upsert_anchors(conn, anchors)

    # Also write JSON for backwards-compat with downstream modules
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(anchors, f, indent=2, ensure_ascii=False)

    return output_path


if __name__ == "__main__":
    output = run_anchor_detection()
    print(f"Anchors written to: {output}")
