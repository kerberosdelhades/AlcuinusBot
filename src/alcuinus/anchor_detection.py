"""
Anchor detection — identify messages containing one or more URLs.

These "anchor" messages are the pivot points around which discussion
clusters form. The URLs they carry are what the subsequent conversation
reacts to, debates, or builds upon.
"""

import json
import os
from urlextract import URLExtract


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
    input_path: str = "data/channel_messages.json",
    output_path: str = "data/anchors.json",
) -> str:
    """Convenience wrapper: load messages, detect anchors, write output.

    Returns path to the anchors JSON file.
    """
    with open(input_path, encoding="utf-8") as f:
        messages = json.load(f)

    anchors = detect_anchors(messages)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(anchors, f, indent=2, ensure_ascii=False)

    return output_path


if __name__ == "__main__":
    output = run_anchor_detection()
    print(f"Anchors written to: {output}")
