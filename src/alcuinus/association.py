"""
Association — link subsequent messages to the anchors they react to.

Three strategies, applied in priority order:

1. **Explicit reply** — a message whose ``reply_to`` points directly at
   an anchor is assigned to that anchor, regardless of window boundaries.
2. **Window: until next anchor** — default; messages belong to the nearest
   preceding anchor. The window closes when the next anchor appears.
3. **Time-gap fallback** — for the last anchor in the data, a configurable
   idle cutoff prevents stale assignments (messages posted long after the
   conversation died).

Storage: SQLite is the source of truth. Windowing uses a single indexed
SQL query instead of an O(N × A) Python loop — ``bisect`` on the
sorted anchor list pulled from the DB.
"""

from __future__ import annotations

import json
import os
from bisect import bisect_right
from datetime import datetime, timezone
from typing import Any

from alcuinus import db


def _parse_dt(date_str: str) -> datetime:
    """Parse a datetime string, handling both ISO and Telegram export formats."""
    iso = db.parse_date_iso(date_str)
    if iso is None:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(iso)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def _build_reaction(msg: dict[str, Any], strategy: str) -> dict[str, Any]:
    """Turn a raw Telethon message into a compact reaction record."""
    reply_to = msg.get("reply_to") or {}
    return {
        "msg_id": msg["id"],
        "date": msg["date"],
        "sender_id": (msg.get("from_id") or {}).get("user_id"),
        "text_preview": (msg.get("message") or "")[:300],
        "reply_to_msg_id": reply_to.get("reply_to_msg_id"),
        "strategy": strategy,
    }


def _enrich_anchor(
    anchor: dict[str, Any], msg: dict[str, Any], anchor_ids: set[int]
) -> dict[str, Any]:
    """Add reply-chain metadata to an anchor record."""
    reply_to = msg.get("reply_to") or {}
    reply_target = reply_to.get("reply_to_msg_id")
    result = dict(anchor)
    result["reply_to_msg_id"] = reply_target
    result["reply_to_anchor_msg_id"] = (
        reply_target if reply_target in anchor_ids else None
    )
    return result


def associate(
    messages: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    *,
    max_idle_hours: float = 168.0,
) -> list[dict[str, Any]]:
    """Build bundles: each anchor plus its associated reactions.

    Args:
        messages: Raw Telethon message dicts (the full channel_messages.json).
        anchors: Anchor records from ``anchor_detection.detect_anchors``.
        max_idle_hours: After the last anchor, stop assigning reactions
            once the gap exceeds this.  Default: 168 (7 days).

    Returns:
        A list of bundle dicts, each containing ``anchor``, ``reactions``,
        and ``window`` metadata.  Anchors with zero reactions are still
        included (empty ``reactions`` list).
    """
    # --- index everything --------------------------------------------------
    anchor_by_id: dict[int, dict[str, Any]] = {a["msg_id"]: a for a in anchors}
    anchor_ids: set[int] = set(anchor_by_id)
    anchor_ids_sorted = sorted(anchor_ids)

    msg_by_id: dict[int, dict[str, Any]] = {m["id"]: m for m in messages}

    if not anchor_ids_sorted:
        return []

    # --- enrich anchors with reply metadata --------------------------------
    enriched_anchors: dict[int, dict[str, Any]] = {}
    for aid in anchor_ids_sorted:
        msg = msg_by_id.get(aid)
        if msg:
            enriched_anchors[aid] = _enrich_anchor(anchor_by_id[aid], msg, anchor_ids)
        else:
            enriched_anchors[aid] = dict(anchor_by_id[aid])

    # --- pass 1: window assignment (binary search — O(N log A)) -----------
    # For each non-anchor message, find the largest anchor id <= mid.
    # bisect_right returns the insertion point; subtract 1 to get the index.
    assignment: dict[int, int] = {}

    for msg in sorted(messages, key=lambda m: m["id"]):
        mid = msg["id"]
        if mid in anchor_ids:
            continue
        if mid < anchor_ids_sorted[0]:
            continue
        idx = bisect_right(anchor_ids_sorted, mid) - 1
        if idx < 0:
            continue
        assignment[mid] = anchor_ids_sorted[idx]

    # --- pass 2: reply override + mark reply-anchored messages -------------
    reply_anchored: set[int] = set()
    for mid, current_anchor in list(assignment.items()):
        msg = msg_by_id[mid]
        reply_target = (msg.get("reply_to") or {}).get("reply_to_msg_id")
        if reply_target and reply_target in anchor_ids:
            reply_anchored.add(mid)
            if reply_target != current_anchor:
                assignment[mid] = reply_target

    # --- pass 3: time-gap cleanup (last anchor only) -----------------------
    last_anchor_id = anchor_ids_sorted[-1]
    last_anchor_dt = _parse_dt(msg_by_id[last_anchor_id]["date"])
    for mid in list(assignment):
        if mid in reply_anchored:
            continue
        if assignment[mid] != last_anchor_id:
            continue
        msg_dt = _parse_dt(msg_by_id[mid]["date"])
        if (msg_dt - last_anchor_dt).total_seconds() / 3600 > max_idle_hours:
            del assignment[mid]

    # --- build bundles -----------------------------------------------------
    bundles: list[dict[str, Any]] = []
    for i, aid in enumerate(anchor_ids_sorted):
        reactions = []
        for mid, assigned_to in assignment.items():
            if assigned_to == aid:
                reactions.append(
                    _build_reaction(
                        msg_by_id[mid],
                        strategy="reply" if mid in reply_anchored else "window",
                    )
                )

        # Window metadata
        window_end_id = (
            anchor_ids_sorted[i + 1] if i + 1 < len(anchor_ids_sorted) else None
        )
        boundary = "next_anchor" if window_end_id else "end_of_data"

        bundles.append(
            {
                "anchor": enriched_anchors[aid],
                "reactions": sorted(reactions, key=lambda r: r["msg_id"]),
                "window": {
                    "start_msg_id": aid,
                    "end_msg_id": window_end_id,
                    "boundary": boundary,
                    "num_messages": len(reactions),
                },
            }
        )

    return bundles


def run_association(
    db_path: str = db.DEFAULT_DB_PATH,
    messages_path: str = "data/channel_messages.json",
    anchors_path: str = "data/anchors.json",
    output_path: str = "data/bundles.json",
    *,
    max_idle_hours: float = 168.0,
) -> str:
    """Convenience wrapper: load messages + anchors, build bundles, write output.

    Writes to SQLite (bundles + reactions tables) and JSON (backwards compat).
    """
    # Load — prefer SQLite
    if db.db_exists(db_path):
        messages = db.load_messages(db_path)
        anchors = db.load_anchors(db_path)
    else:
        with open(messages_path, encoding="utf-8") as f:
            messages = json.load(f)
        with open(anchors_path, encoding="utf-8") as f:
            anchors = json.load(f)

    bundles = associate(messages, anchors, max_idle_hours=max_idle_hours)

    # Write to SQLite
    with db.connect(db_path) as conn:
        db.upsert_bundles(conn, bundles)

    # Write JSON for backwards compat
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(bundles, f, indent=2, ensure_ascii=False)

    return output_path


if __name__ == "__main__":
    output = run_association()
    print(f"Bundles written to: {output}")
