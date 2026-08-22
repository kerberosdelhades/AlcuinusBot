"""
SQLite storage layer — single source of truth for AlcuinusBot data.

Replaces 5 large JSON files (channel_messages.json, anchors.json, bundles.json,
chunks.json, link_metadata.json) with a single SQLite database at
``data/alcuinus.db``. Clusters, decay profiles, digest, and syllabus stay as
JSON — they're small, output-shaped, and decoupled from the hot path.

Schema rationale (see ROADMAP.md "Auditoría F" — consolidated from the 2026-08-09 evaluation):

- ``messages`` is the canonical store. Every other table references it by
  ``id`` (Telegram msg_id). Foreign keys enforced via PRAGMA.
- ``anchor_urls`` is normalized — multi-URL anchors have one row per URL.
- ``reactions`` is normalized away from ``bundles`` so we can index by anchor
  and by message independently.
- ``chunks`` keeps the same ``chunk_id`` as before (string PK) so the Zvec
  index id-mapping is unaffected.

Design principles:

- **Backwards compatible**: every public function returns dicts with the same
  shape as the previous JSON, so call sites in anchor_detection /
  association / chunking / etc. need only swap the loader, not the consumer.
- **Idempotent migration**: ``migrate_from_json()`` is safe to re-run.
- **Stdlib only**: no new Python dependencies.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_DB_PATH = "data/alcuinus.db"
DEFAULT_DATA_DIR = "data"

# Source JSON paths (for migration + backwards compat)
JSON_MESSAGES = "data/channel_messages.json"
JSON_ANCHORS = "data/anchors.json"
JSON_BUNDLES = "data/bundles.json"
JSON_CHUNKS = "data/chunks.json"
JSON_METADATA = "data/link_metadata.json"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;       -- crash-safe, allows concurrent readers
PRAGMA synchronous = NORMAL;     -- WAL-friendly durability

-- Core messages: one row per Telegram message
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY,           -- Telegram msg_id
    date            TEXT,                          -- raw input (may be NULL, may be HTML-export format)
    date_iso        TEXT,                          -- normalized ISO 8601 UTC; NULL when unparseable
    sender_id       TEXT,                          -- user_id int, username str, or "Name <id>"
    message         TEXT,                          -- raw message text
    forwarded_from  TEXT,                          -- name of forwarder (NULL if not forwarded)
    reply_to_msg_id INTEGER,                       -- FK messages(id); NULL if top-level
    has_media       INTEGER NOT NULL DEFAULT 0,    -- 0/1
    FOREIGN KEY (reply_to_msg_id) REFERENCES messages(id)
);
CREATE INDEX IF NOT EXISTS idx_messages_date       ON messages(date_iso);
CREATE INDEX IF NOT EXISTS idx_messages_reply_to  ON messages(reply_to_msg_id);
CREATE INDEX IF NOT EXISTS idx_messages_sender     ON messages(sender_id);

-- URL-bearing messages (subset of messages with at least one URL)
CREATE TABLE IF NOT EXISTS anchors (
    msg_id         INTEGER PRIMARY KEY,
    date           TEXT,
    sender_id      TEXT,
    text_preview   TEXT,
    forwarded_from TEXT,                       -- denormalized from messages for convenience
    FOREIGN KEY (msg_id) REFERENCES messages(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_anchors_date ON anchors(date);

-- One row per URL on a multi-URL anchor (normalized many-to-many)
CREATE TABLE IF NOT EXISTS anchor_urls (
    msg_id   INTEGER NOT NULL,
    url_idx  INTEGER NOT NULL,           -- position within anchor's URL list (0-based)
    url      TEXT    NOT NULL,
    PRIMARY KEY (msg_id, url_idx),
    FOREIGN KEY (msg_id) REFERENCES anchors(msg_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_anchor_urls_url ON anchor_urls(url);

-- Bundles: one per anchor. Window metadata only here.
CREATE TABLE IF NOT EXISTS bundles (
    anchor_msg_id        INTEGER PRIMARY KEY,
    window_start_msg_id  INTEGER NOT NULL,
    window_end_msg_id    INTEGER,                -- NULL for last bundle
    window_boundary      TEXT,                   -- 'next_anchor' or 'end_of_data'
    FOREIGN KEY (anchor_msg_id) REFERENCES anchors(msg_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_bundles_window_start ON bundles(window_start_msg_id);
CREATE INDEX IF NOT EXISTS idx_bundles_window_end   ON bundles(window_end_msg_id);

-- Reactions: many per bundle. The windowed follow-up messages.
CREATE TABLE IF NOT EXISTS reactions (
    anchor_msg_id     INTEGER NOT NULL,
    reaction_msg_id   INTEGER NOT NULL,
    sender_id         TEXT,
    date              TEXT,
    text_preview      TEXT,
    reply_to_msg_id   INTEGER,
    strategy          TEXT,                       -- 'window' or 'reply'
    PRIMARY KEY (anchor_msg_id, reaction_msg_id),
    FOREIGN KEY (anchor_msg_id) REFERENCES bundles(anchor_msg_id) ON DELETE CASCADE,
    FOREIGN KEY (reaction_msg_id) REFERENCES messages(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_reactions_anchor   ON reactions(anchor_msg_id);
CREATE INDEX IF NOT EXISTS idx_reactions_message  ON reactions(reaction_msg_id);

-- Link metadata
CREATE TABLE IF NOT EXISTS link_metadata (
    url           TEXT PRIMARY KEY,
    title         TEXT,
    description   TEXT,
    status        TEXT,                       -- 'ok', 'unsupported', 'not_html', 'error', 'not_found'
    error_message TEXT,
    fetched_at    TEXT
);

-- Chunks
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id           TEXT PRIMARY KEY,         -- e.g. 'bundle_13_child_0'
    bundle_anchor_id   INTEGER NOT NULL,         -- FK anchors(msg_id)
    is_parent          INTEGER NOT NULL DEFAULT 0,
    token_estimate     INTEGER NOT NULL DEFAULT 0,
    text               TEXT NOT NULL,
    text_hash          TEXT,
    FOREIGN KEY (bundle_anchor_id) REFERENCES anchors(msg_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_chunks_bundle ON chunks(bundle_anchor_id);

-- Bookkeeping: last migration run + checksum for idempotency
CREATE TABLE IF NOT EXISTS _meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def parse_date_iso(date_str: str | None) -> str | None:
    """Normalize a date string to ISO 8601 UTC for sorting. Returns None on
    unparseable input. Handles:
    - None / empty / whitespace -> None
    - 'DD.MM.YYYY HH:MM:SS UTC+XX:XX' (Telegram HTML export)
    - ISO 8601 with trailing Z or +00:00 (Telethon)
    - ISO 8601 with explicit offset

    Output is always ISO 8601 with explicit UTC offset (+00:00) or None.
    """
    if not date_str or not date_str.strip():
        return None
    s = date_str.strip()
    # Normalize Z suffix
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    # Try ISO first
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except ValueError:
        pass
    # Try Telegram export format: 'DD.MM.YYYY HH:MM:SS UTC+XX:XX'
    parts = s.split(" ")
    if len(parts) >= 2:
        try:
            d, mo, y = parts[0].split(".")
            time_parts = parts[1].split(":")
            h = time_parts[0]
            mi = time_parts[1]
            sec = time_parts[2] if len(time_parts) > 2 else "00"
            return f"{y}-{mo.zfill(2)}-{d.zfill(2)}T{h}:{mi}:{sec}+00:00"
        except (ValueError, IndexError):
            pass
    return None


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

@contextmanager
def connect(db_path: str = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    """Context manager that yields a connection with row factory and FK on.

    Use as::

        with connect() as conn:
            rows = conn.execute("SELECT ...").fetchall()
    """
    # Ensure parent dir exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # Foreign keys are per-connection
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Create tables if they don't exist. Idempotent.

    Also applies incremental schema migrations for existing DBs that
    predate a given column/table. Migrations are additive only — never
    destructive.
    """
    # First create the file if it doesn't exist
    if not Path(db_path).exists():
        with connect(db_path) as conn:
            conn.executescript(SCHEMA)
        return

    with connect(db_path) as conn:
        conn.executescript(SCHEMA)

        # Incremental migrations ------------------------------------------------
        # Each block is idempotent. Add new ones at the bottom.

        # v1 → v2: add anchors.forwarded_from (denormalized from messages)
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(anchors)")}
        if "forwarded_from" not in cols:
            conn.execute("ALTER TABLE anchors ADD COLUMN forwarded_from TEXT")
            # Backfill from messages table
            conn.execute("""
                UPDATE anchors
                   SET forwarded_from = (
                       SELECT m.forwarded_from FROM messages m
                        WHERE m.id = anchors.msg_id
                   )
                 WHERE forwarded_from IS NULL
            """)

        # v2 → v3: add chunks.text_hash (for incremental Zvec embedding diff)
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(chunks)")}
        if "text_hash" not in cols:
            conn.execute("ALTER TABLE chunks ADD COLUMN text_hash TEXT")


def db_exists(db_path: str = DEFAULT_DB_PATH) -> bool:
    return Path(db_path).exists()


# ---------------------------------------------------------------------------
# Row -> dict helpers
# ---------------------------------------------------------------------------

def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {k: row[k] for k in row.keys()}


# ---------------------------------------------------------------------------
# Write API
# ---------------------------------------------------------------------------

def upsert_messages(conn: sqlite3.Connection, messages: list[dict]) -> int:
    """Insert or replace messages. Returns count written."""
    rows = []
    for m in messages:
        from_id = m.get("from_id") or {}
        sender_id = from_id.get("user_id")
        if sender_id is not None:
            sender_id = str(sender_id)
        fwd = m.get("fwd_from") or {}
        forwarded_from = fwd.get("from_name")
        reply_to = m.get("reply_to") or {}
        reply_to_id = reply_to.get("reply_to_msg_id")
        date_raw = m.get("date")
        date_iso = parse_date_iso(date_raw)
        rows.append((
            m["id"],
            date_raw,
            date_iso,
            sender_id,
            m.get("message") or "",
            forwarded_from,
            reply_to_id,
            1 if m.get("has_media") else 0,
        ))
    conn.executemany(
        """INSERT OR REPLACE INTO messages
           (id, date, date_iso, sender_id, message, forwarded_from,
            reply_to_msg_id, has_media)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    return len(rows)


def upsert_anchors(
    conn: sqlite3.Connection, anchors: list[dict]
) -> tuple[int, int]:
    """Insert anchors + their URLs. Returns (anchor_count, url_count)."""
    anchor_rows = []
    url_rows = []
    for a in anchors:
        anchor_rows.append((
            a["msg_id"],
            a.get("date"),
            str(a.get("sender_id") or ""),
            a.get("text_preview") or "",
            a.get("forwarded_from"),
        ))
        for idx, url in enumerate(a.get("urls", [])):
            url_rows.append((a["msg_id"], idx, url))
    conn.executemany(
        "INSERT OR REPLACE INTO anchors "
        "(msg_id, date, sender_id, text_preview, forwarded_from) "
        "VALUES (?, ?, ?, ?, ?)",
        anchor_rows,
    )
    # Wipe & re-insert URLs for each anchor (handles re-runs cleanly)
    if anchor_rows:
        msg_ids = tuple(r[0] for r in anchor_rows)
        qmarks = ",".join("?" * len(msg_ids))
        conn.execute(
            f"DELETE FROM anchor_urls WHERE msg_id IN ({qmarks})", msg_ids
        )
    if url_rows:
        conn.executemany(
            "INSERT INTO anchor_urls (msg_id, url_idx, url) VALUES (?, ?, ?)",
            url_rows,
        )
    return len(anchor_rows), len(url_rows)


def upsert_bundles(
    conn: sqlite3.Connection, bundles: list[dict]
) -> tuple[int, int]:
    """Insert bundles + reactions. Returns (bundle_count, reaction_count)."""
    bundle_rows = []
    reaction_rows = []
    for b in bundles:
        anchor_id = b["anchor"]["msg_id"]
        win = b.get("window", {})
        bundle_rows.append((
            anchor_id,
            win.get("start_msg_id", anchor_id),
            win.get("end_msg_id"),
            win.get("boundary", "next_anchor"),
        ))
        for r in b.get("reactions", []):
            reaction_rows.append((
                anchor_id,
                r["msg_id"],
                str(r.get("sender_id") or ""),
                r.get("date"),
                r.get("text_preview") or "",
                r.get("reply_to_msg_id"),
                r.get("strategy", "window"),
            ))
    conn.executemany(
        "INSERT OR REPLACE INTO bundles "
        "(anchor_msg_id, window_start_msg_id, window_end_msg_id, window_boundary) "
        "VALUES (?, ?, ?, ?)",
        bundle_rows,
    )
    if reaction_rows:
        # Wipe existing reactions for these bundles, then re-insert
        anchor_ids = tuple(set(r[0] for r in reaction_rows))
        qmarks = ",".join("?" * len(anchor_ids))
        conn.execute(
            f"DELETE FROM reactions WHERE anchor_msg_id IN ({qmarks})",
            anchor_ids,
        )
        conn.executemany(
            "INSERT INTO reactions "
            "(anchor_msg_id, reaction_msg_id, sender_id, date, "
            " text_preview, reply_to_msg_id, strategy) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            reaction_rows,
        )
    return len(bundle_rows), len(reaction_rows)


def _hash_text(text: str) -> str:
    """Return SHA-256 hex digest of *text* (UTF-8). Used for incremental
    Zvec embedding diff — when the hash changes, the chunk needs re-embedding."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def upsert_chunks(conn: sqlite3.Connection, chunks: list[dict]) -> int:
    """Insert or replace chunks. Returns count written."""
    rows = [
        (
            c["chunk_id"],
            c["bundle_anchor_id"],
            1 if c.get("is_parent") else 0,
            int(c.get("token_estimate") or 0),
            c["text"],
            _hash_text(c["text"]),
        )
        for c in chunks
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO chunks "
        "(chunk_id, bundle_anchor_id, is_parent, token_estimate, text, text_hash) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    return len(rows)


def upsert_link_metadata(
    conn: sqlite3.Connection, metadata: list[dict]
) -> int:
    """Insert or replace link metadata. Returns count written."""
    rows = [
        (
            m["url"],
            m.get("title"),
            m.get("description"),
            m.get("status"),
            m.get("error"),
            m.get("fetched_at"),
        )
        for m in metadata
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO link_metadata "
        "(url, title, description, status, error_message, fetched_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    return len(rows)


# ---------------------------------------------------------------------------
# Read API — returns dicts in the same shape as the previous JSON loaders,
# so call sites can swap json.load() for these functions without other changes.
# ---------------------------------------------------------------------------

def load_messages(db_path: str = DEFAULT_DB_PATH) -> list[dict]:
    """Return all messages, sorted by id ascending. Shape matches
    the Telethon / chat_export dicts the pipeline already consumes."""
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM messages ORDER BY id"
        ).fetchall()
    out = []
    for r in rows:
        d = _row_to_dict(r)
        # Preserve empty-string-vs-None distinction (legacy JSON had "")
        sender_id = d["sender_id"] if d["sender_id"] is not None else None
        fwd = d["forwarded_from"] if d["forwarded_from"] is not None else None
        out.append({
            "id": d["id"],
            "date": d["date"],
            "from_id": {"user_id": sender_id} if sender_id is not None else {},
            "message": d["message"],
            "fwd_from": {"from_name": fwd} if fwd is not None else None,
            "reply_to": {"reply_to_msg_id": d["reply_to_msg_id"]} if d["reply_to_msg_id"] else None,
            "has_media": bool(d["has_media"]),
        })
    return out


def load_anchors(db_path: str = DEFAULT_DB_PATH) -> list[dict]:
    """Return all anchors (with URLs joined), sorted by msg_id ascending."""
    with connect(db_path) as conn:
        anchor_rows = conn.execute(
            "SELECT * FROM anchors ORDER BY msg_id"
        ).fetchall()
        url_rows = conn.execute(
            "SELECT msg_id, url FROM anchor_urls ORDER BY msg_id, url_idx"
        ).fetchall()
    urls_by_anchor: dict[int, list[str]] = {}
    for r in url_rows:
        urls_by_anchor.setdefault(r["msg_id"], []).append(r["url"])
    out = []
    for r in anchor_rows:
        d = _row_to_dict(r)
        out.append({
            "msg_id": d["msg_id"],
            "date": d["date"],
            "sender_id": d["sender_id"],
            "urls": urls_by_anchor.get(d["msg_id"], []),
            "text_preview": d["text_preview"],
            "forwarded_from": d.get("forwarded_from"),
        })
    return out


def load_bundles(db_path: str = DEFAULT_DB_PATH) -> list[dict]:
    """Return all bundles, sorted by anchor msg_id ascending. Shape matches
    the previous bundles.json output.

    The bundle's anchor.date is the **raw** date string from the messages
    table (Telethon ISO or Telegram HTML export format), not the
    normalized ``date_iso``. This preserves the format downstream
    consumers (e.g. ``decay.py`` heuristic) expect."""
    with connect(db_path) as conn:
        bundle_rows = conn.execute(
            "SELECT * FROM bundles ORDER BY anchor_msg_id"
        ).fetchall()
        # Reactions for all bundles in one go
        reaction_rows = conn.execute(
            "SELECT * FROM reactions ORDER BY anchor_msg_id, reaction_msg_id"
        ).fetchall()
    reactions_by_anchor: dict[int, list[dict]] = {}
    for r in reaction_rows:
        d = _row_to_dict(r)
        reactions_by_anchor.setdefault(d["anchor_msg_id"], []).append({
            "msg_id": d["reaction_msg_id"],
            "date": d["date"],
            "sender_id": d["sender_id"],
            "text_preview": d["text_preview"],
            "reply_to_msg_id": d["reply_to_msg_id"],
            "strategy": d["strategy"],
        })
    # We need full anchor records. Build a single JOIN query so we get
    # both anchor data and the raw message date in one round trip.
    with connect(db_path) as conn:
        joined = conn.execute("""
            SELECT a.msg_id, a.date AS anchor_date, a.sender_id,
                   a.text_preview, a.forwarded_from,
                   m.date AS msg_date, m.reply_to_msg_id AS msg_reply_to
              FROM anchors a
              JOIN messages m ON m.id = a.msg_id
        """).fetchall()
    anchor_lookup: dict[int, dict] = {}
    for r in joined:
        d = dict(r)
        anchor_lookup[d["msg_id"]] = {
            "msg_id": d["msg_id"],
            "date": d["msg_date"] or d["anchor_date"],   # raw, from messages
            "sender_id": d["sender_id"],
            "urls": [],   # populated below
            "text_preview": d["text_preview"],
            "forwarded_from": d["forwarded_from"],
            "reply_to_msg_id": d["msg_reply_to"],
        }
    # URLs
    with connect(db_path) as conn:
        url_rows = conn.execute(
            "SELECT msg_id, url FROM anchor_urls ORDER BY msg_id, url_idx"
        ).fetchall()
    for r in url_rows:
        if r["msg_id"] in anchor_lookup:
            anchor_lookup[r["msg_id"]]["urls"].append(r["url"])

    out = []
    for b in bundle_rows:
        d = _row_to_dict(b)
        anchor_id = d["anchor_msg_id"]
        anchor = anchor_lookup.get(anchor_id, {})
        # Reply-chain metadata (matches the shape produced by association.py)
        reply_target = anchor.get("reply_to_msg_id")
        anchor_enriched = dict(anchor)
        anchor_enriched["reply_to_anchor_msg_id"] = (
            reply_target if reply_target in anchor_lookup else None
        )
        out.append({
            "anchor": anchor_enriched,
            "reactions": reactions_by_anchor.get(anchor_id, []),
            "window": {
                "start_msg_id": d["window_start_msg_id"],
                "end_msg_id": d["window_end_msg_id"],
                "boundary": d["window_boundary"],
                "num_messages": len(reactions_by_anchor.get(anchor_id, [])),
            },
        })
    return out


def load_chunks(db_path: str = DEFAULT_DB_PATH) -> list[dict]:
    """Return all chunks, sorted by chunk_id."""
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM chunks ORDER BY chunk_id"
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def load_link_metadata(db_path: str = DEFAULT_DB_PATH) -> list[dict]:
    """Return all link metadata records (list of dicts, shape matches the
    previous link_metadata.json output)."""
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT url, title, description, status, "
            "       error_message AS error, fetched_at "
            "FROM link_metadata ORDER BY url"
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


# Convenience lookups used by individual modules --------------------------------

def get_distinct_urls(db_path: str = DEFAULT_DB_PATH) -> list[str]:
    """All unique URLs across all anchors, in stable order."""
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT url FROM anchor_urls ORDER BY url"
        ).fetchall()
    return [r["url"] for r in rows]


def get_message_count(db_path: str = DEFAULT_DB_PATH) -> int:
    with connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) AS n FROM messages").fetchone()["n"]


def get_max_message_id(db_path: str = DEFAULT_DB_PATH) -> int | None:
    with connect(db_path) as conn:
        row = conn.execute("SELECT MAX(id) AS m FROM messages").fetchone()
    return row["m"]


def get_chunk_hashes(db_path: str = DEFAULT_DB_PATH) -> dict[str, str]:
    """Return {chunk_id: text_hash} for all chunks. Used by the embedding
    diff to detect which chunks need re-embedding."""
    with connect(db_path) as conn:
        rows = conn.execute("SELECT chunk_id, text_hash FROM chunks").fetchall()
    return {r["chunk_id"]: r["text_hash"] for r in rows if r["text_hash"]}


def get_meta(db_path: str = DEFAULT_DB_PATH, *, key: str) -> str | None:
    with connect(db_path) as conn:
        row = conn.execute("SELECT value FROM _meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(db_path: str = DEFAULT_DB_PATH, *, key: str, value: str) -> None:
    with connect(db_path) as conn:
        conn.execute("INSERT OR REPLACE INTO _meta (key, value) VALUES (?, ?)", (key, value))


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def migrate_from_json(
    data_dir: str = DEFAULT_DATA_DIR,
    db_path: str | None = None,
) -> dict[str, int]:
    """Migrate from JSON files into SQLite. Idempotent.

    Returns a dict with counts for reporting. If any source JSON is missing,
    that table is skipped (caller can run migrations incrementally).
    """
    if db_path is None:
        db_path = os.path.join(data_dir, "alcuinus.db")

    init_db(db_path)
    counts: dict[str, int] = {}

    with connect(db_path) as conn:
        # Wipe (idempotent re-runs of full migration)
        conn.executescript("""
            DELETE FROM chunks;
            DELETE FROM link_metadata;
            DELETE FROM reactions;
            DELETE FROM bundles;
            DELETE FROM anchor_urls;
            DELETE FROM anchors;
            DELETE FROM messages;
        """)

        # Messages
        msgs_path = os.path.join(data_dir, "channel_messages.json")
        if os.path.exists(msgs_path):
            with open(msgs_path, encoding="utf-8") as f:
                msgs = json.load(f)
            counts["messages"] = upsert_messages(conn, msgs)
            print(f"  messages:  {counts['messages']:>7,}")

        # Anchors
        anchors_path = os.path.join(data_dir, "anchors.json")
        if os.path.exists(anchors_path):
            with open(anchors_path, encoding="utf-8") as f:
                anchors = json.load(f)
            a_count, u_count = upsert_anchors(conn, anchors)
            counts["anchors"] = a_count
            counts["anchor_urls"] = u_count
            print(f"  anchors:   {a_count:>7,}  ({u_count:,} URLs)")

        # Backfill anchors.forwarded_from from messages.forwarded_from
        # (JSON anchors carry this field; the column was added in a
        # post-migration schema bump, so re-runs of the original migration
        # leave it NULL on anchors that came from the first run.)
        conn.execute("""
            UPDATE anchors
               SET forwarded_from = (
                   SELECT m.forwarded_from FROM messages m
                    WHERE m.id = anchors.msg_id
               )
             WHERE forwarded_from IS NULL
        """)

        # Bundles
        bundles_path = os.path.join(data_dir, "bundles.json")
        if os.path.exists(bundles_path):
            with open(bundles_path, encoding="utf-8") as f:
                bundles = json.load(f)
            b_count, r_count = upsert_bundles(conn, bundles)
            counts["bundles"] = b_count
            counts["reactions"] = r_count
            print(f"  bundles:   {b_count:>7,}  ({r_count:,} reactions)")

        # Chunks
        chunks_path = os.path.join(data_dir, "chunks.json")
        if os.path.exists(chunks_path):
            with open(chunks_path, encoding="utf-8") as f:
                chunks = json.load(f)
            counts["chunks"] = upsert_chunks(conn, chunks)
            print(f"  chunks:    {counts['chunks']:>7,}")

        # Link metadata
        meta_path = os.path.join(data_dir, "link_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            counts["link_metadata"] = upsert_link_metadata(conn, meta)
            print(f"  metadata:  {counts['link_metadata']:>7,}")

        # Bookkeeping
        conn.execute(
            "INSERT OR REPLACE INTO _meta (key, value) VALUES (?, ?)",
            ("migrated_at", datetime.now(timezone.utc).isoformat()),
        )
        conn.execute(
            "INSERT OR REPLACE INTO _meta (key, value) VALUES (?, ?)",
            ("source", "json"),
        )

    return counts


def verify_migration(
    data_dir: str = DEFAULT_DATA_DIR,
    db_path: str | None = None,
) -> list[str]:
    """Cross-check SQLite vs JSON. Returns a list of issues (empty = OK)."""
    if db_path is None:
        db_path = os.path.join(data_dir, "alcuinus.db")
    issues: list[str] = []

    def _count_json(path: str) -> int | None:
        full = os.path.join(data_dir, path)
        if not os.path.exists(full):
            return None
        with open(full, encoding="utf-8") as f:
            return len(json.load(f))

    def _count_table(table: str) -> int:
        with connect(db_path) as conn:
            return conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]

    for fname, table in [
        ("channel_messages.json", "messages"),
        ("anchors.json", "anchors"),
        ("bundles.json", "bundles"),
        ("chunks.json", "chunks"),
        ("link_metadata.json", "link_metadata"),
    ]:
        json_count = _count_json(fname)
        db_count = _count_table(table)
        if json_count is None:
            continue
        if json_count != db_count:
            issues.append(
                f"  {table}: JSON has {json_count:,}, DB has {db_count:,}"
            )

    # FK integrity
    with connect(db_path) as conn:
        orphan_anchors = conn.execute(
            "SELECT COUNT(*) AS n FROM anchors a "
            "LEFT JOIN messages m ON a.msg_id = m.id "
            "WHERE m.id IS NULL"
        ).fetchone()["n"]
        if orphan_anchors:
            issues.append(f"  anchors with no message: {orphan_anchors}")
        orphan_chunks = conn.execute(
            "SELECT COUNT(*) AS n FROM chunks c "
            "LEFT JOIN anchors a ON c.bundle_anchor_id = a.msg_id "
            "WHERE a.msg_id IS NULL"
        ).fetchone()["n"]
        if orphan_chunks:
            issues.append(f"  chunks with no anchor: {orphan_chunks}")
        orphan_reactions = conn.execute(
            "SELECT COUNT(*) AS n FROM reactions r "
            "LEFT JOIN messages m ON r.reaction_msg_id = m.id "
            "WHERE m.id IS NULL"
        ).fetchone()["n"]
        if orphan_reactions:
            issues.append(f"  reactions with no message: {orphan_reactions}")

    return issues


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="AlcuinusBot SQLite storage")
    sub = parser.add_subparsers(dest="cmd")

    p_mig = sub.add_parser("migrate", help="Migrate JSON files into SQLite")
    p_mig.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p_mig.add_argument("--db", default=None)

    p_verify = sub.add_parser("verify", help="Verify SQLite vs JSON counts")
    p_verify.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p_verify.add_argument("--db", default=None)

    p_init = sub.add_parser("init", help="Create empty DB (no migration)")
    p_init.add_argument("--db", default=DEFAULT_DB_PATH)

    args = parser.parse_args()
    if args.cmd == "migrate":
        print(f"Migrating from {args.data_dir}/ into SQLite...")
        counts = migrate_from_json(args.data_dir, args.db)
        print(f"  done. {sum(counts.values()):,} rows total.")
    elif args.cmd == "verify":
        issues = verify_migration(args.data_dir, args.db)
        if not issues:
            print("✅ Migration verified — counts and FKs match.")
        else:
            print("❌ Migration issues:")
            for i in issues:
                print(i)
            raise SystemExit(1)
    elif args.cmd == "init":
        init_db(args.db)
        print(f"✅ Empty DB initialized at {args.db}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
