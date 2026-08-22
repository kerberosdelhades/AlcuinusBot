"""
Tests for the SQLite storage layer.

Run with: pytest tests/test_db.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / ".venv" / "lib" / "python3.14" / "site-packages"))

from alcuinus import db


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path):
    """A fresh empty DB in a temp dir."""
    path = str(tmp_path / "test.db")
    db.init_db(path)
    return path


@pytest.fixture
def sample_messages():
    return [
        {
            "id": 1,
            "date": "01.01.2025 10:00:00 UTC+00:00",
            "from_id": {"user_id": "alice"},
            "message": "Hello world",
            "fwd_from": None,
            "reply_to": None,
            "has_media": False,
        },
        {
            "id": 2,
            "date": "01.01.2025 11:00:00 UTC+00:00",
            "from_id": {"user_id": "bob"},
            "message": "Check this: https://example.com",
            "fwd_from": {"from_name": "carol"},
            "reply_to": {"reply_to_msg_id": 1},
            "has_media": True,
        },
        {
            "id": 3,
            "date": None,
            "from_id": {"user_id": ""},  # empty string, not None
            "message": "Reply to 2",
            "fwd_from": None,
            "reply_to": {"reply_to_msg_id": 2},
            "has_media": False,
        },
    ]


@pytest.fixture
def sample_anchors():
    return [
        {
            "msg_id": 2,
            "date": "01.01.2025 11:00:00 UTC+00:00",
            "sender_id": "bob",
            "urls": ["https://example.com", "https://example.org"],
            "text_preview": "Check this: https://example.com",
            "forwarded_from": "carol",
        },
    ]


@pytest.fixture
def sample_bundles():
    return [
        {
            "anchor": {
                "msg_id": 2,
                "date": "01.01.2025 11:00:00 UTC+00:00",
                "sender_id": "bob",
                "urls": ["https://example.com", "https://example.org"],
                "text_preview": "Check this: https://example.com",
                "forwarded_from": "carol",
                "reply_to_msg_id": 1,
                "reply_to_anchor_msg_id": None,
            },
            "reactions": [
                {
                    "msg_id": 3,
                    "date": None,
                    "sender_id": "",
                    "text_preview": "Reply to 2",
                    "reply_to_msg_id": 2,
                    "strategy": "reply",
                },
            ],
            "window": {
                "start_msg_id": 2,
                "end_msg_id": None,
                "boundary": "end_of_data",
                "num_messages": 1,
            },
        },
    ]


@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": "bundle_2_child_0",
            "bundle_anchor_id": 2,
            "is_parent": False,
            "token_estimate": 42,
            "text": "child chunk text",
        },
        {
            "chunk_id": "bundle_2_parent_0",
            "bundle_anchor_id": 2,
            "is_parent": True,
            "token_estimate": 120,
            "text": "parent chunk text",
        },
    ]


@pytest.fixture
def sample_metadata():
    return [
        {"url": "https://example.com", "title": "Example", "description": "An example",
         "status": "ok", "error": None, "fetched_at": "2025-01-01T10:00:00+00:00"},
        {"url": "https://example.org", "title": None, "description": None,
         "status": "unsupported", "error": None, "fetched_at": "2025-01-01T10:00:01+00:00"},
    ]


# ---------------------------------------------------------------------------
# Schema / init
# ---------------------------------------------------------------------------

def test_init_db_creates_all_tables(tmp_db):
    """init_db should create the expected tables."""
    with db.connect(tmp_db) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    table_names = {r["name"] for r in rows}
    expected = {"messages", "anchors", "anchor_urls", "bundles", "reactions",
                "link_metadata", "chunks", "_meta"}
    assert expected.issubset(table_names), f"Missing tables: {expected - table_names}"


def test_init_db_is_idempotent(tmp_db):
    """init_db called twice on the same path is safe."""
    db.init_db(tmp_db)
    db.init_db(tmp_db)  # no error


def test_wal_mode_is_enabled(tmp_db):
    """WAL journal mode is set for concurrent reads + crash safety."""
    with db.connect(tmp_db) as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal", f"Expected WAL, got {mode}"


def test_foreign_keys_are_enabled(tmp_db):
    """PRAGMA foreign_keys is set per-connection."""
    with db.connect(tmp_db) as conn:
        fk_on = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    assert fk_on == 1, f"Expected foreign_keys=1, got {fk_on}"


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def test_parse_date_iso_html_export():
    assert db.parse_date_iso("01.01.2025 10:00:00 UTC+00:00") == "2025-01-01T10:00:00+00:00"


def test_parse_date_iso_telethon_iso():
    assert db.parse_date_iso("2025-01-01T10:00:00+00:00") == "2025-01-01T10:00:00+00:00"


def test_parse_date_iso_z_suffix():
    assert db.parse_date_iso("2025-01-01T10:00:00Z") == "2025-01-01T10:00:00+00:00"


def test_parse_date_iso_empty():
    assert db.parse_date_iso("") is None
    assert db.parse_date_iso("   ") is None
    assert db.parse_date_iso(None) is None


def test_parse_date_iso_unparseable():
    assert db.parse_date_iso("not a date") is None


# ---------------------------------------------------------------------------
# Message round-trip
# ---------------------------------------------------------------------------

def test_upsert_and_load_messages(tmp_db, sample_messages):
    with db.connect(tmp_db) as conn:
        db.upsert_messages(conn, sample_messages)
    loaded = db.load_messages(tmp_db)
    assert len(loaded) == 3
    # First message: alice, has message
    assert loaded[0]["id"] == 1
    assert loaded[0]["from_id"] == {"user_id": "alice"}
    assert loaded[0]["message"] == "Hello world"
    # Second message: forwarded from carol
    assert loaded[1]["fwd_from"] == {"from_name": "carol"}
    # Third message: empty string sender (not None)
    assert loaded[2]["from_id"] == {"user_id": ""}


def test_upsert_messages_replaces(tmp_db, sample_messages):
    """INSERT OR REPLACE — same id replaces the row."""
    with db.connect(tmp_db) as conn:
        db.upsert_messages(conn, sample_messages)
        # Re-upsert with modified text
        modified = [dict(m, message="UPDATED") for m in sample_messages[:1]]
        db.upsert_messages(conn, modified)
    loaded = db.load_messages(tmp_db)
    assert len(loaded) == 3  # not 4
    assert loaded[0]["message"] == "UPDATED"


def test_load_messages_preserves_date_iso_sortable(tmp_db, sample_messages):
    with db.connect(tmp_db) as conn:
        db.upsert_messages(conn, sample_messages)
    with db.connect(tmp_db) as conn:
        rows = conn.execute(
            "SELECT id, date_iso FROM messages ORDER BY date_iso IS NULL, date_iso"
        ).fetchall()
    # Sorted by date_iso with NULLs last
    assert rows[0]["id"] == 1  # 2025-01-01T10:00:00
    assert rows[1]["id"] == 2  # 2025-01-01T11:00:00
    assert rows[2]["id"] == 3  # NULL


# ---------------------------------------------------------------------------
# Anchors and URLs
# ---------------------------------------------------------------------------

def test_upsert_anchors_with_multiple_urls(tmp_db, sample_anchors, sample_messages):
    with db.connect(tmp_db) as conn:
        db.upsert_messages(conn, sample_messages)
        a_count, u_count = db.upsert_anchors(conn, sample_anchors)
    assert a_count == 1
    assert u_count == 2

    loaded = db.load_anchors(tmp_db)
    assert len(loaded) == 1
    assert loaded[0]["msg_id"] == 2
    assert loaded[0]["urls"] == ["https://example.com", "https://example.org"]
    assert loaded[0]["forwarded_from"] == "carol"


def test_anchors_url_replacement_is_clean(tmp_db, sample_anchors, sample_messages):
    """Re-upserting an anchor replaces its URLs, not duplicates them."""
    with db.connect(tmp_db) as conn:
        db.upsert_messages(conn, sample_messages)
        db.upsert_anchors(conn, sample_anchors)
    # Modify URLs and re-upsert
    modified = [dict(sample_anchors[0], urls=["https://new.example.com"])]
    with db.connect(tmp_db) as conn:
        db.upsert_anchors(conn, modified)
    loaded = db.load_anchors(tmp_db)
    assert loaded[0]["urls"] == ["https://new.example.com"]


# ---------------------------------------------------------------------------
# Bundles and reactions
# ---------------------------------------------------------------------------

def test_upsert_bundles_with_reactions(tmp_db, sample_messages, sample_anchors, sample_bundles):
    with db.connect(tmp_db) as conn:
        db.upsert_messages(conn, sample_messages)
        db.upsert_anchors(conn, sample_anchors)
        b_count, r_count = db.upsert_bundles(conn, sample_bundles)
    assert b_count == 1
    assert r_count == 1

    loaded = db.load_bundles(tmp_db)
    assert len(loaded) == 1
    assert loaded[0]["anchor"]["msg_id"] == 2
    assert len(loaded[0]["reactions"]) == 1
    assert loaded[0]["reactions"][0]["msg_id"] == 3


def test_load_bundles_preserves_empty_string_sender(tmp_db, sample_messages, sample_anchors, sample_bundles):
    """Empty-string sender_id must round-trip as '' (not None)."""
    with db.connect(tmp_db) as conn:
        db.upsert_messages(conn, sample_messages)
        db.upsert_anchors(conn, sample_anchors)
        db.upsert_bundles(conn, sample_bundles)
    loaded = db.load_bundles(tmp_db)
    # Reaction 3 has sender_id="" in input
    assert loaded[0]["reactions"][0]["sender_id"] == ""


def test_load_bundles_uses_raw_message_date(tmp_db, sample_messages, sample_anchors, sample_bundles):
    """anchor.date is the raw string (DD.MM.YYYY for HTML export), not date_iso."""
    with db.connect(tmp_db) as conn:
        db.upsert_messages(conn, sample_messages)
        db.upsert_anchors(conn, sample_anchors)
        db.upsert_bundles(conn, sample_bundles)
    loaded = db.load_bundles(tmp_db)
    assert loaded[0]["anchor"]["date"] == "01.01.2025 11:00:00 UTC+00:00"

# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------

def test_upsert_and_load_chunks(tmp_db, sample_chunks, sample_messages, sample_anchors):
    with db.connect(tmp_db) as conn:
        db.upsert_messages(conn, sample_messages)
        db.upsert_anchors(conn, sample_anchors)
        db.upsert_chunks(conn, sample_chunks)
    loaded = db.load_chunks(tmp_db)
    assert len(loaded) == 2
    parents = [c for c in loaded if c["is_parent"]]
    children = [c for c in loaded if not c["is_parent"]]
    assert len(parents) == 1
    assert len(children) == 1
    assert children[0]["chunk_id"] == "bundle_2_child_0"


# ---------------------------------------------------------------------------
# Link metadata
# ---------------------------------------------------------------------------

def test_upsert_and_load_link_metadata(tmp_db, sample_metadata):
    with db.connect(tmp_db) as conn:
        db.upsert_link_metadata(conn, sample_metadata)
    loaded = db.load_link_metadata(tmp_db)
    assert len(loaded) == 2
    statuses = {r["status"] for r in loaded}
    assert statuses == {"ok", "unsupported"}


# ---------------------------------------------------------------------------
# FK integrity
# ---------------------------------------------------------------------------

def test_foreign_keys_prevent_orphan_anchor(tmp_db, sample_anchors):
    """An anchor cannot reference a non-existent message."""
    with db.connect(tmp_db) as conn:
        # Don't insert messages first
        with pytest.raises(Exception):  # sqlite3.IntegrityError
            db.upsert_anchors(conn, sample_anchors)


def test_foreign_keys_prevent_orphan_chunk(tmp_db, sample_chunks):
    """A chunk cannot reference a non-existent anchor."""
    with db.connect(tmp_db) as conn:
        with pytest.raises(Exception):
            db.upsert_chunks(conn, sample_chunks)


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def test_migration_round_trip(tmp_path, sample_messages, sample_anchors,
                              sample_bundles, sample_chunks, sample_metadata):
    """End-to-end: write JSON files, migrate, verify counts match."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db_path = str(tmp_path / "alcuinus.db")

    # Write sample JSONs
    (data_dir / "channel_messages.json").write_text(json.dumps(sample_messages))
    (data_dir / "anchors.json").write_text(json.dumps(sample_anchors))
    (data_dir / "bundles.json").write_text(json.dumps(sample_bundles))
    (data_dir / "chunks.json").write_text(json.dumps(sample_chunks))
    (data_dir / "link_metadata.json").write_text(json.dumps(sample_metadata))

    # Run migration
    counts = db.migrate_from_json(str(data_dir), db_path)
    assert counts["messages"] == 3
    assert counts["anchors"] == 1
    assert counts["bundles"] == 1
    assert counts["reactions"] == 1
    assert counts["chunks"] == 2
    assert counts["link_metadata"] == 2

    # Verify no FK issues
    issues = db.verify_migration(str(data_dir), db_path)
    assert issues == [], f"FK issues: {issues}"


def test_migration_is_idempotent(tmp_path, sample_messages, sample_anchors,
                                  sample_bundles, sample_chunks, sample_metadata):
    """Re-running migration produces the same result."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db_path = str(tmp_path / "alcuinus.db")

    for f, content in [
        ("channel_messages.json", sample_messages),
        ("anchors.json", sample_anchors),
        ("bundles.json", sample_bundles),
        ("chunks.json", sample_chunks),
        ("link_metadata.json", sample_metadata),
    ]:
        (data_dir / f).write_text(json.dumps(content))

    db.migrate_from_json(str(data_dir), db_path)
    first = db.load_messages(db_path)

    db.migrate_from_json(str(data_dir), db_path)
    second = db.load_messages(db_path)

    assert first == second


# ---------------------------------------------------------------------------
# Schema migrations
# ---------------------------------------------------------------------------

def test_schema_migration_adds_forwarded_from_column(tmp_path):
    """A DB created before the forwarded_from column was added gets it on init_db."""
    # Create a DB without the new column
    db_path = str(tmp_path / "test.db")
    with db.connect(db_path) as conn:
        # Create anchors table WITHOUT forwarded_from (mimic old schema)
        conn.execute("""
            CREATE TABLE anchors (
                msg_id INTEGER PRIMARY KEY,
                date TEXT,
                sender_id TEXT,
                text_preview TEXT
            )
        """)
        # Insert a row
        conn.execute(
            "INSERT INTO anchors (msg_id, date, sender_id, text_preview) VALUES (1, '2025', 'a', 't')"
        )

    # init_db should add the column and run the migration
    db.init_db(db_path)

    with db.connect(db_path) as conn:
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(anchors)")}
    assert "forwarded_from" in cols


def test_schema_migration_adds_text_hash_column(tmp_path):
    """A DB created before the text_hash column was added gets it on init_db."""
    db_path = str(tmp_path / "test.db")
    with db.connect(db_path) as conn:
        # Create minimal tables WITHOUT text_hash on chunks (mimic old schema).
        # Must include all columns referenced by SCHEMA indexes so that
        # init_db's executescript(SCHEMA) doesn't fail on CREATE INDEX.
        conn.execute("""
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY,
                date TEXT,
                date_iso TEXT,
                sender_id TEXT,
                message TEXT,
                forwarded_from TEXT,
                reply_to_msg_id INTEGER,
                has_media INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE anchors (
                msg_id INTEGER PRIMARY KEY,
                date TEXT,
                sender_id TEXT,
                text_preview TEXT,
                forwarded_from TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                bundle_anchor_id INTEGER NOT NULL,
                is_parent INTEGER NOT NULL DEFAULT 0,
                token_estimate INTEGER NOT NULL DEFAULT 0,
                text TEXT NOT NULL,
                FOREIGN KEY (bundle_anchor_id) REFERENCES anchors(msg_id) ON DELETE CASCADE
            )
        """)
        # Insert rows to satisfy FK
        conn.execute("INSERT INTO messages (id) VALUES (1)")
        conn.execute("INSERT INTO anchors (msg_id) VALUES (1)")
        conn.execute(
            "INSERT INTO chunks (chunk_id, bundle_anchor_id, is_parent, token_estimate, text) "
            "VALUES ('test_chunk', 1, 0, 10, 'hello')"
        )

    # init_db should add the column
    db.init_db(db_path)

    with db.connect(db_path) as conn:
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(chunks)")}
    assert "text_hash" in cols


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------

def test_get_distinct_urls(tmp_db, sample_messages, sample_anchors):
    with db.connect(tmp_db) as conn:
        db.upsert_messages(conn, sample_messages)
        db.upsert_anchors(conn, sample_anchors)
    urls = db.get_distinct_urls(tmp_db)
    assert sorted(urls) == ["https://example.com", "https://example.org"]


def test_get_max_message_id(tmp_db, sample_messages):
    with db.connect(tmp_db) as conn:
        db.upsert_messages(conn, sample_messages)
    assert db.get_max_message_id(tmp_db) == 3


def test_upsert_chunks_stores_hash(tmp_db, sample_messages, sample_anchors, sample_chunks):
    """upsert_chunks should compute and store a 64-char SHA-256 hash per chunk."""
    with db.connect(tmp_db) as conn:
        db.upsert_messages(conn, sample_messages)
        db.upsert_anchors(conn, sample_anchors)
        db.upsert_chunks(conn, sample_chunks)
    loaded = db.load_chunks(tmp_db)
    assert len(loaded) == 2
    for c in loaded:
        assert c["text_hash"] is not None
        assert len(c["text_hash"]) == 64
        # Verify it's a valid hex string
        int(c["text_hash"], 16)


def test_get_chunk_hashes(tmp_db, sample_messages, sample_anchors, sample_chunks):
    """get_chunk_hashes returns {chunk_id: text_hash} with correct size and 64-char hashes."""
    with db.connect(tmp_db) as conn:
        db.upsert_messages(conn, sample_messages)
        db.upsert_anchors(conn, sample_anchors)
        db.upsert_chunks(conn, sample_chunks)
    hashes = db.get_chunk_hashes(tmp_db)
    assert len(hashes) == 2
    for chunk_id, h in hashes.items():
        assert len(h) == 64
        int(h, 16)  # valid hex


def test_set_and_get_meta(tmp_db):
    """set_meta / get_meta round-trip; nonexistent key returns None."""
    db.set_meta(tmp_db, key="test_key", value="test_value")
    assert db.get_meta(tmp_db, key="test_key") == "test_value"
    assert db.get_meta(tmp_db, key="nonexistent") is None
