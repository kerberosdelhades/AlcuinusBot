"""
Tests for extraction_v2 — Telethon reintegration.

Tests cover:
  1. normalize_message — 4 tests
  2. fetch_messages — 1 test (mock async iterator)
  3. run_extraction_v2 — 3 tests (incremental, full, sqlite write)
  4. main() CLI — 2 tests (incremental flag, full flag)

All tests mock Telegram — no real network connections.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / ".venv" / "lib" / "python3.14" / "site-packages"))

from alcuinus import db
from alcuinus import extraction_v2


# ---------------------------------------------------------------------------
# Task 1: normalize_message tests
# ---------------------------------------------------------------------------

def test_normalize_message_basic():
    """Basic message with from_id, no reply, no media."""
    raw = {
        "id": 100,
        "date": "2025-01-01 10:00:00+00:00",
        "message": "Hello world",
        "from_id": {"_": "PeerUser", "user_id": 12345},
        "fwd_from": None,
        "reply_to": None,
        "media": None,
    }
    result = extraction_v2.normalize_message(raw)
    assert result["id"] == 100
    assert result["date"] == "2025-01-01 10:00:00+00:00"
    assert result["message"] == "Hello world"
    assert result["from_id"] == {"user_id": 12345}
    assert result["fwd_from"] is None
    assert result["reply_to"] is None
    assert result["has_media"] is False


def test_normalize_message_with_reply():
    """Message with reply_to={"reply_to_msg_id": 5000}."""
    raw = {
        "id": 101,
        "date": "2025-01-01 11:00:00+00:00",
        "message": "Replying to you",
        "from_id": {"_": "PeerUser", "user_id": 67890},
        "fwd_from": None,
        "reply_to": {"reply_to_msg_id": 5000},
        "media": None,
    }
    result = extraction_v2.normalize_message(raw)
    assert result["reply_to"] == {"reply_to_msg_id": 5000}
    assert result["from_id"] == {"user_id": 67890}
    assert result["has_media"] is False


def test_normalize_message_forwarded():
    """Forwarded message with fwd_from and media present."""
    raw = {
        "id": 102,
        "date": "2025-01-01 12:00:00+00:00",
        "message": "Forwarded content",
        "from_id": {"_": "PeerUser", "user_id": 11111},
        "fwd_from": {"from_name": "Original Channel"},
        "reply_to": None,
        "media": {"_": "MessageMediaPhoto", "photo": {"_": "Photo"}},
    }
    result = extraction_v2.normalize_message(raw)
    assert result["fwd_from"] == {"from_name": "Original Channel"}
    assert result["has_media"] is True
    assert result["from_id"] == {"user_id": 11111}


def test_normalize_message_no_from_id():
    """Message from a channel (from_id=None) -> from_id={}."""
    raw = {
        "id": 103,
        "date": "2025-01-01 13:00:00+00:00",
        "message": "Channel post",
        "from_id": None,
        "fwd_from": None,
        "reply_to": None,
        "media": None,
    }
    result = extraction_v2.normalize_message(raw)
    assert result["from_id"] == {}
    assert result["has_media"] is False


# ---------------------------------------------------------------------------
# Task 2: fetch_messages test
# ---------------------------------------------------------------------------

def test_fetch_messages_returns_normalized():
    """Mock client.iter_messages as an async generator yielding MagicMock
    objects with .to_dict() returning controlled dicts. Verify normalization."""

    # Controlled raw dicts that Telethon's msg.to_dict() would produce
    raw_msgs = [
        {
            "id": 1,
            "date": "2025-01-01 10:00:00+00:00",
            "message": "First",
            "from_id": {"_": "PeerUser", "user_id": 100},
            "fwd_from": None,
            "reply_to": None,
            "media": None,
        },
        {
            "id": 2,
            "date": "2025-01-01 11:00:00+00:00",
            "message": "Second",
            "from_id": None,
            "fwd_from": {"from_name": "Fwd Channel"},
            "reply_to": {"reply_to_msg_id": 1},
            "media": {"_": "MessageMediaPhoto"},
        },
    ]

    # Build MagicMock objects with .to_dict() returning our controlled dicts
    mock_msgs = []
    for raw in raw_msgs:
        m = MagicMock()
        m.to_dict.return_value = raw
        mock_msgs.append(m)

    # Mock client with iter_messages as an async generator
    async def mock_iter_messages(channel, min_id=0, limit=None):
        for m in mock_msgs:
            yield m

    mock_client = MagicMock()
    mock_client.iter_messages = mock_iter_messages

    results = asyncio.run(extraction_v2.fetch_messages(
        mock_client, channel_id=1234567890, min_id=0
    ))

    assert len(results) == 2
    assert results[0]["id"] == 1
    assert results[0]["from_id"] == {"user_id": 100}
    assert results[0]["has_media"] is False
    assert results[1]["id"] == 2
    assert results[1]["from_id"] == {}
    assert results[1]["fwd_from"] == {"from_name": "Fwd Channel"}
    assert results[1]["reply_to"] == {"reply_to_msg_id": 1}
    assert results[1]["has_media"] is True


# ---------------------------------------------------------------------------
# Task 3: run_extraction_v2 tests
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_config():
    """Fake config dict for _load_config monkeypatching."""
    return {
        "api_id": "12345",
        "api_hash": "abcdef123456",
        "source_channel": "-1009876543210",
        "source_channel_name": "test-channel",
    }


@pytest.fixture
def mock_telegram_client():
    """Mock TelegramClient so async block works without real Telegram."""
    mock_client_instance = MagicMock()
    mock_client_instance.start = AsyncMock()
    mock_client_instance.disconnect = AsyncMock()
    mock_constructor = MagicMock(return_value=mock_client_instance)
    return mock_constructor, mock_client_instance


def test_run_extraction_v2_incremental_uses_max_id(tmp_path, mock_config, mock_telegram_client):
    """Creates a DB with one message (id=5000), sets incremental=True,
    mocks fetch_messages to capture min_id, verifies min_id=5000."""
    db_path = str(tmp_path / "test.db")
    db.init_db(db_path)

    # Insert one message with id=5000
    seed_msgs = [{
        "id": 5000,
        "date": "2025-01-01 10:00:00+00:00",
        "from_id": {"user_id": "alice"},
        "message": "Seed message",
        "fwd_from": None,
        "reply_to": None,
        "has_media": False,
    }]
    with db.connect(db_path) as conn:
        db.upsert_messages(conn, seed_msgs)

    captured_min_id = {}

    async def mock_fetch_messages(client, channel_id, min_id=0, limit=None):
        captured_min_id["min_id"] = min_id
        return []

    mock_constructor, mock_client_instance = mock_telegram_client

    with patch.object(extraction_v2, "_load_config", return_value=mock_config), \
         patch.object(extraction_v2, "fetch_messages", side_effect=mock_fetch_messages), \
         patch.object(extraction_v2, "TelegramClient", mock_constructor):
        result = run_extraction_v2_with_path(db_path, incremental=True, output_dir=str(tmp_path))

    assert captured_min_id["min_id"] == 5000
    assert result["incremental"] is True
    assert result["messages_fetched"] == 0


def test_run_extraction_v2_full_uses_min_id_zero(tmp_path, mock_config, mock_telegram_client):
    """Sets incremental=False, mocks fetch_messages, verifies min_id=0."""
    db_path = str(tmp_path / "test.db")
    db.init_db(db_path)

    # Insert a message so the DB is non-empty (should still use min_id=0)
    seed_msgs = [{
        "id": 5000,
        "date": "2025-01-01 10:00:00+00:00",
        "from_id": {"user_id": "alice"},
        "message": "Seed message",
        "fwd_from": None,
        "reply_to": None,
        "has_media": False,
    }]
    with db.connect(db_path) as conn:
        db.upsert_messages(conn, seed_msgs)

    captured_min_id = {}

    async def mock_fetch_messages(client, channel_id, min_id=0, limit=None):
        captured_min_id["min_id"] = min_id
        return []

    mock_constructor, mock_client_instance = mock_telegram_client

    with patch.object(extraction_v2, "_load_config", return_value=mock_config), \
         patch.object(extraction_v2, "fetch_messages", side_effect=mock_fetch_messages), \
         patch.object(extraction_v2, "TelegramClient", mock_constructor):
        result = run_extraction_v2_with_path(db_path, incremental=False, output_dir=str(tmp_path))

    assert captured_min_id["min_id"] == 0
    assert result["incremental"] is False
    assert result["messages_fetched"] == 0


def test_run_extraction_v2_writes_to_sqlite(tmp_path, mock_config, mock_telegram_client):
    """Mocks fetch_messages to return 2 messages, verifies they're written
    to SQLite via db.load_messages()."""
    db_path = str(tmp_path / "test.db")
    db.init_db(db_path)

    fake_messages = [
        {
            "id": 100,
            "date": "2025-01-01 10:00:00+00:00",
            "from_id": {"user_id": 200},
            "message": "First message",
            "fwd_from": None,
            "reply_to": None,
            "has_media": False,
        },
        {
            "id": 101,
            "date": "2025-01-01 11:00:00+00:00",
            "from_id": {"user_id": 201},
            "message": "Second message",
            "fwd_from": {"from_name": "Orig Channel"},
            "reply_to": {"reply_to_msg_id": 100},
            "has_media": True,
        },
    ]

    async def mock_fetch_messages(client, channel_id, min_id=0, limit=None):
        return fake_messages

    mock_constructor, mock_client_instance = mock_telegram_client

    with patch.object(extraction_v2, "_load_config", return_value=mock_config), \
         patch.object(extraction_v2, "fetch_messages", side_effect=mock_fetch_messages), \
         patch.object(extraction_v2, "TelegramClient", mock_constructor):
        result = run_extraction_v2_with_path(db_path, incremental=False, output_dir=str(tmp_path))

    assert result["messages_fetched"] == 2

    # Verify messages were written to SQLite
    loaded = db.load_messages(db_path)
    assert len(loaded) == 2
    assert loaded[0]["id"] == 100
    assert loaded[0]["message"] == "First message"
    assert loaded[1]["id"] == 101
    assert loaded[1]["fwd_from"] == {"from_name": "Orig Channel"}
    assert loaded[1]["reply_to"] == {"reply_to_msg_id": 100}
    assert loaded[1]["has_media"] is True


def run_extraction_v2_with_path(db_path, incremental=True, output_dir="data"):
    """Helper that calls run_extraction_v2 with explicit db_path.

    This avoids relying on db.DEFAULT_DB_PATH — passes db_path positionally.
    """
    return extraction_v2.run_extraction_v2(
        db_path=db_path,
        incremental=incremental,
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# Task 4: main() CLI tests
# ---------------------------------------------------------------------------

def test_cli_incremental_flag(monkeypatch):
    """sys.argv=["extraction_v2", "--incremental"], mock run_extraction_v2,
    verify incremental=True."""
    captured = {}

    def mock_run_extraction_v2(**kwargs):
        captured.update(kwargs)
        return {"messages_fetched": 0, "min_id": 0, "incremental": kwargs.get("incremental", True)}

    monkeypatch.setattr(sys, "argv", ["extraction_v2", "--incremental"])
    with patch.object(extraction_v2, "run_extraction_v2", side_effect=mock_run_extraction_v2):
        extraction_v2.main()

    assert captured.get("incremental") is True


def test_cli_full_flag(monkeypatch):
    """sys.argv=["extraction_v2", "--full"], mock run_extraction_v2,
    verify incremental=False."""
    captured = {}

    def mock_run_extraction_v2(**kwargs):
        captured.update(kwargs)
        return {"messages_fetched": 0, "min_id": 0, "incremental": kwargs.get("incremental", True)}

    monkeypatch.setattr(sys, "argv", ["extraction_v2", "--full"])
    with patch.object(extraction_v2, "run_extraction_v2", side_effect=mock_run_extraction_v2):
        extraction_v2.main()

    assert captured.get("incremental") is False
