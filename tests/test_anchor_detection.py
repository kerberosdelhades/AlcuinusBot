"""Tests for anchor detection."""

import json
import tempfile
from pathlib import Path

import pytest

from alcuinus.anchor_detection import build_anchor, detect_anchors, extract_urls


class TestExtractUrls:
    def test_single_url(self):
        assert extract_urls("Check out https://arxiv.org/abs/2312.00752") == [
            "https://arxiv.org/abs/2312.00752"
        ]

    def test_multiple_urls(self):
        text = "See https://github.com/X and also https://arxiv.org/abs/42"
        assert extract_urls(text) == [
            "https://github.com/X",
            "https://arxiv.org/abs/42",
        ]

    def test_no_url(self):
        assert extract_urls("Just some text") == []
        assert extract_urls("") == []

    def test_skips_non_http(self):
        assert extract_urls("ftp://files.example.com") == []

    def test_handles_none_input(self):
        # extract_urls expects a str — caller guards, but test the contract
        with pytest.raises(TypeError):
            extract_urls(None)  # type: ignore[arg-type]


class TestBuildAnchor:
    def test_returns_anchor_for_url_message(self):
        msg = {
            "id": 42,
            "date": "2025-01-01 12:00:00+00:00",
            "message": "Paper: https://arxiv.org/abs/9999",
            "from_id": {"_": "PeerUser", "user_id": 123},
            "fwd_from": None,
        }
        anchor = build_anchor(msg)
        assert anchor is not None
        assert anchor["msg_id"] == 42
        assert anchor["sender_id"] == 123
        assert anchor["urls"] == ["https://arxiv.org/abs/9999"]
        assert anchor["forwarded_from"] is None

    def test_returns_none_for_no_url(self):
        msg = {
            "id": 1,
            "date": "2025-01-01 12:00:00+00:00",
            "message": "Hello world",
            "from_id": None,
            "fwd_from": None,
        }
        assert build_anchor(msg) is None

    def test_forwarded_from_name(self):
        msg = {
            "id": 7,
            "date": "2025-01-01 12:00:00+00:00",
            "message": "https://example.com/article",
            "from_id": None,
            "fwd_from": {"from_name": "Alice"},
        }
        anchor = build_anchor(msg)
        assert anchor is not None
        assert anchor["forwarded_from"] == "Alice"

    def test_truncates_long_text(self):
        msg = {
            "id": 1,
            "date": "2025-01-01 12:00:00+00:00",
            "message": "https://x.com " + ("bla " * 200),
            "from_id": None,
            "fwd_from": None,
        }
        anchor = build_anchor(msg)
        assert anchor is not None
        assert len(anchor["text_preview"]) <= 300

    def test_empty_message_field(self):
        msg = {
            "id": 1,
            "date": "2025-01-01 12:00:00+00:00",
            "from_id": None,
            "fwd_from": None,
        }
        assert build_anchor(msg) is None


class TestDetectAnchors:
    def test_empty_list(self):
        assert detect_anchors([]) == []

    def test_finds_all_anchors(self):
        msgs = [
            {
                "id": 1,
                "date": "2025-01-01 12:00:00+00:00",
                "message": "hello",
                "from_id": None,
                "fwd_from": None,
            },
            {
                "id": 2,
                "date": "2025-01-01 12:01:00+00:00",
                "message": "https://github.com/X/super-repo",
                "from_id": {"_": "PeerUser", "user_id": 10},
                "fwd_from": None,
            },
            {
                "id": 3,
                "date": "2025-01-01 12:02:00+00:00",
                "message": "https://arxiv.org/abs/123",
                "from_id": {"_": "PeerUser", "user_id": 20},
                "fwd_from": None,
            },
            {
                "id": 4,
                "date": "2025-01-01 12:03:00+00:00",
                "message": "thanks!",
                "from_id": None,
                "fwd_from": None,
            },
        ]
        anchors = detect_anchors(msgs)
        assert len(anchors) == 2
        assert [a["msg_id"] for a in anchors] == [2, 3]

    def test_sorted_by_msg_id(self):
        msgs = [
            {
                "id": 30,
                "date": "2025-01-01 12:00:00+00:00",
                "message": "https://b.com",
                "from_id": None,
                "fwd_from": None,
            },
            {
                "id": 10,
                "date": "2025-01-01 12:00:00+00:00",
                "message": "https://a.com",
                "from_id": None,
                "fwd_from": None,
            },
        ]
        anchors = detect_anchors(msgs)
        assert [a["msg_id"] for a in anchors] == [10, 30]


class TestIntegration:
    def test_round_trip(self):
        """Detect anchors from real messages, verify JSON round-trips."""
        data_dir = Path(__file__).resolve().parent.parent / "data"
        messages_path = data_dir / "channel_messages.json"

        if not messages_path.exists():
            pytest.skip("No channel_messages.json to test against")

        with open(messages_path, encoding="utf-8") as f:
            messages = json.load(f)

        anchors = detect_anchors(messages)

        # Every anchor must have the expected keys
        required_keys = {"msg_id", "date", "sender_id", "forwarded_from", "urls", "text_preview"}
        for a in anchors:
            assert required_keys == set(a.keys()), f"Missing keys in anchor {a['msg_id']}"
            assert isinstance(a["urls"], list) and len(a["urls"]) > 0
            assert all(u.startswith("http") for u in a["urls"])

        # Write to temp and read back
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(anchors, f, indent=2, ensure_ascii=False)
            tmp = f.name

        try:
            with open(tmp, encoding="utf-8") as f:
                reloaded = json.load(f)
            assert reloaded == anchors
        finally:
            Path(tmp).unlink()

    def test_matches_expected_url_count(self):
        """Smoke test: anchor count should be >= messages with 'http'."""
        data_dir = Path(__file__).resolve().parent.parent / "data"
        messages_path = data_dir / "channel_messages.json"

        if not messages_path.exists():
            pytest.skip("No channel_messages.json to test against")

        with open(messages_path, encoding="utf-8") as f:
            messages = json.load(f)

        httpish = sum(
            1 for m in messages
            if m.get("message") and "http" in m["message"]
        )
        anchors = detect_anchors(messages)

        # Some http-ish messages may not contain valid http:// URLs
        # (e.g., "http" in a code snippet). So anchors ≤ httpish.
        assert len(anchors) <= httpish
        assert len(anchors) >= 1  # We know there are many
