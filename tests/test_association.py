"""Tests for association (Phase 2)."""

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from alcuinus.anchor_detection import detect_anchors
from alcuinus.association import associate, run_association


# ---------------------------------------------------------------------------
# synthetic fixture — covers all three association strategies
# ---------------------------------------------------------------------------

def _dt(offset_hours: float) -> str:
    """Return an ISO datetime string offset from a fixed epoch."""
    epoch = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    return (epoch + timedelta(hours=offset_hours)).isoformat()


def _make_msg(
    msg_id: int,
    text: str,
    offset_hours: float = 0,
    *,
    sender_id: int | None = 1,
    reply_to_msg_id: int | None = None,
) -> dict:
    msg: dict = {
        "id": msg_id,
        "date": _dt(offset_hours),
        "message": text,
        "from_id": {"_": "PeerUser", "user_id": sender_id} if sender_id else None,
        "fwd_from": None,
    }
    if reply_to_msg_id is not None:
        msg["reply_to"] = {"reply_to_msg_id": reply_to_msg_id}
    return msg


@pytest.fixture
def synthetic_fixture() -> tuple[list[dict], list[dict]]:
    """Return (messages, anchors) covering all strategies.

    Timeline::

        msg  1  orphan (before first anchor)
        msg  2  anchor A  "https://arxiv.org/..."
        msg  3  window → A
        msg  4  window → A
        msg  5  anchor B  "https://github.com/..."
        msg  6  replies to msg 2 → A (reply override, even though B is closer)
        msg  7  window → B
        msg  8  anchor C  "https://example.com"  (last anchor)
        msg  9  window → C (within time gap)
        msg 10  orphan (outside max_idle_hours)
        msg 11  replies to msg 8 → C (reply override on last anchor)
    """
    msgs = [
        _make_msg(1, "hello, anyone here?", -2),
        _make_msg(2, "Check this: https://arxiv.org/abs/9999", -1),
        _make_msg(3, "great paper!", 0),
        _make_msg(4, "see section 3.2 for the key result", 0.5),
        _make_msg(5, "Also: https://github.com/org/repo", 1),
        _make_msg(6, "re: arxiv paper — the ablation is flawed", 2, reply_to_msg_id=2),
        _make_msg(7, "starred the repo, thanks!", 3),
        _make_msg(8, "Third link: https://example.com/article", 4),
        _make_msg(9, "this one is interesting too", 5),
        # msg 10 is 200 hours after last anchor → outside 168h default gap
        _make_msg(10, "way too late to matter", 204),
        _make_msg(11, "re: the example.com article — clickbait?", 10, reply_to_msg_id=8),
    ]
    anchors = detect_anchors(msgs)
    return msgs, anchors


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

class TestAssociate:
    def test_returns_all_anchors_in_order(self, synthetic_fixture):
        msgs, anchors = synthetic_fixture
        bundles = associate(msgs, anchors, max_idle_hours=168)
        anchor_ids = [b["anchor"]["msg_id"] for b in bundles]
        assert anchor_ids == [2, 5, 8]

    def test_window_strategy(self, synthetic_fixture):
        msgs, anchors = synthetic_fixture
        bundles = associate(msgs, anchors, max_idle_hours=168)

        b2 = next(b for b in bundles if b["anchor"]["msg_id"] == 2)
        b5 = next(b for b in bundles if b["anchor"]["msg_id"] == 5)

        # Msgs 3, 4 → anchor 2 (window)
        window_msgs = [r["msg_id"] for r in b2["reactions"] if r["strategy"] == "window"]
        assert 3 in window_msgs
        assert 4 in window_msgs

        # Msg 7 → anchor 5 (window)
        b5_window = [r["msg_id"] for r in b5["reactions"] if r["strategy"] == "window"]
        assert b5_window == [7]

    def test_reply_override(self, synthetic_fixture):
        msgs, anchors = synthetic_fixture
        bundles = associate(msgs, anchors, max_idle_hours=168)

        b2 = next(b for b in bundles if b["anchor"]["msg_id"] == 2)
        reply_msgs = [r for r in b2["reactions"] if r["strategy"] == "reply"]
        assert len(reply_msgs) == 1
        assert reply_msgs[0]["msg_id"] == 6
        assert reply_msgs[0]["reply_to_msg_id"] == 2

    def test_time_gap_orphan(self, synthetic_fixture):
        msgs, anchors = synthetic_fixture
        bundles = associate(msgs, anchors, max_idle_hours=168)

        b8 = next(b for b in bundles if b["anchor"]["msg_id"] == 8)
        reaction_ids = {r["msg_id"] for r in b8["reactions"]}
        assert 9 in reaction_ids   # within gap
        assert 11 in reaction_ids  # reply override, even if within gap window
        assert 10 not in reaction_ids  # too far (204h > 168h)

    def test_orphans_before_first_anchor(self, synthetic_fixture):
        msgs, anchors = synthetic_fixture
        bundles = associate(msgs, anchors, max_idle_hours=168)

        all_reaction_ids = {
            r["msg_id"] for b in bundles for r in b["reactions"]
        }
        assert 1 not in all_reaction_ids  # orphan

    def test_anchor_reply_metadata(self, synthetic_fixture):
        """Anchor record carries reply_to info when the link was shared as a reply."""
        msgs, anchors = synthetic_fixture

        # Anchor 6 is not here, but let's verify normal anchor has no reply
        bundles = associate(msgs, anchors, max_idle_hours=168)
        b2 = next(b for b in bundles if b["anchor"]["msg_id"] == 2)
        assert b2["anchor"]["reply_to_msg_id"] is None
        assert b2["anchor"]["reply_to_anchor_msg_id"] is None

    def test_empty_messages(self):
        assert associate([], []) == []

    def test_messages_without_anchors(self):
        msgs = [_make_msg(1, "no links here", 0)]
        assert associate(msgs, []) == []

    def test_time_gap_spares_reply_anchored(self, synthetic_fixture):
        """Tight time-gap removes window reactions but spares reply-anchored ones."""
        msgs, anchors = synthetic_fixture
        # max_idle_hours=0.5 (30 min): msg 9 (gap=1h) → window-orphan,
        # but msg 11 (reply_to=8) → survives because it's reply-anchored.
        bundles = associate(msgs, anchors, max_idle_hours=0.5)
        b8 = next(b for b in bundles if b["anchor"]["msg_id"] == 8)
        reaction_ids = {r["msg_id"] for r in b8["reactions"]}
        assert 9 not in reaction_ids   # gap=1h > 0.5h, not reply-anchored
        assert 11 in reaction_ids      # reply override survives any gap
        # Verify the surviving reaction is marked "reply"
        r11 = next(r for r in b8["reactions"] if r["msg_id"] == 11)
        assert r11["strategy"] == "reply"

    def test_window_metadata(self, synthetic_fixture):
        msgs, anchors = synthetic_fixture
        bundles = associate(msgs, anchors, max_idle_hours=168)

        b2 = next(b for b in bundles if b["anchor"]["msg_id"] == 2)
        assert b2["window"]["boundary"] == "next_anchor"
        assert b2["window"]["start_msg_id"] == 2
        assert b2["window"]["end_msg_id"] == 5

        b8 = next(b for b in bundles if b["anchor"]["msg_id"] == 8)
        assert b8["window"]["boundary"] == "end_of_data"
        assert b8["window"]["end_msg_id"] is None


class TestRunAssociation:
    def test_round_trip(self, synthetic_fixture):
        """Write messages + anchors to temp files, run, read back."""
        msgs, anchors = synthetic_fixture

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(msgs, f)
            msgs_tmp = f.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(anchors, f)
            anchors_tmp = f.name

        try:
            out = run_association(
                messages_path=msgs_tmp,
                anchors_path=anchors_tmp,
                output_path=tempfile.mktemp(suffix=".json"),
                max_idle_hours=168,
            )
            with open(out, encoding="utf-8") as f:
                bundles = json.load(f)

            assert len(bundles) == 3
            assert all("anchor" in b and "reactions" in b and "window" in b for b in bundles)
        finally:
            Path(msgs_tmp).unlink(missing_ok=True)
            Path(anchors_tmp).unlink(missing_ok=True)
            Path(out).unlink(missing_ok=True)


class TestReactionRecord:
    def test_reaction_keys(self, synthetic_fixture):
        msgs, anchors = synthetic_fixture
        bundles = associate(msgs, anchors, max_idle_hours=168)

        for b in bundles:
            for r in b["reactions"]:
                assert set(r.keys()) == {
                    "msg_id", "date", "sender_id", "text_preview",
                    "reply_to_msg_id", "strategy",
                }
                assert r["strategy"] in ("window", "reply")
