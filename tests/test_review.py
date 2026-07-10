"""Tests for Phase 10 — review cycle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alcuinus.review import (
    check_staleness,
    load_current_state,
    load_snapshot,
    run_review,
    save_snapshot,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_decay():
    return {
        "classified_at": "2026-01-10 00:00:00",
        "classifications": {
            "0": {
                "decay_profile": "ephemeral",
                "cluster_label": "cluster_0",
                "keywords": ["hakko", "soldering"],
                "size": 6,
                "bundle_count": 3,
            },
            "1": {
                "decay_profile": "semi-stable",
                "cluster_label": "cluster_1",
                "keywords": ["python", "lcd"],
                "size": 12,
                "bundle_count": 4,
            },
            "2": {
                "decay_profile": "evergreen",
                "cluster_label": "cluster_2",
                "keywords": ["chatgpt", "gpt"],
                "size": 19,
                "bundle_count": 5,
            },
        },
    }


def _make_clusters():
    return {
        "k": 3,
        "clusters": {
            "0": {"keywords": ["hakko"], "size": 6, "bundle_ids": [1], "chunk_ids": []},
            "1": {"keywords": ["python"], "size": 12, "bundle_ids": [2], "chunk_ids": []},
            "2": {"keywords": ["chatgpt"], "size": 19, "bundle_ids": [3], "chunk_ids": []},
        },
        "assignments": {},
    }


# ---------------------------------------------------------------------------
# check_staleness
# ---------------------------------------------------------------------------


class TestCheckStaleness:
    def test_ephemeral_past_retention_flagged(self):
        decay = _make_decay()
        clusters = _make_clusters()
        # classified_at is 2026-01-10, ephemeral retention is 3 months
        # It's now 2026-07 — that's ~6 months > 3 months → flagged
        report = check_staleness(decay, clusters)

        assert report["clusters_total"] == 3
        assert report["summary"]["remove_ephemeral"] >= 1

        ephemeral_flags = [f for f in report["flagged"] if f["cluster_id"] == "0"]
        assert len(ephemeral_flags) >= 1
        assert ephemeral_flags[0]["action"] == "remove_ephemeral"

    def test_semi_stable_past_retention_flagged(self):
        decay = {
            "classified_at": "2023-01-10 00:00:00",
            "classifications": {
                "0": {
                    "decay_profile": "semi-stable",
                    "cluster_label": "cluster_0",
                    "keywords": ["old"],
                    "size": 5,
                    "bundle_count": 1,
                },
            },
        }
        clusters = {"k": 1, "clusters": {"0": {"keywords": ["old"], "size": 5, "bundle_ids": [], "chunk_ids": []}}, "assignments": {}}
        report = check_staleness(decay, clusters)

        assert report["summary"]["demote_semi_stable"] >= 1

    def test_evergreen_not_flagged_within_threshold(self):
        decay = _make_decay()
        clusters = _make_clusters()
        report = check_staleness(decay, clusters)

        # Evergreen should NOT be flagged (only 6 months since classification)
        evergreen_flags = [f for f in report["flagged"] if f["cluster_id"] == "2"]
        assert len(evergreen_flags) == 0

    def test_evergreen_flagged_after_24_months(self):
        decay = {
            "classified_at": "2023-01-10 00:00:00",
            "classifications": {
                "0": {
                    "decay_profile": "evergreen",
                    "cluster_label": "cluster_0",
                    "keywords": ["ancient"],
                    "size": 10,
                    "bundle_count": 2,
                },
            },
        }
        clusters = {"k": 1, "clusters": {"0": {"keywords": ["ancient"], "size": 10, "bundle_ids": [], "chunk_ids": []}}, "assignments": {}}
        report = check_staleness(decay, clusters)

        # ~42 months > 24 months → flagged for review
        assert report["summary"]["promote_evergreen"] >= 1

    def test_no_stale_items(self):
        decay = {
            "classified_at": "2026-07-01 00:00:00",  # 9 days ago
            "classifications": {
                "0": {
                    "decay_profile": "semi-stable",
                    "cluster_label": "cluster_0",
                    "keywords": ["fresh"],
                    "size": 5,
                    "bundle_count": 1,
                },
            },
        }
        clusters = {"k": 1, "clusters": {"0": {"keywords": ["fresh"], "size": 5, "bundle_ids": [], "chunk_ids": []}}, "assignments": {}}
        report = check_staleness(decay, clusters)

        assert report["flagged"] == []
        assert report["summary"]["no_change"] == 1


# ---------------------------------------------------------------------------
# snapshot save/load
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_save_and_load(self, tmp_path):
        decay = _make_decay()
        path = tmp_path / "snapshot.json"

        save_snapshot(decay, str(path))
        assert path.exists()

        loaded = load_snapshot(str(path))
        assert loaded is not None
        assert loaded["total_clusters"] == 3
        assert "0" in loaded["profiles_snapshot"]

    def test_load_missing_returns_none(self, tmp_path):
        result = load_snapshot(str(tmp_path / "nonexistent.json"))
        assert result is None


# ---------------------------------------------------------------------------
# run_review — I/O round-trip
# ---------------------------------------------------------------------------


class TestRunReview:
    def test_roundtrip(self, tmp_path):
        decay = {
            "classified_at": "2026-01-10 00:00:00",
            "classifications": {
                "0": {
                    "decay_profile": "ephemeral",
                    "cluster_label": "cluster_0",
                    "keywords": ["test"],
                    "size": 3,
                    "bundle_count": 1,
                },
            },
        }
        clusters = {
            "k": 1,
            "clusters": {"0": {"keywords": ["test"], "size": 3, "bundle_ids": [], "chunk_ids": []}},
            "assignments": {},
        }

        # Need minimal data files for regenerate
        bundles = [{"anchor": {"msg_id": 1, "urls": [], "text_preview": ""}, "reactions": []}]
        meta = [{"url": "https://x.com", "title": "X"}]

        d_path = tmp_path / "decay.json"
        c_path = tmp_path / "clusters.json"
        b_path = tmp_path / "bundles.json"
        m_path = tmp_path / "meta.json"
        s_path = tmp_path / "snapshot.json"
        out_path = tmp_path / "review.json"

        d_path.write_text(json.dumps(decay))
        c_path.write_text(json.dumps(clusters))
        b_path.write_text(json.dumps(bundles))
        m_path.write_text(json.dumps(meta))

        result = run_review(
            decay_path=str(d_path),
            clusters_path=str(c_path),
            output_path=str(out_path),
            snapshot_path=str(s_path),
            regenerate=False,
        )

        assert result == str(out_path)
        assert out_path.exists()

        report = json.loads(out_path.read_text())
        assert report["clusters_total"] == 1
        assert "flagged" in report
        assert "summary" in report

        # Snapshot should have been saved
        assert s_path.exists()
