"""Tests for Phase 8 — digest output."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from alcuinus.output import (
    connection_insight,
    emerging_themes,
    format_digest,
    influential_links,
    load_data,
    run_output,
    top_topics,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_cluster_data():
    return {
        "k": 3,
        "clusters": {
            "0": {
                "label": "cluster_0",
                "keywords": ["hakko", "soldering", "tip", "fx", "8801"],
                "size": 6,
                "bundle_ids": [1, 2, 3],
                "chunk_ids": ["c1", "c2"],
            },
            "1": {
                "label": "cluster_1",
                "keywords": ["python", "lcd", "library", "micro", "py"],
                "size": 12,
                "bundle_ids": [4, 5, 6, 7],
                "chunk_ids": ["c3", "c4"],
            },
            "2": {
                "label": "cluster_2",
                "keywords": ["chatgpt", "gpt", "cheat", "sheet"],
                "size": 19,
                "bundle_ids": [8, 9, 10, 11, 12],
                "chunk_ids": ["c5", "c6"],
            },
        },
        "assignments": {"c1": "0", "c2": "0", "c3": "1", "c4": "1", "c5": "2", "c6": "2"},
    }


def _make_decay_data():
    return {
        "classifications": {
            "0": {
                "decay_profile": "ephemeral",
                "cluster_label": "cluster_0",
                "keywords": ["hakko", "soldering"],
                "size": 6,
                "bundle_count": 3,
            },
            "1": {
                "decay_profile": "evergreen",
                "cluster_label": "cluster_1",
                "keywords": ["python", "lcd"],
                "size": 12,
                "bundle_count": 4,
            },
            "2": {
                "decay_profile": "semi-stable",
                "cluster_label": "cluster_2",
                "keywords": ["chatgpt", "gpt"],
                "size": 19,
                "bundle_count": 5,
            },
        }
    }


def _make_bundles():
    return [
        {
            "anchor": {
                "msg_id": 1,
                "urls": ["https://example.com/hakko-tip"],
                "text_preview": "hakko t18-d16 soldering tip for fx-8801",
            },
            "reactions": [{"text_preview": "nice"}, {"text_preview": "need one"}, {"text_preview": "pricey"}],
        },
        {
            "anchor": {
                "msg_id": 4,
                "urls": ["https://github.com/dhylands/python_lcd"],
                "text_preview": "Python LCD library for MicroPython",
            },
            "reactions": [{"text_preview": "useful"}, {"text_preview": "testing"}],
        },
        {
            "anchor": {
                "msg_id": 8,
                "urls": ["https://example.com/chatgpt-sheet"],
                "text_preview": "ChatGPT cheat sheet",
            },
            "reactions": [{"text_preview": "links"}],
        },
    ]


# ---------------------------------------------------------------------------
# top_topics
# ---------------------------------------------------------------------------


class TestTopTopics:
    def test_returns_top_n_by_bundle_count(self):
        clusters = {
            "0": {"bundle_ids": [1], "size": 5, "keywords": ["a"]},
            "1": {"bundle_ids": [1, 2, 3], "size": 10, "keywords": ["b"]},
            "2": {"bundle_ids": [1, 2], "size": 7, "keywords": ["c"]},
        }
        decay = {"0": {"decay_profile": "ephemeral"}, "1": {"decay_profile": "evergreen"}, "2": {"decay_profile": "semi-stable"}}

        result = top_topics(clusters, decay, top_n=2)
        assert len(result) == 2
        assert result[0]["cluster_id"] == "1"  # 3 bundles
        assert result[0]["decay_profile"] == "evergreen"
        assert result[1]["cluster_id"] == "2"  # 2 bundles

    def test_missing_decay_defaults_to_semi_stable(self):
        clusters = {"0": {"bundle_ids": [1], "size": 1, "keywords": ["a"]}}
        decay = {}
        result = top_topics(clusters, decay)
        assert result[0]["decay_profile"] == "semi-stable"


# ---------------------------------------------------------------------------
# emerging_themes
# ---------------------------------------------------------------------------


class TestEmergingThemes:
    def test_returns_ephemeral_only(self):
        clusters = {
            "0": {"bundle_ids": [1, 2], "size": 10, "keywords": ["a"]},
            "1": {"bundle_ids": [1], "size": 5, "keywords": ["b"]},
            "2": {"bundle_ids": [1, 2, 3], "size": 15, "keywords": ["c"]},
        }
        decay = {
            "0": {"decay_profile": "ephemeral"},
            "1": {"decay_profile": "ephemeral"},
            "2": {"decay_profile": "evergreen"},
        }

        result = emerging_themes(clusters, decay, top_n=2)
        assert len(result) == 2
        for r in result:
            assert r["cluster_id"] in ("0", "1")

    def test_empty_when_no_ephemeral(self):
        clusters = {"0": {"bundle_ids": [1], "size": 5, "keywords": ["a"]}}
        decay = {"0": {"decay_profile": "evergreen"}}
        result = emerging_themes(clusters, decay)
        assert result == []


# ---------------------------------------------------------------------------
# influential_links
# ---------------------------------------------------------------------------


class TestInfluentialLinks:
    def test_finds_most_discussed_per_cluster(self):
        clusters = {
            "0": {
                "bundle_ids": [1, 2],
                "keywords": ["hakko"],
                "size": 6,
                "chunk_ids": [],
            },
        }
        bundles = [
            {
                "anchor": {"msg_id": 1, "urls": ["https://a.com"], "text_preview": "A"},
                "reactions": [{"x": 1}],
            },
            {
                "anchor": {"msg_id": 2, "urls": ["https://b.com"], "text_preview": "B"},
                "reactions": [{}, {}, {}],  # 3 reactions — more influential
            },
        ]
        link_meta = {"https://a.com": {"title": "A"}, "https://b.com": {"title": "B"}}

        result = influential_links(clusters, bundles, link_meta, top_n=5)
        assert len(result) >= 1
        assert result[0]["title"] == "B"  # most reactions
        assert result[0]["reactions"] == 3

    def test_handles_missing_meta(self):
        clusters = {"0": {"bundle_ids": [1], "keywords": [], "size": 1, "chunk_ids": []}}
        bundles = [{"anchor": {"msg_id": 1, "urls": ["https://missing.com"], "text_preview": "X"}, "reactions": []}]

        result = influential_links(clusters, bundles, {})
        assert result[0]["title"] == "https://missing.com"


# ---------------------------------------------------------------------------
# format_digest
# ---------------------------------------------------------------------------


class TestFormatDigest:
    def test_renders_complete_digest(self):
        topics = [
            {"cluster_id": "1", "keywords": ["python", "lcd"], "bundle_count": 4, "chunk_count": 12, "decay_profile": "evergreen"},
            {"cluster_id": "0", "keywords": ["hakko", "soldering"], "bundle_count": 3, "chunk_count": 6, "decay_profile": "ephemeral"},
        ]
        emerging = [
            {"cluster_id": "0", "keywords": ["hakko", "soldering"], "bundle_count": 3},
        ]
        links = [
            {"cluster_id": "1", "url": "https://gh.com", "title": "python_lcd", "reactions": 5, "anchor_text": "..."},
        ]
        insight = "The soldering cluster connects with Python LCD — both are DIY electronics."

        digest = format_digest(topics, emerging, links, insight)

        assert "Weekly Digest" in digest
        assert "python" in digest
        assert "hakko" in digest
        assert "Evergreen" in digest or "🟢" in digest
        assert "python_lcd" in digest
        assert insight in digest
        assert "AlcuinusBot" in digest


# ---------------------------------------------------------------------------
# connection_insight (mocked)
# ---------------------------------------------------------------------------


class TestConnectionInsight:
    def test_generates_insight(self):
        clusters = {
            "0": {"keywords": ["hakko", "soldering"], "size": 6},
            "1": {"keywords": ["python", "lcd"], "size": 12},
        }
        decay = {
            "0": {"decay_profile": "ephemeral"},
            "1": {"decay_profile": "evergreen"},
        }

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "The soldering cluster connects with Python LCD."

        with patch("mistralai.client.Mistral") as MockClient:
            instance = MockClient.return_value
            instance.chat.complete.return_value = mock_response

            result = connection_insight(clusters, decay, "test-key")

        assert "soldering" in result


# ---------------------------------------------------------------------------
# run_output — I/O round-trip
# ---------------------------------------------------------------------------


class TestRunOutput:
    def test_roundtrip_no_insight(self, tmp_path):
        """Generate digest without LLM call (insight disabled)."""

        clusters_data = _make_cluster_data()
        decay_data = _make_decay_data()
        bundles = _make_bundles()

        # Write temp files
        c_path = tmp_path / "clusters.json"
        c_path.write_text(json.dumps(clusters_data))
        d_path = tmp_path / "decay.json"
        d_path.write_text(json.dumps(decay_data))
        b_path = tmp_path / "bundles.json"
        b_path.write_text(json.dumps(bundles))
        m_path = tmp_path / "meta.json"
        m_path.write_text(json.dumps([
            {"url": "https://example.com/hakko-tip", "title": "Hakko T18-D16", "description": "Soldering tip"},
            {"url": "https://github.com/dhylands/python_lcd", "title": "python_lcd", "description": "LCD library"},
            {"url": "https://example.com/chatgpt-sheet", "title": "ChatGPT Sheet", "description": "Cheat sheet"},
        ]))
        out_path = tmp_path / "digest.txt"

        result = run_output(
            clusters_path=str(c_path),
            decay_path=str(d_path),
            bundles_path=str(b_path),
            metadata_path=str(m_path),
            output_path=str(out_path),
            generate_insight=False,
        )

        assert result == str(out_path)
        assert out_path.exists()

        content = out_path.read_text()
        assert "Weekly Digest" in content
        assert "hakko" in content
        assert "python" in content
        assert "AlcuinusBot" in content
