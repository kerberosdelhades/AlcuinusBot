"""Tests for Phase 9 — syllabus generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alcuinus.syllabus import (
    build_syllabus_sections,
    format_syllabus,
    run_syllabus,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_clusters():
    return {
        "k": 2,
        "clusters": {
            "0": {
                "label": "cluster_0",
                "keywords": ["python", "lcd", "library"],
                "size": 12,
                "bundle_ids": [1, 2],
                "chunk_ids": ["c1", "c2"],
            },
            "1": {
                "label": "cluster_1",
                "keywords": ["hakko", "soldering", "tip"],
                "size": 6,
                "bundle_ids": [3],
                "chunk_ids": ["c3"],
            },
        },
        "assignments": {"c1": "0", "c2": "0", "c3": "1"},
    }


def _make_decay():
    return {
        "classifications": {
            "0": {"decay_profile": "evergreen", "cluster_label": "cluster_0", "keywords": ["python", "lcd"], "size": 12, "bundle_count": 2},
            "1": {"decay_profile": "ephemeral", "cluster_label": "cluster_1", "keywords": ["hakko", "soldering"], "size": 6, "bundle_count": 1},
        }
    }


def _make_bundles():
    return [
        {
            "anchor": {
                "msg_id": 1,
                "urls": ["https://github.com/dhylands/python_lcd"],
                "text_preview": "Python LCD library",
            },
            "reactions": [{"x": 1}, {"x": 2}, {"x": 3}],
        },
        {
            "anchor": {
                "msg_id": 2,
                "urls": ["https://github.com/dhylands/python_lcd/blob/master/lcd/lcd_api.py"],
                "text_preview": "LCD API",
            },
            "reactions": [{"x": 1}],
        },
        {
            "anchor": {
                "msg_id": 3,
                "urls": ["https://www.batterfly.com/shop/en/hakko-t18-d16"],
                "text_preview": "Hakko soldering tip",
            },
            "reactions": [{"x": 1}, {"x": 2}],
        },
    ]


def _make_metadata():
    return [
        {"url": "https://github.com/dhylands/python_lcd", "title": "dhylands/python_lcd", "description": "Python based library for talking to character based LCDs."},
        {"url": "https://github.com/dhylands/python_lcd/blob/master/lcd/lcd_api.py", "title": "dhylands/python_lcd", "description": ""},
        {"url": "https://www.batterfly.com/shop/en/hakko-t18-d16", "title": "Hakko T18-D16", "description": "Soldering tip for FX-8801"},
    ]


# ---------------------------------------------------------------------------
# build_syllabus_sections
# ---------------------------------------------------------------------------


class TestBuildSyllabusSections:
    def test_organizes_by_tier(self):
        clusters = _make_clusters()["clusters"]
        decay = _make_decay()["classifications"]
        bundles = _make_bundles()
        link_meta = {r["url"]: r for r in _make_metadata()}

        result = build_syllabus_sections(clusters, decay, bundles, link_meta)

        assert "evergreen" in result
        assert "semi-stable" in result
        assert "ephemeral" in result

        # Evergreen has the python/lcd cluster
        assert len(result["evergreen"]) == 1
        assert result["evergreen"][0]["keywords"] == ["python", "lcd", "library"]

        # Ephemeral has the hakko cluster
        assert len(result["ephemeral"]) == 1
        assert result["ephemeral"][0]["keywords"] == ["hakko", "soldering", "tip"]

    def test_missing_decay_defaults_to_semi_stable(self):
        clusters = {"0": {"keywords": ["test"], "bundle_ids": [1], "size": 1, "chunk_ids": []}}
        decay = {}
        bundles = [{"anchor": {"msg_id": 1, "urls": ["https://x.com"], "text_preview": "X"}, "reactions": []}]
        link_meta = {}

        result = build_syllabus_sections(clusters, decay, bundles, link_meta)
        assert len(result["semi-stable"]) == 1

    def test_links_sorted_by_reactions(self):
        clusters = {"0": {"keywords": ["test"], "bundle_ids": [4, 5], "size": 2, "chunk_ids": []}}
        decay = {"0": {"decay_profile": "evergreen"}}
        bundles = [
            {"anchor": {"msg_id": 4, "urls": ["https://a.com"], "text_preview": "A"}, "reactions": [{}]},
            {"anchor": {"msg_id": 5, "urls": ["https://b.com"], "text_preview": "B"}, "reactions": [{}, {}, {}]},
        ]
        link_meta = {"https://a.com": {"title": "A"}, "https://b.com": {"title": "B"}}

        result = build_syllabus_sections(clusters, decay, bundles, link_meta)
        links = result["evergreen"][0]["links"]
        assert links[0]["url"] == "https://b.com"  # more reactions first
        assert links[1]["url"] == "https://a.com"


# ---------------------------------------------------------------------------
# format_syllabus
# ---------------------------------------------------------------------------


class TestFormatSyllabus:
    def test_renders_complete_syllabus(self):
        tiered = {
            "evergreen": [
                {
                    "keywords": ["python", "lcd"],
                    "bundle_count": 2,
                    "chunk_count": 12,
                    "links": [
                        {"url": "https://gh.com/py", "title": "python_lcd", "description": "LCD library for Python", "reactions": 5},
                    ],
                },
            ],
            "ephemeral": [
                {
                    "keywords": ["hakko", "soldering"],
                    "bundle_count": 1,
                    "chunk_count": 6,
                    "links": [
                        {"url": "https://shop.com", "title": "Hakko T18", "description": "", "reactions": 2},
                    ],
                },
            ],
            "semi-stable": [],
        }

        output = format_syllabus(tiered)

        assert "Guía de Estudio" in output
        assert "Foundational" in output
        assert "python_lcd" in output
        assert "Hakko T18" in output
        assert "AlcuinusBot" in output
        # Evergreen should appear before ephemeral
        assert output.index("Foundational") < output.index("Recent & ephemeral")


# ---------------------------------------------------------------------------
# run_syllabus — I/O round-trip
# ---------------------------------------------------------------------------


class TestRunSyllabus:
    def test_roundtrip(self, tmp_path):
        c_path = tmp_path / "clusters.json"
        c_path.write_text(json.dumps(_make_clusters()))
        d_path = tmp_path / "decay.json"
        d_path.write_text(json.dumps(_make_decay()))
        b_path = tmp_path / "bundles.json"
        b_path.write_text(json.dumps(_make_bundles()))
        m_path = tmp_path / "meta.json"
        m_path.write_text(json.dumps(_make_metadata()))
        out_path = tmp_path / "syllabus.md"

        result = run_syllabus(
            clusters_path=str(c_path),
            decay_path=str(d_path),
            bundles_path=str(b_path),
            metadata_path=str(m_path),
            output_path=str(out_path),
        )

        assert result == str(out_path)
        assert out_path.exists()

        content = out_path.read_text()
        assert "python_lcd" in content
        assert "Hakko T18" in content
        assert "Foundational" in content
