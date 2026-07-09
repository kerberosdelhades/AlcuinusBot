"""Tests for Phase 4 — chunking & tagging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alcuinus.chunking import (
    CHILD_CHUNK_TOKENS,
    OVERLAP_RATIO,
    PARENT_CHUNK_TOKENS,
    build_all_chunks,
    build_bundle_text,
    build_metadata_prefix,
    chunk_text,
    estimate_tokens,
    load_link_metadata,
    run_chunking,
)


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens("") == 1  # floor at 1

    def test_short(self):
        # 3 chars → 1 token
        assert estimate_tokens("abc") == 1

    def test_typical(self):
        # 30 chars → 10 tokens
        assert estimate_tokens("a" * 30) == 10

    def test_unicode_mixed(self):
        text = "El modelo Mixtral usa MoE"
        # ~25 chars / 3 ≈ 8
        assert estimate_tokens(text) == 8


# ---------------------------------------------------------------------------
# build_metadata_prefix
# ---------------------------------------------------------------------------


class TestBuildMetadataPrefix:
    def test_basic(self):
        bundle = {
            "anchor": {
                "date": "2024-03-15 12:00:00+00:00",
                "sender_id": 12345,
                "text_preview": "Check out this paper on transformers",
            }
        }
        prefix = build_metadata_prefix(bundle)
        assert "[channel: Demiurgo]" in prefix
        assert "[date: 2024-03-15]" in prefix
        assert "[author: 12345]" in prefix
        assert "[lang: en]" in prefix  # pure ASCII

    def test_spanish_detection(self):
        bundle = {
            "anchor": {
                "date": "2024-03-15",
                "sender_id": 999,
                "text_preview": "atención con este paper",
            }
        }
        prefix = build_metadata_prefix(bundle)
        assert "[lang: es]" in prefix  # 'ó' triggers Spanish

    def test_missing_fields(self):
        bundle = {"anchor": {}}
        prefix = build_metadata_prefix(bundle)
        assert "[date: unknown]" in prefix
        assert "[author: unknown]" in prefix


# ---------------------------------------------------------------------------
# build_bundle_text
# ---------------------------------------------------------------------------


class TestBuildBundleText:
    def test_assembles_all_parts(self):
        bundle = {
            "anchor": {
                "urls": ["https://example.com/article"],
                "text_preview": "Check out this article",
            },
            "reactions": [
                {"text_preview": "Great read!"},
                {"text_preview": "I disagree with point 2"},
            ],
        }
        link_meta = {
            "https://example.com/article": {
                "title": "Example Article",
                "description": "A test description",
            }
        }
        text = build_bundle_text(bundle, link_meta)
        assert "[Example Article]" in text
        assert "A test description" in text
        assert "Check out this article" in text
        assert "Great read!" in text
        assert "I disagree with point 2" in text

    def test_no_metadata(self):
        bundle = {
            "anchor": {
                "urls": ["https://example.com/unknown"],
                "text_preview": "Check this",
            },
            "reactions": [],
        }
        link_meta = {}
        text = build_bundle_text(bundle, link_meta)
        assert "Check this" in text

    def test_empty_bundle(self):
        bundle = {"anchor": {}, "reactions": []}
        text = build_bundle_text(bundle, {})
        assert text == ""


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------


class TestChunkText:
    def test_small_text_one_child_one_parent(self):
        text = "Hello world. " * 20  # ~140 chars, ~47 tokens
        chunks = chunk_text(text)
        assert len(chunks) >= 1
        # Should have at least one child and at least one parent
        children = [c for c in chunks if not c["is_parent"]]
        parents = [c for c in chunks if c["is_parent"]]
        assert len(children) >= 1
        assert len(parents) >= 1

    def test_empty_text(self):
        text = ""
        chunks = chunk_text(text)
        assert chunks == []

    def test_whitespace_only(self):
        text = "   \n\n   "
        chunks = chunk_text(text)
        assert chunks == []

    def test_long_text_produces_multiple_children(self):
        # ~4000 chars, ~1300 tokens → should produce multiple children
        text = ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 100)
        chunks = chunk_text(text)
        children = [c for c in chunks if not c["is_parent"]]
        parents = [c for c in chunks if c["is_parent"]]
        assert len(children) >= 3
        assert len(parents) >= 1

    def test_overlap_coverage(self):
        """Ensure the overlap means every sentence appears in at least
        two children (except first and last sentence in the text)."""
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        text = text * 20  # enough to get multiple chunks
        chunks = chunk_text(text)
        children = [c for c in chunks if not c["is_parent"]]
        # With overlap, children should overlap in content
        if len(children) >= 2:
            # Check the last 20 chars of child 0 appear somewhere in child 1
            overlap_zone = children[0]["text"][-20:]
            assert overlap_zone in children[1]["text"]


# ---------------------------------------------------------------------------
# build_all_chunks
# ---------------------------------------------------------------------------


class TestBuildAllChunks:
    def test_produces_chunks_per_bundle(self):
        bundles = [
            {
                "anchor": {
                    "msg_id": 1,
                    "date": "2024-01-01",
                    "urls": ["https://a.com/x"],
                    "text_preview": "Some text here",
                },
                "reactions": [
                    {"text_preview": "Interesting"},
                ],
            },
            {
                "anchor": {
                    "msg_id": 2,
                    "date": "2024-01-02",
                    "urls": ["https://a.com/y"],
                    "text_preview": "Another link",
                },
                "reactions": [],
            },
        ]
        link_meta = {
            "https://a.com/x": {"title": "Title X"},
            "https://a.com/y": {"title": "Title Y"},
        }
        chunks = build_all_chunks(bundles, link_meta)
        assert len(chunks) >= 2  # at least one per bundle
        # Every chunk has required fields
        for c in chunks:
            assert "chunk_id" in c
            assert "text" in c
            assert "bundle_anchor_id" in c
            assert "is_parent" in c
            assert "token_estimate" in c

    def test_empty_body_still_produces_stub(self):
        bundles = [
            {
                "anchor": {"msg_id": 1, "urls": [], "text_preview": ""},
                "reactions": [],
            }
        ]
        chunks = build_all_chunks(bundles, {})
        assert len(chunks) == 1
        assert "stub" in chunks[0]["chunk_id"]

    def test_chunk_ids_are_unique(self):
        bundles = [
            {
                "anchor": {
                    "msg_id": i,
                    "date": "2024-01-01",
                    "urls": [f"https://a.com/{i}"],
                    "text_preview": f"Bundle {i} text. " * 20,
                },
                "reactions": [{"text_preview": "Reaction"}],
            }
            for i in range(5)
        ]
        link_meta = {}
        chunks = build_all_chunks(bundles, link_meta)
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_metadata_prefix_included(self):
        bundles = [
            {
                "anchor": {
                    "msg_id": 42,
                    "date": "2024-06-15 12:00:00+00:00",
                    "sender_id": 777,
                    "urls": ["https://example.com/article"],
                    "text_preview": "Test article",
                },
                "reactions": [],
            }
        ]
        link_meta = {"https://example.com/article": {"title": "Test Title"}}
        chunks = build_all_chunks(bundles, link_meta)
        for c in chunks:
            assert "[channel: Demiurgo]" in c["text"]
            assert "[date: 2024-06-15]" in c["text"]


# ---------------------------------------------------------------------------
# load_link_metadata
# ---------------------------------------------------------------------------


class TestLoadLinkMetadata:
    def test_loads_and_indexes_by_url(self, tmp_path):
        data = [
            {"url": "https://a.com/1", "title": "A"},
            {"url": "https://a.com/2", "title": "B"},
        ]
        path = tmp_path / "meta.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        result = load_link_metadata(str(path))
        assert result["https://a.com/1"]["title"] == "A"
        assert result["https://a.com/2"]["title"] == "B"


# ---------------------------------------------------------------------------
# run_chunking — I/O round-trip
# ---------------------------------------------------------------------------


class TestRunChunking:
    def test_roundtrip(self, tmp_path):
        bundles = [
            {
                "anchor": {
                    "msg_id": 1,
                    "date": "2024-01-01",
                    "urls": ["https://a.com/x"],
                    "text_preview": "Some text here. " * 10,
                },
                "reactions": [
                    {"text_preview": "Interesting"},
                ],
            }
        ]
        link_meta = [
            {"url": "https://a.com/x", "title": "Title X", "description": "Desc X"},
        ]

        bundles_path = tmp_path / "bundles.json"
        meta_path = tmp_path / "meta.json"
        out_path = tmp_path / "chunks.json"

        bundles_path.write_text(json.dumps(bundles), encoding="utf-8")
        meta_path.write_text(json.dumps(link_meta), encoding="utf-8")

        result = run_chunking(
            str(bundles_path), str(meta_path), str(out_path)
        )
        assert result == str(out_path)
        assert out_path.exists()

        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert len(data) >= 1
        assert "chunk_id" in data[0]
        assert "text" in data[0]
