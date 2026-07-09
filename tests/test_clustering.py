"""Tests for Phase 6 — bundle clustering."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zvec

from alcuinus.clustering import (
    DEFAULT_K,
    TOP_KEYWORDS,
    _get_stopwords,
    cluster_vectors,
    extract_keywords,
    fetch_vectors_from_zvec,
    run_clustering,
)
from alcuinus.embedding import build_schema, create_index

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_vectors(n: int = 20, dim: int = 1024) -> dict[str, dict]:
    """Create synthetic vectors for testing."""
    rng = np.random.RandomState(42)
    vectors = {}
    for i in range(n):
        vec = rng.randn(dim).tolist()
        vectors[f"chunk_{i}"] = {
            "vector": vec,
            "text": f"This is chunk number {i} about topic {i % 4}",
            "bundle_anchor_id": i // 3,
            "is_parent": i % 2 == 0,
        }
    return vectors


def _make_zvec_with_vectors(tmp_path: Path, n: int = 20) -> str:
    """Create a Zvec index with synthetic vectors."""
    index_path = str(tmp_path / "test_index")
    schema = build_schema()
    collection = create_index(index_path, schema)

    rng = np.random.RandomState(42)
    docs = []
    for i in range(n):
        doc = zvec.Doc(
            id=f"chunk_{i}",
            vectors={"embedding": rng.randn(1024).tolist()},
            fields={
                "text": f"This is chunk number {i} about topic {i % 4}",
                "bundle_anchor_id": i // 3,
                "is_parent": i % 2 == 0,
                "token_estimate": 10 + i,
            },
        )
        docs.append(doc)

    collection.insert(docs)
    collection.flush()
    return index_path


# ---------------------------------------------------------------------------
# cluster_vectors
# ---------------------------------------------------------------------------


class TestClusterVectors:
    def test_basic_clustering(self):
        vectors = _make_vectors(20)
        result = cluster_vectors(vectors, k=4)

        assert result["k"] == 4
        assert len(result["clusters"]) == 4
        assert len(result["assignments"]) == 20

        # All chunks assigned
        assigned_ids = set(result["assignments"].keys())
        assert assigned_ids == set(vectors.keys())

    def test_cluster_sizes_sum_to_total(self):
        vectors = _make_vectors(20)
        result = cluster_vectors(vectors, k=4)

        total = sum(c["size"] for c in result["clusters"].values())
        assert total == 20

    def test_k_auto_reduced_when_small_data(self):
        vectors = _make_vectors(3)
        result = cluster_vectors(vectors, k=10)

        # K should be auto-reduced to 1 (3 // 2 = 1)
        assert result["k"] == 1

    def test_single_cluster(self):
        vectors = _make_vectors(5)
        result = cluster_vectors(vectors, k=1)

        assert result["k"] == 1
        assert len(result["clusters"]) == 1
        assert len(result["assignments"]) == 5


# ---------------------------------------------------------------------------
# extract_keywords
# ---------------------------------------------------------------------------


class TestExtractKeywords:
    def test_basic_keywords(self):
        vectors = {
            "c1": {"text": "machine learning deep neural networks training"},
            "c2": {"text": "machine learning classification algorithms supervised"},
            "c3": {"text": "web development javascript react frontend"},
            "c4": {"text": "web development css html responsive design"},
        }
        clusters = {
            "0": {"chunk_ids": ["c1", "c2"], "size": 2, "label": "cluster_0"},
            "1": {"chunk_ids": ["c3", "c4"], "size": 2, "label": "cluster_1"},
        }

        keywords = extract_keywords(clusters, vectors, top_n=3)

        assert "cluster_0" in keywords
        assert "cluster_1" in keywords
        # ML cluster should have ML-related keywords
        assert any("machine" in k or "learning" in k for k in keywords["cluster_0"])
        # Web cluster should have web-related keywords
        assert any("web" in k or "development" in k or "javascript" in k for k in keywords["cluster_1"])

    def test_empty_cluster(self):
        clusters = {
            "0": {"chunk_ids": [], "size": 0, "label": "cluster_0"},
        }
        keywords = extract_keywords(clusters, {}, top_n=3)
        assert keywords == {}

    def test_stopwords_filtered(self):
        vectors = {
            "c1": {"text": "the the the the python is great"},
            "c2": {"text": "the the the the python programming language"},
        }
        clusters = {
            "0": {"chunk_ids": ["c1", "c2"], "size": 2, "label": "cluster_0"},
        }

        keywords = extract_keywords(clusters, vectors, top_n=3)
        # "the" should not appear in keywords
        assert "the" not in keywords.get("cluster_0", [])


# ---------------------------------------------------------------------------
# _get_stopwords
# ---------------------------------------------------------------------------


class TestGetStopwords:
    def test_returns_set(self):
        sw = _get_stopwords()
        assert isinstance(sw, set)
        assert len(sw) > 50

    def test_includes_common_words(self):
        sw = _get_stopwords()
        assert "the" in sw
        assert "https" in sw
        assert "el" in sw
        assert "de" in sw


# ---------------------------------------------------------------------------
# fetch_vectors_from_zvec (real Zvec index)
# ---------------------------------------------------------------------------


class TestFetchVectorsFromZvec:
    def test_fetches_all_docs(self, tmp_path):
        index_path = _make_zvec_with_vectors(tmp_path, n=10)
        result = fetch_vectors_from_zvec(index_path)

        assert len(result) == 10
        for chunk_id, info in result.items():
            assert "vector" in info
            assert len(info["vector"]) == 1024
            assert "text" in info
            assert "bundle_anchor_id" in info

    def test_empty_index(self, tmp_path):
        index_path = str(tmp_path / "empty_index")
        schema = build_schema()
        collection = create_index(index_path, schema)
        collection.flush()
        del collection

        result = fetch_vectors_from_zvec(index_path)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# run_clustering — I/O round-trip
# ---------------------------------------------------------------------------


class TestRunClustering:
    def test_roundtrip(self, tmp_path):
        # Create Zvec index with synthetic data
        index_path = _make_zvec_with_vectors(tmp_path, n=20)

        # Create matching chunks.json
        chunks = [
            {
                "chunk_id": f"chunk_{i}",
                "text": f"This is chunk number {i} about topic {i % 4}",
                "bundle_anchor_id": i // 3,
                "is_parent": i % 2 == 0,
                "token_estimate": 10 + i,
            }
            for i in range(20)
        ]
        chunks_path = tmp_path / "chunks.json"
        chunks_path.write_text(json.dumps(chunks), encoding="utf-8")

        out_path = tmp_path / "clusters.json"

        result = run_clustering(
            index_path=index_path,
            chunks_path=str(chunks_path),
            output_path=str(out_path),
            k=4,
        )

        assert result == str(out_path)
        assert out_path.exists()

        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["k"] == 4
        assert len(data["clusters"]) == 4
        assert len(data["assignments"]) == 20

        # Each cluster has keywords and bundle_ids
        for cluster_key, info in data["clusters"].items():
            assert "keywords" in info
            assert "bundle_ids" in info
            assert len(info["bundle_ids"]) > 0
