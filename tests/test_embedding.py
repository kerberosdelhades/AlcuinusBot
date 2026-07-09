"""Tests for Phase 5 — embedding + Zvec storage."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import zvec

from alcuinus.embedding import (
    BATCH_SIZE,
    DEFAULT_INDEX_PATH,
    EMBED_DIM,
    EMBED_MODEL,
    build_schema,
    create_index,
    embed_texts,
    embed_and_store,
    open_index,
    run_embedding,
)


# ---------------------------------------------------------------------------
# build_schema
# ---------------------------------------------------------------------------


class TestBuildSchema:
    def test_returns_schema(self):
        schema = build_schema()
        assert isinstance(schema, zvec.CollectionSchema)

    def test_has_embedding_vector(self):
        schema = build_schema()
        # Just verify it doesn't raise
        assert schema is not None


# ---------------------------------------------------------------------------
# create_index / open_index
# ---------------------------------------------------------------------------


class TestCreateIndex:
    def test_creates_directory(self, tmp_path):
        index_path = str(tmp_path / "test_index")
        schema = build_schema()
        collection = create_index(index_path, schema)
        assert Path(index_path).exists()
        collection.destroy()

    def test_overwrites_existing(self, tmp_path):
        index_path = str(tmp_path / "test_index")
        schema = build_schema()

        c1 = create_index(index_path, schema)
        c1.destroy()

        c2 = create_index(index_path, schema)
        assert Path(index_path).exists()
        c2.destroy()


class TestOpenIndex:
    def test_opens_existing(self, tmp_path):
        index_path = str(tmp_path / "test_index")
        schema = build_schema()

        c1 = create_index(index_path, schema)
        c1.flush()
        del c1  # release lock

        c2 = open_index(index_path)
        assert c2 is not None
        c2.destroy()

    def test_raises_on_missing(self, tmp_path):
        with pytest.raises(Exception):
            open_index(str(tmp_path / "nonexistent"))


# ---------------------------------------------------------------------------
# embed_texts (mocked)
# ---------------------------------------------------------------------------


class TestEmbedTexts:
    def test_single_batch(self):
        """Mock the Mistral API to return controlled embeddings."""
        mock_result = MagicMock()
        mock_result.data = [
            MagicMock(embedding=[0.1] * EMBED_DIM, index=0),
            MagicMock(embedding=[0.2] * EMBED_DIM, index=1),
        ]

        with patch("alcuinus.embedding.Mistral") as MockClient:
            instance = MockClient.return_value
            instance.embeddings.create.return_value = mock_result

            result = embed_texts(
                ["text one", "text two"],
                api_key="test-key",
            )

        assert len(result) == 2
        assert len(result[0]) == EMBED_DIM
        assert result[0][0] == pytest.approx(0.1)
        assert result[1][0] == pytest.approx(0.2)

    def test_multiple_batches(self):
        """With batch_size=2 and 5 inputs, should make 3 API calls."""
        call_count = 0

        def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            batch = kwargs.get("inputs", [])
            result = MagicMock()
            result.data = [
                MagicMock(embedding=[float(i)] * EMBED_DIM, index=i)
                for i in range(len(batch))
            ]
            return result

        with patch("alcuinus.embedding.Mistral") as MockClient:
            instance = MockClient.return_value
            instance.embeddings.create.side_effect = mock_create

            result = embed_texts(
                ["a", "b", "c", "d", "e"],
                api_key="test-key",
                batch_size=2,
            )

        assert len(result) == 5
        assert call_count == 3  # ceil(5/2) = 3

    def test_none_embedding_raises(self):
        mock_result = MagicMock()
        mock_result.data = [MagicMock(embedding=None, index=0)]

        with patch("alcuinus.embedding.Mistral") as MockClient:
            instance = MockClient.return_value
            instance.embeddings.create.return_value = mock_result

            with pytest.raises(ValueError, match="None"):
                embed_texts(["text"], api_key="test-key")

    def test_empty_input(self):
        result = embed_texts([], api_key="test-key")
        assert result == []


# ---------------------------------------------------------------------------
# embed_and_store (mocked API + real Zvec)
# ---------------------------------------------------------------------------


class TestEmbedAndStore:
    @pytest.fixture
    def sample_chunks(self, tmp_path):
        chunks = [
            {
                "chunk_id": f"chunk_{i}",
                "text": f"This is chunk number {i} with some text content.",
                "bundle_anchor_id": i,
                "is_parent": i % 2 == 0,
                "token_estimate": 10 + i,
            }
            for i in range(5)
        ]
        path = tmp_path / "chunks.json"
        path.write_text(json.dumps(chunks), encoding="utf-8")
        return str(path), chunks

    def test_full_pipeline(self, tmp_path, sample_chunks):
        chunks_path, chunks = sample_chunks
        index_path = str(tmp_path / "index")

        # Mock Mistral API
        mock_result = MagicMock()
        mock_result.data = [
            MagicMock(embedding=[float(i) / 10] * EMBED_DIM, index=i)
            for i in range(len(chunks))
        ]

        with patch("alcuinus.embedding.Mistral") as MockClient:
            instance = MockClient.return_value
            instance.embeddings.create.return_value = mock_result

            result = embed_and_store(
                chunks_path=chunks_path,
                index_path=index_path,
                api_key="test-key",
            )

        assert result == index_path
        assert Path(index_path).exists()

        # Verify we can open and query the index
        collection = open_index(index_path)
        # Fetch all by ID
        ids = [c["chunk_id"] for c in chunks]
        fetched = collection.fetch(ids=ids, include_vector=True)
        assert len(fetched) == len(chunks)
        collection.destroy()

    def test_raises_on_empty_chunks(self, tmp_path):
        empty_path = tmp_path / "empty.json"
        empty_path.write_text("[]", encoding="utf-8")

        with pytest.raises(ValueError, match="No chunks"):
            embed_and_store(chunks_path=str(empty_path), api_key="test-key")

    def test_raises_on_missing_api_key(self, tmp_path, sample_chunks):
        chunks_path, _ = sample_chunks

        with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
            embed_and_store(
                chunks_path=chunks_path,
                index_path=str(tmp_path / "idx"),
                api_key="",
            )
