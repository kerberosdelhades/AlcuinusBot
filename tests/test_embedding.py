"""Tests for Phase 5 — embedding + Zvec storage."""

from __future__ import annotations

import hashlib
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
    compute_embedding_delta,
    create_index,
    embed_and_store,
    embed_and_store_incremental,
    embed_texts,
    get_indexed_chunk_ids,
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


# ---------------------------------------------------------------------------
# get_indexed_chunk_ids
# ---------------------------------------------------------------------------


class TestGetIndexedChunkIds:
    def test_get_indexed_chunk_ids(self, tmp_path):
        """Create a Zvec index with 5 docs, probe with 10 IDs, assert only 5 returned."""
        index_path = str(tmp_path / "test_index")
        schema = build_schema()
        collection = create_index(index_path, schema)

        # Insert 5 docs (chunk_0 through chunk_4)
        docs = []
        for i in range(5):
            doc = zvec.Doc(
                id=f"chunk_{i}",
                vectors={"embedding": [float(i)] * EMBED_DIM},
                fields={
                    "text": f"content {i}",
                    "bundle_anchor_id": i,
                    "is_parent": i % 2 == 0,
                    "token_estimate": 10 + i,
                },
            )
            docs.append(doc)
        collection.insert(docs)
        collection.flush()
        del collection  # release lock so get_indexed_chunk_ids can reopen

        # Probe with 10 IDs (chunk_0 through chunk_9)
        probe_ids = [f"chunk_{i}" for i in range(10)]
        result = get_indexed_chunk_ids(index_path, probe_ids)

        assert len(result) == 5
        assert result == {f"chunk_{i}" for i in range(5)}

        # Clean up
        c = open_index(index_path)
        c.destroy()


# ---------------------------------------------------------------------------
# compute_embedding_delta
# ---------------------------------------------------------------------------


class TestComputeEmbeddingDelta:
    def test_compute_embedding_delta_all_new(self):
        """All chunks new, no index — everything needs embedding."""
        chunk_ids = ["chunk_0", "chunk_1", "chunk_2"]
        chunk_hashes = {"chunk_0": "h0", "chunk_1": "h1", "chunk_2": "h2"}
        indexed_ids: set[str] = set()
        indexed_hashes: dict[str, str] = {}

        result = compute_embedding_delta(
            chunk_ids, chunk_hashes, indexed_ids, indexed_hashes
        )

        assert result["to_embed"] == ["chunk_0", "chunk_1", "chunk_2"]
        assert result["to_delete"] == set()
        assert result["unchanged"] == set()

    def test_compute_embedding_delta_no_changes(self):
        """All hashes match — nothing to do."""
        chunk_ids = ["chunk_0", "chunk_1", "chunk_2"]
        chunk_hashes = {"chunk_0": "h0", "chunk_1": "h1", "chunk_2": "h2"}
        indexed_ids = {"chunk_0", "chunk_1", "chunk_2"}
        indexed_hashes = {"chunk_0": "h0", "chunk_1": "h1", "chunk_2": "h2"}

        result = compute_embedding_delta(
            chunk_ids, chunk_hashes, indexed_ids, indexed_hashes
        )

        assert result["to_embed"] == []
        assert result["to_delete"] == set()
        assert result["unchanged"] == {"chunk_0", "chunk_1", "chunk_2"}

    def test_compute_embedding_delta_text_changed(self):
        """One chunk hash differs — only that chunk needs re-embedding."""
        chunk_ids = ["chunk_0", "chunk_1", "chunk_2"]
        chunk_hashes = {"chunk_0": "h0", "chunk_1": "h1_changed", "chunk_2": "h2"}
        indexed_ids = {"chunk_0", "chunk_1", "chunk_2"}
        indexed_hashes = {"chunk_0": "h0", "chunk_1": "h1", "chunk_2": "h2"}

        result = compute_embedding_delta(
            chunk_ids, chunk_hashes, indexed_ids, indexed_hashes
        )

        assert result["to_embed"] == ["chunk_1"]
        assert result["to_delete"] == set()
        assert result["unchanged"] == {"chunk_0", "chunk_2"}

    def test_compute_embedding_delta_stale_chunk(self):
        """Chunk in index but not in SQLite — should be marked for deletion."""
        chunk_ids = ["chunk_0", "chunk_1"]
        chunk_hashes = {"chunk_0": "h0", "chunk_1": "h1"}
        indexed_ids = {"chunk_0", "chunk_1", "chunk_2"}
        indexed_hashes = {"chunk_0": "h0", "chunk_1": "h1", "chunk_2": "h2"}

        result = compute_embedding_delta(
            chunk_ids, chunk_hashes, indexed_ids, indexed_hashes
        )

        assert result["to_embed"] == []
        assert result["to_delete"] == {"chunk_2"}
        assert result["unchanged"] == {"chunk_0", "chunk_1"}


# ---------------------------------------------------------------------------
# embed_and_store_incremental (mocked API + real Zvec + real SQLite)
# ---------------------------------------------------------------------------


class TestEmbedAndStoreIncremental:
    @pytest.fixture
    def sample_db(self, tmp_path):
        """Create a real SQLite DB with 5 anchors + 5 chunks, return db_path."""
        from alcuinus import db

        db_path = str(tmp_path / "test.db")
        db.init_db(db_path)

        anchors = [
            {
                "msg_id": i,
                "date": "2025-01-01",
                "sender_id": "1",
                "urls": [],
                "text_preview": "test",
            }
            for i in range(5)
        ]
        chunks = [
            {
                "chunk_id": f"chunk_{i}",
                "text": f"content {i}",
                "bundle_anchor_id": i,
                "is_parent": i % 2 == 0,
                "token_estimate": 10 + i,
            }
            for i in range(5)
        ]

        # We need messages for FK constraints on anchors
        messages = [
            {
                "id": i,
                "date": "2025-01-01",
                "from_id": {"user_id": 1},
                "message": f"msg {i}",
                "fwd_from": None,
                "reply_to": None,
                "has_media": False,
            }
            for i in range(5)
        ]

        with db.connect(db_path) as conn:
            db.upsert_messages(conn, messages)
            db.upsert_anchors(conn, anchors)
            db.upsert_chunks(conn, chunks)

        return db_path, chunks

    def _make_mock_result(self, n):
        """Create a mock Mistral response with n embeddings."""
        mock_result = MagicMock()
        mock_result.data = [
            MagicMock(embedding=[float(i) / 10] * EMBED_DIM, index=i)
            for i in range(n)
        ]
        return mock_result

    def test_first_run_creates_index(self, tmp_path, sample_db):
        """No existing index → full embed, verify 5 chunks in index."""
        db_path, chunks = sample_db
        index_path = str(tmp_path / "index")

        mock_result = self._make_mock_result(5)

        with patch("alcuinus.embedding.Mistral") as MockClient:
            instance = MockClient.return_value
            instance.embeddings.create.return_value = mock_result

            result = embed_and_store_incremental(
                db_path=db_path,
                index_path=index_path,
                api_key="test-key",
            )

        assert result == index_path
        assert Path(index_path).exists()

        # Verify 5 chunks in index
        collection = open_index(index_path)
        ids = [c["chunk_id"] for c in chunks]
        fetched = collection.fetch(ids=ids, include_vector=True)
        assert len(fetched) == 5
        collection.destroy()

    def test_second_run_no_changes(self, tmp_path, sample_db):
        """Second run → API not called, chunks still there."""
        db_path, chunks = sample_db
        index_path = str(tmp_path / "index")

        mock_result = self._make_mock_result(5)

        with patch("alcuinus.embedding.Mistral") as MockClient:
            instance = MockClient.return_value
            instance.embeddings.create.return_value = mock_result

            # First run — creates index
            embed_and_store_incremental(
                db_path=db_path,
                index_path=index_path,
                api_key="test-key",
            )

        # Second run — no changes, API should NOT be called
        with patch("alcuinus.embedding.Mistral") as MockClient:
            instance = MockClient.return_value
            instance.embeddings.create.return_value = mock_result

            result = embed_and_store_incremental(
                db_path=db_path,
                index_path=index_path,
                api_key="test-key",
            )

            # API should not have been called
            instance.embeddings.create.assert_not_called()

        assert result == index_path

        # Verify chunks still there
        collection = open_index(index_path)
        ids = [c["chunk_id"] for c in chunks]
        fetched = collection.fetch(ids=ids, include_vector=False)
        assert len(fetched) == 5
        collection.destroy()

    def test_second_run_with_text_change(self, tmp_path, sample_db):
        """Change one chunk's text, second run → API called once with 1 text."""
        from alcuinus import db

        db_path, chunks = sample_db
        index_path = str(tmp_path / "index")

        mock_result = self._make_mock_result(5)

        with patch("alcuinus.embedding.Mistral") as MockClient:
            instance = MockClient.return_value
            instance.embeddings.create.return_value = mock_result

            # First run — creates index
            embed_and_store_incremental(
                db_path=db_path,
                index_path=index_path,
                api_key="test-key",
            )

        # Change chunk_2's text in SQLite
        with db.connect(db_path) as conn:
            conn.execute(
                "UPDATE chunks SET text = ?, text_hash = ? WHERE chunk_id = ?",
                ("content 2 modified",
                 hashlib.sha256(b"content 2 modified").hexdigest(),
                 "chunk_2"),
            )

        # Second run — should embed only chunk_2
        call_inputs = []

        def mock_create(**kwargs):
            call_inputs.append(kwargs.get("inputs", []))
            result = MagicMock()
            result.data = [
                MagicMock(embedding=[0.5] * EMBED_DIM, index=i)
                for i in range(len(kwargs.get("inputs", [])))
            ]
            return result

        with patch("alcuinus.embedding.Mistral") as MockClient:
            instance = MockClient.return_value
            instance.embeddings.create.side_effect = mock_create

            result = embed_and_store_incremental(
                db_path=db_path,
                index_path=index_path,
                api_key="test-key",
            )

        assert result == index_path

        # API should have been called once with exactly 1 text
        total_calls = len(call_inputs)
        total_texts = sum(len(c) for c in call_inputs)
        assert total_calls == 1, f"Expected 1 API call, got {total_calls}"
        assert total_texts == 1, f"Expected 1 text embedded, got {total_texts}"

        # Clean up
        collection = open_index(index_path)
        collection.destroy()


# ---------------------------------------------------------------------------
# run_embedding (incremental auto-detect)
# ---------------------------------------------------------------------------


class TestRunEmbeddingIncremental:
    def test_run_embedding_uses_incremental(self, monkeypatch):
        """Monkeypatch embed_and_store_incremental and verify run_embedding
        calls it with force_full=False."""
        captured = {}

        def fake_incremental(*args, **kwargs):
            captured.update(kwargs)
            return "fake/path"

        monkeypatch.setattr(
            "alcuinus.embedding.embed_and_store_incremental",
            fake_incremental,
        )

        result = run_embedding()

        assert result == "fake/path"
        assert captured.get("force_full") is False
