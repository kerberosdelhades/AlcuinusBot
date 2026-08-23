"""End-to-end pipeline test — replaces delete_this.py.

Chains phases 5–10 (embedding → clustering → decay → output → review) using
synthetic data in SQLite, a real Zvec index in tmp_path, a mocked Mistral
client, and the heuristic decay path. No API keys required.

This test deliberately avoids importing anchor_detection, chunking,
association, and extraction (they transitively import pytopicgram/urlextract
which hangs). Instead, synthetic messages/anchors/bundles/chunks are created
directly in SQLite.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import zvec

from alcuinus import db
from alcuinus.embedding import (
    EMBED_DIM,
    build_schema,
    embed_and_store_incremental,
    open_index,
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic data
# ---------------------------------------------------------------------------

NUM_MESSAGES = 20
NUM_ANCHORS = 10
NUM_BUNDLES = 10
NUM_CHUNKS = 20
K = 3


def _make_messages() -> list[dict]:
    """20 synthetic Telegram messages."""
    return [
        {
            "id": i,
            "date": "2025-01-01T10:00:00+00:00",
            "from_id": {"user_id": i},
            "message": f"Message {i}",
            "fwd_from": None,
            "reply_to": None,
            "has_media": False,
        }
        for i in range(NUM_MESSAGES)
    ]


def _make_anchors() -> list[dict]:
    """10 synthetic anchors (messages 0–9 have URLs)."""
    return [
        {
            "msg_id": i,
            "date": "2025-01-01T10:00:00+00:00",
            "sender_id": "1",
            "urls": [f"https://example.com/{i}"],
            "text_preview": f"Link {i}",
        }
        for i in range(NUM_ANCHORS)
    ]


def _make_bundles() -> list[dict]:
    """10 synthetic bundles — one per anchor, with window metadata."""
    return [
        {
            "anchor": {
                "msg_id": i,
                "date": "2025-01-01T10:00:00+00:00",
                "sender_id": "1",
                "urls": [f"https://example.com/{i}"],
                "text_preview": f"Link {i}",
            },
            "reactions": [
                {
                    "msg_id": NUM_ANCHORS + i,
                    "date": "2025-01-01T11:00:00+00:00",
                    "sender_id": "2",
                    "text_preview": f"Reaction to link {i}",
                    "reply_to_msg_id": i,
                    "strategy": "window",
                }
            ],
            "window": {
                "start_msg_id": i,
                "end_msg_id": NUM_MESSAGES - 1 if i == NUM_ANCHORS - 1 else i + 1,
                "boundary": "next_anchor" if i < NUM_ANCHORS - 1 else "end_of_data",
            },
        }
        for i in range(NUM_BUNDLES)
    ]


def _make_chunks() -> list[dict]:
    """20 synthetic chunks — 3 topics for meaningful k=3 clustering."""
    return [
        {
            "chunk_id": f"chunk_{i}",
            "text": f"content about topic {i % 3}",
            "bundle_anchor_id": i % NUM_ANCHORS,
            "is_parent": i % 2 == 0,
            "token_estimate": 10 + i,
        }
        for i in range(NUM_CHUNKS)
    ]


def _make_link_metadata() -> list[dict]:
    """Synthetic link metadata for all anchor URLs."""
    return [
        {
            "url": f"https://example.com/{i}",
            "title": f"Example Page {i}",
            "description": f"Description for link {i}",
            "status": "ok",
            "error": None,
            "fetched_at": "2025-01-01T12:00:00+00:00",
        }
        for i in range(NUM_ANCHORS)
    ]


@pytest.fixture
def pipeline_env(tmp_path: Path) -> dict:
    """Create a full synthetic pipeline environment in tmp_path.

    Sets up:
      - SQLite DB at tmp_path/test.db with messages, anchors, bundles, chunks
      - JSON files in tmp_path for phases that read from JSON:
        chunks.json, bundles.json, link_metadata.json
    Returns a dict of paths and the synthetic data.
    """
    db_path = str(tmp_path / "test.db")
    db.init_db(db_path)

    messages = _make_messages()
    anchors = _make_anchors()
    bundles = _make_bundles()
    chunks = _make_chunks()
    link_meta = _make_link_metadata()

    # Write to SQLite
    with db.connect(db_path) as conn:
        db.upsert_messages(conn, messages)
        db.upsert_anchors(conn, anchors)
        db.upsert_bundles(conn, bundles)
        db.upsert_chunks(conn, chunks)
        db.upsert_link_metadata(conn, link_meta)

    # Write JSON files for phases that read from JSON (decay, output, review)
    chunks_path = tmp_path / "chunks.json"
    chunks_path.write_text(json.dumps(chunks), encoding="utf-8")

    bundles_path = tmp_path / "bundles.json"
    bundles_path.write_text(json.dumps(bundles), encoding="utf-8")

    metadata_path = tmp_path / "link_metadata.json"
    metadata_path.write_text(json.dumps(link_meta), encoding="utf-8")

    # Verify the data landed
    assert db.get_message_count(db_path) == NUM_MESSAGES
    assert len(db.load_anchors(db_path)) == NUM_ANCHORS
    assert len(db.load_bundles(db_path)) == NUM_BUNDLES
    assert len(db.load_chunks(db_path)) == NUM_CHUNKS

    return {
        "tmp_path": tmp_path,
        "db_path": db_path,
        "index_path": str(tmp_path / "zvec_index"),
        "chunks_path": str(chunks_path),
        "bundles_path": str(bundles_path),
        "metadata_path": str(metadata_path),
        "clusters_path": str(tmp_path / "clusters.json"),
        "decay_path": str(tmp_path / "decay_profiles.json"),
        "digest_path": str(tmp_path / "digest.txt"),
        "review_path": str(tmp_path / "review_report.json"),
        "snapshot_path": str(tmp_path / "review_snapshot.json"),
        "messages": messages,
        "anchors": anchors,
        "bundles": bundles,
        "chunks": chunks,
        "link_meta": link_meta,
    }


def _make_mock_embeddings(n: int) -> MagicMock:
    """Create a mock Mistral embeddings response with n 1024d vectors.

    Vectors are designed so chunks with the same topic (i % 3) get similar
    embeddings — this ensures KMeans produces 3 meaningful clusters.
    """
    # Topic-base vectors: 3 distinct directions in 1024d space
    topic_bases = [
        [0.9] * EMBED_DIM,
        [0.1] * EMBED_DIM,
        [0.5] * EMBED_DIM,
    ]

    mock_result = MagicMock()
    mock_result.data = [
        MagicMock(
            embedding=topic_bases[i % 3],
            index=i,
        )
        for i in range(n)
    ]
    return mock_result


# ---------------------------------------------------------------------------
# Phase 5 — Embedding + Zvec
# ---------------------------------------------------------------------------


class TestPhase5Embedding:
    """Phase 5: mock Mistral, run embed_and_store_incremental, verify Zvec index."""

    def test_creates_zvec_index(self, pipeline_env: dict):
        """Mock Mistral → embed all chunks → real Zvec index in tmp_path."""
        env = pipeline_env
        chunks = env["chunks"]
        mock_result = _make_mock_embeddings(len(chunks))

        with patch("alcuinus.embedding.Mistral") as MockClient:
            instance = MockClient.return_value
            instance.embeddings.create.return_value = mock_result

            result = embed_and_store_incremental(
                db_path=env["db_path"],
                index_path=env["index_path"],
                api_key="test-key",
            )

        assert result == env["index_path"]
        assert Path(env["index_path"]).exists()

        # Open the index and verify
        collection = open_index(env["index_path"])
        ids = [c["chunk_id"] for c in chunks]
        fetched = collection.fetch(ids=ids, include_vector=True)

        # All chunks should be in the index
        assert len(fetched) == NUM_CHUNKS, (
            f"Expected {NUM_CHUNKS} docs in Zvec, got {len(fetched)}"
        )

        # All vectors should be 1024d
        for doc_id, doc in fetched.items():
            vec = doc.vector("embedding")
            assert vec is not None, f"doc {doc_id} has None embedding"
            assert len(vec) == EMBED_DIM, (
                f"doc {doc_id} embedding dim {len(vec)} != {EMBED_DIM}"
            )

        collection.destroy()


# ---------------------------------------------------------------------------
# Phase 6 — Clustering
# ---------------------------------------------------------------------------


class TestPhase6Clustering:
    """Phase 6: run clustering against Zvec index + SQLite, verify output."""

    def test_clusters_produced(self, pipeline_env: dict):
        """Run run_clustering with k=3, verify structure and coverage."""
        from alcuinus.clustering import run_clustering

        env = pipeline_env
        chunks = env["chunks"]

        # First create the Zvec index (Phase 5)
        mock_result = _make_mock_embeddings(len(chunks))
        with patch("alcuinus.embedding.Mistral") as MockClient:
            instance = MockClient.return_value
            instance.embeddings.create.return_value = mock_result
            embed_and_store_incremental(
                db_path=env["db_path"],
                index_path=env["index_path"],
                api_key="test-key",
            )

        # Now run clustering
        result = run_clustering(
            index_path=env["index_path"],
            db_path=env["db_path"],
            chunks_path=env["chunks_path"],
            output_path=env["clusters_path"],
            k=K,
        )

        assert result == env["clusters_path"]
        assert Path(env["clusters_path"]).exists()

        # Load and verify the clusters
        with open(env["clusters_path"], encoding="utf-8") as f:
            clusters_data = json.load(f)

        assert clusters_data["k"] == K, f"Expected k={K}, got {clusters_data['k']}"
        assert len(clusters_data["clusters"]) == K, (
            f"Expected {K} clusters, got {len(clusters_data['clusters'])}"
        )

        # All chunks must be assigned
        assignments = clusters_data["assignments"]
        assert len(assignments) == NUM_CHUNKS, (
            f"Expected {NUM_CHUNKS} assignments, got {len(assignments)}"
        )
        assigned_ids = set(assignments.keys())
        expected_ids = {c["chunk_id"] for c in chunks}
        assert assigned_ids == expected_ids, "Assignment IDs don't match chunk IDs"

        # Cluster sizes must sum to total
        total_assigned = sum(c["size"] for c in clusters_data["clusters"].values())
        assert total_assigned == NUM_CHUNKS, (
            f"Total assigned {total_assigned} != {NUM_CHUNKS}"
        )

        # Every cluster must have keywords
        for cluster_key, info in clusters_data["clusters"].items():
            assert "keywords" in info, f"Cluster {cluster_key} missing keywords"
            assert "bundle_ids" in info, f"Cluster {cluster_key} missing bundle_ids"
            assert len(info["bundle_ids"]) > 0, (
                f"Cluster {cluster_key} has no bundle_ids"
            )

        # Verify the clusters file is valid JSON with expected structure
        assert "clusters" in clusters_data
        assert "assignments" in clusters_data
        assert "k" in clusters_data


# ---------------------------------------------------------------------------
# Phase 7 — Decay (heuristic)
# ---------------------------------------------------------------------------


class TestPhase7Decay:
    """Phase 7: run_decay(use_llm=False) — heuristic, no API key."""

    def test_decay_profiles_produced(self, pipeline_env: dict):
        """Run heuristic decay, verify profiles for all clusters."""
        from alcuinus.clustering import run_clustering
        from alcuinus.decay import run_decay

        env = pipeline_env
        chunks = env["chunks"]

        # Phase 5: create Zvec index
        mock_result = _make_mock_embeddings(len(chunks))
        with patch("alcuinus.embedding.Mistral") as MockClient:
            instance = MockClient.return_value
            instance.embeddings.create.return_value = mock_result
            embed_and_store_incremental(
                db_path=env["db_path"],
                index_path=env["index_path"],
                api_key="test-key",
            )

        # Phase 6: clustering
        run_clustering(
            index_path=env["index_path"],
            db_path=env["db_path"],
            chunks_path=env["chunks_path"],
            output_path=env["clusters_path"],
            k=K,
        )

        # Phase 7: decay (heuristic — no LLM, no API key)
        result = run_decay(
            clusters_path=env["clusters_path"],
            chunks_path=env["chunks_path"],
            bundles_path=env["bundles_path"],
            output_path=env["decay_path"],
            use_llm=False,
            db_path=env["db_path"],
        )

        assert result == env["decay_path"]
        assert Path(env["decay_path"]).exists()

        with open(env["decay_path"], encoding="utf-8") as f:
            decay_data = json.load(f)

        # Structural assertions
        assert "classified_at" in decay_data
        assert "method" in decay_data
        assert decay_data["method"] == "heuristic", (
            f"Expected heuristic method, got {decay_data['method']}"
        )
        assert "profiles" in decay_data
        assert "classifications" in decay_data

        # Load clusters to verify 1:1 mapping
        with open(env["clusters_path"], encoding="utf-8") as f:
            clusters_data = json.load(f)
        cluster_keys = set(clusters_data["clusters"].keys())
        profile_keys = set(decay_data["classifications"].keys())
        assert profile_keys == cluster_keys, (
            f"Decay profile keys {profile_keys} != cluster keys {cluster_keys}"
        )

        # Each classification must have valid structure
        valid_profiles = {"evergreen", "semi-stable", "ephemeral"}
        for key, info in decay_data["classifications"].items():
            assert "decay_profile" in info, (
                f"Classification {key} missing decay_profile"
            )
            assert info["decay_profile"] in valid_profiles, (
                f"Classification {key} has invalid profile: {info['decay_profile']}"
            )
            assert "cluster_label" in info
            assert "size" in info
            assert "bundle_count" in info
            assert "keywords" in info
            assert "profile_info" in info


# ---------------------------------------------------------------------------
# Phase 8 — Output (digest, no LLM insight)
# ---------------------------------------------------------------------------


class TestPhase8Output:
    """Phase 8: run_output(generate_insight=False) — skip LLM insight."""

    def test_digest_produced(self, pipeline_env: dict):
        """Run output with generate_insight=False, verify digest structure."""
        from alcuinus.clustering import run_clustering
        from alcuinus.decay import run_decay
        from alcuinus.output import run_output

        env = pipeline_env
        chunks = env["chunks"]

        # Phase 5: create Zvec index
        mock_result = _make_mock_embeddings(len(chunks))
        with patch("alcuinus.embedding.Mistral") as MockClient:
            instance = MockClient.return_value
            instance.embeddings.create.return_value = mock_result
            embed_and_store_incremental(
                db_path=env["db_path"],
                index_path=env["index_path"],
                api_key="test-key",
            )

        # Phase 6: clustering
        run_clustering(
            index_path=env["index_path"],
            db_path=env["db_path"],
            chunks_path=env["chunks_path"],
            output_path=env["clusters_path"],
            k=K,
        )

        # Phase 7: decay (heuristic)
        run_decay(
            clusters_path=env["clusters_path"],
            chunks_path=env["chunks_path"],
            bundles_path=env["bundles_path"],
            output_path=env["decay_path"],
            use_llm=False,
            db_path=env["db_path"],
        )

        # Phase 8: output (skip LLM insight)
        result = run_output(
            clusters_path=env["clusters_path"],
            decay_path=env["decay_path"],
            bundles_path=env["bundles_path"],
            metadata_path=env["metadata_path"],
            output_path=env["digest_path"],
            generate_insight=False,
        )

        assert result == env["digest_path"]
        assert Path(env["digest_path"]).exists()

        with open(env["digest_path"], encoding="utf-8") as f:
            digest = f.read()

        # Structural assertions — digest must have header, topics, footer
        assert "Weekly Digest" in digest, "Digest missing 'Weekly Digest' header"
        assert "AlcuinusBot" in digest, "Digest missing 'AlcuinusBot' footer"
        assert "Top Topics" in digest, "Digest missing 'Top Topics' section"
        assert len(digest) > 100, f"Digest too short ({len(digest)} chars)"

        # Should have topic entries (clusters with keywords)
        with open(env["clusters_path"], encoding="utf-8") as f:
            clusters_data = json.load(f)
        # Each cluster should have keywords that may appear in the digest
        assert len(clusters_data["clusters"]) > 0


# ---------------------------------------------------------------------------
# Phase 10 — Review
# ---------------------------------------------------------------------------


class TestPhase10Review:
    """Phase 10: run_review(regenerate=False) — no regeneration."""

    def test_review_report_produced(self, pipeline_env: dict):
        """Run review, verify report structure."""
        from alcuinus.clustering import run_clustering
        from alcuinus.decay import run_decay
        from alcuinus.review import run_review

        env = pipeline_env
        chunks = env["chunks"]

        # Phase 5: create Zvec index
        mock_result = _make_mock_embeddings(len(chunks))
        with patch("alcuinus.embedding.Mistral") as MockClient:
            instance = MockClient.return_value
            instance.embeddings.create.return_value = mock_result
            embed_and_store_incremental(
                db_path=env["db_path"],
                index_path=env["index_path"],
                api_key="test-key",
            )

        # Phase 6: clustering
        run_clustering(
            index_path=env["index_path"],
            db_path=env["db_path"],
            chunks_path=env["chunks_path"],
            output_path=env["clusters_path"],
            k=K,
        )

        # Phase 7: decay (heuristic)
        run_decay(
            clusters_path=env["clusters_path"],
            chunks_path=env["chunks_path"],
            bundles_path=env["bundles_path"],
            output_path=env["decay_path"],
            use_llm=False,
            db_path=env["db_path"],
        )

        # Phase 10: review (no regeneration)
        result = run_review(
            decay_path=env["decay_path"],
            clusters_path=env["clusters_path"],
            output_path=env["review_path"],
            snapshot_path=env["snapshot_path"],
            regenerate=False,
        )

        assert result == env["review_path"]
        assert Path(env["review_path"]).exists()

        with open(env["review_path"], encoding="utf-8") as f:
            report = json.load(f)

        # Structural assertions
        assert "reviewed_at" in report, "Report missing reviewed_at"
        assert "classified_at" in report, "Report missing classified_at"
        assert "months_since_classification" in report, (
            "Report missing months_since_classification"
        )
        assert "clusters_total" in report, "Report missing clusters_total"
        assert "flagged" in report, "Report missing flagged"
        assert "summary" in report, "Report missing summary"
        assert "profile_changes" in report, "Report missing profile_changes"

        # clusters_total should match our K
        assert report["clusters_total"] == K, (
            f"Expected {K} clusters_total, got {report['clusters_total']}"
        )

        # summary should have action keys
        expected_actions = {
            "promote_evergreen",
            "demote_semi_stable",
            "remove_ephemeral",
            "no_change",
        }
        assert set(report["summary"].keys()) == expected_actions, (
            f"Summary keys {set(report['summary'].keys())} != {expected_actions}"
        )

        # Sum of all action counts should equal clusters_total
        total_actions = sum(report["summary"].values())
        assert total_actions == report["clusters_total"], (
            f"Action counts sum {total_actions} != clusters_total {report['clusters_total']}"
        )

        # Snapshot should have been saved
        assert Path(env["snapshot_path"]).exists(), "Snapshot file not saved"

        # flagged is a list (may be empty for fresh classification)
        assert isinstance(report["flagged"], list)
        assert isinstance(report["profile_changes"], list)
