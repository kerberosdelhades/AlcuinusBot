"""Tests for Phase 7 — decay classification."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from alcuinus.decay import (
    DECAY_PROFILES,
    classify_heuristic,
    classify_llm,
    run_decay,
)


# ---------------------------------------------------------------------------
# classify_heuristic
# ---------------------------------------------------------------------------


class TestClassifyHeuristic:
    def test_ephemeral_on_keyword_signals(self):
        info = {"keywords": ["released", "launch", "update"]}
        result = classify_heuristic(info, [], [])
        assert result == "ephemeral"

    def test_evergreen_on_keyword_signals(self):
        info = {"keywords": ["paper", "architecture", "transformer"]}
        result = classify_heuristic(info, [], [])
        assert result == "evergreen"

    def test_semi_stable_default(self):
        info = {"keywords": ["python", "tool", "library"]}
        result = classify_heuristic(info, [], [])
        assert result == "semi-stable"

    def test_ephemeral_on_recent_date(self):
        info = {"keywords": ["python", "tool"]}
        from datetime import datetime
        now = datetime.now()
        recent = f"{now.year}-{now.month:02d}-01"
        result = classify_heuristic(info, [], [recent])
        assert result == "ephemeral"

    def test_semi_stable_on_old_date(self):
        info = {"keywords": ["python", "tool"]}
        result = classify_heuristic(info, [], ["2022-06-15"])
        assert result == "semi-stable"

    def test_mixed_signals_ephemeral_wins(self):
        # 2 ephemeral signals, 1 evergreen → ephemeral
        info = {"keywords": ["released", "update", "paper"]}
        result = classify_heuristic(info, [], [])
        assert result == "ephemeral"


# ---------------------------------------------------------------------------
# classify_llm (mocked)
# ---------------------------------------------------------------------------


class TestClassifyLlm:
    def test_basic_classification(self):
        clusters = {
            "0": {
                "label": "cluster_0",
                "keywords": ["transformer", "attention", "paper"],
                "size": 20,
                "bundle_ids": [1, 2, 3],
                "chunk_ids": ["c1", "c2"],
            },
            "1": {
                "label": "cluster_1",
                "keywords": ["released", "update", "new"],
                "size": 10,
                "bundle_ids": [4],
                "chunk_ids": ["c3"],
            },
        }
        vectors = {
            "c1": {"text": "Foundational transformer paper"},
            "c2": {"text": "Attention mechanism explained"},
            "c3": {"text": "New model released today"},
        }

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"cluster_0": "evergreen", "cluster_1": "ephemeral"}'

        with patch("mistralai.client.Mistral") as MockClient:
            instance = MockClient.return_value
            instance.chat.complete.return_value = mock_response

            result = classify_llm(clusters, vectors, "test-key")

        assert result["0"] == "evergreen"
        assert result["1"] == "ephemeral"

    def test_handles_markdown_code_blocks(self):
        clusters = {
            "0": {"label": "cluster_0", "keywords": ["test"], "size": 5, "bundle_ids": [1], "chunk_ids": []},
        }
        vectors = {}

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '```json\n{"cluster_0": "semi-stable"}\n```'

        with patch("mistralai.client.Mistral") as MockClient:
            instance = MockClient.return_value
            instance.chat.complete.return_value = mock_response

            result = classify_llm(clusters, vectors, "test-key")

        assert result["0"] == "semi-stable"

    def test_invalid_profile_defaults_to_semi_stable(self):
        clusters = {
            "0": {"label": "cluster_0", "keywords": ["test"], "size": 5, "bundle_ids": [1], "chunk_ids": []},
        }
        vectors = {}

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"cluster_0": "invalid_profile"}'

        with patch("mistralai.client.Mistral") as MockClient:
            instance = MockClient.return_value
            instance.chat.complete.return_value = mock_response

            result = classify_llm(clusters, vectors, "test-key")

        assert result["0"] == "semi-stable"

    def test_empty_response_raises(self):
        clusters = {"0": {"label": "cluster_0", "keywords": [], "size": 1, "bundle_ids": [], "chunk_ids": []}}
        vectors = {}

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""

        with patch("mistralai.client.Mistral") as MockClient:
            instance = MockClient.return_value
            instance.chat.complete.return_value = mock_response

            with pytest.raises(ValueError, match="empty"):
                classify_llm(clusters, vectors, "test-key")


# ---------------------------------------------------------------------------
# DECAY_PROFILES
# ---------------------------------------------------------------------------


class TestDecayProfiles:
    def test_has_all_profiles(self):
        assert "evergreen" in DECAY_PROFILES
        assert "semi-stable" in DECAY_PROFILES
        assert "ephemeral" in DECAY_PROFILES

    def test_evergreen_is_permanent(self):
        assert DECAY_PROFILES["evergreen"]["retention_months"] is None

    def test_semi_stable_has_retention(self):
        assert DECAY_PROFILES["semi-stable"]["retention_months"] == 18

    def test_ephemeral_has_retention(self):
        assert DECAY_PROFILES["ephemeral"]["retention_months"] == 3


# ---------------------------------------------------------------------------
# run_decay — I/O round-trip (heuristic mode, no LLM)
# ---------------------------------------------------------------------------


class TestRunDecay:
    def test_roundtrip_heuristic(self, tmp_path):
        # Create minimal cluster data
        clusters_data = {
            "k": 2,
            "clusters": {
                "0": {
                    "label": "cluster_0",
                    "chunk_ids": ["c1"],
                    "size": 1,
                    "bundle_ids": [1],
                    "keywords": ["paper", "architecture"],
                },
                "1": {
                    "label": "cluster_1",
                    "chunk_ids": ["c2"],
                    "size": 1,
                    "bundle_ids": [2],
                    "keywords": ["released", "update"],
                },
            },
            "assignments": {"c1": "0", "c2": "1"},
        }
        chunks = [
            {"chunk_id": "c1", "text": "Foundational paper on transformers", "bundle_anchor_id": 1, "is_parent": True, "token_estimate": 50},
            {"chunk_id": "c2", "text": "New model released today", "bundle_anchor_id": 2, "is_parent": False, "token_estimate": 20},
        ]
        bundles = [
            {"anchor": {"msg_id": 1, "date": "2022-06-15", "urls": [], "text_preview": ""}, "reactions": []},
            {"anchor": {"msg_id": 2, "date": "2026-06-01", "urls": [], "text_preview": ""}, "reactions": []},
        ]

        clusters_path = tmp_path / "clusters.json"
        chunks_path = tmp_path / "chunks.json"
        bundles_path = tmp_path / "bundles.json"
        out_path = tmp_path / "decay.json"

        clusters_path.write_text(json.dumps(clusters_data))
        chunks_path.write_text(json.dumps(chunks))
        bundles_path.write_text(json.dumps(bundles))

        result = run_decay(
            clusters_path=str(clusters_path),
            chunks_path=str(chunks_path),
            bundles_path=str(bundles_path),
            output_path=str(out_path),
            use_llm=False,
        )

        assert result == str(out_path)
        assert out_path.exists()

        data = json.loads(out_path.read_text())
        assert "classified_at" in data
        assert data["method"] == "heuristic"
        assert len(data["classifications"]) == 2

        # Cluster 0 (paper, architecture, old) → evergreen
        assert data["classifications"]["0"]["decay_profile"] == "evergreen"
        # Cluster 1 (released, update, recent) → ephemeral
        assert data["classifications"]["1"]["decay_profile"] == "ephemeral"

    def test_raises_on_missing_api_key(self, tmp_path):
        clusters_path = tmp_path / "c.json"
        clusters_path.write_text('{"k":1,"clusters":{},"assignments":{}}')
        chunks_path = tmp_path / "ch.json"
        chunks_path.write_text("[]")
        bundles_path = tmp_path / "b.json"
        bundles_path.write_text("[]")

        with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
            run_decay(
                clusters_path=str(clusters_path),
                chunks_path=str(chunks_path),
                bundles_path=str(bundles_path),
                output_path=str(tmp_path / "out.json"),
                api_key="",
                use_llm=True,
            )
