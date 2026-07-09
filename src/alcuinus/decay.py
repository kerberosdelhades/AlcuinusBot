"""
Phase 7 — Decay classification

Tags each cluster with a decay profile: evergreen, semi-stable, or ephemeral.

Two classification strategies:
1. **LLM-based** (primary): sends cluster keywords + sample texts to Mistral
   and asks for a classification. One batch call for all clusters.
2. **Heuristic** (fallback): classifies based on date ranges and keyword
   patterns. No API calls, instant, but less accurate.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DECAY_PROFILES = {
    "evergreen": {
        "label": "Evergreen",
        "retention_months": None,  # permanent
        "description": "Foundational knowledge — papers, architecture, core concepts. Permanent, surfaced to newcomers.",
    },
    "semi-stable": {
        "label": "Semi-stable",
        "retention_months": 18,
        "description": "Important but time-bound — benchmarks, scaling laws, prompting techniques. 12-24 month retention.",
    },
    "ephemeral": {
        "label": "Ephemeral",
        "retention_months": 3,
        "description": "News, announcements, transient tools. Short retention, flagged for review/removal.",
    },
}

# Keywords that suggest each profile
_EVERGREEN_SIGNALS = {
    "paper", "architecture", "transformer", "attention", "fundamental",
    "theory", "concept", "introduction", "tutorial", "guide", "basics",
    "overview", "explained", "primer", "foundations",
}

_EPHEMERAL_SIGNALS = {
    "released", "launch", "announcement", "update", "version", "new",
    "today", "this week", "breaking", "news", "just", "v2", "v3", "v4",
    "gpt-5", "gpt-4", "claude", "gemini", "mistral", "llama",
}

DEFAULT_CLUSTERS_PATH = "data/clusters.json"
DEFAULT_CHUNKS_PATH = "data/chunks.json"
DEFAULT_BUNDLES_PATH = "data/bundles.json"
DEFAULT_OUTPUT_PATH = "data/decay_profiles.json"


# ---------------------------------------------------------------------------
# Heuristic classification
# ---------------------------------------------------------------------------


def classify_heuristic(
    cluster_info: dict,
    cluster_texts: list[str],
    cluster_dates: list[str],
) -> str:
    """Classify a cluster using heuristics (no LLM).

    Returns one of "evergreen", "semi-stable", "ephemeral".
    """
    keywords = set(kw.lower() for kw in cluster_info.get("keywords", []))

    # Check for ephemeral signals
    ephemeral_hits = keywords & _EPHEMERAL_SIGNALS
    if len(ephemeral_hits) >= 2:
        return "ephemeral"

    # Check for evergreen signals
    evergreen_hits = keywords & _EVERGREEN_SIGNALS
    if len(evergreen_hits) >= 2:
        return "evergreen"

    # Date-based: clusters with very recent content lean ephemeral
    if cluster_dates:
        try:
            most_recent = max(d[:10] for d in cluster_dates if d)
            year = int(most_recent[:4])
            month = int(most_recent[5:7])
            now = datetime.now()
            months_ago = (now.year - year) * 12 + (now.month - month)

            if months_ago <= 3:
                return "ephemeral"
            elif months_ago >= 18:
                return "semi-stable"  # old but not tagged evergreen
        except (ValueError, IndexError):
            pass

    # Default
    return "semi-stable"


# ---------------------------------------------------------------------------
# LLM-based classification
# ---------------------------------------------------------------------------


def classify_llm(
    clusters: dict[str, dict],
    vectors: dict[str, dict],
    api_key: str,
    model: str = "mistral-small-latest",
) -> dict[str, str]:
    """Classify all clusters in one LLM call.

    Returns dict mapping cluster_label → decay_profile.
    """
    from mistralai.client import Mistral

    client = Mistral(api_key=api_key)

    # Build a summary for each cluster
    summaries = []
    cluster_keys = []
    for key in sorted(clusters.keys(), key=int):
        info = clusters[key]
        keywords = ", ".join(info.get("keywords", []))
        size = info.get("size", 0)
        bundle_count = len(info.get("bundle_ids", []))

        # Sample up to 3 chunk texts
        sample_texts = []
        for cid in info.get("chunk_ids", [])[:3]:
            text = vectors.get(cid, {}).get("text", "")
            if text:
                sample_texts.append(text[:200])

        summary = (
            f"Cluster {key}: {size} chunks, {bundle_count} bundles. "
            f"Keywords: [{keywords}]. "
            f"Sample texts: {'; '.join(sample_texts[:3])}"
        )
        summaries.append(summary)
        cluster_keys.append(key)

    prompt = f"""You are classifying discussion clusters from a technical AI/ML Telegram group into decay profiles.

Profiles:
- evergreen: foundational knowledge (papers, architecture, core concepts). Permanent.
- semi-stable: important but time-bound (benchmarks, scaling laws, techniques). 12-24 month retention.
- ephemeral: news, announcements, transient tools. Short retention (months).

Classify each cluster below. Respond ONLY with valid JSON: {{"cluster_0": "evergreen", "cluster_1": "ephemeral", ...}}

Clusters:
{chr(10).join(summaries)}"""

    result = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    response_text = str(result.choices[0].message.content or "").strip()  # type: ignore[union-attr]
    if not response_text:
        raise ValueError("LLM returned empty response")

    # Parse JSON from response
    # Handle markdown code blocks
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        response_text = "\n".join(lines[1:-1])

    classifications = json.loads(response_text)

    # Normalize LLM keys: "cluster_0" → "0"
    normalized: dict[str, str] = {}
    for llm_key, profile in classifications.items():
        numeric_key = llm_key.replace("cluster_", "")
        if numeric_key in clusters:
            normalized[numeric_key] = profile
        elif llm_key in clusters:
            normalized[llm_key] = profile
    classifications = normalized

    # Validate
    valid_profiles = {"evergreen", "semi-stable", "ephemeral"}
    for key, profile in classifications.items():
        if profile not in valid_profiles:
            classifications[key] = "semi-stable"

    return classifications


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_decay(
    clusters_path: str = DEFAULT_CLUSTERS_PATH,
    chunks_path: str = DEFAULT_CHUNKS_PATH,
    bundles_path: str = DEFAULT_BUNDLES_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    api_key: str | None = None,
    use_llm: bool = True,
) -> str:
    """Classify clusters by decay profile and write output.

    Returns path to the decay profiles JSON file.
    """
    # Load data
    with open(clusters_path, encoding="utf-8") as f:
        cluster_data = json.load(f)
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)
    with open(bundles_path, encoding="utf-8") as f:
        bundles = json.load(f)

    clusters = cluster_data["clusters"]

    # Build lookup tables
    chunk_map = {c["chunk_id"]: c for c in chunks}
    bundle_map = {b["anchor"]["msg_id"]: b for b in bundles}

    # Build vectors-like structure for text access
    vectors = {}
    for chunk in chunks:
        vectors[chunk["chunk_id"]] = {"text": chunk["text"]}

    # Collect dates and texts per cluster
    cluster_context: dict[str, dict] = {}
    for key, info in clusters.items():
        texts = []
        dates = []
        for cid in info.get("chunk_ids", []):
            chunk = chunk_map.get(cid, {})
            if chunk.get("text"):
                texts.append(chunk["text"])
            bid = chunk.get("bundle_anchor_id")
            bundle = bundle_map.get(bid, {})
            anchor_date = bundle.get("anchor", {}).get("date", "")
            if anchor_date:
                dates.append(anchor_date)
        cluster_context[key] = {"texts": texts, "dates": dates}

    # Classify
    if use_llm:
        if api_key is None:
            api_key = os.environ.get("MISTRAL_API_KEY", "")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not set for LLM classification.")

        try:
            classifications = classify_llm(clusters, vectors, api_key)
        except Exception as exc:
            print(f"LLM classification failed ({exc}), falling back to heuristic")
            classifications = {}
            for key, info in clusters.items():
                ctx = cluster_context[key]
                classifications[key] = classify_heuristic(
                    info, ctx["texts"], ctx["dates"]
                )
    else:
        classifications = {}
        for key, info in clusters.items():
            ctx = cluster_context[key]
            classifications[key] = classify_heuristic(
                info, ctx["texts"], ctx["dates"]
            )

    # Build output
    result = {
        "classified_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": "llm" if use_llm else "heuristic",
        "profiles": DECAY_PROFILES,
        "classifications": {},
    }

    for key, profile in classifications.items():
        result["classifications"][key] = {
            "cluster_label": clusters[key].get("label", f"cluster_{key}"),
            "decay_profile": profile,
            "profile_info": DECAY_PROFILES.get(profile, {}),
            "size": clusters[key].get("size", 0),
            "bundle_count": len(clusters[key].get("bundle_ids", [])),
            "keywords": clusters[key].get("keywords", []),
        }

    # Write output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return output_path


if __name__ == "__main__":
    output = run_decay()
    print(f"Decay profiles written to: {output}")
