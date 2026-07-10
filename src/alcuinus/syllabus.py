"""
Phase 9 — Syllabus (living study guide)

Produces a persistent, topic-organized study guide from cluster and
decay data. Organized by decay tier:
    - Evergreen at top (foundational, read first)
    - Semi-stable (important but time-bound)
    - Ephemeral at bottom (historical interest)

Distinct from Phase 8 digest: the syllabus is structural, not temporal.
It answers "what should a newcomer read first?" not "what happened this week?"
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CLUSTERS_PATH = "data/clusters.json"
DEFAULT_DECAY_PATH = "data/decay_profiles.json"
DEFAULT_BUNDLES_PATH = "data/bundles.json"
DEFAULT_METADATA_PATH = "data/link_metadata.json"
DEFAULT_OUTPUT_PATH = "data/syllabus.md"

TIER_ORDER = ["evergreen", "semi-stable", "ephemeral"]
TIER_LABELS = {
    "evergreen": "🟢 Foundational",
    "semi-stable": "🟡 Current topics",
    "ephemeral": "🔴 Recent & ephemeral",
}

CHANNEL_NAME = "Kreitek IA"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(
    clusters_path: str = DEFAULT_CLUSTERS_PATH,
    decay_path: str = DEFAULT_DECAY_PATH,
    bundles_path: str = DEFAULT_BUNDLES_PATH,
    metadata_path: str = DEFAULT_METADATA_PATH,
) -> tuple[dict, dict, list, dict[str, dict]]:
    """Load all data for syllabus generation."""
    with open(clusters_path, encoding="utf-8") as f:
        clusters = json.load(f)
    with open(decay_path, encoding="utf-8") as f:
        decay = json.load(f)
    with open(bundles_path, encoding="utf-8") as f:
        bundles = json.load(f)
    with open(metadata_path, encoding="utf-8") as f:
        meta_list = json.load(f)

    link_meta = {r["url"]: r for r in meta_list}
    return clusters, decay, bundles, link_meta


# ---------------------------------------------------------------------------
# Syllabus builder
# ---------------------------------------------------------------------------


def build_syllabus_sections(
    clusters: dict[str, dict],
    decay: dict[str, dict],
    bundles: list[dict],
    link_meta: dict[str, dict],
) -> dict[str, list[dict]]:
    """Organize clusters into tiered sections.

    Returns dict mapping tier → list of cluster entries, each with:
        {keywords, bundle_count, links: [{url, title, description}]}
    """
    # Map bundles by anchor msg_id
    bundle_map = {b["anchor"]["msg_id"]: b for b in bundles}

    # Group clusters by decay tier
    tiered: dict[str, list[dict]] = {t: [] for t in TIER_ORDER}

    for key, info in clusters.items():
        profile = decay.get(key, {}).get("decay_profile", "semi-stable")
        if profile not in tiered:
            profile = "semi-stable"

        # Gather links for this cluster
        links = []
        for bid in info.get("bundle_ids", []):
            b = bundle_map.get(bid)
            if b is None:
                continue
            for url in b["anchor"].get("urls", []):
                meta = link_meta.get(url, {})
                links.append({
                    "url": url,
                    "title": meta.get("title") or url,
                    "description": meta.get("description", ""),
                    "reactions": len(b.get("reactions", [])),
                })

        # Sort links by reaction count
        links.sort(key=lambda x: x["reactions"], reverse=True)

        entry = {
            "keywords": info.get("keywords", []),
            "bundle_count": len(info.get("bundle_ids", [])),
            "chunk_count": info.get("size", 0),
            "links": links[:10],  # top 10 links per cluster
        }
        tiered[profile].append(entry)

    # Sort within each tier by bundle count
    for tier in tiered:
        tiered[tier].sort(key=lambda x: x["bundle_count"], reverse=True)

    return tiered


def format_syllabus(
    tiered: dict[str, list[dict]],
) -> str:
    """Format tiered clusters into a Markdown study guide."""
    date_str = datetime.now().strftime("%B %d, %Y")
    lines = [
        f"# 📚 Guía de Estudio — {CHANNEL_NAME}",
        "",
        f"*Living study guide. Last updated: {date_str}.*",
        f"*Read top to bottom: foundational content first, ephemeral last.*",
        "",
        "---",
        "",
    ]

    for tier in TIER_ORDER:
        entries = tiered.get(tier, [])
        if not entries:
            continue

        lines.append(f"## {TIER_LABELS[tier]}")
        lines.append("")

        for i, entry in enumerate(entries, 1):
            kw = ", ".join(entry["keywords"][:4]) if entry["keywords"] else "untagged"
            lines.append(
                f"### {i}. {kw}"
            )
            lines.append(
                f"*{entry['bundle_count']} links, {entry['chunk_count']} messages*"
            )
            lines.append("")

            for link in entry["links"]:
                title = link["title"][:120]
                desc = link["description"]
                if desc:
                    lines.append(f"- [{title}]({link['url']})")
                    lines.append(f"  {desc[:200]}")
                else:
                    lines.append(f"- [{title}]({link['url']})")

            lines.append("")

        lines.append("---")
        lines.append("")

    lines.append("_— AlcuinusBot —_")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_syllabus(
    clusters_path: str = DEFAULT_CLUSTERS_PATH,
    decay_path: str = DEFAULT_DECAY_PATH,
    bundles_path: str = DEFAULT_BUNDLES_PATH,
    metadata_path: str = DEFAULT_METADATA_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> str:
    """Load data, build syllabus sections, write output.

    Returns path to the syllabus Markdown file.
    """
    clusters_data, decay_data, bundles, link_meta = load_data(
        clusters_path, decay_path, bundles_path, metadata_path
    )

    clusters = clusters_data["clusters"]
    decay = decay_data["classifications"]

    tiered = build_syllabus_sections(clusters, decay, bundles, link_meta)
    syllabus = format_syllabus(tiered)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(syllabus)

    return output_path


if __name__ == "__main__":
    output = run_syllabus()
    print(f"Syllabus written to: {output}")
    print()
    with open(output) as f:
        print(f.read())
