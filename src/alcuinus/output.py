"""
Phase 8 — Output (Digest)

Generates a structured weekly digest from clustering and decay data,
formatted for posting to the docs Telegram channel.

Digest format:
    - Top 5 topics (by bundle count, with decay profiles)
    - 3 emerging themes (ephemeral + newest clusters)
    - 5 most influential links (by reaction count per cluster)
    - 1 connection insight (LLM-generated cross-cluster link)
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
DEFAULT_OUTPUT_PATH = "data/digest.txt"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(
    clusters_path: str = DEFAULT_CLUSTERS_PATH,
    decay_path: str = DEFAULT_DECAY_PATH,
    bundles_path: str = DEFAULT_BUNDLES_PATH,
    metadata_path: str = DEFAULT_METADATA_PATH,
) -> tuple[dict, dict, list, dict[str, dict]]:
    """Load all data for digest generation."""
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
# Digest sections
# ---------------------------------------------------------------------------


def top_topics(cluster_info: dict, decay_info: dict, top_n: int = 5) -> list[dict]:
    """Return top N clusters by bundle count, with decay profiles."""
    ranked = sorted(
        cluster_info.items(),
        key=lambda kv: len(kv[1].get("bundle_ids", [])),
        reverse=True,
    )[:top_n]

    result = []
    for key, info in ranked:
        profile = decay_info.get(key, {}).get("decay_profile", "semi-stable")
        result.append({
            "cluster_id": key,
            "keywords": info.get("keywords", []),
            "bundle_count": len(info.get("bundle_ids", [])),
            "chunk_count": info.get("size", 0),
            "decay_profile": profile,
        })
    return result


def emerging_themes(cluster_info: dict, decay_info: dict, top_n: int = 3) -> list[dict]:
    """Return emerging themes: ephemeral clusters, sorted by size descending."""
    ephemeral = []
    for key, info in cluster_info.items():
        profile = decay_info.get(key, {}).get("decay_profile", "")
        if profile == "ephemeral":
            ephemeral.append((key, info))

    ranked = sorted(
        ephemeral,
        key=lambda kv: len(kv[1].get("bundle_ids", [])),
        reverse=True,
    )[:top_n]

    return [
        {
            "cluster_id": key,
            "keywords": info.get("keywords", []),
            "bundle_count": len(info.get("bundle_ids", [])),
        }
        for key, info in ranked
    ]


def influential_links(
    cluster_info: dict,
    bundles: list[dict],
    link_meta: dict[str, dict],
    top_n: int = 5,
) -> list[dict]:
    """Return the most influential links: bundles with most reactions per cluster.

    A bundle's influence is measured by its reaction count — more discussion
    means more influence within the cluster.
    """
    # Map bundle by anchor msg_id
    bundle_map = {b["anchor"]["msg_id"]: b for b in bundles}

    # Per cluster, find the bundle with most reactions
    candidates = []
    for key, info in cluster_info.items():
        best_bundle = None
        best_reactions = -1
        for bid in info.get("bundle_ids", []):
            b = bundle_map.get(bid)
            if b is None:
                continue
            n = len(b.get("reactions", []))
            if n > best_reactions:
                best_reactions = n
                best_bundle = b
            if best_bundle is None:
                best_bundle = b  # at least pick one

        if best_bundle:
            urls = best_bundle["anchor"].get("urls", [])
            url = urls[0] if urls else ""
            meta = link_meta.get(url, {})
            candidates.append({
                "cluster_id": key,
                "url": url,
                "title": meta.get("title") or url,
                "reactions": best_reactions,
                "anchor_text": best_bundle["anchor"].get("text_preview", "")[:100],
            })

    # Sort by reaction count descending
    candidates.sort(key=lambda x: x["reactions"], reverse=True)
    return candidates[:top_n]


def connection_insight(
    cluster_info: dict,
    decay_info: dict,
    api_key: str,
    model: str = "mistral-small-latest",
) -> str:
    """Generate a cross-cluster connection insight via LLM."""
    from mistralai.client import Mistral

    client = Mistral(api_key=api_key)

    # Build cluster summaries
    summaries = []
    for key in sorted(cluster_info.keys(), key=int):
        info = cluster_info[key]
        kw = ", ".join(info.get("keywords", [])[:5])
        profile = decay_info.get(key, {}).get("decay_profile", "?")
        summaries.append(f"Cluster {key} [{profile}]: {info.get('size', 0)} chunks, keywords: {kw}")

    prompt = f"""You are analyzing discussion clusters from a technical AI/ML Telegram group.

Clusters:
{chr(10).join(summaries)}

Write ONE sentence (max 120 chars) that connects two of these clusters in a surprising or insightful way. Example: "The MicroPython LCD discussion intersects with the Hakko hardware cluster — both involve DIY electronics tooling for small device development." Be specific and reference actual cluster content. Respond with just the sentence, nothing else."""

    result = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=100,
    )

    text = str(result.choices[0].message.content or "").strip()  # type: ignore[union-attr]
    return text


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


EMOJI = {
    "evergreen": "🟢",
    "semi-stable": "🟡",
    "ephemeral": "🔴",
}


def format_digest(
    topics: list[dict],
    emerging: list[dict],
    links: list[dict],
    insight: str,
) -> str:
    """Format the digest as Markdown-text suitable for Telegram."""
    date_str = datetime.now().strftime("%B %d, %Y")

    lines = [
        f"📊 *Weekly Digest — Kreitek IA* ({date_str})",
        "",
    ]

    # Top topics
    lines.append("🔥 *Top Topics*")
    for i, t in enumerate(topics, 1):
        emoji = EMOJI.get(t["decay_profile"], "⚪")
        kw = ", ".join(t["keywords"][:3])
        lines.append(f"  {i}. {emoji} {kw}")
        lines.append(f"     _{t['bundle_count']} bundles, {t['chunk_count']} chunks_")
    lines.append("")

    # Emerging themes
    if emerging:
        lines.append("🆕 *Emerging Themes*")
        for i, e in enumerate(emerging, 1):
            kw = ", ".join(e["keywords"][:3])
            lines.append(f"  {i}. {kw} ({e['bundle_count']} bundles)")
        lines.append("")

    # Influential links
    if links:
        lines.append("🔗 *Most Discussed Links*")
        for i, link in enumerate(links, 1):
            title = link["title"][:80]
            lines.append(f"  {i}. [{title}]({link['url']})")
            lines.append(f"     _{link['reactions']} reactions_")
        lines.append("")

    # Connection insight
    if insight:
        lines.append(f"💡 *Connection Insight*")
        lines.append(f"  {insight}")
        lines.append("")

    lines.append("_— AlcuinusBot_")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_output(
    clusters_path: str = DEFAULT_CLUSTERS_PATH,
    decay_path: str = DEFAULT_DECAY_PATH,
    bundles_path: str = DEFAULT_BUNDLES_PATH,
    metadata_path: str = DEFAULT_METADATA_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    api_key: str | None = None,
    generate_insight: bool = True,
) -> str:
    """Load all data, generate digest, write output.

    Returns path to the digest text file.
    """
    clusters_data, decay_data, bundles, link_meta = load_data(
        clusters_path, decay_path, bundles_path, metadata_path
    )

    clusters = clusters_data["clusters"]
    decay = decay_data["classifications"]

    # Generate sections
    topics = top_topics(clusters, decay)
    emerging = emerging_themes(clusters, decay)
    links = influential_links(clusters, bundles, link_meta)

    # LLM insight
    insight = ""
    if generate_insight:
        if api_key is None:
            api_key = os.environ.get("MISTRAL_API_KEY", "")
        if api_key:
            try:
                insight = connection_insight(clusters, decay, api_key)
            except Exception as exc:
                insight = f"(LLM insight unavailable: {exc})"
        else:
            insight = "(No MISTRAL_API_KEY set — skipping connection insight)"

    # Format
    digest = format_digest(topics, emerging, links, insight)

    # Write
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(digest)

    return output_path


if __name__ == "__main__":
    output = run_output()
    print(f"Digest written to: {output}")
    print()
    with open(output) as f:
        print(f.read())
