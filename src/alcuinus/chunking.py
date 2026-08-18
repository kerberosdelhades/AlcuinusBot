"""
Phase 4 — Chunking & Tagging

Split bundles into retrievable chunks with rich metadata.
Parent-child chunking: small child chunks (256 tokens) for retrieval
precision, larger parent chunks (1024 tokens) for LLM context.

Overlap: 15% (NVIDIA FinanceBench optimal — 2024 benchmark tested
10%, 15%, 20%; 15% best on 1,024-token chunks).

Storage: SQLite is the source of truth. Bundles and link metadata are
loaded from SQLite; chunks are written to both the ``chunks`` table and
the legacy ``chunks.json`` file for backwards compat.
"""

from __future__ import annotations

import json
import os
from typing import Any

from alcuinus import db

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHILD_CHUNK_TOKENS = 256
PARENT_CHUNK_TOKENS = 1024
OVERLAP_RATIO = 0.15  # NVIDIA FinanceBench optimal
# Source channel name, used in chunk metadata prefix.  Configurable via env
# var ``ALCUINUS_CHANNEL_NAME``; defaults to a generic value now that the
# source isn't strictly the "Demiurgo" subgroup.
CHANNEL = os.environ.get("ALCUINUS_CHANNEL_NAME", "Kreitek IA")


# ---------------------------------------------------------------------------
# Token estimation (no external tokenizer dependency)
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Rough token count. 3 chars/token is conservative for es/en mix.

    This is good enough for chunk boundary decisions. Exact token counts
    are only needed at the embedding API level (Phase 5), where Mistral
    counts internally.
    """
    return max(1, len(text) // 3)


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------


def load_link_metadata(
    db_path: str = db.DEFAULT_DB_PATH,
    json_path: str = "data/link_metadata.json",
) -> dict[str, dict]:
    """Load link metadata from SQLite (or JSON fallback), keyed by URL.

    ``json_path`` is the legacy-JSON fallback used when the SQLite DB is
    absent (fresh clone / one-release compat window).
    """
    if db.db_exists(db_path):
        records = db.load_link_metadata(db_path)
    else:
        with open(json_path, encoding="utf-8") as f:
            records = json.load(f)
    return {r["url"]: r for r in records}


# ---------------------------------------------------------------------------
# Bundle → text assembly
# ---------------------------------------------------------------------------


def build_bundle_text(
    bundle: dict,
    link_meta: dict[str, dict],
) -> str:
    """Concatenate anchor metadata + opinion texts for one bundle."""
    parts: list[str] = []
    anchor = bundle.get("anchor", {})
    urls = anchor.get("urls", [])

    # Title + description from Phase 3 metadata
    for url in urls:
        meta = link_meta.get(url, {})
        if meta.get("title"):
            parts.append(f"[{meta['title']}]")
        if meta.get("description"):
            parts.append(meta["description"])

    # Anchor text preview
    text_preview = anchor.get("text_preview", "")
    if text_preview:
        parts.append(text_preview)

    # Reaction/opinion texts
    for reaction in bundle.get("reactions", []):
        r_text = reaction.get("text_preview", "")
        if r_text:
            parts.append(r_text)

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Metadata prefix builder
# ---------------------------------------------------------------------------


def build_metadata_prefix(bundle: dict) -> str:
    """Build a metadata prefix string for filtering in retrieval.

    Format: ``[channel: X] [date: YYYY-MM-DD] [author: id] [lang: xx]``

    Language detection is heuristic (first non-ASCII char → guess Spanish).
    """
    anchor = bundle.get("anchor", {})
    date = anchor.get("date", "")
    date_short = date[:10] if date else "unknown"

    sender_id = anchor.get("sender_id") or "unknown"
    author = str(sender_id)

    text = anchor.get("text_preview", "")
    lang = "es" if any(ord(c) > 127 for c in text) else "en"

    return f"[channel: {CHANNEL}] [date: {date_short}] [author: {author}] [lang: {lang}]"


def build_full_metadata_prefix(
    bundle: dict,
    link_meta: dict[str, dict],
) -> str:
    """Extended metadata prefix including link title from Phase 3."""
    anchor = bundle.get("anchor", {})
    base = build_metadata_prefix(bundle)

    # Add link title if available
    urls = anchor.get("urls", [])
    titles = []
    for url in urls:
        meta = link_meta.get(url, {})
        if meta.get("title"):
            titles.append(meta["title"])
    if titles:
        base += f" [link: {titles[0][:120]}]"

    return base


# ---------------------------------------------------------------------------
# Chunking engine
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    child_size: int = CHILD_CHUNK_TOKENS,
    parent_size: int = PARENT_CHUNK_TOKENS,
    overlap_ratio: float = OVERLAP_RATIO,
) -> list[dict]:
    """Split text into parent-child chunks with overlap.

    Returns a list of chunk records, each with ``{id, text, is_parent}``.
    Parent chunks wrap child chunks with full context.
    """
    text = text.strip()
    chunks: list[dict] = []
    child_id = 0
    parent_id = 0

    child_char = child_size * 3
    parent_char = parent_size * 3
    child_overlap = int(child_size * overlap_ratio) * 3
    parent_overlap = int(parent_size * overlap_ratio) * 3

    def _slide(start: int, chunk_char: int, overlap_char: int) -> int:
        """Advance start position by chunk - overlap, but never into the
        tail region where we'd endlessly inch forward 1 char at a time."""
        step = max(chunk_char - overlap_char, 1)
        next_pos = start + step
        if next_pos >= len(text):
            return len(text)  # signal: we're done
        return next_pos

    # Generate child chunks (small, high precision)
    child_start = 0
    while child_start < len(text):
        child_end = min(child_start + child_char, len(text))
        child_chunk_text = text[child_start:child_end].strip()
        if child_chunk_text:
            chunks.append({
                "id": f"child_{child_id}",
                "text": child_chunk_text,
                "is_parent": False,
            })
            child_id += 1
        child_start = _slide(child_start, child_char, child_overlap)
        if child_start >= len(text):
            break

    # Generate parent chunks (large, full LLM context)
    parent_start = 0
    while parent_start < len(text):
        parent_end = min(parent_start + parent_char, len(text))
        parent_chunk_text = text[parent_start:parent_end].strip()
        if parent_chunk_text:
            chunks.append({
                "id": f"parent_{parent_id}",
                "text": parent_chunk_text,
                "is_parent": True,
            })
            parent_id += 1
        parent_start = _slide(parent_start, parent_char, parent_overlap)
        if parent_start >= len(text):
            break

    return chunks



# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def build_all_chunks(
    bundles: list[dict],
    link_meta: dict[str, dict],
) -> list[dict]:
    """Build chunk records for every bundle.

    Each chunk record::

        {
            "chunk_id": str,
            "text": str,            # metadata prefix + chunk content
            "bundle_anchor_id": int,  # msg_id of the anchor
            "is_parent": bool,
            "token_estimate": int,
        }
    """
    all_chunks: list[dict] = []

    for bundle in bundles:
        anchor = bundle.get("anchor", {})
        anchor_id = anchor.get("msg_id", 0)

        # Build metadata prefix and full text
        prefix = build_metadata_prefix(bundle)
        full_prefix = build_full_metadata_prefix(bundle, link_meta)
        body = build_bundle_text(bundle, link_meta)

        if not body.strip():
            # No content to chunk — store a single stub record
            all_chunks.append({
                "chunk_id": f"bundle_{anchor_id}_stub",
                "text": f"{full_prefix}\n\n{prefix}",
                "bundle_anchor_id": anchor_id,
                "is_parent": False,
                "token_estimate": estimate_tokens(prefix),
            })
            continue

        # Prepend full metadata prefix to parent chunks, base prefix to children
        raw_chunks = chunk_text(body)

        for raw in raw_chunks:
            if raw["is_parent"]:
                chunk_text_final = f"{full_prefix}\n\n{raw['text']}"
            else:
                chunk_text_final = f"{prefix}\n\n{raw['text']}"

            all_chunks.append({
                "chunk_id": f"bundle_{anchor_id}_{raw['id']}",
                "text": chunk_text_final,
                "bundle_anchor_id": anchor_id,
                "is_parent": raw["is_parent"],
                "token_estimate": estimate_tokens(chunk_text_final),
            })

    return all_chunks


def run_chunking(
    db_path: str = db.DEFAULT_DB_PATH,
    bundles_path: str = "data/bundles.json",
    link_metadata_path: str = "data/link_metadata.json",
    output_path: str = "data/chunks.json",
) -> str:
    """Convenience wrapper: load bundles + metadata, chunk, write output.

    Writes to SQLite (chunks table) and JSON (backwards compat).
    """
    # Load bundles from SQLite (preferred) or JSON
    if db.db_exists(db_path):
        bundles = db.load_bundles(db_path)
    else:
        with open(bundles_path, encoding="utf-8") as f:
            bundles = json.load(f)

    link_meta = load_link_metadata(db_path, json_path=link_metadata_path)

    chunks = build_all_chunks(bundles, link_meta)

    # Write to SQLite
    with db.connect(db_path) as conn:
        db.upsert_chunks(conn, chunks)

    # Write JSON for backwards compat
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    return output_path


if __name__ == "__main__":
    output = run_chunking()
    print(f"Chunks written to: {output}")
