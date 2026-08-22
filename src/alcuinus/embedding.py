"""
Phase 5 — Embedding + Zvec

Generate 1024d embeddings via Mistral's ``mistral-embed`` API and store
them in a local Zvec index (in-process, Apache 2.0).

Zvec API notes (discovered during verification):
- ``create_and_open(path, schema)`` — path must NOT exist yet
- ``collection.insert([Doc(...)])`` — batch insert
- ``collection.fetch(ids=[...], include_vector=True)`` — fetch by ID
- ``collection.query(Query(...), topk=N)`` — ANN search
- Python wrappers are lowercase: ``fetch``, ``destroy``, ``doc.id``,
  ``doc.vector('name')``, ``doc.score`` (property)
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import zvec
from mistralai.client import Mistral

from alcuinus import db

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EMBED_MODEL = "mistral-embed"
EMBED_DIM = 1024
BATCH_SIZE = 50  # chunks per API call (conservative default)
DEFAULT_INDEX_PATH = "data/zvec_index"
DEFAULT_CHUNKS_PATH = "data/chunks.json"


# ---------------------------------------------------------------------------
# Zvec schema
# ---------------------------------------------------------------------------


def build_schema() -> zvec.CollectionSchema:
    """Build the Zvec collection schema for chunks.

    Schema:
        - ID field (auto, string): chunk_id
        - Vector: embedding (FP32, 1024d)
        - Scalar: text (string), bundle_anchor_id (int), is_parent (bool),
          token_estimate (int)
    """
    return zvec.CollectionSchema(
        name="chunks",
        vectors=zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32, EMBED_DIM),
        fields=[
            zvec.FieldSchema("text", zvec.DataType.STRING),
            zvec.FieldSchema("bundle_anchor_id", zvec.DataType.INT64),
            zvec.FieldSchema("is_parent", zvec.DataType.BOOL),
            zvec.FieldSchema("token_estimate", zvec.DataType.INT64),
        ],
    )


# ---------------------------------------------------------------------------
# Embedding API
# ---------------------------------------------------------------------------


def embed_texts(
    texts: list[str],
    *,
    api_key: str,
    model: str = EMBED_MODEL,
    batch_size: int = BATCH_SIZE,
) -> list[list[float]]:
    """Embed a list of texts via Mistral API, in batches.

    Returns a list of embedding vectors (list of 1024 floats), one per input.
    """
    client = Mistral(api_key=api_key)
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        result = client.embeddings.create(model=model, inputs=batch)
        for item in result.data:
            if item.embedding is None:
                raise ValueError(f"Embedding returned None for batch item {item.index}")
            all_embeddings.append(item.embedding)

    return all_embeddings


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------


def create_index(index_path: str, schema: zvec.CollectionSchema) -> zvec.Collection:
    """Create a new Zvec index. Deletes existing index at that path if present."""
    path = Path(index_path)
    if path.exists():
        shutil.rmtree(path)
    return zvec.create_and_open(path=str(path), schema=schema)


def open_index(index_path: str) -> zvec.Collection:
    """Open an existing Zvec index."""
    return zvec.open(path=index_path)


def _hash_text(text: str) -> str:
    """Return SHA-256 hex digest of *text* (UTF-8). Duplicate of
    ``db._hash_text`` — duplicated here to avoid importing a private name."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Index probing
# ---------------------------------------------------------------------------


def get_indexed_chunk_ids(
    index_path: str,
    known_ids: list[str],
) -> set[str]:
    """Given a list of chunk_ids (from SQLite), return the subset that
    exists in the Zvec index. Uses fetch() to probe."""
    collection = open_index(index_path)
    docs = collection.fetch(ids=known_ids, include_vector=False)
    return set(docs.keys())


# ---------------------------------------------------------------------------
# Embedding delta computation (pure function)
# ---------------------------------------------------------------------------


def compute_embedding_delta(
    chunk_ids: list[str],
    chunk_hashes: dict[str, str],
    indexed_ids: set[str],
    indexed_hashes: dict[str, str],
) -> dict:
    """Compute which chunks need embedding, deletion, or are unchanged.

    Returns: {"to_embed": list[str], "to_delete": set[str], "unchanged": set[str]}
    """
    sqlite_ids = set(chunk_ids)
    to_embed = []
    unchanged = set()
    for cid in chunk_ids:
        if cid not in indexed_ids:
            to_embed.append(cid)
        elif chunk_hashes.get(cid) != indexed_hashes.get(cid):
            to_embed.append(cid)
        else:
            unchanged.add(cid)
    to_delete = indexed_ids - sqlite_ids
    return {"to_embed": to_embed, "to_delete": to_delete, "unchanged": unchanged}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def embed_and_store(
    chunks_path: str = DEFAULT_CHUNKS_PATH,
    index_path: str = DEFAULT_INDEX_PATH,
    api_key: str | None = None,
    batch_size: int = BATCH_SIZE,
) -> str:
    """Load chunks, embed via Mistral, insert into Zvec.

    Returns the path to the Zvec index directory.
    """
    if api_key is None:
        api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        raise ValueError(
            "MISTRAL_API_KEY not set. Export it or pass api_key= explicitly."
        )

    # Load chunks
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}")

    # Extract texts for embedding
    texts = [c["text"] for c in chunks]

    # Generate embeddings
    embeddings = embed_texts(texts, api_key=api_key, batch_size=batch_size)
    assert len(embeddings) == len(chunks), (
        f"Embedding count mismatch: {len(embeddings)} vs {len(chunks)}"
    )

    # Build schema and create index
    schema = build_schema()
    collection = create_index(index_path, schema)

    # Build Doc objects and insert in batches (Zvec limit: 1024 docs/batch)
    ZVEC_BATCH = 1024
    for i in range(0, len(chunks), ZVEC_BATCH):
        batch_docs = []
        for chunk, embedding in zip(
            chunks[i : i + ZVEC_BATCH],
            embeddings[i : i + ZVEC_BATCH],
        ):
            doc = zvec.Doc(
                id=chunk["chunk_id"],
                vectors={"embedding": embedding},
                fields={
                    "text": chunk["text"],
                    "bundle_anchor_id": chunk["bundle_anchor_id"],
                    "is_parent": chunk["is_parent"],
                    "token_estimate": chunk["token_estimate"],
                },
            )
            batch_docs.append(doc)
        collection.insert(batch_docs)

    collection.flush()

    return index_path


def _upsert_docs(
    collection: zvec.Collection,
    chunks: list[dict],
    embeddings: list[list[float]],
) -> None:
    """Build zvec.Doc objects from *chunks* + *embeddings* and upsert them
    into *collection* in batches of 1024 (Zvec batch limit)."""
    ZVEC_BATCH = 1024
    for i in range(0, len(chunks), ZVEC_BATCH):
        batch_docs = []
        for chunk, embedding in zip(
            chunks[i : i + ZVEC_BATCH],
            embeddings[i : i + ZVEC_BATCH],
        ):
            doc = zvec.Doc(
                id=chunk["chunk_id"],
                vectors={"embedding": embedding},
                fields={
                    "text": chunk["text"],
                    "bundle_anchor_id": chunk["bundle_anchor_id"],
                    "is_parent": chunk["is_parent"],
                    "token_estimate": chunk["token_estimate"],
                },
            )
            batch_docs.append(doc)
        collection.upsert(batch_docs)


def embed_and_store_incremental(
    db_path: str = db.DEFAULT_DB_PATH,
    index_path: str = DEFAULT_INDEX_PATH,
    api_key: str | None = None,
    batch_size: int = BATCH_SIZE,
    force_full: bool = False,
) -> str:
    """Incremental embedding pipeline: embed new/changed chunks, delete stale
    ones, and leave the rest untouched.

    Returns the path to the Zvec index directory.
    """
    # 1. Get api_key from env if not provided
    if api_key is None:
        api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        raise ValueError(
            "MISTRAL_API_KEY not set. Export it or pass api_key= explicitly."
        )

    # 2. Load chunks from SQLite
    chunks = db.load_chunks(db_path)
    if not chunks:
        raise ValueError(f"No chunks found in {db_path}")

    # 3. Compute chunk_hashes
    chunk_hashes = {
        c["chunk_id"]: c.get("text_hash") or _hash_text(c["text"]) for c in chunks
    }
    chunk_ids = [c["chunk_id"] for c in chunks]

    # 4. Check if index exists
    index_exists = Path(index_path).exists()

    # 5. Full (re)build path
    if not index_exists or force_full:
        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts, api_key=api_key, batch_size=batch_size)
        assert len(embeddings) == len(chunks), (
            f"Embedding count mismatch: {len(embeddings)} vs {len(chunks)}"
        )

        # create_index nukes (rmtree) any existing index and creates fresh.
        # This is the right behavior for both cold start and force_full.
        schema = build_schema()
        collection = create_index(index_path, schema)

        _upsert_docs(collection, chunks, embeddings)
        collection.flush()
        collection.optimize()
        db.set_meta(db_path, key="indexed_hashes", value=json.dumps(chunk_hashes))
        return index_path

    # 6. Incremental path
    stored = db.get_meta(db_path, key="indexed_hashes")
    indexed_hashes = json.loads(stored) if stored else {}

    collection = open_index(index_path)
    fetched = collection.fetch(ids=chunk_ids, include_vector=False)
    indexed_ids = set(fetched.keys())

    delta = compute_embedding_delta(
        chunk_ids, chunk_hashes, indexed_ids, indexed_hashes
    )
    to_embed = delta["to_embed"]
    to_delete = delta["to_delete"]
    unchanged = delta["unchanged"]

    if not to_embed and not to_delete:
        print("no changes — index is up to date")
        return index_path

    if to_embed:
        delta_chunks = [c for c in chunks if c["chunk_id"] in set(to_embed)]
        texts = [c["text"] for c in delta_chunks]
        embeddings = embed_texts(texts, api_key=api_key, batch_size=batch_size)
        _upsert_docs(collection, delta_chunks, embeddings)

    if to_delete:
        collection.delete(ids=list(to_delete))

    collection.flush()
    collection.optimize()

    # Update stored hashes
    for cid in to_embed:
        indexed_hashes[cid] = chunk_hashes[cid]
    for cid in to_delete:
        indexed_hashes.pop(cid, None)
    db.set_meta(db_path, key="indexed_hashes", value=json.dumps(indexed_hashes))

    print(
        f"incremental update: embedded {len(to_embed)}, "
        f"deleted {len(to_delete)}, unchanged {len(unchanged)}"
    )
    return index_path


def run_embedding(
    chunks_path: str = DEFAULT_CHUNKS_PATH,  # kept for backwards compat; chunks are read from SQLite
    index_path: str = DEFAULT_INDEX_PATH,
    force_full: bool = False,
) -> str:
    """Embed chunks and store in Zvec. Auto-detects incremental vs full.

    If the Zvec index exists and force_full=False, only new/changed
    chunks are embedded. If the index doesn't exist or force_full=True,
    all chunks are embedded from scratch.

    Returns path to the Zvec index directory.
    """
    return embed_and_store_incremental(
        index_path=index_path,
        force_full=force_full,
    )


if __name__ == "__main__":
    output = run_embedding()
    print(f"Zvec index written to: {output}")
