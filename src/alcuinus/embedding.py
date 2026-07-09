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

import json
import os
import shutil
from pathlib import Path
from typing import Any

import zvec
from mistralai.client import Mistral

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

    # Build Doc objects and insert
    docs = []
    for chunk, embedding in zip(chunks, embeddings):
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
        docs.append(doc)

    collection.insert(docs)
    collection.flush()

    return index_path


def run_embedding(
    chunks_path: str = DEFAULT_CHUNKS_PATH,
    index_path: str = DEFAULT_INDEX_PATH,
) -> str:
    """Convenience wrapper: embed all chunks and store in Zvec.

    Returns path to the Zvec index directory.
    """
    return embed_and_store(chunks_path=chunks_path, index_path=index_path)


if __name__ == "__main__":
    output = run_embedding()
    print(f"Zvec index written to: {output}")
