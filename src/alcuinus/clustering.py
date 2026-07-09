"""
Phase 6 — Bundle clustering

Groups bundles by topic using KMeans on mistral-embed vectors from Zvec.
Extracts representative keywords per cluster via TF-IDF on chunk text.

**Why KMeans over BERTopic for 71 bundles:**
- BERTopic uses UMAP (needs ~200+ points for stable manifold learning).
  At 71 bundles, UMAP is unreliable and can hallucinate structure.
- KMeans is deterministic, fast, and works well at this scale.
- If corpus grows to 500+ bundles, BERTopic becomes the better choice.
  The ``ClusteringEngine`` abstraction makes swapping easy.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any

import numpy as np
import zvec
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_K = 8  # number of clusters (heuristic: ~9 bundles per cluster at 71)
DEFAULT_INDEX_PATH = "data/zvec_index"
DEFAULT_CHUNKS_PATH = "data/chunks.json"
DEFAULT_OUTPUT_PATH = "data/clusters.json"
TOP_KEYWORDS = 5  # keywords per cluster


# ---------------------------------------------------------------------------
# Clustering engine
# ---------------------------------------------------------------------------


def fetch_vectors_from_zvec(
    index_path: str = DEFAULT_INDEX_PATH,
) -> dict[str, dict]:
    """Fetch all documents from Zvec index.

    Returns dict mapping chunk_id → {"vector": list[float], "text": str,
    "bundle_anchor_id": int, "is_parent": bool}.
    """
    collection = zvec.open(path=index_path)

    # Get all IDs from the index (query with a dummy vector to get all)
    # We use a zero vector to get everything — ANN returns nearest first,
    # but at this scale (174 docs) it returns all.
    dummy = [0.0] * 1024
    hits = collection.query(
        zvec.Query(field_name="embedding", vector=dummy),
        topk=10000,  # large enough to get everything
    )

    # Now fetch full data for each hit
    ids = [h.id for h in hits]
    docs = collection.fetch(ids=ids, include_vector=True)

    result = {}
    for doc_id, doc in docs.items():
        result[doc_id] = {
            "vector": doc.vector("embedding"),
            "text": doc.field("text") or "",
            "bundle_anchor_id": doc.field("bundle_anchor_id"),
            "is_parent": doc.field("is_parent"),
        }

    return result


def cluster_vectors(
    vectors: dict[str, dict],
    k: int = DEFAULT_K,
) -> dict[str, Any]:
    """Run KMeans clustering on the vectors.

    Returns:
        {
            "k": int,
            "clusters": {
                "0": {"label": "cluster_0", "chunk_ids": [...], "size": N},
                ...
            },
            "assignments": {"chunk_id": cluster_label, ...}
        }
    """
    if len(vectors) < k:
        k = max(1, len(vectors) // 2)

    ids = list(vectors.keys())
    X = np.array([vectors[cid]["vector"] for cid in ids])

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    clusters: dict[str, dict] = defaultdict(lambda: {"chunk_ids": [], "size": 0})
    assignments: dict[str, str] = {}

    for chunk_id, label in zip(ids, labels):
        cluster_key = str(label)
        clusters[cluster_key]["chunk_ids"].append(chunk_id)
        clusters[cluster_key]["size"] += 1
        assignments[chunk_id] = cluster_key

    # Add labels
    for key in clusters:
        clusters[key]["label"] = f"cluster_{key}"

    return {
        "k": k,
        "clusters": dict(clusters),
        "assignments": assignments,
    }


def extract_keywords(
    clusters: dict[str, dict],
    vectors: dict[str, dict],
    top_n: int = TOP_KEYWORDS,
) -> dict[str, list[str]]:
    """Extract top keywords per cluster using TF-IDF on chunk texts.

    Returns dict mapping cluster_label → list of keywords.
    """
    # Collect all texts per cluster
    cluster_texts: dict[str, list[str]] = defaultdict(list)
    for cluster_key, info in clusters.items():
        for chunk_id in info["chunk_ids"]:
            text = vectors.get(chunk_id, {}).get("text", "")
            if text:
                cluster_texts[cluster_key].append(text)

    # Run TF-IDF on the concatenated cluster texts
    corpus = []
    cluster_keys = []
    for key in sorted(cluster_texts.keys()):
        combined = " ".join(cluster_texts[key])
        corpus.append(combined)
        cluster_keys.append(key)

    if not corpus:
        return {}

    # Remove Spanish/English stopwords + common metadata tokens
    stop_words = list(_get_stopwords())

    tfidf = TfidfVectorizer(
        max_features=500,
        stop_words=stop_words,
        min_df=1,
        max_df=0.95,
    )

    try:
        tfidf_matrix = tfidf.fit_transform(corpus)
    except ValueError:
        # Empty vocabulary (all stop words)
        return {f"cluster_{k}": [] for k in cluster_keys}

    feature_names = tfidf.get_feature_names_out()

    keywords: dict[str, list[str]] = {}
    for i, key in enumerate(cluster_keys):
        row = tfidf_matrix[i].toarray().flatten()
        top_indices = row.argsort()[-top_n:][::-1]
        top_terms = [feature_names[j] for j in top_indices if row[j] > 0]
        keywords[f"cluster_{key}"] = top_terms

    return keywords


def _get_stopwords() -> set[str]:
    """Custom stopwords for es/en mixed Telegram content."""
    # Common English stopwords
    en = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "must", "to", "of",
        "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
        "during", "before", "after", "above", "below", "between", "out", "off",
        "over", "under", "again", "further", "then", "once", "here", "there",
        "when", "where", "why", "how", "all", "both", "each", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own",
        "same", "so", "than", "too", "very", "just", "because", "but", "and",
        "or", "if", "while", "about", "up", "down", "this", "that", "these",
        "those", "it", "its", "i", "me", "my", "we", "our", "you", "your",
        "he", "him", "his", "she", "her", "they", "them", "their", "what",
        "which", "who", "whom", "http", "https", "com", "www", "amp",
    }
    # Common Spanish stopwords
    es = {
        "el", "la", "los", "las", "un", "una", "unos", "unas", "y", "o",
        "pero", "sino", "que", "como", "cuando", "donde", "si", "no", "ni",
        "de", "del", "en", "por", "para", "con", "sin", "sobre", "entre",
        "desde", "hasta", "hacia", "segun", "durante", "ante", "bajo",
        "contra", "desde", "hacia", "segun", "sobre", "tras", "a", "al",
        "lo", "le", "les", "me", "te", "se", "nos", "os", "su", "sus",
        "mi", "mis", "tu", "tus", "nuestro", "nuestra", "vuestro", "vuestra",
        "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
        "aquel", "aquella", "aquellos", "aquellas", "ser", "estar", "haber",
        "tener", "hacer", "ir", "poder", "decir", "dar", "saber", "querer",
        "ver", "venir", "llevar", "poner", "salir", "volver", "tomar",
        "como", "más", "muy", "también", "ya", "solo", "hay", "era",
        "sido", "ser", "fue", "son", "están", "está", "ha", "han",
    }
    return en | es


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_clustering(
    index_path: str = DEFAULT_INDEX_PATH,
    chunks_path: str = DEFAULT_CHUNKS_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    k: int = DEFAULT_K,
) -> str:
    """Convenience wrapper: fetch vectors, cluster, extract keywords, write output.

    Returns path to the clusters JSON file.
    """
    # Fetch vectors from Zvec
    vectors = fetch_vectors_from_zvec(index_path)

    # Cluster
    result = cluster_vectors(vectors, k=k)

    # Extract keywords
    keywords = extract_keywords(result["clusters"], vectors)

    # Attach keywords to clusters
    for cluster_key, info in result["clusters"].items():
        label = info["label"]
        info["keywords"] = keywords.get(label, [])

    # Load chunk metadata for bundle info
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)
    chunk_meta = {c["chunk_id"]: c for c in chunks}

    # Add bundle info to each cluster
    for cluster_key, info in result["clusters"].items():
        bundle_ids = set()
        for cid in info["chunk_ids"]:
            meta = chunk_meta.get(cid, {})
            bundle_ids.add(meta.get("bundle_anchor_id"))
        info["bundle_ids"] = sorted(bundle_ids)

    # Write output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return output_path


if __name__ == "__main__":
    output = run_clustering()
    print(f"Clusters written to: {output}")
