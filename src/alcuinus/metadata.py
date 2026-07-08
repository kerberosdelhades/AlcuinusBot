"""
Phase 3 — Metadata extraction from URLs.

For each unique URL found in anchors, fetch the page title and meta
description. Strategy is chosen by URL domain:

- **Generic HTML** — HTTP GET + BeautifulSoup (<title>, <meta description>).
- **GitHub API** — structured metadata for repos (description, topics).
  Non-repo paths (blobs, commits, issues) fall back to generic HTML.
- **arXiv API** — paper metadata (title, authors, abstract).
- **Unsupported** — YouTube, PDFs, images, etc. get a ``status: "unsupported"``
  record so they can be tracked without wasted fetches.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUEST_DELAY = 1.0  # seconds between requests (polite default)
REQUEST_TIMEOUT = 15  # seconds per HTTP request
USER_AGENT = (
    "AlcuinusBot/0.1 (+https://hq.ijuanes.ovh; Telegram link-analysis bot)"
)

# GitHub: match repo-level URLs only (owner/repo).
# Blobs, commits, issues, pulls, etc. are NOT matched — they fall through
# to generic HTML extraction so we still get the page <title>.
_GITHUB_REPO_RE = re.compile(
    r"^https?://github\.com/"  #
    r"([a-zA-Z0-9_.-]+)/"  # owner
    r"([a-zA-Z0-9_.-]+)"  # repo
    r"(?:/.*)?$"  # optional trailing stuff
)

# GitHub: strip /blob/..., /tree/..., /commit/... paths
_GITHUB_EXTRA_PATH = re.compile(r"/(?:blob|tree|commit|issues|pull|releases)/.*$")

# arXiv: /abs/{id} or /pdf/{id}.  Strip trailing .pdf if present.
_ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([^/\s#?]+?)(?:\.pdf)?$")

# Domains we know we cannot extract from — don't bother hitting them.
_UNSUPPORTED_DOMAINS = frozenset(
    {
        "youtube.com",
        "youtu.be",
        "twitter.com",
        "x.com",
        "t.co",
        "play.google.com",
        "bing.com",
    }
)

# Path extensions we skip (PDFs, images, binary files).
_UNSUPPORTED_EXTENSIONS = frozenset(
    {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".mp4", ".webm", ".zip", ".gz"}
)


# ---------------------------------------------------------------------------
# URL classification
# ---------------------------------------------------------------------------

def classify_url(url: str) -> str:
    """Return the strategy to use for a URL.

    Returns one of ``"github"``, ``"arxiv"``, ``"html"``, or ``"unsupported"``.
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    path = (parsed.path or "").lower()

    # ArXiv
    if "arxiv.org" in host:
        return "arxiv"

    # GitHub — only repo-level URLs (not user profiles, not blob/commit pages)
    if "github.com" in host and bool(_GITHUB_REPO_RE.match(parsed._replace(netloc=host).geturl())):
        # Repos with extra path segments (e.g. /blob/main/README.md)
        # are still valid GitHub API targets after stripping.
        return "github"

    # Known unsupported domains
    if host in _UNSUPPORTED_DOMAINS:
        return "unsupported"

    # Unsupported file extensions
    if any(path.endswith(ext) for ext in _UNSUPPORTED_EXTENSIONS):
        return "unsupported"

    # Fallback: generic HTML
    return "html"


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def fetch_github(url: str) -> dict:
    """Extract metadata from a GitHub repo via the public API.

    URL examples handled:
        https://github.com/{owner}/{repo}
        https://github.com/{owner}/{repo}/blob/main/README.md
    """
    # Normalise: strip trailing path segments so we get the repo root
    api_url = _GITHUB_EXTRA_PATH.sub("", url)
    api_url = api_url.replace("github.com", "api.github.com/repos", 1)

    try:
        resp = requests.get(
            api_url,
            headers={"Accept": "application/vnd.github+json", "User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 404:
            return {
                "url": url,
                "title": None,
                "description": None,
                "source": "github",
                "status": "not_found",
                "fetched_at": _now_iso(),
            }
        resp.raise_for_status()
        data = resp.json()
        return {
            "url": url,
            "title": data.get("full_name"),  # "owner/repo"
            "description": data.get("description"),
            "source": "github",
            "status": "ok",
            "fetched_at": _now_iso(),
        }

    except requests.RequestException as exc:
        return {
            "url": url,
            "title": None,
            "description": None,
            "source": "github",
            "status": f"error: {exc}",
            "fetched_at": _now_iso(),
        }


def fetch_arxiv(url: str) -> dict:
    """Extract metadata from an arXiv paper via the public API."""
    match = _ARXIV_ID_RE.search(url)
    if not match:
        return {
            "url": url,
            "title": None,
            "description": None,
            "source": "arxiv",
            "status": "bad_id",
            "fetched_at": _now_iso(),
        }

    arxiv_id = match.group(1)
    api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1"

    try:
        resp = requests.get(api_url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        entry = soup.find("entry")
        if entry is None:
            return {
                "url": url,
                "title": None,
                "description": None,
                "source": "arxiv",
                "status": "not_found",
                "fetched_at": _now_iso(),
            }

        title_el = entry.find("title")
        summary_el = entry.find("summary")

        return {
            "url": url,
            "title": title_el.text.strip() if title_el else None,
            "description": summary_el.text.strip()[:500] if summary_el else None,
            "source": "arxiv",
            "status": "ok",
            "fetched_at": _now_iso(),
        }

    except requests.RequestException as exc:
        return {
            "url": url,
            "title": None,
            "description": None,
            "source": "arxiv",
            "status": f"error: {exc}",
            "fetched_at": _now_iso(),
        }


def fetch_html(url: str) -> dict:
    """Extract <title> and <meta description> from a generic HTML page."""
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()

        # Only parse HTML content types
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            return {
                "url": url,
                "title": None,
                "description": None,
                "source": "html",
                "status": f"not_html ({content_type})",
                "fetched_at": _now_iso(),
            }

        soup = BeautifulSoup(resp.text, "html.parser")

        # Title
        title_tag = soup.find("title")
        title = title_tag.text.strip() if title_tag else None

        # Meta description (multiple forms)
        description = None
        for meta in soup.find_all("meta"):
            name = str(meta.get("name", "")).lower()
            prop = str(meta.get("property", "")).lower()
            if name in ("description", "og:description") or prop == "og:description":
                content = str(meta.get("content", "")).strip()
                if content:
                    description = content
                    break

        return {
            "url": url,
            "title": title,
            "description": description,
            "source": "html",
            "status": "ok",
            "fetched_at": _now_iso(),
        }

    except requests.RequestException as exc:
        return {
            "url": url,
            "title": None,
            "description": None,
            "source": "html",
            "status": f"error: {exc}",
            "fetched_at": _now_iso(),
        }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def fetch_metadata(
    url: str,
    *,
    cache: dict[str, dict] | None = None,
) -> dict:
    """Fetch metadata for a single URL. Uses *cache* to avoid duplicate requests.

    Returns a dict always containing at least:
    ``{url, title, description, source, status, fetched_at}``.
    """
    if cache is not None and url in cache:
        return cache[url]

    strategy = classify_url(url)

    if strategy == "github":
        record = fetch_github(url)
    elif strategy == "arxiv":
        record = fetch_arxiv(url)
    elif strategy == "html":
        record = fetch_html(url)
    else:
        record = {
            "url": url,
            "title": None,
            "description": None,
            "source": "unsupported",
            "status": "unsupported",
            "fetched_at": _now_iso(),
        }

    if cache is not None:
        cache[url] = record
    return record


def fetch_all_metadata(
    anchors: list[dict],
    *,
    delay: float = REQUEST_DELAY,
) -> list[dict]:
    """Extract metadata for every unique URL in *anchors*.

    URLs are de-duplicated across the entire anchor set.
    A delay is inserted between requests to be polite to servers.
    """
    # Collect unique URLs
    urls: set[str] = set()
    for anchor in anchors:
        for u in anchor.get("urls", []):
            urls.add(u)

    cache: dict[str, dict] = {}
    results: list[dict] = []

    for i, url in enumerate(sorted(urls)):
        if i > 0 and delay > 0:
            time.sleep(delay)
        record = fetch_metadata(url, cache=cache)
        results.append(record)

    return results


def run_metadata(
    input_path: str = "data/anchors.json",
    output_path: str = "data/link_metadata.json",
    delay: float = REQUEST_DELAY,
) -> str:
    """Convenience wrapper: load anchors, fetch metadata, write output.

    Returns path to the metadata JSON file.
    """
    with open(input_path, encoding="utf-8") as f:
        anchors = json.load(f)

    results = fetch_all_metadata(anchors, delay=delay)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return output_path


if __name__ == "__main__":
    output = run_metadata()
    print(f"Link metadata written to: {output}")
