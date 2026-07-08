"""Tests for Phase 3 — metadata extraction."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from alcuinus.metadata import (
    _ARXIV_ID_RE,
    _GITHUB_EXTRA_PATH,
    _GITHUB_REPO_RE,
    classify_url,
    fetch_all_metadata,
    fetch_arxiv,
    fetch_github,
    fetch_html,
    fetch_metadata,
    run_metadata,
)


# ---------------------------------------------------------------------------
# classify_url
# ---------------------------------------------------------------------------


class TestClassifyUrl:
    def test_github(self):
        assert classify_url("https://github.com/owner/repo") == "github"

    def test_github_with_trailing_path(self):
        # blob paths still resolve to the same repo via API
        assert (
            classify_url(
                "https://github.com/owner/repo/blob/main/README.md"
            )
            == "github"
        )

    def test_github_gist_not_repo(self):
        # gist URLs are not repos
        assert (
            classify_url("https://gist.github.com/user/abc123") == "html"
        )

    def test_arxiv_abs(self):
        assert (
            classify_url("https://arxiv.org/abs/2312.00752") == "arxiv"
        )

    def test_arxiv_pdf(self):
        assert (
            classify_url("https://arxiv.org/pdf/2312.00752.pdf") == "arxiv"
        )

    def test_youtube_unsupported(self):
        assert (
            classify_url("https://www.youtube.com/watch?v=abc123")
            == "unsupported"
        )
        assert (
            classify_url("https://youtu.be/abc123") == "unsupported"
        )

    def test_twitter_unsupported(self):
        assert (
            classify_url("https://twitter.com/user/status/123") == "unsupported"
        )
        assert (
            classify_url("https://x.com/user/status/123") == "unsupported"
        )

    def test_pdf_unsupported(self):
        assert (
            classify_url("https://example.com/paper.pdf") == "unsupported"
        )

    def test_png_unsupported(self):
        assert (
            classify_url("https://example.com/image.png") == "unsupported"
        )

    def test_generic_html(self):
        assert classify_url("https://example.com/article") == "html"

    def test_reddit_is_html(self):
        assert (
            classify_url(
                "https://www.reddit.com/r/ChatGPTPro/comments/abc"
            )
            == "html"
        )

    def test_www_stripped(self):
        # www subdomain should not affect classification
        assert (
            classify_url("https://www.github.com/owner/repo") == "github"
        )


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------


class TestGitHubRegex:
    def test_match_simple_repo(self):
        m = _GITHUB_REPO_RE.match("https://github.com/owner/repo")
        assert m is not None
        assert m.group(1) == "owner"
        assert m.group(2) == "repo"

    def test_match_with_blob(self):
        m = _GITHUB_REPO_RE.match(
            "https://github.com/owner/repo/blob/main/foo.py"
        )
        assert m is not None
        assert m.group(1) == "owner"
        assert m.group(2) == "repo"

    def test_match_with_commit(self):
        m = _GITHUB_REPO_RE.match(
            "https://github.com/owner/repo/commit/deadbeef"
        )
        assert m is not None

    def test_no_match_user_profile(self):
        # github.com/user (no repo) should not match
        assert _GITHUB_REPO_RE.match("https://github.com/torvalds") is None

    def test_no_match_settings(self):
        assert _GITHUB_REPO_RE.match("https://github.com/settings") is None

    def test_extra_path_stripper(self):
        stripped = _GITHUB_EXTRA_PATH.sub(
            "", "https://github.com/owner/repo/blob/main/README.md"
        )
        assert stripped == "https://github.com/owner/repo"

    def test_extra_path_stripper_commit(self):
        stripped = _GITHUB_EXTRA_PATH.sub(
            "", "https://github.com/owner/repo/commit/deadbeef"
        )
        assert stripped == "https://github.com/owner/repo"


class TestArxivRegex:
    def test_match_abs(self):
        m = _ARXIV_ID_RE.search("https://arxiv.org/abs/2312.00752")
        assert m is not None
        assert m.group(1) == "2312.00752"

    def test_match_pdf(self):
        m = _ARXIV_ID_RE.search("https://arxiv.org/pdf/2312.00752.pdf")
        assert m is not None
        assert m.group(1) == "2312.00752"

    def test_no_match_search(self):
        # search pages are not paper IDs
        m = _ARXIV_ID_RE.search(
            "https://arxiv.org/search?query=transformers"
        )
        assert m is None


# ---------------------------------------------------------------------------
# Fetchers (mocked)
# ---------------------------------------------------------------------------


class TestFetchGithub:
    def test_ok(self, monkeypatch):
        import requests

        class FakeResp:
            status_code = 200
            headers = {}

            @staticmethod
            def json():
                return {
                    "full_name": "owner/repo",
                    "description": "A test repo",
                }

            @staticmethod
            def raise_for_status():
                pass

        def fake_get(url, **kwargs):
            return FakeResp()

        monkeypatch.setattr(requests, "get", fake_get)

        result = fetch_github("https://github.com/owner/repo")
        assert result["title"] == "owner/repo"
        assert result["description"] == "A test repo"
        assert result["source"] == "github"
        assert result["status"] == "ok"

    def test_not_found(self, monkeypatch):
        import requests

        class FakeResp:
            status_code = 404
            headers = {}

            @staticmethod
            def raise_for_status():
                raise requests.HTTPError("404")

        monkeypatch.setattr(requests, "get", lambda *a, **kw: FakeResp())

        result = fetch_github("https://github.com/owner/nonexistent")
        assert result["status"] == "not_found"
        assert result["source"] == "github"

    def test_network_error(self, monkeypatch):
        import requests

        def fake_get(*args, **kwargs):
            raise requests.ConnectionError("timeout")

        monkeypatch.setattr(requests, "get", fake_get)

        result = fetch_github("https://github.com/owner/repo")
        assert result["source"] == "github"
        assert result["status"].startswith("error:")
        assert result["title"] is None


class TestFetchArxiv:
    def test_ok(self, monkeypatch):
        import requests

        xml = """<feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
              <title>Test Paper</title>
              <summary>A test abstract.</summary>
            </entry>
        </feed>"""

        class FakeResp:
            status_code = 200
            text = xml
            headers = {}

            @staticmethod
            def raise_for_status():
                pass

        monkeypatch.setattr(requests, "get", lambda *a, **kw: FakeResp())

        result = fetch_arxiv("https://arxiv.org/abs/2312.00752")
        assert result["title"] == "Test Paper"
        assert result["description"] == "A test abstract."
        assert result["source"] == "arxiv"
        assert result["status"] == "ok"

    def test_bad_id(self):
        result = fetch_arxiv("https://arxiv.org/search?q=test")
        assert result["status"] == "bad_id"
        assert result["title"] is None

    def test_not_found(self, monkeypatch):
        import requests

        xml = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'

        class FakeResp:
            status_code = 200
            text = xml
            headers = {}

            @staticmethod
            def raise_for_status():
                pass

        monkeypatch.setattr(requests, "get", lambda *a, **kw: FakeResp())

        result = fetch_arxiv("https://arxiv.org/abs/9999.99999")
        assert result["status"] == "not_found"

    def test_network_error(self, monkeypatch):
        import requests

        monkeypatch.setattr(
            requests,
            "get",
            lambda *a, **kw: (_ for _ in ()).throw(
                requests.ConnectionError("timeout")
            ),
        )

        result = fetch_arxiv("https://arxiv.org/abs/2312.00752")
        assert result["status"].startswith("error:")


class TestFetchHtml:
    def test_ok(self, monkeypatch):
        import requests

        html = """<html>
            <head>
                <title>Test Page</title>
                <meta name="description" content="A test description">
            </head>
        </html>"""

        class FakeResp:
            status_code = 200
            text = html
            headers = {"Content-Type": "text/html; charset=utf-8"}

            @staticmethod
            def raise_for_status():
                pass

        monkeypatch.setattr(requests, "get", lambda *a, **kw: FakeResp())

        result = fetch_html("https://example.com/page")
        assert result["title"] == "Test Page"
        assert result["description"] == "A test description"
        assert result["source"] == "html"
        assert result["status"] == "ok"

    def test_no_meta_description(self, monkeypatch):
        import requests

        html = "<html><head><title>Title Only</title></head></html>"

        class FakeResp:
            status_code = 200
            text = html
            headers = {"Content-Type": "text/html; charset=utf-8"}

            @staticmethod
            def raise_for_status():
                pass

        monkeypatch.setattr(requests, "get", lambda *a, **kw: FakeResp())

        result = fetch_html("https://example.com/page")
        assert result["title"] == "Title Only"
        assert result["description"] is None
        assert result["status"] == "ok"

    def test_og_description(self, monkeypatch):
        import requests

        html = """<html><head>
            <meta property="og:description" content="OG desc">
        </head></html>"""

        class FakeResp:
            status_code = 200
            text = html
            headers = {"Content-Type": "text/html; charset=utf-8"}

            @staticmethod
            def raise_for_status():
                pass

        monkeypatch.setattr(requests, "get", lambda *a, **kw: FakeResp())

        result = fetch_html("https://example.com/page")
        assert result["description"] == "OG desc"

    def test_not_html_content_type(self, monkeypatch):
        import requests

        class FakeResp:
            status_code = 200
            text = "binary stuff"
            headers = {"Content-Type": "application/pdf"}

            @staticmethod
            def raise_for_status():
                pass

        monkeypatch.setattr(requests, "get", lambda *a, **kw: FakeResp())

        result = fetch_html("https://example.com/doc.pdf")
        assert result["status"].startswith("not_html")
        assert result["title"] is None

    def test_network_error(self, monkeypatch):
        import requests

        monkeypatch.setattr(
            requests,
            "get",
            lambda *a, **kw: (_ for _ in ()).throw(
                requests.ConnectionError("timeout")
            ),
        )

        result = fetch_html("https://example.com/page")
        assert result["status"].startswith("error:")


# ---------------------------------------------------------------------------
# fetch_metadata — dispatch + caching
# ---------------------------------------------------------------------------


class TestFetchMetadata:
    def test_unsupported_returns_immediately(self):
        result = fetch_metadata("https://youtube.com/watch?v=abc")
        assert result["source"] == "unsupported"
        assert result["status"] == "unsupported"
        assert result["title"] is None

    def test_cache_deduplication(self, monkeypatch):
        """When the same URL is fetched twice, the second call must return
        the cached record without making another network request."""
        import requests

        call_count = 0

        def fake_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            html = "<html><head><title>Cached</title></head></html>"

            class Resp:
                status_code = 200
                text = html
                headers = {"Content-Type": "text/html"}

                @staticmethod
                def raise_for_status():
                    pass

            return Resp()

        monkeypatch.setattr(requests, "get", fake_get)

        cache: dict[str, dict] = {}
        r1 = fetch_metadata("https://example.com/page", cache=cache)
        r2 = fetch_metadata("https://example.com/page", cache=cache)

        assert r1 is r2  # same object
        assert call_count == 1  # only one network call


# ---------------------------------------------------------------------------
# fetch_all_metadata — integration with fake anchors
# ---------------------------------------------------------------------------


class TestFetchAllMetadata:
    def test_gathers_unique_urls(self, monkeypatch):
        """Should fetch each unique URL exactly once across all anchors."""
        import requests

        urls_seen: list[str] = []

        def fake_get(url, **kwargs):
            urls_seen.append(url)
            html = "<html><head><title>X</title></head></html>"

            class Resp:
                status_code = 200
                text = html
                headers = {"Content-Type": "text/html"}

                @staticmethod
                def raise_for_status():
                    pass

            return Resp()

        monkeypatch.setattr(requests, "get", fake_get)

        anchors = [
            {"urls": ["https://a.com/1", "https://a.com/2"]},
            {"urls": ["https://a.com/1"]},  # duplicate
            {"urls": ["https://a.com/3"]},
        ]

        results = fetch_all_metadata(anchors, delay=0.0)
        assert len(results) == 3  # unique URLs only
        assert len(urls_seen) == 3


# ---------------------------------------------------------------------------
# run_metadata — JSON I/O round-trip
# ---------------------------------------------------------------------------


class TestRunMetadata:
    def test_roundtrip(self, monkeypatch, tmp_path):
        """Write anchors to a temp file, run the pipeline, read output back."""
        import requests

        anchors = [{"urls": ["https://example.com/a", "https://example.com/b"]}]
        input_path = tmp_path / "anchors.json"
        input_path.write_text(json.dumps(anchors), encoding="utf-8")
        output_path = tmp_path / "out.json"

        html = "<html><head><title>T</title></head></html>"

        def fake_get(*args, **kwargs):
            class Resp:
                status_code = 200
                text = html
                headers = {"Content-Type": "text/html"}

                @staticmethod
                def raise_for_status():
                    pass

            return Resp()

        monkeypatch.setattr(requests, "get", fake_get)

        result_path = run_metadata(
            str(input_path), str(output_path), delay=0.0
        )
        assert result_path == str(output_path)
        assert output_path.exists()

        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert len(data) == 2
        assert data[0]["url"].startswith("https://example.com/")
        assert data[0]["title"] == "T"
        assert data[0]["source"] == "html"
        assert data[0]["status"] == "ok"
        assert "fetched_at" in data[0]
