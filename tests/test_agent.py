import datetime as dt
from pathlib import Path

import pytest

import agent


def test_sanitize_filename_is_stable():
    url = "https://example.com/docs/page"
    title = "Hello World!"
    fname1 = agent.sanitize_filename(url, title)
    fname2 = agent.sanitize_filename(url, title)
    assert fname1 == fname2
    assert fname1.endswith(".md")


def test_convert_to_markdown_includes_metadata():
    html = """
    <html><head><title>My Page</title></head>
    <body><h1>My Page</h1><p>Content</p><script>ignore()</script></body></html>
    """
    md = agent.convert_to_markdown(html, "https://example.com/page")
    assert "title: My Page" in md
    assert "source: https://example.com/page" in md
    assert "Content" in md
    assert "ignore" not in md


def test_not_modified_returns_cached_path(monkeypatch, tmp_path):
    url = "https://example.com/docs"
    cached_file = tmp_path / "cached.md"
    cached_file.write_text("# Cached\n", encoding="utf-8")

    cache = {
        url: {
            "filename": "cached.md",
            "hash": "abc123",
            "fetched_at": dt.datetime.utcnow().isoformat() + "Z",
        }
    }

    def fake_fetch_with_requests(*args, **kwargs):
        raise agent.NotModified()

    monkeypatch.setattr(agent, "fetch_with_requests", fake_fetch_with_requests)

    path = agent.process_url(
        url=url,
        dest_dir=tmp_path,
        limiter=None,
        cache=cache,
        cache_path=tmp_path / "cache.json",
        robots_policy=None,
        use_browser=False,
        force=False,
        browser_timeout_ms=1000,
        browser_wait_until="load",
    )

    assert path == cached_file
