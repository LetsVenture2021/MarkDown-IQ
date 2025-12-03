#!/usr/bin/env python3
"""
Documentation Harvester Agent

Fetches public documentation pages, converts them to Markdown, and stores them
locally with simple metadata for downstream AI use.

Use responsibly: respect robots.txt and site Terms of Service.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse
from urllib import robotparser

try:
    import requests
    from bs4 import BeautifulSoup
    from markdownify import markdownify as html_to_md
except ImportError as exc:  # pragma: no cover - dependency guard
    missing = (
        "Missing dependencies. Install with: pip install -r requirements.txt\n"
        f"Original error: {exc}"
    )
    print(missing, file=sys.stderr)
    sys.exit(1)

USER_AGENT = "DocHarvester/0.1 (+https://example.com; for personal documentation use)"
DEFAULT_MIN_DELAY = 1.0
DEFAULT_CACHE_PATH = Path(".cache/harvest_cache.json")


class NotModified(Exception):
    """Raised when the server returns 304 Not Modified."""


class RateLimiter:
    def __init__(self, min_interval: float = DEFAULT_MIN_DELAY):
        self.min_interval = max(0.0, min_interval)
        self._last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        sleep_for = self.min_interval - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)
        self._last = time.monotonic()


class RobotsPolicy:
    def __init__(self, user_agent: str = USER_AGENT):
        self.user_agent = user_agent
        self._parsers: Dict[str, robotparser.RobotFileParser] = {}

    def is_allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False
        base = f"{parsed.scheme}://{parsed.netloc}"
        rp = self._parsers.get(base)
        if not rp:
            rp = robotparser.RobotFileParser()
            rp.set_url(base + "/robots.txt")
            try:
                rp.read()
            except Exception:
                # If robots.txt cannot be read, default to disallow to be safe.
                return False
            self._parsers[base] = rp
        return rp.can_fetch(self.user_agent, url)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch URLs and convert to Markdown.")
    parser.add_argument("--url", action="append", help="URL to fetch (repeatable).")
    parser.add_argument(
        "--url-file",
        type=Path,
        help="Path to file containing URLs (one per line, '#' for comments).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs"),
        help="Directory to write Markdown files into (default: docs/).",
    )
    parser.add_argument(
        "--min-delay",
        type=float,
        default=DEFAULT_MIN_DELAY,
        help=f"Minimum seconds between fetches (default: {DEFAULT_MIN_DELAY}).",
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help="Path to cache file for ETag/Last-Modified tracking.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cache and force re-fetch.",
    )
    parser.add_argument(
        "--skip-robots",
        action="store_true",
        help="Skip robots.txt enforcement (use only if permitted).",
    )
    parser.add_argument(
        "--use-browser",
        action="store_true",
        help="Fetch page with Playwright (headless Chromium) for dynamic content.",
    )
    parser.add_argument(
        "--browser-timeout",
        type=int,
        default=30000,
        help="Playwright navigation timeout in ms (default: 30000).",
    )
    parser.add_argument(
        "--browser-wait-until",
        type=str,
        choices=["load", "domcontentloaded", "networkidle", "commit"],
        default="networkidle",
        help="Playwright wait_until state (default: networkidle).",
    )
    return parser.parse_args()


def collect_urls(args: argparse.Namespace) -> List[str]:
    urls: List[str] = []
    seen: Set[str] = set()

    if args.url:
        for u in args.url:
            if u and u.strip():
                cleaned = u.strip()
                if cleaned not in seen:
                    urls.append(cleaned)
                    seen.add(cleaned)

    if args.url_file and args.url_file.exists():
        for line in args.url_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped not in seen:
                urls.append(stripped)
                seen.add(stripped)

    return urls


def fetch_html(url: str) -> str:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return resp.text


def extract_title(soup: BeautifulSoup) -> str:
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    if h1 and h1.text:
        return h1.text.strip()
    return "Untitled"


def sanitize_filename(url: str, title: str) -> str:
    base = title or url
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", base).strip("-").lower()
    if not slug:
        slug = "page"
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{url_hash}.md"


def load_cache(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_cache(path: Path, cache: Dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


@contextmanager
def maybe_rate_limited(limiter: Optional[RateLimiter]):
    if limiter:
        limiter.wait()
    yield


def fetch_with_requests(
    url: str,
    cache_entry: Optional[dict],
    limiter: Optional[RateLimiter],
    force: bool = False,
) -> tuple[str, Optional[str], Optional[str]]:
    headers = {"User-Agent": USER_AGENT}
    if cache_entry and not force:
        if etag := cache_entry.get("etag"):
            headers["If-None-Match"] = etag
        if last_mod := cache_entry.get("last_modified"):
            headers["If-Modified-Since"] = last_mod

    with maybe_rate_limited(limiter):
        resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code == 304:
        raise NotModified()
    resp.raise_for_status()
    return resp.text, resp.headers.get("ETag"), resp.headers.get("Last-Modified")


def fetch_with_playwright(
    url: str,
    limiter: Optional[RateLimiter],
    timeout_ms: int,
    wait_until: str,
) -> tuple[str, None, None]:
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Playwright not installed. Install with: pip install playwright && playwright install chromium"
        ) from exc

    with maybe_rate_limited(limiter):
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(user_agent=USER_AGENT)
            page.goto(url, wait_until=wait_until, timeout=timeout_ms)
            html = page.content()
            browser.close()
    return html, None, None


def convert_to_markdown(html: str, url: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Drop script/style to keep noise out of Markdown.
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    title = extract_title(soup)
    body_html = str(soup.body or soup)
    markdown = html_to_md(body_html, heading_style="ATX").strip()

    metadata = (
        "---\n"
        f"title: {title}\n"
        f"source: {url}\n"
        f"fetched_at: {dt.datetime.utcnow().isoformat()}Z\n"
        "---\n\n"
    )
    return metadata + markdown + "\n"


def write_markdown(markdown: str, dest_dir: Path, filename: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / filename
    path.write_text(markdown, encoding="utf-8")
    return path


def process_url(
    url: str,
    dest_dir: Path,
    limiter: Optional[RateLimiter],
    cache: Dict[str, dict],
    cache_path: Path,
    robots_policy: Optional[RobotsPolicy],
    use_browser: bool,
    force: bool,
    browser_timeout_ms: int,
    browser_wait_until: str,
) -> Path:
    if robots_policy and not robots_policy.is_allowed(url):
        raise PermissionError(f"Blocked by robots.txt: {url}")

    cache_entry = cache.get(url)
    html: Optional[str] = None
    etag: Optional[str] = None
    last_modified: Optional[str] = None

    try:
        if use_browser:
            html, etag, last_modified = fetch_with_playwright(
                url,
                limiter=limiter,
                timeout_ms=browser_timeout_ms,
                wait_until=browser_wait_until,
            )
        else:
            html, etag, last_modified = fetch_with_requests(
                url,
                cache_entry=cache_entry,
                limiter=limiter,
                force=force,
            )
    except NotModified:
        if cache_entry and (existing := cache_entry.get("filename")):
            path = dest_dir / existing
            if path.exists():
                return path
        # If cache said not modified but file missing, force fetch.
        html, etag, last_modified = fetch_with_requests(
            url,
            cache_entry=None,
            limiter=limiter,
            force=True,
        )

    html_hash = hashlib.sha256(html.encode("utf-8")).hexdigest()
    title = extract_title(BeautifulSoup(html, "html.parser"))
    filename = sanitize_filename(url, title)

    # Skip rewrite if content hash unchanged and file exists.
    existing_path = dest_dir / filename
    if cache_entry and cache_entry.get("hash") == html_hash and existing_path.exists():
        return existing_path

    markdown = convert_to_markdown(html, url)
    path = write_markdown(markdown, dest_dir, filename)

    cache[url] = {
        "etag": etag,
        "last_modified": last_modified,
        "filename": filename,
        "title": title,
        "hash": html_hash,
        "fetched_at": dt.datetime.utcnow().isoformat() + "Z",
    }
    save_cache(cache_path, cache)
    return path


def main() -> int:
    args = parse_args()
    urls = collect_urls(args)
    if not urls:
        print("No URLs provided. Use --url or --url-file.", file=sys.stderr)
        return 1

    limiter = RateLimiter(args.min_delay) if args.min_delay > 0 else None
    robots_policy = None if args.skip_robots else RobotsPolicy(USER_AGENT)
    cache = load_cache(args.cache_file)

    handled_exceptions = (
        PermissionError,
        RuntimeError,
        requests.RequestException,
        OSError,
        ValueError,
    )

    for url in urls:
        try:
            dest = process_url(
                url=url,
                dest_dir=args.out_dir,
                limiter=limiter,
                cache=cache,
                cache_path=args.cache_file,
                robots_policy=robots_policy,
                use_browser=args.use_browser,
                force=args.force,
                browser_timeout_ms=args.browser_timeout,
                browser_wait_until=args.browser_wait_until,
            )
            print(f"Saved {url} -> {dest}")
        except handled_exceptions as exc:  # pragma: no cover - operational guard
            print(f"Failed to process {url}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
