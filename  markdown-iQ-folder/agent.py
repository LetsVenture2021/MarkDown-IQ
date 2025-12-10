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
import mimetypes
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
    MISSING_DEPS_MSG = (
        "Missing dependencies. Install with: pip install -r requirements.txt\n"
        f"Original error: {exc}"
    )
    print(MISSING_DEPS_MSG, file=sys.stderr)
    sys.exit(1)

USER_AGENT = "DocHarvester/0.1 (+https://example.com; for personal documentation use)"
DEFAULT_MIN_DELAY = 1.0
DEFAULT_MAX_RETRIES = 2
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
            except (OSError, IOError, requests.RequestException):
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
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Retry count for transient errors (default: {DEFAULT_MAX_RETRIES}).",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=1.5,
        help="Seconds for first retry backoff; doubles each attempt (default: 1.5).",
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help="Path to cache file for ETag/Last-Modified tracking.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cache read/write entirely.",
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
    parser.add_argument(
        "--crawl",
        action="store_true",
        help="Discover same-domain links from seed URLs before fetching.",
    )
    parser.add_argument(
        "--max-crawl-pages",
        type=int,
        default=20,
        help="Maximum discovered URLs to add when crawling (default: 20).",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=[],
        help="Regex allowlist for discovered URLs (repeatable).",
    )
    parser.add_argument(
        "--deny-pattern",
        action="append",
        default=[],
        help="Regex denylist for discovered URLs (repeatable).",
    )
    parser.add_argument(
        "--strip-selectors",
        action="append",
        default=[],
        help="CSS selectors to remove before Markdown conversion (repeatable).",
    )
    parser.add_argument(
        "--keep-selectors",
        action="append",
        default=[],
        help="If set, only keep elements matching these selectors (repeatable).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Write run report to JSON or CSV at this path.",
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


def normalize_links(soup: BeautifulSoup, base_url: str) -> None:
    parsed_base = urlparse(base_url)
    if not parsed_base.scheme or not parsed_base.netloc:
        return
    for tag in soup.find_all("a", href=True):
        href = tag["href"]
        tag["href"] = requests.compat.urljoin(base_url, href)


def prune_soup(
    soup: BeautifulSoup,
    strip_selectors: Optional[List[str]] = None,
    keep_selectors: Optional[List[str]] = None,
) -> BeautifulSoup:
    strip_selectors = strip_selectors or []
    keep_selectors = keep_selectors or []
    body = soup.body or soup

    for selector in strip_selectors:
        for tag in body.select(selector):
            tag.decompose()

    if keep_selectors:
        kept_nodes = []
        for selector in keep_selectors:
            kept_nodes.extend(body.select(selector))
        new_body = BeautifulSoup("<body></body>", "html.parser").body
        for node in kept_nodes:
            new_body.append(node.extract())
        if soup.body:
            soup.body.replace_with(new_body)
        else:
            soup = BeautifulSoup("<html></html>", "html.parser")
            soup.append(new_body)
    return soup


def extract_title(soup: BeautifulSoup) -> str:
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    if h1 and h1.text:
        return h1.text.strip()
    return "Untitled"


def sanitize_filename(url: str, title: str, extension: str = ".md") -> str:
    base = title or url
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", base).strip("-").lower()
    if not slug:
        slug = "page"
    suffix = extension or ""
    if suffix and not suffix.startswith("."):
        suffix = "." + suffix
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{url_hash}{suffix}"


def derive_title_from_url(url: str) -> str:
    parsed = urlparse(url)
    path = Path(parsed.path)
    if path.stem:
        return path.stem
    if parsed.netloc:
        return parsed.netloc
    return "Document"


def guess_binary_extension(content_type: str, url: str, default: str = ".bin") -> str:
    ct = (content_type or "").split(";", 1)[0].strip().lower()
    if "pdf" in ct:
        return ".pdf"
    if ct:
        guessed = mimetypes.guess_extension(ct)
        if guessed:
            return guessed
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix
    if suffix:
        return suffix
    return default


def is_pdf_response(content_type: str, raw_bytes: bytes, url: str) -> bool:
    ct = (content_type or "").lower()
    if "pdf" in ct:
        return True
    if url.lower().split("?", 1)[0].endswith(".pdf"):
        return True
    return raw_bytes.startswith(b"%PDF") if raw_bytes else False


def load_cache(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # If cache is corrupted JSON, move it aside so users can inspect it.
        try:
            corrupt_name = path.with_name(path.name + ".corrupt-" + dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"))
            path.replace(corrupt_name)
            print(f"Warning: cache file corrupted; moved to {corrupt_name}", file=sys.stderr)
        except OSError:
            # If we cannot rename, silently continue and return empty cache.
            pass
        return {}
    except OSError:
        # IO problems (permission, etc.) — return empty cache but surface nothing
        # to keep behavior consistent with prior implementation.
        return {}


def save_cache(path: Path, cache: Dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def with_retries(
    func, max_retries: int, backoff: float, retry_exceptions: tuple[type[Exception], ...]
):
    attempt = 0
    while True:
        try:
            return func()
        except retry_exceptions:
            attempt += 1
            if attempt > max_retries:
                raise
            sleep_for = backoff * (2 ** (attempt - 1))
            time.sleep(sleep_for)


def compile_patterns(patterns: List[str]) -> List[re.Pattern[str]]:
    compiled = []
    for pat in patterns:
        try:
            compiled.append(re.compile(pat))
        except re.error:
            continue
    return compiled


def url_matches_patterns(
    url: str, allow: List[re.Pattern[str]], deny: List[re.Pattern[str]]
) -> bool:
    if deny and any(p.search(url) for p in deny):
        return False
    if allow:
        return any(p.search(url) for p in allow)
    return True


def discover_urls(
    seed_urls: List[str],
    max_pages: int,
    allow_patterns: List[re.Pattern[str]],
    deny_patterns: List[re.Pattern[str]],
    limiter: Optional[RateLimiter],
    robots_policy: Optional[RobotsPolicy],
) -> List[str]:
    queue = list(seed_urls)
    seen: Set[str] = set(seed_urls)
    discovered: List[str] = []

    while queue and len(discovered) < max_pages:
        current = queue.pop(0)
        parsed_current = urlparse(current)
        base_domain = parsed_current.netloc
        try:
            with maybe_rate_limited(limiter):
                html = fetch_html(current)
        except (requests.RequestException, OSError):
            continue

        soup = BeautifulSoup(html, "html.parser")
        for link in soup.find_all("a", href=True):
            target = requests.compat.urljoin(current, link["href"])
            parsed_target = urlparse(target)
            if parsed_target.netloc != base_domain:
                continue
            if target in seen:
                continue
            if robots_policy and not robots_policy.is_allowed(target):
                continue
            if not url_matches_patterns(target, allow_patterns, deny_patterns):
                continue

            discovered.append(target)
            seen.add(target)
            queue.append(target)
            if len(discovered) >= max_pages:
                break

    return discovered


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
) -> tuple[str, bytes, str, Optional[str], Optional[str]]:
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
    return (
        resp.text,
        resp.content,
        resp.headers.get("Content-Type", ""),
        resp.headers.get("ETag"),
        resp.headers.get("Last-Modified"),
    )


def fetch_with_playwright(
    url: str,
    limiter: Optional[RateLimiter],
    timeout_ms: int,
    wait_until: str,
) -> tuple[str, bytes, str, None, None]:
    try:
        from playwright.sync_api import sync_playwright  # type: ignore  # pylint: disable=import-outside-toplevel
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
    return html, html.encode("utf-8"), "text/html", None, None


def convert_to_markdown(
    html: str,
    url: str,
    strip_selectors: Optional[List[str]] = None,
    keep_selectors: Optional[List[str]] = None,
) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Drop script/style to keep noise out of Markdown.
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    normalize_links(soup, url)
    soup = prune_soup(soup, strip_selectors=strip_selectors, keep_selectors=keep_selectors)
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


def build_binary_stub_markdown(
    title: str,
    url: str,
    binary_filename: str,
    content_type: str,
) -> str:
    metadata = (
        "---\n"
        f"title: {title}\n"
        f"source: {url}\n"
        f"fetched_at: {dt.datetime.utcnow().isoformat()}Z\n"
        f"content_type: {content_type or 'application/octet-stream'}\n"
        f"local_file: {binary_filename}\n"
        "---\n\n"
    )
    body = (
        "This entry references a binary download saved alongside this file. "
        f"Open `{binary_filename}` with a compatible viewer to inspect the original document."
    )
    return metadata + body + "\n"


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
    use_cache: bool,
    max_retries: int,
    retry_backoff: float,
    strip_selectors: Optional[List[str]],
    keep_selectors: Optional[List[str]],
) -> tuple[Path, str, str, Optional[Path]]:
    if robots_policy and not robots_policy.is_allowed(url):
        raise PermissionError(f"Blocked by robots.txt: {url}")

    cache_entry = cache.get(url) if use_cache else None
    html: Optional[str] = None
    raw_bytes: bytes = b""
    content_type: str = ""
    etag: Optional[str] = None
    last_modified: Optional[str] = None

    retryable = (requests.RequestException, RuntimeError, OSError, ValueError)

    def perform_fetch():
        if use_browser:
            return fetch_with_playwright(
                url,
                limiter=limiter,
                timeout_ms=browser_timeout_ms,
                wait_until=browser_wait_until,
            )
        return fetch_with_requests(
            url,
            cache_entry=cache_entry,
            limiter=limiter,
            force=force,
        )

    try:
        html, raw_bytes, content_type, etag, last_modified = with_retries(
            perform_fetch, max_retries=max_retries, backoff=retry_backoff, retry_exceptions=retryable
        )
    except NotModified:
        if cache_entry and (existing := cache_entry.get("filename")):
            path = dest_dir / existing
            binary_candidate = None
            if binary_name := cache_entry.get("binary_filename"):
                candidate_path = dest_dir / binary_name
                if candidate_path.exists():
                    binary_candidate = candidate_path
                else:
                    path = None  # Force refetch if binary missing
            if path and path.exists():
                return path, cache_entry.get("hash", ""), cache_entry.get("title", ""), binary_candidate
        # If cache said not modified but file missing, force fetch.
        html, raw_bytes, content_type, etag, last_modified = with_retries(
            lambda: fetch_with_requests(
                url,
                cache_entry=None,
                limiter=limiter,
                force=True,
            ),
            max_retries=max_retries,
            backoff=retry_backoff,
            retry_exceptions=retryable,
        )

    html_text = html or ""
    content_bytes = raw_bytes or html_text.encode("utf-8")
    content_hash = hashlib.sha256(content_bytes).hexdigest()

    if cache_entry and cache_entry.get("hash") == content_hash:
        cached_filename = cache_entry.get("filename")
        if cached_filename:
            cached_path = dest_dir / cached_filename
            binary_path: Optional[Path] = None
            binary_name = cache_entry.get("binary_filename")
            if binary_name:
                candidate = dest_dir / binary_name
                if candidate.exists():
                    binary_path = candidate
                else:
                    cached_path = None
            if cached_path and cached_path.exists():
                return cached_path, content_hash, cache_entry.get("title", derive_title_from_url(url)), binary_path

    if is_pdf_response(content_type, raw_bytes, url):
        title = cache_entry.get("title") if cache_entry else None
        if not title:
            title = derive_title_from_url(url)
        base_name = sanitize_filename(url, title, extension="")
        markdown_filename = f"{base_name}.md"
        binary_extension = guess_binary_extension(content_type, url, default=".pdf")
        binary_filename = f"{base_name}{binary_extension}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        binary_path = dest_dir / binary_filename
        binary_payload = raw_bytes or html_text.encode("utf-8")
        binary_path.write_bytes(binary_payload)
        markdown = build_binary_stub_markdown(title, url, binary_filename, content_type or "application/pdf")
        markdown_path = write_markdown(markdown, dest_dir, markdown_filename)
        if use_cache:
            cache[url] = {
                "etag": etag,
                "last_modified": last_modified,
                "filename": markdown_filename,
                "binary_filename": binary_filename,
                "title": title,
                "hash": content_hash,
                "fetched_at": dt.datetime.utcnow().isoformat() + "Z",
            }
            save_cache(cache_path, cache)
        return markdown_path, content_hash, title, binary_path

    soup = BeautifulSoup(html_text, "html.parser")
    title = extract_title(soup)
    filename = sanitize_filename(url, title)

    markdown = convert_to_markdown(
        html_text,
        url,
        strip_selectors=strip_selectors,
        keep_selectors=keep_selectors,
    )
    path = write_markdown(markdown, dest_dir, filename)

    if use_cache:
        cache[url] = {
            "etag": etag,
            "last_modified": last_modified,
            "filename": filename,
            "binary_filename": None,
            "title": title,
            "hash": content_hash,
            "fetched_at": dt.datetime.utcnow().isoformat() + "Z",
        }
        save_cache(cache_path, cache)
    return path, content_hash, title, None


def write_report(report_path: Path, results: List[dict]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = report_path.suffix.lower()
    if suffix == ".csv":
        import csv  # pylint: disable=import-outside-toplevel

        fieldnames = sorted({k for r in results for k in r.keys()})
        with report_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
    else:
        report_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")


def shorten_error_message(exc: Exception, limit: int = 600) -> str:
    text = str(exc).strip()
    if not text:
        text = exc.__class__.__name__
    if len(text) > limit:
        return text[:limit] + "... (truncated)"
    return text


def main() -> int:
    args = parse_args()
    urls = collect_urls(args)
    if not urls:
        print("No URLs provided. Use --url or --url-file.", file=sys.stderr)
        return 1

    if args.use_browser:
        # Check Playwright availability without importing unused symbols
        import importlib.util  # pylint: disable=import-outside-toplevel

        if importlib.util.find_spec("playwright.sync_api") is None:
            print(
                "Playwright not installed. Install with: pip install playwright && playwright install chromium",
                file=sys.stderr,
            )
            return 1

    limiter = RateLimiter(args.min_delay) if args.min_delay > 0 else None
    robots_policy = None if args.skip_robots else RobotsPolicy(USER_AGENT)
    allow_patterns = compile_patterns(args.allow_pattern)
    deny_patterns = compile_patterns(args.deny_pattern)

    if args.crawl and urls:
        discovered = discover_urls(
            seed_urls=urls,
            max_pages=args.max_crawl_pages,
            allow_patterns=allow_patterns,
            deny_patterns=deny_patterns,
            limiter=limiter,
            robots_policy=robots_policy,
        )
        for new_url in discovered:
            if new_url not in urls:
                urls.append(new_url)

    cache = {} if args.no_cache else load_cache(args.cache_file)

    handled_exceptions = (
        PermissionError,
        RuntimeError,
        requests.RequestException,
        OSError,
        ValueError,
    )

    results: List[dict] = []
    failures = 0

    for url in urls:
        try:
            dest, html_hash, title, binary_path = process_url(
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
                use_cache=not args.no_cache,
                max_retries=max(0, args.max_retries),
                retry_backoff=max(0.0, args.retry_backoff),
                strip_selectors=args.strip_selectors,
                keep_selectors=args.keep_selectors,
            )
            print(f"Saved {url} -> {dest}")
            result_row = {
                "url": url,
                "status": "ok",
                "path": str(dest),
                "hash": html_hash,
                "title": title,
            }
            if binary_path:
                result_row["binary_path"] = str(binary_path)
            results.append(result_row)
        except handled_exceptions as exc:  # pragma: no cover - operational guard
            failures += 1
            message = shorten_error_message(exc)
            print(f"Failed to process {url}: {message}", file=sys.stderr)
            results.append({"url": url, "status": "error", "error": message})

    print(f"Completed: {len(urls) - failures} ok, {failures} failed.")
    if args.report:
        try:
            write_report(args.report, results)
            print(f"Wrote report to {args.report}")
        except (OSError, TypeError) as exc:  # pragma: no cover - guard
            print(f"Failed to write report: {exc}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
