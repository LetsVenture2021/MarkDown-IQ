<!-- Copilot / AI agent instructions for MarkDown-IQ -->
# Copilot Instructions — Documentation Harvester Agent

Purpose: Help engineers and AI-assisted agents make safe, focused edits to
this repository. These notes are intentionally concise and actionable.

**Big Picture**
- **Project:** single-script CLI that fetches public documentation pages,
  converts them to Markdown, and writes files to `docs/` with YAML frontmatter.
- **Core module:** `agent.py` — contains CLI parsing, fetch logic, optional
  Playwright browser fetching, crawl/discovery, sanitization and caching.
- **Data flow:** input URLs (`--url` or `--url-file` / `urls.txt`) -> fetch
  (requests or Playwright) -> HTML prune/normalize -> `markdownify` -> write
  `.md` with metadata + update cache (`.cache/harvest_cache.json`).

**Useful files & commands**
- **Entry point:** `agent.py` (main CLI) — run locally from repo root:
  - `python agent.py --url-file urls.txt` or single URL with `--url`.
- **Dependencies:** `requirements.txt` — use a venv and install:
  - `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- **Playwright (optional):** after installing requirements run:
  - `playwright install chromium` (only if `--use-browser` is needed).
- **Tests:** run `pytest` from repo root (tests exercise core pure functions
  like `sanitize_filename`, `convert_to_markdown`, `RobotsPolicy`, cache I/O).

**Project-specific behaviors to preserve**
- **Caching semantics:** cache keys are full URLs; entries store `etag`,
  `last_modified`, `filename`, `hash`, and `fetched_at`. The agent relies on
  304 handling + content hash to avoid rewriting existing files. Don't change
  cache layout without updating `load_cache`/`save_cache` and tests.
- **Filename stability:** `sanitize_filename(url, title)` must be stable across
  runs (tests assert this). Avoid changing slugging or the 8-char hash
  suffixing scheme without updating tests.
- **Robots enforcement:** `RobotsPolicy` uses stdlib `robotparser` and caches
  parsers per domain. Tests monkeypatch `robotparser.RobotFileParser` — keep
  the caching behavior to avoid repeated network calls in tests.
- **Optional Playwright import:** Playwright is optional; code gracefully
  raises helpful errors when not present. For edits that touch browser logic,
  maintain the runtime-optional import pattern.

**Patterns & conventions**
- Prefer small, testable pure functions (e.g., `convert_to_markdown`,
  `sanitize_filename`, `compile_patterns`) rather than large side-effecting
  changes. Tests exercise these functions directly.
- CLI flags map 1:1 to `parse_args()` and are used throughout the codebase;
  add flags there and propagate explicitly into `process_url` rather than
  reading global state.
- Network operations are wrapped with `with_retries` and `RateLimiter`; if you
  change retry/backoff semantics update callers and tests accordingly.

**When editing code**
- Keep changes minimal and backwards-compatible for the CLI surface.
- Run `pytest` locally after changes. Tests are small and should pass with
  no network access (they monkeypatch network calls).
- If you modify caching, filename formats, or metadata frontmatter, update
  `tests/test_agent.py` to reflect the new behavior.

**Examples for common tasks**
- Harvest static pages (no browser):
  `python agent.py --url-file urls.txt --out-dir docs --min-delay 1.0`
- Harvest with Playwright for dynamic pages:
  `python agent.py --url-file urls.txt --use-browser --browser-wait-until networkidle`
- Run tests:
  `source .venv/bin/activate && pytest -q`

**Safety & etiquette**
- This tool is intended to fetch only publicly available pages. Respect
  `robots.txt` (default) and site Terms of Service. Use `--skip-robots` only
  when you are certain you have permission to bypass it.

If anything here is unclear or you want more examples (e.g., common PR
edits workflow, or explanation of `process_url` internals), tell me which
section to expand and I'll iterate.
