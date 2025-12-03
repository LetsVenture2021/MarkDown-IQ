# Documentation Harvester Agent

Agent for fetching public documentation pages (e.g., OpenAI docs), converting them to Markdown, and storing them in the project for downstream AI use.

## Quickstart
- Ensure you have Python 3.10+.
- Create a virtual environment and install dependencies: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
- Add target URLs to `urls.txt` (one per line, `#` for comments).
- Run the harvester: `python agent.py --url-file urls.txt`.
- Markdown files land in `docs/` with YAML frontmatter (title, source, fetched_at).

## Usage
```
python agent.py --url https://platform.openai.com/docs/overview
python agent.py --url-file urls.txt --out-dir docs --min-delay 1.5
python agent.py --url-file urls.txt --use-browser   # Playwright for dynamic pages
python agent.py --url-file urls.txt --crawl --max-crawl-pages 10 --allow-pattern docs
python agent.py --url-file urls.txt --strip-selectors nav --strip-selectors footer
python agent.py --url-file urls.txt --force         # Ignore cache
python agent.py --url-file urls.txt --report run.json --max-retries 3 --retry-backoff 2.0
```

Options:
- `--url`: fetch a single URL (repeatable).
- `--url-file`: path to a file containing URLs (one per line, `#` = comment).
- `--out-dir`: output directory for generated `.md` files (default: `docs`).
- `--min-delay`: throttle between requests (seconds, default 1.0).
- `--max-retries` / `--retry-backoff`: transient error retries with exponential backoff.
- `--cache-file`: where to store fetch metadata (default: `.cache/harvest_cache.json`).
- `--no-cache`: bypass cache reads/writes.
- `--force`: ignore cache and re-fetch.
- `--skip-robots`: bypass robots.txt enforcement (only when permitted by site terms).
- `--use-browser`: fetch with Playwright headless Chromium for dynamic pages.
- `--browser-timeout`: Playwright navigation timeout (ms).
- `--browser-wait-until`: Playwright wait condition (`load`, `domcontentloaded`, `networkidle`, `commit`).
- `--crawl`: discover same-domain links from seed URLs before fetching.
- `--allow-pattern` / `--deny-pattern`: regex filters for crawled links.
- `--strip-selectors` / `--keep-selectors`: remove or only keep matching CSS selectors before Markdown conversion.
- `--report`: write a JSON or CSV report of run results.

## Responsibilities and limits
- Only fetch pages you are permitted to access; respect site Terms of Service and `robots.txt`.
- This agent is for static, publicly available pages; dynamic or authenticated content will need a headless browser (not included here).
- Network access may be restricted in some environments; run locally where outbound HTTP is allowed.
- Cache is based on ETag/Last-Modified and content hashes; 304 responses will skip rewrites when the file already exists.

## Project layout
- `agent.py` – CLI to fetch HTML, convert to Markdown, and save with metadata.
- `requirements.txt` – Python dependencies.
- `urls.txt` – Sample URL list.
- `docs/` – Output folder for harvested Markdown.

## Playwright setup (optional, for dynamic pages)
```
pip install -r requirements.txt
playwright install chromium
```
