# MarkDown-iQ Documentation Harvester

Small CLI utility that downloads documentation pages, converts them to Markdown, and stores them in `docs/` for downstream use.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python agent.py --url https://example.com --out-dir docs
```

## Options

Run `python agent.py --help` to see all supported flags including caching, crawling, Playwright rendering, and reporting.
