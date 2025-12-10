import agent


def test_sanitize_filename_generates_hash_suffix():
    result = agent.sanitize_filename("https://example.com/docs", "Example Title")
    assert result.startswith("example-title-")
    assert result.endswith(".md")


def test_url_collection_merges_sources(tmp_path):
    url_file = tmp_path / "urls.txt"
    url_file.write_text("https://example.com\n# comment\nhttps://example.com/docs\n")
    args = type("Args", (), {"url": ["https://example.com"], "url_file": url_file})
    urls = agent.collect_urls(args)
    assert urls == ["https://example.com", "https://example.com/docs"]


def test_sanitize_filename_allows_other_extensions():
    result = agent.sanitize_filename("https://example.com/doc.pdf", "Doc Title", extension=".pdf")
    assert result.endswith(".pdf")
    assert result.startswith("doc-title-")
