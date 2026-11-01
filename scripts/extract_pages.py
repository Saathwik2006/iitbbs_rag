"""Step 1 of ingestion: fetch public pages and extract clean text.

Reads a list of public URLs from data/raw/seed_urls.txt (one per line,
'#' for comments), fetches each page, strips nav/script/style boilerplate
with BeautifulSoup, and writes one JSON object per line to
data/processed/documents.jsonl.

Usage:
    python scripts/extract_pages.py

This script only reads public, already-published web pages supplied by
you in seed_urls.txt. It does not authenticate, bypass any login, or
touch the ERP system -- see README.md "Known limitations" for why.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
SEED_FILE = RAW_DIR / "seed_urls.txt"
OUTPUT_FILE = PROCESSED_DIR / "documents.jsonl"

REQUEST_TIMEOUT = 15
REQUEST_DELAY_SECONDS = 1.0  # be polite to the source server


def read_seed_urls(path: Path) -> list[str]:
    if not path.exists():
        print(
            f"[error] {path} not found.\n"
            "Create it with one public URL per line, e.g.:\n"
            "  https://www.iitbbs.ac.in/academics.php\n"
            "  https://www.iitbbs.ac.in/notices.php\n"
            "Lines starting with '#' are treated as comments."
        )
        sys.exit(1)

    urls = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


def extract_text(html: str) -> tuple[str, str]:
    """Return (title, clean_text) from a raw HTML page."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    clean_text = "\n".join(lines)

    return title, clean_text


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    urls = read_seed_urls(SEED_FILE)

    if not urls:
        print(f"[error] {SEED_FILE} has no URLs. Nothing to extract.")
        sys.exit(1)

    documents = []
    for url in urls:
        print(f"Fetching {url} ...")
        try:
            resp = requests.get(
                url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "iitbbs-rag-assistant/1.0"}
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            print(f"  [skip] failed to fetch: {exc}")
            continue

        title, text = extract_text(resp.text)
        if not text:
            print("  [skip] no extractable text")
            continue

        documents.append(
            {
                "doc_id": str(uuid.uuid4()),
                "url": url,
                "title": title or url,
                "text": text,
                "fetched_at": datetime.now(UTC).isoformat(),
            }
        )
        time.sleep(REQUEST_DELAY_SECONDS)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(documents)} documents to {OUTPUT_FILE}")
    if len(documents) < len(urls):
        print(f"({len(urls) - len(documents)} URL(s) were skipped -- see log above)")


if __name__ == "__main__":
    main()
