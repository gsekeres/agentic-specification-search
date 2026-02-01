#!/usr/bin/env python3
"""
01_collect_dois.py

Collect ALL openICPSR package DOIs (prefix 10.3886) from DataCite using cursor pagination.
No filtering at this stage - we fetch the full universe (~72,000 DOIs) and filter for AEA in Script 2.

Outputs a clean JSONL file with one record per package/version.

Usage:
    python scripts/01_collect_dois.py
"""

import json
import os
import re
import sqlite3
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterator
from urllib.parse import urlencode

import requests


# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
METADATA_DIR = DATA_DIR / "metadata"
CACHE_DIR = DATA_DIR / "cache"

OUTPUT_FILE = METADATA_DIR / "icpsr_openicpsr_packages.jsonl"
CACHE_DB = CACHE_DIR / "datacite_responses.db"

DATACITE_API = "https://api.datacite.org/dois"
USER_AGENT = "AEA-Spec-Search/1.0 (mailto:gs754@cornell.edu)"
EMAIL = "gs754@cornell.edu"

# Rate limiting: 800 requests per 5 minutes (safe margin under 1000)
RATE_LIMIT_TOKENS = 800
RATE_LIMIT_PERIOD = 300  # seconds


class TokenBucketLimiter:
    """Simple token bucket rate limiter."""

    def __init__(self, tokens_per_period: int, period_seconds: int):
        self.tokens = tokens_per_period
        self.max_tokens = tokens_per_period
        self.period = period_seconds
        self.last_refill = time.time()

    def wait(self):
        """Wait if necessary to respect rate limit."""
        now = time.time()
        elapsed = now - self.last_refill

        # Refill tokens if period has passed
        if elapsed >= self.period:
            self.tokens = self.max_tokens
            self.last_refill = now

        # Wait if no tokens available
        if self.tokens <= 0:
            sleep_time = self.period - elapsed
            if sleep_time > 0:
                print(f"Rate limit: waiting {sleep_time:.1f}s for token refill...")
                time.sleep(sleep_time)
            self.tokens = self.max_tokens
            self.last_refill = time.time()

        self.tokens -= 1


class SQLiteCache:
    """SQLite-based cache for API responses."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS responses (
                    cursor TEXT PRIMARY KEY,
                    response_json TEXT,
                    fetched_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS packages (
                    package_doi TEXT PRIMARY KEY,
                    package_json TEXT,
                    added_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS state (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            conn.commit()

    def get_last_cursor(self) -> Optional[str]:
        """Get the last processed cursor for resumability."""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                "SELECT value FROM state WHERE key = 'last_cursor'"
            ).fetchone()
            return result[0] if result else None

    def save_cursor(self, cursor: str):
        """Save the last processed cursor."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO state (key, value) VALUES ('last_cursor', ?)",
                (cursor,)
            )
            conn.commit()

    def get_response(self, cursor: str) -> Optional[dict]:
        """Get cached response for a cursor."""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                "SELECT response_json FROM responses WHERE cursor = ?",
                (cursor,)
            ).fetchone()
            return json.loads(result[0]) if result else None

    def save_response(self, cursor: str, response: dict):
        """Cache a response."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO responses (cursor, response_json, fetched_at) VALUES (?, ?, ?)",
                (cursor, json.dumps(response), datetime.utcnow().isoformat())
            )
            conn.commit()

    def get_all_packages(self) -> dict:
        """Get all cached packages."""
        with sqlite3.connect(self.db_path) as conn:
            results = conn.execute("SELECT package_doi, package_json FROM packages").fetchall()
            return {row[0]: json.loads(row[1]) for row in results}

    def save_package(self, package: dict):
        """Cache a package."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO packages (package_doi, package_json, added_at) VALUES (?, ?, ?)",
                (package['package_doi'], json.dumps(package), datetime.utcnow().isoformat())
            )
            conn.commit()

    def clear_state(self):
        """Clear cursor state to force full re-fetch."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM state WHERE key = 'last_cursor'")
            conn.commit()


def is_file_level_doi(doi: str) -> bool:
    """
    Check if DOI is a file-level DOI (should be excluded).
    File-level DOIs have extra digits after the version: E123456V1-12345
    """
    # Match pattern: E followed by digits, V followed by digits, then dash and more digits
    pattern = r'E\d+V\d+-\d+'
    return bool(re.search(pattern, doi.upper()))


def parse_package(item: dict) -> Optional[dict]:
    """
    Parse a DataCite item into our package schema.
    Returns None if the DOI should be excluded (e.g., file-level DOI).
    """
    attrs = item.get('attributes', {})
    doi = attrs.get('doi', '').upper()

    if not doi:
        return None

    # Skip file-level DOIs
    if is_file_level_doi(doi):
        return None

    # Extract base_id and version from DOI
    # DOI format: 10.3886/E123456V1 or 10.3886/E123456
    base_match = re.search(r'E(\d+)', doi)
    version_match = re.search(r'V(\d+)', doi)

    if not base_match:
        return None

    base_id = f"E{base_match.group(1)}"
    version = f"V{version_match.group(1)}" if version_match else "V1"

    # Get title
    titles = attrs.get('titles', [])
    title = titles[0].get('title', '') if titles else ''

    # Get dates
    dates = attrs.get('dates', [])
    created = None
    updated = None
    issued = None
    for date_entry in dates:
        date_type = date_entry.get('dateType', '')
        date_value = date_entry.get('date', '')
        if date_type == 'Created':
            created = date_value
        elif date_type == 'Updated':
            updated = date_value
        elif date_type == 'Issued':
            issued = date_value

    # Get related identifiers
    related = attrs.get('relatedIdentifiers', [])
    related_identifiers = []
    for rel in related:
        related_identifiers.append({
            'relatedIdentifier': rel.get('relatedIdentifier', ''),
            'relatedIdentifierType': rel.get('relatedIdentifierType', ''),
            'relationType': rel.get('relationType', '')
        })

    # Construct landing page URL
    # Extract numeric ID from base_id
    numeric_id = base_match.group(1)
    version_num = version_match.group(1) if version_match else "1"
    landing_page = f"https://www.openicpsr.org/openicpsr/project/{numeric_id}/version/V{version_num}/view"

    return {
        'package_doi': doi,
        'base_id': base_id,
        'version': version,
        'title': title,
        'publisher': attrs.get('publisher', ''),
        'created': created,
        'updated': updated,
        'issued': issued,
        'landing_page_url': landing_page,
        'related_identifiers': related_identifiers
    }


def fetch_with_retry(url: str, headers: dict, max_retries: int = 5) -> dict:
    """Fetch URL with retry logic and 429 handling."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                jitter = random.uniform(0, 10)
                sleep_time = retry_after + jitter + (2 ** attempt)
                print(f"Rate limited (429). Waiting {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                continue

            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            sleep_time = 10 * (attempt + 1)
            print(f"Timeout. Retrying in {sleep_time}s...")
            time.sleep(sleep_time)
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            sleep_time = 10 * (attempt + 1)
            print(f"Request error: {e}. Retrying in {sleep_time}s...")
            time.sleep(sleep_time)

    raise Exception(f"Failed after {max_retries} retries")


def build_query_url(cursor: str = '*') -> str:
    """Build the DataCite query URL for ALL 10.3886 DOIs (no publisher filter)."""
    params = {
        'query': 'prefix:10.3886',
        'fields[dois]': 'doi,titles,publisher,relatedIdentifiers,dates,types',
        'page[size]': '1000',
        'page[cursor]': cursor,
        'mailto': EMAIL
    }
    return f"{DATACITE_API}?{urlencode(params)}"


def collect_dois():
    """Main function to collect ALL openICPSR package DOIs (prefix 10.3886)."""
    print("=" * 60)
    print("openICPSR DOI Collection (Full Universe)")
    print("=" * 60)

    # Ensure directories exist
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize cache and rate limiter
    cache = SQLiteCache(CACHE_DB)
    limiter = TokenBucketLimiter(RATE_LIMIT_TOKENS, RATE_LIMIT_PERIOD)

    headers = {
        'Accept': 'application/vnd.api+json',
        'User-Agent': USER_AGENT
    }

    # Get starting cursor (resume if available)
    cursor = cache.get_last_cursor() or '*'
    all_packages = cache.get_all_packages()

    if cursor != '*':
        print(f"Resuming from cursor: {cursor[:50]}...")
        print(f"Already have {len(all_packages)} packages cached")
    else:
        print("Starting fresh collection")

    api_calls = 0
    cache_hits = 0
    start_time = time.time()

    while cursor:
        # Check if we have this response cached
        cached_response = cache.get_response(cursor)

        if cached_response:
            response = cached_response
            cache_hits += 1
        else:
            # Respect rate limit
            limiter.wait()

            url = build_query_url(cursor)
            print(f"Fetching page (cursor: {cursor[:30] if cursor != '*' else '*'}...)")

            response = fetch_with_retry(url, headers)
            cache.save_response(cursor, response)
            api_calls += 1

        # Process items
        items = response.get('data', [])
        new_in_page = 0

        for item in items:
            pkg = parse_package(item)
            if pkg:
                # Use package_doi as key to handle all versions
                key = pkg['package_doi']
                if key not in all_packages:
                    all_packages[key] = pkg
                    cache.save_package(pkg)
                    new_in_page += 1

        print(f"  -> Found {len(items)} items, {new_in_page} new packages")

        # Get next cursor
        links = response.get('links', {})
        next_url = links.get('next')

        if next_url:
            # Extract cursor from next URL
            # The next URL contains the full URL with cursor parameter
            if 'page%5Bcursor%5D=' in next_url or 'page[cursor]=' in next_url:
                import urllib.parse
                parsed = urllib.parse.urlparse(next_url)
                query_params = urllib.parse.parse_qs(parsed.query)
                cursor = query_params.get('page[cursor]', [None])[0]
            else:
                cursor = None

            if cursor:
                cache.save_cursor(cursor)
        else:
            cursor = None

    # Write output file
    print(f"\nWriting {len(all_packages)} packages to {OUTPUT_FILE}")

    # Sort by package_doi for consistent output
    sorted_packages = sorted(all_packages.values(), key=lambda x: x['package_doi'])

    with open(OUTPUT_FILE, 'w') as f:
        for pkg in sorted_packages:
            f.write(json.dumps(pkg) + '\n')

    # Print summary
    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("=== DOI Collection Summary ===")
    print("=" * 60)
    print(f"Total packages: {len(all_packages)}")

    # Count by publisher (should all be AEA)
    publishers = {}
    for pkg in all_packages.values():
        pub = pkg.get('publisher', 'Unknown')
        publishers[pub] = publishers.get(pub, 0) + 1

    print("\nBy publisher:")
    for pub, count in sorted(publishers.items(), key=lambda x: -x[1]):
        print(f"  {pub}: {count}")

    print(f"\nAPI calls made: {api_calls}")
    print(f"Cache hits: {cache_hits}")
    print(f"Elapsed time: {elapsed:.1f}s")
    print(f"\nOutput written to: {OUTPUT_FILE}")

    # Clear cursor state after successful completion
    cache.clear_state()


if __name__ == '__main__':
    collect_dois()
