#!/usr/bin/env python3
"""
02_identify_journals.py

Map each AEA package DOI to its linked article DOI(s) and infer journal.
Uses DataCite relatedIdentifiers from step 1 and AEA migration table as fallback.

Usage:
    python scripts/02_identify_journals.py
"""

import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from io import StringIO

import requests


# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
METADATA_DIR = DATA_DIR / "metadata"
CACHE_DIR = DATA_DIR / "cache"

INPUT_FILE = METADATA_DIR / "icpsr_openicpsr_packages.jsonl"
OUTPUT_FILE = METADATA_DIR / "aea_package_to_journal.jsonl"
AEA_TABLE_CACHE = CACHE_DIR / "aea_migration_table.csv"

AEA_MIGRATION_TABLE_URL = (
    "https://raw.githubusercontent.com/AEADataEditor/aea-supplement-migration/"
    "master/data/generated/table.aea.icpsr.mapping.csv"
)

# DOI stem to journal mapping
DOI_TO_JOURNAL = {
    '10.1257/aer.': 'AER',
    '10.1257/pandp.': 'AER: P&P',
    '10.1257/aeri.': 'AER: Insights',
    '10.1257/pol.': 'AEJ: Policy',
    '10.1257/mic.': 'AEJ: Micro',
    '10.1257/mac.': 'AEJ: Macro',
    '10.1257/app.': 'AEJ: Applied',
    '10.1257/jep.': 'JEP',
    '10.1257/jel.': 'JEL',
}


def load_jsonl(filepath: Path) -> list:
    """Load a JSONL file into a list of dicts."""
    if not filepath.exists():
        return []
    with open(filepath, 'r') as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(filepath: Path, data: list):
    """Save a list of dicts to a JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def load_existing_mappings(filepath: Path) -> dict:
    """Load existing mappings keyed by package_doi."""
    mappings = load_jsonl(filepath)
    return {m['package_doi']: m for m in mappings}


def download_aea_migration_table() -> str:
    """Download the AEA migration table, using cache if available."""
    # Check cache first
    if AEA_TABLE_CACHE.exists():
        # Use cache if less than 7 days old
        cache_age = datetime.now().timestamp() - AEA_TABLE_CACHE.stat().st_mtime
        if cache_age < 7 * 24 * 60 * 60:  # 7 days
            print(f"Using cached AEA migration table ({cache_age/3600:.1f}h old)")
            return AEA_TABLE_CACHE.read_text()

    # Download fresh
    print("Downloading AEA migration table from GitHub...")
    response = requests.get(AEA_MIGRATION_TABLE_URL, timeout=30)
    response.raise_for_status()

    # Cache the response
    AEA_TABLE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    AEA_TABLE_CACHE.write_text(response.text)

    return response.text


def load_aea_migration_table() -> dict:
    """
    Load AEA migration table and return mapping from openICPSR ID to article DOI.
    Returns dict: {base_id: {'article_doi': ..., 'journal': ...}}

    The table has columns: "", "doi", "icpsr_doi", "title"
    - doi: article DOI (10.1257/...)
    - icpsr_doi: package DOI (10.3886/E...)
    """
    csv_text = download_aea_migration_table()

    # Parse CSV
    reader = csv.DictReader(StringIO(csv_text))

    mapping = {}
    for row in reader:
        # Get article DOI from 'doi' column
        article_doi = row.get('doi', '').strip()

        # Get ICPSR package DOI from 'icpsr_doi' column
        icpsr_doi = row.get('icpsr_doi', '').strip()

        if not article_doi or not icpsr_doi:
            continue

        # Extract base_id from icpsr_doi (e.g., "10.3886/E112306V1" -> "E112306")
        base_match = re.search(r'E(\d+)', icpsr_doi.upper())
        if not base_match:
            continue

        base_id = f"E{base_match.group(1)}"

        mapping[base_id] = {
            'article_doi': article_doi.lower().strip(),
            'journal': identify_journal_from_doi(article_doi.lower().strip())
        }

    print(f"Loaded {len(mapping)} entries from AEA migration table")
    return mapping


def identify_journal_from_doi(article_doi: str) -> Optional[str]:
    """Identify journal from article DOI stem."""
    article_doi_lower = article_doi.lower()
    for stem, journal in DOI_TO_JOURNAL.items():
        if article_doi_lower.startswith(stem):
            return journal
    return None


def identify_from_related_ids(pkg: dict) -> dict:
    """
    Try to identify journal from package's relatedIdentifiers.
    Returns mapping dict with article_dois, journal_code, confidence, source.
    """
    related = pkg.get('related_identifiers', [])
    article_dois = set()  # Use set to avoid duplicates
    journal_code = None

    for rel in related:
        rel_type = rel.get('relatedIdentifierType', '').upper()
        rel_id = rel.get('relatedIdentifier', '')

        if rel_type == 'DOI' and rel_id:
            article_doi = rel_id.lower()

            # Check if this is an AEA article DOI
            journal = identify_journal_from_doi(article_doi)
            if journal:
                article_dois.add(article_doi)
                if not journal_code:  # Use first match
                    journal_code = journal

    if journal_code:
        return {
            'article_dois': sorted(list(article_dois)),  # Convert to sorted list
            'journal_code': journal_code,
            'confidence': 'high',
            'source': 'datacite_relatedIdentifier'
        }

    return {
        'article_dois': [],
        'journal_code': None,
        'confidence': None,
        'source': 'unresolved'
    }


def identify_from_aea_table(pkg: dict, aea_table: dict) -> dict:
    """
    Try to identify journal from AEA migration table.
    """
    base_id = pkg.get('base_id', '')

    if base_id in aea_table:
        entry = aea_table[base_id]
        return {
            'article_dois': [entry['article_doi']] if entry['article_doi'] else [],
            'journal_code': entry['journal'],
            'confidence': 'high' if entry['journal'] else 'low',
            'source': 'aea_migration_table'
        }

    return {
        'article_dois': [],
        'journal_code': None,
        'confidence': None,
        'source': 'unresolved'
    }


def identify_journals():
    """
    Main function to identify journals for AEA packages.

    Filters the full ICPSR universe to AEA packages only.
    A package is "AEA" if it has a relatedIdentifier pointing to a 10.1257/* DOI
    or is in the AEA migration table. Non-AEA packages are excluded from output.
    """
    print("=" * 60)
    print("AEA Package Journal Identification")
    print("=" * 60)

    # Check that input file exists
    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        print("Please run 01_collect_dois.py first.")
        return

    # Load packages (full ICPSR universe)
    print(f"Loading packages from {INPUT_FILE}...")
    packages = load_jsonl(INPUT_FILE)
    print(f"Loaded {len(packages)} total ICPSR packages")

    # Load existing mappings for resumability
    existing = load_existing_mappings(OUTPUT_FILE)
    print(f"Found {len(existing)} existing AEA mappings")

    # Load AEA migration table (one-time download)
    aea_table = load_aea_migration_table()

    # Process packages - only include those that are AEA-linked
    results = []
    stats = {
        'total_icpsr': len(packages),
        'aea_found': 0,
        'from_datacite': 0,
        'from_aea_table': 0,
        'non_aea_excluded': 0,
        'skipped_existing': 0,
        'journals': {}
    }

    for pkg in packages:
        pkg_doi = pkg['package_doi']

        # Skip if already processed (resumability) - only if it was an AEA package
        if pkg_doi in existing:
            results.append(existing[pkg_doi])
            stats['skipped_existing'] += 1
            stats['aea_found'] += 1
            journal = existing[pkg_doi].get('journal_code')
            if journal:
                stats['journals'][journal] = stats['journals'].get(journal, 0) + 1
            continue

        # Try DataCite relatedIdentifiers first
        mapping = identify_from_related_ids(pkg)

        # Fallback to AEA table if unresolved
        if mapping['source'] == 'unresolved':
            mapping = identify_from_aea_table(pkg, aea_table)

        # Only include packages that have a journal mapping (are AEA)
        if mapping['journal_code'] is not None:
            stats['aea_found'] += 1

            # Track source stats
            if mapping['source'] == 'datacite_relatedIdentifier':
                stats['from_datacite'] += 1
            elif mapping['source'] == 'aea_migration_table':
                stats['from_aea_table'] += 1

            # Track journal counts
            journal = mapping['journal_code']
            stats['journals'][journal] = stats['journals'].get(journal, 0) + 1

            # Build result record
            result = {
                'package_doi': pkg_doi,
                **mapping
            }
            results.append(result)
        else:
            # Not an AEA package - exclude from output
            stats['non_aea_excluded'] += 1

    # Write output (only AEA packages)
    print(f"\nWriting {len(results)} AEA package mappings to {OUTPUT_FILE}")
    save_jsonl(OUTPUT_FILE, results)

    # Print summary
    print("\n" + "=" * 60)
    print("=== Journal Identification Summary ===")
    print("=" * 60)
    print(f"Total ICPSR packages scanned: {stats['total_icpsr']}")
    print(f"AEA packages found: {stats['aea_found']}")
    print(f"  - From DataCite relatedIdentifiers: {stats['from_datacite']}")
    print(f"  - From AEA migration table: {stats['from_aea_table']}")
    print(f"Non-AEA packages (excluded): {stats['non_aea_excluded']}")

    if stats['skipped_existing']:
        print(f"Skipped (already in cache): {stats['skipped_existing']}")

    print("\nBy journal:")
    for journal, count in sorted(stats['journals'].items(), key=lambda x: -x[1]):
        print(f"  {journal}: {count}")

    api_calls = 0 if AEA_TABLE_CACHE.exists() else 1
    print(f"\nAPI calls made: {api_calls} (AEA table download)")
    print(f"\nOutput written to: {OUTPUT_FILE}")


if __name__ == '__main__':
    identify_journals()
