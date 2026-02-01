#!/usr/bin/env python3
"""
create_unified_csv.py

Aggregates specification_results.csv files from all analyzed papers into
a single unified_results.csv file.

Usage:
    python scripts/create_unified_csv.py
"""

import os
import json
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime

# Configuration
BASE_DIR = Path(__file__).parent.parent
EXTRACTED_DIR = BASE_DIR / "data" / "downloads" / "extracted"
OUTPUT_FILE = BASE_DIR / "unified_results.csv"
TRACKING_FILE = BASE_DIR / "data" / "tracking" / "completed_analyses.jsonl"
DATACITE_DB = BASE_DIR / "data" / "cache" / "datacite_responses.db"

# Journal code mapping from article DOI prefix
JOURNAL_MAP = {
    'aer': 'AER',
    'app': 'AEJ: Applied',
    'pol': 'AEJ: Policy',
    'mic': 'AEJ: Micro',
    'mac': 'AEJ: Macro',
    'pandp': 'AER: P&P',
    'jel': 'JEL',
    'jep': 'JEP',
}

# Required columns for unified output (removed unused: t_stat, model_type, estimation_script)
REQUIRED_COLUMNS = [
    'paper_id',
    'journal',
    'paper_title',
    'spec_id',
    'spec_tree_path',
    'outcome_var',
    'treatment_var',
    'coefficient',
    'std_error',
    'p_value',
    'ci_lower',
    'ci_upper',
    'n_obs',
    'r_squared',
    'coefficient_vector_json',
    'sample_desc',
    'fixed_effects',
    'controls_desc',
    'cluster_var',
]


def load_package_metadata():
    """Load package metadata (title, journal) from datacite database."""
    metadata = {}
    base_records = {}  # Store base DOI records (without version suffix) for fallback

    if not DATACITE_DB.exists():
        print(f"Warning: Datacite database not found: {DATACITE_DB}")
        return metadata

    conn = sqlite3.connect(DATACITE_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT package_doi, package_json FROM packages")

    for row in cursor.fetchall():
        package_doi, package_json = row
        try:
            pkg = json.loads(package_json)
            # Extract base_id and version to create paper_id (e.g., "113597-V2")
            base_id = pkg.get('base_id', '').replace('E', '')
            version = pkg.get('version', 'V1')
            paper_id = f"{base_id}-{version}"

            # Get title
            title = pkg.get('title', '')
            # Clean up title prefixes
            for prefix in ['Data and Code for: ', 'Data and Code for ',
                          'Replication data for: ', 'Replication data for ',
                          'Code for: ', 'Code for ', '"', '"']:
                if title.startswith(prefix):
                    title = title[len(prefix):]
            # Also strip trailing quotes
            title = title.rstrip('"').rstrip('"')

            # Derive journal from article DOI
            journal = ''
            related = pkg.get('related_identifiers', [])
            for rel in related:
                rel_doi = rel.get('relatedIdentifier', '')
                if rel_doi.startswith('10.1257/'):
                    # Extract journal code from DOI (e.g., 10.1257/app.20130489 -> app)
                    parts = rel_doi.replace('10.1257/', '').split('.')
                    if parts:
                        journal_code = parts[0]
                        journal = JOURNAL_MAP.get(journal_code, journal_code.upper())
                        break

            # Check if this is a base DOI record (no version in DOI)
            is_base_record = not any(f'V{i}' in package_doi for i in range(1, 10))
            if is_base_record and journal:
                # Store for fallback lookup
                base_records[base_id] = {'title': title, 'journal': journal}

            metadata[paper_id] = {'title': title, 'journal': journal}

        except (json.JSONDecodeError, KeyError):
            continue

    conn.close()

    # Fill in missing journals from base records
    for paper_id, meta in metadata.items():
        if not meta['journal']:
            base_id = paper_id.split('-')[0]
            if base_id in base_records:
                meta['journal'] = base_records[base_id]['journal']
                if not meta['title']:
                    meta['title'] = base_records[base_id]['title']

    return metadata


def find_specification_results():
    """Find all specification_results.csv files in extracted packages."""
    results_files = []

    if not EXTRACTED_DIR.exists():
        print(f"Warning: Extracted directory not found: {EXTRACTED_DIR}")
        return results_files

    for package_dir in EXTRACTED_DIR.iterdir():
        if not package_dir.is_dir():
            continue

        # Recursively search for specification_results.csv (handles nested structures)
        for results_file in package_dir.rglob("specification_results.csv"):
            results_files.append(results_file)
            break  # Only take the first one per package

    return results_files


def load_and_validate_csv(filepath, metadata):
    """Load a specification_results.csv and validate its columns."""
    try:
        df = pd.read_csv(filepath)

        # Extract paper_id from path - find the directory matching pattern XXXXXX-VX
        paper_id = None
        for parent in filepath.parents:
            if parent.name and '-V' in parent.name and parent.name.split('-')[0].isdigit():
                paper_id = parent.name
                break
            # Stop at extracted directory
            if parent.name == 'extracted':
                break

        if paper_id is None:
            paper_id = filepath.parent.name  # Fallback

        if 'paper_id' not in df.columns:
            df['paper_id'] = paper_id

        # Look up metadata for this paper
        pkg_meta = metadata.get(paper_id, {})
        df['journal'] = pkg_meta.get('journal', '')
        df['paper_title'] = pkg_meta.get('title', '')

        # Add missing columns with None
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                df[col] = None

        return df[REQUIRED_COLUMNS]

    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def compute_stats(df):
    """Compute summary statistics for the unified results."""
    stats = {
        'total_papers': df['paper_id'].nunique(),
        'total_specifications': len(df),
        'specs_per_paper': len(df) / df['paper_id'].nunique() if df['paper_id'].nunique() > 0 else 0,
        'journals': df['journal'].value_counts().to_dict() if 'journal' in df.columns else {},
        'methods': {},
        'significance_rate': (df['p_value'] < 0.05).mean() if 'p_value' in df.columns else None,
    }

    # Count by spec_tree_path (method)
    if 'spec_tree_path' in df.columns:
        method_counts = df['spec_tree_path'].apply(
            lambda x: x.split('/')[1] if pd.notna(x) and '/' in str(x) else 'unknown'
        ).value_counts().to_dict()
        stats['methods'] = method_counts

    return stats


def update_tracking(papers_processed):
    """Update the tracking file with processed papers."""
    TRACKING_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(TRACKING_FILE, 'a') as f:
        for paper_id in papers_processed:
            record = {
                'paper_id': paper_id,
                'unified_at': datetime.now().isoformat(),
            }
            f.write(json.dumps(record) + '\n')


def main():
    print("=" * 60)
    print("Creating Unified Specification Results CSV")
    print("=" * 60)

    # Load package metadata from datacite database
    print("\nLoading package metadata from datacite database...")
    metadata = load_package_metadata()
    print(f"  Loaded metadata for {len(metadata)} packages")

    # Find all results files
    results_files = find_specification_results()
    print(f"\nFound {len(results_files)} specification_results.csv files")

    if not results_files:
        print("No results files found. Nothing to aggregate.")
        return

    # Load and concatenate
    dfs = []
    papers_processed = []

    for filepath in results_files:
        df = load_and_validate_csv(filepath, metadata)
        if df is not None and len(df) > 0:
            dfs.append(df)
            paper_id = df['paper_id'].iloc[0]
            papers_processed.append(paper_id)
            journal = df['journal'].iloc[0] or 'unknown'
            print(f"  Loaded {len(df)} specs from {paper_id} ({journal})")

    if not dfs:
        print("No valid data loaded.")
        return

    # Combine
    unified_df = pd.concat(dfs, ignore_index=True)

    # Sort by paper_id, then spec_id
    unified_df = unified_df.sort_values(['paper_id', 'spec_id'])

    # Save
    unified_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved unified results to: {OUTPUT_FILE}")

    # Compute and display stats
    stats = compute_stats(unified_df)
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"Total papers: {stats['total_papers']}")
    print(f"Total specifications: {stats['total_specifications']}")
    print(f"Average specs per paper: {stats['specs_per_paper']:.1f}")

    if stats['significance_rate'] is not None:
        print(f"Significance rate (p<0.05): {stats['significance_rate']*100:.1f}%")

    if stats['journals']:
        print("\nJournal distribution:")
        for journal, count in sorted(stats['journals'].items(), key=lambda x: -x[1]):
            print(f"  {journal}: {count}")

    if stats['methods']:
        print("\nMethod distribution:")
        for method, count in sorted(stats['methods'].items(), key=lambda x: -x[1]):
            print(f"  {method}: {count}")

    # Update tracking
    update_tracking(papers_processed)
    print(f"\nUpdated tracking file: {TRACKING_FILE}")


if __name__ == "__main__":
    main()
