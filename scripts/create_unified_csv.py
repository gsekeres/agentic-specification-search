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
import pandas as pd
from pathlib import Path
from datetime import datetime

# Configuration
BASE_DIR = Path(__file__).parent.parent
EXTRACTED_DIR = BASE_DIR / "data" / "downloads" / "extracted"
OUTPUT_FILE = BASE_DIR / "unified_results.csv"
TRACKING_FILE = BASE_DIR / "data" / "tracking" / "completed_analyses.jsonl"

# Required columns for unified output
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
    't_stat',
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
    'model_type',
    'estimation_script'
]


def find_specification_results():
    """Find all specification_results.csv files in extracted packages."""
    results_files = []

    if not EXTRACTED_DIR.exists():
        print(f"Warning: Extracted directory not found: {EXTRACTED_DIR}")
        return results_files

    for package_dir in EXTRACTED_DIR.iterdir():
        if not package_dir.is_dir():
            continue

        # Look for specification_results.csv in package root
        results_file = package_dir / "specification_results.csv"
        if results_file.exists():
            results_files.append(results_file)
            continue

        # Also check subdirectories (some packages have nested structure)
        for subdir in package_dir.iterdir():
            if subdir.is_dir():
                results_file = subdir / "specification_results.csv"
                if results_file.exists():
                    results_files.append(results_file)

    return results_files


def load_and_validate_csv(filepath):
    """Load a specification_results.csv and validate its columns."""
    try:
        df = pd.read_csv(filepath)

        # Extract paper_id from path if not in data
        paper_id = filepath.parent.name
        if '-V' not in paper_id:
            # Nested structure - get parent
            paper_id = filepath.parent.parent.name

        if 'paper_id' not in df.columns:
            df['paper_id'] = paper_id

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
        df = load_and_validate_csv(filepath)
        if df is not None and len(df) > 0:
            dfs.append(df)
            papers_processed.append(df['paper_id'].iloc[0])
            print(f"  Loaded {len(df)} specs from {filepath.parent.name}")

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
