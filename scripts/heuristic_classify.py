#!/usr/bin/env python3
"""
Heuristic Paper Classifier

Classifies AEA replication packages using keyword-based heuristics.
No API calls required - uses title and description patterns.

Usage:
    python heuristic_classify.py --input metadata.jsonl --output classified.jsonl
"""

import argparse
import json
import re
from pathlib import Path
from dataclasses import asdict

try:
    from datacite_fetcher import AEAPackageMetadata, load_metadata, save_metadata
except ImportError:
    from .datacite_fetcher import AEAPackageMetadata, load_metadata, save_metadata


# Keywords for classification
EXPERIMENTAL_KEYWORDS = [
    'experiment', 'rct', 'randomized', 'randomised', 'random assignment',
    'lab experiment', 'field experiment', 'laboratory', 'treatment group',
    'control group', 'randomization', 'randomisation', 'experimental design',
    'a]b test', 'behavioral experiment', 'behavioural experiment'
]

OBSERVATIONAL_KEYWORDS = [
    'difference-in-difference', 'diff-in-diff', 'did', 'difference in difference',
    'regression discontinuity', 'rdd', 'rd design', 'discontinuity design',
    'instrumental variable', ' iv ', '2sls', 'two-stage', 'two stage',
    'panel data', 'fixed effect', 'fixed-effect', 'event study', 'event-study',
    'synthetic control', 'synthetic diff', 'staggered', 'twfe',
    'propensity score', 'matching', 'triple difference', 'ddd',
    'bunching', 'kink', 'natural experiment', 'quasi-experiment',
    'causal effect', 'causal impact', 'identification strategy'
]

THEORETICAL_KEYWORDS = [
    'simulation', 'calibrat', 'structural model', 'structural estimation',
    'theoretical', 'model economy', 'monte carlo', 'numerical'
]

RESTRICTED_DATA_KEYWORDS = [
    'restricted', 'confidential', 'proprietary', 'nda', 'data use agreement',
    'administrative data', 'tax record', 'medical record', 'health record',
    'census bureau', 'fsrdc', 'rdc access', 'secure data', 'linked data'
]

PUBLIC_DATA_KEYWORDS = [
    'cps', 'current population survey', 'acs', 'american community survey',
    'ipums', 'census public', 'nhis', 'nlsy', 'psid', 'sipp',
    'world bank', 'fred', 'bls', 'bea data', 'openly available',
    'publicly available', 'public use', 'open data', 'download'
]

# Journal detection patterns
JOURNAL_PATTERNS = {
    'AER': ['american economic review', 'aer:', 'aer '],
    'AER-Insights': ['aer: insights', 'aer insights'],
    'AEJ-Applied': ['aej: applied', 'aej-applied', 'applied economics'],
    'AEJ-Policy': ['aej: economic policy', 'aej-policy', 'economic policy'],
    'AEJ-Macro': ['aej: macro', 'aej-macro', 'macroeconomics'],
    'AEJ-Micro': ['aej: micro', 'aej-micro', 'microeconomics'],
    'JEL': ['journal of economic literature', 'jel:'],
    'JEP': ['journal of economic perspectives', 'jep:'],
}

# Software detection
SOFTWARE_PATTERNS = {
    'stata': ['.do', 'stata', '.dta', 'reghdfe', 'estout'],
    'r': ['.r', 'rstudio', 'tidyverse', 'ggplot', '.rds', '.rdata'],
    'python': ['.py', 'python', 'pandas', 'numpy', 'jupyter', '.ipynb'],
    'matlab': ['.m', 'matlab', '.mat'],
    'julia': ['.jl', 'julia'],
    'sas': ['.sas', 'sas '],
}


def classify_package(m: AEAPackageMetadata) -> AEAPackageMetadata:
    """Classify a single package using heuristics."""
    text = f"{m.title} {m.description} {' '.join(m.subjects)}".lower()

    # Classify observational vs experimental
    exp_score = sum(1 for kw in EXPERIMENTAL_KEYWORDS if kw in text)
    obs_score = sum(1 for kw in OBSERVATIONAL_KEYWORDS if kw in text)
    theory_score = sum(1 for kw in THEORETICAL_KEYWORDS if kw in text)

    if exp_score > obs_score and exp_score > 0:
        m.is_observational = False
    elif theory_score > obs_score and theory_score > exp_score:
        m.is_observational = False  # Theoretical/simulation
    elif obs_score > 0 or (exp_score == 0 and theory_score == 0):
        # Default to observational if no experimental/theory keywords
        # Most econ papers are observational
        m.is_observational = True
    else:
        m.is_observational = None  # Ambiguous

    # Classify data availability
    restricted_score = sum(1 for kw in RESTRICTED_DATA_KEYWORDS if kw in text)
    public_score = sum(1 for kw in PUBLIC_DATA_KEYWORDS if kw in text)

    if restricted_score > public_score and restricted_score > 0:
        m.has_public_data = False
    elif public_score > restricted_score and public_score > 0:
        m.has_public_data = True
    else:
        m.has_public_data = None  # Unknown

    # Detect journal
    m.journal = 'unknown'
    for journal, patterns in JOURNAL_PATTERNS.items():
        for pattern in patterns:
            if pattern in text:
                m.journal = journal
                break
        if m.journal != 'unknown':
            break

    # If journal still unknown, try to infer from title patterns
    if m.journal == 'unknown':
        title_lower = m.title.lower()
        if 'replication' in title_lower or 'data for' in title_lower:
            # Default based on common patterns
            if any(kw in text for kw in ['policy', 'regulation', 'government']):
                m.journal = 'AEJ-Policy'
            elif any(kw in text for kw in ['macro', 'business cycle', 'monetary']):
                m.journal = 'AEJ-Macro'
            elif any(kw in text for kw in ['auction', 'mechanism', 'game']):
                m.journal = 'AEJ-Micro'
            else:
                m.journal = 'AEJ-Applied'  # Most common

    # Detect software
    m.primary_software = 'unknown'
    software_scores = {}
    for sw, patterns in SOFTWARE_PATTERNS.items():
        score = sum(1 for p in patterns if p in text)
        if score > 0:
            software_scores[sw] = score

    if software_scores:
        m.primary_software = max(software_scores, key=software_scores.get)
        if len(software_scores) > 1:
            m.primary_software = 'mixed'

    # Default to stata (most common in econ)
    if m.primary_software == 'unknown':
        m.primary_software = 'stata'

    return m


def classify_all(metadata_list: list[AEAPackageMetadata]) -> list[AEAPackageMetadata]:
    """Classify all packages."""
    return [classify_package(m) for m in metadata_list]


def main():
    parser = argparse.ArgumentParser(
        description='Classify AEA papers using heuristics (no API needed)'
    )
    parser.add_argument(
        '--input', '-i',
        type=Path,
        required=True,
        help='Input metadata file (jsonl)'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output file (default: input with _classified suffix)'
    )

    args = parser.parse_args()

    # Load metadata
    print(f"Loading metadata from {args.input}...")
    metadata = load_metadata(args.input)
    print(f"Loaded {len(metadata)} packages")

    # Classify
    print("\nClassifying packages...")
    classified = classify_all(metadata)

    # Summary
    print("\n" + "=" * 50)
    print("Classification Summary")
    print("=" * 50)

    observational = [m for m in classified if m.is_observational is True]
    experimental = [m for m in classified if m.is_observational is False]
    unknown_obs = [m for m in classified if m.is_observational is None]

    print(f"\nStudy Type:")
    print(f"  Observational: {len(observational)} ({100*len(observational)/len(classified):.1f}%)")
    print(f"  Experimental/Theory: {len(experimental)} ({100*len(experimental)/len(classified):.1f}%)")
    print(f"  Unknown: {len(unknown_obs)} ({100*len(unknown_obs)/len(classified):.1f}%)")

    public = [m for m in classified if m.has_public_data is True]
    restricted = [m for m in classified if m.has_public_data is False]
    unknown_data = [m for m in classified if m.has_public_data is None]

    print(f"\nData Availability:")
    print(f"  Public: {len(public)} ({100*len(public)/len(classified):.1f}%)")
    print(f"  Restricted: {len(restricted)} ({100*len(restricted)/len(classified):.1f}%)")
    print(f"  Unknown: {len(unknown_data)} ({100*len(unknown_data)/len(classified):.1f}%)")

    # Journal distribution
    journals = {}
    for m in classified:
        j = m.journal or 'unknown'
        journals[j] = journals.get(j, 0) + 1

    print(f"\nJournal Distribution:")
    for j, count in sorted(journals.items(), key=lambda x: -x[1]):
        print(f"  {j}: {count} ({100*count/len(classified):.1f}%)")

    # Software distribution
    software = {}
    for m in classified:
        s = m.primary_software or 'unknown'
        software[s] = software.get(s, 0) + 1

    print(f"\nSoftware Distribution:")
    for s, count in sorted(software.items(), key=lambda x: -x[1]):
        print(f"  {s}: {count} ({100*count/len(classified):.1f}%)")

    # Target selection stats
    obs_public = [m for m in classified if m.is_observational and m.has_public_data]
    print(f"\n{'='*50}")
    print(f"ELIGIBLE FOR ANALYSIS (observational + public data):")
    print(f"  {len(obs_public)} packages ({100*len(obs_public)/len(classified):.1f}%)")

    # Save
    output_path = args.output
    if not output_path:
        output_path = args.input.with_stem(args.input.stem.replace('_fresh', '') + '_classified')

    save_metadata(classified, output_path, 'jsonl')
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
