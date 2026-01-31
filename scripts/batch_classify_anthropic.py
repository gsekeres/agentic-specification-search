#!/usr/bin/env python3
"""
Batch Paper Classifier using Anthropic API

Classifies AEA replication packages in batches for efficiency.
Uses the Anthropic Python SDK instead of CLI for faster processing.

Usage:
    python batch_classify_anthropic.py --input metadata.jsonl --output classified.jsonl --batch-size 10
"""

import argparse
import json
import os
import time
from pathlib import Path
from dataclasses import asdict

try:
    import anthropic
except ImportError:
    print("Error: anthropic package not installed. Run: pip install anthropic")
    exit(1)

try:
    from datacite_fetcher import AEAPackageMetadata, load_metadata, save_metadata
except ImportError:
    from .datacite_fetcher import AEAPackageMetadata, load_metadata, save_metadata


BATCH_CLASSIFICATION_PROMPT = """You are classifying economics replication packages. For each package below, provide a JSON classification.

Classification criteria:
- is_observational: TRUE if observational study (diff-in-diff, RD, IV, panel data, event studies, synthetic control). FALSE if experimental (RCT, lab, field experiment) or theoretical/simulation only.
- has_public_data: TRUE if uses publicly available data (CPS, ACS, Census, public admin data). FALSE if proprietary/restricted. null if unknown.
- primary_software: "stata", "r", "python", "matlab", "mixed", or "unknown"
- journal: "AER", "AEJ-Applied", "AEJ-Policy", "AEJ-Macro", "AEJ-Micro", "JEL", "JEP", "AER-Insights", or "unknown"

Packages to classify:
{packages}

Return a JSON array with one object per package, in the same order:
[
  {{"doi": "...", "is_observational": true/false, "has_public_data": true/false/null, "primary_software": "stata", "journal": "AER"}},
  ...
]

ONLY return the JSON array, no other text."""


def format_package_for_prompt(m: AEAPackageMetadata, idx: int) -> str:
    """Format a single package for the batch prompt."""
    return f"""
{idx}. DOI: {m.doi}
   Title: {m.title[:200]}
   Year: {m.publication_year}
   Description: {m.description[:300] if m.description else 'N/A'}
   Keywords: {', '.join(m.subjects[:5]) if m.subjects else 'N/A'}"""


def classify_batch(client: anthropic.Anthropic, packages: list[AEAPackageMetadata]) -> list[dict]:
    """Classify a batch of packages using a single API call."""
    # Format packages for prompt
    packages_text = "\n".join(
        format_package_for_prompt(p, i+1) for i, p in enumerate(packages)
    )

    prompt = BATCH_CLASSIFICATION_PROMPT.format(packages=packages_text)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse JSON from response
        content = response.content[0].text.strip()

        # Try to extract JSON array
        if content.startswith('['):
            results = json.loads(content)
        else:
            # Try to find JSON in response
            import re
            match = re.search(r'\[[\s\S]*\]', content)
            if match:
                results = json.loads(match.group())
            else:
                print(f"  Warning: Could not parse JSON from response")
                return [{}] * len(packages)

        return results

    except Exception as e:
        print(f"  Error in batch classification: {e}")
        return [{}] * len(packages)


def classify_all(
    metadata_list: list[AEAPackageMetadata],
    batch_size: int = 10,
    delay_seconds: float = 0.5
) -> list[AEAPackageMetadata]:
    """Classify all packages in batches."""

    # Initialize Anthropic client
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    results = []
    total_batches = (len(metadata_list) + batch_size - 1) // batch_size

    for i in range(0, len(metadata_list), batch_size):
        batch = metadata_list[i:i + batch_size]
        batch_num = i // batch_size + 1

        print(f"Batch {batch_num}/{total_batches} ({len(batch)} packages)...")

        classifications = classify_batch(client, batch)

        # Apply classifications to metadata
        for j, (m, c) in enumerate(zip(batch, classifications)):
            if c:
                m.is_observational = c.get('is_observational')
                m.has_public_data = c.get('has_public_data')
                m.primary_software = c.get('primary_software')
                m.journal = c.get('journal')

            results.append(m)

        # Rate limiting
        if i + batch_size < len(metadata_list):
            time.sleep(delay_seconds)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Batch classify AEA papers using Anthropic API'
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
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=10,
        help='Number of packages per API call (default: 10)'
    )
    parser.add_argument(
        '--limit', '-n',
        type=int,
        help='Maximum packages to classify'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.5,
        help='Delay between batches in seconds (default: 0.5)'
    )

    args = parser.parse_args()

    # Load metadata
    print(f"Loading metadata from {args.input}...")
    metadata = load_metadata(args.input)
    print(f"Loaded {len(metadata)} packages")

    # Limit if specified
    if args.limit:
        metadata = metadata[:args.limit]
        print(f"Limited to {args.limit} packages")

    # Classify
    print()
    print(f"Classifying in batches of {args.batch_size}...")
    print()

    classified = classify_all(
        metadata,
        batch_size=args.batch_size,
        delay_seconds=args.delay
    )

    # Summary
    print()
    print("Classification Summary")
    print("=" * 40)

    observational = [m for m in classified if m.is_observational is True]
    experimental = [m for m in classified if m.is_observational is False]
    unknown = [m for m in classified if m.is_observational is None]

    print(f"Observational: {len(observational)}")
    print(f"Experimental/Other: {len(experimental)}")
    print(f"Unknown: {len(unknown)}")

    if observational:
        public_data = [m for m in observational if m.has_public_data]
        print(f"Observational with public data: {len(public_data)}")

    # Journal distribution
    journals = {}
    for m in classified:
        j = m.journal or 'unknown'
        journals[j] = journals.get(j, 0) + 1
    print("\nJournal distribution:")
    for j, count in sorted(journals.items(), key=lambda x: -x[1]):
        print(f"  {j}: {count}")

    # Save
    output_path = args.output
    if not output_path:
        output_path = args.input.with_stem(args.input.stem + '_classified')

    save_metadata(classified, output_path, 'jsonl')


if __name__ == '__main__':
    main()
