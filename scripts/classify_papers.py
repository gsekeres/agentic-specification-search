"""
Paper Classifier for AEA Replication Packages

Uses Claude CLI to classify papers based on their metadata:
- Is it observational (vs experimental/simulation)?
- Does it have public data?
- What is the primary software?
- What journal is it from?

Usage:
    python classify_papers.py --input metadata.jsonl --output classified.jsonl
"""

import argparse
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

try:
    from .datacite_fetcher import AEAPackageMetadata, load_metadata, save_metadata
except ImportError:
    from datacite_fetcher import AEAPackageMetadata, load_metadata, save_metadata


CLASSIFICATION_PROMPT = """You are classifying an economics replication package for a research project studying p-hacking in observational studies.

Given this paper metadata, classify it:

Title: {title}
Authors: {creators}
Year: {year}
Description: {description}
Subjects/Keywords: {subjects}

Answer these questions with JSON:

1. is_observational: Is this an observational study (using real-world data to estimate causal effects)?
   - TRUE if: diff-in-diff, regression discontinuity, instrumental variables, panel data analysis, cross-sectional regression, event studies, synthetic control
   - FALSE if: lab experiment, field experiment (RCT), survey experiment, theoretical/simulation only, calibration exercise, structural estimation without reduced-form

2. has_public_data: Based on the title/description, does this likely have fully public replication data?
   - TRUE if: uses publicly available datasets (CPS, ACS, IPUMS, Census, administrative data from public sources)
   - FALSE if: mentions proprietary data, confidential data, restricted access, NDA, licensed data
   - UNKNOWN if: cannot determine

3. primary_software: What software is this likely coded in?
   - "stata", "r", "python", "matlab", "julia", "sas", "mixed", or "unknown"

4. journal: What AEA journal is this from (based on publication patterns)?
   - "AER" (American Economic Review)
   - "AEJ-Applied" (AEJ: Applied Economics)
   - "AEJ-Policy" (AEJ: Economic Policy)
   - "AEJ-Macro" (AEJ: Macroeconomics)
   - "AEJ-Micro" (AEJ: Microeconomics)
   - "JEL" (Journal of Economic Literature)
   - "JEP" (Journal of Economic Perspectives)
   - "AER-Insights"
   - "unknown"

Return ONLY valid JSON in this format:
{{"is_observational": true/false, "has_public_data": true/false/null, "primary_software": "stata", "journal": "AER"}}
"""


def classify_with_claude(metadata: AEAPackageMetadata, timeout: int = 60) -> dict:
    """
    Use Claude CLI to classify a paper.

    Returns dict with classification fields.
    """
    prompt = CLASSIFICATION_PROMPT.format(
        title=metadata.title,
        creators=", ".join(metadata.creators[:5]),
        year=metadata.publication_year,
        description=metadata.description[:500],
        subjects=", ".join(metadata.subjects[:10]),
    )

    try:
        result = subprocess.run(
            ['claude', '-p', prompt],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode != 0:
            print(f"  Claude error: {result.stderr[:100]}")
            return {}

        # Parse JSON from response
        output = result.stdout.strip()

        # Try to extract JSON from the response
        # Claude might include explanation text, so look for JSON object
        import re
        json_match = re.search(r'\{[^}]+\}', output)
        if json_match:
            return json.loads(json_match.group())
        else:
            print(f"  No JSON found in response")
            return {}

    except subprocess.TimeoutExpired:
        print(f"  Claude timeout")
        return {}
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        return {}
    except FileNotFoundError:
        print("  Claude CLI not found")
        return {}


def classify_batch(
    metadata_list: list[AEAPackageMetadata],
    delay_seconds: float = 1.0,
    skip_classified: bool = True
) -> list[AEAPackageMetadata]:
    """
    Classify a batch of papers.

    Args:
        metadata_list: List of metadata to classify
        delay_seconds: Delay between API calls
        skip_classified: Skip already-classified papers

    Returns:
        Updated metadata list with classifications
    """
    results = []

    for i, m in enumerate(metadata_list):
        print(f"[{i+1}/{len(metadata_list)}] {m.title[:60]}...")

        # Skip if already classified
        if skip_classified and m.is_observational is not None:
            print(f"  Already classified, skipping")
            results.append(m)
            continue

        classification = classify_with_claude(m)

        if classification:
            m.is_observational = classification.get('is_observational')
            m.has_public_data = classification.get('has_public_data')
            m.primary_software = classification.get('primary_software')
            m.journal = classification.get('journal')
            print(f"  observational={m.is_observational}, public_data={m.has_public_data}, sw={m.primary_software}")
        else:
            print(f"  Classification failed")

        results.append(m)

        # Rate limit
        if i < len(metadata_list) - 1:
            time.sleep(delay_seconds)

    return results


def filter_observational(
    metadata_list: list[AEAPackageMetadata],
    require_public_data: bool = False
) -> list[AEAPackageMetadata]:
    """Filter to only observational studies."""
    filtered = [m for m in metadata_list if m.is_observational]

    if require_public_data:
        filtered = [m for m in filtered if m.has_public_data]

    return filtered


def main():
    parser = argparse.ArgumentParser(
        description='Classify AEA papers as observational/experimental'
    )
    parser.add_argument(
        '--input', '-i',
        type=Path,
        required=True,
        help='Input metadata file (jsonl or json)'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output file (default: input with _classified suffix)'
    )
    parser.add_argument(
        '--limit', '-n',
        type=int,
        help='Maximum papers to classify'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay between Claude calls in seconds (default: 1.0)'
    )
    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='Re-classify already classified papers'
    )

    args = parser.parse_args()

    # Load metadata
    print(f"Loading metadata from {args.input}...")
    metadata = load_metadata(args.input)
    print(f"Loaded {len(metadata)} papers")

    # Limit if specified
    if args.limit:
        metadata = metadata[:args.limit]
        print(f"Limited to {len(metadata)} papers")

    # Classify
    print()
    print("Classifying papers with Claude...")
    print()

    classified = classify_batch(
        metadata,
        delay_seconds=args.delay,
        skip_classified=not args.no_skip
    )

    # Summary
    print()
    print("Classification Summary")
    print("=" * 40)

    observational = [m for m in classified if m.is_observational]
    experimental = [m for m in classified if m.is_observational is False]
    unknown = [m for m in classified if m.is_observational is None]

    print(f"Observational: {len(observational)}")
    print(f"Experimental/Other: {len(experimental)}")
    print(f"Unknown: {len(unknown)}")

    if observational:
        public_data = [m for m in observational if m.has_public_data]
        print(f"Observational with public data: {len(public_data)}")

    # Save
    output_path = args.output
    if not output_path:
        output_path = args.input.with_stem(args.input.stem + '_classified')

    save_metadata(classified, output_path, 'jsonl')


if __name__ == '__main__':
    main()
