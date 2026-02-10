#!/usr/bin/env python3
"""
select_new_papers.py

Selects ~110 new papers from the AEA package-to-journal mapping for
batch 200 expansion. Uses stratified random sampling by journal.

Usage:
    python scripts/select_new_papers.py [--n 110] [--seed 42]
"""

import json
import re
import random
import argparse
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).parent.parent
JOURNAL_MAPPING_FILE = BASE_DIR / "data" / "metadata" / "aea_package_to_journal.jsonl"
STATUS_FILE = BASE_DIR / "data" / "tracking" / "spec_search_status.json"
OUTPUT_FILE = BASE_DIR / "data" / "metadata" / "batch_200_selection.jsonl"

# Journal targets (approximate, will be adjusted proportionally)
JOURNAL_TARGETS = {
    "AER": 20,
    "AEJ: Applied": 17,
    "AEJ: Policy": 16,
    "AER: Insights": 11,
    "AEJ: Macro": 11,
    "AEJ: Micro": 8,
    "AER: P&P": 11,
    "JEL": 4,
    "JEP": 4,
}


def load_existing_paper_ids():
    """Load paper IDs already analyzed or excluded."""
    existing = set()
    if STATUS_FILE.exists():
        with open(STATUS_FILE) as f:
            status = json.load(f)
        for p in status.get("packages_with_data", []):
            existing.add(p["id"])
        for p in status.get("packages_without_data", []):
            existing.add(p["id"])
    return existing


def load_candidate_packages():
    """Load all versioned packages from journal mapping, keeping highest version per base ID."""
    packages = {}  # base_id -> best entry

    with open(JOURNAL_MAPPING_FILE) as f:
        for line in f:
            entry = json.loads(line.strip())
            doi = entry["package_doi"]

            # Only use versioned entries (e.g., E109242V3)
            m = re.search(r"E(\d+)(V\d+)$", doi)
            if not m:
                continue

            base_num = m.group(1)
            version = m.group(2)
            paper_id = f"{base_num}-{version}"

            # Keep highest version per base package
            if base_num not in packages or version > packages[base_num]["version"]:
                packages[base_num] = {
                    "paper_id": paper_id,
                    "base_num": base_num,
                    "version": version,
                    "journal": entry["journal_code"],
                    "article_dois": entry.get("article_dois", []),
                    "confidence": entry.get("confidence", ""),
                    "package_doi": doi,
                }

    return packages


def stratified_sample(candidates, targets, n_total, seed=42):
    """Stratified random sample by journal."""
    random.seed(seed)

    # Group by journal
    by_journal = {}
    for pid, info in candidates.items():
        j = info["journal"]
        by_journal.setdefault(j, []).append(info)

    print(f"\nAvailable candidates by journal:")
    for j in sorted(by_journal.keys()):
        print(f"  {j}: {len(by_journal[j])}")

    # Calculate proportional draws
    total_target = sum(targets.values())
    selected = []

    for journal, target in sorted(targets.items(), key=lambda x: -x[1]):
        pool = by_journal.get(journal, [])
        if not pool:
            print(f"  Warning: No candidates for {journal}")
            continue

        # Scale target proportionally to n_total
        scaled_target = max(1, round(target * n_total / total_target))
        n_draw = min(scaled_target, len(pool))

        drawn = random.sample(pool, n_draw)
        selected.extend(drawn)
        print(f"  {journal}: drew {n_draw} / {len(pool)} available (target: {target})")

    # If we're short, draw more from the largest pools
    shortfall = n_total - len(selected)
    if shortfall > 0:
        selected_ids = {s["paper_id"] for s in selected}
        remaining = [
            info for pid, info in candidates.items()
            if info["paper_id"] not in selected_ids
        ]
        random.shuffle(remaining)
        extra = remaining[:shortfall]
        selected.extend(extra)
        print(f"\n  Drew {len(extra)} extra to reach target")

    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=110, help="Number of papers to select")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 60)
    print("Paper Selection for Batch 200")
    print("=" * 60)

    # Load existing
    existing_ids = load_existing_paper_ids()
    print(f"\nExisting papers (analyzed or excluded): {len(existing_ids)}")

    # Load candidates
    all_packages = load_candidate_packages()
    print(f"Total versioned packages in mapping: {len(all_packages)}")

    # Filter out existing
    candidates = {
        k: v for k, v in all_packages.items()
        if v["paper_id"] not in existing_ids
    }
    print(f"Candidates after excluding existing: {len(candidates)}")

    # Stratified sample
    selected = stratified_sample(candidates, JOURNAL_TARGETS, args.n, args.seed)

    # Write output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        for entry in selected:
            f.write(json.dumps(entry) + "\n")

    print(f"\n{'=' * 60}")
    print(f"Selected {len(selected)} papers")
    print(f"Written to: {OUTPUT_FILE}")

    # Summary
    journal_counts = Counter(e["journal"] for e in selected)
    print(f"\nSelection by journal:")
    for j, c in journal_counts.most_common():
        print(f"  {j}: {c}")


if __name__ == "__main__":
    main()
