#!/usr/bin/env python3
"""
Script to track which packages need downloading.
Outputs the next batch of packages to download.
"""

import json
import os
from pathlib import Path

def get_remaining_packages():
    metadata_path = Path('/Users/gabesekeres/Dropbox/Papers/competition_science/aea/data/metadata/all_replication_2022_classified.jsonl')
    extracted_dir = Path('/Users/gabesekeres/Dropbox/Papers/competition_science/aea/data/downloads/extracted')
    raw_dir = Path('/Users/gabesekeres/Dropbox/Papers/competition_science/aea/data/downloads/raw_packages')

    # Get already downloaded/extracted project IDs
    done = set()
    if extracted_dir.exists():
        for name in os.listdir(extracted_dir):
            if name.startswith('.'):
                continue
            pid = name.split('-')[0]
            done.add(pid)

    if raw_dir.exists():
        for name in os.listdir(raw_dir):
            if name.endswith('.zip'):
                pid = name.split('-')[0]
                done.add(pid)

    # Get eligible packages not yet downloaded
    eligible = []
    with open(metadata_path) as f:
        for line in f:
            d = json.loads(line)
            if d.get('is_observational') == True and d.get('has_public_data') == True:
                if d.get('project_id') not in done:
                    eligible.append(d)

    # Sort by year descending
    eligible.sort(key=lambda x: (-x.get('publication_year', 0), x.get('project_id', '')))

    return eligible

if __name__ == '__main__':
    remaining = get_remaining_packages()
    print(f"Remaining packages to download: {len(remaining)}")
    print()
    for pkg in remaining[:20]:
        url = f"https://www.openicpsr.org/openicpsr/project/{pkg['project_id']}/version/V1/view"
        print(f"{pkg['project_id']} | {pkg.get('publication_year')} | {pkg.get('title','')[:50]}")
        print(f"  URL: {url}")
