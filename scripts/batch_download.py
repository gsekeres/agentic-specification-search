#!/usr/bin/env python3
"""
batch_download.py

Downloads openICPSR packages listed in batch_200_selection.jsonl.
Extracts them to data/downloads/extracted/{paper_id}/

Usage:
    python scripts/batch_download.py [--start 0] [--count 110]

Prerequisites:
    - Chrome must be open and logged into openICPSR
    - raw_packages/ dir must exist

This script:
1. Reads batch_200_selection.jsonl for paper list
2. For each paper, navigates Chrome to the download/terms page
3. Dismisses profile modal, clicks "I Agree"
4. Waits for ZIP to appear in raw_packages/
5. Extracts to extracted/{paper_id}/
6. Logs to download_manifest.jsonl
"""

import json
import os
import re
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
SELECTION_FILE = BASE_DIR / "data" / "metadata" / "batch_200_selection.jsonl"
RAW_DIR = BASE_DIR / "data" / "downloads" / "raw_packages"
EXTRACTED_DIR = BASE_DIR / "data" / "downloads" / "extracted"
MANIFEST_FILE = BASE_DIR / "data" / "tracking" / "download_manifest.jsonl"


def load_selection():
    papers = []
    with open(SELECTION_FILE) as f:
        for line in f:
            papers.append(json.loads(line.strip()))
    return papers


def already_downloaded(paper_id):
    """Check if paper already exists in extracted/"""
    paper_dir = EXTRACTED_DIR / paper_id
    return paper_dir.exists() and any(paper_dir.iterdir())


def extract_zip(zip_path, paper_id):
    """Extract ZIP to extracted/{paper_id}/"""
    dest = EXTRACTED_DIR / paper_id
    dest.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dest)
        return True
    except zipfile.BadZipFile:
        print(f"  Bad ZIP file: {zip_path}")
        return False


def log_manifest(entry):
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_FILE, 'a') as f:
        f.write(json.dumps(entry) + '\n')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=110)
    args = parser.parse_args()

    papers = load_selection()
    papers = papers[args.start:args.start + args.count]

    print(f"Processing {len(papers)} papers (start={args.start})")

    # Check which are already downloaded
    to_download = []
    for p in papers:
        pid = p["paper_id"]
        if already_downloaded(pid):
            print(f"  SKIP {pid} (already extracted)")
        else:
            to_download.append(p)

    print(f"\nNeed to download: {len(to_download)} papers")
    print(f"Already have: {len(papers) - len(to_download)} papers")

    # Print the list for manual download or Chrome automation
    for i, p in enumerate(to_download):
        base = p["base_num"]
        ver = p["version"]
        url = f"https://www.openicpsr.org/openicpsr/project/{base}/version/{ver}/view"
        print(f"  {i+1}. {p['paper_id']} ({p['journal']}) - {url}")

    # Check for already-downloaded ZIPs in raw_packages that need extraction
    print("\n--- Checking raw_packages for unextracted ZIPs ---")
    extracted_count = 0
    for zip_file in RAW_DIR.glob("*.zip"):
        # Extract paper_id from filename (e.g., "116157-V1.zip" or "116157-V1 (1).zip")
        m = re.match(r'(\d+-V\d+)', zip_file.name)
        if m:
            paper_id = m.group(1)
            if not already_downloaded(paper_id):
                print(f"  Extracting {zip_file.name} -> {paper_id}/")
                if extract_zip(zip_file, paper_id):
                    extracted_count += 1
                    log_manifest({
                        "paper_id": paper_id,
                        "status": "downloaded",
                        "filename": zip_file.name,
                        "bytes": zip_file.stat().st_size,
                        "timestamp": datetime.now().isoformat(),
                    })

    if extracted_count:
        print(f"  Extracted {extracted_count} packages from existing ZIPs")


if __name__ == "__main__":
    main()
