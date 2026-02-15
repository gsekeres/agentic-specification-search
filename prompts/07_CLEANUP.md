# Data Cleanup After Specification Search

## Overview

After running specification searches on downloaded packages, clean up intermediate files to save disk space while preserving the analysis results.

## What to Remove

### 1. Raw Package ZIP Files

Location: `data/downloads/raw_packages/*.zip`

These ZIP files are no longer needed after extraction. Remove all `.zip` files but keep the `.gitkeep` file.

```bash
rm data/downloads/raw_packages/*.zip
```

### 2. Extracted Package Contents (except results)

Location: `data/downloads/extracted/*/`

Each extracted package folder contains the original replication materials (data files, code, documentation). After specification search, only keep:

- `specification_results.csv` - The structured results from all specifications
- `inference_results.csv` - Inference-only recomputations (if present)
- `SPECIFICATION_SEARCH.md` - The summary report
- `SPECIFICATION_SURFACE.json` - The approved pre-run spec universe (surface)
- `SPECIFICATION_SURFACE.md` - Human-readable surface summary/rationale
- `SPEC_SURFACE_REVIEW.md` - Pre-run verifier critique/edits (if present)
- `diagnostics_results.csv` - Separate diagnostics table (if present)
- `spec_diagnostics_map.csv` - Spec-run ↔ diagnostic-run linkage (if present)

Remove everything else (data files, original code, etc.).

## Cleanup Commands

### Option A: Manual Cleanup

```bash
# Remove ZIP files
rm data/downloads/raw_packages/*.zip

# For each extracted package, remove everything except results
for dir in data/downloads/extracted/*/; do
    find "$dir" -type f \
        ! -name "specification_results.csv" \
        ! -name "inference_results.csv" \
        ! -name "SPECIFICATION_SEARCH.md" \
        ! -name "SPECIFICATION_SURFACE.json" \
        ! -name "SPECIFICATION_SURFACE.md" \
        ! -name "SPEC_SURFACE_REVIEW.md" \
        ! -name "diagnostics_results.csv" \
        ! -name "spec_diagnostics_map.csv" \
        -delete
    find "$dir" -type d -empty -delete
done
```

### Option B: Python Script

```python
import os
from pathlib import Path

BASE_DIR = Path("data/downloads")

# Remove ZIP files
for zip_file in (BASE_DIR / "raw_packages").glob("*.zip"):
    print(f"Removing: {zip_file}")
    zip_file.unlink()

# Clean extracted folders
KEEP_FILES = {
    "specification_results.csv",
    "inference_results.csv",
    "SPECIFICATION_SEARCH.md",
    "SPECIFICATION_SURFACE.json",
    "SPECIFICATION_SURFACE.md",
    "SPEC_SURFACE_REVIEW.md",
    "diagnostics_results.csv",
    "spec_diagnostics_map.csv",
}

for package_dir in (BASE_DIR / "extracted").iterdir():
    if not package_dir.is_dir():
        continue

    # Find and remove non-result files
    for file_path in package_dir.rglob("*"):
        if file_path.is_file() and file_path.name not in KEEP_FILES:
            print(f"Removing: {file_path}")
            file_path.unlink()

    # Remove empty directories
    for dir_path in sorted(package_dir.rglob("*"), reverse=True):
        if dir_path.is_dir() and not any(dir_path.iterdir()):
            dir_path.rmdir()
```

## Space Savings

Typical savings per package:
- ZIP file: 1MB - 10GB (varies widely)
- Extracted data/code: Similar to ZIP size

After cleanup, each package folder contains only:
- `specification_results.csv`: ~10-500 KB
- `inference_results.csv`: ~10-500 KB (if present)
- `SPECIFICATION_SEARCH.md`: ~5-30 KB
- `SPECIFICATION_SURFACE.json`: typically small (KB)
- `SPECIFICATION_SURFACE.md`: typically small (KB)
- `SPEC_SURFACE_REVIEW.md`: typically small (KB; if present)
- `diagnostics_results.csv`, `spec_diagnostics_map.csv`: typically small (KB–MB; if present)

## When to Run Cleanup

Run cleanup after:
1. All specification searches are complete
2. Results have been verified (check unified_results.csv)
3. Any desired manual inspection is done

## Preserving Original Data

If you need to re-run specification searches later:
- Re-download packages using the download agent
- Or keep a backup of ZIP files in a separate location before cleanup

## Files to Always Preserve

Never delete:
- `unified_results.csv` - Combined results from all papers
- `data/tracking/*.json` - Status tracking files
- `data/metadata/*.jsonl` - Package metadata
- `data/cache/datacite_responses.db` - API response cache
