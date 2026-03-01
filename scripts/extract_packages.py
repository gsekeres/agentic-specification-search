"""
Extract downloaded openICPSR ZIP packages.

Usage:
    python scripts/extract_packages.py                  # extract all unextracted
    python scripts/extract_packages.py --paper-id 112431-V1
    python scripts/extract_packages.py --force          # re-extract all
"""

import argparse
import zipfile
from pathlib import Path

from download_utils import (
    EXTRACTED_DIR,
    RAW_PACKAGES_DIR,
    is_valid_zip,
    load_tracking,
)

# Files/dirs to skip during extraction
SKIP_PATTERNS = {"__MACOSX", ".DS_Store", "Thumbs.db"}


def should_skip(name: str) -> bool:
    """Check if a ZIP member should be skipped."""
    parts = Path(name).parts
    return any(p in SKIP_PATTERNS for p in parts)


def extract_package(zip_path: Path, dest_dir: Path, force: bool = False) -> bool:
    """
    Extract a ZIP package to dest_dir.

    If the ZIP has a single top-level directory, flatten it so contents
    go directly into dest_dir rather than dest_dir/subdir/.
    """
    if dest_dir.exists() and not force:
        # Check if already extracted (has files beyond .gitkeep)
        contents = [p for p in dest_dir.iterdir() if p.name != ".gitkeep"]
        if contents:
            print(f"  Already extracted: {dest_dir.name} (use --force to re-extract)")
            return True

    if not is_valid_zip(zip_path):
        print(f"  ERROR: Invalid ZIP: {zip_path}")
        return False

    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Determine if there's a single top-level directory
            members = [m for m in zf.namelist() if not should_skip(m)]
            top_dirs = set()
            for m in members:
                parts = Path(m).parts
                if parts:
                    top_dirs.add(parts[0])

            strip_prefix = ""
            if len(top_dirs) == 1:
                sole_top = top_dirs.pop()
                # Check it's actually a directory (not a single file)
                if any("/" in m or "\\" in m for m in members):
                    strip_prefix = sole_top + "/"

            extracted_count = 0
            for member in zf.infolist():
                name = member.filename
                if should_skip(name):
                    continue
                if member.is_dir():
                    continue

                # Strip single top-level directory if present
                rel_name = name
                if strip_prefix and rel_name.startswith(strip_prefix):
                    rel_name = rel_name[len(strip_prefix):]

                if not rel_name:
                    continue

                target = dest_dir / rel_name
                target.parent.mkdir(parents=True, exist_ok=True)

                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                extracted_count += 1

            print(f"  Extracted {extracted_count} files -> {dest_dir}")
            return True

    except (zipfile.BadZipFile, OSError) as e:
        print(f"  ERROR extracting {zip_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Extract openICPSR ZIP packages")
    parser.add_argument("--paper-id", help="Extract specific paper (e.g., 112431-V1)")
    parser.add_argument("--force", action="store_true", help="Re-extract even if already done")
    args = parser.parse_args()

    RAW_PACKAGES_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    if args.paper_id:
        # Extract a specific package
        zip_path = RAW_PACKAGES_DIR / f"{args.paper_id}.zip"
        if not zip_path.exists():
            print(f"ERROR: ZIP not found: {zip_path}")
            return
        dest = EXTRACTED_DIR / args.paper_id
        extract_package(zip_path, dest, force=args.force)
        return

    # Extract all ZIPs in raw_packages that haven't been extracted yet
    zips = sorted(RAW_PACKAGES_DIR.glob("*.zip"))
    if not zips:
        print("No ZIP files found in raw_packages/")
        return

    print(f"Found {len(zips)} ZIP files")
    success = 0
    for zip_path in zips:
        paper_id = zip_path.stem
        dest = EXTRACTED_DIR / paper_id
        if extract_package(zip_path, dest, force=args.force):
            success += 1

    print(f"\nExtracted {success}/{len(zips)} packages")


if __name__ == "__main__":
    main()
