#!/bin/bash
# Cleanup script to run after specification search completes
# Usage: cleanup_after_spec_search.sh <paper_id>

PAPER_ID=$1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PACKAGE_DIR="$BASE_DIR/data/downloads/extracted/$PAPER_ID"

if [ -z "$PAPER_ID" ]; then
    echo "Usage: $0 <paper_id>"
    exit 1
fi

if [ ! -d "$PACKAGE_DIR" ]; then
    echo "Package directory not found: $PACKAGE_DIR"
    exit 1
fi

# Files to preserve inside the extracted package dir
KEEP_FILES=(
    "specification_results.csv"
    "SPECIFICATION_SEARCH.md"
    "SPECIFICATION_SURFACE.json"
    "SPECIFICATION_SURFACE.md"
    "SPEC_SURFACE_REVIEW.md"
    "diagnostics_results.csv"
    "spec_diagnostics_map.csv"
)

# Check if results exist before cleanup
if [ -f "$PACKAGE_DIR/specification_results.csv" ] && [ -f "$PACKAGE_DIR/SPECIFICATION_SEARCH.md" ]; then
    # Move results to a safe location temporarily
    mkdir -p /tmp/spec_search_backup_$PAPER_ID
    for f in "${KEEP_FILES[@]}"; do
        if [ -f "$PACKAGE_DIR/$f" ]; then
            cp "$PACKAGE_DIR/$f" /tmp/spec_search_backup_$PAPER_ID/
        fi
    done
    
    # Delete everything in the package directory
    rm -rf "$PACKAGE_DIR"/*
    
    # Restore the results
    for f in "${KEEP_FILES[@]}"; do
        if [ -f "/tmp/spec_search_backup_$PAPER_ID/$f" ]; then
            mv "/tmp/spec_search_backup_$PAPER_ID/$f" "$PACKAGE_DIR/"
        fi
    done
    rm -rf /tmp/spec_search_backup_$PAPER_ID
    
    echo "Cleaned up $PAPER_ID - kept core outputs (results + surface + optional diagnostics)"
else
    echo "Results not found in $PACKAGE_DIR - skipping cleanup"
    exit 1
fi
