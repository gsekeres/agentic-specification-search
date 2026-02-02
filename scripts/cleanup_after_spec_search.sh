#!/bin/bash
# Cleanup script to run after specification search completes
# Usage: cleanup_after_spec_search.sh <paper_id>

PAPER_ID=$1
BASE_DIR="/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search"
PACKAGE_DIR="$BASE_DIR/data/downloads/extracted/$PAPER_ID"

if [ -z "$PAPER_ID" ]; then
    echo "Usage: $0 <paper_id>"
    exit 1
fi

if [ ! -d "$PACKAGE_DIR" ]; then
    echo "Package directory not found: $PACKAGE_DIR"
    exit 1
fi

# Check if results exist before cleanup
if [ -f "$PACKAGE_DIR/specification_results.csv" ] && [ -f "$PACKAGE_DIR/SPECIFICATION_SEARCH.md" ]; then
    # Move results to a safe location temporarily
    mkdir -p /tmp/spec_search_backup_$PAPER_ID
    cp "$PACKAGE_DIR/specification_results.csv" /tmp/spec_search_backup_$PAPER_ID/
    cp "$PACKAGE_DIR/SPECIFICATION_SEARCH.md" /tmp/spec_search_backup_$PAPER_ID/
    
    # Delete everything in the package directory
    rm -rf "$PACKAGE_DIR"/*
    
    # Restore the results
    mv /tmp/spec_search_backup_$PAPER_ID/specification_results.csv "$PACKAGE_DIR/"
    mv /tmp/spec_search_backup_$PAPER_ID/SPECIFICATION_SEARCH.md "$PACKAGE_DIR/"
    rmdir /tmp/spec_search_backup_$PAPER_ID
    
    echo "Cleaned up $PAPER_ID - kept only specification_results.csv and SPECIFICATION_SEARCH.md"
else
    echo "Results not found in $PACKAGE_DIR - skipping cleanup"
    exit 1
fi
