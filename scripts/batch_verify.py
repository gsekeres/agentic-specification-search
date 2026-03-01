#!/usr/bin/env python3
"""
batch_verify.py
===============
Mechanical verification for papers that haven't been through the agent verifier.
Applies the classification rules from prompts/06_post_run_verifier.md:

- is_valid: run_success == 1
- is_baseline: spec_id == "baseline" or starts with "baseline__"
- is_core_test: namespace is baseline, design/*, or rc/*
- category: determined by spec_id namespace prefix
"""

import csv
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTRACTED = ROOT / "data" / "downloads" / "extracted"
VERIFICATION = ROOT / "data" / "verification"


def classify_category(spec_id: str, run_success: bool) -> str:
    """Determine category from spec_id namespace."""
    if not run_success:
        return "invalid_failure"
    s = spec_id.lower()
    if s == "baseline" or s.startswith("baseline__"):
        return "core_method"
    if s.startswith("design/"):
        return "core_method"
    if s.startswith("rc/controls/"):
        return "core_controls"
    if s.startswith("rc/sample/"):
        return "core_sample"
    if s.startswith("rc/fe/"):
        return "core_fe"
    if s.startswith("rc/form/"):
        return "core_funcform"
    if s.startswith("rc/preprocess/"):
        return "core_preprocess"
    if s.startswith("rc/data/"):
        return "core_data"
    if s.startswith("rc/weights/"):
        return "core_weights"
    if s.startswith("rc/"):
        return "core_method"  # generic rc
    if s.startswith("infer/"):
        return "noncore_inference"
    if s.startswith("diag/") or s.startswith("sens/") or s.startswith("post/") or s.startswith("explore/"):
        return "noncore_other"
    return "core_method"


def is_core_namespace(spec_id: str) -> bool:
    s = spec_id.lower()
    if s == "baseline" or s.startswith("baseline__"):
        return True
    if s.startswith("design/") or s.startswith("rc/"):
        return True
    return False


def is_baseline_spec(spec_id: str) -> bool:
    s = spec_id.lower()
    return s == "baseline" or s.startswith("baseline__")


def verify_paper(paper_id: str) -> dict:
    """Run mechanical verification for one paper. Returns summary dict."""
    pkg_dir = EXTRACTED / paper_id
    spec_csv = pkg_dir / "specification_results.csv"
    surface_json = pkg_dir / "SPECIFICATION_SURFACE.json"
    out_dir = VERIFICATION / paper_id

    if not spec_csv.exists():
        return {"paper_id": paper_id, "status": "no_spec_results"}

    # Read specification_results.csv
    with open(spec_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {"paper_id": paper_id, "status": "empty_spec_results"}

    # Load surface for baseline group info (optional)
    surface = None
    if surface_json.exists():
        try:
            with open(surface_json) as f:
                surface = json.load(f)
        except Exception:
            pass

    # Build baseline groups from data
    baseline_groups = {}
    for row in rows:
        bgid = row.get("baseline_group_id", "G1") or "G1"
        if bgid not in baseline_groups:
            baseline_groups[bgid] = {
                "baseline_group_id": bgid,
                "claim_summary": "",
                "expected_sign": None,
                "baseline_spec_run_ids": [],
                "baseline_spec_ids": [],
                "notes": "Mechanical verification",
            }
        spec_id = row.get("spec_id", "")
        if is_baseline_spec(spec_id):
            run_success = str(row.get("run_success", "0")).strip() == "1"
            if run_success:
                baseline_groups[bgid]["baseline_spec_run_ids"].append(row.get("spec_run_id", ""))
                baseline_groups[bgid]["baseline_spec_ids"].append(spec_id)
                # Grab outcome/treatment from first baseline
                if not baseline_groups[bgid]["claim_summary"]:
                    ov = row.get("outcome_var", "")
                    tv = row.get("treatment_var", "")
                    baseline_groups[bgid]["claim_summary"] = f"Effect of {tv} on {ov}"

    # If surface has baseline group info, use it for claim summaries
    if surface and "baseline_groups" in surface:
        for sg in surface["baseline_groups"]:
            bgid = sg.get("baseline_group_id", "")
            if bgid in baseline_groups:
                if sg.get("claim_summary"):
                    baseline_groups[bgid]["claim_summary"] = sg["claim_summary"]
                if sg.get("expected_sign"):
                    baseline_groups[bgid]["expected_sign"] = sg["expected_sign"]

    # Classify each row
    spec_map_rows = []
    counts = {"total": 0, "core": 0, "noncore": 0, "invalid": 0}
    cat_counts = {}

    for row in rows:
        counts["total"] += 1
        spec_id = row.get("spec_id", "")
        run_success = str(row.get("run_success", "0")).strip() == "1"
        bgid = row.get("baseline_group_id", "G1") or "G1"

        valid = 1 if run_success else 0
        baseline = 1 if (is_baseline_spec(spec_id) and run_success) else 0
        core = 1 if (is_core_namespace(spec_id) and run_success) else 0
        category = classify_category(spec_id, run_success)

        if not run_success:
            counts["invalid"] += 1
            why = f"run_success=0; {row.get('run_error', 'unknown error')[:60]}"
        elif core:
            counts["core"] += 1
            why = "Mechanical namespace classification"
        else:
            counts["noncore"] += 1
            why = "Non-core namespace"

        cat_counts[category] = cat_counts.get(category, 0) + 1

        # Find closest baseline
        closest_bl = ""
        if bgid in baseline_groups and baseline_groups[bgid]["baseline_spec_run_ids"]:
            closest_bl = baseline_groups[bgid]["baseline_spec_run_ids"][0]

        spec_map_rows.append({
            "paper_id": paper_id,
            "spec_run_id": row.get("spec_run_id", ""),
            "spec_id": spec_id,
            "spec_tree_path": row.get("spec_tree_path", ""),
            "outcome_var": row.get("outcome_var", ""),
            "treatment_var": row.get("treatment_var", ""),
            "baseline_group_id": bgid,
            "closest_baseline_spec_run_id": closest_bl,
            "is_baseline": baseline,
            "is_valid": valid,
            "is_core_test": core,
            "category": category,
            "why": why[:80],
            "confidence": 0.90,
        })

    # Write outputs
    out_dir.mkdir(parents=True, exist_ok=True)

    # verification_baselines.json
    baselines_out = {
        "paper_id": paper_id,
        "baseline_groups": list(baseline_groups.values()),
        "global_notes": "Mechanical verification (batch_verify.py). Classification based on spec_id namespace rules.",
    }
    with open(out_dir / "verification_baselines.json", "w") as f:
        json.dump(baselines_out, f, indent=2)

    # verification_spec_map.csv
    fieldnames = [
        "paper_id", "spec_run_id", "spec_id", "spec_tree_path",
        "outcome_var", "treatment_var", "baseline_group_id",
        "closest_baseline_spec_run_id", "is_baseline", "is_valid",
        "is_core_test", "category", "why", "confidence",
    ]
    with open(out_dir / "verification_spec_map.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(spec_map_rows)

    # VERIFICATION_REPORT.md
    report_lines = [
        f"# Verification Report: {paper_id}",
        "",
        "## Method",
        "Mechanical verification via `batch_verify.py`. Classification based on spec_id namespace rules.",
        "",
        "## Baseline Groups",
    ]
    for bg in baseline_groups.values():
        report_lines.append(f"- **{bg['baseline_group_id']}**: {bg['claim_summary']}")
        report_lines.append(f"  - Baseline spec_run_ids: {bg['baseline_spec_run_ids']}")
        if bg.get("expected_sign"):
            report_lines.append(f"  - Expected sign: {bg['expected_sign']}")
    report_lines.extend([
        "",
        "## Counts",
        f"- **Total rows**: {counts['total']}",
        f"- **Core**: {counts['core']} ({100*counts['core']/max(counts['total'],1):.0f}%)",
        f"- **Non-core**: {counts['noncore']}",
        f"- **Invalid**: {counts['invalid']}",
        "",
        "## Category Breakdown",
    ])
    for cat, n in sorted(cat_counts.items()):
        report_lines.append(f"- {cat}: {n}")

    with open(out_dir / "VERIFICATION_REPORT.md", "w") as f:
        f.write("\n".join(report_lines) + "\n")

    return {
        "paper_id": paper_id,
        "status": "ok",
        "total": counts["total"],
        "core": counts["core"],
        "noncore": counts["noncore"],
        "invalid": counts["invalid"],
    }


def main():
    import pandas as pd
    # Find papers needing verification
    unified = pd.read_csv(ROOT / "unified_results.csv")
    all_pids = sorted(unified["paper_id"].unique())
    already = set(
        d for d in os.listdir(VERIFICATION)
        if os.path.isdir(VERIFICATION / d) and (VERIFICATION / d / "verification_spec_map.csv").exists()
    )
    need = [p for p in all_pids if p not in already]
    print(f"Papers needing verification: {len(need)} (already done: {len(already)})")

    results = []
    for pid in need:
        r = verify_paper(pid)
        results.append(r)
        if r["status"] == "ok":
            print(f"  {pid}: {r['total']} total, {r['core']} core, {r['invalid']} invalid")
        else:
            print(f"  {pid}: {r['status']}")

    ok = [r for r in results if r["status"] == "ok"]
    print(f"\nDone: {len(ok)}/{len(need)} papers verified successfully")
    print(f"Total core specs added: {sum(r.get('core', 0) for r in ok)}")


if __name__ == "__main__":
    main()
