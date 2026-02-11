#!/usr/bin/env python3
"""
00_summarize_verification.py
============================

Summarize outputs from per-paper verification agents.

Reads:
  agentic_specification_search/data/verification/*/verification_spec_map.csv (optional)
  agentic_specification_search/data/verification/*/verification_baselines.json (optional)
  agentic_specification_search/data/verification/*/VERIFICATION_REPORT.md (optional)

Writes:
  agentic_specification_search/estimation/data/verification_summary_by_paper.csv
  agentic_specification_search/estimation/data/verification_summary_by_category.csv

This is intentionally lightweight (no attempt to parse full narrative content),
but it extracts a few high-signal flags from the reports to aid filtering.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


def _extract_flags(report_text: str) -> list[str]:
    flags: list[str] = []
    t = report_text

    def has(pat: str) -> bool:
        return re.search(pat, t, flags=re.IGNORECASE | re.MULTILINE) is not None

    if has(r"\bsynthetic\b"):
        flags.append("synthetic_specs")
    if has(r"\bsimulated\b"):
        flags.append("simulated_data")
    if has(r"cannot be properly replicated|cannot test the paper|restricted data|confidential"):
        flags.append("data_limitation")
    if has(r"wrong sign"):
        flags.append("wrong_sign")
    if has(r"p-?value[^\n]*exceed"):
        flags.append("pvalue_gt_1")
    if has(r"only\s+2\s+clusters|two\s+clusters|degenerate\s+cluster"):
        flags.append("degenerate_cluster")

    # Deduplicate while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for f in flags:
        if f in seen:
            continue
        seen.add(f)
        uniq.append(f)
    return uniq


def main() -> None:
    base_dir = Path(__file__).parent.parent.parent  # agentic_specification_search
    verification_dir = base_dir / "data" / "verification"
    out_dir = base_dir / "estimation" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    cat_rows: list[pd.DataFrame] = []

    paper_dirs = sorted([p for p in verification_dir.glob("*") if p.is_dir()])
    if not paper_dirs:
        raise FileNotFoundError(f"No verification directories found in {verification_dir}")

    for paper_dir in paper_dirs:
        paper_id = paper_dir.name

        map_path = paper_dir / "verification_spec_map.csv"
        baselines_path = paper_dir / "verification_baselines.json"
        report_path = paper_dir / "VERIFICATION_REPORT.md"

        has_map = map_path.exists()
        has_baselines = baselines_path.exists()
        has_report = report_path.exists()

        # ---------------------------------------------------------------------
        # Spec-map stats (if present)
        # ---------------------------------------------------------------------
        total = None
        core = None
        invalid = None
        core_share = None

        if has_map:
            try:
                df = pd.read_csv(map_path)
            except Exception as e:
                df = pd.DataFrame()
                has_map = False
                map_error = str(e)
            else:
                map_error = ""
                total = int(len(df))
                core = int(df["is_core_test"].sum()) if "is_core_test" in df.columns else 0
                invalid = int((df["category"] == "invalid").sum()) if "category" in df.columns else 0
                core_share = float(core / total) if total > 0 else float("nan")

                if "category" in df.columns:
                    vc = df["category"].value_counts(dropna=False).reset_index()
                    vc.columns = ["category", "count"]
                    vc.insert(0, "paper_id", paper_id)
                    cat_rows.append(vc)
        else:
            map_error = ""

        # ---------------------------------------------------------------------
        # Baselines JSON stats (if present)
        # ---------------------------------------------------------------------
        baselines_ok = False
        n_baseline_groups = None
        n_baseline_specs = None
        baselines_error = ""
        if has_baselines:
            try:
                raw = baselines_path.read_text(encoding="utf-8", errors="replace").strip()
                d = json.loads(raw) if raw else {}
                groups = d.get("baseline_groups", []) or []
                baselines_ok = True
                n_baseline_groups = int(len(groups))
                spec_run_ids: list[str] = []
                spec_ids: list[str] = []
                for g in groups:
                    spec_run_ids.extend(list(g.get("baseline_spec_run_ids", []) or []))
                    spec_ids.extend(list(g.get("baseline_spec_ids", []) or []))

                if spec_run_ids:
                    n_baseline_specs = int(len(set(map(str, spec_run_ids))))
                else:
                    n_baseline_specs = int(len(set(map(str, spec_ids)))) if spec_ids else 0
            except Exception as e:
                baselines_error = str(e)
        else:
            n_baseline_groups = 0
            n_baseline_specs = 0

        # ---------------------------------------------------------------------
        # Narrative flags (report) + structural flags (missing/empty artifacts)
        # ---------------------------------------------------------------------
        flags: list[str] = []
        if has_report:
            flags = _extract_flags(report_path.read_text(encoding="utf-8", errors="replace"))

        if not has_map:
            flags.append("missing_spec_map")
        if not has_report:
            flags.append("missing_report")
        if not has_baselines:
            flags.append("missing_baselines")
        if has_baselines and baselines_ok and (n_baseline_groups == 0):
            flags.append("empty_baselines")
        if has_baselines and (not baselines_ok):
            flags.append("baselines_parse_error")
        if has_map and map_error:
            flags.append("spec_map_read_error")

        # Deduplicate flags while preserving order
        seen: set[str] = set()
        uniq_flags: list[str] = []
        for f in flags:
            if f in seen:
                continue
            seen.add(f)
            uniq_flags.append(f)

        rows.append(
            {
                "paper_id": paper_id,
                "has_spec_map": bool(has_map),
                "has_baselines_json": bool(has_baselines),
                "has_report": bool(has_report),
                "baselines_ok": bool(baselines_ok) if has_baselines else False,
                "baseline_groups": n_baseline_groups,
                "baseline_specs": n_baseline_specs,
                "total_specs": total,
                "core_specs": core,
                "noncore_specs": (int(total - core) if (total is not None and core is not None) else None),
                "invalid_specs": invalid,
                "core_share": core_share,
                "flags": "|".join(uniq_flags) if uniq_flags else "",
                "baselines_error": baselines_error,
                "spec_map_error": map_error,
            }
        )

    by_paper = pd.DataFrame(rows)
    # Prefer sorting valid core_share first; push NaNs to bottom.
    by_paper["core_share_sort"] = by_paper["core_share"].fillna(-1.0)
    by_paper = by_paper.sort_values(
        ["core_share_sort", "invalid_specs", "total_specs"],
        ascending=[True, False, False],
    ).drop(columns=["core_share_sort"])
    by_cat = pd.concat(cat_rows, ignore_index=True) if cat_rows else pd.DataFrame(columns=["paper_id", "category", "count"])

    out_paper = out_dir / "verification_summary_by_paper.csv"
    out_cat = out_dir / "verification_summary_by_category.csv"
    by_paper.to_csv(out_paper, index=False)
    by_cat.to_csv(out_cat, index=False)

    print(f"Wrote: {out_paper}")
    print(f"Wrote: {out_cat}")

    # Small console summary
    print("\nOverall:")
    print(f"  papers (dirs): {by_paper['paper_id'].nunique()}")
    print(f"  with spec_map: {int(by_paper['has_spec_map'].sum())}")
    print(f"  with baselines.json: {int(by_paper['has_baselines_json'].sum())}")
    print(f"  with report: {int(by_paper['has_report'].sum())}")

    # Only aggregate where map exists
    with_map = by_paper[by_paper["has_spec_map"]].copy()
    total_specs = int(with_map["total_specs"].fillna(0).sum())
    core_specs = int(with_map["core_specs"].fillna(0).sum())
    invalid_specs = int(with_map["invalid_specs"].fillna(0).sum())
    core_share = core_specs / max(total_specs, 1)
    print(f"  total specs (mapped): {total_specs}")
    print(f"  core share (mapped): {core_share:.3f}")
    print(f"  invalid specs (mapped): {invalid_specs}")

    print("\nLowest core-share papers:")
    print(with_map.sort_values(["core_share", "invalid_specs", "total_specs"], ascending=[True, False, False]).head(10).to_string(index=False))

    flagged = by_paper[by_paper["flags"].astype(str).str.len() > 0].copy()
    if len(flagged) > 0:
        print("\nFlagged papers:")
        print(flagged.sort_values(["flags", "core_share"]).to_string(index=False))


if __name__ == "__main__":
    main()
