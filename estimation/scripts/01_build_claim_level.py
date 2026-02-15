#!/usr/bin/env python3
"""
01_build_claim_level.py
=======================

Build claim-level dataset for Sample A (40 i4r papers).
Extracts the baseline t-statistic for each paper's canonical claim.

Output: estimation/data/claim_level.csv
"""

from __future__ import annotations

import json
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "estimation" / "data"
STATUS_FILE = BASE_DIR / "data" / "tracking" / "spec_search_status.json"
UNIFIED_RESULTS = BASE_DIR / "unified_results.csv"
I4R_CLAIM_MAP_FILE = DATA_DIR / "i4r_claim_map.csv"

OUTPUT_FILE = DATA_DIR / "claim_level.csv"


def load_i4r_papers():
    """Load list of i4r papers from status file."""
    if not STATUS_FILE.exists():
        # Fallback: use the i4r claim map (built in the data stage) as the paper list.
        if I4R_CLAIM_MAP_FILE.exists():
            try:
                cm = pd.read_csv(I4R_CLAIM_MAP_FILE)
                ids = sorted(set(cm["paper_id"].astype(str).tolist())) if "paper_id" in cm.columns else []
            except Exception:
                ids = []
            print(f"  Warning: missing status file {STATUS_FILE}; using i4r_claim_map.csv with {len(ids)} paper_ids.")
            return pd.DataFrame(
                [
                    {"paper_id": pid, "title": pid, "journal": "", "year": "", "method": "", "n_specs": 0}
                    for pid in ids
                ]
            )

        print(f"  Warning: missing status file {STATUS_FILE}; no i4r paper list available.")
        return pd.DataFrame(columns=["paper_id", "title", "journal", "year", "method", "n_specs"])

    with open(STATUS_FILE, 'r') as f:
        status = json.load(f)

    i4r_papers = []
    for paper in status.get('packages_with_data', []):
        if paper.get('i4r', False):
            i4r_papers.append({
                'paper_id': paper['id'],
                'title': paper.get('title', paper['id']),
                'journal': paper.get('journal', ''),
                'year': paper.get('year', ''),
                'method': paper.get('method', ''),
                'n_specs': paper.get('n_specs', 0)
            })

    return pd.DataFrame(i4r_papers)


def _orientation_sign_by_paper(df_unified: pd.DataFrame) -> dict[str, int]:
    """Infer sign orientation per paper from the within-paper median t-statistic."""
    df = df_unified[['paper_id', 'coefficient', 'std_error']].copy()
    df['t_stat'] = df['coefficient'] / df['std_error']
    df = df[df['t_stat'].notna() & np.isfinite(df['t_stat'])].copy()
    median_t = df.groupby('paper_id')['t_stat'].median()
    # If median is exactly 0, default to +1.
    return {pid: int(1 if t >= 0 else -1) for pid, t in median_t.items()}


def _expected_sign_to_int(s: str | None) -> int:
    s = (s or "").strip().lower()
    if s.startswith("-") or "negative" in s:
        return -1
    if s.startswith("+") or "positive" in s:
        return 1
    return 0


def extract_baseline_tstat(df_unified, paper_id, orientation_sign_map=None, claim_map_row: dict | None = None):
    """
    Extract the baseline t-statistic for a paper's canonical claim.

    The baseline specification is identified by spec_id='baseline'.
    Returns the t-statistic (coefficient / std_error).
    """
    paper_df = df_unified[df_unified['paper_id'] == paper_id]

    if paper_df.empty:
        return None

    selection_rule = None
    baseline = pd.DataFrame()

    # Preferred: explicit per-paper mapping for i4r canonical claim
    if claim_map_row is not None:
        spec_id = str(claim_map_row.get("map_spec_id", "")).strip()
        outcome_var = str(claim_map_row.get("map_outcome_var", "")).strip()
        treatment_var = str(claim_map_row.get("map_treatment_var", "")).strip()

        if spec_id:
            baseline = paper_df[paper_df["spec_id"].astype(str) == spec_id]
            if outcome_var:
                baseline = baseline[baseline["outcome_var"].astype(str) == outcome_var]
            if treatment_var:
                baseline = baseline[baseline["treatment_var"].astype(str) == treatment_var]

            if not baseline.empty:
                selection_rule = f"i4r_claim_map: {spec_id} ({outcome_var} ~ {treatment_var})"

    # Fallbacks (heuristics)
    if baseline.empty:
        # Prefer explicit baseline spec_id
        baseline = paper_df[paper_df['spec_id'] == 'baseline']
        if not baseline.empty:
            selection_rule = "spec_id==baseline"
        else:
            # Next: any method-level row tagged as baseline in the tree path
            baseline = paper_df[paper_df['spec_tree_path'].astype(str).str.contains('#baseline', na=False)]
            if not baseline.empty:
                selection_rule = "spec_tree_path contains #baseline"
            else:
                # Fallback: use first specification
                baseline = paper_df.iloc[[0]]
                selection_rule = "fallback: first spec"

    # If multiple baseline-like rows exist, take a deterministic choice.
    # Prefer successful rows if run_success is available.
    if "run_success" in baseline.columns:
        rs = pd.to_numeric(baseline["run_success"], errors="coerce").fillna(0).astype(int)
        baseline = baseline.loc[rs == 1].copy()

    # Require a finite focal estimate and standard error.
    baseline["__coef"] = pd.to_numeric(baseline.get("coefficient"), errors="coerce")
    baseline["__se"] = pd.to_numeric(baseline.get("std_error"), errors="coerce")
    baseline = baseline[baseline["__coef"].notna() & baseline["__se"].notna() & np.isfinite(baseline["__se"]) & (baseline["__se"] > 0)].copy()
    if baseline.empty:
        return None

    baseline = baseline.sort_values('spec_id').iloc[[0]]
    baseline = baseline.drop(columns=["__coef", "__se"], errors="ignore")

    row = baseline.iloc[0]

    # Compute t-statistic
    coef = row.get('coefficient', np.nan)
    se = row.get('std_error', np.nan)

    if pd.isna(coef) or pd.isna(se) or se == 0:
        return None

    t_stat = coef / se
    orientation_sign = None
    if claim_map_row is not None:
        es = _expected_sign_to_int(str(claim_map_row.get("map_expected_sign", "")))
        if es in (-1, 1):
            orientation_sign = es

    if orientation_sign is None and orientation_sign_map is not None:
        orientation_sign = int(orientation_sign_map.get(paper_id, 1))

    if orientation_sign is None:
        orientation_sign = int(1 if t_stat >= 0 else -1)

    # If we don't have an expected sign (verification baselines), the median-sign
    # heuristic can misorient papers where the canonical claim is negative but many
    # other outcomes are positive. In those cases, flip orientation so the selected
    # canonical estimate is positive on the "evidence" scale.
    if claim_map_row is not None:
        es = _expected_sign_to_int(str(claim_map_row.get("map_expected_sign", "")))
        map_source = str(claim_map_row.get("map_source", "")).strip()
        if es == 0 and map_source == "heuristic_baseline_like":
            if (t_stat * orientation_sign) < 0:
                orientation_sign = -orientation_sign
                if selection_rule:
                    selection_rule = f"{selection_rule} [auto_flip_unknown_sign]"

    return {
        'paper_id': paper_id,
        't_AI': t_stat,
        't_AI_oriented': float(t_stat * orientation_sign),
        't_AI_abs': float(abs(t_stat)),
        'orientation_sign': orientation_sign,
        'spec_id': row.get('spec_id', ''),
        'baseline_selection_rule': selection_rule,
        'coefficient': coef,
        'std_error': se,
        'p_value': row.get('p_value', np.nan),
        'n_obs': row.get('n_obs', np.nan),
        'outcome_var': row.get('outcome_var', ''),
        'treatment_var': row.get('treatment_var', ''),
        'spec_tree_path': row.get('spec_tree_path', '')
    }


def main():
    print("=" * 60)
    print("Building Claim-Level Dataset (Sample A)")
    print("=" * 60)

    # Load i4r papers
    print("\nLoading i4r paper list...")
    i4r_df = load_i4r_papers()
    print(f"Found {len(i4r_df)} i4r papers")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if len(i4r_df) == 0:
        cols = [
            'paper_id', 'title', 'journal', 'year', 'method',
            'spec_id', 'baseline_selection_rule',
            't_AI', 't_AI_oriented', 't_AI_abs', 'orientation_sign',
            'coefficient', 'std_error', 'p_value', 'n_obs',
            'outcome_var', 'treatment_var', 'spec_tree_path', 'n_specs',
            'i4r_map_source', 'i4r_map_score', 'i4r_map_group', 'i4r_map_expected_sign', 'i4r_map_needs_review'
        ]
        pd.DataFrame(columns=cols).to_csv(OUTPUT_FILE, index=False)
        print(f"\nNo i4r papers available; wrote empty {OUTPUT_FILE}")
        return

    # Load unified results
    print("\nLoading unified results...")
    df_unified = pd.read_csv(UNIFIED_RESULTS)
    print(f"Loaded {len(df_unified)} specifications")

    # Infer orientation sign per paper once (used to align directions so larger values
    # correspond to stronger support for the paper's main claim).
    print("\nInferring orientation sign per paper (median t-stat within paper)...")
    orientation_sign_map = _orientation_sign_by_paper(df_unified)

    # Load optional explicit i4r claim map
    claim_map = None
    if I4R_CLAIM_MAP_FILE.exists():
        try:
            cm = pd.read_csv(I4R_CLAIM_MAP_FILE)
            claim_map = {str(r["paper_id"]): r.to_dict() for _, r in cm.iterrows() if "paper_id" in r}
            print(f"\nLoaded i4r claim map with {len(claim_map)} entries from {I4R_CLAIM_MAP_FILE}")
        except Exception as e:
            print(f"\nWarning: failed to load i4r claim map ({I4R_CLAIM_MAP_FILE}): {e}")
            claim_map = None
    else:
        print(f"\nNo i4r claim map found at {I4R_CLAIM_MAP_FILE}; using baseline heuristics.")

    # Extract baseline t-statistics for each paper
    print("\nExtracting baseline t-statistics...")
    results = []

    for _, paper in i4r_df.iterrows():
        paper_id = paper['paper_id']
        cm_row = claim_map.get(paper_id) if claim_map is not None else None
        extracted = extract_baseline_tstat(
            df_unified,
            paper_id,
            orientation_sign_map=orientation_sign_map,
            claim_map_row=cm_row,
        )

        row = {
            'paper_id': paper_id,
            'title': paper.get('title', ''),
            'journal': paper.get('journal', ''),
            'year': paper.get('year', ''),
            'method': paper.get('method', ''),
            'n_specs': paper.get('n_specs', 0),
        }

        if extracted is None:
            print(f"  Warning: No baseline found for {paper_id}")
            row.update(
                {
                    't_AI': np.nan,
                    't_AI_oriented': np.nan,
                    't_AI_abs': np.nan,
                    'orientation_sign': np.nan,
                    'spec_id': '',
                    'baseline_selection_rule': '',
                    'coefficient': np.nan,
                    'std_error': np.nan,
                    'p_value': np.nan,
                    'n_obs': np.nan,
                    'outcome_var': '',
                    'treatment_var': '',
                    'spec_tree_path': '',
                }
            )
        else:
            row.update(extracted)

        # Carry mapping metadata (if available), even when missing baseline.
        if cm_row is not None:
            row["i4r_map_source"] = cm_row.get("map_source", "")
            row["i4r_map_score"] = cm_row.get("map_score", np.nan)
            row["i4r_map_group"] = cm_row.get("map_baseline_group_id", "")
            row["i4r_map_expected_sign"] = cm_row.get("map_expected_sign", "")
            row["i4r_map_needs_review"] = cm_row.get("needs_review", np.nan)

        results.append(row)

    # Create DataFrame
    claim_df = pd.DataFrame(results)

    # Reorder columns
    cols = ['paper_id', 'title', 'journal', 'year', 'method',
            'spec_id', 'baseline_selection_rule',
            't_AI', 't_AI_oriented', 't_AI_abs', 'orientation_sign',
            'coefficient', 'std_error', 'p_value', 'n_obs',
            'outcome_var', 'treatment_var', 'spec_tree_path', 'n_specs',
            'i4r_map_source', 'i4r_map_score', 'i4r_map_group', 'i4r_map_expected_sign', 'i4r_map_needs_review']
    claim_df = claim_df[[c for c in cols if c in claim_df.columns]]

    # Save
    print(f"\nSaving {len(claim_df)} claims to {OUTPUT_FILE}")
    claim_df.to_csv(OUTPUT_FILE, index=False)

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"Total claims: {len(claim_df)}")
    print(f"Mean |t|: {claim_df['t_AI'].abs().mean():.3f}")
    print(f"Median |t|: {claim_df['t_AI'].abs().median():.3f}")
    print(f"% significant at 5%: {100 * (claim_df['p_value'] < 0.05).mean():.1f}%")
    print(f"% t > 1.96: {100 * (claim_df['t_AI'].abs() > 1.96).mean():.1f}%")
    if 't_AI_oriented' in claim_df.columns:
        print(f"\nOriented evidence index (sign aligned):")
        print(f"  Mean: {claim_df['t_AI_oriented'].mean():.3f}")
        print(f"  Median: {claim_df['t_AI_oriented'].median():.3f}")
        print(f"  % > 1.96: {100 * (claim_df['t_AI_oriented'] > 1.96).mean():.1f}%")

    print("\nBy journal:")
    print(claim_df.groupby('journal')['t_AI'].agg(['count', 'mean', 'std']))

    print("\nDone!")


if __name__ == "__main__":
    main()
