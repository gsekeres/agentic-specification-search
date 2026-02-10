#!/usr/bin/env python3
"""
02_build_spec_level.py
======================

Build specification-level dataset for Sample B (all papers with spec searches).
Extracts all specifications with tree metadata for dependence estimation.

Output: estimation/data/spec_level.csv
"""

import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "estimation" / "data"
STATUS_FILE = BASE_DIR / "data" / "tracking" / "spec_search_status.json"
UNIFIED_RESULTS = BASE_DIR / "unified_results.csv"

OUTPUT_FILE = DATA_DIR / "spec_level.csv"
OUTPUT_FILE_VERIFIED = DATA_DIR / "spec_level_verified.csv"
OUTPUT_FILE_VERIFIED_CORE = DATA_DIR / "spec_level_verified_core.csv"

VERIFICATION_DIR = BASE_DIR / "data" / "verification"


def _load_verification_maps() -> pd.DataFrame:
    """
    Load per-paper verification maps if present.

    Important: spec_id is not always unique within a paper (some scripts reuse
    spec_id across outcomes). To avoid many-to-many merges, we merge on
    (paper_id, spec_id, outcome_var, treatment_var).
    """
    if not VERIFICATION_DIR.exists():
        return pd.DataFrame()

    csvs = sorted(VERIFICATION_DIR.glob("*/verification_spec_map.csv"))
    if not csvs:
        return pd.DataFrame()

    dfs: list[pd.DataFrame] = []
    for p in csvs:
        try:
            d = pd.read_csv(p)
        except Exception as e:
            print(f"  Warning: failed to read verification map {p}: {e}")
            continue
        dfs.append(d)

    if not dfs:
        return pd.DataFrame()

    v = pd.concat(dfs, ignore_index=True)

    # Keep only merge keys + fields needed downstream; prefix to avoid collisions.
    keep = [
        "paper_id",
        "spec_id",
        "outcome_var",
        "treatment_var",
        "baseline_group_id",
        "closest_baseline_spec_id",
        "is_baseline",
        "is_core_test",
        "category",
        "why",
        "confidence",
    ]
    missing = [c for c in keep if c not in v.columns]
    if missing:
        raise ValueError(f"Verification map missing required columns: {missing}")

    v = v[keep].copy()
    v = v.rename(
        columns={
            "baseline_group_id": "v_baseline_group_id",
            "closest_baseline_spec_id": "v_closest_baseline_spec_id",
            "is_baseline": "v_is_baseline",
            "is_core_test": "v_is_core_test",
            "category": "v_category",
            "why": "v_why",
            "confidence": "v_confidence",
        }
    )

    return v


def _load_expected_sign_map() -> dict[tuple[str, str], str]:
    """Map (paper_id, baseline_group_id) -> expected_sign from verification_baselines.json."""
    if not VERIFICATION_DIR.exists():
        return {}

    out: dict[tuple[str, str], str] = {}
    for p in sorted(VERIFICATION_DIR.glob("*/verification_baselines.json")):
        paper_id = p.parent.name
        try:
            d = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        except Exception as e:
            print(f"  Warning: failed to read {p}: {e}")
            continue
        for g in d.get("baseline_groups", []) or []:
            gid = g.get("baseline_group_id")
            if not gid:
                continue
            out[(paper_id, str(gid))] = str(g.get("expected_sign", "")).strip()
    return out


def _expected_sign_to_int(s: str) -> int:
    s = (s or "").strip().lower()
    if s.startswith("-") or "negative" in s:
        return -1
    if s.startswith("+") or "positive" in s:
        return 1
    # For "0"/"null"/"unknown", fall back elsewhere.
    return 0


def parse_tree_path(spec_tree_path):
    """
    Parse specification tree path into components for distance calculation.

    Example paths:
        "methods/cross_sectional_ols.md"
        "methods/difference_in_differences.md#fixed-effects"
        "robustness/leave_one_out.md"
        "robustness/clustering_variations.md"

    Returns:
        dict with tree_depth, branch components
    """
    if pd.isna(spec_tree_path) or not spec_tree_path:
        return {'tree_depth': 0, 'branch_0': 'unknown', 'branch_1': '', 'branch_2': ''}

    # Remove .md extension and fragment
    path = spec_tree_path.replace('.md', '')
    if '#' in path:
        path, fragment = path.split('#', 1)
    else:
        fragment = ''

    # Split into components
    parts = path.split('/')

    return {
        'tree_depth': len(parts) + (1 if fragment else 0),
        'branch_0': parts[0] if len(parts) > 0 else '',
        'branch_1': parts[1] if len(parts) > 1 else '',
        'branch_2': fragment if fragment else (parts[2] if len(parts) > 2 else '')
    }


def compute_tree_distance(path1, path2):
    """
    Compute tree distance between two specifications.

    Distance = number of non-matching branch levels.
    """
    p1 = parse_tree_path(path1)
    p2 = parse_tree_path(path2)

    distance = 0
    for key in ['branch_0', 'branch_1', 'branch_2']:
        if p1[key] != p2[key]:
            distance += 1

    return distance


def load_paper_metadata():
    """Load paper metadata from status file."""
    with open(STATUS_FILE, 'r') as f:
        status = json.load(f)

    metadata = {}
    for paper in status.get('packages_with_data', []):
        metadata[paper['id']] = {
            'title': paper.get('title', paper['id']),
            'journal': paper.get('journal', ''),
            'year': paper.get('year', ''),
            'method': paper.get('method', ''),
            'i4r': paper.get('i4r', False)
        }

    return metadata


def main():
    print("=" * 60)
    print("Building Specification-Level Dataset (Sample B)")
    print("=" * 60)

    # Load unified results
    print("\nLoading unified results...")
    df = pd.read_csv(UNIFIED_RESULTS)
    print(f"Loaded {len(df)} specifications")

    # Load paper metadata
    print("\nLoading paper metadata...")
    metadata = load_paper_metadata()

    # Compute t-statistic (evidence index Z)
    print("\nComputing evidence index Z...")
    df['t_stat'] = df['coefficient'] / df['std_error']

    # Filter out invalid t-statistics (inf, nan, extremely large values)
    valid_mask = (
        df['t_stat'].notna() &
        np.isfinite(df['t_stat']) &
        (df['t_stat'].abs() < 20)  # Remove extreme outliers (|t| > 20 is unrealistic)
    )
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        print(f"  Filtering out {n_invalid} invalid t-statistics")
    df = df[valid_mask].copy()

    # Orient sign so that larger Z corresponds to stronger support for the paper's
    # main claim. We approximate the direction of the claim by the within-paper
    # median t-statistic sign (robust to outliers).
    paper_median_t = df.groupby('paper_id')['t_stat'].median()
    orientation_sign = paper_median_t.apply(lambda x: 1 if x >= 0 else -1).to_dict()
    df['orientation_sign'] = df['paper_id'].map(orientation_sign).fillna(1).astype(int)
    df['Z_raw'] = df['t_stat']
    df['Z'] = df['t_stat'] * df['orientation_sign']
    df['Z_abs'] = df['Z'].abs()

    # ---------------------------------------------------------------------
    # Evidence index alternative: Z_logp = -log10(p)
    #
    # Goal: nonnegative index where 0 is "null" (p=1) and larger values are
    # more significant. We use the reported p_value when it is valid, and
    # fall back to a stable normal approximation from t_stat otherwise.
    # ---------------------------------------------------------------------
    p_raw = pd.to_numeric(df.get("p_value"), errors="coerce")
    t_abs = np.abs(pd.to_numeric(df["t_stat"], errors="coerce").to_numpy(dtype=float))
    p_norm = 2.0 * norm.sf(t_abs)
    p_eff = p_norm.copy()
    if p_raw is not None:
        p_raw_np = p_raw.to_numpy(dtype=float)
        ok = np.isfinite(p_raw_np) & (p_raw_np > 0.0) & (p_raw_np <= 1.0)
        p_eff[ok] = p_raw_np[ok]
    p_eff = np.clip(p_eff, 1e-300, 1.0)
    df["p_value_eff"] = p_eff
    df["Z_logp"] = -np.log10(p_eff)

    # Merge verification maps (if available) to identify which specs are genuine
    # core tests of a baseline claim and to assign baseline groups.
    print("\nMerging verification maps (if available)...")
    vmap = _load_verification_maps()
    if len(vmap) == 0:
        print("  No verification maps found; skipping.")
        df['v_has_map'] = False
    else:
        print(f"  Loaded verification maps for {vmap['paper_id'].nunique()} papers")

        # Left join on (paper_id, spec_id, outcome_var, treatment_var) to avoid
        # many-to-many blowups when spec_id is reused across outcomes.
        merge_keys = ["paper_id", "spec_id", "outcome_var", "treatment_var"]
        for k in merge_keys:
            if k not in df.columns:
                raise ValueError(f"unified_results missing merge key: {k}")

        # Deduplicate verification map on merge keys (keep first)
        n_before_dedup = len(vmap)
        vmap = vmap.drop_duplicates(subset=merge_keys, keep="first")
        n_after_dedup = len(vmap)
        if n_before_dedup != n_after_dedup:
            print(f"  Deduplicated verification map: {n_before_dedup} → {n_after_dedup} rows")

        before = len(df)
        df = df.merge(vmap, how="left", on=merge_keys, validate="m:1", indicator=True)
        after = len(df)
        if after != before:
            raise RuntimeError("Verification merge changed row count; merge keys are not unique in verification map.")
        df['v_has_map'] = df["_merge"].eq("both")
        df = df.drop(columns=["_merge"])

        # Attach expected sign at the baseline-group level.
        expected_sign_map = _load_expected_sign_map()
        if expected_sign_map:
            df["v_expected_sign"] = df.apply(
                lambda r: expected_sign_map.get((r["paper_id"], str(r.get("v_baseline_group_id", ""))), ""),
                axis=1,
            )
        else:
            df["v_expected_sign"] = ""

        # Orientation at (paper, baseline group) level: use expected_sign if given,
        # otherwise fall back to median sign within the baseline group.
        group_med = (
            df[df["v_baseline_group_id"].notna()]
            .groupby(["paper_id", "v_baseline_group_id"])["t_stat"]
            .median()
        )

        def _orient_row(r):
            exp = _expected_sign_to_int(str(r.get("v_expected_sign", "")))
            if exp in (-1, 1):
                return exp
            pid = r["paper_id"]
            gid = r.get("v_baseline_group_id")
            if pd.isna(gid):
                return int(r.get("orientation_sign", 1))
            med = group_med.get((pid, gid), np.nan)
            if pd.isna(med):
                return int(r.get("orientation_sign", 1))
            return int(1 if med >= 0 else -1)

        df["orientation_sign_vgroup"] = df.apply(_orient_row, axis=1).astype(int)
        df["Z_vgroup"] = df["t_stat"] * df["orientation_sign_vgroup"]

    # Parse tree paths
    print("\nParsing tree paths...")
    tree_info = df['spec_tree_path'].apply(parse_tree_path)
    df['tree_depth'] = tree_info.apply(lambda x: x['tree_depth'])
    df['branch_0'] = tree_info.apply(lambda x: x['branch_0'])
    df['branch_1'] = tree_info.apply(lambda x: x['branch_1'])
    df['branch_2'] = tree_info.apply(lambda x: x['branch_2'])

    # Add paper metadata
    print("\nMerging paper metadata...")
    df['title'] = df['paper_id'].map(lambda x: metadata.get(x, {}).get('title', ''))
    df['year'] = df['paper_id'].map(lambda x: metadata.get(x, {}).get('year', ''))
    df['method'] = df['paper_id'].map(lambda x: metadata.get(x, {}).get('method', ''))
    df['i4r'] = df['paper_id'].map(lambda x: metadata.get(x, {}).get('i4r', False))

    # Create within-paper ordering (for AR(1) estimation)
    print("\nCreating specification ordering...")
    df = df.sort_values(['paper_id', 'tree_depth', 'branch_0', 'branch_1', 'spec_id'])
    df['spec_order'] = df.groupby('paper_id').cumcount()

    # Select columns
    cols = [
        'paper_id', 'journal', 'title', 'year', 'method', 'i4r',
        'spec_id', 'spec_tree_path', 'tree_depth', 'branch_0', 'branch_1', 'branch_2',
        'spec_order', 'orientation_sign', 'Z', 'Z_raw', 'Z_abs', 'Z_logp', 't_stat', 'coefficient', 'std_error', 'p_value', 'p_value_eff',
        'n_obs', 'r_squared', 'outcome_var', 'treatment_var',
        'fixed_effects', 'controls_desc', 'cluster_var',
        # Verification-derived fields (present only for verified papers)
        'v_has_map', 'v_baseline_group_id', 'v_closest_baseline_spec_id',
        'v_is_baseline', 'v_is_core_test', 'v_category', 'v_why', 'v_confidence',
        'v_expected_sign', 'orientation_sign_vgroup', 'Z_vgroup'
    ]
    df = df[[c for c in cols if c in df.columns]]

    # Save
    print(f"\nSaving {len(df)} specifications to {OUTPUT_FILE}")
    df.to_csv(OUTPUT_FILE, index=False)

    # Save verification-augmented outputs if available
    if "v_has_map" in df.columns and df["v_has_map"].any():
        print(f"\nSaving verification-augmented dataset to {OUTPUT_FILE_VERIFIED}")
        df.to_csv(OUTPUT_FILE_VERIFIED, index=False)

        core_mask = (df["v_has_map"]) & (df["v_is_core_test"] == 1) & (df["v_category"] != "invalid")
        core_df = df[core_mask].copy()
        print(f"Saving verified-core subset ({len(core_df)} rows) to {OUTPUT_FILE_VERIFIED_CORE}")
        core_df.to_csv(OUTPUT_FILE_VERIFIED_CORE, index=False)

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    n_papers = df['paper_id'].nunique()
    print(f"Total specifications: {len(df)}")
    print(f"Total papers: {n_papers}")
    print(f"Mean specs per paper: {len(df) / n_papers:.1f}")

    print(f"\nMean |Z|: {df['Z'].abs().mean():.3f}")
    print(f"Median |Z|: {df['Z'].abs().median():.3f}")
    print(f"SD(Z): {df['Z'].std():.3f}")

    print(f"\n% significant at 5%: {100 * (df['p_value'] < 0.05).mean():.1f}%")
    print(f"% |Z| > 1.96: {100 * (df['Z'].abs() > 1.96).mean():.1f}%")

    print("\nBy sample:")
    print(f"  i4r papers (Sample A): {df[df['i4r']]['paper_id'].nunique()} papers, "
          f"{len(df[df['i4r']])} specs")
    print(f"  Other papers (Sample B extension): {df[~df['i4r']]['paper_id'].nunique()} papers, "
          f"{len(df[~df['i4r']])} specs")

    print("\nTree depth distribution:")
    print(df.groupby('tree_depth')['Z'].agg(['count', 'mean', 'std']))

    print("\nDone!")


if __name__ == "__main__":
    main()
