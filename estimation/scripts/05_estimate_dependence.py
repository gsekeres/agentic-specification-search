#!/usr/bin/env python3
"""
05_estimate_dependence.py
=========================

Estimate dependence parameter phi (and effective independence Delta = 1 - phi).

Approach: AR(1) serial correlation along specification traversal under multiple
orderings.  The ordering with the best model fit (R²) is selected as the
preferred estimate.

Output: estimation/results/dependence.json
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "estimation" / "data"
RESULTS_DIR = BASE_DIR / "estimation" / "results"
SPEC_LEVEL_FILE = DATA_DIR / "spec_level.csv"
SPEC_LEVEL_VERIFIED_CORE_FILE = DATA_DIR / "spec_level_verified_core.csv"
OUTPUT_FILE = RESULTS_DIR / "dependence.json"

# Available orderings for AR(1) estimation
ORDERINGS = ["spec_order", "lex_path", "bfs", "dfs", "by_category", "random"]


def _sort_group(paper_df: pd.DataFrame, ordering: str, seed: int) -> pd.DataFrame:
    """Return paper_df sorted according to the given ordering."""
    if ordering == "spec_order":
        return paper_df.sort_values("spec_order")
    elif ordering == "lex_path":
        return paper_df.sort_values("spec_tree_path")
    elif ordering == "bfs":
        # Breadth-first: sort by tree_depth ascending, then spec_order
        return paper_df.sort_values(["tree_depth", "spec_order"])
    elif ordering == "dfs":
        # Depth-first: sort by tree_depth descending, then spec_order
        return paper_df.sort_values(["tree_depth", "spec_order"], ascending=[False, True])
    elif ordering == "by_category":
        if "v_category" not in paper_df.columns:
            return None  # signal to skip
        return paper_df.sort_values(["v_category", "spec_order"])
    elif ordering == "random":
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(paper_df))
        return paper_df.iloc[idx]
    else:
        raise ValueError(f"Unknown ordering: {ordering}")


def estimate_ar1_by_paper(df, z_col="Z", group_cols=("paper_id",), ordering="spec_order"):
    """
    Estimate AR(1) coefficient for each group and pool, under a given ordering.

    Z_{i,s+1} = phi * Z_{i,s} + u_{i,s+1}

    Returns dict with pooled phi, SE, CI, R², n_groups, and group-level estimates.
    """
    phi_estimates = []
    total_ss_res = 0.0
    total_ss_tot = 0.0

    group_cols = tuple(group_cols)
    groups = df[list(group_cols)].drop_duplicates()

    for idx_g, (_, g) in enumerate(groups.iterrows()):
        mask = pd.Series(True, index=df.index)
        for c in group_cols:
            mask &= df[c].eq(g[c])
        paper_df = df[mask].copy()

        # Use a per-group seed for random ordering (deterministic)
        sorted_df = _sort_group(paper_df, ordering, seed=12345 + idx_g)
        if sorted_df is None:
            continue  # ordering not applicable (e.g. by_category without v_category)
        paper_df = sorted_df

        if len(paper_df) < 3:
            continue

        z = paper_df[z_col].values
        if np.std(z) < 1e-6:
            continue

        # AR(1) regression: z[t+1] on z[t]
        z_lag = z[:-1]
        z_lead = z[1:]

        if len(z_lag) >= 2 and np.std(z_lag) > 1e-6 and np.std(z_lead) > 1e-6:
            corr, _ = stats.pearsonr(z_lag, z_lead)
            if not np.isfinite(corr):
                continue
            group_id = "|".join(str(g[c]) for c in group_cols)

            # OLS with intercept: z_lead = alpha + beta * z_lag
            beta = corr * np.std(z_lead) / np.std(z_lag)
            alpha = np.mean(z_lead) - beta * np.mean(z_lag)
            z_lead_hat = alpha + beta * z_lag
            ss_res = np.sum((z_lead - z_lead_hat) ** 2)
            ss_tot = np.sum((z_lead - np.mean(z_lead)) ** 2)

            phi_estimates.append({
                'group_id': group_id,
                'phi': corr,
                'n_specs': len(paper_df),
                'r_squared': float(corr ** 2),
            })

            total_ss_res += ss_res
            total_ss_tot += ss_tot

    phi_df = pd.DataFrame(phi_estimates)

    if len(phi_df) == 0:
        return {
            'pooled_phi': np.nan, 'pooled_phi_se': np.nan,
            'pooled_phi_ci_lower': np.nan, 'pooled_phi_ci_upper': np.nan,
            'r_squared': np.nan, 'n_groups': 0, 'group_level': [],
        }

    # Weighted average (weight by n_specs)
    weights = phi_df['n_specs'].values
    pooled_phi = np.average(phi_df['phi'].values, weights=weights)

    # Pooled R²
    r_squared = 1 - total_ss_res / total_ss_tot if total_ss_tot > 0 else np.nan

    # Bootstrap CI
    n_boot = 1000
    rng = np.random.default_rng(12345)
    phi_boots = []
    for _ in range(n_boot):
        idx = rng.choice(len(phi_df), len(phi_df), replace=True)
        w_b = phi_df.iloc[idx]['n_specs'].values
        phi_b = phi_df.iloc[idx]['phi'].values
        phi_boots.append(np.average(phi_b, weights=w_b))

    phi_se = np.std(phi_boots)
    phi_ci = np.percentile(phi_boots, [2.5, 97.5])

    return {
        'pooled_phi': float(pooled_phi),
        'pooled_phi_se': float(phi_se),
        'pooled_phi_ci_lower': float(phi_ci[0]),
        'pooled_phi_ci_upper': float(phi_ci[1]),
        'r_squared': float(r_squared),
        'n_groups': len(phi_df),
        'group_level': phi_df.to_dict('records'),
    }


def main():
    print("=" * 60)
    print("Estimating Dependence Parameter (Multi-Ordering AR(1))")
    print("=" * 60)

    def run_one(df_in: pd.DataFrame, dataset_label: str) -> dict:
        print("\n" + "=" * 60)
        print(f"Dataset: {dataset_label}")
        print("=" * 60)

        # Choose evidence index column.
        z_col = (
            "Z_abs"
            if "Z_abs" in df_in.columns
            else ("Z_logp" if "Z_logp" in df_in.columns else ("Z_vgroup" if "Z_vgroup" in df_in.columns else "Z"))
        )
        if z_col not in df_in.columns:
            raise ValueError(f"Missing required Z column in {dataset_label}: {z_col}")

        # Grouping: if verification provides baseline groups, estimate dependence within
        # (paper_id, baseline_group_id) rather than across unrelated outcomes/claims.
        group_cols = ["paper_id"]
        if dataset_label == "verified_core" and "v_baseline_group_id" in df_in.columns:
            group_cols = ["paper_id", "v_baseline_group_id"]

        df = df_in.copy()
        df = df[df[z_col].notna() & np.isfinite(df[z_col])].copy()
        print(f"Loaded {len(df)} rows; {df['paper_id'].nunique()} papers; z_col={z_col}; group_cols={group_cols}")

        out: dict = {"dataset": dataset_label, "z_col": z_col, "group_cols": group_cols}

        # =====================================================================
        # Estimate AR(1) under each ordering
        # =====================================================================
        ar1_orderings: dict = {}
        best_ordering = None
        best_r2 = -np.inf

        for ordering in ORDERINGS:
            print(f"\n--- Ordering: {ordering} ---")

            # Skip by_category if column not available
            if ordering == "by_category" and "v_category" not in df.columns:
                print("  SKIP (v_category column absent)")
                continue

            ar1 = estimate_ar1_by_paper(df, z_col=z_col, group_cols=tuple(group_cols), ordering=ordering)

            if np.isnan(ar1['pooled_phi']):
                print("  phi = NaN (skipping)")
                continue

            phi = ar1['pooled_phi']
            r2 = ar1['r_squared']
            se = ar1['pooled_phi_se']
            n_g = ar1['n_groups']

            print(f"  phi = {phi:.4f} (SE = {se:.4f}), R² = {r2:.4f}, n_groups = {n_g}")

            ar1_orderings[ordering] = {
                'phi': phi,
                'phi_se': se,
                'phi_ci_lower': ar1['pooled_phi_ci_lower'],
                'phi_ci_upper': ar1['pooled_phi_ci_upper'],
                'Delta': float(1 - phi),
                'r_squared': r2,
                'n_groups': n_g,
                'group_level': ar1['group_level'],
            }

            # Select best (excluding random)
            if ordering != "random" and r2 > best_r2:
                best_r2 = r2
                best_ordering = ordering

        out['ar1_orderings'] = ar1_orderings

        # =====================================================================
        # Preferred estimate
        # =====================================================================
        print("\n" + "-" * 40)
        print("Summary: Preferred Dependence Estimate")
        print("-" * 40)

        if best_ordering is not None:
            pref = ar1_orderings[best_ordering]
            preferred_phi = pref['phi']
            preferred_Delta = pref['Delta']

            print(f"\nPreferred ordering: {best_ordering} (R² = {best_r2:.4f})")
            print(f"phi = {preferred_phi:.4f}")
            print(f"Delta (effective independence) = {preferred_Delta:.4f}")

            out['preferred'] = {
                'method': 'ar1',
                'ordering': best_ordering,
                'phi': float(preferred_phi),
                'Delta': float(preferred_Delta),
                'r_squared': float(best_r2),
            }
        else:
            print("\nNo valid AR(1) estimates found.")
            out['preferred'] = {
                'method': 'none',
                'phi': float('nan'),
                'Delta': float('nan'),
            }

        # Cross-check table
        print(f"\nCross-check across orderings:")
        for ord_name, ord_data in ar1_orderings.items():
            marker = " <-- preferred" if ord_name == best_ordering else ""
            print(f"  {ord_name:15s}: phi={ord_data['phi']:.4f}, Delta={ord_data['Delta']:.4f}, R²={ord_data['r_squared']:.4f}{marker}")

        return out

    if not SPEC_LEVEL_FILE.exists():
        print("Error: spec_level.csv not found. Run 02_build_spec_level.py first.")
        return

    # Load full dataset
    print("\nLoading specification-level data (all specs)...")
    df_all = pd.read_csv(SPEC_LEVEL_FILE)
    print(f"Loaded {len(df_all)} specifications from {df_all['paper_id'].nunique()} papers")

    by_dataset: dict[str, dict] = {}

    if SPEC_LEVEL_VERIFIED_CORE_FILE.exists():
        print("\nLoading verification-filtered core dataset...")
        df_vc = pd.read_csv(SPEC_LEVEL_VERIFIED_CORE_FILE)
        by_dataset["verified_core"] = run_one(df_vc, "verified_core")

    by_dataset["all_specs"] = run_one(df_all, "all_specs")

    primary_label = "verified_core" if "verified_core" in by_dataset else "all_specs"
    primary = by_dataset[primary_label]
    results = {
        "primary_dataset": primary_label,
        **{k: v for k, v in primary.items() if k != "dataset"},
    }
    # Keep secondary results (for appendix/debugging)
    for lab, res in by_dataset.items():
        if lab == primary_label:
            continue
        results[lab] = res

    # =========================================================================
    # Save results
    # =========================================================================
    print("\n" + "-" * 40)
    print(f"Saving results to {OUTPUT_FILE}")
    print("-" * 40)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
