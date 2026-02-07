#!/usr/bin/env python3
"""
05_estimate_dependence.py
=========================

Estimate dependence parameter phi (and effective independence Delta = 1 - phi).

Two approaches:
1. Distance-based: Correlation as function of tree distance
2. AR(1): Serial correlation along specification traversal

Output: estimation/results/dependence.json
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit, minimize

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "estimation" / "data"
RESULTS_DIR = BASE_DIR / "estimation" / "results"
SPEC_LEVEL_FILE = DATA_DIR / "spec_level.csv"
SPEC_LEVEL_VERIFIED_CORE_FILE = DATA_DIR / "spec_level_verified_core.csv"
OUTPUT_FILE = RESULTS_DIR / "dependence.json"


def compute_tree_distance(path1, path2):
    """
    Compute tree distance between two specification paths.

    Distance = number of differing branch levels.
    """
    def parse_path(p):
        if pd.isna(p) or not p:
            return ['unknown']
        p = p.replace('.md', '').split('#')[0]
        return p.split('/')

    parts1 = parse_path(path1)
    parts2 = parse_path(path2)

    # Count differences
    max_len = max(len(parts1), len(parts2))
    distance = 0
    for i in range(max_len):
        p1 = parts1[i] if i < len(parts1) else ''
        p2 = parts2[i] if i < len(parts2) else ''
        if p1 != p2:
            distance += 1

    return distance


def estimate_pairwise_correlations(df, z_col="Z", group_cols=("paper_id",)):
    """
    Estimate correlation of Z as function of tree distance.

    Returns: DataFrame with (distance, correlation, n_pairs)
    """
    group_cols = tuple(group_cols)
    # Unique groups
    groups = df[list(group_cols)].drop_duplicates()

    distance_corrs = {d: [] for d in range(6)}  # Max distance 5

    for _, g in groups.iterrows():
        mask = pd.Series(True, index=df.index)
        for c in group_cols:
            mask &= df[c].eq(g[c])
        paper_df = df[mask].copy()
        if len(paper_df) < 2:
            continue

        z_values = paper_df[z_col].values
        paths = paper_df['spec_tree_path'].values

        # Compute pairwise correlations
        for i in range(len(paper_df)):
            for j in range(i + 1, len(paper_df)):
                d = compute_tree_distance(paths[i], paths[j])
                if d < 6:
                    distance_corrs[d].append((z_values[i], z_values[j]))

    # Compute correlation at each distance
    results = []
    for d in range(6):
        pairs = distance_corrs[d]
        if len(pairs) >= 10:
            z1 = np.array([p[0] for p in pairs])
            z2 = np.array([p[1] for p in pairs])
            corr, pval = stats.pearsonr(z1, z2)
            results.append({
                'distance': d,
                'correlation': corr,
                'n_pairs': len(pairs),
                'p_value': pval
            })

    return pd.DataFrame(results)


def fit_exponential_decay(corr_df):
    """
    Fit correlation decay: rho(d) = phi^d

    Returns: phi estimate and confidence interval
    """
    d = corr_df['distance'].values
    rho = corr_df['correlation'].values

    # Filter to valid correlations
    valid = (rho > 0.01) & (d > 0)  # Require positive correlation > 0.01
    if sum(valid) < 2:
        return {
            'phi': np.nan,
            'phi_se': np.nan,
            'phi_ci_lower': np.nan,
            'phi_ci_upper': np.nan
        }

    d_valid = d[valid]
    rho_valid = rho[valid]

    # Log-linear fit: log(rho) = d * log(phi)
    log_rho = np.log(rho_valid)

    # Simple regression through origin
    phi = np.exp(np.sum(d_valid * log_rho) / np.sum(d_valid ** 2))

    # Bootstrap for SE
    n_boot = 1000
    rng = np.random.default_rng(12345)
    phi_boots = []
    n = len(d_valid)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        d_b = d_valid[idx]
        rho_b = rho_valid[idx]
        log_rho_b = np.log(np.clip(rho_b, 1e-10, 1))
        phi_b = np.exp(np.sum(d_b * log_rho_b) / np.sum(d_b ** 2))
        phi_boots.append(phi_b)

    phi_se = np.std(phi_boots)
    phi_ci = np.percentile(phi_boots, [2.5, 97.5])

    return {
        'phi': float(phi),
        'phi_se': float(phi_se),
        'phi_ci_lower': float(phi_ci[0]),
        'phi_ci_upper': float(phi_ci[1]),
    }


def estimate_ar1_by_paper(df, z_col="Z", group_cols=("paper_id",)):
    """
    Estimate AR(1) coefficient for each group and pool.

    Z_{i,s+1} = phi * Z_{i,s} + u_{i,s+1}
    """
    phi_estimates = []

    group_cols = tuple(group_cols)
    groups = df[list(group_cols)].drop_duplicates()
    for _, g in groups.iterrows():
        mask = pd.Series(True, index=df.index)
        for c in group_cols:
            mask &= df[c].eq(g[c])
        paper_df = df[mask].sort_values('spec_order').copy()
        if len(paper_df) < 3:
            continue

        z = paper_df[z_col].values
        if np.std(z) < 1e-6:
            continue

        # AR(1) regression: z[t+1] on z[t]
        z_lag = z[:-1]
        z_lead = z[1:]

        if len(z_lag) >= 2:
            corr, _ = stats.pearsonr(z_lag, z_lead)
            group_id = "|".join(str(g[c]) for c in group_cols)
            phi_estimates.append({
                'group_id': group_id,
                'phi': corr,
                'n_specs': len(paper_df)
            })

    phi_df = pd.DataFrame(phi_estimates)

    if len(phi_df) == 0:
        return {'pooled_phi': np.nan}

    # Weighted average (weight by n_specs)
    weights = phi_df['n_specs'].values
    pooled_phi = np.average(phi_df['phi'].values, weights=weights)

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
        'n_groups': len(phi_df),
        'group_level': phi_df.to_dict('records'),
    }

def fit_equicorrelated(corr_df):
    """
    No-decay model: rho(d) = rho_bar for all d > 0.
    Just the mean correlation across all distances > 0.
    """
    mask = corr_df['distance'] > 0
    if mask.sum() == 0:
        return {'rho_bar': np.nan, 'Delta': np.nan}
    rho_vals = corr_df.loc[mask, 'correlation'].values
    n_pairs = corr_df.loc[mask, 'n_pairs'].values
    rho_bar = float(np.average(rho_vals, weights=n_pairs))
    return {
        'model': 'equicorrelated',
        'rho_bar': rho_bar,
        'Delta': float(1 - rho_bar),
        'n_distances': int(mask.sum()),
    }


def fit_linear_decay(corr_df):
    """
    Linear decay: rho(d) = max(0, a - b*d).
    OLS of rho on d for d > 0 with non-negativity constraint on predicted values.
    Implied Delta at d=1: 1 - max(0, a - b).
    """
    mask = (corr_df['distance'] > 0) & corr_df['correlation'].notna()
    if mask.sum() < 2:
        return {'a': np.nan, 'b': np.nan, 'Delta_at_d1': np.nan}
    d = corr_df.loc[mask, 'distance'].values.astype(float)
    rho = corr_df.loc[mask, 'correlation'].values.astype(float)
    weights = corr_df.loc[mask, 'n_pairs'].values.astype(float)

    # Weighted OLS: rho = a - b*d
    X = np.column_stack([np.ones_like(d), -d])
    W = np.diag(weights / weights.sum())
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ rho
    try:
        beta = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        return {'a': np.nan, 'b': np.nan, 'Delta_at_d1': np.nan}
    a, b = float(beta[0]), float(beta[1])
    rho_d1 = max(0.0, a - b)
    return {
        'model': 'linear_decay',
        'a': a,
        'b': b,
        'rho_at_d1': rho_d1,
        'Delta_at_d1': float(1 - rho_d1),
    }


def fit_power_law_decay(corr_df):
    """
    Power-law decay: rho(d) = a * d^(-b).
    Log-log regression: log(rho) = log(a) - b*log(d) for d > 0, rho > 0.
    Implied Delta at d=1: 1 - a (since d^(-b) = 1 at d=1).
    """
    mask = (corr_df['distance'] > 0) & (corr_df['correlation'] > 0.01)
    if mask.sum() < 2:
        return {'a': np.nan, 'b': np.nan, 'Delta_at_d1': np.nan}
    d = corr_df.loc[mask, 'distance'].values.astype(float)
    rho = corr_df.loc[mask, 'correlation'].values.astype(float)

    log_d = np.log(d)
    log_rho = np.log(rho)

    # OLS: log(rho) = log(a) - b*log(d)
    X = np.column_stack([np.ones_like(log_d), -log_d])
    try:
        beta = np.linalg.lstsq(X, log_rho, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {'a': np.nan, 'b': np.nan, 'Delta_at_d1': np.nan}
    log_a, b = float(beta[0]), float(beta[1])
    a = float(np.exp(log_a))
    rho_d1 = a  # a * 1^(-b) = a
    return {
        'model': 'power_law',
        'a': a,
        'b': b,
        'rho_at_d1': rho_d1,
        'Delta_at_d1': float(1 - rho_d1),
    }


def fit_constant_plus_exponential(corr_df):
    """
    Constant + exponential: rho(d) = c + (1-c)*phi^d.
    Captures a "floor" correlation that doesn't decay.
    NLS fit on d > 0 data.
    Implied Delta at d=1: 1 - (c + (1-c)*phi).
    """
    mask = (corr_df['distance'] > 0) & corr_df['correlation'].notna()
    if mask.sum() < 3:
        return {'c': np.nan, 'phi': np.nan, 'Delta_at_d1': np.nan}
    d = corr_df.loc[mask, 'distance'].values.astype(float)
    rho = corr_df.loc[mask, 'correlation'].values.astype(float)

    def model_fn(d_arr, c, phi):
        return c + (1 - c) * phi ** d_arr

    # Try NLS with bounds: c in [0, 0.99], phi in [0, 0.99]
    try:
        popt, pcov = curve_fit(
            model_fn, d, rho,
            p0=[0.1, 0.5],
            bounds=([0.0, 0.0], [0.99, 0.99]),
            maxfev=5000,
        )
        c_hat, phi_hat = float(popt[0]), float(popt[1])
        rho_d1 = c_hat + (1 - c_hat) * phi_hat
        # Standard errors from covariance
        se = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan, np.nan]
        return {
            'model': 'constant_plus_exponential',
            'c': c_hat,
            'phi': phi_hat,
            'c_se': float(se[0]),
            'phi_se': float(se[1]),
            'rho_at_d1': rho_d1,
            'Delta_at_d1': float(1 - rho_d1),
        }
    except (RuntimeError, ValueError):
        return {'c': np.nan, 'phi': np.nan, 'Delta_at_d1': np.nan}


def main():
    print("=" * 60)
    print("Estimating Dependence Parameter")
    print("=" * 60)

    def run_one(df_in: pd.DataFrame, dataset_label: str) -> dict:
        print("\n" + "=" * 60)
        print(f"Dataset: {dataset_label}")
        print("=" * 60)

        # Choose evidence index column.
        # Main build uses |Z| (absolute t-statistic), stored as Z_abs when available.
        # Fall back to Z_logp = -log10(p) (nonnegative, interpretable scale) for older builds,
        # then to sign-oriented t-statistics.
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

        # =========================================================================
        # Method 1: Distance-based dependence
        # =========================================================================
        print("\n" + "-" * 40)
        print("Method 1: Distance-Based Dependence")
        print("-" * 40)

        print("\nComputing pairwise correlations by tree distance...")
        corr_df = estimate_pairwise_correlations(df, z_col=z_col, group_cols=tuple(group_cols))

        print("\nCorrelation by distance:")
        for _, row in corr_df.iterrows():
            print(f"  d={row['distance']}: rho={row['correlation']:.3f} (n={row['n_pairs']})")

        print("\nFitting exponential decay: rho(d) = phi^d")
        decay_fit = fit_exponential_decay(corr_df)
        if np.isnan(decay_fit['phi']):
            print("  phi = NaN (insufficient positive correlations)")
        else:
            print(f"  phi = {decay_fit['phi']:.3f} (SE = {decay_fit['phi_se']:.3f})")
            print(f"  95% CI: [{decay_fit['phi_ci_lower']:.3f}, {decay_fit['phi_ci_upper']:.3f}]")

        out['distance_based'] = {
            'correlation_by_distance': corr_df.to_dict('records'),
            'decay_fit': decay_fit,
            'Delta': 1 - decay_fit['phi'] if not np.isnan(decay_fit['phi']) else np.nan,
        }

        # =========================================================================
        # Method 2: AR(1) by traversal
        # =========================================================================
        print("\n" + "-" * 40)
        print("Method 2: AR(1) by Specification Traversal")
        print("-" * 40)

        print("\nEstimating AR(1) for each group...")
        ar1_results = estimate_ar1_by_paper(df, z_col=z_col, group_cols=tuple(group_cols))

        print(f"\nPooled phi = {ar1_results['pooled_phi']:.3f} "
              f"(SE = {ar1_results['pooled_phi_se']:.3f})")
        print(f"95% CI: [{ar1_results['pooled_phi_ci_lower']:.3f}, "
              f"{ar1_results['pooled_phi_ci_upper']:.3f}]")
        print(f"Number of groups: {ar1_results['n_groups']}")

        # Group-level heterogeneity
        group_phis = [p['phi'] for p in ar1_results.get('group_level', [])]
        if group_phis:
            print(f"\nGroup-level heterogeneity:")
            print(f"  Mean: {np.mean(group_phis):.3f}")
            print(f"  SD: {np.std(group_phis):.3f}")
            print(f"  Min: {np.min(group_phis):.3f}")
            print(f"  Max: {np.max(group_phis):.3f}")

        out['ar1'] = {
            'pooled': {
                'phi': ar1_results['pooled_phi'],
                'phi_se': ar1_results['pooled_phi_se'],
                'phi_ci_lower': ar1_results['pooled_phi_ci_lower'],
                'phi_ci_upper': ar1_results['pooled_phi_ci_upper'],
                'Delta': 1 - ar1_results['pooled_phi'],
            },
            'n_groups': ar1_results['n_groups'],
            'group_level': ar1_results.get('group_level', []),
        }

        # =========================================================================
        # Alternative decay models
        # =========================================================================
        print("\n" + "-" * 40)
        print("Alternative Dependence Models")
        print("-" * 40)

        alt_models: dict = {}

        # 1. Equicorrelated (no decay)
        eq = fit_equicorrelated(corr_df)
        alt_models['equicorrelated'] = eq
        print(f"\n  Equicorrelated: rho_bar={eq.get('rho_bar', np.nan):.3f}, Delta={eq.get('Delta', np.nan):.3f}")

        # 2. Linear decay
        lin = fit_linear_decay(corr_df)
        alt_models['linear_decay'] = lin
        print(f"  Linear decay: a={lin.get('a', np.nan):.3f}, b={lin.get('b', np.nan):.3f}, Delta(d=1)={lin.get('Delta_at_d1', np.nan):.3f}")

        # 3. Power-law decay
        pw = fit_power_law_decay(corr_df)
        alt_models['power_law'] = pw
        print(f"  Power-law: a={pw.get('a', np.nan):.3f}, b={pw.get('b', np.nan):.3f}, Delta(d=1)={pw.get('Delta_at_d1', np.nan):.3f}")

        # 4. Constant + exponential
        ce = fit_constant_plus_exponential(corr_df)
        alt_models['constant_plus_exponential'] = ce
        print(f"  Const+exp: c={ce.get('c', np.nan):.3f}, phi={ce.get('phi', np.nan):.3f}, Delta(d=1)={ce.get('Delta_at_d1', np.nan):.3f}")

        out['alternative_models'] = alt_models

        # =========================================================================
        # Summary: Preferred estimate
        # =========================================================================
        print("\n" + "-" * 40)
        print("Summary: Preferred Dependence Estimate")
        print("-" * 40)

        # Preferred: distance-based decay fit (does not depend on an arbitrary
        # traversal ordering). Fall back to AR(1) if decay fit is unavailable.
        if not np.isnan(decay_fit['phi']):
            preferred_phi = decay_fit['phi']
            preferred_method = 'distance_based'
        elif not np.isnan(ar1_results.get('pooled_phi', np.nan)):
            preferred_phi = ar1_results['pooled_phi']
            preferred_method = 'ar1'
        else:
            preferred_phi = np.nan
            preferred_method = 'none'

        preferred_Delta = 1 - preferred_phi

        print(f"\nPreferred method: {preferred_method}")
        print(f"phi = {preferred_phi:.3f}")
        print(f"Delta (effective independence) = {preferred_Delta:.3f}")

        out['preferred'] = {
            'method': preferred_method,
            'phi': float(preferred_phi),
            'Delta': float(preferred_Delta),
        }
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
