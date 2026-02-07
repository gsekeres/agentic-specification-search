#!/usr/bin/env python3
"""
17_dependence_heterogeneity.py
===============================

Estimate paper-level dependence parameter phi_i for each paper in Sample B.

For each paper with >= 10 specifications:
  1. Compute all pairwise tree distances (number of differing path components).
  2. Compute Pearson correlation of |Z| for pairs at each distance d.
  3. Fit exponential decay rho(d) = phi_i^d via log-linear regression.
  4. Record phi_i, number of specs S_i, paper_id.

Reads:
  - estimation/data/spec_level_verified_core.csv (preferred)
  - estimation/data/spec_level.csv (fallback)
  - estimation/results/dependence.json (for pooled phi reference line)

Output:
  - estimation/results/dependence_heterogeneity.csv
  - estimation/figures/fig_phi_vs_nspecs.pdf
  - overleaf/tex/v8_figures/fig_phi_vs_nspecs.pdf (if directory exists)
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "estimation" / "data"
RESULTS_DIR = BASE_DIR / "estimation" / "results"
FIG_DIR = BASE_DIR / "estimation" / "figures"
OL_FIG_DIR = Path(__file__).parent.parent.parent.parent / "overleaf" / "tex" / "v8_figures"

MIN_SPECS = 10
OUTPUT_CSV = RESULTS_DIR / "dependence_heterogeneity.csv"
OUTPUT_FIG = "fig_phi_vs_nspecs.pdf"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Load specification-level data, trying verified core first."""
    preferred = DATA_DIR / "spec_level_verified_core.csv"
    fallback = DATA_DIR / "spec_level.csv"

    if preferred.exists():
        print(f"Loading {preferred.name} ...")
        df = pd.read_csv(preferred)
    elif fallback.exists():
        print(f"Warning: {preferred.name} not found; falling back to {fallback.name}")
        df = pd.read_csv(fallback)
    else:
        raise FileNotFoundError(
            f"Neither {preferred} nor {fallback} found. "
            "Run upstream build scripts first."
        )

    print(f"  Loaded {len(df):,} rows, {df['paper_id'].nunique()} papers")
    return df


def compute_tree_distance(path1: str, path2: str) -> int:
    """
    Compute tree distance between two specification paths.

    Distance = number of differing branch levels.
    Each path looks like "methods/cross_sectional_ols.md" or
    "robustness/clustering_variations.md#robust".  We strip the .md suffix,
    split on '#' to take the directory part, then split on '/' and count
    differing levels.
    """
    def parse_path(p: str) -> list[str]:
        if pd.isna(p) or not p:
            return ["unknown"]
        p = p.replace(".md", "").split("#")[0]
        return p.split("/")

    parts1 = parse_path(path1)
    parts2 = parse_path(path2)

    max_len = max(len(parts1), len(parts2))
    distance = 0
    for i in range(max_len):
        p1 = parts1[i] if i < len(parts1) else ""
        p2 = parts2[i] if i < len(parts2) else ""
        if p1 != p2:
            distance += 1

    return distance


def estimate_phi_for_paper(
    paper_df: pd.DataFrame, z_col: str
) -> dict | None:
    """
    Estimate phi_i for a single paper using pairwise correlations by
    tree distance and log-linear regression.

    Returns dict with phi_i, n_specs, n_pairs or None if estimation fails.
    """
    z_values = paper_df[z_col].values
    paths = paper_df["spec_tree_path"].values
    n = len(paper_df)

    # Collect (z_i, z_j) pairs by distance
    distance_pairs: dict[int, list[tuple[float, float]]] = {}
    for i in range(n):
        for j in range(i + 1, n):
            d = compute_tree_distance(paths[i], paths[j])
            if d not in distance_pairs:
                distance_pairs[d] = []
            distance_pairs[d].append((z_values[i], z_values[j]))

    # Compute correlation at each distance
    corr_by_d: dict[int, float] = {}
    n_pairs_total = 0
    for d in sorted(distance_pairs.keys()):
        pairs = distance_pairs[d]
        if len(pairs) < 5:
            continue
        z1 = np.array([p[0] for p in pairs])
        z2 = np.array([p[1] for p in pairs])
        if np.std(z1) < 1e-10 or np.std(z2) < 1e-10:
            continue
        corr, _ = stats.pearsonr(z1, z2)
        corr_by_d[d] = corr
        n_pairs_total += len(pairs)

    # Fit exponential decay: log(rho) = d * log(phi) for d > 0, rho > 0.01
    valid_d = []
    valid_log_rho = []
    for d, rho in corr_by_d.items():
        if d > 0 and rho > 0.01:
            valid_d.append(d)
            valid_log_rho.append(np.log(rho))

    if len(valid_d) < 1:
        return None

    valid_d = np.array(valid_d, dtype=float)
    valid_log_rho = np.array(valid_log_rho, dtype=float)

    # Log-linear regression through origin: log(rho) = d * log(phi)
    # => log(phi) = sum(d * log(rho)) / sum(d^2)
    log_phi = np.sum(valid_d * valid_log_rho) / np.sum(valid_d ** 2)
    phi_i = np.exp(log_phi)

    return {
        "phi_i": float(phi_i),
        "n_specs": n,
        "n_pairs": n_pairs_total,
    }


def load_pooled_phi() -> float | None:
    """Load the pooled phi estimate from dependence.json."""
    dep_file = RESULTS_DIR / "dependence.json"
    if not dep_file.exists():
        print(f"  Warning: {dep_file} not found; cannot draw pooled phi reference line.")
        return None
    try:
        with open(dep_file) as f:
            dep = json.load(f)
        phi = dep["distance_based"]["decay_fit"]["phi"]
        print(f"  Pooled phi from dependence.json: {phi:.4f}")
        return float(phi)
    except (KeyError, json.JSONDecodeError) as e:
        print(f"  Warning: could not read pooled phi from dependence.json: {e}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("Paper-Level Dependence Heterogeneity")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    df = load_data()

    # Choose |Z| column
    if "Z_abs" in df.columns:
        z_col = "Z_abs"
    elif "Z" in df.columns:
        z_col = "Z"
        df[z_col] = df[z_col].abs()
    else:
        raise KeyError("Data must contain 'Z_abs' or 'Z' column.")

    df = df[df[z_col].notna() & np.isfinite(df[z_col])].copy()
    print(f"  Using z_col={z_col}, {len(df):,} valid rows")

    # ------------------------------------------------------------------
    # Filter papers with >= MIN_SPECS specifications
    # ------------------------------------------------------------------
    paper_counts = df.groupby("paper_id").size()
    eligible = paper_counts[paper_counts >= MIN_SPECS].index
    print(f"  {len(eligible)} papers with >= {MIN_SPECS} specifications")

    # ------------------------------------------------------------------
    # Estimate phi_i for each paper
    # ------------------------------------------------------------------
    results = []
    for i, pid in enumerate(eligible):
        paper_df = df[df["paper_id"] == pid].copy()
        est = estimate_phi_for_paper(paper_df, z_col)
        if est is not None:
            results.append({"paper_id": pid, **est})
        if (i + 1) % 20 == 0 or (i + 1) == len(eligible):
            print(f"  Processed {i + 1}/{len(eligible)} papers "
                  f"({len(results)} with valid phi)")

    if len(results) == 0:
        print("No papers produced a valid phi estimate. Exiting.")
        return

    results_df = pd.DataFrame(results)
    print(f"\n  Estimated phi_i for {len(results_df)} papers")
    print(f"  phi_i summary:")
    print(f"    Mean:   {results_df['phi_i'].mean():.3f}")
    print(f"    Median: {results_df['phi_i'].median():.3f}")
    print(f"    SD:     {results_df['phi_i'].std():.3f}")
    print(f"    Min:    {results_df['phi_i'].min():.3f}")
    print(f"    Max:    {results_df['phi_i'].max():.3f}")

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Saved {OUTPUT_CSV}")

    # ------------------------------------------------------------------
    # Reference line: spec-weighted mean of paper-level phis
    # (The distance-based pooled phi from dependence.json is inflated by
    # between-paper heterogeneity in mean |Z|; the paper-level weighted
    # mean is the correct cross-paper summary.)
    # ------------------------------------------------------------------
    weights = results_df["n_specs"].values.astype(float)
    weighted_mean_phi = float(np.average(results_df["phi_i"].values, weights=weights))
    print(f"  Spec-weighted mean phi_i: {weighted_mean_phi:.4f}")

    # ------------------------------------------------------------------
    # Plot: phi_i vs number of specs
    # ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "text.usetex": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)

    ax.scatter(
        results_df["n_specs"],
        results_df["phi_i"],
        s=20,
        alpha=0.6,
        linewidths=0,
        color="steelblue",
        zorder=3,
    )

    ax.axhline(
        weighted_mean_phi,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=rf"Weighted mean $\hat\varphi = {weighted_mean_phi:.2f}$",
        zorder=2,
    )
    ax.legend(frameon=False, fontsize=10)

    ax.set_xlabel("Specifications per paper", fontsize=12)
    ax.set_ylabel(r"$\hat\varphi_i$", fontsize=12)
    ax.set_xlim(left=0)

    # ------------------------------------------------------------------
    # Save figure
    # ------------------------------------------------------------------
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / OUTPUT_FIG
    fig.savefig(out_path, dpi=300)
    print(f"  Saved {out_path}")

    if OL_FIG_DIR.exists():
        ol_path = OL_FIG_DIR / OUTPUT_FIG
        fig.savefig(ol_path, dpi=300)
        print(f"  Saved {ol_path}")
    else:
        print(f"  Overleaf directory not found ({OL_FIG_DIR}); skipping copy.")

    plt.close(fig)
    print("\nDone!")


if __name__ == "__main__":
    main()
