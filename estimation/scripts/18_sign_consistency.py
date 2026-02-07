#!/usr/bin/env python3
"""
18_sign_consistency.py
======================

Compute sign-flip rate as a function of tree distance.

For each paper:
  1. Determine the baseline sign (sign of the baseline spec's coefficient).
  2. For each non-baseline spec, compute tree distance to the baseline and
     whether the sign of the coefficient matches the baseline sign.
  3. Aggregate across papers: at each distance d, compute the fraction of
     specs that maintain the baseline sign.

Reads:
  - estimation/data/spec_level_verified_core.csv (preferred)
  - estimation/data/spec_level.csv (fallback)

Output:
  - estimation/results/sign_consistency.csv
  - estimation/figures/fig_sign_consistency.pdf
  - overleaf/tex/v8_figures/fig_sign_consistency.pdf (if directory exists)
"""

from __future__ import annotations

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

OUTPUT_CSV = RESULTS_DIR / "sign_consistency.csv"
OUTPUT_FIG = "fig_sign_consistency.pdf"


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


def identify_baseline_row(group: pd.DataFrame) -> pd.Series | None:
    """
    Return the baseline row for a paper/group.

    Strategy (in order of preference):
      1. Row where v_is_baseline == 1
      2. Row whose spec_id exactly equals 'baseline'
      3. Row whose spec_tree_path contains '#baseline'
      4. First row (spec_order == 0) as last resort
    """
    # 1. Verified baseline flag
    if "v_is_baseline" in group.columns:
        bl = group[group["v_is_baseline"] == 1.0]
        if len(bl) > 0:
            return bl.iloc[0]

    # 2. spec_id == "baseline"
    if "spec_id" in group.columns:
        bl = group[group["spec_id"].astype(str).str.lower() == "baseline"]
        if len(bl) > 0:
            return bl.iloc[0]

    # 3. spec_tree_path contains "#baseline"
    if "spec_tree_path" in group.columns:
        bl = group[
            group["spec_tree_path"]
            .astype(str)
            .str.contains("#baseline", case=False, na=False)
        ]
        if len(bl) > 0:
            return bl.iloc[0]

    # 4. First spec by order
    if "spec_order" in group.columns:
        return group.sort_values("spec_order").iloc[0]

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("Sign Consistency vs Tree Distance")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    df = load_data()

    # Need coefficient column for sign
    if "coefficient" in df.columns:
        coeff_col = "coefficient"
    elif "Z" in df.columns:
        coeff_col = "Z"
        print("  Warning: using Z (sign-oriented) as proxy for coefficient sign.")
    else:
        raise KeyError("Data must contain 'coefficient' or 'Z' column.")

    df = df[df[coeff_col].notna()].copy()
    print(f"  Using coefficient column: {coeff_col}")

    # ------------------------------------------------------------------
    # Determine grouping: use (paper_id, v_baseline_group_id) if available
    # ------------------------------------------------------------------
    if "v_baseline_group_id" in df.columns:
        group_cols = ["paper_id", "v_baseline_group_id"]
        print("  Grouping by (paper_id, v_baseline_group_id)")
    else:
        group_cols = ["paper_id"]
        print("  Grouping by paper_id only")

    # ------------------------------------------------------------------
    # For each group, identify baseline and compute sign consistency
    # ------------------------------------------------------------------
    all_records: list[dict] = []  # Each entry: (distance, sign_match)
    n_groups_processed = 0
    n_groups_skipped = 0

    groups = df.groupby(group_cols)
    n_total_groups = len(groups)

    for group_key, grp in groups:
        if len(grp) < 2:
            n_groups_skipped += 1
            continue

        baseline_row = identify_baseline_row(grp)
        if baseline_row is None:
            n_groups_skipped += 1
            continue

        baseline_coeff = baseline_row[coeff_col]
        if baseline_coeff == 0 or pd.isna(baseline_coeff):
            n_groups_skipped += 1
            continue

        baseline_sign = np.sign(baseline_coeff)
        baseline_path = baseline_row["spec_tree_path"]
        baseline_idx = baseline_row.name  # index in df

        # Compare each non-baseline spec to the baseline
        for idx, row in grp.iterrows():
            if idx == baseline_idx:
                continue
            spec_coeff = row[coeff_col]
            if pd.isna(spec_coeff) or spec_coeff == 0:
                continue

            d = compute_tree_distance(baseline_path, row["spec_tree_path"])
            sign_match = 1 if np.sign(spec_coeff) == baseline_sign else 0
            all_records.append({"distance": d, "sign_match": sign_match})

        n_groups_processed += 1
        if n_groups_processed % 50 == 0:
            print(f"  Processed {n_groups_processed}/{n_total_groups} groups ...")

    print(f"\n  Groups processed: {n_groups_processed}")
    print(f"  Groups skipped:   {n_groups_skipped}")
    print(f"  Total spec-baseline pairs: {len(all_records)}")

    if len(all_records) == 0:
        print("No valid pairs found. Exiting.")
        return

    records_df = pd.DataFrame(all_records)

    # ------------------------------------------------------------------
    # Aggregate by distance
    # ------------------------------------------------------------------
    agg = (
        records_df.groupby("distance")
        .agg(n_specs=("sign_match", "size"), sign_match_rate=("sign_match", "mean"))
        .reset_index()
    )

    # Filter to distances with meaningful sample sizes
    agg = agg[agg["n_specs"] >= 5].copy()
    agg = agg.sort_values("distance").reset_index(drop=True)

    print("\n  Sign consistency by distance:")
    for _, row in agg.iterrows():
        print(
            f"    d={int(row['distance'])}: "
            f"rate={row['sign_match_rate']:.3f} "
            f"(n={int(row['n_specs'])})"
        )

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    agg.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Saved {OUTPUT_CSV}")

    # ------------------------------------------------------------------
    # Plot: sign consistency vs tree distance
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

    distances = agg["distance"].values
    rates = agg["sign_match_rate"].values
    n_specs = agg["n_specs"].values

    # Binomial standard errors
    se = np.sqrt(rates * (1 - rates) / n_specs)

    # Point sizes proportional to n_specs (scaled for readability)
    size_scale = n_specs / n_specs.max() * 120 + 20

    ax.errorbar(
        distances,
        rates,
        yerr=1.96 * se,
        fmt="none",
        ecolor="gray",
        elinewidth=1.0,
        capsize=3,
        capthick=1.0,
        zorder=2,
    )

    ax.scatter(
        distances,
        rates,
        s=size_scale,
        color="steelblue",
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )

    # Connect points with a line
    ax.plot(distances, rates, color="steelblue", linewidth=1.0, alpha=0.5, zorder=1)

    # Reference line at 0.5 (random sign)
    ax.axhline(
        0.5,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Random (0.5)",
        zorder=1,
    )

    ax.set_xlabel(r"Tree distance $d$", fontsize=12)
    ax.set_ylabel("Sign consistency rate", fontsize=12)
    ax.legend(frameon=False, fontsize=10)

    # Set axis limits
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=-0.2, right=distances.max() + 0.5)

    # Integer ticks on x-axis
    ax.set_xticks(np.arange(0, int(distances.max()) + 1))

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
