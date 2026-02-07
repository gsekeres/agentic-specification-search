#!/usr/bin/env python3
"""
23_within_paper_dispersion.py
==============================

Within-paper dispersion of |Z| across specifications.

For each paper (with >= 5 specs), show a horizontal box plot of |Z| across
specifications, sorted by the baseline |Z| value.  A vertical reference
line at |Z| = 1.96 highlights the conventional significance threshold.

Reads:
  - estimation/data/spec_level_verified_core.csv (preferred)
  - estimation/data/spec_level.csv (fallback)

Output:
  - estimation/figures/fig_within_paper_dispersion.pdf
  - overleaf/tex/v8_figures/fig_within_paper_dispersion.pdf (if directory exists)
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "estimation" / "data"
RESULTS_DIR = BASE_DIR / "estimation" / "results"
FIG_DIR = BASE_DIR / "estimation" / "figures"
OL_FIG_DIR = Path(__file__).parent.parent.parent.parent / "overleaf" / "tex" / "v8_figures"

OUTPUT_NAME = "fig_within_paper_dispersion"  # base name, pages get _1, _2, etc.
MIN_SPECS = 5
PAPERS_PER_PAGE = 40


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


def identify_baseline_z(group: pd.DataFrame) -> float | None:
    """
    Return the baseline |Z| for a paper group.

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
            return float(bl["abs_z"].iloc[0])

    # 2. spec_id == "baseline"
    if "spec_id" in group.columns:
        bl = group[group["spec_id"].astype(str).str.lower() == "baseline"]
        if len(bl) > 0:
            return float(bl["abs_z"].iloc[0])

    # 3. spec_tree_path contains "#baseline"
    if "spec_tree_path" in group.columns:
        bl = group[group["spec_tree_path"].astype(str).str.contains("#baseline", case=False, na=False)]
        if len(bl) > 0:
            return float(bl["abs_z"].iloc[0])

    # 4. First spec by order
    if "spec_order" in group.columns:
        first = group.sort_values("spec_order").iloc[0]
        return float(first["abs_z"])

    return None


def main() -> None:
    print("=" * 60)
    print("Within-Paper Dispersion of |Z|")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load and prepare data
    # ------------------------------------------------------------------
    df = load_data()

    # Compute |Z|
    if "Z_abs" in df.columns:
        df["abs_z"] = df["Z_abs"]
    elif "Z" in df.columns:
        df["abs_z"] = df["Z"].abs()
    else:
        raise KeyError("Data must contain 'Z_abs' or 'Z' column.")

    # Drop missing |Z|
    df = df.dropna(subset=["abs_z"]).copy()

    # ------------------------------------------------------------------
    # Filter papers with >= MIN_SPECS specifications
    # ------------------------------------------------------------------
    paper_counts = df.groupby("paper_id").size()
    eligible_papers = paper_counts[paper_counts >= MIN_SPECS].index
    df = df[df["paper_id"].isin(eligible_papers)].copy()
    print(f"  {len(eligible_papers)} papers with >= {MIN_SPECS} specifications")

    # ------------------------------------------------------------------
    # Compute baseline |Z| per paper and sort
    # ------------------------------------------------------------------
    baseline_z = {}
    for pid, grp in df.groupby("paper_id"):
        bz = identify_baseline_z(grp)
        if bz is not None:
            baseline_z[pid] = bz

    # Only keep papers where we could identify a baseline
    papers_with_baseline = [p for p in eligible_papers if p in baseline_z]
    print(f"  {len(papers_with_baseline)} papers with identifiable baseline |Z|")

    if len(papers_with_baseline) == 0:
        warnings.warn("No papers with identifiable baseline. Cannot produce plot.")
        return

    # Sort by baseline |Z| ascending
    sorted_papers = sorted(papers_with_baseline, key=lambda p: baseline_z[p])

    # Build list of |Z| arrays for box plots (one per paper, in sorted order)
    box_data = []
    paper_labels = []
    baseline_vals = []
    for pid in sorted_papers:
        grp = df[df["paper_id"] == pid]
        box_data.append(grp["abs_z"].values)
        paper_labels.append(str(pid))
        baseline_vals.append(baseline_z[pid])

    n_papers = len(box_data)

    # ------------------------------------------------------------------
    # Plot (split into pages of ~PAPERS_PER_PAGE)
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

    # Shared x-axis cap across all pages
    x_cap = max(10.0, np.percentile(np.concatenate(box_data), 99))

    n_pages = math.ceil(n_papers / PAPERS_PER_PAGE)
    print(f"  {n_papers} papers -> {n_pages} page(s) ({PAPERS_PER_PAGE} per page)")

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    for page_idx in range(n_pages):
        start = page_idx * PAPERS_PER_PAGE
        end = min(start + PAPERS_PER_PAGE, n_papers)
        page_box_data = box_data[start:end]
        page_baseline_vals = baseline_vals[start:end]
        page_n = len(page_box_data)

        fig_height = max(5.0, 0.25 * page_n)
        fig, ax = plt.subplots(figsize=(6.0, fig_height), constrained_layout=True)

        positions = np.arange(1, page_n + 1)

        bp = ax.boxplot(
            page_box_data,
            positions=positions,
            vert=False,
            widths=0.6,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker=".", markersize=2, alpha=0.4, color="steelblue"),
            medianprops=dict(color="black", linewidth=1.2),
            boxprops=dict(facecolor="lightsteelblue", edgecolor="steelblue", linewidth=0.8),
            whiskerprops=dict(color="steelblue", linewidth=0.8),
            capprops=dict(color="steelblue", linewidth=0.8),
        )

        # Overlay baseline |Z| as red markers
        ax.scatter(
            page_baseline_vals,
            positions,
            marker="D",
            s=14,
            color="tab:red",
            zorder=5,
            label="Baseline spec",
        )

        # Vertical reference at |Z| = 1.96
        ax.axvline(1.96, color="black", linestyle="--", linewidth=1.0, label=r"$|Z| = 1.96$")

        ax.set_xlabel(r"$|Z|$", fontsize=12)
        ax.set_ylabel("Paper (sorted by baseline $|Z|$)", fontsize=11)
        ax.set_yticks(positions)
        # Global paper numbers (1-indexed)
        ax.set_yticklabels([str(i) for i in range(start + 1, end + 1)], fontsize=5)
        ax.set_xlim(left=0, right=x_cap)

        ax.legend(frameon=False, fontsize=9, loc="lower right")

        # Save this page
        page_suffix = f"_{page_idx + 1}" if n_pages > 1 else ""
        fname = f"{OUTPUT_NAME}{page_suffix}.pdf"

        out_path = FIG_DIR / fname
        fig.savefig(out_path, dpi=300)
        print(f"  Saved {out_path}")

        if OL_FIG_DIR.exists():
            ol_path = OL_FIG_DIR / fname
            fig.savefig(ol_path, dpi=300)
            print(f"  Saved {ol_path}")

        plt.close(fig)

    print("\nDone!")


if __name__ == "__main__":
    main()
