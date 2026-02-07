#!/usr/bin/env python3
"""
19_funnel_plot.py
=================

Specification-level funnel plot: |Z| vs 1/SE (precision).

Reads:
  - estimation/data/spec_level_verified_core.csv (preferred)
  - estimation/data/spec_level.csv (fallback)

Output:
  - estimation/figures/fig_funnel_plot.pdf
  - overleaf/tex/v8_figures/fig_funnel_plot.pdf (if directory exists)
"""

from __future__ import annotations

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

OUTPUT_NAME = "fig_funnel_plot.pdf"


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


def main() -> None:
    print("=" * 60)
    print("Funnel Plot: |Z| vs Precision (1/SE)")
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

    # Compute precision = sqrt(n)
    # (1/SE is not comparable across papers with different outcome units;
    # sqrt(n) provides a unit-free, cross-paper-comparable precision measure.)
    if "n_obs" in df.columns:
        n = pd.to_numeric(df["n_obs"], errors="coerce")
        valid = n.notna() & (n > 0)
        df.loc[valid, "precision"] = np.sqrt(n[valid])
        precision_label = r"$\sqrt{n}$"
    elif "std_error" in df.columns:
        se = pd.to_numeric(df["std_error"], errors="coerce")
        valid = se.notna() & (se > 0)
        df.loc[valid, "precision"] = 1.0 / se[valid]
        precision_label = r"Precision ($1/\mathrm{SE}$)"
    else:
        raise KeyError("Data must contain 'n_obs' or 'std_error' to compute precision.")

    # Drop rows missing either axis
    plot_df = df.dropna(subset=["abs_z", "precision"]).copy()
    print(f"  Plotting {len(plot_df):,} specifications with valid |Z| and precision")

    # ------------------------------------------------------------------
    # Plot
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
        plot_df["precision"],
        plot_df["abs_z"],
        s=8,
        alpha=0.3,
        linewidths=0,
        color="steelblue",
        rasterized=True,
    )

    # Reference line at |Z| = 1.96
    ax.axhline(1.96, color="black", linestyle="--", linewidth=1.0, label=r"$|Z| = 1.96$")

    ax.set_xlabel(precision_label, fontsize=12)
    ax.set_ylabel(r"$|Z|$", fontsize=12)
    ax.legend(frameon=False, fontsize=10)

    # Log scale on x-axis (precision spans several orders of magnitude)
    ax.set_xscale("log")

    # Cap y-axis for readability (show up to ~99th percentile or 10, whichever is larger)
    y_cap = max(10.0, np.percentile(plot_df["abs_z"].dropna(), 99))
    ax.set_ylim(bottom=0, top=y_cap)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / OUTPUT_NAME
    fig.savefig(out_path, dpi=300)
    print(f"  Saved {out_path}")

    if OL_FIG_DIR.exists():
        ol_path = OL_FIG_DIR / OUTPUT_NAME
        fig.savefig(ol_path, dpi=300)
        print(f"  Saved {ol_path}")
    else:
        print(f"  Overleaf directory not found ({OL_FIG_DIR}); skipping copy.")

    plt.close(fig)
    print("\nDone!")


if __name__ == "__main__":
    main()
