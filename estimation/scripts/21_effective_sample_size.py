#!/usr/bin/env python3
"""
21_effective_sample_size.py
===========================

Plot effective independent sample size n_eff = Delta * n as a function of
total specifications n, for the estimated Delta and robustness variants.

Reads:
  - estimation/results/dependence.json

Output:
  - estimation/figures/fig_effective_sample_size.pdf
  - overleaf/tex/v8_figures/fig_effective_sample_size.pdf (if directory exists)
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "estimation" / "data"
RESULTS_DIR = BASE_DIR / "estimation" / "results"
FIG_DIR = BASE_DIR / "estimation" / "figures"
OL_FIG_DIR = Path(__file__).parent.parent.parent.parent / "overleaf" / "tex" / "v8_figures"

DEPENDENCE_FILE = RESULTS_DIR / "dependence.json"
OUTPUT_NAME = "fig_effective_sample_size.pdf"

N_MAX = 500


def load_dependence() -> dict:
    """Load dependence estimates from JSON."""
    if not DEPENDENCE_FILE.exists():
        warnings.warn(
            f"Dependence file not found: {DEPENDENCE_FILE}. "
            "Using default Delta values.",
            stacklevel=2,
        )
        return {}

    print(f"Loading {DEPENDENCE_FILE.name} ...")
    with open(DEPENDENCE_FILE, "r") as f:
        dep = json.load(f)
    return dep


def main() -> None:
    print("=" * 60)
    print("Effective Sample Size: n_eff = Delta * n")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load dependence parameters
    # ------------------------------------------------------------------
    dep = load_dependence()

    # Extract preferred (distance-based) Delta
    Delta_preferred = dep.get("distance_based", {}).get("Delta", None)
    if Delta_preferred is None:
        Delta_preferred = dep.get("preferred", {}).get("Delta", 0.21)
        print(f"  Using preferred Delta = {Delta_preferred:.4f}")
    else:
        print(f"  Distance-based Delta = {Delta_preferred:.4f}")

    # Extract AR(1) Delta
    Delta_ar1 = dep.get("ar1", {}).get("pooled", {}).get("Delta", None)
    if Delta_ar1 is not None:
        print(f"  AR(1) Delta = {Delta_ar1:.4f}")
    else:
        print("  AR(1) Delta not available; will omit from plot.")

    # ------------------------------------------------------------------
    # Build curves
    # ------------------------------------------------------------------
    n = np.arange(1, N_MAX + 1)

    n_eff_preferred = Delta_preferred * n
    n_eff_independence = n.astype(float)  # benchmark: Delta = 1
    n_eff_ar1 = Delta_ar1 * n if Delta_ar1 is not None else None

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

    fig, ax = plt.subplots(figsize=(5.5, 4.0), constrained_layout=True)

    # Independence benchmark
    ax.plot(
        n, n_eff_independence,
        linestyle=":",
        color="gray",
        linewidth=1.5,
        label=r"Independence ($\Delta = 1$)",
    )

    # Preferred (distance-based) estimate
    ax.plot(
        n, n_eff_preferred,
        linestyle="-",
        color="black",
        linewidth=2.5,
        label=rf"Distance-based ($\hat{{\Delta}} = {Delta_preferred:.3f}$)",
    )

    # AR(1) robustness
    if n_eff_ar1 is not None:
        ax.plot(
            n, n_eff_ar1,
            linestyle="--",
            color="tab:blue",
            linewidth=1.8,
            label=rf"AR(1) ($\hat{{\Delta}} = {Delta_ar1:.3f}$)",
        )

    ax.set_xlabel(r"Total specifications $n$", fontsize=12)
    ax.set_ylabel(r"Effective independent tests $n_{\mathrm{eff}}$", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    ax.set_xlim(0, N_MAX)
    ax.set_ylim(0, N_MAX)

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
