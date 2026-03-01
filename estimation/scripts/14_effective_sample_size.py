#!/usr/bin/env python3
"""
21_effective_sample_size.py
===========================

Plot effective independent sample size n_eff = Delta * n as a function of
total specifications n, for all 5 non-random AR(1) orderings plus independence.

Reads:
  - estimation/results/dependence.json

Output:
  - estimation/figures/fig_effective_sample_size.pdf
  - overleaf/tex/v8_figures/fig_effective_sample_size.pdf (if directory exists)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / "estimation" / "results"
FIG_DIR = BASE_DIR / "estimation" / "figures"
OL_FIG_DIR = BASE_DIR.parent / "overleaf" / "tex" / "v8_figures"

DEPENDENCE_FILE = RESULTS_DIR / "dependence.json"
OUTPUT_NAME = "fig_effective_sample_size.pdf"

N_MAX = 500

# Non-random orderings to display
ORDERING_KEYS = ["spec_order", "lex_path", "bfs", "dfs", "by_category"]
ORDERING_LABELS = {
    "spec_order": "Document order",
    "lex_path": "Lexicographic path",
    "bfs": "Breadth-first",
    "dfs": "Depth-first",
    "by_category": "By category",
}
ORDERING_COLORS = {
    "spec_order": "tab:orange",
    "lex_path": "tab:green",
    "bfs": "tab:purple",
    "dfs": "tab:red",
    "by_category": "black",
}


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

    dep = load_dependence()

    # Extract preferred ordering
    preferred_ordering = dep.get("preferred", {}).get("ordering", "by_category")
    print(f"  Preferred ordering: {preferred_ordering}")

    # Extract AR(1) orderings
    ar1_ords = dep.get("ar1_orderings", {})

    n = np.arange(1, N_MAX + 1)

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
        n, n.astype(float),
        linestyle=":",
        color="gray",
        linewidth=1.5,
        label=r"Independence ($\Delta = 1$)",
    )

    # Plot each non-random ordering
    for key in ORDERING_KEYS:
        if key not in ar1_ords:
            print(f"  Skipping {key}: not in dependence.json")
            continue
        Delta_val = float(ar1_ords[key].get("Delta", np.nan))
        if not np.isfinite(Delta_val):
            continue

        is_preferred = (key == preferred_ordering)
        label = ORDERING_LABELS.get(key, key)
        color = ORDERING_COLORS.get(key, "tab:blue")

        ax.plot(
            n, Delta_val * n,
            linestyle="-" if is_preferred else "--",
            color=color,
            linewidth=2.5 if is_preferred else 1.5,
            label=rf"{label} ($\widehat{{\Delta}} = {Delta_val:.3f}$)",
        )
        print(f"  {label}: Delta = {Delta_val:.4f}" + (" [preferred]" if is_preferred else ""))

    ax.set_xlabel(r"Total specifications $n$", fontsize=12)
    ax.set_ylabel(r"Effective independent tests $n_{\mathrm{eff}}$", fontsize=12)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

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
