#!/usr/bin/env python3
"""
22_window_surface.py
====================

Plot the separation score S(B) as a heat map over the (z_lo, z_hi) grid.

S(B) = log( p_H(B) / p_bad(B) )

where p_bad(B) = (pi_N * p_N(B) + pi_L * p_L(B)) / (pi_N + pi_L)
and p_k(B) = F_k(z_hi) - F_k(z_lo) for each type k using the
truncated-normal CDF.

Reads:
  - estimation/results/mixture_params_abs_t.json  (spec_level.baseline_only)
  - estimation/results/counterfactual_params.json  (optimal window, if available)

Output:
  - estimation/figures/fig_window_surface.pdf
  - overleaf/tex/v8_figures/fig_window_surface.pdf (if directory exists)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "estimation" / "data"
RESULTS_DIR = BASE_DIR / "estimation" / "results"
FIG_DIR = BASE_DIR / "estimation" / "figures"
OL_FIG_DIR = Path(__file__).parent.parent.parent.parent / "overleaf" / "tex" / "v8_figures"

OUTPUT_NAME = "fig_window_surface.pdf"

# Grid parameters
Z_LO_MIN = 1.96
Z_LO_MAX = 12.0
Z_LO_STEP = 0.25

Z_HI_OFFSET = 0.5  # z_hi starts at z_lo + this offset
Z_HI_MAX = 20.0
Z_HI_STEP = 0.25

# Thresholds for valid separation score
P_H_MIN_GRID = 0.01     # require p_H > this for score to be non-NaN
P_BAD_MIN = 0.001        # require p_bad > this
P_H_FLOOR = 0.05         # require p_H >= this for inclusion in the final surface


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict | None:
    """Load a JSON file, returning None if missing."""
    if not path.exists():
        print(f"  Warning: {path} not found.")
        return None
    with open(path, "r") as f:
        return json.load(f)


def truncnorm_cdf(x: float, mu: float, sigma: float, lo: float) -> float:
    """
    CDF of TruncatedNormal(mu, sigma, lo, +inf) evaluated at x.
    Returns 0 if x <= lo, 1 if x -> +inf.
    """
    sigma = max(sigma, 1e-12)
    if x <= lo:
        return 0.0
    alpha = (lo - mu) / sigma
    denom = max(float(stats.norm.sf(alpha)), 1e-15)
    numer = float(stats.norm.cdf((x - mu) / sigma)) - float(stats.norm.cdf(alpha))
    return max(0.0, min(1.0, numer / denom))


def pass_probability(z_lo: float, z_hi: float, mu: float, sigma: float, lo: float) -> float:
    """Probability that a draw from TruncNorm(mu, sigma, lo) lands in [z_lo, z_hi]."""
    cdf_hi = truncnorm_cdf(z_hi, mu, sigma, lo)
    cdf_lo = truncnorm_cdf(z_lo, mu, sigma, lo) if z_lo > lo else 0.0
    return max(0.0, cdf_hi - cdf_lo)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Window Surface: Separation Score S(B)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load parameters
    # ------------------------------------------------------------------
    print("\nLoading parameters...")

    mixture_raw = load_json(RESULTS_DIR / "mixture_params_abs_t.json")
    if mixture_raw is None:
        print("ERROR: mixture_params_abs_t.json not found. Exiting.")
        return

    bl = mixture_raw.get("spec_level", {}).get("baseline_only", {})
    if not bl:
        print("ERROR: spec_level.baseline_only not found in mixture_params_abs_t.json")
        return

    pi = bl["pi"]
    mu = bl["mu"]
    sigma = bl["sigma"]
    trunc_lo = float(bl.get("truncation_lo", 0.0))

    print(f"  Mixture (spec_level.baseline_only):")
    for k in ["N", "H", "L"]:
        print(f"    {k}: pi={pi[k]:.4f}, mu={mu[k]:.4f}, sigma={sigma[k]:.4f}")
    print(f"    truncation_lo = {trunc_lo}")

    pi_bad_total = pi["N"] + pi["L"]

    # Load optimal window from counterfactual_params.json (if available)
    cf_params = load_json(RESULTS_DIR / "counterfactual_params.json")
    opt_z_lo = None
    opt_z_hi = None
    if cf_params is not None:
        ew = cf_params.get("evidence_window", {})
        if ew:
            opt_z_lo = ew.get("z_lo")
            opt_z_hi = ew.get("z_hi")
            if opt_z_lo is not None and opt_z_hi is not None:
                print(f"  Optimal window from counterfactual_params: [{opt_z_lo:.4f}, {opt_z_hi:.4f}]")

    # ------------------------------------------------------------------
    # Build grid
    # ------------------------------------------------------------------
    print("\nBuilding separation score grid...")

    z_lo_vals = np.arange(Z_LO_MIN, Z_LO_MAX + Z_LO_STEP / 2, Z_LO_STEP)
    z_hi_vals = np.arange(Z_LO_MIN + Z_HI_OFFSET, Z_HI_MAX + Z_HI_STEP / 2, Z_HI_STEP)

    n_lo = len(z_lo_vals)
    n_hi = len(z_hi_vals)
    print(f"  Grid size: {n_lo} x {n_hi} = {n_lo * n_hi} cells")

    # S(B) matrix: rows = z_hi index, cols = z_lo index
    S = np.full((n_hi, n_lo), np.nan)
    best_score = -np.inf
    best_z_lo = None
    best_z_hi = None

    n_computed = 0
    for i, z_hi in enumerate(z_hi_vals):
        for j, z_lo in enumerate(z_lo_vals):
            # Require z_hi > z_lo + a small gap
            if z_hi <= z_lo + 0.5 - 1e-6:
                continue

            # Compute pass probabilities for each type
            p_N = pass_probability(z_lo, z_hi, mu["N"], sigma["N"], trunc_lo)
            p_H = pass_probability(z_lo, z_hi, mu["H"], sigma["H"], trunc_lo)
            p_L = pass_probability(z_lo, z_hi, mu["L"], sigma["L"], trunc_lo)

            # Require p_H >= floor
            if p_H < P_H_FLOOR:
                continue

            # Compute p_bad
            if pi_bad_total > 1e-12:
                p_bad = (pi["N"] * p_N + pi["L"] * p_L) / pi_bad_total
            else:
                p_bad = 0.0

            # Compute separation score
            if p_H > P_H_MIN_GRID and p_bad > P_BAD_MIN:
                score = np.log(p_H / p_bad)
                S[i, j] = score
                n_computed += 1

                if score > best_score:
                    best_score = score
                    best_z_lo = z_lo
                    best_z_hi = z_hi

    print(f"  Computed {n_computed} valid score cells.")
    if best_z_lo is not None:
        print(f"  Best grid score: S={best_score:.3f} at z_lo={best_z_lo:.2f}, z_hi={best_z_hi:.2f}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    print("\nGenerating figure...")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "text.usetex": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

    # Use pcolormesh for heat map
    # pcolormesh expects edges, so we compute cell edges
    z_lo_edges = np.append(z_lo_vals - Z_LO_STEP / 2, z_lo_vals[-1] + Z_LO_STEP / 2)
    z_hi_edges = np.append(z_hi_vals - Z_HI_STEP / 2, z_hi_vals[-1] + Z_HI_STEP / 2)

    # Mask NaN values
    S_masked = np.ma.masked_invalid(S)

    pcm = ax.pcolormesh(
        z_lo_edges,
        z_hi_edges,
        S_masked,
        cmap="viridis",
        shading="flat",
        rasterized=True,
    )
    cbar = fig.colorbar(pcm, ax=ax, label=r"Separation score $S(B)$")

    # Mark the optimal window from counterfactual_params.json
    if opt_z_lo is not None and opt_z_hi is not None:
        ax.plot(
            opt_z_lo, opt_z_hi,
            marker="*",
            markersize=14,
            color="red",
            markeredgecolor="black",
            markeredgewidth=0.5,
            zorder=5,
            label=r"Optimal $B$",
        )
        ax.legend(frameon=False, fontsize=10, loc="upper left")

    # Diagonal reference (z_hi = z_lo)
    diag_range = np.linspace(Z_LO_MIN, min(Z_LO_MAX, Z_HI_MAX), 100)
    ax.plot(diag_range, diag_range, color="white", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel(r"$z_\ell$", fontsize=13)
    ax.set_ylabel(r"$z_h$", fontsize=13)

    ax.set_xlim(z_lo_edges[0], z_lo_edges[-1])
    ax.set_ylim(z_hi_edges[0], z_hi_edges[-1])

    # ------------------------------------------------------------------
    # Save figure
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
