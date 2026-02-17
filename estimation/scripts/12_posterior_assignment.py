#!/usr/bin/env python3
"""
12_posterior_assignment.py
=========================

Compute posterior type probabilities P(k|Z) for each specification under
the fitted three-component truncated-normal mixture model.

For each observation z and component k in {N, H, L}:
    f_k(z) = phi((z - mu_k) / sigma_k) / sigma_k
             / (1 - Phi(-mu_k / sigma_k))
    P(k|z) = pi_k * f_k(z) / sum_j pi_j * f_j(z)

Inputs:
    estimation/results/mixture_params_abs_t.json   (spec_level_verified_core.baseline_only_sigma_fixed_1 preferred)
    estimation/data/spec_level_verified_core.csv   (column Z_abs)

Outputs:
    estimation/results/posterior_assignments.csv
    estimation/figures/fig_posterior_heatmap.pdf
    overleaf/tex/v8_figures/fig_posterior_heatmap.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "estimation" / "data"
RESULTS_DIR = BASE_DIR / "estimation" / "results"
FIG_DIR = BASE_DIR / "estimation" / "figures"
OL_FIG_DIR = BASE_DIR / "overleaf" / "tex" / "v8_figures"

MIXTURE_FILE = RESULTS_DIR / "mixture_params_abs_t.json"
SPEC_FILE = DATA_DIR / "spec_level_verified_core.csv"
OUTPUT_CSV = RESULTS_DIR / "posterior_assignments.csv"
OUTPUT_FIG = "fig_posterior_heatmap.pdf"

# ── Component colours ────────────────────────────────────────────────────────
COLORS = {"N": "#E69F00", "H": "#56B4E9", "L": "#CC79A7"}
COMPONENT_ORDER = ["N", "H", "L"]


# ── Truncated-normal PDF ─────────────────────────────────────────────────────

def truncnorm_pdf(z: np.ndarray, mu: float, sigma: float, lo: float = 0.0) -> np.ndarray:
    """
    PDF of N(mu, sigma^2) truncated to [lo, +inf).

    f(z) = phi((z - mu)/sigma) / sigma / (1 - Phi((lo - mu)/sigma))
    """
    sigma = max(sigma, 1e-12)
    z = np.asarray(z, dtype=float)
    xi = (z - mu) / sigma
    log_pdf = norm.logpdf(xi) - np.log(sigma)
    log_norm = norm.logsf((lo - mu) / sigma)   # log(1 - Phi((lo - mu)/sigma))
    return np.exp(log_pdf - log_norm)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # Ensure output directories exist
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load mixture parameters
    # ------------------------------------------------------------------
    with open(MIXTURE_FILE, "r") as f:
        all_params = json.load(f)

    vc = all_params.get("spec_level_verified_core", {}) or {}
    params = vc.get("baseline_only_sigma_fixed_1")
    source = "spec_level_verified_core.baseline_only_sigma_fixed_1"
    if params is None:
        params = vc.get("baseline_only")
        source = "spec_level_verified_core.baseline_only"
    if params is None:
        raise RuntimeError("No baseline_only mixture params found under spec_level_verified_core in mixture_params_abs_t.json")
    pi = params["pi"]       # dict with keys N, H, L
    mu = params["mu"]       # dict with keys N, H, L
    sigma = params["sigma"] # dict with keys N, H, L
    lo = float(params.get("truncation_lo", 0.0))

    print(f"Mixture parameters ({source}):")
    for k in COMPONENT_ORDER:
        print(f"  {k}: pi={pi[k]:.4f}, mu={mu[k]:.4f}, sigma={sigma[k]:.4f}")
    print(f"  truncation_lo={lo}")

    # ------------------------------------------------------------------
    # 2. Load data
    # ------------------------------------------------------------------
    df = pd.read_csv(SPEC_FILE)
    if "Z_abs" in df.columns:
        z_col = "Z_abs"
    elif "Z" in df.columns:
        z_col = "Z"
    else:
        raise RuntimeError("Cannot find Z_abs or Z column in spec_level_verified_core.csv")

    z = pd.to_numeric(df[z_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(z)
    z_valid = z[valid]
    print(f"\nLoaded {len(z)} rows, {valid.sum()} finite observations from column '{z_col}'.")

    # ------------------------------------------------------------------
    # 3. Compute posterior probabilities
    # ------------------------------------------------------------------
    # f_k(z) for each component (n_obs x 3)
    f = np.column_stack([
        truncnorm_pdf(z_valid, mu[k], sigma[k], lo=lo) for k in COMPONENT_ORDER
    ])

    # pi_k * f_k(z)
    pi_arr = np.array([pi[k] for k in COMPONENT_ORDER])
    weighted = f * pi_arr[np.newaxis, :]

    # Normalise to get posteriors
    denom = weighted.sum(axis=1, keepdims=True)
    denom = np.where(denom > 0, denom, 1e-300)  # avoid division by zero
    posteriors = weighted / denom

    # ------------------------------------------------------------------
    # 4. Build output dataframe
    # ------------------------------------------------------------------
    out = pd.DataFrame({
        "Z_abs": z_valid,
        "P_N": posteriors[:, 0],
        "P_H": posteriors[:, 1],
        "P_L": posteriors[:, 2],
    })
    out["assigned_type"] = out[["P_N", "P_H", "P_L"]].idxmax(axis=1).str.replace("P_", "")

    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {OUTPUT_CSV}  ({len(out)} rows)")

    # Summary
    counts = out["assigned_type"].value_counts()
    print("\nAssignment counts:")
    for k in COMPONENT_ORDER:
        print(f"  {k}: {counts.get(k, 0)}")

    # ------------------------------------------------------------------
    # 5. Stacked bar chart (sorted by |Z|)
    # ------------------------------------------------------------------
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["mathtext.fontset"] = "cm"
    matplotlib.rcParams["font.size"] = 12

    # Sort by Z_abs
    sort_idx = np.argsort(out["Z_abs"].values)
    p_n = out["P_N"].values[sort_idx]
    p_h = out["P_H"].values[sort_idx]
    p_l = out["P_L"].values[sort_idx]

    n_specs = len(out)
    x = np.arange(n_specs)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.bar(x, p_n, width=1.0, color=COLORS["N"], label=r"$N$", linewidth=0)
    ax.bar(x, p_h, width=1.0, bottom=p_n, color=COLORS["H"], label=r"$M$", linewidth=0)
    ax.bar(x, p_l, width=1.0, bottom=p_n + p_h, color=COLORS["L"], label=r"$E$", linewidth=0)

    ax.set_xlim(-0.5, n_specs - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"Specifications (sorted by $|Z|$)")
    ax.set_ylabel("Posterior probability")
    ax.legend(loc="upper left", frameon=False, fontsize=10)

    # Remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    # Save to FIG_DIR
    fig.savefig(
        FIG_DIR / OUTPUT_FIG,
        bbox_inches="tight",
        facecolor="white",
        dpi=300,
    )
    print(f"Saved {FIG_DIR / OUTPUT_FIG}")

    # Save to OL_FIG_DIR if it exists
    if OL_FIG_DIR.is_dir():
        fig.savefig(
            OL_FIG_DIR / OUTPUT_FIG,
            bbox_inches="tight",
            transparent=True,
            dpi=300,
        )
        print(f"Saved {OL_FIG_DIR / OUTPUT_FIG}")

    plt.close(fig)


if __name__ == "__main__":
    main()
