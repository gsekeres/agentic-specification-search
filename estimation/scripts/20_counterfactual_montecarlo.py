#!/usr/bin/env python3
"""
20_counterfactual_montecarlo.py
===============================

Monte Carlo validation of the binomial approximation used in counterfactuals.

Simulates 1000 papers, drawing types from the sigma=1 fixed mixture
and drawing independent |Z| values from the truncated-normal components.
Compares simulated null-only FDR at each threshold m with the analytical
null-only FDR from the counterfactual operating points.

Reads:
  - estimation/results/counterfactual_params.json (flat structure)

Output:
  - estimation/results/montecarlo_validation.json
  - estimation/figures/fig_montecarlo_validation.pdf
  - overleaf/tex/v8_figures/fig_montecarlo_validation.pdf (if directory exists)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / "estimation" / "results"
FIG_DIR = BASE_DIR / "estimation" / "figures"
OL_FIG_DIR = Path(__file__).parent.parent.parent.parent / "overleaf" / "tex" / "v8_figures"

OUTPUT_NAME = "fig_montecarlo_validation.pdf"
OUTPUT_JSON = RESULTS_DIR / "montecarlo_validation.json"

# Simulation parameters
N_PAPERS = 1000
RNG_SEED = 42


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


def truncnorm_rvs(
    mu: float,
    sigma: float,
    lo: float,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw from TruncatedNormal(mu, sigma, lo=lo, hi=+inf) using scipy."""
    sigma = max(sigma, 1e-12)
    a = (lo - mu) / sigma  # lower bound in standard units
    b = np.inf              # upper bound
    return stats.truncnorm.rvs(a, b, loc=mu, scale=sigma, size=size, random_state=rng)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Monte Carlo Validation of Binomial Counterfactual")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load parameters (flat structure)
    # ------------------------------------------------------------------
    print("\nLoading parameters...")

    cf_params = load_json(RESULTS_DIR / "counterfactual_params.json")
    if cf_params is None:
        print("ERROR: counterfactual_params.json not found. Exiting.")
        return

    # Mixture: read from top-level (flat structure)
    mix = cf_params.get("mixture_params", {})
    if not mix:
        print("ERROR: mixture_params not found in counterfactual_params.json")
        return

    pi = mix["pi"]       # dict with keys N, H, L
    mu = mix["mu"]
    sigma = mix["sigma"]
    trunc_lo = float(mix.get("truncation_lo", 0.0))

    mixture_source = cf_params.get("mixture_source", "unknown")
    print(f"  Mixture ({mixture_source}):")
    for k in ["N", "H", "L"]:
        print(f"    {k}: pi={pi[k]:.4f}, mu={mu[k]:.4f}, sigma={sigma[k]:.4f}")
    print(f"    truncation_lo = {trunc_lo}")

    # Cost / horizon from flat structure
    lam = float(cf_params.get("cost_parameters", {}).get("lambda_baseline", 1 / 170))
    print(f"  lambda_baseline = {lam:.6f}")

    # Evidence window from flat structure
    ew = cf_params.get("evidence_window", {})
    z_lo = float(ew.get("z_lo", 1.96))
    z_hi_raw = ew.get("z_hi", None)
    z_hi = float(z_hi_raw) if z_hi_raw is not None else np.inf
    print(f"  Evidence window B = [{z_lo:.4f}, {z_hi:.4f}]")

    # Calibrated values from flat structure
    cal = cf_params.get("calibration", {})
    main_res = cf_params.get("main_result", {})
    n_eff_old = int(main_res.get("n_eff_old", cal.get("calibrated_n_eff_old", 50)))
    n_eff_new = int(main_res.get("n_eff_new", int(np.ceil(n_eff_old / lam))))

    print(f"  n_eff_old = {n_eff_old}, n_eff_new = {n_eff_new}")

    # ------------------------------------------------------------------
    # Monte Carlo simulation
    # ------------------------------------------------------------------
    print(f"\nRunning Monte Carlo simulation with {N_PAPERS} papers...")
    rng = np.random.default_rng(RNG_SEED)

    # Type labels and probabilities
    types = ["N", "H", "L"]
    pi_arr = np.array([pi[k] for k in types])
    pi_arr = pi_arr / pi_arr.sum()  # ensure normalized

    # Draw paper types
    type_draws = rng.choice(types, size=N_PAPERS, p=pi_arr)
    type_counts = {k: int((type_draws == k).sum()) for k in types}
    print(f"  Type draws: N={type_counts['N']}, H={type_counts['H']}, L={type_counts['L']}")

    # For each regime, simulate pass counts
    def simulate_regime(n_eff: int, regime_label: str) -> dict:
        """Simulate n_eff independent |Z| draws per paper and count passes."""
        print(f"\n  Simulating regime '{regime_label}' (n_eff={n_eff})...")

        pass_counts = np.zeros(N_PAPERS, dtype=int)

        for i in range(N_PAPERS):
            k = type_draws[i]
            # Draw n_eff independent |Z| values from TruncatedNormal(mu_k, sigma_k, lo=0)
            z_vals = truncnorm_rvs(
                mu=mu[k],
                sigma=sigma[k],
                lo=trunc_lo,
                size=n_eff,
                rng=rng,
            )
            # Count how many fall in the evidence window B = [z_lo, z_hi]
            in_window = ((z_vals >= z_lo) & (z_vals <= z_hi)).sum()
            pass_counts[i] = int(in_window)

        # For each threshold m, compute simulated FDR
        m_max = min(n_eff, int(pass_counts.max()) + 1) if pass_counts.max() > 0 else 1
        m_values = list(range(1, m_max + 1))

        sim_results = []
        for m_val in m_values:
            qualified = pass_counts >= m_val
            n_qualified = int(qualified.sum())
            if n_qualified == 0:
                fdr_sim = np.nan
            else:
                # Null-only FDR = fraction of qualified papers that are type N
                null_qualified = (type_draws == "N") & qualified
                fdr_sim = float(null_qualified.sum()) / float(n_qualified)

            sim_results.append({
                "m": m_val,
                "n_qualified": n_qualified,
                "FDR_sim": fdr_sim,
            })

        return {
            "regime": regime_label,
            "n_eff": n_eff,
            "results": sim_results,
        }

    old_sim = simulate_regime(n_eff_old, "old")
    new_sim = simulate_regime(n_eff_new, "new")

    # ------------------------------------------------------------------
    # Merge simulated and analytical FDR (null-only)
    # ------------------------------------------------------------------
    print("\nMerging simulated and analytical FDR (null-only)...")

    # Use operating points from flat structure
    op_points = cf_params.get("operating_points", [])
    if op_points:
        op_df = pd.DataFrame(op_points)
        print(f"  Using operating points ({len(op_df)} rows)")
    else:
        # Fallback: load operating_points.csv
        print("  WARNING: operating_points not found in JSON, falling back to operating_points.csv")
        op_path = RESULTS_DIR / "operating_points.csv"
        if not op_path.exists():
            print(f"ERROR: {op_path} not found. Exiting.")
            return
        op_df = pd.read_csv(op_path)

    merged_rows = []

    for sim_data in [old_sim, new_sim]:
        regime = sim_data["regime"]
        n_eff = sim_data["n_eff"]

        # Get analytical rows for this regime
        analyt = op_df[op_df["regime"] == regime].copy()

        for entry in sim_data["results"]:
            m_val = entry["m"]
            fdr_sim = entry["FDR_sim"]

            # Find analytical null-only FDR for this m
            match = analyt[analyt["m"] == m_val]
            if len(match) > 0:
                fdr_analyt = float(match.iloc[0]["FDR_null"])
            else:
                fdr_analyt = np.nan

            merged_rows.append({
                "regime": regime,
                "n_eff": n_eff,
                "m": m_val,
                "FDR_simulated": fdr_sim,
                "FDR_analytical": fdr_analyt,
                "n_qualified": entry["n_qualified"],
            })

    merged_df = pd.DataFrame(merged_rows)

    # Filter to rows where both FDR values are finite
    plot_df = merged_df.dropna(subset=["FDR_simulated", "FDR_analytical"]).copy()
    plot_df = plot_df[np.isfinite(plot_df["FDR_simulated"]) & np.isfinite(plot_df["FDR_analytical"])]

    print(f"  {len(plot_df)} points with both simulated and analytical FDR")

    # ------------------------------------------------------------------
    # Save JSON output
    # ------------------------------------------------------------------
    print("\nSaving results...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_data = {
        "description": "Monte Carlo validation of binomial counterfactual approximation (null-only FDR, sigma=1 fixed mixture)",
        "n_papers": N_PAPERS,
        "rng_seed": RNG_SEED,
        "mixture_source": mixture_source,
        "lambda": lam,
        "n_eff_old": n_eff_old,
        "n_eff_new": n_eff_new,
        "evidence_window": {"z_lo": z_lo, "z_hi": z_hi if np.isfinite(z_hi) else None},
        "type_draws": type_counts,
        "old_regime": old_sim["results"],
        "new_regime": new_sim["results"],
        "merged_comparison": plot_df.to_dict("records"),
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"  Saved {OUTPUT_JSON}")

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

    fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)

    # Separate old and new regime data
    old_plot = plot_df[plot_df["regime"] == "old"]
    new_plot = plot_df[plot_df["regime"] == "new"]

    ax.scatter(
        old_plot["FDR_analytical"],
        old_plot["FDR_simulated"],
        s=30,
        color="C0",
        edgecolors="none",
        alpha=0.7,
        label=r"Old regime ($n_{\mathrm{eff}}=" + str(n_eff_old) + r"$)",
        zorder=3,
    )

    ax.scatter(
        new_plot["FDR_analytical"],
        new_plot["FDR_simulated"],
        s=30,
        color="C1",
        edgecolors="none",
        alpha=0.7,
        label=r"New regime ($n_{\mathrm{eff}}=" + str(n_eff_new) + r"$)",
        zorder=3,
    )

    # 45-degree line
    all_vals = pd.concat([
        plot_df["FDR_analytical"],
        plot_df["FDR_simulated"],
    ]).dropna()
    if len(all_vals) > 0:
        lo_val = max(0.0, all_vals.min() * 0.8)
        hi_val = min(1.0, all_vals.max() * 1.2)
    else:
        lo_val, hi_val = 0.0, 1.0

    ax.plot(
        [lo_val, hi_val],
        [lo_val, hi_val],
        color="grey",
        linestyle="--",
        linewidth=1.0,
        zorder=1,
        label="45-degree line",
    )

    ax.set_xlabel("Analytical FDR (null-only)", fontsize=12)
    ax.set_ylabel("Simulated FDR (null-only)", fontsize=12)
    ax.legend(frameon=False, fontsize=9)

    ax.set_xlim(left=lo_val, right=hi_val)
    ax.set_ylim(bottom=lo_val, top=hi_val)
    ax.set_aspect("equal", adjustable="box")

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
