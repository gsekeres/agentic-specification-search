#!/usr/bin/env python3
"""
10_bootstrap_mixture_ci.py
=========================

Parametric bootstrap confidence intervals for the **baseline** K=3 truncated-normal
mixture parameters used in the screening counterfactuals.

Baseline model:
  |Z| := |t| (no clipping/capping),
  |Z| | k ~ TruncNormal(mu_k, sigma_k; lo=0), k in {N, H, L},
  with sigma constrained to 1.0 (for comparability across fits).

Algorithm:
  1. Load fitted params from mixture_params_abs_t.json
     (key: spec_level.baseline_only_sigma_fixed_1).
  2. For B bootstrap replications:
     a. Draw n_obs samples from the fitted truncated-normal mixture.
     b. Re-fit the 3-component truncated-normal mixture (sigma=1 fixed).
     c. Record pi, mu for each component (sorted by mu).
  3. Output point estimates, bootstrap SE, and 2.5/97.5 percentiles.

Reads:
  - estimation/results/mixture_params_abs_t.json

Output:
  - estimation/results/bootstrap_mixture_ci.json
  - estimation/figures/fig_bootstrap_mixture_ci.pdf
  - overleaf/tex/v8_figures/fig_bootstrap_mixture_ci.pdf (if directory exists)
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import truncnorm as sp_truncnorm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "estimation" / "data"
RESULTS_DIR = BASE_DIR / "estimation" / "results"
FIG_DIR = BASE_DIR / "estimation" / "figures"
OL_FIG_DIR = BASE_DIR / "overleaf" / "tex" / "v8_figures"
SCRIPTS_DIR = Path(__file__).parent

MIXTURE_FILE = RESULTS_DIR / "mixture_params_abs_t.json"
OUTPUT_JSON = RESULTS_DIR / "bootstrap_mixture_ci.json"
OUTPUT_NAME = "fig_bootstrap_mixture_ci.pdf"

# Bootstrap parameters
B = 500
SEED = 42
COMPONENT_ORDER = ["N", "H", "L"]


# ---------------------------------------------------------------------------
# Import fit_truncnorm_mixture from 04_fit_mixture.py
# ---------------------------------------------------------------------------
def _import_fit_mixture():
    spec = importlib.util.spec_from_file_location(
        "fit_mixture", SCRIPTS_DIR / "04_fit_mixture.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.fit_truncnorm_mixture


# ---------------------------------------------------------------------------
# Sampling from a truncated normal
# ---------------------------------------------------------------------------
def sample_truncnorm(mu: float, sigma: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw samples from N(mu, sigma^2) truncated to [0, +inf)."""
    sigma = float(max(sigma, 1e-8))
    a = (0.0 - float(mu)) / sigma
    b = np.inf
    return sp_truncnorm.rvs(a, b, loc=float(mu), scale=sigma, size=int(size), random_state=rng)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("Parametric Bootstrap CIs for Mixture Parameters")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load fitted parameters (baseline K=3 truncnorm, sigma=1 fixed)
    with open(MIXTURE_FILE, "r") as f:
        all_params = json.load(f)
    params = all_params.get("spec_level", {}).get("baseline_only_sigma_fixed_1")
    if params is None:
        raise KeyError("Missing spec_level.baseline_only_sigma_fixed_1 in mixture_params_abs_t.json")
    pi = params["pi"]
    mu = params["mu"]
    sigma = params["sigma"]

    print("\nPoint estimates (truncnorm K=3, sigma=1 fixed):")
    for k in COMPONENT_ORDER:
        print(f"  {k}: pi={pi[k]:.4f}, mu={mu[k]:.4f}, sigma={sigma[k]:.4f}")

    # 2. Bootstrap sample size
    n_obs = int(params.get("n_obs", 0))
    if n_obs <= 0:
        raise ValueError("Invalid n_obs in mixture parameters.")
    print(f"\nn_obs = {n_obs}")

    # 3. Import the fitting function
    fit_truncnorm_mixture = _import_fit_mixture()

    # 4. Bootstrap
    pi_arr = np.array([pi[k] for k in COMPONENT_ORDER])
    mu_arr = np.array([mu[k] for k in COMPONENT_ORDER])
    sig_arr = np.array([sigma[k] for k in COMPONENT_ORDER])

    # Storage: (B, 6) for pi_N, pi_H, pi_L, mu_N, mu_H, mu_L (sigma fixed at 1)
    boot_params = np.full((B, 6), np.nan)

    rng = np.random.default_rng(SEED)
    t0 = time.time()
    n_failed = 0

    for b in range(B):
        if (b + 1) % 50 == 0 or b == 0:
            elapsed = time.time() - t0
            print(f"  Bootstrap replication {b + 1}/{B}  ({elapsed:.1f}s elapsed)")

        # a. Draw n_obs samples from the fitted truncated-normal mixture
        components = rng.choice(3, size=n_obs, p=pi_arr)
        samples = np.empty(n_obs)
        for k in range(3):
            mask = components == k
            n_k = mask.sum()
            if n_k > 0:
                samples[mask] = sample_truncnorm(
                    mu_arr[k], sig_arr[k], n_k, rng
                )

        # b. Re-fit the truncated-normal mixture (sigma=1 fixed)
        try:
            result = fit_truncnorm_mixture(
                samples, n_components=3, n_init=10, lo=0.0,
                sigma_constraint="fixed_1",
                random_state=int(b),
            )
            boot_params[b, 0:3] = [result["pi"][k] for k in COMPONENT_ORDER]
            boot_params[b, 3:6] = [result["mu"][k] for k in COMPONENT_ORDER]
        except Exception as e:
            n_failed += 1
            if n_failed <= 5:
                print(f"    Warning: bootstrap {b} failed: {e}")

    elapsed = time.time() - t0
    print(f"\nBootstrap complete: {B} replications in {elapsed:.1f}s")
    if n_failed > 0:
        print(f"  {n_failed} replications failed (NaN rows)")

    # Remove failed rows for summary statistics
    valid_mask = np.all(np.isfinite(boot_params), axis=1)
    boot_valid = boot_params[valid_mask]
    n_valid = boot_valid.shape[0]
    print(f"  {n_valid} valid replications")

    # 5. Compute summaries (sigma fixed at 1, so only 6 params)
    param_names = [
        "pi_N", "pi_H", "pi_L",
        "mu_N", "mu_H", "mu_L",
    ]
    point_estimates = list(pi_arr) + list(mu_arr)

    summary = {}
    for i, name in enumerate(param_names):
        vals = boot_valid[:, i]
        summary[name] = {
            "point_estimate": float(point_estimates[i]),
            "bootstrap_se": float(np.std(vals, ddof=1)),
            "ci_2_5": float(np.percentile(vals, 2.5)),
            "ci_97_5": float(np.percentile(vals, 97.5)),
            "bootstrap_mean": float(np.mean(vals)),
        }

    output = {
        "B": B,
        "n_valid": n_valid,
        "n_obs": n_obs,
        "seed": SEED,
        "distribution": "truncnorm",
        "sigma_constraint": "fixed_1",
        "mixture_source": "spec_level:baseline_only_sigma_fixed_1",
        "parameters": summary,
    }

    OUTPUT_JSON.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nWrote {OUTPUT_JSON}")

    # Print summary table
    print(f"\n{'Parameter':<12} {'Point':>8} {'SE':>8} {'CI 2.5%':>8} {'CI 97.5%':>9}")
    print("-" * 50)
    for name in param_names:
        s = summary[name]
        print(f"{name:<12} {s['point_estimate']:8.4f} {s['bootstrap_se']:8.4f} "
              f"{s['ci_2_5']:8.4f} {s['ci_97_5']:9.4f}")

    # ------------------------------------------------------------------
    # 6. Plot: 3x3 grid of bootstrap distributions
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

    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)

    nice_labels = {
        "pi_N": r"$\pi_N$",
        "pi_H": r"$\pi_M$",
        "pi_L": r"$\pi_E$",
        "mu_N": r"$\mu_N$",
        "mu_H": r"$\mu_M$",
        "mu_L": r"$\mu_E$",
    }

    for i, name in enumerate(param_names):
        row, col = divmod(i, 3)
        ax = axes[row, col]

        vals = boot_valid[:, i]
        pt = point_estimates[i]
        lo_ci = summary[name]["ci_2_5"]
        hi_ci = summary[name]["ci_97_5"]

        ax.hist(vals, bins=30, color="steelblue", alpha=0.7, edgecolor="white",
                linewidth=0.5)
        ax.axvline(pt, color="black", linewidth=1.5, linestyle="-",
                   label="Point est.")
        ax.axvline(lo_ci, color="firebrick", linewidth=1.2, linestyle="--",
                   label="95\\% CI")
        ax.axvline(hi_ci, color="firebrick", linewidth=1.2, linestyle="--")

        ax.set_xlabel(nice_labels[name], fontsize=11)
        ax.set_ylabel("Count" if col == 0 else "", fontsize=10)

        if i == 0:
            ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        f"Parametric Bootstrap ($B={B}$): Truncated-Normal Mixture ($\\sigma=1$)",
        fontsize=13,
    )

    # Save
    out_path = FIG_DIR / OUTPUT_NAME
    fig.savefig(out_path, dpi=300)
    print(f"\nSaved {out_path}")

    if OL_FIG_DIR.is_dir():
        ol_path = OL_FIG_DIR / OUTPUT_NAME
        fig.savefig(ol_path, dpi=300)
        print(f"Saved {ol_path}")
    else:
        print(f"Overleaf directory not found ({OL_FIG_DIR}); skipping copy.")

    plt.close(fig)
    print("\nDone!")


if __name__ == "__main__":
    main()
