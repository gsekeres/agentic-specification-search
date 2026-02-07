#!/usr/bin/env python3
"""
13_bootstrap_lrt.py
===================

Bootstrap likelihood ratio test for K=2 vs K=3 vs K=4 components
in the truncated-normal mixture model.

Algorithm:
  For each null K0 in {2, 3}:
    1. Fit K0 and K0+1 component mixtures on real data.
    2. Compute LR_obs = 2*(logL_{K0+1} - logL_{K0}).
    3. For B=200 bootstrap replications, draw from the K0 null,
       re-fit both K0 and K0+1 models, and compute LR_b.
    4. p-value = fraction of LR_b >= LR_obs.

Reads:
  - estimation/data/spec_level_verified_core.csv (column Z_abs)

Output:
  - estimation/results/bootstrap_lrt.json
  - estimation/figures/fig_bootstrap_lrt.pdf
  - overleaf/tex/v8_figures/fig_bootstrap_lrt.pdf (if directory exists)
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import truncnorm as sp_truncnorm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "estimation" / "data"
RESULTS_DIR = BASE_DIR / "estimation" / "results"
FIG_DIR = BASE_DIR / "estimation" / "figures"
OL_FIG_DIR = Path(__file__).parent.parent.parent.parent / "overleaf" / "tex" / "v8_figures"
SCRIPTS_DIR = Path(__file__).parent

DATA_FILE = DATA_DIR / "spec_level_verified_core.csv"
OUTPUT_JSON = RESULTS_DIR / "bootstrap_lrt.json"
OUTPUT_NAME = "fig_bootstrap_lrt.pdf"

# Bootstrap parameters
B = 200
SEED = 42
WINSORIZE_THRESHOLD = 20.0


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
# Sampling from a truncated-normal mixture
# ---------------------------------------------------------------------------
def sample_truncnorm(mu: float, sigma: float, lo: float, size: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Draw samples from N(mu, sigma^2) truncated to [lo, +inf)."""
    sigma = max(sigma, 1e-8)
    a = (lo - mu) / sigma
    b_param = np.inf
    return sp_truncnorm.rvs(a, b_param, loc=mu, scale=sigma, size=size,
                            random_state=rng)


def sample_from_mixture(pi_arr: np.ndarray, mu_arr: np.ndarray,
                        sig_arr: np.ndarray, lo: float, n: int,
                        rng: np.random.Generator) -> np.ndarray:
    """Draw n samples from a truncated-normal mixture."""
    K = len(pi_arr)
    components = rng.choice(K, size=n, p=pi_arr)
    samples = np.empty(n)
    for k in range(K):
        mask = components == k
        n_k = mask.sum()
        if n_k > 0:
            samples[mask] = sample_truncnorm(mu_arr[k], sig_arr[k], lo, n_k, rng)
    # Winsorize to match pipeline
    samples = np.minimum(samples, WINSORIZE_THRESHOLD)
    return samples


# ---------------------------------------------------------------------------
# Extract arrays from fit result
# ---------------------------------------------------------------------------
def _extract_arrays(result: dict):
    """Extract pi, mu, sigma arrays from a fit result dict."""
    labels = list(result["pi"].keys())
    pi = np.array([result["pi"][l] for l in labels])
    mu = np.array([result["mu"][l] for l in labels])
    sigma = np.array([result["sigma"][l] for l in labels])
    return pi, mu, sigma


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("Bootstrap Likelihood Ratio Test: K=2 vs K=3 vs K=4")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    df = pd.read_csv(DATA_FILE)
    z = pd.to_numeric(df["Z_abs"], errors="coerce").to_numpy(dtype=float)
    z = z[np.isfinite(z)]
    z = np.minimum(z, WINSORIZE_THRESHOLD)
    n_obs = len(z)
    print(f"\nLoaded {n_obs} observations from {DATA_FILE.name}")

    # Import fitting function
    fit_truncnorm_mixture = _import_fit_mixture()

    lo = 0.0
    tests = []  # will hold results for K0=2 and K0=3

    for K0 in [2, 3]:
        K1 = K0 + 1
        test_label = f"K={K0} vs K={K1}"
        print(f"\n{'='*60}")
        print(f"Test: {test_label}")
        print(f"{'='*60}")

        # a. Fit K0 and K0+1 on real data
        print(f"\n  Fitting K={K0} on real data (n_init=25)...")
        result_k0 = fit_truncnorm_mixture(z, n_components=K0, n_init=25,
                                          random_state=42, lo=lo)
        logL_k0 = result_k0["log_likelihood"]
        print(f"    logL(K={K0}) = {logL_k0:.2f}")

        print(f"  Fitting K={K1} on real data (n_init=25)...")
        result_k1 = fit_truncnorm_mixture(z, n_components=K1, n_init=25,
                                          random_state=42, lo=lo)
        logL_k1 = result_k1["log_likelihood"]
        print(f"    logL(K={K1}) = {logL_k1:.2f}")

        LR_obs = 2.0 * (logL_k1 - logL_k0)
        print(f"  LR_obs = {LR_obs:.4f}")

        # b. Extract null (K0) mixture parameters for sampling
        pi_null, mu_null, sig_null = _extract_arrays(result_k0)

        # c. Bootstrap
        rng = np.random.default_rng(SEED)
        lr_boot = np.full(B, np.nan)
        n_failed = 0
        t0 = time.time()

        for b in range(B):
            if (b + 1) % 20 == 0 or b == 0:
                elapsed = time.time() - t0
                print(f"    Bootstrap {b + 1}/{B}  ({elapsed:.1f}s elapsed)")

            # Draw from null
            boot_data = sample_from_mixture(pi_null, mu_null, sig_null,
                                            lo, n_obs, rng)

            try:
                res_b_k0 = fit_truncnorm_mixture(boot_data, n_components=K0,
                                                 n_init=10, random_state=int(b),
                                                 lo=lo)
                res_b_k1 = fit_truncnorm_mixture(boot_data, n_components=K1,
                                                 n_init=10, random_state=int(b),
                                                 lo=lo)
                lr_b = 2.0 * (res_b_k1["log_likelihood"] - res_b_k0["log_likelihood"])
                lr_boot[b] = max(lr_b, 0.0)  # LR should be non-negative
            except Exception as e:
                n_failed += 1
                if n_failed <= 5:
                    print(f"      Warning: bootstrap {b} failed: {e}")

        elapsed = time.time() - t0
        print(f"  Bootstrap complete: {elapsed:.1f}s")

        valid = np.isfinite(lr_boot)
        lr_valid = lr_boot[valid]
        n_valid = len(lr_valid)
        print(f"  {n_valid} valid / {B} total replications")

        if n_failed > 0:
            print(f"  {n_failed} replications failed")

        # d. p-value
        p_value = float(np.mean(lr_valid >= LR_obs)) if n_valid > 0 else np.nan
        print(f"  p-value = {p_value:.4f}")

        test_result = {
            "test": test_label,
            "K0": K0,
            "K1": K1,
            "LR_obs": float(LR_obs),
            "p_value": float(p_value),
            "B": B,
            "n_valid": n_valid,
            "n_obs": n_obs,
            "logL_K0": float(logL_k0),
            "logL_K1": float(logL_k1),
            "bootstrap_LR_percentiles": {
                "p5": float(np.percentile(lr_valid, 5)) if n_valid > 0 else None,
                "p25": float(np.percentile(lr_valid, 25)) if n_valid > 0 else None,
                "p50": float(np.percentile(lr_valid, 50)) if n_valid > 0 else None,
                "p75": float(np.percentile(lr_valid, 75)) if n_valid > 0 else None,
                "p95": float(np.percentile(lr_valid, 95)) if n_valid > 0 else None,
                "p99": float(np.percentile(lr_valid, 99)) if n_valid > 0 else None,
                "max": float(np.max(lr_valid)) if n_valid > 0 else None,
            },
        }
        tests.append((test_label, test_result, lr_valid, LR_obs, p_value))

    # Save JSON
    output = {"seed": SEED, "B": B, "n_obs": n_obs}
    for label, result, _, _, _ in tests:
        output[label] = result

    OUTPUT_JSON.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nWrote {OUTPUT_JSON}")

    # ------------------------------------------------------------------
    # Plot: two-panel figure
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

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    for idx, (label, result, lr_valid, lr_obs, p_val) in enumerate(tests):
        ax = axes[idx]

        # Histogram of bootstrap LR values
        n_bins = 30
        ax.hist(lr_valid, bins=n_bins, color="steelblue", alpha=0.7,
                edgecolor="white", linewidth=0.5, label="Bootstrap LR")

        # Shade rejection region (values >= LR_obs)
        if lr_obs is not None and np.isfinite(lr_obs):
            # Get histogram bin edges to shade the rejection region
            counts, bin_edges = np.histogram(lr_valid, bins=n_bins)
            for i in range(len(counts)):
                if bin_edges[i + 1] >= lr_obs:
                    ax.bar(
                        (bin_edges[i] + bin_edges[i + 1]) / 2,
                        counts[i],
                        width=bin_edges[i + 1] - bin_edges[i],
                        color="firebrick",
                        alpha=0.5,
                        edgecolor="white",
                        linewidth=0.5,
                    )

            ax.axvline(lr_obs, color="black", linewidth=1.5, linestyle="-",
                       label=f"$\\mathrm{{LR}}_{{\\mathrm{{obs}}}} = {lr_obs:.1f}$")

        ax.set_xlabel("Likelihood Ratio Statistic", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"{label}  ($p = {p_val:.3f}$)", fontsize=12)
        ax.legend(frameon=False, fontsize=9)

    fig.suptitle(
        f"Bootstrap LRT ($B={B}$): Number of Mixture Components",
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
