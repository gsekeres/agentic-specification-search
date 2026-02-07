#!/usr/bin/env python3
"""
14_leave_one_out_cv.py
======================

Leave-one-paper-out cross-validation for the truncated-normal mixture model.

For each paper i:
  1. Remove all specs from paper i (training set = everything else).
  2. Fit a K-component truncated-normal mixture on the training set.
  3. Compute the held-out log-likelihood on paper i's specs.

Repeat for K in {2, 3, 4} to compare model selection.

Outputs:
  - estimation/results/leave_one_out_cv.json
  - estimation/figures/fig_leave_one_out_cv.pdf
  - overleaf/tex/v8_figures/fig_leave_one_out_cv.pdf
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "estimation" / "data"
RESULTS_DIR = BASE_DIR / "estimation" / "results"
FIG_DIR = BASE_DIR / "estimation" / "figures"
OL_FIG_DIR = Path(__file__).parent.parent.parent.parent / "overleaf" / "tex" / "v8_figures"

SPEC_FILE = DATA_DIR / "spec_level_verified_core.csv"
OUTPUT_JSON = RESULTS_DIR / "leave_one_out_cv.json"
OUTPUT_FIG = "fig_leave_one_out_cv.pdf"

WINSORIZE_THRESHOLD = 20.0

# ── Import fitting function from 04_fit_mixture.py ──────────────────────────
SCRIPTS_DIR = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("fit_mixture", SCRIPTS_DIR / "04_fit_mixture.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
fit_truncnorm_mixture = _mod.fit_truncnorm_mixture


# ── Held-out log-likelihood helpers ─────────────────────────────────────────

def truncnorm_logpdf(z, mu, sigma, lo=0.0):
    sigma = max(sigma, 1e-8)
    xi = (z - mu) / sigma
    logpdf = norm.logpdf(xi) - np.log(sigma)
    logZ = norm.logsf((lo - mu) / sigma)
    return logpdf - logZ


def mixture_loglik(z, params, lo=0.0):
    """Compute log-likelihood of data z under fitted mixture params dict."""
    labels = sorted(params["pi"].keys())
    log_pi = np.log(np.array([params["pi"][k] for k in labels]).clip(1e-12))
    logpdf = np.column_stack([
        truncnorm_logpdf(z, params["mu"][k], params["sigma"][k], lo=lo)
        for k in labels
    ])
    return logsumexp(logpdf + log_pi[None, :], axis=1).sum()


def log(msg: str = "") -> None:
    """Print with immediate flush so background runs show progress."""
    print(msg, flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    log("=" * 60)
    log("Leave-One-Paper-Out Cross-Validation")
    log("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    df = pd.read_csv(SPEC_FILE)
    z = pd.to_numeric(df["Z_abs"], errors="coerce").to_numpy(dtype=float)
    z = np.minimum(np.clip(z, 0.0, None), WINSORIZE_THRESHOLD)
    df["Z_w"] = z

    valid = np.isfinite(z)
    df = df[valid].copy()
    log(f"Loaded {len(df):,} specs from {df['paper_id'].nunique()} papers "
        f"(after winsorising at {WINSORIZE_THRESHOLD})")

    paper_ids = df["paper_id"].unique()
    n_papers = len(paper_ids)
    log(f"Unique papers: {n_papers}")

    # ------------------------------------------------------------------
    # 2. LOO-CV for each K
    # ------------------------------------------------------------------
    K_values = [2, 3, 4]
    cv_results = {}

    for K in K_values:
        log(f"\n--- K = {K} ---")
        total_ll = 0.0
        total_obs = 0
        paper_lls = []

        for i, pid in enumerate(paper_ids):
            mask_held = df["paper_id"] == pid
            z_held = df.loc[mask_held, "Z_w"].to_numpy(dtype=float)
            z_train = df.loc[~mask_held, "Z_w"].to_numpy(dtype=float)

            if len(z_held) == 0 or len(z_train) < max(10, K * 3):
                continue

            # Fit on training set
            try:
                params = fit_truncnorm_mixture(
                    z_train, n_components=K, n_init=15, random_state=42, lo=0.0
                )
            except Exception as e:
                log(f"  Paper {pid}: fitting failed ({e})")
                continue

            # Held-out log-likelihood
            ll_held = mixture_loglik(z_held, params, lo=0.0)

            if not np.isfinite(ll_held):
                continue

            total_ll += ll_held
            total_obs += len(z_held)
            paper_lls.append({
                "paper_id": str(pid),
                "n_specs": int(len(z_held)),
                "loglik": float(ll_held),
            })

            if (i + 1) % 10 == 0:
                log(f"  Completed {i + 1}/{n_papers} papers "
                    f"(running CV loglik = {total_ll:.2f})")

        mean_ll = total_ll / total_obs if total_obs > 0 else float("nan")
        cv_results[f"K={K}"] = {
            "total_cv_loglik": float(total_ll),
            "mean_per_obs_cv_loglik": float(mean_ll),
            "n_papers_used": len(paper_lls),
            "n_obs_total": int(total_obs),
            "paper_lls": paper_lls,
        }
        log(f"  K={K}: total CV loglik = {total_ll:.2f}, "
            f"mean per-obs = {mean_ll:.4f} "
            f"({len(paper_lls)} papers, {total_obs} obs)")

    # ------------------------------------------------------------------
    # 3. Save JSON
    # ------------------------------------------------------------------
    OUTPUT_JSON.write_text(json.dumps(cv_results, indent=2) + "\n")
    log(f"\nWrote {OUTPUT_JSON}")

    # ------------------------------------------------------------------
    # 4. Bar chart
    # ------------------------------------------------------------------
    matplotlib.rcParams.update({
        "font.family": "serif",
        "text.usetex": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    ks = sorted(cv_results.keys())
    k_labels = [k.replace("K=", "") for k in ks]
    mean_lls = [cv_results[k]["mean_per_obs_cv_loglik"] for k in ks]

    fig, ax = plt.subplots(figsize=(4.5, 3.5), constrained_layout=True)
    bars = ax.bar(k_labels, mean_lls, color="steelblue", edgecolor="white", width=0.5)

    # Annotate bars
    for bar, val in zip(bars, mean_lls):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_xlabel(r"Number of components ($K$)", fontsize=12)
    ax.set_ylabel("Mean per-obs CV log-likelihood", fontsize=12)

    fig.savefig(FIG_DIR / OUTPUT_FIG, bbox_inches="tight", facecolor="white", dpi=300)
    log(f"Saved {FIG_DIR / OUTPUT_FIG}")

    if OL_FIG_DIR.is_dir():
        fig.savefig(OL_FIG_DIR / OUTPUT_FIG, bbox_inches="tight", transparent=True, dpi=300)
        log(f"Saved {OL_FIG_DIR / OUTPUT_FIG}")

    plt.close(fig)
    log("\nDone!")


if __name__ == "__main__":
    main()
