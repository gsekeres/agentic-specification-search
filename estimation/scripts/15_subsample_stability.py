#!/usr/bin/env python3
"""
15_subsample_stability.py
=========================

Subsample stability of truncated-normal mixture parameters by journal and
random half-splits.

1. Fit K=3 mixture on the full sample (benchmark).
2. For each journal with >= 20 specs, fit K=3 mixture on that journal's data.
3. Split papers randomly into two halves (seed=42), fit mixture on each half.
4. Scatter plot comparing each subsample's (pi, mu, sigma) against the
   full-sample estimates.

Outputs:
  - estimation/results/subsample_stability.json
  - estimation/figures/fig_subsample_stability.pdf
  - overleaf/tex/v8_figures/fig_subsample_stability.pdf
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "estimation" / "data"
RESULTS_DIR = BASE_DIR / "estimation" / "results"
FIG_DIR = BASE_DIR / "estimation" / "figures"
OL_FIG_DIR = Path(__file__).parent.parent.parent.parent / "overleaf" / "tex" / "v8_figures"

SPEC_FILE = DATA_DIR / "spec_level_verified_core.csv"
OUTPUT_JSON = RESULTS_DIR / "subsample_stability.json"
OUTPUT_FIG = "fig_subsample_stability.pdf"

WINSORIZE_THRESHOLD = 20.0
MIN_SPECS_PER_JOURNAL = 20

# ── Import fitting function from 04_fit_mixture.py ──────────────────────────
SCRIPTS_DIR = Path(__file__).parent
spec = importlib.util.spec_from_file_location("fit_mixture", SCRIPTS_DIR / "04_fit_mixture.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
fit_truncnorm_mixture = mod.fit_truncnorm_mixture


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_params(result: dict) -> dict:
    """Extract a flat dict of pi/mu/sigma for each label."""
    labels = sorted(result["pi"].keys())
    out = {}
    for k in labels:
        out[f"pi_{k}"] = float(result["pi"][k])
        out[f"mu_{k}"] = float(result["mu"][k])
        out[f"sigma_{k}"] = float(result["sigma"][k])
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Subsample Stability of Mixture Parameters")
    print("=" * 60)

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
    print(f"Loaded {len(df):,} specs from {df['paper_id'].nunique()} papers")

    # ------------------------------------------------------------------
    # 2. Full-sample fit
    # ------------------------------------------------------------------
    print("\nFitting full-sample K=3 mixture...")
    full_params = fit_truncnorm_mixture(
        df["Z_w"].to_numpy(), n_components=3, n_init=15, random_state=42, lo=0.0
    )
    full_flat = _extract_params(full_params)
    print("  Full-sample params:")
    for k in sorted(full_params["pi"].keys()):
        print(f"    {k}: pi={full_params['pi'][k]:.4f}, mu={full_params['mu'][k]:.4f}, sigma={full_params['sigma'][k]:.4f}")

    output: dict = {
        "full_sample": {
            "n_obs": int(full_params["n_obs"]),
            "params": {
                "pi": full_params["pi"],
                "mu": full_params["mu"],
                "sigma": full_params["sigma"],
            },
        },
        "journal_subsamples": {},
        "random_halves": {},
    }

    # ------------------------------------------------------------------
    # 3. Journal subsamples
    # ------------------------------------------------------------------
    journal_fits: list[dict] = []  # for plotting

    if "journal" in df.columns:
        journal_counts = df.groupby("journal").size()
        eligible = journal_counts[journal_counts >= MIN_SPECS_PER_JOURNAL].index.tolist()
        print(f"\nJournal subsamples ({len(eligible)} journals with >= {MIN_SPECS_PER_JOURNAL} specs):")

        for journal in sorted(eligible):
            z_j = df.loc[df["journal"] == journal, "Z_w"].to_numpy()
            try:
                params_j = fit_truncnorm_mixture(
                    z_j, n_components=3, n_init=15, random_state=42, lo=0.0
                )
            except Exception as e:
                print(f"  {journal}: fitting failed ({e})")
                continue

            flat_j = _extract_params(params_j)
            output["journal_subsamples"][journal] = {
                "n_obs": int(params_j["n_obs"]),
                "params": {
                    "pi": params_j["pi"],
                    "mu": params_j["mu"],
                    "sigma": params_j["sigma"],
                },
            }
            journal_fits.append({"label": journal, "type": "journal", **flat_j})
            print(f"  {journal} (n={params_j['n_obs']}): "
                  + ", ".join(f"pi_{k}={params_j['pi'][k]:.3f}" for k in sorted(params_j["pi"])))
    else:
        print("\nNo 'journal' column found; skipping journal subsamples.")

    # ------------------------------------------------------------------
    # 4. Random half-splits
    # ------------------------------------------------------------------
    print("\nRandom half-splits (seed=42)...")
    rng = np.random.default_rng(42)
    paper_ids = df["paper_id"].unique()
    perm = rng.permutation(paper_ids)
    mid = len(perm) // 2
    half_a_papers = set(perm[:mid])
    half_b_papers = set(perm[mid:])

    half_fits: list[dict] = []

    for label, paper_set in [("Half A", half_a_papers), ("Half B", half_b_papers)]:
        z_h = df.loc[df["paper_id"].isin(paper_set), "Z_w"].to_numpy()
        n_papers_h = df.loc[df["paper_id"].isin(paper_set), "paper_id"].nunique()
        try:
            params_h = fit_truncnorm_mixture(
                z_h, n_components=3, n_init=15, random_state=42, lo=0.0
            )
        except Exception as e:
            print(f"  {label}: fitting failed ({e})")
            continue

        flat_h = _extract_params(params_h)
        output["random_halves"][label] = {
            "n_obs": int(params_h["n_obs"]),
            "n_papers": int(n_papers_h),
            "params": {
                "pi": params_h["pi"],
                "mu": params_h["mu"],
                "sigma": params_h["sigma"],
            },
        }
        half_fits.append({"label": label, "type": "random_half", **flat_h})
        print(f"  {label} ({n_papers_h} papers, n={params_h['n_obs']}): "
              + ", ".join(f"pi_{k}={params_h['pi'][k]:.3f}" for k in sorted(params_h["pi"])))

    # ------------------------------------------------------------------
    # 5. Save JSON
    # ------------------------------------------------------------------
    OUTPUT_JSON.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nWrote {OUTPUT_JSON}")

    # ------------------------------------------------------------------
    # 6. Scatter plot: subsample vs full-sample parameters
    # ------------------------------------------------------------------
    matplotlib.rcParams.update({
        "font.family": "serif",
        "text.usetex": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # Collect all (full, sub) pairs for each parameter dimension
    labels_k = sorted(full_params["pi"].keys())  # e.g. ['H', 'L', 'N']

    param_names = []
    for k in labels_k:
        param_names.extend([f"pi_{k}", f"mu_{k}", f"sigma_{k}"])

    # Build arrays for journal and half-split points
    all_fits = journal_fits + half_fits

    if len(all_fits) == 0:
        print("No subsample fits to plot. Skipping figure.")
        return

    n_params = len(param_names)
    # 3 panels: pi, mu, sigma (each panel shows all components)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    param_groups = [
        (r"$\pi$", [f"pi_{k}" for k in labels_k]),
        (r"$\mu$", [f"mu_{k}" for k in labels_k]),
        (r"$\sigma$", [f"sigma_{k}" for k in labels_k]),
    ]

    marker_journal = "o"
    marker_half = "s"
    color_journal = "steelblue"
    color_half = "#E69F00"

    for ax, (group_label, pnames) in zip(axes, param_groups):
        for fit_info in all_fits:
            for pn in pnames:
                x_full = full_flat[pn]
                y_sub = fit_info[pn]
                if fit_info["type"] == "journal":
                    ax.scatter(
                        x_full, y_sub,
                        marker=marker_journal, s=40, alpha=0.7,
                        color=color_journal, edgecolors="white", linewidths=0.5,
                        zorder=3,
                    )
                else:
                    ax.scatter(
                        x_full, y_sub,
                        marker=marker_half, s=55, alpha=0.8,
                        color=color_half, edgecolors="white", linewidths=0.5,
                        zorder=4,
                    )

        # 45-degree line
        all_x = [full_flat[pn] for pn in pnames]
        all_y = [fit_info[pn] for fit_info in all_fits for pn in pnames]
        lo = min(min(all_x), min(all_y)) * 0.9
        hi = max(max(all_x), max(all_y)) * 1.1
        lo = max(lo, 0.0)
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.5)

        ax.set_xlabel(f"Full-sample {group_label}", fontsize=11)
        ax.set_ylabel(f"Subsample {group_label}", fontsize=11)

    # Legend (only once)
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker=marker_journal, color="w", markerfacecolor=color_journal,
               markersize=7, label="Journal"),
        Line2D([0], [0], marker=marker_half, color="w", markerfacecolor=color_half,
               markersize=7, label="Random half"),
    ]
    axes[2].legend(handles=legend_handles, frameon=False, fontsize=10, loc="upper left")

    fig.savefig(FIG_DIR / OUTPUT_FIG, bbox_inches="tight", facecolor="white", dpi=300)
    print(f"Saved {FIG_DIR / OUTPUT_FIG}")

    if OL_FIG_DIR.is_dir():
        fig.savefig(OL_FIG_DIR / OUTPUT_FIG, bbox_inches="tight", transparent=True, dpi=300)
        print(f"Saved {OL_FIG_DIR / OUTPUT_FIG}")

    plt.close(fig)
    print("\nDone!")


if __name__ == "__main__":
    main()
