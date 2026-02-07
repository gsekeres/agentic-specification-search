#!/usr/bin/env python3
"""
15_journal_subgroup.py
======================

Journal subgroup analysis: fit K=3 truncated-normal mixture on AER papers
versus all non-AER (AEJ + AER: P&P) papers and compare parameters to the
full-sample estimates.

Outputs:
  - estimation/results/journal_subgroup.json
  - estimation/figures/fig_journal_subgroup.pdf
  - overleaf/tex/v8_figures/fig_journal_subgroup.pdf
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
OUTPUT_JSON = RESULTS_DIR / "journal_subgroup.json"
OUTPUT_FIG = "fig_journal_subgroup.pdf"

WINSORIZE_THRESHOLD = 20.0

# ── Import fitting function from 04_fit_mixture.py ───────────────────────────
SCRIPTS_DIR = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("fit_mixture", SCRIPTS_DIR / "04_fit_mixture.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
fit_truncnorm_mixture = _mod.fit_truncnorm_mixture


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Journal Subgroup Analysis (AER vs. non-AER)")
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
    # 2. Full-sample fit (sigma=1 fixed, matching baseline)
    # ------------------------------------------------------------------
    print("\nFitting full-sample K=3 mixture (sigma=1 fixed)...")
    full_params = fit_truncnorm_mixture(
        df["Z_w"].to_numpy(), n_components=3, n_init=40,
        random_state=42, lo=0.0, sigma_constraint="fixed_1",
    )
    print("  Full-sample params:")
    for k in sorted(full_params["pi"].keys()):
        print(f"    {k}: pi={full_params['pi'][k]:.4f}, mu={full_params['mu'][k]:.4f}")

    # ------------------------------------------------------------------
    # 3. AER vs non-AER subgroups
    # ------------------------------------------------------------------
    subgroups = {
        "AER": df[df["journal"] == "AER"],
        "Non-AER": df[df["journal"] != "AER"],
    }

    output: dict = {
        "full_sample": {
            "n_obs": int(full_params["n_obs"]),
            "n_papers": int(df["paper_id"].nunique()),
            "params": {
                "pi": full_params["pi"],
                "mu": full_params["mu"],
                "sigma": full_params["sigma"],
            },
            "log_likelihood": full_params["log_likelihood"],
            "aic": full_params["aic"],
            "bic": full_params["bic"],
        },
        "subgroups": {},
    }

    subgroup_fits: list[dict] = []

    for label, sub_df in subgroups.items():
        n_papers = sub_df["paper_id"].nunique()
        z_sub = sub_df["Z_w"].to_numpy()
        print(f"\n{label}: {n_papers} papers, {len(z_sub):,} specs")

        try:
            params = fit_truncnorm_mixture(
                z_sub, n_components=3, n_init=40,
                random_state=42, lo=0.0, sigma_constraint="fixed_1",
            )
        except Exception as e:
            print(f"  Fitting failed: {e}")
            continue

        for k in sorted(params["pi"].keys()):
            print(f"    {k}: pi={params['pi'][k]:.4f}, mu={params['mu'][k]:.4f}")

        output["subgroups"][label] = {
            "n_obs": int(params["n_obs"]),
            "n_papers": int(n_papers),
            "params": {
                "pi": params["pi"],
                "mu": params["mu"],
                "sigma": params["sigma"],
            },
            "log_likelihood": params["log_likelihood"],
            "aic": params["aic"],
            "bic": params["bic"],
        }

        # Collect flat params for plotting
        flat = {}
        for k in sorted(params["pi"].keys()):
            flat[f"pi_{k}"] = float(params["pi"][k])
            flat[f"mu_{k}"] = float(params["mu"][k])
        subgroup_fits.append({"label": label, **flat})

    # ------------------------------------------------------------------
    # 4. Per-journal fits (all 6 journals)
    # ------------------------------------------------------------------
    print("\n\nPer-journal fits:")
    journal_fits: list[dict] = []
    journals = sorted(df["journal"].dropna().unique())
    for journal in journals:
        z_j = df.loc[df["journal"] == journal, "Z_w"].to_numpy()
        n_papers_j = df.loc[df["journal"] == journal, "paper_id"].nunique()
        print(f"\n  {journal}: {n_papers_j} papers, {len(z_j):,} specs")

        try:
            params_j = fit_truncnorm_mixture(
                z_j, n_components=3, n_init=40,
                random_state=42, lo=0.0, sigma_constraint="fixed_1",
            )
        except Exception as e:
            print(f"    Fitting failed: {e}")
            continue

        for k in sorted(params_j["pi"].keys()):
            print(f"    {k}: pi={params_j['pi'][k]:.4f}, mu={params_j['mu'][k]:.4f}")

        output["subgroups"][journal] = {
            "n_obs": int(params_j["n_obs"]),
            "n_papers": int(n_papers_j),
            "params": {
                "pi": params_j["pi"],
                "mu": params_j["mu"],
                "sigma": params_j["sigma"],
            },
            "log_likelihood": params_j["log_likelihood"],
            "aic": params_j["aic"],
            "bic": params_j["bic"],
        }

        flat = {}
        for k in sorted(params_j["pi"].keys()):
            flat[f"pi_{k}"] = float(params_j["pi"][k])
            flat[f"mu_{k}"] = float(params_j["mu"][k])
        journal_fits.append({"label": journal, **flat})

    # ------------------------------------------------------------------
    # 5. Save JSON
    # ------------------------------------------------------------------
    OUTPUT_JSON.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nWrote {OUTPUT_JSON}")

    # ------------------------------------------------------------------
    # 6. Figure: grouped bar chart of pi and mu by subgroup
    # ------------------------------------------------------------------
    matplotlib.rcParams.update({
        "font.family": "serif",
        "text.usetex": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # Internal keys from fitter: N, H, L (sorted by ascending mu).
    # Paper display convention: N, M, E.
    DISPLAY_MAP = {"N": "N", "H": "M", "L": "E"}
    key_order = ["N", "H", "L"]  # ascending-mu order from fitter
    display_names = [DISPLAY_MAP[k] for k in key_order]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)

    bar_groups = ["Full sample", "AER", "Non-AER"]
    colors = ["#333333", "#1f77b4", "#ff7f0e"]
    n_groups = len(bar_groups)
    n_types = len(key_order)
    x = np.arange(n_types)
    width = 0.25

    # Helper: get params for a group label
    def get_params(group_label):
        if group_label == "Full sample":
            return output["full_sample"]["params"]
        return output["subgroups"][group_label]["params"]

    # Panel 1: mixing weights pi
    ax = axes[0]
    for i, (glabel, color) in enumerate(zip(bar_groups, colors)):
        params = get_params(glabel)
        vals = [params["pi"][k] for k in key_order]
        offset = (i - (n_groups - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=glabel, color=color, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"${d}$" for d in display_names], fontsize=12)
    ax.set_ylabel(r"Mixing weight $\pi_k$", fontsize=12)
    ax.legend(frameon=False, fontsize=10)

    # Panel 2: means mu
    ax = axes[1]
    for i, (glabel, color) in enumerate(zip(bar_groups, colors)):
        params = get_params(glabel)
        vals = [params["mu"][k] for k in key_order]
        offset = (i - (n_groups - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=glabel, color=color, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"${d}$" for d in display_names], fontsize=12)
    ax.set_ylabel(r"Component mean $\mu_k$", fontsize=12)

    fig.savefig(FIG_DIR / OUTPUT_FIG, bbox_inches="tight", facecolor="white", dpi=300)
    print(f"Saved {FIG_DIR / OUTPUT_FIG}")

    if OL_FIG_DIR.is_dir():
        fig.savefig(OL_FIG_DIR / OUTPUT_FIG, bbox_inches="tight", transparent=True, dpi=300)
        print(f"Saved {OL_FIG_DIR / OUTPUT_FIG}")

    plt.close(fig)
    print("\nDone!")


if __name__ == "__main__":
    main()
