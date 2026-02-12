#!/usr/bin/env python3
"""
25_variance_decomposition.py
============================

Decompose the variance of |Z| into between-paper and within-paper
components using a one-way random effects (ANOVA) decomposition.

Reads:
  - estimation/data/spec_level_verified_core.csv  (preferred)
  - estimation/data/spec_level.csv                (fallback)

Writes:
  - estimation/results/variance_decomposition.json
  - overleaf/tex/v8_tables/tab_variance_decomposition.tex
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "estimation" / "data"
RESULTS_DIR = BASE_DIR / "estimation" / "results"
OL_TABLE_DIR = Path(__file__).resolve().parents[3] / "overleaf" / "tex" / "v8_tables"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fmt(x: float, nd: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return ""
    return f"{float(x):.{nd}f}"


def _pct(x: float) -> str:
    if x is None or not np.isfinite(x):
        return ""
    return f"{100 * float(x):.1f}\\%"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n")
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Variance decomposition
# ---------------------------------------------------------------------------
def variance_decomposition(df: pd.DataFrame, z_col: str = "Z_abs") -> dict:
    """
    One-way random-effects variance decomposition of |Z| by paper.

    Total variance = Between-paper variance + Within-paper variance

    Between-paper variance:  variance of paper-level means of |Z|.
    Within-paper variance:   average of within-paper variances of |Z|
                             (weighted by group size for consistency with ANOVA).

    ICC (intraclass correlation) = between / total.
    Within-paper share = 1 - ICC.
    """
    z = df[[z_col, "paper_id"]].dropna().copy()
    z.rename(columns={z_col: "z"}, inplace=True)

    N = len(z)
    n_papers = z["paper_id"].nunique()

    # Grand mean and total variance
    grand_mean = z["z"].mean()
    total_var = z["z"].var(ddof=1)

    # Paper-level means and sizes
    grp = z.groupby("paper_id")["z"]
    paper_means = grp.mean()
    paper_sizes = grp.size()

    # Between-paper variance: variance of paper means
    between_var = float(paper_means.var(ddof=1))

    # Within-paper variance: average within-paper variance
    # Use size-weighted average for consistency with ANOVA decomposition
    paper_vars = grp.var(ddof=1)  # within-paper variance for each paper
    # Papers with only 1 spec have NaN variance; treat as 0
    paper_vars = paper_vars.fillna(0.0)
    within_var_unweighted = float(paper_vars.mean())
    within_var_weighted = float((paper_vars * paper_sizes).sum() / paper_sizes.sum())

    # ANOVA-style decomposition:
    # SST = SS_between + SS_within
    ss_total = float(((z["z"] - grand_mean) ** 2).sum())
    ss_between = float(((paper_means - grand_mean) ** 2 * paper_sizes).sum())
    ss_within = ss_total - ss_between

    ms_between = ss_between / (n_papers - 1) if n_papers > 1 else np.nan
    ms_within = ss_within / (N - n_papers) if N > n_papers else np.nan

    # ICC using ANOVA estimators
    # n0 = adjusted average group size (Searle et al.)
    n_bar = N / n_papers
    n0 = (N - (paper_sizes ** 2).sum() / N) / (n_papers - 1) if n_papers > 1 else n_bar

    # sigma2_between (ANOVA estimator)
    sigma2_within_anova = ms_within
    sigma2_between_anova = (ms_between - ms_within) / n0 if n0 > 0 else np.nan
    # Truncate at 0 (between variance cannot be negative)
    sigma2_between_anova = max(0.0, sigma2_between_anova) if np.isfinite(sigma2_between_anova) else np.nan

    sigma2_total_anova = sigma2_between_anova + sigma2_within_anova
    icc = sigma2_between_anova / sigma2_total_anova if sigma2_total_anova > 0 else np.nan
    within_share = 1.0 - icc if np.isfinite(icc) else np.nan

    results = {
        "n_obs": N,
        "n_papers": n_papers,
        "grand_mean_z_abs": float(grand_mean),
        "total_variance": float(total_var),
        "between_paper_variance_simple": between_var,
        "within_paper_variance_unweighted": within_var_unweighted,
        "within_paper_variance_weighted": within_var_weighted,
        "anova": {
            "ss_total": ss_total,
            "ss_between": ss_between,
            "ss_within": ss_within,
            "ms_between": float(ms_between),
            "ms_within": float(ms_within),
            "n0_adjusted_group_size": float(n0),
            "sigma2_between": float(sigma2_between_anova),
            "sigma2_within": float(sigma2_within_anova),
            "sigma2_total": float(sigma2_total_anova),
        },
        "icc": float(icc),
        "within_paper_share": float(within_share),
        "specs_per_paper_mean": float(n_bar),
        "specs_per_paper_median": float(paper_sizes.median()),
    }
    return results


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------
def build_latex_table(results: dict, label: str) -> str:
    """Build a compact booktabs table for the variance decomposition."""
    anova = results["anova"]
    rows = [
        f"Observations & {results['n_obs']:,} \\\\",
        f"Papers & {results['n_papers']:,} \\\\",
        f"Mean $|Z|$ & {_fmt(results['grand_mean_z_abs'])} \\\\",
        r"\addlinespace",
        rf"Total variance of $|Z|$ & {_fmt(results['total_variance'])} \\",
        rf"Between-paper variance ($\hat\sigma^2_b$) & {_fmt(anova['sigma2_between'])} \\",
        rf"Within-paper variance ($\hat\sigma^2_w$) & {_fmt(anova['sigma2_within'])} \\",
        r"\addlinespace",
        rf"ICC ($\hat\sigma^2_b / (\hat\sigma^2_b + \hat\sigma^2_w)$) & {_fmt(results['icc'])} \\",
        rf"Within-paper share ($1 - \mathrm{{ICC}}$) & {_fmt(results['within_paper_share'])} \\",
    ]

    body = "\n".join(rows)
    tab = rf"""\begin{{tabular}}{{lc}}
\toprule
 & {label} \\
\midrule
{body}
\bottomrule
\end{{tabular}}"""
    return tab


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("25_variance_decomposition.py")
    print("=" * 50)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OL_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data (prefer verified core, fall back to all specs)
    # ------------------------------------------------------------------
    vc_path = DATA_DIR / "spec_level_verified_core.csv"
    spec_path = DATA_DIR / "spec_level.csv"

    if vc_path.exists():
        df = pd.read_csv(vc_path)
        source_label = "Verified core"
        source_file = "spec_level_verified_core.csv"
        print(f"  Loaded {source_file}: {len(df)} rows, {df['paper_id'].nunique()} papers")
    elif spec_path.exists():
        df = pd.read_csv(spec_path)
        source_label = "All specs"
        source_file = "spec_level.csv"
        print(f"  WARNING: verified core not found; falling back to {source_file}")
        print(f"  Loaded {source_file}: {len(df)} rows, {df['paper_id'].nunique()} papers")
    else:
        print(f"  ERROR: neither {vc_path} nor {spec_path} found. Exiting.")
        return

    # ------------------------------------------------------------------
    # Compute decomposition
    # ------------------------------------------------------------------
    print("\nComputing variance decomposition...")
    results = variance_decomposition(df, z_col="Z_abs")
    results["source_file"] = source_file

    print(f"  Total variance:          {results['total_variance']:.4f}")
    print(f"  Between-paper (ANOVA):   {results['anova']['sigma2_between']:.4f}")
    print(f"  Within-paper  (ANOVA):   {results['anova']['sigma2_within']:.4f}")
    print(f"  ICC:                     {results['icc']:.4f}")
    print(f"  Within-paper share:      {results['within_paper_share']:.4f}")

    # ------------------------------------------------------------------
    # Also compute for the full spec_level if both exist
    # ------------------------------------------------------------------
    all_results = {"verified_core": results}

    if vc_path.exists() and spec_path.exists():
        df_all = pd.read_csv(spec_path)
        results_all = variance_decomposition(df_all, z_col="Z_abs")
        results_all["source_file"] = "spec_level.csv"
        all_results["all_specs"] = results_all
        print(f"\n  [All specs]  ICC = {results_all['icc']:.4f},  within share = {results_all['within_paper_share']:.4f}")

    # ------------------------------------------------------------------
    # Write JSON
    # ------------------------------------------------------------------
    json_path = RESULTS_DIR / "variance_decomposition.json"
    json_path.write_text(json.dumps(all_results, indent=2, default=str) + "\n")
    print(f"\n  wrote {json_path}")

    # ------------------------------------------------------------------
    # Write LaTeX table
    # ------------------------------------------------------------------
    tex = build_latex_table(results, source_label)
    tex_path = OL_TABLE_DIR / "tab_variance_decomposition.tex"
    _write(tex_path, tex)

    print("\nDone.")


if __name__ == "__main__":
    main()
