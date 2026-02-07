#!/usr/bin/env python3
"""
24_summary_statistics.py
========================

Produce a descriptive-statistics table for Sample A (claim level) and
Sample B (spec level, all and verified core).

Reads:
  - estimation/data/claim_level.csv          (Sample A, 40 claim-level rows)
  - estimation/data/spec_level.csv           (Sample B, all specifications)
  - estimation/data/spec_level_verified_core.csv  (Sample B, verified-core subset)
  - estimation/data/i4r_comparison.csv       (40-paper I4R comparison)

Writes:
  - estimation/results/summary_statistics.json
  - overleaf/tex/v8_tables/tab_summary_statistics.tex
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
def _fmt(x: float, nd: int = 2) -> str:
    """Format a float to *nd* decimal places; return empty string for NaN."""
    if x is None or not np.isfinite(x):
        return ""
    return f"{float(x):.{nd}f}"


def _fmt_int(x) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return ""


def _pct(x: float) -> str:
    """Format a fraction as a percentage string, e.g. 0.421 -> '42.1\\%'."""
    if x is None or not np.isfinite(x):
        return ""
    return f"{100 * float(x):.1f}\\%"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n")
    print(f"  wrote {path}")


def _iqr_str(series: pd.Series, nd: int = 2) -> str:
    """Return '[Q1, Q3]' string."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    return f"[{_fmt(q1, nd)}, {_fmt(q3, nd)}]"


# ---------------------------------------------------------------------------
# Compute summary stats for one sample
# ---------------------------------------------------------------------------
def compute_stats(
    df: pd.DataFrame,
    z_col: str,
    label: str,
) -> dict:
    """Return a dict of summary statistics for a single sample."""
    n_papers = int(df["paper_id"].nunique())
    n_specs = len(df)

    # Specs per paper
    specs_per_paper = df.groupby("paper_id").size()

    z = df[z_col].dropna()
    z_abs = z.abs()

    frac_sig = float((z_abs >= 1.96).mean()) if len(z_abs) > 0 else np.nan

    # Journals
    if "journal" in df.columns:
        journal_counts = df.drop_duplicates("paper_id")["journal"].value_counts().to_dict()
    else:
        journal_counts = {}

    stats = {
        "label": label,
        "n_papers": n_papers,
        "n_specs": n_specs,
        "specs_per_paper_mean": float(specs_per_paper.mean()),
        "specs_per_paper_median": float(specs_per_paper.median()),
        "specs_per_paper_q1": float(specs_per_paper.quantile(0.25)),
        "specs_per_paper_q3": float(specs_per_paper.quantile(0.75)),
        "z_abs_mean": float(z_abs.mean()) if len(z_abs) > 0 else np.nan,
        "z_abs_median": float(z_abs.median()) if len(z_abs) > 0 else np.nan,
        "z_abs_sd": float(z_abs.std()) if len(z_abs) > 0 else np.nan,
        "z_abs_q1": float(z_abs.quantile(0.25)) if len(z_abs) > 0 else np.nan,
        "z_abs_q3": float(z_abs.quantile(0.75)) if len(z_abs) > 0 else np.nan,
        "frac_significant_005": frac_sig,
        "journal_composition": journal_counts,
    }
    return stats


# ---------------------------------------------------------------------------
# Build the LaTeX table
# ---------------------------------------------------------------------------
def build_latex_table(
    stats_a: dict,
    stats_b_all: dict,
    stats_b_vc: dict,
) -> str:
    """Produce a booktabs tabular with three data columns."""

    def _row(description: str, a_val: str, b_val: str, bvc_val: str) -> str:
        return f"{description} & {a_val} & {b_val} & {bvc_val} \\\\"

    rows: list[str] = []

    # --- Panel: Sample size ---
    rows.append(r"\multicolumn{4}{l}{\emph{Sample size}} \\")
    rows.append(r"\midrule")
    rows.append(_row(
        "Papers",
        _fmt_int(stats_a["n_papers"]),
        _fmt_int(stats_b_all["n_papers"]),
        _fmt_int(stats_b_vc["n_papers"]),
    ))
    rows.append(_row(
        "Specifications",
        _fmt_int(stats_a["n_specs"]),
        _fmt_int(stats_b_all["n_specs"]),
        _fmt_int(stats_b_vc["n_specs"]),
    ))

    # --- Panel: Specs per paper ---
    rows.append(r"\addlinespace")
    rows.append(r"\multicolumn{4}{l}{\emph{Specifications per paper}} \\")
    rows.append(r"\midrule")
    # For Sample A each paper has 1 claim-level row, so specs_per_paper == 1
    rows.append(_row(
        "Mean",
        _fmt(stats_a["specs_per_paper_mean"], 1),
        _fmt(stats_b_all["specs_per_paper_mean"], 1),
        _fmt(stats_b_vc["specs_per_paper_mean"], 1),
    ))
    rows.append(_row(
        "Median",
        _fmt(stats_a["specs_per_paper_median"], 1),
        _fmt(stats_b_all["specs_per_paper_median"], 1),
        _fmt(stats_b_vc["specs_per_paper_median"], 1),
    ))
    rows.append(_row(
        "IQR",
        f"[{_fmt(stats_a['specs_per_paper_q1'], 1)}, {_fmt(stats_a['specs_per_paper_q3'], 1)}]",
        f"[{_fmt(stats_b_all['specs_per_paper_q1'], 1)}, {_fmt(stats_b_all['specs_per_paper_q3'], 1)}]",
        f"[{_fmt(stats_b_vc['specs_per_paper_q1'], 1)}, {_fmt(stats_b_vc['specs_per_paper_q3'], 1)}]",
    ))

    # --- Panel: |Z| distribution ---
    rows.append(r"\addlinespace")
    rows.append(r"\multicolumn{4}{l}{\emph{$|Z|$ distribution}} \\")
    rows.append(r"\midrule")
    rows.append(_row(
        "Mean",
        _fmt(stats_a["z_abs_mean"]),
        _fmt(stats_b_all["z_abs_mean"]),
        _fmt(stats_b_vc["z_abs_mean"]),
    ))
    rows.append(_row(
        "Median",
        _fmt(stats_a["z_abs_median"]),
        _fmt(stats_b_all["z_abs_median"]),
        _fmt(stats_b_vc["z_abs_median"]),
    ))
    rows.append(_row(
        "Std.\\ dev.",
        _fmt(stats_a["z_abs_sd"]),
        _fmt(stats_b_all["z_abs_sd"]),
        _fmt(stats_b_vc["z_abs_sd"]),
    ))
    rows.append(_row(
        "IQR",
        f"[{_fmt(stats_a['z_abs_q1'])}, {_fmt(stats_a['z_abs_q3'])}]",
        f"[{_fmt(stats_b_all['z_abs_q1'])}, {_fmt(stats_b_all['z_abs_q3'])}]",
        f"[{_fmt(stats_b_vc['z_abs_q1'])}, {_fmt(stats_b_vc['z_abs_q3'])}]",
    ))
    rows.append(_row(
        r"Frac.\ significant ($p<0.05$)",
        _pct(stats_a["frac_significant_005"]),
        _pct(stats_b_all["frac_significant_005"]),
        _pct(stats_b_vc["frac_significant_005"]),
    ))

    # --- Panel: Sample composition ---
    rows.append(r"\addlinespace")
    rows.append(r"\multicolumn{4}{l}{\emph{Sample composition (papers by journal)}} \\")
    rows.append(r"\midrule")

    # Collect all journals across all samples (paper-level)
    all_journals: set[str] = set()
    for s in (stats_a, stats_b_all, stats_b_vc):
        all_journals.update(s["journal_composition"].keys())

    for j in sorted(all_journals):
        a_n = stats_a["journal_composition"].get(j, 0)
        b_n = stats_b_all["journal_composition"].get(j, 0)
        bvc_n = stats_b_vc["journal_composition"].get(j, 0)
        # Escape & for LaTeX
        j_tex = j.replace("&", r"\&")
        rows.append(_row(
            f"\\quad {j_tex}",
            _fmt_int(a_n) if a_n > 0 else "---",
            _fmt_int(b_n) if b_n > 0 else "---",
            _fmt_int(bvc_n) if bvc_n > 0 else "---",
        ))

    body = "\n".join(rows)
    tab = rf"""\begin{{tabular}}{{lccc}}
\toprule
 & Sample A & Sample B & Sample B \\
 & (claim level) & (all specs) & (verified core) \\
\midrule
{body}
\bottomrule
\end{{tabular}}"""
    return tab


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("24_summary_statistics.py")
    print("=" * 50)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OL_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    # Sample A: claim level
    claim_path = DATA_DIR / "claim_level.csv"
    if not claim_path.exists():
        print(f"  WARNING: {claim_path} not found; skipping Sample A.")
        df_claim = pd.DataFrame()
    else:
        df_claim = pd.read_csv(claim_path)
        print(f"  Loaded claim_level.csv: {len(df_claim)} rows, {df_claim['paper_id'].nunique()} papers")

    # Sample B: all specs
    spec_path = DATA_DIR / "spec_level.csv"
    if not spec_path.exists():
        print(f"  WARNING: {spec_path} not found; skipping Sample B (all).")
        df_spec = pd.DataFrame()
    else:
        df_spec = pd.read_csv(spec_path)
        print(f"  Loaded spec_level.csv: {len(df_spec)} rows, {df_spec['paper_id'].nunique()} papers")

    # Sample B: verified core
    vc_path = DATA_DIR / "spec_level_verified_core.csv"
    if not vc_path.exists():
        print(f"  WARNING: {vc_path} not found; skipping Sample B (verified core).")
        df_vc = pd.DataFrame()
    else:
        df_vc = pd.read_csv(vc_path)
        print(f"  Loaded spec_level_verified_core.csv: {len(df_vc)} rows, {df_vc['paper_id'].nunique()} papers")

    # I4R comparison (informational)
    i4r_path = DATA_DIR / "i4r_comparison.csv"
    if not i4r_path.exists():
        print(f"  WARNING: {i4r_path} not found; skipping I4R comparison info.")
        df_i4r = pd.DataFrame()
    else:
        df_i4r = pd.read_csv(i4r_path)
        print(f"  Loaded i4r_comparison.csv: {len(df_i4r)} rows")

    # ------------------------------------------------------------------
    # Compute statistics
    # ------------------------------------------------------------------
    print("\nComputing summary statistics...")

    # Sample A uses t_AI_abs as the |Z| measure
    stats_a = compute_stats(df_claim, z_col="t_AI_abs", label="Sample A (claim level)") if len(df_claim) > 0 else {}

    # Sample B uses Z_abs
    stats_b_all = compute_stats(df_spec, z_col="Z_abs", label="Sample B (all specs)") if len(df_spec) > 0 else {}
    stats_b_vc = compute_stats(df_vc, z_col="Z_abs", label="Sample B (verified core)") if len(df_vc) > 0 else {}

    # ------------------------------------------------------------------
    # Write JSON
    # ------------------------------------------------------------------
    out = {
        "sample_a_claim_level": stats_a,
        "sample_b_all_specs": stats_b_all,
        "sample_b_verified_core": stats_b_vc,
    }

    # Add I4R comparison summary if available
    if len(df_i4r) > 0:
        out["i4r_comparison"] = {
            "n_papers": int(df_i4r["paper_id"].nunique()),
            "t_orig_mean": float(df_i4r["t_orig"].mean()),
            "t_i4r_mean": float(df_i4r["t_i4r"].mean()),
            "t_AI_abs_mean": float(df_i4r["t_AI_abs"].mean()),
            "agreement_counts": df_i4r["agreement_status"].value_counts().to_dict(),
        }

    json_path = RESULTS_DIR / "summary_statistics.json"
    json_path.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(f"  wrote {json_path}")

    # ------------------------------------------------------------------
    # Write LaTeX table
    # ------------------------------------------------------------------
    if stats_a and stats_b_all and stats_b_vc:
        tex = build_latex_table(stats_a, stats_b_all, stats_b_vc)
        tex_path = OL_TABLE_DIR / "tab_summary_statistics.tex"
        _write(tex_path, tex)
    else:
        print("  WARNING: insufficient data to produce LaTeX table.")

    print("\nDone.")


if __name__ == "__main__":
    main()
