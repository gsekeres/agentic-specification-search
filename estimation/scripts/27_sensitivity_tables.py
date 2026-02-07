#!/usr/bin/env python3
"""
27_sensitivity_tables.py
========================

Generate two LaTeX tables for the counterfactual screening appendix:
  1. tab_disclosure_scaling.tex -- per-m_old disclosure scaling (baseline lambda=1/14, B=[1.96,10])
  2. tab_sensitivity_summary.tex -- per-m_old scaling under each B and lambda variant

All computations use the null-only FDR variant (sigma=1 fixed).

Reads:
  - estimation/results/counterfactual_params.json (nullfdr_variant)

Output:
  - overleaf/tex/v8_tables/tab_disclosure_scaling.tex
  - overleaf/tex/v8_tables/tab_sensitivity_summary.tex
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binom, norm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / "estimation" / "results"
OL_TABLE_DIR = (
    Path(__file__).parent.parent.parent.parent / "overleaf" / "tex" / "v8_tables"
)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def pass_prob_truncnorm(mu: float, sigma: float, z_lo: float, z_hi: float | None) -> float:
    """P(z_lo <= X <= z_hi) for X ~ TruncatedNormal(mu, sigma, 0, inf)."""
    denom = 1.0 - norm.cdf(-mu / sigma)
    if denom <= 0:
        return 0.0
    cdf_lo = norm.cdf((z_lo - mu) / sigma)
    if z_hi is None or z_hi == float("inf"):
        return (1.0 - cdf_lo) / denom
    cdf_hi = norm.cdf((z_hi - mu) / sigma)
    return max(0.0, (cdf_hi - cdf_lo) / denom)


def compute_pass_probs(mus, sigmas, z_lo, z_hi):
    return [pass_prob_truncnorm(mu, sig, z_lo, z_hi) for mu, sig in zip(mus, sigmas)]


def fdr_null(m: int, n_eff: int, pi: list[float], pp: list[float]) -> float:
    """Null-only FDR = pi_N * Q_N / sum(pi_k * Q_k)."""
    if n_eff < m or m < 1:
        return 1.0
    Qs = [1.0 - binom.cdf(m - 1, n_eff, pp[k]) for k in range(3)]
    Qbar = sum(pi[k] * Qs[k] for k in range(3))
    if Qbar <= 0:
        return 1.0
    return (pi[0] * Qs[0]) / Qbar


def calibrate_per_m(m_old, lam, pi, pp, fdr_target=0.05):
    """For a given m_old, find n_eff_old at FDR boundary, then m_new."""
    n_eff_old = m_old
    for n in range(m_old, 50000):
        if fdr_null(m_old, n, pi, pp) > fdr_target:
            n_eff_old = n - 1
            break
    else:
        n_eff_old = 50000

    n_eff_new = int(np.ceil(n_eff_old / lam))

    m_new = n_eff_new
    for mc in range(1, n_eff_new + 1):
        if fdr_null(mc, n_eff_new, pi, pp) <= fdr_target:
            m_new = mc
            break

    ratio = m_new / m_old if m_old > 0 else float("inf")
    return {"m_old": m_old, "n_eff_old": n_eff_old, "m_new": m_new,
            "n_eff_new": n_eff_new, "ratio": ratio}


def main() -> None:
    print("=" * 60)
    print("Generating sensitivity tables (null-only FDR, sigma=1)")
    print("=" * 60)

    params = load_json(RESULTS_DIR / "counterfactual_params.json")
    nv = params["nullfdr_variant"]
    pi = [nv["mixture_params"]["pi"][k] for k in ["N", "H", "L"]]
    mus = [nv["mixture_params"]["mu"][k] for k in ["N", "H", "L"]]
    sigmas = [nv["mixture_params"]["sigma"][k] for k in ["N", "H", "L"]]
    pp_baseline = [nv["pass_probabilities"][k] for k in ["N", "H", "L"]]
    lam_baseline = params["cost_parameters"]["lambda_baseline"]
    z_lo = nv["evidence_window"]["z_lo"]
    z_hi = nv["evidence_window"]["z_hi"]

    print(f"  pi = {[f'{x:.3f}' for x in pi]}")
    print(f"  pp = {[f'{x:.4f}' for x in pp_baseline]}")
    print(f"  lambda = {lam_baseline:.4f}, B = [{z_lo:.2f}, {z_hi:.1f}]")

    # ==================================================================
    # Table 1: Baseline disclosure scaling (lambda=1/14, B=[1.96,10])
    # ==================================================================
    print("\n--- Disclosure Scaling (baseline) ---")
    scaling_rows = []
    for m_old in range(1, 11):
        row = calibrate_per_m(m_old, lam_baseline, pi, pp_baseline)
        scaling_rows.append(row)
        print(f"  m_old={m_old}: m_new={row['m_new']}, ratio={row['ratio']:.1f}")

    lines = [r"\begin{tabular}{ccccc}", r"\toprule",
             r"$m^{\mathrm{old}}$ & $n_{\mathrm{eff}}^{\mathrm{old}}$ & "
             r"$m^{\mathrm{new}}$ & $n_{\mathrm{eff}}^{\mathrm{new}}$ & "
             r"$m^{\mathrm{new}}/m^{\mathrm{old}}$ \\",
             r"\midrule"]
    for row in scaling_rows:
        bold = row["m_old"] == 3
        fmt = r"\textbf{%s}" if bold else "%s"
        vals = [fmt % str(row["m_old"]), fmt % str(row["n_eff_old"]),
                fmt % str(row["m_new"]), fmt % str(row["n_eff_new"]),
                fmt % f"{row['ratio']:.1f}"]
        lines.append(" & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    scaling_tex = "\n".join(lines) + "\n"

    # ==================================================================
    # Table 2: Per-m scaling under each B and lambda variant
    # ==================================================================
    print("\n--- Sensitivity Summary (per-m, all variants) ---")

    # Define variants
    variants = []

    # B variants (keep lambda=1/14)
    for z_hi_var, label in [(10.0, r"$B=[1.96,10]$"),
                            (15.0, r"$B=[1.96,15]$"),
                            (None, r"$B=[1.96,\infty)$")]:
        pp_var = compute_pass_probs(mus, sigmas, z_lo, z_hi_var)
        variants.append({"label": label, "lam": lam_baseline, "pp": pp_var,
                         "type": "B"})
        zstr = str(z_hi_var) if z_hi_var else "inf"
        print(f"  B variant z_hi={zstr}: pp = {[f'{x:.4f}' for x in pp_var]}")

    # Lambda variants (keep B=[1.96,10])
    for lam_inv, label in [(30, r"$\lambda=1/30$"),
                           (21, r"$\lambda=1/21$"),
                           (14, r"$\lambda=1/14$"),
                           (7, r"$\lambda=1/7$")]:
        variants.append({"label": label, "lam": 1.0 / lam_inv,
                         "pp": pp_baseline, "type": "lam"})

    # Compute per-m results for each variant
    all_results = {}
    for v in variants:
        key = v["label"]
        all_results[key] = []
        for m_old in range(1, 11):
            row = calibrate_per_m(m_old, v["lam"], pi, v["pp"])
            all_results[key].append(row)
        r3 = all_results[key][2]  # m_old=3
        print(f"  {key}: m_old=3 -> m_new={r3['m_new']}, ratio={r3['ratio']:.1f}")

    # Build LaTeX: one panel for B variants, one for lambda variants
    b_variants = [v for v in variants if v["type"] == "B"]
    l_variants = [v for v in variants if v["type"] == "lam"]

    def make_panel(panel_variants, panel_label):
        n_var = len(panel_variants)
        col_spec = "c" + "cc" * n_var
        slines = []
        slines.append(r"\begin{tabular}{" + col_spec + "}")
        slines.append(r"\toprule")

        # Header row 1: variant labels spanning 2 columns each
        hdr1 = r" & "
        for i, v in enumerate(panel_variants):
            hdr1 += r"\multicolumn{2}{c}{" + v["label"] + "}"
            if i < n_var - 1:
                hdr1 += " & "
        hdr1 += r" \\"
        slines.append(hdr1)

        # cmidrules
        for i in range(n_var):
            start = 2 + 2 * i
            end = start + 1
            slines.append(r"\cmidrule(lr){" + f"{start}-{end}" + "}")

        # Header row 2: m_old | m_new ratio | m_new ratio | ...
        hdr2 = r"$m^{\mathrm{old}}$"
        for _ in panel_variants:
            hdr2 += r" & $m^{\mathrm{new}}$ & Ratio"
        hdr2 += r" \\"
        slines.append(hdr2)
        slines.append(r"\midrule")

        # Data rows
        for m_idx in range(10):
            m_old = m_idx + 1
            bold = m_old == 3
            fmt = r"\textbf{%s}" if bold else "%s"
            row_str = fmt % str(m_old)
            for v in panel_variants:
                r = all_results[v["label"]][m_idx]
                row_str += " & " + fmt % str(r["m_new"])
                row_str += " & " + fmt % f"{r['ratio']:.1f}"
            row_str += r" \\"
            slines.append(row_str)

        slines += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(slines)

    # Combine into one table with two panels
    out_lines = []
    out_lines.append(make_panel(b_variants, "A"))
    out_lines.append("")
    out_lines.append(r"\vspace{0.5cm}")
    out_lines.append("")
    out_lines.append(make_panel(l_variants, "B"))

    summary_tex = "\n".join(out_lines) + "\n"

    # ==================================================================
    # Save
    # ==================================================================
    OL_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    p1 = OL_TABLE_DIR / "tab_disclosure_scaling.tex"
    with open(p1, "w") as f:
        f.write(scaling_tex)
    print(f"\n  Saved {p1}")

    p2 = OL_TABLE_DIR / "tab_sensitivity_summary.tex"
    with open(p2, "w") as f:
        f.write(summary_tex)
    print(f"  Saved {p2}")

    print("\nDone!")


if __name__ == "__main__":
    main()
