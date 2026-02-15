#!/usr/bin/env python3
"""
27_sensitivity_tables.py
========================

Generate a comprehensive disclosure scaling table for the counterfactual appendix.

One large table with:
  - Rows: m_old values (1, 2, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
  - Columns: baseline + sensitivity variants, each showing m_new
  - Highlighted row: m_old=50

Replaces the old tab_disclosure_scaling.tex and tab_sensitivity_summary.tex.

All computations use null-only FDR (sigma=1 fixed mixture).

Reads:
  - estimation/results/counterfactual_params.json (flat structure)

Output:
  - overleaf/tex/v8_tables/tab_disclosure_scaling_full.tex
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
OL_TABLE_DIR = BASE_DIR / "overleaf" / "tex" / "v8_tables"


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def pass_prob_truncnorm(mu: float, sigma: float, z_lo: float, z_hi: float | None) -> float:
    """P(z_lo <= X <= z_hi) for X ~ TruncatedNormal(mu, sigma, 0, inf)."""
    sigma = max(sigma, 1e-8)
    denom = 1.0 - norm.cdf(-mu / sigma)
    if denom <= 0:
        return 0.0
    cdf_lo = norm.cdf((z_lo - mu) / sigma)
    if z_hi is None or z_hi == float("inf") or not np.isfinite(z_hi):
        return max(0.0, (1.0 - cdf_lo) / denom)
    cdf_hi = norm.cdf((z_hi - mu) / sigma)
    return max(0.0, (cdf_hi - cdf_lo) / denom)


def compute_pass_probs(mus, sigmas, z_lo, z_hi):
    return [pass_prob_truncnorm(mu, sig, z_lo, z_hi) for mu, sig in zip(mus, sigmas)]


def fdr_null(m: int, n_eff: int, pi: list[float], pp: list[float]) -> float:
    """Null-only FDR = pi_N * Q_N / sum(pi_k * Q_k)."""
    if n_eff < m or m < 1:
        return 1.0
    Qs = [1.0 - binom.cdf(m - 1, n_eff, pp[k]) for k in range(len(pi))]
    Qbar = sum(pi[k] * Qs[k] for k in range(len(pi)))
    if Qbar <= 0:
        return 1.0
    return (pi[0] * Qs[0]) / Qbar


def calibrate_per_m(m_old, lam, pi, pp, fdr_target=0.05):
    """For a given m_old, find n_eff_old at FDR boundary, then m_new."""
    n_eff_old = m_old
    for n in range(m_old, 100001):
        if fdr_null(m_old, n, pi, pp) > fdr_target:
            n_eff_old = n - 1
            break
    else:
        n_eff_old = 100000

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
    print("Generating comprehensive disclosure scaling table")
    print("=" * 60)

    params = load_json(RESULTS_DIR / "counterfactual_params.json")

    # Read from flat structure (no nullfdr_variant nesting)
    mix = params["mixture_params"]
    pi = [mix["pi"][k] for k in ["N", "H", "L"]]
    mus = [mix["mu"][k] for k in ["N", "H", "L"]]
    sigmas = [mix["sigma"][k] for k in ["N", "H", "L"]]

    ew = params["evidence_window"]
    z_lo_baseline = ew["z_lo"]
    z_hi_baseline = ew["z_hi"]

    lam_baseline = params["cost_parameters"]["lambda_baseline"]
    pp_baseline = compute_pass_probs(mus, sigmas, z_lo_baseline, z_hi_baseline)

    print(f"  pi = {[f'{x:.3f}' for x in pi]}")
    print(f"  pp = {[f'{x:.4f}' for x in pp_baseline]}")
    print(f"  lambda = {lam_baseline:.6f} (1/{1/lam_baseline:.0f})")
    print(f"  B = [{z_lo_baseline:.2f}, {z_hi_baseline:.1f}]")

    # ------------------------------------------------------------------
    # Define m_old rows — include empirical percentiles of specs per paper
    # ------------------------------------------------------------------
    # Percentiles of author-reported n_regressions_original (95 in-sample papers):
    # P5=0, P10=1, P25=20, P50=50, P75=108, P90=201
    # P5=0 is not meaningful for disclosure; mark the rest on table rows
    percentile_m = {1: "P10", 20: "P25", 50: "P50", 108: "P75", 201: "P90"}
    m_old_values = sorted(set([1, 2, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                               + [108, 201]))
    highlight_m = 50

    # ------------------------------------------------------------------
    # Define variants (each is a dict with label, lambda, pass_probs)
    # ------------------------------------------------------------------
    variants = []

    # Baseline
    variants.append({
        "label": "Baseline",
        "short": r"$\lambda\!=\!1/" + f"{1/lam_baseline:.0f}" + r"$",
        "lam": lam_baseline,
        "pp": pp_baseline,
    })

    # Lambda variants (round-number sensitivity)
    for lam_inv in [50, 100, 250, 500]:
        lam_v = 1.0 / lam_inv
        if abs(lam_v - lam_baseline) < 1e-8:
            continue
        variants.append({
            "label": f"lam=1/{lam_inv}",
            "short": rf"$\lambda\!=\!1/{lam_inv}$",
            "lam": lam_v,
            "pp": pp_baseline,
        })

    # Lambda quantile variants from timing data
    cost_params = params.get("cost_parameters", {})
    for pctl, pctl_label in [("lambda_p25", "P25"), ("lambda_p50", "P50"),
                              ("lambda_p75", "P75"), ("lambda_p90", "P90")]:
        lam_q = cost_params.get(pctl)
        if lam_q is not None and abs(lam_q - lam_baseline) > 1e-8:
            lam_inv_q = round(1.0 / lam_q)
            variants.append({
                "label": f"lam_{pctl_label}",
                "short": rf"$\lambda^{{{pctl_label}}}$",
                "lam": lam_q,
                "pp": pp_baseline,
            })

    # z_lo variants (keep lambda=baseline, z_hi=10)
    for z_lo_v in [1.0, 1.5, 2.5, 3.0]:
        pp_v = compute_pass_probs(mus, sigmas, z_lo_v, z_hi_baseline)
        variants.append({
            "label": f"z_lo={z_lo_v}",
            "short": rf"$z_{{\mathrm{{lo}}}}\!=\!{z_lo_v:.1f}$",
            "lam": lam_baseline,
            "pp": pp_v,
        })

    # z_hi variants
    for z_hi_v, z_hi_label in [(15.0, "15"), (None, r"\infty")]:
        pp_v = compute_pass_probs(mus, sigmas, z_lo_baseline, z_hi_v)
        variants.append({
            "label": f"z_hi={z_hi_label}",
            "short": rf"$z_{{\mathrm{{hi}}}}\!=\!{z_hi_label}$",
            "lam": lam_baseline,
            "pp": pp_v,
        })

    # Mixture variants: sigma-free
    # Load sigma-free mixture
    try:
        mix_file = load_json(RESULTS_DIR / "mixture_params_abs_t.json")
        sf = mix_file.get("spec_level", {}).get("trim_sensitivity", {}).get("trim_abs_le_10")
        if sf and "N" in sf.get("pi", {}):
            sf_pi = [sf["pi"][k] for k in ["N", "H", "L"]]
            sf_mus = [sf["mu"][k] for k in ["N", "H", "L"]]
            sf_sigmas = [sf["sigma"][k] for k in ["N", "H", "L"]]
            sf_pp = compute_pass_probs(sf_mus, sf_sigmas, z_lo_baseline, z_hi_baseline)
            variants.append({
                "label": "sigma-free",
                "short": r"$\sigma$-free",
                "lam": lam_baseline,
                "pp": sf_pp,
                "pi": sf_pi,
            })
    except Exception:
        pass

    print(f"\n  {len(variants)} variants defined")

    # ------------------------------------------------------------------
    # Compute results for each (m_old, variant)
    # ------------------------------------------------------------------
    all_results = {}
    for v in variants:
        key = v["label"]
        v_pi = v.get("pi", pi)
        all_results[key] = {}
        for m_old in m_old_values:
            row = calibrate_per_m(m_old, v["lam"], v_pi, v["pp"])
            all_results[key][m_old] = row
        # Print headline for m_old=50
        r50 = all_results[key][50]
        print(f"  {key}: m_old=50 -> m_new={r50['m_new']}, ratio={r50['ratio']:.1f}")

    # ------------------------------------------------------------------
    # Build LaTeX table
    # ------------------------------------------------------------------
    n_var = len(variants)
    col_spec = "r" + "r" * n_var  # m_old column + one column per variant

    lines = []
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header row
    hdr = r"$m^{\mathrm{old}}$"
    for v in variants:
        hdr += " & " + v["short"]
    hdr += r" \\"
    lines.append(hdr)
    lines.append(r"\midrule")

    # Data rows
    for m_old in m_old_values:
        bold = (m_old == highlight_m)
        is_pctile = m_old in percentile_m
        fmt = r"\textbf{%s}" if bold else "%s"

        # Build the m_old label, appending percentile tag if applicable
        m_label = str(m_old)
        if is_pctile:
            m_label += r"\rlap{\textsuperscript{\tiny " + percentile_m[m_old] + "}}"
        row_str = fmt % m_label

        for v in variants:
            r = all_results[v["label"]][m_old]
            m_new_str = str(r["m_new"])
            row_str += " & " + (fmt % m_new_str)
        row_str += r" \\"

        # Add a midrule before the highlighted row for visual separation
        if bold:
            lines.append(r"\midrule")
        lines.append(row_str)
        if bold:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    table_tex = "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    OL_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    p1 = OL_TABLE_DIR / "tab_disclosure_scaling_full.tex"
    with open(p1, "w") as f:
        f.write(table_tex)
    print(f"\n  Saved {p1}")

    # Also save the old files for backward compatibility
    # tab_disclosure_scaling.tex — just baseline, first 10 m_old values
    scaling_lines = [r"\begin{tabular}{ccccc}", r"\toprule",
                     r"$m^{\mathrm{old}}$ & $n_{\mathrm{eff}}^{\mathrm{old}}$ & "
                     r"$m^{\mathrm{new}}$ & $n_{\mathrm{eff}}^{\mathrm{new}}$ & "
                     r"$m^{\mathrm{new}}/m^{\mathrm{old}}$ \\",
                     r"\midrule"]
    for m_old in range(1, 11):
        row = calibrate_per_m(m_old, lam_baseline, pi, pp_baseline)
        bold = (m_old == highlight_m)
        fmt = r"\textbf{%s}" if bold else "%s"
        vals = [fmt % str(row["m_old"]), fmt % str(row["n_eff_old"]),
                fmt % str(row["m_new"]), fmt % str(row["n_eff_new"]),
                fmt % f"{row['ratio']:.1f}"]
        scaling_lines.append(" & ".join(vals) + r" \\")
    scaling_lines += [r"\bottomrule", r"\end{tabular}"]

    p2 = OL_TABLE_DIR / "tab_disclosure_scaling.tex"
    with open(p2, "w") as f:
        f.write("\n".join(scaling_lines) + "\n")
    print(f"  Saved {p2}")

    print("\nDone!")


if __name__ == "__main__":
    main()
