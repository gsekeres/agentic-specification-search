#!/usr/bin/env python3
"""
27_mixture_comparison_table.py
==============================

Build a LaTeX comparison table of all mixture model specifications.
Reads from mixture_params_abs_t.json and produces tab_mixture_comparison.tex.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "estimation" / "results"
OVERLEAF = Path(__file__).resolve().parents[3] / "overleaf" / "tex" / "v8_tables"


def main():
    with open(RESULTS / "mixture_params_abs_t.json") as f:
        d = json.load(f)

    mu_free = d["mu_free_sigma1_comparison"]
    grid = d["systematic_grid"]

    # Collect rows: (label, K, distribution, sigma, sample, params_dict)
    rows = []

    # --- Panel A: K comparison (folded, sigma=1) ---
    for K in [2, 3, 4]:
        key = f"foldnorm_K={K}_trim10"
        r = mu_free[key]
        rows.append(("K_panel", K, "Folded", r"$\sigma=1$", r"$|t|\le 10$", r))

    # --- Panel B: Distribution comparison (K=3, sigma=1, trim10) ---
    rows.append(("dist_panel", 3, "Truncated", r"$\sigma=1$", r"$|t|\le 10$",
                 mu_free["truncnorm_K=3_trim10"]))

    # --- Panel C: Sigma comparison (K=3, truncated, trim10) ---
    rows.append(("sigma_panel", 3, "Truncated", "Free", r"$|t|\le 10$",
                 grid["K=3_sigma=free_trim10"]))
    rows.append(("sigma_panel", 3, "Truncated", r"$\sigma\ge 1$", r"$|t|\le 10$",
                 grid["K=3_sigma=geq_1_trim10"]))

    # --- Panel D: Sample comparison (folded, K=3, sigma=1) ---
    rows.append(("sample_panel", 3, "Folded", r"$\sigma=1$", "Full",
                 mu_free["foldnorm_K=3_full"]))

    def fmt(x, nd=1):
        if x is None:
            return "---"
        return f"{x:,.{nd}f}"

    def fmt_params(r):
        """Extract pi and mu for up to 4 components."""
        pi = r["pi"]
        mu = r["mu"]
        labels = list(pi.keys())
        K = len(labels)
        # For K=2: Low, High -> show as N, M
        parts_pi = []
        parts_mu = []
        for l in labels:
            parts_pi.append(f"{pi[l]:.2f}")
            parts_mu.append(f"{mu[l]:.1f}")
        # Pad to 3 columns
        while len(parts_pi) < 3:
            parts_pi.append("---")
            parts_mu.append("---")
        return parts_pi[:3], parts_mu[:3]

    # Build table
    lines = []
    lines.append(r"\begin{tabular}{clll rrr rrr rr}")
    lines.append(r"\toprule")
    lines.append(r"& & & & \multicolumn{3}{c}{Weights $\hat\pi_k$} & \multicolumn{3}{c}{Means $\hat\mu_k$} & & \\")
    lines.append(r"\cmidrule(lr){5-7}\cmidrule(lr){8-10}")
    lines.append(r"$K$ & Family & $\sigma$ & Sample & $N$ & $M$ & $E$ & $N$ & $M$ & $E$ & AIC & BIC \\")
    lines.append(r"\midrule")

    # Panel A header
    lines.append(r"\multicolumn{12}{l}{\emph{Panel A: Number of components}} \\")
    lines.append(r"\addlinespace")

    prev_panel = None
    for panel, K, dist, sigma, sample, r in rows:
        if panel != prev_panel and prev_panel is not None:
            if panel == "dist_panel":
                lines.append(r"\addlinespace")
                lines.append(r"\multicolumn{12}{l}{\emph{Panel B: Distributional family}} \\")
                lines.append(r"\addlinespace")
            elif panel == "sigma_panel":
                lines.append(r"\addlinespace")
                lines.append(r"\multicolumn{12}{l}{\emph{Panel C: Variance constraint}} \\")
                lines.append(r"\addlinespace")
            elif panel == "sample_panel":
                lines.append(r"\addlinespace")
                lines.append(r"\multicolumn{12}{l}{\emph{Panel D: Evidence window}} \\")
                lines.append(r"\addlinespace")
        prev_panel = panel

        pis, mus = fmt_params(r)
        aic = r["aic"]
        bic = r["bic"]

        # Mark the main spec with bold
        is_main = (panel == "K_panel" and K == 3)
        if is_main:
            line = rf"\textbf{{{K}}} & \textbf{{{dist}}} & \textbf{{{sigma}}} & \textbf{{{sample}}}"
            line += f" & \\textbf{{{pis[0]}}} & \\textbf{{{pis[1]}}} & \\textbf{{{pis[2]}}}"
            line += f" & \\textbf{{{mus[0]}}} & \\textbf{{{mus[1]}}} & \\textbf{{{mus[2]}}}"
            line += rf" & \textbf{{{fmt(aic)}}} & \textbf{{{fmt(bic)}}} \\"
        else:
            line = f"{K} & {dist} & {sigma} & {sample}"
            line += f" & {pis[0]} & {pis[1]} & {pis[2]}"
            line += f" & {mus[0]} & {mus[1]} & {mus[2]}"
            line += f" & {fmt(aic)} & {fmt(bic)} \\\\"

        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    tex = "\n".join(lines) + "\n"
    out_path = OVERLEAF / "tab_mixture_comparison.tex"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tex)
    print(f"Wrote {out_path}")

    # Also show a text summary
    print("\nModel comparison:")
    print(f"{'K':>2} {'Dist':<10} {'sigma':<8} {'Sample':<10} {'AIC':>10} {'BIC':>10}")
    print("-" * 60)
    for panel, K, dist, sigma, sample, r in rows:
        print(f"{K:>2} {dist:<10} {sigma:<8} {sample:<10} {r['aic']:>10.1f} {r['bic']:>10.1f}")


if __name__ == "__main__":
    main()
