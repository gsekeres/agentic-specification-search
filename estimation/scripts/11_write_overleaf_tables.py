#!/usr/bin/env python3
"""
11_write_overleaf_tables.py
===========================

Write manuscript-ready LaTeX tabulars used in Appendix~G.

Reads:
  - estimation/results/mixture_params.json
  - estimation/results/mixture_params_abs_t.json
  - estimation/results/dependence.json
  - estimation/results/counterfactual.csv
  - estimation/results/counterfactual_dependence_sensitivity.csv
  - estimation/results/inference_audit_i4r.csv

Writes:
  - overleaf/tex/v8_tables/tab_mixture_params_z.tex
  - overleaf/tex/v8_tables/tab_mixture_params_abs_t.tex
  - overleaf/tex/v8_tables/tab_dependence_summary.tex
  - overleaf/tex/v8_tables/tab_counterfactual_sensitivity.tex
  - overleaf/tex/v8_tables/tab_inference_audit_i4r.tex
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _fmt(x: float, nd: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return ""
    return f"{float(x):.{nd}f}"


def _fmt_int(x: float | int) -> str:
    try:
        return str(int(x))
    except Exception:
        return ""


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n")


def write_mixture_table(params: dict, out_path: Path) -> None:
    n = int(params.get("n_obs", 0))
    bic = float(params.get("bic", np.nan))
    pi = params.get("pi", {})
    mu = params.get("mu", {})
    sigma = params.get("sigma", {})

    line = " & ".join(
        [
            _fmt_int(n),
            _fmt(pi.get("N")),
            _fmt(pi.get("H")),
            _fmt(pi.get("L")),
            _fmt(mu.get("N")),
            _fmt(mu.get("H")),
            _fmt(mu.get("L")),
            _fmt(sigma.get("N")),
            _fmt(sigma.get("H")),
            _fmt(sigma.get("L")),
            f"{bic:.1f}" if np.isfinite(bic) else "",
        ]
    )

    tab = rf"""\begin{{tabular}}{{ccccccccccc}}
\toprule
$n$ & $\hat\pi_N$ & $\hat\pi_M$ & $\hat\pi_E$ & $\hat\mu_N$ & $\hat\mu_M$ & $\hat\mu_E$ & $\hat\sigma_N$ & $\hat\sigma_M$ & $\hat\sigma_E$ & BIC \\
\midrule
{line} \\
\bottomrule
\end{{tabular}}"""
    _write(out_path, tab)


def write_dependence_table(dep: dict, out_path: Path) -> None:
    dist = dep.get("distance_based", {})
    decay = dist.get("decay_fit", {})
    ar1 = dep.get("ar1", {}).get("pooled", {})

    rows: list[tuple[str, float, float, float | None, float | None]] = []
    if decay:
        phi = float(decay.get("phi", np.nan))
        Delta = float(dist.get("Delta", dep.get("preferred", {}).get("Delta", np.nan)))
        rows.append(
            (
                "Distance-based",
                phi,
                Delta,
                decay.get("phi_ci_lower", None),
                decay.get("phi_ci_upper", None),
            )
        )
    if ar1:
        phi = float(ar1.get("phi", np.nan))
        Delta = float(ar1.get("Delta", np.nan))
        rows.append(
            (
                "AR(1) traversal",
                phi,
                Delta,
                ar1.get("phi_ci_lower", None),
                ar1.get("phi_ci_upper", None),
            )
        )

    body = []
    for name, phi, Delta, lo, hi in rows:
        ci = ""
        if lo is not None and hi is not None and np.isfinite(float(lo)) and np.isfinite(float(hi)):
            ci = rf"[{_fmt(float(lo), 3)}, {_fmt(float(hi), 3)}]"
        body.append(" & ".join([name, _fmt(phi, 3), _fmt(Delta, 3), ci]) + r" \\")

    tab = rf"""\begin{{tabular}}{{lccc}}
\toprule
Method & $\hat\phi$ & $\widehat\Delta=1-\hat\phi$ & 95\% CI for $\hat\phi$ \\
\midrule
{chr(10).join(body)}
\bottomrule
\end{{tabular}}"""
    _write(out_path, tab)


def write_counterfactual_table(
    cf_path: Path, cf_dep_sens_path: Path, out_path: Path, *, lambda_mid: float = 1 / 14
) -> None:
    cf = pd.read_csv(cf_path)
    dep = pd.read_csv(cf_dep_sens_path)

    # Panel A: lambda sensitivity (preferred dependence; rho=0.10, FDR=0.10)
    panelA = cf[(cf["rho_target"] == 0.10) & (cf["FDR_target"] == 0.10)].copy()
    panelA = panelA.sort_values("lambda")

    # Panel B: dependence sensitivity at lambda=1/14 (rho=0.10, FDR=0.10)
    panelB = dep[(dep["rho_target"] == 0.10) & (dep["FDR_target"] == 0.10)].copy()
    panelB = panelB[np.isclose(panelB["lambda"], float(lambda_mid))]
    # Drop redundant duplicate rows (preferred and distance_based are identical in the current build)
    panelB = panelB.drop_duplicates(subset=["Delta", "m_old", "m_new", "m_ratio"])
    panelB = panelB.sort_values("Delta")

    # Panel C: target sensitivity at lambda=1/14, preferred dependence (distance_based == preferred here)
    panelC = cf[np.isclose(cf["lambda"], float(lambda_mid))].copy()
    panelC = panelC.sort_values(["FDR_target", "rho_target"])

    def _row_A(r: pd.Series) -> str:
        lam = float(r["lambda"])
        inv = 1.0 / lam if lam > 0 else np.nan
        return " & ".join(
            [
                _fmt(lam, 3),
                _fmt(inv, 1),
                _fmt_int(r["m_old"]),
                _fmt_int(r["m_new"]),
                _fmt(r.get("m_ratio", np.nan), 2),
            ]
        )

    def _row_B(r: pd.Series) -> str:
        lab = str(r["dependence_label"])
        # prettier labels
        lab_map = {
            "preferred": "Preferred (distance-based)",
            "distance_based": "Distance-based",
            "ar1_ci_low_Delta": "AR(1) CI (low $\\Delta$)",
            "ar1_ci_high_Delta": "AR(1) CI (high $\\Delta$)",
        }
        lab = lab_map.get(lab, lab)
        return " & ".join([lab, _fmt(r["Delta"], 3), _fmt_int(r["m_old"]), _fmt_int(r["m_new"]), _fmt(r["m_ratio"], 2)])

    def _m_new(rho: float, fdr: float) -> str:
        sub = panelC[(panelC["rho_target"] == rho) & (panelC["FDR_target"] == fdr)]
        if len(sub) != 1:
            return ""
        return _fmt_int(sub["m_new"].iloc[0])

    def _m_old(rho: float, fdr: float) -> str:
        sub = panelC[(panelC["rho_target"] == rho) & (panelC["FDR_target"] == fdr)]
        if len(sub) != 1:
            return ""
        return _fmt_int(sub["m_old"].iloc[0])

    panelA_lines = [(_row_A(r) + r" \\") for _, r in panelA.iterrows()]
    panelB_lines = [(_row_B(r) + r" \\") for _, r in panelB.iterrows()]

    # Build panel C matrix (report m_new at each rho; include m_old in last col for reference)
    rhos = [0.05, 0.10, 0.20]
    fdrs = [0.05, 0.10, 0.20]
    panelC_lines = []
    for fdr in fdrs:
        row = [
            _fmt(fdr, 2),
            _m_new(rhos[0], fdr),
            _m_new(rhos[1], fdr),
            _m_new(rhos[2], fdr),
            _m_old(rhos[1], fdr),
        ]
        panelC_lines.append(" & ".join(row) + r" \\")

    tab = rf"""\begin{{tabular}}{{lcccc}}
\toprule
\multicolumn{{5}}{{l}}{{\emph{{A. Cost ratio sensitivity (preferred dependence; $\bar\rho=0.10$, FDR target $=0.10$)}}}} \\
\midrule
$\lambda$ & $1/\lambda$ & $m^{{\mathrm{{old}}}}$ & $m^{{\mathrm{{new}}}}$ & $m^{{\mathrm{{new}}}}/m^{{\mathrm{{old}}}}$ \\
{chr(10).join(panelA_lines)}
\midrule
\multicolumn{{5}}{{l}}{{\emph{{B. Dependence sensitivity ($\lambda=1/14$; $\bar\rho=0.10$, FDR target $=0.10$)}}}} \\
\midrule
Dependence estimate & $\widehat\Delta$ & $m^{{\mathrm{{old}}}}$ & $m^{{\mathrm{{new}}}}$ & $m^{{\mathrm{{new}}}}/m^{{\mathrm{{old}}}}$ \\
{chr(10).join(panelB_lines)}
\midrule
\multicolumn{{5}}{{l}}{{\emph{{C. Target sensitivity ($\lambda=1/14$; preferred dependence)}}}} \\
\midrule
FDR target & $m^{{\mathrm{{new}}}}(\bar\rho=0.05)$ & $m^{{\mathrm{{new}}}}(\bar\rho=0.10)$ & $m^{{\mathrm{{new}}}}(\bar\rho=0.20)$ & $m^{{\mathrm{{old}}}}$ \\
{chr(10).join(panelC_lines)}
\bottomrule
\end{{tabular}}"""
    _write(out_path, tab)


def write_inference_audit_table(audit_path: Path, out_path: Path) -> None:
    df = pd.read_csv(audit_path)
    if "suspected_cluster_ignored_in_sm_branch" not in df.columns:
        raise RuntimeError("inference_audit_i4r.csv missing suspected_cluster_ignored_in_sm_branch.")

    flagged = df[pd.to_numeric(df["suspected_cluster_ignored_in_sm_branch"], errors="coerce").fillna(0).astype(int) == 1].copy()
    flagged = flagged.sort_values("paper_id")

    if len(flagged) == 0:
        tab = r"""\begin{tabular}{lc}
\toprule
Paper ID & Flagged \\
\midrule
(none) & 0 \\
\bottomrule
\end{tabular}"""
        _write(out_path, tab)
        return

    lines = []
    for _, r in flagged.iterrows():
        pid = str(r.get("paper_id", ""))
        path = str(r.get("path", ""))
        short = Path(path).name if path else ""
        note = "Potential clustering omission in statsmodels fallback"
        lines.append(" & ".join([pid, rf"\texttt{{{short}}}", note]) + r" \\")

    tab = rf"""\begin{{tabular}}{{lll}}
\toprule
Paper ID & Script & Flag description \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}"""
    _write(out_path, tab)


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    results_dir = base_dir / "estimation" / "results"
    overleaf_tables = base_dir.parent / "overleaf" / "tex" / "v8_tables"
    overleaf_tables.mkdir(parents=True, exist_ok=True)

    mix = json.loads((results_dir / "mixture_params.json").read_text())
    mix_abs = json.loads((results_dir / "mixture_params_abs_t.json").read_text())
    dep = json.loads((results_dir / "dependence.json").read_text())

    # Prefer sigma=1 fixed gamma fit; fall back to sigma-free baseline.
    mix_params = mix.get("spec_level", {}).get("baseline_only_sigma_fixed_1", None)
    if mix_params is None:
        mix_params = mix.get("spec_level", {}).get("baseline_only", None)
    if mix_params is None:
        raise RuntimeError("mixture_params.json missing sigma-fixed or baseline gamma mixture results.")
    # Prefer sigma=1 fixed (trimmed |Z|<=10) as baseline; fall back to sigma-free.
    mix_abs_params = mix_abs.get("spec_level", {}).get("trim_sensitivity", {}).get("trim_abs_le_10_sigma_fixed_1", None)
    if mix_abs_params is None:
        mix_abs_params = mix_abs.get("spec_level", {}).get("baseline_only", None)
    if mix_abs_params is None:
        raise RuntimeError("mixture_params_abs_t.json missing sigma-fixed or baseline mixture results.")

    write_mixture_table(mix_params, overleaf_tables / "tab_mixture_params_z.tex")
    write_mixture_table(mix_abs_params, overleaf_tables / "tab_mixture_params_abs_t.tex")
    write_dependence_table(dep, overleaf_tables / "tab_dependence_summary.tex")

    write_counterfactual_table(
        results_dir / "counterfactual.csv",
        results_dir / "counterfactual_dependence_sensitivity.csv",
        overleaf_tables / "tab_counterfactual_sensitivity.tex",
    )

    write_inference_audit_table(results_dir / "inference_audit_i4r.csv", overleaf_tables / "tab_inference_audit_i4r.tex")


if __name__ == "__main__":
    main()
