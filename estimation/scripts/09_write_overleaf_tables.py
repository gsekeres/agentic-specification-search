#!/usr/bin/env python3
"""
11_write_overleaf_tables.py
===========================

Write manuscript-ready LaTeX tabulars used in Appendix~G.

Reads:
  - estimation/results/mixture_params_abs_t.json
  - estimation/results/dependence.json
  - estimation/results/counterfactual.csv
  - estimation/results/counterfactual_dependence_sensitivity.csv
  - estimation/results/counterfactual_params.json
  - estimation/results/inference_audit_i4r.csv

Writes:
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
    ar1_ords = dep.get("ar1_orderings", {})
    pref_ordering = dep.get("preferred", {}).get("ordering", "")

    label_map = {
        "spec_order": "Document order",
        "lex_path": "Lexicographic path",
        "bfs": "Breadth-first",
        "dfs": "Depth-first",
        "by_category": "By category",
        "random": "Random (null)",
    }
    # Display order: preferred first, then remaining (excl. random), then random
    display_order = []
    if pref_ordering in ar1_ords:
        display_order.append(pref_ordering)
    for k in ["spec_order", "lex_path", "bfs", "dfs", "by_category"]:
        if k in ar1_ords and k != pref_ordering:
            display_order.append(k)
    if "random" in ar1_ords:
        display_order.append("random")

    body = []
    for k in display_order:
        v = ar1_ords[k]
        phi = float(v.get("phi", np.nan))
        Delta = float(v.get("Delta", np.nan))
        lo = v.get("phi_ci_lower", None)
        hi = v.get("phi_ci_upper", None)
        r2 = float(v.get("r_squared", np.nan))
        ci = ""
        if lo is not None and hi is not None and np.isfinite(float(lo)) and np.isfinite(float(hi)):
            ci = rf"[{_fmt(float(lo), 3)}, {_fmt(float(hi), 3)}]"
        label = label_map.get(k, k)
        if k == pref_ordering:
            label = rf"\textbf{{{label}}}"
        body.append(" & ".join([label, _fmt(phi, 3), _fmt(Delta, 3), ci, _fmt(r2, 3)]) + r" \\")

    tab = rf"""\begin{{tabular}}{{lcccc}}
\toprule
Ordering & $\hat\phi$ & $\widehat\Delta=1-\hat\phi$ & 95\% CI for $\hat\phi$ & $R^2$ \\
\midrule
{chr(10).join(body)}
\bottomrule
\end{{tabular}}"""
    _write(out_path, tab)


def write_counterfactual_table(
    cf_path: Path, cf_dep_sens_path: Path, out_path: Path, *,
    params_path: Path | None = None,
) -> None:
    # Read lambda from counterfactual_params.json (flat structure)
    lambda_mid = None
    if params_path is not None and params_path.exists():
        _par = json.loads(params_path.read_text())
        lambda_mid = _par.get("cost_parameters", {}).get("lambda_baseline")
    if lambda_mid is None:
        lambda_mid = 1 / 170

    cf = pd.read_csv(cf_path)
    dep = pd.read_csv(cf_dep_sens_path)

    # Panel A: lambda sensitivity at FDR=0.05
    panelA = cf[cf["FDR_target"] == 0.05].copy() if "FDR_target" in cf.columns else cf.copy()
    panelA = panelA.sort_values("lambda")

    # Panel B: dependence sensitivity — all 5 non-random orderings
    panelB = dep.copy()
    if "lambda" in panelB.columns:
        panelB = panelB[np.isclose(panelB["lambda"], float(lambda_mid))]
    # Keep only actual orderings (not CI bounds)
    ordering_labels = ["spec_order", "lex_path", "bfs", "dfs", "by_category"]
    if "dependence_label" in panelB.columns:
        panelB = panelB[panelB["dependence_label"].isin(ordering_labels)]
    panelB = panelB.drop_duplicates(subset=["dependence_label"])
    panelB = panelB.sort_values("Delta")

    def _row_A(r: pd.Series) -> str:
        lam = float(r["lambda"])
        inv = 1.0 / lam if lam > 0 else np.nan
        m_ratio = float(r.get("m_ratio", np.nan))
        return " & ".join(
            [
                _fmt(lam, 4),
                _fmt(inv, 0),
                _fmt_int(r["m_old"]),
                _fmt_int(r["m_new"]),
                _fmt(m_ratio, 2),
            ]
        )

    # Prettier labels for orderings
    label_map = {
        "spec_order": "Document order",
        "lex_path": "Lexicographic path",
        "bfs": "Breadth-first",
        "dfs": "Depth-first",
        "by_category": r"\textbf{By category}",
    }

    def _row_B(r: pd.Series) -> str:
        lab = str(r["dependence_label"])
        lab = label_map.get(lab, lab)
        Delta = float(r.get("Delta", np.nan))
        n_old = r.get("n_old_implied", np.nan)
        n_old_str = _fmt(float(n_old), 1) if np.isfinite(float(n_old)) else ""
        return " & ".join([lab, _fmt(Delta, 3), n_old_str, _fmt_int(r["m_old"]), _fmt_int(r["m_new"])])

    panelA_lines = [(_row_A(r) + r" \\") for _, r in panelA.iterrows()]
    panelB_lines = [(_row_B(r) + r" \\") for _, r in panelB.iterrows()]

    tab = rf"""\begin{{tabular}}{{lcccc}}
\toprule
\multicolumn{{5}}{{l}}{{\emph{{A. Cost ratio sensitivity (FDR target $=0.05$, $m^{{\mathrm{{old}}}}=50$)}}}} \\
\midrule
$\lambda$ & $1/\lambda$ & $m^{{\mathrm{{old}}}}$ & $m^{{\mathrm{{new}}}}$ & $m^{{\mathrm{{new}}}}/m^{{\mathrm{{old}}}}$ \\
{chr(10).join(panelA_lines)}
\midrule
\multicolumn{{5}}{{l}}{{\emph{{B. Dependence sensitivity (interpretation: implied $n^{{\mathrm{{old}}}}$)}}}} \\
\midrule
Ordering & $\widehat\Delta$ & $n^{{\mathrm{{old}}}}_{{implied}}$ & $m^{{\mathrm{{old}}}}$ & $m^{{\mathrm{{new}}}}$ \\
{chr(10).join(panelB_lines)}
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
    overleaf_tables = base_dir / "overleaf" / "tex" / "v8_tables"
    overleaf_tables.mkdir(parents=True, exist_ok=True)

    mix_abs = json.loads((results_dir / "mixture_params_abs_t.json").read_text())
    dep = json.loads((results_dir / "dependence.json").read_text())

    # Prefer sigma=1 fixed (baseline-only, no |t| capping) as baseline; fall back to sigma-free.
    mix_abs_params = mix_abs.get("spec_level", {}).get("baseline_only_sigma_fixed_1", None)
    if mix_abs_params is None:
        mix_abs_params = mix_abs.get("spec_level", {}).get("baseline_only", None)
    if mix_abs_params is None:
        raise RuntimeError("mixture_params_abs_t.json missing sigma-fixed or baseline mixture results.")

    write_mixture_table(mix_abs_params, overleaf_tables / "tab_mixture_params_abs_t.tex")
    write_dependence_table(dep, overleaf_tables / "tab_dependence_summary.tex")

    write_counterfactual_table(
        results_dir / "counterfactual.csv",
        results_dir / "counterfactual_dependence_sensitivity.csv",
        overleaf_tables / "tab_counterfactual_sensitivity.tex",
        params_path=results_dir / "counterfactual_params.json",
    )

    write_inference_audit_table(results_dir / "inference_audit_i4r.csv", overleaf_tables / "tab_inference_audit_i4r.tex")


if __name__ == "__main__":
    main()
