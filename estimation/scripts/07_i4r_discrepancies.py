#!/usr/bin/env python3
"""
07_i4r_discrepancies.py
=======================

Create a richer discrepancy file for Sample A (i4r vs agentic reproductions)
and write a small LaTeX-ready table for Appendix G.

Outputs:
  - estimation/results/i4r_discrepancies.csv
  - overleaf/tex/v8_tables/tab_i4r_discrepancies_top10.tex
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _latex_escape(s: str) -> str:
    # Minimal escaping for tabular text.
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = str(s)
    for k, v in repl.items():
        out = out.replace(k, v)
    return out


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "estimation" / "data"
    results_dir = base_dir / "estimation" / "results"
    overleaf_tables = base_dir / "overleaf" / "tex" / "v8_tables"
    results_dir.mkdir(parents=True, exist_ok=True)
    overleaf_tables.mkdir(parents=True, exist_ok=True)

    cmp_path = data_dir / "i4r_comparison.csv"
    claim_path = data_dir / "claim_level.csv"
    if not (cmp_path.exists() and claim_path.exists()):
        raise FileNotFoundError("Missing inputs. Run 01_build_claim_level.py and 03_extract_i4r_baseline.py first.")

    cmp = pd.read_csv(cmp_path)
    claim = pd.read_csv(claim_path)

    # Recompute the "used" AI t-stat for transparency.
    if "t_AI_oriented" in cmp.columns and cmp["t_AI_oriented"].notna().any():
        cmp["t_AI_used"] = cmp["t_AI_oriented"]
        ai_col_name = "t_AI_oriented"
    else:
        cmp["t_AI_used"] = cmp["t_AI"]
        ai_col_name = "t_AI"

    cmp["diff_AI_i4r_used"] = cmp["t_AI_used"] - cmp["t_i4r"]
    cmp["abs_diff_AI_i4r_used"] = cmp["diff_AI_i4r_used"].abs()

    keep_claim_cols = [
        "paper_id",
        "spec_id",
        "baseline_selection_rule",
        "outcome_var",
        "treatment_var",
        "coefficient",
        "std_error",
        "spec_tree_path",
    ]
    keep_claim_cols = [c for c in keep_claim_cols if c in claim.columns]

    out = cmp.merge(claim[keep_claim_cols], on="paper_id", how="left")

    # Attach exclusion recommendations from paper audit (if present).
    audit_path = results_dir / "i4r_paper_audit.csv"
    if audit_path.exists():
        audit = pd.read_csv(audit_path)
        keep_audit = ["paper_id", "exclude_i4r", "exclude_reason"]
        keep_audit = [c for c in keep_audit if c in audit.columns]
        if keep_audit:
            out = out.merge(audit[keep_audit], on="paper_id", how="left")
    if "exclude_i4r" not in out.columns:
        out["exclude_i4r"] = np.nan
        out["exclude_reason"] = ""

    # Merge oracle (matched reproduction) data if available.
    oracle_path = data_dir / "i4r_oracle_claim_map.csv"
    if oracle_path.exists():
        oracle = pd.read_csv(oracle_path)
        merge_cols = [c for c in ["paper_id", "oracle_abs_t_stat", "oracle_abs_diff_abs_t_to_i4r"] if c in oracle.columns]
        if merge_cols:
            out = out.merge(oracle[merge_cols], on="paper_id", how="left")

    out_path = results_dir / "i4r_discrepancies.csv"
    out.to_csv(out_path, index=False)

    # ------------------------------------------------------------------
    # Helper: agreement classification on |t|
    # ------------------------------------------------------------------
    def _classify_absdiff(x: float) -> str:
        if not np.isfinite(x):
            return "missing"
        if x < 0.1:
            return "exact"
        if x < 0.5:
            return "close"
        return "discrepant"

    # Use |t| for claim-level comparison (i4r stores magnitudes).
    if "t_AI_abs" in out.columns and out["t_AI_abs"].notna().any():
        out["t_AI_plot"] = out["t_AI_abs"].astype(float)
    else:
        out["t_AI_plot"] = out["t_AI_used"].abs()
    out["abs_diff_abs_t_to_i4r"] = (out["t_AI_plot"] - out["t_i4r"]).abs()
    out["agreement_abs"] = out["abs_diff_abs_t_to_i4r"].apply(_classify_absdiff)

    # Top 10 discrepancy table
    top = out.sort_values("abs_diff_AI_i4r_used", ascending=False).head(10).copy()
    # Keep it readable: shorten claim description
    top["claim_short"] = top["claim_description"].astype(str).str.slice(0, 55)
    top["claim_short"] = top["claim_short"].where(top["claim_description"].astype(str).str.len() <= 55, top["claim_short"] + "…")

    lines = []
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\toprule")
    lines.append(r"Paper ID & Claim & $t^{\mathrm{i4r}}$ & $t^{\mathrm{AI}}$ & $|t^{\mathrm{AI}}-t^{\mathrm{i4r}}|$ \\")
    lines.append(r"\midrule")
    for _, r in top.iterrows():
        pid = _latex_escape(r.get("paper_id", ""))
        claim_s = _latex_escape(r.get("claim_short", ""))
        t_i4r = float(r.get("t_i4r", np.nan))
        t_ai = float(r.get("t_AI_used", np.nan))
        ad = float(r.get("abs_diff_AI_i4r_used", np.nan))
        lines.append(f"{pid} & {claim_s} & {t_i4r:.2f} & {t_ai:.2f} & {ad:.2f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    tex_path = overleaf_tables / "tab_i4r_discrepancies_top10.tex"
    tex_path.write_text("\n".join(lines) + "\n")

    # Full claim-by-claim table (all Sample A claims)
    full = out.sort_values(["paper_id"]).copy()
    full["claim_short"] = full["claim_description"].astype(str).str.slice(0, 60)
    full["claim_short"] = full["claim_short"].where(full["claim_description"].astype(str).str.len() <= 60, full["claim_short"] + "…")

    full_lines = []
    full_lines.append(r"\begin{tabular}{llccccc}")
    full_lines.append(r"\toprule")
    full_lines.append(
        r"Paper ID & Claim & $t^{\mathrm{orig}}$ & $t^{\mathrm{i4r}}$ & $|t^{\mathrm{AI}}|$ & $||t^{\mathrm{AI}}|-t^{\mathrm{i4r}}|$ & Status \\"
    )
    full_lines.append(r"\midrule")
    for _, r in full.iterrows():
        pid = _latex_escape(r.get("paper_id", ""))
        claim_s = _latex_escape(r.get("claim_short", ""))
        t_orig = float(r.get("t_orig", np.nan))
        t_i4r = float(r.get("t_i4r", np.nan))
        t_ai = float(r.get("t_AI_plot", np.nan))
        ad = float(r.get("abs_diff_abs_t_to_i4r", np.nan))
        status = _latex_escape(r.get("agreement_abs", "missing"))
        full_lines.append(f"{pid} & {claim_s} & {t_orig:.2f} & {t_i4r:.2f} & {t_ai:.2f} & {ad:.2f} & {status} \\\\")
    full_lines.append(r"\bottomrule")
    full_lines.append(r"\end{tabular}")

    tex_full = overleaf_tables / "tab_i4r_claim_results.tex"
    tex_full.write_text("\n".join(full_lines) + "\n")

    print(f"Wrote {out_path}")
    print(f"Wrote {tex_path} (AI column used: {ai_col_name})")
    print(f"Wrote {tex_full}")

    # ------------------------------------------------------------------
    # Verified-comparable subset (exclude simulated/synthetic/incomplete)
    # Uses oracle (matched) reproductions for agreement comparison.
    # ------------------------------------------------------------------
    if "exclude_i4r" in out.columns:
        v = out[out["exclude_i4r"].fillna(1).astype(int) == 0].copy()
        if len(v) >= 5:
            v_path = results_dir / "i4r_discrepancies_verified.csv"
            v.to_csv(v_path, index=False)

            # Use oracle (matched) reproductions if available
            use_oracle = "oracle_abs_t_stat" in v.columns and v["oracle_abs_t_stat"].notna().any()
            if use_oracle:
                v["t_compare"] = v["oracle_abs_t_stat"].astype(float)
                v["diff_compare"] = v["oracle_abs_diff_abs_t_to_i4r"].astype(float)
            else:
                v["t_compare"] = v["t_AI_plot"].astype(float)
                v["diff_compare"] = v["abs_diff_abs_t_to_i4r"].astype(float)

            v["agreement_compare"] = v["diff_compare"].apply(_classify_absdiff)

            # Top 10 discrepancy table (verified, matched)
            top_v = v.sort_values("diff_compare", ascending=False).head(10).copy()
            top_v["claim_short"] = top_v["claim_description"].astype(str).str.slice(0, 55)
            top_v["claim_short"] = top_v["claim_short"].where(
                top_v["claim_description"].astype(str).str.len() <= 55, top_v["claim_short"] + "…"
            )

            lines_v = []
            lines_v.append(r"\begin{tabular}{llccc}")
            lines_v.append(r"\toprule")
            lines_v.append(
                r"Paper ID & Claim & $|t^{\mathrm{ind}}|$ & $|t^{\mathrm{match}}|$ & $\big||t^{\mathrm{match}}|-|t^{\mathrm{ind}}|\big|$ \\"
            )
            lines_v.append(r"\midrule")
            for _, r in top_v.iterrows():
                pid = _latex_escape(r.get("paper_id", ""))
                claim_s = _latex_escape(r.get("claim_short", ""))
                t_i4r = float(r.get("t_i4r", np.nan))
                t_compare = float(r.get("t_compare", np.nan))
                diff = float(r.get("diff_compare", np.nan))
                lines_v.append(f"{pid} & {claim_s} & {t_i4r:.2f} & {t_compare:.2f} & {diff:.2f} \\\\")
            lines_v.append(r"\bottomrule")
            lines_v.append(r"\end{tabular}")

            tex_v = overleaf_tables / "tab_i4r_discrepancies_top10_verified.tex"
            tex_v.write_text("\n".join(lines_v) + "\n")

            print(f"Wrote {v_path} (verified subset, oracle={use_oracle})")
            print(f"Wrote {tex_v} (verified subset)")

            # Full claim table for verified subset
            v = v.sort_values(["paper_id"]).copy()
            v["claim_short"] = v["claim_description"].astype(str).str.slice(0, 60)
            v["claim_short"] = v["claim_short"].where(v["claim_description"].astype(str).str.len() <= 60, v["claim_short"] + "…")

            v_lines = []
            v_lines.append(r"\begin{tabular}{llccccc}")
            v_lines.append(r"\toprule")
            v_lines.append(
                r"Paper ID & Claim & $t^{\mathrm{orig}}$ & $|t^{\mathrm{ind}}|$ & $|t^{\mathrm{match}}|$ & $\big||t^{\mathrm{match}}|-|t^{\mathrm{ind}}|\big|$ & Status \\"
            )
            v_lines.append(r"\midrule")
            for _, r in v.iterrows():
                pid = _latex_escape(r.get("paper_id", ""))
                claim_s = _latex_escape(r.get("claim_short", ""))
                t_orig = float(r.get("t_orig", np.nan))
                t_i4r = float(r.get("t_i4r", np.nan))
                t_compare = float(r.get("t_compare", np.nan))
                diff = float(r.get("diff_compare", np.nan))
                status = _latex_escape(r.get("agreement_compare", "missing"))
                v_lines.append(f"{pid} & {claim_s} & {t_orig:.2f} & {t_i4r:.2f} & {t_compare:.2f} & {diff:.2f} & {status} \\\\")
            v_lines.append(r"\bottomrule")
            v_lines.append(r"\end{tabular}")

            tex_full_v = overleaf_tables / "tab_i4r_claim_results_verified.tex"
            tex_full_v.write_text("\n".join(v_lines) + "\n")
            print(f"Wrote {tex_full_v} (verified subset)")


if __name__ == "__main__":
    main()
