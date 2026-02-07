#!/usr/bin/env python3
"""
08_i4r_paper_audit.py
=====================

Paper-by-paper audit for Sample A (the 40 i4r papers).

Goal: make it easy to see (i) what the verification agents said the baseline
groups are, (ii) which baseline group/spec the pipeline currently maps to the
i4r canonical claim, and (iii) which papers should be treated as *not*
comparable to i4r (simulated data, missing verification artifacts, etc.).

Writes:
  - estimation/results/i4r_paper_audit.csv   (one row per paper)
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]  # agentic_specification_search
DATA_DIR = BASE_DIR / "estimation" / "data"
RESULTS_DIR = BASE_DIR / "estimation" / "results"
VERIFICATION_DIR = BASE_DIR / "data" / "verification"

UNIFIED_RESULTS = BASE_DIR / "unified_results.csv"
I4R_CLAIM_MAP = DATA_DIR / "i4r_claim_map.csv"
I4R_COMPARISON = DATA_DIR / "i4r_comparison.csv"
VERIFY_SUMMARY = DATA_DIR / "verification_summary_by_paper.csv"

OUT_CSV = RESULTS_DIR / "i4r_paper_audit.csv"


def _load_i4r_data() -> dict[str, tuple[float, float, str]]:
    script_path = BASE_DIR / "estimation" / "scripts" / "03_extract_i4r_baseline.py"
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "I4R_DATA":
                    return ast.literal_eval(node.value)  # type: ignore[arg-type]
    raise RuntimeError("Failed to find I4R_DATA in 03_extract_i4r_baseline.py")


def _safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _flag_contains(flags: str, pat: str) -> bool:
    return re.search(pat, flags or "", flags=re.IGNORECASE) is not None


def _exclude_reason(flags: str) -> str:
    """
    Conservative exclusion for i4r comparison: exclude if replication is clearly
    not a real re-analysis (simulated/synthetic) or if verification artifacts are
    missing/invalid.
    """
    if _flag_contains(flags, r"missing_spec_map|missing_baselines|baselines_parse_error|empty_baselines"):
        return "incomplete_verification"
    if _flag_contains(flags, r"simulated_data"):
        return "simulated_data"
    if _flag_contains(flags, r"synthetic_specs"):
        return "synthetic_specs"
    # Keep wrong_sign/degenerate_cluster as *warnings* not hard excludes.
    return ""


@dataclass(frozen=True)
class GroupCandidate:
    baseline_group_id: str
    claim_summary: str
    expected_sign: str
    notes: str
    baseline_spec_ids: list[str]
    baseline_outcome_vars: list[str]
    baseline_treatment_vars: list[str]
    # Representative baseline row found in unified results (if any)
    rep_spec_id: str
    rep_outcome_var: str
    rep_treatment_var: str
    rep_t_stat: float
    rep_n_obs: float


def _coerce_num(s) -> float:
    try:
        v = float(s)
        return v
    except Exception:
        return float("nan")


def _pick_representative_row(
    unified_paper: pd.DataFrame,
    spec_id: str,
    outcome_vars: list[str],
    treatment_vars: list[str],
) -> tuple[str, str, str, float, float]:
    """
    Pick one representative row for a baseline spec_id.

    We try to match on (spec_id, outcome_var, treatment_var) if provided; if not found,
    fall back to matching on spec_id only. If multiple, choose largest n_obs, then
    lexicographic path.
    """
    sub = unified_paper[unified_paper["spec_id"].astype(str) == str(spec_id)].copy()
    if len(sub) == 0:
        return "", "", "", float("nan"), float("nan")

    # Try matching on any (outcome,treatment) combo listed by verifier (if provided).
    if outcome_vars or treatment_vars:
        cand = []
        oset = list(outcome_vars) if outcome_vars else [None]
        tset = list(treatment_vars) if treatment_vars else [None]
        for ov in oset:
            for tv in tset:
                c = sub
                if ov is not None and str(ov).strip() != "":
                    c = c[c["outcome_var"].astype(str) == str(ov)]
                if tv is not None and str(tv).strip() != "":
                    c = c[c["treatment_var"].astype(str) == str(tv)]
                if len(c) > 0:
                    cand.append(c)
        if cand:
            sub = pd.concat(cand, ignore_index=True).drop_duplicates()

    sub = sub.copy()
    sub["n_obs_num"] = pd.to_numeric(sub.get("n_obs"), errors="coerce")
    sub["t_stat_num"] = pd.to_numeric(sub.get("coefficient"), errors="coerce") / pd.to_numeric(sub.get("std_error"), errors="coerce")
    sub = sub[sub["t_stat_num"].notna() & np.isfinite(sub["t_stat_num"])].copy()
    if len(sub) == 0:
        return "", "", "", float("nan"), float("nan")

    sub = sub.sort_values(["n_obs_num", "spec_tree_path", "spec_id"], ascending=[False, True, True])
    r = sub.iloc[0]
    return (
        str(r.get("spec_id", "")),
        str(r.get("outcome_var", "")),
        str(r.get("treatment_var", "")),
        float(r.get("t_stat_num", np.nan)),
        float(r.get("n_obs_num", np.nan)),
    )


def _load_baselines(paper_id: str) -> tuple[list[GroupCandidate], str]:
    """
    Load verification_baselines.json and return structured baseline groups.
    Returns (groups, error_string).
    """
    p = VERIFICATION_DIR / paper_id / "verification_baselines.json"
    if not p.exists():
        return [], "missing_baselines_json"

    try:
        raw = _safe_read_text(p).strip()
        d = json.loads(raw) if raw else {}
    except Exception as e:
        return [], f"baselines_parse_error: {e}"

    groups_raw = d.get("baseline_groups", []) or []
    if not groups_raw:
        return [], "empty_baseline_groups"

    return_groups: list[GroupCandidate] = []
    unified = pd.read_csv(UNIFIED_RESULTS)
    up = unified[unified["paper_id"].astype(str) == str(paper_id)].copy()
    # Compute t-stat once
    up["t_stat"] = pd.to_numeric(up["coefficient"], errors="coerce") / pd.to_numeric(up["std_error"], errors="coerce")

    for g in groups_raw:
        gid = str(g.get("baseline_group_id", "")).strip()
        claim = str(g.get("claim_summary", "")).strip()
        exp = str(g.get("expected_sign", "")).strip()
        notes = str(g.get("notes", "")).strip()
        spec_ids = [str(s) for s in (g.get("baseline_spec_ids", []) or [])]
        outs = [str(s) for s in (g.get("baseline_outcome_vars", []) or [])]
        treats = [str(s) for s in (g.get("baseline_treatment_vars", []) or [])]

        # Representative: try the *last* baseline_spec_id (often the most controlled).
        rep_spec_id = spec_ids[-1] if spec_ids else ""
        rep = _pick_representative_row(up, rep_spec_id, outs, treats) if rep_spec_id else ("", "", "", np.nan, np.nan)

        return_groups.append(
            GroupCandidate(
                baseline_group_id=gid,
                claim_summary=claim,
                expected_sign=exp,
                notes=notes,
                baseline_spec_ids=spec_ids,
                baseline_outcome_vars=outs,
                baseline_treatment_vars=treats,
                rep_spec_id=rep[0],
                rep_outcome_var=rep[1],
                rep_treatment_var=rep[2],
                rep_t_stat=float(rep[3]),
                rep_n_obs=float(rep[4]),
            )
        )

    return return_groups, ""


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    i4r = _load_i4r_data()
    ids = list(i4r.keys())

    unified = pd.read_csv(UNIFIED_RESULTS)
    unified["t_stat"] = pd.to_numeric(unified["coefficient"], errors="coerce") / pd.to_numeric(unified["std_error"], errors="coerce")

    claim_map = pd.read_csv(I4R_CLAIM_MAP) if I4R_CLAIM_MAP.exists() else pd.DataFrame()
    cmp = pd.read_csv(I4R_COMPARISON) if I4R_COMPARISON.exists() else pd.DataFrame()
    verify = pd.read_csv(VERIFY_SUMMARY) if VERIFY_SUMMARY.exists() else pd.DataFrame()

    claim_map = claim_map[claim_map["paper_id"].isin(ids)].copy() if len(claim_map) else claim_map
    cmp = cmp[cmp["paper_id"].isin(ids)].copy() if len(cmp) else cmp
    verify = verify[verify["paper_id"].isin(ids)].copy() if len(verify) else verify

    # Index for quick lookups
    cm_idx = {str(r["paper_id"]): r.to_dict() for _, r in claim_map.iterrows()} if len(claim_map) else {}
    cmp_idx = {str(r["paper_id"]): r.to_dict() for _, r in cmp.iterrows()} if len(cmp) else {}
    v_idx = {str(r["paper_id"]): r.to_dict() for _, r in verify.iterrows()} if len(verify) else {}

    rows: list[dict] = []
    for pid in ids:
        t_orig, t_i4r, claim_desc = i4r[pid]
        flags = str(v_idx.get(pid, {}).get("flags", "") or "")
        exclude_reason = _exclude_reason(flags)
        exclude_i4r = 1 if exclude_reason else 0

        cm = cm_idx.get(pid, {})
        chosen_gid = str(cm.get("map_baseline_group_id", "") or "")
        chosen_spec = str(cm.get("map_spec_id", "") or "")
        chosen_out = str(cm.get("map_outcome_var", "") or "")
        chosen_tr = str(cm.get("map_treatment_var", "") or "")
        chosen_t = _coerce_num(cm.get("t_stat", np.nan))
        map_source = str(cm.get("map_source", "") or "")
        map_score = _coerce_num(cm.get("map_score", np.nan))
        needs_review = int(cm.get("needs_review", 0) or 0) if cm else 1

        map_diff = float("nan")
        if np.isfinite(chosen_t):
            map_diff = float(abs(abs(chosen_t) - float(t_i4r)))

        groups, baselines_err = _load_baselines(pid)
        n_groups = len(groups)

        # Paper metadata from i4r_comparison (title/journal) if available
        cmp_row = cmp_idx.get(pid, {})
        title = str(cmp_row.get("title", "") or "")
        journal = str(cmp_row.get("journal", "") or "")
        t_ai_oriented = _coerce_num(cmp_row.get("t_AI_oriented", np.nan))
        t_ai_used = t_ai_oriented if np.isfinite(t_ai_oriented) else _coerce_num(cmp_row.get("t_AI", np.nan))

        rows.append(
            {
                "paper_id": pid,
                "journal": journal,
                "title": title,
                "claim_description": claim_desc,
                "t_orig": float(t_orig),
                "t_i4r": float(t_i4r),
                "t_AI_oriented": t_ai_oriented,
                "t_AI_used": t_ai_used,
                "map_source": map_source,
                "map_score": map_score,
                "map_baseline_group_id": chosen_gid,
                "map_spec_id": chosen_spec,
                "map_outcome_var": chosen_out,
                "map_treatment_var": chosen_tr,
                "t_mapped": chosen_t,
                "map_diff_abs_t_to_i4r": map_diff,
                "needs_review": needs_review,
                "verification_flags": flags,
                "exclude_i4r": exclude_i4r,
                "exclude_reason": exclude_reason,
                "n_baseline_groups": n_groups,
                "baselines_error": baselines_err,
            }
        )

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(
        ["exclude_i4r", "needs_review", "map_diff_abs_t_to_i4r"],
        ascending=[False, False, False],
    )
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    print(f"Wrote {OUT_CSV}")
    print(f"Exclude_i4r: {int(out_df['exclude_i4r'].sum())} / {len(out_df)}")
    print(f"Needs review (map heuristic): {int(out_df['needs_review'].sum())} / {len(out_df)}")


if __name__ == "__main__":
    main()
