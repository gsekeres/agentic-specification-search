#!/usr/bin/env python3
"""
01b_build_i4r_oracle_claim_map.py
=================================

Build an "oracle" mapping for Sample A (40 i4r papers) that selects, for each paper,
the verifier-labeled *core test* (including baselines) whose |t|-stat is closest to
the i4r benchmark t-statistic.

Purpose:
  - Diagnose whether claim-mapping / choice of estimand within a paper is driving
    agentic-vs-i4r discrepancies.
  - Provide an "i4r-aligned agentic" series for plotting (should match i4r by design).

IMPORTANT: This mapping uses t^i4r as the selection criterion, so it is a diagnostic /
oracle benchmark, not a primary pipeline mapping.

Outputs:
  - estimation/data/i4r_oracle_claim_map.csv
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]  # agentic_specification_search
DATA_DIR = BASE_DIR / "estimation" / "data"
VERIFICATION_DIR = BASE_DIR / "data" / "verification"
UNIFIED_RESULTS = BASE_DIR / "unified_results.csv"

OUT_PATH = DATA_DIR / "i4r_oracle_claim_map.csv"


def _load_i4r_data() -> dict[str, tuple[float, float, str]]:
    script_path = BASE_DIR / "estimation" / "scripts" / "03_extract_i4r_baseline.py"
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "I4R_DATA":
                    return ast.literal_eval(node.value)  # type: ignore[arg-type]
    raise RuntimeError("Failed to find I4R_DATA in 03_extract_i4r_baseline.py")


def _pick_representative_row(unified_paper: pd.DataFrame, spec_id: str, outcome_var: str, treatment_var: str) -> pd.Series | None:
    """
    Match a baseline spec-map row into unified_results.

    Prefer exact match on (spec_id,outcome_var,treatment_var); if missing, fall back to spec_id only.
    If multiple, pick largest n_obs then lexicographic spec_tree_path.
    """
    sub = unified_paper[unified_paper["spec_id"].astype(str) == str(spec_id)].copy()
    if len(sub) == 0:
        return None

    cand = sub.copy()
    if str(outcome_var).strip():
        cand = cand[cand["outcome_var"].astype(str) == str(outcome_var)]
    if str(treatment_var).strip():
        cand = cand[cand["treatment_var"].astype(str) == str(treatment_var)]
    if len(cand) == 0:
        cand = sub

    cand = cand.copy()
    cand["n_obs_num"] = pd.to_numeric(cand.get("n_obs"), errors="coerce")
    cand = cand.sort_values(["n_obs_num", "spec_tree_path", "spec_id"], ascending=[False, True, True])
    return cand.iloc[0]


def main() -> None:
    i4r = _load_i4r_data()
    paper_ids = list(i4r.keys())

    if not UNIFIED_RESULTS.exists():
        raise FileNotFoundError(f"Missing {UNIFIED_RESULTS}")
    unified = pd.read_csv(UNIFIED_RESULTS)
    required = {"paper_id", "spec_id", "outcome_var", "treatment_var", "coefficient", "std_error", "spec_tree_path"}
    missing = required - set(unified.columns)
    if missing:
        raise ValueError(f"unified_results.csv missing required columns: {sorted(missing)}")

    rows: list[dict] = []
    for pid in paper_ids:
        t_orig, t_i4r, claim_desc = i4r[pid]
        base = {
            "paper_id": pid,
            "claim_description": claim_desc,
            "t_orig": float(t_orig),
            "t_i4r": float(t_i4r),
        }

        sm_path = VERIFICATION_DIR / pid / "verification_spec_map.csv"
        if not sm_path.exists():
            rows.append({**base, "oracle_status": "missing_spec_map"})
            continue

        try:
            sm = pd.read_csv(sm_path)
        except Exception as e:
            rows.append({**base, "oracle_status": f"spec_map_parse_error:{e}"})
            continue

        required_sm = {"spec_id", "outcome_var", "treatment_var", "baseline_group_id", "is_core_test", "category", "is_baseline"}
        if not required_sm.issubset(set(sm.columns)):
            rows.append({**base, "oracle_status": "spec_map_missing_columns"})
            continue

        is_core = pd.to_numeric(sm["is_core_test"], errors="coerce").fillna(0).astype(int)
        cat = sm["category"].astype(str).str.strip().str.lower()
        core_rows = sm[(is_core == 1) & (cat != "invalid")].copy()
        if len(core_rows) == 0:
            rows.append({**base, "oracle_status": "no_core_rows"})
            continue

        unified_paper = unified[unified["paper_id"] == pid].copy()
        if len(unified_paper) == 0:
            rows.append({**base, "oracle_status": "paper_missing_in_unified"})
            continue

        candidates = []
        for _, r in core_rows.drop_duplicates(subset=["spec_id", "outcome_var", "treatment_var"]).iterrows():
            rep = _pick_representative_row(
                unified_paper,
                spec_id=str(r.get("spec_id", "")),
                outcome_var=str(r.get("outcome_var", "")),
                treatment_var=str(r.get("treatment_var", "")),
            )
            if rep is None:
                continue
            bgid = str(r.get("baseline_group_id", "") or "").strip()
            if bgid == "":
                # Core tests should typically map to a baseline group; skip unassigned rows.
                continue
            coef = rep.get("coefficient", np.nan)
            se = rep.get("std_error", np.nan)
            if not np.isfinite(coef) or not np.isfinite(se) or float(se) == 0:
                continue
            t_stat = float(coef) / float(se)
            candidates.append(
                {
                    "baseline_group_id": bgid,
                    "is_baseline": int(pd.to_numeric(r.get("is_baseline", 0), errors="coerce") or 0),
                    "spec_id": str(rep.get("spec_id", "") or ""),
                    "outcome_var": str(rep.get("outcome_var", "") or ""),
                    "treatment_var": str(rep.get("treatment_var", "") or ""),
                    "spec_tree_path": str(rep.get("spec_tree_path", "") or ""),
                    "cluster_var": str(rep.get("cluster_var", "") or ""),
                    "n_obs": float(rep.get("n_obs", np.nan)) if np.isfinite(rep.get("n_obs", np.nan)) else np.nan,
                    "t_stat": t_stat,
                    "abs_t_stat": float(abs(t_stat)),
                    "abs_diff_abs_t_to_i4r": float(abs(abs(t_stat) - float(t_i4r))),
                }
            )

        if not candidates:
            rows.append({**base, "oracle_status": "no_candidates_in_unified"})
            continue

        cand_df = pd.DataFrame(candidates)
        cand_df = cand_df.sort_values(["abs_diff_abs_t_to_i4r", "n_obs", "spec_tree_path", "spec_id"], ascending=[True, False, True, True])
        best = cand_df.iloc[0].to_dict()

        rows.append(
            {
                **base,
                "oracle_status": "ok",
                "oracle_n_candidates": int(len(cand_df)),
                "oracle_baseline_group_id": best.get("baseline_group_id", ""),
                "oracle_is_baseline": int(best.get("is_baseline", 0) or 0),
                "oracle_spec_id": best.get("spec_id", ""),
                "oracle_outcome_var": best.get("outcome_var", ""),
                "oracle_treatment_var": best.get("treatment_var", ""),
                "oracle_cluster_var": best.get("cluster_var", ""),
                "oracle_spec_tree_path": best.get("spec_tree_path", ""),
                "oracle_t_stat": float(best.get("t_stat", np.nan)),
                "oracle_abs_t_stat": float(best.get("abs_t_stat", np.nan)),
                "oracle_abs_diff_abs_t_to_i4r": float(best.get("abs_diff_abs_t_to_i4r", np.nan)),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")
    ok = out[out["oracle_status"] == "ok"].copy()
    if len(ok) > 0:
        print(f"Oracle-mapped: {len(ok)} / {len(out)}")
        print(
            "Median | |t_oracle| - t_i4r |:",
            float(ok["oracle_abs_diff_abs_t_to_i4r"].median()),
        )


if __name__ == "__main__":
    main()
