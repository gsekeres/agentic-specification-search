#!/usr/bin/env python3
"""
01a_build_i4r_claim_map.py
==========================

Build an explicit mapping from i4r paper_id -> (spec_id, outcome_var, treatment_var)
to ensure we extract *the intended estimand* for Sample A comparisons.

Priority order for choosing a row:
  1) Use verification baselines (if present) and pick the baseline group that best
     matches the i4r claim description via simple token overlap; within that group,
     pick the *last* baseline spec_id listed (often the most controlled/preferred).
  2) Otherwise, fall back to baseline-like rows in unified_results/specification_results
     (spec_id==baseline or spec_tree_path contains '#baseline'), choosing the largest-N.
  3) Final fallback: first row for that paper in unified_results.

Outputs:
  - estimation/data/i4r_claim_map.csv
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]  # agentic_specification_search


def _load_i4r_data() -> dict[str, tuple[float, float, str]]:
    script_path = BASE_DIR / "estimation" / "scripts" / "03_extract_i4r_baseline.py"
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "I4R_DATA":
                    return ast.literal_eval(node.value)  # type: ignore[arg-type]
    raise RuntimeError("Failed to find I4R_DATA in 03_extract_i4r_baseline.py")


def _tokens(s: str) -> set[str]:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    toks = {t for t in s.split() if len(t) >= 3}
    # Drop very generic terms
    stop = {"effect", "effects", "evidence", "paper", "study", "using", "results"}
    return {t for t in toks if t not in stop}


def _overlap_score(a: str, b: str) -> float:
    ta = _tokens(a)
    tb = _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _pick_from_verification(
    paper_id: str,
    claim_desc: str,
    unified: pd.DataFrame,
    verification_dir: Path,
) -> dict | None:
    vb = verification_dir / paper_id / "verification_baselines.json"
    if not vb.exists():
        return None

    try:
        raw = vb.read_text(encoding="utf-8", errors="replace").strip()
        if not raw:
            return None
        d = json.loads(raw)
    except Exception as e:
        # Treat malformed verification outputs as absent.
        print(f"  Warning: failed to parse {vb}: {e}")
        return None
    groups = d.get("baseline_groups", []) or []
    if not groups:
        return None

    scored = []
    for g in groups:
        scored.append((_overlap_score(claim_desc, str(g.get("claim_summary", ""))), g))
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best = scored[0]

    baseline_spec_ids = list(best.get("baseline_spec_ids", []) or [])
    if not baseline_spec_ids:
        return None
    chosen_spec_id = str(baseline_spec_ids[-1])

    outcome_vars = list(best.get("baseline_outcome_vars", []) or [])
    treatment_vars = list(best.get("baseline_treatment_vars", []) or [])
    chosen_outcome = str(outcome_vars[0]) if outcome_vars else ""
    chosen_treat = str(treatment_vars[0]) if treatment_vars else ""

    sub = unified[unified["paper_id"] == paper_id].copy()
    cand = sub[(sub["spec_id"] == chosen_spec_id)].copy()
    if chosen_outcome:
        cand = cand[cand["outcome_var"].astype(str) == chosen_outcome]
    if chosen_treat:
        cand = cand[cand["treatment_var"].astype(str) == chosen_treat]

    if len(cand) == 0:
        # Fall back: match only on spec_id within paper.
        cand = sub[sub["spec_id"] == chosen_spec_id].copy()

    if len(cand) == 0:
        return {
            "paper_id": paper_id,
            "claim_description": claim_desc,
            "map_source": "verification_baselines",
            "map_score": best_score,
            "map_baseline_group_id": str(best.get("baseline_group_id", "")),
            "map_expected_sign": str(best.get("expected_sign", "")),
            "map_spec_id": chosen_spec_id,
            "map_outcome_var": chosen_outcome,
            "map_treatment_var": chosen_treat,
            "map_status": "missing_in_unified",
        }

    # Deterministic pick: if multiple, choose largest N then smallest spec_tree_path.
    cand = cand.copy()
    cand["n_obs_num"] = pd.to_numeric(cand.get("n_obs"), errors="coerce")
    cand = cand.sort_values(["n_obs_num", "spec_tree_path", "spec_id"], ascending=[False, True, True])
    r = cand.iloc[0]

    t = float(r["coefficient"]) / float(r["std_error"]) if float(r["std_error"]) != 0 else np.nan
    return {
        "paper_id": paper_id,
        "claim_description": claim_desc,
        "map_source": "verification_baselines",
        "map_score": best_score,
        "map_baseline_group_id": str(best.get("baseline_group_id", "")),
        "map_expected_sign": str(best.get("expected_sign", "")),
        "map_spec_id": str(r.get("spec_id", "")),
        "map_outcome_var": str(r.get("outcome_var", "")),
        "map_treatment_var": str(r.get("treatment_var", "")),
        "t_stat": t,
        "n_obs": r.get("n_obs", np.nan),
        "cluster_var": r.get("cluster_var", ""),
        "spec_tree_path": r.get("spec_tree_path", ""),
        "map_status": "ok",
    }


def _pick_baseline_like(paper_id: str, unified: pd.DataFrame) -> dict | None:
    sub = unified[unified["paper_id"] == paper_id].copy()
    if len(sub) == 0:
        return None

    sub["n_obs_num"] = pd.to_numeric(sub.get("n_obs"), errors="coerce")
    sub["t_stat"] = sub["coefficient"] / sub["std_error"]

    baseline_like = sub[sub["spec_id"].astype(str).eq("baseline")].copy()
    if len(baseline_like) == 0:
        baseline_like = sub[sub["spec_tree_path"].astype(str).str.contains("#baseline", na=False)].copy()
    if len(baseline_like) == 0:
        baseline_like = sub[sub["spec_id"].astype(str).str.contains("baseline", na=False)].copy()

    if len(baseline_like) == 0:
        # Fallback to the first spec (sorted for determinism)
        baseline_like = sub.sort_values(["spec_tree_path", "spec_id"]).head(1).copy()

    baseline_like = baseline_like.sort_values(["n_obs_num", "spec_tree_path", "spec_id"], ascending=[False, True, True])
    r = baseline_like.iloc[0]
    return {
        "paper_id": paper_id,
        "map_spec_id": str(r.get("spec_id", "")),
        "map_outcome_var": str(r.get("outcome_var", "")),
        "map_treatment_var": str(r.get("treatment_var", "")),
        "t_stat": float(r.get("t_stat", np.nan)),
        "n_obs": r.get("n_obs", np.nan),
        "cluster_var": r.get("cluster_var", ""),
        "spec_tree_path": r.get("spec_tree_path", ""),
    }


def main() -> None:
    i4r = _load_i4r_data()
    paper_ids = list(i4r.keys())

    unified_path = BASE_DIR / "unified_results.csv"
    verification_dir = BASE_DIR / "data" / "verification"
    out_path = BASE_DIR / "estimation" / "data" / "i4r_claim_map.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    unified = pd.read_csv(unified_path)
    required = {"paper_id", "spec_id", "outcome_var", "treatment_var", "coefficient", "std_error", "spec_tree_path"}
    missing = required - set(unified.columns)
    if missing:
        raise ValueError(f"unified_results.csv missing required columns: {sorted(missing)}")

    rows: list[dict] = []

    for pid in paper_ids:
        t_orig, t_i4r, claim_desc = i4r[pid]
        base: dict = {
            "paper_id": pid,
            "claim_description": claim_desc,
            "t_orig": float(t_orig),
            "t_i4r": float(t_i4r),
        }

        picked = _pick_from_verification(pid, claim_desc, unified, verification_dir)
        if picked is None:
            picked = _pick_baseline_like(pid, unified)
            if picked is None:
                rows.append({**base, "map_status": "paper_missing"})
                continue
            rows.append(
                {
                    **base,
                    "map_source": "heuristic_baseline_like",
                    "map_score": np.nan,
                    "map_baseline_group_id": "",
                    "map_expected_sign": "",
                    "map_status": "ok",
                    **picked,
                }
            )
            continue

        # picked already includes status
        if picked.get("map_status") != "ok":
            rows.append({**base, **picked})
            continue

        rows.append({**base, **picked})

    out = pd.DataFrame(rows)
    # Compute mismatch diagnostic: compare |t| to i4r benchmark
    if "t_stat" in out.columns:
        out["abs_t_stat"] = out["t_stat"].abs()
        out["abs_diff_abs_t_to_i4r"] = (out["abs_t_stat"] - out["t_i4r"]).abs()

    # Needs review heuristic
    out["needs_review"] = 0
    out.loc[out["map_status"].ne("ok"), "needs_review"] = 1
    out.loc[out["map_source"].eq("verification_baselines") & out["map_score"].fillna(0).lt(0.05), "needs_review"] = 1
    out.loc[out.get("abs_diff_abs_t_to_i4r", pd.Series(dtype=float)).fillna(0).gt(3.0), "needs_review"] = 1

    out = out.sort_values(["needs_review", "abs_diff_abs_t_to_i4r"], ascending=[False, False])
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(f"Needs review: {int(out['needs_review'].sum())} / {len(out)}")
    print(out.head(15)[["paper_id", "map_source", "map_spec_id", "map_outcome_var", "map_treatment_var", "t_i4r", "t_stat", "needs_review"]].to_string(index=False))


if __name__ == "__main__":
    main()
