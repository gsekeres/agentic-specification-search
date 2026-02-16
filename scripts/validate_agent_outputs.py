#!/usr/bin/env python3
"""
validate_agent_outputs.py
=========================

Validate runner + verifier artifacts against the typed spec-tree contract.

Checks per-paper outputs in:
  data/downloads/extracted/{PAPER_ID}/

And (optionally) verifier outputs in:
  data/verification/{PAPER_ID}/

Exit code:
  0 if no ERROR issues
  1 if any ERROR issues

Usage:
  python scripts/validate_agent_outputs.py --paper-id 112431-V1
  python scripts/validate_agent_outputs.py --all
  python scripts/validate_agent_outputs.py --all --report-csv estimation/data/agent_validation.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from fnmatch import fnmatch
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTED_DIR = ROOT / "data" / "downloads" / "extracted"
DEFAULT_VERIFICATION_DIR = ROOT / "data" / "verification"


REQUIRED_SPEC_COLS: list[str] = [
    "paper_id",
    "spec_run_id",
    "baseline_group_id",
    "spec_id",
    "spec_tree_path",
    "outcome_var",
    "treatment_var",
    "coefficient",
    "std_error",
    "p_value",
    "ci_lower",
    "ci_upper",
    "n_obs",
    "r_squared",
    "coefficient_vector_json",
    "sample_desc",
    "fixed_effects",
    "controls_desc",
    "cluster_var",
    "run_success",
    "run_error",
]

REQUIRED_INFER_COLS: list[str] = [
    "paper_id",
    "inference_run_id",
    "spec_run_id",
    "baseline_group_id",
    "spec_id",
    "spec_tree_path",
    "outcome_var",
    "treatment_var",
    "coefficient",
    "std_error",
    "p_value",
    "ci_lower",
    "ci_upper",
    "n_obs",
    "r_squared",
    "cluster_var",
    "coefficient_vector_json",
    "run_success",
    "run_error",
]

REQUIRED_VERIFICATION_MAP_COLS: list[str] = [
    "paper_id",
    "baseline_group_id",
    "spec_run_id",
    "closest_baseline_spec_run_id",
    "is_baseline",
    "is_valid",
    "is_core_test",
    "category",
    "why",
    "confidence",
]


@dataclass(frozen=True)
class Issue:
    severity: str  # "ERROR" | "WARN"
    paper_id: str
    area: str
    path: str
    code: str
    message: str
    baseline_group_id: str = ""
    spec_run_id: str = ""
    inference_run_id: str = ""
    row_index: int | None = None


def _issue(
    issues: list[Issue],
    severity: str,
    paper_id: str,
    area: str,
    path: Path,
    code: str,
    message: str,
    *,
    baseline_group_id: str = "",
    spec_run_id: str = "",
    inference_run_id: str = "",
    row_index: int | None = None,
) -> None:
    issues.append(
        Issue(
            severity=severity,
            paper_id=paper_id,
            area=area,
            path=str(path),
            code=code,
            message=message,
            baseline_group_id=str(baseline_group_id or ""),
            spec_run_id=str(spec_run_id or ""),
            inference_run_id=str(inference_run_id or ""),
            row_index=row_index,
        )
    )


def _read_csv(path: Path, issues: list[Issue], paper_id: str, area: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        _issue(issues, "ERROR", paper_id, area, path, "missing_file", "File not found.")
        return pd.DataFrame()
    except Exception as e:
        _issue(issues, "ERROR", paper_id, area, path, "read_failed", f"Failed to read CSV: {e}")
        return pd.DataFrame()


def _check_required_cols(df: pd.DataFrame, required: list[str], issues: list[Issue], paper_id: str, area: str, path: Path) -> bool:
    missing = [c for c in required if c not in df.columns]
    if missing:
        _issue(
            issues,
            "ERROR",
            paper_id,
            area,
            path,
            "missing_columns",
            f"Missing required columns: {missing}",
        )
        return False
    return True


def _safe_str(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x)


def _parse_json_field(
    issues: list[Issue],
    paper_id: str,
    area: str,
    path: Path,
    payload: str,
    *,
    baseline_group_id: str = "",
    spec_run_id: str = "",
    inference_run_id: str = "",
    row_index: int | None = None,
) -> None:
    s = _safe_str(payload)
    if not s.strip():
        _issue(
            issues,
            "ERROR",
            paper_id,
            area,
            path,
            "empty_json",
            "coefficient_vector_json is empty; expected valid JSON (use '{}' if needed).",
            baseline_group_id=baseline_group_id,
            spec_run_id=spec_run_id,
            inference_run_id=inference_run_id,
            row_index=row_index,
        )
        return
    try:
        json.loads(s)
    except Exception as e:
        _issue(
            issues,
            "ERROR",
            paper_id,
            area,
            path,
            "invalid_json",
            f"coefficient_vector_json is not valid JSON: {e}",
            baseline_group_id=baseline_group_id,
            spec_run_id=spec_run_id,
            inference_run_id=inference_run_id,
            row_index=row_index,
        )


def _validate_surface(
    paper_id: str,
    pkg_dir: Path,
    issues: list[Issue],
) -> tuple[dict[str, list[str] | None], dict[str, list[str]]]:
    """
    Returns:
      allowed_core_spec_patterns_by_group
      allowed_infer_spec_patterns_by_group
    """
    path = pkg_dir / "SPECIFICATION_SURFACE.json"
    if not path.exists():
        _issue(issues, "ERROR", paper_id, "surface", path, "missing_surface", "Missing SPECIFICATION_SURFACE.json.")
        return {}, {}

    try:
        d = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        _issue(issues, "ERROR", paper_id, "surface", path, "invalid_surface_json", f"Failed to parse JSON: {e}")
        return {}, {}

    if str(d.get("paper_id", "")).strip() != paper_id:
        _issue(
            issues,
            "ERROR",
            paper_id,
            "surface",
            path,
            "paper_id_mismatch",
            f"surface.paper_id={d.get('paper_id')} does not match directory paper_id={paper_id}",
        )

    groups = d.get("baseline_groups", [])
    if not isinstance(groups, list) or not groups:
        _issue(issues, "ERROR", paper_id, "surface", path, "missing_baseline_groups", "surface.baseline_groups is empty or missing.")
        return {}, {}

    core_patterns: dict[str, list[str] | None] = {}
    infer_patterns: dict[str, list[str]] = {}

    for g in groups:
        gid = str(g.get("baseline_group_id", "")).strip()
        if not gid:
            _issue(issues, "ERROR", paper_id, "surface", path, "missing_baseline_group_id", "A baseline_group is missing baseline_group_id.")
            continue

        if not str(g.get("design_code", "")).strip():
            _issue(
                issues,
                "ERROR",
                paper_id,
                "surface",
                path,
                "missing_design_code",
                "baseline_group.design_code is missing/empty.",
                baseline_group_id=gid,
            )

        claim = g.get("claim_object", {})
        if not isinstance(claim, dict):
            claim = {}
        for k in ["outcome_concept", "treatment_concept", "estimand_concept", "target_population"]:
            if not str(claim.get(k, "")).strip():
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "surface",
                    path,
                    "missing_claim_object_field",
                    f"claim_object.{k} is missing/empty.",
                    baseline_group_id=gid,
                )

        baseline_specs = g.get("baseline_specs", [])
        if not isinstance(baseline_specs, list) or not baseline_specs:
            _issue(
                issues,
                "ERROR",
                paper_id,
                "surface",
                path,
                "missing_baseline_specs",
                "baseline_group.baseline_specs is empty or missing.",
                baseline_group_id=gid,
            )

        if "core_universe" not in g:
            _issue(
                issues,
                "ERROR",
                paper_id,
                "surface",
                path,
                "missing_core_universe",
                "baseline_group.core_universe is missing (required).",
                baseline_group_id=gid,
            )
            core_patterns[gid] = None
            core = {}
        else:
            core = g.get("core_universe", {})
            if not isinstance(core, dict):
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "surface",
                    path,
                    "invalid_core_universe",
                    "baseline_group.core_universe must be an object with design_spec_ids and rc_spec_ids lists.",
                    baseline_group_id=gid,
                )
                core_patterns[gid] = None
                core = {}

        design_spec_ids = core.get("design_spec_ids", [])
        rc_spec_ids = core.get("rc_spec_ids", [])
        if not isinstance(design_spec_ids, list) or not isinstance(rc_spec_ids, list):
            _issue(
                issues,
                "ERROR",
                paper_id,
                "surface",
                path,
                "invalid_core_universe",
                "core_universe.design_spec_ids and core_universe.rc_spec_ids must be lists.",
                baseline_group_id=gid,
            )
            design_spec_ids = [] if not isinstance(design_spec_ids, list) else design_spec_ids
            rc_spec_ids = [] if not isinstance(rc_spec_ids, list) else rc_spec_ids

        # Only define matching patterns when core_universe exists and is parseable.
        if core_patterns.get(gid, "__unset__") is not None:
            baseline_spec_ids = core.get("baseline_spec_ids", [])
            if not isinstance(baseline_spec_ids, list):
                baseline_spec_ids = []
            pats = ["baseline", "baseline__*", "baseline/*"] + [str(s) for s in baseline_spec_ids] + [str(s) for s in design_spec_ids] + [str(s) for s in rc_spec_ids]
            if len(pats) <= 2:
                _issue(
                    issues,
                    "WARN",
                    paper_id,
                    "surface",
                    path,
                    "empty_core_universe",
                    "core_universe has no design_spec_ids or rc_spec_ids (baseline-only surface).",
                    baseline_group_id=gid,
                )
            core_patterns[gid] = pats

        ip = g.get("inference_plan", {})
        if not isinstance(ip, dict):
            ip = {}
        canonical = ip.get("canonical", {})
        if not isinstance(canonical, dict):
            canonical = {}
        canon_id = str(canonical.get("spec_id", "")).strip()
        if not canon_id:
            _issue(
                issues,
                "ERROR",
                paper_id,
                "surface",
                path,
                "missing_canonical_inference",
                "inference_plan.canonical.spec_id is missing/empty (required).",
                baseline_group_id=gid,
            )

        variants = ip.get("variants", [])
        if variants is None:
            variants = []
        if not isinstance(variants, list):
            _issue(
                issues,
                "ERROR",
                paper_id,
                "surface",
                path,
                "invalid_inference_variants",
                "inference_plan.variants must be a list.",
                baseline_group_id=gid,
            )
            variants = []
        var_ids = [str(v.get("spec_id", "")).strip() for v in variants if isinstance(v, dict)]
        infer_patterns[gid] = [s for s in [canon_id, *var_ids] if s]

        budgets = g.get("budgets", {})
        if not isinstance(budgets, dict) or not budgets:
            _issue(
                issues,
                "WARN",
                paper_id,
                "surface",
                path,
                "missing_budgets",
                "baseline_group.budgets is missing/empty (recommended).",
                baseline_group_id=gid,
            )

    return core_patterns, infer_patterns


def _validate_spec_results(
    paper_id: str,
    pkg_dir: Path,
    issues: list[Issue],
    core_patterns_by_group: dict[str, list[str] | None],
) -> pd.DataFrame:
    path = pkg_dir / "specification_results.csv"
    df = _read_csv(path, issues, paper_id, "spec_results")
    if df.empty:
        return df

    _check_required_cols(df, REQUIRED_SPEC_COLS, issues, paper_id, "spec_results", path)

    # Paper id consistency
    if "paper_id" in df.columns:
        bad = df["paper_id"].astype(str).ne(paper_id)
        if bad.any():
            _issue(
                issues,
                "ERROR",
                paper_id,
                "spec_results",
                path,
                "paper_id_column_mismatch",
                f"paper_id column contains {bad.sum()} rows not equal to {paper_id}.",
            )

    if "spec_run_id" not in df.columns:
        _issue(issues, "ERROR", paper_id, "spec_results", path, "missing_spec_run_id", "Missing spec_run_id column.")
    else:
        df["spec_run_id"] = df["spec_run_id"].astype(str)
        if df["spec_run_id"].str.strip().eq("").any():
            _issue(
                issues,
                "ERROR",
                paper_id,
                "spec_results",
                path,
                "empty_spec_run_id",
                "Some spec_run_id values are empty.",
            )
        dup = df.duplicated(subset=["spec_run_id"], keep=False)
        if dup.any():
            ex = df.loc[dup, "spec_run_id"].astype(str).head(5).tolist()
            _issue(
                issues,
                "ERROR",
                paper_id,
                "spec_results",
                path,
                "duplicate_spec_run_id",
                f"Duplicate spec_run_id values found (examples: {ex}).",
            )

    # Baseline group non-empty
    if "baseline_group_id" not in df.columns:
        _issue(issues, "ERROR", paper_id, "spec_results", path, "missing_baseline_group_id", "Missing baseline_group_id column.")
        bg = pd.Series(dtype=str)
    else:
        bg = df["baseline_group_id"].astype(str).str.strip()
        if bg.eq("").any():
            _issue(
                issues,
                "ERROR",
                paper_id,
                "spec_results",
                path,
                "empty_baseline_group_id",
                "Some baseline_group_id values are empty.",
            )

    # Namespace rules: no infer/diag/sens/post/explore in spec table.
    if "spec_id" not in df.columns:
        _issue(issues, "ERROR", paper_id, "spec_results", path, "missing_spec_id", "Missing spec_id column.")
        spec_id = pd.Series(dtype=str)
    else:
        spec_id = df["spec_id"].astype(str)
        forbidden = spec_id.str.startswith(("infer/", "diag/", "sens/", "post/", "explore/"))
        if forbidden.any():
            bad_ids = spec_id[forbidden].value_counts().head(5).to_dict()
            _issue(
                issues,
                "ERROR",
                paper_id,
                "spec_results",
                path,
                "forbidden_spec_id_namespace",
                f"specification_results.csv contains non-estimate namespaces (examples: {bad_ids}).",
            )

        allowed_prefix = spec_id.eq("baseline") | spec_id.str.startswith(("baseline__", "baseline/", "design/", "rc/"))
        if (~allowed_prefix).any():
            bad_ids = spec_id[~allowed_prefix].value_counts().head(8).to_dict()
            _issue(
                issues,
                "ERROR",
                paper_id,
                "spec_results",
                path,
                "unknown_spec_id_namespace",
                f"specification_results.csv contains spec_id outside baseline/design/rc (examples: {bad_ids}).",
            )

    # Surface patterns: each row should match the surface core universe for its baseline group.
    if core_patterns_by_group:
        for i, row in df.iterrows():
            gid = str(row.get("baseline_group_id", "")).strip()
            sid = str(row.get("spec_id", "")).strip()
            rid = str(row.get("spec_run_id", "")).strip()
            if not gid or not sid:
                continue
            if gid not in core_patterns_by_group:
                _issue(
                    issues,
                    "WARN",
                    paper_id,
                    "spec_results",
                    path,
                    "baseline_group_missing_in_surface",
                    f"baseline_group_id={gid} appears in results but not in surface baseline_groups.",
                    baseline_group_id=gid,
                    spec_run_id=rid,
                    row_index=int(i),
                )
                continue
            pats = core_patterns_by_group.get(gid)
            if pats is None:
                # Surface exists but did not provide an executable core universe for this group.
                continue
            if not any(fnmatch(sid, p) for p in pats):
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "spec_results",
                    path,
                    "spec_id_not_in_surface_core_universe",
                    f"spec_id={sid} does not match any core_universe pattern for baseline_group_id={gid}.",
                    baseline_group_id=gid,
                    spec_run_id=rid,
                    row_index=int(i),
                )

    # run_success/run_error rules + numeric sanity for successful rows
    if "run_success" not in df.columns:
        _issue(issues, "ERROR", paper_id, "spec_results", path, "missing_run_success", "Missing run_success column.")
        rs = pd.Series([-1] * len(df), index=df.index, dtype=int)
    else:
        rs_raw = pd.to_numeric(df["run_success"], errors="coerce")
        bad_rs = ~(rs_raw.isin([0, 1]))
        if bad_rs.any():
            _issue(
                issues,
                "ERROR",
                paper_id,
                "spec_results",
                path,
                "invalid_run_success_values",
                f"run_success contains values outside {{0,1}} (n={int(bad_rs.sum())}).",
            )
        rs = rs_raw.fillna(-1).astype(int)

    if "run_error" not in df.columns:
        _issue(issues, "ERROR", paper_id, "spec_results", path, "missing_run_error", "Missing run_error column.")
        run_error = pd.Series([""] * len(df), index=df.index, dtype=str)
    else:
        run_error = df["run_error"].fillna("").astype(str)

    coef = pd.to_numeric(df.get("coefficient"), errors="coerce")
    se = pd.to_numeric(df.get("std_error"), errors="coerce")
    pval = pd.to_numeric(df.get("p_value"), errors="coerce")
    ci_lo = pd.to_numeric(df.get("ci_lower"), errors="coerce")
    ci_hi = pd.to_numeric(df.get("ci_upper"), errors="coerce")

    for i, row in df.iterrows():
        gid = str(row.get("baseline_group_id", "")).strip()
        rid = str(row.get("spec_run_id", "")).strip()
        sid = str(row.get("spec_id", "")).strip()

        if "coefficient_vector_json" in df.columns:
            _parse_json_field(
                issues,
                paper_id,
                "spec_results",
                path,
                row.get("coefficient_vector_json", ""),
                baseline_group_id=gid,
                spec_run_id=rid,
                row_index=int(i),
            )

        if rs.iloc[i] == 0:
            if run_error.iloc[i].strip() == "":
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "spec_results",
                    path,
                    "missing_run_error",
                    "run_success=0 but run_error is empty.",
                    baseline_group_id=gid,
                    spec_run_id=rid,
                    row_index=int(i),
                )
            # If failures report finite coef/se/p, flag as warning (often indicates half-failed extraction).
            if np.isfinite(coef.iloc[i]) or np.isfinite(se.iloc[i]) or np.isfinite(pval.iloc[i]):
                _issue(
                    issues,
                    "WARN",
                    paper_id,
                    "spec_results",
                    path,
                    "failure_has_numeric_fields",
                    "run_success=0 but coefficient/std_error/p_value are non-missing; verify extraction and labeling.",
                    baseline_group_id=gid,
                    spec_run_id=rid,
                    row_index=int(i),
                )
            continue

        if rs.iloc[i] == 1:
            if not np.isfinite(coef.iloc[i]):
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "spec_results",
                    path,
                    "missing_coef_success",
                    f"run_success=1 but coefficient is missing/invalid (spec_id={sid}).",
                    baseline_group_id=gid,
                    spec_run_id=rid,
                    row_index=int(i),
                )
            if not np.isfinite(se.iloc[i]) or not (se.iloc[i] > 0):
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "spec_results",
                    path,
                    "missing_se_success",
                    f"run_success=1 but std_error is missing/invalid (spec_id={sid}).",
                    baseline_group_id=gid,
                    spec_run_id=rid,
                    row_index=int(i),
                )
            if not np.isfinite(pval.iloc[i]) or pval.iloc[i] < 0 or pval.iloc[i] > 1:
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "spec_results",
                    path,
                    "invalid_p_value_success",
                    f"run_success=1 but p_value is missing/invalid (p={pval.iloc[i]}).",
                    baseline_group_id=gid,
                    spec_run_id=rid,
                    row_index=int(i),
                )
            if np.isfinite(ci_lo.iloc[i]) and np.isfinite(ci_hi.iloc[i]) and (ci_lo.iloc[i] > ci_hi.iloc[i]):
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "spec_results",
                    path,
                    "invalid_ci_bounds",
                    f"ci_lower > ci_upper (ci_lower={ci_lo.iloc[i]}, ci_upper={ci_hi.iloc[i]}).",
                    baseline_group_id=gid,
                    spec_run_id=rid,
                    row_index=int(i),
                )
            if np.isfinite(ci_lo.iloc[i]) and np.isfinite(ci_hi.iloc[i]) and np.isfinite(coef.iloc[i]):
                if (coef.iloc[i] < ci_lo.iloc[i]) or (coef.iloc[i] > ci_hi.iloc[i]):
                    _issue(
                        issues,
                        "WARN",
                        paper_id,
                        "spec_results",
                        path,
                        "coef_outside_ci",
                        "coefficient lies outside [ci_lower, ci_upper]; check extraction or rounding.",
                        baseline_group_id=gid,
                        spec_run_id=rid,
                        row_index=int(i),
                    )

    # Baseline existence per baseline group (surface-driven expectation)
    if core_patterns_by_group and ("baseline_group_id" in df.columns) and ("spec_id" in df.columns):
        for gid in sorted(set(bg) - {""}):
            sub = df[df["baseline_group_id"].astype(str).str.strip() == gid]
            if len(sub) == 0:
                continue
            is_baseline = sub["spec_id"].astype(str).eq("baseline") | sub["spec_id"].astype(str).str.startswith(("baseline__", "baseline/"))
            if not bool(is_baseline.any()):
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "spec_results",
                    path,
                    "missing_baseline_row_for_group",
                    "No baseline row (spec_id==baseline or baseline__*) found for this baseline_group_id.",
                    baseline_group_id=gid,
                )

    # Search log presence (prompt-required)
    md = pkg_dir / "SPECIFICATION_SEARCH.md"
    if not md.exists():
        _issue(issues, "WARN", paper_id, "spec_results", md, "missing_search_log", "Missing SPECIFICATION_SEARCH.md (recommended).")

    return df


def _validate_inference_results(
    paper_id: str,
    pkg_dir: Path,
    issues: list[Issue],
    spec_results: pd.DataFrame,
    infer_patterns_by_group: dict[str, list[str]],
) -> pd.DataFrame:
    path = pkg_dir / "inference_results.csv"
    if not path.exists():
        # If the surface requested variants, missing inference table is an error.
        needs = False
        for pats in infer_patterns_by_group.values():
            # Treat any pattern beyond the canonical (i.e., >1) as "variants requested".
            if len(pats) > 1:
                needs = True
                break
        if needs:
            _issue(
                issues,
                "ERROR",
                paper_id,
                "inference_results",
                path,
                "missing_inference_results",
                "Surface includes inference variants but inference_results.csv is missing.",
            )
        return pd.DataFrame()

    df = _read_csv(path, issues, paper_id, "inference_results")
    if df.empty:
        return df

    if not _check_required_cols(df, REQUIRED_INFER_COLS, issues, paper_id, "inference_results", path):
        return df

    # Basic id integrity
    df["inference_run_id"] = df["inference_run_id"].astype(str)
    if df["inference_run_id"].str.strip().eq("").any():
        _issue(issues, "ERROR", paper_id, "inference_results", path, "empty_inference_run_id", "Some inference_run_id values are empty.")
    dup = df.duplicated(subset=["inference_run_id"], keep=False)
    if dup.any():
        ex = df.loc[dup, "inference_run_id"].astype(str).head(5).tolist()
        _issue(
            issues,
            "ERROR",
            paper_id,
            "inference_results",
            path,
            "duplicate_inference_run_id",
            f"Duplicate inference_run_id values found (examples: {ex}).",
        )

    # spec_id must be infer/*
    spec_id = df["spec_id"].astype(str)
    bad = ~spec_id.str.startswith("infer/")
    if bad.any():
        bad_ids = spec_id[bad].value_counts().head(8).to_dict()
        _issue(
            issues,
            "ERROR",
            paper_id,
            "inference_results",
            path,
            "non_infer_spec_ids_in_inference_table",
            f"inference_results.csv contains spec_id not starting with infer/ (examples: {bad_ids}).",
        )

    # Link to base estimate spec_run_id
    base_ids = set(spec_results["spec_run_id"].astype(str).tolist()) if (spec_results is not None and not spec_results.empty) else set()
    df["spec_run_id"] = df["spec_run_id"].astype(str)
    missing_base = ~df["spec_run_id"].isin(base_ids)
    if missing_base.any():
        ex = df.loc[missing_base, "spec_run_id"].astype(str).head(8).tolist()
        _issue(
            issues,
            "ERROR",
            paper_id,
            "inference_results",
            path,
            "inference_spec_run_id_missing_in_spec_results",
            f"inference_results references spec_run_id not found in specification_results.csv (examples: {ex}).",
        )

    # Surface inference-plan patterns (if present)
    if infer_patterns_by_group:
        for i, row in df.iterrows():
            gid = str(row.get("baseline_group_id", "")).strip()
            sid = str(row.get("spec_id", "")).strip()
            irid = str(row.get("inference_run_id", "")).strip()
            if not gid or not sid:
                continue
            pats = infer_patterns_by_group.get(gid)
            if not pats:
                _issue(
                    issues,
                    "WARN",
                    paper_id,
                    "inference_results",
                    path,
                    "baseline_group_missing_in_surface",
                    f"baseline_group_id={gid} appears in inference_results but not in surface baseline_groups.",
                    baseline_group_id=gid,
                    inference_run_id=irid,
                    row_index=int(i),
                )
                continue
            if not any(fnmatch(sid, p) for p in pats):
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "inference_results",
                    path,
                    "infer_spec_id_not_in_surface_inference_plan",
                    f"spec_id={sid} does not match any inference_plan pattern for baseline_group_id={gid}.",
                    baseline_group_id=gid,
                    inference_run_id=irid,
                    row_index=int(i),
                )

    # run_success/run_error + numeric sanity
    rs = pd.to_numeric(df["run_success"], errors="coerce")
    bad_rs = ~(rs.isin([0, 1]))
    if bad_rs.any():
        _issue(
            issues,
            "ERROR",
            paper_id,
            "inference_results",
            path,
            "invalid_run_success_values",
            f"run_success contains values outside {{0,1}} (n={int(bad_rs.sum())}).",
        )
    rs = rs.fillna(-1).astype(int)
    run_error = df["run_error"].fillna("").astype(str)

    coef = pd.to_numeric(df["coefficient"], errors="coerce")
    se = pd.to_numeric(df["std_error"], errors="coerce")
    pval = pd.to_numeric(df["p_value"], errors="coerce")

    # Base-row coefficient comparison (warning-level)
    base_coef = {}
    base_bg = {}
    if base_ids:
        base = spec_results.copy()
        base["spec_run_id"] = base["spec_run_id"].astype(str)
        base_coef = pd.to_numeric(base["coefficient"], errors="coerce").to_dict()
        base_bg = base["baseline_group_id"].astype(str).to_dict()

    for i, row in df.iterrows():
        gid = str(row.get("baseline_group_id", "")).strip()
        rid = str(row.get("spec_run_id", "")).strip()
        irid = str(row.get("inference_run_id", "")).strip()
        _parse_json_field(
            issues,
            paper_id,
            "inference_results",
            path,
            row.get("coefficient_vector_json", ""),
            baseline_group_id=gid,
            spec_run_id=rid,
            inference_run_id=irid,
            row_index=int(i),
        )

        if rs.iloc[i] == 0:
            if run_error.iloc[i].strip() == "":
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "inference_results",
                    path,
                    "missing_run_error",
                    "run_success=0 but run_error is empty.",
                    baseline_group_id=gid,
                    spec_run_id=rid,
                    inference_run_id=irid,
                    row_index=int(i),
                )
            continue

        if rs.iloc[i] == 1:
            if not np.isfinite(coef.iloc[i]):
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "inference_results",
                    path,
                    "missing_coef_success",
                    "run_success=1 but coefficient is missing/invalid.",
                    baseline_group_id=gid,
                    spec_run_id=rid,
                    inference_run_id=irid,
                    row_index=int(i),
                )
            if not np.isfinite(se.iloc[i]) or not (se.iloc[i] > 0):
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "inference_results",
                    path,
                    "missing_se_success",
                    "run_success=1 but std_error is missing/invalid.",
                    baseline_group_id=gid,
                    spec_run_id=rid,
                    inference_run_id=irid,
                    row_index=int(i),
                )
            if not np.isfinite(pval.iloc[i]) or pval.iloc[i] < 0 or pval.iloc[i] > 1:
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "inference_results",
                    path,
                    "invalid_p_value_success",
                    f"run_success=1 but p_value is missing/invalid (p={pval.iloc[i]}).",
                    baseline_group_id=gid,
                    spec_run_id=rid,
                    inference_run_id=irid,
                    row_index=int(i),
                )

        # Linkage consistency checks
        if rid and rid in base_ids and gid:
            # baseline_group_id should match base spec row
            # (warning-level; could indicate mislinking)
            base_gid = str(spec_results.loc[spec_results["spec_run_id"].astype(str) == rid, "baseline_group_id"].astype(str).iloc[0])
            if base_gid.strip() and base_gid.strip() != gid.strip():
                _issue(
                    issues,
                    "WARN",
                    paper_id,
                    "inference_results",
                    path,
                    "baseline_group_id_mismatch_to_base",
                    f"inference row baseline_group_id={gid} differs from base spec baseline_group_id={base_gid}.",
                    baseline_group_id=gid,
                    spec_run_id=rid,
                    inference_run_id=irid,
                    row_index=int(i),
                )

            # coefficient should match base estimate
            base_c = pd.to_numeric(
                spec_results.loc[spec_results["spec_run_id"].astype(str) == rid, "coefficient"],
                errors="coerce",
            ).iloc[0]
            if np.isfinite(base_c) and np.isfinite(coef.iloc[i]):
                if not np.isclose(float(base_c), float(coef.iloc[i]), rtol=1e-6, atol=1e-10):
                    _issue(
                        issues,
                        "WARN",
                        paper_id,
                        "inference_results",
                        path,
                        "coef_differs_from_base",
                        f"coefficient differs from base spec (base={base_c}, infer={coef.iloc[i]}).",
                        baseline_group_id=gid,
                        spec_run_id=rid,
                        inference_run_id=irid,
                        row_index=int(i),
                    )

    return df


def _validate_verification_outputs(
    paper_id: str,
    verification_dir: Path,
    issues: list[Issue],
    spec_results: pd.DataFrame,
) -> None:
    vdir = verification_dir / paper_id
    if not vdir.exists():
        _issue(
            issues,
            "WARN",
            paper_id,
            "verification",
            vdir,
            "missing_verification_dir",
            "Missing data/verification/{PAPER_ID}/ (post-run verifier outputs not found).",
        )
        return

    map_path = vdir / "verification_spec_map.csv"
    df = _read_csv(map_path, issues, paper_id, "verification")
    if df.empty:
        return
    if not _check_required_cols(df, REQUIRED_VERIFICATION_MAP_COLS, issues, paper_id, "verification", map_path):
        return

    df["spec_run_id"] = df["spec_run_id"].astype(str)
    dup = df.duplicated(subset=["spec_run_id"], keep=False)
    if dup.any():
        ex = df.loc[dup, "spec_run_id"].astype(str).head(8).tolist()
        _issue(
            issues,
            "ERROR",
            paper_id,
            "verification",
            map_path,
            "duplicate_spec_run_id",
            f"verification_spec_map.csv has duplicate spec_run_id (examples: {ex}).",
        )

    # Coverage: spec_run_ids should match exactly.
    if spec_results is not None and not spec_results.empty and "spec_run_id" in spec_results.columns:
        spec_ids = set(spec_results["spec_run_id"].astype(str).tolist())
        map_ids = set(df["spec_run_id"].astype(str).tolist())
        missing = sorted(spec_ids - map_ids)
        extra = sorted(map_ids - spec_ids)
        if missing:
            _issue(
                issues,
                "ERROR",
                paper_id,
                "verification",
                map_path,
                "spec_run_id_missing_in_verification_map",
                f"verification_spec_map.csv is missing {len(missing)} spec_run_id values (examples: {missing[:5]}).",
            )
        if extra:
            _issue(
                issues,
                "ERROR",
                paper_id,
                "verification",
                map_path,
                "spec_run_id_extra_in_verification_map",
                f"verification_spec_map.csv contains {len(extra)} spec_run_id not in specification_results.csv (examples: {extra[:5]}).",
            )

    # Baselines JSON presence + parse
    baselines_path = vdir / "verification_baselines.json"
    if not baselines_path.exists():
        _issue(issues, "WARN", paper_id, "verification", baselines_path, "missing_baselines_json", "Missing verification_baselines.json.")
    else:
        try:
            d = json.loads(baselines_path.read_text(encoding="utf-8", errors="replace"))
        except Exception as e:
            _issue(issues, "ERROR", paper_id, "verification", baselines_path, "invalid_baselines_json", f"Failed to parse JSON: {e}")
        else:
            groups = d.get("baseline_groups", [])
            if not isinstance(groups, list):
                _issue(issues, "ERROR", paper_id, "verification", baselines_path, "invalid_baselines_schema", "baseline_groups must be a list.")
            elif not groups:
                _issue(issues, "WARN", paper_id, "verification", baselines_path, "empty_baselines", "baseline_groups is empty.")

    report_path = vdir / "VERIFICATION_REPORT.md"
    if not report_path.exists():
        _issue(issues, "WARN", paper_id, "verification", report_path, "missing_verification_report", "Missing VERIFICATION_REPORT.md.")


def validate_paper(paper_id: str, extracted_dir: Path, verification_dir: Path) -> list[Issue]:
    issues: list[Issue] = []
    pkg_dir = extracted_dir / paper_id
    if not pkg_dir.exists():
        _issue(issues, "ERROR", paper_id, "package", pkg_dir, "missing_package_dir", "Missing extracted package directory.")
        return issues

    core_patterns, infer_patterns = _validate_surface(paper_id, pkg_dir, issues)
    spec_df = _validate_spec_results(paper_id, pkg_dir, issues, core_patterns)
    _validate_inference_results(paper_id, pkg_dir, issues, spec_df, infer_patterns)
    _validate_verification_outputs(paper_id, verification_dir, issues, spec_df)
    return issues


def _paper_ids_from_extracted(extracted_dir: Path, *, include_incomplete: bool) -> list[str]:
    if not extracted_dir.exists():
        return []
    out: list[str] = []
    for p in extracted_dir.iterdir():
        if not p.is_dir() or p.name.startswith("."):
            continue
        if include_incomplete:
            out.append(p.name)
            continue
        if (p / "SPECIFICATION_SURFACE.json").exists() or (p / "specification_results.csv").exists():
            out.append(p.name)
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate runner + verifier outputs against the spec-tree contract.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--paper-id", action="append", help="Paper ID to validate (repeatable).")
    g.add_argument("--all", action="store_true", help="Validate all extracted packages.")
    ap.add_argument("--extracted-dir", type=str, default=str(DEFAULT_EXTRACTED_DIR), help="Path to extracted packages dir.")
    ap.add_argument("--verification-dir", type=str, default=str(DEFAULT_VERIFICATION_DIR), help="Path to verification dir.")
    ap.add_argument("--report-csv", type=str, default="", help="Write issues to CSV at this path.")
    ap.add_argument("--max-print", type=int, default=80, help="Max issues to print.")
    ap.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include extracted directories even if they have no surface/results artifacts.",
    )

    args = ap.parse_args()
    extracted_dir = Path(args.extracted_dir)
    verification_dir = Path(args.verification_dir)

    paper_ids = args.paper_id if args.paper_id else _paper_ids_from_extracted(extracted_dir, include_incomplete=bool(args.include_incomplete))
    if not paper_ids:
        print(f"No paper IDs found under {extracted_dir}", file=sys.stderr)
        return 1

    all_issues: list[Issue] = []
    for pid in paper_ids:
        all_issues.extend(validate_paper(pid, extracted_dir, verification_dir))

    # Summaries
    df = pd.DataFrame([asdict(x) for x in all_issues]) if all_issues else pd.DataFrame(columns=[f.name for f in Issue.__dataclass_fields__.values()])

    n_err = int((df["severity"] == "ERROR").sum()) if len(df) else 0
    n_warn = int((df["severity"] == "WARN").sum()) if len(df) else 0

    if args.report_csv:
        out_path = Path(args.report_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Wrote report: {out_path}")

    if len(df) == 0:
        print(f"OK: validated {len(paper_ids)} papers with 0 issues.")
        return 0

    # Per-paper breakdown
    by_paper = df.groupby(["paper_id", "severity"]).size().unstack(fill_value=0)
    for c in ["ERROR", "WARN"]:
        if c not in by_paper.columns:
            by_paper[c] = 0
    by_paper = by_paper.reset_index().sort_values(["ERROR", "WARN", "paper_id"], ascending=[False, False, True])
    cols = ["paper_id"] + [c for c in ["ERROR", "WARN"] if c in by_paper.columns]
    print("\nSummary by paper:")
    print(by_paper[cols].to_string(index=False))

    # Print a small sample of issues
    print("\nIssues (first rows):")
    show = df.sort_values(["severity", "paper_id", "area", "code"], ascending=[True, True, True, True]).head(int(args.max_print))
    for _, r in show.iterrows():
        loc_parts = []
        if r.get("baseline_group_id"):
            loc_parts.append(f"gid={r['baseline_group_id']}")
        if r.get("spec_run_id"):
            loc_parts.append(f"spec_run_id={r['spec_run_id']}")
        if r.get("inference_run_id"):
            loc_parts.append(f"infer_run_id={r['inference_run_id']}")
        if pd.notna(r.get("row_index")):
            loc_parts.append(f"row={int(r['row_index'])}")
        loc = (" [" + ", ".join(loc_parts) + "]") if loc_parts else ""
        print(f"{r['severity']}: {r['paper_id']} {r['area']} {r['code']}{loc} -> {r['message']} ({r['path']})")

    if len(df) > len(show):
        print(f"... {len(df) - len(show)} more issues not shown")

    print(f"\nTotals: ERROR={n_err}, WARN={n_warn}")
    return 1 if n_err > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
