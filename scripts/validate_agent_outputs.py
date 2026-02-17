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
import hashlib
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
SPEC_TREE_DIR = ROOT / "specification_tree"


REQUIRED_SUCCESS_PAYLOAD_KEYS: tuple[str, ...] = (
    "coefficients",
    "inference",
    "software",
    "surface_hash",
)

ALLOWED_PAYLOAD_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        # Required contract keys
        "coefficients",
        "inference",
        "software",
        "surface_hash",
        # Failure plumbing
        "error",
        "error_details",
        "partial",
        # Core RC axis blocks
        "controls",
        "sample",
        "fixed_effects",
        "preprocess",
        "estimation",
        "weights",
        "data_construction",
        "functional_form",
        "joint",
        # Common audit/metadata blocks
        "focal",
        "bundle",
        "warnings",
        "notes",
        # Flexible extension points (keep top-level schema stable)
        "design",
        "extra",
        "universe",
        "sampling",
    }
)

REQUIRED_ERROR_DETAILS_FIELDS: tuple[str, ...] = (
    "stage",
    "exception_type",
    "exception_message",
)

RC_AXIS_TO_REQUIRED_BLOCK: dict[str, str] = {
    "controls": "controls",
    "sample": "sample",
    "fe": "fixed_effects",
    "preprocess": "preprocess",
    "estimation": "estimation",
    "weights": "weights",
    "data": "data_construction",
    "form": "functional_form",
    "joint": "joint",
}


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

# Optional non-core tables (validated only if present).
REQUIRED_EXPLORATION_COLS: list[str] = [
    "paper_id",
    "exploration_run_id",
    "spec_run_id",  # optional link to a base spec row; column required for schema stability
    "spec_id",
    "spec_tree_path",
    "baseline_group_id",
    "exploration_json",
    "run_success",
    "run_error",
]

REQUIRED_SENSITIVITY_COLS: list[str] = [
    "paper_id",
    "sensitivity_run_id",
    "spec_run_id",  # optional link to a base spec row
    "spec_id",
    "spec_tree_path",
    "baseline_group_id",
    "sensitivity_scope",
    "sensitivity_context_id",
    "sensitivity_json",
    "run_success",
    "run_error",
]

REQUIRED_POSTPROCESS_COLS: list[str] = [
    "paper_id",
    "postprocess_run_id",
    "spec_id",
    "spec_tree_path",
    "baseline_group_id",
    "postprocess_json",
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


def _compute_surface_hash(surface: dict) -> str:
    canon = json.dumps(surface, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    h = hashlib.sha256(canon.encode("utf-8")).hexdigest()
    return f"sha256:{h}"


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
            "coefficient_vector_json is empty; expected a valid JSON object. For run_success=1 rows include reserved keys (coefficients/inference/software/surface_hash). For run_success=0 rows include at least {'error': run_error, 'error_details': {...}}.",
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


def _canonicalize_spec_tree_path(stp: str) -> tuple[str, str]:
    """
    Returns (file_part, anchor_part).

    Accepts either:
      - designs/foo.md#anchor
      - modules/bar.md#anchor
      - specification_tree/designs/foo.md#anchor
    """
    stp = (stp or "").strip()
    if stp == "custom":
        return ("custom", "")

    file_part, anchor = (stp.split("#", 1) + [""])[:2]
    file_part = file_part.strip()
    anchor = anchor.strip()
    prefix = "specification_tree/"
    if file_part.startswith(prefix):
        file_part = file_part[len(prefix) :]
    return (file_part, anchor)


def _validate_spec_tree_path(
    *,
    issues: list[Issue],
    paper_id: str,
    area: str,
    path: Path,
    spec_tree_path: str,
    spec_id: str,
    baseline_group_id: str = "",
    spec_run_id: str = "",
    inference_run_id: str = "",
    row_index: int | None = None,
) -> None:
    stp = _safe_str(spec_tree_path).strip()
    sid = str(spec_id or "").strip()

    # Allow only "custom" as the non-tree escape hatch.
    if not stp:
        _issue(
            issues,
            "ERROR",
            paper_id,
            area,
            path,
            "invalid_spec_tree_path",
            f"spec_tree_path is empty (spec_id={sid}).",
            baseline_group_id=baseline_group_id,
            spec_run_id=spec_run_id,
            inference_run_id=inference_run_id,
            row_index=row_index,
        )
        return
    if stp == "custom":
        return

    file_part, anchor = _canonicalize_spec_tree_path(stp)
    if (not file_part) or (".md" not in file_part):
        _issue(
            issues,
            "ERROR",
            paper_id,
            area,
            path,
            "invalid_spec_tree_path",
            f"spec_tree_path must reference a spec-tree .md file (or be 'custom'); got '{stp}' (spec_id={sid}).",
            baseline_group_id=baseline_group_id,
            spec_run_id=spec_run_id,
            inference_run_id=inference_run_id,
            row_index=row_index,
        )
        return

    p = Path(file_part)
    if p.is_absolute():
        _issue(
            issues,
            "ERROR",
            paper_id,
            area,
            path,
            "absolute_spec_tree_path",
            f"spec_tree_path must be repo-relative (got absolute path '{stp}').",
            baseline_group_id=baseline_group_id,
            spec_run_id=spec_run_id,
            inference_run_id=inference_run_id,
            row_index=row_index,
        )
        return
    if any(part == ".." for part in p.parts):
        _issue(
            issues,
            "ERROR",
            paper_id,
            area,
            path,
            "spec_tree_path_traversal",
            f"spec_tree_path must not contain '..' segments ('{stp}').",
            baseline_group_id=baseline_group_id,
            spec_run_id=spec_run_id,
            inference_run_id=inference_run_id,
            row_index=row_index,
        )
        return

    if not anchor:
        _issue(
            issues,
            "WARN",
            paper_id,
            area,
            path,
            "spec_tree_path_missing_anchor",
            f"spec_tree_path is missing a #section-anchor ('{stp}'); add an anchor when possible (spec_id={sid}).",
            baseline_group_id=baseline_group_id,
            spec_run_id=spec_run_id,
            inference_run_id=inference_run_id,
            row_index=row_index,
        )

    full = SPEC_TREE_DIR / p
    if not full.exists():
        _issue(
            issues,
            "ERROR",
            paper_id,
            area,
            path,
            "spec_tree_path_file_missing",
            f"spec_tree_path file does not exist under specification_tree/ ('{file_part}' from '{stp}', spec_id={sid}).",
            baseline_group_id=baseline_group_id,
            spec_run_id=spec_run_id,
            inference_run_id=inference_run_id,
            row_index=row_index,
        )


def _validate_surface(
    paper_id: str,
    pkg_dir: Path,
    issues: list[Issue],
) -> tuple[dict[str, list[str] | None], dict[str, list[str]], dict[str, str]]:
    """
    Returns:
      allowed_core_spec_patterns_by_group
      allowed_infer_spec_patterns_by_group
      canonical_inference_spec_id_by_group
    """
    path = pkg_dir / "SPECIFICATION_SURFACE.json"
    if not path.exists():
        _issue(issues, "ERROR", paper_id, "surface", path, "missing_surface", "Missing SPECIFICATION_SURFACE.json.")
        return {}, {}, {}

    try:
        d = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        _issue(issues, "ERROR", paper_id, "surface", path, "invalid_surface_json", f"Failed to parse JSON: {e}")
        return {}, {}, {}

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
        return {}, {}, {}

    core_patterns: dict[str, list[str] | None] = {}
    infer_patterns: dict[str, list[str]] = {}
    canon_infer_by_gid: dict[str, str] = {}

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
        else:
            dc = str(g.get("design_code", "")).strip()
            design_path = SPEC_TREE_DIR / "designs" / f"{dc}.md"
            if not design_path.exists():
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "surface",
                    path,
                    "invalid_design_code",
                    f"baseline_group.design_code='{dc}' does not match a file under specification_tree/designs/ (expected {design_path}).",
                    baseline_group_id=gid,
                )

        da = g.get("design_audit", None)
        if not isinstance(da, dict) or not da:
            _issue(
                issues,
                "ERROR",
                paper_id,
                "surface",
                path,
                "missing_design_audit",
                "baseline_group.design_audit is missing/empty (required).",
                baseline_group_id=gid,
            )
        elif not str(da.get("estimator", "")).strip():
            _issue(
                issues,
                "ERROR",
                paper_id,
                "surface",
                path,
                "missing_design_audit_estimator",
                "baseline_group.design_audit.estimator is missing/empty (required).",
                baseline_group_id=gid,
            )
        elif set(da.keys()) == {"estimator"}:
            _issue(
                issues,
                "ERROR",
                paper_id,
                "surface",
                path,
                "design_audit_estimator_only",
                "baseline_group.design_audit only contains 'estimator'. Add at least one other design-defining field (e.g., fixed_effects, cluster_vars, weights, panel_unit/panel_time, randomization_unit, model_formula).",
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
        else:
            bad_design = [str(s) for s in design_spec_ids if str(s).strip() and (not str(s).strip().startswith("design/"))]
            if bad_design:
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "surface",
                    path,
                    "invalid_design_spec_id_pattern",
                    f"core_universe.design_spec_ids must start with 'design/' (examples: {bad_design[:5]}).",
                    baseline_group_id=gid,
                )
            bad_rc = [str(s) for s in rc_spec_ids if str(s).strip() and (not str(s).strip().startswith("rc/"))]
            if bad_rc:
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "surface",
                    path,
                    "invalid_rc_spec_id_pattern",
                    f"core_universe.rc_spec_ids must start with 'rc/' (examples: {bad_rc[:5]}).",
                    baseline_group_id=gid,
                )

        # Only define matching patterns when core_universe exists and is parseable.
        if core_patterns.get(gid, "__unset__") is not None:
            baseline_spec_ids = core.get("baseline_spec_ids", [])
            if not isinstance(baseline_spec_ids, list):
                baseline_spec_ids = []
            else:
                bad_base = [
                    str(s)
                    for s in baseline_spec_ids
                    if str(s).strip()
                    and (not (str(s).strip() == "baseline" or str(s).strip().startswith(("baseline__", "baseline/"))))
                ]
                if bad_base:
                    _issue(
                        issues,
                        "ERROR",
                        paper_id,
                        "surface",
                        path,
                        "invalid_baseline_spec_id_pattern",
                        f"core_universe.baseline_spec_ids must equal 'baseline' or start with 'baseline__'/'baseline/' (examples: {bad_base[:5]}).",
                        baseline_group_id=gid,
                    )
            pats = ["baseline"] + [str(s) for s in baseline_spec_ids] + [str(s) for s in design_spec_ids] + [str(s) for s in rc_spec_ids]
            if (len(design_spec_ids) == 0) and (len(rc_spec_ids) == 0):
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
        elif not canon_id.startswith("infer/"):
            _issue(
                issues,
                "ERROR",
                paper_id,
                "surface",
                path,
                "invalid_canonical_inference_spec_id",
                f"inference_plan.canonical.spec_id must start with 'infer/' (got '{canon_id}').",
                baseline_group_id=gid,
            )
        else:
            canon_infer_by_gid[gid] = canon_id

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
        bad_var = [vid for vid in var_ids if vid and (not vid.startswith("infer/"))]
        if bad_var:
            _issue(
                issues,
                "ERROR",
                paper_id,
                "surface",
                path,
                "invalid_inference_variant_spec_id",
                f"inference_plan.variants spec_id must start with 'infer/' (examples: {bad_var[:5]}).",
                baseline_group_id=gid,
            )
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

    return core_patterns, infer_patterns, canon_infer_by_gid


def _validate_spec_results(
    paper_id: str,
    pkg_dir: Path,
    issues: list[Issue],
    core_patterns_by_group: dict[str, list[str] | None],
    canon_infer_by_gid: dict[str, str],
) -> pd.DataFrame:
    path = pkg_dir / "specification_results.csv"
    df = _read_csv(path, issues, paper_id, "spec_results")
    if df.empty:
        return df

    _check_required_cols(df, REQUIRED_SPEC_COLS, issues, paper_id, "spec_results", path)

    expected_surface_hash = ""
    design_code_by_gid: dict[str, str] = {}
    design_audit_by_gid: dict[str, dict] = {}
    surf_path = pkg_dir / "SPECIFICATION_SURFACE.json"
    if surf_path.exists():
        try:
            surface = json.loads(surf_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            surface = {}
        if isinstance(surface, dict):
            expected_surface_hash = _compute_surface_hash(surface)
            for g in surface.get("baseline_groups", []) or []:
                if not isinstance(g, dict):
                    continue
                gid = str(g.get("baseline_group_id", "")).strip()
                dc = str(g.get("design_code", "")).strip()
                if gid and dc:
                    design_code_by_gid[gid] = dc
                da = g.get("design_audit")
                if gid and isinstance(da, dict) and da:
                    design_audit_by_gid[gid] = da

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
    n_obs = pd.to_numeric(df.get("n_obs"), errors="coerce")
    r2 = pd.to_numeric(df.get("r_squared"), errors="coerce")

    for i, row in df.iterrows():
        gid = str(row.get("baseline_group_id", "")).strip()
        rid = str(row.get("spec_run_id", "")).strip()
        sid = str(row.get("spec_id", "")).strip()

        _validate_spec_tree_path(
            issues=issues,
            paper_id=paper_id,
            area="spec_results",
            path=path,
            spec_tree_path=row.get("spec_tree_path", ""),
            spec_id=sid,
            baseline_group_id=gid,
            spec_run_id=rid,
            row_index=int(i),
        )

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

        s = _safe_str(row.get("coefficient_vector_json", "")).strip()
        try:
            payload = json.loads(s) if s else None
        except Exception:
            payload = None

        if (rs.iloc[i] == 1) and expected_surface_hash and isinstance(payload, dict):
            sh = str(payload.get("surface_hash", "")).strip()
            if sh and (sh != expected_surface_hash):
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "spec_results",
                    path,
                    "surface_hash_mismatch",
                    f"coefficient_vector_json.surface_hash='{sh}' does not match SPECIFICATION_SURFACE.json hash='{expected_surface_hash}'.",
                    baseline_group_id=gid,
                    spec_run_id=rid,
                    row_index=int(i),
                )

        if (rs.iloc[i] == 1) and isinstance(payload, dict) and sid:
            dc = design_code_by_gid.get(gid, "")
            if dc:
                dblk = payload.get("design")
                drow = dblk.get(dc) if isinstance(dblk, dict) else None
                has = isinstance(drow, dict) and bool(drow)
                if not has:
                    _issue(
                        issues,
                        "ERROR",
                        paper_id,
                        "spec_results",
                        path,
                        "missing_design_block",
                        f"run_success=1 but coefficient_vector_json.design.{dc} is missing/empty (required; see specification_tree/DESIGN_AUDIT_FIELDS.md).",
                        baseline_group_id=gid,
                        spec_run_id=rid,
                        row_index=int(i),
                    )
                else:
                    surface_da = design_audit_by_gid.get(gid, {})
                    if isinstance(surface_da, dict) and surface_da:
                        is_baseline_or_rc = (sid == "baseline") or sid.startswith(("baseline__", "baseline/")) or sid.startswith("rc/")
                        for k, v in surface_da.items():
                            if k not in drow:
                                _issue(
                                    issues,
                                    "ERROR",
                                    paper_id,
                                    "spec_results",
                                    path,
                                    "design_audit_missing_key",
                                    f"coefficient_vector_json.design.{dc} is missing key '{k}' from surface design_audit.",
                                    baseline_group_id=gid,
                                    spec_run_id=rid,
                                    row_index=int(i),
                                )
                                continue
                            if is_baseline_or_rc and (drow.get(k) != v):
                                _issue(
                                    issues,
                                    "ERROR",
                                    paper_id,
                                    "spec_results",
                                    path,
                                    "design_audit_value_mismatch",
                                    f"coefficient_vector_json.design.{dc}.{k} does not match surface design_audit for baseline/rc row.",
                                    baseline_group_id=gid,
                                    spec_run_id=rid,
                                    row_index=int(i),
                                )

        # run_error should be a short single-line summary; put rich details in JSON.
        re_msg = run_error.iloc[i] if i in run_error.index else ""
        if ("\n" in re_msg) or ("\r" in re_msg):
            _issue(
                issues,
                "WARN",
                paper_id,
                "spec_results",
                path,
                "run_error_contains_newline",
                "run_error contains a newline; keep run_error as a single-line summary and store traceback/details under coefficient_vector_json.error_details.",
                baseline_group_id=gid,
                spec_run_id=rid,
                row_index=int(i),
            )
        if (rs.iloc[i] == 1) and re_msg.strip():
            _issue(
                issues,
                "WARN",
                paper_id,
                "spec_results",
                path,
                "run_error_nonempty_on_success",
                "run_success=1 but run_error is non-empty; set run_error='' and store any warnings under coefficient_vector_json.warnings.",
                baseline_group_id=gid,
                spec_run_id=rid,
                row_index=int(i),
            )

        if rs.iloc[i] == 0:
            # Failures must carry an explicit error payload in coefficient_vector_json.
            # (The row-level run_error column is the short summary; JSON can include richer context.)
            if "coefficient_vector_json" in df.columns:
                if isinstance(payload, dict):
                    if not str(payload.get("error", "")).strip():
                        _issue(
                            issues,
                            "ERROR",
                            paper_id,
                            "spec_results",
                            path,
                            "missing_failure_error_payload",
                            "run_success=0 but coefficient_vector_json lacks a non-empty 'error' field.",
                            baseline_group_id=gid,
                            spec_run_id=rid,
                            row_index=int(i),
                        )
                    unknown = sorted(set(payload.keys()) - set(ALLOWED_PAYLOAD_TOP_LEVEL_KEYS))
                    if unknown:
                        _issue(
                            issues,
                            "ERROR",
                            paper_id,
                            "spec_results",
                            path,
                            "unknown_payload_keys",
                            f"coefficient_vector_json contains unknown top-level keys: {unknown}. Move paper-specific fields under coefficient_vector_json.extra (dict) to keep the schema stable.",
                            baseline_group_id=gid,
                            spec_run_id=rid,
                            row_index=int(i),
                        )
                    ed = payload.get("error_details")
                    if not isinstance(ed, dict) or not ed:
                        _issue(
                            issues,
                            "ERROR",
                            paper_id,
                            "spec_results",
                            path,
                            "missing_error_details",
                            "run_success=0 but coefficient_vector_json.error_details is missing/empty (required).",
                            baseline_group_id=gid,
                            spec_run_id=rid,
                            row_index=int(i),
                        )
                    else:
                        missing = [k for k in REQUIRED_ERROR_DETAILS_FIELDS if not str(ed.get(k, "")).strip()]
                        if missing:
                            _issue(
                                issues,
                                "ERROR",
                                paper_id,
                                "spec_results",
                                path,
                                "missing_error_details_fields",
                                f"run_success=0 but error_details is missing required fields: {missing}.",
                                baseline_group_id=gid,
                                spec_run_id=rid,
                                row_index=int(i),
                            )
                elif payload is not None:
                    _issue(
                        issues,
                        "ERROR",
                        paper_id,
                        "spec_results",
                        path,
                        "invalid_failure_error_payload",
                        "run_success=0 but coefficient_vector_json must be a JSON object with an 'error' field.",
                        baseline_group_id=gid,
                        spec_run_id=rid,
                        row_index=int(i),
                    )

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
            # Failures must not report scalar numeric fields in the CSV (store partials in JSON instead).
            if (
                np.isfinite(coef.iloc[i])
                or np.isfinite(se.iloc[i])
                or np.isfinite(pval.iloc[i])
                or np.isfinite(ci_lo.iloc[i])
                or np.isfinite(ci_hi.iloc[i])
                or np.isfinite(n_obs.iloc[i])
                or np.isfinite(r2.iloc[i])
            ):
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "spec_results",
                    path,
                    "failure_has_numeric_fields",
                    "run_success=0 but scalar numeric fields are non-missing; set them to NaN and store partial outputs under coefficient_vector_json.partial.",
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

            # Successful rows must use the reserved-key payload schema.
            if not isinstance(payload, dict):
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "spec_results",
                    path,
                    "invalid_success_payload",
                    "run_success=1 but coefficient_vector_json must be a JSON object (dict).",
                    baseline_group_id=gid,
                    spec_run_id=rid,
                    row_index=int(i),
                )
            else:
                missing_keys = [k for k in REQUIRED_SUCCESS_PAYLOAD_KEYS if k not in payload]
                if missing_keys:
                    _issue(
                        issues,
                        "ERROR",
                        paper_id,
                        "spec_results",
                        path,
                        "missing_required_payload_keys",
                        f"run_success=1 but coefficient_vector_json is missing required keys: {missing_keys}",
                        baseline_group_id=gid,
                        spec_run_id=rid,
                        row_index=int(i),
                    )

                unknown = sorted(set(payload.keys()) - set(ALLOWED_PAYLOAD_TOP_LEVEL_KEYS))
                if unknown:
                    _issue(
                        issues,
                        "ERROR",
                        paper_id,
                        "spec_results",
                        path,
                        "unknown_payload_keys",
                        f"coefficient_vector_json contains unknown top-level keys: {unknown}. Move paper-specific fields under coefficient_vector_json.extra (dict) to keep the schema stable.",
                        baseline_group_id=gid,
                        spec_run_id=rid,
                        row_index=int(i),
                    )

                coefs = payload.get("coefficients")
                if not isinstance(coefs, dict):
                    _issue(
                        issues,
                        "ERROR",
                        paper_id,
                        "spec_results",
                        path,
                        "invalid_coefficients_block",
                        "coefficient_vector_json.coefficients must be a JSON object (dict).",
                        baseline_group_id=gid,
                        spec_run_id=rid,
                        row_index=int(i),
                    )
                else:
                    bad_type_keys: list[str] = []
                    nonfinite_keys: list[str] = []
                    for k, v in coefs.items():
                        if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
                            bad_type_keys.append(str(k))
                            continue
                        try:
                            fv = float(v)
                        except Exception:
                            bad_type_keys.append(str(k))
                            continue
                        if not np.isfinite(fv):
                            nonfinite_keys.append(str(k))
                    if bad_type_keys:
                        _issue(
                            issues,
                            "ERROR",
                            paper_id,
                            "spec_results",
                            path,
                            "invalid_coefficient_values",
                            f"coefficient_vector_json.coefficients must map parameter -> numeric scalar; found non-numeric values (examples: {bad_type_keys[:5]}).",
                            baseline_group_id=gid,
                            spec_run_id=rid,
                            row_index=int(i),
                        )
                    if nonfinite_keys:
                        _issue(
                            issues,
                            "WARN",
                            paper_id,
                            "spec_results",
                            path,
                            "nonfinite_coefficient_values",
                            f"coefficient_vector_json.coefficients contains non-finite values (examples: {nonfinite_keys[:5]}).",
                            baseline_group_id=gid,
                            spec_run_id=rid,
                            row_index=int(i),
                        )

                inf = payload.get("inference")
                if not isinstance(inf, dict) or not str(inf.get("spec_id", "")).strip():
                    _issue(
                        issues,
                        "ERROR",
                        paper_id,
                        "spec_results",
                        path,
                        "missing_inference_block",
                        "run_success=1 but coefficient_vector_json.inference.spec_id is missing/empty.",
                        baseline_group_id=gid,
                        spec_run_id=rid,
                        row_index=int(i),
                    )
                else:
                    inf_sid = str(inf.get("spec_id", "")).strip()
                    if not inf_sid.startswith("infer/"):
                        _issue(
                            issues,
                            "ERROR",
                            paper_id,
                            "spec_results",
                            path,
                            "invalid_inference_spec_id",
                            f"coefficient_vector_json.inference.spec_id must start with 'infer/' (got '{inf_sid}').",
                            baseline_group_id=gid,
                            spec_run_id=rid,
                            row_index=int(i),
                        )
                    expected = str(canon_infer_by_gid.get(gid, "") or "").strip()
                    if expected and (inf_sid != expected):
                        _issue(
                            issues,
                            "ERROR",
                            paper_id,
                            "spec_results",
                            path,
                            "non_canonical_inference_used",
                            f"Estimate rows must use the baseline group's canonical inference choice. Expected inference.spec_id='{expected}' but got '{inf_sid}'.",
                            baseline_group_id=gid,
                            spec_run_id=rid,
                            row_index=int(i),
                        )

                sw = payload.get("software")
                if not isinstance(sw, dict):
                    _issue(
                        issues,
                        "ERROR",
                        paper_id,
                        "spec_results",
                        path,
                        "missing_software_block",
                        "run_success=1 but coefficient_vector_json.software is missing/invalid.",
                        baseline_group_id=gid,
                        spec_run_id=rid,
                        row_index=int(i),
                    )
                else:
                    if not str(sw.get("runner_language", "")).strip() or not str(sw.get("runner_version", "")).strip():
                        _issue(
                            issues,
                            "ERROR",
                            paper_id,
                            "spec_results",
                            path,
                            "invalid_software_block",
                            "software block must include non-empty runner_language and runner_version.",
                            baseline_group_id=gid,
                            spec_run_id=rid,
                            row_index=int(i),
                        )
                    pkgs = sw.get("packages")
                    if not isinstance(pkgs, dict) or len(pkgs) == 0:
                        _issue(
                            issues,
                            "ERROR",
                            paper_id,
                            "spec_results",
                            path,
                            "missing_software_packages",
                            "software.packages must be a non-empty object mapping package -> version.",
                            baseline_group_id=gid,
                            spec_run_id=rid,
                            row_index=int(i),
                        )

                sh = payload.get("surface_hash")
                if not str(sh or "").strip():
                    _issue(
                        issues,
                        "ERROR",
                        paper_id,
                        "spec_results",
                        path,
                        "missing_surface_hash",
                        "run_success=1 but coefficient_vector_json.surface_hash is missing/empty.",
                        baseline_group_id=gid,
                        spec_run_id=rid,
                        row_index=int(i),
                    )
                else:
                    sh_s = str(sh).strip()
                    ok = sh_s.startswith("sha256:") and (len(sh_s) == 7 + 64) and all(c in "0123456789abcdef" for c in sh_s[7:].lower())
                    if not ok:
                        _issue(
                            issues,
                            "ERROR",
                            paper_id,
                            "spec_results",
                            path,
                            "invalid_surface_hash_format",
                            f"surface_hash must look like 'sha256:<64 hex>'; got '{sh_s}'.",
                            baseline_group_id=gid,
                            spec_run_id=rid,
                            row_index=int(i),
                        )

                # RC rows must include their axis-appropriate audit block.
                if sid.startswith("rc/"):
                    parts = sid.split("/")
                    axis = parts[1] if len(parts) > 1 else ""
                    block = RC_AXIS_TO_REQUIRED_BLOCK.get(axis)
                    if block:
                        b = payload.get(block)
                        if not isinstance(b, dict) or not b:
                            _issue(
                                issues,
                                "ERROR",
                                paper_id,
                                "spec_results",
                                path,
                                "missing_rc_axis_block",
                                f"{sid} requires a non-empty coefficient_vector_json.{block} object.",
                                baseline_group_id=gid,
                                spec_run_id=rid,
                                row_index=int(i),
                            )
                        else:
                            bspec = str(b.get("spec_id", "")).strip()
                            if bspec != sid:
                                _issue(
                                    issues,
                                    "ERROR",
                                    paper_id,
                                    "spec_results",
                                    path,
                                    "rc_axis_block_spec_id_mismatch",
                                    f"{sid} requires coefficient_vector_json.{block}.spec_id == '{sid}' (got '{bspec}').",
                                    baseline_group_id=gid,
                                    spec_run_id=rid,
                                    row_index=int(i),
                                )

                # rc/form: enforce functional_form interpretability fields.
                if sid.startswith("rc/form/"):
                    ff = payload.get("functional_form")
                    if not isinstance(ff, dict) or not ff:
                        _issue(
                            issues,
                            "ERROR",
                            paper_id,
                            "spec_results",
                            path,
                            "missing_functional_form_block",
                            "rc/form row requires a non-empty functional_form object in coefficient_vector_json.",
                            baseline_group_id=gid,
                            spec_run_id=rid,
                            row_index=int(i),
                        )
                    else:
                        if not str(ff.get("interpretation", "")).strip():
                            _issue(
                                issues,
                                "ERROR",
                                paper_id,
                                "spec_results",
                                path,
                                "missing_functional_form_interpretation",
                                "rc/form row requires a non-empty functional_form.interpretation string.",
                                baseline_group_id=gid,
                                spec_run_id=rid,
                                row_index=int(i),
                            )
                        op = str(ff.get("operation", "")).strip().lower()
                        if op in {"binarize", "threshold"}:
                            thresh = ff.get("threshold", "")
                            direction = str(ff.get("direction", "")).strip()
                            units = str(ff.get("units", "")).strip()
                            if thresh in ("", None) or (not direction) or (not units):
                                _issue(
                                    issues,
                                    "ERROR",
                                    paper_id,
                                    "spec_results",
                                    path,
                                    "missing_functional_form_threshold_fields",
                                    "binarize/threshold functional_form requires threshold, direction, and units fields.",
                                    baseline_group_id=gid,
                                    spec_run_id=rid,
                                    row_index=int(i),
                                )
                            else:
                                # Threshold metadata must be explicit (avoid placeholders like 'unspecified').
                                placeholder = {"unspecified", "unknown", "n/a"}
                                if str(thresh).strip().lower() in placeholder or direction.lower() in placeholder:
                                    _issue(
                                        issues,
                                        "ERROR",
                                        paper_id,
                                        "spec_results",
                                        path,
                                        "functional_form_threshold_unspecified",
                                        "binarize/threshold functional_form has placeholder values for threshold/direction; record the actual cutoff and inequality when possible.",
                                        baseline_group_id=gid,
                                        spec_run_id=rid,
                                        row_index=int(i),
                                    )
                                if direction and direction.lower() not in placeholder and direction not in {">", ">=", "<", "<="}:
                                    _issue(
                                        issues,
                                        "WARN",
                                        paper_id,
                                        "spec_results",
                                        path,
                                        "functional_form_direction_unusual",
                                        f"functional_form.direction='{direction}' is unusual; prefer one of {{'>','>=','<','<='}}.",
                                        baseline_group_id=gid,
                                        spec_run_id=rid,
                                        row_index=int(i),
                                    )
                                if units.lower() in placeholder or units.lower().startswith("same units as"):
                                    _issue(
                                        issues,
                                        "ERROR",
                                        paper_id,
                                        "spec_results",
                                        path,
                                        "functional_form_units_placeholder",
                                        "functional_form.units looks like a placeholder; record the actual units (or 'unitless') when possible.",
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
                    "No baseline row (spec_id starts with 'baseline') found for this baseline_group_id.",
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

    expected_surface_hash = ""
    surf_path = pkg_dir / "SPECIFICATION_SURFACE.json"
    if surf_path.exists():
        try:
            surface = json.loads(surf_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            surface = {}
        if isinstance(surface, dict):
            expected_surface_hash = _compute_surface_hash(surface)

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
    ci_lo = pd.to_numeric(df.get("ci_lower"), errors="coerce")
    ci_hi = pd.to_numeric(df.get("ci_upper"), errors="coerce")
    n_obs = pd.to_numeric(df.get("n_obs"), errors="coerce")
    r2 = pd.to_numeric(df.get("r_squared"), errors="coerce")

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
        sid = str(row.get("spec_id", "")).strip()
        _validate_spec_tree_path(
            issues=issues,
            paper_id=paper_id,
            area="inference_results",
            path=path,
            spec_tree_path=row.get("spec_tree_path", ""),
            spec_id=sid,
            baseline_group_id=gid,
            spec_run_id=rid,
            inference_run_id=irid,
            row_index=int(i),
        )

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

        s = _safe_str(row.get("coefficient_vector_json", "")).strip()
        try:
            payload = json.loads(s) if s else None
        except Exception:
            payload = None

        if (rs.iloc[i] == 1) and expected_surface_hash and isinstance(payload, dict):
            sh = str(payload.get("surface_hash", "")).strip()
            if sh and (sh != expected_surface_hash):
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "inference_results",
                    path,
                    "surface_hash_mismatch",
                    f"coefficient_vector_json.surface_hash='{sh}' does not match SPECIFICATION_SURFACE.json hash='{expected_surface_hash}'.",
                    baseline_group_id=gid,
                    spec_run_id=rid,
                    inference_run_id=irid,
                    row_index=int(i),
                )

        # run_error should be a short single-line summary; put rich details in JSON.
        re_msg = run_error.iloc[i] if i in run_error.index else ""
        if ("\n" in re_msg) or ("\r" in re_msg):
            _issue(
                issues,
                "WARN",
                paper_id,
                "inference_results",
                path,
                "run_error_contains_newline",
                "run_error contains a newline; keep run_error as a single-line summary and store traceback/details under coefficient_vector_json.error_details.",
                baseline_group_id=gid,
                spec_run_id=rid,
                inference_run_id=irid,
                row_index=int(i),
            )
        if (rs.iloc[i] == 1) and re_msg.strip():
            _issue(
                issues,
                "WARN",
                paper_id,
                "inference_results",
                path,
                "run_error_nonempty_on_success",
                "run_success=1 but run_error is non-empty; set run_error='' and store any warnings under coefficient_vector_json.warnings.",
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
            # Failures must carry an explicit error payload in coefficient_vector_json.
            if isinstance(payload, dict):
                if not str(payload.get("error", "")).strip():
                    _issue(
                        issues,
                        "ERROR",
                        paper_id,
                        "inference_results",
                        path,
                        "missing_failure_error_payload",
                        "run_success=0 but coefficient_vector_json lacks a non-empty 'error' field.",
                        baseline_group_id=gid,
                        spec_run_id=rid,
                        inference_run_id=irid,
                        row_index=int(i),
                    )
                unknown = sorted(set(payload.keys()) - set(ALLOWED_PAYLOAD_TOP_LEVEL_KEYS))
                if unknown:
                    _issue(
                        issues,
                        "ERROR",
                        paper_id,
                        "inference_results",
                        path,
                        "unknown_payload_keys",
                        f"coefficient_vector_json contains unknown top-level keys: {unknown}. Move paper-specific fields under coefficient_vector_json.extra (dict) to keep the schema stable.",
                        baseline_group_id=gid,
                        spec_run_id=rid,
                        inference_run_id=irid,
                        row_index=int(i),
                    )
                ed = payload.get("error_details")
                if not isinstance(ed, dict) or not ed:
                    _issue(
                        issues,
                        "ERROR",
                        paper_id,
                        "inference_results",
                        path,
                        "missing_error_details",
                        "run_success=0 but coefficient_vector_json.error_details is missing/empty (required).",
                        baseline_group_id=gid,
                        spec_run_id=rid,
                        inference_run_id=irid,
                        row_index=int(i),
                    )
                else:
                    missing = [k for k in REQUIRED_ERROR_DETAILS_FIELDS if not str(ed.get(k, "")).strip()]
                    if missing:
                        _issue(
                            issues,
                            "ERROR",
                            paper_id,
                            "inference_results",
                            path,
                            "missing_error_details_fields",
                            f"run_success=0 but error_details is missing required fields: {missing}.",
                            baseline_group_id=gid,
                            spec_run_id=rid,
                            inference_run_id=irid,
                            row_index=int(i),
                        )
            elif payload is not None:
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "inference_results",
                    path,
                    "invalid_failure_error_payload",
                    "run_success=0 but coefficient_vector_json must be a JSON object with an 'error' field.",
                    baseline_group_id=gid,
                    spec_run_id=rid,
                    inference_run_id=irid,
                    row_index=int(i),
                )

            # Failures must not report scalar numeric fields in the CSV (store partials in JSON instead).
            if (
                np.isfinite(coef.iloc[i])
                or np.isfinite(se.iloc[i])
                or np.isfinite(pval.iloc[i])
                or np.isfinite(ci_lo.iloc[i])
                or np.isfinite(ci_hi.iloc[i])
                or np.isfinite(n_obs.iloc[i])
                or np.isfinite(r2.iloc[i])
            ):
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "inference_results",
                    path,
                    "failure_has_numeric_fields",
                    "run_success=0 but scalar numeric fields are non-missing; set them to NaN and store partial outputs under coefficient_vector_json.partial.",
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

            # Successful rows must use the reserved-key payload schema.
            if not isinstance(payload, dict):
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "inference_results",
                    path,
                    "invalid_success_payload",
                    "run_success=1 but coefficient_vector_json must be a JSON object (dict).",
                    baseline_group_id=gid,
                    spec_run_id=rid,
                    inference_run_id=irid,
                    row_index=int(i),
                )
            else:
                missing_keys = [k for k in REQUIRED_SUCCESS_PAYLOAD_KEYS if k not in payload]
                if missing_keys:
                    _issue(
                        issues,
                        "ERROR",
                        paper_id,
                        "inference_results",
                        path,
                        "missing_required_payload_keys",
                        f"run_success=1 but coefficient_vector_json is missing required keys: {missing_keys}",
                        baseline_group_id=gid,
                        spec_run_id=rid,
                        inference_run_id=irid,
                        row_index=int(i),
                    )

                unknown = sorted(set(payload.keys()) - set(ALLOWED_PAYLOAD_TOP_LEVEL_KEYS))
                if unknown:
                    _issue(
                        issues,
                        "ERROR",
                        paper_id,
                        "inference_results",
                        path,
                        "unknown_payload_keys",
                        f"coefficient_vector_json contains unknown top-level keys: {unknown}. Move paper-specific fields under coefficient_vector_json.extra (dict) to keep the schema stable.",
                        baseline_group_id=gid,
                        spec_run_id=rid,
                        inference_run_id=irid,
                        row_index=int(i),
                    )

                coefs = payload.get("coefficients")
                if not isinstance(coefs, dict):
                    _issue(
                        issues,
                        "ERROR",
                        paper_id,
                        "inference_results",
                        path,
                        "invalid_coefficients_block",
                        "coefficient_vector_json.coefficients must be a JSON object (dict).",
                        baseline_group_id=gid,
                        spec_run_id=rid,
                        inference_run_id=irid,
                        row_index=int(i),
                    )
                else:
                    bad_type_keys: list[str] = []
                    nonfinite_keys: list[str] = []
                    for k, v in coefs.items():
                        if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
                            bad_type_keys.append(str(k))
                            continue
                        try:
                            fv = float(v)
                        except Exception:
                            bad_type_keys.append(str(k))
                            continue
                        if not np.isfinite(fv):
                            nonfinite_keys.append(str(k))
                    if bad_type_keys:
                        _issue(
                            issues,
                            "ERROR",
                            paper_id,
                            "inference_results",
                            path,
                            "invalid_coefficient_values",
                            f"coefficient_vector_json.coefficients must map parameter -> numeric scalar; found non-numeric values (examples: {bad_type_keys[:5]}).",
                            baseline_group_id=gid,
                            spec_run_id=rid,
                            inference_run_id=irid,
                            row_index=int(i),
                        )
                    if nonfinite_keys:
                        _issue(
                            issues,
                            "WARN",
                            paper_id,
                            "inference_results",
                            path,
                            "nonfinite_coefficient_values",
                            f"coefficient_vector_json.coefficients contains non-finite values (examples: {nonfinite_keys[:5]}).",
                            baseline_group_id=gid,
                            spec_run_id=rid,
                            inference_run_id=irid,
                            row_index=int(i),
                        )

                inf = payload.get("inference")
                if not isinstance(inf, dict) or not str(inf.get("spec_id", "")).strip():
                    _issue(
                        issues,
                        "ERROR",
                        paper_id,
                        "inference_results",
                        path,
                        "missing_inference_block",
                        "run_success=1 but coefficient_vector_json.inference.spec_id is missing/empty.",
                        baseline_group_id=gid,
                        spec_run_id=rid,
                        inference_run_id=irid,
                        row_index=int(i),
                    )
                else:
                    inf_sid = str(inf.get("spec_id", "")).strip()
                    if not inf_sid.startswith("infer/"):
                        _issue(
                            issues,
                            "ERROR",
                            paper_id,
                            "inference_results",
                            path,
                            "invalid_inference_spec_id",
                            f"coefficient_vector_json.inference.spec_id must start with 'infer/' (got '{inf_sid}').",
                            baseline_group_id=gid,
                            spec_run_id=rid,
                            inference_run_id=irid,
                            row_index=int(i),
                        )
                    if inf_sid != sid:
                        _issue(
                            issues,
                            "ERROR",
                            paper_id,
                            "inference_results",
                            path,
                            "inference_block_spec_id_mismatch",
                            f"In inference_results.csv, coefficient_vector_json.inference.spec_id must equal the row spec_id. Expected '{sid}' but got '{inf_sid}'.",
                            baseline_group_id=gid,
                            spec_run_id=rid,
                            inference_run_id=irid,
                            row_index=int(i),
                        )

                sw = payload.get("software")
                if not isinstance(sw, dict):
                    _issue(
                        issues,
                        "ERROR",
                        paper_id,
                        "inference_results",
                        path,
                        "missing_software_block",
                        "run_success=1 but coefficient_vector_json.software is missing/invalid.",
                        baseline_group_id=gid,
                        spec_run_id=rid,
                        inference_run_id=irid,
                        row_index=int(i),
                    )
                else:
                    if not str(sw.get("runner_language", "")).strip() or not str(sw.get("runner_version", "")).strip():
                        _issue(
                            issues,
                            "ERROR",
                            paper_id,
                            "inference_results",
                            path,
                            "invalid_software_block",
                            "software block must include non-empty runner_language and runner_version.",
                            baseline_group_id=gid,
                            spec_run_id=rid,
                            inference_run_id=irid,
                            row_index=int(i),
                        )
                    pkgs = sw.get("packages")
                    if not isinstance(pkgs, dict) or len(pkgs) == 0:
                        _issue(
                            issues,
                            "ERROR",
                            paper_id,
                            "inference_results",
                            path,
                            "missing_software_packages",
                            "software.packages must be a non-empty object mapping package -> version.",
                            baseline_group_id=gid,
                            spec_run_id=rid,
                            inference_run_id=irid,
                            row_index=int(i),
                            )

                sh = payload.get("surface_hash")
                if not str(sh or "").strip():
                    _issue(
                        issues,
                        "ERROR",
                        paper_id,
                        "inference_results",
                        path,
                        "missing_surface_hash",
                        "run_success=1 but coefficient_vector_json.surface_hash is missing/empty.",
                        baseline_group_id=gid,
                        spec_run_id=rid,
                        inference_run_id=irid,
                        row_index=int(i),
                    )
                else:
                    sh_s = str(sh).strip()
                    ok = sh_s.startswith("sha256:") and (len(sh_s) == 7 + 64) and all(c in "0123456789abcdef" for c in sh_s[7:].lower())
                    if not ok:
                        _issue(
                            issues,
                            "ERROR",
                            paper_id,
                            "inference_results",
                            path,
                            "invalid_surface_hash_format",
                            f"surface_hash must look like 'sha256:<64 hex>'; got '{sh_s}'.",
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


def _validate_exploration_results(
    paper_id: str,
    pkg_dir: Path,
    issues: list[Issue],
    spec_results: pd.DataFrame,
) -> None:
    path = pkg_dir / "exploration_results.csv"
    if not path.exists():
        return
    df = _read_csv(path, issues, paper_id, "exploration_results")
    if df.empty:
        return

    if not _check_required_cols(df, REQUIRED_EXPLORATION_COLS, issues, paper_id, "exploration_results", path):
        return

    # Paper id consistency
    if "paper_id" in df.columns:
        bad = df["paper_id"].astype(str).ne(paper_id)
        if bad.any():
            _issue(
                issues,
                "ERROR",
                paper_id,
                "exploration_results",
                path,
                "paper_id_column_mismatch",
                f"paper_id column contains {bad.sum()} rows not equal to {paper_id}.",
            )

    # exploration_run_id uniqueness
    df["exploration_run_id"] = df["exploration_run_id"].astype(str)
    if df["exploration_run_id"].str.strip().eq("").any():
        _issue(
            issues,
            "ERROR",
            paper_id,
            "exploration_results",
            path,
            "empty_exploration_run_id",
            "Some exploration_run_id values are empty.",
        )
    dup = df.duplicated(subset=["exploration_run_id"], keep=False)
    if dup.any():
        ex = df.loc[dup, "exploration_run_id"].astype(str).head(5).tolist()
        _issue(
            issues,
            "ERROR",
            paper_id,
            "exploration_results",
            path,
            "duplicate_exploration_run_id",
            f"Duplicate exploration_run_id values found (examples: {ex}).",
        )

    # Typed spec ids
    spec_id = df["spec_id"].astype(str).str.strip()
    bad = ~spec_id.str.startswith("explore/")
    if bad.any():
        bad_ids = spec_id[bad].value_counts().head(8).to_dict()
        _issue(
            issues,
            "ERROR",
            paper_id,
            "exploration_results",
            path,
            "invalid_spec_id_namespace",
            f"exploration_results.csv contains spec_id not starting with explore/ (examples: {bad_ids}).",
        )

    # Optional linkage to base spec_run_id
    base_ids = set(spec_results.get("spec_run_id", pd.Series(dtype=str)).astype(str).tolist()) if not spec_results.empty else set()
    link = df["spec_run_id"].astype(str).str.strip()
    missing = link.ne("") & ~link.isin(base_ids)
    if missing.any():
        ex = link[missing].head(5).tolist()
        _issue(
            issues,
            "WARN",
            paper_id,
            "exploration_results",
            path,
            "unknown_spec_run_id_link",
            f"exploration_results references spec_run_id not found in specification_results.csv (examples: {ex}).",
        )

    # Per-row JSON + run_success rules
    rs = pd.to_numeric(df["run_success"], errors="coerce").fillna(-1).astype(int)
    run_error = df["run_error"].astype(str)
    for i in range(len(df)):
        sid = str(df.at[i, "spec_id"]).strip()
        rid = str(df.at[i, "exploration_run_id"]).strip()
        stp = str(df.at[i, "spec_tree_path"]).strip()
        gid = str(df.at[i, "baseline_group_id"]).strip()

        _validate_spec_tree_path(
            paper_id,
            "exploration_results",
            path,
            stp,
            sid,
            issues,
            baseline_group_id=gid,
            spec_run_id=rid,
            row_index=int(i),
        )

        if rs.iloc[i] not in (0, 1):
            _issue(
                issues,
                "ERROR",
                paper_id,
                "exploration_results",
                path,
                "invalid_run_success",
                f"run_success must be 0/1 (got {df.at[i,'run_success']}).",
                baseline_group_id=gid,
                spec_run_id=rid,
                row_index=int(i),
            )
            continue

        raw = df.at[i, "exploration_json"]
        try:
            payload = json.loads(raw) if isinstance(raw, str) and raw.strip() else None
        except Exception as e:
            _issue(
                issues,
                "ERROR",
                paper_id,
                "exploration_results",
                path,
                "invalid_exploration_json",
                f"Failed to parse exploration_json as JSON: {e}",
                baseline_group_id=gid,
                spec_run_id=rid,
                row_index=int(i),
            )
            continue
        if not isinstance(payload, dict):
            _issue(
                issues,
                "ERROR",
                paper_id,
                "exploration_results",
                path,
                "invalid_exploration_json",
                "exploration_json must be a JSON object (dict).",
                baseline_group_id=gid,
                spec_run_id=rid,
                row_index=int(i),
            )
            continue

        if rs.iloc[i] == 1:
            if not isinstance(payload.get("software"), dict) or not payload.get("software"):
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "exploration_results",
                    path,
                    "missing_software_block",
                    "run_success=1 but exploration_json.software is missing/empty.",
                    baseline_group_id=gid,
                    spec_run_id=rid,
                    row_index=int(i),
                )
            if not str(payload.get("surface_hash", "")).strip():
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "exploration_results",
                    path,
                    "missing_surface_hash",
                    "run_success=1 but exploration_json.surface_hash is missing/empty.",
                    baseline_group_id=gid,
                    spec_run_id=rid,
                    row_index=int(i),
                )
            exp = payload.get("exploration")
            if not isinstance(exp, dict) or not str(exp.get("spec_id", "")).strip():
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "exploration_results",
                    path,
                    "missing_exploration_block",
                    "run_success=1 but exploration_json.exploration.spec_id is missing/empty.",
                    baseline_group_id=gid,
                    spec_run_id=rid,
                    row_index=int(i),
                )
            else:
                esid = str(exp.get("spec_id", "")).strip()
                if esid != sid:
                    _issue(
                        issues,
                        "ERROR",
                        paper_id,
                        "exploration_results",
                        path,
                        "exploration_block_spec_id_mismatch",
                        f"exploration_json.exploration.spec_id must equal the row spec_id. Expected '{sid}' but got '{esid}'.",
                        baseline_group_id=gid,
                        spec_run_id=rid,
                        row_index=int(i),
                    )
        else:
            if not str(run_error.iloc[i]).strip():
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "exploration_results",
                    path,
                    "missing_run_error",
                    "run_success=0 but run_error is empty.",
                    baseline_group_id=gid,
                    spec_run_id=rid,
                    row_index=int(i),
                )
            if not str(payload.get("error", "")).strip():
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "exploration_results",
                    path,
                    "missing_error_field",
                    "run_success=0 but exploration_json.error is missing/empty.",
                    baseline_group_id=gid,
                    spec_run_id=rid,
                    row_index=int(i),
                )
            ed = payload.get("error_details")
            if not isinstance(ed, dict) or not ed:
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "exploration_results",
                    path,
                    "missing_error_details",
                    "run_success=0 but exploration_json.error_details is missing/empty.",
                    baseline_group_id=gid,
                    spec_run_id=rid,
                    row_index=int(i),
                )
            else:
                missing_ed = [k for k in REQUIRED_ERROR_DETAILS_FIELDS if not str(ed.get(k, "")).strip()]
                if missing_ed:
                    _issue(
                        issues,
                        "ERROR",
                        paper_id,
                        "exploration_results",
                        path,
                        "missing_error_details_fields",
                        f"exploration_json.error_details is missing required fields: {missing_ed}",
                        baseline_group_id=gid,
                        spec_run_id=rid,
                        row_index=int(i),
                    )


def _validate_sensitivity_results(
    paper_id: str,
    pkg_dir: Path,
    issues: list[Issue],
    spec_results: pd.DataFrame,
) -> None:
    path = pkg_dir / "sensitivity_results.csv"
    if not path.exists():
        return
    df = _read_csv(path, issues, paper_id, "sensitivity_results")
    if df.empty:
        return

    if not _check_required_cols(df, REQUIRED_SENSITIVITY_COLS, issues, paper_id, "sensitivity_results", path):
        return

    # Paper id consistency
    if "paper_id" in df.columns:
        bad = df["paper_id"].astype(str).ne(paper_id)
        if bad.any():
            _issue(
                issues,
                "ERROR",
                paper_id,
                "sensitivity_results",
                path,
                "paper_id_column_mismatch",
                f"paper_id column contains {bad.sum()} rows not equal to {paper_id}.",
            )

    # sensitivity_run_id uniqueness
    df["sensitivity_run_id"] = df["sensitivity_run_id"].astype(str)
    if df["sensitivity_run_id"].str.strip().eq("").any():
        _issue(
            issues,
            "ERROR",
            paper_id,
            "sensitivity_results",
            path,
            "empty_sensitivity_run_id",
            "Some sensitivity_run_id values are empty.",
        )
    dup = df.duplicated(subset=["sensitivity_run_id"], keep=False)
    if dup.any():
        ex = df.loc[dup, "sensitivity_run_id"].astype(str).head(5).tolist()
        _issue(
            issues,
            "ERROR",
            paper_id,
            "sensitivity_results",
            path,
            "duplicate_sensitivity_run_id",
            f"Duplicate sensitivity_run_id values found (examples: {ex}).",
        )

    # Typed spec ids
    spec_id = df["spec_id"].astype(str).str.strip()
    bad = ~spec_id.str.startswith("sens/")
    if bad.any():
        bad_ids = spec_id[bad].value_counts().head(8).to_dict()
        _issue(
            issues,
            "ERROR",
            paper_id,
            "sensitivity_results",
            path,
            "invalid_spec_id_namespace",
            f"sensitivity_results.csv contains spec_id not starting with sens/ (examples: {bad_ids}).",
        )

    scope = df["sensitivity_scope"].astype(str).str.strip()
    allowed_scope = {"paper", "baseline_group", "spec"}
    bad_scope = ~scope.isin(allowed_scope)
    if bad_scope.any():
        ex = scope[bad_scope].value_counts().head(8).to_dict()
        _issue(
            issues,
            "ERROR",
            paper_id,
            "sensitivity_results",
            path,
            "invalid_sensitivity_scope",
            f"sensitivity_scope must be one of {sorted(allowed_scope)} (examples: {ex}).",
        )

    # Optional linkage to base spec_run_id
    base_ids = set(spec_results.get("spec_run_id", pd.Series(dtype=str)).astype(str).tolist()) if not spec_results.empty else set()
    link = df["spec_run_id"].astype(str).str.strip()
    missing = link.ne("") & ~link.isin(base_ids)
    if missing.any():
        ex = link[missing].head(5).tolist()
        _issue(
            issues,
            "WARN",
            paper_id,
            "sensitivity_results",
            path,
            "unknown_spec_run_id_link",
            f"sensitivity_results references spec_run_id not found in specification_results.csv (examples: {ex}).",
        )

    rs = pd.to_numeric(df["run_success"], errors="coerce").fillna(-1).astype(int)
    run_error = df["run_error"].astype(str)
    for i in range(len(df)):
        sid = str(df.at[i, "spec_id"]).strip()
        rid = str(df.at[i, "sensitivity_run_id"]).strip()
        stp = str(df.at[i, "spec_tree_path"]).strip()
        gid = str(df.at[i, "baseline_group_id"]).strip()

        _validate_spec_tree_path(
            paper_id,
            "sensitivity_results",
            path,
            stp,
            sid,
            issues,
            baseline_group_id=gid,
            spec_run_id=rid,
            row_index=int(i),
        )

        if rs.iloc[i] not in (0, 1):
            _issue(
                issues,
                "ERROR",
                paper_id,
                "sensitivity_results",
                path,
                "invalid_run_success",
                f"run_success must be 0/1 (got {df.at[i,'run_success']}).",
                baseline_group_id=gid,
                spec_run_id=rid,
                row_index=int(i),
            )
            continue

        raw = df.at[i, "sensitivity_json"]
        try:
            payload = json.loads(raw) if isinstance(raw, str) and raw.strip() else None
        except Exception as e:
            _issue(
                issues,
                "ERROR",
                paper_id,
                "sensitivity_results",
                path,
                "invalid_sensitivity_json",
                f"Failed to parse sensitivity_json as JSON: {e}",
                baseline_group_id=gid,
                spec_run_id=rid,
                row_index=int(i),
            )
            continue
        if not isinstance(payload, dict):
            _issue(
                issues,
                "ERROR",
                paper_id,
                "sensitivity_results",
                path,
                "invalid_sensitivity_json",
                "sensitivity_json must be a JSON object (dict).",
                baseline_group_id=gid,
                spec_run_id=rid,
                row_index=int(i),
            )
            continue

        if rs.iloc[i] == 1:
            if not isinstance(payload.get("software"), dict) or not payload.get("software"):
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "sensitivity_results",
                    path,
                    "missing_software_block",
                    "run_success=1 but sensitivity_json.software is missing/empty.",
                    baseline_group_id=gid,
                    spec_run_id=rid,
                    row_index=int(i),
                )
            if not str(payload.get("surface_hash", "")).strip():
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "sensitivity_results",
                    path,
                    "missing_surface_hash",
                    "run_success=1 but sensitivity_json.surface_hash is missing/empty.",
                    baseline_group_id=gid,
                    spec_run_id=rid,
                    row_index=int(i),
                )
            sens = payload.get("sensitivity")
            if not isinstance(sens, dict) or not str(sens.get("spec_id", "")).strip():
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "sensitivity_results",
                    path,
                    "missing_sensitivity_block",
                    "run_success=1 but sensitivity_json.sensitivity.spec_id is missing/empty.",
                    baseline_group_id=gid,
                    spec_run_id=rid,
                    row_index=int(i),
                )
            else:
                ssid = str(sens.get("spec_id", "")).strip()
                if ssid != sid:
                    _issue(
                        issues,
                        "ERROR",
                        paper_id,
                        "sensitivity_results",
                        path,
                        "sensitivity_block_spec_id_mismatch",
                        f"sensitivity_json.sensitivity.spec_id must equal the row spec_id. Expected '{sid}' but got '{ssid}'.",
                        baseline_group_id=gid,
                        spec_run_id=rid,
                        row_index=int(i),
                    )
        else:
            if not str(run_error.iloc[i]).strip():
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "sensitivity_results",
                    path,
                    "missing_run_error",
                    "run_success=0 but run_error is empty.",
                    baseline_group_id=gid,
                    spec_run_id=rid,
                    row_index=int(i),
                )
            if not str(payload.get("error", "")).strip():
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "sensitivity_results",
                    path,
                    "missing_error_field",
                    "run_success=0 but sensitivity_json.error is missing/empty.",
                    baseline_group_id=gid,
                    spec_run_id=rid,
                    row_index=int(i),
                )
            ed = payload.get("error_details")
            if not isinstance(ed, dict) or not ed:
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "sensitivity_results",
                    path,
                    "missing_error_details",
                    "run_success=0 but sensitivity_json.error_details is missing/empty.",
                    baseline_group_id=gid,
                    spec_run_id=rid,
                    row_index=int(i),
                )
            else:
                missing_ed = [k for k in REQUIRED_ERROR_DETAILS_FIELDS if not str(ed.get(k, "")).strip()]
                if missing_ed:
                    _issue(
                        issues,
                        "ERROR",
                        paper_id,
                        "sensitivity_results",
                        path,
                        "missing_error_details_fields",
                        f"sensitivity_json.error_details is missing required fields: {missing_ed}",
                        baseline_group_id=gid,
                        spec_run_id=rid,
                        row_index=int(i),
                    )


def _validate_postprocess_results(
    paper_id: str,
    pkg_dir: Path,
    issues: list[Issue],
) -> None:
    path = pkg_dir / "postprocess_results.csv"
    if not path.exists():
        return
    df = _read_csv(path, issues, paper_id, "postprocess_results")
    if df.empty:
        return

    if not _check_required_cols(df, REQUIRED_POSTPROCESS_COLS, issues, paper_id, "postprocess_results", path):
        return

    # Paper id consistency
    if "paper_id" in df.columns:
        bad = df["paper_id"].astype(str).ne(paper_id)
        if bad.any():
            _issue(
                issues,
                "ERROR",
                paper_id,
                "postprocess_results",
                path,
                "paper_id_column_mismatch",
                f"paper_id column contains {bad.sum()} rows not equal to {paper_id}.",
            )

    # postprocess_run_id uniqueness
    df["postprocess_run_id"] = df["postprocess_run_id"].astype(str)
    if df["postprocess_run_id"].str.strip().eq("").any():
        _issue(
            issues,
            "ERROR",
            paper_id,
            "postprocess_results",
            path,
            "empty_postprocess_run_id",
            "Some postprocess_run_id values are empty.",
        )
    dup = df.duplicated(subset=["postprocess_run_id"], keep=False)
    if dup.any():
        ex = df.loc[dup, "postprocess_run_id"].astype(str).head(5).tolist()
        _issue(
            issues,
            "ERROR",
            paper_id,
            "postprocess_results",
            path,
            "duplicate_postprocess_run_id",
            f"Duplicate postprocess_run_id values found (examples: {ex}).",
        )

    # Typed spec ids
    spec_id = df["spec_id"].astype(str).str.strip()
    bad = ~spec_id.str.startswith("post/")
    if bad.any():
        bad_ids = spec_id[bad].value_counts().head(8).to_dict()
        _issue(
            issues,
            "ERROR",
            paper_id,
            "postprocess_results",
            path,
            "invalid_spec_id_namespace",
            f"postprocess_results.csv contains spec_id not starting with post/ (examples: {bad_ids}).",
        )

    rs = pd.to_numeric(df["run_success"], errors="coerce").fillna(-1).astype(int)
    run_error = df["run_error"].astype(str)
    for i in range(len(df)):
        sid = str(df.at[i, "spec_id"]).strip()
        rid = str(df.at[i, "postprocess_run_id"]).strip()
        stp = str(df.at[i, "spec_tree_path"]).strip()
        gid = str(df.at[i, "baseline_group_id"]).strip()

        _validate_spec_tree_path(
            paper_id,
            "postprocess_results",
            path,
            stp,
            sid,
            issues,
            baseline_group_id=gid,
            spec_run_id=rid,
            row_index=int(i),
        )

        if rs.iloc[i] not in (0, 1):
            _issue(
                issues,
                "ERROR",
                paper_id,
                "postprocess_results",
                path,
                "invalid_run_success",
                f"run_success must be 0/1 (got {df.at[i,'run_success']}).",
                baseline_group_id=gid,
                spec_run_id=rid,
                row_index=int(i),
            )
            continue

        raw = df.at[i, "postprocess_json"]
        try:
            payload = json.loads(raw) if isinstance(raw, str) and raw.strip() else None
        except Exception as e:
            _issue(
                issues,
                "ERROR",
                paper_id,
                "postprocess_results",
                path,
                "invalid_postprocess_json",
                f"Failed to parse postprocess_json as JSON: {e}",
                baseline_group_id=gid,
                spec_run_id=rid,
                row_index=int(i),
            )
            continue
        if not isinstance(payload, dict):
            _issue(
                issues,
                "ERROR",
                paper_id,
                "postprocess_results",
                path,
                "invalid_postprocess_json",
                "postprocess_json must be a JSON object (dict).",
                baseline_group_id=gid,
                spec_run_id=rid,
                row_index=int(i),
            )
            continue

        if rs.iloc[i] == 1:
            if not isinstance(payload.get("software"), dict) or not payload.get("software"):
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "postprocess_results",
                    path,
                    "missing_software_block",
                    "run_success=1 but postprocess_json.software is missing/empty.",
                    baseline_group_id=gid,
                    spec_run_id=rid,
                    row_index=int(i),
                )
            if not str(payload.get("surface_hash", "")).strip():
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "postprocess_results",
                    path,
                    "missing_surface_hash",
                    "run_success=1 but postprocess_json.surface_hash is missing/empty.",
                    baseline_group_id=gid,
                    spec_run_id=rid,
                    row_index=int(i),
                )
            pp = payload.get("postprocess")
            if not isinstance(pp, dict) or not str(pp.get("spec_id", "")).strip():
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "postprocess_results",
                    path,
                    "missing_postprocess_block",
                    "run_success=1 but postprocess_json.postprocess.spec_id is missing/empty.",
                    baseline_group_id=gid,
                    spec_run_id=rid,
                    row_index=int(i),
                )
            else:
                psid = str(pp.get("spec_id", "")).strip()
                if psid != sid:
                    _issue(
                        issues,
                        "ERROR",
                        paper_id,
                        "postprocess_results",
                        path,
                        "postprocess_block_spec_id_mismatch",
                        f"postprocess_json.postprocess.spec_id must equal the row spec_id. Expected '{sid}' but got '{psid}'.",
                        baseline_group_id=gid,
                        spec_run_id=rid,
                        row_index=int(i),
                    )
        else:
            if not str(run_error.iloc[i]).strip():
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "postprocess_results",
                    path,
                    "missing_run_error",
                    "run_success=0 but run_error is empty.",
                    baseline_group_id=gid,
                    spec_run_id=rid,
                    row_index=int(i),
                )
            if not str(payload.get("error", "")).strip():
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "postprocess_results",
                    path,
                    "missing_error_field",
                    "run_success=0 but postprocess_json.error is missing/empty.",
                    baseline_group_id=gid,
                    spec_run_id=rid,
                    row_index=int(i),
                )
            ed = payload.get("error_details")
            if not isinstance(ed, dict) or not ed:
                _issue(
                    issues,
                    "ERROR",
                    paper_id,
                    "postprocess_results",
                    path,
                    "missing_error_details",
                    "run_success=0 but postprocess_json.error_details is missing/empty.",
                    baseline_group_id=gid,
                    spec_run_id=rid,
                    row_index=int(i),
                )
            else:
                missing_ed = [k for k in REQUIRED_ERROR_DETAILS_FIELDS if not str(ed.get(k, "")).strip()]
                if missing_ed:
                    _issue(
                        issues,
                        "ERROR",
                        paper_id,
                        "postprocess_results",
                        path,
                        "missing_error_details_fields",
                        f"postprocess_json.error_details is missing required fields: {missing_ed}",
                        baseline_group_id=gid,
                        spec_run_id=rid,
                        row_index=int(i),
                    )


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

    # spec_tree_path should point to a spec-tree markdown node (or be "custom" only when unavoidable).
    if "spec_tree_path" in df.columns:
        for i, row in df.iterrows():
            _validate_spec_tree_path(
                issues=issues,
                paper_id=paper_id,
                area="verification",
                path=map_path,
                spec_tree_path=row.get("spec_tree_path", ""),
                spec_id=str(row.get("spec_id", "")).strip(),
                baseline_group_id=str(row.get("baseline_group_id", "")).strip(),
                spec_run_id=str(row.get("spec_run_id", "")).strip(),
                row_index=int(i),
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

    core_patterns, infer_patterns, canon_infer_by_gid = _validate_surface(paper_id, pkg_dir, issues)
    spec_df = _validate_spec_results(paper_id, pkg_dir, issues, core_patterns, canon_infer_by_gid)
    _validate_inference_results(paper_id, pkg_dir, issues, spec_df, infer_patterns)
    _validate_exploration_results(paper_id, pkg_dir, issues, spec_df)
    _validate_sensitivity_results(paper_id, pkg_dir, issues, spec_df)
    _validate_postprocess_results(paper_id, pkg_dir, issues)
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
