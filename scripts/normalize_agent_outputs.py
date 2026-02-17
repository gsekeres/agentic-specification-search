#!/usr/bin/env python3
"""
normalize_agent_outputs.py
=========================

Normalize per-paper agent outputs to match the current spec-tree contract.

This script edits extracted-package CSVs in-place:
  - specification_results.csv
  - inference_results.csv (if present)

Key normalizations:
  - coefficient_vector_json becomes a JSON object with reserved audit keys:
      coefficients, inference, software, surface_hash (+ axis blocks, error, etc.)
  - legacy flat coefficient dicts are wrapped into {"coefficients": {...}}.
  - missing required audit blocks are added when reasonably inferable.

Usage:
  python scripts/normalize_agent_outputs.py --paper-id 112431-V1
  python scripts/normalize_agent_outputs.py --all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTED_DIR = ROOT / "data" / "downloads" / "extracted"

NUMERIC_SCALAR_COLS: tuple[str, ...] = (
    "coefficient",
    "std_error",
    "p_value",
    "ci_lower",
    "ci_upper",
    "n_obs",
    "r_squared",
)

def _ensure_run_success_and_error(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure `run_success` and `run_error` columns exist and are usable.

    - If `run_success` is missing, infer it from finite (coef, se) with se>0.
    - If `run_error` is missing, create it.
    - For `run_success==0` rows with empty `run_error`, populate a short message.
    """
    out = df.copy()

    coef = pd.to_numeric(out.get("coefficient"), errors="coerce")
    se = pd.to_numeric(out.get("std_error"), errors="coerce")
    has_valid_se = np.isfinite(se) & (se > 0)
    has_valid_coef = np.isfinite(coef)
    inferred_success = (has_valid_coef & has_valid_se).astype(int)

    if "run_success" not in out.columns:
        out["run_success"] = inferred_success
    else:
        rs = pd.to_numeric(out["run_success"], errors="coerce")
        out["run_success"] = rs.fillna(inferred_success).astype(int)

    if "run_error" not in out.columns:
        out["run_error"] = ""
    out["run_error"] = out["run_error"].fillna("").astype(str)

    needs_err = (out["run_success"] == 0) & (out["run_error"].str.strip() == "")
    if not needs_err.any():
        return out

    # Prefer a notes column if present (common legacy failure format).
    if "notes" in out.columns:
        notes = out["notes"].fillna("").astype(str)
        use_notes = needs_err & (notes.str.strip() != "")
        out.loc[use_notes, "run_error"] = notes[use_notes]
        needs_err = (out["run_success"] == 0) & (out["run_error"].str.strip() == "")
        if not needs_err.any():
            return out

    # Try to extract `error` from coefficient_vector_json for failure rows.
    if "coefficient_vector_json" in out.columns:
        for idx in out.index[needs_err]:
            payload = _load_json_obj(str(out.at[idx, "coefficient_vector_json"]))
            msg = str(payload.get("error", "") or "").strip()
            if msg:
                out.at[idx, "run_error"] = msg
        needs_err = (out["run_success"] == 0) & (out["run_error"].str.strip() == "")
        if not needs_err.any():
            return out

    # Last resort: synthesize a concrete missing/invalid message.
    def _missing_parts(i: int) -> str:
        parts: list[str] = []
        if not bool(has_valid_coef.iloc[i]):
            parts.append("coef")
        if not bool(has_valid_se.iloc[i]):
            parts.append("se")
        return "+".join(parts)

    msgs: list[str] = []
    for i in range(len(out)):
        if not bool(needs_err.iloc[i]):
            msgs.append("")
            continue
        mp = _missing_parts(i)
        msgs.append(f"missing/invalid:{mp}" if mp else "run_success=0")

    out.loc[needs_err, "run_error"] = pd.Series(msgs, index=out.index)[needs_err]
    return out

RESERVED_PAYLOAD_KEYS: set[str] = {
    # Required contract keys
    "coefficients",
    "inference",
    "software",
    "surface_hash",
    # Failure plumbing
    "error",
    "error_details",
    "partial",
    # Flexible extension points (keep top-level schema stable)
    "design",
    "extra",
    "universe",
    "sampling",
    # Common audit blocks
    "controls",
    "sample",
    "fixed_effects",
    "preprocess",
    "estimation",
    "weights",
    "data_construction",
    "functional_form",
    "joint",
    # Other common metadata blocks
    "focal",
    "bundle",
    "warnings",
    "notes",
}

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
        # Flexible extension points
        "design",
        "extra",
        "universe",
        "sampling",
    }
)

AXIS_BLOCK_KEYS: tuple[str, ...] = (
    "controls",
    "sample",
    "fixed_effects",
    "preprocess",
    "estimation",
    "weights",
    "data_construction",
    "functional_form",
    "joint",
)

PACKAGE_VERSION_CANDIDATES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "statsmodels",
    "linearmodels",
    "pyfixest",
    "scipy",
    "rdrobust",
    "openpyxl",
    "pyreadstat",
)


def _safe_int(x, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x):
    if x is None:
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def _load_json_obj(s: str) -> dict:
    s = (s or "").strip()
    if not s:
        return {}
    try:
        obj = json.loads(s)
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _surface_hash(surface: dict) -> str:
    canon = json.dumps(surface, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    h = hashlib.sha256(canon.encode("utf-8")).hexdigest()
    return f"sha256:{h}"


def _software_block() -> dict:
    pkgs: dict[str, str] = {}
    for name in PACKAGE_VERSION_CANDIDATES:
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
        except Exception:
            continue
    return {
        "runner_language": "python",
        "runner_version": ".".join(map(str, sys.version_info[:3])),
        "packages": pkgs,
    }


def _inference_plan_maps(surface: dict) -> tuple[dict[str, dict], dict[str, dict[str, dict]]]:
    """
    Returns:
      canonical_by_gid: baseline_group_id -> {"spec_id": ..., "params": {...}}
      all_by_gid: baseline_group_id -> (spec_id -> {"spec_id": ..., "params": {...}})
    """
    canonical_by_gid: dict[str, dict] = {}
    all_by_gid: dict[str, dict[str, dict]] = {}

    for g in surface.get("baseline_groups", []) or []:
        if not isinstance(g, dict):
            continue
        gid = str(g.get("baseline_group_id", "")).strip()
        if not gid:
            continue
        ip = g.get("inference_plan", {}) or {}
        canonical = ip.get("canonical", {}) if isinstance(ip, dict) else {}
        canon_id = str(canonical.get("spec_id", "")).strip() if isinstance(canonical, dict) else ""
        canon_params = canonical.get("params", {}) if isinstance(canonical, dict) else {}
        canon_params = canon_params if isinstance(canon_params, dict) else {}
        if canon_id:
            canonical_by_gid[gid] = {"spec_id": canon_id, "params": canon_params}

        by_spec: dict[str, dict] = {}
        if canon_id:
            by_spec[canon_id] = {"spec_id": canon_id, "params": canon_params}
        variants = ip.get("variants", []) if isinstance(ip, dict) else []
        if isinstance(variants, list):
            for v in variants:
                if not isinstance(v, dict):
                    continue
                vid = str(v.get("spec_id", "")).strip()
                params = v.get("params", {})
                params = params if isinstance(params, dict) else {}
                if vid:
                    by_spec[vid] = {"spec_id": vid, "params": params}
        all_by_gid[gid] = by_spec

    return canonical_by_gid, all_by_gid


def _baseline_rows(df_spec: pd.DataFrame) -> dict[str, dict]:
    """
    baseline_group_id -> parsed payload for spec_id=='baseline' (run_success==1 preferred).
    """
    out: dict[str, dict] = {}
    if df_spec.empty:
        return out

    sub = df_spec.copy()
    sub["baseline_group_id"] = sub["baseline_group_id"].astype(str).str.strip()
    sub["spec_id"] = sub["spec_id"].astype(str).str.strip()
    if "run_success" in sub.columns:
        sub["_rs"] = pd.to_numeric(sub["run_success"], errors="coerce").fillna(0).astype(int)
    else:
        sub["_rs"] = 0

    base = sub[sub["spec_id"] == "baseline"].copy()
    if base.empty:
        return out

    base = base.sort_values(["baseline_group_id", "_rs"], ascending=[True, False])
    for gid, gdf in base.groupby("baseline_group_id", dropna=False):
        row = gdf.iloc[0]
        payload = _load_json_obj(str(row.get("coefficient_vector_json", "")))
        out[str(gid)] = payload
    return out


def _coef_keys_from_payload(payload: dict) -> list[str]:
    """
    Extract coefficient names from either:
      - payload["coefficients"] (preferred), or
      - legacy flat dict (numeric top-level entries).
    """
    if isinstance(payload.get("coefficients"), dict):
        return list(payload.get("coefficients", {}).keys())
    keys: list[str] = []
    for k, v in payload.items():
        if k in RESERVED_PAYLOAD_KEYS:
            continue
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            keys.append(str(k))
    return keys


def _split_legacy_payload(payload: dict) -> tuple[dict, dict]:
    """
    Returns (coefficients, metadata).

    If payload already has a 'coefficients' dict, treat remaining keys as metadata.
    Otherwise, treat numeric top-level entries as coefficients and everything else as metadata.
    """
    if isinstance(payload.get("coefficients"), dict):
        coefs = payload.get("coefficients", {}) or {}
        meta = {k: v for k, v in payload.items() if k != "coefficients"}
        return (coefs, meta)

    coefs: dict[str, float] = {}
    meta: dict = {}
    for k, v in payload.items():
        if k in RESERVED_PAYLOAD_KEYS:
            meta[k] = v
            continue
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            coefs[str(k)] = float(v)
        else:
            meta[k] = v
    return (coefs, meta)


def _move_unknown_payload_keys_to_extra(payload: dict) -> None:
    """
    Keep the top-level payload schema stable by moving unknown keys under payload["extra"].
    """
    extra = payload.get("extra")
    if not isinstance(extra, dict):
        extra = {}
    unknown = [k for k in list(payload.keys()) if k not in ALLOWED_PAYLOAD_TOP_LEVEL_KEYS]
    if not unknown:
        if extra:
            payload["extra"] = extra
        return
    for k in unknown:
        extra[k] = payload.pop(k)
    payload["extra"] = extra


def _ensure_error_details(payload: dict, run_error: str) -> None:
    """
    Ensure failures carry a minimal structured error_details object.
    """
    ed = payload.get("error_details")
    if isinstance(ed, dict) and ed:
        # Fill any missing required fields with safe fallbacks.
        ed.setdefault("stage", "unknown")
        ed.setdefault("exception_type", "unknown")
        msg = str(ed.get("exception_message", "")).strip()
        if not msg:
            ed["exception_message"] = str(payload.get("error", "") or run_error or "run_success=0").strip()
        payload["error_details"] = ed
        return
    payload["error_details"] = {
        "stage": "unknown",
        "exception_type": "unknown",
        "exception_message": str(payload.get("error", "") or run_error or "run_success=0").strip(),
    }


def _ensure_controls_block(
    *,
    payload: dict,
    spec_id: str,
    treatment_var: str,
    baseline_controls: set[str] | None,
) -> None:
    if isinstance(payload.get("controls"), dict) and payload.get("controls"):
        return
    parts = spec_id.split("/")
    family = parts[2] if len(parts) > 2 else ""
    variant = "/".join(parts[3:]) if len(parts) > 3 else ""
    coefs = payload.get("coefficients", {}) if isinstance(payload.get("coefficients"), dict) else {}
    included = sorted([k for k in coefs.keys() if k not in {"Intercept", treatment_var}])
    block: dict = {
        "spec_id": spec_id,
        "family": family,
        "variant": variant,
        "included": included,
        "n_controls": int(len(included)),
    }
    if baseline_controls is not None and len(baseline_controls) > 0:
        inc_set = set(included)
        dropped = sorted(list(baseline_controls - inc_set))
        added = sorted(list(inc_set - baseline_controls))
        # Avoid dumping huge dropped-lists for progression/bivariate specs; keep small diffs.
        if len(dropped) <= 25:
            block["dropped"] = dropped
        if len(added) <= 25:
            block["added"] = added
    payload["controls"] = block


def _ensure_sample_block(
    *,
    payload: dict,
    spec_id: str,
    outcome_var: str,
    baseline_n_obs: float | None,
    row_n_obs: float | None,
) -> None:
    if isinstance(payload.get("sample"), dict) and payload.get("sample"):
        return
    parts = spec_id.split("/")
    axis = parts[2] if len(parts) > 2 else ""
    variant = "/".join(parts[3:]) if len(parts) > 3 else ""
    block: dict = {
        "spec_id": spec_id,
        "axis": axis,
        "variant": variant,
    }
    if baseline_n_obs is not None and np.isfinite(baseline_n_obs):
        block["n_obs_before"] = int(baseline_n_obs)
    if row_n_obs is not None and np.isfinite(row_n_obs):
        block["n_obs_after"] = int(row_n_obs)

    # Light parser for common variants
    if axis == "outliers" and variant.startswith("trim_y_"):
        # trim_y_1_99 / trim_y_5_95
        try:
            lo, hi = variant.replace("trim_y_", "").split("_")
            block["rule"] = "trim"
            block["params"] = {"var": outcome_var, "lower_q": int(lo) / 100.0, "upper_q": int(hi) / 100.0}
        except Exception:
            pass
    if axis == "time" and variant.startswith("drop_"):
        try:
            yr = int(variant.replace("drop_", ""))
            block["rule"] = "drop_year"
            block["params"] = {"year": yr}
        except Exception:
            pass
    if axis == "time" and variant.startswith("short_window_"):
        # short_window_1930_1950
        toks = variant.replace("short_window_", "").split("_")
        if len(toks) == 2:
            try:
                block["rule"] = "window"
                block["params"] = {"start": int(toks[0]), "end": int(toks[1])}
            except Exception:
                pass

    payload["sample"] = block


def _ensure_fixed_effects_block(
    *,
    payload: dict,
    spec_id: str,
    baseline_fe: str | None,
    row_fe: str | None,
) -> None:
    if isinstance(payload.get("fixed_effects"), dict) and payload.get("fixed_effects"):
        return
    parts = spec_id.split("/")
    family = parts[2] if len(parts) > 2 else ""
    variant = "/".join(parts[3:]) if len(parts) > 3 else ""
    block: dict = {
        "spec_id": spec_id,
        "family": family,
        "variant": variant,
        "baseline_fe": (baseline_fe or ""),
        "new_fe": (row_fe or ""),
    }
    if family == "drop" and variant:
        block["dropped"] = [variant]
    if family == "add" and variant:
        block["added"] = [variant]
    payload["fixed_effects"] = block


def _ensure_weights_block(
    *,
    payload: dict,
    spec_id: str,
    baseline_weight_var: str | None,
) -> None:
    if isinstance(payload.get("weights"), dict) and payload.get("weights"):
        return
    parts = spec_id.split("/")
    family = parts[2] if len(parts) > 2 else ""
    variant = "/".join(parts[3:]) if len(parts) > 3 else ""
    weight_var = ""
    if family == "main":
        if variant.startswith("unweighted"):
            weight_var = ""
        elif variant == "paper_weights":
            weight_var = baseline_weight_var or ""
        else:
            # Best-effort: treat the variant name as the intended weight variable label.
            weight_var = variant
    block: dict = {
        "spec_id": spec_id,
        "family": family,
        "variant": variant,
        "baseline_weight_var": (baseline_weight_var or ""),
        "weight_var": weight_var,
    }
    payload["weights"] = block


def _ensure_preprocess_block(payload: dict, spec_id: str) -> None:
    if isinstance(payload.get("preprocess"), dict) and payload.get("preprocess"):
        return
    parts = spec_id.split("/")
    target = parts[2] if len(parts) > 2 else ""
    variant = "/".join(parts[3:]) if len(parts) > 3 else ""
    payload["preprocess"] = {"spec_id": spec_id, "target": target, "variant": variant}


def _ensure_estimation_block(payload: dict, spec_id: str) -> None:
    if isinstance(payload.get("estimation"), dict) and payload.get("estimation"):
        return
    parts = spec_id.split("/")
    wrapper = parts[2] if len(parts) > 2 else ""
    family = parts[3] if len(parts) > 3 else ""
    variant = "/".join(parts[4:]) if len(parts) > 4 else ""
    block: dict = {
        "spec_id": spec_id,
        "wrapper": wrapper,
        "family": family,
        "variant": variant,
    }
    if wrapper == "dml" and family:
        block["score_family"] = family
    payload["estimation"] = block


def _ensure_joint_block(payload: dict, spec_id: str) -> None:
    if isinstance(payload.get("joint"), dict) and payload.get("joint"):
        return
    parts = spec_id.split("/")
    family = parts[2] if len(parts) > 2 else ""
    axes = [a for a in family.split("_") if a]
    payload["joint"] = {"spec_id": spec_id, "axes_changed": axes, "details": {}}


def _ensure_functional_form_interpretation(payload: dict, spec_id: str) -> None:
    ff = payload.get("functional_form")
    if not isinstance(ff, dict) or not ff:
        # If missing, create minimal block.
        payload["functional_form"] = {"spec_id": spec_id, "interpretation": "See spec_id for transformation; interpret coefficient accordingly."}
        return
    if not str(ff.get("spec_id", "")).strip():
        ff["spec_id"] = spec_id
    if not str(ff.get("interpretation", "")).strip():
        # Best-effort generic fallback.
        ff["interpretation"] = "Functional-form robustness variant; interpret coefficient under the transformed variable definition described here."
    op = str(ff.get("operation", "")).strip().lower()
    if op in {"binarize", "threshold"}:
        # Try to recover threshold/direction from recode_rule when possible.
        placeholder = {"unspecified", "unknown", "n/a"}
        thresh_raw = ff.get("threshold", None)
        thresh = None if (isinstance(thresh_raw, str) and thresh_raw.strip().lower() in placeholder) else thresh_raw
        if thresh in ("", None):
            thresh = None
        direction_raw = str(ff.get("direction", "")).strip()
        direction = "" if direction_raw.lower() in placeholder else direction_raw
        units_raw = str(ff.get("units", "")).strip()
        units = ""
        if units_raw and (units_raw.lower() not in placeholder) and (not units_raw.lower().startswith("same units as")):
            units = units_raw

        if (thresh is None) or (not direction):
            rr = str(ff.get("recode_rule", "")).strip()
            if rr:
                import re

                m = re.search(r"([<>]=?)\\s*([-+]?[0-9]*\\.?[0-9]+)", rr)
                if m:
                    direction = direction or m.group(1)
                    try:
                        thresh = float(m.group(2))
                    except Exception:
                        pass
        if thresh is not None:
            ff["threshold"] = thresh
        else:
            ff.pop("threshold", None)
        if direction:
            ff["direction"] = direction
        else:
            ff.pop("direction", None)
        if units:
            ff["units"] = units
        else:
            ff.pop("units", None)
    payload["functional_form"] = ff


def _normalize_spec_results(pkg_dir: Path, surface: dict, surface_hash: str, software: dict) -> None:
    path = pkg_dir / "specification_results.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = _ensure_run_success_and_error(df)

    # baseline_group_id -> design_code / design_audit (for design audit blocks)
    design_code_by_gid: dict[str, str] = {}
    design_audit_by_gid: dict[str, dict] = {}
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

    # Build baseline maps for relative comparisons.
    base_payload_by_gid = _baseline_rows(df)
    baseline_controls_by_gid: dict[str, set[str]] = {}
    baseline_fe_by_gid: dict[str, str] = {}
    baseline_n_by_gid: dict[str, float] = {}
    baseline_w_by_gid: dict[str, str] = {}
    baseline_treat_by_gid: dict[str, str] = {}

    # Pull baseline FE/N from the baseline rows in the CSV (preferred).
    if "baseline_group_id" in df.columns:
        sub = df[df["spec_id"].astype(str).str.strip() == "baseline"].copy()
        if "run_success" in sub.columns:
            sub["_rs"] = pd.to_numeric(sub["run_success"], errors="coerce").fillna(0).astype(int)
        else:
            sub["_rs"] = 0
        sub = sub.sort_values(["baseline_group_id", "_rs"], ascending=[True, False])
        for gid, gdf in sub.groupby(sub["baseline_group_id"].astype(str).str.strip(), dropna=False):
            row = gdf.iloc[0]
            baseline_fe_by_gid[str(gid)] = str(row.get("fixed_effects", "") or "").strip()
            baseline_n_by_gid[str(gid)] = _safe_float(row.get("n_obs"))
            baseline_treat_by_gid[str(gid)] = str(row.get("treatment_var", "") or "").strip()

    # Baseline controls from baseline coefficient keys (excluding Intercept and treatment var).
    for gid, bp in base_payload_by_gid.items():
        coef_keys = _coef_keys_from_payload(bp)
        t = baseline_treat_by_gid.get(gid, "")
        baseline_controls_by_gid[gid] = set([k for k in coef_keys if k not in {"Intercept", t}])

    # Baseline weight var from the surface (preferred)
    for g in surface.get("baseline_groups", []) or []:
        if not isinstance(g, dict):
            continue
        gid = str(g.get("baseline_group_id", "")).strip()
        if not gid:
            continue
        w = ""
        bs = g.get("baseline_specs", []) or []
        if isinstance(bs, list) and bs:
            b0 = bs[0] if isinstance(bs[0], dict) else {}
            w = str(b0.get("weight_var", "")).strip()
        baseline_w_by_gid[gid] = w

    canon_by_gid, infer_by_gid = _inference_plan_maps(surface)

    def _row_inference(gid: str, spec_id: str) -> dict:
        gid = str(gid).strip()
        # spec_results should always use canonical inference.
        return canon_by_gid.get(gid, {"spec_id": "", "params": {}})

    out_rows = []
    for _, row in df.iterrows():
        sid = str(row.get("spec_id", "")).strip()
        gid = str(row.get("baseline_group_id", "")).strip()
        run_success = _safe_int(row.get("run_success", 0), default=0)
        run_error = str(row.get("run_error", "") or "").strip()
        outcome_var = str(row.get("outcome_var", "")).strip()
        treatment_var = str(row.get("treatment_var", "")).strip()

        legacy = _load_json_obj(str(row.get("coefficient_vector_json", "")))
        coefs, meta = _split_legacy_payload(legacy)

        payload: dict = dict(meta)
        payload["coefficients"] = coefs

        # Required audit keys
        payload["inference"] = _row_inference(gid, sid)
        payload["software"] = software
        payload["surface_hash"] = surface_hash

        if run_success == 0:
            payload.setdefault("error", run_error or "run_success=0")
            _ensure_error_details(payload, run_error)
            # Ensure scalar numeric fields are empty for failures
            for c in NUMERIC_SCALAR_COLS:
                if c in df.columns:
                    row[c] = np.nan
        else:
            # RC axis blocks (best-effort)
            baseline_controls = baseline_controls_by_gid.get(gid)
            if sid.startswith("rc/controls/"):
                _ensure_controls_block(payload=payload, spec_id=sid, treatment_var=treatment_var, baseline_controls=baseline_controls)
            if sid.startswith("rc/sample/"):
                _ensure_sample_block(
                    payload=payload,
                    spec_id=sid,
                    outcome_var=outcome_var,
                    baseline_n_obs=baseline_n_by_gid.get(gid),
                    row_n_obs=_safe_float(row.get("n_obs")),
                )
            if sid.startswith("rc/fe/"):
                _ensure_fixed_effects_block(
                    payload=payload,
                    spec_id=sid,
                    baseline_fe=baseline_fe_by_gid.get(gid, ""),
                    row_fe=str(row.get("fixed_effects", "") or "").strip(),
                )
            if sid.startswith("rc/weights/"):
                _ensure_weights_block(payload=payload, spec_id=sid, baseline_weight_var=baseline_w_by_gid.get(gid, ""))
            if sid.startswith("rc/preprocess/"):
                _ensure_preprocess_block(payload, sid)
            if sid.startswith("rc/estimation/"):
                _ensure_estimation_block(payload, sid)
            if sid.startswith("rc/joint/"):
                _ensure_joint_block(payload, sid)
            if sid.startswith("rc/form/"):
                _ensure_functional_form_interpretation(payload, sid)

            # Design audit block. Store at least an estimator label, and (when present)
            # copy baseline-group design_audit from the surface.
            dc_row = ""
            if sid.startswith("design/"):
                parts = sid.split("/")
                if len(parts) > 1:
                    dc_row = parts[1]
            if not dc_row:
                dc_row = design_code_by_gid.get(gid, "")

            if dc_row:
                surface_da = design_audit_by_gid.get(gid, {})
                is_baseline_or_rc = (sid == "baseline") or sid.startswith(("baseline__", "baseline/")) or sid.startswith("rc/")

                design_block = payload.get("design")
                if not isinstance(design_block, dict):
                    design_block = {}
                dc_payload = design_block.get(dc_row)
                if not isinstance(dc_payload, dict):
                    dc_payload = {}

                est = str(dc_payload.get("estimator", "") or "").strip()
                if not est:
                    # Prefer explicit estimator variants from the spec_id when available.
                    if sid.startswith(f"design/{dc_row}/estimator/"):
                        est = sid.split("/", 3)[-1].split("/", 1)[0].strip()
                    else:
                        est = "paper_baseline"
                    dc_payload["estimator"] = est

                # Merge surface design audit metadata.
                if isinstance(surface_da, dict) and surface_da:
                    for k, v in surface_da.items():
                        if is_baseline_or_rc:
                            dc_payload[k] = v
                        else:
                            dc_payload.setdefault(k, v)

                design_block[dc_row] = dc_payload
                payload["design"] = design_block

        # Ensure axis blocks self-identify the executed spec_id.
        for k in AXIS_BLOCK_KEYS:
            b = payload.get(k)
            if isinstance(b, dict) and b:
                b["spec_id"] = sid
                payload[k] = b

        _move_unknown_payload_keys_to_extra(payload)

        row["coefficient_vector_json"] = json.dumps(payload, ensure_ascii=False)
        out_rows.append(row)

    out = pd.DataFrame(out_rows)
    out.to_csv(path, index=False)


def _normalize_inference_results(pkg_dir: Path, surface: dict, surface_hash: str, software: dict) -> None:
    path = pkg_dir / "inference_results.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = _ensure_run_success_and_error(df)

    canon_by_gid, infer_by_gid = _inference_plan_maps(surface)

    def _infer_row_inference(gid: str, infer_spec_id: str) -> dict:
        gid = str(gid).strip()
        sid = str(infer_spec_id).strip()
        by = infer_by_gid.get(gid, {})
        return by.get(sid, {"spec_id": sid, "params": {}})

    out_rows = []
    for _, row in df.iterrows():
        gid = str(row.get("baseline_group_id", "")).strip()
        sid = str(row.get("spec_id", "")).strip()
        run_success = _safe_int(row.get("run_success", 0), default=0)
        run_error = str(row.get("run_error", "") or "").strip()

        legacy = _load_json_obj(str(row.get("coefficient_vector_json", "")))
        coefs, meta = _split_legacy_payload(legacy)
        payload: dict = dict(meta)
        payload["coefficients"] = coefs
        payload["inference"] = _infer_row_inference(gid, sid)
        payload["software"] = software
        payload["surface_hash"] = surface_hash

        if run_success == 0:
            payload.setdefault("error", run_error or "run_success=0")
            _ensure_error_details(payload, run_error)
            for c in NUMERIC_SCALAR_COLS:
                if c in df.columns:
                    row[c] = np.nan

        _move_unknown_payload_keys_to_extra(payload)

        row["coefficient_vector_json"] = json.dumps(payload, ensure_ascii=False)
        out_rows.append(row)

    out = pd.DataFrame(out_rows)
    out.to_csv(path, index=False)


def normalize_paper(paper_id: str, extracted_dir: Path) -> None:
    pkg_dir = extracted_dir / paper_id
    surf_path = pkg_dir / "SPECIFICATION_SURFACE.json"
    if not surf_path.exists():
        raise FileNotFoundError(f"Missing {surf_path}")
    surface = json.loads(surf_path.read_text(encoding="utf-8", errors="replace"))
    sh = _surface_hash(surface)
    sw = _software_block()

    _normalize_spec_results(pkg_dir, surface, sh, sw)
    _normalize_inference_results(pkg_dir, surface, sh, sw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-id", type=str, default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--extracted-dir", type=str, default=str(DEFAULT_EXTRACTED_DIR))
    args = ap.parse_args()

    extracted_dir = Path(args.extracted_dir)
    if not extracted_dir.exists():
        raise FileNotFoundError(f"Missing extracted dir: {extracted_dir}")

    if args.all:
        paper_ids = sorted([p.name for p in extracted_dir.glob("*") if p.is_dir()])
    else:
        if not args.paper_id.strip():
            raise SystemExit("Provide --paper-id or --all")
        paper_ids = [args.paper_id.strip()]

    for pid in paper_ids:
        normalize_paper(pid, extracted_dir)
        print(f"Normalized {pid}")


if __name__ == "__main__":
    main()
