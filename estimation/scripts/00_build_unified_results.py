#!/usr/bin/env python3
"""
00_build_unified_results.py
===========================

Build unified, schema-normalized result tables from per-paper outputs in:
  data/downloads/extracted/{PAPER_ID}/

Writes:
  - unified_results.csv                  (estimate rows only: baseline/design/rc)
  - unified_inference_results.csv        (inference-only rows: infer/*, if any found)
  - estimation/data/unified_results_duplicate_spec_run_ids.csv
  - estimation/data/unified_results_duplicate_fingerprints.csv

This script enforces a stable column set for downstream estimation and adds:
  - run_success / run_error (if missing)
  - deterministic duplicate-tracking fields (fingerprint + group/rank flags)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXTRACTED_DIR = ROOT / "data" / "downloads" / "extracted"

OUT_UNIFIED = ROOT / "unified_results.csv"
OUT_UNIFIED_INFER = ROOT / "unified_inference_results.csv"

REPORT_DIR = ROOT / "estimation" / "data"
REPORT_DUP_SPEC_RUN_ID = REPORT_DIR / "unified_results_duplicate_spec_run_ids.csv"
REPORT_DUP_FINGERPRINT = REPORT_DIR / "unified_results_duplicate_fingerprints.csv"


STANDARD_SPEC_COLS: list[str] = [
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

STANDARD_INFER_COLS: list[str] = [
    "paper_id",
    "inference_run_id",
    "spec_run_id",  # reference estimate row (may be blank when unavailable)
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


def _coerce_json_str(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "{}"
    s = str(x)
    return s if s.strip() else "{}"


def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out


def _compute_run_success_and_error(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    coef = pd.to_numeric(out.get("coefficient"), errors="coerce")
    se = pd.to_numeric(out.get("std_error"), errors="coerce")
    has_valid_se = np.isfinite(se) & (se > 0)
    has_valid_coef = np.isfinite(coef)
    inferred_success = (has_valid_coef & has_valid_se).astype(int)

    if "run_success" not in out.columns:
        out["run_success"] = inferred_success
    else:
        out["run_success"] = pd.to_numeric(out["run_success"], errors="coerce").fillna(inferred_success).astype(int)

    if "run_error" not in out.columns:
        out["run_error"] = ""
    out["run_error"] = out["run_error"].fillna("").astype(str)

    # If the source has a notes column (common failure format), use it as the error message.
    if "notes" in out.columns:
        notes = out["notes"].fillna("").astype(str)
        needs = (out["run_success"] == 0) & (out["run_error"].str.strip() == "") & (notes.str.strip() != "")
        out.loc[needs, "run_error"] = notes[needs]

    # Fallback: synthesize a short error when we still have nothing.
    needs = (out["run_success"] == 0) & (out["run_error"].str.strip() == "")
    if needs.any():
        missing_parts: list[str] = []
        missing_parts.append(np.where(~has_valid_coef, "coef", ""))
        missing_parts.append(np.where(~has_valid_se, "se", ""))
        msg = []
        for i in range(len(out)):
            if not bool(needs.iloc[i]):
                msg.append("")
                continue
            parts = [p for p in (missing_parts[0][i], missing_parts[1][i]) if p]
            if parts:
                msg.append(f"missing/invalid:{'+'.join(parts)}")
            else:
                msg.append("run_success=0")
        out.loc[needs, "run_error"] = pd.Series(msg, index=out.index)[needs]

    return out


def _spec_fingerprint(df: pd.DataFrame, extra_fields: list[str] | None = None) -> pd.Series:
    extra_fields = extra_fields or []
    fields = [
        "paper_id",
        "baseline_group_id",
        "spec_id",
        "spec_tree_path",
        "outcome_var",
        "treatment_var",
        "sample_desc",
        "fixed_effects",
        "controls_desc",
        "cluster_var",
        *extra_fields,
    ]
    fields = [f for f in fields if f in df.columns]

    def _row_hash(row) -> str:
        obj = {}
        for f in fields:
            v = row.get(f, "")
            if v is None or (isinstance(v, float) and np.isnan(v)):
                v = ""
            obj[f] = str(v)
        payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    return df.apply(_row_hash, axis=1)


def _add_duplicate_tracking(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    sort_cols = key_cols.copy()
    ascending = [True] * len(sort_cols)
    if "run_success" in out.columns:
        sort_cols.append("run_success")
        ascending.append(False)
    sort_cols.append("spec_run_id")
    ascending.append(True)
    out = out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    g = out.groupby(key_cols, dropna=False)
    out["dup_group_size"] = g["spec_run_id"].transform("size").astype(int)
    out["dup_rank"] = g.cumcount().astype(int) + 1
    out["dup_canonical_spec_run_id"] = g["spec_run_id"].transform("first").astype(str)
    out["dup_is_duplicate"] = ((out["dup_group_size"] > 1) & (out["dup_rank"] > 1)).astype(int)
    return out


def _paper_title_map() -> dict[str, str]:
    out: dict[str, str] = {}
    for surf in sorted(EXTRACTED_DIR.glob("*/SPECIFICATION_SURFACE.json")):
        paper_id = surf.parent.name
        try:
            d = json.loads(surf.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        title = str(d.get("paper_title", "")).strip()
        if title:
            out[paper_id] = title
    return out


def _read_spec_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "paper_id" not in df.columns:
        df["paper_id"] = path.parent.name
    if "journal" not in df.columns:
        df["journal"] = ""
    df["paper_id"] = df["paper_id"].astype(str)
    df["spec_run_id"] = df["spec_run_id"].astype(str)
    df["spec_id"] = df["spec_id"].astype(str)
    df["coefficient_vector_json"] = df.get("coefficient_vector_json", "{}").apply(_coerce_json_str)
    df = _compute_run_success_and_error(df)
    df = _ensure_cols(df, STANDARD_SPEC_COLS)
    return df


def _read_inference_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "paper_id" not in df.columns:
        df["paper_id"] = path.parent.name
    df["paper_id"] = df["paper_id"].astype(str)
    if "inference_run_id" not in df.columns:
        # If this file is a direct dump of `infer/*` rows, treat its `spec_run_id` as the inference ID.
        df["inference_run_id"] = df.get("spec_run_id", "")
    df["inference_run_id"] = df["inference_run_id"].astype(str)
    df["spec_run_id"] = df.get("spec_run_id", "").astype(str)
    df["spec_id"] = df["spec_id"].astype(str)
    df["coefficient_vector_json"] = df.get("coefficient_vector_json", "{}").apply(_coerce_json_str)
    df = _compute_run_success_and_error(df)
    df = _ensure_cols(df, STANDARD_INFER_COLS)
    return df


def main() -> None:
    if not EXTRACTED_DIR.exists():
        raise FileNotFoundError(f"Missing extracted directory: {EXTRACTED_DIR}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    spec_paths = sorted(EXTRACTED_DIR.glob("*/specification_results.csv"))
    if not spec_paths:
        raise FileNotFoundError(f"No per-paper specification_results.csv found under {EXTRACTED_DIR}")

    title_map = _paper_title_map()

    spec_rows: list[pd.DataFrame] = []
    infer_rows: list[pd.DataFrame] = []

    for p in spec_paths:
        df = _read_spec_results(p)

        # Split infer rows (if they still exist in specification_results.csv).
        infer_mask = df["spec_id"].astype(str).str.startswith("infer/")
        if infer_mask.any():
            infer_df = df.loc[infer_mask].copy()
            # Convert to inference schema; mapping to a base estimate spec_run_id is not always available.
            infer_df = infer_df.rename(columns={"spec_run_id": "inference_run_id"})
            infer_df["spec_run_id"] = ""  # reference estimate row unknown in this format
            infer_df = _ensure_cols(infer_df, STANDARD_INFER_COLS)
            infer_rows.append(infer_df[STANDARD_INFER_COLS + [c for c in infer_df.columns if c not in STANDARD_INFER_COLS]])

        core_df = df.loc[~infer_mask].copy()
        core_df["paper_title"] = core_df["paper_id"].map(title_map).fillna("")

        # Fingerprint + duplicate tracking for core estimates.
        extra_fp_fields = [c for c in ["se_type", "weights_desc"] if c in core_df.columns]
        core_df["spec_fingerprint"] = _spec_fingerprint(core_df, extra_fields=extra_fp_fields)
        core_df = _add_duplicate_tracking(core_df, key_cols=["paper_id", "spec_fingerprint"])

        spec_rows.append(core_df)

        # Preferred inference file (if present).
        infer_path = p.parent / "inference_results.csv"
        if infer_path.exists():
            inf = _read_inference_results(infer_path)
            infer_rows.append(inf)

    unified = pd.concat(spec_rows, ignore_index=True)

    # Enforce uniqueness of (paper_id, spec_run_id) for downstream joins/verification merges.
    dup_mask = unified.duplicated(subset=["paper_id", "spec_run_id"], keep=False)
    if dup_mask.any():
        dups = unified.loc[dup_mask].copy()
        dups = dups.sort_values(["paper_id", "spec_run_id", "run_success"], ascending=[True, True, False])
        dups.to_csv(REPORT_DUP_SPEC_RUN_ID, index=False)

        # Deduplicate deterministically: prefer run_success==1, then first occurrence.
        unified = unified.sort_values(["paper_id", "spec_run_id", "run_success"], ascending=[True, True, False])
        unified = unified.drop_duplicates(subset=["paper_id", "spec_run_id"], keep="first").reset_index(drop=True)
        print(f"Warning: duplicate (paper_id, spec_run_id) rows found; wrote {REPORT_DUP_SPEC_RUN_ID} and kept first per key.")

    # Report duplicate fingerprints (potential duplicates by signature).
    fp_dups = unified[unified["dup_group_size"] > 1].copy()
    if len(fp_dups) > 0:
        fp_dups.to_csv(REPORT_DUP_FINGERPRINT, index=False)

    # Standard column ordering first; preserve any extras at the end.
    base_cols = [
        "paper_id",
        "paper_title",
        "journal",
        *[c for c in STANDARD_SPEC_COLS if c != "paper_id"],
        "spec_fingerprint",
        "dup_group_size",
        "dup_rank",
        "dup_canonical_spec_run_id",
        "dup_is_duplicate",
    ]
    base_cols = [c for c in base_cols if c in unified.columns]
    extra_cols = [c for c in unified.columns if c not in base_cols]
    unified = unified[base_cols + extra_cols]

    unified.to_csv(OUT_UNIFIED, index=False)
    print(f"Wrote {OUT_UNIFIED} ({len(unified)} rows)")

    if infer_rows:
        infer = pd.concat(infer_rows, ignore_index=True)

        infer = infer.sort_values(["paper_id", "inference_run_id"]).reset_index(drop=True)
        infer.to_csv(OUT_UNIFIED_INFER, index=False)
        print(f"Wrote {OUT_UNIFIED_INFER} ({len(infer)} rows)")
    else:
        if OUT_UNIFIED_INFER.exists():
            OUT_UNIFIED_INFER.unlink()
        print("No inference rows found; did not write unified_inference_results.csv")

    if REPORT_DUP_SPEC_RUN_ID.exists():
        print(f"Spec-run-id duplicates report: {REPORT_DUP_SPEC_RUN_ID}")
    if REPORT_DUP_FINGERPRINT.exists():
        print(f"Fingerprint duplicates report: {REPORT_DUP_FINGERPRINT}")


if __name__ == "__main__":
    main()
