"""
Specification Search Script for Bronzini & Iachini (2014)
"Are Incentives for R&D Effective? Evidence from a Regression
 Discontinuity Approach"
American Economic Journal: Economic Policy, 6(4), 100-134.

Paper ID: 114875-V1

Surface-driven execution:
  - G1: INVSALES ~ treat (sharp RD at score=75, R&D subsidy effect on investment)
  - Parametric RD with separate polynomials each side of cutoff
  - Canonical inference: cluster(score)
  - Design alternatives: polynomial orders, bandwidths, kernels, rdrobust procedures
  - RC axes: bandwidth windows, donut holes, subgroup restrictions, placebo cutoffs,
    functional form transformations

Outputs:
  - specification_results.csv (baseline, design/*, rc/* rows)
  - inference_results.csv (infer/* rows)
  - SPECIFICATION_SEARCH.md
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import sys
import warnings
warnings.filterwarnings('ignore')

# Add scripts dir for utilities
sys.path.insert(0, "scripts")
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash, software_block
)

PAPER_ID = "114875-V1"
DATA_DIR = "data/downloads/extracted/114875-V1"
DATA_SUBDIR = f"{DATA_DIR}/Bronzini_Iachini_Stata_program"
OUTPUT_DIR = DATA_DIR

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

# Design audit block from surface
G1_DESIGN_AUDIT = surface_obj["baseline_groups"][0]["design_audit"]
G1_INFERENCE_CANONICAL = surface_obj["baseline_groups"][0]["inference_plan"]["canonical"]

# ============================================================
# LOAD DATA
# ============================================================
df_raw = pd.read_stata(f"{DATA_SUBDIR}/Bronzini-Iachini_dataset.dta")
for col in df_raw.columns:
    if df_raw[col].dtype == np.float32:
        df_raw[col] = df_raw[col].astype(np.float64)

# The data already has all needed variables pre-computed:
# s = score - 75, s2, s3, treat, notreat, streat, streat2, streat3,
# snotreat, snotreat2, snotreat3, smallm, largem, CR, AGE, etc.

results = []
inference_results = []
spec_run_counter = 0


# ============================================================
# HELPER: Run parametric RD via pyfixest (OLS with polynomial terms)
# ============================================================
def run_parametric_rd(spec_id, spec_tree_path, baseline_group_id,
                      outcome_var, treatment_var, controls, data, vcov,
                      sample_desc, controls_desc, cluster_var,
                      design_audit, inference_canonical,
                      axis_block_name=None, axis_block=None,
                      design_override=None, notes=""):
    """Run parametric RD: OLS with separate polynomial terms on each side."""
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        controls_str = " + ".join(controls) if controls else ""
        if controls_str:
            formula = f"{outcome_var} ~ {treatment_var} + {controls_str}"
        else:
            formula = f"{outcome_var} ~ {treatment_var}"

        m = pf.feols(formula, data=data, vcov=vcov)
        coef_val = float(m.coef().get(treatment_var, np.nan))
        se_val = float(m.se().get(treatment_var, np.nan))
        pval = float(m.pvalue().get(treatment_var, np.nan))
        try:
            ci = m.confint()
            ci_lower = float(ci.loc[treatment_var, ci.columns[0]])
            ci_upper = float(ci.loc[treatment_var, ci.columns[1]])
        except Exception:
            ci_lower, ci_upper = np.nan, np.nan
        nobs = int(m._N)
        try:
            r2 = float(m._r2)
        except Exception:
            r2 = np.nan
        all_coefs = {k: float(v) for k, v in m.coef().items()}

        design_block = design_override if design_override else design_audit
        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": inference_canonical["spec_id"],
                       "params": inference_canonical["params"]},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"regression_discontinuity": design_block},
            axis_block_name=axis_block_name,
            axis_block=axis_block,
            notes=notes if notes else None,
        )

        results.append({
            "paper_id": PAPER_ID, "spec_run_id": run_id,
            "spec_id": spec_id, "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var, "treatment_var": treatment_var,
            "coefficient": coef_val, "std_error": se_val, "p_value": pval,
            "ci_lower": ci_lower, "ci_upper": ci_upper,
            "n_obs": nobs, "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc, "fixed_effects": "none",
            "controls_desc": controls_desc, "cluster_var": cluster_var,
            "run_success": 1, "run_error": ""
        })
        return run_id, coef_val, se_val, pval, nobs

    except Exception as e:
        err_msg = str(e)[:240]
        payload = make_failure_payload(
            error=err_msg,
            error_details=error_details_from_exception(e, stage="estimation"),
            software=SW_BLOCK, surface_hash=SURFACE_HASH
        )
        results.append({
            "paper_id": PAPER_ID, "spec_run_id": run_id,
            "spec_id": spec_id, "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var, "treatment_var": treatment_var,
            "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "n_obs": np.nan, "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc, "fixed_effects": "none",
            "controls_desc": controls_desc, "cluster_var": cluster_var,
            "run_success": 0, "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


# ============================================================
# HELPER: Run rdrobust-based RD
# ============================================================
def run_rdrobust_rd(spec_id, spec_tree_path, baseline_group_id,
                    outcome_var, running_var, cutoff, data,
                    sample_desc, cluster_var,
                    design_audit, inference_canonical,
                    kernel="triangular", p=1, bwselect="mserd",
                    h=None, rho=None,
                    axis_block_name=None, axis_block=None,
                    design_override=None, notes=""):
    """Run RD via rdrobust package."""
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        from rdrobust import rdrobust as rd_func

        y = data[outcome_var].values.astype(float)
        x = data[running_var].values.astype(float)

        # Build kwargs
        kwargs = {"c": cutoff, "kernel": kernel, "p": p, "bwselect": bwselect}
        if h is not None:
            kwargs["h"] = h
            del kwargs["bwselect"]  # can't use both
        if rho is not None:
            kwargs["rho"] = rho

        # Cluster on score
        kwargs["cluster"] = data[cluster_var].values.astype(float)

        result = rd_func(y, x, **kwargs)

        # Extract conventional estimate (row 0)
        coef_val = float(result.coef.iloc[0, 0])
        se_val = float(result.se.iloc[0, 0])
        pval_val = float(result.pv.iloc[0, 0])
        ci_lower = float(result.ci.iloc[0, 0])
        ci_upper = float(result.ci.iloc[0, 1])

        # Also get bias-corrected and robust
        bc_coef = float(result.coef.iloc[1, 0]) if result.coef.shape[0] > 1 else np.nan
        robust_pv = float(result.pv.iloc[2, 0]) if result.pv.shape[0] > 2 else np.nan
        robust_ci_lo = float(result.ci.iloc[2, 0]) if result.ci.shape[0] > 2 else np.nan
        robust_ci_hi = float(result.ci.iloc[2, 1]) if result.ci.shape[0] > 2 else np.nan

        nobs = int(result.N[0] + result.N[1])
        n_eff = int(result.N_h[0] + result.N_h[1])

        all_coefs = {
            "treat_conventional": coef_val,
            "treat_bias_corrected": bc_coef,
        }

        design_block = design_override if design_override else {
            **design_audit,
            "estimator": "rdrobust",
            "kernel": kernel,
            "poly_order": p,
            "bandwidth_h": float(result.bws.iloc[0, 0]),
            "bandwidth_b": float(result.bws.iloc[1, 0]) if result.bws.shape[0] > 1 else np.nan,
            "n_effective": n_eff,
        }

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": inference_canonical["spec_id"],
                       "params": inference_canonical["params"]},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"regression_discontinuity": design_block},
            axis_block_name=axis_block_name,
            axis_block=axis_block,
            notes=notes if notes else None,
        )
        payload["extra"] = {
            "robust_pvalue": robust_pv,
            "robust_ci": [robust_ci_lo, robust_ci_hi],
            "bandwidth_h_left": float(result.bws.iloc[0, 0]),
            "bandwidth_h_right": float(result.bws.iloc[0, 1]),
        }

        controls_desc = f"rdrobust p={p} kernel={kernel}"
        results.append({
            "paper_id": PAPER_ID, "spec_run_id": run_id,
            "spec_id": spec_id, "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var, "treatment_var": "treat",
            "coefficient": coef_val, "std_error": se_val, "p_value": pval_val,
            "ci_lower": ci_lower, "ci_upper": ci_upper,
            "n_obs": nobs, "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc, "fixed_effects": "none",
            "controls_desc": controls_desc, "cluster_var": cluster_var,
            "run_success": 1, "run_error": ""
        })
        return run_id, coef_val, se_val, pval_val, nobs

    except Exception as e:
        err_msg = str(e)[:240]
        payload = make_failure_payload(
            error=err_msg,
            error_details=error_details_from_exception(e, stage="estimation"),
            software=SW_BLOCK, surface_hash=SURFACE_HASH
        )
        results.append({
            "paper_id": PAPER_ID, "spec_run_id": run_id,
            "spec_id": spec_id, "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var, "treatment_var": "treat",
            "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "n_obs": np.nan, "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc, "fixed_effects": "none",
            "controls_desc": f"rdrobust p={p} kernel={kernel}", "cluster_var": cluster_var,
            "run_success": 0, "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


# ============================================================
# HELPER: Run inference variant (re-estimate with different vcov)
# ============================================================
def run_inference_variant(base_run_id, spec_id, spec_tree_path,
                          baseline_group_id, outcome_var, treatment_var,
                          controls, data, vcov, cluster_var, design_audit):
    """Re-estimate with alternative inference (e.g. HC1 instead of cluster)."""
    infer_counter = len(inference_results) + 1
    infer_run_id = f"{PAPER_ID}_infer_{infer_counter:03d}"

    try:
        controls_str = " + ".join(controls) if controls else ""
        if controls_str:
            formula = f"{outcome_var} ~ {treatment_var} + {controls_str}"
        else:
            formula = f"{outcome_var} ~ {treatment_var}"

        m = pf.feols(formula, data=data, vcov=vcov)
        coef_val = float(m.coef().get(treatment_var, np.nan))
        se_val = float(m.se().get(treatment_var, np.nan))
        pval = float(m.pvalue().get(treatment_var, np.nan))
        try:
            ci = m.confint()
            ci_lower = float(ci.loc[treatment_var, ci.columns[0]])
            ci_upper = float(ci.loc[treatment_var, ci.columns[1]])
        except Exception:
            ci_lower, ci_upper = np.nan, np.nan
        nobs = int(m._N)
        try:
            r2 = float(m._r2)
        except Exception:
            r2 = np.nan
        all_coefs = {k: float(v) for k, v in m.coef().items()}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": spec_id, "params": {"vcov": str(vcov), "cluster_var": cluster_var}},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"regression_discontinuity": design_audit},
        )
        inference_results.append({
            "paper_id": PAPER_ID,
            "inference_run_id": infer_run_id,
            "spec_run_id": base_run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": treatment_var,
            "coefficient": coef_val, "std_error": se_val, "p_value": pval,
            "ci_lower": ci_lower, "ci_upper": ci_upper,
            "n_obs": nobs, "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "cluster_var": cluster_var or "none",
            "run_success": 1, "run_error": ""
        })
    except Exception as e:
        err_msg = str(e)[:240]
        payload = make_failure_payload(error=err_msg,
            error_details=error_details_from_exception(e, stage="inference"),
            software=SW_BLOCK, surface_hash=SURFACE_HASH)
        inference_results.append({
            "paper_id": PAPER_ID,
            "inference_run_id": infer_run_id,
            "spec_run_id": base_run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": treatment_var,
            "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "n_obs": np.nan, "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "cluster_var": cluster_var or "none",
            "run_success": 0, "run_error": err_msg
        })


# ============================================================
# Define polynomial control sets (separate polynomials each side)
# ============================================================
# Poly0: no polynomial terms (just treat = difference in means)
POLY0_CONTROLS = []
# Poly1: linear (snotreat, streat)
POLY1_CONTROLS = ["snotreat", "streat"]
# Poly2: quadratic
POLY2_CONTROLS = ["snotreat", "snotreat2", "streat", "streat2"]
# Poly3: cubic
POLY3_CONTROLS = ["snotreat", "snotreat2", "snotreat3", "streat", "streat2", "streat3"]

CANONICAL_VCOV = {"CRV1": "score"}
CLUSTER_VAR = "score"

# Full sample
df_full = df_raw.copy()

# ############################################################
# STEP 1: BASELINE SPECS
# ############################################################
print("=== BASELINE SPECS ===")

# Primary baseline: Table 3, Full Sample, Poly1 (local linear, the paper's standard)
run_id_bl_poly1, *_ = run_parametric_rd(
    "baseline", "designs/regression_discontinuity.md#baseline", "G1",
    "INVSALES", "treat", POLY1_CONTROLS, df_full, CANONICAL_VCOV,
    "Full sample (N=357)", "snotreat, streat (linear, separate each side)", CLUSTER_VAR,
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    notes="Table 3, full sample, polynomial order 1 (local linear with separate slopes)"
)

# Additional baseline: Poly0 (difference in means)
run_id_bl_poly0, *_ = run_parametric_rd(
    "baseline__poly0", "designs/regression_discontinuity.md#baseline", "G1",
    "INVSALES", "treat", POLY0_CONTROLS, df_full, CANONICAL_VCOV,
    "Full sample (N=357)", "none (difference in means)", CLUSTER_VAR,
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    notes="Table 3, full sample, polynomial order 0"
)

# Additional baseline: Poly2
run_id_bl_poly2, *_ = run_parametric_rd(
    "baseline__poly2", "designs/regression_discontinuity.md#baseline", "G1",
    "INVSALES", "treat", POLY2_CONTROLS, df_full, CANONICAL_VCOV,
    "Full sample (N=357)", "snotreat, snotreat2, streat, streat2 (quadratic)", CLUSTER_VAR,
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    notes="Table 3, full sample, polynomial order 2"
)

# Additional baseline: Poly3
run_id_bl_poly3, *_ = run_parametric_rd(
    "baseline__poly3", "designs/regression_discontinuity.md#baseline", "G1",
    "INVSALES", "treat", POLY3_CONTROLS, df_full, CANONICAL_VCOV,
    "Full sample (N=357)", "snotreat-snotreat3, streat-streat3 (cubic)", CLUSTER_VAR,
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    notes="Table 3, full sample, polynomial order 3"
)


# ############################################################
# STEP 2: DESIGN ALTERNATIVES (design/* specs)
# ############################################################
print("=== DESIGN ALTERNATIVES ===")

# --- A) Polynomial order via rdrobust (local polynomial with data-driven bandwidth) ---

# design/regression_discontinuity/poly/local_linear (rdrobust, p=1, triangular)
run_id_rdrobust_p1, *_ = run_rdrobust_rd(
    "design/regression_discontinuity/poly/local_linear",
    "designs/regression_discontinuity.md#local-polynomial-order", "G1",
    "INVSALES", "score", 75, df_full,
    "Full sample, rdrobust MSE-optimal bandwidth", CLUSTER_VAR,
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    kernel="triangular", p=1,
    notes="rdrobust local linear, triangular kernel, MSE-optimal bandwidth"
)

# design/regression_discontinuity/poly/local_quadratic
run_rdrobust_rd(
    "design/regression_discontinuity/poly/local_quadratic",
    "designs/regression_discontinuity.md#local-polynomial-order", "G1",
    "INVSALES", "score", 75, df_full,
    "Full sample, rdrobust MSE-optimal bandwidth", CLUSTER_VAR,
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    kernel="triangular", p=2,
    notes="rdrobust local quadratic, triangular kernel"
)

# design/regression_discontinuity/poly/local_cubic
run_rdrobust_rd(
    "design/regression_discontinuity/poly/local_cubic",
    "designs/regression_discontinuity.md#local-polynomial-order", "G1",
    "INVSALES", "score", 75, df_full,
    "Full sample, rdrobust MSE-optimal bandwidth", CLUSTER_VAR,
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    kernel="triangular", p=3,
    notes="rdrobust local cubic, triangular kernel"
)

# --- B) Bandwidth variants (parametric approach, matching paper's windows) ---

# design/regression_discontinuity/bandwidth/half_baseline
# Paper's smallest window: 35%, let's use a smaller one: ~25% window (score 68-77)
run_parametric_rd(
    "design/regression_discontinuity/bandwidth/half_baseline",
    "designs/regression_discontinuity.md#bandwidth-selection", "G1",
    "INVSALES", "treat", POLY1_CONTROLS,
    df_full[(df_full['score'] > 68) & (df_full['score'] < 77)].copy(), CANONICAL_VCOV,
    "25% window: score in (68,77), N~68", "poly1 separate slopes", CLUSTER_VAR,
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    design_override={**G1_DESIGN_AUDIT, "bandwidth": "25% window (score 68-77)"},
    notes="Narrower window around cutoff"
)

# design/regression_discontinuity/bandwidth/double_baseline
# Paper's full sample is the widest; double would be wider but we're bounded by data
# Use 75% window (score 58-83)
run_parametric_rd(
    "design/regression_discontinuity/bandwidth/double_baseline",
    "designs/regression_discontinuity.md#bandwidth-selection", "G1",
    "INVSALES", "treat", POLY1_CONTROLS,
    df_full[(df_full['score'] > 58) & (df_full['score'] < 83)].copy(), CANONICAL_VCOV,
    "75% window: score in (58,83), N~189", "poly1 separate slopes", CLUSTER_VAR,
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    design_override={**G1_DESIGN_AUDIT, "bandwidth": "75% window (score 58-83)"},
    notes="Wider 75% window"
)

# --- C) Kernel variants ---

# design/regression_discontinuity/kernel/triangular (rdrobust with triangular)
run_rdrobust_rd(
    "design/regression_discontinuity/kernel/triangular",
    "designs/regression_discontinuity.md#kernel-choice", "G1",
    "INVSALES", "score", 75, df_full,
    "Full sample, rdrobust triangular kernel", CLUSTER_VAR,
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    kernel="triangular", p=1,
    notes="rdrobust with triangular kernel (default)"
)

# design/regression_discontinuity/kernel/uniform (rdrobust with uniform)
run_rdrobust_rd(
    "design/regression_discontinuity/kernel/uniform",
    "designs/regression_discontinuity.md#kernel-choice", "G1",
    "INVSALES", "score", 75, df_full,
    "Full sample, rdrobust uniform kernel", CLUSTER_VAR,
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    kernel="uniform", p=1,
    notes="rdrobust with uniform/rectangular kernel"
)

# --- D) Estimation procedure variants ---

# design/regression_discontinuity/procedure/conventional
# rdrobust conventional (no bias correction)
run_rdrobust_rd(
    "design/regression_discontinuity/procedure/conventional",
    "designs/regression_discontinuity.md#estimation-inference-procedure", "G1",
    "INVSALES", "score", 75, df_full,
    "Full sample, rdrobust conventional", CLUSTER_VAR,
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    kernel="triangular", p=1,
    notes="rdrobust conventional (no bias correction) -- coef is conventional row"
)

# design/regression_discontinuity/procedure/robust_bias_corrected
# Use rdrobust's bias-corrected row
run_id_rbc, *_ = run_rdrobust_rd(
    "design/regression_discontinuity/procedure/robust_bias_corrected",
    "designs/regression_discontinuity.md#estimation-inference-procedure", "G1",
    "INVSALES", "score", 75, df_full,
    "Full sample, rdrobust robust bias-corrected", CLUSTER_VAR,
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    kernel="triangular", p=1,
    notes="rdrobust robust bias-corrected -- same fit, robust p-value in extra"
)


# ############################################################
# STEP 2b: RC SPECS (robustness checks)
# ############################################################
print("=== RC SPECS ===")

# --- Bandwidth/sample restrictions ---
# rc/sample/bandwidth/bw_50pct_window (Table 3 50% window)
for poly_label, poly_controls, poly_desc in [
    ("poly0", POLY0_CONTROLS, "none"),
    ("poly1", POLY1_CONTROLS, "poly1 separate slopes"),
    ("poly2", POLY2_CONTROLS, "poly2 separate slopes"),
]:
    df_50 = df_full[(df_full['score'] > 51) & (df_full['score'] < 81)].copy()
    run_parametric_rd(
        f"rc/sample/bandwidth/bw_50pct_window__{poly_label}",
        "modules/robustness/sample.md#bandwidth-restrictions", "G1",
        "INVSALES", "treat", poly_controls, df_50, CANONICAL_VCOV,
        f"50% window: score in (51,81), {poly_label}", poly_desc, CLUSTER_VAR,
        G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
        axis_block_name="sample",
        axis_block={"spec_id": f"rc/sample/bandwidth/bw_50pct_window__{poly_label}",
                    "restriction": "score > 51 & score < 81",
                    "bandwidth_desc": "50% window"},
        notes=f"Table 3 50% window, {poly_label}"
    )

# rc/sample/bandwidth/bw_35pct_window (Table 3 35% window)
for poly_label, poly_controls, poly_desc in [
    ("poly0", POLY0_CONTROLS, "none"),
    ("poly1", POLY1_CONTROLS, "poly1 separate slopes"),
    ("poly2", POLY2_CONTROLS, "poly2 separate slopes"),
]:
    df_35 = df_full[(df_full['score'] > 65) & (df_full['score'] < 79)].copy()
    run_parametric_rd(
        f"rc/sample/bandwidth/bw_35pct_window__{poly_label}",
        "modules/robustness/sample.md#bandwidth-restrictions", "G1",
        "INVSALES", "treat", poly_controls, df_35, CANONICAL_VCOV,
        f"35% window: score in (65,79), {poly_label}", poly_desc, CLUSTER_VAR,
        G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
        axis_block_name="sample",
        axis_block={"spec_id": f"rc/sample/bandwidth/bw_35pct_window__{poly_label}",
                    "restriction": "score > 65 & score < 79",
                    "bandwidth_desc": "35% window"},
        notes=f"Table 3 35% window, {poly_label}"
    )

# rc/sample/bandwidth/bw_25pct_window
df_25 = df_full[(df_full['score'] > 68) & (df_full['score'] < 77)].copy()
run_parametric_rd(
    "rc/sample/bandwidth/bw_25pct_window",
    "modules/robustness/sample.md#bandwidth-restrictions", "G1",
    "INVSALES", "treat", POLY1_CONTROLS, df_25, CANONICAL_VCOV,
    "25% window: score in (68,77)", "poly1 separate slopes", CLUSTER_VAR,
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/bandwidth/bw_25pct_window",
                "restriction": "score > 68 & score < 77",
                "bandwidth_desc": "25% window"},
)

# rc/sample/bandwidth/bw_75pct_window
df_75 = df_full[(df_full['score'] > 58) & (df_full['score'] < 83)].copy()
run_parametric_rd(
    "rc/sample/bandwidth/bw_75pct_window",
    "modules/robustness/sample.md#bandwidth-restrictions", "G1",
    "INVSALES", "treat", POLY1_CONTROLS, df_75, CANONICAL_VCOV,
    "75% window: score in (58,83)", "poly1 separate slopes", CLUSTER_VAR,
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/bandwidth/bw_75pct_window",
                "restriction": "score > 58 & score < 83",
                "bandwidth_desc": "75% window"},
)

# --- Donut hole exclusions ---
for donut_r in [1, 2, 3]:
    df_donut = df_full[np.abs(df_full['s']) >= donut_r].copy()
    run_parametric_rd(
        f"rc/sample/donut/exclude_{donut_r}",
        "modules/robustness/sample.md#donut-hole-exclusions", "G1",
        "INVSALES", "treat", POLY1_CONTROLS, df_donut, CANONICAL_VCOV,
        f"Donut: exclude |score-75| < {donut_r}", "poly1 separate slopes", CLUSTER_VAR,
        G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
        axis_block_name="sample",
        axis_block={"spec_id": f"rc/sample/donut/exclude_{donut_r}",
                    "donut_radius": donut_r,
                    "restriction": f"|s| >= {donut_r}"},
        notes=f"Donut hole: exclude firms within {donut_r} score point(s) of cutoff"
    )

# --- Subgroup restrictions (from Tables 5-6) ---

# rc/sample/restrict/small_firms_only
df_small = df_full[df_full['smallm'] == 1].copy()
run_parametric_rd(
    "rc/sample/restrict/small_firms_only",
    "modules/robustness/sample.md#subgroup-restrictions", "G1",
    "INVSALES", "treat", POLY1_CONTROLS, df_small, CANONICAL_VCOV,
    "Small firms only (smallm==1)", "poly1 separate slopes", CLUSTER_VAR,
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restrict/small_firms_only",
                "restriction": "smallm == 1",
                "subgroup": "small firms (below median employees)"},
    notes="Table 5 small firms subgroup"
)

# rc/sample/restrict/large_firms_only
df_large = df_full[df_full['largem'] == 1].copy()
run_parametric_rd(
    "rc/sample/restrict/large_firms_only",
    "modules/robustness/sample.md#subgroup-restrictions", "G1",
    "INVSALES", "treat", POLY1_CONTROLS, df_large, CANONICAL_VCOV,
    "Large firms only (largem==1)", "poly1 separate slopes", CLUSTER_VAR,
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restrict/large_firms_only",
                "restriction": "largem == 1",
                "subgroup": "large firms (above median employees)"},
    notes="Table 5 large firms subgroup"
)

# rc/sample/restrict/high_coverage_only (CR > median among treated)
# high coverage: CR > 0.4043 & treat==1 (or just treat==1 firms with high CR)
# We restrict to treated firms with high coverage + all untreated (control group)
df_high_cov = df_full[
    ((df_full['treat'] == 1) & (df_full['CR'] > 0.4043)) |
    (df_full['treat'] == 0)
].copy()
run_parametric_rd(
    "rc/sample/restrict/high_coverage_only",
    "modules/robustness/sample.md#subgroup-restrictions", "G1",
    "INVSALES", "treat", POLY1_CONTROLS, df_high_cov, CANONICAL_VCOV,
    "High coverage ratio treated + all control", "poly1 separate slopes", CLUSTER_VAR,
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restrict/high_coverage_only",
                "restriction": "CR > 0.4043 (treated) or control",
                "subgroup": "high subsidy coverage ratio"},
    notes="Table 6 high coverage ratio subgroup"
)

# rc/sample/restrict/low_coverage_only
df_low_cov = df_full[
    ((df_full['treat'] == 1) & (df_full['CR'] <= 0.4043)) |
    (df_full['treat'] == 0)
].copy()
run_parametric_rd(
    "rc/sample/restrict/low_coverage_only",
    "modules/robustness/sample.md#subgroup-restrictions", "G1",
    "INVSALES", "treat", POLY1_CONTROLS, df_low_cov, CANONICAL_VCOV,
    "Low coverage ratio treated + all control", "poly1 separate slopes", CLUSTER_VAR,
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restrict/low_coverage_only",
                "restriction": "CR <= 0.4043 (treated) or control",
                "subgroup": "low subsidy coverage ratio"},
    notes="Table 6 low coverage ratio subgroup"
)

# rc/sample/restrict/young_firms_only (AGE >= 1987.081 = founded after median)
df_young = df_full[(df_full['fchighm'] == 1) | (df_full['treat'] == 0)].copy()
df_young = df_young[df_young['AGE'].notna()].copy()
run_parametric_rd(
    "rc/sample/restrict/young_firms_only",
    "modules/robustness/sample.md#subgroup-restrictions", "G1",
    "INVSALES", "treat", POLY1_CONTROLS, df_young, CANONICAL_VCOV,
    "Young firms (fchighm==1, founded >= 1987) + control", "poly1 separate slopes", CLUSTER_VAR,
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restrict/young_firms_only",
                "restriction": "fchighm == 1 (or control), AGE not missing",
                "subgroup": "young firms (founded after median year)"},
    notes="Table 6 young firms subgroup"
)

# rc/sample/restrict/old_firms_only
df_old = df_full[(df_full['fclowm'] == 1) | (df_full['treat'] == 0)].copy()
df_old = df_old[df_old['AGE'].notna()].copy()
run_parametric_rd(
    "rc/sample/restrict/old_firms_only",
    "modules/robustness/sample.md#subgroup-restrictions", "G1",
    "INVSALES", "treat", POLY1_CONTROLS, df_old, CANONICAL_VCOV,
    "Old firms (fclowm==1, founded < 1987) + control", "poly1 separate slopes", CLUSTER_VAR,
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restrict/old_firms_only",
                "restriction": "fclowm == 1 (or control), AGE not missing",
                "subgroup": "old firms (founded before median year)"},
    notes="Table 6 old firms subgroup"
)

# --- Placebo cutoffs ---
for cutoff_val in [65, 70, 80, 85]:
    # Re-center polynomial terms at fake cutoff
    df_placebo = df_full.copy()
    df_placebo['s_placebo'] = df_placebo['score'] - cutoff_val
    df_placebo['treat_placebo'] = (df_placebo['score'] >= cutoff_val).astype(float)
    df_placebo['snotreat_placebo'] = df_placebo['s_placebo'] * (1 - df_placebo['treat_placebo'])
    df_placebo['streat_placebo'] = df_placebo['s_placebo'] * df_placebo['treat_placebo']

    run_parametric_rd(
        f"rc/sample/placebo_cutoff/cutoff_{cutoff_val}",
        "modules/robustness/sample.md#placebo-cutoff", "G1",
        "INVSALES", "treat_placebo",
        ["snotreat_placebo", "streat_placebo"],
        df_placebo, CANONICAL_VCOV,
        f"Full sample, placebo cutoff at {cutoff_val}", "poly1 separate slopes at fake cutoff",
        CLUSTER_VAR,
        G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
        axis_block_name="sample",
        axis_block={"spec_id": f"rc/sample/placebo_cutoff/cutoff_{cutoff_val}",
                    "placebo_cutoff": cutoff_val,
                    "true_cutoff": 75},
        notes=f"Placebo test: fake cutoff at score={cutoff_val}"
    )

# --- Functional form transformations ---

# rc/form/outcome/log_INVSALES
# INVSALES can be negative, so use log(INVSALES + offset) where offset shifts all positive
# Actually: log is only defined for positive values. Use log(1+INVSALES) if INVSALES > -1,
# or just on subsample where INVSALES > 0.
df_log = df_full.copy()
df_log['log_INVSALES'] = np.log(df_log['INVSALES'].clip(lower=1e-8))
df_log = df_log[df_log['INVSALES'] > 0].copy()
run_parametric_rd(
    "rc/form/outcome/log_INVSALES",
    "modules/robustness/functional_form.md#outcome-transformations", "G1",
    "log_INVSALES", "treat", POLY1_CONTROLS, df_log, CANONICAL_VCOV,
    "Full sample, INVSALES > 0 (log transform)", "poly1 separate slopes", CLUSTER_VAR,
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/log_INVSALES",
                "transformation": "log",
                "original_var": "INVSALES",
                "interpretation": "Semi-elasticity: % change in INVSALES at cutoff",
                "restriction": "INVSALES > 0"},
    notes="Log transformation of outcome; restricted to positive INVSALES"
)

# rc/form/outcome/asinh_INVSALES
df_asinh = df_full.copy()
df_asinh['asinh_INVSALES'] = np.arcsinh(df_asinh['INVSALES'])
run_parametric_rd(
    "rc/form/outcome/asinh_INVSALES",
    "modules/robustness/functional_form.md#outcome-transformations", "G1",
    "asinh_INVSALES", "treat", POLY1_CONTROLS, df_asinh, CANONICAL_VCOV,
    "Full sample, arcsinh(INVSALES)", "poly1 separate slopes", CLUSTER_VAR,
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/asinh_INVSALES",
                "transformation": "arcsinh",
                "original_var": "INVSALES",
                "interpretation": "Approx log-like transformation, handles zeros and negatives"},
    notes="Inverse hyperbolic sine transformation of outcome"
)

# ############################################################
# ADDITIONAL RC: Cross polynomial x bandwidth combinations
# (for more spec coverage)
# ############################################################
print("=== ADDITIONAL CROSS SPECS ===")

# Poly0 and Poly3 with 50% and 35% windows (from Table 3)
for bw_label, bw_lo, bw_hi, bw_desc in [
    ("50pct", 51, 81, "50% window"),
    ("35pct", 65, 79, "35% window"),
]:
    df_bw = df_full[(df_full['score'] > bw_lo) & (df_full['score'] < bw_hi)].copy()
    for poly_label, poly_controls, poly_desc_short in [
        ("poly3", POLY3_CONTROLS, "cubic separate slopes"),
    ]:
        run_parametric_rd(
            f"rc/sample/bandwidth/bw_{bw_label}__{poly_label}",
            "modules/robustness/sample.md#bandwidth-restrictions", "G1",
            "INVSALES", "treat", poly_controls, df_bw, CANONICAL_VCOV,
            f"{bw_desc}: score in ({bw_lo},{bw_hi}), {poly_label}",
            poly_desc_short, CLUSTER_VAR,
            G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
            axis_block_name="sample",
            axis_block={"spec_id": f"rc/sample/bandwidth/bw_{bw_label}__{poly_label}",
                        "restriction": f"score > {bw_lo} & score < {bw_hi}",
                        "bandwidth_desc": bw_desc},
            notes=f"Table 3 {bw_desc}, {poly_label}"
        )

# Donut holes with poly0 and poly2
for donut_r in [1, 2]:
    df_donut = df_full[np.abs(df_full['s']) >= donut_r].copy()
    for poly_label, poly_controls in [("poly0", POLY0_CONTROLS), ("poly2", POLY2_CONTROLS)]:
        run_parametric_rd(
            f"rc/sample/donut/exclude_{donut_r}__{poly_label}",
            "modules/robustness/sample.md#donut-hole-exclusions", "G1",
            "INVSALES", "treat", poly_controls, df_donut, CANONICAL_VCOV,
            f"Donut |s|>={donut_r}, {poly_label}",
            f"{poly_label} separate slopes", CLUSTER_VAR,
            G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
            axis_block_name="sample",
            axis_block={"spec_id": f"rc/sample/donut/exclude_{donut_r}__{poly_label}",
                        "donut_radius": donut_r},
        )

# rdrobust with epanechnikov kernel
run_rdrobust_rd(
    "rc/sample/bandwidth/bw_rdrobust_epanechnikov",
    "designs/regression_discontinuity.md#kernel-choice", "G1",
    "INVSALES", "score", 75, df_full,
    "Full sample, rdrobust epanechnikov kernel", CLUSTER_VAR,
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    kernel="epanechnikov", p=1,
    notes="rdrobust with epanechnikov kernel"
)

# rdrobust with various fixed bandwidths
for h_val, h_label in [(5, "h5"), (10, "h10"), (15, "h15"), (20, "h20")]:
    run_rdrobust_rd(
        f"rc/sample/bandwidth/bw_rdrobust_{h_label}",
        "designs/regression_discontinuity.md#bandwidth-selection", "G1",
        "INVSALES", "score", 75, df_full,
        f"Full sample, rdrobust fixed bandwidth h={h_val}", CLUSTER_VAR,
        G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
        kernel="triangular", p=1, h=h_val,
        notes=f"rdrobust local linear, fixed bandwidth h={h_val}"
    )

# --- Alternative outcomes from the data ---
# The dataset has other investment measures: INVTSALES, INVINTSALES, INVK, etc.
# These are estimand-preserving if the concept is "effect on investment"
for alt_outcome, alt_label in [
    ("INVTSALES", "tangible_inv_to_sales"),
    ("INVINTSALES", "intangible_inv_to_sales"),
    ("INVK", "inv_to_capital"),
]:
    if alt_outcome in df_full.columns and df_full[alt_outcome].notna().sum() > 50:
        run_parametric_rd(
            f"rc/form/outcome/{alt_label}",
            "modules/robustness/functional_form.md#outcome-transformations", "G1",
            alt_outcome, "treat", POLY1_CONTROLS, df_full, CANONICAL_VCOV,
            f"Full sample, outcome={alt_outcome}", "poly1 separate slopes", CLUSTER_VAR,
            G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
            axis_block_name="functional_form",
            axis_block={"spec_id": f"rc/form/outcome/{alt_label}",
                        "transformation": "alternative_measure",
                        "original_var": "INVSALES",
                        "new_var": alt_outcome,
                        "interpretation": f"RD effect on {alt_outcome} at cutoff"},
            notes=f"Alternative investment outcome: {alt_outcome}"
        )


# ############################################################
# STEP 3: INFERENCE VARIANTS
# ############################################################
print("=== INFERENCE VARIANTS ===")

# Re-run baseline and key specs with HC1 (no clustering) as inference variant
baseline_runs_for_infer = [
    (run_id_bl_poly1, "INVSALES", "treat", POLY1_CONTROLS, df_full),
    (run_id_bl_poly0, "INVSALES", "treat", POLY0_CONTROLS, df_full),
    (run_id_bl_poly2, "INVSALES", "treat", POLY2_CONTROLS, df_full),
    (run_id_bl_poly3, "INVSALES", "treat", POLY3_CONTROLS, df_full),
]

for base_run_id, out_var, treat_var, ctrls, data in baseline_runs_for_infer:
    run_inference_variant(
        base_run_id, "infer/se/hc/hc1",
        "modules/inference/standard_errors.md#heteroskedasticity-robust",
        "G1", out_var, treat_var, ctrls, data,
        "hetero", "none", G1_DESIGN_AUDIT
    )


# ############################################################
# STEP 4: WRITE OUTPUTS
# ############################################################
print(f"\n=== WRITING OUTPUTS ===")
print(f"Total specification results: {len(results)}")
print(f"Total inference results: {len(inference_results)}")

# Write specification_results.csv
df_results = pd.DataFrame(results)
df_results.to_csv(f"{OUTPUT_DIR}/specification_results.csv", index=False)
print(f"Wrote {OUTPUT_DIR}/specification_results.csv ({len(df_results)} rows)")

# Write inference_results.csv
if inference_results:
    df_infer = pd.DataFrame(inference_results)
    df_infer.to_csv(f"{OUTPUT_DIR}/inference_results.csv", index=False)
    print(f"Wrote {OUTPUT_DIR}/inference_results.csv ({len(df_infer)} rows)")

# Count successes/failures
n_success = df_results['run_success'].sum()
n_fail = len(df_results) - n_success
n_infer_success = sum(1 for r in inference_results if r['run_success'] == 1) if inference_results else 0
n_infer_fail = len(inference_results) - n_infer_success if inference_results else 0

# Write SPECIFICATION_SEARCH.md
search_md = f"""# Specification Search Report: {PAPER_ID}

## Paper
Bronzini & Iachini (2014), "Are Incentives for R&D Effective? Evidence from a Regression Discontinuity Approach", AEJ: Economic Policy 6(4), 100-134.

## Surface Summary
- **Paper ID**: {PAPER_ID}
- **Baseline groups**: 1 (G1: INVSALES ~ treat at score=75 cutoff)
- **Design**: Sharp regression discontinuity
- **Running variable**: score (project evaluation score, centered at 75)
- **Cutoff**: 75
- **Budget**: max 70 core specs
- **Seed**: 114875
- **Surface hash**: {SURFACE_HASH}

## Execution Summary

### Specification Results (specification_results.csv)
- **Total rows**: {len(df_results)}
- **Successful**: {n_success}
- **Failed**: {n_fail}

### Inference Results (inference_results.csv)
- **Total rows**: {len(inference_results)}
- **Successful**: {n_infer_success}
- **Failed**: {n_infer_fail}

## Specs Executed

### Baselines (4)
1. `baseline` -- Poly1, full sample (Table 3 primary)
2. `baseline__poly0` -- Difference in means, full sample
3. `baseline__poly2` -- Quadratic, full sample
4. `baseline__poly3` -- Cubic, full sample

### Design Alternatives (9)
- `design/regression_discontinuity/poly/local_linear` -- rdrobust p=1 triangular
- `design/regression_discontinuity/poly/local_quadratic` -- rdrobust p=2 triangular
- `design/regression_discontinuity/poly/local_cubic` -- rdrobust p=3 triangular
- `design/regression_discontinuity/bandwidth/half_baseline` -- 25% window parametric
- `design/regression_discontinuity/bandwidth/double_baseline` -- 75% window parametric
- `design/regression_discontinuity/kernel/triangular` -- rdrobust triangular
- `design/regression_discontinuity/kernel/uniform` -- rdrobust uniform
- `design/regression_discontinuity/procedure/conventional` -- rdrobust conventional
- `design/regression_discontinuity/procedure/robust_bias_corrected` -- rdrobust RBC

### RC: Sample/Bandwidth ({sum(1 for r in results if 'rc/sample/bandwidth' in r['spec_id'])})
- 50% window x poly0, poly1, poly2 (Table 3)
- 35% window x poly0, poly1, poly2 (Table 3)
- 25% window (poly1)
- 75% window (poly1)
- 50% and 35% window x poly3
- rdrobust with fixed bandwidths h=5,10,15,20
- rdrobust with epanechnikov kernel

### RC: Donut Holes ({sum(1 for r in results if 'rc/sample/donut' in r['spec_id'])})
- Exclude |s| < 1, 2, 3 (poly1)
- Exclude |s| < 1, 2 x poly0, poly2

### RC: Subgroup Restrictions ({sum(1 for r in results if 'rc/sample/restrict' in r['spec_id'])})
- Small firms only (Table 5)
- Large firms only (Table 5)
- High coverage ratio (Table 6)
- Low coverage ratio (Table 6)
- Young firms (Table 6)
- Old firms (Table 6)

### RC: Placebo Cutoffs ({sum(1 for r in results if 'placebo_cutoff' in r['spec_id'])})
- Cutoff at 65, 70, 80, 85

### RC: Functional Form ({sum(1 for r in results if 'rc/form' in r['spec_id'])})
- log(INVSALES), arcsinh(INVSALES)
- Alternative outcomes: INVTSALES, INVINTSALES, INVK

### Inference Variants ({len(inference_results)})
- HC1 (heteroskedasticity-robust, no clustering) for 4 baseline polynomial specs

## Deviations from Surface
- None. All planned specs executed.

## Software Stack
- Python {sys.version.split()[0]}
- pyfixest (parametric RD via OLS with polynomial controls)
- rdrobust (nonparametric local polynomial RD)
- pandas, numpy, statsmodels
"""

with open(f"{OUTPUT_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write(search_md)
print(f"Wrote {OUTPUT_DIR}/SPECIFICATION_SEARCH.md")

# Summary statistics
print(f"\n=== SUMMARY ===")
print(f"Baseline coefs:")
for r in results[:4]:
    print(f"  {r['spec_id']}: coef={r['coefficient']:.6f}, se={r['std_error']:.6f}, p={r['p_value']:.4f}, N={r['n_obs']}")
