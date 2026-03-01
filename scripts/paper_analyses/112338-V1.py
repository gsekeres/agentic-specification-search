"""
Specification Search Script for Duggan & Scott Morton (2010)
"The Effect of Medicare Part D on Pharmaceutical Prices and Utilization"
American Economic Journal: Economic Policy

Paper ID: 112338-V1

The IMS Health sales/dose data (ims0106data2.dta, ims0106all.dta) is proprietary
and NOT included in the replication package. Outcome variables (lppd0603, ldoses0603,
etc.) cannot be constructed without this data.

Strategy: Extract all available regression coefficients from the Stata log file
(regs-partd-final.log) for specifications that match the surface. Specifications
that require new regressions (not in the paper's tables) are recorded as failures
with reason "proprietary IMS data unavailable."

Outputs:
  - specification_results.csv (core specs: baseline, rc/*)
  - inference_results.csv (inference variants: infer/*)
  - SPECIFICATION_SEARCH.md (run log)
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
PAPER_ID = "112338-V1"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PACKAGE_DIR = os.path.join(REPO_ROOT, "data", "downloads", "extracted", PAPER_ID)
DATA_DIR = os.path.join(PACKAGE_DIR, "data")
SURFACE_FILE = os.path.join(PACKAGE_DIR, "SPECIFICATION_SURFACE.json")
LOG_FILE = os.path.join(DATA_DIR, "regs-partd-final.log")

# Import helper utilities
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash as compute_surface_hash,
    software_block
)

# =============================================================================
# Load surface and compute hash
# =============================================================================
with open(SURFACE_FILE, 'r') as f:
    surface = json.load(f)
SURFACE_HASH = compute_surface_hash(surface)
SOFTWARE = software_block()

# Canonical inference for both groups
CANONICAL_INFERENCE_G1 = {
    "spec_id": "infer/se/hc/hc1",
    "params": {},
    "notes": "Robust (HC1) standard errors, matching paper's `reg ..., robust`."
}
CANONICAL_INFERENCE_G2 = CANONICAL_INFERENCE_G1.copy()

# Design audit -- copied verbatim from surface design_audit for each baseline group
DESIGN_AUDIT_G1 = {
    "cross_sectional_ols": surface["baseline_groups"][0]["design_audit"].copy()
}
DESIGN_AUDIT_G2 = {
    "cross_sectional_ols": surface["baseline_groups"][1]["design_audit"].copy()
}

DATA_UNAVAILABLE_MSG = "Proprietary IMS Health sales/dose data not available in replication package; outcome variables cannot be constructed"

# =============================================================================
# Result accumulators
# =============================================================================
spec_results = []
inference_results = []
spec_counter = [0]
inference_counter = [0]


def next_spec_run_id():
    spec_counter[0] += 1
    return f"{PAPER_ID}_spec_{spec_counter[0]:03d}"


def next_inference_run_id():
    inference_counter[0] += 1
    return f"{PAPER_ID}_infer_{inference_counter[0]:03d}"


def add_success_row(spec_id, spec_tree_path, baseline_group_id,
                    outcome_var, treatment_var, coefficient, std_error, p_value,
                    ci_lower, ci_upper, n_obs, r_squared,
                    sample_desc, fixed_effects, controls_desc, cluster_var,
                    coef_vector_json, weights_desc="meps0203scripts"):
    """Add a successful specification result row."""
    spec_results.append({
        "paper_id": PAPER_ID,
        "spec_run_id": next_spec_run_id(),
        "spec_id": spec_id,
        "spec_tree_path": spec_tree_path,
        "baseline_group_id": baseline_group_id,
        "outcome_var": outcome_var,
        "treatment_var": treatment_var,
        "coefficient": coefficient,
        "std_error": std_error,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_obs": n_obs,
        "r_squared": r_squared,
        "coefficient_vector_json": json.dumps(coef_vector_json),
        "sample_desc": sample_desc,
        "fixed_effects": fixed_effects,
        "controls_desc": controls_desc,
        "cluster_var": cluster_var,
        "run_success": 1,
        "run_error": ""
    })


def add_failure_row(spec_id, spec_tree_path, baseline_group_id,
                    outcome_var, treatment_var, error_msg,
                    sample_desc="", fixed_effects="", controls_desc="",
                    cluster_var=""):
    """Add a failed specification result row."""
    err_payload = make_failure_payload(
        error=error_msg,
        error_details={
            "stage": "data_construction",
            "exception_type": "DataUnavailableError",
            "exception_message": error_msg
        },
        software=SOFTWARE,
        surface_hash=SURFACE_HASH
    )
    spec_results.append({
        "paper_id": PAPER_ID,
        "spec_run_id": next_spec_run_id(),
        "spec_id": spec_id,
        "spec_tree_path": spec_tree_path,
        "baseline_group_id": baseline_group_id,
        "outcome_var": outcome_var,
        "treatment_var": treatment_var,
        "coefficient": np.nan,
        "std_error": np.nan,
        "p_value": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "n_obs": np.nan,
        "r_squared": np.nan,
        "coefficient_vector_json": json.dumps(err_payload),
        "sample_desc": sample_desc,
        "fixed_effects": fixed_effects,
        "controls_desc": controls_desc,
        "cluster_var": cluster_var,
        "run_success": 0,
        "run_error": error_msg
    })


def add_inference_row(base_spec_run_id, spec_id, spec_tree_path,
                      baseline_group_id, coefficient, std_error, p_value,
                      ci_lower, ci_upper, n_obs, r_squared,
                      coef_vector_json, outcome_var="", treatment_var="",
                      cluster_var=""):
    """Add an inference variant result row."""
    inference_results.append({
        "paper_id": PAPER_ID,
        "inference_run_id": next_inference_run_id(),
        "spec_run_id": base_spec_run_id,
        "spec_id": spec_id,
        "spec_tree_path": spec_tree_path,
        "baseline_group_id": baseline_group_id,
        "outcome_var": outcome_var,
        "treatment_var": treatment_var,
        "coefficient": coefficient,
        "std_error": std_error,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_obs": n_obs,
        "r_squared": r_squared,
        "coefficient_vector_json": json.dumps(coef_vector_json),
        "cluster_var": cluster_var,
        "run_success": 1,
        "run_error": ""
    })


def add_inference_failure(base_spec_run_id, spec_id, spec_tree_path,
                          baseline_group_id, error_msg,
                          outcome_var="", treatment_var="", cluster_var=""):
    """Add a failed inference variant."""
    err_payload = make_failure_payload(
        error=error_msg,
        error_details={
            "stage": "data_construction",
            "exception_type": "DataUnavailableError",
            "exception_message": error_msg
        },
        software=SOFTWARE,
        surface_hash=SURFACE_HASH
    )
    inference_results.append({
        "paper_id": PAPER_ID,
        "inference_run_id": next_inference_run_id(),
        "spec_run_id": base_spec_run_id,
        "spec_id": spec_id,
        "spec_tree_path": spec_tree_path,
        "baseline_group_id": baseline_group_id,
        "outcome_var": outcome_var,
        "treatment_var": treatment_var,
        "coefficient": np.nan,
        "std_error": np.nan,
        "p_value": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "n_obs": np.nan,
        "r_squared": np.nan,
        "coefficient_vector_json": json.dumps(err_payload),
        "cluster_var": cluster_var,
        "run_success": 0,
        "run_error": error_msg
    })


def build_log_payload(coefficients_dict, inference_dict, design_audit,
                      extra_blocks=None):
    """Build a contract-compliant coefficient_vector_json from log-extracted data."""
    payload = make_success_payload(
        coefficients=coefficients_dict,
        inference=inference_dict,
        software=SOFTWARE,
        surface_hash=SURFACE_HASH,
        design=design_audit,
        extra={"source": "stata_log_extraction",
               "note": "Coefficients extracted from regs-partd-final.log; IMS data proprietary"}
    )
    if extra_blocks:
        for k, v in extra_blocks.items():
            payload[k] = v
    return payload


# =============================================================================
# G1: Price Effects (Table 2) -- Log-extracted specifications
# =============================================================================
print("=" * 60)
print("G1: Price Effects (Table 2)")
print("=" * 60)

G1_SAMPLE_BASE = "imsgrouprank03<=1000, otc==2, imsgbo==2, fdayear<=2003, smallcat!=."
G1_SAMPLE_TRIM = G1_SAMPLE_BASE + ", lppd0603 in [-1.1, 1.095]"
G1_SAMPLE_TRIM_NOCANCER = G1_SAMPLE_TRIM + ", thercat!=8"

# --- G1 Baseline: Table 2 Col 4 (preferred spec) ---
print("  Running baseline (Table 2 Col 4)...")
add_success_row(
    spec_id="baseline",
    spec_tree_path="specification_tree/designs/cross_sectional_ols.md",
    baseline_group_id="G1",
    outcome_var="lppd0603",
    treatment_var="mcar0203mepsrx",
    coefficient=-0.1363694,
    std_error=0.0561339,
    p_value=0.015,
    ci_lower=-0.2466499,
    ci_upper=-0.0260888,
    n_obs=517,
    r_squared=0.0235,
    sample_desc=G1_SAMPLE_TRIM_NOCANCER,
    fixed_effects="",
    controls_desc="yrs03onmkt, anygen",
    cluster_var="",
    coef_vector_json=build_log_payload(
        coefficients_dict={"mcar0203mepsrx": -0.1363694, "yrs03onmkt": 0.0006297,
                           "anygen": 0.0104217, "_cons": 0.126273},
        inference_dict=CANONICAL_INFERENCE_G1,
        design_audit=DESIGN_AUDIT_G1
    )
)
G1_BASELINE_RUN_ID = spec_results[-1]["spec_run_id"]

# --- G1 Additional baselines ---
# Table 2 Col 1: bivariate (no controls, no trim, no cancer exclusion)
print("  Running baseline Table 2 Col 1...")
add_success_row(
    spec_id="baseline__table2_col1",
    spec_tree_path="specification_tree/designs/cross_sectional_ols.md",
    baseline_group_id="G1",
    outcome_var="lppd0603",
    treatment_var="mcar0203mepsrx",
    coefficient=-0.1278322,
    std_error=0.0572712,
    p_value=0.026,
    ci_lower=-0.2403311,
    ci_upper=-0.0153334,
    n_obs=548,
    r_squared=0.0160,
    sample_desc=G1_SAMPLE_BASE,
    fixed_effects="",
    controls_desc="",
    cluster_var="",
    coef_vector_json=build_log_payload(
        coefficients_dict={"mcar0203mepsrx": -0.1278322, "_cons": 0.1336053},
        inference_dict=CANONICAL_INFERENCE_G1,
        design_audit=DESIGN_AUDIT_G1,
        extra_blocks={"controls": {"spec_id": "baseline__table2_col1",
                                   "family": "none", "n_controls": 0}}
    )
)

# Table 2 Col 2: add controls, no trim, no cancer exclusion
print("  Running baseline Table 2 Col 2...")
add_success_row(
    spec_id="baseline__table2_col2",
    spec_tree_path="specification_tree/designs/cross_sectional_ols.md",
    baseline_group_id="G1",
    outcome_var="lppd0603",
    treatment_var="mcar0203mepsrx",
    coefficient=-0.1271804,
    std_error=0.0570665,
    p_value=0.026,
    ci_lower=-0.2392782,
    ci_upper=-0.0150827,
    n_obs=548,
    r_squared=0.0163,
    sample_desc=G1_SAMPLE_BASE,
    fixed_effects="",
    controls_desc="yrs03onmkt, anygen",
    cluster_var="",
    coef_vector_json=build_log_payload(
        coefficients_dict={"mcar0203mepsrx": -0.1271804, "yrs03onmkt": 0.0003043,
                           "anygen": 0.0049368, "_cons": 0.1278569},
        inference_dict=CANONICAL_INFERENCE_G1,
        design_audit=DESIGN_AUDIT_G1
    )
)

# Table 2 Col 3: add trim, no cancer exclusion
print("  Running baseline Table 2 Col 3...")
add_success_row(
    spec_id="baseline__table2_col3",
    spec_tree_path="specification_tree/designs/cross_sectional_ols.md",
    baseline_group_id="G1",
    outcome_var="lppd0603",
    treatment_var="mcar0203mepsrx",
    coefficient=-0.1376287,
    std_error=0.0556076,
    p_value=0.014,
    ci_lower=-0.2468653,
    ci_upper=-0.0283922,
    n_obs=538,
    r_squared=0.0239,
    sample_desc=G1_SAMPLE_TRIM,
    fixed_effects="",
    controls_desc="yrs03onmkt, anygen",
    cluster_var="",
    coef_vector_json=build_log_payload(
        coefficients_dict={"mcar0203mepsrx": -0.1376287, "yrs03onmkt": 0.0006061,
                           "anygen": 0.0103332, "_cons": 0.1270293},
        inference_dict=CANONICAL_INFERENCE_G1,
        design_audit=DESIGN_AUDIT_G1,
        extra_blocks={"sample": {"spec_id": "baseline__table2_col3",
                                 "trim": "[-1.1, 1.095]", "exclude_cancer": False}}
    )
)

# =============================================================================
# G1 RC specs available from the log
# =============================================================================

# rc/controls/loo/drop_yrs03onmkt -- same as baseline but no yrs03onmkt
# NOT in the log; Col 1 has no controls at all. Must run from data => failure.
print("  Recording rc/controls/loo/drop_yrs03onmkt (data unavailable)...")
add_failure_row(
    spec_id="rc/controls/loo/drop_yrs03onmkt",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G1",
    outcome_var="lppd0603",
    treatment_var="mcar0203mepsrx",
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc=G1_SAMPLE_TRIM_NOCANCER,
    controls_desc="anygen (drop yrs03onmkt)"
)

# rc/controls/loo/drop_anygen
print("  Recording rc/controls/loo/drop_anygen (data unavailable)...")
add_failure_row(
    spec_id="rc/controls/loo/drop_anygen",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G1",
    outcome_var="lppd0603",
    treatment_var="mcar0203mepsrx",
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc=G1_SAMPLE_TRIM_NOCANCER,
    controls_desc="yrs03onmkt (drop anygen)"
)

# rc/controls/single/add_protected -- Table 5 Col 1 has protected + mcar0203prot
# But the surface wants just adding protected as a control (no interaction).
# Not in the log as-is => failure.
print("  Recording rc/controls/single/add_protected (data unavailable)...")
add_failure_row(
    spec_id="rc/controls/single/add_protected",
    spec_tree_path="specification_tree/modules/robustness/controls.md#single",
    baseline_group_id="G1",
    outcome_var="lppd0603",
    treatment_var="mcar0203mepsrx",
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc=G1_SAMPLE_TRIM_NOCANCER,
    controls_desc="yrs03onmkt, anygen, protected"
)

# rc/controls/single/add_smallcat
print("  Recording rc/controls/single/add_smallcat (data unavailable)...")
add_failure_row(
    spec_id="rc/controls/single/add_smallcat",
    spec_tree_path="specification_tree/modules/robustness/controls.md#single",
    baseline_group_id="G1",
    outcome_var="lppd0603",
    treatment_var="mcar0203mepsrx",
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc=G1_SAMPLE_TRIM_NOCANCER,
    controls_desc="yrs03onmkt, anygen, smallcat"
)

# rc/controls/single/add_lppd0201 (lagged price change)
print("  Recording rc/controls/single/add_lppd0201 (data unavailable)...")
add_failure_row(
    spec_id="rc/controls/single/add_lppd0201",
    spec_tree_path="specification_tree/modules/robustness/controls.md#single",
    baseline_group_id="G1",
    outcome_var="lppd0603",
    treatment_var="mcar0203mepsrx",
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc=G1_SAMPLE_TRIM_NOCANCER,
    controls_desc="yrs03onmkt, anygen, lppd0201"
)

# rc/controls/sets/interactions_protected -- Table 5 Col 1
# This spec adds protected + mcar0203prot (interaction). Log has it with cluster(ther1).
# The spec as an RC in G1 should use HC1 (canonical). The log uses cluster.
# We extract the coefficients from Table 5 Col 1 but note the inference is clustered.
# For the estimate row in spec_results, we need canonical inference.
# Since we can't re-run with HC1, record as failure for the canonical inference version.
print("  Recording rc/controls/sets/interactions_protected (data unavailable for HC1)...")
add_failure_row(
    spec_id="rc/controls/sets/interactions_protected",
    spec_tree_path="specification_tree/modules/robustness/controls.md#sets",
    baseline_group_id="G1",
    outcome_var="lppd0603",
    treatment_var="mcar0203mepsrx",
    error_msg=DATA_UNAVAILABLE_MSG + "; Table 5 Col 1 uses cluster(ther1), not HC1",
    sample_desc=G1_SAMPLE_TRIM_NOCANCER,
    controls_desc="yrs03onmkt, anygen, protected, mcar0203prot"
)

# rc/controls/sets/interactions_smallcat -- Table 5 Col 2
print("  Recording rc/controls/sets/interactions_smallcat (data unavailable for HC1)...")
add_failure_row(
    spec_id="rc/controls/sets/interactions_smallcat",
    spec_tree_path="specification_tree/modules/robustness/controls.md#sets",
    baseline_group_id="G1",
    outcome_var="lppd0603",
    treatment_var="mcar0203mepsrx",
    error_msg=DATA_UNAVAILABLE_MSG + "; Table 5 Col 2 uses cluster(ther1), not HC1",
    sample_desc=G1_SAMPLE_TRIM_NOCANCER,
    controls_desc="yrs03onmkt, anygen, smallcat, scmcar0203"
)

# rc/controls/sets/interactions_both -- Table 5 Col 3
print("  Recording rc/controls/sets/interactions_both (data unavailable for HC1)...")
add_failure_row(
    spec_id="rc/controls/sets/interactions_both",
    spec_tree_path="specification_tree/modules/robustness/controls.md#sets",
    baseline_group_id="G1",
    outcome_var="lppd0603",
    treatment_var="mcar0203mepsrx",
    error_msg=DATA_UNAVAILABLE_MSG + "; Table 5 Col 3 uses cluster(ther1), not HC1",
    sample_desc=G1_SAMPLE_TRIM_NOCANCER,
    controls_desc="yrs03onmkt, anygen, smallcat, scmcar0203, protected, mcar0203prot"
)

# rc/controls/sets/full_with_lagged
print("  Recording rc/controls/sets/full_with_lagged (data unavailable)...")
add_failure_row(
    spec_id="rc/controls/sets/full_with_lagged",
    spec_tree_path="specification_tree/modules/robustness/controls.md#sets",
    baseline_group_id="G1",
    outcome_var="lppd0603",
    treatment_var="mcar0203mepsrx",
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc=G1_SAMPLE_TRIM_NOCANCER,
    controls_desc="yrs03onmkt, anygen, protected, mcar0203prot, smallcat, scmcar0203, lppd0201"
)

# rc/sample/outliers/no_trim -- Table 2 Col 2 (controls, no trim, no cancer excl)
# But actually Col 2 also doesn't exclude cancer. The surface wants no-trim + no-cancer.
# Col 2 is: controls + no trim + no cancer exclusion. That's close but we need
# the combination with cancer exclusion to be the RC variant (no_trim but yes cancer excl).
# This exact combo is NOT in the log. Record as failure.
print("  Recording rc/sample/outliers/no_trim (data unavailable)...")
add_failure_row(
    spec_id="rc/sample/outliers/no_trim",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    baseline_group_id="G1",
    outcome_var="lppd0603",
    treatment_var="mcar0203mepsrx",
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc=G1_SAMPLE_BASE + ", thercat!=8 (no outlier trim)",
    controls_desc="yrs03onmkt, anygen"
)

# rc/sample/outliers/trim_y_5_95
print("  Recording rc/sample/outliers/trim_y_5_95 (data unavailable)...")
add_failure_row(
    spec_id="rc/sample/outliers/trim_y_5_95",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    baseline_group_id="G1",
    outcome_var="lppd0603",
    treatment_var="mcar0203mepsrx",
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc=G1_SAMPLE_BASE + ", thercat!=8, lppd0603 trimmed at 5th/95th percentile"
)

# rc/sample/outliers/trim_y_2_98
print("  Recording rc/sample/outliers/trim_y_2_98 (data unavailable)...")
add_failure_row(
    spec_id="rc/sample/outliers/trim_y_2_98",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    baseline_group_id="G1",
    outcome_var="lppd0603",
    treatment_var="mcar0203mepsrx",
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc=G1_SAMPLE_BASE + ", thercat!=8, lppd0603 trimmed at 2nd/98th percentile"
)

# rc/sample/subset/include_cancer -- Table 2 Col 3 (trim, include cancer)
print("  Running rc/sample/subset/include_cancer (from Table 2 Col 3)...")
add_success_row(
    spec_id="rc/sample/subset/include_cancer",
    spec_tree_path="specification_tree/modules/robustness/sample.md#subset",
    baseline_group_id="G1",
    outcome_var="lppd0603",
    treatment_var="mcar0203mepsrx",
    coefficient=-0.1376287,
    std_error=0.0556076,
    p_value=0.014,
    ci_lower=-0.2468653,
    ci_upper=-0.0283922,
    n_obs=538,
    r_squared=0.0239,
    sample_desc=G1_SAMPLE_TRIM + " (cancer drugs included)",
    fixed_effects="",
    controls_desc="yrs03onmkt, anygen",
    cluster_var="",
    coef_vector_json=build_log_payload(
        coefficients_dict={"mcar0203mepsrx": -0.1376287, "yrs03onmkt": 0.0006061,
                           "anygen": 0.0103332, "_cons": 0.1270293},
        inference_dict=CANONICAL_INFERENCE_G1,
        design_audit=DESIGN_AUDIT_G1,
        extra_blocks={"sample": {"spec_id": "rc/sample/subset/include_cancer",
                                 "family": "subset", "cancer_included": True}}
    )
)

# rc/sample/subset/top292 -- Table 2 Col 6
print("  Running rc/sample/subset/top292 (from Table 2 Col 6)...")
add_success_row(
    spec_id="rc/sample/subset/top292",
    spec_tree_path="specification_tree/modules/robustness/sample.md#subset",
    baseline_group_id="G1",
    outcome_var="lppd0603",
    treatment_var="mcar0203mepsrx",
    coefficient=-0.1427032,
    std_error=0.0544166,
    p_value=0.009,
    ci_lower=-0.2500204,
    ci_upper=-0.0353861,
    n_obs=200,
    r_squared=0.0486,
    sample_desc="imsgrouprank03<=292, otc==2, imsgbo==2, fdayear<=2003, smallcat!=., lppd0603 in [-1.1, 1.095], thercat!=8",
    fixed_effects="",
    controls_desc="yrs03onmkt, anygen",
    cluster_var="",
    coef_vector_json=build_log_payload(
        coefficients_dict={"mcar0203mepsrx": -0.1427032, "yrs03onmkt": 0.0020975,
                           "anygen": -0.0076148, "_cons": 0.1269107},
        inference_dict=CANONICAL_INFERENCE_G1,
        design_audit=DESIGN_AUDIT_G1,
        extra_blocks={"sample": {"spec_id": "rc/sample/subset/top292",
                                 "family": "subset", "rank_cutoff": 292}}
    )
)

# rc/sample/subset/top300 -- not exactly in the log, record as failure
print("  Recording rc/sample/subset/top300 (data unavailable)...")
add_failure_row(
    spec_id="rc/sample/subset/top300",
    spec_tree_path="specification_tree/modules/robustness/sample.md#subset",
    baseline_group_id="G1",
    outcome_var="lppd0603",
    treatment_var="mcar0203mepsrx",
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="imsgrouprank03<=300 + baseline restrictions"
)

# rc/sample/subset/top500
print("  Recording rc/sample/subset/top500 (data unavailable)...")
add_failure_row(
    spec_id="rc/sample/subset/top500",
    spec_tree_path="specification_tree/modules/robustness/sample.md#subset",
    baseline_group_id="G1",
    outcome_var="lppd0603",
    treatment_var="mcar0203mepsrx",
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="imsgrouprank03<=500 + baseline restrictions"
)

# rc/sample/subset/drop_new_generic
print("  Recording rc/sample/subset/drop_new_generic (data unavailable)...")
add_failure_row(
    spec_id="rc/sample/subset/drop_new_generic",
    spec_tree_path="specification_tree/modules/robustness/sample.md#subset",
    baseline_group_id="G1",
    outcome_var="lppd0603",
    treatment_var="mcar0203mepsrx",
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc=G1_SAMPLE_TRIM_NOCANCER + ", drop drugs with new generic entry"
)

# rc/sample/subset/no_otc_filter
print("  Recording rc/sample/subset/no_otc_filter (data unavailable)...")
add_failure_row(
    spec_id="rc/sample/subset/no_otc_filter",
    spec_tree_path="specification_tree/modules/robustness/sample.md#subset",
    baseline_group_id="G1",
    outcome_var="lppd0603",
    treatment_var="mcar0203mepsrx",
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="No OTC filter, otherwise baseline restrictions"
)

# rc/form/treatment/spending_share -- Table 2 Col 5 (mcar0203mepspd)
print("  Running rc/form/treatment/spending_share (from Table 2 Col 5)...")
add_success_row(
    spec_id="rc/form/treatment/spending_share",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    baseline_group_id="G1",
    outcome_var="lppd0603",
    treatment_var="mcar0203mepspd",
    coefficient=-0.1329132,
    std_error=0.0562365,
    p_value=0.018,
    ci_lower=-0.2433953,
    ci_upper=-0.022431,
    n_obs=517,
    r_squared=0.0230,
    sample_desc=G1_SAMPLE_TRIM_NOCANCER,
    fixed_effects="",
    controls_desc="yrs03onmkt, anygen",
    cluster_var="",
    coef_vector_json=build_log_payload(
        coefficients_dict={"mcar0203mepspd": -0.1329132, "yrs03onmkt": 0.0006995,
                           "anygen": 0.010523, "_cons": 0.1251799},
        inference_dict=CANONICAL_INFERENCE_G1,
        design_audit=DESIGN_AUDIT_G1,
        extra_blocks={"functional_form": {
            "spec_id": "rc/form/treatment/spending_share",
            "interpretation": "Medicare spending share instead of prescription share as treatment",
            "treatment_definition": "mcar0203mepspd (Medicare share of drug spending, pooled 02-03)"
        }}
    )
)

# rc/form/treatment/self_pay_decomposed -- Table 4 Col 2
print("  Running rc/form/treatment/self_pay_decomposed (from Table 4 Col 2)...")
add_success_row(
    spec_id="rc/form/treatment/self_pay_decomposed",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    baseline_group_id="G1",
    outcome_var="lppd0603",
    treatment_var="mcself0203mepsrx",
    coefficient=-0.2252987,
    std_error=0.070014,
    p_value=0.001,
    ci_lower=-0.3628488,
    ci_upper=-0.0877486,
    n_obs=517,
    r_squared=0.0300,
    sample_desc=G1_SAMPLE_TRIM_NOCANCER,
    fixed_effects="",
    controls_desc="yrs03onmkt, anygen, mcoth0203mepsrx",
    cluster_var="",
    coef_vector_json=build_log_payload(
        coefficients_dict={"mcself0203mepsrx": -0.2252987, "mcoth0203mepsrx": 0.0593237,
                           "yrs03onmkt": 0.0009996, "anygen": 0.0112921, "_cons": 0.1144759},
        inference_dict=CANONICAL_INFERENCE_G1,
        design_audit=DESIGN_AUDIT_G1,
        extra_blocks={"functional_form": {
            "spec_id": "rc/form/treatment/self_pay_decomposed",
            "interpretation": "Decompose Medicare share into self-pay and other-Medicare components",
            "focal_treatment": "mcself0203mepsrx"
        }}
    )
)

# rc/form/treatment/dual_decomposed -- Table 4 Col 3
print("  Running rc/form/treatment/dual_decomposed (from Table 4 Col 3)...")
add_success_row(
    spec_id="rc/form/treatment/dual_decomposed",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    baseline_group_id="G1",
    outcome_var="lppd0603",
    treatment_var="mcself0203mepsrx",
    coefficient=-0.2428983,
    std_error=0.077271,
    p_value=0.002,
    ci_lower=-0.3947062,
    ci_upper=-0.0910903,
    n_obs=517,
    r_squared=0.0331,
    sample_desc=G1_SAMPLE_TRIM_NOCANCER,
    fixed_effects="",
    controls_desc="yrs03onmkt, anygen, mcoth0203mepsrx2, dual0203mepsrx",
    cluster_var="",
    coef_vector_json=build_log_payload(
        coefficients_dict={"mcself0203mepsrx": -0.2428983, "mcoth0203mepsrx2": 0.1898703,
                           "dual0203mepsrx": -0.1823221, "yrs03onmkt": 0.0009956,
                           "anygen": 0.0095834, "_cons": 0.1176387},
        inference_dict=CANONICAL_INFERENCE_G1,
        design_audit=DESIGN_AUDIT_G1,
        extra_blocks={"functional_form": {
            "spec_id": "rc/form/treatment/dual_decomposed",
            "interpretation": "Decompose Medicare share into self-pay, dual-eligible, and other components",
            "focal_treatment": "mcself0203mepsrx"
        }}
    )
)

# rc/form/outcome/level_price_change (ppd0603 instead of lppd0603)
print("  Recording rc/form/outcome/level_price_change (data unavailable)...")
add_failure_row(
    spec_id="rc/form/outcome/level_price_change",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    baseline_group_id="G1",
    outcome_var="ppd0603",
    treatment_var="mcar0203mepsrx",
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc=G1_SAMPLE_TRIM_NOCANCER,
    controls_desc="yrs03onmkt, anygen"
)

# explore/outcome/log_sales_change -- Table 4 Col 7
print("  Running explore/outcome/log_sales_change (from Table 4 Col 7)...")
add_success_row(
    spec_id="explore/outcome/log_sales_change",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    baseline_group_id="G1",
    outcome_var="lsalesq0603",
    treatment_var="mcar0203mepsrx",
    coefficient=0.2727952,
    std_error=0.3136442,
    p_value=0.385,
    ci_lower=-0.34339,
    ci_upper=0.8889803,
    n_obs=517,
    r_squared=0.2925,
    sample_desc="imsgrouprank03<=1000, otc==2, imsgbo==2, fdayear<=2003, smallcat!=., lsalesq0603 in [-3.7, 1.72], thercat!=8",
    fixed_effects="",
    controls_desc="yrs03onmkt, anygen",
    cluster_var="",
    coef_vector_json=build_log_payload(
        coefficients_dict={"mcar0203mepsrx": 0.2727952, "yrs03onmkt": -0.0018285,
                           "anygen": -1.071165, "_cons": -0.1073883},
        inference_dict=CANONICAL_INFERENCE_G1,
        design_audit=DESIGN_AUDIT_G1,
        extra_blocks={"functional_form": {
            "spec_id": "explore/outcome/log_sales_change",
            "interpretation": "Log change in sales revenue (combines price and quantity effects)",
            "outcome_definition": "lsalesq0603 = log(salesq2006 / (CPI_adj * salesq2003))"
        }}
    )
)

# explore/outcome/log_doses_change -- this is G2's primary outcome used as G1 explore
print("  Recording explore/outcome/log_doses_change (data unavailable)...")
add_failure_row(
    spec_id="explore/outcome/log_doses_change",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    baseline_group_id="G1",
    outcome_var="ldoses0603",
    treatment_var="mcar0203mepsrx",
    error_msg=DATA_UNAVAILABLE_MSG + "; Table 3 Col 4 provides this but with different trim bounds",
    sample_desc=G1_SAMPLE_TRIM_NOCANCER + " (with ldoses0603 trim bounds)"
)

# rc/weights/unweighted
print("  Recording rc/weights/unweighted (data unavailable)...")
add_failure_row(
    spec_id="rc/weights/unweighted",
    spec_tree_path="specification_tree/modules/robustness/weights.md",
    baseline_group_id="G1",
    outcome_var="lppd0603",
    treatment_var="mcar0203mepsrx",
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc=G1_SAMPLE_TRIM_NOCANCER,
    controls_desc="yrs03onmkt, anygen"
)

# rc/weights/alt_02plus03_scripts
print("  Recording rc/weights/alt_02plus03_scripts (data unavailable)...")
add_failure_row(
    spec_id="rc/weights/alt_02plus03_scripts",
    spec_tree_path="specification_tree/modules/robustness/weights.md",
    baseline_group_id="G1",
    outcome_var="lppd0603",
    treatment_var="mcar0203mepsrx",
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc=G1_SAMPLE_TRIM_NOCANCER,
    controls_desc="yrs03onmkt, anygen"
)

# rc/fe/add_therapeutic_category
print("  Recording rc/fe/add_therapeutic_category (data unavailable)...")
add_failure_row(
    spec_id="rc/fe/add_therapeutic_category",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md",
    baseline_group_id="G1",
    outcome_var="lppd0603",
    treatment_var="mcar0203mepsrx",
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc=G1_SAMPLE_TRIM_NOCANCER,
    fixed_effects="ther1",
    controls_desc="yrs03onmkt, anygen"
)

# rc/joint/no_trim_include_cancer
print("  Running rc/joint/no_trim_include_cancer (from Table 2 Col 2)...")
# Table 2 Col 2 = controls + no trim + including cancer
add_success_row(
    spec_id="rc/joint/no_trim_include_cancer",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="lppd0603",
    treatment_var="mcar0203mepsrx",
    coefficient=-0.1271804,
    std_error=0.0570665,
    p_value=0.026,
    ci_lower=-0.2392782,
    ci_upper=-0.0150827,
    n_obs=548,
    r_squared=0.0163,
    sample_desc=G1_SAMPLE_BASE + " (no trim, cancer included)",
    fixed_effects="",
    controls_desc="yrs03onmkt, anygen",
    cluster_var="",
    coef_vector_json=build_log_payload(
        coefficients_dict={"mcar0203mepsrx": -0.1271804, "yrs03onmkt": 0.0003043,
                           "anygen": 0.0049368, "_cons": 0.1278569},
        inference_dict=CANONICAL_INFERENCE_G1,
        design_audit=DESIGN_AUDIT_G1,
        extra_blocks={"joint": {
            "spec_id": "rc/joint/no_trim_include_cancer",
            "components": ["no_trim", "include_cancer"]
        }}
    )
)

# rc/joint/top292_with_interactions
print("  Recording rc/joint/top292_with_interactions (data unavailable)...")
add_failure_row(
    spec_id="rc/joint/top292_with_interactions",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="lppd0603",
    treatment_var="mcar0203mepsrx",
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="imsgrouprank03<=292 + baseline trim + thercat!=8"
)

# rc/joint/spending_share_no_cancer
print("  Recording rc/joint/spending_share_no_cancer (Table 2 Col 5 is this)...")
# Table 2 Col 5 IS spending share + no cancer + trim => exactly this spec
add_success_row(
    spec_id="rc/joint/spending_share_no_cancer",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="lppd0603",
    treatment_var="mcar0203mepspd",
    coefficient=-0.1329132,
    std_error=0.0562365,
    p_value=0.018,
    ci_lower=-0.2433953,
    ci_upper=-0.022431,
    n_obs=517,
    r_squared=0.0230,
    sample_desc=G1_SAMPLE_TRIM_NOCANCER,
    fixed_effects="",
    controls_desc="yrs03onmkt, anygen",
    cluster_var="",
    coef_vector_json=build_log_payload(
        coefficients_dict={"mcar0203mepspd": -0.1329132, "yrs03onmkt": 0.0006995,
                           "anygen": 0.010523, "_cons": 0.1251799},
        inference_dict=CANONICAL_INFERENCE_G1,
        design_audit=DESIGN_AUDIT_G1,
        extra_blocks={"joint": {
            "spec_id": "rc/joint/spending_share_no_cancer",
            "components": ["spending_share", "exclude_cancer"]
        }}
    )
)

# rc/joint/unweighted_no_trim
print("  Recording rc/joint/unweighted_no_trim (data unavailable)...")
add_failure_row(
    spec_id="rc/joint/unweighted_no_trim",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="lppd0603",
    treatment_var="mcar0203mepsrx",
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="unweighted, no outlier trim"
)


# =============================================================================
# G1 Inference variants (for baseline spec)
# =============================================================================
print("\n  G1 Inference variants...")

# The Table 5 regressions use cluster(ther1), which gives us the clustered SE
# for the SAME baseline sample/controls. Table 5 Col 1 adds protected+interaction
# which is NOT the baseline spec. However, Table 4 Col 1 re-runs the exact baseline.
# Table 4 Col 1 = same as Table 2 Col 4 (same coefficients).
# Table 5 Col 1 = different controls (adds protected, mcar0203prot).
# So we do NOT have a cluster(ther1) version of the exact baseline (just mcar+yrs+anygen).
# Record as failures since data not available.

add_inference_failure(
    base_spec_run_id=G1_BASELINE_RUN_ID,
    spec_id="infer/se/cluster/therapeutic_category",
    spec_tree_path="specification_tree/modules/robustness/controls.md",
    baseline_group_id="G1",
    error_msg=DATA_UNAVAILABLE_MSG + "; cluster(ther1) version of baseline not in log",
    outcome_var="lppd0603",
    treatment_var="mcar0203mepsrx",
    cluster_var="ther1"
)

add_inference_failure(
    base_spec_run_id=G1_BASELINE_RUN_ID,
    spec_id="infer/se/hc/hc2",
    spec_tree_path="specification_tree/designs/cross_sectional_ols.md",
    baseline_group_id="G1",
    error_msg=DATA_UNAVAILABLE_MSG,
    outcome_var="lppd0603",
    treatment_var="mcar0203mepsrx"
)

add_inference_failure(
    base_spec_run_id=G1_BASELINE_RUN_ID,
    spec_id="infer/se/hc/hc3",
    spec_tree_path="specification_tree/designs/cross_sectional_ols.md",
    baseline_group_id="G1",
    error_msg=DATA_UNAVAILABLE_MSG,
    outcome_var="lppd0603",
    treatment_var="mcar0203mepsrx"
)


# =============================================================================
# G2: Quantity Effects (Table 3)
# =============================================================================
print("\n" + "=" * 60)
print("G2: Quantity Effects (Table 3)")
print("=" * 60)

G2_SAMPLE_BASE = "imsgrouprank03<=1000, otc==2, imsgbo==2, fdayear<=2003, smallcat!=."
G2_SAMPLE_TRIM = G2_SAMPLE_BASE + ", ldoses0603 in [-3.95, 1.51]"
G2_SAMPLE_TRIM_NOCANCER = G2_SAMPLE_TRIM + ", thercat!=8"

# --- G2 Baseline: Table 3 Col 4 ---
print("  Running baseline (Table 3 Col 4)...")
add_success_row(
    spec_id="baseline",
    spec_tree_path="specification_tree/designs/cross_sectional_ols.md",
    baseline_group_id="G2",
    outcome_var="ldoses0603",
    treatment_var="mcar0203mepsrx",
    coefficient=0.3891537,
    std_error=0.3155127,
    p_value=0.218,
    ci_lower=-0.2307021,
    ci_upper=1.00901,
    n_obs=517,
    r_squared=0.2942,
    sample_desc=G2_SAMPLE_TRIM_NOCANCER,
    fixed_effects="",
    controls_desc="yrs03onmkt, anygen",
    cluster_var="",
    coef_vector_json=build_log_payload(
        coefficients_dict={"mcar0203mepsrx": 0.3891537, "yrs03onmkt": -0.0009481,
                           "anygen": -1.087196, "_cons": -0.2351405},
        inference_dict=CANONICAL_INFERENCE_G2,
        design_audit=DESIGN_AUDIT_G2
    )
)
G2_BASELINE_RUN_ID = spec_results[-1]["spec_run_id"]

# --- G2 Additional baselines ---
# Table 3 Col 1: bivariate
print("  Running baseline Table 3 Col 1...")
add_success_row(
    spec_id="baseline__table3_col1",
    spec_tree_path="specification_tree/designs/cross_sectional_ols.md",
    baseline_group_id="G2",
    outcome_var="ldoses0603",
    treatment_var="mcar0203mepsrx",
    coefficient=0.5159561,
    std_error=0.3201647,
    p_value=0.108,
    ci_lower=-0.1129493,
    ci_upper=1.144862,
    n_obs=548,
    r_squared=0.0088,
    sample_desc=G2_SAMPLE_BASE,
    fixed_effects="",
    controls_desc="",
    cluster_var="",
    coef_vector_json=build_log_payload(
        coefficients_dict={"mcar0203mepsrx": 0.5159561, "_cons": -0.8264239},
        inference_dict=CANONICAL_INFERENCE_G2,
        design_audit=DESIGN_AUDIT_G2,
        extra_blocks={"controls": {"spec_id": "baseline__table3_col1", "n_controls": 0}}
    )
)

# Table 3 Col 2: add controls
print("  Running baseline Table 3 Col 2...")
add_success_row(
    spec_id="baseline__table3_col2",
    spec_tree_path="specification_tree/designs/cross_sectional_ols.md",
    baseline_group_id="G2",
    outcome_var="ldoses0603",
    treatment_var="mcar0203mepsrx",
    coefficient=0.4337699,
    std_error=0.3195468,
    p_value=0.175,
    ci_lower=-0.1939267,
    ci_upper=1.061467,
    n_obs=548,
    r_squared=0.2617,
    sample_desc=G2_SAMPLE_BASE,
    fixed_effects="",
    controls_desc="yrs03onmkt, anygen",
    cluster_var="",
    coef_vector_json=build_log_payload(
        coefficients_dict={"mcar0203mepsrx": 0.4337699, "yrs03onmkt": -0.0036854,
                           "anygen": -1.071677, "_cons": -0.2507287},
        inference_dict=CANONICAL_INFERENCE_G2,
        design_audit=DESIGN_AUDIT_G2
    )
)

# Table 3 Col 3: add trim
print("  Running baseline Table 3 Col 3...")
add_success_row(
    spec_id="baseline__table3_col3",
    spec_tree_path="specification_tree/designs/cross_sectional_ols.md",
    baseline_group_id="G2",
    outcome_var="ldoses0603",
    treatment_var="mcar0203mepsrx",
    coefficient=0.3805496,
    std_error=0.3128296,
    p_value=0.224,
    ci_lower=-0.2339779,
    ci_upper=0.9950771,
    n_obs=538,
    r_squared=0.2929,
    sample_desc=G2_SAMPLE_TRIM,
    fixed_effects="",
    controls_desc="yrs03onmkt, anygen",
    cluster_var="",
    coef_vector_json=build_log_payload(
        coefficients_dict={"mcar0203mepsrx": 0.3805496, "yrs03onmkt": -0.001389,
                           "anygen": -1.084452, "_cons": -0.2281303},
        inference_dict=CANONICAL_INFERENCE_G2,
        design_audit=DESIGN_AUDIT_G2,
        extra_blocks={"sample": {"spec_id": "baseline__table3_col3",
                                 "trim": "[-3.95, 1.51]", "exclude_cancer": False}}
    )
)

# =============================================================================
# G2 RC specs
# =============================================================================

# rc/controls/loo/drop_yrs03onmkt
print("  Recording rc/controls/loo/drop_yrs03onmkt (data unavailable)...")
add_failure_row(
    spec_id="rc/controls/loo/drop_yrs03onmkt",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G2",
    outcome_var="ldoses0603",
    treatment_var="mcar0203mepsrx",
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc=G2_SAMPLE_TRIM_NOCANCER,
    controls_desc="anygen (drop yrs03onmkt)"
)

# rc/controls/loo/drop_anygen
print("  Recording rc/controls/loo/drop_anygen (data unavailable)...")
add_failure_row(
    spec_id="rc/controls/loo/drop_anygen",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G2",
    outcome_var="ldoses0603",
    treatment_var="mcar0203mepsrx",
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc=G2_SAMPLE_TRIM_NOCANCER,
    controls_desc="yrs03onmkt (drop anygen)"
)

# rc/controls/single/add_protected
print("  Recording rc/controls/single/add_protected (data unavailable)...")
add_failure_row(
    spec_id="rc/controls/single/add_protected",
    spec_tree_path="specification_tree/modules/robustness/controls.md#single",
    baseline_group_id="G2",
    outcome_var="ldoses0603",
    treatment_var="mcar0203mepsrx",
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc=G2_SAMPLE_TRIM_NOCANCER,
    controls_desc="yrs03onmkt, anygen, protected"
)

# rc/controls/single/add_smallcat
print("  Recording rc/controls/single/add_smallcat (data unavailable)...")
add_failure_row(
    spec_id="rc/controls/single/add_smallcat",
    spec_tree_path="specification_tree/modules/robustness/controls.md#single",
    baseline_group_id="G2",
    outcome_var="ldoses0603",
    treatment_var="mcar0203mepsrx",
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc=G2_SAMPLE_TRIM_NOCANCER,
    controls_desc="yrs03onmkt, anygen, smallcat"
)

# rc/controls/single/add_ldoses0201
print("  Recording rc/controls/single/add_ldoses0201 (data unavailable)...")
add_failure_row(
    spec_id="rc/controls/single/add_ldoses0201",
    spec_tree_path="specification_tree/modules/robustness/controls.md#single",
    baseline_group_id="G2",
    outcome_var="ldoses0603",
    treatment_var="mcar0203mepsrx",
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc=G2_SAMPLE_TRIM_NOCANCER,
    controls_desc="yrs03onmkt, anygen, ldoses0201"
)

# rc/controls/sets/interactions_protected
print("  Recording rc/controls/sets/interactions_protected (data unavailable)...")
add_failure_row(
    spec_id="rc/controls/sets/interactions_protected",
    spec_tree_path="specification_tree/modules/robustness/controls.md#sets",
    baseline_group_id="G2",
    outcome_var="ldoses0603",
    treatment_var="mcar0203mepsrx",
    error_msg=DATA_UNAVAILABLE_MSG + "; Table 5 Col 5 uses cluster(ther1), not HC1",
    sample_desc=G2_SAMPLE_TRIM_NOCANCER,
    controls_desc="yrs03onmkt, anygen, protected, mcar0203prot"
)

# rc/controls/sets/interactions_smallcat
print("  Recording rc/controls/sets/interactions_smallcat (data unavailable)...")
add_failure_row(
    spec_id="rc/controls/sets/interactions_smallcat",
    spec_tree_path="specification_tree/modules/robustness/controls.md#sets",
    baseline_group_id="G2",
    outcome_var="ldoses0603",
    treatment_var="mcar0203mepsrx",
    error_msg=DATA_UNAVAILABLE_MSG + "; Table 5 Col 6 uses cluster(ther1), not HC1",
    sample_desc=G2_SAMPLE_TRIM_NOCANCER,
    controls_desc="yrs03onmkt, anygen, smallcat, scmcar0203"
)

# rc/controls/sets/interactions_both
print("  Recording rc/controls/sets/interactions_both (data unavailable)...")
add_failure_row(
    spec_id="rc/controls/sets/interactions_both",
    spec_tree_path="specification_tree/modules/robustness/controls.md#sets",
    baseline_group_id="G2",
    outcome_var="ldoses0603",
    treatment_var="mcar0203mepsrx",
    error_msg=DATA_UNAVAILABLE_MSG + "; Table 5 Col 7 uses cluster(ther1), not HC1",
    sample_desc=G2_SAMPLE_TRIM_NOCANCER,
    controls_desc="yrs03onmkt, anygen, smallcat, scmcar0203, protected, mcar0203prot"
)

# rc/sample/outliers/no_trim
print("  Recording rc/sample/outliers/no_trim (data unavailable)...")
add_failure_row(
    spec_id="rc/sample/outliers/no_trim",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    baseline_group_id="G2",
    outcome_var="ldoses0603",
    treatment_var="mcar0203mepsrx",
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc=G2_SAMPLE_BASE + ", thercat!=8 (no outlier trim)"
)

# rc/sample/outliers/trim_y_5_95
print("  Recording rc/sample/outliers/trim_y_5_95 (data unavailable)...")
add_failure_row(
    spec_id="rc/sample/outliers/trim_y_5_95",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    baseline_group_id="G2",
    outcome_var="ldoses0603",
    treatment_var="mcar0203mepsrx",
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc=G2_SAMPLE_BASE + ", thercat!=8, ldoses0603 trimmed at 5th/95th pctile"
)

# rc/sample/subset/include_cancer -- Table 3 Col 3 (trim, include cancer)
print("  Running rc/sample/subset/include_cancer (from Table 3 Col 3)...")
add_success_row(
    spec_id="rc/sample/subset/include_cancer",
    spec_tree_path="specification_tree/modules/robustness/sample.md#subset",
    baseline_group_id="G2",
    outcome_var="ldoses0603",
    treatment_var="mcar0203mepsrx",
    coefficient=0.3805496,
    std_error=0.3128296,
    p_value=0.224,
    ci_lower=-0.2339779,
    ci_upper=0.9950771,
    n_obs=538,
    r_squared=0.2929,
    sample_desc=G2_SAMPLE_TRIM + " (cancer drugs included)",
    fixed_effects="",
    controls_desc="yrs03onmkt, anygen",
    cluster_var="",
    coef_vector_json=build_log_payload(
        coefficients_dict={"mcar0203mepsrx": 0.3805496, "yrs03onmkt": -0.001389,
                           "anygen": -1.084452, "_cons": -0.2281303},
        inference_dict=CANONICAL_INFERENCE_G2,
        design_audit=DESIGN_AUDIT_G2,
        extra_blocks={"sample": {"spec_id": "rc/sample/subset/include_cancer",
                                 "family": "subset", "cancer_included": True}}
    )
)

# rc/sample/subset/top293 -- Table 3 Col 6
print("  Running rc/sample/subset/top293 (from Table 3 Col 6)...")
add_success_row(
    spec_id="rc/sample/subset/top293",
    spec_tree_path="specification_tree/modules/robustness/sample.md#subset",
    baseline_group_id="G2",
    outcome_var="ldoses0603",
    treatment_var="mcar0203mepsrx",
    coefficient=0.5167769,
    std_error=0.4701634,
    p_value=0.273,
    ci_lower=-0.4104516,
    ci_upper=1.444005,
    n_obs=200,
    r_squared=0.3176,
    sample_desc="imsgrouprank03<=293, otc==2, imsgbo==2, fdayear<=2003, smallcat!=., ldoses0603 in [-3.95, 1.51], thercat!=8",
    fixed_effects="",
    controls_desc="yrs03onmkt, anygen",
    cluster_var="",
    coef_vector_json=build_log_payload(
        coefficients_dict={"mcar0203mepsrx": 0.5167769, "yrs03onmkt": 0.0066473,
                           "anygen": -1.199203, "_cons": -0.2920525},
        inference_dict=CANONICAL_INFERENCE_G2,
        design_audit=DESIGN_AUDIT_G2,
        extra_blocks={"sample": {"spec_id": "rc/sample/subset/top293",
                                 "family": "subset", "rank_cutoff": 293}}
    )
)

# rc/sample/subset/top300
print("  Recording rc/sample/subset/top300 (data unavailable)...")
add_failure_row(
    spec_id="rc/sample/subset/top300",
    spec_tree_path="specification_tree/modules/robustness/sample.md#subset",
    baseline_group_id="G2",
    outcome_var="ldoses0603",
    treatment_var="mcar0203mepsrx",
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="imsgrouprank03<=300 + baseline restrictions"
)

# rc/sample/subset/drop_generic_facing -- Table 3 Col 7
print("  Running rc/sample/subset/drop_generic_facing (from Table 3 Col 7)...")
add_success_row(
    spec_id="rc/sample/subset/drop_generic_facing",
    spec_tree_path="specification_tree/modules/robustness/sample.md#subset",
    baseline_group_id="G2",
    outcome_var="ldoses0603",
    treatment_var="mcar0203mepsrx",
    coefficient=0.2651332,
    std_error=0.1372073,
    p_value=0.054,
    ci_lower=-0.0048255,
    ci_upper=0.5350919,
    n_obs=318,
    r_squared=0.1081,
    sample_desc=G2_SAMPLE_TRIM_NOCANCER + ", anygen==0",
    fixed_effects="",
    controls_desc="yrs03onmkt (anygen dropped: all zero in sample)",
    cluster_var="",
    coef_vector_json=build_log_payload(
        coefficients_dict={"mcar0203mepsrx": 0.2651332, "yrs03onmkt": -0.0232019,
                           "anygen": 0.0, "_cons": 0.0008013},
        inference_dict=CANONICAL_INFERENCE_G2,
        design_audit=DESIGN_AUDIT_G2,
        extra_blocks={"sample": {"spec_id": "rc/sample/subset/drop_generic_facing",
                                 "family": "subset", "restriction": "anygen==0"}}
    )
)

# rc/form/treatment/spending_share -- Table 3 Col 5
print("  Running rc/form/treatment/spending_share (from Table 3 Col 5)...")
add_success_row(
    spec_id="rc/form/treatment/spending_share",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    baseline_group_id="G2",
    outcome_var="ldoses0603",
    treatment_var="mcar0203mepspd",
    coefficient=0.3211932,
    std_error=0.3134185,
    p_value=0.306,
    ci_lower=-0.2945486,
    ci_upper=0.9369349,
    n_obs=517,
    r_squared=0.2925,
    sample_desc=G2_SAMPLE_TRIM_NOCANCER,
    fixed_effects="",
    controls_desc="yrs03onmkt, anygen",
    cluster_var="",
    coef_vector_json=build_log_payload(
        coefficients_dict={"mcar0203mepspd": 0.3211932, "yrs03onmkt": -0.0011564,
                           "anygen": -1.087987, "_cons": -0.2079498},
        inference_dict=CANONICAL_INFERENCE_G2,
        design_audit=DESIGN_AUDIT_G2,
        extra_blocks={"functional_form": {
            "spec_id": "rc/form/treatment/spending_share",
            "interpretation": "Medicare spending share instead of prescription share",
            "treatment_definition": "mcar0203mepspd"
        }}
    )
)

# rc/form/treatment/self_pay_decomposed -- Table 4 Col 5
print("  Running rc/form/treatment/self_pay_decomposed (from Table 4 Col 5)...")
add_success_row(
    spec_id="rc/form/treatment/self_pay_decomposed",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    baseline_group_id="G2",
    outcome_var="ldoses0603",
    treatment_var="mcself0203mepsrx",
    coefficient=0.4607672,
    std_error=0.4235884,
    p_value=0.277,
    ci_lower=-0.371418,
    ci_upper=1.292952,
    n_obs=517,
    r_squared=0.2943,
    sample_desc=G2_SAMPLE_TRIM_NOCANCER,
    fixed_effects="",
    controls_desc="yrs03onmkt, anygen, mcoth0203mepsrx",
    cluster_var="",
    coef_vector_json=build_log_payload(
        coefficients_dict={"mcself0203mepsrx": 0.4607672, "mcoth0203mepsrx": 0.2317276,
                           "yrs03onmkt": -0.0012549, "anygen": -1.087941, "_cons": -0.225536},
        inference_dict=CANONICAL_INFERENCE_G2,
        design_audit=DESIGN_AUDIT_G2,
        extra_blocks={"functional_form": {
            "spec_id": "rc/form/treatment/self_pay_decomposed",
            "interpretation": "Decompose Medicare share into self-pay and other components",
            "focal_treatment": "mcself0203mepsrx"
        }}
    )
)

# explore/outcome/log_sales_change -- Table 4 Col 7 (lsalesq0603)
# This uses lsalesq0603 as outcome on G2 sample -- but Table 4 Col 7 uses G1 sample trim bounds.
# Not a direct G2 spec; skip or record as failure.
print("  Recording explore/outcome/log_sales_change (data unavailable)...")
add_failure_row(
    spec_id="explore/outcome/log_sales_change",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    baseline_group_id="G2",
    outcome_var="lsalesq0603",
    treatment_var="mcar0203mepsrx",
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="log_sales_change on G2 sample not in log"
)

# rc/weights/unweighted
print("  Recording rc/weights/unweighted (data unavailable)...")
add_failure_row(
    spec_id="rc/weights/unweighted",
    spec_tree_path="specification_tree/modules/robustness/weights.md",
    baseline_group_id="G2",
    outcome_var="ldoses0603",
    treatment_var="mcar0203mepsrx",
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc=G2_SAMPLE_TRIM_NOCANCER
)

# rc/joint/no_trim_include_cancer -- Table 3 Col 2 (controls, no trim, include cancer)
print("  Running rc/joint/no_trim_include_cancer (from Table 3 Col 2)...")
add_success_row(
    spec_id="rc/joint/no_trim_include_cancer",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G2",
    outcome_var="ldoses0603",
    treatment_var="mcar0203mepsrx",
    coefficient=0.4337699,
    std_error=0.3195468,
    p_value=0.175,
    ci_lower=-0.1939267,
    ci_upper=1.061467,
    n_obs=548,
    r_squared=0.2617,
    sample_desc=G2_SAMPLE_BASE + " (no trim, cancer included)",
    fixed_effects="",
    controls_desc="yrs03onmkt, anygen",
    cluster_var="",
    coef_vector_json=build_log_payload(
        coefficients_dict={"mcar0203mepsrx": 0.4337699, "yrs03onmkt": -0.0036854,
                           "anygen": -1.071677, "_cons": -0.2507287},
        inference_dict=CANONICAL_INFERENCE_G2,
        design_audit=DESIGN_AUDIT_G2,
        extra_blocks={"joint": {
            "spec_id": "rc/joint/no_trim_include_cancer",
            "components": ["no_trim", "include_cancer"]
        }}
    )
)

# rc/joint/drop_generic_no_cancer -- Table 3 Col 7 IS this already
print("  Running rc/joint/drop_generic_no_cancer (same as Table 3 Col 7)...")
add_success_row(
    spec_id="rc/joint/drop_generic_no_cancer",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G2",
    outcome_var="ldoses0603",
    treatment_var="mcar0203mepsrx",
    coefficient=0.2651332,
    std_error=0.1372073,
    p_value=0.054,
    ci_lower=-0.0048255,
    ci_upper=0.5350919,
    n_obs=318,
    r_squared=0.1081,
    sample_desc=G2_SAMPLE_TRIM_NOCANCER + ", anygen==0",
    fixed_effects="",
    controls_desc="yrs03onmkt (anygen dropped: all zero)",
    cluster_var="",
    coef_vector_json=build_log_payload(
        coefficients_dict={"mcar0203mepsrx": 0.2651332, "yrs03onmkt": -0.0232019,
                           "anygen": 0.0, "_cons": 0.0008013},
        inference_dict=CANONICAL_INFERENCE_G2,
        design_audit=DESIGN_AUDIT_G2,
        extra_blocks={"joint": {
            "spec_id": "rc/joint/drop_generic_no_cancer",
            "components": ["drop_generic_facing", "exclude_cancer"]
        }}
    )
)

# =============================================================================
# G2 Inference variants
# =============================================================================
print("\n  G2 Inference variants...")

add_inference_failure(
    base_spec_run_id=G2_BASELINE_RUN_ID,
    spec_id="infer/se/cluster/therapeutic_category",
    spec_tree_path="specification_tree/modules/robustness/controls.md",
    baseline_group_id="G2",
    error_msg=DATA_UNAVAILABLE_MSG + "; cluster(ther1) version of baseline not in log",
    outcome_var="ldoses0603",
    treatment_var="mcar0203mepsrx",
    cluster_var="ther1"
)

add_inference_failure(
    base_spec_run_id=G2_BASELINE_RUN_ID,
    spec_id="infer/se/hc/hc3",
    spec_tree_path="specification_tree/designs/cross_sectional_ols.md",
    baseline_group_id="G2",
    error_msg=DATA_UNAVAILABLE_MSG,
    outcome_var="ldoses0603",
    treatment_var="mcar0203mepsrx"
)


# =============================================================================
# Write outputs
# =============================================================================
print("\n" + "=" * 60)
print("Writing outputs...")
print("=" * 60)

# specification_results.csv
# Separate explore/* rows into exploration_results.csv
core_rows = [r for r in spec_results if not r["spec_id"].startswith("explore/")]
explore_rows = [r for r in spec_results if r["spec_id"].startswith("explore/")]

spec_df = pd.DataFrame(core_rows)
spec_df.to_csv(os.path.join(PACKAGE_DIR, "specification_results.csv"), index=False)
print(f"  specification_results.csv: {len(core_rows)} rows")

if explore_rows:
    explore_df = pd.DataFrame(explore_rows)
    # Add required exploration columns (keep spec_run_id AND add exploration_run_id)
    explore_df["exploration_run_id"] = explore_df["spec_run_id"]
    explore_df["exploration_json"] = explore_df["coefficient_vector_json"]
    explore_df.to_csv(os.path.join(PACKAGE_DIR, "exploration_results.csv"), index=False)
    print(f"  exploration_results.csv: {len(explore_rows)} rows")

# inference_results.csv
infer_df = pd.DataFrame(inference_results)
infer_df.to_csv(os.path.join(PACKAGE_DIR, "inference_results.csv"), index=False)
print(f"  inference_results.csv: {len(inference_results)} rows")

# Summary stats
n_success = sum(1 for r in core_rows if r["run_success"] == 1)
n_failure = sum(1 for r in core_rows if r["run_success"] == 0)
n_infer_success = sum(1 for r in inference_results if r["run_success"] == 1)
n_infer_failure = sum(1 for r in inference_results if r["run_success"] == 0)

print(f"\n  Core specs: {n_success} succeeded, {n_failure} failed (proprietary data)")
print(f"  Inference:  {n_infer_success} succeeded, {n_infer_failure} failed")
print(f"  Explore:    {len(explore_rows)} rows")

# =============================================================================
# Write SPECIFICATION_SEARCH.md
# =============================================================================
search_md = f"""# Specification Search: {PAPER_ID}

## Surface Summary
- **Paper**: Duggan & Scott Morton (2010), "The Effect of Medicare Part D on Pharmaceutical Prices and Utilization", AER
- **Baseline groups**: 2 (G1: price effects, G2: quantity effects)
- **Design**: cross_sectional_ols
- **Budgets**: G1 max 80 core specs, G2 max 60 core specs
- **Seeds**: G1=112338, G2=112339
- **Surface hash**: {SURFACE_HASH}

## Critical Data Limitation

**The IMS Health pharmaceutical sales/dose data (ims0106data2.dta, ims0106all.dta) is proprietary
and NOT included in the ICPSR replication package.** This data provides the drug-level sales
revenue and dose quantities for 2001-2006 that are needed to construct ALL outcome variables:

- `lppd0603` (log change in price-per-day, 2003-2006)
- `ldoses0603` (log change in total doses, 2003-2006)
- `lsalesq0603` (log change in sales revenue, 2003-2006)
- `ppd0603` (level change in price-per-day)

Without this data, we cannot run any regressions beyond what is recorded in the Stata log file
(`regs-partd-final.log`). The log contains output for all regressions in the paper's Tables 2-5.

**Approach**: Extract coefficients from the Stata log for all specifications that match the
surface's approved universe. Record all other specifications as failures with the reason
"proprietary IMS data unavailable."

## Execution Summary

### G1: Price Effects (Table 2)

| Category | Planned | Executed | Failed |
|----------|---------|----------|--------|
| Baseline | 4 | 4 | 0 |
| RC specs | 28 | 6 | 22 |
| **Total** | **32** | **10** | **22** |

Specifications extracted from log:
- `baseline` (Table 2 Col 4) -- coef=-0.1364, p=0.015, N=517
- `baseline__table2_col1` (Table 2 Col 1) -- coef=-0.1278, p=0.026, N=548
- `baseline__table2_col2` (Table 2 Col 2) -- coef=-0.1272, p=0.026, N=548
- `baseline__table2_col3` (Table 2 Col 3) -- coef=-0.1376, p=0.014, N=538
- `rc/sample/subset/include_cancer` (=Table 2 Col 3) -- coef=-0.1376, p=0.014, N=538
- `rc/sample/subset/top292` (Table 2 Col 6) -- coef=-0.1427, p=0.009, N=200
- `rc/form/treatment/spending_share` (Table 2 Col 5) -- coef=-0.1329, p=0.018, N=517
- `rc/form/treatment/self_pay_decomposed` (Table 4 Col 2) -- coef=-0.2253, p=0.001, N=517
- `rc/form/treatment/dual_decomposed` (Table 4 Col 3) -- coef=-0.2429, p=0.002, N=517
- `rc/joint/no_trim_include_cancer` (=Table 2 Col 2) -- coef=-0.1272, p=0.026, N=548
- `rc/joint/spending_share_no_cancer` (=Table 2 Col 5) -- coef=-0.1329, p=0.018, N=517

### G2: Quantity Effects (Table 3)

| Category | Planned | Executed | Failed |
|----------|---------|----------|--------|
| Baseline | 4 | 4 | 0 |
| RC specs | 20 | 7 | 13 |
| **Total** | **24** | **11** | **13** |

Specifications extracted from log:
- `baseline` (Table 3 Col 4) -- coef=0.3892, p=0.218, N=517
- `baseline__table3_col1` (Table 3 Col 1) -- coef=0.5160, p=0.108, N=548
- `baseline__table3_col2` (Table 3 Col 2) -- coef=0.4338, p=0.175, N=548
- `baseline__table3_col3` (Table 3 Col 3) -- coef=0.3805, p=0.224, N=538
- `rc/sample/subset/include_cancer` (=Table 3 Col 3) -- coef=0.3805, p=0.224, N=538
- `rc/sample/subset/top293` (Table 3 Col 6) -- coef=0.5168, p=0.273, N=200
- `rc/sample/subset/drop_generic_facing` (Table 3 Col 7) -- coef=0.2651, p=0.054, N=318
- `rc/form/treatment/spending_share` (Table 3 Col 5) -- coef=0.3212, p=0.306, N=517
- `rc/form/treatment/self_pay_decomposed` (Table 4 Col 5) -- coef=0.4608, p=0.277, N=517
- `rc/joint/no_trim_include_cancer` (=Table 3 Col 2) -- coef=0.4338, p=0.175, N=548
- `rc/joint/drop_generic_no_cancer` (=Table 3 Col 7) -- coef=0.2651, p=0.054, N=318

### Exploration (non-core)
- `explore/outcome/log_sales_change` (G1, Table 4 Col 7) -- coef=0.2728, p=0.385, N=517 [SUCCESS]
- `explore/outcome/log_sales_change` (G2) -- FAILED (data unavailable)
- `explore/outcome/log_doses_change` (G1) -- FAILED (data unavailable)

### Inference Variants
- All 5 inference variants FAILED: proprietary data prevents re-estimation with alternative SEs

## Overall Counts

| | Planned | Succeeded | Failed |
|---|---|---|---|
| Core specs (G1+G2) | {n_success + n_failure} | {n_success} | {n_failure} |
| Inference variants | {n_infer_success + n_infer_failure} | {n_infer_success} | {n_infer_failure} |
| Exploration | {len(explore_rows)} | {sum(1 for r in explore_rows if r['run_success']==1)} | {sum(1 for r in explore_rows if r['run_success']==0)} |

## Failure Reason

All failures share the same root cause: **The IMS Health pharmaceutical sales and dose data
(`ims0106data2.dta`, `ims0106all.dta`) is proprietary (purchased from IMS Health / IQVIA) and
was not included in the ICPSR replication package.** The readme explicitly states: "Because of
the proprietary nature of our data we cannot share these files."

Without this data, ALL outcome variables (price changes, dose changes, sales changes) cannot
be computed, making it impossible to run any regression not already recorded in the Stata log.

## Deviations from Surface

1. The surface requests ~80 G1 specs and ~60 G2 specs. Only {n_success} core specs could be
   extracted from the log (matching paper Tables 2-5).
2. Table 5 regressions use `cluster(ther1)` rather than HC1 (the canonical inference).
   These are recorded as failures for the canonical inference version, since the log only
   contains clustered SEs for these specifications.
3. Some surface RC specs exactly correspond to paper table columns (e.g., `rc/sample/subset/top292`
   = Table 2 Col 6), while others require novel regressions not in any table.

## Software Stack

- Source: Stata log file extraction (original analysis used Stata)
- Coefficients, SEs, p-values, CIs, N, R-squared all extracted verbatim from `regs-partd-final.log`
- Python packages used for output generation: pandas {pd.__version__}, numpy {np.__version__}
"""

with open(os.path.join(PACKAGE_DIR, "SPECIFICATION_SEARCH.md"), 'w') as f:
    f.write(search_md)
print(f"  SPECIFICATION_SEARCH.md written")

print("\nDone.")
