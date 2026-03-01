"""
Specification Search Script for Davis, Fuchs & Gertler (2014)
"Cash for Coolers: Evaluating a Large-Scale Appliance Replacement Program in Mexico"
American Economic Journal: Economic Policy, 6(3), 207-238.

Paper ID: 114879-V1

DATA UNAVAILABILITY NOTE:
The central dataset is a two-year panel of household-level electric billing records
from the Mexican Federal Electricity Commission (CFE). These data were provided under
a Non-Disclosure Agreement (NDA) and are NOT included in the replication package.
Only the Stata code (.do files) is posted. The only data files provided are:
  - census.dta (ancillary census analysis, not for main DiD regressions)
  - avgefficiencies.dta (ancillary appliance efficiency analysis)

Since the household billing panel is required for ALL specifications in the surface
(baseline, design, and RC variants), every spec is recorded as run_success=0 with
the data-unavailable error.

Surface-driven execution:
  - G1: usage ~ rrefr (DiD with TWFE, household x calendar-month + county x month FE)
  - Cluster SE at county level
  - All specs fail due to confidential data

Outputs:
  - specification_results.csv (all rows with run_success=0)
  - inference_results.csv (all rows with run_success=0)
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
PAPER_ID = "114879-V1"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PACKAGE_DIR = os.path.join(REPO_ROOT, "data", "downloads", "extracted", PAPER_ID)
DATA_DIR = os.path.join(PACKAGE_DIR, "data-appendix")
SURFACE_FILE = os.path.join(PACKAGE_DIR, "SPECIFICATION_SURFACE.json")

# Import helper utilities
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from agent_output_utils import (
    make_failure_payload,
    surface_hash as compute_surface_hash,
    software_block
)

# =============================================================================
# Load surface and compute hash
# =============================================================================
with open(SURFACE_FILE, 'r') as f:
    surface = json.load(f)
SURFACE_HASH = compute_surface_hash(surface)
SOFTWARE = software_block()

# Extract surface metadata
G1 = surface["baseline_groups"][0]
DESIGN_AUDIT = G1["design_audit"]
CANONICAL_INFERENCE = G1["inference_plan"]["canonical"]
INFERENCE_VARIANTS = G1["inference_plan"]["variants"]

DATA_UNAVAILABLE_MSG = (
    "Confidential household billing data from Mexican Federal Electricity Commission (CFE) "
    "not available; provided under NDA, only code is posted in replication package"
)

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
    return spec_results[-1]["spec_run_id"]


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


# =============================================================================
# Common parameters
# =============================================================================
OUTCOME = "usage"
TREATMENT = "rrefr"
BG = "G1"

# =============================================================================
# BASELINE SPECS (from surface core_universe.baseline_spec_ids)
# =============================================================================

# baseline__col1_simple_fe: Table 3 Col 2 (rrefr + rAC, hhXm + CxM FE, random controls)
run_id_b1 = add_failure_row(
    spec_id="baseline__col1_simple_fe",
    spec_tree_path="specification_tree/methods/difference_in_differences.md#twfe",
    baseline_group_id=BG,
    outcome_var=OUTCOME,
    treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Random control group, drop usage > 200000",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC",
    cluster_var="county"
)

# baseline__col3_ac_summer: Table 3 Col 3 (rrefr + rAC + rAC_summer, hhXm + CxM FE, random)
run_id_b2 = add_failure_row(
    spec_id="baseline__col3_ac_summer",
    spec_tree_path="specification_tree/methods/difference_in_differences.md#twfe",
    baseline_group_id=BG,
    outcome_var=OUTCOME,
    treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Random control group, drop usage > 200000",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC + rAC_summer",
    cluster_var="county"
)

# baseline__location_matched: same spec but with location-matched control group
run_id_b3 = add_failure_row(
    spec_id="baseline__location_matched",
    spec_tree_path="specification_tree/methods/difference_in_differences.md#twfe",
    baseline_group_id=BG,
    outcome_var=OUTCOME,
    treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Location-matched control group (mainregressions_matched.do)",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC + rAC_summer",
    cluster_var="county"
)

# baseline__usage_matched: same spec but with usage-matched control group
run_id_b4 = add_failure_row(
    spec_id="baseline__usage_matched",
    spec_tree_path="specification_tree/methods/difference_in_differences.md#twfe",
    baseline_group_id=BG,
    outcome_var=OUTCOME,
    treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Usage-matched control group (mainregressions_fancy.do)",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC + rAC_summer",
    cluster_var="county"
)

# =============================================================================
# DESIGN VARIANTS (from surface core_universe.design_spec_ids)
# =============================================================================

add_failure_row(
    spec_id="design/difference_in_differences/estimator/twfe",
    spec_tree_path="specification_tree/methods/difference_in_differences.md#twfe",
    baseline_group_id=BG,
    outcome_var=OUTCOME,
    treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Random control group, drop usage > 200000",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC + rAC_summer",
    cluster_var="county"
)

# =============================================================================
# RC VARIANTS (from surface core_universe.rc_spec_ids -- exact list)
# =============================================================================

# --- Controls ---
add_failure_row(
    spec_id="rc/controls/loo/drop_rAC",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Random controls, drop usage > 200000",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC_summer only (dropped rAC)",
    cluster_var="county"
)

add_failure_row(
    spec_id="rc/controls/loo/drop_rAC_summer",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Random controls, drop usage > 200000",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC only (dropped rAC_summer)",
    cluster_var="county"
)

add_failure_row(
    spec_id="rc/controls/sets/none_beyond_fe",
    spec_tree_path="specification_tree/modules/robustness/controls.md#sets",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Random controls, drop usage > 200000",
    fixed_effects="hhXm + CxM",
    controls_desc="no controls beyond FE",
    cluster_var="county"
)

add_failure_row(
    spec_id="rc/controls/sets/with_ac",
    spec_tree_path="specification_tree/modules/robustness/controls.md#sets",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Random controls, drop usage > 200000",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC",
    cluster_var="county"
)

add_failure_row(
    spec_id="rc/controls/sets/with_ac_summer",
    spec_tree_path="specification_tree/modules/robustness/controls.md#sets",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Random controls, drop usage > 200000",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC + rAC_summer",
    cluster_var="county"
)

# --- FE Structure ---
add_failure_row(
    spec_id="rc/fe/structure/hhXm_month",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#structure",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Random controls, drop usage > 200000",
    fixed_effects="hhXm + month",
    controls_desc="rAC",
    cluster_var="county"
)

add_failure_row(
    spec_id="rc/fe/structure/hhXm_CxM",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#structure",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Random controls, drop usage > 200000",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC + rAC_summer",
    cluster_var="county"
)

# --- Sample Restrictions ---
add_failure_row(
    spec_id="rc/sample/restrict/matched_controls",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restrict",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Location-matched control group",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC + rAC_summer",
    cluster_var="county"
)

add_failure_row(
    spec_id="rc/sample/restrict/location_matched",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restrict",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Location-matched control group",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC + rAC_summer",
    cluster_var="county"
)

add_failure_row(
    spec_id="rc/sample/restrict/usage_matched",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restrict",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Usage-matched control group",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC + rAC_summer",
    cluster_var="county"
)

add_failure_row(
    spec_id="rc/sample/restrict/no_control_hh",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restrict",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="No control HHs (particip!=1), random controls dropped",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC + rAC_summer",
    cluster_var="county"
)

add_failure_row(
    spec_id="rc/sample/restrict/drop_transition_month",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restrict",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Random controls, drop transition month (fractional rrefr/rAC)",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC + rAC_summer",
    cluster_var="county"
)

# --- Sample Outliers ---
add_failure_row(
    spec_id="rc/sample/outliers/trim_usage_1_99",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Random controls, trim usage at 1st/99th pctile",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC + rAC_summer",
    cluster_var="county"
)

add_failure_row(
    spec_id="rc/sample/outliers/trim_usage_5_95",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Random controls, trim usage at 5th/95th pctile",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC + rAC_summer",
    cluster_var="county"
)

add_failure_row(
    spec_id="rc/sample/outliers/drop_usage_gt_200000",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Random controls, drop usage > 200000 (paper default outlier rule)",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC + rAC_summer",
    cluster_var="county"
)

add_failure_row(
    spec_id="rc/sample/outliers/trim_usage_2_98",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Random controls, trim usage at 2nd/98th pctile",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC + rAC_summer",
    cluster_var="county"
)

# --- Seasonal/temporal sample restrictions ---
add_failure_row(
    spec_id="rc/sample/restrict/summer_months_only",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restrict",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Random controls, summer months only (moy 5-10)",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC + rAC_summer",
    cluster_var="county"
)

add_failure_row(
    spec_id="rc/sample/restrict/winter_months_only",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restrict",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Random controls, winter months only (moy 1-4, 11-12)",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC + rAC_summer",
    cluster_var="county"
)

add_failure_row(
    spec_id="rc/sample/restrict/pre_treatment_only",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restrict",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Random controls, pre-treatment periods only",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC + rAC_summer",
    cluster_var="county"
)

add_failure_row(
    spec_id="rc/sample/restrict/post_treatment_only",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restrict",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Random controls, post-treatment periods only",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC + rAC_summer",
    cluster_var="county"
)

add_failure_row(
    spec_id="rc/sample/restrict/high_usage_tercile",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restrict",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Random controls, highest pre-treatment usage tercile",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC + rAC_summer",
    cluster_var="county"
)

add_failure_row(
    spec_id="rc/sample/restrict/low_usage_tercile",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restrict",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Random controls, lowest pre-treatment usage tercile",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC + rAC_summer",
    cluster_var="county"
)

add_failure_row(
    spec_id="rc/sample/restrict/exclude_agency_outliers",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restrict",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Random controls, exclude agencies with extreme participation rates",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC + rAC_summer",
    cluster_var="county"
)

# --- Functional Form ---
add_failure_row(
    spec_id="rc/form/outcome/log_usage",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md#log",
    baseline_group_id=BG,
    outcome_var="log(usage)", treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Random controls, drop usage > 200000",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC + rAC_summer",
    cluster_var="county"
)

add_failure_row(
    spec_id="rc/form/outcome/asinh_usage",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md#asinh",
    baseline_group_id=BG,
    outcome_var="asinh(usage)", treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Random controls, drop usage > 200000",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC + rAC_summer",
    cluster_var="county"
)

add_failure_row(
    spec_id="rc/form/outcome/log1p_usage",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md#log",
    baseline_group_id=BG,
    outcome_var="log(1+usage)", treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Random controls, drop usage > 200000",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC + rAC_summer",
    cluster_var="county"
)

add_failure_row(
    spec_id="rc/form/outcome/usage_per_day",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md#rescale",
    baseline_group_id=BG,
    outcome_var="usage_per_day", treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Random controls, drop usage > 200000, usage normalized by billing days",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC + rAC_summer",
    cluster_var="county"
)

# --- Data Construction / Control Group ---
add_failure_row(
    spec_id="rc/data/control_group/random",
    spec_tree_path="specification_tree/modules/robustness/data_construction.md#control-group",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Random control group",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC + rAC_summer",
    cluster_var="county"
)

add_failure_row(
    spec_id="rc/data/control_group/location_matched",
    spec_tree_path="specification_tree/modules/robustness/data_construction.md#control-group",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Location-matched control group (mainregressions_matched.do)",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC + rAC_summer",
    cluster_var="county"
)

add_failure_row(
    spec_id="rc/data/control_group/usage_matched",
    spec_tree_path="specification_tree/modules/robustness/data_construction.md#control-group",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Usage-matched control group (mainregressions_fancy.do)",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC + rAC_summer",
    cluster_var="county"
)

# --- Joint (combinatorial) variants from surface ---
add_failure_row(
    spec_id="rc/joint/random_hhXm_month",
    spec_tree_path="specification_tree/modules/robustness/joint.md#cg-fe",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Random controls, drop usage > 200000",
    fixed_effects="hhXm + month",
    controls_desc="rAC",
    cluster_var="county"
)

add_failure_row(
    spec_id="rc/joint/random_hhXm_CxM",
    spec_tree_path="specification_tree/modules/robustness/joint.md#cg-fe",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Random controls, drop usage > 200000",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC",
    cluster_var="county"
)

add_failure_row(
    spec_id="rc/joint/random_hhXm_CxM_acSummer",
    spec_tree_path="specification_tree/modules/robustness/joint.md#cg-fe",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Random controls, drop usage > 200000",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC + rAC_summer",
    cluster_var="county"
)

add_failure_row(
    spec_id="rc/joint/location_hhXm_month",
    spec_tree_path="specification_tree/modules/robustness/joint.md#cg-fe",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Location-matched controls",
    fixed_effects="hhXm + month",
    controls_desc="rAC",
    cluster_var="county"
)

add_failure_row(
    spec_id="rc/joint/location_hhXm_CxM",
    spec_tree_path="specification_tree/modules/robustness/joint.md#cg-fe",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Location-matched controls",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC",
    cluster_var="county"
)

add_failure_row(
    spec_id="rc/joint/location_hhXm_CxM_acSummer",
    spec_tree_path="specification_tree/modules/robustness/joint.md#cg-fe",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Location-matched controls",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC + rAC_summer",
    cluster_var="county"
)

add_failure_row(
    spec_id="rc/joint/usage_hhXm_month",
    spec_tree_path="specification_tree/modules/robustness/joint.md#cg-fe",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Usage-matched controls",
    fixed_effects="hhXm + month",
    controls_desc="rAC",
    cluster_var="county"
)

add_failure_row(
    spec_id="rc/joint/usage_hhXm_CxM",
    spec_tree_path="specification_tree/modules/robustness/joint.md#cg-fe",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Usage-matched controls",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC",
    cluster_var="county"
)

add_failure_row(
    spec_id="rc/joint/usage_hhXm_CxM_acSummer",
    spec_tree_path="specification_tree/modules/robustness/joint.md#cg-fe",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Usage-matched controls",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC + rAC_summer",
    cluster_var="county"
)

add_failure_row(
    spec_id="rc/joint/random_no_controls_hhXm_CxM",
    spec_tree_path="specification_tree/modules/robustness/joint.md#cg-fe-ctrl",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Random controls, drop usage > 200000",
    fixed_effects="hhXm + CxM",
    controls_desc="no controls beyond FE",
    cluster_var="county"
)

add_failure_row(
    spec_id="rc/joint/random_drop_transition_hhXm_CxM",
    spec_tree_path="specification_tree/modules/robustness/joint.md#cg-sample",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Random controls, drop transition month",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC + rAC_summer",
    cluster_var="county"
)

add_failure_row(
    spec_id="rc/joint/location_drop_transition_hhXm_CxM",
    spec_tree_path="specification_tree/modules/robustness/joint.md#cg-sample",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Location-matched controls, drop transition month",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC + rAC_summer",
    cluster_var="county"
)

add_failure_row(
    spec_id="rc/joint/random_no_control_hh_hhXm_CxM",
    spec_tree_path="specification_tree/modules/robustness/joint.md#cg-sample",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Random controls, drop control HHs (particip!=1)",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC + rAC_summer",
    cluster_var="county"
)

add_failure_row(
    spec_id="rc/joint/random_no_control_hh_no_transition",
    spec_tree_path="specification_tree/modules/robustness/joint.md#cg-sample",
    baseline_group_id=BG,
    outcome_var=OUTCOME, treatment_var=TREATMENT,
    error_msg=DATA_UNAVAILABLE_MSG,
    sample_desc="Random controls, drop control HHs and transition month",
    fixed_effects="hhXm + CxM",
    controls_desc="rAC + rAC_summer",
    cluster_var="county"
)

# =============================================================================
# INFERENCE VARIANTS (for each baseline spec)
# =============================================================================

baseline_run_ids = [run_id_b1, run_id_b2, run_id_b3, run_id_b4]
for base_run_id in baseline_run_ids:
    for variant in INFERENCE_VARIANTS:
        add_inference_failure(
            base_spec_run_id=base_run_id,
            spec_id=variant["spec_id"],
            spec_tree_path="specification_tree/modules/inference/standard_errors.md#cluster",
            baseline_group_id=BG,
            error_msg=DATA_UNAVAILABLE_MSG,
            outcome_var=OUTCOME,
            treatment_var=TREATMENT,
            cluster_var=variant["params"].get("cluster_var", "")
        )

# =============================================================================
# WRITE OUTPUTS
# =============================================================================

# specification_results.csv
df_spec = pd.DataFrame(spec_results)
df_spec.to_csv(os.path.join(PACKAGE_DIR, "specification_results.csv"), index=False)
print(f"Wrote {len(df_spec)} spec rows to specification_results.csv")

# inference_results.csv
df_infer = pd.DataFrame(inference_results)
df_infer.to_csv(os.path.join(PACKAGE_DIR, "inference_results.csv"), index=False)
print(f"Wrote {len(df_infer)} inference rows to inference_results.csv")

# =============================================================================
# SPECIFICATION_SEARCH.md
# =============================================================================
n_total = len(df_spec)
n_baseline = 4
n_design = 1
n_rc = n_total - n_baseline - n_design
n_infer_total = len(df_infer)

search_md = f"""# Specification Search: {PAPER_ID}

## Paper
Davis, Fuchs & Gertler (2014). "Cash for Coolers: Evaluating a Large-Scale Appliance
Replacement Program in Mexico." *American Economic Journal: Economic Policy*, 6(3), 207-238.

## Surface Summary

- **Baseline groups**: 1 (G1: usage ~ rrefr, DiD with TWFE)
- **Design**: difference_in_differences (TWFE with double-demeaning via reg2hdfe)
- **Baselines**: 4 (Table3-Col2-Random, Table3-Col3-Random, location-matched, usage-matched)
- **Design variants**: 1 (TWFE)
- **RC variants**: {n_rc} (controls, FE structure, sample restrictions, outliers, functional form, control group, joint)
- **Budget**: 70 max core specs
- **Seed**: 114879
- **Controls-subset sampling**: none (only 2 controls available)

## Data Availability

**CRITICAL: The central household billing data is NOT available.**

The paper uses a two-year panel of household-level electric billing records from the
Mexican Federal Electricity Commission (CFE). These data were provided to UC Berkeley's
Energy Institute under a Non-Disclosure Agreement (NDA). The ReadMe explicitly states:

> "This agreement prevents us from sharing these data with other researchers, though we
> have been able to post all code."

Only the following data files are included in the replication package:
- `census.dta` -- ancillary Mexican Census analysis (not for main regressions)
- `avgefficiencies.dta` -- ancillary appliance efficiency analysis

The code files are available:
- `national39.do` -- data construction from ~575 raw CFE billing files
- `mainregressions_random.do` -- main regressions with random control group
- `mainregressions_matched.do` -- regressions with location-matched control group
- `mainregressions_fancy.do` -- regressions with usage-matched control group + trend tests

## Execution Summary

| Category | Planned | Executed | Failed |
|----------|---------|----------|--------|
| Baselines | {n_baseline} | 0 | {n_baseline} |
| Design variants | {n_design} | 0 | {n_design} |
| RC variants | {n_rc} | 0 | {n_rc} |
| **Total spec rows** | **{n_total}** | **0** | **{n_total}** |
| Inference variants | {n_infer_total} | 0 | {n_infer_total} |

**All {n_total} specification rows and {n_infer_total} inference rows recorded as failures.**
Reason: Confidential household billing data from CFE not available in replication package (NDA).

## Planned Specifications (all failed)

### Baselines (from surface core_universe.baseline_spec_ids)
1. `baseline__col1_simple_fe`: Table 3 Col 2 -- usage ~ rrefr + rAC | hhXm + CxM, cluster(county), random controls
2. `baseline__col3_ac_summer`: Table 3 Col 3 -- usage ~ rrefr + rAC + rAC_summer | hhXm + CxM, cluster(county), random controls
3. `baseline__location_matched`: same as Col 3 with location-matched control group
4. `baseline__usage_matched`: same as Col 3 with usage-matched control group

### Design Variants (from surface core_universe.design_spec_ids)
5. `design/difference_in_differences/estimator/twfe`: TWFE (only feasible estimator)

### RC: Controls (5 specs)
6-10. LOO drops (rAC, rAC_summer), control set progressions (none, with_ac, with_ac_summer)

### RC: FE Structure (2 specs)
11-12. Simpler FE (hhXm + month) vs preferred (hhXm + CxM)

### RC: Sample Restrictions (11 specs)
13-23. Matched controls, no control HHs, drop transition month, summer/winter only,
pre/post-treatment only, high/low usage tercile, exclude agency outliers

### RC: Outliers (4 specs)
24-27. Trim usage at 1/99, 2/98, 5/95 percentiles; drop usage > 200000

### RC: Functional Form (4 specs)
28-31. Log, asinh, log(1+x), usage_per_day transformations of outcome

### RC: Control Group (3 specs)
32-34. Random, location-matched, usage-matched control groups

### RC: Joint Combinatorial (14 specs)
35-48. Control group x FE structure x controls grid (9 specs),
control group x sample restriction combinations (5 specs)

### Inference Variants ({n_infer_total} rows)
- `infer/se/cluster/state` (coarser state-level clustering) x 4 baselines
- `infer/se/hc/hc1` (robust HC1 only) x 4 baselines

## Software Stack
- Python {SOFTWARE.get('runner_version', 'N/A')}
- pandas, numpy (for output generation only; no estimation performed)

## Deviations from Surface
None. All {n_total} spec_ids and {n_infer_total} inference variants faithfully enumerated from the
surface's core_universe but none could be executed due to confidential data. The surface
itself is well-designed for this paper's DiD structure.
"""

with open(os.path.join(PACKAGE_DIR, "SPECIFICATION_SEARCH.md"), 'w') as f:
    f.write(search_md)
print(f"Wrote SPECIFICATION_SEARCH.md")

print(f"\nDone. {n_total} spec rows, {n_infer_total} inference rows (all failures due to NDA data).")
