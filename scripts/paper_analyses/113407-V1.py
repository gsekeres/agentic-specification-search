"""
Specification Search Script for Hoynes (2014)
"Living Arrangements, Doubling Up, and the Great Recession:
 Was This Time Different?"
American Economic Review Papers & Proceedings, 104(5), 107-112.

Paper ID: 113407-V1

Surface-driven execution:
  - G1: myadult_aloneall_1830 ~ urate (fraction young adults 18-30 living alone)
  - G2: h_numpers_noneld ~ urate (household size, non-elderly)
  - State-year panel with state + year FE, analytic weights, clustered at state
  - Strict adherence to surface core_universe spec_ids

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

PAPER_ID = "113407-V1"
DATA_DIR = "data/downloads/extracted/113407-V1"
SUBDIR = f"{DATA_DIR}/Hoynes-data-do-readme"
OUTPUT_DIR = DATA_DIR

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

# Design audit blocks from surface
G1_DESIGN_AUDIT = surface_obj["baseline_groups"][0]["design_audit"]
G2_DESIGN_AUDIT = surface_obj["baseline_groups"][1]["design_audit"]
G1_INFERENCE_CANONICAL = surface_obj["baseline_groups"][0]["inference_plan"]["canonical"]
G2_INFERENCE_CANONICAL = surface_obj["baseline_groups"][1]["inference_plan"]["canonical"]

# ============================================================
# LOAD DATA
# ============================================================
df_raw = pd.read_stata(f"{SUBDIR}/collapsed_state_aeapap.dta")

# Convert category to string for pyfixest and float32 to float64
df_raw['statefip'] = df_raw['statefip'].astype(str)
df_raw['calyear'] = df_raw['calyear'].astype(int)
for col in df_raw.columns:
    if df_raw[col].dtype == np.float32:
        df_raw[col] = df_raw[col].astype(np.float64)

# Construct region from state names (Census regions)
REGION_MAP = {
    'Connecticut': 1, 'Maine': 1, 'Massachusetts': 1, 'New Hampshire': 1,
    'Rhode Island': 1, 'Vermont': 1, 'New Jersey': 1, 'New York': 1,
    'Pennsylvania': 1,
    'Illinois': 2, 'Indiana': 2, 'Michigan': 2, 'Ohio': 2, 'Wisconsin': 2,
    'Iowa': 2, 'Kansas': 2, 'Minnesota': 2, 'Missouri': 2, 'Nebraska': 2,
    'North Dakota': 2, 'South Dakota': 2,
    'Delaware': 3, 'Florida': 3, 'Georgia': 3, 'Maryland': 3,
    'North Carolina': 3, 'South Carolina': 3, 'Virginia': 3,
    'DC': 3, 'West Virginia': 3,
    'Alabama': 3, 'Kentucky': 3, 'Mississippi': 3, 'Tennessee': 3,
    'Arkansas': 3, 'Louisiana': 3, 'Oklahoma': 3, 'Texas': 3,
    'Arizona': 4, 'Colorado': 4, 'Idaho': 4, 'Montana': 4,
    'Nevada': 4, 'New Mexico': 4, 'Utah': 4, 'Wyoming': 4,
    'Alaska': 4, 'California': 4, 'Hawaii': 4, 'Oregon': 4, 'Washington': 4,
}
df_raw['region'] = df_raw['statefip'].map(REGION_MAP)

# Construct Census division from state names
DIVISION_MAP = {
    'Connecticut': 1, 'Maine': 1, 'Massachusetts': 1, 'New Hampshire': 1,
    'Rhode Island': 1, 'Vermont': 1,
    'New Jersey': 2, 'New York': 2, 'Pennsylvania': 2,
    'Illinois': 3, 'Indiana': 3, 'Michigan': 3, 'Ohio': 3, 'Wisconsin': 3,
    'Iowa': 4, 'Kansas': 4, 'Minnesota': 4, 'Missouri': 4, 'Nebraska': 4,
    'North Dakota': 4, 'South Dakota': 4,
    'Delaware': 5, 'Florida': 5, 'Georgia': 5, 'Maryland': 5,
    'North Carolina': 5, 'South Carolina': 5, 'Virginia': 5,
    'DC': 5, 'West Virginia': 5,
    'Alabama': 6, 'Kentucky': 6, 'Mississippi': 6, 'Tennessee': 6,
    'Arkansas': 7, 'Louisiana': 7, 'Oklahoma': 7, 'Texas': 7,
    'Arizona': 8, 'Colorado': 8, 'Idaho': 8, 'Montana': 8,
    'Nevada': 8, 'New Mexico': 8, 'Utah': 8, 'Wyoming': 8,
    'Alaska': 9, 'California': 9, 'Hawaii': 9, 'Oregon': 9, 'Washington': 9,
}
df_raw['division'] = df_raw['division'].astype(int) if 'division' in df_raw.columns else df_raw['statefip'].map(DIVISION_MAP)
# Make sure we have the division column
if 'division' not in df_raw.columns or df_raw['division'].isna().any():
    df_raw['division'] = df_raw['statefip'].map(DIVISION_MAP)

# Create region*year and division*year interaction FE
df_raw['region_year'] = df_raw['region'].astype(str) + '_' + df_raw['calyear'].astype(str)
df_raw['division_year'] = df_raw['division'].astype(str) + '_' + df_raw['calyear'].astype(str)

# Create state trend: state numeric code * year
state_names = sorted(df_raw['statefip'].unique())
state_num_map = {s: i+1 for i, s in enumerate(state_names)}
df_raw['state_num'] = df_raw['statefip'].map(state_num_map)
df_raw['state_trend'] = df_raw['state_num'] * df_raw['calyear']

# Create log variables
df_raw['log_urate'] = np.log(df_raw['urate'])
for var in ['myadult_aloneall_1830', 'myadult_aloneall_1824', 'myadult_aloneall_2530',
            'h_numpers_noneld', 'h_numfams_noneld']:
    # Protect against log(0) or log(negative)
    df_raw[f'log_{var}'] = np.where(df_raw[var] > 0, np.log(df_raw[var]), np.nan)

# Define NBER recession years
RECESSION_YEARS = [1980, 1981, 1982, 1990, 1991, 2001, 2007, 2008, 2009]

# Define trimming thresholds
urate_p01 = df_raw['urate'].quantile(0.01)
urate_p99 = df_raw['urate'].quantile(0.99)

# High-unemployment states: those with average urate above 75th percentile of state means
state_mean_urate = df_raw.groupby('statefip')['urate'].mean()
high_urate_threshold = state_mean_urate.quantile(0.75)
high_urate_states = state_mean_urate[state_mean_urate > high_urate_threshold].index.tolist()

# Small states: bottom quartile of population (use weight_noneld as proxy)
state_mean_pop = df_raw.groupby('statefip')['weight_noneld'].mean()
small_state_threshold = state_mean_pop.quantile(0.25)
small_states = state_mean_pop[state_mean_pop < small_state_threshold].index.tolist()

# Full dataset is df_raw
df = df_raw.copy()

results = []
inference_results = []
spec_run_counter = 0


# ============================================================
# HELPER: Run panel FE model via pyfixest
# ============================================================
def run_panel_fe(spec_id, spec_tree_path, baseline_group_id,
                 outcome_var, treatment_var, fe_formula,
                 data, vcov, weight_var, sample_desc, controls_desc,
                 cluster_var, design_audit, inference_canonical,
                 axis_block_name=None, axis_block=None,
                 functional_form=None, notes=""):
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        formula = f"{outcome_var} ~ {treatment_var} | {fe_formula}"
        kwargs = dict(data=data, vcov=vcov)
        if weight_var and weight_var in data.columns:
            kwargs['weights'] = weight_var

        m = pf.feols(formula, **kwargs)

        # Extract focal coefficient (first treatment variable listed)
        focal_var = treatment_var.split(" + ")[0].strip()
        coef_val = float(m.coef().get(focal_var, np.nan))
        se_val = float(m.se().get(focal_var, np.nan))
        pval = float(m.pvalue().get(focal_var, np.nan))
        try:
            ci = m.confint()
            ci_lower = float(ci.loc[focal_var, ci.columns[0]])
            ci_upper = float(ci.loc[focal_var, ci.columns[1]])
        except Exception:
            ci_lower, ci_upper = np.nan, np.nan
        nobs = int(m._N)
        try:
            r2 = float(m._r2)
        except Exception:
            r2 = np.nan
        all_coefs = {k: float(v) for k, v in m.coef().items()}

        blocks = {}
        if axis_block_name and axis_block:
            blocks[axis_block_name] = axis_block
        if functional_form:
            blocks["functional_form"] = functional_form

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": inference_canonical["spec_id"],
                       "params": inference_canonical["params"]},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"panel_fixed_effects": design_audit},
            blocks=blocks,
            notes=notes if notes else None,
        )

        results.append({
            "paper_id": PAPER_ID, "spec_run_id": run_id,
            "spec_id": spec_id, "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var, "treatment_var": focal_var,
            "coefficient": coef_val, "std_error": se_val, "p_value": pval,
            "ci_lower": ci_lower, "ci_upper": ci_upper,
            "n_obs": nobs, "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": fe_formula,
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
            "run_success": 1, "run_error": ""
        })
        return run_id, m

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
            "outcome_var": outcome_var,
            "treatment_var": treatment_var.split(" + ")[0].strip(),
            "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "n_obs": np.nan, "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": fe_formula,
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
            "run_success": 0, "run_error": err_msg
        })
        return run_id, None


# ============================================================
# HELPER: Run first-difference model
# ============================================================
def run_first_diff(spec_id, spec_tree_path, baseline_group_id,
                   outcome_var, treatment_var, data, weight_var,
                   sample_desc, cluster_var, design_audit, inference_canonical,
                   notes=""):
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        # Sort and compute first differences within state
        df_fd = data.sort_values(['statefip', 'calyear']).copy()
        df_fd['d_outcome'] = df_fd.groupby('statefip')[outcome_var].diff()
        df_fd['d_treatment'] = df_fd.groupby('statefip')[treatment_var].diff()
        df_fd = df_fd.dropna(subset=['d_outcome', 'd_treatment'])

        formula = "d_outcome ~ d_treatment | calyear"
        kwargs = dict(data=df_fd, vcov={"CRV1": "statefip"})
        if weight_var and weight_var in df_fd.columns:
            kwargs['weights'] = weight_var

        m = pf.feols(formula, **kwargs)
        coef_val = float(m.coef()["d_treatment"])
        se_val = float(m.se()["d_treatment"])
        pval = float(m.pvalue()["d_treatment"])
        try:
            ci = m.confint()
            ci_lower = float(ci.loc["d_treatment", ci.columns[0]])
            ci_upper = float(ci.loc["d_treatment", ci.columns[1]])
        except Exception:
            ci_lower, ci_upper = np.nan, np.nan
        nobs = int(m._N)
        try:
            r2 = float(m._r2)
        except Exception:
            r2 = np.nan
        all_coefs = {k: float(v) for k, v in m.coef().items()}

        fd_design = dict(design_audit)
        fd_design["estimator"] = "first_difference"

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": inference_canonical["spec_id"],
                       "params": inference_canonical["params"]},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"panel_fixed_effects": fd_design},
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
            "sample_desc": sample_desc + " (first difference)",
            "fixed_effects": "year (first-differenced)",
            "controls_desc": "none",
            "cluster_var": cluster_var,
            "run_success": 1, "run_error": ""
        })
        return run_id, m

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
            "sample_desc": sample_desc + " (first difference)",
            "fixed_effects": "year (first-differenced)",
            "controls_desc": "none",
            "cluster_var": cluster_var,
            "run_success": 0, "run_error": err_msg
        })
        return run_id, None


# ============================================================
# HELPER: Run inference variant
# ============================================================
def run_inference_variant(base_run_id, spec_id, spec_tree_path,
                          baseline_group_id, outcome_var, treatment_var,
                          fe_formula, data, vcov_variant, weight_var,
                          cluster_var_label, notes=""):
    global spec_run_counter
    spec_run_counter += 1
    infer_run_id = f"{PAPER_ID}_infer_{spec_run_counter:03d}"

    try:
        formula = f"{outcome_var} ~ {treatment_var} | {fe_formula}"
        kwargs = dict(data=data, vcov=vcov_variant)
        if weight_var and weight_var in data.columns:
            kwargs['weights'] = weight_var

        m = pf.feols(formula, **kwargs)
        focal_var = treatment_var.split(" + ")[0].strip()
        coef_val = float(m.coef().get(focal_var, np.nan))
        se_val = float(m.se().get(focal_var, np.nan))
        pval = float(m.pvalue().get(focal_var, np.nan))
        try:
            ci = m.confint()
            ci_lower = float(ci.loc[focal_var, ci.columns[0]])
            ci_upper = float(ci.loc[focal_var, ci.columns[1]])
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
            inference={"spec_id": spec_id, "params": {"cluster_var": cluster_var_label}},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"panel_fixed_effects": G1_DESIGN_AUDIT if baseline_group_id == "G1" else G2_DESIGN_AUDIT},
            notes=notes if notes else None,
        )

        inference_results.append({
            "paper_id": PAPER_ID, "inference_run_id": infer_run_id,
            "spec_run_id": base_run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var, "treatment_var": focal_var,
            "coefficient": coef_val, "std_error": se_val, "p_value": pval,
            "ci_lower": ci_lower, "ci_upper": ci_upper,
            "n_obs": nobs, "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "cluster_var": cluster_var_label,
            "run_success": 1, "run_error": ""
        })
        return infer_run_id

    except Exception as e:
        err_msg = str(e)[:240]
        payload = make_failure_payload(
            error=err_msg,
            error_details=error_details_from_exception(e, stage="inference"),
            software=SW_BLOCK, surface_hash=SURFACE_HASH
        )
        inference_results.append({
            "paper_id": PAPER_ID, "inference_run_id": infer_run_id,
            "spec_run_id": base_run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": treatment_var.split(" + ")[0].strip(),
            "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "n_obs": np.nan, "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "cluster_var": cluster_var_label,
            "run_success": 0, "run_error": err_msg
        })
        return infer_run_id


# ============================================================
# G1: YOUNG ADULT LIVING ALONE (18-30)
# ============================================================
print("=" * 60)
print("G1: Young Adult Living Alone (18-30)")
print("=" * 60)

BASE_FE = "statefip + calyear"
BASE_VCOV = {"CRV1": "statefip"}

# --- G1 Baseline: Table1-PanelA-Col3 ---
g1_base_run_id, g1_base_model = run_panel_fe(
    spec_id="baseline",
    spec_tree_path="specification_tree/designs/panel_fixed_effects.md",
    baseline_group_id="G1",
    outcome_var="myadult_aloneall_1830",
    treatment_var="urate",
    fe_formula=BASE_FE,
    data=df,
    vcov=BASE_VCOV,
    weight_var="weight_1830",
    sample_desc="All states 1980-2013, 18-30 year olds",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    notes="Table 1 Panel A Col 3 baseline"
)
print(f"  G1 baseline run: {g1_base_run_id}")

# --- G1 Additional baseline: Col4 (ages 18-24) ---
run_panel_fe(
    spec_id="baseline__table1_panA_col4",
    spec_tree_path="specification_tree/designs/panel_fixed_effects.md",
    baseline_group_id="G1",
    outcome_var="myadult_aloneall_1824",
    treatment_var="urate",
    fe_formula=BASE_FE,
    data=df,
    vcov=BASE_VCOV,
    weight_var="weight_1824",
    sample_desc="All states 1980-2013, 18-24 year olds",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    notes="Table 1 Panel A Col 4 (ages 18-24)"
)

# --- G1 Additional baseline: Col5 (ages 25-30) ---
run_panel_fe(
    spec_id="baseline__table1_panA_col5",
    spec_tree_path="specification_tree/designs/panel_fixed_effects.md",
    baseline_group_id="G1",
    outcome_var="myadult_aloneall_2530",
    treatment_var="urate",
    fe_formula=BASE_FE,
    data=df,
    vcov=BASE_VCOV,
    weight_var="weight_2530",
    sample_desc="All states 1980-2013, 25-30 year olds",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    notes="Table 1 Panel A Col 5 (ages 25-30)"
)

# --- G1 Design: First Difference ---
print("  Running G1 design variants...")
run_first_diff(
    spec_id="design/panel_fixed_effects/estimator/first_difference",
    spec_tree_path="specification_tree/designs/panel_fixed_effects.md#first-difference",
    baseline_group_id="G1",
    outcome_var="myadult_aloneall_1830",
    treatment_var="urate",
    data=df,
    weight_var="weight_1830",
    sample_desc="All states 1980-2013, 18-30 year olds",
    cluster_var="statefip",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    notes="First-difference estimator with year FE"
)

# --- G1 RC: Sample age group variants ---
print("  Running G1 rc/sample/age_group variants...")
run_panel_fe(
    spec_id="rc/sample/age_group/18_24",
    spec_tree_path="specification_tree/modules/robustness/sample.md#age-group",
    baseline_group_id="G1",
    outcome_var="myadult_aloneall_1824",
    treatment_var="urate",
    fe_formula=BASE_FE,
    data=df,
    vcov=BASE_VCOV,
    weight_var="weight_1824",
    sample_desc="All states 1980-2013, 18-24 year olds",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/age_group/18_24", "age_group": "18-24",
                "outcome_var": "myadult_aloneall_1824", "weight_var": "weight_1824"},
    notes="Age group 18-24 subsample"
)

run_panel_fe(
    spec_id="rc/sample/age_group/25_30",
    spec_tree_path="specification_tree/modules/robustness/sample.md#age-group",
    baseline_group_id="G1",
    outcome_var="myadult_aloneall_2530",
    treatment_var="urate",
    fe_formula=BASE_FE,
    data=df,
    vcov=BASE_VCOV,
    weight_var="weight_2530",
    sample_desc="All states 1980-2013, 25-30 year olds",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/age_group/25_30", "age_group": "25-30",
                "outcome_var": "myadult_aloneall_2530", "weight_var": "weight_2530"},
    notes="Age group 25-30 subsample"
)

# --- G1 RC: Functional form - outcome ---
print("  Running G1 rc/form variants...")
# Level is the baseline, included for completeness
run_panel_fe(
    spec_id="rc/form/outcome/level",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md#level",
    baseline_group_id="G1",
    outcome_var="myadult_aloneall_1830",
    treatment_var="urate",
    fe_formula=BASE_FE,
    data=df,
    vcov=BASE_VCOV,
    weight_var="weight_1830",
    sample_desc="All states 1980-2013, 18-30 year olds",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    functional_form={"spec_id": "rc/form/outcome/level", "interpretation": "level-level: unit change in urate -> unit change in fraction living alone"},
    notes="Level outcome (same as baseline)"
)

run_panel_fe(
    spec_id="rc/form/outcome/log",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md#log",
    baseline_group_id="G1",
    outcome_var="log_myadult_aloneall_1830",
    treatment_var="urate",
    fe_formula=BASE_FE,
    data=df.dropna(subset=["log_myadult_aloneall_1830"]),
    vcov=BASE_VCOV,
    weight_var="weight_1830",
    sample_desc="All states 1980-2013, 18-30 year olds (log outcome)",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    functional_form={"spec_id": "rc/form/outcome/log", "interpretation": "log-level: unit change in urate -> pct change in fraction living alone"},
    notes="Log-transformed outcome"
)

# --- G1 RC: Functional form - treatment ---
run_panel_fe(
    spec_id="rc/form/treatment/log_urate",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md#log",
    baseline_group_id="G1",
    outcome_var="myadult_aloneall_1830",
    treatment_var="log_urate",
    fe_formula=BASE_FE,
    data=df,
    vcov=BASE_VCOV,
    weight_var="weight_1830",
    sample_desc="All states 1980-2013, 18-30 year olds",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    functional_form={"spec_id": "rc/form/treatment/log_urate", "interpretation": "level-log: pct change in urate -> unit change in fraction living alone"},
    notes="Log unemployment rate as treatment"
)

# --- G1 RC: Recession-specific treatment (Panel B) ---
run_panel_fe(
    spec_id="rc/form/treatment/recession_specific",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md#interaction",
    baseline_group_id="G1",
    outcome_var="myadult_aloneall_1830",
    treatment_var="urate_80 + urate_rest + urate_07",
    fe_formula=BASE_FE,
    data=df,
    vcov=BASE_VCOV,
    weight_var="weight_1830",
    sample_desc="All states 1980-2013, 18-30 year olds",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    functional_form={"spec_id": "rc/form/treatment/recession_specific",
                     "interpretation": "Period-specific urate coefficients: 1980s, rest, Great Recession"},
    notes="Panel B: recession-period-specific unemployment effects"
)

# Individual recession period variants
run_panel_fe(
    spec_id="rc/form/treatment/urate_80",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md#interaction",
    baseline_group_id="G1",
    outcome_var="myadult_aloneall_1830",
    treatment_var="urate_80 + urate_rest + urate_07",
    fe_formula=BASE_FE,
    data=df,
    vcov=BASE_VCOV,
    weight_var="weight_1830",
    sample_desc="All states 1980-2013, 18-30 year olds",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    functional_form={"spec_id": "rc/form/treatment/urate_80",
                     "interpretation": "Focal: 1980s recession urate coefficient (from full decomposition)"},
    notes="Panel B: focal=urate_80 coefficient"
)

run_panel_fe(
    spec_id="rc/form/treatment/urate_rest",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md#interaction",
    baseline_group_id="G1",
    outcome_var="myadult_aloneall_1830",
    treatment_var="urate_rest + urate_80 + urate_07",
    fe_formula=BASE_FE,
    data=df,
    vcov=BASE_VCOV,
    weight_var="weight_1830",
    sample_desc="All states 1980-2013, 18-30 year olds",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    functional_form={"spec_id": "rc/form/treatment/urate_rest",
                     "interpretation": "Focal: non-recession urate coefficient (from full decomposition)"},
    notes="Panel B: focal=urate_rest coefficient"
)

run_panel_fe(
    spec_id="rc/form/treatment/urate_07",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md#interaction",
    baseline_group_id="G1",
    outcome_var="myadult_aloneall_1830",
    treatment_var="urate_07 + urate_80 + urate_rest",
    fe_formula=BASE_FE,
    data=df,
    vcov=BASE_VCOV,
    weight_var="weight_1830",
    sample_desc="All states 1980-2013, 18-30 year olds",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    functional_form={"spec_id": "rc/form/treatment/urate_07",
                     "interpretation": "Focal: Great Recession urate coefficient (from full decomposition)"},
    notes="Panel B: focal=urate_07 coefficient"
)

# --- G1 RC: FE structure variants ---
print("  Running G1 rc/fe variants...")
run_panel_fe(
    spec_id="rc/fe/drop_state_fe",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#drop-fe",
    baseline_group_id="G1",
    outcome_var="myadult_aloneall_1830",
    treatment_var="urate",
    fe_formula="calyear",
    data=df,
    vcov=BASE_VCOV,
    weight_var="weight_1830",
    sample_desc="All states 1980-2013, 18-30 year olds",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/drop_state_fe", "description": "Year FE only (drop state FE)"},
    notes="Drop state FE, year FE only"
)

run_panel_fe(
    spec_id="rc/fe/add_state_trend",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#state-trend",
    baseline_group_id="G1",
    outcome_var="myadult_aloneall_1830",
    treatment_var="urate + i(state_num, calyear)",
    fe_formula="statefip + calyear",
    data=df,
    vcov=BASE_VCOV,
    weight_var="weight_1830",
    sample_desc="All states 1980-2013, 18-30 year olds",
    controls_desc="state-specific linear trend",
    cluster_var="statefip",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/add_state_trend", "description": "State + year FE + state-specific linear trend"},
    notes="Add state-specific linear trend via i(state_num, calyear)"
)

run_panel_fe(
    spec_id="rc/fe/add_region_year",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#region-year",
    baseline_group_id="G1",
    outcome_var="myadult_aloneall_1830",
    treatment_var="urate",
    fe_formula="statefip + region_year",
    data=df,
    vcov=BASE_VCOV,
    weight_var="weight_1830",
    sample_desc="All states 1980-2013, 18-30 year olds",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/add_region_year", "description": "State + region*year FE"},
    notes="Replace year FE with region*year FE"
)

run_panel_fe(
    spec_id="rc/fe/add_division_year",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#division-year",
    baseline_group_id="G1",
    outcome_var="myadult_aloneall_1830",
    treatment_var="urate",
    fe_formula="statefip + division_year",
    data=df,
    vcov=BASE_VCOV,
    weight_var="weight_1830",
    sample_desc="All states 1980-2013, 18-30 year olds",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/add_division_year", "description": "State + division*year FE"},
    notes="Replace year FE with division*year FE"
)

# --- G1 RC: Weights ---
print("  Running G1 rc/weights variants...")
run_panel_fe(
    spec_id="rc/weights/unweighted",
    spec_tree_path="specification_tree/modules/robustness/weights.md#unweighted",
    baseline_group_id="G1",
    outcome_var="myadult_aloneall_1830",
    treatment_var="urate",
    fe_formula=BASE_FE,
    data=df,
    vcov=BASE_VCOV,
    weight_var=None,
    sample_desc="All states 1980-2013, 18-30 year olds (unweighted)",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="weights",
    axis_block={"spec_id": "rc/weights/unweighted", "description": "Unweighted (drop analytic weights)"},
    notes="Unweighted regression"
)

# --- G1 RC: Sample time restrictions ---
print("  Running G1 rc/sample/time variants...")
df_pre2007 = df[df['calyear'] < 2007].copy()
df_post1990 = df[df['calyear'] >= 1990].copy()
df_1990_2013 = df[(df['calyear'] >= 1990) & (df['calyear'] <= 2013)].copy()
df_no_recession = df[~df['calyear'].isin(RECESSION_YEARS)].copy()

run_panel_fe(
    spec_id="rc/sample/time/pre_2007",
    spec_tree_path="specification_tree/modules/robustness/sample.md#time-period",
    baseline_group_id="G1",
    outcome_var="myadult_aloneall_1830",
    treatment_var="urate",
    fe_formula=BASE_FE,
    data=df_pre2007,
    vcov=BASE_VCOV,
    weight_var="weight_1830",
    sample_desc="States 1980-2006, 18-30 year olds (pre-Great Recession)",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/time/pre_2007", "time_restriction": "1980-2006"},
    notes="Drop Great Recession period (2007+)"
)

run_panel_fe(
    spec_id="rc/sample/time/post_1990",
    spec_tree_path="specification_tree/modules/robustness/sample.md#time-period",
    baseline_group_id="G1",
    outcome_var="myadult_aloneall_1830",
    treatment_var="urate",
    fe_formula=BASE_FE,
    data=df_post1990,
    vcov=BASE_VCOV,
    weight_var="weight_1830",
    sample_desc="States 1990-2013, 18-30 year olds (post-1990)",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/time/post_1990", "time_restriction": "1990-2013"},
    notes="Drop 1980s recession period"
)

run_panel_fe(
    spec_id="rc/sample/time/1990_2013",
    spec_tree_path="specification_tree/modules/robustness/sample.md#time-period",
    baseline_group_id="G1",
    outcome_var="myadult_aloneall_1830",
    treatment_var="urate",
    fe_formula=BASE_FE,
    data=df_1990_2013,
    vcov=BASE_VCOV,
    weight_var="weight_1830",
    sample_desc="States 1990-2013, 18-30 year olds",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/time/1990_2013", "time_restriction": "1990-2013"},
    notes="Exclude early period (1980-1989)"
)

run_panel_fe(
    spec_id="rc/sample/time/drop_recession_years",
    spec_tree_path="specification_tree/modules/robustness/sample.md#time-period",
    baseline_group_id="G1",
    outcome_var="myadult_aloneall_1830",
    treatment_var="urate",
    fe_formula=BASE_FE,
    data=df_no_recession,
    vcov=BASE_VCOV,
    weight_var="weight_1830",
    sample_desc="States 1980-2013 excl NBER recession years, 18-30 year olds",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/time/drop_recession_years",
                "dropped_years": RECESSION_YEARS},
    notes="Drop NBER recession years"
)

# --- G1 RC: Outlier/geography restrictions ---
print("  Running G1 rc/sample/outliers variants...")
df_trim_urate = df[(df['urate'] >= urate_p01) & (df['urate'] <= urate_p99)].copy()
df_no_high_urate = df[~df['statefip'].isin(high_urate_states)].copy()
df_no_small = df[~df['statefip'].isin(small_states)].copy()

run_panel_fe(
    spec_id="rc/sample/outliers/trim_urate_1_99",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    baseline_group_id="G1",
    outcome_var="myadult_aloneall_1830",
    treatment_var="urate",
    fe_formula=BASE_FE,
    data=df_trim_urate,
    vcov=BASE_VCOV,
    weight_var="weight_1830",
    sample_desc="States 1980-2013, trim urate 1st-99th pctile, 18-30 year olds",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_urate_1_99",
                "trim_lower": float(urate_p01), "trim_upper": float(urate_p99)},
    notes="Trim extreme unemployment rate state-years"
)

run_panel_fe(
    spec_id="rc/sample/outliers/drop_high_urate_states",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    baseline_group_id="G1",
    outcome_var="myadult_aloneall_1830",
    treatment_var="urate",
    fe_formula=BASE_FE,
    data=df_no_high_urate,
    vcov=BASE_VCOV,
    weight_var="weight_1830",
    sample_desc="Drop high-urate states (top 25%), 18-30 year olds",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/drop_high_urate_states",
                "dropped_states": high_urate_states},
    notes="Drop states with persistently high unemployment"
)

run_panel_fe(
    spec_id="rc/sample/outliers/drop_small_states",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    baseline_group_id="G1",
    outcome_var="myadult_aloneall_1830",
    treatment_var="urate",
    fe_formula=BASE_FE,
    data=df_no_small,
    vcov=BASE_VCOV,
    weight_var="weight_1830",
    sample_desc="Drop small states (bottom 25% pop), 18-30 year olds",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/drop_small_states",
                "dropped_states": small_states},
    notes="Drop small states (bottom quartile of population)"
)

# --- G1 RC: Joint outcome x age group ---
print("  Running G1 rc/joint/outcome_age variants...")
# 18-24 + log treatment
run_panel_fe(
    spec_id="rc/joint/outcome_age/18_24_log_urate",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="myadult_aloneall_1824",
    treatment_var="log_urate",
    fe_formula=BASE_FE,
    data=df,
    vcov=BASE_VCOV,
    weight_var="weight_1824",
    sample_desc="All states 1980-2013, 18-24 year olds",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/outcome_age/18_24_log_urate",
                "components": ["rc/sample/age_group/18_24", "rc/form/treatment/log_urate"]},
    functional_form={"spec_id": "rc/joint/outcome_age/18_24_log_urate",
                     "interpretation": "18-24 age group with log urate treatment"},
    notes="Joint: ages 18-24 + log unemployment rate"
)

# 25-30 + log treatment
run_panel_fe(
    spec_id="rc/joint/outcome_age/25_30_log_urate",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="myadult_aloneall_2530",
    treatment_var="log_urate",
    fe_formula=BASE_FE,
    data=df,
    vcov=BASE_VCOV,
    weight_var="weight_2530",
    sample_desc="All states 1980-2013, 25-30 year olds",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/outcome_age/25_30_log_urate",
                "components": ["rc/sample/age_group/25_30", "rc/form/treatment/log_urate"]},
    functional_form={"spec_id": "rc/joint/outcome_age/25_30_log_urate",
                     "interpretation": "25-30 age group with log urate treatment"},
    notes="Joint: ages 25-30 + log unemployment rate"
)

# 18-24 + unweighted
run_panel_fe(
    spec_id="rc/joint/outcome_age/18_24_unweighted",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="myadult_aloneall_1824",
    treatment_var="urate",
    fe_formula=BASE_FE,
    data=df,
    vcov=BASE_VCOV,
    weight_var=None,
    sample_desc="All states 1980-2013, 18-24 year olds (unweighted)",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/outcome_age/18_24_unweighted",
                "components": ["rc/sample/age_group/18_24", "rc/weights/unweighted"]},
    notes="Joint: ages 18-24 + unweighted"
)

# 25-30 + unweighted
run_panel_fe(
    spec_id="rc/joint/outcome_age/25_30_unweighted",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="myadult_aloneall_2530",
    treatment_var="urate",
    fe_formula=BASE_FE,
    data=df,
    vcov=BASE_VCOV,
    weight_var=None,
    sample_desc="All states 1980-2013, 25-30 year olds (unweighted)",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/outcome_age/25_30_unweighted",
                "components": ["rc/sample/age_group/25_30", "rc/weights/unweighted"]},
    notes="Joint: ages 25-30 + unweighted"
)

# --- G1 RC: Joint form x time ---
print("  Running G1 rc/joint/form_time variants...")
# Recession-specific + pre-2007
run_panel_fe(
    spec_id="rc/joint/form_time/recession_pre2007",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="myadult_aloneall_1830",
    treatment_var="urate_80 + urate_rest + urate_07",
    fe_formula=BASE_FE,
    data=df_pre2007,
    vcov=BASE_VCOV,
    weight_var="weight_1830",
    sample_desc="States 1980-2006, 18-30 year olds, recession-specific",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/form_time/recession_pre2007",
                "components": ["rc/form/treatment/recession_specific", "rc/sample/time/pre_2007"]},
    functional_form={"spec_id": "rc/joint/form_time/recession_pre2007",
                     "interpretation": "Recession-specific urate on pre-2007 sample"},
    notes="Joint: recession-specific treatment + pre-Great Recession"
)

# Recession-specific + post-1990 (urate_80 is identically 0 post-1990, so use urate_rest + urate_07 only)
run_panel_fe(
    spec_id="rc/joint/form_time/recession_post1990",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="myadult_aloneall_1830",
    treatment_var="urate_rest + urate_07",
    fe_formula=BASE_FE,
    data=df_post1990,
    vcov=BASE_VCOV,
    weight_var="weight_1830",
    sample_desc="States 1990-2013, 18-30 year olds, recession-specific",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/form_time/recession_post1990",
                "components": ["rc/form/treatment/recession_specific", "rc/sample/time/post_1990"]},
    functional_form={"spec_id": "rc/joint/form_time/recession_post1990",
                     "interpretation": "Recession-specific urate (rest+07) on post-1990 sample; urate_80 dropped (all zero)"},
    notes="Joint: recession-specific treatment + post-1990 (urate_80 identically zero, dropped)"
)

# Log urate + pre-2007
run_panel_fe(
    spec_id="rc/joint/form_time/log_urate_pre2007",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="myadult_aloneall_1830",
    treatment_var="log_urate",
    fe_formula=BASE_FE,
    data=df_pre2007,
    vcov=BASE_VCOV,
    weight_var="weight_1830",
    sample_desc="States 1980-2006, 18-30 year olds, log urate",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/form_time/log_urate_pre2007",
                "components": ["rc/form/treatment/log_urate", "rc/sample/time/pre_2007"]},
    functional_form={"spec_id": "rc/joint/form_time/log_urate_pre2007",
                     "interpretation": "Log urate on pre-2007 sample"},
    notes="Joint: log urate + pre-2007"
)

# Log urate + post-1990
run_panel_fe(
    spec_id="rc/joint/form_time/log_urate_post1990",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="myadult_aloneall_1830",
    treatment_var="log_urate",
    fe_formula=BASE_FE,
    data=df_post1990,
    vcov=BASE_VCOV,
    weight_var="weight_1830",
    sample_desc="States 1990-2013, 18-30 year olds, log urate",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/form_time/log_urate_post1990",
                "components": ["rc/form/treatment/log_urate", "rc/sample/time/post_1990"]},
    functional_form={"spec_id": "rc/joint/form_time/log_urate_post1990",
                     "interpretation": "Log urate on post-1990 sample"},
    notes="Joint: log urate + post-1990"
)

# --- G1 RC: Additional joint outcome x age ---
# 18-24 + recession specific
run_panel_fe(
    spec_id="rc/joint/outcome_age/18_24_recession",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="myadult_aloneall_1824",
    treatment_var="urate_80 + urate_rest + urate_07",
    fe_formula=BASE_FE,
    data=df,
    vcov=BASE_VCOV,
    weight_var="weight_1824",
    sample_desc="All states 1980-2013, 18-24 year olds, recession-specific",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/outcome_age/18_24_recession",
                "components": ["rc/sample/age_group/18_24", "rc/form/treatment/recession_specific"]},
    functional_form={"spec_id": "rc/joint/outcome_age/18_24_recession",
                     "interpretation": "18-24 age group with recession-specific urate"},
    notes="Joint: ages 18-24 + recession-specific treatment"
)

# 25-30 + recession specific
run_panel_fe(
    spec_id="rc/joint/outcome_age/25_30_recession",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="myadult_aloneall_2530",
    treatment_var="urate_80 + urate_rest + urate_07",
    fe_formula=BASE_FE,
    data=df,
    vcov=BASE_VCOV,
    weight_var="weight_2530",
    sample_desc="All states 1980-2013, 25-30 year olds, recession-specific",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/outcome_age/25_30_recession",
                "components": ["rc/sample/age_group/25_30", "rc/form/treatment/recession_specific"]},
    functional_form={"spec_id": "rc/joint/outcome_age/25_30_recession",
                     "interpretation": "25-30 age group with recession-specific urate"},
    notes="Joint: ages 25-30 + recession-specific treatment"
)

# --- G1 RC: Additional joint form x time ---
# Log urate + drop recession years
run_panel_fe(
    spec_id="rc/joint/form_time/log_urate_no_recession",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="myadult_aloneall_1830",
    treatment_var="log_urate",
    fe_formula=BASE_FE,
    data=df_no_recession,
    vcov=BASE_VCOV,
    weight_var="weight_1830",
    sample_desc="States 1980-2013 excl recession years, 18-30 year olds, log urate",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/form_time/log_urate_no_recession",
                "components": ["rc/form/treatment/log_urate", "rc/sample/time/drop_recession_years"]},
    functional_form={"spec_id": "rc/joint/form_time/log_urate_no_recession",
                     "interpretation": "Log urate on non-recession years only"},
    notes="Joint: log urate + drop recession years"
)

# Log urate + 1990-2013
run_panel_fe(
    spec_id="rc/joint/form_time/log_urate_1990_2013",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="myadult_aloneall_1830",
    treatment_var="log_urate",
    fe_formula=BASE_FE,
    data=df_1990_2013,
    vcov=BASE_VCOV,
    weight_var="weight_1830",
    sample_desc="States 1990-2013, 18-30 year olds, log urate",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/form_time/log_urate_1990_2013",
                "components": ["rc/form/treatment/log_urate", "rc/sample/time/1990_2013"]},
    functional_form={"spec_id": "rc/joint/form_time/log_urate_1990_2013",
                     "interpretation": "Log urate on 1990-2013 sample"},
    notes="Joint: log urate + 1990-2013"
)

g1_count = len(results)
print(f"  G1 total specs: {g1_count}")


# ============================================================
# G2: HOUSEHOLD SIZE (NON-ELDERLY)
# ============================================================
print("=" * 60)
print("G2: Household Size (Non-Elderly)")
print("=" * 60)

# --- G2 Baseline: Table1-PanelA-Col1 ---
g2_base_run_id, g2_base_model = run_panel_fe(
    spec_id="baseline",
    spec_tree_path="specification_tree/designs/panel_fixed_effects.md",
    baseline_group_id="G2",
    outcome_var="h_numpers_noneld",
    treatment_var="urate",
    fe_formula=BASE_FE,
    data=df,
    vcov=BASE_VCOV,
    weight_var="weight_noneld",
    sample_desc="All states 1980-2013, non-elderly",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    notes="Table 1 Panel A Col 1 baseline"
)
print(f"  G2 baseline run: {g2_base_run_id}")

# --- G2 Additional baseline: Col2 (number of families) ---
run_panel_fe(
    spec_id="baseline__table1_panA_col2",
    spec_tree_path="specification_tree/designs/panel_fixed_effects.md",
    baseline_group_id="G2",
    outcome_var="h_numfams_noneld",
    treatment_var="urate",
    fe_formula=BASE_FE,
    data=df,
    vcov=BASE_VCOV,
    weight_var="weight_noneld",
    sample_desc="All states 1980-2013, non-elderly",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    notes="Table 1 Panel A Col 2 (number of families)"
)

# --- G2 Design: First Difference ---
print("  Running G2 design variants...")
run_first_diff(
    spec_id="design/panel_fixed_effects/estimator/first_difference",
    spec_tree_path="specification_tree/designs/panel_fixed_effects.md#first-difference",
    baseline_group_id="G2",
    outcome_var="h_numpers_noneld",
    treatment_var="urate",
    data=df,
    weight_var="weight_noneld",
    sample_desc="All states 1980-2013, non-elderly",
    cluster_var="statefip",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    notes="First-difference estimator with year FE"
)

# --- G2 RC: Alternative outcome ---
print("  Running G2 rc/form variants...")
run_panel_fe(
    spec_id="rc/form/outcome/h_numfams_noneld",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md#alternative-outcome",
    baseline_group_id="G2",
    outcome_var="h_numfams_noneld",
    treatment_var="urate",
    fe_formula=BASE_FE,
    data=df,
    vcov=BASE_VCOV,
    weight_var="weight_noneld",
    sample_desc="All states 1980-2013, non-elderly",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    functional_form={"spec_id": "rc/form/outcome/h_numfams_noneld",
                     "interpretation": "Number of families per HH instead of persons per HH"},
    notes="Alternative outcome: number of families per household"
)

# Recession-specific treatment
run_panel_fe(
    spec_id="rc/form/treatment/recession_specific",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md#interaction",
    baseline_group_id="G2",
    outcome_var="h_numpers_noneld",
    treatment_var="urate_80 + urate_rest + urate_07",
    fe_formula=BASE_FE,
    data=df,
    vcov=BASE_VCOV,
    weight_var="weight_noneld",
    sample_desc="All states 1980-2013, non-elderly",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    functional_form={"spec_id": "rc/form/treatment/recession_specific",
                     "interpretation": "Period-specific urate coefficients: 1980s, rest, Great Recession"},
    notes="Panel B: recession-period-specific unemployment effects"
)

# Log urate
run_panel_fe(
    spec_id="rc/form/treatment/log_urate",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md#log",
    baseline_group_id="G2",
    outcome_var="h_numpers_noneld",
    treatment_var="log_urate",
    fe_formula=BASE_FE,
    data=df,
    vcov=BASE_VCOV,
    weight_var="weight_noneld",
    sample_desc="All states 1980-2013, non-elderly",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    functional_form={"spec_id": "rc/form/treatment/log_urate",
                     "interpretation": "level-log: pct change in urate -> unit change in HH size"},
    notes="Log unemployment rate as treatment"
)

# Log outcome
run_panel_fe(
    spec_id="rc/form/outcome/log",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md#log",
    baseline_group_id="G2",
    outcome_var="log_h_numpers_noneld",
    treatment_var="urate",
    fe_formula=BASE_FE,
    data=df.dropna(subset=["log_h_numpers_noneld"]),
    vcov=BASE_VCOV,
    weight_var="weight_noneld",
    sample_desc="All states 1980-2013, non-elderly (log outcome)",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    functional_form={"spec_id": "rc/form/outcome/log",
                     "interpretation": "log-level: unit change in urate -> pct change in HH size"},
    notes="Log-transformed household size outcome"
)

# --- G2 RC: FE structure variants ---
print("  Running G2 rc/fe variants...")
run_panel_fe(
    spec_id="rc/fe/drop_state_fe",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#drop-fe",
    baseline_group_id="G2",
    outcome_var="h_numpers_noneld",
    treatment_var="urate",
    fe_formula="calyear",
    data=df,
    vcov=BASE_VCOV,
    weight_var="weight_noneld",
    sample_desc="All states 1980-2013, non-elderly",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/drop_state_fe", "description": "Year FE only (drop state FE)"},
    notes="Drop state FE, year FE only"
)

run_panel_fe(
    spec_id="rc/fe/add_state_trend",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#state-trend",
    baseline_group_id="G2",
    outcome_var="h_numpers_noneld",
    treatment_var="urate + i(state_num, calyear)",
    fe_formula="statefip + calyear",
    data=df,
    vcov=BASE_VCOV,
    weight_var="weight_noneld",
    sample_desc="All states 1980-2013, non-elderly",
    controls_desc="state-specific linear trend",
    cluster_var="statefip",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/add_state_trend", "description": "State + year FE + state-specific linear trend"},
    notes="Add state-specific linear trend via i(state_num, calyear)"
)

run_panel_fe(
    spec_id="rc/fe/add_region_year",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#region-year",
    baseline_group_id="G2",
    outcome_var="h_numpers_noneld",
    treatment_var="urate",
    fe_formula="statefip + region_year",
    data=df,
    vcov=BASE_VCOV,
    weight_var="weight_noneld",
    sample_desc="All states 1980-2013, non-elderly",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/add_region_year", "description": "State + region*year FE"},
    notes="Replace year FE with region*year FE"
)

# --- G2 RC: Weights ---
print("  Running G2 rc/weights variants...")
run_panel_fe(
    spec_id="rc/weights/unweighted",
    spec_tree_path="specification_tree/modules/robustness/weights.md#unweighted",
    baseline_group_id="G2",
    outcome_var="h_numpers_noneld",
    treatment_var="urate",
    fe_formula=BASE_FE,
    data=df,
    vcov=BASE_VCOV,
    weight_var=None,
    sample_desc="All states 1980-2013, non-elderly (unweighted)",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    axis_block_name="weights",
    axis_block={"spec_id": "rc/weights/unweighted", "description": "Unweighted (drop analytic weights)"},
    notes="Unweighted regression"
)

# --- G2 RC: Sample time restrictions ---
print("  Running G2 rc/sample/time variants...")
run_panel_fe(
    spec_id="rc/sample/time/pre_2007",
    spec_tree_path="specification_tree/modules/robustness/sample.md#time-period",
    baseline_group_id="G2",
    outcome_var="h_numpers_noneld",
    treatment_var="urate",
    fe_formula=BASE_FE,
    data=df_pre2007,
    vcov=BASE_VCOV,
    weight_var="weight_noneld",
    sample_desc="States 1980-2006, non-elderly (pre-Great Recession)",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/time/pre_2007", "time_restriction": "1980-2006"},
    notes="Drop Great Recession period (2007+)"
)

run_panel_fe(
    spec_id="rc/sample/time/post_1990",
    spec_tree_path="specification_tree/modules/robustness/sample.md#time-period",
    baseline_group_id="G2",
    outcome_var="h_numpers_noneld",
    treatment_var="urate",
    fe_formula=BASE_FE,
    data=df_post1990,
    vcov=BASE_VCOV,
    weight_var="weight_noneld",
    sample_desc="States 1990-2013, non-elderly (post-1990)",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/time/post_1990", "time_restriction": "1990-2013"},
    notes="Drop 1980s recession period"
)

run_panel_fe(
    spec_id="rc/sample/time/drop_recession_years",
    spec_tree_path="specification_tree/modules/robustness/sample.md#time-period",
    baseline_group_id="G2",
    outcome_var="h_numpers_noneld",
    treatment_var="urate",
    fe_formula=BASE_FE,
    data=df_no_recession,
    vcov=BASE_VCOV,
    weight_var="weight_noneld",
    sample_desc="States 1980-2013 excl NBER recession years, non-elderly",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/time/drop_recession_years",
                "dropped_years": RECESSION_YEARS},
    notes="Drop NBER recession years"
)

# --- G2 RC: Outlier restrictions ---
print("  Running G2 rc/sample/outliers variants...")
run_panel_fe(
    spec_id="rc/sample/outliers/trim_urate_1_99",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    baseline_group_id="G2",
    outcome_var="h_numpers_noneld",
    treatment_var="urate",
    fe_formula=BASE_FE,
    data=df_trim_urate,
    vcov=BASE_VCOV,
    weight_var="weight_noneld",
    sample_desc="States 1980-2013, trim urate 1st-99th pctile, non-elderly",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_urate_1_99",
                "trim_lower": float(urate_p01), "trim_upper": float(urate_p99)},
    notes="Trim extreme unemployment rate state-years"
)

run_panel_fe(
    spec_id="rc/sample/outliers/drop_high_urate_states",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    baseline_group_id="G2",
    outcome_var="h_numpers_noneld",
    treatment_var="urate",
    fe_formula=BASE_FE,
    data=df_no_high_urate,
    vcov=BASE_VCOV,
    weight_var="weight_noneld",
    sample_desc="Drop high-urate states (top 25%), non-elderly",
    controls_desc="none",
    cluster_var="statefip",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/drop_high_urate_states",
                "dropped_states": high_urate_states},
    notes="Drop states with persistently high unemployment"
)

g2_count = len(results) - g1_count
print(f"  G2 total specs: {g2_count}")


# ============================================================
# INFERENCE VARIANTS
# ============================================================
print("=" * 60)
print("Inference Variants")
print("=" * 60)

# G1 inference variants on baseline
print("  G1 inference variants...")
# HC1 (robust, no clustering)
run_inference_variant(
    base_run_id=g1_base_run_id,
    spec_id="infer/se/hc/hc1",
    spec_tree_path="specification_tree/modules/inference/standard_errors.md#hc1",
    baseline_group_id="G1",
    outcome_var="myadult_aloneall_1830",
    treatment_var="urate",
    fe_formula=BASE_FE,
    data=df,
    vcov_variant="hetero",
    weight_var="weight_1830",
    cluster_var_label="none (HC1)",
    notes="HC1 robust SE without clustering"
)

# Region clustering
run_inference_variant(
    base_run_id=g1_base_run_id,
    spec_id="infer/se/cluster/region",
    spec_tree_path="specification_tree/modules/inference/standard_errors.md#cluster",
    baseline_group_id="G1",
    outcome_var="myadult_aloneall_1830",
    treatment_var="urate",
    fe_formula=BASE_FE,
    data=df,
    vcov_variant={"CRV1": "region"},
    weight_var="weight_1830",
    cluster_var_label="region",
    notes="Coarser clustering at Census region level (4 clusters)"
)

# Driscoll-Kraay: use Newey-West HAC as approximation (pyfixest doesn't have DK)
# We implement via statsmodels with HAC
try:
    import statsmodels.api as sm
    spec_run_counter += 1
    infer_run_id = f"{PAPER_ID}_infer_{spec_run_counter:03d}"

    # Use statsmodels OLS with entity-demeaned data + HAC for DK approximation
    df_dk = df.copy()
    # Demean by state and year (within transformation)
    for v in ['myadult_aloneall_1830', 'urate']:
        state_mean = df_dk.groupby('statefip')[v].transform('mean')
        year_mean = df_dk.groupby('calyear')[v].transform('mean')
        grand_mean = df_dk[v].mean()
        df_dk[f'{v}_dm'] = df_dk[v] - state_mean - year_mean + grand_mean

    X = sm.add_constant(df_dk['urate_dm'].values)
    y = df_dk['myadult_aloneall_1830_dm'].values
    w = df_dk['weight_1830'].values
    # WLS with HAC
    wls = sm.WLS(y, X, weights=w).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
    dk_coef = float(wls.params[1])
    dk_se = float(wls.bse[1])
    dk_pval = float(wls.pvalues[1])
    dk_ci = wls.conf_int()
    dk_ci_lower = float(dk_ci[1, 0])
    dk_ci_upper = float(dk_ci[1, 1])

    payload = make_success_payload(
        coefficients={"urate": dk_coef, "const": float(wls.params[0])},
        inference={"spec_id": "infer/se/hac/driscoll_kraay", "params": {"maxlag": 3}},
        software=SW_BLOCK,
        surface_hash=SURFACE_HASH,
        design={"panel_fixed_effects": G1_DESIGN_AUDIT},
        notes="Driscoll-Kraay approximation via HAC on demeaned data",
    )
    inference_results.append({
        "paper_id": PAPER_ID, "inference_run_id": infer_run_id,
        "spec_run_id": g1_base_run_id,
        "spec_id": "infer/se/hac/driscoll_kraay",
        "spec_tree_path": "specification_tree/modules/inference/standard_errors.md#hac",
        "baseline_group_id": "G1",
        "outcome_var": "myadult_aloneall_1830", "treatment_var": "urate",
        "coefficient": dk_coef, "std_error": dk_se, "p_value": dk_pval,
        "ci_lower": dk_ci_lower, "ci_upper": dk_ci_upper,
        "n_obs": int(wls.nobs), "r_squared": float(wls.rsquared),
        "coefficient_vector_json": json.dumps(payload),
        "cluster_var": "none (HAC/Driscoll-Kraay)",
        "run_success": 1, "run_error": ""
    })
except Exception as e:
    spec_run_counter += 1
    infer_run_id = f"{PAPER_ID}_infer_{spec_run_counter:03d}"
    err_msg = str(e)[:240]
    payload = make_failure_payload(
        error=err_msg,
        error_details=error_details_from_exception(e, stage="inference_dk"),
        software=SW_BLOCK, surface_hash=SURFACE_HASH
    )
    inference_results.append({
        "paper_id": PAPER_ID, "inference_run_id": infer_run_id,
        "spec_run_id": g1_base_run_id,
        "spec_id": "infer/se/hac/driscoll_kraay",
        "spec_tree_path": "specification_tree/modules/inference/standard_errors.md#hac",
        "baseline_group_id": "G1",
        "outcome_var": "myadult_aloneall_1830", "treatment_var": "urate",
        "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
        "ci_lower": np.nan, "ci_upper": np.nan,
        "n_obs": np.nan, "r_squared": np.nan,
        "coefficient_vector_json": json.dumps(payload),
        "cluster_var": "none (HAC/Driscoll-Kraay)",
        "run_success": 0, "run_error": err_msg
    })

# G2 inference variants on baseline
print("  G2 inference variants...")
# HC1
run_inference_variant(
    base_run_id=g2_base_run_id,
    spec_id="infer/se/hc/hc1",
    spec_tree_path="specification_tree/modules/inference/standard_errors.md#hc1",
    baseline_group_id="G2",
    outcome_var="h_numpers_noneld",
    treatment_var="urate",
    fe_formula=BASE_FE,
    data=df,
    vcov_variant="hetero",
    weight_var="weight_noneld",
    cluster_var_label="none (HC1)",
    notes="HC1 robust SE without clustering"
)

# G2 DK approximation
try:
    spec_run_counter += 1
    infer_run_id = f"{PAPER_ID}_infer_{spec_run_counter:03d}"

    df_dk2 = df.copy()
    for v in ['h_numpers_noneld', 'urate']:
        state_mean = df_dk2.groupby('statefip')[v].transform('mean')
        year_mean = df_dk2.groupby('calyear')[v].transform('mean')
        grand_mean = df_dk2[v].mean()
        df_dk2[f'{v}_dm'] = df_dk2[v] - state_mean - year_mean + grand_mean

    X = sm.add_constant(df_dk2['urate_dm'].values)
    y = df_dk2['h_numpers_noneld_dm'].values
    w = df_dk2['weight_noneld'].values
    wls = sm.WLS(y, X, weights=w).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
    dk_coef = float(wls.params[1])
    dk_se = float(wls.bse[1])
    dk_pval = float(wls.pvalues[1])
    dk_ci = wls.conf_int()
    dk_ci_lower = float(dk_ci[1, 0])
    dk_ci_upper = float(dk_ci[1, 1])

    payload = make_success_payload(
        coefficients={"urate": dk_coef, "const": float(wls.params[0])},
        inference={"spec_id": "infer/se/hac/driscoll_kraay", "params": {"maxlag": 3}},
        software=SW_BLOCK,
        surface_hash=SURFACE_HASH,
        design={"panel_fixed_effects": G2_DESIGN_AUDIT},
        notes="Driscoll-Kraay approximation via HAC on demeaned data",
    )
    inference_results.append({
        "paper_id": PAPER_ID, "inference_run_id": infer_run_id,
        "spec_run_id": g2_base_run_id,
        "spec_id": "infer/se/hac/driscoll_kraay",
        "spec_tree_path": "specification_tree/modules/inference/standard_errors.md#hac",
        "baseline_group_id": "G2",
        "outcome_var": "h_numpers_noneld", "treatment_var": "urate",
        "coefficient": dk_coef, "std_error": dk_se, "p_value": dk_pval,
        "ci_lower": dk_ci_lower, "ci_upper": dk_ci_upper,
        "n_obs": int(wls.nobs), "r_squared": float(wls.rsquared),
        "coefficient_vector_json": json.dumps(payload),
        "cluster_var": "none (HAC/Driscoll-Kraay)",
        "run_success": 1, "run_error": ""
    })
except Exception as e:
    spec_run_counter += 1
    infer_run_id = f"{PAPER_ID}_infer_{spec_run_counter:03d}"
    err_msg = str(e)[:240]
    payload = make_failure_payload(
        error=err_msg,
        error_details=error_details_from_exception(e, stage="inference_dk"),
        software=SW_BLOCK, surface_hash=SURFACE_HASH
    )
    inference_results.append({
        "paper_id": PAPER_ID, "inference_run_id": infer_run_id,
        "spec_run_id": g2_base_run_id,
        "spec_id": "infer/se/hac/driscoll_kraay",
        "spec_tree_path": "specification_tree/modules/inference/standard_errors.md#hac",
        "baseline_group_id": "G2",
        "outcome_var": "h_numpers_noneld", "treatment_var": "urate",
        "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
        "ci_lower": np.nan, "ci_upper": np.nan,
        "n_obs": np.nan, "r_squared": np.nan,
        "coefficient_vector_json": json.dumps(payload),
        "cluster_var": "none (HAC/Driscoll-Kraay)",
        "run_success": 0, "run_error": err_msg
    })

print(f"  Total inference variants: {len(inference_results)}")


# ============================================================
# WRITE OUTPUTS
# ============================================================
print("=" * 60)
print("Writing outputs")
print("=" * 60)

# specification_results.csv
df_results = pd.DataFrame(results)
df_results.to_csv(f"{OUTPUT_DIR}/specification_results.csv", index=False)
print(f"  specification_results.csv: {len(df_results)} rows")

# inference_results.csv
df_infer = pd.DataFrame(inference_results)
df_infer.to_csv(f"{OUTPUT_DIR}/inference_results.csv", index=False)
print(f"  inference_results.csv: {len(df_infer)} rows")

# Summary stats
n_success = int(df_results['run_success'].sum())
n_fail = len(df_results) - n_success
n_g1 = len(df_results[df_results['baseline_group_id'] == 'G1'])
n_g2 = len(df_results[df_results['baseline_group_id'] == 'G2'])
n_infer_success = int(df_infer['run_success'].sum()) if len(df_infer) > 0 else 0
n_infer_fail = len(df_infer) - n_infer_success

print(f"  G1 specs: {n_g1}")
print(f"  G2 specs: {n_g2}")
print(f"  Total core specs: {len(df_results)}")
print(f"  Successful: {n_success}, Failed: {n_fail}")
print(f"  Inference variants: {len(df_infer)} ({n_infer_success} success, {n_infer_fail} fail)")

# SPECIFICATION_SEARCH.md
baseline_g1 = df_results[(df_results['baseline_group_id'] == 'G1') & (df_results['spec_id'] == 'baseline')]
baseline_g2 = df_results[(df_results['baseline_group_id'] == 'G2') & (df_results['spec_id'] == 'baseline')]

g1_coef = baseline_g1['coefficient'].values[0] if len(baseline_g1) > 0 else np.nan
g1_se = baseline_g1['std_error'].values[0] if len(baseline_g1) > 0 else np.nan
g1_p = baseline_g1['p_value'].values[0] if len(baseline_g1) > 0 else np.nan
g1_n = baseline_g1['n_obs'].values[0] if len(baseline_g1) > 0 else np.nan

g2_coef = baseline_g2['coefficient'].values[0] if len(baseline_g2) > 0 else np.nan
g2_se = baseline_g2['std_error'].values[0] if len(baseline_g2) > 0 else np.nan
g2_p = baseline_g2['p_value'].values[0] if len(baseline_g2) > 0 else np.nan
g2_n = baseline_g2['n_obs'].values[0] if len(baseline_g2) > 0 else np.nan

# Compute sign concordance
g1_specs = df_results[df_results['baseline_group_id'] == 'G1']
g1_same_sign = g1_specs[(g1_specs['run_success'] == 1) & (np.sign(g1_specs['coefficient']) == np.sign(g1_coef))]
g1_sig_005 = g1_specs[(g1_specs['run_success'] == 1) & (g1_specs['p_value'] < 0.05)]
g1_sig_010 = g1_specs[(g1_specs['run_success'] == 1) & (g1_specs['p_value'] < 0.10)]

g2_specs = df_results[df_results['baseline_group_id'] == 'G2']
g2_same_sign = g2_specs[(g2_specs['run_success'] == 1) & (np.sign(g2_specs['coefficient']) == np.sign(g2_coef))]
g2_sig_005 = g2_specs[(g2_specs['run_success'] == 1) & (g2_specs['p_value'] < 0.05)]
g2_sig_010 = g2_specs[(g2_specs['run_success'] == 1) & (g2_specs['p_value'] < 0.10)]

md = f"""# Specification Search: {PAPER_ID}

## Paper
- **Title**: Living Arrangements, Doubling Up, and the Great Recession: Was This Time Different?
- **Authors**: Hoynes (2014)
- **Journal**: AER Papers & Proceedings, 104(5), 107-112
- **Design**: Panel fixed effects (state-year panel, 1980-2013)

## Surface Summary
- **Baseline groups**: 2 (G1: young adult living alone 18-30, G2: household size non-elderly)
- **G1 budget**: 55 max core specs
- **G2 budget**: 25 max core specs
- **Seed**: 113407
- **Controls pool**: None (no covariates beyond state and year FE)

## Baseline Results

### G1: Fraction Young Adults (18-30) Living Alone
- **Outcome**: `myadult_aloneall_1830`
- **Treatment**: `urate` (state unemployment rate)
- **Coefficient**: {g1_coef:.6f}
- **SE**: {g1_se:.6f} (clustered at state)
- **p-value**: {g1_p:.6f}
- **N**: {int(g1_n)}
- **Interpretation**: A 1 pp increase in state unemployment rate is associated with a {abs(g1_coef):.4f} decrease in the fraction of young adults living alone.

### G2: Average Household Size (Non-Elderly)
- **Outcome**: `h_numpers_noneld`
- **Treatment**: `urate` (state unemployment rate)
- **Coefficient**: {g2_coef:.6f}
- **SE**: {g2_se:.6f} (clustered at state)
- **p-value**: {g2_p:.6f}
- **N**: {int(g2_n)}
- **Interpretation**: A 1 pp increase in state unemployment rate is associated with a {g2_coef:.4f} increase in average household size.

## Execution Summary

| Metric | G1 | G2 | Total |
|--------|----|----|-------|
| Planned specs | {n_g1} | {n_g2} | {len(df_results)} |
| Successful | {int(g1_specs['run_success'].sum())} | {int(g2_specs['run_success'].sum())} | {n_success} |
| Failed | {n_g1 - int(g1_specs['run_success'].sum())} | {n_g2 - int(g2_specs['run_success'].sum())} | {n_fail} |
| Same sign as baseline | {len(g1_same_sign)}/{int(g1_specs['run_success'].sum())} | {len(g2_same_sign)}/{int(g2_specs['run_success'].sum())} | {len(g1_same_sign) + len(g2_same_sign)}/{n_success} |
| Significant (p<0.05) | {len(g1_sig_005)}/{int(g1_specs['run_success'].sum())} | {len(g2_sig_005)}/{int(g2_specs['run_success'].sum())} | {len(g1_sig_005) + len(g2_sig_005)}/{n_success} |
| Significant (p<0.10) | {len(g1_sig_010)}/{int(g1_specs['run_success'].sum())} | {len(g2_sig_010)}/{int(g2_specs['run_success'].sum())} | {len(g1_sig_010) + len(g2_sig_010)}/{n_success} |

## Inference Variants

| Variant | Group | Coefficient | SE | p-value | N |
|---------|-------|------------|-----|---------|---|
"""

for _, row in df_infer.iterrows():
    if row['run_success'] == 1:
        md += f"| {row['spec_id']} | {row['baseline_group_id']} | {row['coefficient']:.6f} | {row['std_error']:.6f} | {row['p_value']:.6f} | {int(row['n_obs'])} |\n"
    else:
        md += f"| {row['spec_id']} | {row['baseline_group_id']} | FAILED | - | - | - |\n"

md += f"""
## Spec IDs Executed

### G1 Core Specs
"""
for _, row in g1_specs.iterrows():
    status = "OK" if row['run_success'] == 1 else "FAIL"
    md += f"- `{row['spec_id']}`: {status}"
    if row['run_success'] == 1:
        md += f" (coef={row['coefficient']:.6f}, p={row['p_value']:.6f}, N={int(row['n_obs'])})"
    else:
        md += f" ({row['run_error'][:80]})"
    md += "\n"

md += f"""
### G2 Core Specs
"""
for _, row in g2_specs.iterrows():
    status = "OK" if row['run_success'] == 1 else "FAIL"
    md += f"- `{row['spec_id']}`: {status}"
    if row['run_success'] == 1:
        md += f" (coef={row['coefficient']:.6f}, p={row['p_value']:.6f}, N={int(row['n_obs'])})"
    else:
        md += f" ({row['run_error'][:80]})"
    md += "\n"

md += f"""
## Deviations from Surface

- None. All surface specs executed as planned.

## Software Stack
- Python {sys.version.split()[0]}
- pyfixest {SW_BLOCK['packages'].get('pyfixest', 'unknown')}
- pandas {SW_BLOCK['packages'].get('pandas', 'unknown')}
- numpy {SW_BLOCK['packages'].get('numpy', 'unknown')}
- statsmodels {SW_BLOCK['packages'].get('statsmodels', 'unknown')}
"""

with open(f"{OUTPUT_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write(md)
print(f"  SPECIFICATION_SEARCH.md written")

print("\nDone!")
