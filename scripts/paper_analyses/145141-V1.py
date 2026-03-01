"""
Specification Search Script for Bursztyn, Ferman, Fiorin, Kanz, Rao (2018)
"Status Goods: Experimental Evidence from Platinum Credit Cards"
AER 2018 — openICPSR package 145141-V1

Actually: This package is about public recognition and effort/WTP.
The paper has three experimental settings:
  1) YMCA field experiment (G1): effect of public recognition on gym attendance
  2) Charity online experiments (G2): effect of public recognition on real-effort task points
  3) WTP for public recognition (G3): elicited via BDM mechanism across hypothetical performance levels

Surface-driven execution:
  - G1: attendance ~ image [+ controls], HC1 robust SE (YMCA)
  - G2: pts ~ SR ownpay o1 o2, cluster(id) (Charity real-effort, within-subject)
  - G3: wtp ~ visits visits^2 [or interval interval^2], cluster(id) (WTP elicitation)

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
import os
import warnings
from scipy.optimize import minimize
from scipy.stats import norm, t as t_dist

warnings.filterwarnings('ignore')

# Add scripts dir for utilities
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash, software_block
)

PAPER_ID = "145141-V1"
DATA_DIR = os.path.join(REPO_ROOT, "data/downloads/extracted/145141-V1")
RAW_DIR = os.path.join(DATA_DIR, "Raw Data")

# Load surface
with open(os.path.join(DATA_DIR, "SPECIFICATION_SURFACE.json")) as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

results = []
inference_results = []
spec_run_counter = 0
inference_run_counter = 0


# ============================================================================
# HELPER: run a single OLS specification
# ============================================================================
def next_run_id():
    global spec_run_counter
    spec_run_counter += 1
    return f"{PAPER_ID}_run_{spec_run_counter:03d}"


def next_infer_id():
    global inference_run_counter
    inference_run_counter += 1
    return f"{PAPER_ID}_infer_{inference_run_counter:03d}"


def run_ols_spec(
    spec_id, spec_tree_path, baseline_group_id,
    outcome_var, treatment_var, controls, fixed_effects_str, fe_formula,
    data, vcov, sample_desc, controls_desc, cluster_var="",
    design_audit=None, design_code="randomized_experiment",
    inference_spec_id="infer/se/hc/hc1", inference_params=None,
    axis_block_name=None, axis_block=None, notes=""
):
    run_id = next_run_id()
    try:
        regvars = [outcome_var, treatment_var] + list(controls)
        df_reg = data.dropna(subset=regvars).copy()

        controls_str = " + ".join(controls) if controls else ""
        if controls_str and fe_formula:
            formula = f"{outcome_var} ~ {treatment_var} + {controls_str} | {fe_formula}"
        elif controls_str:
            formula = f"{outcome_var} ~ {treatment_var} + {controls_str}"
        elif fe_formula:
            formula = f"{outcome_var} ~ {treatment_var} | {fe_formula}"
        else:
            formula = f"{outcome_var} ~ {treatment_var}"

        m = pf.feols(formula, data=df_reg, vcov=vcov)

        coef_val = float(m.coef().get(treatment_var, np.nan))
        se_val = float(m.se().get(treatment_var, np.nan))
        pval = float(m.pvalue().get(treatment_var, np.nan))

        try:
            ci = m.confint()
            ci_lower = float(ci.loc[treatment_var, ci.columns[0]]) if treatment_var in ci.index else np.nan
            ci_upper = float(ci.loc[treatment_var, ci.columns[1]]) if treatment_var in ci.index else np.nan
        except Exception:
            ci_lower = np.nan
            ci_upper = np.nan

        nobs = int(m._N)
        try:
            r2 = float(m._r2)
        except Exception:
            r2 = np.nan

        all_coefs = {k: float(v) for k, v in m.coef().items()}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": inference_spec_id,
                       "params": inference_params or {}},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={design_code: design_audit or {}},
            axis_block_name=axis_block_name,
            axis_block=axis_block,
        )
        if notes:
            payload["notes"] = notes

        results.append({
            "paper_id": PAPER_ID,
            "spec_run_id": run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": treatment_var,
            "coefficient": coef_val,
            "std_error": se_val,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": nobs,
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": fixed_effects_str,
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
            "run_success": 1,
            "run_error": ""
        })
        return run_id, coef_val, se_val, pval, nobs

    except Exception as e:
        err_msg = str(e)[:240]
        err_details = error_details_from_exception(e, stage="estimation")
        payload = make_failure_payload(
            error=err_msg, error_details=err_details,
            software=SW_BLOCK, surface_hash=SURFACE_HASH
        )
        results.append({
            "paper_id": PAPER_ID,
            "spec_run_id": run_id,
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
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": fixed_effects_str,
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
            "run_success": 0,
            "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


def add_inference_row(
    base_run_id, spec_id, spec_tree_path, baseline_group_id,
    outcome_var, treatment_var, controls, fe_formula,
    data, vcov, cluster_var="", notes=""
):
    """Run the same model with a different vcov for inference_results."""
    inf_id = next_infer_id()
    try:
        regvars = [outcome_var, treatment_var] + list(controls)
        df_reg = data.dropna(subset=regvars).copy()

        controls_str = " + ".join(controls) if controls else ""
        if controls_str and fe_formula:
            formula = f"{outcome_var} ~ {treatment_var} + {controls_str} | {fe_formula}"
        elif controls_str:
            formula = f"{outcome_var} ~ {treatment_var} + {controls_str}"
        elif fe_formula:
            formula = f"{outcome_var} ~ {treatment_var} | {fe_formula}"
        else:
            formula = f"{outcome_var} ~ {treatment_var}"

        m = pf.feols(formula, data=df_reg, vcov=vcov)

        coef_val = float(m.coef().get(treatment_var, np.nan))
        se_val = float(m.se().get(treatment_var, np.nan))
        pval = float(m.pvalue().get(treatment_var, np.nan))

        try:
            ci = m.confint()
            ci_lower = float(ci.loc[treatment_var, ci.columns[0]]) if treatment_var in ci.index else np.nan
            ci_upper = float(ci.loc[treatment_var, ci.columns[1]]) if treatment_var in ci.index else np.nan
        except Exception:
            ci_lower = np.nan
            ci_upper = np.nan

        nobs = int(m._N)
        try:
            r2 = float(m._r2)
        except Exception:
            r2 = np.nan

        all_coefs = {k: float(v) for k, v in m.coef().items()}

        payload = {
            "coefficients": all_coefs,
            "inference": {"spec_id": spec_id, "params": {}},
            "software": SW_BLOCK,
            "surface_hash": SURFACE_HASH,
        }

        inference_results.append({
            "paper_id": PAPER_ID,
            "inference_run_id": inf_id,
            "spec_run_id": base_run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "coefficient": coef_val,
            "std_error": se_val,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": nobs,
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 1,
            "run_error": ""
        })

    except Exception as e:
        err_msg = str(e)[:240]
        err_details = error_details_from_exception(e, stage="inference")
        payload = make_failure_payload(
            error=err_msg, error_details=err_details,
            software=SW_BLOCK, surface_hash=SURFACE_HASH
        )
        inference_results.append({
            "paper_id": PAPER_ID,
            "inference_run_id": inf_id,
            "spec_run_id": base_run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "coefficient": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": np.nan,
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 0,
            "run_error": err_msg
        })


# ============================================================================
# DATA LOADING: YMCA EXPERIMENT
# ============================================================================
print("=== Building YMCA data ===")

# Build attendance data from scans
scans = pd.read_csv(os.path.join(RAW_DIR, "ymca_scans.csv"))
scans.columns = [c.lower() for c in scans.columns]

# Parse scan dates
scans['scandate'] = pd.to_datetime(scans['usrdatetime'], format='%d%b%Y %H:%M:%S', errors='coerce')
scans['month'] = scans['scandate'].dt.month
scans['day'] = scans['scandate'].dt.day
scans['year'] = scans['scandate'].dt.year
scans = scans.drop(columns=['usrdatetime', 'scandate']).drop_duplicates()

# Attendance during experiment (June 15 - July 15, 2017)
scans['att_exp'] = 0
scans.loc[(scans['year'] == 2017) & (scans['month'] == 6) & (scans['day'] >= 15), 'att_exp'] = 1
scans.loc[(scans['year'] == 2017) & (scans['month'] == 7) & (scans['day'] <= 15), 'att_exp'] = 1

attendance = scans.groupby('ymca_id')['att_exp'].sum().reset_index()
attendance.columns = ['ymca_id', 'attendance']

# Monthly attendance counts
monthly = scans.groupby(['ymca_id', 'year', 'month']).size().reset_index(name='att')
monthly_pivot = monthly.pivot_table(index='ymca_id', columns=['year', 'month'], values='att', fill_value=0)

# Merge attendance
ymca_att = attendance.copy()

# Load membership data
membership = pd.read_csv(os.path.join(RAW_DIR, "ymca_membership_data.csv"))
membership.columns = [c.lower() for c in membership.columns]

ymca_full = membership.merge(ymca_att, on='ymca_id', how='left')
# Replace attendance=0 if person was member in June 2017
ymca_full.loc[ymca_full['attendance'].isna() & (ymca_full['m2017_6'] == 1), 'attendance'] = 0

# Build monthly attendance from scans for pre-period
preperiod_months = [(2016, m) for m in range(5, 13)] + [(2017, m) for m in range(1, 6)]
monthly_wide = monthly.pivot_table(index='ymca_id', columns=['year', 'month'], values='att', fill_value=0)

att_preperiod = pd.Series(0, index=ymca_full.index, dtype=float)
for ymca_id_val in ymca_full['ymca_id'].values:
    total = 0
    for yr, mo in preperiod_months:
        if ymca_id_val in monthly_wide.index and (yr, mo) in monthly_wide.columns:
            total += monthly_wide.loc[ymca_id_val, (yr, mo)]
    att_preperiod.loc[ymca_full['ymca_id'] == ymca_id_val] = total

ymca_full['att_preperiod'] = att_preperiod.values
ymca_full['past'] = ymca_full['att_preperiod'] / 13.0

# Load treatment assignment
treatment = pd.read_csv(os.path.join(RAW_DIR, "ymca_treatment_assignment.csv"))
treatment.columns = [c.lower() for c in treatment.columns]

# Load survey data
survey = pd.read_csv(os.path.join(RAW_DIR, "ymca_survey_data.csv"))
survey.columns = [c.lower() for c in survey.columns]

# Merge treatment
survey = survey.merge(treatment[['survey_id', 'treatment']], on='survey_id', how='left')

# Create bdm and image dummies
survey['bdm'] = (survey['treatment'] == 2).astype(int)
survey['image'] = np.where(survey['bdm'] == 1, np.nan, (survey['treatment'] == 1).astype(float))

# Rename beliefs
survey = survey.rename(columns={
    'q25': 'beliefs_morethan4',
    'q26_1': 'beliefs_w_image',
    'q27_1': 'beliefs_wout_image',
    'q28_1': 'beliefs_wout_exp',
    'q30_1': 'beliefs_w_1',
    'q30_2': 'beliefs_w_3'
})
survey.loc[survey['beliefs_morethan4'] == 2, 'beliefs_morethan4'] = 0

# Generate WTP variables and coherent/monotonic flags
intervals = [0, 1, 2, 3, 4, 56, 78, 912, 1317, 1822, 2328]
switch_count = pd.Series(0, index=survey.index)
switch_yes_no = pd.Series(0, index=survey.index)

prev = (survey[f'times_0_1'] == 2).astype(int)
for x in intervals:
    col = f'times_{x}_1'
    img_x = (survey[col] == 2).astype(int)
    switch_count += (img_x != prev).astype(int)
    switch_yes_no += (img_x < prev).astype(int)
    prev = img_x.copy()

    # WTP: negative if don't want image, positive if want image
    survey[f'wtp{x}'] = np.where(
        img_x == 0,
        -1 * survey[f'times_{x}_no_1'],
        survey[f'times_{x}_yes_1']
    )

survey['switch'] = switch_count
survey['switch_yes_no'] = switch_yes_no

survey['coherent'] = 1
survey.loc[survey['switch'] > 2, 'coherent'] = 0
survey.loc[(survey[f'times_0_1'] == 2) & (survey['switch'] == 2), 'coherent'] = 0

survey['monotonic'] = 1
survey.loc[survey['switch_yes_no'] > 0, 'monotonic'] = 0

survey['inexperiment'] = 1
survey['coherent_sample'] = np.where((survey['coherent'] == 0) | (survey['bdm'] == 1), 0, 1)
survey['robust_sample'] = np.where(survey['bdm'] == 1, 0, 1)
survey['monotonic_sample'] = np.where((survey['monotonic'] == 0) | (survey['bdm'] == 1), 0, 1)

# Keep relevant columns
survey_keep = survey[['ymca_id'] + [f'wtp{x}' for x in intervals] +
                     ['beliefs_w_image', 'beliefs_wout_image', 'beliefs_wout_exp',
                      'beliefs_w_1', 'beliefs_w_3', 'beliefs_morethan4',
                      'inexperiment', 'bdm', 'coherent', 'monotonic',
                      'coherent_sample', 'robust_sample', 'monotonic_sample',
                      'image', 'switch', 'switch_yes_no']].copy()

# Merge survey with attendance/full-pop data
ymca_exp = ymca_full.merge(survey_keep, on='ymca_id', how='right')
ymca_exp.loc[ymca_exp['attendance'].isna() & (ymca_exp['inexperiment'] == 1), 'attendance'] = 0

# Create id
ymca_exp = ymca_exp.reset_index(drop=True)
ymca_exp['id'] = np.arange(1, len(ymca_exp) + 1)

print(f"YMCA experiment data: {len(ymca_exp)} obs")
print(f"  coherent_sample: {(ymca_exp['coherent_sample'] == 1).sum()}")
print(f"  monotonic_sample: {(ymca_exp['monotonic_sample'] == 1).sum()}")

# ============================================================================
# G1: YMCA ATTENDANCE SPECIFICATIONS
# ============================================================================
print("\n=== G1: YMCA attendance specs ===")

g1_audit = surface_obj["baseline_groups"][0]["design_audit"]
g1_canonical = surface_obj["baseline_groups"][0]["inference_plan"]["canonical"]

coh = ymca_exp[ymca_exp['coherent_sample'] == 1].copy()
mon = ymca_exp[ymca_exp['monotonic_sample'] == 1].copy()
rob = ymca_exp[ymca_exp['robust_sample'] == 1].copy()

# --- Baseline: Table 2, Col 2 (coherent): attendance ~ image past, r ---
run_id_b1, *_ = run_ols_spec(
    "baseline__table2_col2_coh",
    "specification_tree/designs/randomized_experiment.md",
    "G1", "attendance", "image", ["past"], "", "",
    coh, "hetero",
    "coherent_sample", "past",
    design_audit=g1_audit, design_code="randomized_experiment",
    inference_spec_id=g1_canonical["spec_id"],
    notes="Table 2 Col 2 (Coherent): reg attendance image past, r"
)

# --- Baseline: Table 2, Col 3 (coherent): attendance ~ image past beliefs_w_image, r ---
run_id_b2, *_ = run_ols_spec(
    "baseline__table2_col3_coh",
    "specification_tree/designs/randomized_experiment.md",
    "G1", "attendance", "image", ["past", "beliefs_w_image"], "", "",
    coh, "hetero",
    "coherent_sample", "past + beliefs_w_image",
    design_audit=g1_audit, design_code="randomized_experiment",
    inference_spec_id=g1_canonical["spec_id"],
    notes="Table 2 Col 3 (Coherent): reg attendance image past beliefs_w_image, r"
)

# --- Design variant: diff-in-means (no controls) ---
run_id_dim, *_ = run_ols_spec(
    "design/randomized_experiment/estimator/diff_in_means",
    "specification_tree/designs/randomized_experiment.md#diff-in-means",
    "G1", "attendance", "image", [], "", "",
    coh, "hetero",
    "coherent_sample", "none",
    design_audit=g1_audit, design_code="randomized_experiment",
    inference_spec_id=g1_canonical["spec_id"],
    notes="Diff-in-means: reg attendance image, r (coherent sample)"
)

# --- RC: control sets ---
# No controls
run_ols_spec(
    "rc/controls/sets/none",
    "specification_tree/modules/robustness/controls.md#sets",
    "G1", "attendance", "image", [], "", "",
    coh, "hetero",
    "coherent_sample", "none",
    design_audit=g1_audit, design_code="randomized_experiment",
    inference_spec_id=g1_canonical["spec_id"],
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/none", "family": "sets", "included": [], "n_controls": 0},
)

# Past only
run_ols_spec(
    "rc/controls/sets/past_only",
    "specification_tree/modules/robustness/controls.md#sets",
    "G1", "attendance", "image", ["past"], "", "",
    coh, "hetero",
    "coherent_sample", "past",
    design_audit=g1_audit, design_code="randomized_experiment",
    inference_spec_id=g1_canonical["spec_id"],
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/past_only", "family": "sets", "included": ["past"], "n_controls": 1},
)

# Past + beliefs
run_ols_spec(
    "rc/controls/sets/past_and_beliefs",
    "specification_tree/modules/robustness/controls.md#sets",
    "G1", "attendance", "image", ["past", "beliefs_w_image"], "", "",
    coh, "hetero",
    "coherent_sample", "past + beliefs_w_image",
    design_audit=g1_audit, design_code="randomized_experiment",
    inference_spec_id=g1_canonical["spec_id"],
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/past_and_beliefs", "family": "sets",
                "included": ["past", "beliefs_w_image"], "n_controls": 2},
)

# --- RC: sample definitions ---
# Monotonic sample
run_ols_spec(
    "rc/sample/definition/monotonic_sample",
    "specification_tree/modules/robustness/sample.md#definition",
    "G1", "attendance", "image", ["past"], "", "",
    mon, "hetero",
    "monotonic_sample", "past",
    design_audit=g1_audit, design_code="randomized_experiment",
    inference_spec_id=g1_canonical["spec_id"],
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/definition/monotonic_sample",
                "restriction": "monotonic_sample==1"},
)

# Robust (full) sample
run_ols_spec(
    "rc/sample/definition/robust_sample",
    "specification_tree/modules/robustness/sample.md#definition",
    "G1", "attendance", "image", ["past"], "", "",
    rob, "hetero",
    "robust_sample (excl BDM only)", "past",
    design_audit=g1_audit, design_code="randomized_experiment",
    inference_spec_id=g1_canonical["spec_id"],
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/definition/robust_sample",
                "restriction": "robust_sample==1"},
)

# --- RC: Outlier trimming on attendance ---
# Trim y 1/99
coh_trimmed_1_99 = coh.copy()
lo, hi = coh_trimmed_1_99['attendance'].quantile([0.01, 0.99])
coh_trimmed_1_99 = coh_trimmed_1_99[(coh_trimmed_1_99['attendance'] >= lo) & (coh_trimmed_1_99['attendance'] <= hi)]

run_ols_spec(
    "rc/sample/outliers/trim_y_1_99",
    "specification_tree/modules/robustness/sample.md#outliers",
    "G1", "attendance", "image", ["past"], "", "",
    coh_trimmed_1_99, "hetero",
    "coherent_sample, attendance trimmed 1-99%", "past",
    design_audit=g1_audit, design_code="randomized_experiment",
    inference_spec_id=g1_canonical["spec_id"],
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_1_99",
                "trim_lower": 0.01, "trim_upper": 0.99},
)

# Trim y 5/95
coh_trimmed_5_95 = coh.copy()
lo, hi = coh_trimmed_5_95['attendance'].quantile([0.05, 0.95])
coh_trimmed_5_95 = coh_trimmed_5_95[(coh_trimmed_5_95['attendance'] >= lo) & (coh_trimmed_5_95['attendance'] <= hi)]

run_ols_spec(
    "rc/sample/outliers/trim_y_5_95",
    "specification_tree/modules/robustness/sample.md#outliers",
    "G1", "attendance", "image", ["past"], "", "",
    coh_trimmed_5_95, "hetero",
    "coherent_sample, attendance trimmed 5-95%", "past",
    design_audit=g1_audit, design_code="randomized_experiment",
    inference_spec_id=g1_canonical["spec_id"],
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_5_95",
                "trim_lower": 0.05, "trim_upper": 0.95},
)

# Winsorize y 1/99
coh_wins = coh.copy()
lo, hi = coh_wins['attendance'].quantile([0.01, 0.99])
coh_wins['attendance'] = coh_wins['attendance'].clip(lo, hi)

run_ols_spec(
    "rc/sample/outliers/winsorize_y_1_99",
    "specification_tree/modules/robustness/sample.md#outliers",
    "G1", "attendance", "image", ["past"], "", "",
    coh_wins, "hetero",
    "coherent_sample, attendance winsorized 1-99%", "past",
    design_audit=g1_audit, design_code="randomized_experiment",
    inference_spec_id=g1_canonical["spec_id"],
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/winsorize_y_1_99",
                "winsorize_lower": 0.01, "winsorize_upper": 0.99},
)

# --- RC: Functional form ---
# log(1+attendance)
coh_log = coh.copy()
coh_log['log1p_attendance'] = np.log1p(coh_log['attendance'])

run_ols_spec(
    "rc/form/outcome/log1p",
    "specification_tree/modules/robustness/functional_form.md",
    "G1", "log1p_attendance", "image", ["past"], "", "",
    coh_log, "hetero",
    "coherent_sample, log(1+attendance)", "past",
    design_audit=g1_audit, design_code="randomized_experiment",
    inference_spec_id=g1_canonical["spec_id"],
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/log1p",
                "transform": "log1p", "original_var": "attendance",
                "interpretation": "semi-elasticity of attendance to image"},
)

# asinh(attendance)
coh_asinh = coh.copy()
coh_asinh['asinh_attendance'] = np.arcsinh(coh_asinh['attendance'])

run_ols_spec(
    "rc/form/outcome/asinh",
    "specification_tree/modules/robustness/functional_form.md",
    "G1", "asinh_attendance", "image", ["past"], "", "",
    coh_asinh, "hetero",
    "coherent_sample, asinh(attendance)", "past",
    design_audit=g1_audit, design_code="randomized_experiment",
    inference_spec_id=g1_canonical["spec_id"],
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/asinh",
                "transform": "asinh", "original_var": "attendance",
                "interpretation": "approximate semi-elasticity"},
)

# Standardized attendance
coh_std = coh.copy()
att_mean = coh_std['attendance'].mean()
att_sd = coh_std['attendance'].std()
coh_std['std_attendance'] = (coh_std['attendance'] - att_mean) / att_sd

run_ols_spec(
    "rc/form/outcome/standardized",
    "specification_tree/modules/robustness/functional_form.md",
    "G1", "std_attendance", "image", ["past"], "", "",
    coh_std, "hetero",
    "coherent_sample, standardized attendance", "past",
    design_audit=g1_audit, design_code="randomized_experiment",
    inference_spec_id=g1_canonical["spec_id"],
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/standardized",
                "transform": "standardized", "original_var": "attendance",
                "original_mean": float(att_mean), "original_sd": float(att_sd),
                "interpretation": "effect in SD units"},
)

# --- RC: preprocessing/coding ---
# Topcode attendance at 22
coh_tc22 = coh.copy()
coh_tc22['attendance'] = coh_tc22['attendance'].clip(upper=22)

run_ols_spec(
    "rc/preprocess/coding/attendance_topcode_22",
    "specification_tree/modules/robustness/preprocess.md#coding",
    "G1", "attendance", "image", ["past"], "", "",
    coh_tc22, "hetero",
    "coherent_sample, attendance topcoded at 22", "past",
    design_audit=g1_audit, design_code="randomized_experiment",
    inference_spec_id=g1_canonical["spec_id"],
    axis_block_name="preprocess",
    axis_block={"spec_id": "rc/preprocess/coding/attendance_topcode_22",
                "topcode": 22},
)

# Topcode attendance at 15
coh_tc15 = coh.copy()
coh_tc15['attendance'] = coh_tc15['attendance'].clip(upper=15)

run_ols_spec(
    "rc/preprocess/coding/attendance_topcode_15",
    "specification_tree/modules/robustness/functional_form.md",
    "G1", "attendance", "image", ["past"], "", "",
    coh_tc15, "hetero",
    "coherent_sample, attendance topcoded at 15", "past",
    design_audit=g1_audit, design_code="randomized_experiment",
    inference_spec_id=g1_canonical["spec_id"],
    axis_block_name="preprocess",
    axis_block={"spec_id": "rc/preprocess/coding/attendance_topcode_15",
                "topcode": 15},
)

# G1 inference variant: HC3
add_inference_row(
    run_id_b1, "infer/se/hc/hc3",
    "specification_tree/modules/inference/se.md#hc3",
    "G1", "attendance", "image", ["past"], "",
    coh, "HC3",
    notes="HC3 SEs on baseline G1 spec"
)

print(f"G1 specs done. Total results so far: {len(results)}")


# ============================================================================
# DATA LOADING: CHARITY EXPERIMENT
# ============================================================================
print("\n=== Building Charity data ===")

def load_charity_sample(filename, sample_name):
    df = pd.read_csv(os.path.join(RAW_DIR, filename))
    df.columns = [c.lower() for c in df.columns]

    # attention check flag
    df['flag_attention_check'] = df['attention_check'].notna().astype(int)

    # Generate consistency indicators
    # switch_preference count
    switch_pref = pd.Series(0, index=df.index)
    for i in range(100, 1800, 100):
        j = i - 100
        col_j = f'preference_{j}'
        col_i = f'preference_{i}'
        if col_j in df.columns and col_i in df.columns:
            switch_pref += (df[col_j] != df[col_i]).astype(int)
    df['switch_preference'] = switch_pref

    # consistent: all Y, all N, or one switch from N->Y
    df['consistent'] = ((df['switch_preference'] == 0) |
                        ((df['switch_preference'] == 1) & (df['preference_0'] == 0))).astype(int)

    # consistent_b: also allow N->Y->N
    df['consistent_b'] = df['consistent'].copy()
    df.loc[(df['switch_preference'] == 2) & (df['preference_0'] == 0), 'consistent_b'] = 1

    # Generate WTP variables
    for i in range(18):
        j = i * 100
        anom_col = f'pref_anom_{j}_1'
        rec_col = f'pref_rec_{j}_1'
        wtp_col = f'wtp{i}'

        if anom_col in df.columns:
            df[anom_col] = pd.to_numeric(df[anom_col], errors='coerce')
            df[wtp_col] = -1 * df[anom_col]
        else:
            df[wtp_col] = np.nan

        if rec_col in df.columns:
            df[rec_col] = pd.to_numeric(df[rec_col], errors='coerce')
            df.loc[df[wtp_col].isna(), wtp_col] = df[rec_col]

    # approx_monotonic
    df['approx_monotonic'] = 1
    for i in range(2, 18):
        j_max = i - 2
        for k in range(j_max + 1):
            df.loc[df[f'wtp{i}'] < df[f'wtp{k}'], 'approx_monotonic'] = 0

    # Points variables
    for col in ['earnpts', 'anompts', 'recogpts', 'earndon', 'anomdon', 'recogdon',
                'earnown', 'finalpoints', 'finaldonation', 'finalearned', 'clicked']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['sample'] = sample_name
    return df


prolific = load_charity_sample('prolific_survey_data.csv', 'Prolific')
berkeley = load_charity_sample('berkeley_survey_data.csv', 'Berkeley')
bu_qm222 = load_charity_sample('bu_qm222_survey_data.csv', 'BU')

# Combine charity data
charity = pd.concat([prolific, berkeley, bu_qm222], ignore_index=True)

# Assign unique IDs
np.random.seed(100)
charity['id'] = np.random.randint(1000000, 9999999, size=len(charity))

# high_pts flag: anyone with > 3000 in ANY round
charity['high_pts'] = 0
for rnd in ['anompts', 'earnpts', 'recogpts']:
    charity.loc[charity[rnd] > 3000, 'high_pts'] = 1
charity = charity[charity['high_pts'] == 0].copy()

print(f"Charity data: {len(charity)} obs total (after high_pts drop)")

# Reshape to long for points analysis (3 obs per person)
def make_charity_long(df):
    """Reshape to long: one row per round per person."""
    rows = []
    for idx, row in df.iterrows():
        base = row.to_dict()
        for round_name, pts_col, don_col, sr_val, op_val in [
            ('anom', 'anompts', 'anomdon', 0, 0),
            ('recog', 'recogpts', 'recogdon', 1, 0),
            ('earn', 'earnpts', 'earndon', 0, 1)
        ]:
            r = {k: v for k, v in base.items() if k not in ['anompts', 'recogpts', 'earnpts']}
            r['pts'] = row[pts_col]
            r['round'] = round_name
            r['SR'] = sr_val
            r['ownpay'] = op_val
            rows.append(r)
    return pd.DataFrame(rows)


# Build long data more efficiently
charity_long_list = []
for sname in ['Prolific', 'Berkeley', 'BU']:
    sub = charity[charity['sample'] == sname].copy()

    # reshape
    anom = sub.copy()
    anom['pts'] = anom['anompts']
    anom['round'] = 'anom'
    anom['SR'] = 0
    anom['ownpay'] = 0

    recog = sub.copy()
    recog['pts'] = recog['recogpts']
    recog['round'] = 'recog'
    recog['SR'] = 1
    recog['ownpay'] = 0

    earn = sub.copy()
    earn['pts'] = earn['earnpts']
    earn['round'] = 'earn'
    earn['SR'] = 0
    earn['ownpay'] = 1

    combined = pd.concat([anom, recog, earn], ignore_index=True)
    charity_long_list.append(combined)

charity_long = pd.concat(charity_long_list, ignore_index=True)

# Label rounds with order
condition_col = 'condition'
charity_long['order'] = np.nan

order_map = {
    'AER': {'anom': 1, 'recog': 2, 'earn': 3},
    'ARE': {'anom': 1, 'recog': 3, 'earn': 2},
    'EAR': {'anom': 2, 'recog': 1, 'earn': 3},  # note: EAR = Earn-Anom-Recog but do file reverses
    'ERA': {'anom': 3, 'recog': 1, 'earn': 2},
    'RAE': {'anom': 2, 'recog': 3, 'earn': 1},
    'REA': {'anom': 3, 'recog': 2, 'earn': 1},
}
# The label_rounds.do actually maps differently. Let me use its exact mapping:
order_map = {
    'AER': {'anom': 1, 'earn': 3, 'recog': 2},  # A=1, E=2->3, R=3->2? No...
    'ARE': {'anom': 1, 'recog': 3, 'earn': 2},
    'EAR': {'anom': 2, 'earn': 3, 'recog': 1},   # E first, then A, then R
    'ERA': {'anom': 3, 'earn': 2, 'recog': 1},   # E first, R second, A third
    'RAE': {'anom': 2, 'earn': 1, 'recog': 3},   # R first, A second, E third
    'REA': {'anom': 3, 'earn': 2, 'recog': 1},   # R first, E second, A third
}
# Let me re-read the label_rounds.do more carefully:
# AER: anom=1, recog=2, earn=3  (A first, E second, R third)
# ARE: anom=1, recog=3, earn=2  (A first, R second, E third)
# EAR: anom=2, recog=1, earn=3  -- wait, do says recog=1? That's wrong for EAR
# Let me just use the exact do file mapping:
order_map = {
    'AER': {'anom': 1, 'recog': 2, 'earn': 3},
    'ARE': {'anom': 1, 'recog': 3, 'earn': 2},
    'EAR': {'anom': 2, 'recog': 1, 'earn': 3},
    'ERA': {'anom': 3, 'recog': 1, 'earn': 2},
    'RAE': {'anom': 2, 'recog': 3, 'earn': 1},
    'REA': {'anom': 3, 'recog': 2, 'earn': 1},
}

for cond, rnd_map in order_map.items():
    for rnd, ord_val in rnd_map.items():
        mask = (charity_long[condition_col] == cond) & (charity_long['round'] == rnd)
        charity_long.loc[mask, 'order'] = ord_val

# Generate order dummies
charity_long['o1'] = (charity_long['order'] == 1).astype(int)
charity_long['o2'] = (charity_long['order'] == 2).astype(int)

print(f"Charity long data: {len(charity_long)} obs")

# ============================================================================
# G2: CHARITY REAL-EFFORT SPECIFICATIONS
# ============================================================================
print("\n=== G2: Charity real-effort specs ===")

g2_audit = surface_obj["baseline_groups"][1]["design_audit"]
g2_canonical = surface_obj["baseline_groups"][1]["inference_plan"]["canonical"]

flag_cond = "flag_attention_check == 0"
excl_cond = "consistent_b == 1"

def charity_sample(df, sample_name):
    return df[(df['sample'] == sample_name) &
              (df['flag_attention_check'] == 0) &
              (df['consistent_b'] == 1)].copy()


# --- Baselines: Table 5 columns ---
for sname, spec_suffix, sample_label in [
    ('Prolific', 'prolific', 'Prolific'),
    ('Berkeley', 'berkeley', 'Berkeley'),
    ('BU', 'bu', 'BU')
]:
    df_s = charity_sample(charity_long, sname)
    run_id, *_ = run_ols_spec(
        f"baseline__table5_col{['prolific','berkeley','bu'].index(spec_suffix)+1}_{spec_suffix}",
        "specification_tree/designs/randomized_experiment.md",
        "G2", "pts", "SR", ["ownpay", "o1", "o2"], "", "",
        df_s, {"CRV1": "id"},
        f"{sample_label}, attention+consistency filter", "ownpay + o1 + o2",
        cluster_var="id",
        design_audit=g2_audit, design_code="randomized_experiment",
        inference_spec_id=g2_canonical["spec_id"],
        inference_params={"cluster_var": "id"},
        notes=f"Table 5: reg pts SR ownpay o1 o2, cluster(id). {sample_label} sample."
    )

    # Inference variant: HC1 (no clustering)
    add_inference_row(
        run_id, "infer/se/hc/hc1",
        "specification_tree/modules/inference/se.md#hc1",
        "G2", "pts", "SR", ["ownpay", "o1", "o2"], "",
        df_s, "hetero",
        notes=f"HC1 SEs on G2 baseline, {sample_label}"
    )

# --- Design variant: diff-in-means (SR only, no controls) ---
for sname in ['Prolific', 'Berkeley', 'BU']:
    df_s = charity_sample(charity_long, sname)
    run_ols_spec(
        "design/randomized_experiment/estimator/diff_in_means",
        "specification_tree/designs/randomized_experiment.md#diff-in-means",
        "G2", "pts", "SR", [], "", "",
        df_s, {"CRV1": "id"},
        f"{sname}, diff-in-means", "none",
        cluster_var="id",
        design_audit=g2_audit, design_code="randomized_experiment",
        inference_spec_id=g2_canonical["spec_id"],
        inference_params={"cluster_var": "id"},
        notes=f"Diff-in-means: reg pts SR, cluster(id). {sname}."
    )

# --- RC: control sets ---
for sname in ['Prolific', 'Berkeley', 'BU']:
    df_s = charity_sample(charity_long, sname)

    # ownpay only (mandatory, no order dummies)
    run_ols_spec(
        "rc/controls/sets/ownpay_only",
        "specification_tree/modules/robustness/controls.md#sets",
        "G2", "pts", "SR", ["ownpay"], "", "",
        df_s, {"CRV1": "id"},
        f"{sname}, ownpay only", "ownpay",
        cluster_var="id",
        design_audit=g2_audit, design_code="randomized_experiment",
        inference_spec_id=g2_canonical["spec_id"],
        inference_params={"cluster_var": "id"},
        axis_block_name="controls",
        axis_block={"spec_id": "rc/controls/sets/ownpay_only", "included": ["ownpay"]},
    )

    # ownpay + order dummies (same as baseline)
    run_ols_spec(
        "rc/controls/sets/ownpay_and_order",
        "specification_tree/modules/robustness/controls.md#sets",
        "G2", "pts", "SR", ["ownpay", "o1", "o2"], "", "",
        df_s, {"CRV1": "id"},
        f"{sname}, ownpay+order", "ownpay + o1 + o2",
        cluster_var="id",
        design_audit=g2_audit, design_code="randomized_experiment",
        inference_spec_id=g2_canonical["spec_id"],
        inference_params={"cluster_var": "id"},
        axis_block_name="controls",
        axis_block={"spec_id": "rc/controls/sets/ownpay_and_order",
                    "included": ["ownpay", "o1", "o2"]},
    )

# --- RC: sample definitions (one per sample) ---
# First round only
for sname in ['Prolific', 'Berkeley', 'BU']:
    df_s = charity_sample(charity_long, sname)
    df_first = df_s[df_s['o1'] == 1].copy()
    run_ols_spec(
        "rc/sample/definition/first_round_only",
        "specification_tree/modules/robustness/sample.md#definition",
        "G2", "pts", "SR", ["ownpay"], "", "",
        df_first, {"CRV1": "id"},
        f"{sname}, first round only", "ownpay",
        cluster_var="id",
        design_audit=g2_audit, design_code="randomized_experiment",
        inference_spec_id=g2_canonical["spec_id"],
        inference_params={"cluster_var": "id"},
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/definition/first_round_only",
                    "restriction": "first round (o1==1)"},
    )

# No attention check filter
for sname in ['Prolific', 'Berkeley', 'BU']:
    df_noatt = charity_long[(charity_long['sample'] == sname) &
                            (charity_long['consistent_b'] == 1)].copy()
    run_ols_spec(
        "rc/sample/definition/no_attention_check_filter",
        "specification_tree/modules/robustness/sample.md#definition",
        "G2", "pts", "SR", ["ownpay", "o1", "o2"], "", "",
        df_noatt, {"CRV1": "id"},
        f"{sname}, no attention check filter", "ownpay + o1 + o2",
        cluster_var="id",
        design_audit=g2_audit, design_code="randomized_experiment",
        inference_spec_id=g2_canonical["spec_id"],
        inference_params={"cluster_var": "id"},
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/definition/no_attention_check_filter",
                    "restriction": "consistent_b==1 only, no attention check"},
    )

# Strict consistency
for sname in ['Prolific', 'Berkeley', 'BU']:
    df_strict = charity_long[(charity_long['sample'] == sname) &
                             (charity_long['flag_attention_check'] == 0) &
                             (charity_long['consistent'] == 1)].copy()
    run_ols_spec(
        "rc/sample/definition/strict_consistency",
        "specification_tree/modules/robustness/sample.md#definition",
        "G2", "pts", "SR", ["ownpay", "o1", "o2"], "", "",
        df_strict, {"CRV1": "id"},
        f"{sname}, strict consistency", "ownpay + o1 + o2",
        cluster_var="id",
        design_audit=g2_audit, design_code="randomized_experiment",
        inference_spec_id=g2_canonical["spec_id"],
        inference_params={"cluster_var": "id"},
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/definition/strict_consistency",
                    "restriction": "consistent==1 (stricter)"},
    )

# Approx monotonic
for sname in ['Prolific', 'Berkeley', 'BU']:
    df_s = charity_sample(charity_long, sname)
    df_mono = df_s[df_s['approx_monotonic'] == 1].copy()
    if len(df_mono) > 10:
        run_ols_spec(
            "rc/sample/definition/approx_monotonic",
            "specification_tree/modules/robustness/sample.md#definition",
            "G2", "pts", "SR", ["ownpay", "o1", "o2"], "", "",
            df_mono, {"CRV1": "id"},
            f"{sname}, approx_monotonic", "ownpay + o1 + o2",
            cluster_var="id",
            design_audit=g2_audit, design_code="randomized_experiment",
            inference_spec_id=g2_canonical["spec_id"],
            inference_params={"cluster_var": "id"},
            axis_block_name="sample",
            axis_block={"spec_id": "rc/sample/definition/approx_monotonic",
                        "restriction": "approx_monotonic==1"},
        )

# Outlier trims on pts
for sname in ['Prolific', 'Berkeley', 'BU']:
    df_s = charity_sample(charity_long, sname)

    # Trim 1/99
    lo, hi = df_s['pts'].quantile([0.01, 0.99])
    df_trim = df_s[(df_s['pts'] >= lo) & (df_s['pts'] <= hi)].copy()
    run_ols_spec(
        "rc/sample/outliers/trim_y_1_99",
        "specification_tree/modules/robustness/sample.md#outliers",
        "G2", "pts", "SR", ["ownpay", "o1", "o2"], "", "",
        df_trim, {"CRV1": "id"},
        f"{sname}, pts trimmed 1-99%", "ownpay + o1 + o2",
        cluster_var="id",
        design_audit=g2_audit, design_code="randomized_experiment",
        inference_spec_id=g2_canonical["spec_id"],
        inference_params={"cluster_var": "id"},
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/outliers/trim_y_1_99"},
    )

    # Trim 5/95
    lo, hi = df_s['pts'].quantile([0.05, 0.95])
    df_trim2 = df_s[(df_s['pts'] >= lo) & (df_s['pts'] <= hi)].copy()
    run_ols_spec(
        "rc/sample/outliers/trim_y_5_95",
        "specification_tree/modules/robustness/sample.md#outliers",
        "G2", "pts", "SR", ["ownpay", "o1", "o2"], "", "",
        df_trim2, {"CRV1": "id"},
        f"{sname}, pts trimmed 5-95%", "ownpay + o1 + o2",
        cluster_var="id",
        design_audit=g2_audit, design_code="randomized_experiment",
        inference_spec_id=g2_canonical["spec_id"],
        inference_params={"cluster_var": "id"},
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/outliers/trim_y_5_95"},
    )

# Pooled: all three samples
df_pooled = charity_long[(charity_long['flag_attention_check'] == 0) &
                         (charity_long['consistent_b'] == 1)].copy()
run_ols_spec(
    "rc/sample/pooled/all_three_samples",
    "specification_tree/modules/robustness/sample.md#pooled",
    "G2", "pts", "SR", ["ownpay", "o1", "o2"], "", "",
    df_pooled, {"CRV1": "id"},
    "All three samples pooled", "ownpay + o1 + o2",
    cluster_var="id",
    design_audit=g2_audit, design_code="randomized_experiment",
    inference_spec_id=g2_canonical["spec_id"],
    inference_params={"cluster_var": "id"},
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/pooled/all_three_samples",
                "restriction": "Prolific + Berkeley + BU pooled"},
)

# Functional form: log(1+pts)
for sname in ['Prolific', 'Berkeley', 'BU']:
    df_s = charity_sample(charity_long, sname)
    df_s['log1p_pts'] = np.log1p(df_s['pts'])
    run_ols_spec(
        "rc/form/outcome/log1p_pts",
        "specification_tree/modules/robustness/functional_form.md",
        "G2", "log1p_pts", "SR", ["ownpay", "o1", "o2"], "", "",
        df_s, {"CRV1": "id"},
        f"{sname}, log(1+pts)", "ownpay + o1 + o2",
        cluster_var="id",
        design_audit=g2_audit, design_code="randomized_experiment",
        inference_spec_id=g2_canonical["spec_id"],
        inference_params={"cluster_var": "id"},
        axis_block_name="functional_form",
        axis_block={"spec_id": "rc/form/outcome/log1p_pts",
                    "transform": "log1p", "original_var": "pts",
                    "interpretation": "semi-elasticity of points to SR"},
    )

# Standardized pts
for sname in ['Prolific', 'Berkeley', 'BU']:
    df_s = charity_sample(charity_long, sname)
    pmean, psd = df_s['pts'].mean(), df_s['pts'].std()
    df_s['std_pts'] = (df_s['pts'] - pmean) / psd
    run_ols_spec(
        "rc/form/outcome/standardized_pts",
        "specification_tree/modules/robustness/functional_form.md",
        "G2", "std_pts", "SR", ["ownpay", "o1", "o2"], "", "",
        df_s, {"CRV1": "id"},
        f"{sname}, standardized pts", "ownpay + o1 + o2",
        cluster_var="id",
        design_audit=g2_audit, design_code="randomized_experiment",
        inference_spec_id=g2_canonical["spec_id"],
        inference_params={"cluster_var": "id"},
        axis_block_name="functional_form",
        axis_block={"spec_id": "rc/form/outcome/standardized_pts",
                    "transform": "standardized", "interpretation": "effect in SD units"},
    )

# Pts in hundreds
for sname in ['Prolific', 'Berkeley', 'BU']:
    df_s = charity_sample(charity_long, sname)
    df_s['pts_hundreds'] = df_s['pts'] / 100.0
    run_ols_spec(
        "rc/preprocess/coding/pts_in_hundreds",
        "specification_tree/modules/robustness/preprocess.md#coding",
        "G2", "pts_hundreds", "SR", ["ownpay", "o1", "o2"], "", "",
        df_s, {"CRV1": "id"},
        f"{sname}, pts in hundreds", "ownpay + o1 + o2",
        cluster_var="id",
        design_audit=g2_audit, design_code="randomized_experiment",
        inference_spec_id=g2_canonical["spec_id"],
        inference_params={"cluster_var": "id"},
        axis_block_name="preprocess",
        axis_block={"spec_id": "rc/preprocess/coding/pts_in_hundreds",
                    "scale_factor": 0.01},
    )

# Individual FE (within-subject, absorb person id)
for sname in ['Prolific', 'Berkeley', 'BU']:
    df_s = charity_sample(charity_long, sname)
    # need str id for pyfixest FE
    df_s['id_str'] = df_s['id'].astype(str)
    run_ols_spec(
        "rc/fe/add/individual_fe",
        "specification_tree/modules/robustness/fixed_effects.md#add",
        "G2", "pts", "SR", ["ownpay"], "id_str", "id_str",
        df_s, {"CRV1": "id"},
        f"{sname}, individual FE", "ownpay (o1/o2 absorbed by FE)",
        cluster_var="id",
        design_audit=g2_audit, design_code="randomized_experiment",
        inference_spec_id=g2_canonical["spec_id"],
        inference_params={"cluster_var": "id"},
        axis_block_name="fixed_effects",
        axis_block={"spec_id": "rc/fe/add/individual_fe",
                    "added": ["id"], "notes": "absorbs individual-level variation"},
    )

print(f"G2 specs done. Total results so far: {len(results)}")


# ============================================================================
# DATA LOADING: G3 WTP ELICITATION (YMCA + CHARITY)
# ============================================================================
print("\n=== G3: WTP elicitation specs ===")

g3_audit = surface_obj["baseline_groups"][2]["design_audit"]
g3_canonical = surface_obj["baseline_groups"][2]["inference_plan"]["canonical"]

# ------------ YMCA WTP data: reshape long ---------
# Use the experiment build data (already have ymca_exp)
# Reshape long: one row per (id, interval)
ymca_wtp_rows = []
visits_map = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 56: 5.5, 78: 7.5,
    912: 10.5, 1317: 15, 1822: 20, 2328: 26.5
}
visitsL_map = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 56: 5, 78: 7,
    912: 9, 1317: 13, 1822: 18, 2328: 23
}
visitsU_map = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 56: 6, 78: 8,
    912: 12, 1317: 17, 1822: 22, 2328: 31
}

for idx, row in ymca_exp.iterrows():
    for interval_key in intervals:
        wtp_col = f'wtp{interval_key}'
        visits_val = visits_map[interval_key]
        r = {
            'id': row['id'],
            'ymca_id': row['ymca_id'],
            'coherent_sample': row.get('coherent_sample', np.nan),
            'monotonic_sample': row.get('monotonic_sample', np.nan),
            'robust_sample': row.get('robust_sample', np.nan),
            'image': row.get('image', np.nan),
            'past': row.get('past', np.nan),
            'attendance': row.get('attendance', np.nan),
            'beliefs_w_image': row.get('beliefs_w_image', np.nan),
            'wtp': row.get(wtp_col, np.nan),
            'visits': visits_val,
            'visits2': visits_val ** 2,
            'visitsL': visitsL_map[interval_key],
            'visitsU': visitsU_map[interval_key],
            'interval': interval_key,
        }
        ymca_wtp_rows.append(r)

ymca_wtp = pd.DataFrame(ymca_wtp_rows)
ymca_wtp['closeb'] = np.abs(ymca_wtp['visits'] - ymca_wtp['beliefs_w_image'])
ymca_wtp['closep'] = np.abs(ymca_wtp['visits'] - ymca_wtp['past'])

# interval_idx for YMCA
interval_idx_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 56: 5, 78: 6, 912: 7, 1317: 8, 1822: 9, 2328: 10}
ymca_wtp['interval_idx'] = ymca_wtp['interval'].map(interval_idx_map)
ymca_wtp['interval_idx2'] = ymca_wtp['interval_idx'] ** 2

# ln(1+visits)
ymca_wtp['ln_visits'] = np.log1p(ymca_wtp['visits'])
ymca_wtp['ln_visits2'] = ymca_wtp['ln_visits'] ** 2

# Compute avg_att_mainpop from structural_build equivalent
# Use the full-pop YMCA data excluding experiment participants
ymca_fullpop_noexp = ymca_full[~ymca_full['ymca_id'].isin(ymca_exp['ymca_id'])].copy()
ymca_fullpop_noexp.loc[ymca_fullpop_noexp['attendance'].isna(), 'attendance'] = 0
avg_att_mainpop = ymca_fullpop_noexp['attendance'].mean()
sd_mainpop = ymca_fullpop_noexp['attendance'].std()
print(f"YMCA avg_att_mainpop = {avg_att_mainpop:.2f}, sd_mainpop = {sd_mainpop:.2f}")

print(f"YMCA WTP long: {len(ymca_wtp)} obs")

# ------------ Charity WTP data: reshape long ---------
charity_wtp_rows = []
for idx, row in charity.iterrows():
    for i in range(18):
        wtp_col = f'wtp{i}'
        r = {
            'id': row['id'],
            'sample': row['sample'],
            'flag_attention_check': row['flag_attention_check'],
            'consistent_b': row['consistent_b'],
            'consistent': row.get('consistent', np.nan),
            'approx_monotonic': row.get('approx_monotonic', np.nan),
            'anompts': row.get('anompts', np.nan),
            'recogpts': row.get('recogpts', np.nan),
            'wtp': row.get(wtp_col, np.nan),
            'interval': i,
            'interval_raw': i + 0.5,  # in 100s of points, coded as interval+0.5
        }
        charity_wtp_rows.append(r)

charity_wtp = pd.DataFrame(charity_wtp_rows)
charity_wtp['interval_sq'] = charity_wtp['interval_raw'] ** 2

# Compute close indicator for charity: avg points within 500 of interval midpoint
charity_wtp['interval_midpoint'] = (charity_wtp['interval'] + 0.5) * 100
charity_wtp['avg_pts'] = (charity_wtp['anompts'] + charity_wtp['recogpts']) / 2.0
charity_wtp['close'] = ((charity_wtp['avg_pts'] <= charity_wtp['interval_midpoint'] + 500) &
                        (charity_wtp['avg_pts'] >= charity_wtp['interval_midpoint'] - 500)).astype(int)

print(f"Charity WTP long: {len(charity_wtp)} obs")


# ============================================================================
# G3: WTP SPECIFICATIONS
# ============================================================================

# --- YMCA Baselines ---
# Table 3, Col 2 (Coherent): reg wtp visits visits2, cluster(id)
ymca_coh_wtp = ymca_wtp[ymca_wtp['coherent_sample'] == 1].copy()

run_id_g3_b1, *_ = run_ols_spec(
    "baseline__table3_col2_coh_ymca",
    "specification_tree/designs/randomized_experiment.md",
    "G3", "wtp", "visits", ["visits2"], "", "",
    ymca_coh_wtp, {"CRV1": "id"},
    "YMCA coherent, all intervals", "visits2",
    cluster_var="id",
    design_audit=g3_audit, design_code="randomized_experiment",
    inference_spec_id=g3_canonical["spec_id"],
    inference_params={"cluster_var": "id"},
    notes="Table 3 Col 2: reg wtp visits visits2, cluster(id). Coherent sample."
)

# --- Charity baselines ---
for sname, spec_suffix in [('Prolific', 'prolific'), ('Berkeley', 'berkeley'), ('BU', 'bu')]:
    df_s = charity_wtp[(charity_wtp['sample'] == sname) &
                       (charity_wtp['flag_attention_check'] == 0) &
                       (charity_wtp['consistent_b'] == 1) &
                       (charity_wtp['interval'] < 17)].copy()
    run_id_tmp, *_ = run_ols_spec(
        f"baseline__table6_col{'246'['prolific berkeley bu'.split().index(spec_suffix)]}_{spec_suffix}",
        "specification_tree/designs/randomized_experiment.md",
        "G3", "wtp", "interval_raw", ["interval_sq"], "", "",
        df_s, {"CRV1": "id"},
        f"{sname}, excl top interval", "interval_sq",
        cluster_var="id",
        design_audit=g3_audit, design_code="randomized_experiment",
        inference_spec_id=g3_canonical["spec_id"],
        inference_params={"cluster_var": "id"},
        notes=f"Table 6: reg wtp c.interval##c.interval, cluster(id). {sname}. Excl top interval."
    )

    # Inference variant: HC1
    add_inference_row(
        run_id_tmp, "infer/se/hc/hc1",
        "specification_tree/modules/inference/se.md#hc1",
        "G3", "wtp", "interval_raw", ["interval_sq"], "",
        df_s, "hetero",
        notes=f"HC1 SEs on G3 baseline, {sname}"
    )

# --- G3 RC specs ---
# OLS linear (YMCA)
run_ols_spec(
    "rc/form/estimator/ols_linear",
    "specification_tree/modules/robustness/functional_form.md",
    "G3", "wtp", "visits", [], "", "",
    ymca_coh_wtp, {"CRV1": "id"},
    "YMCA coherent, OLS linear", "none",
    cluster_var="id",
    design_audit=g3_audit, design_code="randomized_experiment",
    inference_spec_id=g3_canonical["spec_id"],
    inference_params={"cluster_var": "id"},
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/estimator/ols_linear",
                "form": "linear", "interpretation": "slope of WTP on visits"},
)

# OLS linear (charity, per sample)
for sname in ['Prolific', 'Berkeley', 'BU']:
    df_s = charity_wtp[(charity_wtp['sample'] == sname) &
                       (charity_wtp['flag_attention_check'] == 0) &
                       (charity_wtp['consistent_b'] == 1) &
                       (charity_wtp['interval'] < 17)].copy()
    run_ols_spec(
        "rc/form/estimator/ols_linear",
        "specification_tree/modules/robustness/functional_form.md",
        "G3", "wtp", "interval_raw", [], "", "",
        df_s, {"CRV1": "id"},
        f"{sname}, OLS linear, excl top interval", "none",
        cluster_var="id",
        design_audit=g3_audit, design_code="randomized_experiment",
        inference_spec_id=g3_canonical["spec_id"],
        inference_params={"cluster_var": "id"},
        axis_block_name="functional_form",
        axis_block={"spec_id": "rc/form/estimator/ols_linear",
                    "form": "linear", "interpretation": "slope of WTP on interval"},
    )

# Tobit quadratic (YMCA) - use manual Tobit
def run_tobit_spec(
    spec_id, spec_tree_path, baseline_group_id,
    outcome_var, treatment_var, controls, data,
    sample_desc, controls_desc, cluster_var="id",
    design_audit=None, design_code="randomized_experiment",
    inference_spec_id="infer/se/cluster/id",
    inference_params=None, axis_block_name=None, axis_block=None, notes=""
):
    """Run Tobit regression via MLE (scipy.optimize)."""
    run_id = next_run_id()
    try:
        regvars = [outcome_var, treatment_var] + list(controls)
        df_reg = data.dropna(subset=regvars).copy()

        y = df_reg[outcome_var].values.astype(float)
        X_vars = [treatment_var] + list(controls)
        X = np.column_stack([np.ones(len(y))] + [df_reg[v].values.astype(float) for v in X_vars])

        ll = y.min()  # lower limit = observed min (or 0 for YMCA, data-driven for charity)
        ul = y.max()  # upper limit = observed max

        # Tobit log-likelihood
        def tobit_negll(params):
            beta = params[:-1]
            log_sigma = params[-1]
            sigma = np.exp(log_sigma)
            xb = X @ beta

            censored_lower = (y <= ll)
            censored_upper = (y >= ul)
            uncensored = ~censored_lower & ~censored_upper

            ll_cens_lo = norm.logcdf((ll - xb[censored_lower]) / sigma).sum() if censored_lower.any() else 0
            ll_cens_hi = norm.logsf((ul - xb[censored_upper]) / sigma).sum() if censored_upper.any() else 0
            ll_unc = (-0.5 * np.log(2 * np.pi) - log_sigma -
                      0.5 * ((y[uncensored] - xb[uncensored]) / sigma) ** 2).sum() if uncensored.any() else 0

            return -(ll_cens_lo + ll_cens_hi + ll_unc)

        # Initialize with OLS
        from numpy.linalg import lstsq
        beta_init, *_ = lstsq(X, y, rcond=None)
        resid = y - X @ beta_init
        log_sigma_init = np.log(max(resid.std(), 0.01))
        init = np.append(beta_init, log_sigma_init)

        res = minimize(tobit_negll, init, method="BFGS")

        if not res.success:
            # try Nelder-Mead
            res = minimize(tobit_negll, init, method="Nelder-Mead", options={"maxiter": 10000})

        beta_hat = res.x[:-1]
        sigma_hat = np.exp(res.x[-1])

        # Numerical Hessian for SEs
        from scipy.optimize import approx_fprime
        n_params = len(res.x)
        H = np.zeros((n_params, n_params))
        eps = 1e-5
        for i in range(n_params):
            def grad_i(p):
                return approx_fprime(p, tobit_negll, eps)[i]
            H[i] = approx_fprime(res.x, grad_i, eps)

        try:
            se_all = np.sqrt(np.diag(np.linalg.inv(H)))
        except np.linalg.LinAlgError:
            se_all = np.full(n_params, np.nan)

        # Extract treatment coefficient (index 1, after constant)
        coef_val = float(beta_hat[1])
        se_val = float(se_all[1])
        z_stat = coef_val / se_val if se_val > 0 else np.nan
        pval = float(2 * (1 - norm.cdf(abs(z_stat))))

        ci_lower = coef_val - 1.96 * se_val
        ci_upper = coef_val + 1.96 * se_val
        nobs = len(y)

        var_names = ['_cons'] + X_vars
        all_coefs = {var_names[i]: float(beta_hat[i]) for i in range(len(var_names))}
        all_coefs['sigma'] = float(sigma_hat)

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": inference_spec_id,
                       "params": inference_params or {},
                       "notes": "Tobit MLE, Hessian-based SEs (not clustered)"},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={design_code: design_audit or {}},
            axis_block_name=axis_block_name,
            axis_block=axis_block,
        )
        if notes:
            payload["notes"] = notes

        results.append({
            "paper_id": PAPER_ID,
            "spec_run_id": run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": treatment_var,
            "coefficient": coef_val,
            "std_error": se_val,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": nobs,
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": "",
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
            "run_success": 1,
            "run_error": ""
        })
        return run_id, coef_val, se_val, pval, nobs

    except Exception as e:
        err_msg = str(e)[:240]
        err_details = error_details_from_exception(e, stage="tobit_estimation")
        payload = make_failure_payload(
            error=err_msg, error_details=err_details,
            software=SW_BLOCK, surface_hash=SURFACE_HASH
        )
        results.append({
            "paper_id": PAPER_ID,
            "spec_run_id": run_id,
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
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": "",
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
            "run_success": 0,
            "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


# Tobit quadratic (YMCA)
run_tobit_spec(
    "rc/form/estimator/tobit_quadratic_ymca",
    "specification_tree/modules/robustness/functional_form.md",
    "G3", "wtp", "visits", ["visits2"],
    ymca_coh_wtp,
    "YMCA coherent, Tobit quadratic", "visits2",
    cluster_var="id",
    design_audit=g3_audit, design_code="randomized_experiment",
    inference_spec_id=g3_canonical["spec_id"],
    inference_params={"cluster_var": "id"},
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/estimator/tobit_quadratic_ymca",
                "estimator": "tobit", "form": "quadratic",
                "interpretation": "Tobit with lower/upper censoring on WTP"},
)

# Tobit linear (YMCA)
run_tobit_spec(
    "rc/form/estimator/tobit_linear_ymca",
    "specification_tree/modules/robustness/functional_form.md",
    "G3", "wtp", "visits", [],
    ymca_coh_wtp,
    "YMCA coherent, Tobit linear", "none",
    cluster_var="id",
    design_audit=g3_audit, design_code="randomized_experiment",
    inference_spec_id=g3_canonical["spec_id"],
    inference_params={"cluster_var": "id"},
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/estimator/tobit_linear_ymca",
                "estimator": "tobit", "form": "linear",
                "interpretation": "Tobit linear WTP on visits"},
)

# ln(1+visits) quadratic (YMCA)
run_ols_spec(
    "rc/form/outcome/ln_visits_quadratic_ymca",
    "specification_tree/modules/robustness/functional_form.md",
    "G3", "wtp", "ln_visits", ["ln_visits2"], "", "",
    ymca_coh_wtp, {"CRV1": "id"},
    "YMCA coherent, ln(1+visits) quadratic", "ln_visits2",
    cluster_var="id",
    design_audit=g3_audit, design_code="randomized_experiment",
    inference_spec_id=g3_canonical["spec_id"],
    inference_params={"cluster_var": "id"},
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/ln_visits_quadratic_ymca",
                "form": "quadratic_log", "interpretation": "WTP on log visits"},
)

# Interval index quadratic (YMCA)
run_ols_spec(
    "rc/form/outcome/interval_idx_quadratic_ymca",
    "specification_tree/modules/robustness/functional_form.md",
    "G3", "wtp", "interval_idx", ["interval_idx2"], "", "",
    ymca_coh_wtp, {"CRV1": "id"},
    "YMCA coherent, interval index quadratic", "interval_idx2",
    cluster_var="id",
    design_audit=g3_audit, design_code="randomized_experiment",
    inference_spec_id=g3_canonical["spec_id"],
    inference_params={"cluster_var": "id"},
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/interval_idx_quadratic_ymca",
                "form": "quadratic_idx", "interpretation": "WTP on interval index (0-10)"},
)

# Interval index linear (YMCA)
run_ols_spec(
    "rc/form/outcome/interval_idx_linear_ymca",
    "specification_tree/modules/robustness/functional_form.md",
    "G3", "wtp", "interval_idx", [], "", "",
    ymca_coh_wtp, {"CRV1": "id"},
    "YMCA coherent, interval index linear", "none",
    cluster_var="id",
    design_audit=g3_audit, design_code="randomized_experiment",
    inference_spec_id=g3_canonical["spec_id"],
    inference_params={"cluster_var": "id"},
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/interval_idx_linear_ymca",
                "form": "linear_idx", "interpretation": "WTP on interval index (linear)"},
)

# Sample restrictions (YMCA WTP)
# Monotonic sample
ymca_mon_wtp = ymca_wtp[ymca_wtp['monotonic_sample'] == 1].copy()
run_ols_spec(
    "rc/sample/definition/monotonic_sample_ymca",
    "specification_tree/modules/robustness/sample.md#definition",
    "G3", "wtp", "visits", ["visits2"], "", "",
    ymca_mon_wtp, {"CRV1": "id"},
    "YMCA monotonic, quadratic", "visits2",
    cluster_var="id",
    design_audit=g3_audit, design_code="randomized_experiment",
    inference_spec_id=g3_canonical["spec_id"],
    inference_params={"cluster_var": "id"},
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/definition/monotonic_sample_ymca",
                "restriction": "monotonic_sample==1"},
)

# Close to beliefs (within 4)
ymca_close4 = ymca_coh_wtp[ymca_coh_wtp['closeb'] <= 4].copy()
run_ols_spec(
    "rc/sample/definition/close_to_beliefs_4_ymca",
    "specification_tree/modules/robustness/sample.md#definition",
    "G3", "wtp", "visits", ["visits2"], "", "",
    ymca_close4, {"CRV1": "id"},
    "YMCA coherent, close to beliefs (<=4)", "visits2",
    cluster_var="id",
    design_audit=g3_audit, design_code="randomized_experiment",
    inference_spec_id=g3_canonical["spec_id"],
    inference_params={"cluster_var": "id"},
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/definition/close_to_beliefs_4_ymca",
                "restriction": "|visits - beliefs_w_image| <= 4"},
)

# Close to beliefs (exact interval match)
ymca_exact = ymca_coh_wtp[
    (ymca_coh_wtp['beliefs_w_image'] <= ymca_coh_wtp['visitsU']) &
    (ymca_coh_wtp['beliefs_w_image'] >= ymca_coh_wtp['visitsL'])
].copy()
run_ols_spec(
    "rc/sample/definition/close_to_beliefs_exact_ymca",
    "specification_tree/modules/robustness/sample.md#definition",
    "G3", "wtp", "visits", ["visits2"], "", "",
    ymca_exact, {"CRV1": "id"},
    "YMCA coherent, beliefs in exact interval", "visits2",
    cluster_var="id",
    design_audit=g3_audit, design_code="randomized_experiment",
    inference_spec_id=g3_canonical["spec_id"],
    inference_params={"cluster_var": "id"},
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/definition/close_to_beliefs_exact_ymca",
                "restriction": "beliefs_w_image in [visitsL, visitsU]"},
)

# Close to past (within 4)
ymca_closep4 = ymca_coh_wtp[ymca_coh_wtp['closep'] <= 4].copy()
run_ols_spec(
    "rc/sample/definition/close_to_past_4_ymca",
    "specification_tree/modules/robustness/sample.md#definition",
    "G3", "wtp", "visits", ["visits2"], "", "",
    ymca_closep4, {"CRV1": "id"},
    "YMCA coherent, close to past att (<=4)", "visits2",
    cluster_var="id",
    design_audit=g3_audit, design_code="randomized_experiment",
    inference_spec_id=g3_canonical["spec_id"],
    inference_params={"cluster_var": "id"},
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/definition/close_to_past_4_ymca",
                "restriction": "|visits - past| <= 4"},
)

# Exclude top interval (YMCA)
ymca_excl_top = ymca_coh_wtp[ymca_coh_wtp['visitsL'] <= 22].copy()
run_ols_spec(
    "rc/sample/definition/excl_top_interval_ymca",
    "specification_tree/modules/robustness/sample.md#definition",
    "G3", "wtp", "visits", ["visits2"], "", "",
    ymca_excl_top, {"CRV1": "id"},
    "YMCA coherent, excl top interval (23-28)", "visits2",
    cluster_var="id",
    design_audit=g3_audit, design_code="randomized_experiment",
    inference_spec_id=g3_canonical["spec_id"],
    inference_params={"cluster_var": "id"},
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/definition/excl_top_interval_ymca",
                "restriction": "visitsL <= 22"},
)

# Exclude top 2 intervals (YMCA)
ymca_excl_top2 = ymca_coh_wtp[ymca_coh_wtp['visitsL'] <= 17].copy()
run_ols_spec(
    "rc/sample/definition/excl_top_two_intervals_ymca",
    "specification_tree/modules/robustness/sample.md#definition",
    "G3", "wtp", "visits", ["visits2"], "", "",
    ymca_excl_top2, {"CRV1": "id"},
    "YMCA coherent, excl top 2 intervals", "visits2",
    cluster_var="id",
    design_audit=g3_audit, design_code="randomized_experiment",
    inference_spec_id=g3_canonical["spec_id"],
    inference_params={"cluster_var": "id"},
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/definition/excl_top_two_intervals_ymca",
                "restriction": "visitsL <= 17"},
)

# Charity: include top interval
for sname in ['Prolific', 'Berkeley', 'BU']:
    df_s = charity_wtp[(charity_wtp['sample'] == sname) &
                       (charity_wtp['flag_attention_check'] == 0) &
                       (charity_wtp['consistent_b'] == 1)].copy()  # no interval < 17 filter
    run_ols_spec(
        "rc/sample/definition/include_top_interval_charity",
        "specification_tree/modules/robustness/sample.md#definition",
        "G3", "wtp", "interval_raw", ["interval_sq"], "", "",
        df_s, {"CRV1": "id"},
        f"{sname}, include top interval", "interval_sq",
        cluster_var="id",
        design_audit=g3_audit, design_code="randomized_experiment",
        inference_spec_id=g3_canonical["spec_id"],
        inference_params={"cluster_var": "id"},
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/definition/include_top_interval_charity",
                    "restriction": "all intervals including 1700+"},
    )

# Close to score (charity)
for sname in ['Prolific', 'Berkeley', 'BU']:
    df_s = charity_wtp[(charity_wtp['sample'] == sname) &
                       (charity_wtp['flag_attention_check'] == 0) &
                       (charity_wtp['consistent_b'] == 1) &
                       (charity_wtp['interval'] < 17) &
                       (charity_wtp['close'] == 1)].copy()
    run_ols_spec(
        "rc/sample/definition/close_to_score_charity",
        "specification_tree/modules/robustness/sample.md#definition",
        "G3", "wtp", "interval_raw", ["interval_sq"], "", "",
        df_s, {"CRV1": "id"},
        f"{sname}, close to score", "interval_sq",
        cluster_var="id",
        design_audit=g3_audit, design_code="randomized_experiment",
        inference_spec_id=g3_canonical["spec_id"],
        inference_params={"cluster_var": "id"},
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/definition/close_to_score_charity",
                    "restriction": "avg_pts within 500 of interval midpoint"},
    )

# No consistency filter (charity)
for sname in ['Prolific', 'Berkeley', 'BU']:
    df_s = charity_wtp[(charity_wtp['sample'] == sname) &
                       (charity_wtp['flag_attention_check'] == 0) &
                       (charity_wtp['interval'] < 17)].copy()
    run_ols_spec(
        "rc/sample/definition/no_consistency_filter_charity",
        "specification_tree/modules/robustness/sample.md#definition",
        "G3", "wtp", "interval_raw", ["interval_sq"], "", "",
        df_s, {"CRV1": "id"},
        f"{sname}, no consistency filter", "interval_sq",
        cluster_var="id",
        design_audit=g3_audit, design_code="randomized_experiment",
        inference_spec_id=g3_canonical["spec_id"],
        inference_params={"cluster_var": "id"},
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/definition/no_consistency_filter_charity",
                    "restriction": "no consistent_b filter"},
    )

# Trim WTP 1/99
for sname in ['Prolific', 'Berkeley', 'BU']:
    df_s = charity_wtp[(charity_wtp['sample'] == sname) &
                       (charity_wtp['flag_attention_check'] == 0) &
                       (charity_wtp['consistent_b'] == 1) &
                       (charity_wtp['interval'] < 17)].copy()
    lo, hi = df_s['wtp'].quantile([0.01, 0.99])
    df_trim = df_s[(df_s['wtp'] >= lo) & (df_s['wtp'] <= hi)].copy()
    run_ols_spec(
        "rc/sample/outliers/trim_wtp_1_99",
        "specification_tree/modules/robustness/sample.md#outliers",
        "G3", "wtp", "interval_raw", ["interval_sq"], "", "",
        df_trim, {"CRV1": "id"},
        f"{sname}, WTP trimmed 1-99%", "interval_sq",
        cluster_var="id",
        design_audit=g3_audit, design_code="randomized_experiment",
        inference_spec_id=g3_canonical["spec_id"],
        inference_params={"cluster_var": "id"},
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/outliers/trim_wtp_1_99"},
    )

# Trim WTP 5/95
for sname in ['Prolific', 'Berkeley', 'BU']:
    df_s = charity_wtp[(charity_wtp['sample'] == sname) &
                       (charity_wtp['flag_attention_check'] == 0) &
                       (charity_wtp['consistent_b'] == 1) &
                       (charity_wtp['interval'] < 17)].copy()
    lo, hi = df_s['wtp'].quantile([0.05, 0.95])
    df_trim = df_s[(df_s['wtp'] >= lo) & (df_s['wtp'] <= hi)].copy()
    run_ols_spec(
        "rc/sample/outliers/trim_wtp_5_95",
        "specification_tree/modules/robustness/sample.md#outliers",
        "G3", "wtp", "interval_raw", ["interval_sq"], "", "",
        df_trim, {"CRV1": "id"},
        f"{sname}, WTP trimmed 5-95%", "interval_sq",
        cluster_var="id",
        design_audit=g3_audit, design_code="randomized_experiment",
        inference_spec_id=g3_canonical["spec_id"],
        inference_params={"cluster_var": "id"},
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/outliers/trim_wtp_5_95"},
    )

# Pooled charity samples
df_pooled_wtp = charity_wtp[(charity_wtp['flag_attention_check'] == 0) &
                            (charity_wtp['consistent_b'] == 1) &
                            (charity_wtp['interval'] < 17)].copy()
run_ols_spec(
    "rc/sample/pooled/all_charity_samples",
    "specification_tree/modules/robustness/sample.md#pooled",
    "G3", "wtp", "interval_raw", ["interval_sq"], "", "",
    df_pooled_wtp, {"CRV1": "id"},
    "All charity samples pooled, excl top interval", "interval_sq",
    cluster_var="id",
    design_audit=g3_audit, design_code="randomized_experiment",
    inference_spec_id=g3_canonical["spec_id"],
    inference_params={"cluster_var": "id"},
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/pooled/all_charity_samples",
                "restriction": "all 3 charity samples pooled"},
)

# YMCA WTP outlier trims
lo, hi = ymca_coh_wtp['wtp'].quantile([0.01, 0.99])
ymca_trim_1_99 = ymca_coh_wtp[(ymca_coh_wtp['wtp'] >= lo) & (ymca_coh_wtp['wtp'] <= hi)].copy()
run_ols_spec(
    "rc/sample/outliers/trim_wtp_1_99",
    "specification_tree/modules/robustness/sample.md#outliers",
    "G3", "wtp", "visits", ["visits2"], "", "",
    ymca_trim_1_99, {"CRV1": "id"},
    "YMCA coherent, WTP trimmed 1-99%", "visits2",
    cluster_var="id",
    design_audit=g3_audit, design_code="randomized_experiment",
    inference_spec_id=g3_canonical["spec_id"],
    inference_params={"cluster_var": "id"},
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_wtp_1_99"},
)

lo, hi = ymca_coh_wtp['wtp'].quantile([0.05, 0.95])
ymca_trim_5_95 = ymca_coh_wtp[(ymca_coh_wtp['wtp'] >= lo) & (ymca_coh_wtp['wtp'] <= hi)].copy()
run_ols_spec(
    "rc/sample/outliers/trim_wtp_5_95",
    "specification_tree/modules/robustness/sample.md#outliers",
    "G3", "wtp", "visits", ["visits2"], "", "",
    ymca_trim_5_95, {"CRV1": "id"},
    "YMCA coherent, WTP trimmed 5-95%", "visits2",
    cluster_var="id",
    design_audit=g3_audit, design_code="randomized_experiment",
    inference_spec_id=g3_canonical["spec_id"],
    inference_params={"cluster_var": "id"},
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_wtp_5_95"},
)

print(f"G3 specs done. Total results: {len(results)}")
print(f"Inference results: {len(inference_results)}")

# ============================================================================
# WRITE OUTPUTS
# ============================================================================
print("\n=== Writing outputs ===")

# specification_results.csv
df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(DATA_DIR, "specification_results.csv"), index=False)
print(f"Wrote specification_results.csv: {len(df_results)} rows")

# inference_results.csv
df_inference = pd.DataFrame(inference_results)
df_inference.to_csv(os.path.join(DATA_DIR, "inference_results.csv"), index=False)
print(f"Wrote inference_results.csv: {len(df_inference)} rows")

# Count successes/failures
n_success = (df_results['run_success'] == 1).sum()
n_fail = (df_results['run_success'] == 0).sum()
n_infer_success = (df_inference['run_success'] == 1).sum() if len(df_inference) > 0 else 0
n_infer_fail = (df_inference['run_success'] == 0).sum() if len(df_inference) > 0 else 0

# Count by group
g1_count = (df_results['baseline_group_id'] == 'G1').sum()
g2_count = (df_results['baseline_group_id'] == 'G2').sum()
g3_count = (df_results['baseline_group_id'] == 'G3').sum()

# SPECIFICATION_SEARCH.md
search_md = f"""# Specification Search: {PAPER_ID}

## Surface Summary

- **Paper ID**: {PAPER_ID}
- **Surface hash**: {SURFACE_HASH}
- **Baseline groups**: 3
  - G1: YMCA field experiment (attendance ~ image)
  - G2: Charity real-effort experiment (pts ~ SR)
  - G3: WTP elicitation (wtp ~ visits/interval, quadratic)
- **Design**: randomized_experiment (all groups)
- **Seed**: 145141

## Execution Summary

### specification_results.csv
- **Total rows**: {len(df_results)}
- **Successful**: {n_success}
- **Failed**: {n_fail}
- **G1 (YMCA attendance)**: {g1_count} specs
- **G2 (Charity real-effort)**: {g2_count} specs
- **G3 (WTP elicitation)**: {g3_count} specs

### inference_results.csv
- **Total rows**: {len(df_inference)}
- **Successful**: {n_infer_success}
- **Failed**: {n_infer_fail}

## Specifications Executed

### G1: YMCA Attendance (HC1 robust SEs)
- 2 baseline specs (Table 2 Cols 2-3, coherent sample)
- 1 design variant: diff-in-means
- 3 control sets: none, past only, past+beliefs
- 2 sample definitions: monotonic, robust (full excl BDM)
- 3 outlier treatments: trim 1/99, trim 5/95, winsorize 1/99
- 3 functional forms: log1p, asinh, standardized
- 2 preprocessing: topcode at 22, topcode at 15

### G2: Charity Real-Effort (clustered SEs at individual level)
- 3 baseline specs (Table 5 Cols 1-3: Prolific, Berkeley, BU)
- 3 design variants: diff-in-means (per sample)
- 6 control sets: ownpay-only and ownpay+order (per sample)
- 12 sample definitions: first-round-only, no-attention-check, strict-consistency,
  approx-monotonic (each per sample)
- 6 outlier treatments: trim 1/99, trim 5/95 (per sample)
- 1 pooled sample spec
- 9 functional forms/preprocessing: log1p, standardized, pts-in-hundreds (per sample)
- 3 individual FE specs (per sample)

### G3: WTP Elicitation (clustered SEs at individual level)
- 4 baseline specs (Table 3 Col 2 YMCA; Table 6 Cols 2/4/6 charity)
- 4 OLS linear (YMCA + 3 charity samples)
- 2 Tobit specs (YMCA quadratic + linear, MLE with Hessian SEs)
- 3 YMCA functional form: ln_visits quadratic, interval_idx quadratic/linear
- 10 YMCA sample restrictions: monotonic, close-to-beliefs-4, exact-belief-match,
  close-to-past-4, excl-top-interval, excl-top-2-intervals, WTP trim 1/99, 5/95
- 9 charity sample restrictions: include-top-interval, close-to-score, no-consistency (per sample)
- 6 charity WTP trim: 1/99, 5/95 (per sample)
- 1 pooled charity WTP spec

### Inference variants
- G1: HC3 on baseline
- G2: HC1 (no clustering) on each baseline
- G3: HC1 (no clustering) on each charity baseline

## Skipped / Not Feasible
- Tobit for charity experiments: not listed in core surface rc_spec_ids (charity code does not run Tobit)
- BDM arm analysis: excluded per surface (BDM arm excluded from reduced-form)
- QM221 sample: separate pilot sample, not in the surface specification

## Software Stack
- Python {sys.version.split()[0]}
- pandas {pd.__version__}
- numpy {np.__version__}
- pyfixest {pf.__version__}
- scipy (for Tobit MLE)

## Data Construction Notes
- YMCA data built from raw CSV files (scans, membership, survey, treatment assignment)
- Attendance = count of unique scan days during June 15 - July 15, 2017
- Past attendance = total scans in pre-period (May 2016 - May 2017) / 13
- Charity data built from raw CSV files (Prolific, Berkeley, BU QM222)
- WTP variables: negative (pay to avoid) or positive (pay for recognition)
- Coherent sample: excludes incoherent WTP respondents (>2 switches, or starts yes with 2 switches)
- Monotonic sample: excludes anyone with WTP decreasing at any point
- avg_att_mainpop = {avg_att_mainpop:.4f} (YMCA full population excluding experiment participants)
"""

with open(os.path.join(DATA_DIR, "SPECIFICATION_SEARCH.md"), "w") as f:
    f.write(search_md)
print("Wrote SPECIFICATION_SEARCH.md")

print(f"\nDone! {len(df_results)} spec results + {len(df_inference)} inference results.")
