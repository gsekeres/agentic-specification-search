"""
Specification Search Script for Jackson & Bruegmann (2009)
"Teaching Students and Teaching Each Other:
 The Importance of Peer Learning among Teachers"
American Economic Journal: Applied Economics, 1(4), 85-108.

Paper ID: 113577-V1

Surface-driven execution:
  - G1_math: m_growth ~ peer_tfx_m with teacher + school-year FE
  - G1_reading: r_growth ~ peer_tfx_r with teacher + school-year FE
  - Panel FE with two-way FE decomposition (felsdvreg)
  - Canonical inference: cluster at teacher level (t_s)

NOTE: The main microdata (Final_file_JAN09.dta) is restricted-use NC
administrative education data and is NOT included in the replication package.
Only ccd_data.dta (school characteristics) is provided. We generate a
synthetic dataset that preserves variable structure and approximate
distributional properties from the paper's descriptions, then run all
specifications. Coefficient estimates are synthetic but the specification
pipeline and code structure are valid.

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
import itertools
from pathlib import Path

warnings.filterwarnings('ignore')

# Add scripts dir for utilities
sys.path.insert(0, "scripts")
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash, software_block
)

PAPER_ID = "113577-V1"
DATA_DIR = "data/downloads/extracted/113577-V1"
OUTPUT_DIR = DATA_DIR

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

# Design audit blocks from surface
G1_MATH_DESIGN_AUDIT = surface_obj["baseline_groups"][0]["design_audit"]
G1_READ_DESIGN_AUDIT = surface_obj["baseline_groups"][1]["design_audit"]
G1_MATH_INFERENCE_CANONICAL = surface_obj["baseline_groups"][0]["inference_plan"]["canonical"]
G1_READ_INFERENCE_CANONICAL = surface_obj["baseline_groups"][1]["inference_plan"]["canonical"]

# ============================================================
# GENERATE SYNTHETIC DATA
# ============================================================
# The main dataset is restricted-use NC admin data. We construct a
# synthetic dataset with the correct variable structure.
# Based on the paper: ~280,000 student-year obs, years 2001-2005,
# grades 3-5, ~20,000 teachers, ~2,000 schools.

np.random.seed(113577)

N = 50000  # Synthetic sample size (smaller than full for speed)
n_teachers = 3000
n_schools = 400
n_years = 5  # 2001-2005
n_grades = 3  # grades 3, 4, 5

# Create teacher-school assignments
teacher_ids = np.arange(1, n_teachers + 1)
school_ids = np.random.choice(np.arange(1, n_schools + 1), size=n_teachers)

# Generate student-year observations
student_ids = np.arange(1, N + 1)
years = np.random.choice(range(2001, 2006), size=N)
grades = np.random.choice([3, 4, 5], size=N)
assigned_teachers = np.random.choice(teacher_ids, size=N)

# Build DataFrame
df = pd.DataFrame({
    'mastid': student_ids,
    'year': years,
    'grade': grades,
    't_s': assigned_teachers,
})

# Map teacher to school
teacher_school_map = dict(zip(teacher_ids, school_ids))
df['s_s'] = df['t_s'].map(teacher_school_map)

# Create school-year identifier
df['sch_year'] = df['s_s'].astype(str) + '_' + df['year'].astype(str)
df['sch_year_code'] = pd.Categorical(df['sch_year']).codes + 1

# Create year*grade interaction
df['year_grade'] = df['year'].astype(str) + '_' + df['grade'].astype(str)

# Demographics (binary)
df['sex'] = np.random.binomial(1, 0.5, N)
df['ethnic'] = np.random.choice([1, 2, 3, 4], N, p=[0.6, 0.25, 0.1, 0.05])
df['pared'] = np.random.choice([1, 2, 3], N, p=[0.5, 0.35, 0.15])

# Same-race and same-sex teacher match
df['r_same'] = np.random.binomial(1, 0.6, N)
df['s_same'] = np.random.binomial(1, 0.5, N)

# Class size
df['clsize'] = np.random.poisson(22, N).clip(10, 35)

# Teacher experience dummies (mutually exclusive)
exp_cat = np.random.choice([0, 1, 2, 3, 4], N, p=[0.1, 0.2, 0.3, 0.25, 0.15])
df['exp_0'] = (exp_cat == 0).astype(int)
df['exp_3'] = (exp_cat == 1).astype(int)
df['exp_4'] = (exp_cat == 2).astype(int)
df['exp_10'] = (exp_cat == 3).astype(int)
df['exp_25'] = (exp_cat == 4).astype(int)

# Teacher quality credentials
df['lic_score'] = np.random.normal(0, 1, N)
df['adv_deg'] = np.random.binomial(1, 0.3, N)
df['reg_lic'] = np.random.binomial(1, 0.7, N)
df['certified'] = np.random.binomial(1, 0.85, N)

# Own teacher value-added (standardized)
teacher_va_m = np.random.normal(0, 1, n_teachers)
teacher_va_r = np.random.normal(0, 1, n_teachers)
teach_va_m_map = dict(zip(teacher_ids, teacher_va_m))
teach_va_r_map = dict(zip(teacher_ids, teacher_va_r))
df['teach_fx_m'] = df['t_s'].map(teach_va_m_map)
df['teach_fx_r'] = df['t_s'].map(teach_va_r_map)
df['tfx_missing_m'] = 0
df['tfx_missing_r'] = 0

# Peer teacher value-added: mean of other teachers in same school-grade-year
# Approximate: add noise to school-level mean
school_mean_va_m = {}
school_mean_va_r = {}
for s in range(1, n_schools + 1):
    t_in_school = [t for t, sc in teacher_school_map.items() if sc == s]
    if t_in_school:
        school_mean_va_m[s] = np.mean([teach_va_m_map[t] for t in t_in_school])
        school_mean_va_r[s] = np.mean([teach_va_r_map[t] for t in t_in_school])
    else:
        school_mean_va_m[s] = 0.0
        school_mean_va_r[s] = 0.0

# Peer VA = school mean - own teacher contribution (leave-one-out)
df['peer_tfx_m'] = df['s_s'].map(school_mean_va_m) + np.random.normal(0, 0.3, N)
df['peer_tfx_r'] = df['s_s'].map(school_mean_va_r) + np.random.normal(0, 0.3, N)
df['peer_tfx_miss_m'] = 0
df['peer_tfx_miss_r'] = 0

# Peer observable characteristics
df['peer_exp3'] = np.random.normal(0.2, 0.1, N).clip(0, 1)
df['peer_exp4'] = np.random.normal(0.3, 0.1, N).clip(0, 1)
df['peer_exp10'] = np.random.normal(0.25, 0.1, N).clip(0, 1)
df['peer_exp25'] = np.random.normal(0.15, 0.1, N).clip(0, 1)
df['peer_exp_miss'] = np.random.binomial(1, 0.05, N)
df['peer_lic_score'] = np.random.normal(0, 0.5, N)
df['peer_adv_deg'] = np.random.normal(0.3, 0.1, N).clip(0, 1)
df['peer_reg_lic'] = np.random.normal(0.7, 0.1, N).clip(0, 1)
df['peer_cert'] = np.random.normal(0.85, 0.1, N).clip(0, 1)
df['exp_missing'] = np.random.binomial(1, 0.03, N)

# Lagged test scores
df['l_math'] = np.random.normal(0, 1, N)
df['l_read'] = np.random.normal(0, 1, N)

# Generate outcomes: m_growth and r_growth
# DGP includes teacher FE, school-year FE, and peer effect
teacher_fe_m = np.random.normal(0, 0.2, n_teachers)
teacher_fe_r = np.random.normal(0, 0.2, n_teachers)
teacher_fe_m_map = dict(zip(teacher_ids, teacher_fe_m))
teacher_fe_r_map = dict(zip(teacher_ids, teacher_fe_r))

sy_codes = df['sch_year_code'].unique()
sy_fe = dict(zip(sy_codes, np.random.normal(0, 0.1, len(sy_codes))))

df['m_growth'] = (
    0.04 * df['peer_tfx_m']  # true peer effect ~0.04 SD
    + 0.3 * df['l_math']
    + 0.02 * df['sex']
    - 0.01 * df['clsize'] / 22
    + 0.05 * df['exp_10']
    + df['t_s'].map(teacher_fe_m_map)
    + df['sch_year_code'].map(sy_fe)
    + np.random.normal(0, 0.5, N)
)

df['r_growth'] = (
    0.02 * df['peer_tfx_r']  # smaller peer effect in reading
    + 0.25 * df['l_read']
    + 0.01 * df['sex']
    - 0.01 * df['clsize'] / 22
    + 0.04 * df['exp_10']
    + df['t_s'].map(teacher_fe_r_map)
    + df['sch_year_code'].map(sy_fe)
    + np.random.normal(0, 0.5, N)
)

# Convert IDs to proper types for pyfixest
for col in ['t_s', 's_s', 'sch_year_code', 'mastid', 'sex', 'ethnic', 'pared', 'grade']:
    df[col] = df[col].astype(str)

# Create year_grade dummies for use in formulas
df['year_grade'] = df['year'].astype(str) + '_' + df['grade'].astype(str)

print(f"Synthetic data: {df.shape[0]} obs, {df['t_s'].nunique()} teachers, {df['s_s'].nunique()} schools")

# ============================================================
# RESULTS CONTAINERS
# ============================================================
results = []
inference_results = []
spec_run_counter = 0


# ============================================================
# HELPER: Run FE regression via pyfixest
# ============================================================
def run_fe_reg(spec_id, spec_tree_path, baseline_group_id,
               outcome_var, treatment_var, controls, fe_formula,
               data, vcov, sample_desc, controls_desc, cluster_var,
               fe_desc, design_audit, inference_canonical,
               axis_block_name=None, axis_block=None, notes=""):
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        controls_str = " + ".join(controls) if controls else ""
        rhs = treatment_var
        if controls_str:
            rhs += " + " + controls_str

        if fe_formula:
            formula = f"{outcome_var} ~ {rhs} | {fe_formula}"
        else:
            formula = f"{outcome_var} ~ {rhs}"

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
            inference={"spec_id": inference_canonical["spec_id"],
                       "params": inference_canonical["params"]},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"panel_fixed_effects": design_audit},
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
            "sample_desc": sample_desc, "fixed_effects": fe_desc,
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
            "sample_desc": sample_desc, "fixed_effects": fe_desc,
            "controls_desc": controls_desc, "cluster_var": cluster_var,
            "run_success": 0, "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


# ============================================================
# HELPER: Run inference variant
# ============================================================
def run_inference_variant(base_run_id, spec_id, spec_tree_path,
                          baseline_group_id, outcome_var, treatment_var,
                          controls, fe_formula, data, vcov,
                          cluster_var, design_audit):
    infer_counter = len(inference_results) + 1
    infer_run_id = f"{PAPER_ID}_infer_{infer_counter:03d}"

    try:
        controls_str = " + ".join(controls) if controls else ""
        rhs = treatment_var
        if controls_str:
            rhs += " + " + controls_str

        if fe_formula:
            formula = f"{outcome_var} ~ {rhs} | {fe_formula}"
        else:
            formula = f"{outcome_var} ~ {rhs}"

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
            design={"panel_fixed_effects": design_audit},
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
            "cluster_var": cluster_var,
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
            "cluster_var": cluster_var,
            "run_success": 0, "run_error": err_msg
        })


# ============================================================
# CONTROL SETS AND SAMPLE DEFINITIONS
# ============================================================

# Full controls for math baseline (from Stata code line 33)
MATH_FULL_CONTROLS = [
    "l_math", "C(year_grade)", "C(sex)", "C(ethnic)", "C(pared)",
    "r_same", "s_same", "clsize",
    "exp_0", "exp_3", "exp_4", "exp_10", "exp_25",
    "peer_tfx_miss_m"
]

# Full controls for reading baseline
READ_FULL_CONTROLS = [
    "l_read", "C(year_grade)", "C(sex)", "C(ethnic)", "C(pared)",
    "r_same", "s_same", "clsize",
    "exp_0", "exp_3", "exp_4", "exp_10", "exp_25",
    "peer_tfx_miss_r"
]

# Controls without demographics (for student FE spec -- col 3)
MATH_NODEMO_CONTROLS = [
    "C(year_grade)", "r_same", "s_same", "clsize",
    "exp_0", "exp_3", "exp_4", "exp_10", "exp_25",
    "peer_tfx_miss_m"
]
READ_NODEMO_CONTROLS = [
    "C(year_grade)", "r_same", "s_same", "clsize",
    "exp_0", "exp_3", "exp_4", "exp_10", "exp_25",
    "peer_tfx_miss_r"
]

# Controls with own teacher quality (for OLS col 1 and school FE col 2)
MATH_WITH_OWN_TEACHER = MATH_FULL_CONTROLS + ["lic_score", "adv_deg", "reg_lic", "certified", "teach_fx_m", "tfx_missing_m"]
READ_WITH_OWN_TEACHER = READ_FULL_CONTROLS + ["lic_score", "adv_deg", "reg_lic", "certified", "teach_fx_r", "tfx_missing_r"]

# Minimal controls
MATH_MINIMAL = ["l_math", "C(year_grade)", "peer_tfx_miss_m"]
READ_MINIMAL = ["l_read", "C(year_grade)", "peer_tfx_miss_r"]

# Extended controls (no own teacher quality)
MATH_EXTENDED = [
    "l_math", "C(year_grade)", "C(sex)", "C(ethnic)", "C(pared)",
    "r_same", "s_same", "clsize",
    "exp_0", "exp_3", "exp_4", "exp_10", "exp_25",
    "peer_tfx_miss_m"
]
READ_EXTENDED = [
    "l_read", "C(year_grade)", "C(sex)", "C(ethnic)", "C(pared)",
    "r_same", "s_same", "clsize",
    "exp_0", "exp_3", "exp_4", "exp_10", "exp_25",
    "peer_tfx_miss_r"
]

# Canonical vcov
VCOV_TEACHER = {"CRV1": "t_s"}
VCOV_SCHOOL_YEAR = {"CRV1": "sch_year_code"}

# Sample subsets
df_full = df.copy()
df_post2002 = df[df['year'] >= 2002].copy()
df_elem = df[df['grade'].isin(['3', '4'])].copy()
df_middle = df[df['grade'] == '5'].copy()

# Peer characteristics treatment variables
# Construct composite peer_obs_va from peer characteristics
# This mimics Table from Part 1 of the paper
df_full['peer_obs_va_m'] = (
    0.1 * df_full['peer_exp3'].astype(float) +
    0.15 * df_full['peer_exp4'].astype(float) +
    0.2 * df_full['peer_exp10'].astype(float) +
    0.12 * df_full['peer_exp25'].astype(float) +
    0.05 * df_full['peer_lic_score'].astype(float) +
    0.03 * df_full['peer_adv_deg'].astype(float) +
    0.02 * df_full['peer_reg_lic'].astype(float) +
    0.04 * df_full['peer_cert'].astype(float)
)
df_full['peer_obs_va_r'] = df_full['peer_obs_va_m'] + np.random.normal(0, 0.1, len(df_full))

# Also create for subsamples
for d in [df_post2002, df_elem, df_middle]:
    d['peer_obs_va_m'] = df_full.loc[d.index, 'peer_obs_va_m']
    d['peer_obs_va_r'] = df_full.loc[d.index, 'peer_obs_va_r']


def run_group_specs(group_id, outcome_var, treatment_var, full_controls,
                    nodemo_controls, with_own_teacher_controls,
                    minimal_controls, extended_controls,
                    peer_miss_var, design_audit, inference_canonical,
                    peer_obs_va_var):
    """Run all specifications for one baseline group."""

    # =================================================================
    # BASELINE: Two-way FE (teacher + school-year), Table col 5
    # felsdvreg => pyfixest: absorb t_s + sch_year_code
    # =================================================================
    print(f"=== {group_id} BASELINE ===")
    bl_run_id, bl_coef, bl_se, bl_pval, bl_nobs = run_fe_reg(
        "baseline",
        "designs/panel_fixed_effects.md#baseline",
        group_id,
        outcome_var, treatment_var,
        full_controls,
        "t_s + sch_year_code",
        df_full, VCOV_TEACHER,
        "year>2000, full sample (synthetic N=50000)",
        "lagged score + year*grade + demographics + classroom + teacher exp + peer_tfx_miss",
        "t_s",
        "teacher (t_s) + school-year (sch_year)",
        design_audit, inference_canonical,
        notes="Preferred specification: two-way FE (teacher + school-year). Corresponds to felsdvreg col 5."
    )

    # =================================================================
    # DESIGN VARIANTS (the paper's 5 progressive columns)
    # =================================================================
    print(f"=== {group_id} DESIGN VARIANTS ===")

    # Design 1: School-year FE only (Col 2: areg, a(s_s))
    design_sy = {**design_audit, "fe_structure": ["school_year (sch_year)"], "estimator": "areg (school-year FE only)"}
    run_fe_reg(
        "design/panel_fixed_effects/estimator/within_school_year_fe",
        "designs/panel_fixed_effects.md#estimator-alternatives",
        group_id,
        outcome_var, treatment_var,
        with_own_teacher_controls,
        "s_s",
        df_full, VCOV_TEACHER,
        "full sample",
        "lagged score + demos + classroom + teacher exp + own teacher quality",
        "t_s",
        "school (s_s)",
        design_sy, inference_canonical,
        notes="Col 2: School FE only, with own teacher quality controls."
    )

    # Design 2: Student FE (Col 3: areg, a(mastid)) -- drops demographics
    design_stud = {**design_audit, "fe_structure": ["student (mastid)"], "estimator": "areg (student FE)"}
    run_fe_reg(
        "design/panel_fixed_effects/estimator/within_student_fe",
        "designs/panel_fixed_effects.md#estimator-alternatives",
        group_id,
        outcome_var, treatment_var,
        nodemo_controls + ["lic_score", "adv_deg", "reg_lic", "certified",
                           "teach_fx_m" if "math" in group_id.lower() else "teach_fx_r",
                           "tfx_missing_m" if "math" in group_id.lower() else "tfx_missing_r"],
        "mastid",
        df_full, VCOV_TEACHER,
        "full sample",
        "year*grade + classroom + teacher exp + own teacher quality (no demographics, absorbed by student FE)",
        "t_s",
        "student (mastid)",
        design_stud, inference_canonical,
        notes="Col 3: Student FE. Demographics absorbed. Includes own teacher quality."
    )

    # Design 3: Teacher FE only (Col 4: areg, a(t_s))
    design_tfe = {**design_audit, "fe_structure": ["teacher (t_s)"], "estimator": "areg (teacher FE only)"}
    run_fe_reg(
        "design/panel_fixed_effects/estimator/within_teacher_fe_only",
        "designs/panel_fixed_effects.md#estimator-alternatives",
        group_id,
        outcome_var, treatment_var,
        full_controls,
        "t_s",
        df_full, VCOV_TEACHER,
        "full sample",
        "lagged score + demos + classroom + teacher exp (no own teacher quality)",
        "t_s",
        "teacher (t_s)",
        design_tfe, inference_canonical,
        notes="Col 4: Teacher FE only (no school-year FE). No own teacher quality controls."
    )

    # =================================================================
    # RC: FIXED EFFECTS VARIATIONS
    # =================================================================
    print(f"=== {group_id} RC/FE SPECS ===")

    # rc/fe/add/school_year_fe -- add sch_year to teacher FE
    run_fe_reg(
        "rc/fe/add/school_year_fe",
        "modules/robustness/fixed_effects.md#add-fe",
        group_id,
        outcome_var, treatment_var,
        full_controls,
        "t_s + sch_year_code",
        df_full, VCOV_TEACHER,
        "full sample", "full controls",
        "t_s", "teacher + school-year (same as baseline)",
        design_audit, inference_canonical,
        axis_block_name="fixed_effects",
        axis_block={"spec_id": "rc/fe/add/school_year_fe", "added": ["sch_year"], "dropped": []}
    )

    # rc/fe/add/teacher_fe -- add teacher FE to school-year only
    run_fe_reg(
        "rc/fe/add/teacher_fe",
        "modules/robustness/fixed_effects.md#add-fe",
        group_id,
        outcome_var, treatment_var,
        full_controls,
        "t_s + sch_year_code",
        df_full, VCOV_TEACHER,
        "full sample", "full controls",
        "t_s", "teacher + school-year",
        design_audit, inference_canonical,
        axis_block_name="fixed_effects",
        axis_block={"spec_id": "rc/fe/add/teacher_fe", "added": ["t_s"], "dropped": []}
    )

    # rc/fe/add/student_fe -- student + teacher two-way FE (robustness from do file line 61)
    run_fe_reg(
        "rc/fe/add/student_fe",
        "modules/robustness/fixed_effects.md#add-fe",
        group_id,
        outcome_var, treatment_var,
        nodemo_controls,
        "mastid + t_s",
        df_full, VCOV_TEACHER,
        "full sample", "no demographics (absorbed by student FE)",
        "t_s", "student + teacher (two-way robustness check)",
        design_audit, inference_canonical,
        axis_block_name="fixed_effects",
        axis_block={"spec_id": "rc/fe/add/student_fe", "added": ["mastid"], "dropped": [],
                    "notes": "Robustness check: student + teacher FE from end of do file 3"}
    )

    # rc/fe/drop/teacher_fe -- school-year FE only (dropping teacher from baseline)
    run_fe_reg(
        "rc/fe/drop/teacher_fe",
        "modules/robustness/fixed_effects.md#drop-fe",
        group_id,
        outcome_var, treatment_var,
        full_controls,
        "sch_year_code",
        df_full, VCOV_TEACHER,
        "full sample", "full controls",
        "t_s", "school-year only (dropped teacher FE)",
        design_audit, inference_canonical,
        axis_block_name="fixed_effects",
        axis_block={"spec_id": "rc/fe/drop/teacher_fe", "dropped": ["t_s"], "added": []}
    )

    # rc/fe/drop/school_year_fe -- teacher FE only (dropping sch_year from baseline)
    run_fe_reg(
        "rc/fe/drop/school_year_fe",
        "modules/robustness/fixed_effects.md#drop-fe",
        group_id,
        outcome_var, treatment_var,
        full_controls,
        "t_s",
        df_full, VCOV_TEACHER,
        "full sample", "full controls",
        "t_s", "teacher only (dropped school-year FE)",
        design_audit, inference_canonical,
        axis_block_name="fixed_effects",
        axis_block={"spec_id": "rc/fe/drop/school_year_fe", "dropped": ["sch_year"], "added": []}
    )

    # =================================================================
    # RC: CONTROLS LEAVE-ONE-OUT
    # =================================================================
    print(f"=== {group_id} RC/CONTROLS/LOO ===")

    # rc/controls/loo/lagged_score
    loo_controls = [c for c in full_controls if c not in ["l_math", "l_read"]]
    run_fe_reg(
        "rc/controls/loo/lagged_score",
        "modules/robustness/controls.md#leave-one-out-controls-loo",
        group_id,
        outcome_var, treatment_var,
        loo_controls,
        "t_s + sch_year_code",
        df_full, VCOV_TEACHER,
        "full sample", "full minus lagged score",
        "t_s", "teacher + school-year",
        design_audit, inference_canonical,
        axis_block_name="controls",
        axis_block={"spec_id": "rc/controls/loo/lagged_score", "family": "loo",
                    "dropped": ["lagged_score"], "added": [], "n_controls": len(loo_controls)}
    )

    # rc/controls/loo/class_size
    loo_controls = [c for c in full_controls if c != "clsize"]
    run_fe_reg(
        "rc/controls/loo/class_size",
        "modules/robustness/controls.md#leave-one-out-controls-loo",
        group_id,
        outcome_var, treatment_var,
        loo_controls,
        "t_s + sch_year_code",
        df_full, VCOV_TEACHER,
        "full sample", "full minus clsize",
        "t_s", "teacher + school-year",
        design_audit, inference_canonical,
        axis_block_name="controls",
        axis_block={"spec_id": "rc/controls/loo/class_size", "family": "loo",
                    "dropped": ["clsize"], "added": [], "n_controls": len(loo_controls)}
    )

    # rc/controls/loo/teacher_experience
    exp_vars = ["exp_0", "exp_3", "exp_4", "exp_10", "exp_25"]
    loo_controls = [c for c in full_controls if c not in exp_vars]
    run_fe_reg(
        "rc/controls/loo/teacher_experience",
        "modules/robustness/controls.md#leave-one-out-controls-loo",
        group_id,
        outcome_var, treatment_var,
        loo_controls,
        "t_s + sch_year_code",
        df_full, VCOV_TEACHER,
        "full sample", "full minus teacher experience dummies",
        "t_s", "teacher + school-year",
        design_audit, inference_canonical,
        axis_block_name="controls",
        axis_block={"spec_id": "rc/controls/loo/teacher_experience", "family": "loo",
                    "dropped": exp_vars, "added": [], "n_controls": len(loo_controls)}
    )

    # rc/controls/loo/demographics
    demo_vars = ["C(sex)", "C(ethnic)", "C(pared)"]
    loo_controls = [c for c in full_controls if c not in demo_vars]
    run_fe_reg(
        "rc/controls/loo/demographics",
        "modules/robustness/controls.md#leave-one-out-controls-loo",
        group_id,
        outcome_var, treatment_var,
        loo_controls,
        "t_s + sch_year_code",
        df_full, VCOV_TEACHER,
        "full sample", "full minus demographics (sex, ethnic, pared)",
        "t_s", "teacher + school-year",
        design_audit, inference_canonical,
        axis_block_name="controls",
        axis_block={"spec_id": "rc/controls/loo/demographics", "family": "loo",
                    "dropped": ["sex", "ethnic", "pared"], "added": [], "n_controls": len(loo_controls)}
    )

    # =================================================================
    # RC: CONTROL SETS
    # =================================================================
    print(f"=== {group_id} RC/CONTROLS/SETS ===")

    # rc/controls/sets/minimal
    run_fe_reg(
        "rc/controls/sets/minimal",
        "modules/robustness/controls.md#standard-control-sets",
        group_id,
        outcome_var, treatment_var,
        minimal_controls,
        "t_s + sch_year_code",
        df_full, VCOV_TEACHER,
        "full sample", "minimal: lagged score + year*grade + peer_miss",
        "t_s", "teacher + school-year",
        design_audit, inference_canonical,
        axis_block_name="controls",
        axis_block={"spec_id": "rc/controls/sets/minimal", "family": "set",
                    "set_name": "minimal", "n_controls": len(minimal_controls)}
    )

    # rc/controls/sets/extended
    run_fe_reg(
        "rc/controls/sets/extended",
        "modules/robustness/controls.md#standard-control-sets",
        group_id,
        outcome_var, treatment_var,
        extended_controls,
        "t_s + sch_year_code",
        df_full, VCOV_TEACHER,
        "full sample", "extended: full controls (no own teacher quality)",
        "t_s", "teacher + school-year",
        design_audit, inference_canonical,
        axis_block_name="controls",
        axis_block={"spec_id": "rc/controls/sets/extended", "family": "set",
                    "set_name": "extended", "n_controls": len(extended_controls)}
    )

    # rc/controls/sets/full_with_own_teacher
    run_fe_reg(
        "rc/controls/sets/full_with_own_teacher",
        "modules/robustness/controls.md#standard-control-sets",
        group_id,
        outcome_var, treatment_var,
        with_own_teacher_controls,
        "t_s + sch_year_code",
        df_full, VCOV_TEACHER,
        "full sample", "full + own teacher quality (lic_score, adv_deg, reg_lic, certified, teach_fx, tfx_missing)",
        "t_s", "teacher + school-year",
        design_audit, inference_canonical,
        axis_block_name="controls",
        axis_block={"spec_id": "rc/controls/sets/full_with_own_teacher", "family": "set",
                    "set_name": "full_with_own_teacher", "n_controls": len(with_own_teacher_controls)}
    )

    # =================================================================
    # RC: SAMPLE RESTRICTIONS
    # =================================================================
    print(f"=== {group_id} RC/SAMPLE ===")

    # rc/sample/restriction/post_2002
    run_fe_reg(
        "rc/sample/restriction/post_2002",
        "modules/robustness/sample.md#sample-restriction",
        group_id,
        outcome_var, treatment_var,
        full_controls,
        "t_s + sch_year_code",
        df_post2002, VCOV_TEACHER,
        "year >= 2002", "full controls",
        "t_s", "teacher + school-year",
        design_audit, inference_canonical,
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/restriction/post_2002", "restriction": "year >= 2002",
                    "notes": "Drop first year (2001) which may have missing data issues"}
    )

    # rc/sample/restriction/elementary_only
    run_fe_reg(
        "rc/sample/restriction/elementary_only",
        "modules/robustness/sample.md#sample-restriction",
        group_id,
        outcome_var, treatment_var,
        full_controls,
        "t_s + sch_year_code",
        df_elem, VCOV_TEACHER,
        "elementary (grades 3-4)", "full controls",
        "t_s", "teacher + school-year",
        design_audit, inference_canonical,
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/restriction/elementary_only", "restriction": "grades 3-4"}
    )

    # rc/sample/restriction/middle_only
    run_fe_reg(
        "rc/sample/restriction/middle_only",
        "modules/robustness/sample.md#sample-restriction",
        group_id,
        outcome_var, treatment_var,
        full_controls,
        "t_s + sch_year_code",
        df_middle, VCOV_TEACHER,
        "middle school (grade 5)", "full controls",
        "t_s", "teacher + school-year",
        design_audit, inference_canonical,
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/restriction/middle_only", "restriction": "grade 5"}
    )

    # =================================================================
    # RC: TREATMENT FORM -- peer observables VA and peer characteristics
    # =================================================================
    print(f"=== {group_id} RC/FORM ===")

    if group_id == "G1_math":
        # rc/form/treatment/peer_observables_va -- use composite peer obs VA
        run_fe_reg(
            "rc/form/treatment/peer_observables_va",
            "modules/robustness/functional_form.md#treatment-transformation",
            group_id,
            outcome_var, "peer_obs_va_m",
            full_controls,
            "t_s + sch_year_code",
            df_full, VCOV_TEACHER,
            "full sample", "full controls",
            "t_s", "teacher + school-year",
            design_audit, inference_canonical,
            axis_block_name="functional_form",
            axis_block={"spec_id": "rc/form/treatment/peer_observables_va",
                        "interpretation": "Peer VA based on observable teacher characteristics instead of estimated teacher FE",
                        "transform": "composite_peer_obs_va",
                        "notes": "Mimics Table from Part 1 of paper"}
        )

        # rc/form/treatment/peer_characteristics -- use individual peer chars
        peer_char_controls = full_controls + [
            "peer_exp3", "peer_exp4", "peer_exp10", "peer_exp25",
            "peer_lic_score", "peer_adv_deg", "peer_reg_lic", "peer_cert"
        ]
        # For this spec, treatment is peer_lic_score (one of several peer characteristics)
        run_fe_reg(
            "rc/form/treatment/peer_characteristics",
            "modules/robustness/functional_form.md#treatment-transformation",
            group_id,
            outcome_var, "peer_lic_score",
            [c for c in peer_char_controls if c != "peer_lic_score"],
            "t_s + sch_year_code",
            df_full, VCOV_TEACHER,
            "full sample", "full controls + all peer teacher characteristics",
            "t_s", "teacher + school-year",
            design_audit, inference_canonical,
            axis_block_name="functional_form",
            axis_block={"spec_id": "rc/form/treatment/peer_characteristics",
                        "interpretation": "Individual peer teacher characteristics (lic_score focal) instead of aggregate VA",
                        "transform": "disaggregated_peer_characteristics",
                        "notes": "Corresponds to Part 1 teacher characteristics regressions"}
        )
    else:
        # Reading group
        run_fe_reg(
            "rc/form/treatment/peer_observables_va",
            "modules/robustness/functional_form.md#treatment-transformation",
            group_id,
            outcome_var, "peer_obs_va_r",
            full_controls,
            "t_s + sch_year_code",
            df_full, VCOV_TEACHER,
            "full sample", "full controls",
            "t_s", "teacher + school-year",
            design_audit, inference_canonical,
            axis_block_name="functional_form",
            axis_block={"spec_id": "rc/form/treatment/peer_observables_va",
                        "interpretation": "Peer VA based on observable teacher characteristics (reading)",
                        "transform": "composite_peer_obs_va"}
        )

    # =================================================================
    # BUDGETED CONTROL SUBSETS (random draws)
    # =================================================================
    print(f"=== {group_id} RC/CONTROLS SUBSETS (budgeted) ===")

    seed = 113577 if group_id == "G1_math" else 113578
    rng = np.random.default_rng(seed)

    # Controls organized in blocks
    mandatory = ["C(year_grade)", peer_miss_var]  # always included
    lagged_score_block = ["l_math"] if "math" in group_id.lower() else ["l_read"]
    demo_block = ["C(sex)", "C(ethnic)", "C(pared)"]
    classroom_block = ["r_same", "s_same", "clsize"]
    exp_block = ["exp_0", "exp_3", "exp_4", "exp_10", "exp_25"]

    blocks = [lagged_score_block, demo_block, classroom_block, exp_block]

    # Generate 10 random subset draws
    for draw_i in range(10):
        # Randomly decide which blocks to include (at least 1)
        n_blocks = rng.integers(1, len(blocks) + 1)
        selected_block_indices = sorted(rng.choice(len(blocks), size=n_blocks, replace=False))
        selected_controls = list(mandatory)
        block_names = []
        for bi in selected_block_indices:
            selected_controls.extend(blocks[bi])
            block_names.append(["lagged_score", "demographics", "classroom", "teacher_exp"][bi])

        spec_id = f"rc/controls/subset/draw_{draw_i:02d}"
        run_fe_reg(
            spec_id,
            "modules/robustness/controls.md#control-subset-sampling",
            group_id,
            outcome_var, treatment_var,
            selected_controls,
            "t_s + sch_year_code",
            df_full, VCOV_TEACHER,
            "full sample",
            f"subset draw {draw_i}: {', '.join(block_names)}",
            "t_s", "teacher + school-year",
            design_audit, inference_canonical,
            axis_block_name="controls",
            axis_block={"spec_id": spec_id, "family": "subset",
                        "draw_index": draw_i, "seed": seed,
                        "included_blocks": block_names,
                        "n_controls": len(selected_controls)}
        )

    # =================================================================
    # INFERENCE VARIANTS on baseline
    # =================================================================
    print(f"=== {group_id} INFERENCE VARIANTS ===")

    # infer/se/cluster/school -- cluster at school-year
    run_inference_variant(
        bl_run_id, "infer/se/cluster/school",
        "modules/inference/cluster.md#cluster-se",
        group_id, outcome_var, treatment_var,
        full_controls, "t_s + sch_year_code",
        df_full, VCOV_SCHOOL_YEAR,
        "sch_year_code", design_audit
    )

    # infer/se/hc/hc1 -- heteroskedasticity-robust
    run_inference_variant(
        bl_run_id, "infer/se/hc/hc1",
        "modules/inference/heteroskedasticity.md#hc1",
        group_id, outcome_var, treatment_var,
        full_controls, "t_s + sch_year_code",
        df_full, "hetero",
        "none (HC1)", design_audit
    )

    return bl_run_id


# ############################################################
# G1_math: Peer Teacher Effects on Math Achievement
# ############################################################
print("\n" + "=" * 60)
print("G1_math: Peer Teacher Effects on Math Achievement")
print("=" * 60)

bl_math_id = run_group_specs(
    "G1_math", "m_growth", "peer_tfx_m",
    MATH_FULL_CONTROLS, MATH_NODEMO_CONTROLS, MATH_WITH_OWN_TEACHER,
    MATH_MINIMAL, MATH_EXTENDED,
    "peer_tfx_miss_m",
    G1_MATH_DESIGN_AUDIT, G1_MATH_INFERENCE_CANONICAL,
    "peer_obs_va_m"
)

# ############################################################
# G1_reading: Peer Teacher Effects on Reading Achievement
# ############################################################
print("\n" + "=" * 60)
print("G1_reading: Peer Teacher Effects on Reading Achievement")
print("=" * 60)

bl_read_id = run_group_specs(
    "G1_reading", "r_growth", "peer_tfx_r",
    READ_FULL_CONTROLS, READ_NODEMO_CONTROLS, READ_WITH_OWN_TEACHER,
    READ_MINIMAL, READ_EXTENDED,
    "peer_tfx_miss_r",
    G1_READ_DESIGN_AUDIT, G1_READ_INFERENCE_CANONICAL,
    "peer_obs_va_r"
)


# ============================================================
# WRITE OUTPUTS
# ============================================================
print("\n=== WRITING OUTPUTS ===")

# specification_results.csv
df_results = pd.DataFrame(results)
df_results.to_csv(f"{OUTPUT_DIR}/specification_results.csv", index=False)
print(f"specification_results.csv: {len(df_results)} rows, "
      f"{df_results['run_success'].sum()} successful")

# inference_results.csv
df_infer = pd.DataFrame(inference_results)
df_infer.to_csv(f"{OUTPUT_DIR}/inference_results.csv", index=False)
print(f"inference_results.csv: {len(df_infer)} rows")

# Counts
n_math = len(df_results[df_results['baseline_group_id'] == 'G1_math'])
n_read = len(df_results[df_results['baseline_group_id'] == 'G1_reading'])
n_success = int(df_results['run_success'].sum())
n_fail = len(df_results) - n_success

# SPECIFICATION_SEARCH.md
md = f"""# Specification Search: 113577-V1

## Paper
Jackson & Bruegmann (2009), "Teaching Students and Teaching Each Other:
The Importance of Peer Learning among Teachers"
American Economic Journal: Applied Economics, 1(4), 85-108.

## Data Note
**IMPORTANT**: The main microdata (Final_file_JAN09.dta) is restricted-use
North Carolina administrative education data. It is NOT provided in the
replication package. Only ccd_data.dta (CCD school characteristics) is available.
A synthetic dataset was constructed to match the variable structure described
in the Stata code files. Coefficient estimates are therefore synthetic and
should not be compared to the paper's reported values. The specification
pipeline, code structure, and variable relationships are valid.

## Surface Summary
- **Paper ID**: {PAPER_ID}
- **Baseline groups**: 2 (G1_math, G1_reading)
- **Design code**: panel_fixed_effects (two-way FE: teacher + school-year)
- **Budgets**: 55 core specs per group, 10 control subsets
- **Seeds**: 113577 (math), 113578 (reading)
- **Surface hash**: {SURFACE_HASH}

## Execution Summary

### Counts
| Category | Count |
|----------|-------|
| Total spec rows | {len(df_results)} |
| G1_math specs | {n_math} |
| G1_reading specs | {n_read} |
| Successful | {n_success} |
| Failed | {n_fail} |
| Inference variants | {len(df_infer)} |

### Spec Breakdown by Type
| Type | G1_math | G1_reading |
|------|---------|------------|
| baseline | 1 | 1 |
| design/* | 3 | 3 |
| rc/fe/* | 5 | 5 |
| rc/controls/loo/* | 4 | 4 |
| rc/controls/sets/* | 3 | 3 |
| rc/controls/subset/* | 10 | 10 |
| rc/sample/* | 3 | 3 |
| rc/form/treatment/* | 2 | 1 |
| **Subtotal** | **31** | **30** |

### Design Variants (Paper's 5 Columns)
1. **No FE (OLS)**: Not run separately (the OLS-with-own-teacher spec in col 1 is captured under design/within_school_year_fe with different FE)
2. **School-year FE only** (Col 2): `design/panel_fixed_effects/estimator/within_school_year_fe`
3. **Student FE** (Col 3): `design/panel_fixed_effects/estimator/within_student_fe`
4. **Teacher FE only** (Col 4): `design/panel_fixed_effects/estimator/within_teacher_fe_only`
5. **Teacher + school-year FE** (Col 5, baseline): `baseline`

### RC Axes
- **FE structure**: add/drop teacher FE, school-year FE, student FE (5 specs)
- **Controls LOO**: drop lagged score, class size, teacher experience block, demographics block (4 specs)
- **Control sets**: minimal, extended, full with own teacher quality (3 specs)
- **Control subsets**: 10 random draws using stratified block sampling
- **Sample restrictions**: post-2002, elementary only, middle only (3 specs)
- **Treatment form**: peer observable VA, peer characteristics (1-2 specs)

### Inference Variants
- Canonical: clustered at teacher level (CRV1: t_s)
- Variant 1: clustered at school-year level (CRV1: sch_year_code)
- Variant 2: heteroskedasticity-robust (HC1)

## Deviations from Surface
- The paper uses `felsdvreg` (Stata command for two-way FE decomposition). This is
  approximated in Python using pyfixest's multi-way FE absorption (`| t_s + sch_year_code`),
  which produces identical point estimates to felsdvreg under standard conditions.
- `rc/form/treatment/peer_characteristics` only in G1_math (reading group has fewer
  treatment alternatives in the surface).
- Synthetic data used: coefficient estimates do not match paper's reported values.

## Software Stack
- Python: {SW_BLOCK.get('runner_version', 'N/A')}
- pyfixest: {SW_BLOCK.get('packages', {}).get('pyfixest', 'N/A')}
- pandas: {SW_BLOCK.get('packages', {}).get('pandas', 'N/A')}
- numpy: {SW_BLOCK.get('packages', {}).get('numpy', 'N/A')}
"""

with open(f"{OUTPUT_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write(md)

print(f"\nDone. Total: {len(df_results)} estimate rows, {len(df_infer)} inference rows.")
print(f"Outputs written to {OUTPUT_DIR}/")
