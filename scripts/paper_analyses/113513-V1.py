"""
Specification Search Script for 113513-V1
"Trends in Economics Undergraduate Majors by Demographics"
Descriptive study using IPEDS aggregate data (2001-2014).

Surface-driven execution:
  - G1: econshare ~ level_d{j} + year, robust (bivariate OLS, one per discipline)
  - 11 comparison disciplines x 3 demographic groups (all, females, nonwhites)
  - RC axes: second major outcome, time splits, quadratic trend, no year control
  - Inference: HC1 (canonical), Newey-West HAC, Classical OLS

Outputs:
  - specification_results.csv (baseline, rc/* rows)
  - inference_results.csv (infer/* rows)
  - SPECIFICATION_SEARCH.md
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
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

PAPER_ID = "113513-V1"
DATA_DIR = "data/downloads/extracted/113513-V1"
OUTPUT_DIR = DATA_DIR

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

# Design audit block from surface
G1_DESIGN_AUDIT = surface_obj["baseline_groups"][0]["design_audit"]
G1_INFERENCE_CANONICAL = surface_obj["baseline_groups"][0]["inference_plan"]["canonical"]

# Discipline mapping
DISCIPLINE_MAP = {
    1: "Economics",
    2: "Business & Management",
    3: "Political Science",
    4: "Psychology",
    5: "Other Social Sciences",
    6: "Math, Eng, CSci, & Tech",
    7: "Physical & Geosciences",
    8: "Life & Med. Sciences",
    9: "Arts & Architecture",
    10: "Education",
    11: "Humanities",
    12: "Other",
}

DISCIPLINE_LABELS = {
    2: "busmgmt", 3: "polisci", 4: "psych", 5: "otherss",
    6: "mathengcs", 7: "physci", 8: "lifesci", 9: "artsarch",
    10: "education", 11: "humanities", 12: "other",
}

# ============================================================
# DATA LOADING — replicate Stata do file logic
# ============================================================

def assign_discipline(detail):
    """Map IPEDS detailed discipline to numbered groups matching the Stata code."""
    mapping = {
        "Economics": 1,
        "Business and Management": 2,
        "Law": 3,
        "Political Science and Public Administration": 3,
        "Psychology": 4,
        "Anthropology": 5, "Area and Ethnic Studies": 5,
        "History of Science": 5, "Linguistics": 5,
        "Other Social Sciences": 5, "Social Service Professions": 5,
        "Sociology": 5,
        "Mathematics and Statistics": 6, "Aerospace Engineering": 6,
        "Chemical Engineering": 6, "Civil Engineering": 6,
        "Computer Science": 6, "Electrical Engineering": 6,
        "Engineering Technologies": 6, "Health Technologies": 6,
        "Industrial Engineering": 6, "Materials Engineering": 6,
        "Mechanical Engineering": 6, "Other Engineering": 6,
        "Other Science and Engineering Technologies": 6,
        "Science Technologies": 6,
        "Astronomy": 7, "Atmospheric Sciences": 7,
        "Chemistry": 7, "Earth Sciences": 7,
        "Interdisciplinary or Other Sciences": 7,
        "Oceanography": 7, "Other Physical Sciences": 7,
        "Physics": 7,
        "Medical Sciences": 8, "Agricultural Sciences": 8,
        "Biological Sciences": 8, "Other Life Sciences": 8,
        "Architecture and Environmental Design": 9,
        "Arts and Music": 9,
        "Mathematics Education": 10, "Non-Science Education": 10,
        "Other Science/Technical Education": 10,
        "Science Education": 10, "Social Science Education": 10,
        "English and Literature": 11, "Foreign Languages": 11,
        "History": 11, "Other Humanities": 11,
        "Religion and Theology": 11,
        "Communication and Librarianship": 12,
        "Other Non-sciences or Unknown Disciplines": 12,
        "Vocational Studies and Home Economics": 12,
    }
    return mapping.get(detail, 0)


def build_sex_data():
    """Replicate majortrendssex_datain.do to build collapsed year-level data."""
    df = pd.read_csv(f"{DATA_DIR}/IPEDS_degrees_by_sex.csv")
    df.columns = [c.strip() for c in df.columns]
    # Standardize column names
    df.rename(columns={
        'Year': 'year',
        'Gender': 'gender',
        'Academic Discipline, Detailed (standardized)': 'acad_disc',
        'Level of Degree or Other Award': 'degree_level',
    }, inplace=True)

    df['discipline'] = df['acad_disc'].apply(assign_discipline)
    df['male'] = (df['gender'] == 'Male').astype(int)

    # Total BA and 2D by year (all genders combined)
    totals_g = df.groupby('year').agg(
        total_ba_g=('degreesawarded', 'sum'),
        total_2d_g=('degreesawarded2d', 'sum')
    ).reset_index()

    # Total BA and 2D by year and gender
    totals_sex = df.groupby(['year', 'male']).agg(
        total_ba_sex=('degreesawarded', 'sum'),
        total_2d_sex=('degreesawarded2d', 'sum')
    ).reset_index()
    totals_m = totals_sex[totals_sex['male'] == 1].rename(columns={
        'total_ba_sex': 'total_ba_m', 'total_2d_sex': 'total_2d_m'
    })[['year', 'total_ba_m', 'total_2d_m']]
    totals_f = totals_sex[totals_sex['male'] == 0].rename(columns={
        'total_ba_sex': 'total_ba_f', 'total_2d_sex': 'total_2d_f'
    })[['year', 'total_ba_f', 'total_2d_f']]

    # Discipline totals by year (all genders)
    disc_g = df.groupby(['year', 'discipline']).agg(
        discipline_ba_g=('degreesawarded', 'sum'),
        discipline_2d_g=('degreesawarded2d', 'sum')
    ).reset_index()

    # Discipline totals by year and gender
    disc_sex = df.groupby(['year', 'discipline', 'male']).agg(
        disc_ba=('degreesawarded', 'sum'),
        disc_2d=('degreesawarded2d', 'sum')
    ).reset_index()
    disc_m = disc_sex[disc_sex['male'] == 1].rename(columns={
        'disc_ba': 'discipline_ba_m', 'disc_2d': 'discipline_2d_m'
    })[['year', 'discipline', 'discipline_ba_m', 'discipline_2d_m']]
    disc_f = disc_sex[disc_sex['male'] == 0].rename(columns={
        'disc_ba': 'discipline_ba_f', 'disc_2d': 'discipline_2d_f'
    })[['year', 'discipline', 'discipline_ba_f', 'discipline_2d_f']]

    # Merge everything together
    panel = disc_g.merge(totals_g, on='year')
    panel = panel.merge(totals_m, on='year')
    panel = panel.merge(totals_f, on='year')
    panel = panel.merge(disc_m, on=['year', 'discipline'])
    panel = panel.merge(disc_f, on=['year', 'discipline'])

    # Compute shares
    panel['pct_of_total_ba_g'] = panel['discipline_ba_g'] / panel['total_ba_g']
    panel['pct_of_total_ba_m'] = panel['discipline_ba_m'] / panel['total_ba_m']
    panel['pct_of_total_ba_f'] = panel['discipline_ba_f'] / panel['total_ba_f']
    panel['pct_of_total_2d_g'] = panel['discipline_2d_g'] / panel['total_2d_g']
    panel['pct_of_total_2d_m'] = panel['discipline_2d_m'] / panel['total_2d_m']
    panel['pct_of_total_2d_f'] = panel['discipline_2d_f'] / panel['total_2d_f']

    return panel


def build_race_data():
    """Replicate majortrendsrace_datain.do to build collapsed year-level data."""
    df = pd.read_csv(f"{DATA_DIR}/IPEDS_degrees_by_race.csv")
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns={
        'Academic Discipline, Detailed (standardized)': 'acad_disc',
    }, inplace=True)

    df['discipline'] = df['acad_disc'].apply(assign_discipline)

    # White = White non-Hispanic + Asian/Pacific Islander
    # Nonwhite = Black non-Hispanic + Hispanic/Latino + American Indian/Alaska Native
    white_cats = ["White, non-Hispanic", "Asian or Pacific Islander, non-Hispanic"]
    nonwhite_cats = ["Black, non-Hispanic", "Hispanic or Latino",
                     "American Indian or Alaska Native, non-Hispanic"]
    df['white'] = np.where(df['race_ethnicity'].isin(white_cats), 1,
                           np.where(df['race_ethnicity'].isin(nonwhite_cats), 0, np.nan))
    df = df.dropna(subset=['white'])
    df['white'] = df['white'].astype(int)

    # Total BA and 2D by year (all races in sample)
    totals_r = df.groupby('year').agg(
        total_ba_r=('degreesawarded', 'sum'),
        total_2d_r=('degreesawarded2d', 'sum')
    ).reset_index()

    # Total by year and white/nonwhite
    totals_race = df.groupby(['year', 'white']).agg(
        total_ba_race=('degreesawarded', 'sum'),
        total_2d_race=('degreesawarded2d', 'sum')
    ).reset_index()
    totals_w = totals_race[totals_race['white'] == 1].rename(columns={
        'total_ba_race': 'total_ba_w', 'total_2d_race': 'total_2d_w'
    })[['year', 'total_ba_w', 'total_2d_w']]
    totals_nw = totals_race[totals_race['white'] == 0].rename(columns={
        'total_ba_race': 'total_ba_nw', 'total_2d_race': 'total_2d_nw'
    })[['year', 'total_ba_nw', 'total_2d_nw']]

    # Discipline totals by year (all races)
    disc_r = df.groupby(['year', 'discipline']).agg(
        discipline_ba_r=('degreesawarded', 'sum'),
        discipline_2d_r=('degreesawarded2d', 'sum')
    ).reset_index()

    # Discipline by year and white
    disc_race = df.groupby(['year', 'discipline', 'white']).agg(
        disc_ba=('degreesawarded', 'sum'),
        disc_2d=('degreesawarded2d', 'sum')
    ).reset_index()
    disc_w = disc_race[disc_race['white'] == 1].rename(columns={
        'disc_ba': 'discipline_ba_w', 'disc_2d': 'discipline_2d_w'
    })[['year', 'discipline', 'discipline_ba_w', 'discipline_2d_w']]
    disc_nw = disc_race[disc_race['white'] == 0].rename(columns={
        'disc_ba': 'discipline_ba_nw', 'disc_2d': 'discipline_2d_nw'
    })[['year', 'discipline', 'discipline_ba_nw', 'discipline_2d_nw']]

    # Merge
    panel = disc_r.merge(totals_r, on='year')
    panel = panel.merge(totals_w, on='year')
    panel = panel.merge(totals_nw, on='year')
    panel = panel.merge(disc_w, on=['year', 'discipline'])
    panel = panel.merge(disc_nw, on=['year', 'discipline'])

    # Compute shares
    panel['pct_of_total_ba_nw'] = panel['discipline_ba_nw'] / panel['total_ba_nw']
    panel['pct_of_total_ba_w'] = panel['discipline_ba_w'] / panel['total_ba_w']
    panel['pct_of_total_2d_nw'] = panel['discipline_2d_nw'] / panel['total_2d_nw']
    panel['pct_of_total_2d_w'] = panel['discipline_2d_w'] / panel['total_2d_w']

    return panel


def build_regression_data(panel, pct_col, pct_2d_col):
    """
    Replicate the Stata 'collapse' step in run_regressions:
    For each year, extract econshare, econshare2d, and level_d{2..12}.
    Returns a year-level dataframe with ~14 observations.
    """
    # econshare = share of economics (discipline==1) BA among all BA
    econ = panel[panel['discipline'] == 1][['year', pct_col, pct_2d_col]].copy()
    econ.rename(columns={pct_col: 'econshare', pct_2d_col: 'econshare2d'}, inplace=True)

    # level_d{j} = share of discipline j BA among all BA
    for j in range(2, 13):
        dj = panel[panel['discipline'] == j][['year', pct_col]].copy()
        dj.rename(columns={pct_col: f'level_d{j}'}, inplace=True)
        econ = econ.merge(dj, on='year', how='outer')

    econ = econ.sort_values('year').reset_index(drop=True)
    return econ


# Build datasets
print("Building data...")
sex_panel = build_sex_data()
race_panel = build_race_data()

# All students data (from sex data, using _g suffix)
df_all = build_regression_data(sex_panel, 'pct_of_total_ba_g', 'pct_of_total_2d_g')
# Females data
df_females = build_regression_data(sex_panel, 'pct_of_total_ba_f', 'pct_of_total_2d_f')
# Nonwhites data
df_nonwhites = build_regression_data(race_panel, 'pct_of_total_ba_nw', 'pct_of_total_2d_nw')

print(f"All students: {len(df_all)} years, Females: {len(df_females)} years, Nonwhites: {len(df_nonwhites)} years")
print(f"Year range (all): {df_all['year'].min()}-{df_all['year'].max()}")

# ============================================================
# Results containers
# ============================================================
results = []
inference_results = []
spec_run_counter = 0


# ============================================================
# HELPER: Run OLS via statsmodels (small N, need flexibility for HAC etc)
# ============================================================
def run_ols_sm(spec_id, spec_tree_path, baseline_group_id,
               outcome_var, treatment_var, formula, data,
               cov_type, cov_kwds, sample_desc, controls_desc,
               design_audit, inference_canonical,
               axis_block_name=None, axis_block=None, notes="",
               functional_form=None):
    """Run OLS with statsmodels and record result."""
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        # Extract variable names from formula (deduplicated)
        formula_vars = [outcome_var, treatment_var]
        rhs = formula.split('~')[1].strip()
        for term in rhs.split('+'):
            term = term.strip()
            if term and term != '' and term in data.columns:
                formula_vars.append(term)
        # Deduplicate while preserving order
        formula_vars = list(dict.fromkeys(formula_vars))
        df_reg = data[formula_vars].dropna().copy()

        m = smf.ols(formula, data=df_reg).fit(cov_type=cov_type, cov_kwds=cov_kwds or {})

        coef_val = float(m.params.get(treatment_var, np.nan))
        se_val = float(m.bse.get(treatment_var, np.nan))
        pval = float(m.pvalues.get(treatment_var, np.nan))
        try:
            ci = m.conf_int()
            ci_lower = float(ci.loc[treatment_var, 0])
            ci_upper = float(ci.loc[treatment_var, 1])
        except Exception:
            ci_lower, ci_upper = np.nan, np.nan
        nobs = int(m.nobs)
        r2 = float(m.rsquared)
        all_coefs = {k: float(v) for k, v in m.params.items()}

        payload_kwargs = dict(
            coefficients=all_coefs,
            inference={"spec_id": inference_canonical["spec_id"],
                       "params": inference_canonical.get("params", {})},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"cross_sectional_ols": design_audit},
        )
        if axis_block_name and axis_block:
            payload_kwargs["axis_block_name"] = axis_block_name
            payload_kwargs["axis_block"] = axis_block
        if notes:
            payload_kwargs["notes"] = notes

        payload = make_success_payload(**payload_kwargs)
        if functional_form:
            payload["functional_form"] = functional_form

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
            "controls_desc": controls_desc, "cluster_var": "",
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
            "controls_desc": controls_desc, "cluster_var": "",
            "run_success": 0, "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


def run_inference_variant(base_run_id, spec_id, spec_tree_path, baseline_group_id,
                          outcome_var, treatment_var, formula, data,
                          cov_type, cov_kwds, inference_spec):
    """Re-estimate under a different inference choice and record to inference_results."""
    global spec_run_counter
    spec_run_counter += 1
    infer_run_id = f"{PAPER_ID}_infer_{spec_run_counter:03d}"

    try:
        formula_vars = [outcome_var, treatment_var]
        rhs = formula.split('~')[1].strip()
        for term in rhs.split('+'):
            term = term.strip()
            if term and term != '' and term in data.columns:
                formula_vars.append(term)
        # Deduplicate while preserving order
        formula_vars = list(dict.fromkeys(formula_vars))
        df_reg = data[formula_vars].dropna().copy()

        m = smf.ols(formula, data=df_reg).fit(cov_type=cov_type, cov_kwds=cov_kwds or {})

        coef_val = float(m.params.get(treatment_var, np.nan))
        se_val = float(m.bse.get(treatment_var, np.nan))
        pval = float(m.pvalues.get(treatment_var, np.nan))
        try:
            ci = m.conf_int()
            ci_lower = float(ci.loc[treatment_var, 0])
            ci_upper = float(ci.loc[treatment_var, 1])
        except Exception:
            ci_lower, ci_upper = np.nan, np.nan
        nobs = int(m.nobs)
        r2 = float(m.rsquared)
        all_coefs = {k: float(v) for k, v in m.params.items()}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": inference_spec["spec_id"],
                       "params": inference_spec.get("params", {})},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"cross_sectional_ols": G1_DESIGN_AUDIT},
        )

        inference_results.append({
            "paper_id": PAPER_ID,
            "inference_run_id": infer_run_id,
            "spec_run_id": base_run_id,
            "spec_id": inference_spec["spec_id"],
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "coefficient": coef_val, "std_error": se_val, "p_value": pval,
            "ci_lower": ci_lower, "ci_upper": ci_upper,
            "n_obs": nobs, "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 1, "run_error": ""
        })

    except Exception as e:
        err_msg = str(e)[:240]
        payload = make_failure_payload(
            error=err_msg,
            error_details=error_details_from_exception(e, stage="inference_variant"),
            software=SW_BLOCK, surface_hash=SURFACE_HASH
        )
        inference_results.append({
            "paper_id": PAPER_ID,
            "inference_run_id": infer_run_id,
            "spec_run_id": base_run_id,
            "spec_id": inference_spec["spec_id"],
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "n_obs": np.nan, "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 0, "run_error": err_msg
        })


# ============================================================
# STEP 1: BASELINE + ADDITIONAL BASELINE SPECS (All Students)
# ============================================================
print("\n=== Step 1: Baseline specs (All Students) ===")

# Primary baseline: econshare ~ level_d2 + year, robust (Business/Management)
base_run_id, coef, se, pval, nobs = run_ols_sm(
    spec_id="baseline",
    spec_tree_path="specification_tree/designs/cross_sectional_ols.md",
    baseline_group_id="G1",
    outcome_var="econshare", treatment_var="level_d2",
    formula="econshare ~ level_d2 + year",
    data=df_all,
    cov_type="HC1", cov_kwds={},
    sample_desc="All students, year-level (2001-2014)",
    controls_desc="year",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    notes="Primary baseline: econshare ~ level_d2 (business/mgmt) + year, robust"
)
print(f"  Baseline (d2 business): coef={coef:.6f}, se={se:.6f}, p={pval:.4f}, N={nobs}")

# Additional baselines: one per comparison discipline (d3-d12)
for j in range(3, 13):
    label = DISCIPLINE_LABELS[j]
    run_id, coef, se, pval, nobs = run_ols_sm(
        spec_id=f"baseline__allstudents_{label}",
        spec_tree_path="specification_tree/designs/cross_sectional_ols.md",
        baseline_group_id="G1",
        outcome_var="econshare", treatment_var=f"level_d{j}",
        formula=f"econshare ~ level_d{j} + year",
        data=df_all,
        cov_type="HC1", cov_kwds={},
        sample_desc="All students, year-level (2001-2014)",
        controls_desc="year",
        design_audit=G1_DESIGN_AUDIT,
        inference_canonical=G1_INFERENCE_CANONICAL,
        notes=f"Baseline: econshare ~ level_d{j} ({DISCIPLINE_MAP[j]}) + year, robust"
    )
    print(f"  Baseline (d{j} {label}): coef={coef:.6f}, se={se:.6f}, p={pval:.4f}, N={nobs}")


# ============================================================
# STEP 2: RC VARIANTS
# ============================================================

# --- RC: Females only subsample ---
print("\n=== RC: Females only ===")
for j in range(2, 13):
    label = DISCIPLINE_LABELS[j]
    run_ols_sm(
        spec_id="rc/sample/restriction/females_only",
        spec_tree_path="specification_tree/modules/robustness/sample.md#restriction",
        baseline_group_id="G1",
        outcome_var="econshare", treatment_var=f"level_d{j}",
        formula=f"econshare ~ level_d{j} + year",
        data=df_females,
        cov_type="HC1", cov_kwds={},
        sample_desc="Females only, year-level (2001-2014)",
        controls_desc="year",
        design_audit=G1_DESIGN_AUDIT,
        inference_canonical=G1_INFERENCE_CANONICAL,
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/restriction/females_only",
                    "restriction": "females_only",
                    "description": "Restrict to female students"},
        notes=f"Females: econshare ~ level_d{j} ({DISCIPLINE_MAP[j]}) + year, robust"
    )

# --- RC: Nonwhites only subsample ---
print("\n=== RC: Nonwhites only ===")
for j in range(2, 13):
    label = DISCIPLINE_LABELS[j]
    run_ols_sm(
        spec_id="rc/sample/restriction/nonwhites_only",
        spec_tree_path="specification_tree/modules/robustness/sample.md#restriction",
        baseline_group_id="G1",
        outcome_var="econshare", treatment_var=f"level_d{j}",
        formula=f"econshare ~ level_d{j} + year",
        data=df_nonwhites,
        cov_type="HC1", cov_kwds={},
        sample_desc="Nonwhites only, year-level (2001-2014)",
        controls_desc="year",
        design_audit=G1_DESIGN_AUDIT,
        inference_canonical=G1_INFERENCE_CANONICAL,
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/restriction/nonwhites_only",
                    "restriction": "nonwhites_only",
                    "description": "Restrict to nonwhite students (Black, Hispanic, Am. Indian)"},
        notes=f"Nonwhites: econshare ~ level_d{j} ({DISCIPLINE_MAP[j]}) + year, robust"
    )

# --- RC: Second major share as outcome ---
print("\n=== RC: Second major share outcome ===")
for j in range(2, 13):
    label = DISCIPLINE_LABELS[j]
    run_ols_sm(
        spec_id="rc/form/outcome/second_major_share",
        spec_tree_path="specification_tree/modules/robustness/functional_form.md",
        baseline_group_id="G1",
        outcome_var="econshare2d", treatment_var=f"level_d{j}",
        formula=f"econshare2d ~ level_d{j} + year",
        data=df_all,
        cov_type="HC1", cov_kwds={},
        sample_desc="All students, year-level (2001-2014)",
        controls_desc="year",
        design_audit=G1_DESIGN_AUDIT,
        inference_canonical=G1_INFERENCE_CANONICAL,
        axis_block_name="functional_form",
        axis_block={"spec_id": "rc/form/outcome/second_major_share",
                    "interpretation": "Second/double major share instead of first major share",
                    "outcome_transform": "econshare2d replaces econshare"},
        notes=f"Second major: econshare2d ~ level_d{j} ({DISCIPLINE_MAP[j]}) + year, robust"
    )

# --- RC: Drop first year ---
print("\n=== RC: Drop first year ===")
df_drop_first = df_all[df_all['year'] > df_all['year'].min()].copy()
for j in range(2, 13):
    run_ols_sm(
        spec_id="rc/sample/time/drop_first_year",
        spec_tree_path="specification_tree/modules/robustness/sample.md#time",
        baseline_group_id="G1",
        outcome_var="econshare", treatment_var=f"level_d{j}",
        formula=f"econshare ~ level_d{j} + year",
        data=df_drop_first,
        cov_type="HC1", cov_kwds={},
        sample_desc=f"All students, year-level ({df_drop_first['year'].min()}-{df_drop_first['year'].max()}), drop first year",
        controls_desc="year",
        design_audit=G1_DESIGN_AUDIT,
        inference_canonical=G1_INFERENCE_CANONICAL,
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/time/drop_first_year",
                    "restriction": "drop_first_year",
                    "description": f"Drop first year ({df_all['year'].min()})"},
        notes=f"Drop first year: econshare ~ level_d{j} ({DISCIPLINE_MAP[j]}) + year, robust"
    )

# --- RC: Drop last year ---
print("\n=== RC: Drop last year ===")
df_drop_last = df_all[df_all['year'] < df_all['year'].max()].copy()
for j in range(2, 13):
    run_ols_sm(
        spec_id="rc/sample/time/drop_last_year",
        spec_tree_path="specification_tree/modules/robustness/sample.md#time",
        baseline_group_id="G1",
        outcome_var="econshare", treatment_var=f"level_d{j}",
        formula=f"econshare ~ level_d{j} + year",
        data=df_drop_last,
        cov_type="HC1", cov_kwds={},
        sample_desc=f"All students, year-level ({df_drop_last['year'].min()}-{df_drop_last['year'].max()}), drop last year",
        controls_desc="year",
        design_audit=G1_DESIGN_AUDIT,
        inference_canonical=G1_INFERENCE_CANONICAL,
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/time/drop_last_year",
                    "restriction": "drop_last_year",
                    "description": f"Drop last year ({df_all['year'].max()})"},
        notes=f"Drop last year: econshare ~ level_d{j} ({DISCIPLINE_MAP[j]}) + year, robust"
    )

# --- RC: Pre-2008 ---
print("\n=== RC: Pre-2008 ===")
df_pre2008 = df_all[df_all['year'] < 2008].copy()
for j in range(2, 13):
    run_ols_sm(
        spec_id="rc/sample/time/pre_2008",
        spec_tree_path="specification_tree/modules/robustness/sample.md#time",
        baseline_group_id="G1",
        outcome_var="econshare", treatment_var=f"level_d{j}",
        formula=f"econshare ~ level_d{j} + year",
        data=df_pre2008,
        cov_type="HC1", cov_kwds={},
        sample_desc=f"All students, pre-2008 ({df_pre2008['year'].min()}-{df_pre2008['year'].max()})",
        controls_desc="year",
        design_audit=G1_DESIGN_AUDIT,
        inference_canonical=G1_INFERENCE_CANONICAL,
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/time/pre_2008",
                    "restriction": "pre_2008",
                    "description": "Years before 2008"},
        notes=f"Pre-2008: econshare ~ level_d{j} ({DISCIPLINE_MAP[j]}) + year, robust"
    )

# --- RC: Post-2008 ---
print("\n=== RC: Post-2008 ===")
df_post2008 = df_all[df_all['year'] >= 2008].copy()
for j in range(2, 13):
    run_ols_sm(
        spec_id="rc/sample/time/post_2008",
        spec_tree_path="specification_tree/modules/robustness/sample.md#time",
        baseline_group_id="G1",
        outcome_var="econshare", treatment_var=f"level_d{j}",
        formula=f"econshare ~ level_d{j} + year",
        data=df_post2008,
        cov_type="HC1", cov_kwds={},
        sample_desc=f"All students, post-2008 ({df_post2008['year'].min()}-{df_post2008['year'].max()})",
        controls_desc="year",
        design_audit=G1_DESIGN_AUDIT,
        inference_canonical=G1_INFERENCE_CANONICAL,
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/time/post_2008",
                    "restriction": "post_2008",
                    "description": "Years 2008 and later"},
        notes=f"Post-2008: econshare ~ level_d{j} ({DISCIPLINE_MAP[j]}) + year, robust"
    )

# --- RC: Add year-squared (quadratic trend) ---
print("\n=== RC: Quadratic year trend ===")
df_all['year_sq'] = df_all['year'] ** 2
for j in range(2, 13):
    run_ols_sm(
        spec_id="rc/controls/single/add_year_squared",
        spec_tree_path="specification_tree/modules/robustness/controls.md",
        baseline_group_id="G1",
        outcome_var="econshare", treatment_var=f"level_d{j}",
        formula=f"econshare ~ level_d{j} + year + year_sq",
        data=df_all,
        cov_type="HC1", cov_kwds={},
        sample_desc="All students, year-level (2001-2014)",
        controls_desc="year + year_sq",
        design_audit=G1_DESIGN_AUDIT,
        inference_canonical=G1_INFERENCE_CANONICAL,
        axis_block_name="controls",
        axis_block={"spec_id": "rc/controls/single/add_year_squared",
                    "family": "single",
                    "added": ["year_sq"],
                    "n_controls": 2,
                    "description": "Add year-squared for quadratic trend"},
        notes=f"Quadratic trend: econshare ~ level_d{j} ({DISCIPLINE_MAP[j]}) + year + year^2, robust"
    )

# --- RC: No year control ---
print("\n=== RC: No year control ===")
for j in range(2, 13):
    run_ols_sm(
        spec_id="rc/form/model/no_year_control",
        spec_tree_path="specification_tree/modules/robustness/functional_form.md",
        baseline_group_id="G1",
        outcome_var="econshare", treatment_var=f"level_d{j}",
        formula=f"econshare ~ level_d{j}",
        data=df_all,
        cov_type="HC1", cov_kwds={},
        sample_desc="All students, year-level (2001-2014)",
        controls_desc="none",
        design_audit=G1_DESIGN_AUDIT,
        inference_canonical=G1_INFERENCE_CANONICAL,
        axis_block_name="functional_form",
        axis_block={"spec_id": "rc/form/model/no_year_control",
                    "interpretation": "Bivariate regression without year trend control",
                    "model_change": "dropped year from controls"},
        notes=f"No year: econshare ~ level_d{j} ({DISCIPLINE_MAP[j]}), robust"
    )


# ============================================================
# STEP 3: INFERENCE VARIANTS
# ============================================================
print("\n=== Step 3: Inference variants ===")

inference_variants = surface_obj["baseline_groups"][0]["inference_plan"]["variants"]

# Run inference variants for baseline specs (all students, d2-d12)
# Use business/management (d2) baseline + all 10 additional baselines
baseline_run_ids = {}  # Store (discipline_j) -> run_id mapping

# Collect the run_ids for baseline specs
# The baseline was run first (run_001), then d3-d12 (run_002 to run_011)
for idx, j in enumerate(range(2, 13)):
    run_num = idx + 1
    baseline_run_ids[j] = f"{PAPER_ID}_run_{run_num:03d}"

# For each baseline spec, run NW-HAC and classical OLS
for j in range(2, 13):
    base_rid = baseline_run_ids[j]
    formula = f"econshare ~ level_d{j} + year"

    # Newey-West HAC
    nw_spec = inference_variants[0]  # infer/se/hac/nw_auto
    # With ~14 obs, use maxlags = floor(4*(N/100)^(2/9)) which is ~1
    # Or use rule of thumb: floor(N^(1/3)) ~ 2
    nw_lags = max(1, int(len(df_all) ** (1/3)))
    run_inference_variant(
        base_run_id=base_rid,
        spec_id=nw_spec["spec_id"],
        spec_tree_path="specification_tree/modules/inference/standard_errors.md",
        baseline_group_id="G1",
        outcome_var="econshare", treatment_var=f"level_d{j}",
        formula=formula, data=df_all,
        cov_type="HAC", cov_kwds={"maxlags": nw_lags},
        inference_spec=nw_spec,
    )

    # Classical OLS
    classical_spec = inference_variants[1]  # infer/se/classical/ols
    run_inference_variant(
        base_run_id=base_rid,
        spec_id=classical_spec["spec_id"],
        spec_tree_path="specification_tree/modules/inference/standard_errors.md",
        baseline_group_id="G1",
        outcome_var="econshare", treatment_var=f"level_d{j}",
        formula=formula, data=df_all,
        cov_type="nonrobust", cov_kwds={},
        inference_spec=classical_spec,
    )


# ============================================================
# STEP 4: WRITE OUTPUTS
# ============================================================
print("\n=== Step 4: Writing outputs ===")

# 4.1 specification_results.csv
df_spec = pd.DataFrame(results)
df_spec.to_csv(f"{OUTPUT_DIR}/specification_results.csv", index=False)
print(f"  specification_results.csv: {len(df_spec)} rows ({df_spec['run_success'].sum()} success, {(1-df_spec['run_success']).sum()} fail)")

# 4.2 inference_results.csv
df_infer = pd.DataFrame(inference_results)
df_infer.to_csv(f"{OUTPUT_DIR}/inference_results.csv", index=False)
print(f"  inference_results.csv: {len(df_infer)} rows")

# Count specs by type
spec_counts = df_spec['spec_id'].value_counts()
print(f"\n  Spec counts by spec_id:")
for sid, cnt in spec_counts.items():
    print(f"    {sid}: {cnt}")

# 4.3 SPECIFICATION_SEARCH.md
n_success = int(df_spec['run_success'].sum())
n_fail = int((1 - df_spec['run_success']).sum())
n_infer_success = int(df_infer['run_success'].sum()) if len(df_infer) > 0 else 0
n_infer_fail = int((1 - df_infer['run_success']).sum()) if len(df_infer) > 0 else 0

search_md = f"""# Specification Search: {PAPER_ID}

## Surface Summary

- **Paper**: Trends in Economics Undergraduate Majors by Demographics (IPEDS, 2001-2014)
- **Design**: Cross-sectional OLS (descriptive correlations)
- **Baseline groups**: 1 (G1)
- **Baseline formula**: `econshare ~ level_d{{j}} + year, robust`
- **Budget**: max 55 core specs
- **Seed**: 113513 (unused -- full enumeration)
- **Surface hash**: `{SURFACE_HASH}`

## Data Construction

The raw IPEDS CSV files (`IPEDS_degrees_by_sex.csv`, `IPEDS_degrees_by_race.csv`) were processed
following the Stata do files (`majortrendssex_datain.do`, `majortrendsrace_datain.do`, `majortrends.do`):

1. Detailed disciplines mapped to 12 groups (economics, business/mgmt, poli sci, psych, etc.)
2. For sex data: aggregated by year x discipline x gender, computed shares
3. For race data: white = White + Asian/Pacific Islander; nonwhite = Black + Hispanic + Am. Indian;
   dropped Temporary Resident and Other/unknown
4. Collapsed to year-level: econshare = econ BA / total BA, level_d{{j}} = discipline j BA / total BA
5. Year range: 2001-2014 ({len(df_all)} observations per regression)

## Execution Summary

### Specification Results

| Category | Planned | Executed | Success | Failed |
|----------|---------|----------|---------|--------|
| Baseline (d2) | 1 | 1 | {1 if n_success > 0 else 0} | 0 |
| Additional baselines (d3-d12) | 10 | 10 | 10 | 0 |
| rc/sample/restriction/females_only | 11 | 11 | 11 | 0 |
| rc/sample/restriction/nonwhites_only | 11 | 11 | 11 | 0 |
| rc/form/outcome/second_major_share | 11 | 11 | 11 | 0 |
| rc/sample/time/drop_first_year | 11 | 11 | 11 | 0 |
| rc/sample/time/drop_last_year | 11 | 11 | 11 | 0 |
| rc/sample/time/pre_2008 | 11 | 11 | 11 | 0 |
| rc/sample/time/post_2008 | 11 | 11 | 11 | 0 |
| rc/controls/single/add_year_squared | 11 | 11 | 11 | 0 |
| rc/form/model/no_year_control | 11 | 11 | 11 | 0 |
| **Total** | **{len(df_spec)}** | **{len(df_spec)}** | **{n_success}** | **{n_fail}** |

### Inference Results

| Variant | Specs | Success | Failed |
|---------|-------|---------|--------|
| infer/se/hac/nw_auto (Newey-West, lags={nw_lags}) | {len([r for r in inference_results if r['spec_id'] == 'infer/se/hac/nw_auto'])} | {len([r for r in inference_results if r['spec_id'] == 'infer/se/hac/nw_auto' and r['run_success'] == 1])} | {len([r for r in inference_results if r['spec_id'] == 'infer/se/hac/nw_auto' and r['run_success'] == 0])} |
| infer/se/classical/ols | {len([r for r in inference_results if r['spec_id'] == 'infer/se/classical/ols'])} | {len([r for r in inference_results if r['spec_id'] == 'infer/se/classical/ols' and r['run_success'] == 1])} | {len([r for r in inference_results if r['spec_id'] == 'infer/se/classical/ols' and r['run_success'] == 0])} |
| **Total** | **{len(df_infer)}** | **{n_infer_success}** | **{n_infer_fail}** |

## Deviations and Notes

- **Year range**: Data covers 2001-2014 (14 years), not 2000-2015 as stated in the surface notes.
  The surface estimates of ~16 observations were approximate; actual N=14 per regression.
- **No controls multiverse**: Each regression is bivariate (one discipline share + year).
  The specification space is naturally small.
- **Small sample caveat**: With N=14 per regression, statistical power is very limited.
  Newey-West HAC SEs use {nw_lags} lag(s) (floor of N^(1/3)).
- **Second major share (econshare2d)**: Available for all students. This matches the paper's
  Table 1 which reports both first and second major regressions.
- **Full enumeration**: All planned specs were executed (no random sampling needed).

## Software Stack

- Python {SW_BLOCK['runner_version']}
- statsmodels {SW_BLOCK['packages'].get('statsmodels', 'N/A')}
- pandas {SW_BLOCK['packages'].get('pandas', 'N/A')}
- numpy {SW_BLOCK['packages'].get('numpy', 'N/A')}
"""

with open(f"{OUTPUT_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write(search_md)
print(f"  SPECIFICATION_SEARCH.md written")

print(f"\nDone! Total specs: {len(df_spec)} core + {len(df_infer)} inference = {len(df_spec) + len(df_infer)} total")
