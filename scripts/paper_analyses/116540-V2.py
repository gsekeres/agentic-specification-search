#!/usr/bin/env python3
"""
Specification Search: 116540-V2
Paper: Individual Development Accounts and Homeownership among Low-Income Renters
Journal: AEJ: Policy

This script runs a systematic specification search following the i4r methodology.
Method classification: Cross-sectional OLS / RCT analysis
"""

import pandas as pd
import numpy as np
import json
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

# Try to import pyfixest, fall back to statsmodels if not available
try:
    import pyfixest as pf
    HAS_PYFIXEST = True
except ImportError:
    HAS_PYFIXEST = False

import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.discrete.discrete_model import Logit, Probit

# =============================================================================
# Configuration
# =============================================================================

PAPER_ID = "116540-V2"
JOURNAL = "AEJ-Policy"
PAPER_TITLE = "Individual Development Accounts and Homeownership among Low-Income Renters"
DATA_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/116540-V2/AEJPol-2011-0054_R1_data/AEJPol-2011-0054_R1_shared_data_file.dta"
OUTPUT_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/116540-V2"
SCRIPT_PATH = "scripts/paper_analyses/116540-V2.py"

# =============================================================================
# Data Preparation
# =============================================================================

def load_and_prepare_data():
    """Load and prepare the data for analysis."""
    df = pd.read_stata(DATA_PATH)

    # Convert categorical variables to numeric
    # Treatment
    df['treat'] = (df['treat'] == 'Treatment').astype(float)

    # Homeownership
    df['own_home_u17'] = (df['own_home_u17'] == 'yes').astype(float)
    df['own_home_u42'] = (df['own_home_u42'] == 'yes').astype(float)

    # Sample indicator
    df['sample14'] = (df['sample14'] == 'yes').astype(float)

    # Gender
    df['female'] = (df['female_u17'] == 'female').astype(float)

    # Race caucasian
    df['race_cau'] = (df['race_cau_u17'] == 'caucasian').astype(float)

    # Subsidized housing
    df['section8'] = (df['section8_u17'] == 'yes').astype(float)
    df['pub_home'] = (df['pub_home_u17'] == 'yes').astype(float)

    # Unsubsidized (not section 8 and not public housing)
    df['unsubsidized'] = ((df['section8'] == 0) & (df['pub_home'] == 0)).astype(float)

    # Other binary variables
    df['own_bank'] = (df['own_bank_u17'] == 'yes').astype(float)
    df['ins_heal'] = (df['ins_heal_u17'] == 'yes').astype(float)
    df['own_bus'] = (df['own_bus_u17'] == 'yes').astype(float)
    df['own_prop'] = (df['own_prop_u17'] == 'yes').astype(float)
    df['own_ira'] = (df['own_ira_u17'] == 'yes').astype(float)
    df['src_welf'] = (df['src_welf_u17'] == 'yes').astype(float)
    df['own_car'] = (df['own_car_u17'] == 'yes').astype(float)

    # Married indicator
    df['married'] = (df['marital_u17'] == 'married').astype(float)

    # Education categories based on labels
    ed_map = {
        'grade school, middle school, or jr high': 1,
        'some hs': 2,
        'graduate hs or ged': 3,
        'some college': 4,
        'grad 2yr college': 5,
        'grad 4yr college': 6,
        'some grad school': 7,
        'finished grad school': 8
    }
    df['ed_num'] = df['ed_u17'].map(ed_map)
    df['less_than_hs'] = (df['ed_num'] < 3).astype(float)
    df['hs_grad'] = (df['ed_num'] == 3).astype(float)
    df['some_college'] = ((df['ed_num'] >= 4) & (df['ed_num'] <= 5)).astype(float)
    df['college_grad'] = (df['ed_num'] >= 6).astype(float)

    # Create tri_ed (0=less than college, 1=some college, 2=college grad)
    df['tri_ed'] = 0
    df.loc[df['ed_num'] >= 4, 'tri_ed'] = 1
    df.loc[df['ed_num'] >= 6, 'tri_ed'] = 2

    # Education dummies
    df['tri_ed_1'] = (df['tri_ed'] == 1).astype(float)
    df['tri_ed_2'] = (df['tri_ed'] == 2).astype(float)

    # Age
    df['age'] = df['age_u17'].astype(float)
    df['age_35_plus'] = (df['age'] >= 35).astype(float)

    # Cohort
    df['cohort_num'] = df['cohort'].astype(float)
    df['late_cohort'] = ((df['cohort_num'] == 12) | (df['cohort_num'] == 13)).astype(float)

    # Children
    df['hh_child'] = df['hh_child_u17'].astype(float)
    df['has_children'] = (df['hh_child'] > 0).astype(float)

    # Adults
    df['hh_adult'] = df['hh_adult_u17'].astype(float)

    # Income
    df['inc_tot'] = df['inc_tot_u17'].astype(float)

    # Assets and liabilities
    df['ass_tot'] = df['ass_tot_u17'].astype(float)
    df['lib_tot'] = df['lib_tot_u17'].astype(float)

    # Scales
    df['ci_scale'] = df['ci_scale_u17'].astype(float)
    df['gv_scale'] = df['gv_scale_u17'].astype(float)
    df['gt_scale'] = df['gt_scale_u17'].astype(float)
    df['str_scale'] = df['str_scale_u17'].astype(float)
    df['own_scale'] = df['own_scale_u17'].astype(float)

    # Satisfaction variables
    df['sat_heal_good'] = df['sat_heal_u17'].isin(['1', 'top category', 1]).astype(float)
    df['sat_fin_good'] = df['sat_fin2_u17'].isin(['very satisfied', 'somewhat satisfied']).astype(float)

    # Filter to baseline renters with wave 4 data
    df_analysis = df[(df['own_home_u17'] == 0) & (df['sample14'] == 1)].copy()

    # Create high income indicator (above median among analytic sample)
    inc_median = df_analysis['inc_tot'].median()
    df_analysis['high_income'] = (df_analysis['inc_tot'] >= inc_median).astype(float)

    # Asset categories (based on do file: 1421 cutoffs)
    df_analysis['cat_ass_1'] = ((df_analysis['ass_tot'] >= 1421) & (df_analysis['ass_tot'] < 2842)).astype(float)
    df_analysis['cat_ass_2'] = ((df_analysis['ass_tot'] >= 2842) & (df_analysis['ass_tot'] < 4263)).astype(float)
    df_analysis['cat_ass_3'] = (df_analysis['ass_tot'] >= 4263).astype(float)
    df_analysis['cat_ass_miss'] = df_analysis['ass_tot'].isna().astype(float)

    df_analysis['cat_lib_1'] = ((df_analysis['lib_tot'] >= 1421) & (df_analysis['lib_tot'] < 2842)).astype(float)
    df_analysis['cat_lib_2'] = ((df_analysis['lib_tot'] >= 2842) & (df_analysis['lib_tot'] < 4263)).astype(float)
    df_analysis['cat_lib_3'] = (df_analysis['lib_tot'] >= 4263).astype(float)
    df_analysis['cat_lib_miss'] = df_analysis['lib_tot'].isna().astype(float)

    return df_analysis


# =============================================================================
# Regression Functions
# =============================================================================

def run_ols_regression(df, outcome, treatment, controls, cluster_var=None, weights=None):
    """Run OLS regression with robust or clustered standard errors."""
    # Drop missing values
    all_vars = [outcome, treatment] + controls
    if cluster_var:
        all_vars.append(cluster_var)
    if weights:
        all_vars.append(weights)

    df_reg = df[all_vars].dropna()

    if len(df_reg) < 10:
        return None

    # Prepare X and y
    y = df_reg[outcome]
    X = df_reg[[treatment] + controls]
    X = sm.add_constant(X)

    try:
        if weights:
            model = OLS(y, X, weights=df_reg[weights]).fit(cov_type='HC1')
        else:
            model = OLS(y, X).fit(cov_type='HC1')

        # Get treatment coefficient
        treat_idx = 1  # After constant
        coef = model.params.iloc[treat_idx]
        se = model.bse.iloc[treat_idx]
        tstat = model.tvalues.iloc[treat_idx]
        pval = model.pvalues.iloc[treat_idx]
        ci = model.conf_int().iloc[treat_idx]

        # Build coefficient vector
        coef_vector = {
            "treatment": {
                "var": treatment,
                "coef": float(coef),
                "se": float(se),
                "pval": float(pval)
            },
            "controls": [],
            "diagnostics": {}
        }

        for i, var in enumerate(controls):
            coef_vector["controls"].append({
                "var": var,
                "coef": float(model.params.iloc[i + 2]),  # +2 for const and treatment
                "se": float(model.bse.iloc[i + 2]),
                "pval": float(model.pvalues.iloc[i + 2])
            })

        return {
            "coefficient": float(coef),
            "std_error": float(se),
            "t_stat": float(tstat),
            "p_value": float(pval),
            "ci_lower": float(ci[0]),
            "ci_upper": float(ci[1]),
            "n_obs": int(len(df_reg)),
            "r_squared": float(model.rsquared),
            "coefficient_vector_json": json.dumps(coef_vector)
        }
    except Exception as e:
        print(f"  Error: {e}")
        return None


def run_logit_regression(df, outcome, treatment, controls):
    """Run logit regression for binary outcome."""
    all_vars = [outcome, treatment] + controls
    df_reg = df[all_vars].dropna()

    if len(df_reg) < 10:
        return None

    y = df_reg[outcome]
    X = df_reg[[treatment] + controls]
    X = sm.add_constant(X)

    try:
        model = Logit(y, X).fit(disp=0, cov_type='HC1')

        treat_idx = 1
        coef = model.params.iloc[treat_idx]
        se = model.bse.iloc[treat_idx]
        tstat = model.tvalues.iloc[treat_idx]
        pval = model.pvalues.iloc[treat_idx]
        ci = model.conf_int().iloc[treat_idx]

        # Marginal effect at means
        mfx = model.get_margeff(at='mean')
        mfx_coef = mfx.margeff[0]
        mfx_se = mfx.margeff_se[0]

        coef_vector = {
            "treatment": {
                "var": treatment,
                "coef": float(coef),
                "se": float(se),
                "pval": float(pval),
                "marginal_effect": float(mfx_coef),
                "marginal_effect_se": float(mfx_se)
            },
            "controls": [],
            "diagnostics": {"model": "logit"}
        }

        return {
            "coefficient": float(mfx_coef),  # Report marginal effect for comparability
            "std_error": float(mfx_se),
            "t_stat": float(mfx_coef / mfx_se),
            "p_value": float(pval),
            "ci_lower": float(mfx_coef - 1.96 * mfx_se),
            "ci_upper": float(mfx_coef + 1.96 * mfx_se),
            "n_obs": int(len(df_reg)),
            "r_squared": float(model.prsquared),
            "coefficient_vector_json": json.dumps(coef_vector)
        }
    except Exception as e:
        print(f"  Logit error: {e}")
        return None


def run_probit_regression(df, outcome, treatment, controls):
    """Run probit regression for binary outcome."""
    all_vars = [outcome, treatment] + controls
    df_reg = df[all_vars].dropna()

    if len(df_reg) < 10:
        return None

    y = df_reg[outcome]
    X = df_reg[[treatment] + controls]
    X = sm.add_constant(X)

    try:
        model = Probit(y, X).fit(disp=0, cov_type='HC1')

        treat_idx = 1
        coef = model.params.iloc[treat_idx]
        se = model.bse.iloc[treat_idx]
        pval = model.pvalues.iloc[treat_idx]

        # Marginal effect at means
        mfx = model.get_margeff(at='mean')
        mfx_coef = mfx.margeff[0]
        mfx_se = mfx.margeff_se[0]

        coef_vector = {
            "treatment": {
                "var": treatment,
                "coef": float(coef),
                "se": float(se),
                "pval": float(pval),
                "marginal_effect": float(mfx_coef),
                "marginal_effect_se": float(mfx_se)
            },
            "diagnostics": {"model": "probit"}
        }

        return {
            "coefficient": float(mfx_coef),
            "std_error": float(mfx_se),
            "t_stat": float(mfx_coef / mfx_se),
            "p_value": float(pval),
            "ci_lower": float(mfx_coef - 1.96 * mfx_se),
            "ci_upper": float(mfx_coef + 1.96 * mfx_se),
            "n_obs": int(len(df_reg)),
            "r_squared": float(model.prsquared),
            "coefficient_vector_json": json.dumps(coef_vector)
        }
    except Exception as e:
        print(f"  Probit error: {e}")
        return None


# =============================================================================
# Specification Search
# =============================================================================

def create_result_row(spec_id, spec_tree_path, outcome_var, treatment_var,
                      reg_result, sample_desc, controls_desc, cluster_var, model_type):
    """Create a result row dictionary."""
    if reg_result is None:
        return None

    return {
        "paper_id": PAPER_ID,
        "journal": JOURNAL,
        "paper_title": PAPER_TITLE,
        "spec_id": spec_id,
        "spec_tree_path": spec_tree_path,
        "outcome_var": outcome_var,
        "treatment_var": treatment_var,
        "coefficient": reg_result["coefficient"],
        "std_error": reg_result["std_error"],
        "t_stat": reg_result["t_stat"],
        "p_value": reg_result["p_value"],
        "ci_lower": reg_result["ci_lower"],
        "ci_upper": reg_result["ci_upper"],
        "n_obs": reg_result["n_obs"],
        "r_squared": reg_result["r_squared"],
        "coefficient_vector_json": reg_result["coefficient_vector_json"],
        "sample_desc": sample_desc,
        "fixed_effects": "None",
        "controls_desc": controls_desc,
        "cluster_var": cluster_var,
        "model_type": model_type,
        "estimation_script": SCRIPT_PATH
    }


def run_specification_search(df):
    """Run comprehensive specification search."""
    results = []

    # Define control sets
    # Full control set from the paper
    full_controls = [
        'unsubsidized', 'age_35_plus', 'high_income',
        'cat_ass_1', 'cat_ass_2', 'cat_ass_3', 'cat_ass_miss',
        'cat_lib_1', 'cat_lib_2', 'cat_lib_3', 'cat_lib_miss',
        'tri_ed_1', 'tri_ed_2',
        'female', 'race_cau', 'married', 'own_bank', 'late_cohort',
        'ins_heal', 'hh_adult', 'has_children',
        'own_bus', 'own_prop', 'own_ira', 'src_welf', 'own_car',
        'own_scale', 'str_scale', 'gv_scale', 'gt_scale',
        'sat_heal_good', 'sat_fin_good', 'ci_scale'
    ]

    # Remove controls with too many missing values for cleaner analysis
    minimal_controls = ['female', 'race_cau', 'married', 'age_35_plus', 'high_income']

    demographic_controls = ['female', 'race_cau', 'married', 'age_35_plus', 'hh_adult', 'has_children']

    socioeconomic_controls = demographic_controls + ['high_income', 'own_bank', 'own_car', 'ins_heal']

    baseline_controls = socioeconomic_controls + [
        'unsubsidized', 'late_cohort', 'tri_ed_1', 'tri_ed_2',
        'own_bus', 'own_prop', 'own_ira', 'src_welf'
    ]

    outcome = 'own_home_u42'
    treatment = 'treat'

    print("=" * 80)
    print("SPECIFICATION SEARCH: 116540-V2")
    print("Individual Development Accounts and Homeownership among Low-Income Renters")
    print("=" * 80)
    print()

    # =========================================================================
    # 1. BASELINE SPECIFICATION
    # =========================================================================
    print("1. BASELINE SPECIFICATION")
    print("-" * 40)

    reg = run_ols_regression(df, outcome, treatment, baseline_controls)
    row = create_result_row(
        spec_id="baseline",
        spec_tree_path="methods/cross_sectional_ols.md#baseline",
        outcome_var=outcome,
        treatment_var=treatment,
        reg_result=reg,
        sample_desc="Baseline renters with Wave 4 data",
        controls_desc="Full baseline controls",
        cluster_var="robust HC1",
        model_type="OLS"
    )
    if row:
        results.append(row)
        print(f"  Baseline: coef={row['coefficient']:.4f}, se={row['std_error']:.4f}, p={row['p_value']:.4f}, n={row['n_obs']}")

    # =========================================================================
    # 2. CONTROL VARIABLE PROGRESSION (Build-up)
    # =========================================================================
    print()
    print("2. CONTROL VARIABLE PROGRESSION")
    print("-" * 40)

    # 2.1 Bivariate (no controls)
    reg = run_ols_regression(df, outcome, treatment, [])
    row = create_result_row(
        spec_id="robust/build/bivariate",
        spec_tree_path="robustness/control_progression.md#bivariate",
        outcome_var=outcome,
        treatment_var=treatment,
        reg_result=reg,
        sample_desc="Baseline renters with Wave 4 data",
        controls_desc="No controls",
        cluster_var="robust HC1",
        model_type="OLS"
    )
    if row:
        results.append(row)
        print(f"  Bivariate: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}")

    # 2.2 Demographics only
    reg = run_ols_regression(df, outcome, treatment, demographic_controls)
    row = create_result_row(
        spec_id="robust/build/demographics",
        spec_tree_path="robustness/control_progression.md#demographics",
        outcome_var=outcome,
        treatment_var=treatment,
        reg_result=reg,
        sample_desc="Baseline renters with Wave 4 data",
        controls_desc="Demographic controls only",
        cluster_var="robust HC1",
        model_type="OLS"
    )
    if row:
        results.append(row)
        print(f"  Demographics: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}")

    # 2.3 Socioeconomic
    reg = run_ols_regression(df, outcome, treatment, socioeconomic_controls)
    row = create_result_row(
        spec_id="robust/build/socioeconomic",
        spec_tree_path="robustness/control_progression.md#socioeconomic",
        outcome_var=outcome,
        treatment_var=treatment,
        reg_result=reg,
        sample_desc="Baseline renters with Wave 4 data",
        controls_desc="Demographic + socioeconomic controls",
        cluster_var="robust HC1",
        model_type="OLS"
    )
    if row:
        results.append(row)
        print(f"  Socioeconomic: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}")

    # 2.4 Full controls
    reg = run_ols_regression(df, outcome, treatment, full_controls)
    row = create_result_row(
        spec_id="robust/build/full",
        spec_tree_path="robustness/control_progression.md#full",
        outcome_var=outcome,
        treatment_var=treatment,
        reg_result=reg,
        sample_desc="Baseline renters with Wave 4 data",
        controls_desc="Full control set including scales",
        cluster_var="robust HC1",
        model_type="OLS"
    )
    if row:
        results.append(row)
        print(f"  Full: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}")

    # =========================================================================
    # 3. LEAVE-ONE-OUT ROBUSTNESS
    # =========================================================================
    print()
    print("3. LEAVE-ONE-OUT ROBUSTNESS")
    print("-" * 40)

    for ctrl in baseline_controls:
        remaining_controls = [c for c in baseline_controls if c != ctrl]
        reg = run_ols_regression(df, outcome, treatment, remaining_controls)
        row = create_result_row(
            spec_id=f"robust/loo/drop_{ctrl}",
            spec_tree_path="robustness/leave_one_out.md",
            outcome_var=outcome,
            treatment_var=treatment,
            reg_result=reg,
            sample_desc="Baseline renters with Wave 4 data",
            controls_desc=f"Baseline controls minus {ctrl}",
            cluster_var="robust HC1",
            model_type="OLS"
        )
        if row:
            results.append(row)
            print(f"  Drop {ctrl}: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}")

    # =========================================================================
    # 4. SAMPLE RESTRICTIONS
    # =========================================================================
    print()
    print("4. SAMPLE RESTRICTIONS")
    print("-" * 40)

    # 4.1 Unsubsidized renters only
    df_unsub = df[df['unsubsidized'] == 1].copy()
    reg = run_ols_regression(df_unsub, outcome, treatment,
                             [c for c in baseline_controls if c != 'unsubsidized'])
    row = create_result_row(
        spec_id="robust/sample/unsubsidized_only",
        spec_tree_path="robustness/sample_restrictions.md",
        outcome_var=outcome,
        treatment_var=treatment,
        reg_result=reg,
        sample_desc="Unsubsidized renters only",
        controls_desc="Baseline controls (minus unsubsidized)",
        cluster_var="robust HC1",
        model_type="OLS"
    )
    if row:
        results.append(row)
        print(f"  Unsubsidized only: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}, n={row['n_obs']}")

    # 4.2 Female only
    df_female = df[df['female'] == 1].copy()
    reg = run_ols_regression(df_female, outcome, treatment,
                             [c for c in baseline_controls if c != 'female'])
    row = create_result_row(
        spec_id="robust/sample/female_only",
        spec_tree_path="robustness/sample_restrictions.md#female_only",
        outcome_var=outcome,
        treatment_var=treatment,
        reg_result=reg,
        sample_desc="Female respondents only",
        controls_desc="Baseline controls (minus female)",
        cluster_var="robust HC1",
        model_type="OLS"
    )
    if row:
        results.append(row)
        print(f"  Female only: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}, n={row['n_obs']}")

    # 4.3 Male only
    df_male = df[df['female'] == 0].copy()
    reg = run_ols_regression(df_male, outcome, treatment,
                             [c for c in baseline_controls if c != 'female'])
    row = create_result_row(
        spec_id="robust/sample/male_only",
        spec_tree_path="robustness/sample_restrictions.md#male_only",
        outcome_var=outcome,
        treatment_var=treatment,
        reg_result=reg,
        sample_desc="Male respondents only",
        controls_desc="Baseline controls (minus female)",
        cluster_var="robust HC1",
        model_type="OLS"
    )
    if row:
        results.append(row)
        print(f"  Male only: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}, n={row['n_obs']}")

    # 4.4 Married only
    df_married = df[df['married'] == 1].copy()
    reg = run_ols_regression(df_married, outcome, treatment,
                             [c for c in baseline_controls if c != 'married'])
    row = create_result_row(
        spec_id="robust/sample/married_only",
        spec_tree_path="robustness/sample_restrictions.md#married_only",
        outcome_var=outcome,
        treatment_var=treatment,
        reg_result=reg,
        sample_desc="Married respondents only",
        controls_desc="Baseline controls (minus married)",
        cluster_var="robust HC1",
        model_type="OLS"
    )
    if row:
        results.append(row)
        print(f"  Married only: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}, n={row['n_obs']}")

    # 4.5 Unmarried only
    df_unmarried = df[df['married'] == 0].copy()
    reg = run_ols_regression(df_unmarried, outcome, treatment,
                             [c for c in baseline_controls if c != 'married'])
    row = create_result_row(
        spec_id="robust/sample/unmarried_only",
        spec_tree_path="robustness/sample_restrictions.md#unmarried_only",
        outcome_var=outcome,
        treatment_var=treatment,
        reg_result=reg,
        sample_desc="Unmarried respondents only",
        controls_desc="Baseline controls (minus married)",
        cluster_var="robust HC1",
        model_type="OLS"
    )
    if row:
        results.append(row)
        print(f"  Unmarried only: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}, n={row['n_obs']}")

    # 4.6 Young (under 35)
    df_young = df[df['age_35_plus'] == 0].copy()
    reg = run_ols_regression(df_young, outcome, treatment,
                             [c for c in baseline_controls if c != 'age_35_plus'])
    row = create_result_row(
        spec_id="robust/sample/young",
        spec_tree_path="robustness/sample_restrictions.md#young",
        outcome_var=outcome,
        treatment_var=treatment,
        reg_result=reg,
        sample_desc="Respondents under 35",
        controls_desc="Baseline controls (minus age_35_plus)",
        cluster_var="robust HC1",
        model_type="OLS"
    )
    if row:
        results.append(row)
        print(f"  Young (<35): coef={row['coefficient']:.4f}, p={row['p_value']:.4f}, n={row['n_obs']}")

    # 4.7 Old (35+)
    df_old = df[df['age_35_plus'] == 1].copy()
    reg = run_ols_regression(df_old, outcome, treatment,
                             [c for c in baseline_controls if c != 'age_35_plus'])
    row = create_result_row(
        spec_id="robust/sample/old",
        spec_tree_path="robustness/sample_restrictions.md#old",
        outcome_var=outcome,
        treatment_var=treatment,
        reg_result=reg,
        sample_desc="Respondents 35 and older",
        controls_desc="Baseline controls (minus age_35_plus)",
        cluster_var="robust HC1",
        model_type="OLS"
    )
    if row:
        results.append(row)
        print(f"  Old (35+): coef={row['coefficient']:.4f}, p={row['p_value']:.4f}, n={row['n_obs']}")

    # 4.8 High income
    df_hi = df[df['high_income'] == 1].copy()
    reg = run_ols_regression(df_hi, outcome, treatment,
                             [c for c in baseline_controls if c != 'high_income'])
    row = create_result_row(
        spec_id="robust/sample/high_income",
        spec_tree_path="robustness/sample_restrictions.md#high_income",
        outcome_var=outcome,
        treatment_var=treatment,
        reg_result=reg,
        sample_desc="High income respondents (above median)",
        controls_desc="Baseline controls (minus high_income)",
        cluster_var="robust HC1",
        model_type="OLS"
    )
    if row:
        results.append(row)
        print(f"  High income: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}, n={row['n_obs']}")

    # 4.9 Low income
    df_lo = df[df['high_income'] == 0].copy()
    reg = run_ols_regression(df_lo, outcome, treatment,
                             [c for c in baseline_controls if c != 'high_income'])
    row = create_result_row(
        spec_id="robust/sample/low_income",
        spec_tree_path="robustness/sample_restrictions.md#low_income",
        outcome_var=outcome,
        treatment_var=treatment,
        reg_result=reg,
        sample_desc="Low income respondents (below median)",
        controls_desc="Baseline controls (minus high_income)",
        cluster_var="robust HC1",
        model_type="OLS"
    )
    if row:
        results.append(row)
        print(f"  Low income: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}, n={row['n_obs']}")

    # 4.10 White only
    df_white = df[df['race_cau'] == 1].copy()
    reg = run_ols_regression(df_white, outcome, treatment,
                             [c for c in baseline_controls if c != 'race_cau'])
    row = create_result_row(
        spec_id="robust/sample/white_only",
        spec_tree_path="robustness/sample_restrictions.md#white_only",
        outcome_var=outcome,
        treatment_var=treatment,
        reg_result=reg,
        sample_desc="White respondents only",
        controls_desc="Baseline controls (minus race_cau)",
        cluster_var="robust HC1",
        model_type="OLS"
    )
    if row:
        results.append(row)
        print(f"  White only: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}, n={row['n_obs']}")

    # 4.11 Non-white only
    df_nonwhite = df[df['race_cau'] == 0].copy()
    reg = run_ols_regression(df_nonwhite, outcome, treatment,
                             [c for c in baseline_controls if c != 'race_cau'])
    row = create_result_row(
        spec_id="robust/sample/nonwhite_only",
        spec_tree_path="robustness/sample_restrictions.md#nonwhite_only",
        outcome_var=outcome,
        treatment_var=treatment,
        reg_result=reg,
        sample_desc="Non-white respondents only",
        controls_desc="Baseline controls (minus race_cau)",
        cluster_var="robust HC1",
        model_type="OLS"
    )
    if row:
        results.append(row)
        print(f"  Non-white only: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}, n={row['n_obs']}")

    # 4.12 With children
    df_children = df[df['has_children'] == 1].copy()
    reg = run_ols_regression(df_children, outcome, treatment,
                             [c for c in baseline_controls if c != 'has_children'])
    row = create_result_row(
        spec_id="robust/sample/with_children",
        spec_tree_path="robustness/sample_restrictions.md#with_children",
        outcome_var=outcome,
        treatment_var=treatment,
        reg_result=reg,
        sample_desc="Respondents with children",
        controls_desc="Baseline controls (minus has_children)",
        cluster_var="robust HC1",
        model_type="OLS"
    )
    if row:
        results.append(row)
        print(f"  With children: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}, n={row['n_obs']}")

    # 4.13 Without children
    df_nochildren = df[df['has_children'] == 0].copy()
    reg = run_ols_regression(df_nochildren, outcome, treatment,
                             [c for c in baseline_controls if c != 'has_children'])
    row = create_result_row(
        spec_id="robust/sample/without_children",
        spec_tree_path="robustness/sample_restrictions.md#without_children",
        outcome_var=outcome,
        treatment_var=treatment,
        reg_result=reg,
        sample_desc="Respondents without children",
        controls_desc="Baseline controls (minus has_children)",
        cluster_var="robust HC1",
        model_type="OLS"
    )
    if row:
        results.append(row)
        print(f"  Without children: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}, n={row['n_obs']}")

    # 4.14 With bank account
    df_bank = df[df['own_bank'] == 1].copy()
    reg = run_ols_regression(df_bank, outcome, treatment,
                             [c for c in baseline_controls if c != 'own_bank'])
    row = create_result_row(
        spec_id="robust/sample/with_bank",
        spec_tree_path="robustness/sample_restrictions.md#with_bank",
        outcome_var=outcome,
        treatment_var=treatment,
        reg_result=reg,
        sample_desc="Respondents with bank account",
        controls_desc="Baseline controls (minus own_bank)",
        cluster_var="robust HC1",
        model_type="OLS"
    )
    if row:
        results.append(row)
        print(f"  With bank account: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}, n={row['n_obs']}")

    # 4.15 Less than HS education
    df_lths = df[df['less_than_hs'] == 1].copy()
    reg = run_ols_regression(df_lths, outcome, treatment,
                             [c for c in baseline_controls if c not in ['tri_ed_1', 'tri_ed_2']])
    row = create_result_row(
        spec_id="robust/sample/less_than_hs",
        spec_tree_path="robustness/sample_restrictions.md#less_than_hs",
        outcome_var=outcome,
        treatment_var=treatment,
        reg_result=reg,
        sample_desc="Respondents with less than HS education",
        controls_desc="Baseline controls (minus education)",
        cluster_var="robust HC1",
        model_type="OLS"
    )
    if row:
        results.append(row)
        print(f"  Less than HS: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}, n={row['n_obs']}")

    # 4.16 HS or more
    df_hsplus = df[(df['hs_grad'] == 1) | (df['some_college'] == 1) | (df['college_grad'] == 1)].copy()
    reg = run_ols_regression(df_hsplus, outcome, treatment, baseline_controls)
    row = create_result_row(
        spec_id="robust/sample/hs_or_more",
        spec_tree_path="robustness/sample_restrictions.md#hs_or_more",
        outcome_var=outcome,
        treatment_var=treatment,
        reg_result=reg,
        sample_desc="Respondents with HS or more education",
        controls_desc="Baseline controls",
        cluster_var="robust HC1",
        model_type="OLS"
    )
    if row:
        results.append(row)
        print(f"  HS or more: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}, n={row['n_obs']}")

    # =========================================================================
    # 5. ESTIMATION METHOD VARIATIONS
    # =========================================================================
    print()
    print("5. ESTIMATION METHOD VARIATIONS")
    print("-" * 40)

    # 5.1 Logit model
    reg = run_logit_regression(df, outcome, treatment, baseline_controls)
    row = create_result_row(
        spec_id="robust/method/logit",
        spec_tree_path="methods/discrete_choice.md#logit",
        outcome_var=outcome,
        treatment_var=treatment,
        reg_result=reg,
        sample_desc="Baseline renters with Wave 4 data",
        controls_desc="Baseline controls",
        cluster_var="robust HC1",
        model_type="Logit (marginal effect)"
    )
    if row:
        results.append(row)
        print(f"  Logit: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}")

    # 5.2 Probit model
    reg = run_probit_regression(df, outcome, treatment, baseline_controls)
    row = create_result_row(
        spec_id="robust/method/probit",
        spec_tree_path="methods/discrete_choice.md#probit",
        outcome_var=outcome,
        treatment_var=treatment,
        reg_result=reg,
        sample_desc="Baseline renters with Wave 4 data",
        controls_desc="Baseline controls",
        cluster_var="robust HC1",
        model_type="Probit (marginal effect)"
    )
    if row:
        results.append(row)
        print(f"  Probit: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}")

    # 5.3 LPM without controls (just treatment)
    reg = run_ols_regression(df, outcome, treatment, [])
    row = create_result_row(
        spec_id="robust/method/lpm_no_controls",
        spec_tree_path="methods/cross_sectional_ols.md#bivariate",
        outcome_var=outcome,
        treatment_var=treatment,
        reg_result=reg,
        sample_desc="Baseline renters with Wave 4 data",
        controls_desc="No controls",
        cluster_var="robust HC1",
        model_type="LPM"
    )
    if row:
        results.append(row)
        print(f"  LPM no controls: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}")

    # =========================================================================
    # 6. HETEROGENEITY ANALYSIS (Interactions)
    # =========================================================================
    print()
    print("6. HETEROGENEITY ANALYSIS (Interactions)")
    print("-" * 40)

    interaction_vars = [
        ('high_income', 'High income'),
        ('female', 'Female'),
        ('race_cau', 'White'),
        ('married', 'Married'),
        ('tri_ed_1', 'Some college'),
        ('has_children', 'Has children'),
        ('late_cohort', 'Late cohort'),
        ('own_bank', 'Has bank account'),
        ('src_welf', 'Welfare receipt'),
        ('own_car', 'Car ownership'),
        ('ins_heal', 'Health insurance')
    ]

    for var, label in interaction_vars:
        # Create interaction term
        df_int = df.copy()
        df_int['treat_x_' + var] = df_int['treat'] * df_int[var]

        # Controls without the interacting variable
        int_controls = [c for c in baseline_controls if c != var]
        int_controls.append(var)
        int_controls.append('treat_x_' + var)

        reg = run_ols_regression(df_int, outcome, treatment, int_controls)
        if reg:
            # Get interaction coefficient
            all_vars = [outcome, treatment] + int_controls
            df_reg = df_int[all_vars].dropna()
            y = df_reg[outcome]
            X = df_reg[[treatment] + int_controls]
            X = sm.add_constant(X)
            model = OLS(y, X).fit(cov_type='HC1')

            int_idx = list(model.params.index).index('treat_x_' + var)
            int_coef = model.params.iloc[int_idx]
            int_se = model.bse.iloc[int_idx]
            int_pval = model.pvalues.iloc[int_idx]

            row = {
                "paper_id": PAPER_ID,
                "journal": JOURNAL,
                "paper_title": PAPER_TITLE,
                "spec_id": f"robust/het/interaction_{var}",
                "spec_tree_path": "robustness/heterogeneity.md#interactions",
                "outcome_var": outcome,
                "treatment_var": f"treat x {var}",
                "coefficient": float(int_coef),
                "std_error": float(int_se),
                "t_stat": float(int_coef / int_se),
                "p_value": float(int_pval),
                "ci_lower": float(int_coef - 1.96 * int_se),
                "ci_upper": float(int_coef + 1.96 * int_se),
                "n_obs": reg["n_obs"],
                "r_squared": reg["r_squared"],
                "coefficient_vector_json": reg["coefficient_vector_json"],
                "sample_desc": "Baseline renters with Wave 4 data",
                "fixed_effects": "None",
                "controls_desc": f"Baseline + treat x {label}",
                "cluster_var": "robust HC1",
                "model_type": "OLS",
                "estimation_script": SCRIPT_PATH
            }
            results.append(row)
            print(f"  Treat x {label}: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}")

    # =========================================================================
    # 7. INFERENCE VARIATIONS
    # =========================================================================
    print()
    print("7. INFERENCE VARIATIONS")
    print("-" * 40)

    # 7.1 Heteroskedasticity-robust (HC0)
    all_vars = [outcome, treatment] + baseline_controls
    df_reg = df[all_vars].dropna()
    y = df_reg[outcome]
    X = df_reg[[treatment] + baseline_controls]
    X = sm.add_constant(X)

    for hc_type in ['HC0', 'HC2', 'HC3']:
        try:
            model = OLS(y, X).fit(cov_type=hc_type)
            coef = model.params.iloc[1]
            se = model.bse.iloc[1]
            pval = model.pvalues.iloc[1]
            ci = model.conf_int().iloc[1]

            row = {
                "paper_id": PAPER_ID,
                "journal": JOURNAL,
                "paper_title": PAPER_TITLE,
                "spec_id": f"robust/se/{hc_type.lower()}",
                "spec_tree_path": "robustness/clustering_variations.md#hc",
                "outcome_var": outcome,
                "treatment_var": treatment,
                "coefficient": float(coef),
                "std_error": float(se),
                "t_stat": float(coef / se),
                "p_value": float(pval),
                "ci_lower": float(ci[0]),
                "ci_upper": float(ci[1]),
                "n_obs": int(len(df_reg)),
                "r_squared": float(model.rsquared),
                "coefficient_vector_json": "{}",
                "sample_desc": "Baseline renters with Wave 4 data",
                "fixed_effects": "None",
                "controls_desc": "Baseline controls",
                "cluster_var": hc_type,
                "model_type": "OLS",
                "estimation_script": SCRIPT_PATH
            }
            results.append(row)
            print(f"  {hc_type}: se={row['std_error']:.4f}, p={row['p_value']:.4f}")
        except Exception as e:
            print(f"  {hc_type} error: {e}")

    # 7.2 Cluster by cohort
    if 'cohort_num' in df.columns:
        df_clust = df.copy()
        df_clust['cohort_cluster'] = df_clust['cohort_num'].astype(int)
        all_vars = [outcome, treatment] + baseline_controls + ['cohort_cluster']
        df_reg = df_clust[all_vars].dropna()
        y = df_reg[outcome]
        X = df_reg[[treatment] + baseline_controls]
        X = sm.add_constant(X)

        try:
            model = OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df_reg['cohort_cluster']})
            coef = model.params.iloc[1]
            se = model.bse.iloc[1]
            pval = model.pvalues.iloc[1]
            ci = model.conf_int().iloc[1]

            row = {
                "paper_id": PAPER_ID,
                "journal": JOURNAL,
                "paper_title": PAPER_TITLE,
                "spec_id": "robust/cluster/cohort",
                "spec_tree_path": "robustness/clustering_variations.md#cohort",
                "outcome_var": outcome,
                "treatment_var": treatment,
                "coefficient": float(coef),
                "std_error": float(se),
                "t_stat": float(coef / se),
                "p_value": float(pval),
                "ci_lower": float(ci[0]),
                "ci_upper": float(ci[1]),
                "n_obs": int(len(df_reg)),
                "r_squared": float(model.rsquared),
                "coefficient_vector_json": "{}",
                "sample_desc": "Baseline renters with Wave 4 data",
                "fixed_effects": "None",
                "controls_desc": "Baseline controls",
                "cluster_var": "cohort",
                "model_type": "OLS",
                "estimation_script": SCRIPT_PATH
            }
            results.append(row)
            print(f"  Cluster by cohort: se={row['std_error']:.4f}, p={row['p_value']:.4f}")
        except Exception as e:
            print(f"  Cluster cohort error: {e}")

    # =========================================================================
    # 8. FUNCTIONAL FORM VARIATIONS
    # =========================================================================
    print()
    print("8. FUNCTIONAL FORM VARIATIONS")
    print("-" * 40)

    # Note: Since outcome is binary, functional form variations are limited
    # We'll run specifications with different control transformations

    # 8.1 Add age squared
    df_sq = df.copy()
    df_sq['age_sq'] = df_sq['age'] ** 2
    controls_sq = baseline_controls + ['age', 'age_sq']
    controls_sq = [c for c in controls_sq if c != 'age_35_plus']

    reg = run_ols_regression(df_sq, outcome, treatment, controls_sq)
    row = create_result_row(
        spec_id="robust/form/age_quadratic",
        spec_tree_path="robustness/functional_form.md#quadratic",
        outcome_var=outcome,
        treatment_var=treatment,
        reg_result=reg,
        sample_desc="Baseline renters with Wave 4 data",
        controls_desc="Baseline + age + age^2",
        cluster_var="robust HC1",
        model_type="OLS"
    )
    if row:
        results.append(row)
        print(f"  Age quadratic: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}")

    # 8.2 Log income
    df_log = df.copy()
    df_log['log_income'] = np.log(df_log['inc_tot'] + 1)
    controls_log = [c for c in baseline_controls if c != 'high_income'] + ['log_income']

    reg = run_ols_regression(df_log, outcome, treatment, controls_log)
    row = create_result_row(
        spec_id="robust/form/log_income",
        spec_tree_path="robustness/functional_form.md#log",
        outcome_var=outcome,
        treatment_var=treatment,
        reg_result=reg,
        sample_desc="Baseline renters with Wave 4 data",
        controls_desc="Baseline with log(income)",
        cluster_var="robust HC1",
        model_type="OLS"
    )
    if row:
        results.append(row)
        print(f"  Log income: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}")

    # 8.3 Continuous income instead of binary
    controls_cont_inc = [c for c in baseline_controls if c != 'high_income']
    df_cont = df.copy()
    df_cont['inc_scaled'] = df_cont['inc_tot'] / 1000  # Scale to thousands
    controls_cont_inc.append('inc_scaled')

    reg = run_ols_regression(df_cont, outcome, treatment, controls_cont_inc)
    row = create_result_row(
        spec_id="robust/form/continuous_income",
        spec_tree_path="robustness/functional_form.md#continuous",
        outcome_var=outcome,
        treatment_var=treatment,
        reg_result=reg,
        sample_desc="Baseline renters with Wave 4 data",
        controls_desc="Baseline with continuous income",
        cluster_var="robust HC1",
        model_type="OLS"
    )
    if row:
        results.append(row)
        print(f"  Continuous income: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}")

    # 8.4 Income quartiles
    df_qrt = df.copy()
    df_qrt['inc_q2'] = ((df_qrt['inc_tot'] >= df_qrt['inc_tot'].quantile(0.25)) &
                        (df_qrt['inc_tot'] < df_qrt['inc_tot'].quantile(0.50))).astype(float)
    df_qrt['inc_q3'] = ((df_qrt['inc_tot'] >= df_qrt['inc_tot'].quantile(0.50)) &
                        (df_qrt['inc_tot'] < df_qrt['inc_tot'].quantile(0.75))).astype(float)
    df_qrt['inc_q4'] = (df_qrt['inc_tot'] >= df_qrt['inc_tot'].quantile(0.75)).astype(float)

    controls_qrt = [c for c in baseline_controls if c != 'high_income'] + ['inc_q2', 'inc_q3', 'inc_q4']

    reg = run_ols_regression(df_qrt, outcome, treatment, controls_qrt)
    row = create_result_row(
        spec_id="robust/form/income_quartiles",
        spec_tree_path="robustness/functional_form.md#categorical",
        outcome_var=outcome,
        treatment_var=treatment,
        reg_result=reg,
        sample_desc="Baseline renters with Wave 4 data",
        controls_desc="Baseline with income quartiles",
        cluster_var="robust HC1",
        model_type="OLS"
    )
    if row:
        results.append(row)
        print(f"  Income quartiles: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}")

    # =========================================================================
    # 9. PLACEBO TESTS
    # =========================================================================
    print()
    print("9. PLACEBO TESTS")
    print("-" * 40)

    # 9.1 Placebo outcome: owns business at wave 4 (should not be affected)
    # Note: We don't have Wave 4 business ownership, so we'll use baseline
    # This tests whether treatment predicts baseline characteristics (it shouldn't)

    placebo_outcomes = [
        ('own_bus', 'Business ownership (baseline)'),
        ('own_car', 'Car ownership (baseline)'),
        ('own_ira', 'IRA ownership (baseline)'),
        ('sat_heal_good', 'Health satisfaction (baseline)'),
        ('sat_fin_good', 'Financial satisfaction (baseline)')
    ]

    for poutcome, plabel in placebo_outcomes:
        # Remove the placebo outcome from controls
        placebo_controls = [c for c in baseline_controls if c != poutcome]

        reg = run_ols_regression(df, poutcome, treatment, placebo_controls)
        row = create_result_row(
            spec_id=f"robust/placebo/{poutcome}",
            spec_tree_path="robustness/placebo_tests.md",
            outcome_var=poutcome,
            treatment_var=treatment,
            reg_result=reg,
            sample_desc="Baseline renters with Wave 4 data",
            controls_desc=f"Baseline controls (minus {poutcome})",
            cluster_var="robust HC1",
            model_type="OLS"
        )
        if row:
            results.append(row)
            print(f"  Placebo {plabel}: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}")

    # =========================================================================
    # 10. ADDITIONAL SPECIFICATIONS TO REACH 50+
    # =========================================================================
    print()
    print("10. ADDITIONAL SPECIFICATIONS")
    print("-" * 40)

    # 10.1 Education subgroups
    # College graduates
    df_college = df[df['college_grad'] == 1].copy()
    controls_no_ed = [c for c in baseline_controls if c not in ['tri_ed_1', 'tri_ed_2']]
    reg = run_ols_regression(df_college, outcome, treatment, controls_no_ed)
    row = create_result_row(
        spec_id="robust/sample/college_grad",
        spec_tree_path="robustness/sample_restrictions.md#college_grad",
        outcome_var=outcome,
        treatment_var=treatment,
        reg_result=reg,
        sample_desc="College graduates only",
        controls_desc="Baseline controls (minus education)",
        cluster_var="robust HC1",
        model_type="OLS"
    )
    if row:
        results.append(row)
        print(f"  College grads: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}, n={row['n_obs']}")

    # Some college
    df_somecoll = df[df['some_college'] == 1].copy()
    reg = run_ols_regression(df_somecoll, outcome, treatment, controls_no_ed)
    row = create_result_row(
        spec_id="robust/sample/some_college",
        spec_tree_path="robustness/sample_restrictions.md#some_college",
        outcome_var=outcome,
        treatment_var=treatment,
        reg_result=reg,
        sample_desc="Some college only",
        controls_desc="Baseline controls (minus education)",
        cluster_var="robust HC1",
        model_type="OLS"
    )
    if row:
        results.append(row)
        print(f"  Some college: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}, n={row['n_obs']}")

    # 10.2 Cohort subgroups
    # Early cohorts
    df_early = df[df['late_cohort'] == 0].copy()
    controls_no_cohort = [c for c in baseline_controls if c != 'late_cohort']
    reg = run_ols_regression(df_early, outcome, treatment, controls_no_cohort)
    row = create_result_row(
        spec_id="robust/sample/early_cohort",
        spec_tree_path="robustness/sample_restrictions.md#early_cohort",
        outcome_var=outcome,
        treatment_var=treatment,
        reg_result=reg,
        sample_desc="Early survey cohorts",
        controls_desc="Baseline controls (minus cohort)",
        cluster_var="robust HC1",
        model_type="OLS"
    )
    if row:
        results.append(row)
        print(f"  Early cohort: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}, n={row['n_obs']}")

    # Late cohorts
    df_late = df[df['late_cohort'] == 1].copy()
    reg = run_ols_regression(df_late, outcome, treatment, controls_no_cohort)
    row = create_result_row(
        spec_id="robust/sample/late_cohort",
        spec_tree_path="robustness/sample_restrictions.md#late_cohort",
        outcome_var=outcome,
        treatment_var=treatment,
        reg_result=reg,
        sample_desc="Late survey cohorts",
        controls_desc="Baseline controls (minus cohort)",
        cluster_var="robust HC1",
        model_type="OLS"
    )
    if row:
        results.append(row)
        print(f"  Late cohort: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}, n={row['n_obs']}")

    # 10.3 Welfare receipt subgroups
    df_welfare = df[df['src_welf'] == 1].copy()
    controls_no_welf = [c for c in baseline_controls if c != 'src_welf']
    reg = run_ols_regression(df_welfare, outcome, treatment, controls_no_welf)
    row = create_result_row(
        spec_id="robust/sample/welfare_recipient",
        spec_tree_path="robustness/sample_restrictions.md#welfare",
        outcome_var=outcome,
        treatment_var=treatment,
        reg_result=reg,
        sample_desc="Welfare recipients only",
        controls_desc="Baseline controls (minus welfare)",
        cluster_var="robust HC1",
        model_type="OLS"
    )
    if row:
        results.append(row)
        print(f"  Welfare recipients: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}, n={row['n_obs']}")

    df_nowelfare = df[df['src_welf'] == 0].copy()
    reg = run_ols_regression(df_nowelfare, outcome, treatment, controls_no_welf)
    row = create_result_row(
        spec_id="robust/sample/non_welfare",
        spec_tree_path="robustness/sample_restrictions.md#non_welfare",
        outcome_var=outcome,
        treatment_var=treatment,
        reg_result=reg,
        sample_desc="Non-welfare recipients only",
        controls_desc="Baseline controls (minus welfare)",
        cluster_var="robust HC1",
        model_type="OLS"
    )
    if row:
        results.append(row)
        print(f"  Non-welfare: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}, n={row['n_obs']}")

    # 10.4 Health insurance subgroups
    df_insured = df[df['ins_heal'] == 1].copy()
    controls_no_ins = [c for c in baseline_controls if c != 'ins_heal']
    reg = run_ols_regression(df_insured, outcome, treatment, controls_no_ins)
    row = create_result_row(
        spec_id="robust/sample/insured",
        spec_tree_path="robustness/sample_restrictions.md#insured",
        outcome_var=outcome,
        treatment_var=treatment,
        reg_result=reg,
        sample_desc="Health insured only",
        controls_desc="Baseline controls (minus insurance)",
        cluster_var="robust HC1",
        model_type="OLS"
    )
    if row:
        results.append(row)
        print(f"  Insured: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}, n={row['n_obs']}")

    df_uninsured = df[df['ins_heal'] == 0].copy()
    reg = run_ols_regression(df_uninsured, outcome, treatment, controls_no_ins)
    row = create_result_row(
        spec_id="robust/sample/uninsured",
        spec_tree_path="robustness/sample_restrictions.md#uninsured",
        outcome_var=outcome,
        treatment_var=treatment,
        reg_result=reg,
        sample_desc="Uninsured only",
        controls_desc="Baseline controls (minus insurance)",
        cluster_var="robust HC1",
        model_type="OLS"
    )
    if row:
        results.append(row)
        print(f"  Uninsured: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}, n={row['n_obs']}")

    # 10.5 Single covariate specifications
    print()
    print("Single covariate specifications:")
    single_covariates = ['female', 'race_cau', 'married', 'age_35_plus', 'high_income',
                         'own_bank', 'has_children', 'late_cohort', 'ins_heal']

    for cov in single_covariates:
        reg = run_ols_regression(df, outcome, treatment, [cov])
        row = create_result_row(
            spec_id=f"robust/single/{cov}",
            spec_tree_path="robustness/single_covariate.md",
            outcome_var=outcome,
            treatment_var=treatment,
            reg_result=reg,
            sample_desc="Baseline renters with Wave 4 data",
            controls_desc=f"Only {cov}",
            cluster_var="robust HC1",
            model_type="OLS"
        )
        if row:
            results.append(row)
            print(f"  Single {cov}: coef={row['coefficient']:.4f}, p={row['p_value']:.4f}")

    return results


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("Loading data...")
    df = load_and_prepare_data()
    print(f"Analysis sample size: {len(df)}")
    print()

    print("Running specification search...")
    results = run_specification_search(df)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Summary statistics
    print()
    print("=" * 80)
    print("SPECIFICATION SEARCH SUMMARY")
    print("=" * 80)
    print(f"Total specifications run: {len(results_df)}")
    print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
    print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
    print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")
    print()
    print(f"Coefficient range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")
    print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
    print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")

    # Save results
    output_csv = f"{OUTPUT_PATH}/specification_results.csv"
    results_df.to_csv(output_csv, index=False)
    print()
    print(f"Results saved to: {output_csv}")
