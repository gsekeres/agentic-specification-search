"""
Specification Search for Paper 223321-V1
=========================================
Title: Immigrant Age at Arrival and the Intergenerational Transmission of
       Ethnic Identification among Mexican Americans
Authors: Brian Duncan and Stephen J. Trejo

This script replicates the main analysis and runs a systematic specification search.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import json
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data/downloads/extracted/223321-V1"
DATA_FILE = DATA_DIR / "data/analysis/Census2000-ACS2001-2019_d1.dta"
OUTPUT_FILE = DATA_DIR / "specification_results.csv"

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load and prepare the analysis dataset."""
    print("Loading data...")
    df = pd.read_stata(DATA_FILE)

    # Convert categorical columns to numeric
    cat_cols = df.select_dtypes(include=['category']).columns
    for col in cat_cols:
        df[col] = df[col].cat.codes

    print(f"Data shape: {df.shape}")
    print(f"not_hisp mean: {df['not_hisp'].mean():.4f}")
    print(f"par_Arrived0_8 mean: {df['par_Arrived0_8'].mean():.4f}")

    return df

# ============================================================================
# CONTROL VARIABLE DEFINITIONS
# ============================================================================

def get_control_variables(df):
    """Define control variable sets."""

    # Year dummies (drop 2000 as reference)
    year_vars = [x for x in df.columns if x.startswith('year_') and x != 'year_2000']

    # Child age dummies (drop age 0 as reference)
    age_vars = [x for x in df.columns if x.startswith('child_age_') and x != 'child_age_0']

    # Parent age dummies (drop age 25 as reference)
    par_age_vars = [x for x in df.columns if x.startswith('par_age_')
                    and not x.startswith('par_age35') and x != 'par_age_25']

    # Baseline controls
    baseline_controls = ['female', 'par_female'] + year_vars + age_vars + par_age_vars

    # English ability dummies (drop "not at all" as reference)
    eng_dummies = ['par_English_notwell', 'par_English_well',
                   'par_English_verywell', 'par_English_only']

    # Education dummies (drop "no schooling" as reference)
    educ_dummies = ['par_EDUC_nursery_grade_4', 'par_EDUC_grade_5_8', 'par_EDUC_grade_9',
                    'par_EDUC_grade_10', 'par_EDUC_grade_11', 'par_EDUC_grade_12',
                    'par_EDUC_college_1', 'par_EDUC_college_2', 'par_EDUC_college_4',
                    'par_EDUC_college_5plus']

    # Family structure dummies
    family_dummies = ['p_present_momonly', 'p_present_dadonly']

    # Other parent ethnicity dummies (intermarriage)
    other_parent_dummies = ['Oth_Parrent_FBHisp', 'Oth_Parrent_FBnonHisp',
                            'Oth_Parrent_USHisp', 'Oth_Parrent_USnonHisp']

    # State dummies (drop California as reference)
    state_dummies = [col for col in df.columns if col.startswith('state_')
                     and col not in ['state_california', 'state_CA_TX_AZ_NM']]

    # Full controls
    full_controls = (baseline_controls + eng_dummies + educ_dummies +
                     family_dummies + other_parent_dummies + state_dummies)

    # Control groups for leave-one-out
    control_groups = {
        'year': year_vars,
        'child_age': age_vars,
        'par_age': par_age_vars,
        'child_sex': ['female'],
        'par_sex': ['par_female'],
        'par_english': eng_dummies,
        'par_education': educ_dummies,
        'family_structure': family_dummies,
        'intermarriage': other_parent_dummies,
        'state': state_dummies
    }

    return {
        'baseline': baseline_controls,
        'full': full_controls,
        'groups': control_groups,
        'eng': eng_dummies,
        'educ': educ_dummies,
        'family': family_dummies,
        'intermarriage': other_parent_dummies,
        'state': state_dummies
    }

# ============================================================================
# REGRESSION FUNCTIONS
# ============================================================================

def run_wls(formula, data, weights, spec_id, spec_tree_path, description, additional_info=None):
    """Run weighted least squares and extract results."""
    try:
        model = smf.wls(formula, data=data, weights=weights).fit(cov_type='HC1')

        if 'par_Arrived0_8' in model.params.index:
            coef = model.params['par_Arrived0_8']
            se = model.bse['par_Arrived0_8']
            pval = model.pvalues['par_Arrived0_8']
            ci_lower, ci_upper = model.conf_int().loc['par_Arrived0_8']
        else:
            coef, se, pval, ci_lower, ci_upper = np.nan, np.nan, np.nan, np.nan, np.nan

        result = {
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'description': description,
            'coef': float(coef) if not np.isnan(coef) else np.nan,
            'se': float(se) if not np.isnan(se) else np.nan,
            'pval': float(pval) if not np.isnan(pval) else np.nan,
            'ci_lower': float(ci_lower) if not np.isnan(ci_lower) else np.nan,
            'ci_upper': float(ci_upper) if not np.isnan(ci_upper) else np.nan,
            'n_obs': int(model.nobs),
            'r_squared': float(model.rsquared),
            'f_stat': float(model.fvalue) if hasattr(model, 'fvalue') else np.nan,
            'dep_var_mean': float(data['not_hisp'].mean())
        }

        if additional_info:
            result.update(additional_info)

        coef_vector = {'par_Arrived0_8': {'coef': float(coef), 'se': float(se), 'pval': float(pval)}}
        result['coefficient_vector_json'] = json.dumps(coef_vector)

        return result
    except Exception as e:
        return {
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'description': description,
            'coef': np.nan,
            'se': np.nan,
            'pval': np.nan,
            'error': str(e),
            'coefficient_vector_json': '{}'
        }

# ============================================================================
# SPECIFICATION SEARCH
# ============================================================================

def run_baseline_specifications(df, controls):
    """Run baseline specifications from the paper (Table 2)."""
    results = []

    # Column 1: No controls (bivariate)
    formula = "not_hisp ~ par_Arrived0_8"
    result = run_wls(formula, df, df['perwt'],
                     'baseline', 'methods/cross_sectional_ols.md#baseline',
                     'Table 2 Column 1: No controls (bivariate)')
    results.append(result)
    print(f"Baseline (Col 1): coef={result['coef']:.4f}, p={result['pval']:.4f}")

    # Column 2: Baseline controls
    formula = "not_hisp ~ par_Arrived0_8 + " + " + ".join(controls['baseline'])
    result = run_wls(formula, df, df['perwt'],
                     'ols/controls/baseline', 'methods/cross_sectional_ols.md#control-sets',
                     'Table 2 Column 2: Baseline controls')
    results.append(result)
    print(f"Baseline (Col 2): coef={result['coef']:.4f}, p={result['pval']:.4f}")

    # Column 3: Full controls
    formula = "not_hisp ~ par_Arrived0_8 + " + " + ".join(controls['full'])
    result = run_wls(formula, df, df['perwt'],
                     'ols/controls/full', 'methods/cross_sectional_ols.md#control-sets',
                     'Table 2 Column 3: Full controls')
    results.append(result)
    print(f"Baseline (Col 3): coef={result['coef']:.4f}, p={result['pval']:.4f}")

    return results

def run_se_variations(df, controls):
    """Run standard error variations."""
    results = []
    full_formula = "not_hisp ~ par_Arrived0_8 + " + " + ".join(controls['full'])

    # Classical SE
    model = smf.wls(full_formula, data=df, weights=df['perwt']).fit()
    result = {
        'spec_id': 'ols/se/classical',
        'spec_tree_path': 'methods/cross_sectional_ols.md#standard-errors',
        'description': 'Full controls with classical SE',
        'coef': float(model.params['par_Arrived0_8']),
        'se': float(model.bse['par_Arrived0_8']),
        'pval': float(model.pvalues['par_Arrived0_8']),
        'n_obs': int(model.nobs),
        'r_squared': float(model.rsquared),
        'coefficient_vector_json': '{}'
    }
    results.append(result)

    # HC2
    model = smf.wls(full_formula, data=df, weights=df['perwt']).fit(cov_type='HC2')
    result = {
        'spec_id': 'ols/se/hc2',
        'spec_tree_path': 'methods/cross_sectional_ols.md#standard-errors',
        'description': 'Full controls with HC2 SE',
        'coef': float(model.params['par_Arrived0_8']),
        'se': float(model.bse['par_Arrived0_8']),
        'pval': float(model.pvalues['par_Arrived0_8']),
        'n_obs': int(model.nobs),
        'r_squared': float(model.rsquared),
        'coefficient_vector_json': '{}'
    }
    results.append(result)

    # HC3
    model = smf.wls(full_formula, data=df, weights=df['perwt']).fit(cov_type='HC3')
    result = {
        'spec_id': 'ols/se/hc3',
        'spec_tree_path': 'methods/cross_sectional_ols.md#standard-errors',
        'description': 'Full controls with HC3 SE',
        'coef': float(model.params['par_Arrived0_8']),
        'se': float(model.bse['par_Arrived0_8']),
        'pval': float(model.pvalues['par_Arrived0_8']),
        'n_obs': int(model.nobs),
        'r_squared': float(model.rsquared),
        'coefficient_vector_json': '{}'
    }
    results.append(result)

    # State clustered
    model = smf.wls(full_formula, data=df, weights=df['perwt']).fit(
        cov_type='cluster', cov_kwds={'groups': df['statefip']})
    result = {
        'spec_id': 'robust/cluster/state',
        'spec_tree_path': 'robustness/clustering_variations.md#single-level-clustering',
        'description': 'Full controls with state-clustered SE',
        'coef': float(model.params['par_Arrived0_8']),
        'se': float(model.bse['par_Arrived0_8']),
        'pval': float(model.pvalues['par_Arrived0_8']),
        'n_obs': int(model.nobs),
        'r_squared': float(model.rsquared),
        'n_clusters': int(df['statefip'].nunique()),
        'coefficient_vector_json': '{}'
    }
    results.append(result)

    # Year clustered
    model = smf.wls(full_formula, data=df, weights=df['perwt']).fit(
        cov_type='cluster', cov_kwds={'groups': df['year']})
    result = {
        'spec_id': 'robust/cluster/year',
        'spec_tree_path': 'robustness/clustering_variations.md#single-level-clustering',
        'description': 'Full controls with year-clustered SE',
        'coef': float(model.params['par_Arrived0_8']),
        'se': float(model.bse['par_Arrived0_8']),
        'pval': float(model.pvalues['par_Arrived0_8']),
        'n_obs': int(model.nobs),
        'r_squared': float(model.rsquared),
        'n_clusters': int(df['year'].nunique()),
        'coefficient_vector_json': '{}'
    }
    results.append(result)

    return results

def run_leave_one_out(df, controls, baseline_coef):
    """Run leave-one-out robustness checks."""
    results = []
    full_formula = "not_hisp ~ par_Arrived0_8 + " + " + ".join(controls['full'])

    for group_name, group_vars in controls['groups'].items():
        loo_controls = [c for c in controls['full'] if c not in group_vars]
        formula = "not_hisp ~ par_Arrived0_8 + " + " + ".join(loo_controls)

        result = run_wls(formula, df, df['perwt'],
                         f'robust/loo/drop_{group_name}',
                         'robustness/leave_one_out.md',
                         f'Leave-one-out: drop {group_name} controls',
                         {'dropped_variable': group_name})

        coef_change_pct = ((result['coef'] - baseline_coef) / baseline_coef * 100) if baseline_coef != 0 else np.nan
        result['coef_change_pct'] = coef_change_pct
        result['baseline_coef'] = baseline_coef

        results.append(result)
        print(f"Drop {group_name}: coef={result['coef']:.4f}, change={coef_change_pct:.1f}%")

    return results

def run_single_covariate(df, controls, bivariate_coef):
    """Run single covariate robustness checks."""
    results = []

    for group_name, group_vars in controls['groups'].items():
        formula = "not_hisp ~ par_Arrived0_8 + " + " + ".join(group_vars)

        result = run_wls(formula, df, df['perwt'],
                         f'robust/single/{group_name}',
                         'robustness/single_covariate.md',
                         f'Treatment + {group_name} only')

        change = ((result['coef'] - bivariate_coef) / bivariate_coef * 100) if bivariate_coef != 0 else np.nan
        result['change_from_bivariate_pct'] = change
        result['bivariate_coef'] = bivariate_coef

        results.append(result)
        print(f"Single {group_name}: coef={result['coef']:.4f}, change={change:.1f}%")

    # Bivariate
    result = run_wls("not_hisp ~ par_Arrived0_8", df, df['perwt'],
                     'robust/single/none', 'robustness/single_covariate.md',
                     'Treatment only (bivariate)')
    results.append(result)

    return results

def run_sample_restrictions(df, controls):
    """Run sample restriction robustness checks."""
    results = []
    full_formula = "not_hisp ~ par_Arrived0_8 + " + " + ".join(controls['full'])

    # Time-based
    restrictions = [
        ('early_period', df['year'] <= 9, 'Early period (2000-2009)'),
        ('late_period', df['year'] >= 10, 'Late period (2010-2019)'),
        ('exclude_first_year', df['year'] > 0, 'Exclude first year (2000)'),
        ('exclude_last_year', df['year'] < 19, 'Exclude last year (2019)'),
        ('male_only', df['female'] == 0, 'Male children only'),
        ('female_only', df['female'] == 1, 'Female children only'),
        ('mother_immigrant', df['par_female'] == 1, 'Mother is immigrant parent'),
        ('father_immigrant', df['par_female'] == 0, 'Father is immigrant parent'),
        ('both_parents', df['p_present_both'] == 1, 'Living with both parents'),
        ('young_children', df['age'] <= 8, 'Young children (age 0-8)'),
        ('older_children', df['age'] >= 9, 'Older children (age 9-17)'),
        ('border_states', df['state_CA_TX_AZ_NM'] == 1, 'Border states (CA, TX, AZ, NM)'),
        ('non_border_states', df['state_CA_TX_AZ_NM'] == 0, 'Non-border states'),
        ('california_only', df['state_california'] == 1, 'California only'),
        ('texas_only', df['state_texas'] == 1, 'Texas only'),
        ('exclude_california', df['state_california'] == 0, 'Exclude California'),
        ('exclude_texas', df['state_texas'] == 0, 'Exclude Texas'),
    ]

    for name, mask, description in restrictions:
        df_sub = df[mask]
        result = run_wls(full_formula, df_sub, df_sub['perwt'],
                         f'robust/sample/{name}',
                         'robustness/sample_restrictions.md',
                         description)
        results.append(result)
        print(f"{name}: coef={result['coef']:.4f}, n={result['n_obs']}")

    return results

def run_functional_form(df, controls):
    """Run functional form variations."""
    results = []
    full_formula = "not_hisp ~ par_Arrived0_8 + " + " + ".join(controls['full'])

    # Continuous age at arrival
    formula = "not_hisp ~ par_AgeAtArrival + " + " + ".join(controls['full'])
    model = smf.wls(formula, data=df, weights=df['perwt']).fit(cov_type='HC1')
    result = {
        'spec_id': 'robust/form/continuous_treatment',
        'spec_tree_path': 'robustness/functional_form.md#treatment-transformations',
        'description': 'Continuous age at arrival (linear)',
        'coef': float(model.params['par_AgeAtArrival']),
        'se': float(model.bse['par_AgeAtArrival']),
        'pval': float(model.pvalues['par_AgeAtArrival']),
        'n_obs': int(model.nobs),
        'r_squared': float(model.rsquared),
        'coefficient_vector_json': '{}'
    }
    results.append(result)
    print(f"Continuous treatment: coef={result['coef']:.5f}")

    # Quadratic
    df['par_AgeAtArrival_sq'] = df['par_AgeAtArrival'] ** 2
    formula = "not_hisp ~ par_AgeAtArrival + par_AgeAtArrival_sq + " + " + ".join(controls['full'])
    model = smf.wls(formula, data=df, weights=df['perwt']).fit(cov_type='HC1')
    result = {
        'spec_id': 'robust/form/quadratic_treatment',
        'spec_tree_path': 'robustness/functional_form.md#nonlinear-specifications',
        'description': 'Quadratic age at arrival',
        'coef': float(model.params['par_AgeAtArrival']),
        'se': float(model.bse['par_AgeAtArrival']),
        'pval': float(model.pvalues['par_AgeAtArrival']),
        'coef_squared': float(model.params['par_AgeAtArrival_sq']),
        'n_obs': int(model.nobs),
        'r_squared': float(model.rsquared),
        'coefficient_vector_json': '{}'
    }
    results.append(result)
    print(f"Quadratic: linear={result['coef']:.5f}, squared={result['coef_squared']:.6f}")

    # Logit
    try:
        logit_model = smf.logit(full_formula, data=df).fit(disp=0)
        mfx = logit_model.get_margeff(at='mean')
        result = {
            'spec_id': 'ols/method/logit',
            'spec_tree_path': 'methods/discrete_choice.md#logit',
            'description': 'Logit regression',
            'coef': float(logit_model.params['par_Arrived0_8']),
            'se': float(logit_model.bse['par_Arrived0_8']),
            'pval': float(logit_model.pvalues['par_Arrived0_8']),
            'marginal_effect': float(mfx.margeff[0]),
            'n_obs': int(logit_model.nobs),
            'r_squared': float(logit_model.prsquared),
            'coefficient_vector_json': '{}'
        }
        results.append(result)
        print(f"Logit: coef={result['coef']:.4f}, ME={result['marginal_effect']:.4f}")
    except Exception as e:
        print(f"Logit error: {e}")

    # Probit
    try:
        probit_model = smf.probit(full_formula, data=df).fit(disp=0)
        mfx = probit_model.get_margeff(at='mean')
        result = {
            'spec_id': 'ols/method/probit',
            'spec_tree_path': 'methods/discrete_choice.md#probit',
            'description': 'Probit regression',
            'coef': float(probit_model.params['par_Arrived0_8']),
            'se': float(probit_model.bse['par_Arrived0_8']),
            'pval': float(probit_model.pvalues['par_Arrived0_8']),
            'marginal_effect': float(mfx.margeff[0]),
            'n_obs': int(probit_model.nobs),
            'r_squared': float(probit_model.prsquared),
            'coefficient_vector_json': '{}'
        }
        results.append(result)
        print(f"Probit: coef={result['coef']:.4f}, ME={result['marginal_effect']:.4f}")
    except Exception as e:
        print(f"Probit error: {e}")

    # Unweighted OLS
    model = smf.ols("not_hisp ~ par_Arrived0_8", data=df).fit(cov_type='HC1')
    result = {
        'spec_id': 'ols/method/ols_unweighted_bivariate',
        'spec_tree_path': 'methods/cross_sectional_ols.md#estimation-method',
        'description': 'Unweighted OLS, no controls',
        'coef': float(model.params['par_Arrived0_8']),
        'se': float(model.bse['par_Arrived0_8']),
        'pval': float(model.pvalues['par_Arrived0_8']),
        'n_obs': int(model.nobs),
        'r_squared': float(model.rsquared),
        'coefficient_vector_json': '{}'
    }
    results.append(result)
    print(f"Unweighted bivariate: coef={result['coef']:.4f}")

    model = smf.ols(full_formula, data=df).fit(cov_type='HC1')
    result = {
        'spec_id': 'ols/method/ols_unweighted_full',
        'spec_tree_path': 'methods/cross_sectional_ols.md#estimation-method',
        'description': 'Unweighted OLS with full controls',
        'coef': float(model.params['par_Arrived0_8']),
        'se': float(model.bse['par_Arrived0_8']),
        'pval': float(model.pvalues['par_Arrived0_8']),
        'n_obs': int(model.nobs),
        'r_squared': float(model.rsquared),
        'coefficient_vector_json': '{}'
    }
    results.append(result)
    print(f"Unweighted full: coef={result['coef']:.4f}")

    return results

def run_interactions(df, controls):
    """Run interaction specifications."""
    results = []
    full_formula = "not_hisp ~ par_Arrived0_8 + " + " + ".join(controls['full'])

    # Gender interaction
    df['par_Arrived0_8_female'] = df['par_Arrived0_8'] * df['female']
    formula = "not_hisp ~ par_Arrived0_8 + par_Arrived0_8_female + " + " + ".join(controls['full'])
    model = smf.wls(formula, data=df, weights=df['perwt']).fit(cov_type='HC1')
    result = {
        'spec_id': 'ols/interact/gender',
        'spec_tree_path': 'methods/cross_sectional_ols.md#interaction-effects',
        'description': 'Treatment x child gender interaction',
        'coef': float(model.params['par_Arrived0_8']),
        'se': float(model.bse['par_Arrived0_8']),
        'pval': float(model.pvalues['par_Arrived0_8']),
        'interact_coef': float(model.params['par_Arrived0_8_female']),
        'n_obs': int(model.nobs),
        'r_squared': float(model.rsquared),
        'coefficient_vector_json': '{}'
    }
    results.append(result)

    # Parent gender interaction
    df['par_Arrived0_8_par_female'] = df['par_Arrived0_8'] * df['par_female']
    formula = "not_hisp ~ par_Arrived0_8 + par_Arrived0_8_par_female + " + " + ".join(controls['full'])
    model = smf.wls(formula, data=df, weights=df['perwt']).fit(cov_type='HC1')
    result = {
        'spec_id': 'ols/interact/par_gender',
        'spec_tree_path': 'methods/cross_sectional_ols.md#interaction-effects',
        'description': 'Treatment x parent gender interaction',
        'coef': float(model.params['par_Arrived0_8']),
        'se': float(model.bse['par_Arrived0_8']),
        'pval': float(model.pvalues['par_Arrived0_8']),
        'interact_coef': float(model.params['par_Arrived0_8_par_female']),
        'n_obs': int(model.nobs),
        'r_squared': float(model.rsquared),
        'coefficient_vector_json': '{}'
    }
    results.append(result)

    # Time period interaction
    df['late_period'] = (df['year'] >= 10).astype(int)
    df['par_Arrived0_8_late'] = df['par_Arrived0_8'] * df['late_period']
    formula = "not_hisp ~ par_Arrived0_8 + par_Arrived0_8_late + late_period + " + " + ".join(controls['full'])
    model = smf.wls(formula, data=df, weights=df['perwt']).fit(cov_type='HC1')
    result = {
        'spec_id': 'ols/interact/time_period',
        'spec_tree_path': 'methods/cross_sectional_ols.md#interaction-effects',
        'description': 'Treatment x late period interaction',
        'coef': float(model.params['par_Arrived0_8']),
        'se': float(model.bse['par_Arrived0_8']),
        'pval': float(model.pvalues['par_Arrived0_8']),
        'interact_coef': float(model.params['par_Arrived0_8_late']),
        'n_obs': int(model.nobs),
        'r_squared': float(model.rsquared),
        'coefficient_vector_json': '{}'
    }
    results.append(result)

    return results

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the full specification search."""
    print("=" * 80)
    print("SPECIFICATION SEARCH: Paper 223321-V1")
    print("=" * 80)

    # Load data
    df = load_data()
    controls = get_control_variables(df)

    all_results = []

    # Baseline specifications
    print("\n--- Baseline Specifications ---")
    results = run_baseline_specifications(df, controls)
    all_results.extend(results)
    baseline_coef = results[2]['coef']  # Full controls coefficient
    bivariate_coef = results[0]['coef']  # Bivariate coefficient

    # SE variations
    print("\n--- SE Variations ---")
    results = run_se_variations(df, controls)
    all_results.extend(results)

    # Demographics only
    demo_controls = ['female'] + [x for x in df.columns if x.startswith('year_') and x != 'year_2000'] + \
                    [x for x in df.columns if x.startswith('child_age_') and x != 'child_age_0']
    formula = "not_hisp ~ par_Arrived0_8 + " + " + ".join(demo_controls)
    result = run_wls(formula, df, df['perwt'],
                     'ols/controls/demographics', 'methods/cross_sectional_ols.md#control-sets',
                     'Demographic controls only')
    all_results.append(result)

    # No controls
    result = run_wls("not_hisp ~ par_Arrived0_8", df, df['perwt'],
                     'ols/controls/none', 'methods/cross_sectional_ols.md#control-sets',
                     'No controls')
    all_results.append(result)

    # Leave-one-out
    print("\n--- Leave-One-Out ---")
    results = run_leave_one_out(df, controls, baseline_coef)
    all_results.extend(results)

    # Single covariate
    print("\n--- Single Covariate ---")
    results = run_single_covariate(df, controls, bivariate_coef)
    all_results.extend(results)

    # Sample restrictions
    print("\n--- Sample Restrictions ---")
    results = run_sample_restrictions(df, controls)
    all_results.extend(results)

    # Functional form
    print("\n--- Functional Form ---")
    results = run_functional_form(df, controls)
    all_results.extend(results)

    # Interactions
    print("\n--- Interactions ---")
    results = run_interactions(df, controls)
    all_results.extend(results)

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n{'=' * 80}")
    print(f"Saved {len(results_df)} specifications to {OUTPUT_FILE}")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main()
