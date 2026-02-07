#!/usr/bin/env python3
"""
Specification Search for Paper 223321-V1
Effect of Parent's Age at Arrival on Child Not Identifying as Hispanic

AER Papers and Proceedings 2025

This script runs 50+ robustness specifications following the i4r methodology
and the specification tree at specification_tree/.
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# Configuration
PAPER_ID = "223321-V1"
JOURNAL = "AER P&P"
PAPER_TITLE = "Effect of Parent's Age at Arrival on Child Not Identifying as Hispanic"
DATA_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/223321-V1/data/analysis/Census2000-ACS2001-2019_d1.dta"
OUTPUT_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/223321-V1/specification_results.csv"

# Load data
print("Loading data...")
df = pd.read_stata(DATA_PATH)

# Convert categorical columns to numeric where needed
df['year_num'] = df['year'].astype(str).astype(int)
df['statefip_num'] = df['statefip'].cat.codes
df['sex_num'] = df['sex'].cat.codes
df['age_num'] = df['age'].cat.codes
df['par_age_num'] = df['par_age']

# Create some derived variables
df['not_hisp'] = df['not_hisp'].astype(float)
df['par_Arrived0_8'] = df['par_Arrived0_8'].astype(float)
df['perwt'] = df['perwt'].astype(float)

# Define control variable groups
baseline_controls = ['year_num', 'sex_num', 'par_sex', 'age_num', 'par_age_num']
additional_controls_list = ['par_english_ability', 'par_educ', 'p_present_cat3']

# Define all control dummy variables from the data
year_dummies = [c for c in df.columns if c.startswith('year_') and c != 'year_num' and c not in ['year_2000']]
child_age_dummies = [c for c in df.columns if c.startswith('child_age_') and c != 'child_age_0']
par_age_dummies = [c for c in df.columns if c.startswith('par_age_') and c not in ['par_age_25', 'par_age_num']]
state_dummies = [c for c in df.columns if c.startswith('state_') and c != 'state_california' and c not in ['state_CA_TX_AZ_NM']]
par_english_dummies = [c for c in df.columns if c.startswith('par_English_') and c != 'par_English_notatall']
par_educ_dummies = [c for c in df.columns if c.startswith('par_EDUC_') and c not in ['par_EDUC_no_schooling', 'par_EDUC_less12', 'par_EDUC_12', 'par_EDUC_more12']]
intermarriage_dummies = ['Oth_Parrent_FBHisp', 'Oth_Parrent_FBnonHisp', 'Oth_Parrent_USHisp', 'Oth_Parrent_USnonHisp']
family_structure_dummies = ['p_present_momonly', 'p_present_dadonly']

# Full control set
full_controls = year_dummies + child_age_dummies + par_age_dummies + ['female', 'par_female'] + par_english_dummies + par_educ_dummies + family_structure_dummies + intermarriage_dummies + state_dummies

# Convert needed columns
for col in full_controls:
    if col in df.columns:
        df[col] = df[col].astype(float)
df['female'] = df['female'].astype(float)
df['par_female'] = df['par_female'].astype(float)

# Results storage
results = []

def run_weighted_ols(df_sub, formula, weights_col='perwt', spec_id='baseline', spec_tree_path='methods/cross_sectional_ols.md',
                     sample_desc='Full sample', controls_desc='', cluster_var=None):
    """Run weighted OLS regression and extract results."""
    try:
        # Using statsmodels WLS with robust SEs
        y_var = formula.split('~')[0].strip()
        x_vars = formula.split('~')[1].strip()

        model = smf.wls(formula, data=df_sub, weights=df_sub[weights_col]).fit(cov_type='HC1')

        treatment_var = 'par_Arrived0_8'

        # Extract results
        coef = model.params.get(treatment_var, np.nan)
        se = model.bse.get(treatment_var, np.nan)
        tstat = model.tvalues.get(treatment_var, np.nan)
        pval = model.pvalues.get(treatment_var, np.nan)

        # Confidence interval
        ci = model.conf_int().loc[treatment_var] if treatment_var in model.params.index else [np.nan, np.nan]

        # Build coefficient vector JSON
        coef_vector = {
            "treatment": {
                "var": treatment_var,
                "coef": float(coef) if not np.isnan(coef) else None,
                "se": float(se) if not np.isnan(se) else None,
                "pval": float(pval) if not np.isnan(pval) else None
            },
            "controls": [],
            "diagnostics": {
                "r_squared": float(model.rsquared) if hasattr(model, 'rsquared') else None,
                "f_stat": float(model.fvalue) if hasattr(model, 'fvalue') else None
            }
        }

        # Add a few control coefficients (not all - too many)
        for var in list(model.params.index)[:10]:
            if var not in [treatment_var, 'Intercept']:
                coef_vector["controls"].append({
                    "var": var,
                    "coef": float(model.params[var]),
                    "se": float(model.bse[var]),
                    "pval": float(model.pvalues[var])
                })

        return {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': y_var,
            'treatment_var': treatment_var,
            'coefficient': coef,
            'std_error': se,
            't_stat': tstat,
            'p_value': pval,
            'ci_lower': ci[0] if not np.isnan(ci[0]) else np.nan,
            'ci_upper': ci[1] if not np.isnan(ci[1]) else np.nan,
            'n_obs': int(model.nobs),
            'r_squared': model.rsquared,
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc,
            'fixed_effects': 'None (controls as dummies)',
            'controls_desc': controls_desc,
            'cluster_var': cluster_var if cluster_var else 'None (robust HC1)',
            'model_type': 'WLS',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
    except Exception as e:
        print(f"Error in spec {spec_id}: {e}")
        return None

print("Running specifications...")

# ==============================================================================
# BASELINE SPECIFICATIONS (Table 2 replication)
# ==============================================================================

# Baseline 1: No controls (Column 1 of Table 2)
formula = 'not_hisp ~ par_Arrived0_8'
result = run_weighted_ols(df, formula, spec_id='baseline_no_controls',
                          spec_tree_path='methods/cross_sectional_ols.md#baseline',
                          controls_desc='No controls')
if result: results.append(result)
print(f"Spec 1: {result['spec_id']} - coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Baseline 2: With basic controls (Column 2 of Table 2)
controls_str = ' + '.join(year_dummies + child_age_dummies + par_age_dummies + ['female', 'par_female'])
formula = f'not_hisp ~ par_Arrived0_8 + {controls_str}'
result = run_weighted_ols(df, formula, spec_id='baseline_basic_controls',
                          spec_tree_path='methods/cross_sectional_ols.md#baseline',
                          controls_desc='Year, child age, parent age, sex dummies')
if result: results.append(result)
print(f"Spec 2: {result['spec_id']} - coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Baseline 3: Full specification (Column 3 of Table 2) - MAIN RESULT
controls_str = ' + '.join(year_dummies + child_age_dummies + par_age_dummies + ['female', 'par_female'] +
                          par_english_dummies + par_educ_dummies + family_structure_dummies +
                          intermarriage_dummies + state_dummies)
formula = f'not_hisp ~ par_Arrived0_8 + {controls_str}'
result = run_weighted_ols(df, formula, spec_id='baseline',
                          spec_tree_path='methods/cross_sectional_ols.md#baseline',
                          controls_desc='Full controls: year, age, sex, English, education, family structure, intermarriage, state')
if result: results.append(result)
baseline_coef = result['coefficient']
baseline_se = result['std_error']
print(f"Spec 3 (MAIN BASELINE): {result['spec_id']} - coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# ==============================================================================
# CONTROL VARIABLE VARIATIONS (Leave-one-out and build-up)
# ==============================================================================

# Define control groups for leave-one-out
control_groups = {
    'year': year_dummies,
    'child_age': child_age_dummies,
    'par_age': par_age_dummies,
    'sex': ['female', 'par_female'],
    'par_english': par_english_dummies,
    'par_educ': par_educ_dummies,
    'family_structure': family_structure_dummies,
    'intermarriage': intermarriage_dummies,
    'state': state_dummies
}

# Leave-one-out: Drop each control GROUP
for group_name, group_vars in control_groups.items():
    remaining_controls = []
    for g, v in control_groups.items():
        if g != group_name:
            remaining_controls.extend(v)

    if len(remaining_controls) > 0:
        controls_str = ' + '.join(remaining_controls)
        formula = f'not_hisp ~ par_Arrived0_8 + {controls_str}'
        result = run_weighted_ols(df, formula,
                                  spec_id=f'robust/loo/drop_{group_name}',
                                  spec_tree_path='robustness/leave_one_out.md',
                                  controls_desc=f'Full controls minus {group_name}')
        if result: results.append(result)
        print(f"LOO drop {group_name}: coef={result['coefficient']:.4f}")

# Build-up specifications
print("\nRunning control build-up specifications...")

# Bivariate
formula = 'not_hisp ~ par_Arrived0_8'
result = run_weighted_ols(df, formula, spec_id='robust/build/bivariate',
                          spec_tree_path='robustness/control_progression.md',
                          controls_desc='No controls')
if result: results.append(result)

# Add demographics (year, age, sex)
controls_str = ' + '.join(year_dummies + child_age_dummies + ['female'])
formula = f'not_hisp ~ par_Arrived0_8 + {controls_str}'
result = run_weighted_ols(df, formula, spec_id='robust/build/demographics',
                          spec_tree_path='robustness/control_progression.md',
                          controls_desc='Year, child age, sex')
if result: results.append(result)

# Add parent demographics
controls_str = ' + '.join(year_dummies + child_age_dummies + ['female', 'par_female'] + par_age_dummies)
formula = f'not_hisp ~ par_Arrived0_8 + {controls_str}'
result = run_weighted_ols(df, formula, spec_id='robust/build/par_demographics',
                          spec_tree_path='robustness/control_progression.md',
                          controls_desc='Year, age, sex + parent age/sex')
if result: results.append(result)

# Add English proficiency
controls_str = ' + '.join(year_dummies + child_age_dummies + ['female', 'par_female'] + par_age_dummies + par_english_dummies)
formula = f'not_hisp ~ par_Arrived0_8 + {controls_str}'
result = run_weighted_ols(df, formula, spec_id='robust/build/add_english',
                          spec_tree_path='robustness/control_progression.md',
                          controls_desc='+ parent English ability')
if result: results.append(result)

# Add education
controls_str = ' + '.join(year_dummies + child_age_dummies + ['female', 'par_female'] + par_age_dummies + par_english_dummies + par_educ_dummies)
formula = f'not_hisp ~ par_Arrived0_8 + {controls_str}'
result = run_weighted_ols(df, formula, spec_id='robust/build/add_education',
                          spec_tree_path='robustness/control_progression.md',
                          controls_desc='+ parent education')
if result: results.append(result)

# Add family structure
controls_str = ' + '.join(year_dummies + child_age_dummies + ['female', 'par_female'] + par_age_dummies + par_english_dummies + par_educ_dummies + family_structure_dummies)
formula = f'not_hisp ~ par_Arrived0_8 + {controls_str}'
result = run_weighted_ols(df, formula, spec_id='robust/build/add_family',
                          spec_tree_path='robustness/control_progression.md',
                          controls_desc='+ family structure')
if result: results.append(result)

# Add intermarriage
controls_str = ' + '.join(year_dummies + child_age_dummies + ['female', 'par_female'] + par_age_dummies + par_english_dummies + par_educ_dummies + family_structure_dummies + intermarriage_dummies)
formula = f'not_hisp ~ par_Arrived0_8 + {controls_str}'
result = run_weighted_ols(df, formula, spec_id='robust/build/add_intermarriage',
                          spec_tree_path='robustness/control_progression.md',
                          controls_desc='+ intermarriage')
if result: results.append(result)

# ==============================================================================
# SAMPLE RESTRICTIONS
# ==============================================================================
print("\nRunning sample restriction specifications...")

# By time period
for period, years in [('early', range(2000, 2010)), ('late', range(2010, 2020))]:
    df_sub = df[df['year_num'].isin(years)]
    formula = f'not_hisp ~ par_Arrived0_8 + {" + ".join(full_controls)}'
    result = run_weighted_ols(df_sub, formula,
                              spec_id=f'robust/sample/{period}_period',
                              spec_tree_path='robustness/sample_restrictions.md',
                              sample_desc=f'{period.title()} period: {min(years)}-{max(years)}',
                              controls_desc='Full controls')
    if result: results.append(result)
    print(f"Sample {period}: coef={result['coefficient']:.4f}, n={result['n_obs']}")

# Drop each decade start
for year in [2000, 2010]:
    df_sub = df[df['year_num'] != year]
    formula = f'not_hisp ~ par_Arrived0_8 + {" + ".join(full_controls)}'
    result = run_weighted_ols(df_sub, formula,
                              spec_id=f'robust/sample/drop_year_{year}',
                              spec_tree_path='robustness/sample_restrictions.md',
                              sample_desc=f'Excluding year {year}',
                              controls_desc='Full controls')
    if result: results.append(result)

# By child age groups
for age_group, ages in [('young_children', range(0, 6)), ('school_age', range(6, 13)), ('teens', range(13, 18))]:
    df_sub = df[df['age_num'].isin(ages)]
    formula = f'not_hisp ~ par_Arrived0_8 + {" + ".join(full_controls)}'
    result = run_weighted_ols(df_sub, formula,
                              spec_id=f'robust/sample/{age_group}',
                              spec_tree_path='robustness/sample_restrictions.md',
                              sample_desc=f'Child age group: {age_group}',
                              controls_desc='Full controls')
    if result: results.append(result)
    print(f"Age group {age_group}: coef={result['coefficient']:.4f}, n={result['n_obs']}")

# By gender
for gender, female_val in [('male', 0), ('female', 1)]:
    df_sub = df[df['female'] == female_val]
    formula = f'not_hisp ~ par_Arrived0_8 + {" + ".join([c for c in full_controls if c != "female"])}'
    result = run_weighted_ols(df_sub, formula,
                              spec_id=f'robust/sample/{gender}_only',
                              spec_tree_path='robustness/sample_restrictions.md',
                              sample_desc=f'{gender.title()} children only',
                              controls_desc='Full controls (excluding gender)')
    if result: results.append(result)
    print(f"Gender {gender}: coef={result['coefficient']:.4f}, n={result['n_obs']}")

# By parent gender
for par_gender, par_female_val in [('mother', 1), ('father', 0)]:
    df_sub = df[df['par_female'] == par_female_val]
    formula = f'not_hisp ~ par_Arrived0_8 + {" + ".join([c for c in full_controls if c != "par_female"])}'
    result = run_weighted_ols(df_sub, formula,
                              spec_id=f'robust/sample/par_{par_gender}',
                              spec_tree_path='robustness/sample_restrictions.md',
                              sample_desc=f'Immigrant parent is {par_gender}',
                              controls_desc='Full controls (excluding parent gender)')
    if result: results.append(result)
    print(f"Parent {par_gender}: coef={result['coefficient']:.4f}, n={result['n_obs']}")

# By family structure
for structure, val in [('both_parents', 0), ('single_parent', 1)]:
    if structure == 'both_parents':
        df_sub = df[df['p_present_both'] == 1]
    else:
        df_sub = df[(df['p_present_momonly'] == 1) | (df['p_present_dadonly'] == 1)]
    formula = f'not_hisp ~ par_Arrived0_8 + {" + ".join([c for c in full_controls if c not in family_structure_dummies])}'
    result = run_weighted_ols(df_sub, formula,
                              spec_id=f'robust/sample/{structure}',
                              spec_tree_path='robustness/sample_restrictions.md',
                              sample_desc=f'Family structure: {structure}',
                              controls_desc='Full controls (excluding family structure)')
    if result: results.append(result)
    print(f"Family {structure}: coef={result['coefficient']:.4f}, n={result['n_obs']}")

# Key states
big_states = ['state_california', 'state_texas', 'state_arizona']
for state_var in big_states:
    state_name = state_var.replace('state_', '')
    df_sub = df[df[state_var] == 1]
    # Remove state dummies for single-state analysis
    controls_no_state = [c for c in full_controls if not c.startswith('state_')]
    formula = f'not_hisp ~ par_Arrived0_8 + {" + ".join(controls_no_state)}'
    result = run_weighted_ols(df_sub, formula,
                              spec_id=f'robust/sample/{state_name}_only',
                              spec_tree_path='robustness/sample_restrictions.md',
                              sample_desc=f'{state_name.title()} only',
                              controls_desc='Full controls (excluding state)')
    if result: results.append(result)
    print(f"State {state_name}: coef={result['coefficient']:.4f}, n={result['n_obs']}")

# Traditional Mexican states vs others
df_sub = df[df['state_CA_TX_AZ_NM'] == 1]
controls_no_state = [c for c in full_controls if not c.startswith('state_')]
formula = f'not_hisp ~ par_Arrived0_8 + {" + ".join(controls_no_state)}'
result = run_weighted_ols(df_sub, formula,
                          spec_id='robust/sample/traditional_states',
                          spec_tree_path='robustness/sample_restrictions.md',
                          sample_desc='CA, TX, AZ, NM only',
                          controls_desc='Full controls (excluding state)')
if result: results.append(result)

df_sub = df[df['state_CA_TX_AZ_NM'] == 0]
formula = f'not_hisp ~ par_Arrived0_8 + {" + ".join(controls_no_state)}'
result = run_weighted_ols(df_sub, formula,
                          spec_id='robust/sample/non_traditional_states',
                          spec_tree_path='robustness/sample_restrictions.md',
                          sample_desc='Outside CA, TX, AZ, NM',
                          controls_desc='Full controls (excluding state)')
if result: results.append(result)

# ==============================================================================
# ALTERNATIVE TREATMENT DEFINITIONS
# ==============================================================================
print("\nRunning alternative treatment specifications...")

# Continuous treatment: parent's actual age at arrival
formula = f'not_hisp ~ par_AgeAtArrival + {" + ".join(full_controls)}'
result = run_weighted_ols(df, formula, spec_id='robust/treatment/continuous',
                          spec_tree_path='robustness/measurement.md',
                          controls_desc='Full controls, continuous treatment')
if result:
    result['treatment_var'] = 'par_AgeAtArrival'
    results.append(result)
    print(f"Continuous treatment: coef={result['coefficient']:.4f}")

# Different age cutoffs
for cutoff in [5, 6, 7, 9, 10]:
    df[f'par_Arrived0_{cutoff}'] = (df['par_AgeAtArrival'] <= cutoff).astype(float)
    formula = f'not_hisp ~ par_Arrived0_{cutoff} + {" + ".join(full_controls)}'
    result = run_weighted_ols(df, formula, spec_id=f'robust/treatment/cutoff_{cutoff}',
                              spec_tree_path='robustness/measurement.md',
                              controls_desc=f'Treatment: arrived age 0-{cutoff}')
    if result:
        result['treatment_var'] = f'par_Arrived0_{cutoff}'
        results.append(result)
        print(f"Cutoff {cutoff}: coef={result['coefficient']:.4f}")

# ==============================================================================
# INFERENCE VARIATIONS
# ==============================================================================
print("\nRunning inference variation specifications...")

# Unweighted regression
formula = f'not_hisp ~ par_Arrived0_8 + {" + ".join(full_controls)}'
try:
    model = smf.ols(formula, data=df).fit(cov_type='HC1')
    result = {
        'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
        'spec_id': 'robust/weights/unweighted',
        'spec_tree_path': 'robustness/measurement.md',
        'outcome_var': 'not_hisp', 'treatment_var': 'par_Arrived0_8',
        'coefficient': model.params['par_Arrived0_8'],
        'std_error': model.bse['par_Arrived0_8'],
        't_stat': model.tvalues['par_Arrived0_8'],
        'p_value': model.pvalues['par_Arrived0_8'],
        'ci_lower': model.conf_int().loc['par_Arrived0_8'][0],
        'ci_upper': model.conf_int().loc['par_Arrived0_8'][1],
        'n_obs': int(model.nobs), 'r_squared': model.rsquared,
        'coefficient_vector_json': '{}',
        'sample_desc': 'Full sample', 'fixed_effects': 'None',
        'controls_desc': 'Full controls', 'cluster_var': 'None (HC1)',
        'model_type': 'OLS (unweighted)', 'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    }
    results.append(result)
    print(f"Unweighted: coef={result['coefficient']:.4f}")
except Exception as e:
    print(f"Error in unweighted: {e}")

# HC2 and HC3 standard errors
for hc_type in ['HC2', 'HC3']:
    try:
        model = smf.wls(formula, data=df, weights=df['perwt']).fit(cov_type=hc_type)
        result = {
            'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
            'spec_id': f'robust/se/{hc_type.lower()}',
            'spec_tree_path': 'robustness/clustering_variations.md',
            'outcome_var': 'not_hisp', 'treatment_var': 'par_Arrived0_8',
            'coefficient': model.params['par_Arrived0_8'],
            'std_error': model.bse['par_Arrived0_8'],
            't_stat': model.tvalues['par_Arrived0_8'],
            'p_value': model.pvalues['par_Arrived0_8'],
            'ci_lower': model.conf_int().loc['par_Arrived0_8'][0],
            'ci_upper': model.conf_int().loc['par_Arrived0_8'][1],
            'n_obs': int(model.nobs), 'r_squared': model.rsquared,
            'coefficient_vector_json': '{}',
            'sample_desc': 'Full sample', 'fixed_effects': 'None',
            'controls_desc': 'Full controls', 'cluster_var': f'None ({hc_type})',
            'model_type': 'WLS', 'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
        results.append(result)
        print(f"{hc_type}: se={result['std_error']:.4f}")
    except Exception as e:
        print(f"Error in {hc_type}: {e}")

# Cluster by state
try:
    model = smf.wls(formula, data=df, weights=df['perwt']).fit(cov_type='cluster', cov_kwds={'groups': df['statefip']})
    result = {
        'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
        'spec_id': 'robust/cluster/state',
        'spec_tree_path': 'robustness/clustering_variations.md',
        'outcome_var': 'not_hisp', 'treatment_var': 'par_Arrived0_8',
        'coefficient': model.params['par_Arrived0_8'],
        'std_error': model.bse['par_Arrived0_8'],
        't_stat': model.tvalues['par_Arrived0_8'],
        'p_value': model.pvalues['par_Arrived0_8'],
        'ci_lower': model.conf_int().loc['par_Arrived0_8'][0],
        'ci_upper': model.conf_int().loc['par_Arrived0_8'][1],
        'n_obs': int(model.nobs), 'r_squared': model.rsquared,
        'coefficient_vector_json': '{}',
        'sample_desc': 'Full sample', 'fixed_effects': 'None',
        'controls_desc': 'Full controls', 'cluster_var': 'State',
        'model_type': 'WLS', 'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    }
    results.append(result)
    print(f"Clustered by state: se={result['std_error']:.4f}")
except Exception as e:
    print(f"Error in state clustering: {e}")

# Cluster by year
try:
    model = smf.wls(formula, data=df, weights=df['perwt']).fit(cov_type='cluster', cov_kwds={'groups': df['year']})
    result = {
        'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
        'spec_id': 'robust/cluster/year',
        'spec_tree_path': 'robustness/clustering_variations.md',
        'outcome_var': 'not_hisp', 'treatment_var': 'par_Arrived0_8',
        'coefficient': model.params['par_Arrived0_8'],
        'std_error': model.bse['par_Arrived0_8'],
        't_stat': model.tvalues['par_Arrived0_8'],
        'p_value': model.pvalues['par_Arrived0_8'],
        'ci_lower': model.conf_int().loc['par_Arrived0_8'][0],
        'ci_upper': model.conf_int().loc['par_Arrived0_8'][1],
        'n_obs': int(model.nobs), 'r_squared': model.rsquared,
        'coefficient_vector_json': '{}',
        'sample_desc': 'Full sample', 'fixed_effects': 'None',
        'controls_desc': 'Full controls', 'cluster_var': 'Year',
        'model_type': 'WLS', 'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    }
    results.append(result)
    print(f"Clustered by year: se={result['std_error']:.4f}")
except Exception as e:
    print(f"Error in year clustering: {e}")

# ==============================================================================
# HETEROGENEITY ANALYSIS
# ==============================================================================
print("\nRunning heterogeneity specifications...")

# Interaction with child gender
try:
    df['treat_x_female'] = df['par_Arrived0_8'] * df['female']
    formula_het = f'not_hisp ~ par_Arrived0_8 + female + treat_x_female + {" + ".join([c for c in full_controls if c != "female"])}'
    model = smf.wls(formula_het, data=df, weights=df['perwt']).fit(cov_type='HC1')
    result = {
        'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
        'spec_id': 'robust/het/interaction_gender',
        'spec_tree_path': 'robustness/heterogeneity.md',
        'outcome_var': 'not_hisp', 'treatment_var': 'treat_x_female',
        'coefficient': model.params['treat_x_female'],
        'std_error': model.bse['treat_x_female'],
        't_stat': model.tvalues['treat_x_female'],
        'p_value': model.pvalues['treat_x_female'],
        'ci_lower': model.conf_int().loc['treat_x_female'][0],
        'ci_upper': model.conf_int().loc['treat_x_female'][1],
        'n_obs': int(model.nobs), 'r_squared': model.rsquared,
        'coefficient_vector_json': json.dumps({
            'main_effect': {'coef': float(model.params['par_Arrived0_8']), 'se': float(model.bse['par_Arrived0_8'])},
            'interaction': {'coef': float(model.params['treat_x_female']), 'se': float(model.bse['treat_x_female'])}
        }),
        'sample_desc': 'Full sample', 'fixed_effects': 'None',
        'controls_desc': 'Full controls + gender interaction', 'cluster_var': 'None (HC1)',
        'model_type': 'WLS', 'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    }
    results.append(result)
    print(f"Gender interaction: coef={result['coefficient']:.4f}")
except Exception as e:
    print(f"Error in gender interaction: {e}")

# Interaction with parent gender
try:
    df['treat_x_par_female'] = df['par_Arrived0_8'] * df['par_female']
    formula_het = f'not_hisp ~ par_Arrived0_8 + par_female + treat_x_par_female + {" + ".join([c for c in full_controls if c != "par_female"])}'
    model = smf.wls(formula_het, data=df, weights=df['perwt']).fit(cov_type='HC1')
    result = {
        'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
        'spec_id': 'robust/het/interaction_par_gender',
        'spec_tree_path': 'robustness/heterogeneity.md',
        'outcome_var': 'not_hisp', 'treatment_var': 'treat_x_par_female',
        'coefficient': model.params['treat_x_par_female'],
        'std_error': model.bse['treat_x_par_female'],
        't_stat': model.tvalues['treat_x_par_female'],
        'p_value': model.pvalues['treat_x_par_female'],
        'ci_lower': model.conf_int().loc['treat_x_par_female'][0],
        'ci_upper': model.conf_int().loc['treat_x_par_female'][1],
        'n_obs': int(model.nobs), 'r_squared': model.rsquared,
        'coefficient_vector_json': json.dumps({
            'main_effect': {'coef': float(model.params['par_Arrived0_8']), 'se': float(model.bse['par_Arrived0_8'])},
            'interaction': {'coef': float(model.params['treat_x_par_female']), 'se': float(model.bse['treat_x_par_female'])}
        }),
        'sample_desc': 'Full sample', 'fixed_effects': 'None',
        'controls_desc': 'Full controls + parent gender interaction', 'cluster_var': 'None (HC1)',
        'model_type': 'WLS', 'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    }
    results.append(result)
    print(f"Parent gender interaction: coef={result['coefficient']:.4f}")
except Exception as e:
    print(f"Error in parent gender interaction: {e}")

# Interaction with living with both parents
try:
    df['treat_x_both_parents'] = df['par_Arrived0_8'] * df['p_present_both']
    formula_het = f'not_hisp ~ par_Arrived0_8 + p_present_both + treat_x_both_parents + {" + ".join([c for c in full_controls if c not in family_structure_dummies + ["p_present_both"]])}'
    model = smf.wls(formula_het, data=df, weights=df['perwt']).fit(cov_type='HC1')
    result = {
        'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
        'spec_id': 'robust/het/interaction_family',
        'spec_tree_path': 'robustness/heterogeneity.md',
        'outcome_var': 'not_hisp', 'treatment_var': 'treat_x_both_parents',
        'coefficient': model.params['treat_x_both_parents'],
        'std_error': model.bse['treat_x_both_parents'],
        't_stat': model.tvalues['treat_x_both_parents'],
        'p_value': model.pvalues['treat_x_both_parents'],
        'ci_lower': model.conf_int().loc['treat_x_both_parents'][0],
        'ci_upper': model.conf_int().loc['treat_x_both_parents'][1],
        'n_obs': int(model.nobs), 'r_squared': model.rsquared,
        'coefficient_vector_json': '{}',
        'sample_desc': 'Full sample', 'fixed_effects': 'None',
        'controls_desc': 'Full controls + family structure interaction', 'cluster_var': 'None (HC1)',
        'model_type': 'WLS', 'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    }
    results.append(result)
    print(f"Family structure interaction: coef={result['coefficient']:.4f}")
except Exception as e:
    print(f"Error in family interaction: {e}")

# Interaction with parent English fluency
try:
    df['treat_x_english_fluent'] = df['par_Arrived0_8'] * df['par_english_fluent']
    formula_het = f'not_hisp ~ par_Arrived0_8 + par_english_fluent + treat_x_english_fluent + {" + ".join([c for c in full_controls if c not in par_english_dummies + ["par_english_fluent"]])}'
    model = smf.wls(formula_het, data=df, weights=df['perwt']).fit(cov_type='HC1')
    result = {
        'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
        'spec_id': 'robust/het/interaction_english',
        'spec_tree_path': 'robustness/heterogeneity.md',
        'outcome_var': 'not_hisp', 'treatment_var': 'treat_x_english_fluent',
        'coefficient': model.params['treat_x_english_fluent'],
        'std_error': model.bse['treat_x_english_fluent'],
        't_stat': model.tvalues['treat_x_english_fluent'],
        'p_value': model.pvalues['treat_x_english_fluent'],
        'ci_lower': model.conf_int().loc['treat_x_english_fluent'][0],
        'ci_upper': model.conf_int().loc['treat_x_english_fluent'][1],
        'n_obs': int(model.nobs), 'r_squared': model.rsquared,
        'coefficient_vector_json': '{}',
        'sample_desc': 'Full sample', 'fixed_effects': 'None',
        'controls_desc': 'Full controls + English fluency interaction', 'cluster_var': 'None (HC1)',
        'model_type': 'WLS', 'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    }
    results.append(result)
    print(f"English fluency interaction: coef={result['coefficient']:.4f}")
except Exception as e:
    print(f"Error in English interaction: {e}")

# By parent education level
for educ_group, educ_var in [('low_education', 'par_EDUC_less12'), ('high_education', 'par_EDUC_more12')]:
    df_sub = df[df[educ_var] == 1]
    controls_no_educ = [c for c in full_controls if c not in par_educ_dummies]
    formula = f'not_hisp ~ par_Arrived0_8 + {" + ".join(controls_no_educ)}'
    result = run_weighted_ols(df_sub, formula,
                              spec_id=f'robust/het/by_{educ_group}',
                              spec_tree_path='robustness/heterogeneity.md',
                              sample_desc=f'Parent {educ_group.replace("_", " ")}',
                              controls_desc='Full controls (excluding education)')
    if result: results.append(result)
    print(f"Parent {educ_group}: coef={result['coefficient']:.4f}, n={result['n_obs']}")

# ==============================================================================
# PLACEBO TESTS
# ==============================================================================
print("\nRunning placebo specifications...")

# Placebo: Effect on parent's own Hispanic identification (should be weak/zero with controls)
df['par_not_hisp'] = df['par_not_hisp'].astype(float)
formula = f'par_not_hisp ~ par_Arrived0_8 + {" + ".join(full_controls)}'
result = run_weighted_ols(df, formula, spec_id='robust/placebo/parent_hisp',
                          spec_tree_path='robustness/placebo_tests.md',
                          controls_desc='Full controls, outcome=parent not Hispanic')
if result:
    result['outcome_var'] = 'par_not_hisp'
    results.append(result)
    print(f"Placebo (parent): coef={result['coefficient']:.4f}")

# ==============================================================================
# ESTIMATION METHOD VARIATIONS
# ==============================================================================
print("\nRunning estimation method variations...")

# Logit/Probit (discrete choice since outcome is binary)
try:
    from statsmodels.discrete.discrete_model import Logit, Probit

    # Prepare data
    X_vars = ['par_Arrived0_8'] + full_controls[:20]  # Limit controls for convergence
    X = df[X_vars].copy()
    X = sm.add_constant(X)
    y = df['not_hisp']

    # Logit
    logit_model = Logit(y, X).fit(disp=0, maxiter=100)

    result = {
        'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
        'spec_id': 'robust/estimation/logit',
        'spec_tree_path': 'methods/discrete_choice.md',
        'outcome_var': 'not_hisp', 'treatment_var': 'par_Arrived0_8',
        'coefficient': logit_model.params['par_Arrived0_8'],
        'std_error': logit_model.bse['par_Arrived0_8'],
        't_stat': logit_model.tvalues['par_Arrived0_8'],
        'p_value': logit_model.pvalues['par_Arrived0_8'],
        'ci_lower': logit_model.conf_int().loc['par_Arrived0_8'][0],
        'ci_upper': logit_model.conf_int().loc['par_Arrived0_8'][1],
        'n_obs': int(logit_model.nobs), 'r_squared': logit_model.prsquared,
        'coefficient_vector_json': '{}',
        'sample_desc': 'Full sample', 'fixed_effects': 'None',
        'controls_desc': 'Subset of controls (for convergence)', 'cluster_var': 'None',
        'model_type': 'Logit', 'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    }
    results.append(result)
    print(f"Logit: coef={result['coefficient']:.4f}")

    # Probit
    probit_model = Probit(y, X).fit(disp=0, maxiter=100)

    result = {
        'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
        'spec_id': 'robust/estimation/probit',
        'spec_tree_path': 'methods/discrete_choice.md',
        'outcome_var': 'not_hisp', 'treatment_var': 'par_Arrived0_8',
        'coefficient': probit_model.params['par_Arrived0_8'],
        'std_error': probit_model.bse['par_Arrived0_8'],
        't_stat': probit_model.tvalues['par_Arrived0_8'],
        'p_value': probit_model.pvalues['par_Arrived0_8'],
        'ci_lower': probit_model.conf_int().loc['par_Arrived0_8'][0],
        'ci_upper': probit_model.conf_int().loc['par_Arrived0_8'][1],
        'n_obs': int(probit_model.nobs), 'r_squared': probit_model.prsquared,
        'coefficient_vector_json': '{}',
        'sample_desc': 'Full sample', 'fixed_effects': 'None',
        'controls_desc': 'Subset of controls (for convergence)', 'cluster_var': 'None',
        'model_type': 'Probit', 'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    }
    results.append(result)
    print(f"Probit: coef={result['coefficient']:.4f}")

except Exception as e:
    print(f"Error in logit/probit: {e}")

# Linear Probability Model without weights (for comparison)
formula = f'not_hisp ~ par_Arrived0_8 + {" + ".join(full_controls)}'
result = run_weighted_ols(df, formula, spec_id='ols/method/lpm_weighted',
                          spec_tree_path='methods/cross_sectional_ols.md',
                          controls_desc='Full controls (LPM with weights)')
if result: results.append(result)

# ==============================================================================
# SAVE RESULTS
# ==============================================================================

print(f"\n{'='*60}")
print(f"SPECIFICATION SEARCH COMPLETE")
print(f"{'='*60}")
print(f"Total specifications run: {len(results)}")

# Convert to DataFrame and save
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_PATH, index=False)
print(f"Results saved to: {OUTPUT_PATH}")

# Summary statistics
print(f"\nSUMMARY:")
print(f"  Total specifications: {len(results_df)}")
print(f"  Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
print(f"  Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
print(f"  Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")
print(f"  Median coefficient: {results_df['coefficient'].median():.4f}")
print(f"  Mean coefficient: {results_df['coefficient'].mean():.4f}")
print(f"  Range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")
