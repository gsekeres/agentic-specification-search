"""
Specification Search for Paper 114372-V1

Paper: Social Networks and Economic Behavior - Dictator Game Experiments
This paper studies how social networks and individual characteristics affect
giving behavior in dictator games and friendship formation.

Main Analyses:
1. Tables 1-2: Cross-sectional OLS on dictator game giving (database_56_network.dta)
2. Tables 3-4: Conditional logit on friendship choices (linkingdatabase.dta)
3. Tables 5-6: Cross-sectional OLS on simulated earnings (database_earnings.dta)

Primary method: cross_sectional_ols
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Set paths
BASE_PATH = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search'
DATA_PATH = f'{BASE_PATH}/data/downloads/extracted/114372-V1/data/'
OUTPUT_PATH = f'{BASE_PATH}/data/downloads/extracted/114372-V1/'

# Paper metadata
PAPER_ID = '114372-V1'
JOURNAL = 'AER'  # American Economic Review (assumed based on AEA package)
PAPER_TITLE = 'Social Networks and Economic Behavior: Dictator Game Experiments'

# ============================================================================
# Helper Functions
# ============================================================================

def extract_results(model, outcome_var, treatment_var, spec_id, spec_tree_path,
                    sample_desc='', fixed_effects='', controls_desc='',
                    cluster_var='', model_type='OLS', df=None):
    """Extract results from a fitted statsmodels model into standard format."""

    # Get coefficient for treatment variable
    coef = model.params.get(treatment_var, np.nan)
    se = model.bse.get(treatment_var, np.nan)
    t_stat = model.tvalues.get(treatment_var, np.nan)
    pval = model.pvalues.get(treatment_var, np.nan)

    # Confidence intervals
    conf_int = model.conf_int()
    if treatment_var in conf_int.index:
        ci_lower = conf_int.loc[treatment_var, 0]
        ci_upper = conf_int.loc[treatment_var, 1]
    else:
        ci_lower = np.nan
        ci_upper = np.nan

    # Get R-squared (if available)
    r_squared = getattr(model, 'rsquared', np.nan)
    if hasattr(model, 'prsquared'):  # Pseudo R-squared for discrete choice
        r_squared = model.prsquared

    # Build coefficient vector JSON
    coef_vector = {
        'treatment': {
            'var': treatment_var,
            'coef': float(coef) if not pd.isna(coef) else None,
            'se': float(se) if not pd.isna(se) else None,
            'pval': float(pval) if not pd.isna(pval) else None
        },
        'controls': [],
        'fixed_effects': [],
        'diagnostics': {
            'r_squared': float(r_squared) if not pd.isna(r_squared) else None,
            'f_stat': float(getattr(model, 'fvalue', np.nan)) if hasattr(model, 'fvalue') and not pd.isna(getattr(model, 'fvalue', np.nan)) else None
        }
    }

    # Add all coefficients
    for var in model.params.index:
        if var != treatment_var and var != 'Intercept':
            coef_vector['controls'].append({
                'var': var,
                'coef': float(model.params[var]) if not pd.isna(model.params[var]) else None,
                'se': float(model.bse[var]) if not pd.isna(model.bse[var]) else None,
                'pval': float(model.pvalues[var]) if not pd.isna(model.pvalues[var]) else None
            })

    return {
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'outcome_var': outcome_var,
        'treatment_var': treatment_var,
        'coefficient': coef,
        'std_error': se,
        't_stat': t_stat,
        'p_value': pval,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_obs': int(model.nobs),
        'r_squared': r_squared,
        'coefficient_vector_json': json.dumps(coef_vector),
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    }


def run_ols_robust(formula, df, spec_id, spec_tree_path, outcome_var, treatment_var,
                   sample_desc='', fixed_effects='', controls_desc='', model_type='OLS'):
    """Run OLS with robust standard errors."""
    model = smf.ols(formula, data=df).fit(cov_type='HC1')
    return extract_results(model, outcome_var, treatment_var, spec_id, spec_tree_path,
                          sample_desc, fixed_effects, controls_desc, 'robust', model_type, df)


def run_ols_clustered(formula, df, cluster_var, spec_id, spec_tree_path,
                      outcome_var, treatment_var, sample_desc='', fixed_effects='',
                      controls_desc='', model_type='OLS'):
    """Run OLS with clustered standard errors."""
    model = smf.ols(formula, data=df).fit(cov_type='cluster',
                                          cov_kwds={'groups': df[cluster_var]})
    return extract_results(model, outcome_var, treatment_var, spec_id, spec_tree_path,
                          sample_desc, fixed_effects, controls_desc, cluster_var, model_type, df)


# ============================================================================
# Load Data
# ============================================================================

print("Loading data...")

# Dataset 1: Dictator game data
df_dictator = pd.read_stata(DATA_PATH + 'database_56_network.dta')

# Dataset 2: Earnings data
df_earnings = pd.read_stata(DATA_PATH + 'database_earnings.dta')

# Dataset 3: Linking/friendship data
df_linking = pd.read_stata(DATA_PATH + 'linkingdatabase.dta')

print(f"Dictator game data: {df_dictator.shape[0]} observations")
print(f"Earnings data: {df_earnings.shape[0]} observations")
print(f"Linking data: {df_linking.shape[0]} observations")

# ============================================================================
# Analysis 1: Dictator Game (Tables 1 & 2)
# ============================================================================

results = []

print("\n" + "="*70)
print("ANALYSIS 1: DICTATOR GAME - Amount given in dictator game")
print("="*70)

# Filter to estimation sample
df_dict_sample = df_dictator[df_dictator['_est_ols'] == 1].copy()
print(f"Estimation sample: {len(df_dict_sample)} observations")

# Define variables
outcome_dictator = 'amount'
treatment_dictator = 'height'  # Main variable of interest: height (deviation from mean)

# Controls from Table 1
controls_t1 = ['asian', 'shy', 'shy_recipient', 'popular', 'popular_recipient',
               'samerace', 'sameheight', 'sameconf']

# Additional controls for Table 2 (with network variables)
controls_t2 = controls_t1 + ['close', 'between', 'Power', 'order']

# --------------------------------------------------
# BASELINE SPECIFICATIONS (Tables 1 & 2 replication)
# --------------------------------------------------

print("\n--- Baseline Replication (Table 1) ---")

# Table 1: Basic model
formula_t1 = f'{outcome_dictator} ~ height + ' + ' + '.join(controls_t1)
result = run_ols_clustered(formula_t1, df_dict_sample, 'population',
                           'baseline', 'methods/cross_sectional_ols.md',
                           outcome_dictator, treatment_dictator,
                           sample_desc='Dictator game participants',
                           controls_desc='Demographics and partner characteristics')
results.append(result)
print(f"Table 1: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Table 2: With network variables
print("\n--- Baseline with Network Controls (Table 2) ---")
# Need to handle missing values in network variables
df_dict_t2 = df_dict_sample.dropna(subset=['close', 'between', 'Power', 'order', 'partners_dr'])

formula_t2 = f'{outcome_dictator} ~ height + partners_dr + order + ' + ' + '.join(controls_t1) + ' + close + between + Power'
if len(df_dict_t2) > 20:
    result = run_ols_clustered(formula_t2, df_dict_t2, 'population',
                               'baseline_network', 'methods/cross_sectional_ols.md',
                               outcome_dictator, treatment_dictator,
                               sample_desc='Dictator game with network variables',
                               controls_desc='Demographics, partner characteristics, and network centrality')
    results.append(result)
    print(f"Table 2: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# --------------------------------------------------
# METHOD-SPECIFIC VARIATIONS
# --------------------------------------------------

print("\n--- Method-Specific Variations ---")

# Standard Errors variations
for se_type, se_label in [('HC1', 'robust'), ('HC2', 'hc2'), ('HC3', 'hc3')]:
    model = smf.ols(formula_t1, data=df_dict_sample).fit(cov_type=se_type)
    result = extract_results(model, outcome_dictator, treatment_dictator,
                            f'ols/se/{se_label}', 'methods/cross_sectional_ols.md#standard-errors',
                            sample_desc='Dictator game participants',
                            controls_desc='Demographics and partner characteristics',
                            cluster_var=se_label)
    results.append(result)
    print(f"SE {se_label}: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Control set variations
print("\n--- Control Set Variations ---")

# No controls (bivariate)
formula_bivariate = f'{outcome_dictator} ~ height'
result = run_ols_clustered(formula_bivariate, df_dict_sample, 'population',
                           'ols/controls/none', 'methods/cross_sectional_ols.md#control-sets',
                           outcome_dictator, treatment_dictator,
                           sample_desc='Dictator game participants',
                           controls_desc='None (bivariate)')
results.append(result)
print(f"Bivariate: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Demographics only
demo_controls = ['asian', 'shy', 'popular']
formula_demo = f'{outcome_dictator} ~ height + ' + ' + '.join(demo_controls)
result = run_ols_clustered(formula_demo, df_dict_sample, 'population',
                           'ols/controls/demographics', 'methods/cross_sectional_ols.md#control-sets',
                           outcome_dictator, treatment_dictator,
                           sample_desc='Dictator game participants',
                           controls_desc='Demographics only (asian, shy, popular)')
results.append(result)
print(f"Demographics only: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# --------------------------------------------------
# ROBUSTNESS: Leave-One-Out
# --------------------------------------------------

print("\n--- Leave-One-Out Robustness ---")

for control in controls_t1:
    remaining = [c for c in controls_t1 if c != control]
    formula_loo = f'{outcome_dictator} ~ height + ' + ' + '.join(remaining)
    result = run_ols_clustered(formula_loo, df_dict_sample, 'population',
                               f'robust/loo/drop_{control}', 'robustness/leave_one_out.md',
                               outcome_dictator, treatment_dictator,
                               sample_desc='Dictator game participants',
                               controls_desc=f'Baseline controls minus {control}')
    results.append(result)
    print(f"Drop {control}: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# --------------------------------------------------
# ROBUSTNESS: Single Covariate
# --------------------------------------------------

print("\n--- Single Covariate Robustness ---")

# Bivariate (no controls) - already done above
result = run_ols_clustered(formula_bivariate, df_dict_sample, 'population',
                           'robust/single/none', 'robustness/single_covariate.md',
                           outcome_dictator, treatment_dictator,
                           sample_desc='Dictator game participants',
                           controls_desc='None')
results.append(result)

for control in controls_t1:
    formula_single = f'{outcome_dictator} ~ height + {control}'
    result = run_ols_clustered(formula_single, df_dict_sample, 'population',
                               f'robust/single/{control}', 'robustness/single_covariate.md',
                               outcome_dictator, treatment_dictator,
                               sample_desc='Dictator game participants',
                               controls_desc=f'Single control: {control}')
    results.append(result)
    print(f"Single {control}: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# --------------------------------------------------
# ROBUSTNESS: Clustering Variations
# --------------------------------------------------

print("\n--- Clustering Variations ---")

# No clustering (robust only)
result = run_ols_robust(formula_t1, df_dict_sample,
                        'robust/cluster/none', 'robustness/clustering_variations.md',
                        outcome_dictator, treatment_dictator,
                        sample_desc='Dictator game participants',
                        controls_desc='Baseline controls')
results.append(result)
print(f"No clustering: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Cluster by population (baseline)
result = run_ols_clustered(formula_t1, df_dict_sample, 'population',
                           'robust/cluster/unit', 'robustness/clustering_variations.md',
                           outcome_dictator, treatment_dictator,
                           sample_desc='Dictator game participants',
                           controls_desc='Baseline controls')
results.append(result)
print(f"Cluster by population: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# --------------------------------------------------
# ROBUSTNESS: Sample Restrictions
# --------------------------------------------------

print("\n--- Sample Restrictions ---")

# Trim extreme outcomes (1% and 99%)
q01 = df_dict_sample[outcome_dictator].quantile(0.01)
q99 = df_dict_sample[outcome_dictator].quantile(0.99)
df_trimmed = df_dict_sample[(df_dict_sample[outcome_dictator] >= q01) &
                            (df_dict_sample[outcome_dictator] <= q99)]
if len(df_trimmed) > 20:
    result = run_ols_clustered(formula_t1, df_trimmed, 'population',
                               'robust/sample/trim_1pct', 'robustness/sample_restrictions.md',
                               outcome_dictator, treatment_dictator,
                               sample_desc=f'Trimmed 1%/99% (n={len(df_trimmed)})',
                               controls_desc='Baseline controls')
    results.append(result)
    print(f"Trimmed 1%: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Complete cases
df_complete = df_dict_sample.dropna(subset=[outcome_dictator, treatment_dictator] + controls_t1)
if len(df_complete) > 20:
    result = run_ols_clustered(formula_t1, df_complete, 'population',
                               'robust/sample/complete_cases', 'robustness/sample_restrictions.md',
                               outcome_dictator, treatment_dictator,
                               sample_desc=f'Complete cases (n={len(df_complete)})',
                               controls_desc='Baseline controls')
    results.append(result)
    print(f"Complete cases: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Subgroup: Asian only
df_asian = df_dict_sample[df_dict_sample['asian'] == 1]
if len(df_asian) > 20:
    # Need to drop asian from controls
    controls_no_asian = [c for c in controls_t1 if c != 'asian']
    formula_no_asian = f'{outcome_dictator} ~ height + ' + ' + '.join(controls_no_asian)
    result = run_ols_clustered(formula_no_asian, df_asian, 'population',
                               'robust/sample/asian_only', 'robustness/sample_restrictions.md',
                               outcome_dictator, treatment_dictator,
                               sample_desc=f'Asian subsample (n={len(df_asian)})',
                               controls_desc='Baseline controls minus asian')
    results.append(result)
    print(f"Asian only: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Subgroup: Non-Asian
df_non_asian = df_dict_sample[df_dict_sample['asian'] == 0]
if len(df_non_asian) > 20:
    result = run_ols_clustered(formula_no_asian, df_non_asian, 'population',
                               'robust/sample/non_asian_only', 'robustness/sample_restrictions.md',
                               outcome_dictator, treatment_dictator,
                               sample_desc=f'Non-Asian subsample (n={len(df_non_asian)})',
                               controls_desc='Baseline controls minus asian')
    results.append(result)
    print(f"Non-Asian: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# --------------------------------------------------
# ROBUSTNESS: Functional Form
# --------------------------------------------------

print("\n--- Functional Form Variations ---")

# Standardized outcome
df_dict_sample['amount_std'] = (df_dict_sample['amount'] - df_dict_sample['amount'].mean()) / df_dict_sample['amount'].std()
formula_std = 'amount_std ~ height + ' + ' + '.join(controls_t1)
result = run_ols_clustered(formula_std, df_dict_sample, 'population',
                           'robust/form/y_standardized', 'robustness/functional_form.md',
                           'amount_std', treatment_dictator,
                           sample_desc='Dictator game participants',
                           controls_desc='Baseline controls')
results.append(result)
print(f"Standardized Y: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Quadratic in treatment
formula_quad = f'{outcome_dictator} ~ height + I(height**2) + ' + ' + '.join(controls_t1)
result = run_ols_clustered(formula_quad, df_dict_sample, 'population',
                           'robust/form/quadratic', 'robustness/functional_form.md',
                           outcome_dictator, treatment_dictator,
                           sample_desc='Dictator game participants',
                           controls_desc='Baseline controls + height squared')
results.append(result)
print(f"Quadratic: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")


# ============================================================================
# Analysis 2: Earnings Analysis (Tables 5 & 6)
# ============================================================================

print("\n" + "="*70)
print("ANALYSIS 2: SIMULATED EARNINGS")
print("="*70)

outcome_earnings = 'receivesum'
treatment_earnings = 'Height'  # Note: capital H in this dataset

# Controls for Table 5
controls_t5 = ['asian', 'Shy', 'Confident', 'onlychild', 'Optimistic', 'braces', 'glasses']

# Additional controls for Table 6
controls_t6 = controls_t5 + ['popular', 'between', 'close', 'Power']

# --------------------------------------------------
# BASELINE SPECIFICATIONS (Tables 5 & 6)
# --------------------------------------------------

print("\n--- Baseline Replication (Table 5) ---")

# Table 5
formula_t5 = f'{outcome_earnings} ~ Height + ' + ' + '.join(controls_t5)
result = run_ols_robust(formula_t5, df_earnings,
                        'baseline_earnings', 'methods/cross_sectional_ols.md',
                        outcome_earnings, treatment_earnings,
                        sample_desc='Simulated earnings sample',
                        controls_desc='Demographics (Table 5)')
results.append(result)
print(f"Table 5: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

print("\n--- Baseline with Network Controls (Table 6) ---")

# Table 6
formula_t6 = f'{outcome_earnings} ~ Height + ' + ' + '.join(controls_t6)
result = run_ols_robust(formula_t6, df_earnings,
                        'baseline_earnings_network', 'methods/cross_sectional_ols.md',
                        outcome_earnings, treatment_earnings,
                        sample_desc='Simulated earnings sample',
                        controls_desc='Demographics and network variables (Table 6)')
results.append(result)
print(f"Table 6: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# --------------------------------------------------
# EARNINGS: Leave-One-Out
# --------------------------------------------------

print("\n--- Leave-One-Out (Earnings) ---")

for control in controls_t5:
    remaining = [c for c in controls_t5 if c != control]
    formula_loo = f'{outcome_earnings} ~ Height + ' + ' + '.join(remaining)
    result = run_ols_robust(formula_loo, df_earnings,
                            f'robust/loo_earnings/drop_{control}', 'robustness/leave_one_out.md',
                            outcome_earnings, treatment_earnings,
                            sample_desc='Simulated earnings sample',
                            controls_desc=f'Table 5 controls minus {control}')
    results.append(result)
    print(f"Drop {control}: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# --------------------------------------------------
# EARNINGS: Single Covariate
# --------------------------------------------------

print("\n--- Single Covariate (Earnings) ---")

# Bivariate
formula_biv_earn = f'{outcome_earnings} ~ Height'
result = run_ols_robust(formula_biv_earn, df_earnings,
                        'robust/single_earnings/none', 'robustness/single_covariate.md',
                        outcome_earnings, treatment_earnings,
                        sample_desc='Simulated earnings sample',
                        controls_desc='None')
results.append(result)
print(f"Bivariate: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

for control in controls_t5:
    formula_single = f'{outcome_earnings} ~ Height + {control}'
    result = run_ols_robust(formula_single, df_earnings,
                            f'robust/single_earnings/{control}', 'robustness/single_covariate.md',
                            outcome_earnings, treatment_earnings,
                            sample_desc='Simulated earnings sample',
                            controls_desc=f'Single control: {control}')
    results.append(result)
    print(f"Single {control}: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# --------------------------------------------------
# EARNINGS: Functional Form
# --------------------------------------------------

print("\n--- Functional Form (Earnings) ---")

# Log outcome (add small constant to handle zeros)
df_earnings['log_receivesum'] = np.log(df_earnings['receivesum'] + 0.01)
formula_log = 'log_receivesum ~ Height + ' + ' + '.join(controls_t5)
result = run_ols_robust(formula_log, df_earnings,
                        'robust/form_earnings/y_log', 'robustness/functional_form.md',
                        'log_receivesum', treatment_earnings,
                        sample_desc='Simulated earnings sample',
                        controls_desc='Table 5 controls, log outcome')
results.append(result)
print(f"Log outcome: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Standardized outcome
df_earnings['receivesum_std'] = (df_earnings['receivesum'] - df_earnings['receivesum'].mean()) / df_earnings['receivesum'].std()
formula_std_earn = 'receivesum_std ~ Height + ' + ' + '.join(controls_t5)
result = run_ols_robust(formula_std_earn, df_earnings,
                        'robust/form_earnings/y_standardized', 'robustness/functional_form.md',
                        'receivesum_std', treatment_earnings,
                        sample_desc='Simulated earnings sample',
                        controls_desc='Table 5 controls, standardized outcome')
results.append(result)
print(f"Standardized Y: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")


# ============================================================================
# Analysis 3: Friendship/Linking (Tables 3 & 4) - Conditional Logit
# ============================================================================

print("\n" + "="*70)
print("ANALYSIS 3: FRIENDSHIP FORMATION - Conditional Logit")
print("="*70)

# The linking analysis uses conditional logit (clogit in Stata)
# In Python, we can approximate this with grouped fixed effects logit

outcome_link = 'link_dr'
treatment_link = 'shy_recipient'  # One of the key treatment variables

# Controls for Table 3
controls_t3 = ['samerace', 'sameheight', 'sameconf', 'sameboyfriend', 'height_partner']

# Additional controls for Table 4
controls_t4 = controls_t3 + ['d_dr_2', 'd_dr_3', 'd_dr_4']

# Drop missing values
df_link_sample = df_linking.dropna(subset=[outcome_link, treatment_link] + controls_t3)
print(f"Linking sample: {len(df_link_sample)} observations")

# Note: For true conditional logit we would need specialized software
# We approximate with fixed effects logit using population dummies
# However, this may have convergence issues with many groups

# Simple logit with population fixed effects (approximation)
print("\n--- Logit Approximation (without group FE for simplicity) ---")

# Run standard logit as approximation
formula_logit = f'{outcome_link} ~ shy_recipient + ' + ' + '.join(controls_t3)
try:
    model = smf.logit(formula_logit, data=df_link_sample).fit(disp=0)
    result = extract_results(model, outcome_link, treatment_link,
                            'discrete/binary/logit', 'methods/discrete_choice.md',
                            sample_desc='Friendship pairs',
                            controls_desc='Pair characteristics (Table 3)',
                            model_type='Logit')
    results.append(result)
    print(f"Logit (Table 3): coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"Logit failed: {e}")

# Logit with additional controls (Table 4)
df_link_t4 = df_linking.dropna(subset=[outcome_link, treatment_link] + controls_t4)
formula_logit_t4 = f'{outcome_link} ~ shy_recipient + ' + ' + '.join(controls_t4)
try:
    model = smf.logit(formula_logit_t4, data=df_link_t4).fit(disp=0)
    result = extract_results(model, outcome_link, treatment_link,
                            'discrete/binary/logit_extended', 'methods/discrete_choice.md',
                            sample_desc='Friendship pairs',
                            controls_desc='Pair characteristics + friend-of-friend (Table 4)',
                            model_type='Logit')
    results.append(result)
    print(f"Logit (Table 4): coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"Logit T4 failed: {e}")

# Probit comparison
print("\n--- Probit Comparison ---")
try:
    model = smf.probit(formula_logit, data=df_link_sample).fit(disp=0)
    result = extract_results(model, outcome_link, treatment_link,
                            'discrete/binary/probit', 'methods/discrete_choice.md#model-type-binary-outcome',
                            sample_desc='Friendship pairs',
                            controls_desc='Pair characteristics (Table 3)',
                            model_type='Probit')
    results.append(result)
    print(f"Probit: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"Probit failed: {e}")

# Linear Probability Model
print("\n--- Linear Probability Model ---")
result = run_ols_robust(formula_logit, df_link_sample,
                        'discrete/binary/lpm', 'methods/discrete_choice.md#model-type-binary-outcome',
                        outcome_link, treatment_link,
                        sample_desc='Friendship pairs',
                        controls_desc='Pair characteristics (Table 3)',
                        model_type='LPM')
results.append(result)
print(f"LPM: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# LPM with clustering
result = run_ols_clustered(formula_logit, df_link_sample, 'population',
                           'discrete/binary/lpm_clustered', 'methods/discrete_choice.md',
                           outcome_link, treatment_link,
                           sample_desc='Friendship pairs',
                           controls_desc='Pair characteristics, clustered by population',
                           model_type='LPM')
results.append(result)
print(f"LPM clustered: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# --------------------------------------------------
# LINKING: Leave-One-Out
# --------------------------------------------------

print("\n--- Leave-One-Out (Linking) ---")

for control in controls_t3:
    remaining = [c for c in controls_t3 if c != control]
    formula_loo = f'{outcome_link} ~ shy_recipient + ' + ' + '.join(remaining)
    result = run_ols_clustered(formula_loo, df_link_sample, 'population',
                               f'robust/loo_linking/drop_{control}', 'robustness/leave_one_out.md',
                               outcome_link, treatment_link,
                               sample_desc='Friendship pairs',
                               controls_desc=f'Table 3 controls minus {control}',
                               model_type='LPM')
    results.append(result)
    print(f"Drop {control}: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")


# ============================================================================
# Save Results
# ============================================================================

print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
output_file = OUTPUT_PATH + 'specification_results.csv'
results_df.to_csv(output_file, index=False)
print(f"Saved {len(results_df)} specifications to {output_file}")

# Summary statistics
print("\n--- Summary Statistics ---")
print(f"Total specifications: {len(results_df)}")
print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")
print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")
print(f"Coefficient range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")

print("\n--- By Analysis ---")
for analysis in ['dictator', 'earnings', 'linking', 'discrete']:
    subset = results_df[results_df['spec_id'].str.contains(analysis, case=False) |
                        results_df['sample_desc'].str.contains(analysis, case=False, na=False)]
    if len(subset) == 0:
        # Try to identify by outcome variable
        if analysis == 'dictator':
            subset = results_df[results_df['outcome_var'] == 'amount']
        elif analysis == 'earnings':
            subset = results_df[results_df['outcome_var'].str.contains('receivesum', case=False, na=False)]
        elif analysis in ['linking', 'discrete']:
            subset = results_df[results_df['outcome_var'] == 'link_dr']

    if len(subset) > 0:
        sig_pct = 100 * (subset['p_value'] < 0.05).mean()
        print(f"{analysis}: {len(subset)} specs, {sig_pct:.1f}% significant at 5%")

print("\nDone!")
