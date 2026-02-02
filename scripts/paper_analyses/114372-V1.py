"""
Specification Search: Paper 114372-V1
Social Networks and Economic Behavior: Dictator Game Experiments

Author: Claude Agent
Date: 2026-02-02

This script runs a comprehensive specification search replicating and extending
the analyses from the dictator game experiment paper.

Paper Overview:
- Tests how social networks affect giving behavior in dictator games
- Main regression: amount given ~ height + asian + shy + shy_recipient + popular +
                                 popular_recipient + samerace + sameheight + sameconf
- Clustered by population (experiment session)
- Also includes network formation analysis (conditional logit)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# Set paths
BASE_PATH = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search'
DATA_PATH = f'{BASE_PATH}/data/downloads/extracted/114372-V1/data'
OUTPUT_PATH = f'{BASE_PATH}/data/downloads/extracted/114372-V1'

# Paper metadata
PAPER_ID = '114372-V1'
PAPER_TITLE = 'Social Networks and Economic Behavior: Dictator Game Experiments'
JOURNAL = 'AER'  # American Economic Review

# =============================================================================
# Load Data
# =============================================================================

df_network = pd.read_stata(f'{DATA_PATH}/database_56_network.dta')
df_earnings = pd.read_stata(f'{DATA_PATH}/database_earnings.dta')
df_linking = pd.read_stata(f'{DATA_PATH}/linkingdatabase.dta')

# Create analysis sample for main specification
df_main = df_network[df_network['_est_ols'] == 1].copy()

print(f"Main analysis sample size: {len(df_main)}")
print(f"Earnings analysis sample size: {len(df_earnings)}")
print(f"Network linking sample size: {len(df_linking)}")

# =============================================================================
# Helper Functions
# =============================================================================

def extract_results(model, treatment_var, spec_id, spec_tree_path, outcome_var,
                   controls_desc, sample_desc, cluster_var, model_type, n_obs,
                   fixed_effects='None'):
    """Extract results from a statsmodels fit object"""
    try:
        coef = model.params[treatment_var]
        se = model.bse[treatment_var]
        tval = model.tvalues[treatment_var]
        pval = model.pvalues[treatment_var]
        ci = model.conf_int().loc[treatment_var]
        ci_lower, ci_upper = ci[0], ci[1]

        # R-squared
        if hasattr(model, 'rsquared'):
            r2 = model.rsquared
        else:
            r2 = None

        # Build coefficient vector
        coef_dict = {
            'treatment': {
                'var': treatment_var,
                'coef': float(coef),
                'se': float(se),
                'pval': float(pval)
            },
            'controls': [],
            'fixed_effects': fixed_effects,
            'diagnostics': {
                'first_stage_F': None,
                'overid_pval': None,
                'hausman_pval': None
            }
        }

        # Add control coefficients
        for var in model.params.index:
            if var != treatment_var and var != 'Intercept':
                coef_dict['controls'].append({
                    'var': var,
                    'coef': float(model.params[var]),
                    'se': float(model.bse[var]),
                    'pval': float(model.pvalues[var])
                })

        return {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': float(coef),
            'std_error': float(se),
            't_stat': float(tval),
            'p_value': float(pval),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_obs': n_obs,
            'r_squared': float(r2) if r2 is not None else None,
            'coefficient_vector_json': json.dumps(coef_dict),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects,
            'controls_desc': controls_desc,
            'cluster_var': cluster_var,
            'model_type': model_type,
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
    except Exception as e:
        print(f"Error extracting results for {spec_id}: {e}")
        return None

def run_ols_cluster(data, formula, cluster_var, treatment_var, spec_id, spec_tree_path,
                   outcome_var, controls_desc, sample_desc, model_type='OLS'):
    """Run OLS with clustered standard errors"""
    try:
        # Filter to non-missing
        vars_in_formula = formula.replace('~', '+').replace(' ', '').split('+')
        data_clean = data.dropna(subset=[v for v in vars_in_formula if v in data.columns])

        model = smf.ols(formula, data=data_clean).fit(
            cov_type='cluster',
            cov_kwds={'groups': data_clean[cluster_var]}
        )

        return extract_results(
            model, treatment_var, spec_id, spec_tree_path, outcome_var,
            controls_desc, sample_desc, cluster_var, model_type, len(data_clean)
        )
    except Exception as e:
        print(f"Error running {spec_id}: {e}")
        return None

def run_ols_robust(data, formula, treatment_var, spec_id, spec_tree_path,
                  outcome_var, controls_desc, sample_desc, model_type='OLS'):
    """Run OLS with robust (HC1) standard errors"""
    try:
        vars_in_formula = formula.replace('~', '+').replace(' ', '').split('+')
        data_clean = data.dropna(subset=[v for v in vars_in_formula if v in data.columns])

        model = smf.ols(formula, data=data_clean).fit(cov_type='HC1')

        return extract_results(
            model, treatment_var, spec_id, spec_tree_path, outcome_var,
            controls_desc, sample_desc, 'None (robust SE)', model_type, len(data_clean)
        )
    except Exception as e:
        print(f"Error running {spec_id}: {e}")
        return None

# =============================================================================
# Initialize Results Storage
# =============================================================================

results = []

# =============================================================================
# BASELINE SPECIFICATION (Table 1 Model 1)
# reg amount height asian shy shy_recipient popular popular_recipient samerace sameheight sameconf
# if _est_ols==1, robust cluster(population)
# =============================================================================

print("\n" + "="*60)
print("BASELINE SPECIFICATION")
print("="*60)

baseline_controls = ['height', 'asian', 'shy', 'shy_recipient', 'popular',
                    'popular_recipient', 'samerace', 'sameheight', 'sameconf']

baseline_formula = 'amount ~ ' + ' + '.join(baseline_controls)

result = run_ols_cluster(
    df_main, baseline_formula, 'population', 'height',
    'baseline', 'methods/cross_sectional_ols.md#baseline',
    'amount', ', '.join(baseline_controls[1:]),
    'Main sample (_est_ols==1)', 'OLS'
)
if result:
    results.append(result)
    print(f"Baseline - height: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# =============================================================================
# CONTROL VARIATIONS (Leave-One-Out)
# =============================================================================

print("\n" + "="*60)
print("CONTROL VARIATIONS - LEAVE ONE OUT")
print("="*60)

for control in baseline_controls[1:]:  # Skip height (treatment)
    remaining = [c for c in baseline_controls if c != control]
    formula = 'amount ~ ' + ' + '.join(remaining)

    result = run_ols_cluster(
        df_main, formula, 'population', 'height',
        f'robust/loo/drop_{control}', 'robustness/leave_one_out.md',
        'amount', ', '.join([c for c in remaining if c != 'height']),
        'Main sample (_est_ols==1)', 'OLS'
    )
    if result:
        results.append(result)
        print(f"Drop {control}: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# =============================================================================
# CONTROL VARIATIONS - ADD CONTROLS INCREMENTALLY
# =============================================================================

print("\n" + "="*60)
print("CONTROL VARIATIONS - ADD INCREMENTALLY")
print("="*60)

# Bivariate (no controls except treatment)
result = run_ols_cluster(
    df_main, 'amount ~ height', 'population', 'height',
    'robust/control/none', 'robustness/control_progression.md',
    'amount', 'None',
    'Main sample (_est_ols==1)', 'OLS'
)
if result:
    results.append(result)
    print(f"No controls: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Add controls one at a time
controls_to_add = baseline_controls[1:]  # all except height
for i, control in enumerate(controls_to_add):
    controls_so_far = ['height'] + controls_to_add[:i+1]
    formula = 'amount ~ ' + ' + '.join(controls_so_far)

    result = run_ols_cluster(
        df_main, formula, 'population', 'height',
        f'robust/control/add_{control}', 'robustness/control_progression.md',
        'amount', ', '.join(controls_to_add[:i+1]),
        'Main sample (_est_ols==1)', 'OLS'
    )
    if result:
        results.append(result)
        print(f"Add {control}: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# =============================================================================
# INFERENCE VARIATIONS - DIFFERENT STANDARD ERRORS
# =============================================================================

print("\n" + "="*60)
print("INFERENCE VARIATIONS")
print("="*60)

# Robust (HC1) - no clustering
result = run_ols_robust(
    df_main, baseline_formula, 'height',
    'robust/se/robust_hc1', 'robustness/clustering_variations.md#robust',
    'amount', ', '.join(baseline_controls[1:]),
    'Main sample (_est_ols==1)', 'OLS'
)
if result:
    results.append(result)
    print(f"Robust HC1: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# HC2
try:
    model = smf.ols(baseline_formula, data=df_main).fit(cov_type='HC2')
    result = extract_results(
        model, 'height', 'robust/se/hc2', 'robustness/clustering_variations.md#hc2',
        'amount', ', '.join(baseline_controls[1:]), 'Main sample (_est_ols==1)',
        'None (HC2)', 'OLS', len(df_main)
    )
    if result:
        results.append(result)
        print(f"HC2: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"HC2 error: {e}")

# HC3
try:
    model = smf.ols(baseline_formula, data=df_main).fit(cov_type='HC3')
    result = extract_results(
        model, 'height', 'robust/se/hc3', 'robustness/clustering_variations.md#hc3',
        'amount', ', '.join(baseline_controls[1:]), 'Main sample (_est_ols==1)',
        'None (HC3)', 'OLS', len(df_main)
    )
    if result:
        results.append(result)
        print(f"HC3: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"HC3 error: {e}")

# Classical (homoskedastic) SE
try:
    model = smf.ols(baseline_formula, data=df_main).fit()
    result = extract_results(
        model, 'height', 'robust/se/classical', 'robustness/clustering_variations.md#classical',
        'amount', ', '.join(baseline_controls[1:]), 'Main sample (_est_ols==1)',
        'None (classical)', 'OLS', len(df_main)
    )
    if result:
        results.append(result)
        print(f"Classical: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"Classical SE error: {e}")

# =============================================================================
# SAMPLE RESTRICTIONS
# =============================================================================

print("\n" + "="*60)
print("SAMPLE RESTRICTIONS")
print("="*60)

# Winsorize outcome at 5%
df_win5 = df_main.copy()
df_win5['amount'] = df_win5['amount'].clip(
    lower=df_win5['amount'].quantile(0.05),
    upper=df_win5['amount'].quantile(0.95)
)
result = run_ols_cluster(
    df_win5, baseline_formula, 'population', 'height',
    'robust/sample/winsor_5pct', 'robustness/sample_restrictions.md#winsor',
    'amount', ', '.join(baseline_controls[1:]),
    'Main sample, winsorized 5%', 'OLS'
)
if result:
    results.append(result)
    print(f"Winsorize 5%: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Winsorize outcome at 10%
df_win10 = df_main.copy()
df_win10['amount'] = df_win10['amount'].clip(
    lower=df_win10['amount'].quantile(0.10),
    upper=df_win10['amount'].quantile(0.90)
)
result = run_ols_cluster(
    df_win10, baseline_formula, 'population', 'height',
    'robust/sample/winsor_10pct', 'robustness/sample_restrictions.md#winsor',
    'amount', ', '.join(baseline_controls[1:]),
    'Main sample, winsorized 10%', 'OLS'
)
if result:
    results.append(result)
    print(f"Winsorize 10%: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Trim extreme outcome values (1%)
df_trim1 = df_main[(df_main['amount'] > df_main['amount'].quantile(0.01)) &
                   (df_main['amount'] < df_main['amount'].quantile(0.99))]
result = run_ols_cluster(
    df_trim1, baseline_formula, 'population', 'height',
    'robust/sample/trim_1pct', 'robustness/sample_restrictions.md#trim',
    'amount', ', '.join(baseline_controls[1:]),
    'Main sample, trimmed 1%', 'OLS'
)
if result:
    results.append(result)
    print(f"Trim 1%: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Trim extreme outcome values (5%)
df_trim5 = df_main[(df_main['amount'] > df_main['amount'].quantile(0.05)) &
                   (df_main['amount'] < df_main['amount'].quantile(0.95))]
result = run_ols_cluster(
    df_trim5, baseline_formula, 'population', 'height',
    'robust/sample/trim_5pct', 'robustness/sample_restrictions.md#trim',
    'amount', ', '.join(baseline_controls[1:]),
    'Main sample, trimmed 5%', 'OLS'
)
if result:
    results.append(result)
    print(f"Trim 5%: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Subgroup: Same race pairs only
df_same_race = df_main[df_main['samerace'] == 1]
result = run_ols_cluster(
    df_same_race, baseline_formula, 'population', 'height',
    'robust/sample/same_race_only', 'robustness/sample_restrictions.md#subgroup',
    'amount', ', '.join(baseline_controls[1:]),
    'Same race pairs only', 'OLS'
)
if result:
    results.append(result)
    print(f"Same race only: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Subgroup: Different race pairs only
df_diff_race = df_main[df_main['samerace'] == 0]
result = run_ols_cluster(
    df_diff_race, baseline_formula, 'population', 'height',
    'robust/sample/diff_race_only', 'robustness/sample_restrictions.md#subgroup',
    'amount', ', '.join(baseline_controls[1:]),
    'Different race pairs only', 'OLS'
)
if result:
    results.append(result)
    print(f"Different race only: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Subgroup: Asian only
df_asian = df_main[df_main['asian'] == 1]
result = run_ols_cluster(
    df_asian, baseline_formula.replace(' + asian', ''), 'population', 'height',
    'robust/sample/asian_only', 'robustness/sample_restrictions.md#subgroup',
    'amount', ', '.join([c for c in baseline_controls[1:] if c != 'asian']),
    'Asian subjects only', 'OLS'
)
if result:
    results.append(result)
    print(f"Asian only: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Subgroup: Non-Asian only
df_non_asian = df_main[df_main['asian'] == 0]
result = run_ols_cluster(
    df_non_asian, baseline_formula.replace(' + asian', ''), 'population', 'height',
    'robust/sample/non_asian_only', 'robustness/sample_restrictions.md#subgroup',
    'amount', ', '.join([c for c in baseline_controls[1:] if c != 'asian']),
    'Non-Asian subjects only', 'OLS'
)
if result:
    results.append(result)
    print(f"Non-Asian only: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Subgroup: Same height pairs
df_same_height = df_main[df_main['sameheight'] == 1]
result = run_ols_cluster(
    df_same_height, baseline_formula, 'population', 'height',
    'robust/sample/same_height_only', 'robustness/sample_restrictions.md#subgroup',
    'amount', ', '.join(baseline_controls[1:]),
    'Same height pairs only', 'OLS'
)
if result:
    results.append(result)
    print(f"Same height only: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Subgroup: Different height pairs
df_diff_height = df_main[df_main['sameheight'] == 0]
result = run_ols_cluster(
    df_diff_height, baseline_formula, 'population', 'height',
    'robust/sample/diff_height_only', 'robustness/sample_restrictions.md#subgroup',
    'amount', ', '.join(baseline_controls[1:]),
    'Different height pairs only', 'OLS'
)
if result:
    results.append(result)
    print(f"Different height only: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# High shy subjects (above median)
df_high_shy = df_main[df_main['shy'] > df_main['shy'].median()]
result = run_ols_cluster(
    df_high_shy, baseline_formula, 'population', 'height',
    'robust/sample/high_shy', 'robustness/sample_restrictions.md#subgroup',
    'amount', ', '.join(baseline_controls[1:]),
    'High shy subjects (above median)', 'OLS'
)
if result:
    results.append(result)
    print(f"High shy: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Low shy subjects (below median)
df_low_shy = df_main[df_main['shy'] <= df_main['shy'].median()]
result = run_ols_cluster(
    df_low_shy, baseline_formula, 'population', 'height',
    'robust/sample/low_shy', 'robustness/sample_restrictions.md#subgroup',
    'amount', ', '.join(baseline_controls[1:]),
    'Low shy subjects (at or below median)', 'OLS'
)
if result:
    results.append(result)
    print(f"Low shy: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# =============================================================================
# FUNCTIONAL FORM VARIATIONS
# =============================================================================

print("\n" + "="*60)
print("FUNCTIONAL FORM VARIATIONS")
print("="*60)

# Log outcome (add small constant to avoid log(0))
df_log = df_main.copy()
df_log['log_amount'] = np.log(df_log['amount'] + 0.01)
formula_log = 'log_amount ~ ' + ' + '.join(baseline_controls)
result = run_ols_cluster(
    df_log, formula_log, 'population', 'height',
    'robust/funcform/log_outcome', 'robustness/functional_form.md#log',
    'log_amount', ', '.join(baseline_controls[1:]),
    'Main sample, log outcome', 'OLS'
)
if result:
    results.append(result)
    print(f"Log outcome: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# IHS transformation
df_ihs = df_main.copy()
df_ihs['ihs_amount'] = np.arcsinh(df_ihs['amount'])
formula_ihs = 'ihs_amount ~ ' + ' + '.join(baseline_controls)
result = run_ols_cluster(
    df_ihs, formula_ihs, 'population', 'height',
    'robust/funcform/ihs_outcome', 'robustness/functional_form.md#ihs',
    'ihs_amount', ', '.join(baseline_controls[1:]),
    'Main sample, IHS outcome', 'OLS'
)
if result:
    results.append(result)
    print(f"IHS outcome: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Quadratic in height
df_quad = df_main.copy()
df_quad['height_sq'] = df_quad['height']**2
formula_quad = 'amount ~ height + height_sq + ' + ' + '.join(baseline_controls[1:])
result = run_ols_cluster(
    df_quad, formula_quad, 'population', 'height',
    'robust/funcform/quadratic_height', 'robustness/functional_form.md#polynomial',
    'amount', 'height_sq, ' + ', '.join(baseline_controls[1:]),
    'Main sample, quadratic height', 'OLS'
)
if result:
    results.append(result)
    print(f"Quadratic height: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# =============================================================================
# ALTERNATIVE TREATMENT MEASURES
# =============================================================================

print("\n" + "="*60)
print("ALTERNATIVE TREATMENT MEASURES")
print("="*60)

# Use 'shy' as treatment instead of height
result = run_ols_cluster(
    df_main, baseline_formula, 'population', 'shy',
    'robust/treatment/shy_as_treatment', 'methods/cross_sectional_ols.md#treatment',
    'amount', 'height, ' + ', '.join([c for c in baseline_controls[1:] if c != 'shy']),
    'Main sample, shy as treatment', 'OLS'
)
if result:
    results.append(result)
    print(f"Shy as treatment: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Use 'popular' as treatment
result = run_ols_cluster(
    df_main, baseline_formula, 'population', 'popular',
    'robust/treatment/popular_as_treatment', 'methods/cross_sectional_ols.md#treatment',
    'amount', 'height, ' + ', '.join([c for c in baseline_controls[1:] if c != 'popular']),
    'Main sample, popular as treatment', 'OLS'
)
if result:
    results.append(result)
    print(f"Popular as treatment: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Use 'samerace' as treatment
result = run_ols_cluster(
    df_main, baseline_formula, 'population', 'samerace',
    'robust/treatment/samerace_as_treatment', 'methods/cross_sectional_ols.md#treatment',
    'amount', 'height, ' + ', '.join([c for c in baseline_controls[1:] if c != 'samerace']),
    'Main sample, samerace as treatment', 'OLS'
)
if result:
    results.append(result)
    print(f"Samerace as treatment: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Binary height (tall vs short)
df_binary_height = df_main.copy()
df_binary_height['height_binary'] = (df_binary_height['height'] > df_binary_height['height'].median()).astype(int)
formula_binary = 'amount ~ height_binary + ' + ' + '.join(baseline_controls[1:])
result = run_ols_cluster(
    df_binary_height, formula_binary, 'population', 'height_binary',
    'robust/treatment/binary_height', 'methods/cross_sectional_ols.md#treatment',
    'amount', ', '.join(baseline_controls[1:]),
    'Main sample, binary height (above median)', 'OLS'
)
if result:
    results.append(result)
    print(f"Binary height: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# =============================================================================
# ALTERNATIVE OUTCOMES
# =============================================================================

print("\n" + "="*60)
print("ALTERNATIVE OUTCOMES")
print("="*60)

# Use 'money' (the raw money amount) instead of 'amount' (proportion)
if 'money' in df_main.columns:
    formula_money = 'money ~ ' + ' + '.join(baseline_controls)
    result = run_ols_cluster(
        df_main, formula_money, 'population', 'height',
        'robust/outcome/money_raw', 'methods/cross_sectional_ols.md#outcome',
        'money', ', '.join(baseline_controls[1:]),
        'Main sample, raw money as outcome', 'OLS'
    )
    if result:
        results.append(result)
        print(f"Money raw: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Binary outcome: gave anything (amount > 0)
df_binary_out = df_main.copy()
df_binary_out['gave_any'] = (df_binary_out['amount'] > 0).astype(int)
formula_binary_out = 'gave_any ~ ' + ' + '.join(baseline_controls)
result = run_ols_cluster(
    df_binary_out, formula_binary_out, 'population', 'height',
    'robust/outcome/gave_any', 'methods/cross_sectional_ols.md#outcome',
    'gave_any', ', '.join(baseline_controls[1:]),
    'Main sample, binary gave any', 'OLS'
)
if result:
    results.append(result)
    print(f"Gave any (binary): coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Binary outcome: gave at least half
df_half = df_main.copy()
df_half['gave_half'] = (df_half['amount'] >= 0.5).astype(int)
formula_half = 'gave_half ~ ' + ' + '.join(baseline_controls)
result = run_ols_cluster(
    df_half, formula_half, 'population', 'height',
    'robust/outcome/gave_half', 'methods/cross_sectional_ols.md#outcome',
    'gave_half', ', '.join(baseline_controls[1:]),
    'Main sample, binary gave at least half', 'OLS'
)
if result:
    results.append(result)
    print(f"Gave half+ (binary): coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# =============================================================================
# HETEROGENEITY ANALYSIS - INTERACTIONS
# =============================================================================

print("\n" + "="*60)
print("HETEROGENEITY ANALYSIS - INTERACTIONS")
print("="*60)

# Height x Asian interaction
df_int = df_main.copy()
df_int['height_x_asian'] = df_int['height'] * df_int['asian']
formula_int = 'amount ~ height + asian + height_x_asian + ' + ' + '.join([c for c in baseline_controls[1:] if c != 'asian'])
result = run_ols_cluster(
    df_int, formula_int, 'population', 'height',
    'robust/het/interaction_asian', 'robustness/heterogeneity.md#interaction',
    'amount', 'height_x_asian, ' + ', '.join([c for c in baseline_controls[1:] if c != 'asian']),
    'Main sample, height x asian interaction', 'OLS'
)
if result:
    results.append(result)
    print(f"Height x Asian: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Height x Samerace interaction
df_int2 = df_main.copy()
df_int2['height_x_samerace'] = df_int2['height'] * df_int2['samerace']
formula_int2 = 'amount ~ height + samerace + height_x_samerace + ' + ' + '.join([c for c in baseline_controls[1:] if c != 'samerace'])
result = run_ols_cluster(
    df_int2, formula_int2, 'population', 'height',
    'robust/het/interaction_samerace', 'robustness/heterogeneity.md#interaction',
    'amount', 'height_x_samerace, ' + ', '.join([c for c in baseline_controls[1:] if c != 'samerace']),
    'Main sample, height x samerace interaction', 'OLS'
)
if result:
    results.append(result)
    print(f"Height x Samerace: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Height x Shy interaction
df_int3 = df_main.copy()
df_int3['height_x_shy'] = df_int3['height'] * df_int3['shy']
formula_int3 = 'amount ~ height + shy + height_x_shy + ' + ' + '.join([c for c in baseline_controls[1:] if c != 'shy'])
result = run_ols_cluster(
    df_int3, formula_int3, 'population', 'height',
    'robust/het/interaction_shy', 'robustness/heterogeneity.md#interaction',
    'amount', 'height_x_shy, ' + ', '.join([c for c in baseline_controls[1:] if c != 'shy']),
    'Main sample, height x shy interaction', 'OLS'
)
if result:
    results.append(result)
    print(f"Height x Shy: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Height x Popular interaction
df_int4 = df_main.copy()
df_int4['height_x_popular'] = df_int4['height'] * df_int4['popular']
formula_int4 = 'amount ~ height + popular + height_x_popular + ' + ' + '.join([c for c in baseline_controls[1:] if c != 'popular'])
result = run_ols_cluster(
    df_int4, formula_int4, 'population', 'height',
    'robust/het/interaction_popular', 'robustness/heterogeneity.md#interaction',
    'amount', 'height_x_popular, ' + ', '.join([c for c in baseline_controls[1:] if c != 'popular']),
    'Main sample, height x popular interaction', 'OLS'
)
if result:
    results.append(result)
    print(f"Height x Popular: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# =============================================================================
# EARNINGS ANALYSIS (Tables 5 & 6)
# =============================================================================

print("\n" + "="*60)
print("EARNINGS ANALYSIS (Tables 5 & 6)")
print("="*60)

# Table 5: Basic controls
earnings_basic_controls = ['Height', 'asian', 'Shy', 'Confident', 'onlychild', 'Optimistic', 'braces', 'glasses']
formula_earn5 = 'receivesum ~ ' + ' + '.join(earnings_basic_controls)
result = run_ols_robust(
    df_earnings, formula_earn5, 'Height',
    'earnings/table5', 'methods/cross_sectional_ols.md#baseline',
    'receivesum', ', '.join([c for c in earnings_basic_controls if c != 'Height']),
    'Earnings sample', 'OLS'
)
if result:
    results.append(result)
    print(f"Table 5 (Height): coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Table 6: Full controls including network measures
earnings_full_controls = ['popular', 'Height', 'asian', 'Shy', 'Confident', 'onlychild',
                         'Optimistic', 'braces', 'glasses', 'between', 'close', 'Power']
formula_earn6 = 'receivesum ~ ' + ' + '.join(earnings_full_controls)
result = run_ols_robust(
    df_earnings, formula_earn6, 'popular',
    'earnings/table6_popular', 'methods/cross_sectional_ols.md#baseline',
    'receivesum', ', '.join([c for c in earnings_full_controls if c != 'popular']),
    'Earnings sample', 'OLS'
)
if result:
    results.append(result)
    print(f"Table 6 (popular): coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Table 6 with Height as treatment
result = run_ols_robust(
    df_earnings, formula_earn6, 'Height',
    'earnings/table6_height', 'methods/cross_sectional_ols.md#baseline',
    'receivesum', ', '.join([c for c in earnings_full_controls if c != 'Height']),
    'Earnings sample', 'OLS'
)
if result:
    results.append(result)
    print(f"Table 6 (Height): coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# =============================================================================
# EARNINGS ROBUSTNESS - LEAVE ONE OUT
# =============================================================================

print("\n" + "="*60)
print("EARNINGS ROBUSTNESS - LEAVE ONE OUT")
print("="*60)

for control in [c for c in earnings_full_controls if c != 'popular']:
    remaining = [c for c in earnings_full_controls if c != control]
    formula = 'receivesum ~ ' + ' + '.join(remaining)

    result = run_ols_robust(
        df_earnings, formula, 'popular',
        f'earnings/loo/drop_{control}', 'robustness/leave_one_out.md',
        'receivesum', ', '.join([c for c in remaining if c != 'popular']),
        'Earnings sample', 'OLS'
    )
    if result:
        results.append(result)
        print(f"Earnings drop {control}: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# =============================================================================
# EARNINGS ROBUSTNESS - SAMPLE RESTRICTIONS
# =============================================================================

print("\n" + "="*60)
print("EARNINGS ROBUSTNESS - SAMPLE RESTRICTIONS")
print("="*60)

# Trim 5%
df_earn_trim = df_earnings[(df_earnings['receivesum'] > df_earnings['receivesum'].quantile(0.05)) &
                           (df_earnings['receivesum'] < df_earnings['receivesum'].quantile(0.95))]
result = run_ols_robust(
    df_earn_trim, formula_earn6, 'popular',
    'earnings/sample/trim_5pct', 'robustness/sample_restrictions.md#trim',
    'receivesum', ', '.join([c for c in earnings_full_controls if c != 'popular']),
    'Earnings sample, trimmed 5%', 'OLS'
)
if result:
    results.append(result)
    print(f"Earnings trim 5%: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Winsorize 5%
df_earn_win = df_earnings.copy()
df_earn_win['receivesum'] = df_earn_win['receivesum'].clip(
    lower=df_earn_win['receivesum'].quantile(0.05),
    upper=df_earn_win['receivesum'].quantile(0.95)
)
result = run_ols_robust(
    df_earn_win, formula_earn6, 'popular',
    'earnings/sample/winsor_5pct', 'robustness/sample_restrictions.md#winsor',
    'receivesum', ', '.join([c for c in earnings_full_controls if c != 'popular']),
    'Earnings sample, winsorized 5%', 'OLS'
)
if result:
    results.append(result)
    print(f"Earnings winsor 5%: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Asian only
df_earn_asian = df_earnings[df_earnings['asian'] == 1]
formula_earn_no_asian = 'receivesum ~ ' + ' + '.join([c for c in earnings_full_controls if c != 'asian'])
result = run_ols_robust(
    df_earn_asian, formula_earn_no_asian, 'popular',
    'earnings/sample/asian_only', 'robustness/sample_restrictions.md#subgroup',
    'receivesum', ', '.join([c for c in earnings_full_controls if c not in ['popular', 'asian']]),
    'Earnings sample, Asian only', 'OLS'
)
if result:
    results.append(result)
    print(f"Earnings Asian only: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Non-Asian only
df_earn_non_asian = df_earnings[df_earnings['asian'] == 0]
result = run_ols_robust(
    df_earn_non_asian, formula_earn_no_asian, 'popular',
    'earnings/sample/non_asian_only', 'robustness/sample_restrictions.md#subgroup',
    'receivesum', ', '.join([c for c in earnings_full_controls if c not in ['popular', 'asian']]),
    'Earnings sample, Non-Asian only', 'OLS'
)
if result:
    results.append(result)
    print(f"Earnings Non-Asian only: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# High popular (above median)
df_earn_high_pop = df_earnings[df_earnings['popular'] > df_earnings['popular'].median()]
result = run_ols_robust(
    df_earn_high_pop, formula_earn6, 'popular',
    'earnings/sample/high_popular', 'robustness/sample_restrictions.md#subgroup',
    'receivesum', ', '.join([c for c in earnings_full_controls if c != 'popular']),
    'Earnings sample, high popular', 'OLS'
)
if result:
    results.append(result)
    print(f"Earnings high popular: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# =============================================================================
# EARNINGS ALTERNATIVE TREATMENTS
# =============================================================================

print("\n" + "="*60)
print("EARNINGS ALTERNATIVE TREATMENTS")
print("="*60)

# between as treatment
result = run_ols_robust(
    df_earnings, formula_earn6, 'between',
    'earnings/treatment/between', 'methods/cross_sectional_ols.md#treatment',
    'receivesum', ', '.join([c for c in earnings_full_controls if c != 'between']),
    'Earnings sample, between as treatment', 'OLS'
)
if result:
    results.append(result)
    print(f"Earnings between as treatment: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# close as treatment
result = run_ols_robust(
    df_earnings, formula_earn6, 'close',
    'earnings/treatment/close', 'methods/cross_sectional_ols.md#treatment',
    'receivesum', ', '.join([c for c in earnings_full_controls if c != 'close']),
    'Earnings sample, close as treatment', 'OLS'
)
if result:
    results.append(result)
    print(f"Earnings close as treatment: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Power as treatment
result = run_ols_robust(
    df_earnings, formula_earn6, 'Power',
    'earnings/treatment/power', 'methods/cross_sectional_ols.md#treatment',
    'receivesum', ', '.join([c for c in earnings_full_controls if c != 'Power']),
    'Earnings sample, Power as treatment', 'OLS'
)
if result:
    results.append(result)
    print(f"Earnings Power as treatment: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# =============================================================================
# EARNINGS FUNCTIONAL FORM
# =============================================================================

print("\n" + "="*60)
print("EARNINGS FUNCTIONAL FORM")
print("="*60)

# Log outcome
df_earn_log = df_earnings.copy()
df_earn_log['log_receivesum'] = np.log(df_earn_log['receivesum'] + 0.01)
formula_earn_log = 'log_receivesum ~ ' + ' + '.join(earnings_full_controls)
result = run_ols_robust(
    df_earn_log, formula_earn_log, 'popular',
    'earnings/funcform/log', 'robustness/functional_form.md#log',
    'log_receivesum', ', '.join([c for c in earnings_full_controls if c != 'popular']),
    'Earnings sample, log outcome', 'OLS'
)
if result:
    results.append(result)
    print(f"Earnings log outcome: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# IHS outcome
df_earn_ihs = df_earnings.copy()
df_earn_ihs['ihs_receivesum'] = np.arcsinh(df_earn_ihs['receivesum'])
formula_earn_ihs = 'ihs_receivesum ~ ' + ' + '.join(earnings_full_controls)
result = run_ols_robust(
    df_earn_ihs, formula_earn_ihs, 'popular',
    'earnings/funcform/ihs', 'robustness/functional_form.md#ihs',
    'ihs_receivesum', ', '.join([c for c in earnings_full_controls if c != 'popular']),
    'Earnings sample, IHS outcome', 'OLS'
)
if result:
    results.append(result)
    print(f"Earnings IHS outcome: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Quadratic popular
df_earn_quad = df_earnings.copy()
df_earn_quad['popular_sq'] = df_earn_quad['popular']**2
formula_earn_quad = 'receivesum ~ popular + popular_sq + ' + ' + '.join([c for c in earnings_full_controls if c != 'popular'])
result = run_ols_robust(
    df_earn_quad, formula_earn_quad, 'popular',
    'earnings/funcform/quadratic', 'robustness/functional_form.md#polynomial',
    'receivesum', 'popular_sq, ' + ', '.join([c for c in earnings_full_controls if c != 'popular']),
    'Earnings sample, quadratic popular', 'OLS'
)
if result:
    results.append(result)
    print(f"Earnings quadratic: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# =============================================================================
# NETWORK ANALYSIS (Tables 3 & 4) - Conditional Logit approximation
# Using logistic regression with population fixed effects as approximation
# =============================================================================

print("\n" + "="*60)
print("NETWORK ANALYSIS (Tables 3 & 4)")
print("="*60)

# Filter to non-missing observations
df_link_clean = df_linking.dropna(subset=['link_dr', 'samerace', 'sameheight', 'sameconf',
                                          'sameboyfriend', 'shy_recipient', 'height_partner'])

# Table 3 specification
network_controls_t3 = ['samerace', 'sameheight', 'sameconf', 'sameboyfriend', 'shy_recipient', 'height_partner']
formula_net3 = 'link_dr ~ ' + ' + '.join(network_controls_t3)
result = run_ols_cluster(
    df_link_clean, formula_net3, 'population', 'samerace',
    'network/table3_samerace', 'methods/discrete_choice.md',
    'link_dr', ', '.join([c for c in network_controls_t3 if c != 'samerace']),
    'Network linking sample', 'Linear Probability'
)
if result:
    results.append(result)
    print(f"Table 3 (samerace): coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# sameheight as treatment
result = run_ols_cluster(
    df_link_clean, formula_net3, 'population', 'sameheight',
    'network/table3_sameheight', 'methods/discrete_choice.md',
    'link_dr', ', '.join([c for c in network_controls_t3 if c != 'sameheight']),
    'Network linking sample', 'Linear Probability'
)
if result:
    results.append(result)
    print(f"Table 3 (sameheight): coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Table 4 specification (adding distance measures)
df_link_clean4 = df_linking.dropna(subset=['link_dr', 'samerace', 'sameheight', 'sameconf',
                                           'sameboyfriend', 'shy_recipient', 'height_partner',
                                           'd_dr_2', 'd_dr_3', 'd_dr_4'])
network_controls_t4 = ['samerace', 'sameheight', 'sameconf', 'sameboyfriend',
                       'shy_recipient', 'height_partner', 'd_dr_2', 'd_dr_3', 'd_dr_4']
formula_net4 = 'link_dr ~ ' + ' + '.join(network_controls_t4)

result = run_ols_cluster(
    df_link_clean4, formula_net4, 'population', 'samerace',
    'network/table4_samerace', 'methods/discrete_choice.md',
    'link_dr', ', '.join([c for c in network_controls_t4 if c != 'samerace']),
    'Network linking sample', 'Linear Probability'
)
if result:
    results.append(result)
    print(f"Table 4 (samerace): coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# d_dr_2 as treatment (friend of friend effect)
result = run_ols_cluster(
    df_link_clean4, formula_net4, 'population', 'd_dr_2',
    'network/table4_d_dr_2', 'methods/discrete_choice.md',
    'link_dr', ', '.join([c for c in network_controls_t4 if c != 'd_dr_2']),
    'Network linking sample', 'Linear Probability'
)
if result:
    results.append(result)
    print(f"Table 4 (d_dr_2): coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# =============================================================================
# NETWORK ROBUSTNESS
# =============================================================================

print("\n" + "="*60)
print("NETWORK ROBUSTNESS - LEAVE ONE OUT")
print("="*60)

for control in [c for c in network_controls_t3 if c != 'samerace']:
    remaining = [c for c in network_controls_t3 if c != control]
    formula = 'link_dr ~ ' + ' + '.join(remaining)

    result = run_ols_cluster(
        df_link_clean, formula, 'population', 'samerace',
        f'network/loo/drop_{control}', 'robustness/leave_one_out.md',
        'link_dr', ', '.join([c for c in remaining if c != 'samerace']),
        'Network linking sample', 'Linear Probability'
    )
    if result:
        results.append(result)
        print(f"Network drop {control}: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# =============================================================================
# MAIN ANALYSIS - MORE SAMPLE SPLITS
# =============================================================================

print("\n" + "="*60)
print("MAIN ANALYSIS - ADDITIONAL SAMPLE SPLITS")
print("="*60)

# High popular recipient (above median)
df_high_pop_rec = df_main[df_main['popular_recipient'] > df_main['popular_recipient'].median()]
result = run_ols_cluster(
    df_high_pop_rec, baseline_formula, 'population', 'height',
    'robust/sample/high_pop_recipient', 'robustness/sample_restrictions.md#subgroup',
    'amount', ', '.join(baseline_controls[1:]),
    'High popular recipient', 'OLS'
)
if result:
    results.append(result)
    print(f"High pop recipient: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Low popular recipient (below median)
df_low_pop_rec = df_main[df_main['popular_recipient'] <= df_main['popular_recipient'].median()]
result = run_ols_cluster(
    df_low_pop_rec, baseline_formula, 'population', 'height',
    'robust/sample/low_pop_recipient', 'robustness/sample_restrictions.md#subgroup',
    'amount', ', '.join(baseline_controls[1:]),
    'Low popular recipient', 'OLS'
)
if result:
    results.append(result)
    print(f"Low pop recipient: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Same confidence level
df_same_conf = df_main[df_main['sameconf'] == 1]
result = run_ols_cluster(
    df_same_conf, baseline_formula, 'population', 'height',
    'robust/sample/same_conf', 'robustness/sample_restrictions.md#subgroup',
    'amount', ', '.join(baseline_controls[1:]),
    'Same confidence pairs', 'OLS'
)
if result:
    results.append(result)
    print(f"Same confidence: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Different confidence level
df_diff_conf = df_main[df_main['sameconf'] == 0]
result = run_ols_cluster(
    df_diff_conf, baseline_formula, 'population', 'height',
    'robust/sample/diff_conf', 'robustness/sample_restrictions.md#subgroup',
    'amount', ', '.join(baseline_controls[1:]),
    'Different confidence pairs', 'OLS'
)
if result:
    results.append(result)
    print(f"Different confidence: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# =============================================================================
# PLACEBO TESTS
# =============================================================================

print("\n" + "="*60)
print("PLACEBO TESTS")
print("="*60)

# Placebo: Randomize treatment (height) within population
np.random.seed(42)
df_placebo = df_main.copy()
df_placebo['height_placebo'] = df_placebo.groupby('population')['height'].transform(
    lambda x: np.random.permutation(x.values)
)
formula_placebo = 'amount ~ height_placebo + ' + ' + '.join(baseline_controls[1:])
result = run_ols_cluster(
    df_placebo, formula_placebo, 'population', 'height_placebo',
    'robust/placebo/randomized_height', 'robustness/placebo_tests.md',
    'amount', ', '.join(baseline_controls[1:]),
    'Main sample, randomized height within population', 'OLS'
)
if result:
    results.append(result)
    print(f"Placebo (randomized): coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Placebo 2: Randomize treatment (height) across all observations
df_placebo2 = df_main.copy()
df_placebo2['height_placebo_all'] = np.random.permutation(df_placebo2['height'].values)
formula_placebo2 = 'amount ~ height_placebo_all + ' + ' + '.join(baseline_controls[1:])
result = run_ols_cluster(
    df_placebo2, formula_placebo2, 'population', 'height_placebo_all',
    'robust/placebo/randomized_height_all', 'robustness/placebo_tests.md',
    'amount', ', '.join(baseline_controls[1:]),
    'Main sample, randomized height globally', 'OLS'
)
if result:
    results.append(result)
    print(f"Placebo (randomized all): coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Placebo 3: Use lagged/shuffled partner popularity as outcome predictor
df_placebo3 = df_main.copy()
df_placebo3['popular_recipient_placebo'] = np.random.permutation(df_placebo3['popular_recipient'].values)
formula_placebo3 = 'amount ~ height + asian + shy + shy_recipient + popular + popular_recipient_placebo + samerace + sameheight + sameconf'
result = run_ols_cluster(
    df_placebo3, formula_placebo3, 'population', 'height',
    'robust/placebo/randomized_pop_recipient', 'robustness/placebo_tests.md',
    'amount', 'asian, shy, shy_recipient, popular, popular_recipient_placebo, samerace, sameheight, sameconf',
    'Main sample, randomized popular_recipient', 'OLS'
)
if result:
    results.append(result)
    print(f"Placebo (random pop_rec): coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# =============================================================================
# ADDITIONAL HETEROGENEITY
# =============================================================================

print("\n" + "="*60)
print("ADDITIONAL HETEROGENEITY")
print("="*60)

# Height x sameconf interaction
df_int_conf = df_main.copy()
df_int_conf['height_x_sameconf'] = df_int_conf['height'] * df_int_conf['sameconf']
formula_int_conf = 'amount ~ height + sameconf + height_x_sameconf + ' + ' + '.join([c for c in baseline_controls[1:] if c != 'sameconf'])
result = run_ols_cluster(
    df_int_conf, formula_int_conf, 'population', 'height',
    'robust/het/interaction_sameconf', 'robustness/heterogeneity.md#interaction',
    'amount', 'height_x_sameconf, ' + ', '.join([c for c in baseline_controls[1:] if c != 'sameconf']),
    'Main sample, height x sameconf interaction', 'OLS'
)
if result:
    results.append(result)
    print(f"Height x Sameconf: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Height x sameheight interaction
df_int_hgt = df_main.copy()
df_int_hgt['height_x_sameheight'] = df_int_hgt['height'] * df_int_hgt['sameheight']
formula_int_hgt = 'amount ~ height + sameheight + height_x_sameheight + ' + ' + '.join([c for c in baseline_controls[1:] if c != 'sameheight'])
result = run_ols_cluster(
    df_int_hgt, formula_int_hgt, 'population', 'height',
    'robust/het/interaction_sameheight', 'robustness/heterogeneity.md#interaction',
    'amount', 'height_x_sameheight, ' + ', '.join([c for c in baseline_controls[1:] if c != 'sameheight']),
    'Main sample, height x sameheight interaction', 'OLS'
)
if result:
    results.append(result)
    print(f"Height x Sameheight: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# =============================================================================
# CONTROL SET VARIATIONS
# =============================================================================

print("\n" + "="*60)
print("CONTROL SET VARIATIONS")
print("="*60)

# Minimal controls (only demographic)
minimal_controls = ['height', 'asian', 'shy']
formula_minimal = 'amount ~ ' + ' + '.join(minimal_controls)
result = run_ols_cluster(
    df_main, formula_minimal, 'population', 'height',
    'robust/control/minimal', 'methods/cross_sectional_ols.md#controls',
    'amount', ', '.join(minimal_controls[1:]),
    'Main sample, minimal controls', 'OLS'
)
if result:
    results.append(result)
    print(f"Minimal controls: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Network-only controls
network_only = ['height', 'close', 'between', 'Power'] if 'close' in df_main.columns and 'between' in df_main.columns else ['height', 'popular', 'popular_recipient']
formula_network = 'amount ~ ' + ' + '.join([c for c in network_only if c in df_main.columns])
result = run_ols_cluster(
    df_main, formula_network, 'population', 'height',
    'robust/control/network_only', 'methods/cross_sectional_ols.md#controls',
    'amount', ', '.join([c for c in network_only[1:] if c in df_main.columns]),
    'Main sample, network controls only', 'OLS'
)
if result:
    results.append(result)
    print(f"Network only controls: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# Pair characteristics only
pair_controls = ['height', 'samerace', 'sameheight', 'sameconf']
formula_pair = 'amount ~ ' + ' + '.join(pair_controls)
result = run_ols_cluster(
    df_main, formula_pair, 'population', 'height',
    'robust/control/pair_characteristics', 'methods/cross_sectional_ols.md#controls',
    'amount', ', '.join(pair_controls[1:]),
    'Main sample, pair characteristics only', 'OLS'
)
if result:
    results.append(result)
    print(f"Pair characteristics only: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# =============================================================================
# SAVE RESULTS
# =============================================================================

print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Convert to DataFrame and save
results_df = pd.DataFrame(results)
print(f"\nTotal specifications: {len(results_df)}")

# Save to CSV
output_file = f'{OUTPUT_PATH}/specification_results.csv'
results_df.to_csv(output_file, index=False)
print(f"Results saved to: {output_file}")

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

# Get baseline coefficient for reference (height in main dictator game analysis)
baseline_coef = results_df[results_df['spec_id'] == 'baseline']['coefficient'].values[0]
print(f"Baseline coefficient: {baseline_coef:.6f}")

# Statistics for dictator game height specifications only
dictator_height_specs = results_df[
    (results_df['treatment_var'] == 'height') &
    (~results_df['spec_id'].str.startswith('earnings/')) &
    (~results_df['spec_id'].str.startswith('network/'))
]

print(f"\nDictator game height specifications: {len(dictator_height_specs)}")
print(f"  Positive coefficients: {(dictator_height_specs['coefficient'] > 0).sum()} ({100*(dictator_height_specs['coefficient'] > 0).mean():.1f}%)")
print(f"  Significant at 5%: {(dictator_height_specs['p_value'] < 0.05).sum()} ({100*(dictator_height_specs['p_value'] < 0.05).mean():.1f}%)")
print(f"  Significant at 1%: {(dictator_height_specs['p_value'] < 0.01).sum()} ({100*(dictator_height_specs['p_value'] < 0.01).mean():.1f}%)")
print(f"  Median coefficient: {dictator_height_specs['coefficient'].median():.6f}")
print(f"  Mean coefficient: {dictator_height_specs['coefficient'].mean():.6f}")
print(f"  Range: [{dictator_height_specs['coefficient'].min():.6f}, {dictator_height_specs['coefficient'].max():.6f}]")

# Overall summary
print(f"\nAll specifications: {len(results_df)}")
print(f"  Unique treatment variables: {results_df['treatment_var'].unique()}")
print(f"  Unique outcomes: {results_df['outcome_var'].unique()}")

print("\n" + "="*60)
print("SPECIFICATION SEARCH COMPLETE")
print("="*60)
