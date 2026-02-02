"""
Specification Search: Paper 111185-V1
Title: Optimal Climate Policy When Damages Are Unknown
Journal: AEJ: Economic Policy
Author: Ivan Rudik

This script performs a systematic specification search on the empirical component
of the paper: the regression estimating damage function parameters from
meta-analysis data (Table 1).

Method: Cross-sectional OLS (meta-analysis regression)
Main specification: log(damages) ~ log(temperature)
Data: Howard and Sterner (2017) meta-analysis of climate damage estimates
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.quantile_regression import QuantReg
import json
import warnings
warnings.filterwarnings('ignore')

# Set paths
PACKAGE_DIR = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/111185-V1'
DATA_FILE = f'{PACKAGE_DIR}/estimate_damage_parameters/10640_2017_166_MOESM10_ESM.dta'
OUTPUT_FILE = f'{PACKAGE_DIR}/specification_results.csv'

# Paper metadata
PAPER_ID = '111185-V1'
JOURNAL = 'AEJ: Policy'
PAPER_TITLE = 'Optimal Climate Policy When Damages Are Unknown'

# Results container
results = []

def add_result(spec_id, spec_tree_path, model, outcome_var, treatment_var,
               sample_desc, controls_desc, cluster_var, model_type, n_obs,
               coefficient_vector_json=None, fixed_effects='None'):
    """Add a result to the results list."""

    try:
        # Get coefficient, SE, t-stat, p-value
        coef = model.params[treatment_var]
        se = model.bse[treatment_var]
        t_stat = model.tvalues[treatment_var]
        p_val = model.pvalues[treatment_var]

        # Confidence intervals
        conf_int = model.conf_int()
        ci_lower = conf_int.loc[treatment_var, 0]
        ci_upper = conf_int.loc[treatment_var, 1]

        # R-squared
        try:
            r_squared = model.rsquared
        except:
            r_squared = np.nan

    except Exception as e:
        print(f"Error extracting results for {spec_id}: {e}")
        return

    # Build coefficient vector JSON
    if coefficient_vector_json is None:
        coef_vector = {
            'treatment': {
                'var': treatment_var,
                'coef': float(coef),
                'se': float(se),
                'pval': float(p_val),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper)
            },
            'controls': [],
            'fixed_effects': [],
            'diagnostics': {}
        }

        # Add other coefficients
        for var in model.params.index:
            if var != treatment_var and var != 'Intercept':
                coef_vector['controls'].append({
                    'var': var,
                    'coef': float(model.params[var]),
                    'se': float(model.bse[var]),
                    'pval': float(model.pvalues[var])
                })

        coefficient_vector_json = json.dumps(coef_vector)

    results.append({
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
        'p_value': p_val,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_obs': n_obs,
        'r_squared': r_squared,
        'coefficient_vector_json': coefficient_vector_json,
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })

def add_result_manual(spec_id, spec_tree_path, outcome_var, treatment_var,
                      coef, se, t_stat, p_val, ci_lower, ci_upper, n_obs, r_squared,
                      sample_desc, controls_desc, cluster_var, model_type,
                      coefficient_vector_json=None, fixed_effects='None'):
    """Add a result manually (for models without standard statsmodels interface)."""

    if coefficient_vector_json is None:
        coef_vector = {
            'treatment': {
                'var': treatment_var,
                'coef': float(coef),
                'se': float(se),
                'pval': float(p_val),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper)
            },
            'controls': [],
            'fixed_effects': [],
            'diagnostics': {}
        }
        coefficient_vector_json = json.dumps(coef_vector)

    results.append({
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
        'p_value': p_val,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_obs': n_obs,
        'r_squared': r_squared,
        'coefficient_vector_json': coefficient_vector_json,
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })

# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================
print("Loading data...")
df = pd.read_stata(DATA_FILE)

# Replicate transformations from the Stata do file
# GDP Loss is defined as GDP_loss == (Y^g - Y^n)/Y^g = 1 - Y^n/Y^g
# Damage function is equivalent to Y^g/Y^n-1
# Translate GDP loss into damage function terms: damages = GDP_loss/(1-GDP_loss)
df['correct_d'] = (df['D_new']/100) / (1 - df['D_new']/100)
df['log_correct'] = np.log(df['correct_d'])
df['logt'] = np.log(df['t'])

# Create additional variables for robustness
df['t_sq'] = df['t']**2
df['logt_sq'] = df['logt']**2

# Convert categorical variables
df['Method_cat'] = df['Method'].astype('category')
df['Year_num'] = df['Year'].astype(float)

# Create indicator variables for subgroup analysis
df['Grey_binary'] = (df['Grey'] == 1).astype(int)
df['Market_binary'] = (df['Market'] == 1).astype(int)
df['is_enumerative'] = (df['Method'] == 'enumerative').astype(int)
df['is_statistical'] = (df['Method'] == 'statistical').astype(int)
df['is_survey'] = (df['Method'] == 'Survey').astype(int)
df['pre_2008'] = (df['Year'] < 2008).astype(int)
df['post_2008'] = (df['Year'] >= 2008).astype(int)
df['high_temp'] = (df['t'] > df['t'].median()).astype(int)

# Clean data for regression (remove invalid values)
df_clean = df[np.isfinite(df['log_correct']) & np.isfinite(df['logt'])].copy()

print(f"Total observations: {len(df)}")
print(f"Valid observations for baseline: {len(df_clean)}")

# =============================================================================
# BASELINE SPECIFICATION (exact replication of Table 1)
# =============================================================================
print("\n" + "="*60)
print("BASELINE SPECIFICATION")
print("="*60)

# Baseline: log(damages) ~ log(temperature)
model_baseline = smf.ols('log_correct ~ logt', data=df_clean).fit()
print(f"Baseline - log(d2) estimate: {model_baseline.params['logt']:.4f} (SE: {model_baseline.bse['logt']:.4f})")

add_result(
    spec_id='baseline',
    spec_tree_path='methods/cross_sectional_ols.md',
    model=model_baseline,
    outcome_var='log_correct',
    treatment_var='logt',
    sample_desc='Full sample, Howard and Sterner (2017) meta-analysis',
    controls_desc='None (bivariate)',
    cluster_var='None',
    model_type='OLS',
    n_obs=len(df_clean)
)

# =============================================================================
# INFERENCE VARIATIONS (5-8 specs)
# =============================================================================
print("\n" + "="*60)
print("INFERENCE VARIATIONS")
print("="*60)

# 1. Robust (HC1) standard errors
model_hc1 = smf.ols('log_correct ~ logt', data=df_clean).fit(cov_type='HC1')
add_result('robust/se/hc1', 'robustness/clustering_variations.md', model_hc1,
           'log_correct', 'logt', 'Full sample', 'None', 'None (HC1 robust)', 'OLS-HC1', len(df_clean))
print(f"HC1 robust SE: {model_hc1.bse['logt']:.4f}")

# 2. HC2 standard errors
model_hc2 = smf.ols('log_correct ~ logt', data=df_clean).fit(cov_type='HC2')
add_result('robust/se/hc2', 'robustness/clustering_variations.md', model_hc2,
           'log_correct', 'logt', 'Full sample', 'None', 'None (HC2 robust)', 'OLS-HC2', len(df_clean))
print(f"HC2 robust SE: {model_hc2.bse['logt']:.4f}")

# 3. HC3 standard errors (small sample)
model_hc3 = smf.ols('log_correct ~ logt', data=df_clean).fit(cov_type='HC3')
add_result('robust/se/hc3', 'robustness/clustering_variations.md', model_hc3,
           'log_correct', 'logt', 'Full sample', 'None', 'None (HC3 robust)', 'OLS-HC3', len(df_clean))
print(f"HC3 robust SE: {model_hc3.bse['logt']:.4f}")

# 4. Cluster by Method
df_clean_method = df_clean.dropna(subset=['Method'])
model_cluster_method = smf.ols('log_correct ~ logt', data=df_clean_method).fit(
    cov_type='cluster', cov_kwds={'groups': df_clean_method['Method']})
add_result('robust/cluster/method', 'robustness/clustering_variations.md', model_cluster_method,
           'log_correct', 'logt', 'Full sample', 'None', 'Method', 'OLS-clustered', len(df_clean_method))
print(f"Clustered by Method SE: {model_cluster_method.bse['logt']:.4f}")

# 5. Cluster by Primary Author
df_clean_author = df_clean.dropna(subset=['Primary_Author'])
model_cluster_author = smf.ols('log_correct ~ logt', data=df_clean_author).fit(
    cov_type='cluster', cov_kwds={'groups': df_clean_author['Primary_Author']})
add_result('robust/cluster/author', 'robustness/clustering_variations.md', model_cluster_author,
           'log_correct', 'logt', 'Full sample', 'None', 'Primary_Author', 'OLS-clustered', len(df_clean_author))
print(f"Clustered by Author SE: {model_cluster_author.bse['logt']:.4f}")

# 6. Cluster by Model
df_clean_model = df_clean.dropna(subset=['Model'])
# Only cluster if there are multiple groups
if df_clean_model['Model'].nunique() > 1:
    model_cluster_mod = smf.ols('log_correct ~ logt', data=df_clean_model).fit(
        cov_type='cluster', cov_kwds={'groups': df_clean_model['Model']})
    add_result('robust/cluster/model', 'robustness/clustering_variations.md', model_cluster_mod,
               'log_correct', 'logt', 'Full sample', 'None', 'Model', 'OLS-clustered', len(df_clean_model))
    print(f"Clustered by Model SE: {model_cluster_mod.bse['logt']:.4f}")

# =============================================================================
# FUNCTIONAL FORM VARIATIONS (10 specs)
# =============================================================================
print("\n" + "="*60)
print("FUNCTIONAL FORM VARIATIONS")
print("="*60)

# 1. Level-level (temperature in levels)
df_clean['correct_d_pos'] = df_clean['correct_d'].clip(lower=0.001)
model_level = smf.ols('correct_d_pos ~ t', data=df_clean).fit()
add_result('robust/form/level_level', 'robustness/functional_form.md', model_level,
           'correct_d', 't', 'Full sample', 'None (linear)', 'None', 'OLS', len(df_clean))
print(f"Level-level: coef = {model_level.params['t']:.4f}")

# 2. Log-level (log outcome, level treatment)
model_log_level = smf.ols('log_correct ~ t', data=df_clean).fit()
add_result('robust/form/log_level', 'robustness/functional_form.md', model_log_level,
           'log_correct', 't', 'Full sample', 'None (semi-log)', 'None', 'OLS', len(df_clean))
print(f"Log-level: coef = {model_log_level.params['t']:.4f}")

# 3. Level-log (level outcome, log treatment)
model_level_log = smf.ols('correct_d_pos ~ logt', data=df_clean).fit()
add_result('robust/form/level_log', 'robustness/functional_form.md', model_level_log,
           'correct_d', 'logt', 'Full sample', 'None', 'None', 'OLS', len(df_clean))
print(f"Level-log: coef = {model_level_log.params['logt']:.4f}")

# 4. Quadratic in log temperature
model_quad = smf.ols('log_correct ~ logt + logt_sq', data=df_clean).fit()
add_result('robust/form/quadratic_logt', 'robustness/functional_form.md', model_quad,
           'log_correct', 'logt', 'Full sample', 'logt_sq', 'None', 'OLS', len(df_clean))
print(f"Quadratic: logt coef = {model_quad.params['logt']:.4f}, logt_sq coef = {model_quad.params['logt_sq']:.4f}")

# 5. Quadratic in temperature (levels)
model_quad_level = smf.ols('log_correct ~ t + t_sq', data=df_clean).fit()
add_result('robust/form/quadratic_t', 'robustness/functional_form.md', model_quad_level,
           'log_correct', 't', 'Full sample', 't_sq', 'None', 'OLS', len(df_clean))
print(f"Quadratic (level): t coef = {model_quad_level.params['t']:.4f}")

# 6. Asinh transformation of outcome
df_clean['asinh_correct'] = np.arcsinh(df_clean['correct_d'])
model_asinh = smf.ols('asinh_correct ~ logt', data=df_clean).fit()
add_result('robust/form/y_asinh', 'robustness/functional_form.md', model_asinh,
           'asinh_correct', 'logt', 'Full sample', 'None (asinh outcome)', 'None', 'OLS', len(df_clean))
print(f"Asinh outcome: coef = {model_asinh.params['logt']:.4f}")

# 7. Quantile regression - median
try:
    X = sm.add_constant(df_clean['logt'])
    model_q50 = QuantReg(df_clean['log_correct'], X).fit(q=0.5)
    add_result('robust/form/quantile_50', 'robustness/functional_form.md', model_q50,
               'log_correct', 'logt', 'Full sample', 'None (median)', 'None', 'Quantile-0.5', len(df_clean))
    print(f"Quantile 50: coef = {model_q50.params['logt']:.4f}")
except Exception as e:
    print(f"Quantile 50 failed: {e}")

# 8. Quantile regression - 25th percentile
try:
    model_q25 = QuantReg(df_clean['log_correct'], X).fit(q=0.25)
    add_result('robust/form/quantile_25', 'robustness/functional_form.md', model_q25,
               'log_correct', 'logt', 'Full sample', 'None (25th pct)', 'None', 'Quantile-0.25', len(df_clean))
    print(f"Quantile 25: coef = {model_q25.params['logt']:.4f}")
except Exception as e:
    print(f"Quantile 25 failed: {e}")

# 9. Quantile regression - 75th percentile
try:
    model_q75 = QuantReg(df_clean['log_correct'], X).fit(q=0.75)
    add_result('robust/form/quantile_75', 'robustness/functional_form.md', model_q75,
               'log_correct', 'logt', 'Full sample', 'None (75th pct)', 'None', 'Quantile-0.75', len(df_clean))
    print(f"Quantile 75: coef = {model_q75.params['logt']:.4f}")
except Exception as e:
    print(f"Quantile 75 failed: {e}")

# 10. Cubic polynomial
df_clean['logt_cu'] = df_clean['logt']**3
model_cubic = smf.ols('log_correct ~ logt + logt_sq + logt_cu', data=df_clean).fit()
add_result('robust/form/cubic', 'robustness/functional_form.md', model_cubic,
           'log_correct', 'logt', 'Full sample', 'logt_sq + logt_cu', 'None', 'OLS', len(df_clean))
print(f"Cubic: logt coef = {model_cubic.params['logt']:.4f}")

# =============================================================================
# SAMPLE RESTRICTIONS (15 specs)
# =============================================================================
print("\n" + "="*60)
print("SAMPLE RESTRICTIONS")
print("="*60)

# 1. Trim 1% outliers on outcome
q01 = df_clean['log_correct'].quantile(0.01)
q99 = df_clean['log_correct'].quantile(0.99)
df_trim1 = df_clean[(df_clean['log_correct'] > q01) & (df_clean['log_correct'] < q99)]
if len(df_trim1) >= 10:
    model_trim1 = smf.ols('log_correct ~ logt', data=df_trim1).fit()
    add_result('robust/sample/trim_1pct', 'robustness/sample_restrictions.md', model_trim1,
               'log_correct', 'logt', f'Trimmed 1% outliers (n={len(df_trim1)})', 'None', 'None', 'OLS', len(df_trim1))
    print(f"Trim 1%: coef = {model_trim1.params['logt']:.4f} (n={len(df_trim1)})")

# 2. Trim 5% outliers on outcome
q05 = df_clean['log_correct'].quantile(0.05)
q95 = df_clean['log_correct'].quantile(0.95)
df_trim5 = df_clean[(df_clean['log_correct'] > q05) & (df_clean['log_correct'] < q95)]
if len(df_trim5) >= 10:
    model_trim5 = smf.ols('log_correct ~ logt', data=df_trim5).fit()
    add_result('robust/sample/trim_5pct', 'robustness/sample_restrictions.md', model_trim5,
               'log_correct', 'logt', f'Trimmed 5% outliers (n={len(df_trim5)})', 'None', 'None', 'OLS', len(df_trim5))
    print(f"Trim 5%: coef = {model_trim5.params['logt']:.4f} (n={len(df_trim5)})")

# 3. Winsorize 1%
df_wins1 = df_clean.copy()
df_wins1['log_correct'] = df_wins1['log_correct'].clip(lower=q01, upper=q99)
model_wins1 = smf.ols('log_correct ~ logt', data=df_wins1).fit()
add_result('robust/sample/winsor_1pct', 'robustness/sample_restrictions.md', model_wins1,
           'log_correct', 'logt', 'Winsorized 1%', 'None', 'None', 'OLS', len(df_wins1))
print(f"Winsor 1%: coef = {model_wins1.params['logt']:.4f}")

# 4. Winsorize 5%
df_wins5 = df_clean.copy()
df_wins5['log_correct'] = df_wins5['log_correct'].clip(lower=q05, upper=q95)
model_wins5 = smf.ols('log_correct ~ logt', data=df_wins5).fit()
add_result('robust/sample/winsor_5pct', 'robustness/sample_restrictions.md', model_wins5,
           'log_correct', 'logt', 'Winsorized 5%', 'None', 'None', 'OLS', len(df_wins5))
print(f"Winsor 5%: coef = {model_wins5.params['logt']:.4f}")

# 5. Early period (pre-2008)
df_early = df_clean[df_clean['Year'] < 2008]
if len(df_early) >= 10:
    model_early = smf.ols('log_correct ~ logt', data=df_early).fit()
    add_result('robust/sample/early_period', 'robustness/sample_restrictions.md', model_early,
               'log_correct', 'logt', f'Pre-2008 studies (n={len(df_early)})', 'None', 'None', 'OLS', len(df_early))
    print(f"Pre-2008: coef = {model_early.params['logt']:.4f} (n={len(df_early)})")

# 6. Late period (post-2008)
df_late = df_clean[df_clean['Year'] >= 2008]
if len(df_late) >= 10:
    model_late = smf.ols('log_correct ~ logt', data=df_late).fit()
    add_result('robust/sample/late_period', 'robustness/sample_restrictions.md', model_late,
               'log_correct', 'logt', f'Post-2008 studies (n={len(df_late)})', 'None', 'None', 'OLS', len(df_late))
    print(f"Post-2008: coef = {model_late.params['logt']:.4f} (n={len(df_late)})")

# 7. Pre-2005
df_pre05 = df_clean[df_clean['Year'] < 2005]
if len(df_pre05) >= 10:
    model_pre05 = smf.ols('log_correct ~ logt', data=df_pre05).fit()
    add_result('robust/sample/pre_2005', 'robustness/sample_restrictions.md', model_pre05,
               'log_correct', 'logt', f'Pre-2005 studies (n={len(df_pre05)})', 'None', 'None', 'OLS', len(df_pre05))
    print(f"Pre-2005: coef = {model_pre05.params['logt']:.4f} (n={len(df_pre05)})")

# 8. Post-2010
df_post10 = df_clean[df_clean['Year'] >= 2010]
if len(df_post10) >= 10:
    model_post10 = smf.ols('log_correct ~ logt', data=df_post10).fit()
    add_result('robust/sample/post_2010', 'robustness/sample_restrictions.md', model_post10,
               'log_correct', 'logt', f'Post-2010 studies (n={len(df_post10)})', 'None', 'None', 'OLS', len(df_post10))
    print(f"Post-2010: coef = {model_post10.params['logt']:.4f} (n={len(df_post10)})")

# 9. Exclude grey literature
df_no_grey = df_clean[df_clean['Grey'] != 1]
if len(df_no_grey) >= 10:
    model_no_grey = smf.ols('log_correct ~ logt', data=df_no_grey).fit()
    add_result('robust/sample/exclude_grey', 'robustness/sample_restrictions.md', model_no_grey,
               'log_correct', 'logt', f'Exclude grey literature (n={len(df_no_grey)})', 'None', 'None', 'OLS', len(df_no_grey))
    print(f"No grey: coef = {model_no_grey.params['logt']:.4f} (n={len(df_no_grey)})")

# 10. Market estimates only
df_market = df_clean[df_clean['Market'] == 1]
if len(df_market) >= 10:
    model_market = smf.ols('log_correct ~ logt', data=df_market).fit()
    add_result('robust/sample/market_only', 'robustness/sample_restrictions.md', model_market,
               'log_correct', 'logt', f'Market estimates only (n={len(df_market)})', 'None', 'None', 'OLS', len(df_market))
    print(f"Market only: coef = {model_market.params['logt']:.4f} (n={len(df_market)})")

# 11. Non-market estimates
df_nonmarket = df_clean[df_clean['Market'] != 1]
if len(df_nonmarket) >= 10:
    model_nonmarket = smf.ols('log_correct ~ logt', data=df_nonmarket).fit()
    add_result('robust/sample/nonmarket', 'robustness/sample_restrictions.md', model_nonmarket,
               'log_correct', 'logt', f'Non-market estimates (n={len(df_nonmarket)})', 'None', 'None', 'OLS', len(df_nonmarket))
    print(f"Non-market: coef = {model_nonmarket.params['logt']:.4f} (n={len(df_nonmarket)})")

# 12. Enumerative method only
df_enum = df_clean[df_clean['Method'] == 'enumerative']
if len(df_enum) >= 10:
    model_enum = smf.ols('log_correct ~ logt', data=df_enum).fit()
    add_result('robust/sample/method_enumerative', 'robustness/sample_restrictions.md', model_enum,
               'log_correct', 'logt', f'Enumerative method (n={len(df_enum)})', 'None', 'None', 'OLS', len(df_enum))
    print(f"Enumerative: coef = {model_enum.params['logt']:.4f} (n={len(df_enum)})")

# 13. Statistical method only
df_stat = df_clean[df_clean['Method'] == 'statistical']
if len(df_stat) >= 10:
    model_stat = smf.ols('log_correct ~ logt', data=df_stat).fit()
    add_result('robust/sample/method_statistical', 'robustness/sample_restrictions.md', model_stat,
               'log_correct', 'logt', f'Statistical method (n={len(df_stat)})', 'None', 'None', 'OLS', len(df_stat))
    print(f"Statistical: coef = {model_stat.params['logt']:.4f} (n={len(df_stat)})")

# 14. High temperature scenarios (above median t)
df_high_t = df_clean[df_clean['t'] > df_clean['t'].median()]
if len(df_high_t) >= 10:
    model_high_t = smf.ols('log_correct ~ logt', data=df_high_t).fit()
    add_result('robust/sample/high_temp', 'robustness/sample_restrictions.md', model_high_t,
               'log_correct', 'logt', f'High temp scenarios (n={len(df_high_t)})', 'None', 'None', 'OLS', len(df_high_t))
    print(f"High temp: coef = {model_high_t.params['logt']:.4f} (n={len(df_high_t)})")

# 15. Low temperature scenarios (below median t)
df_low_t = df_clean[df_clean['t'] <= df_clean['t'].median()]
if len(df_low_t) >= 10:
    model_low_t = smf.ols('log_correct ~ logt', data=df_low_t).fit()
    add_result('robust/sample/low_temp', 'robustness/sample_restrictions.md', model_low_t,
               'log_correct', 'logt', f'Low temp scenarios (n={len(df_low_t)})', 'None', 'None', 'OLS', len(df_low_t))
    print(f"Low temp: coef = {model_low_t.params['logt']:.4f} (n={len(df_low_t)})")

# 16. Drop high-leverage points (Cook's D > 4/n)
from statsmodels.stats.outliers_influence import OLSInfluence
influence = OLSInfluence(model_baseline)
cooks_d = influence.cooks_distance[0]
threshold = 4 / len(df_clean)
df_no_infl = df_clean[cooks_d < threshold]
if len(df_no_infl) >= 10:
    model_no_infl = smf.ols('log_correct ~ logt', data=df_no_infl).fit()
    add_result('robust/sample/drop_influential', 'robustness/sample_restrictions.md', model_no_infl,
               'log_correct', 'logt', f'Drop influential obs (n={len(df_no_infl)})', 'None', 'None', 'OLS', len(df_no_infl))
    print(f"No influential: coef = {model_no_infl.params['logt']:.4f} (n={len(df_no_infl)})")

# =============================================================================
# CONTROL VARIABLES (10 specs)
# =============================================================================
print("\n" + "="*60)
print("CONTROL VARIABLE VARIATIONS")
print("="*60)

# 1. Control for publication year
model_year = smf.ols('log_correct ~ logt + Year_num', data=df_clean).fit()
add_result('robust/control/year', 'robustness/control_progression.md', model_year,
           'log_correct', 'logt', 'Full sample', 'Year', 'None', 'OLS', len(df_clean))
print(f"Control Year: coef = {model_year.params['logt']:.4f}")

# 2. Control for Grey literature indicator
model_grey_ctrl = smf.ols('log_correct ~ logt + Grey_binary', data=df_clean).fit()
add_result('robust/control/grey', 'robustness/control_progression.md', model_grey_ctrl,
           'log_correct', 'logt', 'Full sample', 'Grey indicator', 'None', 'OLS', len(df_clean))
print(f"Control Grey: coef = {model_grey_ctrl.params['logt']:.4f}")

# 3. Control for Market indicator
model_market_ctrl = smf.ols('log_correct ~ logt + Market_binary', data=df_clean).fit()
add_result('robust/control/market', 'robustness/control_progression.md', model_market_ctrl,
           'log_correct', 'logt', 'Full sample', 'Market indicator', 'None', 'OLS', len(df_clean))
print(f"Control Market: coef = {model_market_ctrl.params['logt']:.4f}")

# 4. Control for Year and Grey
model_year_grey = smf.ols('log_correct ~ logt + Year_num + Grey_binary', data=df_clean).fit()
add_result('robust/control/year_grey', 'robustness/control_progression.md', model_year_grey,
           'log_correct', 'logt', 'Full sample', 'Year + Grey', 'None', 'OLS', len(df_clean))
print(f"Control Year+Grey: coef = {model_year_grey.params['logt']:.4f}")

# 5. Control for Year, Grey, Market
model_full = smf.ols('log_correct ~ logt + Year_num + Grey_binary + Market_binary', data=df_clean).fit()
add_result('robust/control/full', 'robustness/control_progression.md', model_full,
           'log_correct', 'logt', 'Full sample', 'Year + Grey + Market', 'None', 'OLS', len(df_clean))
print(f"Full controls: coef = {model_full.params['logt']:.4f}")

# 6. Method indicators
model_method = smf.ols('log_correct ~ logt + is_enumerative + is_statistical + is_survey', data=df_clean).fit()
add_result('robust/control/method_indicators', 'robustness/control_progression.md', model_method,
           'log_correct', 'logt', 'Full sample', 'Method indicators', 'None', 'OLS', len(df_clean))
print(f"Method indicators: coef = {model_method.params['logt']:.4f}")

# 7. Kitchen sink (all controls)
model_kitchen = smf.ols('log_correct ~ logt + Year_num + Grey_binary + Market_binary + is_enumerative + is_statistical',
                        data=df_clean).fit()
add_result('robust/control/kitchen_sink', 'robustness/control_progression.md', model_kitchen,
           'log_correct', 'logt', 'Full sample', 'All available controls', 'None', 'OLS', len(df_clean))
print(f"Kitchen sink: coef = {model_kitchen.params['logt']:.4f}")

# 8. Method fixed effects
df_clean['Method_fe'] = pd.Categorical(df_clean['Method'])
model_method_fe = smf.ols('log_correct ~ logt + C(Method_fe)', data=df_clean).fit()
add_result('robust/control/method_fe', 'robustness/control_progression.md', model_method_fe,
           'log_correct', 'logt', 'Full sample', 'Method FE', 'None', 'OLS-FE', len(df_clean), fixed_effects='Method')
print(f"Method FE: coef = {model_method_fe.params['logt']:.4f}")

# 9. Year linear trend
df_clean['Year_centered'] = df_clean['Year_num'] - df_clean['Year_num'].mean()
model_year_trend = smf.ols('log_correct ~ logt + Year_centered', data=df_clean).fit()
add_result('robust/control/year_trend', 'robustness/control_progression.md', model_year_trend,
           'log_correct', 'logt', 'Full sample', 'Year trend (centered)', 'None', 'OLS', len(df_clean))
print(f"Year trend: coef = {model_year_trend.params['logt']:.4f}")

# 10. Year squared (nonlinear trend)
df_clean['Year_centered_sq'] = df_clean['Year_centered']**2
model_year_quad = smf.ols('log_correct ~ logt + Year_centered + Year_centered_sq', data=df_clean).fit()
add_result('robust/control/year_quadratic', 'robustness/control_progression.md', model_year_quad,
           'log_correct', 'logt', 'Full sample', 'Year quadratic trend', 'None', 'OLS', len(df_clean))
print(f"Year quad: coef = {model_year_quad.params['logt']:.4f}")

# =============================================================================
# HETEROGENEITY ANALYSIS (10 specs)
# =============================================================================
print("\n" + "="*60)
print("HETEROGENEITY ANALYSIS")
print("="*60)

# 1. Interaction with Grey literature
model_het_grey = smf.ols('log_correct ~ logt * Grey_binary', data=df_clean).fit()
add_result('robust/het/grey_interaction', 'robustness/heterogeneity.md', model_het_grey,
           'log_correct', 'logt', 'Full sample', 'logt x Grey interaction', 'None', 'OLS', len(df_clean))
print(f"Grey interaction: main effect = {model_het_grey.params['logt']:.4f}")

# 2. Interaction with Market
model_het_market = smf.ols('log_correct ~ logt * Market_binary', data=df_clean).fit()
add_result('robust/het/market_interaction', 'robustness/heterogeneity.md', model_het_market,
           'log_correct', 'logt', 'Full sample', 'logt x Market interaction', 'None', 'OLS', len(df_clean))
print(f"Market interaction: main effect = {model_het_market.params['logt']:.4f}")

# 3. Interaction with Year (linear)
model_het_year = smf.ols('log_correct ~ logt * Year_centered', data=df_clean).fit()
add_result('robust/het/year_interaction', 'robustness/heterogeneity.md', model_het_year,
           'log_correct', 'logt', 'Full sample', 'logt x Year interaction', 'None', 'OLS', len(df_clean))
print(f"Year interaction: main effect = {model_het_year.params['logt']:.4f}")

# 4. Interaction with enumerative method
model_het_enum = smf.ols('log_correct ~ logt * is_enumerative', data=df_clean).fit()
add_result('robust/het/enumerative_interaction', 'robustness/heterogeneity.md', model_het_enum,
           'log_correct', 'logt', 'Full sample', 'logt x Enumerative interaction', 'None', 'OLS', len(df_clean))
print(f"Enumerative interaction: main effect = {model_het_enum.params['logt']:.4f}")

# 5. Interaction with statistical method
model_het_stat = smf.ols('log_correct ~ logt * is_statistical', data=df_clean).fit()
add_result('robust/het/statistical_interaction', 'robustness/heterogeneity.md', model_het_stat,
           'log_correct', 'logt', 'Full sample', 'logt x Statistical interaction', 'None', 'OLS', len(df_clean))
print(f"Statistical interaction: main effect = {model_het_stat.params['logt']:.4f}")

# 6. Interaction with pre-2008
model_het_pre08 = smf.ols('log_correct ~ logt * pre_2008', data=df_clean).fit()
add_result('robust/het/pre2008_interaction', 'robustness/heterogeneity.md', model_het_pre08,
           'log_correct', 'logt', 'Full sample', 'logt x Pre-2008 interaction', 'None', 'OLS', len(df_clean))
print(f"Pre-2008 interaction: main effect = {model_het_pre08.params['logt']:.4f}")

# 7. Interaction with high temperature
model_het_high_t = smf.ols('log_correct ~ logt * high_temp', data=df_clean).fit()
add_result('robust/het/high_temp_interaction', 'robustness/heterogeneity.md', model_het_high_t,
           'log_correct', 'logt', 'Full sample', 'logt x High temp interaction', 'None', 'OLS', len(df_clean))
print(f"High temp interaction: main effect = {model_het_high_t.params['logt']:.4f}")

# 8-10. Split by Year terciles
df_clean['Year_tercile'] = pd.qcut(df_clean['Year'], 3, labels=['early', 'mid', 'late'], duplicates='drop')
for tercile in ['early', 'mid', 'late']:
    df_tercile = df_clean[df_clean['Year_tercile'] == tercile]
    if len(df_tercile) >= 5:
        model_tercile = smf.ols('log_correct ~ logt', data=df_tercile).fit()
        add_result(f'robust/het/year_tercile_{tercile}', 'robustness/heterogeneity.md', model_tercile,
                   'log_correct', 'logt', f'Year tercile: {tercile} (n={len(df_tercile)})', 'None', 'None', 'OLS', len(df_tercile))
        print(f"Year tercile {tercile}: coef = {model_tercile.params['logt']:.4f} (n={len(df_tercile)})")

# =============================================================================
# LEAVE-ONE-OUT STUDIES (jackknife)
# =============================================================================
print("\n" + "="*60)
print("LEAVE-ONE-OUT ANALYSIS")
print("="*60)

# Get unique studies
studies = df_clean['Study'].unique()
for i, study in enumerate(studies[:15]):  # Limit to 15 most common
    df_loo = df_clean[df_clean['Study'] != study]
    if len(df_loo) >= 10:
        model_loo = smf.ols('log_correct ~ logt', data=df_loo).fit()
        study_short = study[:30] if len(study) > 30 else study
        add_result(f'robust/loo/drop_study_{i+1}', 'robustness/leave_one_out.md', model_loo,
                   'log_correct', 'logt', f'Drop {study_short} (n={len(df_loo)})', 'None', 'None', 'OLS', len(df_loo))
        print(f"LOO {i+1}: coef = {model_loo.params['logt']:.4f} (drop {study_short[:20]})")

# =============================================================================
# WEIGHTED REGRESSIONS
# =============================================================================
print("\n" + "="*60)
print("WEIGHTED REGRESSIONS")
print("="*60)

# 1. Weight by inverse variance proxy (using 1/t as proxy for precision)
df_clean['weight_inv_t'] = 1 / df_clean['t']
model_wls1 = smf.wls('log_correct ~ logt', data=df_clean, weights=df_clean['weight_inv_t']).fit()
add_result('robust/weights/inverse_temp', 'robustness/measurement.md', model_wls1,
           'log_correct', 'logt', 'Full sample', 'None', 'None', 'WLS (weight=1/t)', len(df_clean))
print(f"WLS (1/t): coef = {model_wls1.params['logt']:.4f}")

# 2. Weight by year (more recent = higher weight)
df_clean['weight_year'] = df_clean['Year_num'] - df_clean['Year_num'].min() + 1
model_wls2 = smf.wls('log_correct ~ logt', data=df_clean, weights=df_clean['weight_year']).fit()
add_result('robust/weights/year_weight', 'robustness/measurement.md', model_wls2,
           'log_correct', 'logt', 'Full sample', 'None', 'None', 'WLS (weight=year)', len(df_clean))
print(f"WLS (year): coef = {model_wls2.params['logt']:.4f}")

# 3. Weight by 1/year (older = higher weight)
df_clean['weight_inv_year'] = 1 / df_clean['weight_year']
model_wls3 = smf.wls('log_correct ~ logt', data=df_clean, weights=df_clean['weight_inv_year']).fit()
add_result('robust/weights/inverse_year', 'robustness/measurement.md', model_wls3,
           'log_correct', 'logt', 'Full sample', 'None', 'None', 'WLS (weight=1/year)', len(df_clean))
print(f"WLS (1/year): coef = {model_wls3.params['logt']:.4f}")

# =============================================================================
# ALTERNATIVE OUTCOME DEFINITIONS
# =============================================================================
print("\n" + "="*60)
print("ALTERNATIVE OUTCOMES")
print("="*60)

# 1. Using D_new directly (% GDP loss)
model_dnew = smf.ols('D_new ~ logt', data=df_clean).fit()
add_result('robust/outcome/dnew_level', 'robustness/measurement.md', model_dnew,
           'D_new', 'logt', 'Full sample', 'None', 'None', 'OLS', len(df_clean))
print(f"D_new (level): coef = {model_dnew.params['logt']:.4f}")

# 2. Log of D_new
df_clean['log_dnew'] = np.log(df_clean['D_new'].clip(lower=0.01))
model_log_dnew = smf.ols('log_dnew ~ logt', data=df_clean[np.isfinite(df_clean['log_dnew'])]).fit()
add_result('robust/outcome/dnew_log', 'robustness/measurement.md', model_log_dnew,
           'log_dnew', 'logt', 'Full sample (D_new > 0)', 'None', 'None', 'OLS', len(df_clean[np.isfinite(df_clean['log_dnew'])]))
print(f"Log D_new: coef = {model_log_dnew.params['logt']:.4f}")

# 3. Using original D
df_clean_d = df_clean[df_clean['D'].notna() & (df_clean['D'] != 0)]
if len(df_clean_d) >= 10:
    df_clean_d['log_D'] = np.log(np.abs(df_clean_d['D']).clip(lower=0.01))
    model_orig_d = smf.ols('log_D ~ logt', data=df_clean_d).fit()
    add_result('robust/outcome/original_d', 'robustness/measurement.md', model_orig_d,
               'log_D', 'logt', 'Full sample (original D)', 'None', 'None', 'OLS', len(df_clean_d))
    print(f"Original D: coef = {model_orig_d.params['logt']:.4f}")

# =============================================================================
# PLACEBO / FALSIFICATION TESTS
# =============================================================================
print("\n" + "="*60)
print("PLACEBO TESTS")
print("="*60)

# 1. Permutation test - shuffle temperature
np.random.seed(42)
df_placebo = df_clean.copy()
df_placebo['logt_shuffled'] = np.random.permutation(df_placebo['logt'].values)
model_placebo1 = smf.ols('log_correct ~ logt_shuffled', data=df_placebo).fit()
add_result('robust/placebo/shuffled_temp', 'robustness/placebo_tests.md', model_placebo1,
           'log_correct', 'logt_shuffled', 'Full sample (shuffled temp)', 'None', 'None', 'OLS-placebo', len(df_placebo))
print(f"Placebo (shuffle temp): coef = {model_placebo1.params['logt_shuffled']:.4f}, p = {model_placebo1.pvalues['logt_shuffled']:.4f}")

# 2. Permutation test 2
np.random.seed(123)
df_placebo['logt_shuffled2'] = np.random.permutation(df_placebo['logt'].values)
model_placebo2 = smf.ols('log_correct ~ logt_shuffled2', data=df_placebo).fit()
add_result('robust/placebo/shuffled_temp_2', 'robustness/placebo_tests.md', model_placebo2,
           'log_correct', 'logt_shuffled2', 'Full sample (shuffled temp seed 2)', 'None', 'None', 'OLS-placebo', len(df_placebo))
print(f"Placebo 2: coef = {model_placebo2.params['logt_shuffled2']:.4f}")

# 3. Regress on Year only (no temperature) - should be weak
model_year_only = smf.ols('log_correct ~ Year_num', data=df_clean).fit()
add_result('robust/placebo/year_only', 'robustness/placebo_tests.md', model_year_only,
           'log_correct', 'Year_num', 'Full sample', 'None', 'None', 'OLS-placebo', len(df_clean))
print(f"Year only: coef = {model_year_only.params['Year_num']:.4f}, p = {model_year_only.pvalues['Year_num']:.4f}")

# =============================================================================
# BOOTSTRAP STANDARD ERRORS
# =============================================================================
print("\n" + "="*60)
print("BOOTSTRAP ANALYSIS")
print("="*60)

# Run bootstrap for baseline specification
n_boot = 1000
np.random.seed(42)
boot_coefs = []

for b in range(n_boot):
    boot_sample = df_clean.sample(n=len(df_clean), replace=True)
    try:
        boot_model = smf.ols('log_correct ~ logt', data=boot_sample).fit()
        boot_coefs.append(boot_model.params['logt'])
    except:
        pass

boot_coefs = np.array(boot_coefs)
boot_se = np.std(boot_coefs)
boot_mean = np.mean(boot_coefs)
boot_ci_lower = np.percentile(boot_coefs, 2.5)
boot_ci_upper = np.percentile(boot_coefs, 97.5)

# Add bootstrap result
add_result_manual(
    'robust/se/bootstrap',
    'robustness/clustering_variations.md',
    'log_correct', 'logt',
    boot_mean, boot_se, boot_mean/boot_se,
    2 * (1 - sm.distributions.ECDF(np.abs(boot_coefs))(np.abs(boot_mean))),  # two-sided p-value approximation
    boot_ci_lower, boot_ci_upper, len(df_clean), model_baseline.rsquared,
    'Full sample', 'None', 'Bootstrap (1000 reps)', 'OLS-bootstrap'
)
print(f"Bootstrap SE: {boot_se:.4f} (vs OLS SE: {model_baseline.bse['logt']:.4f})")

# =============================================================================
# SAVE RESULTS
# =============================================================================
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
results_df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved {len(results_df)} specifications to {OUTPUT_FILE}")

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Total specifications: {len(results_df)}")
print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({(results_df['coefficient'] > 0).mean()*100:.1f}%)")
print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({(results_df['p_value'] < 0.05).mean()*100:.1f}%)")
print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({(results_df['p_value'] < 0.01).mean()*100:.1f}%)")
print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")
print(f"Coefficient range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")

# Breakdown by category
print("\n" + "="*60)
print("BREAKDOWN BY CATEGORY")
print("="*60)

def get_category(spec_id):
    if spec_id == 'baseline':
        return 'Baseline'
    elif 'se/' in spec_id or 'cluster/' in spec_id:
        return 'Inference variations'
    elif 'form/' in spec_id:
        return 'Functional form'
    elif 'sample/' in spec_id:
        return 'Sample restrictions'
    elif 'control/' in spec_id:
        return 'Control variations'
    elif 'het/' in spec_id:
        return 'Heterogeneity'
    elif 'loo/' in spec_id:
        return 'Leave-one-out'
    elif 'weights/' in spec_id:
        return 'Weights'
    elif 'outcome/' in spec_id:
        return 'Alternative outcomes'
    elif 'placebo/' in spec_id:
        return 'Placebo tests'
    else:
        return 'Other'

results_df['category'] = results_df['spec_id'].apply(get_category)
category_summary = results_df.groupby('category').agg({
    'coefficient': ['count', lambda x: (x > 0).mean() * 100],
    'p_value': lambda x: (x < 0.05).mean() * 100
}).round(1)
category_summary.columns = ['N', '% Positive', '% Sig 5%']
print(category_summary)

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
