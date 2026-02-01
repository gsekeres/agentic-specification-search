"""
Specification Search: "The Price of Experience"
Paper ID: 235621-V1
Journal: AER
Authors: Jeong, Kim, Manovskii (2015)

Main Hypothesis: The relative price of experience has increased over time,
explaining the flattening of age-earnings profiles and changing returns to experience.

Method: Panel Fixed Effects (wage regressions on PSID panel data)
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

import statsmodels.api as sm
from scipy import stats

# ============================================================================
# Configuration
# ============================================================================

BASE_PATH = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/235621-V1/'
OUTPUT_PATH = BASE_PATH

# ============================================================================
# Data Loading and Preparation
# ============================================================================

print("Loading data...")
df = pd.read_stata(BASE_PATH + 'basedata.dta')

# Convert all columns to appropriate types
for col in df.columns:
    if df[col].dtype in ['float32', 'float16', 'int8', 'int16', 'int32']:
        df[col] = df[col].astype(np.float64)

# Create derived variables
df['e2'] = df['e'] ** 2
df['s2'] = df['s'] ** 2
df['e_s'] = df['e'] * df['s']
df['e3'] = df['e'] ** 3

# Clean data
df_clean = df.dropna(subset=['lw', 'e', 's', 'male', 'black', 'year', 'cohort_5yr']).copy()
print(f"Clean sample size: {len(df_clean)}")

# ============================================================================
# Helper Functions
# ============================================================================

def run_regression(df_sub, y_var, x_vars, cluster_var=None):
    """Run OLS regression with optional clustering."""
    df_reg = df_sub.dropna(subset=[y_var] + x_vars).copy()

    y = df_reg[y_var].values.astype(np.float64)
    X_data = df_reg[x_vars].values.astype(np.float64)
    X = sm.add_constant(X_data)

    if cluster_var and cluster_var in df_reg.columns:
        groups = df_reg[cluster_var].values
        model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': groups}, use_t=True)
    else:
        model = sm.OLS(y, X).fit(cov_type='HC1', use_t=True)

    # Create proper index
    params = pd.Series(model.params, index=['const'] + x_vars)
    bse = pd.Series(model.bse, index=['const'] + x_vars)
    pvals = pd.Series(model.pvalues, index=['const'] + x_vars)

    return {
        'params': params,
        'bse': bse,
        'pvalues': pvals,
        'nobs': int(model.nobs),
        'rsquared': float(model.rsquared)
    }


def create_result_row(spec_id, spec_tree_path, model_result, treatment_var,
                      controls_desc, fixed_effects, cluster_var, sample_desc):
    """Create a standardized result row."""

    coef = float(model_result['params'][treatment_var])
    se = float(model_result['bse'][treatment_var])
    pval = float(model_result['pvalues'][treatment_var])
    tstat = coef / se if se > 0 else np.nan

    # Build coefficient vector
    controls = []
    for var in model_result['params'].index:
        if var != treatment_var and var != 'const':
            controls.append({
                'var': var,
                'coef': float(model_result['params'][var]),
                'se': float(model_result['bse'][var]),
                'pval': float(model_result['pvalues'][var])
            })

    coef_vector = {
        'treatment': {
            'var': treatment_var,
            'coef': coef,
            'se': se,
            'pval': pval
        },
        'controls': controls,
        'fixed_effects': fixed_effects.split(', ') if fixed_effects else [],
        'diagnostics': {}
    }

    return {
        'paper_id': '235621-V1',
        'journal': 'AER',
        'paper_title': 'The Price of Experience',
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'outcome_var': 'lw',
        'treatment_var': treatment_var,
        'coefficient': coef,
        'std_error': se,
        't_stat': tstat,
        'p_value': pval,
        'ci_lower': coef - 1.96 * se,
        'ci_upper': coef + 1.96 * se,
        'n_obs': model_result['nobs'],
        'r_squared': model_result['rsquared'],
        'coefficient_vector_json': json.dumps(coef_vector),
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': 'OLS',
        'estimation_script': 'scripts/paper_analyses/235621-V1.py'
    }


def add_year_dummies(df_sub):
    """Add year dummies to dataframe."""
    dummies = pd.get_dummies(df_sub['year'], prefix='yr', drop_first=True)
    dummies = dummies.astype(float)
    return pd.concat([df_sub.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1), dummies.columns.tolist()


def add_cohort_dummies(df_sub):
    """Add cohort dummies to dataframe."""
    dummies = pd.get_dummies(df_sub['cohort_5yr'], prefix='coh', drop_first=True)
    dummies = dummies.astype(float)
    return pd.concat([df_sub.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1), dummies.columns.tolist()


# ============================================================================
# Results Storage
# ============================================================================

results = []

# ============================================================================
# BASELINE SPECIFICATION
# ============================================================================

print("\nRunning Baseline Specification...")

df_base, yr_cols = add_year_dummies(df_clean)
baseline_x = ['e', 'e2', 's', 'male', 'black'] + yr_cols

model = run_regression(df_base, 'lw', baseline_x, 'cohort_5yr')
results.append(create_result_row(
    'baseline', 'methods/panel_fixed_effects.md', model, 'e',
    'e2, s, male, black', 'year', 'cohort_5yr',
    f'PSID 1968-2007, N={model["nobs"]}'
))
print(f"  e coef = {model['params']['e']:.4f} (se = {model['bse']['e']:.4f})")

# ============================================================================
# PANEL FE VARIATIONS
# ============================================================================

print("\nRunning Panel FE Variations...")

# 1. No fixed effects (pooled OLS)
print("  panel/fe/none...")
x_no_fe = ['e', 'e2', 's', 'male', 'black']
model = run_regression(df_clean, 'lw', x_no_fe, 'cohort_5yr')
results.append(create_result_row(
    'panel/fe/none', 'methods/panel_fixed_effects.md#fixed-effects-structure', model, 'e',
    'e2, s, male, black', 'none', 'cohort_5yr',
    f'PSID 1968-2007, N={model["nobs"]}'
))

# 2. Cohort FE only
print("  panel/fe/cohort...")
df_coh, coh_cols = add_cohort_dummies(df_clean)
x_coh = ['e', 'e2', 's', 'male', 'black'] + coh_cols
model = run_regression(df_coh, 'lw', x_coh, 'year')
results.append(create_result_row(
    'panel/fe/cohort', 'methods/panel_fixed_effects.md#fixed-effects-structure', model, 'e',
    'e2, s, male, black', 'cohort_5yr', 'year',
    f'PSID 1968-2007, N={model["nobs"]}'
))

# 3. Year FE only (same as baseline but different cluster)
print("  panel/fe/year...")
model = run_regression(df_base, 'lw', baseline_x, 'year')
results.append(create_result_row(
    'panel/fe/year', 'methods/panel_fixed_effects.md#fixed-effects-structure', model, 'e',
    'e2, s, male, black', 'year', 'year',
    f'PSID 1968-2007, N={model["nobs"]}'
))

# 4. Two-way FE (year + cohort)
print("  panel/fe/twoway...")
df_tw, yr_cols = add_year_dummies(df_clean)
df_tw, coh_cols = add_cohort_dummies(df_tw)
x_tw = ['e', 'e2', 's', 'male', 'black'] + yr_cols + coh_cols
model = run_regression(df_tw, 'lw', x_tw, None)
results.append(create_result_row(
    'panel/fe/twoway', 'methods/panel_fixed_effects.md#fixed-effects-structure', model, 'e',
    'e2, s, male, black', 'year, cohort_5yr', 'none (robust)',
    f'PSID 1968-2007, N={model["nobs"]}'
))

# ============================================================================
# LEAVE-ONE-OUT ROBUSTNESS
# ============================================================================

print("\nRunning Leave-One-Out Robustness...")

controls_to_drop = ['e2', 's', 'male', 'black']

for control in controls_to_drop:
    print(f"  robust/loo/drop_{control}...")
    df_loo, yr_cols = add_year_dummies(df_clean)
    x_loo = [c for c in ['e', 'e2', 's', 'male', 'black'] if c != control] + yr_cols
    model = run_regression(df_loo, 'lw', x_loo, 'cohort_5yr')
    results.append(create_result_row(
        f'robust/loo/drop_{control}', 'robustness/leave_one_out.md', model, 'e',
        ', '.join([c for c in ['e2', 's', 'male', 'black'] if c != control]),
        'year', 'cohort_5yr',
        f'PSID 1968-2007, N={model["nobs"]}'
    ))

# ============================================================================
# SINGLE COVARIATE ROBUSTNESS
# ============================================================================

print("\nRunning Single Covariate Robustness...")

# Bivariate (no controls)
print("  robust/single/none...")
df_biv, yr_cols = add_year_dummies(df_clean)
x_biv = ['e'] + yr_cols
model = run_regression(df_biv, 'lw', x_biv, 'cohort_5yr')
results.append(create_result_row(
    'robust/single/none', 'robustness/single_covariate.md', model, 'e',
    'none', 'year', 'cohort_5yr',
    f'PSID 1968-2007, N={model["nobs"]}'
))

# Single covariate additions
for control in ['e2', 's', 'male', 'black']:
    print(f"  robust/single/{control}...")
    df_sing, yr_cols = add_year_dummies(df_clean)
    x_sing = ['e', control] + yr_cols
    model = run_regression(df_sing, 'lw', x_sing, 'cohort_5yr')
    results.append(create_result_row(
        f'robust/single/{control}', 'robustness/single_covariate.md', model, 'e',
        control, 'year', 'cohort_5yr',
        f'PSID 1968-2007, N={model["nobs"]}'
    ))

# ============================================================================
# CLUSTERING VARIATIONS
# ============================================================================

print("\nRunning Clustering Variations...")

df_clus, yr_cols = add_year_dummies(df_clean)
x_clus = ['e', 'e2', 's', 'male', 'black'] + yr_cols

# No clustering (robust SE)
print("  robust/cluster/none...")
model = run_regression(df_clus, 'lw', x_clus, None)
results.append(create_result_row(
    'robust/cluster/none', 'robustness/clustering_variations.md', model, 'e',
    'e2, s, male, black', 'year', 'none (HC1 robust)',
    f'PSID 1968-2007, N={model["nobs"]}'
))

# Cluster by year
print("  robust/cluster/year...")
model = run_regression(df_clus, 'lw', x_clus, 'year')
results.append(create_result_row(
    'robust/cluster/year', 'robustness/clustering_variations.md', model, 'e',
    'e2, s, male, black', 'year', 'year',
    f'PSID 1968-2007, N={model["nobs"]}'
))

# ============================================================================
# SAMPLE RESTRICTIONS
# ============================================================================

print("\nRunning Sample Restriction Robustness...")

# Early period (1968-1987)
print("  robust/sample/early_period...")
df_early = df_clean[df_clean['year'] <= 1987].copy()
df_early, yr_cols = add_year_dummies(df_early)
x_early = ['e', 'e2', 's', 'male', 'black'] + yr_cols
model = run_regression(df_early, 'lw', x_early, 'cohort_5yr')
results.append(create_result_row(
    'robust/sample/early_period', 'robustness/sample_restrictions.md', model, 'e',
    'e2, s, male, black', 'year', 'cohort_5yr',
    f'PSID 1968-1987, N={model["nobs"]}'
))

# Late period (1988-2007)
print("  robust/sample/late_period...")
df_late = df_clean[df_clean['year'] >= 1988].copy()
df_late, yr_cols = add_year_dummies(df_late)
x_late = ['e', 'e2', 's', 'male', 'black'] + yr_cols
model = run_regression(df_late, 'lw', x_late, 'cohort_5yr')
results.append(create_result_row(
    'robust/sample/late_period', 'robustness/sample_restrictions.md', model, 'e',
    'e2, s, male, black', 'year', 'cohort_5yr',
    f'PSID 1988-2007, N={model["nobs"]}'
))

# Male only
print("  robust/sample/male_only...")
df_male = df_clean[df_clean['male'] == 1].copy()
df_male, yr_cols = add_year_dummies(df_male)
x_male = ['e', 'e2', 's', 'black'] + yr_cols
model = run_regression(df_male, 'lw', x_male, 'cohort_5yr')
results.append(create_result_row(
    'robust/sample/male_only', 'robustness/sample_restrictions.md', model, 'e',
    'e2, s, black', 'year', 'cohort_5yr',
    f'PSID 1968-2007 Males, N={model["nobs"]}'
))

# Female only
print("  robust/sample/female_only...")
df_female = df_clean[df_clean['male'] == 0].copy()
df_female, yr_cols = add_year_dummies(df_female)
x_female = ['e', 'e2', 's', 'black'] + yr_cols
model = run_regression(df_female, 'lw', x_female, 'cohort_5yr')
results.append(create_result_row(
    'robust/sample/female_only', 'robustness/sample_restrictions.md', model, 'e',
    'e2, s, black', 'year', 'cohort_5yr',
    f'PSID 1968-2007 Females, N={model["nobs"]}'
))

# College educated only
print("  robust/sample/college_only...")
df_coll = df_clean[df_clean['college'] == 1].copy()
df_coll, yr_cols = add_year_dummies(df_coll)
x_coll = ['e', 'e2', 's', 'male', 'black'] + yr_cols
model = run_regression(df_coll, 'lw', x_coll, 'cohort_5yr')
results.append(create_result_row(
    'robust/sample/college_only', 'robustness/sample_restrictions.md', model, 'e',
    'e2, s, male, black', 'year', 'cohort_5yr',
    f'PSID 1968-2007 College, N={model["nobs"]}'
))

# Non-college only
print("  robust/sample/noncollege_only...")
df_noncoll = df_clean[df_clean['college'] == 0].copy()
df_noncoll, yr_cols = add_year_dummies(df_noncoll)
x_noncoll = ['e', 'e2', 's', 'male', 'black'] + yr_cols
model = run_regression(df_noncoll, 'lw', x_noncoll, 'cohort_5yr')
results.append(create_result_row(
    'robust/sample/noncollege_only', 'robustness/sample_restrictions.md', model, 'e',
    'e2, s, male, black', 'year', 'cohort_5yr',
    f'PSID 1968-2007 Non-College, N={model["nobs"]}'
))

# Trim 1% outliers
print("  robust/sample/trim_1pct...")
q01 = df_clean['lw'].quantile(0.01)
q99 = df_clean['lw'].quantile(0.99)
df_trim = df_clean[(df_clean['lw'] >= q01) & (df_clean['lw'] <= q99)].copy()
df_trim, yr_cols = add_year_dummies(df_trim)
x_trim = ['e', 'e2', 's', 'male', 'black'] + yr_cols
model = run_regression(df_trim, 'lw', x_trim, 'cohort_5yr')
results.append(create_result_row(
    'robust/sample/trim_1pct', 'robustness/sample_restrictions.md', model, 'e',
    'e2, s, male, black', 'year', 'cohort_5yr',
    f'PSID 1968-2007 trimmed 1%, N={model["nobs"]}'
))

# ============================================================================
# FUNCTIONAL FORM ROBUSTNESS
# ============================================================================

print("\nRunning Functional Form Robustness...")

# Cubic in experience
print("  robust/form/cubic_experience...")
df_form, yr_cols = add_year_dummies(df_clean)
x_cubic = ['e', 'e2', 'e3', 's', 'male', 'black'] + yr_cols
model = run_regression(df_form, 'lw', x_cubic, 'cohort_5yr')
results.append(create_result_row(
    'robust/form/cubic_experience', 'robustness/functional_form.md', model, 'e',
    'e2, e3, s, male, black', 'year', 'cohort_5yr',
    f'PSID 1968-2007, N={model["nobs"]}'
))

# Quadratic in schooling
print("  robust/form/quadratic_schooling...")
x_s2 = ['e', 'e2', 's', 's2', 'male', 'black'] + yr_cols
model = run_regression(df_form, 'lw', x_s2, 'cohort_5yr')
results.append(create_result_row(
    'robust/form/quadratic_schooling', 'robustness/functional_form.md', model, 'e',
    'e2, s, s2, male, black', 'year', 'cohort_5yr',
    f'PSID 1968-2007, N={model["nobs"]}'
))

# Experience x Schooling interaction
print("  robust/form/exp_school_interaction...")
x_inter = ['e', 'e2', 's', 'e_s', 'male', 'black'] + yr_cols
model = run_regression(df_form, 'lw', x_inter, 'cohort_5yr')
results.append(create_result_row(
    'robust/form/exp_school_interaction', 'robustness/functional_form.md', model, 'e',
    'e2, s, e_s, male, black', 'year', 'cohort_5yr',
    f'PSID 1968-2007, N={model["nobs"]}'
))

# Linear only (no e^2)
print("  robust/form/linear_experience...")
x_lin = ['e', 's', 'male', 'black'] + yr_cols
model = run_regression(df_form, 'lw', x_lin, 'cohort_5yr')
results.append(create_result_row(
    'robust/form/linear_experience', 'robustness/functional_form.md', model, 'e',
    's, male, black', 'year', 'cohort_5yr',
    f'PSID 1968-2007, N={model["nobs"]}'
))

# ============================================================================
# CUSTOM: DECADE SUBSAMPLES (Time-varying returns)
# ============================================================================

print("\nRunning Custom Specifications (Decade Subsamples)...")

decades = [('1970s', 1970, 1979), ('1980s', 1980, 1989),
           ('1990s', 1990, 1999), ('2000s', 2000, 2007)]

for decade_name, start, end in decades:
    print(f"  custom/decade/{decade_name}...")
    df_dec = df_clean[(df_clean['year'] >= start) & (df_clean['year'] <= end)].copy()
    if len(df_dec) > 1000:
        df_dec, yr_cols = add_year_dummies(df_dec)
        x_dec = ['e', 'e2', 's', 'male', 'black'] + yr_cols
        model = run_regression(df_dec, 'lw', x_dec, 'cohort_5yr')
        results.append(create_result_row(
            f'custom/decade/{decade_name}', 'custom', model, 'e',
            'e2, s, male, black', 'year', 'cohort_5yr',
            f'PSID {start}-{end}, N={model["nobs"]}'
        ))

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*60)
print("Saving Results...")
print("="*60)

results_df = pd.DataFrame(results)
output_file = OUTPUT_PATH + 'specification_results.csv'
results_df.to_csv(output_file, index=False)
print(f"Saved {len(results_df)} specifications to {output_file}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

print(f"\nTotal specifications: {len(results_df)}")
print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")
print(f"\nCoefficient range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")
print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")

print("\n" + "="*60)
print("BASELINE RESULT")
print("="*60)
baseline_row = results_df[results_df['spec_id'] == 'baseline'].iloc[0]
print(f"Experience coefficient: {baseline_row['coefficient']:.4f}")
print(f"Standard error: {baseline_row['std_error']:.4f}")
print(f"p-value: {baseline_row['p_value']:.6f}")
print(f"95% CI: [{baseline_row['ci_lower']:.4f}, {baseline_row['ci_upper']:.4f}]")

print("\nDone!")
