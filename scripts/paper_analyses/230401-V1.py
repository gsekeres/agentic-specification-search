"""
Specification Search Analysis: 230401-V1
Paper: Are Loans to Minority-Owned Firms Mispriced?

Main Hypothesis: Minority-owned businesses face higher loan interest rates compared to
white-owned businesses, even after controlling for firm, loan, and lender characteristics.

Method: Cross-sectional OLS with state and time fixed effects
Outcome: loanrate_w2 (winsorized interest rate)
Treatment: Race indicators (hisp_50, black_50, asian_50, native_50) vs white_50
Identification: Controlling for observable firm/loan/lender characteristics + FE

Author: Automated Specification Search
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.quantile_regression import QuantReg
import warnings
import json
from datetime import datetime
import os

warnings.filterwarnings('ignore')

# Paths
DATA_PATH = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/230401-V1/final.dta'
OUTPUT_DIR = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/230401-V1'

# Load data
print("Loading data...")
df = pd.read_stata(DATA_PATH)
print(f"Loaded {len(df)} observations")

# Convert all columns to numeric where possible
for col in df.columns:
    if df[col].dtype == 'category':
        df[col + '_cat'] = pd.Categorical(df[col]).codes
        # Keep original for reference

# Ensure numeric types for key variables
df['ind_num'] = pd.Categorical(df['ind']).codes if 'ind' in df.columns else 0
df['statehead_num'] = pd.Categorical(df['statehead']).codes if 'statehead' in df.columns else 0
df['lender_num'] = pd.Categorical(df['lender']).codes if df['lender'].dtype == 'category' else df['lender']

# Convert ceoown to numeric
if 'ceoown' in df.columns:
    df['ceoown_num'] = pd.to_numeric(df['ceoown'].astype(str), errors='coerce')
else:
    df['ceoown_num'] = 0

# Ensure float types for continuous variables
float_cols = ['busage', 'ceoexp', 'ceoage', 'sales_val', 'assets_val', 'loss21',
              'revenuegrow', 'employeegrow', 'conditiongood', 'ltd', 'family',
              'woman_owned', 'creditscore', 'loan', 'newcredit', 'purpose_debt',
              'fixed', 'term', 'smallbank', 'creditunion', 'CDFI', 'fintech',
              'nonbank', 'relength_yr', 'hhi_c', 'c_branches', 'timeid',
              'loanrate_w2', 'loanspread_w2', 'coll', 'collval', 'relength', 'q30']
for col in float_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Create indicator variables for industry (7 categories, drop first)
for i in range(7):
    df[f'ind_{i+1}'] = (df['ind_num'] == i).astype(float)

# Create indicator variables for credit score (3 categories, drop first)
for i in range(3):
    df[f'creditscore_{i+1}'] = (df['creditscore'] == i+1).astype(float)

# Create state fixed effects (absorb largest category)
state_dummies = pd.get_dummies(df['statehead_num'], prefix='state', drop_first=True, dtype=float)
df = pd.concat([df, state_dummies], axis=1)

# Create time fixed effects
time_dummies = pd.get_dummies(df['timeid'], prefix='time', drop_first=True, dtype=float)
df = pd.concat([df, time_dummies], axis=1)

# Define variables
OUTCOME = 'loanrate_w2'
TREATMENTS = ['hisp_50', 'black_50', 'asian_50', 'native_50']
COMPARISON = 'white_50'

# Control variable sets (all numeric)
FIRM_CHARS = ['busage', 'ceoexp', 'ceoage', 'ceoown_num', 'sales_val', 'assets_val',
              'loss21', 'revenuegrow', 'employeegrow', 'conditiongood', 'ltd',
              'family', 'woman_owned']
LOAN_CHARS = ['loan', 'newcredit', 'purpose_debt', 'fixed', 'term']
LENDER_CHARS = ['smallbank', 'creditunion', 'CDFI', 'fintech', 'nonbank',
                'relength_yr', 'hhi_c', 'c_branches']

# Get list of state and time dummies
STATE_DUMMIES = [c for c in df.columns if c.startswith('state_')]
TIME_DUMMIES = [c for c in df.columns if c.startswith('time_')]
IND_DUMMIES = [f'ind_{i}' for i in range(2, 8)]  # drop first for reference
CREDITSCORE_DUMMIES = [f'creditscore_{i}' for i in range(2, 4)]  # drop first for reference

FE_VARS = STATE_DUMMIES + TIME_DUMMIES

ALL_CONTROLS = FIRM_CHARS + IND_DUMMIES + CREDITSCORE_DUMMIES + LOAN_CHARS + LENDER_CHARS

# Results storage
results = []

def run_regression(df_sub, outcome, treatment, controls, cluster_var=None,
                   spec_id='', spec_tree_path='', add_fe=True, cov_type='HC1'):
    """Run OLS regression and store results"""

    # Build control list
    ctrl_list = controls.copy()
    if add_fe:
        ctrl_list = ctrl_list + FE_VARS

    # Filter to treatment vs white comparison
    mask = (df_sub[treatment] == 1) | (df_sub[COMPARISON] == 1)
    df_reg = df_sub[mask].copy()

    # Drop rows with missing values
    all_vars = [outcome, treatment] + ctrl_list
    all_vars = [v for v in all_vars if v in df_reg.columns]
    df_reg = df_reg.dropna(subset=all_vars)

    if len(df_reg) < 50:
        return None

    # Prepare X and y - ensure all numeric
    y = df_reg[outcome].astype(float)
    X_vars = [treatment] + ctrl_list
    X = df_reg[X_vars].astype(float)
    X = sm.add_constant(X)

    try:
        # Run regression
        if cluster_var and cluster_var in df_reg.columns:
            groups = df_reg[cluster_var].astype(int)
            model = sm.OLS(y, X, missing='drop').fit(
                cov_type='cluster',
                cov_kwds={'groups': groups}
            )
        else:
            model = sm.OLS(y, X, missing='drop').fit(cov_type=cov_type)

        # Extract treatment coefficient
        coef = model.params[treatment]
        se = model.bse[treatment]
        pval = model.pvalues[treatment]
        ci = model.conf_int().loc[treatment]

        # Build coefficient vector
        coef_vector = {
            'treatment': {
                'var': treatment,
                'coef': float(coef),
                'se': float(se),
                'pval': float(pval),
                'ci_lower': float(ci[0]),
                'ci_upper': float(ci[1])
            },
            'controls': [],
            'n_obs': int(model.nobs),
            'r_squared': float(model.rsquared),
            'adj_r_squared': float(model.rsquared_adj),
            'f_stat': float(model.fvalue) if model.fvalue else None,
            'f_pval': float(model.f_pvalue) if model.f_pvalue else None
        }

        # Add control coefficients (non-FE)
        for ctrl in controls:
            if ctrl in model.params.index:
                coef_vector['controls'].append({
                    'var': ctrl,
                    'coef': float(model.params[ctrl]),
                    'se': float(model.bse[ctrl]),
                    'pval': float(model.pvalues[ctrl])
                })

        result = {
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'treatment_var': treatment,
            'outcome_var': outcome,
            'coefficient': coef,
            'std_error': se,
            'p_value': pval,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'n_obs': model.nobs,
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'f_stat': model.fvalue,
            'controls_used': ','.join(controls),
            'fixed_effects': 'state,time' if add_fe else 'none',
            'cluster_var': cluster_var if cluster_var else 'none',
            'cov_type': cov_type,
            'coefficient_vector_json': json.dumps(coef_vector)
        }

        return result

    except Exception as e:
        print(f"Error in {spec_id} ({treatment}): {e}")
        return None


def run_quantile_regression(df_sub, outcome, treatment, controls, quantile=0.5,
                            spec_id='', spec_tree_path='', add_fe=True):
    """Run quantile regression"""

    ctrl_list = controls.copy()
    if add_fe:
        ctrl_list = ctrl_list + FE_VARS

    mask = (df_sub[treatment] == 1) | (df_sub[COMPARISON] == 1)
    df_reg = df_sub[mask].copy()

    all_vars = [outcome, treatment] + ctrl_list
    all_vars = [v for v in all_vars if v in df_reg.columns]
    df_reg = df_reg.dropna(subset=all_vars)

    if len(df_reg) < 50:
        return None

    y = df_reg[outcome].astype(float)
    X_vars = [treatment] + ctrl_list
    X = df_reg[X_vars].astype(float)
    X = sm.add_constant(X)

    try:
        model = QuantReg(y, X).fit(q=quantile)

        coef = model.params[treatment]
        se = model.bse[treatment]
        pval = model.pvalues[treatment]

        result = {
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'treatment_var': treatment,
            'outcome_var': outcome,
            'coefficient': coef,
            'std_error': se,
            'p_value': pval,
            'ci_lower': coef - 1.96*se,
            'ci_upper': coef + 1.96*se,
            'n_obs': len(y),
            'r_squared': model.prsquared,
            'adj_r_squared': None,
            'f_stat': None,
            'controls_used': ','.join(controls),
            'fixed_effects': 'state,time' if add_fe else 'none',
            'cluster_var': 'none',
            'cov_type': f'quantile_{quantile}',
            'coefficient_vector_json': json.dumps({
                'treatment': {'var': treatment, 'coef': float(coef), 'se': float(se), 'pval': float(pval)},
                'quantile': quantile
            })
        }
        return result
    except Exception as e:
        print(f"Error in quantile {spec_id}: {e}")
        return None


print("\n" + "="*70)
print("RUNNING SPECIFICATION SEARCH")
print("="*70)

# ============================================================================
# 1. BASELINE SPECIFICATIONS (replicate Table 2)
# ============================================================================
print("\n1. BASELINE SPECIFICATIONS (Table 2)")
print("-" * 50)

for treat in TREATMENTS:
    # Full model (column 5 in Table 2)
    result = run_regression(
        df, OUTCOME, treat,
        controls=ALL_CONTROLS,
        cluster_var='statehead_num',
        spec_id='baseline',
        spec_tree_path='methods/cross_sectional_ols.md#baseline'
    )
    if result:
        results.append(result)
        print(f"  {treat}: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")

# ============================================================================
# 2. OLS CONTROL SET VARIATIONS
# ============================================================================
print("\n2. CONTROL SET VARIATIONS")
print("-" * 50)

# No controls
for treat in TREATMENTS:
    result = run_regression(
        df, OUTCOME, treat,
        controls=[],
        cluster_var='statehead_num',
        spec_id='ols/controls/none',
        spec_tree_path='methods/cross_sectional_ols.md#control-sets',
        add_fe=False
    )
    if result:
        results.append(result)
        print(f"  {treat} (no controls): coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# FE only
for treat in TREATMENTS:
    result = run_regression(
        df, OUTCOME, treat,
        controls=[],
        cluster_var='statehead_num',
        spec_id='ols/controls/fe_only',
        spec_tree_path='methods/cross_sectional_ols.md#control-sets',
        add_fe=True
    )
    if result:
        results.append(result)
        print(f"  {treat} (FE only): coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# Firm characteristics only
for treat in TREATMENTS:
    result = run_regression(
        df, OUTCOME, treat,
        controls=FIRM_CHARS + IND_DUMMIES + CREDITSCORE_DUMMIES,
        cluster_var='statehead_num',
        spec_id='ols/controls/firm_only',
        spec_tree_path='methods/cross_sectional_ols.md#control-sets'
    )
    if result:
        results.append(result)
        print(f"  {treat} (firm only): coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# Firm + loan characteristics
for treat in TREATMENTS:
    result = run_regression(
        df, OUTCOME, treat,
        controls=FIRM_CHARS + IND_DUMMIES + CREDITSCORE_DUMMIES + LOAN_CHARS,
        cluster_var='statehead_num',
        spec_id='ols/controls/firm_loan',
        spec_tree_path='methods/cross_sectional_ols.md#control-sets'
    )
    if result:
        results.append(result)
        print(f"  {treat} (firm+loan): coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# ============================================================================
# 3. STANDARD ERROR VARIATIONS
# ============================================================================
print("\n3. STANDARD ERROR VARIATIONS")
print("-" * 50)

# Classical SE (no clustering)
for treat in TREATMENTS:
    result = run_regression(
        df, OUTCOME, treat,
        controls=ALL_CONTROLS,
        cluster_var=None,
        spec_id='ols/se/classical',
        spec_tree_path='methods/cross_sectional_ols.md#standard-errors',
        cov_type='nonrobust'
    )
    if result:
        results.append(result)
        print(f"  {treat} (classical): se={result['std_error']:.4f}")

# Robust SE (no clustering)
for treat in TREATMENTS:
    result = run_regression(
        df, OUTCOME, treat,
        controls=ALL_CONTROLS,
        cluster_var=None,
        spec_id='ols/se/robust',
        spec_tree_path='methods/cross_sectional_ols.md#standard-errors',
        cov_type='HC1'
    )
    if result:
        results.append(result)
        print(f"  {treat} (HC1): se={result['std_error']:.4f}")

# HC3 robust SE
for treat in TREATMENTS:
    result = run_regression(
        df, OUTCOME, treat,
        controls=ALL_CONTROLS,
        cluster_var=None,
        spec_id='ols/se/hc3',
        spec_tree_path='methods/cross_sectional_ols.md#standard-errors',
        cov_type='HC3'
    )
    if result:
        results.append(result)
        print(f"  {treat} (HC3): se={result['std_error']:.4f}")

# ============================================================================
# 4. CLUSTERING VARIATIONS
# ============================================================================
print("\n4. CLUSTERING VARIATIONS")
print("-" * 50)

# No clustering
for treat in TREATMENTS:
    result = run_regression(
        df, OUTCOME, treat,
        controls=ALL_CONTROLS,
        cluster_var=None,
        spec_id='robust/cluster/none',
        spec_tree_path='robustness/clustering_variations.md#single-level-clustering',
        cov_type='HC1'
    )
    if result:
        results.append(result)
        print(f"  {treat} (no cluster): se={result['std_error']:.4f}")

# Cluster by state (baseline)
for treat in TREATMENTS:
    result = run_regression(
        df, OUTCOME, treat,
        controls=ALL_CONTROLS,
        cluster_var='statehead_num',
        spec_id='robust/cluster/state',
        spec_tree_path='robustness/clustering_variations.md#single-level-clustering'
    )
    if result:
        results.append(result)
        print(f"  {treat} (state cluster): se={result['std_error']:.4f}")

# Cluster by industry
for treat in TREATMENTS:
    result = run_regression(
        df, OUTCOME, treat,
        controls=ALL_CONTROLS,
        cluster_var='ind_num',
        spec_id='robust/cluster/industry',
        spec_tree_path='robustness/clustering_variations.md#single-level-clustering'
    )
    if result:
        results.append(result)
        print(f"  {treat} (industry cluster): se={result['std_error']:.4f}")

# ============================================================================
# 5. LEAVE-ONE-OUT ROBUSTNESS
# ============================================================================
print("\n5. LEAVE-ONE-OUT ROBUSTNESS")
print("-" * 50)

# Focus on black_50 (primary treatment in paper)
primary_treat = 'black_50'

for control_to_drop in FIRM_CHARS + LOAN_CHARS + LENDER_CHARS:
    loo_controls = [c for c in ALL_CONTROLS if c != control_to_drop]
    result = run_regression(
        df, OUTCOME, primary_treat,
        controls=loo_controls,
        cluster_var='statehead_num',
        spec_id=f'robust/loo/drop_{control_to_drop}',
        spec_tree_path='robustness/leave_one_out.md'
    )
    if result:
        results.append(result)
        print(f"  drop {control_to_drop}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# ============================================================================
# 6. SINGLE COVARIATE ANALYSIS
# ============================================================================
print("\n6. SINGLE COVARIATE ANALYSIS")
print("-" * 50)

# Bivariate (no controls)
result = run_regression(
    df, OUTCOME, primary_treat,
    controls=[],
    cluster_var='statehead_num',
    spec_id='robust/single/none',
    spec_tree_path='robustness/single_covariate.md',
    add_fe=False
)
if result:
    results.append(result)
    bivariate_coef = result['coefficient']
    print(f"  bivariate: coef={result['coefficient']:.4f}")

# Single covariate
for single_ctrl in FIRM_CHARS[:6] + LOAN_CHARS[:3] + LENDER_CHARS[:3]:
    result = run_regression(
        df, OUTCOME, primary_treat,
        controls=[single_ctrl],
        cluster_var='statehead_num',
        spec_id=f'robust/single/{single_ctrl}',
        spec_tree_path='robustness/single_covariate.md',
        add_fe=False
    )
    if result:
        results.append(result)
        print(f"  + {single_ctrl}: coef={result['coefficient']:.4f}")

# ============================================================================
# 7. FUNCTIONAL FORM VARIATIONS
# ============================================================================
print("\n7. FUNCTIONAL FORM VARIATIONS")
print("-" * 50)

# Log outcome (add 0.01 to avoid log(0))
df['loanrate_log'] = np.log(df['loanrate_w2'] + 0.01)

for treat in TREATMENTS:
    result = run_regression(
        df, 'loanrate_log', treat,
        controls=ALL_CONTROLS,
        cluster_var='statehead_num',
        spec_id='robust/form/y_log',
        spec_tree_path='robustness/functional_form.md#outcome-variable-transformations'
    )
    if result:
        results.append(result)
        print(f"  {treat} (log outcome): coef={result['coefficient']:.4f}")

# Asinh transformation
df['loanrate_asinh'] = np.arcsinh(df['loanrate_w2'])

for treat in TREATMENTS:
    result = run_regression(
        df, 'loanrate_asinh', treat,
        controls=ALL_CONTROLS,
        cluster_var='statehead_num',
        spec_id='robust/form/y_asinh',
        spec_tree_path='robustness/functional_form.md#outcome-variable-transformations'
    )
    if result:
        results.append(result)
        print(f"  {treat} (asinh outcome): coef={result['coefficient']:.4f}")

# ============================================================================
# 8. QUANTILE REGRESSIONS
# ============================================================================
print("\n8. QUANTILE REGRESSIONS")
print("-" * 50)

for treat in TREATMENTS:
    # Median regression
    result = run_quantile_regression(
        df, OUTCOME, treat,
        controls=ALL_CONTROLS,
        quantile=0.5,
        spec_id='ols/method/quantile_median',
        spec_tree_path='methods/cross_sectional_ols.md#estimation-method'
    )
    if result:
        results.append(result)
        print(f"  {treat} (q50): coef={result['coefficient']:.4f}")

    # 25th percentile
    result = run_quantile_regression(
        df, OUTCOME, treat,
        controls=ALL_CONTROLS,
        quantile=0.25,
        spec_id='ols/method/quantile_25',
        spec_tree_path='methods/cross_sectional_ols.md#estimation-method'
    )
    if result:
        results.append(result)
        print(f"  {treat} (q25): coef={result['coefficient']:.4f}")

    # 75th percentile
    result = run_quantile_regression(
        df, OUTCOME, treat,
        controls=ALL_CONTROLS,
        quantile=0.75,
        spec_id='ols/method/quantile_75',
        spec_tree_path='methods/cross_sectional_ols.md#estimation-method'
    )
    if result:
        results.append(result)
        print(f"  {treat} (q75): coef={result['coefficient']:.4f}")

# ============================================================================
# 9. SAMPLE RESTRICTIONS
# ============================================================================
print("\n9. SAMPLE RESTRICTIONS")
print("-" * 50)

# Trim outliers (1%)
q01 = df[OUTCOME].quantile(0.01)
q99 = df[OUTCOME].quantile(0.99)
df_trimmed = df[(df[OUTCOME] >= q01) & (df[OUTCOME] <= q99)]

for treat in TREATMENTS:
    result = run_regression(
        df_trimmed, OUTCOME, treat,
        controls=ALL_CONTROLS,
        cluster_var='statehead_num',
        spec_id='robust/sample/trim_1pct',
        spec_tree_path='robustness/sample_restrictions.md#outlier-handling'
    )
    if result:
        results.append(result)
        print(f"  {treat} (trim 1%): coef={result['coefficient']:.4f}, n={result['n_obs']}")

# Higher loan amounts only (>=500k)
df_highloan = df[df['q30'] >= 500000]

for treat in TREATMENTS:
    result = run_regression(
        df_highloan, OUTCOME, treat,
        controls=ALL_CONTROLS,
        cluster_var='statehead_num',
        spec_id='robust/sample/high_loan_amount',
        spec_tree_path='robustness/sample_restrictions.md'
    )
    if result:
        results.append(result)
        print(f"  {treat} (high loan): coef={result['coefficient']:.4f}, n={result['n_obs']}")

# Lower loan amounts (<500k)
df_lowloan = df[df['q30'] < 500000]

for treat in TREATMENTS:
    result = run_regression(
        df_lowloan, OUTCOME, treat,
        controls=ALL_CONTROLS,
        cluster_var='statehead_num',
        spec_id='robust/sample/low_loan_amount',
        spec_tree_path='robustness/sample_restrictions.md'
    )
    if result:
        results.append(result)
        print(f"  {treat} (low loan): coef={result['coefficient']:.4f}, n={result['n_obs']}")

# Bank lenders only (lender_num <= 1 since 0=large bank, 1=small bank)
df_bank = df[df['lender_num'] <= 1]

for treat in TREATMENTS:
    result = run_regression(
        df_bank, OUTCOME, treat,
        controls=ALL_CONTROLS,
        cluster_var='statehead_num',
        spec_id='robust/sample/bank_only',
        spec_tree_path='robustness/sample_restrictions.md'
    )
    if result:
        results.append(result)
        print(f"  {treat} (bank only): coef={result['coefficient']:.4f}, n={result['n_obs']}")

# Non-bank lenders
df_nonbank = df[df['lender_num'] > 1]

for treat in TREATMENTS:
    result = run_regression(
        df_nonbank, OUTCOME, treat,
        controls=ALL_CONTROLS,
        cluster_var='statehead_num',
        spec_id='robust/sample/nonbank_only',
        spec_tree_path='robustness/sample_restrictions.md'
    )
    if result:
        results.append(result)
        print(f"  {treat} (nonbank only): coef={result['coefficient']:.4f}, n={result['n_obs']}")

# Short relationship (<= 5 years)
df_shortrel = df[df['relength'] <= 2]

for treat in TREATMENTS:
    result = run_regression(
        df_shortrel, OUTCOME, treat,
        controls=ALL_CONTROLS,
        cluster_var='statehead_num',
        spec_id='robust/sample/short_relationship',
        spec_tree_path='robustness/sample_restrictions.md'
    )
    if result:
        results.append(result)
        print(f"  {treat} (short rel): coef={result['coefficient']:.4f}, n={result['n_obs']}")

# Long relationship (> 5 years)
df_longrel = df[df['relength'] >= 3]

for treat in TREATMENTS:
    result = run_regression(
        df_longrel, OUTCOME, treat,
        controls=ALL_CONTROLS,
        cluster_var='statehead_num',
        spec_id='robust/sample/long_relationship',
        spec_tree_path='robustness/sample_restrictions.md'
    )
    if result:
        results.append(result)
        print(f"  {treat} (long rel): coef={result['coefficient']:.4f}, n={result['n_obs']}")

# Loan vs line of credit
df_loan = df[df['loan'] == 1]

for treat in TREATMENTS:
    result = run_regression(
        df_loan, OUTCOME, treat,
        controls=ALL_CONTROLS,
        cluster_var='statehead_num',
        spec_id='robust/sample/loan_only',
        spec_tree_path='robustness/sample_restrictions.md'
    )
    if result:
        results.append(result)
        print(f"  {treat} (loan only): coef={result['coefficient']:.4f}, n={result['n_obs']}")

df_loc = df[df['loan'] == 0]

for treat in TREATMENTS:
    result = run_regression(
        df_loc, OUTCOME, treat,
        controls=ALL_CONTROLS,
        cluster_var='statehead_num',
        spec_id='robust/sample/loc_only',
        spec_tree_path='robustness/sample_restrictions.md'
    )
    if result:
        results.append(result)
        print(f"  {treat} (LOC only): coef={result['coefficient']:.4f}, n={result['n_obs']}")

# New credit vs renewal
df_new = df[df['newcredit'] == 1]

for treat in TREATMENTS:
    result = run_regression(
        df_new, OUTCOME, treat,
        controls=ALL_CONTROLS,
        cluster_var='statehead_num',
        spec_id='robust/sample/new_credit',
        spec_tree_path='robustness/sample_restrictions.md'
    )
    if result:
        results.append(result)
        print(f"  {treat} (new credit): coef={result['coefficient']:.4f}, n={result['n_obs']}")

df_renewal = df[df['newcredit'] == 0]

for treat in TREATMENTS:
    result = run_regression(
        df_renewal, OUTCOME, treat,
        controls=ALL_CONTROLS,
        cluster_var='statehead_num',
        spec_id='robust/sample/renewal',
        spec_tree_path='robustness/sample_restrictions.md'
    )
    if result:
        results.append(result)
        print(f"  {treat} (renewal): coef={result['coefficient']:.4f}, n={result['n_obs']}")

# ============================================================================
# 10. ALTERNATIVE OUTCOMES
# ============================================================================
print("\n10. ALTERNATIVE OUTCOMES")
print("-" * 50)

# Loan spread
for treat in TREATMENTS:
    result = run_regression(
        df, 'loanspread_w2', treat,
        controls=ALL_CONTROLS,
        cluster_var='statehead_num',
        spec_id='custom/outcome_loanspread',
        spec_tree_path='custom'
    )
    if result:
        results.append(result)
        print(f"  {treat} (loan spread): coef={result['coefficient']:.4f}")

# Collateral required
for treat in TREATMENTS:
    result = run_regression(
        df, 'coll', treat,
        controls=ALL_CONTROLS,
        cluster_var='statehead_num',
        spec_id='custom/outcome_collateral',
        spec_tree_path='custom'
    )
    if result:
        results.append(result)
        print(f"  {treat} (collateral): coef={result['coefficient']:.4f}")

# Collateral value
for treat in TREATMENTS:
    result = run_regression(
        df, 'collval', treat,
        controls=ALL_CONTROLS,
        cluster_var='statehead_num',
        spec_id='custom/outcome_collval',
        spec_tree_path='custom'
    )
    if result:
        results.append(result)
        print(f"  {treat} (coll value): coef={result['coefficient']:.4f}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

results_df = pd.DataFrame(results)
output_path = os.path.join(OUTPUT_DIR, 'specification_results.csv')
results_df.to_csv(output_path, index=False)
print(f"Saved {len(results_df)} specifications to {output_path}")

# Print summary
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

for treat in TREATMENTS:
    treat_results = results_df[results_df['treatment_var'] == treat]
    if len(treat_results) > 0:
        print(f"\n{treat}:")
        print(f"  N specifications: {len(treat_results)}")
        print(f"  Coefficient range: [{treat_results['coefficient'].min():.4f}, {treat_results['coefficient'].max():.4f}]")
        print(f"  Mean coefficient: {treat_results['coefficient'].mean():.4f}")
        print(f"  Significant at 5%: {(treat_results['p_value'] < 0.05).sum()} / {len(treat_results)}")
        print(f"  Significant at 10%: {(treat_results['p_value'] < 0.10).sum()} / {len(treat_results)}")

print("\nAnalysis complete!")
