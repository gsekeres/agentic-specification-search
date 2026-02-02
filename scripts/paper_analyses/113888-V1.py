#!/usr/bin/env python3
"""
Specification Search for Paper 113888-V1
"Keep it Simple: A Field Experiment on Information and Credit Application Behavior"

This is a randomized controlled trial (RCT) studying the effects of financial training
on microenterprise outcomes in the Dominican Republic.

Treatment: Two types of training sessions:
    - Accounting training (treat_acc)
    - Rule-of-thumb/Separation training (treat_rot)

Main outcomes:
    - e_zBusPrac: Standardized index of business practices
    - e_zSales: Standardized sales index
    - e_repAnyMistake: Reporting any mistake indicator
    - e_salesWkBad_w01: Sales in bad week (winsorized)

Method: Cross-sectional OLS with ANCOVA (baseline covariate adjustment)
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

# Try to import pyfixest, fall back to statsmodels
try:
    import pyfixest as pf
    HAS_PYFIXEST = True
except ImportError:
    HAS_PYFIXEST = False

import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

# Configuration
PAPER_ID = "113888-V1"
PAPER_TITLE = "Keep it Simple: A Field Experiment on Information and Credit Application Behavior"
JOURNAL = "AER"
DATA_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/113888-V1/Keep-it-Simple-Replication-Files/kisDataFinal.dta"
OUTPUT_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/113888-V1/specification_results.csv"

# Load data
print("Loading data...")
df = pd.read_stata(DATA_PATH)
print(f"Data shape: {df.shape}")

# Create business ownership indicator FIRST (before converting categories)
# e_busOwn is categorical with 'yes'/'no' values
# Use cat.codes: 1 = 'yes', 0 = 'no', -1 = NaN
df['e_busOwn_num'] = (df['e_busOwn'].cat.codes == 1).astype(float)
print(f"Business owners: {df['e_busOwn_num'].sum()}")

# Create barrio numeric codes and string version (before conversion)
df['barrio_num'] = pd.Categorical(df['barrio']).codes
df['barrio_str'] = df['barrio'].astype(str)

# Identify yes/no categorical columns and convert them to 0/1
yes_no_cols = []
for col in df.columns:
    if df[col].dtype.name == 'category':
        cats = df[col].cat.categories.tolist()
        if set([c.lower() if isinstance(c, str) else c for c in cats]) == {'yes', 'no'}:
            yes_no_cols.append(col)
            # yes=1, no=0
            df[col + '_num'] = (df[col].astype(str).str.lower() == 'yes').astype(float)

# Store sex before conversion
df['b_sex_num'] = (df['b_sex'].astype(str).str.lower() == 'female').astype(float)

# Convert categorical columns to numeric
for col in df.columns:
    if col in ['e_busOwn_num', 'barrio_num', 'b_sex_num'] or col.endswith('_num'):
        continue
    if df[col].dtype.name == 'category':
        df[col] = df[col].astype(str)
        # Try to convert to numeric if possible
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            pass

# Create numeric versions of key variables
df['treat'] = pd.to_numeric(df['treat'], errors='coerce')
df['treat_acc'] = pd.to_numeric(df['treat_acc'], errors='coerce')
df['treat_rot'] = pd.to_numeric(df['treat_rot'], errors='coerce')

# Primary outcomes
primary_outcomes = ['e_zBusPrac', 'e_zSales', 'e_repAnyMistake', 'e_salesWkBad_w01']

# Secondary outcomes for Table 2
secondary_outcomes_busprac = ['e_cashSep', 'e_keepAcct', 'e_acctSep', 'e_cashSetAside',
                              'e_calcSales', 'e_save', 'e_saveTotal_w01']
secondary_outcomes_repqual = ['e_profitDiffWinWk', 'e_profitAbsDiffWk']
secondary_outcomes_busperf = ['e_empTotal', 'e_salesAvgWk_w01']

# Controls used in paper
paper_controls = ['i_bus1', 'i_bus2', 'i_bus3', 'i_bus4', 'monto_dese', 'savings']

# Convert controls to numeric
for ctrl in paper_controls:
    df[ctrl] = pd.to_numeric(df[ctrl], errors='coerce')

# Store all results
results = []

def run_regression(formula, data, cluster_var=None, spec_id="", spec_tree_path="",
                   outcome_var="", treatment_var="", controls_desc="",
                   sample_desc="Full sample", fixed_effects="None",
                   model_type="OLS"):
    """Run regression and extract results."""
    try:
        # Use statsmodels for OLS
        model = smf.ols(formula, data=data.dropna(subset=[c for c in data.columns if c in formula]))

        if cluster_var and cluster_var in data.columns:
            # Clustered standard errors
            result = model.fit(cov_type='cluster',
                               cov_kwds={'groups': data.dropna(subset=[c for c in data.columns if c in formula])[cluster_var]})
        else:
            # Robust standard errors
            result = model.fit(cov_type='HC1')

        # Find treatment coefficient
        treat_coef = None
        treat_se = None
        treat_pval = None
        treat_tstat = None

        for param in result.params.index:
            if 'treat' in param.lower():
                treat_coef = result.params[param]
                treat_se = result.bse[param]
                treat_pval = result.pvalues[param]
                treat_tstat = result.tvalues[param]
                treatment_var = param
                break

        if treat_coef is None:
            return None

        # Confidence interval
        ci_lower = treat_coef - 1.96 * treat_se
        ci_upper = treat_coef + 1.96 * treat_se

        # Build coefficient vector JSON
        coef_vector = {
            "treatment": {
                "var": treatment_var,
                "coef": float(treat_coef),
                "se": float(treat_se),
                "pval": float(treat_pval)
            },
            "controls": [],
            "fixed_effects": [],
            "diagnostics": {
                "r_squared": float(result.rsquared),
                "adj_r_squared": float(result.rsquared_adj),
                "f_stat": float(result.fvalue) if result.fvalue else None,
                "f_pval": float(result.f_pvalue) if result.f_pvalue else None
            }
        }

        # Add control coefficients
        for param in result.params.index:
            if param != 'Intercept' and 'treat' not in param.lower():
                coef_vector["controls"].append({
                    "var": param,
                    "coef": float(result.params[param]),
                    "se": float(result.bse[param]),
                    "pval": float(result.pvalues[param])
                })

        return {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': float(treat_coef),
            'std_error': float(treat_se),
            't_stat': float(treat_tstat),
            'p_value': float(treat_pval),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_obs': int(result.nobs),
            'r_squared': float(result.rsquared),
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects,
            'controls_desc': controls_desc,
            'cluster_var': cluster_var if cluster_var else "None",
            'model_type': model_type,
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
    except Exception as e:
        print(f"Error in {spec_id}: {e}")
        return None

# ============================================================================
# BASELINE SPECIFICATIONS (Table 2 replication)
# ============================================================================
print("\n" + "="*80)
print("BASELINE SPECIFICATIONS")
print("="*80)

# The paper's main specification (Table 2) for business owners
# Outcome ~ baseline_outcome + treat_acc + treat_rot, clustered by barrio
# Sample: business owners only (e_busOwn == 1)

# Create business owner sample
df_bus = df[df['e_busOwn_num'] == 1].copy()
print(f"Business owner sample size: {len(df_bus)}")

# Main outcomes for Table 2
main_outcomes = ['zBusPrac', 'repAnyMistake', 'zSales', 'salesWkBad_w01']

for outcome in main_outcomes:
    e_var = f'e_{outcome}'
    b_var = f'b_{outcome}'

    if e_var not in df_bus.columns or b_var not in df_bus.columns:
        continue

    # Convert to numeric
    df_bus[e_var] = pd.to_numeric(df_bus[e_var], errors='coerce')
    df_bus[b_var] = pd.to_numeric(df_bus[b_var], errors='coerce')

    # Baseline with separate treatment arms
    formula = f'{e_var} ~ {b_var} + treat_acc + treat_rot'

    # Run for treat_rot (Rule-of-Thumb)
    result = run_regression(
        formula=formula,
        data=df_bus,
        cluster_var='barrio_num',
        spec_id='baseline',
        spec_tree_path='methods/cross_sectional_ols.md',
        outcome_var=e_var,
        treatment_var='treat_rot',
        controls_desc=f'Baseline {b_var}',
        sample_desc='Business owners only',
        model_type='OLS'
    )
    if result:
        result['treatment_var'] = 'treat_rot'
        # Extract treat_rot coefficient
        model = smf.ols(formula, data=df_bus.dropna(subset=[e_var, b_var, 'treat_acc', 'treat_rot'])).fit(
            cov_type='cluster', cov_kwds={'groups': df_bus.dropna(subset=[e_var, b_var, 'treat_acc', 'treat_rot'])['barrio_num']})
        result['coefficient'] = float(model.params['treat_rot'])
        result['std_error'] = float(model.bse['treat_rot'])
        result['t_stat'] = float(model.tvalues['treat_rot'])
        result['p_value'] = float(model.pvalues['treat_rot'])
        result['ci_lower'] = result['coefficient'] - 1.96 * result['std_error']
        result['ci_upper'] = result['coefficient'] + 1.96 * result['std_error']
        results.append(result)
        print(f"Baseline {e_var} (treat_rot): coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# ============================================================================
# BASELINE WITH "ANY TREATMENT" (pooled)
# ============================================================================
print("\n" + "="*80)
print("BASELINE WITH POOLED TREATMENT")
print("="*80)

for outcome in main_outcomes:
    e_var = f'e_{outcome}'
    b_var = f'b_{outcome}'

    if e_var not in df_bus.columns or b_var not in df_bus.columns:
        continue

    df_bus[e_var] = pd.to_numeric(df_bus[e_var], errors='coerce')
    df_bus[b_var] = pd.to_numeric(df_bus[b_var], errors='coerce')

    # Pooled treatment
    formula = f'{e_var} ~ {b_var} + treat'

    result = run_regression(
        formula=formula,
        data=df_bus,
        cluster_var='barrio_num',
        spec_id='baseline_pooled',
        spec_tree_path='methods/cross_sectional_ols.md',
        outcome_var=e_var,
        treatment_var='treat',
        controls_desc=f'Baseline {b_var}',
        sample_desc='Business owners only',
        model_type='OLS'
    )
    if result:
        results.append(result)
        print(f"Baseline pooled {e_var}: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

# ============================================================================
# CONTROL VARIATIONS: Leave-One-Out
# ============================================================================
print("\n" + "="*80)
print("CONTROL VARIATIONS: LEAVE-ONE-OUT")
print("="*80)

# Main outcome for control variations
main_outcome = 'e_zBusPrac'
main_baseline = 'b_zBusPrac'

df_bus[main_outcome] = pd.to_numeric(df_bus[main_outcome], errors='coerce')
df_bus[main_baseline] = pd.to_numeric(df_bus[main_baseline], errors='coerce')

# With controls
full_controls = ['i_bus1', 'i_bus2', 'i_bus3', 'i_bus4', 'monto_dese', 'savings']

# Leave-one-out for each control
for ctrl in full_controls:
    remaining = [c for c in full_controls if c != ctrl]
    ctrl_str = ' + '.join([main_baseline] + remaining)
    formula = f'{main_outcome} ~ {ctrl_str} + treat_rot'

    result = run_regression(
        formula=formula,
        data=df_bus,
        cluster_var='barrio_num',
        spec_id=f'robust/control/drop_{ctrl}',
        spec_tree_path='robustness/leave_one_out.md',
        outcome_var=main_outcome,
        treatment_var='treat_rot',
        controls_desc=f'Full controls minus {ctrl}',
        sample_desc='Business owners only',
        model_type='OLS'
    )
    if result:
        results.append(result)
        print(f"Drop {ctrl}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# ============================================================================
# CONTROL VARIATIONS: Build-up / Progressive
# ============================================================================
print("\n" + "="*80)
print("CONTROL VARIATIONS: BUILD-UP")
print("="*80)

# No controls (just baseline outcome)
formula = f'{main_outcome} ~ {main_baseline} + treat_rot'
result = run_regression(
    formula=formula,
    data=df_bus,
    cluster_var='barrio_num',
    spec_id='robust/build/baseline_only',
    spec_tree_path='robustness/control_progression.md',
    outcome_var=main_outcome,
    treatment_var='treat_rot',
    controls_desc='Baseline outcome only',
    sample_desc='Business owners only',
    model_type='OLS'
)
if result:
    results.append(result)
    print(f"Baseline only: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# Bivariate (no baseline, no controls)
formula = f'{main_outcome} ~ treat_rot'
result = run_regression(
    formula=formula,
    data=df_bus,
    cluster_var='barrio_num',
    spec_id='robust/build/bivariate',
    spec_tree_path='robustness/control_progression.md',
    outcome_var=main_outcome,
    treatment_var='treat_rot',
    controls_desc='None',
    sample_desc='Business owners only',
    model_type='OLS'
)
if result:
    results.append(result)
    print(f"Bivariate: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# Add controls incrementally
controls_to_add = []
for i, ctrl in enumerate(full_controls):
    controls_to_add.append(ctrl)
    ctrl_str = ' + '.join([main_baseline] + controls_to_add)
    formula = f'{main_outcome} ~ {ctrl_str} + treat_rot'

    result = run_regression(
        formula=formula,
        data=df_bus,
        cluster_var='barrio_num',
        spec_id=f'robust/build/add_{ctrl}',
        spec_tree_path='robustness/control_progression.md',
        outcome_var=main_outcome,
        treatment_var='treat_rot',
        controls_desc=f'Baseline + {", ".join(controls_to_add)}',
        sample_desc='Business owners only',
        model_type='OLS'
    )
    if result:
        results.append(result)
        print(f"Add {ctrl}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# Full controls
ctrl_str = ' + '.join([main_baseline] + full_controls)
formula = f'{main_outcome} ~ {ctrl_str} + treat_rot'
result = run_regression(
    formula=formula,
    data=df_bus,
    cluster_var='barrio_num',
    spec_id='robust/build/full_controls',
    spec_tree_path='robustness/control_progression.md',
    outcome_var=main_outcome,
    treatment_var='treat_rot',
    controls_desc='All controls',
    sample_desc='Business owners only',
    model_type='OLS'
)
if result:
    results.append(result)
    print(f"Full controls: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# ============================================================================
# ALTERNATIVE OUTCOMES
# ============================================================================
print("\n" + "="*80)
print("ALTERNATIVE OUTCOMES")
print("="*80)

# For yes/no outcomes, use the _num versions we created
alt_outcomes = [
    ('e_cashSep_num', 'b_cashSep_num', 'Cash separation'),
    ('e_keepAcct_num', 'b_keepAcct_num', 'Keep accounts'),
    ('e_acctSep_num', 'b_acctSep_num', 'Account separation'),
    ('e_cashSetAside_num', 'b_cashSetAside_num', 'Cash set aside'),
    ('e_calcSales_num', 'b_calcSales_num', 'Calculate sales'),
    ('e_saveTotal_w01', 'b_saveTotal_w01', 'Total savings (winsorized)'),
    ('e_empTotal', 'b_empTotal', 'Total employees'),
    ('e_salesAvgWk_w01', 'b_salesAvgWk_w01', 'Avg weekly sales (winsorized)'),
    ('e_loanAmt', 'b_loanAmt', 'Loan amount'),
    ('e_save_num', 'b_save_num', 'Saves'),
]

for e_var, b_var, desc in alt_outcomes:
    if e_var not in df_bus.columns:
        continue

    df_bus[e_var] = pd.to_numeric(df_bus[e_var], errors='coerce')
    if b_var in df_bus.columns:
        df_bus[b_var] = pd.to_numeric(df_bus[b_var], errors='coerce')
        formula = f'{e_var} ~ {b_var} + treat_rot'
        ctrl_desc = f'Baseline {b_var}'
    else:
        formula = f'{e_var} ~ treat_rot'
        ctrl_desc = 'None'

    result = run_regression(
        formula=formula,
        data=df_bus,
        cluster_var='barrio_num',
        spec_id=f'robust/outcome/{e_var}',
        spec_tree_path='robustness/measurement.md',
        outcome_var=e_var,
        treatment_var='treat_rot',
        controls_desc=ctrl_desc,
        sample_desc='Business owners only',
        model_type='OLS'
    )
    if result:
        results.append(result)
        print(f"{desc}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")

# ============================================================================
# ALTERNATIVE TREATMENT DEFINITIONS
# ============================================================================
print("\n" + "="*80)
print("ALTERNATIVE TREATMENT DEFINITIONS")
print("="*80)

# Accounting treatment only
formula = f'{main_outcome} ~ {main_baseline} + treat_acc'
result = run_regression(
    formula=formula,
    data=df_bus,
    cluster_var='barrio_num',
    spec_id='robust/treatment/accounting_only',
    spec_tree_path='methods/cross_sectional_ols.md',
    outcome_var=main_outcome,
    treatment_var='treat_acc',
    controls_desc=f'Baseline {main_baseline}',
    sample_desc='Business owners only',
    model_type='OLS'
)
if result:
    results.append(result)
    print(f"Accounting treatment: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# Any treatment (pooled)
formula = f'{main_outcome} ~ {main_baseline} + treat'
result = run_regression(
    formula=formula,
    data=df_bus,
    cluster_var='barrio_num',
    spec_id='robust/treatment/any_treatment',
    spec_tree_path='methods/cross_sectional_ols.md',
    outcome_var=main_outcome,
    treatment_var='treat',
    controls_desc=f'Baseline {main_baseline}',
    sample_desc='Business owners only',
    model_type='OLS'
)
if result:
    results.append(result)
    print(f"Any treatment: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# Both treatments separately
formula = f'{main_outcome} ~ {main_baseline} + treat_acc + treat_rot'
model = smf.ols(formula, data=df_bus.dropna(subset=[main_outcome, main_baseline, 'treat_acc', 'treat_rot'])).fit(
    cov_type='cluster', cov_kwds={'groups': df_bus.dropna(subset=[main_outcome, main_baseline, 'treat_acc', 'treat_rot'])['barrio_num']})

# treat_acc coefficient
result = {
    'paper_id': PAPER_ID,
    'journal': JOURNAL,
    'paper_title': PAPER_TITLE,
    'spec_id': 'robust/treatment/both_arms_acc',
    'spec_tree_path': 'methods/cross_sectional_ols.md',
    'outcome_var': main_outcome,
    'treatment_var': 'treat_acc',
    'coefficient': float(model.params['treat_acc']),
    'std_error': float(model.bse['treat_acc']),
    't_stat': float(model.tvalues['treat_acc']),
    'p_value': float(model.pvalues['treat_acc']),
    'ci_lower': float(model.params['treat_acc'] - 1.96 * model.bse['treat_acc']),
    'ci_upper': float(model.params['treat_acc'] + 1.96 * model.bse['treat_acc']),
    'n_obs': int(model.nobs),
    'r_squared': float(model.rsquared),
    'coefficient_vector_json': json.dumps({}),
    'sample_desc': 'Business owners only',
    'fixed_effects': 'None',
    'controls_desc': f'Baseline {main_baseline}',
    'cluster_var': 'barrio_num',
    'model_type': 'OLS',
    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
}
results.append(result)
print(f"Both arms (treat_acc): coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# ============================================================================
# INFERENCE VARIATIONS
# ============================================================================
print("\n" + "="*80)
print("INFERENCE VARIATIONS")
print("="*80)

formula = f'{main_outcome} ~ {main_baseline} + treat_rot'

# Robust SEs (no clustering)
result = run_regression(
    formula=formula,
    data=df_bus,
    cluster_var=None,
    spec_id='robust/cluster/robust_hc1',
    spec_tree_path='robustness/clustering_variations.md',
    outcome_var=main_outcome,
    treatment_var='treat_rot',
    controls_desc=f'Baseline {main_baseline}',
    sample_desc='Business owners only',
    model_type='OLS'
)
if result:
    results.append(result)
    print(f"Robust HC1: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# Cluster by barrio (paper's main)
result = run_regression(
    formula=formula,
    data=df_bus,
    cluster_var='barrio_num',
    spec_id='robust/cluster/barrio',
    spec_tree_path='robustness/clustering_variations.md',
    outcome_var=main_outcome,
    treatment_var='treat_rot',
    controls_desc=f'Baseline {main_baseline}',
    sample_desc='Business owners only',
    model_type='OLS'
)
if result:
    results.append(result)
    print(f"Cluster barrio: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# HC2 robust SEs
model = smf.ols(formula, data=df_bus.dropna(subset=[main_outcome, main_baseline, 'treat_rot'])).fit(cov_type='HC2')
result = {
    'paper_id': PAPER_ID,
    'journal': JOURNAL,
    'paper_title': PAPER_TITLE,
    'spec_id': 'robust/cluster/robust_hc2',
    'spec_tree_path': 'robustness/clustering_variations.md',
    'outcome_var': main_outcome,
    'treatment_var': 'treat_rot',
    'coefficient': float(model.params['treat_rot']),
    'std_error': float(model.bse['treat_rot']),
    't_stat': float(model.tvalues['treat_rot']),
    'p_value': float(model.pvalues['treat_rot']),
    'ci_lower': float(model.params['treat_rot'] - 1.96 * model.bse['treat_rot']),
    'ci_upper': float(model.params['treat_rot'] + 1.96 * model.bse['treat_rot']),
    'n_obs': int(model.nobs),
    'r_squared': float(model.rsquared),
    'coefficient_vector_json': json.dumps({}),
    'sample_desc': 'Business owners only',
    'fixed_effects': 'None',
    'controls_desc': f'Baseline {main_baseline}',
    'cluster_var': 'None (HC2)',
    'model_type': 'OLS',
    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
}
results.append(result)
print(f"HC2: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# HC3 robust SEs
model = smf.ols(formula, data=df_bus.dropna(subset=[main_outcome, main_baseline, 'treat_rot'])).fit(cov_type='HC3')
result = {
    'paper_id': PAPER_ID,
    'journal': JOURNAL,
    'paper_title': PAPER_TITLE,
    'spec_id': 'robust/cluster/robust_hc3',
    'spec_tree_path': 'robustness/clustering_variations.md',
    'outcome_var': main_outcome,
    'treatment_var': 'treat_rot',
    'coefficient': float(model.params['treat_rot']),
    'std_error': float(model.bse['treat_rot']),
    't_stat': float(model.tvalues['treat_rot']),
    'p_value': float(model.pvalues['treat_rot']),
    'ci_lower': float(model.params['treat_rot'] - 1.96 * model.bse['treat_rot']),
    'ci_upper': float(model.params['treat_rot'] + 1.96 * model.bse['treat_rot']),
    'n_obs': int(model.nobs),
    'r_squared': float(model.rsquared),
    'coefficient_vector_json': json.dumps({}),
    'sample_desc': 'Business owners only',
    'fixed_effects': 'None',
    'controls_desc': f'Baseline {main_baseline}',
    'cluster_var': 'None (HC3)',
    'model_type': 'OLS',
    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
}
results.append(result)
print(f"HC3: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# ============================================================================
# SAMPLE RESTRICTIONS
# ============================================================================
print("\n" + "="*80)
print("SAMPLE RESTRICTIONS")
print("="*80)

# Full sample (not just business owners)
formula = f'{main_outcome} ~ {main_baseline} + treat_rot'
result = run_regression(
    formula=formula,
    data=df,
    cluster_var='barrio_num',
    spec_id='robust/sample/full_sample',
    spec_tree_path='robustness/sample_restrictions.md',
    outcome_var=main_outcome,
    treatment_var='treat_rot',
    controls_desc=f'Baseline {main_baseline}',
    sample_desc='Full sample (all respondents)',
    model_type='OLS'
)
if result:
    results.append(result)
    print(f"Full sample: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")

# By gender (mostly female sample)
# b_sex_num was already created above (1=female, 0=male)

# Female only
df_female = df_bus[df_bus['b_sex_num'] == 1]
result = run_regression(
    formula=formula,
    data=df_female,
    cluster_var='barrio_num',
    spec_id='robust/sample/female_only',
    spec_tree_path='robustness/sample_restrictions.md',
    outcome_var=main_outcome,
    treatment_var='treat_rot',
    controls_desc=f'Baseline {main_baseline}',
    sample_desc='Female business owners only',
    model_type='OLS'
)
if result:
    results.append(result)
    print(f"Female only: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")

# Male only
df_male = df_bus[df_bus['b_sex_num'] == 0]
result = run_regression(
    formula=formula,
    data=df_male,
    cluster_var='barrio_num',
    spec_id='robust/sample/male_only',
    spec_tree_path='robustness/sample_restrictions.md',
    outcome_var=main_outcome,
    treatment_var='treat_rot',
    controls_desc=f'Baseline {main_baseline}',
    sample_desc='Male business owners only',
    model_type='OLS'
)
if result:
    results.append(result)
    print(f"Male only: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")

# By age
df_bus['b_age'] = pd.to_numeric(df_bus['b_age'], errors='coerce')
median_age = df_bus['b_age'].median()

df_young = df_bus[df_bus['b_age'] <= median_age]
result = run_regression(
    formula=formula,
    data=df_young,
    cluster_var='barrio_num',
    spec_id='robust/sample/young_age',
    spec_tree_path='robustness/sample_restrictions.md',
    outcome_var=main_outcome,
    treatment_var='treat_rot',
    controls_desc=f'Baseline {main_baseline}',
    sample_desc=f'Age <= {median_age:.0f}',
    model_type='OLS'
)
if result:
    results.append(result)
    print(f"Young: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")

df_old = df_bus[df_bus['b_age'] > median_age]
result = run_regression(
    formula=formula,
    data=df_old,
    cluster_var='barrio_num',
    spec_id='robust/sample/old_age',
    spec_tree_path='robustness/sample_restrictions.md',
    outcome_var=main_outcome,
    treatment_var='treat_rot',
    controls_desc=f'Baseline {main_baseline}',
    sample_desc=f'Age > {median_age:.0f}',
    model_type='OLS'
)
if result:
    results.append(result)
    print(f"Old: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")

# By education
df_bus['educHi'] = pd.to_numeric(df_bus['educHi'], errors='coerce')

df_low_ed = df_bus[df_bus['educHi'] == 0]
result = run_regression(
    formula=formula,
    data=df_low_ed,
    cluster_var='barrio_num',
    spec_id='robust/sample/low_education',
    spec_tree_path='robustness/sample_restrictions.md',
    outcome_var=main_outcome,
    treatment_var='treat_rot',
    controls_desc=f'Baseline {main_baseline}',
    sample_desc='Low education',
    model_type='OLS'
)
if result:
    results.append(result)
    print(f"Low education: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")

df_high_ed = df_bus[df_bus['educHi'] == 1]
result = run_regression(
    formula=formula,
    data=df_high_ed,
    cluster_var='barrio_num',
    spec_id='robust/sample/high_education',
    spec_tree_path='robustness/sample_restrictions.md',
    outcome_var=main_outcome,
    treatment_var='treat_rot',
    controls_desc=f'Baseline {main_baseline}',
    sample_desc='High education',
    model_type='OLS'
)
if result:
    results.append(result)
    print(f"High education: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")

# Winsorize outcome at 1%
df_bus_wins = df_bus.copy()
q01 = df_bus_wins[main_outcome].quantile(0.01)
q99 = df_bus_wins[main_outcome].quantile(0.99)
df_bus_wins[f'{main_outcome}_wins'] = df_bus_wins[main_outcome].clip(lower=q01, upper=q99)

formula_wins = f'{main_outcome}_wins ~ {main_baseline} + treat_rot'
result = run_regression(
    formula=formula_wins,
    data=df_bus_wins,
    cluster_var='barrio_num',
    spec_id='robust/sample/winsorize_1pct',
    spec_tree_path='robustness/sample_restrictions.md',
    outcome_var=f'{main_outcome}_wins',
    treatment_var='treat_rot',
    controls_desc=f'Baseline {main_baseline}',
    sample_desc='Outcome winsorized at 1%/99%',
    model_type='OLS'
)
if result:
    results.append(result)
    print(f"Winsorized 1%: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# Trim outcome at 1%
df_bus_trim = df_bus[(df_bus[main_outcome] >= q01) & (df_bus[main_outcome] <= q99)]
result = run_regression(
    formula=formula,
    data=df_bus_trim,
    cluster_var='barrio_num',
    spec_id='robust/sample/trim_1pct',
    spec_tree_path='robustness/sample_restrictions.md',
    outcome_var=main_outcome,
    treatment_var='treat_rot',
    controls_desc=f'Baseline {main_baseline}',
    sample_desc='Trim top/bottom 1%',
    model_type='OLS'
)
if result:
    results.append(result)
    print(f"Trim 1%: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")

# ============================================================================
# HETEROGENEITY ANALYSIS (Table 4 style)
# ============================================================================
print("\n" + "="*80)
print("HETEROGENEITY ANALYSIS")
print("="*80)

# By skill level
df_bus['skill'] = pd.to_numeric(df_bus['skill'], errors='coerce')

df_low_skill = df_bus[df_bus['skill'] == 0]
result = run_regression(
    formula=formula,
    data=df_low_skill,
    cluster_var='barrio_num',
    spec_id='robust/het/low_skill',
    spec_tree_path='robustness/heterogeneity.md',
    outcome_var=main_outcome,
    treatment_var='treat_rot',
    controls_desc=f'Baseline {main_baseline}',
    sample_desc='Low skill',
    model_type='OLS'
)
if result:
    results.append(result)
    print(f"Low skill: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")

df_high_skill = df_bus[df_bus['skill'] == 1]
result = run_regression(
    formula=formula,
    data=df_high_skill,
    cluster_var='barrio_num',
    spec_id='robust/het/high_skill',
    spec_tree_path='robustness/heterogeneity.md',
    outcome_var=main_outcome,
    treatment_var='treat_rot',
    controls_desc=f'Baseline {main_baseline}',
    sample_desc='High skill',
    model_type='OLS'
)
if result:
    results.append(result)
    print(f"High skill: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")

# By interest in having an account
df_bus['b_wantAcct'] = pd.to_numeric(df_bus['b_wantAcct'], errors='coerce')

df_no_want = df_bus[df_bus['b_wantAcct'] == 0]
result = run_regression(
    formula=formula,
    data=df_no_want,
    cluster_var='barrio_num',
    spec_id='robust/het/no_want_acct',
    spec_tree_path='robustness/heterogeneity.md',
    outcome_var=main_outcome,
    treatment_var='treat_rot',
    controls_desc=f'Baseline {main_baseline}',
    sample_desc='No interest in account',
    model_type='OLS'
)
if result:
    results.append(result)
    print(f"No want account: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")

df_want = df_bus[df_bus['b_wantAcct'] == 1]
result = run_regression(
    formula=formula,
    data=df_want,
    cluster_var='barrio_num',
    spec_id='robust/het/want_acct',
    spec_tree_path='robustness/heterogeneity.md',
    outcome_var=main_outcome,
    treatment_var='treat_rot',
    controls_desc=f'Baseline {main_baseline}',
    sample_desc='Interest in account',
    model_type='OLS'
)
if result:
    results.append(result)
    print(f"Want account: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")

# By baseline business practice quartiles
for q in [1, 2, 3, 4]:
    q_var = f'bBusPracQ{q}'
    df_bus[q_var] = pd.to_numeric(df_bus[q_var], errors='coerce')
    df_q = df_bus[df_bus[q_var] == 1]

    result = run_regression(
        formula=formula,
        data=df_q,
        cluster_var='barrio_num',
        spec_id=f'robust/het/baseline_busprac_q{q}',
        spec_tree_path='robustness/heterogeneity.md',
        outcome_var=main_outcome,
        treatment_var='treat_rot',
        controls_desc=f'Baseline {main_baseline}',
        sample_desc=f'Baseline business practice quartile {q}',
        model_type='OLS'
    )
    if result:
        results.append(result)
        print(f"BusPrac Q{q}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")

# Interaction with skill
df_bus['treat_rot_skill'] = df_bus['treat_rot'] * df_bus['skill']
formula_int = f'{main_outcome} ~ {main_baseline} + treat_rot + skill + treat_rot_skill'
result = run_regression(
    formula=formula_int,
    data=df_bus,
    cluster_var='barrio_num',
    spec_id='robust/het/interaction_skill',
    spec_tree_path='robustness/heterogeneity.md',
    outcome_var=main_outcome,
    treatment_var='treat_rot',
    controls_desc=f'Baseline + skill interaction',
    sample_desc='Business owners',
    model_type='OLS'
)
if result:
    results.append(result)
    print(f"Skill interaction: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# Interaction with baseline business practices
df_bus['treat_rot_busprac'] = df_bus['treat_rot'] * df_bus[main_baseline]
formula_int2 = f'{main_outcome} ~ {main_baseline} + treat_rot + treat_rot_busprac'
result = run_regression(
    formula=formula_int2,
    data=df_bus,
    cluster_var='barrio_num',
    spec_id='robust/het/interaction_baseline_busprac',
    spec_tree_path='robustness/heterogeneity.md',
    outcome_var=main_outcome,
    treatment_var='treat_rot',
    controls_desc=f'Baseline + baseline interaction',
    sample_desc='Business owners',
    model_type='OLS'
)
if result:
    results.append(result)
    print(f"Baseline BusPrac interaction: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# ============================================================================
# FUNCTIONAL FORM VARIATIONS
# ============================================================================
print("\n" + "="*80)
print("FUNCTIONAL FORM VARIATIONS")
print("="*80)

# For sales outcome (continuous)
sales_outcome = 'e_salesWkBad_w01'
sales_baseline = 'b_salesWkBad_w01'

df_bus[sales_outcome] = pd.to_numeric(df_bus[sales_outcome], errors='coerce')
df_bus[sales_baseline] = pd.to_numeric(df_bus[sales_baseline], errors='coerce')

# Log sales (add 1 to handle zeros)
df_bus['log_sales'] = np.log(df_bus[sales_outcome] + 1)
df_bus['log_sales_base'] = np.log(df_bus[sales_baseline] + 1)

formula_log = 'log_sales ~ log_sales_base + treat_rot'
result = run_regression(
    formula=formula_log,
    data=df_bus,
    cluster_var='barrio_num',
    spec_id='robust/funcform/log_sales',
    spec_tree_path='robustness/functional_form.md',
    outcome_var='log_sales',
    treatment_var='treat_rot',
    controls_desc='Baseline log sales',
    sample_desc='Business owners',
    model_type='OLS'
)
if result:
    results.append(result)
    print(f"Log sales: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# IHS transformation
df_bus['ihs_sales'] = np.arcsinh(df_bus[sales_outcome])
df_bus['ihs_sales_base'] = np.arcsinh(df_bus[sales_baseline])

formula_ihs = 'ihs_sales ~ ihs_sales_base + treat_rot'
result = run_regression(
    formula=formula_ihs,
    data=df_bus,
    cluster_var='barrio_num',
    spec_id='robust/funcform/ihs_sales',
    spec_tree_path='robustness/functional_form.md',
    outcome_var='ihs_sales',
    treatment_var='treat_rot',
    controls_desc='Baseline IHS sales',
    sample_desc='Business owners',
    model_type='OLS'
)
if result:
    results.append(result)
    print(f"IHS sales: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# Levels
formula_levels = f'{sales_outcome} ~ {sales_baseline} + treat_rot'
result = run_regression(
    formula=formula_levels,
    data=df_bus,
    cluster_var='barrio_num',
    spec_id='robust/funcform/levels_sales',
    spec_tree_path='robustness/functional_form.md',
    outcome_var=sales_outcome,
    treatment_var='treat_rot',
    controls_desc=f'Baseline {sales_baseline}',
    sample_desc='Business owners',
    model_type='OLS'
)
if result:
    results.append(result)
    print(f"Levels sales: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# Quadratic in baseline
df_bus['baseline_sq'] = df_bus[main_baseline] ** 2
formula_quad = f'{main_outcome} ~ {main_baseline} + baseline_sq + treat_rot'
result = run_regression(
    formula=formula_quad,
    data=df_bus,
    cluster_var='barrio_num',
    spec_id='robust/funcform/quadratic_baseline',
    spec_tree_path='robustness/functional_form.md',
    outcome_var=main_outcome,
    treatment_var='treat_rot',
    controls_desc='Baseline + baseline squared',
    sample_desc='Business owners',
    model_type='OLS'
)
if result:
    results.append(result)
    print(f"Quadratic baseline: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# ============================================================================
# PLACEBO / FALSIFICATION TESTS
# ============================================================================
print("\n" + "="*80)
print("PLACEBO TESTS")
print("="*80)

# Placebo: Use baseline outcome as "outcome" (should be no effect since treatment
# was assigned after baseline)
formula_placebo = f'{main_baseline} ~ treat_rot'
result = run_regression(
    formula=formula_placebo,
    data=df_bus,
    cluster_var='barrio_num',
    spec_id='robust/placebo/baseline_as_outcome',
    spec_tree_path='robustness/placebo_tests.md',
    outcome_var=main_baseline,
    treatment_var='treat_rot',
    controls_desc='None',
    sample_desc='Business owners (placebo)',
    model_type='OLS'
)
if result:
    results.append(result)
    print(f"Placebo (baseline as outcome): coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# Placebo: Demographics as outcome (shouldn't be affected)
demo_vars = ['b_age', 'b_numKids']
for demo in demo_vars:
    df_bus[demo] = pd.to_numeric(df_bus[demo], errors='coerce')
    formula_demo = f'{demo} ~ treat_rot'
    result = run_regression(
        formula=formula_demo,
        data=df_bus,
        cluster_var='barrio_num',
        spec_id=f'robust/placebo/{demo}',
        spec_tree_path='robustness/placebo_tests.md',
        outcome_var=demo,
        treatment_var='treat_rot',
        controls_desc='None',
        sample_desc='Business owners (placebo)',
        model_type='OLS'
    )
    if result:
        results.append(result)
        print(f"Placebo ({demo}): coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# ============================================================================
# ADDITIONAL OUTCOMES FOR MULTIPLE OUTCOMES
# ============================================================================
print("\n" + "="*80)
print("ADDITIONAL OUTCOMES")
print("="*80)

# Sales outcomes
additional_outcomes = [
    ('e_zSales', 'b_zSales', 'Standardized sales'),
    ('e_salesAvgWk_w01', 'b_salesAvgWk_w01', 'Avg weekly sales'),
    ('e_salesWkGood_w01', 'b_salesWkGood_w01', 'Sales good week'),
    ('e_repAnyMistake', 'b_repAnyMistake', 'Any reporting mistake'),
]

for e_var, b_var, desc in additional_outcomes:
    if e_var not in df_bus.columns:
        continue

    df_bus[e_var] = pd.to_numeric(df_bus[e_var], errors='coerce')
    if b_var in df_bus.columns:
        df_bus[b_var] = pd.to_numeric(df_bus[b_var], errors='coerce')
        formula = f'{e_var} ~ {b_var} + treat_rot'
        ctrl = f'Baseline {b_var}'
    else:
        formula = f'{e_var} ~ treat_rot'
        ctrl = 'None'

    result = run_regression(
        formula=formula,
        data=df_bus,
        cluster_var='barrio_num',
        spec_id=f'outcome/{e_var}',
        spec_tree_path='methods/cross_sectional_ols.md',
        outcome_var=e_var,
        treatment_var='treat_rot',
        controls_desc=ctrl,
        sample_desc='Business owners',
        model_type='OLS'
    )
    if result:
        results.append(result)
        print(f"{desc}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# ============================================================================
# ESTIMATION METHOD VARIATIONS
# ============================================================================
print("\n" + "="*80)
print("ESTIMATION METHOD VARIATIONS")
print("="*80)

# Quantile regression (median)
try:
    from statsmodels.regression.quantile_regression import QuantReg

    df_reg = df_bus.dropna(subset=[main_outcome, main_baseline, 'treat_rot']).copy()
    X = sm.add_constant(df_reg[[main_baseline, 'treat_rot']])
    y = df_reg[main_outcome]

    qr_model = QuantReg(y, X).fit(q=0.5)

    result = {
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': 'robust/method/quantile_median',
        'spec_tree_path': 'methods/cross_sectional_ols.md',
        'outcome_var': main_outcome,
        'treatment_var': 'treat_rot',
        'coefficient': float(qr_model.params['treat_rot']),
        'std_error': float(qr_model.bse['treat_rot']),
        't_stat': float(qr_model.tvalues['treat_rot']),
        'p_value': float(qr_model.pvalues['treat_rot']),
        'ci_lower': float(qr_model.params['treat_rot'] - 1.96 * qr_model.bse['treat_rot']),
        'ci_upper': float(qr_model.params['treat_rot'] + 1.96 * qr_model.bse['treat_rot']),
        'n_obs': int(len(df_reg)),
        'r_squared': float(qr_model.prsquared),
        'coefficient_vector_json': json.dumps({}),
        'sample_desc': 'Business owners',
        'fixed_effects': 'None',
        'controls_desc': f'Baseline {main_baseline}',
        'cluster_var': 'None',
        'model_type': 'Quantile (median)',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    }
    results.append(result)
    print(f"Quantile median: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # 25th percentile
    qr_model_25 = QuantReg(y, X).fit(q=0.25)
    result = {
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': 'robust/method/quantile_25',
        'spec_tree_path': 'methods/cross_sectional_ols.md',
        'outcome_var': main_outcome,
        'treatment_var': 'treat_rot',
        'coefficient': float(qr_model_25.params['treat_rot']),
        'std_error': float(qr_model_25.bse['treat_rot']),
        't_stat': float(qr_model_25.tvalues['treat_rot']),
        'p_value': float(qr_model_25.pvalues['treat_rot']),
        'ci_lower': float(qr_model_25.params['treat_rot'] - 1.96 * qr_model_25.bse['treat_rot']),
        'ci_upper': float(qr_model_25.params['treat_rot'] + 1.96 * qr_model_25.bse['treat_rot']),
        'n_obs': int(len(df_reg)),
        'r_squared': float(qr_model_25.prsquared),
        'coefficient_vector_json': json.dumps({}),
        'sample_desc': 'Business owners',
        'fixed_effects': 'None',
        'controls_desc': f'Baseline {main_baseline}',
        'cluster_var': 'None',
        'model_type': 'Quantile (25th)',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    }
    results.append(result)
    print(f"Quantile 25th: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # 75th percentile
    qr_model_75 = QuantReg(y, X).fit(q=0.75)
    result = {
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': 'robust/method/quantile_75',
        'spec_tree_path': 'methods/cross_sectional_ols.md',
        'outcome_var': main_outcome,
        'treatment_var': 'treat_rot',
        'coefficient': float(qr_model_75.params['treat_rot']),
        'std_error': float(qr_model_75.bse['treat_rot']),
        't_stat': float(qr_model_75.tvalues['treat_rot']),
        'p_value': float(qr_model_75.pvalues['treat_rot']),
        'ci_lower': float(qr_model_75.params['treat_rot'] - 1.96 * qr_model_75.bse['treat_rot']),
        'ci_upper': float(qr_model_75.params['treat_rot'] + 1.96 * qr_model_75.bse['treat_rot']),
        'n_obs': int(len(df_reg)),
        'r_squared': float(qr_model_75.prsquared),
        'coefficient_vector_json': json.dumps({}),
        'sample_desc': 'Business owners',
        'fixed_effects': 'None',
        'controls_desc': f'Baseline {main_baseline}',
        'cluster_var': 'None',
        'model_type': 'Quantile (75th)',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    }
    results.append(result)
    print(f"Quantile 75th: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

except Exception as e:
    print(f"Quantile regression error: {e}")

# ============================================================================
# MORE SAMPLE RESTRICTIONS - DROP BY BARRIO
# ============================================================================
print("\n" + "="*80)
print("JACKKNIFE BY NEIGHBORHOOD (BARRIO)")
print("="*80)

# Reset formula for main outcome
main_outcome = 'e_zBusPrac'
main_baseline = 'b_zBusPrac'
formula = f'{main_outcome} ~ {main_baseline} + treat_rot'

# Get top 5 largest barrios and drop each one
# Use barrio_str which was created before conversion
barrio_counts = df_bus.groupby('barrio_str').size().sort_values(ascending=False)
top_barrios = barrio_counts.head(5).index.tolist()
print(f"Top barrios to jackknife: {top_barrios}")

for barrio in top_barrios:
    # Safe barrio name for spec_id (remove spaces and special chars)
    safe_barrio = str(barrio).replace(' ', '_').replace('.', '')[:20]
    df_drop = df_bus[df_bus['barrio_str'] != barrio].copy()
    n_dropped = len(df_bus) - len(df_drop)
    print(f"  Testing drop of {barrio}: {n_dropped} obs dropped, {len(df_drop)} remaining")
    if len(df_drop) < 50:
        print(f"  Skipping - too few observations")
        continue
    result = run_regression(
        formula=formula,
        data=df_drop,
        cluster_var='barrio_num',
        spec_id=f'robust/sample/drop_barrio_{safe_barrio}',
        spec_tree_path='robustness/sample_restrictions.md',
        outcome_var=main_outcome,
        treatment_var='treat_rot',
        controls_desc=f'Baseline {main_baseline}',
        sample_desc=f'Drop barrio {barrio}',
        model_type='OLS'
    )
    if result:
        results.append(result)
        print(f"Drop barrio {safe_barrio}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")

# ============================================================================
# ADDITIONAL HETEROGENEITY - BY SAVINGS
# ============================================================================
print("\n" + "="*80)
print("HETEROGENEITY BY SAVINGS")
print("="*80)

df_bus['savings'] = pd.to_numeric(df_bus['savings'], errors='coerce')

df_no_save = df_bus[df_bus['savings'] == 0]
result = run_regression(
    formula=formula,
    data=df_no_save,
    cluster_var='barrio_num',
    spec_id='robust/het/no_savings',
    spec_tree_path='robustness/heterogeneity.md',
    outcome_var=main_outcome,
    treatment_var='treat_rot',
    controls_desc=f'Baseline {main_baseline}',
    sample_desc='No savings at baseline',
    model_type='OLS'
)
if result:
    results.append(result)
    print(f"No savings: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")

df_save = df_bus[df_bus['savings'] == 1]
result = run_regression(
    formula=formula,
    data=df_save,
    cluster_var='barrio_num',
    spec_id='robust/het/has_savings',
    spec_tree_path='robustness/heterogeneity.md',
    outcome_var=main_outcome,
    treatment_var='treat_rot',
    controls_desc=f'Baseline {main_baseline}',
    sample_desc='Has savings at baseline',
    model_type='OLS'
)
if result:
    results.append(result)
    print(f"Has savings: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")

# ============================================================================
# ADDITIONAL HETEROGENEITY - BY LOAN AMOUNT
# ============================================================================
print("\n" + "="*80)
print("HETEROGENEITY BY LOAN AMOUNT")
print("="*80)

df_bus['monto_dese'] = pd.to_numeric(df_bus['monto_dese'], errors='coerce')
median_loan = df_bus['monto_dese'].median()

df_low_loan = df_bus[df_bus['monto_dese'] <= median_loan]
result = run_regression(
    formula=formula,
    data=df_low_loan,
    cluster_var='barrio_num',
    spec_id='robust/het/low_loan_amount',
    spec_tree_path='robustness/heterogeneity.md',
    outcome_var=main_outcome,
    treatment_var='treat_rot',
    controls_desc=f'Baseline {main_baseline}',
    sample_desc=f'Loan <= {median_loan:.0f}',
    model_type='OLS'
)
if result:
    results.append(result)
    print(f"Low loan: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")

df_high_loan = df_bus[df_bus['monto_dese'] > median_loan]
result = run_regression(
    formula=formula,
    data=df_high_loan,
    cluster_var='barrio_num',
    spec_id='robust/het/high_loan_amount',
    spec_tree_path='robustness/heterogeneity.md',
    outcome_var=main_outcome,
    treatment_var='treat_rot',
    controls_desc=f'Baseline {main_baseline}',
    sample_desc=f'Loan > {median_loan:.0f}',
    model_type='OLS'
)
if result:
    results.append(result)
    print(f"High loan: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Convert to DataFrame
results_df = pd.DataFrame(results)
print(f"Total specifications: {len(results_df)}")

# Save to CSV
results_df.to_csv(OUTPUT_PATH, index=False)
print(f"Results saved to: {OUTPUT_PATH}")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"Total specifications: {len(results_df)}")
print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")
print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")
print(f"Coefficient range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")

# By category
print("\n" + "-"*40)
print("BY SPECIFICATION CATEGORY")
print("-"*40)

def get_category(spec_id):
    if spec_id.startswith('baseline'):
        return 'Baseline'
    elif 'control' in spec_id or 'build' in spec_id:
        return 'Control variations'
    elif 'sample' in spec_id:
        return 'Sample restrictions'
    elif 'outcome' in spec_id:
        return 'Alternative outcomes'
    elif 'treatment' in spec_id:
        return 'Alternative treatments'
    elif 'cluster' in spec_id or 'se/' in spec_id:
        return 'Inference variations'
    elif 'method' in spec_id:
        return 'Estimation method'
    elif 'funcform' in spec_id:
        return 'Functional form'
    elif 'het' in spec_id:
        return 'Heterogeneity'
    elif 'placebo' in spec_id:
        return 'Placebo tests'
    else:
        return 'Other'

results_df['category'] = results_df['spec_id'].apply(get_category)

category_summary = results_df.groupby('category').agg({
    'spec_id': 'count',
    'coefficient': ['mean', lambda x: (x > 0).mean()],
    'p_value': [lambda x: (x < 0.05).mean(), lambda x: (x < 0.01).mean()]
}).round(3)

category_summary.columns = ['N', 'Mean Coef', '% Positive', '% Sig 5%', '% Sig 1%']
print(category_summary.to_string())

print("\nDone!")
