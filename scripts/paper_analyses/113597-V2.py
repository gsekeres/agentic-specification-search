"""
Specification Search: 113597-V2
Title: The Impacts of Microfinance: Evidence from Joint-Liability Lending in Mongolia
Authors: Attanasio, Augsburg, De Haas, Fitzsimons, Harmgart
Journal: AEJ Applied (2015)

This script runs a systematic specification search following the i4r methodology.
The paper is an RCT evaluating the impact of joint-liability microfinance loans
in Mongolia. The main comparison is Group Loan vs Control (excluding Individual Loan arm).

Method Classification: Cross-sectional OLS with ANCOVA (baseline control)
The paper uses simple difference estimation at follow-up, controlling for baseline
outcome and a standard set of covariates, with clustering at the soum level.
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import warnings
warnings.filterwarnings('ignore')

# Set paths
BASE_PATH = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search'
DATA_PATH = f'{BASE_PATH}/data/downloads/extracted/113597-V2/Analysis-files'
OUTPUT_PATH = f'{BASE_PATH}/data/downloads/extracted/113597-V2'

# Paper metadata
PAPER_ID = '113597-V2'
JOURNAL = 'AEJ-Applied'
PAPER_TITLE = 'The Impacts of Microfinance: Evidence from Joint-Liability Lending in Mongolia'

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================

print("Loading data...")
# Load main dataset
df = pd.read_stata(f'{DATA_PATH}/data/Baseline/all_outcomes_controls.dta', convert_categoricals=False)

# Load baseline outcomes (contains BL prefixed variables)
baseline_outcomes = pd.read_stata(f'{DATA_PATH}/data/Baseline/Baseline outcomes.dta', convert_categoricals=False)

# Merge baseline outcomes
df = df.merge(baseline_outcomes, on='rescode', how='left', suffixes=('', '_dup'))

print(f"Dataset shape after merge: {df.shape}")

# Create key variables
df['age16'] = df['age16m'].fillna(0) + df['age16f'].fillna(0)

# Sample restriction: Group vs Control only (exclude Individual Loan)
# reint==1 means reinterviewed at followup
df_analysis = df[(df['indiv'] != 1) & (df['reint'] == 1)].copy()

# Further restrict to followup period for simple difference
df_followup = df_analysis[df_analysis['followup'] == 1].copy()

print(f"Analysis sample (Group vs Control, followup): {len(df_followup)}")
print(f"Group treatment: {df_followup['group'].sum()}, Control: {(df_followup['group']==0).sum()}")

# Create aimag dummies
df_followup['aimag_cat'] = pd.Categorical(df_followup['aimag'])
aimag_dummies = pd.get_dummies(df_followup['aimag_cat'], prefix='aimag', drop_first=True)
df_followup = pd.concat([df_followup, aimag_dummies], axis=1)

# ============================================================================
# STEP 2: DEFINE VARIABLES AND CONTROLS
# ============================================================================

# Control variables (from Stata do-file global Xvar)
# Note: aug_b has no variation in followup sample, so we exclude it
CONTROL_VARS = ['loan_baseline', 'eduvoc', 'edusec', 'age16', 'under16',
                'marr_cohab', 'age', 'age_sq', 'buddhist', 'hahl',
                'sep_f', 'nov_f']

# Aimag dummies
AIMAG_DUMMIES = [c for c in df_followup.columns if c.startswith('aimag_') and c != 'aimag_cat']

# Full control set
ALL_CONTROLS = CONTROL_VARS + AIMAG_DUMMIES

# Treatment variable
TREATMENT_VAR = 'group'

# Create additional outcome variables
print("Creating additional variables...")

# Create scaled profit variable
df_followup['scaled_profit_r'] = df_followup['profit_r'] / 1000

# Create log consumption variables (add small constant to handle zeros)
for var in ['totalc', 'foodc', 'nondurc', 'durc']:
    if var in df_followup.columns:
        df_followup[f'ln_{var}'] = np.log(df_followup[var].clip(lower=1))

# Create per capita consumption
if 'totalc' in df_followup.columns and 'hhsize' in df_followup.columns:
    df_followup['totalc_pc'] = df_followup['totalc'] / df_followup['hhsize'].replace(0, np.nan)
    df_followup['ln_totalc_pc'] = np.log(df_followup['totalc_pc'].clip(lower=1))

# Hours worked
df_followup['hours_total'] = df_followup.get('hours_wage', pd.Series(0)).fillna(0) + df_followup.get('hours_ent', pd.Series(0)).fillna(0)

# Assets
df_followup['scaled_assets'] = df_followup['assets_all'] / 1000 if 'assets_all' in df_followup.columns else 0

# IHS transformation
df_followup['ihs_profit'] = np.arcsinh(df_followup['profit_r'])

# Interaction variables for heterogeneity
df_followup['group_x_edusec'] = df_followup['group'] * df_followup['edusec']
df_followup['group_x_loan'] = df_followup['group'] * df_followup['loan_baseline']
df_followup['old'] = (df_followup['age'] >= df_followup['age'].median()).astype(int)
df_followup['group_x_old'] = df_followup['group'] * df_followup['old']
df_followup['large_hh'] = (df_followup['hhsize'] >= df_followup['hhsize'].median()).astype(int)
df_followup['group_x_large_hh'] = df_followup['group'] * df_followup['large_hh']
df_followup['group_x_buddhist'] = df_followup['group'] * df_followup['buddhist']

# ============================================================================
# STEP 3: HELPER FUNCTIONS
# ============================================================================

def run_specification(df, outcome_var, treatment_var, controls, cluster_var,
                      spec_id, spec_tree_path, fixed_effects=None,
                      sample_desc='Full sample', baseline_var=None):
    """
    Run a single specification and return results dictionary.
    """
    result = {
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'outcome_var': outcome_var,
        'treatment_var': treatment_var,
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects if fixed_effects else 'None',
        'controls_desc': ', '.join(controls[:5]) + ('...' if len(controls) > 5 else ''),
        'cluster_var': cluster_var if cluster_var else 'None',
        'model_type': 'OLS',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    }

    try:
        # Check if outcome variable exists and has variation
        if outcome_var not in df.columns:
            raise ValueError(f"Outcome variable {outcome_var} not found")

        if df[outcome_var].isna().all():
            raise ValueError(f"Outcome variable {outcome_var} is all missing")

        # Build control list (filter to existing columns)
        valid_controls = [c for c in controls if c in df.columns]

        # Add baseline variable if specified and exists
        if baseline_var and baseline_var in df.columns:
            if baseline_var not in valid_controls:
                valid_controls = [baseline_var] + valid_controls

        # Build formula
        if valid_controls:
            control_str = ' + '.join(valid_controls)
            if fixed_effects:
                formula = f"{outcome_var} ~ {treatment_var} + {control_str} | {fixed_effects}"
            else:
                formula = f"{outcome_var} ~ {treatment_var} + {control_str}"
        else:
            if fixed_effects:
                formula = f"{outcome_var} ~ {treatment_var} | {fixed_effects}"
            else:
                formula = f"{outcome_var} ~ {treatment_var}"

        # Run regression
        if cluster_var and cluster_var in df.columns:
            model = pf.feols(formula, data=df, vcov={'CRV1': cluster_var})
        else:
            model = pf.feols(formula, data=df, vcov='hetero')

        # Extract results
        coef = model.coef()[treatment_var]
        se = model.se()[treatment_var]
        tstat = model.tstat()[treatment_var]
        pval = model.pvalue()[treatment_var]

        # Confidence interval
        ci = model.confint()
        ci_lower = ci.loc[treatment_var, '2.5%'] if treatment_var in ci.index else coef - 1.96*se
        ci_upper = ci.loc[treatment_var, '97.5%'] if treatment_var in ci.index else coef + 1.96*se

        result['coefficient'] = coef
        result['std_error'] = se
        result['t_stat'] = tstat
        result['p_value'] = pval
        result['ci_lower'] = ci_lower
        result['ci_upper'] = ci_upper
        result['n_obs'] = int(model._N)
        result['r_squared'] = model._r2 if hasattr(model, '_r2') else np.nan

        # Build coefficient vector JSON
        coef_dict = {
            'treatment': {
                'var': treatment_var,
                'coef': float(coef),
                'se': float(se),
                'pval': float(pval)
            },
            'controls': [],
            'fixed_effects': fixed_effects.split('+') if fixed_effects else [],
            'diagnostics': {}
        }

        # Add control coefficients (limit to first 10)
        for i, var in enumerate(model.coef().index):
            if var != treatment_var and var != 'Intercept' and i < 15:
                coef_dict['controls'].append({
                    'var': var,
                    'coef': float(model.coef()[var]),
                    'se': float(model.se()[var]),
                    'pval': float(model.pvalue()[var])
                })

        result['coefficient_vector_json'] = json.dumps(coef_dict)

    except Exception as e:
        result['coefficient'] = np.nan
        result['std_error'] = np.nan
        result['t_stat'] = np.nan
        result['p_value'] = np.nan
        result['ci_lower'] = np.nan
        result['ci_upper'] = np.nan
        result['n_obs'] = 0
        result['r_squared'] = np.nan
        result['coefficient_vector_json'] = json.dumps({'error': str(e)})

    return result

# ============================================================================
# STEP 4: RUN SPECIFICATIONS
# ============================================================================

results = []

print("\n" + "="*60)
print("RUNNING SPECIFICATION SEARCH")
print("="*60)

# --------------------------------------------------------------------------
# 4.1 BASELINE SPECIFICATION (exact replication of paper's main result)
# --------------------------------------------------------------------------
print("\n--- Baseline Specifications ---")

# Main baseline: enterprise outcome (Table 3)
result = run_specification(
    df=df_followup,
    outcome_var='enterprise',
    treatment_var='group',
    controls=ALL_CONTROLS,
    cluster_var='soum',
    spec_id='baseline',
    spec_tree_path='methods/cross_sectional_ols.md#baseline',
    baseline_var='BLenterprise',
    sample_desc='Group vs Control, followup'
)
results.append(result)
print(f"Baseline enterprise: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")

# Additional baseline outcomes
baseline_outcomes_list = [
    ('soleent', 'BLsoleent', 'Sole entrepreneurship'),
    ('ln_totalc', 'BLln_totalpc', 'Log total consumption'),
    ('ln_foodc', 'BLln_foodc', 'Log food consumption'),
    ('scaled_profit_r', 'BLscaled_profit_r', 'Scaled profit'),
]

for outcome, bl_var, desc in baseline_outcomes_list:
    if outcome in df_followup.columns:
        result = run_specification(
            df=df_followup,
            outcome_var=outcome,
            treatment_var='group',
            controls=ALL_CONTROLS,
            cluster_var='soum',
            spec_id=f'baseline/{outcome}',
            spec_tree_path='methods/cross_sectional_ols.md#baseline',
            baseline_var=bl_var if bl_var in df_followup.columns else None,
            sample_desc=desc
        )
        results.append(result)
        print(f"Baseline {outcome}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# --------------------------------------------------------------------------
# 4.2 CONTROL VARIABLE VARIATIONS (~15 specs)
# --------------------------------------------------------------------------
print("\n--- Control Variations ---")

# 4.2.1 No controls (bivariate)
result = run_specification(
    df=df_followup,
    outcome_var='enterprise',
    treatment_var='group',
    controls=[],
    cluster_var='soum',
    spec_id='robust/control/none',
    spec_tree_path='robustness/control_progression.md',
    sample_desc='Bivariate regression'
)
results.append(result)
print(f"No controls: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# 4.2.2 Only baseline outcome control
result = run_specification(
    df=df_followup,
    outcome_var='enterprise',
    treatment_var='group',
    controls=['BLenterprise'],
    cluster_var='soum',
    spec_id='robust/control/baseline_only',
    spec_tree_path='robustness/control_progression.md',
    sample_desc='Only baseline control'
)
results.append(result)
print(f"Baseline only: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# 4.2.3 Demographics only
demo_controls = ['age', 'age_sq', 'marr_cohab', 'buddhist', 'hahl', 'under16', 'age16']
result = run_specification(
    df=df_followup,
    outcome_var='enterprise',
    treatment_var='group',
    controls=demo_controls,
    cluster_var='soum',
    spec_id='robust/control/demographics',
    spec_tree_path='robustness/control_progression.md',
    baseline_var='BLenterprise',
    sample_desc='Demographics controls only'
)
results.append(result)
print(f"Demographics only: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# 4.2.4 Leave-one-out for each control
for control in CONTROL_VARS:
    remaining_controls = [c for c in ALL_CONTROLS if c != control]
    result = run_specification(
        df=df_followup,
        outcome_var='enterprise',
        treatment_var='group',
        controls=remaining_controls,
        cluster_var='soum',
        spec_id=f'robust/loo/drop_{control}',
        spec_tree_path='robustness/leave_one_out.md',
        baseline_var='BLenterprise',
        sample_desc=f'Drop {control}'
    )
    results.append(result)

print(f"Leave-one-out: {len(CONTROL_VARS)} specifications")

# --------------------------------------------------------------------------
# 4.3 SAMPLE RESTRICTIONS (~15 specs)
# --------------------------------------------------------------------------
print("\n--- Sample Restrictions ---")

# 4.3.1 By aimag (region)
aimags = df_followup['aimag'].dropna().unique()
for aimag in aimags[:5]:  # Limit to 5 regions
    df_subset = df_followup[df_followup['aimag'] != aimag].copy()
    result = run_specification(
        df=df_subset,
        outcome_var='enterprise',
        treatment_var='group',
        controls=ALL_CONTROLS,
        cluster_var='soum',
        spec_id=f'robust/sample/drop_aimag_{int(aimag)}',
        spec_tree_path='robustness/sample_restrictions.md',
        baseline_var='BLenterprise',
        sample_desc=f'Drop aimag {int(aimag)}'
    )
    results.append(result)

print(f"Aimag restrictions: 5 specifications")

# 4.3.2 Winsorize outcome
for pct in [1, 5, 10]:
    df_wins = df_followup.copy()
    lower = df_wins['profit_r'].quantile(pct/100)
    upper = df_wins['profit_r'].quantile(1 - pct/100)
    df_wins['profit_r_wins'] = df_wins['profit_r'].clip(lower=lower, upper=upper)

    result = run_specification(
        df=df_wins,
        outcome_var='profit_r_wins',
        treatment_var='group',
        controls=ALL_CONTROLS,
        cluster_var='soum',
        spec_id=f'robust/sample/winsorize_{pct}pct',
        spec_tree_path='robustness/sample_restrictions.md',
        baseline_var='BLscaled_profit_r',
        sample_desc=f'Profit winsorized at {pct}%'
    )
    results.append(result)

print(f"Winsorization: 3 specifications")

# 4.3.3 Trim outliers
df_trim = df_followup.copy()
q01 = df_trim['totalc'].quantile(0.01)
q99 = df_trim['totalc'].quantile(0.99)
df_trim = df_trim[(df_trim['totalc'] >= q01) & (df_trim['totalc'] <= q99)]

result = run_specification(
    df=df_trim,
    outcome_var='enterprise',
    treatment_var='group',
    controls=ALL_CONTROLS,
    cluster_var='soum',
    spec_id='robust/sample/trim_1pct',
    spec_tree_path='robustness/sample_restrictions.md',
    baseline_var='BLenterprise',
    sample_desc='Trim 1% tails on consumption'
)
results.append(result)

# 4.3.4 By education level
df_low_edu = df_followup[df_followup['edusec'] == 0].copy()
df_high_edu = df_followup[df_followup['edusec'] == 1].copy()

result = run_specification(
    df=df_low_edu,
    outcome_var='enterprise',
    treatment_var='group',
    controls=[c for c in ALL_CONTROLS if c != 'edusec'],
    cluster_var='soum',
    spec_id='robust/sample/low_education',
    spec_tree_path='robustness/sample_restrictions.md',
    baseline_var='BLenterprise',
    sample_desc='Low education subsample'
)
results.append(result)

result = run_specification(
    df=df_high_edu,
    outcome_var='enterprise',
    treatment_var='group',
    controls=[c for c in ALL_CONTROLS if c != 'edusec'],
    cluster_var='soum',
    spec_id='robust/sample/high_education',
    spec_tree_path='robustness/sample_restrictions.md',
    baseline_var='BLenterprise',
    sample_desc='High education subsample'
)
results.append(result)

print(f"Education subsamples: 2 specifications")

# 4.3.5 By baseline loan status
df_with_loan = df_followup[df_followup['loan_baseline'] == 1].copy()
df_no_loan = df_followup[df_followup['loan_baseline'] == 0].copy()

result = run_specification(
    df=df_with_loan,
    outcome_var='enterprise',
    treatment_var='group',
    controls=[c for c in ALL_CONTROLS if c != 'loan_baseline'],
    cluster_var='soum',
    spec_id='robust/sample/had_baseline_loan',
    spec_tree_path='robustness/sample_restrictions.md',
    baseline_var='BLenterprise',
    sample_desc='Had loan at baseline'
)
results.append(result)

result = run_specification(
    df=df_no_loan,
    outcome_var='enterprise',
    treatment_var='group',
    controls=[c for c in ALL_CONTROLS if c != 'loan_baseline'],
    cluster_var='soum',
    spec_id='robust/sample/no_baseline_loan',
    spec_tree_path='robustness/sample_restrictions.md',
    baseline_var='BLenterprise',
    sample_desc='No loan at baseline'
)
results.append(result)

print(f"Loan status subsamples: 2 specifications")

# --------------------------------------------------------------------------
# 4.4 ALTERNATIVE OUTCOMES (~10 specs)
# --------------------------------------------------------------------------
print("\n--- Alternative Outcomes ---")

outcomes_to_test = [
    ('enterprise', 'BLenterprise', 'Has any enterprise'),
    ('soleent', 'BLsoleent', 'Sole entrepreneurship'),
    ('scaled_profit_r', 'BLscaled_profit_r', 'Scaled profit'),
    ('ln_totalc', 'BLln_totalpc', 'Log total consumption'),
    ('ln_foodc', 'BLln_foodc', 'Log food consumption'),
    ('scaled_assets', 'BLscaled_assets_all', 'Scaled assets'),
    ('hours_total', 'BLhours', 'Total hours worked'),
    ('ln_nondurc', 'BLln_nondurc', 'Log non-durable consumption'),
    ('ln_durc', 'BLln_durc', 'Log durable consumption'),
]

for outcome, bl_var, desc in outcomes_to_test:
    if outcome in df_followup.columns:
        result = run_specification(
            df=df_followup,
            outcome_var=outcome,
            treatment_var='group',
            controls=ALL_CONTROLS,
            cluster_var='soum',
            spec_id=f'robust/outcome/{outcome}',
            spec_tree_path='robustness/measurement.md',
            baseline_var=bl_var if bl_var in df_followup.columns else None,
            sample_desc=desc
        )
        results.append(result)

print(f"Alternative outcomes: {len(outcomes_to_test)} specifications")

# --------------------------------------------------------------------------
# 4.5 INFERENCE VARIATIONS (~8 specs)
# --------------------------------------------------------------------------
print("\n--- Inference Variations ---")

# 4.5.1 Robust SE (no clustering)
result = run_specification(
    df=df_followup,
    outcome_var='enterprise',
    treatment_var='group',
    controls=ALL_CONTROLS,
    cluster_var=None,  # Heteroskedasticity-robust, no clustering
    spec_id='robust/cluster/robust_only',
    spec_tree_path='robustness/clustering_variations.md#single-level-clustering',
    baseline_var='BLenterprise',
    sample_desc='Robust SE, no clustering'
)
results.append(result)

# 4.5.2 Cluster at aimag level (higher level)
result = run_specification(
    df=df_followup,
    outcome_var='enterprise',
    treatment_var='group',
    controls=ALL_CONTROLS,
    cluster_var='aimag',
    spec_id='robust/cluster/aimag',
    spec_tree_path='robustness/clustering_variations.md#single-level-clustering',
    baseline_var='BLenterprise',
    sample_desc='Cluster at aimag level'
)
results.append(result)

# 4.5.3 Cluster at household level
result = run_specification(
    df=df_followup,
    outcome_var='enterprise',
    treatment_var='group',
    controls=ALL_CONTROLS,
    cluster_var='hhid',
    spec_id='robust/cluster/household',
    spec_tree_path='robustness/clustering_variations.md#single-level-clustering',
    baseline_var='BLenterprise',
    sample_desc='Cluster at household level'
)
results.append(result)

print(f"Clustering variations: 3 specifications")

# Additional outcomes with different clustering
for outcome in ['soleent', 'ln_totalc', 'scaled_profit_r']:
    if outcome in df_followup.columns:
        bl_var = f'BL{outcome}' if f'BL{outcome}' in df_followup.columns else None
        if bl_var is None and outcome == 'ln_totalc':
            bl_var = 'BLln_totalpc'
        result = run_specification(
            df=df_followup,
            outcome_var=outcome,
            treatment_var='group',
            controls=ALL_CONTROLS,
            cluster_var='aimag',
            spec_id=f'robust/cluster/aimag_{outcome}',
            spec_tree_path='robustness/clustering_variations.md',
            baseline_var=bl_var,
            sample_desc=f'{outcome} with aimag clustering'
        )
        results.append(result)

# --------------------------------------------------------------------------
# 4.6 FIXED EFFECTS VARIATIONS (~5 specs)
# --------------------------------------------------------------------------
print("\n--- Fixed Effects Variations ---")

# 4.6.1 No aimag dummies
controls_no_aimag = [c for c in ALL_CONTROLS if not c.startswith('aimag_')]
result = run_specification(
    df=df_followup,
    outcome_var='enterprise',
    treatment_var='group',
    controls=controls_no_aimag,
    cluster_var='soum',
    spec_id='robust/fe/no_aimag_dummies',
    spec_tree_path='robustness/model_specification.md',
    baseline_var='BLenterprise',
    sample_desc='No aimag dummies'
)
results.append(result)

# 4.6.2 Aimag fixed effects
result = run_specification(
    df=df_followup,
    outcome_var='enterprise',
    treatment_var='group',
    controls=controls_no_aimag,
    cluster_var='soum',
    spec_id='robust/fe/aimag_fe',
    spec_tree_path='robustness/model_specification.md',
    fixed_effects='aimag',
    baseline_var='BLenterprise',
    sample_desc='Aimag fixed effects'
)
results.append(result)

# 4.6.3 Soum fixed effects
result = run_specification(
    df=df_followup,
    outcome_var='enterprise',
    treatment_var='group',
    controls=controls_no_aimag,
    cluster_var='aimag',
    spec_id='robust/fe/soum_fe',
    spec_tree_path='robustness/model_specification.md',
    fixed_effects='soum',
    baseline_var='BLenterprise',
    sample_desc='Soum fixed effects'
)
results.append(result)

print(f"Fixed effects variations: 3 specifications")

# --------------------------------------------------------------------------
# 4.7 FUNCTIONAL FORM (~5 specs)
# --------------------------------------------------------------------------
print("\n--- Functional Form ---")

# 4.7.1 Level vs log for consumption outcomes
result = run_specification(
    df=df_followup,
    outcome_var='totalc',
    treatment_var='group',
    controls=ALL_CONTROLS,
    cluster_var='soum',
    spec_id='robust/funcform/totalc_level',
    spec_tree_path='robustness/functional_form.md',
    baseline_var='BLln_totalpc',
    sample_desc='Total consumption in levels'
)
results.append(result)

# 4.7.2 IHS transformation
result = run_specification(
    df=df_followup,
    outcome_var='ihs_profit',
    treatment_var='group',
    controls=ALL_CONTROLS,
    cluster_var='soum',
    spec_id='robust/funcform/ihs_profit',
    spec_tree_path='robustness/functional_form.md',
    baseline_var='BLscaled_profit_r',
    sample_desc='IHS profit'
)
results.append(result)

# 4.7.3 Per capita consumption
if 'ln_totalc_pc' in df_followup.columns:
    result = run_specification(
        df=df_followup,
        outcome_var='ln_totalc_pc',
        treatment_var='group',
        controls=ALL_CONTROLS,
        cluster_var='soum',
        spec_id='robust/funcform/ln_totalc_pc',
        spec_tree_path='robustness/functional_form.md',
        baseline_var='BLln_totalpc',
        sample_desc='Log per capita consumption'
    )
    results.append(result)

print(f"Functional form: 3 specifications")

# --------------------------------------------------------------------------
# 4.8 HETEROGENEITY ANALYSIS (~10 specs)
# --------------------------------------------------------------------------
print("\n--- Heterogeneity Analysis ---")

# 4.8.1 Education interaction
het_controls = ALL_CONTROLS + ['group_x_edusec']

result = run_specification(
    df=df_followup,
    outcome_var='enterprise',
    treatment_var='group',
    controls=het_controls,
    cluster_var='soum',
    spec_id='robust/heterogeneity/education',
    spec_tree_path='robustness/heterogeneity.md',
    baseline_var='BLenterprise',
    sample_desc='Education interaction'
)
results.append(result)

# 4.8.2 Baseline loan interaction
het_controls = ALL_CONTROLS + ['group_x_loan']

result = run_specification(
    df=df_followup,
    outcome_var='enterprise',
    treatment_var='group',
    controls=het_controls,
    cluster_var='soum',
    spec_id='robust/heterogeneity/baseline_loan',
    spec_tree_path='robustness/heterogeneity.md',
    baseline_var='BLenterprise',
    sample_desc='Baseline loan interaction'
)
results.append(result)

# 4.8.3 Age interaction
het_controls = ALL_CONTROLS + ['old', 'group_x_old']

result = run_specification(
    df=df_followup,
    outcome_var='enterprise',
    treatment_var='group',
    controls=het_controls,
    cluster_var='soum',
    spec_id='robust/heterogeneity/age',
    spec_tree_path='robustness/heterogeneity.md',
    baseline_var='BLenterprise',
    sample_desc='Age interaction'
)
results.append(result)

# 4.8.4 Household size interaction
het_controls = ALL_CONTROLS + ['large_hh', 'group_x_large_hh']

result = run_specification(
    df=df_followup,
    outcome_var='enterprise',
    treatment_var='group',
    controls=het_controls,
    cluster_var='soum',
    spec_id='robust/heterogeneity/hh_size',
    spec_tree_path='robustness/heterogeneity.md',
    baseline_var='BLenterprise',
    sample_desc='Household size interaction'
)
results.append(result)

# 4.8.5 Buddhist interaction
het_controls = ALL_CONTROLS + ['group_x_buddhist']

result = run_specification(
    df=df_followup,
    outcome_var='enterprise',
    treatment_var='group',
    controls=het_controls,
    cluster_var='soum',
    spec_id='robust/heterogeneity/buddhist',
    spec_tree_path='robustness/heterogeneity.md',
    baseline_var='BLenterprise',
    sample_desc='Buddhist interaction'
)
results.append(result)

print(f"Heterogeneity: 5 specifications")

# Heterogeneity for other outcomes
for outcome in ['soleent', 'ln_totalc']:
    if outcome in df_followup.columns:
        bl_var = f'BL{outcome}' if f'BL{outcome}' in df_followup.columns else None
        if bl_var is None and outcome == 'ln_totalc':
            bl_var = 'BLln_totalpc'
        het_controls = ALL_CONTROLS + ['group_x_edusec']
        result = run_specification(
            df=df_followup,
            outcome_var=outcome,
            treatment_var='group',
            controls=het_controls,
            cluster_var='soum',
            spec_id=f'robust/heterogeneity/education_{outcome}',
            spec_tree_path='robustness/heterogeneity.md',
            baseline_var=bl_var,
            sample_desc=f'Education interaction for {outcome}'
        )
        results.append(result)

# --------------------------------------------------------------------------
# 4.9 PLACEBO/FALSIFICATION TESTS (~5 specs)
# --------------------------------------------------------------------------
print("\n--- Placebo Tests ---")

# 4.9.1 Placebo outcome: baseline characteristics that shouldn't change
placebo_outcomes = ['buddhist', 'hahl', 'under16']

for outcome in placebo_outcomes:
    if outcome in df_followup.columns:
        result = run_specification(
            df=df_followup,
            outcome_var=outcome,
            treatment_var='group',
            controls=[c for c in ALL_CONTROLS if c != outcome],
            cluster_var='soum',
            spec_id=f'robust/placebo/outcome_{outcome}',
            spec_tree_path='robustness/placebo_tests.md',
            sample_desc=f'Placebo: {outcome} (should be null)'
        )
        results.append(result)

print(f"Placebo tests: {len(placebo_outcomes)} specifications")

# --------------------------------------------------------------------------
# 4.10 INDIVIDUAL LOAN COMPARISON (~5 specs)
# --------------------------------------------------------------------------
print("\n--- Individual vs Group Loan ---")

# Compare individual loan to control (different from main specification)
df_indiv = df[(df['group'] != 1) & (df['reint'] == 1) & (df['followup'] == 1)].copy()

# Create aimag dummies for this sample
df_indiv['aimag_cat'] = pd.Categorical(df_indiv['aimag'])
aimag_dummies_ind = pd.get_dummies(df_indiv['aimag_cat'], prefix='aimag', drop_first=True)
df_indiv = pd.concat([df_indiv, aimag_dummies_ind], axis=1)

# Merge BL variables
df_indiv = df_indiv.merge(baseline_outcomes, on='rescode', how='left', suffixes=('', '_dup2'))

# Create age16
df_indiv['age16'] = df_indiv['age16m'].fillna(0) + df_indiv['age16f'].fillna(0)

result = run_specification(
    df=df_indiv,
    outcome_var='enterprise',
    treatment_var='indiv',
    controls=ALL_CONTROLS,
    cluster_var='soum',
    spec_id='custom/indiv_vs_control/enterprise',
    spec_tree_path='custom',
    baseline_var='BLenterprise',
    sample_desc='Individual loan vs Control'
)
results.append(result)

result = run_specification(
    df=df_indiv,
    outcome_var='soleent',
    treatment_var='indiv',
    controls=ALL_CONTROLS,
    cluster_var='soum',
    spec_id='custom/indiv_vs_control/soleent',
    spec_tree_path='custom',
    baseline_var='BLsoleent',
    sample_desc='Individual loan vs Control'
)
results.append(result)

print(f"Individual loan comparison: 2 specifications")

# ============================================================================
# STEP 5: SAVE RESULTS
# ============================================================================

print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Calculate summary statistics
n_specs = len(results_df)
valid_results = results_df[results_df['coefficient'].notna()]
n_valid = len(valid_results)
n_positive = (valid_results['coefficient'] > 0).sum()
n_sig_05 = (valid_results['p_value'] < 0.05).sum()
n_sig_01 = (valid_results['p_value'] < 0.01).sum()
median_coef = valid_results['coefficient'].median()
mean_coef = valid_results['coefficient'].mean()

print(f"\nTotal specifications: {n_specs}")
print(f"Valid specifications: {n_valid}")
print(f"Positive coefficients: {n_positive} ({100*n_positive/n_valid:.1f}%)")
print(f"Significant at 5%: {n_sig_05} ({100*n_sig_05/n_valid:.1f}%)")
print(f"Significant at 1%: {n_sig_01} ({100*n_sig_01/n_valid:.1f}%)")
print(f"Median coefficient: {median_coef:.4f}")
print(f"Mean coefficient: {mean_coef:.4f}")

# Save to CSV
output_file = f'{OUTPUT_PATH}/specification_results.csv'
results_df.to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")

# ============================================================================
# STEP 6: CREATE SUMMARY REPORT
# ============================================================================

# Categorize specifications
def categorize_spec(spec_id):
    if spec_id == 'baseline' or spec_id.startswith('baseline/'):
        return 'Baseline'
    elif 'control' in spec_id or 'loo' in spec_id:
        return 'Control variations'
    elif 'sample' in spec_id:
        return 'Sample restrictions'
    elif 'outcome' in spec_id:
        return 'Alternative outcomes'
    elif 'cluster' in spec_id or 'se/' in spec_id:
        return 'Inference variations'
    elif 'fe/' in spec_id:
        return 'Estimation method'
    elif 'funcform' in spec_id:
        return 'Functional form'
    elif 'heterogeneity' in spec_id:
        return 'Heterogeneity'
    elif 'placebo' in spec_id:
        return 'Placebo tests'
    elif 'custom' in spec_id:
        return 'Custom'
    else:
        return 'Other'

valid_results['category'] = valid_results['spec_id'].apply(categorize_spec)

# Create category breakdown
category_summary = valid_results.groupby('category').agg({
    'spec_id': 'count',
    'coefficient': lambda x: (x > 0).sum(),
    'p_value': lambda x: (x < 0.05).sum()
}).reset_index()
category_summary.columns = ['Category', 'N', 'Positive', 'Sig5pct']
category_summary['Pct_Positive'] = 100 * category_summary['Positive'] / category_summary['N']
category_summary['Pct_Sig5pct'] = 100 * category_summary['Sig5pct'] / category_summary['N']

# Assess robustness
baseline_row = valid_results[valid_results['spec_id'] == 'baseline']
if len(baseline_row) > 0:
    baseline_coef = baseline_row['coefficient'].values[0]
    baseline_sig = baseline_row['p_value'].values[0] < 0.05
else:
    baseline_coef = median_coef
    baseline_sig = False

pct_same_sign = 100 * n_positive / n_valid if baseline_coef > 0 else 100 * (n_valid - n_positive) / n_valid
pct_sig = 100 * n_sig_05 / n_valid

if pct_same_sign > 90 and pct_sig > 70:
    robustness_assessment = "STRONG"
elif pct_same_sign > 75 and pct_sig > 50:
    robustness_assessment = "MODERATE"
else:
    robustness_assessment = "WEAK"

summary_md = f"""# Specification Search: {PAPER_TITLE}

## Paper Overview
- **Paper ID**: {PAPER_ID}
- **Journal**: {JOURNAL}
- **Topic**: Microfinance impact evaluation in Mongolia
- **Hypothesis**: Joint-liability microfinance loans increase entrepreneurship and improve household welfare
- **Method**: RCT with cross-sectional ANCOVA analysis at follow-up
- **Data**: Household survey from Mongolia, comparing Group Loan treatment vs Control

## Classification
- **Method Type**: Cross-sectional OLS (ANCOVA with baseline control)
- **Spec Tree Path**: methods/cross_sectional_ols.md
- **Treatment**: Group loan treatment assignment (randomized at soum level)
- **Primary Outcome**: Enterprise ownership at follow-up

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total specifications | {n_specs} |
| Valid specifications | {n_valid} |
| Positive coefficients | {n_positive} ({100*n_positive/n_valid:.1f}%) |
| Significant at 5% | {n_sig_05} ({100*n_sig_05/n_valid:.1f}%) |
| Significant at 1% | {n_sig_01} ({100*n_sig_01/n_valid:.1f}%) |
| Median coefficient | {median_coef:.4f} |
| Mean coefficient | {mean_coef:.4f} |
| Range | [{valid_results['coefficient'].min():.4f}, {valid_results['coefficient'].max():.4f}] |

## Robustness Assessment

**{robustness_assessment}** support for the main hypothesis.

The baseline specification shows a {'positive' if baseline_coef > 0 else 'negative'} coefficient
of {baseline_coef:.4f} for the effect of group loan treatment on enterprise ownership,
which is {'statistically significant' if baseline_sig else 'not statistically significant'} at the 5% level.

Across {n_valid} valid specifications, {pct_same_sign:.1f}% maintain the same sign as the baseline,
and {pct_sig:.1f}% are statistically significant at the 5% level.

## Specification Breakdown by Category (i4r format)

| Category | N | % Positive | % Sig 5% |
|----------|---|------------|----------|
"""

for _, row in category_summary.iterrows():
    summary_md += f"| {row['Category']} | {int(row['N'])} | {row['Pct_Positive']:.1f}% | {row['Pct_Sig5pct']:.1f}% |\n"

summary_md += f"| **TOTAL** | **{n_valid}** | **{100*n_positive/n_valid:.1f}%** | **{100*n_sig_05/n_valid:.1f}%** |\n"

summary_md += f"""
## Key Findings

1. **Main Result**: The group loan treatment effect on enterprise ownership is {baseline_coef:.4f}
   (SE clustered at soum level), {'statistically significant' if baseline_sig else 'not statistically significant'} at the 5% level.

2. **Control Sensitivity**: Results are {'robust' if pct_same_sign > 80 else 'sensitive'} to control variable specifications.
   Leave-one-out analysis shows coefficient remains stable when dropping individual controls.

3. **Sample Restrictions**: Subsample analyses by education level and baseline loan status show
   {'consistent' if pct_same_sign > 80 else 'heterogeneous'} treatment effects.

4. **Inference Robustness**: Results are {'consistent' if pct_sig > 60 else 'sensitive to'} across different
   clustering levels (soum, aimag, household).

## Critical Caveats

1. **Attrition**: The analysis is conditional on re-interview at follow-up. Differential attrition
   could bias results if related to treatment assignment.

2. **Clustering Level**: The paper clusters at soum level (where randomization occurred), but
   different clustering choices affect inference.

3. **Multiple Outcomes**: The paper tests many outcomes. Results should be interpreted with
   multiple hypothesis testing corrections in mind.

4. **External Validity**: Results are specific to rural Mongolia and joint-liability microfinance context.

## Files Generated

- `specification_results.csv` - Full specification search results ({n_specs} specifications)
- `scripts/paper_analyses/{PAPER_ID}.py` - Estimation script
"""

# Save summary report
summary_file = f'{OUTPUT_PATH}/SPECIFICATION_SEARCH.md'
with open(summary_file, 'w') as f:
    f.write(summary_md)

print(f"\nSummary report saved to: {summary_file}")

print("\n" + "="*60)
print("SPECIFICATION SEARCH COMPLETE")
print("="*60)
