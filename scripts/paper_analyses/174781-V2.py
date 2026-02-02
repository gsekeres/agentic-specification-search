"""
Specification Search: Work and Mental Health among Rohingya Refugees
Paper ID: 174781-V2
Journal: AER

This script runs a systematic specification search following the i4r methodology.

Method: Cross-sectional OLS (RCT)
- The paper is a randomized controlled trial comparing:
  - Work treatment (cash-for-work program)
  - Cash treatment (unconditional cash transfer)
  - Control (smaller cash transfer)

Main outcome: Mental health index (composite of PHQ, stress, life satisfaction, etc.)
Main treatment: Work vs Cash comparison
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

# Import regression packages
import pyfixest as pf
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Paths
BASE_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search"
PACKAGE_PATH = f"{BASE_PATH}/data/downloads/extracted/174781-V2/replication_AER"
OUTPUT_PATH = f"{BASE_PATH}/data/downloads/extracted/174781-V2"

# Paper metadata
PAPER_ID = "174781-V2"
JOURNAL = "AER"
PAPER_TITLE = "Work and Mental Health among Rohingya Refugees"

# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================

print("Loading data...")
df = pd.read_stata(f"{PACKAGE_PATH}/data/3.Endline/3_processed/endline_processed.dta")

# Convert categorical variables to numeric
df['female'] = (df['b_resp_gender'].astype(str).str.lower().str.contains('female')).astype(int)
df['hh_head'] = (df['b_hh_head'].astype(str) == '1').astype(int)

# Create enumerator dummies
df['enum_code'] = df['enumname'].astype(str).astype('category').cat.codes
df['camp_code'] = df['b_campname'].astype(str).astype('category').cat.codes

# Create numeric version of block for clustering
df['block'] = df['b_block']

# Define treatment variables (from paper)
# b_treat_work = 1 if in work treatment
# b_treat_largecash = 1 if in large cash treatment
# b_treat_smallcash = 1 if in control (small cash)
# Baseline spec compares work vs cash (excluding control) or includes all three

# Define main outcomes (Table 2 outcomes)
MENTAL_OUTCOMES = [
    'mental_health_index',  # Primary outcome - composite index
    'phq_sd_scale',        # PHQ depression scale
    'stress_index',        # Stress index
    'life_satisfaction',   # Life satisfaction
    'sociability_a',       # Sociability
    'selfworth_index',     # Self-worth
    'control_index',       # Perceived control
    'stability'            # Stability
]

# Baseline versions for ANCOVA controls
BASELINE_OUTCOMES = [f'b_{out}' for out in MENTAL_OUTCOMES]

# Control variables used in paper
# Paper uses pdslasso (post-double-selection LASSO) with these possible controls
POSSIBLE_CONTROLS = ['female', 'age', 'marry_dum', 'b_member_count',
                     'education_formal', 'education_religious', 'hh_head']

# Define the treatment variable
# Main comparison is Work vs Cash (both treatment groups)
# Paper compares: b_treat_work and b_treat_largecash (with b_treat_smallcash as reference)

print(f"Sample size: {len(df)}")
print(f"Treatment groups: Work={df['b_treat_work'].sum()}, Cash={df['b_treat_largecash'].sum()}, Control={df['b_treat_smallcash'].sum()}")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def run_ols_spec(df_in, outcome, treatment_vars, controls, fe_vars=None, cluster_var=None,
                 spec_id='', spec_tree_path='', sample_desc='Full sample'):
    """
    Run OLS regression and return standardized results dict.
    """
    try:
        # Build formula
        df_reg = df_in.copy()

        # Handle missing values
        all_vars = [outcome] + treatment_vars + controls
        if fe_vars:
            all_vars += fe_vars
        if cluster_var:
            all_vars += [cluster_var]

        # Drop missing
        df_reg = df_reg.dropna(subset=[v for v in all_vars if v in df_reg.columns])

        if len(df_reg) < 50:
            return None

        # Build regression formula
        if controls:
            controls_str = ' + '.join(controls)
            formula = f"{outcome} ~ {' + '.join(treatment_vars)} + {controls_str}"
        else:
            formula = f"{outcome} ~ {' + '.join(treatment_vars)}"

        # Add fixed effects
        if fe_vars:
            formula += f" | {' + '.join(fe_vars)}"

        # Run regression
        if fe_vars:
            if cluster_var:
                model = pf.feols(formula, data=df_reg, vcov={'CRV1': cluster_var})
            else:
                model = pf.feols(formula, data=df_reg, vcov='hetero')
        else:
            # No FE - use statsmodels
            if cluster_var:
                # Cluster robust SE
                model_sm = smf.ols(formula, data=df_reg).fit(
                    cov_type='cluster',
                    cov_kwds={'groups': df_reg[cluster_var]}
                )
            else:
                model_sm = smf.ols(formula, data=df_reg).fit(cov_type='HC1')

            # Get results
            main_treat = treatment_vars[0]
            coef = model_sm.params[main_treat]
            se = model_sm.bse[main_treat]
            tstat = model_sm.tvalues[main_treat]
            pval = model_sm.pvalues[main_treat]
            ci = model_sm.conf_int().loc[main_treat]
            n_obs = int(model_sm.nobs)
            r2 = model_sm.rsquared

            # Build coefficient vector
            coef_vector = {
                'treatment': {
                    'var': main_treat,
                    'coef': float(coef),
                    'se': float(se),
                    'pval': float(pval)
                },
                'controls': [],
                'fixed_effects': fe_vars if fe_vars else [],
                'diagnostics': {
                    'r_squared': float(r2),
                    'adj_r_squared': float(model_sm.rsquared_adj)
                }
            }

            # Add other treatment if present
            if len(treatment_vars) > 1:
                for t in treatment_vars[1:]:
                    coef_vector['controls'].append({
                        'var': t,
                        'coef': float(model_sm.params[t]),
                        'se': float(model_sm.bse[t]),
                        'pval': float(model_sm.pvalues[t])
                    })

            # Add controls
            for c in controls:
                if c in model_sm.params:
                    coef_vector['controls'].append({
                        'var': c,
                        'coef': float(model_sm.params[c]),
                        'se': float(model_sm.bse[c]),
                        'pval': float(model_sm.pvalues[c])
                    })

            return {
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': spec_id,
                'spec_tree_path': spec_tree_path,
                'outcome_var': outcome,
                'treatment_var': main_treat,
                'coefficient': float(coef),
                'std_error': float(se),
                't_stat': float(tstat),
                'p_value': float(pval),
                'ci_lower': float(ci[0]),
                'ci_upper': float(ci[1]),
                'n_obs': n_obs,
                'r_squared': float(r2),
                'coefficient_vector_json': json.dumps(coef_vector),
                'sample_desc': sample_desc,
                'fixed_effects': ', '.join(fe_vars) if fe_vars else 'None',
                'controls_desc': ', '.join(controls) if controls else 'None',
                'cluster_var': cluster_var if cluster_var else 'None',
                'model_type': 'OLS'
            }

        # With fixed effects using pyfixest
        main_treat = treatment_vars[0]
        coef = model.coef()[main_treat]
        se = model.se()[main_treat]
        tstat = model.tstat()[main_treat]
        pval = model.pvalue()[main_treat]
        ci = model.confint().loc[main_treat]
        n_obs = int(model._N)
        r2 = model._r2

        # Build coefficient vector
        coef_vector = {
            'treatment': {
                'var': main_treat,
                'coef': float(coef),
                'se': float(se),
                'pval': float(pval)
            },
            'controls': [],
            'fixed_effects': fe_vars if fe_vars else [],
            'diagnostics': {
                'r_squared': float(r2)
            }
        }

        # Add other coefficients
        for var in model.coef().index:
            if var != main_treat and var != 'Intercept':
                coef_vector['controls'].append({
                    'var': var,
                    'coef': float(model.coef()[var]),
                    'se': float(model.se()[var]),
                    'pval': float(model.pvalue()[var])
                })

        return {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': outcome,
            'treatment_var': main_treat,
            'coefficient': float(coef),
            'std_error': float(se),
            't_stat': float(tstat),
            'p_value': float(pval),
            'ci_lower': float(ci[0]),
            'ci_upper': float(ci[1]),
            'n_obs': n_obs,
            'r_squared': float(r2),
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc,
            'fixed_effects': ', '.join(fe_vars) if fe_vars else 'None',
            'controls_desc': ', '.join(controls) if controls else 'None',
            'cluster_var': cluster_var if cluster_var else 'None',
            'model_type': 'FE-OLS'
        }

    except Exception as e:
        print(f"  Error in {spec_id}: {e}")
        return None

# =============================================================================
# RUN SPECIFICATIONS
# =============================================================================

results = []

# Paper baseline: Work treatment vs Cash treatment
# Uses camp and enumerator FE, clusters at block level
# Controls for baseline outcome + gender + lasso-selected controls

print("\n" + "="*80)
print("BASELINE SPECIFICATIONS")
print("="*80)

# BASELINE - Main outcome (Mental Health Index) - Table 2 Column 1
# Paper uses pdslasso - we approximate with baseline outcome + key controls
baseline_controls = ['b_mental_health_index', 'female']  # Paper always includes these
baseline_fe = ['camp_code', 'enum_code']

for outcome in MENTAL_OUTCOMES:
    baseline_out = f'b_{outcome}'
    controls = [baseline_out, 'female'] if baseline_out in df.columns else ['female']

    result = run_ols_spec(
        df, outcome,
        treatment_vars=['b_treat_work', 'b_treat_largecash'],
        controls=controls,
        fe_vars=baseline_fe,
        cluster_var='block',
        spec_id='baseline' if outcome == 'mental_health_index' else f'baseline/{outcome}',
        spec_tree_path='methods/cross_sectional_ols.md#baseline',
        sample_desc='Full sample'
    )
    if result:
        results.append(result)
        print(f"  {outcome}: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

print(f"\nBaseline specs completed: {len(results)}")

# =============================================================================
# CONTROL VARIABLE PROGRESSION (10+ specs)
# =============================================================================

print("\n" + "="*80)
print("CONTROL PROGRESSION")
print("="*80)

# Main outcome for robustness
main_outcome = 'mental_health_index'
main_baseline = 'b_mental_health_index'

# 1. Bivariate (no controls)
result = run_ols_spec(
    df, main_outcome,
    treatment_vars=['b_treat_work', 'b_treat_largecash'],
    controls=[],
    fe_vars=['camp_code', 'enum_code'],
    cluster_var='block',
    spec_id='robust/build/bivariate',
    spec_tree_path='robustness/control_progression.md'
)
if result:
    results.append(result)
    print(f"  Bivariate: coef={result['coefficient']:.4f}")

# 2. Only baseline outcome (ANCOVA)
result = run_ols_spec(
    df, main_outcome,
    treatment_vars=['b_treat_work', 'b_treat_largecash'],
    controls=[main_baseline],
    fe_vars=['camp_code', 'enum_code'],
    cluster_var='block',
    spec_id='robust/build/ancova_only',
    spec_tree_path='robustness/control_progression.md'
)
if result:
    results.append(result)
    print(f"  ANCOVA only: coef={result['coefficient']:.4f}")

# 3. Demographics only
result = run_ols_spec(
    df, main_outcome,
    treatment_vars=['b_treat_work', 'b_treat_largecash'],
    controls=['female', 'age', 'marry_dum'],
    fe_vars=['camp_code', 'enum_code'],
    cluster_var='block',
    spec_id='robust/build/demographics',
    spec_tree_path='robustness/control_progression.md'
)
if result:
    results.append(result)
    print(f"  Demographics: coef={result['coefficient']:.4f}")

# 4. Add household controls
result = run_ols_spec(
    df, main_outcome,
    treatment_vars=['b_treat_work', 'b_treat_largecash'],
    controls=['female', 'age', 'marry_dum', 'b_member_count', 'hh_head'],
    fe_vars=['camp_code', 'enum_code'],
    cluster_var='block',
    spec_id='robust/build/household',
    spec_tree_path='robustness/control_progression.md'
)
if result:
    results.append(result)
    print(f"  + Household: coef={result['coefficient']:.4f}")

# 5. Add education controls
result = run_ols_spec(
    df, main_outcome,
    treatment_vars=['b_treat_work', 'b_treat_largecash'],
    controls=['female', 'age', 'marry_dum', 'b_member_count', 'education_formal', 'education_religious'],
    fe_vars=['camp_code', 'enum_code'],
    cluster_var='block',
    spec_id='robust/build/education',
    spec_tree_path='robustness/control_progression.md'
)
if result:
    results.append(result)
    print(f"  + Education: coef={result['coefficient']:.4f}")

# 6. Full controls (paper baseline + all)
result = run_ols_spec(
    df, main_outcome,
    treatment_vars=['b_treat_work', 'b_treat_largecash'],
    controls=[main_baseline] + POSSIBLE_CONTROLS,
    fe_vars=['camp_code', 'enum_code'],
    cluster_var='block',
    spec_id='robust/build/full',
    spec_tree_path='robustness/control_progression.md'
)
if result:
    results.append(result)
    print(f"  Full controls: coef={result['coefficient']:.4f}")

# 7-13. Leave-one-out for each control
all_controls = [main_baseline] + POSSIBLE_CONTROLS
for drop_control in all_controls:
    remaining = [c for c in all_controls if c != drop_control]
    result = run_ols_spec(
        df, main_outcome,
        treatment_vars=['b_treat_work', 'b_treat_largecash'],
        controls=remaining,
        fe_vars=['camp_code', 'enum_code'],
        cluster_var='block',
        spec_id=f'robust/control/drop_{drop_control}',
        spec_tree_path='robustness/leave_one_out.md'
    )
    if result:
        results.append(result)
        print(f"  Drop {drop_control}: coef={result['coefficient']:.4f}")

print(f"\nControl progression specs: {len(results)}")

# =============================================================================
# SAMPLE RESTRICTIONS (10+ specs)
# =============================================================================

print("\n" + "="*80)
print("SAMPLE RESTRICTIONS")
print("="*80)

# Use baseline spec controls
baseline_ctrl = [main_baseline, 'female']

# 1-3. By camp
for camp in df['b_campname'].unique():
    df_sub = df[df['b_campname'] == camp]
    camp_name = f'camp_{camp}'
    result = run_ols_spec(
        df_sub, main_outcome,
        treatment_vars=['b_treat_work', 'b_treat_largecash'],
        controls=baseline_ctrl,
        fe_vars=['enum_code'],  # Only enumerator FE within camp
        cluster_var='block',
        spec_id=f'robust/sample/drop_{camp_name}',
        spec_tree_path='robustness/sample_restrictions.md',
        sample_desc=f'Camp {camp} only'
    )
    if result:
        results.append(result)
        print(f"  Camp {camp}: coef={result['coefficient']:.4f}, n={result['n_obs']}")

# 4. Male only
df_male = df[df['female'] == 0]
result = run_ols_spec(
    df_male, main_outcome,
    treatment_vars=['b_treat_work', 'b_treat_largecash'],
    controls=[main_baseline],  # exclude gender
    fe_vars=['camp_code', 'enum_code'],
    cluster_var='block',
    spec_id='robust/sample/male_only',
    spec_tree_path='robustness/sample_restrictions.md',
    sample_desc='Male only'
)
if result:
    results.append(result)
    print(f"  Male only: coef={result['coefficient']:.4f}, n={result['n_obs']}")

# 5. Female only
df_female = df[df['female'] == 1]
result = run_ols_spec(
    df_female, main_outcome,
    treatment_vars=['b_treat_work', 'b_treat_largecash'],
    controls=[main_baseline],
    fe_vars=['camp_code', 'enum_code'],
    cluster_var='block',
    spec_id='robust/sample/female_only',
    spec_tree_path='robustness/sample_restrictions.md',
    sample_desc='Female only'
)
if result:
    results.append(result)
    print(f"  Female only: coef={result['coefficient']:.4f}, n={result['n_obs']}")

# 6. Young (below median age)
median_age = df['age'].median()
df_young = df[df['age'] <= median_age]
result = run_ols_spec(
    df_young, main_outcome,
    treatment_vars=['b_treat_work', 'b_treat_largecash'],
    controls=baseline_ctrl,
    fe_vars=['camp_code', 'enum_code'],
    cluster_var='block',
    spec_id='robust/sample/young',
    spec_tree_path='robustness/sample_restrictions.md',
    sample_desc=f'Age <= {median_age}'
)
if result:
    results.append(result)
    print(f"  Young: coef={result['coefficient']:.4f}, n={result['n_obs']}")

# 7. Old (above median age)
df_old = df[df['age'] > median_age]
result = run_ols_spec(
    df_old, main_outcome,
    treatment_vars=['b_treat_work', 'b_treat_largecash'],
    controls=baseline_ctrl,
    fe_vars=['camp_code', 'enum_code'],
    cluster_var='block',
    spec_id='robust/sample/old',
    spec_tree_path='robustness/sample_restrictions.md',
    sample_desc=f'Age > {median_age}'
)
if result:
    results.append(result)
    print(f"  Old: coef={result['coefficient']:.4f}, n={result['n_obs']}")

# 8. Married only
df_married = df[df['marry_dum'] == 1]
result = run_ols_spec(
    df_married, main_outcome,
    treatment_vars=['b_treat_work', 'b_treat_largecash'],
    controls=[main_baseline, 'female'],
    fe_vars=['camp_code', 'enum_code'],
    cluster_var='block',
    spec_id='robust/sample/married_only',
    spec_tree_path='robustness/sample_restrictions.md',
    sample_desc='Married only'
)
if result:
    results.append(result)
    print(f"  Married: coef={result['coefficient']:.4f}, n={result['n_obs']}")

# 9. Formally educated
df_edu = df[df['education_formal'] == 1]
result = run_ols_spec(
    df_edu, main_outcome,
    treatment_vars=['b_treat_work', 'b_treat_largecash'],
    controls=[main_baseline, 'female'],
    fe_vars=['camp_code', 'enum_code'],
    cluster_var='block',
    spec_id='robust/sample/educated',
    spec_tree_path='robustness/sample_restrictions.md',
    sample_desc='Formally educated'
)
if result:
    results.append(result)
    print(f"  Educated: coef={result['coefficient']:.4f}, n={result['n_obs']}")

# 10. Not formally educated
df_noedu = df[df['education_formal'] == 0]
result = run_ols_spec(
    df_noedu, main_outcome,
    treatment_vars=['b_treat_work', 'b_treat_largecash'],
    controls=[main_baseline, 'female'],
    fe_vars=['camp_code', 'enum_code'],
    cluster_var='block',
    spec_id='robust/sample/not_educated',
    spec_tree_path='robustness/sample_restrictions.md',
    sample_desc='Not formally educated'
)
if result:
    results.append(result)
    print(f"  Not educated: coef={result['coefficient']:.4f}, n={result['n_obs']}")

# 11. Winsorize outcome at 1%/99%
df_wins = df.copy()
for out in MENTAL_OUTCOMES:
    df_wins[f'{out}_w1'] = df_wins[out].clip(
        lower=df_wins[out].quantile(0.01),
        upper=df_wins[out].quantile(0.99)
    )
result = run_ols_spec(
    df_wins, 'mental_health_index_w1',
    treatment_vars=['b_treat_work', 'b_treat_largecash'],
    controls=baseline_ctrl,
    fe_vars=['camp_code', 'enum_code'],
    cluster_var='block',
    spec_id='robust/sample/winsor_1pct',
    spec_tree_path='robustness/sample_restrictions.md',
    sample_desc='Outcome winsorized at 1%'
)
if result:
    results.append(result)
    print(f"  Winsorized 1%: coef={result['coefficient']:.4f}")

# 12. Trim extreme values
df_trim = df[(df[main_outcome] > df[main_outcome].quantile(0.01)) &
             (df[main_outcome] < df[main_outcome].quantile(0.99))]
result = run_ols_spec(
    df_trim, main_outcome,
    treatment_vars=['b_treat_work', 'b_treat_largecash'],
    controls=baseline_ctrl,
    fe_vars=['camp_code', 'enum_code'],
    cluster_var='block',
    spec_id='robust/sample/trim_1pct',
    spec_tree_path='robustness/sample_restrictions.md',
    sample_desc='Trimmed 1% tails'
)
if result:
    results.append(result)
    print(f"  Trimmed 1%: coef={result['coefficient']:.4f}, n={result['n_obs']}")

print(f"\nSample restriction specs: {len(results)}")

# =============================================================================
# ALTERNATIVE OUTCOMES (8 specs)
# =============================================================================

print("\n" + "="*80)
print("ALTERNATIVE OUTCOMES")
print("="*80)

# Already ran main outcomes in baseline. Now try PHQ components
phq_components = ['phq_1', 'phq_2', 'phq_3', 'phq_4', 'phq_5', 'phq_6', 'phq_7', 'phq_8', 'phq_9']

for outcome in phq_components:
    baseline_out = f'b_{outcome}'
    controls = [baseline_out, 'female'] if baseline_out in df.columns else ['female']

    result = run_ols_spec(
        df, outcome,
        treatment_vars=['b_treat_work', 'b_treat_largecash'],
        controls=controls,
        fe_vars=['camp_code', 'enum_code'],
        cluster_var='block',
        spec_id=f'robust/outcome/{outcome}',
        spec_tree_path='robustness/measurement.md',
        sample_desc='Full sample'
    )
    if result:
        results.append(result)
        print(f"  {outcome}: coef={result['coefficient']:.4f}")

# Depression binary
result = run_ols_spec(
    df, 'depressed',
    treatment_vars=['b_treat_work', 'b_treat_largecash'],
    controls=['b_depressed', 'female'],
    fe_vars=['camp_code', 'enum_code'],
    cluster_var='block',
    spec_id='robust/outcome/depressed_binary',
    spec_tree_path='robustness/measurement.md',
    sample_desc='Full sample'
)
if result:
    results.append(result)
    print(f"  Depressed (binary): coef={result['coefficient']:.4f}")

print(f"\nAlternative outcome specs: {len(results)}")

# =============================================================================
# INFERENCE VARIATIONS (8 specs)
# =============================================================================

print("\n" + "="*80)
print("INFERENCE/CLUSTERING VARIATIONS")
print("="*80)

# 1. Robust SE (no clustering)
result = run_ols_spec(
    df, main_outcome,
    treatment_vars=['b_treat_work', 'b_treat_largecash'],
    controls=baseline_ctrl,
    fe_vars=['camp_code', 'enum_code'],
    cluster_var=None,  # Robust HC SE
    spec_id='robust/cluster/none',
    spec_tree_path='robustness/clustering_variations.md',
    sample_desc='Robust SE (no clustering)'
)
if result:
    results.append(result)
    print(f"  Robust SE: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}")

# 2. Cluster by camp
result = run_ols_spec(
    df, main_outcome,
    treatment_vars=['b_treat_work', 'b_treat_largecash'],
    controls=baseline_ctrl,
    fe_vars=['enum_code'],  # Remove camp FE if clustering by camp
    cluster_var='camp_code',
    spec_id='robust/cluster/camp',
    spec_tree_path='robustness/clustering_variations.md',
    sample_desc='Cluster by camp'
)
if result:
    results.append(result)
    print(f"  Cluster camp: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}")

# 3. Cluster by enumerator
result = run_ols_spec(
    df, main_outcome,
    treatment_vars=['b_treat_work', 'b_treat_largecash'],
    controls=baseline_ctrl,
    fe_vars=['camp_code'],  # Remove enum FE if clustering
    cluster_var='enum_code',
    spec_id='robust/cluster/enumerator',
    spec_tree_path='robustness/clustering_variations.md',
    sample_desc='Cluster by enumerator'
)
if result:
    results.append(result)
    print(f"  Cluster enum: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}")

# 4. No fixed effects at all
result = run_ols_spec(
    df, main_outcome,
    treatment_vars=['b_treat_work', 'b_treat_largecash'],
    controls=baseline_ctrl,
    fe_vars=None,
    cluster_var='block',
    spec_id='robust/estimation/no_fe',
    spec_tree_path='robustness/model_specification.md',
    sample_desc='No fixed effects'
)
if result:
    results.append(result)
    print(f"  No FE: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}")

# 5. Only camp FE
result = run_ols_spec(
    df, main_outcome,
    treatment_vars=['b_treat_work', 'b_treat_largecash'],
    controls=baseline_ctrl,
    fe_vars=['camp_code'],
    cluster_var='block',
    spec_id='robust/estimation/camp_fe_only',
    spec_tree_path='robustness/model_specification.md',
    sample_desc='Camp FE only'
)
if result:
    results.append(result)
    print(f"  Camp FE only: coef={result['coefficient']:.4f}")

# 6. Only enumerator FE
result = run_ols_spec(
    df, main_outcome,
    treatment_vars=['b_treat_work', 'b_treat_largecash'],
    controls=baseline_ctrl,
    fe_vars=['enum_code'],
    cluster_var='block',
    spec_id='robust/estimation/enum_fe_only',
    spec_tree_path='robustness/model_specification.md',
    sample_desc='Enumerator FE only'
)
if result:
    results.append(result)
    print(f"  Enum FE only: coef={result['coefficient']:.4f}")

print(f"\nInference specs: {len(results)}")

# =============================================================================
# HETEROGENEITY ANALYSIS (10+ specs)
# =============================================================================

print("\n" + "="*80)
print("HETEROGENEITY ANALYSIS")
print("="*80)

# Create interaction terms
df['work_female'] = df['b_treat_work'] * df['female']
df['cash_female'] = df['b_treat_largecash'] * df['female']

# 1. Gender interaction
result = run_ols_spec(
    df, main_outcome,
    treatment_vars=['b_treat_work', 'b_treat_largecash', 'work_female', 'cash_female'],
    controls=[main_baseline, 'female'],
    fe_vars=['camp_code', 'enum_code'],
    cluster_var='block',
    spec_id='robust/het/interaction_gender',
    spec_tree_path='robustness/heterogeneity.md',
    sample_desc='Work x Gender interaction'
)
if result:
    results.append(result)
    print(f"  Gender interaction: work coef={result['coefficient']:.4f}")

# Create age interactions
df['work_age'] = df['b_treat_work'] * df['age']
df['cash_age'] = df['b_treat_largecash'] * df['age']

# 2. Age interaction
result = run_ols_spec(
    df, main_outcome,
    treatment_vars=['b_treat_work', 'b_treat_largecash', 'work_age', 'cash_age'],
    controls=[main_baseline, 'female', 'age'],
    fe_vars=['camp_code', 'enum_code'],
    cluster_var='block',
    spec_id='robust/het/interaction_age',
    spec_tree_path='robustness/heterogeneity.md',
    sample_desc='Work x Age interaction'
)
if result:
    results.append(result)
    print(f"  Age interaction: work coef={result['coefficient']:.4f}")

# Create education interactions
df['work_edu'] = df['b_treat_work'] * df['education_formal']
df['cash_edu'] = df['b_treat_largecash'] * df['education_formal']

# 3. Education interaction
result = run_ols_spec(
    df, main_outcome,
    treatment_vars=['b_treat_work', 'b_treat_largecash', 'work_edu', 'cash_edu'],
    controls=[main_baseline, 'female', 'education_formal'],
    fe_vars=['camp_code', 'enum_code'],
    cluster_var='block',
    spec_id='robust/het/interaction_education',
    spec_tree_path='robustness/heterogeneity.md',
    sample_desc='Work x Education interaction'
)
if result:
    results.append(result)
    print(f"  Education interaction: work coef={result['coefficient']:.4f}")

# Create baseline outcome interactions
df['work_baseline'] = df['b_treat_work'] * df[main_baseline]
df['cash_baseline'] = df['b_treat_largecash'] * df[main_baseline]

# 4. Baseline outcome interaction
result = run_ols_spec(
    df, main_outcome,
    treatment_vars=['b_treat_work', 'b_treat_largecash', 'work_baseline', 'cash_baseline'],
    controls=[main_baseline, 'female'],
    fe_vars=['camp_code', 'enum_code'],
    cluster_var='block',
    spec_id='robust/het/interaction_baseline_y',
    spec_tree_path='robustness/heterogeneity.md',
    sample_desc='Work x Baseline MH interaction'
)
if result:
    results.append(result)
    print(f"  Baseline interaction: work coef={result['coefficient']:.4f}")

# Create married interaction
df['work_married'] = df['b_treat_work'] * df['marry_dum']
df['cash_married'] = df['b_treat_largecash'] * df['marry_dum']

# 5. Married interaction
result = run_ols_spec(
    df, main_outcome,
    treatment_vars=['b_treat_work', 'b_treat_largecash', 'work_married', 'cash_married'],
    controls=[main_baseline, 'female', 'marry_dum'],
    fe_vars=['camp_code', 'enum_code'],
    cluster_var='block',
    spec_id='robust/het/interaction_married',
    spec_tree_path='robustness/heterogeneity.md',
    sample_desc='Work x Married interaction'
)
if result:
    results.append(result)
    print(f"  Married interaction: work coef={result['coefficient']:.4f}")

# Check for violence exposure heterogeneity if available
if 'b_any_killed' in df.columns:
    df['work_violence'] = df['b_treat_work'] * df['b_any_killed']
    df['cash_violence'] = df['b_treat_largecash'] * df['b_any_killed']

    result = run_ols_spec(
        df, main_outcome,
        treatment_vars=['b_treat_work', 'b_treat_largecash', 'work_violence', 'cash_violence'],
        controls=[main_baseline, 'female', 'b_any_killed'],
        fe_vars=['camp_code', 'enum_code'],
        cluster_var='block',
        spec_id='robust/het/interaction_violence',
        spec_tree_path='robustness/heterogeneity.md',
        sample_desc='Work x Violence exposure'
    )
    if result:
        results.append(result)
        print(f"  Violence interaction: work coef={result['coefficient']:.4f}")

# Split-sample heterogeneity
# 6. Low baseline mental health
median_mh = df[main_baseline].median()
df_low_mh = df[df[main_baseline] <= median_mh]
result = run_ols_spec(
    df_low_mh, main_outcome,
    treatment_vars=['b_treat_work', 'b_treat_largecash'],
    controls=[main_baseline, 'female'],
    fe_vars=['camp_code', 'enum_code'],
    cluster_var='block',
    spec_id='robust/het/by_baseline_outcome_low',
    spec_tree_path='robustness/heterogeneity.md',
    sample_desc='Low baseline mental health'
)
if result:
    results.append(result)
    print(f"  Low baseline MH: coef={result['coefficient']:.4f}, n={result['n_obs']}")

# 7. High baseline mental health
df_high_mh = df[df[main_baseline] > median_mh]
result = run_ols_spec(
    df_high_mh, main_outcome,
    treatment_vars=['b_treat_work', 'b_treat_largecash'],
    controls=[main_baseline, 'female'],
    fe_vars=['camp_code', 'enum_code'],
    cluster_var='block',
    spec_id='robust/het/by_baseline_outcome_high',
    spec_tree_path='robustness/heterogeneity.md',
    sample_desc='High baseline mental health'
)
if result:
    results.append(result)
    print(f"  High baseline MH: coef={result['coefficient']:.4f}, n={result['n_obs']}")

print(f"\nHeterogeneity specs: {len(results)}")

# =============================================================================
# PLACEBO TESTS (5 specs)
# =============================================================================

print("\n" + "="*80)
print("PLACEBO TESTS")
print("="*80)

# 1. Effect on baseline outcome (should be zero by random assignment)
result = run_ols_spec(
    df, main_baseline,
    treatment_vars=['b_treat_work', 'b_treat_largecash'],
    controls=['female'],
    fe_vars=['camp_code', 'enum_code'],
    cluster_var='block',
    spec_id='robust/placebo/outcome_predetermined',
    spec_tree_path='robustness/placebo_tests.md',
    sample_desc='Placebo: effect on baseline outcome'
)
if result:
    results.append(result)
    print(f"  Baseline outcome placebo: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# 2-9. Placebo on other baseline outcomes
for outcome in MENTAL_OUTCOMES[1:]:  # Skip first since already done
    baseline_out = f'b_{outcome}'
    if baseline_out in df.columns:
        result = run_ols_spec(
            df, baseline_out,
            treatment_vars=['b_treat_work', 'b_treat_largecash'],
            controls=['female'],
            fe_vars=['camp_code', 'enum_code'],
            cluster_var='block',
            spec_id=f'robust/placebo/baseline_{outcome}',
            spec_tree_path='robustness/placebo_tests.md',
            sample_desc=f'Placebo: effect on baseline {outcome}'
        )
        if result:
            results.append(result)
            print(f"  Baseline {outcome} placebo: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

print(f"\nPlacebo specs: {len(results)}")

# =============================================================================
# TREATMENT VARIATIONS (5 specs)
# =============================================================================

print("\n" + "="*80)
print("TREATMENT VARIATIONS")
print("="*80)

# 1. Work vs Control only (excluding large cash)
df_wc = df[df['b_treat_largecash'] == 0]
result = run_ols_spec(
    df_wc, main_outcome,
    treatment_vars=['b_treat_work'],
    controls=[main_baseline, 'female'],
    fe_vars=['camp_code', 'enum_code'],
    cluster_var='block',
    spec_id='robust/treatment/work_vs_control',
    spec_tree_path='robustness/measurement.md',
    sample_desc='Work vs Control (exclude large cash)'
)
if result:
    results.append(result)
    print(f"  Work vs Control: coef={result['coefficient']:.4f}, n={result['n_obs']}")

# 2. Cash vs Control only (excluding work)
df_cc = df[df['b_treat_work'] == 0]
result = run_ols_spec(
    df_cc, main_outcome,
    treatment_vars=['b_treat_largecash'],
    controls=[main_baseline, 'female'],
    fe_vars=['camp_code', 'enum_code'],
    cluster_var='block',
    spec_id='robust/treatment/cash_vs_control',
    spec_tree_path='robustness/measurement.md',
    sample_desc='Cash vs Control (exclude work)'
)
if result:
    results.append(result)
    print(f"  Cash vs Control: coef={result['coefficient']:.4f}, n={result['n_obs']}")

# 3. Work vs Cash only (excluding control)
df_wcash = df[df['b_treat_smallcash'] == 0]
result = run_ols_spec(
    df_wcash, main_outcome,
    treatment_vars=['b_treat_work'],
    controls=[main_baseline, 'female'],
    fe_vars=['camp_code', 'enum_code'],
    cluster_var='block',
    spec_id='robust/treatment/work_vs_cash',
    spec_tree_path='robustness/measurement.md',
    sample_desc='Work vs Cash (exclude control)'
)
if result:
    results.append(result)
    print(f"  Work vs Cash: coef={result['coefficient']:.4f}, n={result['n_obs']}")

# 4. Any treatment vs control
df['any_treatment'] = (df['b_treat_work'] == 1) | (df['b_treat_largecash'] == 1)
df['any_treatment'] = df['any_treatment'].astype(int)
result = run_ols_spec(
    df, main_outcome,
    treatment_vars=['any_treatment'],
    controls=[main_baseline, 'female'],
    fe_vars=['camp_code', 'enum_code'],
    cluster_var='block',
    spec_id='robust/treatment/any_vs_control',
    spec_tree_path='robustness/measurement.md',
    sample_desc='Any treatment vs control'
)
if result:
    results.append(result)
    print(f"  Any treatment: coef={result['coefficient']:.4f}")

print(f"\nTreatment variation specs: {len(results)}")

# =============================================================================
# ADDITIONAL ROBUSTNESS - ALL OUTCOMES WITH FULL CONTROLS
# =============================================================================

print("\n" + "="*80)
print("ALL OUTCOMES WITH FULL CONTROLS")
print("="*80)

for outcome in MENTAL_OUTCOMES:
    baseline_out = f'b_{outcome}'
    controls = [baseline_out] + POSSIBLE_CONTROLS if baseline_out in df.columns else POSSIBLE_CONTROLS

    result = run_ols_spec(
        df, outcome,
        treatment_vars=['b_treat_work', 'b_treat_largecash'],
        controls=controls,
        fe_vars=['camp_code', 'enum_code'],
        cluster_var='block',
        spec_id=f'robust/controls/full_{outcome}',
        spec_tree_path='robustness/control_progression.md',
        sample_desc='Full controls'
    )
    if result:
        results.append(result)
        print(f"  {outcome} (full): coef={result['coefficient']:.4f}")

print(f"\nFull control specs: {len(results)}")

# =============================================================================
# SUMMARY AND OUTPUT
# =============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

# Convert to DataFrame
results_df = pd.DataFrame(results)
print(f"\nTotal specifications run: {len(results_df)}")

# Summary statistics for main treatment (work)
work_results = results_df[results_df['treatment_var'] == 'b_treat_work']
print(f"\nWork treatment coefficient summary:")
print(f"  Mean: {work_results['coefficient'].mean():.4f}")
print(f"  Median: {work_results['coefficient'].median():.4f}")
print(f"  Min: {work_results['coefficient'].min():.4f}")
print(f"  Max: {work_results['coefficient'].max():.4f}")
print(f"  Std: {work_results['coefficient'].std():.4f}")

# Significance rates
print(f"\nSignificance rates:")
print(f"  Positive coefficients: {(work_results['coefficient'] > 0).mean()*100:.1f}%")
print(f"  Significant at 5%: {(work_results['p_value'] < 0.05).mean()*100:.1f}%")
print(f"  Significant at 1%: {(work_results['p_value'] < 0.01).mean()*100:.1f}%")

# Save results
output_file = f"{OUTPUT_PATH}/specification_results.csv"
results_df.to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")

# Create specification breakdown
print("\n" + "="*80)
print("SPECIFICATION BREAKDOWN BY CATEGORY")
print("="*80)

# Parse spec_id to get category
def get_category(spec_id):
    if spec_id.startswith('baseline'):
        return 'Baseline'
    elif 'build' in spec_id or 'control' in spec_id:
        return 'Control variations'
    elif 'sample' in spec_id:
        return 'Sample restrictions'
    elif 'outcome' in spec_id:
        return 'Alternative outcomes'
    elif 'treatment' in spec_id or 'controls/full' in spec_id:
        return 'Treatment/control variations'
    elif 'cluster' in spec_id or 'estimation' in spec_id:
        return 'Inference variations'
    elif 'het' in spec_id:
        return 'Heterogeneity'
    elif 'placebo' in spec_id:
        return 'Placebo tests'
    else:
        return 'Other'

results_df['category'] = results_df['spec_id'].apply(get_category)

category_summary = results_df.groupby('category').agg({
    'coefficient': ['count', lambda x: (x > 0).mean()*100],
    'p_value': [lambda x: (x < 0.05).mean()*100, lambda x: (x < 0.01).mean()*100]
}).round(1)
category_summary.columns = ['N', '% Positive', '% Sig 5%', '% Sig 1%']
print(category_summary)

print(f"\n{'='*80}")
print(f"SPECIFICATION SEARCH COMPLETE: {len(results_df)} specifications")
print(f"{'='*80}")
