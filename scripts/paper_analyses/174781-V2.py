"""
Specification Search for Paper 174781-V2
"Work and Mental Health among Refugees"

This paper studies the effects of work vs cash transfers on mental health outcomes
among Rohingya refugees in Bangladesh using a randomized controlled trial (RCT).

Method: Cross-sectional OLS (RCT design)
Primary outcome: Psychosocial Index (mental_health_index)
Treatment variables: b_treat_work, b_treat_largecash (vs control/small cash)
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

# Set paths
BASE_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/174781-V2"
DATA_PATH = f"{BASE_PATH}/replication_AER/data/3.Endline/3_processed/endline_processed.dta"
OUTPUT_PATH = f"{BASE_PATH}/specification_results.csv"

# Paper metadata
PAPER_ID = "174781-V2"
JOURNAL = "AER"
PAPER_TITLE = "Work and Mental Health among Rohingya Refugees"

# Load data
print("Loading data...")
df = pd.read_stata(DATA_PATH)
print(f"Data loaded: {len(df)} observations, {len(df.columns)} variables")

# Key variables from the paper
# Treatment variables (binary indicators)
TREATMENT_VARS = ['b_treat_work', 'b_treat_largecash']
TREATMENT_MAIN = 'b_treat_work'  # Main treatment of interest (Work vs Control)

# Primary outcome
PRIMARY_OUTCOME = 'mental_health_index'

# Secondary outcomes from the paper
MENTAL_OUTCOMES = ['phq_sd_scale', 'stress_index', 'life_satisfaction',
                   'sociability_a', 'selfworth_index', 'control_index', 'stability']

MISC_OUTCOMES = ['health_a', 'cog_index', 'risk_choice', 'time_choice']

# Fixed effects (camp and enumerator)
FE_VARS = ['b_campname', 'enumname']

# Baseline controls the paper uses in pdslasso
BASELINE_CONTROLS = ['b_resp_gender', 'age', 'marry_dum', 'b_member_count',
                     'education_formal', 'education_religious', 'b_hh_head']

# Cluster variable
CLUSTER_VAR = 'b_block'

# Check data
print("\nChecking key variables...")
for var in [PRIMARY_OUTCOME] + TREATMENT_VARS + FE_VARS + [CLUSTER_VAR]:
    if var in df.columns:
        print(f"  {var}: N={df[var].notna().sum()}, unique={df[var].nunique()}")
    else:
        print(f"  {var}: NOT FOUND")

# Check which baseline controls exist
available_controls = [c for c in BASELINE_CONTROLS if c in df.columns]
print(f"\nAvailable baseline controls: {available_controls}")

# Clean data - drop observations with missing outcome or treatments
df_analysis = df.dropna(subset=[PRIMARY_OUTCOME, 'b_treat_work', 'b_treat_largecash', 'b_campname', 'enumname']).copy()
print(f"\nAnalysis sample: {len(df_analysis)} observations")

def extract_results(model, outcome_var, treatment_var, spec_id, spec_tree_path,
                    sample_desc="Full sample", fe_desc="None", controls_desc="None",
                    cluster_var=None, model_type="OLS"):
    """Extract results from pyfixest model to standard format"""

    try:
        # Use tidy() which gives a nice DataFrame
        tidy_df = model.tidy()

        if treatment_var not in tidy_df.index:
            print(f"  Treatment var {treatment_var} not in model results")
            return None

        row = tidy_df.loc[treatment_var]
        coef = float(row['Estimate'])
        se = float(row['Std. Error'])
        tstat = float(row['t value'])
        pval = float(row['Pr(>|t|)'])
        ci_lower = float(row['2.5%'])
        ci_upper = float(row['97.5%'])

        # Get all coefficients for coefficient_vector_json
        all_coefs = {}
        for var in tidy_df.index:
            all_coefs[var] = {
                "coef": float(tidy_df.loc[var, 'Estimate']),
                "se": float(tidy_df.loc[var, 'Std. Error']),
                "pval": float(tidy_df.loc[var, 'Pr(>|t|)'])
            }

        # Get R-squared from model attributes if available
        r2 = None
        if hasattr(model, '_r2'):
            r2 = float(model._r2)
        elif hasattr(model, 'r2'):
            try:
                r2 = float(model.r2)
            except:
                pass

        coef_vector = {
            "treatment": {"var": treatment_var, "coef": coef, "se": se, "pval": pval},
            "all_coefficients": all_coefs,
            "n_obs": int(model._N),
            "r_squared": r2
        }

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
            't_stat': tstat,
            'p_value': pval,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_obs': int(model._N),
            'r_squared': r2,
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc,
            'fixed_effects': fe_desc,
            'controls_desc': controls_desc,
            'cluster_var': cluster_var,
            'model_type': model_type,
            'estimation_script': f"spec_search_{PAPER_ID}.py"
        }
    except Exception as e:
        print(f"Error extracting results for {spec_id}: {e}")
        return None

# Store results
results = []

# ============================================================================
# BASELINE SPECIFICATION
# ============================================================================
print("\n" + "="*70)
print("Running baseline specification (exact replication)")
print("="*70)

# The paper uses pdslasso with camp and enumerator FE, clustered at block level
# Baseline specification from Table 2
try:
    # Baseline with camp and enumerator FE, clustered by block
    formula = f"{PRIMARY_OUTCOME} ~ b_treat_work + b_treat_largecash | b_campname + enumname"
    baseline_model = pf.feols(formula, data=df_analysis, vcov={'CRV1': CLUSTER_VAR})

    # Work treatment effect
    result = extract_results(
        baseline_model,
        PRIMARY_OUTCOME,
        'b_treat_work',
        spec_id='baseline',
        spec_tree_path='methods/cross_sectional_ols.md#baseline',
        fe_desc='camp + enumerator',
        cluster_var=CLUSTER_VAR,
        model_type='FE-OLS'
    )
    if result:
        results.append(result)
        print(f"Baseline (Work): coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

    # Cash treatment effect
    result_cash = extract_results(
        baseline_model,
        PRIMARY_OUTCOME,
        'b_treat_largecash',
        spec_id='baseline_cash',
        spec_tree_path='methods/cross_sectional_ols.md#baseline',
        fe_desc='camp + enumerator',
        cluster_var=CLUSTER_VAR,
        model_type='FE-OLS'
    )
    if result_cash:
        results.append(result_cash)
        print(f"Baseline (Cash): coef={result_cash['coefficient']:.4f}, se={result_cash['std_error']:.4f}, p={result_cash['p_value']:.4f}")

except Exception as e:
    print(f"Error in baseline: {e}")

# ============================================================================
# OLS METHOD VARIATIONS
# ============================================================================
print("\n" + "="*70)
print("Running OLS method variations")
print("="*70)

# 1. No fixed effects
try:
    formula = f"{PRIMARY_OUTCOME} ~ b_treat_work + b_treat_largecash"
    model = pf.feols(formula, data=df_analysis, vcov={'CRV1': CLUSTER_VAR})
    result = extract_results(model, PRIMARY_OUTCOME, 'b_treat_work',
        spec_id='ols/fe/none', spec_tree_path='methods/cross_sectional_ols.md#fixed-effects',
        fe_desc='none', cluster_var=CLUSTER_VAR)
    if result:
        results.append(result)
        print(f"No FE: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"Error: {e}")

# 2. Camp FE only
try:
    formula = f"{PRIMARY_OUTCOME} ~ b_treat_work + b_treat_largecash | b_campname"
    model = pf.feols(formula, data=df_analysis, vcov={'CRV1': CLUSTER_VAR})
    result = extract_results(model, PRIMARY_OUTCOME, 'b_treat_work',
        spec_id='ols/fe/camp', spec_tree_path='methods/cross_sectional_ols.md#fixed-effects',
        fe_desc='camp', cluster_var=CLUSTER_VAR)
    if result:
        results.append(result)
        print(f"Camp FE only: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"Error: {e}")

# ============================================================================
# STANDARD ERROR VARIATIONS
# ============================================================================
print("\n" + "="*70)
print("Running SE variations")
print("="*70)

# Heteroskedasticity robust (no clustering)
try:
    formula = f"{PRIMARY_OUTCOME} ~ b_treat_work + b_treat_largecash | b_campname + enumname"
    model = pf.feols(formula, data=df_analysis, vcov='hetero')
    result = extract_results(model, PRIMARY_OUTCOME, 'b_treat_work',
        spec_id='ols/se/robust', spec_tree_path='methods/cross_sectional_ols.md#standard-errors',
        fe_desc='camp + enumerator', cluster_var='none (robust)', model_type='FE-OLS')
    if result:
        results.append(result)
        print(f"Robust SE: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}")
except Exception as e:
    print(f"Error: {e}")

# ============================================================================
# CONTROL VARIATIONS
# ============================================================================
print("\n" + "="*70)
print("Running control variations")
print("="*70)

# Find available baseline controls
actual_controls = []
for c in BASELINE_CONTROLS:
    if c in df_analysis.columns and df_analysis[c].notna().sum() > 100:
        actual_controls.append(c)
print(f"Available controls: {actual_controls}")

# 1. No controls (bivariate with FE)
try:
    formula = f"{PRIMARY_OUTCOME} ~ b_treat_work + b_treat_largecash | b_campname + enumname"
    model = pf.feols(formula, data=df_analysis, vcov={'CRV1': CLUSTER_VAR})
    result = extract_results(model, PRIMARY_OUTCOME, 'b_treat_work',
        spec_id='ols/controls/none', spec_tree_path='methods/cross_sectional_ols.md#control-sets',
        fe_desc='camp + enumerator', controls_desc='none', cluster_var=CLUSTER_VAR)
    if result:
        results.append(result)
        print(f"No controls: coef={result['coefficient']:.4f}")
except Exception as e:
    print(f"Error: {e}")

# 2. Demographics only
demographics = ['b_resp_gender', 'age', 'marry_dum']
demographics_avail = [d for d in demographics if d in df_analysis.columns]
if demographics_avail:
    try:
        ctrl_str = ' + '.join(demographics_avail)
        formula = f"{PRIMARY_OUTCOME} ~ b_treat_work + b_treat_largecash + {ctrl_str} | b_campname + enumname"
        model = pf.feols(formula, data=df_analysis.dropna(subset=demographics_avail), vcov={'CRV1': CLUSTER_VAR})
        result = extract_results(model, PRIMARY_OUTCOME, 'b_treat_work',
            spec_id='ols/controls/demographics', spec_tree_path='methods/cross_sectional_ols.md#control-sets',
            fe_desc='camp + enumerator', controls_desc='demographics', cluster_var=CLUSTER_VAR)
        if result:
            results.append(result)
            print(f"Demographics: coef={result['coefficient']:.4f}")
    except Exception as e:
        print(f"Error: {e}")

# 3. Full controls
if actual_controls:
    try:
        df_temp = df_analysis.dropna(subset=actual_controls)
        ctrl_str = ' + '.join(actual_controls)
        formula = f"{PRIMARY_OUTCOME} ~ b_treat_work + b_treat_largecash + {ctrl_str} | b_campname + enumname"
        model = pf.feols(formula, data=df_temp, vcov={'CRV1': CLUSTER_VAR})
        result = extract_results(model, PRIMARY_OUTCOME, 'b_treat_work',
            spec_id='ols/controls/full', spec_tree_path='methods/cross_sectional_ols.md#control-sets',
            fe_desc='camp + enumerator', controls_desc='full baseline controls', cluster_var=CLUSTER_VAR)
        if result:
            results.append(result)
            print(f"Full controls: coef={result['coefficient']:.4f}")
    except Exception as e:
        print(f"Error: {e}")

# ============================================================================
# SAMPLE RESTRICTIONS
# ============================================================================
print("\n" + "="*70)
print("Running sample restrictions")
print("="*70)

# 1. Male subsample
try:
    df_male = df_analysis[df_analysis['b_resp_gender'] == 0].copy()
    if len(df_male) > 50:
        formula = f"{PRIMARY_OUTCOME} ~ b_treat_work + b_treat_largecash | b_campname + enumname"
        model = pf.feols(formula, data=df_male, vcov={'CRV1': CLUSTER_VAR})
        result = extract_results(model, PRIMARY_OUTCOME, 'b_treat_work',
            spec_id='ols/sample/subgroup_male', spec_tree_path='robustness/sample_restrictions.md',
            sample_desc='Male only', fe_desc='camp + enumerator', cluster_var=CLUSTER_VAR)
        if result:
            results.append(result)
            print(f"Male only: coef={result['coefficient']:.4f}, n={result['n_obs']}")
except Exception as e:
    print(f"Error: {e}")

# 2. Female subsample
try:
    df_female = df_analysis[df_analysis['b_resp_gender'] == 1].copy()
    if len(df_female) > 50:
        formula = f"{PRIMARY_OUTCOME} ~ b_treat_work + b_treat_largecash | b_campname + enumname"
        model = pf.feols(formula, data=df_female, vcov={'CRV1': CLUSTER_VAR})
        result = extract_results(model, PRIMARY_OUTCOME, 'b_treat_work',
            spec_id='ols/sample/subgroup_female', spec_tree_path='robustness/sample_restrictions.md',
            sample_desc='Female only', fe_desc='camp + enumerator', cluster_var=CLUSTER_VAR)
        if result:
            results.append(result)
            print(f"Female only: coef={result['coefficient']:.4f}, n={result['n_obs']}")
except Exception as e:
    print(f"Error: {e}")

# 3. Trimmed sample (1% on outcome)
try:
    q01 = df_analysis[PRIMARY_OUTCOME].quantile(0.01)
    q99 = df_analysis[PRIMARY_OUTCOME].quantile(0.99)
    df_trimmed = df_analysis[(df_analysis[PRIMARY_OUTCOME] > q01) & (df_analysis[PRIMARY_OUTCOME] < q99)].copy()
    if len(df_trimmed) > 50:
        formula = f"{PRIMARY_OUTCOME} ~ b_treat_work + b_treat_largecash | b_campname + enumname"
        model = pf.feols(formula, data=df_trimmed, vcov={'CRV1': CLUSTER_VAR})
        result = extract_results(model, PRIMARY_OUTCOME, 'b_treat_work',
            spec_id='robust/sample/trim_1pct', spec_tree_path='robustness/sample_restrictions.md',
            sample_desc='Trimmed 1%/99%', fe_desc='camp + enumerator', cluster_var=CLUSTER_VAR)
        if result:
            results.append(result)
            print(f"Trimmed 1%: coef={result['coefficient']:.4f}, n={result['n_obs']}")
except Exception as e:
    print(f"Error: {e}")

# 4. Excluding each camp (leave-one-out by camp)
camps = df_analysis['b_campname'].dropna().unique()
for camp in camps:
    try:
        df_excl = df_analysis[df_analysis['b_campname'] != camp].copy()
        if len(df_excl) > 50:
            formula = f"{PRIMARY_OUTCOME} ~ b_treat_work + b_treat_largecash | b_campname + enumname"
            model = pf.feols(formula, data=df_excl, vcov={'CRV1': CLUSTER_VAR})
            result = extract_results(model, PRIMARY_OUTCOME, 'b_treat_work',
                spec_id=f'robust/sample/exclude_camp_{int(camp)}',
                spec_tree_path='robustness/sample_restrictions.md',
                sample_desc=f'Excluding camp {int(camp)}', fe_desc='camp + enumerator', cluster_var=CLUSTER_VAR)
            if result:
                results.append(result)
                print(f"Exclude camp {int(camp)}: coef={result['coefficient']:.4f}, n={result['n_obs']}")
    except Exception as e:
        print(f"Error excluding camp {camp}: {e}")

# ============================================================================
# CLUSTERING VARIATIONS
# ============================================================================
print("\n" + "="*70)
print("Running clustering variations")
print("="*70)

# 1. No clustering (robust only)
try:
    formula = f"{PRIMARY_OUTCOME} ~ b_treat_work + b_treat_largecash | b_campname + enumname"
    model = pf.feols(formula, data=df_analysis, vcov='hetero')
    result = extract_results(model, PRIMARY_OUTCOME, 'b_treat_work',
        spec_id='robust/cluster/none', spec_tree_path='robustness/clustering_variations.md#single-level-clustering',
        fe_desc='camp + enumerator', cluster_var='none (robust)')
    if result:
        results.append(result)
        print(f"No clustering: se={result['std_error']:.4f}")
except Exception as e:
    print(f"Error: {e}")

# 2. Cluster by camp
try:
    formula = f"{PRIMARY_OUTCOME} ~ b_treat_work + b_treat_largecash | b_campname + enumname"
    model = pf.feols(formula, data=df_analysis, vcov={'CRV1': 'b_campname'})
    result = extract_results(model, PRIMARY_OUTCOME, 'b_treat_work',
        spec_id='robust/cluster/camp', spec_tree_path='robustness/clustering_variations.md#single-level-clustering',
        fe_desc='camp + enumerator', cluster_var='camp')
    if result:
        results.append(result)
        print(f"Cluster by camp: se={result['std_error']:.4f}")
except Exception as e:
    print(f"Error: {e}")

# 3. Cluster by enumerator
try:
    formula = f"{PRIMARY_OUTCOME} ~ b_treat_work + b_treat_largecash | b_campname + enumname"
    model = pf.feols(formula, data=df_analysis, vcov={'CRV1': 'enumname'})
    result = extract_results(model, PRIMARY_OUTCOME, 'b_treat_work',
        spec_id='robust/cluster/enumerator', spec_tree_path='robustness/clustering_variations.md#single-level-clustering',
        fe_desc='camp + enumerator', cluster_var='enumerator')
    if result:
        results.append(result)
        print(f"Cluster by enumerator: se={result['std_error']:.4f}")
except Exception as e:
    print(f"Error: {e}")

# ============================================================================
# LEAVE-ONE-OUT (CONTROLS)
# ============================================================================
print("\n" + "="*70)
print("Running leave-one-out for controls")
print("="*70)

if actual_controls:
    for drop_ctrl in actual_controls:
        try:
            remaining = [c for c in actual_controls if c != drop_ctrl]
            df_temp = df_analysis.dropna(subset=remaining) if remaining else df_analysis.copy()
            if remaining:
                ctrl_str = ' + '.join(remaining)
                formula = f"{PRIMARY_OUTCOME} ~ b_treat_work + b_treat_largecash + {ctrl_str} | b_campname + enumname"
            else:
                formula = f"{PRIMARY_OUTCOME} ~ b_treat_work + b_treat_largecash | b_campname + enumname"
            model = pf.feols(formula, data=df_temp, vcov={'CRV1': CLUSTER_VAR})
            result = extract_results(model, PRIMARY_OUTCOME, 'b_treat_work',
                spec_id=f'robust/loo/drop_{drop_ctrl}', spec_tree_path='robustness/leave_one_out.md',
                fe_desc='camp + enumerator', controls_desc=f'all except {drop_ctrl}', cluster_var=CLUSTER_VAR)
            if result:
                results.append(result)
                print(f"Drop {drop_ctrl}: coef={result['coefficient']:.4f}")
        except Exception as e:
            print(f"Error dropping {drop_ctrl}: {e}")

# ============================================================================
# SINGLE COVARIATE
# ============================================================================
print("\n" + "="*70)
print("Running single covariate specifications")
print("="*70)

# Bivariate (no controls)
try:
    formula = f"{PRIMARY_OUTCOME} ~ b_treat_work + b_treat_largecash | b_campname + enumname"
    model = pf.feols(formula, data=df_analysis, vcov={'CRV1': CLUSTER_VAR})
    result = extract_results(model, PRIMARY_OUTCOME, 'b_treat_work',
        spec_id='robust/single/none', spec_tree_path='robustness/single_covariate.md',
        fe_desc='camp + enumerator', controls_desc='none', cluster_var=CLUSTER_VAR)
    if result:
        results.append(result)
        print(f"Bivariate: coef={result['coefficient']:.4f}")
except Exception as e:
    print(f"Error: {e}")

# Single control additions
for ctrl in actual_controls:
    try:
        df_temp = df_analysis.dropna(subset=[ctrl])
        formula = f"{PRIMARY_OUTCOME} ~ b_treat_work + b_treat_largecash + {ctrl} | b_campname + enumname"
        model = pf.feols(formula, data=df_temp, vcov={'CRV1': CLUSTER_VAR})
        result = extract_results(model, PRIMARY_OUTCOME, 'b_treat_work',
            spec_id=f'robust/single/{ctrl}', spec_tree_path='robustness/single_covariate.md',
            fe_desc='camp + enumerator', controls_desc=f'{ctrl} only', cluster_var=CLUSTER_VAR)
        if result:
            results.append(result)
            print(f"+ {ctrl}: coef={result['coefficient']:.4f}")
    except Exception as e:
        print(f"Error adding {ctrl}: {e}")

# ============================================================================
# SECONDARY OUTCOMES
# ============================================================================
print("\n" + "="*70)
print("Running secondary outcomes")
print("="*70)

for outcome in MENTAL_OUTCOMES + MISC_OUTCOMES:
    if outcome not in df_analysis.columns:
        print(f"  {outcome}: NOT FOUND")
        continue

    df_temp = df_analysis.dropna(subset=[outcome]).copy()
    if len(df_temp) < 100:
        print(f"  {outcome}: insufficient observations")
        continue

    try:
        formula = f"{outcome} ~ b_treat_work + b_treat_largecash | b_campname + enumname"
        model = pf.feols(formula, data=df_temp, vcov={'CRV1': CLUSTER_VAR})
        result = extract_results(model, outcome, 'b_treat_work',
            spec_id=f'outcome/{outcome}', spec_tree_path='methods/cross_sectional_ols.md',
            fe_desc='camp + enumerator', cluster_var=CLUSTER_VAR)
        if result:
            results.append(result)
            print(f"{outcome}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
    except Exception as e:
        print(f"Error for {outcome}: {e}")

# ============================================================================
# ADDITIONAL HETEROGENEITY (FROM PAPER)
# ============================================================================
print("\n" + "="*70)
print("Running additional heterogeneity specifications")
print("="*70)

# Interaction with baseline depression
if 'b_depressed' in df_analysis.columns:
    try:
        # Create interaction
        df_temp = df_analysis.copy()
        df_temp['work_depressed'] = df_temp['b_treat_work'] * df_temp['b_depressed']
        formula = f"{PRIMARY_OUTCOME} ~ b_treat_work + b_treat_largecash + b_depressed + work_depressed | b_campname + enumname"
        model = pf.feols(formula, data=df_temp.dropna(subset=['b_depressed']), vcov={'CRV1': CLUSTER_VAR})
        result = extract_results(model, PRIMARY_OUTCOME, 'b_treat_work',
            spec_id='ols/interact/depression', spec_tree_path='methods/cross_sectional_ols.md#interaction-effects',
            fe_desc='camp + enumerator', controls_desc='interaction with baseline depression', cluster_var=CLUSTER_VAR)
        if result:
            results.append(result)
            print(f"Work x Depression: coef={result['coefficient']:.4f}")
    except Exception as e:
        print(f"Error: {e}")

# Interaction with violence exposure
if 'b_any_killed' in df_analysis.columns:
    try:
        df_temp = df_analysis.copy()
        df_temp['work_violence'] = df_temp['b_treat_work'] * df_temp['b_any_killed']
        formula = f"{PRIMARY_OUTCOME} ~ b_treat_work + b_treat_largecash + b_any_killed + work_violence | b_campname + enumname"
        model = pf.feols(formula, data=df_temp.dropna(subset=['b_any_killed']), vcov={'CRV1': CLUSTER_VAR})
        result = extract_results(model, PRIMARY_OUTCOME, 'b_treat_work',
            spec_id='ols/interact/violence', spec_tree_path='methods/cross_sectional_ols.md#interaction-effects',
            fe_desc='camp + enumerator', controls_desc='interaction with violence exposure', cluster_var=CLUSTER_VAR)
        if result:
            results.append(result)
            print(f"Work x Violence: coef={result['coefficient']:.4f}")
    except Exception as e:
        print(f"Error: {e}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*70)
print("Saving results")
print("="*70)

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved {len(results_df)} specifications to {OUTPUT_PATH}")

# Summary statistics
if len(results_df) > 0:
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"Total specifications: {len(results_df)}")
    print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({(results_df['coefficient'] > 0).mean()*100:.1f}%)")
    print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({(results_df['p_value'] < 0.05).mean()*100:.1f}%)")
    print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({(results_df['p_value'] < 0.01).mean()*100:.1f}%)")
    print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
    print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")
    print(f"Range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")

    # Print results by category
    print("\n" + "="*70)
    print("RESULTS BY CATEGORY")
    print("="*70)

    for spec_type in ['baseline', 'ols/', 'robust/', 'outcome/']:
        subset = results_df[results_df['spec_id'].str.startswith(spec_type)]
        if len(subset) > 0:
            print(f"\n{spec_type}:")
            print(f"  N: {len(subset)}")
            print(f"  Sig at 5%: {(subset['p_value'] < 0.05).sum()} ({(subset['p_value'] < 0.05).mean()*100:.1f}%)")
            print(f"  Mean coef: {subset['coefficient'].mean():.4f}")
else:
    print("No results generated!")
