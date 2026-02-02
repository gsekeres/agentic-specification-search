#!/usr/bin/env python3
"""
Specification Search: 173341-V1
Paper: "Vulnerability and Clientelism" by Bobonis, Gertler, Gonzalez-Navarro and Nichter (2022)
Journal: American Economic Review (AER)

This script conducts a systematic specification search following the i4r methodology.

Method Classification: Panel Fixed Effects / Cross-sectional OLS with Municipality FE
The paper studies the effect of a cisterns treatment (RCT) and rainfall shocks on:
- Clientelistic requests to politicians
- Vulnerability outcomes
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Paths
BASE_PATH = Path('/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search')
DATA_PATH = BASE_PATH / 'data/downloads/extracted/173341-V1/data/final_data'
OUTPUT_PATH = BASE_PATH / 'data/downloads/extracted/173341-V1'

# Paper metadata
PAPER_ID = '173341-V1'
JOURNAL = 'AER'
PAPER_TITLE = 'Vulnerability and Clientelism'

# Results storage
results = []

def add_result(spec_id, spec_tree_path, outcome_var, treatment_var, coef, se, t_stat, p_value,
               ci_lower, ci_upper, n_obs, r_squared, coef_vector_json, sample_desc,
               fixed_effects, controls_desc, cluster_var, model_type):
    """Add a specification result to the results list."""
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
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_obs': n_obs,
        'r_squared': r_squared,
        'coefficient_vector_json': json.dumps(coef_vector_json) if isinstance(coef_vector_json, dict) else coef_vector_json,
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })

def run_regression(df, formula, cluster_var=None, treatment_var='treatment'):
    """
    Run OLS regression with formula interface.
    """
    try:
        df_reg = df.dropna(subset=df.columns[df.columns.isin(
            [c.strip() for c in formula.replace('~', '+').replace('C(', ' ').replace(')', ' ').split('+')]
        )]).copy()

        if len(df_reg) < 50:
            return None

        if cluster_var and cluster_var in df_reg.columns:
            model = smf.ols(formula, data=df_reg).fit(
                cov_type='cluster',
                cov_kwds={'groups': df_reg[cluster_var].astype(int).values}
            )
        else:
            model = smf.ols(formula, data=df_reg).fit(cov_type='HC1')

        if treatment_var not in model.params.index:
            # Try with brackets for categorical vars
            if f'C({treatment_var})' in str(model.params.index):
                return None
            return None

        coef = model.params[treatment_var]
        se = model.bse[treatment_var]
        t_stat = model.tvalues[treatment_var]
        p_val = model.pvalues[treatment_var]
        ci = model.conf_int().loc[treatment_var]

        return {
            'coef': float(coef),
            'se': float(se),
            't_stat': float(t_stat),
            'p_value': float(p_val),
            'ci_lower': float(ci[0]),
            'ci_upper': float(ci[1]),
            'n_obs': int(model.nobs),
            'r_squared': float(model.rsquared),
            'coef_vector': {
                'treatment': {'var': treatment_var, 'coef': float(coef), 'se': float(se), 'pval': float(p_val)},
                'n_obs': int(model.nobs),
                'r_squared': float(model.rsquared)
            }
        }
    except Exception as e:
        print(f"    Regression error: {e}")
        return None

# =============================================================================
# Load Data
# =============================================================================
print("Loading data...")

# Load the three main datasets
df_stacked = pd.read_stata(DATA_PATH / 'clientelism_individual_data_stacked.dta')
df_household = pd.read_stata(DATA_PATH / 'clientelism_household_data.dta')
df_individual = pd.read_stata(DATA_PATH / 'clientelism_individual_data.dta')

# Convert columns to numeric to avoid dtype issues
for col in ['treatment', 'year2012', 'year2013', 'mun_id', 'b_clusters', 'frequent_interactor']:
    if col in df_stacked.columns:
        df_stacked[col] = pd.to_numeric(df_stacked[col], errors='coerce')
    if col in df_household.columns:
        df_household[col] = pd.to_numeric(df_household[col], errors='coerce')
    if col in df_individual.columns:
        df_individual[col] = pd.to_numeric(df_individual[col], errors='coerce')

print(f"Stacked data: {len(df_stacked)} observations")
print(f"Household data: {len(df_household)} observations")
print(f"Individual data: {len(df_individual)} observations")

# Create interaction variables for stacked data (Table 3 specification)
df_stacked['treat_rain_std'] = df_stacked['treatment'] * df_stacked['rainfall_std_stacked']
df_stacked['treat_freq'] = df_stacked['treatment'] * df_stacked['frequent_interactor']
df_stacked['rain_std_freq'] = df_stacked['rainfall_std_stacked'] * df_stacked['frequent_interactor']

# =============================================================================
# SPECIFICATION SEARCH
# =============================================================================

print("\n" + "="*80)
print("RUNNING SPECIFICATION SEARCH")
print("="*80)

spec_count = 0

# =============================================================================
# 1. BASELINE SPECIFICATIONS (Table 3 Replication)
# =============================================================================
print("\n1. Running baseline specifications...")

# Table 3 Column 3: Effect of Cisterns and Rainfall on Private Requests with Mun FE
result = run_regression(
    df_stacked,
    'ask_private_stacked ~ treatment + rainfall_std_stacked + year2012 + C(mun_id)',
    cluster_var='b_clusters',
    treatment_var='treatment'
)
if result:
    spec_count += 1
    add_result(
        spec_id='baseline',
        spec_tree_path='methods/panel_fixed_effects.md#baseline',
        outcome_var='ask_private_stacked',
        treatment_var='treatment',
        coef=result['coef'],
        se=result['se'],
        t_stat=result['t_stat'],
        p_value=result['p_value'],
        ci_lower=result['ci_lower'],
        ci_upper=result['ci_upper'],
        n_obs=result['n_obs'],
        r_squared=result['r_squared'],
        coef_vector_json=result['coef_vector'],
        sample_desc='Full stacked sample (2012+2013)',
        fixed_effects='Municipality + Year',
        controls_desc='Rainfall shock',
        cluster_var='b_clusters (neighborhood)',
        model_type='OLS with FE'
    )
    print(f"  Baseline: coef={result['coef']:.4f}, se={result['se']:.4f}, p={result['p_value']:.4f}")

# Rainfall effect
result_rain = run_regression(
    df_stacked,
    'ask_private_stacked ~ treatment + rainfall_std_stacked + year2012 + C(mun_id)',
    cluster_var='b_clusters',
    treatment_var='rainfall_std_stacked'
)
if result_rain:
    spec_count += 1
    add_result(
        spec_id='baseline_rainfall',
        spec_tree_path='methods/panel_fixed_effects.md#baseline',
        outcome_var='ask_private_stacked',
        treatment_var='rainfall_std_stacked',
        coef=result_rain['coef'],
        se=result_rain['se'],
        t_stat=result_rain['t_stat'],
        p_value=result_rain['p_value'],
        ci_lower=result_rain['ci_lower'],
        ci_upper=result_rain['ci_upper'],
        n_obs=result_rain['n_obs'],
        r_squared=result_rain['r_squared'],
        coef_vector_json=result_rain['coef_vector'],
        sample_desc='Full stacked sample (2012+2013)',
        fixed_effects='Municipality + Year',
        controls_desc='Cisterns treatment',
        cluster_var='b_clusters (neighborhood)',
        model_type='OLS with FE'
    )
    print(f"  Rainfall effect: coef={result_rain['coef']:.4f}, p={result_rain['p_value']:.4f}")

# Table 3 Column 4: With interaction term
result_int = run_regression(
    df_stacked,
    'ask_private_stacked ~ treatment + rainfall_std_stacked + treat_rain_std + year2012 + C(mun_id)',
    cluster_var='b_clusters',
    treatment_var='treat_rain_std'
)
if result_int:
    spec_count += 1
    add_result(
        spec_id='baseline_interaction',
        spec_tree_path='methods/panel_fixed_effects.md#baseline',
        outcome_var='ask_private_stacked',
        treatment_var='treat_rain_std',
        coef=result_int['coef'],
        se=result_int['se'],
        t_stat=result_int['t_stat'],
        p_value=result_int['p_value'],
        ci_lower=result_int['ci_lower'],
        ci_upper=result_int['ci_upper'],
        n_obs=result_int['n_obs'],
        r_squared=result_int['r_squared'],
        coef_vector_json=result_int['coef_vector'],
        sample_desc='Full stacked sample (2012+2013)',
        fixed_effects='Municipality + Year',
        controls_desc='Treatment, Rainfall, Interaction',
        cluster_var='b_clusters (neighborhood)',
        model_type='OLS with FE'
    )
    print(f"  Interaction effect: coef={result_int['coef']:.4f}, p={result_int['p_value']:.4f}")

# =============================================================================
# 2. ALTERNATIVE OUTCOMES
# =============================================================================
print("\n2. Running alternative outcomes...")

outcome_vars = [
    ('ask_private_stacked', 'Request any private good'),
    ('ask_nowater_private_stacked', 'Request private good excluding water'),
    ('askrec_private_stacked', 'Request and receive private good'),
    ('ask_public_stacked', 'Request any public good'),
]

for outcome, desc in outcome_vars:
    if outcome in df_stacked.columns:
        result = run_regression(
            df_stacked,
            f'{outcome} ~ treatment + rainfall_std_stacked + year2012 + C(mun_id)',
            cluster_var='b_clusters',
            treatment_var='treatment'
        )
        if result:
            spec_count += 1
            add_result(
                spec_id=f'robust/outcome/{outcome}',
                spec_tree_path='robustness/measurement.md',
                outcome_var=outcome,
                treatment_var='treatment',
                coef=result['coef'],
                se=result['se'],
                t_stat=result['t_stat'],
                p_value=result['p_value'],
                ci_lower=result['ci_lower'],
                ci_upper=result['ci_upper'],
                n_obs=result['n_obs'],
                r_squared=result['r_squared'],
                coef_vector_json=result['coef_vector'],
                sample_desc='Full stacked sample (2012+2013)',
                fixed_effects='Municipality + Year',
                controls_desc='Rainfall shock',
                cluster_var='b_clusters (neighborhood)',
                model_type='OLS with FE'
            )
            print(f"  {outcome}: coef={result['coef']:.4f}, p={result['p_value']:.4f}")

# =============================================================================
# 3. CONTROL VARIATIONS
# =============================================================================
print("\n3. Running control variations...")

# No rainfall control
result = run_regression(
    df_stacked,
    'ask_private_stacked ~ treatment + year2012 + C(mun_id)',
    cluster_var='b_clusters',
    treatment_var='treatment'
)
if result:
    spec_count += 1
    add_result(
        spec_id='robust/control/no_rainfall',
        spec_tree_path='robustness/leave_one_out.md',
        outcome_var='ask_private_stacked',
        treatment_var='treatment',
        coef=result['coef'],
        se=result['se'],
        t_stat=result['t_stat'],
        p_value=result['p_value'],
        ci_lower=result['ci_lower'],
        ci_upper=result['ci_upper'],
        n_obs=result['n_obs'],
        r_squared=result['r_squared'],
        coef_vector_json=result['coef_vector'],
        sample_desc='Full stacked sample',
        fixed_effects='Municipality + Year',
        controls_desc='None (no rainfall)',
        cluster_var='b_clusters',
        model_type='OLS with FE'
    )
    print(f"  No rainfall control: coef={result['coef']:.4f}, p={result['p_value']:.4f}")

# Only rainfall (no treatment)
result = run_regression(
    df_stacked,
    'ask_private_stacked ~ rainfall_std_stacked + year2012 + C(mun_id)',
    cluster_var='b_clusters',
    treatment_var='rainfall_std_stacked'
)
if result:
    spec_count += 1
    add_result(
        spec_id='robust/control/rainfall_only',
        spec_tree_path='robustness/leave_one_out.md',
        outcome_var='ask_private_stacked',
        treatment_var='rainfall_std_stacked',
        coef=result['coef'],
        se=result['se'],
        t_stat=result['t_stat'],
        p_value=result['p_value'],
        ci_lower=result['ci_lower'],
        ci_upper=result['ci_upper'],
        n_obs=result['n_obs'],
        r_squared=result['r_squared'],
        coef_vector_json=result['coef_vector'],
        sample_desc='Full stacked sample',
        fixed_effects='Municipality + Year',
        controls_desc='Rainfall only',
        cluster_var='b_clusters',
        model_type='OLS with FE'
    )
    print(f"  Rainfall only: coef={result['coef']:.4f}, p={result['p_value']:.4f}")

# Add frequent_interactor as control
result = run_regression(
    df_stacked,
    'ask_private_stacked ~ treatment + rainfall_std_stacked + frequent_interactor + year2012 + C(mun_id)',
    cluster_var='b_clusters',
    treatment_var='treatment'
)
if result:
    spec_count += 1
    add_result(
        spec_id='robust/control/add_frequent_interactor',
        spec_tree_path='robustness/control_progression.md',
        outcome_var='ask_private_stacked',
        treatment_var='treatment',
        coef=result['coef'],
        se=result['se'],
        t_stat=result['t_stat'],
        p_value=result['p_value'],
        ci_lower=result['ci_lower'],
        ci_upper=result['ci_upper'],
        n_obs=result['n_obs'],
        r_squared=result['r_squared'],
        coef_vector_json=result['coef_vector'],
        sample_desc='Full stacked sample',
        fixed_effects='Municipality + Year',
        controls_desc='Rainfall + Frequent interactor',
        cluster_var='b_clusters',
        model_type='OLS with FE'
    )
    print(f"  With frequent_interactor: coef={result['coef']:.4f}, p={result['p_value']:.4f}")

# =============================================================================
# 4. FIXED EFFECTS VARIATIONS
# =============================================================================
print("\n4. Running fixed effects variations...")

# No FE (pooled OLS)
result = run_regression(
    df_stacked,
    'ask_private_stacked ~ treatment + rainfall_std_stacked + year2012',
    cluster_var='b_clusters',
    treatment_var='treatment'
)
if result:
    spec_count += 1
    add_result(
        spec_id='panel/fe/none',
        spec_tree_path='methods/panel_fixed_effects.md#fixed-effects-structure',
        outcome_var='ask_private_stacked',
        treatment_var='treatment',
        coef=result['coef'],
        se=result['se'],
        t_stat=result['t_stat'],
        p_value=result['p_value'],
        ci_lower=result['ci_lower'],
        ci_upper=result['ci_upper'],
        n_obs=result['n_obs'],
        r_squared=result['r_squared'],
        coef_vector_json=result['coef_vector'],
        sample_desc='Full stacked sample',
        fixed_effects='None (pooled)',
        controls_desc='Rainfall + Year',
        cluster_var='b_clusters',
        model_type='Pooled OLS'
    )
    print(f"  No FE: coef={result['coef']:.4f}, p={result['p_value']:.4f}")

# Municipality FE only (no year dummy)
result = run_regression(
    df_stacked,
    'ask_private_stacked ~ treatment + rainfall_std_stacked + C(mun_id)',
    cluster_var='b_clusters',
    treatment_var='treatment'
)
if result:
    spec_count += 1
    add_result(
        spec_id='panel/fe/unit',
        spec_tree_path='methods/panel_fixed_effects.md#fixed-effects-structure',
        outcome_var='ask_private_stacked',
        treatment_var='treatment',
        coef=result['coef'],
        se=result['se'],
        t_stat=result['t_stat'],
        p_value=result['p_value'],
        ci_lower=result['ci_lower'],
        ci_upper=result['ci_upper'],
        n_obs=result['n_obs'],
        r_squared=result['r_squared'],
        coef_vector_json=result['coef_vector'],
        sample_desc='Full stacked sample',
        fixed_effects='Municipality only',
        controls_desc='Rainfall',
        cluster_var='b_clusters',
        model_type='OLS with FE'
    )
    print(f"  Mun FE only: coef={result['coef']:.4f}, p={result['p_value']:.4f}")

# =============================================================================
# 5. CLUSTERING VARIATIONS
# =============================================================================
print("\n5. Running clustering variations...")

# Robust SE (no clustering)
result = run_regression(
    df_stacked,
    'ask_private_stacked ~ treatment + rainfall_std_stacked + year2012 + C(mun_id)',
    cluster_var=None,
    treatment_var='treatment'
)
if result:
    spec_count += 1
    add_result(
        spec_id='robust/cluster/none',
        spec_tree_path='robustness/clustering_variations.md#single-level-clustering',
        outcome_var='ask_private_stacked',
        treatment_var='treatment',
        coef=result['coef'],
        se=result['se'],
        t_stat=result['t_stat'],
        p_value=result['p_value'],
        ci_lower=result['ci_lower'],
        ci_upper=result['ci_upper'],
        n_obs=result['n_obs'],
        r_squared=result['r_squared'],
        coef_vector_json=result['coef_vector'],
        sample_desc='Full stacked sample',
        fixed_effects='Municipality + Year',
        controls_desc='Rainfall',
        cluster_var='None (robust HC1)',
        model_type='OLS with FE'
    )
    print(f"  Robust SE (no cluster): se={result['se']:.4f}")

# Cluster by municipality
result = run_regression(
    df_stacked,
    'ask_private_stacked ~ treatment + rainfall_std_stacked + year2012 + C(mun_id)',
    cluster_var='mun_id',
    treatment_var='treatment'
)
if result:
    spec_count += 1
    add_result(
        spec_id='robust/cluster/mun_id',
        spec_tree_path='robustness/clustering_variations.md#single-level-clustering',
        outcome_var='ask_private_stacked',
        treatment_var='treatment',
        coef=result['coef'],
        se=result['se'],
        t_stat=result['t_stat'],
        p_value=result['p_value'],
        ci_lower=result['ci_lower'],
        ci_upper=result['ci_upper'],
        n_obs=result['n_obs'],
        r_squared=result['r_squared'],
        coef_vector_json=result['coef_vector'],
        sample_desc='Full stacked sample',
        fixed_effects='Municipality + Year',
        controls_desc='Rainfall',
        cluster_var='mun_id (municipality)',
        model_type='OLS with FE'
    )
    print(f"  Cluster by municipality: se={result['se']:.4f}")

# =============================================================================
# 6. SAMPLE RESTRICTIONS - TIME PERIODS
# =============================================================================
print("\n6. Running sample restrictions by time...")

# 2012 only
df_2012 = df_stacked[df_stacked['year2012'] == 1].copy()
result = run_regression(
    df_2012,
    'ask_private_stacked ~ treatment + rainfall_std_stacked + C(mun_id)',
    cluster_var='b_clusters',
    treatment_var='treatment'
)
if result:
    spec_count += 1
    add_result(
        spec_id='robust/sample/year_2012',
        spec_tree_path='robustness/sample_restrictions.md#time-based-restrictions',
        outcome_var='ask_private_stacked',
        treatment_var='treatment',
        coef=result['coef'],
        se=result['se'],
        t_stat=result['t_stat'],
        p_value=result['p_value'],
        ci_lower=result['ci_lower'],
        ci_upper=result['ci_upper'],
        n_obs=result['n_obs'],
        r_squared=result['r_squared'],
        coef_vector_json=result['coef_vector'],
        sample_desc='2012 only',
        fixed_effects='Municipality',
        controls_desc='Rainfall',
        cluster_var='b_clusters',
        model_type='OLS with FE'
    )
    print(f"  2012 only: coef={result['coef']:.4f}, n={result['n_obs']}")

# 2013 only
df_2013 = df_stacked[df_stacked['year2013'] == 1].copy()
result = run_regression(
    df_2013,
    'ask_private_stacked ~ treatment + rainfall_std_stacked + C(mun_id)',
    cluster_var='b_clusters',
    treatment_var='treatment'
)
if result:
    spec_count += 1
    add_result(
        spec_id='robust/sample/year_2013',
        spec_tree_path='robustness/sample_restrictions.md#time-based-restrictions',
        outcome_var='ask_private_stacked',
        treatment_var='treatment',
        coef=result['coef'],
        se=result['se'],
        t_stat=result['t_stat'],
        p_value=result['p_value'],
        ci_lower=result['ci_lower'],
        ci_upper=result['ci_upper'],
        n_obs=result['n_obs'],
        r_squared=result['r_squared'],
        coef_vector_json=result['coef_vector'],
        sample_desc='2013 only',
        fixed_effects='Municipality',
        controls_desc='Rainfall',
        cluster_var='b_clusters',
        model_type='OLS with FE'
    )
    print(f"  2013 only: coef={result['coef']:.4f}, n={result['n_obs']}")

# =============================================================================
# 7. SAMPLE RESTRICTIONS - BY MUNICIPALITY (drop each one)
# =============================================================================
print("\n7. Running sample restrictions by municipality...")

mun_ids = sorted(df_stacked['mun_id'].dropna().unique())
mun_count = 0
for mun in mun_ids:
    df_excl = df_stacked[df_stacked['mun_id'] != mun].copy()
    result = run_regression(
        df_excl,
        'ask_private_stacked ~ treatment + rainfall_std_stacked + year2012 + C(mun_id)',
        cluster_var='b_clusters',
        treatment_var='treatment'
    )
    if result:
        spec_count += 1
        mun_count += 1
        add_result(
            spec_id=f'robust/sample/drop_mun_{int(mun)}',
            spec_tree_path='robustness/sample_restrictions.md#geographic-unit-restrictions',
            outcome_var='ask_private_stacked',
            treatment_var='treatment',
            coef=result['coef'],
            se=result['se'],
            t_stat=result['t_stat'],
            p_value=result['p_value'],
            ci_lower=result['ci_lower'],
            ci_upper=result['ci_upper'],
            n_obs=result['n_obs'],
            r_squared=result['r_squared'],
            coef_vector_json=result['coef_vector'],
            sample_desc=f'Excluding municipality {int(mun)}',
            fixed_effects='Municipality + Year',
            controls_desc='Rainfall',
            cluster_var='b_clusters',
            model_type='OLS with FE'
        )

print(f"  Dropped {mun_count} municipalities individually")

# =============================================================================
# 8. HETEROGENEITY ANALYSIS
# =============================================================================
print("\n8. Running heterogeneity analyses...")

# By clientelist relationship (Table 5 replication)
result = run_regression(
    df_stacked,
    'ask_private_stacked ~ treatment + treat_freq + rainfall_std_stacked + rain_std_freq + frequent_interactor + year2012 + C(mun_id)',
    cluster_var='b_clusters',
    treatment_var='treat_freq'
)
if result:
    spec_count += 1
    add_result(
        spec_id='robust/heterogeneity/frequent_interactor',
        spec_tree_path='robustness/heterogeneity.md',
        outcome_var='ask_private_stacked',
        treatment_var='treat_freq',
        coef=result['coef'],
        se=result['se'],
        t_stat=result['t_stat'],
        p_value=result['p_value'],
        ci_lower=result['ci_lower'],
        ci_upper=result['ci_upper'],
        n_obs=result['n_obs'],
        r_squared=result['r_squared'],
        coef_vector_json=result['coef_vector'],
        sample_desc='Full stacked sample',
        fixed_effects='Municipality + Year',
        controls_desc='Full interaction model',
        cluster_var='b_clusters',
        model_type='OLS with FE'
    )
    print(f"  Clientelist interaction: coef={result['coef']:.4f}, p={result['p_value']:.4f}")

# Subsample: frequent interactors only
df_freq = df_stacked[df_stacked['frequent_interactor'] == 1].copy()
result = run_regression(
    df_freq,
    'ask_private_stacked ~ treatment + rainfall_std_stacked + year2012 + C(mun_id)',
    cluster_var='b_clusters',
    treatment_var='treatment'
)
if result:
    spec_count += 1
    add_result(
        spec_id='robust/sample/frequent_interactors',
        spec_tree_path='robustness/sample_restrictions.md#demographic-subgroups',
        outcome_var='ask_private_stacked',
        treatment_var='treatment',
        coef=result['coef'],
        se=result['se'],
        t_stat=result['t_stat'],
        p_value=result['p_value'],
        ci_lower=result['ci_lower'],
        ci_upper=result['ci_upper'],
        n_obs=result['n_obs'],
        r_squared=result['r_squared'],
        coef_vector_json=result['coef_vector'],
        sample_desc='Frequent interactors only',
        fixed_effects='Municipality + Year',
        controls_desc='Rainfall',
        cluster_var='b_clusters',
        model_type='OLS with FE'
    )
    print(f"  Frequent interactors only: coef={result['coef']:.4f}, n={result['n_obs']}")

# Subsample: non-frequent interactors
df_nonfreq = df_stacked[df_stacked['frequent_interactor'] == 0].copy()
result = run_regression(
    df_nonfreq,
    'ask_private_stacked ~ treatment + rainfall_std_stacked + year2012 + C(mun_id)',
    cluster_var='b_clusters',
    treatment_var='treatment'
)
if result:
    spec_count += 1
    add_result(
        spec_id='robust/sample/non_frequent_interactors',
        spec_tree_path='robustness/sample_restrictions.md#demographic-subgroups',
        outcome_var='ask_private_stacked',
        treatment_var='treatment',
        coef=result['coef'],
        se=result['se'],
        t_stat=result['t_stat'],
        p_value=result['p_value'],
        ci_lower=result['ci_lower'],
        ci_upper=result['ci_upper'],
        n_obs=result['n_obs'],
        r_squared=result['r_squared'],
        coef_vector_json=result['coef_vector'],
        sample_desc='Non-frequent interactors only',
        fixed_effects='Municipality + Year',
        controls_desc='Rainfall',
        cluster_var='b_clusters',
        model_type='OLS with FE'
    )
    print(f"  Non-frequent interactors: coef={result['coef']:.4f}, n={result['n_obs']}")

# =============================================================================
# 9. HOUSEHOLD DATA SPECIFICATIONS (Table 2)
# =============================================================================
print("\n9. Running household data specifications (Table 2)...")

household_outcomes = [
    ('d_Happiness', '-(CES-D) Scale 2013'),
    ('d_Health', 'SRHS Index 2013'),
    ('d_Child_Food_Security', 'Child Food Security Index 2013'),
    ('d_Overall_index', 'Overall Vulnerability Index 2013'),
]

for outcome, desc in household_outcomes:
    if outcome in df_household.columns:
        # Effect of treatment on vulnerability with mun FE
        result = run_regression(
            df_household,
            f'{outcome} ~ treatment + C(mun_id)',
            cluster_var='b_clusters',
            treatment_var='treatment'
        )
        if result:
            spec_count += 1
            add_result(
                spec_id=f'robust/outcome/hh_{outcome}_treatment',
                spec_tree_path='robustness/measurement.md',
                outcome_var=outcome,
                treatment_var='treatment',
                coef=result['coef'],
                se=result['se'],
                t_stat=result['t_stat'],
                p_value=result['p_value'],
                ci_lower=result['ci_lower'],
                ci_upper=result['ci_upper'],
                n_obs=result['n_obs'],
                r_squared=result['r_squared'],
                coef_vector_json=result['coef_vector'],
                sample_desc='Household data',
                fixed_effects='Municipality',
                controls_desc='None',
                cluster_var='b_clusters',
                model_type='OLS with FE'
            )
            print(f"  {outcome} treatment effect: coef={result['coef']:.4f}")

# Effect of rainfall on vulnerability (no FE as in paper)
for outcome, desc in household_outcomes:
    if outcome in df_household.columns and 'rainfall_2013_std' in df_household.columns:
        result = run_regression(
            df_household,
            f'{outcome} ~ rainfall_2013_std',
            cluster_var='b_clusters',
            treatment_var='rainfall_2013_std'
        )
        if result:
            spec_count += 1
            add_result(
                spec_id=f'robust/outcome/hh_{outcome}_rainfall',
                spec_tree_path='robustness/measurement.md',
                outcome_var=outcome,
                treatment_var='rainfall_2013_std',
                coef=result['coef'],
                se=result['se'],
                t_stat=result['t_stat'],
                p_value=result['p_value'],
                ci_lower=result['ci_lower'],
                ci_upper=result['ci_upper'],
                n_obs=result['n_obs'],
                r_squared=result['r_squared'],
                coef_vector_json=result['coef_vector'],
                sample_desc='Household data',
                fixed_effects='None',
                controls_desc='None',
                cluster_var='b_clusters',
                model_type='OLS'
            )

# =============================================================================
# 10. INDIVIDUAL DATA SPECIFICATIONS (Table 1)
# =============================================================================
print("\n10. Running individual data specifications (Table 1)...")

individual_outcomes = [
    'frequent_interactor',
    'vote_same_party',
    'vote_together',
    'c_declared_dummy',
    'declare_body',
    'declare_house',
    'declare_rally',
]

for outcome in individual_outcomes:
    if outcome in df_individual.columns and 'rainfall_2012_std' in df_individual.columns:
        result = run_regression(
            df_individual,
            f'{outcome} ~ rainfall_2012_std',
            cluster_var='b_clusters',
            treatment_var='rainfall_2012_std'
        )
        if result:
            spec_count += 1
            add_result(
                spec_id=f'robust/outcome/indiv_{outcome}',
                spec_tree_path='robustness/measurement.md',
                outcome_var=outcome,
                treatment_var='rainfall_2012_std',
                coef=result['coef'],
                se=result['se'],
                t_stat=result['t_stat'],
                p_value=result['p_value'],
                ci_lower=result['ci_lower'],
                ci_upper=result['ci_upper'],
                n_obs=result['n_obs'],
                r_squared=result['r_squared'],
                coef_vector_json=result['coef_vector'],
                sample_desc='Individual data',
                fixed_effects='None',
                controls_desc='None',
                cluster_var='b_clusters',
                model_type='OLS'
            )
            print(f"  {outcome}: coef={result['coef']:.4f}")

# =============================================================================
# 11. WINSORIZATION AND TRIMMING
# =============================================================================
print("\n11. Running winsorization and trimming...")

# Winsorize rainfall at 5%
df_wins = df_stacked.copy()
p5, p95 = df_wins['rainfall_std_stacked'].quantile([0.05, 0.95])
df_wins['rainfall_wins'] = df_wins['rainfall_std_stacked'].clip(lower=p5, upper=p95)

result = run_regression(
    df_wins,
    'ask_private_stacked ~ treatment + rainfall_wins + year2012 + C(mun_id)',
    cluster_var='b_clusters',
    treatment_var='treatment'
)
if result:
    spec_count += 1
    add_result(
        spec_id='robust/sample/winsor_rainfall_5pct',
        spec_tree_path='robustness/sample_restrictions.md#outlier-handling',
        outcome_var='ask_private_stacked',
        treatment_var='treatment',
        coef=result['coef'],
        se=result['se'],
        t_stat=result['t_stat'],
        p_value=result['p_value'],
        ci_lower=result['ci_lower'],
        ci_upper=result['ci_upper'],
        n_obs=result['n_obs'],
        r_squared=result['r_squared'],
        coef_vector_json=result['coef_vector'],
        sample_desc='Rainfall winsorized at 5%/95%',
        fixed_effects='Municipality + Year',
        controls_desc='Winsorized rainfall',
        cluster_var='b_clusters',
        model_type='OLS with FE'
    )
    print(f"  Winsorized rainfall 5%: coef={result['coef']:.4f}")

# Winsorize at 1%
p1, p99 = df_wins['rainfall_std_stacked'].quantile([0.01, 0.99])
df_wins['rainfall_wins_1'] = df_wins['rainfall_std_stacked'].clip(lower=p1, upper=p99)

result = run_regression(
    df_wins,
    'ask_private_stacked ~ treatment + rainfall_wins_1 + year2012 + C(mun_id)',
    cluster_var='b_clusters',
    treatment_var='treatment'
)
if result:
    spec_count += 1
    add_result(
        spec_id='robust/sample/winsor_rainfall_1pct',
        spec_tree_path='robustness/sample_restrictions.md#outlier-handling',
        outcome_var='ask_private_stacked',
        treatment_var='treatment',
        coef=result['coef'],
        se=result['se'],
        t_stat=result['t_stat'],
        p_value=result['p_value'],
        ci_lower=result['ci_lower'],
        ci_upper=result['ci_upper'],
        n_obs=result['n_obs'],
        r_squared=result['r_squared'],
        coef_vector_json=result['coef_vector'],
        sample_desc='Rainfall winsorized at 1%/99%',
        fixed_effects='Municipality + Year',
        controls_desc='Winsorized rainfall',
        cluster_var='b_clusters',
        model_type='OLS with FE'
    )
    print(f"  Winsorized rainfall 1%: coef={result['coef']:.4f}")

# =============================================================================
# 12. FUNCTIONAL FORM VARIATIONS
# =============================================================================
print("\n12. Running functional form variations...")

# Quadratic rainfall
df_stacked['rainfall_sq'] = df_stacked['rainfall_std_stacked'] ** 2

result = run_regression(
    df_stacked,
    'ask_private_stacked ~ treatment + rainfall_std_stacked + rainfall_sq + year2012 + C(mun_id)',
    cluster_var='b_clusters',
    treatment_var='treatment'
)
if result:
    spec_count += 1
    add_result(
        spec_id='robust/funcform/quadratic_rainfall',
        spec_tree_path='robustness/functional_form.md',
        outcome_var='ask_private_stacked',
        treatment_var='treatment',
        coef=result['coef'],
        se=result['se'],
        t_stat=result['t_stat'],
        p_value=result['p_value'],
        ci_lower=result['ci_lower'],
        ci_upper=result['ci_upper'],
        n_obs=result['n_obs'],
        r_squared=result['r_squared'],
        coef_vector_json=result['coef_vector'],
        sample_desc='Full stacked sample',
        fixed_effects='Municipality + Year',
        controls_desc='Rainfall + Rainfall squared',
        cluster_var='b_clusters',
        model_type='OLS with FE'
    )
    print(f"  Quadratic rainfall: coef={result['coef']:.4f}")

# =============================================================================
# 13. PLACEBO TESTS
# =============================================================================
print("\n13. Running placebo tests...")

# Placebo outcome: public goods (should be less affected)
result = run_regression(
    df_stacked,
    'ask_public_stacked ~ treatment + rainfall_std_stacked + year2012 + C(mun_id)',
    cluster_var='b_clusters',
    treatment_var='treatment'
)
if result:
    spec_count += 1
    add_result(
        spec_id='robust/placebo/public_goods',
        spec_tree_path='robustness/placebo_tests.md',
        outcome_var='ask_public_stacked',
        treatment_var='treatment',
        coef=result['coef'],
        se=result['se'],
        t_stat=result['t_stat'],
        p_value=result['p_value'],
        ci_lower=result['ci_lower'],
        ci_upper=result['ci_upper'],
        n_obs=result['n_obs'],
        r_squared=result['r_squared'],
        coef_vector_json=result['coef_vector'],
        sample_desc='Full stacked sample',
        fixed_effects='Municipality + Year',
        controls_desc='Rainfall',
        cluster_var='b_clusters',
        model_type='OLS with FE'
    )
    print(f"  Placebo (public goods): coef={result['coef']:.4f}, p={result['p_value']:.4f}")

# =============================================================================
# 14. TREATMENT INTENSITY VARIATIONS
# =============================================================================
print("\n14. Running treatment intensity variations...")

# Treatment interacted with high rainfall indicator
df_stacked['high_rainfall'] = (df_stacked['rainfall_std_stacked'] > 0).astype(int)
df_stacked['treat_high_rain'] = df_stacked['treatment'] * df_stacked['high_rainfall']

result = run_regression(
    df_stacked,
    'ask_private_stacked ~ treatment + high_rainfall + treat_high_rain + year2012 + C(mun_id)',
    cluster_var='b_clusters',
    treatment_var='treat_high_rain'
)
if result:
    spec_count += 1
    add_result(
        spec_id='robust/treatment/high_rainfall_interaction',
        spec_tree_path='robustness/heterogeneity.md',
        outcome_var='ask_private_stacked',
        treatment_var='treat_high_rain',
        coef=result['coef'],
        se=result['se'],
        t_stat=result['t_stat'],
        p_value=result['p_value'],
        ci_lower=result['ci_lower'],
        ci_upper=result['ci_upper'],
        n_obs=result['n_obs'],
        r_squared=result['r_squared'],
        coef_vector_json=result['coef_vector'],
        sample_desc='Full stacked sample',
        fixed_effects='Municipality + Year',
        controls_desc='Treatment x High Rainfall',
        cluster_var='b_clusters',
        model_type='OLS with FE'
    )
    print(f"  Treatment x High Rainfall: coef={result['coef']:.4f}")

# =============================================================================
# 15. ADDITIONAL HETEROGENEITY BY ASSOCIATION MEMBERSHIP
# =============================================================================
print("\n15. Running heterogeneity by association membership...")

if 'mem_assoc' in df_stacked.columns:
    df_stacked['mem_assoc'] = pd.to_numeric(df_stacked['mem_assoc'], errors='coerce')
    df_stacked['treat_assoc'] = df_stacked['treatment'] * df_stacked['mem_assoc']

    result = run_regression(
        df_stacked,
        'ask_private_stacked ~ treatment + treat_assoc + mem_assoc + rainfall_std_stacked + year2012 + C(mun_id)',
        cluster_var='b_clusters',
        treatment_var='treat_assoc'
    )
    if result:
        spec_count += 1
        add_result(
            spec_id='robust/heterogeneity/association_member',
            spec_tree_path='robustness/heterogeneity.md',
            outcome_var='ask_private_stacked',
            treatment_var='treat_assoc',
            coef=result['coef'],
            se=result['se'],
            t_stat=result['t_stat'],
            p_value=result['p_value'],
            ci_lower=result['ci_lower'],
            ci_upper=result['ci_upper'],
            n_obs=result['n_obs'],
            r_squared=result['r_squared'],
            coef_vector_json=result['coef_vector'],
            sample_desc='Full stacked sample',
            fixed_effects='Municipality + Year',
            controls_desc='Treatment x Association Member',
            cluster_var='b_clusters',
            model_type='OLS with FE'
        )
        print(f"  Treatment x Association: coef={result['coef']:.4f}")

    # Subsample: association members only
    df_assoc = df_stacked[df_stacked['mem_assoc'] == 1].copy()
    result = run_regression(
        df_assoc,
        'ask_private_stacked ~ treatment + rainfall_std_stacked + year2012 + C(mun_id)',
        cluster_var='b_clusters',
        treatment_var='treatment'
    )
    if result:
        spec_count += 1
        add_result(
            spec_id='robust/sample/association_members',
            spec_tree_path='robustness/sample_restrictions.md#demographic-subgroups',
            outcome_var='ask_private_stacked',
            treatment_var='treatment',
            coef=result['coef'],
            se=result['se'],
            t_stat=result['t_stat'],
            p_value=result['p_value'],
            ci_lower=result['ci_lower'],
            ci_upper=result['ci_upper'],
            n_obs=result['n_obs'],
            r_squared=result['r_squared'],
            coef_vector_json=result['coef_vector'],
            sample_desc='Association members only',
            fixed_effects='Municipality + Year',
            controls_desc='Rainfall',
            cluster_var='b_clusters',
            model_type='OLS with FE'
        )
        print(f"  Association members only: coef={result['coef']:.4f}, n={result['n_obs']}")

# =============================================================================
# 16. YEAR-SPECIFIC TREATMENT EFFECTS
# =============================================================================
print("\n16. Running year-specific treatment effects...")

df_stacked['treat_2012'] = df_stacked['treatment'] * df_stacked['year2012']
df_stacked['treat_2013'] = df_stacked['treatment'] * df_stacked['year2013']

result = run_regression(
    df_stacked,
    'ask_private_stacked ~ treat_2012 + treat_2013 + year2012 + C(mun_id)',
    cluster_var='b_clusters',
    treatment_var='treat_2012'
)
if result:
    spec_count += 1
    add_result(
        spec_id='robust/treatment/year_specific_2012',
        spec_tree_path='robustness/heterogeneity.md',
        outcome_var='ask_private_stacked',
        treatment_var='treat_2012',
        coef=result['coef'],
        se=result['se'],
        t_stat=result['t_stat'],
        p_value=result['p_value'],
        ci_lower=result['ci_lower'],
        ci_upper=result['ci_upper'],
        n_obs=result['n_obs'],
        r_squared=result['r_squared'],
        coef_vector_json=result['coef_vector'],
        sample_desc='Full stacked sample',
        fixed_effects='Municipality + Year',
        controls_desc='Year-specific treatment',
        cluster_var='b_clusters',
        model_type='OLS with FE'
    )
    print(f"  Treatment x 2012: coef={result['coef']:.4f}")

result = run_regression(
    df_stacked,
    'ask_private_stacked ~ treat_2012 + treat_2013 + year2012 + C(mun_id)',
    cluster_var='b_clusters',
    treatment_var='treat_2013'
)
if result:
    spec_count += 1
    add_result(
        spec_id='robust/treatment/year_specific_2013',
        spec_tree_path='robustness/heterogeneity.md',
        outcome_var='ask_private_stacked',
        treatment_var='treat_2013',
        coef=result['coef'],
        se=result['se'],
        t_stat=result['t_stat'],
        p_value=result['p_value'],
        ci_lower=result['ci_lower'],
        ci_upper=result['ci_upper'],
        n_obs=result['n_obs'],
        r_squared=result['r_squared'],
        coef_vector_json=result['coef_vector'],
        sample_desc='Full stacked sample',
        fixed_effects='Municipality + Year',
        controls_desc='Year-specific treatment',
        cluster_var='b_clusters',
        model_type='OLS with FE'
    )
    print(f"  Treatment x 2013: coef={result['coef']:.4f}")

# =============================================================================
# 17. RAINFALL YEAR-SPECIFIC EFFECTS
# =============================================================================
print("\n17. Running rainfall year-specific effects...")

df_stacked['rain_2012'] = df_stacked['rainfall_std_stacked'] * df_stacked['year2012']
df_stacked['rain_2013'] = df_stacked['rainfall_std_stacked'] * df_stacked['year2013']

result = run_regression(
    df_stacked,
    'ask_private_stacked ~ rain_2012 + rain_2013 + year2012 + C(mun_id)',
    cluster_var='b_clusters',
    treatment_var='rain_2012'
)
if result:
    spec_count += 1
    add_result(
        spec_id='robust/treatment/rainfall_2012',
        spec_tree_path='robustness/heterogeneity.md',
        outcome_var='ask_private_stacked',
        treatment_var='rain_2012',
        coef=result['coef'],
        se=result['se'],
        t_stat=result['t_stat'],
        p_value=result['p_value'],
        ci_lower=result['ci_lower'],
        ci_upper=result['ci_upper'],
        n_obs=result['n_obs'],
        r_squared=result['r_squared'],
        coef_vector_json=result['coef_vector'],
        sample_desc='Full stacked sample',
        fixed_effects='Municipality + Year',
        controls_desc='Year-specific rainfall',
        cluster_var='b_clusters',
        model_type='OLS with FE'
    )
    print(f"  Rainfall x 2012: coef={result['coef']:.4f}")

result = run_regression(
    df_stacked,
    'ask_private_stacked ~ rain_2012 + rain_2013 + year2012 + C(mun_id)',
    cluster_var='b_clusters',
    treatment_var='rain_2013'
)
if result:
    spec_count += 1
    add_result(
        spec_id='robust/treatment/rainfall_2013',
        spec_tree_path='robustness/heterogeneity.md',
        outcome_var='ask_private_stacked',
        treatment_var='rain_2013',
        coef=result['coef'],
        se=result['se'],
        t_stat=result['t_stat'],
        p_value=result['p_value'],
        ci_lower=result['ci_lower'],
        ci_upper=result['ci_upper'],
        n_obs=result['n_obs'],
        r_squared=result['r_squared'],
        coef_vector_json=result['coef_vector'],
        sample_desc='Full stacked sample',
        fixed_effects='Municipality + Year',
        controls_desc='Year-specific rainfall',
        cluster_var='b_clusters',
        model_type='OLS with FE'
    )
    print(f"  Rainfall x 2013: coef={result['coef']:.4f}")

# =============================================================================
# 18. PROBIT MODELS
# =============================================================================
print("\n18. Running probit models for binary outcomes...")

try:
    df_probit = df_stacked[['ask_private_stacked', 'treatment', 'rainfall_std_stacked', 'year2012']].dropna()
    model = smf.probit('ask_private_stacked ~ treatment + rainfall_std_stacked + year2012',
                       data=df_probit).fit(disp=0)

    mfx = model.get_margeff()
    spec_count += 1
    add_result(
        spec_id='robust/estimation/probit',
        spec_tree_path='robustness/model_specification.md',
        outcome_var='ask_private_stacked',
        treatment_var='treatment',
        coef=float(mfx.margeff[0]),
        se=float(mfx.margeff_se[0]),
        t_stat=float(mfx.tvalues[0]),
        p_value=float(mfx.pvalues[0]),
        ci_lower=float(mfx.conf_int()[0][0]),
        ci_upper=float(mfx.conf_int()[0][1]),
        n_obs=int(model.nobs),
        r_squared=float(model.prsquared),
        coef_vector_json={'treatment': {'coef': float(mfx.margeff[0]), 'se': float(mfx.margeff_se[0])}},
        sample_desc='Full stacked sample',
        fixed_effects='None (probit)',
        controls_desc='Rainfall + Year',
        cluster_var='None',
        model_type='Probit (marginal effects)'
    )
    print(f"  Probit marginal effect: coef={mfx.margeff[0]:.4f}")
except Exception as e:
    print(f"  Probit model failed: {e}")

# =============================================================================
# 19. ADDITIONAL HOUSEHOLD-LEVEL CONTROLS
# =============================================================================
print("\n19. Running with additional household-level controls...")

household_controls = ['b_total_members', 'b_female_hoh', 'owns_house']
avail_controls = [c for c in household_controls if c in df_household.columns]

if avail_controls:
    control_str = ' + '.join(avail_controls)
    for outcome in ['d_Overall_index', 'd_Health', 'd_Happiness']:
        if outcome in df_household.columns:
            result = run_regression(
                df_household,
                f'{outcome} ~ treatment + {control_str} + C(mun_id)',
                cluster_var='b_clusters',
                treatment_var='treatment'
            )
            if result:
                spec_count += 1
                add_result(
                    spec_id=f'robust/control/hh_{outcome}_with_controls',
                    spec_tree_path='robustness/control_progression.md',
                    outcome_var=outcome,
                    treatment_var='treatment',
                    coef=result['coef'],
                    se=result['se'],
                    t_stat=result['t_stat'],
                    p_value=result['p_value'],
                    ci_lower=result['ci_lower'],
                    ci_upper=result['ci_upper'],
                    n_obs=result['n_obs'],
                    r_squared=result['r_squared'],
                    coef_vector_json=result['coef_vector'],
                    sample_desc='Household data with controls',
                    fixed_effects='Municipality',
                    controls_desc=', '.join(avail_controls),
                    cluster_var='b_clusters',
                    model_type='OLS with FE'
                )
    print(f"  Added household controls: {avail_controls}")

# =============================================================================
# 20. ADDITIONAL OUTCOME VARIATIONS FOR RAINFALL
# =============================================================================
print("\n20. Running additional outcome variations for rainfall...")

for outcome in ['ask_nowater_private_stacked', 'askrec_private_stacked', 'ask_public_stacked']:
    if outcome in df_stacked.columns:
        result = run_regression(
            df_stacked,
            f'{outcome} ~ treatment + rainfall_std_stacked + year2012 + C(mun_id)',
            cluster_var='b_clusters',
            treatment_var='rainfall_std_stacked'
        )
        if result:
            spec_count += 1
            add_result(
                spec_id=f'robust/outcome/{outcome}_rainfall',
                spec_tree_path='robustness/measurement.md',
                outcome_var=outcome,
                treatment_var='rainfall_std_stacked',
                coef=result['coef'],
                se=result['se'],
                t_stat=result['t_stat'],
                p_value=result['p_value'],
                ci_lower=result['ci_lower'],
                ci_upper=result['ci_upper'],
                n_obs=result['n_obs'],
                r_squared=result['r_squared'],
                coef_vector_json=result['coef_vector'],
                sample_desc='Full stacked sample',
                fixed_effects='Municipality + Year',
                controls_desc='Treatment',
                cluster_var='b_clusters',
                model_type='OLS with FE'
            )
            print(f"  {outcome} (rainfall): coef={result['coef']:.4f}")

# =============================================================================
# SAVE RESULTS
# =============================================================================
print("\n" + "="*80)
print(f"SPECIFICATION SEARCH COMPLETE: {spec_count} specifications")
print("="*80)

# Convert to DataFrame and save
results_df = pd.DataFrame(results)
output_file = OUTPUT_PATH / 'specification_results.csv'
results_df.to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")

# Summary statistics
print("\n" + "-"*40)
print("SUMMARY STATISTICS")
print("-"*40)
print(f"Total specifications: {len(results_df)}")
print(f"Unique outcomes: {results_df['outcome_var'].nunique()}")
print(f"Unique treatments: {results_df['treatment_var'].nunique()}")

# Significance counts for treatment coefficient (main outcome)
main_results = results_df[results_df['outcome_var'] == 'ask_private_stacked']
if len(main_results) > 0:
    print(f"\nFor main outcome (ask_private_stacked):")
    print(f"  Specifications: {len(main_results)}")
    print(f"  Positive coefficients: {(main_results['coefficient'] > 0).sum()} ({100*(main_results['coefficient'] > 0).mean():.1f}%)")
    print(f"  Significant at 5%: {(main_results['p_value'] < 0.05).sum()} ({100*(main_results['p_value'] < 0.05).mean():.1f}%)")
    print(f"  Significant at 1%: {(main_results['p_value'] < 0.01).sum()} ({100*(main_results['p_value'] < 0.01).mean():.1f}%)")
    print(f"  Median coefficient: {main_results['coefficient'].median():.4f}")
    print(f"  Mean coefficient: {main_results['coefficient'].mean():.4f}")
    print(f"  Range: [{main_results['coefficient'].min():.4f}, {main_results['coefficient'].max():.4f}]")

# Overall summary
print(f"\nOverall (all outcomes/treatments):")
print(f"  Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
print(f"  Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
print(f"  Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")

print("\nScript completed successfully.")
