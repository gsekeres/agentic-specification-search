"""
Specification Search: Optimal Life Cycle Unemployment Insurance
Paper ID: 112870-V2
Journal: AER
Authors: Claudio Michelacci and Hernan Ruffo

This script runs a systematic specification search on the empirical analyses in the paper.
The main analysis focuses on the CPS panel regression examining how unemployment rates
respond to UI benefits across age groups.

Method: Panel Fixed Effects (state-year panel)
"""

import pandas as pd
import numpy as np
import json
import warnings
from scipy import stats
import os

warnings.filterwarnings('ignore')

# Try to import pyfixest, fall back to statsmodels if not available
try:
    import pyfixest as pf
    USE_PYFIXEST = True
except ImportError:
    USE_PYFIXEST = False

import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

# =============================================================================
# Configuration
# =============================================================================

PAPER_ID = "112870-V2"
PAPER_TITLE = "Optimal Life Cycle Unemployment Insurance"
JOURNAL = "AER"
PACKAGE_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/112870-V2/DataCodes"

# =============================================================================
# Data Loading and Preparation
# =============================================================================

def load_data():
    """Load the CPS state-level panel data"""
    df = pd.read_stata(os.path.join(PACKAGE_DIR, 'statecps.dta'))

    # Create necessary variables
    # age_i2 represents age groups: 2=young(20-29), 3=prime(30-39), 4=middle(40-49), 5=older(50-59), 6=old(60+)

    # Create age group interactions with benefits (as in the original)
    for age in df['age_i2'].unique():
        if pd.notna(age):
            age_int = int(age)
            df[f'lnwba_age{age_int}'] = df['lnwba'] * (df['age_i2'] == age).astype(int)

    # Create age group dummies
    for age in df['age_i2'].unique():
        if pd.notna(age):
            age_int = int(age)
            df[f'age_group_{age_int}'] = (df['age_i2'] == age).astype(int)

    # Create year dummies
    for year in df['year'].unique():
        if pd.notna(year):
            year_int = int(year)
            df[f'year_{year_int}'] = (df['year'] == year).astype(int)

    # Create period/part dummies
    df['part_2'] = (df['part'] == 2).astype(int)

    # State dummies
    for state in df['state_60'].unique():
        if pd.notna(state):
            state_int = int(state)
            df[f'state_{state_int}'] = (df['state_60'] == state).astype(int)

    # Create panel identifiers
    df['state_id'] = df['state_60'].astype(int)
    df['year_int'] = df['year'].astype(int)
    df['period_int'] = df['period'].astype(int)

    return df

# =============================================================================
# Regression Helper Functions
# =============================================================================

def run_ols_clustered(df, outcome, treatment, controls=None, fe_vars=None, cluster_var=None):
    """
    Run OLS regression with optional fixed effects and clustering
    """
    df_clean = df.dropna(subset=[outcome, treatment]).copy()

    # Build regressor list
    regressors = [treatment]
    if controls:
        regressors.extend([c for c in controls if c in df_clean.columns])

    # Add fixed effects if specified
    if fe_vars:
        for fe in fe_vars:
            fe_dummies = [c for c in df_clean.columns if c.startswith(f'{fe}_')]
            # Drop one dummy for identification
            regressors.extend(fe_dummies[1:] if len(fe_dummies) > 1 else fe_dummies)

    # Remove any missing regressors
    regressors = [r for r in regressors if r in df_clean.columns]

    # Clean the data
    all_vars = [outcome] + regressors
    if cluster_var:
        all_vars.append(cluster_var)
    df_clean = df_clean.dropna(subset=[v for v in all_vars if v in df_clean.columns])

    if len(df_clean) < 10:
        return None

    X = df_clean[regressors].values
    X = sm.add_constant(X)
    y = df_clean[outcome].values

    try:
        model = OLS(y, X)
        if cluster_var and cluster_var in df_clean.columns:
            groups = df_clean[cluster_var].values
            result = model.fit(cov_type='cluster', cov_kwds={'groups': groups})
        else:
            result = model.fit(cov_type='HC1')

        # Extract treatment coefficient (first regressor after constant)
        treat_idx = 1  # After constant

        return {
            'coefficient': result.params[treat_idx],
            'std_error': result.bse[treat_idx],
            't_stat': result.tvalues[treat_idx],
            'p_value': result.pvalues[treat_idx],
            'ci_lower': result.conf_int()[treat_idx, 0],
            'ci_upper': result.conf_int()[treat_idx, 1],
            'n_obs': int(result.nobs),
            'r_squared': result.rsquared,
            'all_params': dict(zip(['const'] + regressors, result.params)),
            'all_se': dict(zip(['const'] + regressors, result.bse)),
            'all_pval': dict(zip(['const'] + regressors, result.pvalues))
        }
    except Exception as e:
        return None


def run_interaction_model(df, outcome, base_treatment, age_groups, controls=None, fe_vars=None, cluster_var=None):
    """
    Run model with treatment x age group interactions
    Returns coefficients for each age group
    """
    df_clean = df.dropna(subset=[outcome]).copy()

    # Build regressor list with age-specific treatment effects
    regressors = []
    treatment_vars = []
    for age in sorted(age_groups):
        var_name = f'{base_treatment}_age{int(age)}'
        if var_name in df_clean.columns:
            regressors.append(var_name)
            treatment_vars.append(var_name)

    # Add age group dummies (excluding one for identification)
    age_dummies = [f'age_group_{int(age)}' for age in sorted(age_groups)[1:]]
    regressors.extend([d for d in age_dummies if d in df_clean.columns])

    if controls:
        regressors.extend([c for c in controls if c in df_clean.columns])

    # Add fixed effects
    if fe_vars:
        for fe in fe_vars:
            fe_dummies = [c for c in df_clean.columns if c.startswith(f'{fe}_')]
            regressors.extend(fe_dummies[1:] if len(fe_dummies) > 1 else fe_dummies)

    # Clean data
    all_vars = [outcome] + regressors
    if cluster_var:
        all_vars.append(cluster_var)
    df_clean = df_clean.dropna(subset=[v for v in all_vars if v in df_clean.columns])

    if len(df_clean) < 10:
        return None

    X = df_clean[regressors].values
    X = sm.add_constant(X)
    y = df_clean[outcome].values

    try:
        model = OLS(y, X)
        if cluster_var and cluster_var in df_clean.columns:
            groups = df_clean[cluster_var].values
            result = model.fit(cov_type='cluster', cov_kwds={'groups': groups})
        else:
            result = model.fit(cov_type='HC1')

        param_names = ['const'] + regressors
        results_dict = {
            'n_obs': int(result.nobs),
            'r_squared': result.rsquared,
            'age_effects': {}
        }

        # Extract age-specific treatment effects
        for i, var in enumerate(treatment_vars):
            idx = param_names.index(var)
            results_dict['age_effects'][var] = {
                'coefficient': result.params[idx],
                'std_error': result.bse[idx],
                't_stat': result.tvalues[idx],
                'p_value': result.pvalues[idx],
                'ci_lower': result.conf_int()[idx, 0],
                'ci_upper': result.conf_int()[idx, 1]
            }

        return results_dict
    except Exception as e:
        return None


# =============================================================================
# Specification Search
# =============================================================================

def run_specification_search():
    """
    Run comprehensive specification search following i4r methodology
    Target: 50+ specifications
    """

    print("Loading data...")
    df = load_data()

    results = []

    # Define key variables
    outcome_var = 'lnun'  # Log unemployment rate
    treatment_var = 'lnwba'  # Log weekly benefit amount

    # Control variables from the paper
    demographic_controls = ['m_married', 'spwork_yn', 'r_white']
    education_controls = ['ed1', 'ed2', 'ed3', 'ed4']
    all_controls = demographic_controls + education_controls

    # Age groups
    age_groups = [2, 3, 4, 5, 6]  # From the data

    # Fixed effect sets
    state_fe_vars = [c for c in df.columns if c.startswith('state_')]
    year_fe_vars = [c for c in df.columns if c.startswith('year_')]

    spec_count = 0

    # ==========================================================================
    # BASELINE SPECIFICATIONS
    # ==========================================================================
    print("Running baseline specifications...")

    # Baseline 1: Main specification with all controls (replicating Figure 1b)
    result = run_interaction_model(
        df, outcome_var, treatment_var, age_groups,
        controls=all_controls,
        fe_vars=['state', 'year', 'part'],
        cluster_var=None
    )
    if result:
        # For baseline, use the average effect across age groups
        avg_coef = np.mean([v['coefficient'] for v in result['age_effects'].values()])
        avg_se = np.mean([v['std_error'] for v in result['age_effects'].values()])

        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'baseline',
            'spec_tree_path': 'methods/panel_fixed_effects.md',
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': avg_coef,
            'std_error': avg_se,
            't_stat': avg_coef / avg_se if avg_se > 0 else np.nan,
            'p_value': 2 * (1 - stats.norm.cdf(abs(avg_coef / avg_se))) if avg_se > 0 else np.nan,
            'ci_lower': avg_coef - 1.96 * avg_se,
            'ci_upper': avg_coef + 1.96 * avg_se,
            'n_obs': result['n_obs'],
            'r_squared': result['r_squared'],
            'coefficient_vector_json': json.dumps(result['age_effects']),
            'sample_desc': 'Full sample, all age groups',
            'fixed_effects': 'State + Year + Part',
            'controls_desc': 'Demographics + Education',
            'cluster_var': 'None',
            'model_type': 'Panel FE with age interactions',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })
        spec_count += 1

    # Baseline 2: Pooled treatment effect
    result = run_ols_clustered(
        df, outcome_var, treatment_var,
        controls=all_controls,
        fe_vars=['state', 'year', 'part'],
        cluster_var='state_id'
    )
    if result:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'baseline_pooled',
            'spec_tree_path': 'methods/panel_fixed_effects.md',
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': result['coefficient'],
            'std_error': result['std_error'],
            't_stat': result['t_stat'],
            'p_value': result['p_value'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'n_obs': result['n_obs'],
            'r_squared': result['r_squared'],
            'coefficient_vector_json': json.dumps({'treatment': {'coef': result['coefficient'], 'se': result['std_error']}}),
            'sample_desc': 'Full sample, pooled across ages',
            'fixed_effects': 'State + Year + Part',
            'controls_desc': 'Demographics + Education',
            'cluster_var': 'State',
            'model_type': 'Panel FE pooled',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })
        spec_count += 1

    # ==========================================================================
    # CONTROL VARIABLE VARIATIONS (10-15 specs)
    # ==========================================================================
    print("Running control variable variations...")

    # No controls
    result = run_ols_clustered(
        df, outcome_var, treatment_var,
        controls=None,
        fe_vars=['state', 'year'],
        cluster_var='state_id'
    )
    if result:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/control/none',
            'spec_tree_path': 'robustness/control_progression.md',
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': result['coefficient'],
            'std_error': result['std_error'],
            't_stat': result['t_stat'],
            'p_value': result['p_value'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'n_obs': result['n_obs'],
            'r_squared': result['r_squared'],
            'coefficient_vector_json': json.dumps({'treatment': {'coef': result['coefficient'], 'se': result['std_error']}}),
            'sample_desc': 'Full sample',
            'fixed_effects': 'State + Year',
            'controls_desc': 'None',
            'cluster_var': 'State',
            'model_type': 'Panel FE',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })
        spec_count += 1

    # Demographics only
    result = run_ols_clustered(
        df, outcome_var, treatment_var,
        controls=demographic_controls,
        fe_vars=['state', 'year'],
        cluster_var='state_id'
    )
    if result:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/control/demographics',
            'spec_tree_path': 'robustness/control_progression.md',
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': result['coefficient'],
            'std_error': result['std_error'],
            't_stat': result['t_stat'],
            'p_value': result['p_value'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'n_obs': result['n_obs'],
            'r_squared': result['r_squared'],
            'coefficient_vector_json': json.dumps({'treatment': {'coef': result['coefficient'], 'se': result['std_error']}}),
            'sample_desc': 'Full sample',
            'fixed_effects': 'State + Year',
            'controls_desc': 'Demographics only (married, spouse work, white)',
            'cluster_var': 'State',
            'model_type': 'Panel FE',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })
        spec_count += 1

    # Education only
    result = run_ols_clustered(
        df, outcome_var, treatment_var,
        controls=education_controls,
        fe_vars=['state', 'year'],
        cluster_var='state_id'
    )
    if result:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/control/education',
            'spec_tree_path': 'robustness/control_progression.md',
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': result['coefficient'],
            'std_error': result['std_error'],
            't_stat': result['t_stat'],
            'p_value': result['p_value'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'n_obs': result['n_obs'],
            'r_squared': result['r_squared'],
            'coefficient_vector_json': json.dumps({'treatment': {'coef': result['coefficient'], 'se': result['std_error']}}),
            'sample_desc': 'Full sample',
            'fixed_effects': 'State + Year',
            'controls_desc': 'Education only (ed1-ed4)',
            'cluster_var': 'State',
            'model_type': 'Panel FE',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })
        spec_count += 1

    # Leave-one-out: Drop each control one at a time
    for control in all_controls:
        remaining_controls = [c for c in all_controls if c != control]
        result = run_ols_clustered(
            df, outcome_var, treatment_var,
            controls=remaining_controls,
            fe_vars=['state', 'year'],
            cluster_var='state_id'
        )
        if result:
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': f'robust/control/drop_{control}',
                'spec_tree_path': 'robustness/leave_one_out.md',
                'outcome_var': outcome_var,
                'treatment_var': treatment_var,
                'coefficient': result['coefficient'],
                'std_error': result['std_error'],
                't_stat': result['t_stat'],
                'p_value': result['p_value'],
                'ci_lower': result['ci_lower'],
                'ci_upper': result['ci_upper'],
                'n_obs': result['n_obs'],
                'r_squared': result['r_squared'],
                'coefficient_vector_json': json.dumps({'treatment': {'coef': result['coefficient'], 'se': result['std_error']}}),
                'sample_desc': 'Full sample',
                'fixed_effects': 'State + Year',
                'controls_desc': f'All controls except {control}',
                'cluster_var': 'State',
                'model_type': 'Panel FE',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })
            spec_count += 1

    # ==========================================================================
    # FIXED EFFECTS VARIATIONS (5-8 specs)
    # ==========================================================================
    print("Running fixed effects variations...")

    # No fixed effects (pooled OLS)
    result = run_ols_clustered(
        df, outcome_var, treatment_var,
        controls=all_controls,
        fe_vars=None,
        cluster_var='state_id'
    )
    if result:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/estimation/no_fe',
            'spec_tree_path': 'methods/panel_fixed_effects.md#no-fixed-effects',
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': result['coefficient'],
            'std_error': result['std_error'],
            't_stat': result['t_stat'],
            'p_value': result['p_value'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'n_obs': result['n_obs'],
            'r_squared': result['r_squared'],
            'coefficient_vector_json': json.dumps({'treatment': {'coef': result['coefficient'], 'se': result['std_error']}}),
            'sample_desc': 'Full sample',
            'fixed_effects': 'None',
            'controls_desc': 'All controls',
            'cluster_var': 'State',
            'model_type': 'Pooled OLS',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })
        spec_count += 1

    # State FE only
    result = run_ols_clustered(
        df, outcome_var, treatment_var,
        controls=all_controls,
        fe_vars=['state'],
        cluster_var='state_id'
    )
    if result:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/estimation/state_fe_only',
            'spec_tree_path': 'methods/panel_fixed_effects.md#unit-fe',
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': result['coefficient'],
            'std_error': result['std_error'],
            't_stat': result['t_stat'],
            'p_value': result['p_value'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'n_obs': result['n_obs'],
            'r_squared': result['r_squared'],
            'coefficient_vector_json': json.dumps({'treatment': {'coef': result['coefficient'], 'se': result['std_error']}}),
            'sample_desc': 'Full sample',
            'fixed_effects': 'State only',
            'controls_desc': 'All controls',
            'cluster_var': 'State',
            'model_type': 'State FE',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })
        spec_count += 1

    # Year FE only
    result = run_ols_clustered(
        df, outcome_var, treatment_var,
        controls=all_controls,
        fe_vars=['year'],
        cluster_var='state_id'
    )
    if result:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/estimation/year_fe_only',
            'spec_tree_path': 'methods/panel_fixed_effects.md#time-fe',
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': result['coefficient'],
            'std_error': result['std_error'],
            't_stat': result['t_stat'],
            'p_value': result['p_value'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'n_obs': result['n_obs'],
            'r_squared': result['r_squared'],
            'coefficient_vector_json': json.dumps({'treatment': {'coef': result['coefficient'], 'se': result['std_error']}}),
            'sample_desc': 'Full sample',
            'fixed_effects': 'Year only',
            'controls_desc': 'All controls',
            'cluster_var': 'State',
            'model_type': 'Year FE',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })
        spec_count += 1

    # Two-way FE (state + year)
    result = run_ols_clustered(
        df, outcome_var, treatment_var,
        controls=all_controls,
        fe_vars=['state', 'year'],
        cluster_var='state_id'
    )
    if result:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/estimation/twoway_fe',
            'spec_tree_path': 'methods/panel_fixed_effects.md#twoway-fe',
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': result['coefficient'],
            'std_error': result['std_error'],
            't_stat': result['t_stat'],
            'p_value': result['p_value'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'n_obs': result['n_obs'],
            'r_squared': result['r_squared'],
            'coefficient_vector_json': json.dumps({'treatment': {'coef': result['coefficient'], 'se': result['std_error']}}),
            'sample_desc': 'Full sample',
            'fixed_effects': 'State + Year',
            'controls_desc': 'All controls',
            'cluster_var': 'State',
            'model_type': 'Two-way FE',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })
        spec_count += 1

    # ==========================================================================
    # CLUSTERING VARIATIONS (5-8 specs)
    # ==========================================================================
    print("Running clustering variations...")

    # No clustering (robust SE only)
    result = run_ols_clustered(
        df, outcome_var, treatment_var,
        controls=all_controls,
        fe_vars=['state', 'year'],
        cluster_var=None
    )
    if result:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/cluster/none',
            'spec_tree_path': 'robustness/clustering_variations.md',
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': result['coefficient'],
            'std_error': result['std_error'],
            't_stat': result['t_stat'],
            'p_value': result['p_value'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'n_obs': result['n_obs'],
            'r_squared': result['r_squared'],
            'coefficient_vector_json': json.dumps({'treatment': {'coef': result['coefficient'], 'se': result['std_error']}}),
            'sample_desc': 'Full sample',
            'fixed_effects': 'State + Year',
            'controls_desc': 'All controls',
            'cluster_var': 'None (robust SE)',
            'model_type': 'Panel FE',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })
        spec_count += 1

    # Cluster by year
    result = run_ols_clustered(
        df, outcome_var, treatment_var,
        controls=all_controls,
        fe_vars=['state', 'year'],
        cluster_var='year_int'
    )
    if result:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/cluster/year',
            'spec_tree_path': 'robustness/clustering_variations.md',
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': result['coefficient'],
            'std_error': result['std_error'],
            't_stat': result['t_stat'],
            'p_value': result['p_value'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'n_obs': result['n_obs'],
            'r_squared': result['r_squared'],
            'coefficient_vector_json': json.dumps({'treatment': {'coef': result['coefficient'], 'se': result['std_error']}}),
            'sample_desc': 'Full sample',
            'fixed_effects': 'State + Year',
            'controls_desc': 'All controls',
            'cluster_var': 'Year',
            'model_type': 'Panel FE',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })
        spec_count += 1

    # ==========================================================================
    # SAMPLE RESTRICTIONS (10-15 specs)
    # ==========================================================================
    print("Running sample restrictions...")

    # By age group
    for age in age_groups:
        df_subset = df[df['age_i2'] == age].copy()
        result = run_ols_clustered(
            df_subset, outcome_var, treatment_var,
            controls=all_controls,
            fe_vars=['state', 'year'],
            cluster_var='state_id'
        )
        if result:
            age_labels = {2: '20-29', 3: '30-39', 4: '40-49', 5: '50-59', 6: '60+'}
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': f'robust/sample/age_group_{age}',
                'spec_tree_path': 'robustness/sample_restrictions.md',
                'outcome_var': outcome_var,
                'treatment_var': treatment_var,
                'coefficient': result['coefficient'],
                'std_error': result['std_error'],
                't_stat': result['t_stat'],
                'p_value': result['p_value'],
                'ci_lower': result['ci_lower'],
                'ci_upper': result['ci_upper'],
                'n_obs': result['n_obs'],
                'r_squared': result['r_squared'],
                'coefficient_vector_json': json.dumps({'treatment': {'coef': result['coefficient'], 'se': result['std_error']}}),
                'sample_desc': f'Age group {age_labels.get(age, str(age))} only',
                'fixed_effects': 'State + Year',
                'controls_desc': 'All controls',
                'cluster_var': 'State',
                'model_type': 'Panel FE',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })
            spec_count += 1

    # Young vs Old workers (as in paper)
    # Young: age groups 2-3 (20-39)
    df_young = df[df['age_i2'].isin([2, 3])].copy()
    result = run_ols_clustered(
        df_young, outcome_var, treatment_var,
        controls=all_controls,
        fe_vars=['state', 'year'],
        cluster_var='state_id'
    )
    if result:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/sample/young_workers',
            'spec_tree_path': 'robustness/sample_restrictions.md',
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': result['coefficient'],
            'std_error': result['std_error'],
            't_stat': result['t_stat'],
            'p_value': result['p_value'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'n_obs': result['n_obs'],
            'r_squared': result['r_squared'],
            'coefficient_vector_json': json.dumps({'treatment': {'coef': result['coefficient'], 'se': result['std_error']}}),
            'sample_desc': 'Young workers (age 20-39)',
            'fixed_effects': 'State + Year',
            'controls_desc': 'All controls',
            'cluster_var': 'State',
            'model_type': 'Panel FE',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })
        spec_count += 1

    # Old: age groups 5-6 (50+)
    df_old = df[df['age_i2'].isin([5, 6])].copy()
    result = run_ols_clustered(
        df_old, outcome_var, treatment_var,
        controls=all_controls,
        fe_vars=['state', 'year'],
        cluster_var='state_id'
    )
    if result:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/sample/old_workers',
            'spec_tree_path': 'robustness/sample_restrictions.md',
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': result['coefficient'],
            'std_error': result['std_error'],
            't_stat': result['t_stat'],
            'p_value': result['p_value'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'n_obs': result['n_obs'],
            'r_squared': result['r_squared'],
            'coefficient_vector_json': json.dumps({'treatment': {'coef': result['coefficient'], 'se': result['std_error']}}),
            'sample_desc': 'Old workers (age 50+)',
            'fixed_effects': 'State + Year',
            'controls_desc': 'All controls',
            'cluster_var': 'State',
            'model_type': 'Panel FE',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })
        spec_count += 1

    # Time period restrictions
    years = sorted(df['year'].unique())
    mid_year = np.median(years)

    # Early period
    df_early = df[df['year'] <= mid_year].copy()
    result = run_ols_clustered(
        df_early, outcome_var, treatment_var,
        controls=all_controls,
        fe_vars=['state', 'year'],
        cluster_var='state_id'
    )
    if result:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/sample/early_period',
            'spec_tree_path': 'robustness/sample_restrictions.md',
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': result['coefficient'],
            'std_error': result['std_error'],
            't_stat': result['t_stat'],
            'p_value': result['p_value'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'n_obs': result['n_obs'],
            'r_squared': result['r_squared'],
            'coefficient_vector_json': json.dumps({'treatment': {'coef': result['coefficient'], 'se': result['std_error']}}),
            'sample_desc': f'Early period (1984-{int(mid_year)})',
            'fixed_effects': 'State + Year',
            'controls_desc': 'All controls',
            'cluster_var': 'State',
            'model_type': 'Panel FE',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })
        spec_count += 1

    # Late period
    df_late = df[df['year'] > mid_year].copy()
    result = run_ols_clustered(
        df_late, outcome_var, treatment_var,
        controls=all_controls,
        fe_vars=['state', 'year'],
        cluster_var='state_id'
    )
    if result:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/sample/late_period',
            'spec_tree_path': 'robustness/sample_restrictions.md',
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': result['coefficient'],
            'std_error': result['std_error'],
            't_stat': result['t_stat'],
            'p_value': result['p_value'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'n_obs': result['n_obs'],
            'r_squared': result['r_squared'],
            'coefficient_vector_json': json.dumps({'treatment': {'coef': result['coefficient'], 'se': result['std_error']}}),
            'sample_desc': f'Late period ({int(mid_year)+1}-2000)',
            'fixed_effects': 'State + Year',
            'controls_desc': 'All controls',
            'cluster_var': 'State',
            'model_type': 'Panel FE',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })
        spec_count += 1

    # Drop individual years
    for year in sorted(df['year'].unique())[:5]:  # First 5 years
        df_drop = df[df['year'] != year].copy()
        result = run_ols_clustered(
            df_drop, outcome_var, treatment_var,
            controls=all_controls,
            fe_vars=['state', 'year'],
            cluster_var='state_id'
        )
        if result:
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': f'robust/sample/drop_year_{int(year)}',
                'spec_tree_path': 'robustness/sample_restrictions.md',
                'outcome_var': outcome_var,
                'treatment_var': treatment_var,
                'coefficient': result['coefficient'],
                'std_error': result['std_error'],
                't_stat': result['t_stat'],
                'p_value': result['p_value'],
                'ci_lower': result['ci_lower'],
                'ci_upper': result['ci_upper'],
                'n_obs': result['n_obs'],
                'r_squared': result['r_squared'],
                'coefficient_vector_json': json.dumps({'treatment': {'coef': result['coefficient'], 'se': result['std_error']}}),
                'sample_desc': f'Excluding year {int(year)}',
                'fixed_effects': 'State + Year',
                'controls_desc': 'All controls',
                'cluster_var': 'State',
                'model_type': 'Panel FE',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })
            spec_count += 1

    # Outlier handling
    # Trim 1%
    lower = df['lnun'].quantile(0.01)
    upper = df['lnun'].quantile(0.99)
    df_trim = df[(df['lnun'] >= lower) & (df['lnun'] <= upper)].copy()
    result = run_ols_clustered(
        df_trim, outcome_var, treatment_var,
        controls=all_controls,
        fe_vars=['state', 'year'],
        cluster_var='state_id'
    )
    if result:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/sample/trim_1pct',
            'spec_tree_path': 'robustness/sample_restrictions.md',
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': result['coefficient'],
            'std_error': result['std_error'],
            't_stat': result['t_stat'],
            'p_value': result['p_value'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'n_obs': result['n_obs'],
            'r_squared': result['r_squared'],
            'coefficient_vector_json': json.dumps({'treatment': {'coef': result['coefficient'], 'se': result['std_error']}}),
            'sample_desc': 'Trimmed 1% tails of outcome',
            'fixed_effects': 'State + Year',
            'controls_desc': 'All controls',
            'cluster_var': 'State',
            'model_type': 'Panel FE',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })
        spec_count += 1

    # Trim 5%
    lower = df['lnun'].quantile(0.05)
    upper = df['lnun'].quantile(0.95)
    df_trim = df[(df['lnun'] >= lower) & (df['lnun'] <= upper)].copy()
    result = run_ols_clustered(
        df_trim, outcome_var, treatment_var,
        controls=all_controls,
        fe_vars=['state', 'year'],
        cluster_var='state_id'
    )
    if result:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/sample/trim_5pct',
            'spec_tree_path': 'robustness/sample_restrictions.md',
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': result['coefficient'],
            'std_error': result['std_error'],
            't_stat': result['t_stat'],
            'p_value': result['p_value'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'n_obs': result['n_obs'],
            'r_squared': result['r_squared'],
            'coefficient_vector_json': json.dumps({'treatment': {'coef': result['coefficient'], 'se': result['std_error']}}),
            'sample_desc': 'Trimmed 5% tails of outcome',
            'fixed_effects': 'State + Year',
            'controls_desc': 'All controls',
            'cluster_var': 'State',
            'model_type': 'Panel FE',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })
        spec_count += 1

    # ==========================================================================
    # FUNCTIONAL FORM VARIATIONS (3-5 specs)
    # ==========================================================================
    print("Running functional form variations...")

    # Levels (unemployment rate) instead of logs
    df['un_rate'] = np.exp(df['lnun'])
    df['wba'] = np.exp(df['lnwba'])

    result = run_ols_clustered(
        df, 'un_rate', 'wba',
        controls=all_controls,
        fe_vars=['state', 'year'],
        cluster_var='state_id'
    )
    if result:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/funcform/levels',
            'spec_tree_path': 'robustness/functional_form.md',
            'outcome_var': 'un_rate',
            'treatment_var': 'wba',
            'coefficient': result['coefficient'],
            'std_error': result['std_error'],
            't_stat': result['t_stat'],
            'p_value': result['p_value'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'n_obs': result['n_obs'],
            'r_squared': result['r_squared'],
            'coefficient_vector_json': json.dumps({'treatment': {'coef': result['coefficient'], 'se': result['std_error']}}),
            'sample_desc': 'Full sample',
            'fixed_effects': 'State + Year',
            'controls_desc': 'All controls',
            'cluster_var': 'State',
            'model_type': 'Panel FE (levels)',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })
        spec_count += 1

    # Log outcome, level treatment
    result = run_ols_clustered(
        df, 'lnun', 'wba',
        controls=all_controls,
        fe_vars=['state', 'year'],
        cluster_var='state_id'
    )
    if result:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/funcform/log_level',
            'spec_tree_path': 'robustness/functional_form.md',
            'outcome_var': 'lnun',
            'treatment_var': 'wba',
            'coefficient': result['coefficient'],
            'std_error': result['std_error'],
            't_stat': result['t_stat'],
            'p_value': result['p_value'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'n_obs': result['n_obs'],
            'r_squared': result['r_squared'],
            'coefficient_vector_json': json.dumps({'treatment': {'coef': result['coefficient'], 'se': result['std_error']}}),
            'sample_desc': 'Full sample',
            'fixed_effects': 'State + Year',
            'controls_desc': 'All controls',
            'cluster_var': 'State',
            'model_type': 'Panel FE (log-level)',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })
        spec_count += 1

    # Level outcome, log treatment
    result = run_ols_clustered(
        df, 'un_rate', 'lnwba',
        controls=all_controls,
        fe_vars=['state', 'year'],
        cluster_var='state_id'
    )
    if result:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/funcform/level_log',
            'spec_tree_path': 'robustness/functional_form.md',
            'outcome_var': 'un_rate',
            'treatment_var': 'lnwba',
            'coefficient': result['coefficient'],
            'std_error': result['std_error'],
            't_stat': result['t_stat'],
            'p_value': result['p_value'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'n_obs': result['n_obs'],
            'r_squared': result['r_squared'],
            'coefficient_vector_json': json.dumps({'treatment': {'coef': result['coefficient'], 'se': result['std_error']}}),
            'sample_desc': 'Full sample',
            'fixed_effects': 'State + Year',
            'controls_desc': 'All controls',
            'cluster_var': 'State',
            'model_type': 'Panel FE (level-log)',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })
        spec_count += 1

    # ==========================================================================
    # HETEROGENEITY ANALYSIS (5-10 specs)
    # ==========================================================================
    print("Running heterogeneity analysis...")

    # Interactions with demographics
    for het_var in ['m_married', 'r_white']:
        df[f'{treatment_var}_x_{het_var}'] = df[treatment_var] * df[het_var]

        result = run_ols_clustered(
            df, outcome_var, f'{treatment_var}_x_{het_var}',
            controls=all_controls + [treatment_var],
            fe_vars=['state', 'year'],
            cluster_var='state_id'
        )
        if result:
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': f'robust/heterogeneity/{het_var}',
                'spec_tree_path': 'robustness/heterogeneity.md',
                'outcome_var': outcome_var,
                'treatment_var': f'{treatment_var}_x_{het_var}',
                'coefficient': result['coefficient'],
                'std_error': result['std_error'],
                't_stat': result['t_stat'],
                'p_value': result['p_value'],
                'ci_lower': result['ci_lower'],
                'ci_upper': result['ci_upper'],
                'n_obs': result['n_obs'],
                'r_squared': result['r_squared'],
                'coefficient_vector_json': json.dumps({'treatment': {'coef': result['coefficient'], 'se': result['std_error']}}),
                'sample_desc': 'Full sample',
                'fixed_effects': 'State + Year',
                'controls_desc': f'All controls + {het_var} interaction',
                'cluster_var': 'State',
                'model_type': 'Panel FE with interaction',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })
            spec_count += 1

    # Subsample by married status
    for married in [0, 1]:
        df_sub = df[df['m_married'] == married].copy()
        result = run_ols_clustered(
            df_sub, outcome_var, treatment_var,
            controls=[c for c in all_controls if c != 'm_married'],
            fe_vars=['state', 'year'],
            cluster_var='state_id'
        )
        if result:
            married_label = 'married' if married == 1 else 'unmarried'
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': f'robust/heterogeneity/subsample_{married_label}',
                'spec_tree_path': 'robustness/heterogeneity.md',
                'outcome_var': outcome_var,
                'treatment_var': treatment_var,
                'coefficient': result['coefficient'],
                'std_error': result['std_error'],
                't_stat': result['t_stat'],
                'p_value': result['p_value'],
                'ci_lower': result['ci_lower'],
                'ci_upper': result['ci_upper'],
                'n_obs': result['n_obs'],
                'r_squared': result['r_squared'],
                'coefficient_vector_json': json.dumps({'treatment': {'coef': result['coefficient'], 'se': result['std_error']}}),
                'sample_desc': f'{married_label.capitalize()} workers only',
                'fixed_effects': 'State + Year',
                'controls_desc': 'All controls except marriage',
                'cluster_var': 'State',
                'model_type': 'Panel FE',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })
            spec_count += 1

    # Subsample by race
    for white in [0, 1]:
        df_sub = df[df['r_white'] == white].copy()
        result = run_ols_clustered(
            df_sub, outcome_var, treatment_var,
            controls=[c for c in all_controls if c != 'r_white'],
            fe_vars=['state', 'year'],
            cluster_var='state_id'
        )
        if result:
            race_label = 'white' if white == 1 else 'nonwhite'
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': f'robust/heterogeneity/subsample_{race_label}',
                'spec_tree_path': 'robustness/heterogeneity.md',
                'outcome_var': outcome_var,
                'treatment_var': treatment_var,
                'coefficient': result['coefficient'],
                'std_error': result['std_error'],
                't_stat': result['t_stat'],
                'p_value': result['p_value'],
                'ci_lower': result['ci_lower'],
                'ci_upper': result['ci_upper'],
                'n_obs': result['n_obs'],
                'r_squared': result['r_squared'],
                'coefficient_vector_json': json.dumps({'treatment': {'coef': result['coefficient'], 'se': result['std_error']}}),
                'sample_desc': f'{race_label.capitalize()} workers only',
                'fixed_effects': 'State + Year',
                'controls_desc': 'All controls except race',
                'cluster_var': 'State',
                'model_type': 'Panel FE',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })
            spec_count += 1

    # ==========================================================================
    # PLACEBO TESTS (3-5 specs)
    # ==========================================================================
    print("Running placebo tests...")

    # Lagged treatment (1-year lag)
    df['lnwba_lag1'] = df.groupby('state_id')['lnwba'].shift(1)
    result = run_ols_clustered(
        df.dropna(subset=['lnwba_lag1']), outcome_var, 'lnwba_lag1',
        controls=all_controls + [treatment_var],
        fe_vars=['state', 'year'],
        cluster_var='state_id'
    )
    if result:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/placebo/lag1_treatment',
            'spec_tree_path': 'robustness/placebo_tests.md',
            'outcome_var': outcome_var,
            'treatment_var': 'lnwba_lag1',
            'coefficient': result['coefficient'],
            'std_error': result['std_error'],
            't_stat': result['t_stat'],
            'p_value': result['p_value'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'n_obs': result['n_obs'],
            'r_squared': result['r_squared'],
            'coefficient_vector_json': json.dumps({'treatment': {'coef': result['coefficient'], 'se': result['std_error']}}),
            'sample_desc': 'Full sample',
            'fixed_effects': 'State + Year',
            'controls_desc': 'All controls + current treatment',
            'cluster_var': 'State',
            'model_type': 'Panel FE (placebo)',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })
        spec_count += 1

    # Lead treatment (1-year lead) - should not predict current outcome
    df['lnwba_lead1'] = df.groupby('state_id')['lnwba'].shift(-1)
    result = run_ols_clustered(
        df.dropna(subset=['lnwba_lead1']), outcome_var, 'lnwba_lead1',
        controls=all_controls + [treatment_var],
        fe_vars=['state', 'year'],
        cluster_var='state_id'
    )
    if result:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/placebo/lead1_treatment',
            'spec_tree_path': 'robustness/placebo_tests.md',
            'outcome_var': outcome_var,
            'treatment_var': 'lnwba_lead1',
            'coefficient': result['coefficient'],
            'std_error': result['std_error'],
            't_stat': result['t_stat'],
            'p_value': result['p_value'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'n_obs': result['n_obs'],
            'r_squared': result['r_squared'],
            'coefficient_vector_json': json.dumps({'treatment': {'coef': result['coefficient'], 'se': result['std_error']}}),
            'sample_desc': 'Full sample',
            'fixed_effects': 'State + Year',
            'controls_desc': 'All controls + current treatment',
            'cluster_var': 'State',
            'model_type': 'Panel FE (placebo)',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })
        spec_count += 1

    # Randomized treatment (permutation)
    np.random.seed(42)
    df['lnwba_random'] = np.random.permutation(df['lnwba'].values)
    result = run_ols_clustered(
        df, outcome_var, 'lnwba_random',
        controls=all_controls,
        fe_vars=['state', 'year'],
        cluster_var='state_id'
    )
    if result:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/placebo/randomized_treatment',
            'spec_tree_path': 'robustness/placebo_tests.md',
            'outcome_var': outcome_var,
            'treatment_var': 'lnwba_random',
            'coefficient': result['coefficient'],
            'std_error': result['std_error'],
            't_stat': result['t_stat'],
            'p_value': result['p_value'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'n_obs': result['n_obs'],
            'r_squared': result['r_squared'],
            'coefficient_vector_json': json.dumps({'treatment': {'coef': result['coefficient'], 'se': result['std_error']}}),
            'sample_desc': 'Full sample',
            'fixed_effects': 'State + Year',
            'controls_desc': 'All controls',
            'cluster_var': 'State',
            'model_type': 'Panel FE (placebo)',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })
        spec_count += 1

    # ==========================================================================
    # ADDITIONAL SPECIFICATIONS TO REACH 50+
    # ==========================================================================
    print("Running additional specifications...")

    # Different time trends
    df['trend'] = df['year'] - df['year'].min()
    df['trend_sq'] = df['trend'] ** 2

    # With linear trend
    result = run_ols_clustered(
        df, outcome_var, treatment_var,
        controls=all_controls + ['trend'],
        fe_vars=['state'],
        cluster_var='state_id'
    )
    if result:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/funcform/linear_trend',
            'spec_tree_path': 'robustness/functional_form.md',
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': result['coefficient'],
            'std_error': result['std_error'],
            't_stat': result['t_stat'],
            'p_value': result['p_value'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'n_obs': result['n_obs'],
            'r_squared': result['r_squared'],
            'coefficient_vector_json': json.dumps({'treatment': {'coef': result['coefficient'], 'se': result['std_error']}}),
            'sample_desc': 'Full sample',
            'fixed_effects': 'State',
            'controls_desc': 'All controls + linear trend',
            'cluster_var': 'State',
            'model_type': 'Panel FE with trend',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })
        spec_count += 1

    # With quadratic trend
    result = run_ols_clustered(
        df, outcome_var, treatment_var,
        controls=all_controls + ['trend', 'trend_sq'],
        fe_vars=['state'],
        cluster_var='state_id'
    )
    if result:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/funcform/quadratic_trend',
            'spec_tree_path': 'robustness/functional_form.md',
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': result['coefficient'],
            'std_error': result['std_error'],
            't_stat': result['t_stat'],
            'p_value': result['p_value'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'n_obs': result['n_obs'],
            'r_squared': result['r_squared'],
            'coefficient_vector_json': json.dumps({'treatment': {'coef': result['coefficient'], 'se': result['std_error']}}),
            'sample_desc': 'Full sample',
            'fixed_effects': 'State',
            'controls_desc': 'All controls + quadratic trend',
            'cluster_var': 'State',
            'model_type': 'Panel FE with trend',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })
        spec_count += 1

    # Prime-age workers only (30-49)
    df_prime = df[df['age_i2'].isin([3, 4])].copy()
    result = run_ols_clustered(
        df_prime, outcome_var, treatment_var,
        controls=all_controls,
        fe_vars=['state', 'year'],
        cluster_var='state_id'
    )
    if result:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/sample/prime_age',
            'spec_tree_path': 'robustness/sample_restrictions.md',
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': result['coefficient'],
            'std_error': result['std_error'],
            't_stat': result['t_stat'],
            'p_value': result['p_value'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'n_obs': result['n_obs'],
            'r_squared': result['r_squared'],
            'coefficient_vector_json': json.dumps({'treatment': {'coef': result['coefficient'], 'se': result['std_error']}}),
            'sample_desc': 'Prime-age workers (30-49) only',
            'fixed_effects': 'State + Year',
            'controls_desc': 'All controls',
            'cluster_var': 'State',
            'model_type': 'Panel FE',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })
        spec_count += 1

    # Part 1 only (first half of year)
    df_part1 = df[df['part'] == 1].copy()
    result = run_ols_clustered(
        df_part1, outcome_var, treatment_var,
        controls=all_controls,
        fe_vars=['state', 'year'],
        cluster_var='state_id'
    )
    if result:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/sample/part1_only',
            'spec_tree_path': 'robustness/sample_restrictions.md',
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': result['coefficient'],
            'std_error': result['std_error'],
            't_stat': result['t_stat'],
            'p_value': result['p_value'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'n_obs': result['n_obs'],
            'r_squared': result['r_squared'],
            'coefficient_vector_json': json.dumps({'treatment': {'coef': result['coefficient'], 'se': result['std_error']}}),
            'sample_desc': 'First half of year only',
            'fixed_effects': 'State + Year',
            'controls_desc': 'All controls',
            'cluster_var': 'State',
            'model_type': 'Panel FE',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })
        spec_count += 1

    # Part 2 only (second half of year)
    df_part2 = df[df['part'] == 2].copy()
    result = run_ols_clustered(
        df_part2, outcome_var, treatment_var,
        controls=all_controls,
        fe_vars=['state', 'year'],
        cluster_var='state_id'
    )
    if result:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/sample/part2_only',
            'spec_tree_path': 'robustness/sample_restrictions.md',
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': result['coefficient'],
            'std_error': result['std_error'],
            't_stat': result['t_stat'],
            'p_value': result['p_value'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'n_obs': result['n_obs'],
            'r_squared': result['r_squared'],
            'coefficient_vector_json': json.dumps({'treatment': {'coef': result['coefficient'], 'se': result['std_error']}}),
            'sample_desc': 'Second half of year only',
            'fixed_effects': 'State + Year',
            'controls_desc': 'All controls',
            'cluster_var': 'State',
            'model_type': 'Panel FE',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })
        spec_count += 1

    # First difference specification
    df_sorted = df.sort_values(['state_id', 'period_int'])
    df['lnun_diff'] = df_sorted.groupby('state_id')['lnun'].diff()
    df['lnwba_diff'] = df_sorted.groupby('state_id')['lnwba'].diff()

    result = run_ols_clustered(
        df.dropna(subset=['lnun_diff', 'lnwba_diff']), 'lnun_diff', 'lnwba_diff',
        controls=None,
        fe_vars=['year'],
        cluster_var='state_id'
    )
    if result:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/estimation/first_difference',
            'spec_tree_path': 'methods/panel_fixed_effects.md#first-difference',
            'outcome_var': 'lnun_diff',
            'treatment_var': 'lnwba_diff',
            'coefficient': result['coefficient'],
            'std_error': result['std_error'],
            't_stat': result['t_stat'],
            'p_value': result['p_value'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'n_obs': result['n_obs'],
            'r_squared': result['r_squared'],
            'coefficient_vector_json': json.dumps({'treatment': {'coef': result['coefficient'], 'se': result['std_error']}}),
            'sample_desc': 'Full sample (first-differenced)',
            'fixed_effects': 'Year',
            'controls_desc': 'None (first differenced)',
            'cluster_var': 'State',
            'model_type': 'First Difference',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })
        spec_count += 1

    print(f"\nTotal specifications run: {spec_count}")

    return results


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(f"Specification Search: {PAPER_TITLE}")
    print(f"Paper ID: {PAPER_ID}")
    print("=" * 70)

    # Run specification search
    results = run_specification_search()

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    output_path = os.path.join(PACKAGE_DIR, 'specification_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    print(f"\nTotal specifications: {len(results_df)}")

    # Filter to only those with valid coefficients
    valid_results = results_df.dropna(subset=['coefficient', 'p_value'])

    if len(valid_results) > 0:
        print(f"Valid specifications: {len(valid_results)}")
        print(f"\nCoefficient statistics (treatment effect):")
        print(f"  Mean: {valid_results['coefficient'].mean():.4f}")
        print(f"  Median: {valid_results['coefficient'].median():.4f}")
        print(f"  Std Dev: {valid_results['coefficient'].std():.4f}")
        print(f"  Min: {valid_results['coefficient'].min():.4f}")
        print(f"  Max: {valid_results['coefficient'].max():.4f}")

        pos_coefs = (valid_results['coefficient'] > 0).sum()
        print(f"\nPositive coefficients: {pos_coefs} ({100*pos_coefs/len(valid_results):.1f}%)")

        sig_05 = (valid_results['p_value'] < 0.05).sum()
        sig_01 = (valid_results['p_value'] < 0.01).sum()
        print(f"Significant at 5%: {sig_05} ({100*sig_05/len(valid_results):.1f}%)")
        print(f"Significant at 1%: {sig_01} ({100*sig_01/len(valid_results):.1f}%)")

    print("\n" + "=" * 70)
    print("Specification search complete!")
    print("=" * 70)
