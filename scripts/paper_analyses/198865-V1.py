#!/usr/bin/env python3
"""
Specification Search: 198865-V1
"Estimating Models of Supply and Demand: Instruments and Covariance Restrictions"
by Alexander MacKay and Nathan H. Miller
American Economic Journal: Microeconomics

This paper develops methods for estimating supply and demand models using
covariance restrictions as an alternative to instrumental variables.

Method Classification:
- Primary: Instrumental Variables (for demand estimation)
- Paper develops alternative "Covariance Restriction" (CR) estimator
- Applications: Cement industry and Airlines

Key estimates: Price elasticities from demand models
"""

import pandas as pd
import numpy as np
import json
import warnings
from scipy.optimize import minimize_scalar, minimize
from scipy import stats
import statsmodels.api as sm

warnings.filterwarnings('ignore')

# Set paths
BASE_PATH = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search'
DATA_PATH = f'{BASE_PATH}/data/downloads/extracted/198865-V1'

# Paper metadata
PAPER_ID = '198865-V1'
JOURNAL = 'AEJ-Micro'
PAPER_TITLE = 'Estimating Models of Supply and Demand: Instruments and Covariance Restrictions'

# Results container
results = []

def add_result(spec_id, spec_tree_path, outcome_var, treatment_var,
               coefficient, std_error, t_stat, p_value, n_obs,
               r_squared=None, coefficient_vector_json=None,
               sample_desc='Full sample', fixed_effects='Market FE',
               controls_desc='None', cluster_var='epa_market',
               model_type='OLS', first_stage_F=None):
    """Add a specification result to the results list."""

    ci_lower = coefficient - 1.96 * std_error
    ci_upper = coefficient + 1.96 * std_error

    results.append({
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'outcome_var': outcome_var,
        'treatment_var': treatment_var,
        'coefficient': float(coefficient),
        'std_error': float(std_error),
        't_stat': float(t_stat),
        'p_value': float(p_value),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'n_obs': int(n_obs),
        'r_squared': float(r_squared) if r_squared is not None else None,
        'coefficient_vector_json': json.dumps(coefficient_vector_json) if coefficient_vector_json else None,
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
        'first_stage_F': float(first_stage_F) if first_stage_F is not None else None
    })

def run_ols_with_fe(df, y_col, x_cols, fe_col, cluster_col):
    """Run OLS with fixed effects and clustered standard errors."""
    # Create design matrix
    fe_dummies = pd.get_dummies(df[fe_col], prefix='fe', drop_first=True)
    X = pd.concat([df[x_cols], fe_dummies], axis=1).astype(float)
    X = sm.add_constant(X)
    y = df[y_col].astype(float)

    # Run regression
    model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df[cluster_col]})
    return model

def run_2sls(df, y_col, endog_col, instruments, fe_col, cluster_col, controls=None):
    """Run 2SLS IV estimation."""
    # First stage
    fe_dummies = pd.get_dummies(df[fe_col], prefix='fe', drop_first=True).astype(float)
    X_first = pd.concat([df[instruments].astype(float), fe_dummies], axis=1)
    if controls:
        X_first = pd.concat([X_first, df[controls].astype(float)], axis=1)
    X_first = sm.add_constant(X_first)

    first_stage = sm.OLS(df[endog_col].astype(float), X_first).fit()
    fitted_endog = first_stage.fittedvalues

    # First stage F-stat
    r_matrix = np.zeros((len(instruments), len(first_stage.params)))
    for i, instr in enumerate(instruments):
        idx = first_stage.params.index.get_loc(instr)
        r_matrix[i, idx] = 1
    f_test = first_stage.f_test(r_matrix)
    # Handle different statsmodels versions
    fval = f_test.fvalue
    if hasattr(fval, '__len__'):
        first_stage_F = float(fval[0][0]) if hasattr(fval[0], '__len__') else float(fval[0])
    else:
        first_stage_F = float(fval)

    # Second stage
    X_second = pd.concat([pd.Series(fitted_endog, name='fitted_endog'), fe_dummies], axis=1)
    if controls:
        X_second = pd.concat([X_second, df[controls].astype(float)], axis=1)
    X_second = sm.add_constant(X_second)

    second_stage = sm.OLS(df[y_col].astype(float), X_second).fit(
        cov_type='cluster', cov_kwds={'groups': df[cluster_col]})

    return second_stage, first_stage_F

# ============================================================================
# PART 1: CEMENT DEMAND ANALYSIS
# ============================================================================

print("=" * 80)
print("CEMENT DEMAND ANALYSIS")
print("=" * 80)

# Load cement data
cement_raw = pd.read_stata(f'{DATA_PATH}/data/cement/demand_est_data.dta')

# Apply the same transformations as the original code
variables_to_convert = ['epa_quantity', 'epa_customsvalue', 'epa_cif',
                        'epa_market_shipped', 'epa_market_price', 'epa_market_capacity']
for var in variables_to_convert:
    if var in cement_raw.columns:
        cement_raw.loc[cement_raw['year'] < 1993, var] = cement_raw.loc[cement_raw['year'] < 1993, var] * 0.907185

cement_raw.loc[cement_raw['year'] < 1994, 'production'] = cement_raw.loc[cement_raw['year'] < 1994, 'production'] * 0.907185

# Filter to years >= 1984 (as in paper)
cement = cement_raw[cement_raw['year'] >= 1984].copy()

# Create additional variables
cement['price'] = np.exp(cement['log_price'].astype(float))
cement['shipped'] = np.exp(cement['log_shipped'].astype(float))

# Drop missing
instruments = ['cpi_coal_price', 'cpi_electricity_price', 'cpi_ng_price', 'cpi_wage_hr']
demand_controls = ['log_population', 'log_units', 'log_unemployment_rate', 'log_buildings']
cement = cement.dropna(subset=['log_shipped', 'log_price'] + instruments)

# Ensure numeric types
for col in ['log_shipped', 'log_price', 'year'] + instruments + demand_controls:
    if col in cement.columns:
        cement[col] = pd.to_numeric(cement[col], errors='coerce')

cement = cement.dropna(subset=['log_shipped', 'log_price'] + instruments)
cement = cement.reset_index(drop=True)

print(f"Cement data: {len(cement)} observations, {cement['epa_market'].nunique()} markets")

# ============================================================================
# CEMENT: BASELINE SPECIFICATIONS
# ============================================================================

print("\n--- Baseline Specifications (Cement) ---")

# Spec 1: OLS Log-Linear Demand (baseline replication)
model_ols = run_ols_with_fe(cement, 'log_shipped', ['log_price'], 'epa_market', 'epa_market')
coef_ols = model_ols.params['log_price']
se_ols = model_ols.bse['log_price']
print(f"OLS Log-Linear: coef = {coef_ols:.4f}, se = {se_ols:.4f}")

add_result(
    spec_id='baseline',
    spec_tree_path='methods/instrumental_variables.md#baseline',
    outcome_var='log_shipped',
    treatment_var='log_price',
    coefficient=coef_ols,
    std_error=se_ols,
    t_stat=model_ols.tvalues['log_price'],
    p_value=model_ols.pvalues['log_price'],
    n_obs=len(cement),
    r_squared=model_ols.rsquared,
    sample_desc='Cement data 1984+',
    fixed_effects='Market FE',
    controls_desc='None',
    cluster_var='epa_market',
    model_type='OLS'
)

# Spec 2: 2SLS Log-Linear Demand
model_2sls, first_stage_F = run_2sls(cement, 'log_shipped', 'log_price', instruments, 'epa_market', 'epa_market')
coef_2sls = model_2sls.params['fitted_endog']
se_2sls = model_2sls.bse['fitted_endog']
print(f"2SLS Log-Linear: coef = {coef_2sls:.4f}, se = {se_2sls:.4f}, F = {first_stage_F:.2f}")

add_result(
    spec_id='iv/method/2sls',
    spec_tree_path='methods/instrumental_variables.md#estimation-method',
    outcome_var='log_shipped',
    treatment_var='log_price',
    coefficient=coef_2sls,
    std_error=se_2sls,
    t_stat=model_2sls.tvalues['fitted_endog'],
    p_value=model_2sls.pvalues['fitted_endog'],
    n_obs=len(cement),
    r_squared=model_2sls.rsquared,
    sample_desc='Cement data 1984+',
    fixed_effects='Market FE',
    controls_desc='IV: cost shifters',
    cluster_var='epa_market',
    model_type='2SLS',
    first_stage_F=first_stage_F
)

# Spec 3: OLS for comparison (same as baseline but explicit)
add_result(
    spec_id='iv/method/ols',
    spec_tree_path='methods/instrumental_variables.md#estimation-method',
    outcome_var='log_shipped',
    treatment_var='log_price',
    coefficient=coef_ols,
    std_error=se_ols,
    t_stat=model_ols.tvalues['log_price'],
    p_value=model_ols.pvalues['log_price'],
    n_obs=len(cement),
    r_squared=model_ols.rsquared,
    sample_desc='Cement data 1984+',
    fixed_effects='Market FE',
    controls_desc='OLS (ignoring endogeneity)',
    cluster_var='epa_market',
    model_type='OLS'
)

# ============================================================================
# CEMENT: INSTRUMENT VARIATIONS
# ============================================================================

print("\n--- Instrument Variations ---")

# Individual instruments
for instr_name in instruments:
    try:
        model_single, fs_F = run_2sls(cement, 'log_shipped', 'log_price', [instr_name], 'epa_market', 'epa_market')
        coef_s = model_single.params['fitted_endog']
        se_s = model_single.bse['fitted_endog']
        print(f"2SLS with {instr_name}: coef = {coef_s:.4f}, F = {fs_F:.2f}")

        add_result(
            spec_id=f'iv/instruments/single_{instr_name}',
            spec_tree_path='methods/instrumental_variables.md#instrument-sets',
            outcome_var='log_shipped',
            treatment_var='log_price',
            coefficient=coef_s,
            std_error=se_s,
            t_stat=model_single.tvalues['fitted_endog'],
            p_value=model_single.pvalues['fitted_endog'],
            n_obs=len(cement),
            r_squared=model_single.rsquared,
            sample_desc='Cement data 1984+',
            fixed_effects='Market FE',
            controls_desc=f'Single instrument: {instr_name}',
            cluster_var='epa_market',
            model_type='2SLS',
            first_stage_F=fs_F
        )
    except Exception as e:
        print(f"  Error with {instr_name}: {e}")

# Energy instruments only (subset)
energy_instruments = ['cpi_coal_price', 'cpi_electricity_price', 'cpi_ng_price']
model_energy, energy_F = run_2sls(cement, 'log_shipped', 'log_price', energy_instruments, 'epa_market', 'epa_market')
print(f"2SLS energy instruments: coef = {model_energy.params['fitted_endog']:.4f}, F = {energy_F:.2f}")

add_result(
    spec_id='iv/instruments/subset_energy',
    spec_tree_path='methods/instrumental_variables.md#instrument-sets',
    outcome_var='log_shipped',
    treatment_var='log_price',
    coefficient=model_energy.params['fitted_endog'],
    std_error=model_energy.bse['fitted_endog'],
    t_stat=model_energy.tvalues['fitted_endog'],
    p_value=model_energy.pvalues['fitted_endog'],
    n_obs=len(cement),
    r_squared=model_energy.rsquared,
    sample_desc='Cement data 1984+',
    fixed_effects='Market FE',
    controls_desc='Energy price instruments only',
    cluster_var='epa_market',
    model_type='2SLS',
    first_stage_F=energy_F
)

# ============================================================================
# CEMENT: FIXED EFFECTS VARIATIONS
# ============================================================================

print("\n--- Fixed Effects Variations ---")

# No fixed effects
X_nofe = cement[['log_price']].astype(float)
X_nofe = sm.add_constant(X_nofe)
model_nofe = sm.OLS(cement['log_shipped'].astype(float), X_nofe).fit(
    cov_type='cluster', cov_kwds={'groups': cement['epa_market']})
print(f"OLS no FE: coef = {model_nofe.params['log_price']:.4f}")

add_result(
    spec_id='iv/fe/none',
    spec_tree_path='methods/instrumental_variables.md#fixed-effects',
    outcome_var='log_shipped',
    treatment_var='log_price',
    coefficient=model_nofe.params['log_price'],
    std_error=model_nofe.bse['log_price'],
    t_stat=model_nofe.tvalues['log_price'],
    p_value=model_nofe.pvalues['log_price'],
    n_obs=len(cement),
    r_squared=model_nofe.rsquared,
    sample_desc='Cement data 1984+',
    fixed_effects='None',
    cluster_var='epa_market',
    model_type='OLS'
)

# Year FE only
model_yearfe = run_ols_with_fe(cement, 'log_shipped', ['log_price'], 'year', 'epa_market')
print(f"OLS year FE: coef = {model_yearfe.params['log_price']:.4f}")

add_result(
    spec_id='iv/fe/time',
    spec_tree_path='methods/instrumental_variables.md#fixed-effects',
    outcome_var='log_shipped',
    treatment_var='log_price',
    coefficient=model_yearfe.params['log_price'],
    std_error=model_yearfe.bse['log_price'],
    t_stat=model_yearfe.tvalues['log_price'],
    p_value=model_yearfe.pvalues['log_price'],
    n_obs=len(cement),
    r_squared=model_yearfe.rsquared,
    sample_desc='Cement data 1984+',
    fixed_effects='Year FE',
    cluster_var='epa_market',
    model_type='OLS'
)

# Two-way FE (market + year)
mkt_dum = pd.get_dummies(cement['epa_market'], prefix='mkt', drop_first=True).astype(float)
yr_dum = pd.get_dummies(cement['year'], prefix='yr', drop_first=True).astype(float)
X_twoway = pd.concat([cement[['log_price']].astype(float), mkt_dum, yr_dum], axis=1)
X_twoway = sm.add_constant(X_twoway)
model_twoway = sm.OLS(cement['log_shipped'].astype(float), X_twoway).fit(
    cov_type='cluster', cov_kwds={'groups': cement['epa_market']})
print(f"OLS two-way FE: coef = {model_twoway.params['log_price']:.4f}")

add_result(
    spec_id='iv/fe/twoway',
    spec_tree_path='methods/instrumental_variables.md#fixed-effects',
    outcome_var='log_shipped',
    treatment_var='log_price',
    coefficient=model_twoway.params['log_price'],
    std_error=model_twoway.bse['log_price'],
    t_stat=model_twoway.tvalues['log_price'],
    p_value=model_twoway.pvalues['log_price'],
    n_obs=len(cement),
    r_squared=model_twoway.rsquared,
    sample_desc='Cement data 1984+',
    fixed_effects='Market + Year FE',
    cluster_var='epa_market',
    model_type='OLS'
)

# ============================================================================
# CEMENT: CONTROL VARIATIONS
# ============================================================================

print("\n--- Control Variations ---")

# Drop rows with missing controls
cement_ctrl = cement.dropna(subset=demand_controls).copy()

# With all controls
fe_dum = pd.get_dummies(cement_ctrl['epa_market'], prefix='fe', drop_first=True).astype(float)
X_ctrl = pd.concat([cement_ctrl[['log_price'] + demand_controls].astype(float), fe_dum], axis=1)
X_ctrl = sm.add_constant(X_ctrl)
model_ctrl = sm.OLS(cement_ctrl['log_shipped'].astype(float), X_ctrl).fit(
    cov_type='cluster', cov_kwds={'groups': cement_ctrl['epa_market']})
print(f"OLS with controls: coef = {model_ctrl.params['log_price']:.4f}")

add_result(
    spec_id='iv/controls/full',
    spec_tree_path='methods/instrumental_variables.md#control-sets',
    outcome_var='log_shipped',
    treatment_var='log_price',
    coefficient=model_ctrl.params['log_price'],
    std_error=model_ctrl.bse['log_price'],
    t_stat=model_ctrl.tvalues['log_price'],
    p_value=model_ctrl.pvalues['log_price'],
    n_obs=len(cement_ctrl),
    r_squared=model_ctrl.rsquared,
    sample_desc='Cement data 1984+',
    fixed_effects='Market FE',
    controls_desc=', '.join(demand_controls),
    cluster_var='epa_market',
    model_type='OLS'
)

# Leave-one-out controls
for ctrl in demand_controls:
    remaining = [c for c in demand_controls if c != ctrl]
    X_loo = pd.concat([cement_ctrl[['log_price'] + remaining].astype(float), fe_dum], axis=1)
    X_loo = sm.add_constant(X_loo)
    model_loo = sm.OLS(cement_ctrl['log_shipped'].astype(float), X_loo).fit(
        cov_type='cluster', cov_kwds={'groups': cement_ctrl['epa_market']})
    print(f"LOO drop {ctrl}: coef = {model_loo.params['log_price']:.4f}")

    add_result(
        spec_id=f'robust/loo/drop_{ctrl}',
        spec_tree_path='robustness/leave_one_out.md',
        outcome_var='log_shipped',
        treatment_var='log_price',
        coefficient=model_loo.params['log_price'],
        std_error=model_loo.bse['log_price'],
        t_stat=model_loo.tvalues['log_price'],
        p_value=model_loo.pvalues['log_price'],
        n_obs=len(cement_ctrl),
        r_squared=model_loo.rsquared,
        sample_desc='Cement data 1984+',
        fixed_effects='Market FE',
        controls_desc=f'Drop {ctrl}',
        cluster_var='epa_market',
        model_type='OLS'
    )

# Add controls incrementally
for i, ctrl in enumerate(demand_controls):
    inc_ctrls = demand_controls[:i+1]
    X_inc = pd.concat([cement_ctrl[['log_price'] + inc_ctrls].astype(float), fe_dum], axis=1)
    X_inc = sm.add_constant(X_inc)
    model_inc = sm.OLS(cement_ctrl['log_shipped'].astype(float), X_inc).fit(
        cov_type='cluster', cov_kwds={'groups': cement_ctrl['epa_market']})
    print(f"Add {ctrl}: coef = {model_inc.params['log_price']:.4f}")

    add_result(
        spec_id=f'robust/control/add_{ctrl}',
        spec_tree_path='robustness/control_progression.md',
        outcome_var='log_shipped',
        treatment_var='log_price',
        coefficient=model_inc.params['log_price'],
        std_error=model_inc.bse['log_price'],
        t_stat=model_inc.tvalues['log_price'],
        p_value=model_inc.pvalues['log_price'],
        n_obs=len(cement_ctrl),
        r_squared=model_inc.rsquared,
        sample_desc='Cement data 1984+',
        fixed_effects='Market FE',
        controls_desc=f'Controls: {", ".join(inc_ctrls)}',
        cluster_var='epa_market',
        model_type='OLS'
    )

# ============================================================================
# CEMENT: SAMPLE RESTRICTIONS
# ============================================================================

print("\n--- Sample Restrictions ---")

# Early period
cement_early = cement[cement['year'] <= 1994].copy()
if len(cement_early) > 30:
    model_early = run_ols_with_fe(cement_early, 'log_shipped', ['log_price'], 'epa_market', 'epa_market')
    print(f"Early period: coef = {model_early.params['log_price']:.4f}, n = {len(cement_early)}")

    add_result(
        spec_id='robust/sample/early_period',
        spec_tree_path='robustness/sample_restrictions.md',
        outcome_var='log_shipped',
        treatment_var='log_price',
        coefficient=model_early.params['log_price'],
        std_error=model_early.bse['log_price'],
        t_stat=model_early.tvalues['log_price'],
        p_value=model_early.pvalues['log_price'],
        n_obs=len(cement_early),
        r_squared=model_early.rsquared,
        sample_desc='Cement 1984-1994',
        fixed_effects='Market FE',
        cluster_var='epa_market',
        model_type='OLS'
    )

# Late period
cement_late = cement[cement['year'] >= 1995].copy()
if len(cement_late) > 30:
    model_late = run_ols_with_fe(cement_late, 'log_shipped', ['log_price'], 'epa_market', 'epa_market')
    print(f"Late period: coef = {model_late.params['log_price']:.4f}, n = {len(cement_late)}")

    add_result(
        spec_id='robust/sample/late_period',
        spec_tree_path='robustness/sample_restrictions.md',
        outcome_var='log_shipped',
        treatment_var='log_price',
        coefficient=model_late.params['log_price'],
        std_error=model_late.bse['log_price'],
        t_stat=model_late.tvalues['log_price'],
        p_value=model_late.pvalues['log_price'],
        n_obs=len(cement_late),
        r_squared=model_late.rsquared,
        sample_desc='Cement 1995+',
        fixed_effects='Market FE',
        cluster_var='epa_market',
        model_type='OLS'
    )

# Drop individual markets
markets_to_drop = cement['epa_market'].value_counts().head(5).index.tolist()
for mkt in markets_to_drop:
    cement_drop = cement[cement['epa_market'] != mkt].copy()
    if len(cement_drop) > 30:
        model_drop = run_ols_with_fe(cement_drop, 'log_shipped', ['log_price'], 'epa_market', 'epa_market')
        mkt_clean = mkt.replace(' ', '_').replace('.', '')
        print(f"Drop {mkt}: coef = {model_drop.params['log_price']:.4f}")

        add_result(
            spec_id=f'robust/sample/drop_{mkt_clean}',
            spec_tree_path='robustness/sample_restrictions.md',
            outcome_var='log_shipped',
            treatment_var='log_price',
            coefficient=model_drop.params['log_price'],
            std_error=model_drop.bse['log_price'],
            t_stat=model_drop.tvalues['log_price'],
            p_value=model_drop.pvalues['log_price'],
            n_obs=len(cement_drop),
            r_squared=model_drop.rsquared,
            sample_desc=f'Drop {mkt}',
            fixed_effects='Market FE',
            cluster_var='epa_market',
            model_type='OLS'
        )

# Winsorize
cement_win = cement.copy()
for col in ['log_shipped', 'log_price']:
    cement_win[col] = cement_win[col].clip(
        lower=cement_win[col].quantile(0.01),
        upper=cement_win[col].quantile(0.99))
model_win = run_ols_with_fe(cement_win, 'log_shipped', ['log_price'], 'epa_market', 'epa_market')
print(f"Winsorize 1%: coef = {model_win.params['log_price']:.4f}")

add_result(
    spec_id='robust/sample/winsor_1pct',
    spec_tree_path='robustness/sample_restrictions.md',
    outcome_var='log_shipped',
    treatment_var='log_price',
    coefficient=model_win.params['log_price'],
    std_error=model_win.bse['log_price'],
    t_stat=model_win.tvalues['log_price'],
    p_value=model_win.pvalues['log_price'],
    n_obs=len(cement_win),
    r_squared=model_win.rsquared,
    sample_desc='Winsorized 1%/99%',
    fixed_effects='Market FE',
    cluster_var='epa_market',
    model_type='OLS'
)

# Trim
cement_trim = cement[
    (cement['log_shipped'] > cement['log_shipped'].quantile(0.05)) &
    (cement['log_shipped'] < cement['log_shipped'].quantile(0.95))
].copy()
if len(cement_trim) > 30:
    model_trim = run_ols_with_fe(cement_trim, 'log_shipped', ['log_price'], 'epa_market', 'epa_market')
    print(f"Trim 5%: coef = {model_trim.params['log_price']:.4f}, n = {len(cement_trim)}")

    add_result(
        spec_id='robust/sample/trim_5pct',
        spec_tree_path='robustness/sample_restrictions.md',
        outcome_var='log_shipped',
        treatment_var='log_price',
        coefficient=model_trim.params['log_price'],
        std_error=model_trim.bse['log_price'],
        t_stat=model_trim.tvalues['log_price'],
        p_value=model_trim.pvalues['log_price'],
        n_obs=len(cement_trim),
        r_squared=model_trim.rsquared,
        sample_desc='Trimmed 5%/95%',
        fixed_effects='Market FE',
        cluster_var='epa_market',
        model_type='OLS'
    )

# ============================================================================
# CEMENT: CLUSTERING VARIATIONS
# ============================================================================

print("\n--- Clustering Variations ---")

# HC1 robust (no clustering)
model_hc1 = run_ols_with_fe(cement, 'log_shipped', ['log_price'], 'epa_market', 'epa_market')
# Re-run with HC1
fe_dum_base = pd.get_dummies(cement['epa_market'], prefix='fe', drop_first=True).astype(float)
X_base = pd.concat([cement[['log_price']].astype(float), fe_dum_base], axis=1)
X_base = sm.add_constant(X_base)
model_hc1 = sm.OLS(cement['log_shipped'].astype(float), X_base).fit(cov_type='HC1')
print(f"HC1 robust: coef = {model_hc1.params['log_price']:.4f}, se = {model_hc1.bse['log_price']:.4f}")

add_result(
    spec_id='robust/cluster/none',
    spec_tree_path='robustness/clustering_variations.md',
    outcome_var='log_shipped',
    treatment_var='log_price',
    coefficient=model_hc1.params['log_price'],
    std_error=model_hc1.bse['log_price'],
    t_stat=model_hc1.tvalues['log_price'],
    p_value=model_hc1.pvalues['log_price'],
    n_obs=len(cement),
    r_squared=model_hc1.rsquared,
    sample_desc='Cement data 1984+',
    fixed_effects='Market FE',
    controls_desc='Heteroskedasticity-robust SE',
    cluster_var='None (HC1)',
    model_type='OLS'
)

# Cluster by year
model_yr_cl = sm.OLS(cement['log_shipped'].astype(float), X_base).fit(
    cov_type='cluster', cov_kwds={'groups': cement['year']})
print(f"Cluster by year: se = {model_yr_cl.bse['log_price']:.4f}")

add_result(
    spec_id='robust/cluster/time',
    spec_tree_path='robustness/clustering_variations.md',
    outcome_var='log_shipped',
    treatment_var='log_price',
    coefficient=model_yr_cl.params['log_price'],
    std_error=model_yr_cl.bse['log_price'],
    t_stat=model_yr_cl.tvalues['log_price'],
    p_value=model_yr_cl.pvalues['log_price'],
    n_obs=len(cement),
    r_squared=model_yr_cl.rsquared,
    sample_desc='Cement data 1984+',
    fixed_effects='Market FE',
    cluster_var='year',
    model_type='OLS'
)

# ============================================================================
# CEMENT: FUNCTIONAL FORM VARIATIONS
# ============================================================================

print("\n--- Functional Form Variations ---")

# Level-level
cement_lev = cement.copy()
cement_lev['shipped_level'] = np.exp(cement_lev['log_shipped'].astype(float))
cement_lev['price_level'] = np.exp(cement_lev['log_price'].astype(float))
X_lev = pd.concat([cement_lev[['price_level']].astype(float), fe_dum_base], axis=1)
X_lev = sm.add_constant(X_lev)
model_lev = sm.OLS(cement_lev['shipped_level'].astype(float), X_lev).fit(
    cov_type='cluster', cov_kwds={'groups': cement_lev['epa_market']})
print(f"Level-level: coef = {model_lev.params['price_level']:.4f}")

add_result(
    spec_id='robust/form/y_level',
    spec_tree_path='robustness/functional_form.md',
    outcome_var='shipped_level',
    treatment_var='price_level',
    coefficient=model_lev.params['price_level'],
    std_error=model_lev.bse['price_level'],
    t_stat=model_lev.tvalues['price_level'],
    p_value=model_lev.pvalues['price_level'],
    n_obs=len(cement_lev),
    r_squared=model_lev.rsquared,
    sample_desc='Cement 1984+ (levels)',
    fixed_effects='Market FE',
    controls_desc='Level-level specification',
    cluster_var='epa_market',
    model_type='OLS'
)

# Semi-elasticity (log Q, level P)
X_semi = pd.concat([cement_lev[['price_level']].astype(float), fe_dum_base], axis=1)
X_semi = sm.add_constant(X_semi)
model_semi = sm.OLS(cement_lev['log_shipped'].astype(float), X_semi).fit(
    cov_type='cluster', cov_kwds={'groups': cement_lev['epa_market']})
print(f"Semi-elasticity: coef = {model_semi.params['price_level']:.6f}")

add_result(
    spec_id='robust/form/log_level',
    spec_tree_path='robustness/functional_form.md',
    outcome_var='log_shipped',
    treatment_var='price_level',
    coefficient=model_semi.params['price_level'],
    std_error=model_semi.bse['price_level'],
    t_stat=model_semi.tvalues['price_level'],
    p_value=model_semi.pvalues['price_level'],
    n_obs=len(cement_lev),
    r_squared=model_semi.rsquared,
    sample_desc='Cement data 1984+',
    fixed_effects='Market FE',
    controls_desc='Log-level (semi-elasticity)',
    cluster_var='epa_market',
    model_type='OLS'
)

# Quadratic
cement_quad = cement.copy()
cement_quad['log_price_sq'] = cement_quad['log_price'].astype(float) ** 2
X_quad = pd.concat([cement_quad[['log_price', 'log_price_sq']].astype(float), fe_dum_base], axis=1)
X_quad = sm.add_constant(X_quad)
model_quad = sm.OLS(cement_quad['log_shipped'].astype(float), X_quad).fit(
    cov_type='cluster', cov_kwds={'groups': cement_quad['epa_market']})
print(f"Quadratic: linear = {model_quad.params['log_price']:.4f}, sq = {model_quad.params['log_price_sq']:.4f}")

add_result(
    spec_id='robust/form/quadratic',
    spec_tree_path='robustness/functional_form.md',
    outcome_var='log_shipped',
    treatment_var='log_price',
    coefficient=model_quad.params['log_price'],
    std_error=model_quad.bse['log_price'],
    t_stat=model_quad.tvalues['log_price'],
    p_value=model_quad.pvalues['log_price'],
    n_obs=len(cement_quad),
    r_squared=model_quad.rsquared,
    sample_desc='Cement data 1984+',
    fixed_effects='Market FE',
    controls_desc='Quadratic in log price',
    cluster_var='epa_market',
    model_type='OLS'
)

# ============================================================================
# PART 2: AIRLINES DEMAND ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("AIRLINES DEMAND ANALYSIS")
print("=" * 80)

# Load airlines data
airlines_raw = pd.read_stata(f'{DATA_PATH}/data/airlines/eco2901_problemset_01_2012_airlines_data.dta')
airlines = airlines_raw.copy()

# Market size
airlines['msize'] = 1000 * (airlines['pop04_origin'].astype(float) + airlines['pop04_dest'].astype(float)) / 2

# Market share
airlines['share'] = airlines['passengers'].astype(float) / airlines['msize']

# Route-quarter identifier
airlines['route_quarter'] = airlines['route_id'].astype(str) + '_' + airlines['quarter'].astype(str)

# Outside good share
share0 = airlines.groupby('route_quarter')['share'].transform('sum')
airlines['share0'] = 1 - share0

# Group shares (direct vs connecting)
airlines['share_stop'] = airlines.groupby('route_quarter').apply(
    lambda x: (x['share'] * (1 - x['direct'])).sum(), include_groups=False
).reindex(airlines['route_quarter']).values
airlines['share_nonstop'] = airlines.groupby('route_quarter').apply(
    lambda x: (x['share'] * x['direct']).sum(), include_groups=False
).reindex(airlines['route_quarter']).values

airlines['share_group'] = np.where(airlines['direct'] == 1, airlines['share_nonstop'], airlines['share_stop'])
airlines['share_within'] = airlines['share'] / airlines['share_group']

# Log-odds ratios
airlines['log_s_s0'] = np.log(airlines['share'] / airlines['share0'])
airlines['log_s_within'] = np.log(airlines['share_within'])

# Scale variables
airlines['price100'] = airlines['price'].astype(float) / 100
airlines['miles1000'] = airlines['avg_miles'].astype(float) / 1000
airlines['hub_avg'] = (airlines['HUB_origin'].astype(float) + airlines['HUB_dest'].astype(float)) / 2

# Clean data
airlines = airlines.replace([np.inf, -np.inf], np.nan)
airlines = airlines.dropna(subset=['log_s_s0', 'log_s_within', 'price100'])
airlines = airlines.reset_index(drop=True)

# Market and firm IDs
airlines['mktid'] = pd.Categorical(airlines['route_quarter']).codes
airlines['firmid'] = pd.Categorical(airlines['airline']).codes

print(f"Airlines data: {len(airlines)} observations")

# Control variables
air_controls = ['direct', 'HUB_origin', 'HUB_dest', 'miles1000']

# ============================================================================
# AIRLINES: BASELINE NESTED LOGIT
# ============================================================================

print("\n--- Airlines Baseline (Nested Logit) ---")

# OLS nested logit with market (route_quarter) FE
# Within-transform the data
air_vars = ['log_s_s0', 'price100', 'log_s_within'] + air_controls
airlines_dm = airlines[air_vars + ['mktid', 'route_id', 'airline']].copy()

# Demean by market
for col in air_vars:
    airlines_dm[col] = airlines_dm.groupby('mktid')[col].transform(lambda x: x - x.mean())

# Create airline dummies
airline_dum = pd.get_dummies(airlines['airline'], prefix='air', drop_first=True).astype(float)
airline_dum_dm = airline_dum.groupby(airlines['mktid']).transform(lambda x: x - x.mean())

# Run regression
X_air = pd.concat([airlines_dm[['price100', 'log_s_within'] + air_controls].astype(float), airline_dum_dm], axis=1)
X_air = sm.add_constant(X_air)
y_air = airlines_dm['log_s_s0'].astype(float)

model_air_ols = sm.OLS(y_air, X_air).fit(cov_type='cluster', cov_kwds={'groups': airlines['route_id']})
print(f"Airlines OLS: alpha = {model_air_ols.params['price100']:.4f}, sigma = {model_air_ols.params['log_s_within']:.4f}")

add_result(
    spec_id='baseline_airlines',
    spec_tree_path='methods/instrumental_variables.md#baseline',
    outcome_var='log_s_s0',
    treatment_var='price100',
    coefficient=model_air_ols.params['price100'],
    std_error=model_air_ols.bse['price100'],
    t_stat=model_air_ols.tvalues['price100'],
    p_value=model_air_ols.pvalues['price100'],
    n_obs=len(airlines),
    r_squared=model_air_ols.rsquared,
    sample_desc='Airlines 2004 Q1-Q4',
    fixed_effects='Market (route-quarter) FE',
    controls_desc='direct, HUB, miles, airline FE',
    cluster_var='route_id',
    model_type='OLS Nested Logit'
)

# ============================================================================
# AIRLINES: CONTROL VARIATIONS
# ============================================================================

print("\n--- Airlines Control Variations ---")

# No product controls (just price and sigma)
X_air_noctl = pd.concat([airlines_dm[['price100', 'log_s_within']].astype(float), airline_dum_dm], axis=1)
X_air_noctl = sm.add_constant(X_air_noctl)
model_air_noctl = sm.OLS(y_air, X_air_noctl).fit(cov_type='cluster', cov_kwds={'groups': airlines['route_id']})
print(f"Airlines no controls: alpha = {model_air_noctl.params['price100']:.4f}")

add_result(
    spec_id='iv/controls/none_airlines',
    spec_tree_path='methods/instrumental_variables.md#control-sets',
    outcome_var='log_s_s0',
    treatment_var='price100',
    coefficient=model_air_noctl.params['price100'],
    std_error=model_air_noctl.bse['price100'],
    t_stat=model_air_noctl.tvalues['price100'],
    p_value=model_air_noctl.pvalues['price100'],
    n_obs=len(airlines),
    r_squared=model_air_noctl.rsquared,
    sample_desc='Airlines 2004',
    fixed_effects='Market FE',
    controls_desc='No product controls',
    cluster_var='route_id',
    model_type='OLS Nested Logit'
)

# Leave-one-out controls
for ctrl in air_controls:
    remaining = [c for c in air_controls if c != ctrl]
    X_loo = pd.concat([airlines_dm[['price100', 'log_s_within'] + remaining].astype(float), airline_dum_dm], axis=1)
    X_loo = sm.add_constant(X_loo)
    model_loo = sm.OLS(y_air, X_loo).fit(cov_type='cluster', cov_kwds={'groups': airlines['route_id']})
    print(f"Airlines drop {ctrl}: alpha = {model_loo.params['price100']:.4f}")

    add_result(
        spec_id=f'robust/loo/drop_{ctrl}_airlines',
        spec_tree_path='robustness/leave_one_out.md',
        outcome_var='log_s_s0',
        treatment_var='price100',
        coefficient=model_loo.params['price100'],
        std_error=model_loo.bse['price100'],
        t_stat=model_loo.tvalues['price100'],
        p_value=model_loo.pvalues['price100'],
        n_obs=len(airlines),
        r_squared=model_loo.rsquared,
        sample_desc='Airlines 2004',
        fixed_effects='Market FE',
        controls_desc=f'Drop {ctrl}',
        cluster_var='route_id',
        model_type='OLS Nested Logit'
    )

# ============================================================================
# AIRLINES: SAMPLE RESTRICTIONS
# ============================================================================

print("\n--- Airlines Sample Restrictions ---")

# By quarter
for q in sorted(airlines['quarter'].unique()):
    air_q = airlines[airlines['quarter'] == q].copy()
    if len(air_q) > 100:
        # Demean
        for col in air_vars:
            air_q[col] = air_q.groupby('mktid')[col].transform(lambda x: x - x.mean())

        air_dum_q = pd.get_dummies(airlines[airlines['quarter'] == q]['airline'], prefix='air', drop_first=True).astype(float)
        air_dum_q_dm = air_dum_q.groupby(airlines[airlines['quarter'] == q]['mktid']).transform(lambda x: x - x.mean())

        X_q = pd.concat([air_q[['price100', 'log_s_within'] + air_controls].astype(float), air_dum_q_dm], axis=1)
        X_q = sm.add_constant(X_q)
        y_q = air_q['log_s_s0'].astype(float)

        model_q = sm.OLS(y_q, X_q).fit(cov_type='cluster', cov_kwds={'groups': airlines[airlines['quarter'] == q]['route_id']})
        print(f"Quarter {int(q)}: alpha = {model_q.params['price100']:.4f}")

        add_result(
            spec_id=f'robust/sample/quarter_{int(q)}',
            spec_tree_path='robustness/sample_restrictions.md',
            outcome_var='log_s_s0',
            treatment_var='price100',
            coefficient=model_q.params['price100'],
            std_error=model_q.bse['price100'],
            t_stat=model_q.tvalues['price100'],
            p_value=model_q.pvalues['price100'],
            n_obs=len(air_q),
            r_squared=model_q.rsquared,
            sample_desc=f'Airlines Q{int(q)} 2004',
            fixed_effects='Market FE',
            cluster_var='route_id',
            model_type='OLS Nested Logit'
        )

# Direct flights only
air_direct = airlines[airlines['direct'] == 1].copy()
if len(air_direct) > 100:
    for col in air_vars:
        air_direct[col] = air_direct.groupby('mktid')[col].transform(lambda x: x - x.mean())

    X_direct = pd.concat([air_direct[['price100', 'HUB_origin', 'HUB_dest', 'miles1000']].astype(float)], axis=1)
    X_direct = sm.add_constant(X_direct)
    y_direct = air_direct['log_s_s0'].astype(float)

    model_direct = sm.OLS(y_direct, X_direct).fit(cov_type='cluster', cov_kwds={'groups': air_direct['route_id']})
    print(f"Direct flights only: alpha = {model_direct.params['price100']:.4f}")

    add_result(
        spec_id='robust/sample/direct_only',
        spec_tree_path='robustness/sample_restrictions.md',
        outcome_var='log_s_s0',
        treatment_var='price100',
        coefficient=model_direct.params['price100'],
        std_error=model_direct.bse['price100'],
        t_stat=model_direct.tvalues['price100'],
        p_value=model_direct.pvalues['price100'],
        n_obs=len(air_direct),
        r_squared=model_direct.rsquared,
        sample_desc='Direct flights only',
        fixed_effects='Market FE',
        cluster_var='route_id',
        model_type='OLS Logit'
    )

# Connecting flights only
air_connect = airlines[airlines['direct'] == 0].copy()
if len(air_connect) > 100:
    for col in air_vars:
        air_connect[col] = air_connect.groupby('mktid')[col].transform(lambda x: x - x.mean())

    X_conn = pd.concat([air_connect[['price100', 'HUB_origin', 'HUB_dest', 'miles1000']].astype(float)], axis=1)
    X_conn = sm.add_constant(X_conn)
    y_conn = air_connect['log_s_s0'].astype(float)

    model_conn = sm.OLS(y_conn, X_conn).fit(cov_type='cluster', cov_kwds={'groups': air_connect['route_id']})
    print(f"Connecting flights only: alpha = {model_conn.params['price100']:.4f}")

    add_result(
        spec_id='robust/sample/connecting_only',
        spec_tree_path='robustness/sample_restrictions.md',
        outcome_var='log_s_s0',
        treatment_var='price100',
        coefficient=model_conn.params['price100'],
        std_error=model_conn.bse['price100'],
        t_stat=model_conn.tvalues['price100'],
        p_value=model_conn.pvalues['price100'],
        n_obs=len(air_connect),
        r_squared=model_conn.rsquared,
        sample_desc='Connecting flights only',
        fixed_effects='Market FE',
        cluster_var='route_id',
        model_type='OLS Logit'
    )

# Drop major airlines
for airline_drop in ['American Airlines (AA)', 'Delta (DL)', 'Southwest (WN)']:
    air_drop = airlines[airlines['airline'] != airline_drop].copy()
    if len(air_drop) > 100:
        for col in air_vars:
            air_drop[col] = air_drop.groupby('mktid')[col].transform(lambda x: x - x.mean())

        air_dum_drop = pd.get_dummies(air_drop['airline'], prefix='air', drop_first=True).astype(float)
        air_dum_drop_dm = air_dum_drop.groupby(air_drop['mktid']).transform(lambda x: x - x.mean())

        X_drop = pd.concat([air_drop[['price100', 'log_s_within'] + air_controls].astype(float), air_dum_drop_dm], axis=1)
        X_drop = sm.add_constant(X_drop)
        y_drop = air_drop['log_s_s0'].astype(float)

        model_drop = sm.OLS(y_drop, X_drop).fit(cov_type='cluster', cov_kwds={'groups': air_drop['route_id']})
        airline_clean = airline_drop.replace(' ', '_').replace('(', '').replace(')', '')
        print(f"Drop {airline_drop}: alpha = {model_drop.params['price100']:.4f}")

        add_result(
            spec_id=f'robust/sample/drop_{airline_clean}',
            spec_tree_path='robustness/sample_restrictions.md',
            outcome_var='log_s_s0',
            treatment_var='price100',
            coefficient=model_drop.params['price100'],
            std_error=model_drop.bse['price100'],
            t_stat=model_drop.tvalues['price100'],
            p_value=model_drop.pvalues['price100'],
            n_obs=len(air_drop),
            r_squared=model_drop.rsquared,
            sample_desc=f'Drop {airline_drop}',
            fixed_effects='Market FE',
            cluster_var='route_id',
            model_type='OLS Nested Logit'
        )

# ============================================================================
# AIRLINES: HETEROGENEITY
# ============================================================================

print("\n--- Airlines Heterogeneity ---")

# Legacy vs LCC carriers
airlines['is_legacy'] = airlines['airline'].isin([
    'American Airlines (AA)', 'Continental (CO)', 'Delta (DL)',
    'Northwest (NW)', 'US Airways (US)', 'United (UA)'
]).astype(int)

for carrier_type, label in [(1, 'legacy'), (0, 'lcc')]:
    air_type = airlines[airlines['is_legacy'] == carrier_type].copy()
    if len(air_type) > 100:
        for col in air_vars:
            air_type[col] = air_type.groupby('mktid')[col].transform(lambda x: x - x.mean())

        air_dum_type = pd.get_dummies(air_type['airline'], prefix='air', drop_first=True).astype(float)
        air_dum_type_dm = air_dum_type.groupby(air_type['mktid']).transform(lambda x: x - x.mean())

        X_type = pd.concat([air_type[['price100', 'log_s_within'] + air_controls].astype(float), air_dum_type_dm], axis=1)
        X_type = sm.add_constant(X_type)
        y_type = air_type['log_s_s0'].astype(float)

        model_type = sm.OLS(y_type, X_type).fit(cov_type='cluster', cov_kwds={'groups': air_type['route_id']})
        print(f"{label.upper()} carriers: alpha = {model_type.params['price100']:.4f}")

        add_result(
            spec_id=f'robust/heterogeneity/{label}_carriers',
            spec_tree_path='robustness/heterogeneity.md',
            outcome_var='log_s_s0',
            treatment_var='price100',
            coefficient=model_type.params['price100'],
            std_error=model_type.bse['price100'],
            t_stat=model_type.tvalues['price100'],
            p_value=model_type.pvalues['price100'],
            n_obs=len(air_type),
            r_squared=model_type.rsquared,
            sample_desc=f'{label.upper()} carriers only',
            fixed_effects='Market FE',
            cluster_var='route_id',
            model_type='OLS Nested Logit'
        )

# By route distance
airlines['long_haul'] = (airlines['avg_miles'] > airlines['avg_miles'].median()).astype(int)
for dist_type, label in [(1, 'long_haul'), (0, 'short_haul')]:
    air_dist = airlines[airlines['long_haul'] == dist_type].copy()
    if len(air_dist) > 100:
        for col in air_vars:
            air_dist[col] = air_dist.groupby('mktid')[col].transform(lambda x: x - x.mean())

        air_dum_dist = pd.get_dummies(air_dist['airline'], prefix='air', drop_first=True).astype(float)
        air_dum_dist_dm = air_dum_dist.groupby(air_dist['mktid']).transform(lambda x: x - x.mean())

        X_dist = pd.concat([air_dist[['price100', 'log_s_within'] + air_controls].astype(float), air_dum_dist_dm], axis=1)
        X_dist = sm.add_constant(X_dist)
        y_dist = air_dist['log_s_s0'].astype(float)

        model_dist = sm.OLS(y_dist, X_dist).fit(cov_type='cluster', cov_kwds={'groups': air_dist['route_id']})
        print(f"{label}: alpha = {model_dist.params['price100']:.4f}")

        add_result(
            spec_id=f'robust/heterogeneity/{label}',
            spec_tree_path='robustness/heterogeneity.md',
            outcome_var='log_s_s0',
            treatment_var='price100',
            coefficient=model_dist.params['price100'],
            std_error=model_dist.bse['price100'],
            t_stat=model_dist.tvalues['price100'],
            p_value=model_dist.pvalues['price100'],
            n_obs=len(air_dist),
            r_squared=model_dist.rsquared,
            sample_desc=f'{label.replace("_", " ").title()} routes',
            fixed_effects='Market FE',
            cluster_var='route_id',
            model_type='OLS Nested Logit'
        )

# ============================================================================
# AIRLINES: CLUSTERING VARIATIONS
# ============================================================================

print("\n--- Airlines Clustering Variations ---")

# Cluster by airline
model_air_cl = sm.OLS(y_air, X_air).fit(cov_type='cluster', cov_kwds={'groups': airlines['airline']})
print(f"Cluster by airline: se = {model_air_cl.bse['price100']:.4f}")

add_result(
    spec_id='robust/cluster/airline',
    spec_tree_path='robustness/clustering_variations.md',
    outcome_var='log_s_s0',
    treatment_var='price100',
    coefficient=model_air_cl.params['price100'],
    std_error=model_air_cl.bse['price100'],
    t_stat=model_air_cl.tvalues['price100'],
    p_value=model_air_cl.pvalues['price100'],
    n_obs=len(airlines),
    r_squared=model_air_cl.rsquared,
    sample_desc='Airlines 2004',
    fixed_effects='Market FE',
    cluster_var='airline',
    model_type='OLS Nested Logit'
)

# HC1 robust
model_air_hc1 = sm.OLS(y_air, X_air).fit(cov_type='HC1')
print(f"HC1 robust: se = {model_air_hc1.bse['price100']:.4f}")

add_result(
    spec_id='robust/cluster/none_airlines',
    spec_tree_path='robustness/clustering_variations.md',
    outcome_var='log_s_s0',
    treatment_var='price100',
    coefficient=model_air_hc1.params['price100'],
    std_error=model_air_hc1.bse['price100'],
    t_stat=model_air_hc1.tvalues['price100'],
    p_value=model_air_hc1.pvalues['price100'],
    n_obs=len(airlines),
    r_squared=model_air_hc1.rsquared,
    sample_desc='Airlines 2004',
    fixed_effects='Market FE',
    cluster_var='None (HC1)',
    model_type='OLS Nested Logit'
)

# ============================================================================
# AIRLINES: FUNCTIONAL FORM
# ============================================================================

print("\n--- Airlines Functional Form ---")

# Simple logit (no nesting, sigma=0)
X_simple = pd.concat([airlines_dm[['price100'] + air_controls].astype(float), airline_dum_dm], axis=1)
X_simple = sm.add_constant(X_simple)
model_simple = sm.OLS(y_air, X_simple).fit(cov_type='cluster', cov_kwds={'groups': airlines['route_id']})
print(f"Simple logit (no nest): alpha = {model_simple.params['price100']:.4f}")

add_result(
    spec_id='robust/form/simple_logit',
    spec_tree_path='robustness/functional_form.md',
    outcome_var='log_s_s0',
    treatment_var='price100',
    coefficient=model_simple.params['price100'],
    std_error=model_simple.bse['price100'],
    t_stat=model_simple.tvalues['price100'],
    p_value=model_simple.pvalues['price100'],
    n_obs=len(airlines),
    r_squared=model_simple.rsquared,
    sample_desc='Airlines 2004',
    fixed_effects='Market FE',
    controls_desc='Simple logit (sigma=0)',
    cluster_var='route_id',
    model_type='OLS Simple Logit'
)

# Price in levels
airlines_dm['price_level'] = airlines['price'].astype(float)
airlines_dm['price_level'] = airlines_dm.groupby(airlines['mktid'])['price_level'].transform(lambda x: x - x.mean())

X_lev_air = pd.concat([airlines_dm[['price_level', 'log_s_within'] + air_controls].astype(float), airline_dum_dm], axis=1)
X_lev_air = sm.add_constant(X_lev_air)
model_lev_air = sm.OLS(y_air, X_lev_air).fit(cov_type='cluster', cov_kwds={'groups': airlines['route_id']})
print(f"Price in levels: alpha = {model_lev_air.params['price_level']:.6f}")

add_result(
    spec_id='robust/form/price_levels',
    spec_tree_path='robustness/functional_form.md',
    outcome_var='log_s_s0',
    treatment_var='price',
    coefficient=model_lev_air.params['price_level'],
    std_error=model_lev_air.bse['price_level'],
    t_stat=model_lev_air.tvalues['price_level'],
    p_value=model_lev_air.pvalues['price_level'],
    n_obs=len(airlines),
    r_squared=model_lev_air.rsquared,
    sample_desc='Airlines 2004',
    fixed_effects='Market FE',
    controls_desc='Price in levels',
    cluster_var='route_id',
    model_type='OLS Nested Logit'
)

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
output_path = f'{DATA_PATH}/specification_results.csv'
results_df.to_csv(output_path, index=False)
print(f"Saved {len(results_df)} specifications to {output_path}")

# Summary statistics
print(f"\n--- Summary Statistics ---")
print(f"Total specifications: {len(results_df)}")
print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
print(f"Negative coefficients: {(results_df['coefficient'] < 0).sum()} ({100*(results_df['coefficient'] < 0).mean():.1f}%)")
print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")

# Breakdown by category
print("\n--- Breakdown by Category ---")
results_df['category'] = results_df['spec_id'].apply(lambda x: x.split('/')[0] if '/' in x else 'baseline')
for cat in sorted(results_df['category'].unique()):
    cat_df = results_df[results_df['category'] == cat]
    sig_rate = 100 * (cat_df['p_value'] < 0.05).mean()
    print(f"{cat}: {len(cat_df)} specs, {sig_rate:.1f}% sig at 5%")
