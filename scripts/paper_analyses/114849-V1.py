"""
Specification Search for Paper 114849-V1
Title: Does Prison Make People Criminals? Evidence from Italian Collective Pardons
Journal: AEJ Applied

Main Hypothesis: Prison population has a causal effect on crime rates (deterrence/incapacitation)
Identification: Instrumental Variables using collective pardons as exogenous variation
Treatment: Change in prison population (lwchange_jail)
Outcome: Change in crime rate (lchange_all)
Instrument: Fraction of pardoned inmates (lwexit_free_amnesty)

Method: Instrumental Variables (2SLS)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up paths
PACKAGE_DIR = Path('/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/114849-V1')
DATA_DIR = PACKAGE_DIR / 'data_AEJ'
OUTPUT_DIR = PACKAGE_DIR

# Paper metadata
PAPER_ID = '114849-V1'
JOURNAL = 'AEJ-Applied'
PAPER_TITLE = 'Does Prison Make People Criminals? Evidence from Italian Collective Pardons'

# Load data
df = pd.read_stata(DATA_DIR / 'master_panel.dta')

# =============================================================================
# DATA PREPARATION (Replicating Stata transformations)
# =============================================================================

# Create region numeric code
df = df.sort_values(['region', 'year']).reset_index(drop=True)
df['dcode'] = df.groupby('region').ngroup() + 1

# Drop national aggregate (Italia)
df = df[df['region'] != 'Italia'].copy()

# Generate overcrowding variables
df['rovercrowding'] = df['jail'] / df['beds']
df['fovercrowding'] = (df['jail'] > df['beds']).astype(int)
df['fsingle'] = df['cells_1p'] / df['beds'] * 100
df['fdormitories'] = df['dormitories'] / df['beds'] * 100

# Generate South indicator
south_codes = [1, 2, 3, 4, 11, 13, 14, 15, 18]  # Based on do file
df['south'] = df['dcode'].isin(south_codes).astype(int)
df.loc[df['dcode'] == 7, 'south'] = np.nan  # Italia excluded

# Generate decade variable
df['decade'] = 60 + 10*(df['year'] >= 1970) + 10*(df['year'] >= 1980) + 10*(df['year'] >= 1990)

# Set up panel structure
df = df.sort_values(['dcode', 'year']).reset_index(drop=True)

# Generate pardon weights
pardon_years = {
    2006: 0.42, 1990: (0.03*(10/12) + 0.72*(2/12)), 1986: 0.04, 1981: 0.03,
    1978: 0.4, 1970: 0.61, 1968: 0.18, 1966: 0.58,
    1963: 0.94, 1959: 0.48, 1953: 0.03, 1949: 0.02, 1948: 0.92
}

df['pardonweight'] = df['year'].map(pardon_years).fillna(0)
df['pardonweight'] = df['pardonweight'].replace(0, 1)

df['pardon'] = df['year'].isin([1959, 1963, 1966, 1970, 1978, 1981, 1986, 1990]).astype(int)

# Create lagged variables
df['L_pardonweight'] = df.groupby('dcode')['pardonweight'].shift(1)
df['Lpardonweight'] = 1 - df['L_pardonweight']
df.loc[df['year'] == 1962, 'Lpardonweight'] = 0

df['L_jail'] = df.groupby('dcode')['jail'].shift(1)
df['L_exit_free_amnesty'] = df.groupby('dcode')['exit_free_amnesty'].shift(1)
df['L_all'] = df.groupby('dcode')['all'].shift(1)

# Per 100,000 residents transformation
pop_vars = ['jail', 'exit_free_amnesty', 'all', 'beds']
for var in pop_vars:
    if var in df.columns:
        df[var] = df[var] / df['population'] * 100

# Recompute lagged after transformation
df['L_jail'] = df.groupby('dcode')['jail'].shift(1)
df['L_exit_free_amnesty'] = df.groupby('dcode')['exit_free_amnesty'].shift(1)
df['L_all'] = df.groupby('dcode')['all'].shift(1)

# Generate weighted exit amnesty (adjustment for pardon timing)
df['wexit_free_amnesty'] = df['exit_free_amnesty'] * df['pardonweight'] + df['L_exit_free_amnesty'] * df['Lpardonweight']
df.loc[df['year'] == 1962, 'wexit_free_amnesty'] = df.loc[df['year'] == 1962, 'exit_free_amnesty'] * df.loc[df['year'] == 1962, 'pardonweight']

# Neighboring values - using simple approach
df['Njail'] = df.groupby('year')['jail'].transform('mean')
df['Nexit_free_amnesty'] = df.groupby('year')['exit_free_amnesty'].transform('mean')

df['L_Njail'] = df.groupby('dcode')['Njail'].shift(1)
df['L_Nexit_free_amnesty'] = df.groupby('dcode')['Nexit_free_amnesty'].shift(1)

df['Nwexit_free_amnesty'] = df['Nexit_free_amnesty'] * df['pardonweight'] + df['L_Nexit_free_amnesty'] * df['Lpardonweight']
df.loc[df['year'] == 1962, 'Nwexit_free_amnesty'] = df.loc[df['year'] == 1962, 'Nexit_free_amnesty'] * df.loc[df['year'] == 1962, 'pardonweight']

# Change in crime and jail
df['change_jail'] = df['jail'] - df['L_jail']
df['change_all'] = df['all'] - df['L_all']
df['lchange_jail'] = np.log(df['jail']) - np.log(df['L_jail'])
df['lchange_all'] = np.log(df['all']) - np.log(df['L_all'])

# Denominator for fraction calculations
df['denominator'] = df['L_jail'] * 0.5 + df['jail'] * 0.5

# Key instrument variable: fraction of pardoned inmates
df['lwexit_free_amnesty'] = df['wexit_free_amnesty'] / df['denominator']
df['lexit_free_amnesty'] = df['exit_free_amnesty'] / df['denominator']

# Weighted change in jail (adjusted)
df['L_wexit_free_amnesty'] = df.groupby('dcode')['wexit_free_amnesty'].shift(1)
df['wchange_jail'] = (df['jail'] - df['wexit_free_amnesty'] + df['exit_free_amnesty']) - \
                     (df['L_jail'] - df['L_wexit_free_amnesty'] + df['L_exit_free_amnesty'])
df.loc[df['year'] == 1962, 'wchange_jail'] = np.nan

# For log change, need to ensure positive values
jail_adj = df['jail'] - df['wexit_free_amnesty'] + df['exit_free_amnesty']
L_jail_adj = df['L_jail'] - df['L_wexit_free_amnesty'] + df['L_exit_free_amnesty']
df['lwchange_jail'] = np.log(jail_adj.clip(lower=0.001)) - np.log(L_jail_adj.clip(lower=0.001))

# Year dummies for pardon-specific periods
df['IS63year'] = df['year'] * ((df['year'] >= 1963) & (df['year'] < 1966)).astype(int)
df['IS66year'] = df['year'] * ((df['year'] >= 1966) & (df['year'] < 1968)).astype(int)
df['IS68year'] = df['year'] * ((df['year'] >= 1968) & (df['year'] < 1970)).astype(int)
df['IS70year'] = df['year'] * ((df['year'] >= 1970) & (df['year'] < 1978)).astype(int)
df['IS78year'] = df['year'] * ((df['year'] >= 1978) & (df['year'] < 1981)).astype(int)
df['IS81year'] = df['year'] * ((df['year'] >= 1981) & (df['year'] < 1986)).astype(int)
df['IS86year'] = df['year'] * ((df['year'] >= 1986) & (df['year'] < 1990)).astype(int)
df['IS90year'] = df['year'] * ((df['year'] >= 1990) & (df['year'] < 2006)).astype(int)

df['IC63year'] = ((df['year'] >= 1963) & (df['year'] < 1966)).astype(int)
df['IC66year'] = ((df['year'] >= 1966) & (df['year'] < 1968)).astype(int)
df['IC68year'] = ((df['year'] >= 1968) & (df['year'] < 1970)).astype(int)
df['IC70year'] = ((df['year'] >= 1970) & (df['year'] < 1978)).astype(int)
df['IC78year'] = ((df['year'] >= 1978) & (df['year'] < 1981)).astype(int)
df['IC81year'] = ((df['year'] >= 1981) & (df['year'] < 1986)).astype(int)
df['IC86year'] = ((df['year'] >= 1986) & (df['year'] < 1990)).astype(int)
df['IC90year'] = ((df['year'] >= 1990) & (df['year'] < 2006)).astype(int)

# Year 90/91 dummies for Umbria earthquake
df['year90'] = ((df['year'] == 1990) & df['dcode'].isin([13, 5, 14, 16, 8, 10, 9, 4, 15, 12, 6, 19])).astype(int)
df['year91_umbria'] = ((df['year'] == 1991) & (df['region'] == 'Umbria')).astype(int)

# Create year dummies for baseline
year_dummies = pd.get_dummies(df['year'], prefix='Dyear', drop_first=True)
df = pd.concat([df, year_dummies], axis=1)

# Additional control variables
control_vars_raw = ['pil', 'cfi', 'dis', 'pop1535', 'high', 'uni', 'police', 'fdormitories', 'rovercrowding']
for var in control_vars_raw:
    if var in df.columns:
        df[f'change_{var}'] = df[var] - df.groupby('dcode')[var].shift(1)
        df[f'lchange_{var}'] = np.log(df[var].clip(lower=0.001)) - np.log(df.groupby('dcode')[var].shift(1).clip(lower=0.001))

# Create region dummies
region_dummies = pd.get_dummies(df['region'], prefix='Dreg', drop_first=True)
df = pd.concat([df, region_dummies], axis=1)

# =============================================================================
# ANALYSIS FUNCTIONS - MANUAL 2SLS
# =============================================================================

import statsmodels.api as sm


def manual_2sls(df_sample, y_var, endog_var, instrument_var, exog_controls=None, cluster_var=None):
    """
    Manual 2SLS implementation for more robust handling
    """
    # Prepare data
    df_reg = df_sample.copy()
    all_vars = [y_var, endog_var, instrument_var]
    if exog_controls:
        all_vars = all_vars + exog_controls
    df_reg = df_reg.dropna(subset=all_vars)

    if len(df_reg) < 30:
        return {'success': False, 'error': 'Insufficient observations'}

    y = df_reg[y_var].values
    endog = df_reg[endog_var].values
    instrument = df_reg[instrument_var].values

    # Build exogenous matrix
    if exog_controls:
        X_exog = df_reg[exog_controls].values
        X_exog = sm.add_constant(X_exog)
    else:
        X_exog = np.ones((len(df_reg), 1))

    try:
        # First stage: regress endogenous on instrument + exog
        Z = np.column_stack([instrument, X_exog])
        fs_model = sm.OLS(endog, Z)
        if cluster_var and cluster_var in df_reg.columns:
            fs_result = fs_model.fit(cov_type='cluster', cov_kwds={'groups': df_reg[cluster_var].values})
        else:
            fs_result = fs_model.fit(cov_type='HC1')

        # Get first stage stats
        fs_coef = fs_result.params[0]  # coefficient on instrument
        fs_se = fs_result.bse[0]
        fs_tstat = fs_result.tvalues[0]
        fs_pval = fs_result.pvalues[0]
        fs_F = fs_tstat ** 2  # F-stat for single instrument

        # Predicted values of endogenous variable
        endog_hat = fs_result.fittedvalues

        # Second stage: regress y on fitted endogenous + exog
        X_2sls = np.column_stack([endog_hat, X_exog])
        ss_model = sm.OLS(y, X_2sls)
        if cluster_var and cluster_var in df_reg.columns:
            ss_result = ss_model.fit(cov_type='cluster', cov_kwds={'groups': df_reg[cluster_var].values})
        else:
            ss_result = ss_model.fit(cov_type='HC1')

        # Note: Standard errors need adjustment for 2SLS
        # For proper inference, we should use residuals from actual values
        residuals = y - (ss_result.params[0] * endog + X_exog @ ss_result.params[1:])

        # Compute corrected standard errors
        n = len(y)
        k = X_2sls.shape[1]

        # Get coefficient on endogenous variable
        coef = ss_result.params[0]

        # Compute proper 2SLS standard errors
        # Using the formula: Var(beta) = sigma^2 * (X'PzX)^(-1)
        # where Pz is the projection matrix onto instruments
        sigma2 = np.sum(residuals**2) / (n - k)

        # Compute X'PzX
        Pz = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
        X_full = np.column_stack([endog, X_exog])
        XPzX = X_full.T @ Pz @ X_full

        try:
            XPzX_inv = np.linalg.inv(XPzX)
            var_beta = sigma2 * XPzX_inv
            se = np.sqrt(var_beta[0, 0])
        except:
            se = ss_result.bse[0]  # fallback

        tstat = coef / se
        from scipy import stats
        pval = 2 * (1 - stats.t.cdf(abs(tstat), n - k))
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se

        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / ss_tot

        return {
            'coef': coef,
            'se': se,
            't_stat': tstat,
            'p_value': pval,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_obs': n,
            'r_squared': r2,
            'first_stage_F': fs_F,
            'first_stage_coef': fs_coef,
            'first_stage_se': fs_se,
            'first_stage_pval': fs_pval,
            'success': True
        }

    except Exception as e:
        return {
            'coef': np.nan,
            'se': np.nan,
            't_stat': np.nan,
            'p_value': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n_obs': np.nan,
            'r_squared': np.nan,
            'first_stage_F': np.nan,
            'first_stage_coef': np.nan,
            'first_stage_se': np.nan,
            'first_stage_pval': np.nan,
            'success': False,
            'error': str(e)
        }


def run_ols(df_sample, y_var, x_var, controls=None, cluster_var=None):
    """
    Run OLS regression
    """
    df_reg = df_sample.dropna(subset=[y_var, x_var])

    exog_vars = [x_var]
    if controls:
        df_reg = df_reg.dropna(subset=controls)
        exog_vars = exog_vars + controls

    X = sm.add_constant(df_reg[exog_vars])
    y = df_reg[y_var]

    if len(df_reg) < 30:
        return {'success': False, 'coef': np.nan, 'se': np.nan, 't_stat': np.nan,
                'p_value': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan,
                'n_obs': np.nan, 'r_squared': np.nan}

    try:
        if cluster_var and cluster_var in df_reg.columns:
            model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': df_reg[cluster_var]})
        else:
            model = sm.OLS(y, X).fit(cov_type='HC1')

        coef = model.params[x_var]
        se = model.bse[x_var]
        tstat = model.tvalues[x_var]
        pval = model.pvalues[x_var]
        ci_lower, ci_upper = model.conf_int().loc[x_var]

        return {
            'coef': coef,
            'se': se,
            't_stat': tstat,
            'p_value': pval,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_obs': int(model.nobs),
            'r_squared': model.rsquared,
            'success': True
        }
    except Exception as e:
        return {
            'coef': np.nan,
            'se': np.nan,
            't_stat': np.nan,
            'p_value': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n_obs': np.nan,
            'r_squared': np.nan,
            'success': False,
            'error': str(e)
        }


# =============================================================================
# SPECIFICATION SEARCH
# =============================================================================

results = []

# Get pardon-specific trend variables
trend_vars = ['IS63year', 'IS66year', 'IS68year', 'IS70year', 'IS78year', 'IS81year', 'IS86year', 'IS90year',
              'IC63year', 'IC66year', 'IC68year', 'IC70year', 'IC78year', 'IC81year', 'IC86year', 'IC90year']
trend_vars = [v for v in trend_vars if v in df.columns]

year_dummy_cols = [col for col in df.columns if col.startswith('Dyear')]
region_dummy_cols = [col for col in df.columns if col.startswith('Dreg')]

# Working sample (exclude missing)
df_sample = df.dropna(subset=['lchange_all', 'lwchange_jail', 'lwexit_free_amnesty']).copy()

print(f"Working sample size: {len(df_sample)}")

# =============================================================================
# 1. BASELINE SPECIFICATION
# =============================================================================

print("\n1. Running baseline specification...")

baseline_controls = trend_vars + ['year90', 'year91_umbria']
baseline_result = manual_2sls(
    df_sample,
    y_var='lchange_all',
    endog_var='lwchange_jail',
    instrument_var='lwexit_free_amnesty',
    exog_controls=baseline_controls,
    cluster_var='dcode'
)

results.append({
    'paper_id': PAPER_ID,
    'journal': JOURNAL,
    'paper_title': PAPER_TITLE,
    'spec_id': 'baseline',
    'spec_tree_path': 'methods/instrumental_variables.md#baseline',
    'outcome_var': 'lchange_all',
    'treatment_var': 'lwchange_jail',
    'coefficient': baseline_result['coef'],
    'std_error': baseline_result['se'],
    't_stat': baseline_result['t_stat'],
    'p_value': baseline_result['p_value'],
    'ci_lower': baseline_result['ci_lower'],
    'ci_upper': baseline_result['ci_upper'],
    'n_obs': baseline_result['n_obs'],
    'r_squared': baseline_result['r_squared'],
    'coefficient_vector_json': json.dumps({
        'treatment': {'var': 'lwchange_jail', 'coef': float(baseline_result['coef']) if pd.notna(baseline_result['coef']) else None,
                      'se': float(baseline_result['se']) if pd.notna(baseline_result['se']) else None,
                      'pval': float(baseline_result['p_value']) if pd.notna(baseline_result['p_value']) else None},
        'first_stage': {'instrument': 'lwexit_free_amnesty',
                       'coef': float(baseline_result['first_stage_coef']) if pd.notna(baseline_result.get('first_stage_coef')) else None,
                       'F_stat': float(baseline_result['first_stage_F']) if pd.notna(baseline_result.get('first_stage_F')) else None},
        'controls': baseline_controls,
        'diagnostics': {'first_stage_F': float(baseline_result['first_stage_F']) if pd.notna(baseline_result.get('first_stage_F')) else None}
    }),
    'sample_desc': 'Full panel 1963-1995, 18 regions (excluding Italia)',
    'fixed_effects': 'None (pardon-specific trends instead)',
    'controls_desc': 'Pardon-specific linear trends, year90, year91_umbria dummies',
    'cluster_var': 'dcode (region)',
    'model_type': 'IV-2SLS',
    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
})

print(f"  Baseline: coef={baseline_result['coef']:.4f}, se={baseline_result['se']:.4f}, p={baseline_result['p_value']:.4f}, F={baseline_result.get('first_stage_F', np.nan):.2f}")

# =============================================================================
# 2. IV METHOD VARIATIONS
# =============================================================================

print("\n2. Running IV method variations...")

# 2a. OLS (ignoring endogeneity)
ols_result = run_ols(df_sample, 'lchange_all', 'lwchange_jail', baseline_controls, 'dcode')
results.append({
    'paper_id': PAPER_ID,
    'journal': JOURNAL,
    'paper_title': PAPER_TITLE,
    'spec_id': 'iv/method/ols',
    'spec_tree_path': 'methods/instrumental_variables.md#estimation-method',
    'outcome_var': 'lchange_all',
    'treatment_var': 'lwchange_jail',
    'coefficient': ols_result['coef'],
    'std_error': ols_result['se'],
    't_stat': ols_result['t_stat'],
    'p_value': ols_result['p_value'],
    'ci_lower': ols_result['ci_lower'],
    'ci_upper': ols_result['ci_upper'],
    'n_obs': ols_result['n_obs'],
    'r_squared': ols_result['r_squared'],
    'coefficient_vector_json': json.dumps({'treatment': {'var': 'lwchange_jail',
                                                          'coef': float(ols_result['coef']) if pd.notna(ols_result['coef']) else None,
                                                          'se': float(ols_result['se']) if pd.notna(ols_result['se']) else None,
                                                          'pval': float(ols_result['p_value']) if pd.notna(ols_result['p_value']) else None}}),
    'sample_desc': 'Full panel',
    'fixed_effects': 'None',
    'controls_desc': 'Pardon-specific trends',
    'cluster_var': 'dcode',
    'model_type': 'OLS',
    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
})
print(f"  OLS: coef={ols_result['coef']:.4f}, p={ols_result['p_value']:.4f}")

# =============================================================================
# 3. FIRST STAGE AND REDUCED FORM
# =============================================================================

print("\n3. Running first stage and reduced form...")

# First stage
fs_result = run_ols(df_sample, 'lwchange_jail', 'lwexit_free_amnesty', baseline_controls, 'dcode')
results.append({
    'paper_id': PAPER_ID,
    'journal': JOURNAL,
    'paper_title': PAPER_TITLE,
    'spec_id': 'iv/first_stage/baseline',
    'spec_tree_path': 'methods/instrumental_variables.md#first-stage',
    'outcome_var': 'lwchange_jail',
    'treatment_var': 'lwexit_free_amnesty',
    'coefficient': fs_result['coef'],
    'std_error': fs_result['se'],
    't_stat': fs_result['t_stat'],
    'p_value': fs_result['p_value'],
    'ci_lower': fs_result['ci_lower'],
    'ci_upper': fs_result['ci_upper'],
    'n_obs': fs_result['n_obs'],
    'r_squared': fs_result['r_squared'],
    'coefficient_vector_json': json.dumps({'instrument': {'var': 'lwexit_free_amnesty',
                                                           'coef': float(fs_result['coef']) if pd.notna(fs_result['coef']) else None,
                                                           'se': float(fs_result['se']) if pd.notna(fs_result['se']) else None,
                                                           'pval': float(fs_result['p_value']) if pd.notna(fs_result['p_value']) else None}}),
    'sample_desc': 'Full panel',
    'fixed_effects': 'None',
    'controls_desc': 'Pardon-specific trends',
    'cluster_var': 'dcode',
    'model_type': 'OLS (First Stage)',
    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
})
print(f"  First Stage: coef={fs_result['coef']:.4f}, p={fs_result['p_value']:.4f}")

# Reduced form
rf_result = run_ols(df_sample, 'lchange_all', 'lwexit_free_amnesty', baseline_controls, 'dcode')
results.append({
    'paper_id': PAPER_ID,
    'journal': JOURNAL,
    'paper_title': PAPER_TITLE,
    'spec_id': 'iv/first_stage/reduced_form',
    'spec_tree_path': 'methods/instrumental_variables.md#first-stage',
    'outcome_var': 'lchange_all',
    'treatment_var': 'lwexit_free_amnesty',
    'coefficient': rf_result['coef'],
    'std_error': rf_result['se'],
    't_stat': rf_result['t_stat'],
    'p_value': rf_result['p_value'],
    'ci_lower': rf_result['ci_lower'],
    'ci_upper': rf_result['ci_upper'],
    'n_obs': rf_result['n_obs'],
    'r_squared': rf_result['r_squared'],
    'coefficient_vector_json': json.dumps({'instrument': {'var': 'lwexit_free_amnesty',
                                                           'coef': float(rf_result['coef']) if pd.notna(rf_result['coef']) else None,
                                                           'se': float(rf_result['se']) if pd.notna(rf_result['se']) else None,
                                                           'pval': float(rf_result['p_value']) if pd.notna(rf_result['p_value']) else None}}),
    'sample_desc': 'Full panel',
    'fixed_effects': 'None',
    'controls_desc': 'Pardon-specific trends',
    'cluster_var': 'dcode',
    'model_type': 'OLS (Reduced Form)',
    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
})
print(f"  Reduced Form: coef={rf_result['coef']:.4f}, p={rf_result['p_value']:.4f}")

# =============================================================================
# 4. FIXED EFFECTS VARIATIONS
# =============================================================================

print("\n4. Running fixed effects variations...")

# 4a. With year dummies instead of pardon-specific trends
year_controls = [col for col in year_dummy_cols if col in df_sample.columns][:25]
iv_year_fe = manual_2sls(df_sample, 'lchange_all', 'lwchange_jail', 'lwexit_free_amnesty',
                          year_controls + ['year90', 'year91_umbria'], 'dcode')
results.append({
    'paper_id': PAPER_ID,
    'journal': JOURNAL,
    'paper_title': PAPER_TITLE,
    'spec_id': 'iv/fe/time',
    'spec_tree_path': 'methods/instrumental_variables.md#fixed-effects',
    'outcome_var': 'lchange_all',
    'treatment_var': 'lwchange_jail',
    'coefficient': iv_year_fe['coef'],
    'std_error': iv_year_fe['se'],
    't_stat': iv_year_fe['t_stat'],
    'p_value': iv_year_fe['p_value'],
    'ci_lower': iv_year_fe['ci_lower'],
    'ci_upper': iv_year_fe['ci_upper'],
    'n_obs': iv_year_fe['n_obs'],
    'r_squared': iv_year_fe['r_squared'],
    'coefficient_vector_json': json.dumps({'treatment': {'var': 'lwchange_jail',
                                                          'coef': float(iv_year_fe['coef']) if pd.notna(iv_year_fe['coef']) else None,
                                                          'se': float(iv_year_fe['se']) if pd.notna(iv_year_fe['se']) else None,
                                                          'pval': float(iv_year_fe['p_value']) if pd.notna(iv_year_fe['p_value']) else None},
                                            'first_stage_F': float(iv_year_fe.get('first_stage_F')) if pd.notna(iv_year_fe.get('first_stage_F')) else None}),
    'sample_desc': 'Full panel',
    'fixed_effects': 'Year dummies',
    'controls_desc': 'Year dummies',
    'cluster_var': 'dcode',
    'model_type': 'IV-2SLS',
    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
})
print(f"  Year FE: coef={iv_year_fe['coef']:.4f}, p={iv_year_fe['p_value']:.4f}")

# 4b. With region dummies
region_controls = [col for col in region_dummy_cols if col in df_sample.columns]
iv_region_fe = manual_2sls(df_sample, 'lchange_all', 'lwchange_jail', 'lwexit_free_amnesty',
                           trend_vars + region_controls + ['year90', 'year91_umbria'], 'dcode')
results.append({
    'paper_id': PAPER_ID,
    'journal': JOURNAL,
    'paper_title': PAPER_TITLE,
    'spec_id': 'iv/fe/unit',
    'spec_tree_path': 'methods/instrumental_variables.md#fixed-effects',
    'outcome_var': 'lchange_all',
    'treatment_var': 'lwchange_jail',
    'coefficient': iv_region_fe['coef'],
    'std_error': iv_region_fe['se'],
    't_stat': iv_region_fe['t_stat'],
    'p_value': iv_region_fe['p_value'],
    'ci_lower': iv_region_fe['ci_lower'],
    'ci_upper': iv_region_fe['ci_upper'],
    'n_obs': iv_region_fe['n_obs'],
    'r_squared': iv_region_fe['r_squared'],
    'coefficient_vector_json': json.dumps({'treatment': {'var': 'lwchange_jail',
                                                          'coef': float(iv_region_fe['coef']) if pd.notna(iv_region_fe['coef']) else None,
                                                          'se': float(iv_region_fe['se']) if pd.notna(iv_region_fe['se']) else None,
                                                          'pval': float(iv_region_fe['p_value']) if pd.notna(iv_region_fe['p_value']) else None}}),
    'sample_desc': 'Full panel',
    'fixed_effects': 'Region dummies',
    'controls_desc': 'Pardon-specific trends + region dummies',
    'cluster_var': 'dcode',
    'model_type': 'IV-2SLS',
    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
})
print(f"  Region FE: coef={iv_region_fe['coef']:.4f}, p={iv_region_fe['p_value']:.4f}")

# 4c. No controls (simple IV)
iv_no_controls = manual_2sls(df_sample, 'lchange_all', 'lwchange_jail', 'lwexit_free_amnesty',
                              None, 'dcode')
results.append({
    'paper_id': PAPER_ID,
    'journal': JOURNAL,
    'paper_title': PAPER_TITLE,
    'spec_id': 'iv/fe/none',
    'spec_tree_path': 'methods/instrumental_variables.md#fixed-effects',
    'outcome_var': 'lchange_all',
    'treatment_var': 'lwchange_jail',
    'coefficient': iv_no_controls['coef'],
    'std_error': iv_no_controls['se'],
    't_stat': iv_no_controls['t_stat'],
    'p_value': iv_no_controls['p_value'],
    'ci_lower': iv_no_controls['ci_lower'],
    'ci_upper': iv_no_controls['ci_upper'],
    'n_obs': iv_no_controls['n_obs'],
    'r_squared': iv_no_controls['r_squared'],
    'coefficient_vector_json': json.dumps({'treatment': {'var': 'lwchange_jail',
                                                          'coef': float(iv_no_controls['coef']) if pd.notna(iv_no_controls['coef']) else None}}),
    'sample_desc': 'Full panel',
    'fixed_effects': 'None',
    'controls_desc': 'None',
    'cluster_var': 'dcode',
    'model_type': 'IV-2SLS',
    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
})
print(f"  No FE: coef={iv_no_controls['coef']:.4f}, p={iv_no_controls['p_value']:.4f}")

# =============================================================================
# 5. FUNCTIONAL FORM VARIATIONS (LEVELS VS LOGS)
# =============================================================================

print("\n5. Running functional form variations...")

# 5a. In levels (not logs)
df_levels = df.dropna(subset=['change_all', 'wchange_jail', 'wexit_free_amnesty']).copy()
iv_levels = manual_2sls(df_levels, 'change_all', 'wchange_jail', 'wexit_free_amnesty',
                         trend_vars + ['year90', 'year91_umbria'], 'dcode')
results.append({
    'paper_id': PAPER_ID,
    'journal': JOURNAL,
    'paper_title': PAPER_TITLE,
    'spec_id': 'robust/form/y_level',
    'spec_tree_path': 'robustness/functional_form.md#outcome-variable-transformations',
    'outcome_var': 'change_all',
    'treatment_var': 'wchange_jail',
    'coefficient': iv_levels['coef'],
    'std_error': iv_levels['se'],
    't_stat': iv_levels['t_stat'],
    'p_value': iv_levels['p_value'],
    'ci_lower': iv_levels['ci_lower'],
    'ci_upper': iv_levels['ci_upper'],
    'n_obs': iv_levels['n_obs'],
    'r_squared': iv_levels['r_squared'],
    'coefficient_vector_json': json.dumps({'treatment': {'var': 'wchange_jail',
                                                          'coef': float(iv_levels['coef']) if pd.notna(iv_levels['coef']) else None}}),
    'sample_desc': 'Full panel',
    'fixed_effects': 'None',
    'controls_desc': 'Pardon-specific trends',
    'cluster_var': 'dcode',
    'model_type': 'IV-2SLS',
    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
})
print(f"  Levels: coef={iv_levels['coef']:.4f}, p={iv_levels['p_value']:.4f}")

# =============================================================================
# 6. CLUSTERING VARIATIONS
# =============================================================================

print("\n6. Running clustering variations...")

# 6a. Cluster by year
iv_cluster_year = manual_2sls(df_sample, 'lchange_all', 'lwchange_jail', 'lwexit_free_amnesty',
                               baseline_controls, 'year')
results.append({
    'paper_id': PAPER_ID,
    'journal': JOURNAL,
    'paper_title': PAPER_TITLE,
    'spec_id': 'robust/cluster/time',
    'spec_tree_path': 'robustness/clustering_variations.md#single-level-clustering',
    'outcome_var': 'lchange_all',
    'treatment_var': 'lwchange_jail',
    'coefficient': iv_cluster_year['coef'],
    'std_error': iv_cluster_year['se'],
    't_stat': iv_cluster_year['t_stat'],
    'p_value': iv_cluster_year['p_value'],
    'ci_lower': iv_cluster_year['ci_lower'],
    'ci_upper': iv_cluster_year['ci_upper'],
    'n_obs': iv_cluster_year['n_obs'],
    'r_squared': iv_cluster_year['r_squared'],
    'coefficient_vector_json': json.dumps({'treatment': {'var': 'lwchange_jail',
                                                          'coef': float(iv_cluster_year['coef']) if pd.notna(iv_cluster_year['coef']) else None,
                                                          'se': float(iv_cluster_year['se']) if pd.notna(iv_cluster_year['se']) else None}}),
    'sample_desc': 'Full panel',
    'fixed_effects': 'None',
    'controls_desc': 'Pardon-specific trends',
    'cluster_var': 'year',
    'model_type': 'IV-2SLS',
    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
})
print(f"  Cluster by year: coef={iv_cluster_year['coef']:.4f}, se={iv_cluster_year['se']:.4f}")

# 6b. Robust SE (no clustering)
iv_robust = manual_2sls(df_sample, 'lchange_all', 'lwchange_jail', 'lwexit_free_amnesty',
                         baseline_controls, None)
results.append({
    'paper_id': PAPER_ID,
    'journal': JOURNAL,
    'paper_title': PAPER_TITLE,
    'spec_id': 'robust/cluster/none',
    'spec_tree_path': 'robustness/clustering_variations.md#single-level-clustering',
    'outcome_var': 'lchange_all',
    'treatment_var': 'lwchange_jail',
    'coefficient': iv_robust['coef'],
    'std_error': iv_robust['se'],
    't_stat': iv_robust['t_stat'],
    'p_value': iv_robust['p_value'],
    'ci_lower': iv_robust['ci_lower'],
    'ci_upper': iv_robust['ci_upper'],
    'n_obs': iv_robust['n_obs'],
    'r_squared': iv_robust['r_squared'],
    'coefficient_vector_json': json.dumps({'treatment': {'var': 'lwchange_jail',
                                                          'coef': float(iv_robust['coef']) if pd.notna(iv_robust['coef']) else None}}),
    'sample_desc': 'Full panel',
    'fixed_effects': 'None',
    'controls_desc': 'Pardon-specific trends',
    'cluster_var': 'None (robust SE)',
    'model_type': 'IV-2SLS',
    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
})
print(f"  Robust SE: coef={iv_robust['coef']:.4f}, se={iv_robust['se']:.4f}")

# =============================================================================
# 7. SAMPLE RESTRICTIONS
# =============================================================================

print("\n7. Running sample restrictions...")

# 7a. Exclude each region one at a time
regions = df_sample['region'].unique()
for reg in regions[:8]:
    df_excl = df_sample[df_sample['region'] != reg].copy()
    result = manual_2sls(df_excl, 'lchange_all', 'lwchange_jail', 'lwexit_free_amnesty',
                          baseline_controls, 'dcode')
    reg_clean = reg.replace(" ", "_").replace("&", "and").replace("'", "")
    results.append({
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': f'robust/sample/exclude_{reg_clean}',
        'spec_tree_path': 'robustness/sample_restrictions.md#geographic-unit-restrictions',
        'outcome_var': 'lchange_all',
        'treatment_var': 'lwchange_jail',
        'coefficient': result['coef'],
        'std_error': result['se'],
        't_stat': result['t_stat'],
        'p_value': result['p_value'],
        'ci_lower': result['ci_lower'],
        'ci_upper': result['ci_upper'],
        'n_obs': result['n_obs'],
        'r_squared': result['r_squared'],
        'coefficient_vector_json': json.dumps({'treatment': {'var': 'lwchange_jail',
                                                              'coef': float(result['coef']) if pd.notna(result['coef']) else None}}),
        'sample_desc': f'Excluding {reg}',
        'fixed_effects': 'None',
        'controls_desc': 'Pardon-specific trends',
        'cluster_var': 'dcode',
        'model_type': 'IV-2SLS',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })
    print(f"  Exclude {reg}: coef={result['coef']:.4f}, p={result['p_value']:.4f}")

# 7b. Exclude each decade
for dec in [60, 70, 80, 90]:
    df_excl = df_sample[df_sample['decade'] != dec].copy()
    if len(df_excl) > 50:
        result = manual_2sls(df_excl, 'lchange_all', 'lwchange_jail', 'lwexit_free_amnesty',
                              baseline_controls, 'dcode')
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': f'robust/sample/exclude_decade_{dec}s',
            'spec_tree_path': 'robustness/sample_restrictions.md#time-based-restrictions',
            'outcome_var': 'lchange_all',
            'treatment_var': 'lwchange_jail',
            'coefficient': result['coef'],
            'std_error': result['se'],
            't_stat': result['t_stat'],
            'p_value': result['p_value'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'n_obs': result['n_obs'],
            'r_squared': result['r_squared'],
            'coefficient_vector_json': json.dumps({'treatment': {'var': 'lwchange_jail',
                                                                  'coef': float(result['coef']) if pd.notna(result['coef']) else None}}),
            'sample_desc': f'Excluding {dec}s decade',
            'fixed_effects': 'None',
            'controls_desc': 'Pardon-specific trends',
            'cluster_var': 'dcode',
            'model_type': 'IV-2SLS',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })
        print(f"  Exclude {dec}s: coef={result['coef']:.4f}, p={result['p_value']:.4f}")

# 7c. Early vs late period
mid_year = df_sample['year'].median()
df_early = df_sample[df_sample['year'] <= mid_year].copy()
df_late = df_sample[df_sample['year'] > mid_year].copy()

for period, df_period in [('early_period', df_early), ('late_period', df_late)]:
    result = manual_2sls(df_period, 'lchange_all', 'lwchange_jail', 'lwexit_free_amnesty',
                          baseline_controls, 'dcode')
    results.append({
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': f'robust/sample/{period}',
        'spec_tree_path': 'robustness/sample_restrictions.md#time-based-restrictions',
        'outcome_var': 'lchange_all',
        'treatment_var': 'lwchange_jail',
        'coefficient': result['coef'],
        'std_error': result['se'],
        't_stat': result['t_stat'],
        'p_value': result['p_value'],
        'ci_lower': result['ci_lower'],
        'ci_upper': result['ci_upper'],
        'n_obs': result['n_obs'],
        'r_squared': result['r_squared'],
        'coefficient_vector_json': json.dumps({'treatment': {'var': 'lwchange_jail',
                                                              'coef': float(result['coef']) if pd.notna(result['coef']) else None}}),
        'sample_desc': period.replace('_', ' ').title(),
        'fixed_effects': 'None',
        'controls_desc': 'Pardon-specific trends',
        'cluster_var': 'dcode',
        'model_type': 'IV-2SLS',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })
    print(f"  {period}: coef={result['coef']:.4f}, p={result['p_value']:.4f}")

# 7d. Trim outliers (1%)
df_trim = df_sample.copy()
for var in ['lchange_all', 'lwchange_jail', 'lwexit_free_amnesty']:
    q01 = df_trim[var].quantile(0.01)
    q99 = df_trim[var].quantile(0.99)
    df_trim = df_trim[(df_trim[var] >= q01) & (df_trim[var] <= q99)]

result = manual_2sls(df_trim, 'lchange_all', 'lwchange_jail', 'lwexit_free_amnesty',
                      baseline_controls, 'dcode')
results.append({
    'paper_id': PAPER_ID,
    'journal': JOURNAL,
    'paper_title': PAPER_TITLE,
    'spec_id': 'robust/sample/trim_1pct',
    'spec_tree_path': 'robustness/sample_restrictions.md#outlier-handling',
    'outcome_var': 'lchange_all',
    'treatment_var': 'lwchange_jail',
    'coefficient': result['coef'],
    'std_error': result['se'],
    't_stat': result['t_stat'],
    'p_value': result['p_value'],
    'ci_lower': result['ci_lower'],
    'ci_upper': result['ci_upper'],
    'n_obs': result['n_obs'],
    'r_squared': result['r_squared'],
    'coefficient_vector_json': json.dumps({'treatment': {'var': 'lwchange_jail',
                                                          'coef': float(result['coef']) if pd.notna(result['coef']) else None}}),
    'sample_desc': 'Trimmed 1% tails',
    'fixed_effects': 'None',
    'controls_desc': 'Pardon-specific trends',
    'cluster_var': 'dcode',
    'model_type': 'IV-2SLS',
    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
})
print(f"  Trim 1%: coef={result['coef']:.4f}, p={result['p_value']:.4f}")

# =============================================================================
# 8. ADDITIONAL CONTROLS (TIME-VARYING)
# =============================================================================

print("\n8. Running specifications with additional controls...")

# Minimal controls
result = manual_2sls(df_sample, 'lchange_all', 'lwchange_jail', 'lwexit_free_amnesty',
                      ['year90', 'year91_umbria'], 'dcode')
results.append({
    'paper_id': PAPER_ID,
    'journal': JOURNAL,
    'paper_title': PAPER_TITLE,
    'spec_id': 'iv/controls/minimal',
    'spec_tree_path': 'methods/instrumental_variables.md#control-sets',
    'outcome_var': 'lchange_all',
    'treatment_var': 'lwchange_jail',
    'coefficient': result['coef'],
    'std_error': result['se'],
    't_stat': result['t_stat'],
    'p_value': result['p_value'],
    'ci_lower': result['ci_lower'],
    'ci_upper': result['ci_upper'],
    'n_obs': result['n_obs'],
    'r_squared': result['r_squared'],
    'coefficient_vector_json': json.dumps({'treatment': {'var': 'lwchange_jail',
                                                          'coef': float(result['coef']) if pd.notna(result['coef']) else None}}),
    'sample_desc': 'Full panel',
    'fixed_effects': 'None',
    'controls_desc': 'Minimal (year90, year91_umbria)',
    'cluster_var': 'dcode',
    'model_type': 'IV-2SLS',
    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
})
print(f"  Minimal controls: coef={result['coef']:.4f}, p={result['p_value']:.4f}")

# =============================================================================
# 9. DIFFERENT OUTCOME VARIABLES (CRIME TYPES)
# =============================================================================

print("\n9. Running specifications with different crime types...")

crime_types = {
    'totpen': 'Total reported crimes',
    'furpen': 'Theft',
    'omipen': 'Homicide',
    'rappen': 'Robbery'
}

for crime_var, crime_desc in crime_types.items():
    if crime_var in df.columns:
        df[f'lchange_{crime_var}'] = np.log(df[crime_var].clip(lower=0.001)) - np.log(df.groupby('dcode')[crime_var].shift(1).clip(lower=0.001))
        df_crime = df.dropna(subset=[f'lchange_{crime_var}', 'lwchange_jail', 'lwexit_free_amnesty']).copy()

        if len(df_crime) > 50:
            result = manual_2sls(df_crime, f'lchange_{crime_var}', 'lwchange_jail', 'lwexit_free_amnesty',
                                  baseline_controls, 'dcode')
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': f'custom/outcome/{crime_var}',
                'spec_tree_path': 'custom',
                'outcome_var': f'lchange_{crime_var}',
                'treatment_var': 'lwchange_jail',
                'coefficient': result['coef'],
                'std_error': result['se'],
                't_stat': result['t_stat'],
                'p_value': result['p_value'],
                'ci_lower': result['ci_lower'],
                'ci_upper': result['ci_upper'],
                'n_obs': result['n_obs'],
                'r_squared': result['r_squared'],
                'coefficient_vector_json': json.dumps({'treatment': {'var': 'lwchange_jail',
                                                                      'coef': float(result['coef']) if pd.notna(result['coef']) else None},
                                                        'outcome': crime_desc}),
                'sample_desc': f'Full panel, outcome = {crime_desc}',
                'fixed_effects': 'None',
                'controls_desc': 'Pardon-specific trends',
                'cluster_var': 'dcode',
                'model_type': 'IV-2SLS',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })
            print(f"  {crime_desc}: coef={result['coef']:.4f}, p={result['p_value']:.4f}")

# =============================================================================
# 10. ALTERNATIVE INSTRUMENT ADJUSTMENTS
# =============================================================================

print("\n10. Running with alternative instrument adjustments...")

df_unadj = df.dropna(subset=['lchange_all', 'lchange_jail', 'lexit_free_amnesty']).copy()
result = manual_2sls(df_unadj, 'lchange_all', 'lchange_jail', 'lexit_free_amnesty',
                      baseline_controls, 'dcode')
results.append({
    'paper_id': PAPER_ID,
    'journal': JOURNAL,
    'paper_title': PAPER_TITLE,
    'spec_id': 'iv/instruments/alternative',
    'spec_tree_path': 'methods/instrumental_variables.md#instrument-sets',
    'outcome_var': 'lchange_all',
    'treatment_var': 'lchange_jail',
    'coefficient': result['coef'],
    'std_error': result['se'],
    't_stat': result['t_stat'],
    'p_value': result['p_value'],
    'ci_lower': result['ci_lower'],
    'ci_upper': result['ci_upper'],
    'n_obs': result['n_obs'],
    'r_squared': result['r_squared'],
    'coefficient_vector_json': json.dumps({'treatment': {'var': 'lchange_jail',
                                                          'coef': float(result['coef']) if pd.notna(result['coef']) else None},
                                            'note': 'Unadjusted for pardon timing'}),
    'sample_desc': 'Full panel',
    'fixed_effects': 'None',
    'controls_desc': 'Pardon-specific trends',
    'cluster_var': 'dcode',
    'model_type': 'IV-2SLS',
    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
})
print(f"  Unadjusted instrument: coef={result['coef']:.4f}, p={result['p_value']:.4f}")

# =============================================================================
# SAVE RESULTS
# =============================================================================

print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

results_df = pd.DataFrame(results)

output_file = OUTPUT_DIR / 'specification_results.csv'
results_df.to_csv(output_file, index=False)
print(f"\nSaved {len(results_df)} specifications to {output_file}")

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

valid_results = results_df[results_df['coefficient'].notna()]
print(f"\nTotal specifications: {len(results_df)}")
print(f"Valid results: {len(valid_results)}")

if len(valid_results) > 0:
    print(f"\nCoefficient statistics:")
    print(f"  Mean: {valid_results['coefficient'].mean():.4f}")
    print(f"  Median: {valid_results['coefficient'].median():.4f}")
    print(f"  Std: {valid_results['coefficient'].std():.4f}")
    print(f"  Min: {valid_results['coefficient'].min():.4f}")
    print(f"  Max: {valid_results['coefficient'].max():.4f}")

    sig_05 = (valid_results['p_value'] < 0.05).sum()
    sig_01 = (valid_results['p_value'] < 0.01).sum()
    positive = (valid_results['coefficient'] > 0).sum()

    print(f"\nSignificance:")
    print(f"  Significant at 5%: {sig_05} ({100*sig_05/len(valid_results):.1f}%)")
    print(f"  Significant at 1%: {sig_01} ({100*sig_01/len(valid_results):.1f}%)")
    print(f"  Positive coefficients: {positive} ({100*positive/len(valid_results):.1f}%)")

print("\nDone!")
