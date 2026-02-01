"""
Specification Search: Paper 198865-V1
"Estimating Models of Supply and Demand: Instruments and Covariance Restrictions"
Alexander MacKay and Nathan H. Miller
American Economic Journal: Microeconomics

This script replicates and extends the Airlines empirical application from Section 5.3.
The paper demonstrates identification using covariance restrictions between demand and cost shocks.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize_scalar
import json
import warnings
warnings.filterwarnings('ignore')

# Define paths
BASE_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search"
DATA_PATH = f"{BASE_PATH}/data/downloads/extracted/198865-V1/data/airlines/eco2901_problemset_01_2012_airlines_data.dta"

# Paper metadata
PAPER_ID = "198865-V1"
JOURNAL = "AEJ: Micro"
PAPER_TITLE = "Estimating Models of Supply and Demand: Instruments and Covariance Restrictions"

##############################################################################
# STEP 1: Process Airlines Data (replicating Stata preprocessing)
##############################################################################

def process_airlines_data():
    """
    Replicate the data processing from a07_process_airlines_data.do
    """
    df = pd.read_stata(DATA_PATH)

    # Rename for consistency
    df = df.rename(columns={'HUB_origin': 'HUB_orig'})

    # Market size and shares
    df['msize'] = 1000 * (df['pop04_origin'] + df['pop04_dest']) / 2
    df['share'] = df['passengers'] / df['msize']

    # Outside good share
    df['share0'] = df.groupby(['route_id', 'quarter'])['share'].transform('sum')
    df['share0'] = 1 - df['share0']

    # Group shares (direct vs connecting)
    df['share_stop'] = df.groupby(['route_id', 'quarter']).apply(
        lambda x: (x['share'] * (1 - x['direct'])).sum()
    ).reindex(df.set_index(['route_id', 'quarter']).index).values

    df['share_nonstop'] = df.groupby(['route_id', 'quarter']).apply(
        lambda x: (x['share'] * x['direct']).sum()
    ).reindex(df.set_index(['route_id', 'quarter']).index).values

    df['share_group'] = np.where(df['direct'] == 1, df['share_nonstop'], df['share_stop'])
    df['share_within'] = df['share'] / df['share_group']

    # Log transformations
    df['log_s_s0'] = np.log(df['share'] / df['share0'])
    df['log_s_within'] = np.log(df['share_within'])

    # Normalize price and distance
    df['price100'] = df['price'] / 100
    df['miles1000'] = df['avg_miles'] / 1000

    # Route-quarter identifier
    df['route_quarter'] = 10 * df['route_id'] + df['quarter']

    # Quarter and airline dummies
    quarter_dummies = pd.get_dummies(df['quarter'], prefix='tdum', drop_first=False)
    airline_dummies = pd.get_dummies(df['airline'], prefix='airdum', drop_first=False)
    df = pd.concat([df, quarter_dummies, airline_dummies], axis=1)

    # Legacy carrier indicator for instruments
    legacy_carriers = ['American (AA)', 'Continental (CO)', 'Delta (DL)',
                       'Northwest (NW)', 'US Airways (US)', 'United (UA)']
    df['leg'] = df['airline'].isin(legacy_carriers).astype(int)

    # Instruments: Legacy hub sizes
    df['temp1a'] = df['leg'] * 0.5 * (df['HUB_orig'] + df['HUB_dest'])
    temp2a = df.groupby(['route_id', 'quarter', 'leg'])['temp1a'].transform('mean')
    df['z_leghub_a'] = df.groupby(['route_id', 'quarter'])['temp1a'].transform(
        lambda x: x[df.loc[x.index, 'leg'] == 1].mean() if any(df.loc[x.index, 'leg'] == 1) else 0
    )

    # Instrument: Southwest presence
    df['sw'] = (df['airline'] == 'Southwest (WN)').astype(int)
    df['z_sw'] = df.groupby(['route_id', 'quarter'])['sw'].transform('max')

    # Instrument: Average non-stop for other carriers
    temp1 = df.groupby(['route_id', 'quarter'])['direct'].transform('sum')
    temp2 = df.groupby(['route_id', 'quarter'])['direct'].transform('count')
    temp3 = df.groupby(['route_id', 'quarter', 'airline'])['direct'].transform('sum')
    temp4 = df.groupby(['route_id', 'quarter', 'airline'])['direct'].transform('count')
    df['z_direct'] = (temp1 - temp3) / (temp2 - temp4)

    # Instrument: Competitor hub sizes
    temp0a = df['leg'] * 0.5 * (df['HUB_orig'] + df['HUB_dest'])
    temp1a = df.groupby(['route_id', 'quarter'])['temp1a'].transform('sum')
    temp2 = df.groupby(['route_id', 'quarter'])['direct'].transform('count')
    temp3a = df.groupby(['route_id', 'quarter', 'airline'])['temp1a'].transform('sum')
    temp4 = df.groupby(['route_id', 'quarter', 'airline'])['direct'].transform('count')
    df['z_comphub_a'] = (temp1a - temp3a) / (temp2 - temp4)

    # Hub sizes based on connections (simplified version)
    df['HUB2_orig'] = df.groupby(['airline', 'quarter', 'route_city'])['route_id'].transform('nunique')
    df['HUB2_dest'] = df['HUB2_orig']  # Simplified

    # Create firm and market IDs
    df['firmid'] = pd.factorize(df['airline'])[0] + 1
    df['mktid'] = pd.factorize(df['route_quarter'])[0] + 1

    # Drop missing values
    df = df.dropna(subset=['z_direct', 'log_s_s0'])

    return df


def demean_by_market(df, variables, market_var='mktid'):
    """
    Demean variables by market (route-quarter) for within-market estimation.
    """
    df_dm = df.copy()
    for var in variables:
        if var in df.columns:
            df_dm[var] = df_dm.groupby(market_var)[var].transform(lambda x: x - x.mean())
    return df_dm


##############################################################################
# STEP 2: Estimation Functions
##############################################################################

def run_ols(df, y_var, x_vars, cluster_var=None):
    """Run OLS regression with optional clustering."""
    df_clean = df[[y_var] + x_vars].dropna()
    y = df_clean[y_var]
    X = sm.add_constant(df_clean[x_vars])

    model = sm.OLS(y, X).fit()

    if cluster_var is not None and cluster_var in df.columns:
        # Cluster-robust standard errors
        model = sm.OLS(y, X).fit(cov_type='cluster',
                                  cov_kwds={'groups': df.loc[df_clean.index, cluster_var]})

    return model


def run_2sls(df, y_var, endog_vars, exog_vars, instruments):
    """
    Run 2SLS regression.

    Parameters:
    -----------
    df : DataFrame
    y_var : str - dependent variable
    endog_vars : list - endogenous regressors
    exog_vars : list - exogenous controls
    instruments : list - instrumental variables
    """
    df_clean = df[[y_var] + endog_vars + exog_vars + instruments].dropna()
    y = df_clean[y_var]

    # First stage
    first_stages = {}
    fitted_endogs = pd.DataFrame(index=df_clean.index)

    for endog in endog_vars:
        X_first = sm.add_constant(df_clean[exog_vars + instruments])
        model_first = sm.OLS(df_clean[endog], X_first).fit()
        first_stages[endog] = model_first
        fitted_endogs[f'{endog}_hat'] = model_first.fittedvalues

    # Second stage
    X_second = sm.add_constant(pd.concat([
        fitted_endogs[[f'{e}_hat' for e in endog_vars]].rename(columns={f'{e}_hat': e for e in endog_vars}),
        df_clean[exog_vars]
    ], axis=1))

    model_second = sm.OLS(y, X_second).fit()

    # Correct standard errors (need to use original endogenous variables for residuals)
    X_original = sm.add_constant(df_clean[endog_vars + exog_vars])
    resid = y - model_second.predict(X_original)
    n = len(y)
    k = X_second.shape[1]
    sigma2 = (resid ** 2).sum() / (n - k)

    # First stage F-statistic for each endogenous variable
    f_stats = {}
    for endog in endog_vars:
        # Get R2 from first stage
        r2_full = first_stages[endog].rsquared
        # R2 without instruments (only exogenous)
        X_reduced = sm.add_constant(df_clean[exog_vars])
        r2_reduced = sm.OLS(df_clean[endog], X_reduced).fit().rsquared

        q = len(instruments)
        n = len(df_clean)
        k = len(exog_vars) + q + 1

        f_stat = ((r2_full - r2_reduced) / q) / ((1 - r2_full) / (n - k))
        f_stats[endog] = f_stat

    return model_second, first_stages, f_stats


def run_nested_logit_demand(df, sigma, demean=True):
    """
    Estimate nested logit demand for a given sigma (nesting parameter).

    Model: log(s_j/s_0) = alpha * price + sigma * log(s_j|g) + X*beta + xi
    """
    # Demean by market
    x_vars = ['direct', 'HUB_orig', 'HUB_dest', 'miles1000']

    # Get airline dummy columns (excluding one for identification)
    airline_cols = [c for c in df.columns if c.startswith('airdum_')][:10]
    all_vars = ['log_s_s0', 'price100', 'log_s_within'] + x_vars + airline_cols

    df_clean = df[all_vars + ['mktid']].dropna()

    if demean:
        df_dm = demean_by_market(df_clean, all_vars)
    else:
        df_dm = df_clean

    # Adjust LHS for sigma
    df_dm['y_adj'] = df_dm['log_s_s0'] - sigma * df_dm['log_s_within']

    # Run regression
    X = sm.add_constant(df_dm[['price100'] + x_vars + airline_cols])
    model = sm.OLS(df_dm['y_adj'], X).fit()

    return model


##############################################################################
# STEP 3: Run Specification Search
##############################################################################

def create_result_dict(spec_id, spec_tree_path, model, treatment_var, outcome_var,
                       sample_desc, fixed_effects, controls_desc, cluster_var,
                       model_type, first_stage_F=None, all_coefs=None):
    """Create a standardized result dictionary."""

    if treatment_var in model.params.index:
        coef = model.params[treatment_var]
        se = model.bse[treatment_var]
        tstat = model.tvalues[treatment_var]
        pval = model.pvalues[treatment_var]
        ci = model.conf_int().loc[treatment_var]
        ci_lower, ci_upper = ci[0], ci[1]
    else:
        coef = se = tstat = pval = ci_lower = ci_upper = np.nan

    # Build coefficient vector JSON
    coef_vector = {
        "treatment": {
            "var": treatment_var,
            "coef": float(coef) if not np.isnan(coef) else None,
            "se": float(se) if not np.isnan(se) else None,
            "pval": float(pval) if not np.isnan(pval) else None
        },
        "controls": [],
        "fixed_effects": fixed_effects.split(", ") if fixed_effects else [],
        "diagnostics": {
            "first_stage_F": first_stage_F,
            "overid_pval": None,
            "hausman_pval": None
        }
    }

    # Add other coefficients
    for var in model.params.index:
        if var not in [treatment_var, 'const']:
            coef_vector["controls"].append({
                "var": var,
                "coef": float(model.params[var]),
                "se": float(model.bse[var]),
                "pval": float(model.pvalues[var])
            })

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
        'n_obs': int(model.nobs),
        'r_squared': model.rsquared,
        'coefficient_vector_json': json.dumps(coef_vector),
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    }


def main():
    print("Processing airlines data...")
    df = process_airlines_data()
    print(f"Data shape: {df.shape}")

    # Define variables
    outcome_var = 'log_s_s0'
    treatment_vars = ['price100', 'log_s_within']  # Both endogenous

    x_vars = ['direct', 'HUB_orig', 'HUB_dest', 'miles1000']
    airline_cols = [c for c in df.columns if c.startswith('airdum_')][:10]
    all_controls = x_vars + airline_cols

    instruments = ['z_sw', 'z_direct', 'z_comphub_a', 'z_leghub_a', 'HUB2_orig', 'HUB2_dest']

    # Demean by market for within-market estimation
    all_reg_vars = [outcome_var] + treatment_vars + all_controls + instruments
    df_dm = demean_by_market(df, all_reg_vars)

    results = []

    ##########################################################################
    # Baseline: OLS (replicating Table in paper)
    ##########################################################################
    print("\n=== Running Baseline OLS ===")

    model_ols = run_ols(df_dm, outcome_var, treatment_vars + all_controls)
    results.append(create_result_dict(
        spec_id='baseline',
        spec_tree_path='methods/instrumental_variables.md#baseline',
        model=model_ols,
        treatment_var='price100',
        outcome_var=outcome_var,
        sample_desc='Full sample, market-demeaned',
        fixed_effects='Market (route-quarter)',
        controls_desc='Direct, hub sizes, distance, airline dummies',
        cluster_var=None,
        model_type='OLS'
    ))

    print(f"OLS price coefficient: {model_ols.params['price100']:.4f} (SE: {model_ols.bse['price100']:.4f})")
    print(f"OLS sigma (log_s_within): {model_ols.params['log_s_within']:.4f}")

    ##########################################################################
    # IV/2SLS Specifications
    ##########################################################################
    print("\n=== Running 2SLS ===")

    # 2SLS with all instruments
    model_2sls, first_stages, f_stats = run_2sls(
        df_dm, outcome_var, treatment_vars, all_controls, instruments
    )

    results.append(create_result_dict(
        spec_id='iv/method/2sls',
        spec_tree_path='methods/instrumental_variables.md#estimation-method',
        model=model_2sls,
        treatment_var='price100',
        outcome_var=outcome_var,
        sample_desc='Full sample, market-demeaned',
        fixed_effects='Market (route-quarter)',
        controls_desc='Direct, hub sizes, distance, airline dummies',
        cluster_var=None,
        model_type='2SLS',
        first_stage_F=f_stats.get('price100')
    ))

    print(f"2SLS price coefficient: {model_2sls.params['price100']:.4f}")
    print(f"2SLS sigma (log_s_within): {model_2sls.params['log_s_within']:.4f}")
    print(f"First stage F (price): {f_stats.get('price100', 'N/A'):.2f}")

    ##########################################################################
    # OLS for comparison (ignoring endogeneity)
    ##########################################################################
    results.append(create_result_dict(
        spec_id='iv/method/ols',
        spec_tree_path='methods/instrumental_variables.md#estimation-method',
        model=model_ols,
        treatment_var='price100',
        outcome_var=outcome_var,
        sample_desc='Full sample, market-demeaned',
        fixed_effects='Market (route-quarter)',
        controls_desc='Direct, hub sizes, distance, airline dummies',
        cluster_var=None,
        model_type='OLS (comparison)'
    ))

    ##########################################################################
    # Reduced form (direct effect of instruments on Y)
    ##########################################################################
    print("\n=== Running Reduced Form ===")

    model_rf = run_ols(df_dm, outcome_var, instruments + all_controls)
    results.append(create_result_dict(
        spec_id='iv/first_stage/reduced_form',
        spec_tree_path='methods/instrumental_variables.md#first-stage',
        model=model_rf,
        treatment_var='z_sw',  # Southwest presence as main IV
        outcome_var=outcome_var,
        sample_desc='Full sample, market-demeaned',
        fixed_effects='Market (route-quarter)',
        controls_desc='All instruments + controls',
        cluster_var=None,
        model_type='Reduced Form'
    ))

    ##########################################################################
    # First Stage for price
    ##########################################################################
    model_first_price = run_ols(df_dm, 'price100', instruments + all_controls)
    results.append(create_result_dict(
        spec_id='iv/first_stage/baseline',
        spec_tree_path='methods/instrumental_variables.md#first-stage',
        model=model_first_price,
        treatment_var='z_sw',
        outcome_var='price100',
        sample_desc='Full sample, market-demeaned',
        fixed_effects='Market (route-quarter)',
        controls_desc='All instruments + controls',
        cluster_var=None,
        model_type='First Stage (price)'
    ))

    ##########################################################################
    # Instrument subsets
    ##########################################################################
    print("\n=== Instrument Subset Variations ===")

    # Single strongest instrument (Southwest)
    model_2sls_sw, _, f_stats_sw = run_2sls(
        df_dm, outcome_var, ['price100'], all_controls + ['log_s_within'], ['z_sw']
    )
    results.append(create_result_dict(
        spec_id='iv/instruments/single',
        spec_tree_path='methods/instrumental_variables.md#instrument-sets',
        model=model_2sls_sw,
        treatment_var='price100',
        outcome_var=outcome_var,
        sample_desc='Full sample, market-demeaned',
        fixed_effects='Market (route-quarter)',
        controls_desc='log_s_within + controls',
        cluster_var=None,
        model_type='2SLS (SW only)',
        first_stage_F=f_stats_sw.get('price100')
    ))

    # Hub-based instruments only
    hub_ivs = ['z_comphub_a', 'z_leghub_a', 'HUB2_orig', 'HUB2_dest']
    model_2sls_hub, _, f_stats_hub = run_2sls(
        df_dm, outcome_var, ['price100'], all_controls + ['log_s_within'], hub_ivs
    )
    results.append(create_result_dict(
        spec_id='iv/instruments/subset',
        spec_tree_path='methods/instrumental_variables.md#instrument-sets',
        model=model_2sls_hub,
        treatment_var='price100',
        outcome_var=outcome_var,
        sample_desc='Full sample, market-demeaned',
        fixed_effects='Market (route-quarter)',
        controls_desc='log_s_within + controls',
        cluster_var=None,
        model_type='2SLS (Hub IVs)',
        first_stage_F=f_stats_hub.get('price100')
    ))

    ##########################################################################
    # Control set variations
    ##########################################################################
    print("\n=== Control Set Variations ===")

    # No controls (besides airline dummies)
    model_ols_nocontrols = run_ols(df_dm, outcome_var, treatment_vars + airline_cols)
    results.append(create_result_dict(
        spec_id='iv/controls/minimal',
        spec_tree_path='methods/instrumental_variables.md#control-sets',
        model=model_ols_nocontrols,
        treatment_var='price100',
        outcome_var=outcome_var,
        sample_desc='Full sample, market-demeaned',
        fixed_effects='Market (route-quarter)',
        controls_desc='Airline dummies only',
        cluster_var=None,
        model_type='OLS'
    ))

    # Full controls (with all x_vars)
    results.append(create_result_dict(
        spec_id='iv/controls/full',
        spec_tree_path='methods/instrumental_variables.md#control-sets',
        model=model_ols,  # Same as baseline
        treatment_var='price100',
        outcome_var=outcome_var,
        sample_desc='Full sample, market-demeaned',
        fixed_effects='Market (route-quarter)',
        controls_desc='Direct, hub sizes, distance, airline dummies',
        cluster_var=None,
        model_type='OLS'
    ))

    ##########################################################################
    # Sample restrictions
    ##########################################################################
    print("\n=== Sample Restrictions ===")

    # Direct flights only
    df_direct = df_dm[df['direct'] == 1].copy()
    if len(df_direct) > 100:
        model_direct = run_ols(df_direct, outcome_var,
                               [v for v in treatment_vars + all_controls if v != 'direct'])
        results.append(create_result_dict(
            spec_id='iv/sample/restricted',
            spec_tree_path='methods/instrumental_variables.md#sample-restrictions',
            model=model_direct,
            treatment_var='price100',
            outcome_var=outcome_var,
            sample_desc='Direct flights only',
            fixed_effects='Market (route-quarter)',
            controls_desc='Hub sizes, distance, airline dummies',
            cluster_var=None,
            model_type='OLS'
        ))

    # Trimmed sample (drop extreme prices)
    p5, p95 = df_dm['price100'].quantile([0.05, 0.95])
    df_trimmed = df_dm[(df_dm['price100'] >= p5) & (df_dm['price100'] <= p95)].copy()
    model_trimmed = run_ols(df_trimmed, outcome_var, treatment_vars + all_controls)
    results.append(create_result_dict(
        spec_id='iv/sample/trimmed',
        spec_tree_path='methods/instrumental_variables.md#sample-restrictions',
        model=model_trimmed,
        treatment_var='price100',
        outcome_var=outcome_var,
        sample_desc='Trimmed 5-95% price',
        fixed_effects='Market (route-quarter)',
        controls_desc='Direct, hub sizes, distance, airline dummies',
        cluster_var=None,
        model_type='OLS'
    ))

    ##########################################################################
    # Robustness: Leave-one-out
    ##########################################################################
    print("\n=== Leave-One-Out Robustness ===")

    for drop_var in x_vars:
        remaining = [v for v in all_controls if v != drop_var]
        model_loo = run_ols(df_dm, outcome_var, treatment_vars + remaining)
        results.append(create_result_dict(
            spec_id=f'robust/loo/drop_{drop_var}',
            spec_tree_path='robustness/leave_one_out.md',
            model=model_loo,
            treatment_var='price100',
            outcome_var=outcome_var,
            sample_desc='Full sample, market-demeaned',
            fixed_effects='Market (route-quarter)',
            controls_desc=f'Dropped: {drop_var}',
            cluster_var=None,
            model_type='OLS'
        ))

    ##########################################################################
    # Robustness: Single covariate
    ##########################################################################
    print("\n=== Single Covariate Robustness ===")

    # Bivariate (no controls besides treatment vars)
    model_bivar = run_ols(df_dm, outcome_var, treatment_vars)
    results.append(create_result_dict(
        spec_id='robust/single/none',
        spec_tree_path='robustness/single_covariate.md',
        model=model_bivar,
        treatment_var='price100',
        outcome_var=outcome_var,
        sample_desc='Full sample, market-demeaned',
        fixed_effects='Market (route-quarter)',
        controls_desc='No controls',
        cluster_var=None,
        model_type='OLS'
    ))

    for single_var in x_vars:
        model_single = run_ols(df_dm, outcome_var, treatment_vars + [single_var])
        results.append(create_result_dict(
            spec_id=f'robust/single/{single_var}',
            spec_tree_path='robustness/single_covariate.md',
            model=model_single,
            treatment_var='price100',
            outcome_var=outcome_var,
            sample_desc='Full sample, market-demeaned',
            fixed_effects='Market (route-quarter)',
            controls_desc=f'Only: {single_var}',
            cluster_var=None,
            model_type='OLS'
        ))

    ##########################################################################
    # Robustness: Clustering variations
    ##########################################################################
    print("\n=== Clustering Variations ===")

    # Add cluster variables back to demeaned data
    df_dm['route_id'] = df['route_id'].values
    df_dm['quarter'] = df['quarter'].values
    df_dm['firmid'] = df['firmid'].values

    # Cluster by route
    model_cluster_route = run_ols(df_dm, outcome_var, treatment_vars + all_controls,
                                   cluster_var='route_id')
    results.append(create_result_dict(
        spec_id='robust/cluster/unit',
        spec_tree_path='robustness/clustering_variations.md',
        model=model_cluster_route,
        treatment_var='price100',
        outcome_var=outcome_var,
        sample_desc='Full sample, market-demeaned',
        fixed_effects='Market (route-quarter)',
        controls_desc='Direct, hub sizes, distance, airline dummies',
        cluster_var='route_id',
        model_type='OLS (clustered)'
    ))

    # Cluster by airline
    model_cluster_airline = run_ols(df_dm, outcome_var, treatment_vars + all_controls,
                                     cluster_var='firmid')
    results.append(create_result_dict(
        spec_id='robust/cluster/industry',
        spec_tree_path='robustness/clustering_variations.md',
        model=model_cluster_airline,
        treatment_var='price100',
        outcome_var=outcome_var,
        sample_desc='Full sample, market-demeaned',
        fixed_effects='Market (route-quarter)',
        controls_desc='Direct, hub sizes, distance, airline dummies',
        cluster_var='firmid (airline)',
        model_type='OLS (clustered)'
    ))

    ##########################################################################
    # Functional form variations
    ##########################################################################
    print("\n=== Functional Form Variations ===")

    # Log price (instead of price100)
    df_dm['log_price'] = np.log(df['price'].clip(lower=0.01)).values
    df_dm['log_price'] = df_dm.groupby('mktid')['log_price'].transform(lambda x: x - x.mean())

    # Clean up any inf/nan values
    df_logprice = df_dm.dropna(subset=['log_price'] + [outcome_var] + ['log_s_within'] + all_controls)
    df_logprice = df_logprice.replace([np.inf, -np.inf], np.nan).dropna()

    model_logprice = run_ols(df_logprice,
                              outcome_var, ['log_price', 'log_s_within'] + all_controls)
    results.append(create_result_dict(
        spec_id='robust/functional/log_price',
        spec_tree_path='robustness/functional_form.md',
        model=model_logprice,
        treatment_var='log_price',
        outcome_var=outcome_var,
        sample_desc='Full sample, market-demeaned',
        fixed_effects='Market (route-quarter)',
        controls_desc='Direct, hub sizes, distance, airline dummies',
        cluster_var=None,
        model_type='OLS (log price)'
    ))

    # Quadratic price
    df_dm['price100_sq'] = df_dm['price100'] ** 2
    model_quad = run_ols(df_dm, outcome_var,
                          ['price100', 'price100_sq', 'log_s_within'] + all_controls)
    results.append(create_result_dict(
        spec_id='robust/functional/quadratic',
        spec_tree_path='robustness/functional_form.md',
        model=model_quad,
        treatment_var='price100',
        outcome_var=outcome_var,
        sample_desc='Full sample, market-demeaned',
        fixed_effects='Market (route-quarter)',
        controls_desc='Quadratic price + controls',
        cluster_var=None,
        model_type='OLS (quadratic)'
    ))

    ##########################################################################
    # Different nesting structures (sigma values)
    ##########################################################################
    print("\n=== Nesting Parameter Variations ===")

    for sigma in [0.6, 0.7, 0.8, 0.9]:
        model_sigma = run_nested_logit_demand(df, sigma)
        results.append(create_result_dict(
            spec_id=f'custom/sigma_{sigma}',
            spec_tree_path='methods/instrumental_variables.md',
            model=model_sigma,
            treatment_var='price100',
            outcome_var='y_adj',
            sample_desc=f'Full sample, sigma={sigma}',
            fixed_effects='Market (route-quarter)',
            controls_desc='Direct, hub sizes, distance, airline dummies',
            cluster_var=None,
            model_type=f'Nested Logit (sigma={sigma})'
        ))

    ##########################################################################
    # Save Results
    ##########################################################################
    print("\n=== Saving Results ===")

    results_df = pd.DataFrame(results)

    # Save to package directory
    output_path = f"{BASE_PATH}/data/downloads/extracted/{PAPER_ID}/specification_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Saved {len(results)} specifications to {output_path}")

    # Summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total specifications: {len(results)}")
    print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
    print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
    print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")
    print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
    print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")
    print(f"Range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")

    return results_df


if __name__ == "__main__":
    results_df = main()
