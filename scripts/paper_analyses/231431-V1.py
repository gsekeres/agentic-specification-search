#!/usr/bin/env python3
"""
Paper Analysis: 231431-V1
=========================
Parker & Souleles (2019) "Reported Effects vs. Revealed-Preference Estimates:
Evidence from the Propensity to Spend Tax Rebates"
AER: Insights, Vol. 1 No. 2

This paper estimates the marginal propensity to consume (MPC) from 2008
Economic Stimulus Payments (ESP) using Consumer Expenditure Survey (CEX) data.
The key specification is an IV/2SLS regression where ESP dollar amounts are
instrumented by an indicator for ESP receipt (random timing).

Baseline specification (Appendix 2 / Table 3 Column 3):
    dcn = esp + lesp + l2esp + age + dnad + dnkd + yymm_FE
    instruments: iesp, liesp, l2iesp for esp, lesp, l2esp
    Weights: analytic weights (weight)
    Clustering: by household (newid)

Primary method: Instrumental Variables (2SLS)
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS, IVLIML, IVGMM
from scipy import stats

# =============================================================================
# CONFIGURATION
# =============================================================================

PAPER_ID = "231431-V1"
JOURNAL = "AER: Insights"
PAPER_TITLE = "Reported Effects vs Revealed-Preference Estimates: Evidence from the Propensity to Spend Tax Rebates"
METHOD_TYPE = "instrumental_variables"

BASE_DIR = Path("/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search")
PACKAGE_DIR = BASE_DIR / "data" / "downloads" / "extracted" / PAPER_ID
DATA_DIR = PACKAGE_DIR / "Replication_Parker_AERI" / "CEX" / "data"
OUTPUT_FILE = PACKAGE_DIR / "specification_results.csv"

# Key variables
OUTCOME_VARS = {
    'dcn': 'Change in nondurable consumption (dollars)',
    'dctsmed': 'Change in total consumption incl. some durables (dollars)',
    'dlcn': 'Log change in nondurable consumption (x100, percent)',
    'dlctsmed': 'Log change in total consumption (x100, percent)',
}
TREATMENT_ENDOGENOUS = ['esp', 'lesp', 'l2esp']
TREATMENT_INSTRUMENTS = ['iesp', 'liesp', 'l2iesp']
CONTROL_VARS = ['age', 'dnad', 'dnkd']
CLUSTER_VAR = 'newid'


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load and prepare the data, replicating Stata merge + filter logic."""
    df = pd.read_stata(DATA_DIR / "PSJMReplication.dta")
    ssquest = pd.read_stata(DATA_DIR / "SSquest.dta")
    flag = pd.read_stata(DATA_DIR / "flagl2iespmissMJJ.dta")

    # Merge m:1 on newid
    df = df.merge(ssquest, on='newid', how='left')

    # Merge 1:1 on newid intview (keep matched only)
    df = df.merge(flag, on=['newid', 'intview'], how='inner')

    # Create sample filter (replicates Stata: samplerprp = baselines3 & flagl2iespmissMJJ ~= 1)
    df['samplerprp'] = (df['baselines3'] == 1) & (df['flagl2iespmissMJJ'] != 1)
    df = df[df['samplerprp']].copy()

    # Scale log changes to percent (replicating Stata: replace dlcn = 100*dlcn)
    df['dlcn'] = 100 * df['dlcn']
    df['dlctsmed'] = 100 * df['dlctsmed']

    # Create time FE dummies
    yymm_vals = sorted(df['yymm'].unique())
    # Drop first dummy for identification (Stata drops ym1)
    for i, val in enumerate(yymm_vals[1:], start=2):
        df[f'ym{i}'] = (df['yymm'] == val).astype(float)
    yymm_dummies = [f'ym{i}' for i in range(2, len(yymm_vals) + 1)]

    # Create howused indicators for interaction regressions
    df['allhh'] = 1
    for var in TREATMENT_ENDOGENOUS + TREATMENT_INSTRUMENTS:
        df[f'spend{var}'] = (df['howused'] == 1).astype(float) * df[var]
        df[f'nspend{var}'] = (df['howused'] != 1).astype(float) * df[var]
        df[f'save{var}'] = (df['howused'] == 2).astype(float) * df[var]
        df[f'payd{var}'] = (df['howused'] == 3).astype(float) * df[var]

    # Income terciles
    hh_df = df.drop_duplicates(subset='newid')
    inc_valid = hh_df.loc[hh_df['income'].notna(), 'income']
    df['income3'] = np.nan
    df.loc[(df['income'] < 35000) & df['income'].notna(), 'income3'] = 1
    df.loc[(df['income'] >= 35000) & (df['income'] < 70000) & df['income'].notna(), 'income3'] = 2
    df.loc[(df['income'] >= 70000) & df['income'].notna(), 'income3'] = 3

    df['low3inc'] = (df['income3'] == 1).astype(float)
    df['mid3inc'] = (df['income3'] == 2).astype(float)
    df['high3inc'] = (df['income3'] == 3).astype(float)

    # Liquidity split
    df['liqassii2'] = np.nan
    df.loc[(df['liqassii'] < 2000) & df['liqassii'].notna(), 'liqassii2'] = 1
    df.loc[(df['liqassii'] >= 2000) & df['liqassii'].notna(), 'liqassii2'] = 2

    df['low2liq'] = ((df['liqassii2'] == 1)).astype(float)
    df['high2liq'] = ((df['liqassii2'] == 2)).astype(float)

    return df, yymm_dummies


# =============================================================================
# REGRESSION RUNNERS
# =============================================================================

def run_iv_2sls(df, outcome, endogenous, instruments, controls, yymm_dummies,
                cluster_var, weights_var='weight', method='2sls'):
    """
    Run IV/2SLS regression using linearmodels.
    Returns coefficient dict with all relevant info.
    """
    data = df.dropna(subset=[outcome] + endogenous + instruments + controls + [cluster_var, weights_var]).copy()
    if len(data) == 0:
        return None

    y = data[outcome]
    # Exogenous: constant + controls + yymm dummies
    exog_cols = ['const'] + controls + yymm_dummies
    data['const'] = 1.0
    X_exog = data[exog_cols]
    X_endog = data[endogenous]
    Z = data[instruments]

    w = data[weights_var]
    # Apply analytic weights (multiply by sqrt of weight)
    sw = np.sqrt(w)
    y_w = y * sw
    X_exog_w = X_exog.multiply(sw, axis=0)
    X_endog_w = X_endog.multiply(sw, axis=0)
    Z_w = Z.multiply(sw, axis=0)

    try:
        if method == '2sls':
            model = IV2SLS(y_w, X_exog_w, X_endog_w, Z_w)
        elif method == 'liml':
            model = IVLIML(y_w, X_exog_w, X_endog_w, Z_w)
        elif method == 'gmm':
            model = IVGMM(y_w, X_exog_w, X_endog_w, Z_w)
        else:
            model = IV2SLS(y_w, X_exog_w, X_endog_w, Z_w)

        result = model.fit(cov_type='clustered', clusters=data[cluster_var])
        return result, data
    except Exception as e:
        print(f"  IV error: {e}")
        return None, data


def run_areg(df, outcome, regressors, controls, yymm_dummies, cluster_var,
             weights_var='weight', absorb='yymm'):
    """
    Run absorbed FE regression (like Stata areg ... , a(yymm)).
    Uses WLS with time dummies.
    """
    data = df.dropna(subset=[outcome] + regressors + controls + [cluster_var, weights_var]).copy()
    if len(data) == 0:
        return None

    all_rhs = regressors + controls + yymm_dummies
    data['const'] = 1.0
    all_rhs_with_const = ['const'] + all_rhs

    y = data[outcome]
    X = data[all_rhs_with_const]

    try:
        model = sm.WLS(y, X, weights=data[weights_var])
        result = model.fit(cov_type='cluster', cov_kwds={'groups': data[cluster_var]})
        return result, data
    except Exception as e:
        print(f"  areg error: {e}")
        return None, data


def run_ols_simple(df, outcome, regressors, controls, yymm_dummies, cluster_var,
                   weights_var='weight'):
    """Run simple OLS/WLS regression."""
    data = df.dropna(subset=[outcome] + regressors + controls + [cluster_var, weights_var]).copy()
    if len(data) == 0:
        return None

    all_rhs = regressors + controls + yymm_dummies
    data['const'] = 1.0
    all_rhs_with_const = ['const'] + all_rhs

    y = data[outcome]
    X = data[all_rhs_with_const]

    try:
        model = sm.WLS(y, X, weights=data[weights_var])
        result = model.fit(cov_type='cluster', cov_kwds={'groups': data[cluster_var]})
        return result, data
    except Exception as e:
        print(f"  OLS error: {e}")
        return None, data


# =============================================================================
# RESULT FORMATTER
# =============================================================================

def format_iv_result(result, data, outcome, treatment_var, spec_id, spec_tree_path,
                     sample_desc, controls, fe_desc, cluster_var, method='2sls',
                     endogenous=None, instruments=None):
    """Format IV result into standard output dict."""
    if result is None:
        return None
    if endogenous is None:
        endogenous = TREATMENT_ENDOGENOUS
    if instruments is None:
        instruments = TREATMENT_INSTRUMENTS

    # Primary treatment variable is first endogenous var
    tv = treatment_var if treatment_var in result.params.index else endogenous[0]
    try:
        coef = float(result.params[tv])
        se = float(result.std_errors[tv])
        pval = float(result.pvalues[tv])
    except KeyError:
        print(f"  Warning: {tv} not in results for {spec_id}")
        return None

    tstat = coef / se if se > 0 else np.nan
    ci_lower = coef - 1.96 * se
    ci_upper = coef + 1.96 * se
    n_obs = int(result.nobs)
    r2 = float(result.r2) if hasattr(result, 'r2') else np.nan
    n_clusters = data[cluster_var].nunique()

    # Build coefficient vector
    coef_vector = {
        "treatment": {
            "var": tv,
            "coef": coef,
            "se": se,
            "pval": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        },
        "controls": [],
        "fixed_effects_absorbed": ["yymm"],
        "diagnostics": {
            "n_instruments": len(instruments),
            "n_endogenous": len(endogenous),
            "method": method,
            "n_clusters": n_clusters,
        }
    }

    # Add other endogenous vars
    for ev in endogenous:
        if ev != tv and ev in result.params.index:
            coef_vector[f"endogenous_{ev}"] = {
                "var": ev,
                "coef": float(result.params[ev]),
                "se": float(result.std_errors[ev]),
                "pval": float(result.pvalues[ev]),
            }

    # Add control coefficients
    for ctrl in controls:
        if ctrl in result.params.index:
            coef_vector["controls"].append({
                "var": ctrl,
                "coef": float(result.params[ctrl]),
                "se": float(result.std_errors[ctrl]),
                "pval": float(result.pvalues[ctrl]),
            })

    # Compute cumulative MPC (sum of distributed lag coefficients)
    cum_coefs = []
    for ev in endogenous:
        if ev in result.params.index:
            cum_coefs.append(float(result.params[ev]))
    if cum_coefs:
        # Cumulative MPC over 3 quarters = 3*esp + 2*lesp + l2esp
        if len(cum_coefs) == 3:
            cum3 = 3 * cum_coefs[0] + 2 * cum_coefs[1] + cum_coefs[2]
            coef_vector["diagnostics"]["cumulative_mpc_3q"] = cum3

    return {
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'outcome_var': outcome,
        'treatment_var': tv,
        'coefficient': coef,
        'std_error': se,
        't_stat': tstat,
        'p_value': pval,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_obs': n_obs,
        'r_squared': r2,
        'coefficient_vector_json': json.dumps(coef_vector),
        'sample_desc': sample_desc,
        'fixed_effects': fe_desc,
        'controls_desc': ", ".join(controls) if controls else "None",
        'cluster_var': cluster_var,
        'model_type': method.upper(),
        'estimation_script': f"scripts/paper_analyses/{PAPER_ID}.py"
    }


def format_ols_result(result, data, outcome, treatment_var, spec_id, spec_tree_path,
                      sample_desc, controls, fe_desc, cluster_var):
    """Format OLS/WLS result into standard output dict."""
    if result is None:
        return None

    try:
        coef = float(result.params[treatment_var])
        se = float(result.bse[treatment_var])
        pval = float(result.pvalues[treatment_var])
    except KeyError:
        print(f"  Warning: {treatment_var} not in results for {spec_id}")
        return None

    tstat = float(result.tvalues[treatment_var])
    ci_lower = coef - 1.96 * se
    ci_upper = coef + 1.96 * se
    n_obs = int(result.nobs)
    r2 = float(result.rsquared)
    n_clusters = data[cluster_var].nunique()

    coef_vector = {
        "treatment": {
            "var": treatment_var,
            "coef": coef,
            "se": se,
            "pval": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        },
        "controls": [],
        "fixed_effects_absorbed": ["yymm"],
        "diagnostics": {
            "method": "OLS/WLS",
            "n_clusters": n_clusters,
        }
    }

    for ctrl in controls:
        if ctrl in result.params.index:
            coef_vector["controls"].append({
                "var": ctrl,
                "coef": float(result.params[ctrl]),
                "se": float(result.bse[ctrl]),
                "pval": float(result.pvalues[ctrl]),
            })

    return {
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'outcome_var': outcome,
        'treatment_var': treatment_var,
        'coefficient': coef,
        'std_error': se,
        't_stat': tstat,
        'p_value': pval,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_obs': n_obs,
        'r_squared': r2,
        'coefficient_vector_json': json.dumps(coef_vector),
        'sample_desc': sample_desc,
        'fixed_effects': fe_desc,
        'controls_desc': ", ".join(controls) if controls else "None",
        'cluster_var': cluster_var,
        'model_type': "OLS",
        'estimation_script': f"scripts/paper_analyses/{PAPER_ID}.py"
    }


# =============================================================================
# SPECIFICATION SEARCH
# =============================================================================

def run_all_specifications(df, yymm_dummies):
    """Run all specifications."""
    results = []
    spec_count = 0

    def add_result(r):
        nonlocal spec_count
        if r is not None:
            results.append(r)
            spec_count += 1
            print(f"  [{spec_count}] {r['spec_id']}: coef={r['coefficient']:.4f}, se={r['std_error']:.4f}, p={r['p_value']:.4f}, n={r['n_obs']}")

    # =========================================================================
    # 1. BASELINE: IV/2SLS -- Table 3/Appendix 2 Column 3 (MPC, nondurable)
    # =========================================================================
    print("\n--- BASELINE REPLICATIONS ---")

    # Baseline: dcn ~ esp + lesp + l2esp (IV by iesp, liesp, l2iesp) + controls + yymm FE
    res, dat = run_iv_2sls(df, 'dcn', TREATMENT_ENDOGENOUS, TREATMENT_INSTRUMENTS,
                           CONTROL_VARS, yymm_dummies, CLUSTER_VAR)
    add_result(format_iv_result(res, dat, 'dcn', 'esp', 'baseline',
                                'methods/instrumental_variables.md',
                                'Full sample, nondurable consumption',
                                CONTROL_VARS, 'yymm', CLUSTER_VAR))

    # Baseline variant: total consumption
    res, dat = run_iv_2sls(df, 'dctsmed', TREATMENT_ENDOGENOUS, TREATMENT_INSTRUMENTS,
                           CONTROL_VARS, yymm_dummies, CLUSTER_VAR)
    add_result(format_iv_result(res, dat, 'dctsmed', 'esp', 'baseline_total_consumption',
                                'methods/instrumental_variables.md',
                                'Full sample, total consumption',
                                CONTROL_VARS, 'yymm', CLUSTER_VAR))

    # =========================================================================
    # 2. IV METHOD VARIATIONS
    # =========================================================================
    print("\n--- IV METHOD VARIATIONS ---")

    # 2a. LIML
    res, dat = run_iv_2sls(df, 'dcn', TREATMENT_ENDOGENOUS, TREATMENT_INSTRUMENTS,
                           CONTROL_VARS, yymm_dummies, CLUSTER_VAR, method='liml')
    add_result(format_iv_result(res, dat, 'dcn', 'esp', 'iv/method/liml',
                                'methods/instrumental_variables.md#estimation-method',
                                'Full sample, nondurable consumption',
                                CONTROL_VARS, 'yymm', CLUSTER_VAR, method='liml'))

    # 2b. GMM
    res, dat = run_iv_2sls(df, 'dcn', TREATMENT_ENDOGENOUS, TREATMENT_INSTRUMENTS,
                           CONTROL_VARS, yymm_dummies, CLUSTER_VAR, method='gmm')
    add_result(format_iv_result(res, dat, 'dcn', 'esp', 'iv/method/gmm_2step',
                                'methods/instrumental_variables.md#estimation-method',
                                'Full sample, nondurable consumption',
                                CONTROL_VARS, 'yymm', CLUSTER_VAR, method='gmm'))

    # 2c. OLS (ignoring endogeneity -- reduced form with indicator)
    res, dat = run_areg(df, 'dcn', TREATMENT_INSTRUMENTS, CONTROL_VARS, yymm_dummies, CLUSTER_VAR)
    add_result(format_ols_result(res, dat, 'dcn', 'iesp', 'iv/method/ols_indicator',
                                 'methods/instrumental_variables.md#estimation-method',
                                 'Full sample, OLS with ESP indicator',
                                 CONTROL_VARS, 'yymm', CLUSTER_VAR))

    # 2d. Reduced form (direct effect of instrument on outcome)
    res, dat = run_areg(df, 'dcn', TREATMENT_INSTRUMENTS, CONTROL_VARS, yymm_dummies, CLUSTER_VAR)
    add_result(format_ols_result(res, dat, 'dcn', 'iesp', 'iv/first_stage/reduced_form',
                                 'methods/instrumental_variables.md#first-stage',
                                 'Reduced form: effect of ESP indicator on consumption',
                                 CONTROL_VARS, 'yymm', CLUSTER_VAR))

    # 2e. First stage: esp on iesp
    res, dat = run_areg(df, 'esp', ['iesp'], CONTROL_VARS, yymm_dummies, CLUSTER_VAR)
    add_result(format_ols_result(res, dat, 'esp', 'iesp', 'iv/first_stage/baseline',
                                 'methods/instrumental_variables.md#first-stage',
                                 'First stage: ESP amount on ESP indicator',
                                 CONTROL_VARS, 'yymm', CLUSTER_VAR))

    # =========================================================================
    # 3. OUTCOME VARIABLE VARIATIONS
    # =========================================================================
    print("\n--- OUTCOME VARIATIONS ---")

    # 3a. Log nondurable (percent)
    res, dat = run_areg(df, 'dlcn', TREATMENT_INSTRUMENTS, CONTROL_VARS, yymm_dummies, CLUSTER_VAR)
    add_result(format_ols_result(res, dat, 'dlcn', 'iesp', 'robust/form/y_log_nondurable',
                                 'robustness/functional_form.md',
                                 'Log nondurable consumption (percent)',
                                 CONTROL_VARS, 'yymm', CLUSTER_VAR))

    # 3b. Log total consumption (percent)
    res, dat = run_areg(df, 'dlctsmed', TREATMENT_INSTRUMENTS, CONTROL_VARS, yymm_dummies, CLUSTER_VAR)
    add_result(format_ols_result(res, dat, 'dlctsmed', 'iesp', 'robust/form/y_log_total',
                                 'robustness/functional_form.md',
                                 'Log total consumption (percent)',
                                 CONTROL_VARS, 'yymm', CLUSTER_VAR))

    # 3c. IV on dctsmed (total, MPC)
    res, dat = run_iv_2sls(df, 'dctsmed', TREATMENT_ENDOGENOUS, TREATMENT_INSTRUMENTS,
                           CONTROL_VARS, yymm_dummies, CLUSTER_VAR)
    add_result(format_iv_result(res, dat, 'dctsmed', 'esp', 'robust/form/y_total_iv',
                                'robustness/functional_form.md',
                                'Total consumption, IV/2SLS',
                                CONTROL_VARS, 'yymm', CLUSTER_VAR))

    # 3d. Dollar nondurable (indicator, areg)
    res, dat = run_areg(df, 'dcn', TREATMENT_INSTRUMENTS, CONTROL_VARS, yymm_dummies, CLUSTER_VAR)
    add_result(format_ols_result(res, dat, 'dcn', 'iesp', 'robust/form/y_dollar_indicator',
                                 'robustness/functional_form.md',
                                 'Dollar nondurable, indicator treatment',
                                 CONTROL_VARS, 'yymm', CLUSTER_VAR))

    # 3e. Dollar total (indicator, areg)
    res, dat = run_areg(df, 'dctsmed', TREATMENT_INSTRUMENTS, CONTROL_VARS, yymm_dummies, CLUSTER_VAR)
    add_result(format_ols_result(res, dat, 'dctsmed', 'iesp', 'robust/form/y_dollar_total_indicator',
                                 'robustness/functional_form.md',
                                 'Dollar total consumption, indicator treatment',
                                 CONTROL_VARS, 'yymm', CLUSTER_VAR))

    # 3f-3i. Subcategories of consumption
    for subcat in ['dcfdhome', 'dcfdaway', 'dcappar', 'dchealth']:
        if subcat in df.columns:
            res, dat = run_iv_2sls(df, subcat, TREATMENT_ENDOGENOUS, TREATMENT_INSTRUMENTS,
                                   CONTROL_VARS, yymm_dummies, CLUSTER_VAR)
            add_result(format_iv_result(res, dat, subcat, 'esp',
                                        f'robust/form/y_{subcat}',
                                        'robustness/functional_form.md',
                                        f'{subcat} subcategory, IV/2SLS',
                                        CONTROL_VARS, 'yymm', CLUSTER_VAR))

    # =========================================================================
    # 4. CONTROL SET VARIATIONS (Leave-one-out)
    # =========================================================================
    print("\n--- LEAVE-ONE-OUT CONTROLS ---")

    for ctrl in CONTROL_VARS:
        remaining = [c for c in CONTROL_VARS if c != ctrl]
        res, dat = run_iv_2sls(df, 'dcn', TREATMENT_ENDOGENOUS, TREATMENT_INSTRUMENTS,
                               remaining, yymm_dummies, CLUSTER_VAR)
        add_result(format_iv_result(res, dat, 'dcn', 'esp',
                                    f'robust/loo/drop_{ctrl}',
                                    'robustness/leave_one_out.md',
                                    f'Drop {ctrl}',
                                    remaining, 'yymm', CLUSTER_VAR))

    # No controls at all
    res, dat = run_iv_2sls(df, 'dcn', TREATMENT_ENDOGENOUS, TREATMENT_INSTRUMENTS,
                           [], yymm_dummies, CLUSTER_VAR)
    add_result(format_iv_result(res, dat, 'dcn', 'esp', 'iv/controls/none',
                                'methods/instrumental_variables.md#control-sets',
                                'No demographic controls',
                                [], 'yymm', CLUSTER_VAR))

    # =========================================================================
    # 5. SINGLE COVARIATE REGRESSIONS
    # =========================================================================
    print("\n--- SINGLE COVARIATE ---")

    for ctrl in CONTROL_VARS:
        res, dat = run_iv_2sls(df, 'dcn', TREATMENT_ENDOGENOUS, TREATMENT_INSTRUMENTS,
                               [ctrl], yymm_dummies, CLUSTER_VAR)
        add_result(format_iv_result(res, dat, 'dcn', 'esp',
                                    f'robust/single/{ctrl}',
                                    'robustness/single_covariate.md',
                                    f'Only {ctrl} as control',
                                    [ctrl], 'yymm', CLUSTER_VAR))

    # =========================================================================
    # 6. FIXED EFFECTS VARIATIONS
    # =========================================================================
    print("\n--- FIXED EFFECTS VARIATIONS ---")

    # No time FE (no yymm dummies)
    res, dat = run_iv_2sls(df, 'dcn', TREATMENT_ENDOGENOUS, TREATMENT_INSTRUMENTS,
                           CONTROL_VARS, [], CLUSTER_VAR)
    add_result(format_iv_result(res, dat, 'dcn', 'esp', 'iv/fe/none',
                                'methods/instrumental_variables.md#fixed-effects',
                                'No time fixed effects',
                                CONTROL_VARS, 'None', CLUSTER_VAR))

    # =========================================================================
    # 7. CLUSTERING VARIATIONS
    # =========================================================================
    print("\n--- CLUSTERING VARIATIONS ---")

    # 7a. Robust SE (no clustering)
    data_tmp = df.dropna(subset=['dcn'] + TREATMENT_ENDOGENOUS + TREATMENT_INSTRUMENTS +
                         CONTROL_VARS + [CLUSTER_VAR, 'weight']).copy()
    data_tmp['const'] = 1.0
    exog_cols = ['const'] + CONTROL_VARS + yymm_dummies
    sw = np.sqrt(data_tmp['weight'])
    try:
        model_hc = IV2SLS(data_tmp['dcn'] * sw,
                          data_tmp[exog_cols].multiply(sw, axis=0),
                          data_tmp[TREATMENT_ENDOGENOUS].multiply(sw, axis=0),
                          data_tmp[TREATMENT_INSTRUMENTS].multiply(sw, axis=0))
        res_hc = model_hc.fit(cov_type='robust')
        add_result(format_iv_result(res_hc, data_tmp, 'dcn', 'esp', 'robust/cluster/none',
                                    'robustness/clustering_variations.md',
                                    'Robust SE, no clustering',
                                    CONTROL_VARS, 'yymm', CLUSTER_VAR))
    except Exception as e:
        print(f"  HC error: {e}")

    # 7b. Cluster by yymm (time period)
    res, dat = run_iv_2sls(df, 'dcn', TREATMENT_ENDOGENOUS, TREATMENT_INSTRUMENTS,
                           CONTROL_VARS, yymm_dummies, 'yymm')
    if res is not None:
        add_result(format_iv_result(res, dat, 'dcn', 'esp', 'robust/cluster/time',
                                    'robustness/clustering_variations.md',
                                    'Clustered by time period',
                                    CONTROL_VARS, 'yymm', 'yymm'))

    # 7c. Robust OLS SEs (for indicator-based regression)
    res, dat = run_areg(df, 'dcn', TREATMENT_INSTRUMENTS, CONTROL_VARS, yymm_dummies, CLUSTER_VAR)
    if res is not None:
        # Re-run with HC1 (no clustering)
        data_hc = df.dropna(subset=['dcn'] + TREATMENT_INSTRUMENTS + CONTROL_VARS +
                            [CLUSTER_VAR, 'weight']).copy()
        data_hc['const'] = 1.0
        all_rhs = ['const'] + TREATMENT_INSTRUMENTS + CONTROL_VARS + yymm_dummies
        try:
            model_hc1 = sm.WLS(data_hc['dcn'], data_hc[all_rhs], weights=data_hc['weight'])
            res_hc1 = model_hc1.fit(cov_type='HC1')
            add_result(format_ols_result(res_hc1, data_hc, 'dcn', 'iesp', 'robust/se/hc1',
                                         'robustness/clustering_variations.md',
                                         'HC1 robust SE, indicator treatment',
                                         CONTROL_VARS, 'yymm', CLUSTER_VAR))
        except Exception as e:
            print(f"  HC1 error: {e}")

    # =========================================================================
    # 8. SAMPLE RESTRICTIONS
    # =========================================================================
    print("\n--- SAMPLE RESTRICTIONS ---")

    # 8a. Income tercile subsamples
    for inc_label, inc_val in [('low_income', 1), ('mid_income', 2), ('high_income', 3)]:
        df_sub = df[df['income3'] == inc_val].copy()
        if len(df_sub) > 50:
            res, dat = run_iv_2sls(df_sub, 'dcn', TREATMENT_ENDOGENOUS, TREATMENT_INSTRUMENTS,
                                   CONTROL_VARS, yymm_dummies, CLUSTER_VAR)
            add_result(format_iv_result(res, dat, 'dcn', 'esp',
                                        f'robust/sample/{inc_label}',
                                        'robustness/sample_restrictions.md',
                                        f'Income tercile: {inc_label}',
                                        CONTROL_VARS, 'yymm', CLUSTER_VAR))

    # 8b. Income tercile with total consumption
    for inc_label, inc_val in [('low_income', 1), ('mid_income', 2), ('high_income', 3)]:
        df_sub = df[df['income3'] == inc_val].copy()
        if len(df_sub) > 50:
            res, dat = run_iv_2sls(df_sub, 'dctsmed', TREATMENT_ENDOGENOUS, TREATMENT_INSTRUMENTS,
                                   CONTROL_VARS, yymm_dummies, CLUSTER_VAR)
            add_result(format_iv_result(res, dat, 'dctsmed', 'esp',
                                        f'robust/sample/{inc_label}_total',
                                        'robustness/sample_restrictions.md',
                                        f'Income tercile: {inc_label}, total consumption',
                                        CONTROL_VARS, 'yymm', CLUSTER_VAR))

    # 8c. Liquidity splits
    for liq_label, liq_val in [('low_liquidity', 1), ('high_liquidity', 2)]:
        df_sub = df[df['liqassii2'] == liq_val].copy()
        if len(df_sub) > 50:
            res, dat = run_iv_2sls(df_sub, 'dcn', TREATMENT_ENDOGENOUS, TREATMENT_INSTRUMENTS,
                                   CONTROL_VARS, yymm_dummies, CLUSTER_VAR)
            add_result(format_iv_result(res, dat, 'dcn', 'esp',
                                        f'robust/sample/{liq_label}',
                                        'robustness/sample_restrictions.md',
                                        f'Liquidity split: {liq_label}',
                                        CONTROL_VARS, 'yymm', CLUSTER_VAR))

    # 8d. Liquidity splits with total consumption
    for liq_label, liq_val in [('low_liquidity', 1), ('high_liquidity', 2)]:
        df_sub = df[df['liqassii2'] == liq_val].copy()
        if len(df_sub) > 50:
            res, dat = run_iv_2sls(df_sub, 'dctsmed', TREATMENT_ENDOGENOUS, TREATMENT_INSTRUMENTS,
                                   CONTROL_VARS, yymm_dummies, CLUSTER_VAR)
            add_result(format_iv_result(res, dat, 'dctsmed', 'esp',
                                        f'robust/sample/{liq_label}_total',
                                        'robustness/sample_restrictions.md',
                                        f'Liquidity split: {liq_label}, total consumption',
                                        CONTROL_VARS, 'yymm', CLUSTER_VAR))

    # 8e. Trim outliers: drop top/bottom 1% of dcn
    q01 = df['dcn'].quantile(0.01)
    q99 = df['dcn'].quantile(0.99)
    df_trim = df[(df['dcn'] >= q01) & (df['dcn'] <= q99)].copy()
    res, dat = run_iv_2sls(df_trim, 'dcn', TREATMENT_ENDOGENOUS, TREATMENT_INSTRUMENTS,
                           CONTROL_VARS, yymm_dummies, CLUSTER_VAR)
    add_result(format_iv_result(res, dat, 'dcn', 'esp', 'robust/sample/trimmed_1pct',
                                'robustness/sample_restrictions.md',
                                'Trimmed top/bottom 1% of dcn',
                                CONTROL_VARS, 'yymm', CLUSTER_VAR))

    # 8f. Trim 5%
    q05 = df['dcn'].quantile(0.05)
    q95 = df['dcn'].quantile(0.95)
    df_trim5 = df[(df['dcn'] >= q05) & (df['dcn'] <= q95)].copy()
    res, dat = run_iv_2sls(df_trim5, 'dcn', TREATMENT_ENDOGENOUS, TREATMENT_INSTRUMENTS,
                           CONTROL_VARS, yymm_dummies, CLUSTER_VAR)
    add_result(format_iv_result(res, dat, 'dcn', 'esp', 'robust/sample/trimmed_5pct',
                                'robustness/sample_restrictions.md',
                                'Trimmed top/bottom 5% of dcn',
                                CONTROL_VARS, 'yymm', CLUSTER_VAR))

    # 8g. Winsorize 1%
    df_wins = df.copy()
    df_wins['dcn'] = df_wins['dcn'].clip(lower=q01, upper=q99)
    res, dat = run_iv_2sls(df_wins, 'dcn', TREATMENT_ENDOGENOUS, TREATMENT_INSTRUMENTS,
                           CONTROL_VARS, yymm_dummies, CLUSTER_VAR)
    add_result(format_iv_result(res, dat, 'dcn', 'esp', 'robust/form/y_winsorized',
                                'robustness/functional_form.md',
                                'Winsorized dcn at 1st/99th percentile',
                                CONTROL_VARS, 'yymm', CLUSTER_VAR))

    # =========================================================================
    # 9. HETEROGENEITY: BY REPORTED SPENDING BEHAVIOR
    # =========================================================================
    print("\n--- HETEROGENEITY: REPORTED SPENDING ---")

    # 9a. Spend vs non-spend (indicator interacted) -- replicates Table 3 Panel A
    # This is the paper's main contribution: comparing revealed vs reported MPC
    # Use areg with spend/nspend interactions
    spend_regs = ['spendiesp', 'spendliesp', 'spendl2iesp',
                  'nspendiesp', 'nspendliesp', 'nspendl2iesp']
    res, dat = run_areg(df, 'dcn', spend_regs, CONTROL_VARS, yymm_dummies, CLUSTER_VAR)
    if res is not None:
        add_result(format_ols_result(res, dat, 'dcn', 'spendiesp',
                                     'robust/heterogeneity/spend_vs_nspend',
                                     'robustness/heterogeneity.md',
                                     'Spend vs non-spend, indicator treatment',
                                     CONTROL_VARS, 'yymm', CLUSTER_VAR))
        add_result(format_ols_result(res, dat, 'dcn', 'nspendiesp',
                                     'robust/heterogeneity/nspend_indicator',
                                     'robustness/heterogeneity.md',
                                     'Non-spend group, indicator treatment',
                                     CONTROL_VARS, 'yymm', CLUSTER_VAR))

    # 9b. Three-way split: spend, save, pay debt
    howused_regs = ['spendiesp', 'spendliesp', 'spendl2iesp',
                    'saveiesp', 'saveliesp', 'savel2iesp',
                    'paydiesp', 'paydliesp', 'paydl2iesp']
    res, dat = run_areg(df, 'dcn', howused_regs, CONTROL_VARS, yymm_dummies, CLUSTER_VAR)
    if res is not None:
        for grp, var in [('spend', 'spendiesp'), ('save', 'saveiesp'), ('paydebt', 'paydiesp')]:
            add_result(format_ols_result(res, dat, 'dcn', var,
                                         f'robust/heterogeneity/howused_{grp}',
                                         'robustness/heterogeneity.md',
                                         f'How-used group: {grp}',
                                         CONTROL_VARS, 'yymm', CLUSTER_VAR))

    # 9c. IV versions: spend vs nspend (MPC)
    spend_endog = ['spendesp', 'spendlesp', 'spendl2esp',
                   'nspendesp', 'nspendlesp', 'nspendl2esp']
    spend_instr = ['spendiesp', 'spendliesp', 'spendl2iesp',
                   'nspendiesp', 'nspendliesp', 'nspendl2iesp']
    res, dat = run_iv_2sls(df, 'dcn', spend_endog, spend_instr,
                           CONTROL_VARS, yymm_dummies, CLUSTER_VAR)
    if res is not None:
        add_result(format_iv_result(res, dat, 'dcn', 'spendesp',
                                    'robust/heterogeneity/spend_mpc',
                                    'robustness/heterogeneity.md',
                                    'MPC for spend group (IV)',
                                    CONTROL_VARS, 'yymm', CLUSTER_VAR,
                                    endogenous=spend_endog, instruments=spend_instr))
        add_result(format_iv_result(res, dat, 'dcn', 'nspendesp',
                                    'robust/heterogeneity/nspend_mpc',
                                    'robustness/heterogeneity.md',
                                    'MPC for non-spend group (IV)',
                                    CONTROL_VARS, 'yymm', CLUSTER_VAR,
                                    endogenous=spend_endog, instruments=spend_instr))

    # 9d. IV with three-way split: spend, save, paydebt (MPC)
    howused_endog = ['spendesp', 'spendlesp', 'spendl2esp',
                     'saveesp', 'savelesp', 'savel2esp',
                     'paydesp', 'paydlesp', 'paydl2esp']
    howused_instr = ['spendiesp', 'spendliesp', 'spendl2iesp',
                     'saveiesp', 'saveliesp', 'savel2iesp',
                     'paydiesp', 'paydliesp', 'paydl2iesp']
    res, dat = run_iv_2sls(df, 'dcn', howused_endog, howused_instr,
                           CONTROL_VARS, yymm_dummies, CLUSTER_VAR)
    if res is not None:
        for grp, var in [('spend', 'spendesp'), ('save', 'saveesp'), ('paydebt', 'paydesp')]:
            add_result(format_iv_result(res, dat, 'dcn', var,
                                        f'robust/heterogeneity/howused_{grp}_mpc',
                                        'robustness/heterogeneity.md',
                                        f'MPC for how-used group: {grp}',
                                        CONTROL_VARS, 'yymm', CLUSTER_VAR,
                                        endogenous=howused_endog, instruments=howused_instr))

    # =========================================================================
    # 10. TOTAL CONSUMPTION PARALLELS (mirror all key specs for dctsmed)
    # =========================================================================
    print("\n--- TOTAL CONSUMPTION PARALLELS ---")

    # Already have baseline_total_consumption above
    # LIML for total
    res, dat = run_iv_2sls(df, 'dctsmed', TREATMENT_ENDOGENOUS, TREATMENT_INSTRUMENTS,
                           CONTROL_VARS, yymm_dummies, CLUSTER_VAR, method='liml')
    add_result(format_iv_result(res, dat, 'dctsmed', 'esp', 'iv/method/liml_total',
                                'methods/instrumental_variables.md#estimation-method',
                                'Total consumption, LIML',
                                CONTROL_VARS, 'yymm', CLUSTER_VAR, method='liml'))

    # OLS indicator for total
    res, dat = run_areg(df, 'dctsmed', TREATMENT_INSTRUMENTS, CONTROL_VARS, yymm_dummies, CLUSTER_VAR)
    add_result(format_ols_result(res, dat, 'dctsmed', 'iesp', 'iv/method/ols_indicator_total',
                                 'methods/instrumental_variables.md#estimation-method',
                                 'Total consumption, OLS indicator',
                                 CONTROL_VARS, 'yymm', CLUSTER_VAR))

    # No controls for total
    res, dat = run_iv_2sls(df, 'dctsmed', TREATMENT_ENDOGENOUS, TREATMENT_INSTRUMENTS,
                           [], yymm_dummies, CLUSTER_VAR)
    add_result(format_iv_result(res, dat, 'dctsmed', 'esp', 'iv/controls/none_total',
                                'methods/instrumental_variables.md#control-sets',
                                'Total consumption, no controls',
                                [], 'yymm', CLUSTER_VAR))

    # Heterogeneity: spend vs nspend for total
    spend_regs_tsmed = ['spendiesp', 'spendliesp', 'spendl2iesp',
                        'nspendiesp', 'nspendliesp', 'nspendl2iesp']
    res, dat = run_areg(df, 'dctsmed', spend_regs_tsmed, CONTROL_VARS, yymm_dummies, CLUSTER_VAR)
    if res is not None:
        add_result(format_ols_result(res, dat, 'dctsmed', 'spendiesp',
                                     'robust/heterogeneity/spend_total',
                                     'robustness/heterogeneity.md',
                                     'Spend group, total consumption, indicator',
                                     CONTROL_VARS, 'yymm', CLUSTER_VAR))
        add_result(format_ols_result(res, dat, 'dctsmed', 'nspendiesp',
                                     'robust/heterogeneity/nspend_total',
                                     'robustness/heterogeneity.md',
                                     'Non-spend group, total consumption, indicator',
                                     CONTROL_VARS, 'yymm', CLUSTER_VAR))

    # =========================================================================
    # 11. INSTRUMENT SET VARIATIONS
    # =========================================================================
    print("\n--- INSTRUMENT SET VARIATIONS ---")

    # 11a. Just contemporaneous (no lags)
    res, dat = run_iv_2sls(df, 'dcn', ['esp'], ['iesp'],
                           CONTROL_VARS, yymm_dummies, CLUSTER_VAR)
    add_result(format_iv_result(res, dat, 'dcn', 'esp', 'iv/instruments/single',
                                'methods/instrumental_variables.md#instrument-sets',
                                'Single instrument, contemporaneous only',
                                CONTROL_VARS, 'yymm', CLUSTER_VAR,
                                endogenous=['esp'], instruments=['iesp']))

    # 11b. Contemporaneous + 1 lag
    res, dat = run_iv_2sls(df, 'dcn', ['esp', 'lesp'], ['iesp', 'liesp'],
                           CONTROL_VARS, yymm_dummies, CLUSTER_VAR)
    add_result(format_iv_result(res, dat, 'dcn', 'esp', 'iv/instruments/two_lags',
                                'methods/instrumental_variables.md#instrument-sets',
                                'Contemporaneous + 1 lag',
                                CONTROL_VARS, 'yymm', CLUSTER_VAR,
                                endogenous=['esp', 'lesp'], instruments=['iesp', 'liesp']))

    # 11c. Check vs EFT instrument (if available)
    for inst_type, inst_vars in [('check', ['iespchck', 'liespchck']),
                                  ('eft', ['iespeft', 'liespeft'])]:
        if all(v in df.columns for v in inst_vars):
            # Use just contemporaneous for simplicity
            try:
                res, dat = run_iv_2sls(df, 'dcn', ['esp'], [inst_vars[0]],
                                       CONTROL_VARS, yymm_dummies, CLUSTER_VAR)
                add_result(format_iv_result(res, dat, 'dcn', 'esp',
                                            f'iv/instruments/{inst_type}',
                                            'methods/instrumental_variables.md#instrument-sets',
                                            f'Instrument: {inst_type} ESP indicator',
                                            CONTROL_VARS, 'yymm', CLUSTER_VAR,
                                            endogenous=['esp'], instruments=[inst_vars[0]]))
            except:
                pass

    # =========================================================================
    # 12. CONTROL PROGRESSION (build-up)
    # =========================================================================
    print("\n--- CONTROL PROGRESSION ---")

    # Already have no controls (iv/controls/none)
    # Add one at a time
    for i, ctrl in enumerate(CONTROL_VARS):
        ctrl_set = CONTROL_VARS[:i+1]
        res, dat = run_iv_2sls(df, 'dcn', TREATMENT_ENDOGENOUS, TREATMENT_INSTRUMENTS,
                               ctrl_set, yymm_dummies, CLUSTER_VAR)
        add_result(format_iv_result(res, dat, 'dcn', 'esp',
                                    f'robust/progression/add_{ctrl}',
                                    'robustness/control_progression.md',
                                    f'Controls: {", ".join(ctrl_set)}',
                                    ctrl_set, 'yymm', CLUSTER_VAR))

    # =========================================================================
    # 13. MODEL SPECIFICATION VARIATIONS
    # =========================================================================
    print("\n--- MODEL SPECIFICATION ---")

    # 13a. Unweighted IV
    data_uw = df.dropna(subset=['dcn'] + TREATMENT_ENDOGENOUS + TREATMENT_INSTRUMENTS +
                        CONTROL_VARS + [CLUSTER_VAR, 'weight']).copy()
    data_uw['const'] = 1.0
    exog_cols = ['const'] + CONTROL_VARS + yymm_dummies
    try:
        model_uw = IV2SLS(data_uw['dcn'], data_uw[exog_cols],
                          data_uw[TREATMENT_ENDOGENOUS], data_uw[TREATMENT_INSTRUMENTS])
        res_uw = model_uw.fit(cov_type='clustered', clusters=data_uw[CLUSTER_VAR])
        add_result(format_iv_result(res_uw, data_uw, 'dcn', 'esp',
                                    'robust/model/unweighted',
                                    'robustness/model_specification.md',
                                    'Unweighted IV/2SLS',
                                    CONTROL_VARS, 'yymm', CLUSTER_VAR))
    except Exception as e:
        print(f"  Unweighted error: {e}")

    # 13b. Unweighted IV for total consumption
    try:
        model_uw_t = IV2SLS(data_uw['dctsmed'], data_uw[exog_cols],
                            data_uw[TREATMENT_ENDOGENOUS], data_uw[TREATMENT_INSTRUMENTS])
        res_uw_t = model_uw_t.fit(cov_type='clustered', clusters=data_uw[CLUSTER_VAR])
        add_result(format_iv_result(res_uw_t, data_uw, 'dctsmed', 'esp',
                                    'robust/model/unweighted_total',
                                    'robustness/model_specification.md',
                                    'Unweighted IV/2SLS, total consumption',
                                    CONTROL_VARS, 'yymm', CLUSTER_VAR))
    except Exception as e:
        print(f"  Unweighted total error: {e}")

    # =========================================================================
    # 14. PLACEBO / FALSIFICATION TESTS
    # =========================================================================
    print("\n--- PLACEBO TESTS ---")

    # 14a. Check if ESP indicator predicts pre-determined variables
    for predet in ['age', 'income', 'liqassii']:
        if predet in df.columns:
            df_predet = df[df[predet].notna()].copy()
            ctrl_no_predet = [c for c in CONTROL_VARS if c != predet]
            res, dat = run_areg(df_predet, predet, ['iesp'], ctrl_no_predet, yymm_dummies, CLUSTER_VAR)
            if res is not None:
                add_result(format_ols_result(res, dat, predet, 'iesp',
                                             f'iv/balance/{predet}',
                                             'methods/instrumental_variables.md#placebos-and-falsification',
                                             f'Balance test: {predet} on ESP indicator',
                                             ctrl_no_predet, 'yymm', CLUSTER_VAR))

    # =========================================================================
    # 15. ADDITIONAL SUBCATEGORY OUTCOMES
    # =========================================================================
    print("\n--- SUBCATEGORY OUTCOMES ---")

    subcat_vars = ['dcfdhome', 'dcfdaway', 'dcalcbev', 'dcappar', 'dchealth',
                   'dcread', 'dcother', 'dchousdur', 'dctransdur', 'dctentert',
                   'dceduca', 'dcveh', 'dcgasptr', 'dcutilhop', 'dcpersmisc', 'dctobac']
    for subcat in subcat_vars:
        if subcat in df.columns:
            res, dat = run_areg(df, subcat, TREATMENT_INSTRUMENTS, CONTROL_VARS, yymm_dummies, CLUSTER_VAR)
            if res is not None:
                add_result(format_ols_result(res, dat, subcat, 'iesp',
                                             f'custom/subcat_{subcat}',
                                             'custom',
                                             f'Subcategory: {subcat}, indicator',
                                             CONTROL_VARS, 'yymm', CLUSTER_VAR))

    # =========================================================================
    # 16. QUANTILE REGRESSIONS
    # =========================================================================
    print("\n--- QUANTILE REGRESSIONS ---")

    from statsmodels.regression.quantile_regression import QuantReg

    for q in [0.25, 0.50, 0.75]:
        data_q = df.dropna(subset=['dcn'] + TREATMENT_INSTRUMENTS + CONTROL_VARS + ['weight']).copy()
        data_q['const'] = 1.0
        X_cols = ['const'] + TREATMENT_INSTRUMENTS + CONTROL_VARS + yymm_dummies
        try:
            model_q = QuantReg(data_q['dcn'], data_q[X_cols])
            res_q = model_q.fit(q=q)
            coef_q = float(res_q.params['iesp'])
            se_q = float(res_q.bse['iesp'])
            pval_q = float(res_q.pvalues['iesp'])
            tstat_q = coef_q / se_q if se_q > 0 else np.nan

            coef_vector = {
                "treatment": {"var": "iesp", "coef": coef_q, "se": se_q, "pval": pval_q,
                              "ci_lower": coef_q - 1.96 * se_q, "ci_upper": coef_q + 1.96 * se_q},
                "controls": [], "fixed_effects_absorbed": ["yymm"],
                "diagnostics": {"method": f"quantile_{int(q*100)}", "quantile": q}
            }
            r = {
                'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
                'spec_id': f'robust/form/quantile_{int(q*100)}',
                'spec_tree_path': 'robustness/functional_form.md',
                'outcome_var': 'dcn', 'treatment_var': 'iesp',
                'coefficient': coef_q, 'std_error': se_q, 't_stat': tstat_q,
                'p_value': pval_q, 'ci_lower': coef_q - 1.96 * se_q,
                'ci_upper': coef_q + 1.96 * se_q, 'n_obs': int(len(data_q)),
                'r_squared': float(res_q.prsquared),
                'coefficient_vector_json': json.dumps(coef_vector),
                'sample_desc': f'Quantile {int(q*100)}th, indicator treatment',
                'fixed_effects': 'yymm', 'controls_desc': ", ".join(CONTROL_VARS),
                'cluster_var': 'None (quantile)', 'model_type': 'QUANTILE',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            }
            add_result(r)
        except Exception as e:
            print(f"  Quantile {int(q*100)} error: {e}")

    # =========================================================================
    # 17. ASINH TRANSFORMATION
    # =========================================================================
    print("\n--- ASINH TRANSFORMATION ---")

    # asinh(dcn + shift) to handle negative values
    df_asinh = df.copy()
    # Shift so minimum is positive, then asinh
    shift = abs(df_asinh['dcn'].min()) + 1
    df_asinh['dcn_asinh'] = np.arcsinh(df_asinh['dcn'])
    res, dat = run_areg(df_asinh, 'dcn_asinh', TREATMENT_INSTRUMENTS, CONTROL_VARS, yymm_dummies, CLUSTER_VAR)
    if res is not None:
        add_result(format_ols_result(res, dat, 'dcn_asinh', 'iesp', 'robust/form/y_asinh',
                                     'robustness/functional_form.md',
                                     'Arcsinh nondurable consumption, indicator',
                                     CONTROL_VARS, 'yymm', CLUSTER_VAR))

    # =========================================================================
    # 18. CHECK/EFT DELIVERY METHOD SUBSAMPLES
    # =========================================================================
    print("\n--- DELIVERY METHOD SUBSAMPLES ---")

    # Check recipients only
    df_chk = df[df['hhchckever'] == 1].copy()
    if len(df_chk) > 50:
        res, dat = run_iv_2sls(df_chk, 'dcn', TREATMENT_ENDOGENOUS, TREATMENT_INSTRUMENTS,
                               CONTROL_VARS, yymm_dummies, CLUSTER_VAR)
        add_result(format_iv_result(res, dat, 'dcn', 'esp', 'robust/sample/check_only',
                                    'robustness/sample_restrictions.md',
                                    'Check recipients only',
                                    CONTROL_VARS, 'yymm', CLUSTER_VAR))

    # EFT recipients only
    df_eft = df[df['hheftever'] == 1].copy()
    if len(df_eft) > 50:
        res, dat = run_iv_2sls(df_eft, 'dcn', TREATMENT_ENDOGENOUS, TREATMENT_INSTRUMENTS,
                               CONTROL_VARS, yymm_dummies, CLUSTER_VAR)
        add_result(format_iv_result(res, dat, 'dcn', 'esp', 'robust/sample/eft_only',
                                    'robustness/sample_restrictions.md',
                                    'EFT recipients only',
                                    CONTROL_VARS, 'yymm', CLUSTER_VAR))

    print(f"\n=== Total specifications run: {spec_count} ===")
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print(f"Specification Search: {PAPER_ID}")
    print(f"Paper: {PAPER_TITLE}")
    print(f"Method: {METHOD_TYPE}")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df, yymm_dummies = load_data()
    print(f"Loaded {len(df)} observations, {df['newid'].nunique()} unique households")

    # Run specification search
    print("\nRunning specifications...")
    results = run_all_specifications(df, yymm_dummies)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(results)} specifications to {OUTPUT_FILE}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total specifications: {len(results)}")
    if len(results) > 0:
        baseline = results_df[results_df['spec_id'] == 'baseline']
        if len(baseline) > 0:
            bl_coef = baseline.iloc[0]['coefficient']
            bl_pval = baseline.iloc[0]['p_value']
            print(f"Baseline coefficient (esp on dcn): {bl_coef:.4f}")
            print(f"Baseline p-value: {bl_pval:.4f}")

        print(f"\nCoefficient range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")
        print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
        print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} / {len(results)}")
        print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} / {len(results)}")
        print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} / {len(results)}")

    print("\nDone!")
    return results_df


if __name__ == "__main__":
    results_df = main()
