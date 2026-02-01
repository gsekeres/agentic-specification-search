"""
Specification Search Analysis for Paper 194841-V1

"Rational Inattention and the Business Cycle Effects of Productivity and News Shocks"
by Bartosz Mackowiak and Mirko Wiederholt

This paper is primarily theoretical (DSGE with rational inattention), but the key empirical
analysis is the Coibion-Gorodnichenko (CG) regression that tests for information rigidity
in the Survey of Professional Forecasters data.

CG Regression: forecast_error = alpha + beta * forecast_revision + epsilon

Where:
- forecast_error = log(outcome) - log(forecast)
- forecast_revision = log(forecast) - log(previous_forecast)

A positive beta indicates information rigidity (sticky expectations).

Method Classification: cross_sectional_ols
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.quantile_regression import QuantReg
from scipy import stats
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration
PAPER_ID = "194841-V1"
PAPER_TITLE = "Rational Inattention and the Business Cycle Effects of Productivity and News Shocks"
JOURNAL = "AER"  # American Economic Review
DATA_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/194841-V1"
OUTPUT_DIR = DATA_DIR

# Method classification
METHOD_CODE = "cross_sectional_ols"
METHOD_TREE_PATH = "specification_tree/methods/cross_sectional_ols.md"


def load_data():
    """Load and prepare SPF data for CG regression."""
    spf = pd.read_excel(os.path.join(DATA_DIR, 'SPF_data_replication.xlsx'), header=1)

    # Get year and quarter columns
    year_qtr = spf.iloc[1:, 0:2].copy()
    year_qtr.columns = ['year', 'quarter']

    # Get the final data columns (y(t+3|t), y(t+3|t-1), y(t+3))
    final_data = spf.iloc[1:, 19:22].copy()
    final_data.columns = ['forecast', 'prev_forecast', 'outcome']

    # Combine
    df = pd.concat([year_qtr, final_data], axis=1)
    df = df.dropna(subset=['forecast', 'prev_forecast', 'outcome'])
    df = df.astype({
        'year': int, 'quarter': int,
        'forecast': float, 'prev_forecast': float, 'outcome': float
    })
    df = df.reset_index(drop=True)

    # Create log variables as in the original MATLAB code
    df['log_forecast'] = np.log(df['forecast'])
    df['log_prev_forecast'] = np.log(df['prev_forecast'])
    df['log_outcome'] = np.log(df['outcome'])

    # Create forecast error and forecast revision
    df['forecast_error'] = df['log_outcome'] - df['log_forecast']
    df['forecast_revision'] = df['log_forecast'] - df['log_prev_forecast']

    # Create time variable
    df['time_index'] = range(len(df))
    df['year_quarter'] = df['year'] + (df['quarter'] - 1) / 4

    return df


def remove_outliers(df, fe_percentile=99, fr_percentile=99):
    """Remove outliers as in the original MATLAB code."""
    df_clean = df.copy()

    fe_threshold = np.percentile(df['forecast_error'], fe_percentile)
    fr_threshold = np.percentile(df['forecast_revision'], fr_percentile)

    # Set outliers to NaN (like MATLAB code)
    df_clean.loc[df_clean['forecast_error'] > fe_threshold, 'forecast_error_clean'] = np.nan
    df_clean.loc[df_clean['forecast_error'] <= fe_threshold, 'forecast_error_clean'] = df_clean['forecast_error']

    df_clean.loc[df_clean['forecast_revision'] > fr_threshold, 'forecast_revision_clean'] = np.nan
    df_clean.loc[df_clean['forecast_revision'] <= fr_threshold, 'forecast_revision_clean'] = df_clean['forecast_revision']

    return df_clean


def run_ols_regression(y, X, robust=True, hc_type='HC1'):
    """Run OLS regression with various standard error options."""
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const, missing='drop')

    if robust:
        results = model.fit(cov_type=hc_type)
    else:
        results = model.fit()

    return results


def extract_results(results, spec_id, spec_tree_path, outcome_var, treatment_var,
                    sample_desc, controls_desc, cluster_var, model_type, n_obs_full=None):
    """Extract standardized results from statsmodels regression."""

    # Get treatment coefficient (assuming it's the first non-constant variable)
    treatment_idx = 1 if 'const' in results.params.index else 0
    treatment_name = results.params.index[treatment_idx]

    coef = results.params[treatment_name]
    se = results.bse[treatment_name]
    t_stat = results.tvalues[treatment_name]
    p_val = results.pvalues[treatment_name]

    # Confidence interval
    ci = results.conf_int().loc[treatment_name]
    ci_lower = ci[0]
    ci_upper = ci[1]

    # Build coefficient vector JSON
    coef_vector = {
        "treatment": {
            "var": treatment_name,
            "coef": float(coef),
            "se": float(se),
            "pval": float(p_val),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper)
        },
        "intercept": {
            "var": "const",
            "coef": float(results.params.get('const', np.nan)),
            "se": float(results.bse.get('const', np.nan)),
            "pval": float(results.pvalues.get('const', np.nan))
        },
        "controls": [],
        "diagnostics": {
            "r_squared": float(results.rsquared),
            "adj_r_squared": float(results.rsquared_adj),
            "f_stat": float(results.fvalue) if hasattr(results, 'fvalue') else None,
            "f_pval": float(results.f_pvalue) if hasattr(results, 'f_pvalue') else None,
            "aic": float(results.aic) if hasattr(results, 'aic') else None,
            "bic": float(results.bic) if hasattr(results, 'bic') else None
        },
        "n_obs": int(results.nobs)
    }

    # Add any control variables
    for var in results.params.index:
        if var not in ['const', treatment_name]:
            coef_vector["controls"].append({
                "var": var,
                "coef": float(results.params[var]),
                "se": float(results.bse[var]),
                "pval": float(results.pvalues[var])
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
        't_stat': t_stat,
        'p_value': p_val,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_obs': int(results.nobs),
        'r_squared': results.rsquared,
        'coefficient_vector_json': json.dumps(coef_vector),
        'sample_desc': sample_desc,
        'fixed_effects': 'None',
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    }


def run_specification_search():
    """Run complete specification search."""

    results_list = []

    # Load data
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} observations")

    # Apply outlier removal as in original paper
    df = remove_outliers(df)

    # Use clean variables for most analyses
    y_full = df['forecast_error_clean']
    X_full = df['forecast_revision_clean']

    # ==========================================================================
    # BASELINE SPECIFICATION (exact replication)
    # ==========================================================================
    print("\n--- Running Baseline Specification ---")

    # Baseline: OLS with classical SE (as in fitlm MATLAB)
    mask = ~(y_full.isna() | X_full.isna())
    y = y_full[mask]
    X = X_full[mask]

    results = run_ols_regression(y, X, robust=False)
    print(f"Baseline coefficient: {results.params.iloc[1]:.6f}")
    print(f"Baseline SE: {results.bse.iloc[1]:.6f}")
    print(f"N obs: {results.nobs}")

    results_list.append(extract_results(
        results,
        spec_id='baseline',
        spec_tree_path='methods/cross_sectional_ols.md#baseline',
        outcome_var='forecast_error',
        treatment_var='forecast_revision',
        sample_desc='Full sample (1969-2009), 99th percentile outliers removed',
        controls_desc='None (bivariate regression)',
        cluster_var='None',
        model_type='OLS'
    ))

    # ==========================================================================
    # STANDARD ERROR VARIATIONS
    # ==========================================================================
    print("\n--- Running Standard Error Variations ---")

    # OLS with Robust SE (HC1)
    results_hc1 = run_ols_regression(y, X, robust=True, hc_type='HC1')
    results_list.append(extract_results(
        results_hc1,
        spec_id='ols/se/robust',
        spec_tree_path='methods/cross_sectional_ols.md#standard-errors',
        outcome_var='forecast_error',
        treatment_var='forecast_revision',
        sample_desc='Full sample with 99th percentile outliers removed',
        controls_desc='None (bivariate regression)',
        cluster_var='HC1 robust',
        model_type='OLS'
    ))

    # OLS with HC2
    results_hc2 = run_ols_regression(y, X, robust=True, hc_type='HC2')
    results_list.append(extract_results(
        results_hc2,
        spec_id='ols/se/hc2',
        spec_tree_path='methods/cross_sectional_ols.md#standard-errors',
        outcome_var='forecast_error',
        treatment_var='forecast_revision',
        sample_desc='Full sample with 99th percentile outliers removed',
        controls_desc='None (bivariate regression)',
        cluster_var='HC2 robust',
        model_type='OLS'
    ))

    # OLS with HC3
    results_hc3 = run_ols_regression(y, X, robust=True, hc_type='HC3')
    results_list.append(extract_results(
        results_hc3,
        spec_id='ols/se/hc3',
        spec_tree_path='methods/cross_sectional_ols.md#standard-errors',
        outcome_var='forecast_error',
        treatment_var='forecast_revision',
        sample_desc='Full sample with 99th percentile outliers removed',
        controls_desc='None (bivariate regression)',
        cluster_var='HC3 robust',
        model_type='OLS'
    ))

    # Newey-West HAC SE (account for autocorrelation in time series)
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const)
    results_nw = model.fit(cov_type='HAC', cov_kwds={'maxlags': 4})
    results_list.append(extract_results(
        results_nw,
        spec_id='robust/se/newey_west',
        spec_tree_path='robustness/clustering_variations.md#alternative-se-methods',
        outcome_var='forecast_error',
        treatment_var='forecast_revision',
        sample_desc='Full sample with 99th percentile outliers removed',
        controls_desc='None (bivariate regression)',
        cluster_var='Newey-West HAC (4 lags)',
        model_type='OLS'
    ))

    # ==========================================================================
    # SAMPLE RESTRICTIONS (from paper Section 5.3)
    # ==========================================================================
    print("\n--- Running Sample Restrictions ---")

    # Subsample 1 (first 80 observations) - as in paper
    df_sub1 = df.iloc[:80]
    y_sub1 = df_sub1['forecast_error_clean']
    X_sub1 = df_sub1['forecast_revision_clean']
    mask1 = ~(y_sub1.isna() | X_sub1.isna())

    results_sub1 = run_ols_regression(y_sub1[mask1], X_sub1[mask1], robust=False)
    results_list.append(extract_results(
        results_sub1,
        spec_id='robust/sample/early_period',
        spec_tree_path='robustness/sample_restrictions.md#time-based-restrictions',
        outcome_var='forecast_error',
        treatment_var='forecast_revision',
        sample_desc='First 80 observations (approx 1969-1988)',
        controls_desc='None (bivariate regression)',
        cluster_var='None',
        model_type='OLS'
    ))

    # Subsample 2 (observations 81+) - as in paper
    df_sub2 = df.iloc[80:]
    y_sub2 = df_sub2['forecast_error_clean']
    X_sub2 = df_sub2['forecast_revision_clean']
    mask2 = ~(y_sub2.isna() | X_sub2.isna())

    results_sub2 = run_ols_regression(y_sub2[mask2], X_sub2[mask2], robust=False)
    results_list.append(extract_results(
        results_sub2,
        spec_id='robust/sample/late_period',
        spec_tree_path='robustness/sample_restrictions.md#time-based-restrictions',
        outcome_var='forecast_error',
        treatment_var='forecast_revision',
        sample_desc='Observations 81+ (approx 1989-2009)',
        controls_desc='None (bivariate regression)',
        cluster_var='None',
        model_type='OLS'
    ))

    # Pre-Great Recession (up to 2007)
    df_pre_crisis = df[df['year'] <= 2007]
    y_pre = df_pre_crisis['forecast_error_clean']
    X_pre = df_pre_crisis['forecast_revision_clean']
    mask_pre = ~(y_pre.isna() | X_pre.isna())

    results_pre = run_ols_regression(y_pre[mask_pre], X_pre[mask_pre], robust=False)
    results_list.append(extract_results(
        results_pre,
        spec_id='robust/sample/pre_crisis',
        spec_tree_path='robustness/sample_restrictions.md#time-based-restrictions',
        outcome_var='forecast_error',
        treatment_var='forecast_revision',
        sample_desc='Pre-Great Recession (1969-2007)',
        controls_desc='None (bivariate regression)',
        cluster_var='None',
        model_type='OLS'
    ))

    # Post-2007 (Great Recession and after)
    df_post_crisis = df[df['year'] > 2007]
    y_post = df_post_crisis['forecast_error_clean']
    X_post = df_post_crisis['forecast_revision_clean']
    mask_post = ~(y_post.isna() | X_post.isna())

    if mask_post.sum() > 5:  # Need minimum observations
        results_post = run_ols_regression(y_post[mask_post], X_post[mask_post], robust=False)
        results_list.append(extract_results(
            results_post,
            spec_id='robust/sample/post_crisis',
            spec_tree_path='robustness/sample_restrictions.md#time-based-restrictions',
            outcome_var='forecast_error',
            treatment_var='forecast_revision',
            sample_desc='Post-Great Recession (2008-2009)',
            controls_desc='None (bivariate regression)',
            cluster_var='None',
            model_type='OLS'
        ))

    # ==========================================================================
    # OUTLIER HANDLING VARIATIONS
    # ==========================================================================
    print("\n--- Running Outlier Handling Variations ---")

    # No outlier removal
    y_raw = df['forecast_error']
    X_raw = df['forecast_revision']

    results_raw = run_ols_regression(y_raw, X_raw, robust=False)
    results_list.append(extract_results(
        results_raw,
        spec_id='robust/sample/no_outlier_removal',
        spec_tree_path='robustness/sample_restrictions.md#outlier-handling',
        outcome_var='forecast_error',
        treatment_var='forecast_revision',
        sample_desc='Full sample, no outlier removal',
        controls_desc='None (bivariate regression)',
        cluster_var='None',
        model_type='OLS'
    ))

    # Trim 1% (top and bottom)
    df_trim1 = df.copy()
    fe_1 = np.percentile(df['forecast_error'], 1)
    fe_99 = np.percentile(df['forecast_error'], 99)
    fr_1 = np.percentile(df['forecast_revision'], 1)
    fr_99 = np.percentile(df['forecast_revision'], 99)

    mask_trim1 = (
        (df['forecast_error'] > fe_1) & (df['forecast_error'] < fe_99) &
        (df['forecast_revision'] > fr_1) & (df['forecast_revision'] < fr_99)
    )

    results_trim1 = run_ols_regression(
        df.loc[mask_trim1, 'forecast_error'],
        df.loc[mask_trim1, 'forecast_revision'],
        robust=False
    )
    results_list.append(extract_results(
        results_trim1,
        spec_id='robust/sample/trim_1pct',
        spec_tree_path='robustness/sample_restrictions.md#outlier-handling',
        outcome_var='forecast_error',
        treatment_var='forecast_revision',
        sample_desc='Full sample, trimmed at 1%/99% on both variables',
        controls_desc='None (bivariate regression)',
        cluster_var='None',
        model_type='OLS'
    ))

    # Trim 5%
    fe_5 = np.percentile(df['forecast_error'], 5)
    fe_95 = np.percentile(df['forecast_error'], 95)
    fr_5 = np.percentile(df['forecast_revision'], 5)
    fr_95 = np.percentile(df['forecast_revision'], 95)

    mask_trim5 = (
        (df['forecast_error'] > fe_5) & (df['forecast_error'] < fe_95) &
        (df['forecast_revision'] > fr_5) & (df['forecast_revision'] < fr_95)
    )

    results_trim5 = run_ols_regression(
        df.loc[mask_trim5, 'forecast_error'],
        df.loc[mask_trim5, 'forecast_revision'],
        robust=False
    )
    results_list.append(extract_results(
        results_trim5,
        spec_id='robust/sample/trim_5pct',
        spec_tree_path='robustness/sample_restrictions.md#outlier-handling',
        outcome_var='forecast_error',
        treatment_var='forecast_revision',
        sample_desc='Full sample, trimmed at 5%/95% on both variables',
        controls_desc='None (bivariate regression)',
        cluster_var='None',
        model_type='OLS'
    ))

    # Winsorize 1%
    df_winsor1 = df.copy()
    df_winsor1['fe_winsor'] = df['forecast_error'].clip(lower=fe_1, upper=fe_99)
    df_winsor1['fr_winsor'] = df['forecast_revision'].clip(lower=fr_1, upper=fr_99)

    results_winsor1 = run_ols_regression(df_winsor1['fe_winsor'], df_winsor1['fr_winsor'], robust=False)
    results_list.append(extract_results(
        results_winsor1,
        spec_id='robust/sample/winsor_1pct',
        spec_tree_path='robustness/sample_restrictions.md#outlier-handling',
        outcome_var='forecast_error',
        treatment_var='forecast_revision',
        sample_desc='Full sample, winsorized at 1%/99% on both variables',
        controls_desc='None (bivariate regression)',
        cluster_var='None',
        model_type='OLS'
    ))

    # Winsorize 5%
    df_winsor5 = df.copy()
    df_winsor5['fe_winsor'] = df['forecast_error'].clip(lower=fe_5, upper=fe_95)
    df_winsor5['fr_winsor'] = df['forecast_revision'].clip(lower=fr_5, upper=fr_95)

    results_winsor5 = run_ols_regression(df_winsor5['fe_winsor'], df_winsor5['fr_winsor'], robust=False)
    results_list.append(extract_results(
        results_winsor5,
        spec_id='robust/sample/winsor_5pct',
        spec_tree_path='robustness/sample_restrictions.md#outlier-handling',
        outcome_var='forecast_error',
        treatment_var='forecast_revision',
        sample_desc='Full sample, winsorized at 5%/95% on both variables',
        controls_desc='None (bivariate regression)',
        cluster_var='None',
        model_type='OLS'
    ))

    # ==========================================================================
    # FUNCTIONAL FORM VARIATIONS
    # ==========================================================================
    print("\n--- Running Functional Form Variations ---")

    # Quadratic specification
    df['forecast_revision_sq'] = df['forecast_revision_clean'] ** 2
    mask_quad = ~(y_full.isna() | X_full.isna())

    X_quad = df.loc[mask_quad, ['forecast_revision_clean', 'forecast_revision_sq']]
    X_quad_const = sm.add_constant(X_quad)
    model_quad = sm.OLS(y_full[mask_quad], X_quad_const)
    results_quad = model_quad.fit()

    results_list.append(extract_results(
        results_quad,
        spec_id='robust/form/quadratic',
        spec_tree_path='robustness/functional_form.md#nonlinear-specifications',
        outcome_var='forecast_error',
        treatment_var='forecast_revision',
        sample_desc='Full sample with 99th percentile outliers removed',
        controls_desc='forecast_revision^2',
        cluster_var='None',
        model_type='OLS'
    ))

    # Level-level (using raw levels instead of logs)
    y_levels = df['outcome'] - df['forecast']
    X_levels = df['forecast'] - df['prev_forecast']

    results_levels = run_ols_regression(y_levels, X_levels, robust=False)
    results_list.append(extract_results(
        results_levels,
        spec_id='robust/form/y_level',
        spec_tree_path='robustness/functional_form.md#outcome-variable-transformations',
        outcome_var='forecast_error_levels',
        treatment_var='forecast_revision_levels',
        sample_desc='Full sample, levels instead of logs',
        controls_desc='None (bivariate regression)',
        cluster_var='None',
        model_type='OLS'
    ))

    # Asinh transformation (handles extreme values better)
    df['fe_asinh'] = np.arcsinh(df['forecast_error'])
    df['fr_asinh'] = np.arcsinh(df['forecast_revision'])

    results_asinh = run_ols_regression(df['fe_asinh'], df['fr_asinh'], robust=False)
    results_list.append(extract_results(
        results_asinh,
        spec_id='robust/form/y_asinh',
        spec_tree_path='robustness/functional_form.md#outcome-variable-transformations',
        outcome_var='forecast_error_asinh',
        treatment_var='forecast_revision_asinh',
        sample_desc='Full sample, inverse hyperbolic sine transformation',
        controls_desc='None (bivariate regression)',
        cluster_var='None',
        model_type='OLS'
    ))

    # ==========================================================================
    # QUANTILE REGRESSIONS
    # ==========================================================================
    print("\n--- Running Quantile Regressions ---")

    # Median regression (LAD)
    X_qr = sm.add_constant(X)
    model_q50 = QuantReg(y, X_qr)
    results_q50 = model_q50.fit(q=0.5)

    # For quantile regression, need to manually extract results
    coef_q50 = results_q50.params.iloc[1]
    se_q50 = results_q50.bse.iloc[1]
    t_q50 = coef_q50 / se_q50
    p_q50 = 2 * (1 - stats.norm.cdf(abs(t_q50)))

    results_list.append({
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': 'robust/form/quantile_50',
        'spec_tree_path': 'robustness/functional_form.md#alternative-estimators',
        'outcome_var': 'forecast_error',
        'treatment_var': 'forecast_revision',
        'coefficient': coef_q50,
        'std_error': se_q50,
        't_stat': t_q50,
        'p_value': p_q50,
        'ci_lower': coef_q50 - 1.96 * se_q50,
        'ci_upper': coef_q50 + 1.96 * se_q50,
        'n_obs': int(results_q50.nobs),
        'r_squared': np.nan,  # Not well-defined for quantile regression
        'coefficient_vector_json': json.dumps({
            'treatment': {'var': 'forecast_revision', 'coef': float(coef_q50),
                         'se': float(se_q50), 'pval': float(p_q50)},
            'quantile': 0.5
        }),
        'sample_desc': 'Full sample with 99th percentile outliers removed',
        'fixed_effects': 'None',
        'controls_desc': 'None (bivariate regression)',
        'cluster_var': 'None',
        'model_type': 'Quantile (0.5)',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })

    # 25th percentile regression
    results_q25 = model_q50.fit(q=0.25)
    coef_q25 = results_q25.params.iloc[1]
    se_q25 = results_q25.bse.iloc[1]
    t_q25 = coef_q25 / se_q25
    p_q25 = 2 * (1 - stats.norm.cdf(abs(t_q25)))

    results_list.append({
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': 'robust/form/quantile_25',
        'spec_tree_path': 'robustness/functional_form.md#alternative-estimators',
        'outcome_var': 'forecast_error',
        'treatment_var': 'forecast_revision',
        'coefficient': coef_q25,
        'std_error': se_q25,
        't_stat': t_q25,
        'p_value': p_q25,
        'ci_lower': coef_q25 - 1.96 * se_q25,
        'ci_upper': coef_q25 + 1.96 * se_q25,
        'n_obs': int(results_q25.nobs),
        'r_squared': np.nan,
        'coefficient_vector_json': json.dumps({
            'treatment': {'var': 'forecast_revision', 'coef': float(coef_q25),
                         'se': float(se_q25), 'pval': float(p_q25)},
            'quantile': 0.25
        }),
        'sample_desc': 'Full sample with 99th percentile outliers removed',
        'fixed_effects': 'None',
        'controls_desc': 'None (bivariate regression)',
        'cluster_var': 'None',
        'model_type': 'Quantile (0.25)',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })

    # 75th percentile regression
    results_q75 = model_q50.fit(q=0.75)
    coef_q75 = results_q75.params.iloc[1]
    se_q75 = results_q75.bse.iloc[1]
    t_q75 = coef_q75 / se_q75
    p_q75 = 2 * (1 - stats.norm.cdf(abs(t_q75)))

    results_list.append({
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': 'robust/form/quantile_75',
        'spec_tree_path': 'robustness/functional_form.md#alternative-estimators',
        'outcome_var': 'forecast_error',
        'treatment_var': 'forecast_revision',
        'coefficient': coef_q75,
        'std_error': se_q75,
        't_stat': t_q75,
        'p_value': p_q75,
        'ci_lower': coef_q75 - 1.96 * se_q75,
        'ci_upper': coef_q75 + 1.96 * se_q75,
        'n_obs': int(results_q75.nobs),
        'r_squared': np.nan,
        'coefficient_vector_json': json.dumps({
            'treatment': {'var': 'forecast_revision', 'coef': float(coef_q75),
                         'se': float(se_q75), 'pval': float(p_q75)},
            'quantile': 0.75
        }),
        'sample_desc': 'Full sample with 99th percentile outliers removed',
        'fixed_effects': 'None',
        'controls_desc': 'None (bivariate regression)',
        'cluster_var': 'None',
        'model_type': 'Quantile (0.75)',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })

    # ==========================================================================
    # WITH CONTROLS (time trend)
    # ==========================================================================
    print("\n--- Running Specifications with Controls ---")

    # Add time trend as control
    df['time_trend'] = range(len(df))
    mask_ctrl = ~(y_full.isna() | X_full.isna())

    X_trend = df.loc[mask_ctrl, ['forecast_revision_clean', 'time_trend']]
    X_trend_const = sm.add_constant(X_trend)
    model_trend = sm.OLS(y_full[mask_ctrl], X_trend_const)
    results_trend = model_trend.fit()

    results_list.append(extract_results(
        results_trend,
        spec_id='ols/controls/time_trend',
        spec_tree_path='methods/cross_sectional_ols.md#control-sets',
        outcome_var='forecast_error',
        treatment_var='forecast_revision',
        sample_desc='Full sample with 99th percentile outliers removed',
        controls_desc='Linear time trend',
        cluster_var='None',
        model_type='OLS'
    ))

    # Add quadratic time trend
    df['time_trend_sq'] = df['time_trend'] ** 2
    X_trend2 = df.loc[mask_ctrl, ['forecast_revision_clean', 'time_trend', 'time_trend_sq']]
    X_trend2_const = sm.add_constant(X_trend2)
    model_trend2 = sm.OLS(y_full[mask_ctrl], X_trend2_const)
    results_trend2 = model_trend2.fit()

    results_list.append(extract_results(
        results_trend2,
        spec_id='ols/controls/quadratic_time_trend',
        spec_tree_path='methods/cross_sectional_ols.md#control-sets',
        outcome_var='forecast_error',
        treatment_var='forecast_revision',
        sample_desc='Full sample with 99th percentile outliers removed',
        controls_desc='Linear and quadratic time trend',
        cluster_var='None',
        model_type='OLS'
    ))

    # ==========================================================================
    # INTERACTION TERMS
    # ==========================================================================
    print("\n--- Running Interaction Specifications ---")

    # Interaction with time trend (test for time-varying information rigidity)
    df['fr_time_interact'] = df['forecast_revision_clean'] * df['time_trend']
    mask_int = ~(y_full.isna() | df['forecast_revision_clean'].isna())

    X_interact = df.loc[mask_int, ['forecast_revision_clean', 'time_trend', 'fr_time_interact']]
    X_interact_const = sm.add_constant(X_interact)
    model_interact = sm.OLS(y_full[mask_int], X_interact_const)
    results_interact = model_interact.fit()

    results_list.append(extract_results(
        results_interact,
        spec_id='robust/form/interact_time',
        spec_tree_path='robustness/functional_form.md#interaction-terms',
        outcome_var='forecast_error',
        treatment_var='forecast_revision',
        sample_desc='Full sample with 99th percentile outliers removed',
        controls_desc='Time trend, forecast_revision x time interaction',
        cluster_var='None',
        model_type='OLS'
    ))

    # ==========================================================================
    # ROBUST REGRESSION (M-estimation)
    # ==========================================================================
    print("\n--- Running Robust Regression ---")

    from statsmodels.robust.robust_linear_model import RLM

    X_rlm = sm.add_constant(X)
    model_rlm = RLM(y, X_rlm, M=sm.robust.norms.HuberT())
    results_rlm = model_rlm.fit()

    # Manually extract RLM results (RLM doesn't have rsquared)
    coef_rlm = results_rlm.params.iloc[1]
    se_rlm = results_rlm.bse.iloc[1]
    t_rlm = coef_rlm / se_rlm
    p_rlm = 2 * (1 - stats.norm.cdf(abs(t_rlm)))

    results_list.append({
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': 'ols/method/robust',
        'spec_tree_path': 'methods/cross_sectional_ols.md#estimation-method',
        'outcome_var': 'forecast_error',
        'treatment_var': 'forecast_revision',
        'coefficient': coef_rlm,
        'std_error': se_rlm,
        't_stat': t_rlm,
        'p_value': p_rlm,
        'ci_lower': coef_rlm - 1.96 * se_rlm,
        'ci_upper': coef_rlm + 1.96 * se_rlm,
        'n_obs': int(results_rlm.nobs),
        'r_squared': np.nan,  # Not well-defined for robust regression
        'coefficient_vector_json': json.dumps({
            'treatment': {'var': 'forecast_revision', 'coef': float(coef_rlm),
                         'se': float(se_rlm), 'pval': float(p_rlm)},
            'method': 'Huber M-estimation'
        }),
        'sample_desc': 'Full sample with 99th percentile outliers removed',
        'fixed_effects': 'None',
        'controls_desc': 'None (bivariate regression)',
        'cluster_var': 'Huber robust',
        'model_type': 'Robust (Huber-M)',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })

    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================
    print("\n--- Saving Results ---")

    results_df = pd.DataFrame(results_list)
    output_path = os.path.join(OUTPUT_DIR, 'specification_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"Saved {len(results_df)} specifications to {output_path}")

    return results_df


def generate_summary_statistics(results_df):
    """Generate summary statistics for the specification search."""

    # Basic stats
    n_specs = len(results_df)
    n_positive = (results_df['coefficient'] > 0).sum()
    pct_positive = n_positive / n_specs * 100

    # Significance at different levels
    n_sig_05 = (results_df['p_value'] < 0.05).sum()
    n_sig_01 = (results_df['p_value'] < 0.01).sum()
    pct_sig_05 = n_sig_05 / n_specs * 100
    pct_sig_01 = n_sig_01 / n_specs * 100

    # Coefficient statistics
    median_coef = results_df['coefficient'].median()
    mean_coef = results_df['coefficient'].mean()
    min_coef = results_df['coefficient'].min()
    max_coef = results_df['coefficient'].max()
    std_coef = results_df['coefficient'].std()

    summary = {
        'n_specifications': n_specs,
        'n_positive': n_positive,
        'pct_positive': pct_positive,
        'n_sig_05': n_sig_05,
        'pct_sig_05': pct_sig_05,
        'n_sig_01': n_sig_01,
        'pct_sig_01': pct_sig_01,
        'median_coefficient': median_coef,
        'mean_coefficient': mean_coef,
        'std_coefficient': std_coef,
        'min_coefficient': min_coef,
        'max_coefficient': max_coef
    }

    return summary


if __name__ == "__main__":
    # Run specification search
    results_df = run_specification_search()

    # Generate summary
    summary = generate_summary_statistics(results_df)

    print("\n" + "="*60)
    print("SPECIFICATION SEARCH SUMMARY")
    print("="*60)
    print(f"Total specifications: {summary['n_specifications']}")
    print(f"Positive coefficients: {summary['n_positive']} ({summary['pct_positive']:.1f}%)")
    print(f"Significant at 5%: {summary['n_sig_05']} ({summary['pct_sig_05']:.1f}%)")
    print(f"Significant at 1%: {summary['n_sig_01']} ({summary['pct_sig_01']:.1f}%)")
    print(f"Median coefficient: {summary['median_coefficient']:.6f}")
    print(f"Mean coefficient: {summary['mean_coefficient']:.6f}")
    print(f"Range: [{summary['min_coefficient']:.6f}, {summary['max_coefficient']:.6f}]")
    print("="*60)
