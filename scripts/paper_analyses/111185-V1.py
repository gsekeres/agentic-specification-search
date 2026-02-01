#!/usr/bin/env python3
"""
Specification Search: 111185-V1
"Optimal climate policy when damages are unknown" - Ivan Rudik
American Economic Journal: Economic Policy

This script replicates Table 1 damage parameter estimation and runs
systematic specification searches.

Usage:
    cd /path/to/111185-V1
    python3 /path/to/111185-V1.py
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.robust.robust_linear_model import RLM
from scipy import stats
import json
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = 'estimate_damage_parameters/10640_2017_166_MOESM10_ESM.dta'
OUTPUT_PATH = 'specification_results.csv'
N_BOOTSTRAP = 1000
RANDOM_SEED = 12345

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def add_result(results_list, spec_id, spec_tree_path, model,
               treatment_var='logt', outcome_var='log_correct',
               sample_desc='Full sample', controls_desc='None',
               cluster_var=None, model_type='OLS', notes=''):
    """Helper function to store regression results."""
    coef = model.params[treatment_var]
    se = model.bse[treatment_var]
    t_stat = model.tvalues[treatment_var]
    pval = model.pvalues[treatment_var]
    ci = model.conf_int().loc[treatment_var]

    # Build coefficient vector JSON safely
    try:
        r2 = float(model.rsquared) if hasattr(model, 'rsquared') and not np.isnan(model.rsquared) else None
    except:
        r2 = None
    try:
        r2_adj = float(model.rsquared_adj) if hasattr(model, 'rsquared_adj') and not np.isnan(model.rsquared_adj) else None
    except:
        r2_adj = None
    try:
        f_stat = float(model.fvalue) if hasattr(model, 'fvalue') and model.fvalue is not None else None
    except:
        f_stat = None
    try:
        f_pval = float(model.f_pvalue) if hasattr(model, 'f_pvalue') and model.f_pvalue is not None else None
    except:
        f_pval = None

    coef_vec = {
        'treatment': {'var': treatment_var, 'coef': float(coef), 'se': float(se), 'pval': float(pval)},
        'controls': [],
        'fixed_effects': [],
        'diagnostics': {
            'r_squared': r2,
            'adj_r_squared': r2_adj,
            'f_stat': f_stat,
            'f_pval': f_pval
        }
    }

    # Add intercept and other coefficients
    for var in model.params.index:
        if var != treatment_var:
            coef_vec['controls'].append({
                'var': var,
                'coef': float(model.params[var]),
                'se': float(model.bse[var]),
                'pval': float(model.pvalues[var])
            })

    results_list.append({
        'paper_id': '111185-V1',
        'journal': 'AEJ-Economic Policy',
        'paper_title': 'Optimal climate policy when damages are unknown',
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'outcome_var': outcome_var,
        'treatment_var': treatment_var,
        'coefficient': float(coef),
        'std_error': float(se),
        't_stat': float(t_stat),
        'p_value': float(pval),
        'ci_lower': float(ci[0]),
        'ci_upper': float(ci[1]),
        'n_obs': int(model.nobs),
        'r_squared': r2,
        'coefficient_vector_json': json.dumps(coef_vec),
        'sample_desc': sample_desc,
        'fixed_effects': 'None',
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': 'scripts/paper_analyses/111185-V1.py',
        'notes': notes
    })


def main():
    # ============================================================
    # DATA PREPARATION
    # ============================================================
    print("Loading data...")
    df = pd.read_stata(DATA_PATH)
    print(f"Raw data shape: {df.shape}")

    # Filter to valid observations (D_new > 0 to avoid log(0))
    df = df[df['D_new'] > 0].copy()
    print(f"After filtering D_new > 0: {len(df)}")

    # Translate % GDP loss into damages in the 1/(1+D) set up
    # GDP Loss = (Y^g - Y^n)/Y^g = 1 - Y^n/Y^g
    # Damage function: Y^g/Y^n - 1 = damages = GDP_loss/(1-GDP_loss)
    df['correct_d'] = (df['D_new']/100) / (1 - df['D_new']/100)

    # Generate log transformations
    df['log_correct'] = np.log(df['correct_d'])
    df = df[df['t'] > 0].copy()
    df['logt'] = np.log(df['t'])

    # Prepare regression sample
    df_reg = df[['log_correct', 'logt', 'correct_d', 't', 'D_new', 'Study',
                 'Primary_Author', 'Year', 'Market', 'Nonmarket', 'Preindustrial',
                 'Primary_model', 'Model', 'Method']].dropna(subset=['log_correct', 'logt'])
    print(f"Final regression sample size: {len(df_reg)}")

    results = []

    # ============================================================
    # BASELINE REGRESSION
    # ============================================================
    print("\nRunning baseline regression...")
    baseline_model = smf.ols('log_correct ~ logt', data=df_reg).fit()
    print(f"Baseline: coef={baseline_model.params['logt']:.4f}, "
          f"se={baseline_model.bse['logt']:.4f}, "
          f"p={baseline_model.pvalues['logt']:.6f}")

    add_result(results, 'baseline', 'methods/cross_sectional_ols.md#baseline',
               baseline_model, notes='Exact replication of Table 1 damage parameter estimation')

    coef_baseline = baseline_model.params['logt']
    se_baseline = baseline_model.bse['logt']
    r2_baseline = baseline_model.rsquared
    n_baseline = int(baseline_model.nobs)

    # ============================================================
    # STANDARD ERROR VARIATIONS
    # ============================================================
    print("\nRunning standard error variations...")
    for cov_type in ['HC1', 'HC2', 'HC3']:
        model = smf.ols('log_correct ~ logt', data=df_reg).fit(cov_type=cov_type)
        add_result(results, f'ols/se/{cov_type.lower()}',
                   'methods/cross_sectional_ols.md#standard-errors', model,
                   notes=f'{cov_type} heteroskedasticity-robust standard errors')

    # ============================================================
    # QUANTILE REGRESSION
    # ============================================================
    print("Running quantile regressions...")
    for q, name in [(0.25, '25'), (0.5, 'median'), (0.75, '75')]:
        qr = QuantReg(df_reg['log_correct'], sm.add_constant(df_reg['logt'])).fit(q=q)
        add_result(results, f'ols/method/quantile_{name}',
                   'methods/cross_sectional_ols.md#estimation-method', qr,
                   treatment_var='logt', model_type=f'Quantile({q})',
                   notes=f'{int(q*100)}th percentile regression')

    # ============================================================
    # ROBUST REGRESSION
    # ============================================================
    print("Running robust regression...")
    rlm_model = RLM(df_reg['log_correct'], sm.add_constant(df_reg['logt']),
                    M=sm.robust.norms.HuberT()).fit()
    add_result(results, 'ols/method/robust',
               'methods/cross_sectional_ols.md#estimation-method', rlm_model,
               treatment_var='logt', model_type='Robust(Huber)',
               notes='M-estimation with Huber T norm')

    # ============================================================
    # SAMPLE RESTRICTIONS
    # ============================================================
    print("Running sample restrictions...")

    # Trim outliers on Y
    for pct in [5, 10]:
        q_lo = df_reg['log_correct'].quantile(pct/100)
        q_hi = df_reg['log_correct'].quantile(1 - pct/100)
        df_trim = df_reg[(df_reg['log_correct'] >= q_lo) & (df_reg['log_correct'] <= q_hi)]
        model = smf.ols('log_correct ~ logt', data=df_trim).fit()
        add_result(results, f'robust/sample/trim_{pct}pct',
                   'robustness/sample_restrictions.md#outlier-handling', model,
                   sample_desc=f'Trim {pct}% tails on Y, N={len(df_trim)}',
                   notes=f'Drop top and bottom {pct}% of log(damage)')

    # Drop extreme observations
    for extrema, func, label in [('largest', 'max', '<'), ('smallest', 'min', '>')]:
        val = getattr(df_reg['log_correct'], func)()
        df_drop = df_reg[eval(f"df_reg['log_correct'] {label} val")]
        model = smf.ols('log_correct ~ logt', data=df_drop).fit()
        add_result(results, f'robust/sample/drop_{extrema}',
                   'robustness/sample_restrictions.md#geographic-restrictions', model,
                   sample_desc=f'Drop {extrema} damage observation, N={len(df_drop)}',
                   notes=f'Exclude observation with {func}imum damage')

    # ============================================================
    # FUNCTIONAL FORM VARIATIONS
    # ============================================================
    print("Running functional form variations...")

    # Level-level
    model = smf.ols('correct_d ~ t', data=df_reg).fit()
    add_result(results, 'robust/form/level_level', 'robustness/functional_form.md',
               model, treatment_var='t', outcome_var='correct_d',
               notes='Level-level specification (no log transforms)')

    # Semi-log specifications
    model = smf.ols('correct_d ~ logt', data=df_reg).fit()
    add_result(results, 'robust/form/level_log', 'robustness/functional_form.md',
               model, treatment_var='logt', outcome_var='correct_d',
               notes='Semi-log: level(Y) ~ log(X)')

    model = smf.ols('log_correct ~ t', data=df_reg).fit()
    add_result(results, 'robust/form/log_level', 'robustness/functional_form.md',
               model, treatment_var='t', outcome_var='log_correct',
               notes='Log-level: log(Y) ~ level(X)')

    # Polynomial specifications
    df_reg['logt2'] = df_reg['logt'] ** 2
    df_reg['logt3'] = df_reg['logt'] ** 3
    df_reg['t2'] = df_reg['t'] ** 2

    model = smf.ols('log_correct ~ logt + logt2', data=df_reg).fit()
    add_result(results, 'robust/form/quadratic',
               'robustness/functional_form.md#nonlinear-specifications', model,
               treatment_var='logt', controls_desc='logt^2',
               notes='Quadratic specification: log(Y) ~ log(X) + log(X)^2')

    model = smf.ols('log_correct ~ logt + logt2 + logt3', data=df_reg).fit()
    add_result(results, 'robust/form/cubic',
               'robustness/functional_form.md#nonlinear-specifications', model,
               treatment_var='logt', controls_desc='logt^2 + logt^3',
               notes='Cubic specification')

    model = smf.ols('correct_d ~ t + t2', data=df_reg).fit()
    add_result(results, 'robust/form/level_quadratic',
               'robustness/functional_form.md#nonlinear-specifications', model,
               treatment_var='t', outcome_var='correct_d', controls_desc='t^2',
               notes='Quadratic in levels')

    # ============================================================
    # LEAVE-ONE-OUT
    # ============================================================
    print("Running leave-one-out analysis...")
    for idx in df_reg.index:
        study_name = str(df_reg.loc[idx, 'Study'])[:30] if pd.notna(df_reg.loc[idx, 'Study']) else f'obs_{idx}'
        df_loo = df_reg.drop(idx)
        model = smf.ols('log_correct ~ logt', data=df_loo).fit()
        add_result(results, f'robust/loo/drop_obs_{idx}', 'robustness/leave_one_out.md',
                   model, sample_desc=f'Drop observation {idx}: {study_name}',
                   notes=f'Leave-one-out: drop {study_name}')

    # ============================================================
    # INFLUENTIAL OBSERVATIONS
    # ============================================================
    print("Running influential observation analysis...")
    influence = baseline_model.get_influence()
    cooks_d = influence.cooks_distance[0]
    threshold = 4 / len(df_reg)
    df_no_influential = df_reg[cooks_d < threshold]
    if len(df_no_influential) < len(df_reg):
        model = smf.ols('log_correct ~ logt', data=df_no_influential).fit()
        add_result(results, 'robust/sample/drop_influential',
                   'robustness/sample_restrictions.md', model,
                   sample_desc=f"Drop high Cook's D (>{threshold:.4f}), N={len(df_no_influential)}",
                   notes="Exclude observations with Cook's D > 4/n")

    # ============================================================
    # WEIGHTED LEAST SQUARES
    # ============================================================
    print("Running WLS...")
    abs_resid = np.abs(baseline_model.resid)
    weights = 1 / (abs_resid + 0.001)
    weights = weights / weights.mean()
    model = sm.WLS(df_reg['log_correct'], sm.add_constant(df_reg['logt']), weights=weights).fit()
    add_result(results, 'ols/method/wls', 'methods/cross_sectional_ols.md#estimation-method',
               model, treatment_var='logt', model_type='WLS',
               notes='WLS with inverse residual weights')

    # ============================================================
    # BOOTSTRAP STANDARD ERRORS
    # ============================================================
    print(f"Running bootstrap ({N_BOOTSTRAP} replications)...")
    np.random.seed(RANDOM_SEED)
    boot_coefs = []
    for _ in range(N_BOOTSTRAP):
        boot_idx = np.random.choice(len(df_reg), size=len(df_reg), replace=True)
        df_boot = df_reg.iloc[boot_idx]
        try:
            model = smf.ols('log_correct ~ logt', data=df_boot).fit()
            boot_coefs.append(model.params['logt'])
        except:
            pass

    boot_se = np.std(boot_coefs)
    boot_ci_lo = np.percentile(boot_coefs, 2.5)
    boot_ci_hi = np.percentile(boot_coefs, 97.5)

    results.append({
        'paper_id': '111185-V1',
        'journal': 'AEJ-Economic Policy',
        'paper_title': 'Optimal climate policy when damages are unknown',
        'spec_id': 'ols/se/bootstrap',
        'spec_tree_path': 'methods/cross_sectional_ols.md#standard-errors',
        'outcome_var': 'log_correct',
        'treatment_var': 'logt',
        'coefficient': float(coef_baseline),
        'std_error': float(boot_se),
        't_stat': float(coef_baseline / boot_se),
        'p_value': float(2 * (1 - stats.norm.cdf(abs(coef_baseline / boot_se)))),
        'ci_lower': float(boot_ci_lo),
        'ci_upper': float(boot_ci_hi),
        'n_obs': n_baseline,
        'r_squared': float(r2_baseline),
        'coefficient_vector_json': json.dumps({
            'treatment': {'var': 'logt', 'coef': float(coef_baseline), 'se': float(boot_se)},
            'bootstrap': {'n_reps': N_BOOTSTRAP, 'method': 'pairs'}
        }),
        'sample_desc': 'Full sample',
        'fixed_effects': 'None',
        'controls_desc': 'None',
        'cluster_var': None,
        'model_type': 'OLS (Bootstrap SE)',
        'estimation_script': 'scripts/paper_analyses/111185-V1.py',
        'notes': f'Bootstrap standard errors ({N_BOOTSTRAP} replications)'
    })

    # ============================================================
    # SUBSAMPLES
    # ============================================================
    print("Running subsample analyses...")

    # Temperature subsamples
    t_median = df_reg['t'].median()
    for label, condition in [('high_temperature', df_reg['t'] >= t_median),
                              ('low_temperature', df_reg['t'] < t_median)]:
        df_sub = df_reg[condition]
        if len(df_sub) >= 5:
            model = smf.ols('log_correct ~ logt', data=df_sub).fit()
            add_result(results, f'robust/sample/{label}',
                       'robustness/sample_restrictions.md#demographic-subgroups', model,
                       sample_desc=f'Temperature {">=": "high", "<": "low"}[label.split("_")[0]] median ({t_median:.2f}), N={len(df_sub)}',
                       notes=f'Subsample: {label.replace("_", " ")} studies')

    # Damage subsamples
    d_median = df_reg['D_new'].median()
    for label, condition in [('high_damage', df_reg['D_new'] >= d_median),
                              ('low_damage', df_reg['D_new'] < d_median)]:
        df_sub = df_reg[condition]
        if len(df_sub) >= 5:
            model = smf.ols('log_correct ~ logt', data=df_sub).fit()
            add_result(results, f'robust/sample/{label}',
                       'robustness/sample_restrictions.md#demographic-subgroups', model,
                       sample_desc=f'Damage {">=": "high", "<": "low"}[label.split("_")[0]] median ({d_median:.2f}%), N={len(df_sub)}',
                       notes=f'Subsample: {label.replace("_", " ")} estimates')

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    print("\nSaving results...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Total specifications: {len(results_df)}")
    print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
    print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
    print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")
    print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
    print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")
    print(f"Range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
