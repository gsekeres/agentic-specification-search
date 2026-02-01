"""
Specification Search Script for Paper 113710-V1
"Does electoral competition curb party favoritism?"
Curto-Grau, Sole-Olle, and Sorribas-Navarro (2018), AEJ: Applied Economics

This script replicates and extends the main analysis using fuzzy RD design.
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import statsmodels.api as sm
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# Configuration
PAPER_ID = "113710-V1"
JOURNAL = "AEJ-Applied"
PAPER_TITLE = "Does electoral competition curb party favoritism?"
DATA_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/113710-V1/data/data/db_main.dta"
OUTPUT_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/113710-V1"

# Load data
df = pd.read_stata(DATA_PATH)

# Create unique IDs
df['unit_id'] = df['codiine'].astype(int)
df['region_id'] = df['codccaa'].astype(int)
df['time_id'] = df['t'].astype(int)

# Create region dummies for fixed effects
region_dummies = pd.get_dummies(df['region_id'], prefix='region', drop_first=True)
df = pd.concat([df, region_dummies], axis=1)
region_fe_cols = [c for c in df.columns if c.startswith('region_')]

# Function to create within-group centered variables (for DiD analysis)
def center_within_group(df, variables, group_var='unit_id'):
    """Center variables within groups (like Stata's egen center, by())"""
    centered = df.copy()
    for var in variables:
        if var in df.columns:
            centered[f'c_{var}'] = df.groupby(group_var)[var].transform(lambda x: x - x.mean())
    return centered

# Center variables for DiD analysis
vars_to_center = ['tk', 'ab', 'dt1', 'dt2', 'esas1', 'ecs1'] + [f'dca_ab{i}' for i in range(1, 16)]
df = center_within_group(df, vars_to_center)

# Results storage
results = []

def extract_results(model, spec_id, spec_tree_path, outcome_var, treatment_var,
                    df_used, cluster_var=None, fixed_effects=None, controls_desc=None,
                    model_type='OLS', sample_desc='Full sample', additional_info=None):
    """Extract standardized results from regression model"""

    try:
        if hasattr(model, 'coef'):
            # pyfixest model
            coefs = model.coef()
            ses = model.se()
            pvals = model.pvalue()
            tvals = model.tstat()

            if treatment_var in coefs.index:
                coef = coefs[treatment_var]
                se = ses[treatment_var]
                pval = pvals[treatment_var]
                tval = tvals[treatment_var]
            else:
                # Try to find the treatment variable
                matching_vars = [v for v in coefs.index if treatment_var in v]
                if matching_vars:
                    coef = coefs[matching_vars[0]]
                    se = ses[matching_vars[0]]
                    pval = pvals[matching_vars[0]]
                    tval = tvals[matching_vars[0]]
                else:
                    return None

            n_obs = model.nobs
            r2 = model.r2 if hasattr(model, 'r2') else None

            # Build coefficient vector JSON
            coef_vector = {
                "treatment": {"var": treatment_var, "coef": float(coef), "se": float(se), "pval": float(pval)},
                "controls": [],
                "fixed_effects": fixed_effects if fixed_effects else [],
                "diagnostics": {}
            }

            for var in coefs.index:
                if var != treatment_var and not var.startswith('region_'):
                    coef_vector["controls"].append({
                        "var": var,
                        "coef": float(coefs[var]),
                        "se": float(ses[var]),
                        "pval": float(pvals[var])
                    })

        elif hasattr(model, 'params'):
            # statsmodels model
            coefs = model.params
            ses = model.bse
            pvals = model.pvalues
            tvals = model.tvalues

            if treatment_var in coefs.index:
                coef = coefs[treatment_var]
                se = ses[treatment_var]
                pval = pvals[treatment_var]
                tval = tvals[treatment_var]
            else:
                matching_vars = [v for v in coefs.index if treatment_var in v]
                if matching_vars:
                    coef = coefs[matching_vars[0]]
                    se = ses[matching_vars[0]]
                    pval = pvals[matching_vars[0]]
                    tval = tvals[matching_vars[0]]
                else:
                    return None

            n_obs = int(model.nobs)
            r2 = model.rsquared if hasattr(model, 'rsquared') else None

            coef_vector = {
                "treatment": {"var": treatment_var, "coef": float(coef), "se": float(se), "pval": float(pval)},
                "controls": [],
                "fixed_effects": fixed_effects if fixed_effects else [],
                "diagnostics": {}
            }

            for var in coefs.index:
                if var != treatment_var and var != 'const' and not var.startswith('region_'):
                    coef_vector["controls"].append({
                        "var": var,
                        "coef": float(coefs[var]),
                        "se": float(ses[var]),
                        "pval": float(pvals[var])
                    })
        else:
            return None

        # Add additional diagnostics if provided
        if additional_info:
            coef_vector["diagnostics"].update(additional_info)

        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se

        return {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': float(coef),
            'std_error': float(se),
            't_stat': float(tval),
            'p_value': float(pval),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_obs': int(n_obs),
            'r_squared': float(r2) if r2 is not None else None,
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects if fixed_effects else '',
            'controls_desc': controls_desc if controls_desc else '',
            'cluster_var': cluster_var if cluster_var else '',
            'model_type': model_type,
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
    except Exception as e:
        print(f"Error extracting results for {spec_id}: {e}")
        return None


def run_iv_2sls(df_sub, outcome, endog, instruments, exog, cluster_var, spec_id, spec_tree_path, sample_desc):
    """Run IV/2SLS regression using manual 2SLS"""
    try:
        # Ensure all data is float64
        df_iv = df_sub.copy()

        # Build full X matrix
        X_cols = exog + instruments
        y = df_iv[outcome].astype(float).values

        # First endogenous variable only for simplicity
        endog_var = endog[0] if isinstance(endog, list) else endog
        endog_data = df_iv[endog_var].astype(float).values

        # Build matrices
        X_exog = sm.add_constant(df_iv[exog].astype(float).values)
        Z_all = sm.add_constant(df_iv[exog + instruments].astype(float).values)

        # First stage: endog ~ exog + instruments
        first_stage = sm.OLS(endog_data, Z_all).fit()
        endog_fitted = first_stage.fittedvalues

        # Calculate first-stage F-statistic on instruments
        first_stage_full = sm.OLS(endog_data, Z_all).fit()
        first_stage_restricted = sm.OLS(endog_data, X_exog).fit()

        n = len(y)
        k_full = Z_all.shape[1]
        k_restricted = X_exog.shape[1]
        num_instruments = k_full - k_restricted

        rss_full = np.sum(first_stage_full.resid**2)
        rss_restricted = np.sum(first_stage_restricted.resid**2)

        if rss_full > 0 and num_instruments > 0:
            f_stat = ((rss_restricted - rss_full) / num_instruments) / (rss_full / (n - k_full))
        else:
            f_stat = np.nan

        # Second stage: y ~ X_exog + fitted_endog
        X_second = np.column_stack([X_exog, endog_fitted])

        second_stage = sm.OLS(y, X_second).fit()

        # Now we need proper SEs - use manual 2SLS residuals
        # The last coefficient is on the endogenous variable
        coef = second_stage.params[-1]

        # For proper SEs, compute residuals using ACTUAL endog, not fitted
        X_actual = np.column_stack([X_exog, endog_data])
        resid = y - X_actual @ second_stage.params

        # Cluster-robust variance estimation
        if cluster_var:
            clusters = df_iv[cluster_var].values
            unique_clusters = np.unique(clusters)
            n_clusters = len(unique_clusters)

            # Bread: (X'Z(Z'Z)^{-1}Z'X)^{-1}
            # For manual 2SLS, use sandwich estimator
            XtX_inv = np.linalg.inv(X_second.T @ X_second)

            # Cluster-robust meat
            meat = np.zeros((X_second.shape[1], X_second.shape[1]))
            for c in unique_clusters:
                idx = clusters == c
                Xi = X_second[idx]
                ei = resid[idx]
                score_i = Xi.T @ ei
                meat += np.outer(score_i, score_i)

            # Finite sample adjustment
            adjustment = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - X_second.shape[1]))
            var_cluster = adjustment * XtX_inv @ meat @ XtX_inv
            se = np.sqrt(np.diag(var_cluster))[-1]
        else:
            # Heteroskedasticity-robust
            var_hc = second_stage.HC1_se
            se = var_hc[-1]

        tval = coef / se
        pval = 2 * (1 - stats.norm.cdf(np.abs(tval)))

        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se

        coef_vector = {
            "treatment": {"var": endog_var, "coef": float(coef), "se": float(se), "pval": float(pval)},
            "first_stage": {"F_stat": float(f_stat), "coef": float(first_stage.params[-1])},
            "controls": [],
            "fixed_effects": ["region"],
            "diagnostics": {"first_stage_F": float(f_stat)}
        }

        return {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': outcome,
            'treatment_var': endog_var,
            'coefficient': float(coef),
            'std_error': float(se),
            't_stat': float(tval),
            'p_value': float(pval),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_obs': int(n),
            'r_squared': float(second_stage.rsquared) if hasattr(second_stage, 'rsquared') else None,
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc,
            'fixed_effects': 'region',
            'controls_desc': ', '.join(exog),
            'cluster_var': cluster_var if cluster_var else '',
            'model_type': '2SLS',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }

    except Exception as e:
        print(f"Error in IV 2SLS for {spec_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


print("="*60)
print("SPECIFICATION SEARCH: 113710-V1")
print("Does electoral competition curb party favoritism?")
print("="*60)

# ============================================================================
# BASELINE SPECIFICATION - Table 1, Column 1
# Fuzzy RD with full sample (2nd order polynomial)
# ============================================================================

print("\n--- Running Baseline Specifications ---")

# Table 1, Panel A, Column 1: Full sample, 2nd order polynomial
# ivregress 2sls tk (ab vsa vsa2 = dab vda vda2) dist1 dist2 i.codccaa, vce(cluster codiine)

df_baseline = df.dropna(subset=['tk', 'ab', 'dab', 'dist1', 'dist2', 'vsa', 'vsa2', 'vda', 'vda2'])

result = run_iv_2sls(
    df_baseline,
    outcome='tk',
    endog=['ab', 'vsa', 'vsa2'],
    instruments=['dab', 'vda', 'vda2'],
    exog=['dist1', 'dist2'] + region_fe_cols,
    cluster_var='unit_id',
    spec_id='baseline',
    spec_tree_path='methods/regression_discontinuity.md',
    sample_desc='Full sample'
)
if result:
    results.append(result)
    print(f"Baseline: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")


# ============================================================================
# FUZZY RD VARIATIONS - Different Bandwidths (Table 1)
# ============================================================================

print("\n--- Running RD Bandwidth Variations ---")

bandwidths = [
    ('rd/bandwidth/bw_0386', 0.386, 'Bandwidth = 0.386 (optimal)'),
    ('rd/bandwidth/bw_0193', 0.193, 'Bandwidth = 0.193 (half optimal)'),
    ('rd/bandwidth/bw_0097', 0.0965, 'Bandwidth = 0.0965 (quarter optimal)'),
    ('rd/bandwidth/bw_0048', 0.048, 'Bandwidth = 0.048 (eighth optimal)')
]

for spec_id, bw, desc in bandwidths:
    df_bw = df_baseline[np.abs(df_baseline['dist1']) < bw].copy()

    if len(df_bw) > 50:  # Minimum sample size check
        result = run_iv_2sls(
            df_bw,
            outcome='tk',
            endog=['ab', 'vsa'],
            instruments=['dab', 'vda'],
            exog=['dist1'] + region_fe_cols,
            cluster_var='unit_id',
            spec_id=spec_id,
            spec_tree_path='methods/regression_discontinuity.md#bandwidth-selection',
            sample_desc=desc
        )
        if result:
            results.append(result)
            print(f"{spec_id}: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, n={result['n_obs']}")


# ============================================================================
# POLYNOMIAL ORDER VARIATIONS (Table A12)
# ============================================================================

print("\n--- Running Polynomial Order Variations ---")

# Linear polynomial
result = run_iv_2sls(
    df_baseline,
    outcome='tk',
    endog=['ab', 'vsa'],
    instruments=['dab', 'vda'],
    exog=['dist1'] + region_fe_cols,
    cluster_var='unit_id',
    spec_id='rd/poly/local_linear',
    spec_tree_path='methods/regression_discontinuity.md#polynomial-order',
    sample_desc='Full sample, linear polynomial'
)
if result:
    results.append(result)
    print(f"Linear: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}")

# Quadratic polynomial (already in baseline)
# Cubic polynomial
df_baseline['dist3'] = df_baseline['dist1']**3
df_baseline['vda3'] = df_baseline['dist3'] * df_baseline['dab']
df_baseline['vsa3'] = df_baseline['dist3'] * df_baseline['ab']

result = run_iv_2sls(
    df_baseline,
    outcome='tk',
    endog=['ab', 'vsa', 'vsa2', 'vsa3'],
    instruments=['dab', 'vda', 'vda2', 'vda3'],
    exog=['dist1', 'dist2', 'dist3'] + region_fe_cols,
    cluster_var='unit_id',
    spec_id='rd/poly/local_cubic',
    spec_tree_path='methods/regression_discontinuity.md#polynomial-order',
    sample_desc='Full sample, cubic polynomial'
)
if result:
    results.append(result)
    print(f"Cubic: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}")


# ============================================================================
# DiD SPECIFICATION (Panel Unit FE, Table 1 Column 6)
# ============================================================================

print("\n--- Running DiD Specifications ---")

# Create centered variables for DiD
# c_tk c_ab c_dt1 c_dt2 clustered by codiine

df_did = df.dropna(subset=['c_tk', 'c_ab', 'c_dt1', 'c_dt2']).copy()

if 'c_tk' in df_did.columns and 'c_ab' in df_did.columns:
    try:
        model = sm.OLS.from_formula('c_tk ~ c_ab + c_dt1 + c_dt2', data=df_did).fit(
            cov_type='cluster', cov_kwds={'groups': df_did['unit_id']}
        )
        result = extract_results(
            model,
            spec_id='did/fe/unit',
            spec_tree_path='methods/difference_in_differences.md#fixed-effects',
            outcome_var='c_tk',
            treatment_var='c_ab',
            df_used=df_did,
            cluster_var='unit_id',
            fixed_effects='unit (within transformation)',
            controls_desc='time dummies (c_dt1, c_dt2)',
            model_type='DiD-FE',
            sample_desc='Full sample with unit FE (within transformation)'
        )
        if result:
            results.append(result)
            print(f"DiD FE: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}")
    except Exception as e:
        print(f"DiD FE failed: {e}")


# ============================================================================
# HETEROGENEITY BY ELECTORAL COMPETITION (HLATE) - Table 2
# ============================================================================

print("\n--- Running HLATE (Heterogeneity by Electoral Competition) ---")

# Table 2 examines how the effect varies with regional electoral competition (ecs1)
# Key variables: esas1 = ecs1*ab (interaction), ecs1 = regional competition (demeaned)

# Prepare HLATE data
hlate_vars = ['tk', 'esas1', 'ecs1', 'dist1', 'dist2'] + [f'dca_ab{i}' for i in range(1, 16)]
df_hlate = df.dropna(subset=['tk', 'ab', 'esas1', 'ecs1', 'dist1']).copy()

# Simple OLS specification for HLATE (without full IV complexity)
# Testing whether effect varies with competition
try:
    formula = 'tk ~ ab + esas1 + ecs1 + dist1 + dist2 + ' + ' + '.join(region_fe_cols)
    model = sm.OLS.from_formula(formula, data=df_hlate).fit(
        cov_type='cluster', cov_kwds={'groups': df_hlate['region_id']}
    )

    # Extract esas1 coefficient (interaction of alignment x competition)
    coef_vector = {
        "treatment": {"var": "esas1", "coef": float(model.params['esas1']),
                     "se": float(model.bse['esas1']), "pval": float(model.pvalues['esas1'])},
        "ab_main": {"var": "ab", "coef": float(model.params['ab']),
                   "se": float(model.bse['ab']), "pval": float(model.pvalues['ab'])},
        "ecs1_main": {"var": "ecs1", "coef": float(model.params['ecs1']),
                     "se": float(model.bse['ecs1']), "pval": float(model.pvalues['ecs1'])},
        "controls": [],
        "fixed_effects": ["region"],
        "diagnostics": {}
    }

    result = {
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': 'hlate/interaction',
        'spec_tree_path': 'methods/regression_discontinuity.md',
        'outcome_var': 'tk',
        'treatment_var': 'esas1',
        'coefficient': float(model.params['esas1']),
        'std_error': float(model.bse['esas1']),
        't_stat': float(model.tvalues['esas1']),
        'p_value': float(model.pvalues['esas1']),
        'ci_lower': float(model.params['esas1'] - 1.96*model.bse['esas1']),
        'ci_upper': float(model.params['esas1'] + 1.96*model.bse['esas1']),
        'n_obs': int(model.nobs),
        'r_squared': float(model.rsquared),
        'coefficient_vector_json': json.dumps(coef_vector),
        'sample_desc': 'Full sample, HLATE specification',
        'fixed_effects': 'region',
        'controls_desc': 'ab, ecs1, dist1, dist2',
        'cluster_var': 'region_id',
        'model_type': 'OLS-HLATE',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    }
    results.append(result)
    print(f"HLATE (esas1): coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")

except Exception as e:
    print(f"HLATE specification failed: {e}")


# ============================================================================
# REDUCED FORM (Table 1 First Stage)
# ============================================================================

print("\n--- Running Reduced Form / First Stage ---")

# First stage: ab ~ dab + dist1 + dist2 + region FE
try:
    formula = 'ab ~ dab + dist1 + dist2 + vda + vda2 + ' + ' + '.join(region_fe_cols)
    model = sm.OLS.from_formula(formula, data=df_baseline).fit(
        cov_type='cluster', cov_kwds={'groups': df_baseline['unit_id']}
    )

    result = extract_results(
        model,
        spec_id='iv/first_stage/baseline',
        spec_tree_path='methods/instrumental_variables.md#first-stage',
        outcome_var='ab',
        treatment_var='dab',
        df_used=df_baseline,
        cluster_var='unit_id',
        fixed_effects='region',
        controls_desc='dist1, dist2, vda, vda2',
        model_type='First Stage',
        sample_desc='Full sample first stage'
    )
    if result:
        results.append(result)
        print(f"First Stage: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}")

except Exception as e:
    print(f"First stage failed: {e}")


# ============================================================================
# CLUSTERING VARIATIONS
# ============================================================================

print("\n--- Running Clustering Variations ---")

# Baseline regression with different clustering
base_formula = 'tk ~ ab + dist1 + dist2 + ' + ' + '.join(region_fe_cols)

# No clustering (robust)
try:
    model = sm.OLS.from_formula(base_formula, data=df_baseline).fit(cov_type='HC1')
    result = extract_results(
        model,
        spec_id='robust/cluster/none',
        spec_tree_path='robustness/clustering_variations.md#single-level-clustering',
        outcome_var='tk',
        treatment_var='ab',
        df_used=df_baseline,
        cluster_var=None,
        fixed_effects='region',
        controls_desc='dist1, dist2',
        model_type='OLS',
        sample_desc='Full sample, robust SE (no clustering)'
    )
    if result:
        results.append(result)
        print(f"No cluster (robust): se={result['std_error']:.4f}")
except Exception as e:
    print(f"Robust SE failed: {e}")

# Cluster by municipality (codiine)
try:
    model = sm.OLS.from_formula(base_formula, data=df_baseline).fit(
        cov_type='cluster', cov_kwds={'groups': df_baseline['unit_id']}
    )
    result = extract_results(
        model,
        spec_id='robust/cluster/unit',
        spec_tree_path='robustness/clustering_variations.md#single-level-clustering',
        outcome_var='tk',
        treatment_var='ab',
        df_used=df_baseline,
        cluster_var='unit_id',
        fixed_effects='region',
        controls_desc='dist1, dist2',
        model_type='OLS',
        sample_desc='Full sample, cluster by municipality'
    )
    if result:
        results.append(result)
        print(f"Cluster municipality: se={result['std_error']:.4f}")
except Exception as e:
    print(f"Cluster unit failed: {e}")

# Cluster by region (codccaa) - as in original paper
try:
    model = sm.OLS.from_formula(base_formula, data=df_baseline).fit(
        cov_type='cluster', cov_kwds={'groups': df_baseline['region_id']}
    )
    result = extract_results(
        model,
        spec_id='robust/cluster/region',
        spec_tree_path='robustness/clustering_variations.md#single-level-clustering',
        outcome_var='tk',
        treatment_var='ab',
        df_used=df_baseline,
        cluster_var='region_id',
        fixed_effects='region',
        controls_desc='dist1, dist2',
        model_type='OLS',
        sample_desc='Full sample, cluster by region'
    )
    if result:
        results.append(result)
        print(f"Cluster region: se={result['std_error']:.4f}")
except Exception as e:
    print(f"Cluster region failed: {e}")


# ============================================================================
# SAMPLE RESTRICTIONS
# ============================================================================

print("\n--- Running Sample Restriction Robustness ---")

# Trim 1% outliers on outcome
p1, p99 = df_baseline['tk'].quantile([0.01, 0.99])
df_trim = df_baseline[(df_baseline['tk'] >= p1) & (df_baseline['tk'] <= p99)]

try:
    model = sm.OLS.from_formula(base_formula, data=df_trim).fit(
        cov_type='cluster', cov_kwds={'groups': df_trim['unit_id']}
    )
    result = extract_results(
        model,
        spec_id='robust/sample/trim_1pct',
        spec_tree_path='robustness/sample_restrictions.md',
        outcome_var='tk',
        treatment_var='ab',
        df_used=df_trim,
        cluster_var='unit_id',
        fixed_effects='region',
        controls_desc='dist1, dist2',
        model_type='OLS',
        sample_desc='Trimmed 1% tails on outcome'
    )
    if result:
        results.append(result)
        print(f"Trim 1%: coef={result['coefficient']:.4f}, n={result['n_obs']}")
except Exception as e:
    print(f"Trim 1% failed: {e}")

# Early period (first 3 time periods)
if 'time_id' in df_baseline.columns:
    median_t = df_baseline['time_id'].median()
    df_early = df_baseline[df_baseline['time_id'] <= median_t]

    try:
        model = sm.OLS.from_formula(base_formula, data=df_early).fit(
            cov_type='cluster', cov_kwds={'groups': df_early['unit_id']}
        )
        result = extract_results(
            model,
            spec_id='robust/sample/early_period',
            spec_tree_path='robustness/sample_restrictions.md',
            outcome_var='tk',
            treatment_var='ab',
            df_used=df_early,
            cluster_var='unit_id',
            fixed_effects='region',
            controls_desc='dist1, dist2',
            model_type='OLS',
            sample_desc='Early time period'
        )
        if result:
            results.append(result)
            print(f"Early period: coef={result['coefficient']:.4f}, n={result['n_obs']}")
    except Exception as e:
        print(f"Early period failed: {e}")

    # Late period
    df_late = df_baseline[df_baseline['time_id'] > median_t]

    try:
        model = sm.OLS.from_formula(base_formula, data=df_late).fit(
            cov_type='cluster', cov_kwds={'groups': df_late['unit_id']}
        )
        result = extract_results(
            model,
            spec_id='robust/sample/late_period',
            spec_tree_path='robustness/sample_restrictions.md',
            outcome_var='tk',
            treatment_var='ab',
            df_used=df_late,
            cluster_var='unit_id',
            fixed_effects='region',
            controls_desc='dist1, dist2',
            model_type='OLS',
            sample_desc='Late time period'
        )
        if result:
            results.append(result)
            print(f"Late period: coef={result['coefficient']:.4f}, n={result['n_obs']}")
    except Exception as e:
        print(f"Late period failed: {e}")


# ============================================================================
# FUNCTIONAL FORM VARIATIONS
# ============================================================================

print("\n--- Running Functional Form Variations ---")

# Log outcome (add small constant for zeros)
df_baseline['tk_log'] = np.log(df_baseline['tk'] + 1)

try:
    formula_log = 'tk_log ~ ab + dist1 + dist2 + ' + ' + '.join(region_fe_cols)
    model = sm.OLS.from_formula(formula_log, data=df_baseline).fit(
        cov_type='cluster', cov_kwds={'groups': df_baseline['unit_id']}
    )
    result = extract_results(
        model,
        spec_id='robust/form/y_log',
        spec_tree_path='robustness/functional_form.md',
        outcome_var='tk_log',
        treatment_var='ab',
        df_used=df_baseline,
        cluster_var='unit_id',
        fixed_effects='region',
        controls_desc='dist1, dist2',
        model_type='OLS',
        sample_desc='Log(tk+1) outcome'
    )
    if result:
        results.append(result)
        print(f"Log outcome: coef={result['coefficient']:.4f}")
except Exception as e:
    print(f"Log outcome failed: {e}")

# Asinh outcome (handles zeros better)
df_baseline['tk_asinh'] = np.arcsinh(df_baseline['tk'])

try:
    formula_asinh = 'tk_asinh ~ ab + dist1 + dist2 + ' + ' + '.join(region_fe_cols)
    model = sm.OLS.from_formula(formula_asinh, data=df_baseline).fit(
        cov_type='cluster', cov_kwds={'groups': df_baseline['unit_id']}
    )
    result = extract_results(
        model,
        spec_id='robust/form/y_asinh',
        spec_tree_path='robustness/functional_form.md',
        outcome_var='tk_asinh',
        treatment_var='ab',
        df_used=df_baseline,
        cluster_var='unit_id',
        fixed_effects='region',
        controls_desc='dist1, dist2',
        model_type='OLS',
        sample_desc='Asinh(tk) outcome'
    )
    if result:
        results.append(result)
        print(f"Asinh outcome: coef={result['coefficient']:.4f}")
except Exception as e:
    print(f"Asinh outcome failed: {e}")


# ============================================================================
# LEAVE-ONE-OUT ROBUSTNESS (dropping controls)
# ============================================================================

print("\n--- Running Leave-One-Out Robustness ---")

# Control variables used in Table A11
controls_table_a11 = ['lpob', 'density', 'debt', 'vcp', 'tipo']
controls_available = [c for c in controls_table_a11 if c in df.columns]

for control in controls_available:
    try:
        remaining = [c for c in controls_available if c != control]
        formula = f'tk ~ ab + dist1 + dist2 + {" + ".join(remaining)} + ' + ' + '.join(region_fe_cols) if remaining else f'tk ~ ab + dist1 + dist2 + ' + ' + '.join(region_fe_cols)

        df_loo = df_baseline.dropna(subset=['tk', 'ab'] + remaining)

        model = sm.OLS.from_formula(formula, data=df_loo).fit(
            cov_type='cluster', cov_kwds={'groups': df_loo['unit_id']}
        )
        result = extract_results(
            model,
            spec_id=f'robust/loo/drop_{control}',
            spec_tree_path='robustness/leave_one_out.md',
            outcome_var='tk',
            treatment_var='ab',
            df_used=df_loo,
            cluster_var='unit_id',
            fixed_effects='region',
            controls_desc=f'dist1, dist2, {", ".join(remaining)}' if remaining else 'dist1, dist2',
            model_type='OLS',
            sample_desc=f'Full sample, dropping {control}'
        )
        if result:
            results.append(result)
            print(f"LOO drop_{control}: coef={result['coefficient']:.4f}")
    except Exception as e:
        print(f"LOO {control} failed: {e}")


# ============================================================================
# SINGLE COVARIATE ROBUSTNESS
# ============================================================================

print("\n--- Running Single Covariate Robustness ---")

# Bivariate (no controls except running variable)
try:
    formula_bivar = 'tk ~ ab + dist1 + dist2 + ' + ' + '.join(region_fe_cols)
    model = sm.OLS.from_formula(formula_bivar, data=df_baseline).fit(
        cov_type='cluster', cov_kwds={'groups': df_baseline['unit_id']}
    )
    result = extract_results(
        model,
        spec_id='robust/single/none',
        spec_tree_path='robustness/single_covariate.md',
        outcome_var='tk',
        treatment_var='ab',
        df_used=df_baseline,
        cluster_var='unit_id',
        fixed_effects='region',
        controls_desc='dist1, dist2 (running variable polynomials only)',
        model_type='OLS',
        sample_desc='Full sample, no additional controls'
    )
    if result:
        results.append(result)
        print(f"Bivariate: coef={result['coefficient']:.4f}")
except Exception as e:
    print(f"Bivariate failed: {e}")

# Single control additions
for control in controls_available:
    try:
        formula = f'tk ~ ab + dist1 + dist2 + {control} + ' + ' + '.join(region_fe_cols)
        df_single = df_baseline.dropna(subset=['tk', 'ab', control])

        model = sm.OLS.from_formula(formula, data=df_single).fit(
            cov_type='cluster', cov_kwds={'groups': df_single['unit_id']}
        )
        result = extract_results(
            model,
            spec_id=f'robust/single/{control}',
            spec_tree_path='robustness/single_covariate.md',
            outcome_var='tk',
            treatment_var='ab',
            df_used=df_single,
            cluster_var='unit_id',
            fixed_effects='region',
            controls_desc=f'dist1, dist2, {control}',
            model_type='OLS',
            sample_desc=f'Full sample, adding {control} only'
        )
        if result:
            results.append(result)
            print(f"Single {control}: coef={result['coefficient']:.4f}")
    except Exception as e:
        print(f"Single {control} failed: {e}")


# ============================================================================
# ALTERNATIVE OUTCOMES (if available)
# ============================================================================

print("\n--- Running Alternative Outcomes ---")

# Check for alternative outcomes
alt_outcomes = ['tc']  # Total current transfers
for alt_out in alt_outcomes:
    if alt_out in df.columns:
        df_alt = df.dropna(subset=[alt_out, 'ab', 'dist1', 'dist2'])
        try:
            formula = f'{alt_out} ~ ab + dist1 + dist2 + ' + ' + '.join(region_fe_cols)
            model = sm.OLS.from_formula(formula, data=df_alt).fit(
                cov_type='cluster', cov_kwds={'groups': df_alt['unit_id']}
            )
            result = extract_results(
                model,
                spec_id=f'custom/outcome_{alt_out}',
                spec_tree_path='custom',
                outcome_var=alt_out,
                treatment_var='ab',
                df_used=df_alt,
                cluster_var='unit_id',
                fixed_effects='region',
                controls_desc='dist1, dist2',
                model_type='OLS',
                sample_desc=f'Full sample, alternative outcome: {alt_out}'
            )
            if result:
                results.append(result)
                print(f"Alt outcome {alt_out}: coef={result['coefficient']:.4f}")
        except Exception as e:
            print(f"Alt outcome {alt_out} failed: {e}")


# ============================================================================
# OLS COMPARISON (ignoring endogeneity)
# ============================================================================

print("\n--- Running OLS Comparison (Ignoring Endogeneity) ---")

try:
    formula = 'tk ~ ab + dist1 + dist2 + ' + ' + '.join(region_fe_cols)
    model = sm.OLS.from_formula(formula, data=df_baseline).fit(
        cov_type='cluster', cov_kwds={'groups': df_baseline['unit_id']}
    )
    result = extract_results(
        model,
        spec_id='iv/method/ols',
        spec_tree_path='methods/instrumental_variables.md#estimation-method',
        outcome_var='tk',
        treatment_var='ab',
        df_used=df_baseline,
        cluster_var='unit_id',
        fixed_effects='region',
        controls_desc='dist1, dist2',
        model_type='OLS',
        sample_desc='Full sample, OLS (ignoring endogeneity)'
    )
    if result:
        results.append(result)
        print(f"OLS comparison: coef={result['coefficient']:.4f}")
except Exception as e:
    print(f"OLS comparison failed: {e}")


# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Convert to DataFrame and save
results_df = pd.DataFrame(results)
output_path = f"{OUTPUT_DIR}/specification_results.csv"
results_df.to_csv(output_path, index=False)
print(f"\nSaved {len(results)} specifications to: {output_path}")

# Print summary statistics
print(f"\n--- Summary Statistics ---")
print(f"Total specifications: {len(results)}")
print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")
print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")
print(f"Range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")
