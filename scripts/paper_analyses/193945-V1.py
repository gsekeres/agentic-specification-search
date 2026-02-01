"""
Specification Search Analysis for Paper 193945-V1
"Location, Location, Location" by Card, Rothstein, and Yi

This paper examines urban wage premiums using ACS data at the commuting zone (CZ) level.
Main hypothesis: City size is positively associated with wages (the urban wage premium).
Key finding: Larger cities have higher wages, partly due to composition (skill sorting)
and partly due to CZ-specific wage premiums.

Method Classification: Cross-sectional OLS
- Unit of observation: Commuting zones (691 CZs)
- Outcome variables: log wages, CZ effects from wage models
- Treatment variable: log size (log of weighted count of workers/adults)
- Key controls: fraction college-educated, skill measures

"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.quantile_regression import QuantReg
import json
import warnings
warnings.filterwarnings('ignore')

# Paths
PACKAGE_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/193945-V1"
OUTPUT_DIR = PACKAGE_DIR

# Paper metadata
PAPER_ID = "193945-V1"
JOURNAL = "AEJ: Applied"
PAPER_TITLE = "Location, Location, Location"

def load_data():
    """Load the CZ-level data from Stata files."""

    # Load the main effects file
    czeffects = pd.read_stata(f"{PACKAGE_DIR}/portion1_acs/tocensus/czeffects.dta")

    # Load the post-disclosure analysis file which has more complete data
    atab3 = pd.read_stata(f"{PACKAGE_DIR}/portion3_postdisclosure/results/atab3_full.dta")

    # Merge the datasets
    df = atab3.merge(czeffects[['cz', 'logwage', 'cz_effects_m2', 'cz_effects_m3']],
                     on='cz', how='left', suffixes=('', '_czeffects'))

    # Clean and prepare
    df = df.dropna(subset=['cz_effects_m1', 'lnsize', 'frachighed', 'wcount'])

    print(f"Loaded {len(df)} commuting zones")
    print(f"Variables: {list(df.columns)}")

    return df

def run_regression(df, formula, weights=None, vcov='HC1', spec_id='',
                   spec_tree_path='', treatment_var='lnsize', outcome_var='cz_effects_m1'):
    """Run a regression and extract results in standard format."""

    try:
        if weights is not None:
            model = smf.wls(formula, data=df, weights=df[weights]).fit(cov_type=vcov)
        else:
            model = smf.ols(formula, data=df).fit(cov_type=vcov)

        # Get treatment coefficient
        if treatment_var in model.params:
            coef = model.params[treatment_var]
            se = model.bse[treatment_var]
            tstat = model.tvalues[treatment_var]
            pval = model.pvalues[treatment_var]
            ci = model.conf_int().loc[treatment_var]
        else:
            coef = se = tstat = pval = np.nan
            ci = [np.nan, np.nan]

        # Build coefficient vector
        coef_vector = {
            "treatment": {
                "var": treatment_var,
                "coef": float(coef) if not pd.isna(coef) else None,
                "se": float(se) if not pd.isna(se) else None,
                "pval": float(pval) if not pd.isna(pval) else None
            },
            "controls": [],
            "fixed_effects": [],
            "diagnostics": {
                "f_stat": float(model.fvalue) if hasattr(model, 'fvalue') and not pd.isna(model.fvalue) else None,
                "f_pval": float(model.f_pvalue) if hasattr(model, 'f_pvalue') and not pd.isna(model.f_pvalue) else None
            }
        }

        # Add other coefficients
        for var in model.params.index:
            if var not in [treatment_var, 'Intercept']:
                coef_vector["controls"].append({
                    "var": var,
                    "coef": float(model.params[var]),
                    "se": float(model.bse[var]),
                    "pval": float(model.pvalues[var])
                })

        result = {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': float(coef) if not pd.isna(coef) else None,
            'std_error': float(se) if not pd.isna(se) else None,
            't_stat': float(tstat) if not pd.isna(tstat) else None,
            'p_value': float(pval) if not pd.isna(pval) else None,
            'ci_lower': float(ci[0]) if not pd.isna(ci[0]) else None,
            'ci_upper': float(ci[1]) if not pd.isna(ci[1]) else None,
            'n_obs': int(model.nobs),
            'r_squared': float(model.rsquared),
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': f"{int(model.nobs)} commuting zones",
            'fixed_effects': 'none',
            'controls_desc': ', '.join([c['var'] for c in coef_vector['controls']]) if coef_vector['controls'] else 'none',
            'cluster_var': 'none',
            'model_type': 'WLS' if weights else 'OLS',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }

        return result

    except Exception as e:
        print(f"Error in {spec_id}: {str(e)}")
        return None

def run_quantile_regression(df, formula, q=0.5, spec_id='', spec_tree_path='',
                            treatment_var='lnsize', outcome_var='cz_effects_m1'):
    """Run quantile regression."""
    try:
        # Parse formula
        parts = formula.split('~')
        y_var = parts[0].strip()
        x_vars = [v.strip() for v in parts[1].split('+')]

        y = df[y_var].values
        X = sm.add_constant(df[x_vars].values)

        model = QuantReg(y, X).fit(q=q)

        # Treatment is first variable after constant
        coef_idx = 1  # lnsize should be first
        coef = model.params[coef_idx]
        se = model.bse[coef_idx]
        tstat = model.tvalues[coef_idx]
        pval = model.pvalues[coef_idx]
        ci = model.conf_int()[coef_idx]

        coef_vector = {
            "treatment": {"var": treatment_var, "coef": float(coef), "se": float(se), "pval": float(pval)},
            "controls": [],
            "fixed_effects": [],
            "diagnostics": {"quantile": q}
        }

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
            't_stat': float(tstat),
            'p_value': float(pval),
            'ci_lower': float(ci[0]),
            'ci_upper': float(ci[1]),
            'n_obs': int(model.nobs),
            'r_squared': float(model.prsquared) if hasattr(model, 'prsquared') else None,
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': f"{int(model.nobs)} commuting zones",
            'fixed_effects': 'none',
            'controls_desc': 'none',
            'cluster_var': 'none',
            'model_type': f'QuantReg (q={q})',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
    except Exception as e:
        print(f"Error in quantile {spec_id}: {str(e)}")
        return None

def main():
    """Run all specifications."""

    print("=" * 60)
    print(f"Specification Search for {PAPER_ID}")
    print(f"Paper: {PAPER_TITLE}")
    print("=" * 60)

    # Load data
    df = load_data()

    results = []

    # ============================================================
    # BASELINE SPECIFICATIONS
    # These replicate the paper's main results from Appendix Table 1
    # ============================================================

    print("\n--- Baseline Specifications ---")

    # Baseline 1: Raw log wage on log size (App Table 1, Row 1, Col 1)
    result = run_regression(
        df, 'logwage ~ lnsize', weights='wcount',
        spec_id='baseline',
        spec_tree_path='methods/cross_sectional_ols.md#baseline',
        outcome_var='logwage'
    )
    if result:
        results.append(result)
        print(f"Baseline (logwage ~ lnsize): coef={result['coefficient']:.5f}, se={result['std_error']:.5f}, p={result['p_value']:.4f}")

    # Baseline 2: CZ effects (model 1) on log size - simplest wage premium measure
    result = run_regression(
        df, 'cz_effects_m1 ~ lnsize', weights='wcount',
        spec_id='baseline_cz_m1',
        spec_tree_path='methods/cross_sectional_ols.md#baseline',
        outcome_var='cz_effects_m1'
    )
    if result:
        results.append(result)
        print(f"Baseline (cz_m1 ~ lnsize): coef={result['coefficient']:.5f}, se={result['std_error']:.5f}")

    # Baseline 3: CZ effects (model 3 with industry FE) on log size (App Table 1, Row 1, Col 3)
    result = run_regression(
        df, 'cz_effects_m3 ~ lnsize', weights='wcount',
        spec_id='baseline_cz_m3',
        spec_tree_path='methods/cross_sectional_ols.md#baseline',
        outcome_var='cz_effects_m3'
    )
    if result:
        results.append(result)
        print(f"Baseline (cz_m3 ~ lnsize): coef={result['coefficient']:.5f}, se={result['std_error']:.5f}")

    # ============================================================
    # METHOD-SPECIFIC VARIATIONS (Cross-sectional OLS)
    # ============================================================

    print("\n--- Method Variations ---")

    # OLS/method/ols - unweighted
    result = run_regression(
        df, 'cz_effects_m1 ~ lnsize', weights=None,
        spec_id='ols/method/ols',
        spec_tree_path='methods/cross_sectional_ols.md#estimation-method',
        outcome_var='cz_effects_m1'
    )
    if result:
        results.append(result)
        print(f"OLS unweighted: coef={result['coefficient']:.5f}")

    # OLS/method/wls - weighted (baseline uses this)
    result = run_regression(
        df, 'cz_effects_m1 ~ lnsize', weights='wcount',
        spec_id='ols/method/wls',
        spec_tree_path='methods/cross_sectional_ols.md#estimation-method',
        outcome_var='cz_effects_m1'
    )
    if result:
        results.append(result)
        print(f"WLS weighted: coef={result['coefficient']:.5f}")

    # Quantile regressions
    for q, label in [(0.25, '25'), (0.5, 'median'), (0.75, '75')]:
        result = run_quantile_regression(
            df, 'cz_effects_m1 ~ lnsize', q=q,
            spec_id=f'ols/method/quantile_{label}',
            spec_tree_path='methods/cross_sectional_ols.md#estimation-method',
            outcome_var='cz_effects_m1'
        )
        if result:
            results.append(result)
            print(f"Quantile {q}: coef={result['coefficient']:.5f}")

    # ============================================================
    # CONTROL VARIATIONS
    # ============================================================

    print("\n--- Control Variations ---")

    # Controls/none - bivariate (already done above)
    result = run_regression(
        df, 'cz_effects_m1 ~ lnsize', weights='wcount',
        spec_id='ols/controls/none',
        spec_tree_path='methods/cross_sectional_ols.md#control-sets',
        outcome_var='cz_effects_m1'
    )
    if result:
        results.append(result)

    # Controls/full - with fraction college educated
    result = run_regression(
        df, 'cz_effects_m1 ~ lnsize + frachighed', weights='wcount',
        spec_id='ols/controls/full',
        spec_tree_path='methods/cross_sectional_ols.md#control-sets',
        outcome_var='cz_effects_m1'
    )
    if result:
        results.append(result)
        print(f"With frachighed control: coef={result['coefficient']:.5f}")

    # ============================================================
    # FUNCTIONAL FORM VARIATIONS
    # ============================================================

    print("\n--- Functional Form Variations ---")

    # Level outcome (raw wage)
    result = run_regression(
        df, 'logwage ~ lnsize', weights='wcount',
        spec_id='ols/form/log_dep',
        spec_tree_path='methods/cross_sectional_ols.md#functional-form',
        outcome_var='logwage'
    )
    if result:
        results.append(result)

    # Quadratic in size
    df['lnsize_sq'] = df['lnsize'] ** 2
    result = run_regression(
        df, 'cz_effects_m1 ~ lnsize + lnsize_sq', weights='wcount',
        spec_id='ols/form/quadratic',
        spec_tree_path='methods/cross_sectional_ols.md#functional-form',
        outcome_var='cz_effects_m1'
    )
    if result:
        results.append(result)
        print(f"Quadratic: coef={result['coefficient']:.5f}")

    # ============================================================
    # STANDARD ERROR VARIATIONS
    # ============================================================

    print("\n--- Standard Error Variations ---")

    # Classical SE
    result = run_regression(
        df, 'cz_effects_m1 ~ lnsize', weights='wcount', vcov='nonrobust',
        spec_id='ols/se/classical',
        spec_tree_path='methods/cross_sectional_ols.md#standard-errors',
        outcome_var='cz_effects_m1'
    )
    if result:
        results.append(result)
        print(f"Classical SE: se={result['std_error']:.5f}")

    # HC1 robust (default)
    result = run_regression(
        df, 'cz_effects_m1 ~ lnsize', weights='wcount', vcov='HC1',
        spec_id='ols/se/robust',
        spec_tree_path='methods/cross_sectional_ols.md#standard-errors',
        outcome_var='cz_effects_m1'
    )
    if result:
        results.append(result)
        print(f"HC1 robust SE: se={result['std_error']:.5f}")

    # HC3 robust
    result = run_regression(
        df, 'cz_effects_m1 ~ lnsize', weights='wcount', vcov='HC3',
        spec_id='ols/se/hc3',
        spec_tree_path='methods/cross_sectional_ols.md#standard-errors',
        outcome_var='cz_effects_m1'
    )
    if result:
        results.append(result)
        print(f"HC3 robust SE: se={result['std_error']:.5f}")

    # ============================================================
    # SAMPLE RESTRICTIONS
    # ============================================================

    print("\n--- Sample Restrictions ---")

    # Trim 1% outliers on outcome
    q_low = df['cz_effects_m1'].quantile(0.01)
    q_high = df['cz_effects_m1'].quantile(0.99)
    df_trim = df[(df['cz_effects_m1'] > q_low) & (df['cz_effects_m1'] < q_high)]

    result = run_regression(
        df_trim, 'cz_effects_m1 ~ lnsize', weights='wcount',
        spec_id='robust/sample/trim_1pct',
        spec_tree_path='robustness/sample_restrictions.md#outlier-handling',
        outcome_var='cz_effects_m1'
    )
    if result:
        results.append(result)
        print(f"Trim 1%: coef={result['coefficient']:.5f}, n={result['n_obs']}")

    # Trim 5% outliers
    q_low = df['cz_effects_m1'].quantile(0.05)
    q_high = df['cz_effects_m1'].quantile(0.95)
    df_trim5 = df[(df['cz_effects_m1'] > q_low) & (df['cz_effects_m1'] < q_high)]

    result = run_regression(
        df_trim5, 'cz_effects_m1 ~ lnsize', weights='wcount',
        spec_id='robust/sample/trim_5pct',
        spec_tree_path='robustness/sample_restrictions.md#outlier-handling',
        outcome_var='cz_effects_m1'
    )
    if result:
        results.append(result)
        print(f"Trim 5%: coef={result['coefficient']:.5f}, n={result['n_obs']}")

    # Large CZs only (top 50%)
    median_size = df['wcount'].median()
    df_large = df[df['wcount'] >= median_size]

    result = run_regression(
        df_large, 'cz_effects_m1 ~ lnsize', weights='wcount',
        spec_id='robust/sample/large_czs',
        spec_tree_path='robustness/sample_restrictions.md#geographic-unit-restrictions',
        outcome_var='cz_effects_m1'
    )
    if result:
        results.append(result)
        print(f"Large CZs only: coef={result['coefficient']:.5f}, n={result['n_obs']}")

    # Small CZs only (bottom 50%)
    df_small = df[df['wcount'] < median_size]

    result = run_regression(
        df_small, 'cz_effects_m1 ~ lnsize', weights='wcount',
        spec_id='robust/sample/small_czs',
        spec_tree_path='robustness/sample_restrictions.md#geographic-unit-restrictions',
        outcome_var='cz_effects_m1'
    )
    if result:
        results.append(result)
        print(f"Small CZs only: coef={result['coefficient']:.5f}, n={result['n_obs']}")

    # Drop largest CZ (Los Angeles)
    df_no_la = df[df['wcount'] < df['wcount'].max()]

    result = run_regression(
        df_no_la, 'cz_effects_m1 ~ lnsize', weights='wcount',
        spec_id='robust/sample/drop_largest',
        spec_tree_path='robustness/sample_restrictions.md#geographic-unit-restrictions',
        outcome_var='cz_effects_m1'
    )
    if result:
        results.append(result)
        print(f"Drop largest CZ: coef={result['coefficient']:.5f}")

    # Drop smallest CZs (bottom 10%)
    p10 = df['wcount'].quantile(0.10)
    df_no_smallest = df[df['wcount'] >= p10]

    result = run_regression(
        df_no_smallest, 'cz_effects_m1 ~ lnsize', weights='wcount',
        spec_id='robust/sample/drop_smallest_10pct',
        spec_tree_path='robustness/sample_restrictions.md#geographic-unit-restrictions',
        outcome_var='cz_effects_m1'
    )
    if result:
        results.append(result)
        print(f"Drop smallest 10%: coef={result['coefficient']:.5f}, n={result['n_obs']}")

    # ============================================================
    # SINGLE COVARIATE ANALYSIS
    # ============================================================

    print("\n--- Single Covariate Analysis ---")

    # Bivariate
    result = run_regression(
        df, 'cz_effects_m1 ~ lnsize', weights='wcount',
        spec_id='robust/single/none',
        spec_tree_path='robustness/single_covariate.md',
        outcome_var='cz_effects_m1'
    )
    if result:
        results.append(result)
        bivariate_coef = result['coefficient']
        print(f"Bivariate: coef={result['coefficient']:.5f}")

    # With frachighed
    result = run_regression(
        df, 'cz_effects_m1 ~ lnsize + frachighed', weights='wcount',
        spec_id='robust/single/frachighed',
        spec_tree_path='robustness/single_covariate.md',
        outcome_var='cz_effects_m1'
    )
    if result:
        results.append(result)
        print(f"+ frachighed: coef={result['coefficient']:.5f}")

    # ============================================================
    # ALTERNATIVE OUTCOME MEASURES
    # ============================================================

    print("\n--- Alternative Outcomes ---")

    # CZ effects from model 2 (basic Mincer + extra controls)
    result = run_regression(
        df, 'cz_effects_m2 ~ lnsize', weights='wcount',
        spec_id='custom/outcome_m2',
        spec_tree_path='custom',
        outcome_var='cz_effects_m2'
    )
    if result:
        results.append(result)
        print(f"CZ effects model 2: coef={result['coefficient']:.5f}")

    # CZ effects from model 3 (with industry FE)
    result = run_regression(
        df, 'cz_effects_m3 ~ lnsize', weights='wcount',
        spec_id='custom/outcome_m3',
        spec_tree_path='custom',
        outcome_var='cz_effects_m3'
    )
    if result:
        results.append(result)
        print(f"CZ effects model 3: coef={result['coefficient']:.5f}")

    # ACS-predicted LEHD premium
    result = run_regression(
        df, 'psi_acspredict ~ lnsize', weights='wcount',
        spec_id='custom/outcome_psi_predict',
        spec_tree_path='custom',
        outcome_var='psi_acspredict'
    )
    if result:
        results.append(result)
        print(f"Predicted LEHD premium: coef={result['coefficient']:.5f}")

    # ============================================================
    # CLUSTERING VARIATIONS
    # ============================================================

    print("\n--- Clustering Variations ---")
    # Note: With CZ-level data, standard clustering at unit level is not applicable
    # We show robust SE variations instead

    result = run_regression(
        df, 'cz_effects_m1 ~ lnsize', weights='wcount', vcov='HC0',
        spec_id='robust/cluster/none',
        spec_tree_path='robustness/clustering_variations.md#single-level-clustering',
        outcome_var='cz_effects_m1'
    )
    if result:
        results.append(result)
        print(f"HC0 (no df correction): se={result['std_error']:.5f}")

    result = run_regression(
        df, 'cz_effects_m1 ~ lnsize', weights='wcount', vcov='HC2',
        spec_id='robust/se/hc2',
        spec_tree_path='robustness/clustering_variations.md#alternative-se-methods',
        outcome_var='cz_effects_m1'
    )
    if result:
        results.append(result)
        print(f"HC2: se={result['std_error']:.5f}")

    # ============================================================
    # FUNCTIONAL FORM ROBUSTNESS
    # ============================================================

    print("\n--- Functional Form Robustness ---")

    # Standardized outcome
    df['cz_effects_m1_std'] = (df['cz_effects_m1'] - df['cz_effects_m1'].mean()) / df['cz_effects_m1'].std()
    df['lnsize_std'] = (df['lnsize'] - df['lnsize'].mean()) / df['lnsize'].std()

    result = run_regression(
        df, 'cz_effects_m1_std ~ lnsize_std', weights='wcount',
        spec_id='robust/form/y_standardized',
        spec_tree_path='robustness/functional_form.md#outcome-variable-transformations',
        treatment_var='lnsize_std',
        outcome_var='cz_effects_m1_std'
    )
    if result:
        results.append(result)
        print(f"Standardized: coef={result['coefficient']:.5f}")

    # ============================================================
    # SAVE RESULTS
    # ============================================================

    print("\n" + "=" * 60)
    print(f"Total specifications run: {len(results)}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    output_path = f"{OUTPUT_DIR}/specification_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

    # Summary statistics
    print("\n--- Summary Statistics ---")
    print(f"Total specifications: {len(results_df)}")
    print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
    print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
    print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")
    print(f"Coefficient range: [{results_df['coefficient'].min():.5f}, {results_df['coefficient'].max():.5f}]")
    print(f"Median coefficient: {results_df['coefficient'].median():.5f}")
    print(f"Mean coefficient: {results_df['coefficient'].mean():.5f}")

    return results_df

if __name__ == "__main__":
    results_df = main()
