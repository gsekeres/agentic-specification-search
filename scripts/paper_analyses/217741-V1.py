"""
Specification Search Analysis for Paper 217741-V1
"AI and Women's Employment in Europe"
AEA Papers and Proceedings (2025)

This script replicates the paper's main findings and runs a systematic
specification search following the specification tree framework.

Paper Summary:
- Research Question: How does AI exposure affect female employment shares in Europe?
- Method: Cross-sectional OLS with WLS (weighted by labor supply)
- Main Treatment Variables: PCT_aiW (Webb AI exposure), PCT_aiF (Felten AI exposure)
- Outcome: DHSshEmployee - percent change in female employment shares (DHS formula)
- Sample: 16 European countries, 2011-2019, using 2011 cross-section for main results
- Controls: Sector FE (sec1-sec5), Country FE (cty2-cty16)
- Clustering: Country x Sector
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import WLS, OLS
from scipy import stats
import json
import warnings
import os

warnings.filterwarnings('ignore')

# Paths
DATA_PATH = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/217741-V1/Pool16_AI.dta'
OUTPUT_DIR = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/217741-V1'

# Results storage
results = []

def prepare_data():
    """Prepare the dataset following the Stata code."""
    df = pd.read_stata(DATA_PATH)

    # Create country dummies
    df = df.sort_values('country')
    country_order = sorted(df['country'].unique())
    for i, c in enumerate(country_order, 1):
        df[f'cty{i}'] = (df['country'] == c).astype(int)

    df['cty'] = df['country'].map({c: i+1 for i, c in enumerate(country_order)})

    # Create AI x country interactions
    for i in range(1, 17):
        df[f'PCT_aiW_cty{i}'] = df['PCT_aiW'] * df[f'cty{i}']
        df[f'PCT_aiF_cty{i}'] = df['PCT_aiF'] * df[f'cty{i}']

    # Sort
    df = df.sort_values(['country', 'id', 'year'])

    # Total female employment by country year
    df['emplT_F'] = df.groupby(['country', 'year'])['empl_female'].transform('sum')

    # Female employment share
    df['shFE2'] = df['empl_female'] / df['emplT']

    # Filter to ISCO 08
    df_08 = df[df['isco'] == 8.0].copy()

    # Compute start and end years
    start_end = df_08.groupby(['country', 'id']).agg(
        startyear08=('year', 'min'),
        endyear08=('year', 'max')
    ).reset_index()

    df_08 = df_08.merge(start_end, on=['country', 'id'], how='left')
    df_08['diff08'] = df_08['endyear08'] - df_08['startyear08']

    # Get shFE2 at start and end year
    start_df = df_08[df_08['year'] == df_08['startyear08']][['country', 'id', 'shFE2']].drop_duplicates()
    start_df.columns = ['country', 'id', 'shFE21']
    end_df = df_08[df_08['year'] == df_08['endyear08']][['country', 'id', 'shFE2']].drop_duplicates()
    end_df.columns = ['country', 'id', 'shFE22']

    df_08 = df_08.merge(start_df, on=['country', 'id'], how='left')
    df_08 = df_08.merge(end_df, on=['country', 'id'], how='left')

    # Compute DHS change
    df_08['MshFE2'] = (df_08['shFE22'] + df_08['shFE21']) / 2
    df_08.loc[df_08['diff08'] == 0, 'MshFE2'] = np.nan
    df_08['DHSshFE2'] = (df_08['shFE22'] - df_08['shFE21']) / df_08['MshFE2'] * 100

    # Filter to year > 2010
    df_08 = df_08[df_08['year'] > 2010].copy()

    # Winsorize at 1% and 99% by country
    def winsorize_by_country(group):
        q1 = group['DHSshFE2'].quantile(0.01)
        q99 = group['DHSshFE2'].quantile(0.99)
        return group[(group['DHSshFE2'] > q1) & (group['DHSshFE2'] < q99)]

    df_clean = df_08.groupby('cty', group_keys=False).apply(winsorize_by_country)

    # Create id for sector-occupation
    df_clean['idp'] = pd.factorize(df_clean['sector'].astype(str) + '_' + df_clean['occupdigit'].astype(str))[0]

    # Compute shActAv
    df_clean = df_clean.rename(columns={'shActive_so': 'shAct'})

    for i in range(1, 17):
        mask = df_clean['cty'] == i
        means = df_clean[mask].groupby('idp')['shAct'].mean()
        df_clean.loc[mask, f'shActAveg_cty{i}'] = df_clean[mask]['idp'].map(means)

    df_clean['shActAv'] = np.nan
    for i in range(1, 17):
        mask = df_clean['cty'] == i
        df_clean.loc[mask, 'shActAv'] = df_clean.loc[mask, f'shActAveg_cty{i}']

    # Create double cluster variable
    df_clean['double_cluster'] = pd.factorize(df_clean['country'].astype(str) + '_' + df_clean['sector'].astype(str))[0]

    # Create DHSshEmployee
    df_clean['DHSshEmployee'] = df_clean['DHSshFE2']

    # Country classifications
    df_clean['clas1'] = 2
    df_clean.loc[df_clean['country'].isin(['PT', 'ES', 'IT', 'GR', 'FR', 'IE', 'BE', 'AT']), 'clas1'] = 1

    df_clean['clas3'] = 2
    df_clean.loc[df_clean['country'].isin(['ES', 'NL', 'IT', 'LT', 'BE', 'FI', 'AT', 'DE', 'UK', 'LU']), 'clas3'] = 1

    df_clean['clas4'] = 2
    df_clean.loc[df_clean['country'].isin(['IT', 'GR', 'LU', 'IE', 'ES', 'UK', 'BE']), 'clas4'] = 1

    df_clean['clas5'] = 2
    df_clean.loc[df_clean['country'].isin(['IT', 'GR', 'LU', 'BE', 'IE', 'ES']), 'clas5'] = 1

    df_clean['occu1'] = df_clean['occup1digit']

    return df_clean


def run_regression(df, treatment_var, controls, weights=None, cluster_var=None,
                   spec_id=None, spec_tree_path=None, description=None, sample_filter=None):
    """
    Run a WLS/OLS regression and return structured results.
    """
    # Apply sample filter if provided
    if sample_filter is not None:
        df = df[sample_filter].copy()

    # Prepare the data
    vars_needed = ['DHSshEmployee', treatment_var] + controls
    if weights:
        vars_needed.append(weights)
    if cluster_var:
        vars_needed.append(cluster_var)

    df_reg = df.dropna(subset=vars_needed).copy()

    if len(df_reg) < 10:
        return None

    # Build design matrix
    y = df_reg['DHSshEmployee'].values
    X = df_reg[[treatment_var] + controls].values
    X = np.column_stack([X, np.ones(len(df_reg))])

    col_names = [treatment_var] + controls + ['const']

    # Run regression
    if weights:
        w = df_reg[weights].values
        model = WLS(y, X, weights=w)
    else:
        model = OLS(y, X)

    if cluster_var:
        clusters = df_reg[cluster_var].values
        result = model.fit(cov_type='cluster', cov_kwds={'groups': clusters})
    else:
        result = model.fit(cov_type='HC1')  # robust SE

    # Extract treatment coefficient
    treat_idx = 0
    treat_coef = result.params[treat_idx]
    treat_se = result.bse[treat_idx]
    treat_tstat = result.tvalues[treat_idx]
    treat_pval = result.pvalues[treat_idx]
    ci = result.conf_int()[treat_idx]

    # Build coefficient vector
    coef_vector = {
        'treatment': {
            'var': treatment_var,
            'coef': float(treat_coef),
            'se': float(treat_se),
            'tstat': float(treat_tstat),
            'pval': float(treat_pval),
            'ci_lower': float(ci[0]),
            'ci_upper': float(ci[1])
        },
        'controls': [],
        'diagnostics': {
            'n_obs': int(result.nobs),
            'r_squared': float(result.rsquared),
            'f_stat': float(result.fvalue) if hasattr(result, 'fvalue') and result.fvalue is not None else np.nan,
            'f_pval': float(result.f_pvalue) if hasattr(result, 'f_pvalue') and result.f_pvalue is not None else np.nan
        }
    }

    # Add control coefficients
    for i, var in enumerate(controls):
        coef_vector['controls'].append({
            'var': var,
            'coef': float(result.params[i + 1]),
            'se': float(result.bse[i + 1]),
            'pval': float(result.pvalues[i + 1])
        })

    # Build result record
    result_record = {
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'description': description,
        'treatment_var': treatment_var,
        'coefficient': treat_coef,
        'std_error': treat_se,
        't_statistic': treat_tstat,
        'p_value': treat_pval,
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'n_obs': int(result.nobs),
        'r_squared': result.rsquared,
        'controls': ', '.join(controls) if controls else 'none',
        'weights': weights,
        'cluster_var': cluster_var,
        'coefficient_vector_json': json.dumps(coef_vector)
    }

    return result_record


def main():
    print("="*70)
    print("Specification Search for Paper 217741-V1")
    print("AI and Women's Employment in Europe")
    print("="*70)

    # Prepare data
    print("\nPreparing data...")
    df_full = prepare_data()

    # Use 2011 cross-section as main sample (paper's main specification)
    df = df_full[df_full['year'] == 2011].copy()
    print(f"Sample size (2011 cross-section): {len(df)}")

    # Define baseline controls
    sector_cols = ['sec1', 'sec2', 'sec3', 'sec4', 'sec5']
    country_cols = [f'cty{i}' for i in range(2, 17)]
    baseline_controls = sector_cols + country_cols

    global results

    # ===================================================================
    # BASELINE SPECIFICATIONS
    # ===================================================================
    print("\n" + "-"*50)
    print("Running baseline specifications...")
    print("-"*50)

    # Baseline with PCT_aiW
    res = run_regression(
        df, 'PCT_aiW', baseline_controls,
        weights='shActAv', cluster_var='double_cluster',
        spec_id='baseline/aiW',
        spec_tree_path='methods/cross_sectional_ols.md#baseline',
        description='Baseline: Webb AI exposure on female employment change'
    )
    if res:
        results.append(res)
        print(f"Baseline (AIW): coef={res['coefficient']:.4f}, se={res['std_error']:.4f}, p={res['p_value']:.4f}, n={res['n_obs']}")

    # Baseline with PCT_aiF
    res = run_regression(
        df, 'PCT_aiF', baseline_controls,
        weights='shActAv', cluster_var='double_cluster',
        spec_id='baseline/aiF',
        spec_tree_path='methods/cross_sectional_ols.md#baseline',
        description='Baseline: Felten AI exposure on female employment change'
    )
    if res:
        results.append(res)
        print(f"Baseline (AIF): coef={res['coefficient']:.4f}, se={res['std_error']:.4f}, p={res['p_value']:.4f}, n={res['n_obs']}")

    # ===================================================================
    # OLS METHOD VARIATIONS
    # ===================================================================
    print("\n" + "-"*50)
    print("Running OLS method variations...")
    print("-"*50)

    # No weights (OLS instead of WLS)
    for treat, label in [('PCT_aiW', 'aiW'), ('PCT_aiF', 'aiF')]:
        res = run_regression(
            df, treat, baseline_controls,
            weights=None, cluster_var='double_cluster',
            spec_id=f'ols/method/ols_{label}',
            spec_tree_path='methods/cross_sectional_ols.md#estimation-method',
            description=f'OLS (unweighted): {label}'
        )
        if res:
            results.append(res)
            print(f"OLS unweighted ({label}): coef={res['coefficient']:.4f}, p={res['p_value']:.4f}")

    # ===================================================================
    # STANDARD ERROR VARIATIONS
    # ===================================================================
    print("\n" + "-"*50)
    print("Running SE variations...")
    print("-"*50)

    for treat, label in [('PCT_aiW', 'aiW'), ('PCT_aiF', 'aiF')]:
        # Robust SE (no clustering)
        res = run_regression(
            df, treat, baseline_controls,
            weights='shActAv', cluster_var=None,
            spec_id=f'ols/se/robust_{label}',
            spec_tree_path='robustness/clustering_variations.md#single-level-clustering',
            description=f'Robust SE (HC1), no clustering: {label}'
        )
        if res:
            results.append(res)
            print(f"Robust SE ({label}): coef={res['coefficient']:.4f}, se={res['std_error']:.4f}")

        # Cluster by country only
        df['country_cluster'] = pd.factorize(df['country'])[0]
        res = run_regression(
            df, treat, baseline_controls,
            weights='shActAv', cluster_var='country_cluster',
            spec_id=f'robust/cluster/country_{label}',
            spec_tree_path='robustness/clustering_variations.md#single-level-clustering',
            description=f'Cluster by country only: {label}'
        )
        if res:
            results.append(res)
            print(f"Cluster country ({label}): coef={res['coefficient']:.4f}, se={res['std_error']:.4f}")

        # Cluster by sector only
        df['sector_cluster'] = pd.factorize(df['sector'])[0]
        res = run_regression(
            df, treat, baseline_controls,
            weights='shActAv', cluster_var='sector_cluster',
            spec_id=f'robust/cluster/sector_{label}',
            spec_tree_path='robustness/clustering_variations.md#single-level-clustering',
            description=f'Cluster by sector only: {label}'
        )
        if res:
            results.append(res)
            print(f"Cluster sector ({label}): coef={res['coefficient']:.4f}, se={res['std_error']:.4f}")

    # ===================================================================
    # CONTROL SET VARIATIONS
    # ===================================================================
    print("\n" + "-"*50)
    print("Running control set variations...")
    print("-"*50)

    for treat, label in [('PCT_aiW', 'aiW'), ('PCT_aiF', 'aiF')]:
        # No controls (bivariate)
        res = run_regression(
            df, treat, [],
            weights='shActAv', cluster_var='double_cluster',
            spec_id=f'ols/controls/none_{label}',
            spec_tree_path='methods/cross_sectional_ols.md#control-sets',
            description=f'Bivariate (no controls): {label}'
        )
        if res:
            results.append(res)
            print(f"No controls ({label}): coef={res['coefficient']:.4f}, p={res['p_value']:.4f}")

        # Sector FE only
        res = run_regression(
            df, treat, sector_cols,
            weights='shActAv', cluster_var='double_cluster',
            spec_id=f'ols/controls/sector_only_{label}',
            spec_tree_path='methods/cross_sectional_ols.md#control-sets',
            description=f'Sector FE only: {label}'
        )
        if res:
            results.append(res)
            print(f"Sector FE only ({label}): coef={res['coefficient']:.4f}, p={res['p_value']:.4f}")

        # Country FE only
        res = run_regression(
            df, treat, country_cols,
            weights='shActAv', cluster_var='double_cluster',
            spec_id=f'ols/controls/country_only_{label}',
            spec_tree_path='methods/cross_sectional_ols.md#control-sets',
            description=f'Country FE only: {label}'
        )
        if res:
            results.append(res)
            print(f"Country FE only ({label}): coef={res['coefficient']:.4f}, p={res['p_value']:.4f}")

    # ===================================================================
    # LEAVE-ONE-OUT: DROP EACH SECTOR DUMMY
    # ===================================================================
    print("\n" + "-"*50)
    print("Running leave-one-out on sector controls...")
    print("-"*50)

    for treat, label in [('PCT_aiW', 'aiW'), ('PCT_aiF', 'aiF')]:
        for drop_sec in sector_cols:
            loo_controls = [c for c in baseline_controls if c != drop_sec]
            res = run_regression(
                df, treat, loo_controls,
                weights='shActAv', cluster_var='double_cluster',
                spec_id=f'robust/loo/drop_{drop_sec}_{label}',
                spec_tree_path='robustness/leave_one_out.md',
                description=f'Leave-one-out: drop {drop_sec}: {label}'
            )
            if res:
                results.append(res)
    print(f"Completed {len(sector_cols) * 2} leave-one-out specifications on sectors")

    # ===================================================================
    # LEAVE-ONE-OUT: DROP EACH COUNTRY
    # ===================================================================
    print("\n" + "-"*50)
    print("Running leave-one-out on country controls...")
    print("-"*50)

    for treat, label in [('PCT_aiW', 'aiW'), ('PCT_aiF', 'aiF')]:
        for drop_cty in country_cols:
            loo_controls = [c for c in baseline_controls if c != drop_cty]
            res = run_regression(
                df, treat, loo_controls,
                weights='shActAv', cluster_var='double_cluster',
                spec_id=f'robust/loo/drop_{drop_cty}_{label}',
                spec_tree_path='robustness/leave_one_out.md',
                description=f'Leave-one-out: drop {drop_cty}: {label}'
            )
            if res:
                results.append(res)
    print(f"Completed {len(country_cols) * 2} leave-one-out specifications on countries")

    # ===================================================================
    # SINGLE COVARIATE ANALYSIS
    # ===================================================================
    print("\n" + "-"*50)
    print("Running single covariate analysis...")
    print("-"*50)

    for treat, label in [('PCT_aiW', 'aiW'), ('PCT_aiF', 'aiF')]:
        # Treatment + each sector only
        for sec in sector_cols:
            res = run_regression(
                df, treat, [sec],
                weights='shActAv', cluster_var='double_cluster',
                spec_id=f'robust/single/{sec}_{label}',
                spec_tree_path='robustness/single_covariate.md',
                description=f'Single covariate: {sec}: {label}'
            )
            if res:
                results.append(res)
    print(f"Completed single covariate specifications")

    # ===================================================================
    # SAMPLE RESTRICTIONS: SUBGROUPS BY COUNTRY CLASSIFICATION
    # ===================================================================
    print("\n" + "-"*50)
    print("Running sample restriction specifications (country groups)...")
    print("-"*50)

    classifications = [
        ('clas4', 1, 'low_participation'),
        ('clas4', 2, 'high_participation'),
        ('clas5', 1, 'low_participation_level'),
        ('clas5', 2, 'high_participation_level'),
        ('clas3', 1, 'high_upskilling'),
        ('clas3', 2, 'low_upskilling'),
        ('clas1', 1, 'education_gap'),
        ('clas1', 2, 'close_to_us_edu'),
    ]

    for treat, label in [('PCT_aiW', 'aiW'), ('PCT_aiF', 'aiF')]:
        for clas_var, clas_val, clas_name in classifications:
            sample_filter = df[clas_var] == clas_val
            res = run_regression(
                df, treat, baseline_controls,
                weights='shActAv', cluster_var='double_cluster',
                spec_id=f'robust/sample/{clas_name}_{label}',
                spec_tree_path='robustness/sample_restrictions.md',
                description=f'Subsample: {clas_name}: {label}',
                sample_filter=sample_filter
            )
            if res:
                results.append(res)
                print(f"Subsample {clas_name} ({label}): coef={res['coefficient']:.4f}, n={res['n_obs']}")

    # ===================================================================
    # SAMPLE RESTRICTIONS: LEAVE-ONE-COUNTRY-OUT
    # ===================================================================
    print("\n" + "-"*50)
    print("Running leave-one-country-out specifications...")
    print("-"*50)

    countries = sorted(df['country'].unique())
    for treat, label in [('PCT_aiW', 'aiW'), ('PCT_aiF', 'aiF')]:
        for country in countries:
            sample_filter = df['country'] != country
            res = run_regression(
                df, treat, baseline_controls,
                weights='shActAv', cluster_var='double_cluster',
                spec_id=f'robust/sample/exclude_{country}_{label}',
                spec_tree_path='robustness/sample_restrictions.md',
                description=f'Exclude country {country}: {label}',
                sample_filter=sample_filter
            )
            if res:
                results.append(res)
    print(f"Completed {len(countries) * 2} leave-one-country-out specifications")

    # ===================================================================
    # SAMPLE RESTRICTIONS: LEAVE-ONE-OCCUPATION-OUT (Table A3)
    # ===================================================================
    print("\n" + "-"*50)
    print("Running leave-one-occupation-out specifications (Table A3)...")
    print("-"*50)

    occupations = sorted(df['occu1'].dropna().unique())
    for treat, label in [('PCT_aiW', 'aiW'), ('PCT_aiF', 'aiF')]:
        for occ in occupations:
            if pd.isna(occ):
                continue
            sample_filter = df['occu1'] != occ
            res = run_regression(
                df, treat, baseline_controls,
                weights='shActAv', cluster_var='double_cluster',
                spec_id=f'robust/sample/exclude_occup{int(occ)}_{label}',
                spec_tree_path='robustness/sample_restrictions.md',
                description=f'Exclude occupation {int(occ)}: {label}',
                sample_filter=sample_filter
            )
            if res:
                results.append(res)
    print(f"Completed leave-one-occupation-out specifications")

    # ===================================================================
    # FUNCTIONAL FORM VARIATIONS
    # ===================================================================
    print("\n" + "-"*50)
    print("Running functional form variations...")
    print("-"*50)

    for treat, label in [('PCT_aiW', 'aiW'), ('PCT_aiF', 'aiF')]:
        # Quadratic in treatment
        df['treat_sq'] = df[treat] ** 2
        res = run_regression(
            df, treat, baseline_controls + ['treat_sq'],
            weights='shActAv', cluster_var='double_cluster',
            spec_id=f'ols/form/quadratic_{label}',
            spec_tree_path='methods/cross_sectional_ols.md#functional-form',
            description=f'Quadratic in AI exposure: {label}'
        )
        if res:
            results.append(res)
            print(f"Quadratic ({label}): coef={res['coefficient']:.4f}, p={res['p_value']:.4f}")

    # ===================================================================
    # ALTERNATIVE YEARS (pooled panel)
    # ===================================================================
    print("\n" + "-"*50)
    print("Running alternative time periods...")
    print("-"*50)

    for year in [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]:
        df_year = df_full[df_full['year'] == year].copy()
        if len(df_year) < 100:
            continue
        for treat, label in [('PCT_aiW', 'aiW'), ('PCT_aiF', 'aiF')]:
            res = run_regression(
                df_year, treat, baseline_controls,
                weights='shActAv', cluster_var='double_cluster',
                spec_id=f'robust/sample/year_{year}_{label}',
                spec_tree_path='robustness/sample_restrictions.md',
                description=f'Year {year} cross-section: {label}'
            )
            if res:
                results.append(res)
                print(f"Year {year} ({label}): coef={res['coefficient']:.4f}, n={res['n_obs']}")

    # ===================================================================
    # INTERACTION WITH COUNTRY (heterogeneous effects)
    # ===================================================================
    print("\n" + "-"*50)
    print("Running country-specific effects...")
    print("-"*50)

    # Create country interaction terms for the regression
    # This replicates the coefficients used for Figure A1
    for treat, label in [('PCT_aiW', 'aiW'), ('PCT_aiF', 'aiF')]:
        interact_cols = [f'{treat}_cty{i}' for i in range(1, 17)]
        all_controls = sector_cols + country_cols + interact_cols

        df_int = df.dropna(subset=[treat, 'DHSshEmployee', 'shActAv']).copy()
        for i in range(1, 17):
            df_int[f'{treat}_cty{i}'] = df_int[treat] * df_int[f'cty{i}']

        # Skip if too many missing
        if df_int.dropna(subset=all_controls).shape[0] < 100:
            continue

        res = run_regression(
            df_int, interact_cols[0], sector_cols + country_cols + interact_cols[1:],
            weights='shActAv', cluster_var='double_cluster',
            spec_id=f'ols/interact/country_{label}',
            spec_tree_path='methods/cross_sectional_ols.md#interaction-effects',
            description=f'Country-specific effects: {label}'
        )
        if res:
            results.append(res)
            print(f"Country interactions ({label}): estimated")

    # ===================================================================
    # SAVE RESULTS
    # ===================================================================
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    results_df = pd.DataFrame(results)
    output_file = os.path.join(OUTPUT_DIR, 'specification_results.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved {len(results)} specification results to:")
    print(f"  {output_file}")

    # Summary statistics
    print("\n" + "-"*50)
    print("SUMMARY STATISTICS")
    print("-"*50)

    print(f"\nTotal specifications run: {len(results)}")

    # By treatment variable
    for treat in ['PCT_aiW', 'PCT_aiF']:
        subset = results_df[results_df['treatment_var'] == treat]
        if len(subset) > 0:
            print(f"\n{treat}:")
            print(f"  Specifications: {len(subset)}")
            print(f"  Coefficient range: [{subset['coefficient'].min():.4f}, {subset['coefficient'].max():.4f}]")
            print(f"  Median coefficient: {subset['coefficient'].median():.4f}")
            print(f"  Significant at 5%: {(subset['p_value'] < 0.05).sum()} ({(subset['p_value'] < 0.05).mean()*100:.1f}%)")
            print(f"  Significant at 1%: {(subset['p_value'] < 0.01).sum()} ({(subset['p_value'] < 0.01).mean()*100:.1f}%)")

    return results_df


if __name__ == '__main__':
    results_df = main()
