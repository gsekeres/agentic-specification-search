#!/usr/bin/env python3
"""
Specification Search: 217741-V1
Paper: "AI and Women's Employment in Europe"
AEA P&P 2025

Method: Cross-sectional OLS with sector and country fixed effects
Outcome: Change in female employment shares (DHSshFE2/DHSshEmployee)
Treatment: AI exposure (PCT_aiW from Webb, PCT_aiF from Felten)
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
PAPER_ID = "217741-V1"
JOURNAL = "AER P&P"
PAPER_TITLE = "AI and Women's Employment in Europe"
DATA_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/217741-V1/Pool16_AI.dta"
OUTPUT_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/217741-V1"

# =============================================================================
# DATA PREPARATION - Replicate Stata transformations
# =============================================================================
def prepare_data():
    """Load and prepare data following the Stata do files."""
    df = pd.read_stata(DATA_PATH)

    # Generate country dummies
    df = df.sort_values('country')
    country_dummies = pd.get_dummies(df['country'], prefix='cty', drop_first=False)
    df = pd.concat([df, country_dummies], axis=1)

    # Create country numeric ID
    country_order = ['AT', 'BE', 'DE', 'EE', 'ES', 'FI', 'FR', 'GR', 'IE', 'IT', 'LT', 'LU', 'LV', 'NL', 'PT', 'UK']
    df['cty'] = df['country'].map({c: i+1 for i, c in enumerate(country_order)})

    # Ensure country dummies exist with correct names
    for i, c in enumerate(country_order, 1):
        col_name = f'cty{i}'
        if col_name not in df.columns:
            df[col_name] = (df['country'] == c).astype(int)
        else:
            df[col_name] = df[col_name].astype(int)

    # Create country-specific AI interaction terms
    for ai_var in ['PCT_aiW', 'PCT_aiF']:
        for i in range(1, 17):
            df[f'{ai_var}_cty{i}'] = df[ai_var] * df[f'cty{i}']

    # Sort and prepare for outcome calculation
    df = df.sort_values(['country', 'year', 'sector'])

    # Calculate total employment by country-year
    df['emplT_F'] = df.groupby(['country', 'year'])['empl_female'].transform('sum')
    df['emplT'] = df.groupby(['country', 'year'])['empl'].transform('sum')

    # Female employment shares
    df['shFE2'] = df['empl_female'] / df['emplT']

    # Sort for DHS calculation
    df = df.sort_values(['country', 'id', 'year'])

    # Calculate DHS (Davis-Haltiwanger-Schuh) percent change for ISCO08
    # Following the Stata code
    df_isco08 = df[df['isco'] == 8].copy()

    # Get start and end year by country-id
    start_year = df_isco08.groupby(['country', 'id'])['year'].min().reset_index(name='startyear08')
    end_year = df_isco08.groupby(['country', 'id'])['year'].max().reset_index(name='endyear08')

    df_isco08 = df_isco08.merge(start_year, on=['country', 'id'])
    df_isco08 = df_isco08.merge(end_year, on=['country', 'id'])

    # Get shFE2 at start and end years
    df_start = df_isco08[df_isco08['year'] == df_isco08['startyear08']][['country', 'id', 'shFE2']].copy()
    df_start.columns = ['country', 'id', 'shFE21']

    df_end = df_isco08[df_isco08['year'] == df_isco08['endyear08']][['country', 'id', 'shFE2']].copy()
    df_end.columns = ['country', 'id', 'shFE22']

    df_isco08 = df_isco08.merge(df_start, on=['country', 'id'], how='left')
    df_isco08 = df_isco08.merge(df_end, on=['country', 'id'], how='left')

    # Calculate DHS
    df_isco08['diff08'] = df_isco08['endyear08'] - df_isco08['startyear08']
    df_isco08['DHSshFE2'] = df_isco08['shFE22'] - df_isco08['shFE21']
    df_isco08['MshFE2'] = (df_isco08['shFE22'] + df_isco08['shFE21']) / 2
    df_isco08.loc[df_isco08['diff08'] == 0, 'MshFE2'] = np.nan
    df_isco08['DHSshFE2'] = (df_isco08['DHSshFE2'] / df_isco08['MshFE2']) * 100

    # Merge back
    df = df.merge(df_isco08[['country', 'id', 'year', 'DHSshFE2']].drop_duplicates(),
                  on=['country', 'id', 'year'], how='left')

    # Country classification variables
    # clas1: deviations from US female education
    df['clas1'] = 2
    df.loc[df['country'].isin(['PT', 'ES', 'IT', 'GR', 'FR', 'IE', 'BE', 'AT']), 'clas1'] = 1

    # clas3: relative upskilling
    df['clas3'] = 2
    df.loc[df['country'].isin(['ES', 'NL', 'IT', 'LT', 'BE', 'FI', 'AT', 'DE', 'UK', 'LU']), 'clas3'] = 1

    # clas4: female LFP relative to total in 2011
    df['clas4'] = 2
    df.loc[df['country'].isin(['IT', 'GR', 'LU', 'IE', 'ES', 'UK', 'BE']), 'clas4'] = 1

    # clas5: female LFP levels in 2011
    df['clas5'] = 2
    df.loc[df['country'].isin(['IT', 'GR', 'LU', 'BE', 'IE', 'ES']), 'clas5'] = 1

    # Filter to post-2010
    df = df[df['year'] > 2010].copy()

    # Winsorization at 1% by country (drop if outside 1-99%)
    df = df.sort_values('cty')

    def winsorize_drop(group):
        if 'DHSshFE2' in group.columns and group['DHSshFE2'].notna().any():
            q1 = group['DHSshFE2'].quantile(0.01)
            q99 = group['DHSshFE2'].quantile(0.99)
            return group[(group['DHSshFE2'] > q1) & (group['DHSshFE2'] < q99)]
        return group

    df = df.groupby('cty', group_keys=False).apply(winsorize_drop)

    # Create idp (group of sector-occupation digit)
    df['idp'] = df.groupby(['sector', 'occup1digit']).ngroup()

    # Calculate average labor supply by cell
    df = df.rename(columns={'shActive_so': 'shAct'})

    for cty in range(1, 17):
        df[f'shActAveg_cty{cty}'] = df[df['cty'] == cty].groupby('idp')['shAct'].transform('mean')

    df['shActAv'] = np.nan
    for cty in range(1, 17):
        df.loc[df['cty'] == cty, 'shActAv'] = df.loc[df['cty'] == cty, f'shActAveg_cty{cty}']

    # Create double cluster variable
    df['double_cluster'] = df.groupby(['country', 'sector']).ngroup()

    # The outcome variable
    df['DHSshEmployee'] = df['DHSshFE2']

    # Rename occupation for clarity
    df['occu1'] = df['occup1digit']

    return df

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def run_wls_regression(df_sub, formula, weights=None, cluster_var=None):
    """Run WLS regression with clustering."""
    try:
        df_reg = df_sub.dropna(subset=[col for col in df_sub.columns if col in formula.replace('~', ' ').replace('+', ' ').replace('|', ' ').split()])

        if len(df_reg) < 10:
            return None

        # Check for fixed effects syntax
        if '|' in formula:
            # Use pyfixest for FE models
            if weights is not None and weights in df_reg.columns:
                model = pf.feols(formula, data=df_reg, weights=weights, vcov={'CRV1': cluster_var} if cluster_var else 'hetero')
            else:
                model = pf.feols(formula, data=df_reg, vcov={'CRV1': cluster_var} if cluster_var else 'hetero')
            return model
        else:
            # Use statsmodels for non-FE models
            if weights is not None and weights in df_reg.columns:
                model = smf.wls(formula, data=df_reg, weights=df_reg[weights]).fit(
                    cov_type='cluster', cov_kwds={'groups': df_reg[cluster_var]} if cluster_var else None)
            else:
                model = smf.ols(formula, data=df_reg).fit(
                    cov_type='cluster', cov_kwds={'groups': df_reg[cluster_var]} if cluster_var else None)
            return model
    except Exception as e:
        print(f"  Error in regression: {e}")
        return None

def extract_results(model, treatment_var, spec_id, spec_tree_path, outcome_var, df_sub,
                    sample_desc, fixed_effects, controls_desc, cluster_var, model_type, weights_desc="shActAv"):
    """Extract results from a fitted model."""
    try:
        if hasattr(model, 'coef'):
            # pyfixest model
            coef = model.coef()[treatment_var] if treatment_var in model.coef() else None
            se = model.se()[treatment_var] if treatment_var in model.se() else None
            pval = model.pvalue()[treatment_var] if treatment_var in model.pvalue() else None
            tstat = model.tstat()[treatment_var] if treatment_var in model.tstat() else None
            n_obs = model.nobs
            r2 = model.r2 if hasattr(model, 'r2') else None

            # Get all coefficients
            coef_vector = {}
            for var in model.coef().index:
                coef_vector[var] = {
                    'coef': float(model.coef()[var]),
                    'se': float(model.se()[var]),
                    'pval': float(model.pvalue()[var])
                }
        else:
            # statsmodels model
            coef = model.params.get(treatment_var, None)
            se = model.bse.get(treatment_var, None)
            pval = model.pvalues.get(treatment_var, None)
            tstat = model.tvalues.get(treatment_var, None)
            n_obs = int(model.nobs)
            r2 = model.rsquared if hasattr(model, 'rsquared') else None

            coef_vector = {}
            for var in model.params.index:
                coef_vector[var] = {
                    'coef': float(model.params[var]),
                    'se': float(model.bse[var]),
                    'pval': float(model.pvalues[var])
                }

        if coef is None:
            return None

        ci_lower = coef - 1.96 * se if se else None
        ci_upper = coef + 1.96 * se if se else None

        return {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': float(coef) if coef else None,
            'std_error': float(se) if se else None,
            't_stat': float(tstat) if tstat else None,
            'p_value': float(pval) if pval else None,
            'ci_lower': float(ci_lower) if ci_lower else None,
            'ci_upper': float(ci_upper) if ci_upper else None,
            'n_obs': n_obs,
            'r_squared': float(r2) if r2 else None,
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects,
            'controls_desc': controls_desc,
            'cluster_var': cluster_var,
            'model_type': model_type,
            'weights': weights_desc,
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
    except Exception as e:
        print(f"  Error extracting results: {e}")
        return None

# =============================================================================
# MAIN SPECIFICATION SEARCH
# =============================================================================
def run_specification_search():
    """Run comprehensive specification search."""
    print("Loading and preparing data...")
    df = prepare_data()

    # Filter to year 2011 as in the main analysis
    df_2011 = df[df['year'] == 2011].copy()

    print(f"Sample size: {len(df_2011)}")
    print(f"Countries: {df_2011['country'].nunique()}")

    results = []
    spec_count = 0

    # Define controls
    sector_controls = ['sec1', 'sec2', 'sec3', 'sec4', 'sec5']  # sec6 is omitted (Services)
    country_controls = [f'cty{i}' for i in range(2, 17)]  # cty1 (AT) is omitted
    all_controls = sector_controls + country_controls
    controls_formula = ' + '.join(all_controls)

    # ==========================================================================
    # 1. BASELINE SPECIFICATIONS (both treatment measures)
    # ==========================================================================
    print("\n=== BASELINE SPECIFICATIONS ===")

    for treat_var, treat_name in [('PCT_aiW', 'Webb'), ('PCT_aiF', 'Felten')]:
        formula = f"DHSshEmployee ~ {treat_var} + {controls_formula}"
        model = run_wls_regression(df_2011, formula, weights='shActAv', cluster_var='double_cluster')

        if model:
            result = extract_results(
                model, treat_var, f'baseline_{treat_name}',
                'methods/cross_sectional_ols.md#baseline',
                'DHSshEmployee', df_2011,
                f'Full sample, year 2011, {treat_name} AI measure',
                'sector + country', controls_formula,
                'double_cluster (country x sector)', 'WLS'
            )
            if result:
                results.append(result)
                spec_count += 1
                print(f"  Spec {spec_count}: baseline_{treat_name} - coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # ==========================================================================
    # 2. CONTROL VARIABLE VARIATIONS
    # ==========================================================================
    print("\n=== CONTROL VARIATIONS ===")

    # 2.1 No controls (just fixed effects)
    for treat_var, treat_name in [('PCT_aiW', 'Webb'), ('PCT_aiF', 'Felten')]:
        formula = f"DHSshEmployee ~ {treat_var}"
        model = run_wls_regression(df_2011, formula, weights='shActAv', cluster_var='double_cluster')

        if model:
            result = extract_results(
                model, treat_var, f'ols/controls/none_{treat_name}',
                'robustness/control_progression.md',
                'DHSshEmployee', df_2011,
                f'No controls, {treat_name}',
                'none', 'none',
                'double_cluster', 'WLS'
            )
            if result:
                results.append(result)
                spec_count += 1
                print(f"  Spec {spec_count}: no controls {treat_name} - coef={result['coefficient']:.4f}")

    # 2.2 Sector FE only (no country FE)
    for treat_var, treat_name in [('PCT_aiW', 'Webb'), ('PCT_aiF', 'Felten')]:
        formula = f"DHSshEmployee ~ {treat_var} + {' + '.join(sector_controls)}"
        model = run_wls_regression(df_2011, formula, weights='shActAv', cluster_var='double_cluster')

        if model:
            result = extract_results(
                model, treat_var, f'ols/fe/sector_only_{treat_name}',
                'methods/cross_sectional_ols.md#fixed-effects',
                'DHSshEmployee', df_2011,
                f'Sector FE only, {treat_name}',
                'sector', ' + '.join(sector_controls),
                'double_cluster', 'WLS'
            )
            if result:
                results.append(result)
                spec_count += 1
                print(f"  Spec {spec_count}: sector FE only {treat_name} - coef={result['coefficient']:.4f}")

    # 2.3 Country FE only (no sector FE)
    for treat_var, treat_name in [('PCT_aiW', 'Webb'), ('PCT_aiF', 'Felten')]:
        formula = f"DHSshEmployee ~ {treat_var} + {' + '.join(country_controls)}"
        model = run_wls_regression(df_2011, formula, weights='shActAv', cluster_var='double_cluster')

        if model:
            result = extract_results(
                model, treat_var, f'ols/fe/country_only_{treat_name}',
                'methods/cross_sectional_ols.md#fixed-effects',
                'DHSshEmployee', df_2011,
                f'Country FE only, {treat_name}',
                'country', ' + '.join(country_controls),
                'double_cluster', 'WLS'
            )
            if result:
                results.append(result)
                spec_count += 1
                print(f"  Spec {spec_count}: country FE only {treat_name} - coef={result['coefficient']:.4f}")

    # 2.4 Leave-one-out: drop each sector control
    for treat_var, treat_name in [('PCT_aiW', 'Webb'), ('PCT_aiF', 'Felten')]:
        for drop_sec in sector_controls:
            remaining_sec = [s for s in sector_controls if s != drop_sec]
            formula = f"DHSshEmployee ~ {treat_var} + {' + '.join(remaining_sec + country_controls)}"
            model = run_wls_regression(df_2011, formula, weights='shActAv', cluster_var='double_cluster')

            if model:
                result = extract_results(
                    model, treat_var, f'robust/loo/drop_{drop_sec}_{treat_name}',
                    'robustness/leave_one_out.md',
                    'DHSshEmployee', df_2011,
                    f'Drop {drop_sec}, {treat_name}',
                    'sector (partial) + country', f'Drop {drop_sec}',
                    'double_cluster', 'WLS'
                )
                if result:
                    results.append(result)
                    spec_count += 1
                    print(f"  Spec {spec_count}: LOO drop {drop_sec} {treat_name} - coef={result['coefficient']:.4f}")

    # ==========================================================================
    # 3. SAMPLE RESTRICTIONS - Country Groups (Tables A2)
    # ==========================================================================
    print("\n=== SAMPLE RESTRICTIONS - COUNTRY GROUPS ===")

    country_groups = {
        'clas4_1': ('clas4', 1, 'Low LFP participation (relative)'),
        'clas4_2': ('clas4', 2, 'High LFP participation (relative)'),
        'clas5_1': ('clas5', 1, 'Low LFP participation (levels)'),
        'clas5_2': ('clas5', 2, 'High LFP participation (levels)'),
        'clas3_1': ('clas3', 1, 'Higher relative upskilling'),
        'clas3_2': ('clas3', 2, 'Lower relative upskilling'),
        'clas1_1': ('clas1', 1, 'Below US female education'),
        'clas1_2': ('clas1', 2, 'Near/above US female education'),
    }

    for treat_var, treat_name in [('PCT_aiW', 'Webb'), ('PCT_aiF', 'Felten')]:
        for group_id, (var, val, desc) in country_groups.items():
            df_sub = df_2011[df_2011[var] == val].copy()
            formula = f"DHSshEmployee ~ {treat_var} + {controls_formula}"
            model = run_wls_regression(df_sub, formula, weights='shActAv', cluster_var='double_cluster')

            if model:
                result = extract_results(
                    model, treat_var, f'robust/sample/{group_id}_{treat_name}',
                    'robustness/sample_restrictions.md',
                    'DHSshEmployee', df_sub,
                    f'{desc}, {treat_name}',
                    'sector + country', controls_formula,
                    'double_cluster', 'WLS'
                )
                if result:
                    results.append(result)
                    spec_count += 1
                    print(f"  Spec {spec_count}: {group_id} {treat_name} - coef={result['coefficient']:.4f}, n={result['n_obs']}")

    # ==========================================================================
    # 4. SAMPLE RESTRICTIONS - Occupation Exclusion (Table A3)
    # ==========================================================================
    print("\n=== SAMPLE RESTRICTIONS - OCCUPATION EXCLUSION ===")

    for treat_var, treat_name in [('PCT_aiW', 'Webb'), ('PCT_aiF', 'Felten')]:
        for occ in range(1, 10):
            df_sub = df_2011[df_2011['occu1'] != occ].copy()
            formula = f"DHSshEmployee ~ {treat_var} + {controls_formula}"
            model = run_wls_regression(df_sub, formula, weights='shActAv', cluster_var='double_cluster')

            if model:
                result = extract_results(
                    model, treat_var, f'robust/sample/exclude_occup{occ}_{treat_name}',
                    'robustness/sample_restrictions.md',
                    'DHSshEmployee', df_sub,
                    f'Exclude occupation {occ}, {treat_name}',
                    'sector + country', controls_formula,
                    'double_cluster', 'WLS'
                )
                if result:
                    results.append(result)
                    spec_count += 1
                    print(f"  Spec {spec_count}: exclude occup{occ} {treat_name} - coef={result['coefficient']:.4f}")

    # ==========================================================================
    # 5. SAMPLE RESTRICTIONS - Drop Each Country
    # ==========================================================================
    print("\n=== SAMPLE RESTRICTIONS - DROP EACH COUNTRY ===")

    countries = df_2011['country'].unique()
    for treat_var, treat_name in [('PCT_aiW', 'Webb'), ('PCT_aiF', 'Felten')]:
        for country in countries:
            df_sub = df_2011[df_2011['country'] != country].copy()
            # Need to adjust country controls for dropped country
            formula = f"DHSshEmployee ~ {treat_var} + {controls_formula}"
            model = run_wls_regression(df_sub, formula, weights='shActAv', cluster_var='double_cluster')

            if model:
                result = extract_results(
                    model, treat_var, f'robust/sample/drop_{country}_{treat_name}',
                    'robustness/sample_restrictions.md',
                    'DHSshEmployee', df_sub,
                    f'Drop {country}, {treat_name}',
                    'sector + country', controls_formula,
                    'double_cluster', 'WLS'
                )
                if result:
                    results.append(result)
                    spec_count += 1
                    print(f"  Spec {spec_count}: drop {country} {treat_name} - coef={result['coefficient']:.4f}")

    # ==========================================================================
    # 6. SAMPLE RESTRICTIONS - Sector Subsamples
    # ==========================================================================
    print("\n=== SAMPLE RESTRICTIONS - BY SECTOR ===")

    sector_names = {
        'sec1': 'Agriculture',
        'sec2': 'Construction',
        'sec3': 'Financial Services',
        'sec4': 'Manufacturing',
        'sec5': 'Public',
        'sec6': 'Services'
    }

    for treat_var, treat_name in [('PCT_aiW', 'Webb'), ('PCT_aiF', 'Felten')]:
        for sec_var, sec_name in sector_names.items():
            if sec_var == 'sec6':
                df_sub = df_2011[(df_2011[['sec1', 'sec2', 'sec3', 'sec4', 'sec5']].sum(axis=1) == 0)].copy()
            else:
                df_sub = df_2011[df_2011[sec_var] == 1].copy()

            if len(df_sub) > 30:
                formula = f"DHSshEmployee ~ {treat_var} + {' + '.join(country_controls)}"
                model = run_wls_regression(df_sub, formula, weights='shActAv', cluster_var='double_cluster')

                if model:
                    result = extract_results(
                        model, treat_var, f'robust/sample/{sec_var}_{treat_name}',
                        'robustness/sample_restrictions.md',
                        'DHSshEmployee', df_sub,
                        f'{sec_name} sector only, {treat_name}',
                        'country', ' + '.join(country_controls),
                        'double_cluster', 'WLS'
                    )
                    if result:
                        results.append(result)
                        spec_count += 1
                        print(f"  Spec {spec_count}: {sec_name} {treat_name} - coef={result['coefficient']:.4f}")

    # ==========================================================================
    # 7. CLUSTERING VARIATIONS
    # ==========================================================================
    print("\n=== CLUSTERING VARIATIONS ===")

    for treat_var, treat_name in [('PCT_aiW', 'Webb'), ('PCT_aiF', 'Felten')]:
        formula = f"DHSshEmployee ~ {treat_var} + {controls_formula}"

        # 7.1 Cluster by country only
        model = run_wls_regression(df_2011, formula, weights='shActAv', cluster_var='country')
        if model:
            result = extract_results(
                model, treat_var, f'robust/cluster/country_{treat_name}',
                'robustness/clustering_variations.md',
                'DHSshEmployee', df_2011,
                f'Cluster by country, {treat_name}',
                'sector + country', controls_formula,
                'country', 'WLS'
            )
            if result:
                results.append(result)
                spec_count += 1
                print(f"  Spec {spec_count}: cluster country {treat_name} - coef={result['coefficient']:.4f}")

        # 7.2 Cluster by sector only
        model = run_wls_regression(df_2011, formula, weights='shActAv', cluster_var='sector')
        if model:
            result = extract_results(
                model, treat_var, f'robust/cluster/sector_{treat_name}',
                'robustness/clustering_variations.md',
                'DHSshEmployee', df_2011,
                f'Cluster by sector, {treat_name}',
                'sector + country', controls_formula,
                'sector', 'WLS'
            )
            if result:
                results.append(result)
                spec_count += 1
                print(f"  Spec {spec_count}: cluster sector {treat_name} - coef={result['coefficient']:.4f}")

        # 7.3 Robust (heteroskedasticity-consistent) SE only
        model = run_wls_regression(df_2011, formula, weights='shActAv', cluster_var=None)
        if model:
            result = extract_results(
                model, treat_var, f'robust/cluster/robust_hc_{treat_name}',
                'robustness/clustering_variations.md',
                'DHSshEmployee', df_2011,
                f'Robust HC SE, {treat_name}',
                'sector + country', controls_formula,
                'none (robust HC)', 'WLS'
            )
            if result:
                results.append(result)
                spec_count += 1
                print(f"  Spec {spec_count}: robust HC {treat_name} - coef={result['coefficient']:.4f}")

    # ==========================================================================
    # 8. WEIGHTS VARIATIONS
    # ==========================================================================
    print("\n=== WEIGHTS VARIATIONS ===")

    for treat_var, treat_name in [('PCT_aiW', 'Webb'), ('PCT_aiF', 'Felten')]:
        formula = f"DHSshEmployee ~ {treat_var} + {controls_formula}"

        # 8.1 Unweighted
        model = run_wls_regression(df_2011, formula, weights=None, cluster_var='double_cluster')
        if model:
            result = extract_results(
                model, treat_var, f'robust/weights/unweighted_{treat_name}',
                'robustness/model_specification.md',
                'DHSshEmployee', df_2011,
                f'Unweighted OLS, {treat_name}',
                'sector + country', controls_formula,
                'double_cluster', 'OLS'
            )
            if result:
                result['weights'] = 'unweighted'
                results.append(result)
                spec_count += 1
                print(f"  Spec {spec_count}: unweighted {treat_name} - coef={result['coefficient']:.4f}")

    # ==========================================================================
    # 9. FUNCTIONAL FORM VARIATIONS
    # ==========================================================================
    print("\n=== FUNCTIONAL FORM VARIATIONS ===")

    for treat_var, treat_name in [('PCT_aiW', 'Webb'), ('PCT_aiF', 'Felten')]:
        # 9.1 Quadratic treatment
        df_2011[f'{treat_var}_sq'] = df_2011[treat_var] ** 2
        formula = f"DHSshEmployee ~ {treat_var} + {treat_var}_sq + {controls_formula}"
        model = run_wls_regression(df_2011, formula, weights='shActAv', cluster_var='double_cluster')
        if model:
            result = extract_results(
                model, treat_var, f'robust/form/quadratic_{treat_name}',
                'robustness/functional_form.md',
                'DHSshEmployee', df_2011,
                f'Quadratic treatment, {treat_name}',
                'sector + country', f'{treat_var} + {treat_var}^2 + {controls_formula}',
                'double_cluster', 'WLS'
            )
            if result:
                results.append(result)
                spec_count += 1
                print(f"  Spec {spec_count}: quadratic {treat_name} - coef={result['coefficient']:.4f}")

        # 9.2 Log treatment (AI exposure is percentile, so log is meaningful)
        df_2011[f'log_{treat_var}'] = np.log(df_2011[treat_var] + 1)
        formula = f"DHSshEmployee ~ log_{treat_var} + {controls_formula}"
        model = run_wls_regression(df_2011, formula, weights='shActAv', cluster_var='double_cluster')
        if model:
            result = extract_results(
                model, f'log_{treat_var}', f'robust/form/log_treatment_{treat_name}',
                'robustness/functional_form.md',
                'DHSshEmployee', df_2011,
                f'Log treatment, {treat_name}',
                'sector + country', f'log({treat_var}) + {controls_formula}',
                'double_cluster', 'WLS'
            )
            if result:
                result['treatment_var'] = treat_var
                results.append(result)
                spec_count += 1
                print(f"  Spec {spec_count}: log treatment {treat_name} - coef={result['coefficient']:.4f}")

        # 9.3 Treatment deciles (categorical)
        df_2011[f'{treat_var}_decile'] = pd.qcut(df_2011[treat_var], 10, labels=False, duplicates='drop')
        formula = f"DHSshEmployee ~ C({treat_var}_decile) + {controls_formula}"
        try:
            model = run_wls_regression(df_2011, formula, weights='shActAv', cluster_var='double_cluster')
            if model:
                result = extract_results(
                    model, f'C({treat_var}_decile)[T.9]', f'robust/form/decile_treatment_{treat_name}',
                    'robustness/functional_form.md',
                    'DHSshEmployee', df_2011,
                    f'Treatment deciles, {treat_name} (top decile vs bottom)',
                    'sector + country', f'{treat_var} deciles + {controls_formula}',
                    'double_cluster', 'WLS'
                )
                if result:
                    result['treatment_var'] = treat_var
                    results.append(result)
                    spec_count += 1
                    print(f"  Spec {spec_count}: decile treatment {treat_name}")
        except:
            pass

    # ==========================================================================
    # 10. HETEROGENEITY - Country-Specific Coefficients
    # ==========================================================================
    print("\n=== HETEROGENEITY - COUNTRY-SPECIFIC ===")

    # This replicates the Figure A1 analysis - interact AI with country dummies
    for treat_var, treat_name in [('PCT_aiW', 'Webb'), ('PCT_aiF', 'Felten')]:
        # Get country-specific AI interactions
        interactions = [f'{treat_var}_cty{i}' for i in range(1, 17)]
        formula = f"DHSshEmployee ~ {' + '.join(interactions)} + {controls_formula}"
        model = run_wls_regression(df_2011, formula, weights='shActAv', cluster_var='double_cluster')

        if model:
            result = extract_results(
                model, f'{treat_var}_cty1', f'robust/het/country_interaction_{treat_name}',
                'robustness/heterogeneity.md',
                'DHSshEmployee', df_2011,
                f'Country-specific AI effects, {treat_name}',
                'sector + country', f'AI x country interactions + {controls_formula}',
                'double_cluster', 'WLS'
            )
            if result:
                results.append(result)
                spec_count += 1
                print(f"  Spec {spec_count}: country interactions {treat_name}")

    # ==========================================================================
    # 11. ALTERNATIVE OUTCOME - Using Different Years
    # ==========================================================================
    print("\n=== ALTERNATIVE TIME PERIODS ===")

    # Try different years if available
    for year in [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]:
        df_year = df[df['year'] == year].copy()
        if len(df_year) > 100:
            for treat_var, treat_name in [('PCT_aiW', 'Webb'), ('PCT_aiF', 'Felten')]:
                formula = f"DHSshEmployee ~ {treat_var} + {controls_formula}"
                model = run_wls_regression(df_year, formula, weights='shActAv', cluster_var='double_cluster')

                if model:
                    result = extract_results(
                        model, treat_var, f'robust/sample/year_{year}_{treat_name}',
                        'robustness/sample_restrictions.md',
                        'DHSshEmployee', df_year,
                        f'Year {year}, {treat_name}',
                        'sector + country', controls_formula,
                        'double_cluster', 'WLS'
                    )
                    if result:
                        results.append(result)
                        spec_count += 1
                        print(f"  Spec {spec_count}: year {year} {treat_name} - coef={result['coefficient']:.4f}")

    # ==========================================================================
    # 12. ALTERNATIVE OUTCOME - Female Share Level
    # ==========================================================================
    print("\n=== ALTERNATIVE OUTCOME - FEMALE SHARE LEVEL ===")

    for treat_var, treat_name in [('PCT_aiW', 'Webb'), ('PCT_aiF', 'Felten')]:
        formula = f"shWomen ~ {treat_var} + {controls_formula}"
        model = run_wls_regression(df_2011, formula, weights='shActAv', cluster_var='double_cluster')

        if model:
            result = extract_results(
                model, treat_var, f'robust/outcome/shWomen_{treat_name}',
                'robustness/measurement.md',
                'shWomen', df_2011,
                f'Female share level (not change), {treat_name}',
                'sector + country', controls_formula,
                'double_cluster', 'WLS'
            )
            if result:
                results.append(result)
                spec_count += 1
                print(f"  Spec {spec_count}: shWomen outcome {treat_name} - coef={result['coefficient']:.4f}")

    # ==========================================================================
    # 13. HETEROGENEITY - By High/Low AI Exposure
    # ==========================================================================
    print("\n=== HETEROGENEITY - AI EXPOSURE QUARTILES ===")

    for treat_var, treat_name in [('PCT_aiW', 'Webb'), ('PCT_aiF', 'Felten')]:
        # High AI exposure (top quartile)
        q75 = df_2011[treat_var].quantile(0.75)
        df_high = df_2011[df_2011[treat_var] >= q75].copy()
        formula = f"DHSshEmployee ~ {treat_var} + {controls_formula}"
        model = run_wls_regression(df_high, formula, weights='shActAv', cluster_var='double_cluster')

        if model:
            result = extract_results(
                model, treat_var, f'robust/het/high_exposure_{treat_name}',
                'robustness/heterogeneity.md',
                'DHSshEmployee', df_high,
                f'Top quartile AI exposure, {treat_name}',
                'sector + country', controls_formula,
                'double_cluster', 'WLS'
            )
            if result:
                results.append(result)
                spec_count += 1
                print(f"  Spec {spec_count}: high exposure {treat_name} - coef={result['coefficient']:.4f}")

        # Low AI exposure (bottom quartile)
        q25 = df_2011[treat_var].quantile(0.25)
        df_low = df_2011[df_2011[treat_var] <= q25].copy()
        model = run_wls_regression(df_low, formula, weights='shActAv', cluster_var='double_cluster')

        if model:
            result = extract_results(
                model, treat_var, f'robust/het/low_exposure_{treat_name}',
                'robustness/heterogeneity.md',
                'DHSshEmployee', df_low,
                f'Bottom quartile AI exposure, {treat_name}',
                'sector + country', controls_formula,
                'double_cluster', 'WLS'
            )
            if result:
                results.append(result)
                spec_count += 1
                print(f"  Spec {spec_count}: low exposure {treat_name} - coef={result['coefficient']:.4f}")

    # ==========================================================================
    # 14. PLACEBO - Using Pre-Treatment Period (2010 and earlier)
    # ==========================================================================
    print("\n=== PLACEBO TESTS ===")

    # Pre-2011 data for placebo (if DHS can be computed)
    df_pre = df[df['year'] < 2011].copy()
    if len(df_pre) > 100 and df_pre['DHSshEmployee'].notna().sum() > 50:
        for treat_var, treat_name in [('PCT_aiW', 'Webb'), ('PCT_aiF', 'Felten')]:
            formula = f"DHSshEmployee ~ {treat_var} + {controls_formula}"
            model = run_wls_regression(df_pre, formula, weights='shActAv', cluster_var='double_cluster')

            if model:
                result = extract_results(
                    model, treat_var, f'robust/placebo/pre_2011_{treat_name}',
                    'robustness/placebo_tests.md',
                    'DHSshEmployee', df_pre,
                    f'Pre-2011 placebo, {treat_name}',
                    'sector + country', controls_formula,
                    'double_cluster', 'WLS'
                )
                if result:
                    results.append(result)
                    spec_count += 1
                    print(f"  Spec {spec_count}: pre-2011 placebo {treat_name} - coef={result['coefficient']:.4f}")

    # ==========================================================================
    # 15. OUTLIER TREATMENT
    # ==========================================================================
    print("\n=== OUTLIER TREATMENT ===")

    for treat_var, treat_name in [('PCT_aiW', 'Webb'), ('PCT_aiF', 'Felten')]:
        # 15.1 More aggressive winsorization (5%)
        df_wins = df_2011.copy()
        q5 = df_wins['DHSshEmployee'].quantile(0.05)
        q95 = df_wins['DHSshEmployee'].quantile(0.95)
        df_wins = df_wins[(df_wins['DHSshEmployee'] > q5) & (df_wins['DHSshEmployee'] < q95)]

        formula = f"DHSshEmployee ~ {treat_var} + {controls_formula}"
        model = run_wls_regression(df_wins, formula, weights='shActAv', cluster_var='double_cluster')

        if model:
            result = extract_results(
                model, treat_var, f'robust/sample/winsor_5pct_{treat_name}',
                'robustness/sample_restrictions.md',
                'DHSshEmployee', df_wins,
                f'5% winsorization, {treat_name}',
                'sector + country', controls_formula,
                'double_cluster', 'WLS'
            )
            if result:
                results.append(result)
                spec_count += 1
                print(f"  Spec {spec_count}: 5% winsor {treat_name} - coef={result['coefficient']:.4f}")

        # 15.2 Trim top and bottom 10%
        q10 = df_2011['DHSshEmployee'].quantile(0.10)
        q90 = df_2011['DHSshEmployee'].quantile(0.90)
        df_trim = df_2011[(df_2011['DHSshEmployee'] > q10) & (df_2011['DHSshEmployee'] < q90)]

        model = run_wls_regression(df_trim, formula, weights='shActAv', cluster_var='double_cluster')

        if model:
            result = extract_results(
                model, treat_var, f'robust/sample/trim_10pct_{treat_name}',
                'robustness/sample_restrictions.md',
                'DHSshEmployee', df_trim,
                f'10% trimming, {treat_name}',
                'sector + country', controls_formula,
                'double_cluster', 'WLS'
            )
            if result:
                results.append(result)
                spec_count += 1
                print(f"  Spec {spec_count}: 10% trim {treat_name} - coef={result['coefficient']:.4f}")

    # ==========================================================================
    # 16. ALTERNATIVE TREATMENT - Both AI Measures Together
    # ==========================================================================
    print("\n=== ALTERNATIVE TREATMENT - BOTH AI MEASURES ===")

    formula = f"DHSshEmployee ~ PCT_aiW + PCT_aiF + {controls_formula}"
    model = run_wls_regression(df_2011, formula, weights='shActAv', cluster_var='double_cluster')

    if model:
        for treat_var, treat_name in [('PCT_aiW', 'Webb'), ('PCT_aiF', 'Felten')]:
            result = extract_results(
                model, treat_var, f'robust/treatment/both_measures_{treat_name}',
                'robustness/model_specification.md',
                'DHSshEmployee', df_2011,
                f'Both AI measures, effect of {treat_name}',
                'sector + country', f'PCT_aiW + PCT_aiF + {controls_formula}',
                'double_cluster', 'WLS'
            )
            if result:
                results.append(result)
                spec_count += 1
                print(f"  Spec {spec_count}: both measures {treat_name} - coef={result['coefficient']:.4f}")

    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================
    print(f"\n=== TOTAL SPECIFICATIONS: {spec_count} ===")

    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    output_path = f"{OUTPUT_DIR}/specification_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved results to: {output_path}")

    return results_df

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    results = run_specification_search()
    print("\nDone!")
