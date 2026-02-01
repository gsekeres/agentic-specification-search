"""
Specification Search Analysis for Paper 113597-V2
Attanasio et al. (2014) - The Impacts of Microfinance: Evidence from Joint-Liability Lending in Mongolia

This paper is a Randomized Controlled Trial (RCT) evaluating the impact of microfinance
(group lending vs. control) in Mongolia. The main estimation strategy is a cross-sectional
comparison at follow-up, controlling for baseline outcomes (ANCOVA).

Key features:
- Treatment: group (group lending treatment = 1, control = 0)
- Sample: Exclude individual liability arm (indiv != 1) for main analysis
- Method: OLS with baseline outcome as control, robust SE clustered at soum level
- Controls: loan_baseline, eduvoc, edusec, age16, under16, marr_cohab, age, age_sq,
           buddhist, hahl, aug_b, sep_f, nov_f + aimag dummies

Authors: Orazio Attanasio, Britta Augsburg, Ralph De Haas, Emla Fitzsimons, Heike Harmgart
Journal: AEJ Applied Economics
"""

import pandas as pd
import numpy as np
import pyreadstat
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/113597-V2/Analysis-files"
OUTPUT_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/113597-V2"

def load_and_prepare_data():
    """Load and merge all necessary data files, replicating the Stata do-file logic."""

    # Load main data
    df, _ = pyreadstat.read_dta(f"{BASE_PATH}/data/Baseline/all_outcomes_controls.dta")

    # Merge Debt data
    debt, _ = pyreadstat.read_dta(f"{BASE_PATH}/data/Followup/Debt.dta")
    df = df.merge(debt, on=['rescode', 'hhid', 'aimag', 'soum', 'treatment'],
                  how='left', suffixes=('', '_debt'))

    # Merge Baseline outcomes
    bl_out, _ = pyreadstat.read_dta(f"{BASE_PATH}/data/Baseline/Baseline outcomes.dta")
    df = df.merge(bl_out, on='rescode', how='left')

    # Drop if not reinterviewed
    df = df[df['reint'] == 1].copy()

    # Create age16 (combination of male and female 16+)
    df['age16'] = df['age16m'].fillna(0) + df['age16f'].fillna(0)

    # Create aimag dummies
    aimag_dummies = pd.get_dummies(df['aimag'], prefix='aimag', drop_first=False)
    df = pd.concat([df, aimag_dummies], axis=1)

    # Create outcome variables for Table 2
    df['nloans_x'] = df['nloans_x'].fillna(0)
    for v in ['amount1_x', 'amount2_x', 'amount3_x']:
        if v in df.columns:
            df[v] = df[v].fillna(0)

    df['tot_amount_x'] = df['amount1_x'] + df['amount2_x'] + df['amount3_x']

    # Create loan dummies
    df['dum_loan_x'] = (df['nloans_x'] > 0).astype(int)

    # Scale profit and assets
    df['scaled_assets_all'] = df['assets_all'] / 1000
    df['scaled_profit'] = df['profit'] / 1000

    return df


def get_aimag_dummies(df):
    """Get list of aimag dummy columns."""
    return [c for c in df.columns if c.startswith('aimag_')]


def run_regression(df, outcome, treatment='group', controls=None, baseline_var=None,
                   cluster_var='soum', fe_vars=None, sample_filter=None, robust=True):
    """
    Run OLS regression with specified controls and clustering.

    Returns dictionary with coefficient info.
    """
    # Apply sample filter
    analysis_df = df.copy()
    if sample_filter is not None:
        analysis_df = analysis_df[sample_filter].copy()

    # Default: follow-up period only, exclude individual liability arm
    analysis_df = analysis_df[(analysis_df['followup'] == 1) & (analysis_df['indiv'] != 1)].copy()

    # Drop missing values for key variables
    key_vars = [outcome, treatment]
    if baseline_var and baseline_var in analysis_df.columns:
        key_vars.append(baseline_var)
    if controls:
        key_vars.extend([c for c in controls if c in analysis_df.columns])
    if cluster_var and cluster_var in analysis_df.columns:
        key_vars.append(cluster_var)

    analysis_df = analysis_df.dropna(subset=[v for v in key_vars if v in analysis_df.columns])

    if len(analysis_df) < 20:
        return None

    # Build formula
    rhs_vars = [treatment]
    if baseline_var and baseline_var in analysis_df.columns:
        rhs_vars.append(baseline_var)
    if controls:
        rhs_vars.extend([c for c in controls if c in analysis_df.columns])

    formula = f"{outcome} ~ " + " + ".join(rhs_vars)

    try:
        # Handle clustering
        if cluster_var and cluster_var in analysis_df.columns:
            model = smf.ols(formula, data=analysis_df).fit(
                cov_type='cluster',
                cov_kwds={'groups': analysis_df[cluster_var]}
            )
            n_clusters = analysis_df[cluster_var].nunique()
        else:
            # Use robust (HC1) standard errors when no clustering
            model = smf.ols(formula, data=analysis_df).fit(cov_type='HC1')
            n_clusters = None

        # Get control group mean
        control_mask = (analysis_df[treatment] == 0) if treatment in analysis_df.columns else True
        control_mean = analysis_df.loc[control_mask, outcome].mean()

        # Build result dictionary
        result = {
            'outcome': outcome,
            'treatment_var': treatment,
            'treatment_coef': model.params.get(treatment, np.nan),
            'treatment_se': model.bse.get(treatment, np.nan),
            'treatment_pval': model.pvalues.get(treatment, np.nan),
            'treatment_ci_lower': model.conf_int().loc[treatment, 0] if treatment in model.params else np.nan,
            'treatment_ci_upper': model.conf_int().loc[treatment, 1] if treatment in model.params else np.nan,
            'n_obs': int(model.nobs),
            'r_squared': model.rsquared,
            'control_mean': control_mean,
            'cluster_var': cluster_var,
            'n_clusters': n_clusters,
            'controls_used': controls if controls else [],
            'baseline_control': baseline_var,
            'coefficient_vector': {
                var: {
                    'coef': float(model.params[var]),
                    'se': float(model.bse[var]),
                    'pval': float(model.pvalues[var])
                } for var in model.params.index
            }
        }

        return result

    except Exception as e:
        print(f"Error running regression for {outcome}: {e}")
        return None


def run_specification_search(df):
    """Run all specifications from the specification tree."""

    results = []

    # Define control sets
    base_controls = ['loan_baseline', 'eduvoc', 'edusec', 'age16', 'under16',
                     'marr_cohab', 'age', 'age_sq', 'buddhist', 'hahl',
                     'aug_b', 'sep_f', 'nov_f']
    aimag_dummies = get_aimag_dummies(df)
    full_controls = base_controls + aimag_dummies

    # Main outcome variables (from Table 2)
    main_outcomes = [
        ('dum_loan_x', 'BLdum_loan_x'),  # Has loan from XacBank
        ('enterprise', 'BLenterprise'),  # Has enterprise
        ('scaled_profit', 'BLscaled_profit'),  # Scaled profit
        ('scaled_assets_all', 'BLscaled_assets_all'),  # Scaled assets
    ]

    # 1. BASELINE SPECIFICATION (replicating paper's Table 2 main spec)
    print("Running baseline specifications...")
    for outcome, bl_var in main_outcomes:
        if outcome not in df.columns:
            continue

        result = run_regression(
            df, outcome=outcome,
            controls=full_controls,
            baseline_var=bl_var,
            cluster_var='soum'
        )

        if result:
            result['spec_id'] = 'baseline'
            result['spec_tree_path'] = 'methods/cross_sectional_ols.md'
            result['spec_description'] = 'Paper baseline: ANCOVA with full controls, clustered at soum'
            results.append(result)

    # 2. OLS CONTROL SET VARIATIONS
    print("Running control set variations...")
    control_variations = [
        ('ols/controls/none', [], None, 'No controls'),
        ('ols/controls/minimal', ['edusec', 'age', 'marr_cohab'], None, 'Minimal demographics'),
        ('ols/controls/baseline', full_controls, None, 'Full controls without BL outcome'),
        ('ols/controls/full', full_controls, 'use_bl', 'Full controls with BL outcome (baseline)')
    ]

    for spec_id, controls, bl_flag, desc in control_variations:
        for outcome, bl_var in main_outcomes:
            if outcome not in df.columns:
                continue

            use_bl = bl_var if bl_flag == 'use_bl' else None
            result = run_regression(
                df, outcome=outcome,
                controls=controls if controls else None,
                baseline_var=use_bl,
                cluster_var='soum'
            )

            if result:
                result['spec_id'] = spec_id
                result['spec_tree_path'] = 'methods/cross_sectional_ols.md#control-sets'
                result['spec_description'] = desc
                results.append(result)

    # 3. CLUSTERING VARIATIONS
    print("Running clustering variations...")
    cluster_variations = [
        ('robust/cluster/none', None, 'No clustering (robust SE)'),
        ('robust/cluster/soum', 'soum', 'Cluster by soum (paper default)'),
        ('robust/cluster/aimag', 'aimag', 'Cluster by aimag (higher level)'),
    ]

    for spec_id, cluster, desc in cluster_variations:
        for outcome, bl_var in main_outcomes:
            if outcome not in df.columns:
                continue

            result = run_regression(
                df, outcome=outcome,
                controls=full_controls,
                baseline_var=bl_var,
                cluster_var=cluster
            )

            if result:
                result['spec_id'] = spec_id
                result['spec_tree_path'] = 'robustness/clustering_variations.md'
                result['spec_description'] = desc
                results.append(result)

    # 4. LEAVE-ONE-OUT
    print("Running leave-one-out specifications...")
    for var_to_drop in base_controls:
        loo_controls = [c for c in base_controls if c != var_to_drop] + aimag_dummies

        for outcome, bl_var in main_outcomes:
            if outcome not in df.columns:
                continue

            result = run_regression(
                df, outcome=outcome,
                controls=loo_controls,
                baseline_var=bl_var,
                cluster_var='soum'
            )

            if result:
                result['spec_id'] = f'robust/loo/drop_{var_to_drop}'
                result['spec_tree_path'] = 'robustness/leave_one_out.md'
                result['spec_description'] = f'Drop {var_to_drop} from controls'
                result['dropped_variable'] = var_to_drop
                results.append(result)

    # 5. SINGLE COVARIATE
    print("Running single covariate specifications...")
    # First: bivariate (treatment only)
    for outcome, bl_var in main_outcomes:
        if outcome not in df.columns:
            continue

        result = run_regression(
            df, outcome=outcome,
            controls=None,
            baseline_var=None,
            cluster_var='soum'
        )

        if result:
            result['spec_id'] = 'robust/single/none'
            result['spec_tree_path'] = 'robustness/single_covariate.md'
            result['spec_description'] = 'Bivariate: treatment only'
            results.append(result)

    # Single covariate additions
    for single_var in base_controls:
        for outcome, bl_var in main_outcomes:
            if outcome not in df.columns:
                continue

            result = run_regression(
                df, outcome=outcome,
                controls=[single_var],
                baseline_var=None,
                cluster_var='soum'
            )

            if result:
                result['spec_id'] = f'robust/single/{single_var}'
                result['spec_tree_path'] = 'robustness/single_covariate.md'
                result['spec_description'] = f'Treatment + {single_var} only'
                results.append(result)

    # 6. FIXED EFFECTS VARIATIONS (using aimag FE)
    print("Running fixed effects variations...")
    # Without aimag dummies (no FE)
    for outcome, bl_var in main_outcomes:
        if outcome not in df.columns:
            continue

        result = run_regression(
            df, outcome=outcome,
            controls=base_controls,
            baseline_var=bl_var,
            cluster_var='soum'
        )

        if result:
            result['spec_id'] = 'ols/fe/none'
            result['spec_tree_path'] = 'methods/cross_sectional_ols.md#fixed-effects'
            result['spec_description'] = 'No region (aimag) fixed effects'
            results.append(result)

    # 7. FUNCTIONAL FORM VARIATIONS
    print("Running functional form variations...")
    # Log outcome for profit/assets
    for outcome, bl_var in [('scaled_profit', 'BLscaled_profit'), ('scaled_assets_all', 'BLscaled_assets_all')]:
        if outcome not in df.columns:
            continue

        # Create log version (add small constant to handle zeros)
        df_temp = df.copy()
        df_temp[f'ln_{outcome}'] = np.log(df_temp[outcome] + 1)

        # Corresponding log baseline if available
        ln_bl_var = None
        if bl_var in df_temp.columns:
            df_temp[f'ln_{bl_var}'] = np.log(df_temp[bl_var].clip(lower=0) + 1)
            ln_bl_var = f'ln_{bl_var}'

        result = run_regression(
            df_temp, outcome=f'ln_{outcome}',
            controls=full_controls,
            baseline_var=ln_bl_var,
            cluster_var='soum'
        )

        if result:
            result['spec_id'] = 'ols/form/log_dep'
            result['spec_tree_path'] = 'robustness/functional_form.md'
            result['spec_description'] = f'Log({outcome}+1)'
            result['original_outcome'] = outcome
            results.append(result)

    # 8. SAMPLE RESTRICTIONS
    print("Running sample restriction specifications...")

    # Winsorize outcomes at 1%/99%
    for outcome, bl_var in main_outcomes:
        if outcome not in df.columns:
            continue

        df_temp = df.copy()
        p01 = df_temp[outcome].quantile(0.01)
        p99 = df_temp[outcome].quantile(0.99)
        df_temp[f'{outcome}_wins'] = df_temp[outcome].clip(lower=p01, upper=p99)

        result = run_regression(
            df_temp, outcome=f'{outcome}_wins',
            controls=full_controls,
            baseline_var=bl_var,
            cluster_var='soum'
        )

        if result:
            result['spec_id'] = 'ols/sample/winsorized'
            result['spec_tree_path'] = 'robustness/sample_restrictions.md'
            result['spec_description'] = f'Winsorized {outcome} at 1%/99%'
            result['original_outcome'] = outcome
            results.append(result)

    # By education subgroups (edusec = 0 vs edusec = 1)
    for edu_val, edu_label in [(0, 'low_edu'), (1, 'high_edu')]:
        for outcome, bl_var in main_outcomes:
            if outcome not in df.columns:
                continue

            df_sub = df[df['edusec'] == edu_val].copy()

            # Remove edusec from controls for this subsample
            sub_controls = [c for c in full_controls if c != 'edusec']

            result = run_regression(
                df_sub, outcome=outcome,
                controls=sub_controls,
                baseline_var=bl_var,
                cluster_var='soum'
            )

            if result:
                result['spec_id'] = f'ols/sample/subgroup_{edu_label}'
                result['spec_tree_path'] = 'robustness/sample_restrictions.md'
                result['spec_description'] = f'Subsample: {edu_label} (edusec={edu_val})'
                results.append(result)

    return results


def results_to_dataframe(results):
    """Convert results list to pandas DataFrame."""
    rows = []
    for r in results:
        row = {
            'spec_id': r['spec_id'],
            'spec_tree_path': r['spec_tree_path'],
            'spec_description': r.get('spec_description', ''),
            'outcome': r['outcome'],
            'treatment_var': r['treatment_var'],
            'treatment_coef': r['treatment_coef'],
            'treatment_se': r['treatment_se'],
            'treatment_pval': r['treatment_pval'],
            'treatment_ci_lower': r['treatment_ci_lower'],
            'treatment_ci_upper': r['treatment_ci_upper'],
            'n_obs': r['n_obs'],
            'r_squared': r['r_squared'],
            'control_mean': r['control_mean'],
            'cluster_var': r['cluster_var'],
            'n_clusters': r.get('n_clusters'),
            'baseline_control': r.get('baseline_control'),
            'dropped_variable': r.get('dropped_variable', ''),
            'coefficient_vector_json': json.dumps(r.get('coefficient_vector', {}))
        }
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("SPECIFICATION SEARCH: 113597-V2 (Mongolia Microfinance RCT)")
    print("=" * 60)

    print("\n1. Loading and preparing data...")
    df = load_and_prepare_data()
    print(f"   Data shape: {df.shape}")
    print(f"   Follow-up observations (group vs control): {len(df[(df['followup']==1) & (df['indiv']!=1)])}")

    print("\n2. Running specification search...")
    results = run_specification_search(df)
    print(f"   Total specifications run: {len(results)}")

    print("\n3. Converting to DataFrame and saving...")
    results_df = results_to_dataframe(results)

    # Save results
    output_file = f"{OUTPUT_PATH}/specification_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"   Results saved to: {output_file}")

    # Print summary
    print("\n4. Summary of results:")
    print("-" * 60)

    # Baseline results
    baseline = results_df[results_df['spec_id'] == 'baseline']
    print("\nBaseline specification results:")
    for _, row in baseline.iterrows():
        sig = '*' if row['treatment_pval'] < 0.05 else ''
        sig = '**' if row['treatment_pval'] < 0.01 else sig
        sig = '***' if row['treatment_pval'] < 0.001 else sig
        print(f"  {row['outcome']}: coef={row['treatment_coef']:.4f} (se={row['treatment_se']:.4f}){sig}, N={row['n_obs']}")

    # Count specifications by type
    print("\nSpecifications by type:")
    spec_counts = results_df['spec_id'].apply(lambda x: x.split('/')[0] if '/' in x else x).value_counts()
    for spec_type, count in spec_counts.items():
        print(f"  {spec_type}: {count}")

    return results_df


if __name__ == "__main__":
    results_df = main()
