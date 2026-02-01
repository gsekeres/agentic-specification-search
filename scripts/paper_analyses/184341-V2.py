"""
Specification Search Analysis for Paper 184341-V2
"Emotional and Behavioral Impacts of Telementoring and Homeschooling Support on Children"
Authors: Hashibul Hassan, Asad Islam, Abu Siddique, Liang Choon Wang

Method: Cross-sectional OLS (Randomized Controlled Trial)
Treatment: Telementoring intervention (treat)
Outcomes: SDQ subscale scores (emotional, conduct, hyperactivity, peer, total difficulties)
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings('ignore')

# Paths
BASE_PATH = Path("/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search")
DATA_PATH = BASE_PATH / "data/downloads/extracted/184341-V2/Detailed Replication Package V3.0/Raw data"
OUTPUT_PATH = BASE_PATH / "data/downloads/extracted/184341-V2"

# Load and merge data (replicating the Stata do file)
def load_and_merge_data():
    """Load and merge all raw datasets to create the analysis dataset."""

    # Load secondary data from pre-pandemic project
    tm_ecd = pd.read_stata(DATA_PATH / "tm_ecd_data.dta")

    # Load sample with treatment assignment
    sample = pd.read_stata(DATA_PATH / "Sample.dta")

    # Merge tm_ecd with sample
    df = tm_ecd.merge(sample, on='CHILD_ID', how='outer')

    # Rename treatment variable
    if 'newtreat' in df.columns:
        df['treat'] = df['newtreat']

    # Load rapid baseline survey
    rapid_baseline = pd.read_excel(DATA_PATH / "Rapid-baseline-data.xlsx")
    rapid_baseline = rapid_baseline.rename(columns={
        'rsq2_1': 'CHILD_ID',
        'rsq2_4': 'pvt_tutor',
        'rsq2_23': 'number_child'
    })

    df = df.merge(rapid_baseline[['CHILD_ID', 'pvt_tutor', 'number_child']],
                  on='CHILD_ID', how='left')

    # Load Endline 1 Parent Survey
    e1_parent = pd.read_excel(DATA_PATH / "Telementoring-E1-Parent-Survey.xlsx")

    # Rename SDQ variables for E1
    sdq_vars = ['tm_sdq2', 'tm_sdq3', 'tm_sdq5', 'tm_sdq6', 'tm_sdq7', 'tm_sdq8',
                'tm_sdq10', 'tm_sdq11', 'tm_sdq12', 'tm_sdq13', 'tm_sdq14', 'tm_sdq15',
                'tm_sdq16', 'tm_sdq18', 'tm_sdq19', 'tm_sdq21', 'tm_sdq22', 'tm_sdq23',
                'tm_sdq24', 'tm_sdq25']

    rename_dict = {var: f'tm_e1_{var.replace("tm_", "")}' for var in sdq_vars if var in e1_parent.columns}
    e1_parent = e1_parent.rename(columns=rename_dict)

    df = df.merge(e1_parent, on='CHILD_ID', how='left')

    # Load Endline 1 Assessment
    e1_assess = pd.read_excel(DATA_PATH / "Telementoring-E1-Assessment.xlsx")
    e1_assess = e1_assess.rename(columns={
        'tm_cogr1_grade': 'tm_e1_grade',
        'tm_cogr1_gen': 'tm_e1_gender'
    })

    df = df.merge(e1_assess[['CHILD_ID', 'tm_e1_grade', 'tm_e1_gender']],
                  on='CHILD_ID', how='left')

    # Generate e1_comp (completed endline 1)
    df['e1_comp'] = ((df['tm_e1_grade'].notna()) & (df['hh_reli'].notna())).astype(int)

    # Load Endline 2 Parent Survey
    e2_parent = pd.read_excel(DATA_PATH / "Telementoring-E2-Parent-Survey.xlsx")

    # Rename SDQ variables for E2 (anu_* -> tm_e2_*)
    e2_rename = {}
    for var in e2_parent.columns:
        if var.startswith('anu_sdq'):
            e2_rename[var] = f'tm_e2_{var.replace("anu_", "")}'
    e2_parent = e2_parent.rename(columns=e2_rename)

    df = df.merge(e2_parent, on='CHILD_ID', how='left')

    # Load Endline 2 Assessment
    e2_assess = pd.read_excel(DATA_PATH / "Telementoring-E2-Assessment.xlsx")
    e2_assess = e2_assess.rename(columns={
        'anu_e1_grade': 'tm_e2_grade',
        'anu_e1_gender': 'tm_e2_gender'
    })

    df = df.merge(e2_assess[['CHILD_ID', 'tm_e2_grade', 'tm_e2_gender']],
                  on='CHILD_ID', how='left')

    # Generate e2_comp (completed endline 2)
    df['e2_comp'] = ((df['tm_e2_grade'].notna()) & (df['tm_e2_sdq2'].notna())).astype(int)

    # Generate gender variable (combine from multiple sources)
    df['gender'] = df['tm_e1_gender']
    df.loc[df['gender'].isna(), 'gender'] = df.loc[df['gender'].isna(), 'tm_e2_gender']
    if 'child_gen' in df.columns:
        df.loc[df['gender'].isna(), 'gender'] = df.loc[df['gender'].isna(), 'child_gen']

    return df


def compute_sdq_scores(df, endline='e1'):
    """Compute SDQ subscale scores following the Stata code logic."""

    prefix = f'tm_{endline}_sdq'

    # Get SDQ item columns
    sdq_items = {
        'emotion': [3, 8, 13, 16, 24],  # somatic, worries, unhappy, clingy, afraid
        'conduct': [5, 7, 12, 18, 22],   # tantrum, obeys(r), fights, lies, steals
        'hyper': [2, 10, 15, 21, 25],    # restless, fidgety, distrac, reflect(r), attends(r)
        'peer': [6, 11, 14, 19, 23]      # loner, friend(r), popular(r), bullied, oldbest
    }

    # Items that need reverse coding (0->2, 1->1, 2->0)
    reverse_items = {7, 11, 14, 21, 25}

    # Create temporary copies of items
    items_dict = {}
    for item_num in range(2, 26):
        if item_num in [1, 4, 9, 17, 20]:  # Items not used
            continue
        col = f'{prefix}{item_num}'
        if col in df.columns:
            if item_num in reverse_items:
                # Reverse code
                items_dict[item_num] = df[col].replace({0: 2, 1: 1, 2: 0})
            else:
                items_dict[item_num] = df[col]

    # Compute subscale scores
    for subscale, items in sdq_items.items():
        cols = []
        for item in items:
            if item in items_dict:
                cols.append(items_dict[item])

        if len(cols) > 0:
            # Stack columns
            item_df = pd.concat(cols, axis=1)
            # Count non-missing
            n_valid = item_df.notna().sum(axis=1)
            # Compute mean only if more than 2 valid responses
            mean_score = item_df.mean(axis=1, skipna=True)
            # Multiply by 5 and round
            score = np.where(n_valid > 2, np.round(mean_score * 5), np.nan)
            df[f'sdq_{subscale}_{endline}'] = score

    # Compute total difficulties score
    subscale_cols = [f'sdq_emotion_{endline}', f'sdq_conduct_{endline}',
                     f'sdq_hyper_{endline}', f'sdq_peer_{endline}']

    existing_cols = [c for c in subscale_cols if c in df.columns]
    if len(existing_cols) == 4:
        df[f'sdq_totdiff_{endline}'] = df[existing_cols].sum(axis=1, skipna=False)

    return df


def compute_control_variables(df):
    """Compute control variables following the Stata code."""

    # Child age as of 1 Sep 2020
    if 'child_dob' in df.columns:
        target_date = pd.Timestamp('2020-09-01')
        df['child_age'] = (target_date - pd.to_datetime(df['child_dob'])).dt.days / 365.25

    # Father's education in years
    edu_mapping = {1: 0, 10: 0, 2: 2, 3: 5, 4: 8, 5: 10, 6: 11, 7: 12, 8: 14, 9: 18}
    if 'fathers_edu' in df.columns:
        df['FEdun'] = df['fathers_edu'].map(edu_mapping)

    # Mother's education in years
    if 'mothers_edu' in df.columns:
        df['MEdun'] = df['mothers_edu'].map(edu_mapping)

    # Total family income
    if 'fathers_income' in df.columns and 'mothers_income' in df.columns:
        df['total_income'] = df['fathers_income'].fillna(0) + df['mothers_income'].fillna(0)

    # Number of children
    if 'number_child' in df.columns:
        df['children_no'] = 1 + df['number_child'].fillna(0)

    # Religion dummy (Islam = 1)
    if 'hh_reli' in df.columns:
        df['reli_dummy'] = (df['hh_reli'] == 1).astype(float)

    return df


def run_regression(df, outcome, treatment, controls, fe_vars=None, sample_filter=None,
                   se_type='HC1', spec_id='baseline', spec_tree_path='custom'):
    """Run OLS regression and return results in standard format."""

    # Apply sample filter
    if sample_filter is not None:
        df_reg = df[sample_filter].copy()
    else:
        df_reg = df.copy()

    # Build control list (exclude any controls that are all NaN)
    valid_controls = []
    for c in controls:
        if c in df_reg.columns and df_reg[c].notna().any():
            valid_controls.append(c)

    # Create formula
    if fe_vars:
        # Include fixed effects as dummies
        fe_dummies = []
        for fe in fe_vars:
            if fe in df_reg.columns:
                fe_dummies.append(f'C({fe})')
        if fe_dummies:
            formula = f'{outcome} ~ {treatment} + {" + ".join(valid_controls)} + {" + ".join(fe_dummies)}'
        else:
            formula = f'{outcome} ~ {treatment} + {" + ".join(valid_controls)}'
    else:
        if valid_controls:
            formula = f'{outcome} ~ {treatment} + {" + ".join(valid_controls)}'
        else:
            formula = f'{outcome} ~ {treatment}'

    # Drop missing observations
    vars_needed = [outcome, treatment] + valid_controls
    if fe_vars:
        vars_needed += [v for v in fe_vars if v in df_reg.columns]

    df_reg = df_reg.dropna(subset=[v for v in vars_needed if v in df_reg.columns])

    if len(df_reg) < 30:
        return None

    try:
        # Run regression with robust standard errors
        model = smf.ols(formula, data=df_reg)
        result = model.fit(cov_type=se_type)

        # Extract treatment coefficient
        coef = result.params.get(treatment, np.nan)
        se = result.bse.get(treatment, np.nan)
        pval = result.pvalues.get(treatment, np.nan)

        # Confidence interval
        ci = result.conf_int().loc[treatment] if treatment in result.conf_int().index else [np.nan, np.nan]

        # Control means
        control_mean = df_reg.loc[df_reg[treatment] == 0, outcome].mean()
        control_se = df_reg.loc[df_reg[treatment] == 0, outcome].std() / np.sqrt((df_reg[treatment] == 0).sum())

        # Build coefficient vector
        coef_vector = {
            'treatment': {
                'var': treatment,
                'coef': float(coef),
                'se': float(se),
                'pval': float(pval),
                'ci_lower': float(ci[0]),
                'ci_upper': float(ci[1])
            },
            'controls': [],
            'n_obs': int(result.nobs),
            'r_squared': float(result.rsquared),
            'adj_r_squared': float(result.rsquared_adj),
            'f_stat': float(result.fvalue) if hasattr(result, 'fvalue') and result.fvalue is not None else None,
            'f_pval': float(result.f_pvalue) if hasattr(result, 'f_pvalue') and result.f_pvalue is not None else None,
            'control_mean': float(control_mean),
            'control_se': float(control_se)
        }

        # Add control coefficients
        for c in valid_controls:
            if c in result.params.index:
                coef_vector['controls'].append({
                    'var': c,
                    'coef': float(result.params[c]),
                    'se': float(result.bse[c]),
                    'pval': float(result.pvalues[c])
                })

        return {
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome': outcome,
            'treatment': treatment,
            'coef': float(coef),
            'se': float(se),
            'pval': float(pval),
            'ci_lower': float(ci[0]),
            'ci_upper': float(ci[1]),
            'n_obs': int(result.nobs),
            'r_squared': float(result.rsquared),
            'adj_r_squared': float(result.rsquared_adj),
            'control_mean': float(control_mean),
            'control_se': float(control_se),
            'se_type': se_type,
            'controls_used': valid_controls,
            'fe_vars': fe_vars if fe_vars else [],
            'coefficient_vector_json': json.dumps(coef_vector)
        }

    except Exception as e:
        print(f"Error in {spec_id}: {e}")
        return None


def main():
    print("Loading and merging data...")
    df = load_and_merge_data()

    print("Computing SDQ scores...")
    df = compute_sdq_scores(df, 'e1')
    df = compute_sdq_scores(df, 'e2')

    print("Computing control variables...")
    df = compute_control_variables(df)

    print(f"Dataset has {len(df)} observations")
    print(f"Endline 1 completed: {df['e1_comp'].sum()}")
    print(f"Endline 2 completed: {df['e2_comp'].sum()}")

    # Define variables
    outcomes_e1 = ['sdq_totdiff_e1', 'sdq_emotion_e1', 'sdq_conduct_e1', 'sdq_hyper_e1', 'sdq_peer_e1']
    outcomes_e2 = ['sdq_totdiff_e2', 'sdq_emotion_e2', 'sdq_conduct_e2', 'sdq_hyper_e2', 'sdq_peer_e2']

    control_list = ['child_age', 'gender', 'baseline_literacy_score', 'baseline_numeracy_score',
                    'pvt_tutor', 'birth_order', 'children_no', 'FEdun', 'MEdun', 'total_income', 'reli_dummy']

    fe_vars_e1 = ['tm_e1_grade', 'union_code']
    fe_vars_e2 = ['tm_e2_grade', 'union_code']

    results = []

    # ==========================================================================
    # BASELINE SPECIFICATIONS
    # ==========================================================================
    print("\n--- Running Baseline Specifications ---")

    # Endline 1 outcomes
    for outcome in outcomes_e1:
        res = run_regression(
            df, outcome, 'treat', control_list, fe_vars_e1,
            sample_filter=df['e1_comp'] == 1,
            spec_id='baseline',
            spec_tree_path='methods/cross_sectional_ols.md#baseline'
        )
        if res:
            res['endline'] = 'e1'
            results.append(res)
            print(f"  {outcome}: coef={res['coef']:.3f}, se={res['se']:.3f}, p={res['pval']:.4f}, n={res['n_obs']}")

    # Endline 2 outcomes
    for outcome in outcomes_e2:
        res = run_regression(
            df, outcome, 'treat', control_list, fe_vars_e2,
            sample_filter=df['e2_comp'] == 1,
            spec_id='baseline',
            spec_tree_path='methods/cross_sectional_ols.md#baseline'
        )
        if res:
            res['endline'] = 'e2'
            results.append(res)
            print(f"  {outcome}: coef={res['coef']:.3f}, se={res['se']:.3f}, p={res['pval']:.4f}, n={res['n_obs']}")

    # ==========================================================================
    # STANDARD ERRORS VARIATIONS
    # ==========================================================================
    print("\n--- Running SE Variations ---")

    primary_outcome_e1 = 'sdq_totdiff_e1'
    primary_outcome_e2 = 'sdq_totdiff_e2'

    se_types = [
        ('robust/se/hc1', 'HC1', 'robustness/clustering_variations.md#single-level-clustering'),
        ('robust/se/hc2', 'HC2', 'robustness/clustering_variations.md#single-level-clustering'),
        ('robust/se/hc3', 'HC3', 'robustness/clustering_variations.md#single-level-clustering'),
    ]

    for spec_id, se_type, tree_path in se_types:
        for outcome, fe_vars, sample_filter, endline in [
            (primary_outcome_e1, fe_vars_e1, df['e1_comp'] == 1, 'e1'),
            (primary_outcome_e2, fe_vars_e2, df['e2_comp'] == 1, 'e2')
        ]:
            res = run_regression(df, outcome, 'treat', control_list, fe_vars, sample_filter,
                                se_type=se_type, spec_id=spec_id, spec_tree_path=tree_path)
            if res:
                res['endline'] = endline
                results.append(res)
                print(f"  {spec_id} ({endline}): se={res['se']:.3f}, p={res['pval']:.4f}")

    # Clustered by union_code
    for outcome, fe_vars, sample_filter, endline in [
        (primary_outcome_e1, fe_vars_e1, df['e1_comp'] == 1, 'e1'),
        (primary_outcome_e2, fe_vars_e2, df['e2_comp'] == 1, 'e2')
    ]:
        res = run_regression(df, outcome, 'treat', control_list, fe_vars, sample_filter,
                            se_type='cluster', spec_id='robust/cluster/union',
                            spec_tree_path='robustness/clustering_variations.md#single-level-clustering')
        if res:
            res['endline'] = endline
            results.append(res)

    # ==========================================================================
    # CONTROL SET VARIATIONS
    # ==========================================================================
    print("\n--- Running Control Set Variations ---")

    # No controls
    for outcome, fe_vars, sample_filter, endline in [
        (primary_outcome_e1, fe_vars_e1, df['e1_comp'] == 1, 'e1'),
        (primary_outcome_e2, fe_vars_e2, df['e2_comp'] == 1, 'e2')
    ]:
        res = run_regression(df, outcome, 'treat', [], fe_vars, sample_filter,
                            spec_id='ols/controls/none',
                            spec_tree_path='methods/cross_sectional_ols.md#control-sets')
        if res:
            res['endline'] = endline
            results.append(res)
            print(f"  No controls ({endline}): coef={res['coef']:.3f}, p={res['pval']:.4f}")

    # Demographic controls only
    demo_controls = ['child_age', 'gender', 'birth_order', 'children_no']
    for outcome, fe_vars, sample_filter, endline in [
        (primary_outcome_e1, fe_vars_e1, df['e1_comp'] == 1, 'e1'),
        (primary_outcome_e2, fe_vars_e2, df['e2_comp'] == 1, 'e2')
    ]:
        res = run_regression(df, outcome, 'treat', demo_controls, fe_vars, sample_filter,
                            spec_id='ols/controls/demographics',
                            spec_tree_path='methods/cross_sectional_ols.md#control-sets')
        if res:
            res['endline'] = endline
            results.append(res)
            print(f"  Demographics only ({endline}): coef={res['coef']:.3f}, p={res['pval']:.4f}")

    # Without fixed effects
    for outcome, sample_filter, endline in [
        (primary_outcome_e1, df['e1_comp'] == 1, 'e1'),
        (primary_outcome_e2, df['e2_comp'] == 1, 'e2')
    ]:
        res = run_regression(df, outcome, 'treat', control_list, None, sample_filter,
                            spec_id='ols/fe/none',
                            spec_tree_path='methods/cross_sectional_ols.md#fixed-effects')
        if res:
            res['endline'] = endline
            results.append(res)
            print(f"  No FE ({endline}): coef={res['coef']:.3f}, p={res['pval']:.4f}")

    # ==========================================================================
    # LEAVE-ONE-OUT ROBUSTNESS
    # ==========================================================================
    print("\n--- Running Leave-One-Out ---")

    for control in control_list:
        loo_controls = [c for c in control_list if c != control]

        for outcome, fe_vars, sample_filter, endline in [
            (primary_outcome_e1, fe_vars_e1, df['e1_comp'] == 1, 'e1'),
            (primary_outcome_e2, fe_vars_e2, df['e2_comp'] == 1, 'e2')
        ]:
            res = run_regression(df, outcome, 'treat', loo_controls, fe_vars, sample_filter,
                                spec_id=f'robust/loo/drop_{control}',
                                spec_tree_path='robustness/leave_one_out.md')
            if res:
                res['endline'] = endline
                res['dropped_variable'] = control
                results.append(res)

        print(f"  Dropped {control}")

    # ==========================================================================
    # SINGLE COVARIATE
    # ==========================================================================
    print("\n--- Running Single Covariate ---")

    # Bivariate (no controls, no FE)
    for outcome, sample_filter, endline in [
        (primary_outcome_e1, df['e1_comp'] == 1, 'e1'),
        (primary_outcome_e2, df['e2_comp'] == 1, 'e2')
    ]:
        res = run_regression(df, outcome, 'treat', [], None, sample_filter,
                            spec_id='robust/single/none',
                            spec_tree_path='robustness/single_covariate.md')
        if res:
            res['endline'] = endline
            results.append(res)
            print(f"  Bivariate ({endline}): coef={res['coef']:.3f}, p={res['pval']:.4f}")

    # Single covariate specifications
    for control in control_list:
        for outcome, fe_vars, sample_filter, endline in [
            (primary_outcome_e1, fe_vars_e1, df['e1_comp'] == 1, 'e1'),
            (primary_outcome_e2, fe_vars_e2, df['e2_comp'] == 1, 'e2')
        ]:
            res = run_regression(df, outcome, 'treat', [control], fe_vars, sample_filter,
                                spec_id=f'robust/single/{control}',
                                spec_tree_path='robustness/single_covariate.md')
            if res:
                res['endline'] = endline
                res['single_control'] = control
                results.append(res)

    # ==========================================================================
    # SAMPLE RESTRICTIONS
    # ==========================================================================
    print("\n--- Running Sample Restrictions ---")

    # Gender subgroups
    for outcome, fe_vars, sample_filter, endline in [
        (primary_outcome_e1, fe_vars_e1, df['e1_comp'] == 1, 'e1'),
        (primary_outcome_e2, fe_vars_e2, df['e2_comp'] == 1, 'e2')
    ]:
        # Boys only (gender == 1)
        res = run_regression(df, outcome, 'treat',
                            [c for c in control_list if c != 'gender'],
                            fe_vars,
                            sample_filter & (df['gender'] == 1),
                            spec_id='robust/sample/male_only',
                            spec_tree_path='robustness/sample_restrictions.md#demographic-subgroups')
        if res:
            res['endline'] = endline
            results.append(res)
            print(f"  Boys only ({endline}): coef={res['coef']:.3f}, p={res['pval']:.4f}, n={res['n_obs']}")

        # Girls only (gender == 0)
        res = run_regression(df, outcome, 'treat',
                            [c for c in control_list if c != 'gender'],
                            fe_vars,
                            sample_filter & (df['gender'] == 0),
                            spec_id='robust/sample/female_only',
                            spec_tree_path='robustness/sample_restrictions.md#demographic-subgroups')
        if res:
            res['endline'] = endline
            results.append(res)
            print(f"  Girls only ({endline}): coef={res['coef']:.3f}, p={res['pval']:.4f}, n={res['n_obs']}")

    # Age subgroups (young/old based on median)
    if 'child_age' in df.columns and df['child_age'].notna().any():
        median_age = df['child_age'].median()

        for outcome, fe_vars, sample_filter, endline in [
            (primary_outcome_e1, fe_vars_e1, df['e1_comp'] == 1, 'e1'),
            (primary_outcome_e2, fe_vars_e2, df['e2_comp'] == 1, 'e2')
        ]:
            # Young children
            res = run_regression(df, outcome, 'treat',
                                [c for c in control_list if c != 'child_age'],
                                fe_vars,
                                sample_filter & (df['child_age'] <= median_age),
                                spec_id='robust/sample/young',
                                spec_tree_path='robustness/sample_restrictions.md#demographic-subgroups')
            if res:
                res['endline'] = endline
                results.append(res)

            # Old children
            res = run_regression(df, outcome, 'treat',
                                [c for c in control_list if c != 'child_age'],
                                fe_vars,
                                sample_filter & (df['child_age'] > median_age),
                                spec_id='robust/sample/old',
                                spec_tree_path='robustness/sample_restrictions.md#demographic-subgroups')
            if res:
                res['endline'] = endline
                results.append(res)

    # Income subgroups (high/low based on median)
    if 'total_income' in df.columns and df['total_income'].notna().any():
        median_income = df['total_income'].median()

        for outcome, fe_vars, sample_filter, endline in [
            (primary_outcome_e1, fe_vars_e1, df['e1_comp'] == 1, 'e1'),
            (primary_outcome_e2, fe_vars_e2, df['e2_comp'] == 1, 'e2')
        ]:
            # Low income
            res = run_regression(df, outcome, 'treat',
                                [c for c in control_list if c != 'total_income'],
                                fe_vars,
                                sample_filter & (df['total_income'] <= median_income),
                                spec_id='robust/sample/low_income',
                                spec_tree_path='robustness/sample_restrictions.md#demographic-subgroups')
            if res:
                res['endline'] = endline
                results.append(res)

            # High income
            res = run_regression(df, outcome, 'treat',
                                [c for c in control_list if c != 'total_income'],
                                fe_vars,
                                sample_filter & (df['total_income'] > median_income),
                                spec_id='robust/sample/high_income',
                                spec_tree_path='robustness/sample_restrictions.md#demographic-subgroups')
            if res:
                res['endline'] = endline
                results.append(res)

    # Trimmed sample (drop extreme SDQ scores)
    for outcome, fe_vars, sample_filter, endline in [
        (primary_outcome_e1, fe_vars_e1, df['e1_comp'] == 1, 'e1'),
        (primary_outcome_e2, fe_vars_e2, df['e2_comp'] == 1, 'e2')
    ]:
        if outcome in df.columns:
            q01 = df.loc[sample_filter, outcome].quantile(0.01)
            q99 = df.loc[sample_filter, outcome].quantile(0.99)
            trim_filter = sample_filter & (df[outcome] >= q01) & (df[outcome] <= q99)

            res = run_regression(df, outcome, 'treat', control_list, fe_vars, trim_filter,
                                spec_id='robust/sample/trim_1pct',
                                spec_tree_path='robustness/sample_restrictions.md#outlier-handling')
            if res:
                res['endline'] = endline
                results.append(res)
                print(f"  Trimmed 1% ({endline}): coef={res['coef']:.3f}, n={res['n_obs']}")

    # ==========================================================================
    # FUNCTIONAL FORM VARIATIONS
    # ==========================================================================
    print("\n--- Running Functional Form Variations ---")

    # Standardized outcome
    for outcome, fe_vars, sample_filter, endline in [
        (primary_outcome_e1, fe_vars_e1, df['e1_comp'] == 1, 'e1'),
        (primary_outcome_e2, fe_vars_e2, df['e2_comp'] == 1, 'e2')
    ]:
        df_temp = df.copy()
        if outcome in df_temp.columns:
            outcome_mean = df_temp.loc[sample_filter, outcome].mean()
            outcome_std = df_temp.loc[sample_filter, outcome].std()
            if outcome_std > 0:
                df_temp[f'{outcome}_std'] = (df_temp[outcome] - outcome_mean) / outcome_std

                res = run_regression(df_temp, f'{outcome}_std', 'treat', control_list, fe_vars, sample_filter,
                                    spec_id='robust/form/y_standardized',
                                    spec_tree_path='robustness/functional_form.md#outcome-variable-transformations')
                if res:
                    res['endline'] = endline
                    results.append(res)
                    print(f"  Standardized ({endline}): coef={res['coef']:.3f}, p={res['pval']:.4f}")

    # ==========================================================================
    # INTERACTION EFFECTS (Gender heterogeneity as in paper)
    # ==========================================================================
    print("\n--- Running Interaction Effects ---")

    for outcome, fe_vars, sample_filter, endline in [
        (primary_outcome_e1, fe_vars_e1, df['e1_comp'] == 1, 'e1'),
        (primary_outcome_e2, fe_vars_e2, df['e2_comp'] == 1, 'e2')
    ]:
        # Create interaction term
        df_temp = df.copy()
        # Convert to numeric to handle any non-numeric types
        df_temp['treat_numeric'] = pd.to_numeric(df_temp['treat'], errors='coerce')
        df_temp['gender_numeric'] = pd.to_numeric(df_temp['gender'], errors='coerce')
        df_temp['treat_x_gender'] = df_temp['treat_numeric'] * df_temp['gender_numeric']

        # Run with interaction
        interact_controls = [c for c in control_list if c != 'gender'] + ['gender_numeric', 'treat_x_gender']

        res = run_regression(df_temp, outcome, 'treat_numeric', interact_controls, fe_vars, sample_filter,
                            spec_id='ols/interact/gender',
                            spec_tree_path='methods/cross_sectional_ols.md#interaction-effects')
        if res:
            res['endline'] = endline
            results.append(res)
            print(f"  Gender interaction ({endline}): coef={res['coef']:.3f}")

    # ==========================================================================
    # ALL OUTCOMES (for completeness)
    # ==========================================================================
    print("\n--- Running All Outcome Variations ---")

    # Run baseline for all outcomes, not just the primary
    all_outcomes = outcomes_e1 + outcomes_e2

    # Already done in baseline section above

    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================
    print("\n--- Saving Results ---")

    results_df = pd.DataFrame(results)

    # Save to CSV
    output_file = OUTPUT_PATH / "specification_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Saved {len(results_df)} specifications to {output_file}")

    # Summary statistics
    print(f"\nTotal specifications run: {len(results_df)}")
    print(f"Unique spec_ids: {results_df['spec_id'].nunique()}")

    # Check for significant results
    sig_05 = (results_df['pval'] < 0.05).sum()
    sig_01 = (results_df['pval'] < 0.01).sum()
    print(f"Significant at 5%: {sig_05} ({100*sig_05/len(results_df):.1f}%)")
    print(f"Significant at 1%: {sig_01} ({100*sig_01/len(results_df):.1f}%)")

    return results_df


if __name__ == "__main__":
    results_df = main()
