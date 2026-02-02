"""
Specification Search for Paper 116433-V1
"Referrals: Peer Screening and Enforcement in a Consumer Credit Field Experiment"
By Bryan, Karlan, and Zinman

Method: cross_sectional_ols (2x2 factorial RCT)
Method Tree Path: specification_tree/methods/cross_sectional_ols.md

This script runs 50+ specifications following the i4r methodology.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_PATH = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/116433-V1/20130234_Data/Referrals_data.dta'
OUTPUT_PATH = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/116433-V1/specification_results.csv'

# Paper metadata
PAPER_ID = '116433-V1'
JOURNAL = 'AER'
PAPER_TITLE = 'Referrals: Peer Screening and Enforcement in a Consumer Credit Field Experiment'

# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_and_prepare_data():
    """Load data and create all necessary variables following original do-files."""
    df = pd.read_stata(DATA_PATH)

    # Convert treatment to numeric codes
    treatment_map = {
        'Approved': 1,          # app - only referrer gets reward if approved
        'Approved + Repay': 2,  # apprep - referrer gets reward if approved AND repay
        'Repay': 3,             # rep - referrer gets reward if repay
        'Repay --> Approved': 4 # repapp - referrer gets reward if repay, THEN approved
    }
    df['treatment_num'] = df['treatment'].map(treatment_map)

    # Create enforcement and selection dummies (per Table 5/6 do-files)
    df['enforcement'] = ((df['treatment_num'] == 2) | (df['treatment_num'] == 3)).astype(int)
    df['selection'] = ((df['treatment_num'] == 3) | (df['treatment_num'] == 4)).astype(int)
    df['enforcement_x_selection'] = df['enforcement'] * df['selection']

    # Create treatment arm dummies (per Table 2 do-file)
    df['app'] = (df['treatment_num'] == 1).astype(int)
    df['apprep'] = (df['treatment_num'] == 2).astype(int)
    df['rep'] = (df['treatment_num'] == 3).astype(int)
    df['repapp'] = (df['treatment_num'] == 4).astype(int)

    # Create referrer control variables (per Table 2/TableA1 do-files)
    df['referrer_itcmissing'] = ((df['referrer_itcscore'] == 0) |
                                  (df['referrer_itcscore'].isna())).astype(int)
    df['referrer_itcscore'] = df['referrer_itcscore'].fillna(0)

    # Job type dummies for referrer
    df['referrer_gov'] = (df['referrer_jobtype'] == 'Government').astype(int)
    df['referrer_clean'] = (df['referrer_jobtype'] == 'Cleaner').astype(int) if 'Cleaner' in df['referrer_jobtype'].cat.categories else 0
    df['referrer_secure'] = (df['referrer_jobtype'] == 'Security').astype(int) if 'Security' in df['referrer_jobtype'].cat.categories else 0
    df['referrer_retail'] = (df['referrer_jobtype'] == 'Retail').astype(int) if 'Retail' in df['referrer_jobtype'].cat.categories else 0
    df['referrer_it'] = (df['referrer_jobtype'] == 'IT').astype(int) if 'IT' in df['referrer_jobtype'].cat.categories else 0
    df['referrer_ag'] = (df['referrer_jobtype'] == 'Agriculture').astype(int) if 'Agriculture' in df['referrer_jobtype'].cat.categories else 0

    # Education and salary for referrer
    referrer_edu_codes = {cat: i for i, cat in enumerate(df['referrer_highedu'].cat.categories)}
    df['referrer_highedu_num'] = df['referrer_highedu'].map(referrer_edu_codes)
    df['referrer_education'] = (df['referrer_highedu_num'] >= 2).astype(int) if df['referrer_highedu_num'].notna().any() else 0

    salary_codes = {cat: i for i, cat in enumerate(df['referrer_salaryoccurence'].cat.categories)}
    df['referrer_salaryocc_num'] = df['referrer_salaryoccurence'].map(salary_codes)
    df['referrer_salaryM'] = (df['referrer_salaryocc_num'] == 2).astype(int) if df['referrer_salaryocc_num'].notna().any() else 0

    # Scaled variables
    df['referrer_disposableincome000'] = df['referrer_disposableincome'] / 1000
    df['referrer_requestedamount000'] = df['referrer_requestedamount'] / 1000

    # Convert referrer_female to numeric
    if df['referrer_female'].dtype.name == 'category':
        female_map = {cat: 1 if 'female' in str(cat).lower() or 'yes' in str(cat).lower() or cat == 1 else 0
                      for cat in df['referrer_female'].cat.categories}
        df['referrer_female_num'] = df['referrer_female'].map(female_map)
    else:
        df['referrer_female_num'] = df['referrer_female']

    # Create referred control variables (per Table 3/TableA2 do-files)
    df['referred_itcmissing'] = df['referred_itcscore'].isna().astype(int)
    df['referred_itcscore'] = df['referred_itcscore'].fillna(0)

    # Education and salary for referred
    if df['referred_highedu'].dtype.name == 'category':
        referred_edu_codes = {cat: i for i, cat in enumerate(df['referred_highedu'].cat.categories)}
        df['referred_highedu_num'] = df['referred_highedu'].map(referred_edu_codes)
    else:
        df['referred_highedu_num'] = df['referred_highedu']
    df['referred_education'] = (df['referred_highedu_num'] >= 2).astype(int) if df['referred_highedu_num'].notna().any() else 0

    if df['referred_salaryoccurence'].dtype.name == 'category':
        ref_salary_codes = {cat: i for i, cat in enumerate(df['referred_salaryoccurence'].cat.categories)}
        df['referred_salaryocc_num'] = df['referred_salaryoccurence'].map(ref_salary_codes)
    else:
        df['referred_salaryocc_num'] = df['referred_salaryoccurence']
    df['referred_salaryM'] = (df['referred_salaryocc_num'] == 2).astype(int) if df['referred_salaryocc_num'].notna().any() else 0

    # Scaled variables for referred
    df['referred_disposableincome000'] = df['referred_disposableincome'] / 1000
    df['referred_requestedamount000'] = df['referred_requestedamount'] / 1000

    # Convert referred_female to numeric
    if df['referred_female'].dtype.name == 'category':
        female_map_ref = {cat: 1 if 'female' in str(cat).lower() or 'yes' in str(cat).lower() or cat == 1 else 0
                         for cat in df['referred_female'].cat.categories}
        df['referred_female_num'] = df['referred_female'].map(female_map_ref)
    else:
        df['referred_female_num'] = df['referred_female']

    # Branch dummies
    if df['referrer_branchname'].dtype.name == 'category':
        branch_dummies = pd.get_dummies(df['referrer_branchname'], prefix='branch_ref', drop_first=True)
        df = pd.concat([df, branch_dummies], axis=1)

    # Relationship type
    if df['relationship'].dtype.name == 'category':
        rel_map = {cat: i for i, cat in enumerate(df['relationship'].cat.categories)}
        df['relationship_num'] = df['relationship'].map(rel_map)

    return df

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def run_ols(df, formula, vcov_type='HC1', cluster_var=None):
    """Run OLS regression with robust or clustered standard errors."""
    try:
        df_clean = df.dropna(subset=[v.strip() for v in formula.split('~')[0].split('+')[0].strip().split()])

        model = smf.ols(formula, data=df_clean)

        if cluster_var and cluster_var in df_clean.columns:
            # Clustered SE
            result = model.fit(cov_type='cluster',
                              cov_kwds={'groups': df_clean[cluster_var]})
        else:
            # Robust SE
            result = model.fit(cov_type=vcov_type)

        return result
    except Exception as e:
        print(f"Error in regression: {e}")
        return None

def extract_results(result, treatment_var, spec_id, spec_tree_path, outcome_var,
                   sample_desc='', fixed_effects='None', controls_desc='',
                   cluster_var='None', model_type='OLS'):
    """Extract results from statsmodels regression into standard format."""
    if result is None:
        return None

    try:
        # Get treatment coefficient
        if treatment_var not in result.params.index:
            print(f"Treatment var {treatment_var} not found in results")
            return None

        coef = result.params[treatment_var]
        se = result.bse[treatment_var]
        t_stat = result.tvalues[treatment_var]
        p_val = result.pvalues[treatment_var]
        ci = result.conf_int().loc[treatment_var]

        # Build coefficient vector JSON
        coef_vector = {
            'treatment': {
                'var': treatment_var,
                'coef': float(coef),
                'se': float(se),
                'pval': float(p_val)
            },
            'controls': [],
            'fixed_effects': fixed_effects.split(', ') if fixed_effects != 'None' else [],
            'diagnostics': {
                'r_squared': float(result.rsquared),
                'adj_r_squared': float(result.rsquared_adj),
                'f_stat': float(result.fvalue) if hasattr(result, 'fvalue') and result.fvalue is not None else None,
                'f_pval': float(result.f_pvalue) if hasattr(result, 'f_pvalue') and result.f_pvalue is not None else None
            }
        }

        # Add control coefficients
        for var in result.params.index:
            if var != treatment_var and var != 'Intercept':
                coef_vector['controls'].append({
                    'var': var,
                    'coef': float(result.params[var]),
                    'se': float(result.bse[var]),
                    'pval': float(result.pvalues[var])
                })

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
            't_stat': float(t_stat),
            'p_value': float(p_val),
            'ci_lower': float(ci[0]),
            'ci_upper': float(ci[1]),
            'n_obs': int(result.nobs),
            'r_squared': float(result.rsquared),
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects,
            'controls_desc': controls_desc,
            'cluster_var': cluster_var,
            'model_type': model_type
        }
    except Exception as e:
        print(f"Error extracting results: {e}")
        return None

# =============================================================================
# SPECIFICATION SEARCH
# =============================================================================

def run_specification_search():
    """Run all specifications for the paper."""

    print("Loading and preparing data...")
    df = load_and_prepare_data()

    results = []

    # Define control variables
    baseline_controls = ['referrer_female_num', 'referrer_age', 'referrer_education',
                        'referrer_salaryM', 'referrer_disposableincome000',
                        'referrer_applicationscore', 'referrer_itcscore',
                        'referrer_itcmissing', 'referrer_requestedamount000',
                        'referrer_requestedterm']

    # Clean control variable names (remove any that don't exist or have too many NaNs)
    baseline_controls = [c for c in baseline_controls if c in df.columns and df[c].notna().sum() > 100]

    # Primary outcomes (loan performance)
    outcomes = ['repaid', 'charged_off', 'interest', 'portion']

    # Treatment variables
    treatment_vars = ['enforcement', 'selection']

    print(f"Running specification search with {len(baseline_controls)} baseline controls...")
    print(f"Baseline controls: {baseline_controls}")

    # =========================================================================
    # SECTION 1: BASELINE SPECIFICATIONS (Table 5/6 replication)
    # =========================================================================
    print("\n--- Running Baseline Specifications ---")

    for outcome in outcomes:
        df_outcome = df[df[outcome].notna()].copy()
        if len(df_outcome) < 50:
            print(f"Skipping {outcome}: insufficient observations ({len(df_outcome)})")
            continue

        # Baseline: enforcement + selection (Table 6 style)
        formula = f'{outcome} ~ enforcement + selection'
        result = run_ols(df_outcome, formula, vcov_type='HC1')

        for treat_var in treatment_vars:
            res = extract_results(
                result, treat_var,
                spec_id=f'baseline_{outcome}_{treat_var}',
                spec_tree_path='methods/cross_sectional_ols.md#baseline',
                outcome_var=outcome,
                sample_desc='Full sample with loan outcomes',
                controls_desc='None',
                cluster_var='None',
                model_type='OLS'
            )
            if res:
                results.append(res)

    # =========================================================================
    # SECTION 2: CONTROL VARIATIONS
    # =========================================================================
    print("\n--- Running Control Variations ---")

    # 2.1 No controls vs full controls
    control_sets = {
        'none': [],
        'demographics': [c for c in ['referrer_female_num', 'referrer_age', 'referrer_education'] if c in baseline_controls],
        'financial': [c for c in ['referrer_disposableincome000', 'referrer_requestedamount000',
                                  'referrer_applicationscore', 'referrer_itcscore'] if c in baseline_controls],
        'full': baseline_controls
    }

    for outcome in outcomes:
        df_outcome = df[df[outcome].notna()].copy()
        if len(df_outcome) < 50:
            continue

        for control_name, controls in control_sets.items():
            if controls:
                controls_str = ' + '.join(controls)
                formula = f'{outcome} ~ enforcement + selection + {controls_str}'
            else:
                formula = f'{outcome} ~ enforcement + selection'

            result = run_ols(df_outcome, formula, vcov_type='HC1')

            for treat_var in treatment_vars:
                res = extract_results(
                    result, treat_var,
                    spec_id=f'ols/controls/{control_name}_{outcome}_{treat_var}',
                    spec_tree_path='methods/cross_sectional_ols.md#control-sets',
                    outcome_var=outcome,
                    sample_desc='Full sample with loan outcomes',
                    controls_desc=control_name,
                    cluster_var='None',
                    model_type='OLS'
                )
                if res:
                    results.append(res)

    # 2.2 Leave-one-out controls
    print("\n--- Running Leave-One-Out Controls ---")

    for outcome in ['repaid', 'charged_off']:  # Focus on main outcomes
        df_outcome = df[df[outcome].notna()].copy()
        if len(df_outcome) < 50 or len(baseline_controls) < 2:
            continue

        for drop_control in baseline_controls:
            remaining_controls = [c for c in baseline_controls if c != drop_control]
            if remaining_controls:
                controls_str = ' + '.join(remaining_controls)
                formula = f'{outcome} ~ enforcement + selection + {controls_str}'
            else:
                formula = f'{outcome} ~ enforcement + selection'

            result = run_ols(df_outcome, formula, vcov_type='HC1')

            for treat_var in treatment_vars:
                res = extract_results(
                    result, treat_var,
                    spec_id=f'robust/loo/drop_{drop_control}_{outcome}_{treat_var}',
                    spec_tree_path='robustness/leave_one_out.md',
                    outcome_var=outcome,
                    sample_desc='Full sample with loan outcomes',
                    controls_desc=f'All except {drop_control}',
                    cluster_var='None',
                    model_type='OLS'
                )
                if res:
                    results.append(res)

    # =========================================================================
    # SECTION 3: INFERENCE VARIATIONS (Clustering/SE)
    # =========================================================================
    print("\n--- Running Inference Variations ---")

    se_types = ['HC0', 'HC1', 'HC2', 'HC3']

    for outcome in ['repaid', 'charged_off']:
        df_outcome = df[df[outcome].notna()].copy()
        if len(df_outcome) < 50:
            continue

        formula = f'{outcome} ~ enforcement + selection'

        for se_type in se_types:
            result = run_ols(df_outcome, formula, vcov_type=se_type)

            for treat_var in treatment_vars:
                res = extract_results(
                    result, treat_var,
                    spec_id=f'robust/se/{se_type.lower()}_{outcome}_{treat_var}',
                    spec_tree_path='robustness/clustering_variations.md#alternative-se-methods',
                    outcome_var=outcome,
                    sample_desc='Full sample with loan outcomes',
                    controls_desc='None',
                    cluster_var=f'{se_type} robust SE',
                    model_type='OLS'
                )
                if res:
                    results.append(res)

    # Try clustering by branch
    for outcome in ['repaid', 'charged_off']:
        df_outcome = df[df[outcome].notna()].copy()
        if len(df_outcome) < 50:
            continue

        # Get branch columns
        branch_cols = [c for c in df_outcome.columns if c.startswith('branch_ref')]
        if branch_cols:
            # Create a branch identifier
            df_outcome['branch_id'] = df_outcome[branch_cols].idxmax(axis=1)

            formula = f'{outcome} ~ enforcement + selection'
            result = run_ols(df_outcome, formula, cluster_var='branch_id')

            for treat_var in treatment_vars:
                res = extract_results(
                    result, treat_var,
                    spec_id=f'robust/cluster/branch_{outcome}_{treat_var}',
                    spec_tree_path='robustness/clustering_variations.md#single-level-clustering',
                    outcome_var=outcome,
                    sample_desc='Full sample with loan outcomes',
                    controls_desc='None',
                    cluster_var='branch',
                    model_type='OLS'
                )
                if res:
                    results.append(res)

    # =========================================================================
    # SECTION 4: SAMPLE RESTRICTIONS
    # =========================================================================
    print("\n--- Running Sample Restrictions ---")

    for outcome in ['repaid', 'charged_off', 'interest', 'portion']:
        df_outcome = df[df[outcome].notna()].copy()
        if len(df_outcome) < 30:
            continue

        formula = f'{outcome} ~ enforcement + selection'

        # 4.1 Drop extreme outcome values (trim)
        for pct in [5, 10]:
            lower = df_outcome[outcome].quantile(pct/100)
            upper = df_outcome[outcome].quantile(1 - pct/100)
            df_trim = df_outcome[(df_outcome[outcome] >= lower) & (df_outcome[outcome] <= upper)]

            if len(df_trim) >= 30:
                result = run_ols(df_trim, formula, vcov_type='HC1')

                for treat_var in treatment_vars:
                    res = extract_results(
                        result, treat_var,
                        spec_id=f'robust/sample/trim_{pct}pct_{outcome}_{treat_var}',
                        spec_tree_path='robustness/sample_restrictions.md#outlier-handling',
                        outcome_var=outcome,
                        sample_desc=f'Trimmed at {pct}% tails',
                        controls_desc='None',
                        cluster_var='None',
                        model_type='OLS'
                    )
                    if res:
                        results.append(res)

        # 4.2 Winsorize
        for pct in [5, 10]:
            df_wins = df_outcome.copy()
            lower = df_wins[outcome].quantile(pct/100)
            upper = df_wins[outcome].quantile(1 - pct/100)
            df_wins[outcome] = df_wins[outcome].clip(lower, upper)

            result = run_ols(df_wins, formula, vcov_type='HC1')

            for treat_var in treatment_vars:
                res = extract_results(
                    result, treat_var,
                    spec_id=f'robust/sample/winsor_{pct}pct_{outcome}_{treat_var}',
                    spec_tree_path='robustness/sample_restrictions.md#outlier-handling',
                    outcome_var=outcome,
                    sample_desc=f'Winsorized at {pct}%',
                    controls_desc='None',
                    cluster_var='None',
                    model_type='OLS'
                )
                if res:
                    results.append(res)

    # 4.3 By referrer age groups
    df_loans = df[df['loan'] == 1].copy()

    if 'referrer_age' in df_loans.columns and df_loans['referrer_age'].notna().sum() > 30:
        median_age = df_loans['referrer_age'].median()

        for outcome in ['repaid', 'charged_off']:
            df_outcome = df_loans[df_loans[outcome].notna()].copy()
            if len(df_outcome) < 30:
                continue

            # Young referrers
            df_young = df_outcome[df_outcome['referrer_age'] <= median_age]
            if len(df_young) >= 20:
                formula = f'{outcome} ~ enforcement + selection'
                result = run_ols(df_young, formula, vcov_type='HC1')

                for treat_var in treatment_vars:
                    res = extract_results(
                        result, treat_var,
                        spec_id=f'robust/sample/young_referrer_{outcome}_{treat_var}',
                        spec_tree_path='robustness/sample_restrictions.md#demographic-subgroups',
                        outcome_var=outcome,
                        sample_desc=f'Referrer age <= {median_age:.0f}',
                        controls_desc='None',
                        cluster_var='None',
                        model_type='OLS'
                    )
                    if res:
                        results.append(res)

            # Old referrers
            df_old = df_outcome[df_outcome['referrer_age'] > median_age]
            if len(df_old) >= 20:
                result = run_ols(df_old, formula, vcov_type='HC1')

                for treat_var in treatment_vars:
                    res = extract_results(
                        result, treat_var,
                        spec_id=f'robust/sample/old_referrer_{outcome}_{treat_var}',
                        spec_tree_path='robustness/sample_restrictions.md#demographic-subgroups',
                        outcome_var=outcome,
                        sample_desc=f'Referrer age > {median_age:.0f}',
                        controls_desc='None',
                        cluster_var='None',
                        model_type='OLS'
                    )
                    if res:
                        results.append(res)

    # 4.4 By referrer gender
    if 'referrer_female_num' in df_loans.columns:
        for outcome in ['repaid', 'charged_off']:
            df_outcome = df_loans[df_loans[outcome].notna()].copy()
            if len(df_outcome) < 30:
                continue

            formula = f'{outcome} ~ enforcement + selection'

            # Female referrers
            df_female = df_outcome[df_outcome['referrer_female_num'] == 1]
            if len(df_female) >= 15:
                result = run_ols(df_female, formula, vcov_type='HC1')

                for treat_var in treatment_vars:
                    res = extract_results(
                        result, treat_var,
                        spec_id=f'robust/sample/female_referrer_{outcome}_{treat_var}',
                        spec_tree_path='robustness/sample_restrictions.md#demographic-subgroups',
                        outcome_var=outcome,
                        sample_desc='Female referrers only',
                        controls_desc='None',
                        cluster_var='None',
                        model_type='OLS'
                    )
                    if res:
                        results.append(res)

            # Male referrers
            df_male = df_outcome[df_outcome['referrer_female_num'] == 0]
            if len(df_male) >= 15:
                result = run_ols(df_male, formula, vcov_type='HC1')

                for treat_var in treatment_vars:
                    res = extract_results(
                        result, treat_var,
                        spec_id=f'robust/sample/male_referrer_{outcome}_{treat_var}',
                        spec_tree_path='robustness/sample_restrictions.md#demographic-subgroups',
                        outcome_var=outcome,
                        sample_desc='Male referrers only',
                        controls_desc='None',
                        cluster_var='None',
                        model_type='OLS'
                    )
                    if res:
                        results.append(res)

    # 4.5 By income levels
    if 'referrer_disposableincome000' in df_loans.columns:
        median_income = df_loans['referrer_disposableincome000'].median()

        for outcome in ['repaid', 'charged_off']:
            df_outcome = df_loans[df_loans[outcome].notna()].copy()
            if len(df_outcome) < 30:
                continue

            formula = f'{outcome} ~ enforcement + selection'

            # Low income
            df_low = df_outcome[df_outcome['referrer_disposableincome000'] <= median_income]
            if len(df_low) >= 15:
                result = run_ols(df_low, formula, vcov_type='HC1')

                for treat_var in treatment_vars:
                    res = extract_results(
                        result, treat_var,
                        spec_id=f'robust/sample/low_income_{outcome}_{treat_var}',
                        spec_tree_path='robustness/sample_restrictions.md#demographic-subgroups',
                        outcome_var=outcome,
                        sample_desc='Low income referrers',
                        controls_desc='None',
                        cluster_var='None',
                        model_type='OLS'
                    )
                    if res:
                        results.append(res)

            # High income
            df_high = df_outcome[df_outcome['referrer_disposableincome000'] > median_income]
            if len(df_high) >= 15:
                result = run_ols(df_high, formula, vcov_type='HC1')

                for treat_var in treatment_vars:
                    res = extract_results(
                        result, treat_var,
                        spec_id=f'robust/sample/high_income_{outcome}_{treat_var}',
                        spec_tree_path='robustness/sample_restrictions.md#demographic-subgroups',
                        outcome_var=outcome,
                        sample_desc='High income referrers',
                        controls_desc='None',
                        cluster_var='None',
                        model_type='OLS'
                    )
                    if res:
                        results.append(res)

    # =========================================================================
    # SECTION 5: ALTERNATIVE OUTCOMES
    # =========================================================================
    print("\n--- Running Alternative Outcomes ---")

    # 5.1 Binary outcomes - whether repaid anything
    df_loans = df[df['loan'] == 1].copy()

    if 'portion' in df_loans.columns:
        df_loans['repaid_any'] = (df_loans['portion'] > 0).astype(int)
        df_outcome = df_loans[df_loans['repaid_any'].notna()].copy()

        if len(df_outcome) >= 30:
            formula = 'repaid_any ~ enforcement + selection'
            result = run_ols(df_outcome, formula, vcov_type='HC1')

            for treat_var in treatment_vars:
                res = extract_results(
                    result, treat_var,
                    spec_id=f'robust/outcome/repaid_any_{treat_var}',
                    spec_tree_path='robustness/functional_form.md#outcome-variable-transformations',
                    outcome_var='repaid_any',
                    sample_desc='Loans only',
                    controls_desc='None',
                    cluster_var='None',
                    model_type='OLS (LPM)'
                )
                if res:
                    results.append(res)

    # 5.2 Log transformations
    for outcome in ['interest', 'portion']:
        df_outcome = df_loans[df_loans[outcome].notna()].copy()
        if len(df_outcome) < 30:
            continue

        # asinh transformation (handles zeros)
        df_outcome[f'{outcome}_asinh'] = np.arcsinh(df_outcome[outcome])

        formula = f'{outcome}_asinh ~ enforcement + selection'
        result = run_ols(df_outcome, formula, vcov_type='HC1')

        for treat_var in treatment_vars:
            res = extract_results(
                result, treat_var,
                spec_id=f'robust/form/asinh_{outcome}_{treat_var}',
                spec_tree_path='robustness/functional_form.md#outcome-variable-transformations',
                outcome_var=f'{outcome}_asinh',
                sample_desc='Loans only',
                controls_desc='None',
                cluster_var='None',
                model_type='OLS'
            )
            if res:
                results.append(res)

    # =========================================================================
    # SECTION 6: ALTERNATIVE TREATMENT DEFINITIONS
    # =========================================================================
    print("\n--- Running Alternative Treatment Definitions ---")

    # 6.1 Individual treatment arms vs control (treatment 1)
    for outcome in ['repaid', 'charged_off']:
        df_outcome = df_loans[df_loans[outcome].notna()].copy()
        if len(df_outcome) < 30:
            continue

        # Each treatment vs. treatment 1 (Approved only)
        for treat_arm, treat_name in [(2, 'apprep'), (3, 'rep'), (4, 'repapp')]:
            df_sub = df_outcome[df_outcome['treatment_num'].isin([1, treat_arm])].copy()
            df_sub['treat_arm'] = (df_sub['treatment_num'] == treat_arm).astype(int)

            if len(df_sub) >= 20:
                formula = f'{outcome} ~ treat_arm'
                result = run_ols(df_sub, formula, vcov_type='HC1')

                res = extract_results(
                    result, 'treat_arm',
                    spec_id=f'robust/treatment/{treat_name}_vs_app_{outcome}',
                    spec_tree_path='methods/cross_sectional_ols.md#alternative-treatments',
                    outcome_var=outcome,
                    sample_desc=f'Treatment {treat_arm} vs 1',
                    controls_desc='None',
                    cluster_var='None',
                    model_type='OLS'
                )
                if res:
                    results.append(res)

    # 6.2 With interaction term
    for outcome in ['repaid', 'charged_off']:
        df_outcome = df_loans[df_loans[outcome].notna()].copy()
        if len(df_outcome) < 30:
            continue

        formula = f'{outcome} ~ enforcement + selection + enforcement_x_selection'
        result = run_ols(df_outcome, formula, vcov_type='HC1')

        for treat_var in ['enforcement', 'selection', 'enforcement_x_selection']:
            res = extract_results(
                result, treat_var,
                spec_id=f'robust/treatment/interaction_{outcome}_{treat_var}',
                spec_tree_path='methods/cross_sectional_ols.md#interaction-effects',
                outcome_var=outcome,
                sample_desc='Loans only',
                controls_desc='None',
                cluster_var='None',
                model_type='OLS'
            )
            if res:
                results.append(res)

    # =========================================================================
    # SECTION 7: HETEROGENEITY ANALYSES
    # =========================================================================
    print("\n--- Running Heterogeneity Analyses ---")

    # 7.1 Interaction with referrer characteristics
    het_vars = [
        ('referrer_female_num', 'gender'),
        ('referrer_education', 'education'),
    ]

    for outcome in ['repaid', 'charged_off']:
        df_outcome = df_loans[df_loans[outcome].notna()].copy()
        if len(df_outcome) < 30:
            continue

        for het_var, het_name in het_vars:
            if het_var not in df_outcome.columns:
                continue

            df_het = df_outcome[df_outcome[het_var].notna()].copy()
            if len(df_het) < 30:
                continue

            # Make sure het_var is numeric
            if df_het[het_var].dtype.name == 'category':
                df_het[het_var] = pd.to_numeric(df_het[het_var].cat.codes, errors='coerce')
            else:
                df_het[het_var] = pd.to_numeric(df_het[het_var], errors='coerce')

            df_het = df_het[df_het[het_var].notna()]
            if len(df_het) < 30:
                continue

            # Create interaction
            df_het['enforce_x_het'] = df_het['enforcement'].astype(float) * df_het[het_var].astype(float)
            df_het['select_x_het'] = df_het['selection'].astype(float) * df_het[het_var].astype(float)

            formula = f'{outcome} ~ enforcement + selection + {het_var} + enforce_x_het + select_x_het'
            result = run_ols(df_het, formula, vcov_type='HC1')

            for treat_var in ['enforcement', 'selection', 'enforce_x_het', 'select_x_het']:
                res = extract_results(
                    result, treat_var,
                    spec_id=f'robust/het/interaction_{het_name}_{outcome}_{treat_var}',
                    spec_tree_path='robustness/heterogeneity.md#interaction-specifications',
                    outcome_var=outcome,
                    sample_desc='Loans only',
                    controls_desc=f'Interaction with {het_name}',
                    cluster_var='None',
                    model_type='OLS'
                )
                if res:
                    results.append(res)

    # 7.2 By relationship type
    if 'relationship' in df_loans.columns:
        for outcome in ['repaid', 'charged_off']:
            df_outcome = df_loans[df_loans[outcome].notna()].copy()
            if len(df_outcome) < 20:
                continue

            for rel_type in ['Work', 'Relative']:
                df_rel = df_outcome[df_outcome['relationship'] == rel_type].copy()
                if len(df_rel) >= 15:
                    formula = f'{outcome} ~ enforcement + selection'
                    result = run_ols(df_rel, formula, vcov_type='HC1')

                    for treat_var in treatment_vars:
                        res = extract_results(
                            result, treat_var,
                            spec_id=f'robust/het/relationship_{rel_type.lower()}_{outcome}_{treat_var}',
                            spec_tree_path='robustness/heterogeneity.md#demographic-subgroups',
                            outcome_var=outcome,
                            sample_desc=f'Relationship type: {rel_type}',
                            controls_desc='None',
                            cluster_var='None',
                            model_type='OLS'
                        )
                        if res:
                            results.append(res)

    # =========================================================================
    # SECTION 8: PLACEBO TESTS
    # =========================================================================
    print("\n--- Running Placebo Tests ---")

    # Test if treatment predicts referrer characteristics (should be ~0)
    placebo_outcomes = ['referrer_age', 'referrer_disposableincome000', 'referrer_applicationscore']
    placebo_outcomes = [p for p in placebo_outcomes if p in df.columns]

    for outcome in placebo_outcomes:
        df_outcome = df[df[outcome].notna()].copy()
        if len(df_outcome) < 100:
            continue

        formula = f'{outcome} ~ enforcement + selection'
        result = run_ols(df_outcome, formula, vcov_type='HC1')

        for treat_var in treatment_vars:
            res = extract_results(
                result, treat_var,
                spec_id=f'robust/placebo/balance_{outcome}_{treat_var}',
                spec_tree_path='robustness/placebo_tests.md',
                outcome_var=outcome,
                sample_desc='Full sample (balance check)',
                controls_desc='None',
                cluster_var='None',
                model_type='OLS'
            )
            if res:
                results.append(res)

    # =========================================================================
    # SECTION 9: FULL MODEL WITH CONTROLS
    # =========================================================================
    print("\n--- Running Full Model with Controls ---")

    if len(baseline_controls) >= 3:
        controls_str = ' + '.join(baseline_controls)

        for outcome in outcomes:
            df_outcome = df_loans[df_loans[outcome].notna()].copy()
            if len(df_outcome) < 30:
                continue

            formula = f'{outcome} ~ enforcement + selection + {controls_str}'
            result = run_ols(df_outcome, formula, vcov_type='HC1')

            for treat_var in treatment_vars:
                res = extract_results(
                    result, treat_var,
                    spec_id=f'ols/full_model_{outcome}_{treat_var}',
                    spec_tree_path='methods/cross_sectional_ols.md#control-sets',
                    outcome_var=outcome,
                    sample_desc='Loans only',
                    controls_desc='Full controls',
                    cluster_var='None',
                    model_type='OLS'
                )
                if res:
                    results.append(res)

    # =========================================================================
    # SECTION 10: ESTIMATION METHOD VARIATIONS
    # =========================================================================
    print("\n--- Running Estimation Method Variations ---")

    # Probit/Logit for binary outcomes
    for outcome in ['repaid', 'charged_off']:
        df_outcome = df_loans[df_loans[outcome].notna()].copy()
        if len(df_outcome) < 30:
            continue

        # Logit
        try:
            formula = f'{outcome} ~ enforcement + selection'
            logit_model = smf.logit(formula, data=df_outcome).fit(disp=0)

            for treat_var in treatment_vars:
                if treat_var in logit_model.params.index:
                    coef = logit_model.params[treat_var]
                    se = logit_model.bse[treat_var]
                    t_stat = coef / se
                    p_val = logit_model.pvalues[treat_var]
                    ci = logit_model.conf_int().loc[treat_var]

                    # Marginal effect (at means)
                    marg_effect = coef * (np.exp(-coef) / (1 + np.exp(-coef))**2)

                    results.append({
                        'paper_id': PAPER_ID,
                        'journal': JOURNAL,
                        'paper_title': PAPER_TITLE,
                        'spec_id': f'ols/method/logit_{outcome}_{treat_var}',
                        'spec_tree_path': 'methods/discrete_choice.md',
                        'outcome_var': outcome,
                        'treatment_var': treat_var,
                        'coefficient': float(coef),
                        'std_error': float(se),
                        't_stat': float(t_stat),
                        'p_value': float(p_val),
                        'ci_lower': float(ci[0]),
                        'ci_upper': float(ci[1]),
                        'n_obs': int(logit_model.nobs),
                        'r_squared': float(logit_model.prsquared),
                        'coefficient_vector_json': json.dumps({'treatment': {'var': treat_var, 'coef': float(coef), 'se': float(se), 'pval': float(p_val)}}),
                        'sample_desc': 'Loans only',
                        'fixed_effects': 'None',
                        'controls_desc': 'None',
                        'cluster_var': 'None',
                        'model_type': 'Logit'
                    })
        except Exception as e:
            print(f"Logit failed for {outcome}: {e}")

    # =========================================================================
    # SECTION 11: ADDITIONAL ROBUSTNESS
    # =========================================================================
    print("\n--- Running Additional Robustness ---")

    # Add incrementally controls
    for outcome in ['repaid', 'charged_off']:
        df_outcome = df_loans[df_loans[outcome].notna()].copy()
        if len(df_outcome) < 30:
            continue

        controls_so_far = []
        for i, control in enumerate(baseline_controls[:5]):  # First 5 controls
            controls_so_far.append(control)
            controls_str = ' + '.join(controls_so_far)
            formula = f'{outcome} ~ enforcement + selection + {controls_str}'

            result = run_ols(df_outcome, formula, vcov_type='HC1')

            for treat_var in treatment_vars:
                res = extract_results(
                    result, treat_var,
                    spec_id=f'robust/control/add_{i+1}_{outcome}_{treat_var}',
                    spec_tree_path='robustness/control_progression.md',
                    outcome_var=outcome,
                    sample_desc='Loans only',
                    controls_desc=f'First {i+1} controls',
                    cluster_var='None',
                    model_type='OLS'
                )
                if res:
                    results.append(res)

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    print(f"\n--- Saving {len(results)} specifications ---")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Results saved to {OUTPUT_PATH}")

    return results_df

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    results = run_specification_search()

    print("\n" + "="*60)
    print("SPECIFICATION SEARCH SUMMARY")
    print("="*60)
    print(f"Total specifications: {len(results)}")
    print(f"Positive coefficients: {(results['coefficient'] > 0).sum()} ({(results['coefficient'] > 0).mean()*100:.1f}%)")
    print(f"Significant at 5%: {(results['p_value'] < 0.05).sum()} ({(results['p_value'] < 0.05).mean()*100:.1f}%)")
    print(f"Significant at 1%: {(results['p_value'] < 0.01).sum()} ({(results['p_value'] < 0.01).mean()*100:.1f}%)")
    print(f"Median coefficient: {results['coefficient'].median():.4f}")
    print(f"Mean coefficient: {results['coefficient'].mean():.4f}")
    print(f"Range: [{results['coefficient'].min():.4f}, {results['coefficient'].max():.4f}]")
