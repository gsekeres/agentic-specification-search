"""
Specification Search: Paper 158081-V7
Real-Time Population Survey (RPS) - Work From Home Analysis

This script replicates the specification search for Bick, Blandin, and Mertens (2023)
"Work from Home before and after the COVID-19 Outbreak" using RPS data.

Primary outcome: WFH share (share of work days spent working from home)
Primary treatment: College education (Bachelor's degree or higher)

Run from package directory:
    python scripts/paper_analyses/158081-V7.py
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

PAPER_ID = "158081-V7"
JOURNAL = "AEJ-Macroeconomics"
PAPER_TITLE = "Work from Home before and after the COVID-19 Outbreak (RPS Data)"

DATA_FILE = 'rps_data_release_v5.2.csv'
OUTPUT_FILE = 'specification_results.csv'

# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_data(filepath):
    """Load and prepare RPS data for analysis."""

    print("Loading data...")
    df = pd.read_csv(filepath, low_memory=False)

    # Convert numeric columns
    def safe_numeric(series):
        return pd.to_numeric(series, errors='coerce')

    numeric_cols = ['days_commuting', 'days_working', 'wearn', 'wage', 'uhrs', 'ahrs', 'age']
    for col in numeric_cols:
        df[col + '_num'] = safe_numeric(df[col])

    # Create WFH variables
    df['wfh_days'] = df['days_working_num'] - df['days_commuting_num']
    df['wfh_share'] = df['wfh_days'] / df['days_working_num']

    # Filter to employed at work with valid data
    df_working = df[
        (df['emp_detail'] == 'Employed, at work') &
        (df['days_working_num'] > 0) &
        (df['wfh_share'].notna())
    ].copy()

    # Create treatment: College education (using str.contains for encoding robustness)
    df_working['college'] = (
        df_working['educ'].str.contains('Bachelor', case=False, na=False) |
        df_working['educ'].str.contains('Graduate degree', case=False, na=False)
    ).astype(int)

    # Demographics
    df_working['female'] = (df_working['sex'] == 'Female').astype(int)
    df_working['male'] = (df_working['sex'] == 'Male').astype(int)
    df_working['married'] = df_working['marst'].str.contains('Married, spouse currently lives', case=False, na=False).astype(int)
    df_working['has_children'] = df_working['nchild'].isin(['1 child', '2 children', '3 or more children']).astype(int)
    df_working['age_sq'] = df_working['age_num'] ** 2

    # Education dummies
    df_working['less_hs'] = df_working['educ'].str.contains('Did not complete', case=False, na=False).astype(int)
    df_working['hs_only'] = df_working['educ'].str.contains('High school graduate', case=False, na=False).astype(int)
    df_working['some_college'] = df_working['educ'].str.contains('Some college', case=False, na=False).astype(int)
    df_working['associates'] = df_working['educ'].str.contains('Associate', case=False, na=False).astype(int)
    df_working['bachelors'] = df_working['educ'].str.contains('Bachelor', case=False, na=False).astype(int)
    df_working['graduate'] = df_working['educ'].str.contains('Graduate degree', case=False, na=False).astype(int)

    # Race
    df_working['white'] = (df_working['race_comb'] == 'Non-Hispanic white').astype(int)
    df_working['black'] = (df_working['race_comb'] == 'Non-Hispanic black').astype(int)
    df_working['hispanic_d'] = (df_working['race_comb'] == 'Hispanic').astype(int)

    # Year dummies
    for y in df_working['year'].unique():
        df_working[f'year_{y}'] = (df_working['year'] == y).astype(int)

    # Industry dummies
    industry_map = {
        'Health Care and Social Assistance': 'ind_healthcare',
        'Education': 'ind_education',
        'Professional, Technical, or Business Services': 'ind_professional',
        'Banking, Finance, or Insurance': 'ind_finance',
        'Retail Trade': 'ind_retail',
        'Manufacturing': 'ind_manufacturing',
        'Government, including Military': 'ind_government',
        'Information Services (including Publishing or Media)': 'ind_information',
        'Construction': 'ind_construction'
    }
    for ind_name, col_name in industry_map.items():
        df_working[col_name] = (df_working['ind18'] == ind_name).astype(int)

    # Income
    df_working['income_high'] = df_working['hhinc'].isin(['$100,000-$125,000', '$125,000-$150,000', '$150,000 or more']).astype(int)
    df_working['income_low'] = df_working['hhinc'].isin(['$0-$25,000', '$25,000-$50,000']).astype(int)

    # Weights
    df_working['weight'] = safe_numeric(df_working['wgt'])

    # Age groups
    df_working['age_25_34'] = ((df_working['age_num'] >= 25) & (df_working['age_num'] < 35)).astype(int)
    df_working['age_35_44'] = ((df_working['age_num'] >= 35) & (df_working['age_num'] < 45)).astype(int)
    df_working['age_45_54'] = ((df_working['age_num'] >= 45) & (df_working['age_num'] < 55)).astype(int)
    df_working['age_55_64'] = (df_working['age_num'] >= 55).astype(int)

    # Alternative outcomes
    df_working['any_wfh'] = (df_working['wfh_share'] > 0).astype(int)
    df_working['full_wfh'] = (df_working['wfh_share'] == 1).astype(int)
    df_working['wfh_share_ihs'] = np.arcsinh(df_working['wfh_share'])

    print(f"Prepared {len(df_working)} observations")
    print(f"College share: {df_working['college'].mean():.3f}")
    print(f"Mean WFH share: {df_working['wfh_share'].mean():.3f}")

    return df_working

# =============================================================================
# REGRESSION HELPER
# =============================================================================

def run_regression(data, formula, treatment_var='college', spec_id='', spec_tree_path='',
                   outcome_var='wfh_share', cluster_var=None, weights=None,
                   sample_desc='', fe_desc='None', controls_desc='', model_type='OLS'):
    """Run regression and return standardized results dictionary."""

    try:
        # Get formula variables
        formula_vars = [v.strip() for v in formula.replace('~', '+').replace('*', '+').split('+')]
        formula_vars = [v for v in formula_vars if v and v != '1']

        data_clean = data.dropna(subset=formula_vars)

        if len(data_clean) < 30:
            return None

        if weights is not None and weights in data.columns:
            data_clean = data_clean.dropna(subset=[weights])
            weight_vals = data_clean[weights]
            model = smf.wls(formula, data=data_clean, weights=weight_vals).fit(cov_type='HC1')
        else:
            if cluster_var and cluster_var in data_clean.columns:
                data_clean = data_clean.dropna(subset=[cluster_var])
                model = smf.ols(formula, data=data_clean).fit(cov_type='cluster',
                                                               cov_kwds={'groups': data_clean[cluster_var]})
            else:
                model = smf.ols(formula, data=data_clean).fit(cov_type='HC1')

        if treatment_var not in model.params:
            return None

        coef = model.params[treatment_var]
        se = model.bse[treatment_var]
        t_stat = model.tvalues[treatment_var]
        p_val = model.pvalues[treatment_var]
        ci_lower, ci_upper = model.conf_int().loc[treatment_var]

        coef_vector = {
            "treatment": {"var": treatment_var, "coef": float(coef), "se": float(se), "pval": float(p_val)},
            "controls": [{"var": v, "coef": float(model.params[v]), "se": float(model.bse[v]), "pval": float(model.pvalues[v])}
                         for v in model.params.index if v != treatment_var and v != 'Intercept'],
            "n_obs": int(model.nobs),
            "r_squared": float(model.rsquared),
            "adj_r_squared": float(model.rsquared_adj)
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
            't_stat': float(t_stat),
            'p_value': float(p_val),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_obs': int(model.nobs),
            'r_squared': float(model.rsquared),
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc,
            'fixed_effects': fe_desc,
            'controls_desc': controls_desc,
            'cluster_var': cluster_var if cluster_var else 'None',
            'model_type': model_type,
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }

    except Exception as e:
        print(f"Error in {spec_id}: {str(e)[:60]}")
        return None

# =============================================================================
# MAIN SPECIFICATION SEARCH
# =============================================================================

def run_specification_search(df):
    """Run the full specification search."""

    results = []

    # Control variable sets
    basic_controls = 'age_num + age_sq + female'
    demographic_controls = 'age_num + age_sq + female + married + has_children + white + black + hispanic_d'
    full_controls = 'age_num + age_sq + female + married + has_children + white + black + hispanic_d + income_high + income_low'
    industry_controls = 'ind_healthcare + ind_education + ind_professional + ind_finance + ind_retail + ind_manufacturing'

    # =========================================================================
    # BASELINE
    # =========================================================================
    print("\nRunning baseline...")
    formula = f'wfh_share ~ college + {basic_controls}'
    res = run_regression(df, formula, spec_id='baseline',
                         spec_tree_path='methods/cross_sectional_ols.md#baseline',
                         sample_desc='Employed at work', controls_desc='Age, age squared, female')
    if res:
        results.append(res)
        print(f"  Baseline: coef={res['coefficient']:.4f}, se={res['std_error']:.4f}")

    # =========================================================================
    # CONTROL VARIATIONS
    # =========================================================================
    print("\nRunning control variations...")

    # No controls
    res = run_regression(df, 'wfh_share ~ college', spec_id='robust/control/none',
                         spec_tree_path='robustness/control_progression.md', controls_desc='None')
    if res: results.append(res)

    # Add incrementally
    control_list = ['age_num', 'age_sq', 'female', 'married', 'has_children', 'white', 'black', 'hispanic_d', 'income_high', 'income_low']
    controls_so_far = []
    for ctrl in control_list:
        controls_so_far.append(ctrl)
        formula = f'wfh_share ~ college + {" + ".join(controls_so_far)}'
        res = run_regression(df, formula, spec_id=f'robust/control/add_{ctrl}',
                             spec_tree_path='robustness/control_progression.md')
        if res: results.append(res)

    # With industry
    formula = f'wfh_share ~ college + {full_controls} + {industry_controls}'
    res = run_regression(df, formula, spec_id='robust/control/with_industry',
                         spec_tree_path='methods/cross_sectional_ols.md#control-sets')
    if res: results.append(res)

    # Leave-one-out
    base_ctrls = ['age_num', 'age_sq', 'female', 'married', 'has_children']
    for drop in base_ctrls:
        remaining = [c for c in base_ctrls if c != drop]
        formula = f'wfh_share ~ college + {" + ".join(remaining)}'
        res = run_regression(df, formula, spec_id=f'robust/loo/drop_{drop}',
                             spec_tree_path='robustness/leave_one_out.md')
        if res: results.append(res)

    # =========================================================================
    # SAMPLE RESTRICTIONS
    # =========================================================================
    print("\nRunning sample restrictions...")
    formula = f'wfh_share ~ college + {demographic_controls}'

    for year in sorted(df['year'].unique()):
        res = run_regression(df[df['year'] == year], formula,
                             spec_id=f'robust/sample/year_{year}',
                             spec_tree_path='robustness/sample_restrictions.md')
        if res: results.append(res)

    # Other sample restrictions
    sample_conditions = [
        ('post_pandemic', df['year'] >= 2022),
        ('exclude_first_year', df['year'] != df['year'].min()),
        ('exclude_last_year', df['year'] != df['year'].max()),
        ('interior_wfh', (df['wfh_share'] > 0) & (df['wfh_share'] < 1)),
        ('fulltime_only', df['days_working_num'] >= 5),
        ('parttime_only', df['days_working_num'] < 5),
    ]
    for name, cond in sample_conditions:
        res = run_regression(df[cond], formula, spec_id=f'robust/sample/{name}',
                             spec_tree_path='robustness/sample_restrictions.md')
        if res: results.append(res)

    # COVID peak
    df_peak = df[(df['year'] == 2020) & (df['month'].isin([4,5,6]))]
    res = run_regression(df_peak, formula, spec_id='robust/sample/covid_peak',
                         spec_tree_path='robustness/sample_restrictions.md')
    if res: results.append(res)

    # =========================================================================
    # ALTERNATIVE OUTCOMES
    # =========================================================================
    print("\nRunning alternative outcomes...")

    for outcome, spec_name in [('wfh_days', 'wfh_days_count'), ('any_wfh', 'any_wfh_binary'),
                               ('full_wfh', 'full_wfh_binary'), ('days_commuting_num', 'days_commuting'),
                               ('days_working_num', 'days_working')]:
        res = run_regression(df, f'{outcome} ~ college + {demographic_controls}',
                             spec_id=f'robust/outcome/{spec_name}', outcome_var=outcome,
                             spec_tree_path='robustness/measurement.md')
        if res: results.append(res)

    # =========================================================================
    # ALTERNATIVE TREATMENTS
    # =========================================================================
    print("\nRunning alternative treatments...")

    for treat, spec_name in [('graduate', 'graduate_degree'), ('bachelors', 'bachelors_only'),
                              ('ind_professional', 'professional_industry'), ('ind_information', 'information_industry'),
                              ('ind_finance', 'finance_industry'), ('ind_education', 'education_industry'),
                              ('income_high', 'high_income')]:
        formula = f'wfh_share ~ {treat} + age_num + age_sq + female + married + has_children'
        res = run_regression(df, formula, treatment_var=treat,
                             spec_id=f'robust/treatment/{spec_name}',
                             spec_tree_path='robustness/measurement.md')
        if res: results.append(res)

    # =========================================================================
    # INFERENCE VARIATIONS
    # =========================================================================
    print("\nRunning inference variations...")
    formula = f'wfh_share ~ college + {demographic_controls}'

    res = run_regression(df, formula, spec_id='robust/cluster/wave', cluster_var='wave',
                         spec_tree_path='robustness/clustering_variations.md')
    if res: results.append(res)

    res = run_regression(df, formula, spec_id='robust/cluster/year', cluster_var='year',
                         spec_tree_path='robustness/clustering_variations.md')
    if res: results.append(res)

    # Classical SE
    model = smf.ols(formula, data=df.dropna(subset=['wfh_share', 'college', 'age_num', 'female'])).fit()
    results.append({
        'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
        'spec_id': 'robust/se/classical', 'spec_tree_path': 'robustness/clustering_variations.md',
        'outcome_var': 'wfh_share', 'treatment_var': 'college',
        'coefficient': float(model.params['college']), 'std_error': float(model.bse['college']),
        't_stat': float(model.tvalues['college']), 'p_value': float(model.pvalues['college']),
        'ci_lower': float(model.conf_int().loc['college', 0]),
        'ci_upper': float(model.conf_int().loc['college', 1]),
        'n_obs': int(model.nobs), 'r_squared': float(model.rsquared),
        'coefficient_vector_json': '{}', 'sample_desc': '', 'fixed_effects': 'None',
        'controls_desc': '', 'cluster_var': 'None (classical SE)', 'model_type': 'OLS',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })

    # =========================================================================
    # ESTIMATION METHOD VARIATIONS
    # =========================================================================
    print("\nRunning estimation method variations...")

    # WLS
    res = run_regression(df, formula, spec_id='ols/method/wls_weighted', weights='weight',
                         model_type='WLS', spec_tree_path='methods/cross_sectional_ols.md#estimation-method')
    if res: results.append(res)

    # Year FE
    year_dummies = ' + '.join([f'year_{y}' for y in [2021, 2022, 2023, 2024]])
    formula_fe = f'wfh_share ~ college + {demographic_controls} + {year_dummies}'
    res = run_regression(df, formula_fe, spec_id='ols/fe/year', fe_desc='Year FE',
                         spec_tree_path='methods/cross_sectional_ols.md#fixed-effects')
    if res: results.append(res)

    # Industry FE
    formula_ind = f'wfh_share ~ college + {demographic_controls} + {industry_controls}'
    res = run_regression(df, formula_ind, spec_id='ols/fe/industry', fe_desc='Industry FE',
                         spec_tree_path='methods/cross_sectional_ols.md#fixed-effects')
    if res: results.append(res)

    # Both
    formula_both = f'wfh_share ~ college + {demographic_controls} + {year_dummies} + {industry_controls}'
    res = run_regression(df, formula_both, spec_id='ols/fe/year_industry', fe_desc='Year + Industry FE',
                         spec_tree_path='methods/cross_sectional_ols.md#fixed-effects')
    if res: results.append(res)

    # =========================================================================
    # FUNCTIONAL FORM
    # =========================================================================
    print("\nRunning functional form variations...")

    res = run_regression(df, f'wfh_share_ihs ~ college + {demographic_controls}',
                         spec_id='robust/funcform/ihs_outcome', outcome_var='wfh_share_ihs',
                         spec_tree_path='robustness/functional_form.md')
    if res: results.append(res)

    res = run_regression(df, f'wfh_share ~ college + age_25_34 + age_35_44 + age_45_54 + age_55_64 + female',
                         spec_id='robust/funcform/age_categories',
                         spec_tree_path='robustness/functional_form.md')
    if res: results.append(res)

    res = run_regression(df, f'wfh_share ~ some_college + associates + bachelors + graduate + {basic_controls}',
                         spec_id='robust/funcform/educ_categories', treatment_var='bachelors',
                         spec_tree_path='robustness/functional_form.md')
    if res: results.append(res)

    # =========================================================================
    # HETEROGENEITY
    # =========================================================================
    print("\nRunning heterogeneity analyses...")
    formula = f'wfh_share ~ college + {demographic_controls}'

    # By gender
    res = run_regression(df[df['male'] == 1], formula.replace(' + female', ''),
                         spec_id='robust/het/by_gender_male', spec_tree_path='robustness/heterogeneity.md')
    if res: results.append(res)
    res = run_regression(df[df['female'] == 1], formula.replace(' + female', ''),
                         spec_id='robust/het/by_gender_female', spec_tree_path='robustness/heterogeneity.md')
    if res: results.append(res)

    # By age
    for label, cond in [('young', df['age_num'] < 35),
                        ('middle', (df['age_num'] >= 35) & (df['age_num'] < 50)),
                        ('old', df['age_num'] >= 50)]:
        res = run_regression(df[cond], formula, spec_id=f'robust/het/by_age_{label}',
                             spec_tree_path='robustness/heterogeneity.md')
        if res: results.append(res)

    # By income
    res = run_regression(df[df['income_high'] == 1], formula.replace(' + income_high + income_low', ''),
                         spec_id='robust/het/by_income_high', spec_tree_path='robustness/heterogeneity.md')
    if res: results.append(res)
    res = run_regression(df[df['income_low'] == 1], formula.replace(' + income_high + income_low', ''),
                         spec_id='robust/het/by_income_low', spec_tree_path='robustness/heterogeneity.md')
    if res: results.append(res)

    # By marital status
    res = run_regression(df[df['married'] == 1], formula.replace(' + married', ''),
                         spec_id='robust/het/by_married', spec_tree_path='robustness/heterogeneity.md')
    if res: results.append(res)
    res = run_regression(df[df['married'] == 0], formula.replace(' + married', ''),
                         spec_id='robust/het/by_unmarried', spec_tree_path='robustness/heterogeneity.md')
    if res: results.append(res)

    # By children
    res = run_regression(df[df['has_children'] == 1], formula.replace(' + has_children', ''),
                         spec_id='robust/het/by_children_yes', spec_tree_path='robustness/heterogeneity.md')
    if res: results.append(res)
    res = run_regression(df[df['has_children'] == 0], formula.replace(' + has_children', ''),
                         spec_id='robust/het/by_children_no', spec_tree_path='robustness/heterogeneity.md')
    if res: results.append(res)

    # Interactions
    for var in ['female', 'married', 'has_children', 'income_high']:
        res = run_regression(df, f'wfh_share ~ college * {var} + age_num + age_sq + female + married + has_children'.replace(f' + {var}', ''),
                             spec_id=f'robust/het/interaction_{var}', spec_tree_path='robustness/heterogeneity.md')
        if res: results.append(res)

    # =========================================================================
    # PLACEBO TESTS
    # =========================================================================
    print("\nRunning placebo tests...")

    res = run_regression(df, f'days_working_num ~ college + {demographic_controls}',
                         spec_id='robust/placebo/days_working_outcome', outcome_var='days_working_num',
                         spec_tree_path='robustness/placebo_tests.md')
    if res: results.append(res)

    for ind in ['ind_construction', 'ind_retail']:
        res = run_regression(df, f'wfh_share ~ {ind} + {demographic_controls}',
                             spec_id=f'robust/placebo/{ind.replace("ind_", "")}_industry', treatment_var=ind,
                             spec_tree_path='robustness/placebo_tests.md')
        if res: results.append(res)

    return results

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Load and prepare data
    df = prepare_data(DATA_FILE)

    # Run specification search
    results = run_specification_search(df)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "="*80)
    print("SPECIFICATION SEARCH COMPLETE")
    print("="*80)
    print(f"Total specifications: {len(results_df)}")

    college_specs = results_df[results_df['treatment_var'] == 'college']
    print(f"College treatment specs: {len(college_specs)}")
    print(f"  Positive: {(college_specs['coefficient'] > 0).sum()} ({100*(college_specs['coefficient'] > 0).mean():.1f}%)")
    print(f"  Sig at 5%: {(college_specs['p_value'] < 0.05).sum()} ({100*(college_specs['p_value'] < 0.05).mean():.1f}%)")
    print(f"  Mean coef: {college_specs['coefficient'].mean():.4f}")
    print(f"  Median coef: {college_specs['coefficient'].median():.4f}")
    print(f"\nResults saved to: {OUTPUT_FILE}")
