#!/usr/bin/env python3
"""
Specification Search: 183985-V2
Paper: Emoticons as Performance Feedback for College Students: A Large-Classroom Field Experiment
Authors: Darshak Patel and Justin Roush
Journal: AER: P&P

Method: Difference-in-Differences with panel fixed effects
Treatment: Emoticon feedback on exam 1 (vs numeric feedback only)
Main Outcomes: Exam scores (log), quiz scores, homework completion/scores, attendance
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

PAPER_ID = "183985-V2"
PAPER_TITLE = "Emoticons as Performance Feedback for College Students: A Large-Classroom Field Experiment"
JOURNAL = "AER: P&P"
DATA_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/183985-V2/aeapp_patelroush_data_v2.dta"
OUTPUT_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/183985-V2/specification_results.csv"

# ============================================================================
# Data Loading and Preparation
# ============================================================================

def load_and_prepare_data():
    """Load and prepare the dataset following the original Stata code."""
    df = pd.read_stata(DATA_PATH)

    # Create emoji category (based on test1 scores)
    df['emoji'] = 1  # Sad (default)
    df.loc[(df['test1'] > 67) & (df['test1'] < 80), 'emoji'] = 2  # Neut.Low
    df.loc[(df['test1'] > 79) & (df['test1'] < 91), 'emoji'] = 3  # Neut.High
    df.loc[df['test1'] > 90, 'emoji'] = 4  # Happy

    # Create log test scores
    df['ltest1'] = np.log(df['test1'].clip(lower=1))
    df['ltest2'] = np.log(df['test2'].clip(lower=1))

    return df


def prepare_test_panel(df):
    """Prepare panel data for test score analysis."""
    # Filter to estimation sample (flag_test == 0)
    df_wide = df[df['flag_test'] == 0].copy()

    # Create long format by stacking test1 and test2
    df_t1 = df_wide[['name_id', 'treatment', 'emoji', 'male', 'freshman',
                     'sophomore', 'junior', 'senior', 'econmajor', 'hours_semester',
                     'econ_first', 'job', 'whitenonhispanic', 'black', 'whitehispanic',
                     'asian', 'race_other', 'gpa_categorical', 'class_cat', 'test1']].copy()
    df_t1['test'] = df_wide['test1']
    df_t1['post'] = 0

    df_t2 = df_wide[['name_id', 'treatment', 'emoji', 'male', 'freshman',
                     'sophomore', 'junior', 'senior', 'econmajor', 'hours_semester',
                     'econ_first', 'job', 'whitenonhispanic', 'black', 'whitehispanic',
                     'asian', 'race_other', 'gpa_categorical', 'class_cat', 'test1']].copy()
    df_t2['test'] = df_wide['test2']
    df_t2['post'] = 1

    df_long = pd.concat([df_t1, df_t2], ignore_index=True)
    df_long['ltest'] = np.log(df_long['test'].clip(lower=1))
    df_long['treat_post'] = df_long['treatment'] * df_long['post']
    df_long['female'] = 1 - df_long['male']

    return df_long


def prepare_quiz_panel(df):
    """Prepare panel data for quiz analysis."""
    # Quiz columns (excluding 1, 2, 17, 20 per original code)
    quiz_cols = [f'quiz_{i}' for i in range(3, 21) if i not in [1, 2, 17, 20]]

    # Filter to valid observations
    df_filt = df[df['flag_test'] == 0].copy()

    id_vars = ['name_id', 'treatment', 'emoji', 'male', 'freshman',
               'sophomore', 'junior', 'senior', 'econmajor', 'hours_semester',
               'econ_first', 'job', 'whitenonhispanic', 'black', 'whitehispanic',
               'asian', 'race_other', 'gpa_categorical', 'class_cat', 'test1']

    df_long = pd.melt(
        df_filt[id_vars + quiz_cols],
        id_vars=id_vars,
        value_vars=quiz_cols,
        var_name='class_day_str',
        value_name='q_score'
    )

    df_long['class_day'] = df_long['class_day_str'].str.replace('quiz_', '').astype(int)
    df_long['post'] = (df_long['class_day'] > 12).astype(int)

    # Attendance indicator
    df_long['attend'] = ((df_long['q_score'] > 0) & (~df_long['q_score'].isna())).astype(int)

    # Conditional score (only if attended)
    df_long['q_score_cond'] = df_long['q_score'].copy()
    df_long.loc[df_long['q_score'] == 0, 'q_score_cond'] = np.nan

    df_long['treat_post'] = df_long['treatment'] * df_long['post']
    df_long['female'] = 1 - df_long['male']

    return df_long


def prepare_homework_panel(df):
    """Prepare panel data for homework analysis."""
    hw_cols = ['online_1', 'online_2', 'online_3', 'online_4', 'online_5',
               'online_6', 'online_7']

    df_filt = df[df['flag_test'] == 0].copy()

    id_vars = ['name_id', 'treatment', 'emoji', 'male', 'freshman',
               'sophomore', 'junior', 'senior', 'econmajor', 'hours_semester',
               'econ_first', 'job', 'whitenonhispanic', 'black', 'whitehispanic',
               'asian', 'race_other', 'gpa_categorical', 'class_cat', 'test1']

    df_long = pd.melt(
        df_filt[id_vars + hw_cols],
        id_vars=id_vars,
        value_vars=hw_cols,
        var_name='hw_number_str',
        value_name='hw_score'
    )

    df_long['hw_number'] = df_long['hw_number_str'].str.replace('online_', '').astype(int)
    df_long['post'] = (df_long['hw_number'] > 4).astype(int)

    df_long['hw_score'] = df_long['hw_score'].fillna(0) * 10

    df_long['hw_binary'] = (df_long['hw_score'] > 0).astype(int)

    df_long['hw_score_cond'] = df_long['hw_score'].copy()
    df_long.loc[df_long['hw_score'] == 0, 'hw_score_cond'] = np.nan

    df_long['treat_post'] = df_long['treatment'] * df_long['post']
    df_long['female'] = 1 - df_long['male']

    return df_long


# ============================================================================
# Result Recording
# ============================================================================

def extract_results(model, spec_id, spec_tree_path, outcome_var, treatment_var,
                    sample_desc, fixed_effects, controls_desc, cluster_var,
                    model_type):
    """Extract results from a pyfixest model."""
    try:
        tidy = model.tidy()

        if treatment_var not in tidy.index:
            print(f"  Warning: {treatment_var} not in coefficients for {spec_id}")
            return None

        row = tidy.loc[treatment_var]
        coef = row['Estimate']
        se = row['Std. Error']
        tstat = row['t value']
        pval = row['Pr(>|t|)']
        ci_lower = row['2.5%']
        ci_upper = row['97.5%']
        r2 = model._r2
        nobs = model._N

        coef_vector = {
            "treatment": {"var": treatment_var, "coef": float(coef), "se": float(se), "pval": float(pval)},
            "controls": [],
            "fixed_effects": fixed_effects.split(' + ') if fixed_effects and fixed_effects != 'None' else [],
            "diagnostics": {}
        }

        for var in tidy.index:
            if var != treatment_var:
                coef_vector["controls"].append({
                    "var": str(var),
                    "coef": float(tidy.loc[var, 'Estimate']),
                    "se": float(tidy.loc[var, 'Std. Error']),
                    "pval": float(tidy.loc[var, 'Pr(>|t|)'])
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
            't_stat': float(tstat),
            'p_value': float(pval),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_obs': int(nobs),
            'r_squared': float(r2) if r2 is not None else None,
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects,
            'controls_desc': controls_desc,
            'cluster_var': cluster_var,
            'model_type': model_type,
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
    except Exception as e:
        print(f"  Error extracting for {spec_id}: {e}")
        return None


def run_spec(formula, data, vcov, spec_id, spec_tree_path, outcome_var,
             treatment_var, sample_desc, fixed_effects, controls_desc,
             cluster_var, model_type, results):
    """Run a specification and record results."""
    try:
        model = pf.feols(formula, data=data, vcov=vcov)
        result = extract_results(model, spec_id, spec_tree_path, outcome_var, treatment_var,
                                 sample_desc, fixed_effects, controls_desc, cluster_var, model_type)
        if result:
            results.append(result)
            return True
    except Exception as e:
        print(f"  {spec_id} failed: {e}")
    return False


# ============================================================================
# Specification Functions
# ============================================================================

def run_test_specifications(df_test, results):
    """Run specifications for test score outcome."""

    print("  Running test score specifications...")

    # Baseline
    run_spec("ltest ~ post + treat_post | name_id", df_test, 'hetero',
             'baseline', 'methods/difference_in_differences.md',
             'ltest', 'treat_post', 'Full estimation sample', 'name_id',
             'post', 'robust', 'Panel FE DiD', results)

    # FE variations
    run_spec("ltest ~ treatment + post + treat_post", df_test, 'hetero',
             'did/fe/none', 'methods/difference_in_differences.md#fixed-effects',
             'ltest', 'treat_post', 'Full sample', 'None',
             'treatment + post', 'robust', 'Pooled OLS', results)

    run_spec("ltest ~ treatment + treat_post | post", df_test, 'hetero',
             'did/fe/time_only', 'methods/difference_in_differences.md#fixed-effects',
             'ltest', 'treat_post', 'Full sample', 'post',
             'treatment', 'robust', 'Time FE', results)

    run_spec("ltest ~ treat_post | name_id + post", df_test, 'hetero',
             'did/fe/twoway', 'methods/difference_in_differences.md#fixed-effects',
             'ltest', 'treat_post', 'Full sample', 'name_id + post',
             'None', 'robust', 'TWFE', results)

    # Controls
    df_demo = df_test.dropna(subset=['male', 'freshman', 'sophomore', 'junior'])
    run_spec("ltest ~ treatment + post + treat_post + male + freshman + sophomore + junior",
             df_demo, 'hetero',
             'did/controls/demographics', 'methods/difference_in_differences.md#control-sets',
             'ltest', 'treat_post', 'With demographics', 'None',
             'demographics', 'robust', 'OLS with controls', results)

    df_gpa = df_test.dropna(subset=['gpa_categorical'])
    run_spec("ltest ~ post + treat_post + gpa_categorical | name_id", df_gpa, 'hetero',
             'did/controls/gpa', 'methods/difference_in_differences.md#control-sets',
             'ltest', 'treat_post', 'With GPA', 'name_id',
             'gpa_categorical', 'robust', 'Panel FE + GPA', results)

    df_full = df_test.dropna(subset=['male', 'freshman', 'sophomore', 'junior', 'econmajor', 'job', 'gpa_categorical'])
    run_spec("ltest ~ treatment + post + treat_post + male + freshman + sophomore + junior + econmajor + job + gpa_categorical",
             df_full, 'hetero',
             'did/controls/full', 'methods/difference_in_differences.md#control-sets',
             'ltest', 'treat_post', 'Full controls', 'None',
             'all demographics', 'robust', 'OLS full controls', results)

    # Functional form
    run_spec("test ~ post + treat_post | name_id", df_test, 'hetero',
             'robust/funcform/levels', 'robustness/functional_form.md',
             'test', 'treat_post', 'Full sample', 'name_id',
             'post', 'robust', 'Panel FE (levels)', results)

    df_test['ihs_test'] = np.arcsinh(df_test['test'])
    run_spec("ihs_test ~ post + treat_post | name_id", df_test, 'hetero',
             'robust/funcform/ihs', 'robustness/functional_form.md',
             'ihs_test', 'treat_post', 'Full sample', 'name_id',
             'post', 'robust', 'Panel FE (IHS)', results)

    df_test['test_std'] = (df_test['test'] - df_test['test'].mean()) / df_test['test'].std()
    run_spec("test_std ~ post + treat_post | name_id", df_test, 'hetero',
             'robust/funcform/standardized', 'robustness/functional_form.md',
             'test_std', 'treat_post', 'Standardized', 'name_id',
             'post', 'robust', 'Panel FE (std)', results)

    # Sample restrictions
    q10 = df_test['test1'].quantile(0.1)
    df_sub = df_test[df_test['test1'] > q10]
    run_spec("ltest ~ post + treat_post | name_id", df_sub, 'hetero',
             'robust/sample/drop_low_baseline', 'robustness/sample_restrictions.md',
             'ltest', 'treat_post', 'Drop bottom 10%', 'name_id',
             'post', 'robust', 'Panel FE', results)

    q90 = df_test['test1'].quantile(0.9)
    df_sub = df_test[df_test['test1'] < q90]
    run_spec("ltest ~ post + treat_post | name_id", df_sub, 'hetero',
             'robust/sample/drop_high_baseline', 'robustness/sample_restrictions.md',
             'ltest', 'treat_post', 'Drop top 10%', 'name_id',
             'post', 'robust', 'Panel FE', results)

    df_wins = df_test.copy()
    q05, q95 = df_wins['test'].quantile([0.05, 0.95])
    df_wins['test_wins'] = df_wins['test'].clip(lower=q05, upper=q95)
    df_wins['ltest_wins'] = np.log(df_wins['test_wins'].clip(lower=1))
    run_spec("ltest_wins ~ post + treat_post | name_id", df_wins, 'hetero',
             'robust/sample/winsor_5pct', 'robustness/sample_restrictions.md',
             'ltest_wins', 'treat_post', 'Winsorized 5%', 'name_id',
             'post', 'robust', 'Panel FE', results)

    q01, q99 = df_test['test'].quantile([0.01, 0.99])
    df_trim = df_test[(df_test['test'] >= q01) & (df_test['test'] <= q99)]
    run_spec("ltest ~ post + treat_post | name_id", df_trim, 'hetero',
             'robust/sample/trim_1pct', 'robustness/sample_restrictions.md',
             'ltest', 'treat_post', 'Trimmed 1%', 'name_id',
             'post', 'robust', 'Panel FE', results)

    df_complete = df_test.dropna(subset=['male', 'freshman', 'sophomore', 'junior', 'senior', 'econmajor', 'job', 'gpa_categorical'])
    run_spec("ltest ~ post + treat_post | name_id", df_complete, 'hetero',
             'robust/sample/complete_demographics', 'robustness/sample_restrictions.md',
             'ltest', 'treat_post', 'Complete demographics', 'name_id',
             'post', 'robust', 'Panel FE', results)

    df_nozero = df_test[df_test['test'] > 0]
    run_spec("ltest ~ post + treat_post | name_id", df_nozero, 'hetero',
             'robust/sample/drop_zeros', 'robustness/sample_restrictions.md',
             'ltest', 'treat_post', 'Drop zeros', 'name_id',
             'post', 'robust', 'Panel FE', results)

    # Heterogeneity: Emoji
    emoji_names = {1: 'sad', 2: 'neut_low', 3: 'neut_high', 4: 'happy'}
    for emoji_cat, emoji_name in emoji_names.items():
        df_sub = df_test[df_test['emoji'] == emoji_cat]
        if len(df_sub) > 40:
            run_spec("ltest ~ post + treat_post | name_id", df_sub, 'hetero',
                     f'robust/het/emoji_{emoji_name}', 'robustness/heterogeneity.md',
                     'ltest', 'treat_post', f'Emoji={emoji_name}', 'name_id',
                     'post', 'robust', 'Panel FE', results)

    # Heterogeneity: Gender
    df_male = df_test[df_test['male'] == 1]
    run_spec("ltest ~ post + treat_post | name_id", df_male, 'hetero',
             'robust/het/by_gender_male', 'robustness/heterogeneity.md',
             'ltest', 'treat_post', 'Male only', 'name_id',
             'post', 'robust', 'Panel FE', results)

    df_female = df_test[df_test['male'] == 0]
    run_spec("ltest ~ post + treat_post | name_id", df_female, 'hetero',
             'robust/het/by_gender_female', 'robustness/heterogeneity.md',
             'ltest', 'treat_post', 'Female only', 'name_id',
             'post', 'robust', 'Panel FE', results)

    df_gender = df_test.dropna(subset=['female']).copy()
    df_gender['treat_post_female'] = df_gender['treat_post'] * df_gender['female']
    df_gender['post_female'] = df_gender['post'] * df_gender['female']
    run_spec("ltest ~ post + treat_post + post_female + treat_post_female | name_id", df_gender, 'hetero',
             'robust/het/interaction_gender', 'robustness/heterogeneity.md',
             'ltest', 'treat_post', 'Gender interaction', 'name_id',
             'post + gender', 'robust', 'Panel FE', results)

    # Heterogeneity: Year
    df_under = df_test[(df_test['freshman'] == 1) | (df_test['sophomore'] == 1)]
    if len(df_under) > 40:
        run_spec("ltest ~ post + treat_post | name_id", df_under, 'hetero',
                 'robust/het/underclassmen', 'robustness/heterogeneity.md',
                 'ltest', 'treat_post', 'Underclassmen', 'name_id',
                 'post', 'robust', 'Panel FE', results)

    df_upper = df_test[(df_test['junior'] == 1) | (df_test['senior'] == 1)]
    if len(df_upper) > 40:
        run_spec("ltest ~ post + treat_post | name_id", df_upper, 'hetero',
                 'robust/het/upperclassmen', 'robustness/heterogeneity.md',
                 'ltest', 'treat_post', 'Upperclassmen', 'name_id',
                 'post', 'robust', 'Panel FE', results)

    # Heterogeneity: Major and job
    df_econ = df_test[df_test['econmajor'] == 1]
    if len(df_econ) > 40:
        run_spec("ltest ~ post + treat_post | name_id", df_econ, 'hetero',
                 'robust/het/econ_major', 'robustness/heterogeneity.md',
                 'ltest', 'treat_post', 'Econ major', 'name_id',
                 'post', 'robust', 'Panel FE', results)

    df_nonecon = df_test[df_test['econmajor'] == 0]
    run_spec("ltest ~ post + treat_post | name_id", df_nonecon, 'hetero',
             'robust/het/non_econ_major', 'robustness/heterogeneity.md',
             'ltest', 'treat_post', 'Non-econ major', 'name_id',
             'post', 'robust', 'Panel FE', results)

    df_job = df_test[df_test['job'] == 1]
    run_spec("ltest ~ post + treat_post | name_id", df_job, 'hetero',
             'robust/het/has_job', 'robustness/heterogeneity.md',
             'ltest', 'treat_post', 'Has job', 'name_id',
             'post', 'robust', 'Panel FE', results)

    df_nojob = df_test[df_test['job'] == 0]
    run_spec("ltest ~ post + treat_post | name_id", df_nojob, 'hetero',
             'robust/het/no_job', 'robustness/heterogeneity.md',
             'ltest', 'treat_post', 'No job', 'name_id',
             'post', 'robust', 'Panel FE', results)

    # Heterogeneity: GPA
    df_gpa = df_test.dropna(subset=['gpa_categorical'])
    median_gpa = df_gpa['gpa_categorical'].median()
    df_high_gpa = df_gpa[df_gpa['gpa_categorical'] >= median_gpa]
    run_spec("ltest ~ post + treat_post | name_id", df_high_gpa, 'hetero',
             'robust/het/high_gpa', 'robustness/heterogeneity.md',
             'ltest', 'treat_post', 'High GPA', 'name_id',
             'post', 'robust', 'Panel FE', results)

    df_low_gpa = df_gpa[df_gpa['gpa_categorical'] < median_gpa]
    run_spec("ltest ~ post + treat_post | name_id", df_low_gpa, 'hetero',
             'robust/het/low_gpa', 'robustness/heterogeneity.md',
             'ltest', 'treat_post', 'Low GPA', 'name_id',
             'post', 'robust', 'Panel FE', results)

    # Heterogeneity: Race
    df_white = df_test[df_test['whitenonhispanic'] == 1]
    run_spec("ltest ~ post + treat_post | name_id", df_white, 'hetero',
             'robust/het/white', 'robustness/heterogeneity.md',
             'ltest', 'treat_post', 'White', 'name_id',
             'post', 'robust', 'Panel FE', results)

    df_nonwhite = df_test[df_test['whitenonhispanic'] == 0]
    run_spec("ltest ~ post + treat_post | name_id", df_nonwhite, 'hetero',
             'robust/het/nonwhite', 'robustness/heterogeneity.md',
             'ltest', 'treat_post', 'Non-white', 'name_id',
             'post', 'robust', 'Panel FE', results)

    # Heterogeneity: First econ
    df_first = df_test[df_test['econ_first'] == 1]
    if len(df_first) > 40:
        run_spec("ltest ~ post + treat_post | name_id", df_first, 'hetero',
                 'robust/het/first_econ', 'robustness/heterogeneity.md',
                 'ltest', 'treat_post', 'First econ', 'name_id',
                 'post', 'robust', 'Panel FE', results)

    df_notfirst = df_test[df_test['econ_first'] == 0]
    if len(df_notfirst) > 40:
        run_spec("ltest ~ post + treat_post | name_id", df_notfirst, 'hetero',
                 'robust/het/not_first_econ', 'robustness/heterogeneity.md',
                 'ltest', 'treat_post', 'Not first econ', 'name_id',
                 'post', 'robust', 'Panel FE', results)

    # Heterogeneity: Course load
    df_hours = df_test.dropna(subset=['hours_semester'])
    median_hours = df_hours['hours_semester'].median()
    df_high_hours = df_hours[df_hours['hours_semester'] >= median_hours]
    run_spec("ltest ~ post + treat_post | name_id", df_high_hours, 'hetero',
             'robust/het/high_course_load', 'robustness/heterogeneity.md',
             'ltest', 'treat_post', 'High hours', 'name_id',
             'post', 'robust', 'Panel FE', results)

    df_low_hours = df_hours[df_hours['hours_semester'] < median_hours]
    run_spec("ltest ~ post + treat_post | name_id", df_low_hours, 'hetero',
             'robust/het/low_course_load', 'robustness/heterogeneity.md',
             'ltest', 'treat_post', 'Low hours', 'name_id',
             'post', 'robust', 'Panel FE', results)

    # Heterogeneity: Baseline test
    median_t1 = df_test['test1'].median()
    df_high_t1 = df_test[df_test['test1'] >= median_t1]
    run_spec("ltest ~ post + treat_post | name_id", df_high_t1, 'hetero',
             'robust/het/high_baseline_test', 'robustness/heterogeneity.md',
             'ltest', 'treat_post', 'High baseline', 'name_id',
             'post', 'robust', 'Panel FE', results)

    df_low_t1 = df_test[df_test['test1'] < median_t1]
    run_spec("ltest ~ post + treat_post | name_id", df_low_t1, 'hetero',
             'robust/het/low_baseline_test', 'robustness/heterogeneity.md',
             'ltest', 'treat_post', 'Low baseline', 'name_id',
             'post', 'robust', 'Panel FE', results)

    # Alternative outcomes
    df_pass = df_test.copy()
    df_pass['pass'] = (df_pass['test'] >= 60).astype(int)
    run_spec("pass ~ post + treat_post | name_id", df_pass, 'hetero',
             'robust/outcome/pass_fail', 'methods/difference_in_differences.md',
             'pass', 'treat_post', 'Pass/fail', 'name_id',
             'post', 'robust', 'Panel FE (LPM)', results)

    df_b = df_test.copy()
    df_b['grade_b'] = (df_b['test'] >= 80).astype(int)
    run_spec("grade_b ~ post + treat_post | name_id", df_b, 'hetero',
             'robust/outcome/grade_b_or_better', 'methods/difference_in_differences.md',
             'grade_b', 'treat_post', 'B or better', 'name_id',
             'post', 'robust', 'Panel FE (LPM)', results)

    df_a = df_test.copy()
    df_a['grade_a'] = (df_a['test'] >= 90).astype(int)
    run_spec("grade_a ~ post + treat_post | name_id", df_a, 'hetero',
             'robust/outcome/grade_a', 'methods/difference_in_differences.md',
             'grade_a', 'treat_post', 'A grade', 'name_id',
             'post', 'robust', 'Panel FE (LPM)', results)

    return results


def run_quiz_specifications(df_quiz, results):
    """Run quiz specifications."""

    print("  Running quiz specifications...")

    df_qscore = df_quiz.dropna(subset=['q_score_cond'])

    run_spec("q_score_cond ~ post + treat_post | name_id", df_qscore, 'hetero',
             'did/outcome/quiz_score', 'methods/difference_in_differences.md',
             'q_score_cond', 'treat_post', 'Quiz cond', 'name_id',
             'post', 'robust', 'Panel FE', results)

    run_spec("attend ~ post + treat_post | name_id", df_quiz, 'hetero',
             'did/outcome/attendance', 'methods/difference_in_differences.md',
             'attend', 'treat_post', 'Attendance', 'name_id',
             'post', 'robust', 'Panel FE (LPM)', results)

    run_spec("attend ~ treatment + post + treat_post", df_quiz, 'hetero',
             'did/outcome/attendance_nofe', 'methods/difference_in_differences.md#fixed-effects',
             'attend', 'treat_post', 'Attendance', 'None',
             'treatment + post', 'robust', 'Pooled OLS', results)

    run_spec("attend ~ treat_post | name_id + class_day", df_quiz, 'hetero',
             'did/outcome/attend_twfe', 'methods/difference_in_differences.md#fixed-effects',
             'attend', 'treat_post', 'Attendance', 'name_id + class_day',
             'None', 'robust', 'TWFE', results)

    run_spec("q_score_cond ~ treat_post | name_id + class_day", df_qscore, 'hetero',
             'did/outcome/quiz_twfe', 'methods/difference_in_differences.md#fixed-effects',
             'q_score_cond', 'treat_post', 'Quiz TWFE', 'name_id + class_day',
             'None', 'robust', 'TWFE', results)

    # Emoji heterogeneity
    emoji_names = {1: 'sad', 2: 'neut_low', 3: 'neut_high', 4: 'happy'}
    for emoji_cat, emoji_name in emoji_names.items():
        df_sub = df_qscore[df_qscore['emoji'] == emoji_cat]
        if len(df_sub) > 100:
            run_spec("q_score_cond ~ post + treat_post | name_id", df_sub, 'hetero',
                     f'robust/het/quiz_emoji_{emoji_name}', 'robustness/heterogeneity.md',
                     'q_score_cond', 'treat_post', f'Quiz {emoji_name}', 'name_id',
                     'post', 'robust', 'Panel FE', results)

    # Gender
    df_male = df_qscore[df_qscore['male'] == 1]
    run_spec("q_score_cond ~ post + treat_post | name_id", df_male, 'hetero',
             'robust/het/quiz_male', 'robustness/heterogeneity.md',
             'q_score_cond', 'treat_post', 'Quiz male', 'name_id',
             'post', 'robust', 'Panel FE', results)

    df_female = df_qscore[df_qscore['male'] == 0]
    run_spec("q_score_cond ~ post + treat_post | name_id", df_female, 'hetero',
             'robust/het/quiz_female', 'robustness/heterogeneity.md',
             'q_score_cond', 'treat_post', 'Quiz female', 'name_id',
             'post', 'robust', 'Panel FE', results)

    df_male_att = df_quiz[df_quiz['male'] == 1]
    run_spec("attend ~ post + treat_post | name_id", df_male_att, 'hetero',
             'robust/het/attend_male', 'robustness/heterogeneity.md',
             'attend', 'treat_post', 'Attend male', 'name_id',
             'post', 'robust', 'Panel FE', results)

    df_female_att = df_quiz[df_quiz['male'] == 0]
    run_spec("attend ~ post + treat_post | name_id", df_female_att, 'hetero',
             'robust/het/attend_female', 'robustness/heterogeneity.md',
             'attend', 'treat_post', 'Attend female', 'name_id',
             'post', 'robust', 'Panel FE', results)

    return results


def run_homework_specifications(df_hw, results):
    """Run homework specifications."""

    print("  Running homework specifications...")

    run_spec("hw_binary ~ post + treat_post | name_id", df_hw, 'hetero',
             'did/outcome/hw_attempted', 'methods/difference_in_differences.md',
             'hw_binary', 'treat_post', 'HW attempt', 'name_id',
             'post', 'robust', 'Panel FE (LPM)', results)

    df_pos = df_hw.dropna(subset=['hw_score_cond'])
    run_spec("hw_score_cond ~ post + treat_post | name_id", df_pos, 'hetero',
             'did/outcome/hw_score', 'methods/difference_in_differences.md',
             'hw_score_cond', 'treat_post', 'HW score', 'name_id',
             'post', 'robust', 'Panel FE', results)

    run_spec("hw_binary ~ treatment + post + treat_post", df_hw, 'hetero',
             'did/outcome/hw_attempted_nofe', 'methods/difference_in_differences.md#fixed-effects',
             'hw_binary', 'treat_post', 'HW attempt', 'None',
             'treatment + post', 'robust', 'Pooled OLS', results)

    run_spec("hw_binary ~ treat_post | name_id + hw_number", df_hw, 'hetero',
             'did/outcome/hw_twfe', 'methods/difference_in_differences.md#fixed-effects',
             'hw_binary', 'treat_post', 'HW TWFE', 'name_id + hw_number',
             'None', 'robust', 'TWFE', results)

    # Emoji heterogeneity
    emoji_names = {1: 'sad', 2: 'neut_low', 3: 'neut_high', 4: 'happy'}
    for emoji_cat, emoji_name in emoji_names.items():
        df_sub = df_hw[df_hw['emoji'] == emoji_cat]
        if len(df_sub) > 100:
            run_spec("hw_binary ~ post + treat_post | name_id", df_sub, 'hetero',
                     f'robust/het/hw_emoji_{emoji_name}', 'robustness/heterogeneity.md',
                     'hw_binary', 'treat_post', f'HW {emoji_name}', 'name_id',
                     'post', 'robust', 'Panel FE', results)

    # Gender
    df_male = df_hw[df_hw['male'] == 1]
    run_spec("hw_binary ~ post + treat_post | name_id", df_male, 'hetero',
             'robust/het/hw_male', 'robustness/heterogeneity.md',
             'hw_binary', 'treat_post', 'HW male', 'name_id',
             'post', 'robust', 'Panel FE', results)

    df_female = df_hw[df_hw['male'] == 0]
    run_spec("hw_binary ~ post + treat_post | name_id", df_female, 'hetero',
             'robust/het/hw_female', 'robustness/heterogeneity.md',
             'hw_binary', 'treat_post', 'HW female', 'name_id',
             'post', 'robust', 'Panel FE', results)

    return results


def run_clustering_specifications(df_test, results):
    """Run clustering variations."""

    print("  Running clustering specifications...")

    run_spec("ltest ~ post + treat_post | name_id", df_test, 'HC1',
             'robust/se/hc1', 'robustness/clustering_variations.md',
             'ltest', 'treat_post', 'Full sample', 'name_id',
             'post', 'HC1', 'Panel FE', results)

    run_spec("ltest ~ post + treat_post | name_id", df_test, 'HC2',
             'robust/se/hc2', 'robustness/clustering_variations.md',
             'ltest', 'treat_post', 'Full sample', 'name_id',
             'post', 'HC2', 'Panel FE', results)

    run_spec("ltest ~ post + treat_post | name_id", df_test, 'HC3',
             'robust/se/hc3', 'robustness/clustering_variations.md',
             'ltest', 'treat_post', 'Full sample', 'name_id',
             'post', 'HC3', 'Panel FE', results)

    run_spec("ltest ~ post + treat_post | name_id", df_test, {'CRV1': 'treatment'},
             'robust/cluster/treatment', 'robustness/clustering_variations.md',
             'ltest', 'treat_post', 'Full sample', 'name_id',
             'post', 'cluster(treatment)', 'Panel FE', results)

    run_spec("ltest ~ post + treat_post | name_id", df_test, {'CRV1': 'emoji'},
             'robust/cluster/emoji', 'robustness/clustering_variations.md',
             'ltest', 'treat_post', 'Full sample', 'name_id',
             'post', 'cluster(emoji)', 'Panel FE', results)

    df_gpa = df_test.dropna(subset=['gpa_categorical'])
    run_spec("ltest ~ post + treat_post | name_id", df_gpa, {'CRV1': 'gpa_categorical'},
             'robust/cluster/gpa', 'robustness/clustering_variations.md',
             'ltest', 'treat_post', 'With GPA', 'name_id',
             'post', 'cluster(gpa)', 'Panel FE', results)

    return results


def run_placebo_specifications(df_test, df_quiz, df_hw, df_raw, results):
    """Run placebo tests."""

    print("  Running placebo specifications...")

    df_wide = df_raw[df_raw['flag_test'] == 0].drop_duplicates(subset='name_id').copy()
    df_wide['ltest1'] = np.log(df_wide['test1'].clip(lower=1))
    run_spec("ltest1 ~ treatment", df_wide.dropna(subset=['ltest1']), 'hetero',
             'robust/placebo/baseline_balance', 'robustness/placebo_tests.md',
             'ltest1', 'treatment', 'Balance check', 'None',
             'None', 'robust', 'OLS', results)

    df_pre = df_quiz[df_quiz['post'] == 0].dropna(subset=['q_score_cond']).copy()
    if len(df_pre) > 100:
        df_pre['fake_post'] = (df_pre['class_day'] > 7).astype(int)
        df_pre['fake_treat_post'] = df_pre['treatment'] * df_pre['fake_post']
        run_spec("q_score_cond ~ fake_post + fake_treat_post | name_id", df_pre, 'hetero',
                 'robust/placebo/pre_quiz_trend', 'robustness/placebo_tests.md',
                 'q_score_cond', 'fake_treat_post', 'Pre-quiz placebo', 'name_id',
                 'fake_post', 'robust', 'Panel FE', results)

    df_pre_hw = df_hw[df_hw['post'] == 0].copy()
    if len(df_pre_hw) > 100:
        df_pre_hw['fake_post'] = (df_pre_hw['hw_number'] > 2).astype(int)
        df_pre_hw['fake_treat_post'] = df_pre_hw['treatment'] * df_pre_hw['fake_post']
        run_spec("hw_binary ~ fake_post + fake_treat_post | name_id", df_pre_hw, 'hetero',
                 'robust/placebo/pre_hw_trend', 'robustness/placebo_tests.md',
                 'hw_binary', 'fake_treat_post', 'Pre-HW placebo', 'name_id',
                 'fake_post', 'robust', 'Panel FE', results)

    df_pre_att = df_quiz[df_quiz['post'] == 0].copy()
    if len(df_pre_att) > 100:
        df_pre_att['fake_post'] = (df_pre_att['class_day'] > 7).astype(int)
        df_pre_att['fake_treat_post'] = df_pre_att['treatment'] * df_pre_att['fake_post']
        run_spec("attend ~ fake_post + fake_treat_post | name_id", df_pre_att, 'hetero',
                 'robust/placebo/pre_attend_trend', 'robustness/placebo_tests.md',
                 'attend', 'fake_treat_post', 'Pre-attend placebo', 'name_id',
                 'fake_post', 'robust', 'Panel FE', results)

    return results


def run_first_difference_specifications(df_raw, results):
    """Run first difference specifications."""

    print("  Running first difference specifications...")

    df_change = df_raw[df_raw['flag_test'] == 0].dropna(subset=['test1', 'test2']).copy()
    df_change['test_change'] = df_change['test2'] - df_change['test1']
    run_spec("test_change ~ treatment", df_change, 'hetero',
             'robust/funcform/first_difference', 'robustness/functional_form.md',
             'test_change', 'treatment', 'First diff', 'None',
             'None', 'robust', 'OLS', results)

    df_change['ltest_change'] = np.log(df_change['test2'].clip(lower=1)) - np.log(df_change['test1'].clip(lower=1))
    run_spec("ltest_change ~ treatment", df_change, 'hetero',
             'robust/funcform/log_first_diff', 'robustness/functional_form.md',
             'ltest_change', 'treatment', 'Log first diff', 'None',
             'None', 'robust', 'OLS', results)

    df_demo = df_change.dropna(subset=['male', 'freshman', 'sophomore', 'junior'])
    run_spec("test_change ~ treatment + male + freshman + sophomore + junior", df_demo, 'hetero',
             'robust/funcform/first_diff_controls', 'robustness/functional_form.md',
             'test_change', 'treatment', 'First diff + controls', 'None',
             'demographics', 'robust', 'OLS', results)

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    print(f"Starting specification search for {PAPER_ID}")
    print("=" * 60)

    print("Loading data...")
    df_raw = load_and_prepare_data()

    df_test = prepare_test_panel(df_raw)
    df_quiz = prepare_quiz_panel(df_raw)
    df_hw = prepare_homework_panel(df_raw)

    print(f"Test panel: {len(df_test)} obs")
    print(f"Quiz panel: {len(df_quiz)} obs")
    print(f"Homework panel: {len(df_hw)} obs")

    results = []

    print("\nRunning specifications...")
    results = run_test_specifications(df_test, results)
    print(f"  After test: {len(results)} specs")

    results = run_quiz_specifications(df_quiz, results)
    print(f"  After quiz: {len(results)} specs")

    results = run_homework_specifications(df_hw, results)
    print(f"  After HW: {len(results)} specs")

    results = run_clustering_specifications(df_test, results)
    print(f"  After clustering: {len(results)} specs")

    results = run_placebo_specifications(df_test, df_quiz, df_hw, df_raw, results)
    print(f"  After placebo: {len(results)} specs")

    results = run_first_difference_specifications(df_raw, results)
    print(f"  After first diff: {len(results)} specs")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(results_df)} specs to {OUTPUT_PATH}")

    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    if len(results_df) > 0:
        print(f"\nTotal: {len(results_df)}")
        print(f"Positive coef: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
        print(f"Sig at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
        print(f"Sig at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")
        print(f"Sig at 10%: {(results_df['p_value'] < 0.10).sum()} ({100*(results_df['p_value'] < 0.10).mean():.1f}%)")

        ltest_results = results_df[results_df['outcome_var'] == 'ltest']
        if len(ltest_results) > 0:
            print(f"\nLog test outcome ({len(ltest_results)} specs):")
            print(f"  Median coef: {ltest_results['coefficient'].median():.4f}")
            print(f"  Mean coef: {ltest_results['coefficient'].mean():.4f}")
            print(f"  Range: [{ltest_results['coefficient'].min():.4f}, {ltest_results['coefficient'].max():.4f}]")

    return results_df


if __name__ == "__main__":
    results = main()
