"""
Specification Search Analysis for Paper 183985-V1
"Emoticons as Performance Feedback for College Students: A Large-Classroom Field Experiment"
By Darshak Patel and Justin Roush

This script replicates the main DiD specifications and runs systematic robustness checks.

The paper studies whether emoticon-based performance feedback affects student effort and
performance. Treatment students receive emoticon feedback based on their Exam 1 score.

Key outcomes:
- Test scores (Exam 1 pre-treatment, Exam 2 post-treatment)
- Quiz attendance
- Quiz scores (conditional on attendance)
- Homework attempt
- Homework scores (conditional on attempt)

Design: DiD with panel fixed effects (student FE)
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings('ignore')

# Try to import pyfixest; fall back to linearmodels if needed
try:
    import pyfixest as pf
    HAS_PYFIXEST = True
except ImportError:
    HAS_PYFIXEST = False

try:
    from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS
    HAS_LINEARMODELS = True
except ImportError:
    HAS_LINEARMODELS = False

# Paths
DATA_DIR = Path("/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/183985-V1")
OUTPUT_DIR = DATA_DIR

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def load_data():
    """Load the Stata dataset."""
    data_path = DATA_DIR / "aeapp_patelroush_data.dta"
    df = pd.read_stata(data_path)
    return df

def prepare_exam_data(df):
    """
    Prepare data for exam analysis (main DiD specification).
    Replicates the Stata code's data preparation.

    Structure:
    - Each student has test1 (pre) and test2 (post)
    - Reshape to long format with test_num indicator
    - Filter to students where flag_test == 0 (picked up exam 1 and didn't drop)
    """
    df = df.copy()

    # Filter to estimation sample (flag_test == 0)
    df = df[df['flag_test'] == 0].copy()

    # Create emoji variable based on test1 scores
    df['emoji'] = 1  # Sad (default, <68)
    df.loc[(df['test1'] > 67) & (df['test1'] < 80), 'emoji'] = 2  # Neutral Low
    df.loc[(df['test1'] > 79) & (df['test1'] < 91), 'emoji'] = 3  # Neutral High
    df.loc[df['test1'] > 90, 'emoji'] = 4  # Happy

    # Create female variable (if male exists)
    if 'male' in df.columns:
        df['female'] = (df['male'] == 0).astype(int)

    # Reshape from wide to long (test1, test2 -> test with test_num indicator)
    id_vars = ['name_id', 'treatment', 'male', 'emoji', 'flag_test']
    if 'female' in df.columns:
        id_vars.append('female')

    # Keep only columns we need for reshaping
    df_wide = df[id_vars + ['test1', 'test2']].copy()

    df_long = pd.melt(
        df_wide,
        id_vars=id_vars,
        value_vars=['test1', 'test2'],
        var_name='test_var',
        value_name='test'
    )

    # Create test_num (1 or 2)
    df_long['test_num'] = df_long['test_var'].apply(lambda x: 1 if x == 'test1' else 2)

    # Drop rows with missing test scores
    df_long = df_long.dropna(subset=['test'])

    # Create log test score
    df_long['ltest'] = np.log(df_long['test'].replace(0, np.nan))

    # Create post indicator (test_num == 2 means post-treatment)
    df_long['post'] = (df_long['test_num'] == 2).astype(int)

    # Create treatment x post interaction
    df_long['treat_post'] = df_long['treatment'] * df_long['post']

    return df_long

def prepare_quiz_data(df):
    """
    Prepare data for quiz/attendance analysis.

    Quiz data is wide: quiz_1, quiz_2, ..., quiz_26 columns
    Reshape to long with class_day indicator
    Filter as per Stata code:
    - Keep class_day <= 20 (last quiz before test 2)
    - Drop days 1, 2 (all received 100%)
    - Drop day 17 (outlier)
    - Drop day 20 (all earned 5s)

    Post indicator: class_day > 12 (feedback received after quiz 11/12)
    """
    df = df.copy()

    # Filter to estimation sample
    df = df[df['flag_test'] == 0].copy()

    # Create emoji variable
    df['emoji'] = 1
    df.loc[(df['test1'] > 67) & (df['test1'] < 80), 'emoji'] = 2
    df.loc[(df['test1'] > 79) & (df['test1'] < 91), 'emoji'] = 3
    df.loc[df['test1'] > 90, 'emoji'] = 4

    if 'male' in df.columns:
        df['female'] = (df['male'] == 0).astype(int)

    # Find quiz columns
    quiz_cols = [c for c in df.columns if c.startswith('quiz_') and c[5:].isdigit()]

    if not quiz_cols:
        return None

    # ID columns to keep
    id_vars = ['name_id', 'treatment', 'emoji']
    if 'male' in df.columns:
        id_vars.append('male')
    if 'female' in df.columns:
        id_vars.append('female')

    df_long = pd.melt(
        df[id_vars + quiz_cols],
        id_vars=id_vars,
        value_vars=quiz_cols,
        var_name='quiz_var',
        value_name='q_score'
    )

    # Extract class_day from quiz variable name
    df_long['class_day'] = df_long['quiz_var'].str.extract(r'(\d+)').astype(int)

    # Create post indicator (feedback received after quiz 12)
    df_long['post'] = (df_long['class_day'] > 12).astype(int)

    # Filter as per Stata code
    df_long = df_long[df_long['class_day'] <= 20]  # Drop after last quiz before test
    df_long = df_long[~df_long['class_day'].isin([1, 2])]  # All received 100%
    df_long = df_long[df_long['class_day'] != 17]  # Outlier
    df_long = df_long[df_long['class_day'] != 20]  # All students earned 5s

    # Scale quiz score to 100
    df_long['q_score_scaled'] = 100 * (df_long['q_score'] / 5)

    # Create attendance indicator
    df_long['attend'] = ((df_long['q_score'] > 0) & (df_long['q_score'].notna())).astype(int)

    # For quiz score analysis, set to NA if didn't attend
    df_long['q_score_cond'] = df_long['q_score_scaled'].copy()
    df_long.loc[df_long['attend'] == 0, 'q_score_cond'] = np.nan

    # Create interaction
    df_long['treat_post'] = df_long['treatment'] * df_long['post']

    return df_long

def prepare_homework_data(df):
    """
    Prepare data for homework analysis.

    HW data is wide: online_1, online_2, ..., online_14 columns
    Reshape to long
    Filter as per Stata code:
    - Drop chapter 8 (group homework)
    - Remap chapter 10 to 8 (no hw for ch9)
    - Drop chapters > 8

    Post indicator: chapter > 4 (feedback after ch 4)
    """
    df = df.copy()

    # Filter to estimation sample
    df = df[df['flag_test'] == 0].copy()

    # Create emoji variable
    df['emoji'] = 1
    df.loc[(df['test1'] > 67) & (df['test1'] < 80), 'emoji'] = 2
    df.loc[(df['test1'] > 79) & (df['test1'] < 91), 'emoji'] = 3
    df.loc[df['test1'] > 90, 'emoji'] = 4

    if 'male' in df.columns:
        df['female'] = (df['male'] == 0).astype(int)

    # Find online homework columns
    hw_cols = [c for c in df.columns if c.startswith('online_') and c[7:].isdigit()]

    if not hw_cols:
        return None

    # ID columns
    id_vars = ['name_id', 'treatment', 'emoji']
    if 'male' in df.columns:
        id_vars.append('male')
    if 'female' in df.columns:
        id_vars.append('female')

    df_long = pd.melt(
        df[id_vars + hw_cols],
        id_vars=id_vars,
        value_vars=hw_cols,
        var_name='hw_var',
        value_name='hw_score'
    )

    # Extract chapter from hw variable name
    df_long['chapter'] = df_long['hw_var'].str.extract(r'(\d+)').astype(int)

    # Fill missing with 0
    df_long['hw_score'] = df_long['hw_score'].fillna(0)

    # Scale to 100
    df_long['hw_score_scaled'] = 100 * (df_long['hw_score'] / 10)

    # Drop chapter 8 (group homework)
    df_long = df_long[df_long['chapter'] != 8]

    # Remap chapter 10 to 8
    df_long.loc[df_long['chapter'] == 10, 'chapter'] = 8

    # Drop chapters > 8
    df_long = df_long[df_long['chapter'] <= 8]

    # Post indicator (feedback after chapter 4)
    df_long['post'] = (df_long['chapter'] > 4).astype(int)

    # Attempted HW indicator
    df_long['attempted_hw'] = ((df_long['hw_score'] > 0) & (df_long['hw_score'].notna())).astype(int)

    # For score analysis, set to NA if not attempted
    df_long['hw_score_cond'] = df_long['hw_score_scaled'].copy()
    df_long.loc[df_long['attempted_hw'] == 0, 'hw_score_cond'] = np.nan

    # Create interaction
    df_long['treat_post'] = df_long['treatment'] * df_long['post']

    return df_long

# ============================================================================
# REGRESSION FUNCTIONS
# ============================================================================

def run_panel_fe_regression(df, outcome, entity_id, time_id, treatment_var='treat_post',
                            additional_vars=None, cluster_var=None, fe_type='unit',
                            vcov_type='hetero'):
    """
    Run a panel fixed effects regression using pyfixest.

    Parameters:
    -----------
    df : DataFrame
    outcome : str - dependent variable
    entity_id : str - unit identifier
    time_id : str - time identifier
    treatment_var : str - treatment variable (typically treat_post for DiD)
    additional_vars : list - additional regressors (not absorbed)
    cluster_var : str - clustering variable for standard errors
    fe_type : str - 'unit', 'time', 'twoway', 'none'
    vcov_type : str - 'hetero', 'iid', or cluster specification

    Returns:
    --------
    dict with regression results
    """
    df = df.copy()

    # Build list of variables to keep
    keep_vars = [outcome, treatment_var, entity_id, time_id]
    if additional_vars:
        keep_vars.extend(additional_vars)
    if cluster_var and cluster_var not in keep_vars:
        keep_vars.append(cluster_var)

    # Only keep rows without NA in required columns
    df = df.dropna(subset=[outcome, treatment_var, entity_id, time_id])
    if additional_vars:
        df = df.dropna(subset=additional_vars)

    if len(df) == 0:
        return {'error': 'No data after dropping NAs'}

    results = {}

    try:
        if HAS_PYFIXEST:
            # Build formula
            if additional_vars:
                rhs = treatment_var + ' + ' + ' + '.join(additional_vars)
            else:
                rhs = treatment_var

            # Add fixed effects
            if fe_type == 'twoway':
                formula = f"{outcome} ~ {rhs} | {entity_id} + {time_id}"
            elif fe_type == 'unit':
                formula = f"{outcome} ~ {rhs} | {entity_id}"
            elif fe_type == 'time':
                formula = f"{outcome} ~ {rhs} | {time_id}"
            else:
                formula = f"{outcome} ~ {rhs}"

            # Set vcov
            if cluster_var:
                vcov = {'CRV1': cluster_var}
            else:
                vcov = vcov_type

            model = pf.feols(formula, data=df, vcov=vcov)

            results['coef'] = float(model.coef()[treatment_var])
            results['se'] = float(model.se()[treatment_var])
            results['pval'] = float(model.pvalue()[treatment_var])
            results['tstat'] = float(model.tstat()[treatment_var])
            results['n_obs'] = int(model._N)  # pyfixest uses _N
            results['r2'] = float(model._r2) if model._r2 is not None else None  # pyfixest uses _r2

            # Get CI from tidy output
            tidy = model.tidy()
            if treatment_var in tidy.index:
                results['ci_lower'] = float(tidy.loc[treatment_var, '2.5%'])
                results['ci_upper'] = float(tidy.loc[treatment_var, '97.5%'])

            # Get other coefficients
            coef_dict = {}
            for var in model.coef().index:
                coef_dict[var] = {
                    'coef': float(model.coef()[var]),
                    'se': float(model.se()[var]),
                    'pval': float(model.pvalue()[var])
                }
            results['coefficients'] = coef_dict

        else:
            # Fall back to OLS with dummy variables
            y = df[outcome]
            X_vars = [treatment_var]
            if additional_vars:
                X_vars.extend(additional_vars)
            X = df[X_vars].copy()

            # Create numeric IDs for dummies
            df['_entity_id'] = pd.Categorical(df[entity_id]).codes
            df['_time_id'] = pd.Categorical(df[time_id]).codes

            if fe_type in ['unit', 'twoway']:
                entity_dummies = pd.get_dummies(df['_entity_id'], prefix='entity', drop_first=True)
                X = pd.concat([X, entity_dummies], axis=1)
            if fe_type in ['time', 'twoway']:
                time_dummies = pd.get_dummies(df['_time_id'], prefix='time', drop_first=True)
                X = pd.concat([X, time_dummies], axis=1)

            X = sm.add_constant(X)
            model = sm.OLS(y, X)

            if cluster_var:
                result = model.fit(cov_type='cluster', cov_kwds={'groups': df[cluster_var]})
            else:
                result = model.fit(cov_type='HC1')

            results['coef'] = float(result.params[treatment_var])
            results['se'] = float(result.bse[treatment_var])
            results['pval'] = float(result.pvalues[treatment_var])
            results['tstat'] = float(result.tvalues[treatment_var])
            results['n_obs'] = int(result.nobs)
            results['r2'] = float(result.rsquared)

            ci = result.conf_int()
            results['ci_lower'] = float(ci.loc[treatment_var, 0])
            results['ci_upper'] = float(ci.loc[treatment_var, 1])

            coef_dict = {}
            for var in result.params.index:
                if not var.startswith(('entity', 'time', 'const')):
                    coef_dict[var] = {
                        'coef': float(result.params[var]),
                        'se': float(result.bse[var]),
                        'pval': float(result.pvalues[var])
                    }
            results['coefficients'] = coef_dict

    except Exception as e:
        results['error'] = str(e)

    return results

def run_pooled_ols(df, outcome, treatment_var='treat_post', additional_vars=None, robust=True):
    """Run pooled OLS without fixed effects."""
    df = df.copy().dropna(subset=[outcome, treatment_var])
    if additional_vars:
        df = df.dropna(subset=additional_vars)

    if len(df) == 0:
        return {'error': 'No data after dropping NAs'}

    y = df[outcome]
    X_vars = [treatment_var]
    if additional_vars:
        X_vars.extend(additional_vars)
    X = df[X_vars]
    X = sm.add_constant(X)

    model = sm.OLS(y, X)
    result = model.fit(cov_type='HC1' if robust else 'nonrobust')

    results = {
        'coef': float(result.params[treatment_var]),
        'se': float(result.bse[treatment_var]),
        'pval': float(result.pvalues[treatment_var]),
        'tstat': float(result.tvalues[treatment_var]),
        'n_obs': int(result.nobs),
        'r2': float(result.rsquared)
    }

    ci = result.conf_int()
    results['ci_lower'] = float(ci.loc[treatment_var, 0])
    results['ci_upper'] = float(ci.loc[treatment_var, 1])

    return results

# ============================================================================
# SPECIFICATION SEARCH
# ============================================================================

def create_result_row(spec_id, spec_tree_path, outcome_var, results, additional_info=None):
    """Create a standardized result row for the output CSV."""
    row = {
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'outcome_var': outcome_var,
        'coef': results.get('coef'),
        'se': results.get('se'),
        'pval': results.get('pval'),
        'tstat': results.get('tstat'),
        'ci_lower': results.get('ci_lower'),
        'ci_upper': results.get('ci_upper'),
        'n_obs': results.get('n_obs'),
        'r2': results.get('r2'),
        'error': results.get('error'),
        'coefficient_vector_json': json.dumps(results.get('coefficients', {}))
    }

    if additional_info:
        row.update(additional_info)

    return row

def run_specification_search():
    """Main function to run all specifications."""

    print("Loading data...")
    df_raw = load_data()
    print(f"Raw data shape: {df_raw.shape}")
    print(f"Columns: {list(df_raw.columns)}")

    results = []

    # ========================================================================
    # EXAM ANALYSIS (Main DiD specification)
    # ========================================================================
    print("\n" + "="*60)
    print("EXAM ANALYSIS")
    print("="*60)

    df_exam = prepare_exam_data(df_raw)
    print(f"Exam data shape: {df_exam.shape}")
    print(f"Treatment distribution:\n{df_exam['treatment'].value_counts()}")
    print(f"Post distribution:\n{df_exam['post'].value_counts()}")
    print(f"treat_post distribution:\n{df_exam['treat_post'].value_counts()}")

    entity_id = 'name_id'
    time_id = 'test_num'

    # ------------------------------------------------------------------
    # BASELINE: Main DiD with unit FE (Table 2, Column 5)
    # Stata: xtreg ltest i.post i.treatment i.treatment#i.post if flag_test==0&log_sample==1,vce(robust) fe
    # ------------------------------------------------------------------
    print("\nRunning baseline exam specification...")
    baseline_exam = run_panel_fe_regression(
        df_exam, 'ltest', entity_id, time_id, 'treat_post',
        additional_vars=['post'],  # post is explicitly included in Stata
        cluster_var=None, fe_type='unit'
    )
    if baseline_exam and 'error' not in baseline_exam:
        print(f"  Baseline coefficient: {baseline_exam['coef']:.4f} (SE: {baseline_exam['se']:.4f}, p={baseline_exam['pval']:.4f})")
        results.append(create_result_row(
            'baseline', 'baseline', 'ltest', baseline_exam,
            {'fe_type': 'unit', 'cluster': 'robust', 'outcome_type': 'exam'}
        ))
    else:
        print(f"  Error: {baseline_exam.get('error') if baseline_exam else 'No data'}")

    # ------------------------------------------------------------------
    # DiD FIXED EFFECTS VARIATIONS
    # ------------------------------------------------------------------
    print("\nRunning DiD FE variations...")

    # Time FE only
    fe_time = run_panel_fe_regression(
        df_exam, 'ltest', entity_id, time_id, 'treat_post',
        additional_vars=['post', 'treatment'], fe_type='time'
    )
    if fe_time and 'error' not in fe_time:
        results.append(create_result_row(
            'did/fe/time_only', 'methods/difference_in_differences.md#fixed-effects',
            'ltest', fe_time, {'fe_type': 'time', 'outcome_type': 'exam'}
        ))

    # No FE (pooled OLS)
    fe_none = run_pooled_ols(df_exam, 'ltest', 'treat_post', additional_vars=['treatment', 'post'])
    if fe_none and 'error' not in fe_none:
        results.append(create_result_row(
            'did/fe/none', 'methods/difference_in_differences.md#fixed-effects',
            'ltest', fe_none, {'fe_type': 'none', 'outcome_type': 'exam'}
        ))

    # Two-way FE (unit + time)
    fe_twoway = run_panel_fe_regression(
        df_exam, 'ltest', entity_id, time_id, 'treat_post',
        additional_vars=None, fe_type='twoway'
    )
    if fe_twoway and 'error' not in fe_twoway:
        results.append(create_result_row(
            'did/fe/twoway', 'methods/difference_in_differences.md#fixed-effects',
            'ltest', fe_twoway, {'fe_type': 'twoway', 'outcome_type': 'exam'}
        ))

    # ------------------------------------------------------------------
    # CLUSTERING VARIATIONS
    # ------------------------------------------------------------------
    print("\nRunning clustering variations...")

    # Cluster by entity (student)
    cluster_entity = run_panel_fe_regression(
        df_exam, 'ltest', entity_id, time_id, 'treat_post',
        additional_vars=['post'], cluster_var=entity_id, fe_type='unit'
    )
    if cluster_entity and 'error' not in cluster_entity:
        results.append(create_result_row(
            'robust/cluster/unit', 'robustness/clustering_variations.md#single-level-clustering',
            'ltest', cluster_entity, {'cluster': 'name_id', 'outcome_type': 'exam'}
        ))

    # Cluster by treatment
    cluster_treat = run_panel_fe_regression(
        df_exam, 'ltest', entity_id, time_id, 'treat_post',
        additional_vars=['post'], cluster_var='treatment', fe_type='unit'
    )
    if cluster_treat and 'error' not in cluster_treat:
        results.append(create_result_row(
            'robust/cluster/treatment_group', 'robustness/clustering_variations.md#single-level-clustering',
            'ltest', cluster_treat, {'cluster': 'treatment', 'outcome_type': 'exam'}
        ))

    # Cluster by emoji (treatment intensity)
    cluster_emoji = run_panel_fe_regression(
        df_exam, 'ltest', entity_id, time_id, 'treat_post',
        additional_vars=['post'], cluster_var='emoji', fe_type='unit'
    )
    if cluster_emoji and 'error' not in cluster_emoji:
        results.append(create_result_row(
            'robust/cluster/emoji', 'robustness/clustering_variations.md#single-level-clustering',
            'ltest', cluster_emoji, {'cluster': 'emoji', 'outcome_type': 'exam'}
        ))

    # ------------------------------------------------------------------
    # SAMPLE RESTRICTIONS
    # ------------------------------------------------------------------
    print("\nRunning sample restrictions...")

    # By emoji group (heterogeneity / treatment intensity)
    emoji_names = {1: 'sad', 2: 'neutral_low', 3: 'neutral_high', 4: 'happy'}
    for emoji_val, emoji_name in emoji_names.items():
        df_emoji = df_exam[df_exam['emoji'] == emoji_val].copy()
        if len(df_emoji) >= 20:  # Need enough observations
            sample_emoji = run_panel_fe_regression(
                df_emoji, 'ltest', entity_id, time_id, 'treat_post',
                additional_vars=['post'], fe_type='unit'
            )
            if sample_emoji and 'error' not in sample_emoji:
                results.append(create_result_row(
                    f'robust/sample/emoji_{emoji_name}',
                    'robustness/sample_restrictions.md',
                    'ltest', sample_emoji,
                    {'emoji_group': emoji_val, 'sample_n': len(df_emoji)//2, 'outcome_type': 'exam'}
                ))

    # Exclude low performers (test1 < 60)
    df_no_low = df_exam[df_exam['test1'] >= 60].copy() if 'test1' in df_exam.columns else None
    if df_no_low is not None and len(df_no_low) >= 20:
        sample_no_low = run_panel_fe_regression(
            df_no_low, 'ltest', entity_id, time_id, 'treat_post',
            additional_vars=['post'], fe_type='unit'
        )
        if sample_no_low and 'error' not in sample_no_low:
            results.append(create_result_row(
                'robust/sample/exclude_low_performers',
                'robustness/sample_restrictions.md',
                'ltest', sample_no_low, {'restriction': 'test1>=60', 'outcome_type': 'exam'}
            ))

    # Winsorize outcome at 1%/99%
    df_winsor = df_exam.copy()
    p1, p99 = df_winsor['ltest'].quantile([0.01, 0.99])
    df_winsor['ltest_winsor'] = df_winsor['ltest'].clip(lower=p1, upper=p99)
    sample_winsor = run_panel_fe_regression(
        df_winsor, 'ltest_winsor', entity_id, time_id, 'treat_post',
        additional_vars=['post'], fe_type='unit'
    )
    if sample_winsor and 'error' not in sample_winsor:
        results.append(create_result_row(
            'robust/sample/winsor_1pct',
            'robustness/sample_restrictions.md',
            'ltest_winsor', sample_winsor, {'transformation': 'winsorized_1_99', 'outcome_type': 'exam'}
        ))

    # ------------------------------------------------------------------
    # FUNCTIONAL FORM
    # ------------------------------------------------------------------
    print("\nRunning functional form variations...")

    # Level outcome (not log)
    form_level = run_panel_fe_regression(
        df_exam, 'test', entity_id, time_id, 'treat_post',
        additional_vars=['post'], fe_type='unit'
    )
    if form_level and 'error' not in form_level:
        results.append(create_result_row(
            'robust/form/y_level', 'robustness/functional_form.md',
            'test', form_level, {'transformation': 'level', 'outcome_type': 'exam'}
        ))

    # Standardized outcome
    df_exam['test_z'] = (df_exam['test'] - df_exam['test'].mean()) / df_exam['test'].std()
    form_std = run_panel_fe_regression(
        df_exam, 'test_z', entity_id, time_id, 'treat_post',
        additional_vars=['post'], fe_type='unit'
    )
    if form_std and 'error' not in form_std:
        results.append(create_result_row(
            'robust/form/y_standardized', 'robustness/functional_form.md',
            'test_z', form_std, {'transformation': 'standardized', 'outcome_type': 'exam'}
        ))

    # ========================================================================
    # QUIZ/ATTENDANCE ANALYSIS
    # ========================================================================
    print("\n" + "="*60)
    print("QUIZ/ATTENDANCE ANALYSIS")
    print("="*60)

    df_quiz = prepare_quiz_data(df_raw)
    if df_quiz is not None and len(df_quiz) > 0:
        print(f"Quiz data shape: {df_quiz.shape}")
        print(f"Post distribution:\n{df_quiz['post'].value_counts()}")

        entity_id_quiz = 'name_id'
        time_id_quiz = 'class_day'

        # ------------------------------------------------------------------
        # BASELINE: Attendance DiD
        # ------------------------------------------------------------------
        print("\nRunning attendance specification...")
        attend_did = run_panel_fe_regression(
            df_quiz, 'attend', entity_id_quiz, time_id_quiz, 'treat_post',
            additional_vars=['post'], fe_type='unit'
        )
        if attend_did and 'error' not in attend_did:
            print(f"  Attendance coefficient: {attend_did['coef']:.4f} (SE: {attend_did['se']:.4f})")
            results.append(create_result_row(
                'baseline_attend', 'methods/difference_in_differences.md',
                'attend', attend_did, {'fe_type': 'unit', 'outcome_type': 'attendance'}
            ))

        # ------------------------------------------------------------------
        # BASELINE: Quiz score DiD (conditional on attendance)
        # ------------------------------------------------------------------
        print("\nRunning quiz score specification...")
        quiz_did = run_panel_fe_regression(
            df_quiz, 'q_score_cond', entity_id_quiz, time_id_quiz, 'treat_post',
            additional_vars=['post'], fe_type='unit'
        )
        if quiz_did and 'error' not in quiz_did:
            print(f"  Quiz score coefficient: {quiz_did['coef']:.4f} (SE: {quiz_did['se']:.4f})")
            results.append(create_result_row(
                'baseline_quiz', 'methods/difference_in_differences.md',
                'q_score_cond', quiz_did, {'fe_type': 'unit', 'outcome_type': 'quiz_score'}
            ))

        # FE variations for attendance
        attend_fe_none = run_pooled_ols(df_quiz, 'attend', 'treat_post', additional_vars=['treatment', 'post'])
        if attend_fe_none and 'error' not in attend_fe_none:
            results.append(create_result_row(
                'did/fe/none_attend', 'methods/difference_in_differences.md#fixed-effects',
                'attend', attend_fe_none, {'fe_type': 'none', 'outcome_type': 'attendance'}
            ))

        attend_fe_twoway = run_panel_fe_regression(
            df_quiz, 'attend', entity_id_quiz, time_id_quiz, 'treat_post',
            additional_vars=None, fe_type='twoway'
        )
        if attend_fe_twoway and 'error' not in attend_fe_twoway:
            results.append(create_result_row(
                'did/fe/twoway_attend', 'methods/difference_in_differences.md#fixed-effects',
                'attend', attend_fe_twoway, {'fe_type': 'twoway', 'outcome_type': 'attendance'}
            ))

        # Clustering variations for attendance
        attend_cluster_unit = run_panel_fe_regression(
            df_quiz, 'attend', entity_id_quiz, time_id_quiz, 'treat_post',
            additional_vars=['post'], cluster_var=entity_id_quiz, fe_type='unit'
        )
        if attend_cluster_unit and 'error' not in attend_cluster_unit:
            results.append(create_result_row(
                'robust/cluster/unit_attend', 'robustness/clustering_variations.md#single-level-clustering',
                'attend', attend_cluster_unit, {'cluster': 'name_id', 'outcome_type': 'attendance'}
            ))

        attend_cluster_time = run_panel_fe_regression(
            df_quiz, 'attend', entity_id_quiz, time_id_quiz, 'treat_post',
            additional_vars=['post'], cluster_var='class_day', fe_type='unit'
        )
        if attend_cluster_time and 'error' not in attend_cluster_time:
            results.append(create_result_row(
                'robust/cluster/time_attend', 'robustness/clustering_variations.md#single-level-clustering',
                'attend', attend_cluster_time, {'cluster': 'class_day', 'outcome_type': 'attendance'}
            ))

        # Sample restrictions for attendance (by emoji)
        for emoji_val, emoji_name in emoji_names.items():
            df_quiz_emoji = df_quiz[df_quiz['emoji'] == emoji_val].copy()
            if len(df_quiz_emoji) >= 100:
                attend_emoji = run_panel_fe_regression(
                    df_quiz_emoji, 'attend', entity_id_quiz, time_id_quiz, 'treat_post',
                    additional_vars=['post'], fe_type='unit'
                )
                if attend_emoji and 'error' not in attend_emoji:
                    results.append(create_result_row(
                        f'robust/sample/emoji_{emoji_name}_attend',
                        'robustness/sample_restrictions.md',
                        'attend', attend_emoji,
                        {'emoji_group': emoji_val, 'outcome_type': 'attendance'}
                    ))

    else:
        print("Could not prepare quiz data")

    # ========================================================================
    # HOMEWORK ANALYSIS
    # ========================================================================
    print("\n" + "="*60)
    print("HOMEWORK ANALYSIS")
    print("="*60)

    df_hw = prepare_homework_data(df_raw)
    if df_hw is not None and len(df_hw) > 0:
        print(f"Homework data shape: {df_hw.shape}")
        print(f"Post distribution:\n{df_hw['post'].value_counts()}")

        entity_id_hw = 'name_id'
        time_id_hw = 'chapter'

        # ------------------------------------------------------------------
        # BASELINE: Homework attempt DiD
        # ------------------------------------------------------------------
        print("\nRunning homework attempt specification...")
        hw_attempt_did = run_panel_fe_regression(
            df_hw, 'attempted_hw', entity_id_hw, time_id_hw, 'treat_post',
            additional_vars=['post'], fe_type='unit'
        )
        if hw_attempt_did and 'error' not in hw_attempt_did:
            print(f"  HW attempt coefficient: {hw_attempt_did['coef']:.4f} (SE: {hw_attempt_did['se']:.4f})")
            results.append(create_result_row(
                'baseline_hw_attempt', 'methods/difference_in_differences.md',
                'attempted_hw', hw_attempt_did, {'fe_type': 'unit', 'outcome_type': 'hw_attempt'}
            ))

        # ------------------------------------------------------------------
        # BASELINE: Homework score DiD (conditional on attempt)
        # ------------------------------------------------------------------
        print("\nRunning homework score specification...")
        hw_score_did = run_panel_fe_regression(
            df_hw, 'hw_score_cond', entity_id_hw, time_id_hw, 'treat_post',
            additional_vars=['post'], fe_type='unit'
        )
        if hw_score_did and 'error' not in hw_score_did:
            print(f"  HW score coefficient: {hw_score_did['coef']:.4f} (SE: {hw_score_did['se']:.4f})")
            results.append(create_result_row(
                'baseline_hw_score', 'methods/difference_in_differences.md',
                'hw_score_cond', hw_score_did, {'fe_type': 'unit', 'outcome_type': 'hw_score'}
            ))

        # FE variations for homework attempt
        hw_attempt_fe_none = run_pooled_ols(df_hw, 'attempted_hw', 'treat_post', additional_vars=['treatment', 'post'])
        if hw_attempt_fe_none and 'error' not in hw_attempt_fe_none:
            results.append(create_result_row(
                'did/fe/none_hw_attempt', 'methods/difference_in_differences.md#fixed-effects',
                'attempted_hw', hw_attempt_fe_none, {'fe_type': 'none', 'outcome_type': 'hw_attempt'}
            ))

        hw_attempt_fe_twoway = run_panel_fe_regression(
            df_hw, 'attempted_hw', entity_id_hw, time_id_hw, 'treat_post',
            additional_vars=None, fe_type='twoway'
        )
        if hw_attempt_fe_twoway and 'error' not in hw_attempt_fe_twoway:
            results.append(create_result_row(
                'did/fe/twoway_hw_attempt', 'methods/difference_in_differences.md#fixed-effects',
                'attempted_hw', hw_attempt_fe_twoway, {'fe_type': 'twoway', 'outcome_type': 'hw_attempt'}
            ))

        # Clustering variations for homework
        hw_attempt_cluster_unit = run_panel_fe_regression(
            df_hw, 'attempted_hw', entity_id_hw, time_id_hw, 'treat_post',
            additional_vars=['post'], cluster_var=entity_id_hw, fe_type='unit'
        )
        if hw_attempt_cluster_unit and 'error' not in hw_attempt_cluster_unit:
            results.append(create_result_row(
                'robust/cluster/unit_hw_attempt', 'robustness/clustering_variations.md#single-level-clustering',
                'attempted_hw', hw_attempt_cluster_unit, {'cluster': 'name_id', 'outcome_type': 'hw_attempt'}
            ))

        hw_attempt_cluster_time = run_panel_fe_regression(
            df_hw, 'attempted_hw', entity_id_hw, time_id_hw, 'treat_post',
            additional_vars=['post'], cluster_var='chapter', fe_type='unit'
        )
        if hw_attempt_cluster_time and 'error' not in hw_attempt_cluster_time:
            results.append(create_result_row(
                'robust/cluster/time_hw_attempt', 'robustness/clustering_variations.md#single-level-clustering',
                'attempted_hw', hw_attempt_cluster_time, {'cluster': 'chapter', 'outcome_type': 'hw_attempt'}
            ))

        # Sample restrictions for homework (by emoji)
        for emoji_val, emoji_name in emoji_names.items():
            df_hw_emoji = df_hw[df_hw['emoji'] == emoji_val].copy()
            if len(df_hw_emoji) >= 50:
                hw_emoji = run_panel_fe_regression(
                    df_hw_emoji, 'attempted_hw', entity_id_hw, time_id_hw, 'treat_post',
                    additional_vars=['post'], fe_type='unit'
                )
                if hw_emoji and 'error' not in hw_emoji:
                    results.append(create_result_row(
                        f'robust/sample/emoji_{emoji_name}_hw_attempt',
                        'robustness/sample_restrictions.md',
                        'attempted_hw', hw_emoji,
                        {'emoji_group': emoji_val, 'outcome_type': 'hw_attempt'}
                    ))

    else:
        print("Could not prepare homework data")

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    results_df = pd.DataFrame(results)
    output_path = OUTPUT_DIR / "specification_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Saved {len(results_df)} specifications to {output_path}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total specifications run: {len(results_df)}")
    print(f"Specifications with errors: {results_df['error'].notna().sum()}")

    # Group by outcome
    print("\nBy outcome variable:")
    print(results_df.groupby('outcome_var').size())

    # Show baseline results
    print("\nBaseline results:")
    baseline_rows = results_df[results_df['spec_id'].str.startswith('baseline')]
    for _, row in baseline_rows.iterrows():
        sig = "***" if row['pval'] < 0.01 else ("**" if row['pval'] < 0.05 else ("*" if row['pval'] < 0.1 else ""))
        print(f"  {row['spec_id']}: coef={row['coef']:.4f}, se={row['se']:.4f}, p={row['pval']:.4f}{sig}")

    return results_df

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("SPECIFICATION SEARCH: 183985-V1")
    print("Emoticons as Performance Feedback")
    print("="*60)
    print(f"pyfixest available: {HAS_PYFIXEST}")
    print(f"linearmodels available: {HAS_LINEARMODELS}")

    results = run_specification_search()
