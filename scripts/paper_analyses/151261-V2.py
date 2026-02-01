#!/usr/bin/env python3
"""
Specification Search: Paper 151261-V2
"The Effects of Working while in School: Evidence from Employment Lotteries"
Le Barbanchon, Ubfal, Araya (AEJ Applied)

Method: Instrumental Variables (2SLS)
- Treatment: Actual program participation (treatment)
- Instrument: Lottery offer (offered)
- Design: RCT with imperfect compliance

This script replicates the survey analysis tables (Tables 5, 7) and runs
systematic specification variations.
"""

import pandas as pd
import numpy as np
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# Set up paths
DATA_PATH = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/151261-V2/replication_AEJ/data/nonconfidential/'
OUTPUT_PATH = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/151261-V2/'

# Paper metadata
PAPER_ID = '151261-V2'
JOURNAL = 'AEJ-Applied'
PAPER_TITLE = 'The Effects of Working while in School: Evidence from Employment Lotteries'

# =============================================================================
# Load Data
# =============================================================================
print("Loading data...")
df = pd.read_stata(DATA_PATH + 'survey_analysis_AEJ.dta')
print(f"Loaded {len(df)} observations")

# =============================================================================
# Define Variables
# =============================================================================

# Treatment and instrument
TREATMENT_VAR = 'treatment'
INSTRUMENT_VAR = 'offered'

# Location dummies (lottery strata)
loc_cols = [f'loc{i}' for i in range(1, 62)]

# Quota controls (stratification variables)
quota_controls = ['trans_quotaloc', 'disability_quotaloc', 'ethnic_afro_quotaloc',
                  'vulnerableall_quotaloc', 'applications']

# Unbalanced covariates (imbalanced at baseline)
unbal_controls = ['school_morning', 'school_afternoon', 'educ_sec_acad', 'educ_terciary']

# Full control set
full_controls = loc_cols + quota_controls + unbal_controls

# Minimal controls (just strata)
minimal_controls = loc_cols + quota_controls

# Main outcomes (Table 5 - Education)
education_outcomes = [
    ('s_highsch_current', 'Currently in High School'),
    ('s_highsch_missedclass', 'Missed Class (conditional on enrollment)'),
    ('s_highsch_whours', 'Hours of School per Week'),
    ('s_TU_dur_study_', 'Time Use: Study Hours'),
    ('s_study_Grade', 'Grades (conditional on enrollment)')
]

# Soft skills outcomes (Table 7)
soft_skills_outcomes = [
    ('s_soft_open', 'Openness'),
    ('s_soft_conscientious', 'Conscientiousness'),
    ('s_soft_extrav', 'Extraversion'),
    ('s_soft_agreeable', 'Agreeableness'),
    ('s_soft_neurotic', 'Neuroticism'),
    ('s_soft_grit', 'Grit')
]

# Work behavior outcomes (Table 7 Panel B)
work_behavior_outcomes = [
    ('s_soft_complydeadline', 'Comply with Deadlines'),
    ('s_soft_adapt', 'Adaptability'),
    ('s_soft_teamwork', 'Teamwork'),
    ('s_soft_punctuality', 'Punctuality'),
    ('s_soft_workindex', 'Work Index'),
    ('s_inpunctuality_interview', 'Interview Punctuality')
]

# Labor market outcomes
labor_outcomes = [
    ('s_emp', 'Employed'),
    ('s_enrolled_edu', 'Enrolled in Education'),
    ('s_wstudy', 'Working and Studying'),
    ('s_nwnstudy', 'Neither Working nor Studying'),
    ('s_tot_income', 'Total Income')
]

# =============================================================================
# IV Estimation Functions
# =============================================================================

def run_2sls(df, outcome, treatment, instrument, controls=None,
             robust_se=True, cluster_var=None, outcome_filter=None):
    """
    Run 2SLS regression manually using numpy/scipy.

    Parameters:
    -----------
    df : DataFrame
    outcome : str - dependent variable
    treatment : str - endogenous variable
    instrument : str - instrumental variable
    controls : list - control variables
    robust_se : bool - use robust standard errors
    cluster_var : str - variable for clustered SEs (optional)
    outcome_filter : str - additional filter (e.g., subsample condition)

    Returns:
    --------
    dict with results
    """
    # Make a copy
    data = df.copy()

    # Apply filter if specified
    if outcome_filter is not None:
        data = data.query(outcome_filter).copy()

    # Get complete cases
    all_vars = [outcome, treatment, instrument]
    if controls:
        all_vars += controls
    if cluster_var:
        all_vars.append(cluster_var)

    # Remove duplicates from all_vars
    all_vars = list(dict.fromkeys(all_vars))

    data = data.dropna(subset=all_vars)

    if len(data) < 50:
        return None

    y = data[outcome].values
    endog = data[treatment].values
    z = data[instrument].values

    # Build control matrix
    if controls:
        # Filter out constant columns
        valid_controls = []
        for c in controls:
            if c in data.columns and data[c].std() > 0:
                valid_controls.append(c)
        X_controls = data[valid_controls].values if valid_controls else np.ones((len(data), 1))
        if len(valid_controls) == 0:
            X_controls = np.ones((len(data), 1))
        else:
            X_controls = np.column_stack([np.ones(len(data)), X_controls])
    else:
        X_controls = np.ones((len(data), 1))

    # First stage: treatment ~ instrument + controls
    X_first = np.column_stack([z, X_controls[:, 1:] if X_controls.shape[1] > 1 else np.array([]).reshape(len(data), 0)])
    X_first = np.column_stack([np.ones(len(data)), X_first])

    try:
        beta_first, _, _, _ = np.linalg.lstsq(X_first, endog, rcond=None)
    except:
        return None

    endog_hat = X_first @ beta_first

    # First stage F-statistic
    resid_first = endog - endog_hat
    ss_res_first = np.sum(resid_first**2)
    ss_tot_first = np.sum((endog - endog.mean())**2)
    r2_first = 1 - ss_res_first / ss_tot_first

    # F-stat for instrument (simplified)
    n = len(data)
    k = X_first.shape[1]
    f_stat = (r2_first / 1) / ((1 - r2_first) / (n - k))

    # Second stage: outcome ~ treatment_hat + controls
    X_second = np.column_stack([endog_hat, X_controls[:, 1:] if X_controls.shape[1] > 1 else np.array([]).reshape(len(data), 0)])
    X_second = np.column_stack([np.ones(len(data)), X_second])

    try:
        beta_second, _, _, _ = np.linalg.lstsq(X_second, y, rcond=None)
    except:
        return None

    # Get residuals using actual treatment (for SE calculation)
    X_actual = np.column_stack([endog, X_controls[:, 1:] if X_controls.shape[1] > 1 else np.array([]).reshape(len(data), 0)])
    X_actual = np.column_stack([np.ones(len(data)), X_actual])
    y_hat = X_actual @ beta_second
    residuals = y - y_hat

    # Robust standard errors
    try:
        XtX_inv = np.linalg.inv(X_first.T @ X_first)
    except:
        return None

    if cluster_var and cluster_var in data.columns:
        # Clustered standard errors
        clusters = data[cluster_var].values
        unique_clusters = np.unique(clusters)
        n_clusters = len(unique_clusters)

        meat = np.zeros((X_first.shape[1], X_first.shape[1]))
        for c in unique_clusters:
            mask = clusters == c
            X_c = X_first[mask]
            r_c = residuals[mask]
            score_c = (X_c.T * r_c).sum(axis=1, keepdims=True)
            meat += score_c @ score_c.T

        # Small sample correction
        correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
        var_matrix = correction * XtX_inv @ meat @ XtX_inv
    else:
        # HC1 robust standard errors
        if robust_se:
            omega = residuals**2
            meat = X_first.T @ (X_first * omega[:, np.newaxis])
            correction = n / (n - k)
            var_matrix = correction * XtX_inv @ meat @ XtX_inv
        else:
            sigma2 = ss_res_first / (n - k)
            var_matrix = sigma2 * XtX_inv

    se = np.sqrt(np.diag(var_matrix))

    # Treatment effect is the coefficient on the instrumented treatment
    coef_treatment = beta_second[1]
    se_treatment = se[1] if len(se) > 1 else se[0]

    # t-stat and p-value
    t_stat = coef_treatment / se_treatment
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - k))

    # CI
    ci_lower = coef_treatment - 1.96 * se_treatment
    ci_upper = coef_treatment + 1.96 * se_treatment

    # R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Control mean
    control_mean = y[endog == 0].mean() if np.sum(endog == 0) > 0 else np.nan

    # Complier control mean
    treated_mean = y[endog == 1].mean() if np.sum(endog == 1) > 0 else np.nan
    ccm = treated_mean - coef_treatment if not np.isnan(treated_mean) else np.nan

    # First stage coefficient
    first_stage_coef = beta_first[1]  # coefficient on instrument

    return {
        'coefficient': coef_treatment,
        'std_error': se_treatment,
        't_stat': t_stat,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_obs': n,
        'r_squared': r2,
        'first_stage_F': f_stat,
        'first_stage_coef': first_stage_coef,
        'control_mean': control_mean,
        'complier_control_mean': ccm,
        'n_clusters': len(unique_clusters) if cluster_var and cluster_var in data.columns else None
    }


def run_ols(df, outcome, treatment, controls=None, robust_se=True, cluster_var=None, outcome_filter=None):
    """
    Run OLS regression (reduced form or comparison).
    """
    data = df.copy()

    if outcome_filter:
        data = data.query(outcome_filter).copy()

    all_vars = [outcome, treatment]
    if controls:
        all_vars += controls
    if cluster_var:
        all_vars.append(cluster_var)

    all_vars = list(dict.fromkeys(all_vars))
    data = data.dropna(subset=all_vars)

    if len(data) < 50:
        return None

    y = data[outcome].values
    x = data[treatment].values

    if controls:
        valid_controls = [c for c in controls if c in data.columns and data[c].std() > 0]
        if valid_controls:
            X = np.column_stack([np.ones(len(data)), x, data[valid_controls].values])
        else:
            X = np.column_stack([np.ones(len(data)), x])
    else:
        X = np.column_stack([np.ones(len(data)), x])

    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except:
        return None

    y_hat = X @ beta
    residuals = y - y_hat

    n = len(data)
    k = X.shape[1]

    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except:
        return None

    if cluster_var and cluster_var in data.columns:
        clusters = data[cluster_var].values
        unique_clusters = np.unique(clusters)
        n_clusters = len(unique_clusters)

        meat = np.zeros((k, k))
        for c in unique_clusters:
            mask = clusters == c
            X_c = X[mask]
            r_c = residuals[mask]
            score_c = (X_c.T * r_c).sum(axis=1, keepdims=True)
            meat += score_c @ score_c.T

        correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
        var_matrix = correction * XtX_inv @ meat @ XtX_inv
    else:
        if robust_se:
            omega = residuals**2
            meat = X.T @ (X * omega[:, np.newaxis])
            correction = n / (n - k)
            var_matrix = correction * XtX_inv @ meat @ XtX_inv
        else:
            sigma2 = np.sum(residuals**2) / (n - k)
            var_matrix = sigma2 * XtX_inv

    se = np.sqrt(np.diag(var_matrix))

    coef = beta[1]
    se_coef = se[1]
    t_stat = coef / se_coef
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - k))

    ci_lower = coef - 1.96 * se_coef
    ci_upper = coef + 1.96 * se_coef

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    control_mean = y[x == 0].mean() if np.sum(x == 0) > 0 else np.nan

    return {
        'coefficient': coef,
        'std_error': se_coef,
        't_stat': t_stat,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_obs': n,
        'r_squared': r2,
        'control_mean': control_mean,
        'n_clusters': len(unique_clusters) if cluster_var and cluster_var in data.columns else None
    }


# =============================================================================
# Run Specification Search
# =============================================================================

results = []

def to_python_float(val):
    """Convert numpy/pandas floats to Python floats for JSON serialization."""
    if val is None:
        return None
    if isinstance(val, (np.floating, np.integer)):
        return float(val)
    if isinstance(val, float) and np.isnan(val):
        return None
    return val


def add_result(spec_id, spec_tree_path, outcome_var, outcome_label, treatment_var,
               result_dict, controls_desc, fixed_effects, cluster_var, model_type,
               sample_desc='Full survey sample'):
    """Helper function to add a result to the results list."""
    if result_dict is None:
        return

    # Build coefficient vector JSON
    coef_vector = {
        'treatment': {
            'var': treatment_var,
            'coef': to_python_float(result_dict['coefficient']),
            'se': to_python_float(result_dict['std_error']),
            'pval': to_python_float(result_dict['p_value'])
        },
        'fixed_effects': fixed_effects.split(' + ') if fixed_effects else [],
        'diagnostics': {
            'first_stage_F': to_python_float(result_dict.get('first_stage_F')),
            'first_stage_coef': to_python_float(result_dict.get('first_stage_coef')),
            'control_mean': to_python_float(result_dict.get('control_mean')),
            'complier_control_mean': to_python_float(result_dict.get('complier_control_mean'))
        }
    }

    results.append({
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'outcome_var': outcome_var,
        'outcome_label': outcome_label,
        'treatment_var': treatment_var,
        'coefficient': to_python_float(result_dict['coefficient']),
        'std_error': to_python_float(result_dict['std_error']),
        't_stat': to_python_float(result_dict['t_stat']),
        'p_value': to_python_float(result_dict['p_value']),
        'ci_lower': to_python_float(result_dict['ci_lower']),
        'ci_upper': to_python_float(result_dict['ci_upper']),
        'n_obs': int(result_dict['n_obs']),
        'r_squared': to_python_float(result_dict.get('r_squared')),
        'coefficient_vector_json': json.dumps(coef_vector),
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })


print("\n" + "="*70)
print("RUNNING SPECIFICATION SEARCH")
print("="*70)

# =============================================================================
# 1. BASELINE SPECIFICATIONS (Table 5 - Education Outcomes)
# =============================================================================
print("\n--- Baseline Specifications (Table 5) ---")

# Main outcome: Currently in High School
for outcome, label in education_outcomes[:1]:  # s_highsch_current
    print(f"  Running baseline for {outcome}...")
    result = run_2sls(df, outcome, TREATMENT_VAR, INSTRUMENT_VAR,
                      controls=full_controls, robust_se=True)
    add_result(
        spec_id='baseline',
        spec_tree_path='methods/instrumental_variables.md#baseline',
        outcome_var=outcome,
        outcome_label=label,
        treatment_var=TREATMENT_VAR,
        result_dict=result,
        controls_desc='Location FE + Quota controls + Unbalanced controls',
        fixed_effects='Location strata',
        cluster_var='None (robust SE)',
        model_type='2SLS'
    )

# Conditional outcomes (need to filter on enrollment)
for outcome, label in education_outcomes[1:]:
    print(f"  Running baseline for {outcome}...")
    filter_cond = 's_highsch_whours == s_highsch_whours' if outcome != 's_highsch_current' else None
    result = run_2sls(df, outcome, TREATMENT_VAR, INSTRUMENT_VAR,
                      controls=full_controls, robust_se=True,
                      outcome_filter=filter_cond if 's_highsch_whours' in df.columns else None)
    add_result(
        spec_id='baseline',
        spec_tree_path='methods/instrumental_variables.md#baseline',
        outcome_var=outcome,
        outcome_label=label,
        treatment_var=TREATMENT_VAR,
        result_dict=result,
        controls_desc='Location FE + Quota controls + Unbalanced controls',
        fixed_effects='Location strata',
        cluster_var='None (robust SE)',
        model_type='2SLS',
        sample_desc='Enrolled students' if 'missedclass' in outcome or 'whours' in outcome or 'Grade' in outcome else 'Full survey sample'
    )

# =============================================================================
# 2. BASELINE SPECIFICATIONS (Table 7 - Soft Skills)
# =============================================================================
print("\n--- Baseline Specifications (Table 7) ---")

for outcome, label in soft_skills_outcomes + work_behavior_outcomes:
    print(f"  Running baseline for {outcome}...")
    result = run_2sls(df, outcome, TREATMENT_VAR, INSTRUMENT_VAR,
                      controls=full_controls, cluster_var='ci_an')
    add_result(
        spec_id='baseline',
        spec_tree_path='methods/instrumental_variables.md#baseline',
        outcome_var=outcome,
        outcome_label=label,
        treatment_var=TREATMENT_VAR,
        result_dict=result,
        controls_desc='Location FE + Quota controls + Unbalanced controls',
        fixed_effects='Location strata',
        cluster_var='ci_an (applicant ID)',
        model_type='2SLS'
    )

# =============================================================================
# 3. LABOR MARKET OUTCOMES (Appendix Table E2)
# =============================================================================
print("\n--- Labor Market Outcomes ---")

for outcome, label in labor_outcomes:
    print(f"  Running baseline for {outcome}...")
    result = run_2sls(df, outcome, TREATMENT_VAR, INSTRUMENT_VAR,
                      controls=full_controls, cluster_var='ci_an')
    add_result(
        spec_id='baseline',
        spec_tree_path='methods/instrumental_variables.md#baseline',
        outcome_var=outcome,
        outcome_label=label,
        treatment_var=TREATMENT_VAR,
        result_dict=result,
        controls_desc='Location FE + Quota controls + Unbalanced controls',
        fixed_effects='Location strata',
        cluster_var='ci_an (applicant ID)',
        model_type='2SLS'
    )

# =============================================================================
# 4. IV METHOD VARIATIONS
# =============================================================================
print("\n--- IV Method Variations ---")

# Primary outcome for method variations: s_highsch_current
main_outcome = 's_highsch_current'
main_label = 'Currently in High School'

# 4a. OLS (ignoring endogeneity)
print("  Running OLS (no IV)...")
result = run_ols(df, main_outcome, TREATMENT_VAR, controls=full_controls, robust_se=True)
add_result(
    spec_id='iv/method/ols',
    spec_tree_path='methods/instrumental_variables.md#estimation-method',
    outcome_var=main_outcome,
    outcome_label=main_label,
    treatment_var=TREATMENT_VAR,
    result_dict=result,
    controls_desc='Location FE + Quota controls + Unbalanced controls',
    fixed_effects='Location strata',
    cluster_var='None (robust SE)',
    model_type='OLS'
)

# 4b. Reduced form (ITT effect)
print("  Running reduced form (ITT)...")
result = run_ols(df, main_outcome, INSTRUMENT_VAR, controls=full_controls, robust_se=True)
add_result(
    spec_id='iv/first_stage/reduced_form',
    spec_tree_path='methods/instrumental_variables.md#first-stage',
    outcome_var=main_outcome,
    outcome_label=main_label,
    treatment_var=INSTRUMENT_VAR,
    result_dict=result,
    controls_desc='Location FE + Quota controls + Unbalanced controls',
    fixed_effects='Location strata',
    cluster_var='None (robust SE)',
    model_type='OLS (reduced form/ITT)'
)

# 4c. First stage
print("  Running first stage...")
result = run_ols(df, TREATMENT_VAR, INSTRUMENT_VAR, controls=full_controls, robust_se=True)
add_result(
    spec_id='iv/first_stage/baseline',
    spec_tree_path='methods/instrumental_variables.md#first-stage',
    outcome_var=TREATMENT_VAR,
    outcome_label='Program Participation',
    treatment_var=INSTRUMENT_VAR,
    result_dict=result,
    controls_desc='Location FE + Quota controls + Unbalanced controls',
    fixed_effects='Location strata',
    cluster_var='None (robust SE)',
    model_type='OLS (first stage)'
)

# =============================================================================
# 5. CONTROL SET VARIATIONS
# =============================================================================
print("\n--- Control Set Variations ---")

# 5a. No controls
print("  Running with no controls...")
result = run_2sls(df, main_outcome, TREATMENT_VAR, INSTRUMENT_VAR, controls=None, robust_se=True)
add_result(
    spec_id='iv/controls/none',
    spec_tree_path='methods/instrumental_variables.md#control-sets',
    outcome_var=main_outcome,
    outcome_label=main_label,
    treatment_var=TREATMENT_VAR,
    result_dict=result,
    controls_desc='None',
    fixed_effects='None',
    cluster_var='None (robust SE)',
    model_type='2SLS'
)

# 5b. Minimal controls (strata only)
print("  Running with minimal controls (strata only)...")
result = run_2sls(df, main_outcome, TREATMENT_VAR, INSTRUMENT_VAR, controls=minimal_controls, robust_se=True)
add_result(
    spec_id='iv/controls/minimal',
    spec_tree_path='methods/instrumental_variables.md#control-sets',
    outcome_var=main_outcome,
    outcome_label=main_label,
    treatment_var=TREATMENT_VAR,
    result_dict=result,
    controls_desc='Location FE + Quota controls only',
    fixed_effects='Location strata',
    cluster_var='None (robust SE)',
    model_type='2SLS'
)

# 5c. With baseline GPA control (Table A9 specification)
print("  Running with GPA control...")
controls_with_gpa = full_controls + ['s_study_GPA']
result = run_2sls(df, main_outcome, TREATMENT_VAR, INSTRUMENT_VAR, controls=controls_with_gpa, robust_se=True)
add_result(
    spec_id='iv/controls/with_gpa',
    spec_tree_path='methods/instrumental_variables.md#control-sets',
    outcome_var=main_outcome,
    outcome_label=main_label,
    treatment_var=TREATMENT_VAR,
    result_dict=result,
    controls_desc='Location FE + Quota controls + Unbalanced + GPA',
    fixed_effects='Location strata',
    cluster_var='None (robust SE)',
    model_type='2SLS'
)

# =============================================================================
# 6. CLUSTERING VARIATIONS
# =============================================================================
print("\n--- Clustering Variations ---")

# 6a. No clustering (robust SE only)
result = run_2sls(df, main_outcome, TREATMENT_VAR, INSTRUMENT_VAR, controls=full_controls,
                  robust_se=True, cluster_var=None)
add_result(
    spec_id='robust/cluster/none',
    spec_tree_path='robustness/clustering_variations.md#single-level-clustering',
    outcome_var=main_outcome,
    outcome_label=main_label,
    treatment_var=TREATMENT_VAR,
    result_dict=result,
    controls_desc='Location FE + Quota controls + Unbalanced controls',
    fixed_effects='Location strata',
    cluster_var='None (robust SE)',
    model_type='2SLS'
)

# 6b. Cluster by applicant (ci_an)
print("  Running with clustering at applicant level...")
result = run_2sls(df, main_outcome, TREATMENT_VAR, INSTRUMENT_VAR, controls=full_controls,
                  cluster_var='ci_an')
add_result(
    spec_id='robust/cluster/applicant',
    spec_tree_path='robustness/clustering_variations.md#single-level-clustering',
    outcome_var=main_outcome,
    outcome_label=main_label,
    treatment_var=TREATMENT_VAR,
    result_dict=result,
    controls_desc='Location FE + Quota controls + Unbalanced controls',
    fixed_effects='Location strata',
    cluster_var='ci_an (applicant ID)',
    model_type='2SLS'
)

# =============================================================================
# 7. LEAVE-ONE-OUT ROBUSTNESS
# =============================================================================
print("\n--- Leave-One-Out Robustness ---")

# Drop each unbalanced control one at a time
for drop_var in unbal_controls:
    remaining = [c for c in full_controls if c != drop_var]
    print(f"  Dropping {drop_var}...")
    result = run_2sls(df, main_outcome, TREATMENT_VAR, INSTRUMENT_VAR,
                      controls=remaining, robust_se=True)
    add_result(
        spec_id=f'robust/loo/drop_{drop_var}',
        spec_tree_path='robustness/leave_one_out.md',
        outcome_var=main_outcome,
        outcome_label=main_label,
        treatment_var=TREATMENT_VAR,
        result_dict=result,
        controls_desc=f'Full controls minus {drop_var}',
        fixed_effects='Location strata',
        cluster_var='None (robust SE)',
        model_type='2SLS'
    )

# Drop quota controls one at a time
for drop_var in quota_controls:
    remaining = [c for c in full_controls if c != drop_var]
    print(f"  Dropping {drop_var}...")
    result = run_2sls(df, main_outcome, TREATMENT_VAR, INSTRUMENT_VAR,
                      controls=remaining, robust_se=True)
    add_result(
        spec_id=f'robust/loo/drop_{drop_var}',
        spec_tree_path='robustness/leave_one_out.md',
        outcome_var=main_outcome,
        outcome_label=main_label,
        treatment_var=TREATMENT_VAR,
        result_dict=result,
        controls_desc=f'Full controls minus {drop_var}',
        fixed_effects='Location strata',
        cluster_var='None (robust SE)',
        model_type='2SLS'
    )

# =============================================================================
# 8. SINGLE COVARIATE ANALYSIS
# =============================================================================
print("\n--- Single Covariate Analysis ---")

# Bivariate (no controls)
result = run_2sls(df, main_outcome, TREATMENT_VAR, INSTRUMENT_VAR, controls=None, robust_se=True)
add_result(
    spec_id='robust/single/none',
    spec_tree_path='robustness/single_covariate.md',
    outcome_var=main_outcome,
    outcome_label=main_label,
    treatment_var=TREATMENT_VAR,
    result_dict=result,
    controls_desc='None (bivariate)',
    fixed_effects='None',
    cluster_var='None (robust SE)',
    model_type='2SLS'
)

# Add single controls
single_controls = unbal_controls + ['applications']
for control in single_controls:
    print(f"  Adding only {control}...")
    result = run_2sls(df, main_outcome, TREATMENT_VAR, INSTRUMENT_VAR,
                      controls=[control], robust_se=True)
    add_result(
        spec_id=f'robust/single/{control}',
        spec_tree_path='robustness/single_covariate.md',
        outcome_var=main_outcome,
        outcome_label=main_label,
        treatment_var=TREATMENT_VAR,
        result_dict=result,
        controls_desc=f'Only {control}',
        fixed_effects='None',
        cluster_var='None (robust SE)',
        model_type='2SLS'
    )

# =============================================================================
# 9. SAMPLE RESTRICTIONS
# =============================================================================
print("\n--- Sample Restrictions ---")

# By gender
print("  Running for females only...")
df_female = df[df['female'] == 1].copy()
result = run_2sls(df_female, main_outcome, TREATMENT_VAR, INSTRUMENT_VAR,
                  controls=full_controls, robust_se=True)
add_result(
    spec_id='robust/sample/female',
    spec_tree_path='robustness/sample_restrictions.md',
    outcome_var=main_outcome,
    outcome_label=main_label,
    treatment_var=TREATMENT_VAR,
    result_dict=result,
    controls_desc='Location FE + Quota controls + Unbalanced controls',
    fixed_effects='Location strata',
    cluster_var='None (robust SE)',
    model_type='2SLS',
    sample_desc='Female only'
)

print("  Running for males only...")
df_male = df[df['female'] == 0].copy()
result = run_2sls(df_male, main_outcome, TREATMENT_VAR, INSTRUMENT_VAR,
                  controls=full_controls, robust_se=True)
add_result(
    spec_id='robust/sample/male',
    spec_tree_path='robustness/sample_restrictions.md',
    outcome_var=main_outcome,
    outcome_label=main_label,
    treatment_var=TREATMENT_VAR,
    result_dict=result,
    controls_desc='Location FE + Quota controls + Unbalanced controls',
    fixed_effects='Location strata',
    cluster_var='None (robust SE)',
    model_type='2SLS',
    sample_desc='Male only'
)

# By ability (using the ability indicators in the data)
if 'low_ability' in df.columns:
    print("  Running for low ability only...")
    df_low = df[df['low_ability'] == 1].copy()
    result = run_2sls(df_low, main_outcome, TREATMENT_VAR, INSTRUMENT_VAR,
                      controls=full_controls, robust_se=True)
    add_result(
        spec_id='robust/sample/low_ability',
        spec_tree_path='robustness/sample_restrictions.md',
        outcome_var=main_outcome,
        outcome_label=main_label,
        treatment_var=TREATMENT_VAR,
        result_dict=result,
        controls_desc='Location FE + Quota controls + Unbalanced controls',
        fixed_effects='Location strata',
        cluster_var='None (robust SE)',
        model_type='2SLS',
        sample_desc='Low ability students'
    )

if 'high_ability' in df.columns:
    print("  Running for high ability only...")
    df_high = df[df['high_ability'] == 1].copy()
    result = run_2sls(df_high, main_outcome, TREATMENT_VAR, INSTRUMENT_VAR,
                      controls=full_controls, robust_se=True)
    add_result(
        spec_id='robust/sample/high_ability',
        spec_tree_path='robustness/sample_restrictions.md',
        outcome_var=main_outcome,
        outcome_label=main_label,
        treatment_var=TREATMENT_VAR,
        result_dict=result,
        controls_desc='Location FE + Quota controls + Unbalanced controls',
        fixed_effects='Location strata',
        cluster_var='None (robust SE)',
        model_type='2SLS',
        sample_desc='High ability students'
    )

# =============================================================================
# 10. ADDITIONAL OUTCOMES - FULL SPECIFICATION SEARCH
# =============================================================================
print("\n--- Additional Outcomes with Control Variations ---")

# Run the full set of variations for key outcomes
key_outcomes = [
    ('s_emp', 'Employed'),
    ('s_enrolled_edu', 'Enrolled in Education'),
    ('s_soft_grit', 'Grit'),
    ('s_TU_dur_study_', 'Study Time')
]

for outcome, label in key_outcomes:
    print(f"\n  Processing {outcome}...")

    # No controls
    result = run_2sls(df, outcome, TREATMENT_VAR, INSTRUMENT_VAR, controls=None, robust_se=True)
    add_result(
        spec_id='iv/controls/none',
        spec_tree_path='methods/instrumental_variables.md#control-sets',
        outcome_var=outcome,
        outcome_label=label,
        treatment_var=TREATMENT_VAR,
        result_dict=result,
        controls_desc='None',
        fixed_effects='None',
        cluster_var='None (robust SE)',
        model_type='2SLS'
    )

    # Minimal controls
    result = run_2sls(df, outcome, TREATMENT_VAR, INSTRUMENT_VAR, controls=minimal_controls, robust_se=True)
    add_result(
        spec_id='iv/controls/minimal',
        spec_tree_path='methods/instrumental_variables.md#control-sets',
        outcome_var=outcome,
        outcome_label=label,
        treatment_var=TREATMENT_VAR,
        result_dict=result,
        controls_desc='Location FE + Quota controls only',
        fixed_effects='Location strata',
        cluster_var='None (robust SE)',
        model_type='2SLS'
    )

    # Full controls
    result = run_2sls(df, outcome, TREATMENT_VAR, INSTRUMENT_VAR, controls=full_controls, robust_se=True)
    add_result(
        spec_id='iv/controls/full',
        spec_tree_path='methods/instrumental_variables.md#control-sets',
        outcome_var=outcome,
        outcome_label=label,
        treatment_var=TREATMENT_VAR,
        result_dict=result,
        controls_desc='Location FE + Quota controls + Unbalanced controls',
        fixed_effects='Location strata',
        cluster_var='None (robust SE)',
        model_type='2SLS'
    )

    # ITT
    result = run_ols(df, outcome, INSTRUMENT_VAR, controls=full_controls, robust_se=True)
    add_result(
        spec_id='iv/first_stage/reduced_form',
        spec_tree_path='methods/instrumental_variables.md#first-stage',
        outcome_var=outcome,
        outcome_label=label,
        treatment_var=INSTRUMENT_VAR,
        result_dict=result,
        controls_desc='Location FE + Quota controls + Unbalanced controls',
        fixed_effects='Location strata',
        cluster_var='None (robust SE)',
        model_type='OLS (reduced form/ITT)'
    )

# =============================================================================
# SAVE RESULTS
# =============================================================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Convert to DataFrame
results_df = pd.DataFrame(results)
print(f"\nTotal specifications run: {len(results_df)}")

# Save to CSV
output_file = OUTPUT_PATH + 'specification_results.csv'
results_df.to_csv(output_file, index=False)
print(f"Results saved to: {output_file}")

# Print summary statistics
print("\n--- SUMMARY STATISTICS ---")
print(f"Total specifications: {len(results_df)}")
print(f"Unique outcomes: {results_df['outcome_var'].nunique()}")
print(f"Unique spec types: {results_df['spec_id'].nunique()}")

sig_05 = (results_df['p_value'] < 0.05).sum()
sig_01 = (results_df['p_value'] < 0.01).sum()
print(f"\nSignificant at 5%: {sig_05} ({100*sig_05/len(results_df):.1f}%)")
print(f"Significant at 1%: {sig_01} ({100*sig_01/len(results_df):.1f}%)")

# Summary by outcome
print("\n--- BY OUTCOME ---")
for outcome in results_df['outcome_var'].unique():
    subset = results_df[results_df['outcome_var'] == outcome]
    baseline = subset[subset['spec_id'] == 'baseline']
    if len(baseline) > 0:
        b = baseline.iloc[0]
        print(f"\n{outcome}:")
        print(f"  Baseline: coef={b['coefficient']:.4f}, se={b['std_error']:.4f}, p={b['p_value']:.4f}")
        print(f"  N specs: {len(subset)}")
        print(f"  Coef range: [{subset['coefficient'].min():.4f}, {subset['coefficient'].max():.4f}]")

print("\n" + "="*70)
print("SPECIFICATION SEARCH COMPLETE")
print("="*70)
