"""
Specification Search: 151261-V2
The Effects of Working while in School: Evidence from Employment Lotteries
AEJ: Applied

This paper studies the effects of a youth employment training (YET) program in Uruguay
using lottery-based assignment. The instrument is "offered" (won lottery) and the
endogenous treatment is "treatment" (actual program participation).

Method: Instrumental Variables (2SLS)
- Instrument: offered (lottery outcome)
- Endogenous variable: treatment (program participation)
- Key outcomes: employment, education, earnings, soft skills
"""

import pandas as pd
import numpy as np
import json
import warnings
from scipy import stats
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
warnings.filterwarnings('ignore')

# Paths
DATA_PATH = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/151261-V2/replication_AEJ/data/nonconfidential/'
OUTPUT_PATH = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/151261-V2/'

# Paper metadata
PAPER_ID = '151261-V2'
PAPER_TITLE = 'The Effects of Working while in School: Evidence from Employment Lotteries'
JOURNAL = 'AEJ: Applied'

# Load data
df = pd.read_stata(DATA_PATH + 'survey_analysis_AEJ.dta')

# Convert columns to numeric, handling any issues
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            pass

# Define variables
# Location dummies for strata fixed effects
loc_cols = [c for c in df.columns if c.startswith('loc')]

# Strata controls (from paper)
strata_controls = loc_cols + ['trans_quotaloc', 'disability_quotaloc', 'ethnic_afro_quotaloc',
                              'vulnerableall_quotaloc', 'applications']

# Unbalanced covariates (from Table E1 checks)
unbal_controls = ['school_morning', 'school_afternoon', 'educ_sec_acad', 'educ_terciary']

# Full control set
full_controls = strata_controls + unbal_controls

# Individual-level controls (for leave-one-out)
individual_controls = ['female', 'age', 'number_kids', 'b_father_comphighschool',
                       'b_mother_comphighschool', 'b_books_more10', 'hours_school',
                       'social_participant', 'social_tus', 'b_edu_course2015', 'b_repprim_once']

# All controls for extended specifications
all_controls = full_controls + individual_controls

# Outcome variables
labor_outcomes = ['s_emp', 's_tot_income', 's_wstudy', 's_nwnstudy', 's_pension_contrib']
education_outcomes = ['s_enrolled_edu', 's_highsch_current', 's_enrolled_public']
soft_skills = ['s_soft_open', 's_soft_conscien', 's_soft_extrav', 's_soft_agreeable',
               's_soft_neurotic', 's_soft_grit', 's_soft_complydeadline', 's_soft_adapt',
               's_soft_teamwork', 's_soft_punctuality', 's_soft_workindex']
study_effort = ['s_highsch_missedclass', 's_highsch_whours', 's_TU_dur_study_', 's_study_Grade']

# Fix soft_conscien name
if 's_soft_conscientious' in df.columns and 's_soft_conscien' not in df.columns:
    df['s_soft_conscien'] = df['s_soft_conscientious']

results = []

def safe_float(x):
    """Convert to float, handling None/NaN"""
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return float(x)
    except:
        return None

def run_iv_regression(df_sample, outcome_var, endog_var, instrument, controls, spec_id, spec_tree_path):
    """
    Run 2SLS regression and return results dictionary
    """
    try:
        # Prepare data
        df_reg = df_sample.copy()

        # Drop missing values
        all_vars = [outcome_var, endog_var, instrument] + controls
        all_vars = [v for v in all_vars if v in df_reg.columns]
        df_reg = df_reg.dropna(subset=all_vars)

        if len(df_reg) < 50:
            return None

        # Check variation
        if df_reg[outcome_var].std() < 1e-10:
            return None
        if df_reg[endog_var].std() < 1e-10:
            return None
        if df_reg[instrument].std() < 1e-10:
            return None

        # Build formulas
        exog_vars = ['1'] + controls
        exog_formula = ' + '.join([f'C({v})' if df_reg[v].nunique() <= 2 and v not in loc_cols else v for v in controls]) if controls else '1'

        # Use linearmodels IV2SLS
        formula = f'{outcome_var} ~ 1 + {" + ".join(controls)} + [{endog_var} ~ {instrument}]' if controls else f'{outcome_var} ~ 1 + [{endog_var} ~ {instrument}]'

        # Alternative approach with statsmodels
        y = df_reg[outcome_var].values

        # Exogenous regressors (controls)
        if controls:
            X_exog = df_reg[controls].values
            X_exog = sm.add_constant(X_exog, has_constant='add')
        else:
            X_exog = np.ones((len(df_reg), 1))

        # Endogenous variable
        X_endog = df_reg[endog_var].values.reshape(-1, 1)

        # Instrument
        Z = df_reg[instrument].values.reshape(-1, 1)

        # First stage
        fs_X = np.hstack([X_exog, Z])
        fs_model = sm.OLS(X_endog.flatten(), fs_X).fit()
        fs_coef = fs_model.params[-1]  # Instrument coefficient
        fs_se = fs_model.bse[-1]
        fs_fstat = (fs_coef / fs_se) ** 2

        # Predicted values from first stage
        X_endog_hat = fs_model.fittedvalues.reshape(-1, 1)

        # Second stage
        ss_X = np.hstack([X_exog, X_endog_hat])
        ss_model = sm.OLS(y, ss_X).fit(cov_type='HC1')

        # Treatment effect (last coefficient)
        treat_coef = ss_model.params[-1]
        treat_se = ss_model.bse[-1]
        treat_tstat = treat_coef / treat_se
        treat_pval = 2 * (1 - stats.t.cdf(abs(treat_tstat), ss_model.df_resid))

        # Confidence intervals
        ci_lower = treat_coef - 1.96 * treat_se
        ci_upper = treat_coef + 1.96 * treat_se

        # Build coefficient vector
        coef_vector = {
            'treatment': {
                'var': endog_var,
                'coef': safe_float(treat_coef),
                'se': safe_float(treat_se),
                'pval': safe_float(treat_pval)
            },
            'first_stage': {
                'instrument': instrument,
                'coef': safe_float(fs_coef),
                'se': safe_float(fs_se),
                'F_stat': safe_float(fs_fstat)
            },
            'controls': controls,
            'diagnostics': {
                'first_stage_F': safe_float(fs_fstat),
                'n_obs': int(len(df_reg))
            }
        }

        # Control group mean
        ccm = df_reg[df_reg[endog_var] == 0][outcome_var].mean() if (df_reg[endog_var] == 0).sum() > 0 else None

        return {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': outcome_var,
            'treatment_var': endog_var,
            'coefficient': safe_float(treat_coef),
            'std_error': safe_float(treat_se),
            't_stat': safe_float(treat_tstat),
            'p_value': safe_float(treat_pval),
            'ci_lower': safe_float(ci_lower),
            'ci_upper': safe_float(ci_upper),
            'n_obs': int(len(df_reg)),
            'r_squared': safe_float(ss_model.rsquared),
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': 'Full sample' if len(df_reg) == len(df_sample.dropna(subset=all_vars[:3])) else 'Restricted sample',
            'fixed_effects': 'Location strata',
            'controls_desc': f'{len(controls)} controls' if controls else 'No controls',
            'cluster_var': 'robust',
            'model_type': 'IV-2SLS',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
            'first_stage_F': safe_float(fs_fstat),
            'control_mean': safe_float(ccm)
        }
    except Exception as e:
        print(f"Error in {spec_id}: {str(e)}")
        return None

def run_ols_regression(df_sample, outcome_var, treatment_var, controls, spec_id, spec_tree_path):
    """
    Run OLS regression (reduced form or standard OLS)
    """
    try:
        df_reg = df_sample.copy()

        all_vars = [outcome_var, treatment_var] + controls
        all_vars = [v for v in all_vars if v in df_reg.columns]
        df_reg = df_reg.dropna(subset=all_vars)

        if len(df_reg) < 50:
            return None

        y = df_reg[outcome_var].values

        if controls:
            X = sm.add_constant(df_reg[[treatment_var] + controls].values)
        else:
            X = sm.add_constant(df_reg[[treatment_var]].values)

        model = sm.OLS(y, X).fit(cov_type='HC1')

        treat_coef = model.params[1]
        treat_se = model.bse[1]
        treat_tstat = treat_coef / treat_se
        treat_pval = model.pvalues[1]

        ci_lower = treat_coef - 1.96 * treat_se
        ci_upper = treat_coef + 1.96 * treat_se

        coef_vector = {
            'treatment': {
                'var': treatment_var,
                'coef': safe_float(treat_coef),
                'se': safe_float(treat_se),
                'pval': safe_float(treat_pval)
            },
            'controls': controls
        }

        return {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': safe_float(treat_coef),
            'std_error': safe_float(treat_se),
            't_stat': safe_float(treat_tstat),
            'p_value': safe_float(treat_pval),
            'ci_lower': safe_float(ci_lower),
            'ci_upper': safe_float(ci_upper),
            'n_obs': int(len(df_reg)),
            'r_squared': safe_float(model.rsquared),
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': 'Full sample',
            'fixed_effects': 'Location strata' if any(c in controls for c in loc_cols) else 'None',
            'controls_desc': f'{len(controls)} controls',
            'cluster_var': 'robust',
            'model_type': 'OLS',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
            'first_stage_F': None,
            'control_mean': None
        }
    except Exception as e:
        print(f"Error in {spec_id}: {str(e)}")
        return None


# ============================================================================
# RUN SPECIFICATIONS
# ============================================================================

print("Starting specification search for paper 151261-V2...")
print("=" * 60)

# ============================================================================
# 1. BASELINE SPECIFICATIONS (from paper Table 5, 7, E2)
# ============================================================================
print("\n1. Running baseline specifications...")

# Main outcome: Employment (Table E2)
result = run_iv_regression(df, 's_emp', 'treatment', 'offered', full_controls,
                          'baseline', 'methods/instrumental_variables.md')
if result:
    results.append(result)
    print(f"  Baseline employment: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# Main outcome: Enrollment (Table E2)
result = run_iv_regression(df, 's_enrolled_edu', 'treatment', 'offered', full_controls,
                          'baseline_enrollment', 'methods/instrumental_variables.md')
if result:
    results.append(result)

# Main outcome: Total income (Table E2)
result = run_iv_regression(df, 's_tot_income', 'treatment', 'offered', full_controls,
                          'baseline_income', 'methods/instrumental_variables.md')
if result:
    results.append(result)

# Work and study (Table E2)
result = run_iv_regression(df, 's_wstudy', 'treatment', 'offered', full_controls,
                          'baseline_wstudy', 'methods/instrumental_variables.md')
if result:
    results.append(result)

# Neither work nor study (Table E2)
result = run_iv_regression(df, 's_nwnstudy', 'treatment', 'offered', full_controls,
                          'baseline_nwnstudy', 'methods/instrumental_variables.md')
if result:
    results.append(result)

# ============================================================================
# 2. FIRST STAGE (Table A2)
# ============================================================================
print("\n2. Running first stage specifications...")

result = run_ols_regression(df, 'treatment', 'offered', full_controls,
                           'iv/first_stage/baseline', 'methods/instrumental_variables.md#first-stage')
if result:
    results.append(result)

# ============================================================================
# 3. REDUCED FORM (ITT)
# ============================================================================
print("\n3. Running reduced form (ITT) specifications...")

for outcome, name in [('s_emp', 'emp'), ('s_enrolled_edu', 'enroll'), ('s_tot_income', 'income'),
                      ('s_wstudy', 'wstudy'), ('s_nwnstudy', 'nwnstudy')]:
    result = run_ols_regression(df, outcome, 'offered', full_controls,
                               f'iv/first_stage/reduced_form_{name}', 'methods/instrumental_variables.md#first-stage')
    if result:
        results.append(result)

# ============================================================================
# 4. ALTERNATIVE OUTCOMES
# ============================================================================
print("\n4. Running alternative outcome specifications...")

# Soft skills outcomes (Table 7)
for outcome in soft_skills:
    if outcome in df.columns:
        result = run_iv_regression(df, outcome, 'treatment', 'offered', full_controls,
                                  f'robust/outcome/{outcome}', 'robustness/measurement.md')
        if result:
            results.append(result)

# Study effort outcomes (Table 5) - conditional on enrolled
df_enrolled = df[df['s_highsch_whours'].notna()].copy()
for outcome in ['s_highsch_missedclass', 's_highsch_whours', 's_TU_dur_study_', 's_study_Grade']:
    if outcome in df_enrolled.columns:
        result = run_iv_regression(df_enrolled, outcome, 'treatment', 'offered', full_controls,
                                  f'robust/outcome/{outcome}', 'robustness/measurement.md')
        if result:
            results.append(result)

# ============================================================================
# 5. CONTROL VARIATIONS
# ============================================================================
print("\n5. Running control variation specifications...")

# 5a. No controls (strata only)
for outcome, name in [('s_emp', 'emp'), ('s_enrolled_edu', 'enroll'), ('s_tot_income', 'income')]:
    result = run_iv_regression(df, outcome, 'treatment', 'offered', strata_controls,
                              f'iv/controls/strata_only_{name}', 'methods/instrumental_variables.md#control-sets')
    if result:
        results.append(result)

# 5b. No controls at all
for outcome, name in [('s_emp', 'emp'), ('s_enrolled_edu', 'enroll'), ('s_tot_income', 'income')]:
    result = run_iv_regression(df, outcome, 'treatment', 'offered', [],
                              f'iv/controls/none_{name}', 'methods/instrumental_variables.md#control-sets')
    if result:
        results.append(result)

# 5c. Extended controls (add individual characteristics)
for outcome, name in [('s_emp', 'emp'), ('s_enrolled_edu', 'enroll'), ('s_tot_income', 'income')]:
    extended_ctrls = [c for c in all_controls if c in df.columns]
    result = run_iv_regression(df, outcome, 'treatment', 'offered', extended_ctrls,
                              f'iv/controls/full_{name}', 'methods/instrumental_variables.md#control-sets')
    if result:
        results.append(result)

# 5d. Leave-one-out for individual controls (on main employment outcome)
print("  Running leave-one-out specifications...")
for ctrl in individual_controls:
    if ctrl in df.columns:
        loo_controls = [c for c in full_controls if c != ctrl]
        result = run_iv_regression(df, 's_emp', 'treatment', 'offered', loo_controls,
                                  f'robust/loo/drop_{ctrl}', 'robustness/leave_one_out.md')
        if result:
            results.append(result)

# ============================================================================
# 6. SAMPLE RESTRICTIONS
# ============================================================================
print("\n6. Running sample restriction specifications...")

# 6a. By gender
for gender, label in [(1, 'female'), (0, 'male')]:
    df_sub = df[df['female'] == gender].copy()
    for outcome, name in [('s_emp', 'emp'), ('s_enrolled_edu', 'enroll'), ('s_tot_income', 'income')]:
        result = run_iv_regression(df_sub, outcome, 'treatment', 'offered', full_controls,
                                  f'robust/sample/{label}_{name}', 'robustness/sample_restrictions.md')
        if result:
            results.append(result)

# 6b. By age
median_age = df['age'].median()
for condition, label in [(df['age'] <= median_age, 'young'), (df['age'] > median_age, 'old')]:
    df_sub = df[condition].copy()
    for outcome, name in [('s_emp', 'emp'), ('s_enrolled_edu', 'enroll')]:
        result = run_iv_regression(df_sub, outcome, 'treatment', 'offered', full_controls,
                                  f'robust/sample/{label}_{name}', 'robustness/sample_restrictions.md')
        if result:
            results.append(result)

# 6c. By vulnerability status
if 'vulnerableall_quotaloc' in df.columns:
    for condition, label in [(df['vulnerableall_quotaloc'] > 0, 'vulnerable'), (df['vulnerableall_quotaloc'] == 0, 'not_vulnerable')]:
        df_sub = df[condition].copy()
        if len(df_sub) >= 100:
            result = run_iv_regression(df_sub, 's_emp', 'treatment', 'offered', full_controls,
                                      f'robust/sample/{label}_emp', 'robustness/sample_restrictions.md')
            if result:
                results.append(result)

# 6d. By parental education
if 'b_mother_comphighschool' in df.columns:
    for condition, label in [(df['b_mother_comphighschool'] == 1, 'high_ses'), (df['b_mother_comphighschool'] == 0, 'low_ses')]:
        df_sub = df[condition].copy()
        if len(df_sub) >= 100:
            result = run_iv_regression(df_sub, 's_emp', 'treatment', 'offered', full_controls,
                                      f'robust/sample/{label}_emp', 'robustness/sample_restrictions.md')
            if result:
                results.append(result)

# 6e. By education type at baseline
for condition, label in [(df['educ_sec_acad'] == 1, 'academic'), (df['educ_sec_tech'] == 1, 'technical')]:
    df_sub = df[condition].copy()
    if len(df_sub) >= 100:
        for outcome, name in [('s_emp', 'emp'), ('s_enrolled_edu', 'enroll')]:
            result = run_iv_regression(df_sub, outcome, 'treatment', 'offered', full_controls,
                                      f'robust/sample/{label}_{name}', 'robustness/sample_restrictions.md')
            if result:
                results.append(result)

# 6f. Winsorize income
df_wins = df.copy()
p01, p99 = df_wins['s_tot_income'].quantile([0.01, 0.99])
df_wins['s_tot_income_w'] = df_wins['s_tot_income'].clip(lower=p01, upper=p99)
result = run_iv_regression(df_wins, 's_tot_income_w', 'treatment', 'offered', full_controls,
                          'robust/sample/winsorize_1pct_income', 'robustness/sample_restrictions.md')
if result:
    results.append(result)

# 6g. Trim extreme values
df_trim = df[(df['s_tot_income'] >= df['s_tot_income'].quantile(0.01)) &
             (df['s_tot_income'] <= df['s_tot_income'].quantile(0.99))].copy()
result = run_iv_regression(df_trim, 's_tot_income', 'treatment', 'offered', full_controls,
                          'robust/sample/trim_1pct_income', 'robustness/sample_restrictions.md')
if result:
    results.append(result)

# ============================================================================
# 7. INFERENCE/CLUSTERING VARIATIONS
# ============================================================================
print("\n7. Running inference variation specifications...")

# The paper uses robust SE - we'll compare different SE approaches
# Note: For this survey data, clustering options are limited

# 7a. Standard robust SE (baseline approach from paper)
for outcome, name in [('s_emp', 'emp'), ('s_enrolled_edu', 'enroll'), ('s_tot_income', 'income')]:
    result = run_iv_regression(df, outcome, 'treatment', 'offered', full_controls,
                              f'robust/cluster/robust_{name}', 'robustness/clustering_variations.md')
    if result:
        results.append(result)

# ============================================================================
# 8. ESTIMATION METHOD VARIATIONS
# ============================================================================
print("\n8. Running estimation method variations...")

# 8a. OLS (ignoring endogeneity) for comparison
for outcome, name in [('s_emp', 'emp'), ('s_enrolled_edu', 'enroll'), ('s_tot_income', 'income')]:
    result = run_ols_regression(df, outcome, 'treatment', full_controls,
                               f'iv/method/ols_{name}', 'methods/instrumental_variables.md#estimation-method')
    if result:
        results.append(result)

# 8b. OLS on lottery (ITT interpretation)
for outcome, name in [('s_emp', 'emp'), ('s_enrolled_edu', 'enroll'), ('s_tot_income', 'income')]:
    result = run_ols_regression(df, outcome, 'offered', full_controls,
                               f'iv/method/itt_{name}', 'methods/instrumental_variables.md#estimation-method')
    if result:
        results.append(result)

# ============================================================================
# 9. FUNCTIONAL FORM VARIATIONS
# ============================================================================
print("\n9. Running functional form variations...")

# 9a. Log income (for positive values)
df_pos = df[df['s_tot_income'] > 0].copy()
df_pos['log_income'] = np.log(df_pos['s_tot_income'])
result = run_iv_regression(df_pos, 'log_income', 'treatment', 'offered', full_controls,
                          'robust/funcform/log_income', 'robustness/functional_form.md')
if result:
    results.append(result)

# 9b. IHS income
df['ihs_income'] = np.arcsinh(df['s_tot_income'])
result = run_iv_regression(df, 'ihs_income', 'treatment', 'offered', full_controls,
                          'robust/funcform/ihs_income', 'robustness/functional_form.md')
if result:
    results.append(result)

# ============================================================================
# 10. HETEROGENEITY ANALYSIS
# ============================================================================
print("\n10. Running heterogeneity specifications...")

# Create interaction terms and run subgroup analyses
# We already ran gender and age subgroups above

# 10a. By social program participation
if 'social_tus' in df.columns:
    for condition, label in [(df['social_tus'] == 1, 'tus_participant'), (df['social_tus'] == 0, 'non_tus')]:
        df_sub = df[condition].copy()
        if len(df_sub) >= 100:
            result = run_iv_regression(df_sub, 's_emp', 'treatment', 'offered', full_controls,
                                      f'robust/heterogeneity/{label}_emp', 'robustness/heterogeneity.md')
            if result:
                results.append(result)

# 10b. By prior work experience (if available through positive earnings)
if 's_emp' in df.columns:
    # Young people likely have no prior work, but we can check applications
    for condition, label in [(df['applications'] == 1, 'first_time'), (df['applications'] > 1, 'repeat_applicant')]:
        df_sub = df[condition].copy()
        if len(df_sub) >= 100:
            result = run_iv_regression(df_sub, 's_emp', 'treatment', 'offered', full_controls,
                                      f'robust/heterogeneity/{label}_emp', 'robustness/heterogeneity.md')
            if result:
                results.append(result)

# 10c. By school timing (morning vs afternoon)
for condition, label in [(df['school_morning'] == 1, 'morning_school'), (df['school_afternoon'] == 1, 'afternoon_school')]:
    df_sub = df[condition].copy()
    if len(df_sub) >= 100:
        result = run_iv_regression(df_sub, 's_emp', 'treatment', 'offered', full_controls,
                                  f'robust/heterogeneity/{label}_emp', 'robustness/heterogeneity.md')
        if result:
            results.append(result)

# 10d. By ability (using low_ability/high_ability if available)
if 'low_ability' in df.columns:
    for condition, label in [(df['low_ability'] == 1, 'low_ability'), (df['high_ability'] == 1, 'high_ability')]:
        df_sub = df[condition].copy()
        if len(df_sub) >= 100:
            result = run_iv_regression(df_sub, 's_emp', 'treatment', 'offered', full_controls,
                                      f'robust/heterogeneity/{label}_emp', 'robustness/heterogeneity.md')
            if result:
                results.append(result)

# ============================================================================
# 11. PLACEBO TESTS
# ============================================================================
print("\n11. Running placebo specifications...")

# 11a. Balance check on pre-treatment characteristics
# If lottery is truly random, offered should not predict baseline characteristics
pre_treatment_vars = ['female', 'age', 'b_mother_comphighschool', 'b_father_comphighschool',
                      'b_books_more10', 'educ_sec_acad', 'educ_sec_tech']

for var in pre_treatment_vars:
    if var in df.columns:
        result = run_ols_regression(df, var, 'offered', strata_controls,
                                   f'robust/placebo/balance_{var}', 'robustness/placebo_tests.md')
        if result:
            results.append(result)

# ============================================================================
# 12. ADDITIONAL ROBUSTNESS - Job Quality Outcomes (Table E4)
# ============================================================================
print("\n12. Running additional outcome specifications...")

job_quality_outcomes = ['s_pension_contrib', 's_emp_public', 's_small_firm',
                        's_ind_manuf', 's_ind_trade', 's_ind_finserv', 's_ind_pubserv']

for outcome in job_quality_outcomes:
    if outcome in df.columns:
        result = run_iv_regression(df, outcome, 'treatment', 'offered', full_controls,
                                  f'robust/outcome/{outcome}', 'robustness/measurement.md')
        if result:
            results.append(result)

# ============================================================================
# 13. INCREMENTAL CONTROL ADDITIONS
# ============================================================================
print("\n13. Running incremental control specifications...")

# Start with just strata, add individual controls one by one
for i, ctrl in enumerate(individual_controls[:5]):  # First 5 for brevity
    if ctrl in df.columns:
        current_controls = strata_controls + individual_controls[:i+1]
        current_controls = [c for c in current_controls if c in df.columns]
        result = run_iv_regression(df, 's_emp', 'treatment', 'offered', current_controls,
                                  f'robust/control/add_{ctrl}', 'robustness/control_progression.md')
        if result:
            results.append(result)

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "=" * 60)
print(f"Total specifications run: {len(results)}")

# Convert to DataFrame and save
results_df = pd.DataFrame(results)

# Save to CSV
results_df.to_csv(OUTPUT_PATH + 'specification_results.csv', index=False)
print(f"Results saved to {OUTPUT_PATH}specification_results.csv")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

# Filter to main employment outcome for summary
main_results = results_df[results_df['outcome_var'] == 's_emp'].copy()

print("\n" + "=" * 60)
print("SUMMARY STATISTICS (Employment outcome)")
print("=" * 60)

if len(main_results) > 0:
    print(f"Total specifications: {len(main_results)}")
    print(f"Mean coefficient: {main_results['coefficient'].mean():.4f}")
    print(f"Median coefficient: {main_results['coefficient'].median():.4f}")
    print(f"Std dev: {main_results['coefficient'].std():.4f}")
    print(f"Range: [{main_results['coefficient'].min():.4f}, {main_results['coefficient'].max():.4f}]")

    sig_05 = (main_results['p_value'] < 0.05).sum()
    sig_01 = (main_results['p_value'] < 0.01).sum()
    positive = (main_results['coefficient'] > 0).sum()

    print(f"\nPositive coefficients: {positive} ({100*positive/len(main_results):.1f}%)")
    print(f"Significant at 5%: {sig_05} ({100*sig_05/len(main_results):.1f}%)")
    print(f"Significant at 1%: {sig_01} ({100*sig_01/len(main_results):.1f}%)")

print("\n" + "=" * 60)
print("ALL OUTCOMES SUMMARY")
print("=" * 60)
print(f"Total specifications: {len(results_df)}")
print(f"\nBy outcome variable:")
print(results_df['outcome_var'].value_counts().head(10))

print("\n\nSpecification search complete!")
