"""
Specification Search for Paper 202864-V1
"Eliciting ambiguity with mixing bets" - Patrick Schmidt

This paper studies ambiguity attitudes through a lab experiment where participants
make choices across different domains (risky urns, ambiguous urns, stock market, social dilemmas).

Method: Cross-sectional OLS and discrete choice (logit/probit)
Main outcomes:
  - Mixing intensity (continuous): how much participants mix between options
  - Multiple mixing (binary): whether participants mix at multiple quota levels
  - Probability belief (continuous): subjective probability from mixing behavior
  - Cooperation (binary): cooperation in prisoner's dilemma

Main explanatory variables:
  - Topic/domain (risk, ambiguous, stock, social)
  - Risk aversion
  - Demographics (age, gender, field of study, experience)
"""

import pandas as pd
import numpy as np
import pyreadr
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# Configuration
PAPER_ID = "202864-V1"
PAPER_TITLE = "Eliciting ambiguity with mixing bets"
JOURNAL = "AER: Insights"
BASE_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search"
DATA_PATH = f"{BASE_PATH}/data/downloads/extracted/{PAPER_ID}/RProject/data"

# Load data
df_summary = pyreadr.read_r(f'{DATA_PATH}/df.RDS')[None]
df_all = pyreadr.read_r(f'{DATA_PATH}/df.all.RDS')[None]

# Rename topics from German to English
def rename_topics(df):
    df = df.copy()
    topic_map = {
        'blauer Ball aus Urne': 'risk',
        'DAX': 'stock',
        'gepunkteter Ball aus Urne': 'ambiguous',
        'Alleingang oder Zusammenarbeit': 'social',
        'gepunkteter Ball nach 10 Ziehungen': 'updated'
    }
    df['topic'] = df['topic'].replace(topic_map)
    return df

df_summary = rename_topics(df_summary)
df_all = rename_topics(df_all)

# Standardize function (matching original R code)
def mystan(x):
    x = pd.to_numeric(x, errors='coerce')
    return (x - np.nanmean(x)) / np.nanstd(x)

# ============================================================================
# DATA PREPARATION
# ============================================================================

# Step 1: Create mixing intensity (attitude) from slider data
df_slider = df_all[df_all['player.elicit_type'] == 'slider'].copy()
df_slider = df_slider[~df_slider['topic'].isin(['updated'])]

# Calculate mixing value for each observation
def calc_mixing_val(row):
    x, q = row['x'], row['q']
    if x >= 1 - q:
        return (1 - x) / q if q > 0 else np.nan
    else:
        return x / (1 - q) if (1 - q) > 0 else np.nan

df_slider['mixing_val'] = df_slider.apply(calc_mixing_val, axis=1)
# Filter to mixing observations only (x not 0 or 1)
df_slider_mix = df_slider[(df_slider['x'] != 0) & (df_slider['x'] != 1)]

# Aggregate to person-topic level
att_person_topic = df_slider_mix.groupby(['participant.code', 'topic'])['mixing_val'].mean().reset_index()
att_person_topic.columns = ['participant_code', 'topic', 'att']

# Aggregate to person level (average across topics)
att_person = df_slider_mix.groupby('participant.code')['mixing_val'].mean().reset_index()
att_person.columns = ['participant_code', 'att']

# Step 2: Create demographics from summary data
person_demo = df_summary.groupby('participant.code').first().reset_index()
person_demo = person_demo[['participant.code', 'demographics.1.player.gender',
                           'demographics.1.player.fieldofstudy',
                           'demographics.1.player.comprehension',
                           'demographics.1.player.probabilityexp1',
                           'demographics.1.player.probabilityexp2',
                           'demographics.1.player.probabilityexp3',
                           'demographics.1.player.probabilityexp4',
                           'demographics.1.player.probabilityexp5',
                           'age', 'risk_aversion', 'cooperate',
                           'i_consistent', 'i_maxmin', 'i_seu']].copy()
person_demo.columns = ['participant_code', 'gender', 'field', 'comprehension',
                       'exp1', 'exp2', 'exp3', 'exp4', 'exp5',
                       'age', 'risk_aversion', 'cooperate',
                       'i_consistent', 'i_maxmin', 'i_seu']

# Create binary variables
person_demo['female'] = (person_demo['gender'] == 'weiblich').astype(float)
person_demo['understood'] = (person_demo['comprehension'] == 'Ja').astype(float)

# Fix age (some have birth year instead of age)
person_demo['age_clean'] = person_demo['age'].apply(lambda x: x if (15 < x < 65) else np.nan)
person_demo['age_clean'] = person_demo['age_clean'].fillna(person_demo['age_clean'].mean())

# Field of study indicators
person_demo['Fmint'] = person_demo['field'].isin(
    ['Mathematik', 'Physik', 'Informatik', 'Biochemie, Chemie, Pharmazie']).astype(float)
person_demo['Fsocial'] = person_demo['field'].isin(
    ['Sprach-und Kulturwissenschaften', 'Erziehungswissenschaften', 'Psychologie']).astype(float)
person_demo['Fecon'] = (person_demo['field'] == 'Wirtschaftswissenschaften').astype(float)
person_demo['Flaw'] = (person_demo['field'] == 'Rechtswissenschaft').astype(float)

# Experience variables
person_demo['exp1_n'] = pd.to_numeric(person_demo['exp1'], errors='coerce').fillna(0)
person_demo['exp2_n'] = pd.to_numeric(person_demo['exp2'], errors='coerce').fillna(0)
person_demo['exp3_n'] = (person_demo['exp3'] == 'Ja').astype(float)
person_demo['exp4_n'] = (person_demo['exp4'] == 500).astype(float)
person_demo['exp5_n'] = person_demo['exp5'].apply(lambda x: 1.0 if x in [25, 0.25] else 0.0)

# Standardized experience score
person_demo['experience'] = (mystan(person_demo['exp1_n']) +
                              mystan(person_demo['exp2_n']) +
                              mystan(person_demo['exp3_n']) +
                              mystan(person_demo['exp4_n']) +
                              mystan(person_demo['exp5_n']))

# Step 3: Create analysis dataset from discrete elicitation (choice_twice)
df_discrete = df_summary[df_summary['player.elicit_type'] == 'choice_twice'].copy()
df_discrete = df_discrete[~df_discrete['topic'].isin(['updated'])]
df_discrete['participant_code'] = df_discrete['participant.code']

# Key outcomes
df_discrete['multiple_mixing'] = (df_discrete['l'] > 0).astype(int)

# Merge with person-level attitude
df_analysis = df_discrete.merge(att_person, on='participant_code', how='left')

# Merge with demographics (select only needed columns to avoid conflicts)
demo_cols = person_demo[['participant_code', 'female', 'risk_aversion',
                          'experience', 'age_clean', 'understood',
                          'Fmint', 'Fsocial', 'Fecon', 'Flaw',
                          'i_consistent', 'cooperate']].copy()

# Drop columns that already exist in df_analysis to avoid merge conflicts
existing_cols = [c for c in demo_cols.columns if c in df_analysis.columns and c != 'participant_code']
for col in existing_cols:
    if col in df_analysis.columns:
        df_analysis = df_analysis.drop(columns=[col])

df_analysis = df_analysis.merge(demo_cols, on='participant_code', how='left')

# Create standardized versions
df_analysis['att_std'] = mystan(df_analysis['att'])
df_analysis['risk_std'] = mystan(df_analysis['risk_aversion'])
df_analysis['exp_std'] = mystan(df_analysis['experience'])
df_analysis['age_std'] = mystan(df_analysis['age_clean'])
df_analysis['p_std'] = mystan(df_analysis['p'])
df_analysis['l_std'] = mystan(df_analysis['l'])

# Fill any missing values
for col in ['att_std', 'risk_std', 'exp_std', 'age_std', 'female']:
    df_analysis[col] = df_analysis[col].fillna(df_analysis[col].mean())

print(f"Analysis dataset: {len(df_analysis)} observations")
print(f"Unique participants: {df_analysis['participant_code'].nunique()}")
print(f"Topics: {df_analysis['topic'].unique()}")

# Results storage
results = []

def extract_results(model, spec_id, spec_tree_path, outcome_var, treatment_var,
                   sample_desc, fixed_effects, controls_desc, cluster_var, model_type,
                   n_obs=None, sample_df=None):
    """Extract results from a statsmodels regression model."""

    # Get coefficient info for treatment variable
    treatment_found = False
    if treatment_var in model.params.index:
        coef = model.params[treatment_var]
        se = model.bse[treatment_var]
        t_stat = model.tvalues[treatment_var]
        p_value = model.pvalues[treatment_var]
        ci = model.conf_int().loc[treatment_var]
        ci_lower, ci_upper = ci[0], ci[1]
        treatment_found = True
    else:
        # For categorical treatment, use first non-intercept coefficient
        treatment_params = [p for p in model.params.index if treatment_var in p and 'Intercept' not in p]
        if treatment_params:
            first_treatment = treatment_params[0]
            coef = model.params[first_treatment]
            se = model.bse[first_treatment]
            t_stat = model.tvalues[first_treatment]
            p_value = model.pvalues[first_treatment]
            ci = model.conf_int().loc[first_treatment]
            ci_lower, ci_upper = ci[0], ci[1]
            treatment_found = True
        else:
            coef = se = t_stat = p_value = ci_lower = ci_upper = np.nan

    # Build coefficient vector JSON
    coef_vector = {
        "treatment": {
            "var": treatment_var,
            "coef": float(coef) if not np.isnan(coef) else None,
            "se": float(se) if not np.isnan(se) else None,
            "pval": float(p_value) if not np.isnan(p_value) else None
        },
        "controls": [],
        "fixed_effects": fixed_effects.split(', ') if fixed_effects else [],
        "diagnostics": {
            "r_squared": float(model.rsquared) if hasattr(model, 'rsquared') else None,
            "adj_r_squared": float(model.rsquared_adj) if hasattr(model, 'rsquared_adj') else None,
            "f_stat": float(model.fvalue) if hasattr(model, 'fvalue') and model.fvalue is not None else None
        }
    }

    # Add control coefficients
    for param in model.params.index:
        if param != treatment_var and 'Intercept' not in param and treatment_var not in param:
            coef_vector["controls"].append({
                "var": param,
                "coef": float(model.params[param]),
                "se": float(model.bse[param]),
                "pval": float(model.pvalues[param])
            })

    return {
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'outcome_var': outcome_var,
        'treatment_var': treatment_var,
        'coefficient': coef,
        'std_error': se,
        't_stat': t_stat,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_obs': n_obs if n_obs else int(model.nobs),
        'r_squared': model.rsquared if hasattr(model, 'rsquared') else np.nan,
        'coefficient_vector_json': json.dumps(coef_vector),
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    }


# ============================================================================
# BASELINE SPECIFICATIONS
# ============================================================================

print("\nRunning baseline specifications...")

# BASELINE 1: Mixing intensity explained by topic
try:
    model = smf.ols('att_std ~ C(topic)', data=df_analysis).fit()
    results.append(extract_results(
        model, 'baseline/topic', 'methods/cross_sectional_ols.md',
        'mixing_intensity_std', 'topic',
        'Person-topic level, discrete elicitation',
        'None', 'Topic dummies', 'None', 'OLS'
    ))
except Exception as e:
    print(f"Baseline topic error: {e}")

# BASELINE 2: Multiple mixing explained by topic
try:
    model = smf.ols('multiple_mixing ~ C(topic)', data=df_analysis).fit()
    results.append(extract_results(
        model, 'baseline/multiple_mixing', 'methods/cross_sectional_ols.md',
        'multiple_mixing', 'topic',
        'Discrete elicitation, person-topic level',
        'None', 'Topic dummies', 'None', 'OLS'
    ))
except Exception as e:
    print(f"Baseline multiple_mixing error: {e}")

# BASELINE 3: Table 1 style regression (ambiguous domain)
try:
    df_amb = df_analysis[df_analysis['topic'] == 'ambiguous'].copy()
    model = smf.ols('multiple_mixing ~ att_std + risk_std + female + exp_std', data=df_amb).fit()
    results.append(extract_results(
        model, 'baseline/table1_ambiguous', 'methods/cross_sectional_ols.md',
        'multiple_mixing', 'att_std',
        'Ambiguous domain only',
        'None', 'att_std, risk_std, female, exp_std', 'None', 'OLS'
    ))
except Exception as e:
    print(f"Baseline Table 1 ambiguous error: {e}")

# BASELINE 4: Table 1 for each domain
for topic_name in ['risk', 'ambiguous', 'stock', 'social']:
    try:
        df_topic = df_analysis[df_analysis['topic'] == topic_name].copy()
        model = smf.ols('multiple_mixing ~ att_std + risk_std + female + exp_std', data=df_topic).fit()
        results.append(extract_results(
            model, f'baseline/table1/{topic_name}', 'methods/cross_sectional_ols.md',
            'multiple_mixing', 'att_std',
            f'{topic_name} domain only',
            'None', 'att_std, risk_std, female, exp_std', 'None', 'OLS'
        ))
    except Exception as e:
        print(f"Baseline Table 1 {topic_name} error: {e}")


# ============================================================================
# CONTROL VARIATIONS (10-15 specs)
# ============================================================================

print("Running control variations...")

all_controls = ['att_std', 'risk_std', 'female', 'exp_std']
df_amb = df_analysis[df_analysis['topic'] == 'ambiguous'].copy()

# Leave-one-out: drop each control
for control in all_controls:
    try:
        remaining = [c for c in all_controls if c != control]
        formula = f'multiple_mixing ~ {" + ".join(remaining)}'
        model = smf.ols(formula, data=df_amb).fit()
        results.append(extract_results(
            model, f'robust/control/drop_{control}', 'robustness/leave_one_out.md',
            'multiple_mixing', remaining[0],
            f'Ambiguous domain, dropped {control}',
            'None', ', '.join(remaining), 'None', 'OLS'
        ))
    except Exception as e:
        print(f"LOO {control} error: {e}")

# No controls (bivariate)
try:
    model = smf.ols('multiple_mixing ~ att_std', data=df_amb).fit()
    results.append(extract_results(
        model, 'robust/control/none', 'robustness/control_progression.md',
        'multiple_mixing', 'att_std',
        'Ambiguous domain, no controls',
        'None', 'None', 'None', 'OLS'
    ))
except Exception as e:
    print(f"No controls error: {e}")

# Add controls incrementally
controls_sequence = [
    ['att_std'],
    ['att_std', 'risk_std'],
    ['att_std', 'risk_std', 'female'],
    ['att_std', 'risk_std', 'female', 'exp_std']
]
for i, controls in enumerate(controls_sequence):
    try:
        formula = f'multiple_mixing ~ {" + ".join(controls)}'
        model = smf.ols(formula, data=df_amb).fit()
        results.append(extract_results(
            model, f'robust/control/add_{i+1}', 'robustness/control_progression.md',
            'multiple_mixing', controls[0],
            f'Ambiguous domain, {i+1} control(s)',
            'None', ', '.join(controls), 'None', 'OLS'
        ))
    except Exception as e:
        print(f"Add controls {i} error: {e}")

# With age
try:
    model = smf.ols('multiple_mixing ~ att_std + risk_std + female + exp_std + age_std', data=df_amb).fit()
    results.append(extract_results(
        model, 'robust/control/add_age', 'robustness/control_progression.md',
        'multiple_mixing', 'att_std',
        'Ambiguous domain, with age control',
        'None', 'att_std, risk_std, female, exp_std, age_std', 'None', 'OLS'
    ))
except Exception as e:
    print(f"Add age error: {e}")


# ============================================================================
# SAMPLE RESTRICTIONS (10-15 specs)
# ============================================================================

print("Running sample restrictions...")

# By topic/domain
for topic_name in ['risk', 'ambiguous', 'stock', 'social']:
    try:
        df_topic = df_analysis[df_analysis['topic'] == topic_name].copy()
        model = smf.ols('multiple_mixing ~ att_std + risk_std + female + exp_std', data=df_topic).fit()
        results.append(extract_results(
            model, f'robust/sample/{topic_name}_only', 'robustness/sample_restrictions.md',
            'multiple_mixing', 'att_std',
            f'{topic_name} domain only',
            'None', 'att_std, risk_std, female, exp_std', 'None', 'OLS'
        ))
    except Exception as e:
        print(f"Sample {topic_name} error: {e}")

# By gender
try:
    df_male = df_analysis[df_analysis['female'] == 0].copy()
    model = smf.ols('multiple_mixing ~ att_std + risk_std + exp_std', data=df_male).fit()
    results.append(extract_results(
        model, 'robust/sample/male_only', 'robustness/sample_restrictions.md',
        'multiple_mixing', 'att_std',
        'Male participants only',
        'None', 'att_std, risk_std, exp_std', 'None', 'OLS'
    ))
except Exception as e:
    print(f"Male only error: {e}")

try:
    df_female = df_analysis[df_analysis['female'] == 1].copy()
    model = smf.ols('multiple_mixing ~ att_std + risk_std + exp_std', data=df_female).fit()
    results.append(extract_results(
        model, 'robust/sample/female_only', 'robustness/sample_restrictions.md',
        'multiple_mixing', 'att_std',
        'Female participants only',
        'None', 'att_std, risk_std, exp_std', 'None', 'OLS'
    ))
except Exception as e:
    print(f"Female only error: {e}")

# By comprehension (understood vs not)
try:
    df_understood = df_analysis[df_analysis['understood'] == 1].copy()
    model = smf.ols('multiple_mixing ~ att_std + risk_std + female + exp_std',
                   data=df_understood[df_understood['topic'] == 'ambiguous']).fit()
    results.append(extract_results(
        model, 'robust/sample/understood_only', 'robustness/sample_restrictions.md',
        'multiple_mixing', 'att_std',
        'Participants who understood instructions, ambiguous domain',
        'None', 'att_std, risk_std, female, exp_std', 'None', 'OLS'
    ))
except Exception as e:
    print(f"Understood only error: {e}")

# By risk aversion level
try:
    df_riskaverse = df_analysis[df_analysis['risk_aversion'] >= 1].copy()
    model = smf.ols('multiple_mixing ~ att_std + female + exp_std',
                   data=df_riskaverse[df_riskaverse['topic'] == 'ambiguous']).fit()
    results.append(extract_results(
        model, 'robust/sample/risk_averse', 'robustness/sample_restrictions.md',
        'multiple_mixing', 'att_std',
        'Risk averse participants (risk_aversion >= 1), ambiguous domain',
        'None', 'att_std, female, exp_std', 'None', 'OLS'
    ))
except Exception as e:
    print(f"Risk averse error: {e}")

try:
    df_riskseeking = df_analysis[df_analysis['risk_aversion'] == 0].copy()
    model = smf.ols('multiple_mixing ~ att_std + female + exp_std',
                   data=df_riskseeking[df_riskseeking['topic'] == 'ambiguous']).fit()
    results.append(extract_results(
        model, 'robust/sample/risk_seeking', 'robustness/sample_restrictions.md',
        'multiple_mixing', 'att_std',
        'Risk seeking participants (risk_aversion = 0), ambiguous domain',
        'None', 'att_std, female, exp_std', 'None', 'OLS'
    ))
except Exception as e:
    print(f"Risk seeking error: {e}")

# Artificial events only (risk and ambiguous urns)
try:
    df_artificial = df_analysis[df_analysis['topic'].isin(['risk', 'ambiguous'])].copy()
    model = smf.ols('multiple_mixing ~ att_std + risk_std + female + exp_std + C(topic)',
                   data=df_artificial).fit()
    results.append(extract_results(
        model, 'robust/sample/artificial_events', 'robustness/sample_restrictions.md',
        'multiple_mixing', 'att_std',
        'Artificial events only (risk and ambiguous urns)',
        'None', 'att_std, risk_std, female, exp_std, topic', 'None', 'OLS'
    ))
except Exception as e:
    print(f"Artificial events error: {e}")

# Natural events only
try:
    df_natural = df_analysis[df_analysis['topic'].isin(['stock', 'social'])].copy()
    model = smf.ols('multiple_mixing ~ att_std + risk_std + female + exp_std + C(topic)',
                   data=df_natural).fit()
    results.append(extract_results(
        model, 'robust/sample/natural_events', 'robustness/sample_restrictions.md',
        'multiple_mixing', 'att_std',
        'Natural events only (stock and social)',
        'None', 'att_std, risk_std, female, exp_std, topic', 'None', 'OLS'
    ))
except Exception as e:
    print(f"Natural events error: {e}")

# Exclude inconsistent participants
try:
    df_consistent = df_analysis[df_analysis['i_consistent'] == 1].copy()
    model = smf.ols('multiple_mixing ~ att_std + risk_std + female + exp_std',
                   data=df_consistent[df_consistent['topic'] == 'ambiguous']).fit()
    results.append(extract_results(
        model, 'robust/sample/consistent_only', 'robustness/sample_restrictions.md',
        'multiple_mixing', 'att_std',
        'Consistent participants only, ambiguous domain',
        'None', 'att_std, risk_std, female, exp_std', 'None', 'OLS'
    ))
except Exception as e:
    print(f"Consistent only error: {e}")


# ============================================================================
# ALTERNATIVE OUTCOMES (5-10 specs)
# ============================================================================

print("Running alternative outcomes...")

# Probability belief as outcome
try:
    model = smf.ols('p ~ att_std + risk_std + female + exp_std', data=df_amb).fit()
    results.append(extract_results(
        model, 'robust/outcome/probability', 'robustness/measurement.md',
        'probability', 'att_std',
        'Ambiguous domain, probability belief as outcome',
        'None', 'att_std, risk_std, female, exp_std', 'None', 'OLS'
    ))
except Exception as e:
    print(f"Probability outcome error: {e}")

# Interval length l (continuous) as outcome
try:
    model = smf.ols('l ~ att_std + risk_std + female + exp_std', data=df_amb).fit()
    results.append(extract_results(
        model, 'robust/outcome/interval_length', 'robustness/measurement.md',
        'interval_length', 'att_std',
        'Ambiguous domain, interval length as outcome',
        'None', 'att_std, risk_std, female, exp_std', 'None', 'OLS'
    ))
except Exception as e:
    print(f"Interval length outcome error: {e}")

# Number of mixing choices (num_mix) as outcome
try:
    model = smf.ols('num_mix ~ att_std + risk_std + female + exp_std', data=df_amb).fit()
    results.append(extract_results(
        model, 'robust/outcome/num_mix', 'robustness/measurement.md',
        'num_mix', 'att_std',
        'Ambiguous domain, number of mixing choices',
        'None', 'att_std, risk_std, female, exp_std', 'None', 'OLS'
    ))
except Exception as e:
    print(f"Num mix outcome error: {e}")

# Cooperation as outcome (for social domain)
try:
    df_social = df_analysis[df_analysis['topic'] == 'social'].copy()
    df_social['cooperate'] = df_social['cooperate'].astype(float)
    model = smf.ols('cooperate ~ att_std + risk_std + female', data=df_social).fit()
    results.append(extract_results(
        model, 'robust/outcome/cooperation', 'robustness/measurement.md',
        'cooperate', 'att_std',
        'Social domain, cooperation as outcome',
        'None', 'att_std, risk_std, female', 'None', 'OLS'
    ))
except Exception as e:
    print(f"Cooperation outcome error: {e}")

# SEU classification as outcome
try:
    model = smf.ols('class_seu ~ att_std + risk_std + female + exp_std', data=df_amb).fit()
    results.append(extract_results(
        model, 'robust/outcome/seu_classification', 'robustness/measurement.md',
        'class_seu', 'att_std',
        'Ambiguous domain, SEU classification as outcome',
        'None', 'att_std, risk_std, female, exp_std', 'None', 'OLS'
    ))
except Exception as e:
    print(f"SEU classification error: {e}")

# Maxmin classification
try:
    model = smf.ols('class_maxmin ~ att_std + risk_std + female + exp_std', data=df_amb).fit()
    results.append(extract_results(
        model, 'robust/outcome/maxmin_classification', 'robustness/measurement.md',
        'class_maxmin', 'att_std',
        'Ambiguous domain, maxmin classification as outcome',
        'None', 'att_std, risk_std, female, exp_std', 'None', 'OLS'
    ))
except Exception as e:
    print(f"Maxmin classification error: {e}")


# ============================================================================
# INFERENCE VARIATIONS (5-8 specs)
# ============================================================================

print("Running inference variations...")

# HC1 robust SE
try:
    model = smf.ols('multiple_mixing ~ att_std + risk_std + female + exp_std',
                   data=df_amb).fit(cov_type='HC1')
    results.append(extract_results(
        model, 'robust/se/hc1', 'robustness/clustering_variations.md',
        'multiple_mixing', 'att_std',
        'Ambiguous domain, HC1 robust SE',
        'None', 'att_std, risk_std, female, exp_std', 'robust_hc1', 'OLS'
    ))
except Exception as e:
    print(f"HC1 error: {e}")

# HC2 robust SE
try:
    model = smf.ols('multiple_mixing ~ att_std + risk_std + female + exp_std',
                   data=df_amb).fit(cov_type='HC2')
    results.append(extract_results(
        model, 'robust/se/hc2', 'robustness/clustering_variations.md',
        'multiple_mixing', 'att_std',
        'Ambiguous domain, HC2 robust SE',
        'None', 'att_std, risk_std, female, exp_std', 'robust_hc2', 'OLS'
    ))
except Exception as e:
    print(f"HC2 error: {e}")

# HC3 robust SE
try:
    model = smf.ols('multiple_mixing ~ att_std + risk_std + female + exp_std',
                   data=df_amb).fit(cov_type='HC3')
    results.append(extract_results(
        model, 'robust/se/hc3', 'robustness/clustering_variations.md',
        'multiple_mixing', 'att_std',
        'Ambiguous domain, HC3 robust SE',
        'None', 'att_std, risk_std, female, exp_std', 'robust_hc3', 'OLS'
    ))
except Exception as e:
    print(f"HC3 error: {e}")

# Cluster by participant (all domains)
try:
    model = smf.ols('multiple_mixing ~ att_std + risk_std + female + exp_std',
                   data=df_analysis).fit(cov_type='cluster',
                                        cov_kwds={'groups': df_analysis['participant_code']})
    results.append(extract_results(
        model, 'robust/cluster/participant', 'robustness/clustering_variations.md',
        'multiple_mixing', 'att_std',
        'All domains, clustered by participant',
        'None', 'att_std, risk_std, female, exp_std', 'participant', 'OLS'
    ))
except Exception as e:
    print(f"Cluster participant error: {e}")

# Cluster by topic
try:
    model = smf.ols('multiple_mixing ~ att_std + risk_std + female + exp_std',
                   data=df_analysis).fit(cov_type='cluster',
                                        cov_kwds={'groups': df_analysis['topic']})
    results.append(extract_results(
        model, 'robust/cluster/topic', 'robustness/clustering_variations.md',
        'multiple_mixing', 'att_std',
        'All domains, clustered by topic',
        'None', 'att_std, risk_std, female, exp_std', 'topic', 'OLS'
    ))
except Exception as e:
    print(f"Cluster topic error: {e}")


# ============================================================================
# ESTIMATION METHOD VARIATIONS (3-5 specs)
# ============================================================================

print("Running estimation method variations...")

# Logit for binary outcome
try:
    df_logit = df_amb.dropna(subset=['multiple_mixing', 'att_std', 'risk_std', 'female', 'exp_std'])
    model = smf.logit('multiple_mixing ~ att_std + risk_std + female + exp_std',
                     data=df_logit).fit(disp=0)
    results.append(extract_results(
        model, 'discrete/binary/logit', 'methods/discrete_choice.md',
        'multiple_mixing', 'att_std',
        'Ambiguous domain, logit model',
        'None', 'att_std, risk_std, female, exp_std', 'None', 'Logit'
    ))
except Exception as e:
    print(f"Logit error: {e}")

# Probit for binary outcome
try:
    model = smf.probit('multiple_mixing ~ att_std + risk_std + female + exp_std',
                      data=df_logit).fit(disp=0)
    results.append(extract_results(
        model, 'discrete/binary/probit', 'methods/discrete_choice.md',
        'multiple_mixing', 'att_std',
        'Ambiguous domain, probit model',
        'None', 'att_std, risk_std, female, exp_std', 'None', 'Probit'
    ))
except Exception as e:
    print(f"Probit error: {e}")

# OLS with topic fixed effects (all domains)
try:
    model = smf.ols('multiple_mixing ~ att_std + risk_std + female + exp_std + C(topic)',
                   data=df_analysis).fit()
    results.append(extract_results(
        model, 'robust/estimation/topic_fe', 'robustness/model_specification.md',
        'multiple_mixing', 'att_std',
        'All domains with topic fixed effects',
        'topic', 'att_std, risk_std, female, exp_std', 'None', 'OLS'
    ))
except Exception as e:
    print(f"Topic FE error: {e}")

# Pooled with topic interactions
try:
    model = smf.ols('multiple_mixing ~ att_std * C(topic) + risk_std + female + exp_std',
                   data=df_analysis).fit()
    results.append(extract_results(
        model, 'robust/estimation/pooled_interactions', 'robustness/model_specification.md',
        'multiple_mixing', 'att_std',
        'All domains with attitude x topic interactions',
        'None', 'att_std * topic, risk_std, female, exp_std', 'None', 'OLS'
    ))
except Exception as e:
    print(f"Pooled interactions error: {e}")


# ============================================================================
# FUNCTIONAL FORM VARIATIONS (3-5 specs)
# ============================================================================

print("Running functional form variations...")

# Log-odds transformation for probability outcome
try:
    df_amb_logit = df_amb.copy()
    df_amb_logit['p_logit'] = np.log((df_amb_logit['p'] + 0.01) / (1 - df_amb_logit['p'] + 0.01))
    model = smf.ols('p_logit ~ att_std + risk_std + female + exp_std', data=df_amb_logit).fit()
    results.append(extract_results(
        model, 'robust/funcform/logit_transform', 'robustness/functional_form.md',
        'p_logit', 'att_std',
        'Ambiguous domain, log-odds transformed probability',
        'None', 'att_std, risk_std, female, exp_std', 'None', 'OLS'
    ))
except Exception as e:
    print(f"Logit transform error: {e}")

# Quadratic in mixing attitude
try:
    df_amb_quad = df_amb.copy()
    df_amb_quad['att_std_sq'] = df_amb_quad['att_std'] ** 2
    model = smf.ols('multiple_mixing ~ att_std + att_std_sq + risk_std + female + exp_std',
                   data=df_amb_quad).fit()
    results.append(extract_results(
        model, 'robust/funcform/quadratic_att', 'robustness/functional_form.md',
        'multiple_mixing', 'att_std',
        'Ambiguous domain, quadratic in mixing attitude',
        'None', 'att_std, att_std^2, risk_std, female, exp_std', 'None', 'OLS'
    ))
except Exception as e:
    print(f"Quadratic error: {e}")

# Categorical risk aversion instead of continuous
try:
    model = smf.ols('multiple_mixing ~ att_std + C(risk_aversion) + female + exp_std',
                   data=df_amb).fit()
    results.append(extract_results(
        model, 'robust/funcform/categorical_risk', 'robustness/functional_form.md',
        'multiple_mixing', 'att_std',
        'Ambiguous domain, categorical risk aversion',
        'None', 'att_std, C(risk_aversion), female, exp_std', 'None', 'OLS'
    ))
except Exception as e:
    print(f"Categorical risk error: {e}")


# ============================================================================
# HETEROGENEITY ANALYSES (5-10 specs)
# ============================================================================

print("Running heterogeneity analyses...")

# Interaction with gender
try:
    model = smf.ols('multiple_mixing ~ att_std * female + risk_std + exp_std', data=df_amb).fit()
    results.append(extract_results(
        model, 'robust/het/interaction_gender', 'robustness/heterogeneity.md',
        'multiple_mixing', 'att_std',
        'Ambiguous domain, attitude x gender interaction',
        'None', 'att_std * female, risk_std, exp_std', 'None', 'OLS'
    ))
except Exception as e:
    print(f"Gender interaction error: {e}")

# Interaction with risk aversion
try:
    model = smf.ols('multiple_mixing ~ att_std * risk_std + female + exp_std', data=df_amb).fit()
    results.append(extract_results(
        model, 'robust/het/interaction_risk', 'robustness/heterogeneity.md',
        'multiple_mixing', 'att_std',
        'Ambiguous domain, attitude x risk aversion interaction',
        'None', 'att_std * risk_std, female, exp_std', 'None', 'OLS'
    ))
except Exception as e:
    print(f"Risk interaction error: {e}")

# Interaction with experience
try:
    model = smf.ols('multiple_mixing ~ att_std * exp_std + risk_std + female', data=df_amb).fit()
    results.append(extract_results(
        model, 'robust/het/interaction_experience', 'robustness/heterogeneity.md',
        'multiple_mixing', 'att_std',
        'Ambiguous domain, attitude x experience interaction',
        'None', 'att_std * exp_std, risk_std, female', 'None', 'OLS'
    ))
except Exception as e:
    print(f"Experience interaction error: {e}")

# Triple interaction: attitude x topic x gender (all domains)
try:
    model = smf.ols('multiple_mixing ~ att_std * C(topic) * female + risk_std + exp_std',
                   data=df_analysis).fit()
    results.append(extract_results(
        model, 'robust/het/triple_diff', 'robustness/heterogeneity.md',
        'multiple_mixing', 'att_std',
        'All domains, triple interaction att x topic x gender',
        'None', 'att_std * topic * female, risk_std, exp_std', 'None', 'OLS'
    ))
except Exception as e:
    print(f"Triple interaction error: {e}")

# By field of study
for field in ['Fecon', 'Fmint', 'Fsocial']:
    try:
        df_field = df_analysis[(df_analysis[field] == 1) & (df_analysis['topic'] == 'ambiguous')].copy()
        if len(df_field) >= 10:
            model = smf.ols('multiple_mixing ~ att_std + risk_std + female + exp_std',
                           data=df_field).fit()
            results.append(extract_results(
                model, f'robust/het/by_{field.lower()}', 'robustness/heterogeneity.md',
                'multiple_mixing', 'att_std',
                f'{field} students, ambiguous domain',
                'None', 'att_std, risk_std, female, exp_std', 'None', 'OLS'
            ))
    except Exception as e:
        print(f"Field {field} error: {e}")

# By high vs low mixing attitude
try:
    median_att = df_analysis['att'].median()
    df_high_att = df_analysis[(df_analysis['att'] > median_att) & (df_analysis['topic'] == 'ambiguous')]
    df_low_att = df_analysis[(df_analysis['att'] <= median_att) & (df_analysis['topic'] == 'ambiguous')]

    model_high = smf.ols('multiple_mixing ~ risk_std + female + exp_std', data=df_high_att).fit()
    results.append(extract_results(
        model_high, 'robust/het/high_mixing_attitude', 'robustness/heterogeneity.md',
        'multiple_mixing', 'risk_std',
        'High mixing attitude (above median), ambiguous domain',
        'None', 'risk_std, female, exp_std', 'None', 'OLS'
    ))

    model_low = smf.ols('multiple_mixing ~ risk_std + female + exp_std', data=df_low_att).fit()
    results.append(extract_results(
        model_low, 'robust/het/low_mixing_attitude', 'robustness/heterogeneity.md',
        'multiple_mixing', 'risk_std',
        'Low mixing attitude (below median), ambiguous domain',
        'None', 'risk_std, female, exp_std', 'None', 'OLS'
    ))
except Exception as e:
    print(f"High/low att error: {e}")


# ============================================================================
# PLACEBO AND ADDITIONAL TESTS (3-5 specs)
# ============================================================================

print("Running placebo tests...")

# Risk domain should show less mixing than ambiguous
try:
    df_risk_amb = df_analysis[df_analysis['topic'].isin(['risk', 'ambiguous'])].copy()
    df_risk_amb['is_ambiguous'] = (df_risk_amb['topic'] == 'ambiguous').astype(int)
    model = smf.ols('multiple_mixing ~ is_ambiguous + att_std + risk_std + female + exp_std',
                   data=df_risk_amb).fit()
    results.append(extract_results(
        model, 'robust/placebo/risk_vs_ambiguous', 'robustness/placebo_tests.md',
        'multiple_mixing', 'is_ambiguous',
        'Risk vs ambiguous domain comparison',
        'None', 'is_ambiguous, att_std, risk_std, female, exp_std', 'None', 'OLS'
    ))
except Exception as e:
    print(f"Risk vs ambiguous error: {e}")

# Updated domain (with updated data)
try:
    df_updated = df_summary[df_summary['player.elicit_type'] == 'choice_twice'].copy()
    df_updated = df_updated[df_updated['topic'].isin(['ambiguous', 'updated'])]
    df_updated['participant_code'] = df_updated['participant.code']
    df_updated = df_updated.merge(att_person, on='participant_code', how='left')
    df_updated = df_updated.merge(person_demo[['participant_code', 'female', 'risk_aversion', 'experience']],
                                   on='participant_code', how='left')
    df_updated['att_std'] = mystan(df_updated['att'])
    df_updated['risk_std'] = mystan(df_updated['risk_aversion'])
    df_updated['exp_std'] = mystan(df_updated['experience'])
    df_updated['multiple_mixing'] = (df_updated['l'] > 0).astype(int)
    df_updated['is_updated'] = (df_updated['topic'] == 'updated').astype(int)

    model = smf.ols('multiple_mixing ~ is_updated + att_std + risk_std + female + exp_std',
                   data=df_updated).fit()
    results.append(extract_results(
        model, 'robust/placebo/updated_vs_ambiguous', 'robustness/placebo_tests.md',
        'multiple_mixing', 'is_updated',
        'Updated vs ambiguous domain (information effect)',
        'None', 'is_updated, att_std, risk_std, female, exp_std', 'None', 'OLS'
    ))
except Exception as e:
    print(f"Updated vs ambiguous error: {e}")


# ============================================================================
# T-TESTS FROM PAPER (comparing domains)
# ============================================================================

print("Running t-tests from paper...")

# Pairwise t-tests for mixing rates across domains
domains = ['risk', 'ambiguous', 'stock', 'social']
for i, d1 in enumerate(domains):
    for d2 in domains[i+1:]:
        try:
            df_d1 = df_analysis[df_analysis['topic'] == d1]['multiple_mixing']
            df_d2 = df_analysis[df_analysis['topic'] == d2]['multiple_mixing']
            t_stat, p_val = stats.ttest_ind(df_d1, df_d2)

            result = {
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': f'ttest/{d1}_vs_{d2}',
                'spec_tree_path': 'robustness/placebo_tests.md',
                'outcome_var': 'multiple_mixing',
                'treatment_var': f'{d1}_vs_{d2}',
                'coefficient': df_d1.mean() - df_d2.mean(),
                'std_error': np.sqrt(df_d1.var()/len(df_d1) + df_d2.var()/len(df_d2)),
                't_stat': t_stat,
                'p_value': p_val,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'n_obs': len(df_d1) + len(df_d2),
                'r_squared': np.nan,
                'coefficient_vector_json': json.dumps({
                    'test_type': 't-test',
                    'd1_mean': float(df_d1.mean()),
                    'd2_mean': float(df_d2.mean()),
                    'd1_n': int(len(df_d1)),
                    'd2_n': int(len(df_d2))
                }),
                'sample_desc': f'{d1} vs {d2} comparison',
                'fixed_effects': 'None',
                'controls_desc': 'None',
                'cluster_var': 'None',
                'model_type': 't-test',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            }
            results.append(result)
        except Exception as e:
            print(f"T-test {d1} vs {d2} error: {e}")


# ============================================================================
# COOPERATION ANALYSIS (Table 2 in supplement)
# ============================================================================

print("Running cooperation analysis...")

try:
    df_coop = df_analysis[df_analysis['topic'] == 'social'].copy()
    df_coop['cooperate'] = df_coop['cooperate'].astype(float)
    df_coop['mult_mix'] = (df_coop['l'] > 0).astype(float)

    # Full model from Table 2
    model = smf.ols('cooperate ~ att_std + risk_std + mult_mix + p_std', data=df_coop).fit()
    results.append(extract_results(
        model, 'table2/cooperation_full', 'methods/cross_sectional_ols.md',
        'cooperate', 'att_std',
        'Social domain, cooperation explained by ambiguity measures',
        'None', 'att_std, risk_std, multiple_mixing, probability (std)', 'None', 'OLS'
    ))

    # Simpler models
    model = smf.ols('cooperate ~ att_std', data=df_coop).fit()
    results.append(extract_results(
        model, 'table2/cooperation_att_only', 'methods/cross_sectional_ols.md',
        'cooperate', 'att_std',
        'Social domain, cooperation explained by mixing intensity only',
        'None', 'att_std', 'None', 'OLS'
    ))

    model = smf.ols('cooperate ~ att_std + risk_std', data=df_coop).fit()
    results.append(extract_results(
        model, 'table2/cooperation_att_risk', 'methods/cross_sectional_ols.md',
        'cooperate', 'att_std',
        'Social domain, cooperation with mixing and risk',
        'None', 'att_std, risk_std', 'None', 'OLS'
    ))
except Exception as e:
    print(f"Cooperation analysis error: {e}")


# ============================================================================
# SAVE RESULTS
# ============================================================================

print(f"\nTotal specifications run: {len(results)}")

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
output_path = f"{BASE_PATH}/data/downloads/extracted/{PAPER_ID}/specification_results.csv"
results_df.to_csv(output_path, index=False)
print(f"Results saved to: {output_path}")

# Summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Total specifications: {len(results_df)}")
print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")
print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")
print(f"Range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")

# By category
print("\n" + "-"*60)
print("BY CATEGORY:")
print("-"*60)
categories = {
    'Baseline': results_df['spec_id'].str.startswith('baseline'),
    'Control variations': results_df['spec_id'].str.startswith('robust/control'),
    'Sample restrictions': results_df['spec_id'].str.startswith('robust/sample'),
    'Alternative outcomes': results_df['spec_id'].str.startswith('robust/outcome'),
    'Inference variations': results_df['spec_id'].str.startswith('robust/se') | results_df['spec_id'].str.startswith('robust/cluster'),
    'Estimation method': results_df['spec_id'].str.startswith('discrete') | results_df['spec_id'].str.startswith('robust/estimation'),
    'Functional form': results_df['spec_id'].str.startswith('robust/funcform'),
    'Heterogeneity': results_df['spec_id'].str.startswith('robust/het'),
    'Placebo tests': results_df['spec_id'].str.startswith('robust/placebo') | results_df['spec_id'].str.startswith('ttest'),
    'Table analyses': results_df['spec_id'].str.startswith('table'),
}

for cat, mask in categories.items():
    n = mask.sum()
    if n > 0:
        pct_pos = 100 * (results_df.loc[mask, 'coefficient'] > 0).mean()
        pct_sig = 100 * (results_df.loc[mask, 'p_value'] < 0.05).mean()
        print(f"{cat}: {n} specs, {pct_pos:.0f}% positive, {pct_sig:.0f}% sig at 5%")

print("\nDone!")
