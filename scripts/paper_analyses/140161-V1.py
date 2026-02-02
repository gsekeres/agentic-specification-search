#!/usr/bin/env python3
"""
Specification Search: Paper 140161-V1
Title: Sharing Fake News/Misinformation on Facebook - Survey Experiment Study

This paper studies how fact-checking affects the sharing of fake news ("Alt-Facts")
on Facebook in France during the 2019 European Parliament elections.

Treatment groups:
- Survey 1: Alt-Facts only (control for sharing Alt-Facts)
- Survey 2: Imposed Fact-Check (forced to see fact-check after Alt-Facts)
- Survey 3: Voluntary Fact-Check (can choose to see fact-check)

Main outcomes:
- want_share_fb: Intent to share Alt-Facts on Facebook (binary)
- share_click2: Actual sharing action of Alt-Facts (binary)
- want_share_facts: Intent to share Fact-Check on Facebook (binary)
- share_facts_click2: Actual sharing action of Fact-Check (binary)

Method: Cross-sectional OLS with robust standard errors
Identification: Random assignment to survey treatments

Author: Specification Search Agent
Date: 2026-02-02
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import json
import warnings
import os

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

PAPER_ID = "140161-V1"
PAPER_TITLE = "Sharing Fake News: Survey Experiment on Fact-Checking Effects"
JOURNAL = "AEJ-Policy"
METHOD = "cross_sectional_ols"
METHOD_TREE_PATH = "methods/cross_sectional_ols.md"

BASE_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search"
DATA_PATH = f"{BASE_PATH}/data/downloads/extracted/140161-V1/original_data"
OUTPUT_PATH = f"{BASE_PATH}/data/downloads/extracted/140161-V1"

# =============================================================================
# Data Loading and Preparation
# =============================================================================

def load_and_prepare_data():
    """Load and prepare the survey data from CSV files."""

    # Load survey files
    survey1 = pd.read_csv(f"{DATA_PATH}/Survey+1_May+27%2C+2019_01.02.csv", low_memory=False)
    survey2 = pd.read_csv(f"{DATA_PATH}/Survey+2_May+27%2C+2019_01.03.csv", low_memory=False)
    survey3 = pd.read_csv(f"{DATA_PATH}/Survey+3_May+27%2C+2019_01.03.csv", low_memory=False)

    # Process each survey
    dfs = []
    for i, survey_df in enumerate([survey1, survey2, survey3], 1):
        df = process_survey(survey_df, survey_id=i)
        dfs.append(df)

    # Combine surveys
    df = pd.concat(dfs, ignore_index=True)

    # Create additional variables
    df = create_analysis_variables(df)

    return df

def process_survey(df, survey_id):
    """Process a single survey file."""

    # Skip metadata rows
    df = df.iloc[3:].copy()
    df = df[df['DistributionChannel'] != 'preview'].copy()

    # Convert duration to numeric and filter
    df['Duration (in seconds)'] = pd.to_numeric(df['Duration (in seconds)'], errors='coerce')
    df = df[df['Duration (in seconds)'] >= 250].copy()

    # Add survey identifier
    df['survey'] = survey_id

    # Rename and process key variables
    # Q2 = Age, Q3 = City size, Q4 = Education, Q5 = Sex
    # Q34 = Share Fake News on Facebook, Q37 = Share with others

    # Extract relevant columns and rename
    column_map = {
        'Q2': 'age',
        'Q3': 'city_size',
        'Q4': 'education',
        'Q5': 'sex',
        'Q10': 'marital_status',
        'Q11': 'income',
        'Q12': 'has_children',
        'Q13': 'fb_usage',
        'Q14': 'fb_friends',
        'Q15': 'fb_share_freq',
        'Q21': 'religion',
        'Q22': 'religious_freq',
        'Q34': 'want_share_fb_raw',
        'Q37': 'want_share_others_raw',
        'Q29': 'vote_first_round',
        'Q30': 'vote_second_round',
        'Q86': 'image_eu',
        'Q18_1': 'charity_money',
        'Q18_2': 'homeless_money',
        'Q18_3': 'charity_work',
        'Q18_4': 'blood_donor',
        'Q16_1': 'share_reason_interest',
        'Q16_2': 'share_reason_influence',
        'Q16_3': 'share_reason_image',
        'Q16_4': 'share_reason_reciprocity',
        'Q19_1': 'help_giveback',
        'Q19_2': 'help_comfort',
        'Q19_3': 'help_cost',
        'Q20_1': 'importance_selfishness'
    }

    # Select and rename columns
    available_cols = [c for c in column_map.keys() if c in df.columns]
    df_subset = df[available_cols + ['survey', 'Duration (in seconds)']].copy()
    df_subset = df_subset.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

    # Convert numeric columns
    for col in df_subset.columns:
        if col != 'survey':
            df_subset[col] = pd.to_numeric(df_subset[col], errors='coerce')

    return df_subset

def create_analysis_variables(df):
    """Create derived variables for analysis."""

    # Main outcome: Want to share fake news on FB (Q34 == 1)
    df['want_share_fb'] = (df['want_share_fb_raw'] == 1).astype(float)
    df.loc[df['want_share_fb_raw'].isna(), 'want_share_fb'] = np.nan

    # Share with others
    df['want_share_others'] = (df['want_share_others_raw'] == 1).astype(float)
    df.loc[df['want_share_others_raw'].isna(), 'want_share_others'] = np.nan

    # Demographics
    df['male'] = (df['sex'] == 1).astype(float)
    df.loc[df['sex'].isna(), 'male'] = np.nan

    df['age_sqr'] = df['age'] ** 2

    # Education levels
    df['low_educ'] = (df['education'] <= 7).astype(float)
    df['mid_educ'] = (df['education'] == 8).astype(float)
    df['high_educ'] = (df['education'] == 9).astype(float)

    # City size
    df['village'] = (df['city_size'] == 1).astype(float)
    df['town'] = (df['city_size'] == 2).astype(float)
    df['city'] = (df['city_size'] == 3).astype(float)

    # Marital status
    df['married'] = (df['marital_status'] == 2).astype(float)
    df['single'] = (df['marital_status'] == 1).astype(float)

    # Children
    df['children'] = (df['has_children'] == 1).astype(float)

    # Religion
    df['catholic'] = (df['religion'] == 1).astype(float)
    df['muslim'] = (df['religion'] == 4).astype(float)
    df['no_religion'] = (df['religion'] == 7).astype(float)
    df['religious'] = (df['religious_freq'] <= 4).astype(float)

    # Facebook usage
    df['use_FB'] = 5 - df['fb_usage']
    df.loc[df['fb_usage'] == 5, 'use_FB'] = np.nan
    df['often_share_fb'] = (df['fb_share_freq'] == 3).astype(float)
    df['fb_friends_clean'] = df['fb_friends'].copy()
    df.loc[df['fb_friends'] > 5000, 'fb_friends_clean'] = np.nan
    df['log_nb_friends_fb'] = np.log(df['fb_friends_clean'] + 1)

    # Political variables - Vote for Le Pen
    df['first_mlp'] = (df['vote_first_round'] == 2).astype(float)
    df.loc[df['vote_first_round'].isna(), 'first_mlp'] = np.nan
    df['second_mlp'] = (df['vote_second_round'] == 1).astype(float)
    df.loc[df['vote_second_round'].isna(), 'second_mlp'] = np.nan

    # EU image (negative)
    df['negative_image_UE'] = df['image_eu']

    # Altruism index (average of charity behaviors)
    df['altruism'] = (df['charity_money'].fillna(0) + df['homeless_money'].fillna(0) +
                      df['charity_work'].fillna(0) + df['blood_donor'].fillna(0)) / 20

    # Reciprocity index
    df['reciprocity'] = (df['help_giveback'].fillna(0) + df['help_comfort'].fillna(0) +
                         df['help_cost'].fillna(0)) / 15

    # Image importance
    df['image'] = df['importance_selfishness'] / 5

    # Share reasons (scaled)
    for col in ['share_reason_interest', 'share_reason_influence', 'share_reason_image', 'share_reason_reciprocity']:
        if col in df.columns:
            df[col] = df[col] / 5

    # Treatment indicators
    df['survey2'] = (df['survey'] == 2).astype(int)
    df['survey3'] = (df['survey'] == 3).astype(int)

    # Age groups
    df['young'] = (df['age'] <= 35).astype(float)
    df['mid_age'] = ((df['age'] > 35) & (df['age'] < 65)).astype(float)
    df['old'] = (df['age'] >= 65).astype(float)

    # Income (clean)
    df['income_clean'] = df['income'].copy()
    df.loc[df['income'] >= 11, 'income_clean'] = np.nan

    return df

# =============================================================================
# Regression Functions
# =============================================================================

def run_ols_robust(df, formula, outcome_var='want_share_fb', treatment_var='survey2'):
    """Run OLS with robust standard errors."""

    try:
        # Create analysis sample
        df_clean = df.dropna(subset=[outcome_var]).copy()

        model = smf.ols(formula, data=df_clean).fit(cov_type='HC1')

        # Extract treatment coefficient
        if treatment_var in model.params.index:
            coef = model.params[treatment_var]
            se = model.bse[treatment_var]
            tstat = model.tvalues[treatment_var]
            pval = model.pvalues[treatment_var]
            ci = model.conf_int().loc[treatment_var]
        else:
            # Try alternative treatment variable names
            for alt in ['survey2', 'survey3', 'C(survey)[T.2]', 'C(survey)[T.3]']:
                if alt in model.params.index:
                    coef = model.params[alt]
                    se = model.bse[alt]
                    tstat = model.tvalues[alt]
                    pval = model.pvalues[alt]
                    ci = model.conf_int().loc[alt]
                    treatment_var = alt
                    break
            else:
                return None

        # Build coefficient vector JSON
        coef_vector = {
            "treatment": {
                "var": treatment_var,
                "coef": float(coef),
                "se": float(se),
                "pval": float(pval)
            },
            "controls": [],
            "diagnostics": {
                "r_squared": float(model.rsquared),
                "adj_r_squared": float(model.rsquared_adj),
                "f_stat": float(model.fvalue) if model.fvalue else None,
                "f_pval": float(model.f_pvalue) if model.f_pvalue else None
            }
        }

        # Add control coefficients
        for var in model.params.index:
            if var not in ['Intercept', treatment_var]:
                coef_vector["controls"].append({
                    "var": var,
                    "coef": float(model.params[var]),
                    "se": float(model.bse[var]),
                    "pval": float(model.pvalues[var])
                })

        return {
            'coefficient': float(coef),
            'std_error': float(se),
            't_stat': float(tstat),
            'p_value': float(pval),
            'ci_lower': float(ci[0]),
            'ci_upper': float(ci[1]),
            'n_obs': int(model.nobs),
            'r_squared': float(model.rsquared),
            'coefficient_vector_json': json.dumps(coef_vector)
        }
    except Exception as e:
        print(f"Error in regression: {e}")
        return None

def run_logit(df, formula, outcome_var='want_share_fb', treatment_var='survey2'):
    """Run Logit with robust standard errors for binary outcomes."""

    try:
        df_clean = df.dropna(subset=[outcome_var]).copy()

        model = smf.logit(formula, data=df_clean).fit(cov_type='HC1', disp=False)

        # Extract treatment coefficient (log-odds)
        if treatment_var in model.params.index:
            coef = model.params[treatment_var]
            se = model.bse[treatment_var]
            tstat = model.tvalues[treatment_var]
            pval = model.pvalues[treatment_var]
            ci = model.conf_int().loc[treatment_var]
        else:
            for alt in ['survey2', 'survey3', 'C(survey)[T.2]', 'C(survey)[T.3]']:
                if alt in model.params.index:
                    coef = model.params[alt]
                    se = model.bse[alt]
                    tstat = model.tvalues[alt]
                    pval = model.pvalues[alt]
                    ci = model.conf_int().loc[alt]
                    treatment_var = alt
                    break
            else:
                return None

        coef_vector = {
            "treatment": {
                "var": treatment_var,
                "coef": float(coef),
                "se": float(se),
                "pval": float(pval)
            },
            "controls": [],
            "diagnostics": {
                "pseudo_r_squared": float(model.prsquared)
            }
        }

        return {
            'coefficient': float(coef),
            'std_error': float(se),
            't_stat': float(tstat),
            'p_value': float(pval),
            'ci_lower': float(ci[0]),
            'ci_upper': float(ci[1]),
            'n_obs': int(model.nobs),
            'r_squared': float(model.prsquared),
            'coefficient_vector_json': json.dumps(coef_vector)
        }
    except Exception as e:
        print(f"Error in logit: {e}")
        return None

# =============================================================================
# Specification Search
# =============================================================================

def run_specification_search(df):
    """Run comprehensive specification search."""

    results = []
    spec_count = 0

    # Define control sets
    strata_controls = ['male', 'low_educ', 'mid_educ']
    socio_controls = ['age', 'age_sqr', 'income_clean', 'married', 'single', 'village', 'town',
                      'children', 'catholic', 'muslim', 'no_religion', 'religious']
    fb_controls = ['use_FB', 'often_share_fb', 'log_nb_friends_fb']
    vote_controls = ['second_mlp', 'negative_image_UE']
    behavioral_controls = ['altruism', 'reciprocity', 'image']
    share_reasons = ['share_reason_interest', 'share_reason_influence',
                     'share_reason_image', 'share_reason_reciprocity']

    all_controls = strata_controls + socio_controls + fb_controls + vote_controls + behavioral_controls

    # Filter to main experiment (surveys 1, 2, 3)
    df_main = df[df['survey'] <= 3].copy()

    # ===========================================================================
    # BASELINE SPECIFICATIONS (Table 2 replication)
    # ===========================================================================

    print("Running baseline specifications...")

    # Baseline 1: No controls
    outcome = 'want_share_fb'
    formula = f"{outcome} ~ C(survey)"
    result = run_ols_robust(df_main, formula, outcome_var=outcome, treatment_var='C(survey)[T.2]')
    if result:
        spec_count += 1
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'baseline',
            'spec_tree_path': 'methods/cross_sectional_ols.md#baseline',
            'outcome_var': outcome,
            'treatment_var': 'survey (Imposed Fact-Check)',
            'sample_desc': 'Main experiment, surveys 1-3',
            'fixed_effects': 'None',
            'controls_desc': 'No controls',
            'cluster_var': 'None (robust SE)',
            'model_type': 'OLS',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
            **result
        })

    # ===========================================================================
    # CONTROL PROGRESSION (robust/build)
    # ===========================================================================

    print("Running control progression...")

    # Build-up specifications
    control_progressions = [
        ('robust/build/bivariate', [], 'No controls'),
        ('robust/build/strata', strata_controls, 'Stratification: male, education'),
        ('robust/build/demographics', strata_controls + ['age', 'age_sqr'], 'Strata + age'),
        ('robust/build/socioeconomic', strata_controls + socio_controls, 'Strata + socioeconomic'),
        ('robust/build/fb_vars', strata_controls + socio_controls + fb_controls, 'Strata + socio + FB'),
        ('robust/build/vote', strata_controls + socio_controls + fb_controls + vote_controls,
         'Strata + socio + FB + political'),
        ('robust/build/full', all_controls, 'All controls'),
    ]

    for spec_id, controls, desc in control_progressions:
        if controls:
            control_str = ' + '.join(controls)
            formula = f"{outcome} ~ C(survey) + {control_str}"
        else:
            formula = f"{outcome} ~ C(survey)"

        result = run_ols_robust(df_main, formula, outcome_var=outcome, treatment_var='C(survey)[T.2]')
        if result:
            spec_count += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': spec_id,
                'spec_tree_path': 'robustness/control_progression.md',
                'outcome_var': outcome,
                'treatment_var': 'survey (Imposed Fact-Check)',
                'sample_desc': 'Main experiment, surveys 1-3',
                'fixed_effects': 'None',
                'controls_desc': desc,
                'cluster_var': 'None (robust SE)',
                'model_type': 'OLS',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
                **result
            })

    # ===========================================================================
    # LEAVE-ONE-OUT CONTROL VARIATIONS
    # ===========================================================================

    print("Running leave-one-out specifications...")

    # Leave out each control from full model
    for control in all_controls:
        remaining = [c for c in all_controls if c != control]
        control_str = ' + '.join(remaining)
        formula = f"{outcome} ~ C(survey) + {control_str}"

        result = run_ols_robust(df_main, formula, outcome_var=outcome, treatment_var='C(survey)[T.2]')
        if result:
            spec_count += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': f'robust/loo/drop_{control}',
                'spec_tree_path': 'robustness/leave_one_out.md',
                'outcome_var': outcome,
                'treatment_var': 'survey (Imposed Fact-Check)',
                'sample_desc': 'Main experiment, surveys 1-3',
                'fixed_effects': 'None',
                'controls_desc': f'Full controls minus {control}',
                'cluster_var': 'None (robust SE)',
                'model_type': 'OLS',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
                **result
            })

    # ===========================================================================
    # ALTERNATIVE OUTCOMES
    # ===========================================================================

    print("Running alternative outcome specifications...")

    alt_outcomes = [
        ('want_share_fb', 'Intent to share Alt-Facts on FB'),
        ('want_share_others', 'Intent to share Alt-Facts with others'),
    ]

    for alt_outcome, desc in alt_outcomes:
        control_str = ' + '.join(strata_controls + socio_controls)
        formula = f"{alt_outcome} ~ C(survey) + {control_str}"

        result = run_ols_robust(df_main, formula, outcome_var=alt_outcome, treatment_var='C(survey)[T.2]')
        if result:
            spec_count += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': f'robust/outcome/{alt_outcome}',
                'spec_tree_path': 'robustness/measurement.md',
                'outcome_var': alt_outcome,
                'treatment_var': 'survey (Imposed Fact-Check)',
                'sample_desc': 'Main experiment, surveys 1-3',
                'fixed_effects': 'None',
                'controls_desc': 'Strata + socioeconomic controls',
                'cluster_var': 'None (robust SE)',
                'model_type': 'OLS',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
                **result
            })

    # ===========================================================================
    # ALTERNATIVE TREATMENT COMPARISONS
    # ===========================================================================

    print("Running alternative treatment comparisons...")

    # Survey 3 (Voluntary Fact-Check) vs Survey 1 (Alt-Facts only)
    control_str = ' + '.join(strata_controls + socio_controls)
    formula = f"{outcome} ~ C(survey) + {control_str}"
    result = run_ols_robust(df_main, formula, outcome_var=outcome, treatment_var='C(survey)[T.3]')
    if result:
        spec_count += 1
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/treatment/voluntary_fc_vs_control',
            'spec_tree_path': 'methods/cross_sectional_ols.md',
            'outcome_var': outcome,
            'treatment_var': 'survey (Voluntary Fact-Check vs Alt-Facts)',
            'sample_desc': 'Main experiment, surveys 1-3',
            'fixed_effects': 'None',
            'controls_desc': 'Strata + socioeconomic controls',
            'cluster_var': 'None (robust SE)',
            'model_type': 'OLS',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
            **result
        })

    # Survey 2 vs Survey 3 only (Imposed vs Voluntary)
    df_23 = df_main[df_main['survey'].isin([2, 3])].copy()
    df_23['survey2_vs_3'] = (df_23['survey'] == 2).astype(int)
    formula = f"{outcome} ~ survey2_vs_3 + {control_str}"
    result = run_ols_robust(df_23, formula, outcome_var=outcome, treatment_var='survey2_vs_3')
    if result:
        spec_count += 1
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/treatment/imposed_vs_voluntary',
            'spec_tree_path': 'methods/cross_sectional_ols.md',
            'outcome_var': outcome,
            'treatment_var': 'Imposed vs Voluntary Fact-Check',
            'sample_desc': 'Surveys 2-3 only',
            'fixed_effects': 'None',
            'controls_desc': 'Strata + socioeconomic controls',
            'cluster_var': 'None (robust SE)',
            'model_type': 'OLS',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
            **result
        })

    # ===========================================================================
    # SAMPLE RESTRICTIONS
    # ===========================================================================

    print("Running sample restriction specifications...")

    # Gender subsamples
    df_male = df_main[df_main['male'] == 1].copy()
    result = run_ols_robust(df_male, f"{outcome} ~ C(survey) + {control_str}",
                            outcome_var=outcome, treatment_var='C(survey)[T.2]')
    if result:
        spec_count += 1
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/sample/male_only',
            'spec_tree_path': 'robustness/sample_restrictions.md',
            'outcome_var': outcome,
            'treatment_var': 'survey (Imposed Fact-Check)',
            'sample_desc': 'Male respondents only',
            'fixed_effects': 'None',
            'controls_desc': 'Strata + socioeconomic controls',
            'cluster_var': 'None (robust SE)',
            'model_type': 'OLS',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
            **result
        })

    df_female = df_main[df_main['male'] == 0].copy()
    result = run_ols_robust(df_female, f"{outcome} ~ C(survey) + {control_str}",
                            outcome_var=outcome, treatment_var='C(survey)[T.2]')
    if result:
        spec_count += 1
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/sample/female_only',
            'spec_tree_path': 'robustness/sample_restrictions.md',
            'outcome_var': outcome,
            'treatment_var': 'survey (Imposed Fact-Check)',
            'sample_desc': 'Female respondents only',
            'fixed_effects': 'None',
            'controls_desc': 'Strata + socioeconomic controls',
            'cluster_var': 'None (robust SE)',
            'model_type': 'OLS',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
            **result
        })

    # Age subsamples
    df_young = df_main[df_main['young'] == 1].copy()
    result = run_ols_robust(df_young, f"{outcome} ~ C(survey) + {control_str}",
                            outcome_var=outcome, treatment_var='C(survey)[T.2]')
    if result:
        spec_count += 1
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/sample/young',
            'spec_tree_path': 'robustness/sample_restrictions.md',
            'outcome_var': outcome,
            'treatment_var': 'survey (Imposed Fact-Check)',
            'sample_desc': 'Young respondents (age <= 35)',
            'fixed_effects': 'None',
            'controls_desc': 'Strata + socioeconomic controls',
            'cluster_var': 'None (robust SE)',
            'model_type': 'OLS',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
            **result
        })

    df_old = df_main[df_main['old'] == 1].copy()
    result = run_ols_robust(df_old, f"{outcome} ~ C(survey) + {control_str}",
                            outcome_var=outcome, treatment_var='C(survey)[T.2]')
    if result:
        spec_count += 1
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/sample/old',
            'spec_tree_path': 'robustness/sample_restrictions.md',
            'outcome_var': outcome,
            'treatment_var': 'survey (Imposed Fact-Check)',
            'sample_desc': 'Old respondents (age >= 65)',
            'fixed_effects': 'None',
            'controls_desc': 'Strata + socioeconomic controls',
            'cluster_var': 'None (robust SE)',
            'model_type': 'OLS',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
            **result
        })

    df_middle = df_main[df_main['mid_age'] == 1].copy()
    result = run_ols_robust(df_middle, f"{outcome} ~ C(survey) + {control_str}",
                            outcome_var=outcome, treatment_var='C(survey)[T.2]')
    if result:
        spec_count += 1
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/sample/middle_aged',
            'spec_tree_path': 'robustness/sample_restrictions.md',
            'outcome_var': outcome,
            'treatment_var': 'survey (Imposed Fact-Check)',
            'sample_desc': 'Middle-aged respondents (35 < age < 65)',
            'fixed_effects': 'None',
            'controls_desc': 'Strata + socioeconomic controls',
            'cluster_var': 'None (robust SE)',
            'model_type': 'OLS',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
            **result
        })

    # Education subsamples
    df_lowedu = df_main[df_main['low_educ'] == 1].copy()
    result = run_ols_robust(df_lowedu, f"{outcome} ~ C(survey) + {control_str}",
                            outcome_var=outcome, treatment_var='C(survey)[T.2]')
    if result:
        spec_count += 1
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/sample/low_education',
            'spec_tree_path': 'robustness/sample_restrictions.md',
            'outcome_var': outcome,
            'treatment_var': 'survey (Imposed Fact-Check)',
            'sample_desc': 'Low education respondents',
            'fixed_effects': 'None',
            'controls_desc': 'Strata + socioeconomic controls',
            'cluster_var': 'None (robust SE)',
            'model_type': 'OLS',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
            **result
        })

    df_highedu = df_main[df_main['high_educ'] == 1].copy()
    result = run_ols_robust(df_highedu, f"{outcome} ~ C(survey) + {control_str}",
                            outcome_var=outcome, treatment_var='C(survey)[T.2]')
    if result:
        spec_count += 1
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/sample/high_education',
            'spec_tree_path': 'robustness/sample_restrictions.md',
            'outcome_var': outcome,
            'treatment_var': 'survey (Imposed Fact-Check)',
            'sample_desc': 'High education respondents',
            'fixed_effects': 'None',
            'controls_desc': 'Strata + socioeconomic controls',
            'cluster_var': 'None (robust SE)',
            'model_type': 'OLS',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
            **result
        })

    # Location subsamples
    df_city = df_main[df_main['city'] == 1].copy()
    result = run_ols_robust(df_city, f"{outcome} ~ C(survey) + {control_str}",
                            outcome_var=outcome, treatment_var='C(survey)[T.2]')
    if result:
        spec_count += 1
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/sample/city_only',
            'spec_tree_path': 'robustness/sample_restrictions.md',
            'outcome_var': outcome,
            'treatment_var': 'survey (Imposed Fact-Check)',
            'sample_desc': 'City residents only',
            'fixed_effects': 'None',
            'controls_desc': 'Strata + socioeconomic controls',
            'cluster_var': 'None (robust SE)',
            'model_type': 'OLS',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
            **result
        })

    df_rural = df_main[(df_main['village'] == 1) | (df_main['town'] == 1)].copy()
    result = run_ols_robust(df_rural, f"{outcome} ~ C(survey) + {control_str}",
                            outcome_var=outcome, treatment_var='C(survey)[T.2]')
    if result:
        spec_count += 1
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/sample/rural_only',
            'spec_tree_path': 'robustness/sample_restrictions.md',
            'outcome_var': outcome,
            'treatment_var': 'survey (Imposed Fact-Check)',
            'sample_desc': 'Rural/town residents only',
            'fixed_effects': 'None',
            'controls_desc': 'Strata + socioeconomic controls',
            'cluster_var': 'None (robust SE)',
            'model_type': 'OLS',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
            **result
        })

    # Political subsamples
    df_mlp = df_main[df_main['second_mlp'] == 1].copy()
    result = run_ols_robust(df_mlp, f"{outcome} ~ C(survey) + {control_str}",
                            outcome_var=outcome, treatment_var='C(survey)[T.2]')
    if result:
        spec_count += 1
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/sample/le_pen_voters',
            'spec_tree_path': 'robustness/sample_restrictions.md',
            'outcome_var': outcome,
            'treatment_var': 'survey (Imposed Fact-Check)',
            'sample_desc': 'Le Pen voters only',
            'fixed_effects': 'None',
            'controls_desc': 'Strata + socioeconomic controls',
            'cluster_var': 'None (robust SE)',
            'model_type': 'OLS',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
            **result
        })

    df_not_mlp = df_main[df_main['second_mlp'] == 0].copy()
    result = run_ols_robust(df_not_mlp, f"{outcome} ~ C(survey) + {control_str}",
                            outcome_var=outcome, treatment_var='C(survey)[T.2]')
    if result:
        spec_count += 1
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/sample/non_le_pen_voters',
            'spec_tree_path': 'robustness/sample_restrictions.md',
            'outcome_var': outcome,
            'treatment_var': 'survey (Imposed Fact-Check)',
            'sample_desc': 'Non-Le Pen voters',
            'fixed_effects': 'None',
            'controls_desc': 'Strata + socioeconomic controls',
            'cluster_var': 'None (robust SE)',
            'model_type': 'OLS',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
            **result
        })

    # Religious vs non-religious
    df_religious = df_main[df_main['religious'] == 1].copy()
    result = run_ols_robust(df_religious, f"{outcome} ~ C(survey) + {control_str}",
                            outcome_var=outcome, treatment_var='C(survey)[T.2]')
    if result:
        spec_count += 1
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/sample/religious',
            'spec_tree_path': 'robustness/sample_restrictions.md',
            'outcome_var': outcome,
            'treatment_var': 'survey (Imposed Fact-Check)',
            'sample_desc': 'Religious respondents',
            'fixed_effects': 'None',
            'controls_desc': 'Strata + socioeconomic controls',
            'cluster_var': 'None (robust SE)',
            'model_type': 'OLS',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
            **result
        })

    df_not_religious = df_main[df_main['religious'] == 0].copy()
    result = run_ols_robust(df_not_religious, f"{outcome} ~ C(survey) + {control_str}",
                            outcome_var=outcome, treatment_var='C(survey)[T.2]')
    if result:
        spec_count += 1
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/sample/not_religious',
            'spec_tree_path': 'robustness/sample_restrictions.md',
            'outcome_var': outcome,
            'treatment_var': 'survey (Imposed Fact-Check)',
            'sample_desc': 'Non-religious respondents',
            'fixed_effects': 'None',
            'controls_desc': 'Strata + socioeconomic controls',
            'cluster_var': 'None (robust SE)',
            'model_type': 'OLS',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
            **result
        })

    # ===========================================================================
    # HETEROGENEITY ANALYSIS (Interactions)
    # ===========================================================================

    print("Running heterogeneity specifications...")

    # Gender interaction
    formula = f"{outcome} ~ C(survey) * male + {control_str}"
    result = run_ols_robust(df_main, formula, outcome_var=outcome, treatment_var='C(survey)[T.2]:male')
    if result:
        spec_count += 1
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/het/interaction_gender',
            'spec_tree_path': 'robustness/heterogeneity.md',
            'outcome_var': outcome,
            'treatment_var': 'survey x male interaction',
            'sample_desc': 'Main experiment, surveys 1-3',
            'fixed_effects': 'None',
            'controls_desc': 'Strata + socioeconomic controls',
            'cluster_var': 'None (robust SE)',
            'model_type': 'OLS',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
            **result
        })

    # Age interaction (young)
    formula = f"{outcome} ~ C(survey) * young + {control_str}"
    result = run_ols_robust(df_main, formula, outcome_var=outcome, treatment_var='C(survey)[T.2]:young')
    if result:
        spec_count += 1
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/het/interaction_age_young',
            'spec_tree_path': 'robustness/heterogeneity.md',
            'outcome_var': outcome,
            'treatment_var': 'survey x young interaction',
            'sample_desc': 'Main experiment, surveys 1-3',
            'fixed_effects': 'None',
            'controls_desc': 'Strata + socioeconomic controls',
            'cluster_var': 'None (robust SE)',
            'model_type': 'OLS',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
            **result
        })

    # Education interaction
    formula = f"{outcome} ~ C(survey) * high_educ + {control_str}"
    result = run_ols_robust(df_main, formula, outcome_var=outcome, treatment_var='C(survey)[T.2]:high_educ')
    if result:
        spec_count += 1
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/het/interaction_education',
            'spec_tree_path': 'robustness/heterogeneity.md',
            'outcome_var': outcome,
            'treatment_var': 'survey x high_educ interaction',
            'sample_desc': 'Main experiment, surveys 1-3',
            'fixed_effects': 'None',
            'controls_desc': 'Strata + socioeconomic controls',
            'cluster_var': 'None (robust SE)',
            'model_type': 'OLS',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
            **result
        })

    # Le Pen voter interaction
    formula = f"{outcome} ~ C(survey) * second_mlp + {control_str}"
    result = run_ols_robust(df_main, formula, outcome_var=outcome, treatment_var='C(survey)[T.2]:second_mlp')
    if result:
        spec_count += 1
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/het/interaction_le_pen',
            'spec_tree_path': 'robustness/heterogeneity.md',
            'outcome_var': outcome,
            'treatment_var': 'survey x Le Pen voter interaction',
            'sample_desc': 'Main experiment, surveys 1-3',
            'fixed_effects': 'None',
            'controls_desc': 'Strata + socioeconomic controls',
            'cluster_var': 'None (robust SE)',
            'model_type': 'OLS',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
            **result
        })

    # FB usage interaction
    formula = f"{outcome} ~ C(survey) * use_FB + {control_str}"
    result = run_ols_robust(df_main, formula, outcome_var=outcome, treatment_var='C(survey)[T.2]:use_FB')
    if result:
        spec_count += 1
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/het/interaction_fb_usage',
            'spec_tree_path': 'robustness/heterogeneity.md',
            'outcome_var': outcome,
            'treatment_var': 'survey x FB usage interaction',
            'sample_desc': 'Main experiment, surveys 1-3',
            'fixed_effects': 'None',
            'controls_desc': 'Strata + socioeconomic controls',
            'cluster_var': 'None (robust SE)',
            'model_type': 'OLS',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
            **result
        })

    # Religious interaction
    formula = f"{outcome} ~ C(survey) * religious + {control_str}"
    result = run_ols_robust(df_main, formula, outcome_var=outcome, treatment_var='C(survey)[T.2]:religious')
    if result:
        spec_count += 1
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/het/interaction_religious',
            'spec_tree_path': 'robustness/heterogeneity.md',
            'outcome_var': outcome,
            'treatment_var': 'survey x religious interaction',
            'sample_desc': 'Main experiment, surveys 1-3',
            'fixed_effects': 'None',
            'controls_desc': 'Strata + socioeconomic controls',
            'cluster_var': 'None (robust SE)',
            'model_type': 'OLS',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
            **result
        })

    # City interaction
    formula = f"{outcome} ~ C(survey) * city + {control_str}"
    result = run_ols_robust(df_main, formula, outcome_var=outcome, treatment_var='C(survey)[T.2]:city')
    if result:
        spec_count += 1
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/het/interaction_city',
            'spec_tree_path': 'robustness/heterogeneity.md',
            'outcome_var': outcome,
            'treatment_var': 'survey x city interaction',
            'sample_desc': 'Main experiment, surveys 1-3',
            'fixed_effects': 'None',
            'controls_desc': 'Strata + socioeconomic controls',
            'cluster_var': 'None (robust SE)',
            'model_type': 'OLS',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
            **result
        })

    # ===========================================================================
    # INFERENCE VARIATIONS
    # ===========================================================================

    print("Running inference variations...")

    # Classical (homoskedastic) SE
    try:
        df_clean = df_main.dropna(subset=[outcome]).copy()
        model = smf.ols(f"{outcome} ~ C(survey) + {control_str}", data=df_clean).fit()
        if 'C(survey)[T.2]' in model.params.index:
            coef = model.params['C(survey)[T.2]']
            se = model.bse['C(survey)[T.2]']
            tstat = model.tvalues['C(survey)[T.2]']
            pval = model.pvalues['C(survey)[T.2]']
            ci = model.conf_int().loc['C(survey)[T.2]']

            spec_count += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': 'robust/inference/classical_se',
                'spec_tree_path': 'robustness/inference_alternatives.md',
                'outcome_var': outcome,
                'treatment_var': 'survey (Imposed Fact-Check)',
                'sample_desc': 'Main experiment, surveys 1-3',
                'fixed_effects': 'None',
                'controls_desc': 'Strata + socioeconomic controls',
                'cluster_var': 'None (classical SE)',
                'model_type': 'OLS',
                'coefficient': float(coef),
                'std_error': float(se),
                't_stat': float(tstat),
                'p_value': float(pval),
                'ci_lower': float(ci[0]),
                'ci_upper': float(ci[1]),
                'n_obs': int(model.nobs),
                'r_squared': float(model.rsquared),
                'coefficient_vector_json': json.dumps({"treatment": {"coef": float(coef), "se": float(se)}}),
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })
    except Exception as e:
        print(f"Error in classical SE: {e}")

    # HC2 robust SE
    try:
        model = smf.ols(f"{outcome} ~ C(survey) + {control_str}", data=df_clean).fit(cov_type='HC2')
        if 'C(survey)[T.2]' in model.params.index:
            coef = model.params['C(survey)[T.2]']
            se = model.bse['C(survey)[T.2]']
            tstat = model.tvalues['C(survey)[T.2]']
            pval = model.pvalues['C(survey)[T.2]']
            ci = model.conf_int().loc['C(survey)[T.2]']

            spec_count += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': 'robust/inference/hc2_se',
                'spec_tree_path': 'robustness/inference_alternatives.md',
                'outcome_var': outcome,
                'treatment_var': 'survey (Imposed Fact-Check)',
                'sample_desc': 'Main experiment, surveys 1-3',
                'fixed_effects': 'None',
                'controls_desc': 'Strata + socioeconomic controls',
                'cluster_var': 'None (HC2 robust SE)',
                'model_type': 'OLS',
                'coefficient': float(coef),
                'std_error': float(se),
                't_stat': float(tstat),
                'p_value': float(pval),
                'ci_lower': float(ci[0]),
                'ci_upper': float(ci[1]),
                'n_obs': int(model.nobs),
                'r_squared': float(model.rsquared),
                'coefficient_vector_json': json.dumps({"treatment": {"coef": float(coef), "se": float(se)}}),
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })
    except Exception as e:
        print(f"Error in HC2 SE: {e}")

    # HC3 robust SE
    try:
        model = smf.ols(f"{outcome} ~ C(survey) + {control_str}", data=df_clean).fit(cov_type='HC3')
        if 'C(survey)[T.2]' in model.params.index:
            coef = model.params['C(survey)[T.2]']
            se = model.bse['C(survey)[T.2]']
            tstat = model.tvalues['C(survey)[T.2]']
            pval = model.pvalues['C(survey)[T.2]']
            ci = model.conf_int().loc['C(survey)[T.2]']

            spec_count += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': 'robust/inference/hc3_se',
                'spec_tree_path': 'robustness/inference_alternatives.md',
                'outcome_var': outcome,
                'treatment_var': 'survey (Imposed Fact-Check)',
                'sample_desc': 'Main experiment, surveys 1-3',
                'fixed_effects': 'None',
                'controls_desc': 'Strata + socioeconomic controls',
                'cluster_var': 'None (HC3 robust SE)',
                'model_type': 'OLS',
                'coefficient': float(coef),
                'std_error': float(se),
                't_stat': float(tstat),
                'p_value': float(pval),
                'ci_lower': float(ci[0]),
                'ci_upper': float(ci[1]),
                'n_obs': int(model.nobs),
                'r_squared': float(model.rsquared),
                'coefficient_vector_json': json.dumps({"treatment": {"coef": float(coef), "se": float(se)}}),
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })
    except Exception as e:
        print(f"Error in HC3 SE: {e}")

    # ===========================================================================
    # ESTIMATION METHOD VARIATIONS
    # ===========================================================================

    print("Running estimation method variations...")

    # Logit model (for binary outcome)
    result = run_logit(df_main, f"{outcome} ~ C(survey) + {control_str}",
                       outcome_var=outcome, treatment_var='C(survey)[T.2]')
    if result:
        spec_count += 1
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/estimation/logit',
            'spec_tree_path': 'methods/discrete_choice.md',
            'outcome_var': outcome,
            'treatment_var': 'survey (Imposed Fact-Check)',
            'sample_desc': 'Main experiment, surveys 1-3',
            'fixed_effects': 'None',
            'controls_desc': 'Strata + socioeconomic controls',
            'cluster_var': 'None (robust SE)',
            'model_type': 'Logit',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
            **result
        })

    # Probit model
    try:
        df_clean = df_main.dropna(subset=[outcome]).copy()
        model = smf.probit(f"{outcome} ~ C(survey) + {control_str}", data=df_clean).fit(cov_type='HC1', disp=False)
        if 'C(survey)[T.2]' in model.params.index:
            coef = model.params['C(survey)[T.2]']
            se = model.bse['C(survey)[T.2]']
            tstat = model.tvalues['C(survey)[T.2]']
            pval = model.pvalues['C(survey)[T.2]']
            ci = model.conf_int().loc['C(survey)[T.2]']

            spec_count += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': 'robust/estimation/probit',
                'spec_tree_path': 'methods/discrete_choice.md',
                'outcome_var': outcome,
                'treatment_var': 'survey (Imposed Fact-Check)',
                'sample_desc': 'Main experiment, surveys 1-3',
                'fixed_effects': 'None',
                'controls_desc': 'Strata + socioeconomic controls',
                'cluster_var': 'None (robust SE)',
                'model_type': 'Probit',
                'coefficient': float(coef),
                'std_error': float(se),
                't_stat': float(tstat),
                'p_value': float(pval),
                'ci_lower': float(ci[0]),
                'ci_upper': float(ci[1]),
                'n_obs': int(model.nobs),
                'r_squared': float(model.prsquared),
                'coefficient_vector_json': json.dumps({"treatment": {"coef": float(coef), "se": float(se)}}),
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })
    except Exception as e:
        print(f"Error in probit: {e}")

    # ===========================================================================
    # PLACEBO/FALSIFICATION TESTS
    # ===========================================================================

    print("Running placebo specifications...")

    # Placebo outcome: No effect expected on altruism (pre-treatment characteristic)
    placebo_outcomes = ['altruism', 'reciprocity', 'image']
    for placebo_outcome in placebo_outcomes:
        if placebo_outcome in df_main.columns:
            formula = f"{placebo_outcome} ~ C(survey) + male + age + age_sqr"
            result = run_ols_robust(df_main, formula, outcome_var=placebo_outcome,
                                    treatment_var='C(survey)[T.2]')
            if result:
                spec_count += 1
                results.append({
                    'paper_id': PAPER_ID,
                    'journal': JOURNAL,
                    'paper_title': PAPER_TITLE,
                    'spec_id': f'robust/placebo/{placebo_outcome}',
                    'spec_tree_path': 'robustness/placebo_tests.md',
                    'outcome_var': placebo_outcome,
                    'treatment_var': 'survey (Imposed Fact-Check)',
                    'sample_desc': 'Main experiment, surveys 1-3',
                    'fixed_effects': 'None',
                    'controls_desc': 'Basic demographics only',
                    'cluster_var': 'None (robust SE)',
                    'model_type': 'OLS',
                    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
                    **result
                })

    # ===========================================================================
    # ADDITIONAL SPECIFICATIONS - Voluntary Fact-Check (Survey 3) treatment
    # ===========================================================================

    print("Running additional treatment effect specifications...")

    # Repeat key specs for Survey 3 treatment
    for spec_suffix, controls, desc in [
        ('bivariate', [], 'No controls'),
        ('strata', strata_controls, 'Stratification controls'),
        ('full', all_controls, 'All controls')
    ]:
        if controls:
            control_str = ' + '.join(controls)
            formula = f"{outcome} ~ C(survey) + {control_str}"
        else:
            formula = f"{outcome} ~ C(survey)"

        result = run_ols_robust(df_main, formula, outcome_var=outcome, treatment_var='C(survey)[T.3]')
        if result:
            spec_count += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': f'robust/treatment/voluntary_fc_{spec_suffix}',
                'spec_tree_path': 'methods/cross_sectional_ols.md',
                'outcome_var': outcome,
                'treatment_var': 'survey (Voluntary Fact-Check vs Control)',
                'sample_desc': 'Main experiment, surveys 1-3',
                'fixed_effects': 'None',
                'controls_desc': desc,
                'cluster_var': 'None (robust SE)',
                'model_type': 'OLS',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
                **result
            })

    print(f"\nTotal specifications run: {spec_count}")
    return results

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(f"Specification Search: {PAPER_ID}")
    print("=" * 70)

    # Load data
    print("\nLoading and preparing data...")
    df = load_and_prepare_data()
    print(f"Loaded {len(df)} observations")
    print(f"Surveys: {df['survey'].value_counts().to_dict()}")

    # Run specification search
    print("\nRunning specification search...")
    results = run_specification_search(df)

    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)

    # Save results
    output_file = f"{OUTPUT_PATH}/specification_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    n_total = len(results_df)
    n_positive = (results_df['coefficient'] > 0).sum()
    n_sig_05 = (results_df['p_value'] < 0.05).sum()
    n_sig_01 = (results_df['p_value'] < 0.01).sum()

    print(f"Total specifications: {n_total}")
    print(f"Positive coefficients: {n_positive} ({100*n_positive/n_total:.1f}%)")
    print(f"Significant at 5%: {n_sig_05} ({100*n_sig_05/n_total:.1f}%)")
    print(f"Significant at 1%: {n_sig_01} ({100*n_sig_01/n_total:.1f}%)")
    print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
    print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")
    print(f"Range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")

    # Breakdown by category
    print("\n" + "-" * 70)
    print("BREAKDOWN BY CATEGORY")
    print("-" * 70)

    def categorize_spec(spec_id):
        if spec_id == 'baseline':
            return 'Baseline'
        elif 'build' in spec_id:
            return 'Control progression'
        elif 'loo' in spec_id:
            return 'Leave-one-out'
        elif 'outcome' in spec_id:
            return 'Alternative outcomes'
        elif 'treatment' in spec_id:
            return 'Alternative treatments'
        elif 'sample' in spec_id:
            return 'Sample restrictions'
        elif 'het' in spec_id:
            return 'Heterogeneity'
        elif 'inference' in spec_id:
            return 'Inference variations'
        elif 'estimation' in spec_id:
            return 'Estimation method'
        elif 'placebo' in spec_id:
            return 'Placebo tests'
        else:
            return 'Other'

    results_df['category'] = results_df['spec_id'].apply(categorize_spec)

    for cat, group in results_df.groupby('category'):
        n = len(group)
        n_pos = (group['coefficient'] > 0).sum()
        n_sig = (group['p_value'] < 0.05).sum()
        print(f"{cat}: {n} specs, {100*n_pos/n:.0f}% positive, {100*n_sig/n:.0f}% sig at 5%")

    print("\n" + "=" * 70)
    print("Specification search complete!")
    print("=" * 70)
