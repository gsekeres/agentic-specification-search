"""
Specification Search Analysis for Paper 202864-V1

Title: Eliciting ambiguity with mixing bets
Journal: AEJ: Micro
Author: Patrick Schmidt

This script runs systematic specification variations on the main regression analysis
from the paper, which examines how mixing behavior varies across domains (risk, ambiguity,
stock, social) and individual characteristics.

Main hypothesis: Ambiguity preferences (mixing behavior) differ across domains and
are associated with individual characteristics like risk aversion, gender, and experience.
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

# ============================================================================
# CONFIGURATION
# ============================================================================

PAPER_ID = "202864-V1"
JOURNAL = "AEJ: Micro"
PAPER_TITLE = "Eliciting ambiguity with mixing bets"

BASE_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search"
DATA_PATH = f"{BASE_PATH}/data/downloads/extracted/202864-V1/RProject/data/df.all.RDS"
OUTPUT_PATH = f"{BASE_PATH}/data/downloads/extracted/202864-V1/specification_results.csv"

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def load_and_prepare_data():
    """Load and prepare data following the paper's gen_data.R and output.R logic"""

    # Load RDS file
    result = pyreadr.read_r(DATA_PATH)
    df = result[None]

    # Rename topics from German to English (following output.R rename_topics function)
    topic_map = {
        'blauer Ball aus Urne': 'risk',
        'DAX': 'stock',
        'gepunkteter Ball aus Urne': 'ambiguous',
        'Alleingang oder Zusammenarbeit': 'social',
        'gepunkteter Ball nach 10 Ziehungen': 'updated'
    }
    df['topic'] = df['topic'].map(topic_map)

    # Rename key columns
    df = df.rename(columns={
        'participant.code': 'person',
        'player.elicit_type': 'type',
        'demographics.1.player.gender': 'gender'
    })

    # Create female indicator
    df['female'] = (df['gender'] == 'weiblich').astype(int)

    # Create field of study dummies
    df['Fmint'] = df['demographics.1.player.fieldofstudy'].isin(
        ['Mathematik', 'Physik', 'Informatik', 'Biochemie, Chemie, Pharmazie']).astype(int)
    df['Fsocial'] = df['demographics.1.player.fieldofstudy'].isin(
        ['Sprach-und Kulturwissenschaften', 'Erziehungswissenschaften', 'Psychologie']).astype(int)
    df['Fecon'] = (df['demographics.1.player.fieldofstudy'] == 'Wirtschaftswissenschaften').astype(int)
    df['Flaw'] = (df['demographics.1.player.fieldofstudy'] == 'Rechtswissenschaft').astype(int)

    # Create comprehension indicator
    df['comprehension'] = (df['demographics.1.player.comprehension'] == 'Ja').astype(int)

    # Create experience indicators
    df['exp1'] = df['demographics.1.player.probabilityexp1']
    df['exp2'] = df['demographics.1.player.probabilityexp2']
    df['exp3'] = (df['demographics.1.player.probabilityexp3'] == 'Ja').astype(int)
    df['exp4'] = (df['demographics.1.player.probabilityexp4'] == 500).astype(int)
    df['exp5'] = ((df['demographics.1.player.probabilityexp5'] == 25) |
                  (df['demographics.1.player.probabilityexp5'] == 0.25)).astype(int)

    # Create cooperate indicator (was boolean/string)
    df['cooperate_num'] = (df['cooperate'] == True).astype(int)

    # Standardize function
    def mystan(x):
        return (x - x.mean()) / x.std()

    # Standardize experience variables and create composite
    df['exp1_s'] = mystan(df['exp1'].fillna(df['exp1'].mean()))
    df['exp2_s'] = mystan(df['exp2'].fillna(df['exp2'].mean()))
    df['exp3_s'] = mystan(df['exp3'].fillna(df['exp3'].mean()))
    df['exp4_s'] = mystan(df['exp4'].fillna(df['exp4'].mean()))
    df['exp5_s'] = mystan(df['exp5'].fillna(df['exp5'].mean()))
    df['experienceall'] = df['exp1_s'] + df['exp2_s'] + df['exp3_s'] + df['exp4_s'] + df['exp5_s']

    return df


def create_analysis_datasets(df):
    """Create person-level and person-topic-level analysis datasets"""

    # Person x Topic level: compute mixing intensity (att) for slider type
    # Following output.R lines 316-322
    df_att = df[df['type'] == 'slider'].copy()
    df_att['diff'] = np.where(
        df_att['x'] >= 1 - df_att['q'],
        (1 - df_att['x']) / df_att['q'],
        df_att['x'] / (1 - df_att['q'])
    )

    # Keep only mixing choices (x not 0 or 1) and exclude 'updated' topic
    df_att = df_att[(df_att['x'] != 0) & (df_att['x'] != 1) & (df_att['topic'] != 'updated')]

    # Aggregate to person-topic level
    att_person_topic = df_att.groupby(['person', 'topic'])['diff'].mean().reset_index()
    att_person_topic = att_person_topic.rename(columns={'diff': 'att'})

    # Person-level mixing intensity (average across topics)
    att_person = df_att.groupby('person')['diff'].mean().reset_index()
    att_person = att_person.rename(columns={'diff': 'att'})

    # Person x Topic x Discrete: compute l3 (number of distinct q with mixing choices)
    # Following output.R lines 361-372
    df_disc = df[(df['type'] == 'choice_twice') & (df['topic'] != 'updated')].copy()
    df_disc['is_mixing'] = ~df_disc['x'].isin([0, 1])

    l_person_topic = df_disc.groupby(['person', 'topic']).agg(
        l3=('q', lambda x: x[df_disc.loc[x.index, 'is_mixing']].nunique()),
        qE=('q', lambda x: x[df_disc.loc[x.index, 'x'] != 0].min() if any(df_disc.loc[x.index, 'x'] != 0) else 1),
        qC=('q', lambda x: x[df_disc.loc[x.index, 'x'] != 1].max() if any(df_disc.loc[x.index, 'x'] != 1) else 0)
    ).reset_index()

    # Compute multiple mixing indicator
    l_person_topic['multiple_mixing'] = (l_person_topic['l3'] > 1).astype(int)

    # Compute probability p
    # p = 1 - mean(q[mixing]) if any mixing, else 1 - mean(qC, qE)
    l_person_topic['p'] = 1 - (l_person_topic['qE'] + l_person_topic['qC']) / 2

    # Get person-level demographics
    person_demo = df.groupby('person').first()[
        ['age', 'risk_aversion', 'female', 'Fmint', 'Fsocial', 'Fecon', 'Flaw',
         'comprehension', 'experienceall', 'cooperate_num']
    ].reset_index()

    # Merge datasets
    analysis_df = l_person_topic.merge(att_person, on='person', how='left')
    analysis_df = analysis_df.merge(person_demo, on='person', how='left')

    # Standardize key variables
    def mystan(x):
        return (x - x.mean()) / x.std()

    analysis_df['risk_aversion_s'] = mystan(analysis_df['risk_aversion'].fillna(analysis_df['risk_aversion'].mean()))
    analysis_df['experienceall_s'] = mystan(analysis_df['experienceall'].fillna(analysis_df['experienceall'].mean()))
    analysis_df['att_s'] = mystan(analysis_df['att'].fillna(analysis_df['att'].mean()))
    analysis_df['l3_s'] = mystan(analysis_df['l3'].fillna(analysis_df['l3'].mean()))
    analysis_df['p_s'] = mystan(analysis_df['p'].fillna(analysis_df['p'].mean()))

    return analysis_df, att_person, person_demo


# ============================================================================
# REGRESSION HELPER FUNCTIONS
# ============================================================================

def run_regression(df, formula, outcome_var, treatment_var, spec_id, spec_tree_path,
                   cluster_var=None, model_type='OLS', sample_desc='Full sample',
                   controls_desc='', fixed_effects='None'):
    """Run a regression and return standardized results dictionary"""

    try:
        if cluster_var:
            model = smf.ols(formula, data=df).fit(
                cov_type='cluster',
                cov_kwds={'groups': df[cluster_var]}
            )
        else:
            model = smf.ols(formula, data=df).fit(cov_type='HC1')

        # Get treatment coefficient
        treat_coef = model.params[treatment_var]
        treat_se = model.bse[treatment_var]
        treat_tstat = model.tvalues[treatment_var]
        treat_pval = model.pvalues[treatment_var]

        # Confidence intervals
        conf_int = model.conf_int()
        ci_lower = conf_int.loc[treatment_var, 0]
        ci_upper = conf_int.loc[treatment_var, 1]

        # Build coefficient vector JSON
        coef_vector = {
            'treatment': {
                'var': treatment_var,
                'coef': float(treat_coef),
                'se': float(treat_se),
                'pval': float(treat_pval)
            },
            'controls': [],
            'fixed_effects': [fixed_effects] if fixed_effects != 'None' else [],
            'diagnostics': {
                'f_stat': float(model.fvalue) if model.fvalue else None,
                'f_pval': float(model.f_pvalue) if model.f_pvalue else None
            }
        }

        # Add control coefficients
        for var in model.params.index:
            if var != treatment_var and var != 'Intercept':
                coef_vector['controls'].append({
                    'var': var,
                    'coef': float(model.params[var]),
                    'se': float(model.bse[var]),
                    'pval': float(model.pvalues[var])
                })

        return {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': float(treat_coef),
            'std_error': float(treat_se),
            't_stat': float(treat_tstat),
            'p_value': float(treat_pval),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_obs': int(model.nobs),
            'r_squared': float(model.rsquared),
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects,
            'controls_desc': controls_desc,
            'cluster_var': cluster_var if cluster_var else 'None',
            'model_type': model_type,
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
    except Exception as e:
        print(f"Error in {spec_id}: {e}")
        return None


def run_ttest(df, topic1, topic2, outcome_var, spec_id, spec_tree_path):
    """Run t-test comparing two topics and return results dictionary"""

    try:
        group1 = df[df['topic'] == topic1][outcome_var].dropna()
        group2 = df[df['topic'] == topic2][outcome_var].dropna()

        # Run two-sample t-test
        t_result = stats.ttest_ind(group1, group2)
        t_stat = float(t_result.statistic)
        p_val = float(t_result.pvalue)

        # Compute difference (treatment effect)
        diff = group2.mean() - group1.mean()
        se = np.sqrt(group1.var()/len(group1) + group2.var()/len(group2))
        ci_lower = diff - 1.96 * se
        ci_upper = diff + 1.96 * se

        return {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': outcome_var,
            'treatment_var': f'{topic2}_vs_{topic1}',
            'coefficient': float(diff),
            'std_error': float(se),
            't_stat': t_stat,
            'p_value': p_val,
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_obs': len(group1) + len(group2),
            'r_squared': np.nan,
            'coefficient_vector_json': json.dumps({
                'treatment': {'var': f'{topic2}_vs_{topic1}', 'coef': float(diff), 'se': float(se), 'pval': p_val},
                'mean_group1': float(group1.mean()),
                'mean_group2': float(group2.mean()),
                'n_group1': len(group1),
                'n_group2': len(group2)
            }),
            'sample_desc': f'{topic1} vs {topic2}',
            'fixed_effects': 'None',
            'controls_desc': 'None',
            'cluster_var': 'None',
            'model_type': 't-test',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
    except Exception as e:
        print(f"Error in t-test {spec_id}: {e}")
        return None


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    """Run all specification variations"""

    print("Loading data...")
    df = load_and_prepare_data()
    analysis_df, att_person, person_demo = create_analysis_datasets(df)

    results = []

    # ========================================================================
    # SECTION 1: BASELINE - T-TESTS (Table 1 main paper)
    # Comparing mixing behavior (l) across domains
    # ========================================================================
    print("\nRunning baseline t-tests...")

    # Prepare discrete choice data for t-tests
    df_discrete = df[(df['type'] == 'choice_twice')].copy()
    df_discrete['is_mixing'] = ~df_discrete['x'].isin([0, 1])

    # Aggregate to person-topic level - count mixing q values
    dft = df_discrete.groupby(['person', 'topic']).agg(
        n_mixing_qs=('is_mixing', 'sum')
    ).reset_index()

    # l = True if multiple mixing choices (n_mixing_qs > 1)
    dft['l'] = (dft['n_mixing_qs'] > 1).astype(int)

    # T-tests comparing risk to other domains (main hypothesis tests)
    ttest_pairs = [
        ('risk', 'ambiguous'),
        ('risk', 'stock'),
        ('risk', 'social'),
        ('ambiguous', 'stock'),
        ('ambiguous', 'social')
    ]

    for topic1, topic2 in ttest_pairs:
        res = run_ttest(dft, topic1, topic2, 'l',
                       f'baseline/ttest/{topic2}_vs_{topic1}',
                       'methods/cross_sectional_ols.md#baseline')
        if res:
            results.append(res)

    # ========================================================================
    # SECTION 2: MAIN REGRESSION - Table 1 (columns 3-6)
    # Predicting multiple mixing (l3>1) from individual characteristics
    # ========================================================================
    print("\nRunning main regressions...")

    # Baseline specification for each domain
    # Following output.R lines 468-472
    baseline_controls = ['att_s', 'risk_aversion_s', 'female', 'experienceall_s']
    control_str = ' + '.join(baseline_controls)

    for topic in ['risk', 'ambiguous', 'stock', 'social']:
        df_topic = analysis_df[analysis_df['topic'] == topic].copy()

        # Baseline specification
        formula = f'multiple_mixing ~ {control_str}'
        res = run_regression(
            df_topic, formula, 'multiple_mixing', 'att_s',
            f'baseline/{topic}',
            'methods/cross_sectional_ols.md#baseline',
            sample_desc=f'{topic} domain',
            controls_desc='risk_aversion_s, female, experienceall_s'
        )
        if res:
            results.append(res)

    # ========================================================================
    # SECTION 3: OLS VARIATIONS
    # ========================================================================
    print("\nRunning OLS method variations...")

    # Focus on ambiguous domain as primary test
    df_ambig = analysis_df[analysis_df['topic'] == 'ambiguous'].copy()

    # 3a. Different standard error approaches
    # Classical SE
    model = smf.ols(f'multiple_mixing ~ {control_str}', data=df_ambig).fit()
    res = run_regression(
        df_ambig, f'multiple_mixing ~ {control_str}', 'multiple_mixing', 'att_s',
        'ols/se/classical', 'methods/cross_sectional_ols.md#standard-errors',
        sample_desc='ambiguous domain', controls_desc='Full controls'
    )
    if res:
        # Override with classical SE
        res['std_error'] = float(model.bse['att_s'])
        res['p_value'] = float(model.pvalues['att_s'])
        res['t_stat'] = float(model.tvalues['att_s'])
        results.append(res)

    # HC1 robust (already done in baseline)

    # HC2 robust
    model_hc2 = smf.ols(f'multiple_mixing ~ {control_str}', data=df_ambig).fit(cov_type='HC2')
    res = run_regression(
        df_ambig, f'multiple_mixing ~ {control_str}', 'multiple_mixing', 'att_s',
        'ols/se/hc2', 'methods/cross_sectional_ols.md#standard-errors',
        sample_desc='ambiguous domain', controls_desc='Full controls'
    )
    if res:
        res['std_error'] = float(model_hc2.bse['att_s'])
        res['p_value'] = float(model_hc2.pvalues['att_s'])
        results.append(res)

    # HC3 robust
    model_hc3 = smf.ols(f'multiple_mixing ~ {control_str}', data=df_ambig).fit(cov_type='HC3')
    res = run_regression(
        df_ambig, f'multiple_mixing ~ {control_str}', 'multiple_mixing', 'att_s',
        'ols/se/hc3', 'methods/cross_sectional_ols.md#standard-errors',
        sample_desc='ambiguous domain', controls_desc='Full controls'
    )
    if res:
        res['std_error'] = float(model_hc3.bse['att_s'])
        res['p_value'] = float(model_hc3.pvalues['att_s'])
        results.append(res)

    # 3b. Control set variations
    # No controls (bivariate)
    res = run_regression(
        df_ambig, 'multiple_mixing ~ att_s', 'multiple_mixing', 'att_s',
        'ols/controls/none', 'methods/cross_sectional_ols.md#control-sets',
        sample_desc='ambiguous domain', controls_desc='None (bivariate)'
    )
    if res:
        results.append(res)

    # Demographics only
    res = run_regression(
        df_ambig, 'multiple_mixing ~ att_s + female', 'multiple_mixing', 'att_s',
        'ols/controls/demographics', 'methods/cross_sectional_ols.md#control-sets',
        sample_desc='ambiguous domain', controls_desc='female only'
    )
    if res:
        results.append(res)

    # Full controls (add field of study)
    full_controls = 'att_s + risk_aversion_s + female + experienceall_s + Fmint + Fsocial + Fecon + Flaw'
    res = run_regression(
        df_ambig, f'multiple_mixing ~ {full_controls}', 'multiple_mixing', 'att_s',
        'ols/controls/full', 'methods/cross_sectional_ols.md#control-sets',
        sample_desc='ambiguous domain', controls_desc='Full controls + field of study'
    )
    if res:
        results.append(res)

    # ========================================================================
    # SECTION 4: ROBUSTNESS - LEAVE ONE OUT
    # ========================================================================
    print("\nRunning leave-one-out checks...")

    for drop_var in baseline_controls:
        remaining = [c for c in baseline_controls if c != drop_var]
        formula = f'multiple_mixing ~ {" + ".join(remaining)}'
        res = run_regression(
            df_ambig, formula, 'multiple_mixing', remaining[0],  # First remaining as treatment
            f'robust/loo/drop_{drop_var}', 'robustness/leave_one_out.md',
            sample_desc='ambiguous domain', controls_desc=f'Dropped {drop_var}'
        )
        if res:
            # We want att_s as treatment when it's still in model
            if 'att_s' in remaining:
                model = smf.ols(formula, data=df_ambig).fit(cov_type='HC1')
                res['treatment_var'] = 'att_s'
                res['coefficient'] = float(model.params['att_s'])
                res['std_error'] = float(model.bse['att_s'])
                res['t_stat'] = float(model.tvalues['att_s'])
                res['p_value'] = float(model.pvalues['att_s'])
            results.append(res)

    # ========================================================================
    # SECTION 5: ROBUSTNESS - SINGLE COVARIATE
    # ========================================================================
    print("\nRunning single covariate checks...")

    # Bivariate
    res = run_regression(
        df_ambig, 'multiple_mixing ~ att_s', 'multiple_mixing', 'att_s',
        'robust/single/none', 'robustness/single_covariate.md',
        sample_desc='ambiguous domain', controls_desc='Bivariate'
    )
    if res:
        results.append(res)

    for ctrl in ['risk_aversion_s', 'female', 'experienceall_s']:
        res = run_regression(
            df_ambig, f'multiple_mixing ~ att_s + {ctrl}', 'multiple_mixing', 'att_s',
            f'robust/single/{ctrl}', 'robustness/single_covariate.md',
            sample_desc='ambiguous domain', controls_desc=f'att_s + {ctrl}'
        )
        if res:
            results.append(res)

    # ========================================================================
    # SECTION 6: ROBUSTNESS - SAMPLE RESTRICTIONS
    # ========================================================================
    print("\nRunning sample restriction checks...")

    # Complete cases only
    df_complete = df_ambig.dropna(subset=baseline_controls + ['multiple_mixing'])
    res = run_regression(
        df_complete, f'multiple_mixing ~ {control_str}', 'multiple_mixing', 'att_s',
        'robust/sample/complete_cases', 'robustness/sample_restrictions.md',
        sample_desc='ambiguous domain, complete cases', controls_desc='Full controls'
    )
    if res:
        results.append(res)

    # Gender subgroups
    df_male = df_ambig[df_ambig['female'] == 0]
    res = run_regression(
        df_male, f'multiple_mixing ~ att_s + risk_aversion_s + experienceall_s',
        'multiple_mixing', 'att_s',
        'robust/sample/male_only', 'robustness/sample_restrictions.md',
        sample_desc='ambiguous domain, male only', controls_desc='Without gender'
    )
    if res:
        results.append(res)

    df_female = df_ambig[df_ambig['female'] == 1]
    res = run_regression(
        df_female, f'multiple_mixing ~ att_s + risk_aversion_s + experienceall_s',
        'multiple_mixing', 'att_s',
        'robust/sample/female_only', 'robustness/sample_restrictions.md',
        sample_desc='ambiguous domain, female only', controls_desc='Without gender'
    )
    if res:
        results.append(res)

    # Comprehension subgroup (only those who understood)
    df_understood = analysis_df[(analysis_df['topic'] == 'ambiguous') & (analysis_df['comprehension'] == 1)]
    res = run_regression(
        df_understood, f'multiple_mixing ~ {control_str}', 'multiple_mixing', 'att_s',
        'robust/sample/understood_only', 'robustness/sample_restrictions.md',
        sample_desc='ambiguous domain, understood only', controls_desc='Full controls'
    )
    if res:
        results.append(res)

    # ========================================================================
    # SECTION 7: ALTERNATIVE OUTCOMES
    # ========================================================================
    print("\nRunning alternative outcome specifications...")

    # Continuous mixing intensity as outcome
    df_with_att = analysis_df[(analysis_df['topic'] == 'ambiguous') & analysis_df['att'].notna()].copy()
    res = run_regression(
        df_with_att, f'att ~ risk_aversion_s + female + experienceall_s',
        'att', 'risk_aversion_s',
        'custom/outcome_att', 'custom',
        sample_desc='ambiguous domain', controls_desc='female, experienceall_s',
        model_type='OLS'
    )
    if res:
        results.append(res)

    # Probability p as outcome
    res = run_regression(
        df_ambig, f'p ~ att_s + risk_aversion_s + female + experienceall_s',
        'p', 'att_s',
        'custom/outcome_probability', 'custom',
        sample_desc='ambiguous domain', controls_desc='Full controls',
        model_type='OLS'
    )
    if res:
        results.append(res)

    # ========================================================================
    # SECTION 8: DOMAIN INTERACTIONS
    # ========================================================================
    print("\nRunning domain interaction specifications...")

    # Pool all domains with domain fixed effects
    analysis_df_pooled = analysis_df[analysis_df['topic'].isin(['risk', 'ambiguous', 'stock', 'social'])].copy()

    # Add domain dummies
    for topic in ['ambiguous', 'stock', 'social']:
        analysis_df_pooled[f'topic_{topic}'] = (analysis_df_pooled['topic'] == topic).astype(int)

    formula_pooled = 'multiple_mixing ~ att_s + risk_aversion_s + female + experienceall_s + topic_ambiguous + topic_stock + topic_social'
    res = run_regression(
        analysis_df_pooled, formula_pooled, 'multiple_mixing', 'att_s',
        'ols/fe/domain', 'methods/cross_sectional_ols.md#fixed-effects',
        sample_desc='All domains pooled', controls_desc='Full controls + domain FE',
        fixed_effects='domain'
    )
    if res:
        results.append(res)

    # Domain interactions with mixing intensity
    formula_interact = 'multiple_mixing ~ att_s * topic_ambiguous + att_s * topic_stock + att_s * topic_social + risk_aversion_s + female + experienceall_s'
    res = run_regression(
        analysis_df_pooled, formula_interact, 'multiple_mixing', 'att_s',
        'ols/interact/domain', 'methods/cross_sectional_ols.md#interaction-effects',
        sample_desc='All domains pooled', controls_desc='Full controls + domain interactions'
    )
    if res:
        results.append(res)

    # ========================================================================
    # SECTION 9: COOPERATION REGRESSION (Table 2 supplement)
    # ========================================================================
    print("\nRunning cooperation regression...")

    df_social = analysis_df[analysis_df['topic'] == 'social'].copy()
    df_social['cooperate'] = df_social['cooperate_num']
    df_social['l3_indicator'] = (df_social['l3'] > 1).astype(int)
    df_social['l3_indicator_s'] = (df_social['l3_indicator'] - df_social['l3_indicator'].mean()) / df_social['l3_indicator'].std()

    res = run_regression(
        df_social, 'cooperate ~ att_s + risk_aversion_s + l3_indicator_s + p_s',
        'cooperate', 'att_s',
        'baseline/cooperation', 'methods/cross_sectional_ols.md#baseline',
        sample_desc='social domain', controls_desc='risk_aversion_s, l3_s, p_s',
        model_type='OLS (LPM)'
    )
    if res:
        results.append(res)

    # ========================================================================
    # SECTION 10: CLUSTERING VARIATIONS
    # ========================================================================
    print("\nRunning clustering variations...")

    # Cluster by person (for pooled regressions)
    res = run_regression(
        analysis_df_pooled, formula_pooled, 'multiple_mixing', 'att_s',
        'robust/cluster/person', 'robustness/clustering_variations.md',
        cluster_var='person',
        sample_desc='All domains pooled', controls_desc='Full controls + domain FE'
    )
    if res:
        results.append(res)

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    print("\nSaving results...")

    # Filter out None results
    results = [r for r in results if r is not None]

    # Create DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved {len(results)} specifications to {OUTPUT_PATH}")

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total specifications: {len(results)}")
    print(f"Positive coefficients: {sum(results_df['coefficient'] > 0)} ({100*sum(results_df['coefficient'] > 0)/len(results):.1f}%)")
    print(f"Significant at 5%: {sum(results_df['p_value'] < 0.05)} ({100*sum(results_df['p_value'] < 0.05)/len(results):.1f}%)")
    print(f"Significant at 1%: {sum(results_df['p_value'] < 0.01)} ({100*sum(results_df['p_value'] < 0.01)/len(results):.1f}%)")

    # By spec category
    results_df['category'] = results_df['spec_id'].str.split('/').str[0]
    print("\nBy category:")
    for cat in results_df['category'].unique():
        cat_df = results_df[results_df['category'] == cat]
        pct_sig = 100 * sum(cat_df['p_value'] < 0.05) / len(cat_df)
        print(f"  {cat}: {len(cat_df)} specs, {pct_sig:.1f}% significant at 5%")

    return results_df


if __name__ == "__main__":
    results_df = main()
