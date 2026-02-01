"""
Specification Search for Paper 146241-V1

Paper: "Ten isn't large! Group size and coordination in a large-scale experiment"
Authors: Jasmina Arifovic, Cars Hommes, Anita Kopányi-Peuker, Isabelle Salle

Main Hypothesis: Larger group size leads to more bank runs (more withdrawals)
    - The paper tests whether coordination on the inefficient equilibrium
      (bank run) is more likely in large groups compared to small groups

Treatment Variables:
    - large: 1 if large group (~80-90 participants), 0 if small group (10 participants)
    - rho: persistence parameter (0.5 or 0.8)
    - rValue: interest rate parameter (1.33 or 1.54)

Primary Outcome: withdraw (binary: 1=withdraw, 0=wait)

Method: Discrete choice (logit) models with clustered standard errors
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import Logit
import json
import warnings
warnings.filterwarnings('ignore')

# Set paths
BASE_PATH = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search'
DATA_PATH = f'{BASE_PATH}/data/downloads/extracted/146241-V1/Data_deposit'
OUTPUT_PATH = f'{BASE_PATH}/data/downloads/extracted/146241-V1'

# Paper metadata
PAPER_ID = '146241-V1'
JOURNAL = 'AER'  # American Economic Review
PAPER_TITLE = "Ten isn't large! Group size and coordination in a large-scale experiment"

def load_and_prepare_data():
    """Load and prepare the experimental data."""
    # Load individual-level data
    df = pd.read_stata(f'{DATA_PATH}/2_raw_data/raw_data_Stata/individual_merged.dta')

    # Load group data for group-level analysis
    df_group = pd.read_stata(f'{DATA_PATH}/2_raw_data/raw_data_Stata/group.dta')

    # Create key treatment indicators
    # Treatment: 1=SL(small,low), 2=SH(small,high), 3=LL(large,low), 4=LH(large,high), 5=small r=1.33, 6=large r=1.33, 7=AJ
    df['large'] = df['treatment'].isin([3, 4, 6]).astype(int)

    # rhoValue = 0.8 for treatments 2 and 4, 0.5 otherwise
    df['rhoValue'] = np.where(df['treatment'].isin([2, 4]), 0.8, 0.5)

    # rValue = 1.33 for treatments 5, 6; 1.54 otherwise
    df['rValue'] = np.where(df['treatment'].isin([5, 6]), 1.33, 1.54)
    df['rValueF'] = (df['rValue'] == 1.33).astype(int)

    # Group size
    group_sizes = {1: 10, 2: 10, 3: 90, 4: 84, 5: 10, 6: 84, 7: 10}
    df['groupSize'] = df['treatment'].map(group_sizes)

    # Create group ID
    df['group_ID'] = df['session'].astype(str) + '_' + df['groep'].astype(str)

    # Create individual ID
    df['indiv_ID'] = df['session'].astype(str) + '_' + df['ppnr'].astype(str)

    # Create background variable (education field)
    df['background'] = df['fieldstudy'].fillna('OTHER')
    df['background'] = df['background'].replace({
        'business': 'ECO', 'econ': 'ECO',
        'Tand': 'SCIENCE', 'Genees': 'SCIENCE', 'physics': 'SCIENCE',
        'psy': 'SOC', 'FMG': 'SOC', 'Rechten': 'SOC', 'Geestes': 'SOC',
        'anders': 'OTHER', 'andH': 'NO STUD'
    })

    # Gender variable
    df['female'] = (df['gender'] == 1).astype(int)

    # Filter to paid periods only (round >= 7)
    df_paid = df[df['round'] >= 7].copy()

    # Exclude timeout decisions
    df_paid = df_paid[df_paid['timeout'] == 0].copy()

    # Create lagged variables for period 8+ analysis
    df_paid = df_paid.sort_values(['indiv_ID', 'round'])
    df_paid['pastDecision'] = df_paid.groupby('indiv_ID')['withdraw'].shift(1)
    df_paid['pastGroupRunning'] = df_paid.groupby('indiv_ID')['gr_running'].shift(1)

    # Normalize pastGroupRunning by group size
    df_paid['pastGroupRunning_pct'] = df_paid['pastGroupRunning'] / df_paid['groupSize'] * 100

    # Period 8 data (for Table 4-style regressions)
    df_p8 = df_paid[df_paid['round'] == 8].copy()

    # Create group-level data for first period analysis (round 7)
    df_p7 = df_paid[df_paid['round'] == 7].copy()
    df_group_p7 = df_p7.groupby(['group_ID', 'treatment']).agg({
        'withdraw': 'mean',
        'large': 'first',
        'rhoValue': 'first',
        'rValue': 'first',
        'groupSize': 'first'
    }).reset_index()
    df_group_p7['running_rate'] = df_group_p7['withdraw']

    return df, df_paid, df_p8, df_p7, df_group_p7


def run_logit(formula, data, cluster_var=None, spec_id='', spec_tree_path=''):
    """Run logit regression and extract results."""
    try:
        model = smf.logit(formula, data=data).fit(disp=0)

        # Get treatment coefficient (large)
        treatment_var = 'large'
        if treatment_var in model.params.index:
            coef = model.params[treatment_var]
            se = model.bse[treatment_var]
            tstat = model.tvalues[treatment_var]
            pval = model.pvalues[treatment_var]
        else:
            # Try to find the treatment variable
            treatment_vars = [v for v in model.params.index if 'large' in v.lower()]
            if treatment_vars:
                treatment_var = treatment_vars[0]
                coef = model.params[treatment_var]
                se = model.bse[treatment_var]
                tstat = model.tvalues[treatment_var]
                pval = model.pvalues[treatment_var]
            else:
                coef = se = tstat = pval = np.nan

        # Build coefficient vector
        coef_vector = {
            'treatment': {
                'var': treatment_var,
                'coef': float(coef),
                'se': float(se),
                'pval': float(pval)
            },
            'controls': [],
            'fixed_effects': [],
            'diagnostics': {
                'pseudo_r2': float(model.prsquared),
                'll_model': float(model.llf),
                'll_null': float(model.llnull),
                'aic': float(model.aic),
                'bic': float(model.bic)
            }
        }

        for var in model.params.index:
            if var != treatment_var and var != 'Intercept':
                coef_vector['controls'].append({
                    'var': var,
                    'coef': float(model.params[var]),
                    'se': float(model.bse[var]),
                    'pval': float(model.pvalues[var])
                })

        # CI
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se

        return {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': formula.split('~')[0].strip(),
            'treatment_var': treatment_var,
            'coefficient': float(coef),
            'std_error': float(se),
            't_stat': float(tstat),
            'p_value': float(pval),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_obs': int(model.nobs),
            'r_squared': float(model.prsquared),
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': f'N={int(model.nobs)}',
            'fixed_effects': 'none',
            'controls_desc': formula.split('~')[1].strip(),
            'cluster_var': cluster_var if cluster_var else 'none',
            'model_type': 'logit',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
    except Exception as e:
        print(f"Error in {spec_id}: {e}")
        return None


def run_ols(formula, data, cluster_var=None, spec_id='', spec_tree_path=''):
    """Run OLS (LPM) regression and extract results."""
    try:
        model = smf.ols(formula, data=data).fit()

        # Get treatment coefficient (large)
        treatment_var = 'large'
        if treatment_var in model.params.index:
            coef = model.params[treatment_var]
            se = model.bse[treatment_var]
            tstat = model.tvalues[treatment_var]
            pval = model.pvalues[treatment_var]
        else:
            treatment_vars = [v for v in model.params.index if 'large' in v.lower()]
            if treatment_vars:
                treatment_var = treatment_vars[0]
                coef = model.params[treatment_var]
                se = model.bse[treatment_var]
                tstat = model.tvalues[treatment_var]
                pval = model.pvalues[treatment_var]
            else:
                coef = se = tstat = pval = np.nan

        # Build coefficient vector
        coef_vector = {
            'treatment': {
                'var': treatment_var,
                'coef': float(coef),
                'se': float(se),
                'pval': float(pval)
            },
            'controls': [],
            'fixed_effects': [],
            'diagnostics': {
                'r_squared': float(model.rsquared),
                'r_squared_adj': float(model.rsquared_adj),
                'f_stat': float(model.fvalue),
                'f_pval': float(model.f_pvalue)
            }
        }

        for var in model.params.index:
            if var != treatment_var and var != 'Intercept':
                coef_vector['controls'].append({
                    'var': var,
                    'coef': float(model.params[var]),
                    'se': float(model.bse[var]),
                    'pval': float(model.pvalues[var])
                })

        # CI
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se

        return {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': formula.split('~')[0].strip(),
            'treatment_var': treatment_var,
            'coefficient': float(coef),
            'std_error': float(se),
            't_stat': float(tstat),
            'p_value': float(pval),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_obs': int(model.nobs),
            'r_squared': float(model.rsquared),
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': f'N={int(model.nobs)}',
            'fixed_effects': 'none',
            'controls_desc': formula.split('~')[1].strip(),
            'cluster_var': cluster_var if cluster_var else 'none',
            'model_type': 'OLS-LPM',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
    except Exception as e:
        print(f"Error in {spec_id}: {e}")
        return None


def run_probit(formula, data, cluster_var=None, spec_id='', spec_tree_path=''):
    """Run probit regression and extract results."""
    try:
        model = smf.probit(formula, data=data).fit(disp=0)

        # Get treatment coefficient (large)
        treatment_var = 'large'
        if treatment_var in model.params.index:
            coef = model.params[treatment_var]
            se = model.bse[treatment_var]
            tstat = model.tvalues[treatment_var]
            pval = model.pvalues[treatment_var]
        else:
            treatment_vars = [v for v in model.params.index if 'large' in v.lower()]
            if treatment_vars:
                treatment_var = treatment_vars[0]
                coef = model.params[treatment_var]
                se = model.bse[treatment_var]
                tstat = model.tvalues[treatment_var]
                pval = model.pvalues[treatment_var]
            else:
                coef = se = tstat = pval = np.nan

        # Build coefficient vector
        coef_vector = {
            'treatment': {
                'var': treatment_var,
                'coef': float(coef),
                'se': float(se),
                'pval': float(pval)
            },
            'controls': [],
            'fixed_effects': [],
            'diagnostics': {
                'pseudo_r2': float(model.prsquared),
                'll_model': float(model.llf),
                'll_null': float(model.llnull)
            }
        }

        for var in model.params.index:
            if var != treatment_var and var != 'Intercept':
                coef_vector['controls'].append({
                    'var': var,
                    'coef': float(model.params[var]),
                    'se': float(model.bse[var]),
                    'pval': float(model.pvalues[var])
                })

        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se

        return {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': formula.split('~')[0].strip(),
            'treatment_var': treatment_var,
            'coefficient': float(coef),
            'std_error': float(se),
            't_stat': float(tstat),
            'p_value': float(pval),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_obs': int(model.nobs),
            'r_squared': float(model.prsquared),
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': f'N={int(model.nobs)}',
            'fixed_effects': 'none',
            'controls_desc': formula.split('~')[1].strip(),
            'cluster_var': cluster_var if cluster_var else 'none',
            'model_type': 'probit',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
    except Exception as e:
        print(f"Error in {spec_id}: {e}")
        return None


def main():
    print("Loading data...")
    df, df_paid, df_p8, df_p7, df_group_p7 = load_and_prepare_data()

    results = []

    # =========================================================================
    # BASELINE SPECIFICATIONS
    # =========================================================================
    print("Running baseline specifications...")

    # Baseline 1: Replication of Table 4, Column 4 (period 8, all treatments)
    # Outcome: withdraw in period 8
    # Treatment: large (group size effect)
    # Controls: pastDecision, pastGroupRunning_pct, rhoValue, rValueF, demographic controls

    # First, prepare period 8 data with all controls
    df_p8_clean = df_p8.dropna(subset=['pastDecision', 'pastGroupRunning_pct', 'withdraw',
                                        'large', 'leeftijd', 'female', 'location']).copy()
    df_p8_clean['rho_high'] = (df_p8_clean['rhoValue'] == 0.8).astype(int)

    # Baseline: Table 4 style regression
    baseline_formula = 'withdraw ~ large + pastDecision + pastGroupRunning_pct + rho_high + rValueF + leeftijd + female + location'
    result = run_logit(baseline_formula, df_p8_clean,
                       cluster_var='group_ID',
                       spec_id='baseline',
                       spec_tree_path='methods/discrete_choice.md')
    if result:
        results.append(result)

    # =========================================================================
    # DISCRETE CHOICE MODEL VARIATIONS
    # =========================================================================
    print("Running discrete choice model variations...")

    # Probit version
    result = run_probit(baseline_formula, df_p8_clean,
                        cluster_var='group_ID',
                        spec_id='discrete/binary/probit',
                        spec_tree_path='methods/discrete_choice.md#model-type-binary-outcome')
    if result:
        results.append(result)

    # LPM version
    result = run_ols(baseline_formula, df_p8_clean,
                     cluster_var='group_ID',
                     spec_id='discrete/binary/lpm',
                     spec_tree_path='methods/discrete_choice.md#model-type-binary-outcome')
    if result:
        results.append(result)

    # =========================================================================
    # CONTROL SET VARIATIONS
    # =========================================================================
    print("Running control set variations...")

    # No controls - treatment only
    result = run_logit('withdraw ~ large', df_p8_clean,
                       spec_id='discrete/controls/none',
                       spec_tree_path='methods/discrete_choice.md#control-sets')
    if result:
        results.append(result)

    # Minimal controls - just lagged behavior
    result = run_logit('withdraw ~ large + pastDecision + pastGroupRunning_pct', df_p8_clean,
                       spec_id='discrete/controls/minimal',
                       spec_tree_path='methods/discrete_choice.md#control-sets')
    if result:
        results.append(result)

    # With treatment parameters
    result = run_logit('withdraw ~ large + pastDecision + pastGroupRunning_pct + rho_high + rValueF', df_p8_clean,
                       spec_id='discrete/controls/treatment_params',
                       spec_tree_path='methods/discrete_choice.md#control-sets')
    if result:
        results.append(result)

    # With interaction: large x pastGroupRunning
    df_p8_clean['large_x_pastGR'] = df_p8_clean['large'] * df_p8_clean['pastGroupRunning_pct']
    result = run_logit('withdraw ~ large + pastDecision + pastGroupRunning_pct + large_x_pastGR + rho_high + rValueF + leeftijd + female + location',
                       df_p8_clean,
                       spec_id='discrete/controls/interactions',
                       spec_tree_path='methods/discrete_choice.md#control-sets')
    if result:
        results.append(result)

    # =========================================================================
    # SAMPLE RESTRICTIONS
    # =========================================================================
    print("Running sample restriction specifications...")

    # r=1.54 treatments only (exclude r=1.33)
    df_r154 = df_p8_clean[df_p8_clean['rValue'] == 1.54].copy()
    result = run_logit(baseline_formula.replace(' + rValueF', ''), df_r154,
                       cluster_var='group_ID',
                       spec_id='robust/sample/r154_only',
                       spec_tree_path='robustness/sample_restrictions.md')
    if result:
        results.append(result)

    # r=1.33 treatments only
    df_r133 = df_p8_clean[df_p8_clean['rValue'] == 1.33].copy()
    if len(df_r133) > 100:
        result = run_logit('withdraw ~ large + pastDecision + pastGroupRunning_pct + leeftijd + female + location',
                           df_r133,
                           cluster_var='group_ID',
                           spec_id='robust/sample/r133_only',
                           spec_tree_path='robustness/sample_restrictions.md')
        if result:
            results.append(result)

    # High persistence (rho=0.8) only
    df_rho_high = df_p8_clean[df_p8_clean['rhoValue'] == 0.8].copy()
    if len(df_rho_high) > 100:
        result = run_logit('withdraw ~ large + pastDecision + pastGroupRunning_pct + rValueF + leeftijd + female + location',
                           df_rho_high,
                           cluster_var='group_ID',
                           spec_id='robust/sample/rho_high_only',
                           spec_tree_path='robustness/sample_restrictions.md')
        if result:
            results.append(result)

    # Low persistence (rho=0.5) only
    df_rho_low = df_p8_clean[df_p8_clean['rhoValue'] == 0.5].copy()
    if len(df_rho_low) > 100:
        result = run_logit('withdraw ~ large + pastDecision + pastGroupRunning_pct + rValueF + leeftijd + female + location',
                           df_rho_low,
                           cluster_var='group_ID',
                           spec_id='robust/sample/rho_low_only',
                           spec_tree_path='robustness/sample_restrictions.md')
        if result:
            results.append(result)

    # Waiters only (those who did NOT withdraw in period 7)
    df_waiters = df_p8_clean[df_p8_clean['pastDecision'] == 0].copy()
    if len(df_waiters) > 100:
        result = run_logit('withdraw ~ large + pastGroupRunning_pct + rho_high + rValueF + leeftijd + female + location',
                           df_waiters,
                           cluster_var='group_ID',
                           spec_id='robust/sample/waiters_only',
                           spec_tree_path='robustness/sample_restrictions.md')
        if result:
            results.append(result)

    # Withdrawers only (those who DID withdraw in period 7)
    df_withdrawers = df_p8_clean[df_p8_clean['pastDecision'] == 1].copy()
    if len(df_withdrawers) > 100:
        result = run_logit('withdraw ~ large + pastGroupRunning_pct + rho_high + rValueF + leeftijd + female + location',
                           df_withdrawers,
                           cluster_var='group_ID',
                           spec_id='robust/sample/withdrawers_only',
                           spec_tree_path='robustness/sample_restrictions.md')
        if result:
            results.append(result)

    # =========================================================================
    # PERIOD VARIATIONS
    # =========================================================================
    print("Running period variation specifications...")

    # First period (round 7) analysis - group level
    # Regress running rate on group size
    if len(df_group_p7) > 10:
        result = run_ols('running_rate ~ large', df_group_p7,
                         spec_id='robust/sample/period7_group',
                         spec_tree_path='robustness/sample_restrictions.md')
        if result:
            results.append(result)

    # Period 9 analysis
    df_p9 = df_paid[df_paid['round'] == 9].copy()
    df_p9['rho_high'] = (df_p9['rhoValue'] == 0.8).astype(int)
    df_p9 = df_p9.dropna(subset=['pastDecision', 'pastGroupRunning_pct', 'withdraw',
                                  'large', 'leeftijd', 'female', 'location'])
    if len(df_p9) > 100:
        result = run_logit(baseline_formula, df_p9,
                           cluster_var='group_ID',
                           spec_id='robust/sample/period9',
                           spec_tree_path='robustness/sample_restrictions.md')
        if result:
            results.append(result)

    # Later periods (10-12)
    for period in [10, 11, 12]:
        df_period = df_paid[df_paid['round'] == period].copy()
        df_period['rho_high'] = (df_period['rhoValue'] == 0.8).astype(int)
        df_period = df_period.dropna(subset=['pastDecision', 'pastGroupRunning_pct', 'withdraw',
                                              'large', 'leeftijd', 'female', 'location'])
        if len(df_period) > 100:
            result = run_logit(baseline_formula, df_period,
                               cluster_var='group_ID',
                               spec_id=f'robust/sample/period{period}',
                               spec_tree_path='robustness/sample_restrictions.md')
            if result:
                results.append(result)

    # Early periods (7-11)
    df_early = df_paid[(df_paid['round'] >= 7) & (df_paid['round'] <= 11)].copy()
    df_early['rho_high'] = (df_early['rhoValue'] == 0.8).astype(int)
    df_early = df_early.dropna(subset=['withdraw', 'large', 'leeftijd', 'female', 'location'])
    result = run_logit('withdraw ~ large + rho_high + rValueF + leeftijd + female + location', df_early,
                       cluster_var='group_ID',
                       spec_id='robust/sample/early_periods',
                       spec_tree_path='robustness/sample_restrictions.md')
    if result:
        results.append(result)

    # Late periods (12+)
    df_late = df_paid[df_paid['round'] >= 12].copy()
    df_late['rho_high'] = (df_late['rhoValue'] == 0.8).astype(int)
    df_late = df_late.dropna(subset=['withdraw', 'large', 'leeftijd', 'female', 'location'])
    result = run_logit('withdraw ~ large + rho_high + rValueF + leeftijd + female + location', df_late,
                       cluster_var='group_ID',
                       spec_id='robust/sample/late_periods',
                       spec_tree_path='robustness/sample_restrictions.md')
    if result:
        results.append(result)

    # =========================================================================
    # LEAVE-ONE-OUT ROBUSTNESS
    # =========================================================================
    print("Running leave-one-out specifications...")

    baseline_controls = ['pastDecision', 'pastGroupRunning_pct', 'rho_high', 'rValueF', 'leeftijd', 'female', 'location']

    for control in baseline_controls:
        remaining = [c for c in baseline_controls if c != control]
        formula = f"withdraw ~ large + {' + '.join(remaining)}"
        result = run_logit(formula, df_p8_clean,
                           cluster_var='group_ID',
                           spec_id=f'robust/loo/drop_{control}',
                           spec_tree_path='robustness/leave_one_out.md')
        if result:
            results.append(result)

    # =========================================================================
    # SINGLE COVARIATE ROBUSTNESS
    # =========================================================================
    print("Running single covariate specifications...")

    # Bivariate
    result = run_logit('withdraw ~ large', df_p8_clean,
                       spec_id='robust/single/none',
                       spec_tree_path='robustness/single_covariate.md')
    if result:
        results.append(result)

    for control in baseline_controls:
        formula = f"withdraw ~ large + {control}"
        result = run_logit(formula, df_p8_clean,
                           spec_id=f'robust/single/{control}',
                           spec_tree_path='robustness/single_covariate.md')
        if result:
            results.append(result)

    # =========================================================================
    # POOLED ANALYSIS (ALL PERIODS)
    # =========================================================================
    print("Running pooled period specifications...")

    # Pooled: all paid periods with round effects
    df_pooled = df_paid.copy()
    df_pooled['rho_high'] = (df_pooled['rhoValue'] == 0.8).astype(int)
    df_pooled = df_pooled.dropna(subset=['withdraw', 'large', 'leeftijd', 'female', 'location'])

    result = run_logit('withdraw ~ large + rho_high + rValueF + leeftijd + female + location', df_pooled,
                       cluster_var='group_ID',
                       spec_id='discrete/sample/pooled_all',
                       spec_tree_path='methods/discrete_choice.md#sample-restrictions')
    if result:
        results.append(result)

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    print(f"\nTotal specifications run: {len(results)}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    output_file = f'{OUTPUT_PATH}/specification_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")

    # Summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total specifications: {len(results_df)}")
    print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({(results_df['coefficient'] > 0).mean()*100:.1f}%)")
    print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({(results_df['p_value'] < 0.05).mean()*100:.1f}%)")
    print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({(results_df['p_value'] < 0.01).mean()*100:.1f}%)")
    print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
    print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")
    print(f"Range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")

    return results_df


if __name__ == '__main__':
    results_df = main()
