"""
Specification Search: 114333-V1
Paper: Team versus Individual Play in Finitely Repeated Prisoner Dilemma Games
Journal: AEJ: Microeconomics (2016)
Authors: Kuhn and Kagel

Main Hypothesis: Teams behave differently than individuals in repeated prisoner's dilemma games.
                 Specifically, teams are hypothesized to be more strategic (backward induction,
                 coordination on cooperation paths, strategic defection timing).

Treatment: treatment = 1 (individuals) vs treatment = 2 (teams)
Outcome: Cooperation rates (realchoice: 1=Cooperate, 2=Defect)
Method: Discrete choice with panel structure (player/team x super-game x round)

Data source: Lab experiment at Ohio State University, 2012
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

# For regression
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Set paths
DATA_DIR = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/114333-V1/Data_files_zip'
OUTPUT_DIR = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/114333-V1'

# ============================================================================
# STEP 1: Load and prepare data
# ============================================================================

# Load choice data
df_all = pd.read_excel(f'{DATA_DIR}/Choice_Data_detailed.xlsx', sheet_name='All data')

# Load defection patterns to get treatment assignment by session
df_defect = pd.read_excel(f'{DATA_DIR}/defection-patterns.xlsx', sheet_name='Data')

# Create session to treatment mapping
# Treatment: 1 = individuals, 2 = teams
session_treatment = df_defect.groupby('Session')['Treatment'].first().to_dict()

# Map treatment to main data
df_all['treatment'] = df_all['session'].map(session_treatment)

# Create cooperation indicator (1 = cooperate, 0 = defect)
# realchoice: 1 = Cooperate, 2 = Defect
df_all['cooperate'] = (df_all['realchoice'] == 1).astype(int)

# Create team indicator
df_all['is_team'] = (df_all['treatment'] == 2).astype(int)

# Create unique player identifier (session + team)
df_all['player_id'] = df_all['session'] + '_' + df_all['team'].astype(str)

# Rename 'block' to 'supergame' for clarity
df_all['supergame'] = df_all['block']

# Create round within supergame
df_all['round'] = df_all.groupby(['session', 'team', 'supergame']).cumcount() + 1

# Get supergame length for each observation
supergame_lengths = df_all.groupby(['session', 'team', 'supergame'])['round'].transform('max')
df_all['supergame_length'] = supergame_lengths

# Create indicators for early/late supergames
df_all['early_supergame'] = (df_all['supergame'] <= 3).astype(int)
df_all['late_supergame'] = (df_all['supergame'] >= 5).astype(int)

# Create round indicators
df_all['first_round'] = (df_all['round'] == 1).astype(int)
df_all['last_round'] = (df_all['round'] == df_all['supergame_length']).astype(int)
df_all['early_round'] = (df_all['round'] <= 4).astype(int)

print("Data prepared successfully")
print(f"Total observations: {len(df_all)}")
print(f"Individuals (treatment=1): {(df_all['treatment']==1).sum()}")
print(f"Teams (treatment=2): {(df_all['treatment']==2).sum()}")
print(f"Overall cooperation rate: {df_all['cooperate'].mean():.3f}")
print(f"Cooperation by individuals: {df_all[df_all['treatment']==1]['cooperate'].mean():.3f}")
print(f"Cooperation by teams: {df_all[df_all['treatment']==2]['cooperate'].mean():.3f}")

# ============================================================================
# STEP 2: Prepare results storage
# ============================================================================

results = []

PAPER_ID = '114333-V1'
PAPER_TITLE = 'Team versus Individual Play in Finitely Repeated Prisoner Dilemma Games'
JOURNAL = 'AEJ: Microeconomics'

def extract_results(model, spec_id, spec_tree_path, outcome_var, treatment_var,
                   n_obs, controls_desc, fixed_effects, cluster_var, model_type,
                   sample_desc='Full sample'):
    """Extract results from a statsmodels regression"""

    # Get coefficient and standard error for treatment
    try:
        coef = model.params[treatment_var]
        se = model.bse[treatment_var]
        tstat = model.tvalues[treatment_var]
        pval = model.pvalues[treatment_var]
        ci = model.conf_int().loc[treatment_var]
        ci_lower = ci[0]
        ci_upper = ci[1]
    except KeyError:
        # Variable name might be different
        treatment_var_alt = 'is_team'
        coef = model.params[treatment_var_alt]
        se = model.bse[treatment_var_alt]
        tstat = model.tvalues[treatment_var_alt]
        pval = model.pvalues[treatment_var_alt]
        ci = model.conf_int().loc[treatment_var_alt]
        ci_lower = ci[0]
        ci_upper = ci[1]

    # Build coefficient vector
    coef_vector = {
        'treatment': {
            'var': treatment_var,
            'coef': float(coef),
            'se': float(se),
            'pval': float(pval)
        },
        'controls': [],
        'fixed_effects': fixed_effects.split(', ') if fixed_effects else [],
        'diagnostics': {}
    }

    # Add controls to coefficient vector
    for var in model.params.index:
        if var not in [treatment_var, 'Intercept', 'const']:
            coef_vector['controls'].append({
                'var': var,
                'coef': float(model.params[var]),
                'se': float(model.bse[var]),
                'pval': float(model.pvalues[var])
            })

    # Add diagnostics
    if hasattr(model, 'rsquared'):
        coef_vector['diagnostics']['r_squared'] = float(model.rsquared)
    if hasattr(model, 'prsquared'):
        coef_vector['diagnostics']['pseudo_r_squared'] = float(model.prsquared)
    if hasattr(model, 'llf'):
        coef_vector['diagnostics']['log_likelihood'] = float(model.llf)

    r_squared = model.rsquared if hasattr(model, 'rsquared') else (model.prsquared if hasattr(model, 'prsquared') else None)

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
        'n_obs': int(n_obs),
        'r_squared': float(r_squared) if r_squared else None,
        'coefficient_vector_json': json.dumps(coef_vector),
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': 'scripts/paper_analyses/114333-V1.py'
    }

# ============================================================================
# STEP 3: BASELINE SPECIFICATION
# Main hypothesis: Teams cooperate differently than individuals
# ============================================================================

print("\n" + "="*60)
print("BASELINE SPECIFICATION")
print("="*60)

# Baseline: Simple comparison of cooperation rates
# OLS regression of cooperation on team indicator
model_baseline = smf.ols('cooperate ~ is_team', data=df_all).fit(cov_type='cluster', cov_kwds={'groups': df_all['session']})
print(model_baseline.summary())

results.append(extract_results(
    model_baseline,
    spec_id='baseline',
    spec_tree_path='methods/discrete_choice.md#baseline',
    outcome_var='cooperate',
    treatment_var='is_team',
    n_obs=len(df_all),
    controls_desc='None',
    fixed_effects='None',
    cluster_var='session',
    model_type='OLS'
))

# ============================================================================
# STEP 4: DISCRETE CHOICE SPECIFICATIONS
# ============================================================================

print("\n" + "="*60)
print("DISCRETE CHOICE SPECIFICATIONS")
print("="*60)

# 4.1 Logit model
model_logit = smf.logit('cooperate ~ is_team', data=df_all).fit(disp=0)
print("\nLogit model:")
print(model_logit.summary())

# Get marginal effects
mfx = model_logit.get_margeff(at='overall')
print("\nMarginal effects (AME):")
print(mfx.summary())

results.append(extract_results(
    model_logit,
    spec_id='discrete/binary/logit',
    spec_tree_path='methods/discrete_choice.md#model-type-binary-outcome',
    outcome_var='cooperate',
    treatment_var='is_team',
    n_obs=len(df_all),
    controls_desc='None',
    fixed_effects='None',
    cluster_var='None',
    model_type='Logit'
))

# 4.2 Probit model
model_probit = smf.probit('cooperate ~ is_team', data=df_all).fit(disp=0)
print("\nProbit model:")
print(model_probit.summary())

results.append(extract_results(
    model_probit,
    spec_id='discrete/binary/probit',
    spec_tree_path='methods/discrete_choice.md#model-type-binary-outcome',
    outcome_var='cooperate',
    treatment_var='is_team',
    n_obs=len(df_all),
    controls_desc='None',
    fixed_effects='None',
    cluster_var='None',
    model_type='Probit'
))

# 4.3 LPM with robust SE
model_lpm_robust = smf.ols('cooperate ~ is_team', data=df_all).fit(cov_type='HC3')
print("\nLPM with robust SE (HC3):")
print(model_lpm_robust.summary())

results.append(extract_results(
    model_lpm_robust,
    spec_id='discrete/binary/lpm',
    spec_tree_path='methods/discrete_choice.md#model-type-binary-outcome',
    outcome_var='cooperate',
    treatment_var='is_team',
    n_obs=len(df_all),
    controls_desc='None',
    fixed_effects='None',
    cluster_var='None (HC3)',
    model_type='LPM'
))

# ============================================================================
# STEP 5: PANEL FIXED EFFECTS SPECIFICATIONS
# ============================================================================

print("\n" + "="*60)
print("PANEL FIXED EFFECTS SPECIFICATIONS")
print("="*60)

# 5.1 Supergame fixed effects
# Create dummies for supergame
for sg in df_all['supergame'].unique():
    df_all[f'sg_{sg}'] = (df_all['supergame'] == sg).astype(int)

model_sg_fe = smf.ols('cooperate ~ is_team + C(supergame)', data=df_all).fit(cov_type='cluster', cov_kwds={'groups': df_all['session']})
print("\nWith supergame FE:")
print(f"Coefficient on is_team: {model_sg_fe.params['is_team']:.4f}, SE: {model_sg_fe.bse['is_team']:.4f}")

results.append(extract_results(
    model_sg_fe,
    spec_id='panel/fe/time',
    spec_tree_path='methods/panel_fixed_effects.md#fixed-effects-structure',
    outcome_var='cooperate',
    treatment_var='is_team',
    n_obs=len(df_all),
    controls_desc='None',
    fixed_effects='supergame',
    cluster_var='session',
    model_type='OLS with FE'
))

# 5.2 Round fixed effects
model_round_fe = smf.ols('cooperate ~ is_team + C(round)', data=df_all).fit(cov_type='cluster', cov_kwds={'groups': df_all['session']})
print("\nWith round FE:")
print(f"Coefficient on is_team: {model_round_fe.params['is_team']:.4f}, SE: {model_round_fe.bse['is_team']:.4f}")

results.append(extract_results(
    model_round_fe,
    spec_id='panel/fe/round',
    spec_tree_path='methods/panel_fixed_effects.md#fixed-effects-structure',
    outcome_var='cooperate',
    treatment_var='is_team',
    n_obs=len(df_all),
    controls_desc='None',
    fixed_effects='round',
    cluster_var='session',
    model_type='OLS with FE'
))

# 5.3 Supergame + Round FE
model_twoway_fe = smf.ols('cooperate ~ is_team + C(supergame) + C(round)', data=df_all).fit(cov_type='cluster', cov_kwds={'groups': df_all['session']})
print("\nWith supergame + round FE:")
print(f"Coefficient on is_team: {model_twoway_fe.params['is_team']:.4f}, SE: {model_twoway_fe.bse['is_team']:.4f}")

results.append(extract_results(
    model_twoway_fe,
    spec_id='panel/fe/twoway',
    spec_tree_path='methods/panel_fixed_effects.md#fixed-effects-structure',
    outcome_var='cooperate',
    treatment_var='is_team',
    n_obs=len(df_all),
    controls_desc='None',
    fixed_effects='supergame, round',
    cluster_var='session',
    model_type='OLS with FE'
))

# ============================================================================
# STEP 6: CONTROL SPECIFICATIONS
# ============================================================================

print("\n" + "="*60)
print("CONTROL SPECIFICATIONS")
print("="*60)

# 6.1 Control for round
model_ctrl_round = smf.ols('cooperate ~ is_team + round', data=df_all).fit(cov_type='cluster', cov_kwds={'groups': df_all['session']})
print("\nWith round control:")
print(f"Coefficient on is_team: {model_ctrl_round.params['is_team']:.4f}, SE: {model_ctrl_round.bse['is_team']:.4f}")

results.append(extract_results(
    model_ctrl_round,
    spec_id='discrete/controls/round',
    spec_tree_path='methods/discrete_choice.md#control-sets',
    outcome_var='cooperate',
    treatment_var='is_team',
    n_obs=len(df_all),
    controls_desc='round',
    fixed_effects='None',
    cluster_var='session',
    model_type='OLS'
))

# 6.2 Control for supergame
model_ctrl_sg = smf.ols('cooperate ~ is_team + supergame', data=df_all).fit(cov_type='cluster', cov_kwds={'groups': df_all['session']})
print("\nWith supergame control:")
print(f"Coefficient on is_team: {model_ctrl_sg.params['is_team']:.4f}, SE: {model_ctrl_sg.bse['is_team']:.4f}")

results.append(extract_results(
    model_ctrl_sg,
    spec_id='discrete/controls/supergame',
    spec_tree_path='methods/discrete_choice.md#control-sets',
    outcome_var='cooperate',
    treatment_var='is_team',
    n_obs=len(df_all),
    controls_desc='supergame',
    fixed_effects='None',
    cluster_var='session',
    model_type='OLS'
))

# 6.3 Full controls
model_ctrl_full = smf.ols('cooperate ~ is_team + round + supergame + supergame_length', data=df_all).fit(cov_type='cluster', cov_kwds={'groups': df_all['session']})
print("\nWith full controls:")
print(f"Coefficient on is_team: {model_ctrl_full.params['is_team']:.4f}, SE: {model_ctrl_full.bse['is_team']:.4f}")

results.append(extract_results(
    model_ctrl_full,
    spec_id='discrete/controls/full',
    spec_tree_path='methods/discrete_choice.md#control-sets',
    outcome_var='cooperate',
    treatment_var='is_team',
    n_obs=len(df_all),
    controls_desc='round, supergame, supergame_length',
    fixed_effects='None',
    cluster_var='session',
    model_type='OLS'
))

# ============================================================================
# STEP 7: CLUSTERING VARIATIONS
# ============================================================================

print("\n" + "="*60)
print("CLUSTERING VARIATIONS")
print("="*60)

# 7.1 No clustering (robust SE)
model_robust = smf.ols('cooperate ~ is_team', data=df_all).fit(cov_type='HC1')
print("\nRobust SE (HC1):")
print(f"Coefficient on is_team: {model_robust.params['is_team']:.4f}, SE: {model_robust.bse['is_team']:.4f}")

results.append(extract_results(
    model_robust,
    spec_id='robust/cluster/none',
    spec_tree_path='robustness/clustering_variations.md#single-level-clustering',
    outcome_var='cooperate',
    treatment_var='is_team',
    n_obs=len(df_all),
    controls_desc='None',
    fixed_effects='None',
    cluster_var='None (HC1)',
    model_type='OLS'
))

# 7.2 Cluster by session
model_cluster_session = smf.ols('cooperate ~ is_team', data=df_all).fit(cov_type='cluster', cov_kwds={'groups': df_all['session']})
print("\nClustered by session:")
print(f"Coefficient on is_team: {model_cluster_session.params['is_team']:.4f}, SE: {model_cluster_session.bse['is_team']:.4f}")

results.append(extract_results(
    model_cluster_session,
    spec_id='robust/cluster/session',
    spec_tree_path='robustness/clustering_variations.md#single-level-clustering',
    outcome_var='cooperate',
    treatment_var='is_team',
    n_obs=len(df_all),
    controls_desc='None',
    fixed_effects='None',
    cluster_var='session',
    model_type='OLS'
))

# 7.3 Cluster by player
model_cluster_player = smf.ols('cooperate ~ is_team', data=df_all).fit(cov_type='cluster', cov_kwds={'groups': df_all['player_id']})
print("\nClustered by player:")
print(f"Coefficient on is_team: {model_cluster_player.params['is_team']:.4f}, SE: {model_cluster_player.bse['is_team']:.4f}")

results.append(extract_results(
    model_cluster_player,
    spec_id='robust/cluster/unit',
    spec_tree_path='robustness/clustering_variations.md#single-level-clustering',
    outcome_var='cooperate',
    treatment_var='is_team',
    n_obs=len(df_all),
    controls_desc='None',
    fixed_effects='None',
    cluster_var='player_id',
    model_type='OLS'
))

# 7.4 Cluster by supergame
model_cluster_sg = smf.ols('cooperate ~ is_team', data=df_all).fit(cov_type='cluster', cov_kwds={'groups': df_all['supergame']})
print("\nClustered by supergame:")
print(f"Coefficient on is_team: {model_cluster_sg.params['is_team']:.4f}, SE: {model_cluster_sg.bse['is_team']:.4f}")

results.append(extract_results(
    model_cluster_sg,
    spec_id='robust/cluster/time',
    spec_tree_path='robustness/clustering_variations.md#single-level-clustering',
    outcome_var='cooperate',
    treatment_var='is_team',
    n_obs=len(df_all),
    controls_desc='None',
    fixed_effects='None',
    cluster_var='supergame',
    model_type='OLS'
))

# ============================================================================
# STEP 8: SAMPLE RESTRICTIONS
# ============================================================================

print("\n" + "="*60)
print("SAMPLE RESTRICTIONS")
print("="*60)

# 8.1 Early supergames only (learning phase)
df_early = df_all[df_all['supergame'] <= 3]
model_early = smf.ols('cooperate ~ is_team', data=df_early).fit(cov_type='cluster', cov_kwds={'groups': df_early['session']})
print(f"\nEarly supergames (1-3): n={len(df_early)}")
print(f"Coefficient on is_team: {model_early.params['is_team']:.4f}, SE: {model_early.bse['is_team']:.4f}")

results.append(extract_results(
    model_early,
    spec_id='robust/sample/early_period',
    spec_tree_path='robustness/sample_restrictions.md#time-based-restrictions',
    outcome_var='cooperate',
    treatment_var='is_team',
    n_obs=len(df_early),
    controls_desc='None',
    fixed_effects='None',
    cluster_var='session',
    model_type='OLS',
    sample_desc='Early supergames (1-3)'
))

# 8.2 Late supergames only (experienced play)
df_late = df_all[df_all['supergame'] >= 5]
model_late = smf.ols('cooperate ~ is_team', data=df_late).fit(cov_type='cluster', cov_kwds={'groups': df_late['session']})
print(f"\nLate supergames (5+): n={len(df_late)}")
print(f"Coefficient on is_team: {model_late.params['is_team']:.4f}, SE: {model_late.bse['is_team']:.4f}")

results.append(extract_results(
    model_late,
    spec_id='robust/sample/late_period',
    spec_tree_path='robustness/sample_restrictions.md#time-based-restrictions',
    outcome_var='cooperate',
    treatment_var='is_team',
    n_obs=len(df_late),
    controls_desc='None',
    fixed_effects='None',
    cluster_var='session',
    model_type='OLS',
    sample_desc='Late supergames (5+)'
))

# 8.3 First round only (initial cooperation decision)
df_first = df_all[df_all['round'] == 1]
model_first = smf.ols('cooperate ~ is_team', data=df_first).fit(cov_type='cluster', cov_kwds={'groups': df_first['session']})
print(f"\nFirst round only: n={len(df_first)}")
print(f"Coefficient on is_team: {model_first.params['is_team']:.4f}, SE: {model_first.bse['is_team']:.4f}")

results.append(extract_results(
    model_first,
    spec_id='robust/sample/first_round',
    spec_tree_path='robustness/sample_restrictions.md#time-based-restrictions',
    outcome_var='cooperate',
    treatment_var='is_team',
    n_obs=len(df_first),
    controls_desc='None',
    fixed_effects='None',
    cluster_var='session',
    model_type='OLS',
    sample_desc='First round of each supergame'
))

# 8.4 Last round only (endgame behavior)
df_last = df_all[df_all['round'] == df_all['supergame_length']]
model_last = smf.ols('cooperate ~ is_team', data=df_last).fit(cov_type='cluster', cov_kwds={'groups': df_last['session']})
print(f"\nLast round only: n={len(df_last)}")
print(f"Coefficient on is_team: {model_last.params['is_team']:.4f}, SE: {model_last.bse['is_team']:.4f}")

results.append(extract_results(
    model_last,
    spec_id='robust/sample/last_round',
    spec_tree_path='robustness/sample_restrictions.md#time-based-restrictions',
    outcome_var='cooperate',
    treatment_var='is_team',
    n_obs=len(df_last),
    controls_desc='None',
    fixed_effects='None',
    cluster_var='session',
    model_type='OLS',
    sample_desc='Last round of each supergame'
))

# 8.5 Early rounds only (rounds 1-4)
df_early_rounds = df_all[df_all['round'] <= 4]
model_early_rounds = smf.ols('cooperate ~ is_team', data=df_early_rounds).fit(cov_type='cluster', cov_kwds={'groups': df_early_rounds['session']})
print(f"\nEarly rounds (1-4): n={len(df_early_rounds)}")
print(f"Coefficient on is_team: {model_early_rounds.params['is_team']:.4f}, SE: {model_early_rounds.bse['is_team']:.4f}")

results.append(extract_results(
    model_early_rounds,
    spec_id='robust/sample/early_rounds',
    spec_tree_path='robustness/sample_restrictions.md#time-based-restrictions',
    outcome_var='cooperate',
    treatment_var='is_team',
    n_obs=len(df_early_rounds),
    controls_desc='None',
    fixed_effects='None',
    cluster_var='session',
    model_type='OLS',
    sample_desc='Early rounds (1-4)'
))

# 8.6 Late rounds only (last 3 rounds of supergame)
df_late_rounds = df_all[df_all['round'] > df_all['supergame_length'] - 3]
model_late_rounds = smf.ols('cooperate ~ is_team', data=df_late_rounds).fit(cov_type='cluster', cov_kwds={'groups': df_late_rounds['session']})
print(f"\nLate rounds (last 3): n={len(df_late_rounds)}")
print(f"Coefficient on is_team: {model_late_rounds.params['is_team']:.4f}, SE: {model_late_rounds.bse['is_team']:.4f}")

results.append(extract_results(
    model_late_rounds,
    spec_id='robust/sample/late_rounds',
    spec_tree_path='robustness/sample_restrictions.md#time-based-restrictions',
    outcome_var='cooperate',
    treatment_var='is_team',
    n_obs=len(df_late_rounds),
    controls_desc='None',
    fixed_effects='None',
    cluster_var='session',
    model_type='OLS',
    sample_desc='Late rounds (last 3 of each supergame)'
))

# 8.7 Exclude first supergame (drop learning period)
df_no_first = df_all[df_all['supergame'] > 1]
model_no_first = smf.ols('cooperate ~ is_team', data=df_no_first).fit(cov_type='cluster', cov_kwds={'groups': df_no_first['session']})
print(f"\nExcluding first supergame: n={len(df_no_first)}")
print(f"Coefficient on is_team: {model_no_first.params['is_team']:.4f}, SE: {model_no_first.bse['is_team']:.4f}")

results.append(extract_results(
    model_no_first,
    spec_id='robust/sample/exclude_first_sg',
    spec_tree_path='robustness/sample_restrictions.md#time-based-restrictions',
    outcome_var='cooperate',
    treatment_var='is_team',
    n_obs=len(df_no_first),
    controls_desc='None',
    fixed_effects='None',
    cluster_var='session',
    model_type='OLS',
    sample_desc='Excluding first supergame'
))

# ============================================================================
# STEP 9: INTERACTION SPECIFICATIONS
# ============================================================================

print("\n" + "="*60)
print("INTERACTION SPECIFICATIONS")
print("="*60)

# 9.1 Team x supergame interaction
model_interact_sg = smf.ols('cooperate ~ is_team * supergame', data=df_all).fit(cov_type='cluster', cov_kwds={'groups': df_all['session']})
print("\nTeam x Supergame interaction:")
print(f"is_team: {model_interact_sg.params['is_team']:.4f}")
print(f"is_team:supergame: {model_interact_sg.params['is_team:supergame']:.4f}")

results.append(extract_results(
    model_interact_sg,
    spec_id='robust/form/interact_time',
    spec_tree_path='robustness/functional_form.md#interaction-terms',
    outcome_var='cooperate',
    treatment_var='is_team',
    n_obs=len(df_all),
    controls_desc='supergame, is_team*supergame',
    fixed_effects='None',
    cluster_var='session',
    model_type='OLS'
))

# 9.2 Team x round interaction
model_interact_round = smf.ols('cooperate ~ is_team * round', data=df_all).fit(cov_type='cluster', cov_kwds={'groups': df_all['session']})
print("\nTeam x Round interaction:")
print(f"is_team: {model_interact_round.params['is_team']:.4f}")
print(f"is_team:round: {model_interact_round.params['is_team:round']:.4f}")

results.append(extract_results(
    model_interact_round,
    spec_id='robust/form/interact_round',
    spec_tree_path='robustness/functional_form.md#interaction-terms',
    outcome_var='cooperate',
    treatment_var='is_team',
    n_obs=len(df_all),
    controls_desc='round, is_team*round',
    fixed_effects='None',
    cluster_var='session',
    model_type='OLS'
))

# 9.3 Team x early_supergame interaction
model_interact_early = smf.ols('cooperate ~ is_team * early_supergame', data=df_all).fit(cov_type='cluster', cov_kwds={'groups': df_all['session']})
print("\nTeam x Early Supergame interaction:")
print(f"is_team: {model_interact_early.params['is_team']:.4f}")

results.append(extract_results(
    model_interact_early,
    spec_id='robust/form/interact_early',
    spec_tree_path='robustness/functional_form.md#interaction-terms',
    outcome_var='cooperate',
    treatment_var='is_team',
    n_obs=len(df_all),
    controls_desc='early_supergame, is_team*early_supergame',
    fixed_effects='None',
    cluster_var='session',
    model_type='OLS'
))

# ============================================================================
# STEP 10: LEAVE-ONE-OUT (Sessions)
# ============================================================================

print("\n" + "="*60)
print("LEAVE-ONE-OUT (Sessions)")
print("="*60)

for session in df_all['session'].unique():
    df_loo = df_all[df_all['session'] != session]
    model_loo = smf.ols('cooperate ~ is_team', data=df_loo).fit(cov_type='cluster', cov_kwds={'groups': df_loo['session']})
    print(f"Excluding {session}: coef={model_loo.params['is_team']:.4f}, SE={model_loo.bse['is_team']:.4f}")

    results.append(extract_results(
        model_loo,
        spec_id=f'robust/loo/drop_{session}',
        spec_tree_path='robustness/leave_one_out.md',
        outcome_var='cooperate',
        treatment_var='is_team',
        n_obs=len(df_loo),
        controls_desc='None',
        fixed_effects='None',
        cluster_var='session',
        model_type='OLS',
        sample_desc=f'Excluding session {session}'
    ))

# ============================================================================
# STEP 11: BY SUPERGAME ANALYSIS
# ============================================================================

print("\n" + "="*60)
print("BY SUPERGAME ANALYSIS")
print("="*60)

for sg in sorted(df_all['supergame'].unique()):
    df_sg = df_all[df_all['supergame'] == sg]
    if len(df_sg) > 0 and df_sg['is_team'].nunique() > 1:
        model_sg = smf.ols('cooperate ~ is_team', data=df_sg).fit(cov_type='cluster', cov_kwds={'groups': df_sg['session']})
        print(f"Supergame {sg}: coef={model_sg.params['is_team']:.4f}, SE={model_sg.bse['is_team']:.4f}, n={len(df_sg)}")

        results.append(extract_results(
            model_sg,
            spec_id=f'discrete/sample/supergame_{sg}',
            spec_tree_path='methods/discrete_choice.md#sample-restrictions',
            outcome_var='cooperate',
            treatment_var='is_team',
            n_obs=len(df_sg),
            controls_desc='None',
            fixed_effects='None',
            cluster_var='session',
            model_type='OLS',
            sample_desc=f'Supergame {sg} only'
        ))

# ============================================================================
# STEP 12: LOGIT WITH CONTROLS
# ============================================================================

print("\n" + "="*60)
print("LOGIT WITH CONTROLS")
print("="*60)

# Logit with round control
model_logit_round = smf.logit('cooperate ~ is_team + round', data=df_all).fit(disp=0)
print(f"\nLogit with round: coef={model_logit_round.params['is_team']:.4f}")

results.append(extract_results(
    model_logit_round,
    spec_id='discrete/binary/logit_round',
    spec_tree_path='methods/discrete_choice.md#model-type-binary-outcome',
    outcome_var='cooperate',
    treatment_var='is_team',
    n_obs=len(df_all),
    controls_desc='round',
    fixed_effects='None',
    cluster_var='None',
    model_type='Logit'
))

# Logit with supergame control
model_logit_sg = smf.logit('cooperate ~ is_team + supergame', data=df_all).fit(disp=0)
print(f"Logit with supergame: coef={model_logit_sg.params['is_team']:.4f}")

results.append(extract_results(
    model_logit_sg,
    spec_id='discrete/binary/logit_supergame',
    spec_tree_path='methods/discrete_choice.md#model-type-binary-outcome',
    outcome_var='cooperate',
    treatment_var='is_team',
    n_obs=len(df_all),
    controls_desc='supergame',
    fixed_effects='None',
    cluster_var='None',
    model_type='Logit'
))

# Logit with full controls
model_logit_full = smf.logit('cooperate ~ is_team + round + supergame', data=df_all).fit(disp=0)
print(f"Logit with full controls: coef={model_logit_full.params['is_team']:.4f}")

results.append(extract_results(
    model_logit_full,
    spec_id='discrete/binary/logit_full',
    spec_tree_path='methods/discrete_choice.md#model-type-binary-outcome',
    outcome_var='cooperate',
    treatment_var='is_team',
    n_obs=len(df_all),
    controls_desc='round, supergame',
    fixed_effects='None',
    cluster_var='None',
    model_type='Logit'
))

# ============================================================================
# STEP 13: ALTERNATIVE OUTCOME - DEFECTION PATTERNS ANALYSIS
# ============================================================================

print("\n" + "="*60)
print("DEFECTION PATTERNS ANALYSIS")
print("="*60)

# Clean the defection patterns data
df_defect_clean = df_defect.copy()
df_defect_clean = df_defect_clean[df_defect_clean['Round in which Coop ends'] != 999]  # Remove censored
df_defect_clean = df_defect_clean[df_defect_clean['Round in which Coop ends'] != 0]  # Remove zeros

# Create team indicator
df_defect_clean['is_team'] = (df_defect_clean['Treatment'] == 2).astype(int)

# Outcome: Round in which cooperation ends
if len(df_defect_clean) > 10:
    model_defect = smf.ols('Q("Round in which Coop ends") ~ is_team', data=df_defect_clean).fit(cov_type='cluster', cov_kwds={'groups': df_defect_clean['Session']})
    print(f"\nRound coop ends on is_team: coef={model_defect.params['is_team']:.4f}, SE={model_defect.bse['is_team']:.4f}")

    results.append(extract_results(
        model_defect,
        spec_id='custom/defect_round',
        spec_tree_path='custom',
        outcome_var='Round in which Coop ends',
        treatment_var='is_team',
        n_obs=len(df_defect_clean),
        controls_desc='None',
        fixed_effects='None',
        cluster_var='Session',
        model_type='OLS',
        sample_desc='Uncensored defection patterns'
    ))

# ============================================================================
# STEP 14: QUADRATIC SPECIFICATIONS
# ============================================================================

print("\n" + "="*60)
print("QUADRATIC SPECIFICATIONS")
print("="*60)

# Quadratic in round
df_all['round_sq'] = df_all['round'] ** 2
model_quad_round = smf.ols('cooperate ~ is_team + round + round_sq', data=df_all).fit(cov_type='cluster', cov_kwds={'groups': df_all['session']})
print(f"\nQuadratic in round: is_team coef={model_quad_round.params['is_team']:.4f}")

results.append(extract_results(
    model_quad_round,
    spec_id='robust/form/quadratic_round',
    spec_tree_path='robustness/functional_form.md#nonlinear-specifications',
    outcome_var='cooperate',
    treatment_var='is_team',
    n_obs=len(df_all),
    controls_desc='round, round_sq',
    fixed_effects='None',
    cluster_var='session',
    model_type='OLS'
))

# Quadratic in supergame
df_all['supergame_sq'] = df_all['supergame'] ** 2
model_quad_sg = smf.ols('cooperate ~ is_team + supergame + supergame_sq', data=df_all).fit(cov_type='cluster', cov_kwds={'groups': df_all['session']})
print(f"Quadratic in supergame: is_team coef={model_quad_sg.params['is_team']:.4f}")

results.append(extract_results(
    model_quad_sg,
    spec_id='robust/form/quadratic_supergame',
    spec_tree_path='robustness/functional_form.md#nonlinear-specifications',
    outcome_var='cooperate',
    treatment_var='is_team',
    n_obs=len(df_all),
    controls_desc='supergame, supergame_sq',
    fixed_effects='None',
    cluster_var='session',
    model_type='OLS'
))

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
output_path = f'{OUTPUT_DIR}/specification_results.csv'
results_df.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")

# Print summary statistics
print(f"\nTotal specifications: {len(results_df)}")
print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({(results_df['coefficient'] > 0).mean()*100:.1f}%)")
print(f"Negative coefficients: {(results_df['coefficient'] < 0).sum()} ({(results_df['coefficient'] < 0).mean()*100:.1f}%)")
print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({(results_df['p_value'] < 0.05).mean()*100:.1f}%)")
print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({(results_df['p_value'] < 0.01).mean()*100:.1f}%)")
print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")
print(f"Range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")

# Print by category
print("\nBy specification category:")
results_df['category'] = results_df['spec_id'].apply(lambda x: x.split('/')[0] if '/' in x else 'baseline')
for cat, group in results_df.groupby('category'):
    sig_pct = (group['p_value'] < 0.05).mean() * 100
    print(f"  {cat}: n={len(group)}, {sig_pct:.1f}% significant at 5%")

print("\nSpecification search complete!")
