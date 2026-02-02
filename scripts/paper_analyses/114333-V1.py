"""
Specification Search for Paper 114333-V1
"Team versus Individual Play in Finitely Repeated Prisoner Dilemma Games"

This script runs 50+ specifications following the i4r methodology.
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Statistical packages
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Try to import pyfixest, fall back to statsmodels if not available
try:
    import pyfixest as pf
    HAS_PYFIXEST = True
except ImportError:
    HAS_PYFIXEST = False
    print("pyfixest not available, using statsmodels")

# ==============================================================================
# Configuration
# ==============================================================================

PAPER_ID = "114333-V1"
PAPER_TITLE = "Team versus Individual Play in Finitely Repeated Prisoner Dilemma Games"
JOURNAL = "AER"  # American Economic Review (2016 copyright date)

# Paths
BASE_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search"
DATA_PATH = f"{BASE_PATH}/data/downloads/extracted/{PAPER_ID}/Data_files_zip"
OUTPUT_PATH = f"{BASE_PATH}/data/downloads/extracted/{PAPER_ID}"
SCRIPT_PATH = f"{BASE_PATH}/scripts/paper_analyses"

# Treatment mapping (from defection patterns data)
INDIVIDUAL_SESSIONS = ['120131_1219', '120131_1338', '120210_1202', '120509_1153', '120509_1446']
TEAM_SESSIONS = ['120125_1258', '120125_1450', '120208_1601', '120507_1150', '120507_1457']

# ==============================================================================
# Data Loading and Preparation
# ==============================================================================

def load_and_prepare_data():
    """Load the choice data and add treatment indicators."""

    # Load main choice data
    df = pd.read_excel(f"{DATA_PATH}/Choice_Data_detailed.xlsx", sheet_name='All data')

    # Create treatment indicators
    # Treatment 1 = individuals (is_team=0), Treatment 2 = teams (is_team=1)
    df['is_team'] = df['session'].isin(TEAM_SESSIONS).astype(int)
    df['is_individual'] = df['session'].isin(INDIVIDUAL_SESSIONS).astype(int)

    # Create cooperation variable (realchoice 1 = cooperate, 2 = defect)
    df['cooperate'] = (df['realchoice'] == 1).astype(int)
    df['defect'] = (df['realchoice'] == 2).astype(int)

    # Create unique player identifier (session + team number)
    df['player_id'] = df['session'] + '_' + df['team'].astype(str)

    # Create numeric player_id for fixed effects
    df['player_num'] = pd.factorize(df['player_id'])[0]

    # Create numeric session_id
    df['session_num'] = pd.factorize(df['session'])[0]

    # Period within block (super-game round: 1-10)
    df['round_in_block'] = ((df['period'] - 1) % 10) + 1

    # Is this the final round of the super-game?
    df['is_final_round'] = (df['round_in_block'] == 10).astype(int)

    # Is this an early round (1-5) or late round (6-10)?
    df['is_late_round'] = (df['round_in_block'] > 5).astype(int)

    # Create outcome-based variables
    # Mutual cooperation indicator (outcome = 105)
    df['mutual_coop'] = (df['outcome'] == 105).astype(int)
    # Mutual defection indicator (outcome = 175)
    df['mutual_defect'] = (df['outcome'] == 175).astype(int)
    # Got suckered (cooperated but opponent defected, outcome = 5)
    df['got_suckered'] = (df['outcome'] == 5).astype(int)
    # Temptation (defected while opponent cooperated, outcome = 75)
    df['temptation'] = (df['outcome'] == 75).astype(int)

    # Lagged variables
    df = df.sort_values(['player_id', 'period'])
    df['lag_cooperate'] = df.groupby('player_id')['cooperate'].shift(1)
    df['lag_mutual_coop'] = df.groupby('player_id')['mutual_coop'].shift(1)
    df['lag_got_suckered'] = df.groupby('player_id')['got_suckered'].shift(1)

    # Opponent's last move (can be inferred from outcome and own choice)
    # If I cooperated and outcome=105, opponent cooperated
    # If I cooperated and outcome=5, opponent defected
    # If I defected and outcome=75, opponent cooperated
    # If I defected and outcome=175, opponent defected
    df['opponent_coop'] = ((df['outcome'] == 105) | (df['outcome'] == 75)).astype(int)
    df['lag_opponent_coop'] = df.groupby('player_id')['opponent_coop'].shift(1)

    # Experience variables
    df['cumulative_periods'] = df.groupby('player_id').cumcount() + 1
    df['log_period'] = np.log(df['period'] + 1)

    # Block indicators for early vs late super-games
    df['early_blocks'] = (df['block'] <= 5).astype(int)
    df['late_blocks'] = (df['block'] > 5).astype(int)

    return df


def load_defection_data():
    """Load the defection patterns data for alternative outcome analysis."""

    defect_df = pd.read_excel(f"{DATA_PATH}/defection-patterns.xlsx", sheet_name='Data')

    # Create treatment indicator (1 = individuals, 2 = teams)
    # Convert to is_team (0 = individuals, 1 = teams)
    defect_df['is_team'] = (defect_df['Treatment'] == 2).astype(int)

    # Clean the round variable (filter out 999 values)
    defect_df = defect_df[defect_df['Round in which Coop ends'] < 100].copy()

    # Create player_id for merging
    defect_df['player_id'] = defect_df['Session'] + '_' + defect_df['Player'].astype(str)

    return defect_df


# ==============================================================================
# Regression Functions
# ==============================================================================

def run_ols_regression(df, formula, cluster_var=None, weights=None):
    """Run OLS regression with optional clustering."""

    try:
        # Simply use the dataframe as-is, let statsmodels handle missing values
        df_clean = df.dropna()

        if cluster_var and cluster_var in df_clean.columns:
            model = smf.ols(formula, data=df_clean).fit(
                cov_type='cluster',
                cov_kwds={'groups': df_clean[cluster_var]}
            )
        else:
            model = smf.ols(formula, data=df_clean).fit(cov_type='HC1')
        return model
    except Exception as e:
        print(f"Error running regression: {e}")
        return None


def run_logit_regression(df, formula, cluster_var=None):
    """Run logistic regression."""

    try:
        if cluster_var and cluster_var in df_clean.columns:
            model = smf.logit(formula, data=df).fit(
                cov_type='cluster',
                cov_kwds={'groups': df[cluster_var]},
                disp=False
            )
        else:
            model = smf.logit(formula, data=df).fit(cov_type='HC1', disp=False)
        return model
    except Exception as e:
        print(f"Error running logit: {e}")
        return None


def extract_results(model, treatment_var, spec_id, spec_tree_path, outcome_var,
                    sample_desc, fixed_effects, controls_desc, cluster_var, model_type,
                    df=None):
    """Extract results from a regression model into standard format."""

    if model is None:
        return None

    try:
        # Get treatment coefficient
        if treatment_var in model.params.index:
            coef = model.params[treatment_var]
            se = model.bse[treatment_var]
            tstat = model.tvalues[treatment_var]
            pval = model.pvalues[treatment_var]
            ci = model.conf_int().loc[treatment_var]
            ci_lower, ci_upper = ci[0], ci[1]
        else:
            # Treatment variable not found
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
            "fixed_effects": fixed_effects.split(', ') if fixed_effects else [],
            "diagnostics": {}
        }

        # Add control coefficients
        for var in model.params.index:
            if var != treatment_var and var != 'Intercept':
                coef_vector["controls"].append({
                    "var": var,
                    "coef": float(model.params[var]),
                    "se": float(model.bse[var]),
                    "pval": float(model.pvalues[var])
                })

        # Get R-squared
        try:
            r_squared = float(model.rsquared)
        except:
            r_squared = None

        # Get N observations
        n_obs = int(model.nobs)

        result = {
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
            'n_obs': n_obs,
            'r_squared': r_squared,
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects,
            'controls_desc': controls_desc,
            'cluster_var': cluster_var if cluster_var else 'robust',
            'model_type': model_type,
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }

        return result

    except Exception as e:
        print(f"Error extracting results: {e}")
        return None


# ==============================================================================
# Specification Functions
# ==============================================================================

def run_baseline_specifications(df, results):
    """Run baseline and core specifications."""

    print("Running baseline specifications...")

    # BASELINE: Main effect of team on cooperation (with player and block FE)
    # This is likely the paper's main specification
    model = run_ols_regression(
        df,
        'cooperate ~ is_team + C(block) + C(round_in_block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'baseline',
        'methods/panel_fixed_effects.md',
        'cooperate', 'Full sample',
        'Block FE, Round FE', 'None',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    # Note: Cannot use session FE because treatment is constant within session
    # (each session is either all teams or all individuals)
    # Instead, we cluster by session for conservative SEs

    # Baseline - simple comparison (no FE)
    model = run_ols_regression(
        df,
        'cooperate ~ is_team',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'panel/fe/none',
        'methods/panel_fixed_effects.md',
        'cooperate', 'Full sample',
        'None', 'None',
        'player_id', 'OLS'
    )
    if result: results.append(result)

    return results


def run_control_variations(df, results):
    """Run control variable variations."""

    print("Running control variations...")

    controls = ['round_in_block', 'is_late_round', 'is_final_round', 'block',
                'log_period', 'lag_cooperate', 'lag_opponent_coop']

    # No controls
    model = run_ols_regression(
        df,
        'cooperate ~ is_team',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/control/none',
        'robustness/leave_one_out.md',
        'cooperate', 'Full sample',
        'None', 'No controls',
        'player_id', 'OLS'
    )
    if result: results.append(result)

    # Add round controls
    model = run_ols_regression(
        df,
        'cooperate ~ is_team + round_in_block',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/control/add_round',
        'robustness/control_progression.md',
        'cooperate', 'Full sample',
        'None', 'Round in block',
        'player_id', 'OLS'
    )
    if result: results.append(result)

    # Add block FE
    model = run_ols_regression(
        df,
        'cooperate ~ is_team + C(block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/control/add_block_fe',
        'robustness/control_progression.md',
        'cooperate', 'Full sample',
        'Block FE', 'Block FE only',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    # Add round FE
    model = run_ols_regression(
        df,
        'cooperate ~ is_team + C(round_in_block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/control/add_round_fe',
        'robustness/control_progression.md',
        'cooperate', 'Full sample',
        'Round FE', 'Round FE only',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    # Full FE specification
    model = run_ols_regression(
        df,
        'cooperate ~ is_team + C(block) + C(round_in_block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/control/full_fe',
        'robustness/control_progression.md',
        'cooperate', 'Full sample',
        'Block FE, Round FE', 'Block and Round FE',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    # With lagged cooperation (dynamic)
    df_lag = df.dropna(subset=['lag_cooperate'])
    model = run_ols_regression(
        df_lag,
        'cooperate ~ is_team + lag_cooperate + C(block) + C(round_in_block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/control/add_lag_coop',
        'robustness/control_progression.md',
        'cooperate', 'Full sample (excluding first round)',
        'Block FE, Round FE', 'Lagged cooperation',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    # With lagged opponent cooperation
    df_lag = df.dropna(subset=['lag_opponent_coop'])
    model = run_ols_regression(
        df_lag,
        'cooperate ~ is_team + lag_opponent_coop + C(block) + C(round_in_block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/control/add_lag_opponent',
        'robustness/control_progression.md',
        'cooperate', 'Full sample (excluding first round)',
        'Block FE, Round FE', 'Lagged opponent cooperation',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    # Full controls with lags
    df_lag = df.dropna(subset=['lag_cooperate', 'lag_opponent_coop'])
    model = run_ols_regression(
        df_lag,
        'cooperate ~ is_team + lag_cooperate + lag_opponent_coop + C(block) + C(round_in_block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/control/full_with_lags',
        'robustness/control_progression.md',
        'cooperate', 'Full sample (excluding first round)',
        'Block FE, Round FE', 'All lags',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    return results


def run_sample_restrictions(df, results):
    """Run sample restriction specifications."""

    print("Running sample restrictions...")

    # Early blocks (first 5 super-games)
    df_early = df[df['block'] <= 5]
    model = run_ols_regression(
        df_early,
        'cooperate ~ is_team + C(block) + C(round_in_block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/sample/early_blocks',
        'robustness/sample_restrictions.md',
        'cooperate', 'First 5 super-games',
        'Block FE, Round FE', 'None',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    # Late blocks (last 5 super-games)
    df_late = df[df['block'] > 5]
    model = run_ols_regression(
        df_late,
        'cooperate ~ is_team + C(block) + C(round_in_block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/sample/late_blocks',
        'robustness/sample_restrictions.md',
        'cooperate', 'Last 5 super-games',
        'Block FE, Round FE', 'None',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    # Early rounds within super-game (1-5)
    df_early_round = df[df['round_in_block'] <= 5]
    model = run_ols_regression(
        df_early_round,
        'cooperate ~ is_team + C(block) + C(round_in_block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/sample/early_rounds',
        'robustness/sample_restrictions.md',
        'cooperate', 'Rounds 1-5 within super-game',
        'Block FE, Round FE', 'None',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    # Late rounds within super-game (6-10)
    df_late_round = df[df['round_in_block'] > 5]
    model = run_ols_regression(
        df_late_round,
        'cooperate ~ is_team + C(block) + C(round_in_block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/sample/late_rounds',
        'robustness/sample_restrictions.md',
        'cooperate', 'Rounds 6-10 within super-game',
        'Block FE, Round FE', 'None',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    # Exclude final round (round 10)
    df_not_final = df[df['round_in_block'] < 10]
    model = run_ols_regression(
        df_not_final,
        'cooperate ~ is_team + C(block) + C(round_in_block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/sample/exclude_final_round',
        'robustness/sample_restrictions.md',
        'cooperate', 'Excluding final round',
        'Block FE, Round FE', 'None',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    # Only final round (round 10)
    df_final = df[df['round_in_block'] == 10]
    model = run_ols_regression(
        df_final,
        'cooperate ~ is_team + C(block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/sample/only_final_round',
        'robustness/sample_restrictions.md',
        'cooperate', 'Final round only',
        'Block FE', 'None',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    # Exclude first round (round 1)
    df_not_first = df[df['round_in_block'] > 1]
    model = run_ols_regression(
        df_not_first,
        'cooperate ~ is_team + C(block) + C(round_in_block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/sample/exclude_first_round',
        'robustness/sample_restrictions.md',
        'cooperate', 'Excluding first round',
        'Block FE, Round FE', 'None',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    # Only first super-game (block 1)
    df_first_block = df[df['block'] == 1]
    model = run_ols_regression(
        df_first_block,
        'cooperate ~ is_team + C(round_in_block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/sample/first_supergame_only',
        'robustness/sample_restrictions.md',
        'cooperate', 'First super-game only',
        'Round FE', 'None',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    # Exclude first super-game
    df_not_first_block = df[df['block'] > 1]
    model = run_ols_regression(
        df_not_first_block,
        'cooperate ~ is_team + C(block) + C(round_in_block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/sample/exclude_first_supergame',
        'robustness/sample_restrictions.md',
        'cooperate', 'Excluding first super-game',
        'Block FE, Round FE', 'None',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    # Drop each session one at a time
    for session in df['session'].unique():
        df_drop = df[df['session'] != session]
        model = run_ols_regression(
            df_drop,
            'cooperate ~ is_team + C(block) + C(round_in_block)',
            cluster_var='player_id'
        )
        result = extract_results(
            model, 'is_team', f'robust/sample/drop_session_{session}',
            'robustness/sample_restrictions.md',
            'cooperate', f'Excluding session {session}',
            'Block FE, Round FE', 'None',
            'player_id', 'OLS-FE'
        )
        if result: results.append(result)

    return results


def run_alternative_outcomes(df, results):
    """Run alternative outcome specifications."""

    print("Running alternative outcomes...")

    # Mutual cooperation (both players cooperate)
    model = run_ols_regression(
        df,
        'mutual_coop ~ is_team + C(block) + C(round_in_block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/outcome/mutual_coop',
        'robustness/measurement.md',
        'mutual_coop', 'Full sample',
        'Block FE, Round FE', 'None',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    # Mutual defection
    model = run_ols_regression(
        df,
        'mutual_defect ~ is_team + C(block) + C(round_in_block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/outcome/mutual_defect',
        'robustness/measurement.md',
        'mutual_defect', 'Full sample',
        'Block FE, Round FE', 'None',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    # Defection outcome
    model = run_ols_regression(
        df,
        'defect ~ is_team + C(block) + C(round_in_block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/outcome/defect',
        'robustness/measurement.md',
        'defect', 'Full sample',
        'Block FE, Round FE', 'None',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    # Got suckered (cooperated but opponent defected)
    model = run_ols_regression(
        df,
        'got_suckered ~ is_team + C(block) + C(round_in_block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/outcome/got_suckered',
        'robustness/measurement.md',
        'got_suckered', 'Full sample',
        'Block FE, Round FE', 'None',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    # Temptation payoff (defected while opponent cooperated)
    model = run_ols_regression(
        df,
        'temptation ~ is_team + C(block) + C(round_in_block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/outcome/temptation',
        'robustness/measurement.md',
        'temptation', 'Full sample',
        'Block FE, Round FE', 'None',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    # Outcome payoff (continuous)
    model = run_ols_regression(
        df,
        'outcome ~ is_team + C(block) + C(round_in_block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/outcome/payoff',
        'robustness/measurement.md',
        'outcome', 'Full sample',
        'Block FE, Round FE', 'None',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    return results


def run_inference_variations(df, results):
    """Run different clustering/inference specifications."""

    print("Running inference variations...")

    # Robust SEs (no clustering)
    model = smf.ols(
        'cooperate ~ is_team + C(block) + C(round_in_block)',
        data=df
    ).fit(cov_type='HC1')
    result = extract_results(
        model, 'is_team', 'robust/cluster/none',
        'robustness/clustering_variations.md',
        'cooperate', 'Full sample',
        'Block FE, Round FE', 'None',
        'robust', 'OLS-FE'
    )
    if result: results.append(result)

    # Cluster by session
    model = run_ols_regression(
        df,
        'cooperate ~ is_team + C(block) + C(round_in_block)',
        cluster_var='session'
    )
    result = extract_results(
        model, 'is_team', 'robust/cluster/session',
        'robustness/clustering_variations.md',
        'cooperate', 'Full sample',
        'Block FE, Round FE', 'None',
        'session', 'OLS-FE'
    )
    if result: results.append(result)

    # Cluster by player
    model = run_ols_regression(
        df,
        'cooperate ~ is_team + C(block) + C(round_in_block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/cluster/player',
        'robustness/clustering_variations.md',
        'cooperate', 'Full sample',
        'Block FE, Round FE', 'None',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    # Cluster by block
    model = run_ols_regression(
        df,
        'cooperate ~ is_team + C(block) + C(round_in_block)',
        cluster_var='block'
    )
    result = extract_results(
        model, 'is_team', 'robust/cluster/block',
        'robustness/clustering_variations.md',
        'cooperate', 'Full sample',
        'Block FE, Round FE', 'None',
        'block', 'OLS-FE'
    )
    if result: results.append(result)

    return results


def run_estimation_variations(df, results):
    """Run different estimation method variations."""

    print("Running estimation variations...")

    # Logit model
    try:
        model = smf.logit(
            'cooperate ~ is_team + C(block) + C(round_in_block)',
            data=df
        ).fit(disp=False)
        result = extract_results(
            model, 'is_team', 'discrete/binary/logit',
            'methods/discrete_choice.md',
            'cooperate', 'Full sample',
            'Block FE, Round FE', 'None',
            'robust', 'Logit'
        )
        if result: results.append(result)
    except Exception as e:
        print(f"Logit failed: {e}")

    # Probit model
    try:
        model = smf.probit(
            'cooperate ~ is_team + C(block) + C(round_in_block)',
            data=df
        ).fit(disp=False)
        result = extract_results(
            model, 'is_team', 'discrete/binary/probit',
            'methods/discrete_choice.md',
            'cooperate', 'Full sample',
            'Block FE, Round FE', 'None',
            'robust', 'Probit'
        )
        if result: results.append(result)
    except Exception as e:
        print(f"Probit failed: {e}")

    # Note: Cannot use session FE because treatment is constant within session
    # (each session is either all teams or all individuals)
    # is_team would be collinear with session fixed effects

    return results


def run_functional_form(df, results):
    """Run functional form variations."""

    print("Running functional form variations...")

    # Quadratic in round
    model = run_ols_regression(
        df,
        'cooperate ~ is_team + round_in_block + I(round_in_block**2) + C(block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/form/quadratic_round',
        'robustness/functional_form.md',
        'cooperate', 'Full sample',
        'Block FE', 'Quadratic in round',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    # Log period
    model = run_ols_regression(
        df,
        'cooperate ~ is_team + log_period + C(block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/form/log_period',
        'robustness/functional_form.md',
        'cooperate', 'Full sample',
        'Block FE', 'Log period',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    # Linear time trend
    model = run_ols_regression(
        df,
        'cooperate ~ is_team + period + C(block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/form/linear_trend',
        'robustness/functional_form.md',
        'cooperate', 'Full sample',
        'Block FE', 'Linear period trend',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    return results


def run_placebo_tests(df, results):
    """Run placebo tests."""

    print("Running placebo tests...")

    # Placebo: effect should be zero in round 1 (before any strategic interaction)
    df_round1 = df[df['round_in_block'] == 1]
    model = run_ols_regression(
        df_round1,
        'cooperate ~ is_team + C(block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/placebo/round1_only',
        'robustness/placebo_tests.md',
        'cooperate', 'Round 1 only (initial cooperation)',
        'Block FE', 'None',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    # Placebo: randomize treatment within session
    np.random.seed(42)
    df_placebo = df.copy()
    session_treatment = df_placebo.groupby('session')['is_team'].first().reset_index()
    session_treatment['fake_team'] = np.random.permutation(session_treatment['is_team'].values)
    df_placebo = df_placebo.merge(session_treatment[['session', 'fake_team']], on='session')

    model = run_ols_regression(
        df_placebo,
        'cooperate ~ fake_team + C(block) + C(round_in_block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'fake_team', 'robust/placebo/randomized_treatment',
        'robustness/placebo_tests.md',
        'cooperate', 'Full sample, randomized treatment',
        'Block FE, Round FE', 'None',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    return results


def run_heterogeneity_analyses(df, results):
    """Run heterogeneity analyses."""

    print("Running heterogeneity analyses...")

    # Heterogeneity by block (early vs late super-games)
    model = run_ols_regression(
        df,
        'cooperate ~ is_team * early_blocks + C(block) + C(round_in_block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/het/by_early_late_blocks',
        'robustness/heterogeneity.md',
        'cooperate', 'Full sample',
        'Block FE, Round FE', 'Interaction with early blocks',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    # Heterogeneity by round (early vs late rounds)
    model = run_ols_regression(
        df,
        'cooperate ~ is_team * is_late_round + C(block) + C(round_in_block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/het/by_early_late_rounds',
        'robustness/heterogeneity.md',
        'cooperate', 'Full sample',
        'Block FE, Round FE', 'Interaction with late rounds',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    # Heterogeneity by final round
    model = run_ols_regression(
        df,
        'cooperate ~ is_team * is_final_round + C(block) + C(round_in_block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/het/by_final_round',
        'robustness/heterogeneity.md',
        'cooperate', 'Full sample',
        'Block FE, Round FE', 'Interaction with final round',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    # Split by early vs late blocks
    df_early = df[df['block'] <= 5]
    model = run_ols_regression(
        df_early,
        'cooperate ~ is_team + C(block) + C(round_in_block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/het/early_blocks_only',
        'robustness/heterogeneity.md',
        'cooperate', 'Blocks 1-5 only',
        'Block FE, Round FE', 'None',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    df_late = df[df['block'] > 5]
    model = run_ols_regression(
        df_late,
        'cooperate ~ is_team + C(block) + C(round_in_block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/het/late_blocks_only',
        'robustness/heterogeneity.md',
        'cooperate', 'Blocks 6-10 only',
        'Block FE, Round FE', 'None',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    # Heterogeneity by each block
    for block_num in range(1, 11):
        df_block = df[df['block'] == block_num]
        model = run_ols_regression(
            df_block,
            'cooperate ~ is_team + C(round_in_block)',
            cluster_var='player_id'
        )
        result = extract_results(
            model, 'is_team', f'robust/het/block_{block_num}',
            'robustness/heterogeneity.md',
            'cooperate', f'Block {block_num} only',
            'Round FE', 'None',
            'player_id', 'OLS-FE'
        )
        if result: results.append(result)

    # Heterogeneity by each round within block
    for round_num in range(1, 11):
        df_round = df[df['round_in_block'] == round_num]
        model = run_ols_regression(
            df_round,
            'cooperate ~ is_team + C(block)',
            cluster_var='player_id'
        )
        result = extract_results(
            model, 'is_team', f'robust/het/round_{round_num}',
            'robustness/heterogeneity.md',
            'cooperate', f'Round {round_num} only',
            'Block FE', 'None',
            'player_id', 'OLS-FE'
        )
        if result: results.append(result)

    return results


def run_defection_timing_analysis(defect_df, results):
    """Run analysis on defection timing data."""

    print("Running defection timing analysis...")

    # Rename column for easier formula specification
    defect_df = defect_df.copy()
    defect_df['coop_end_round'] = defect_df['Round in which Coop ends']
    defect_df['super_game'] = defect_df['Super Game']
    defect_df['coop_path_num'] = defect_df['Coop Path Number']

    # Main outcome: Round in which cooperation ends
    try:
        model = smf.ols('coop_end_round ~ is_team', data=defect_df).fit(
            cov_type='cluster', cov_kwds={'groups': defect_df['player_id']}
        )
        result = extract_results(
            model, 'is_team', 'custom/defection_timing',
            'methods/panel_fixed_effects.md',
            'coop_end_round', 'Defection patterns data',
            'None', 'None',
            'player_id', 'OLS'
        )
        if result: results.append(result)
    except Exception as e:
        print(f"Defection timing failed: {e}")

    # With super-game control
    try:
        model = smf.ols('coop_end_round ~ is_team + super_game', data=defect_df).fit(
            cov_type='cluster', cov_kwds={'groups': defect_df['player_id']}
        )
        result = extract_results(
            model, 'is_team', 'custom/defection_timing_with_sg',
            'methods/panel_fixed_effects.md',
            'coop_end_round', 'Defection patterns data',
            'None', 'Super game control',
            'player_id', 'OLS'
        )
        if result: results.append(result)
    except Exception as e:
        print(f"Defection timing with SG failed: {e}")

    # With coop path number control
    try:
        model = smf.ols('coop_end_round ~ is_team + coop_path_num', data=defect_df).fit(
            cov_type='cluster', cov_kwds={'groups': defect_df['player_id']}
        )
        result = extract_results(
            model, 'is_team', 'custom/defection_timing_with_cooppath',
            'methods/panel_fixed_effects.md',
            'coop_end_round', 'Defection patterns data',
            'None', 'Coop path number',
            'player_id', 'OLS'
        )
        if result: results.append(result)
    except Exception as e:
        print(f"Defection timing with cooppath failed: {e}")

    return results


def run_additional_robustness(df, results):
    """Run additional robustness checks to reach 50+ specifications."""

    print("Running additional robustness checks...")

    # Middle blocks (3-7)
    df_mid = df[(df['block'] >= 3) & (df['block'] <= 7)]
    model = run_ols_regression(
        df_mid,
        'cooperate ~ is_team + C(block) + C(round_in_block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/sample/middle_blocks',
        'robustness/sample_restrictions.md',
        'cooperate', 'Blocks 3-7 only',
        'Block FE, Round FE', 'None',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    # Rounds 2-9 (excluding first and last)
    df_mid_round = df[(df['round_in_block'] >= 2) & (df['round_in_block'] <= 9)]
    model = run_ols_regression(
        df_mid_round,
        'cooperate ~ is_team + C(block) + C(round_in_block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/sample/middle_rounds',
        'robustness/sample_restrictions.md',
        'cooperate', 'Rounds 2-9 (excluding first and last)',
        'Block FE, Round FE', 'None',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    # Last 3 super-games only
    df_last3 = df[df['block'] >= 8]
    model = run_ols_regression(
        df_last3,
        'cooperate ~ is_team + C(block) + C(round_in_block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/sample/last_3_blocks',
        'robustness/sample_restrictions.md',
        'cooperate', 'Last 3 super-games',
        'Block FE, Round FE', 'None',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    # First 3 super-games only
    df_first3 = df[df['block'] <= 3]
    model = run_ols_regression(
        df_first3,
        'cooperate ~ is_team + C(block) + C(round_in_block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/sample/first_3_blocks',
        'robustness/sample_restrictions.md',
        'cooperate', 'First 3 super-games',
        'Block FE, Round FE', 'None',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    # After mutual cooperation (conditional on lag_mutual_coop)
    df_cond = df[df['lag_mutual_coop'] == 1].dropna()
    model = run_ols_regression(
        df_cond,
        'cooperate ~ is_team + C(block) + C(round_in_block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/sample/after_mutual_coop',
        'robustness/sample_restrictions.md',
        'cooperate', 'After mutual cooperation',
        'Block FE, Round FE', 'None',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    # After getting suckered (conditional on lag_got_suckered)
    df_cond = df[df['lag_got_suckered'] == 1].dropna()
    if len(df_cond) > 50:
        model = run_ols_regression(
            df_cond,
            'cooperate ~ is_team + C(block) + C(round_in_block)',
            cluster_var='player_id'
        )
        result = extract_results(
            model, 'is_team', 'robust/sample/after_got_suckered',
            'robustness/sample_restrictions.md',
            'cooperate', 'After getting suckered',
            'Block FE, Round FE', 'None',
            'player_id', 'OLS-FE'
        )
        if result: results.append(result)

    # Only observations where opponent cooperated last round
    df_cond = df[df['lag_opponent_coop'] == 1].dropna()
    model = run_ols_regression(
        df_cond,
        'cooperate ~ is_team + C(block) + C(round_in_block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/sample/opponent_coop_last',
        'robustness/sample_restrictions.md',
        'cooperate', 'After opponent cooperated',
        'Block FE, Round FE', 'None',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    # Only observations where opponent defected last round
    df_cond = df[df['lag_opponent_coop'] == 0].dropna()
    model = run_ols_regression(
        df_cond,
        'cooperate ~ is_team + C(block) + C(round_in_block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/sample/opponent_defect_last',
        'robustness/sample_restrictions.md',
        'cooperate', 'After opponent defected',
        'Block FE, Round FE', 'None',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    # Interaction with round number (continuous)
    model = run_ols_regression(
        df,
        'cooperate ~ is_team * round_in_block + C(block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/het/interaction_round',
        'robustness/heterogeneity.md',
        'cooperate', 'Full sample',
        'Block FE', 'Interaction with round',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    # Interaction with block number (continuous)
    model = run_ols_regression(
        df,
        'cooperate ~ is_team * block + C(round_in_block)',
        cluster_var='player_id'
    )
    result = extract_results(
        model, 'is_team', 'robust/het/interaction_block',
        'robustness/heterogeneity.md',
        'cooperate', 'Full sample',
        'Round FE', 'Interaction with block',
        'player_id', 'OLS-FE'
    )
    if result: results.append(result)

    return results


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    """Run all specifications and save results."""

    print(f"Starting specification search for {PAPER_ID}...")
    print("=" * 60)

    # Load data
    print("Loading data...")
    df = load_and_prepare_data()
    print(f"Loaded {len(df)} observations from choice data")

    defect_df = load_defection_data()
    print(f"Loaded {len(defect_df)} observations from defection patterns data")

    # Initialize results list
    results = []

    # Run all specification categories
    results = run_baseline_specifications(df, results)
    results = run_control_variations(df, results)
    results = run_sample_restrictions(df, results)
    results = run_alternative_outcomes(df, results)
    results = run_inference_variations(df, results)
    results = run_estimation_variations(df, results)
    results = run_functional_form(df, results)
    results = run_placebo_tests(df, results)
    results = run_heterogeneity_analyses(df, results)
    results = run_defection_timing_analysis(defect_df, results)
    results = run_additional_robustness(df, results)

    # Filter out None results
    results = [r for r in results if r is not None]

    print("=" * 60)
    print(f"Total specifications completed: {len(results)}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    output_file = f"{OUTPUT_PATH}/specification_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total specifications: {len(results_df)}")
    print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")
    print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
    print(f"Std coefficient: {results_df['coefficient'].std():.4f}")
    print(f"Min coefficient: {results_df['coefficient'].min():.4f}")
    print(f"Max coefficient: {results_df['coefficient'].max():.4f}")
    print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
    print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
    print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")

    return results_df


if __name__ == "__main__":
    results_df = main()
