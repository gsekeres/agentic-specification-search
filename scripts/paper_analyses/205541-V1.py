#!/usr/bin/env python3
"""
Specification Search: Paper 205541-V1 (AER)
Title: Beliefs and Cooperation in Repeated Prisoner's Dilemma Games

Paper Overview:
- Lab experiment studying cooperation in repeated prisoner's dilemma games
- Two game types: Finite (8 rounds) and Indefinite (random continuation)
- Elicits subjects' beliefs about opponent cooperation probability
- Main hypothesis: Beliefs affect cooperation decisions, and subjects are approximately Bayesian

Method Classification:
- method_code: discrete_choice
- method_tree_path: specification_tree/methods/discrete_choice.md
- The paper uses mixed-effects probit models for binary cooperation choices
- Also uses mixed-effects tobit for bounded belief data

Main outcome: coop (binary cooperation choice)
Key treatment/exposure: belief (stated belief about opponent cooperation)
Key moderators: finite (game type), late (late supergames), round
Clustering: session
Panel structure: Multiple observations per subject (id)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Setup paths and load data
# ============================================================================

BASE_PATH = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search'
DATA_PATH = f'{BASE_PATH}/data/downloads/extracted/205541-V1/data'
OUTPUT_PATH = f'{BASE_PATH}/data/downloads/extracted/205541-V1'

# Load main dataset
df = pd.read_stata(f'{DATA_PATH}/dat_1.dta')

# Paper metadata
PAPER_ID = '205541-V1'
JOURNAL = 'AER'
PAPER_TITLE = 'Beliefs and Cooperation in Repeated Prisoner''s Dilemma Games'

# ============================================================================
# Data preparation (following original Stata code)
# ============================================================================

# Create lagged variables for round 1 analysis
df_round1 = df[df['round'] == 1].copy()
df_round1 = df_round1.sort_values(['id', 'supergame'])

# First cooperation in supergame 1
df_round1['fcoop'] = df_round1.groupby('id').apply(
    lambda x: x[x['supergame']==1]['coop'].values[0] if 1 in x['supergame'].values else np.nan
).reindex(df_round1['id']).values

# Other's cooperation in previous supergame
df_round1['focoop_m1'] = df_round1.groupby('id')['o_coop'].shift(1)

# Length of previous supergame
df_round1['length_m1'] = df_round1.groupby('id')['length'].shift(1)

# Risk measure
df_round1['risk'] = df_round1['bomb_choice']

# For late supergames analysis
df_late = df[(df['late'] == 1) & (df['round'] <= 8)].copy()

# ============================================================================
# Results storage
# ============================================================================
results = []

def add_result(spec_id, spec_tree_path, outcome_var, treatment_var,
               model_result, df_used, cluster_var=None, controls_desc='',
               fixed_effects='None', sample_desc='Full sample', model_type='Probit',
               extra_info=None):
    """Helper function to add results in standardized format."""

    # Extract coefficient info
    try:
        coef = model_result.params[treatment_var]
        se = model_result.bse[treatment_var]
        pval = model_result.pvalues[treatment_var]
        tstat = coef / se
    except:
        # If treatment_var not in model (e.g., margins)
        coef = extra_info.get('coef', np.nan) if extra_info else np.nan
        se = extra_info.get('se', np.nan) if extra_info else np.nan
        pval = extra_info.get('pval', np.nan) if extra_info else np.nan
        tstat = coef / se if se > 0 else np.nan

    ci_lower = coef - 1.96 * se
    ci_upper = coef + 1.96 * se

    # Build coefficient vector
    coef_vector = {
        'treatment': {
            'var': treatment_var,
            'coef': float(coef),
            'se': float(se),
            'pval': float(pval)
        },
        'controls': [],
        'fixed_effects': [fixed_effects] if fixed_effects != 'None' else [],
        'diagnostics': {}
    }

    # Add control coefficients
    for var in model_result.params.index:
        if var not in [treatment_var, 'Intercept', 'const'] and not var.startswith('C('):
            try:
                coef_vector['controls'].append({
                    'var': var,
                    'coef': float(model_result.params[var]),
                    'se': float(model_result.bse[var]),
                    'pval': float(model_result.pvalues[var])
                })
            except:
                pass

    # Add diagnostics
    try:
        coef_vector['diagnostics']['pseudo_r2'] = float(model_result.prsquared) if hasattr(model_result, 'prsquared') else None
        coef_vector['diagnostics']['ll_model'] = float(model_result.llf) if hasattr(model_result, 'llf') else None
        coef_vector['diagnostics']['aic'] = float(model_result.aic) if hasattr(model_result, 'aic') else None
        coef_vector['diagnostics']['bic'] = float(model_result.bic) if hasattr(model_result, 'bic') else None
    except:
        pass

    n_obs = int(model_result.nobs) if hasattr(model_result, 'nobs') else len(df_used)

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
        'r_squared': coef_vector['diagnostics'].get('pseudo_r2'),
        'coefficient_vector_json': json.dumps(coef_vector),
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var if cluster_var else 'None',
        'model_type': model_type,
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    }

    results.append(result)
    return result

# ============================================================================
# BASELINE SPECIFICATIONS
# Replicating Table 6/7: Determinants of Cooperation in Round 1
# ============================================================================

print("="*70)
print("Running Baseline Specifications (Table 6/7 Replication)")
print("="*70)

# Note: The paper uses mixed-effects probit with random effects for subject
# and clustering at session level. Python's statsmodels has limited support
# for this, so we use GEE and regular probit with clustered SE as approximations.

# Baseline 1: Finite game, beliefon only
df_fin_r1 = df_round1[(df_round1['finite'] == 1) & df_round1['beliefon'].notna()].copy()

try:
    model = smf.probit('coop ~ beliefon', data=df_fin_r1).fit(
        cov_type='cluster', cov_kwds={'groups': df_fin_r1['session']}, disp=0)
    add_result(
        spec_id='baseline',
        spec_tree_path='methods/discrete_choice.md',
        outcome_var='coop',
        treatment_var='beliefon',
        model_result=model,
        df_used=df_fin_r1,
        cluster_var='session',
        controls_desc='None',
        fixed_effects='None',
        sample_desc='Finite game, round 1',
        model_type='Probit'
    )
    print(f"Baseline (Finite, beliefon only): coef={model.params['beliefon']:.4f}, p={model.pvalues['beliefon']:.4f}")
except Exception as e:
    print(f"Error in baseline finite: {e}")

# Baseline 2: Finite game with controls (main specification from paper)
df_fin_r1_full = df_fin_r1.dropna(subset=['beliefon', 'supergame', 'focoop_m1', 'fcoop', 'risk'])

try:
    model = smf.probit('coop ~ beliefon + supergame + focoop_m1 + fcoop + risk',
                       data=df_fin_r1_full).fit(
        cov_type='cluster', cov_kwds={'groups': df_fin_r1_full['session']}, disp=0)
    add_result(
        spec_id='baseline_full_controls',
        spec_tree_path='methods/discrete_choice.md',
        outcome_var='coop',
        treatment_var='beliefon',
        model_result=model,
        df_used=df_fin_r1_full,
        cluster_var='session',
        controls_desc='supergame, focoop_m1, fcoop, risk',
        fixed_effects='None',
        sample_desc='Finite game, round 1, complete cases',
        model_type='Probit'
    )
    print(f"Baseline full (Finite): coef={model.params['beliefon']:.4f}, p={model.pvalues['beliefon']:.4f}")
except Exception as e:
    print(f"Error in baseline finite full: {e}")

# Baseline 3: Indefinite game, beliefon only
df_inf_r1 = df_round1[(df_round1['finite'] == 0) & df_round1['beliefon'].notna()].copy()

try:
    model = smf.probit('coop ~ beliefon', data=df_inf_r1).fit(
        cov_type='cluster', cov_kwds={'groups': df_inf_r1['session']}, disp=0)
    add_result(
        spec_id='baseline_indefinite',
        spec_tree_path='methods/discrete_choice.md',
        outcome_var='coop',
        treatment_var='beliefon',
        model_result=model,
        df_used=df_inf_r1,
        cluster_var='session',
        controls_desc='None',
        fixed_effects='None',
        sample_desc='Indefinite game, round 1',
        model_type='Probit'
    )
    print(f"Baseline (Indefinite, beliefon only): coef={model.params['beliefon']:.4f}, p={model.pvalues['beliefon']:.4f}")
except Exception as e:
    print(f"Error in baseline indefinite: {e}")

# Baseline 4: Indefinite game with controls
df_inf_r1_full = df_inf_r1.dropna(subset=['beliefon', 'supergame', 'focoop_m1', 'fcoop', 'risk', 'length_m1'])

try:
    model = smf.probit('coop ~ beliefon + supergame + focoop_m1 + fcoop + risk + length_m1',
                       data=df_inf_r1_full).fit(
        cov_type='cluster', cov_kwds={'groups': df_inf_r1_full['session']}, disp=0)
    add_result(
        spec_id='baseline_indefinite_full',
        spec_tree_path='methods/discrete_choice.md',
        outcome_var='coop',
        treatment_var='beliefon',
        model_result=model,
        df_used=df_inf_r1_full,
        cluster_var='session',
        controls_desc='supergame, focoop_m1, fcoop, risk, length_m1',
        fixed_effects='None',
        sample_desc='Indefinite game, round 1, complete cases',
        model_type='Probit'
    )
    print(f"Baseline full (Indefinite): coef={model.params['beliefon']:.4f}, p={model.pvalues['beliefon']:.4f}")
except Exception as e:
    print(f"Error in baseline indefinite full: {e}")

# ============================================================================
# CORE DISCRETE CHOICE VARIATIONS
# ============================================================================

print("\n" + "="*70)
print("Running Model Type Variations")
print("="*70)

# Logit model
try:
    model = smf.logit('coop ~ beliefon + supergame + focoop_m1 + fcoop + risk',
                      data=df_fin_r1_full).fit(
        cov_type='cluster', cov_kwds={'groups': df_fin_r1_full['session']}, disp=0)
    add_result(
        spec_id='discrete/binary/logit',
        spec_tree_path='methods/discrete_choice.md#model-type-binary-outcome',
        outcome_var='coop',
        treatment_var='beliefon',
        model_result=model,
        df_used=df_fin_r1_full,
        cluster_var='session',
        controls_desc='supergame, focoop_m1, fcoop, risk',
        sample_desc='Finite game, round 1',
        model_type='Logit'
    )
    print(f"Logit (Finite): coef={model.params['beliefon']:.4f}, p={model.pvalues['beliefon']:.4f}")
except Exception as e:
    print(f"Error in logit: {e}")

# Linear Probability Model
try:
    model = smf.ols('coop ~ beliefon + supergame + focoop_m1 + fcoop + risk',
                    data=df_fin_r1_full).fit(
        cov_type='cluster', cov_kwds={'groups': df_fin_r1_full['session']})
    add_result(
        spec_id='discrete/binary/lpm',
        spec_tree_path='methods/discrete_choice.md#model-type-binary-outcome',
        outcome_var='coop',
        treatment_var='beliefon',
        model_result=model,
        df_used=df_fin_r1_full,
        cluster_var='session',
        controls_desc='supergame, focoop_m1, fcoop, risk',
        sample_desc='Finite game, round 1',
        model_type='LPM'
    )
    print(f"LPM (Finite): coef={model.params['beliefon']:.4f}, p={model.pvalues['beliefon']:.4f}")
except Exception as e:
    print(f"Error in LPM: {e}")

# Indefinite game - Logit
try:
    model = smf.logit('coop ~ beliefon + supergame + focoop_m1 + fcoop + risk + length_m1',
                      data=df_inf_r1_full).fit(
        cov_type='cluster', cov_kwds={'groups': df_inf_r1_full['session']}, disp=0)
    add_result(
        spec_id='discrete/binary/logit_indefinite',
        spec_tree_path='methods/discrete_choice.md#model-type-binary-outcome',
        outcome_var='coop',
        treatment_var='beliefon',
        model_result=model,
        df_used=df_inf_r1_full,
        cluster_var='session',
        controls_desc='supergame, focoop_m1, fcoop, risk, length_m1',
        sample_desc='Indefinite game, round 1',
        model_type='Logit'
    )
    print(f"Logit (Indefinite): coef={model.params['beliefon']:.4f}, p={model.pvalues['beliefon']:.4f}")
except Exception as e:
    print(f"Error in logit indefinite: {e}")

# LPM - Indefinite
try:
    model = smf.ols('coop ~ beliefon + supergame + focoop_m1 + fcoop + risk + length_m1',
                    data=df_inf_r1_full).fit(
        cov_type='cluster', cov_kwds={'groups': df_inf_r1_full['session']})
    add_result(
        spec_id='discrete/binary/lpm_indefinite',
        spec_tree_path='methods/discrete_choice.md#model-type-binary-outcome',
        outcome_var='coop',
        treatment_var='beliefon',
        model_result=model,
        df_used=df_inf_r1_full,
        cluster_var='session',
        controls_desc='supergame, focoop_m1, fcoop, risk, length_m1',
        sample_desc='Indefinite game, round 1',
        model_type='LPM'
    )
    print(f"LPM (Indefinite): coef={model.params['beliefon']:.4f}, p={model.pvalues['beliefon']:.4f}")
except Exception as e:
    print(f"Error in LPM indefinite: {e}")

# ============================================================================
# ALTERNATIVE OUTCOME: BELIEF AS DEPENDENT VARIABLE
# Testing belief accuracy (belief - o_coop)
# ============================================================================

print("\n" + "="*70)
print("Running Belief Accuracy Specifications")
print("="*70)

# Create belief error variable
df_late['belief_error'] = df_late['belief'] - df_late['o_coop']

# OLS regression on belief error
df_late_finite = df_late[df_late['finite'] == 1].copy()
df_late_indefinite = df_late[df_late['finite'] == 0].copy()

try:
    model = smf.ols('belief_error ~ round', data=df_late_finite.dropna(subset=['belief_error', 'round'])).fit(
        cov_type='cluster', cov_kwds={'groups': df_late_finite.dropna(subset=['belief_error', 'round'])['session']})
    add_result(
        spec_id='belief_accuracy/finite',
        spec_tree_path='methods/discrete_choice.md',
        outcome_var='belief_error',
        treatment_var='round',
        model_result=model,
        df_used=df_late_finite,
        cluster_var='session',
        controls_desc='None',
        sample_desc='Finite game, late supergames, rounds 1-8',
        model_type='OLS'
    )
    print(f"Belief error by round (Finite): coef={model.params['round']:.4f}, p={model.pvalues['round']:.4f}")
except Exception as e:
    print(f"Error in belief accuracy finite: {e}")

try:
    model = smf.ols('belief_error ~ round', data=df_late_indefinite.dropna(subset=['belief_error', 'round'])).fit(
        cov_type='cluster', cov_kwds={'groups': df_late_indefinite.dropna(subset=['belief_error', 'round'])['session']})
    add_result(
        spec_id='belief_accuracy/indefinite',
        spec_tree_path='methods/discrete_choice.md',
        outcome_var='belief_error',
        treatment_var='round',
        model_result=model,
        df_used=df_late_indefinite,
        cluster_var='session',
        controls_desc='None',
        sample_desc='Indefinite game, late supergames, rounds 1-8',
        model_type='OLS'
    )
    print(f"Belief error by round (Indefinite): coef={model.params['round']:.4f}, p={model.pvalues['round']:.4f}")
except Exception as e:
    print(f"Error in belief accuracy indefinite: {e}")

# ============================================================================
# CONTROL VARIATIONS
# ============================================================================

print("\n" + "="*70)
print("Running Control Variations")
print("="*70)

# No controls
try:
    model = smf.probit('coop ~ beliefon', data=df_fin_r1_full).fit(
        cov_type='cluster', cov_kwds={'groups': df_fin_r1_full['session']}, disp=0)
    add_result(
        spec_id='discrete/controls/none',
        spec_tree_path='methods/discrete_choice.md#control-sets',
        outcome_var='coop',
        treatment_var='beliefon',
        model_result=model,
        df_used=df_fin_r1_full,
        cluster_var='session',
        controls_desc='None',
        sample_desc='Finite game, round 1',
        model_type='Probit'
    )
    print(f"No controls (Finite): coef={model.params['beliefon']:.4f}, p={model.pvalues['beliefon']:.4f}")
except Exception as e:
    print(f"Error: {e}")

# ============================================================================
# SAMPLE RESTRICTIONS
# ============================================================================

print("\n" + "="*70)
print("Running Sample Restrictions")
print("="*70)

# Early vs Late supergames
df_early = df_round1[(df_round1['early'] == 1) & df_round1['beliefon'].notna()].copy()
df_early_full = df_early.dropna(subset=['beliefon', 'supergame', 'focoop_m1', 'fcoop', 'risk'])

# Early supergames - Finite
df_early_fin = df_early_full[df_early_full['finite'] == 1]
if len(df_early_fin) > 10:
    try:
        model = smf.probit('coop ~ beliefon + supergame + focoop_m1 + fcoop + risk',
                           data=df_early_fin).fit(
            cov_type='cluster', cov_kwds={'groups': df_early_fin['session']}, disp=0)
        add_result(
            spec_id='robust/sample/early_period',
            spec_tree_path='robustness/sample_restrictions.md',
            outcome_var='coop',
            treatment_var='beliefon',
            model_result=model,
            df_used=df_early_fin,
            cluster_var='session',
            controls_desc='supergame, focoop_m1, fcoop, risk',
            sample_desc='Finite game, early supergames, round 1',
            model_type='Probit'
        )
        print(f"Early supergames (Finite): coef={model.params['beliefon']:.4f}, p={model.pvalues['beliefon']:.4f}")
    except Exception as e:
        print(f"Error: {e}")

# Late supergames only (round 1)
df_late_r1 = df_round1[(df_round1['late'] == 1) & df_round1['beliefon'].notna()].copy()
df_late_r1_full = df_late_r1.dropna(subset=['beliefon', 'supergame', 'focoop_m1', 'fcoop', 'risk'])

df_late_r1_fin = df_late_r1_full[df_late_r1_full['finite'] == 1]
if len(df_late_r1_fin) > 10:
    try:
        model = smf.probit('coop ~ beliefon + supergame + focoop_m1 + fcoop + risk',
                           data=df_late_r1_fin).fit(
            cov_type='cluster', cov_kwds={'groups': df_late_r1_fin['session']}, disp=0)
        add_result(
            spec_id='robust/sample/late_period',
            spec_tree_path='robustness/sample_restrictions.md',
            outcome_var='coop',
            treatment_var='beliefon',
            model_result=model,
            df_used=df_late_r1_fin,
            cluster_var='session',
            controls_desc='supergame, focoop_m1, fcoop, risk',
            sample_desc='Finite game, late supergames, round 1',
            model_type='Probit'
        )
        print(f"Late supergames (Finite): coef={model.params['beliefon']:.4f}, p={model.pvalues['beliefon']:.4f}")
    except Exception as e:
        print(f"Error: {e}")

# First 4 supergames only
df_first4 = df_round1[(df_round1['supergame'] <= 4) & df_round1['beliefon'].notna()].copy()
df_first4_fin = df_first4[df_first4['finite'] == 1].dropna(subset=['beliefon', 'focoop_m1', 'fcoop', 'risk'])

if len(df_first4_fin) > 10:
    try:
        model = smf.probit('coop ~ beliefon + supergame + focoop_m1 + fcoop + risk',
                           data=df_first4_fin).fit(
            cov_type='cluster', cov_kwds={'groups': df_first4_fin['session']}, disp=0)
        add_result(
            spec_id='robust/sample/first_4_supergames',
            spec_tree_path='robustness/sample_restrictions.md',
            outcome_var='coop',
            treatment_var='beliefon',
            model_result=model,
            df_used=df_first4_fin,
            cluster_var='session',
            controls_desc='supergame, focoop_m1, fcoop, risk',
            sample_desc='Finite game, first 4 supergames, round 1',
            model_type='Probit'
        )
        print(f"First 4 supergames (Finite): coef={model.params['beliefon']:.4f}, p={model.pvalues['beliefon']:.4f}")
    except Exception as e:
        print(f"Error: {e}")

# ============================================================================
# LEAVE-ONE-OUT ROBUSTNESS
# ============================================================================

print("\n" + "="*70)
print("Running Leave-One-Out Specifications")
print("="*70)

controls_fin = ['supergame', 'focoop_m1', 'fcoop', 'risk']

for drop_var in controls_fin:
    remaining = [c for c in controls_fin if c != drop_var]
    formula = f'coop ~ beliefon + {" + ".join(remaining)}'

    try:
        model = smf.probit(formula, data=df_fin_r1_full).fit(
            cov_type='cluster', cov_kwds={'groups': df_fin_r1_full['session']}, disp=0)
        add_result(
            spec_id=f'robust/loo/drop_{drop_var}',
            spec_tree_path='robustness/leave_one_out.md',
            outcome_var='coop',
            treatment_var='beliefon',
            model_result=model,
            df_used=df_fin_r1_full,
            cluster_var='session',
            controls_desc=', '.join(remaining),
            sample_desc='Finite game, round 1',
            model_type='Probit'
        )
        print(f"LOO drop {drop_var}: coef={model.params['beliefon']:.4f}, p={model.pvalues['beliefon']:.4f}")
    except Exception as e:
        print(f"Error dropping {drop_var}: {e}")

# ============================================================================
# SINGLE COVARIATE SPECIFICATIONS
# ============================================================================

print("\n" + "="*70)
print("Running Single Covariate Specifications")
print("="*70)

# Bivariate (already run above as no controls)

for single_var in controls_fin:
    formula = f'coop ~ beliefon + {single_var}'

    try:
        model = smf.probit(formula, data=df_fin_r1_full).fit(
            cov_type='cluster', cov_kwds={'groups': df_fin_r1_full['session']}, disp=0)
        add_result(
            spec_id=f'robust/single/{single_var}',
            spec_tree_path='robustness/single_covariate.md',
            outcome_var='coop',
            treatment_var='beliefon',
            model_result=model,
            df_used=df_fin_r1_full,
            cluster_var='session',
            controls_desc=single_var,
            sample_desc='Finite game, round 1',
            model_type='Probit'
        )
        print(f"Single {single_var}: coef={model.params['beliefon']:.4f}, p={model.pvalues['beliefon']:.4f}")
    except Exception as e:
        print(f"Error single {single_var}: {e}")

# ============================================================================
# CLUSTERING VARIATIONS
# ============================================================================

print("\n" + "="*70)
print("Running Clustering Variations")
print("="*70)

# Robust SE only (no clustering)
try:
    model = smf.probit('coop ~ beliefon + supergame + focoop_m1 + fcoop + risk',
                       data=df_fin_r1_full).fit(cov_type='HC1', disp=0)
    add_result(
        spec_id='robust/cluster/none',
        spec_tree_path='robustness/clustering_variations.md',
        outcome_var='coop',
        treatment_var='beliefon',
        model_result=model,
        df_used=df_fin_r1_full,
        cluster_var='None (Robust SE)',
        controls_desc='supergame, focoop_m1, fcoop, risk',
        sample_desc='Finite game, round 1',
        model_type='Probit'
    )
    print(f"Robust SE (Finite): coef={model.params['beliefon']:.4f}, p={model.pvalues['beliefon']:.4f}")
except Exception as e:
    print(f"Error: {e}")

# Cluster by id (individual)
try:
    model = smf.probit('coop ~ beliefon + supergame + focoop_m1 + fcoop + risk',
                       data=df_fin_r1_full).fit(
        cov_type='cluster', cov_kwds={'groups': df_fin_r1_full['id']}, disp=0)
    add_result(
        spec_id='robust/cluster/unit',
        spec_tree_path='robustness/clustering_variations.md',
        outcome_var='coop',
        treatment_var='beliefon',
        model_result=model,
        df_used=df_fin_r1_full,
        cluster_var='id',
        controls_desc='supergame, focoop_m1, fcoop, risk',
        sample_desc='Finite game, round 1',
        model_type='Probit'
    )
    print(f"Cluster by id (Finite): coef={model.params['beliefon']:.4f}, p={model.pvalues['beliefon']:.4f}")
except Exception as e:
    print(f"Error: {e}")

# ============================================================================
# ALTERNATIVE OUTCOME: BELIEF DIRECTLY
# ============================================================================

print("\n" + "="*70)
print("Running Belief as Outcome Specifications")
print("="*70)

# Test if belief differs by game type (finite vs indefinite)
# Using late supergames, round 1
df_late_r1_both = df[(df['late'] == 1) & (df['round'] == 1)].dropna(subset=['belief', 'finite'])

try:
    model = smf.ols('belief ~ finite', data=df_late_r1_both).fit(
        cov_type='cluster', cov_kwds={'groups': df_late_r1_both['session']})
    add_result(
        spec_id='belief_outcome/finite_effect',
        spec_tree_path='methods/discrete_choice.md',
        outcome_var='belief',
        treatment_var='finite',
        model_result=model,
        df_used=df_late_r1_both,
        cluster_var='session',
        controls_desc='None',
        sample_desc='Late supergames, round 1',
        model_type='OLS'
    )
    print(f"Belief ~ finite: coef={model.params['finite']:.4f}, p={model.pvalues['finite']:.4f}")
except Exception as e:
    print(f"Error: {e}")

# ============================================================================
# MAIN HYPOTHESIS: BELIEF AFFECTS COOPERATION
# Testing whether beliefs predict cooperation
# ============================================================================

print("\n" + "="*70)
print("Running Main Hypothesis: Belief -> Cooperation")
print("="*70)

# Finite game, all rounds with beliefs
df_finite_all = df[(df['finite'] == 1) & df['belief'].notna() & df['late'] == 1].copy()
df_finite_all['coop_m1_x_belief'] = df_finite_all['coop_m1'] * df_finite_all['belief']

# Belief predicting cooperation (main hypothesis)
try:
    model = smf.probit('coop ~ belief', data=df_finite_all).fit(
        cov_type='cluster', cov_kwds={'groups': df_finite_all['session']}, disp=0)
    add_result(
        spec_id='main_hypothesis/belief_coop_finite',
        spec_tree_path='methods/discrete_choice.md',
        outcome_var='coop',
        treatment_var='belief',
        model_result=model,
        df_used=df_finite_all,
        cluster_var='session',
        controls_desc='None',
        sample_desc='Finite game, late supergames, all rounds with beliefs',
        model_type='Probit'
    )
    print(f"Belief -> Coop (Finite): coef={model.params['belief']:.4f}, p={model.pvalues['belief']:.4f}")
except Exception as e:
    print(f"Error: {e}")

# With round controls
try:
    model = smf.probit('coop ~ belief + round', data=df_finite_all).fit(
        cov_type='cluster', cov_kwds={'groups': df_finite_all['session']}, disp=0)
    add_result(
        spec_id='main_hypothesis/belief_coop_finite_round',
        spec_tree_path='methods/discrete_choice.md',
        outcome_var='coop',
        treatment_var='belief',
        model_result=model,
        df_used=df_finite_all,
        cluster_var='session',
        controls_desc='round',
        sample_desc='Finite game, late supergames, all rounds with beliefs',
        model_type='Probit'
    )
    print(f"Belief -> Coop + round (Finite): coef={model.params['belief']:.4f}, p={model.pvalues['belief']:.4f}")
except Exception as e:
    print(f"Error: {e}")

# Indefinite game
df_indefinite_all = df[(df['finite'] == 0) & df['belief'].notna() & df['late'] == 1].copy()

try:
    model = smf.probit('coop ~ belief', data=df_indefinite_all).fit(
        cov_type='cluster', cov_kwds={'groups': df_indefinite_all['session']}, disp=0)
    add_result(
        spec_id='main_hypothesis/belief_coop_indefinite',
        spec_tree_path='methods/discrete_choice.md',
        outcome_var='coop',
        treatment_var='belief',
        model_result=model,
        df_used=df_indefinite_all,
        cluster_var='session',
        controls_desc='None',
        sample_desc='Indefinite game, late supergames, all rounds with beliefs',
        model_type='Probit'
    )
    print(f"Belief -> Coop (Indefinite): coef={model.params['belief']:.4f}, p={model.pvalues['belief']:.4f}")
except Exception as e:
    print(f"Error: {e}")

# With round controls
try:
    model = smf.probit('coop ~ belief + round', data=df_indefinite_all).fit(
        cov_type='cluster', cov_kwds={'groups': df_indefinite_all['session']}, disp=0)
    add_result(
        spec_id='main_hypothesis/belief_coop_indefinite_round',
        spec_tree_path='methods/discrete_choice.md',
        outcome_var='coop',
        treatment_var='belief',
        model_result=model,
        df_used=df_indefinite_all,
        cluster_var='session',
        controls_desc='round',
        sample_desc='Indefinite game, late supergames, all rounds with beliefs',
        model_type='Probit'
    )
    print(f"Belief -> Coop + round (Indefinite): coef={model.params['belief']:.4f}, p={model.pvalues['belief']:.4f}")
except Exception as e:
    print(f"Error: {e}")

# ============================================================================
# POOLED SPECIFICATIONS
# ============================================================================

print("\n" + "="*70)
print("Running Pooled Specifications")
print("="*70)

# Pooled: both game types
df_pooled = df[df['belief'].notna() & (df['late'] == 1)].copy()

try:
    model = smf.probit('coop ~ belief + finite', data=df_pooled).fit(
        cov_type='cluster', cov_kwds={'groups': df_pooled['session']}, disp=0)
    add_result(
        spec_id='pooled/belief_coop_both_games',
        spec_tree_path='methods/discrete_choice.md',
        outcome_var='coop',
        treatment_var='belief',
        model_result=model,
        df_used=df_pooled,
        cluster_var='session',
        controls_desc='finite',
        sample_desc='Both games, late supergames, all rounds with beliefs',
        model_type='Probit'
    )
    print(f"Pooled (both games): coef={model.params['belief']:.4f}, p={model.pvalues['belief']:.4f}")
except Exception as e:
    print(f"Error: {e}")

# With interaction
try:
    model = smf.probit('coop ~ belief * finite + round', data=df_pooled).fit(
        cov_type='cluster', cov_kwds={'groups': df_pooled['session']}, disp=0)
    add_result(
        spec_id='pooled/belief_coop_interaction',
        spec_tree_path='methods/discrete_choice.md',
        outcome_var='coop',
        treatment_var='belief',
        model_result=model,
        df_used=df_pooled,
        cluster_var='session',
        controls_desc='finite, round, belief*finite',
        sample_desc='Both games, late supergames, all rounds with beliefs',
        model_type='Probit'
    )
    print(f"Pooled with interaction: coef={model.params['belief']:.4f}, p={model.pvalues['belief']:.4f}")
except Exception as e:
    print(f"Error: {e}")

# ============================================================================
# ROUND-BY-ROUND ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("Running Round-by-Round Specifications")
print("="*70)

for round_num in range(1, 9):
    df_round = df[(df['finite'] == 1) & (df['late'] == 1) & (df['round'] == round_num) & df['belief'].notna()].copy()

    if len(df_round) > 20:
        try:
            model = smf.probit('coop ~ belief', data=df_round).fit(
                cov_type='cluster', cov_kwds={'groups': df_round['session']}, disp=0)
            add_result(
                spec_id=f'round_analysis/finite_round_{round_num}',
                spec_tree_path='methods/discrete_choice.md',
                outcome_var='coop',
                treatment_var='belief',
                model_result=model,
                df_used=df_round,
                cluster_var='session',
                controls_desc='None',
                sample_desc=f'Finite game, late supergames, round {round_num}',
                model_type='Probit'
            )
            print(f"Finite Round {round_num}: coef={model.params['belief']:.4f}, p={model.pvalues['belief']:.4f}, n={len(df_round)}")
        except Exception as e:
            print(f"Error round {round_num}: {e}")

# ============================================================================
# Save results
# ============================================================================

print("\n" + "="*70)
print("Saving Results")
print("="*70)

# Convert to DataFrame and save
results_df = pd.DataFrame(results)
results_df.to_csv(f'{OUTPUT_PATH}/specification_results.csv', index=False)

print(f"\nTotal specifications run: {len(results)}")
print(f"Results saved to: {OUTPUT_PATH}/specification_results.csv")

# ============================================================================
# Summary statistics
# ============================================================================

print("\n" + "="*70)
print("Summary Statistics")
print("="*70)

# Filter to main hypothesis specifications (belief -> coop)
main_results = results_df[results_df['treatment_var'] == 'belief']

print(f"\nFor belief -> cooperation specifications:")
print(f"  Total specifications: {len(main_results)}")
print(f"  Positive coefficients: {sum(main_results['coefficient'] > 0)} ({100*sum(main_results['coefficient'] > 0)/len(main_results):.1f}%)")
print(f"  Significant at 5%: {sum(main_results['p_value'] < 0.05)} ({100*sum(main_results['p_value'] < 0.05)/len(main_results):.1f}%)")
print(f"  Significant at 1%: {sum(main_results['p_value'] < 0.01)} ({100*sum(main_results['p_value'] < 0.01)/len(main_results):.1f}%)")
print(f"  Median coefficient: {main_results['coefficient'].median():.4f}")
print(f"  Mean coefficient: {main_results['coefficient'].mean():.4f}")
print(f"  Range: [{main_results['coefficient'].min():.4f}, {main_results['coefficient'].max():.4f}]")

# All results
print(f"\nFor all specifications:")
print(f"  Total specifications: {len(results_df)}")
print(f"  Positive coefficients: {sum(results_df['coefficient'] > 0)} ({100*sum(results_df['coefficient'] > 0)/len(results_df):.1f}%)")
print(f"  Significant at 5%: {sum(results_df['p_value'] < 0.05)} ({100*sum(results_df['p_value'] < 0.05)/len(results_df):.1f}%)")
