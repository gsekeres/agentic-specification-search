"""
Specification Search: Paper 205541-V1
=====================================
Title: Cooperation and Beliefs in Games with Repeated Interaction
Topic: Experimental economics - cooperation in finitely and indefinitely repeated
       prisoner's dilemma games with belief elicitation
Method: Discrete choice models (probit/logit) with correlated random effects

Key variables:
- Outcome: coop (cooperation decision, binary 0/1)
- Treatment: finite (finite vs indefinite game horizon), beliefon (beliefs elicited)
- Controls: supergame, focoop_m1 (other cooperated in previous supergame),
            fcoop (cooperated in first supergame), risk (bomb choice), length_m1
- Clustering: session
- Panel: subject (id), repeated measures across supergames and rounds
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Paths
BASE_PATH = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search'
DATA_PATH = f'{BASE_PATH}/data/downloads/extracted/205541-V1/data'
OUTPUT_PATH = f'{BASE_PATH}/data/downloads/extracted/205541-V1'

# Paper metadata
PAPER_ID = '205541-V1'
JOURNAL = 'AER'
PAPER_TITLE = 'Cooperation and Beliefs in Games with Repeated Interaction'

# Load data
print("Loading data...")
df = pd.read_stata(f'{DATA_PATH}/dat_1.dta')
df_wtypes = pd.read_stata(f'{DATA_PATH}/dat_wtypes.dta')
df_combined = pd.read_stata(f'{DATA_PATH}/dat_combined.dta')

# Data preparation - replicate variable construction from est_2.do
def prepare_data(df):
    """Prepare data for analysis following original Stata code"""
    df = df.copy()

    # Create first supergame cooperation indicator
    foo = df.loc[df['round'] == 1].loc[df['supergame'] == 1, ['id', 'coop']].rename(columns={'coop': 'fcoop'})
    df = df.merge(foo, on='id', how='left')

    # Create valid round measure
    df['foo'] = df['round'] * df['validround'].fillna(0)
    df['max_round'] = df.groupby(['finite', 'session', 'supergame', 'id'])['foo'].transform('max')
    df['length'] = df['max_round']

    # Keep only round 1
    df_r1 = df[df['round'] == 1].copy()

    # Create lagged variables for round 1 analysis
    df_r1 = df_r1.sort_values(['id', 'supergame'])
    df_r1['focoop_m1'] = df_r1.groupby('id')['o_coop'].shift(1)
    df_r1['length_m1'] = df_r1.groupby('id')['length'].shift(1)

    # Risk measure
    df_r1['risk'] = df_r1['bomb_choice']

    return df_r1

# Prepare data for main analysis
print("Preparing data...")
df_r1 = prepare_data(df)
df_r1 = df_r1.dropna(subset=['coop'])

# Also prepare full data for round-level analysis
df_full = df.copy()
df_full['risk'] = df_full['bomb_choice']

# Results storage
results = []

def add_result(spec_id, spec_tree_path, outcome_var, treatment_var, model_result,
               sample_desc, model_type, cluster_var, controls_desc, fixed_effects,
               data_df, formula_used):
    """Add a result to the results list"""

    # Get treatment coefficient and stats
    try:
        if treatment_var in model_result.params.index:
            coef = model_result.params[treatment_var]
            se = model_result.bse[treatment_var]
            t_stat = model_result.tvalues[treatment_var]
            p_val = model_result.pvalues[treatment_var]
        else:
            # For models without explicit treatment var
            coef = np.nan
            se = np.nan
            t_stat = np.nan
            p_val = np.nan
    except:
        coef = np.nan
        se = np.nan
        t_stat = np.nan
        p_val = np.nan

    # Compute CI
    ci_lower = coef - 1.96 * se if not np.isnan(se) else np.nan
    ci_upper = coef + 1.96 * se if not np.isnan(se) else np.nan

    # Get R-squared (pseudo for logit/probit)
    try:
        r2 = model_result.prsquared if hasattr(model_result, 'prsquared') else model_result.rsquared
    except:
        r2 = np.nan

    # Build coefficient vector
    coef_vector = {"treatment": {}, "controls": [], "fixed_effects": [], "diagnostics": {}}
    try:
        for var in model_result.params.index:
            if var == treatment_var:
                coef_vector["treatment"] = {
                    "var": var,
                    "coef": float(model_result.params[var]),
                    "se": float(model_result.bse[var]),
                    "pval": float(model_result.pvalues[var])
                }
            elif var != 'Intercept' and var != 'const':
                coef_vector["controls"].append({
                    "var": var,
                    "coef": float(model_result.params[var]),
                    "se": float(model_result.bse[var]),
                    "pval": float(model_result.pvalues[var])
                })
    except:
        pass

    coef_vector["fixed_effects"] = fixed_effects.split('+') if fixed_effects else []
    try:
        coef_vector["diagnostics"] = {
            "pseudo_r2": float(r2) if not np.isnan(r2) else None,
            "ll": float(model_result.llf) if hasattr(model_result, 'llf') else None,
            "aic": float(model_result.aic) if hasattr(model_result, 'aic') else None
        }
    except:
        pass

    result = {
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
        'p_value': p_val,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_obs': int(model_result.nobs),
        'r_squared': r2,
        'coefficient_vector_json': json.dumps(coef_vector),
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    }

    results.append(result)
    print(f"  {spec_id}: coef={coef:.4f}, se={se:.4f}, p={p_val:.4f}, n={int(model_result.nobs)}")
    return result


def run_probit_clustered(formula, data, cluster_var='session'):
    """Run probit with clustered standard errors"""
    try:
        model = smf.probit(formula, data=data)
        result = model.fit(cov_type='cluster', cov_kwds={'groups': data[cluster_var]}, disp=0)
        return result
    except Exception as e:
        print(f"Error in probit: {e}")
        return None


def run_logit_clustered(formula, data, cluster_var='session'):
    """Run logit with clustered standard errors"""
    try:
        model = smf.logit(formula, data=data)
        result = model.fit(cov_type='cluster', cov_kwds={'groups': data[cluster_var]}, disp=0)
        return result
    except Exception as e:
        print(f"Error in logit: {e}")
        return None


def run_ols_clustered(formula, data, cluster_var='session'):
    """Run OLS (LPM) with clustered standard errors"""
    try:
        model = smf.ols(formula, data=data)
        result = model.fit(cov_type='cluster', cov_kwds={'groups': data[cluster_var]}, disp=0)
        return result
    except Exception as e:
        print(f"Error in OLS: {e}")
        return None


# ========================================
# SECTION 1: BASELINE SPECIFICATIONS
# ========================================
print("\n" + "="*60)
print("SECTION 1: BASELINE SPECIFICATIONS")
print("="*60)

# Baseline 1: Finite game - cooperation determinants (Table 6/7 replication)
print("\n1.1 Baseline: Finite game determinants of round-1 cooperation")
df_finite = df_r1[df_r1['finite'] == 1].dropna(subset=['coop', 'beliefon', 'supergame',
                                                        'focoop_m1', 'fcoop', 'risk'])

if len(df_finite) > 50:
    formula = 'coop ~ beliefon + supergame + focoop_m1 + fcoop + risk'
    result = run_probit_clustered(formula, df_finite)
    if result:
        add_result('baseline', 'methods/discrete_choice.md#baseline',
                  'coop', 'beliefon', result, 'Finite game, round 1',
                  'probit', 'session', 'supergame+focoop_m1+fcoop+risk', 'none',
                  df_finite, formula)

# Baseline 2: Indefinite game
print("\n1.2 Baseline: Indefinite game determinants of round-1 cooperation")
df_infinite = df_r1[df_r1['finite'] == 0].dropna(subset=['coop', 'beliefon', 'supergame',
                                                          'focoop_m1', 'fcoop', 'risk', 'length_m1'])

if len(df_infinite) > 50:
    formula = 'coop ~ beliefon + supergame + focoop_m1 + fcoop + risk + length_m1'
    result = run_probit_clustered(formula, df_infinite)
    if result:
        add_result('baseline_indefinite', 'methods/discrete_choice.md#baseline',
                  'coop', 'beliefon', result, 'Indefinite game, round 1',
                  'probit', 'session', 'supergame+focoop_m1+fcoop+risk+length_m1', 'none',
                  df_infinite, formula)


# ========================================
# SECTION 2: MODEL TYPE VARIATIONS
# ========================================
print("\n" + "="*60)
print("SECTION 2: MODEL TYPE VARIATIONS")
print("="*60)

# Logit for finite
print("\n2.1 Logit: Finite game")
if len(df_finite) > 50:
    formula = 'coop ~ beliefon + supergame + focoop_m1 + fcoop + risk'
    result = run_logit_clustered(formula, df_finite)
    if result:
        add_result('discrete/binary/logit_finite', 'methods/discrete_choice.md#model-type-binary-outcome',
                  'coop', 'beliefon', result, 'Finite game, round 1',
                  'logit', 'session', 'supergame+focoop_m1+fcoop+risk', 'none',
                  df_finite, formula)

# Logit for indefinite
print("\n2.2 Logit: Indefinite game")
if len(df_infinite) > 50:
    formula = 'coop ~ beliefon + supergame + focoop_m1 + fcoop + risk + length_m1'
    result = run_logit_clustered(formula, df_infinite)
    if result:
        add_result('discrete/binary/logit_indefinite', 'methods/discrete_choice.md#model-type-binary-outcome',
                  'coop', 'beliefon', result, 'Indefinite game, round 1',
                  'logit', 'session', 'supergame+focoop_m1+fcoop+risk+length_m1', 'none',
                  df_infinite, formula)

# LPM for finite
print("\n2.3 LPM: Finite game")
if len(df_finite) > 50:
    formula = 'coop ~ beliefon + supergame + focoop_m1 + fcoop + risk'
    result = run_ols_clustered(formula, df_finite)
    if result:
        add_result('discrete/binary/lpm_finite', 'methods/discrete_choice.md#model-type-binary-outcome',
                  'coop', 'beliefon', result, 'Finite game, round 1',
                  'OLS-LPM', 'session', 'supergame+focoop_m1+fcoop+risk', 'none',
                  df_finite, formula)

# LPM for indefinite
print("\n2.4 LPM: Indefinite game")
if len(df_infinite) > 50:
    formula = 'coop ~ beliefon + supergame + focoop_m1 + fcoop + risk + length_m1'
    result = run_ols_clustered(formula, df_infinite)
    if result:
        add_result('discrete/binary/lpm_indefinite', 'methods/discrete_choice.md#model-type-binary-outcome',
                  'coop', 'beliefon', result, 'Indefinite game, round 1',
                  'OLS-LPM', 'session', 'supergame+focoop_m1+fcoop+risk+length_m1', 'none',
                  df_infinite, formula)


# ========================================
# SECTION 3: CONTROL VARIATIONS
# ========================================
print("\n" + "="*60)
print("SECTION 3: CONTROL VARIATIONS")
print("="*60)

# No controls - finite
print("\n3.1 No controls: Finite")
if len(df_finite) > 50:
    formula = 'coop ~ beliefon'
    result = run_probit_clustered(formula, df_finite)
    if result:
        add_result('discrete/controls/none_finite', 'methods/discrete_choice.md#control-sets',
                  'coop', 'beliefon', result, 'Finite game, round 1, no controls',
                  'probit', 'session', 'none', 'none', df_finite, formula)

# Minimal controls - finite
print("\n3.2 Minimal controls (supergame only): Finite")
if len(df_finite) > 50:
    formula = 'coop ~ beliefon + supergame'
    result = run_probit_clustered(formula, df_finite)
    if result:
        add_result('discrete/controls/minimal_finite', 'methods/discrete_choice.md#control-sets',
                  'coop', 'beliefon', result, 'Finite game, round 1, minimal controls',
                  'probit', 'session', 'supergame', 'none', df_finite, formula)

# No controls - indefinite
print("\n3.3 No controls: Indefinite")
if len(df_infinite) > 50:
    formula = 'coop ~ beliefon'
    result = run_probit_clustered(formula, df_infinite)
    if result:
        add_result('discrete/controls/none_indefinite', 'methods/discrete_choice.md#control-sets',
                  'coop', 'beliefon', result, 'Indefinite game, round 1, no controls',
                  'probit', 'session', 'none', 'none', df_infinite, formula)

# Minimal controls - indefinite
print("\n3.4 Minimal controls (supergame only): Indefinite")
if len(df_infinite) > 50:
    formula = 'coop ~ beliefon + supergame'
    result = run_probit_clustered(formula, df_infinite)
    if result:
        add_result('discrete/controls/minimal_indefinite', 'methods/discrete_choice.md#control-sets',
                  'coop', 'beliefon', result, 'Indefinite game, round 1, minimal controls',
                  'probit', 'session', 'supergame', 'none', df_infinite, formula)


# ========================================
# SECTION 4: LEAVE-ONE-OUT CONTROL VARIATIONS
# ========================================
print("\n" + "="*60)
print("SECTION 4: LEAVE-ONE-OUT CONTROL VARIATIONS")
print("="*60)

# Finite game controls
finite_controls = ['supergame', 'focoop_m1', 'fcoop', 'risk']
for ctrl in finite_controls:
    print(f"\n4.{finite_controls.index(ctrl)+1} Drop {ctrl}: Finite")
    remaining = [c for c in finite_controls if c != ctrl]
    formula = f'coop ~ beliefon + {" + ".join(remaining)}'
    df_temp = df_finite.dropna(subset=remaining + ['coop', 'beliefon'])
    if len(df_temp) > 50:
        result = run_probit_clustered(formula, df_temp)
        if result:
            add_result(f'robust/control/drop_{ctrl}_finite', 'robustness/leave_one_out.md',
                      'coop', 'beliefon', result, f'Finite game, drop {ctrl}',
                      'probit', 'session', '+'.join(remaining), 'none', df_temp, formula)

# Indefinite game controls
indefinite_controls = ['supergame', 'focoop_m1', 'fcoop', 'risk', 'length_m1']
for ctrl in indefinite_controls:
    print(f"\n4.{len(finite_controls)+indefinite_controls.index(ctrl)+1} Drop {ctrl}: Indefinite")
    remaining = [c for c in indefinite_controls if c != ctrl]
    formula = f'coop ~ beliefon + {" + ".join(remaining)}'
    df_temp = df_infinite.dropna(subset=remaining + ['coop', 'beliefon'])
    if len(df_temp) > 50:
        result = run_probit_clustered(formula, df_temp)
        if result:
            add_result(f'robust/control/drop_{ctrl}_indefinite', 'robustness/leave_one_out.md',
                      'coop', 'beliefon', result, f'Indefinite game, drop {ctrl}',
                      'probit', 'session', '+'.join(remaining), 'none', df_temp, formula)


# ========================================
# SECTION 5: SAMPLE RESTRICTIONS
# ========================================
print("\n" + "="*60)
print("SECTION 5: SAMPLE RESTRICTIONS")
print("="*60)

# Early vs Late supergames
print("\n5.1 Early supergames only: Finite")
df_early_fin = df_finite[df_finite['early'] == 1]
if len(df_early_fin) > 50:
    formula = 'coop ~ beliefon + supergame + focoop_m1 + fcoop + risk'
    df_temp = df_early_fin.dropna(subset=['coop', 'beliefon', 'supergame', 'focoop_m1', 'fcoop', 'risk'])
    if len(df_temp) > 30:
        result = run_probit_clustered(formula, df_temp)
        if result:
            add_result('robust/sample/early_supergames_finite', 'robustness/sample_restrictions.md#time-based-restrictions',
                      'coop', 'beliefon', result, 'Finite game, early supergames',
                      'probit', 'session', 'supergame+focoop_m1+fcoop+risk', 'none', df_temp, formula)

print("\n5.2 Late supergames only: Finite")
df_late_fin = df_finite[df_finite['late'] == 1]
if len(df_late_fin) > 50:
    formula = 'coop ~ beliefon + supergame + focoop_m1 + fcoop + risk'
    df_temp = df_late_fin.dropna(subset=['coop', 'beliefon', 'supergame', 'focoop_m1', 'fcoop', 'risk'])
    if len(df_temp) > 30:
        result = run_probit_clustered(formula, df_temp)
        if result:
            add_result('robust/sample/late_supergames_finite', 'robustness/sample_restrictions.md#time-based-restrictions',
                      'coop', 'beliefon', result, 'Finite game, late supergames',
                      'probit', 'session', 'supergame+focoop_m1+fcoop+risk', 'none', df_temp, formula)

print("\n5.3 Early supergames only: Indefinite")
df_early_inf = df_infinite[df_infinite['early'] == 1]
if len(df_early_inf) > 50:
    formula = 'coop ~ beliefon + supergame + focoop_m1 + fcoop + risk + length_m1'
    df_temp = df_early_inf.dropna(subset=['coop', 'beliefon', 'supergame', 'focoop_m1', 'fcoop', 'risk', 'length_m1'])
    if len(df_temp) > 30:
        result = run_probit_clustered(formula, df_temp)
        if result:
            add_result('robust/sample/early_supergames_indefinite', 'robustness/sample_restrictions.md#time-based-restrictions',
                      'coop', 'beliefon', result, 'Indefinite game, early supergames',
                      'probit', 'session', 'supergame+focoop_m1+fcoop+risk+length_m1', 'none', df_temp, formula)

print("\n5.4 Late supergames only: Indefinite")
df_late_inf = df_infinite[df_infinite['late'] == 1]
if len(df_late_inf) > 50:
    formula = 'coop ~ beliefon + supergame + focoop_m1 + fcoop + risk + length_m1'
    df_temp = df_late_inf.dropna(subset=['coop', 'beliefon', 'supergame', 'focoop_m1', 'fcoop', 'risk', 'length_m1'])
    if len(df_temp) > 30:
        result = run_probit_clustered(formula, df_temp)
        if result:
            add_result('robust/sample/late_supergames_indefinite', 'robustness/sample_restrictions.md#time-based-restrictions',
                      'coop', 'beliefon', result, 'Indefinite game, late supergames',
                      'probit', 'session', 'supergame+focoop_m1+fcoop+risk+length_m1', 'none', df_temp, formula)

# Drop first supergame
print("\n5.5 Drop first supergame: Finite")
df_drop_first_fin = df_finite[df_finite['supergame'] > 1]
if len(df_drop_first_fin) > 50:
    formula = 'coop ~ beliefon + supergame + focoop_m1 + fcoop + risk'
    df_temp = df_drop_first_fin.dropna(subset=['coop', 'beliefon', 'supergame', 'focoop_m1', 'fcoop', 'risk'])
    if len(df_temp) > 30:
        result = run_probit_clustered(formula, df_temp)
        if result:
            add_result('robust/sample/drop_first_supergame_finite', 'robustness/sample_restrictions.md#time-based-restrictions',
                      'coop', 'beliefon', result, 'Finite game, drop supergame 1',
                      'probit', 'session', 'supergame+focoop_m1+fcoop+risk', 'none', df_temp, formula)

print("\n5.6 Drop first supergame: Indefinite")
df_drop_first_inf = df_infinite[df_infinite['supergame'] > 1]
if len(df_drop_first_inf) > 50:
    formula = 'coop ~ beliefon + supergame + focoop_m1 + fcoop + risk + length_m1'
    df_temp = df_drop_first_inf.dropna(subset=['coop', 'beliefon', 'supergame', 'focoop_m1', 'fcoop', 'risk', 'length_m1'])
    if len(df_temp) > 30:
        result = run_probit_clustered(formula, df_temp)
        if result:
            add_result('robust/sample/drop_first_supergame_indefinite', 'robustness/sample_restrictions.md#time-based-restrictions',
                      'coop', 'beliefon', result, 'Indefinite game, drop supergame 1',
                      'probit', 'session', 'supergame+focoop_m1+fcoop+risk+length_m1', 'none', df_temp, formula)

# Drop late supergames
print("\n5.7 Drop late supergames: Finite")
df_no_late_fin = df_finite[df_finite['late'] == 0]
if len(df_no_late_fin) > 50:
    formula = 'coop ~ beliefon + supergame + focoop_m1 + fcoop + risk'
    df_temp = df_no_late_fin.dropna(subset=['coop', 'beliefon', 'supergame', 'focoop_m1', 'fcoop', 'risk'])
    if len(df_temp) > 30:
        result = run_probit_clustered(formula, df_temp)
        if result:
            add_result('robust/sample/drop_late_supergames_finite', 'robustness/sample_restrictions.md#time-based-restrictions',
                      'coop', 'beliefon', result, 'Finite game, excluding late supergames',
                      'probit', 'session', 'supergame+focoop_m1+fcoop+risk', 'none', df_temp, formula)


# ========================================
# SECTION 6: CLUSTERING VARIATIONS
# ========================================
print("\n" + "="*60)
print("SECTION 6: CLUSTERING VARIATIONS")
print("="*60)

# Robust (no clustering) - finite
print("\n6.1 Robust SE (no clustering): Finite")
if len(df_finite) > 50:
    formula = 'coop ~ beliefon + supergame + focoop_m1 + fcoop + risk'
    model = smf.probit(formula, data=df_finite)
    result = model.fit(cov_type='HC1', disp=0)
    if result:
        add_result('robust/cluster/none_finite', 'robustness/clustering_variations.md#single-level-clustering',
                  'coop', 'beliefon', result, 'Finite game, robust SE',
                  'probit', 'none', 'supergame+focoop_m1+fcoop+risk', 'none', df_finite, formula)

# Robust (no clustering) - indefinite
print("\n6.2 Robust SE (no clustering): Indefinite")
if len(df_infinite) > 50:
    formula = 'coop ~ beliefon + supergame + focoop_m1 + fcoop + risk + length_m1'
    model = smf.probit(formula, data=df_infinite)
    result = model.fit(cov_type='HC1', disp=0)
    if result:
        add_result('robust/cluster/none_indefinite', 'robustness/clustering_variations.md#single-level-clustering',
                  'coop', 'beliefon', result, 'Indefinite game, robust SE',
                  'probit', 'none', 'supergame+focoop_m1+fcoop+risk+length_m1', 'none', df_infinite, formula)

# Cluster by id (subject) - finite
print("\n6.3 Cluster by subject: Finite")
if len(df_finite) > 50:
    formula = 'coop ~ beliefon + supergame + focoop_m1 + fcoop + risk'
    result = run_probit_clustered(formula, df_finite, cluster_var='id')
    if result:
        add_result('robust/cluster/unit_finite', 'robustness/clustering_variations.md#single-level-clustering',
                  'coop', 'beliefon', result, 'Finite game, cluster by subject',
                  'probit', 'id', 'supergame+focoop_m1+fcoop+risk', 'none', df_finite, formula)

# Cluster by id (subject) - indefinite
print("\n6.4 Cluster by subject: Indefinite")
if len(df_infinite) > 50:
    formula = 'coop ~ beliefon + supergame + focoop_m1 + fcoop + risk + length_m1'
    result = run_probit_clustered(formula, df_infinite, cluster_var='id')
    if result:
        add_result('robust/cluster/unit_indefinite', 'robustness/clustering_variations.md#single-level-clustering',
                  'coop', 'beliefon', result, 'Indefinite game, cluster by subject',
                  'probit', 'id', 'supergame+focoop_m1+fcoop+risk+length_m1', 'none', df_infinite, formula)


# ========================================
# SECTION 7: HETEROGENEITY ANALYSIS
# ========================================
print("\n" + "="*60)
print("SECTION 7: HETEROGENEITY ANALYSIS")
print("="*60)

# By risk level (median split)
print("\n7.1 Heterogeneity by risk: Low risk, Finite")
risk_median = df_finite['risk'].median()
df_low_risk_fin = df_finite[df_finite['risk'] <= risk_median]
if len(df_low_risk_fin) > 50:
    formula = 'coop ~ beliefon + supergame + focoop_m1 + fcoop'
    df_temp = df_low_risk_fin.dropna(subset=['coop', 'beliefon', 'supergame', 'focoop_m1', 'fcoop'])
    if len(df_temp) > 30:
        result = run_probit_clustered(formula, df_temp)
        if result:
            add_result('robust/het/by_risk_low_finite', 'robustness/heterogeneity.md#socioeconomic-subgroups',
                      'coop', 'beliefon', result, 'Finite game, low risk subjects',
                      'probit', 'session', 'supergame+focoop_m1+fcoop', 'none', df_temp, formula)

print("\n7.2 Heterogeneity by risk: High risk, Finite")
df_high_risk_fin = df_finite[df_finite['risk'] > risk_median]
if len(df_high_risk_fin) > 50:
    formula = 'coop ~ beliefon + supergame + focoop_m1 + fcoop'
    df_temp = df_high_risk_fin.dropna(subset=['coop', 'beliefon', 'supergame', 'focoop_m1', 'fcoop'])
    if len(df_temp) > 30:
        result = run_probit_clustered(formula, df_temp)
        if result:
            add_result('robust/het/by_risk_high_finite', 'robustness/heterogeneity.md#socioeconomic-subgroups',
                      'coop', 'beliefon', result, 'Finite game, high risk subjects',
                      'probit', 'session', 'supergame+focoop_m1+fcoop', 'none', df_temp, formula)

# By first supergame cooperation
print("\n7.3 Heterogeneity by first-supergame cooperation: Cooperators, Finite")
df_first_coop_fin = df_finite[df_finite['fcoop'] == 1]
if len(df_first_coop_fin) > 50:
    formula = 'coop ~ beliefon + supergame + focoop_m1 + risk'
    df_temp = df_first_coop_fin.dropna(subset=['coop', 'beliefon', 'supergame', 'focoop_m1', 'risk'])
    if len(df_temp) > 30:
        result = run_probit_clustered(formula, df_temp)
        if result:
            add_result('robust/het/by_fcoop_cooperators_finite', 'robustness/heterogeneity.md#baseline-characteristics',
                      'coop', 'beliefon', result, 'Finite game, first-supergame cooperators',
                      'probit', 'session', 'supergame+focoop_m1+risk', 'none', df_temp, formula)

print("\n7.4 Heterogeneity by first-supergame cooperation: Defectors, Finite")
df_first_defect_fin = df_finite[df_finite['fcoop'] == 0]
if len(df_first_defect_fin) > 50:
    formula = 'coop ~ beliefon + supergame + focoop_m1 + risk'
    df_temp = df_first_defect_fin.dropna(subset=['coop', 'beliefon', 'supergame', 'focoop_m1', 'risk'])
    if len(df_temp) > 30:
        result = run_probit_clustered(formula, df_temp)
        if result:
            add_result('robust/het/by_fcoop_defectors_finite', 'robustness/heterogeneity.md#baseline-characteristics',
                      'coop', 'beliefon', result, 'Finite game, first-supergame defectors',
                      'probit', 'session', 'supergame+focoop_m1+risk', 'none', df_temp, formula)

# Interaction specifications
print("\n7.5 Interaction: beliefon x supergame, Finite")
if len(df_finite) > 50:
    formula = 'coop ~ beliefon * supergame + focoop_m1 + fcoop + risk'
    df_temp = df_finite.dropna(subset=['coop', 'beliefon', 'supergame', 'focoop_m1', 'fcoop', 'risk'])
    result = run_probit_clustered(formula, df_temp)
    if result:
        add_result('robust/het/interaction_supergame_finite', 'robustness/heterogeneity.md#interaction-specifications',
                  'coop', 'beliefon', result, 'Finite game, beliefon x supergame interaction',
                  'probit', 'session', 'supergame+focoop_m1+fcoop+risk+beliefon:supergame', 'none', df_temp, formula)

print("\n7.6 Interaction: beliefon x fcoop, Finite")
if len(df_finite) > 50:
    formula = 'coop ~ beliefon * fcoop + supergame + focoop_m1 + risk'
    df_temp = df_finite.dropna(subset=['coop', 'beliefon', 'supergame', 'focoop_m1', 'fcoop', 'risk'])
    result = run_probit_clustered(formula, df_temp)
    if result:
        add_result('robust/het/interaction_fcoop_finite', 'robustness/heterogeneity.md#interaction-specifications',
                  'coop', 'beliefon', result, 'Finite game, beliefon x fcoop interaction',
                  'probit', 'session', 'supergame+focoop_m1+risk+beliefon:fcoop', 'none', df_temp, formula)

# Risk heterogeneity for indefinite
print("\n7.7 Heterogeneity by risk: Low risk, Indefinite")
risk_median_inf = df_infinite['risk'].median()
df_low_risk_inf = df_infinite[df_infinite['risk'] <= risk_median_inf]
if len(df_low_risk_inf) > 50:
    formula = 'coop ~ beliefon + supergame + focoop_m1 + fcoop + length_m1'
    df_temp = df_low_risk_inf.dropna(subset=['coop', 'beliefon', 'supergame', 'focoop_m1', 'fcoop', 'length_m1'])
    if len(df_temp) > 30:
        result = run_probit_clustered(formula, df_temp)
        if result:
            add_result('robust/het/by_risk_low_indefinite', 'robustness/heterogeneity.md#socioeconomic-subgroups',
                      'coop', 'beliefon', result, 'Indefinite game, low risk subjects',
                      'probit', 'session', 'supergame+focoop_m1+fcoop+length_m1', 'none', df_temp, formula)

print("\n7.8 Heterogeneity by risk: High risk, Indefinite")
df_high_risk_inf = df_infinite[df_infinite['risk'] > risk_median_inf]
if len(df_high_risk_inf) > 50:
    formula = 'coop ~ beliefon + supergame + focoop_m1 + fcoop + length_m1'
    df_temp = df_high_risk_inf.dropna(subset=['coop', 'beliefon', 'supergame', 'focoop_m1', 'fcoop', 'length_m1'])
    if len(df_temp) > 30:
        result = run_probit_clustered(formula, df_temp)
        if result:
            add_result('robust/het/by_risk_high_indefinite', 'robustness/heterogeneity.md#socioeconomic-subgroups',
                      'coop', 'beliefon', result, 'Indefinite game, high risk subjects',
                      'probit', 'session', 'supergame+focoop_m1+fcoop+length_m1', 'none', df_temp, formula)


# ========================================
# SECTION 8: ADDITIONAL TREATMENT COMPARISONS
# ========================================
print("\n" + "="*60)
print("SECTION 8: TREATMENT COMPARISONS (COMBINED DATA)")
print("="*60)

# Prepare combined data for analysis
df_comb = df_combined.copy()
df_comb['risk'] = df_comb['bomb_choice']

# Create first-supergame cooperation
foo = df_comb.loc[(df_comb['round'] == 1) & (df_comb['supergame'] == 1), ['id', 'session', 'coop']].rename(columns={'coop': 'fcoop'})
df_comb = df_comb.merge(foo, on=['id', 'session'], how='left', suffixes=('', '_first'))

# Filter to round 1
df_comb_r1 = df_comb[df_comb['round'] == 1].copy()
df_comb_r1 = df_comb_r1.sort_values(['id', 'session', 'supergame'])
df_comb_r1['focoop_m1'] = df_comb_r1.groupby(['id', 'session'])['o_coop'].shift(1)

# Effect of finite vs indefinite (pooled)
print("\n8.1 Finite vs Indefinite comparison (pooled)")
df_pooled = df_comb_r1[df_comb_r1['late'] == 1].copy()
df_pooled = df_pooled[(df_pooled['lowr'] == 0) & (df_pooled['hight'] == 0)]
if len(df_pooled) > 50:
    formula = 'coop ~ finite + supergame + focoop_m1 + risk'
    df_temp = df_pooled.dropna(subset=['coop', 'finite', 'supergame', 'focoop_m1', 'risk'])
    if len(df_temp) > 30:
        result = run_probit_clustered(formula, df_temp)
        if result:
            add_result('custom/treatment_finite_pooled', 'custom',
                      'coop', 'finite', result, 'Pooled data, late supergames, finite effect',
                      'probit', 'session', 'supergame+focoop_m1+risk', 'none', df_temp, formula)

# Low R treatment effect
print("\n8.2 Low R treatment effect (indefinite games)")
df_inf_comb = df_comb_r1[(df_comb_r1['finite'] == 0) & (df_comb_r1['late'] == 1) & (df_comb_r1['hight'] == 0)]
if len(df_inf_comb) > 50:
    formula = 'coop ~ lowr + supergame + focoop_m1 + risk'
    df_temp = df_inf_comb.dropna(subset=['coop', 'lowr', 'supergame', 'focoop_m1', 'risk'])
    if len(df_temp) > 30:
        result = run_probit_clustered(formula, df_temp)
        if result:
            add_result('custom/treatment_lowr', 'custom',
                      'coop', 'lowr', result, 'Indefinite game, Low R treatment effect',
                      'probit', 'session', 'supergame+focoop_m1+risk', 'none', df_temp, formula)

# High T treatment effect
print("\n8.3 High T treatment effect (indefinite games)")
df_inf_comb2 = df_comb_r1[(df_comb_r1['finite'] == 0) & (df_comb_r1['late'] == 1) & (df_comb_r1['lowr'] == 0)]
if len(df_inf_comb2) > 50:
    formula = 'coop ~ hight + supergame + focoop_m1 + risk'
    df_temp = df_inf_comb2.dropna(subset=['coop', 'hight', 'supergame', 'focoop_m1', 'risk'])
    if len(df_temp) > 30:
        result = run_probit_clustered(formula, df_temp)
        if result:
            add_result('custom/treatment_hight', 'custom',
                      'coop', 'hight', result, 'Indefinite game, High T treatment effect',
                      'probit', 'session', 'supergame+focoop_m1+risk', 'none', df_temp, formula)


# ========================================
# SECTION 9: ALTERNATIVE OUTCOMES
# ========================================
print("\n" + "="*60)
print("SECTION 9: ALTERNATIVE OUTCOMES")
print("="*60)

# Use belief as outcome (conditional on round and coop)
print("\n9.1 Belief as outcome: Finite, round 1 cooperators")
df_fin_coop = df_full[(df_full['finite'] == 1) & (df_full['round'] == 1) & (df_full['coop'] == 1) & (df_full['late'] == 1)]
if len(df_fin_coop) > 50:
    formula = 'belief ~ supergame + o_coop_m1 + risk'
    df_temp = df_fin_coop.dropna(subset=['belief', 'supergame', 'o_coop_m1', 'risk'])
    if len(df_temp) > 30:
        result = run_ols_clustered(formula, df_temp)
        if result:
            add_result('robust/outcome/belief_cooperators_finite', 'robustness/measurement.md',
                      'belief', 'supergame', result, 'Finite game, beliefs of cooperators',
                      'OLS', 'session', 'o_coop_m1+risk', 'none', df_temp, formula)

print("\n9.2 Belief as outcome: Finite, round 1 defectors")
df_fin_defect = df_full[(df_full['finite'] == 1) & (df_full['round'] == 1) & (df_full['coop'] == 0) & (df_full['late'] == 1)]
if len(df_fin_defect) > 50:
    formula = 'belief ~ supergame + o_coop_m1 + risk'
    df_temp = df_fin_defect.dropna(subset=['belief', 'supergame', 'o_coop_m1', 'risk'])
    if len(df_temp) > 30:
        result = run_ols_clustered(formula, df_temp)
        if result:
            add_result('robust/outcome/belief_defectors_finite', 'robustness/measurement.md',
                      'belief', 'supergame', result, 'Finite game, beliefs of defectors',
                      'OLS', 'session', 'o_coop_m1+risk', 'none', df_temp, formula)


# ========================================
# SECTION 10: ROUND-LEVEL ANALYSIS
# ========================================
print("\n" + "="*60)
print("SECTION 10: ROUND-LEVEL ANALYSIS")
print("="*60)

# Cooperation over all rounds (not just round 1)
print("\n10.1 All rounds: Finite, late supergames")
df_all_rounds_fin = df_full[(df_full['finite'] == 1) & (df_full['late'] == 1) & (df_full['round'] <= 8)]
if len(df_all_rounds_fin) > 100:
    formula = 'coop ~ belief + round + o_coop_m1'
    df_temp = df_all_rounds_fin.dropna(subset=['coop', 'belief', 'round', 'o_coop_m1'])
    if len(df_temp) > 50:
        result = run_probit_clustered(formula, df_temp)
        if result:
            add_result('custom/all_rounds_finite', 'custom',
                      'coop', 'belief', result, 'Finite game, all rounds, belief effect',
                      'probit', 'session', 'round+o_coop_m1', 'none', df_temp, formula)

print("\n10.2 All rounds: Indefinite, late supergames")
df_all_rounds_inf = df_full[(df_full['finite'] == 0) & (df_full['late'] == 1) & (df_full['round'] <= 8)]
if len(df_all_rounds_inf) > 100:
    formula = 'coop ~ belief + round + o_coop_m1'
    df_temp = df_all_rounds_inf.dropna(subset=['coop', 'belief', 'round', 'o_coop_m1'])
    if len(df_temp) > 50:
        result = run_probit_clustered(formula, df_temp)
        if result:
            add_result('custom/all_rounds_indefinite', 'custom',
                      'coop', 'belief', result, 'Indefinite game, all rounds, belief effect',
                      'probit', 'session', 'round+o_coop_m1', 'none', df_temp, formula)


# ========================================
# SECTION 11: ADDITIONAL CONTROL PROGRESSIONS
# ========================================
print("\n" + "="*60)
print("SECTION 11: CONTROL PROGRESSIONS")
print("="*60)

# Build-up controls for finite
print("\n11.1 Add supergame only: Finite")
if len(df_finite) > 50:
    formula = 'coop ~ beliefon + supergame'
    result = run_probit_clustered(formula, df_finite)
    if result:
        add_result('robust/control/add_supergame_finite', 'robustness/control_progression.md',
                  'coop', 'beliefon', result, 'Finite, add supergame',
                  'probit', 'session', 'supergame', 'none', df_finite, formula)

print("\n11.2 Add supergame + focoop_m1: Finite")
if len(df_finite) > 50:
    formula = 'coop ~ beliefon + supergame + focoop_m1'
    df_temp = df_finite.dropna(subset=['coop', 'beliefon', 'supergame', 'focoop_m1'])
    if len(df_temp) > 30:
        result = run_probit_clustered(formula, df_temp)
        if result:
            add_result('robust/control/add_focoop_m1_finite', 'robustness/control_progression.md',
                      'coop', 'beliefon', result, 'Finite, add supergame+focoop_m1',
                      'probit', 'session', 'supergame+focoop_m1', 'none', df_temp, formula)

print("\n11.3 Add supergame + focoop_m1 + fcoop: Finite")
if len(df_finite) > 50:
    formula = 'coop ~ beliefon + supergame + focoop_m1 + fcoop'
    df_temp = df_finite.dropna(subset=['coop', 'beliefon', 'supergame', 'focoop_m1', 'fcoop'])
    if len(df_temp) > 30:
        result = run_probit_clustered(formula, df_temp)
        if result:
            add_result('robust/control/add_fcoop_finite', 'robustness/control_progression.md',
                      'coop', 'beliefon', result, 'Finite, add supergame+focoop_m1+fcoop',
                      'probit', 'session', 'supergame+focoop_m1+fcoop', 'none', df_temp, formula)


# ========================================
# SECTION 12: SESSION-LEVEL DROP ANALYSIS
# ========================================
print("\n" + "="*60)
print("SECTION 12: DROP EACH SESSION")
print("="*60)

sessions = df_finite['session'].unique()
for i, sess in enumerate(sessions[:5]):  # Limit to first 5 sessions
    print(f"\n12.{i+1} Drop session {int(sess)}: Finite")
    df_drop_sess = df_finite[df_finite['session'] != sess]
    if len(df_drop_sess) > 50:
        formula = 'coop ~ beliefon + supergame + focoop_m1 + fcoop + risk'
        df_temp = df_drop_sess.dropna(subset=['coop', 'beliefon', 'supergame', 'focoop_m1', 'fcoop', 'risk'])
        if len(df_temp) > 30:
            result = run_probit_clustered(formula, df_temp)
            if result:
                add_result(f'robust/sample/drop_session_{int(sess)}_finite',
                          'robustness/sample_restrictions.md#influential-observations',
                          'coop', 'beliefon', result, f'Finite, drop session {int(sess)}',
                          'probit', 'session', 'supergame+focoop_m1+fcoop+risk', 'none', df_temp, formula)


# ========================================
# SECTION 13: SUPERGAME-BY-SUPERGAME
# ========================================
print("\n" + "="*60)
print("SECTION 13: BY SUPERGAME")
print("="*60)

for sg in range(1, 5):
    print(f"\n13.{sg} Supergame {sg} only: Finite")
    df_sg = df_finite[df_finite['supergame'] == sg]
    if len(df_sg) > 30:
        formula = 'coop ~ beliefon + focoop_m1 + fcoop + risk'
        df_temp = df_sg.dropna(subset=['coop', 'beliefon', 'focoop_m1', 'fcoop', 'risk'])
        if len(df_temp) > 20:
            result = run_probit_clustered(formula, df_temp)
            if result:
                add_result(f'robust/sample/supergame_{sg}_finite',
                          'robustness/sample_restrictions.md#time-based-restrictions',
                          'coop', 'beliefon', result, f'Finite, supergame {sg} only',
                          'probit', 'session', 'focoop_m1+fcoop+risk', 'none', df_temp, formula)


# ========================================
# SECTION 14: BELIEF ERROR ANALYSIS
# ========================================
print("\n" + "="*60)
print("SECTION 14: BELIEF ERROR ANALYSIS")
print("="*60)

# Create belief error variable
df_full['belief_error'] = df_full['belief'] - df_full['o_coop']

print("\n14.1 Belief error determinants: Finite, late supergames")
df_be_fin = df_full[(df_full['finite'] == 1) & (df_full['late'] == 1) & (df_full['round'] <= 8)]
if len(df_be_fin) > 100:
    formula = 'belief_error ~ round + coop + o_coop_m1'
    df_temp = df_be_fin.dropna(subset=['belief_error', 'round', 'coop', 'o_coop_m1'])
    if len(df_temp) > 50:
        result = run_ols_clustered(formula, df_temp)
        if result:
            add_result('custom/belief_error_finite', 'custom',
                      'belief_error', 'round', result, 'Finite, belief error determinants',
                      'OLS', 'session', 'coop+o_coop_m1', 'none', df_temp, formula)

print("\n14.2 Belief error determinants: Indefinite, late supergames")
df_be_inf = df_full[(df_full['finite'] == 0) & (df_full['late'] == 1) & (df_full['round'] <= 8)]
if len(df_be_inf) > 100:
    formula = 'belief_error ~ round + coop + o_coop_m1'
    df_temp = df_be_inf.dropna(subset=['belief_error', 'round', 'coop', 'o_coop_m1'])
    if len(df_temp) > 50:
        result = run_ols_clustered(formula, df_temp)
        if result:
            add_result('custom/belief_error_indefinite', 'custom',
                      'belief_error', 'round', result, 'Indefinite, belief error determinants',
                      'OLS', 'session', 'coop+o_coop_m1', 'none', df_temp, formula)


# ========================================
# SECTION 15: ADDITIONAL INFERENCE VARIATIONS
# ========================================
print("\n" + "="*60)
print("SECTION 15: ADDITIONAL INFERENCE VARIATIONS")
print("="*60)

# HC3 standard errors
print("\n15.1 HC3 standard errors: Finite")
if len(df_finite) > 50:
    formula = 'coop ~ beliefon + supergame + focoop_m1 + fcoop + risk'
    model = smf.probit(formula, data=df_finite)
    result = model.fit(cov_type='HC3', disp=0)
    if result:
        add_result('robust/se/hc3_finite', 'robustness/clustering_variations.md#alternative-se-methods',
                  'coop', 'beliefon', result, 'Finite game, HC3 SE',
                  'probit', 'none (HC3)', 'supergame+focoop_m1+fcoop+risk', 'none', df_finite, formula)

print("\n15.2 HC3 standard errors: Indefinite")
if len(df_infinite) > 50:
    formula = 'coop ~ beliefon + supergame + focoop_m1 + fcoop + risk + length_m1'
    model = smf.probit(formula, data=df_infinite)
    result = model.fit(cov_type='HC3', disp=0)
    if result:
        add_result('robust/se/hc3_indefinite', 'robustness/clustering_variations.md#alternative-se-methods',
                  'coop', 'beliefon', result, 'Indefinite game, HC3 SE',
                  'probit', 'none (HC3)', 'supergame+focoop_m1+fcoop+risk+length_m1', 'none', df_infinite, formula)


# ========================================
# Save Results
# ========================================
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
output_file = f'{OUTPUT_PATH}/specification_results.csv'
results_df.to_csv(output_file, index=False)
print(f"\nSaved {len(results_df)} specifications to {output_file}")

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

# Category breakdown
print("\n\nSpecification breakdown by category:")
results_df['category'] = results_df['spec_id'].apply(lambda x: x.split('/')[0] if '/' in x else 'baseline')
category_summary = results_df.groupby('category').agg({
    'coefficient': ['count', lambda x: (x > 0).mean(), 'mean'],
    'p_value': lambda x: (x < 0.05).mean()
}).round(3)
category_summary.columns = ['N', 'pct_positive', 'mean_coef', 'pct_sig_05']
print(category_summary)

print("\n\nDone!")
