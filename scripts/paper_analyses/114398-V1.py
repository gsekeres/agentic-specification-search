#!/usr/bin/env python3
"""
Specification Search: 114398-V1
Paper: "Church-Based Childcare" by Rennhoff & Owens
Journal: AER (Papers & Proceedings)

This paper uses structural GMM estimation of a church entry game model.
The original estimation requires MATLAB simulation which is not available.

We conduct reduced-form discrete choice analysis (probit/logit) as approximations
to understand the relationships between competition, demographics, and church
childcare provision.

Key variables:
- Outcome: ccare (binary - whether church provides childcare)
- Treatment/exposure: Competition from other churches (same/different denomination)
- Controls: Demographics, church characteristics, for-profit competitor presence
"""

import pandas as pd
import numpy as np
import json
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_PATH = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search'
DATA_PATH = f'{BASE_PATH}/data/downloads/extracted/114398-V1'
OUTPUT_PATH = DATA_PATH

# Paper metadata
PAPER_ID = '114398-V1'
PAPER_TITLE = 'Church-Based Childcare'
JOURNAL = 'AER-PP'  # AER Papers & Proceedings

# Load data
df = pd.read_csv(f'{DATA_PATH}/analysis_data.csv')

# Define variable groups
OUTCOME = 'ccare'

# Main treatment: competition from other churches
# Paper's key hypothesis: competition effects differ by denomination type
TREATMENT_VARS = ['n_same_4mi', 'n_diff_4mi']  # Primary: 4-mile radius same/diff denomination

COMPETITION_VARS = [
    'n_same_4mi', 'n_same_4_8mi', 'n_same_8_12mi',  # Same denomination by distance
    'n_diff_4mi', 'n_diff_4_8mi', 'n_diff_8_12mi',  # Different denomination by distance
    'n_total_4mi', 'n_total_4_8mi', 'n_total_8_12mi'  # Total competitors
]

# Church characteristics
CHURCH_CHARS = ['wmson', 'small', 'large', 'new_church', 'big50']

# Demographics (demand shifters)
DEMOGRAPHICS = ['pop_1000', 'pct_under5', 'pct_married', 'hh_income', 'pct_dual_income',
                'adherents_1000', 'workers']

# For-profit competitors
FP_VARS = ['dist_to_fp', 'fp_capacity']

# All controls
ALL_CONTROLS = CHURCH_CHARS + DEMOGRAPHICS + FP_VARS

# Location dummies
LOCATION_DUMMIES = ['bwood', 'fairview', 'franklin', 'wm_rural', 'lavergne', 'boro', 'smyrna', 'ruth_rural']

# Store results
results = []

def extract_results(model, spec_id, spec_tree_path, outcome_var, treatment_var,
                    controls_desc, model_type, sample_desc='Full sample',
                    fixed_effects='None', cluster_var='None'):
    """Extract results from a fitted model"""

    # Get coefficients, SEs, etc.
    coef = model.params.get(treatment_var, np.nan)
    se = model.bse.get(treatment_var, np.nan)
    pval = model.pvalues.get(treatment_var, np.nan)
    tstat = model.tvalues.get(treatment_var, np.nan)

    # CI
    try:
        conf = model.conf_int()
        ci_lower = conf.loc[treatment_var, 0] if treatment_var in conf.index else np.nan
        ci_upper = conf.loc[treatment_var, 1] if treatment_var in conf.index else np.nan
    except:
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se

    # Build coefficient vector JSON
    coef_vector = {
        'treatment': {
            'var': treatment_var,
            'coef': float(coef) if not np.isnan(coef) else None,
            'se': float(se) if not np.isnan(se) else None,
            'pval': float(pval) if not np.isnan(pval) else None
        },
        'controls': [],
        'fixed_effects': fixed_effects.split(', ') if fixed_effects != 'None' else [],
        'diagnostics': {
            'pseudo_r2': float(model.prsquared) if hasattr(model, 'prsquared') else None,
            'r_squared': float(model.rsquared) if hasattr(model, 'rsquared') else None,
            'll_model': float(model.llf) if hasattr(model, 'llf') else None,
            'll_null': float(model.llnull) if hasattr(model, 'llnull') else None
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

    # Get marginal effects for logit/probit
    mfx = None
    mfx_se = None
    if model_type in ['Logit', 'Probit']:
        try:
            mfx_result = model.get_margeff(at='overall')
            mfx = mfx_result.margeff[list(model.params.index[1:]).index(treatment_var)]
            mfx_se = mfx_result.margeff_se[list(model.params.index[1:]).index(treatment_var)]
            coef_vector['treatment']['marginal_effect'] = float(mfx)
            coef_vector['treatment']['mfx_se'] = float(mfx_se)
        except:
            pass

    # R-squared or pseudo R-squared
    if hasattr(model, 'prsquared'):
        r2 = model.prsquared
    elif hasattr(model, 'rsquared'):
        r2 = model.rsquared
    else:
        r2 = np.nan

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
        't_stat': tstat,
        'p_value': pval,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_obs': int(model.nobs),
        'r_squared': r2,
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
# The paper's structural model cannot be replicated without MATLAB.
# We run reduced-form probit/logit as baseline approximations.
# ============================================================================

print("="*80)
print("SPECIFICATION SEARCH: 114398-V1")
print("Church-Based Childcare (Rennhoff & Owens)")
print("="*80)

# BASELINE: Logit with main competition variables and baseline controls
print("\n[1] Running baseline specifications...")

# Construct formula with all baseline controls
baseline_controls = CHURCH_CHARS + DEMOGRAPHICS + FP_VARS
control_formula = ' + '.join(baseline_controls)
formula = f'{OUTCOME} ~ n_same_4mi + n_diff_4mi + {control_formula}'

# Baseline Logit
try:
    model_logit = smf.logit(formula, data=df).fit(disp=0)
    results.append(extract_results(
        model_logit, 'baseline',
        'methods/discrete_choice.md#baseline',
        OUTCOME, 'n_same_4mi',
        'Church chars + demographics + FP competition',
        'Logit'
    ))
    # Also add different denomination competition as second treatment
    results.append(extract_results(
        model_logit, 'baseline_diff_denom',
        'methods/discrete_choice.md#baseline',
        OUTCOME, 'n_diff_4mi',
        'Church chars + demographics + FP competition',
        'Logit'
    ))
    print(f"  Baseline Logit: n_same_4mi coef = {model_logit.params['n_same_4mi']:.4f}, p = {model_logit.pvalues['n_same_4mi']:.4f}")
    print(f"                  n_diff_4mi coef = {model_logit.params['n_diff_4mi']:.4f}, p = {model_logit.pvalues['n_diff_4mi']:.4f}")
except Exception as e:
    print(f"  Error in baseline logit: {e}")

# Baseline Probit
try:
    model_probit = smf.probit(formula, data=df).fit(disp=0)
    results.append(extract_results(
        model_probit, 'discrete/binary/probit',
        'methods/discrete_choice.md#model-type-binary-outcome',
        OUTCOME, 'n_same_4mi',
        'Church chars + demographics + FP competition',
        'Probit'
    ))
    results.append(extract_results(
        model_probit, 'discrete/binary/probit_diff_denom',
        'methods/discrete_choice.md#model-type-binary-outcome',
        OUTCOME, 'n_diff_4mi',
        'Church chars + demographics + FP competition',
        'Probit'
    ))
    print(f"  Baseline Probit: n_same_4mi coef = {model_probit.params['n_same_4mi']:.4f}, p = {model_probit.pvalues['n_same_4mi']:.4f}")
except Exception as e:
    print(f"  Error in baseline probit: {e}")

# LPM (Linear Probability Model)
try:
    model_lpm = smf.ols(formula, data=df).fit()
    results.append(extract_results(
        model_lpm, 'discrete/binary/lpm',
        'methods/discrete_choice.md#model-type-binary-outcome',
        OUTCOME, 'n_same_4mi',
        'Church chars + demographics + FP competition',
        'LPM'
    ))
    results.append(extract_results(
        model_lpm, 'discrete/binary/lpm_diff_denom',
        'methods/discrete_choice.md#model-type-binary-outcome',
        OUTCOME, 'n_diff_4mi',
        'Church chars + demographics + FP competition',
        'LPM'
    ))
    print(f"  LPM: n_same_4mi coef = {model_lpm.params['n_same_4mi']:.4f}, p = {model_lpm.pvalues['n_same_4mi']:.4f}")
except Exception as e:
    print(f"  Error in LPM: {e}")


# ============================================================================
# DISCRETE CHOICE VARIATIONS
# ============================================================================

print("\n[2] Running discrete choice variations...")

# Different distance bands for competition
distance_specs = [
    ('n_same_4mi', 'n_diff_4mi', '4mi'),
    ('n_same_4_8mi', 'n_diff_4_8mi', '4_8mi'),
    ('n_same_8_12mi', 'n_diff_8_12mi', '8_12mi'),
]

for same_var, diff_var, dist_label in distance_specs:
    formula = f'{OUTCOME} ~ {same_var} + {diff_var} + {control_formula}'
    try:
        model = smf.logit(formula, data=df).fit(disp=0)
        results.append(extract_results(
            model, f'discrete/distance/{dist_label}_same',
            'methods/discrete_choice.md#control-sets',
            OUTCOME, same_var,
            f'Competition at {dist_label} + baseline controls',
            'Logit'
        ))
        results.append(extract_results(
            model, f'discrete/distance/{dist_label}_diff',
            'methods/discrete_choice.md#control-sets',
            OUTCOME, diff_var,
            f'Competition at {dist_label} + baseline controls',
            'Logit'
        ))
        print(f"  Distance {dist_label}: same coef = {model.params[same_var]:.4f}, diff coef = {model.params[diff_var]:.4f}")
    except Exception as e:
        print(f"  Error for distance {dist_label}: {e}")

# Total competitors (not differentiated by denomination)
for dist in ['4mi', '4_8mi', '8_12mi']:
    var = f'n_total_{dist}'
    formula = f'{OUTCOME} ~ {var} + {control_formula}'
    try:
        model = smf.logit(formula, data=df).fit(disp=0)
        results.append(extract_results(
            model, f'discrete/distance/total_{dist}',
            'methods/discrete_choice.md#control-sets',
            OUTCOME, var,
            f'Total competition at {dist} + baseline controls',
            'Logit'
        ))
        print(f"  Total {dist}: coef = {model.params[var]:.4f}, p = {model.pvalues[var]:.4f}")
    except Exception as e:
        print(f"  Error for total {dist}: {e}")

# All distance bands simultaneously
formula_all_dist = f'{OUTCOME} ~ n_same_4mi + n_same_4_8mi + n_same_8_12mi + n_diff_4mi + n_diff_4_8mi + n_diff_8_12mi + {control_formula}'
try:
    model = smf.logit(formula_all_dist, data=df).fit(disp=0)
    for var in ['n_same_4mi', 'n_same_4_8mi', 'n_same_8_12mi', 'n_diff_4mi', 'n_diff_4_8mi', 'n_diff_8_12mi']:
        results.append(extract_results(
            model, f'discrete/full_distance/{var}',
            'methods/discrete_choice.md#control-sets',
            OUTCOME, var,
            'All distance bands + baseline controls',
            'Logit'
        ))
    print(f"  Full distance model: Pseudo R2 = {model.prsquared:.4f}")
except Exception as e:
    print(f"  Error for full distance model: {e}")


# ============================================================================
# ROBUSTNESS: LEAVE-ONE-OUT
# ============================================================================

print("\n[3] Running leave-one-out robustness...")

for drop_var in baseline_controls:
    remaining_controls = [c for c in baseline_controls if c != drop_var]
    formula = f'{OUTCOME} ~ n_same_4mi + n_diff_4mi + {" + ".join(remaining_controls)}'
    try:
        model = smf.logit(formula, data=df).fit(disp=0)
        results.append(extract_results(
            model, f'robust/loo/drop_{drop_var}',
            'robustness/leave_one_out.md',
            OUTCOME, 'n_same_4mi',
            f'Baseline minus {drop_var}',
            'Logit'
        ))
    except Exception as e:
        print(f"  Error dropping {drop_var}: {e}")

print(f"  Completed {len(baseline_controls)} leave-one-out specifications")


# ============================================================================
# ROBUSTNESS: SINGLE COVARIATE
# ============================================================================

print("\n[4] Running single covariate robustness...")

# Bivariate (no controls)
formula = f'{OUTCOME} ~ n_same_4mi + n_diff_4mi'
try:
    model = smf.logit(formula, data=df).fit(disp=0)
    results.append(extract_results(
        model, 'robust/single/none',
        'robustness/single_covariate.md',
        OUTCOME, 'n_same_4mi',
        'None (bivariate)',
        'Logit'
    ))
    print(f"  Bivariate: n_same_4mi coef = {model.params['n_same_4mi']:.4f}, p = {model.pvalues['n_same_4mi']:.4f}")
except Exception as e:
    print(f"  Error in bivariate: {e}")

# Single covariate
for control in baseline_controls:
    formula = f'{OUTCOME} ~ n_same_4mi + n_diff_4mi + {control}'
    try:
        model = smf.logit(formula, data=df).fit(disp=0)
        results.append(extract_results(
            model, f'robust/single/{control}',
            'robustness/single_covariate.md',
            OUTCOME, 'n_same_4mi',
            f'Only {control}',
            'Logit'
        ))
    except Exception as e:
        print(f"  Error with single {control}: {e}")

print(f"  Completed {len(baseline_controls) + 1} single covariate specifications")


# ============================================================================
# ROBUSTNESS: CLUSTERING
# ============================================================================

print("\n[5] Running clustering variations...")

# Robust SE
try:
    model = smf.logit(f'{OUTCOME} ~ n_same_4mi + n_diff_4mi + {control_formula}', data=df).fit(disp=0, cov_type='HC1')
    results.append(extract_results(
        model, 'robust/se/hc1',
        'robustness/clustering_variations.md#alternative-se-methods',
        OUTCOME, 'n_same_4mi',
        'Baseline controls',
        'Logit',
        cluster_var='HC1 Robust'
    ))
    print(f"  HC1: SE = {model.bse['n_same_4mi']:.4f}")
except Exception as e:
    print(f"  Error with HC1: {e}")

# Cluster by county (wmson)
try:
    model = smf.logit(f'{OUTCOME} ~ n_same_4mi + n_diff_4mi + {control_formula}', data=df).fit(disp=0, cov_type='cluster', cov_kwds={'groups': df['wmson']})
    results.append(extract_results(
        model, 'robust/cluster/county',
        'robustness/clustering_variations.md#single-level-clustering',
        OUTCOME, 'n_same_4mi',
        'Baseline controls',
        'Logit',
        cluster_var='County (wmson)'
    ))
    print(f"  Cluster by county: SE = {model.bse['n_same_4mi']:.4f}")
except Exception as e:
    print(f"  Error clustering by county: {e}")


# ============================================================================
# ROBUSTNESS: SAMPLE RESTRICTIONS
# ============================================================================

print("\n[6] Running sample restriction robustness...")

# By county
for county, county_name in [(1, 'Williamson'), (0, 'Rutherford')]:
    df_sub = df[df['wmson'] == county]
    formula = f'{OUTCOME} ~ n_same_4mi + n_diff_4mi + {" + ".join([c for c in baseline_controls if c != "wmson"])}'
    try:
        model = smf.logit(formula, data=df_sub).fit(disp=0)
        results.append(extract_results(
            model, f'robust/sample/{county_name.lower()}_only',
            'robustness/sample_restrictions.md#geographic-unit-restrictions',
            OUTCOME, 'n_same_4mi',
            'Baseline controls (excl county dummy)',
            'Logit',
            sample_desc=f'{county_name} County only'
        ))
        print(f"  {county_name} only: n={len(df_sub)}, coef = {model.params['n_same_4mi']:.4f}")
    except Exception as e:
        print(f"  Error for {county_name}: {e}")

# Large churches only
df_large = df[df['big50'] == 1]
try:
    model = smf.logit(f'{OUTCOME} ~ n_same_4mi + n_diff_4mi + {control_formula}', data=df_large).fit(disp=0)
    results.append(extract_results(
        model, 'robust/sample/large_only',
        'robustness/sample_restrictions.md#demographic-subgroups',
        OUTCOME, 'n_same_4mi',
        'Baseline controls',
        'Logit',
        sample_desc='Large churches (50+ capacity) only'
    ))
    print(f"  Large churches only: n={len(df_large)}, coef = {model.params['n_same_4mi']:.4f}")
except Exception as e:
    print(f"  Error for large churches: {e}")

# New vs established churches
for new_val, new_label in [(1, 'new'), (0, 'established')]:
    df_sub = df[df['new_church'] == new_val]
    formula = f'{OUTCOME} ~ n_same_4mi + n_diff_4mi + {" + ".join([c for c in baseline_controls if c != "new_church"])}'
    try:
        model = smf.logit(formula, data=df_sub).fit(disp=0)
        results.append(extract_results(
            model, f'robust/sample/{new_label}_churches',
            'robustness/sample_restrictions.md#demographic-subgroups',
            OUTCOME, 'n_same_4mi',
            'Baseline controls (excl new_church)',
            'Logit',
            sample_desc=f'{new_label.capitalize()} churches only'
        ))
        print(f"  {new_label.capitalize()} churches: n={len(df_sub)}, coef = {model.params['n_same_4mi']:.4f}")
    except Exception as e:
        print(f"  Error for {new_label}: {e}")

# High vs low population areas
pop_median = df['pop_1000'].median()
for pop_type, pop_label in [(df['pop_1000'] >= pop_median, 'high_pop'),
                            (df['pop_1000'] < pop_median, 'low_pop')]:
    df_sub = df[pop_type]
    try:
        model = smf.logit(f'{OUTCOME} ~ n_same_4mi + n_diff_4mi + {control_formula}', data=df_sub).fit(disp=0)
        results.append(extract_results(
            model, f'robust/sample/{pop_label}',
            'robustness/sample_restrictions.md#demographic-subgroups',
            OUTCOME, 'n_same_4mi',
            'Baseline controls',
            'Logit',
            sample_desc=f'{pop_label.replace("_", " ").title()} areas'
        ))
        print(f"  {pop_label}: n={len(df_sub)}, coef = {model.params['n_same_4mi']:.4f}")
    except Exception as e:
        print(f"  Error for {pop_label}: {e}")


# ============================================================================
# ROBUSTNESS: FUNCTIONAL FORM
# ============================================================================

print("\n[7] Running functional form robustness...")

# Quadratic competition effects
formula = f'{OUTCOME} ~ n_same_4mi + I(n_same_4mi**2) + n_diff_4mi + I(n_diff_4mi**2) + {control_formula}'
try:
    model = smf.logit(formula, data=df).fit(disp=0)
    results.append(extract_results(
        model, 'robust/form/quadratic',
        'robustness/functional_form.md#nonlinear-specifications',
        OUTCOME, 'n_same_4mi',
        'Baseline controls + quadratic competition',
        'Logit'
    ))
    print(f"  Quadratic: linear coef = {model.params['n_same_4mi']:.4f}, squared coef = {model.params['I(n_same_4mi ** 2)']:.4f}")
except Exception as e:
    print(f"  Error with quadratic: {e}")

# Log-transformed demographics
df_log = df.copy()
for var in ['pop_1000', 'hh_income', 'workers', 'fp_capacity']:
    df_log[f'log_{var}'] = np.log(df[var] + 1)

log_controls = ['wmson', 'small', 'large', 'new_church', 'big50',
                'log_pop_1000', 'pct_under5', 'pct_married', 'log_hh_income',
                'pct_dual_income', 'adherents_1000', 'log_workers',
                'dist_to_fp', 'log_fp_capacity']
formula = f'{OUTCOME} ~ n_same_4mi + n_diff_4mi + {" + ".join(log_controls)}'
try:
    model = smf.logit(formula, data=df_log).fit(disp=0)
    results.append(extract_results(
        model, 'robust/form/controls_log',
        'robustness/functional_form.md#control-variable-transformations',
        OUTCOME, 'n_same_4mi',
        'Log-transformed continuous controls',
        'Logit'
    ))
    print(f"  Log controls: coef = {model.params['n_same_4mi']:.4f}")
except Exception as e:
    print(f"  Error with log controls: {e}")

# Interaction: competition * county
formula = f'{OUTCOME} ~ n_same_4mi * wmson + n_diff_4mi * wmson + {" + ".join([c for c in baseline_controls if c != "wmson"])}'
try:
    model = smf.logit(formula, data=df).fit(disp=0)
    results.append(extract_results(
        model, 'robust/form/interact_county',
        'robustness/functional_form.md#interaction-terms',
        OUTCOME, 'n_same_4mi',
        'Competition x county interaction',
        'Logit'
    ))
    print(f"  Interaction (county): main effect = {model.params['n_same_4mi']:.4f}, interaction = {model.params['n_same_4mi:wmson']:.4f}")
except Exception as e:
    print(f"  Error with county interaction: {e}")


# ============================================================================
# CONTROL SETS
# ============================================================================

print("\n[8] Running control set variations...")

# Only church characteristics
formula = f'{OUTCOME} ~ n_same_4mi + n_diff_4mi + {" + ".join(CHURCH_CHARS)}'
try:
    model = smf.logit(formula, data=df).fit(disp=0)
    results.append(extract_results(
        model, 'discrete/controls/church_only',
        'methods/discrete_choice.md#control-sets',
        OUTCOME, 'n_same_4mi',
        'Church characteristics only',
        'Logit'
    ))
    print(f"  Church chars only: coef = {model.params['n_same_4mi']:.4f}")
except Exception as e:
    print(f"  Error with church chars only: {e}")

# Only demographics
formula = f'{OUTCOME} ~ n_same_4mi + n_diff_4mi + {" + ".join(DEMOGRAPHICS)}'
try:
    model = smf.logit(formula, data=df).fit(disp=0)
    results.append(extract_results(
        model, 'discrete/controls/demographics_only',
        'methods/discrete_choice.md#control-sets',
        OUTCOME, 'n_same_4mi',
        'Demographics only',
        'Logit'
    ))
    print(f"  Demographics only: coef = {model.params['n_same_4mi']:.4f}")
except Exception as e:
    print(f"  Error with demographics only: {e}")

# Including location dummies
loc_dummies_formula = ' + '.join(LOCATION_DUMMIES[:-1])  # Drop one for collinearity
formula = f'{OUTCOME} ~ n_same_4mi + n_diff_4mi + {control_formula} + {loc_dummies_formula}'
try:
    model = smf.logit(formula, data=df).fit(disp=0)
    results.append(extract_results(
        model, 'discrete/controls/with_location_fe',
        'methods/discrete_choice.md#fixed-effects',
        OUTCOME, 'n_same_4mi',
        'Baseline + location dummies',
        'Logit',
        fixed_effects='Location (city)'
    ))
    print(f"  With location FE: coef = {model.params['n_same_4mi']:.4f}")
except Exception as e:
    print(f"  Error with location FE: {e}")


# ============================================================================
# STRUCTURAL CALIBRATION NOTE
# ============================================================================

print("\n[9] Note on structural model...")
print("  The original paper uses structural GMM estimation of a church entry game.")
print("  This requires MATLAB simulation which is not available in Python.")
print("  The above reduced-form specifications approximate the relationships")
print("  between competition, demographics, and childcare provision.")


# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Create DataFrame and save
results_df = pd.DataFrame(results)
results_df.to_csv(f'{OUTPUT_PATH}/specification_results.csv', index=False)
print(f"\nSaved {len(results_df)} specifications to specification_results.csv")

# Summary statistics
print("\n" + "-"*40)
print("SUMMARY STATISTICS")
print("-"*40)
print(f"Total specifications: {len(results_df)}")
print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")
print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")
print(f"Coefficient range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")

print("\nDone!")
