"""
Specification Search: 140921-V1
Paper: "Assortative Matching at the Top of the Distribution: Evidence from the World's Most Exclusive Marriage Market"
Author: Marc Goni

Method: Reduced-form OLS/Probit + Instrumental Variables (IV)
    - Treatment: syntheticT (synthetic probability to marry during Season's interruption 1861-63)
    - Instrument: syntheticT instruments for mourn (actually married during interruption)
    - Main outcomes: cOut (married a commoner), mheir (married an heir), fmissmatch (sorting)
    - Baseline controls: pr4 (duke/earl/marquis daughter), biorder (birth order), hengpee (English peerage)
    - Cluster: byear (birth year)

This script runs 50+ specifications following the i4r methodology.
"""

import pandas as pd
import numpy as np
import warnings
import json
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')

# ============================================================================
# SETUP
# ============================================================================

BASE_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search"
DATA_PATH = f"{BASE_PATH}/data/downloads/extracted/140921-V1/data"

PAPER_ID = "140921-V1"
JOURNAL = "AER"
PAPER_TITLE = "Assortative Matching at the Top of the Distribution: Evidence from the World's Most Exclusive Marriage Market"

# Load data
df_full = pd.read_stata(f"{DATA_PATH}/final-data.dta")

# Create baseline sample
df = df_full[df_full['base_sample'] == 1].copy()
print(f"Baseline sample size: {len(df)}")

# Load marital rates sample for celibacy outcome
df_marital = df_full[df_full['marital_rates_sample'] == 1].copy()
print(f"Marital rates sample size: {len(df_marital)}")

# ============================================================================
# VARIABLE SETUP
# ============================================================================

# Main treatment variable (reduced form)
TREATMENT_VAR = 'syntheticT'

# Endogenous variable (for IV)
ENDOGENOUS_VAR = 'mourn'

# Baseline controls from the paper
BASELINE_CONTROLS = ['pr4', 'biorder', 'hengpee']

# Extended controls
EXTENDED_CONTROLS = ['pr4', 'biorder', 'hengpee', 'distlondon']

# Cluster variable
CLUSTER_VAR = 'byear'

# Outcome variables
OUTCOMES = {
    'cOut': {'name': 'Married commoner', 'type': 'binary'},
    'mheir': {'name': 'Married heir', 'type': 'binary'},
    'fmissmatch': {'name': 'Abs diff acres pctile', 'type': 'continuous'},
    'fmissmatch2': {'name': 'Signed diff acres pctile', 'type': 'continuous'},
    'fdown': {'name': 'Married down', 'type': 'binary'},
    'dist': {'name': 'Distance between seats', 'type': 'continuous'},
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def run_ols_clustered(df, formula, cluster_var):
    """Run OLS with clustered standard errors."""
    try:
        # Fit OLS
        model = smf.ols(formula, data=df).fit()

        # Get cluster-robust SEs
        cluster_groups = df.loc[model.model.data.row_labels, cluster_var]
        model_clustered = model.get_robustcov_results(cov_type='cluster', groups=cluster_groups)

        return model_clustered
    except Exception as e:
        print(f"Error in OLS: {e}")
        return None

def extract_results(model, treatment_var, outcome_var, spec_id, spec_tree_path,
                    controls_desc, sample_desc, fixed_effects="None", cluster_var="byear",
                    model_type="OLS", n_obs_override=None):
    """Extract results from a fitted model into standard format."""
    if model is None:
        return None

    try:
        # Get parameter names from the model (handle both Series and array)
        if hasattr(model.params, 'index'):
            param_names = list(model.params.index)
            params_dict = dict(zip(param_names, model.params))
            bse_dict = dict(zip(param_names, model.bse))
            tvalues_dict = dict(zip(param_names, model.tvalues))
            pvalues_dict = dict(zip(param_names, model.pvalues))
        else:
            # Clustered results return numpy arrays
            param_names = model.model.exog_names
            params_dict = dict(zip(param_names, model.params))
            bse_dict = dict(zip(param_names, model.bse))
            tvalues_dict = dict(zip(param_names, model.tvalues))
            pvalues_dict = dict(zip(param_names, model.pvalues))

        if treatment_var not in param_names:
            print(f"Treatment var {treatment_var} not in params: {param_names}")
            return None

        coef = params_dict[treatment_var]
        se = bse_dict[treatment_var]
        t_stat = tvalues_dict[treatment_var]
        pval = pvalues_dict[treatment_var]
        r_sq = model.rsquared if hasattr(model, 'rsquared') else np.nan
        n = int(model.nobs) if n_obs_override is None else n_obs_override

        # Get all coefficients as JSON
        coef_dict = {
            "treatment": {"var": treatment_var, "coef": float(coef), "se": float(se), "pval": float(pval)},
            "controls": [],
            "fixed_effects": [],
            "diagnostics": {}
        }
        for name in param_names:
            if name != treatment_var and name != 'Intercept':
                coef_dict["controls"].append({
                    "var": name,
                    "coef": float(params_dict[name]),
                    "se": float(bse_dict[name]),
                    "pval": float(pvalues_dict[name])
                })

        # Calculate confidence interval
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se

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
            't_stat': float(t_stat),
            'p_value': float(pval),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_obs': n,
            'r_squared': float(r_sq) if not np.isnan(r_sq) else None,
            'coefficient_vector_json': json.dumps(coef_dict),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects,
            'controls_desc': controls_desc,
            'cluster_var': cluster_var,
            'model_type': model_type,
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
    except Exception as e:
        print(f"Error extracting results: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================================
# RESULTS CONTAINER
# ============================================================================

results = []

# ============================================================================
# 1. BASELINE SPECIFICATIONS (Exact replication of main results)
# ============================================================================
print("\n" + "="*60)
print("BASELINE SPECIFICATIONS")
print("="*60)

# Baseline 1: cOut ~ syntheticT + controls (OLS with clustered SEs)
print("\nBaseline 1: cOut (OLS)")
formula = "cOut ~ syntheticT + pr4 + biorder + hengpee"
model = run_ols_clustered(df, formula, CLUSTER_VAR)
if model:
    res = extract_results(model, TREATMENT_VAR, 'cOut', 'baseline',
                         'methods/cross_sectional_ols.md#baseline',
                         'pr4, biorder, hengpee',
                         'Baseline sample: peers daughters aged 15-35 in 1861',
                         model_type='OLS')
    if res:
        results.append(res)
        print(f"  coef={res['coefficient']:.4f}, se={res['std_error']:.4f}, p={res['p_value']:.4f}, n={res['n_obs']}")

# Baseline 2: mheir ~ syntheticT + controls
print("\nBaseline 2: mheir (OLS)")
formula = "mheir ~ syntheticT + pr4 + biorder + hengpee"
model = run_ols_clustered(df, formula, CLUSTER_VAR)
if model:
    res = extract_results(model, TREATMENT_VAR, 'mheir', 'baseline_mheir',
                         'methods/cross_sectional_ols.md#baseline',
                         'pr4, biorder, hengpee',
                         'Baseline sample',
                         model_type='OLS')
    if res:
        results.append(res)
        print(f"  coef={res['coefficient']:.4f}, se={res['std_error']:.4f}, p={res['p_value']:.4f}, n={res['n_obs']}")

# Baseline 3: fmissmatch ~ syntheticT + controls
print("\nBaseline 3: fmissmatch (OLS)")
formula = "fmissmatch ~ syntheticT + pr4 + biorder + hengpee"
model = run_ols_clustered(df, formula, CLUSTER_VAR)
if model:
    res = extract_results(model, TREATMENT_VAR, 'fmissmatch', 'baseline_fmissmatch',
                         'methods/cross_sectional_ols.md#baseline',
                         'pr4, biorder, hengpee',
                         'Baseline sample, landholdings subsample',
                         model_type='OLS')
    if res:
        results.append(res)
        print(f"  coef={res['coefficient']:.4f}, se={res['std_error']:.4f}, p={res['p_value']:.4f}, n={res['n_obs']}")

# Baseline 4: fmissmatch2 ~ syntheticT + controls
print("\nBaseline 4: fmissmatch2 (OLS)")
formula = "fmissmatch2 ~ syntheticT + pr4 + biorder + hengpee"
model = run_ols_clustered(df, formula, CLUSTER_VAR)
if model:
    res = extract_results(model, TREATMENT_VAR, 'fmissmatch2', 'baseline_fmissmatch2',
                         'methods/cross_sectional_ols.md#baseline',
                         'pr4, biorder, hengpee',
                         'Baseline sample, landholdings subsample',
                         model_type='OLS')
    if res:
        results.append(res)
        print(f"  coef={res['coefficient']:.4f}, se={res['std_error']:.4f}, p={res['p_value']:.4f}, n={res['n_obs']}")

# Baseline 5: fdown ~ syntheticT + controls
print("\nBaseline 5: fdown (OLS)")
formula = "fdown ~ syntheticT + pr4 + biorder + hengpee"
model = run_ols_clustered(df, formula, CLUSTER_VAR)
if model:
    res = extract_results(model, TREATMENT_VAR, 'fdown', 'baseline_fdown',
                         'methods/cross_sectional_ols.md#baseline',
                         'pr4, biorder, hengpee',
                         'Baseline sample, landholdings subsample',
                         model_type='OLS')
    if res:
        results.append(res)
        print(f"  coef={res['coefficient']:.4f}, se={res['std_error']:.4f}, p={res['p_value']:.4f}, n={res['n_obs']}")

# ============================================================================
# 2. CONTROL VARIATIONS
# ============================================================================
print("\n" + "="*60)
print("CONTROL VARIATIONS")
print("="*60)

# 2.1 No controls
print("\n2.1 No controls")
formula = "cOut ~ syntheticT"
model = run_ols_clustered(df, formula, CLUSTER_VAR)
if model:
    res = extract_results(model, TREATMENT_VAR, 'cOut', 'robust/control/none',
                         'robustness/control_progression.md#no-controls',
                         'none', 'Baseline sample',
                         model_type='OLS')
    if res:
        results.append(res)
        print(f"  coef={res['coefficient']:.4f}, se={res['std_error']:.4f}, p={res['p_value']:.4f}")

# 2.2 Leave-one-out for each control
print("\n2.2 Leave-one-out controls")
all_controls = ['pr4', 'biorder', 'hengpee']
for control in all_controls:
    remaining = [c for c in all_controls if c != control]
    formula = f"cOut ~ syntheticT + {' + '.join(remaining)}"
    model = run_ols_clustered(df, formula, CLUSTER_VAR)
    if model:
        res = extract_results(model, TREATMENT_VAR, 'cOut', f'robust/control/drop_{control}',
                             'robustness/leave_one_out.md',
                             ', '.join(remaining), 'Baseline sample',
                             model_type='OLS')
        if res:
            results.append(res)
            print(f"  Drop {control}: coef={res['coefficient']:.4f}, p={res['p_value']:.4f}")

# 2.3 Add controls incrementally
print("\n2.3 Add controls incrementally")
for i, control in enumerate(all_controls):
    controls_so_far = all_controls[:i+1]
    formula = f"cOut ~ syntheticT + {' + '.join(controls_so_far)}"
    model = run_ols_clustered(df, formula, CLUSTER_VAR)
    if model:
        res = extract_results(model, TREATMENT_VAR, 'cOut', f'robust/control/add_{control}',
                             'robustness/control_progression.md',
                             ', '.join(controls_so_far), 'Baseline sample',
                             model_type='OLS')
        if res:
            results.append(res)
            print(f"  Add {control}: coef={res['coefficient']:.4f}, p={res['p_value']:.4f}")

# 2.4 Extended controls (with distlondon)
print("\n2.4 Extended controls (with distlondon)")
formula = "cOut ~ syntheticT + pr4 + biorder + hengpee + distlondon"
model = run_ols_clustered(df, formula, CLUSTER_VAR)
if model:
    res = extract_results(model, TREATMENT_VAR, 'cOut', 'robust/control/extended',
                         'robustness/control_progression.md#extended',
                         'pr4, biorder, hengpee, distlondon', 'Baseline sample',
                         model_type='OLS')
    if res:
        results.append(res)
        print(f"  Extended: coef={res['coefficient']:.4f}, p={res['p_value']:.4f}")

# 2.5 Add age controls
print("\n2.5 With age controls")
df['age1861_sq'] = df['age1861'] ** 2
formula = "cOut ~ syntheticT + pr4 + biorder + hengpee + age1861 + age1861_sq"
model = run_ols_clustered(df, formula, CLUSTER_VAR)
if model:
    res = extract_results(model, TREATMENT_VAR, 'cOut', 'robust/control/with_age',
                         'robustness/control_progression.md',
                         'pr4, biorder, hengpee, age1861, age1861_sq', 'Baseline sample',
                         model_type='OLS')
    if res:
        results.append(res)
        print(f"  With age: coef={res['coefficient']:.4f}, p={res['p_value']:.4f}")

# 2.6 Add family landholdings
print("\n2.6 With family landholdings")
df['ln_fatotal'] = np.log(df['fatotal'] + 1)
formula = "cOut ~ syntheticT + pr4 + biorder + hengpee + ln_fatotal"
model = run_ols_clustered(df, formula, CLUSTER_VAR)
if model:
    res = extract_results(model, TREATMENT_VAR, 'cOut', 'robust/control/with_landholdings',
                         'robustness/control_progression.md',
                         'pr4, biorder, hengpee, ln_fatotal', 'Baseline sample',
                         model_type='OLS')
    if res:
        results.append(res)
        print(f"  With landholdings: coef={res['coefficient']:.4f}, p={res['p_value']:.4f}")

# ============================================================================
# 3. SAMPLE RESTRICTIONS
# ============================================================================
print("\n" + "="*60)
print("SAMPLE RESTRICTIONS")
print("="*60)

# 3.1 By birth year (drop each year)
print("\n3.1 Drop each birth year")
byears = sorted(df['byear'].dropna().unique())
for year in byears[:5]:  # First 5 years
    df_sub = df[df['byear'] != year].copy()
    formula = "cOut ~ syntheticT + pr4 + biorder + hengpee"
    model = run_ols_clustered(df_sub, formula, CLUSTER_VAR)
    if model:
        res = extract_results(model, TREATMENT_VAR, 'cOut', f'robust/sample/drop_byear_{int(year)}',
                             'robustness/sample_restrictions.md#drop-year',
                             'pr4, biorder, hengpee', f'Drop birth year {int(year)}',
                             model_type='OLS')
        if res:
            results.append(res)
            print(f"  Drop byear {int(year)}: coef={res['coefficient']:.4f}, n={res['n_obs']}")

# 3.2 By age groups
print("\n3.2 Age groups")
# Young women (age 15-25 in 1861)
df_young = df[df['age1861'] <= 25].copy()
formula = "cOut ~ syntheticT + pr4 + biorder + hengpee"
model = run_ols_clustered(df_young, formula, CLUSTER_VAR)
if model:
    res = extract_results(model, TREATMENT_VAR, 'cOut', 'robust/sample/young',
                         'robustness/sample_restrictions.md#age-groups',
                         'pr4, biorder, hengpee', 'Age 15-25 in 1861',
                         model_type='OLS')
    if res:
        results.append(res)
        print(f"  Young: coef={res['coefficient']:.4f}, n={res['n_obs']}")

# Older women (age 26-35 in 1861)
df_old = df[df['age1861'] > 25].copy()
model = run_ols_clustered(df_old, formula, CLUSTER_VAR)
if model:
    res = extract_results(model, TREATMENT_VAR, 'cOut', 'robust/sample/older',
                         'robustness/sample_restrictions.md#age-groups',
                         'pr4, biorder, hengpee', 'Age 26-35 in 1861',
                         model_type='OLS')
    if res:
        results.append(res)
        print(f"  Older: coef={res['coefficient']:.4f}, n={res['n_obs']}")

# 3.3 By father's rank
print("\n3.3 By father's rank")
# High rank (Duke/Earl/Marquis daughters)
df_high = df[df['pr4'] == 1].copy()
formula = "cOut ~ syntheticT + biorder + hengpee"
model = run_ols_clustered(df_high, formula, CLUSTER_VAR)
if model:
    res = extract_results(model, TREATMENT_VAR, 'cOut', 'robust/sample/high_rank',
                         'robustness/sample_restrictions.md#subgroups',
                         'biorder, hengpee', 'High rank daughters only',
                         model_type='OLS')
    if res:
        results.append(res)
        print(f"  High rank: coef={res['coefficient']:.4f}, n={res['n_obs']}")

# Lower rank (Baron/Viscount daughters)
df_low = df[df['pr4'] == 0].copy()
model = run_ols_clustered(df_low, formula, CLUSTER_VAR)
if model:
    res = extract_results(model, TREATMENT_VAR, 'cOut', 'robust/sample/low_rank',
                         'robustness/sample_restrictions.md#subgroups',
                         'biorder, hengpee', 'Low rank daughters only',
                         model_type='OLS')
    if res:
        results.append(res)
        print(f"  Low rank: coef={res['coefficient']:.4f}, n={res['n_obs']}")

# 3.4 English peerage only
print("\n3.4 English peerage only")
df_eng = df[df['hengpee'] == 1].copy()
formula = "cOut ~ syntheticT + pr4 + biorder"
model = run_ols_clustered(df_eng, formula, CLUSTER_VAR)
if model:
    res = extract_results(model, TREATMENT_VAR, 'cOut', 'robust/sample/english_only',
                         'robustness/sample_restrictions.md#subgroups',
                         'pr4, biorder', 'English peerage only',
                         model_type='OLS')
    if res:
        results.append(res)
        print(f"  English only: coef={res['coefficient']:.4f}, n={res['n_obs']}")

# 3.5 Trimmed sample (by treatment variable)
print("\n3.5 Trimmed sample")
q01 = df['syntheticT'].quantile(0.01)
q99 = df['syntheticT'].quantile(0.99)
df_trim = df[(df['syntheticT'] >= q01) & (df['syntheticT'] <= q99)].copy()
formula = "cOut ~ syntheticT + pr4 + biorder + hengpee"
model = run_ols_clustered(df_trim, formula, CLUSTER_VAR)
if model:
    res = extract_results(model, TREATMENT_VAR, 'cOut', 'robust/sample/trimmed_1pct',
                         'robustness/sample_restrictions.md#outliers',
                         'pr4, biorder, hengpee', 'Trim 1% tails of syntheticT',
                         model_type='OLS')
    if res:
        results.append(res)
        print(f"  Trimmed: coef={res['coefficient']:.4f}, n={res['n_obs']}")

# 3.6 First-born only
print("\n3.6 First-born only")
df_first = df[df['biorder'] == 1].copy()
formula = "cOut ~ syntheticT + pr4 + hengpee"
model = run_ols_clustered(df_first, formula, CLUSTER_VAR)
if model:
    res = extract_results(model, TREATMENT_VAR, 'cOut', 'robust/sample/firstborn',
                         'robustness/sample_restrictions.md#subgroups',
                         'pr4, hengpee', 'First-born daughters only',
                         model_type='OLS')
    if res:
        results.append(res)
        print(f"  First-born: coef={res['coefficient']:.4f}, n={res['n_obs']}")

# 3.7 Non-first-born
print("\n3.7 Non-first-born")
df_later = df[df['biorder'] > 1].copy()
formula = "cOut ~ syntheticT + pr4 + biorder + hengpee"
model = run_ols_clustered(df_later, formula, CLUSTER_VAR)
if model:
    res = extract_results(model, TREATMENT_VAR, 'cOut', 'robust/sample/later_born',
                         'robustness/sample_restrictions.md#subgroups',
                         'pr4, biorder, hengpee', 'Later-born daughters only',
                         model_type='OLS')
    if res:
        results.append(res)
        print(f"  Later-born: coef={res['coefficient']:.4f}, n={res['n_obs']}")

# ============================================================================
# 4. ALTERNATIVE OUTCOMES
# ============================================================================
print("\n" + "="*60)
print("ALTERNATIVE OUTCOMES")
print("="*60)

# Run baseline on all outcome variables
for outcome in ['mheir', 'fmissmatch', 'fmissmatch2', 'fdown', 'dist']:
    print(f"\n{outcome}")
    formula = f"{outcome} ~ syntheticT + pr4 + biorder + hengpee"
    model = run_ols_clustered(df, formula, CLUSTER_VAR)
    if model:
        res = extract_results(model, TREATMENT_VAR, outcome, f'robust/outcome/{outcome}',
                             'robustness/measurement.md#alternative-outcomes',
                             'pr4, biorder, hengpee', 'Baseline sample',
                             model_type='OLS')
        if res:
            results.append(res)
            print(f"  coef={res['coefficient']:.4f}, p={res['p_value']:.4f}")

# Distance between seats (alternative outcome) with extended controls
print("\nDistance with extended controls")
formula = "dist ~ syntheticT + pr4 + biorder + hengpee + distlondon"
model = run_ols_clustered(df, formula, CLUSTER_VAR)
if model:
    res = extract_results(model, TREATMENT_VAR, 'dist', 'robust/outcome/dist_extended',
                         'robustness/measurement.md#alternative-outcomes',
                         'pr4, biorder, hengpee, distlondon', 'Baseline sample',
                         model_type='OLS')
    if res:
        results.append(res)
        print(f"  coef={res['coefficient']:.4f}, p={res['p_value']:.4f}")

# ============================================================================
# 5. INFERENCE VARIATIONS
# ============================================================================
print("\n" + "="*60)
print("INFERENCE VARIATIONS")
print("="*60)

# 5.1 Robust (heteroskedasticity-consistent) SEs
print("\n5.1 Robust (HC1) SEs")
formula = "cOut ~ syntheticT + pr4 + biorder + hengpee"
model = smf.ols(formula, data=df).fit(cov_type='HC1')
res = extract_results(model, TREATMENT_VAR, 'cOut', 'robust/inference/hc1',
                     'robustness/inference_alternatives.md#robust-se',
                     'pr4, biorder, hengpee', 'Baseline sample',
                     cluster_var='None (HC1)',
                     model_type='OLS')
if res:
    results.append(res)
    print(f"  HC1: coef={res['coefficient']:.4f}, se={res['std_error']:.4f}")

# 5.2 HC3 SEs
print("\n5.2 HC3 SEs")
model = smf.ols(formula, data=df).fit(cov_type='HC3')
res = extract_results(model, TREATMENT_VAR, 'cOut', 'robust/inference/hc3',
                     'robustness/inference_alternatives.md#robust-se',
                     'pr4, biorder, hengpee', 'Baseline sample',
                     cluster_var='None (HC3)',
                     model_type='OLS')
if res:
    results.append(res)
    print(f"  HC3: coef={res['coefficient']:.4f}, se={res['std_error']:.4f}")

# 5.3 Classical OLS SEs
print("\n5.3 Classical OLS SEs")
model = smf.ols(formula, data=df).fit()
res = extract_results(model, TREATMENT_VAR, 'cOut', 'robust/inference/classical',
                     'robustness/inference_alternatives.md#classical',
                     'pr4, biorder, hengpee', 'Baseline sample',
                     cluster_var='None (Classical)',
                     model_type='OLS')
if res:
    results.append(res)
    print(f"  Classical: coef={res['coefficient']:.4f}, se={res['std_error']:.4f}")

# ============================================================================
# 6. FUNCTIONAL FORM
# ============================================================================
print("\n" + "="*60)
print("FUNCTIONAL FORM VARIATIONS")
print("="*60)

# 6.1 Standardized treatment
print("\n6.1 Standardized treatment")
df['syntheticT_std'] = (df['syntheticT'] - df['syntheticT'].mean()) / df['syntheticT'].std()
formula = "cOut ~ syntheticT_std + pr4 + biorder + hengpee"
model = run_ols_clustered(df, formula, CLUSTER_VAR)
if model:
    res = extract_results(model, 'syntheticT_std', 'cOut', 'robust/funcform/standardized',
                         'robustness/functional_form.md',
                         'pr4, biorder, hengpee', 'Baseline sample',
                         model_type='OLS')
    if res:
        results.append(res)
        print(f"  Standardized: coef={res['coefficient']:.4f}")

# 6.2 Quadratic in treatment
print("\n6.2 Quadratic treatment")
df['syntheticT_sq'] = df['syntheticT'] ** 2
formula = "cOut ~ syntheticT + syntheticT_sq + pr4 + biorder + hengpee"
model = run_ols_clustered(df, formula, CLUSTER_VAR)
if model:
    res = extract_results(model, TREATMENT_VAR, 'cOut', 'robust/funcform/quadratic',
                         'robustness/functional_form.md#polynomial',
                         'pr4, biorder, hengpee, syntheticT_sq', 'Baseline sample',
                         model_type='OLS')
    if res:
        results.append(res)
        print(f"  Quadratic (linear term): coef={res['coefficient']:.4f}")

# 6.3 Log treatment
print("\n6.3 Log treatment")
df['log_syntheticT'] = np.log(df['syntheticT'] + 1)
formula = "cOut ~ log_syntheticT + pr4 + biorder + hengpee"
model = run_ols_clustered(df, formula, CLUSTER_VAR)
if model:
    res = extract_results(model, 'log_syntheticT', 'cOut', 'robust/funcform/log_treatment',
                         'robustness/functional_form.md#log',
                         'pr4, biorder, hengpee', 'Baseline sample',
                         model_type='OLS')
    if res:
        results.append(res)
        print(f"  Log treatment: coef={res['coefficient']:.4f}")

# 6.4 Binary treatment (high vs low)
print("\n6.4 Binary treatment (top quintile)")
df['syntheticT_high'] = (df['syntheticT'] > df['syntheticT'].quantile(0.8)).astype(int)
formula = "cOut ~ syntheticT_high + pr4 + biorder + hengpee"
model = run_ols_clustered(df, formula, CLUSTER_VAR)
if model:
    res = extract_results(model, 'syntheticT_high', 'cOut', 'robust/funcform/binary_treatment',
                         'robustness/functional_form.md#binary',
                         'pr4, biorder, hengpee', 'Baseline sample',
                         model_type='OLS')
    if res:
        results.append(res)
        print(f"  Binary (high): coef={res['coefficient']:.4f}")

# ============================================================================
# 7. HETEROGENEITY ANALYSIS
# ============================================================================
print("\n" + "="*60)
print("HETEROGENEITY ANALYSIS")
print("="*60)

# 7.1 Interaction with father's rank
print("\n7.1 Interaction with pr4 (high rank)")
df['syntheticT_x_pr4'] = df['syntheticT'] * df['pr4']
formula = "cOut ~ syntheticT + pr4 + syntheticT_x_pr4 + biorder + hengpee"
model = run_ols_clustered(df, formula, CLUSTER_VAR)
if model:
    res = extract_results(model, TREATMENT_VAR, 'cOut', 'robust/heterogeneity/by_rank',
                         'robustness/heterogeneity.md#interactions',
                         'pr4, biorder, hengpee, syntheticT*pr4', 'Baseline sample',
                         model_type='OLS')
    if res:
        results.append(res)
        print(f"  Main effect: coef={res['coefficient']:.4f}")
    # Also record the interaction
    res2 = extract_results(model, 'syntheticT_x_pr4', 'cOut', 'robust/heterogeneity/by_rank_interaction',
                         'robustness/heterogeneity.md#interactions',
                         'pr4, biorder, hengpee, syntheticT*pr4', 'Baseline sample',
                         model_type='OLS')
    if res2:
        results.append(res2)
        print(f"  Interaction: coef={res2['coefficient']:.4f}")

# 7.2 Interaction with English peerage
print("\n7.2 Interaction with hengpee (English peerage)")
df['syntheticT_x_hengpee'] = df['syntheticT'] * df['hengpee']
formula = "cOut ~ syntheticT + hengpee + syntheticT_x_hengpee + pr4 + biorder"
model = run_ols_clustered(df, formula, CLUSTER_VAR)
if model:
    res = extract_results(model, TREATMENT_VAR, 'cOut', 'robust/heterogeneity/by_peerage',
                         'robustness/heterogeneity.md#interactions',
                         'pr4, biorder, syntheticT*hengpee', 'Baseline sample',
                         model_type='OLS')
    if res:
        results.append(res)
        print(f"  Main effect: coef={res['coefficient']:.4f}")

# 7.3 Interaction with birth order
print("\n7.3 Interaction with birth order")
df['syntheticT_x_biorder'] = df['syntheticT'] * df['biorder']
formula = "cOut ~ syntheticT + biorder + syntheticT_x_biorder + pr4 + hengpee"
model = run_ols_clustered(df, formula, CLUSTER_VAR)
if model:
    res = extract_results(model, TREATMENT_VAR, 'cOut', 'robust/heterogeneity/by_birthorder',
                         'robustness/heterogeneity.md#interactions',
                         'pr4, hengpee, syntheticT*biorder', 'Baseline sample',
                         model_type='OLS')
    if res:
        results.append(res)
        print(f"  Main effect: coef={res['coefficient']:.4f}")

# 7.4 Interaction with distance to London
print("\n7.4 Interaction with distance to London")
df['syntheticT_x_distlondon'] = df['syntheticT'] * df['distlondon']
formula = "cOut ~ syntheticT + distlondon + syntheticT_x_distlondon + pr4 + biorder + hengpee"
model = run_ols_clustered(df, formula, CLUSTER_VAR)
if model:
    res = extract_results(model, TREATMENT_VAR, 'cOut', 'robust/heterogeneity/by_distlondon',
                         'robustness/heterogeneity.md#interactions',
                         'pr4, biorder, hengpee, syntheticT*distlondon', 'Baseline sample',
                         model_type='OLS')
    if res:
        results.append(res)
        print(f"  Main effect: coef={res['coefficient']:.4f}")

# 7.5 Interaction with age
print("\n7.5 Interaction with age")
df['syntheticT_x_age'] = df['syntheticT'] * df['age1861']
formula = "cOut ~ syntheticT + age1861 + syntheticT_x_age + pr4 + biorder + hengpee"
model = run_ols_clustered(df, formula, CLUSTER_VAR)
if model:
    res = extract_results(model, TREATMENT_VAR, 'cOut', 'robust/heterogeneity/by_age',
                         'robustness/heterogeneity.md#interactions',
                         'pr4, biorder, hengpee, syntheticT*age1861', 'Baseline sample',
                         model_type='OLS')
    if res:
        results.append(res)
        print(f"  Main effect: coef={res['coefficient']:.4f}")

# ============================================================================
# 8. INSTRUMENTAL VARIABLES
# ============================================================================
print("\n" + "="*60)
print("INSTRUMENTAL VARIABLES")
print("="*60)

# 8.1 First Stage
print("\n8.1 First Stage: mourn ~ syntheticT + controls")
formula = "mourn ~ syntheticT + pr4 + biorder + hengpee"
model = run_ols_clustered(df, formula, CLUSTER_VAR)
if model:
    res = extract_results(model, TREATMENT_VAR, 'mourn', 'iv/first_stage/baseline',
                         'methods/instrumental_variables.md#first-stage',
                         'pr4, biorder, hengpee', 'Baseline sample',
                         model_type='OLS (First Stage)')
    if res:
        # Get F-stat from the OLS model
        model_plain = smf.ols(formula, data=df).fit()
        print(f"  First stage: coef={res['coefficient']:.4f}, F={model_plain.fvalue:.1f}")
        results.append(res)

# 8.2 Reduced Form
print("\n8.2 Reduced Form: cOut ~ syntheticT + controls")
formula = "cOut ~ syntheticT + pr4 + biorder + hengpee"
model = run_ols_clustered(df, formula, CLUSTER_VAR)
if model:
    res = extract_results(model, TREATMENT_VAR, 'cOut', 'iv/first_stage/reduced_form',
                         'methods/instrumental_variables.md#reduced-form',
                         'pr4, biorder, hengpee', 'Baseline sample',
                         model_type='OLS (Reduced Form)')
    if res:
        results.append(res)
        print(f"  Reduced form: coef={res['coefficient']:.4f}")

# 8.3 2SLS for cOut
print("\n8.3 2SLS: cOut ~ mourn (instrumented by syntheticT)")
try:
    df_iv = df.dropna(subset=['cOut', 'mourn', 'syntheticT', 'pr4', 'biorder', 'hengpee']).copy()

    # First stage
    first_stage = smf.ols("mourn ~ syntheticT + pr4 + biorder + hengpee", data=df_iv).fit()
    df_iv['mourn_hat'] = first_stage.fittedvalues

    # Second stage
    second_stage = smf.ols("cOut ~ mourn_hat + pr4 + biorder + hengpee", data=df_iv).fit()

    # Get clustered SEs
    cluster_groups = df_iv[CLUSTER_VAR]
    second_stage_cluster = second_stage.get_robustcov_results(cov_type='cluster', groups=cluster_groups)

    # Extract results
    coef = second_stage_cluster.params['mourn_hat']
    se = second_stage_cluster.bse['mourn_hat']
    t_stat = second_stage_cluster.tvalues['mourn_hat']
    pval = second_stage_cluster.pvalues['mourn_hat']

    coef_dict = {
        "treatment": {"var": "mourn (IV)", "coef": float(coef), "se": float(se), "pval": float(pval)},
        "controls": [],
        "fixed_effects": [],
        "diagnostics": {"first_stage_F": float(first_stage.fvalue)}
    }

    res = {
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': 'iv/method/2sls',
        'spec_tree_path': 'methods/instrumental_variables.md#2sls',
        'outcome_var': 'cOut',
        'treatment_var': 'mourn (IV)',
        'coefficient': float(coef),
        'std_error': float(se),
        't_stat': float(t_stat),
        'p_value': float(pval),
        'ci_lower': float(coef - 1.96 * se),
        'ci_upper': float(coef + 1.96 * se),
        'n_obs': int(len(df_iv)),
        'r_squared': float(second_stage.rsquared),
        'coefficient_vector_json': json.dumps(coef_dict),
        'sample_desc': 'Baseline sample',
        'fixed_effects': 'None',
        'controls_desc': 'pr4, biorder, hengpee',
        'cluster_var': 'byear',
        'model_type': '2SLS',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    }
    results.append(res)
    print(f"  2SLS: coef={coef:.4f}, se={se:.4f}, F={first_stage.fvalue:.1f}")
except Exception as e:
    print(f"  Error in 2SLS: {e}")

# 8.4 2SLS for fmissmatch
print("\n8.4 2SLS: fmissmatch ~ mourn (instrumented by syntheticT)")
try:
    df_iv = df.dropna(subset=['fmissmatch', 'mourn', 'syntheticT', 'pr4', 'biorder', 'hengpee']).copy()

    # First stage
    first_stage = smf.ols("mourn ~ syntheticT + pr4 + biorder + hengpee", data=df_iv).fit()
    df_iv['mourn_hat'] = first_stage.fittedvalues

    # Second stage
    second_stage = smf.ols("fmissmatch ~ mourn_hat + pr4 + biorder + hengpee", data=df_iv).fit()

    cluster_groups = df_iv[CLUSTER_VAR]
    second_stage_cluster = second_stage.get_robustcov_results(cov_type='cluster', groups=cluster_groups)

    coef = second_stage_cluster.params['mourn_hat']
    se = second_stage_cluster.bse['mourn_hat']

    coef_dict = {
        "treatment": {"var": "mourn (IV)", "coef": float(coef), "se": float(se), "pval": float(second_stage_cluster.pvalues['mourn_hat'])},
        "controls": [],
        "fixed_effects": [],
        "diagnostics": {"first_stage_F": float(first_stage.fvalue)}
    }

    res = {
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': 'iv/method/2sls_fmissmatch',
        'spec_tree_path': 'methods/instrumental_variables.md#2sls',
        'outcome_var': 'fmissmatch',
        'treatment_var': 'mourn (IV)',
        'coefficient': float(coef),
        'std_error': float(se),
        't_stat': float(second_stage_cluster.tvalues['mourn_hat']),
        'p_value': float(second_stage_cluster.pvalues['mourn_hat']),
        'ci_lower': float(coef - 1.96 * se),
        'ci_upper': float(coef + 1.96 * se),
        'n_obs': int(len(df_iv)),
        'r_squared': float(second_stage.rsquared),
        'coefficient_vector_json': json.dumps(coef_dict),
        'sample_desc': 'Baseline sample, landholdings subsample',
        'fixed_effects': 'None',
        'controls_desc': 'pr4, biorder, hengpee',
        'cluster_var': 'byear',
        'model_type': '2SLS',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    }
    results.append(res)
    print(f"  2SLS fmissmatch: coef={coef:.4f}, se={se:.4f}")
except Exception as e:
    print(f"  Error in 2SLS fmissmatch: {e}")

# 8.5 OLS for comparison (ignoring endogeneity)
print("\n8.5 OLS (ignoring endogeneity): cOut ~ mourn + controls")
formula = "cOut ~ mourn + pr4 + biorder + hengpee"
model = run_ols_clustered(df, formula, CLUSTER_VAR)
if model:
    res = extract_results(model, 'mourn', 'cOut', 'iv/method/ols',
                         'methods/instrumental_variables.md#ols-comparison',
                         'pr4, biorder, hengpee', 'Baseline sample',
                         model_type='OLS (no IV)')
    if res:
        results.append(res)
        print(f"  OLS (no IV): coef={res['coefficient']:.4f}")

# ============================================================================
# 9. PLACEBO TESTS
# ============================================================================
print("\n" + "="*60)
print("PLACEBO TESTS")
print("="*60)

# 9.1 Fake treatment timing - use actual mourn as outcome, but shift definition
print("\n9.1 Placebo: Pre-treatment cohorts (born 1825-1830)")
df_early = df[df['byear'] <= 1830].copy()
formula = "cOut ~ syntheticT + pr4 + biorder + hengpee"
model = run_ols_clustered(df_early, formula, CLUSTER_VAR)
if model:
    res = extract_results(model, TREATMENT_VAR, 'cOut', 'robust/placebo/early_cohort',
                         'robustness/placebo_tests.md#timing',
                         'pr4, biorder, hengpee', 'Early cohorts (born 1825-1830)',
                         model_type='OLS')
    if res:
        results.append(res)
        print(f"  Early cohort: coef={res['coefficient']:.4f}, n={res['n_obs']}")

# 9.2 Placebo: Late cohorts
print("\n9.2 Placebo: Late cohorts (born 1841-1846)")
df_late = df[df['byear'] >= 1841].copy()
model = run_ols_clustered(df_late, formula, CLUSTER_VAR)
if model:
    res = extract_results(model, TREATMENT_VAR, 'cOut', 'robust/placebo/late_cohort',
                         'robustness/placebo_tests.md#timing',
                         'pr4, biorder, hengpee', 'Late cohorts (born 1841-1846)',
                         model_type='OLS')
    if res:
        results.append(res)
        print(f"  Late cohort: coef={res['coefficient']:.4f}, n={res['n_obs']}")

# 9.3 Placebo outcome: pr4 (predetermined at birth)
print("\n9.3 Placebo outcome: pr4 (predetermined)")
formula = "pr4 ~ syntheticT + biorder + hengpee"
model = run_ols_clustered(df, formula, CLUSTER_VAR)
if model:
    res = extract_results(model, TREATMENT_VAR, 'pr4', 'robust/placebo/predetermined_pr4',
                         'robustness/placebo_tests.md#predetermined',
                         'biorder, hengpee', 'Baseline sample',
                         model_type='OLS')
    if res:
        results.append(res)
        print(f"  Placebo (pr4): coef={res['coefficient']:.4f}, p={res['p_value']:.4f}")

# 9.4 Placebo outcome: birth order (predetermined)
print("\n9.4 Placebo outcome: biorder (predetermined)")
formula = "biorder ~ syntheticT + pr4 + hengpee"
model = run_ols_clustered(df, formula, CLUSTER_VAR)
if model:
    res = extract_results(model, TREATMENT_VAR, 'biorder', 'robust/placebo/predetermined_biorder',
                         'robustness/placebo_tests.md#predetermined',
                         'pr4, hengpee', 'Baseline sample',
                         model_type='OLS')
    if res:
        results.append(res)
        print(f"  Placebo (biorder): coef={res['coefficient']:.4f}, p={res['p_value']:.4f}")

# ============================================================================
# 10. ADDITIONAL ROBUSTNESS - Multiple Outcomes with Extended Controls
# ============================================================================
print("\n" + "="*60)
print("EXTENDED CONTROLS ON ALL OUTCOMES")
print("="*60)

for outcome in ['cOut', 'mheir', 'fmissmatch', 'fmissmatch2', 'fdown']:
    print(f"\n{outcome} with extended controls")
    formula = f"{outcome} ~ syntheticT + pr4 + biorder + hengpee + distlondon"
    model = run_ols_clustered(df, formula, CLUSTER_VAR)
    if model:
        res = extract_results(model, TREATMENT_VAR, outcome, f'robust/extended/{outcome}',
                             'robustness/control_progression.md#extended',
                             'pr4, biorder, hengpee, distlondon', 'Baseline sample',
                             model_type='OLS')
        if res:
            results.append(res)
            print(f"  coef={res['coefficient']:.4f}, p={res['p_value']:.4f}")

# ============================================================================
# 11. ADDITIONAL SAMPLE RESTRICTIONS - More Birth Years
# ============================================================================
print("\n" + "="*60)
print("ADDITIONAL SAMPLE RESTRICTIONS")
print("="*60)

# Drop more birth years
for year in byears[5:10]:  # Next 5 years
    df_sub = df[df['byear'] != year].copy()
    formula = "cOut ~ syntheticT + pr4 + biorder + hengpee"
    model = run_ols_clustered(df_sub, formula, CLUSTER_VAR)
    if model:
        res = extract_results(model, TREATMENT_VAR, 'cOut', f'robust/sample/drop_byear_{int(year)}',
                             'robustness/sample_restrictions.md#drop-year',
                             'pr4, biorder, hengpee', f'Drop birth year {int(year)}',
                             model_type='OLS')
        if res:
            results.append(res)
            print(f"  Drop byear {int(year)}: coef={res['coefficient']:.4f}, n={res['n_obs']}")

# ============================================================================
# 12. CELIBACY OUTCOME (Different Sample)
# ============================================================================
print("\n" + "="*60)
print("CELIBACY OUTCOME (MARITAL RATES SAMPLE)")
print("="*60)

# Baseline celibacy
print("\nCelibacy baseline")
formula = "celibacy ~ syntheticT + pr4 + biorder + hengpee"
model = run_ols_clustered(df_marital, formula, CLUSTER_VAR)
if model:
    res = extract_results(model, TREATMENT_VAR, 'celibacy', 'robust/outcome/celibacy',
                         'robustness/measurement.md#alternative-outcomes',
                         'pr4, biorder, hengpee', 'Marital rates sample',
                         model_type='OLS')
    if res:
        results.append(res)
        print(f"  Celibacy: coef={res['coefficient']:.4f}, p={res['p_value']:.4f}")

# Celibacy with extended controls
print("\nCelibacy with extended controls")
formula = "celibacy ~ syntheticT + pr4 + biorder + hengpee + distlondon"
model = run_ols_clustered(df_marital, formula, CLUSTER_VAR)
if model:
    res = extract_results(model, TREATMENT_VAR, 'celibacy', 'robust/outcome/celibacy_extended',
                         'robustness/measurement.md#alternative-outcomes',
                         'pr4, biorder, hengpee, distlondon', 'Marital rates sample',
                         model_type='OLS')
    if res:
        results.append(res)
        print(f"  Celibacy extended: coef={res['coefficient']:.4f}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Convert to DataFrame
results_df = pd.DataFrame(results)
print(f"\nTotal specifications run: {len(results_df)}")

# Save to CSV
output_path = f"{DATA_PATH}/../specification_results.csv"
results_df.to_csv(output_path, index=False)
print(f"Results saved to: {output_path}")

# Summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

# Filter to main treatment variable results
main_results = results_df[results_df['treatment_var'] == 'syntheticT']

print(f"\nTotal specs with syntheticT: {len(main_results)}")
print(f"Positive coefficients: {(main_results['coefficient'] > 0).sum()} ({100*(main_results['coefficient'] > 0).mean():.1f}%)")
print(f"Significant at 5%: {(main_results['p_value'] < 0.05).sum()} ({100*(main_results['p_value'] < 0.05).mean():.1f}%)")
print(f"Significant at 1%: {(main_results['p_value'] < 0.01).sum()} ({100*(main_results['p_value'] < 0.01).mean():.1f}%)")
print(f"\nCoefficient range: [{main_results['coefficient'].min():.4f}, {main_results['coefficient'].max():.4f}]")
print(f"Median coefficient: {main_results['coefficient'].median():.4f}")
print(f"Mean coefficient: {main_results['coefficient'].mean():.4f}")

# Breakdown by category
print("\n\nBreakdown by specification category:")

def categorize_spec(spec_id):
    if spec_id.startswith('baseline'):
        return 'baseline'
    elif 'control' in spec_id:
        return 'control'
    elif 'sample' in spec_id:
        return 'sample'
    elif 'outcome' in spec_id or spec_id.startswith('robust/extended'):
        return 'outcome'
    elif 'inference' in spec_id:
        return 'inference'
    elif 'funcform' in spec_id:
        return 'funcform'
    elif 'heterogeneity' in spec_id:
        return 'heterogeneity'
    elif 'placebo' in spec_id:
        return 'placebo'
    elif 'iv/' in spec_id:
        return 'iv'
    else:
        return 'other'

results_df['category'] = results_df['spec_id'].apply(categorize_spec)

for cat in ['baseline', 'control', 'sample', 'outcome', 'inference', 'funcform', 'heterogeneity', 'placebo', 'iv']:
    cat_df = results_df[results_df['category'] == cat]
    if len(cat_df) > 0:
        # For consistent sign counting, use syntheticT or mourn (IV) results
        pct_pos = 100 * (cat_df['coefficient'] > 0).mean()
        pct_sig = 100 * (cat_df['p_value'] < 0.05).mean()
        print(f"  {cat}: N={len(cat_df)}, {pct_pos:.0f}% positive, {pct_sig:.0f}% sig at 5%")

print("\nDone!")
