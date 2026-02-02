"""
Specification Search: 145141-V1
Paper: "Measuring the Welfare Effects of Shame and Pride" by Butera et al. (2021)
Published: AER

This paper studies image concerns (shame and pride) using two main experiments:
1. YMCA experiment: WTP for public recognition of gym attendance
2. Charity experiments (Prolific, Berkeley, BU): WTP for public recognition of charitable contributions

Main specifications:
- Regress WTP on performance intervals (linear and quadratic)
- Use clustered standard errors by individual
- Compare OLS and Tobit models
- Various sample restrictions (coherent, monotonic, close to beliefs)
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings('ignore')

# Paths
PACKAGE_DIR = Path("/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/145141-V1/Packet")
RAW_DIR = PACKAGE_DIR / "Raw Data"
OUTPUT_DIR = PACKAGE_DIR

# Paper metadata
PAPER_ID = "145141-V1"
JOURNAL = "AER"
PAPER_TITLE = "Measuring the Welfare Effects of Shame and Pride"

# Results storage
results = []

def add_result(spec_id, spec_tree_path, outcome_var, treatment_var,
               coef, se, t_stat, p_value, ci_lower, ci_upper,
               n_obs, r_squared, coef_vector_json, sample_desc,
               fixed_effects, controls_desc, cluster_var, model_type):
    """Add a result to the results list."""
    results.append({
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
        'n_obs': n_obs,
        'r_squared': r_squared,
        'coefficient_vector_json': json.dumps(coef_vector_json),
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })

def extract_results(model, treatment_var, model_type='OLS'):
    """Extract results from a statsmodels regression."""
    if treatment_var not in model.params.index:
        return None

    coef = model.params[treatment_var]
    se = model.bse[treatment_var]
    t_stat = model.tvalues[treatment_var]
    p_val = model.pvalues[treatment_var]
    ci = model.conf_int().loc[treatment_var]

    # Build coefficient vector JSON
    coef_vector = {
        'treatment': {
            'var': treatment_var,
            'coef': float(coef),
            'se': float(se),
            'pval': float(p_val)
        },
        'controls': [],
        'diagnostics': {
            'r_squared': float(model.rsquared) if hasattr(model, 'rsquared') else None,
            'n_obs': int(model.nobs)
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

    return {
        'coef': float(coef),
        'se': float(se),
        't_stat': float(t_stat),
        'p_value': float(p_val),
        'ci_lower': float(ci[0]),
        'ci_upper': float(ci[1]),
        'n_obs': int(model.nobs),
        'r_squared': float(model.rsquared) if hasattr(model, 'rsquared') else None,
        'coef_vector': coef_vector
    }

def run_clustered_ols(formula, data, cluster_var):
    """Run OLS with clustered standard errors."""
    try:
        model = smf.ols(formula, data=data).fit(cov_type='cluster',
                                                  cov_kwds={'groups': data[cluster_var]})
        return model
    except Exception as e:
        print(f"Error in OLS: {e}")
        return None

def run_robust_ols(formula, data):
    """Run OLS with robust (HC1) standard errors."""
    try:
        model = smf.ols(formula, data=data).fit(cov_type='HC1')
        return model
    except Exception as e:
        print(f"Error in robust OLS: {e}")
        return None

# ==============================================================================
# PART 1: Load and prepare the YMCA experiment data
# ==============================================================================
print("Loading YMCA experiment data...")

# Load treatment assignment
ymca_treatment = pd.read_csv(RAW_DIR / "ymca_treatment_assignment.csv")

# Load survey data
ymca_survey = pd.read_csv(RAW_DIR / "ymca_survey_data.csv")

# Merge treatment assignment
ymca_survey = ymca_survey.merge(ymca_treatment, on=['ymca_id', 'survey_id'], how='left')

# Load membership data
ymca_membership = pd.read_csv(RAW_DIR / "ymca_membership_data.csv")

# Load attendance data
ymca_scans = pd.read_csv(RAW_DIR / "ymca_scans.csv")

# Calculate past attendance (average monthly visits)
month_cols = [c for c in ymca_membership.columns if c.startswith('m20')]
ymca_membership['past'] = ymca_membership[month_cols].sum(axis=1) / len(month_cols) * 4  # Weekly average

# Merge survey with membership
ymca_df = ymca_survey.merge(ymca_membership[['ymca_id', 'past', 'female', 'age']],
                             on='ymca_id', how='left')

# Create WTP variables
wtp_intervals = [0, 1, 2, 3, 4, 56, 78, 912, 1317, 1822, 2328]
for x in wtp_intervals:
    col_image = f'times_{x}_1'
    col_no = f'times_{x}_no_1'
    col_yes = f'times_{x}_yes_1'

    if col_image in ymca_df.columns:
        ymca_df[f'image{x}'] = (ymca_df[col_image] == 2).astype(float)
        ymca_df[f'wtp{x}'] = np.where(
            ymca_df[f'image{x}'] == 0,
            -1 * ymca_df[col_no] if col_no in ymca_df.columns else np.nan,
            ymca_df[col_yes] if col_yes in ymca_df.columns else np.nan
        )

# Create coherent/monotonic sample indicators
ymca_df['switch'] = 0
ymca_df['switch_yes_no'] = 0
prev_image = ymca_df['image0'] if 'image0' in ymca_df.columns else None

for x in wtp_intervals[1:]:
    if f'image{x}' in ymca_df.columns and prev_image is not None:
        ymca_df['switch'] += (ymca_df[f'image{x}'] != prev_image).astype(int)
        ymca_df['switch_yes_no'] += (ymca_df[f'image{x}'] < prev_image).astype(int)
        prev_image = ymca_df[f'image{x}']

ymca_df['coherent'] = 1
ymca_df.loc[ymca_df['switch'] > 2, 'coherent'] = 0
if 'image0' in ymca_df.columns:
    ymca_df.loc[(ymca_df['image0'] == 1) & (ymca_df['switch'] == 2), 'coherent'] = 0

ymca_df['monotonic'] = 1
ymca_df.loc[ymca_df['switch_yes_no'] > 0, 'monotonic'] = 0

ymca_df['bdm'] = (ymca_df['treatment'] == 2).astype(int)
ymca_df['coherent_sample'] = ((ymca_df['coherent'] == 1) & (ymca_df['bdm'] == 0)).astype(int)
ymca_df['monotonic_sample'] = ((ymca_df['monotonic'] == 1) & (ymca_df['bdm'] == 0)).astype(int)

# Create beliefs variables
if 'q26_1' in ymca_df.columns:
    ymca_df['beliefs_w_image'] = ymca_df['q26_1']
if 'q27_1' in ymca_df.columns:
    ymca_df['beliefs_wout_image'] = ymca_df['q27_1']
if 'q30_1' in ymca_df.columns:
    ymca_df['beliefs_w_1'] = ymca_df['q30_1']
if 'q31_1' in ymca_df.columns:
    ymca_df['payment1'] = ymca_df['q31_1']

# Create motivation variable
if all(c in ymca_df.columns for c in ['payment1', 'beliefs_w_1', 'beliefs_wout_image']):
    ymca_df['motivation'] = ymca_df['payment1'] - 0.5 * (ymca_df['beliefs_w_1'] + ymca_df['beliefs_wout_image'])

# Reshape to long format for WTP analysis
print("Reshaping YMCA data to long format...")
ymca_long_data = []

# Map interval codes to visit numbers
interval_to_visits = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4,
    56: 5.5, 78: 7.5, 912: 10.5, 1317: 15, 1822: 20, 2328: 26.5
}

for idx, row in ymca_df.iterrows():
    for x in wtp_intervals:
        wtp_col = f'wtp{x}'
        if wtp_col in ymca_df.columns and pd.notna(row.get(wtp_col)):
            record = {
                'id': row['ymca_id'],
                'interval': x,
                'visits': interval_to_visits.get(x, x),
                'wtp': row[wtp_col],
                'coherent_sample': row.get('coherent_sample', 0),
                'monotonic_sample': row.get('monotonic_sample', 0),
                'past': row.get('past', np.nan),
                'beliefs_w_image': row.get('beliefs_w_image', np.nan),
                'motivation': row.get('motivation', np.nan),
                'female': row.get('female', np.nan),
                'age': row.get('age', np.nan)
            }
            ymca_long_data.append(record)

ymca_long = pd.DataFrame(ymca_long_data)
ymca_long['visits2'] = ymca_long['visits'] ** 2
ymca_long['closeb'] = np.abs(ymca_long['visits'] - ymca_long['beliefs_w_image'])
ymca_long['closep'] = np.abs(ymca_long['visits'] - ymca_long['past'])

print(f"YMCA long format: {len(ymca_long)} rows, {ymca_long['id'].nunique()} subjects")

# ==============================================================================
# PART 2: Load and prepare the Charity experiment data
# ==============================================================================
print("\nLoading Charity experiment data...")

# Load Prolific data
prolific = pd.read_csv(RAW_DIR / "prolific_survey_data.csv")
# Drop existing Sample column if it exists (from original data)
if 'Sample' in prolific.columns:
    prolific = prolific.drop(columns=['Sample'])
prolific['sample'] = 'Prolific'

# Load Berkeley data
berkeley = pd.read_csv(RAW_DIR / "berkeley_survey_data.csv")
if 'Sample' in berkeley.columns:
    berkeley = berkeley.drop(columns=['Sample'])
berkeley['sample'] = 'Berkeley'

# Load BU data
bu_qm221 = pd.read_csv(RAW_DIR / "bu_qm221_survey_data.csv")
if 'Sample' in bu_qm221.columns:
    bu_qm221 = bu_qm221.drop(columns=['Sample'])
bu_qm221['sample'] = 'BU'

bu_qm222 = pd.read_csv(RAW_DIR / "bu_qm222_survey_data.csv")
if 'Sample' in bu_qm222.columns:
    bu_qm222 = bu_qm222.drop(columns=['Sample'])
bu_qm222['sample'] = 'BU'

# Combine charity datasets
charity_df = pd.concat([prolific, berkeley, bu_qm221, bu_qm222], ignore_index=True, sort=False)

# Standardize column names to lowercase (sample already lowercase)
charity_df.columns = charity_df.columns.str.lower()

# Remove any duplicate columns
charity_df = charity_df.loc[:, ~charity_df.columns.duplicated()]

print(f"  Sample distribution: {charity_df['sample'].value_counts().to_dict()}")

# Rename key columns if they exist
rename_cols = {
    'earnpts': 'earnpts',
    'anompts': 'anompts',
    'recogpts': 'recogpts',
    'practicepts': 'practicepts'
}

# Generate consistency indicators
# Switch count
charity_df['switch_preference'] = 0
pref_cols = [f'preference_{i*100}' for i in range(18)]
existing_pref_cols = [c for c in pref_cols if c in charity_df.columns]

for i in range(1, len(existing_pref_cols)):
    col_curr = existing_pref_cols[i]
    col_prev = existing_pref_cols[i-1]
    charity_df['switch_preference'] += (charity_df[col_curr] != charity_df[col_prev]).astype(int)

# Consistent indicators
charity_df['consistent'] = ((charity_df['switch_preference'] == 0) |
                            ((charity_df['switch_preference'] == 1) & (charity_df.get('preference_0', 0) == 0)))
charity_df['consistent_b'] = charity_df['consistent'].copy()
charity_df.loc[(charity_df['switch_preference'] == 2) & (charity_df.get('preference_0', 0) == 0), 'consistent_b'] = True

# Attention check flag
if 'attention_check' in charity_df.columns:
    charity_df['flag_attention_check'] = charity_df['attention_check'].notna().astype(int)
else:
    charity_df['flag_attention_check'] = 0

# Generate WTP variables from slider data
# The original data has pref_anom_{points}_1 (slider) and pref_rec_{points}_1 (slider)
# Preference: 0 = anonymous, 1 = recognition
# WTP: negative of pref_anom (willing to pay to avoid recognition), or positive pref_rec (willing to pay for recognition)

for i in range(18):
    j = i * 100
    pref_col = f'preference_{j}'
    anon_col = f'pref_anom_{j}_1'
    rec_col = f'pref_rec_{j}_1'

    # Initialize WTP
    charity_df[f'wtp{i}'] = np.nan

    # If preference is 0 (anonymous), use negative of anonymous WTP
    if anon_col in charity_df.columns:
        mask_anon = charity_df[anon_col].notna()
        charity_df.loc[mask_anon, f'wtp{i}'] = -1 * charity_df.loc[mask_anon, anon_col]

    # If preference is 1 (recognition), use recognition WTP (overwrites where available)
    if rec_col in charity_df.columns:
        mask_rec = charity_df[rec_col].notna()
        charity_df.loc[mask_rec, f'wtp{i}'] = charity_df.loc[mask_rec, rec_col]

print(f"  WTP columns created: {sum([f'wtp{i}' in charity_df.columns for i in range(18)])}")

# Generate group size indicators for Prolific
if 'badgegroupsize' in charity_df.columns:
    charity_df['group15'] = (charity_df['badgegroupsize'] == 15).astype(int)
    charity_df['group75'] = (charity_df['badgegroupsize'] == 75).astype(int)
    charity_df['group300'] = (charity_df['badgegroupsize'] == 300).astype(int)

# Generate realized interval
if 'recogpts' in charity_df.columns:
    charity_df['realized_interval'] = np.floor(charity_df['recogpts'] / 100).clip(upper=17)

# Gender variable
if 'gender' in charity_df.columns:
    charity_df['female'] = (charity_df['gender'] == 2).astype(int)

print(f"Charity data: {len(charity_df)} rows")

# Reshape charity data to long format
print("Reshaping Charity data to long format...")
charity_long_data = []

for idx, row in charity_df.iterrows():
    for i in range(17):  # 0-16 intervals (excluding top)
        wtp_col = f'wtp{i}'
        if wtp_col in charity_df.columns:
            wtp_val = row.get(wtp_col)
            if pd.notna(wtp_val):
                record = {
                    'id': row.get('id', idx),
                    'interval': i + 0.5,  # Midpoint
                    'wtp': float(wtp_val),
                    'sample': str(row.get('sample', 'Unknown')),
                    'consistent_b': bool(row.get('consistent_b', True)),
                    'flag_attention_check': int(row.get('flag_attention_check', 0)),
                    'anompts': row.get('anompts', np.nan),
                    'recogpts': row.get('recogpts', np.nan),
                    'earnpts': row.get('earnpts', np.nan),
                    'female': row.get('female', np.nan),
                    'age': row.get('age', np.nan),
                    'group15': int(row.get('group15', 0)) if pd.notna(row.get('group15')) else 0,
                    'group300': int(row.get('group300', 0)) if pd.notna(row.get('group300')) else 0
                }
                charity_long_data.append(record)

charity_long = pd.DataFrame(charity_long_data)

if len(charity_long) > 0:
    charity_long['interval_sq'] = charity_long['interval'] ** 2

    # Generate above median anonymous score indicator
    for sample in charity_long['sample'].unique():
        mask = charity_long['sample'] == sample
        median_anom = charity_long.loc[mask, 'anompts'].median()
        charity_long.loc[mask, 'above_median_anom'] = (charity_long.loc[mask, 'anompts'] >= median_anom).astype(int)
else:
    print("WARNING: No charity long data generated")

print(f"Charity long format: {len(charity_long)} rows")

# ==============================================================================
# PART 3: Run specifications
# ==============================================================================
print("\n" + "="*70)
print("RUNNING SPECIFICATION SEARCH")
print("="*70)

spec_count = 0

# Helper function to safely run specification
def run_spec(spec_id, tree_path, formula, data, treatment_var, outcome_var,
             sample_desc, model_type, controls_desc, cluster_var='id',
             fixed_effects='None', use_robust=False):
    global spec_count

    try:
        data_clean = data.dropna(subset=[treatment_var, outcome_var])
        if len(data_clean) < 30:
            print(f"  Skipping {spec_id}: insufficient data ({len(data_clean)} obs)")
            return

        if use_robust:
            model = run_robust_ols(formula, data_clean)
        else:
            model = run_clustered_ols(formula, data_clean, cluster_var)

        if model is None:
            return

        res = extract_results(model, treatment_var, model_type)
        if res is None:
            return

        add_result(
            spec_id=spec_id,
            spec_tree_path=tree_path,
            outcome_var=outcome_var,
            treatment_var=treatment_var,
            coef=res['coef'],
            se=res['se'],
            t_stat=res['t_stat'],
            p_value=res['p_value'],
            ci_lower=res['ci_lower'],
            ci_upper=res['ci_upper'],
            n_obs=res['n_obs'],
            r_squared=res['r_squared'],
            coef_vector_json=res['coef_vector'],
            sample_desc=sample_desc,
            fixed_effects=fixed_effects,
            controls_desc=controls_desc,
            cluster_var=cluster_var,
            model_type=model_type
        )
        spec_count += 1
        print(f"  [{spec_count}] {spec_id}: coef={res['coef']:.4f}, p={res['p_value']:.4f}, n={res['n_obs']}")

    except Exception as e:
        print(f"  Error in {spec_id}: {e}")

# ==============================================================================
# YMCA EXPERIMENT SPECIFICATIONS
# ==============================================================================
print("\n--- YMCA EXPERIMENT ---")

# Filter to coherent sample
ymca_coh = ymca_long[ymca_long['coherent_sample'] == 1].copy()
ymca_mon = ymca_long[ymca_long['monotonic_sample'] == 1].copy()

# 1. BASELINE - Linear WTP on visits (coherent sample)
print("\n1. Baseline specifications...")
run_spec('baseline', 'methods/cross_sectional_ols.md#baseline',
         'wtp ~ visits', ymca_coh, 'visits', 'wtp',
         'YMCA coherent sample', 'OLS', 'None', 'id')

# 2. Quadratic specification
run_spec('ymca/form/quadratic', 'methods/cross_sectional_ols.md#functional-form',
         'wtp ~ visits + visits2', ymca_coh, 'visits', 'wtp',
         'YMCA coherent sample', 'OLS', 'visits squared', 'id')

# 3. Monotonic sample - linear
run_spec('ymca/sample/monotonic_linear', 'robustness/sample_restrictions.md',
         'wtp ~ visits', ymca_mon, 'visits', 'wtp',
         'YMCA monotonic sample', 'OLS', 'None', 'id')

# 4. Monotonic sample - quadratic
run_spec('ymca/sample/monotonic_quadratic', 'robustness/sample_restrictions.md',
         'wtp ~ visits + visits2', ymca_mon, 'visits', 'wtp',
         'YMCA monotonic sample', 'OLS', 'visits squared', 'id')

# 5-8. Sample restrictions: close to beliefs
print("\n2. Sample restrictions (close to beliefs)...")
for max_dist in [2, 4, 6, 8]:
    ymca_close = ymca_coh[ymca_coh['closeb'] <= max_dist].copy()
    run_spec(f'ymca/sample/close_belief_{max_dist}', 'robustness/sample_restrictions.md',
             'wtp ~ visits + visits2', ymca_close, 'visits', 'wtp',
             f'YMCA close to beliefs (<={max_dist})', 'OLS', 'visits squared', 'id')

# 9-12. Sample restrictions: close to past attendance
print("\n3. Sample restrictions (close to past)...")
for max_dist in [2, 4, 6, 8]:
    ymca_close = ymca_coh[ymca_coh['closep'] <= max_dist].copy()
    run_spec(f'ymca/sample/close_past_{max_dist}', 'robustness/sample_restrictions.md',
             'wtp ~ visits + visits2', ymca_close, 'visits', 'wtp',
             f'YMCA close to past (<={max_dist})', 'OLS', 'visits squared', 'id')

# 13-16. Heterogeneity by past attendance
print("\n4. Heterogeneity by past attendance...")
past_median = ymca_coh['past'].median()
ymca_high_past = ymca_coh[ymca_coh['past'] >= past_median].copy()
ymca_low_past = ymca_coh[ymca_coh['past'] < past_median].copy()

run_spec('ymca/heterogeneity/high_past_linear', 'robustness/heterogeneity.md',
         'wtp ~ visits', ymca_high_past, 'visits', 'wtp',
         'YMCA high past attendance', 'OLS', 'None', 'id')

run_spec('ymca/heterogeneity/high_past_quadratic', 'robustness/heterogeneity.md',
         'wtp ~ visits + visits2', ymca_high_past, 'visits', 'wtp',
         'YMCA high past attendance', 'OLS', 'visits squared', 'id')

run_spec('ymca/heterogeneity/low_past_linear', 'robustness/heterogeneity.md',
         'wtp ~ visits', ymca_low_past, 'visits', 'wtp',
         'YMCA low past attendance', 'OLS', 'None', 'id')

run_spec('ymca/heterogeneity/low_past_quadratic', 'robustness/heterogeneity.md',
         'wtp ~ visits + visits2', ymca_low_past, 'visits', 'wtp',
         'YMCA low past attendance', 'OLS', 'visits squared', 'id')

# 17-18. Interaction with past attendance
print("\n5. Interaction specifications...")
ymca_coh['visits_x_past'] = ymca_coh['visits'] * ymca_coh['past']
ymca_coh['visits2_x_past'] = ymca_coh['visits2'] * ymca_coh['past']

run_spec('ymca/interact/past_linear', 'methods/cross_sectional_ols.md#interaction-effects',
         'wtp ~ visits + past + visits_x_past', ymca_coh, 'visits', 'wtp',
         'YMCA coherent sample', 'OLS', 'past, visits x past', 'id')

run_spec('ymca/interact/past_quadratic', 'methods/cross_sectional_ols.md#interaction-effects',
         'wtp ~ visits + visits2 + past + visits_x_past + visits2_x_past', ymca_coh, 'visits', 'wtp',
         'YMCA coherent sample', 'OLS', 'past, interactions', 'id')

# 19-20. Robust SE instead of clustered
print("\n6. Inference variations...")
run_spec('ymca/se/robust_linear', 'robustness/clustering_variations.md',
         'wtp ~ visits', ymca_coh, 'visits', 'wtp',
         'YMCA coherent sample', 'OLS', 'None', 'id', use_robust=True)

run_spec('ymca/se/robust_quadratic', 'robustness/clustering_variations.md',
         'wtp ~ visits + visits2', ymca_coh, 'visits', 'wtp',
         'YMCA coherent sample', 'OLS', 'visits squared', 'id', use_robust=True)

# 21-24. Functional form: interval index instead of visits
print("\n7. Functional form variations...")
ymca_coh['interval_idx'] = ymca_coh['visits'].rank(method='dense')
ymca_coh['interval_idx2'] = ymca_coh['interval_idx'] ** 2

run_spec('ymca/form/interval_idx_linear', 'robustness/functional_form.md',
         'wtp ~ interval_idx', ymca_coh, 'interval_idx', 'wtp',
         'YMCA coherent sample', 'OLS', 'None', 'id')

run_spec('ymca/form/interval_idx_quadratic', 'robustness/functional_form.md',
         'wtp ~ interval_idx + interval_idx2', ymca_coh, 'interval_idx', 'wtp',
         'YMCA coherent sample', 'OLS', 'interval squared', 'id')

# 25-26. Log visits
ymca_coh['ln_visits'] = np.log(ymca_coh['visits'] + 1)
ymca_coh['ln_visits2'] = ymca_coh['ln_visits'] ** 2

run_spec('ymca/form/ln_visits_linear', 'robustness/functional_form.md',
         'wtp ~ ln_visits', ymca_coh, 'ln_visits', 'wtp',
         'YMCA coherent sample', 'OLS', 'None', 'id')

run_spec('ymca/form/ln_visits_quadratic', 'robustness/functional_form.md',
         'wtp ~ ln_visits + ln_visits2', ymca_coh, 'ln_visits', 'wtp',
         'YMCA coherent sample', 'OLS', 'ln visits squared', 'id')

# 27-30. Exclude top intervals
print("\n8. Excluding top intervals...")
for max_visits in [20, 15, 10, 7.5]:
    ymca_excl = ymca_coh[ymca_coh['visits'] <= max_visits].copy()
    run_spec(f'ymca/sample/excl_above_{int(max_visits)}', 'robustness/sample_restrictions.md',
             'wtp ~ visits + visits2', ymca_excl, 'visits', 'wtp',
             f'YMCA excl visits > {max_visits}', 'OLS', 'visits squared', 'id')

# ==============================================================================
# CHARITY EXPERIMENT SPECIFICATIONS
# ==============================================================================
print("\n--- CHARITY EXPERIMENT ---")

# Filter to consistent sample without attention check failures
charity_clean = charity_long[
    (charity_long['consistent_b'] == True) &
    (charity_long['flag_attention_check'] == 0)
].copy()

# 31-32. Baseline across all samples
print("\n9. Charity baseline specifications...")
run_spec('charity/baseline/linear', 'methods/cross_sectional_ols.md#baseline',
         'wtp ~ interval', charity_clean, 'interval', 'wtp',
         'All charity samples', 'OLS', 'None', 'id')

run_spec('charity/baseline/quadratic', 'methods/cross_sectional_ols.md#baseline',
         'wtp ~ interval + interval_sq', charity_clean, 'interval', 'wtp',
         'All charity samples', 'OLS', 'interval squared', 'id')

# 33-38. By sample
print("\n10. By sample specifications...")
for sample in ['Prolific', 'Berkeley', 'BU']:
    sample_data = charity_clean[charity_clean['sample'] == sample].copy()

    run_spec(f'charity/sample/{sample.lower()}_linear', 'robustness/sample_restrictions.md',
             'wtp ~ interval', sample_data, 'interval', 'wtp',
             f'{sample} sample', 'OLS', 'None', 'id')

    run_spec(f'charity/sample/{sample.lower()}_quadratic', 'robustness/sample_restrictions.md',
             'wtp ~ interval + interval_sq', sample_data, 'interval', 'wtp',
             f'{sample} sample', 'OLS', 'interval squared', 'id')

# 39-44. Heterogeneity by above/below median anonymous score
print("\n11. Heterogeneity by anonymous score...")
for sample in ['Prolific', 'Berkeley', 'BU']:
    sample_data = charity_clean[charity_clean['sample'] == sample].copy()

    above_med = sample_data[sample_data['above_median_anom'] == 1].copy()
    below_med = sample_data[sample_data['above_median_anom'] == 0].copy()

    run_spec(f'charity/heterogeneity/{sample.lower()}_above_med', 'robustness/heterogeneity.md',
             'wtp ~ interval + interval_sq', above_med, 'interval', 'wtp',
             f'{sample} above median anon', 'OLS', 'interval squared', 'id')

    run_spec(f'charity/heterogeneity/{sample.lower()}_below_med', 'robustness/heterogeneity.md',
             'wtp ~ interval + interval_sq', below_med, 'interval', 'wtp',
             f'{sample} below median anon', 'OLS', 'interval squared', 'id')

# 45-46. Prolific with group size controls
print("\n12. Prolific with group size controls...")
prolific_data = charity_clean[charity_clean['sample'] == 'Prolific'].copy()

run_spec('charity/control/prolific_groupsize_linear', 'robustness/control_progression.md',
         'wtp ~ interval + group15 + group300', prolific_data, 'interval', 'wtp',
         'Prolific with group size', 'OLS', 'group size dummies', 'id')

run_spec('charity/control/prolific_groupsize_quadratic', 'robustness/control_progression.md',
         'wtp ~ interval + interval_sq + group15 + group300', prolific_data, 'interval', 'wtp',
         'Prolific with group size', 'OLS', 'interval sq, group size', 'id')

# 47-50. Interaction with group size
print("\n13. Group size interactions...")
prolific_data['interval_x_group15'] = prolific_data['interval'] * prolific_data['group15']
prolific_data['interval_x_group300'] = prolific_data['interval'] * prolific_data['group300']

run_spec('charity/interact/prolific_group15', 'methods/cross_sectional_ols.md#interaction-effects',
         'wtp ~ interval + group15 + interval_x_group15', prolific_data, 'interval', 'wtp',
         'Prolific', 'OLS', 'group15 interaction', 'id')

run_spec('charity/interact/prolific_group300', 'methods/cross_sectional_ols.md#interaction-effects',
         'wtp ~ interval + group300 + interval_x_group300', prolific_data, 'interval', 'wtp',
         'Prolific', 'OLS', 'group300 interaction', 'id')

run_spec('charity/interact/prolific_all_groups', 'methods/cross_sectional_ols.md#interaction-effects',
         'wtp ~ interval + group15 + group300 + interval_x_group15 + interval_x_group300',
         prolific_data, 'interval', 'wtp',
         'Prolific', 'OLS', 'all group interactions', 'id')

# 51-54. Robust SE variations
print("\n14. Inference variations...")
for sample in ['Prolific', 'Berkeley']:
    sample_data = charity_clean[charity_clean['sample'] == sample].copy()

    run_spec(f'charity/se/{sample.lower()}_robust_linear', 'robustness/clustering_variations.md',
             'wtp ~ interval', sample_data, 'interval', 'wtp',
             f'{sample} robust SE', 'OLS', 'None', 'id', use_robust=True)

    run_spec(f'charity/se/{sample.lower()}_robust_quadratic', 'robustness/clustering_variations.md',
             'wtp ~ interval + interval_sq', sample_data, 'interval', 'wtp',
             f'{sample} robust SE', 'OLS', 'interval squared', 'id', use_robust=True)

# 55-58. Functional form: polynomial
print("\n15. Higher order polynomials...")
charity_clean['interval_cubed'] = charity_clean['interval'] ** 3

for sample in ['Prolific', 'Berkeley']:
    sample_data = charity_clean[charity_clean['sample'] == sample].copy()
    sample_data['interval_cubed'] = sample_data['interval'] ** 3

    run_spec(f'charity/form/{sample.lower()}_cubic', 'robustness/functional_form.md',
             'wtp ~ interval + interval_sq + interval_cubed', sample_data, 'interval', 'wtp',
             f'{sample} cubic', 'OLS', 'interval sq and cubed', 'id')

# 59-62. Winsorization
print("\n16. Winsorization...")
for sample in ['Prolific', 'Berkeley']:
    sample_data = charity_clean[charity_clean['sample'] == sample].copy()

    # Winsorize WTP at 5% and 95%
    p5, p95 = sample_data['wtp'].quantile([0.05, 0.95])
    sample_data['wtp_wins'] = sample_data['wtp'].clip(lower=p5, upper=p95)

    run_spec(f'charity/sample/{sample.lower()}_winsorized', 'robustness/sample_restrictions.md',
             'wtp_wins ~ interval + interval_sq', sample_data, 'interval', 'wtp_wins',
             f'{sample} winsorized 5%', 'OLS', 'interval squared', 'id')

# 63-66. Trimming outliers
print("\n17. Trimming outliers...")
for sample in ['Prolific', 'Berkeley']:
    sample_data = charity_clean[charity_clean['sample'] == sample].copy()

    # Trim WTP at 1% and 99%
    p1, p99 = sample_data['wtp'].quantile([0.01, 0.99])
    sample_trimmed = sample_data[(sample_data['wtp'] >= p1) & (sample_data['wtp'] <= p99)].copy()

    run_spec(f'charity/sample/{sample.lower()}_trimmed', 'robustness/sample_restrictions.md',
             'wtp ~ interval + interval_sq', sample_trimmed, 'interval', 'wtp',
             f'{sample} trimmed 1%', 'OLS', 'interval squared', 'id')

# 67-70. Gender heterogeneity
print("\n18. Gender heterogeneity...")
for sample in ['Prolific', 'Berkeley']:
    sample_data = charity_clean[charity_clean['sample'] == sample].copy()
    sample_data = sample_data.dropna(subset=['female'])

    female_data = sample_data[sample_data['female'] == 1].copy()
    male_data = sample_data[sample_data['female'] == 0].copy()

    run_spec(f'charity/heterogeneity/{sample.lower()}_female', 'robustness/heterogeneity.md',
             'wtp ~ interval + interval_sq', female_data, 'interval', 'wtp',
             f'{sample} female', 'OLS', 'interval squared', 'id')

    run_spec(f'charity/heterogeneity/{sample.lower()}_male', 'robustness/heterogeneity.md',
             'wtp ~ interval + interval_sq', male_data, 'interval', 'wtp',
             f'{sample} male', 'OLS', 'interval squared', 'id')

# 71-74. Age heterogeneity for Prolific (where we have age)
print("\n19. Age heterogeneity...")
prolific_age = charity_clean[
    (charity_clean['sample'] == 'Prolific') &
    charity_clean['age'].notna()
].copy()

if len(prolific_age) > 100:
    age_median = prolific_age['age'].median()
    young_data = prolific_age[prolific_age['age'] < age_median].copy()
    old_data = prolific_age[prolific_age['age'] >= age_median].copy()

    run_spec('charity/heterogeneity/prolific_young', 'robustness/heterogeneity.md',
             'wtp ~ interval + interval_sq', young_data, 'interval', 'wtp',
             'Prolific young', 'OLS', 'interval squared', 'id')

    run_spec('charity/heterogeneity/prolific_old', 'robustness/heterogeneity.md',
             'wtp ~ interval + interval_sq', old_data, 'interval', 'wtp',
             'Prolific old', 'OLS', 'interval squared', 'id')

# 75-80. Placebo-like tests: using points from different rounds
print("\n20. Placebo-like tests...")
for sample in ['Prolific', 'Berkeley', 'BU']:
    sample_data = charity_clean[charity_clean['sample'] == sample].copy()

    # Use anonymous round points as predictor
    if 'anompts' in sample_data.columns and sample_data['anompts'].notna().sum() > 100:
        sample_data['anom_interval'] = np.floor(sample_data['anompts'] / 100).clip(upper=17) + 0.5
        sample_data['anom_interval_sq'] = sample_data['anom_interval'] ** 2

        run_spec(f'charity/placebo/{sample.lower()}_anon_pts', 'robustness/placebo_tests.md',
                 'wtp ~ anom_interval + anom_interval_sq', sample_data, 'anom_interval', 'wtp',
                 f'{sample} using anon pts', 'OLS', 'anom interval squared', 'id')

# 81-84. Controls progression - adding demographics
print("\n21. Control progression...")
prolific_controls = charity_clean[
    (charity_clean['sample'] == 'Prolific') &
    charity_clean['age'].notna() &
    charity_clean['female'].notna()
].copy()

if len(prolific_controls) > 100:
    run_spec('charity/control/prolific_no_controls', 'robustness/control_progression.md',
             'wtp ~ interval + interval_sq', prolific_controls, 'interval', 'wtp',
             'Prolific no controls', 'OLS', 'interval squared', 'id')

    run_spec('charity/control/prolific_age', 'robustness/control_progression.md',
             'wtp ~ interval + interval_sq + age', prolific_controls, 'interval', 'wtp',
             'Prolific with age', 'OLS', 'interval sq, age', 'id')

    run_spec('charity/control/prolific_female', 'robustness/control_progression.md',
             'wtp ~ interval + interval_sq + female', prolific_controls, 'interval', 'wtp',
             'Prolific with female', 'OLS', 'interval sq, female', 'id')

    run_spec('charity/control/prolific_all_demo', 'robustness/control_progression.md',
             'wtp ~ interval + interval_sq + age + female', prolific_controls, 'interval', 'wtp',
             'Prolific all demographics', 'OLS', 'interval sq, age, female', 'id')

# 85-88. Excluding certain intervals
print("\n22. Excluding intervals...")
for sample in ['Prolific', 'Berkeley']:
    sample_data = charity_clean[charity_clean['sample'] == sample].copy()

    # Exclude bottom interval
    sample_excl_bottom = sample_data[sample_data['interval'] > 1].copy()
    run_spec(f'charity/sample/{sample.lower()}_excl_bottom', 'robustness/sample_restrictions.md',
             'wtp ~ interval + interval_sq', sample_excl_bottom, 'interval', 'wtp',
             f'{sample} excl bottom interval', 'OLS', 'interval squared', 'id')

    # Exclude top interval
    sample_excl_top = sample_data[sample_data['interval'] < 16].copy()
    run_spec(f'charity/sample/{sample.lower()}_excl_top', 'robustness/sample_restrictions.md',
             'wtp ~ interval + interval_sq', sample_excl_top, 'interval', 'wtp',
             f'{sample} excl top interval', 'OLS', 'interval squared', 'id')

# ==============================================================================
# Save Results
# ==============================================================================
print("\n" + "="*70)
print(f"COMPLETED: {spec_count} specifications")
print("="*70)

# Create DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
output_path = OUTPUT_DIR / "specification_results.csv"
results_df.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")

# Print summary statistics
print("\n--- SUMMARY STATISTICS ---")
print(f"Total specifications: {len(results_df)}")
print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")
print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")
print(f"Coefficient range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")

print("\nDone!")
