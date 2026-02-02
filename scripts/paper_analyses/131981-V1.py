#!/usr/bin/env python3
"""
Specification Search for Paper 131981-V1
"Mental Health Costs of Lockdowns: Evidence from Age-specific Curfews in Turkey"

Method: Regression Discontinuity (Sharp RD)
Running variable: Birth month relative to December 1955 (dif)
Treatment: before1955 (1 if born before December 1955)
Main outcomes: Mental distress indices (z_depression, z_somatic, z_nonsomatic, sum_srq)

Following the i4r methodology with 50+ specifications.
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import warnings
from scipy import stats
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================================
# SETUP
# ============================================================================

BASE_DIR = Path('/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search')
DATA_DIR = BASE_DIR / 'data/downloads/extracted/131981-V1'
OUTPUT_DIR = DATA_DIR

PAPER_ID = '131981-V1'
PAPER_TITLE = 'Mental Health Costs of Lockdowns: Evidence from Age-specific Curfews in Turkey'
JOURNAL = 'American Economic Journal: Applied Economics'

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

print("Loading data...")
df = pd.read_stata(DATA_DIR / 'konda_data_for_analysis.dta')

# Filter for non-response == 0 (as in Stata code)
df = df[df['non_response'] == 0].copy()

print(f"Initial sample size: {len(df)}")

# Clean up categorical variables - replace spaces with underscores
def clean_string(s):
    """Clean string for use as variable name."""
    if pd.isna(s):
        return 'missing'
    return str(s).replace(' ', '_').replace('-', '_').lower()

# Create cleaned education categories
df['education_clean'] = df['education'].apply(clean_string)
df['ethnicity_clean'] = df['ethnicity'].apply(clean_string)

# Create dummies with clean names
education_dummies = pd.get_dummies(df['education_clean'], prefix='edu', drop_first=True)
ethnicity_dummies = pd.get_dummies(df['ethnicity_clean'], prefix='eth', drop_first=True)

# Merge dummies back
df = pd.concat([df, education_dummies, ethnicity_dummies], axis=1)

# Get lists of dummy variable names (now clean)
edu_cols = [c for c in df.columns if c.startswith('edu_')]
eth_cols = [c for c in df.columns if c.startswith('eth_')]

print(f"Education dummies: {edu_cols}")
print(f"Ethnicity dummies: {eth_cols}")

# SRQ-20 symptom variables (for constructing indices)
list1 = ['head_ache', 'mal_appetite', 'sleeplessness', 'scared', 'shaking', 'nervous',
         'indigestion', 'unfocused', 'unhappy', 'weepy', 'unwillingness', 'undecisiveness',
         'disrupted', 'useless', 'uninterest', 'worthless', 'suicidal', 'usually_tired',
         'stomach_discomfort', 'quickly_tired']
list2 = ['head_ache', 'shaking', 'indigestion', 'stomach_discomfort']  # somatic
list3 = [x for x in list1 if x not in list2]  # nonsomatic

# Create outcome indices
# Calculate sum SRQ-20
available_symptoms = [s for s in list1 if s in df.columns]
df['sum_srq'] = df[available_symptoms].sum(axis=1)
df.loc[df[available_symptoms].isna().any(axis=1), 'sum_srq'] = np.nan

# Create standardized indices for treatment effect analysis
# Using z-scores normalized by control group
control_mask = df['before1955'] == 0

def create_index(df, vars_list, control_mask):
    """Create a standardized index from a list of variables, normalized by control group."""
    avail_vars = [v for v in vars_list if v in df.columns]
    if not avail_vars:
        return pd.Series(np.nan, index=df.index)
    temp = df[avail_vars].copy()
    idx = temp.mean(axis=1)
    ctrl_mean = idx[control_mask].mean()
    ctrl_std = idx[control_mask].std()
    if ctrl_std > 0:
        idx_std = (idx - ctrl_mean) / ctrl_std
    else:
        idx_std = idx - ctrl_mean
    return idx_std

df['z_depression'] = create_index(df, list1, control_mask)
df['z_somatic'] = create_index(df, list2, control_mask)
df['z_nonsomatic'] = create_index(df, list3, control_mask)

# Create running variable interaction terms for RD (linear control function)
df['dif_neg'] = df['dif'] * (df['dif'] < 0).astype(int)
df['dif_pos'] = df['dif'] * (df['dif'] >= 0).astype(int)

# For quadratic control function
df['dif2_neg'] = (df['dif']**2) * (df['dif'] < 0).astype(int)
df['dif2_pos'] = (df['dif']**2) * (df['dif'] >= 0).astype(int)

# Create modate for clustering
if 'modate' not in df.columns:
    if 'month' in df.columns and 'year' in df.columns:
        df['modate'] = df['year'].astype(str) + '_' + df['month'].astype(str)
    else:
        df['modate'] = df.index

# ============================================================================
# DEFINE CONTROL VARIABLES
# ============================================================================

# Baseline controls
control_vars_basic = ['female'] + edu_cols + eth_cols

def get_control_string(controls):
    """Convert list of controls to formula string."""
    if not controls:
        return ''
    return ' + '.join(controls)

# ============================================================================
# RESULTS STORAGE
# ============================================================================

results = []

def extract_results(model, spec_id, spec_tree_path, outcome_var, treatment_var,
                    sample_desc, controls_desc, cluster_var, model_type,
                    bandwidth=None, additional_info=None):
    """Extract results from a pyfixest model into standard format."""
    try:
        coef = model.coef()[treatment_var]
        se = model.se()[treatment_var]
        tstat = model.tstat()[treatment_var]
        pval = model.pvalue()[treatment_var]

        # Get number of observations
        try:
            n_obs = model._N
        except:
            try:
                n_obs = len(model._Y)
            except:
                n_obs = None

        # Calculate CI
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se

        # Get R-squared if available
        try:
            r2 = model._r2
        except:
            r2 = np.nan

        # Create coefficient vector JSON
        coef_vector = {
            'treatment': {
                'var': treatment_var,
                'coef': float(coef),
                'se': float(se),
                'pval': float(pval)
            },
            'controls': [],
            'diagnostics': {}
        }

        if additional_info:
            coef_vector['diagnostics'].update(additional_info)

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
            'n_obs': int(n_obs) if n_obs else None,
            'r_squared': float(r2) if not np.isnan(r2) else None,
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc,
            'fixed_effects': 'None (controls in formula)',
            'controls_desc': controls_desc,
            'cluster_var': cluster_var,
            'model_type': model_type,
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }

        if bandwidth:
            result['bandwidth'] = bandwidth

        return result
    except Exception as e:
        print(f"    Error extracting results for {spec_id}: {e}")
        return None

# ============================================================================
# RUN SPECIFICATIONS
# ============================================================================

print("\n" + "="*80)
print("RUNNING SPECIFICATION SEARCH")
print("="*80)

spec_count = 0

# Main outcome variable for primary analysis
MAIN_OUTCOME = 'z_depression'
TREATMENT_VAR = 'before1955'

# Bandwidths to test
BANDWIDTHS = [17, 24, 30, 36, 45, 48, 60, 72]
BASELINE_BW = 45

# Get control formula
control_formula = get_control_string(control_vars_basic)

# ============================================================================
# 1. BASELINE SPECIFICATIONS
# ============================================================================
print("\n--- 1. Baseline Specifications ---")

df_baseline = df[df['dif'].between(-BASELINE_BW, BASELINE_BW)].copy()
formula = f'{MAIN_OUTCOME} ~ {TREATMENT_VAR} + dif_neg + dif_pos + {control_formula}'

try:
    model = pf.feols(formula, data=df_baseline, vcov={'CRV1': 'modate'})
    result = extract_results(
        model, 'baseline', 'methods/regression_discontinuity.md#baseline',
        MAIN_OUTCOME, TREATMENT_VAR,
        f'Bandwidth={BASELINE_BW} months', 'ethnicity, education, female',
        'modate', 'RD-Linear', BASELINE_BW,
        {'bandwidth': BASELINE_BW, 'polynomial_order': 1}
    )
    if result:
        results.append(result)
        spec_count += 1
        print(f"  Baseline: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")
except Exception as e:
    print(f"  Error in baseline: {e}")

# ============================================================================
# 2. BANDWIDTH VARIATIONS (RD-specific)
# ============================================================================
print("\n--- 2. Bandwidth Variations ---")

for bw in BANDWIDTHS:
    try:
        df_bw = df[df['dif'].between(-bw, bw)].copy()
        formula = f'{MAIN_OUTCOME} ~ {TREATMENT_VAR} + dif_neg + dif_pos + {control_formula}'
        model = pf.feols(formula, data=df_bw, vcov={'CRV1': 'modate'})

        result = extract_results(
            model, f'rd/bandwidth/bw_{bw}', 'methods/regression_discontinuity.md#bandwidth-selection',
            MAIN_OUTCOME, TREATMENT_VAR,
            f'Bandwidth={bw} months', 'ethnicity, education, female',
            'modate', 'RD-Linear', bw,
            {'bandwidth': bw, 'polynomial_order': 1}
        )
        if result:
            results.append(result)
            spec_count += 1
            print(f"  BW={bw}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")
    except Exception as e:
        print(f"  Error for BW={bw}: {e}")

# ============================================================================
# 3. POLYNOMIAL ORDER VARIATIONS
# ============================================================================
print("\n--- 3. Polynomial Order Variations ---")

for poly_order in [1, 2]:
    for bw in [30, 45, 60]:
        try:
            df_bw = df[df['dif'].between(-bw, bw)].copy()

            if poly_order == 1:
                formula = f'{MAIN_OUTCOME} ~ {TREATMENT_VAR} + dif_neg + dif_pos + {control_formula}'
                poly_name = 'linear'
            else:
                formula = f'{MAIN_OUTCOME} ~ {TREATMENT_VAR} + dif_neg + dif_pos + dif2_neg + dif2_pos + {control_formula}'
                poly_name = 'quadratic'

            model = pf.feols(formula, data=df_bw, vcov={'CRV1': 'modate'})

            result = extract_results(
                model, f'rd/poly/{poly_name}_bw{bw}', 'methods/regression_discontinuity.md#polynomial-order',
                MAIN_OUTCOME, TREATMENT_VAR,
                f'Bandwidth={bw}, polynomial={poly_order}', 'ethnicity, education, female',
                'modate', f'RD-{poly_name.capitalize()}', bw,
                {'bandwidth': bw, 'polynomial_order': poly_order}
            )
            if result:
                results.append(result)
                spec_count += 1
                print(f"  Poly={poly_order}, BW={bw}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
        except Exception as e:
            print(f"  Error for poly={poly_order}, BW={bw}: {e}")

# ============================================================================
# 4. ALTERNATIVE OUTCOMES
# ============================================================================
print("\n--- 4. Alternative Outcomes ---")

outcome_vars = ['z_depression', 'z_somatic', 'z_nonsomatic', 'sum_srq',
                'outside_week', 'under_curfew', 'never_out']

for outcome in outcome_vars:
    if outcome not in df.columns:
        continue
    try:
        df_bw = df[df['dif'].between(-BASELINE_BW, BASELINE_BW)].copy()
        df_bw = df_bw.dropna(subset=[outcome])

        formula = f'{outcome} ~ {TREATMENT_VAR} + dif_neg + dif_pos + {control_formula}'
        model = pf.feols(formula, data=df_bw, vcov={'CRV1': 'modate'})

        result = extract_results(
            model, f'robust/outcome/{outcome}', 'robustness/measurement.md',
            outcome, TREATMENT_VAR,
            f'Outcome={outcome}, BW={BASELINE_BW}', 'ethnicity, education, female',
            'modate', 'RD-Linear', BASELINE_BW
        )
        if result:
            results.append(result)
            spec_count += 1
            print(f"  Outcome={outcome}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
    except Exception as e:
        print(f"  Error for outcome={outcome}: {e}")

# ============================================================================
# 5. CONTROL VARIATIONS
# ============================================================================
print("\n--- 5. Control Variations ---")

# No controls
try:
    df_bw = df[df['dif'].between(-BASELINE_BW, BASELINE_BW)].copy()
    formula = f'{MAIN_OUTCOME} ~ {TREATMENT_VAR} + dif_neg + dif_pos'
    model = pf.feols(formula, data=df_bw, vcov={'CRV1': 'modate'})

    result = extract_results(
        model, 'rd/controls/none', 'methods/regression_discontinuity.md#control-sets',
        MAIN_OUTCOME, TREATMENT_VAR,
        f'No controls, BW={BASELINE_BW}', 'None',
        'modate', 'RD-Linear', BASELINE_BW
    )
    if result:
        results.append(result)
        spec_count += 1
        print(f"  No controls: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error for no controls: {e}")

# Only female
try:
    df_bw = df[df['dif'].between(-BASELINE_BW, BASELINE_BW)].copy()
    formula = f'{MAIN_OUTCOME} ~ {TREATMENT_VAR} + dif_neg + dif_pos + female'
    model = pf.feols(formula, data=df_bw, vcov={'CRV1': 'modate'})

    result = extract_results(
        model, 'rd/controls/female_only', 'methods/regression_discontinuity.md#control-sets',
        MAIN_OUTCOME, TREATMENT_VAR,
        f'Female only, BW={BASELINE_BW}', 'female',
        'modate', 'RD-Linear', BASELINE_BW
    )
    if result:
        results.append(result)
        spec_count += 1
        print(f"  Female only: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error for female only: {e}")

# Leave-one-out: drop ethnicity
try:
    df_bw = df[df['dif'].between(-BASELINE_BW, BASELINE_BW)].copy()
    controls_no_eth = ['female'] + edu_cols
    formula = f'{MAIN_OUTCOME} ~ {TREATMENT_VAR} + dif_neg + dif_pos + {get_control_string(controls_no_eth)}'
    model = pf.feols(formula, data=df_bw, vcov={'CRV1': 'modate'})

    result = extract_results(
        model, 'robust/loo/drop_ethnicity', 'robustness/leave_one_out.md',
        MAIN_OUTCOME, TREATMENT_VAR,
        f'Drop ethnicity, BW={BASELINE_BW}', 'female, education',
        'modate', 'RD-Linear', BASELINE_BW
    )
    if result:
        results.append(result)
        spec_count += 1
        print(f"  Drop ethnicity: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error dropping ethnicity: {e}")

# Leave-one-out: drop education
try:
    df_bw = df[df['dif'].between(-BASELINE_BW, BASELINE_BW)].copy()
    controls_no_edu = ['female'] + eth_cols
    formula = f'{MAIN_OUTCOME} ~ {TREATMENT_VAR} + dif_neg + dif_pos + {get_control_string(controls_no_edu)}'
    model = pf.feols(formula, data=df_bw, vcov={'CRV1': 'modate'})

    result = extract_results(
        model, 'robust/loo/drop_education', 'robustness/leave_one_out.md',
        MAIN_OUTCOME, TREATMENT_VAR,
        f'Drop education, BW={BASELINE_BW}', 'female, ethnicity',
        'modate', 'RD-Linear', BASELINE_BW
    )
    if result:
        results.append(result)
        spec_count += 1
        print(f"  Drop education: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error dropping education: {e}")

# Leave-one-out: drop female
try:
    df_bw = df[df['dif'].between(-BASELINE_BW, BASELINE_BW)].copy()
    controls_no_female = edu_cols + eth_cols
    formula = f'{MAIN_OUTCOME} ~ {TREATMENT_VAR} + dif_neg + dif_pos + {get_control_string(controls_no_female)}'
    model = pf.feols(formula, data=df_bw, vcov={'CRV1': 'modate'})

    result = extract_results(
        model, 'robust/loo/drop_female', 'robustness/leave_one_out.md',
        MAIN_OUTCOME, TREATMENT_VAR,
        f'Drop female, BW={BASELINE_BW}', 'education, ethnicity',
        'modate', 'RD-Linear', BASELINE_BW
    )
    if result:
        results.append(result)
        spec_count += 1
        print(f"  Drop female: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"  Error dropping female: {e}")

# ============================================================================
# 6. CLUSTERING VARIATIONS
# ============================================================================
print("\n--- 6. Clustering Variations ---")

cluster_options = {
    'robust': 'hetero',
    'modate': {'CRV1': 'modate'},
    'province': {'CRV1': 'province_n'},
}

for cluster_name, vcov in cluster_options.items():
    try:
        df_bw = df[df['dif'].between(-BASELINE_BW, BASELINE_BW)].copy()

        if cluster_name == 'province':
            df_bw = df_bw.dropna(subset=['province_n'])

        formula = f'{MAIN_OUTCOME} ~ {TREATMENT_VAR} + dif_neg + dif_pos + {control_formula}'
        model = pf.feols(formula, data=df_bw, vcov=vcov)

        result = extract_results(
            model, f'robust/cluster/{cluster_name}', 'robustness/clustering_variations.md',
            MAIN_OUTCOME, TREATMENT_VAR,
            f'Cluster={cluster_name}, BW={BASELINE_BW}', 'ethnicity, education, female',
            cluster_name, 'RD-Linear', BASELINE_BW
        )
        if result:
            results.append(result)
            spec_count += 1
            print(f"  Cluster={cluster_name}: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
    except Exception as e:
        print(f"  Error for cluster={cluster_name}: {e}")

# ============================================================================
# 7. DONUT HOLE SPECIFICATIONS
# ============================================================================
print("\n--- 7. Donut Hole Specifications ---")

donut_sizes = [1, 2, 3, 5]
for donut in donut_sizes:
    try:
        df_donut = df[(df['dif'].between(-BASELINE_BW, BASELINE_BW)) &
                      (np.abs(df['dif']) > donut)].copy()

        formula = f'{MAIN_OUTCOME} ~ {TREATMENT_VAR} + dif_neg + dif_pos + {control_formula}'
        model = pf.feols(formula, data=df_donut, vcov={'CRV1': 'modate'})

        result = extract_results(
            model, f'rd/donut/exclude_{donut}mo', 'methods/regression_discontinuity.md#donut-hole-specifications',
            MAIN_OUTCOME, TREATMENT_VAR,
            f'Donut hole >{donut} months, BW={BASELINE_BW}', 'ethnicity, education, female',
            'modate', 'RD-Linear', BASELINE_BW,
            {'donut_size': donut}
        )
        if result:
            results.append(result)
            spec_count += 1
            print(f"  Donut={donut}mo: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")
    except Exception as e:
        print(f"  Error for donut={donut}: {e}")

# ============================================================================
# 8. SAMPLE RESTRICTIONS - Subgroups
# ============================================================================
print("\n--- 8. Sample Restrictions (Subgroups) ---")

# Gender subsamples
for gender, gender_val in [('male', 0), ('female', 1)]:
    try:
        df_gender = df[(df['dif'].between(-BASELINE_BW, BASELINE_BW)) &
                       (df['female'] == gender_val)].copy()

        # Remove female from controls for gender subsample
        gender_controls = edu_cols + eth_cols
        formula = f'{MAIN_OUTCOME} ~ {TREATMENT_VAR} + dif_neg + dif_pos + {get_control_string(gender_controls)}'
        model = pf.feols(formula, data=df_gender, vcov={'CRV1': 'modate'})

        result = extract_results(
            model, f'robust/sample/{gender}_only', 'robustness/sample_restrictions.md',
            MAIN_OUTCOME, TREATMENT_VAR,
            f'{gender.capitalize()} only, BW={BASELINE_BW}', 'education, ethnicity',
            'modate', 'RD-Linear', BASELINE_BW
        )
        if result:
            results.append(result)
            spec_count += 1
            print(f"  {gender.capitalize()} only: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")
    except Exception as e:
        print(f"  Error for {gender}: {e}")

# Marital status subsamples
if 'married' in df.columns:
    for marital, marital_val in [('married', 1), ('not_married', 0)]:
        try:
            df_marital = df[(df['dif'].between(-BASELINE_BW, BASELINE_BW)) &
                           (df['married'] == marital_val)].copy()

            formula = f'{MAIN_OUTCOME} ~ {TREATMENT_VAR} + dif_neg + dif_pos + {control_formula}'
            model = pf.feols(formula, data=df_marital, vcov={'CRV1': 'modate'})

            result = extract_results(
                model, f'robust/sample/{marital}', 'robustness/sample_restrictions.md',
                MAIN_OUTCOME, TREATMENT_VAR,
                f'{marital.replace("_", " ").capitalize()}, BW={BASELINE_BW}', 'ethnicity, education, female',
                'modate', 'RD-Linear', BASELINE_BW
            )
            if result:
                results.append(result)
                spec_count += 1
                print(f"  {marital}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")
        except Exception as e:
            print(f"  Error for {marital}: {e}")

# Education subsamples
if 'highschool' in df.columns:
    for edu, edu_val in [('highschool', 1), ('no_highschool', 0)]:
        try:
            df_edu = df[(df['dif'].between(-BASELINE_BW, BASELINE_BW)) &
                        (df['highschool'] == edu_val)].copy()

            # Remove education controls
            edu_subset_controls = ['female'] + eth_cols
            formula = f'{MAIN_OUTCOME} ~ {TREATMENT_VAR} + dif_neg + dif_pos + {get_control_string(edu_subset_controls)}'
            model = pf.feols(formula, data=df_edu, vcov={'CRV1': 'modate'})

            result = extract_results(
                model, f'robust/sample/{edu}', 'robustness/sample_restrictions.md',
                MAIN_OUTCOME, TREATMENT_VAR,
                f'{edu.replace("_", " ").capitalize()}, BW={BASELINE_BW}', 'female, ethnicity',
                'modate', 'RD-Linear', BASELINE_BW
            )
            if result:
                results.append(result)
                spec_count += 1
                print(f"  {edu}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")
        except Exception as e:
            print(f"  Error for {edu}: {e}")

# Chronic disease subsamples
if 'chronic_disease' in df.columns:
    for chronic, chronic_val in [('chronic', 1), ('no_chronic', 0)]:
        try:
            df_chronic = df[(df['dif'].between(-BASELINE_BW, BASELINE_BW)) &
                           (df['chronic_disease'] == chronic_val)].copy()

            formula = f'{MAIN_OUTCOME} ~ {TREATMENT_VAR} + dif_neg + dif_pos + {control_formula}'
            model = pf.feols(formula, data=df_chronic, vcov={'CRV1': 'modate'})

            result = extract_results(
                model, f'robust/sample/{chronic}_disease', 'robustness/sample_restrictions.md',
                MAIN_OUTCOME, TREATMENT_VAR,
                f'{chronic.replace("_", " ").capitalize()} disease, BW={BASELINE_BW}', 'ethnicity, education, female',
                'modate', 'RD-Linear', BASELINE_BW
            )
            if result:
                results.append(result)
                spec_count += 1
                print(f"  {chronic} disease: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")
        except Exception as e:
            print(f"  Error for {chronic} disease: {e}")

# Psych support subsamples
if 'psych_support' in df.columns:
    for psych, psych_val in [('prior_psych_support', 1), ('no_prior_psych_support', 0)]:
        try:
            df_psych = df[(df['dif'].between(-BASELINE_BW, BASELINE_BW)) &
                         (df['psych_support'] == psych_val)].copy()

            formula = f'{MAIN_OUTCOME} ~ {TREATMENT_VAR} + dif_neg + dif_pos + {control_formula}'
            model = pf.feols(formula, data=df_psych, vcov={'CRV1': 'modate'})

            result = extract_results(
                model, f'robust/sample/{psych}', 'robustness/sample_restrictions.md',
                MAIN_OUTCOME, TREATMENT_VAR,
                f'{psych.replace("_", " ").capitalize()}, BW={BASELINE_BW}', 'ethnicity, education, female',
                'modate', 'RD-Linear', BASELINE_BW
            )
            if result:
                results.append(result)
                spec_count += 1
                print(f"  {psych}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")
        except Exception as e:
            print(f"  Error for {psych}: {e}")

# ============================================================================
# 9. HETEROGENEITY ANALYSIS
# ============================================================================
print("\n--- 9. Heterogeneity Analysis ---")

# Gender interaction
try:
    df_bw = df[df['dif'].between(-BASELINE_BW, BASELINE_BW)].copy()
    df_bw['treat_female'] = df_bw[TREATMENT_VAR] * df_bw['female']

    # Use only education and ethnicity controls (not female)
    het_controls = edu_cols + eth_cols
    formula = f'{MAIN_OUTCOME} ~ {TREATMENT_VAR} + female + treat_female + dif_neg + dif_pos + {get_control_string(het_controls)}'
    model = pf.feols(formula, data=df_bw, vcov={'CRV1': 'modate'})

    interact_coef = model.coef()['treat_female']
    interact_se = model.se()['treat_female']
    interact_pval = model.pvalue()['treat_female']

    result = extract_results(
        model, 'robust/het/interaction_gender', 'robustness/heterogeneity.md#interaction-specifications',
        MAIN_OUTCOME, TREATMENT_VAR,
        f'Gender interaction, BW={BASELINE_BW}', 'education, ethnicity',
        'modate', 'RD-Linear', BASELINE_BW,
        {'interaction_var': 'female', 'interaction_coef': float(interact_coef),
         'interaction_se': float(interact_se), 'interaction_pval': float(interact_pval)}
    )
    if result:
        results.append(result)
        spec_count += 1
        print(f"  Gender interaction: main={result['coefficient']:.4f}, interact={interact_coef:.4f}, p_interact={interact_pval:.4f}")
except Exception as e:
    print(f"  Error for gender interaction: {e}")

# Marital status interaction
if 'married' in df.columns:
    try:
        df_bw = df[df['dif'].between(-BASELINE_BW, BASELINE_BW)].copy()
        df_bw['treat_married'] = df_bw[TREATMENT_VAR] * df_bw['married']

        formula = f'{MAIN_OUTCOME} ~ {TREATMENT_VAR} + married + treat_married + dif_neg + dif_pos + {control_formula}'
        model = pf.feols(formula, data=df_bw, vcov={'CRV1': 'modate'})

        interact_coef = model.coef()['treat_married']
        interact_pval = model.pvalue()['treat_married']

        result = extract_results(
            model, 'robust/het/interaction_married', 'robustness/heterogeneity.md#interaction-specifications',
            MAIN_OUTCOME, TREATMENT_VAR,
            f'Marital status interaction, BW={BASELINE_BW}', 'ethnicity, education, female',
            'modate', 'RD-Linear', BASELINE_BW
        )
        if result:
            results.append(result)
            spec_count += 1
            print(f"  Married interaction: main={result['coefficient']:.4f}, interact={interact_coef:.4f}")
    except Exception as e:
        print(f"  Error for married interaction: {e}")

# Chronic disease interaction
if 'chronic_disease' in df.columns:
    try:
        df_bw = df[df['dif'].between(-BASELINE_BW, BASELINE_BW)].copy()
        df_bw['treat_chronic'] = df_bw[TREATMENT_VAR] * df_bw['chronic_disease']

        formula = f'{MAIN_OUTCOME} ~ {TREATMENT_VAR} + chronic_disease + treat_chronic + dif_neg + dif_pos + {control_formula}'
        model = pf.feols(formula, data=df_bw, vcov={'CRV1': 'modate'})

        interact_coef = model.coef()['treat_chronic']
        interact_pval = model.pvalue()['treat_chronic']

        result = extract_results(
            model, 'robust/het/interaction_chronic', 'robustness/heterogeneity.md#interaction-specifications',
            MAIN_OUTCOME, TREATMENT_VAR,
            f'Chronic disease interaction, BW={BASELINE_BW}', 'ethnicity, education, female',
            'modate', 'RD-Linear', BASELINE_BW
        )
        if result:
            results.append(result)
            spec_count += 1
            print(f"  Chronic interaction: main={result['coefficient']:.4f}, interact={interact_coef:.4f}")
    except Exception as e:
        print(f"  Error for chronic interaction: {e}")

# Prior psych support interaction
if 'psych_support' in df.columns:
    try:
        df_bw = df[df['dif'].between(-BASELINE_BW, BASELINE_BW)].copy()
        df_bw['treat_psych'] = df_bw[TREATMENT_VAR] * df_bw['psych_support']

        formula = f'{MAIN_OUTCOME} ~ {TREATMENT_VAR} + psych_support + treat_psych + dif_neg + dif_pos + {control_formula}'
        model = pf.feols(formula, data=df_bw, vcov={'CRV1': 'modate'})

        interact_coef = model.coef()['treat_psych']
        interact_pval = model.pvalue()['treat_psych']

        result = extract_results(
            model, 'robust/het/interaction_psych_support', 'robustness/heterogeneity.md#interaction-specifications',
            MAIN_OUTCOME, TREATMENT_VAR,
            f'Psych support interaction, BW={BASELINE_BW}', 'ethnicity, education, female',
            'modate', 'RD-Linear', BASELINE_BW
        )
        if result:
            results.append(result)
            spec_count += 1
            print(f"  Psych support interaction: main={result['coefficient']:.4f}, interact={interact_coef:.4f}")
    except Exception as e:
        print(f"  Error for psych support interaction: {e}")

# ============================================================================
# 10. PLACEBO TESTS
# ============================================================================
print("\n--- 10. Placebo Tests ---")

# Placebo cutoffs
placebo_cutoffs = [-24, -12, 12, 24]

for placebo_cut in placebo_cutoffs:
    try:
        df_placebo = df.copy()
        df_placebo['dif_placebo'] = df_placebo['dif'] - placebo_cut
        df_placebo['before_placebo'] = (df_placebo['dif_placebo'] < 0).astype(int)
        df_placebo['dif_placebo_neg'] = df_placebo['dif_placebo'] * (df_placebo['dif_placebo'] < 0).astype(int)
        df_placebo['dif_placebo_pos'] = df_placebo['dif_placebo'] * (df_placebo['dif_placebo'] >= 0).astype(int)

        df_placebo = df_placebo[df_placebo['dif_placebo'].between(-BASELINE_BW, BASELINE_BW)].copy()

        formula = f'{MAIN_OUTCOME} ~ before_placebo + dif_placebo_neg + dif_placebo_pos + {control_formula}'
        model = pf.feols(formula, data=df_placebo, vcov={'CRV1': 'modate'})

        result = extract_results(
            model, f'rd/placebo/cutoff_{placebo_cut:+d}mo', 'methods/regression_discontinuity.md#placebo-cutoff-tests',
            MAIN_OUTCOME, 'before_placebo',
            f'Placebo cutoff at {placebo_cut:+d} months, BW={BASELINE_BW}', 'ethnicity, education, female',
            'modate', 'RD-Linear', BASELINE_BW,
            {'placebo_cutoff_offset': placebo_cut}
        )
        if result:
            results.append(result)
            spec_count += 1
            print(f"  Placebo cutoff {placebo_cut:+d}mo: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
    except Exception as e:
        print(f"  Error for placebo cutoff {placebo_cut}: {e}")

# Placebo outcomes (predetermined covariates)
placebo_outcomes = ['highschool', 'illiterate', 'female', 'married', 'non_turk',
                    'pre_covid_hhsize', 'psych_support', 'chronic_disease']

for placebo_y in placebo_outcomes:
    if placebo_y not in df.columns:
        continue
    try:
        df_bw = df[df['dif'].between(-BASELINE_BW, BASELINE_BW)].copy()
        df_bw = df_bw.dropna(subset=[placebo_y])

        formula = f'{placebo_y} ~ {TREATMENT_VAR} + dif_neg + dif_pos'
        model = pf.feols(formula, data=df_bw, vcov={'CRV1': 'modate'})

        result = extract_results(
            model, f'rd/validity/covariate_{placebo_y}', 'methods/regression_discontinuity.md#validation-tests',
            placebo_y, TREATMENT_VAR,
            f'Covariate balance: {placebo_y}, BW={BASELINE_BW}', 'None',
            'modate', 'RD-Linear', BASELINE_BW
        )
        if result:
            results.append(result)
            spec_count += 1
            print(f"  Balance {placebo_y}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
    except Exception as e:
        print(f"  Error for balance {placebo_y}: {e}")

# ============================================================================
# 11. CHANNEL OUTCOMES
# ============================================================================
print("\n--- 11. Channel Outcomes ---")

channel_vars = ['paid_employment', 'total_employment', 'job_to_return',
                'money_as_usual', 'money_distressed', 'hh_size', 'conflict',
                'limited_physical_act', 'lim_social_interaction']

for channel in channel_vars:
    if channel not in df.columns:
        continue
    try:
        df_bw = df[df['dif'].between(-BASELINE_BW, BASELINE_BW)].copy()
        df_bw = df_bw.dropna(subset=[channel])

        formula = f'{channel} ~ {TREATMENT_VAR} + dif_neg + dif_pos + {control_formula}'
        model = pf.feols(formula, data=df_bw, vcov={'CRV1': 'modate'})

        result = extract_results(
            model, f'robust/outcome/channel_{channel}', 'robustness/measurement.md',
            channel, TREATMENT_VAR,
            f'Channel outcome: {channel}, BW={BASELINE_BW}', 'ethnicity, education, female',
            'modate', 'RD-Linear', BASELINE_BW
        )
        if result:
            results.append(result)
            spec_count += 1
            print(f"  Channel {channel}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
    except Exception as e:
        print(f"  Error for channel {channel}: {e}")

# ============================================================================
# 12. INDIVIDUAL SYMPTOMS
# ============================================================================
print("\n--- 12. Individual SRQ-20 Symptoms ---")

for symptom in available_symptoms[:12]:
    try:
        df_bw = df[df['dif'].between(-BASELINE_BW, BASELINE_BW)].copy()
        df_bw = df_bw.dropna(subset=[symptom])

        formula = f'{symptom} ~ {TREATMENT_VAR} + dif_neg + dif_pos + {control_formula}'
        model = pf.feols(formula, data=df_bw, vcov={'CRV1': 'modate'})

        result = extract_results(
            model, f'robust/outcome/symptom_{symptom}', 'robustness/measurement.md',
            symptom, TREATMENT_VAR,
            f'Individual symptom: {symptom}, BW={BASELINE_BW}', 'ethnicity, education, female',
            'modate', 'RD-Linear', BASELINE_BW
        )
        if result:
            results.append(result)
            spec_count += 1
            print(f"  Symptom {symptom}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
    except Exception as e:
        print(f"  Error for symptom {symptom}: {e}")

# ============================================================================
# 13. OUTLIER HANDLING
# ============================================================================
print("\n--- 13. Outlier Handling ---")

for pct in [1, 5, 10]:
    try:
        df_bw = df[df['dif'].between(-BASELINE_BW, BASELINE_BW)].copy()

        lower = df_bw[MAIN_OUTCOME].quantile(pct/100)
        upper = df_bw[MAIN_OUTCOME].quantile(1 - pct/100)
        df_bw['outcome_wins'] = df_bw[MAIN_OUTCOME].clip(lower=lower, upper=upper)

        formula = f'outcome_wins ~ {TREATMENT_VAR} + dif_neg + dif_pos + {control_formula}'
        model = pf.feols(formula, data=df_bw, vcov={'CRV1': 'modate'})

        result = extract_results(
            model, f'robust/sample/winsor_{pct}pct', 'robustness/sample_restrictions.md#outlier-handling',
            f'{MAIN_OUTCOME}_wins', TREATMENT_VAR,
            f'Winsorized {pct}%, BW={BASELINE_BW}', 'ethnicity, education, female',
            'modate', 'RD-Linear', BASELINE_BW
        )
        if result:
            results.append(result)
            spec_count += 1
            print(f"  Winsor {pct}%: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")
    except Exception as e:
        print(f"  Error for winsor {pct}%: {e}")

# Trimming
for pct in [1, 5]:
    try:
        df_bw = df[df['dif'].between(-BASELINE_BW, BASELINE_BW)].copy()

        lower = df_bw[MAIN_OUTCOME].quantile(pct/100)
        upper = df_bw[MAIN_OUTCOME].quantile(1 - pct/100)
        df_trim = df_bw[(df_bw[MAIN_OUTCOME] >= lower) & (df_bw[MAIN_OUTCOME] <= upper)].copy()

        formula = f'{MAIN_OUTCOME} ~ {TREATMENT_VAR} + dif_neg + dif_pos + {control_formula}'
        model = pf.feols(formula, data=df_trim, vcov={'CRV1': 'modate'})

        result = extract_results(
            model, f'robust/sample/trim_{pct}pct', 'robustness/sample_restrictions.md#outlier-handling',
            MAIN_OUTCOME, TREATMENT_VAR,
            f'Trimmed {pct}%, BW={BASELINE_BW}', 'ethnicity, education, female',
            'modate', 'RD-Linear', BASELINE_BW
        )
        if result:
            results.append(result)
            spec_count += 1
            print(f"  Trim {pct}%: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")
    except Exception as e:
        print(f"  Error for trim {pct}%: {e}")

# ============================================================================
# 14. SYMMETRIC SAMPLE
# ============================================================================
print("\n--- 14. Symmetric Sample ---")

try:
    df_bw = df[df['dif'].between(-BASELINE_BW, BASELINE_BW)].copy()

    n_left = (df_bw['dif'] < 0).sum()
    n_right = (df_bw['dif'] >= 0).sum()
    min_n = min(n_left, n_right)

    np.random.seed(42)
    left_sample = df_bw[df_bw['dif'] < 0].sample(n=min_n, random_state=42)
    right_sample = df_bw[df_bw['dif'] >= 0].sample(n=min_n, random_state=42)
    df_symmetric = pd.concat([left_sample, right_sample])

    formula = f'{MAIN_OUTCOME} ~ {TREATMENT_VAR} + dif_neg + dif_pos + {control_formula}'
    model = pf.feols(formula, data=df_symmetric, vcov={'CRV1': 'modate'})

    result = extract_results(
        model, 'rd/sample/symmetric', 'methods/regression_discontinuity.md#sample-restrictions',
        MAIN_OUTCOME, TREATMENT_VAR,
        f'Symmetric sample, BW={BASELINE_BW}', 'ethnicity, education, female',
        'modate', 'RD-Linear', BASELINE_BW
    )
    if result:
        results.append(result)
        spec_count += 1
        print(f"  Symmetric: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")
except Exception as e:
    print(f"  Error for symmetric: {e}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print(f"COMPLETED: {spec_count} specifications")
print("="*80)

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
output_file = OUTPUT_DIR / 'specification_results.csv'
results_df.to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

if len(results_df) > 0:
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    # Filter to main outcome specifications
    main_specs = results_df[results_df['outcome_var'] == MAIN_OUTCOME]

    if len(main_specs) > 0:
        print(f"\nMain outcome ({MAIN_OUTCOME}) specifications: {len(main_specs)}")
        print(f"  Positive coefficients: {(main_specs['coefficient'] > 0).sum()} ({100*(main_specs['coefficient'] > 0).mean():.1f}%)")
        print(f"  Significant at 5%: {(main_specs['p_value'] < 0.05).sum()} ({100*(main_specs['p_value'] < 0.05).mean():.1f}%)")
        print(f"  Significant at 1%: {(main_specs['p_value'] < 0.01).sum()} ({100*(main_specs['p_value'] < 0.01).mean():.1f}%)")
        print(f"  Median coefficient: {main_specs['coefficient'].median():.4f}")
        print(f"  Mean coefficient: {main_specs['coefficient'].mean():.4f}")
        print(f"  Coefficient range: [{main_specs['coefficient'].min():.4f}, {main_specs['coefficient'].max():.4f}]")

    print(f"\nTotal specifications: {len(results_df)}")

    # Category breakdown
    print("\n--- Breakdown by Category ---")
    categories = {
        'Baseline': results_df['spec_id'].str.startswith('baseline'),
        'Bandwidth': results_df['spec_id'].str.startswith('rd/bandwidth'),
        'Polynomial': results_df['spec_id'].str.startswith('rd/poly'),
        'Control variations': results_df['spec_id'].str.contains('loo|controls'),
        'Sample restrictions': results_df['spec_id'].str.contains('sample|donut'),
        'Clustering': results_df['spec_id'].str.contains('cluster'),
        'Alternative outcomes': results_df['spec_id'].str.contains('outcome|symptom|channel'),
        'Heterogeneity': results_df['spec_id'].str.contains('het'),
        'Placebo/Validity': results_df['spec_id'].str.contains('placebo|validity'),
    }

    for cat, mask in categories.items():
        n = mask.sum()
        if n > 0:
            cat_df = results_df[mask]
            pct_sig = 100 * (cat_df['p_value'] < 0.05).mean()
            print(f"  {cat}: {n} specs, {pct_sig:.1f}% sig at 5%")

print("\nDone!")
