"""
Specification Search: Lopez, Sautmann, Schaner (2020)
Paper ID: 126722-V1
Journal: AEJ Applied
Title: Allocating Health Care Resources Efficiently: The Simple Economics of Vouchers vs. In-Kind Provision

This paper studies the effects of malaria treatment vouchers in Mali using a
randomized controlled trial (RCT). Patients visiting health clinics were randomized
to receive either:
- Patient voucher: voucher given directly to patients for antimalarial drugs
- Doctor voucher: voucher given to doctors to distribute
- Control: no voucher

Key outcomes:
- Malaria treatment prescribed (RXtreat_sev_simple_mal)
- Malaria treatment purchased (treat_sev_simple_mal)
- Voucher usage (used_vouchers_admin)
- Treatment-illness match (expected_mal_match_any)

Method: Cross-sectional RCT with clustering at clinic level
Standard errors clustered at clinic (cscom) level
Original analysis uses double-selection LASSO with date fixed effects as partials

Created by: Specification Search Agent
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import warnings
import os

warnings.filterwarnings('ignore')

# =============================================================================
# SETUP
# =============================================================================

# Paths
BASE_DIR = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search'
DATA_PATH = f'{BASE_DIR}/data/downloads/extracted/126722-V1/Lopez_Sautmann_Schaner_2020/Data/LSS_analysis_datasets_20201108.dta'
OUTPUT_DIR = f'{BASE_DIR}/data/downloads/extracted/126722-V1/Lopez_Sautmann_Schaner_2020'

# Paper metadata
PAPER_ID = '126722-V1'
JOURNAL = 'AEJ-Applied'
PAPER_TITLE = 'Allocating Health Care Resources Efficiently: The Simple Economics of Vouchers vs. In-Kind Provision'

# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================

print("Loading data...")
df = pd.read_stata(DATA_PATH)

# Filter to analysis sample (drop if dropme==1)
df = df[df['dropme'] == 0].copy()
print(f"Analysis sample size: {len(df)}")

# Convert categorical variables to numeric
# Treatment variables
df['patient_voucher'] = (df['patient_voucher'] == 'Yes').astype(int)
df['doctor_voucher'] = (df['doctor_voucher'] == 'Yes').astype(int)
df['patient_info'] = (df['patient_info'] == 'Yes').astype(int)
df['infotreat'] = (df['infotreat'] == 'Yes').astype(int)

# Outcome variables (convert from categorical)
outcome_cats = ['treat_sev_simple_mal', 'RXtreat_sev_simple_mal', 'treat_severe_mal',
                'RXtreat_severe_mal']
for col in outcome_cats:
    if col in df.columns:
        df[col] = (df[col] == 'Yes').astype(float)
        df.loc[df[col].isna() | (df[col] == -1), col] = np.nan

# Numeric outcomes
numeric_outcomes = ['used_vouchers_admin', 'expected_mal_match_any', 'RXexpected_mal_match_any',
                    'expected_mal_match_any_pos', 'expected_mal_match_any_neg',
                    'RXexpected_mal_match_any_pos', 'RXexpected_mal_match_any_neg']
for col in numeric_outcomes:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Control variables (convert from categorical where needed)
cat_controls = ['gender', 'genderpatient', 'respondent', 'ethnic_bambara',
                'speak_french', 'readwrite_fluent_french', 'prischoolorless']
for col in cat_controls:
    if col in df.columns:
        if df[col].dtype.name == 'category':
            # Convert to dummy (1 if first category after sorting, varies by var)
            df[col] = pd.Categorical(df[col]).codes
            df.loc[df[col] == -1, col] = np.nan

# Numeric controls
numeric_controls = ['num_symptoms', 'daysillness99', 'agepatient', 'under5',
                    'pregnancy', 'pred_mal_pos', 'above_med_pos']
for col in numeric_controls:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Symptom dummies
symptom_cols = [f'symptomsscreening_{i}' for i in range(1, 8)]
for col in symptom_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Create clinic identifier from cscom columns
# Find the cscom numeric columns for clinic FE
cscom_cols = [c for c in df.columns if c.startswith('cscom_') and c[6:].isdigit()]
if len(cscom_cols) > 0:
    # Create clinic ID from the cscom dummies
    df['clinic_id'] = 0
    for i, col in enumerate(cscom_cols):
        if col in df.columns:
            df['clinic_id'] += (i + 1) * pd.to_numeric(df[col], errors='coerce').fillna(0)

# Use cscom_day as alternative clinic ID
df['clinic_day'] = df['cscom_day'].astype(str)

# Create date fixed effects columns
date_cols = [c for c in df.columns if c.startswith('date_') and c[5:].isdigit()]
df['date_fe'] = 0
for i, col in enumerate(date_cols):
    if col in df.columns:
        df['date_fe'] += (i + 1) * pd.to_numeric(df[col], errors='coerce').fillna(0)

# Create interaction variables for heterogeneity
df['patient_voucher_high'] = df['patient_voucher'] * df['above_med_pos']
df['doctor_voucher_high'] = df['doctor_voucher'] * df['above_med_pos']
df['patient_voucher_low'] = df['patient_voucher'] * (1 - df['above_med_pos'])
df['doctor_voucher_low'] = df['doctor_voucher'] * (1 - df['above_med_pos'])

# Combined voucher variable
df['any_voucher'] = ((df['patient_voucher'] == 1) | (df['doctor_voucher'] == 1)).astype(int)

print(f"Variables prepared. N = {len(df)}")

# =============================================================================
# DEFINE CONTROL SETS
# =============================================================================

# Date FE dummies
DATE_FE_COLS = date_cols[:35]  # DD1-DD35

# Basic patient controls (from paper's no-lasso specification)
BASIC_CONTROLS = ['num_symptoms', 'daysillness99', 'agepatient', 'under5',
                  'genderpatient', 'pregnancy']

# Respondent controls
RESPONDENT_CONTROLS = ['respondent', 'gender', 'ethnic_bambara', 'speak_french',
                       'readwrite_fluent_french', 'prischoolorless']

# Symptom controls
SYMPTOM_CONTROLS = symptom_cols

# Full control set (no LASSO, Table B10 style)
FULL_CONTROLS = BASIC_CONTROLS + RESPONDENT_CONTROLS + SYMPTOM_CONTROLS

# Clinic dummies
CLINIC_COLS = cscom_cols[:59]  # CL1-CL59

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_valid_controls(df, controls):
    """Return list of controls that exist and have variation"""
    valid = []
    for c in controls:
        if c in df.columns:
            vals = df[c].dropna()
            if len(vals) > 0 and vals.nunique() > 1:
                valid.append(c)
    return valid

def create_formula(outcome, treatments, controls, fe=None):
    """Create pyfixest formula"""
    treat_str = ' + '.join(treatments)
    if controls:
        ctrl_str = ' + '.join(controls)
        rhs = f'{treat_str} + {ctrl_str}'
    else:
        rhs = treat_str

    if fe:
        formula = f'{outcome} ~ {rhs} | {fe}'
    else:
        formula = f'{outcome} ~ {rhs}'

    return formula

def run_regression(df, outcome, treatments, controls=None, fe=None, cluster=None,
                   spec_id='', spec_tree_path='', sample_desc='Full sample'):
    """Run regression and return result dict"""

    # Get valid controls
    if controls:
        valid_controls = get_valid_controls(df, controls)
    else:
        valid_controls = []

    # Create formula
    formula = create_formula(outcome, treatments, valid_controls, fe)

    try:
        # Run regression
        if cluster:
            model = pf.feols(formula, data=df, vcov={'CRV1': cluster})
        else:
            model = pf.feols(formula, data=df, vcov='hetero')

        # Extract main treatment coefficient (patient_voucher if present)
        treatment_var = treatments[0]  # Primary treatment

        coef = model.coef()[treatment_var]
        se = model.se()[treatment_var]
        tstat = model.tstat()[treatment_var]
        pval = model.pvalue()[treatment_var]

        # Confidence interval
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
            'fixed_effects': [fe] if fe else [],
            'diagnostics': {}
        }

        # Add other treatment coefficients
        for t in treatments[1:]:
            if t in model.coef().index:
                coef_vector['controls'].append({
                    'var': t,
                    'coef': float(model.coef()[t]),
                    'se': float(model.se()[t]),
                    'pval': float(model.pvalue()[t])
                })

        # Add control coefficients
        for c in valid_controls:
            if c in model.coef().index:
                coef_vector['controls'].append({
                    'var': c,
                    'coef': float(model.coef()[c]),
                    'se': float(model.se()[c]),
                    'pval': float(model.pvalue()[c])
                })

        result = {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': outcome,
            'treatment_var': treatment_var,
            'coefficient': float(coef),
            'std_error': float(se),
            't_stat': float(tstat),
            'p_value': float(pval),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_obs': int(model._N),
            'r_squared': float(model._r2) if hasattr(model, '_r2') else np.nan,
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc,
            'fixed_effects': fe if fe else 'None',
            'controls_desc': ', '.join(valid_controls) if valid_controls else 'None',
            'cluster_var': cluster if cluster else 'None',
            'model_type': 'OLS with FE' if fe else 'OLS',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }

        return result

    except Exception as e:
        print(f"Error in spec {spec_id}: {str(e)}")
        return None

# =============================================================================
# RUN SPECIFICATIONS
# =============================================================================

results = []

# Define primary outcome and treatments
PRIMARY_OUTCOME = 'treat_sev_simple_mal'  # Purchased any malaria treatment
PRIMARY_TREATMENTS = ['patient_voucher', 'doctor_voucher']

# All outcomes to test
ALL_OUTCOMES = ['treat_sev_simple_mal', 'RXtreat_sev_simple_mal',
                'treat_severe_mal', 'RXtreat_severe_mal',
                'used_vouchers_admin', 'expected_mal_match_any', 'RXexpected_mal_match_any']

# Valid date FE for formulas
date_fe_valid = get_valid_controls(df, DATE_FE_COLS)
date_fe_str = 'date_fe'

print("\n" + "="*60)
print("RUNNING SPECIFICATION SEARCH")
print("="*60)

# -----------------------------------------------------------------------------
# 1. BASELINE SPECIFICATION (replicating Table 3)
# -----------------------------------------------------------------------------
print("\n--- Baseline Specifications ---")

# Baseline with date FE and clustered SEs (no additional controls - simplest version)
result = run_regression(
    df, PRIMARY_OUTCOME, PRIMARY_TREATMENTS,
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='baseline',
    spec_tree_path='methods/cross_sectional_ols.md',
    sample_desc='Full analysis sample'
)
if result:
    results.append(result)
    print(f"Baseline: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")

# -----------------------------------------------------------------------------
# 2. OUTCOME VARIATIONS
# -----------------------------------------------------------------------------
print("\n--- Alternative Outcomes ---")

for outcome in ALL_OUTCOMES:
    if outcome == PRIMARY_OUTCOME:
        continue

    result = run_regression(
        df, outcome, PRIMARY_TREATMENTS,
        controls=date_fe_valid, fe=None, cluster='clinic_day',
        spec_id=f'robust/outcome/{outcome}',
        spec_tree_path='robustness/outcome_variations.md',
        sample_desc='Full analysis sample'
    )
    if result:
        results.append(result)
        print(f"Outcome {outcome}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")

# -----------------------------------------------------------------------------
# 3. CONTROL VARIATIONS
# -----------------------------------------------------------------------------
print("\n--- Control Variations ---")

# 3a. No controls (bivariate)
result = run_regression(
    df, PRIMARY_OUTCOME, PRIMARY_TREATMENTS,
    controls=None, fe=None, cluster='clinic_day',
    spec_id='robust/control/none',
    spec_tree_path='robustness/control_variations.md',
    sample_desc='Full sample, no controls'
)
if result:
    results.append(result)
    print(f"No controls: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# 3b. Date FE only (paper's minimal)
result = run_regression(
    df, PRIMARY_OUTCOME, PRIMARY_TREATMENTS,
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/control/date_fe_only',
    spec_tree_path='robustness/control_variations.md',
    sample_desc='Date FE only'
)
if result:
    results.append(result)
    print(f"Date FE only: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# 3c. Basic patient controls
result = run_regression(
    df, PRIMARY_OUTCOME, PRIMARY_TREATMENTS,
    controls=date_fe_valid + get_valid_controls(df, BASIC_CONTROLS),
    fe=None, cluster='clinic_day',
    spec_id='robust/control/basic_patient',
    spec_tree_path='robustness/control_variations.md',
    sample_desc='Date FE + basic patient controls'
)
if result:
    results.append(result)
    print(f"Basic controls: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# 3d. Full controls (Table B10 style)
result = run_regression(
    df, PRIMARY_OUTCOME, PRIMARY_TREATMENTS,
    controls=date_fe_valid + get_valid_controls(df, FULL_CONTROLS),
    fe=None, cluster='clinic_day',
    spec_id='robust/control/full',
    spec_tree_path='robustness/control_variations.md',
    sample_desc='Date FE + full controls'
)
if result:
    results.append(result)
    print(f"Full controls: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# 3e. Leave-one-out for each control
print("\n--- Leave-One-Out Control Analysis ---")
for ctrl in get_valid_controls(df, BASIC_CONTROLS + RESPONDENT_CONTROLS):
    remaining = [c for c in get_valid_controls(df, BASIC_CONTROLS + RESPONDENT_CONTROLS) if c != ctrl]
    result = run_regression(
        df, PRIMARY_OUTCOME, PRIMARY_TREATMENTS,
        controls=date_fe_valid + remaining,
        fe=None, cluster='clinic_day',
        spec_id=f'robust/control/drop_{ctrl}',
        spec_tree_path='robustness/leave_one_out.md',
        sample_desc=f'Dropped control: {ctrl}'
    )
    if result:
        results.append(result)
        print(f"Drop {ctrl}: coef={result['coefficient']:.4f}")

# 3f. Add controls incrementally
print("\n--- Incremental Control Addition ---")
controls_to_add = get_valid_controls(df, BASIC_CONTROLS + RESPONDENT_CONTROLS)
current_controls = list(date_fe_valid)
for i, ctrl in enumerate(controls_to_add):
    current_controls.append(ctrl)
    result = run_regression(
        df, PRIMARY_OUTCOME, PRIMARY_TREATMENTS,
        controls=current_controls,
        fe=None, cluster='clinic_day',
        spec_id=f'robust/control/add_{ctrl}',
        spec_tree_path='robustness/control_progression.md',
        sample_desc=f'Added controls through: {ctrl}'
    )
    if result:
        results.append(result)
        print(f"Add {ctrl}: coef={result['coefficient']:.4f}")

# -----------------------------------------------------------------------------
# 4. TREATMENT VARIATIONS
# -----------------------------------------------------------------------------
print("\n--- Treatment Variations ---")

# 4a. Only patient voucher
result = run_regression(
    df, PRIMARY_OUTCOME, ['patient_voucher'],
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/treatment/patient_voucher_only',
    spec_tree_path='robustness/treatment_variations.md',
    sample_desc='Patient voucher effect only'
)
if result:
    results.append(result)
    print(f"Patient voucher only: coef={result['coefficient']:.4f}")

# 4b. Only doctor voucher
result = run_regression(
    df, PRIMARY_OUTCOME, ['doctor_voucher'],
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/treatment/doctor_voucher_only',
    spec_tree_path='robustness/treatment_variations.md',
    sample_desc='Doctor voucher effect only'
)
if result:
    results.append(result)
    print(f"Doctor voucher only: coef={result['coefficient']:.4f}")

# 4c. Combined any voucher
result = run_regression(
    df, PRIMARY_OUTCOME, ['any_voucher'],
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/treatment/any_voucher',
    spec_tree_path='robustness/treatment_variations.md',
    sample_desc='Any voucher effect'
)
if result:
    results.append(result)
    print(f"Any voucher: coef={result['coefficient']:.4f}")

# -----------------------------------------------------------------------------
# 5. INFERENCE VARIATIONS
# -----------------------------------------------------------------------------
print("\n--- Inference Variations ---")

# 5a. Robust SE (no clustering)
result = run_regression(
    df, PRIMARY_OUTCOME, PRIMARY_TREATMENTS,
    controls=date_fe_valid, fe=None, cluster=None,
    spec_id='robust/cluster/robust_hc1',
    spec_tree_path='robustness/clustering_variations.md',
    sample_desc='Robust (unclustered) SEs'
)
if result:
    results.append(result)
    print(f"Robust SE: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}")

# 5b. Cluster by clinic_day (default)
result = run_regression(
    df, PRIMARY_OUTCOME, PRIMARY_TREATMENTS,
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/cluster/clinic_day',
    spec_tree_path='robustness/clustering_variations.md',
    sample_desc='Clustered by clinic-day'
)
if result:
    results.append(result)
    print(f"Cluster clinic_day: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}")

# 5c. Cluster by date_fe (time clustering)
result = run_regression(
    df, PRIMARY_OUTCOME, PRIMARY_TREATMENTS,
    controls=date_fe_valid, fe=None, cluster='date_fe',
    spec_id='robust/cluster/date',
    spec_tree_path='robustness/clustering_variations.md',
    sample_desc='Clustered by date'
)
if result:
    results.append(result)
    print(f"Cluster date: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}")

# -----------------------------------------------------------------------------
# 6. SAMPLE RESTRICTIONS
# -----------------------------------------------------------------------------
print("\n--- Sample Restrictions ---")

# 6a. Under 5 only
df_u5 = df[df['under5'] == 1].copy()
result = run_regression(
    df_u5, PRIMARY_OUTCOME, PRIMARY_TREATMENTS,
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/sample/under5',
    spec_tree_path='robustness/sample_restrictions.md',
    sample_desc='Children under 5 only'
)
if result:
    results.append(result)
    print(f"Under 5: coef={result['coefficient']:.4f}, n={result['n_obs']}")

# 6b. 5 and over
df_o5 = df[df['under5'] == 0].copy()
result = run_regression(
    df_o5, PRIMARY_OUTCOME, PRIMARY_TREATMENTS,
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/sample/age5plus',
    spec_tree_path='robustness/sample_restrictions.md',
    sample_desc='Age 5 and over'
)
if result:
    results.append(result)
    print(f"Age 5+: coef={result['coefficient']:.4f}, n={result['n_obs']}")

# 6c. Male patients only
df_male = df[df['genderpatient'] == 1].copy()
result = run_regression(
    df_male, PRIMARY_OUTCOME, PRIMARY_TREATMENTS,
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/sample/male',
    spec_tree_path='robustness/sample_restrictions.md',
    sample_desc='Male patients only'
)
if result:
    results.append(result)
    print(f"Male: coef={result['coefficient']:.4f}, n={result['n_obs']}")

# 6d. Female patients only
df_female = df[df['genderpatient'] == 0].copy()
result = run_regression(
    df_female, PRIMARY_OUTCOME, PRIMARY_TREATMENTS,
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/sample/female',
    spec_tree_path='robustness/sample_restrictions.md',
    sample_desc='Female patients only'
)
if result:
    results.append(result)
    print(f"Female: coef={result['coefficient']:.4f}, n={result['n_obs']}")

# 6e. High malaria risk
df_high = df[df['above_med_pos'] == 1].copy()
result = run_regression(
    df_high, PRIMARY_OUTCOME, PRIMARY_TREATMENTS,
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/sample/high_malaria_risk',
    spec_tree_path='robustness/sample_restrictions.md',
    sample_desc='High predicted malaria risk'
)
if result:
    results.append(result)
    print(f"High risk: coef={result['coefficient']:.4f}, n={result['n_obs']}")

# 6f. Low malaria risk
df_low = df[df['above_med_pos'] == 0].copy()
result = run_regression(
    df_low, PRIMARY_OUTCOME, PRIMARY_TREATMENTS,
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/sample/low_malaria_risk',
    spec_tree_path='robustness/sample_restrictions.md',
    sample_desc='Low predicted malaria risk'
)
if result:
    results.append(result)
    print(f"Low risk: coef={result['coefficient']:.4f}, n={result['n_obs']}")

# 6g. Trim outliers on days ill (top/bottom 1%)
df_trim = df.copy()
p01 = df_trim['daysillness99'].quantile(0.01)
p99 = df_trim['daysillness99'].quantile(0.99)
df_trim = df_trim[(df_trim['daysillness99'] >= p01) & (df_trim['daysillness99'] <= p99)]
result = run_regression(
    df_trim, PRIMARY_OUTCOME, PRIMARY_TREATMENTS,
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/sample/trim_daysill_1pct',
    spec_tree_path='robustness/sample_restrictions.md',
    sample_desc='Trimmed days ill (1-99%)'
)
if result:
    results.append(result)
    print(f"Trim days ill: coef={result['coefficient']:.4f}, n={result['n_obs']}")

# 6h. Multiple symptoms only
df_multi = df[df['num_symptoms'] > 1].copy()
result = run_regression(
    df_multi, PRIMARY_OUTCOME, PRIMARY_TREATMENTS,
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/sample/multiple_symptoms',
    spec_tree_path='robustness/sample_restrictions.md',
    sample_desc='Patients with >1 symptom'
)
if result:
    results.append(result)
    print(f"Multiple symptoms: coef={result['coefficient']:.4f}, n={result['n_obs']}")

# 6i. Single symptom only
df_single = df[df['num_symptoms'] == 1].copy()
result = run_regression(
    df_single, PRIMARY_OUTCOME, PRIMARY_TREATMENTS,
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/sample/single_symptom',
    spec_tree_path='robustness/sample_restrictions.md',
    sample_desc='Patients with 1 symptom'
)
if result:
    results.append(result)
    print(f"Single symptom: coef={result['coefficient']:.4f}, n={result['n_obs']}")

# 6j. Non-pregnant only (only females can be pregnant)
df_nopreg = df[(df['pregnancy'] == 0) | (df['pregnancy'].isna())].copy()
if len(df_nopreg) > 100:
    result = run_regression(
        df_nopreg, PRIMARY_OUTCOME, PRIMARY_TREATMENTS,
        controls=date_fe_valid, fe=None, cluster='clinic_day',
        spec_id='robust/sample/non_pregnant',
        spec_tree_path='robustness/sample_restrictions.md',
        sample_desc='Non-pregnant patients'
    )
    if result:
        results.append(result)
        print(f"Non-pregnant: coef={result['coefficient']:.4f}, n={result['n_obs']}")

# 6k. French speakers only
df_french = df[df['speak_french'] == 1].copy()
result = run_regression(
    df_french, PRIMARY_OUTCOME, PRIMARY_TREATMENTS,
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/sample/french_speaker',
    spec_tree_path='robustness/sample_restrictions.md',
    sample_desc='French speakers only'
)
if result:
    results.append(result)
    print(f"French speakers: coef={result['coefficient']:.4f}, n={result['n_obs']}")

# 6l. Non-French speakers only
df_nofrench = df[df['speak_french'] == 0].copy()
result = run_regression(
    df_nofrench, PRIMARY_OUTCOME, PRIMARY_TREATMENTS,
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/sample/non_french_speaker',
    spec_tree_path='robustness/sample_restrictions.md',
    sample_desc='Non-French speakers only'
)
if result:
    results.append(result)
    print(f"Non-French speakers: coef={result['coefficient']:.4f}, n={result['n_obs']}")

# -----------------------------------------------------------------------------
# 7. HETEROGENEITY ANALYSIS
# -----------------------------------------------------------------------------
print("\n--- Heterogeneity Analysis ---")

# 7a. Interaction with high/low risk (Table 4 style)
result = run_regression(
    df, PRIMARY_OUTCOME,
    ['patient_voucher_high', 'doctor_voucher_high', 'patient_voucher_low', 'doctor_voucher_low', 'above_med_pos'],
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/het/malaria_risk',
    spec_tree_path='robustness/heterogeneity.md',
    sample_desc='Interaction with malaria risk'
)
if result:
    results.append(result)
    print(f"Risk heterogeneity: coef={result['coefficient']:.4f}")

# 7b. Interaction with under5
df['patient_voucher_under5'] = df['patient_voucher'] * df['under5']
df['doctor_voucher_under5'] = df['doctor_voucher'] * df['under5']
result = run_regression(
    df, PRIMARY_OUTCOME,
    ['patient_voucher', 'doctor_voucher', 'patient_voucher_under5', 'doctor_voucher_under5', 'under5'],
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/het/under5',
    spec_tree_path='robustness/heterogeneity.md',
    sample_desc='Interaction with under 5'
)
if result:
    results.append(result)
    print(f"Under5 heterogeneity: coef={result['coefficient']:.4f}")

# 7c. Interaction with gender
df['patient_voucher_male'] = df['patient_voucher'] * df['genderpatient']
df['doctor_voucher_male'] = df['doctor_voucher'] * df['genderpatient']
result = run_regression(
    df, PRIMARY_OUTCOME,
    ['patient_voucher', 'doctor_voucher', 'patient_voucher_male', 'doctor_voucher_male', 'genderpatient'],
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/het/gender',
    spec_tree_path='robustness/heterogeneity.md',
    sample_desc='Interaction with patient gender'
)
if result:
    results.append(result)
    print(f"Gender heterogeneity: coef={result['coefficient']:.4f}")

# 7d. Interaction with number of symptoms
df['patient_voucher_symptoms'] = df['patient_voucher'] * df['num_symptoms']
df['doctor_voucher_symptoms'] = df['doctor_voucher'] * df['num_symptoms']
result = run_regression(
    df, PRIMARY_OUTCOME,
    ['patient_voucher', 'doctor_voucher', 'patient_voucher_symptoms', 'doctor_voucher_symptoms', 'num_symptoms'],
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/het/num_symptoms',
    spec_tree_path='robustness/heterogeneity.md',
    sample_desc='Interaction with number of symptoms'
)
if result:
    results.append(result)
    print(f"Symptoms heterogeneity: coef={result['coefficient']:.4f}")

# 7e. Interaction with patient info
df['patient_voucher_x_info'] = df['patient_voucher'] * df['patient_info']
df['doctor_voucher_x_info'] = df['doctor_voucher'] * df['patient_info']
result = run_regression(
    df, PRIMARY_OUTCOME,
    ['patient_voucher', 'doctor_voucher', 'patient_info', 'patient_voucher_x_info', 'doctor_voucher_x_info'],
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/het/patient_info',
    spec_tree_path='robustness/heterogeneity.md',
    sample_desc='Interaction with patient information'
)
if result:
    results.append(result)
    print(f"Patient info heterogeneity: coef={result['coefficient']:.4f}")

# 7f. Interaction with education
df['patient_voucher_educ'] = df['patient_voucher'] * df['prischoolorless'].fillna(0)
df['doctor_voucher_educ'] = df['doctor_voucher'] * df['prischoolorless'].fillna(0)
result = run_regression(
    df, PRIMARY_OUTCOME,
    ['patient_voucher', 'doctor_voucher', 'patient_voucher_educ', 'doctor_voucher_educ', 'prischoolorless'],
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/het/education',
    spec_tree_path='robustness/heterogeneity.md',
    sample_desc='Interaction with education'
)
if result:
    results.append(result)
    print(f"Education heterogeneity: coef={result['coefficient']:.4f}")

# 7g. Interaction with days ill
df['patient_voucher_days'] = df['patient_voucher'] * df['daysillness99'].fillna(0)
df['doctor_voucher_days'] = df['doctor_voucher'] * df['daysillness99'].fillna(0)
result = run_regression(
    df, PRIMARY_OUTCOME,
    ['patient_voucher', 'doctor_voucher', 'patient_voucher_days', 'doctor_voucher_days', 'daysillness99'],
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/het/days_ill',
    spec_tree_path='robustness/heterogeneity.md',
    sample_desc='Interaction with days ill'
)
if result:
    results.append(result)
    print(f"Days ill heterogeneity: coef={result['coefficient']:.4f}")

# 7h. Interaction with ethnicity (Bambara)
df['patient_voucher_bambara'] = df['patient_voucher'] * df['ethnic_bambara'].fillna(0)
df['doctor_voucher_bambara'] = df['doctor_voucher'] * df['ethnic_bambara'].fillna(0)
result = run_regression(
    df, PRIMARY_OUTCOME,
    ['patient_voucher', 'doctor_voucher', 'patient_voucher_bambara', 'doctor_voucher_bambara', 'ethnic_bambara'],
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/het/ethnicity',
    spec_tree_path='robustness/heterogeneity.md',
    sample_desc='Interaction with Bambara ethnicity'
)
if result:
    results.append(result)
    print(f"Ethnicity heterogeneity: coef={result['coefficient']:.4f}")

# -----------------------------------------------------------------------------
# 8. FUNCTIONAL FORM VARIATIONS
# -----------------------------------------------------------------------------
print("\n--- Functional Form Variations ---")

# For outcomes that can be transformed
# 8a. Predicted malaria prob as outcome (continuous)
result = run_regression(
    df, 'pred_mal_pos', PRIMARY_TREATMENTS,
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/funcform/pred_mal_continuous',
    spec_tree_path='robustness/functional_form.md',
    sample_desc='Predicted malaria (continuous outcome)'
)
if result:
    results.append(result)
    print(f"Pred mal continuous: coef={result['coefficient']:.4f}")

# 8b. Expected match (continuous)
result = run_regression(
    df, 'expected_mal_match_any', PRIMARY_TREATMENTS,
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/funcform/expected_match',
    spec_tree_path='robustness/functional_form.md',
    sample_desc='Expected match (continuous)'
)
if result:
    results.append(result)
    print(f"Expected match: coef={result['coefficient']:.4f}")

# -----------------------------------------------------------------------------
# 9. PLACEBO TESTS
# -----------------------------------------------------------------------------
print("\n--- Placebo Tests ---")

# 9a. Outcome should not be affected: predicted malaria probability
# (Treatment assignment should be random and not affect predicted risk)
result = run_regression(
    df, 'pred_mal_pos', PRIMARY_TREATMENTS,
    controls=None, fe=None, cluster='clinic_day',
    spec_id='robust/placebo/pred_mal_no_effect',
    spec_tree_path='robustness/placebo_tests.md',
    sample_desc='Placebo: voucher should not affect predicted malaria'
)
if result:
    results.append(result)
    print(f"Placebo (pred mal): coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# 9b. Patient age as outcome (should not be affected by treatment)
result = run_regression(
    df, 'agepatient', PRIMARY_TREATMENTS,
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/placebo/age_no_effect',
    spec_tree_path='robustness/placebo_tests.md',
    sample_desc='Placebo: voucher should not affect patient age'
)
if result:
    results.append(result)
    print(f"Placebo (age): coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# 9c. Days ill as outcome (should not be affected by treatment)
result = run_regression(
    df, 'daysillness99', PRIMARY_TREATMENTS,
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/placebo/daysill_no_effect',
    spec_tree_path='robustness/placebo_tests.md',
    sample_desc='Placebo: voucher should not affect days ill'
)
if result:
    results.append(result)
    print(f"Placebo (days ill): coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# 9d. Number of symptoms as outcome
result = run_regression(
    df, 'num_symptoms', PRIMARY_TREATMENTS,
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/placebo/symptoms_no_effect',
    spec_tree_path='robustness/placebo_tests.md',
    sample_desc='Placebo: voucher should not affect symptoms'
)
if result:
    results.append(result)
    print(f"Placebo (symptoms): coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# -----------------------------------------------------------------------------
# 10. ADDITIONAL ROBUSTNESS
# -----------------------------------------------------------------------------
print("\n--- Additional Robustness Specifications ---")

# 10a. Drop extreme predicted malaria probability
df_no_extreme = df[(df['pred_mal_pos'] > 0.05) & (df['pred_mal_pos'] < 0.95)].copy()
result = run_regression(
    df_no_extreme, PRIMARY_OUTCOME, PRIMARY_TREATMENTS,
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/sample/moderate_risk',
    spec_tree_path='robustness/sample_restrictions.md',
    sample_desc='Moderate predicted malaria risk (5-95%)'
)
if result:
    results.append(result)
    print(f"Moderate risk: coef={result['coefficient']:.4f}")

# 10b. Only patients who responded themselves
df_self = df[df['respondent'] == 1].copy()
result = run_regression(
    df_self, PRIMARY_OUTCOME, PRIMARY_TREATMENTS,
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/sample/patient_respondent',
    spec_tree_path='robustness/sample_restrictions.md',
    sample_desc='Patient responded themselves'
)
if result:
    results.append(result)
    print(f"Patient respondent: coef={result['coefficient']:.4f}")

# 10c. Only literate respondents
df_lit = df[df['readwrite_fluent_french'] == 1].copy()
result = run_regression(
    df_lit, PRIMARY_OUTCOME, PRIMARY_TREATMENTS,
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/sample/literate',
    spec_tree_path='robustness/sample_restrictions.md',
    sample_desc='Literate respondents only'
)
if result:
    results.append(result)
    print(f"Literate: coef={result['coefficient']:.4f}")

# 10d. Only illiterate respondents
df_illit = df[df['readwrite_fluent_french'] == 0].copy()
result = run_regression(
    df_illit, PRIMARY_OUTCOME, PRIMARY_TREATMENTS,
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/sample/illiterate',
    spec_tree_path='robustness/sample_restrictions.md',
    sample_desc='Illiterate respondents only'
)
if result:
    results.append(result)
    print(f"Illiterate: coef={result['coefficient']:.4f}")

# 10e. Severe malaria treatment outcome with full controls
result = run_regression(
    df, 'treat_severe_mal', PRIMARY_TREATMENTS,
    controls=date_fe_valid + get_valid_controls(df, FULL_CONTROLS),
    fe=None, cluster='clinic_day',
    spec_id='robust/outcome/severe_full_controls',
    spec_tree_path='robustness/outcome_variations.md',
    sample_desc='Severe malaria treatment, full controls'
)
if result:
    results.append(result)
    print(f"Severe mal full controls: coef={result['coefficient']:.4f}")

# 10f. Prescribed outcome with full controls
result = run_regression(
    df, 'RXtreat_sev_simple_mal', PRIMARY_TREATMENTS,
    controls=date_fe_valid + get_valid_controls(df, FULL_CONTROLS),
    fe=None, cluster='clinic_day',
    spec_id='robust/outcome/prescribed_full_controls',
    spec_tree_path='robustness/outcome_variations.md',
    sample_desc='Prescribed treatment, full controls'
)
if result:
    results.append(result)
    print(f"Prescribed full controls: coef={result['coefficient']:.4f}")

# 10g. Voucher usage with full controls
result = run_regression(
    df, 'used_vouchers_admin', PRIMARY_TREATMENTS,
    controls=date_fe_valid + get_valid_controls(df, FULL_CONTROLS),
    fe=None, cluster='clinic_day',
    spec_id='robust/outcome/voucher_usage_full_controls',
    spec_tree_path='robustness/outcome_variations.md',
    sample_desc='Voucher usage, full controls'
)
if result:
    results.append(result)
    print(f"Voucher usage full controls: coef={result['coefficient']:.4f}")

# 10h. Winsorize outcome at 1%
# For continuous outcomes
df_w = df.copy()
# Note: binary outcomes can't really be winsorized, skip

# 10i. Compare with doctor info (infotreat)
result = run_regression(
    df, PRIMARY_OUTCOME, ['patient_voucher', 'doctor_voucher', 'infotreat'],
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/control/include_doc_info',
    spec_tree_path='robustness/control_variations.md',
    sample_desc='Include doctor information treatment'
)
if result:
    results.append(result)
    print(f"Include doc info: coef={result['coefficient']:.4f}")

# -----------------------------------------------------------------------------
# 11. ADDITIONAL SAMPLE RESTRICTIONS TO REACH 50+ SPECS
# -----------------------------------------------------------------------------
print("\n--- Additional Sample Restrictions ---")

# 11a. Young patients (under median age)
median_age = df['agepatient'].median()
df_young = df[df['agepatient'] < median_age].copy()
result = run_regression(
    df_young, PRIMARY_OUTCOME, PRIMARY_TREATMENTS,
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/sample/young_below_median',
    spec_tree_path='robustness/sample_restrictions.md',
    sample_desc=f'Young patients (age < {median_age:.0f})'
)
if result:
    results.append(result)
    print(f"Young (below median): coef={result['coefficient']:.4f}")

# 11b. Older patients (above median age)
df_old = df[df['agepatient'] >= median_age].copy()
result = run_regression(
    df_old, PRIMARY_OUTCOME, PRIMARY_TREATMENTS,
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/sample/old_above_median',
    spec_tree_path='robustness/sample_restrictions.md',
    sample_desc=f'Older patients (age >= {median_age:.0f})'
)
if result:
    results.append(result)
    print(f"Old (above median): coef={result['coefficient']:.4f}")

# 11c. Bambara ethnicity only
df_bambara = df[df['ethnic_bambara'] == 1].copy()
result = run_regression(
    df_bambara, PRIMARY_OUTCOME, PRIMARY_TREATMENTS,
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/sample/bambara',
    spec_tree_path='robustness/sample_restrictions.md',
    sample_desc='Bambara ethnicity only'
)
if result:
    results.append(result)
    print(f"Bambara: coef={result['coefficient']:.4f}")

# 11d. Non-Bambara
df_nonbambara = df[df['ethnic_bambara'] == 0].copy()
result = run_regression(
    df_nonbambara, PRIMARY_OUTCOME, PRIMARY_TREATMENTS,
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/sample/non_bambara',
    spec_tree_path='robustness/sample_restrictions.md',
    sample_desc='Non-Bambara ethnicity'
)
if result:
    results.append(result)
    print(f"Non-Bambara: coef={result['coefficient']:.4f}")

# 11e. Higher education
df_educ = df[df['prischoolorless'] == 0].copy()
result = run_regression(
    df_educ, PRIMARY_OUTCOME, PRIMARY_TREATMENTS,
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/sample/higher_education',
    spec_tree_path='robustness/sample_restrictions.md',
    sample_desc='Higher than primary education'
)
if result:
    results.append(result)
    print(f"Higher education: coef={result['coefficient']:.4f}")

# 11f. Primary education or less
df_loeduc = df[df['prischoolorless'] == 1].copy()
result = run_regression(
    df_loeduc, PRIMARY_OUTCOME, PRIMARY_TREATMENTS,
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/sample/primary_or_less',
    spec_tree_path='robustness/sample_restrictions.md',
    sample_desc='Primary education or less'
)
if result:
    results.append(result)
    print(f"Primary or less: coef={result['coefficient']:.4f}")

# 11g. Short illness duration (below median)
median_days = df['daysillness99'].median()
df_short = df[df['daysillness99'] < median_days].copy()
result = run_regression(
    df_short, PRIMARY_OUTCOME, PRIMARY_TREATMENTS,
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/sample/short_illness',
    spec_tree_path='robustness/sample_restrictions.md',
    sample_desc=f'Short illness (<{median_days:.0f} days)'
)
if result:
    results.append(result)
    print(f"Short illness: coef={result['coefficient']:.4f}")

# 11h. Longer illness duration (above median)
df_long = df[df['daysillness99'] >= median_days].copy()
result = run_regression(
    df_long, PRIMARY_OUTCOME, PRIMARY_TREATMENTS,
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/sample/long_illness',
    spec_tree_path='robustness/sample_restrictions.md',
    sample_desc=f'Longer illness (>={median_days:.0f} days)'
)
if result:
    results.append(result)
    print(f"Long illness: coef={result['coefficient']:.4f}")

# 11i. Only those with no patient info
df_noinfo = df[df['patient_info'] == 0].copy()
result = run_regression(
    df_noinfo, PRIMARY_OUTCOME, PRIMARY_TREATMENTS,
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/sample/no_patient_info',
    spec_tree_path='robustness/sample_restrictions.md',
    sample_desc='No patient information treatment'
)
if result:
    results.append(result)
    print(f"No patient info: coef={result['coefficient']:.4f}")

# 11j. Only those with patient info
df_info = df[df['patient_info'] == 1].copy()
result = run_regression(
    df_info, PRIMARY_OUTCOME, PRIMARY_TREATMENTS,
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/sample/with_patient_info',
    spec_tree_path='robustness/sample_restrictions.md',
    sample_desc='With patient information treatment'
)
if result:
    results.append(result)
    print(f"With patient info: coef={result['coefficient']:.4f}")

# -----------------------------------------------------------------------------
# 12. ADDITIONAL OUTCOME VARIATIONS WITH CONTROLS
# -----------------------------------------------------------------------------
print("\n--- Additional Outcome Specifications ---")

# Expected match components
for outcome in ['expected_mal_match_any_pos', 'expected_mal_match_any_neg',
                'RXexpected_mal_match_any_pos', 'RXexpected_mal_match_any_neg']:
    if outcome in df.columns:
        result = run_regression(
            df, outcome, PRIMARY_TREATMENTS,
            controls=date_fe_valid, fe=None, cluster='clinic_day',
            spec_id=f'robust/outcome/{outcome}',
            spec_tree_path='robustness/outcome_variations.md',
            sample_desc=f'Outcome: {outcome}'
        )
        if result:
            results.append(result)
            print(f"Outcome {outcome}: coef={result['coefficient']:.4f}")

# -----------------------------------------------------------------------------
# 13. INTERACTION WITH CONTINUOUS VARIABLES
# -----------------------------------------------------------------------------
print("\n--- Additional Heterogeneity Specifications ---")

# Interaction with continuous age
df['patient_voucher_age'] = df['patient_voucher'] * df['agepatient'].fillna(0)
df['doctor_voucher_age'] = df['doctor_voucher'] * df['agepatient'].fillna(0)
result = run_regression(
    df, PRIMARY_OUTCOME,
    ['patient_voucher', 'doctor_voucher', 'patient_voucher_age', 'doctor_voucher_age', 'agepatient'],
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/het/continuous_age',
    spec_tree_path='robustness/heterogeneity.md',
    sample_desc='Interaction with continuous age'
)
if result:
    results.append(result)
    print(f"Age continuous heterogeneity: coef={result['coefficient']:.4f}")

# Interaction with predicted malaria (continuous)
df['patient_voucher_predmal'] = df['patient_voucher'] * df['pred_mal_pos'].fillna(0)
df['doctor_voucher_predmal'] = df['doctor_voucher'] * df['pred_mal_pos'].fillna(0)
result = run_regression(
    df, PRIMARY_OUTCOME,
    ['patient_voucher', 'doctor_voucher', 'patient_voucher_predmal', 'doctor_voucher_predmal', 'pred_mal_pos'],
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/het/pred_malaria_continuous',
    spec_tree_path='robustness/heterogeneity.md',
    sample_desc='Interaction with continuous predicted malaria'
)
if result:
    results.append(result)
    print(f"Pred malaria continuous heterogeneity: coef={result['coefficient']:.4f}")

# Interaction with French speaking
df['patient_voucher_french'] = df['patient_voucher'] * df['speak_french'].fillna(0)
df['doctor_voucher_french'] = df['doctor_voucher'] * df['speak_french'].fillna(0)
result = run_regression(
    df, PRIMARY_OUTCOME,
    ['patient_voucher', 'doctor_voucher', 'patient_voucher_french', 'doctor_voucher_french', 'speak_french'],
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/het/french_speaking',
    spec_tree_path='robustness/heterogeneity.md',
    sample_desc='Interaction with French speaking'
)
if result:
    results.append(result)
    print(f"French speaking heterogeneity: coef={result['coefficient']:.4f}")

# Interaction with literacy
df['patient_voucher_lit'] = df['patient_voucher'] * df['readwrite_fluent_french'].fillna(0)
df['doctor_voucher_lit'] = df['doctor_voucher'] * df['readwrite_fluent_french'].fillna(0)
result = run_regression(
    df, PRIMARY_OUTCOME,
    ['patient_voucher', 'doctor_voucher', 'patient_voucher_lit', 'doctor_voucher_lit', 'readwrite_fluent_french'],
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/het/literacy',
    spec_tree_path='robustness/heterogeneity.md',
    sample_desc='Interaction with literacy'
)
if result:
    results.append(result)
    print(f"Literacy heterogeneity: coef={result['coefficient']:.4f}")

# -----------------------------------------------------------------------------
# 14. COMBINED HETEROGENEITY
# -----------------------------------------------------------------------------
print("\n--- Combined Heterogeneity ---")

# Triple interaction: voucher x malaria risk x patient info
df['pv_high_info'] = df['patient_voucher_high'] * df['patient_info']
df['pv_low_info'] = df['patient_voucher_low'] * df['patient_info']
df['dv_high_info'] = df['doctor_voucher_high'] * df['patient_info']
df['dv_low_info'] = df['doctor_voucher_low'] * df['patient_info']

result = run_regression(
    df, PRIMARY_OUTCOME,
    ['patient_voucher_high', 'doctor_voucher_high', 'patient_voucher_low', 'doctor_voucher_low',
     'above_med_pos', 'patient_info', 'pv_high_info', 'dv_high_info', 'pv_low_info', 'dv_low_info'],
    controls=date_fe_valid, fe=None, cluster='clinic_day',
    spec_id='robust/het/risk_x_info',
    spec_tree_path='robustness/heterogeneity.md',
    sample_desc='Triple interaction: risk x info x voucher'
)
if result:
    results.append(result)
    print(f"Risk x Info triple diff: coef={result['coefficient']:.4f}")

# =============================================================================
# SAVE RESULTS
# =============================================================================

print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Convert to DataFrame
results_df = pd.DataFrame(results)

print(f"\nTotal specifications run: {len(results_df)}")

# Save to CSV
output_path = f'{OUTPUT_DIR}/specification_results.csv'
results_df.to_csv(output_path, index=False)
print(f"Results saved to: {output_path}")

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

# Focus on primary outcome
primary_results = results_df[results_df['outcome_var'] == PRIMARY_OUTCOME]

print(f"\nPrimary outcome ({PRIMARY_OUTCOME}):")
print(f"  N specifications: {len(primary_results)}")
print(f"  Positive coefficients: {(primary_results['coefficient'] > 0).sum()} ({100*(primary_results['coefficient'] > 0).mean():.1f}%)")
print(f"  Significant at 5%: {(primary_results['p_value'] < 0.05).sum()} ({100*(primary_results['p_value'] < 0.05).mean():.1f}%)")
print(f"  Significant at 1%: {(primary_results['p_value'] < 0.01).sum()} ({100*(primary_results['p_value'] < 0.01).mean():.1f}%)")
print(f"  Median coefficient: {primary_results['coefficient'].median():.4f}")
print(f"  Mean coefficient: {primary_results['coefficient'].mean():.4f}")
print(f"  Range: [{primary_results['coefficient'].min():.4f}, {primary_results['coefficient'].max():.4f}]")

# All specifications
print(f"\nAll outcomes:")
print(f"  N specifications: {len(results_df)}")
print(f"  Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
print(f"  Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
print(f"  Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")

print("\n" + "="*60)
print("SPECIFICATION SEARCH COMPLETE")
print("="*60)
