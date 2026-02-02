"""
Specification Search for Paper 174501-V1:
"Interaction, Stereotypes and Performance: Evidence from South Africa"
Corno, La Ferrara, Burns (AER: Insights 2022)

This script runs 50+ specifications testing the effect of mixed-race roommate assignment
on stereotypes (IAT), academic performance, and social outcomes.

Main hypothesis: Living with a roommate of a different race reduces stereotypes
and affects academic performance.

Method: Cross-sectional OLS with fixed effects (randomized experiment)
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# ============================================================================
# Data Loading and Preparation
# ============================================================================

BASE_PATH = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/174501-V1/ForJournals_a/'

# Load the balanced dataset
df = pd.read_stata(BASE_PATH + 'Data/Clean/uctdata_balanced.dta')

# Convert categorical columns to numeric codes for FE
df['round_num'] = (df['round'] == 'Follow-up').astype(int)

# Convert categorical Female to numeric
if df['Female'].dtype.name == 'category':
    df['Female_num'] = (df['Female'] == 'female').astype(float)
else:
    df['Female_num'] = df['Female']

# Convert regprogram to numeric codes for FE
if df['regprogram'].dtype == 'object':
    df['regprogram_code'] = pd.factorize(df['regprogram'])[0]
else:
    df['regprogram_code'] = df['regprogram']

# Ensure Res_base is numeric
df['Res_base'] = pd.to_numeric(df['Res_base'], errors='coerce')

# Create lagged variables for IAT outcomes (baseline values)
df_base = df[df['round'] == 'Baseline'][['individual', 'DscoreraceIAT', 'DscoreacaIAT', 'PCAattitude']].copy()
df_base.columns = ['individual', 'L_DscoreraceIAT', 'L_DscoreacaIAT', 'L_PCAattitude']

# Merge lagged values to follow-up observations
df = df.merge(df_base, on='individual', how='left')

# Filter to follow-up round for main analysis
df_r2 = df[df['round'] == 'Follow-up'].copy()

# Define control variable sets
controls_subsample = ['Female_num', 'Falseuct2012', 'missfalse', 'Foreign', 'foreign_missing',
                      'privateschool_nomiss', 'privateschool_miss', 'durpcabas_nomiss',
                      'durpcabas_miss', 'consbas_nomiss', 'consbas_miss']

rocontrols = ['roFalseuct2012', 'missrofalse', 'roForeign_bas', 'roforeign_missingbas',
              'roprivschool_nomiss', 'roprivschool_miss', 'rodurpcabas_nomiss',
              'rodurpcabas_miss', 'roconsbas_nomiss']

controls_full = ['white', 'coloured', 'Else', 'Female_num', 'Falseuct2012', 'missfalse',
                 'Foreign', 'foreign_missing', 'privateschool_nomiss', 'privateschool_miss',
                 'durpcabas_nomiss', 'durpcabas_miss', 'consbas_nomiss', 'consbas_miss']

# ============================================================================
# Helper Functions
# ============================================================================

def run_ols_regression(df, outcome, treatment, controls=None,
                      fe_vars=None, sample_condition=None):
    """
    Run OLS regression with various options using statsmodels.
    Returns dict with coefficient info.
    """
    data = df.copy()

    # Apply sample condition
    if sample_condition is not None:
        data = data[sample_condition].copy()

    # Drop missing values for outcome and treatment
    data = data.dropna(subset=[outcome, treatment])

    if len(data) < 20:
        return None

    # Filter valid controls (have variation and are numeric)
    valid_controls = []
    if controls:
        for c in controls:
            if c in data.columns:
                # Ensure numeric
                if not pd.api.types.is_numeric_dtype(data[c]):
                    continue
                if data[c].notna().sum() > 10 and data[c].std() > 0:
                    valid_controls.append(c)

    # Build design matrix
    X_vars = [treatment] + valid_controls

    # Add fixed effects as dummies
    if fe_vars:
        for fe in fe_vars:
            if fe in data.columns:
                # Ensure fe column is numeric
                if not pd.api.types.is_numeric_dtype(data[fe]):
                    data[fe] = pd.factorize(data[fe])[0]
                dummies = pd.get_dummies(data[fe].astype(int), prefix=f'fe_{fe}', drop_first=True)
                for col in dummies.columns:
                    data[col] = dummies[col].astype(float)
                    X_vars.append(col)

    # Drop rows with any missing values in X or y
    final_vars = [outcome] + [v for v in X_vars if v in data.columns]
    data = data.dropna(subset=final_vars)

    if len(data) < 20:
        return None

    try:
        # Build X matrix - ensure all numeric
        X = data[[v for v in X_vars if v in data.columns]].copy()
        X = X.astype(float)
        X = sm.add_constant(X)
        y = data[outcome].astype(float)

        # Run OLS
        model = sm.OLS(y, X).fit(cov_type='HC1')

        if treatment not in model.params.index:
            return None

        coef = model.params[treatment]
        se = model.bse[treatment]
        t_stat = model.tvalues[treatment]
        p_val = model.pvalues[treatment]
        ci = model.conf_int()
        ci_lower = ci.loc[treatment, 0]
        ci_upper = ci.loc[treatment, 1]
        n_obs = int(model.nobs)
        r2 = model.rsquared

        return {
            'coefficient': coef,
            'std_error': se,
            't_stat': t_stat,
            'p_value': p_val,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_obs': n_obs,
            'r_squared': r2
        }

    except Exception as e:
        print(f"Error in regression: {e}")
        return None

def create_result_row(spec_id, spec_tree_path, outcome_var, treatment_var,
                     reg_result, sample_desc, fe_desc, controls_desc,
                     cluster_var, model_type='OLS'):
    """Create a standardized result row for the specification search."""
    if reg_result is None:
        return None

    return {
        'paper_id': '174501-V1',
        'journal': 'AER: Insights',
        'paper_title': 'Interaction, Stereotypes and Performance: Evidence from South Africa',
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'outcome_var': outcome_var,
        'treatment_var': treatment_var,
        'coefficient': reg_result['coefficient'],
        'std_error': reg_result['std_error'],
        't_stat': reg_result['t_stat'],
        'p_value': reg_result['p_value'],
        'ci_lower': reg_result['ci_lower'],
        'ci_upper': reg_result['ci_upper'],
        'n_obs': reg_result['n_obs'],
        'r_squared': reg_result['r_squared'],
        'coefficient_vector_json': json.dumps({
            'treatment': {
                'var': treatment_var,
                'coef': float(reg_result['coefficient']),
                'se': float(reg_result['std_error']),
                'pval': float(reg_result['p_value'])
            }
        }),
        'sample_desc': sample_desc,
        'fixed_effects': fe_desc,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': 'spec_search_174501.py'
    }

# ============================================================================
# Run Specification Search
# ============================================================================

results = []

# ============================================================================
# 1. BASELINE SPECIFICATIONS (Table 3-5 main results)
# ============================================================================
print("Running baseline specifications...")

# Table 3: IAT outcomes with lagged dependent variable - WHITES and BLACKS
for outcome, outcome_name in [('DscoreraceIAT', 'Race IAT'), ('DscoreacaIAT', 'Academic IAT')]:
    lagged = 'L_' + outcome

    # Whites
    reg = run_ols_regression(
        df_r2, outcome, 'mixracebas',
        controls=[lagged] + controls_subsample + rocontrols,
        fe_vars=['Res_base'],
        sample_condition=df_r2['white'] == 1
    )
    row = create_result_row(
        f'baseline/table3/{outcome_name.replace(" ", "_").lower()}_white',
        'methods/cross_sectional_ols.md#baseline',
        outcome, 'mixracebas', reg,
        'White students, follow-up',
        'Residence FE', 'Own + roommate controls, lagged DV',
        'roomnum_base', 'OLS'
    )
    if row:
        results.append(row)

    # Blacks
    reg = run_ols_regression(
        df_r2, outcome, 'mixracebas',
        controls=[lagged] + controls_subsample + rocontrols,
        fe_vars=['Res_base'],
        sample_condition=df_r2['black'] == 1
    )
    row = create_result_row(
        f'baseline/table3/{outcome_name.replace(" ", "_").lower()}_black',
        'methods/cross_sectional_ols.md#baseline',
        outcome, 'mixracebas', reg,
        'Black students, follow-up',
        'Residence FE', 'Own + roommate controls, lagged DV',
        'roomnum_base', 'OLS'
    )
    if row:
        results.append(row)

# Table 4: Academic performance outcomes
for outcome in ['GPA', 'examspassed', 'continue', 'PCAperf']:

    # Whites
    reg = run_ols_regression(
        df_r2, outcome, 'mixracebas',
        controls=controls_subsample + rocontrols,
        fe_vars=['Res_base', 'regprogram_code'],
        sample_condition=df_r2['white'] == 1
    )
    row = create_result_row(
        f'baseline/table4/{outcome.lower()}_white',
        'methods/cross_sectional_ols.md#baseline',
        outcome, 'mixracebas', reg,
        'White students, follow-up',
        'Residence + Program FE', 'Own + roommate controls',
        'roomnum_base', 'OLS'
    )
    if row:
        results.append(row)

    # Blacks
    reg = run_ols_regression(
        df_r2, outcome, 'mixracebas',
        controls=controls_subsample + rocontrols,
        fe_vars=['Res_base', 'regprogram_code'],
        sample_condition=df_r2['black'] == 1
    )
    row = create_result_row(
        f'baseline/table4/{outcome.lower()}_black',
        'methods/cross_sectional_ols.md#baseline',
        outcome, 'mixracebas', reg,
        'Black students, follow-up',
        'Residence + Program FE', 'Own + roommate controls',
        'roomnum_base', 'OLS'
    )
    if row:
        results.append(row)

    # Full sample
    reg = run_ols_regression(
        df_r2, outcome, 'mixracebas',
        controls=controls_full + rocontrols,
        fe_vars=['Res_base', 'regprogram_code']
    )
    row = create_result_row(
        f'baseline/table4/{outcome.lower()}_full',
        'methods/cross_sectional_ols.md#baseline',
        outcome, 'mixracebas', reg,
        'Full sample, follow-up',
        'Residence + Program FE', 'Full + roommate controls',
        'roomnum_base', 'OLS'
    )
    if row:
        results.append(row)

# Table 5: Social outcomes
for outcome in ['PCAfriend', 'PCAattitude', 'PCAcomm', 'PCAsocial']:

    # Whites
    reg = run_ols_regression(
        df_r2, outcome, 'mixracebas',
        controls=controls_subsample + rocontrols,
        fe_vars=['Res_base'],
        sample_condition=df_r2['white'] == 1
    )
    row = create_result_row(
        f'baseline/table5/{outcome.lower()}_white',
        'methods/cross_sectional_ols.md#baseline',
        outcome, 'mixracebas', reg,
        'White students, follow-up',
        'Residence FE', 'Own + roommate controls',
        'roomnum_base', 'OLS'
    )
    if row:
        results.append(row)

    # Blacks
    reg = run_ols_regression(
        df_r2, outcome, 'mixracebas',
        controls=controls_subsample + rocontrols,
        fe_vars=['Res_base'],
        sample_condition=df_r2['black'] == 1
    )
    row = create_result_row(
        f'baseline/table5/{outcome.lower()}_black',
        'methods/cross_sectional_ols.md#baseline',
        outcome, 'mixracebas', reg,
        'Black students, follow-up',
        'Residence FE', 'Own + roommate controls',
        'roomnum_base', 'OLS'
    )
    if row:
        results.append(row)

    # Full sample
    reg = run_ols_regression(
        df_r2, outcome, 'mixracebas',
        controls=controls_full + rocontrols,
        fe_vars=['Res_base']
    )
    row = create_result_row(
        f'baseline/table5/{outcome.lower()}_full',
        'methods/cross_sectional_ols.md#baseline',
        outcome, 'mixracebas', reg,
        'Full sample, follow-up',
        'Residence FE', 'Full + roommate controls',
        'roomnum_base', 'OLS'
    )
    if row:
        results.append(row)

print(f"  Baseline specs: {len(results)}")

# ============================================================================
# 2. CONTROL VARIATIONS
# ============================================================================
print("Running control variations...")
n_before = len(results)

# Focus on key outcomes
key_specs = [
    ('DscoreraceIAT', df_r2['white'] == 1, 'white', ['L_DscoreraceIAT'] + controls_subsample + rocontrols, ['Res_base']),
    ('GPA', df_r2['black'] == 1, 'black', controls_subsample + rocontrols, ['Res_base', 'regprogram_code']),
    ('DscoreraceIAT', df_r2['black'] == 1, 'black', ['L_DscoreraceIAT'] + controls_subsample + rocontrols, ['Res_base']),
    ('GPA', df_r2['white'] == 1, 'white', controls_subsample + rocontrols, ['Res_base', 'regprogram_code']),
]

for outcome, sample_cond, sample_name, full_controls, fe_vars in key_specs:
    # No controls
    reg = run_ols_regression(
        df_r2, outcome, 'mixracebas',
        controls=None,
        fe_vars=fe_vars,
        sample_condition=sample_cond
    )
    row = create_result_row(
        f'robust/control/none_{outcome}_{sample_name}',
        'robustness/control_progression.md',
        outcome, 'mixracebas', reg,
        f'{sample_name.capitalize()} students, no controls',
        ' + '.join(fe_vars), 'None',
        'roomnum_base', 'OLS'
    )
    if row:
        results.append(row)

    # Only own controls (no roommate)
    own_controls = [c for c in full_controls if not c.startswith('ro')]
    reg = run_ols_regression(
        df_r2, outcome, 'mixracebas',
        controls=own_controls,
        fe_vars=fe_vars,
        sample_condition=sample_cond
    )
    row = create_result_row(
        f'robust/control/own_only_{outcome}_{sample_name}',
        'robustness/control_progression.md',
        outcome, 'mixracebas', reg,
        f'{sample_name.capitalize()} students, own controls only',
        ' + '.join(fe_vars), 'Own controls only',
        'roomnum_base', 'OLS'
    )
    if row:
        results.append(row)

    # Leave-one-out for key controls
    key_controls_to_drop = ['Female_num', 'Falseuct2012', 'privateschool_nomiss', 'durpcabas_nomiss', 'consbas_nomiss']
    for ctrl in key_controls_to_drop:
        if ctrl in full_controls:
            remaining = [c for c in full_controls if c != ctrl]
            reg = run_ols_regression(
                df_r2, outcome, 'mixracebas',
                controls=remaining,
                fe_vars=fe_vars,
                sample_condition=sample_cond
            )
            row = create_result_row(
                f'robust/control/drop_{ctrl}_{outcome}_{sample_name}',
                'robustness/leave_one_out.md',
                outcome, 'mixracebas', reg,
                f'{sample_name.capitalize()} students, drop {ctrl}',
                ' + '.join(fe_vars), f'All except {ctrl}',
                'roomnum_base', 'OLS'
            )
            if row:
                results.append(row)

print(f"  Control variations: {len(results) - n_before}")

# ============================================================================
# 3. INFERENCE VARIATIONS
# ============================================================================
print("Running inference variations...")
n_before = len(results)

# Standard errors variations - using robust SE (HC0, HC1, HC2, HC3)
for outcome, sample_cond, sample_name in [
    ('DscoreraceIAT', df_r2['white'] == 1, 'white'),
    ('GPA', df_r2['black'] == 1, 'black'),
]:
    if outcome == 'DscoreraceIAT':
        ctrls = ['L_DscoreraceIAT'] + controls_subsample + rocontrols
    else:
        ctrls = controls_subsample + rocontrols

    for cov_type in ['HC0', 'HC2', 'HC3']:
        data = df_r2[sample_cond].copy()
        data = data.dropna(subset=[outcome, 'mixracebas'])

        if len(data) < 20:
            continue

        # Build design matrix
        valid_controls = [c for c in ctrls if c in data.columns and pd.api.types.is_numeric_dtype(data[c]) and data[c].notna().sum() > 10 and data[c].std() > 0]
        X_vars = ['mixracebas'] + valid_controls

        # Add FE
        for fe in ['Res_base']:
            if fe in data.columns:
                dummies = pd.get_dummies(data[fe].astype(int), prefix=f'fe_{fe}', drop_first=True)
                for col in dummies.columns:
                    data[col] = dummies[col].astype(float)
                    X_vars.append(col)

        data = data.dropna(subset=[outcome] + [v for v in X_vars if v in data.columns])

        if len(data) < 20:
            continue

        try:
            X = data[[v for v in X_vars if v in data.columns]].copy().astype(float)
            X = sm.add_constant(X)
            y = data[outcome].astype(float)

            model = sm.OLS(y, X).fit(cov_type=cov_type)

            reg_result = {
                'coefficient': model.params['mixracebas'],
                'std_error': model.bse['mixracebas'],
                't_stat': model.tvalues['mixracebas'],
                'p_value': model.pvalues['mixracebas'],
                'ci_lower': model.conf_int().loc['mixracebas', 0],
                'ci_upper': model.conf_int().loc['mixracebas', 1],
                'n_obs': int(model.nobs),
                'r_squared': model.rsquared
            }

            row = create_result_row(
                f'robust/inference/{cov_type}_{outcome}_{sample_name}',
                'robustness/clustering_variations.md',
                outcome, 'mixracebas', reg_result,
                f'{sample_name.capitalize()} students, {cov_type} SE',
                'Residence FE', 'Full controls',
                f'{cov_type}', 'OLS'
            )
            if row:
                results.append(row)
        except Exception as e:
            print(f"Error: {e}")

print(f"  Inference variations: {len(results) - n_before}")

# ============================================================================
# 4. ALTERNATIVE OUTCOMES
# ============================================================================
print("Running alternative outcomes...")
n_before = len(results)

# Second year outcomes
for outcome in ['GPA2013', 'examspassed2013', 'continue2013', 'PCAperf2013']:
    if outcome not in df_r2.columns:
        continue

    # Blacks
    reg = run_ols_regression(
        df_r2, outcome, 'mixracebas',
        controls=controls_subsample + rocontrols,
        fe_vars=['Res_base', 'regprogram_code'],
        sample_condition=df_r2['black'] == 1
    )
    row = create_result_row(
        f'robust/outcome/year2_{outcome}_black',
        'robustness/measurement.md',
        outcome, 'mixracebas', reg,
        'Black students, year 2 outcome',
        'Residence + Program FE', 'Own + roommate controls',
        'roomnum_base', 'OLS'
    )
    if row:
        results.append(row)

    # Whites
    reg = run_ols_regression(
        df_r2, outcome, 'mixracebas',
        controls=controls_subsample + rocontrols,
        fe_vars=['Res_base', 'regprogram_code'],
        sample_condition=df_r2['white'] == 1
    )
    row = create_result_row(
        f'robust/outcome/year2_{outcome}_white',
        'robustness/measurement.md',
        outcome, 'mixracebas', reg,
        'White students, year 2 outcome',
        'Residence + Program FE', 'Own + roommate controls',
        'roomnum_base', 'OLS'
    )
    if row:
        results.append(row)

# Residential choice outcomes (Table 6)
for outcome in ['stillinres', 'Inresmix_yr2', 'Same_ro_yr2']:
    if outcome not in df_r2.columns:
        continue

    # Blacks
    reg = run_ols_regression(
        df_r2, outcome, 'mixracebas',
        controls=controls_subsample + rocontrols,
        fe_vars=['Res_base'],
        sample_condition=df_r2['black'] == 1
    )
    row = create_result_row(
        f'robust/outcome/{outcome}_black',
        'robustness/measurement.md',
        outcome, 'mixracebas', reg,
        'Black students, residential outcome',
        'Residence FE', 'Own + roommate controls',
        'roomnum_base', 'OLS'
    )
    if row:
        results.append(row)

    # Whites
    reg = run_ols_regression(
        df_r2, outcome, 'mixracebas',
        controls=controls_subsample + rocontrols,
        fe_vars=['Res_base'],
        sample_condition=df_r2['white'] == 1
    )
    row = create_result_row(
        f'robust/outcome/{outcome}_white',
        'robustness/measurement.md',
        outcome, 'mixracebas', reg,
        'White students, residential outcome',
        'Residence FE', 'Own + roommate controls',
        'roomnum_base', 'OLS'
    )
    if row:
        results.append(row)

# Additional friendship measures
for outcome in ['pctFriendsdiff_net', 'pctStudydiffer_net', 'LeisuOth', 'StudyOth']:
    if outcome not in df_r2.columns:
        continue

    # Whites
    reg = run_ols_regression(
        df_r2, outcome, 'mixracebas',
        controls=controls_subsample + rocontrols,
        fe_vars=['Res_base'],
        sample_condition=df_r2['white'] == 1
    )
    row = create_result_row(
        f'robust/outcome/{outcome}_white',
        'robustness/measurement.md',
        outcome, 'mixracebas', reg,
        'White students, friendship outcome',
        'Residence FE', 'Own + roommate controls',
        'roomnum_base', 'OLS'
    )
    if row:
        results.append(row)

    # Blacks
    reg = run_ols_regression(
        df_r2, outcome, 'mixracebas',
        controls=controls_subsample + rocontrols,
        fe_vars=['Res_base'],
        sample_condition=df_r2['black'] == 1
    )
    row = create_result_row(
        f'robust/outcome/{outcome}_black',
        'robustness/measurement.md',
        outcome, 'mixracebas', reg,
        'Black students, friendship outcome',
        'Residence FE', 'Own + roommate controls',
        'roomnum_base', 'OLS'
    )
    if row:
        results.append(row)

print(f"  Alternative outcomes: {len(results) - n_before}")

# ============================================================================
# 5. PLACEBO TESTS
# ============================================================================
print("Running placebo tests...")
n_before = len(results)

# Placebo: baseline IAT should not be affected by treatment
for outcome in ['DscoreraceIATbas', 'DscoreacaIATbas']:
    if outcome not in df_r2.columns:
        continue

    for sample_cond, sample_name in [(df_r2['white'] == 1, 'white'), (df_r2['black'] == 1, 'black')]:
        reg = run_ols_regression(
            df_r2, outcome, 'mixracebas',
            controls=controls_subsample + rocontrols,
            fe_vars=['Res_base'],
            sample_condition=sample_cond
        )
        row = create_result_row(
            f'robust/placebo/baseline_{outcome}_{sample_name}',
            'robustness/placebo_tests.md',
            outcome, 'mixracebas', reg,
            f'{sample_name.capitalize()} students, placebo (baseline outcome)',
            'Residence FE', 'Own + roommate controls',
            'roomnum_base', 'OLS'
        )
        if row:
            results.append(row)

# Placebo: baseline attitude
if 'L_PCAattitude' in df_r2.columns:
    for sample_cond, sample_name in [(df_r2['white'] == 1, 'white'), (df_r2['black'] == 1, 'black')]:
        reg = run_ols_regression(
            df_r2, 'L_PCAattitude', 'mixracebas',
            controls=controls_subsample + rocontrols,
            fe_vars=['Res_base'],
            sample_condition=sample_cond
        )
        row = create_result_row(
            f'robust/placebo/baseline_attitude_{sample_name}',
            'robustness/placebo_tests.md',
            'L_PCAattitude', 'mixracebas', reg,
            f'{sample_name.capitalize()} students, placebo (baseline attitude)',
            'Residence FE', 'Own + roommate controls',
            'roomnum_base', 'OLS'
        )
        if row:
            results.append(row)

print(f"  Placebo tests: {len(results) - n_before}")

# ============================================================================
# 6. SAMPLE RESTRICTIONS
# ============================================================================
print("Running sample restrictions...")
n_before = len(results)

# By gender
for gender, gender_name in [(1, 'female'), (0, 'male')]:
    # Race IAT for whites
    reg = run_ols_regression(
        df_r2, 'DscoreraceIAT', 'mixracebas',
        controls=['L_DscoreraceIAT'] + [c for c in controls_subsample if c != 'Female_num'] + rocontrols,
        fe_vars=['Res_base'],
        sample_condition=(df_r2['white'] == 1) & (df_r2['Female_num'] == gender)
    )
    row = create_result_row(
        f'robust/sample/{gender_name}_IAT_white',
        'robustness/sample_restrictions.md',
        'DscoreraceIAT', 'mixracebas', reg,
        f'White {gender_name} students',
        'Residence FE', 'Own + roommate controls',
        'roomnum_base', 'OLS'
    )
    if row:
        results.append(row)

    # GPA for blacks
    reg = run_ols_regression(
        df_r2, 'GPA', 'mixracebas',
        controls=[c for c in controls_subsample if c != 'Female_num'] + rocontrols,
        fe_vars=['Res_base', 'regprogram_code'],
        sample_condition=(df_r2['black'] == 1) & (df_r2['Female_num'] == gender)
    )
    row = create_result_row(
        f'robust/sample/{gender_name}_GPA_black',
        'robustness/sample_restrictions.md',
        'GPA', 'mixracebas', reg,
        f'Black {gender_name} students',
        'Residence + Program FE', 'Own + roommate controls',
        'roomnum_base', 'OLS'
    )
    if row:
        results.append(row)

    # Race IAT for blacks
    reg = run_ols_regression(
        df_r2, 'DscoreraceIAT', 'mixracebas',
        controls=['L_DscoreraceIAT'] + [c for c in controls_subsample if c != 'Female_num'] + rocontrols,
        fe_vars=['Res_base'],
        sample_condition=(df_r2['black'] == 1) & (df_r2['Female_num'] == gender)
    )
    row = create_result_row(
        f'robust/sample/{gender_name}_IAT_black',
        'robustness/sample_restrictions.md',
        'DscoreraceIAT', 'mixracebas', reg,
        f'Black {gender_name} students',
        'Residence FE', 'Own + roommate controls',
        'roomnum_base', 'OLS'
    )
    if row:
        results.append(row)

    # GPA for whites
    reg = run_ols_regression(
        df_r2, 'GPA', 'mixracebas',
        controls=[c for c in controls_subsample if c != 'Female_num'] + rocontrols,
        fe_vars=['Res_base', 'regprogram_code'],
        sample_condition=(df_r2['white'] == 1) & (df_r2['Female_num'] == gender)
    )
    row = create_result_row(
        f'robust/sample/{gender_name}_GPA_white',
        'robustness/sample_restrictions.md',
        'GPA', 'mixracebas', reg,
        f'White {gender_name} students',
        'Residence + Program FE', 'Own + roommate controls',
        'roomnum_base', 'OLS'
    )
    if row:
        results.append(row)

# By residence - drop each residence
for res in df_r2['Res_base'].dropna().unique():
    if df_r2[df_r2['Res_base'] == res].shape[0] < 20:
        continue

    # GPA for blacks dropping residence
    reg = run_ols_regression(
        df_r2, 'GPA', 'mixracebas',
        controls=controls_subsample + rocontrols,
        fe_vars=['Res_base', 'regprogram_code'],
        sample_condition=(df_r2['black'] == 1) & (df_r2['Res_base'] != res)
    )
    row = create_result_row(
        f'robust/sample/drop_res{int(res)}_GPA_black',
        'robustness/sample_restrictions.md',
        'GPA', 'mixracebas', reg,
        f'Black students, drop residence {int(res)}',
        'Residence + Program FE', 'Own + roommate controls',
        'roomnum_base', 'OLS'
    )
    if row:
        results.append(row)

    # IAT for whites dropping residence
    reg = run_ols_regression(
        df_r2, 'DscoreraceIAT', 'mixracebas',
        controls=['L_DscoreraceIAT'] + controls_subsample + rocontrols,
        fe_vars=['Res_base'],
        sample_condition=(df_r2['white'] == 1) & (df_r2['Res_base'] != res)
    )
    row = create_result_row(
        f'robust/sample/drop_res{int(res)}_IAT_white',
        'robustness/sample_restrictions.md',
        'DscoreraceIAT', 'mixracebas', reg,
        f'White students, drop residence {int(res)}',
        'Residence FE', 'Own + roommate controls',
        'roomnum_base', 'OLS'
    )
    if row:
        results.append(row)

print(f"  Sample restrictions: {len(results) - n_before}")

# ============================================================================
# 7. HETEROGENEITY ANALYSIS
# ============================================================================
print("Running heterogeneity analysis...")
n_before = len(results)

# Create interaction terms
df_r2['mix_x_female'] = df_r2['mixracebas'] * df_r2['Female_num']
df_r2['mix_x_falseuct'] = df_r2['mixracebas'] * df_r2['Falseuct2012'].fillna(0)
df_r2['mix_x_foreign'] = df_r2['mixracebas'] * df_r2['Foreign'].fillna(0)
df_r2['mix_x_private'] = df_r2['mixracebas'] * df_r2['privateschool_nomiss'].fillna(0)

# Heterogeneity by gender
for outcome, sample_cond, sample_name in [
    ('DscoreraceIAT', df_r2['white'] == 1, 'white'),
    ('GPA', df_r2['black'] == 1, 'black'),
    ('DscoreraceIAT', df_r2['black'] == 1, 'black'),
    ('GPA', df_r2['white'] == 1, 'white'),
]:
    if outcome == 'DscoreraceIAT':
        base_controls = ['L_DscoreraceIAT'] + controls_subsample + rocontrols
    else:
        base_controls = controls_subsample + rocontrols

    # Gender interaction
    reg = run_ols_regression(
        df_r2, outcome, 'mixracebas',
        controls=base_controls + ['mix_x_female'],
        fe_vars=['Res_base'],
        sample_condition=sample_cond
    )
    row = create_result_row(
        f'robust/heterogeneity/gender_{outcome}_{sample_name}',
        'robustness/heterogeneity.md',
        outcome, 'mixracebas', reg,
        f'{sample_name.capitalize()} students, gender interaction',
        'Residence FE', 'Full controls + gender interaction',
        'roomnum_base', 'OLS'
    )
    if row:
        results.append(row)

    # Private school interaction
    reg = run_ols_regression(
        df_r2, outcome, 'mixracebas',
        controls=base_controls + ['mix_x_private'],
        fe_vars=['Res_base'],
        sample_condition=sample_cond
    )
    row = create_result_row(
        f'robust/heterogeneity/private_{outcome}_{sample_name}',
        'robustness/heterogeneity.md',
        outcome, 'mixracebas', reg,
        f'{sample_name.capitalize()} students, private school interaction',
        'Residence FE', 'Full controls + private school interaction',
        'roomnum_base', 'OLS'
    )
    if row:
        results.append(row)

    # Foreign interaction
    reg = run_ols_regression(
        df_r2, outcome, 'mixracebas',
        controls=base_controls + ['mix_x_foreign'],
        fe_vars=['Res_base'],
        sample_condition=sample_cond
    )
    row = create_result_row(
        f'robust/heterogeneity/foreign_{outcome}_{sample_name}',
        'robustness/heterogeneity.md',
        outcome, 'mixracebas', reg,
        f'{sample_name.capitalize()} students, foreign interaction',
        'Residence FE', 'Full controls + foreign interaction',
        'roomnum_base', 'OLS'
    )
    if row:
        results.append(row)

print(f"  Heterogeneity analysis: {len(results) - n_before}")

# ============================================================================
# 8. ESTIMATION METHOD VARIATIONS
# ============================================================================
print("Running estimation method variations...")
n_before = len(results)

# No fixed effects
for outcome, sample_cond, sample_name in [
    ('DscoreraceIAT', df_r2['white'] == 1, 'white'),
    ('GPA', df_r2['black'] == 1, 'black'),
    ('DscoreraceIAT', df_r2['black'] == 1, 'black'),
    ('GPA', df_r2['white'] == 1, 'white'),
]:
    if outcome == 'DscoreraceIAT':
        ctrls = ['L_DscoreraceIAT'] + controls_subsample + rocontrols
    else:
        ctrls = controls_subsample + rocontrols

    reg = run_ols_regression(
        df_r2, outcome, 'mixracebas',
        controls=ctrls,
        fe_vars=None,
        sample_condition=sample_cond
    )
    row = create_result_row(
        f'robust/estimation/no_fe_{outcome}_{sample_name}',
        'robustness/model_specification.md',
        outcome, 'mixracebas', reg,
        f'{sample_name.capitalize()} students, no FE',
        'None', 'Full controls',
        'roomnum_base', 'OLS'
    )
    if row:
        results.append(row)

print(f"  Estimation variations: {len(results) - n_before}")

# ============================================================================
# 9. WITHOUT ROOMMATE CONTROLS (Table A7, A9)
# ============================================================================
print("Running specifications without roommate controls...")
n_before = len(results)

# IAT without roommate controls
for outcome in ['DscoreraceIAT', 'DscoreacaIAT']:
    lagged = 'L_' + outcome

    # Whites
    reg = run_ols_regression(
        df_r2, outcome, 'mixracebas',
        controls=[lagged] + controls_subsample,
        fe_vars=['Res_base'],
        sample_condition=df_r2['white'] == 1
    )
    row = create_result_row(
        f'robust/control/no_rocontrols_{outcome}_white',
        'robustness/control_progression.md',
        outcome, 'mixracebas', reg,
        'White students, no roommate controls',
        'Residence FE', 'Own controls only (no roommate)',
        'roomnum_base', 'OLS'
    )
    if row:
        results.append(row)

    # Blacks
    reg = run_ols_regression(
        df_r2, outcome, 'mixracebas',
        controls=[lagged] + controls_subsample,
        fe_vars=['Res_base'],
        sample_condition=df_r2['black'] == 1
    )
    row = create_result_row(
        f'robust/control/no_rocontrols_{outcome}_black',
        'robustness/control_progression.md',
        outcome, 'mixracebas', reg,
        'Black students, no roommate controls',
        'Residence FE', 'Own controls only (no roommate)',
        'roomnum_base', 'OLS'
    )
    if row:
        results.append(row)

# Academic performance without roommate controls
for outcome in ['GPA', 'examspassed', 'continue', 'PCAperf']:
    # Whites
    reg = run_ols_regression(
        df_r2, outcome, 'mixracebas',
        controls=controls_subsample,
        fe_vars=['Res_base', 'regprogram_code'],
        sample_condition=df_r2['white'] == 1
    )
    row = create_result_row(
        f'robust/control/no_rocontrols_{outcome}_white',
        'robustness/control_progression.md',
        outcome, 'mixracebas', reg,
        'White students, no roommate controls',
        'Residence + Program FE', 'Own controls only',
        'roomnum_base', 'OLS'
    )
    if row:
        results.append(row)

    # Blacks
    reg = run_ols_regression(
        df_r2, outcome, 'mixracebas',
        controls=controls_subsample,
        fe_vars=['Res_base', 'regprogram_code'],
        sample_condition=df_r2['black'] == 1
    )
    row = create_result_row(
        f'robust/control/no_rocontrols_{outcome}_black',
        'robustness/control_progression.md',
        outcome, 'mixracebas', reg,
        'Black students, no roommate controls',
        'Residence + Program FE', 'Own controls only',
        'roomnum_base', 'OLS'
    )
    if row:
        results.append(row)

print(f"  Without roommate controls: {len(results) - n_before}")

# ============================================================================
# 10. PRO-SOCIAL BEHAVIOR (Table A13-A14)
# ============================================================================
print("Running pro-social behavior outcomes...")
n_before = len(results)

# Pro-social behavior
for outcome in ['Cooperate', 'Pris_coopbelief']:
    if outcome not in df_r2.columns:
        continue

    # Check if numeric
    if not pd.api.types.is_numeric_dtype(df_r2[outcome]):
        continue

    # Full sample
    reg = run_ols_regression(
        df_r2, outcome, 'mixracebas',
        controls=controls_full + rocontrols,
        fe_vars=['Res_base']
    )
    row = create_result_row(
        f'robust/outcome/{outcome}_full',
        'robustness/measurement.md',
        outcome, 'mixracebas', reg,
        'Full sample, pro-social outcome',
        'Residence FE', 'Full + roommate controls',
        'roomnum_base', 'OLS'
    )
    if row:
        results.append(row)

    # Whites
    reg = run_ols_regression(
        df_r2, outcome, 'mixracebas',
        controls=controls_subsample + rocontrols,
        fe_vars=['Res_base'],
        sample_condition=df_r2['white'] == 1
    )
    row = create_result_row(
        f'robust/outcome/{outcome}_white',
        'robustness/measurement.md',
        outcome, 'mixracebas', reg,
        'White students, pro-social outcome',
        'Residence FE', 'Own + roommate controls',
        'roomnum_base', 'OLS'
    )
    if row:
        results.append(row)

    # Blacks
    reg = run_ols_regression(
        df_r2, outcome, 'mixracebas',
        controls=controls_subsample + rocontrols,
        fe_vars=['Res_base'],
        sample_condition=df_r2['black'] == 1
    )
    row = create_result_row(
        f'robust/outcome/{outcome}_black',
        'robustness/measurement.md',
        outcome, 'mixracebas', reg,
        'Black students, pro-social outcome',
        'Residence FE', 'Own + roommate controls',
        'roomnum_base', 'OLS'
    )
    if row:
        results.append(row)

print(f"  Pro-social behavior: {len(results) - n_before}")

# ============================================================================
# 11. FUNCTIONAL FORM VARIATIONS
# ============================================================================
print("Running functional form variations...")
n_before = len(results)

# Standardized outcomes
for outcome in ['DscoreraceIAT', 'GPA']:
    for sample_cond, sample_name in [(df_r2['white'] == 1, 'white'), (df_r2['black'] == 1, 'black')]:
        data = df_r2[sample_cond].copy()

        # Standardize outcome
        outcome_std = f'{outcome}_std'
        mean_val = data[outcome].mean()
        std_val = data[outcome].std()
        if std_val > 0:
            data[outcome_std] = (data[outcome] - mean_val) / std_val
        else:
            continue

        if outcome == 'DscoreraceIAT':
            ctrls = ['L_DscoreraceIAT'] + controls_subsample + rocontrols
        else:
            ctrls = controls_subsample + rocontrols

        reg = run_ols_regression(
            data, outcome_std, 'mixracebas',
            controls=ctrls,
            fe_vars=['Res_base']
        )
        row = create_result_row(
            f'robust/funcform/standardized_{outcome}_{sample_name}',
            'robustness/functional_form.md',
            outcome_std, 'mixracebas', reg,
            f'{sample_name.capitalize()} students, standardized outcome',
            'Residence FE', 'Full controls',
            'roomnum_base', 'OLS'
        )
        if row:
            results.append(row)

print(f"  Functional form: {len(results) - n_before}")

# ============================================================================
# Save Results
# ============================================================================
print(f"\nTotal specifications: {len(results)}")

# Create DataFrame and save
results_df = pd.DataFrame(results)
output_path = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/174501-V1/specification_results.csv'
results_df.to_csv(output_path, index=False)
print(f"Results saved to: {output_path}")

# Print summary statistics
if len(results_df) > 0:
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total specifications: {len(results_df)}")
    print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
    print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
    print(f"Significant at 10%: {(results_df['p_value'] < 0.10).sum()} ({100*(results_df['p_value'] < 0.10).mean():.1f}%)")
    print(f"Coefficient range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")
    print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
    print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")

    # Summary by category
    print("\n=== BREAKDOWN BY CATEGORY ===")
    results_df['category'] = results_df['spec_id'].apply(lambda x: x.split('/')[0] + '/' + x.split('/')[1] if len(x.split('/')) > 1 else x.split('/')[0])
    for cat in sorted(results_df['category'].unique()):
        subset = results_df[results_df['category'] == cat]
        n_pos = (subset['coefficient'] > 0).sum()
        n_sig = (subset['p_value'] < 0.05).sum()
        print(f"{cat}: N={len(subset)}, Positive={100*n_pos/len(subset):.0f}%, Sig 5%={100*n_sig/len(subset):.0f}%")
else:
    print("No results generated!")
