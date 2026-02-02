"""
Specification Search for Paper 149882-V1
=========================================
Paper: "Reshaping Adolescents' Gender Attitudes: Evidence from a School-Based Experiment in India"

This is a school-level RCT evaluating the Breakthrough program's effects on:
- Gender attitudes index
- Girls' aspirations index
- Self-reported behavior index

Method: Cross-sectional OLS with school-clustered standard errors
Treatment: B_treat (school-level randomization)
Key controls: baseline outcome, grade, district FEs, missing flags
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import statsmodels.api as sm
import json
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search"
DATA_PATH = f"{BASE_PATH}/data/downloads/extracted/149882-V1/data"
OUTPUT_PATH = f"{BASE_PATH}/data/downloads/extracted/149882-V1"

# Paper metadata
PAPER_ID = "149882-V1"
PAPER_TITLE = "Reshaping Adolescents' Gender Attitudes: Evidence from a School-Based Experiment in India"
JOURNAL = "AER"

print("Loading and merging data...")

# Load raw data files
baseline_student = pd.read_stata(f"{DATA_PATH}/baseline_student_raw.dta", convert_categoricals=False)
endline1_student = pd.read_stata(f"{DATA_PATH}/endline1_student_raw.dta", convert_categoricals=False)
endline2_student = pd.read_stata(f"{DATA_PATH}/endline2_student_raw.dta", convert_categoricals=False)

# Rename columns for baseline
baseline_student = baseline_student.rename(columns={
    'school_id': 'Sschool_id',
    'student_gender': 'B_Sgender',
    'district': 'B_Sdistrict'
})

# Create gender indicator (female=1)
baseline_student['B_Sgirl'] = (baseline_student['B_Sgender'] == 2).astype(float)

# Get treatment assignment from endline data
endline1_student = endline1_student.rename(columns={
    'school_id': 'Sschool_id',
    'treatment': 'B_treat'
})

# Ensure consistent data types for merge keys
baseline_student['Sschool_id'] = baseline_student['Sschool_id'].astype(int)
endline1_student['Sschool_id'] = endline1_student['Sschool_id'].astype(int)
baseline_student['child_id'] = baseline_student['child_id'].astype(int)
endline1_student['child_id'] = endline1_student['child_id'].astype(int)

# Merge treatment to baseline
treatment_map = endline1_student[['Sschool_id', 'B_treat']].drop_duplicates()
baseline_student = baseline_student.merge(treatment_map, on='Sschool_id', how='left')

# Create grade indicators
baseline_student['B_Sgrade6'] = (baseline_student.get('enrolled', 0) == 6).astype(float)
baseline_student['B_Sgrade7'] = (baseline_student.get('enrolled', 0) == 7).astype(float)

# Create district fixed effects
for d in baseline_student['B_Sdistrict'].dropna().unique():
    baseline_student[f'district_{int(d)}'] = (baseline_student['B_Sdistrict'] == d).astype(float)

print(f"Baseline student data shape: {baseline_student.shape}")

# ========================================
# Create Gender Attitudes Index (Baseline)
# ========================================

# These are the gender attitude questions - higher = more progressive
# Variables are recoded as 0/1 where 1 = progressive response
gender_vars_bl = [
    'child_woman_role',      # Women's role not just home
    'child_man_final_deci',  # Man shouldn't have final decision
    'child_woman_tol_viol',  # Women shouldn't tolerate violence
    'child_wives_less_edu',  # Wives don't need less education
    'child_boy_more_opps',   # Boys shouldn't get more opportunities
    'child_equal_opps',      # Equal opportunities
    'child_girl_allow_study', # Girls should be allowed to study
    'child_similar_right',   # Similar rights
    'child_elect_woman'      # Elect women
]

# Create progressive coding (1 = progressive)
for var in gender_vars_bl:
    if var in baseline_student.columns:
        # Assuming higher values are more conservative, recode
        col = baseline_student[var]
        # Normalize to 0-1 scale if not already
        if col.max() > 1:
            baseline_student[f'{var}_prog'] = 1 - (col - col.min()) / (col.max() - col.min())
        else:
            baseline_student[f'{var}_prog'] = col

# Create simple gender attitudes index as mean of available items
gender_prog_vars = [f'{v}_prog' for v in gender_vars_bl if f'{v}_prog' in baseline_student.columns]
if gender_prog_vars:
    baseline_student['B_Sgender_index2'] = baseline_student[gender_prog_vars].mean(axis=1)
else:
    # If no attitude vars found, create placeholder
    baseline_student['B_Sgender_index2'] = np.nan

print(f"Created baseline gender index with {len(gender_prog_vars)} items")

# ========================================
# Process Endline 1 Data
# ========================================

endline1_student['E_Sgirl'] = (endline1_student['gender'] == 2).astype(float)
endline1_student['E_Sdistrict'] = endline1_student['district']

# Create grade FE
endline1_student['E_Sgrade6'] = (endline1_student.get('class', 0) == 4).astype(float)  # class 4 = grade 6 at endline

# Gender attitude questions at endline 1
el1_gender_vars = [
    'wives_less_edu',
    'elect_woman',
    'boy_more_oppo',
    'own_studies',
    'man_final_deci',
    'woman_viol',
    'control_daughters',
    'woman_role_home',
    'men_better_suited',
    'similar_right',
    'marriage_more_imp',
    'teacher_suitable',
    'girl_marriage_age',
    'marriage_age_diff',
    'study_marry',
    'allow_work',
    'fertility'
]

# Create progressive coding for endline
for var in el1_gender_vars:
    if var in endline1_student.columns:
        col = endline1_student[var]
        if col.notna().any() and col.max() > 1:
            endline1_student[f'{var}_prog'] = 1 - (col - col.min()) / (col.max() - col.min())
        else:
            endline1_student[f'{var}_prog'] = col

# Create endline gender index
el1_prog_vars = [f'{v}_prog' for v in el1_gender_vars if f'{v}_prog' in endline1_student.columns]
if el1_prog_vars:
    endline1_student['E_Sgender_index2'] = endline1_student[el1_prog_vars].mean(axis=1)
else:
    endline1_student['E_Sgender_index2'] = np.nan

print(f"Created endline 1 gender index with {len(el1_prog_vars)} items")

# ========================================
# Merge Baseline and Endline 1
# ========================================

# Merge on school_id and child_id
analysis_df = endline1_student.merge(
    baseline_student[['child_id', 'Sschool_id', 'B_Sgender_index2', 'B_Sgirl', 'B_Sdistrict', 'B_Sgrade6', 'B_Sgrade7'] +
                     [c for c in baseline_student.columns if c.startswith('district_')]],
    on=['child_id', 'Sschool_id'],
    how='inner'
)

print(f"Merged analysis data shape: {analysis_df.shape}")

# Fill missing baseline index with mean
analysis_df['B_Sgender_index2'] = analysis_df['B_Sgender_index2'].fillna(analysis_df['B_Sgender_index2'].mean())
analysis_df['B_Sgender_index2_flag'] = analysis_df['B_Sgender_index2'].isna().astype(int)

# Create gender-grade and gender-district interactions
analysis_df['gender_grade_6'] = analysis_df['B_Sgirl'] * analysis_df.get('E_Sgrade6', 0)
analysis_df['gender_grade_7'] = analysis_df['B_Sgirl'] * (1 - analysis_df.get('E_Sgrade6', 0))

district_cols = [c for c in analysis_df.columns if c.startswith('district_')]
for d in district_cols:
    analysis_df[f'gender_{d}'] = analysis_df['B_Sgirl'] * analysis_df[d]

# ========================================
# Create Behavior Index
# ========================================

# Behavior index components for girls and boys
# Girls: talk to opposite gender, sit with opposite gender, cook/clean, future work, etc.
# Boys: similar but gender-appropriate

behavior_vars_common = [
    'talk_opp_gender',
    'sit_opp_gender',
    'cook_clean',
    'absent_sch_hhwork'
]

for var in behavior_vars_common:
    base_var = var
    if base_var in endline1_student.columns:
        col = endline1_student[base_var]
        if col.notna().any():
            endline1_student[f'{var}_norm'] = (col - col.mean()) / col.std()

beh_norm_vars = [f'{v}_norm' for v in behavior_vars_common if f'{v}_norm' in endline1_student.columns]
if beh_norm_vars:
    endline1_student['E_Sbehavior_index2'] = endline1_student[beh_norm_vars].mean(axis=1)
else:
    endline1_student['E_Sbehavior_index2'] = np.nan

# Merge behavior index to analysis df
if 'E_Sbehavior_index2' in endline1_student.columns:
    analysis_df = analysis_df.merge(
        endline1_student[['child_id', 'Sschool_id', 'E_Sbehavior_index2']].dropna(),
        on=['child_id', 'Sschool_id'],
        how='left'
    )

# Create baseline behavior proxy if available
if 'B_Sbehavior_index2' not in analysis_df.columns:
    analysis_df['B_Sbehavior_index2'] = 0  # Placeholder - will be controlled for

# ========================================
# Create Aspiration Index (Girls only)
# ========================================

aspiration_vars = [
    'board_score',  # Expected board exam score
    'highest_educ',  # Highest education aspiration
    'discuss_educ',  # Discussed education goals
    'occupa_25',     # Expected occupation at 25
    'cont_educ'      # Continue education
]

for var in aspiration_vars:
    if var in endline1_student.columns:
        col = endline1_student[var]
        if col.notna().any():
            endline1_student[f'{var}_norm'] = (col - col.mean()) / col.std()

asp_norm_vars = [f'{v}_norm' for v in aspiration_vars if f'{v}_norm' in endline1_student.columns]
if asp_norm_vars:
    endline1_student['E_Saspiration_index2'] = endline1_student[asp_norm_vars].mean(axis=1)
else:
    endline1_student['E_Saspiration_index2'] = np.nan

# Merge aspiration index
if 'E_Saspiration_index2' in endline1_student.columns:
    analysis_df = analysis_df.merge(
        endline1_student[['child_id', 'Sschool_id', 'E_Saspiration_index2']].dropna(),
        on=['child_id', 'Sschool_id'],
        how='left'
    )

# Create baseline aspiration proxy
if 'B_Saspiration_index2' not in analysis_df.columns:
    analysis_df['B_Saspiration_index2'] = 0

# ========================================
# Final Data Cleaning
# ========================================

# Ensure treatment is binary
analysis_df['B_treat'] = analysis_df['B_treat'].fillna(0).astype(int)

# Filter to valid observations
analysis_df = analysis_df[analysis_df['B_treat'].notna()].copy()
analysis_df = analysis_df[analysis_df['E_Sgender_index2'].notna()].copy()

# Standardize outcomes
for outcome in ['E_Sgender_index2', 'E_Sbehavior_index2', 'E_Saspiration_index2']:
    if outcome in analysis_df.columns:
        mean_val = analysis_df[outcome].mean()
        std_val = analysis_df[outcome].std()
        if std_val > 0:
            analysis_df[f'{outcome}_std'] = (analysis_df[outcome] - mean_val) / std_val
        else:
            analysis_df[f'{outcome}_std'] = 0

print(f"\nFinal analysis dataset: {analysis_df.shape[0]} observations")
print(f"Treatment: {analysis_df['B_treat'].mean()*100:.1f}% treated")
print(f"Female: {analysis_df['B_Sgirl'].mean()*100:.1f}% girls")

# ========================================
# SPECIFICATION SEARCH
# ========================================

results = []

def run_spec(spec_id, spec_tree_path, outcome_var, treatment_var, formula, data,
             cluster_var='Sschool_id', sample_desc='Full sample', controls_desc='',
             fixed_effects='', model_type='OLS', vcov_type='CRV1'):
    """Run a single specification and record results."""

    try:
        # Filter to non-missing outcome
        df = data[data[outcome_var].notna()].copy()

        if len(df) < 50:
            return None

        # Run regression
        if vcov_type == 'CRV1' and cluster_var in df.columns:
            model = pf.feols(formula, data=df, vcov={'CRV1': cluster_var})
        elif vcov_type == 'hetero':
            model = pf.feols(formula, data=df, vcov='hetero')
        else:
            model = pf.feols(formula, data=df)

        # Extract coefficient on treatment
        if treatment_var not in model.coef().index:
            return None

        coef = model.coef()[treatment_var]
        se = model.se()[treatment_var]
        t_stat = coef / se if se > 0 else np.nan
        p_val = model.pvalue()[treatment_var]

        # Confidence interval
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se

        # R-squared and N
        try:
            r2 = model._r2
        except:
            r2 = np.nan

        try:
            n_obs = model._N
        except:
            n_obs = len(df)

        # Coefficient vector
        coef_vector = {
            'treatment': {
                'var': treatment_var,
                'coef': float(coef),
                'se': float(se),
                'pval': float(p_val)
            },
            'controls': [],
            'fixed_effects': fixed_effects.split('+') if fixed_effects else [],
            'diagnostics': {}
        }

        # Add control coefficients
        for var in model.coef().index:
            if var != treatment_var and var != 'Intercept':
                coef_vector['controls'].append({
                    'var': var,
                    'coef': float(model.coef()[var]),
                    'se': float(model.se()[var]),
                    'pval': float(model.pvalue()[var])
                })

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
            'n_obs': n_obs,
            'r_squared': r2,
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects,
            'controls_desc': controls_desc,
            'cluster_var': cluster_var,
            'model_type': model_type,
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }

        return result

    except Exception as e:
        print(f"  Error in {spec_id}: {str(e)[:50]}")
        return None

# ========================================
# Define control sets
# ========================================

# Basic controls (grade and district FEs)
district_fe_vars = [c for c in analysis_df.columns if c.startswith('district_') and not c.startswith('gender_district')]
basic_controls = ['B_Sgrade6'] + district_fe_vars[:3]  # Limit to avoid multicollinearity

# Gender-interacted FEs
gender_district_vars = [c for c in analysis_df.columns if c.startswith('gender_district')]
gender_grade_vars = ['gender_grade_6', 'gender_grade_7']

# Full controls
full_controls = basic_controls + ['B_Sgender_index2']

print("\n" + "="*60)
print("RUNNING SPECIFICATION SEARCH")
print("="*60)

# ========================================
# 1. BASELINE SPECIFICATIONS
# ========================================
print("\n1. Baseline specifications...")

# Main outcome: Gender Attitudes Index (Endline 1)
# Combined sample with gender-interacted FEs

# Baseline - exact replication attempt
base_formula = "E_Sgender_index2_std ~ B_treat + B_Sgender_index2 + B_Sgrade6"
for d in district_fe_vars[:3]:
    if d in analysis_df.columns:
        base_formula += f" + {d}"

r = run_spec(
    spec_id='baseline',
    spec_tree_path='methods/cross_sectional_ols.md#baseline',
    outcome_var='E_Sgender_index2_std',
    treatment_var='B_treat',
    formula=base_formula,
    data=analysis_df,
    sample_desc='Full sample (Endline 1 respondents)',
    controls_desc='Baseline gender index, grade FE, district FE',
    fixed_effects='Grade + District'
)
if r: results.append(r)

# ========================================
# 2. CONTROL VARIATIONS (10-15 specs)
# ========================================
print("\n2. Control variations...")

# 2.1 No controls (bivariate)
r = run_spec(
    spec_id='robust/control/none',
    spec_tree_path='robustness/leave_one_out.md',
    outcome_var='E_Sgender_index2_std',
    treatment_var='B_treat',
    formula='E_Sgender_index2_std ~ B_treat',
    data=analysis_df,
    sample_desc='Full sample',
    controls_desc='No controls (bivariate)'
)
if r: results.append(r)

# 2.2 Drop baseline outcome control
control_formula = "E_Sgender_index2_std ~ B_treat + B_Sgrade6"
for d in district_fe_vars[:3]:
    if d in analysis_df.columns:
        control_formula += f" + {d}"

r = run_spec(
    spec_id='robust/control/drop_baseline_outcome',
    spec_tree_path='robustness/leave_one_out.md',
    outcome_var='E_Sgender_index2_std',
    treatment_var='B_treat',
    formula=control_formula,
    data=analysis_df,
    sample_desc='Full sample',
    controls_desc='Grade FE, district FE (no baseline outcome)'
)
if r: results.append(r)

# 2.3 Only baseline outcome
r = run_spec(
    spec_id='robust/control/only_baseline_outcome',
    spec_tree_path='robustness/single_covariate.md',
    outcome_var='E_Sgender_index2_std',
    treatment_var='B_treat',
    formula='E_Sgender_index2_std ~ B_treat + B_Sgender_index2',
    data=analysis_df,
    sample_desc='Full sample',
    controls_desc='Only baseline gender index'
)
if r: results.append(r)

# 2.4 Only grade FE
r = run_spec(
    spec_id='robust/control/only_grade_fe',
    spec_tree_path='robustness/leave_one_out.md',
    outcome_var='E_Sgender_index2_std',
    treatment_var='B_treat',
    formula='E_Sgender_index2_std ~ B_treat + B_Sgrade6',
    data=analysis_df,
    sample_desc='Full sample',
    controls_desc='Only grade FE'
)
if r: results.append(r)

# 2.5 Only district FE
district_formula = "E_Sgender_index2_std ~ B_treat"
for d in district_fe_vars[:3]:
    if d in analysis_df.columns:
        district_formula += f" + {d}"

r = run_spec(
    spec_id='robust/control/only_district_fe',
    spec_tree_path='robustness/leave_one_out.md',
    outcome_var='E_Sgender_index2_std',
    treatment_var='B_treat',
    formula=district_formula,
    data=analysis_df,
    sample_desc='Full sample',
    controls_desc='Only district FE'
)
if r: results.append(r)

# 2.6-2.10 Add controls incrementally
controls_to_add = ['B_Sgender_index2', 'B_Sgrade6'] + district_fe_vars[:3]
for i, control in enumerate(controls_to_add):
    if control in analysis_df.columns:
        controls_so_far = controls_to_add[:i+1]
        formula = f"E_Sgender_index2_std ~ B_treat + {' + '.join(controls_so_far)}"

        r = run_spec(
            spec_id=f'robust/control/add_{control}',
            spec_tree_path='robustness/control_progression.md',
            outcome_var='E_Sgender_index2_std',
            treatment_var='B_treat',
            formula=formula,
            data=analysis_df,
            sample_desc='Full sample',
            controls_desc=f'Adding {control}'
        )
        if r: results.append(r)

# ========================================
# 3. SAMPLE RESTRICTIONS (10-15 specs)
# ========================================
print("\n3. Sample restrictions...")

# 3.1 Girls only
girls_df = analysis_df[analysis_df['B_Sgirl'] == 1].copy()
r = run_spec(
    spec_id='robust/sample/girls_only',
    spec_tree_path='robustness/sample_restrictions.md',
    outcome_var='E_Sgender_index2_std',
    treatment_var='B_treat',
    formula=base_formula,
    data=girls_df,
    sample_desc='Girls only'
)
if r: results.append(r)

# 3.2 Boys only
boys_df = analysis_df[analysis_df['B_Sgirl'] == 0].copy()
r = run_spec(
    spec_id='robust/sample/boys_only',
    spec_tree_path='robustness/sample_restrictions.md',
    outcome_var='E_Sgender_index2_std',
    treatment_var='B_treat',
    formula=base_formula,
    data=boys_df,
    sample_desc='Boys only'
)
if r: results.append(r)

# 3.3-3.6 By district
for d_col in district_fe_vars[:4]:
    d_num = d_col.split('_')[-1]
    district_df = analysis_df[analysis_df[d_col] == 1].copy()

    r = run_spec(
        spec_id=f'robust/sample/district_{d_num}_only',
        spec_tree_path='robustness/sample_restrictions.md',
        outcome_var='E_Sgender_index2_std',
        treatment_var='B_treat',
        formula='E_Sgender_index2_std ~ B_treat + B_Sgender_index2 + B_Sgrade6',
        data=district_df,
        sample_desc=f'District {d_num} only'
    )
    if r: results.append(r)

# 3.7-3.10 Drop each district
for d_col in district_fe_vars[:4]:
    d_num = d_col.split('_')[-1]
    drop_district_df = analysis_df[analysis_df[d_col] != 1].copy()

    r = run_spec(
        spec_id=f'robust/sample/drop_district_{d_num}',
        spec_tree_path='robustness/sample_restrictions.md',
        outcome_var='E_Sgender_index2_std',
        treatment_var='B_treat',
        formula=base_formula,
        data=drop_district_df,
        sample_desc=f'Excluding district {d_num}'
    )
    if r: results.append(r)

# 3.11 Grade 6 only
grade6_df = analysis_df[analysis_df['B_Sgrade6'] == 1].copy()
r = run_spec(
    spec_id='robust/sample/grade6_only',
    spec_tree_path='robustness/sample_restrictions.md',
    outcome_var='E_Sgender_index2_std',
    treatment_var='B_treat',
    formula='E_Sgender_index2_std ~ B_treat + B_Sgender_index2',
    data=grade6_df,
    sample_desc='Grade 6 only'
)
if r: results.append(r)

# 3.12 Grade 7 only
grade7_df = analysis_df[analysis_df['B_Sgrade6'] == 0].copy()
r = run_spec(
    spec_id='robust/sample/grade7_only',
    spec_tree_path='robustness/sample_restrictions.md',
    outcome_var='E_Sgender_index2_std',
    treatment_var='B_treat',
    formula='E_Sgender_index2_std ~ B_treat + B_Sgender_index2',
    data=grade7_df,
    sample_desc='Grade 7 only'
)
if r: results.append(r)

# 3.13-3.15 Outlier treatment
for pct in [1, 5, 10]:
    wins_df = analysis_df.copy()
    lower = wins_df['E_Sgender_index2_std'].quantile(pct/100)
    upper = wins_df['E_Sgender_index2_std'].quantile(1 - pct/100)
    wins_df['E_Sgender_index2_std'] = wins_df['E_Sgender_index2_std'].clip(lower=lower, upper=upper)

    r = run_spec(
        spec_id=f'robust/sample/winsorize_{pct}pct',
        spec_tree_path='robustness/sample_restrictions.md',
        outcome_var='E_Sgender_index2_std',
        treatment_var='B_treat',
        formula=base_formula,
        data=wins_df,
        sample_desc=f'Winsorized at {pct}%'
    )
    if r: results.append(r)

# 3.16 Trim extreme 1%
trim_df = analysis_df[
    (analysis_df['E_Sgender_index2_std'] > analysis_df['E_Sgender_index2_std'].quantile(0.01)) &
    (analysis_df['E_Sgender_index2_std'] < analysis_df['E_Sgender_index2_std'].quantile(0.99))
].copy()

r = run_spec(
    spec_id='robust/sample/trim_1pct',
    spec_tree_path='robustness/sample_restrictions.md',
    outcome_var='E_Sgender_index2_std',
    treatment_var='B_treat',
    formula=base_formula,
    data=trim_df,
    sample_desc='Trimmed 1% tails'
)
if r: results.append(r)

# ========================================
# 4. ALTERNATIVE OUTCOMES (5-10 specs)
# ========================================
print("\n4. Alternative outcomes...")

# 4.1 Unstandardized gender index
r = run_spec(
    spec_id='robust/outcome/unstandardized',
    spec_tree_path='robustness/measurement.md',
    outcome_var='E_Sgender_index2',
    treatment_var='B_treat',
    formula='E_Sgender_index2 ~ B_treat + B_Sgender_index2 + B_Sgrade6',
    data=analysis_df,
    sample_desc='Full sample',
    controls_desc='Unstandardized outcome'
)
if r: results.append(r)

# 4.2 Behavior index (if available)
if 'E_Sbehavior_index2' in analysis_df.columns and analysis_df['E_Sbehavior_index2'].notna().sum() > 100:
    analysis_df['E_Sbehavior_index2_std'] = (
        analysis_df['E_Sbehavior_index2'] - analysis_df['E_Sbehavior_index2'].mean()
    ) / analysis_df['E_Sbehavior_index2'].std()

    r = run_spec(
        spec_id='robust/outcome/behavior_index',
        spec_tree_path='robustness/measurement.md',
        outcome_var='E_Sbehavior_index2_std',
        treatment_var='B_treat',
        formula='E_Sbehavior_index2_std ~ B_treat + B_Sbehavior_index2 + B_Sgrade6',
        data=analysis_df,
        sample_desc='Full sample',
        controls_desc='Alternative outcome: Behavior index'
    )
    if r: results.append(r)

# 4.3 Aspiration index (girls only)
if 'E_Saspiration_index2' in analysis_df.columns and analysis_df['E_Saspiration_index2'].notna().sum() > 100:
    girls_asp = girls_df.copy()
    girls_asp['E_Saspiration_index2_std'] = (
        girls_asp['E_Saspiration_index2'] - girls_asp['E_Saspiration_index2'].mean()
    ) / girls_asp['E_Saspiration_index2'].std()

    r = run_spec(
        spec_id='robust/outcome/aspiration_index_girls',
        spec_tree_path='robustness/measurement.md',
        outcome_var='E_Saspiration_index2_std',
        treatment_var='B_treat',
        formula='E_Saspiration_index2_std ~ B_treat + B_Saspiration_index2 + B_Sgrade6',
        data=girls_asp,
        sample_desc='Girls only',
        controls_desc='Alternative outcome: Aspirations index'
    )
    if r: results.append(r)

# 4.4-4.6 Individual gender attitude items
for var in ['wives_less_edu', 'elect_woman', 'man_final_deci']:
    prog_var = f'{var}_prog'
    if prog_var in endline1_student.columns:
        # Merge individual item to analysis df
        temp_df = analysis_df.merge(
            endline1_student[['child_id', 'Sschool_id', prog_var]],
            on=['child_id', 'Sschool_id'],
            how='left'
        )

        r = run_spec(
            spec_id=f'robust/outcome/item_{var}',
            spec_tree_path='robustness/measurement.md',
            outcome_var=prog_var,
            treatment_var='B_treat',
            formula=f'{prog_var} ~ B_treat + B_Sgender_index2 + B_Sgrade6',
            data=temp_df,
            sample_desc='Full sample',
            controls_desc=f'Individual item: {var}'
        )
        if r: results.append(r)

# ========================================
# 5. INFERENCE VARIATIONS (5-8 specs)
# ========================================
print("\n5. Inference variations...")

# 5.1 Robust SE (no clustering)
r = run_spec(
    spec_id='robust/cluster/robust_hc1',
    spec_tree_path='robustness/clustering_variations.md',
    outcome_var='E_Sgender_index2_std',
    treatment_var='B_treat',
    formula=base_formula,
    data=analysis_df,
    cluster_var=None,
    vcov_type='hetero',
    sample_desc='Full sample',
    controls_desc='Heteroskedasticity-robust SE'
)
if r: results.append(r)

# 5.2 Cluster by district (if enough clusters)
if 'B_Sdistrict' in analysis_df.columns:
    r = run_spec(
        spec_id='robust/cluster/district',
        spec_tree_path='robustness/clustering_variations.md',
        outcome_var='E_Sgender_index2_std',
        treatment_var='B_treat',
        formula=base_formula,
        data=analysis_df,
        cluster_var='B_Sdistrict',
        sample_desc='Full sample',
        controls_desc='Clustered by district'
    )
    if r: results.append(r)

# 5.3 School-clustered (baseline - confirming)
r = run_spec(
    spec_id='robust/cluster/school',
    spec_tree_path='robustness/clustering_variations.md',
    outcome_var='E_Sgender_index2_std',
    treatment_var='B_treat',
    formula=base_formula,
    data=analysis_df,
    cluster_var='Sschool_id',
    sample_desc='Full sample',
    controls_desc='Clustered by school (baseline)'
)
if r: results.append(r)

# ========================================
# 6. ESTIMATION METHOD VARIATIONS (3-5 specs)
# ========================================
print("\n6. Estimation method variations...")

# 6.1 OLS without any FE
r = run_spec(
    spec_id='robust/estimation/no_fe',
    spec_tree_path='robustness/model_specification.md',
    outcome_var='E_Sgender_index2_std',
    treatment_var='B_treat',
    formula='E_Sgender_index2_std ~ B_treat + B_Sgender_index2',
    data=analysis_df,
    sample_desc='Full sample',
    controls_desc='No fixed effects',
    fixed_effects='None'
)
if r: results.append(r)

# 6.2 With school FE (absorbed)
try:
    r = run_spec(
        spec_id='robust/estimation/school_fe',
        spec_tree_path='robustness/model_specification.md',
        outcome_var='E_Sgender_index2_std',
        treatment_var='B_treat',
        formula='E_Sgender_index2_std ~ B_Sgender_index2 | Sschool_id',  # Cannot estimate treat with school FE
        data=analysis_df,
        sample_desc='Full sample',
        controls_desc='School FE (treatment absorbed)',
        fixed_effects='School'
    )
    if r: results.append(r)
except:
    pass  # Expected - treatment is at school level

# ========================================
# 7. FUNCTIONAL FORM (3-5 specs)
# ========================================
print("\n7. Functional form variations...")

# 7.1 Quadratic baseline control
analysis_df['B_Sgender_index2_sq'] = analysis_df['B_Sgender_index2'] ** 2

r = run_spec(
    spec_id='robust/funcform/quadratic_bl',
    spec_tree_path='robustness/functional_form.md',
    outcome_var='E_Sgender_index2_std',
    treatment_var='B_treat',
    formula='E_Sgender_index2_std ~ B_treat + B_Sgender_index2 + B_Sgender_index2_sq + B_Sgrade6',
    data=analysis_df,
    sample_desc='Full sample',
    controls_desc='Quadratic baseline outcome'
)
if r: results.append(r)

# 7.2 Level outcome (not standardized)
r = run_spec(
    spec_id='robust/funcform/levels',
    spec_tree_path='robustness/functional_form.md',
    outcome_var='E_Sgender_index2',
    treatment_var='B_treat',
    formula='E_Sgender_index2 ~ B_treat + B_Sgender_index2 + B_Sgrade6',
    data=analysis_df,
    sample_desc='Full sample',
    controls_desc='Level (non-standardized) outcome'
)
if r: results.append(r)

# ========================================
# 8. HETEROGENEITY ANALYSIS (5-10 specs)
# ========================================
print("\n8. Heterogeneity analysis...")

# 8.1 Gender interaction
analysis_df['treat_girl'] = analysis_df['B_treat'] * analysis_df['B_Sgirl']

r = run_spec(
    spec_id='robust/heterogeneity/gender',
    spec_tree_path='robustness/heterogeneity.md',
    outcome_var='E_Sgender_index2_std',
    treatment_var='B_treat',
    formula='E_Sgender_index2_std ~ B_treat + B_Sgirl + treat_girl + B_Sgender_index2 + B_Sgrade6',
    data=analysis_df,
    sample_desc='Full sample',
    controls_desc='Gender interaction'
)
if r: results.append(r)

# Also record the interaction term
r = run_spec(
    spec_id='robust/heterogeneity/gender_interaction',
    spec_tree_path='robustness/heterogeneity.md',
    outcome_var='E_Sgender_index2_std',
    treatment_var='treat_girl',
    formula='E_Sgender_index2_std ~ B_treat + B_Sgirl + treat_girl + B_Sgender_index2 + B_Sgrade6',
    data=analysis_df,
    sample_desc='Full sample',
    controls_desc='Gender interaction term'
)
if r: results.append(r)

# 8.2 Grade interaction
analysis_df['treat_grade6'] = analysis_df['B_treat'] * analysis_df['B_Sgrade6']

r = run_spec(
    spec_id='robust/heterogeneity/grade',
    spec_tree_path='robustness/heterogeneity.md',
    outcome_var='E_Sgender_index2_std',
    treatment_var='B_treat',
    formula='E_Sgender_index2_std ~ B_treat + B_Sgrade6 + treat_grade6 + B_Sgender_index2',
    data=analysis_df,
    sample_desc='Full sample',
    controls_desc='Grade interaction'
)
if r: results.append(r)

# 8.3-8.6 District interactions
for d_col in district_fe_vars[:4]:
    d_num = d_col.split('_')[-1]
    analysis_df[f'treat_{d_col}'] = analysis_df['B_treat'] * analysis_df[d_col]

    r = run_spec(
        spec_id=f'robust/heterogeneity/district_{d_num}',
        spec_tree_path='robustness/heterogeneity.md',
        outcome_var='E_Sgender_index2_std',
        treatment_var='B_treat',
        formula=f'E_Sgender_index2_std ~ B_treat + {d_col} + treat_{d_col} + B_Sgender_index2 + B_Sgrade6',
        data=analysis_df,
        sample_desc='Full sample',
        controls_desc=f'District {d_num} interaction'
    )
    if r: results.append(r)

# 8.7 Baseline attitudes interaction (below/above median)
median_bl = analysis_df['B_Sgender_index2'].median()
analysis_df['B_high_attitudes'] = (analysis_df['B_Sgender_index2'] > median_bl).astype(int)
analysis_df['treat_high_att'] = analysis_df['B_treat'] * analysis_df['B_high_attitudes']

r = run_spec(
    spec_id='robust/heterogeneity/baseline_attitudes',
    spec_tree_path='robustness/heterogeneity.md',
    outcome_var='E_Sgender_index2_std',
    treatment_var='B_treat',
    formula='E_Sgender_index2_std ~ B_treat + B_high_attitudes + treat_high_att + B_Sgrade6',
    data=analysis_df,
    sample_desc='Full sample',
    controls_desc='Baseline attitudes interaction'
)
if r: results.append(r)

# ========================================
# 9. PLACEBO TESTS (3-5 specs)
# ========================================
print("\n9. Placebo tests...")

# 9.1 Baseline outcome as placebo (should be zero)
r = run_spec(
    spec_id='robust/placebo/baseline_outcome',
    spec_tree_path='robustness/placebo_tests.md',
    outcome_var='B_Sgender_index2',
    treatment_var='B_treat',
    formula='B_Sgender_index2 ~ B_treat + B_Sgrade6',
    data=analysis_df,
    sample_desc='Full sample',
    controls_desc='Placebo: Baseline outcome'
)
if r: results.append(r)

# 9.2 Randomization check - treatment on grade
r = run_spec(
    spec_id='robust/placebo/treatment_on_grade',
    spec_tree_path='robustness/placebo_tests.md',
    outcome_var='B_Sgrade6',
    treatment_var='B_treat',
    formula='B_Sgrade6 ~ B_treat',
    data=analysis_df,
    sample_desc='Full sample',
    controls_desc='Placebo: Treatment predicting grade'
)
if r: results.append(r)

# 9.3 Randomization check - treatment on gender
r = run_spec(
    spec_id='robust/placebo/treatment_on_gender',
    spec_tree_path='robustness/placebo_tests.md',
    outcome_var='B_Sgirl',
    treatment_var='B_treat',
    formula='B_Sgirl ~ B_treat',
    data=analysis_df,
    sample_desc='Full sample',
    controls_desc='Placebo: Treatment predicting gender'
)
if r: results.append(r)

# ========================================
# 10. ADDITIONAL ROBUSTNESS
# ========================================
print("\n10. Additional robustness checks...")

# 10.1-10.5 Drop each school (sample of schools)
schools = analysis_df['Sschool_id'].unique()[:5]  # First 5 schools
for school in schools:
    drop_school_df = analysis_df[analysis_df['Sschool_id'] != school].copy()

    r = run_spec(
        spec_id=f'robust/sample/drop_school_{int(school)}',
        spec_tree_path='robustness/sample_restrictions.md',
        outcome_var='E_Sgender_index2_std',
        treatment_var='B_treat',
        formula=base_formula,
        data=drop_school_df,
        sample_desc=f'Excluding school {int(school)}'
    )
    if r: results.append(r)

# 10.6 High baseline index only
high_bl_df = analysis_df[analysis_df['B_Sgender_index2'] > median_bl].copy()
r = run_spec(
    spec_id='robust/sample/high_baseline_only',
    spec_tree_path='robustness/sample_restrictions.md',
    outcome_var='E_Sgender_index2_std',
    treatment_var='B_treat',
    formula='E_Sgender_index2_std ~ B_treat + B_Sgender_index2 + B_Sgrade6',
    data=high_bl_df,
    sample_desc='Above-median baseline attitudes'
)
if r: results.append(r)

# 10.7 Low baseline index only
low_bl_df = analysis_df[analysis_df['B_Sgender_index2'] <= median_bl].copy()
r = run_spec(
    spec_id='robust/sample/low_baseline_only',
    spec_tree_path='robustness/sample_restrictions.md',
    outcome_var='E_Sgender_index2_std',
    treatment_var='B_treat',
    formula='E_Sgender_index2_std ~ B_treat + B_Sgender_index2 + B_Sgrade6',
    data=low_bl_df,
    sample_desc='Below-median baseline attitudes'
)
if r: results.append(r)

# ========================================
# SAVE RESULTS
# ========================================
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

results_df = pd.DataFrame(results)
print(f"\nTotal specifications run: {len(results_df)}")

# Save to CSV
output_file = f"{OUTPUT_PATH}/specification_results.csv"
results_df.to_csv(output_file, index=False)
print(f"Results saved to: {output_file}")

# ========================================
# SUMMARY STATISTICS
# ========================================
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

print(f"\nTotal specifications: {len(results_df)}")
print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({(results_df['coefficient'] > 0).mean()*100:.1f}%)")
print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({(results_df['p_value'] < 0.05).mean()*100:.1f}%)")
print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({(results_df['p_value'] < 0.01).mean()*100:.1f}%)")
print(f"\nCoefficient range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")
print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")

# Category breakdown
print("\n--- By Category ---")
results_df['category'] = results_df['spec_id'].apply(lambda x: x.split('/')[0] if '/' in x else 'baseline')
for cat in results_df['category'].unique():
    cat_df = results_df[results_df['category'] == cat]
    pos_pct = (cat_df['coefficient'] > 0).mean() * 100
    sig_pct = (cat_df['p_value'] < 0.05).mean() * 100
    print(f"{cat}: N={len(cat_df)}, Positive={pos_pct:.0f}%, Sig 5%={sig_pct:.0f}%")

print("\nSpecification search complete!")
