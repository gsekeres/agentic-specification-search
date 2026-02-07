"""
Specification Search for Paper 184341-V2
Title: Emotional and behavioral impacts of homeschooling support on children
Authors: Hashibul Hassan, Asad Islam, Abu Siddique, Liang Choon Wang
Journal: AER: P&P (2025)

Method: Cross-sectional OLS with randomized treatment (RCT)
Treatment: Telementoring program (treat)
Outcomes: SDQ subscales (emotional, conduct, hyperactivity, peer, total difficulties)
Two endlines: 1-month (Endline 1) and 1-year (Endline 2)
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Configuration
PAPER_ID = "184341-V2"
JOURNAL = "AER: P&P"
PAPER_TITLE = "Emotional and behavioral impacts of homeschooling support on children"

# File paths
BASE_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search"
DATA_PATH = f"{BASE_PATH}/data/downloads/extracted/184341-V2/Detailed Replication Package V3.0/Raw data"
OUTPUT_PATH = f"{BASE_PATH}/data/downloads/extracted/184341-V2"

# Load and prepare data
print("Loading data...")

# Load the main sample and ECD data
sample_df = pd.read_stata(f"{DATA_PATH}/Sample.dta")
ecd_df = pd.read_stata(f"{DATA_PATH}/tm_ecd_data.dta")

# Load Excel files
rapid_baseline = pd.read_excel(f"{DATA_PATH}/Rapid-baseline-data.xlsx")
e1_parent = pd.read_excel(f"{DATA_PATH}/Telementoring-E1-Parent-Survey.xlsx")
e1_assess = pd.read_excel(f"{DATA_PATH}/Telementoring-E1-Assessment.xlsx")
e2_parent = pd.read_excel(f"{DATA_PATH}/Telementoring-E2-Parent-Survey.xlsx")
e2_assess = pd.read_excel(f"{DATA_PATH}/Telementoring-E2-Assessment.xlsx")

print(f"Sample shape: {sample_df.shape}")
print(f"ECD shape: {ecd_df.shape}")
print(f"E1 parent shape: {e1_parent.shape}")
print(f"E2 parent shape: {e2_parent.shape}")

# Standardize column names
rapid_baseline = rapid_baseline.rename(columns={'rsq2_1': 'CHILD_ID', 'rsq2_4': 'pvt_tutor', 'rsq2_23': 'number_child'})

# Rename SDQ columns in E1 parent survey to match Stata code expectations
e1_parent_renamed = e1_parent.copy()
for col in e1_parent.columns:
    if col.startswith('tm_sdq'):
        num = col.replace('tm_sdq', '')
        e1_parent_renamed = e1_parent_renamed.rename(columns={col: f'tm_e1_sdq{num}'})
e1_parent = e1_parent_renamed

# Rename columns in E1 assessment
e1_assess = e1_assess.rename(columns={'tm_cogr1_grade': 'tm_e1_grade', 'tm_cogr1_gen': 'tm_e1_gender'})

# Rename SDQ columns in E2 parent survey
e2_parent_renamed = e2_parent.copy()
for col in e2_parent.columns:
    if col.startswith('anu_sdq'):
        num = col.replace('anu_sdq', '')
        e2_parent_renamed = e2_parent_renamed.rename(columns={col: f'tm_e2_sdq{num}'})
e2_parent = e2_parent_renamed

# Rename columns in E2 assessment
e2_assess = e2_assess.rename(columns={'anu_e1_grade': 'tm_e2_grade', 'anu_e1_gender': 'tm_e2_gender'})

print(f"E1 parent columns after rename: {[c for c in e1_parent.columns if 'sdq' in c][:5]}...")
print(f"E2 parent columns after rename: {[c for c in e2_parent.columns if 'sdq' in c][:5]}...")

# Merge all datasets
print("\nMerging datasets...")
df = ecd_df.copy()
df = df.merge(sample_df[['CHILD_ID', 'newtreat']], on='CHILD_ID', how='left')
df = df.rename(columns={'newtreat': 'treat'})

# Merge rapid baseline
cols_to_merge = ['CHILD_ID', 'pvt_tutor', 'number_child']
cols_available = [c for c in cols_to_merge if c in rapid_baseline.columns]
df = df.merge(rapid_baseline[cols_available], on='CHILD_ID', how='left')

# Merge E1 parent survey
df = df.merge(e1_parent, on='CHILD_ID', how='left')

# Merge E1 assessment
df = df.merge(e1_assess, on='CHILD_ID', how='left')

# Merge E2 parent survey
df = df.merge(e2_parent, on='CHILD_ID', how='left')

# Merge E2 assessment
df = df.merge(e2_assess, on='CHILD_ID', how='left')

print(f"Merged data shape: {df.shape}")

# Create derived variables
print("\nCreating derived variables...")

# Child age calculation
if 'child_dob' in df.columns:
    target_date = pd.Timestamp('2020-09-01')
    df['child_dob'] = pd.to_datetime(df['child_dob'], errors='coerce')
    df['child_age'] = (target_date - df['child_dob']).dt.days / 365.25

# Father's education in years
if 'fathers_edu' in df.columns:
    edu_map = {1: 0, 2: 2, 3: 5, 4: 8, 5: 10, 6: 11, 7: 12, 8: 14, 9: 18, 10: 0}
    df['FEdun'] = df['fathers_edu'].map(edu_map)

# Mother's education in years
if 'mothers_edu' in df.columns:
    edu_map = {1: 0, 2: 2, 3: 5, 4: 8, 5: 10, 6: 11, 7: 12, 8: 14, 9: 18, 10: 0}
    df['MEdun'] = df['mothers_edu'].map(edu_map)

# Total family income
if 'fathers_income' in df.columns and 'mothers_income' in df.columns:
    df['total_income'] = df['fathers_income'].fillna(0) + df['mothers_income'].fillna(0)

# Number of children
if 'number_child' in df.columns:
    df['children_no'] = 1 + df['number_child'].fillna(0)

# Religion dummy
if 'hh_reli' in df.columns:
    df['reli_dummy'] = (df['hh_reli'] == 1).astype(float)

# Gender variable
df['gender'] = np.nan
if 'tm_e1_gender' in df.columns:
    df['gender'] = df['tm_e1_gender']
if 'tm_e2_gender' in df.columns:
    df.loc[df['gender'].isna(), 'gender'] = df.loc[df['gender'].isna(), 'tm_e2_gender']
if 'child_gen' in df.columns:
    df.loc[df['gender'].isna(), 'gender'] = df.loc[df['gender'].isna(), 'child_gen']

# ============================================================================
# CREATE SDQ SUBSCALES - Following exact Stata code logic
# ============================================================================
print("\nCreating SDQ subscales...")

def create_sdq_for_endline(df, prefix, endline_suffix):
    """
    Create SDQ subscales following the exact Stata code logic.
    prefix: 'tm_e1_' or 'tm_e2_'
    endline_suffix: '_e1' or '_e2'
    """

    # Create reverse-coded items
    # qobeys: pobeys (sdq7) recoded 0->2, 1->1, 2->0
    if f'{prefix}sdq7' in df.columns:
        df['qobeys'] = df[f'{prefix}sdq7'].map({0: 2, 1: 1, 2: 0})
    # qreflect: preflect (sdq21) recoded
    if f'{prefix}sdq21' in df.columns:
        df['qreflect'] = df[f'{prefix}sdq21'].map({0: 2, 1: 1, 2: 0})
    # qattends: pattends (sdq25) recoded
    if f'{prefix}sdq25' in df.columns:
        df['qattends'] = df[f'{prefix}sdq25'].map({0: 2, 1: 1, 2: 0})
    # qfriend: pfriend (sdq11) recoded
    if f'{prefix}sdq11' in df.columns:
        df['qfriend'] = df[f'{prefix}sdq11'].map({0: 2, 1: 1, 2: 0})
    # qpopular: ppopular (sdq14) recoded
    if f'{prefix}sdq14' in df.columns:
        df['qpopular'] = df[f'{prefix}sdq14'].map({0: 2, 1: 1, 2: 0})

    # Emotional symptoms: psomatic(3), pworries(8), punhappy(13), pclingy(16), pafraid(24)
    emo_items = [f'{prefix}sdq3', f'{prefix}sdq8', f'{prefix}sdq13', f'{prefix}sdq16', f'{prefix}sdq24']
    emo_items = [c for c in emo_items if c in df.columns]
    if len(emo_items) >= 3:
        n_emo = df[emo_items].notna().sum(axis=1)
        df[f'sdq_emotion{endline_suffix}'] = np.where(
            n_emo > 2,
            (df[emo_items].mean(axis=1) * 5).round(),
            np.nan
        )

    # Conduct problems: ptantrum(5), qobeys, pfights(12), plies(18), psteals(22)
    cond_items = [f'{prefix}sdq5', 'qobeys', f'{prefix}sdq12', f'{prefix}sdq18', f'{prefix}sdq22']
    cond_items = [c for c in cond_items if c in df.columns]
    if len(cond_items) >= 3:
        n_cond = df[cond_items].notna().sum(axis=1)
        df[f'sdq_conduct{endline_suffix}'] = np.where(
            n_cond > 2,
            (df[cond_items].mean(axis=1) * 5).round(),
            np.nan
        )

    # Hyperactivity: prestles(2), pfidgety(10), pdistrac(15), qreflect, qattends
    hyper_items = [f'{prefix}sdq2', f'{prefix}sdq10', f'{prefix}sdq15', 'qreflect', 'qattends']
    hyper_items = [c for c in hyper_items if c in df.columns]
    if len(hyper_items) >= 3:
        n_hyper = df[hyper_items].notna().sum(axis=1)
        df[f'sdq_hyper{endline_suffix}'] = np.where(
            n_hyper > 2,
            (df[hyper_items].mean(axis=1) * 5).round(),
            np.nan
        )

    # Peer problems: ploner(6), qfriend, qpopular, pbullied(19), poldbest(23)
    peer_items = [f'{prefix}sdq6', 'qfriend', 'qpopular', f'{prefix}sdq19', f'{prefix}sdq23']
    peer_items = [c for c in peer_items if c in df.columns]
    if len(peer_items) >= 3:
        n_peer = df[peer_items].notna().sum(axis=1)
        df[f'sdq_peer{endline_suffix}'] = np.where(
            n_peer > 2,
            (df[peer_items].mean(axis=1) * 5).round(),
            np.nan
        )

    # Drop temporary variables
    for col in ['qobeys', 'qreflect', 'qattends', 'qfriend', 'qpopular']:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df

# Create SDQ for Endline 1
df = create_sdq_for_endline(df, 'tm_e1_', '_e1')

# Create SDQ for Endline 2
df = create_sdq_for_endline(df, 'tm_e2_', '_e2')

# Create total difficulties scores
for suffix in ['_e1', '_e2']:
    emo_col = f'sdq_emotion{suffix}'
    cond_col = f'sdq_conduct{suffix}'
    hyper_col = f'sdq_hyper{suffix}'
    peer_col = f'sdq_peer{suffix}'

    cols_exist = [c for c in [emo_col, cond_col, hyper_col, peer_col] if c in df.columns]
    if len(cols_exist) == 4:
        df[f'sdq_totdiff{suffix}'] = df[cols_exist].sum(axis=1)

# Create completion indicators
df['e1_comp'] = 0
if 'tm_e1_grade' in df.columns and 'hh_reli' in df.columns:
    df['e1_comp'] = (df['tm_e1_grade'].notna() & df['hh_reli'].notna()).astype(int)

df['e2_comp'] = 0
if 'tm_e2_grade' in df.columns and 'tm_e2_sdq2' in df.columns:
    df['e2_comp'] = (df['tm_e2_grade'].notna() & df['tm_e2_sdq2'].notna()).astype(int)

print(f"E1 complete: {df['e1_comp'].sum()}, E2 complete: {df['e2_comp'].sum()}")

# Check SDQ outcomes
print(f"\nSDQ columns created: {[c for c in df.columns if 'sdq_' in c]}")
print(f"SDQ totdiff E1 non-null: {df['sdq_totdiff_e1'].notna().sum() if 'sdq_totdiff_e1' in df.columns else 0}")
print(f"SDQ totdiff E2 non-null: {df['sdq_totdiff_e2'].notna().sum() if 'sdq_totdiff_e2' in df.columns else 0}")

# Define control variables
control_list = ['child_age', 'gender', 'baseline_literacy_score', 'baseline_numeracy_score',
                'pvt_tutor', 'birth_order', 'children_no', 'FEdun', 'MEdun', 'total_income', 'reli_dummy']

# Filter to available controls
available_controls = [c for c in control_list if c in df.columns and df[c].notna().sum() > 10]
print(f"\nAvailable controls: {available_controls}")

# Define outcomes
outcomes_e1 = ['sdq_totdiff_e1', 'sdq_emotion_e1', 'sdq_conduct_e1', 'sdq_hyper_e1', 'sdq_peer_e1']
outcomes_e2 = ['sdq_totdiff_e2', 'sdq_emotion_e2', 'sdq_conduct_e2', 'sdq_hyper_e2', 'sdq_peer_e2']
outcomes_e1 = [o for o in outcomes_e1 if o in df.columns and df[o].notna().sum() > 30]
outcomes_e2 = [o for o in outcomes_e2 if o in df.columns and df[o].notna().sum() > 30]

print(f"E1 outcomes: {outcomes_e1}")
print(f"E2 outcomes: {outcomes_e2}")

# ============================================================================
# SPECIFICATION SEARCH
# ============================================================================

results = []

def run_regression(df_subset, outcome, treatment, controls, fe_vars=None, cluster_var=None,
                   spec_id='', spec_tree_path='', model_type='OLS', sample_desc='',
                   vcov_type='HC1'):
    """Run regression and return results dictionary"""

    # Prepare data
    df_reg = df_subset.copy()

    # Build regression variables list
    all_vars = [outcome, treatment]
    if controls:
        all_vars.extend(controls)
    if fe_vars:
        all_vars.extend(fe_vars)
    if cluster_var and cluster_var not in all_vars:
        all_vars.append(cluster_var)

    # Drop rows with missing values in key variables
    df_reg = df_reg.dropna(subset=[v for v in all_vars if v in df_reg.columns])

    if len(df_reg) < 30:
        return None

    try:
        # Build the regression formula
        # Start with treatment and controls
        rhs_vars = [treatment]
        if controls:
            rhs_vars.extend(controls)

        # Add fixed effects as categorical dummies
        fe_dummies_added = []
        if fe_vars:
            for fe in fe_vars:
                if fe in df_reg.columns and df_reg[fe].nunique() > 1:
                    # Use C() for categorical treatment in formula
                    rhs_vars.append(f'C({fe})')
                    fe_dummies_added.append(fe)

        formula = f'{outcome} ~ ' + ' + '.join(rhs_vars)

        # Fit the model
        model = smf.ols(formula, data=df_reg).fit(cov_type=vcov_type)

        coef = model.params[treatment]
        se = model.bse[treatment]
        pval = model.pvalues[treatment]
        t_stat = model.tvalues[treatment]
        ci_lower = model.conf_int().loc[treatment, 0]
        ci_upper = model.conf_int().loc[treatment, 1]
        n_obs = int(model.nobs)
        r2 = model.rsquared

        # Coefficient vector
        coef_vec = {
            "treatment": {"var": treatment, "coef": float(coef), "se": float(se), "pval": float(pval)},
            "controls": [],
            "fixed_effects": fe_dummies_added,
            "diagnostics": {"r_squared": float(r2), "f_stat": float(model.fvalue) if model.fvalue else None}
        }
        for ctrl in (controls if controls else []):
            if ctrl in model.params.index:
                coef_vec["controls"].append({
                    "var": ctrl,
                    "coef": float(model.params[ctrl]),
                    "se": float(model.bse[ctrl]),
                    "pval": float(model.pvalues[ctrl])
                })

        return {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': outcome,
            'treatment_var': treatment,
            'coefficient': float(coef),
            'std_error': float(se),
            't_stat': float(t_stat),
            'p_value': float(pval),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_obs': int(n_obs),
            'r_squared': float(r2),
            'coefficient_vector_json': json.dumps(coef_vec),
            'sample_desc': sample_desc,
            'fixed_effects': ', '.join(fe_dummies_added) if fe_dummies_added else 'None',
            'controls_desc': ', '.join(controls) if controls else 'None',
            'cluster_var': cluster_var if cluster_var else 'None',
            'model_type': model_type,
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
    except Exception as e:
        print(f"Error in {spec_id}: {e}")
        return None

# ============================================================================
# RUN SPECIFICATIONS
# ============================================================================

print("\n" + "="*80)
print("RUNNING SPECIFICATION SEARCH")
print("="*80)

# Use E1 sample for main analysis
df_e1 = df[df['e1_comp'] == 1].copy()
df_e2 = df[df['e2_comp'] == 1].copy()

# FE variables
fe_vars_e1 = ['tm_e1_grade', 'union_code']
fe_vars_e1 = [f for f in fe_vars_e1 if f in df.columns]
fe_vars_e2 = ['tm_e2_grade', 'union_code']
fe_vars_e2 = [f for f in fe_vars_e2 if f in df.columns]

# Main outcome for detailed analysis
main_outcome_e1 = 'sdq_totdiff_e1' if 'sdq_totdiff_e1' in outcomes_e1 else (outcomes_e1[0] if outcomes_e1 else None)
main_outcome_e2 = 'sdq_totdiff_e2' if 'sdq_totdiff_e2' in outcomes_e2 else (outcomes_e2[0] if outcomes_e2 else None)

print(f"\nMain outcome E1: {main_outcome_e1}")
print(f"Main outcome E2: {main_outcome_e2}")

spec_count = 0

# ---------------------------------------------------------------------------
# 1. BASELINE SPECIFICATIONS (5 outcomes x 2 endlines = 10 specs)
# ---------------------------------------------------------------------------
print("\n1. Running baseline specifications...")

for outcome in outcomes_e1:
    result = run_regression(
        df_e1, outcome, 'treat', available_controls, fe_vars_e1, None,
        spec_id=f'baseline/{outcome}',
        spec_tree_path='methods/cross_sectional_ols.md#baseline',
        sample_desc='Endline 1 complete sample'
    )
    if result:
        results.append(result)
        spec_count += 1
        print(f"   {outcome}: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}, n={result['n_obs']}")

for outcome in outcomes_e2:
    result = run_regression(
        df_e2, outcome, 'treat', available_controls, fe_vars_e2, None,
        spec_id=f'baseline/{outcome}',
        spec_tree_path='methods/cross_sectional_ols.md#baseline',
        sample_desc='Endline 2 complete sample'
    )
    if result:
        results.append(result)
        spec_count += 1
        print(f"   {outcome}: coef={result['coefficient']:.3f}, p={result['p_value']:.4f}, n={result['n_obs']}")

print(f"   Baseline specs: {spec_count}")

# ---------------------------------------------------------------------------
# 2. CONTROL VARIATIONS (~15 specs)
# ---------------------------------------------------------------------------
print("\n2. Running control variations...")
control_start = spec_count

if main_outcome_e1 and available_controls:
    # No controls
    result = run_regression(
        df_e1, main_outcome_e1, 'treat', [], fe_vars_e1, None,
        spec_id='robust/control/none',
        spec_tree_path='robustness/leave_one_out.md',
        sample_desc='E1 - no controls'
    )
    if result:
        results.append(result)
        spec_count += 1

    # Leave-one-out for each control
    for ctrl in available_controls:
        remaining = [c for c in available_controls if c != ctrl]
        result = run_regression(
            df_e1, main_outcome_e1, 'treat', remaining, fe_vars_e1, None,
            spec_id=f'robust/loo/drop_{ctrl}',
            spec_tree_path='robustness/leave_one_out.md',
            sample_desc=f'E1 - drop {ctrl}'
        )
        if result:
            results.append(result)
            spec_count += 1

    # Add controls incrementally
    for i, ctrl in enumerate(available_controls):
        controls_so_far = available_controls[:i+1]
        result = run_regression(
            df_e1, main_outcome_e1, 'treat', controls_so_far, fe_vars_e1, None,
            spec_id=f'robust/control/add_{ctrl}',
            spec_tree_path='robustness/control_progression.md',
            sample_desc=f'E1 - add controls up to {ctrl}'
        )
        if result:
            results.append(result)
            spec_count += 1

print(f"   Control variation specs: {spec_count - control_start}")

# ---------------------------------------------------------------------------
# 3. SAMPLE RESTRICTIONS (~15 specs)
# ---------------------------------------------------------------------------
print("\n3. Running sample restrictions...")
sample_start = spec_count

if main_outcome_e1:
    # By gender
    for gender_val, gender_name in [(0, 'female'), (1, 'male')]:
        if 'gender' in df_e1.columns:
            df_sub = df_e1[df_e1['gender'] == gender_val]
            if len(df_sub) > 30:
                result = run_regression(
                    df_sub, main_outcome_e1, 'treat', available_controls, fe_vars_e1, None,
                    spec_id=f'robust/sample/{gender_name}_only',
                    spec_tree_path='robustness/sample_restrictions.md',
                    sample_desc=f'E1 - {gender_name} only'
                )
                if result:
                    results.append(result)
                    spec_count += 1

    # By child age (median split)
    if 'child_age' in df_e1.columns and df_e1['child_age'].notna().sum() > 30:
        median_age = df_e1['child_age'].median()
        for age_cond, age_name in [(df_e1['child_age'] <= median_age, 'young'),
                                    (df_e1['child_age'] > median_age, 'old')]:
            df_sub = df_e1[age_cond]
            if len(df_sub) > 30:
                result = run_regression(
                    df_sub, main_outcome_e1, 'treat', available_controls, fe_vars_e1, None,
                    spec_id=f'robust/sample/{age_name}',
                    spec_tree_path='robustness/sample_restrictions.md',
                    sample_desc=f'E1 - {age_name} children'
                )
                if result:
                    results.append(result)
                    spec_count += 1

    # By parent education (median split)
    if 'FEdun' in df_e1.columns and df_e1['FEdun'].notna().sum() > 30:
        median_edu = df_e1['FEdun'].median()
        for edu_cond, edu_name in [(df_e1['FEdun'] <= median_edu, 'low_edu'),
                                    (df_e1['FEdun'] > median_edu, 'high_edu')]:
            df_sub = df_e1[edu_cond]
            if len(df_sub) > 30:
                result = run_regression(
                    df_sub, main_outcome_e1, 'treat', available_controls, fe_vars_e1, None,
                    spec_id=f'robust/sample/{edu_name}',
                    spec_tree_path='robustness/sample_restrictions.md',
                    sample_desc=f'E1 - {edu_name} father'
                )
                if result:
                    results.append(result)
                    spec_count += 1

    # By income (median split)
    if 'total_income' in df_e1.columns and df_e1['total_income'].notna().sum() > 30:
        median_inc = df_e1['total_income'].median()
        for inc_cond, inc_name in [(df_e1['total_income'] <= median_inc, 'low_income'),
                                    (df_e1['total_income'] > median_inc, 'high_income')]:
            df_sub = df_e1[inc_cond]
            if len(df_sub) > 30:
                result = run_regression(
                    df_sub, main_outcome_e1, 'treat', available_controls, fe_vars_e1, None,
                    spec_id=f'robust/sample/{inc_name}',
                    spec_tree_path='robustness/sample_restrictions.md',
                    sample_desc=f'E1 - {inc_name}'
                )
                if result:
                    results.append(result)
                    spec_count += 1

    # By religion
    if 'reli_dummy' in df_e1.columns:
        for reli_val, reli_name in [(1, 'muslim'), (0, 'non_muslim')]:
            df_sub = df_e1[df_e1['reli_dummy'] == reli_val]
            if len(df_sub) > 30:
                result = run_regression(
                    df_sub, main_outcome_e1, 'treat', available_controls, fe_vars_e1, None,
                    spec_id=f'robust/sample/{reli_name}',
                    spec_tree_path='robustness/sample_restrictions.md',
                    sample_desc=f'E1 - {reli_name} households'
                )
                if result:
                    results.append(result)
                    spec_count += 1

    # Winsorize outcome at different levels
    for pct in [1, 5, 10]:
        df_wins = df_e1.copy()
        lower = df_wins[main_outcome_e1].quantile(pct/100)
        upper = df_wins[main_outcome_e1].quantile(1 - pct/100)
        df_wins[main_outcome_e1] = df_wins[main_outcome_e1].clip(lower=lower, upper=upper)
        result = run_regression(
            df_wins, main_outcome_e1, 'treat', available_controls, fe_vars_e1, None,
            spec_id=f'robust/sample/winsor_{pct}pct',
            spec_tree_path='robustness/sample_restrictions.md',
            sample_desc=f'E1 - winsorized {pct}%'
        )
        if result:
            results.append(result)
            spec_count += 1

    # Trim outliers
    for pct in [1, 5]:
        df_trim = df_e1[
            (df_e1[main_outcome_e1] > df_e1[main_outcome_e1].quantile(pct/100)) &
            (df_e1[main_outcome_e1] < df_e1[main_outcome_e1].quantile(1 - pct/100))
        ]
        if len(df_trim) > 30:
            result = run_regression(
                df_trim, main_outcome_e1, 'treat', available_controls, fe_vars_e1, None,
                spec_id=f'robust/sample/trim_{pct}pct',
                spec_tree_path='robustness/sample_restrictions.md',
                sample_desc=f'E1 - trimmed {pct}%'
            )
            if result:
                results.append(result)
                spec_count += 1

print(f"   Sample restriction specs: {spec_count - sample_start}")

# ---------------------------------------------------------------------------
# 4. ALTERNATIVE OUTCOMES (~10 specs)
# ---------------------------------------------------------------------------
print("\n4. Running alternative outcomes...")
outcome_start = spec_count

if main_outcome_e1:
    # Standardized outcome
    df_std = df_e1.copy()
    df_std[f'{main_outcome_e1}_std'] = (df_std[main_outcome_e1] - df_std[main_outcome_e1].mean()) / df_std[main_outcome_e1].std()
    result = run_regression(
        df_std, f'{main_outcome_e1}_std', 'treat', available_controls, fe_vars_e1, None,
        spec_id='robust/outcome/standardized',
        spec_tree_path='robustness/functional_form.md',
        sample_desc='E1 - standardized outcome'
    )
    if result:
        results.append(result)
        spec_count += 1

    # Log transformation (if positive)
    if df_e1[main_outcome_e1].min() >= 0:
        df_log = df_e1.copy()
        df_log[f'{main_outcome_e1}_log'] = np.log(df_log[main_outcome_e1] + 1)
        result = run_regression(
            df_log, f'{main_outcome_e1}_log', 'treat', available_controls, fe_vars_e1, None,
            spec_id='robust/form/y_log',
            spec_tree_path='robustness/functional_form.md',
            sample_desc='E1 - log(outcome+1)'
        )
        if result:
            results.append(result)
            spec_count += 1

        # Asinh transformation
        df_asinh = df_e1.copy()
        df_asinh[f'{main_outcome_e1}_asinh'] = np.arcsinh(df_asinh[main_outcome_e1])
        result = run_regression(
            df_asinh, f'{main_outcome_e1}_asinh', 'treat', available_controls, fe_vars_e1, None,
            spec_id='robust/form/y_asinh',
            spec_tree_path='robustness/functional_form.md',
            sample_desc='E1 - arcsinh(outcome)'
        )
        if result:
            results.append(result)
            spec_count += 1

    # Binary outcome: above median difficulties
    df_bin = df_e1.copy()
    median_val = df_bin[main_outcome_e1].median()
    df_bin['high_difficulties'] = (df_bin[main_outcome_e1] > median_val).astype(int)
    result = run_regression(
        df_bin, 'high_difficulties', 'treat', available_controls, fe_vars_e1, None,
        spec_id='robust/outcome/binary_high',
        spec_tree_path='robustness/functional_form.md',
        sample_desc='E1 - binary: above median difficulties'
    )
    if result:
        results.append(result)
        spec_count += 1

print(f"   Alternative outcome specs: {spec_count - outcome_start}")

# ---------------------------------------------------------------------------
# 5. INFERENCE VARIATIONS (~8 specs)
# ---------------------------------------------------------------------------
print("\n5. Running inference variations...")
inference_start = spec_count

if main_outcome_e1:
    # Classical SE
    result = run_regression(
        df_e1, main_outcome_e1, 'treat', available_controls, fe_vars_e1, None,
        spec_id='robust/se/classical',
        spec_tree_path='robustness/clustering_variations.md',
        sample_desc='E1 - classical SE',
        vcov_type='nonrobust'
    )
    if result:
        results.append(result)
        spec_count += 1

    # HC2 robust SE
    result = run_regression(
        df_e1, main_outcome_e1, 'treat', available_controls, fe_vars_e1, None,
        spec_id='robust/se/hc2',
        spec_tree_path='robustness/clustering_variations.md',
        sample_desc='E1 - HC2 robust SE',
        vcov_type='HC2'
    )
    if result:
        results.append(result)
        spec_count += 1

    # HC3 robust SE
    result = run_regression(
        df_e1, main_outcome_e1, 'treat', available_controls, fe_vars_e1, None,
        spec_id='robust/se/hc3',
        spec_tree_path='robustness/clustering_variations.md',
        sample_desc='E1 - HC3 robust SE',
        vcov_type='HC3'
    )
    if result:
        results.append(result)
        spec_count += 1

    # Cluster by union_code - use robust SEs clustered at group level
    if 'union_code' in df_e1.columns:
        try:
            formula = f'{main_outcome_e1} ~ treat + ' + ' + '.join(available_controls)
            for fe in fe_vars_e1:
                if fe in df_e1.columns:
                    formula += f' + C({fe})'
            model_cluster = smf.ols(formula, data=df_e1.dropna(subset=[main_outcome_e1, 'treat'] + available_controls + fe_vars_e1)).fit(
                cov_type='cluster', cov_kwds={'groups': df_e1.dropna(subset=[main_outcome_e1, 'treat'] + available_controls + fe_vars_e1)['union_code']}
            )
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': 'robust/cluster/union',
                'spec_tree_path': 'robustness/clustering_variations.md',
                'outcome_var': main_outcome_e1,
                'treatment_var': 'treat',
                'coefficient': float(model_cluster.params['treat']),
                'std_error': float(model_cluster.bse['treat']),
                't_stat': float(model_cluster.tvalues['treat']),
                'p_value': float(model_cluster.pvalues['treat']),
                'ci_lower': float(model_cluster.conf_int().loc['treat', 0]),
                'ci_upper': float(model_cluster.conf_int().loc['treat', 1]),
                'n_obs': int(model_cluster.nobs),
                'r_squared': float(model_cluster.rsquared),
                'coefficient_vector_json': json.dumps({"treatment": {"var": "treat", "coef": float(model_cluster.params['treat']), "se": float(model_cluster.bse['treat']), "pval": float(model_cluster.pvalues['treat'])}}),
                'sample_desc': 'E1 - clustered by union',
                'fixed_effects': ', '.join(fe_vars_e1),
                'controls_desc': ', '.join(available_controls),
                'cluster_var': 'union_code',
                'model_type': 'OLS',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })
            spec_count += 1
        except Exception as e:
            print(f"   Error in cluster by union: {e}")

print(f"   Inference variation specs: {spec_count - inference_start}")

# ---------------------------------------------------------------------------
# 6. ESTIMATION METHOD VARIATIONS (~5 specs)
# ---------------------------------------------------------------------------
print("\n6. Running estimation method variations...")
method_start = spec_count

if main_outcome_e1:
    # No fixed effects
    result = run_regression(
        df_e1, main_outcome_e1, 'treat', available_controls, None, None,
        spec_id='robust/estimation/no_fe',
        spec_tree_path='methods/cross_sectional_ols.md',
        sample_desc='E1 - no fixed effects',
        model_type='OLS'
    )
    if result:
        results.append(result)
        spec_count += 1

    # Only grade FE
    if 'tm_e1_grade' in df.columns:
        result = run_regression(
            df_e1, main_outcome_e1, 'treat', available_controls, ['tm_e1_grade'], None,
            spec_id='robust/estimation/grade_fe_only',
            spec_tree_path='methods/cross_sectional_ols.md',
            sample_desc='E1 - only grade FE',
            model_type='OLS+FE'
        )
        if result:
            results.append(result)
            spec_count += 1

    # Only union FE
    if 'union_code' in df.columns:
        result = run_regression(
            df_e1, main_outcome_e1, 'treat', available_controls, ['union_code'], None,
            spec_id='robust/estimation/union_fe_only',
            spec_tree_path='methods/cross_sectional_ols.md',
            sample_desc='E1 - only union FE',
            model_type='OLS+FE'
        )
        if result:
            results.append(result)
            spec_count += 1

print(f"   Estimation method specs: {spec_count - method_start}")

# ---------------------------------------------------------------------------
# 7. HETEROGENEITY ANALYSIS (~10 specs)
# ---------------------------------------------------------------------------
print("\n7. Running heterogeneity analysis...")
het_start = spec_count

if main_outcome_e1:
    # Gender interaction
    if 'gender' in df_e1.columns:
        df_het = df_e1.copy()
        df_het['treat_x_gender'] = df_het['treat'] * df_het['gender']
        controls_het = available_controls + ['treat_x_gender']
        result = run_regression(
            df_het, main_outcome_e1, 'treat', controls_het, fe_vars_e1, None,
            spec_id='robust/het/interaction_gender',
            spec_tree_path='robustness/heterogeneity.md',
            sample_desc='E1 - treatment x gender interaction'
        )
        if result:
            results.append(result)
            spec_count += 1

    # Age interaction
    if 'child_age' in df_e1.columns and df_e1['child_age'].notna().sum() > 30:
        df_het = df_e1.copy()
        df_het['treat_x_age'] = df_het['treat'] * df_het['child_age']
        controls_het = available_controls + ['treat_x_age']
        result = run_regression(
            df_het, main_outcome_e1, 'treat', controls_het, fe_vars_e1, None,
            spec_id='robust/het/interaction_age',
            spec_tree_path='robustness/heterogeneity.md',
            sample_desc='E1 - treatment x age interaction'
        )
        if result:
            results.append(result)
            spec_count += 1

    # Parent education interaction
    if 'FEdun' in df_e1.columns and df_e1['FEdun'].notna().sum() > 30:
        df_het = df_e1.copy()
        df_het['treat_x_edu'] = df_het['treat'] * df_het['FEdun']
        controls_het = available_controls + ['treat_x_edu']
        result = run_regression(
            df_het, main_outcome_e1, 'treat', controls_het, fe_vars_e1, None,
            spec_id='robust/het/interaction_education',
            spec_tree_path='robustness/heterogeneity.md',
            sample_desc='E1 - treatment x father education interaction'
        )
        if result:
            results.append(result)
            spec_count += 1

    # Income interaction
    if 'total_income' in df_e1.columns and df_e1['total_income'].notna().sum() > 30:
        df_het = df_e1.copy()
        df_het['treat_x_income'] = df_het['treat'] * df_het['total_income']
        controls_het = available_controls + ['treat_x_income']
        result = run_regression(
            df_het, main_outcome_e1, 'treat', controls_het, fe_vars_e1, None,
            spec_id='robust/het/interaction_income',
            spec_tree_path='robustness/heterogeneity.md',
            sample_desc='E1 - treatment x income interaction'
        )
        if result:
            results.append(result)
            spec_count += 1

    # Baseline literacy interaction
    if 'baseline_literacy_score' in df_e1.columns and df_e1['baseline_literacy_score'].notna().sum() > 30:
        df_het = df_e1.copy()
        df_het['treat_x_literacy'] = df_het['treat'] * df_het['baseline_literacy_score']
        controls_het = available_controls + ['treat_x_literacy']
        result = run_regression(
            df_het, main_outcome_e1, 'treat', controls_het, fe_vars_e1, None,
            spec_id='robust/het/interaction_baseline_literacy',
            spec_tree_path='robustness/heterogeneity.md',
            sample_desc='E1 - treatment x baseline literacy interaction'
        )
        if result:
            results.append(result)
            spec_count += 1

    # Baseline numeracy interaction
    if 'baseline_numeracy_score' in df_e1.columns and df_e1['baseline_numeracy_score'].notna().sum() > 30:
        df_het = df_e1.copy()
        df_het['treat_x_numeracy'] = df_het['treat'] * df_het['baseline_numeracy_score']
        controls_het = available_controls + ['treat_x_numeracy']
        result = run_regression(
            df_het, main_outcome_e1, 'treat', controls_het, fe_vars_e1, None,
            spec_id='robust/het/interaction_baseline_numeracy',
            spec_tree_path='robustness/heterogeneity.md',
            sample_desc='E1 - treatment x baseline numeracy interaction'
        )
        if result:
            results.append(result)
            spec_count += 1

print(f"   Heterogeneity specs: {spec_count - het_start}")

# ---------------------------------------------------------------------------
# 8. ENDLINE 2 ROBUSTNESS (~10 specs)
# ---------------------------------------------------------------------------
print("\n8. Running Endline 2 robustness...")
e2_start = spec_count

if main_outcome_e2:
    # No controls
    result = run_regression(
        df_e2, main_outcome_e2, 'treat', [], fe_vars_e2, None,
        spec_id='robust/e2/no_controls',
        spec_tree_path='robustness/leave_one_out.md',
        sample_desc='E2 - no controls'
    )
    if result:
        results.append(result)
        spec_count += 1

    # By gender
    if 'gender' in df_e2.columns:
        for gender_val, gender_name in [(0, 'female'), (1, 'male')]:
            df_sub = df_e2[df_e2['gender'] == gender_val]
            if len(df_sub) > 30:
                result = run_regression(
                    df_sub, main_outcome_e2, 'treat', available_controls, fe_vars_e2, None,
                    spec_id=f'robust/e2/sample/{gender_name}_only',
                    spec_tree_path='robustness/sample_restrictions.md',
                    sample_desc=f'E2 - {gender_name} only'
                )
                if result:
                    results.append(result)
                    spec_count += 1

    # No FE
    result = run_regression(
        df_e2, main_outcome_e2, 'treat', available_controls, None, None,
        spec_id='robust/e2/estimation/no_fe',
        spec_tree_path='methods/cross_sectional_ols.md',
        sample_desc='E2 - no fixed effects'
    )
    if result:
        results.append(result)
        spec_count += 1

    # Winsorize
    for pct in [1, 5]:
        df_wins = df_e2.copy()
        lower = df_wins[main_outcome_e2].quantile(pct/100)
        upper = df_wins[main_outcome_e2].quantile(1 - pct/100)
        df_wins[main_outcome_e2] = df_wins[main_outcome_e2].clip(lower=lower, upper=upper)
        result = run_regression(
            df_wins, main_outcome_e2, 'treat', available_controls, fe_vars_e2, None,
            spec_id=f'robust/e2/sample/winsor_{pct}pct',
            spec_tree_path='robustness/sample_restrictions.md',
            sample_desc=f'E2 - winsorized {pct}%'
        )
        if result:
            results.append(result)
            spec_count += 1

print(f"   Endline 2 robustness specs: {spec_count - e2_start}")

# ---------------------------------------------------------------------------
# 9. ADDITIONAL ROBUSTNESS FOR SUBSCALES
# ---------------------------------------------------------------------------
print("\n9. Running additional subscale analyses...")
subscale_start = spec_count

# Run key robustness for each subscale
subscale_outcomes_e1 = [o for o in outcomes_e1 if o != main_outcome_e1]
for outcome in subscale_outcomes_e1:
    # No controls
    result = run_regression(
        df_e1, outcome, 'treat', [], fe_vars_e1, None,
        spec_id=f'robust/subscale/{outcome}/no_controls',
        spec_tree_path='robustness/leave_one_out.md',
        sample_desc=f'{outcome} - no controls'
    )
    if result:
        results.append(result)
        spec_count += 1

    # By gender
    if 'gender' in df_e1.columns:
        for gender_val, gender_name in [(0, 'female'), (1, 'male')]:
            df_sub = df_e1[df_e1['gender'] == gender_val]
            if len(df_sub) > 30:
                result = run_regression(
                    df_sub, outcome, 'treat', available_controls, fe_vars_e1, None,
                    spec_id=f'robust/subscale/{outcome}/{gender_name}',
                    spec_tree_path='robustness/sample_restrictions.md',
                    sample_desc=f'{outcome} - {gender_name} only'
                )
                if result:
                    results.append(result)
                    spec_count += 1

print(f"   Additional subscale specs: {spec_count - subscale_start}")

# ---------------------------------------------------------------------------
# 10. PLACEBO-LIKE TESTS
# ---------------------------------------------------------------------------
print("\n10. Running placebo tests...")
placebo_start = spec_count

# Use baseline scores as outcome (should not be affected by treatment)
if 'baseline_literacy_score' in df_e1.columns:
    result = run_regression(
        df_e1, 'baseline_literacy_score', 'treat',
        [c for c in available_controls if c != 'baseline_literacy_score'],
        fe_vars_e1, None,
        spec_id='robust/placebo/baseline_literacy',
        spec_tree_path='robustness/placebo_tests.md',
        sample_desc='Placebo: baseline literacy (should be null)'
    )
    if result:
        results.append(result)
        spec_count += 1

if 'baseline_numeracy_score' in df_e1.columns:
    result = run_regression(
        df_e1, 'baseline_numeracy_score', 'treat',
        [c for c in available_controls if c != 'baseline_numeracy_score'],
        fe_vars_e1, None,
        spec_id='robust/placebo/baseline_numeracy',
        spec_tree_path='robustness/placebo_tests.md',
        sample_desc='Placebo: baseline numeracy (should be null)'
    )
    if result:
        results.append(result)
        spec_count += 1

print(f"   Placebo specs: {spec_count - placebo_start}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Convert to DataFrame
results_df = pd.DataFrame(results)
print(f"\nTotal specifications: {len(results_df)}")

if len(results_df) > 0:
    # Save to CSV
    output_file = f"{OUTPUT_PATH}/specification_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")

    # ============================================================================
    # SUMMARY STATISTICS
    # ============================================================================
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    # Filter to main treatment coefficient (for SDQ total difficulties)
    main_results = results_df[results_df['outcome_var'].str.contains('totdiff', na=False)]

    if len(main_results) > 0:
        print(f"\nSDQ Total Difficulties Specifications (N={len(main_results)}):")
        print(f"  Positive coefficients: {(main_results['coefficient'] > 0).sum()} ({100*(main_results['coefficient'] > 0).mean():.1f}%)")
        print(f"  Negative coefficients: {(main_results['coefficient'] < 0).sum()} ({100*(main_results['coefficient'] < 0).mean():.1f}%)")
        print(f"  Significant at 5%: {(main_results['p_value'] < 0.05).sum()} ({100*(main_results['p_value'] < 0.05).mean():.1f}%)")
        print(f"  Significant at 1%: {(main_results['p_value'] < 0.01).sum()} ({100*(main_results['p_value'] < 0.01).mean():.1f}%)")
        print(f"  Coefficient range: [{main_results['coefficient'].min():.3f}, {main_results['coefficient'].max():.3f}]")
        print(f"  Median coefficient: {main_results['coefficient'].median():.3f}")
        print(f"  Mean coefficient: {main_results['coefficient'].mean():.3f}")

    # All specifications
    print(f"\nAll Specifications (N={len(results_df)}):")
    print(f"  Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
else:
    print("No results generated!")

print("\n" + "="*80)
print("SPECIFICATION SEARCH COMPLETE")
print("="*80)
