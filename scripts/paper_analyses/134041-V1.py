#!/usr/bin/env python3
"""
Specification search script for 134041-V1:
"How Do Beliefs About the Gender Wage Gap Affect the Demand for Public Policy?"
Sonja Settele, AEJ: Economic Policy

Surface-driven execution: G1 (perception index) and G2 (policy demand index).
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import hashlib
import traceback
import sys
import warnings
warnings.filterwarnings('ignore')

PACKAGE_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/134041-V1"
PAPER_ID = "134041-V1"

# ============================================================
# LOAD SURFACE
# ============================================================
with open(f"{PACKAGE_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface = json.load(f)

surface_hash_val = "sha256:" + hashlib.sha256(
    json.dumps(surface, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
).hexdigest()

# ============================================================
# SOFTWARE BLOCK
# ============================================================
software_block = {
    "runner_language": "python",
    "runner_version": sys.version.split()[0],
    "packages": {
        "pyfixest": pf.__version__ if hasattr(pf, '__version__') else "0.40+",
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
}

# ============================================================
# DATA CLEANING -- replicate Stata pipeline
# ============================================================
print("=== DATA CLEANING ===")

def decode_A_codes(series, n_codes, offset=0):
    """Convert 'A1','A2',... string codes to numeric 1,2,..."""
    mapping = {f"A{i}": i + offset for i in range(1, n_codes + 1)}
    return series.map(mapping)

def clean_wave(raw_path, wave_num):
    """Clean one wave of survey data, replicating the Stata cleaning code."""
    df = pd.read_stata(raw_path, convert_categoricals=False)

    # Employment
    employ_map = {f"A{i}": i for i in range(1, 8)}
    df['employ'] = df['employment'].map(employ_map)
    df['fulltime'] = (df['employ'] == 1).astype(float)
    df['parttime'] = (df['employ'] == 2).astype(float)
    df['selfemp'] = (df['employ'] == 3).astype(float)
    df['unemp'] = (df['employ'] == 4).astype(float)  # Note: in Stata code the variable is 'unemployed' but global uses 'unemp'
    df['student'] = (df['employ'] == 5).astype(float)
    df['employee'] = ((df['employ'] == 1) | (df['employ'] == 2)).astype(float)
    df.loc[df['employ'].isna(), ['fulltime', 'parttime', 'selfemp', 'unemp', 'student', 'employee']] = np.nan

    # Region
    region_map = {f"A{i}": i for i in range(1, 5)}
    df['region_num'] = df['region'].map(region_map)
    df['northeast'] = (df['region_num'] == 1).astype(float)
    df['midwest'] = (df['region_num'] == 2).astype(float)
    df['south'] = (df['region_num'] == 3).astype(float)
    df['west'] = (df['region_num'] == 4).astype(float)

    # Age bracket
    age_map = {f"A{i}": i for i in range(1, 6)}
    df['age'] = df['agebracket'].map(age_map)
    for i in range(1, 6):
        df[f'age{i}'] = (df['age'] == i).astype(float)

    # Gender
    gender_map = {"A1": 0, "A2": 1}
    df['gender'] = df['gender'].map(gender_map) if df['gender'].dtype == object else df['gender']
    df['female'] = df['gender'].copy()
    df['male'] = 1 - df['female']

    # Political orientation
    pol_map = {"A1": -2, "A2": -1, "A3": 0, "A4": 1, "A5": 2}
    df['pol'] = df['demrep'].map(pol_map) if 'demrep' in df.columns else np.nan
    df['otherpol'] = (df['demrep'] == "-oth-").astype(float) if 'demrep' in df.columns else 0
    df['republican'] = (df['pol'] < 0).astype(float)
    df['democrat'] = ((df['pol'] > 0) & df['pol'].notna()).astype(float)
    df['indep'] = (df['pol'] == 0).astype(float)
    # If otherpol==1, set all three to 0
    df.loc[df['otherpol'] == 1, ['republican', 'democrat', 'indep']] = 0

    # Treatment
    df['T1'] = (df['RAND'] == 1).astype(float)
    df['T2'] = (df['RAND'] == 2).astype(float)
    df['rand'] = df['RAND'].copy()

    # Prior beliefs
    df['prior1'] = (df['RAND12'] == 1).astype(float)
    df['prior'] = np.where(df['RAND12'] == 1, df['elicitbgendeMarc'], df['elicitbgendernoinMar'])

    # Posterior belief
    if wave_num == 1:
        # Wave A: RAND4 in {9,10,11}
        df['posterior'] = np.nan
        if 'extrayoung' in df.columns:
            df.loc[df['RAND4'] == 9, 'posterior'] = df.loc[df['RAND4'] == 9, 'extrayoung']
        if 'extraHSS' in df.columns:
            df.loc[df['RAND4'] == 10, 'posterior'] = df.loc[df['RAND4'] == 10, 'extraHSS']
        elif 'extraHS' in df.columns:
            df.loc[df['RAND4'] == 10, 'posterior'] = df.loc[df['RAND4'] == 10, 'extraHS']
        if 'extraoccuu' in df.columns:
            df.loc[df['RAND4'] == 11, 'posterior'] = df.loc[df['RAND4'] == 11, 'extraoccuu']
        elif 'extraoccu' in df.columns:
            df.loc[df['RAND4'] == 11, 'posterior'] = df.loc[df['RAND4'] == 11, 'extraoccu']
    else:
        # Wave B: RAND4 in {10,11}
        df['posterior'] = np.nan
        if 'extrasame' in df.columns:
            df.loc[df['RAND4'] == 10, 'posterior'] = df.loc[df['RAND4'] == 10, 'extrasame']
        if 'extrachild' in df.columns:
            df.loc[df['RAND4'] == 11, 'posterior'] = df.loc[df['RAND4'] == 11, 'extrachild']

    # Manipulation check / perceptions (10-point Likert scale encoded as A1-A10)
    for old, new in [('manicheckSQ001', 'large'), ('manicheckSQ002', 'problem'), ('manicheckSQ003', 'govmore')]:
        if old in df.columns:
            mapping = {f"A{i}": i for i in range(1, 11)}
            df[new] = df[old].map(mapping)

    # womenwages (also 1-10 Likert)
    if 'womenwages' in df.columns and df['womenwages'].dtype == object:
        mapping = {f"A{i}": i for i in range(1, 11)}
        df['womenwages'] = df['womenwages'].map(mapping)

    # Policy preferences (5-point Likert encoded as A1-A5)
    for var in ['quotaanchor', 'AAanchor', 'transparencyanchor', 'legislationanchor', 'childcare']:
        if var in df.columns and df[var].dtype == object:
            mapping = {f"A{i}": i for i in range(1, 6)}
            df[var] = df[var].map(mapping)

    # Wave B: transparencyanchor was actually a different question (UKtool)
    if wave_num == 2:
        df['UKtool'] = df['transparencyanchor'].copy()
        df['transparencyanchor'] = np.nan
    else:
        df['UKtool'] = np.nan

    # Household income
    hhinc_map = {f"A{i}": i for i in range(1, 9)}
    df['hhinc'] = df['hhincbracket'].map(hhinc_map) if 'hhincbracket' in df.columns else np.nan

    hhinccont_map = {1: 6735, 2: 19742, 3: 36701, 4: 61275, 5: 86204, 6: 120686, 7: 170381, 8: 327261}
    df['hhinccont'] = df['hhinc'].map(hhinccont_map)
    df['loghhinc'] = np.log(df['hhinccont'])

    # Education
    if 'demo1' in df.columns:
        educ_map = {f"A{i}": i for i in range(1, 10)}
        df['educ'] = df['demo1'].map(educ_map)
        df['associatemore'] = (df['educ'] > 4).astype(float)
        df.loc[df['educ'].isna(), 'associatemore'] = np.nan

    # Children
    for var in ['childrenSQ001', 'childrenSQ002']:
        if var in df.columns and df[var].dtype == object:
            mapping = {f"A{i}": i for i in range(1, 7)}
            df[var] = df[var].map(mapping)

    if 'childrenSQ001' in df.columns and 'childrenSQ002' in df.columns:
        boy = df['childrenSQ001'] - 1  # subtract 1 per Stata code
        girl = df['childrenSQ002'] - 1
        df['children'] = boy + girl
        df['anychildren'] = ((df['children'] > 0) & df['children'].notna()).astype(float)
    else:
        df['anychildren'] = np.nan

    # fairown (1-10 scale)
    if 'fairown' in df.columns and df['fairown'].dtype == object:
        mapping = {f"A{i}": i for i in range(1, 11)}
        df['fairown'] = df['fairown'].map(mapping)
        df.loc[df['fairown'] == 6, 'fairown'] = np.nan  # "never worked" = missing

    # Wave indicator
    df['wave'] = wave_num

    return df


# Clean both waves
dfA = clean_wave(f"{PACKAGE_DIR}/data/SurveyStageI_WaveA_raw.dta", wave_num=1)
dfB = clean_wave(f"{PACKAGE_DIR}/data/SurveyStageI_WaveB_raw.dta", wave_num=2)

print(f"Wave A after cleaning: {len(dfA)} rows")
print(f"Wave B after cleaning: {len(dfB)} rows")

# ============================================================
# APPEND WAVES
# ============================================================
# Ensure common columns
common_cols = sorted(set(dfA.columns) & set(dfB.columns))
df = pd.concat([dfA[common_cols], dfB[common_cols]], ignore_index=True)

print(f"Combined: {len(df)} rows")

# ============================================================
# PROBABILITY WEIGHTS (from do-file 05)
# ============================================================
df['pweight'] = 1.0
# Wave B corrections for oversampled women in certain age groups
df.loc[(df['wave'] == 2) & (df['gender'] == 0) & (df['age1'] == 1), 'pweight'] = 1.4615
df.loc[(df['wave'] == 2) & (df['gender'] == 1) & (df['age1'] == 1), 'pweight'] = 0.6298
df.loc[(df['wave'] == 2) & (df['gender'] == 0) & (df['age5'] == 1), 'pweight'] = 1.0184
df.loc[(df['wave'] == 2) & (df['gender'] == 1) & (df['age5'] == 1), 'pweight'] = 0.8691

# ============================================================
# Z-SCORE VARIABLES (based on control group mean/sd)
# ============================================================
def zscore_using_control(df, varname, control_mask):
    """Z-score using control group mean and SD, applied to all observations."""
    ctrl_vals = df.loc[control_mask, varname]
    mean_val = ctrl_vals.mean()
    sd_val = ctrl_vals.std()
    if sd_val > 0:
        df[varname] = (df[varname] - mean_val) / sd_val
    return df

control_mask = (df['rand'] == 0)

# z-score manipulation check variables
for var in ['large', 'problem', 'govmore']:
    df = zscore_using_control(df, var, control_mask)

# z-score policy preferences
for var in ['quotaanchor', 'AAanchor', 'legislationanchor', 'childcare']:
    df = zscore_using_control(df, var, control_mask)

# UKtool: only in Wave B, z-score using Wave B control group
control_mask_B = (df['rand'] == 0) & (df['wave'] == 2)
# For transparency anchor (Wave A only), z-score using Wave A control group
# Actually the Stata code z-scores transparencyanchor across both waves,
# but then replaces with UKtool for Wave B and sets transparencyanchor to missing for Wave B
# The z-scoring in Stata happens AFTER appending, using the overall control group for each variable
# The policy index dm_y4 uses transparencyanchor but replaces with UKtool where missing
# Let's handle this carefully:

# z-score UKtool using the overall control group (for UKtool, this is only Wave B respondents)
ukvals = df.loc[control_mask & df['UKtool'].notna(), 'UKtool']
if len(ukvals) > 0:
    uk_mean = ukvals.mean()
    uk_sd = ukvals.std()
    if uk_sd > 0:
        df['UKtool'] = (df['UKtool'] - uk_mean) / uk_sd

# z-score transparencyanchor using overall control group (Wave A only has data)
ta_vals = df.loc[control_mask & df['transparencyanchor'].notna(), 'transparencyanchor']
if len(ta_vals) > 0:
    ta_mean = ta_vals.mean()
    ta_sd = ta_vals.std()
    if ta_sd > 0:
        df['transparencyanchor'] = (df['transparencyanchor'] - ta_mean) / ta_sd

# z-score womenwages
if 'womenwages' in df.columns:
    df = zscore_using_control(df, 'womenwages', control_mask)

# ============================================================
# INTERACTION TERMS (from do-file 05)
# ============================================================
# Re-set T1, T2 using rand (as the Stata code does: replace T1=(rand==1) if rand!=.)
df['T1'] = np.where(df['rand'].notna(), (df['rand'] == 1).astype(float), np.nan)
df['T2'] = np.where(df['rand'].notna(), (df['rand'] == 2).astype(float), np.nan)
df['T1female'] = df['T1'] * df['female']
df['T1democrat'] = df['T1'] * df['democrat']
df['T1indep'] = df['T1'] * df['indep']

# ============================================================
# BUILD SUMMARY INDICES (from do-file 05)
# ============================================================
def build_inverse_covariance_index(df, components, varname):
    """Build Anderson (2008) inverse-covariance-weighted index.

    This replicates the Stata code that:
    1. Computes the correlation matrix of the components
    2. Inverts it
    3. Weights each component by the row sum of the inverse
    4. Normalizes by the sum of all weights
    """
    # Get complete cases for all components
    subdf = df[components].dropna()
    if len(subdf) < 10:
        df[varname] = np.nan
        return df

    # Correlation matrix
    corr_mat = subdf.corr().values

    # Invert
    try:
        inv_mat = np.linalg.inv(corr_mat)
    except np.linalg.LinAlgError:
        inv_mat = np.linalg.pinv(corr_mat)

    # Row sums = weights
    weights = inv_mat.sum(axis=1)

    # Compute index: sum(weight_i * var_i) / sum(weights)
    numerator = np.zeros(len(df))
    for i, comp in enumerate(components):
        numerator += weights[i] * df[comp].fillna(0)

    denominator = weights.sum()
    df[varname] = numerator / denominator

    # Set to NaN where any component is missing
    any_missing = df[components].isna().any(axis=1)
    df.loc[any_missing, varname] = np.nan

    return df


# Perception index (z_mani_index): large, problem, govmore
df = build_inverse_covariance_index(df, ['large', 'problem', 'govmore'], 'z_mani_index')

# Policy demand index (z_lmpolicy_index): quotaanchor, AAanchor, legislationanchor, transparencyanchor/UKtool, childcare
# The Stata code uses dm_y4 = transparencyanchor, then replaces with UKtool where missing
df['policy_transparency'] = df['transparencyanchor'].copy()
df.loc[df['policy_transparency'].isna(), 'policy_transparency'] = df.loc[df['policy_transparency'].isna(), 'UKtool']
df = build_inverse_covariance_index(df, ['quotaanchor', 'AAanchor', 'legislationanchor', 'policy_transparency', 'childcare'], 'z_lmpolicy_index')

print(f"z_mani_index non-null: {df['z_mani_index'].notna().sum()}")
print(f"z_lmpolicy_index non-null: {df['z_lmpolicy_index'].notna().sum()}")

# ============================================================
# Z-SCORE POSTERIOR (done in Table 5 code, after dropping control group)
# ============================================================
# The paper z-scores posterior using aweights=pweight, but only among non-control respondents
# zscore posterior [aweight=pweight], stub(z) -> creates zposterior
# We'll do this on the treatment sample
treat_mask = (df['rand'] != 0)
post_vals = df.loc[treat_mask & df['posterior'].notna(), 'posterior']
pw = df.loc[treat_mask & df['posterior'].notna(), 'pweight']
if len(post_vals) > 0:
    wmean = np.average(post_vals, weights=pw)
    wvar = np.average((post_vals - wmean)**2, weights=pw)
    wsd = np.sqrt(wvar)
    df['zposterior'] = (df['posterior'] - wmean) / wsd
else:
    df['zposterior'] = np.nan

# ============================================================
# SAMPLE SELECTION: Drop pure control group (rand==0) for Table 5 analysis
# ============================================================
df_treat = df[df['rand'] != 0].copy()
print(f"Treatment sample (rand != 0): {len(df_treat)} rows")

# Check key variables
for var in ['T1', 'z_mani_index', 'z_lmpolicy_index', 'posterior', 'zposterior', 'pweight']:
    n_nonmiss = df_treat[var].notna().sum()
    print(f"  {var}: {n_nonmiss} non-missing")

# ============================================================
# DEFINE CONTROL VARIABLE GROUPS
# ============================================================
baseline_controls = [
    'wave', 'gender', 'prior', 'democrat', 'indep', 'otherpol',
    'midwest', 'south', 'west',
    'age1', 'age2', 'age3', 'age4',
    'anychildren', 'loghhinc', 'associatemore',
    'fulltime', 'parttime', 'selfemp', 'unemp', 'student'
]

# Control blocks per surface constraints
region_block = ['midwest', 'south', 'west']
age_block = ['age1', 'age2', 'age3', 'age4']
employment_block = ['fulltime', 'parttime', 'selfemp', 'unemp', 'student']

demographics_only = ['gender', 'age1', 'age2', 'age3', 'age4', 'midwest', 'south', 'west']
demographics_plus_politics = demographics_only + ['democrat', 'indep', 'otherpol']
demographics_plus_economics = demographics_only + ['loghhinc', 'fulltime', 'parttime', 'selfemp', 'unemp', 'student', 'associatemore']

# ============================================================
# DESIGN AUDIT BLOCKS
# ============================================================
design_block_G1 = {"randomized_experiment": surface["baseline_groups"][0]["design_audit"]}
design_block_G2 = {"randomized_experiment": surface["baseline_groups"][1]["design_audit"]}

# ============================================================
# RUN SPECS
# ============================================================
results = []
spec_run_id = 0


def run_spec(spec_id, spec_tree_path, outcome_var, treatment_var, controls, data,
             weight_var='pweight', sample_desc='Treatment sample (rand!=0)',
             baseline_group_id='G1', design_block_val=None, extra_json_blocks=None,
             vcov="hetero"):
    """Run a single OLS specification and record results."""
    global spec_run_id
    spec_run_id += 1
    run_id = f"S{spec_run_id:03d}"

    if design_block_val is None:
        design_block_val = design_block_G1 if baseline_group_id == 'G1' else design_block_G2

    all_vars = [outcome_var, treatment_var] + [c for c in controls if c != treatment_var]
    if weight_var:
        all_vars.append(weight_var)
    all_vars = list(dict.fromkeys(all_vars))

    subdf = data.dropna(subset=all_vars).copy()

    indepvars = [treatment_var] + [c for c in controls if c != treatment_var]
    indepvars = list(dict.fromkeys(indepvars))

    formula = f"{outcome_var} ~ " + " + ".join(indepvars) if indepvars else f"{outcome_var} ~ 1 + {treatment_var}"

    try:
        if weight_var and weight_var in subdf.columns:
            m = pf.feols(formula, data=subdf, vcov=vcov, weights=weight_var)
        else:
            m = pf.feols(formula, data=subdf, vcov=vcov)

        coef = float(m.coef()[treatment_var])
        se = float(m.se()[treatment_var])
        pval = float(m.pvalue()[treatment_var])
        nobs = int(m._N)
        r2 = float(m._r2)
        ci = m.confint()
        ci_lower = float(ci.loc[treatment_var].iloc[0])
        ci_upper = float(ci.loc[treatment_var].iloc[1])

        coef_dict = {k: float(v) for k, v in m.coef().items()}

        cvj = {
            "coefficients": coef_dict,
            "inference": {"spec_id": "infer/se/hc/hc1", "params": {}},
            "software": software_block,
            "surface_hash": surface_hash_val,
            "design": design_block_val,
        }
        if extra_json_blocks:
            for k, v in extra_json_blocks.items():
                cvj[k] = v

        result = {
            'paper_id': PAPER_ID,
            'spec_run_id': run_id,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'baseline_group_id': baseline_group_id,
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': coef,
            'std_error': se,
            'p_value': pval,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_obs': nobs,
            'r_squared': r2,
            'coefficient_vector_json': json.dumps(cvj),
            'sample_desc': sample_desc,
            'fixed_effects': '',
            'controls_desc': f'{len(indepvars)-1} controls' if len(indepvars) > 1 else 'no controls',
            'cluster_var': '',
            'run_success': 1,
            'run_error': '',
        }
        results.append(result)
        print(f"  {run_id} ({spec_id}): coef={coef:.4f}, se={se:.4f}, p={pval:.4f}, N={nobs}")
        return m
    except Exception as e:
        tb_str = traceback.format_exc()
        err_msg = str(e)[:200]
        cvj = {
            "error": err_msg,
            "error_details": {
                "stage": "estimation",
                "exception_type": type(e).__name__,
                "exception_message": str(e),
                "traceback_tail": tb_str[-500:]
            }
        }
        result = {
            'paper_id': PAPER_ID,
            'spec_run_id': run_id,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'baseline_group_id': baseline_group_id,
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': np.nan,
            'std_error': np.nan,
            'p_value': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n_obs': np.nan,
            'r_squared': np.nan,
            'coefficient_vector_json': json.dumps(cvj),
            'sample_desc': sample_desc,
            'fixed_effects': '',
            'controls_desc': '',
            'cluster_var': '',
            'run_success': 0,
            'run_error': err_msg,
        }
        results.append(result)
        print(f"  {run_id} ({spec_id}): FAILED - {err_msg}")
        return None


# ============================================================
# G1: PERCEPTION INDEX (z_mani_index)
# ============================================================
print("\n========== G1: PERCEPTION INDEX ==========")

# --- BASELINE ---
print("\n=== G1 BASELINE ===")
run_spec(
    spec_id='baseline',
    spec_tree_path='designs/randomized_experiment.md#baseline',
    outcome_var='z_mani_index',
    treatment_var='T1',
    controls=baseline_controls,
    data=df_treat,
    baseline_group_id='G1',
    sample_desc='Table 5 Panel A Col 6: z_mani_index ~ T1 + controls [pweight], vce(r)',
)

# Additional baselines: individual perception components
for out_var, label in [
    ('posterior', 'baseline__posterior'),
    ('zposterior', 'baseline__zposterior'),
    ('large', 'baseline__large'),
    ('problem', 'baseline__problem'),
    ('govmore', 'baseline__govmore'),
]:
    run_spec(
        spec_id=label,
        spec_tree_path='designs/randomized_experiment.md#baseline',
        outcome_var=out_var,
        treatment_var='T1',
        controls=baseline_controls,
        data=df_treat,
        baseline_group_id='G1',
        sample_desc=f'Table 5 Panel A: {out_var} ~ T1 + controls',
    )

# --- DESIGN VARIANTS ---
print("\n=== G1 DESIGN VARIANTS ===")

# Diff in means (no controls)
run_spec(
    spec_id='design/randomized_experiment/estimator/diff_in_means',
    spec_tree_path='designs/randomized_experiment.md#itt-implementations',
    outcome_var='z_mani_index',
    treatment_var='T1',
    controls=[],
    data=df_treat,
    baseline_group_id='G1',
    sample_desc='Diff-in-means: z_mani_index ~ T1, no controls',
)

# With covariates (same as baseline, explicit label)
run_spec(
    spec_id='design/randomized_experiment/estimator/with_covariates',
    spec_tree_path='designs/randomized_experiment.md#itt-implementations',
    outcome_var='z_mani_index',
    treatment_var='T1',
    controls=baseline_controls,
    data=df_treat,
    baseline_group_id='G1',
    sample_desc='With covariates: z_mani_index ~ T1 + full controls',
)

# --- RC: LEAVE-ONE-OUT ---
print("\n=== G1 RC: LEAVE-ONE-OUT ===")

# Individual LOO drops
loo_singles = [
    ('wave', 'drop_wave'),
    ('gender', 'drop_gender'),
    ('prior', 'drop_prior'),
    ('democrat', 'drop_democrat'),
    ('indep', 'drop_indep'),
    ('otherpol', 'drop_otherpol'),
    ('anychildren', 'drop_anychildren'),
    ('loghhinc', 'drop_loghhinc'),
    ('associatemore', 'drop_associatemore'),
]

for var, label in loo_singles:
    remaining = [c for c in baseline_controls if c != var]
    run_spec(
        spec_id=f'rc/controls/loo/{label}',
        spec_tree_path='modules/robustness/controls.md#leave-one-out-controls-loo',
        outcome_var='z_mani_index',
        treatment_var='T1',
        controls=remaining,
        data=df_treat,
        baseline_group_id='G1',
        sample_desc=f'LOO: drop {var}',
        extra_json_blocks={"controls": {"spec_id": f"rc/controls/loo/{label}", "family": "loo", "dropped": [var], "n_controls": len(remaining)}},
    )

# Block LOO drops
# Region block
remaining = [c for c in baseline_controls if c not in region_block]
run_spec(
    spec_id='rc/controls/loo/drop_region_block',
    spec_tree_path='modules/robustness/controls.md#leave-one-out-controls-loo',
    outcome_var='z_mani_index',
    treatment_var='T1',
    controls=remaining,
    data=df_treat,
    baseline_group_id='G1',
    sample_desc='LOO: drop region block (midwest, south, west)',
    extra_json_blocks={"controls": {"spec_id": "rc/controls/loo/drop_region_block", "family": "loo", "dropped": region_block, "n_controls": len(remaining)}},
)

# Age block
remaining = [c for c in baseline_controls if c not in age_block]
run_spec(
    spec_id='rc/controls/loo/drop_age_block',
    spec_tree_path='modules/robustness/controls.md#leave-one-out-controls-loo',
    outcome_var='z_mani_index',
    treatment_var='T1',
    controls=remaining,
    data=df_treat,
    baseline_group_id='G1',
    sample_desc='LOO: drop age block (age1-age4)',
    extra_json_blocks={"controls": {"spec_id": "rc/controls/loo/drop_age_block", "family": "loo", "dropped": age_block, "n_controls": len(remaining)}},
)

# Employment block
remaining = [c for c in baseline_controls if c not in employment_block]
run_spec(
    spec_id='rc/controls/loo/drop_employment_block',
    spec_tree_path='modules/robustness/controls.md#leave-one-out-controls-loo',
    outcome_var='z_mani_index',
    treatment_var='T1',
    controls=remaining,
    data=df_treat,
    baseline_group_id='G1',
    sample_desc='LOO: drop employment block (fulltime, parttime, selfemp, unemp, student)',
    extra_json_blocks={"controls": {"spec_id": "rc/controls/loo/drop_employment_block", "family": "loo", "dropped": employment_block, "n_controls": len(remaining)}},
)

# --- RC: CONTROL SETS ---
print("\n=== G1 RC: CONTROL SETS ===")

run_spec(
    spec_id='rc/controls/sets/none',
    spec_tree_path='modules/robustness/controls.md#standard-control-sets',
    outcome_var='z_mani_index',
    treatment_var='T1',
    controls=[],
    data=df_treat,
    baseline_group_id='G1',
    sample_desc='No controls (pure treatment-control)',
    extra_json_blocks={"controls": {"spec_id": "rc/controls/sets/none", "family": "sets", "set_name": "none", "n_controls": 0}},
)

run_spec(
    spec_id='rc/controls/sets/demographics_only',
    spec_tree_path='modules/robustness/controls.md#standard-control-sets',
    outcome_var='z_mani_index',
    treatment_var='T1',
    controls=demographics_only,
    data=df_treat,
    baseline_group_id='G1',
    sample_desc='Demographics only (gender, age, region)',
    extra_json_blocks={"controls": {"spec_id": "rc/controls/sets/demographics_only", "family": "sets", "set_name": "demographics_only", "n_controls": len(demographics_only)}},
)

run_spec(
    spec_id='rc/controls/sets/demographics_plus_politics',
    spec_tree_path='modules/robustness/controls.md#standard-control-sets',
    outcome_var='z_mani_index',
    treatment_var='T1',
    controls=demographics_plus_politics,
    data=df_treat,
    baseline_group_id='G1',
    sample_desc='Demographics + politics',
    extra_json_blocks={"controls": {"spec_id": "rc/controls/sets/demographics_plus_politics", "family": "sets", "set_name": "demographics_plus_politics", "n_controls": len(demographics_plus_politics)}},
)

run_spec(
    spec_id='rc/controls/sets/demographics_plus_economics',
    spec_tree_path='modules/robustness/controls.md#standard-control-sets',
    outcome_var='z_mani_index',
    treatment_var='T1',
    controls=demographics_plus_economics,
    data=df_treat,
    baseline_group_id='G1',
    sample_desc='Demographics + economics',
    extra_json_blocks={"controls": {"spec_id": "rc/controls/sets/demographics_plus_economics", "family": "sets", "set_name": "demographics_plus_economics", "n_controls": len(demographics_plus_economics)}},
)

run_spec(
    spec_id='rc/controls/sets/full_baseline',
    spec_tree_path='modules/robustness/controls.md#standard-control-sets',
    outcome_var='z_mani_index',
    treatment_var='T1',
    controls=baseline_controls,
    data=df_treat,
    baseline_group_id='G1',
    sample_desc='Full baseline controls',
    extra_json_blocks={"controls": {"spec_id": "rc/controls/sets/full_baseline", "family": "sets", "set_name": "full_baseline", "n_controls": len(baseline_controls)}},
)

# --- RC: CONTROL PROGRESSION ---
print("\n=== G1 RC: CONTROL PROGRESSION ===")

progression_specs = [
    ('rc/controls/progression/bivariate', [], 'Bivariate: T1 only'),
    ('rc/controls/progression/wave_only', ['wave'], 'Wave only'),
    ('rc/controls/progression/wave_demographics', ['wave'] + demographics_only, 'Wave + demographics'),
    ('rc/controls/progression/wave_demographics_politics', ['wave'] + demographics_plus_politics, 'Wave + demographics + politics'),
    ('rc/controls/progression/full', baseline_controls, 'Full controls'),
]

for sid, ctrls, desc in progression_specs:
    run_spec(
        spec_id=sid,
        spec_tree_path='modules/robustness/controls.md#control-progression-build-up',
        outcome_var='z_mani_index',
        treatment_var='T1',
        controls=ctrls,
        data=df_treat,
        baseline_group_id='G1',
        sample_desc=f'Progression: {desc}',
        extra_json_blocks={"controls": {"spec_id": sid, "family": "progression", "set_name": desc, "n_controls": len(ctrls)}},
    )

# --- RC: RANDOM CONTROL SUBSETS ---
print("\n=== G1 RC: RANDOM CONTROL SUBSETS ===")

rng = np.random.RandomState(134041)
optional_controls = [c for c in baseline_controls if c != 'wave']  # wave is mandatory

for i in range(1, 11):
    n_draw = rng.randint(3, len(optional_controls))
    drawn = list(rng.choice(optional_controls, size=n_draw, replace=False))
    ctrls = ['wave'] + drawn
    run_spec(
        spec_id=f'rc/controls/subset/random_{i:03d}',
        spec_tree_path='modules/robustness/controls.md#high-dimensional-control-set-search',
        outcome_var='z_mani_index',
        treatment_var='T1',
        controls=ctrls,
        data=df_treat,
        baseline_group_id='G1',
        sample_desc=f'Random subset {i}: {n_draw} optional controls',
        extra_json_blocks={"controls": {"spec_id": f"rc/controls/subset/random_{i:03d}", "family": "subset", "draw_index": i, "included": ctrls, "n_controls": len(ctrls)}},
    )

# --- RC: SAMPLE RESTRICTIONS ---
print("\n=== G1 RC: SAMPLE ===")

# Wave A only
df_waveA = df_treat[df_treat['wave'] == 1].copy()
waveA_controls = [c for c in baseline_controls if c != 'wave']
run_spec(
    spec_id='rc/sample/restriction/wave_a_only',
    spec_tree_path='modules/robustness/sample.md#data-quality-and-eligibility-filters',
    outcome_var='z_mani_index',
    treatment_var='T1',
    controls=waveA_controls,
    data=df_waveA,
    baseline_group_id='G1',
    sample_desc='Wave A only (wave dummy dropped)',
    extra_json_blocks={"sample": {"spec_id": "rc/sample/restriction/wave_a_only", "axis": "restriction", "rule": "filter", "params": {"wave": 1}}},
)

# Wave B only
df_waveB = df_treat[df_treat['wave'] == 2].copy()
run_spec(
    spec_id='rc/sample/restriction/wave_b_only',
    spec_tree_path='modules/robustness/sample.md#data-quality-and-eligibility-filters',
    outcome_var='z_mani_index',
    treatment_var='T1',
    controls=waveA_controls,  # same: drop wave dummy
    data=df_waveB,
    baseline_group_id='G1',
    sample_desc='Wave B only (wave dummy dropped)',
    extra_json_blocks={"sample": {"spec_id": "rc/sample/restriction/wave_b_only", "axis": "restriction", "rule": "filter", "params": {"wave": 2}}},
)

# Incentivized prior only (prior1==1)
df_inc = df_treat[df_treat['prior1'] == 1].copy()
run_spec(
    spec_id='rc/sample/restriction/incentivized_prior_only',
    spec_tree_path='modules/robustness/sample.md#data-quality-and-eligibility-filters',
    outcome_var='z_mani_index',
    treatment_var='T1',
    controls=baseline_controls,
    data=df_inc,
    baseline_group_id='G1',
    sample_desc='Incentivized prior only (prior1==1)',
    extra_json_blocks={"sample": {"spec_id": "rc/sample/restriction/incentivized_prior_only", "axis": "restriction", "rule": "filter", "params": {"prior1": 1}}},
)

# --- RC: WEIGHTS ---
print("\n=== G1 RC: WEIGHTS ===")

run_spec(
    spec_id='rc/weights/unweighted',
    spec_tree_path='modules/robustness/controls.md#standard-control-sets',
    outcome_var='z_mani_index',
    treatment_var='T1',
    controls=baseline_controls,
    data=df_treat,
    weight_var=None,
    baseline_group_id='G1',
    sample_desc='Unweighted (no probability weights)',
    extra_json_blocks={"weights": {"spec_id": "rc/weights/unweighted", "weight_var": None, "notes": "No probability weights applied"}},
)


# ============================================================
# G2: POLICY DEMAND INDEX (z_lmpolicy_index)
# ============================================================
print("\n========== G2: POLICY DEMAND INDEX ==========")

# --- BASELINE ---
print("\n=== G2 BASELINE ===")
run_spec(
    spec_id='baseline',
    spec_tree_path='designs/randomized_experiment.md#baseline',
    outcome_var='z_lmpolicy_index',
    treatment_var='T1',
    controls=baseline_controls,
    data=df_treat,
    baseline_group_id='G2',
    sample_desc='Table 5 Panel B Col 7: z_lmpolicy_index ~ T1 + controls [pweight], vce(r)',
)

# Additional baselines: individual policy components
for out_var, label in [
    ('quotaanchor', 'baseline__quotaanchor'),
    ('AAanchor', 'baseline__AAanchor'),
    ('legislationanchor', 'baseline__legislationanchor'),
    ('UKtool', 'baseline__UKtool'),
    ('childcare', 'baseline__childcare'),
]:
    run_spec(
        spec_id=label,
        spec_tree_path='designs/randomized_experiment.md#baseline',
        outcome_var=out_var,
        treatment_var='T1',
        controls=baseline_controls,
        data=df_treat,
        baseline_group_id='G2',
        sample_desc=f'Table 5 Panel B: {out_var} ~ T1 + controls',
    )

# transparencyanchor is Wave A only -- run on Wave A
run_spec(
    spec_id='baseline__transparencyanchor',
    spec_tree_path='designs/randomized_experiment.md#baseline',
    outcome_var='transparencyanchor',
    treatment_var='T1',
    controls=waveA_controls,
    data=df_waveA,
    baseline_group_id='G2',
    sample_desc='Table 5 Panel B: transparencyanchor ~ T1 + controls (Wave A only)',
)

# --- DESIGN VARIANTS ---
print("\n=== G2 DESIGN VARIANTS ===")

run_spec(
    spec_id='design/randomized_experiment/estimator/diff_in_means',
    spec_tree_path='designs/randomized_experiment.md#itt-implementations',
    outcome_var='z_lmpolicy_index',
    treatment_var='T1',
    controls=[],
    data=df_treat,
    baseline_group_id='G2',
    sample_desc='Diff-in-means: z_lmpolicy_index ~ T1, no controls',
)

run_spec(
    spec_id='design/randomized_experiment/estimator/with_covariates',
    spec_tree_path='designs/randomized_experiment.md#itt-implementations',
    outcome_var='z_lmpolicy_index',
    treatment_var='T1',
    controls=baseline_controls,
    data=df_treat,
    baseline_group_id='G2',
    sample_desc='With covariates: z_lmpolicy_index ~ T1 + full controls',
)

# --- RC: LEAVE-ONE-OUT ---
print("\n=== G2 RC: LEAVE-ONE-OUT ===")

for var, label in loo_singles:
    remaining = [c for c in baseline_controls if c != var]
    run_spec(
        spec_id=f'rc/controls/loo/{label}',
        spec_tree_path='modules/robustness/controls.md#leave-one-out-controls-loo',
        outcome_var='z_lmpolicy_index',
        treatment_var='T1',
        controls=remaining,
        data=df_treat,
        baseline_group_id='G2',
        sample_desc=f'LOO: drop {var}',
        extra_json_blocks={"controls": {"spec_id": f"rc/controls/loo/{label}", "family": "loo", "dropped": [var], "n_controls": len(remaining)}},
    )

# Block LOO drops
remaining = [c for c in baseline_controls if c not in region_block]
run_spec(
    spec_id='rc/controls/loo/drop_region_block',
    spec_tree_path='modules/robustness/controls.md#leave-one-out-controls-loo',
    outcome_var='z_lmpolicy_index',
    treatment_var='T1',
    controls=remaining,
    data=df_treat,
    baseline_group_id='G2',
    sample_desc='LOO: drop region block',
    extra_json_blocks={"controls": {"spec_id": "rc/controls/loo/drop_region_block", "family": "loo", "dropped": region_block, "n_controls": len(remaining)}},
)

remaining = [c for c in baseline_controls if c not in age_block]
run_spec(
    spec_id='rc/controls/loo/drop_age_block',
    spec_tree_path='modules/robustness/controls.md#leave-one-out-controls-loo',
    outcome_var='z_lmpolicy_index',
    treatment_var='T1',
    controls=remaining,
    data=df_treat,
    baseline_group_id='G2',
    sample_desc='LOO: drop age block',
    extra_json_blocks={"controls": {"spec_id": "rc/controls/loo/drop_age_block", "family": "loo", "dropped": age_block, "n_controls": len(remaining)}},
)

remaining = [c for c in baseline_controls if c not in employment_block]
run_spec(
    spec_id='rc/controls/loo/drop_employment_block',
    spec_tree_path='modules/robustness/controls.md#leave-one-out-controls-loo',
    outcome_var='z_lmpolicy_index',
    treatment_var='T1',
    controls=remaining,
    data=df_treat,
    baseline_group_id='G2',
    sample_desc='LOO: drop employment block',
    extra_json_blocks={"controls": {"spec_id": "rc/controls/loo/drop_employment_block", "family": "loo", "dropped": employment_block, "n_controls": len(remaining)}},
)

# --- RC: CONTROL SETS ---
print("\n=== G2 RC: CONTROL SETS ===")

for sid, ctrls, desc in [
    ('rc/controls/sets/none', [], 'No controls'),
    ('rc/controls/sets/demographics_only', demographics_only, 'Demographics only'),
    ('rc/controls/sets/demographics_plus_politics', demographics_plus_politics, 'Demographics + politics'),
    ('rc/controls/sets/demographics_plus_economics', demographics_plus_economics, 'Demographics + economics'),
    ('rc/controls/sets/full_baseline', baseline_controls, 'Full baseline'),
]:
    run_spec(
        spec_id=sid,
        spec_tree_path='modules/robustness/controls.md#standard-control-sets',
        outcome_var='z_lmpolicy_index',
        treatment_var='T1',
        controls=ctrls,
        data=df_treat,
        baseline_group_id='G2',
        sample_desc=desc,
        extra_json_blocks={"controls": {"spec_id": sid, "family": "sets", "set_name": desc, "n_controls": len(ctrls)}},
    )

# --- RC: CONTROL PROGRESSION ---
print("\n=== G2 RC: CONTROL PROGRESSION ===")

for sid, ctrls, desc in progression_specs:
    run_spec(
        spec_id=sid,
        spec_tree_path='modules/robustness/controls.md#control-progression-build-up',
        outcome_var='z_lmpolicy_index',
        treatment_var='T1',
        controls=ctrls,
        data=df_treat,
        baseline_group_id='G2',
        sample_desc=f'Progression: {desc}',
        extra_json_blocks={"controls": {"spec_id": sid, "family": "progression", "set_name": desc, "n_controls": len(ctrls)}},
    )

# --- RC: RANDOM CONTROL SUBSETS (G2, seed=134042, 5 draws) ---
print("\n=== G2 RC: RANDOM CONTROL SUBSETS ===")

rng2 = np.random.RandomState(134042)
for i in range(1, 6):
    n_draw = rng2.randint(3, len(optional_controls))
    drawn = list(rng2.choice(optional_controls, size=n_draw, replace=False))
    ctrls = ['wave'] + drawn
    run_spec(
        spec_id=f'rc/controls/subset/random_{i:03d}',
        spec_tree_path='modules/robustness/controls.md#high-dimensional-control-set-search',
        outcome_var='z_lmpolicy_index',
        treatment_var='T1',
        controls=ctrls,
        data=df_treat,
        baseline_group_id='G2',
        sample_desc=f'Random subset {i}: {n_draw} optional controls',
        extra_json_blocks={"controls": {"spec_id": f"rc/controls/subset/random_{i:03d}", "family": "subset", "draw_index": i, "included": ctrls, "n_controls": len(ctrls)}},
    )

# --- RC: SAMPLE RESTRICTIONS ---
print("\n=== G2 RC: SAMPLE ===")

run_spec(
    spec_id='rc/sample/restriction/wave_a_only',
    spec_tree_path='modules/robustness/sample.md#data-quality-and-eligibility-filters',
    outcome_var='z_lmpolicy_index',
    treatment_var='T1',
    controls=waveA_controls,
    data=df_waveA,
    baseline_group_id='G2',
    sample_desc='Wave A only',
    extra_json_blocks={"sample": {"spec_id": "rc/sample/restriction/wave_a_only", "axis": "restriction", "rule": "filter", "params": {"wave": 1}}},
)

run_spec(
    spec_id='rc/sample/restriction/wave_b_only',
    spec_tree_path='modules/robustness/sample.md#data-quality-and-eligibility-filters',
    outcome_var='z_lmpolicy_index',
    treatment_var='T1',
    controls=waveA_controls,
    data=df_waveB,
    baseline_group_id='G2',
    sample_desc='Wave B only',
    extra_json_blocks={"sample": {"spec_id": "rc/sample/restriction/wave_b_only", "axis": "restriction", "rule": "filter", "params": {"wave": 2}}},
)

run_spec(
    spec_id='rc/sample/restriction/incentivized_prior_only',
    spec_tree_path='modules/robustness/sample.md#data-quality-and-eligibility-filters',
    outcome_var='z_lmpolicy_index',
    treatment_var='T1',
    controls=baseline_controls,
    data=df_inc,
    baseline_group_id='G2',
    sample_desc='Incentivized prior only',
    extra_json_blocks={"sample": {"spec_id": "rc/sample/restriction/incentivized_prior_only", "axis": "restriction", "rule": "filter", "params": {"prior1": 1}}},
)

# --- RC: WEIGHTS ---
print("\n=== G2 RC: WEIGHTS ===")

run_spec(
    spec_id='rc/weights/unweighted',
    spec_tree_path='modules/robustness/controls.md#standard-control-sets',
    outcome_var='z_lmpolicy_index',
    treatment_var='T1',
    controls=baseline_controls,
    data=df_treat,
    weight_var=None,
    baseline_group_id='G2',
    sample_desc='Unweighted',
    extra_json_blocks={"weights": {"spec_id": "rc/weights/unweighted", "weight_var": None}},
)

# ============================================================
# SAVE SPECIFICATION RESULTS
# ============================================================
print(f"\n\nTotal specifications run: {len(results)}")

results_df = pd.DataFrame(results)
results_df.to_csv(f"{PACKAGE_DIR}/specification_results.csv", index=False)
print(f"Saved specification_results.csv with {len(results_df)} rows")

n_success = sum(1 for r in results if r['run_success'] == 1)
n_failed = sum(1 for r in results if r['run_success'] == 0)
n_g1 = sum(1 for r in results if r['baseline_group_id'] == 'G1')
n_g2 = sum(1 for r in results if r['baseline_group_id'] == 'G2')
print(f"Success: {n_success}, Failed: {n_failed}")
print(f"G1 specs: {n_g1}, G2 specs: {n_g2}")

# ============================================================
# INFERENCE RESULTS
# ============================================================
print("\n=== INFERENCE VARIANTS ===")

inference_results = []
inf_run_id = 0

# For each baseline group, compute IID and HC3 variants on the primary baseline
for bg_id, out_var, bg_label in [('G1', 'z_mani_index', 'G1-baseline'), ('G2', 'z_lmpolicy_index', 'G2-baseline')]:
    base_run_id = 'S001' if bg_id == 'G1' else [r['spec_run_id'] for r in results if r['baseline_group_id'] == bg_id and r['spec_id'] == 'baseline'][0]

    formula_inf = f"{out_var} ~ T1 + " + " + ".join(baseline_controls)
    all_vars_inf = [out_var, 'T1'] + baseline_controls + ['pweight']
    df_inf = df_treat.dropna(subset=all_vars_inf).copy()

    for infer_spec_id, vcov_type, vcov_label, spec_tree in [
        ("infer/se/iid", "iid", "Classical (homoskedastic) SEs", "modules/inference/standard_errors.md#ols-default-iid"),
        ("infer/se/hc/hc3", "HC3", "HC3 robust SEs", "modules/inference/standard_errors.md#heteroskedasticity-robust-se-no-clustering"),
    ]:
        inf_run_id += 1
        try:
            if vcov_type == "iid":
                m_inf = pf.feols(formula_inf, data=df_inf, vcov="iid", weights="pweight")
            else:
                # HC3
                m_inf = pf.feols(formula_inf, data=df_inf, vcov="HC3", weights="pweight")

            coef_inf = float(m_inf.coef()['T1'])
            se_inf = float(m_inf.se()['T1'])
            pval_inf = float(m_inf.pvalue()['T1'])
            ci_inf = m_inf.confint()

            inf_cvj = {
                "coefficients": {k: float(v) for k, v in m_inf.coef().items()},
                "inference": {"spec_id": infer_spec_id, "method": vcov_label},
                "software": software_block,
                "surface_hash": surface_hash_val,
            }

            inference_results.append({
                'paper_id': PAPER_ID,
                'inference_run_id': f"I{inf_run_id:03d}",
                'spec_run_id': base_run_id,
                'spec_id': infer_spec_id,
                'spec_tree_path': spec_tree,
                'baseline_group_id': bg_id,
                'outcome_var': out_var,
                'treatment_var': 'T1',
                'coefficient': coef_inf,
                'std_error': se_inf,
                'p_value': pval_inf,
                'ci_lower': float(ci_inf.loc['T1'].iloc[0]),
                'ci_upper': float(ci_inf.loc['T1'].iloc[1]),
                'n_obs': int(m_inf._N),
                'r_squared': float(m_inf._r2),
                'cluster_var': '',
                'coefficient_vector_json': json.dumps(inf_cvj),
                'run_success': 1,
                'run_error': '',
            })
            print(f"  I{inf_run_id:03d} ({bg_label}/{infer_spec_id}): coef={coef_inf:.4f}, se={se_inf:.4f}, p={pval_inf:.4f}")
        except Exception as e:
            inf_run_id_str = f"I{inf_run_id:03d}"
            inference_results.append({
                'paper_id': PAPER_ID,
                'inference_run_id': inf_run_id_str,
                'spec_run_id': base_run_id,
                'spec_id': infer_spec_id,
                'spec_tree_path': spec_tree,
                'baseline_group_id': bg_id,
                'outcome_var': out_var,
                'treatment_var': 'T1',
                'coefficient': np.nan, 'std_error': np.nan, 'p_value': np.nan,
                'ci_lower': np.nan, 'ci_upper': np.nan,
                'n_obs': np.nan, 'r_squared': np.nan, 'cluster_var': '',
                'coefficient_vector_json': json.dumps({"error": str(e), "error_details": {"stage": "inference", "exception_type": type(e).__name__, "exception_message": str(e)}}),
                'run_success': 0,
                'run_error': str(e)[:200],
            })
            print(f"  I{inf_run_id:03d} ({bg_label}/{infer_spec_id}): FAILED - {e}")

if inference_results:
    inf_df = pd.DataFrame(inference_results)
    inf_df.to_csv(f"{PACKAGE_DIR}/inference_results.csv", index=False)
    print(f"Saved inference_results.csv with {len(inf_df)} rows")

print("\nDone!")
