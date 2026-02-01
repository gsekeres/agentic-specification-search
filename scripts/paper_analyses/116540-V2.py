"""
Specification Search Analysis for Paper 116540-V2
AEJ: Policy - American Dream Demonstration: Individual Development Accounts and Homeownership

This paper is a randomized controlled trial studying the effect of Individual Development
Accounts (IDAs) - matched savings accounts - on homeownership. Treatment is random assignment
to IDA eligibility. Main outcome is homeownership at Wave 4 (approximately 10 years after baseline).

Method: Cross-sectional OLS (RCT comparison at follow-up)
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: Load and prepare data
# ============================================================================

df = pd.read_stata('/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/116540-V2/AEJPol-2011-0054_R1_data/AEJPol-2011-0054_R1_shared_data_file.dta')

# Convert categorical variables to numeric
# Treatment
df['treat'] = (df['treat'] == 'Treatment').astype(int)

# Outcomes
df['own_home_u17'] = (df['own_home_u17'] == 'yes').astype(int)
df['own_home_u42'] = (df['own_home_u42'] == 'yes').astype(float)
df.loc[df['own_home_u42'].isna(), 'own_home_u42'] = np.nan

# Sample indicator
df['sample14'] = (df['sample14'] == 'yes').astype(int)

# Female
df['female_u17'] = (df['female_u17'] == 'female').astype(float)

# Race
df['race_cau_u17'] = (df['race_cau_u17'] == 'yes').astype(float)

# Marital status - married vs not
df['married_u17'] = (df['marital_u17'] == 'married').astype(float)

# Binary controls
for var in ['ins_heal_u17', 'src_welf_u17', 'own_car_u17', 'own_prop_u17',
            'own_bus_u17', 'own_ira_u17', 'own_bank_u17', 'pub_home_u17', 'section8_u17']:
    df[var] = (df[var] == 'yes').astype(float)

# Unsubsidized housing (not section 8, not public housing)
df['unsubsidized'] = ((df['section8_u17'] == 0) & (df['pub_home_u17'] == 0)).astype(float)

# Age binary (35+)
df['bin_age_u17'] = (df['age_u17'] > 34).astype(float)
df.loc[df['age_u17'].isna(), 'bin_age_u17'] = np.nan

# Education trichotomy
ed_map = {
    'grade school, middle school, or jr high': 0,
    'some hs': 0,
    'graduate hs or ged': 0,
    'some college': 1,
    'grad 2yr college': 1,
    'grad 4yr college': 2,
    'some grad school': 2,
    'finished grad school': 2
}
df['tri_ed_u17'] = df['ed_u17'].map(ed_map).astype(float)

# Income binary (above median)
df['hiinc'] = np.nan
# Will compute on analytic sample below

# Asset/liability categories
def categorize_assets(x):
    if pd.isna(x):
        return 4  # missing
    elif x < 1421:
        return 0
    elif x < 2842:
        return 1
    elif x < 4263:
        return 2
    else:
        return 3

df['cat_ass_tot'] = df['ass_tot_u17'].apply(categorize_assets)
df['cat_lib_tot'] = df['lib_tot_u17'].apply(categorize_assets)

# Cohort binary (late cohorts 12-13)
df['bin_cohort'] = ((df['cohort'] == 12) | (df['cohort'] == 13)).astype(float)

# Children binary
df['bin_child_u17'] = (df['hh_child_u17'] > 0).astype(float)

# Health satisfaction binary (top 2 categories)
sat_heal_map = {'top category': 0, 1: 1, 2: 2, 3: 3, 'bottom category': 4}
df['sat_heal_num'] = df['sat_heal_u17'].map(sat_heal_map)
df['bin_sat_heal_u17'] = (df['sat_heal_num'] <= 1).astype(float)
df.loc[df['sat_heal_num'].isna(), 'bin_sat_heal_u17'] = np.nan

# Financial satisfaction binary (top 2 categories)
sat_fin_map = {'very satisfied': 0, 'somewhat satisfied': 1, 'somewhat dissatisfied': 2, 'dissatisfied': 3}
df['sat_fin_num'] = df['sat_fin2_u17'].map(sat_fin_map)
df['bin_sat_fin2_u17'] = (df['sat_fin_num'] <= 1).astype(float)
df.loc[df['sat_fin_num'].isna(), 'bin_sat_fin2_u17'] = np.nan

# Scale variables - set to missing if miss_* == 3
for scale, miss in [('str_scale_u17', 'miss_str'), ('gv_scale_u17', 'miss_gv'),
                     ('gt_scale_u17', 'miss_gt'), ('own_scale_u17', 'miss_own'),
                     ('ci_scale_u17', 'miss_ci')]:
    df[f'{scale}_clean'] = df[scale].copy()
    df.loc[df[miss] == 3, f'{scale}_clean'] = np.nan

# ============================================================================
# STEP 2: Create analytic sample
# ============================================================================

# Keep baseline renters only
df = df[df['own_home_u17'] == 0].copy()

# Compute income median on wave 4 sample
median_inc = df.loc[df['sample14'] == 1, 'inc_tot_u17'].median()
df['hiinc'] = (df['inc_tot_u17'] >= median_inc).astype(float)
df.loc[df['inc_tot_u17'].isna(), 'hiinc'] = np.nan

# Define full control list (matching the do file)
controls_full = [
    'unsubsidized', 'bin_age_u17', 'hiinc',
    'female_u17', 'race_cau_u17', 'married_u17', 'own_bank_u17',
    'bin_cohort', 'ins_heal_u17', 'hh_adult_u17', 'bin_child_u17',
    'own_bus_u17', 'own_prop_u17', 'own_ira_u17', 'src_welf_u17',
    'own_car_u17', 'own_scale_u17', 'str_scale_u17_clean',
    'gv_scale_u17_clean', 'gt_scale_u17_clean',
    'bin_sat_heal_u17', 'bin_sat_fin2_u17', 'ci_scale_u17'
]

# Add categorical variables as dummies
df['tri_ed_1'] = (df['tri_ed_u17'] == 1).astype(float)
df['tri_ed_2'] = (df['tri_ed_u17'] == 2).astype(float)
df['cat_ass_1'] = (df['cat_ass_tot'] == 1).astype(float)
df['cat_ass_2'] = (df['cat_ass_tot'] == 2).astype(float)
df['cat_ass_3'] = (df['cat_ass_tot'] == 3).astype(float)
df['cat_ass_4'] = (df['cat_ass_tot'] == 4).astype(float)  # missing
df['cat_lib_1'] = (df['cat_lib_tot'] == 1).astype(float)
df['cat_lib_2'] = (df['cat_lib_tot'] == 2).astype(float)
df['cat_lib_3'] = (df['cat_lib_tot'] == 3).astype(float)
df['cat_lib_4'] = (df['cat_lib_tot'] == 4).astype(float)  # missing

# Update full controls list with dummies
controls_with_dummies = [
    'unsubsidized', 'bin_age_u17', 'hiinc',
    'tri_ed_1', 'tri_ed_2',
    'cat_ass_1', 'cat_ass_2', 'cat_ass_3', 'cat_ass_4',
    'cat_lib_1', 'cat_lib_2', 'cat_lib_3', 'cat_lib_4',
    'female_u17', 'race_cau_u17', 'married_u17', 'own_bank_u17',
    'bin_cohort', 'ins_heal_u17', 'hh_adult_u17', 'bin_child_u17',
    'own_bus_u17', 'own_prop_u17', 'own_ira_u17', 'src_welf_u17',
    'own_car_u17', 'own_scale_u17', 'str_scale_u17_clean',
    'gv_scale_u17_clean', 'gt_scale_u17_clean',
    'bin_sat_heal_u17', 'bin_sat_fin2_u17', 'ci_scale_u17'
]

# Create analytic sample flag
all_vars = ['treat', 'own_home_u42'] + controls_with_dummies
df['miss_any'] = df[all_vars].isna().any(axis=1)
df['analytic'] = (df['miss_any'] == False) & (df['sample14'] == 1)

print(f"Total baseline renters: {len(df)}")
print(f"In Wave 4: {df['sample14'].sum()}")
print(f"Analytic sample (complete cases): {df['analytic'].sum()}")

# Create analytic dataframe
df_analytic = df[df['analytic'] == True].copy()

# ============================================================================
# STEP 3: Run specifications
# ============================================================================

results = []

# Paper metadata
paper_id = "116540-V2"
journal = "AEJ: Policy"
paper_title = "Can Individual Development Accounts Affect Homeownership? Evidence from a Randomized Experiment"

def run_ols(df, outcome_var, treatment_var, controls, se_type='HC1', spec_id='', spec_tree_path=''):
    """
    Run OLS regression and return results dictionary
    """
    if len(controls) > 0:
        formula = f"{outcome_var} ~ {treatment_var} + " + " + ".join(controls)
    else:
        formula = f"{outcome_var} ~ {treatment_var}"

    try:
        model = smf.ols(formula, data=df).fit(cov_type=se_type)

        # Extract coefficients
        coef = model.params[treatment_var]
        se = model.bse[treatment_var]
        tstat = model.tvalues[treatment_var]
        pval = model.pvalues[treatment_var]
        ci = model.conf_int().loc[treatment_var]

        # Build coefficient vector
        coef_vector = {
            "treatment": {
                "var": treatment_var,
                "coef": float(coef),
                "se": float(se),
                "pval": float(pval)
            },
            "controls": [],
            "diagnostics": {
                "r_squared": float(model.rsquared),
                "adj_r_squared": float(model.rsquared_adj),
                "f_stat": float(model.fvalue) if model.fvalue else None,
                "f_pval": float(model.f_pvalue) if model.f_pvalue else None
            }
        }

        # Add control coefficients
        for var in controls:
            if var in model.params:
                coef_vector["controls"].append({
                    "var": var,
                    "coef": float(model.params[var]),
                    "se": float(model.bse[var]),
                    "pval": float(model.pvalues[var])
                })

        return {
            'paper_id': paper_id,
            'journal': journal,
            'paper_title': paper_title,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': float(coef),
            'std_error': float(se),
            't_stat': float(tstat),
            'p_value': float(pval),
            'ci_lower': float(ci[0]),
            'ci_upper': float(ci[1]),
            'n_obs': int(model.nobs),
            'r_squared': float(model.rsquared),
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': 'Baseline renters in Wave 4 with complete data',
            'fixed_effects': 'None',
            'controls_desc': ', '.join(controls) if controls else 'None',
            'cluster_var': 'None (robust SE)',
            'model_type': 'OLS',
            'estimation_script': f'scripts/paper_analyses/{paper_id}.py'
        }
    except Exception as e:
        print(f"Error in {spec_id}: {e}")
        return None

# ============================================================================
# BASELINE SPECIFICATION
# ============================================================================

# Baseline: Full controls with robust SE (matching Table 5)
result = run_ols(
    df_analytic,
    'own_home_u42',
    'treat',
    controls_with_dummies,
    se_type='HC1',
    spec_id='baseline',
    spec_tree_path='methods/cross_sectional_ols.md'
)
if result:
    results.append(result)
    baseline_coef = result['coefficient']
    baseline_se = result['std_error']
    print(f"\nBaseline: coef={baseline_coef:.4f}, se={baseline_se:.4f}, p={result['p_value']:.4f}")

# ============================================================================
# METHOD VARIATIONS (Cross-sectional OLS)
# ============================================================================

# 1. No controls (bivariate)
result = run_ols(
    df_analytic,
    'own_home_u42',
    'treat',
    [],
    se_type='HC1',
    spec_id='ols/controls/none',
    spec_tree_path='methods/cross_sectional_ols.md#control-sets'
)
if result:
    results.append(result)

# 2. Demographics only
demo_controls = ['bin_age_u17', 'female_u17', 'race_cau_u17', 'married_u17',
                 'hh_adult_u17', 'bin_child_u17', 'tri_ed_1', 'tri_ed_2']
result = run_ols(
    df_analytic,
    'own_home_u42',
    'treat',
    demo_controls,
    se_type='HC1',
    spec_id='ols/controls/demographics',
    spec_tree_path='methods/cross_sectional_ols.md#control-sets'
)
if result:
    results.append(result)

# 3. Classical SE (homoskedastic)
result = run_ols(
    df_analytic,
    'own_home_u42',
    'treat',
    controls_with_dummies,
    se_type='nonrobust',
    spec_id='ols/se/classical',
    spec_tree_path='methods/cross_sectional_ols.md#standard-errors'
)
if result:
    results.append(result)

# 4. HC2 robust SE
result = run_ols(
    df_analytic,
    'own_home_u42',
    'treat',
    controls_with_dummies,
    se_type='HC2',
    spec_id='ols/se/hc2',
    spec_tree_path='methods/cross_sectional_ols.md#standard-errors'
)
if result:
    results.append(result)

# 5. HC3 robust SE
result = run_ols(
    df_analytic,
    'own_home_u42',
    'treat',
    controls_with_dummies,
    se_type='HC3',
    spec_id='ols/se/hc3',
    spec_tree_path='methods/cross_sectional_ols.md#standard-errors'
)
if result:
    results.append(result)

# ============================================================================
# SUBGROUP ANALYSES
# ============================================================================

# Unsubsidized renters only
df_unsub = df_analytic[df_analytic['unsubsidized'] == 1].copy()
controls_unsub = [c for c in controls_with_dummies if c != 'unsubsidized']
result = run_ols(
    df_unsub,
    'own_home_u42',
    'treat',
    controls_unsub,
    se_type='HC1',
    spec_id='ols/sample/subgroup_unsubsidized',
    spec_tree_path='methods/cross_sectional_ols.md#sample-restrictions'
)
if result:
    result['sample_desc'] = 'Unsubsidized baseline renters in Wave 4'
    results.append(result)

# Female subsample
df_female = df_analytic[df_analytic['female_u17'] == 1].copy()
controls_female = [c for c in controls_with_dummies if c != 'female_u17']
result = run_ols(
    df_female,
    'own_home_u42',
    'treat',
    controls_female,
    se_type='HC1',
    spec_id='ols/sample/subgroup_female',
    spec_tree_path='methods/cross_sectional_ols.md#sample-restrictions'
)
if result:
    result['sample_desc'] = 'Female baseline renters in Wave 4'
    results.append(result)

# Male subsample
df_male = df_analytic[df_analytic['female_u17'] == 0].copy()
controls_male = [c for c in controls_with_dummies if c != 'female_u17']
result = run_ols(
    df_male,
    'own_home_u42',
    'treat',
    controls_male,
    se_type='HC1',
    spec_id='ols/sample/subgroup_male',
    spec_tree_path='methods/cross_sectional_ols.md#sample-restrictions'
)
if result:
    result['sample_desc'] = 'Male baseline renters in Wave 4'
    results.append(result)

# High income subsample
df_hi = df_analytic[df_analytic['hiinc'] == 1].copy()
controls_hi = [c for c in controls_with_dummies if c != 'hiinc']
result = run_ols(
    df_hi,
    'own_home_u42',
    'treat',
    controls_hi,
    se_type='HC1',
    spec_id='ols/sample/subgroup_high_income',
    spec_tree_path='methods/cross_sectional_ols.md#sample-restrictions'
)
if result:
    result['sample_desc'] = 'Above-median income baseline renters in Wave 4'
    results.append(result)

# Low income subsample
df_lo = df_analytic[df_analytic['hiinc'] == 0].copy()
controls_lo = [c for c in controls_with_dummies if c != 'hiinc']
result = run_ols(
    df_lo,
    'own_home_u42',
    'treat',
    controls_lo,
    se_type='HC1',
    spec_id='ols/sample/subgroup_low_income',
    spec_tree_path='methods/cross_sectional_ols.md#sample-restrictions'
)
if result:
    result['sample_desc'] = 'Below-median income baseline renters in Wave 4'
    results.append(result)

# ============================================================================
# ROBUSTNESS: LEAVE-ONE-OUT
# ============================================================================

# Group controls logically for leave-one-out
control_groups = {
    'unsubsidized': ['unsubsidized'],
    'age': ['bin_age_u17'],
    'income': ['hiinc'],
    'education': ['tri_ed_1', 'tri_ed_2'],
    'assets': ['cat_ass_1', 'cat_ass_2', 'cat_ass_3', 'cat_ass_4'],
    'liabilities': ['cat_lib_1', 'cat_lib_2', 'cat_lib_3', 'cat_lib_4'],
    'female': ['female_u17'],
    'race': ['race_cau_u17'],
    'married': ['married_u17'],
    'bank_account': ['own_bank_u17'],
    'cohort': ['bin_cohort'],
    'health_ins': ['ins_heal_u17'],
    'household_size': ['hh_adult_u17', 'bin_child_u17'],
    'assets_owned': ['own_bus_u17', 'own_prop_u17', 'own_ira_u17', 'own_car_u17'],
    'welfare': ['src_welf_u17'],
    'scales': ['own_scale_u17', 'str_scale_u17_clean', 'gv_scale_u17_clean',
               'gt_scale_u17_clean', 'ci_scale_u17'],
    'satisfaction': ['bin_sat_heal_u17', 'bin_sat_fin2_u17']
}

for group_name, vars_to_drop in control_groups.items():
    remaining_controls = [c for c in controls_with_dummies if c not in vars_to_drop]
    result = run_ols(
        df_analytic,
        'own_home_u42',
        'treat',
        remaining_controls,
        se_type='HC1',
        spec_id=f'robust/loo/drop_{group_name}',
        spec_tree_path='robustness/leave_one_out.md'
    )
    if result:
        results.append(result)

# ============================================================================
# ROBUSTNESS: SINGLE COVARIATE
# ============================================================================

# Bivariate already done above

# Single covariate for key variables
single_vars = ['unsubsidized', 'bin_age_u17', 'hiinc', 'female_u17', 'race_cau_u17',
               'married_u17', 'own_bank_u17', 'bin_cohort', 'ins_heal_u17',
               'src_welf_u17', 'own_car_u17']

for var in single_vars:
    result = run_ols(
        df_analytic,
        'own_home_u42',
        'treat',
        [var],
        se_type='HC1',
        spec_id=f'robust/single/{var}',
        spec_tree_path='robustness/single_covariate.md'
    )
    if result:
        results.append(result)

# ============================================================================
# ROBUSTNESS: FUNCTIONAL FORM
# ============================================================================

# Probit model for binary outcome
from statsmodels.discrete.discrete_model import Probit

try:
    X = df_analytic[['treat'] + controls_with_dummies]
    X = sm.add_constant(X)
    y = df_analytic['own_home_u42']
    probit_model = Probit(y, X).fit(disp=0)

    # Get marginal effects
    mfx = probit_model.get_margeff(at='mean')

    coef = mfx.margeff[0]  # treat is first variable
    se = mfx.margeff_se[0]
    pval = mfx.pvalues[0]
    tstat = coef / se
    ci_lower = coef - 1.96 * se
    ci_upper = coef + 1.96 * se

    coef_vector = {
        "treatment": {"var": "treat", "coef": float(coef), "se": float(se), "pval": float(pval)},
        "controls": [],
        "diagnostics": {"pseudo_r_squared": float(probit_model.prsquared)}
    }

    results.append({
        'paper_id': paper_id,
        'journal': journal,
        'paper_title': paper_title,
        'spec_id': 'robust/form/probit',
        'spec_tree_path': 'robustness/functional_form.md#alternative-estimators',
        'outcome_var': 'own_home_u42',
        'treatment_var': 'treat',
        'coefficient': float(coef),
        'std_error': float(se),
        't_stat': float(tstat),
        'p_value': float(pval),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'n_obs': int(probit_model.nobs),
        'r_squared': float(probit_model.prsquared),
        'coefficient_vector_json': json.dumps(coef_vector),
        'sample_desc': 'Baseline renters in Wave 4 with complete data',
        'fixed_effects': 'None',
        'controls_desc': ', '.join(controls_with_dummies),
        'cluster_var': 'None',
        'model_type': 'Probit (marginal effects)',
        'estimation_script': f'scripts/paper_analyses/{paper_id}.py'
    })
except Exception as e:
    print(f"Probit error: {e}")

# Logit model
from statsmodels.discrete.discrete_model import Logit

try:
    X = df_analytic[['treat'] + controls_with_dummies]
    X = sm.add_constant(X)
    y = df_analytic['own_home_u42']
    logit_model = Logit(y, X).fit(disp=0)

    # Get marginal effects
    mfx = logit_model.get_margeff(at='mean')

    coef = mfx.margeff[0]
    se = mfx.margeff_se[0]
    pval = mfx.pvalues[0]
    tstat = coef / se
    ci_lower = coef - 1.96 * se
    ci_upper = coef + 1.96 * se

    coef_vector = {
        "treatment": {"var": "treat", "coef": float(coef), "se": float(se), "pval": float(pval)},
        "controls": [],
        "diagnostics": {"pseudo_r_squared": float(logit_model.prsquared)}
    }

    results.append({
        'paper_id': paper_id,
        'journal': journal,
        'paper_title': paper_title,
        'spec_id': 'robust/form/logit',
        'spec_tree_path': 'robustness/functional_form.md#alternative-estimators',
        'outcome_var': 'own_home_u42',
        'treatment_var': 'treat',
        'coefficient': float(coef),
        'std_error': float(se),
        't_stat': float(tstat),
        'p_value': float(pval),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'n_obs': int(logit_model.nobs),
        'r_squared': float(logit_model.prsquared),
        'coefficient_vector_json': json.dumps(coef_vector),
        'sample_desc': 'Baseline renters in Wave 4 with complete data',
        'fixed_effects': 'None',
        'controls_desc': ', '.join(controls_with_dummies),
        'cluster_var': 'None',
        'model_type': 'Logit (marginal effects)',
        'estimation_script': f'scripts/paper_analyses/{paper_id}.py'
    })
except Exception as e:
    print(f"Logit error: {e}")

# ============================================================================
# ROBUSTNESS: COHORT CLUSTERING
# ============================================================================

# Cluster by cohort
try:
    formula = "own_home_u42 ~ treat + " + " + ".join(controls_with_dummies)
    model = smf.ols(formula, data=df_analytic).fit(
        cov_type='cluster',
        cov_kwds={'groups': df_analytic['cohort']}
    )

    coef = model.params['treat']
    se = model.bse['treat']
    tstat = model.tvalues['treat']
    pval = model.pvalues['treat']
    ci = model.conf_int().loc['treat']

    coef_vector = {
        "treatment": {"var": "treat", "coef": float(coef), "se": float(se), "pval": float(pval)},
        "controls": [],
        "diagnostics": {"n_clusters": int(df_analytic['cohort'].nunique())}
    }

    results.append({
        'paper_id': paper_id,
        'journal': journal,
        'paper_title': paper_title,
        'spec_id': 'robust/cluster/cohort',
        'spec_tree_path': 'robustness/clustering_variations.md',
        'outcome_var': 'own_home_u42',
        'treatment_var': 'treat',
        'coefficient': float(coef),
        'std_error': float(se),
        't_stat': float(tstat),
        'p_value': float(pval),
        'ci_lower': float(ci[0]),
        'ci_upper': float(ci[1]),
        'n_obs': int(model.nobs),
        'r_squared': float(model.rsquared),
        'coefficient_vector_json': json.dumps(coef_vector),
        'sample_desc': 'Baseline renters in Wave 4 with complete data',
        'fixed_effects': 'None',
        'controls_desc': ', '.join(controls_with_dummies),
        'cluster_var': 'cohort',
        'model_type': 'OLS',
        'estimation_script': f'scripts/paper_analyses/{paper_id}.py'
    })
except Exception as e:
    print(f"Cluster by cohort error: {e}")

# ============================================================================
# INTERACTIONS WITH TREATMENT
# ============================================================================

# Treatment x Female interaction
try:
    df_analytic['treat_x_female'] = df_analytic['treat'] * df_analytic['female_u17']
    controls_interact = controls_with_dummies + ['treat_x_female']

    formula = "own_home_u42 ~ treat + " + " + ".join(controls_interact)
    model = smf.ols(formula, data=df_analytic).fit(cov_type='HC1')

    # Main treatment effect
    coef = model.params['treat']
    se = model.bse['treat']
    tstat = model.tvalues['treat']
    pval = model.pvalues['treat']
    ci = model.conf_int().loc['treat']

    # Interaction effect
    int_coef = model.params['treat_x_female']
    int_se = model.bse['treat_x_female']
    int_pval = model.pvalues['treat_x_female']

    coef_vector = {
        "treatment": {"var": "treat", "coef": float(coef), "se": float(se), "pval": float(pval)},
        "interaction": {"var": "treat_x_female", "coef": float(int_coef), "se": float(int_se), "pval": float(int_pval)},
        "controls": [],
        "diagnostics": {}
    }

    results.append({
        'paper_id': paper_id,
        'journal': journal,
        'paper_title': paper_title,
        'spec_id': 'ols/interact/gender',
        'spec_tree_path': 'methods/cross_sectional_ols.md#interaction-effects',
        'outcome_var': 'own_home_u42',
        'treatment_var': 'treat',
        'coefficient': float(coef),
        'std_error': float(se),
        't_stat': float(tstat),
        'p_value': float(pval),
        'ci_lower': float(ci[0]),
        'ci_upper': float(ci[1]),
        'n_obs': int(model.nobs),
        'r_squared': float(model.rsquared),
        'coefficient_vector_json': json.dumps(coef_vector),
        'sample_desc': 'Baseline renters in Wave 4 with complete data',
        'fixed_effects': 'None',
        'controls_desc': ', '.join(controls_with_dummies) + ' + treat_x_female',
        'cluster_var': 'None (robust SE)',
        'model_type': 'OLS',
        'estimation_script': f'scripts/paper_analyses/{paper_id}.py'
    })
except Exception as e:
    print(f"Interaction error: {e}")

# Treatment x Income interaction
try:
    df_analytic['treat_x_hiinc'] = df_analytic['treat'] * df_analytic['hiinc']
    controls_interact = controls_with_dummies + ['treat_x_hiinc']

    formula = "own_home_u42 ~ treat + " + " + ".join(controls_interact)
    model = smf.ols(formula, data=df_analytic).fit(cov_type='HC1')

    coef = model.params['treat']
    se = model.bse['treat']
    tstat = model.tvalues['treat']
    pval = model.pvalues['treat']
    ci = model.conf_int().loc['treat']

    int_coef = model.params['treat_x_hiinc']
    int_se = model.bse['treat_x_hiinc']
    int_pval = model.pvalues['treat_x_hiinc']

    coef_vector = {
        "treatment": {"var": "treat", "coef": float(coef), "se": float(se), "pval": float(pval)},
        "interaction": {"var": "treat_x_hiinc", "coef": float(int_coef), "se": float(int_se), "pval": float(int_pval)},
        "controls": [],
        "diagnostics": {}
    }

    results.append({
        'paper_id': paper_id,
        'journal': journal,
        'paper_title': paper_title,
        'spec_id': 'ols/interact/income',
        'spec_tree_path': 'methods/cross_sectional_ols.md#interaction-effects',
        'outcome_var': 'own_home_u42',
        'treatment_var': 'treat',
        'coefficient': float(coef),
        'std_error': float(se),
        't_stat': float(tstat),
        'p_value': float(pval),
        'ci_lower': float(ci[0]),
        'ci_upper': float(ci[1]),
        'n_obs': int(model.nobs),
        'r_squared': float(model.rsquared),
        'coefficient_vector_json': json.dumps(coef_vector),
        'sample_desc': 'Baseline renters in Wave 4 with complete data',
        'fixed_effects': 'None',
        'controls_desc': ', '.join(controls_with_dummies) + ' + treat_x_hiinc',
        'cluster_var': 'None (robust SE)',
        'model_type': 'OLS',
        'estimation_script': f'scripts/paper_analyses/{paper_id}.py'
    })
except Exception as e:
    print(f"Income interaction error: {e}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

# Convert to dataframe
results_df = pd.DataFrame(results)

# Save to package directory
output_path = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/116540-V2/specification_results.csv'
results_df.to_csv(output_path, index=False)

print(f"\n{'='*60}")
print(f"RESULTS SUMMARY")
print(f"{'='*60}")
print(f"Total specifications run: {len(results_df)}")
print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")
print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")
print(f"Range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")
print(f"\nResults saved to: {output_path}")
