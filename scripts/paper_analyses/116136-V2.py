"""
Specification Search: 116136-V2
Paper: "Yours, Mine and Ours: Do Divorce Laws Affect the Intertemporal Behavior of Married Couples?"
Author: Alessandra Voena (American Economic Review, 2015)

Method: Panel Fixed Effects with DiD-style treatment (unilateral divorce * property regime interactions)

Primary outcome: household assets (NLSW data)
Treatment: uni_comprop (unilateral divorce * community property interaction)
    - Also: uni_title, uni_eqdistr, comprop, eqdistr as additional treatments/controls
Controls: age dummies (d_age*), year dummies (yrd*), state dummies (std*),
          children dummies (chd*), time since marriage (since1marr*)
Fixed effects: Individual (id)
Clustering: State (state)
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

import pyfixest as pf

# Set paths
DATA_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/116136-V2/replicate_empirics"
OUTPUT_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/116136-V2"

# Paper metadata
PAPER_ID = "116136-V2"
JOURNAL = "AER"
PAPER_TITLE = "Yours, Mine and Ours: Do Divorce Laws Affect the Intertemporal Behavior of Married Couples?"

# Load data
print("Loading data...")
nlsw = pd.read_stata(f"{DATA_PATH}/NLSW_women.dta")
print(f"NLSW data loaded: {nlsw.shape}")

# Prepare data
# Get column lists
d_age_cols = [c for c in nlsw.columns if c.startswith('d_age')]
yrd_cols = [c for c in nlsw.columns if c.startswith('yrd')]
chd_cols = [c for c in nlsw.columns if c.startswith('chd')]
std_cols = [c for c in nlsw.columns if c.startswith('std')]
since_cols = [c for c in nlsw.columns if c.startswith('since1marr')]

# Main treatment and property regime variables
treatment_vars = ['uni_comprop', 'uni_title', 'uni_eqdistr']
property_vars = ['comprop', 'eqdistr']

# Create additional variables for robustness
nlsw['age_sq'] = nlsw['age'] ** 2
nlsw['log_assets'] = np.log(nlsw['assets'].clip(lower=1))
nlsw['assets_pos'] = nlsw['assets'].clip(lower=0)
nlsw['log_assets_pos'] = np.log(nlsw['assets_pos'] + 1)
nlsw['ihs_assets'] = np.arcsinh(nlsw['assets'])

# Create year variable from dummies if needed
if 'year' in nlsw.columns:
    pass
else:
    nlsw['year'] = 1967 + nlsw[[c for c in nlsw.columns if c.startswith('yrd')]].values.argmax(axis=1)

# Winsorize assets at different levels
for pct in [1, 5, 10]:
    lower = nlsw['assets'].quantile(pct/100)
    upper = nlsw['assets'].quantile(1 - pct/100)
    nlsw[f'assets_wins_{pct}'] = nlsw['assets'].clip(lower=lower, upper=upper)

# Ensure id and state are integer for clustering
nlsw['id'] = nlsw['id'].astype('Int64')
nlsw['state'] = nlsw['state'].astype('Int64')

# Define control sets
controls_basic = d_age_cols + yrd_cols  # Age and year dummies only
controls_children = controls_basic + chd_cols  # Add children
controls_state = controls_children + std_cols  # Add state dummies (alternative to state FE)
controls_marriage = controls_children + since_cols  # Add time since marriage
controls_full = controls_marriage  # Full baseline as in paper

# Filter to sample with non-missing assets (for regressions)
df = nlsw[nlsw['assets'].notna()].copy()
print(f"Sample with non-missing assets: {len(df)}")

# Results storage
results = []

def run_regression(formula, data, cluster_var='state', spec_id='', spec_tree_path='',
                   outcome_var='assets', treatment_var='uni_comprop',
                   sample_desc='Full sample', fixed_effects='Individual FE',
                   controls_desc='', model_type='Panel FE'):
    """Run a regression and store results"""
    try:
        # Use pyfixest with correct API
        if cluster_var and cluster_var in data.columns:
            model = pf.feols(formula, data=data, vcov={'CRV1': cluster_var})
        else:
            model = pf.feols(formula, data=data, vcov='hetero')

        # Get coefficient for treatment variable
        coefs = model.coef()
        ses = model.se()
        pvals = model.pvalue()
        tstats = model.tstat()

        # Find the treatment coefficient
        if treatment_var in coefs.index:
            coef = coefs[treatment_var]
            se = ses[treatment_var]
            pval = pvals[treatment_var]
            tstat = tstats[treatment_var]
            ci = model.confint()
            ci_lower = ci.loc[treatment_var, '2.5%']
            ci_upper = ci.loc[treatment_var, '97.5%']
        else:
            # Treatment variable might have different name
            coef = se = pval = tstat = ci_lower = ci_upper = np.nan

        # Use private attributes for model stats
        n_obs = model._N
        r_sq = model._r2
        r_sq_within = model._r2_within if hasattr(model, '_r2_within') else np.nan

        # Build coefficient vector
        coef_vector = {
            "treatment": {
                "var": treatment_var,
                "coef": float(coef) if not np.isnan(coef) else None,
                "se": float(se) if not np.isnan(se) else None,
                "pval": float(pval) if not np.isnan(pval) else None
            },
            "controls": [],
            "fixed_effects_absorbed": fixed_effects.split(', ') if fixed_effects else [],
            "n_obs": int(n_obs),
            "r_squared": float(r_sq) if not np.isnan(r_sq) else None,
            "r_squared_within": float(r_sq_within) if not np.isnan(r_sq_within) else None
        }

        # Add other treatment variables to coefficient vector
        for var in treatment_vars + property_vars:
            if var in coefs.index and var != treatment_var:
                coef_vector["controls"].append({
                    "var": var,
                    "coef": float(coefs[var]),
                    "se": float(ses[var]),
                    "pval": float(pvals[var])
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
            't_stat': tstat,
            'p_value': pval,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_obs': n_obs,
            'r_squared': r_sq,
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects,
            'controls_desc': controls_desc,
            'cluster_var': cluster_var if cluster_var else 'robust',
            'model_type': model_type,
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
        return result

    except Exception as e:
        print(f"Error in {spec_id}: {str(e)[:100]}")
        return None

print("\n" + "="*60)
print("RUNNING SPECIFICATION SEARCH")
print("="*60)

# ===========================================================================
# BASELINE SPECIFICATIONS (exact replication of paper)
# ===========================================================================
print("\n--- Baseline Specifications ---")

# Baseline 1: Table 1 Column 1 - basic controls
formula = f"assets ~ {' + '.join(treatment_vars + property_vars + controls_basic)} | id"
result = run_regression(formula, df, spec_id='baseline',
                        spec_tree_path='methods/panel_fixed_effects.md#baseline',
                        controls_desc='Age + Year dummies')
if result: results.append(result)

# Baseline 2: Table 1 Column 2 - add children controls
formula = f"assets ~ {' + '.join(treatment_vars + property_vars + controls_children)} | id"
result = run_regression(formula, df, spec_id='baseline/children',
                        spec_tree_path='methods/panel_fixed_effects.md#baseline',
                        controls_desc='Age + Year + Children dummies')
if result: results.append(result)

# Baseline 3: Table 1 Column 3 - add state dummies
formula = f"assets ~ {' + '.join(treatment_vars + property_vars + controls_state)} | id"
result = run_regression(formula, df, spec_id='baseline/state_controls',
                        spec_tree_path='methods/panel_fixed_effects.md#baseline',
                        controls_desc='Age + Year + Children + State dummies')
if result: results.append(result)

# Baseline 4: Table 1 Column 4 - full controls (main specification)
formula = f"assets ~ {' + '.join(treatment_vars + property_vars + controls_full)} | id"
result = run_regression(formula, df, spec_id='baseline/full',
                        spec_tree_path='methods/panel_fixed_effects.md#baseline',
                        controls_desc='Full controls (Age + Year + Children + Marriage duration)')
if result: results.append(result)

# ===========================================================================
# FIXED EFFECTS VARIATIONS
# ===========================================================================
print("\n--- Fixed Effects Variations ---")

# No fixed effects (pooled OLS)
formula = f"assets ~ {' + '.join(treatment_vars + property_vars + controls_full)}"
result = run_regression(formula, df, spec_id='panel/fe/none',
                        spec_tree_path='methods/panel_fixed_effects.md#fixed-effects-structure',
                        fixed_effects='None (pooled OLS)',
                        controls_desc='Full controls')
if result: results.append(result)

# Year fixed effects only (simpler specification)
formula = f"assets ~ {' + '.join(treatment_vars + property_vars + d_age_cols + chd_cols + since_cols)} | year"
result = run_regression(formula, df, spec_id='panel/fe/time',
                        spec_tree_path='methods/panel_fixed_effects.md#fixed-effects-structure',
                        fixed_effects='Year FE',
                        controls_desc='Age + Children + Marriage duration')
if result: results.append(result)

# State fixed effects only
formula = f"assets ~ {' + '.join(treatment_vars + property_vars + controls_basic + chd_cols + since_cols)} | state"
result = run_regression(formula, df, spec_id='panel/fe/state',
                        spec_tree_path='methods/panel_fixed_effects.md#fixed-effects-structure',
                        fixed_effects='State FE',
                        controls_desc='Age + Year + Children + Marriage duration')
if result: results.append(result)

# Two-way FE: individual + year
formula = f"assets ~ {' + '.join(treatment_vars + property_vars + d_age_cols + chd_cols + since_cols)} | id + year"
result = run_regression(formula, df, spec_id='panel/fe/twoway',
                        spec_tree_path='methods/panel_fixed_effects.md#fixed-effects-structure',
                        fixed_effects='Individual + Year FE',
                        controls_desc='Age + Children + Marriage duration')
if result: results.append(result)

# Individual + State FE
formula = f"assets ~ {' + '.join(treatment_vars + property_vars + controls_basic + chd_cols + since_cols)} | id + state"
result = run_regression(formula, df, spec_id='panel/fe/id_state',
                        spec_tree_path='methods/panel_fixed_effects.md#fixed-effects-structure',
                        fixed_effects='Individual + State FE',
                        controls_desc='Age + Year + Children + Marriage duration')
if result: results.append(result)

# ===========================================================================
# CONTROL SET VARIATIONS
# ===========================================================================
print("\n--- Control Set Variations ---")

# No controls (treatment + FE only)
formula = f"assets ~ {' + '.join(treatment_vars + property_vars)} | id"
result = run_regression(formula, df, spec_id='panel/controls/none',
                        spec_tree_path='methods/panel_fixed_effects.md#control-sets',
                        controls_desc='None (treatment only)')
if result: results.append(result)

# Age controls only
formula = f"assets ~ {' + '.join(treatment_vars + property_vars + d_age_cols)} | id"
result = run_regression(formula, df, spec_id='panel/controls/age_only',
                        spec_tree_path='methods/panel_fixed_effects.md#control-sets',
                        controls_desc='Age dummies only')
if result: results.append(result)

# Year controls only
formula = f"assets ~ {' + '.join(treatment_vars + property_vars + yrd_cols)} | id"
result = run_regression(formula, df, spec_id='panel/controls/year_only',
                        spec_tree_path='methods/panel_fixed_effects.md#control-sets',
                        controls_desc='Year dummies only')
if result: results.append(result)

# Children controls only
formula = f"assets ~ {' + '.join(treatment_vars + property_vars + chd_cols)} | id"
result = run_regression(formula, df, spec_id='panel/controls/children_only',
                        spec_tree_path='methods/panel_fixed_effects.md#control-sets',
                        controls_desc='Children dummies only')
if result: results.append(result)

# Marriage duration controls only
formula = f"assets ~ {' + '.join(treatment_vars + property_vars + since_cols)} | id"
result = run_regression(formula, df, spec_id='panel/controls/marriage_only',
                        spec_tree_path='methods/panel_fixed_effects.md#control-sets',
                        controls_desc='Marriage duration only')
if result: results.append(result)

# ===========================================================================
# LEAVE-ONE-OUT: DROP CONTROL GROUPS
# ===========================================================================
print("\n--- Leave-One-Out Control Variations ---")

# Drop age dummies
formula = f"assets ~ {' + '.join(treatment_vars + property_vars + yrd_cols + chd_cols + since_cols)} | id"
result = run_regression(formula, df, spec_id='robust/loo/drop_age',
                        spec_tree_path='robustness/leave_one_out.md',
                        controls_desc='No age dummies')
if result: results.append(result)

# Drop year dummies
formula = f"assets ~ {' + '.join(treatment_vars + property_vars + d_age_cols + chd_cols + since_cols)} | id"
result = run_regression(formula, df, spec_id='robust/loo/drop_year',
                        spec_tree_path='robustness/leave_one_out.md',
                        controls_desc='No year dummies')
if result: results.append(result)

# Drop children dummies
formula = f"assets ~ {' + '.join(treatment_vars + property_vars + d_age_cols + yrd_cols + since_cols)} | id"
result = run_regression(formula, df, spec_id='robust/loo/drop_children',
                        spec_tree_path='robustness/leave_one_out.md',
                        controls_desc='No children dummies')
if result: results.append(result)

# Drop marriage duration
formula = f"assets ~ {' + '.join(treatment_vars + property_vars + d_age_cols + yrd_cols + chd_cols)} | id"
result = run_regression(formula, df, spec_id='robust/loo/drop_marriage',
                        spec_tree_path='robustness/leave_one_out.md',
                        controls_desc='No marriage duration')
if result: results.append(result)

# ===========================================================================
# CLUSTERING VARIATIONS
# ===========================================================================
print("\n--- Clustering Variations ---")

# Robust (heteroskedasticity-consistent) SEs
formula = f"assets ~ {' + '.join(treatment_vars + property_vars + controls_full)} | id"
result = run_regression(formula, df, cluster_var=None, spec_id='robust/cluster/robust_hc',
                        spec_tree_path='robustness/clustering_variations.md',
                        controls_desc='Full controls')
if result: results.append(result)

# Cluster by individual
formula = f"assets ~ {' + '.join(treatment_vars + property_vars + controls_full)} | id"
result = run_regression(formula, df, cluster_var='id', spec_id='robust/cluster/individual',
                        spec_tree_path='robustness/clustering_variations.md',
                        controls_desc='Full controls')
if result: results.append(result)

# Cluster by year
formula = f"assets ~ {' + '.join(treatment_vars + property_vars + controls_full)} | id"
result = run_regression(formula, df, cluster_var='year', spec_id='robust/cluster/year',
                        spec_tree_path='robustness/clustering_variations.md',
                        controls_desc='Full controls')
if result: results.append(result)

# ===========================================================================
# SAMPLE RESTRICTIONS
# ===========================================================================
print("\n--- Sample Restrictions ---")

# By time period
years = df['year'].unique()
mid_year = np.median(years)

# Early period
df_early = df[df['year'] < mid_year]
formula = f"assets ~ {' + '.join(treatment_vars + property_vars + controls_full)} | id"
result = run_regression(formula, df_early, spec_id='robust/sample/early_period',
                        spec_tree_path='robustness/sample_restrictions.md',
                        sample_desc=f'Years < {mid_year}',
                        controls_desc='Full controls')
if result: results.append(result)

# Late period
df_late = df[df['year'] >= mid_year]
result = run_regression(formula, df_late, spec_id='robust/sample/late_period',
                        spec_tree_path='robustness/sample_restrictions.md',
                        sample_desc=f'Years >= {mid_year}',
                        controls_desc='Full controls')
if result: results.append(result)

# Drop specific years (first and last)
min_year = df['year'].min()
max_year = df['year'].max()

df_no_first = df[df['year'] != min_year]
result = run_regression(formula, df_no_first, spec_id=f'robust/sample/drop_first_year',
                        spec_tree_path='robustness/sample_restrictions.md',
                        sample_desc=f'Exclude year {int(min_year)}',
                        controls_desc='Full controls')
if result: results.append(result)

df_no_last = df[df['year'] != max_year]
result = run_regression(formula, df_no_last, spec_id=f'robust/sample/drop_last_year',
                        spec_tree_path='robustness/sample_restrictions.md',
                        sample_desc=f'Exclude year {int(max_year)}',
                        controls_desc='Full controls')
if result: results.append(result)

# By property regime - community property states only
df_comprop = df[df['comprop'] == 1]
if len(df_comprop) > 100:
    formula_cp = f"assets ~ uni_comprop + {' + '.join(controls_full)} | id"
    result = run_regression(formula_cp, df_comprop, spec_id='robust/sample/comprop_only',
                            spec_tree_path='robustness/sample_restrictions.md',
                            sample_desc='Community property states only',
                            controls_desc='Full controls')
    if result: results.append(result)

# Equitable distribution states only
df_eqdistr = df[df['eqdistr'] == 1]
if len(df_eqdistr) > 100:
    formula_eq = f"assets ~ uni_eqdistr + {' + '.join(controls_full)} | id"
    result = run_regression(formula_eq, df_eqdistr, spec_id='robust/sample/eqdistr_only',
                            spec_tree_path='robustness/sample_restrictions.md',
                            sample_desc='Equitable distribution states only',
                            controls_desc='Full controls')
    if result: results.append(result)

# Title states only (neither comprop nor eqdistr)
df_title = df[(df['comprop'] == 0) & (df['eqdistr'] == 0)]
if len(df_title) > 100:
    formula_ti = f"assets ~ uni_title + {' + '.join(controls_full)} | id"
    result = run_regression(formula_ti, df_title, spec_id='robust/sample/title_only',
                            spec_tree_path='robustness/sample_restrictions.md',
                            sample_desc='Title states only',
                            controls_desc='Full controls')
    if result: results.append(result)

# ===========================================================================
# OUTLIER TREATMENTS
# ===========================================================================
print("\n--- Outlier Treatments ---")

# Winsorize at different levels
for pct in [1, 5, 10]:
    df_temp = df.copy()
    df_temp['assets'] = df_temp[f'assets_wins_{pct}']
    formula = f"assets ~ {' + '.join(treatment_vars + property_vars + controls_full)} | id"
    result = run_regression(formula, df_temp, spec_id=f'robust/sample/winsorize_{pct}pct',
                            spec_tree_path='robustness/sample_restrictions.md',
                            sample_desc=f'Assets winsorized at {pct}%',
                            controls_desc='Full controls')
    if result: results.append(result)

# Trim extreme values (1%)
df_trim = df[(df['assets'] > df['assets'].quantile(0.01)) &
             (df['assets'] < df['assets'].quantile(0.99))]
result = run_regression(formula, df_trim, spec_id='robust/sample/trim_1pct',
                        spec_tree_path='robustness/sample_restrictions.md',
                        sample_desc='Trim top and bottom 1%',
                        controls_desc='Full controls')
if result: results.append(result)

# Drop negative assets
df_pos = df[df['assets'] > 0]
result = run_regression(formula, df_pos, spec_id='robust/sample/positive_assets',
                        spec_tree_path='robustness/sample_restrictions.md',
                        sample_desc='Positive assets only',
                        controls_desc='Full controls')
if result: results.append(result)

# ===========================================================================
# FUNCTIONAL FORM VARIATIONS
# ===========================================================================
print("\n--- Functional Form Variations ---")

# Log assets
df_log = df[df['assets'] > 0].copy()
formula = f"log_assets ~ {' + '.join(treatment_vars + property_vars + controls_full)} | id"
result = run_regression(formula, df_log, spec_id='robust/funcform/log_outcome',
                        spec_tree_path='robustness/functional_form.md',
                        outcome_var='log_assets',
                        sample_desc='Positive assets only (logged)',
                        controls_desc='Full controls')
if result: results.append(result)

# Inverse hyperbolic sine
formula = f"ihs_assets ~ {' + '.join(treatment_vars + property_vars + controls_full)} | id"
result = run_regression(formula, df, spec_id='robust/funcform/ihs_outcome',
                        spec_tree_path='robustness/functional_form.md',
                        outcome_var='ihs_assets',
                        sample_desc='IHS transformation of assets',
                        controls_desc='Full controls')
if result: results.append(result)

# Age polynomial instead of dummies
formula = f"assets ~ {' + '.join(treatment_vars + property_vars)} + age + age_sq + {' + '.join(yrd_cols + chd_cols + since_cols)} | id"
result = run_regression(formula, df, spec_id='robust/funcform/age_polynomial',
                        spec_tree_path='robustness/functional_form.md',
                        controls_desc='Age polynomial instead of dummies')
if result: results.append(result)

# ===========================================================================
# ALTERNATIVE TREATMENT DEFINITIONS
# ===========================================================================
print("\n--- Alternative Treatment Definitions ---")

# Binary unilateral (any property regime)
df['uni_any'] = ((df['uni_comprop'] == 1) | (df['uni_title'] == 1) | (df['uni_eqdistr'] == 1)).astype(int)
formula = f"assets ~ uni_any + {' + '.join(property_vars + controls_full)} | id"
result = run_regression(formula, df, spec_id='robust/treatment/uni_any',
                        spec_tree_path='robustness/measurement.md',
                        treatment_var='uni_any',
                        controls_desc='Full controls')
if result: results.append(result)

# Each treatment separately
for treat in treatment_vars:
    formula = f"assets ~ {treat} + {' + '.join(property_vars + controls_full)} | id"
    result = run_regression(formula, df, spec_id=f'robust/treatment/{treat}_only',
                            spec_tree_path='robustness/measurement.md',
                            treatment_var=treat,
                            controls_desc='Full controls')
    if result: results.append(result)

# Interaction with property regime dummies
formula = f"assets ~ uni_comprop + uni_title + uni_eqdistr + {' + '.join(controls_full)} | id"
result = run_regression(formula, df, spec_id='robust/treatment/no_property_controls',
                        spec_tree_path='robustness/measurement.md',
                        controls_desc='Full controls, no property regime controls')
if result: results.append(result)

# ===========================================================================
# HETEROGENEITY ANALYSIS
# ===========================================================================
print("\n--- Heterogeneity Analysis ---")

# By age groups
df['young'] = (df['age'] < 35).astype(int)
df['middle'] = ((df['age'] >= 35) & (df['age'] < 50)).astype(int)
df['older'] = (df['age'] >= 50).astype(int)

for age_group in ['young', 'middle', 'older']:
    df_age = df[df[age_group] == 1]
    if len(df_age) > 100:
        formula = f"assets ~ {' + '.join(treatment_vars + property_vars + yrd_cols + chd_cols + since_cols)} | id"
        result = run_regression(formula, df_age, spec_id=f'robust/heterogeneity/age_{age_group}',
                                spec_tree_path='robustness/heterogeneity.md',
                                sample_desc=f'Age group: {age_group}',
                                controls_desc='Full controls (minus age dummies for subsample)')
        if result: results.append(result)

# By marriage duration
df['early_marriage'] = (df['since1marr'] <= 5).astype(int)
df['mid_marriage'] = ((df['since1marr'] > 5) & (df['since1marr'] <= 15)).astype(int)
df['late_marriage'] = (df['since1marr'] > 15).astype(int)

for marr_group in ['early_marriage', 'mid_marriage', 'late_marriage']:
    df_marr = df[df[marr_group] == 1]
    if len(df_marr) > 100:
        formula = f"assets ~ {' + '.join(treatment_vars + property_vars + d_age_cols + yrd_cols + chd_cols)} | id"
        result = run_regression(formula, df_marr, spec_id=f'robust/heterogeneity/{marr_group}',
                                spec_tree_path='robustness/heterogeneity.md',
                                sample_desc=f'Marriage duration: {marr_group}',
                                controls_desc='Full controls (minus marriage duration for subsample)')
        if result: results.append(result)

# Interaction terms
formula = f"assets ~ uni_comprop + uni_comprop:young + uni_comprop:older + {' + '.join(['uni_title', 'uni_eqdistr'] + property_vars + controls_full)} | id"
result = run_regression(formula, df, spec_id='robust/heterogeneity/age_interaction',
                        spec_tree_path='robustness/heterogeneity.md',
                        controls_desc='Full controls with age interactions')
if result: results.append(result)

# ===========================================================================
# PLACEBO TESTS
# ===========================================================================
print("\n--- Placebo Tests ---")

# Fake treatment timing (shift treatment 5 years earlier)
df_fake = df.copy()
df_fake['time_unilateral_fake'] = df_fake['time_unilateral'] - 5
df_fake['uni_comprop_fake'] = ((df_fake['time_unilateral_fake'] >= 0) & (df_fake['comprop'] == 1)).astype(int)

formula = f"assets ~ uni_comprop_fake + {' + '.join(['uni_title', 'uni_eqdistr'] + property_vars + controls_full)} | id"
result = run_regression(formula, df_fake, spec_id='robust/placebo/fake_timing_minus5',
                        spec_tree_path='robustness/placebo_tests.md',
                        treatment_var='uni_comprop_fake',
                        sample_desc='Fake treatment (5 years earlier)',
                        controls_desc='Full controls')
if result: results.append(result)

# Shift 5 years later
df_fake['time_unilateral_fake'] = df_fake['time_unilateral'] + 5
df_fake['uni_comprop_fake'] = ((df_fake['time_unilateral_fake'] >= 0) & (df_fake['comprop'] == 1)).astype(int)

formula = f"assets ~ uni_comprop_fake + {' + '.join(['uni_title', 'uni_eqdistr'] + property_vars + controls_full)} | id"
result = run_regression(formula, df_fake, spec_id='robust/placebo/fake_timing_plus5',
                        spec_tree_path='robustness/placebo_tests.md',
                        treatment_var='uni_comprop_fake',
                        sample_desc='Fake treatment (5 years later)',
                        controls_desc='Full controls')
if result: results.append(result)

# ===========================================================================
# DYNAMIC TREATMENT EFFECTS (EVENT STUDY STYLE)
# ===========================================================================
print("\n--- Dynamic Treatment Effects ---")

# Create event time dummies (as in original do file)
for x in range(1, 14, 3):
    col = f'tduni{x}'
    df[col] = 0
    mask = (df['time_unilateral'] >= x) & (df['time_unilateral'] <= x + 2)
    df.loc[mask, col] = 1

df.loc[df['time_unilateral'] > 13, 'tduni13'] = 1

# Pre-treatment indicator
df['totduni1'] = 0
df.loc[(df['totime_unilateral'] == 1) | (df['totime_unilateral'] == 2), 'totduni1'] = 1

# Event study for community property states
df_comprop_es = df[df['comprop'] == 1].copy()
time_dummies = ['totduni1', 'tduni1', 'tduni4', 'tduni7', 'tduni10', 'tduni13']

if len(df_comprop_es) > 100:
    # Reduced control set for event study
    formula = f"assets ~ {' + '.join(time_dummies + d_age_cols + yrd_cols + chd_cols + since_cols)} | id"
    result = run_regression(formula, df_comprop_es, spec_id='did/dynamic/event_study_comprop',
                            spec_tree_path='methods/difference_in_differences.md#dynamic-effects',
                            treatment_var='tduni1',  # 1-3 years after treatment
                            sample_desc='Community property states, event study',
                            controls_desc='Age + Year + Children + Marriage duration')
    if result: results.append(result)

# ===========================================================================
# ADDITIONAL ROBUSTNESS: DROP INDIVIDUAL STATES
# ===========================================================================
print("\n--- Drop Individual States ---")

# Get top 5 states by frequency
state_counts = df['state'].value_counts()
top_states = state_counts.head(5).index.tolist()

for state in top_states:
    df_no_state = df[df['state'] != state]
    formula = f"assets ~ {' + '.join(treatment_vars + property_vars + controls_full)} | id"
    result = run_regression(formula, df_no_state, spec_id=f'robust/sample/drop_state_{int(state)}',
                            spec_tree_path='robustness/sample_restrictions.md',
                            sample_desc=f'Exclude state {int(state)}',
                            controls_desc='Full controls')
    if result: results.append(result)

# ===========================================================================
# DROP INDIVIDUAL YEARS
# ===========================================================================
print("\n--- Drop Individual Years ---")

# Drop each of top 5 years
year_counts = df['year'].value_counts()
top_years = year_counts.head(5).index.tolist()

for yr in top_years:
    df_no_year = df[df['year'] != yr]
    formula = f"assets ~ {' + '.join(treatment_vars + property_vars + controls_full)} | id"
    result = run_regression(formula, df_no_year, spec_id=f'robust/sample/drop_year_{int(yr)}',
                            spec_tree_path='robustness/sample_restrictions.md',
                            sample_desc=f'Exclude year {int(yr)}',
                            controls_desc='Full controls')
    if result: results.append(result)

# ===========================================================================
# ADDITIONAL SPECIFICATIONS TO REACH 50+
# ===========================================================================
print("\n--- Additional Specifications ---")

# Balanced panel
id_counts = df.groupby('id').size()
balanced_ids = id_counts[id_counts == id_counts.max()].index
df_balanced = df[df['id'].isin(balanced_ids)]
if len(df_balanced) > 100:
    formula = f"assets ~ {' + '.join(treatment_vars + property_vars + controls_full)} | id"
    result = run_regression(formula, df_balanced, spec_id='panel/sample/balanced',
                            spec_tree_path='methods/panel_fixed_effects.md#sample-restrictions',
                            sample_desc='Balanced panel only',
                            controls_desc='Full controls')
    if result: results.append(result)

# Continuously observed
min_obs = 5
high_obs_ids = id_counts[id_counts >= min_obs].index
df_continuous = df[df['id'].isin(high_obs_ids)]
if len(df_continuous) > 100:
    formula = f"assets ~ {' + '.join(treatment_vars + property_vars + controls_full)} | id"
    result = run_regression(formula, df_continuous, spec_id='panel/sample/continuous',
                            spec_tree_path='methods/panel_fixed_effects.md#sample-restrictions',
                            sample_desc=f'Units with {min_obs}+ observations',
                            controls_desc='Full controls')
    if result: results.append(result)

# First differences
df_sorted = df.sort_values(['id', 'year'])
df_fd = df_sorted.groupby('id').apply(lambda x: x.diff()).reset_index(drop=True)
df_fd['id'] = df_sorted['id'].values
df_fd = df_fd.dropna(subset=['assets'])

if len(df_fd) > 100:
    formula = f"assets ~ {' + '.join(treatment_vars + property_vars)} | 0"
    result = run_regression(formula, df_fd, spec_id='panel/method/first_diff',
                            spec_tree_path='methods/panel_fixed_effects.md#estimation-method',
                            fixed_effects='First differences',
                            sample_desc='First differenced data',
                            controls_desc='First differences of treatment')
    if result: results.append(result)

# ===========================================================================
# ALTERNATIVE OUTCOME CODINGS
# ===========================================================================
print("\n--- Alternative Outcome Codings ---")

# Positive assets indicator
df['has_positive_assets'] = (df['assets'] > 0).astype(int)
formula = f"has_positive_assets ~ {' + '.join(treatment_vars + property_vars + controls_full)} | id"
result = run_regression(formula, df, spec_id='robust/outcome/positive_indicator',
                        spec_tree_path='robustness/measurement.md',
                        outcome_var='has_positive_assets',
                        controls_desc='Full controls')
if result: results.append(result)

# Assets quintile
df['assets_quintile'] = pd.qcut(df['assets'].rank(method='first'), 5, labels=False)
formula = f"assets_quintile ~ {' + '.join(treatment_vars + property_vars + controls_full)} | id"
result = run_regression(formula, df, spec_id='robust/outcome/quintile',
                        spec_tree_path='robustness/measurement.md',
                        outcome_var='assets_quintile',
                        controls_desc='Full controls')
if result: results.append(result)

# ===========================================================================
# MORE HETEROGENEITY - INTERACTIONS
# ===========================================================================
print("\n--- Additional Heterogeneity Interactions ---")

# Interaction with marriage duration categories
formula = f"assets ~ uni_comprop + uni_comprop:early_marriage + uni_comprop:late_marriage + {' + '.join(['uni_title', 'uni_eqdistr'] + property_vars + controls_full)} | id"
result = run_regression(formula, df, spec_id='robust/heterogeneity/marriage_interaction',
                        spec_tree_path='robustness/heterogeneity.md',
                        controls_desc='Full controls with marriage duration interactions')
if result: results.append(result)

# ===========================================================================
# MORE SAMPLE RESTRICTIONS - DECADES
# ===========================================================================
print("\n--- Sample by Decades ---")

# 1970s only
df_70s = df[(df['year'] >= 1970) & (df['year'] < 1980)]
if len(df_70s) > 100:
    formula = f"assets ~ {' + '.join(treatment_vars + property_vars + controls_full)} | id"
    result = run_regression(formula, df_70s, spec_id='robust/sample/1970s',
                            spec_tree_path='robustness/sample_restrictions.md',
                            sample_desc='1970s only',
                            controls_desc='Full controls')
    if result: results.append(result)

# 1980s only
df_80s = df[(df['year'] >= 1980) & (df['year'] < 1990)]
if len(df_80s) > 100:
    formula = f"assets ~ {' + '.join(treatment_vars + property_vars + controls_full)} | id"
    result = run_regression(formula, df_80s, spec_id='robust/sample/1980s',
                            spec_tree_path='robustness/sample_restrictions.md',
                            sample_desc='1980s only',
                            controls_desc='Full controls')
    if result: results.append(result)

# 1990s only
df_90s = df[df['year'] >= 1990]
if len(df_90s) > 100:
    formula = f"assets ~ {' + '.join(treatment_vars + property_vars + controls_full)} | id"
    result = run_regression(formula, df_90s, spec_id='robust/sample/1990s',
                            spec_tree_path='robustness/sample_restrictions.md',
                            sample_desc='1990s only',
                            controls_desc='Full controls')
    if result: results.append(result)

# ===========================================================================
# SAVE RESULTS
# ===========================================================================
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Convert to DataFrame
results_df = pd.DataFrame(results)
print(f"\nTotal specifications run: {len(results_df)}")

# Save to CSV
output_file = f"{OUTPUT_PATH}/specification_results.csv"
results_df.to_csv(output_file, index=False)
print(f"Results saved to: {output_file}")

# Summary statistics
if len(results_df) > 0:
    print("\n--- Summary Statistics ---")
    print(f"Total specifications: {len(results_df)}")
    valid_coefs = results_df['coefficient'].dropna()
    valid_pvals = results_df['p_value'].dropna()
    print(f"Positive coefficients: {(valid_coefs > 0).sum()} ({(valid_coefs > 0).mean()*100:.1f}%)")
    print(f"Significant at 5%: {(valid_pvals < 0.05).sum()} ({(valid_pvals < 0.05).mean()*100:.1f}%)")
    print(f"Significant at 1%: {(valid_pvals < 0.01).sum()} ({(valid_pvals < 0.01).mean()*100:.1f}%)")
    print(f"Median coefficient: {valid_coefs.median():.4f}")
    print(f"Mean coefficient: {valid_coefs.mean():.4f}")
    print(f"Coefficient range: [{valid_coefs.min():.4f}, {valid_coefs.max():.4f}]")
else:
    print("No results to summarize")
