"""
Specification Search Script for: Optimal Climate Policy When Damages are Unknown
Paper ID: 111185-V1
Author: Ivan Rudik
Journal: American Economic Journal: Economic Policy, 2020

This script executes the approved specification surface for the only
reduced-form regression in the paper: OLS of log damages on log temperature
(Table 1) using Howard & Sterner (2017) meta-analysis data (N=43).
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels.stats.api as sms
from scipy import stats
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
PAPER_ID = "111185-V1"
PACKAGE_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/111185-V1"
DATA_FILE = os.path.join(PACKAGE_DIR, "estimate_damage_parameters", "10640_2017_166_MOESM10_ESM.dta")

# ============================================================
# Load and prepare data
# ============================================================
print(f"Loading data from: {DATA_FILE}")
df_raw = pd.read_stata(DATA_FILE)
print(f"Raw data shape: {df_raw.shape}")

# GDP Loss -> damage function transformation (matching table_1.do)
df_raw['correct_d'] = (df_raw['D_new'] / 100) / (1 - df_raw['D_new'] / 100)
df_raw['log_correct'] = np.where(df_raw['correct_d'] > 0, np.log(df_raw['correct_d']), np.nan)
df_raw['logt'] = np.log(df_raw['t'])
df_raw['logt_sq'] = df_raw['logt'] ** 2

# Create the regression sample: drop obs with missing log_correct or logt
df = df_raw.dropna(subset=['log_correct', 'logt']).copy()
print(f"Regression sample: N={len(df)} (from {len(df_raw)} raw obs)")

# Create a study-level cluster variable from Primary_Author
df['study_cluster'] = df['Primary_Author'].astype(str)

# ============================================================
# Results storage
# ============================================================
results = []

def record_result(spec_run_id, spec_id, spec_tree_path, baseline_group_id,
                  outcome_var, treatment_var, model, focal_var='logt',
                  sample_desc='', fixed_effects='', controls_desc='',
                  cluster_var='', extra_coef_info=None, n_obs_override=None,
                  se_type='classical', weights_desc=''):
    """Extract and record results from a statsmodels OLS result."""
    try:
        coef = model.params[focal_var]
        se = model.bse[focal_var]
        pval = model.pvalues[focal_var]
        ci = model.conf_int().loc[focal_var]
        n = int(model.nobs) if n_obs_override is None else n_obs_override
        r2 = model.rsquared

        # Build coefficient vector JSON
        coef_vector = {k: float(v) for k, v in model.params.items()}
        if extra_coef_info:
            coef_vector.update(extra_coef_info)

        results.append({
            'paper_id': PAPER_ID,
            'spec_run_id': f"{PAPER_ID}_{spec_run_id}",
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'baseline_group_id': baseline_group_id,
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': coef,
            'std_error': se,
            'p_value': pval,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'n_obs': n,
            'r_squared': r2,
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects,
            'controls_desc': controls_desc,
            'cluster_var': cluster_var,
            'se_type': se_type,
            'weights_desc': weights_desc,
        })
        print(f"  [{spec_run_id}] coef={coef:.4f}, se={se:.4f}, p={pval:.4e}, n={n}, R2={r2:.4f}")
    except Exception as e:
        print(f"  [{spec_run_id}] FAILED: {e}")
        results.append({
            'paper_id': PAPER_ID,
            'spec_run_id': f"{PAPER_ID}_{spec_run_id}",
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
            'coefficient_vector_json': json.dumps({"error": str(e)}),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects,
            'controls_desc': controls_desc,
            'cluster_var': cluster_var,
            'se_type': se_type,
            'weights_desc': weights_desc,
        })

# ============================================================
# STEP 1: BASELINE
# ============================================================
print("\n" + "="*60)
print("STEP 1: BASELINE SPEC")
print("="*60)

baseline_model = smf.ols("log_correct ~ logt", data=df).fit()
record_result(
    spec_run_id='G1_baseline',
    spec_id='baseline',
    spec_tree_path='baseline',
    baseline_group_id='G1',
    outcome_var='log_correct',
    treatment_var='logt',
    model=baseline_model,
    sample_desc='Howard & Sterner (2017) damage estimates, N=43 (6 obs dropped due to non-positive damages)',
    controls_desc='None (bivariate regression)',
    se_type='classical',
)

# ============================================================
# STEP 2: DESIGN SPECS
# ============================================================
print("\n" + "="*60)
print("STEP 2: DESIGN SPECS")
print("="*60)

# design/cross_sectional_ols/estimator/ols -- identical to baseline
design_model = smf.ols("log_correct ~ logt", data=df).fit()
record_result(
    spec_run_id='G1_design_ols',
    spec_id='design/cross_sectional_ols/estimator/ols',
    spec_tree_path='design/cross_sectional_ols/estimator/ols',
    baseline_group_id='G1',
    outcome_var='log_correct',
    treatment_var='logt',
    model=design_model,
    sample_desc='Howard & Sterner (2017), N=43',
    controls_desc='None (bivariate regression)',
    se_type='classical',
)

# ============================================================
# STEP 3: RC/CONTROLS - SINGLE ADDITIONS
# ============================================================
print("\n" + "="*60)
print("STEP 3: RC/CONTROLS - SINGLE ADDITIONS")
print("="*60)

single_controls = [
    ('Year', 'rc/controls/single/add_Year'),
    ('Market', 'rc/controls/single/add_Market'),
    ('Nonmarket', 'rc/controls/single/add_Nonmarket'),
    ('Grey', 'rc/controls/single/add_Grey'),
    ('Preindustrial', 'rc/controls/single/add_Preindustrial'),
    ('Based_On_Other', 'rc/controls/single/add_Based_On_Other'),
    ('Method_1', 'rc/controls/single/add_Method_1'),
    ('Method_2', 'rc/controls/single/add_Method_2'),
    ('Method_3', 'rc/controls/single/add_Method_3'),
    ('Method_5', 'rc/controls/single/add_Method_5'),
]

for ctrl_var, spec_id in single_controls:
    run_id = f"G1_{spec_id.replace('/', '_')}"
    formula = f"log_correct ~ logt + {ctrl_var}"
    m = smf.ols(formula, data=df).fit()
    record_result(
        spec_run_id=run_id,
        spec_id=spec_id,
        spec_tree_path=spec_id,
        baseline_group_id='G1',
        outcome_var='log_correct',
        treatment_var='logt',
        model=m,
        sample_desc='Howard & Sterner (2017), N=43',
        controls_desc=ctrl_var,
        se_type='classical',
    )

# ============================================================
# STEP 4: RC/CONTROLS - NAMED SETS AND PROGRESSIONS
# ============================================================
print("\n" + "="*60)
print("STEP 4: RC/CONTROLS - NAMED SETS AND PROGRESSIONS")
print("="*60)

# study_characteristics_basic: Market + Grey + Preindustrial (study_type block)
ctrl_set = ['Market', 'Grey', 'Preindustrial']
formula = "log_correct ~ logt + " + " + ".join(ctrl_set)
m = smf.ols(formula, data=df).fit()
record_result(
    spec_run_id='G1_rc_controls_sets_study_characteristics_basic',
    spec_id='rc/controls/sets/study_characteristics_basic',
    spec_tree_path='rc/controls/sets/study_characteristics_basic',
    baseline_group_id='G1',
    outcome_var='log_correct',
    treatment_var='logt',
    model=m,
    sample_desc='Howard & Sterner (2017), N=43',
    controls_desc=', '.join(ctrl_set),
    se_type='classical',
)

# study_characteristics_extended: Market + Grey + Preindustrial + Based_On_Other + Nonmarket
ctrl_set = ['Market', 'Grey', 'Preindustrial', 'Based_On_Other', 'Nonmarket']
formula = "log_correct ~ logt + " + " + ".join(ctrl_set)
m = smf.ols(formula, data=df).fit()
record_result(
    spec_run_id='G1_rc_controls_sets_study_characteristics_extended',
    spec_id='rc/controls/sets/study_characteristics_extended',
    spec_tree_path='rc/controls/sets/study_characteristics_extended',
    baseline_group_id='G1',
    outcome_var='log_correct',
    treatment_var='logt',
    model=m,
    sample_desc='Howard & Sterner (2017), N=43',
    controls_desc=', '.join(ctrl_set),
    se_type='classical',
)

# Progression: study_type = Market + Grey + Preindustrial
ctrl_set = ['Market', 'Grey', 'Preindustrial']
formula = "log_correct ~ logt + " + " + ".join(ctrl_set)
m = smf.ols(formula, data=df).fit()
record_result(
    spec_run_id='G1_rc_controls_progression_study_type',
    spec_id='rc/controls/progression/study_type',
    spec_tree_path='rc/controls/progression/study_type',
    baseline_group_id='G1',
    outcome_var='log_correct',
    treatment_var='logt',
    model=m,
    sample_desc='Howard & Sterner (2017), N=43',
    controls_desc=', '.join(ctrl_set),
    se_type='classical',
)

# Progression: study_type_plus_method = Market + Grey + Preindustrial + Method_1 + Method_2 + Method_3 + Method_5
ctrl_set = ['Market', 'Grey', 'Preindustrial', 'Method_1', 'Method_2', 'Method_3', 'Method_5']
formula = "log_correct ~ logt + " + " + ".join(ctrl_set)
m = smf.ols(formula, data=df).fit()
record_result(
    spec_run_id='G1_rc_controls_progression_study_type_plus_method',
    spec_id='rc/controls/progression/study_type_plus_method',
    spec_tree_path='rc/controls/progression/study_type_plus_method',
    baseline_group_id='G1',
    outcome_var='log_correct',
    treatment_var='logt',
    model=m,
    sample_desc='Howard & Sterner (2017), N=43',
    controls_desc=', '.join(ctrl_set),
    se_type='classical',
)

# Progression: full = all 10 controls (Market + Nonmarket + Grey + Preindustrial + Based_On_Other + Year + Method_1 + Method_2 + Method_3 + Method_5)
# NOTE: This exceeds 4-control cap but is an explicitly enumerated named spec
ctrl_set_full = ['Market', 'Nonmarket', 'Grey', 'Preindustrial', 'Based_On_Other', 'Year', 'Method_1', 'Method_2', 'Method_3', 'Method_5']
formula = "log_correct ~ logt + " + " + ".join(ctrl_set_full)
# Check for collinearity issues
try:
    m = smf.ols(formula, data=df).fit()
    # Check if any coefficients are NaN (perfect collinearity)
    if m.params.isna().any():
        print("  WARNING: Perfect collinearity detected in full progression. Dropping collinear vars.")
        # Drop vars with NaN coefficients
        good_vars = [v for v in ctrl_set_full if v in m.params.index and not np.isnan(m.params[v])]
        formula = "log_correct ~ logt + " + " + ".join(good_vars)
        m = smf.ols(formula, data=df).fit()
        ctrl_desc = ', '.join(good_vars) + ' (collinear vars dropped)'
    else:
        ctrl_desc = ', '.join(ctrl_set_full)
    record_result(
        spec_run_id='G1_rc_controls_progression_full',
        spec_id='rc/controls/progression/full',
        spec_tree_path='rc/controls/progression/full',
        baseline_group_id='G1',
        outcome_var='log_correct',
        treatment_var='logt',
        model=m,
        sample_desc='Howard & Sterner (2017), N=43',
        controls_desc=ctrl_desc,
        se_type='classical',
    )
except Exception as e:
    print(f"  Full progression FAILED: {e}")

# ============================================================
# STEP 5: RC/SAMPLE VARIANTS
# ============================================================
print("\n" + "="*60)
print("STEP 5: RC/SAMPLE VARIANTS")
print("="*60)

# --- Outlier trimming ---

# trim_y_1_99: Trim log_correct outside [1%, 99%] percentiles
p1, p99 = df['log_correct'].quantile(0.01), df['log_correct'].quantile(0.99)
df_trim199 = df[(df['log_correct'] >= p1) & (df['log_correct'] <= p99)]
m = smf.ols("log_correct ~ logt", data=df_trim199).fit()
record_result(
    spec_run_id='G1_rc_sample_outliers_trim_y_1_99',
    spec_id='rc/sample/outliers/trim_y_1_99',
    spec_tree_path='rc/sample/outliers/trim_y_1_99',
    baseline_group_id='G1',
    outcome_var='log_correct',
    treatment_var='logt',
    model=m,
    sample_desc=f'Trimmed log_correct at [1%, 99%] percentiles, N={len(df_trim199)}',
    controls_desc='None (bivariate regression)',
    se_type='classical',
)

# trim_y_5_95: Trim log_correct outside [5%, 95%] percentiles
p5, p95 = df['log_correct'].quantile(0.05), df['log_correct'].quantile(0.95)
df_trim595 = df[(df['log_correct'] >= p5) & (df['log_correct'] <= p95)]
m = smf.ols("log_correct ~ logt", data=df_trim595).fit()
record_result(
    spec_run_id='G1_rc_sample_outliers_trim_y_5_95',
    spec_id='rc/sample/outliers/trim_y_5_95',
    spec_tree_path='rc/sample/outliers/trim_y_5_95',
    baseline_group_id='G1',
    outcome_var='log_correct',
    treatment_var='logt',
    model=m,
    sample_desc=f'Trimmed log_correct at [5%, 95%] percentiles, N={len(df_trim595)}',
    controls_desc='None (bivariate regression)',
    se_type='classical',
)

# drop_weitzman_12C: Drop the Weitzman 12C/99% damage estimate
df_no_weitzman = df[df['t'] != 12.0].copy()
m = smf.ols("log_correct ~ logt", data=df_no_weitzman).fit()
record_result(
    spec_run_id='G1_rc_sample_outliers_drop_weitzman_12C',
    spec_id='rc/sample/outliers/drop_weitzman_12C',
    spec_tree_path='rc/sample/outliers/drop_weitzman_12C',
    baseline_group_id='G1',
    outcome_var='log_correct',
    treatment_var='logt',
    model=m,
    sample_desc=f'Dropped Weitzman 12C observation, N={len(df_no_weitzman)}',
    controls_desc='None (bivariate regression)',
    se_type='classical',
)

# cooksd_4_over_n: Drop observations with Cook's D > 4/N
influence = baseline_model.get_influence()
cooks_d = influence.cooks_distance[0]
threshold = 4.0 / len(df)
mask_cooks = cooks_d <= threshold
df_cooks = df[mask_cooks].copy()
m = smf.ols("log_correct ~ logt", data=df_cooks).fit()
record_result(
    spec_run_id='G1_rc_sample_outliers_cooksd_4_over_n',
    spec_id='rc/sample/outliers/cooksd_4_over_n',
    spec_tree_path='rc/sample/outliers/cooksd_4_over_n',
    baseline_group_id='G1',
    outcome_var='log_correct',
    treatment_var='logt',
    model=m,
    sample_desc=f'Dropped obs with Cook\'s D > 4/N (threshold={threshold:.4f}), N={len(df_cooks)} (dropped {len(df)-len(df_cooks)})',
    controls_desc='None (bivariate regression)',
    se_type='classical',
)

# --- Time splits ---

# pre_2006: Studies published before 2006
df_pre2006 = df[df['Year'] < 2006].copy()
m = smf.ols("log_correct ~ logt", data=df_pre2006).fit()
record_result(
    spec_run_id='G1_rc_sample_time_pre_2006',
    spec_id='rc/sample/time/pre_2006',
    spec_tree_path='rc/sample/time/pre_2006',
    baseline_group_id='G1',
    outcome_var='log_correct',
    treatment_var='logt',
    model=m,
    sample_desc=f'Studies published before 2006, N={len(df_pre2006)}',
    controls_desc='None (bivariate regression)',
    se_type='classical',
)

# post_2006: Studies published 2006 and after
df_post2006 = df[df['Year'] >= 2006].copy()
m = smf.ols("log_correct ~ logt", data=df_post2006).fit()
record_result(
    spec_run_id='G1_rc_sample_time_post_2006',
    spec_id='rc/sample/time/post_2006',
    spec_tree_path='rc/sample/time/post_2006',
    baseline_group_id='G1',
    outcome_var='log_correct',
    treatment_var='logt',
    model=m,
    sample_desc=f'Studies published 2006 and after, N={len(df_post2006)}',
    controls_desc='None (bivariate regression)',
    se_type='classical',
)

# --- Quality filters ---

# drop_based_on_other: Drop studies with Based_On_Other=1
df_no_based = df[df['Based_On_Other'] == 0].copy()
m = smf.ols("log_correct ~ logt", data=df_no_based).fit()
record_result(
    spec_run_id='G1_rc_sample_quality_drop_based_on_other',
    spec_id='rc/sample/quality/drop_based_on_other',
    spec_tree_path='rc/sample/quality/drop_based_on_other',
    baseline_group_id='G1',
    outcome_var='log_correct',
    treatment_var='logt',
    model=m,
    sample_desc=f'Dropped Based_On_Other=1 studies, N={len(df_no_based)}',
    controls_desc='None (bivariate regression)',
    se_type='classical',
)

# drop_grey_literature: Drop Grey=1 studies
df_no_grey = df[df['Grey'] == 0].copy()
m = smf.ols("log_correct ~ logt", data=df_no_grey).fit()
record_result(
    spec_run_id='G1_rc_sample_quality_drop_grey_literature',
    spec_id='rc/sample/quality/drop_grey_literature',
    spec_tree_path='rc/sample/quality/drop_grey_literature',
    baseline_group_id='G1',
    outcome_var='log_correct',
    treatment_var='logt',
    model=m,
    sample_desc=f'Dropped grey literature (Grey=1), N={len(df_no_grey)}',
    controls_desc='None (bivariate regression)',
    se_type='classical',
)

# ============================================================
# STEP 6: RC/FORM - FUNCTIONAL FORM VARIANTS
# ============================================================
print("\n" + "="*60)
print("STEP 6: RC/FORM - FUNCTIONAL FORM VARIANTS")
print("="*60)

# quadratic_treatment: log_correct on logt + logt^2
m = smf.ols("log_correct ~ logt + logt_sq", data=df).fit()
# Focal coefficient is the linear term on logt
# Also compute joint F-test for logt + logt^2
from statsmodels.stats.anova import anova_lm
m_restricted = smf.ols("log_correct ~ 1", data=df).fit()
f_test = m.f_test("logt = 0, logt_sq = 0")
fval = f_test.fvalue
if hasattr(fval, '__getitem__'):
    try:
        fval = float(fval[0][0])
    except (TypeError, IndexError):
        fval = float(fval)
else:
    fval = float(fval)
pval_f = float(f_test.pvalue)
extra_info = {
    'joint_F_stat': fval,
    'joint_F_pval': pval_f,
    'coef_logt_sq': float(m.params['logt_sq']),
    'se_logt_sq': float(m.bse['logt_sq']),
    'pval_logt_sq': float(m.pvalues['logt_sq']),
}
record_result(
    spec_run_id='G1_rc_form_model_quadratic_treatment',
    spec_id='rc/form/model/quadratic_treatment',
    spec_tree_path='rc/form/model/quadratic_treatment',
    baseline_group_id='G1',
    outcome_var='log_correct',
    treatment_var='logt + logt^2',
    model=m,
    focal_var='logt',
    sample_desc='Howard & Sterner (2017), N=43',
    controls_desc='logt_sq (quadratic in log temperature)',
    se_type='classical',
    extra_coef_info=extra_info,
)

# ============================================================
# STEP 7: RC/PREPROCESS VARIANTS
# ============================================================
print("\n" + "="*60)
print("STEP 7: RC/PREPROCESS VARIANTS")
print("="*60)

# winsor_1_99: Winsorize log_correct at 1st/99th percentiles
p1, p99 = df['log_correct'].quantile(0.01), df['log_correct'].quantile(0.99)
df_winsor = df.copy()
df_winsor['log_correct_w'] = df_winsor['log_correct'].clip(lower=p1, upper=p99)
m = smf.ols("log_correct_w ~ logt", data=df_winsor).fit()
record_result(
    spec_run_id='G1_rc_preprocess_outcome_winsor_1_99',
    spec_id='rc/preprocess/outcome/winsor_1_99',
    spec_tree_path='rc/preprocess/outcome/winsor_1_99',
    baseline_group_id='G1',
    outcome_var='log_correct (winsorized 1%/99%)',
    treatment_var='logt',
    model=m,
    sample_desc='Howard & Sterner (2017), N=43, log_correct winsorized at 1st/99th pctl',
    controls_desc='None (bivariate regression)',
    se_type='classical',
)

# Temperature adjustments: FUND, NASA, AVG
temp_adj_specs = [
    ('Temp_adj_FUND_curr', 'rc/preprocess/treatment/temperature_adjustment_FUND', 'FUND'),
    ('Temp_adj_NASA', 'rc/preprocess/treatment/temperature_adjustment_NASA', 'NASA'),
    ('Temp_adj_AVG', 'rc/preprocess/treatment/temperature_adjustment_AVG', 'AVG'),
]

for adj_col, spec_id, adj_label in temp_adj_specs:
    run_id = f"G1_{spec_id.replace('/', '_')}"
    df_adj = df.copy()
    df_adj['t_adjusted'] = df_adj['t'] - df_adj[adj_col]
    # Need positive adjusted temperature for log
    df_adj = df_adj[df_adj['t_adjusted'] > 0].copy()
    df_adj['logt_adj'] = np.log(df_adj['t_adjusted'])
    df_adj = df_adj.dropna(subset=['logt_adj', 'log_correct'])
    if len(df_adj) < 5:
        print(f"  [{run_id}] SKIPPED: Only {len(df_adj)} obs with positive adjusted temperature")
        continue
    m = smf.ols("log_correct ~ logt_adj", data=df_adj).fit()
    record_result(
        spec_run_id=run_id,
        spec_id=spec_id,
        spec_tree_path=spec_id,
        baseline_group_id='G1',
        outcome_var='log_correct',
        treatment_var=f'log(t - {adj_col})',
        model=m,
        focal_var='logt_adj',
        sample_desc=f'Howard & Sterner (2017), {adj_label}-adjusted temperature, N={len(df_adj)}',
        controls_desc='None (bivariate regression)',
        se_type='classical',
    )

# ============================================================
# STEP 8: RC/WEIGHTS VARIANTS
# ============================================================
print("\n" + "="*60)
print("STEP 8: RC/WEIGHTS VARIANTS")
print("="*60)

# unweighted: identical to baseline, included as reference
m = smf.ols("log_correct ~ logt", data=df).fit()
record_result(
    spec_run_id='G1_rc_weights_main_unweighted',
    spec_id='rc/weights/main/unweighted',
    spec_tree_path='rc/weights/main/unweighted',
    baseline_group_id='G1',
    outcome_var='log_correct',
    treatment_var='logt',
    model=m,
    sample_desc='Howard & Sterner (2017), N=43',
    controls_desc='None (bivariate regression)',
    se_type='classical',
    weights_desc='unweighted (identical to baseline)',
)

# WLS: inverse variance proxy
# Check if dataset contains any precision measure.
# Possible candidates: number of observations per study group, or Year (as a proxy for quality)
# The dataset doesn't have standard errors per observation, but we can use 1/t^2 as a proxy
# (variance of log damage estimates plausibly increases with temperature magnitude)
# OR use Year as proxy (more recent studies may be more precise).
# Since no direct variance measure is available, we'll use Year (centered) as a precision proxy.
# Actually, a better approach: count obs per Primary_Author (study group) as a weight.
# Let's check if any direct precision measure exists.
# The surface says "if available" and "feasibility depends on H&S dataset containing precision measures"
# There is no explicit SE column. Let's try using 1/(t^2) as a rough inverse-variance proxy
# (higher temperature studies tend to have more uncertain damage estimates).
# This is a common meta-regression weighting strategy.
df_wls = df.copy()
df_wls['weight_inv_t2'] = 1.0 / (df_wls['t'] ** 2)
try:
    m = smf.wls("log_correct ~ logt", data=df_wls, weights=df_wls['weight_inv_t2']).fit()
    record_result(
        spec_run_id='G1_rc_weights_main_wls_inverse_variance_proxy',
        spec_id='rc/weights/main/wls_inverse_variance_proxy',
        spec_tree_path='rc/weights/main/wls_inverse_variance_proxy',
        baseline_group_id='G1',
        outcome_var='log_correct',
        treatment_var='logt',
        model=m,
        sample_desc='Howard & Sterner (2017), N=43, WLS with 1/t^2 weights',
        controls_desc='None (bivariate regression)',
        se_type='classical (WLS)',
        weights_desc='WLS: inverse temperature squared (1/t^2) as precision proxy. No direct study-level SE available in dataset.',
    )
except Exception as e:
    print(f"  WLS FAILED: {e}")

# ============================================================
# STEP 9: INFER/* VARIANTS (inference-only changes)
# ============================================================
print("\n" + "="*60)
print("STEP 9: INFERENCE VARIANTS")
print("="*60)

# HC1 robust standard errors
m = smf.ols("log_correct ~ logt", data=df).fit(cov_type='HC1')
record_result(
    spec_run_id='G1_infer_se_hc_hc1',
    spec_id='infer/se/hc/hc1',
    spec_tree_path='infer/se/hc/hc1',
    baseline_group_id='G1',
    outcome_var='log_correct',
    treatment_var='logt',
    model=m,
    sample_desc='Howard & Sterner (2017), N=43',
    controls_desc='None (bivariate regression)',
    se_type='HC1',
)

# HC2 robust standard errors
m = smf.ols("log_correct ~ logt", data=df).fit(cov_type='HC2')
record_result(
    spec_run_id='G1_infer_se_hc_hc2',
    spec_id='infer/se/hc/hc2',
    spec_tree_path='infer/se/hc/hc2',
    baseline_group_id='G1',
    outcome_var='log_correct',
    treatment_var='logt',
    model=m,
    sample_desc='Howard & Sterner (2017), N=43',
    controls_desc='None (bivariate regression)',
    se_type='HC2',
)

# HC3 robust standard errors
m = smf.ols("log_correct ~ logt", data=df).fit(cov_type='HC3')
record_result(
    spec_run_id='G1_infer_se_hc_hc3',
    spec_id='infer/se/hc/hc3',
    spec_tree_path='infer/se/hc/hc3',
    baseline_group_id='G1',
    outcome_var='log_correct',
    treatment_var='logt',
    model=m,
    sample_desc='Howard & Sterner (2017), N=43',
    controls_desc='None (bivariate regression)',
    se_type='HC3',
)

# Cluster SEs by study (Primary_Author)
try:
    m = smf.ols("log_correct ~ logt", data=df).fit(
        cov_type='cluster',
        cov_kwds={'groups': df['study_cluster']}
    )
    record_result(
        spec_run_id='G1_infer_se_cluster_study',
        spec_id='infer/se/cluster/study',
        spec_tree_path='infer/se/cluster/study',
        baseline_group_id='G1',
        outcome_var='log_correct',
        treatment_var='logt',
        model=m,
        sample_desc='Howard & Sterner (2017), N=43',
        controls_desc='None (bivariate regression)',
        cluster_var='Primary_Author',
        se_type='cluster (Primary_Author)',
    )
except Exception as e:
    print(f"  Cluster SE FAILED: {e}")

# ============================================================
# STEP 10: HIGH-VALUE INTERACTIONS
# ============================================================
print("\n" + "="*60)
print("STEP 10: HIGH-VALUE INTERACTIONS")
print("="*60)

# Joint 1: Drop Weitzman + HC3
df_no_weitzman = df[df['t'] != 12.0].copy()
m = smf.ols("log_correct ~ logt", data=df_no_weitzman).fit(cov_type='HC3')
record_result(
    spec_run_id='G1_rc_joint_outlier_inference_drop_weitzman_hc3',
    spec_id='rc/joint/outlier_inference/drop_weitzman_hc3',
    spec_tree_path='rc/joint/outlier_inference/drop_weitzman_hc3',
    baseline_group_id='G1',
    outcome_var='log_correct',
    treatment_var='logt',
    model=m,
    sample_desc=f'Dropped Weitzman 12C observation, N={len(df_no_weitzman)}',
    controls_desc='None (bivariate regression)',
    se_type='HC3',
)

# Joint 2: Drop Weitzman + study_type controls
ctrl_set = ['Market', 'Grey', 'Preindustrial']
formula = "log_correct ~ logt + " + " + ".join(ctrl_set)
m = smf.ols(formula, data=df_no_weitzman).fit()
record_result(
    spec_run_id='G1_rc_joint_outlier_controls_drop_weitzman_study_type',
    spec_id='rc/joint/outlier_controls/drop_weitzman_study_type',
    spec_tree_path='rc/joint/outlier_controls/drop_weitzman_study_type',
    baseline_group_id='G1',
    outcome_var='log_correct',
    treatment_var='logt',
    model=m,
    sample_desc=f'Dropped Weitzman 12C, N={len(df_no_weitzman)}',
    controls_desc=', '.join(ctrl_set),
    se_type='classical',
)

# Joint 3: HC3 + study_type controls
ctrl_set = ['Market', 'Grey', 'Preindustrial']
formula = "log_correct ~ logt + " + " + ".join(ctrl_set)
m = smf.ols(formula, data=df).fit(cov_type='HC3')
record_result(
    spec_run_id='G1_rc_joint_inference_controls_hc3_study_type',
    spec_id='rc/joint/inference_controls/hc3_study_type',
    spec_tree_path='rc/joint/inference_controls/hc3_study_type',
    baseline_group_id='G1',
    outcome_var='log_correct',
    treatment_var='logt',
    model=m,
    sample_desc='Howard & Sterner (2017), N=43',
    controls_desc=', '.join(ctrl_set),
    se_type='HC3',
)

# Joint 4: Drop Weitzman + quadratic treatment
df_no_weitzman['logt_sq'] = df_no_weitzman['logt'] ** 2
m = smf.ols("log_correct ~ logt + logt_sq", data=df_no_weitzman).fit()
record_result(
    spec_run_id='G1_rc_joint_outlier_form_drop_weitzman_quadratic',
    spec_id='rc/joint/outlier_form/drop_weitzman_quadratic',
    spec_tree_path='rc/joint/outlier_form/drop_weitzman_quadratic',
    baseline_group_id='G1',
    outcome_var='log_correct',
    treatment_var='logt + logt^2',
    model=m,
    focal_var='logt',
    sample_desc=f'Dropped Weitzman 12C, N={len(df_no_weitzman)}',
    controls_desc='logt_sq (quadratic in log temperature)',
    se_type='classical',
)

# Joint 5: FUND temp adjustment + drop Weitzman
df_adj = df.copy()
df_adj['t_adjusted'] = df_adj['t'] - df_adj['Temp_adj_FUND_curr']
df_adj = df_adj[(df_adj['t_adjusted'] > 0) & (df_adj['t'] != 12.0)].copy()
df_adj['logt_adj'] = np.log(df_adj['t_adjusted'])
df_adj = df_adj.dropna(subset=['logt_adj', 'log_correct'])
m = smf.ols("log_correct ~ logt_adj", data=df_adj).fit()
record_result(
    spec_run_id='G1_rc_joint_preprocess_sample_temp_adj_fund_drop_weitzman',
    spec_id='rc/joint/preprocess_sample/temp_adj_fund_drop_weitzman',
    spec_tree_path='rc/joint/preprocess_sample/temp_adj_fund_drop_weitzman',
    baseline_group_id='G1',
    outcome_var='log_correct',
    treatment_var='log(t - Temp_adj_FUND_curr)',
    model=m,
    focal_var='logt_adj',
    sample_desc=f'FUND-adjusted temp + dropped Weitzman, N={len(df_adj)}',
    controls_desc='None (bivariate regression)',
    se_type='classical',
)

# Joint 6: WLS + HC3 (if WLS feasible)
try:
    df_wls = df.copy()
    df_wls['weight_inv_t2'] = 1.0 / (df_wls['t'] ** 2)
    m = smf.wls("log_correct ~ logt", data=df_wls, weights=df_wls['weight_inv_t2']).fit(cov_type='HC3')
    record_result(
        spec_run_id='G1_rc_joint_weights_inference_wls_hc3',
        spec_id='rc/joint/weights_inference/wls_hc3',
        spec_tree_path='rc/joint/weights_inference/wls_hc3',
        baseline_group_id='G1',
        outcome_var='log_correct',
        treatment_var='logt',
        model=m,
        sample_desc='Howard & Sterner (2017), N=43, WLS with 1/t^2 weights',
        controls_desc='None (bivariate regression)',
        se_type='HC3 (WLS)',
        weights_desc='WLS: 1/t^2 weights + HC3 robust SEs',
    )
except Exception as e:
    print(f"  WLS+HC3 FAILED: {e}")

# ============================================================
# STEP 11: DIAGNOSTICS
# ============================================================
print("\n" + "="*60)
print("STEP 11: DIAGNOSTICS")
print("="*60)

diag_results = []

# Cook's D
influence = baseline_model.get_influence()
cooks_d_vals = influence.cooks_distance[0]
diag_results.append({
    'paper_id': PAPER_ID,
    'diag_run_id': f'{PAPER_ID}_diag_cooks_d',
    'diag_spec_id': 'diag/regression/influence/cooks_d',
    'baseline_group_id': 'G1',
    'scope': 'baseline_group',
    'result_json': json.dumps({
        'max_cooks_d': float(np.max(cooks_d_vals)),
        'mean_cooks_d': float(np.mean(cooks_d_vals)),
        'n_above_4_over_n': int(np.sum(cooks_d_vals > 4.0/len(df))),
        'threshold_4_over_n': float(4.0/len(df)),
        'top_5_indices': [int(i) for i in np.argsort(cooks_d_vals)[-5:][::-1]],
        'top_5_values': [float(cooks_d_vals[i]) for i in np.argsort(cooks_d_vals)[-5:][::-1]],
    }),
    'notes': f'Cook\'s D: max={np.max(cooks_d_vals):.4f}, {int(np.sum(cooks_d_vals > 4.0/len(df)))} obs above 4/N threshold'
})
print(f"  Cook's D: max={np.max(cooks_d_vals):.4f}, {int(np.sum(cooks_d_vals > 4.0/len(df)))} obs > 4/N")

# Jarque-Bera normality test
jb_result = stats.jarque_bera(baseline_model.resid)
jb_stat = float(jb_result.statistic) if hasattr(jb_result, 'statistic') else float(jb_result[0])
jb_pval = float(jb_result.pvalue) if hasattr(jb_result, 'pvalue') else float(jb_result[1])
resid_arr = baseline_model.resid.values
jb_skew = float(stats.skew(resid_arr))
jb_kurt = float(stats.kurtosis(resid_arr, fisher=False))
diag_results.append({
    'paper_id': PAPER_ID,
    'diag_run_id': f'{PAPER_ID}_diag_jarque_bera',
    'diag_spec_id': 'diag/regression/normality/jarque_bera',
    'baseline_group_id': 'G1',
    'scope': 'baseline_group',
    'result_json': json.dumps({
        'jb_statistic': jb_stat,
        'jb_p_value': jb_pval,
        'skewness': jb_skew,
        'kurtosis': jb_kurt,
    }),
    'notes': f'JB stat={jb_stat:.3f}, p={jb_pval:.4e}, skew={jb_skew:.3f}, kurt={jb_kurt:.3f}'
})
print(f"  Jarque-Bera: stat={jb_stat:.3f}, p={jb_pval:.4e}")

# Breusch-Pagan heteroskedasticity test
bp_result = sms.het_breuschpagan(baseline_model.resid, baseline_model.model.exog)
diag_results.append({
    'paper_id': PAPER_ID,
    'diag_run_id': f'{PAPER_ID}_diag_breusch_pagan',
    'diag_spec_id': 'diag/regression/heteroskedasticity/breusch_pagan',
    'baseline_group_id': 'G1',
    'scope': 'baseline_group',
    'result_json': json.dumps({
        'lm_statistic': float(bp_result[0]),
        'lm_p_value': float(bp_result[1]),
        'f_statistic': float(bp_result[2]),
        'f_p_value': float(bp_result[3]),
    }),
    'notes': f'BP LM stat={bp_result[0]:.3f}, p={bp_result[1]:.4e}'
})
print(f"  Breusch-Pagan: LM stat={bp_result[0]:.3f}, p={bp_result[1]:.4e}")

# Ramsey RESET test
try:
    from statsmodels.stats.diagnostic import linear_reset
    reset_result = linear_reset(baseline_model, power=3, use_f=True)
    diag_results.append({
        'paper_id': PAPER_ID,
        'diag_run_id': f'{PAPER_ID}_diag_ramsey_reset',
        'diag_spec_id': 'diag/regression/specification/ramsey_reset',
        'baseline_group_id': 'G1',
        'scope': 'baseline_group',
        'result_json': json.dumps({
            'f_statistic': float(reset_result.statistic),
            'f_p_value': float(reset_result.pvalue),
            'df_num': int(reset_result.df_num) if hasattr(reset_result, 'df_num') else None,
            'df_denom': int(reset_result.df_denom) if hasattr(reset_result, 'df_denom') else None,
        }),
        'notes': f'RESET F stat={reset_result.statistic:.3f}, p={reset_result.pvalue:.4e}'
    })
    print(f"  Ramsey RESET: F stat={reset_result.statistic:.3f}, p={reset_result.pvalue:.4e}")
except Exception as e:
    print(f"  Ramsey RESET FAILED: {e}")
    # Fallback: manual RESET test
    try:
        yhat = baseline_model.fittedvalues
        df_reset = df.copy()
        df_reset['yhat2'] = yhat ** 2
        df_reset['yhat3'] = yhat ** 3
        m_aug = smf.ols("log_correct ~ logt + yhat2 + yhat3", data=df_reset).fit()
        f_test = m_aug.f_test("yhat2 = 0, yhat3 = 0")
        diag_results.append({
            'paper_id': PAPER_ID,
            'diag_run_id': f'{PAPER_ID}_diag_ramsey_reset',
            'diag_spec_id': 'diag/regression/specification/ramsey_reset',
            'baseline_group_id': 'G1',
            'scope': 'baseline_group',
            'result_json': json.dumps({
                'f_statistic': float(f_test.fvalue[0][0]),
                'f_p_value': float(f_test.pvalue),
            }),
            'notes': f'Manual RESET: F={float(f_test.fvalue[0][0]):.3f}, p={float(f_test.pvalue):.4e}'
        })
        print(f"  Manual RESET: F={float(f_test.fvalue[0][0]):.3f}, p={float(f_test.pvalue):.4e}")
    except Exception as e2:
        print(f"  Manual RESET also FAILED: {e2}")

# ============================================================
# STEP 12: WRITE OUTPUTS
# ============================================================
print("\n" + "="*60)
print("STEP 12: WRITE OUTPUTS")
print("="*60)

# Write specification_results.csv
results_df = pd.DataFrame(results)
output_csv = os.path.join(PACKAGE_DIR, "specification_results.csv")
results_df.to_csv(output_csv, index=False)
print(f"Wrote {len(results_df)} specs to: {output_csv}")

# Write diagnostics_results.csv
diag_df = pd.DataFrame(diag_results)
diag_csv = os.path.join(PACKAGE_DIR, "diagnostics_results.csv")
diag_df.to_csv(diag_csv, index=False)
print(f"Wrote {len(diag_df)} diagnostics to: {diag_csv}")

# Write spec_diagnostics_map.csv
# Diagnostics are all at baseline_group scope, so link to baseline spec
diag_map = []
for d in diag_results:
    diag_map.append({
        'spec_run_id': f'{PAPER_ID}_G1_baseline',
        'diag_run_id': d['diag_run_id'],
        'scope': d['scope'],
    })
diag_map_df = pd.DataFrame(diag_map)
diag_map_csv = os.path.join(PACKAGE_DIR, "spec_diagnostics_map.csv")
diag_map_df.to_csv(diag_map_csv, index=False)
print(f"Wrote {len(diag_map_df)} diagnostic mappings to: {diag_map_csv}")

# Summary
print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"Total specifications executed: {len(results_df)}")
print(f"Successful: {results_df['coefficient'].notna().sum()}")
print(f"Failed: {results_df['coefficient'].isna().sum()}")
print(f"Diagnostics: {len(diag_df)}")
print(f"\nBaseline coefficient: {results_df.iloc[0]['coefficient']:.4f}")
print(f"Baseline SE: {results_df.iloc[0]['std_error']:.4f}")
print(f"Baseline p-value: {results_df.iloc[0]['p_value']:.4e}")
print(f"\nCoefficient range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")
print(f"p-value range: [{results_df['p_value'].min():.4e}, {results_df['p_value'].max():.4e}]")
