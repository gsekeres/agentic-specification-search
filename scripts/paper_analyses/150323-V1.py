"""
Specification Search: 150323-V1
Paper: Political Turnover, Bureaucratic Turnover and the Quality of Public Services
Authors: Mitra Akhtari, Diana Moreira, Laura Trucco
Method: Regression Discontinuity Design

This script runs a systematic specification search following the i4r methodology.
"""

import pandas as pd
import numpy as np
import json
import warnings
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')

# Configuration
PAPER_ID = "150323-V1"
PAPER_TITLE = "Political Turnover, Bureaucratic Turnover and the Quality of Public Services"
JOURNAL = "American Economic Review"
METHOD = "regression_discontinuity"
DATA_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/150323-V1/Data and Code/Data/Main Data"
OUTPUT_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/150323-V1"

# Results storage
results = []

def add_result(spec_id, spec_tree_path, outcome_var, treatment_var, model, df_sample,
               sample_desc="", controls_desc="", fixed_effects="", cluster_var="",
               model_type="RD", bandwidth=None, polynomial=1, kernel="uniform"):
    """Add a specification result to the results list."""

    try:
        coef = model.params[treatment_var]
        se = model.bse[treatment_var]
        tstat = model.tvalues[treatment_var]
        pval = model.pvalues[treatment_var]
        ci = model.conf_int().loc[treatment_var]
        ci_lower, ci_upper = ci.iloc[0], ci.iloc[1]
        n_obs = int(model.nobs)
        r2 = model.rsquared if hasattr(model, 'rsquared') else None

        coef_vector = {
            "treatment": {
                "var": treatment_var,
                "coef": float(coef),
                "se": float(se),
                "pval": float(pval)
            },
            "controls": [],
            "diagnostics": {
                "bandwidth": bandwidth,
                "polynomial": polynomial,
                "kernel": kernel
            }
        }
        for var in model.params.index:
            if var != treatment_var and var != 'Intercept' and var != 'const':
                coef_vector["controls"].append({
                    "var": var,
                    "coef": float(model.params[var]),
                    "se": float(model.bse[var]),
                    "pval": float(model.pvalues[var])
                })
    except Exception as e:
        print(f"Error extracting results for {spec_id}: {e}")
        return

    results.append({
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
        'n_obs': n_obs,
        'r_squared': float(r2) if r2 is not None else None,
        'coefficient_vector_json': json.dumps(coef_vector),
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })
    print(f"  {spec_id}: coef={coef:.4f}, se={se:.4f}, p={pval:.4f}, n={n_obs}")


def run_rd_regression(df, outcome, bandwidth, controls=None, cluster_var='COD_MUNICIPIO',
                      polynomial=1, use_cluster=True):
    """Run an RD regression within a specified bandwidth."""

    # Filter to bandwidth
    df_bw = df[df['pX'].abs() <= bandwidth].copy()

    # Remove missing values for outcome and key variables
    required_vars = [outcome, 'pX_dummy', 'pX', 'pX_pD']
    df_bw = df_bw.dropna(subset=required_vars)

    if len(df_bw) < 50:
        return None, df_bw

    # Build formula
    if polynomial == 1:
        formula = f"{outcome} ~ pX_dummy + pX + pX_pD"
    elif polynomial == 2:
        df_bw['pX_sq'] = df_bw['pX'] ** 2
        df_bw['pX_pD_sq'] = df_bw['pX_sq'] * df_bw['pX_dummy']
        formula = f"{outcome} ~ pX_dummy + pX + pX_pD + pX_sq + pX_pD_sq"
    else:
        formula = f"{outcome} ~ pX_dummy + pX + pX_pD"

    if controls:
        controls_clean = [c for c in controls if c in df_bw.columns]
        df_bw = df_bw.dropna(subset=controls_clean)
        if len(df_bw) < 50:
            return None, df_bw
        formula += " + " + " + ".join(controls_clean)

    try:
        if use_cluster and cluster_var in df_bw.columns:
            model = smf.ols(formula, data=df_bw).fit(
                cov_type='cluster', cov_kwds={'groups': df_bw[cluster_var]}
            )
        else:
            model = smf.ols(formula, data=df_bw).fit(cov_type='HC1')
        return model, df_bw
    except Exception as e:
        print(f"  Regression error: {e}")
        return None, df_bw


# ============================================================================
# LOAD DATA
# ============================================================================
print("Loading data...")

# School-level municipal schools (main analysis dataset)
df_munic = pd.read_stata(f"{DATA_DIR}/s_MainData_SchlLevel2007_2013_MunicSchools.dta")
print(f"School-level municipal data: {df_munic.shape}")

# School-level non-municipal schools (for placebo/comparison)
df_nonmunic = pd.read_stata(f"{DATA_DIR}/s_MainData_SchlLevel2007_2013_NonMunicSchools.dta")
print(f"School-level non-municipal data: {df_nonmunic.shape}")

# ============================================================================
# PREPARE DATA
# ============================================================================
print("\nPreparing data...")

# Filter to estimation sample (2009 and 2013 post-election years)
# Exclude supplementary elections and large populations
df_munic['year'] = df_munic['year'].astype(float)
df_munic_est = df_munic[
    ((df_munic['year'] == 2009) | (df_munic['year'] == 2013)) &
    ~((df_munic['year'] == 2009) & (df_munic['supplement_2008'] == 1)) &
    ~((df_munic['year'] == 2013) & (df_munic['supplement_2012'] == 1)) &
    ~((df_munic['year'] == 2009) & (df_munic['population_large'] == 1)) &
    ~((df_munic['year'] == 2013) & (df_munic['population_large'] == 1))
].copy()

df_munic_est['year_dummy'] = (df_munic_est['year'] == 2013).astype(int)

print(f"Estimation sample size: {len(df_munic_est)}")
print(f"Unique municipalities: {df_munic_est['COD_MUNICIPIO'].nunique()}")

# Do the same for non-municipal
df_nonmunic['year'] = df_nonmunic['year'].astype(float)
df_nonmunic_est = df_nonmunic[
    ((df_nonmunic['year'] == 2009) | (df_nonmunic['year'] == 2013)) &
    ~((df_nonmunic['year'] == 2009) & (df_nonmunic['supplement_2008'] == 1)) &
    ~((df_nonmunic['year'] == 2013) & (df_nonmunic['supplement_2012'] == 1)) &
    ~((df_nonmunic['year'] == 2009) & (df_nonmunic['population_large'] == 1)) &
    ~((df_nonmunic['year'] == 2013) & (df_nonmunic['population_large'] == 1))
].copy()
df_nonmunic_est['year_dummy'] = (df_nonmunic_est['year'] == 2013).astype(int)

# Define control variables
school_controls = ['urban_schl', 'Nstaff_schl', 'alltchr_docen', 'eqpinternet_schl',
                   'waterpblcnetwork_schl', 'sewerpblcnetwork_schl']
teacher_controls = ['agetchr_docen', 'femaletchr_docen', 'eductchr_BA_docen',
                    'contracttchr_concurso_docen']

# Main outcomes
OUTCOMES = {
    'both_score_4_std': '4th Grade Test Scores',
    'both_score_8_std': '8th Grade Test Scores',
    'tx_abandono_primary': 'Dropout Rate',
    'newtchr': 'New Teachers (Share)',
    'lefttchr': 'Teachers Left (Share)',
    'expthisschl_lessthan2_DPB': 'Headmaster Replacement'
}

# ============================================================================
# BASELINE SPECIFICATIONS
# ============================================================================
print("\n" + "="*60)
print("BASELINE SPECIFICATIONS")
print("="*60)

# Primary outcome: 4th grade test scores
outcome = 'both_score_4_std'
h_opt = 0.09  # Paper's typical bandwidth

model, df_sample = run_rd_regression(df_munic_est, outcome, h_opt,
                                      controls=['both_score_4_baseline', 'year_dummy'] + school_controls)
if model:
    add_result('baseline', 'methods/regression_discontinuity.md#baseline',
               outcome, 'pX_dummy', model, df_sample,
               sample_desc=f"Municipal schools, |pX|<{h_opt}, 2009 & 2013",
               controls_desc="Baseline scores + school controls",
               cluster_var="COD_MUNICIPIO", bandwidth=h_opt)

# ============================================================================
# RD BANDWIDTH VARIATIONS
# ============================================================================
print("\n" + "="*60)
print("BANDWIDTH VARIATIONS")
print("="*60)

bandwidths = {
    'optimal': 0.09,
    'narrow': 0.07,
    'wide': 0.11,
    'very_narrow': 0.05,
    'very_wide': 0.15,
    'half_optimal': 0.045,
    'double_optimal': 0.18
}

for bw_name, bw in bandwidths.items():
    for outcome in ['both_score_4_std', 'newtchr', 'expthisschl_lessthan2_DPB']:
        baseline_var = 'both_score_4_baseline' if outcome == 'both_score_4_std' else None
        controls = ['year_dummy'] + school_controls
        if baseline_var:
            controls = [baseline_var] + controls

        model, df_sample = run_rd_regression(df_munic_est, outcome, bw, controls=controls)
        if model:
            add_result(f'rd/bandwidth/{bw_name}_{outcome[:10]}',
                      'methods/regression_discontinuity.md#bandwidth-selection',
                      outcome, 'pX_dummy', model, df_sample,
                      sample_desc=f"|pX|<{bw}",
                      controls_desc="Baseline + school controls" if baseline_var else "School controls",
                      cluster_var="COD_MUNICIPIO", bandwidth=bw)

# ============================================================================
# POLYNOMIAL ORDER VARIATIONS
# ============================================================================
print("\n" + "="*60)
print("POLYNOMIAL ORDER VARIATIONS")
print("="*60)

for poly in [1, 2]:
    for outcome in ['both_score_4_std', 'newtchr']:
        model, df_sample = run_rd_regression(df_munic_est, outcome, 0.09,
                                             controls=['year_dummy'] + school_controls,
                                             polynomial=poly)
        if model:
            add_result(f'rd/poly/order{poly}_{outcome[:10]}',
                      'methods/regression_discontinuity.md#polynomial-order',
                      outcome, 'pX_dummy', model, df_sample,
                      sample_desc="|pX|<0.09",
                      controls_desc="School controls",
                      cluster_var="COD_MUNICIPIO", bandwidth=0.09, polynomial=poly)

# ============================================================================
# CONTROL SET VARIATIONS
# ============================================================================
print("\n" + "="*60)
print("CONTROL SET VARIATIONS")
print("="*60)

# No controls
for outcome in ['both_score_4_std', 'newtchr', 'expthisschl_lessthan2_DPB', 'tx_abandono_primary']:
    model, df_sample = run_rd_regression(df_munic_est, outcome, 0.09, controls=None)
    if model:
        add_result(f'rd/controls/none_{outcome[:10]}',
                  'methods/regression_discontinuity.md#control-sets',
                  outcome, 'pX_dummy', model, df_sample,
                  sample_desc="|pX|<0.09",
                  controls_desc="No controls",
                  cluster_var="COD_MUNICIPIO", bandwidth=0.09)

# Year dummy only
for outcome in ['both_score_4_std', 'newtchr']:
    model, df_sample = run_rd_regression(df_munic_est, outcome, 0.09, controls=['year_dummy'])
    if model:
        add_result(f'rd/controls/year_only_{outcome[:10]}',
                  'methods/regression_discontinuity.md#control-sets',
                  outcome, 'pX_dummy', model, df_sample,
                  sample_desc="|pX|<0.09",
                  controls_desc="Year dummy only",
                  cluster_var="COD_MUNICIPIO", bandwidth=0.09)

# Full controls
full_controls = ['year_dummy'] + school_controls + teacher_controls
for outcome in ['both_score_4_std', 'newtchr']:
    model, df_sample = run_rd_regression(df_munic_est, outcome, 0.09, controls=full_controls)
    if model:
        add_result(f'rd/controls/full_{outcome[:10]}',
                  'methods/regression_discontinuity.md#control-sets',
                  outcome, 'pX_dummy', model, df_sample,
                  sample_desc="|pX|<0.09",
                  controls_desc="Full controls (school + teacher)",
                  cluster_var="COD_MUNICIPIO", bandwidth=0.09)

# ============================================================================
# LEAVE-ONE-OUT CONTROL VARIATIONS
# ============================================================================
print("\n" + "="*60)
print("LEAVE-ONE-OUT CONTROL VARIATIONS")
print("="*60)

all_controls = ['year_dummy'] + school_controls
for drop_control in school_controls:
    remaining = [c for c in all_controls if c != drop_control]
    model, df_sample = run_rd_regression(df_munic_est, 'both_score_4_std', 0.09, controls=remaining)
    if model:
        add_result(f'robust/loo/drop_{drop_control}',
                  'robustness/leave_one_out.md',
                  'both_score_4_std', 'pX_dummy', model, df_sample,
                  sample_desc="|pX|<0.09",
                  controls_desc=f"Controls minus {drop_control}",
                  cluster_var="COD_MUNICIPIO", bandwidth=0.09)

# ============================================================================
# ALTERNATIVE OUTCOMES
# ============================================================================
print("\n" + "="*60)
print("ALTERNATIVE OUTCOMES")
print("="*60)

for outcome, desc in OUTCOMES.items():
    baseline_var = None
    if outcome == 'both_score_4_std':
        baseline_var = 'both_score_4_baseline'
    elif outcome == 'tx_abandono_primary':
        baseline_var = 'tx_abandono_primary_baseline'

    controls = ['year_dummy'] + school_controls
    if baseline_var and baseline_var in df_munic_est.columns:
        controls = [baseline_var] + controls

    model, df_sample = run_rd_regression(df_munic_est, outcome, 0.09, controls=controls)
    if model:
        add_result(f'robust/outcome/{outcome}',
                  'robustness/measurement.md',
                  outcome, 'pX_dummy', model, df_sample,
                  sample_desc="|pX|<0.09",
                  controls_desc="Baseline + school controls" if baseline_var else "School controls",
                  cluster_var="COD_MUNICIPIO", bandwidth=0.09)

# ============================================================================
# SAMPLE RESTRICTIONS
# ============================================================================
print("\n" + "="*60)
print("SAMPLE RESTRICTIONS")
print("="*60)

# By year
for year in [2009, 2013]:
    df_year = df_munic_est[df_munic_est['year'] == year].copy()
    model, df_sample = run_rd_regression(df_year, 'both_score_4_std', 0.09,
                                          controls=['both_score_4_baseline'] + school_controls)
    if model:
        add_result(f'robust/sample/year_{year}',
                  'robustness/sample_restrictions.md',
                  'both_score_4_std', 'pX_dummy', model, df_sample,
                  sample_desc=f"Year {year} only, |pX|<0.09",
                  controls_desc="Baseline + school controls",
                  cluster_var="COD_MUNICIPIO", bandwidth=0.09)

# Urban vs rural schools
for urban_val, urban_name in [(1, 'urban'), (0, 'rural')]:
    df_urban = df_munic_est[df_munic_est['urban_schl'] == urban_val].copy()
    model, df_sample = run_rd_regression(df_urban, 'both_score_4_std', 0.09,
                                          controls=['year_dummy', 'both_score_4_baseline'])
    if model:
        add_result(f'robust/sample/{urban_name}_schools',
                  'robustness/sample_restrictions.md',
                  'both_score_4_std', 'pX_dummy', model, df_sample,
                  sample_desc=f"{urban_name.capitalize()} schools only, |pX|<0.09",
                  controls_desc="Baseline scores + year dummy",
                  cluster_var="COD_MUNICIPIO", bandwidth=0.09)

# By population size
df_munic_est['pop_median'] = df_munic_est['population'] > df_munic_est['population'].median()
for pop_val, pop_name in [(True, 'large_pop'), (False, 'small_pop')]:
    df_pop = df_munic_est[df_munic_est['pop_median'] == pop_val].copy()
    model, df_sample = run_rd_regression(df_pop, 'both_score_4_std', 0.09,
                                          controls=['year_dummy', 'both_score_4_baseline'] + school_controls)
    if model:
        add_result(f'robust/sample/{pop_name}',
                  'robustness/sample_restrictions.md',
                  'both_score_4_std', 'pX_dummy', model, df_sample,
                  sample_desc=f"{pop_name.replace('_', ' ').title()} municipalities, |pX|<0.09",
                  controls_desc="Baseline + school controls",
                  cluster_var="COD_MUNICIPIO", bandwidth=0.09)

# Schools with internet vs without
for internet_val, internet_name in [(1, 'with_internet'), (0, 'no_internet')]:
    df_int = df_munic_est[df_munic_est['eqpinternet_schl'] == internet_val].copy()
    model, df_sample = run_rd_regression(df_int, 'both_score_4_std', 0.09,
                                          controls=['year_dummy', 'both_score_4_baseline'])
    if model:
        add_result(f'robust/sample/{internet_name}',
                  'robustness/sample_restrictions.md',
                  'both_score_4_std', 'pX_dummy', model, df_sample,
                  sample_desc=f"Schools {internet_name.replace('_', ' ')}, |pX|<0.09",
                  controls_desc="Baseline scores + year dummy",
                  cluster_var="COD_MUNICIPIO", bandwidth=0.09)

# ============================================================================
# DONUT HOLE SPECIFICATIONS
# ============================================================================
print("\n" + "="*60)
print("DONUT HOLE SPECIFICATIONS")
print("="*60)

for donut in [0.01, 0.02, 0.03]:
    df_donut = df_munic_est[(df_munic_est['pX'].abs() > donut) & (df_munic_est['pX'].abs() <= 0.09)].copy()
    model, df_sample = run_rd_regression(df_donut, 'both_score_4_std', 0.09,
                                          controls=['year_dummy', 'both_score_4_baseline'] + school_controls)
    if model:
        add_result(f'rd/donut/exclude_{int(donut*100)}pct',
                  'methods/regression_discontinuity.md#donut-hole-specifications',
                  'both_score_4_std', 'pX_dummy', model, df_sample,
                  sample_desc=f"|pX|>{donut} and |pX|<0.09",
                  controls_desc="Baseline + school controls",
                  cluster_var="COD_MUNICIPIO", bandwidth=0.09)

# ============================================================================
# PLACEBO TESTS - NON-MUNICIPAL SCHOOLS
# ============================================================================
print("\n" + "="*60)
print("PLACEBO TESTS - NON-MUNICIPAL SCHOOLS")
print("="*60)

# Non-municipal schools should NOT be affected by municipal political turnover
for outcome in ['both_score_4_std', 'newtchr', 'expthisschl_lessthan2_DPB']:
    baseline_var = 'both_score_4_baseline' if outcome == 'both_score_4_std' else None
    controls = ['year_dummy'] + school_controls
    if baseline_var and baseline_var in df_nonmunic_est.columns:
        controls = [baseline_var] + controls

    model, df_sample = run_rd_regression(df_nonmunic_est, outcome, 0.09, controls=controls)
    if model:
        add_result(f'robust/placebo/nonmunic_{outcome[:10]}',
                  'robustness/placebo_tests.md',
                  outcome, 'pX_dummy', model, df_sample,
                  sample_desc="Non-municipal schools (placebo), |pX|<0.09",
                  controls_desc="Baseline + school controls" if baseline_var else "School controls",
                  cluster_var="COD_MUNICIPIO", bandwidth=0.09)

# ============================================================================
# PLACEBO CUTOFFS
# ============================================================================
print("\n" + "="*60)
print("PLACEBO CUTOFFS")
print("="*60)

# Test at median of left side (should find no effect)
df_left = df_munic_est[df_munic_est['pX'] < 0].copy()
left_median = df_left['pX'].median()
df_left['pX_placebo'] = df_left['pX'] - left_median
df_left['pX_dummy_placebo'] = (df_left['pX'] > left_median).astype(int)
df_left['pX_pD_placebo'] = df_left['pX_placebo'] * df_left['pX_dummy_placebo']

# Filter to bandwidth around placebo cutoff
df_left_bw = df_left[df_left['pX_placebo'].abs() <= 0.09].copy()
df_left_bw = df_left_bw.dropna(subset=['both_score_4_std', 'pX_placebo', 'pX_dummy_placebo'])

if len(df_left_bw) > 100:
    try:
        formula = "both_score_4_std ~ pX_dummy_placebo + pX_placebo + pX_pD_placebo + year_dummy"
        model = smf.ols(formula, data=df_left_bw).fit(
            cov_type='cluster', cov_kwds={'groups': df_left_bw['COD_MUNICIPIO']}
        )
        add_result('rd/placebo/cutoff_left_median',
                  'methods/regression_discontinuity.md#placebo-cutoff-tests',
                  'both_score_4_std', 'pX_dummy_placebo', model, df_left_bw,
                  sample_desc=f"Left of cutoff, placebo at {left_median:.3f}",
                  controls_desc="Year dummy",
                  cluster_var="COD_MUNICIPIO", bandwidth=0.09)
    except Exception as e:
        print(f"  Placebo cutoff error: {e}")

# Test at median of right side
df_right = df_munic_est[df_munic_est['pX'] > 0].copy()
right_median = df_right['pX'].median()
df_right['pX_placebo'] = df_right['pX'] - right_median
df_right['pX_dummy_placebo'] = (df_right['pX'] > right_median).astype(int)
df_right['pX_pD_placebo'] = df_right['pX_placebo'] * df_right['pX_dummy_placebo']

df_right_bw = df_right[df_right['pX_placebo'].abs() <= 0.09].copy()
df_right_bw = df_right_bw.dropna(subset=['both_score_4_std', 'pX_placebo', 'pX_dummy_placebo'])

if len(df_right_bw) > 100:
    try:
        formula = "both_score_4_std ~ pX_dummy_placebo + pX_placebo + pX_pD_placebo + year_dummy"
        model = smf.ols(formula, data=df_right_bw).fit(
            cov_type='cluster', cov_kwds={'groups': df_right_bw['COD_MUNICIPIO']}
        )
        add_result('rd/placebo/cutoff_right_median',
                  'methods/regression_discontinuity.md#placebo-cutoff-tests',
                  'both_score_4_std', 'pX_dummy_placebo', model, df_right_bw,
                  sample_desc=f"Right of cutoff, placebo at {right_median:.3f}",
                  controls_desc="Year dummy",
                  cluster_var="COD_MUNICIPIO", bandwidth=0.09)
    except Exception as e:
        print(f"  Placebo cutoff error: {e}")

# ============================================================================
# HETEROGENEITY ANALYSES
# ============================================================================
print("\n" + "="*60)
print("HETEROGENEITY ANALYSES")
print("="*60)

# By urban/rural
df_munic_est['pX_dummy_urban'] = df_munic_est['pX_dummy'] * df_munic_est['urban_schl']
df_bw = df_munic_est[df_munic_est['pX'].abs() <= 0.09].copy()
df_bw = df_bw.dropna(subset=['both_score_4_std', 'pX_dummy', 'urban_schl', 'pX_dummy_urban'])

if len(df_bw) > 100:
    try:
        formula = "both_score_4_std ~ pX_dummy + pX + pX_pD + urban_schl + pX_dummy_urban + year_dummy"
        model = smf.ols(formula, data=df_bw).fit(
            cov_type='cluster', cov_kwds={'groups': df_bw['COD_MUNICIPIO']}
        )
        add_result('robust/het/urban_interaction',
                  'robustness/heterogeneity.md',
                  'both_score_4_std', 'pX_dummy', model, df_bw,
                  sample_desc="|pX|<0.09",
                  controls_desc="Includes urban interaction",
                  cluster_var="COD_MUNICIPIO", bandwidth=0.09)
    except Exception as e:
        print(f"  Heterogeneity error: {e}")

# By school size (number of teachers)
df_munic_est['large_school'] = (df_munic_est['alltchr_docen'] > df_munic_est['alltchr_docen'].median()).astype(int)
df_munic_est['pX_dummy_large'] = df_munic_est['pX_dummy'] * df_munic_est['large_school']
df_bw = df_munic_est[df_munic_est['pX'].abs() <= 0.09].copy()
df_bw = df_bw.dropna(subset=['both_score_4_std', 'pX_dummy', 'large_school', 'pX_dummy_large'])

if len(df_bw) > 100:
    try:
        formula = "both_score_4_std ~ pX_dummy + pX + pX_pD + large_school + pX_dummy_large + year_dummy"
        model = smf.ols(formula, data=df_bw).fit(
            cov_type='cluster', cov_kwds={'groups': df_bw['COD_MUNICIPIO']}
        )
        add_result('robust/het/school_size_interaction',
                  'robustness/heterogeneity.md',
                  'both_score_4_std', 'pX_dummy', model, df_bw,
                  sample_desc="|pX|<0.09",
                  controls_desc="Includes school size interaction",
                  cluster_var="COD_MUNICIPIO", bandwidth=0.09)
    except Exception as e:
        print(f"  Heterogeneity error: {e}")

# By infrastructure (internet)
df_munic_est['pX_dummy_internet'] = df_munic_est['pX_dummy'] * df_munic_est['eqpinternet_schl']
df_bw = df_munic_est[df_munic_est['pX'].abs() <= 0.09].copy()
df_bw = df_bw.dropna(subset=['both_score_4_std', 'pX_dummy', 'eqpinternet_schl', 'pX_dummy_internet'])

if len(df_bw) > 100:
    try:
        formula = "both_score_4_std ~ pX_dummy + pX + pX_pD + eqpinternet_schl + pX_dummy_internet + year_dummy"
        model = smf.ols(formula, data=df_bw).fit(
            cov_type='cluster', cov_kwds={'groups': df_bw['COD_MUNICIPIO']}
        )
        add_result('robust/het/internet_interaction',
                  'robustness/heterogeneity.md',
                  'both_score_4_std', 'pX_dummy', model, df_bw,
                  sample_desc="|pX|<0.09",
                  controls_desc="Includes internet interaction",
                  cluster_var="COD_MUNICIPIO", bandwidth=0.09)
    except Exception as e:
        print(f"  Heterogeneity error: {e}")

# ============================================================================
# INFERENCE VARIATIONS
# ============================================================================
print("\n" + "="*60)
print("INFERENCE VARIATIONS")
print("="*60)

# Robust (heteroskedasticity-consistent) SEs
df_bw = df_munic_est[df_munic_est['pX'].abs() <= 0.09].copy()
df_bw = df_bw.dropna(subset=['both_score_4_std', 'pX_dummy', 'pX', 'pX_pD', 'year_dummy'])

for se_type, se_name in [('HC1', 'robust_hc1'), ('HC3', 'robust_hc3')]:
    if len(df_bw) > 100:
        try:
            formula = "both_score_4_std ~ pX_dummy + pX + pX_pD + year_dummy"
            model = smf.ols(formula, data=df_bw).fit(cov_type=se_type)
            add_result(f'robust/inference/{se_name}',
                      'robustness/inference_alternatives.md',
                      'both_score_4_std', 'pX_dummy', model, df_bw,
                      sample_desc="|pX|<0.09",
                      controls_desc="Year dummy",
                      cluster_var=f"{se_type} robust", bandwidth=0.09)
        except Exception as e:
            print(f"  Inference error: {e}")

# ============================================================================
# FUNCTIONAL FORM VARIATIONS
# ============================================================================
print("\n" + "="*60)
print("FUNCTIONAL FORM VARIATIONS")
print("="*60)

# Log transformation of continuous outcomes
for outcome in ['newtchr', 'lefttchr']:
    df_bw = df_munic_est[df_munic_est['pX'].abs() <= 0.09].copy()
    df_bw = df_bw.dropna(subset=[outcome, 'pX_dummy', 'pX', 'pX_pD', 'year_dummy'])
    df_bw[f'log_{outcome}'] = np.log(df_bw[outcome] + 0.01)

    if len(df_bw) > 100:
        try:
            formula = f"log_{outcome} ~ pX_dummy + pX + pX_pD + year_dummy"
            model = smf.ols(formula, data=df_bw).fit(
                cov_type='cluster', cov_kwds={'groups': df_bw['COD_MUNICIPIO']}
            )
            add_result(f'robust/funcform/log_{outcome}',
                      'robustness/functional_form.md',
                      f'log_{outcome}', 'pX_dummy', model, df_bw,
                      sample_desc="|pX|<0.09",
                      controls_desc="Year dummy, log transform",
                      cluster_var="COD_MUNICIPIO", bandwidth=0.09)
        except Exception as e:
            print(f"  Functional form error: {e}")

# IHS transformation
df_bw = df_munic_est[df_munic_est['pX'].abs() <= 0.09].copy()
df_bw = df_bw.dropna(subset=['both_score_4_std', 'pX_dummy', 'pX', 'pX_pD', 'year_dummy'])
df_bw['ihs_score'] = np.arcsinh(df_bw['both_score_4_std'])

if len(df_bw) > 100:
    try:
        formula = "ihs_score ~ pX_dummy + pX + pX_pD + year_dummy"
        model = smf.ols(formula, data=df_bw).fit(
            cov_type='cluster', cov_kwds={'groups': df_bw['COD_MUNICIPIO']}
        )
        add_result('robust/funcform/ihs_score',
                  'robustness/functional_form.md',
                  'ihs_score', 'pX_dummy', model, df_bw,
                  sample_desc="|pX|<0.09",
                  controls_desc="Year dummy, IHS transform",
                  cluster_var="COD_MUNICIPIO", bandwidth=0.09)
    except Exception as e:
        print(f"  Functional form error: {e}")

# ============================================================================
# SYMMETRIC SAMPLE SPECIFICATIONS
# ============================================================================
print("\n" + "="*60)
print("SYMMETRIC SAMPLE SPECIFICATIONS")
print("="*60)

# Create symmetric sample (equal N on each side of cutoff)
df_bw = df_munic_est[df_munic_est['pX'].abs() <= 0.09].copy()
df_left_sym = df_bw[df_bw['pX'] < 0]
df_right_sym = df_bw[df_bw['pX'] >= 0]
n_sym = min(len(df_left_sym), len(df_right_sym))

if n_sym > 100:
    df_sym = pd.concat([df_left_sym.sample(n_sym, random_state=42),
                        df_right_sym.sample(n_sym, random_state=42)])

    model, df_sample = run_rd_regression(df_sym, 'both_score_4_std', 0.09,
                                          controls=['year_dummy'] + school_controls)
    if model:
        add_result('rd/sample/symmetric',
                  'methods/regression_discontinuity.md#sample-restrictions',
                  'both_score_4_std', 'pX_dummy', model, df_sample,
                  sample_desc="Symmetric sample around cutoff",
                  controls_desc="School controls + year dummy",
                  cluster_var="COD_MUNICIPIO", bandwidth=0.09)

# ============================================================================
# FINE-GRAINED BANDWIDTH SENSITIVITY
# ============================================================================
print("\n" + "="*60)
print("FINE-GRAINED BANDWIDTH SENSITIVITY")
print("="*60)

for bw in np.arange(0.04, 0.16, 0.02):
    bw = round(bw, 2)
    model, df_sample = run_rd_regression(df_munic_est, 'both_score_4_std', bw,
                                          controls=['year_dummy', 'both_score_4_baseline'] + school_controls)
    if model:
        add_result(f'rd/validity/sensitivity_bw{int(bw*100):02d}',
                  'methods/regression_discontinuity.md#validation-tests',
                  'both_score_4_std', 'pX_dummy', model, df_sample,
                  sample_desc=f"|pX|<{bw}",
                  controls_desc="Baseline + school controls",
                  cluster_var="COD_MUNICIPIO", bandwidth=bw)

# ============================================================================
# ADDITIONAL BANDWIDTH SENSITIVITY FOR OTHER OUTCOMES
# ============================================================================
print("\n" + "="*60)
print("ADDITIONAL BANDWIDTH SENSITIVITY")
print("="*60)

for outcome in ['newtchr', 'expthisschl_lessthan2_DPB', 'tx_abandono_primary']:
    for bw in [0.06, 0.08, 0.10, 0.12]:
        model, df_sample = run_rd_regression(df_munic_est, outcome, bw,
                                              controls=['year_dummy'] + school_controls)
        if model:
            add_result(f'rd/validity/{outcome[:6]}_bw{int(bw*100):02d}',
                      'methods/regression_discontinuity.md#validation-tests',
                      outcome, 'pX_dummy', model, df_sample,
                      sample_desc=f"|pX|<{bw}",
                      controls_desc="School controls",
                      cluster_var="COD_MUNICIPIO", bandwidth=bw)

# ============================================================================
# ADDITIONAL SAMPLE RESTRICTIONS
# ============================================================================
print("\n" + "="*60)
print("ADDITIONAL SAMPLE RESTRICTIONS")
print("="*60)

# By water/sewage infrastructure
for infra, infra_name in [('waterpblcnetwork_schl', 'water'), ('sewerpblcnetwork_schl', 'sewage')]:
    for val, val_name in [(1, f'with_{infra_name}'), (0, f'no_{infra_name}')]:
        df_infra = df_munic_est[df_munic_est[infra] == val].copy()
        model, df_sample = run_rd_regression(df_infra, 'both_score_4_std', 0.09,
                                              controls=['year_dummy', 'both_score_4_baseline'])
        if model:
            add_result(f'robust/sample/{val_name}',
                      'robustness/sample_restrictions.md',
                      'both_score_4_std', 'pX_dummy', model, df_sample,
                      sample_desc=f"Schools {val_name.replace('_', ' ')}, |pX|<0.09",
                      controls_desc="Baseline + year dummy",
                      cluster_var="COD_MUNICIPIO", bandwidth=0.09)

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Convert to DataFrame
results_df = pd.DataFrame(results)
print(f"Total specifications: {len(results_df)}")

# Save to CSV
output_path = f"{OUTPUT_DIR}/specification_results.csv"
results_df.to_csv(output_path, index=False)
print(f"Results saved to: {output_path}")

# Print summary statistics
print("\n=== SUMMARY STATISTICS ===")
print(f"Total specifications: {len(results_df)}")
print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
print(f"Negative coefficients: {(results_df['coefficient'] < 0).sum()} ({100*(results_df['coefficient'] < 0).mean():.1f}%)")
print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")
print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")
print(f"Range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")

# Summary by category
print("\n=== SUMMARY BY SPECIFICATION CATEGORY ===")
results_df['category'] = results_df['spec_id'].apply(lambda x: x.split('/')[0])
summary = results_df.groupby('category').agg({
    'coefficient': ['count', 'mean', 'median'],
    'p_value': lambda x: (x < 0.05).mean()
}).round(3)
summary.columns = ['N', 'Mean Coef', 'Median Coef', 'Pct Sig 5%']
print(summary)

# Summary by outcome
print("\n=== SUMMARY BY OUTCOME ===")
outcome_summary = results_df.groupby('outcome_var').agg({
    'coefficient': ['count', 'mean', 'median'],
    'p_value': lambda x: (x < 0.05).mean()
}).round(3)
outcome_summary.columns = ['N', 'Mean Coef', 'Median Coef', 'Pct Sig 5%']
print(outcome_summary)
