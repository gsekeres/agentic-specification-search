"""
Specification Search for Paper 146041-V1
"Human Capital in the Presence of Skilled-Biased Technical Change"

This paper estimates the relationship between GDP per worker (l_y) and:
1. Relative skill efficiency (AQ) - measured from domestic wage differentials
2. Relative human capital (Q) - measured from immigrant wage differentials in the US

Method: Cross-sectional OLS regressions
Primary specification: log(AQ) ~ log(GDP per worker), log(Q) ~ log(GDP per worker)

Author: Claude Agent
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
import json

warnings.filterwarnings('ignore')

# Configuration
PAPER_ID = "146041-V1"
PAPER_TITLE = "Human Capital in the Presence of Skilled-Biased Technical Change"
JOURNAL = "AER"  # Based on typical AEA publication patterns
BASE_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/146041-V1/Replication"

# Results storage
results = []

def add_result(spec_id, spec_tree_path, outcome_var, treatment_var, model, sample_desc,
               controls_desc, model_type="OLS", fixed_effects="None", cluster_var="None",
               n_obs=None, r_squared=None, coef_vector=None):
    """Add a regression result to the results list."""

    try:
        coef = model.params[treatment_var]
        se = model.bse[treatment_var]
        t_stat = model.tvalues[treatment_var]
        p_val = model.pvalues[treatment_var]
        ci = model.conf_int().loc[treatment_var]
        ci_lower = ci[0]
        ci_upper = ci[1]
    except:
        coef = se = t_stat = p_val = ci_lower = ci_upper = np.nan

    if n_obs is None:
        n_obs = int(model.nobs)
    if r_squared is None:
        r_squared = model.rsquared if hasattr(model, 'rsquared') else np.nan

    # Build coefficient vector JSON
    if coef_vector is None:
        coef_vector = {
            "treatment": {
                "var": treatment_var,
                "coef": float(coef) if not np.isnan(coef) else None,
                "se": float(se) if not np.isnan(se) else None,
                "pval": float(p_val) if not np.isnan(p_val) else None
            },
            "controls": [],
            "diagnostics": {
                "r_squared": float(r_squared) if not np.isnan(r_squared) else None,
                "n_obs": n_obs
            }
        }
        # Add other coefficients
        for var in model.params.index:
            if var != treatment_var and var != 'Intercept' and var != 'const':
                coef_vector["controls"].append({
                    "var": var,
                    "coef": float(model.params[var]),
                    "se": float(model.bse[var]),
                    "pval": float(model.pvalues[var])
                })

    results.append({
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
        'r_squared': r_squared,
        'coefficient_vector_json': json.dumps(coef_vector),
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })

def run_ols(df, outcome_var, treatment_var, controls=None, robust=True, sample_name="full"):
    """Run OLS regression and return model."""
    df_clean = df.dropna(subset=[outcome_var, treatment_var])
    if controls:
        df_clean = df_clean.dropna(subset=controls)

    if controls and len(controls) > 0:
        formula = f"{outcome_var} ~ {treatment_var} + {' + '.join(controls)}"
    else:
        formula = f"{outcome_var} ~ {treatment_var}"

    cov_type = 'HC1' if robust else 'nonrobust'
    model = smf.ols(formula, data=df_clean).fit(cov_type=cov_type)

    return model

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading data...")

# Load AQ_2000 data (relative skill efficiency - country level)
aq_df = pd.read_stata(f'{BASE_PATH}/temp/AQ_2000.dta')

# Load Q data (relative human capital from migrant analysis)
q_df = pd.read_stata(f'{BASE_PATH}/temp/Q.dta')

# Load Q_origins (unique country observations)
q_origins = pd.read_stata(f'{BASE_PATH}/temp/Q_origins.dta')

# Create log variables for AQ data
for var in ['irAQ53_dum_skti_hrs_secall', 'irAQ53rh_dum_skti_hrs_secall', 'irAQ53rl_dum_skti_hrs_secall',
            'irAQ53_dumx_skti_hrs_secall', 'irAQ53rh_dumx_skti_hrs_secall', 'irAQ53rl_dumx_skti_hrs_secall',
            'irAQ53_dum_skti_bod_secall', 'irAQ53_dum_skti_pop_secall', 'irAQ53_minc_skti_pop_secall',
            'irAQ53_dumse_skti_hrs_secall', 'irAQ53rh_dumse_skti_hrs_secall', 'irAQ53rl_dumse_skti_hrs_secall',
            'irAQ53_blee_skti', 'irAQ53rh_blee_skti', 'irAQ53rl_blee_skti',
            'wrat53_dum_skti_secall', 'H5L3_dum_skti_hrs_secall',
            'wrat53_dumx_skti_secall', 'H5L3_dumx_skti_hrs_secall',
            'wrat53_dumse_skti_secall', 'H5L3_dumse_skti_hrs_secall',
            'irAQ53_dum_skti_hrs_sec1', 'irAQ53_dum_skti_hrs_sec2', 'irAQ53_dum_skti_hrs_sec3', 'irAQ53_dum_skti_hrs_sec4',
            'wrat53_dum_skti_sec1', 'wrat53_dum_skti_sec2', 'wrat53_dum_skti_sec3', 'wrat53_dum_skti_sec4',
            'H5L3_dum_skti_hrs_sec1', 'H5L3_dum_skti_hrs_sec2', 'H5L3_dum_skti_hrs_sec3', 'H5L3_dum_skti_hrs_sec4']:
    if var in aq_df.columns:
        aq_df[f'l_{var}'] = np.log(aq_df[var].replace(0, np.nan))

# Create log variables for Q data
for var in ['irQ53_dum', 'irQ53_pool_dum', 'irQ53sel_dum', 'irQ53yh_dum',
            'irQ53goodeng_dum', 'irQ53nodown_dum', 'irQ53nomism_dum',
            'irQ53sorts_dum', 'irQ53sortr_dum',
            'irAQ53_blee_skti', 'irAQ53rh_blee_skti', 'irAQ53rl_blee_skti',
            'irAQ53_dum_skti_hrs_secall', 'irAQ53rh_dum_skti_hrs_secall', 'irAQ53rl_dum_skti_hrs_secall']:
    if var in q_df.columns:
        q_df[f'l_{var}'] = np.log(q_df[var].replace(0, np.nan))
    if var in q_origins.columns:
        q_origins[f'l_{var}'] = np.log(q_origins[var].replace(0, np.nan))

# Define sample restrictions
aq_micro = aq_df[aq_df['sample_micro'] == 1].copy()
q_us = q_df[q_df['sample'] == 'US Pooled'].copy()
q_pooled = q_df[q_df['sample_migr'] == 1].copy()

print(f"AQ micro sample: {len(aq_micro)} countries")
print(f"Q US immigrants sample: {len(q_us)} countries")
print(f"Q pooled sample: {len(q_pooled)} observations")

# ============================================================================
# PART 1: BASELINE SPECIFICATIONS - Table 1 (AQ regressions)
# ============================================================================

print("\n" + "="*70)
print("PART 1: BASELINE SPECIFICATIONS - AQ ANALYSIS")
print("="*70)

# 1. Baseline: log(AQ) on log(GDP) - micro sample
spec_id = 'baseline'
model = run_ols(aq_micro, 'l_irAQ53_dum_skti_hrs_secall', 'l_y')
add_result(spec_id, 'methods/cross_sectional_ols.md', 'l_irAQ53_dum_skti_hrs_secall', 'l_y', model,
           'Micro sample (12 countries)', 'None', 'OLS')
print(f"1. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}, n={int(model.nobs)}")

# 2. Log wage ratio on log(GDP)
spec_id = 'ols/outcome/wage_ratio'
model = run_ols(aq_micro, 'l_wrat53_dum_skti_secall', 'l_y')
add_result(spec_id, 'methods/cross_sectional_ols.md#functional-form', 'l_wrat53_dum_skti_secall', 'l_y', model,
           'Micro sample', 'None', 'OLS')
print(f"2. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}")

# 3. Log labor supply ratio on log(GDP)
spec_id = 'ols/outcome/labor_supply'
model = run_ols(aq_micro, 'l_H5L3_dum_skti_hrs_secall', 'l_y')
add_result(spec_id, 'methods/cross_sectional_ols.md#functional-form', 'l_H5L3_dum_skti_hrs_secall', 'l_y', model,
           'Micro sample', 'None', 'OLS')
print(f"3. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}")

# ============================================================================
# PART 2: ELASTICITY OF SUBSTITUTION VARIATIONS (sigma)
# ============================================================================

print("\n" + "="*70)
print("PART 2: ELASTICITY OF SUBSTITUTION VARIATIONS")
print("="*70)

# 4-5. Different sigma values (1.3, 2.0 vs baseline 1.5)
for sigma_label, var_suffix in [('sigma_low', 'rl'), ('sigma_high', 'rh')]:
    spec_id = f'ols/param/{sigma_label}'
    outcome_var = f'l_irAQ53{var_suffix}_dum_skti_hrs_secall'
    if outcome_var in aq_micro.columns:
        model = run_ols(aq_micro, outcome_var, 'l_y')
        add_result(spec_id, 'robustness/model_specification.md', outcome_var, 'l_y', model,
                   f'Micro sample, {sigma_label}', 'None', 'OLS')
        print(f"4/5. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}")

# ============================================================================
# PART 3: ALTERNATIVE LABOR AGGREGATION
# ============================================================================

print("\n" + "="*70)
print("PART 3: ALTERNATIVE LABOR AGGREGATION METHODS")
print("="*70)

# 6. Bodies instead of hours (no hours weighting)
spec_id = 'ols/measurement/bodies'
model = run_ols(aq_micro, 'l_irAQ53_dum_skti_bod_secall', 'l_y')
add_result(spec_id, 'robustness/measurement.md', 'l_irAQ53_dum_skti_bod_secall', 'l_y', model,
           'Micro sample, bodies', 'None', 'OLS')
print(f"6. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}")

# 7. Population instead of employed
spec_id = 'ols/measurement/population'
model = run_ols(aq_micro, 'l_irAQ53_dum_skti_pop_secall', 'l_y')
add_result(spec_id, 'robustness/measurement.md', 'l_irAQ53_dum_skti_pop_secall', 'l_y', model,
           'Micro sample, population', 'None', 'OLS')
print(f"7. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}")

# 8. Mincerian returns assumption
spec_id = 'ols/measurement/mincerian'
model = run_ols(aq_micro, 'l_irAQ53_minc_skti_pop_secall', 'l_y')
add_result(spec_id, 'robustness/measurement.md', 'l_irAQ53_minc_skti_pop_secall', 'l_y', model,
           'Micro sample, mincerian', 'None', 'OLS')
print(f"8. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}")

# ============================================================================
# PART 4: EXPERIENCE AND GENDER CONTROLS
# ============================================================================

print("\n" + "="*70)
print("PART 4: EXPERIENCE AND GENDER CONTROLS")
print("="*70)

# 9. With experience and gender controls
spec_id = 'ols/controls/experience_gender'
model = run_ols(aq_micro, 'l_irAQ53_dumx_skti_hrs_secall', 'l_y')
add_result(spec_id, 'robustness/control_progression.md', 'l_irAQ53_dumx_skti_hrs_secall', 'l_y', model,
           'Micro sample, experience+gender controls', 'Experience and gender', 'OLS')
print(f"9. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}")

# 10-11. Experience/gender controls with different sigma
for sigma_label, var_suffix in [('sigma_low', 'rl'), ('sigma_high', 'rh')]:
    spec_id = f'ols/controls/expgen_{sigma_label}'
    outcome_var = f'l_irAQ53{var_suffix}_dumx_skti_hrs_secall'
    if outcome_var in aq_micro.columns:
        model = run_ols(aq_micro, outcome_var, 'l_y')
        add_result(spec_id, 'robustness/control_progression.md', outcome_var, 'l_y', model,
                   f'Micro sample, exp+gender, {sigma_label}', 'Experience, gender', 'OLS')
        print(f"10/11. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}")

# ============================================================================
# PART 5: SELF-EMPLOYMENT SAMPLE
# ============================================================================

print("\n" + "="*70)
print("PART 5: SELF-EMPLOYMENT SAMPLE")
print("="*70)

# Self-employment sample (if available)
aq_se = aq_micro[aq_micro['irAQ53_dumse_skti_hrs_secall'].notna()].copy()

if len(aq_se) > 3:
    # 12. Self-employment sample
    spec_id = 'ols/sample/self_employed'
    model = run_ols(aq_se, 'l_irAQ53_dumse_skti_hrs_secall', 'l_y')
    add_result(spec_id, 'robustness/sample_restrictions.md', 'l_irAQ53_dumse_skti_hrs_secall', 'l_y', model,
               'Self-employment sample', 'None', 'OLS')
    print(f"12. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}")

    # 13-14. Self-employment with different sigma
    for sigma_label, var_suffix in [('sigma_low', 'rl'), ('sigma_high', 'rh')]:
        spec_id = f'ols/sample/selfemp_{sigma_label}'
        outcome_var = f'l_irAQ53{var_suffix}_dumse_skti_hrs_secall'
        if outcome_var in aq_se.columns:
            model = run_ols(aq_se, outcome_var, 'l_y')
            add_result(spec_id, 'robustness/sample_restrictions.md', outcome_var, 'l_y', model,
                       f'Self-employment, {sigma_label}', 'None', 'OLS')
            print(f"13/14. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}")

# ============================================================================
# PART 6: SECTORAL ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("PART 6: SECTORAL ANALYSIS")
print("="*70)

sectors = {1: 'Agriculture', 2: 'Manufacturing', 3: 'LowSkillServices', 4: 'HighSkillServices'}

for sec_num, sec_name in sectors.items():
    # 15-22. By sector
    spec_id = f'ols/sector/{sec_name}'
    outcome_var = f'l_irAQ53_dum_skti_hrs_sec{sec_num}'
    if outcome_var in aq_micro.columns:
        model = run_ols(aq_micro, outcome_var, 'l_y')
        add_result(spec_id, 'robustness/heterogeneity.md', outcome_var, 'l_y', model,
                   f'Micro sample, sector {sec_name}', 'None', 'OLS')
        print(f"15-22. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}")

    # Wage ratio and labor supply by sector
    for var_type, var_prefix in [('wage_ratio', 'l_wrat53_dum_skti_sec'), ('labor_supply', 'l_H5L3_dum_skti_hrs_sec')]:
        spec_id = f'ols/sector/{sec_name}/{var_type}'
        outcome_var = f'{var_prefix}{sec_num}'
        if outcome_var in aq_micro.columns:
            model = run_ols(aq_micro, outcome_var, 'l_y')
            add_result(spec_id, 'robustness/heterogeneity.md', outcome_var, 'l_y', model,
                       f'Micro sample, {sec_name}, {var_type}', 'None', 'OLS')
            print(f"   {spec_id}: coef={model.params['l_y']:.4f}")

# ============================================================================
# PART 7: BARRO-LEE BROAD SAMPLE
# ============================================================================

print("\n" + "="*70)
print("PART 7: BARRO-LEE BROAD SAMPLE")
print("="*70)

# Use all countries with Barro-Lee data
aq_blee = aq_df[aq_df['irAQ53_blee_skti'].notna()].copy()

# 23. Barro-Lee broad sample
spec_id = 'ols/sample/barro_lee_broad'
model = run_ols(aq_blee, 'l_irAQ53_blee_skti', 'l_y')
add_result(spec_id, 'robustness/sample_restrictions.md', 'l_irAQ53_blee_skti', 'l_y', model,
           'Barro-Lee broad sample', 'None', 'OLS')
print(f"23. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}, n={int(model.nobs)}")

# 24-25. Barro-Lee with different sigma
for sigma_label, var_suffix in [('sigma_low', 'rl'), ('sigma_high', 'rh')]:
    spec_id = f'ols/sample/blee_{sigma_label}'
    outcome_var = f'l_irAQ53{var_suffix}_blee_skti'
    if outcome_var in aq_blee.columns:
        model = run_ols(aq_blee, outcome_var, 'l_y')
        add_result(spec_id, 'robustness/sample_restrictions.md', outcome_var, 'l_y', model,
                   f'Barro-Lee broad, {sigma_label}', 'None', 'OLS')
        print(f"24/25. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}")

# ============================================================================
# PART 8: Q (RELATIVE HUMAN CAPITAL) - US IMMIGRANTS SAMPLE
# ============================================================================

print("\n" + "="*70)
print("PART 8: Q ANALYSIS - US IMMIGRANTS SAMPLE")
print("="*70)

# Prepare US immigrant sample
q_us_clean = q_us.dropna(subset=['l_irQ53_dum', 'l_y']).copy()

# 26. Baseline Q regression
spec_id = 'ols/Q/baseline_us'
model = run_ols(q_us_clean, 'l_irQ53_dum', 'l_y')
add_result(spec_id, 'methods/cross_sectional_ols.md', 'l_irQ53_dum', 'l_y', model,
           'US immigrants sample', 'None', 'OLS')
print(f"26. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}, n={int(model.nobs)}")

# 27. US immigrants - micro sample only
q_us_micro = q_us_clean[q_us_clean['sample_micro'] == 1].copy()
if len(q_us_micro) > 3:
    spec_id = 'ols/Q/us_micro'
    model = run_ols(q_us_micro, 'l_irQ53_dum', 'l_y')
    add_result(spec_id, 'robustness/sample_restrictions.md', 'l_irQ53_dum', 'l_y', model,
               'US immigrants, micro sample', 'None', 'OLS')
    print(f"27. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}")

# ============================================================================
# PART 9: Q POOLED ACROSS HOST COUNTRIES
# ============================================================================

print("\n" + "="*70)
print("PART 9: Q POOLED ANALYSIS")
print("="*70)

# Get unique country observations for pooled analysis
q_unique = q_df[q_df['sample_migr'] == 1].drop_duplicates(subset=['country']).copy()

# 28. Pooled Q (no bilateral controls)
spec_id = 'ols/Q/pooled'
model = run_ols(q_unique, 'l_irQ53_pool_dum', 'l_y')
add_result(spec_id, 'methods/cross_sectional_ols.md', 'l_irQ53_pool_dum', 'l_y', model,
           'Pooled across host countries', 'None', 'OLS')
print(f"28. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}")

# 29. Pooled Q - micro sample only
q_unique_micro = q_unique[q_unique['sample_micro'] == 1].copy()
if len(q_unique_micro) > 3:
    spec_id = 'ols/Q/pooled_micro'
    model = run_ols(q_unique_micro, 'l_irQ53_pool_dum', 'l_y')
    add_result(spec_id, 'robustness/sample_restrictions.md', 'l_irQ53_pool_dum', 'l_y', model,
               'Pooled, micro sample', 'None', 'OLS')
    print(f"29. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}")

# ============================================================================
# PART 10: SELECTION ADJUSTMENT
# ============================================================================

print("\n" + "="*70)
print("PART 10: SELECTION ADJUSTMENT")
print("="*70)

# 30. Selection-adjusted Q
if 'l_irQ53sel_dum' in q_us_clean.columns:
    q_us_sel = q_us_clean.dropna(subset=['l_irQ53sel_dum'])
    if len(q_us_sel) > 3:
        spec_id = 'ols/Q/selection_adjusted'
        model = run_ols(q_us_sel, 'l_irQ53sel_dum', 'l_y')
        add_result(spec_id, 'robustness/measurement.md', 'l_irQ53sel_dum', 'l_y', model,
                   'US immigrants, selection adjusted', 'None', 'OLS')
        print(f"30. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}")

# ============================================================================
# PART 11: 10+ YEARS IN US
# ============================================================================

print("\n" + "="*70)
print("PART 11: LONG-TERM IMMIGRANTS (10+ YEARS)")
print("="*70)

# 31. 10+ years in US
if 'l_irQ53yh_dum' in q_us_clean.columns:
    q_us_yh = q_us_clean.dropna(subset=['l_irQ53yh_dum'])
    if len(q_us_yh) > 3:
        spec_id = 'ols/Q/long_term'
        model = run_ols(q_us_yh, 'l_irQ53yh_dum', 'l_y')
        add_result(spec_id, 'robustness/sample_restrictions.md', 'l_irQ53yh_dum', 'l_y', model,
                   'US immigrants, 10+ years', 'None', 'OLS')
        print(f"31. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}")

# ============================================================================
# PART 12: GOOD ENGLISH SPEAKERS
# ============================================================================

print("\n" + "="*70)
print("PART 12: GOOD ENGLISH SPEAKERS")
print("="*70)

# 32. Good English speakers only
if 'l_irQ53goodeng_dum' in q_us_clean.columns:
    q_us_eng = q_us_clean.dropna(subset=['l_irQ53goodeng_dum'])
    if len(q_us_eng) > 3:
        spec_id = 'ols/Q/good_english'
        model = run_ols(q_us_eng, 'l_irQ53goodeng_dum', 'l_y')
        add_result(spec_id, 'robustness/sample_restrictions.md', 'l_irQ53goodeng_dum', 'l_y', model,
                   'US immigrants, good English', 'None', 'OLS')
        print(f"32. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}")

# ============================================================================
# PART 13: SKILL DOWNGRADING
# ============================================================================

print("\n" + "="*70)
print("PART 13: SKILL DOWNGRADING")
print("="*70)

# 33. No skill downgrading
if 'l_irQ53nodown_dum' in q_us_clean.columns:
    q_us_nd = q_us_clean.dropna(subset=['l_irQ53nodown_dum'])
    if len(q_us_nd) > 3:
        spec_id = 'ols/Q/no_downgrading'
        model = run_ols(q_us_nd, 'l_irQ53nodown_dum', 'l_y')
        add_result(spec_id, 'robustness/measurement.md', 'l_irQ53nodown_dum', 'l_y', model,
                   'US immigrants, no skill downgrading', 'None', 'OLS')
        print(f"33. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}")

# 34. No skill mismatch
if 'l_irQ53nomism_dum' in q_us_clean.columns:
    q_us_nm = q_us_clean.dropna(subset=['l_irQ53nomism_dum'])
    if len(q_us_nm) > 3:
        spec_id = 'ols/Q/no_mismatch'
        model = run_ols(q_us_nm, 'l_irQ53nomism_dum', 'l_y')
        add_result(spec_id, 'robustness/measurement.md', 'l_irQ53nomism_dum', 'l_y', model,
                   'US immigrants, no mismatch', 'None', 'OLS')
        print(f"34. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}")

# ============================================================================
# PART 14: SORTING CORRECTIONS
# ============================================================================

print("\n" + "="*70)
print("PART 14: SORTING CORRECTIONS")
print("="*70)

# 35. Sorting - sectors
if 'l_irQ53sorts_dum' in q_us_clean.columns:
    q_us_ss = q_us_clean.dropna(subset=['l_irQ53sorts_dum'])
    if len(q_us_ss) > 3:
        spec_id = 'ols/Q/sorting_sectors'
        model = run_ols(q_us_ss, 'l_irQ53sorts_dum', 'l_y')
        add_result(spec_id, 'robustness/measurement.md', 'l_irQ53sorts_dum', 'l_y', model,
                   'US immigrants, sorting (sectors)', 'None', 'OLS')
        print(f"35. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}")

# 36. Sorting - geographic
if 'l_irQ53sortr_dum' in q_us_clean.columns:
    q_us_sr = q_us_clean.dropna(subset=['l_irQ53sortr_dum'])
    if len(q_us_sr) > 3:
        spec_id = 'ols/Q/sorting_geographic'
        model = run_ols(q_us_sr, 'l_irQ53sortr_dum', 'l_y')
        add_result(spec_id, 'robustness/measurement.md', 'l_irQ53sortr_dum', 'l_y', model,
                   'US immigrants, sorting (geographic)', 'None', 'OLS')
        print(f"36. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}")

# ============================================================================
# PART 15: INFERENCE VARIATIONS
# ============================================================================

print("\n" + "="*70)
print("PART 15: INFERENCE VARIATIONS")
print("="*70)

# 37. Classical (non-robust) standard errors
spec_id = 'ols/se/classical'
model = run_ols(aq_micro, 'l_irAQ53_dum_skti_hrs_secall', 'l_y', robust=False)
add_result(spec_id, 'robustness/clustering_variations.md', 'l_irAQ53_dum_skti_hrs_secall', 'l_y', model,
           'Micro sample, classical SE', 'None', 'OLS')
print(f"37. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}")

# 38. Bootstrap standard errors
from sklearn.utils import resample
def bootstrap_se(df, outcome, treatment, n_bootstrap=500):
    coefs = []
    for _ in range(n_bootstrap):
        sample = resample(df)
        try:
            model = smf.ols(f'{outcome} ~ {treatment}', data=sample).fit()
            coefs.append(model.params[treatment])
        except:
            pass
    return np.std(coefs)

boot_se = bootstrap_se(aq_micro.dropna(subset=['l_irAQ53_dum_skti_hrs_secall', 'l_y']),
                       'l_irAQ53_dum_skti_hrs_secall', 'l_y')
spec_id = 'ols/se/bootstrap'
model = run_ols(aq_micro, 'l_irAQ53_dum_skti_hrs_secall', 'l_y')
# Override SE with bootstrap
results.append({
    'paper_id': PAPER_ID,
    'journal': JOURNAL,
    'paper_title': PAPER_TITLE,
    'spec_id': spec_id,
    'spec_tree_path': 'robustness/clustering_variations.md',
    'outcome_var': 'l_irAQ53_dum_skti_hrs_secall',
    'treatment_var': 'l_y',
    'coefficient': model.params['l_y'],
    'std_error': boot_se,
    't_stat': model.params['l_y'] / boot_se,
    'p_value': 2 * (1 - stats.norm.cdf(abs(model.params['l_y'] / boot_se))),
    'ci_lower': model.params['l_y'] - 1.96 * boot_se,
    'ci_upper': model.params['l_y'] + 1.96 * boot_se,
    'n_obs': int(model.nobs),
    'r_squared': model.rsquared,
    'coefficient_vector_json': json.dumps({"treatment": {"var": "l_y", "coef": float(model.params['l_y']), "se": float(boot_se)}}),
    'sample_desc': 'Micro sample, bootstrap SE',
    'fixed_effects': 'None',
    'controls_desc': 'None',
    'cluster_var': 'None',
    'model_type': 'OLS',
    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
})
print(f"38. {spec_id}: coef={model.params['l_y']:.4f}, boot_se={boot_se:.4f}")

# ============================================================================
# PART 16: SAMPLE RESTRICTIONS
# ============================================================================

print("\n" + "="*70)
print("PART 16: SAMPLE RESTRICTIONS")
print("="*70)

# 39-50. Leave-one-out (drop each country)
countries_in_sample = aq_micro['country'].unique()
for i, country in enumerate(countries_in_sample):
    spec_id = f'robust/loo/drop_{country.replace(" ", "_")}'
    aq_loo = aq_micro[aq_micro['country'] != country].copy()
    if len(aq_loo) > 3:
        model = run_ols(aq_loo, 'l_irAQ53_dum_skti_hrs_secall', 'l_y')
        add_result(spec_id, 'robustness/leave_one_out.md', 'l_irAQ53_dum_skti_hrs_secall', 'l_y', model,
                   f'Micro sample, drop {country}', 'None', 'OLS')
        print(f"39-50. {spec_id}: coef={model.params['l_y']:.4f}, n={int(model.nobs)}")

# ============================================================================
# PART 17: OUTLIER HANDLING
# ============================================================================

print("\n" + "="*70)
print("PART 17: OUTLIER HANDLING")
print("="*70)

# Use Q data for larger sample
q_full = q_us_clean.copy()

# 51. Winsorize at 5%
spec_id = 'robust/sample/winsorize_5pct'
q_wins = q_full.copy()
q_wins['l_irQ53_dum'] = q_wins['l_irQ53_dum'].clip(
    lower=q_wins['l_irQ53_dum'].quantile(0.05),
    upper=q_wins['l_irQ53_dum'].quantile(0.95)
)
if len(q_wins.dropna(subset=['l_irQ53_dum', 'l_y'])) > 3:
    model = run_ols(q_wins, 'l_irQ53_dum', 'l_y')
    add_result(spec_id, 'robustness/sample_restrictions.md', 'l_irQ53_dum', 'l_y', model,
               'US immigrants, winsorized 5%', 'None', 'OLS')
    print(f"51. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}")

# 52. Winsorize at 10%
spec_id = 'robust/sample/winsorize_10pct'
q_wins = q_full.copy()
q_wins['l_irQ53_dum'] = q_wins['l_irQ53_dum'].clip(
    lower=q_wins['l_irQ53_dum'].quantile(0.10),
    upper=q_wins['l_irQ53_dum'].quantile(0.90)
)
if len(q_wins.dropna(subset=['l_irQ53_dum', 'l_y'])) > 3:
    model = run_ols(q_wins, 'l_irQ53_dum', 'l_y')
    add_result(spec_id, 'robustness/sample_restrictions.md', 'l_irQ53_dum', 'l_y', model,
               'US immigrants, winsorized 10%', 'None', 'OLS')
    print(f"52. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}")

# 53. Trim extreme values
spec_id = 'robust/sample/trim_5pct'
q_trim = q_full[
    (q_full['l_irQ53_dum'] > q_full['l_irQ53_dum'].quantile(0.05)) &
    (q_full['l_irQ53_dum'] < q_full['l_irQ53_dum'].quantile(0.95))
].copy()
if len(q_trim.dropna(subset=['l_irQ53_dum', 'l_y'])) > 3:
    model = run_ols(q_trim, 'l_irQ53_dum', 'l_y')
    add_result(spec_id, 'robustness/sample_restrictions.md', 'l_irQ53_dum', 'l_y', model,
               'US immigrants, trimmed 5%', 'None', 'OLS')
    print(f"53. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}")

# ============================================================================
# PART 18: INCOME GROUP ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("PART 18: INCOME GROUP ANALYSIS")
print("="*70)

# Create income groups based on y_relUS
if 'y_relUS_2005' in q_full.columns:
    # 54. High income countries (above median)
    median_income = q_full['y_relUS_2005'].median()
    q_high = q_full[q_full['y_relUS_2005'] >= median_income].copy()
    if len(q_high.dropna(subset=['l_irQ53_dum', 'l_y'])) > 3:
        spec_id = 'robust/sample/high_income'
        model = run_ols(q_high, 'l_irQ53_dum', 'l_y')
        add_result(spec_id, 'robustness/sample_restrictions.md', 'l_irQ53_dum', 'l_y', model,
                   'US immigrants, high income', 'None', 'OLS')
        print(f"54. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}")

    # 55. Low income countries
    q_low = q_full[q_full['y_relUS_2005'] < median_income].copy()
    if len(q_low.dropna(subset=['l_irQ53_dum', 'l_y'])) > 3:
        spec_id = 'robust/sample/low_income'
        model = run_ols(q_low, 'l_irQ53_dum', 'l_y')
        add_result(spec_id, 'robustness/sample_restrictions.md', 'l_irQ53_dum', 'l_y', model,
                   'US immigrants, low income', 'None', 'OLS')
        print(f"55. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}")

# ============================================================================
# PART 19: BILATERAL CONTROLS
# ============================================================================

print("\n" + "="*70)
print("PART 19: BILATERAL CONTROLS (for Q analysis)")
print("="*70)

# Prepare bilateral controls
if 'geo_dist' in q_us_clean.columns and 'contig' in q_us_clean.columns:
    q_bilat = q_us_clean.dropna(subset=['geo_dist', 'contig', 'comlang_off']).copy()
    q_bilat['geo_dist2'] = q_bilat['geo_dist'] ** 2
    q_bilat['geo_dist3'] = q_bilat['geo_dist'] ** 3

    # 56. With distance control
    if len(q_bilat) > 5:
        spec_id = 'ols/Q/distance_control'
        model = run_ols(q_bilat, 'l_irQ53_dum', 'l_y', controls=['geo_dist'])
        add_result(spec_id, 'robustness/control_progression.md', 'l_irQ53_dum', 'l_y', model,
                   'US immigrants', 'Distance', 'OLS')
        print(f"56. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}")

    # 57. With full bilateral controls
    if len(q_bilat) > 5:
        spec_id = 'ols/Q/bilateral_full'
        model = run_ols(q_bilat, 'l_irQ53_dum', 'l_y',
                        controls=['geo_dist', 'geo_dist2', 'geo_dist3', 'contig', 'comlang_off'])
        add_result(spec_id, 'robustness/control_progression.md', 'l_irQ53_dum', 'l_y', model,
                   'US immigrants', 'Distance, contiguity, language', 'OLS')
        print(f"57. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}")

    # 58. Contiguous only
    if len(q_bilat) > 5:
        spec_id = 'ols/Q/contig_control'
        model = run_ols(q_bilat, 'l_irQ53_dum', 'l_y', controls=['contig'])
        add_result(spec_id, 'robustness/control_progression.md', 'l_irQ53_dum', 'l_y', model,
                   'US immigrants', 'Contiguity', 'OLS')
        print(f"58. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}")

    # 59. Common language only
    if len(q_bilat) > 5:
        spec_id = 'ols/Q/language_control'
        model = run_ols(q_bilat, 'l_irQ53_dum', 'l_y', controls=['comlang_off'])
        add_result(spec_id, 'robustness/control_progression.md', 'l_irQ53_dum', 'l_y', model,
                   'US immigrants', 'Common language', 'OLS')
        print(f"59. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}")

# ============================================================================
# PART 20: FUNCTIONAL FORM VARIATIONS
# ============================================================================

print("\n" + "="*70)
print("PART 20: FUNCTIONAL FORM VARIATIONS")
print("="*70)

# 60. Levels instead of logs for AQ
spec_id = 'ols/form/levels_AQ'
model = run_ols(aq_micro, 'irAQ53_dum_skti_hrs_secall', 'l_y')
add_result(spec_id, 'robustness/functional_form.md', 'irAQ53_dum_skti_hrs_secall', 'l_y', model,
           'Micro sample, levels AQ', 'None', 'OLS')
print(f"60. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}")

# 61. Levels for Q
spec_id = 'ols/form/levels_Q'
model = run_ols(q_us_clean, 'irQ53_dum', 'l_y')
add_result(spec_id, 'robustness/functional_form.md', 'irQ53_dum', 'l_y', model,
           'US immigrants, levels Q', 'None', 'OLS')
print(f"61. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}")

# 62. Quadratic in GDP
spec_id = 'ols/form/quadratic_gdp'
aq_quad = aq_micro.copy()
aq_quad['l_y_sq'] = aq_quad['l_y'] ** 2
model = smf.ols('l_irAQ53_dum_skti_hrs_secall ~ l_y + l_y_sq', data=aq_quad).fit(cov_type='HC1')
add_result(spec_id, 'robustness/functional_form.md', 'l_irAQ53_dum_skti_hrs_secall', 'l_y', model,
           'Micro sample, quadratic', 'Quadratic GDP', 'OLS')
print(f"62. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}")

# ============================================================================
# PART 21: QUANTILE REGRESSION
# ============================================================================

print("\n" + "="*70)
print("PART 21: QUANTILE REGRESSION")
print("="*70)

from statsmodels.regression.quantile_regression import QuantReg

q_data = q_us_clean.dropna(subset=['l_irQ53_dum', 'l_y']).copy()
q_data['const'] = 1

for q, name in [(0.25, '25pct'), (0.50, 'median'), (0.75, '75pct')]:
    spec_id = f'ols/method/quantile_{name}'
    try:
        model = QuantReg(q_data['l_irQ53_dum'], q_data[['const', 'l_y']]).fit(q=q)
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': 'methods/cross_sectional_ols.md#estimation-method',
            'outcome_var': 'l_irQ53_dum',
            'treatment_var': 'l_y',
            'coefficient': model.params['l_y'],
            'std_error': model.bse['l_y'],
            't_stat': model.tvalues['l_y'],
            'p_value': model.pvalues['l_y'],
            'ci_lower': model.params['l_y'] - 1.96 * model.bse['l_y'],
            'ci_upper': model.params['l_y'] + 1.96 * model.bse['l_y'],
            'n_obs': len(q_data),
            'r_squared': np.nan,
            'coefficient_vector_json': json.dumps({"treatment": {"var": "l_y", "coef": float(model.params['l_y']), "se": float(model.bse['l_y'])}}),
            'sample_desc': f'US immigrants, {name} quantile',
            'fixed_effects': 'None',
            'controls_desc': 'None',
            'cluster_var': 'None',
            'model_type': f'Quantile ({name})',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })
        print(f"63-65. {spec_id}: coef={model.params['l_y']:.4f}, se={model.bse['l_y']:.4f}")
    except Exception as e:
        print(f"   Error in {spec_id}: {e}")

# ============================================================================
# PART 22: COMPARISON ACROSS AQ MEASURES
# ============================================================================

print("\n" + "="*70)
print("PART 22: AQ MEASURE COMPARISONS")
print("="*70)

# For each measure type, compare elasticity with GDP
aq_measures = [
    ('l_irAQ53_dum_skti_hrs_secall', 'Baseline AQ'),
    ('l_irAQ53_blee_skti', 'Barro-Lee AQ'),
]

for var, desc in aq_measures:
    if var in q_us_clean.columns:
        q_comp = q_us_clean.dropna(subset=[var, 'l_y'])
        if len(q_comp) > 3:
            spec_id = f'ols/compare/{desc.replace(" ", "_").lower()}'
            model = run_ols(q_comp, var, 'l_y')
            add_result(spec_id, 'robustness/model_specification.md', var, 'l_y', model,
                       f'US sample, {desc}', 'None', 'OLS')
            print(f"66-67. {spec_id}: coef={model.params['l_y']:.4f}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
output_path = f'{BASE_PATH}/specification_results.csv'
results_df.to_csv(output_path, index=False)

print(f"\nTotal specifications run: {len(results_df)}")
print(f"Results saved to: {output_path}")

# Summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

# Filter for valid coefficients
valid_results = results_df[results_df['coefficient'].notna()]

print(f"Valid specifications: {len(valid_results)}")
print(f"Positive coefficients: {(valid_results['coefficient'] > 0).sum()} ({100*(valid_results['coefficient'] > 0).mean():.1f}%)")
print(f"Significant at 5%: {(valid_results['p_value'] < 0.05).sum()} ({100*(valid_results['p_value'] < 0.05).mean():.1f}%)")
print(f"Significant at 1%: {(valid_results['p_value'] < 0.01).sum()} ({100*(valid_results['p_value'] < 0.01).mean():.1f}%)")
print(f"\nCoefficient statistics:")
print(f"  Median: {valid_results['coefficient'].median():.4f}")
print(f"  Mean: {valid_results['coefficient'].mean():.4f}")
print(f"  Std Dev: {valid_results['coefficient'].std():.4f}")
print(f"  Min: {valid_results['coefficient'].min():.4f}")
print(f"  Max: {valid_results['coefficient'].max():.4f}")
