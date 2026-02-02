#!/usr/bin/env python3
"""
Specification Search: 215802-V1
Paper: "Long-Run Impacts of Childhood Access to the Safety Net"
Authors: Hoynes, Schanzenbach & Almond
Journal: AER

Method: Panel Fixed Effects / Cross-Sectional OLS (hybrid approach)

Note: The original paper uses county-level Food Stamp Program rollout timing
interacted with birth cohorts to identify effects. The county identifiers
are stripped from the public PSID data (require restricted data application).

IMPORTANT DATA LIMITATIONS:
1. County identifiers are missing (require restricted PSID data)
2. Treatment assignment variables (shareFSPageIU_5, etc.) are 100% missing
3. The health outcome (healthy1986) is TIME-INVARIANT - it doesn't change within
   individuals, so individual FE regressions will fail to identify effects.

This analysis uses available variables with appropriate specifications:
- CROSS-SECTIONAL analysis for time-invariant outcomes (health status)
- PANEL FE analysis for time-varying outcomes (income, work limitation, hospitalization)
- Treatment: Current food stamp receipt (fsyn) and AFDC (afdcyn)
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import warnings
warnings.filterwarnings('ignore')

# Configuration
PAPER_ID = "215802-V1"
JOURNAL = "AER"
PAPER_TITLE = "Long-Run Impacts of Childhood Access to the Safety Net"
BASE_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search"
DATA_PATH = f"{BASE_PATH}/data/downloads/extracted/{PAPER_ID}"

# Load data
print("Loading data...")
df = pd.read_stata(f"{DATA_PATH}/psidAdultHealth.dta")

# ============================================================================
# DATA PREPARATION
# ============================================================================

print("Preparing data...")

# Create individual ID for panel
df['individual_id'] = df['inum1968'].astype(str) + '_' + df['person1968'].astype(str)

# Create binary health outcome (good health = 1)
# healthy1986: 0=excellent, 1=very good, 2=good, 3=fair, 4=poor
df['good_health'] = (df['healthy1986'] <= 2).astype(float)
df.loc[df['healthy1986'].isin([5, 9]), 'good_health'] = np.nan

# Create continuous health scale (inverted: higher = better)
df['health_scale'] = 4 - df['healthy1986']
df.loc[df['healthy1986'].isin([5, 9]), 'health_scale'] = np.nan

# Excellent health (0-1 only)
df['excellent_health'] = (df['healthy1986'] <= 1).astype(float)
df.loc[df['healthy1986'].isin([5, 9]), 'excellent_health'] = np.nan

# Poor health (3-4)
df['poor_health'] = (df['healthy1986'] >= 3).astype(float)
df.loc[df['healthy1986'].isin([5, 9]), 'poor_health'] = np.nan

# Work limitation binary (1 = has limitation)
df['work_limited'] = (df['worklimit'] == 1).astype(float)
df.loc[df['worklimit'].isin([8, 9]), 'work_limited'] = np.nan

# Create employed binary
df['employed'] = (df['empstat'] == 'Working now').astype(float)
df.loc[df['empstat'].isna(), 'employed'] = np.nan

# Create demographic controls
df['female'] = (df['sex'] == 'Female').astype(float)
df['black'] = (df['race'] == 'Black').astype(float)
df.loc[df['race'].isna(), 'black'] = np.nan
df['white'] = (df['race'] == 'White').astype(float)
df.loc[df['race'].isna(), 'white'] = np.nan

df['married'] = (df['marstat'] == 'Married').astype(float)
df.loc[df['marstat'].isna(), 'married'] = np.nan

df['age_sq'] = df['age'] ** 2

# Log income (handle zeros and negatives)
df['log_income'] = np.log(df['totInc'].clip(lower=1))

# Education years (cap at 17)
df['educ_years'] = df['educ'].clip(upper=17)

# Create treatment variables
# Main treatment: current food stamp receipt
df['fs_receipt'] = df['fsyn'].fillna(0)

# Also create AFDC receipt
df['afdc_receipt'] = df['afdcyn'].fillna(0)

# Combined safety net
df['any_welfare'] = ((df['fs_receipt'] == 1) | (df['afdc_receipt'] == 1)).astype(float)

# Birth cohort indicators
df['cohort_early'] = (df['yob'] <= 1965).astype(float)  # Born before FSP expansion
df['cohort_late'] = (df['yob'] > 1970).astype(float)   # Born after FSP fully implemented

# Hospital utilization
df['any_hospital'] = (df['hospdays'] > 0).astype(float)
df.loc[df['hospdays'].isna(), 'any_hospital'] = np.nan

# Log hospital days
df['log_hospdays'] = np.log(df['hospdays'].clip(lower=1))
df.loc[df['hospdays'].isna(), 'log_hospdays'] = np.nan

# ============================================================================
# CREATE ANALYSIS SAMPLES
# ============================================================================

# Full panel sample
analysis_df = df.dropna(subset=['good_health', 'individual_id', 'Datayear', 'age']).copy()
print(f"Full panel sample: {len(analysis_df)} observations, {analysis_df['individual_id'].nunique()} individuals")

# Cross-sectional sample (first observation per person for time-invariant analysis)
cs_df = df.sort_values('Datayear').groupby('individual_id').first().reset_index()
cs_df = cs_df.dropna(subset=['good_health', 'age'])
print(f"Cross-sectional sample: {len(cs_df)} individuals")

# Panel sample with time-varying outcomes
panel_df = df.dropna(subset=['log_income', 'individual_id', 'Datayear', 'age']).copy()
print(f"Panel sample (income): {len(panel_df)} observations")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_results(model, treatment_var, spec_id, spec_tree_path, outcome_var,
                   controls_desc, fixed_effects, cluster_var, model_type, sample_desc):
    """Extract results from pyfixest model."""
    try:
        coef = model.coef()[treatment_var]
        se = model.se()[treatment_var]
        tstat = model.tstat()[treatment_var]
        pval = model.pvalue()[treatment_var]
        ci = model.confint()
        ci_lower = ci.loc[treatment_var, '2.5%']
        ci_upper = ci.loc[treatment_var, '97.5%']
        n_obs = model._N
        r2 = model._r2

        # Build coefficient vector
        coef_vector = {
            "treatment": {
                "var": treatment_var,
                "coef": float(coef),
                "se": float(se),
                "pval": float(pval)
            },
            "controls": [],
            "fixed_effects_absorbed": fixed_effects.split(' + ') if fixed_effects else [],
            "diagnostics": {}
        }

        # Add control coefficients
        for var in model.coef().index:
            if var != treatment_var and var != 'Intercept':
                coef_vector["controls"].append({
                    "var": var,
                    "coef": float(model.coef()[var]),
                    "se": float(model.se()[var]),
                    "pval": float(model.pvalue()[var])
                })

        return {
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
            'n_obs': int(n_obs),
            'r_squared': float(r2),
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects,
            'controls_desc': controls_desc,
            'cluster_var': cluster_var,
            'model_type': model_type,
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
    except Exception as e:
        print(f"Error extracting results for {spec_id}: {e}")
        return None

def run_spec(data, formula, treatment_var, spec_id, spec_tree_path, outcome_var,
             controls_desc, fixed_effects, cluster_var, model_type, sample_desc, vcov=None):
    """Run a single specification and extract results."""
    try:
        if vcov:
            model = pf.feols(formula, data=data, vcov=vcov)
        else:
            model = pf.feols(formula, data=data)
        return extract_results(model, treatment_var, spec_id, spec_tree_path,
                              outcome_var, controls_desc, fixed_effects,
                              cluster_var, model_type, sample_desc)
    except Exception as e:
        print(f"Error in {spec_id}: {e}")
        return None

# ============================================================================
# SPECIFICATION SEARCH
# ============================================================================

results = []

# Define control sets
basic_controls = ['age', 'age_sq', 'female']
demo_controls = basic_controls + ['black', 'educ_years', 'married']
full_controls = demo_controls + ['famsize', 'kids']

basic_controls_str = ' + '.join(basic_controls)
demo_controls_str = ' + '.join(demo_controls)
full_controls_str = ' + '.join(full_controls)

# Controls without female (for panel with individual FE)
panel_controls = ['age', 'age_sq', 'married']
panel_controls_str = ' + '.join(panel_controls)

print("\n" + "="*70)
print("RUNNING SPECIFICATION SEARCH")
print("="*70)

# ============================================================================
# PART 1: CROSS-SECTIONAL ANALYSIS (Time-invariant outcomes like health)
# ============================================================================
print("\n" + "-"*70)
print("PART 1: CROSS-SECTIONAL SPECIFICATIONS (health outcomes)")
print("-"*70)

# 1A. BASELINE CROSS-SECTIONAL SPECIFICATIONS (5 specs)
print("\n--- 1A. Baseline Cross-Sectional ---")

# Baseline: FS receipt on good health (cross-sectional)
res = run_spec(
    cs_df,
    f"good_health ~ fs_receipt + {demo_controls_str}",
    'fs_receipt', 'baseline', 'methods/cross_sectional_ols.md',
    'good_health', 'Demographics', 'none', 'yob',
    'Cross-sectional OLS', 'First observation per person',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# With year FE
res = run_spec(
    cs_df,
    f"good_health ~ fs_receipt + {demo_controls_str} | Datayear",
    'fs_receipt', 'baseline_year_fe', 'methods/cross_sectional_ols.md',
    'good_health', 'Demographics', 'year', 'yob',
    'Cross-sectional OLS', 'First observation per person',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# With cohort FE
res = run_spec(
    cs_df,
    f"good_health ~ fs_receipt + {demo_controls_str} | yob",
    'fs_receipt', 'baseline_cohort_fe', 'methods/cross_sectional_ols.md',
    'good_health', 'Demographics', 'cohort', 'yob',
    'Cross-sectional OLS', 'First observation per person',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# Health scale outcome
res = run_spec(
    cs_df,
    f"health_scale ~ fs_receipt + {demo_controls_str}",
    'fs_receipt', 'baseline_health_scale', 'methods/cross_sectional_ols.md',
    'health_scale', 'Demographics', 'none', 'yob',
    'Cross-sectional OLS', 'First observation per person',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# Any welfare treatment
res = run_spec(
    cs_df,
    f"good_health ~ any_welfare + {demo_controls_str}",
    'any_welfare', 'baseline_any_welfare', 'methods/cross_sectional_ols.md',
    'good_health', 'Demographics', 'none', 'yob',
    'Cross-sectional OLS', 'First observation per person',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# 1B. CONTROL PROGRESSION - CROSS-SECTIONAL (10 specs)
print("\n--- 1B. Control Progression (Cross-sectional) ---")

# Bivariate
res = run_spec(
    cs_df,
    f"good_health ~ fs_receipt",
    'fs_receipt', 'robust/build/bivariate', 'robustness/control_progression.md',
    'good_health', 'None', 'none', 'yob',
    'Cross-sectional OLS', 'First observation per person',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# Add age
res = run_spec(
    cs_df,
    f"good_health ~ fs_receipt + age + age_sq",
    'fs_receipt', 'robust/build/add_age', 'robustness/control_progression.md',
    'good_health', 'Age only', 'none', 'yob',
    'Cross-sectional OLS', 'First observation per person',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# Add gender
res = run_spec(
    cs_df,
    f"good_health ~ fs_receipt + {basic_controls_str}",
    'fs_receipt', 'robust/build/add_gender', 'robustness/control_progression.md',
    'good_health', 'Age + gender', 'none', 'yob',
    'Cross-sectional OLS', 'First observation per person',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# Add race
cs_race = cs_df.dropna(subset=['black'])
res = run_spec(
    cs_race,
    f"good_health ~ fs_receipt + {basic_controls_str} + black",
    'fs_receipt', 'robust/build/add_race', 'robustness/control_progression.md',
    'good_health', 'Age + gender + race', 'none', 'yob',
    'Cross-sectional OLS', 'Non-missing race',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# Add education
cs_educ = cs_df.dropna(subset=['educ_years'])
res = run_spec(
    cs_educ,
    f"good_health ~ fs_receipt + {basic_controls_str} + black + educ_years",
    'fs_receipt', 'robust/build/add_education', 'robustness/control_progression.md',
    'good_health', 'Demographics', 'none', 'yob',
    'Cross-sectional OLS', 'Non-missing education',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# Add marital
cs_full = cs_df.dropna(subset=['married', 'educ_years', 'black'])
res = run_spec(
    cs_full,
    f"good_health ~ fs_receipt + {demo_controls_str}",
    'fs_receipt', 'robust/build/add_marital', 'robustness/control_progression.md',
    'good_health', 'Full demographics', 'none', 'yob',
    'Cross-sectional OLS', 'Non-missing demographics',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# Add household
cs_hh = cs_df.dropna(subset=['married', 'educ_years', 'black', 'famsize', 'kids'])
res = run_spec(
    cs_hh,
    f"good_health ~ fs_receipt + {full_controls_str}",
    'fs_receipt', 'robust/build/full', 'robustness/control_progression.md',
    'good_health', 'Full controls', 'none', 'yob',
    'Cross-sectional OLS', 'Complete cases',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# Kitchen sink
cs_ks = cs_df.dropna(subset=['married', 'educ_years', 'black', 'famsize', 'kids', 'log_income'])
res = run_spec(
    cs_ks,
    f"good_health ~ fs_receipt + {full_controls_str} + log_income",
    'fs_receipt', 'robust/build/kitchen_sink', 'robustness/control_progression.md',
    'good_health', 'Kitchen sink', 'none', 'yob',
    'Cross-sectional OLS', 'Kitchen sink sample',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# Leave-one-out: drop age
res = run_spec(
    cs_full,
    f"good_health ~ fs_receipt + female + black + educ_years + married",
    'fs_receipt', 'robust/loo/drop_age', 'robustness/leave_one_out.md',
    'good_health', 'No age', 'none', 'yob',
    'Cross-sectional OLS', 'Non-missing demographics',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# Leave-one-out: drop education
res = run_spec(
    cs_full,
    f"good_health ~ fs_receipt + age + age_sq + female + black + married",
    'fs_receipt', 'robust/loo/drop_educ', 'robustness/leave_one_out.md',
    'good_health', 'No education', 'none', 'yob',
    'Cross-sectional OLS', 'Non-missing demographics',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# 1C. CLUSTERING VARIATIONS - CROSS-SECTIONAL (6 specs)
print("\n--- 1C. Clustering Variations (Cross-sectional) ---")

# Robust SE (no clustering)
res = run_spec(
    cs_df,
    f"good_health ~ fs_receipt + {demo_controls_str}",
    'fs_receipt', 'robust/cluster/robust_hetero', 'robustness/clustering_variations.md',
    'good_health', 'Demographics', 'none', 'none (robust)',
    'Cross-sectional OLS', 'First observation per person',
    vcov='hetero'
)
if res: results.append(res)

# Cluster by household (inum1968)
res = run_spec(
    cs_df,
    f"good_health ~ fs_receipt + {demo_controls_str}",
    'fs_receipt', 'robust/cluster/household', 'robustness/clustering_variations.md',
    'good_health', 'Demographics', 'none', 'inum1968',
    'Cross-sectional OLS', 'First observation per person',
    vcov={'CRV1': 'inum1968'}
)
if res: results.append(res)

# Cluster by Datayear
res = run_spec(
    cs_df,
    f"good_health ~ fs_receipt + {demo_controls_str}",
    'fs_receipt', 'robust/cluster/year', 'robustness/clustering_variations.md',
    'good_health', 'Demographics', 'none', 'Datayear',
    'Cross-sectional OLS', 'First observation per person',
    vcov={'CRV1': 'Datayear'}
)
if res: results.append(res)

# Cluster by race
race_df = cs_df.dropna(subset=['race'])
res = run_spec(
    race_df,
    f"good_health ~ fs_receipt + {demo_controls_str}",
    'fs_receipt', 'robust/cluster/race', 'robustness/clustering_variations.md',
    'good_health', 'Demographics', 'none', 'race',
    'Cross-sectional OLS', 'Non-missing race',
    vcov={'CRV1': 'race'}
)
if res: results.append(res)

# Cluster by sex
res = run_spec(
    cs_df,
    f"good_health ~ fs_receipt + {demo_controls_str}",
    'fs_receipt', 'robust/cluster/sex', 'robustness/clustering_variations.md',
    'good_health', 'Demographics', 'none', 'sex',
    'Cross-sectional OLS', 'First observation per person',
    vcov={'CRV1': 'sex'}
)
if res: results.append(res)

# HC3 (small sample)
res = run_spec(
    cs_df,
    f"good_health ~ fs_receipt + {demo_controls_str}",
    'fs_receipt', 'robust/se/hc3', 'robustness/clustering_variations.md',
    'good_health', 'Demographics', 'none', 'none (HC3)',
    'Cross-sectional OLS', 'First observation per person',
    vcov='HC3'
)
if res: results.append(res)

# 1D. SAMPLE RESTRICTIONS - CROSS-SECTIONAL (10 specs)
print("\n--- 1D. Sample Restrictions (Cross-sectional) ---")

# Male only
male_cs = cs_df[cs_df['female'] == 0]
res = run_spec(
    male_cs,
    f"good_health ~ fs_receipt + age + age_sq + black + educ_years + married",
    'fs_receipt', 'robust/sample/male_only', 'robustness/sample_restrictions.md',
    'good_health', 'Demographics (no female)', 'none', 'yob',
    'Cross-sectional OLS', 'Male subsample',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# Female only
female_cs = cs_df[cs_df['female'] == 1]
res = run_spec(
    female_cs,
    f"good_health ~ fs_receipt + age + age_sq + black + educ_years + married",
    'fs_receipt', 'robust/sample/female_only', 'robustness/sample_restrictions.md',
    'good_health', 'Demographics (no female)', 'none', 'yob',
    'Cross-sectional OLS', 'Female subsample',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# Black only
black_cs = cs_df[cs_df['black'] == 1]
res = run_spec(
    black_cs,
    f"good_health ~ fs_receipt + age + age_sq + female + educ_years + married",
    'fs_receipt', 'robust/sample/black_only', 'robustness/sample_restrictions.md',
    'good_health', 'Demographics (no race)', 'none', 'yob',
    'Cross-sectional OLS', 'Black subsample',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# White only
white_cs = cs_df[cs_df['white'] == 1]
res = run_spec(
    white_cs,
    f"good_health ~ fs_receipt + age + age_sq + female + educ_years + married",
    'fs_receipt', 'robust/sample/white_only', 'robustness/sample_restrictions.md',
    'good_health', 'Demographics (no race)', 'none', 'yob',
    'Cross-sectional OLS', 'White subsample',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# Early cohorts (born <= 1965)
early_coh = cs_df[cs_df['yob'] <= 1965]
res = run_spec(
    early_coh,
    f"good_health ~ fs_receipt + {demo_controls_str}",
    'fs_receipt', 'robust/sample/early_cohort', 'robustness/sample_restrictions.md',
    'good_health', 'Demographics', 'none', 'yob',
    'Cross-sectional OLS', 'Born 1956-1965',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# Late cohorts (born > 1970)
late_coh = cs_df[cs_df['yob'] > 1970]
res = run_spec(
    late_coh,
    f"good_health ~ fs_receipt + {demo_controls_str}",
    'fs_receipt', 'robust/sample/late_cohort', 'robustness/sample_restrictions.md',
    'good_health', 'Demographics', 'none', 'yob',
    'Cross-sectional OLS', 'Born 1971-1981',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# High education
high_ed_cs = cs_df[cs_df['educ_years'] > 12]
res = run_spec(
    high_ed_cs,
    f"good_health ~ fs_receipt + age + age_sq + female + black + married",
    'fs_receipt', 'robust/sample/high_education', 'robustness/sample_restrictions.md',
    'good_health', 'Demographics (no educ)', 'none', 'yob',
    'Cross-sectional OLS', 'College education',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# Low education
low_ed_cs = cs_df[cs_df['educ_years'] <= 12]
res = run_spec(
    low_ed_cs,
    f"good_health ~ fs_receipt + age + age_sq + female + black + married",
    'fs_receipt', 'robust/sample/low_education', 'robustness/sample_restrictions.md',
    'good_health', 'Demographics (no educ)', 'none', 'yob',
    'Cross-sectional OLS', 'HS or less',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# Heads of household
head_cs = cs_df[cs_df['head'] == 1]
res = run_spec(
    head_cs,
    f"good_health ~ fs_receipt + {demo_controls_str}",
    'fs_receipt', 'robust/sample/heads_only', 'robustness/sample_restrictions.md',
    'good_health', 'Demographics', 'none', 'yob',
    'Cross-sectional OLS', 'Household heads only',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# Non-heads (spouses)
spouse_cs = cs_df[cs_df['head'] == 2]
res = run_spec(
    spouse_cs,
    f"good_health ~ fs_receipt + {demo_controls_str}",
    'fs_receipt', 'robust/sample/non_heads', 'robustness/sample_restrictions.md',
    'good_health', 'Demographics', 'none', 'yob',
    'Cross-sectional OLS', 'Spouses only',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# 1E. ALTERNATIVE TREATMENTS - CROSS-SECTIONAL (4 specs)
print("\n--- 1E. Alternative Treatments (Cross-sectional) ---")

# AFDC receipt
res = run_spec(
    cs_df,
    f"good_health ~ afdc_receipt + {demo_controls_str}",
    'afdc_receipt', 'robust/treatment/afdc', 'robustness/sample_restrictions.md',
    'good_health', 'Demographics', 'none', 'yob',
    'Cross-sectional OLS', 'First observation per person',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# Any welfare
res = run_spec(
    cs_df,
    f"good_health ~ any_welfare + {demo_controls_str}",
    'any_welfare', 'robust/treatment/any_welfare', 'robustness/sample_restrictions.md',
    'good_health', 'Demographics', 'none', 'yob',
    'Cross-sectional OLS', 'First observation per person',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# Both FS and AFDC
cs_df['both_programs'] = ((cs_df['fs_receipt'] == 1) & (cs_df['afdc_receipt'] == 1)).astype(float)
res = run_spec(
    cs_df,
    f"good_health ~ both_programs + {demo_controls_str}",
    'both_programs', 'robust/treatment/both_programs', 'robustness/sample_restrictions.md',
    'good_health', 'Demographics', 'none', 'yob',
    'Cross-sectional OLS', 'First observation per person',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# FS only (not AFDC)
cs_df['fs_only'] = ((cs_df['fs_receipt'] == 1) & (cs_df['afdc_receipt'] == 0)).astype(float)
res = run_spec(
    cs_df,
    f"good_health ~ fs_only + {demo_controls_str}",
    'fs_only', 'robust/treatment/fs_only', 'robustness/sample_restrictions.md',
    'good_health', 'Demographics', 'none', 'yob',
    'Cross-sectional OLS', 'First observation per person',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# 1F. ALTERNATIVE OUTCOMES - CROSS-SECTIONAL (5 specs)
print("\n--- 1F. Alternative Outcomes (Cross-sectional) ---")

# Excellent health
res = run_spec(
    cs_df,
    f"excellent_health ~ fs_receipt + {demo_controls_str}",
    'fs_receipt', 'robust/outcome/excellent_health', 'robustness/sample_restrictions.md',
    'excellent_health', 'Demographics', 'none', 'yob',
    'Cross-sectional OLS', 'First observation per person',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# Poor health
res = run_spec(
    cs_df,
    f"poor_health ~ fs_receipt + {demo_controls_str}",
    'fs_receipt', 'robust/outcome/poor_health', 'robustness/sample_restrictions.md',
    'poor_health', 'Demographics', 'none', 'yob',
    'Cross-sectional OLS', 'First observation per person',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# Work limitation (CS)
work_cs = cs_df.dropna(subset=['work_limited'])
res = run_spec(
    work_cs,
    f"work_limited ~ fs_receipt + {demo_controls_str}",
    'fs_receipt', 'robust/outcome/work_limited_cs', 'robustness/sample_restrictions.md',
    'work_limited', 'Demographics', 'none', 'yob',
    'Cross-sectional OLS', 'Non-missing work limitation',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# Employed (CS)
emp_cs = cs_df.dropna(subset=['employed'])
res = run_spec(
    emp_cs,
    f"employed ~ fs_receipt + {demo_controls_str}",
    'fs_receipt', 'robust/outcome/employed_cs', 'robustness/sample_restrictions.md',
    'employed', 'Demographics', 'none', 'yob',
    'Cross-sectional OLS', 'Non-missing employment',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# Log income (CS)
inc_cs = cs_df.dropna(subset=['log_income'])
res = run_spec(
    inc_cs,
    f"log_income ~ fs_receipt + {demo_controls_str}",
    'fs_receipt', 'robust/outcome/log_income_cs', 'robustness/sample_restrictions.md',
    'log_income', 'Demographics', 'none', 'yob',
    'Cross-sectional OLS', 'Non-missing income',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# 1G. HETEROGENEITY - CROSS-SECTIONAL (10 specs)
print("\n--- 1G. Heterogeneity Analysis (Cross-sectional) ---")

# Interaction with gender
res = run_spec(
    cs_df,
    f"good_health ~ fs_receipt * female + age + age_sq + black + educ_years + married",
    'fs_receipt', 'robust/het/interaction_gender', 'robustness/heterogeneity.md',
    'good_health', 'Demographics + interaction', 'none', 'yob',
    'Cross-sectional OLS', 'First observation per person',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# Interaction with race
race_het = cs_df.dropna(subset=['black'])
res = run_spec(
    race_het,
    f"good_health ~ fs_receipt * black + age + age_sq + female + educ_years + married",
    'fs_receipt', 'robust/het/interaction_race', 'robustness/heterogeneity.md',
    'good_health', 'Demographics + interaction', 'none', 'yob',
    'Cross-sectional OLS', 'Non-missing race',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# Interaction with age (median split)
cs_df['old'] = (cs_df['age'] >= cs_df['age'].median()).astype(float)
res = run_spec(
    cs_df,
    f"good_health ~ fs_receipt * old + age_sq + female + black + educ_years + married",
    'fs_receipt', 'robust/het/interaction_age', 'robustness/heterogeneity.md',
    'good_health', 'Demographics + interaction', 'none', 'yob',
    'Cross-sectional OLS', 'First observation per person',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# Interaction with education
educ_het = cs_df.dropna(subset=['educ_years'])
educ_het['high_educ'] = (educ_het['educ_years'] > 12).astype(float)
res = run_spec(
    educ_het,
    f"good_health ~ fs_receipt * high_educ + age + age_sq + female + black + married",
    'fs_receipt', 'robust/het/interaction_education', 'robustness/heterogeneity.md',
    'good_health', 'Demographics + interaction', 'none', 'yob',
    'Cross-sectional OLS', 'Non-missing education',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# Interaction with marital status
married_het = cs_df.dropna(subset=['married'])
res = run_spec(
    married_het,
    f"good_health ~ fs_receipt * married + age + age_sq + female + black + educ_years",
    'fs_receipt', 'robust/het/interaction_married', 'robustness/heterogeneity.md',
    'good_health', 'Demographics + interaction', 'none', 'yob',
    'Cross-sectional OLS', 'Non-missing marital',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# Interaction with household head
res = run_spec(
    cs_df,
    f"good_health ~ fs_receipt * head + age + age_sq + female + black + educ_years + married",
    'fs_receipt', 'robust/het/interaction_head', 'robustness/heterogeneity.md',
    'good_health', 'Demographics + interaction', 'none', 'yob',
    'Cross-sectional OLS', 'First observation per person',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# Interaction with female head
res = run_spec(
    cs_df,
    f"good_health ~ fs_receipt * femhead + age + age_sq + female + black + educ_years + married",
    'fs_receipt', 'robust/het/interaction_femhead', 'robustness/heterogeneity.md',
    'good_health', 'Demographics + interaction', 'none', 'yob',
    'Cross-sectional OLS', 'First observation per person',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# Interaction with cohort (late)
res = run_spec(
    cs_df,
    f"good_health ~ fs_receipt * cohort_late + age + age_sq + female + black + educ_years + married",
    'fs_receipt', 'robust/het/interaction_cohort', 'robustness/heterogeneity.md',
    'good_health', 'Demographics + interaction', 'none', 'yob',
    'Cross-sectional OLS', 'First observation per person',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# Interaction with family size
famsize_het = cs_df.dropna(subset=['famsize'])
famsize_het['large_family'] = (famsize_het['famsize'] >= 4).astype(float)
res = run_spec(
    famsize_het,
    f"good_health ~ fs_receipt * large_family + age + age_sq + female + black + educ_years + married",
    'fs_receipt', 'robust/het/interaction_famsize', 'robustness/heterogeneity.md',
    'good_health', 'Demographics + interaction', 'none', 'yob',
    'Cross-sectional OLS', 'Non-missing family size',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# By number of kids
kids_het = cs_df.dropna(subset=['kids'])
kids_het['has_kids'] = (kids_het['kids'] > 0).astype(float)
res = run_spec(
    kids_het,
    f"good_health ~ fs_receipt * has_kids + age + age_sq + female + black + educ_years + married",
    'fs_receipt', 'robust/het/interaction_kids', 'robustness/heterogeneity.md',
    'good_health', 'Demographics + interaction', 'none', 'yob',
    'Cross-sectional OLS', 'Non-missing kids',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# 1H. FUNCTIONAL FORM - CROSS-SECTIONAL (5 specs)
print("\n--- 1H. Functional Form (Cross-sectional) ---")

# Age cubed
cs_df['age_cubed'] = cs_df['age'] ** 3
res = run_spec(
    cs_df,
    f"good_health ~ fs_receipt + age + age_sq + age_cubed + female + black + educ_years + married",
    'fs_receipt', 'robust/funcform/age_cubic', 'robustness/functional_form.md',
    'good_health', 'Demographics (age cubic)', 'none', 'yob',
    'Cross-sectional OLS', 'First observation per person',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# Log age
cs_df['log_age'] = np.log(cs_df['age'])
res = run_spec(
    cs_df,
    f"good_health ~ fs_receipt + log_age + female + black + educ_years + married",
    'fs_receipt', 'robust/funcform/log_age', 'robustness/functional_form.md',
    'good_health', 'Demographics (log age)', 'none', 'yob',
    'Cross-sectional OLS', 'First observation per person',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# Education dummies
educ_dum = cs_df.dropna(subset=['educ_years'])
educ_dum['hs_dropout'] = (educ_dum['educ_years'] < 12).astype(float)
educ_dum['some_college'] = ((educ_dum['educ_years'] > 12) & (educ_dum['educ_years'] < 16)).astype(float)
educ_dum['college_grad'] = (educ_dum['educ_years'] >= 16).astype(float)
res = run_spec(
    educ_dum,
    f"good_health ~ fs_receipt + age + age_sq + female + black + hs_dropout + some_college + college_grad + married",
    'fs_receipt', 'robust/funcform/educ_dummies', 'robustness/functional_form.md',
    'good_health', 'Demographics (educ dummies)', 'none', 'yob',
    'Cross-sectional OLS', 'Non-missing education',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# Age x gender interaction
res = run_spec(
    cs_df,
    f"good_health ~ fs_receipt + age * female + age_sq + black + educ_years + married",
    'fs_receipt', 'robust/funcform/age_gender_interact', 'robustness/functional_form.md',
    'good_health', 'Demographics (age x gender)', 'none', 'yob',
    'Cross-sectional OLS', 'First observation per person',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# Health scale as outcome (continuous)
res = run_spec(
    cs_df,
    f"health_scale ~ fs_receipt + {demo_controls_str}",
    'fs_receipt', 'robust/funcform/ordinal', 'robustness/functional_form.md',
    'health_scale', 'Demographics', 'none', 'yob',
    'Cross-sectional OLS', 'First observation per person',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# ============================================================================
# PART 2: PANEL ANALYSIS (Time-varying outcomes)
# ============================================================================
print("\n" + "-"*70)
print("PART 2: PANEL SPECIFICATIONS (time-varying outcomes)")
print("-"*70)

# 2A. INCOME OUTCOMES WITH PANEL FE (8 specs)
print("\n--- 2A. Income Outcomes (Panel FE) ---")

# Baseline: FS on log income with individual + year FE
inc_panel = analysis_df.dropna(subset=['log_income'])
res = run_spec(
    inc_panel,
    f"log_income ~ fs_receipt + {panel_controls_str} | individual_id + Datayear",
    'fs_receipt', 'panel/income/baseline', 'methods/panel_fixed_effects.md',
    'log_income', 'Panel controls', 'individual + year', 'individual_id',
    'Panel FE (TWFE)', 'Full panel (income)',
    vcov={'CRV1': 'individual_id'}
)
if res: results.append(res)

# Unit FE only
res = run_spec(
    inc_panel,
    f"log_income ~ fs_receipt + {panel_controls_str} | individual_id",
    'fs_receipt', 'panel/income/unit_fe', 'methods/panel_fixed_effects.md',
    'log_income', 'Panel controls', 'individual', 'individual_id',
    'Panel FE (unit only)', 'Full panel (income)',
    vcov={'CRV1': 'individual_id'}
)
if res: results.append(res)

# Year FE only
res = run_spec(
    inc_panel,
    f"log_income ~ fs_receipt + {panel_controls_str} | Datayear",
    'fs_receipt', 'panel/income/time_fe', 'methods/panel_fixed_effects.md',
    'log_income', 'Panel controls', 'year', 'individual_id',
    'Panel FE (time only)', 'Full panel (income)',
    vcov={'CRV1': 'individual_id'}
)
if res: results.append(res)

# No FE
res = run_spec(
    inc_panel,
    f"log_income ~ fs_receipt + {panel_controls_str} + female + black + educ_years",
    'fs_receipt', 'panel/income/no_fe', 'methods/panel_fixed_effects.md',
    'log_income', 'Full controls', 'none', 'individual_id',
    'Pooled OLS', 'Full panel (income)',
    vcov={'CRV1': 'individual_id'}
)
if res: results.append(res)

# Male only
male_inc = inc_panel[inc_panel['female'] == 0]
res = run_spec(
    male_inc,
    f"log_income ~ fs_receipt + {panel_controls_str} | individual_id + Datayear",
    'fs_receipt', 'panel/income/male_only', 'methods/panel_fixed_effects.md',
    'log_income', 'Panel controls', 'individual + year', 'individual_id',
    'Panel FE (TWFE)', 'Male subsample',
    vcov={'CRV1': 'individual_id'}
)
if res: results.append(res)

# Female only
female_inc = inc_panel[inc_panel['female'] == 1]
res = run_spec(
    female_inc,
    f"log_income ~ fs_receipt + {panel_controls_str} | individual_id + Datayear",
    'fs_receipt', 'panel/income/female_only', 'methods/panel_fixed_effects.md',
    'log_income', 'Panel controls', 'individual + year', 'individual_id',
    'Panel FE (TWFE)', 'Female subsample',
    vcov={'CRV1': 'individual_id'}
)
if res: results.append(res)

# Black only
black_inc = inc_panel[inc_panel['black'] == 1]
res = run_spec(
    black_inc,
    f"log_income ~ fs_receipt + {panel_controls_str} | individual_id + Datayear",
    'fs_receipt', 'panel/income/black_only', 'methods/panel_fixed_effects.md',
    'log_income', 'Panel controls', 'individual + year', 'individual_id',
    'Panel FE (TWFE)', 'Black subsample',
    vcov={'CRV1': 'individual_id'}
)
if res: results.append(res)

# White only
white_inc = inc_panel[inc_panel['white'] == 1]
res = run_spec(
    white_inc,
    f"log_income ~ fs_receipt + {panel_controls_str} | individual_id + Datayear",
    'fs_receipt', 'panel/income/white_only', 'methods/panel_fixed_effects.md',
    'log_income', 'Panel controls', 'individual + year', 'individual_id',
    'Panel FE (TWFE)', 'White subsample',
    vcov={'CRV1': 'individual_id'}
)
if res: results.append(res)

# 2B. WORK LIMITATION WITH PANEL FE (5 specs)
print("\n--- 2B. Work Limitation (Panel FE) ---")

work_panel = analysis_df.dropna(subset=['work_limited'])

# Baseline
res = run_spec(
    work_panel,
    f"work_limited ~ fs_receipt + {panel_controls_str} | individual_id + Datayear",
    'fs_receipt', 'panel/worklimit/baseline', 'methods/panel_fixed_effects.md',
    'work_limited', 'Panel controls', 'individual + year', 'individual_id',
    'Panel FE (TWFE)', 'Full panel (work limit)',
    vcov={'CRV1': 'individual_id'}
)
if res: results.append(res)

# Unit FE only
res = run_spec(
    work_panel,
    f"work_limited ~ fs_receipt + {panel_controls_str} | individual_id",
    'fs_receipt', 'panel/worklimit/unit_fe', 'methods/panel_fixed_effects.md',
    'work_limited', 'Panel controls', 'individual', 'individual_id',
    'Panel FE (unit only)', 'Full panel (work limit)',
    vcov={'CRV1': 'individual_id'}
)
if res: results.append(res)

# Year FE only
res = run_spec(
    work_panel,
    f"work_limited ~ fs_receipt + {panel_controls_str} | Datayear",
    'fs_receipt', 'panel/worklimit/time_fe', 'methods/panel_fixed_effects.md',
    'work_limited', 'Panel controls', 'year', 'individual_id',
    'Panel FE (time only)', 'Full panel (work limit)',
    vcov={'CRV1': 'individual_id'}
)
if res: results.append(res)

# No FE
res = run_spec(
    work_panel,
    f"work_limited ~ fs_receipt + {panel_controls_str} + female + black + educ_years",
    'fs_receipt', 'panel/worklimit/no_fe', 'methods/panel_fixed_effects.md',
    'work_limited', 'Full controls', 'none', 'individual_id',
    'Pooled OLS', 'Full panel (work limit)',
    vcov={'CRV1': 'individual_id'}
)
if res: results.append(res)

# Early cohorts
early_work = work_panel[work_panel['yob'] <= 1965]
res = run_spec(
    early_work,
    f"work_limited ~ fs_receipt + {panel_controls_str} | individual_id + Datayear",
    'fs_receipt', 'panel/worklimit/early_cohort', 'methods/panel_fixed_effects.md',
    'work_limited', 'Panel controls', 'individual + year', 'individual_id',
    'Panel FE (TWFE)', 'Born 1956-1965',
    vcov={'CRV1': 'individual_id'}
)
if res: results.append(res)

# 2C. EMPLOYMENT WITH PANEL FE (5 specs)
print("\n--- 2C. Employment (Panel FE) ---")

emp_panel = analysis_df.dropna(subset=['employed'])

# Baseline
res = run_spec(
    emp_panel,
    f"employed ~ fs_receipt + {panel_controls_str} | individual_id + Datayear",
    'fs_receipt', 'panel/employed/baseline', 'methods/panel_fixed_effects.md',
    'employed', 'Panel controls', 'individual + year', 'individual_id',
    'Panel FE (TWFE)', 'Full panel (employed)',
    vcov={'CRV1': 'individual_id'}
)
if res: results.append(res)

# Unit FE only
res = run_spec(
    emp_panel,
    f"employed ~ fs_receipt + {panel_controls_str} | individual_id",
    'fs_receipt', 'panel/employed/unit_fe', 'methods/panel_fixed_effects.md',
    'employed', 'Panel controls', 'individual', 'individual_id',
    'Panel FE (unit only)', 'Full panel (employed)',
    vcov={'CRV1': 'individual_id'}
)
if res: results.append(res)

# Year FE only
res = run_spec(
    emp_panel,
    f"employed ~ fs_receipt + {panel_controls_str} | Datayear",
    'fs_receipt', 'panel/employed/time_fe', 'methods/panel_fixed_effects.md',
    'employed', 'Panel controls', 'year', 'individual_id',
    'Panel FE (time only)', 'Full panel (employed)',
    vcov={'CRV1': 'individual_id'}
)
if res: results.append(res)

# Male only
male_emp = emp_panel[emp_panel['female'] == 0]
res = run_spec(
    male_emp,
    f"employed ~ fs_receipt + {panel_controls_str} | individual_id + Datayear",
    'fs_receipt', 'panel/employed/male_only', 'methods/panel_fixed_effects.md',
    'employed', 'Panel controls', 'individual + year', 'individual_id',
    'Panel FE (TWFE)', 'Male subsample',
    vcov={'CRV1': 'individual_id'}
)
if res: results.append(res)

# Female only
female_emp = emp_panel[emp_panel['female'] == 1]
res = run_spec(
    female_emp,
    f"employed ~ fs_receipt + {panel_controls_str} | individual_id + Datayear",
    'fs_receipt', 'panel/employed/female_only', 'methods/panel_fixed_effects.md',
    'employed', 'Panel controls', 'individual + year', 'individual_id',
    'Panel FE (TWFE)', 'Female subsample',
    vcov={'CRV1': 'individual_id'}
)
if res: results.append(res)

# 2D. HOSPITAL UTILIZATION (4 specs)
print("\n--- 2D. Hospital Utilization (Panel FE) ---")

hosp_panel = analysis_df.dropna(subset=['any_hospital'])

# Any hospitalization
res = run_spec(
    hosp_panel,
    f"any_hospital ~ fs_receipt + {panel_controls_str} | individual_id + Datayear",
    'fs_receipt', 'panel/hospital/any', 'methods/panel_fixed_effects.md',
    'any_hospital', 'Panel controls', 'individual + year', 'individual_id',
    'Panel FE (TWFE)', 'Full panel (hospital)',
    vcov={'CRV1': 'individual_id'}
)
if res: results.append(res)

# Unit FE only
res = run_spec(
    hosp_panel,
    f"any_hospital ~ fs_receipt + {panel_controls_str} | individual_id",
    'fs_receipt', 'panel/hospital/unit_fe', 'methods/panel_fixed_effects.md',
    'any_hospital', 'Panel controls', 'individual', 'individual_id',
    'Panel FE (unit only)', 'Full panel (hospital)',
    vcov={'CRV1': 'individual_id'}
)
if res: results.append(res)

# Log hospital days
log_hosp = analysis_df.dropna(subset=['log_hospdays'])
res = run_spec(
    log_hosp,
    f"log_hospdays ~ fs_receipt + {panel_controls_str} | individual_id + Datayear",
    'fs_receipt', 'panel/hospital/log_days', 'methods/panel_fixed_effects.md',
    'log_hospdays', 'Panel controls', 'individual + year', 'individual_id',
    'Panel FE (TWFE)', 'Full panel (hospital)',
    vcov={'CRV1': 'individual_id'}
)
if res: results.append(res)

# Pooled OLS
res = run_spec(
    hosp_panel,
    f"any_hospital ~ fs_receipt + {panel_controls_str} + female + black + educ_years",
    'fs_receipt', 'panel/hospital/no_fe', 'methods/panel_fixed_effects.md',
    'any_hospital', 'Full controls', 'none', 'individual_id',
    'Pooled OLS', 'Full panel (hospital)',
    vcov={'CRV1': 'individual_id'}
)
if res: results.append(res)

# 2E. PLACEBO TESTS (3 specs)
print("\n--- 2E. Placebo Tests ---")

# Placebo: FS on year of birth (should be zero)
res = run_spec(
    cs_df,
    f"yob ~ fs_receipt + age + age_sq + female + black + educ_years",
    'fs_receipt', 'robust/placebo/yob_outcome', 'robustness/placebo_tests.md',
    'yob', 'Demographics', 'none', 'yob',
    'Cross-sectional OLS', 'First observation per person',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# Placebo: Random assignment
np.random.seed(42)
cs_df['random_treat'] = np.random.binomial(1, 0.12, len(cs_df))
res = run_spec(
    cs_df,
    f"good_health ~ random_treat + {demo_controls_str}",
    'random_treat', 'robust/placebo/random_treatment', 'robustness/placebo_tests.md',
    'good_health', 'Demographics', 'none', 'yob',
    'Cross-sectional OLS', 'First observation per person',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# Pre-1990 only (before welfare reform)
pre_90_df = df[df['Datayear'] < 1990].sort_values('Datayear').groupby('individual_id').first().reset_index()
pre_90_df = pre_90_df.dropna(subset=['good_health', 'fs_receipt', 'age'])
res = run_spec(
    pre_90_df,
    f"good_health ~ fs_receipt + {demo_controls_str}",
    'fs_receipt', 'robust/placebo/pre_reform', 'robustness/placebo_tests.md',
    'good_health', 'Demographics', 'none', 'yob',
    'Cross-sectional OLS', 'Pre-1990 first observation',
    vcov={'CRV1': 'yob'}
)
if res: results.append(res)

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

results_df = pd.DataFrame([r for r in results if r is not None])
print(f"\nTotal specifications run: {len(results_df)}")

# Save to CSV
output_path = f"{DATA_PATH}/specification_results.csv"
results_df.to_csv(output_path, index=False)
print(f"Results saved to: {output_path}")

# Summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

# Overall summary
print(f"\nTotal specifications: {len(results_df)}")

# By outcome variable
print("\n--- Results by Outcome Variable ---")
for outcome in results_df['outcome_var'].unique():
    subset = results_df[results_df['outcome_var'] == outcome]
    sig_5 = (subset['p_value'] < 0.05).sum()
    mean_coef = subset['coefficient'].mean()
    print(f"{outcome}: N={len(subset)}, Mean coef={mean_coef:.4f}, Sig 5%={sig_5} ({100*sig_5/len(subset):.1f}%)")

# Filter to main treatment (fs_receipt) and good_health outcome
main_results = results_df[(results_df['treatment_var'] == 'fs_receipt') &
                          (results_df['outcome_var'] == 'good_health')]

if len(main_results) > 0:
    print(f"\n--- Main Analysis: FS Receipt on Good Health ---")
    print(f"N specifications: {len(main_results)}")
    print(f"Mean coefficient: {main_results['coefficient'].mean():.4f}")
    print(f"Median coefficient: {main_results['coefficient'].median():.4f}")
    print(f"Range: [{main_results['coefficient'].min():.4f}, {main_results['coefficient'].max():.4f}]")
    print(f"Std dev: {main_results['coefficient'].std():.4f}")

    n_negative = (main_results['coefficient'] < 0).sum()
    n_sig_05 = (main_results['p_value'] < 0.05).sum()
    n_sig_01 = (main_results['p_value'] < 0.01).sum()

    print(f"\nNegative coefficients: {n_negative} ({100*n_negative/len(main_results):.1f}%)")
    print(f"Significant at 5%: {n_sig_05} ({100*n_sig_05/len(main_results):.1f}%)")
    print(f"Significant at 1%: {n_sig_01} ({100*n_sig_01/len(main_results):.1f}%)")

print("\n" + "="*70)
print("SPECIFICATION SEARCH COMPLETE")
print("="*70)
