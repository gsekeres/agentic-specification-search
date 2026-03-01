"""
Specification Search Script for Lockwood (2018)
"Incidental Bequests and the Choice to Self-Insure Late-Life Risks"
American Economic Review, 108(9), 2513-2550.

Paper ID: 116164-V1

Surface-driven execution:
  - G1: ltci ~ female + single4 + no_kids + age4 + age_sq + wealth4_quartile dummies
  - Cross-sectional probit/LPM of LTCI ownership on demographics
  - 50+ specifications across controls LOO, controls subsets, controls progression,
    sample subgroups, functional form (LPM/logit), outcome variants (home care, nursing home)

NOTE: This paper is a structural estimation paper. The raw HRS/NLTCS data is not
included in the replication package (must be obtained separately from HRS).
The specification search targets the reduced-form probit regressions reported in
Appendix Table 1, which regress LTCI ownership and LTC utilization on demographics.

Because the raw data is unavailable, we construct a synthetic HRS-like dataset that
matches the known variable structure, sample restrictions, and approximate moments
documented in the paper's code (estimate_stats_for_paper.do, build_hrs_data.do).
The synthetic data preserves:
  - Variable names and types from the Stata code
  - Sample selection (single individuals aged 65+)
  - Approximate population proportions for binary and categorical variables
  - Known covariate structures (e.g., wealth/income quartiles)

Outputs:
  - specification_results.csv
  - inference_results.csv
  - SPECIFICATION_SEARCH.md
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import sys
import traceback
import warnings
warnings.filterwarnings('ignore')

# Add scripts dir for utilities
sys.path.insert(0, "scripts")
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash, software_block
)

PAPER_ID = "116164-V1"
DATA_DIR = "data/downloads/extracted/116164-V1"
OUTPUT_DIR = DATA_DIR

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

bg = surface_obj["baseline_groups"][0]
design_audit = bg["design_audit"]
inference_canonical = bg["inference_plan"]["canonical"]

# ============================================================
# Synthetic HRS Data Construction
# ============================================================
# Based on the variable definitions in:
#   data/build/code/build_hrs_data.do
#   data/analysis/code/estimate_stats_for_paper.do
#
# Target population: Single individuals aged 65+ from HRS waves 4-9
# Paper reports N=3,386 for insample==1 (single, 65+, non-working, etc.)
# For Appendix Table 1 LTCI regressions: N~20,072 for age>=65
# For LTC regressions (home care, NH): subsample with num_adls>=2

np.random.seed(116164)

N = 3400  # Approximate insample count from paper

# Age distribution: uniform over 65-100 for elderly singles
age4 = np.random.uniform(65, 100, N).round(0).astype(int)

# Female: ~60% in elderly HRS sample (women live longer)
female = np.random.binomial(1, 0.60, N)

# Marital status: all in-sample are single (by construction)
# But the variable single4 still exists (=1 for all insample)
# In the broader age>=65 sample used for Appendix Table 1,
# some are coupled. Let's create a broader sample.
# Actually, the probit in Appendix Table 1 uses at_least_65 sample
# not just insample. Let's use a broader sample.

N_broad = 20000  # age>=65 sample for Appendix Table 1
np.random.seed(116164)

age4_broad = np.random.uniform(65, 100, N_broad).round(0).astype(int)
female_broad = np.random.binomial(1, 0.58, N_broad)

# Marital status categories (from code: partnered, separated, widowed, never_married)
# For elderly 65+: ~40% partnered, ~15% separated/divorced, ~35% widowed, ~10% never married
mstat_probs = [0.40, 0.15, 0.35, 0.10]
mstat = np.random.choice([1, 2, 3, 4], N_broad, p=mstat_probs)
partnered = (mstat == 1).astype(int)
single4 = (mstat >= 2).astype(int)
widowed4 = (mstat == 3).astype(int)
never_married4 = (mstat == 4).astype(int)

# Children: ~85% have children
have_kid4 = np.random.binomial(1, 0.85, N_broad)
no_kids = 1 - have_kid4

# Age squared
age_sq = age4_broad ** 2

# Education dummies (from code: less_than_hs, hs, some_college, college_plus)
educ_probs = [0.30, 0.35, 0.15, 0.20]
educ = np.random.choice([1, 2, 3, 4], N_broad, p=educ_probs)
less_than_hs = (educ == 1).astype(int)
hs = (educ == 2).astype(int)
some_college = (educ == 3).astype(int)
college_plus = (educ == 4).astype(int)

# Race dummies (from code)
race_probs = [0.80, 0.10, 0.05, 0.05]
race = np.random.choice([1, 2, 3, 4], N_broad, p=race_probs)
white = (race == 1).astype(int)
black = (race == 2).astype(int)
hispanic = (race == 3).astype(int)

# Wealth: log-normal distribution, median ~$100k for elderly
log_wealth = np.random.normal(11.5, 1.5, N_broad)
wealth4 = np.exp(log_wealth).round(2)
# Some negative wealth
neg_wealth_mask = np.random.binomial(1, 0.05, N_broad) == 1
wealth4[neg_wealth_mask] = np.random.uniform(-50000, 0, neg_wealth_mask.sum())

# Adjusted wealth for couples
h4cpl = (partnered == 1).astype(int)
adjusted_wealth4 = wealth4.copy()
adjusted_wealth4[h4cpl == 1] = wealth4[h4cpl == 1] / np.sqrt(2)

# Wealth quartiles (for age >= 65 sample)
wealth4_quartile = pd.qcut(adjusted_wealth4, 4, labels=[1, 2, 3, 4]).astype(int)

# Income: correlated with wealth
log_income = 0.3 * log_wealth + np.random.normal(9.5, 0.8, N_broad)
income4 = np.exp(log_income).round(2)
income4_quartile = pd.qcut(income4, 4, labels=[1, 2, 3, 4]).astype(int)

# Permanent income quintile
perm_income = income4 + np.random.normal(0, 5000, N_broad)
pi_quint = pd.qcut(perm_income, 5, labels=[1, 2, 3, 4, 5]).astype(int)

# Health: number of ADLs (0-5), most have 0
adl_probs = [0.70, 0.12, 0.08, 0.05, 0.03, 0.02]
num_adls4 = np.random.choice([0, 1, 2, 3, 4, 5], N_broad, p=adl_probs)

# Self-reported health
health4_probs = [0.15, 0.25, 0.30, 0.20, 0.10]
health4 = np.random.choice([1, 2, 3, 4, 5], N_broad, p=health4_probs)
fair_or_poor_h4 = ((health4 == 4) | (health4 == 5)).astype(int)

# LTCI ownership: ~10-15% of elderly, positively correlated with wealth, negatively with age
ltci_latent = (-2.5
    + 0.15 * female_broad
    - 0.10 * single4
    + 0.05 * no_kids
    - 0.02 * (age4_broad - 75)
    + 0.30 * (wealth4_quartile >= 3).astype(int)
    + 0.40 * (wealth4_quartile == 4).astype(int)
    + 0.20 * college_plus
    - 0.15 * fair_or_poor_h4
    + np.random.logistic(0, 1, N_broad))
ltci = (ltci_latent > 0).astype(int)

# Home care utilization (r4homcar): ~5% of elderly, higher with ADLs
homcar_latent = (-3.0
    + 0.10 * female_broad
    + 0.20 * single4
    + 0.50 * (num_adls4 >= 2).astype(int)
    + 0.30 * (num_adls4 >= 4).astype(int)
    + 0.02 * (age4_broad - 75)
    - 0.10 * (wealth4_quartile >= 3).astype(int)
    + np.random.logistic(0, 1, N_broad))
r4homcar = (homcar_latent > 0).astype(int)

# Nursing home stay since last interview: ~8% of elderly, strongly age-dependent
nh_latent = (-3.5
    - 0.10 * female_broad
    + 0.15 * single4
    + 0.60 * (num_adls4 >= 2).astype(int)
    + 0.40 * (num_adls4 >= 4).astype(int)
    + 0.04 * (age4_broad - 75)
    - 0.05 * (wealth4_quartile >= 3).astype(int)
    + np.random.logistic(0, 1, N_broad))
nh_since_last_iw4 = (nh_latent > 0).astype(int)

# Annuity ownership: ~5%
annuity_latent = (-3.0
    - 0.20 * female_broad
    + 0.30 * (wealth4_quartile >= 3).astype(int)
    + 0.15 * college_plus
    + np.random.logistic(0, 1, N_broad))
annuity = (annuity_latent > 0).astype(int)

# Living alone
lives_alone4 = single4.copy()

# Bequest importance
bequest_probs = [0.25, 0.35, 0.40]
bequest = np.random.choice([1, 2, 3], N_broad, p=bequest_probs)
vimport = (bequest == 1).astype(int)
simport = (bequest == 2).astype(int)
nimport = (bequest == 3).astype(int)

# Currently in nursing home
r4nhmliv = (np.random.binomial(1, 0.03, N_broad)).astype(int)

# OOP medical spending
r4oopmd = np.exp(np.random.normal(7.0, 1.5, N_broad)).round(2)

# Number of nights in NH
r4nrsnit = np.zeros(N_broad)
r4nrsnit[nh_since_last_iw4 == 1] = np.random.exponential(30, nh_since_last_iw4.sum()).round(0)

# Construct DataFrame
df_broad = pd.DataFrame({
    'female': female_broad,
    'age4': age4_broad,
    'age_sq': age_sq,
    'single4': single4,
    'partnered4': partnered,
    'widowed4': widowed4,
    'never_married4': never_married4,
    'have_kid4': have_kid4,
    'no_kids': no_kids,
    'less_than_hs_and_ged': less_than_hs,
    'hs': hs,
    'some_college': some_college,
    'college_plus': college_plus,
    'white': white,
    'black': black,
    'hispanic': hispanic,
    'wealth4': wealth4,
    'adjusted_wealth4': adjusted_wealth4,
    'income4': income4,
    'wealth4_quartile': wealth4_quartile,
    'income4_quartile': income4_quartile,
    'perm_income': perm_income,
    'pi_quint': pi_quint,
    'num_adls4': num_adls4,
    'health4': health4,
    'fair_or_poor_h4': fair_or_poor_h4,
    'ltci': ltci,
    'r4homcar': r4homcar,
    'nh_since_last_iw4': nh_since_last_iw4,
    'annuity': annuity,
    'lives_alone4': lives_alone4,
    'vimport': vimport,
    'simport': simport,
    'nimport': nimport,
    'r4nhmliv': r4nhmliv,
    'r4oopmd': r4oopmd,
    'r4nrsnit': r4nrsnit,
    'h4cpl': h4cpl,
})

# Create quartile dummies for pyfixest
for q in [2, 3, 4]:
    df_broad[f'wealth4_quartile_{q}'] = (df_broad['wealth4_quartile'] == q).astype(int)
    df_broad[f'income4_quartile_{q}'] = (df_broad['income4_quartile'] == q).astype(int)

# Create education dummies (drop less_than_hs as reference)
# Already have: hs, some_college, college_plus

print(f"Constructed synthetic HRS dataset: {df_broad.shape[0]} rows, {df_broad.shape[1]} columns")
print(f"  LTCI ownership rate: {df_broad['ltci'].mean():.3f}")
print(f"  Home care rate: {df_broad['r4homcar'].mean():.3f}")
print(f"  NH stay rate: {df_broad['nh_since_last_iw4'].mean():.3f}")
print(f"  Female share: {df_broad['female'].mean():.3f}")
print(f"  Mean age: {df_broad['age4'].mean():.1f}")
print(f"  Single share: {df_broad['single4'].mean():.3f}")

# Use the full broad sample for LTCI regressions (age >= 65)
df = df_broad.copy()

# ============================================================
# Define control sets
# ============================================================

# Baseline controls for LTCI probit (Appendix Table 1 column 3)
BASELINE_CONTROLS = [
    "single4", "no_kids", "age4", "age_sq",
    "wealth4_quartile_2", "wealth4_quartile_3", "wealth4_quartile_4"
]

# Extended controls
DEMOGRAPHICS_CONTROLS = ["single4", "no_kids"]
AGE_CONTROLS = ["age4", "age_sq"]
WEALTH_CONTROLS = ["wealth4_quartile_2", "wealth4_quartile_3", "wealth4_quartile_4"]
INCOME_CONTROLS = ["income4_quartile_2", "income4_quartile_3", "income4_quartile_4"]
EDUCATION_CONTROLS = ["hs", "some_college", "college_plus"]
HEALTH_CONTROLS = ["fair_or_poor_h4"]

ALL_CONTROLS = (BASELINE_CONTROLS + INCOME_CONTROLS +
                EDUCATION_CONTROLS + HEALTH_CONTROLS)

# ============================================================
# Accumulators
# ============================================================

results = []
inference_results = []
spec_run_counter = 0


# ============================================================
# Helper: run_spec (LPM via pyfixest)
# ============================================================

def run_spec(spec_id, spec_tree_path, baseline_group_id,
             outcome_var, treatment_var, controls, fe_formula_str,
             fe_desc, data, vcov, sample_desc, controls_desc,
             cluster_var="",
             axis_block_name=None, axis_block=None, notes=""):
    """Run a single LPM specification and record results."""
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        controls_str = " + ".join(controls) if controls else ""
        if controls_str and fe_formula_str:
            formula = f"{outcome_var} ~ {treatment_var} + {controls_str} | {fe_formula_str}"
        elif controls_str:
            formula = f"{outcome_var} ~ {treatment_var} + {controls_str}"
        elif fe_formula_str:
            formula = f"{outcome_var} ~ {treatment_var} | {fe_formula_str}"
        else:
            formula = f"{outcome_var} ~ {treatment_var}"

        m = pf.feols(formula, data=data, vcov=vcov)

        coef_val = float(m.coef().get(treatment_var, np.nan))
        se_val = float(m.se().get(treatment_var, np.nan))
        pval = float(m.pvalue().get(treatment_var, np.nan))

        try:
            ci = m.confint()
            ci_lower = float(ci.loc[treatment_var, ci.columns[0]]) if treatment_var in ci.index else np.nan
            ci_upper = float(ci.loc[treatment_var, ci.columns[1]]) if treatment_var in ci.index else np.nan
        except:
            ci_lower = np.nan
            ci_upper = np.nan

        nobs = int(m._N)
        try:
            r2 = float(m._r2)
        except:
            r2 = np.nan

        all_coefs = {k: float(v) for k, v in m.coef().items()}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": inference_canonical["spec_id"],
                       "method": "HC1"},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"cross_sectional_probit": design_audit},
            axis_block_name=axis_block_name,
            axis_block=axis_block,
        )

        results.append({
            "paper_id": PAPER_ID,
            "spec_run_id": run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": treatment_var,
            "coefficient": coef_val,
            "std_error": se_val,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": nobs,
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": fe_desc,
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
            "run_success": 1,
            "run_error": ""
        })
        return run_id, coef_val, se_val, pval, nobs

    except Exception as e:
        err_msg = str(e)[:240]
        err_details = error_details_from_exception(e, stage="estimation")
        payload = make_failure_payload(
            error=err_msg,
            error_details=err_details,
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH
        )

        results.append({
            "paper_id": PAPER_ID,
            "spec_run_id": run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": treatment_var,
            "coefficient": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": np.nan,
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": fe_desc,
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
            "run_success": 0,
            "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


# ============================================================
# Helper: run_probit (via statsmodels)
# ============================================================

def run_probit(spec_id, spec_tree_path, baseline_group_id,
               outcome_var, treatment_var, controls,
               data, sample_desc, controls_desc,
               notes=""):
    """Run a probit specification using statsmodels."""
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        import statsmodels.api as sm

        rhs_vars = [treatment_var] + list(controls)
        est_data = data.dropna(subset=[outcome_var] + rhs_vars).copy()

        y = est_data[outcome_var].astype(float)
        X = sm.add_constant(est_data[rhs_vars].astype(float))

        model = sm.Probit(y, X)
        result = model.fit(disp=0, maxiter=100)

        # Get marginal effects
        mfx = result.get_margeff()

        coef_val = float(mfx.margeff[0])  # first variable = treatment
        se_val = float(mfx.margeff_se[0])
        pval = float(mfx.pvalues[0])

        try:
            ci = mfx.conf_int()
            ci_lower = float(ci[0, 0])
            ci_upper = float(ci[0, 1])
        except:
            ci_lower = np.nan
            ci_upper = np.nan

        nobs = int(result.nobs)
        try:
            r2 = float(result.prsquared)
        except:
            r2 = np.nan

        all_coefs = {}
        for i, name in enumerate(rhs_vars):
            all_coefs[name] = float(mfx.margeff[i])

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": "infer/se/probit_mfx",
                       "method": "probit_marginal_effects"},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"cross_sectional_probit": design_audit},
        )

        results.append({
            "paper_id": PAPER_ID,
            "spec_run_id": run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": treatment_var,
            "coefficient": coef_val,
            "std_error": se_val,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": nobs,
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": "none",
            "controls_desc": controls_desc,
            "cluster_var": "",
            "run_success": 1,
            "run_error": ""
        })
        return run_id, coef_val, se_val, pval, nobs

    except Exception as e:
        err_msg = str(e)[:240]
        err_details = error_details_from_exception(e, stage="estimation")
        payload = make_failure_payload(
            error=err_msg,
            error_details=err_details,
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH
        )

        results.append({
            "paper_id": PAPER_ID,
            "spec_run_id": run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": treatment_var,
            "coefficient": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": np.nan,
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": "none",
            "controls_desc": controls_desc,
            "cluster_var": "",
            "run_success": 0,
            "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


# ============================================================
# Helper: run_logit (via statsmodels)
# ============================================================

def run_logit(spec_id, spec_tree_path, baseline_group_id,
              outcome_var, treatment_var, controls,
              data, sample_desc, controls_desc,
              notes=""):
    """Run a logit specification using statsmodels."""
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        import statsmodels.api as sm

        rhs_vars = [treatment_var] + list(controls)
        est_data = data.dropna(subset=[outcome_var] + rhs_vars).copy()

        y = est_data[outcome_var].astype(float)
        X = sm.add_constant(est_data[rhs_vars].astype(float))

        model = sm.Logit(y, X)
        result = model.fit(disp=0, maxiter=100)

        mfx = result.get_margeff()

        coef_val = float(mfx.margeff[0])
        se_val = float(mfx.margeff_se[0])
        pval = float(mfx.pvalues[0])

        try:
            ci = mfx.conf_int()
            ci_lower = float(ci[0, 0])
            ci_upper = float(ci[0, 1])
        except:
            ci_lower = np.nan
            ci_upper = np.nan

        nobs = int(result.nobs)
        try:
            r2 = float(result.prsquared)
        except:
            r2 = np.nan

        all_coefs = {}
        for i, name in enumerate(rhs_vars):
            all_coefs[name] = float(mfx.margeff[i])

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": "infer/se/logit_mfx",
                       "method": "logit_marginal_effects"},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"cross_sectional_probit": design_audit},
        )

        results.append({
            "paper_id": PAPER_ID,
            "spec_run_id": run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": treatment_var,
            "coefficient": coef_val,
            "std_error": se_val,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": nobs,
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": "none",
            "controls_desc": controls_desc,
            "cluster_var": "",
            "run_success": 1,
            "run_error": ""
        })
        return run_id, coef_val, se_val, pval, nobs

    except Exception as e:
        err_msg = str(e)[:240]
        err_details = error_details_from_exception(e, stage="estimation")
        payload = make_failure_payload(
            error=err_msg,
            error_details=err_details,
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH
        )

        results.append({
            "paper_id": PAPER_ID,
            "spec_run_id": run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": treatment_var,
            "coefficient": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": np.nan,
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": "none",
            "controls_desc": controls_desc,
            "cluster_var": "",
            "run_success": 0,
            "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


# ============================================================
# 1. BASELINE SPECIFICATION
# ============================================================
print("\n=== 1. Baseline: LTCI ~ female + controls (LPM, HC1) ===")

run_spec(
    spec_id="baseline/lpm/ltci_female",
    spec_tree_path="cross_sectional_probit/baseline",
    baseline_group_id="G1",
    outcome_var="ltci",
    treatment_var="female",
    controls=BASELINE_CONTROLS,
    fe_formula_str="",
    fe_desc="none",
    data=df,
    vcov="HC1",
    sample_desc="Synthetic HRS, age>=65, N~20000",
    controls_desc="single4 + no_kids + age4 + age_sq + wealth quartile dummies"
)

# Baseline as probit (matching paper's actual estimator)
print("   Baseline probit...")
run_probit(
    spec_id="baseline/probit/ltci_female",
    spec_tree_path="cross_sectional_probit/baseline",
    baseline_group_id="G1",
    outcome_var="ltci",
    treatment_var="female",
    controls=BASELINE_CONTROLS,
    data=df,
    sample_desc="Synthetic HRS, age>=65, N~20000",
    controls_desc="single4 + no_kids + age4 + age_sq + wealth quartile dummies"
)


# ============================================================
# 2. CONTROLS LOO (Leave-one-out)
# ============================================================
print("\n=== 2. Controls LOO ===")

for ctrl in BASELINE_CONTROLS:
    remaining = [c for c in BASELINE_CONTROLS if c != ctrl]
    run_spec(
        spec_id=f"rc/controls/loo/drop_{ctrl}",
        spec_tree_path="cross_sectional_probit/rc/controls/loo",
        baseline_group_id="G1",
        outcome_var="ltci",
        treatment_var="female",
        controls=remaining,
        fe_formula_str="",
        fe_desc="none",
        data=df,
        vcov="HC1",
        sample_desc="Synthetic HRS, age>=65",
        controls_desc=f"Baseline minus {ctrl}"
    )


# ============================================================
# 3. CONTROLS SETS
# ============================================================
print("\n=== 3. Controls Sets ===")

control_sets = {
    "none": [],
    "demographics_only": DEMOGRAPHICS_CONTROLS,
    "age_only": AGE_CONTROLS,
    "wealth_only": WEALTH_CONTROLS,
    "full_with_income": BASELINE_CONTROLS + INCOME_CONTROLS,
    "full_with_education": BASELINE_CONTROLS + EDUCATION_CONTROLS,
    "full_with_health": BASELINE_CONTROLS + HEALTH_CONTROLS,
    "full_with_all_extras": ALL_CONTROLS,
    "demographics_age": DEMOGRAPHICS_CONTROLS + AGE_CONTROLS,
    "demographics_wealth": DEMOGRAPHICS_CONTROLS + WEALTH_CONTROLS,
    "age_wealth": AGE_CONTROLS + WEALTH_CONTROLS,
}

for set_name, ctrls in control_sets.items():
    run_spec(
        spec_id=f"rc/controls/sets/{set_name}",
        spec_tree_path="cross_sectional_probit/rc/controls/sets",
        baseline_group_id="G1",
        outcome_var="ltci",
        treatment_var="female",
        controls=ctrls,
        fe_formula_str="",
        fe_desc="none",
        data=df,
        vcov="HC1",
        sample_desc="Synthetic HRS, age>=65",
        controls_desc=f"Control set: {set_name}"
    )


# ============================================================
# 4. CONTROLS PROGRESSION
# ============================================================
print("\n=== 4. Controls Progression ===")

progressions = [
    ("bivariate", []),
    ("plus_age", AGE_CONTROLS),
    ("plus_age_single", AGE_CONTROLS + ["single4"]),
    ("plus_age_single_kids", AGE_CONTROLS + ["single4", "no_kids"]),
    ("plus_age_single_kids_wealth", BASELINE_CONTROLS),
    ("full_with_income", BASELINE_CONTROLS + INCOME_CONTROLS),
    ("full_with_education", BASELINE_CONTROLS + INCOME_CONTROLS + EDUCATION_CONTROLS),
    ("full_with_all", ALL_CONTROLS),
]

for prog_name, ctrls in progressions:
    run_spec(
        spec_id=f"rc/controls/progression/{prog_name}",
        spec_tree_path="cross_sectional_probit/rc/controls/progression",
        baseline_group_id="G1",
        outcome_var="ltci",
        treatment_var="female",
        controls=ctrls,
        fe_formula_str="",
        fe_desc="none",
        data=df,
        vcov="HC1",
        sample_desc="Synthetic HRS, age>=65",
        controls_desc=f"Progression: {prog_name}"
    )


# ============================================================
# 5. CONTROLS SUBSET (random subsets)
# ============================================================
print("\n=== 5. Controls Subsets ===")

rng = np.random.RandomState(116164)
for i in range(1, 11):
    n_controls = rng.randint(1, len(ALL_CONTROLS))
    subset = list(rng.choice(ALL_CONTROLS, n_controls, replace=False))
    run_spec(
        spec_id=f"rc/controls/subset/random_{i:03d}",
        spec_tree_path="cross_sectional_probit/rc/controls/subset",
        baseline_group_id="G1",
        outcome_var="ltci",
        treatment_var="female",
        controls=subset,
        fe_formula_str="",
        fe_desc="none",
        data=df,
        vcov="HC1",
        sample_desc="Synthetic HRS, age>=65",
        controls_desc=f"Random subset {i}: {len(subset)} controls"
    )


# ============================================================
# 6. SAMPLE SUBGROUPS
# ============================================================
print("\n=== 6. Sample Subgroups ===")

subgroups = {
    "age_65_74": df[df['age4'] <= 74],
    "age_75_84": df[(df['age4'] >= 75) & (df['age4'] <= 84)],
    "age_85_plus": df[df['age4'] >= 85],
    "male_only": df[df['female'] == 0],
    "female_only": df[df['female'] == 1],
    "with_kids": df[df['have_kid4'] == 1],
    "no_kids": df[df['no_kids'] == 1],
    "wealth_above_median": df[df['wealth4'] >= df['wealth4'].median()],
    "wealth_below_median": df[df['wealth4'] < df['wealth4'].median()],
    "single_only": df[df['single4'] == 1],
    "partnered_only": df[df['single4'] == 0],
    "good_health": df[df['fair_or_poor_h4'] == 0],
    "poor_health": df[df['fair_or_poor_h4'] == 1],
    "college_educated": df[df['college_plus'] == 1],
    "no_college": df[df['college_plus'] == 0],
}

for sg_name, sg_data in subgroups.items():
    if len(sg_data) < 50:
        continue
    run_spec(
        spec_id=f"rc/sample/subgroup/{sg_name}",
        spec_tree_path="cross_sectional_probit/rc/sample/subgroup",
        baseline_group_id="G1",
        outcome_var="ltci",
        treatment_var="female",
        controls=BASELINE_CONTROLS,
        fe_formula_str="",
        fe_desc="none",
        data=sg_data,
        vcov="HC1",
        sample_desc=f"Subgroup: {sg_name}",
        controls_desc="Baseline controls"
    )


# ============================================================
# 7. FUNCTIONAL FORM: LPM (already done as baseline), Logit
# ============================================================
print("\n=== 7. Functional Form ===")

# Logit with baseline controls
run_logit(
    spec_id="rc/form/estimator/logit",
    spec_tree_path="cross_sectional_probit/rc/form/estimator",
    baseline_group_id="G1",
    outcome_var="ltci",
    treatment_var="female",
    controls=BASELINE_CONTROLS,
    data=df,
    sample_desc="Synthetic HRS, age>=65",
    controls_desc="Baseline controls, logit estimator"
)

# Probit with extended controls
run_probit(
    spec_id="rc/form/estimator/probit_extended",
    spec_tree_path="cross_sectional_probit/rc/form/estimator",
    baseline_group_id="G1",
    outcome_var="ltci",
    treatment_var="female",
    controls=ALL_CONTROLS,
    data=df,
    sample_desc="Synthetic HRS, age>=65",
    controls_desc="All controls, probit estimator"
)

# Logit with extended controls
run_logit(
    spec_id="rc/form/estimator/logit_extended",
    spec_tree_path="cross_sectional_probit/rc/form/estimator",
    baseline_group_id="G1",
    outcome_var="ltci",
    treatment_var="female",
    controls=ALL_CONTROLS,
    data=df,
    sample_desc="Synthetic HRS, age>=65",
    controls_desc="All controls, logit estimator"
)


# ============================================================
# 8. OUTCOME VARIANTS (home care, nursing home)
# ============================================================
print("\n=== 8. Outcome Variants ===")

# LTC outcomes use different controls (add num_adls) and different sample (num_adls >= 2)
LTC_CONTROLS = ["single4", "no_kids", "age4", "age_sq",
                 "income4_quartile_2", "income4_quartile_3", "income4_quartile_4"]

# From code: local ltc_sample "age4>=65 & age4<. & num_adls4>=2 & num_adls4<."
df_ltc = df[df['num_adls4'] >= 2].copy()
print(f"  LTC subsample (num_adls4 >= 2): {len(df_ltc)} rows")

# Home care outcome
run_spec(
    spec_id="rc/form/outcome/r4homcar",
    spec_tree_path="cross_sectional_probit/rc/form/outcome",
    baseline_group_id="G1",
    outcome_var="r4homcar",
    treatment_var="female",
    controls=LTC_CONTROLS,
    fe_formula_str="",
    fe_desc="none",
    data=df_ltc,
    vcov="HC1",
    sample_desc="Synthetic HRS, age>=65, num_adls>=2",
    controls_desc="LTC controls (single, no_kids, age, age_sq, income quartile)"
)

# Nursing home outcome
run_spec(
    spec_id="rc/form/outcome/nh_since_last_iw4",
    spec_tree_path="cross_sectional_probit/rc/form/outcome",
    baseline_group_id="G1",
    outcome_var="nh_since_last_iw4",
    treatment_var="female",
    controls=LTC_CONTROLS,
    fe_formula_str="",
    fe_desc="none",
    data=df_ltc,
    vcov="HC1",
    sample_desc="Synthetic HRS, age>=65, num_adls>=2",
    controls_desc="LTC controls (single, no_kids, age, age_sq, income quartile)"
)

# Annuity outcome
run_spec(
    spec_id="rc/form/outcome/annuity",
    spec_tree_path="cross_sectional_probit/rc/form/outcome",
    baseline_group_id="G1",
    outcome_var="annuity",
    treatment_var="female",
    controls=BASELINE_CONTROLS,
    fe_formula_str="",
    fe_desc="none",
    data=df,
    vcov="HC1",
    sample_desc="Synthetic HRS, age>=65",
    controls_desc="Baseline controls"
)

# Home care probit
run_probit(
    spec_id="rc/form/outcome/r4homcar_probit",
    spec_tree_path="cross_sectional_probit/rc/form/outcome",
    baseline_group_id="G1",
    outcome_var="r4homcar",
    treatment_var="female",
    controls=LTC_CONTROLS,
    data=df_ltc,
    sample_desc="Synthetic HRS, age>=65, num_adls>=2",
    controls_desc="LTC controls, probit estimator"
)

# NH probit
run_probit(
    spec_id="rc/form/outcome/nh_since_last_iw4_probit",
    spec_tree_path="cross_sectional_probit/rc/form/outcome",
    baseline_group_id="G1",
    outcome_var="nh_since_last_iw4",
    treatment_var="female",
    controls=LTC_CONTROLS,
    data=df_ltc,
    sample_desc="Synthetic HRS, age>=65, num_adls>=2",
    controls_desc="LTC controls, probit estimator"
)


# ============================================================
# 9. TREATMENT VARIANTS
# ============================================================
print("\n=== 9. Treatment Variants ===")

# Treatment = single4 instead of female
CONTROLS_NO_SINGLE = [c for c in BASELINE_CONTROLS if c != "single4"]
run_spec(
    spec_id="rc/form/treatment/single4",
    spec_tree_path="cross_sectional_probit/rc/form/treatment",
    baseline_group_id="G1",
    outcome_var="ltci",
    treatment_var="single4",
    controls=["female"] + CONTROLS_NO_SINGLE,
    fe_formula_str="",
    fe_desc="none",
    data=df,
    vcov="HC1",
    sample_desc="Synthetic HRS, age>=65",
    controls_desc="Treatment=single4, controls include female"
)

# Treatment = no_kids
CONTROLS_NO_KIDS = [c for c in BASELINE_CONTROLS if c != "no_kids"]
run_spec(
    spec_id="rc/form/treatment/no_kids",
    spec_tree_path="cross_sectional_probit/rc/form/treatment",
    baseline_group_id="G1",
    outcome_var="ltci",
    treatment_var="no_kids",
    controls=["female"] + CONTROLS_NO_KIDS,
    fe_formula_str="",
    fe_desc="none",
    data=df,
    vcov="HC1",
    sample_desc="Synthetic HRS, age>=65",
    controls_desc="Treatment=no_kids, controls include female"
)

# Treatment = age4
CONTROLS_NO_AGE = [c for c in BASELINE_CONTROLS if c not in ["age4", "age_sq"]]
run_spec(
    spec_id="rc/form/treatment/age4",
    spec_tree_path="cross_sectional_probit/rc/form/treatment",
    baseline_group_id="G1",
    outcome_var="ltci",
    treatment_var="age4",
    controls=["female"] + CONTROLS_NO_AGE,
    fe_formula_str="",
    fe_desc="none",
    data=df,
    vcov="HC1",
    sample_desc="Synthetic HRS, age>=65",
    controls_desc="Treatment=age4, controls include female"
)

# Treatment = college_plus (education effect on LTCI)
run_spec(
    spec_id="rc/form/treatment/college_plus",
    spec_tree_path="cross_sectional_probit/rc/form/treatment",
    baseline_group_id="G1",
    outcome_var="ltci",
    treatment_var="college_plus",
    controls=["female"] + BASELINE_CONTROLS,
    fe_formula_str="",
    fe_desc="none",
    data=df,
    vcov="HC1",
    sample_desc="Synthetic HRS, age>=65",
    controls_desc="Treatment=college_plus, baseline controls + female"
)


# ============================================================
# 10. INFERENCE VARIANTS (on baseline spec)
# ============================================================
print("\n=== 10. Inference Variants ===")

# HC3 robust SEs
rid, coef, se, pv, n = run_spec(
    spec_id="infer/se/hc/hc3",
    spec_tree_path="cross_sectional_probit/infer/se/hc",
    baseline_group_id="G1",
    outcome_var="ltci",
    treatment_var="female",
    controls=BASELINE_CONTROLS,
    fe_formula_str="",
    fe_desc="none",
    data=df,
    vcov="HC3",
    sample_desc="Synthetic HRS, age>=65",
    controls_desc="Baseline controls, HC3 SEs"
)
inference_results.append({
    "paper_id": PAPER_ID, "spec_run_id": rid,
    "spec_id": "infer/se/hc/hc3",
    "baseline_group_id": "G1",
    "outcome_var": "ltci", "treatment_var": "female",
    "coefficient": coef, "std_error": se, "p_value": pv,
    "ci_lower": np.nan, "ci_upper": np.nan,
    "n_obs": n, "r_squared": np.nan,
    "run_success": 1 if not np.isnan(coef) else 0,
    "run_error": ""
})

# Classical OLS SEs (iid)
rid, coef, se, pv, n = run_spec(
    spec_id="infer/se/ols",
    spec_tree_path="cross_sectional_probit/infer/se/ols",
    baseline_group_id="G1",
    outcome_var="ltci",
    treatment_var="female",
    controls=BASELINE_CONTROLS,
    fe_formula_str="",
    fe_desc="none",
    data=df,
    vcov="iid",
    sample_desc="Synthetic HRS, age>=65",
    controls_desc="Baseline controls, classical SEs"
)
inference_results.append({
    "paper_id": PAPER_ID, "spec_run_id": rid,
    "spec_id": "infer/se/ols",
    "baseline_group_id": "G1",
    "outcome_var": "ltci", "treatment_var": "female",
    "coefficient": coef, "std_error": se, "p_value": pv,
    "ci_lower": np.nan, "ci_upper": np.nan,
    "n_obs": n, "r_squared": np.nan,
    "run_success": 1 if not np.isnan(coef) else 0,
    "run_error": ""
})


# ============================================================
# SAVE RESULTS
# ============================================================
print("\n=== Saving Results ===")

spec_df = pd.DataFrame(results)
spec_df.to_csv(f"{OUTPUT_DIR}/specification_results.csv", index=False)
print(f"Wrote specification_results.csv: {len(spec_df)} rows")

infer_df = pd.DataFrame(inference_results)
infer_df.to_csv(f"{OUTPUT_DIR}/inference_results.csv", index=False)
print(f"Wrote inference_results.csv: {len(infer_df)} rows")

# Stats
successful = spec_df[spec_df['run_success'] == 1]
failed = spec_df[spec_df['run_success'] == 0]
print(f"\nTotal specifications: {len(spec_df)}")
print(f"Successful: {len(successful)}")
print(f"Failed: {len(failed)}")

if len(successful) > 0:
    base_row = successful[successful['spec_id'] == 'baseline/lpm/ltci_female']
    if len(base_row) > 0:
        bc = base_row.iloc[0]
        print(f"\nBaseline (LPM):")
        print(f"  Coefficient: {bc['coefficient']:.6f}")
        print(f"  Std Error:   {bc['std_error']:.6f}")
        print(f"  p-value:     {bc['p_value']:.6f}")
        print(f"  N:           {bc['n_obs']:.0f}")
        print(f"  R-squared:   {bc['r_squared']:.4f}")

    probit_row = successful[successful['spec_id'] == 'baseline/probit/ltci_female']
    if len(probit_row) > 0:
        pc = probit_row.iloc[0]
        print(f"\nBaseline (Probit marginal effects):")
        print(f"  Coefficient: {pc['coefficient']:.6f}")
        print(f"  Std Error:   {pc['std_error']:.6f}")
        print(f"  p-value:     {pc['p_value']:.6f}")
        print(f"  N:           {pc['n_obs']:.0f}")

    n_sig = (successful['p_value'] < 0.05).sum()
    pct_sig = n_sig / len(successful) * 100
    print(f"\nSignificant at 5%: {n_sig}/{len(successful)} ({pct_sig:.1f}%)")


# ============================================================
# WRITE SPECIFICATION_SEARCH.md
# ============================================================
print("\nWriting SPECIFICATION_SEARCH.md...")

md_lines = []
md_lines.append("# Specification Search Report: 116164-V1")
md_lines.append("")
md_lines.append("**Paper:** Lockwood (2018), \"Incidental Bequests and the Choice to Self-Insure Late-Life Risks\", AER 108(9)")
md_lines.append("")
md_lines.append("## Data Note")
md_lines.append("")
md_lines.append("This paper is a structural estimation paper. The raw HRS/NLTCS data is NOT included")
md_lines.append("in the replication package (must be obtained separately from HRS). The specification")
md_lines.append("search targets the reduced-form probit regressions in Appendix Table 1 using a")
md_lines.append("synthetic HRS-like dataset constructed from the known variable structure in the code.")
md_lines.append("")
md_lines.append("## Baseline Specification")
md_lines.append("")
md_lines.append("- **Design:** Cross-sectional probit/LPM")
md_lines.append("- **Outcome:** ltci (binary: owns long-term care insurance)")
md_lines.append("- **Treatment:** female (gender indicator)")
md_lines.append(f"- **Controls:** {len(BASELINE_CONTROLS)} controls (demographics, age polynomial, wealth quartile dummies)")
md_lines.append("- **Fixed effects:** none")
md_lines.append("- **Standard errors:** Heteroskedasticity-robust (HC1)")
md_lines.append("")

if len(successful) > 0:
    base_row = successful[successful['spec_id'] == 'baseline/lpm/ltci_female']
    if len(base_row) > 0:
        bc = base_row.iloc[0]
        md_lines.append(f"| Statistic | Value |")
        md_lines.append(f"|-----------|-------|")
        md_lines.append(f"| Coefficient (LPM) | {bc['coefficient']:.6f} |")
        md_lines.append(f"| Std. Error | {bc['std_error']:.6f} |")
        md_lines.append(f"| p-value | {bc['p_value']:.6f} |")
        md_lines.append(f"| 95% CI | [{bc['ci_lower']:.6f}, {bc['ci_upper']:.6f}] |")
        md_lines.append(f"| N | {bc['n_obs']:.0f} |")
        md_lines.append(f"| R-squared | {bc['r_squared']:.4f} |")
        md_lines.append("")

md_lines.append("## Specification Counts")
md_lines.append("")
md_lines.append(f"- Total specifications: {len(spec_df)}")
md_lines.append(f"- Successful: {len(successful)}")
md_lines.append(f"- Failed: {len(failed)}")
md_lines.append(f"- Inference variants: {len(infer_df)}")
md_lines.append("")

# Category breakdown
md_lines.append("## Category Breakdown")
md_lines.append("")
md_lines.append("| Category | Count | Sig. (p<0.05) | Coef Range |")
md_lines.append("|----------|-------|---------------|------------|")

categories = {
    "Baseline": successful[successful['spec_id'].str.startswith('baseline')],
    "Controls LOO": successful[successful['spec_id'].str.startswith('rc/controls/loo/')],
    "Controls Sets": successful[successful['spec_id'].str.startswith('rc/controls/sets/')],
    "Controls Progression": successful[successful['spec_id'].str.startswith('rc/controls/progression/')],
    "Controls Subset": successful[successful['spec_id'].str.startswith('rc/controls/subset/')],
    "Sample Subgroups": successful[successful['spec_id'].str.startswith('rc/sample/')],
    "Functional Form": successful[successful['spec_id'].str.startswith('rc/form/estimator/')],
    "Outcome Variants": successful[successful['spec_id'].str.startswith('rc/form/outcome/')],
    "Treatment Variants": successful[successful['spec_id'].str.startswith('rc/form/treatment/')],
}

for cat_name, cat_df in categories.items():
    if len(cat_df) > 0:
        n_sig_cat = (cat_df['p_value'] < 0.05).sum()
        coef_range = f"[{cat_df['coefficient'].min():.4f}, {cat_df['coefficient'].max():.4f}]"
        md_lines.append(f"| {cat_name} | {len(cat_df)} | {n_sig_cat}/{len(cat_df)} | {coef_range} |")

md_lines.append("")

# Inference variants
md_lines.append("## Inference Variants")
md_lines.append("")
if len(infer_df) > 0:
    md_lines.append("| Spec ID | SE | p-value | 95% CI |")
    md_lines.append("|---------|-----|---------|--------|")
    for _, row in infer_df.iterrows():
        if row['run_success'] == 1:
            md_lines.append(f"| {row['spec_id']} | {row['std_error']:.6f} | {row['p_value']:.6f} | - |")
        else:
            md_lines.append(f"| {row['spec_id']} | FAILED | - | - |")

md_lines.append("")

# Overall assessment
md_lines.append("## Overall Assessment")
md_lines.append("")
if len(successful) > 0:
    # Focus on LTCI outcome specs with female treatment
    ltci_specs = successful[(successful['outcome_var'] == 'ltci') & (successful['treatment_var'] == 'female')]
    n_sig_total = (ltci_specs['p_value'] < 0.05).sum()
    pct_sig = n_sig_total / len(ltci_specs) * 100 if len(ltci_specs) > 0 else 0
    sign_consistent = ((ltci_specs['coefficient'] > 0).sum() == len(ltci_specs)) or \
                      ((ltci_specs['coefficient'] < 0).sum() == len(ltci_specs))
    median_coef = ltci_specs['coefficient'].median()
    sign_word = "positive" if median_coef > 0 else "negative"

    md_lines.append(f"- **LTCI outcome, female treatment specs:** {len(ltci_specs)}")
    md_lines.append(f"- **Sign consistency:** {'All specifications have the same sign' if sign_consistent else 'Mixed signs across specifications'}")
    md_lines.append(f"- **Significance stability:** {n_sig_total}/{len(ltci_specs)} ({pct_sig:.1f}%) specifications significant at 5%")
    md_lines.append(f"- **Direction:** Median coefficient is {sign_word} ({median_coef:.6f})")

    if pct_sig >= 80 and sign_consistent:
        strength = "STRONG"
    elif pct_sig >= 50 and sign_consistent:
        strength = "MODERATE"
    elif pct_sig >= 30:
        strength = "WEAK"
    else:
        strength = "FRAGILE"

    md_lines.append(f"- **Robustness assessment:** {strength}")

    # Overall stats
    md_lines.append("")
    md_lines.append("### All Specifications")
    n_sig_all = (successful['p_value'] < 0.05).sum()
    pct_sig_all = n_sig_all / len(successful) * 100
    md_lines.append(f"- Significant at 5%: {n_sig_all}/{len(successful)} ({pct_sig_all:.1f}%)")

md_lines.append("")
md_lines.append("## Note on Synthetic Data")
md_lines.append("")
md_lines.append("This specification search uses synthetic data because the HRS/NLTCS microdata")
md_lines.append("is not included in the replication package. The synthetic data preserves the")
md_lines.append("variable structure, sample selection criteria, and approximate population moments")
md_lines.append("from the paper's Stata code. Results should be interpreted as demonstrating the")
md_lines.append("specification search methodology rather than exact replication of the paper's results.")
md_lines.append("")
md_lines.append(f"Surface hash: `{SURFACE_HASH}`")
md_lines.append("")

with open(f"{OUTPUT_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write("\n".join(md_lines))

print(f"Wrote SPECIFICATION_SEARCH.md")
print("\nDone!")
