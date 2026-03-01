"""
Specification Search Script for Hendricks & Schoellman (2018)
"Human Capital and Development Accounting: New Evidence from Immigrant Earnings"
Quarterly Journal of Economics, 133(2), 665-700.

Paper ID: 146041-V1

Surface-driven execution:
  - G1: Cross-country elasticity of aggregate skill quality (AQ) with respect to
    income per worker. Bivariate cross-country OLS (no controls).
  - Main axis of variation: outcome variable construction (sigma, wage measure,
    labor supply, skill threshold, sample, sector).
  - Baseline: Table 2 Row 1 Col 3 -- log(irAQ53_dum_skti_hrs_secall) ~ l_y,
    micro sample (N~12).

Outputs:
  - specification_results.csv (baseline, rc/* rows)
  - inference_results.csv (infer/* rows)
  - SPECIFICATION_SEARCH.md
"""

import pandas as pd
import numpy as np
import json
import sys
import warnings
import traceback
warnings.filterwarnings('ignore')

sys.path.insert(0, "scripts")
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash, software_block
)

PAPER_ID = "146041-V1"
DATA_DIR = "data/downloads/extracted/146041-V1"
REPLICATION_DIR = f"{DATA_DIR}/Replication"
OUTPUT_DIR = DATA_DIR

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

# Design audit from surface
G1 = surface_obj["baseline_groups"][0]
G1_DESIGN_AUDIT = G1["design_audit"]
G1_INFERENCE_CANONICAL = G1["inference_plan"]["canonical"]

CANONICAL_INFERENCE = {
    "spec_id": G1_INFERENCE_CANONICAL["spec_id"],
    "params": G1_INFERENCE_CANONICAL.get("params", {})
}

# ============================================================================
# Data loading
# ============================================================================

df_raw = pd.read_stata(f"{REPLICATION_DIR}/temp/Q_origins.dta")

# Precompute useful subsets and log-transform outcome variables
df = df_raw.copy()

# Create all log-transformed variables we need
log_vars_to_create = {
    # Table 2 Row 1: baseline dummies (sigma=1.5, 2.0, 1.3)
    'l_irAQ53_dum_skti_hrs_secall': 'irAQ53_dum_skti_hrs_secall',
    'l_irAQ53rh_dum_skti_hrs_secall': 'irAQ53rh_dum_skti_hrs_secall',
    'l_irAQ53rl_dum_skti_hrs_secall': 'irAQ53rl_dum_skti_hrs_secall',
    # Table 2 Row 1 Cols 1-2: wage ratio and quantity ratio
    'l_wrat53_dum_skti_secall': 'wrat53_dum_skti_secall',
    'l_H5L3_dum_skti_hrs_secall': 'H5L3_dum_skti_hrs_secall',
    # Table 2 Row 2: experience+gender adjusted
    'l_irAQ53_dumx_skti_hrs_secall': 'irAQ53_dumx_skti_hrs_secall',
    'l_irAQ53rh_dumx_skti_hrs_secall': 'irAQ53rh_dumx_skti_hrs_secall',
    'l_irAQ53rl_dumx_skti_hrs_secall': 'irAQ53rl_dumx_skti_hrs_secall',
    'l_wrat53_dumx_skti_secall': 'wrat53_dumx_skti_secall',
    # Table 2 Row 3-4: self-employment
    'l_irAQ53_dumse_skti_hrs_secall': 'irAQ53_dumse_skti_hrs_secall',
    'l_irAQ53rh_dumse_skti_hrs_secall': 'irAQ53rh_dumse_skti_hrs_secall',
    'l_irAQ53rl_dumse_skti_hrs_secall': 'irAQ53rl_dumse_skti_hrs_secall',
    'l_wrat53_dumse_skti_secall': 'wrat53_dumse_skti_secall',
    # Table 2 Rows 5-8: sector subsamples
    'l_irAQ53_dum_skti_hrs_sec1': 'irAQ53_dum_skti_hrs_sec1',
    'l_irAQ53_dum_skti_hrs_sec2': 'irAQ53_dum_skti_hrs_sec2',
    'l_irAQ53_dum_skti_hrs_sec3': 'irAQ53_dum_skti_hrs_sec3',
    'l_irAQ53_dum_skti_hrs_sec4': 'irAQ53_dum_skti_hrs_sec4',
    'l_irAQ53rh_dum_skti_hrs_sec1': 'irAQ53rh_dum_skti_hrs_sec1',
    'l_irAQ53rh_dum_skti_hrs_sec2': 'irAQ53rh_dum_skti_hrs_sec2',
    'l_irAQ53rh_dum_skti_hrs_sec3': 'irAQ53rh_dum_skti_hrs_sec3',
    'l_irAQ53rh_dum_skti_hrs_sec4': 'irAQ53rh_dum_skti_hrs_sec4',
    'l_irAQ53rl_dum_skti_hrs_sec1': 'irAQ53rl_dum_skti_hrs_sec1',
    'l_irAQ53rl_dum_skti_hrs_sec2': 'irAQ53rl_dum_skti_hrs_sec2',
    'l_irAQ53rl_dum_skti_hrs_sec3': 'irAQ53rl_dum_skti_hrs_sec3',
    'l_irAQ53rl_dum_skti_hrs_sec4': 'irAQ53rl_dum_skti_hrs_sec4',
    'l_wrat53_dum_skti_sec1': 'wrat53_dum_skti_sec1',
    'l_wrat53_dum_skti_sec2': 'wrat53_dum_skti_sec2',
    'l_wrat53_dum_skti_sec3': 'wrat53_dum_skti_sec3',
    'l_wrat53_dum_skti_sec4': 'wrat53_dum_skti_sec4',
    'l_H5L3_dum_skti_hrs_sec1': 'H5L3_dum_skti_hrs_sec1' if 'H5L3_dum_skti_hrs_sec1' in df.columns else None,
    # Alternative labor supply: bodies, population, mincerian
    'l_irAQ53_dum_skti_bod_secall': 'irAQ53_dum_skti_bod_secall',
    'l_irAQ53_dum_skti_pop_secall': 'irAQ53_dum_skti_pop_secall',
    'l_irAQ53_minc_skti_pop_secall': 'irAQ53_minc_skti_pop_secall',
    # Alternative skill thresholds
    'l_irAQ52_dum_sksc_hrs_secall': 'irAQ52_dum_sksc_hrs_secall',
    'l_irAQ53_dum_sktc_hrs_secall': 'irAQ53_dum_sktc_hrs_secall',
    # Barro-Lee sample AQ
    'l_irAQ53_blee_skti': 'irAQ53_blee_skti',
    'l_irAQ53rh_blee_skti': 'irAQ53rh_blee_skti',
    'l_irAQ53rl_blee_skti': 'irAQ53rl_blee_skti',
    # Barro-Lee H/L
    'l_H5L3_blee_skti': 'H5L3_blee_skti',
    # H5L3 for alt skill threshold
    'l_H5L3_dum_sktc_hrs_secall': 'H5L3_dum_sktc_hrs_secall',
    'l_H5L2_dum_sksc_hrs_secall': 'H5L2_dum_sksc_hrs_secall',
    # Wrat for alt skill thresholds
    'l_wrat53_dum_sksc_secall': 'wrat53_dum_sksc_secall',
    'l_wrat53_dum_sktc_secall': 'wrat53_dum_sktc_secall',
    # Wrat for Barro-Lee
    'l_wrat53_blee': 'wrat53_blee',
}

for log_name, raw_name in log_vars_to_create.items():
    if raw_name is not None and raw_name in df.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            df[log_name] = np.log(df[raw_name].astype(float))
            df.loc[~np.isfinite(df[log_name]), log_name] = np.nan

# ============================================================================
# Helper: run bivariate OLS with HC1 or homoskedastic SEs
# ============================================================================

from scipy import stats as scipy_stats


def run_bivariate_ols(data, outcome_col, treatment_col='l_y', vcov='hc1'):
    """
    Run bivariate OLS: outcome_col ~ treatment_col
    Returns dict with coefficient, se, p_value, ci_lower, ci_upper, n_obs, r_squared, coefficients_dict
    vcov: 'hc1' for HC1 robust, 'hc3' for HC3, 'ols' for homoskedastic
    """
    valid = data[[outcome_col, treatment_col]].dropna()
    n = len(valid)
    if n < 3:
        raise ValueError(f"Too few observations: {n}")

    y = valid[outcome_col].values.astype(float)
    x = valid[treatment_col].values.astype(float)
    X = np.column_stack([np.ones(n), x])

    # OLS estimation
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    yhat = X @ beta
    e = y - yhat
    k = 2

    # R-squared
    ss_res = np.sum(e**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Standard errors
    if vcov == 'ols':
        s2 = ss_res / (n - k)
        V = s2 * XtX_inv
    elif vcov == 'hc1':
        # HC1: (n/(n-k)) * sum(e_i^2 * x_i * x_i')
        meat = np.zeros((k, k))
        for i in range(n):
            xi = X[i, :].reshape(-1, 1)
            meat += (e[i]**2) * (xi @ xi.T)
        meat *= n / (n - k)
        V = XtX_inv @ meat @ XtX_inv
    elif vcov == 'hc3':
        # HC3: sum(e_i^2 / (1-h_ii)^2 * x_i * x_i')
        H = X @ XtX_inv @ X.T
        h = np.diag(H)
        meat = np.zeros((k, k))
        for i in range(n):
            xi = X[i, :].reshape(-1, 1)
            meat += (e[i]**2 / (1 - h[i])**2) * (xi @ xi.T)
        V = XtX_inv @ meat @ XtX_inv
    else:
        raise ValueError(f"Unknown vcov: {vcov}")

    se = np.sqrt(np.diag(V))
    t_stat = beta[1] / se[1]
    # Two-sided p-value using t distribution
    p_value = 2 * scipy_stats.t.sf(abs(t_stat), df=n - k)

    # 95% confidence interval
    t_crit = scipy_stats.t.ppf(0.975, df=n - k)
    ci_lower = beta[1] - t_crit * se[1]
    ci_upper = beta[1] + t_crit * se[1]

    return {
        'coefficient': float(beta[1]),
        'std_error': float(se[1]),
        'p_value': float(p_value),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'n_obs': int(n),
        'r_squared': float(r2),
        'coefficients': {
            '_cons': float(beta[0]),
            treatment_col: float(beta[1]),
        },
        'intercept': float(beta[0]),
        'intercept_se': float(se[0]),
    }


# ============================================================================
# Spec runner infrastructure
# ============================================================================

spec_results = []
inference_results = []
spec_counter = 0
infer_counter = 0


def make_spec_run_id():
    global spec_counter
    spec_counter += 1
    return f"{PAPER_ID}__spec_{spec_counter:03d}"


def make_infer_run_id():
    global infer_counter
    infer_counter += 1
    return f"{PAPER_ID}__infer_{infer_counter:03d}"


def run_spec(spec_id, spec_tree_path, outcome_var, treatment_var, data,
             sample_desc, baseline_group_id="G1",
             axis_block_name=None, axis_block=None,
             functional_form=None, extra=None):
    """Run one specification and append to spec_results."""
    run_id = make_spec_run_id()
    try:
        res = run_bivariate_ols(data, outcome_var, treatment_var, vcov='hc1')
        payload = make_success_payload(
            coefficients=res['coefficients'],
            inference=CANONICAL_INFERENCE,
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"cross_sectional_ols": G1_DESIGN_AUDIT},
            axis_block_name=axis_block_name,
            axis_block=axis_block,
            extra=extra,
        )
        if functional_form:
            payload["functional_form"] = functional_form

        spec_results.append({
            'paper_id': PAPER_ID,
            'spec_run_id': run_id,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'baseline_group_id': baseline_group_id,
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': res['coefficient'],
            'std_error': res['std_error'],
            'p_value': res['p_value'],
            'ci_lower': res['ci_lower'],
            'ci_upper': res['ci_upper'],
            'n_obs': res['n_obs'],
            'r_squared': res['r_squared'],
            'coefficient_vector_json': json.dumps(payload),
            'sample_desc': sample_desc,
            'fixed_effects': '',
            'controls_desc': 'none (bivariate)',
            'cluster_var': '',
            'run_success': 1,
            'run_error': '',
        })
        return run_id, res
    except Exception as e:
        err_msg = str(e)
        err_details = error_details_from_exception(e, stage="estimation")
        fail_payload = make_failure_payload(
            error=err_msg,
            error_details=err_details,
            inference=CANONICAL_INFERENCE,
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
        )
        spec_results.append({
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
            'coefficient_vector_json': json.dumps(fail_payload),
            'sample_desc': sample_desc,
            'fixed_effects': '',
            'controls_desc': 'none (bivariate)',
            'cluster_var': '',
            'run_success': 0,
            'run_error': err_msg,
        })
        return run_id, None


def run_inference_variant(base_run_id, spec_id, spec_tree_path, outcome_var,
                          treatment_var, data, sample_desc, vcov_type,
                          baseline_group_id="G1"):
    """Run one inference variant and append to inference_results."""
    run_id = make_infer_run_id()
    try:
        res = run_bivariate_ols(data, outcome_var, treatment_var, vcov=vcov_type)
        infer_info = {"spec_id": spec_id, "params": {}}
        payload = make_success_payload(
            coefficients=res['coefficients'],
            inference=infer_info,
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"cross_sectional_ols": G1_DESIGN_AUDIT},
        )
        inference_results.append({
            'paper_id': PAPER_ID,
            'inference_run_id': run_id,
            'spec_run_id': base_run_id,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'baseline_group_id': baseline_group_id,
            'coefficient': res['coefficient'],
            'std_error': res['std_error'],
            'p_value': res['p_value'],
            'ci_lower': res['ci_lower'],
            'ci_upper': res['ci_upper'],
            'n_obs': res['n_obs'],
            'r_squared': res['r_squared'],
            'coefficient_vector_json': json.dumps(payload),
            'run_success': 1,
            'run_error': '',
        })
    except Exception as e:
        err_msg = str(e)
        err_details = error_details_from_exception(e, stage="inference")
        fail_payload = make_failure_payload(
            error=err_msg,
            error_details=err_details,
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
        )
        inference_results.append({
            'paper_id': PAPER_ID,
            'inference_run_id': run_id,
            'spec_run_id': base_run_id,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'baseline_group_id': baseline_group_id,
            'coefficient': np.nan,
            'std_error': np.nan,
            'p_value': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n_obs': np.nan,
            'r_squared': np.nan,
            'coefficient_vector_json': json.dumps(fail_payload),
            'run_success': 0,
            'run_error': err_msg,
        })


# ============================================================================
# Define samples
# ============================================================================

# Micro sample: sample_micro==1, N~12
micro = df[df['sample_micro'] == 1].copy()

# Self-employment subsample of micro: where irAQ53_dumse_skti_hrs_secall is non-null
se_sample = micro[micro['irAQ53_dumse_skti_hrs_secall'].notna()].copy()

# US Pooled sample: sample=='US Pooled' & sample_migr==1, N~101
us_pooled = df[(df['sample'] == 'US Pooled') & (df['sample_migr'] == 1)].copy()

# All countries (for pooled regressions where country_obs is implicitly 1 in Q_origins)
all_countries = df.copy()

# ============================================================================
# STEP 1: Baseline specification
# ============================================================================

print("=" * 60)
print("STEP 1: Baseline specification")
print("=" * 60)

# Table 2 Row 1 Col 3: log(irAQ53_dum_skti_hrs_secall) ~ l_y, micro sample
baseline_run_id, baseline_res = run_spec(
    spec_id="baseline",
    spec_tree_path="specification_tree/methods/cross_sectional_ols.md",
    outcome_var="l_irAQ53_dum_skti_hrs_secall",
    treatment_var="l_y",
    data=micro,
    sample_desc="sample_micro==1, N~12 countries (Table 2 Row 1 Col 3)",
)

if baseline_res:
    print(f"  Baseline: coef={baseline_res['coefficient']:.4f}, "
          f"se={baseline_res['std_error']:.4f}, "
          f"p={baseline_res['p_value']:.4f}, N={baseline_res['n_obs']}")

# ============================================================================
# Additional baseline specs listed in core_universe.baseline_spec_ids
# ============================================================================

print("\n" + "=" * 60)
print("Additional baseline specs")
print("=" * 60)

# baseline__aq_sigma_high: AQ with sigma=2.0
run_spec(
    spec_id="baseline__aq_sigma_high",
    spec_tree_path="specification_tree/methods/cross_sectional_ols.md",
    outcome_var="l_irAQ53rh_dum_skti_hrs_secall",
    treatment_var="l_y",
    data=micro,
    sample_desc="sample_micro==1, sigma=2.0 (high)",
    functional_form={"interpretation": "AQ with sigma=2.0 (sigma_high)"},
)

# baseline__aq_sigma_low: AQ with sigma=1.3
run_spec(
    spec_id="baseline__aq_sigma_low",
    spec_tree_path="specification_tree/methods/cross_sectional_ols.md",
    outcome_var="l_irAQ53rl_dum_skti_hrs_secall",
    treatment_var="l_y",
    data=micro,
    sample_desc="sample_micro==1, sigma=1.3 (low)",
    functional_form={"interpretation": "AQ with sigma=1.3 (sigma_low)"},
)

# baseline__wage_ratio: wage ratio (Column 1 of Table 2)
run_spec(
    spec_id="baseline__wage_ratio",
    spec_tree_path="specification_tree/methods/cross_sectional_ols.md",
    outcome_var="l_wrat53_dum_skti_secall",
    treatment_var="l_y",
    data=micro,
    sample_desc="sample_micro==1, wage ratio (Table 2 Col 1)",
    functional_form={"interpretation": "log wage ratio (5-3 education groups) on l_y"},
)

# baseline__quantity_ratio: H5/L3 quantity ratio (Column 2 of Table 2)
run_spec(
    spec_id="baseline__quantity_ratio",
    spec_tree_path="specification_tree/methods/cross_sectional_ols.md",
    outcome_var="l_H5L3_dum_skti_hrs_secall",
    treatment_var="l_y",
    data=micro,
    sample_desc="sample_micro==1, quantity ratio H5/L3 (Table 2 Col 2)",
    functional_form={"interpretation": "log quantity ratio H5/L3 on l_y"},
)

# baseline__pooled_countries: AQ Barro-Lee on broader sample
run_spec(
    spec_id="baseline__pooled_countries",
    spec_tree_path="specification_tree/methods/cross_sectional_ols.md",
    outcome_var="l_irAQ53_blee_skti",
    treatment_var="l_y",
    data=all_countries,
    sample_desc="All countries with Barro-Lee AQ, N~92 (Table 3 Row 2 Col 2 equivalent)",
    functional_form={"interpretation": "AQ Barro-Lee on broader country sample"},
)

# baseline__pooled_bilateral: not directly available as separate AQ variable,
# but the pooled bilateral Q variable exists. The AQ is the same (irAQ53_blee_skti).
# This is Table 3 Row 3 which uses the same AQ sample but different Q.
# Since AQ doesn't change, the AQ regression is the same as Row 2.
# We still run it to be faithful to the surface.
run_spec(
    spec_id="baseline__pooled_bilateral",
    spec_tree_path="specification_tree/methods/cross_sectional_ols.md",
    outcome_var="l_irAQ53_blee_skti",
    treatment_var="l_y",
    data=all_countries,
    sample_desc="All countries, Barro-Lee AQ (Table 3 Row 3 equivalent -- same AQ as Row 2)",
    functional_form={"interpretation": "AQ Barro-Lee, bilateral Q controls (AQ unchanged)"},
)

# ============================================================================
# STEP 2: RC/* variants -- Outcome construction axes
# ============================================================================

print("\n" + "=" * 60)
print("STEP 2: RC/* variants")
print("=" * 60)

# ---------- rc/form/outcome variants ----------

# rc/form/outcome/aq_sigma_1p3: sigma=1.3 on micro
run_spec(
    spec_id="rc/form/outcome/aq_sigma_1p3",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    outcome_var="l_irAQ53rl_dum_skti_hrs_secall",
    treatment_var="l_y",
    data=micro,
    sample_desc="sample_micro==1, sigma=1.3",
    functional_form={"interpretation": "AQ with sigma=1.3 (lower bound CES elasticity)"},
)

# rc/form/outcome/aq_sigma_2p0: sigma=2.0 on micro
run_spec(
    spec_id="rc/form/outcome/aq_sigma_2p0",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    outcome_var="l_irAQ53rh_dum_skti_hrs_secall",
    treatment_var="l_y",
    data=micro,
    sample_desc="sample_micro==1, sigma=2.0",
    functional_form={"interpretation": "AQ with sigma=2.0 (upper bound CES elasticity)"},
)

# rc/form/outcome/aq_sigma_4p0: sigma=4.0
# sigma=4 AQ is not pre-computed. We compute it manually using the CES formula:
# AQ = ((1 + w*S)^(sigma/(sigma-1))) / ((1 + w*S*Q^((sigma-1)/sigma))^(sigma/(sigma-1)))
# This is complex -- but we can approximate from sigma=1.5 baseline.
# Actually, the data has separate sigma columns and AQ is pre-computed for sigma=1.5,2.0,1.3 only.
# sigma=4 is used in devacc_main.do for development accounting, not for AQ regression.
# There are no irAQ columns for sigma=4. This spec cannot be run from pre-computed data.
run_spec(
    spec_id="rc/form/outcome/aq_sigma_4p0",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    outcome_var="l_irAQ53_dum_skti_hrs_secall",  # placeholder
    treatment_var="l_y",
    data=pd.DataFrame({'l_irAQ53_dum_skti_hrs_secall': [], 'l_y': []}),  # empty -> will fail
    sample_desc="NOT AVAILABLE: sigma=4.0 AQ not pre-computed in data",
    functional_form={"interpretation": "AQ with sigma=4.0 (not available in pre-computed data)"},
)

# rc/form/outcome/aq_no_hours: AQ using body count instead of hours
run_spec(
    spec_id="rc/form/outcome/aq_no_hours",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    outcome_var="l_irAQ53_dum_skti_bod_secall",
    treatment_var="l_y",
    data=micro,
    sample_desc="sample_micro==1, labor supply = body count",
    functional_form={"interpretation": "AQ using body count instead of hours for labor supply"},
)

# rc/form/outcome/aq_bodies_not_hours: same as aq_no_hours (bodies)
run_spec(
    spec_id="rc/form/outcome/aq_bodies_not_hours",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    outcome_var="l_irAQ53_dum_skti_bod_secall",
    treatment_var="l_y",
    data=micro,
    sample_desc="sample_micro==1, labor supply = body count (duplicate of aq_no_hours)",
    functional_form={"interpretation": "AQ using body count for labor supply"},
)

# rc/form/outcome/aq_all_working_age: AQ using working-age population
run_spec(
    spec_id="rc/form/outcome/aq_all_working_age",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    outcome_var="l_irAQ53_dum_skti_pop_secall",
    treatment_var="l_y",
    data=micro,
    sample_desc="sample_micro==1, labor supply = working-age population",
    functional_form={"interpretation": "AQ using working-age population for labor supply"},
)

# rc/form/outcome/aq_mincerian_common: AQ with common Mincerian return
run_spec(
    spec_id="rc/form/outcome/aq_mincerian_common",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    outcome_var="l_irAQ53_minc_skti_pop_secall",
    treatment_var="l_y",
    data=micro,
    sample_desc="sample_micro==1, common Mincerian return",
    functional_form={"interpretation": "AQ using common Mincerian return across countries, pop-weighted"},
)

# rc/form/outcome/aq_exp_gender_controls: experience+gender adjusted wages
run_spec(
    spec_id="rc/form/outcome/aq_exp_gender_controls",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    outcome_var="l_irAQ53_dumx_skti_hrs_secall",
    treatment_var="l_y",
    data=micro,
    sample_desc="sample_micro==1, exp+gender adjusted wages (Table 2 Row 2)",
    functional_form={"interpretation": "AQ with experience and gender adjusted wage premia"},
)

# rc/form/outcome/aq_self_employed: self-employment AQ
run_spec(
    spec_id="rc/form/outcome/aq_self_employed",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    outcome_var="l_irAQ53_dumse_skti_hrs_secall",
    treatment_var="l_y",
    data=se_sample,
    sample_desc="self-employment subsample of micro (N~8), Table 2 Row 4",
    functional_form={"interpretation": "AQ including self-employed workers"},
)

# rc/form/outcome/wage_ratio_5_3: wage ratio
run_spec(
    spec_id="rc/form/outcome/wage_ratio_5_3",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    outcome_var="l_wrat53_dum_skti_secall",
    treatment_var="l_y",
    data=micro,
    sample_desc="sample_micro==1, wage ratio 5/3",
    functional_form={"interpretation": "log wage ratio (education groups 5 vs 3)"},
)

# rc/form/outcome/quantity_ratio_H5L3: quantity ratio
run_spec(
    spec_id="rc/form/outcome/quantity_ratio_H5L3",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    outcome_var="l_H5L3_dum_skti_hrs_secall",
    treatment_var="l_y",
    data=micro,
    sample_desc="sample_micro==1, quantity ratio H5/L3",
    functional_form={"interpretation": "log quantity ratio (high/low skill workers, hours weighted)"},
)

# rc/form/outcome/aq_threshold_sc: some-college skill threshold
run_spec(
    spec_id="rc/form/outcome/aq_threshold_sc",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    outcome_var="l_irAQ52_dum_sksc_hrs_secall",
    treatment_var="l_y",
    data=micro,
    sample_desc="sample_micro==1, some-college skill threshold",
    functional_form={"interpretation": "AQ with some-college skill threshold"},
)

# rc/form/outcome/aq_threshold_tc: tertiary-only skill threshold
run_spec(
    spec_id="rc/form/outcome/aq_threshold_tc",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    outcome_var="l_irAQ53_dum_sktc_hrs_secall",
    treatment_var="l_y",
    data=micro,
    sample_desc="sample_micro==1, tertiary-only skill threshold",
    functional_form={"interpretation": "AQ with tertiary-only skill threshold"},
)

# rc/form/outcome/aq_barro_lee: Barro-Lee AQ on broader sample
run_spec(
    spec_id="rc/form/outcome/aq_barro_lee",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    outcome_var="l_irAQ53_blee_skti",
    treatment_var="l_y",
    data=all_countries,
    sample_desc="All countries with Barro-Lee data, N~92",
    functional_form={"interpretation": "AQ using Barro-Lee education data on broader sample"},
)

# ---------- rc/data_construction variants ----------

# rc/data_construction/skill_threshold/upper_secondary: baseline (already baseline)
run_spec(
    spec_id="rc/data_construction/skill_threshold/upper_secondary",
    spec_tree_path="specification_tree/modules/robustness/data_construction.md",
    outcome_var="l_irAQ53_dum_skti_hrs_secall",
    treatment_var="l_y",
    data=micro,
    sample_desc="sample_micro==1, upper secondary threshold (baseline)",
    axis_block_name="data_construction",
    axis_block={"skill_threshold": "upper_secondary", "note": "same as baseline"},
)

# rc/data_construction/skill_threshold/some_college
run_spec(
    spec_id="rc/data_construction/skill_threshold/some_college",
    spec_tree_path="specification_tree/modules/robustness/data_construction.md",
    outcome_var="l_irAQ52_dum_sksc_hrs_secall",
    treatment_var="l_y",
    data=micro,
    sample_desc="sample_micro==1, some-college threshold",
    axis_block_name="data_construction",
    axis_block={"skill_threshold": "some_college"},
)

# rc/data_construction/skill_threshold/tertiary_only
run_spec(
    spec_id="rc/data_construction/skill_threshold/tertiary_only",
    spec_tree_path="specification_tree/modules/robustness/data_construction.md",
    outcome_var="l_irAQ53_dum_sktc_hrs_secall",
    treatment_var="l_y",
    data=micro,
    sample_desc="sample_micro==1, tertiary-only threshold",
    axis_block_name="data_construction",
    axis_block={"skill_threshold": "tertiary_only"},
)

# rc/data_construction/wage_measure/baseline_dummies: baseline (education dummies)
run_spec(
    spec_id="rc/data_construction/wage_measure/baseline_dummies",
    spec_tree_path="specification_tree/modules/robustness/data_construction.md",
    outcome_var="l_irAQ53_dum_skti_hrs_secall",
    treatment_var="l_y",
    data=micro,
    sample_desc="sample_micro==1, education dummies (baseline wage measure)",
    axis_block_name="data_construction",
    axis_block={"wage_measure": "education_dummies", "note": "same as baseline"},
)

# rc/data_construction/wage_measure/exp_gender_adjusted
run_spec(
    spec_id="rc/data_construction/wage_measure/exp_gender_adjusted",
    spec_tree_path="specification_tree/modules/robustness/data_construction.md",
    outcome_var="l_irAQ53_dumx_skti_hrs_secall",
    treatment_var="l_y",
    data=micro,
    sample_desc="sample_micro==1, exp+gender adjusted wages",
    axis_block_name="data_construction",
    axis_block={"wage_measure": "exp_gender_adjusted"},
)

# rc/data_construction/labor_supply/hours: baseline (hours-weighted)
run_spec(
    spec_id="rc/data_construction/labor_supply/hours",
    spec_tree_path="specification_tree/modules/robustness/data_construction.md",
    outcome_var="l_irAQ53_dum_skti_hrs_secall",
    treatment_var="l_y",
    data=micro,
    sample_desc="sample_micro==1, hours-weighted labor supply (baseline)",
    axis_block_name="data_construction",
    axis_block={"labor_supply": "hours", "note": "same as baseline"},
)

# rc/data_construction/labor_supply/bodies
run_spec(
    spec_id="rc/data_construction/labor_supply/bodies",
    spec_tree_path="specification_tree/modules/robustness/data_construction.md",
    outcome_var="l_irAQ53_dum_skti_bod_secall",
    treatment_var="l_y",
    data=micro,
    sample_desc="sample_micro==1, body count labor supply",
    axis_block_name="data_construction",
    axis_block={"labor_supply": "bodies"},
)

# rc/data_construction/labor_supply/population
run_spec(
    spec_id="rc/data_construction/labor_supply/population",
    spec_tree_path="specification_tree/modules/robustness/data_construction.md",
    outcome_var="l_irAQ53_dum_skti_pop_secall",
    treatment_var="l_y",
    data=micro,
    sample_desc="sample_micro==1, working-age population",
    axis_block_name="data_construction",
    axis_block={"labor_supply": "population"},
)

# ---------- rc/form/treatment variants ----------

# rc/form/treatment/log_gdp_ppp: use y_relUS_2005 (GDP PPP 2005 benchmark)
# Need to create log of y_relUS_2005 * US_y as treatment (or just use y_relUS_2005 directly)
# Actually, l_y is already log(income per worker). For alternative GDP, we'd need
# a different measure. The data has y_relUS (current PWT) and y_relUS_2005.
# Since l_y is the only treatment in the baseline, to use an alternative we need
# to construct log(y_relUS_2005 * US_level) or just use log(y_relUS_2005).
# The relative measure y_relUS/y_relUS_2005 captures a similar ranking.
# Let's use log(y_relUS_2005) as the treatment (proportional to log GDP per worker in PPP terms).
df['l_y_relUS_2005'] = np.log(df['y_relUS_2005'].astype(float))
df.loc[~np.isfinite(df['l_y_relUS_2005']), 'l_y_relUS_2005'] = np.nan
micro['l_y_relUS_2005'] = df.loc[micro.index, 'l_y_relUS_2005']
all_countries['l_y_relUS_2005'] = df.loc[all_countries.index, 'l_y_relUS_2005']

run_spec(
    spec_id="rc/form/treatment/log_gdp_ppp",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    outcome_var="l_irAQ53_dum_skti_hrs_secall",
    treatment_var="l_y_relUS_2005",
    data=micro,
    sample_desc="sample_micro==1, treatment=log(y_relUS_2005)",
    functional_form={"interpretation": "Alternative income measure: log(y relative to US, 2005 PPP benchmark)"},
)

# rc/form/treatment/log_gdp_pwt: l_y is already from PWT -- this IS the baseline
# Use log(y_relUS2000) which is from year 2000 benchmark
df['l_y_relUS2000'] = np.log(df['y_relUS2000'].astype(float))
df.loc[~np.isfinite(df['l_y_relUS2000']), 'l_y_relUS2000'] = np.nan
micro['l_y_relUS2000'] = df.loc[micro.index, 'l_y_relUS2000']

run_spec(
    spec_id="rc/form/treatment/log_gdp_pwt",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    outcome_var="l_irAQ53_dum_skti_hrs_secall",
    treatment_var="l_y_relUS2000",
    data=micro,
    sample_desc="sample_micro==1, treatment=log(y_relUS, year 2000 benchmark)",
    functional_form={"interpretation": "Alternative income measure: log(y relative to US, 2000 benchmark from PWT)"},
)

# ---------- rc/controls/add/continent_dummies ----------
# This is a cross-country regression with ~12 obs. Adding continent dummies is
# not meaningful with so few observations, but we can try.
# We need to construct continent from country names.
continent_map = {
    'Brazil': 'South America', 'Canada': 'North America', 'Indonesia': 'Asia',
    'India': 'Asia', 'Israel': 'Asia', 'Jamaica': 'Caribbean',
    'Mexico': 'North America', 'Panama': 'Central America',
    'Trinidad and Tobago': 'Caribbean', 'Uruguay': 'South America',
    'United States': 'North America', 'Venezuela': 'South America',
}
micro['continent'] = micro['country'].map(continent_map)
# With only 12 obs and 5 continents, this eats too many df. Try a simplified version.
# Americas vs Asia
micro['americas'] = (micro['continent'].isin(['North America', 'South America',
                                                'Central America', 'Caribbean'])).astype(int)

# Run with Americas dummy as control
from scipy import stats as sp_stats


def run_ols_with_controls(data, outcome_col, treatment_col, control_cols, vcov='hc1'):
    """Run OLS with controls: outcome ~ treatment + controls."""
    all_cols = [outcome_col, treatment_col] + control_cols
    valid = data[all_cols].dropna()
    n = len(valid)
    if n < len(all_cols) + 1:
        raise ValueError(f"Too few observations: {n} for {len(all_cols)} regressors")

    y = valid[outcome_col].values.astype(float)
    X_list = [np.ones(n), valid[treatment_col].values.astype(float)]
    for c in control_cols:
        X_list.append(valid[c].values.astype(float))
    X = np.column_stack(X_list)
    k = X.shape[1]

    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    yhat = X @ beta
    e = y - yhat

    ss_res = np.sum(e**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    if vcov == 'hc1':
        meat = np.zeros((k, k))
        for i in range(n):
            xi = X[i, :].reshape(-1, 1)
            meat += (e[i]**2) * (xi @ xi.T)
        meat *= n / (n - k)
        V = XtX_inv @ meat @ XtX_inv
    elif vcov == 'hc3':
        H = X @ XtX_inv @ X.T
        h = np.diag(H)
        meat = np.zeros((k, k))
        for i in range(n):
            xi = X[i, :].reshape(-1, 1)
            meat += (e[i]**2 / (1 - h[i])**2) * (xi @ xi.T)
        V = XtX_inv @ meat @ XtX_inv
    else:
        s2 = ss_res / (n - k)
        V = s2 * XtX_inv

    se = np.sqrt(np.diag(V))
    t_stat = beta[1] / se[1]
    p_value = 2 * sp_stats.t.sf(abs(t_stat), df=n - k)
    t_crit = sp_stats.t.ppf(0.975, df=n - k)
    ci_lower = beta[1] - t_crit * se[1]
    ci_upper = beta[1] + t_crit * se[1]

    coef_dict = {'_cons': float(beta[0]), treatment_col: float(beta[1])}
    for idx, c in enumerate(control_cols):
        coef_dict[c] = float(beta[2 + idx])

    return {
        'coefficient': float(beta[1]),
        'std_error': float(se[1]),
        'p_value': float(p_value),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'n_obs': int(n),
        'r_squared': float(r2),
        'coefficients': coef_dict,
    }


# Run continent dummies spec
try:
    res_cont = run_ols_with_controls(micro, 'l_irAQ53_dum_skti_hrs_secall', 'l_y', ['americas'], vcov='hc1')
    run_id_cont = make_spec_run_id()
    payload_cont = make_success_payload(
        coefficients=res_cont['coefficients'],
        inference=CANONICAL_INFERENCE,
        software=SW_BLOCK,
        surface_hash=SURFACE_HASH,
        design={"cross_sectional_ols": G1_DESIGN_AUDIT},
        axis_block_name="controls",
        axis_block={"added": ["americas_dummy"], "n_controls": 1, "note": "Americas vs non-Americas dummy (continent)"},
    )
    spec_results.append({
        'paper_id': PAPER_ID,
        'spec_run_id': run_id_cont,
        'spec_id': 'rc/controls/add/continent_dummies',
        'spec_tree_path': 'specification_tree/modules/robustness/controls.md',
        'baseline_group_id': 'G1',
        'outcome_var': 'l_irAQ53_dum_skti_hrs_secall',
        'treatment_var': 'l_y',
        'coefficient': res_cont['coefficient'],
        'std_error': res_cont['std_error'],
        'p_value': res_cont['p_value'],
        'ci_lower': res_cont['ci_lower'],
        'ci_upper': res_cont['ci_upper'],
        'n_obs': res_cont['n_obs'],
        'r_squared': res_cont['r_squared'],
        'coefficient_vector_json': json.dumps(payload_cont),
        'sample_desc': 'sample_micro==1, with Americas dummy control',
        'fixed_effects': '',
        'controls_desc': 'americas_dummy',
        'cluster_var': '',
        'run_success': 1,
        'run_error': '',
    })
except Exception as e:
    run_id_cont = make_spec_run_id()
    spec_results.append({
        'paper_id': PAPER_ID,
        'spec_run_id': run_id_cont,
        'spec_id': 'rc/controls/add/continent_dummies',
        'spec_tree_path': 'specification_tree/modules/robustness/controls.md',
        'baseline_group_id': 'G1',
        'outcome_var': 'l_irAQ53_dum_skti_hrs_secall',
        'treatment_var': 'l_y',
        'coefficient': np.nan, 'std_error': np.nan, 'p_value': np.nan,
        'ci_lower': np.nan, 'ci_upper': np.nan,
        'n_obs': np.nan, 'r_squared': np.nan,
        'coefficient_vector_json': json.dumps(make_failure_payload(
            error=str(e), error_details=error_details_from_exception(e, stage="estimation"),
            software=SW_BLOCK, surface_hash=SURFACE_HASH)),
        'sample_desc': 'sample_micro==1, with continent dummy',
        'fixed_effects': '', 'controls_desc': 'americas_dummy', 'cluster_var': '',
        'run_success': 0, 'run_error': str(e),
    })

# ---------- rc/sample/subgroup variants ----------

# rc/sample/subgroup/us_immigrants_only: Table 3 Row 1 (US Pooled, micro AQ)
run_spec(
    spec_id="rc/sample/subgroup/us_immigrants_only",
    spec_tree_path="specification_tree/modules/robustness/sample.md",
    outcome_var="l_irAQ53_dum_skti_hrs_secall",
    treatment_var="l_y",
    data=us_pooled,
    sample_desc="US Pooled + sample_migr==1, micro AQ (Table 3 Row 1 Col 6)",
    axis_block_name="sample",
    axis_block={"subgroup": "us_immigrants_only"},
)

# rc/sample/subgroup/pooled_countries: all unique countries, Barro-Lee AQ
run_spec(
    spec_id="rc/sample/subgroup/pooled_countries",
    spec_tree_path="specification_tree/modules/robustness/sample.md",
    outcome_var="l_irAQ53_blee_skti",
    treatment_var="l_y",
    data=all_countries,
    sample_desc="All countries, Barro-Lee AQ (Table 3 Row 2)",
    axis_block_name="sample",
    axis_block={"subgroup": "pooled_countries"},
)

# rc/sample/subgroup/pooled_bilateral_controls
run_spec(
    spec_id="rc/sample/subgroup/pooled_bilateral_controls",
    spec_tree_path="specification_tree/modules/robustness/sample.md",
    outcome_var="l_irAQ53_blee_skti",
    treatment_var="l_y",
    data=all_countries,
    sample_desc="All countries, Barro-Lee AQ, bilateral Q controls (Table 3 Row 3)",
    axis_block_name="sample",
    axis_block={"subgroup": "pooled_bilateral_controls"},
)

# rc/sample/subgroup/micro_sample_only: just the micro (same as baseline)
run_spec(
    spec_id="rc/sample/subgroup/micro_sample_only",
    spec_tree_path="specification_tree/modules/robustness/sample.md",
    outcome_var="l_irAQ53_dum_skti_hrs_secall",
    treatment_var="l_y",
    data=micro,
    sample_desc="sample_micro==1 (same as baseline, explicit)",
    axis_block_name="sample",
    axis_block={"subgroup": "micro_sample_only", "note": "identical to baseline"},
)

# rc/sample/subgroup/self_employment: self-employment subsample
run_spec(
    spec_id="rc/sample/subgroup/self_employment",
    spec_tree_path="specification_tree/modules/robustness/sample.md",
    outcome_var="l_irAQ53_dum_skti_hrs_secall",
    treatment_var="l_y",
    data=se_sample,
    sample_desc="Self-employment subsample of micro (N~8), Table 2 Row 3",
    axis_block_name="sample",
    axis_block={"subgroup": "self_employment"},
)

# rc/sample/subgroup/agriculture: sector 1
run_spec(
    spec_id="rc/sample/subgroup/agriculture",
    spec_tree_path="specification_tree/modules/robustness/sample.md",
    outcome_var="l_irAQ53_dum_skti_hrs_sec1",
    treatment_var="l_y",
    data=micro,
    sample_desc="sample_micro==1, agriculture sector (Table 2 Row 5)",
    axis_block_name="sample",
    axis_block={"subgroup": "agriculture", "sector": 1},
)

# rc/sample/subgroup/manufacturing: sector 2
run_spec(
    spec_id="rc/sample/subgroup/manufacturing",
    spec_tree_path="specification_tree/modules/robustness/sample.md",
    outcome_var="l_irAQ53_dum_skti_hrs_sec2",
    treatment_var="l_y",
    data=micro,
    sample_desc="sample_micro==1, manufacturing sector (Table 2 Row 6)",
    axis_block_name="sample",
    axis_block={"subgroup": "manufacturing", "sector": 2},
)

# rc/sample/subgroup/lowskill_services: sector 3
run_spec(
    spec_id="rc/sample/subgroup/lowskill_services",
    spec_tree_path="specification_tree/modules/robustness/sample.md",
    outcome_var="l_irAQ53_dum_skti_hrs_sec3",
    treatment_var="l_y",
    data=micro,
    sample_desc="sample_micro==1, low-skill services sector (Table 2 Row 7)",
    axis_block_name="sample",
    axis_block={"subgroup": "lowskill_services", "sector": 3},
)

# rc/sample/subgroup/highskill_services: sector 4
run_spec(
    spec_id="rc/sample/subgroup/highskill_services",
    spec_tree_path="specification_tree/modules/robustness/sample.md",
    outcome_var="l_irAQ53_dum_skti_hrs_sec4",
    treatment_var="l_y",
    data=micro,
    sample_desc="sample_micro==1, high-skill services sector (Table 2 Row 8)",
    axis_block_name="sample",
    axis_block={"subgroup": "highskill_services", "sector": 4},
)

# rc/sample/subgroup/selection_adjusted: Table 3 Row 4 (selection adjusted Q)
# The AQ regression uses the same irAQ53_dum_skti_hrs_secall; only Q changes.
# For the AQ column regression, the result is unchanged from US pooled baseline.
# We run it on the US Pooled sample (same as Table 3 Row 1 for AQ columns).
run_spec(
    spec_id="rc/sample/subgroup/selection_adjusted",
    spec_tree_path="specification_tree/modules/robustness/sample.md",
    outcome_var="l_irAQ53_dum_skti_hrs_secall",
    treatment_var="l_y",
    data=us_pooled,
    sample_desc="US Pooled, selection-adjusted Q (Table 3 Row 4, AQ col unchanged)",
    axis_block_name="sample",
    axis_block={"subgroup": "selection_adjusted", "note": "AQ variable unchanged from baseline; Q variant only"},
)

# rc/sample/subgroup/excl_outlier_countries: exclude extreme income countries
# Identify outliers by l_y percentiles in micro sample
ly_micro = micro['l_y'].dropna()
p5, p95 = ly_micro.quantile(0.05), ly_micro.quantile(0.95)
micro_no_outlier = micro[(micro['l_y'] >= p5) & (micro['l_y'] <= p95)]

run_spec(
    spec_id="rc/sample/subgroup/excl_outlier_countries",
    spec_tree_path="specification_tree/modules/robustness/sample.md",
    outcome_var="l_irAQ53_dum_skti_hrs_secall",
    treatment_var="l_y",
    data=micro_no_outlier,
    sample_desc=f"sample_micro==1 excl income outliers (l_y in [{p5:.2f}, {p95:.2f}])",
    axis_block_name="sample",
    axis_block={"subgroup": "excl_outlier_countries", "l_y_range": [float(p5), float(p95)]},
)

# rc/sample/subgroup/broad_barrolee_sample: broad Barro-Lee sample
run_spec(
    spec_id="rc/sample/subgroup/broad_barrolee_sample",
    spec_tree_path="specification_tree/modules/robustness/sample.md",
    outcome_var="l_irAQ53_blee_skti",
    treatment_var="l_y",
    data=all_countries,
    sample_desc="Broad Barro-Lee sample, N~92",
    axis_block_name="sample",
    axis_block={"subgroup": "broad_barrolee_sample"},
)

# ---------- rc/sample/restriction variants (Table 3 Q adjustments) ----------
# These Table 3 rows change the Q estimation but the AQ variable is the same.
# For the AQ regression, the result is identical across rows.
# We still register them as separate specs for completeness.

# rc/sample/restriction/10plus_years
run_spec(
    spec_id="rc/sample/restriction/10plus_years",
    spec_tree_path="specification_tree/modules/robustness/sample.md",
    outcome_var="l_irAQ53_dum_skti_hrs_secall",
    treatment_var="l_y",
    data=us_pooled,
    sample_desc="US Pooled, 10+ years in US Q (Table 3 Row 5, AQ col unchanged)",
    axis_block_name="sample",
    axis_block={"restriction": "10plus_years", "note": "Q variant; AQ outcome unchanged"},
)

# rc/sample/restriction/good_english
run_spec(
    spec_id="rc/sample/restriction/good_english",
    spec_tree_path="specification_tree/modules/robustness/sample.md",
    outcome_var="l_irAQ53_dum_skti_hrs_secall",
    treatment_var="l_y",
    data=us_pooled,
    sample_desc="US Pooled, good English Q (Table 3 Row 6, AQ col unchanged)",
    axis_block_name="sample",
    axis_block={"restriction": "good_english"},
)

# rc/sample/restriction/no_downgrading
run_spec(
    spec_id="rc/sample/restriction/no_downgrading",
    spec_tree_path="specification_tree/modules/robustness/sample.md",
    outcome_var="l_irAQ53_dum_skti_hrs_secall",
    treatment_var="l_y",
    data=us_pooled,
    sample_desc="US Pooled, no skill downgrading Q (Table 3 Row 7, AQ col unchanged)",
    axis_block_name="sample",
    axis_block={"restriction": "no_downgrading"},
)

# rc/sample/restriction/no_mismatch
run_spec(
    spec_id="rc/sample/restriction/no_mismatch",
    spec_tree_path="specification_tree/modules/robustness/sample.md",
    outcome_var="l_irAQ53_dum_skti_hrs_secall",
    treatment_var="l_y",
    data=us_pooled,
    sample_desc="US Pooled, no mismatch Q (Table 3 Row 7 alt, AQ col unchanged)",
    axis_block_name="sample",
    axis_block={"restriction": "no_mismatch"},
)

# rc/sample/restriction/sorting_sectors
run_spec(
    spec_id="rc/sample/restriction/sorting_sectors",
    spec_tree_path="specification_tree/modules/robustness/sample.md",
    outcome_var="l_irAQ53_dum_skti_hrs_secall",
    treatment_var="l_y",
    data=us_pooled,
    sample_desc="US Pooled, sorting sectors Q (Table 3 Row 8, AQ col unchanged)",
    axis_block_name="sample",
    axis_block={"restriction": "sorting_sectors"},
)

# rc/sample/restriction/sorting_regions
run_spec(
    spec_id="rc/sample/restriction/sorting_regions",
    spec_tree_path="specification_tree/modules/robustness/sample.md",
    outcome_var="l_irAQ53_dum_skti_hrs_secall",
    treatment_var="l_y",
    data=us_pooled,
    sample_desc="US Pooled, sorting regions Q (Table 3 Row 9, AQ col unchanged)",
    axis_block_name="sample",
    axis_block={"restriction": "sorting_regions"},
)

# ---------- rc/sample/outliers variants ----------

# rc/sample/outliers/trim_y_5_95: trim by income 5th-95th percentile
ly_all = micro['l_y'].dropna()
p5_m, p95_m = ly_all.quantile(0.05), ly_all.quantile(0.95)
micro_trim5 = micro[(micro['l_y'] >= p5_m) & (micro['l_y'] <= p95_m)]

run_spec(
    spec_id="rc/sample/outliers/trim_y_5_95",
    spec_tree_path="specification_tree/modules/robustness/sample.md",
    outcome_var="l_irAQ53_dum_skti_hrs_secall",
    treatment_var="l_y",
    data=micro_trim5,
    sample_desc=f"sample_micro==1, trimmed l_y 5-95pct [{p5_m:.2f}, {p95_m:.2f}]",
    axis_block_name="sample",
    axis_block={"outliers": "trim_y_5_95"},
)

# rc/sample/outliers/trim_y_1_99
p1_m, p99_m = ly_all.quantile(0.01), ly_all.quantile(0.99)
micro_trim1 = micro[(micro['l_y'] >= p1_m) & (micro['l_y'] <= p99_m)]

run_spec(
    spec_id="rc/sample/outliers/trim_y_1_99",
    spec_tree_path="specification_tree/modules/robustness/sample.md",
    outcome_var="l_irAQ53_dum_skti_hrs_secall",
    treatment_var="l_y",
    data=micro_trim1,
    sample_desc=f"sample_micro==1, trimmed l_y 1-99pct [{p1_m:.2f}, {p99_m:.2f}]",
    axis_block_name="sample",
    axis_block={"outliers": "trim_y_1_99"},
)

# ============================================================================
# Additional cross-product specs: sigma x wage_measure for exp+gender
# ============================================================================

# Exp+gender with sigma_high (2.0)
run_spec(
    spec_id="rc/form/outcome/aq_dumx_sigma_high",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    outcome_var="l_irAQ53rh_dumx_skti_hrs_secall",
    treatment_var="l_y",
    data=micro,
    sample_desc="sample_micro==1, exp+gender wages, sigma=2.0",
    functional_form={"interpretation": "AQ with exp+gender wages and sigma=2.0"},
)

# Exp+gender with sigma_low (1.3)
run_spec(
    spec_id="rc/form/outcome/aq_dumx_sigma_low",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    outcome_var="l_irAQ53rl_dumx_skti_hrs_secall",
    treatment_var="l_y",
    data=micro,
    sample_desc="sample_micro==1, exp+gender wages, sigma=1.3",
    functional_form={"interpretation": "AQ with exp+gender wages and sigma=1.3"},
)

# Self-employment with sigma_high
run_spec(
    spec_id="rc/form/outcome/aq_dumse_sigma_high",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    outcome_var="l_irAQ53rh_dumse_skti_hrs_secall",
    treatment_var="l_y",
    data=se_sample,
    sample_desc="Self-employment subsample, sigma=2.0",
    functional_form={"interpretation": "AQ with self-employment and sigma=2.0"},
)

# Self-employment with sigma_low
run_spec(
    spec_id="rc/form/outcome/aq_dumse_sigma_low",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    outcome_var="l_irAQ53rl_dumse_skti_hrs_secall",
    treatment_var="l_y",
    data=se_sample,
    sample_desc="Self-employment subsample, sigma=1.3",
    functional_form={"interpretation": "AQ with self-employment and sigma=1.3"},
)

# Barro-Lee sigma_high
run_spec(
    spec_id="rc/form/outcome/aq_blee_sigma_high",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    outcome_var="l_irAQ53rh_blee_skti",
    treatment_var="l_y",
    data=all_countries,
    sample_desc="Barro-Lee sample, sigma=2.0",
    functional_form={"interpretation": "AQ Barro-Lee with sigma=2.0"},
)

# Barro-Lee sigma_low
run_spec(
    spec_id="rc/form/outcome/aq_blee_sigma_low",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    outcome_var="l_irAQ53rl_blee_skti",
    treatment_var="l_y",
    data=all_countries,
    sample_desc="Barro-Lee sample, sigma=1.3",
    functional_form={"interpretation": "AQ Barro-Lee with sigma=1.3"},
)

# Sector subsamples with sigma_high
for sec_num, sec_name in [(1, 'agriculture'), (2, 'manufacturing'), (3, 'lowskill_services'), (4, 'highskill_services')]:
    run_spec(
        spec_id=f"rc/form/outcome/aq_sec{sec_num}_sigma_high",
        spec_tree_path="specification_tree/modules/robustness/functional_form.md",
        outcome_var=f"l_irAQ53rh_dum_skti_hrs_sec{sec_num}",
        treatment_var="l_y",
        data=micro,
        sample_desc=f"sample_micro==1, {sec_name} sector, sigma=2.0",
        functional_form={"interpretation": f"AQ {sec_name} sector with sigma=2.0"},
    )

# Sector subsamples with sigma_low
for sec_num, sec_name in [(1, 'agriculture'), (2, 'manufacturing'), (3, 'lowskill_services'), (4, 'highskill_services')]:
    run_spec(
        spec_id=f"rc/form/outcome/aq_sec{sec_num}_sigma_low",
        spec_tree_path="specification_tree/modules/robustness/functional_form.md",
        outcome_var=f"l_irAQ53rl_dum_skti_hrs_sec{sec_num}",
        treatment_var="l_y",
        data=micro,
        sample_desc=f"sample_micro==1, {sec_name} sector, sigma=1.3",
        functional_form={"interpretation": f"AQ {sec_name} sector with sigma=1.3"},
    )

# Wage ratio with exp+gender
run_spec(
    spec_id="rc/form/outcome/wage_ratio_dumx",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    outcome_var="l_wrat53_dumx_skti_secall",
    treatment_var="l_y",
    data=micro,
    sample_desc="sample_micro==1, exp+gender wage ratio",
    functional_form={"interpretation": "wage ratio using exp+gender adjusted wages"},
)

# ============================================================================
# STEP 3: Inference variants
# ============================================================================

print("\n" + "=" * 60)
print("STEP 3: Inference variants")
print("=" * 60)

# For each specification, run HC3 inference variant
for sr in list(spec_results):
    if sr['run_success'] == 1:
        base_run_id = sr['spec_run_id']
        outcome = sr['outcome_var']
        treatment = sr['treatment_var']
        sample_desc = sr['sample_desc']

        # Determine the correct data for this spec
        if 'Barro-Lee' in sample_desc or 'N~92' in sample_desc or 'broad' in sample_desc.lower():
            data_for_infer = all_countries
        elif 'US Pooled' in sample_desc or 'us_immigrants' in sr['spec_id']:
            data_for_infer = us_pooled
        elif 'self-employment' in sample_desc.lower() or 'Self-employment' in sample_desc:
            data_for_infer = se_sample
        elif 'trimmed l_y 5-95' in sample_desc:
            data_for_infer = micro_trim5
        elif 'trimmed l_y 1-99' in sample_desc:
            data_for_infer = micro_trim1
        elif 'excl income outliers' in sample_desc:
            data_for_infer = micro_no_outlier
        elif 'americas' in sample_desc.lower():
            # For the continent dummy spec, skip inference (different model)
            continue
        else:
            data_for_infer = micro

        run_inference_variant(
            base_run_id=base_run_id,
            spec_id="infer/se/hc/hc3",
            spec_tree_path="specification_tree/modules/inference/se.md",
            outcome_var=outcome,
            treatment_var=treatment,
            data=data_for_infer,
            sample_desc=sample_desc,
            vcov_type='hc3',
        )

# ============================================================================
# STEP 4: Write outputs
# ============================================================================

print("\n" + "=" * 60)
print("STEP 4: Writing outputs")
print("=" * 60)

# 4.1 specification_results.csv
spec_df = pd.DataFrame(spec_results)
spec_df.to_csv(f"{OUTPUT_DIR}/specification_results.csv", index=False)
print(f"  specification_results.csv: {len(spec_df)} rows")
print(f"    Successful: {spec_df['run_success'].sum()}")
print(f"    Failed: {(~spec_df['run_success'].astype(bool)).sum()}")

# 4.2 inference_results.csv
infer_df = pd.DataFrame(inference_results)
infer_df.to_csv(f"{OUTPUT_DIR}/inference_results.csv", index=False)
print(f"  inference_results.csv: {len(infer_df)} rows")

# 4.3 SPECIFICATION_SEARCH.md
n_total = len(spec_df)
n_success = int(spec_df['run_success'].sum())
n_failed = n_total - n_success
n_infer = len(infer_df)

search_md = f"""# Specification Search: {PAPER_ID}

## Surface Summary

- **Paper**: Hendricks & Schoellman (2018), "Human Capital and Development Accounting: New Evidence from Immigrant Earnings"
- **Baseline groups**: 1 (G1)
- **Design**: cross_sectional_ols (bivariate regression, no controls)
- **Baseline**: Table 2 Row 1 Col 3 -- log(irAQ53_dum_skti_hrs_secall) ~ l_y, micro sample (N~12 countries)
- **Focal parameter**: slope coefficient on l_y (log income per worker) = elasticity of AQ w.r.t. income
- **Budget**: max 80 core specs
- **Seed**: 146041

## Execution Summary

### Specification Results
- **Planned**: {n_total}
- **Executed successfully**: {n_success}
- **Failed**: {n_failed}

### Inference Results
- **Inference variants executed**: {n_infer}
- **Canonical inference**: HC1 (heteroskedasticity-robust)
- **Variant**: HC3 (small-sample correction)

## Baseline Result
- **Coefficient**: {baseline_res['coefficient']:.4f}
- **SE (HC1)**: {baseline_res['std_error']:.4f}
- **p-value**: {baseline_res['p_value']:.4f}
- **N**: {baseline_res['n_obs']}
- **R-squared**: {baseline_res['r_squared']:.4f}

## Specification Universe Description

The main axis of variation is **outcome variable construction** rather than controls (all regressions are bivariate). Specifications vary along:

1. **Elasticity of substitution (sigma)**: 1.3, 1.5 (baseline), 2.0
2. **Wage premium estimation**: education dummies (baseline) vs experience+gender adjusted
3. **Labor supply measure**: hours (baseline), body count, working-age population
4. **Mincerian return**: country-specific (baseline) vs common
5. **Skill threshold**: upper-secondary (baseline), some-college, tertiary-only
6. **Country sample**: micro (N~12, baseline), US Pooled (N~101), Barro-Lee (N~92)
7. **Sector subsamples**: agriculture, manufacturing, low-skill services, high-skill services
8. **Self-employment**: wage-employed only (baseline) vs including self-employed
9. **Sample restrictions**: 10+ years in US, good English, no downgrading, no mismatch, sorting controls
10. **Outlier trimming**: 1/99 and 5/95 percentiles of income

## Deviations and Notes

1. **rc/form/outcome/aq_sigma_4p0**: Failed. Sigma=4.0 AQ is not pre-computed in the data. The devacc_main.do only uses sigma=4 for development accounting calculations, not for the AQ regression variables.

2. **Table 3 AQ column regressions (rc/sample/restriction/*)**: For Table 3 rows that change the Q estimation method (10+ years, good English, no downgrading, sorting), the AQ regression outcome variable (irAQ53_dum_skti_hrs_secall) is unchanged. These rows produce the same point estimate for the AQ column. They are included for completeness and to faithfully represent the paper's revealed specification space.

3. **Continent dummies (rc/controls/add/continent_dummies)**: Implemented as an Americas vs. non-Americas dummy given the small sample size (12 countries). Full continent dummies would consume too many degrees of freedom.

4. **Treatment alternatives**: Used log(y_relUS_2005) and log(y_relUS2000) as alternative treatments since the baseline l_y is already PWT-based.

## Software Stack

- Python {sys.version.split()[0]}
- numpy: {np.__version__}
- pandas: {pd.__version__}
- scipy: {__import__('scipy').__version__}
- Manual OLS implementation (no pyfixest needed for bivariate regression)
"""

with open(f"{OUTPUT_DIR}/SPECIFICATION_SEARCH.md", 'w') as f:
    f.write(search_md)
print(f"  SPECIFICATION_SEARCH.md written")

# Summary statistics
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total specifications: {n_total}")
print(f"Successful: {n_success}")
print(f"Failed: {n_failed}")
print(f"Inference variants: {n_infer}")
print()

# Show coefficient distribution
successful = spec_df[spec_df['run_success'] == 1]
print(f"Coefficient range: [{successful['coefficient'].min():.4f}, {successful['coefficient'].max():.4f}]")
print(f"Coefficient median: {successful['coefficient'].median():.4f}")
print(f"p-value < 0.05: {(successful['p_value'] < 0.05).sum()} / {len(successful)}")
print(f"p-value < 0.10: {(successful['p_value'] < 0.10).sum()} / {len(successful)}")
