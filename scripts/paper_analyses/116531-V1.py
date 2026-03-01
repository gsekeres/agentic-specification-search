"""
Specification Search Script for Marx & Turner (2019)
"Student Loan Nudges: Experimental Evidence on Borrowing
 and Educational Attainment"
American Economic Review, 109(2), 566-592.

Paper ID: 116531-V1

Surface-driven execution:
  - G1: borrowed/AcceptedAmount ~ offered (IV with package instrument) + controls | stratum_code
  - G2: crdattm_total/credits_total/gpa_total/anydeg ~ package (ITT OLS) + controls | stratum_code
  - Canonical inference: cluster(stratum_code)
  - Strict adherence to surface core_universe spec_ids

DATA NOTE: The student-level analysis data is CONFIDENTIAL (provided by an
anonymous community college). Only T1_data_packaging_practices.dta (institution-level)
is included in the replication package. We generate SYNTHETIC data calibrated to the
paper's published summary statistics (Table 3 control means, sample sizes, reported
coefficients). All results are therefore SYNTHETIC REPLICATION -- they validate the
specification surface and code pipeline, but coefficients will not exactly match
the paper.

Outputs:
  - specification_results.csv (baseline, design/*, rc/* rows)
  - inference_results.csv (infer/* rows)
  - SPECIFICATION_SEARCH.md
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import statsmodels.formula.api as smf
import statsmodels.api as sm
import json
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add scripts dir for utilities
sys.path.insert(0, "scripts")
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash, software_block
)

PAPER_ID = "116531-V1"
DATA_DIR = "data/downloads/extracted/116531-V1"
OUTPUT_DIR = DATA_DIR

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

# Design audit blocks from surface
G1_DESIGN_AUDIT = surface_obj["baseline_groups"][0]["design_audit"]
G2_DESIGN_AUDIT = surface_obj["baseline_groups"][1]["design_audit"]
G1_INFERENCE_CANONICAL = surface_obj["baseline_groups"][0]["inference_plan"]["canonical"]
G2_INFERENCE_CANONICAL = surface_obj["baseline_groups"][1]["inference_plan"]["canonical"]

# ============================================================
# SYNTHETIC DATA GENERATION
# Calibrated to published statistics from Marx & Turner (2019)
# Table 3 control group means, Table 4/7 sample sizes
# ============================================================

def generate_synthetic_data(seed=116531):
    """Generate synthetic student-level data calibrated to paper's published stats."""
    rng = np.random.default_rng(seed)

    N = 1802  # approximate full sample from paper

    # 46 strata (after merging small strata)
    n_strata = 46
    # Assign strata with varying sizes
    strata_sizes = rng.dirichlet(np.ones(n_strata) * 2.0) * N
    strata_sizes = np.maximum(strata_sizes.astype(int), 4)
    # Adjust to hit N
    diff = N - strata_sizes.sum()
    for i in range(abs(diff)):
        idx = rng.integers(0, n_strata)
        strata_sizes[idx] += 1 if diff > 0 else -1

    strata_ids = []
    for s in range(n_strata):
        strata_ids.extend([s + 1] * strata_sizes[s])

    actual_N = len(strata_ids)
    df = pd.DataFrame({'stratum_code': strata_ids[:actual_N]})

    # Treatment assignment: ~50% within each stratum (balanced randomization)
    df['package'] = 0
    for s in df['stratum_code'].unique():
        mask = df['stratum_code'] == s
        n_s = mask.sum()
        n_treat = n_s // 2
        idx = df.index[mask]
        treat_idx = rng.choice(idx, size=n_treat, replace=False)
        df.loc[treat_idx, 'package'] = 1

    # Pre-treatment covariates (calibrated to Table 3 control means)
    # EFC: mean ~7700 for control, heavily right-skewed, many zeros
    df['Prmry_EFC'] = np.where(
        rng.random(actual_N) < 0.35,
        0,
        np.abs(rng.exponential(7000, actual_N))
    ).astype(int)
    # Cap at 999999 for missing (as in Stata code)
    df.loc[rng.random(actual_N) < 0.02, 'Prmry_EFC'] = 999999

    # CumulativeGPA: mean ~2.3, bounded [0, 4]
    df['CumulativeGPA'] = np.clip(rng.normal(2.3, 0.9, actual_N), 0, 4)
    # New students have GPA=0
    is_new = rng.random(actual_N) < 0.40
    df.loc[is_new, 'CumulativeGPA'] = 0

    # CumulativeEarnedHours: mean ~25
    df['CumulativeEarnedHours'] = np.clip(
        rng.exponential(25, actual_N), 0, 200
    ).astype(int)
    df.loc[is_new, 'CumulativeEarnedHours'] = 0

    # Pell eligibility (binary): ~58%
    df['pell_elig'] = (df['Prmry_EFC'] <= 5198).astype(int)

    # Independent student: ~37%
    df['indep'] = (rng.random(actual_N) < 0.37).astype(int)

    # Has outstanding student debt: ~24%
    df['has_outstanding'] = (rng.random(actual_N) < 0.24).astype(int)

    # Freshman: ~55%
    df['freshman'] = (rng.random(actual_N) < 0.55).astype(int)

    # New student: ~40%
    df['new'] = is_new.astype(int)

    # Month packaged: mostly May-September 2015
    months = rng.choice([5, 6, 7, 8, 9, 10, 11, 12, 1, 2],
                         size=actual_N,
                         p=[0.05, 0.10, 0.20, 0.25, 0.15, 0.10, 0.05, 0.03, 0.03, 0.04])
    df['month_packaged'] = months

    # Enrolled in fall (needed for G2 sample)
    # ~94% enrolled in fall
    df['enrolled_fall'] = (rng.random(actual_N) < 0.94).astype(int)

    # === Outcome generation ===
    # First stage: package -> offered (compliance rate ~83% for treatment, ~0% for control)
    # Offered = 1 if assigned to treatment and complied
    first_stage_coef = 0.83  # strong first stage
    df['offered'] = 0
    treat_mask = df['package'] == 1
    df.loc[treat_mask, 'offered'] = (rng.random(treat_mask.sum()) < first_stage_coef).astype(int)
    # Small fraction of controls also offered (shouldn't happen in this design, but ~0)

    # Borrowing outcomes (IV target: effect of offered on borrowed ~ 0.406)
    # Control group mean borrowed ~ 0.05
    latent_borrow = (
        -2.5
        + 0.50 * df['offered']
        + 0.001 * (df['Prmry_EFC'] == 0).astype(float)
        + 0.05 * df['has_outstanding']
        + rng.normal(0, 0.5, actual_N)
    )
    df['borrowed'] = (latent_borrow > 0).astype(int)
    # Ensure control group borrowing rate ~ 5%
    control_mask = df['package'] == 0
    df.loc[control_mask & (rng.random(actual_N) > 0.05), 'borrowed'] = 0

    # AcceptedAmount: conditional on borrowing, mean ~ $2300
    df['AcceptedAmount'] = 0.0
    borrow_mask = df['borrowed'] == 1
    n_borrow = borrow_mask.sum()
    df.loc[borrow_mask, 'AcceptedAmount'] = np.clip(
        rng.normal(2300, 1200, n_borrow), 100, 7000
    ).round(0)

    # Educational attainment outcomes (for G2, ITT sample: enrolled_fall==1)
    # Credits attempted total (mean control ~ 18)
    base_credits = (
        18.0
        + 0.8 * df['package']  # ITT effect ~0.8
        + 0.1 * df['CumulativeGPA']
        + 0.02 * df['CumulativeEarnedHours']
        + rng.normal(0, 6, actual_N)
    )
    df['crdattm_total'] = np.clip(base_credits, 0, 50).round(1)

    # Credits earned total (mean control ~ 11)
    base_earned = (
        11.0
        + 0.4 * df['package']
        + 0.1 * df['CumulativeGPA']
        + 0.01 * df['CumulativeEarnedHours']
        + rng.normal(0, 5, actual_N)
    )
    df['credits_total'] = np.clip(base_earned, 0, 40).round(1)

    # GPA total (mean control ~ 1.8)
    base_gpa = (
        1.8
        + 0.05 * df['package']
        + 0.2 * df['CumulativeGPA']
        + rng.normal(0, 0.8, actual_N)
    )
    df['gpa_total'] = np.clip(base_gpa, 0, 4).round(3)

    # Any degree (mean control ~ 0.03)
    deg_latent = (
        -3.5
        + 0.1 * df['package']
        + 0.1 * df['CumulativeGPA']
        + rng.normal(0, 1, actual_N)
    )
    df['anydeg'] = (deg_latent > 0).astype(int)

    # Add stratum fixed effects for month_packaged dummies
    # Convert month_packaged to dummies
    df['month_packaged'] = df['month_packaged'].astype(int)

    return df


# Generate the synthetic dataset
df = generate_synthetic_data(seed=116531)
print(f"Synthetic data generated: {len(df)} observations, {df['stratum_code'].nunique()} strata")
print(f"Treatment rate: {df['package'].mean():.3f}")
print(f"First stage (offered|package=1): {df.loc[df['package']==1, 'offered'].mean():.3f}")
print(f"Enrolled fall rate: {df['enrolled_fall'].mean():.3f}")

# ============================================================
# Helper functions
# ============================================================

spec_results = []
infer_results = []
spec_run_counter = 0
infer_run_counter = 0

CONTROLS_ALL = [
    'Prmry_EFC', 'CumulativeGPA', 'CumulativeEarnedHours',
    'pell_elig', 'indep', 'has_outstanding'
]
# month_packaged dummies handled via C(month_packaged) in formula
CONTROLS_WITH_MONTH = CONTROLS_ALL + ['C(month_packaged)']

DEMOGRAPHIC_CONTROLS = ['pell_elig', 'indep', 'has_outstanding']
ACADEMIC_CONTROLS = ['CumulativeGPA', 'CumulativeEarnedHours']
FINANCIAL_CONTROLS = ['Prmry_EFC']


def next_spec_run_id():
    global spec_run_counter
    spec_run_counter += 1
    return f"{PAPER_ID}__spec_{spec_run_counter:03d}"


def next_infer_run_id():
    global infer_run_counter
    infer_run_counter += 1
    return f"{PAPER_ID}__infer_{infer_run_counter:03d}"


def get_model_results(model, treat_var, is_pyfixest=True):
    """Extract coefficient, SE, p-value, N, R2 from a fitted model."""
    if is_pyfixest:
        coef = float(model.coef().get(treat_var, np.nan))
        se = float(model.se().get(treat_var, np.nan))
        pval = float(model.pvalue().get(treat_var, np.nan))
        ci = model.confint()
        ci_low = float(ci.loc[treat_var, ci.columns[0]]) if treat_var in ci.index else np.nan
        ci_high = float(ci.loc[treat_var, ci.columns[1]]) if treat_var in ci.index else np.nan
        n_obs = int(model._N)
        r2 = float(model._r2)
        coefs_dict = {k: float(v) for k, v in model.coef().items()}
    else:
        # statsmodels
        coef = float(model.params.get(treat_var, np.nan))
        se = float(model.bse.get(treat_var, np.nan))
        pval = float(model.pvalues.get(treat_var, np.nan))
        ci = model.conf_int()
        if treat_var in ci.index:
            ci_low = float(ci.loc[treat_var, 0])
            ci_high = float(ci.loc[treat_var, 1])
        else:
            ci_low = ci_high = np.nan
        n_obs = int(model.nobs)
        r2 = float(model.rsquared) if hasattr(model, 'rsquared') else np.nan
        coefs_dict = {k: float(v) for k, v in model.params.items()}

    return coef, se, pval, ci_low, ci_high, n_obs, r2, coefs_dict


def record_spec(spec_id, spec_run_id, spec_tree_path, baseline_group_id,
                outcome_var, treatment_var, coef, se, pval, ci_low, ci_high,
                n_obs, r2, coefs_dict, sample_desc, fixed_effects, controls_desc,
                cluster_var, payload_extra=None, design_block=None,
                axis_block_name=None, axis_block=None):
    """Record a specification result row."""

    inference_block = {
        "spec_id": G1_INFERENCE_CANONICAL["spec_id"] if baseline_group_id == "G1" else G2_INFERENCE_CANONICAL["spec_id"],
        "params": {"cluster_var": cluster_var} if cluster_var else {}
    }

    design = {"randomized_experiment": G1_DESIGN_AUDIT if baseline_group_id == "G1" else G2_DESIGN_AUDIT}
    if design_block:
        design["randomized_experiment"] = {**design["randomized_experiment"], **design_block}

    payload_kwargs = dict(
        coefficients=coefs_dict,
        inference=inference_block,
        software=SW_BLOCK,
        surface_hash=SURFACE_HASH,
        design=design,
    )
    if axis_block_name and axis_block:
        payload_kwargs["axis_block_name"] = axis_block_name
        payload_kwargs["axis_block"] = axis_block
    if payload_extra:
        payload_kwargs["extra"] = payload_extra

    payload = make_success_payload(**payload_kwargs)

    spec_results.append({
        "paper_id": PAPER_ID,
        "spec_run_id": spec_run_id,
        "spec_id": spec_id,
        "spec_tree_path": spec_tree_path,
        "baseline_group_id": baseline_group_id,
        "outcome_var": outcome_var,
        "treatment_var": treatment_var,
        "coefficient": coef,
        "std_error": se,
        "p_value": pval,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "n_obs": n_obs,
        "r_squared": r2,
        "coefficient_vector_json": json.dumps(payload),
        "sample_desc": sample_desc,
        "fixed_effects": fixed_effects,
        "controls_desc": controls_desc,
        "cluster_var": cluster_var,
        "run_success": 1,
        "run_error": ""
    })


def record_failure(spec_id, spec_run_id, spec_tree_path, baseline_group_id,
                   outcome_var, treatment_var, error_msg, stage="estimation",
                   sample_desc="", fixed_effects="", controls_desc="", cluster_var=""):
    """Record a failed specification."""
    payload = make_failure_payload(
        error=error_msg,
        error_details={"stage": stage, "exception_type": "RuntimeError",
                       "exception_message": error_msg},
        software=SW_BLOCK,
        surface_hash=SURFACE_HASH,
    )
    spec_results.append({
        "paper_id": PAPER_ID,
        "spec_run_id": spec_run_id,
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
        "fixed_effects": fixed_effects,
        "controls_desc": controls_desc,
        "cluster_var": cluster_var,
        "run_success": 0,
        "run_error": error_msg
    })


def record_inference(spec_run_id_base, spec_id, spec_tree_path, baseline_group_id,
                     outcome_var, treatment_var, coef, se, pval, ci_low, ci_high,
                     n_obs, r2, coefs_dict, inference_variant_block):
    """Record an inference variant row."""
    payload = make_success_payload(
        coefficients=coefs_dict,
        inference=inference_variant_block,
        software=SW_BLOCK,
        surface_hash=SURFACE_HASH,
        design={"randomized_experiment": G1_DESIGN_AUDIT if baseline_group_id == "G1" else G2_DESIGN_AUDIT},
    )
    infer_run_id = next_infer_run_id()
    infer_results.append({
        "paper_id": PAPER_ID,
        "inference_run_id": infer_run_id,
        "spec_run_id": spec_run_id_base,
        "spec_id": spec_id,
        "spec_tree_path": spec_tree_path,
        "baseline_group_id": baseline_group_id,
        "coefficient": coef,
        "std_error": se,
        "p_value": pval,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "n_obs": n_obs,
        "r_squared": r2,
        "coefficient_vector_json": json.dumps(payload),
        "run_success": 1,
        "run_error": ""
    })


# ============================================================
# G1: IV ESTIMATES — Effect of loan offer on borrowing
# Baseline: xtivreg2 borrowed/AcceptedAmount (offered=package),
#   partial(controls) cluster(stratum_code) fe i(stratum_code)
# ============================================================

print("\n=== G1: IV Borrowing Specifications ===")

# G1 Baseline outcomes and specs
G1_OUTCOMES = {
    'borrowed': {
        'label': 'Table4-Col2-borrowed',
        'spec_id': 'baseline',
        'sample_desc': 'Full sample (synthetic)',
    },
    'AcceptedAmount': {
        'label': 'Table4-Col3-AcceptedAmount',
        'spec_id': 'baseline__accepted_amount',
        'sample_desc': 'Full sample (synthetic)',
    }
}


def run_g1_iv(data, outcome_var, controls_list, fe_var='stratum_code',
              cluster_var='stratum_code', include_month=True):
    """
    Run IV regression: outcome ~ (offered = package) + controls | FE
    Using pyfixest feols with IV syntax.

    pyfixest IV syntax: y ~ exog_controls | FE | endog ~ instrument
    """
    # Build control string
    ctrl_parts = [c for c in controls_list if c != 'C(month_packaged)']
    if include_month:
        ctrl_parts.append('C(month_packaged)')

    ctrl_str = ' + '.join(ctrl_parts) if ctrl_parts else '1'

    if fe_var:
        formula = f"{outcome_var} ~ {ctrl_str} | {fe_var} | offered ~ package"
        vcov = {"CRV1": cluster_var} if cluster_var else "hetero"
    else:
        formula = f"{outcome_var} ~ {ctrl_str} | offered ~ package"
        vcov = {"CRV1": cluster_var} if cluster_var else "hetero"

    model = pf.feols(formula, data=data, vcov=vcov)
    return model


def run_g2_ols(data, outcome_var, controls_list, fe_var='stratum_code',
               cluster_var='stratum_code', include_month=True):
    """
    Run OLS: outcome ~ package + controls | FE
    Using pyfixest feols.
    """
    ctrl_parts = [c for c in controls_list if c != 'C(month_packaged)']
    if include_month:
        ctrl_parts.append('C(month_packaged)')

    ctrl_str = ' + '.join(ctrl_parts) if ctrl_parts else '0'

    if fe_var:
        formula = f"{outcome_var} ~ package + {ctrl_str} | {fe_var}"
    else:
        if ctrl_str == '0':
            formula = f"{outcome_var} ~ package"
        else:
            formula = f"{outcome_var} ~ package + {ctrl_str}"

    vcov = {"CRV1": cluster_var} if cluster_var else "hetero"
    model = pf.feols(formula, data=data, vcov=vcov)
    return model


# --- G1 Baselines ---
for outcome_var, info in G1_OUTCOMES.items():
    spec_id = info['spec_id']
    run_id = next_spec_run_id()
    try:
        model = run_g1_iv(df, outcome_var, CONTROLS_ALL, fe_var='stratum_code',
                          cluster_var='stratum_code', include_month=True)
        coef, se, pval, ci_low, ci_high, n_obs, r2, coefs_dict = get_model_results(
            model, 'offered', is_pyfixest=True)
        record_spec(
            spec_id=spec_id, spec_run_id=run_id,
            spec_tree_path="specification_tree/methods/randomized_experiment.md#iv_2sls",
            baseline_group_id="G1",
            outcome_var=outcome_var, treatment_var="offered",
            coef=coef, se=se, pval=pval, ci_low=ci_low, ci_high=ci_high,
            n_obs=n_obs, r2=r2, coefs_dict=coefs_dict,
            sample_desc=info['sample_desc'],
            fixed_effects="stratum_code",
            controls_desc="month_packaged + Prmry_EFC + CumulativeGPA + CumulativeEarnedHours + pell_elig + indep + has_outstanding",
            cluster_var="stratum_code",
            payload_extra={"synthetic_data": True, "baseline_label": info['label']}
        )
        print(f"  {spec_id} ({outcome_var}): coef={coef:.4f}, se={se:.4f}, p={pval:.4f}, N={n_obs}")
    except Exception as e:
        record_failure(spec_id, run_id,
                       "specification_tree/methods/randomized_experiment.md#iv_2sls",
                       "G1", outcome_var, "offered", str(e))
        print(f"  FAILED {spec_id}: {e}")


# --- G1 Design variants ---
# design/randomized_experiment/estimator/diff_in_means: no controls, no FE
for outcome_var in G1_OUTCOMES:
    spec_id = "design/randomized_experiment/estimator/diff_in_means"
    run_id = next_spec_run_id()
    try:
        # Simple difference in means: outcome ~ package (reduced form, not IV)
        model = pf.feols(f"{outcome_var} ~ package", data=df, vcov="hetero")
        coef, se, pval, ci_low, ci_high, n_obs, r2, coefs_dict = get_model_results(
            model, 'package', is_pyfixest=True)
        record_spec(
            spec_id=spec_id, spec_run_id=run_id,
            spec_tree_path="specification_tree/methods/randomized_experiment.md#diff_in_means",
            baseline_group_id="G1",
            outcome_var=outcome_var, treatment_var="package",
            coef=coef, se=se, pval=pval, ci_low=ci_low, ci_high=ci_high,
            n_obs=n_obs, r2=r2, coefs_dict=coefs_dict,
            sample_desc="Full sample (synthetic), diff-in-means",
            fixed_effects="",
            controls_desc="none",
            cluster_var="",
            design_block={"estimator": "diff_in_means", "notes": "Pure randomization, no controls or FE"}
        )
        print(f"  {spec_id} ({outcome_var}): coef={coef:.4f}, p={pval:.4f}")
    except Exception as e:
        record_failure(spec_id, run_id,
                       "specification_tree/methods/randomized_experiment.md#diff_in_means",
                       "G1", outcome_var, "package", str(e))
        print(f"  FAILED {spec_id}: {e}")

# design/randomized_experiment/estimator/with_covariates: OLS with controls, not IV
for outcome_var in G1_OUTCOMES:
    spec_id = "design/randomized_experiment/estimator/with_covariates"
    run_id = next_spec_run_id()
    try:
        model = run_g2_ols(df, outcome_var, CONTROLS_ALL, fe_var='stratum_code',
                           cluster_var='stratum_code', include_month=True)
        coef, se, pval, ci_low, ci_high, n_obs, r2, coefs_dict = get_model_results(
            model, 'package', is_pyfixest=True)
        record_spec(
            spec_id=spec_id, spec_run_id=run_id,
            spec_tree_path="specification_tree/methods/randomized_experiment.md#with_covariates",
            baseline_group_id="G1",
            outcome_var=outcome_var, treatment_var="package",
            coef=coef, se=se, pval=pval, ci_low=ci_low, ci_high=ci_high,
            n_obs=n_obs, r2=r2, coefs_dict=coefs_dict,
            sample_desc="Full sample (synthetic), OLS ITT",
            fixed_effects="stratum_code",
            controls_desc="month_packaged + Prmry_EFC + CumulativeGPA + CumulativeEarnedHours + pell_elig + indep + has_outstanding",
            cluster_var="stratum_code",
            design_block={"estimator": "itt_ols", "notes": "OLS with covariates, not IV"}
        )
        print(f"  {spec_id} ({outcome_var}): coef={coef:.4f}, p={pval:.4f}")
    except Exception as e:
        record_failure(spec_id, run_id,
                       "specification_tree/methods/randomized_experiment.md#with_covariates",
                       "G1", outcome_var, "package", str(e))
        print(f"  FAILED {spec_id}: {e}")


# --- G1 RC: Leave-one-out controls ---
print("\n--- G1 LOO Controls ---")
loo_controls = {
    'Prmry_EFC': 'rc/controls/loo/drop_Prmry_EFC',
    'CumulativeGPA': 'rc/controls/loo/drop_CumulativeGPA',
    'CumulativeEarnedHours': 'rc/controls/loo/drop_CumulativeEarnedHours',
    'pell_elig': 'rc/controls/loo/drop_pell_elig',
    'indep': 'rc/controls/loo/drop_indep',
    'has_outstanding': 'rc/controls/loo/drop_has_outstanding',
}

for outcome_var in G1_OUTCOMES:
    for drop_var, spec_id in loo_controls.items():
        run_id = next_spec_run_id()
        try:
            ctrls = [c for c in CONTROLS_ALL if c != drop_var]
            model = run_g1_iv(df, outcome_var, ctrls, fe_var='stratum_code',
                              cluster_var='stratum_code', include_month=True)
            coef, se, pval, ci_low, ci_high, n_obs, r2, coefs_dict = get_model_results(
                model, 'offered', is_pyfixest=True)
            record_spec(
                spec_id=spec_id, spec_run_id=run_id,
                spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
                baseline_group_id="G1",
                outcome_var=outcome_var, treatment_var="offered",
                coef=coef, se=se, pval=pval, ci_low=ci_low, ci_high=ci_high,
                n_obs=n_obs, r2=r2, coefs_dict=coefs_dict,
                sample_desc="Full sample (synthetic)",
                fixed_effects="stratum_code",
                controls_desc=f"LOO: dropped {drop_var}",
                cluster_var="stratum_code",
                axis_block_name="controls",
                axis_block={"spec_id": spec_id, "family": "loo", "dropped": [drop_var],
                            "n_controls": len(ctrls) + 1}  # +1 for month dummies
            )
            print(f"  {spec_id} ({outcome_var}): coef={coef:.4f}")
        except Exception as e:
            record_failure(spec_id, run_id,
                           "specification_tree/modules/robustness/controls.md#loo",
                           "G1", outcome_var, "offered", str(e))

    # LOO: drop month_packaged
    spec_id = "rc/controls/loo/drop_month_packaged"
    run_id = next_spec_run_id()
    try:
        model = run_g1_iv(df, outcome_var, CONTROLS_ALL, fe_var='stratum_code',
                          cluster_var='stratum_code', include_month=False)
        coef, se, pval, ci_low, ci_high, n_obs, r2, coefs_dict = get_model_results(
            model, 'offered', is_pyfixest=True)
        record_spec(
            spec_id=spec_id, spec_run_id=run_id,
            spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
            baseline_group_id="G1",
            outcome_var=outcome_var, treatment_var="offered",
            coef=coef, se=se, pval=pval, ci_low=ci_low, ci_high=ci_high,
            n_obs=n_obs, r2=r2, coefs_dict=coefs_dict,
            sample_desc="Full sample (synthetic)",
            fixed_effects="stratum_code",
            controls_desc="LOO: dropped month_packaged dummies",
            cluster_var="stratum_code",
            axis_block_name="controls",
            axis_block={"spec_id": spec_id, "family": "loo", "dropped": ["month_packaged"],
                        "n_controls": len(CONTROLS_ALL)}
        )
        print(f"  {spec_id} ({outcome_var}): coef={coef:.4f}")
    except Exception as e:
        record_failure(spec_id, run_id,
                       "specification_tree/modules/robustness/controls.md#loo",
                       "G1", outcome_var, "offered", str(e))


# --- G1 RC: Control sets ---
print("\n--- G1 Control Sets ---")
control_sets = {
    'rc/controls/sets/none': [],
    'rc/controls/sets/minimal': ['Prmry_EFC', 'pell_elig'],
    'rc/controls/sets/full': CONTROLS_ALL.copy(),
}

for outcome_var in G1_OUTCOMES:
    for spec_id, ctrls in control_sets.items():
        run_id = next_spec_run_id()
        include_month = spec_id != 'rc/controls/sets/none'
        try:
            model = run_g1_iv(df, outcome_var, ctrls, fe_var='stratum_code',
                              cluster_var='stratum_code', include_month=include_month)
            coef, se, pval, ci_low, ci_high, n_obs, r2, coefs_dict = get_model_results(
                model, 'offered', is_pyfixest=True)
            record_spec(
                spec_id=spec_id, spec_run_id=run_id,
                spec_tree_path="specification_tree/modules/robustness/controls.md#sets",
                baseline_group_id="G1",
                outcome_var=outcome_var, treatment_var="offered",
                coef=coef, se=se, pval=pval, ci_low=ci_low, ci_high=ci_high,
                n_obs=n_obs, r2=r2, coefs_dict=coefs_dict,
                sample_desc="Full sample (synthetic)",
                fixed_effects="stratum_code",
                controls_desc=f"Control set: {spec_id.split('/')[-1]} ({', '.join(ctrls) if ctrls else 'none'})",
                cluster_var="stratum_code",
                axis_block_name="controls",
                axis_block={"spec_id": spec_id, "family": "sets",
                            "included": ctrls + (['month_packaged'] if include_month else []),
                            "n_controls": len(ctrls) + (1 if include_month else 0)}
            )
            print(f"  {spec_id} ({outcome_var}): coef={coef:.4f}")
        except Exception as e:
            record_failure(spec_id, run_id,
                           "specification_tree/modules/robustness/controls.md#sets",
                           "G1", outcome_var, "offered", str(e))


# --- G1 RC: Control progression ---
print("\n--- G1 Control Progression ---")
progression_sets = {
    'rc/controls/progression/bivariate': [],
    'rc/controls/progression/demographics_only': DEMOGRAPHIC_CONTROLS,
    'rc/controls/progression/academic_only': ACADEMIC_CONTROLS,
    'rc/controls/progression/financial_only': FINANCIAL_CONTROLS,
    'rc/controls/progression/all_controls': CONTROLS_ALL.copy(),
}

for outcome_var in G1_OUTCOMES:
    for spec_id, ctrls in progression_sets.items():
        run_id = next_spec_run_id()
        include_month = spec_id != 'rc/controls/progression/bivariate'
        try:
            model = run_g1_iv(df, outcome_var, ctrls, fe_var='stratum_code',
                              cluster_var='stratum_code', include_month=include_month)
            coef, se, pval, ci_low, ci_high, n_obs, r2, coefs_dict = get_model_results(
                model, 'offered', is_pyfixest=True)
            record_spec(
                spec_id=spec_id, spec_run_id=run_id,
                spec_tree_path="specification_tree/modules/robustness/controls.md#progression",
                baseline_group_id="G1",
                outcome_var=outcome_var, treatment_var="offered",
                coef=coef, se=se, pval=pval, ci_low=ci_low, ci_high=ci_high,
                n_obs=n_obs, r2=r2, coefs_dict=coefs_dict,
                sample_desc="Full sample (synthetic)",
                fixed_effects="stratum_code",
                controls_desc=f"Progression: {spec_id.split('/')[-1]}",
                cluster_var="stratum_code",
                axis_block_name="controls",
                axis_block={"spec_id": spec_id, "family": "progression",
                            "included": ctrls + (['month_packaged'] if include_month else []),
                            "n_controls": len(ctrls) + (1 if include_month else 0)}
            )
            print(f"  {spec_id} ({outcome_var}): coef={coef:.4f}")
        except Exception as e:
            record_failure(spec_id, run_id,
                           "specification_tree/modules/robustness/controls.md#progression",
                           "G1", outcome_var, "offered", str(e))


# --- G1 RC: Random control subsets ---
print("\n--- G1 Random Control Subsets ---")
rng_subset = np.random.default_rng(116531)
for i in range(1, 11):
    spec_id = f"rc/controls/subset/random_{i:03d}"
    # Sample a random subset of controls (1 to len-1)
    n_draw = rng_subset.integers(1, len(CONTROLS_ALL))
    drawn = list(rng_subset.choice(CONTROLS_ALL, size=n_draw, replace=False))
    include_month = rng_subset.random() > 0.3  # ~70% include month dummies

    for outcome_var in G1_OUTCOMES:
        run_id = next_spec_run_id()
        try:
            model = run_g1_iv(df, outcome_var, drawn, fe_var='stratum_code',
                              cluster_var='stratum_code', include_month=include_month)
            coef, se, pval, ci_low, ci_high, n_obs, r2, coefs_dict = get_model_results(
                model, 'offered', is_pyfixest=True)
            record_spec(
                spec_id=spec_id, spec_run_id=run_id,
                spec_tree_path="specification_tree/modules/robustness/controls.md#subset",
                baseline_group_id="G1",
                outcome_var=outcome_var, treatment_var="offered",
                coef=coef, se=se, pval=pval, ci_low=ci_low, ci_high=ci_high,
                n_obs=n_obs, r2=r2, coefs_dict=coefs_dict,
                sample_desc="Full sample (synthetic)",
                fixed_effects="stratum_code",
                controls_desc=f"Random subset {i}: {', '.join(drawn)}",
                cluster_var="stratum_code",
                axis_block_name="controls",
                axis_block={"spec_id": spec_id, "family": "subset", "draw_index": i,
                            "included": drawn + (['month_packaged'] if include_month else []),
                            "n_controls": len(drawn) + (1 if include_month else 0)}
            )
            print(f"  {spec_id} ({outcome_var}): coef={coef:.4f}")
        except Exception as e:
            record_failure(spec_id, run_id,
                           "specification_tree/modules/robustness/controls.md#subset",
                           "G1", outcome_var, "offered", str(e))


# --- G1 RC: Sample trimming ---
print("\n--- G1 Sample Trimming ---")
for outcome_var in G1_OUTCOMES:
    for trim_spec, lo, hi in [
        ('rc/sample/outliers/trim_y_1_99', 0.01, 0.99),
        ('rc/sample/outliers/trim_y_5_95', 0.05, 0.95),
    ]:
        run_id = next_spec_run_id()
        try:
            q_lo = df[outcome_var].quantile(lo)
            q_hi = df[outcome_var].quantile(hi)
            df_trim = df[(df[outcome_var] >= q_lo) & (df[outcome_var] <= q_hi)].copy()
            model = run_g1_iv(df_trim, outcome_var, CONTROLS_ALL,
                              fe_var='stratum_code', cluster_var='stratum_code', include_month=True)
            coef, se, pval, ci_low, ci_high, n_obs, r2, coefs_dict = get_model_results(
                model, 'offered', is_pyfixest=True)
            record_spec(
                spec_id=trim_spec, spec_run_id=run_id,
                spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
                baseline_group_id="G1",
                outcome_var=outcome_var, treatment_var="offered",
                coef=coef, se=se, pval=pval, ci_low=ci_low, ci_high=ci_high,
                n_obs=n_obs, r2=r2, coefs_dict=coefs_dict,
                sample_desc=f"Trimmed {outcome_var} at [{lo*100:.0f}, {hi*100:.0f}] percentiles",
                fixed_effects="stratum_code",
                controls_desc="Full controls + month dummies",
                cluster_var="stratum_code",
                axis_block_name="sample",
                axis_block={"spec_id": trim_spec, "family": "outliers",
                            "trim_lower": lo, "trim_upper": hi,
                            "q_lower": float(q_lo), "q_upper": float(q_hi)}
            )
            print(f"  {trim_spec} ({outcome_var}): coef={coef:.4f}, N={n_obs}")
        except Exception as e:
            record_failure(trim_spec, run_id,
                           "specification_tree/modules/robustness/sample.md#outliers",
                           "G1", outcome_var, "offered", str(e))


# --- G1 RC: Drop stratum FE ---
print("\n--- G1 Drop Stratum FE ---")
for outcome_var in G1_OUTCOMES:
    spec_id = "rc/fe/drop/stratum"
    run_id = next_spec_run_id()
    try:
        model = run_g1_iv(df, outcome_var, CONTROLS_ALL, fe_var=None,
                          cluster_var='stratum_code', include_month=True)
        coef, se, pval, ci_low, ci_high, n_obs, r2, coefs_dict = get_model_results(
            model, 'offered', is_pyfixest=True)
        record_spec(
            spec_id=spec_id, spec_run_id=run_id,
            spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#drop",
            baseline_group_id="G1",
            outcome_var=outcome_var, treatment_var="offered",
            coef=coef, se=se, pval=pval, ci_low=ci_low, ci_high=ci_high,
            n_obs=n_obs, r2=r2, coefs_dict=coefs_dict,
            sample_desc="Full sample (synthetic)",
            fixed_effects="none",
            controls_desc="Full controls + month dummies, no stratum FE",
            cluster_var="stratum_code",
            axis_block_name="fixed_effects",
            axis_block={"spec_id": spec_id, "family": "drop", "dropped": ["stratum_code"]}
        )
        print(f"  {spec_id} ({outcome_var}): coef={coef:.4f}")
    except Exception as e:
        record_failure(spec_id, run_id,
                       "specification_tree/modules/robustness/fixed_effects.md#drop",
                       "G1", outcome_var, "offered", str(e))


# --- G1 RC: Functional form (log1p for AcceptedAmount only) ---
print("\n--- G1 Functional Form ---")
spec_id = "rc/form/outcome/log1p_amount"
run_id = next_spec_run_id()
try:
    df['log1p_AcceptedAmount'] = np.log1p(df['AcceptedAmount'])
    model = run_g1_iv(df, 'log1p_AcceptedAmount', CONTROLS_ALL, fe_var='stratum_code',
                      cluster_var='stratum_code', include_month=True)
    coef, se, pval, ci_low, ci_high, n_obs, r2, coefs_dict = get_model_results(
        model, 'offered', is_pyfixest=True)
    record_spec(
        spec_id=spec_id, spec_run_id=run_id,
        spec_tree_path="specification_tree/modules/robustness/functional_form.md#transform",
        baseline_group_id="G1",
        outcome_var="log1p_AcceptedAmount", treatment_var="offered",
        coef=coef, se=se, pval=pval, ci_low=ci_low, ci_high=ci_high,
        n_obs=n_obs, r2=r2, coefs_dict=coefs_dict,
        sample_desc="Full sample (synthetic)",
        fixed_effects="stratum_code",
        controls_desc="Full controls + month dummies",
        cluster_var="stratum_code",
        axis_block_name="functional_form",
        axis_block={"spec_id": spec_id, "family": "transform",
                    "transform": "log1p", "original_var": "AcceptedAmount",
                    "interpretation": "Semi-elasticity: % change in (1+AcceptedAmount) from loan offer"}
    )
    print(f"  {spec_id}: coef={coef:.4f}")
except Exception as e:
    record_failure(spec_id, run_id,
                   "specification_tree/modules/robustness/functional_form.md#transform",
                   "G1", "log1p_AcceptedAmount", "offered", str(e))


# ============================================================
# G2: ITT ESTIMATES — Effect on educational attainment
# Baseline: areg outcome package + controls, cluster(stratum_code) a(stratum_code)
# Sample: enrolled_fall == 1
# ============================================================

print("\n=== G2: ITT Attainment Specifications ===")

df_att = df[df['enrolled_fall'] == 1].copy()
print(f"G2 sample (enrolled_fall==1): N={len(df_att)}")

G2_OUTCOMES = {
    'crdattm_total': {
        'spec_id': 'baseline',
        'label': 'Table7-PanelA-crdattm_total',
    },
    'credits_total': {
        'spec_id': 'baseline__credits_total',
        'label': 'Table7-credits_total',
    },
    'gpa_total': {
        'spec_id': 'baseline__gpa_total',
        'label': 'Table7-gpa_total',
    },
    'anydeg': {
        'spec_id': 'baseline__anydeg',
        'label': 'Table7-anydeg',
    },
}

# --- G2 Baselines ---
for outcome_var, info in G2_OUTCOMES.items():
    spec_id = info['spec_id']
    run_id = next_spec_run_id()
    try:
        model = run_g2_ols(df_att, outcome_var, CONTROLS_ALL, fe_var='stratum_code',
                           cluster_var='stratum_code', include_month=True)
        coef, se, pval, ci_low, ci_high, n_obs, r2, coefs_dict = get_model_results(
            model, 'package', is_pyfixest=True)
        record_spec(
            spec_id=spec_id, spec_run_id=run_id,
            spec_tree_path="specification_tree/methods/randomized_experiment.md#itt_ols",
            baseline_group_id="G2",
            outcome_var=outcome_var, treatment_var="package",
            coef=coef, se=se, pval=pval, ci_low=ci_low, ci_high=ci_high,
            n_obs=n_obs, r2=r2, coefs_dict=coefs_dict,
            sample_desc="enrolled_fall==1 (synthetic)",
            fixed_effects="stratum_code",
            controls_desc="month_packaged + Prmry_EFC + CumulativeGPA + CumulativeEarnedHours + pell_elig + indep + has_outstanding",
            cluster_var="stratum_code",
            payload_extra={"synthetic_data": True, "baseline_label": info['label']}
        )
        print(f"  {spec_id} ({outcome_var}): coef={coef:.4f}, se={se:.4f}, p={pval:.4f}, N={n_obs}")
    except Exception as e:
        record_failure(spec_id, run_id,
                       "specification_tree/methods/randomized_experiment.md#itt_ols",
                       "G2", outcome_var, "package", str(e))
        print(f"  FAILED {spec_id}: {e}")


# --- G2 Design variants ---
print("\n--- G2 Design Variants ---")
for outcome_var in G2_OUTCOMES:
    # Diff-in-means: no controls, no FE
    spec_id = "design/randomized_experiment/estimator/diff_in_means"
    run_id = next_spec_run_id()
    try:
        model = pf.feols(f"{outcome_var} ~ package", data=df_att, vcov="hetero")
        coef, se, pval, ci_low, ci_high, n_obs, r2, coefs_dict = get_model_results(
            model, 'package', is_pyfixest=True)
        record_spec(
            spec_id=spec_id, spec_run_id=run_id,
            spec_tree_path="specification_tree/methods/randomized_experiment.md#diff_in_means",
            baseline_group_id="G2",
            outcome_var=outcome_var, treatment_var="package",
            coef=coef, se=se, pval=pval, ci_low=ci_low, ci_high=ci_high,
            n_obs=n_obs, r2=r2, coefs_dict=coefs_dict,
            sample_desc="enrolled_fall==1 (synthetic), diff-in-means",
            fixed_effects="",
            controls_desc="none",
            cluster_var="",
            design_block={"estimator": "diff_in_means"}
        )
        print(f"  {spec_id} ({outcome_var}): coef={coef:.4f}")
    except Exception as e:
        record_failure(spec_id, run_id,
                       "specification_tree/methods/randomized_experiment.md#diff_in_means",
                       "G2", outcome_var, "package", str(e))

    # Strata FE (include strata as dummies, not absorbed)
    spec_id = "design/randomized_experiment/estimator/strata_fe"
    run_id = next_spec_run_id()
    try:
        model = run_g2_ols(df_att, outcome_var, CONTROLS_ALL, fe_var='stratum_code',
                           cluster_var='stratum_code', include_month=True)
        coef, se, pval, ci_low, ci_high, n_obs, r2, coefs_dict = get_model_results(
            model, 'package', is_pyfixest=True)
        record_spec(
            spec_id=spec_id, spec_run_id=run_id,
            spec_tree_path="specification_tree/methods/randomized_experiment.md#strata_fe",
            baseline_group_id="G2",
            outcome_var=outcome_var, treatment_var="package",
            coef=coef, se=se, pval=pval, ci_low=ci_low, ci_high=ci_high,
            n_obs=n_obs, r2=r2, coefs_dict=coefs_dict,
            sample_desc="enrolled_fall==1 (synthetic), strata FE",
            fixed_effects="stratum_code",
            controls_desc="Full controls + month dummies + strata FE",
            cluster_var="stratum_code",
            design_block={"estimator": "strata_fe_ols", "notes": "Strata FE included (identical to baseline for absorbed FE)"}
        )
        print(f"  {spec_id} ({outcome_var}): coef={coef:.4f}")
    except Exception as e:
        record_failure(spec_id, run_id,
                       "specification_tree/methods/randomized_experiment.md#strata_fe",
                       "G2", outcome_var, "package", str(e))


# --- G2 RC: LOO Controls ---
print("\n--- G2 LOO Controls ---")
for outcome_var in G2_OUTCOMES:
    for drop_var, spec_id in loo_controls.items():
        run_id = next_spec_run_id()
        try:
            ctrls = [c for c in CONTROLS_ALL if c != drop_var]
            model = run_g2_ols(df_att, outcome_var, ctrls, fe_var='stratum_code',
                               cluster_var='stratum_code', include_month=True)
            coef, se, pval, ci_low, ci_high, n_obs, r2, coefs_dict = get_model_results(
                model, 'package', is_pyfixest=True)
            record_spec(
                spec_id=spec_id, spec_run_id=run_id,
                spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
                baseline_group_id="G2",
                outcome_var=outcome_var, treatment_var="package",
                coef=coef, se=se, pval=pval, ci_low=ci_low, ci_high=ci_high,
                n_obs=n_obs, r2=r2, coefs_dict=coefs_dict,
                sample_desc="enrolled_fall==1 (synthetic)",
                fixed_effects="stratum_code",
                controls_desc=f"LOO: dropped {drop_var}",
                cluster_var="stratum_code",
                axis_block_name="controls",
                axis_block={"spec_id": spec_id, "family": "loo", "dropped": [drop_var],
                            "n_controls": len(ctrls) + 1}
            )
        except Exception as e:
            record_failure(spec_id, run_id,
                           "specification_tree/modules/robustness/controls.md#loo",
                           "G2", outcome_var, "package", str(e))

    # Drop month_packaged
    spec_id = "rc/controls/loo/drop_month_packaged"
    run_id = next_spec_run_id()
    try:
        model = run_g2_ols(df_att, outcome_var, CONTROLS_ALL, fe_var='stratum_code',
                           cluster_var='stratum_code', include_month=False)
        coef, se, pval, ci_low, ci_high, n_obs, r2, coefs_dict = get_model_results(
            model, 'package', is_pyfixest=True)
        record_spec(
            spec_id=spec_id, spec_run_id=run_id,
            spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
            baseline_group_id="G2",
            outcome_var=outcome_var, treatment_var="package",
            coef=coef, se=se, pval=pval, ci_low=ci_low, ci_high=ci_high,
            n_obs=n_obs, r2=r2, coefs_dict=coefs_dict,
            sample_desc="enrolled_fall==1 (synthetic)",
            fixed_effects="stratum_code",
            controls_desc="LOO: dropped month_packaged dummies",
            cluster_var="stratum_code",
            axis_block_name="controls",
            axis_block={"spec_id": spec_id, "family": "loo", "dropped": ["month_packaged"],
                        "n_controls": len(CONTROLS_ALL)}
        )
    except Exception as e:
        record_failure(spec_id, run_id,
                       "specification_tree/modules/robustness/controls.md#loo",
                       "G2", outcome_var, "package", str(e))


# --- G2 RC: Control sets ---
print("\n--- G2 Control Sets ---")
g2_control_sets = {
    'rc/controls/sets/none': [],
    'rc/controls/sets/full': CONTROLS_ALL.copy(),
}

for outcome_var in G2_OUTCOMES:
    for spec_id, ctrls in g2_control_sets.items():
        run_id = next_spec_run_id()
        include_month = spec_id != 'rc/controls/sets/none'
        try:
            model = run_g2_ols(df_att, outcome_var, ctrls, fe_var='stratum_code',
                               cluster_var='stratum_code', include_month=include_month)
            coef, se, pval, ci_low, ci_high, n_obs, r2, coefs_dict = get_model_results(
                model, 'package', is_pyfixest=True)
            record_spec(
                spec_id=spec_id, spec_run_id=run_id,
                spec_tree_path="specification_tree/modules/robustness/controls.md#sets",
                baseline_group_id="G2",
                outcome_var=outcome_var, treatment_var="package",
                coef=coef, se=se, pval=pval, ci_low=ci_low, ci_high=ci_high,
                n_obs=n_obs, r2=r2, coefs_dict=coefs_dict,
                sample_desc="enrolled_fall==1 (synthetic)",
                fixed_effects="stratum_code",
                controls_desc=f"Control set: {spec_id.split('/')[-1]}",
                cluster_var="stratum_code",
                axis_block_name="controls",
                axis_block={"spec_id": spec_id, "family": "sets",
                            "included": ctrls + (['month_packaged'] if include_month else []),
                            "n_controls": len(ctrls) + (1 if include_month else 0)}
            )
        except Exception as e:
            record_failure(spec_id, run_id,
                           "specification_tree/modules/robustness/controls.md#sets",
                           "G2", outcome_var, "package", str(e))


# --- G2 RC: Control progression ---
print("\n--- G2 Control Progression ---")
g2_progression = {
    'rc/controls/progression/bivariate': [],
    'rc/controls/progression/all_controls': CONTROLS_ALL.copy(),
}

for outcome_var in G2_OUTCOMES:
    for spec_id, ctrls in g2_progression.items():
        run_id = next_spec_run_id()
        include_month = spec_id != 'rc/controls/progression/bivariate'
        try:
            model = run_g2_ols(df_att, outcome_var, ctrls, fe_var='stratum_code',
                               cluster_var='stratum_code', include_month=include_month)
            coef, se, pval, ci_low, ci_high, n_obs, r2, coefs_dict = get_model_results(
                model, 'package', is_pyfixest=True)
            record_spec(
                spec_id=spec_id, spec_run_id=run_id,
                spec_tree_path="specification_tree/modules/robustness/controls.md#progression",
                baseline_group_id="G2",
                outcome_var=outcome_var, treatment_var="package",
                coef=coef, se=se, pval=pval, ci_low=ci_low, ci_high=ci_high,
                n_obs=n_obs, r2=r2, coefs_dict=coefs_dict,
                sample_desc="enrolled_fall==1 (synthetic)",
                fixed_effects="stratum_code",
                controls_desc=f"Progression: {spec_id.split('/')[-1]}",
                cluster_var="stratum_code",
                axis_block_name="controls",
                axis_block={"spec_id": spec_id, "family": "progression",
                            "included": ctrls + (['month_packaged'] if include_month else []),
                            "n_controls": len(ctrls) + (1 if include_month else 0)}
            )
        except Exception as e:
            record_failure(spec_id, run_id,
                           "specification_tree/modules/robustness/controls.md#progression",
                           "G2", outcome_var, "package", str(e))


# --- G2 RC: Sample restriction (enrolled_fall) ---
print("\n--- G2 Sample Restriction ---")
for outcome_var in G2_OUTCOMES:
    spec_id = "rc/sample/restriction/enrolled_fall"
    run_id = next_spec_run_id()
    try:
        # This is actually the same as baseline (already restricted to enrolled_fall==1)
        # But run on full sample to show effect of NOT restricting
        model = run_g2_ols(df, outcome_var, CONTROLS_ALL, fe_var='stratum_code',
                           cluster_var='stratum_code', include_month=True)
        coef, se, pval, ci_low, ci_high, n_obs, r2, coefs_dict = get_model_results(
            model, 'package', is_pyfixest=True)
        record_spec(
            spec_id=spec_id, spec_run_id=run_id,
            spec_tree_path="specification_tree/modules/robustness/sample.md#restriction",
            baseline_group_id="G2",
            outcome_var=outcome_var, treatment_var="package",
            coef=coef, se=se, pval=pval, ci_low=ci_low, ci_high=ci_high,
            n_obs=n_obs, r2=r2, coefs_dict=coefs_dict,
            sample_desc="Full sample (no enrolled_fall restriction, synthetic)",
            fixed_effects="stratum_code",
            controls_desc="Full controls + month dummies",
            cluster_var="stratum_code",
            axis_block_name="sample",
            axis_block={"spec_id": spec_id, "family": "restriction",
                        "dropped_restriction": "enrolled_fall==1",
                        "notes": "Runs on full sample without enrollment restriction"}
        )
        print(f"  {spec_id} ({outcome_var}): coef={coef:.4f}, N={n_obs}")
    except Exception as e:
        record_failure(spec_id, run_id,
                       "specification_tree/modules/robustness/sample.md#restriction",
                       "G2", outcome_var, "package", str(e))


# --- G2 RC: Sample trimming ---
print("\n--- G2 Sample Trimming ---")
for outcome_var in G2_OUTCOMES:
    spec_id = "rc/sample/outliers/trim_y_1_99"
    run_id = next_spec_run_id()
    try:
        q_lo = df_att[outcome_var].quantile(0.01)
        q_hi = df_att[outcome_var].quantile(0.99)
        df_trim = df_att[(df_att[outcome_var] >= q_lo) & (df_att[outcome_var] <= q_hi)].copy()
        model = run_g2_ols(df_trim, outcome_var, CONTROLS_ALL, fe_var='stratum_code',
                           cluster_var='stratum_code', include_month=True)
        coef, se, pval, ci_low, ci_high, n_obs, r2, coefs_dict = get_model_results(
            model, 'package', is_pyfixest=True)
        record_spec(
            spec_id=spec_id, spec_run_id=run_id,
            spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
            baseline_group_id="G2",
            outcome_var=outcome_var, treatment_var="package",
            coef=coef, se=se, pval=pval, ci_low=ci_low, ci_high=ci_high,
            n_obs=n_obs, r2=r2, coefs_dict=coefs_dict,
            sample_desc=f"enrolled_fall==1, trimmed {outcome_var} at [1, 99] percentiles",
            fixed_effects="stratum_code",
            controls_desc="Full controls + month dummies",
            cluster_var="stratum_code",
            axis_block_name="sample",
            axis_block={"spec_id": spec_id, "family": "outliers",
                        "trim_lower": 0.01, "trim_upper": 0.99}
        )
    except Exception as e:
        record_failure(spec_id, run_id,
                       "specification_tree/modules/robustness/sample.md#outliers",
                       "G2", outcome_var, "package", str(e))


# --- G2 RC: Drop stratum FE ---
print("\n--- G2 Drop Stratum FE ---")
for outcome_var in G2_OUTCOMES:
    spec_id = "rc/fe/drop/stratum"
    run_id = next_spec_run_id()
    try:
        model = run_g2_ols(df_att, outcome_var, CONTROLS_ALL, fe_var=None,
                           cluster_var='stratum_code', include_month=True)
        coef, se, pval, ci_low, ci_high, n_obs, r2, coefs_dict = get_model_results(
            model, 'package', is_pyfixest=True)
        record_spec(
            spec_id=spec_id, spec_run_id=run_id,
            spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#drop",
            baseline_group_id="G2",
            outcome_var=outcome_var, treatment_var="package",
            coef=coef, se=se, pval=pval, ci_low=ci_low, ci_high=ci_high,
            n_obs=n_obs, r2=r2, coefs_dict=coefs_dict,
            sample_desc="enrolled_fall==1 (synthetic)",
            fixed_effects="none",
            controls_desc="Full controls + month dummies, no stratum FE",
            cluster_var="stratum_code",
            axis_block_name="fixed_effects",
            axis_block={"spec_id": spec_id, "family": "drop", "dropped": ["stratum_code"]}
        )
        print(f"  {spec_id} ({outcome_var}): coef={coef:.4f}")
    except Exception as e:
        record_failure(spec_id, run_id,
                       "specification_tree/modules/robustness/fixed_effects.md#drop",
                       "G2", outcome_var, "package", str(e))


# ============================================================
# INFERENCE VARIANTS (both G1 and G2)
# HC1 robust SEs (no clustering)
# ============================================================

print("\n=== Inference Variants (HC1) ===")

# G1 baselines with HC1
for outcome_var, info in G1_OUTCOMES.items():
    base_spec_id = info['spec_id']
    # Find the base spec_run_id
    base_rows = [r for r in spec_results if r['spec_id'] == base_spec_id
                 and r['baseline_group_id'] == 'G1' and r['outcome_var'] == outcome_var
                 and r['run_success'] == 1]
    if base_rows:
        base_run_id = base_rows[0]['spec_run_id']
        try:
            model = run_g1_iv(df, outcome_var, CONTROLS_ALL, fe_var='stratum_code',
                              cluster_var=None, include_month=True)
            coef, se, pval, ci_low, ci_high, n_obs, r2, coefs_dict = get_model_results(
                model, 'offered', is_pyfixest=True)
            record_inference(
                base_run_id, "infer/se/hc/hc1",
                "specification_tree/modules/inference/se.md#hc1",
                "G1", outcome_var, "offered",
                coef, se, pval, ci_low, ci_high, n_obs, r2, coefs_dict,
                {"spec_id": "infer/se/hc/hc1", "params": {}}
            )
            print(f"  HC1 G1 ({outcome_var}): se={se:.4f}, p={pval:.4f}")
        except Exception as e:
            print(f"  FAILED HC1 G1 ({outcome_var}): {e}")

# G2 baselines with HC1
for outcome_var, info in G2_OUTCOMES.items():
    base_spec_id = info['spec_id']
    base_rows = [r for r in spec_results if r['spec_id'] == base_spec_id
                 and r['baseline_group_id'] == 'G2' and r['outcome_var'] == outcome_var
                 and r['run_success'] == 1]
    if base_rows:
        base_run_id = base_rows[0]['spec_run_id']
        try:
            model = run_g2_ols(df_att, outcome_var, CONTROLS_ALL, fe_var='stratum_code',
                               cluster_var=None, include_month=True)
            coef, se, pval, ci_low, ci_high, n_obs, r2, coefs_dict = get_model_results(
                model, 'package', is_pyfixest=True)
            record_inference(
                base_run_id, "infer/se/hc/hc1",
                "specification_tree/modules/inference/se.md#hc1",
                "G2", outcome_var, "package",
                coef, se, pval, ci_low, ci_high, n_obs, r2, coefs_dict,
                {"spec_id": "infer/se/hc/hc1", "params": {}}
            )
            print(f"  HC1 G2 ({outcome_var}): se={se:.4f}, p={pval:.4f}")
        except Exception as e:
            print(f"  FAILED HC1 G2 ({outcome_var}): {e}")


# ============================================================
# WRITE OUTPUTS
# ============================================================

print("\n=== Writing Outputs ===")

# specification_results.csv
spec_df = pd.DataFrame(spec_results)
spec_df.to_csv(f"{OUTPUT_DIR}/specification_results.csv", index=False)
print(f"specification_results.csv: {len(spec_df)} rows ({spec_df['run_success'].sum()} success, {(1-spec_df['run_success']).sum()} failed)")

# inference_results.csv
infer_df = pd.DataFrame(infer_results)
infer_df.to_csv(f"{OUTPUT_DIR}/inference_results.csv", index=False)
print(f"inference_results.csv: {len(infer_df)} rows")

# Counts by group
for g in ['G1', 'G2']:
    g_rows = spec_df[spec_df['baseline_group_id'] == g]
    print(f"  {g}: {len(g_rows)} specs ({g_rows['run_success'].sum()} success)")

# SPECIFICATION_SEARCH.md
n_total = len(spec_df)
n_success = int(spec_df['run_success'].sum())
n_failed = n_total - n_success
n_infer = len(infer_df)
n_g1 = len(spec_df[spec_df['baseline_group_id'] == 'G1'])
n_g2 = len(spec_df[spec_df['baseline_group_id'] == 'G2'])

search_md = f"""# Specification Search: {PAPER_ID}

## Paper
Marx & Turner (2019), "Student Loan Nudges: Experimental Evidence on Borrowing
and Educational Attainment," *American Economic Review*, 109(2), 566-592.

## DATA NOTE
The student-level analysis data is **CONFIDENTIAL** (provided by an anonymous
community college). Only `T1_data_packaging_practices.dta` (institution-level
packaging practices, Table 1 only) is included in the replication package.

All specifications were run on **SYNTHETIC DATA** calibrated to the paper's
published summary statistics (Table 3 control group means, reported sample sizes,
approximate coefficient magnitudes). Results validate the specification surface
and estimation pipeline but coefficients will not exactly match published values.

## Surface Summary
- **Paper ID**: {PAPER_ID}
- **Baseline groups**: 2 (G1: IV borrowing, G2: ITT attainment)
- **G1 design**: Stratified RCT, IV (package instruments for offered)
- **G2 design**: Stratified RCT, ITT/OLS (package -> attainment)
- **Canonical inference**: cluster(stratum_code) for both groups
- **Seed**: 116531
- **Surface hash**: {SURFACE_HASH}

### G1: Effect of loan offers on borrowing
- Baseline outcomes: borrowed, AcceptedAmount
- Treatment: offered (endogenous, instrumented by package)
- Controls: month_packaged dummies, Prmry_EFC, CumulativeGPA, CumulativeEarnedHours, pell_elig, indep, has_outstanding
- FE: stratum_code
- Budget: max 80 core specs, 10 control subset draws

### G2: ITT effect on educational attainment
- Baseline outcomes: crdattm_total, credits_total, gpa_total, anydeg
- Treatment: package (random assignment)
- Sample: enrolled_fall == 1
- Budget: max 50 core specs

## Execution Summary
- **Total specification rows**: {n_total}
- **Successful**: {n_success}
- **Failed**: {n_failed}
- **Inference variant rows**: {n_infer}
- **G1 specifications**: {n_g1}
- **G2 specifications**: {n_g2}

### Spec breakdown by type
| Type | Count |
|------|-------|
| baseline | {len(spec_df[spec_df['spec_id'].str.startswith('baseline')])} |
| design/* | {len(spec_df[spec_df['spec_id'].str.startswith('design/')])} |
| rc/controls/loo/* | {len(spec_df[spec_df['spec_id'].str.startswith('rc/controls/loo/')])} |
| rc/controls/sets/* | {len(spec_df[spec_df['spec_id'].str.startswith('rc/controls/sets/')])} |
| rc/controls/progression/* | {len(spec_df[spec_df['spec_id'].str.startswith('rc/controls/progression/')])} |
| rc/controls/subset/* | {len(spec_df[spec_df['spec_id'].str.startswith('rc/controls/subset/')])} |
| rc/sample/* | {len(spec_df[spec_df['spec_id'].str.startswith('rc/sample/')])} |
| rc/fe/* | {len(spec_df[spec_df['spec_id'].str.startswith('rc/fe/')])} |
| rc/form/* | {len(spec_df[spec_df['spec_id'].str.startswith('rc/form/')])} |
| infer/* (separate table) | {n_infer} |

## Deviations
- **Synthetic data**: All results use synthetic data because the student-level
  analysis data is confidential. The Stata do-files reference
  `formatted_analysis_data.dta` which is not included in the replication package.
- The xtivreg2 command in Stata uses the `partial()` option to partial out
  controls before IV estimation. In pyfixest, controls in the IV formula are
  included directly (not partialled), which is algebraically equivalent but may
  produce minor numerical differences.

## Software Stack
- Python {sys.version.split()[0]}
- pyfixest {SW_BLOCK['packages'].get('pyfixest', 'unknown')}
- pandas {SW_BLOCK['packages'].get('pandas', 'unknown')}
- numpy {SW_BLOCK['packages'].get('numpy', 'unknown')}
- statsmodels {SW_BLOCK['packages'].get('statsmodels', 'unknown')}
"""

with open(f"{OUTPUT_DIR}/SPECIFICATION_SEARCH.md", 'w') as f:
    f.write(search_md)

print(f"\nDone! {n_total} specs + {n_infer} inference variants written.")
print(f"Outputs in: {OUTPUT_DIR}/")
