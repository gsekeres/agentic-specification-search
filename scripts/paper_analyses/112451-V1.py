"""
Specification Search Script for Furman & Stern (2011)
"Biological Resource Centers and the Advancement of Technological Knowledge"
RAND Journal of Economics, 42(1), 1-32.

Paper ID: 112451-V1

Surface-driven execution:
  - G1: Effect of BRC Deposit on Citations (DiD with article FE)
  - Baseline: Poisson FE (proxy for paper's conditional FE negative binomial)
  - The paper uses xtnbreg (conditional FE negative binomial) which is not
    available in Python. Poisson FE (pyfixest fepois) is the closest available
    estimator and a common robustness check for count data models.
  - RC axes: controls, sample, functional form, FE, joint variations

Outputs:
  - specification_results.csv (baseline, design/*, rc/* rows)
  - inference_results.csv (infer/* rows)
  - SPECIFICATION_SEARCH.md
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import sys
import warnings
warnings.filterwarnings('ignore')

# Add scripts dir for utilities
sys.path.insert(0, "scripts")
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash, software_block
)

PAPER_ID = "112451-V1"
DATA_DIR = "data/downloads/extracted/112451-V1"
OUTPUT_DIR = DATA_DIR

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

# Design audit from surface
G1_DESIGN_AUDIT = surface_obj["baseline_groups"][0]["design_audit"]
G1_INFERENCE_CANONICAL = surface_obj["baseline_groups"][0]["inference_plan"]["canonical"]

# ============================================================
# LOAD DATA
# ============================================================
df_raw = pd.read_stata(f"{DATA_DIR}/data/brc_core.dta")
for col in df_raw.columns:
    if df_raw[col].dtype == np.float32:
        df_raw[col] = df_raw[col].astype(np.float64)

# Core sample: sample3==1
df_s3 = df_raw[df_raw['sample3'] == 1].copy()
df_s3 = df_s3.dropna(subset=['cites'])  # 23 NAs

# Derived outcomes
df_s3['log_cites_plus1'] = np.log(df_s3['cites'] + 1)
df_s3['asinh_cites'] = np.arcsinh(df_s3['cites'])

# Sample0 (BRC-only articles)
df_s0 = df_raw[df_raw['sample0'] == 1].copy()
df_s0 = df_s0.dropna(subset=['cites'])
df_s0['log_cites_plus1'] = np.log(df_s0['cites'] + 1)
df_s0['asinh_cites'] = np.arcsinh(df_s0['cites'])

# Sample0, window==0 (for Table 4 Cols 3-4)
df_s0_nowin = df_s0[df_s0['window'] == 0].copy()

# ============================================================
# VARIABLE LISTS
# ============================================================
AGE_DUMMIES = [f'age{i}' for i in range(1, 31)]  # 30 age dummies

YEAR_DUMMIES = [
    'year7074', 'year7579',
    'year1980', 'year1981', 'year1982', 'year1983', 'year1984', 'year1985',
    'year1986', 'year1987', 'year1988', 'year1989',
    'year1991', 'year1992', 'year1993',
    'year1994', 'year1995', 'year1996', 'year1997',
    'year1998', 'year1999', 'year2000', 'year2001'
]  # 23 year dummies (omit 1990)

GROUPED_YEAR_DUMMIES = ['year7074', 'y7579', 'y8084', 'y8589', 'y9094']

# Winsorized cites (top 1%)
cites_99 = df_s3['cites'].quantile(0.99)
df_s3['cites_w99'] = df_s3['cites'].clip(upper=cites_99)
df_s3['log_cites_w99_plus1'] = np.log(df_s3['cites_w99'] + 1)

# Trimmed sample (drop top 1% of cites)
df_s3_trim99 = df_s3[df_s3['cites'] <= cites_99].copy()

results = []
inference_results = []
spec_run_counter = 0


# ============================================================
# HELPER: Run Poisson FE
# ============================================================
def run_poisson_fe(spec_id, spec_tree_path, baseline_group_id,
                   outcome_var, treatment_var, controls, fe_formula,
                   data, vcov, sample_desc, controls_desc, cluster_var,
                   design_audit, inference_canonical,
                   axis_block_name=None, axis_block=None,
                   functional_form_block=None, notes=""):
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        controls_str = " + ".join(controls) if controls else ""
        if controls_str and fe_formula:
            formula = f"{outcome_var} ~ {treatment_var} + {controls_str} | {fe_formula}"
        elif controls_str:
            formula = f"{outcome_var} ~ {treatment_var} + {controls_str}"
        elif fe_formula:
            formula = f"{outcome_var} ~ {treatment_var} | {fe_formula}"
        else:
            formula = f"{outcome_var} ~ {treatment_var}"

        m = pf.fepois(formula, data=data, vcov=vcov)
        coef_val = float(m.coef().get(treatment_var, np.nan))
        se_val = float(m.se().get(treatment_var, np.nan))
        pval = float(m.pvalue().get(treatment_var, np.nan))
        try:
            ci = m.confint()
            ci_lower = float(ci.loc[treatment_var, ci.columns[0]])
            ci_upper = float(ci.loc[treatment_var, ci.columns[1]])
        except Exception:
            ci_lower, ci_upper = np.nan, np.nan
        nobs = int(m._N)
        try:
            r2 = float(m._r2)
        except Exception:
            r2 = np.nan
        all_coefs = {k: float(v) for k, v in m.coef().items()}

        blocks = {}
        if axis_block_name and axis_block:
            blocks[axis_block_name] = axis_block
        if functional_form_block:
            blocks["functional_form"] = functional_form_block

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": inference_canonical["spec_id"],
                       "params": inference_canonical.get("params", {})},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"difference_in_differences": design_audit},
            blocks=blocks,
            notes=notes if notes else None,
        )

        results.append({
            "paper_id": PAPER_ID, "spec_run_id": run_id,
            "spec_id": spec_id, "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var, "treatment_var": treatment_var,
            "coefficient": coef_val, "std_error": se_val, "p_value": pval,
            "ci_lower": ci_lower, "ci_upper": ci_upper,
            "n_obs": nobs, "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc, "fixed_effects": fe_formula or "none",
            "controls_desc": controls_desc, "cluster_var": cluster_var,
            "run_success": 1, "run_error": ""
        })
        return run_id, coef_val, se_val, pval, nobs

    except Exception as e:
        err_msg = str(e)[:240]
        payload = make_failure_payload(
            error=err_msg,
            error_details=error_details_from_exception(e, stage="estimation"),
            software=SW_BLOCK, surface_hash=SURFACE_HASH
        )
        results.append({
            "paper_id": PAPER_ID, "spec_run_id": run_id,
            "spec_id": spec_id, "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var, "treatment_var": treatment_var,
            "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "n_obs": np.nan, "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc, "fixed_effects": fe_formula or "none",
            "controls_desc": controls_desc, "cluster_var": cluster_var,
            "run_success": 0, "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


# ============================================================
# HELPER: Run OLS FE
# ============================================================
def run_ols_fe(spec_id, spec_tree_path, baseline_group_id,
               outcome_var, treatment_var, controls, fe_formula,
               data, vcov, sample_desc, controls_desc, cluster_var,
               design_audit, inference_canonical,
               axis_block_name=None, axis_block=None,
               functional_form_block=None, notes=""):
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        controls_str = " + ".join(controls) if controls else ""
        if controls_str and fe_formula:
            formula = f"{outcome_var} ~ {treatment_var} + {controls_str} | {fe_formula}"
        elif controls_str:
            formula = f"{outcome_var} ~ {treatment_var} + {controls_str}"
        elif fe_formula:
            formula = f"{outcome_var} ~ {treatment_var} | {fe_formula}"
        else:
            formula = f"{outcome_var} ~ {treatment_var}"

        m = pf.feols(formula, data=data, vcov=vcov)
        coef_val = float(m.coef().get(treatment_var, np.nan))
        se_val = float(m.se().get(treatment_var, np.nan))
        pval = float(m.pvalue().get(treatment_var, np.nan))
        try:
            ci = m.confint()
            ci_lower = float(ci.loc[treatment_var, ci.columns[0]])
            ci_upper = float(ci.loc[treatment_var, ci.columns[1]])
        except Exception:
            ci_lower, ci_upper = np.nan, np.nan
        nobs = int(m._N)
        try:
            r2 = float(m._r2)
        except Exception:
            r2 = np.nan
        all_coefs = {k: float(v) for k, v in m.coef().items()}

        blocks = {}
        if axis_block_name and axis_block:
            blocks[axis_block_name] = axis_block
        if functional_form_block:
            blocks["functional_form"] = functional_form_block

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": inference_canonical["spec_id"],
                       "params": inference_canonical.get("params", {})},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"difference_in_differences": design_audit},
            blocks=blocks,
            notes=notes if notes else None,
        )

        results.append({
            "paper_id": PAPER_ID, "spec_run_id": run_id,
            "spec_id": spec_id, "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var, "treatment_var": treatment_var,
            "coefficient": coef_val, "std_error": se_val, "p_value": pval,
            "ci_lower": ci_lower, "ci_upper": ci_upper,
            "n_obs": nobs, "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc, "fixed_effects": fe_formula or "none",
            "controls_desc": controls_desc, "cluster_var": cluster_var,
            "run_success": 1, "run_error": ""
        })
        return run_id, coef_val, se_val, pval, nobs

    except Exception as e:
        err_msg = str(e)[:240]
        payload = make_failure_payload(
            error=err_msg,
            error_details=error_details_from_exception(e, stage="estimation"),
            software=SW_BLOCK, surface_hash=SURFACE_HASH
        )
        results.append({
            "paper_id": PAPER_ID, "spec_run_id": run_id,
            "spec_id": spec_id, "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var, "treatment_var": treatment_var,
            "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "n_obs": np.nan, "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc, "fixed_effects": fe_formula or "none",
            "controls_desc": controls_desc, "cluster_var": cluster_var,
            "run_success": 0, "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


# ============================================================
# HELPER: Add inference variant row
# ============================================================
def add_inference_row(base_run_id, spec_id, spec_tree_path, baseline_group_id,
                      outcome_var, treatment_var, model, infer_spec,
                      nobs, r2, sample_desc, fe_str, controls_desc, cluster_var):
    global spec_run_counter
    spec_run_counter += 1
    infer_run_id = f"{PAPER_ID}_infer_{spec_run_counter:03d}"

    try:
        coef_val = float(model.coef().get(treatment_var, np.nan))
        se_val = float(model.se().get(treatment_var, np.nan))
        pval = float(model.pvalue().get(treatment_var, np.nan))
        try:
            ci = model.confint()
            ci_lower = float(ci.loc[treatment_var, ci.columns[0]])
            ci_upper = float(ci.loc[treatment_var, ci.columns[1]])
        except Exception:
            ci_lower, ci_upper = np.nan, np.nan

        all_coefs = {k: float(v) for k, v in model.coef().items()}
        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": infer_spec["spec_id"],
                       "params": infer_spec.get("params", {})},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"difference_in_differences": G1_DESIGN_AUDIT},
        )

        inference_results.append({
            "paper_id": PAPER_ID,
            "inference_run_id": infer_run_id,
            "spec_run_id": base_run_id,
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
            "run_success": 1,
            "run_error": ""
        })
    except Exception as e:
        err_msg = str(e)[:240]
        payload = make_failure_payload(
            error=err_msg,
            error_details=error_details_from_exception(e, stage="inference"),
            software=SW_BLOCK, surface_hash=SURFACE_HASH
        )
        inference_results.append({
            "paper_id": PAPER_ID,
            "inference_run_id": infer_run_id,
            "spec_run_id": base_run_id,
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
            "run_success": 0,
            "run_error": err_msg
        })


# ============================================================
# BASELINE CONTROLS (T4C1 equivalent)
# ============================================================
# Table 4, Col 1: window post_brc age_brc1 $age1 $year1 | rart_num
BASELINE_CONTROLS = ['window', 'age_brc1'] + AGE_DUMMIES + YEAR_DUMMIES
BASELINE_CONTROLS_DESC = "window + age_brc1 + 30 age dummies + 23 year dummies"

# ============================================================
# SPEC 1: BASELINE (Poisson FE proxy for paper's NB FE)
# ============================================================
print("Running baseline...")
run_poisson_fe(
    spec_id="baseline",
    spec_tree_path="specification_tree/designs/difference_in_differences.md",
    baseline_group_id="G1",
    outcome_var="cites",
    treatment_var="post_brc",
    controls=BASELINE_CONTROLS,
    fe_formula="rart_num",
    data=df_s3,
    vcov="hetero",
    sample_desc="sample3==1, Poisson FE (proxy for xtnbreg FE)",
    controls_desc=BASELINE_CONTROLS_DESC,
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    notes="Baseline: Poisson FE proxy for paper's conditional FE negative binomial (xtnbreg). Table 4, Col 1 equivalent."
)

# ============================================================
# SPEC 2: baseline__table4_col2 (add post_brc_yrs)
# ============================================================
print("Running baseline Table4 Col2...")
run_poisson_fe(
    spec_id="baseline__table4_col2",
    spec_tree_path="specification_tree/designs/difference_in_differences.md",
    baseline_group_id="G1",
    outcome_var="cites",
    treatment_var="post_brc",
    controls=BASELINE_CONTROLS + ['post_brc_yrs'],
    fe_formula="rart_num",
    data=df_s3,
    vcov="hetero",
    sample_desc="sample3==1, Poisson FE",
    controls_desc=BASELINE_CONTROLS_DESC + " + post_brc_yrs",
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    notes="Table 4, Col 2 equivalent: adds time-since-deposit trend (post_brc_yrs)."
)

# ============================================================
# SPEC 3: design/difference_in_differences/estimator/twfe
# OLS on cites (linear TWFE)
# ============================================================
print("Running design TWFE (OLS on cites)...")
run_ols_fe(
    spec_id="design/difference_in_differences/estimator/twfe",
    spec_tree_path="specification_tree/designs/difference_in_differences.md#twfe",
    baseline_group_id="G1",
    outcome_var="cites",
    treatment_var="post_brc",
    controls=BASELINE_CONTROLS,
    fe_formula="rart_num",
    data=df_s3,
    vcov="hetero",
    sample_desc="sample3==1, linear TWFE on cites",
    controls_desc=BASELINE_CONTROLS_DESC,
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    functional_form_block={
        "spec_id": "design/difference_in_differences/estimator/twfe",
        "interpretation": "Linear TWFE: OLS on raw cite counts with article FE",
        "outcome_transform": "none (raw cites)",
    },
    notes="Linear TWFE design variant: OLS on raw cites with article FE."
)

# ============================================================
# RC: CONTROLS VARIANTS
# ============================================================

# rc/controls/single/add_post_brc_yrs
print("Running rc/controls/single/add_post_brc_yrs...")
run_poisson_fe(
    spec_id="rc/controls/single/add_post_brc_yrs",
    spec_tree_path="specification_tree/modules/robustness/controls.md#single",
    baseline_group_id="G1",
    outcome_var="cites",
    treatment_var="post_brc",
    controls=BASELINE_CONTROLS + ['post_brc_yrs'],
    fe_formula="rart_num",
    data=df_s3,
    vcov="hetero",
    sample_desc="sample3==1, Poisson FE",
    controls_desc=BASELINE_CONTROLS_DESC + " + post_brc_yrs",
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/single/add_post_brc_yrs",
                "family": "single", "added": ["post_brc_yrs"],
                "n_controls": len(BASELINE_CONTROLS) + 1},
)

# rc/controls/loo/drop_window
print("Running rc/controls/loo/drop_window...")
controls_no_window = ['age_brc1'] + AGE_DUMMIES + YEAR_DUMMIES
run_poisson_fe(
    spec_id="rc/controls/loo/drop_window",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G1",
    outcome_var="cites",
    treatment_var="post_brc",
    controls=controls_no_window,
    fe_formula="rart_num",
    data=df_s3,
    vcov="hetero",
    sample_desc="sample3==1, Poisson FE",
    controls_desc="age_brc1 + 30 age dummies + 23 year dummies (dropped window)",
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_window",
                "family": "loo", "dropped": ["window"],
                "n_controls": len(controls_no_window)},
)

# rc/controls/loo/drop_age_brc1
print("Running rc/controls/loo/drop_age_brc1...")
controls_no_agebrc = ['window'] + AGE_DUMMIES + YEAR_DUMMIES
run_poisson_fe(
    spec_id="rc/controls/loo/drop_age_brc1",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G1",
    outcome_var="cites",
    treatment_var="post_brc",
    controls=controls_no_agebrc,
    fe_formula="rart_num",
    data=df_s3,
    vcov="hetero",
    sample_desc="sample3==1, Poisson FE",
    controls_desc="window + 30 age dummies + 23 year dummies (dropped age_brc1)",
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_age_brc1",
                "family": "loo", "dropped": ["age_brc1"],
                "n_controls": len(controls_no_agebrc)},
)

# rc/controls/loo/drop_window_and_age_brc1
print("Running rc/controls/loo/drop_window_and_age_brc1...")
controls_minimal = AGE_DUMMIES + YEAR_DUMMIES
run_poisson_fe(
    spec_id="rc/controls/loo/drop_window_and_age_brc1",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G1",
    outcome_var="cites",
    treatment_var="post_brc",
    controls=controls_minimal,
    fe_formula="rart_num",
    data=df_s3,
    vcov="hetero",
    sample_desc="sample3==1, Poisson FE",
    controls_desc="30 age dummies + 23 year dummies (dropped window and age_brc1)",
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_window_and_age_brc1",
                "family": "loo", "dropped": ["window", "age_brc1"],
                "n_controls": len(controls_minimal)},
)

# rc/controls/sets/grouped_age_year (Table 4 Cols 3-4 use grouped age/year)
print("Running rc/controls/sets/grouped_age_year...")
controls_grouped = ['window', 'age_brc1', 'age', 'age_2'] + GROUPED_YEAR_DUMMIES
run_poisson_fe(
    spec_id="rc/controls/sets/grouped_age_year",
    spec_tree_path="specification_tree/modules/robustness/controls.md#sets",
    baseline_group_id="G1",
    outcome_var="cites",
    treatment_var="post_brc",
    controls=controls_grouped,
    fe_formula="rart_num",
    data=df_s3,
    vcov="hetero",
    sample_desc="sample3==1, Poisson FE",
    controls_desc="window + age_brc1 + age + age_2 + 5 grouped year dummies",
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/grouped_age_year",
                "family": "sets",
                "description": "Replace individual age/year dummies with grouped (5yr) dummies + quadratic age",
                "n_controls": len(controls_grouped)},
)

# rc/controls/sets/polynomial_age_year (Table 4 Col 4: age age_2 yr yr_2)
print("Running rc/controls/sets/polynomial_age_year...")
controls_poly = ['age', 'age_2', 'yr', 'yr_2']
run_poisson_fe(
    spec_id="rc/controls/sets/polynomial_age_year",
    spec_tree_path="specification_tree/modules/robustness/controls.md#sets",
    baseline_group_id="G1",
    outcome_var="cites",
    treatment_var="post_brc",
    controls=controls_poly,
    fe_formula="rart_num",
    data=df_s3,
    vcov="hetero",
    sample_desc="sample3==1, Poisson FE",
    controls_desc="age + age_2 + yr + yr_2 (polynomial age/year)",
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/polynomial_age_year",
                "family": "sets",
                "description": "Replace dummies with quadratic age and year polynomials",
                "n_controls": len(controls_poly)},
)

# ============================================================
# RC: SAMPLE VARIANTS
# ============================================================

# rc/sample/subset/sample0_brc_only (Table 4 Cols 3-4)
print("Running rc/sample/subset/sample0_brc_only...")
controls_s0 = ['age', 'age_2'] + GROUPED_YEAR_DUMMIES
run_poisson_fe(
    spec_id="rc/sample/subset/sample0_brc_only",
    spec_tree_path="specification_tree/modules/robustness/sample.md#subset",
    baseline_group_id="G1",
    outcome_var="cites",
    treatment_var="post_brc",
    controls=controls_s0,
    fe_formula="rart_num",
    data=df_s0_nowin,
    vcov="hetero",
    sample_desc="sample0==1 & window==0 (BRC-only, Table 4 Col 3)",
    controls_desc="age + age_2 + 5 grouped year dummies",
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subset/sample0_brc_only",
                "family": "subset",
                "description": "BRC-only articles (no matched controls), window==0",
                "filter": "sample0==1 & window==0"},
)

# rc/sample/subset/sample0_brc_only_polynomial (Table 4 Col 4)
print("Running rc/sample/subset/sample0_brc_only_polynomial...")
controls_s0_poly = ['age', 'age_2', 'yr', 'yr_2']
run_poisson_fe(
    spec_id="rc/sample/subset/sample0_brc_only_polynomial",
    spec_tree_path="specification_tree/modules/robustness/sample.md#subset",
    baseline_group_id="G1",
    outcome_var="cites",
    treatment_var="post_brc",
    controls=controls_s0_poly,
    fe_formula="rart_num",
    data=df_s0_nowin,
    vcov="hetero",
    sample_desc="sample0==1 & window==0 (BRC-only, Table 4 Col 4)",
    controls_desc="age + age_2 + yr + yr_2 (polynomial)",
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subset/sample0_brc_only_polynomial",
                "family": "subset",
                "description": "BRC-only, polynomial age/year (Table 4 Col 4)",
                "filter": "sample0==1 & window==0"},
)

# rc/sample/outliers/trim_cites_99
print("Running rc/sample/outliers/trim_cites_99...")
run_poisson_fe(
    spec_id="rc/sample/outliers/trim_cites_99",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    baseline_group_id="G1",
    outcome_var="cites",
    treatment_var="post_brc",
    controls=BASELINE_CONTROLS,
    fe_formula="rart_num",
    data=df_s3_trim99,
    vcov="hetero",
    sample_desc="sample3==1, trim top 1% cites",
    controls_desc=BASELINE_CONTROLS_DESC,
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_cites_99",
                "family": "outliers",
                "description": "Trim observations with cites > 99th percentile",
                "threshold": float(cites_99)},
)

# rc/sample/outliers/winsor_cites_99
print("Running rc/sample/outliers/winsor_cites_99...")
run_poisson_fe(
    spec_id="rc/sample/outliers/winsor_cites_99",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    baseline_group_id="G1",
    outcome_var="cites_w99",
    treatment_var="post_brc",
    controls=BASELINE_CONTROLS,
    fe_formula="rart_num",
    data=df_s3,
    vcov="hetero",
    sample_desc="sample3==1, winsorize cites at 99th pct",
    controls_desc=BASELINE_CONTROLS_DESC,
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/winsor_cites_99",
                "family": "outliers",
                "description": "Winsorize cites at 99th percentile",
                "threshold": float(cites_99)},
)

# ============================================================
# RC: FUNCTIONAL FORM VARIANTS
# ============================================================

# rc/form/model/poisson_fe -- this IS the baseline estimator we use (note in payload)
# Instead, let's run the explicit Table 3 Col 4 Poisson: no age_brc1
print("Running rc/form/model/poisson_fe_no_age_brc1 (Table 3 Col 4 equivalent)...")
run_poisson_fe(
    spec_id="rc/form/model/poisson_fe_table3c4",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md#model",
    baseline_group_id="G1",
    outcome_var="cites",
    treatment_var="post_brc",
    controls=['window'] + AGE_DUMMIES + YEAR_DUMMIES,
    fe_formula="rart_num",
    data=df_s3,
    vcov="hetero",
    sample_desc="sample3==1, Poisson FE (Table 3 Col 4 equivalent)",
    controls_desc="window + 30 age dummies + 23 year dummies",
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    functional_form_block={
        "spec_id": "rc/form/model/poisson_fe_table3c4",
        "interpretation": "Poisson FE, Table 3 Col 4 controls (no age_brc1)",
        "model": "poisson_fe",
    },
)

# rc/form/model/ols_log_cites
print("Running rc/form/model/ols_log_cites...")
run_ols_fe(
    spec_id="rc/form/model/ols_log_cites",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md#model",
    baseline_group_id="G1",
    outcome_var="log_cites_plus1",
    treatment_var="post_brc",
    controls=BASELINE_CONTROLS,
    fe_formula="rart_num",
    data=df_s3,
    vcov="hetero",
    sample_desc="sample3==1, OLS on log(cites+1)",
    controls_desc=BASELINE_CONTROLS_DESC,
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    functional_form_block={
        "spec_id": "rc/form/model/ols_log_cites",
        "interpretation": "Semi-elasticity: % change in cites from BRC deposit",
        "outcome_transform": "log(cites + 1)",
        "model": "ols_fe",
    },
)

# rc/form/model/ols_asinh_cites
print("Running rc/form/model/ols_asinh_cites...")
run_ols_fe(
    spec_id="rc/form/model/ols_asinh_cites",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md#model",
    baseline_group_id="G1",
    outcome_var="asinh_cites",
    treatment_var="post_brc",
    controls=BASELINE_CONTROLS,
    fe_formula="rart_num",
    data=df_s3,
    vcov="hetero",
    sample_desc="sample3==1, OLS on asinh(cites)",
    controls_desc=BASELINE_CONTROLS_DESC,
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    functional_form_block={
        "spec_id": "rc/form/model/ols_asinh_cites",
        "interpretation": "Inverse hyperbolic sine transform of citations",
        "outcome_transform": "asinh(cites)",
        "model": "ols_fe",
    },
)

# rc/form/outcome/log_cites_plus1 (OLS on log(cites+1), no age_brc1 -- Table 3 Col 1-like)
print("Running rc/form/outcome/log_cites_plus1 (Table 3 OLS variant)...")
run_ols_fe(
    spec_id="rc/form/outcome/log_cites_plus1",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md#outcome",
    baseline_group_id="G1",
    outcome_var="log_cites_plus1",
    treatment_var="post_brc",
    controls=['window'] + AGE_DUMMIES + YEAR_DUMMIES,
    fe_formula="rart_num",
    data=df_s3,
    vcov="hetero",
    sample_desc="sample3==1, OLS on log(cites+1), Table 3 controls",
    controls_desc="window + 30 age dummies + 23 year dummies",
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    functional_form_block={
        "spec_id": "rc/form/outcome/log_cites_plus1",
        "interpretation": "OLS FE on log(cites+1), Table 3 Col 4 controls",
        "outcome_transform": "log(cites + 1)",
    },
)

# rc/form/outcome/asinh_cites (OLS on asinh(cites), baseline controls)
print("Running rc/form/outcome/asinh_cites...")
run_ols_fe(
    spec_id="rc/form/outcome/asinh_cites",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md#outcome",
    baseline_group_id="G1",
    outcome_var="asinh_cites",
    treatment_var="post_brc",
    controls=['window'] + AGE_DUMMIES + YEAR_DUMMIES,
    fe_formula="rart_num",
    data=df_s3,
    vcov="hetero",
    sample_desc="sample3==1, OLS on asinh(cites), Table 3 controls",
    controls_desc="window + 30 age dummies + 23 year dummies",
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    functional_form_block={
        "spec_id": "rc/form/outcome/asinh_cites",
        "interpretation": "OLS FE on asinh(cites)",
        "outcome_transform": "asinh(cites)",
    },
)

# rc/form/model/nbreg_grouped_age (Poisson with grouped age dummies)
print("Running rc/form/model/nbreg_grouped_age...")
run_poisson_fe(
    spec_id="rc/form/model/nbreg_grouped_age",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md#model",
    baseline_group_id="G1",
    outcome_var="cites",
    treatment_var="post_brc",
    controls=['window', 'age_brc1', 'age', 'age_2'] + GROUPED_YEAR_DUMMIES,
    fe_formula="rart_num",
    data=df_s3,
    vcov="hetero",
    sample_desc="sample3==1, Poisson FE with grouped age/year",
    controls_desc="window + age_brc1 + age + age_2 + 5 grouped year dummies",
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    functional_form_block={
        "spec_id": "rc/form/model/nbreg_grouped_age",
        "interpretation": "Poisson FE with grouped (5yr) age/year dummies instead of individual dummies",
        "model": "poisson_fe",
    },
)

# ============================================================
# RC: FIXED EFFECTS VARIANTS
# ============================================================

# rc/fe/add_pair_fe (add pair FE alongside article FE)
print("Running rc/fe/add_pair_fe...")
run_poisson_fe(
    spec_id="rc/fe/add_pair_fe",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md",
    baseline_group_id="G1",
    outcome_var="cites",
    treatment_var="post_brc",
    controls=BASELINE_CONTROLS,
    fe_formula="rart_num + pair_num",
    data=df_s3,
    vcov="hetero",
    sample_desc="sample3==1, Poisson FE with article + pair FE",
    controls_desc=BASELINE_CONTROLS_DESC,
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/add_pair_fe",
                "added": ["pair_num"],
                "description": "Add matched-pair FE alongside article FE"},
)

# rc/fe/pair_fe_only (pair FE instead of article FE -- like Table 3 Col 3)
print("Running rc/fe/pair_fe_only (Table 3 Col 3 style)...")
run_poisson_fe(
    spec_id="rc/fe/pair_fe_only",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md",
    baseline_group_id="G1",
    outcome_var="cites",
    treatment_var="post_brc",
    controls=['brc', 'window'] + AGE_DUMMIES + YEAR_DUMMIES,
    fe_formula="pair_num",
    data=df_s3,
    vcov="hetero",
    sample_desc="sample3==1, Poisson FE with pair FE (Table 3 Col 3 style)",
    controls_desc="brc + window + 30 age dummies + 23 year dummies",
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/pair_fe_only",
                "replaced": ["rart_num"],
                "with": ["pair_num"],
                "description": "Pair FE instead of article FE (Table 3 Col 3 style, includes brc level indicator)"},
)

# ============================================================
# RC: PREPROCESS/OUTCOME
# ============================================================

# rc/preprocess/outcome/winsor_cites_99 (winsorize then log)
print("Running rc/preprocess/outcome/winsor_cites_99 (OLS on log winsorized cites)...")
run_ols_fe(
    spec_id="rc/preprocess/outcome/winsor_cites_99",
    spec_tree_path="specification_tree/modules/robustness/preprocess.md",
    baseline_group_id="G1",
    outcome_var="log_cites_w99_plus1",
    treatment_var="post_brc",
    controls=BASELINE_CONTROLS,
    fe_formula="rart_num",
    data=df_s3,
    vcov="hetero",
    sample_desc="sample3==1, OLS on log(winsorized_cites+1)",
    controls_desc=BASELINE_CONTROLS_DESC,
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    functional_form_block={
        "spec_id": "rc/preprocess/outcome/winsor_cites_99",
        "interpretation": "OLS FE on log of winsorized cites (99th pct)",
        "outcome_transform": "log(winsor_cites_99 + 1)",
        "threshold": float(cites_99),
    },
)

# ============================================================
# RC: JOINT VARIATIONS (sample x form)
# ============================================================

# Joint: sample0 x OLS log
print("Running rc/joint/sample_form/sample0_ols_log...")
run_ols_fe(
    spec_id="rc/joint/sample_form/sample0_ols_log",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="log_cites_plus1",
    treatment_var="post_brc",
    controls=['age', 'age_2'] + GROUPED_YEAR_DUMMIES,
    fe_formula="rart_num",
    data=df_s0_nowin,
    vcov="hetero",
    sample_desc="sample0==1 & window==0, OLS on log(cites+1)",
    controls_desc="age + age_2 + 5 grouped year dummies",
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/sample_form/sample0_ols_log",
                "components": ["rc/sample/subset/sample0_brc_only", "rc/form/model/ols_log_cites"],
                "description": "BRC-only sample + OLS on log(cites+1)"},
    functional_form_block={
        "spec_id": "rc/joint/sample_form/sample0_ols_log",
        "interpretation": "BRC-only sample, OLS FE on log(cites+1)",
        "outcome_transform": "log(cites + 1)",
    },
)

# Joint: sample0 x OLS asinh
print("Running rc/joint/sample_form/sample0_ols_asinh...")
run_ols_fe(
    spec_id="rc/joint/sample_form/sample0_ols_asinh",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="asinh_cites",
    treatment_var="post_brc",
    controls=['age', 'age_2'] + GROUPED_YEAR_DUMMIES,
    fe_formula="rart_num",
    data=df_s0_nowin,
    vcov="hetero",
    sample_desc="sample0==1 & window==0, OLS on asinh(cites)",
    controls_desc="age + age_2 + 5 grouped year dummies",
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/sample_form/sample0_ols_asinh",
                "components": ["rc/sample/subset/sample0_brc_only", "rc/form/model/ols_asinh_cites"],
                "description": "BRC-only sample + OLS on asinh(cites)"},
    functional_form_block={
        "spec_id": "rc/joint/sample_form/sample0_ols_asinh",
        "interpretation": "BRC-only sample, OLS FE on asinh(cites)",
        "outcome_transform": "asinh(cites)",
    },
)

# Joint: sample0 x Poisson polynomial
print("Running rc/joint/sample_form/sample0_poisson_poly...")
run_poisson_fe(
    spec_id="rc/joint/sample_form/sample0_poisson_poly",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="cites",
    treatment_var="post_brc",
    controls=['age', 'age_2', 'yr', 'yr_2'],
    fe_formula="rart_num",
    data=df_s0_nowin,
    vcov="hetero",
    sample_desc="sample0==1 & window==0, Poisson FE, polynomial",
    controls_desc="age + age_2 + yr + yr_2",
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/sample_form/sample0_poisson_poly",
                "components": ["rc/sample/subset/sample0_brc_only", "rc/controls/sets/polynomial_age_year"],
                "description": "BRC-only, polynomial age/year (Table 4 Col 4 equivalent)"},
)

# Joint: trim99 x OLS log
print("Running rc/joint/sample_form/trim99_ols_log...")
run_ols_fe(
    spec_id="rc/joint/sample_form/trim99_ols_log",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="log_cites_plus1",
    treatment_var="post_brc",
    controls=BASELINE_CONTROLS,
    fe_formula="rart_num",
    data=df_s3_trim99,
    vcov="hetero",
    sample_desc="sample3==1, trim top 1% cites, OLS on log(cites+1)",
    controls_desc=BASELINE_CONTROLS_DESC,
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/sample_form/trim99_ols_log",
                "components": ["rc/sample/outliers/trim_cites_99", "rc/form/model/ols_log_cites"],
                "description": "Trimmed sample + OLS on log(cites+1)"},
    functional_form_block={
        "spec_id": "rc/joint/sample_form/trim99_ols_log",
        "interpretation": "Trimmed sample, OLS FE on log(cites+1)",
        "outcome_transform": "log(cites + 1)",
    },
)

# Joint: trim99 x OLS asinh
print("Running rc/joint/sample_form/trim99_ols_asinh...")
run_ols_fe(
    spec_id="rc/joint/sample_form/trim99_ols_asinh",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="asinh_cites",
    treatment_var="post_brc",
    controls=BASELINE_CONTROLS,
    fe_formula="rart_num",
    data=df_s3_trim99,
    vcov="hetero",
    sample_desc="sample3==1, trim top 1% cites, OLS on asinh(cites)",
    controls_desc=BASELINE_CONTROLS_DESC,
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/sample_form/trim99_ols_asinh",
                "components": ["rc/sample/outliers/trim_cites_99", "rc/form/model/ols_asinh_cites"],
                "description": "Trimmed sample + OLS on asinh(cites)"},
    functional_form_block={
        "spec_id": "rc/joint/sample_form/trim99_ols_asinh",
        "interpretation": "Trimmed sample, OLS FE on asinh(cites)",
        "outcome_transform": "asinh(cites)",
    },
)

# Joint: winsor99 x Poisson
print("Running rc/joint/sample_form/winsor99_poisson...")
run_poisson_fe(
    spec_id="rc/joint/sample_form/winsor99_poisson",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="cites_w99",
    treatment_var="post_brc",
    controls=BASELINE_CONTROLS,
    fe_formula="rart_num",
    data=df_s3,
    vcov="hetero",
    sample_desc="sample3==1, winsorize cites at 99th, Poisson FE",
    controls_desc=BASELINE_CONTROLS_DESC,
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/sample_form/winsor99_poisson",
                "components": ["rc/sample/outliers/winsor_cites_99"],
                "description": "Winsorized cites + Poisson FE"},
)

# Joint: pair_fe x OLS log
print("Running rc/joint/fe_form/pair_fe_ols_log...")
run_ols_fe(
    spec_id="rc/joint/fe_form/pair_fe_ols_log",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="log_cites_plus1",
    treatment_var="post_brc",
    controls=['brc', 'window'] + AGE_DUMMIES + YEAR_DUMMIES,
    fe_formula="pair_num",
    data=df_s3,
    vcov="hetero",
    sample_desc="sample3==1, OLS log(cites+1) with pair FE",
    controls_desc="brc + window + 30 age dummies + 23 year dummies",
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/fe_form/pair_fe_ols_log",
                "components": ["rc/fe/pair_fe_only", "rc/form/model/ols_log_cites"],
                "description": "Pair FE + OLS on log(cites+1)"},
    functional_form_block={
        "spec_id": "rc/joint/fe_form/pair_fe_ols_log",
        "interpretation": "OLS log(cites+1) with pair FE (Table 3 OLS variant)",
        "outcome_transform": "log(cites + 1)",
    },
)

# Joint: pair_fe x OLS asinh
print("Running rc/joint/fe_form/pair_fe_ols_asinh...")
run_ols_fe(
    spec_id="rc/joint/fe_form/pair_fe_ols_asinh",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="asinh_cites",
    treatment_var="post_brc",
    controls=['brc', 'window'] + AGE_DUMMIES + YEAR_DUMMIES,
    fe_formula="pair_num",
    data=df_s3,
    vcov="hetero",
    sample_desc="sample3==1, OLS asinh(cites) with pair FE",
    controls_desc="brc + window + 30 age dummies + 23 year dummies",
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/fe_form/pair_fe_ols_asinh",
                "components": ["rc/fe/pair_fe_only", "rc/form/model/ols_asinh_cites"],
                "description": "Pair FE + OLS on asinh(cites)"},
    functional_form_block={
        "spec_id": "rc/joint/fe_form/pair_fe_ols_asinh",
        "interpretation": "OLS asinh(cites) with pair FE",
        "outcome_transform": "asinh(cites)",
    },
)

# ============================================================
# Additional functional form + controls combos
# ============================================================

# OLS log + post_brc_yrs
print("Running rc/joint/controls_form/ols_log_post_brc_yrs...")
run_ols_fe(
    spec_id="rc/joint/controls_form/ols_log_post_brc_yrs",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="log_cites_plus1",
    treatment_var="post_brc",
    controls=BASELINE_CONTROLS + ['post_brc_yrs'],
    fe_formula="rart_num",
    data=df_s3,
    vcov="hetero",
    sample_desc="sample3==1, OLS log(cites+1) + post_brc_yrs",
    controls_desc=BASELINE_CONTROLS_DESC + " + post_brc_yrs",
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/controls_form/ols_log_post_brc_yrs",
                "components": ["rc/controls/single/add_post_brc_yrs", "rc/form/model/ols_log_cites"],
                "description": "OLS log(cites+1) + post_brc_yrs control"},
    functional_form_block={
        "spec_id": "rc/joint/controls_form/ols_log_post_brc_yrs",
        "interpretation": "OLS log(cites+1) with time-since-deposit trend",
        "outcome_transform": "log(cites + 1)",
    },
)

# OLS asinh + post_brc_yrs
print("Running rc/joint/controls_form/ols_asinh_post_brc_yrs...")
run_ols_fe(
    spec_id="rc/joint/controls_form/ols_asinh_post_brc_yrs",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="asinh_cites",
    treatment_var="post_brc",
    controls=BASELINE_CONTROLS + ['post_brc_yrs'],
    fe_formula="rart_num",
    data=df_s3,
    vcov="hetero",
    sample_desc="sample3==1, OLS asinh(cites) + post_brc_yrs",
    controls_desc=BASELINE_CONTROLS_DESC + " + post_brc_yrs",
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/controls_form/ols_asinh_post_brc_yrs",
                "components": ["rc/controls/single/add_post_brc_yrs", "rc/form/model/ols_asinh_cites"],
                "description": "OLS asinh(cites) + post_brc_yrs control"},
    functional_form_block={
        "spec_id": "rc/joint/controls_form/ols_asinh_post_brc_yrs",
        "interpretation": "OLS asinh(cites) with time-since-deposit trend",
        "outcome_transform": "asinh(cites)",
    },
)

# Poisson + post_brc_yrs + grouped age/year
print("Running rc/joint/controls_form/poisson_post_brc_yrs_grouped...")
run_poisson_fe(
    spec_id="rc/joint/controls_form/poisson_post_brc_yrs_grouped",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="cites",
    treatment_var="post_brc",
    controls=['window', 'age_brc1', 'post_brc_yrs', 'age', 'age_2'] + GROUPED_YEAR_DUMMIES,
    fe_formula="rart_num",
    data=df_s3,
    vcov="hetero",
    sample_desc="sample3==1, Poisson FE, post_brc_yrs + grouped age/year",
    controls_desc="window + age_brc1 + post_brc_yrs + age + age_2 + grouped year dummies",
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/controls_form/poisson_post_brc_yrs_grouped",
                "components": ["rc/controls/single/add_post_brc_yrs", "rc/controls/sets/grouped_age_year"],
                "description": "Poisson FE + post_brc_yrs + grouped age/year dummies"},
)

# ============================================================
# Additional sample x controls combos
# ============================================================

# sample0 with full individual age/year dummies (if sufficient variation)
print("Running rc/joint/sample_controls/sample0_individual_dummies...")
# sample0 has fewer obs, but try with full dummies
s0_controls = ['age_brc1'] + AGE_DUMMIES + YEAR_DUMMIES
run_poisson_fe(
    spec_id="rc/joint/sample_controls/sample0_individual_dummies",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="cites",
    treatment_var="post_brc",
    controls=s0_controls,
    fe_formula="rart_num",
    data=df_s0_nowin,
    vcov="hetero",
    sample_desc="sample0==1 & window==0, Poisson FE, individual age/year dummies",
    controls_desc="age_brc1 + 30 age dummies + 23 year dummies",
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/sample_controls/sample0_individual_dummies",
                "components": ["rc/sample/subset/sample0_brc_only"],
                "description": "BRC-only sample with individual age/year dummies"},
)

# OLS on raw cites (linear FE, not logged)
print("Running rc/form/model/ols_raw_cites...")
run_ols_fe(
    spec_id="rc/form/model/ols_raw_cites",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md#model",
    baseline_group_id="G1",
    outcome_var="cites",
    treatment_var="post_brc",
    controls=BASELINE_CONTROLS,
    fe_formula="rart_num",
    data=df_s3,
    vcov="hetero",
    sample_desc="sample3==1, OLS on raw cites (linear FE)",
    controls_desc=BASELINE_CONTROLS_DESC,
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    functional_form_block={
        "spec_id": "rc/form/model/ols_raw_cites",
        "interpretation": "Linear FE on raw citation counts (level, not logged)",
        "outcome_transform": "none (raw cites)",
        "model": "ols_fe",
    },
)

# OLS on raw cites, no age_brc1 (Table 3 Col 4 OLS equivalent)
print("Running rc/form/model/ols_raw_cites_table3c4...")
run_ols_fe(
    spec_id="rc/form/model/ols_raw_cites_table3c4",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md#model",
    baseline_group_id="G1",
    outcome_var="cites",
    treatment_var="post_brc",
    controls=['window'] + AGE_DUMMIES + YEAR_DUMMIES,
    fe_formula="rart_num",
    data=df_s3,
    vcov="hetero",
    sample_desc="sample3==1, OLS on raw cites, Table 3 controls",
    controls_desc="window + 30 age dummies + 23 year dummies",
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    functional_form_block={
        "spec_id": "rc/form/model/ols_raw_cites_table3c4",
        "interpretation": "Linear FE on raw cites, Table 3 Col 4 controls",
        "outcome_transform": "none (raw cites)",
        "model": "ols_fe",
    },
)

# ============================================================
# Table 3 replication variants (Col 1 and Col 2 from the OLS section)
# ============================================================

# Table 3 Col 1: OLS on log(cites+1) with age FE only, pair cluster
print("Running rc/joint/fe_form/ols_log_age_fe_only...")
run_ols_fe(
    spec_id="rc/joint/fe_form/ols_log_age_fe_only",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="log_cites_plus1",
    treatment_var="post_brc",
    controls=['brc', 'window'] + AGE_DUMMIES,
    fe_formula="",
    data=df_s3,
    vcov={"CRV1": "pair_num"},
    sample_desc="sample3==1, OLS log(cites+1) with age dummies only (Table 3 Col 1 style)",
    controls_desc="brc + window + 30 age dummies (no year dummies, no article FE)",
    cluster_var="pair_num",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/fe_form/ols_log_age_fe_only",
                "components": [],
                "description": "Table 3 Col 1 style: OLS log(cites+1), age dummies only, cluster(pair_num)"},
    functional_form_block={
        "spec_id": "rc/joint/fe_form/ols_log_age_fe_only",
        "interpretation": "OLS on log(cites+1), cross-sectional with age dummies only",
        "outcome_transform": "log(cites + 1)",
    },
)

# Table 3 Col 2: OLS on log(cites+1) with pair FE + age + year dummies, pair cluster
print("Running rc/joint/fe_form/ols_log_pair_age_year...")
run_ols_fe(
    spec_id="rc/joint/fe_form/ols_log_pair_age_year",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="log_cites_plus1",
    treatment_var="post_brc",
    controls=['brc', 'window'] + AGE_DUMMIES + YEAR_DUMMIES,
    fe_formula="pair_num",
    data=df_s3,
    vcov={"CRV1": "pair_num"},
    sample_desc="sample3==1, OLS log(cites+1) with pair FE (Table 3 Col 2 style)",
    controls_desc="brc + window + 30 age dummies + 23 year dummies",
    cluster_var="pair_num",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/fe_form/ols_log_pair_age_year",
                "components": ["rc/fe/pair_fe_only"],
                "description": "Table 3 Col 2 style: OLS log(cites+1), pair + age + year FE, cluster(pair_num)"},
    functional_form_block={
        "spec_id": "rc/joint/fe_form/ols_log_pair_age_year",
        "interpretation": "OLS on log(cites+1) with pair FE (Table 3 Col 2 style)",
        "outcome_transform": "log(cites + 1)",
    },
)

# ============================================================
# Additional: Poisson FE with cluster at article level
# ============================================================
print("Running rc/form/model/poisson_fe_cluster_article...")
run_poisson_fe(
    spec_id="rc/form/model/poisson_fe_cluster_article",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md#model",
    baseline_group_id="G1",
    outcome_var="cites",
    treatment_var="post_brc",
    controls=BASELINE_CONTROLS,
    fe_formula="rart_num",
    data=df_s3,
    vcov={"CRV1": "rart_num"},
    sample_desc="sample3==1, Poisson FE with cluster(rart_num) SE",
    controls_desc=BASELINE_CONTROLS_DESC,
    cluster_var="rart_num",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    functional_form_block={
        "spec_id": "rc/form/model/poisson_fe_cluster_article",
        "interpretation": "Poisson FE with cluster-robust SE at article level",
        "model": "poisson_fe",
    },
    notes="Poisson FE baseline with cluster-robust SE at article level instead of HC1."
)

# OLS log cluster(rart_num)
print("Running rc/form/model/ols_log_cluster_article...")
run_ols_fe(
    spec_id="rc/form/model/ols_log_cluster_article",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md#model",
    baseline_group_id="G1",
    outcome_var="log_cites_plus1",
    treatment_var="post_brc",
    controls=BASELINE_CONTROLS,
    fe_formula="rart_num",
    data=df_s3,
    vcov={"CRV1": "rart_num"},
    sample_desc="sample3==1, OLS log(cites+1) with cluster(rart_num) SE",
    controls_desc=BASELINE_CONTROLS_DESC,
    cluster_var="rart_num",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    functional_form_block={
        "spec_id": "rc/form/model/ols_log_cluster_article",
        "interpretation": "OLS FE on log(cites+1), cluster-robust SE at article level",
        "outcome_transform": "log(cites + 1)",
        "model": "ols_fe",
    },
)

# ============================================================
# Additional: Drop window and age_brc1 for OLS variants
# ============================================================

# OLS log, no window/age_brc1
print("Running rc/joint/controls_form/ols_log_minimal...")
run_ols_fe(
    spec_id="rc/joint/controls_form/ols_log_minimal",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="log_cites_plus1",
    treatment_var="post_brc",
    controls=AGE_DUMMIES + YEAR_DUMMIES,
    fe_formula="rart_num",
    data=df_s3,
    vcov="hetero",
    sample_desc="sample3==1, OLS log(cites+1), minimal controls",
    controls_desc="30 age dummies + 23 year dummies (no window, no age_brc1)",
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/controls_form/ols_log_minimal",
                "components": ["rc/controls/loo/drop_window_and_age_brc1", "rc/form/model/ols_log_cites"],
                "description": "OLS log(cites+1), dropped window and age_brc1"},
    functional_form_block={
        "spec_id": "rc/joint/controls_form/ols_log_minimal",
        "interpretation": "OLS FE on log(cites+1), minimal controls",
        "outcome_transform": "log(cites + 1)",
    },
)

# OLS asinh, no window/age_brc1
print("Running rc/joint/controls_form/ols_asinh_minimal...")
run_ols_fe(
    spec_id="rc/joint/controls_form/ols_asinh_minimal",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="asinh_cites",
    treatment_var="post_brc",
    controls=AGE_DUMMIES + YEAR_DUMMIES,
    fe_formula="rart_num",
    data=df_s3,
    vcov="hetero",
    sample_desc="sample3==1, OLS asinh(cites), minimal controls",
    controls_desc="30 age dummies + 23 year dummies (no window, no age_brc1)",
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/controls_form/ols_asinh_minimal",
                "components": ["rc/controls/loo/drop_window_and_age_brc1", "rc/form/model/ols_asinh_cites"],
                "description": "OLS asinh(cites), dropped window and age_brc1"},
    functional_form_block={
        "spec_id": "rc/joint/controls_form/ols_asinh_minimal",
        "interpretation": "OLS FE on asinh(cites), minimal controls",
        "outcome_transform": "asinh(cites)",
    },
)

# ============================================================
# Additional: Trim/Winsorize x Poisson with different controls
# ============================================================

# Trim99 x Poisson, grouped age/year
print("Running rc/joint/sample_form/trim99_poisson_grouped...")
df_s3_trim99_copy = df_s3_trim99.copy()
run_poisson_fe(
    spec_id="rc/joint/sample_form/trim99_poisson_grouped",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="cites",
    treatment_var="post_brc",
    controls=['window', 'age_brc1', 'age', 'age_2'] + GROUPED_YEAR_DUMMIES,
    fe_formula="rart_num",
    data=df_s3_trim99_copy,
    vcov="hetero",
    sample_desc="sample3==1, trim top 1%, Poisson FE, grouped age/year",
    controls_desc="window + age_brc1 + age + age_2 + grouped year dummies",
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/sample_form/trim99_poisson_grouped",
                "components": ["rc/sample/outliers/trim_cites_99", "rc/controls/sets/grouped_age_year"],
                "description": "Trimmed sample + Poisson FE + grouped age/year"},
)

# Winsor99 x OLS log
print("Running rc/joint/sample_form/winsor99_ols_log...")
run_ols_fe(
    spec_id="rc/joint/sample_form/winsor99_ols_log",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="log_cites_w99_plus1",
    treatment_var="post_brc",
    controls=BASELINE_CONTROLS,
    fe_formula="rart_num",
    data=df_s3,
    vcov="hetero",
    sample_desc="sample3==1, winsorize cites at 99th, OLS log(cites+1)",
    controls_desc=BASELINE_CONTROLS_DESC,
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/sample_form/winsor99_ols_log",
                "components": ["rc/sample/outliers/winsor_cites_99", "rc/form/model/ols_log_cites"],
                "description": "Winsorized cites + OLS on log(cites+1)"},
    functional_form_block={
        "spec_id": "rc/joint/sample_form/winsor99_ols_log",
        "interpretation": "OLS FE on log(winsorized cites + 1)",
        "outcome_transform": "log(winsor_cites_99 + 1)",
    },
)

# ============================================================
# Additional: Poisson FE with post_brc_yrs, different samples
# ============================================================

# Poisson + post_brc_yrs, trim99
print("Running rc/joint/sample_controls/trim99_post_brc_yrs...")
run_poisson_fe(
    spec_id="rc/joint/sample_controls/trim99_post_brc_yrs",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="cites",
    treatment_var="post_brc",
    controls=BASELINE_CONTROLS + ['post_brc_yrs'],
    fe_formula="rart_num",
    data=df_s3_trim99,
    vcov="hetero",
    sample_desc="sample3==1, trim top 1%, Poisson FE + post_brc_yrs",
    controls_desc=BASELINE_CONTROLS_DESC + " + post_brc_yrs",
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/sample_controls/trim99_post_brc_yrs",
                "components": ["rc/sample/outliers/trim_cites_99", "rc/controls/single/add_post_brc_yrs"],
                "description": "Trimmed sample + post_brc_yrs control"},
)

# OLS log + post_brc_yrs, trim99
print("Running rc/joint/sample_controls_form/trim99_ols_log_post_brc_yrs...")
run_ols_fe(
    spec_id="rc/joint/sample_controls_form/trim99_ols_log_post_brc_yrs",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="log_cites_plus1",
    treatment_var="post_brc",
    controls=BASELINE_CONTROLS + ['post_brc_yrs'],
    fe_formula="rart_num",
    data=df_s3_trim99,
    vcov="hetero",
    sample_desc="sample3==1, trim top 1%, OLS log(cites+1) + post_brc_yrs",
    controls_desc=BASELINE_CONTROLS_DESC + " + post_brc_yrs",
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/sample_controls_form/trim99_ols_log_post_brc_yrs",
                "components": ["rc/sample/outliers/trim_cites_99", "rc/controls/single/add_post_brc_yrs", "rc/form/model/ols_log_cites"],
                "description": "Trimmed + post_brc_yrs + OLS log(cites+1)"},
    functional_form_block={
        "spec_id": "rc/joint/sample_controls_form/trim99_ols_log_post_brc_yrs",
        "interpretation": "Trimmed sample, OLS log(cites+1) with time-since-deposit",
        "outcome_transform": "log(cites + 1)",
    },
)

# ============================================================
# Additional: Poisson with pair FE + article FE, different controls
# ============================================================
print("Running rc/joint/fe_controls/pair_article_fe_post_brc_yrs...")
run_poisson_fe(
    spec_id="rc/joint/fe_controls/pair_article_fe_post_brc_yrs",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="cites",
    treatment_var="post_brc",
    controls=BASELINE_CONTROLS + ['post_brc_yrs'],
    fe_formula="rart_num + pair_num",
    data=df_s3,
    vcov="hetero",
    sample_desc="sample3==1, Poisson FE with article + pair FE + post_brc_yrs",
    controls_desc=BASELINE_CONTROLS_DESC + " + post_brc_yrs",
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/fe_controls/pair_article_fe_post_brc_yrs",
                "components": ["rc/fe/add_pair_fe", "rc/controls/single/add_post_brc_yrs"],
                "description": "Pair + article FE + post_brc_yrs"},
)

# ============================================================
# Additional: Cluster at pair_num level
# ============================================================
print("Running rc/joint/fe_form/poisson_cluster_pair...")
run_poisson_fe(
    spec_id="rc/joint/fe_form/poisson_cluster_pair",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="cites",
    treatment_var="post_brc",
    controls=BASELINE_CONTROLS,
    fe_formula="rart_num",
    data=df_s3,
    vcov={"CRV1": "pair_num"},
    sample_desc="sample3==1, Poisson FE, cluster(pair_num) SE",
    controls_desc=BASELINE_CONTROLS_DESC,
    cluster_var="pair_num",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/fe_form/poisson_cluster_pair",
                "description": "Poisson FE baseline with cluster SE at pair level"},
)

# OLS log cluster(pair_num)
print("Running rc/joint/fe_form/ols_log_cluster_pair...")
run_ols_fe(
    spec_id="rc/joint/fe_form/ols_log_cluster_pair",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="log_cites_plus1",
    treatment_var="post_brc",
    controls=BASELINE_CONTROLS,
    fe_formula="rart_num",
    data=df_s3,
    vcov={"CRV1": "pair_num"},
    sample_desc="sample3==1, OLS log(cites+1), cluster(pair_num) SE",
    controls_desc=BASELINE_CONTROLS_DESC,
    cluster_var="pair_num",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/fe_form/ols_log_cluster_pair",
                "description": "OLS log(cites+1) with cluster SE at pair level"},
    functional_form_block={
        "spec_id": "rc/joint/fe_form/ols_log_cluster_pair",
        "interpretation": "OLS log(cites+1), cluster-robust SE at pair level",
        "outcome_transform": "log(cites + 1)",
    },
)

# ============================================================
# Additional: OLS on raw cites with different controls/samples
# ============================================================

# OLS raw cites, trim99
print("Running rc/joint/sample_form/trim99_ols_raw...")
run_ols_fe(
    spec_id="rc/joint/sample_form/trim99_ols_raw",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="cites",
    treatment_var="post_brc",
    controls=BASELINE_CONTROLS,
    fe_formula="rart_num",
    data=df_s3_trim99,
    vcov="hetero",
    sample_desc="sample3==1, trim top 1%, OLS on raw cites",
    controls_desc=BASELINE_CONTROLS_DESC,
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/sample_form/trim99_ols_raw",
                "components": ["rc/sample/outliers/trim_cites_99", "rc/form/model/ols_raw_cites"],
                "description": "Trimmed sample + OLS on raw cites"},
    functional_form_block={
        "spec_id": "rc/joint/sample_form/trim99_ols_raw",
        "interpretation": "OLS FE on raw cites, trimmed sample",
        "outcome_transform": "none (raw cites)",
    },
)

# ============================================================
# Additional: sample0 variations with OLS raw cites
# ============================================================
print("Running rc/joint/sample_form/sample0_ols_raw...")
run_ols_fe(
    spec_id="rc/joint/sample_form/sample0_ols_raw",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="cites",
    treatment_var="post_brc",
    controls=['age', 'age_2'] + GROUPED_YEAR_DUMMIES,
    fe_formula="rart_num",
    data=df_s0_nowin,
    vcov="hetero",
    sample_desc="sample0==1 & window==0, OLS on raw cites",
    controls_desc="age + age_2 + 5 grouped year dummies",
    cluster_var="",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/sample_form/sample0_ols_raw",
                "components": ["rc/sample/subset/sample0_brc_only", "rc/form/model/ols_raw_cites"],
                "description": "BRC-only sample + OLS on raw cites"},
    functional_form_block={
        "spec_id": "rc/joint/sample_form/sample0_ols_raw",
        "interpretation": "BRC-only, OLS FE on raw cites",
        "outcome_transform": "none (raw cites)",
    },
)

# ============================================================
# INFERENCE VARIANTS
# ============================================================
print("Running inference variants...")

# For a selection of key specs, re-estimate with cluster SE (article and pair level)
# We'll refit the baseline model under different inference choices

# Re-estimate baseline Poisson FE with cluster(rart_num)
try:
    controls_str = " + ".join(BASELINE_CONTROLS)
    formula = f"cites ~ post_brc + {controls_str} | rart_num"
    m_cluster_art = pf.fepois(formula, data=df_s3, vcov={"CRV1": "rart_num"})
    add_inference_row(
        base_run_id=f"{PAPER_ID}_run_001",
        spec_id="infer/se/cluster/article",
        spec_tree_path="specification_tree/modules/inference/cluster.md",
        baseline_group_id="G1",
        outcome_var="cites", treatment_var="post_brc",
        model=m_cluster_art,
        infer_spec={"spec_id": "infer/se/cluster/article", "params": {"cluster_var": "rart_num"}},
        nobs=int(m_cluster_art._N),
        r2=np.nan,
        sample_desc="sample3==1, Poisson FE",
        fe_str="rart_num",
        controls_desc=BASELINE_CONTROLS_DESC,
        cluster_var="rart_num"
    )
except Exception as e:
    print(f"  Inference cluster/article failed: {e}")

# Re-estimate baseline OLS log with cluster(rart_num)
try:
    formula_ols = f"log_cites_plus1 ~ post_brc + {controls_str} | rart_num"
    m_ols_cluster = pf.feols(formula_ols, data=df_s3, vcov={"CRV1": "rart_num"})
    # Find the run_id for the OLS log baseline
    ols_log_run_id = None
    for r in results:
        if r["spec_id"] == "rc/form/model/ols_log_cites":
            ols_log_run_id = r["spec_run_id"]
            break
    if ols_log_run_id:
        add_inference_row(
            base_run_id=ols_log_run_id,
            spec_id="infer/se/cluster/article",
            spec_tree_path="specification_tree/modules/inference/cluster.md",
            baseline_group_id="G1",
            outcome_var="log_cites_plus1", treatment_var="post_brc",
            model=m_ols_cluster,
            infer_spec={"spec_id": "infer/se/cluster/article", "params": {"cluster_var": "rart_num"}},
            nobs=int(m_ols_cluster._N),
            r2=float(m_ols_cluster._r2),
            sample_desc="sample3==1, OLS log(cites+1)",
            fe_str="rart_num",
            controls_desc=BASELINE_CONTROLS_DESC,
            cluster_var="rart_num"
        )
except Exception as e:
    print(f"  Inference OLS log cluster/article failed: {e}")

# HC1 for OLS log baseline
try:
    m_ols_hc1 = pf.feols(formula_ols, data=df_s3, vcov="hetero")
    if ols_log_run_id:
        add_inference_row(
            base_run_id=ols_log_run_id,
            spec_id="infer/se/hc/hc1",
            spec_tree_path="specification_tree/modules/inference/heteroskedasticity.md",
            baseline_group_id="G1",
            outcome_var="log_cites_plus1", treatment_var="post_brc",
            model=m_ols_hc1,
            infer_spec={"spec_id": "infer/se/hc/hc1", "params": {}},
            nobs=int(m_ols_hc1._N),
            r2=float(m_ols_hc1._r2),
            sample_desc="sample3==1, OLS log(cites+1)",
            fe_str="rart_num",
            controls_desc=BASELINE_CONTROLS_DESC,
            cluster_var=""
        )
except Exception as e:
    print(f"  Inference OLS log HC1 failed: {e}")

# Cluster at pair level for baseline Poisson
try:
    m_cluster_pair = pf.fepois(formula, data=df_s3, vcov={"CRV1": "pair_num"})
    add_inference_row(
        base_run_id=f"{PAPER_ID}_run_001",
        spec_id="infer/se/cluster/pair",
        spec_tree_path="specification_tree/modules/inference/cluster.md",
        baseline_group_id="G1",
        outcome_var="cites", treatment_var="post_brc",
        model=m_cluster_pair,
        infer_spec={"spec_id": "infer/se/cluster/pair", "params": {"cluster_var": "pair_num"}},
        nobs=int(m_cluster_pair._N),
        r2=np.nan,
        sample_desc="sample3==1, Poisson FE",
        fe_str="rart_num",
        controls_desc=BASELINE_CONTROLS_DESC,
        cluster_var="pair_num"
    )
except Exception as e:
    print(f"  Inference cluster/pair failed: {e}")


# ============================================================
# SAVE OUTPUTS
# ============================================================
print(f"\nTotal specification results: {len(results)}")
print(f"Total inference results: {len(inference_results)}")

# Save specification_results.csv
df_results = pd.DataFrame(results)
df_results.to_csv(f"{OUTPUT_DIR}/specification_results.csv", index=False)
print(f"Saved specification_results.csv with {len(df_results)} rows")

# Save inference_results.csv
df_infer = pd.DataFrame(inference_results)
df_infer.to_csv(f"{OUTPUT_DIR}/inference_results.csv", index=False)
print(f"Saved inference_results.csv with {len(df_infer)} rows")

# Summary stats
n_success = df_results['run_success'].sum()
n_fail = len(df_results) - n_success
print(f"Successes: {n_success}, Failures: {n_fail}")

# Print summary of coefficients
print("\n=== SPECIFICATION RESULTS SUMMARY ===")
for _, row in df_results.iterrows():
    status = "OK" if row['run_success'] == 1 else "FAIL"
    print(f"  [{status}] {row['spec_id']}: coef={row['coefficient']:.4f}, se={row['std_error']:.4f}, p={row['p_value']:.4f}, N={row['n_obs']}" if row['run_success'] == 1 else f"  [{status}] {row['spec_id']}: {row['run_error'][:60]}")


# ============================================================
# WRITE SPECIFICATION_SEARCH.md
# ============================================================
md_lines = [
    f"# Specification Search: {PAPER_ID}",
    "",
    "## Paper",
    "Furman & Stern (2011), 'Biological Resource Centers and the Advancement of Technological Knowledge'",
    "RAND Journal of Economics, 42(1), 1-32.",
    "",
    "## Surface Summary",
    "- **Baseline groups**: 1 (G1: Effect of BRC deposit on citations)",
    "- **Design**: Difference-in-differences with article fixed effects",
    "- **Estimator**: Poisson FE (proxy for paper's conditional FE negative binomial, xtnbreg)",
    f"- **Budget**: 60 max specs",
    f"- **Seed**: 112451",
    "",
    "## Execution Summary",
    f"- **Total specs planned**: {len(results)}",
    f"- **Specs executed successfully**: {int(n_success)}",
    f"- **Specs failed**: {int(n_fail)}",
    f"- **Inference variants**: {len(inference_results)}",
    "",
    "## Deviations from Surface",
    "- The paper uses `xtnbreg` (conditional fixed-effects negative binomial), which is not",
    "  available in Python. Poisson FE (`pyfixest.fepois`) is used as the primary estimator.",
    "  This is a standard robustness check for count data and gives consistent estimates",
    "  under correct conditional mean specification.",
    "- Bootstrap SE (the paper's canonical inference) cannot be directly replicated in the",
    "  Poisson FE framework. HC1 (heteroskedasticity-robust) SE is used as the canonical",
    "  inference choice, with cluster-robust SE at article and pair levels as variants.",
    "",
    "## RC Axes Executed",
    "",
    "### Controls",
    "- Add post_brc_yrs (time-since-deposit trend)",
    "- LOO: drop window, drop age_brc1, drop both",
    "- Grouped age/year dummies (5-year groups)",
    "- Polynomial age/year (quadratic)",
    "",
    "### Sample",
    "- BRC-only sample (sample0==1 & window==0) with grouped and polynomial controls",
    "- Trim top 1% of citation counts",
    "- Winsorize at 99th percentile",
    "",
    "### Functional Form",
    "- Poisson FE (baseline proxy)",
    "- OLS on log(cites+1)",
    "- OLS on asinh(cites)",
    "- OLS on raw cites (linear FE)",
    "- Winsorized outcome transforms",
    "",
    "### Fixed Effects",
    "- Article FE (baseline)",
    "- Pair FE (instead of or alongside article FE)",
    "- Article + pair FE",
    "",
    "### Joint Variations",
    "- Sample x functional form (BRC-only x OLS/Poisson, trim x OLS/Poisson)",
    "- FE x functional form (pair FE x OLS log/asinh)",
    "- Controls x functional form (post_brc_yrs x OLS variants)",
    "- Cluster SE at article and pair levels across specifications",
    "",
    "## Inference Plan",
    "- **Canonical**: Bootstrap SE (approximated by HC1 robust SE in Python)",
    "- **Variant 1**: Cluster-robust SE at article level (rart_num)",
    "- **Variant 2**: Cluster-robust SE at pair level (pair_num)",
    "- **Variant 3**: HC1 for OLS variants",
    "",
    "## Software Stack",
    f"- Python {sys.version.split()[0]}",
]
for pkg, ver in SW_BLOCK["packages"].items():
    md_lines.append(f"- {pkg} {ver}")

md_lines.extend([
    "",
    "## Failed Specifications",
])
if n_fail == 0:
    md_lines.append("None.")
else:
    for _, row in df_results[df_results['run_success'] == 0].iterrows():
        md_lines.append(f"- `{row['spec_id']}`: {row['run_error']}")

with open(f"{OUTPUT_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write("\n".join(md_lines) + "\n")

print(f"\nSaved SPECIFICATION_SEARCH.md")
print("Done.")
