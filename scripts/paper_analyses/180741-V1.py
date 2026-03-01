#!/usr/bin/env python3
"""
Specification search script for 180741-V1:
"Enabling or Limiting Cognitive Flexibility? Evidence of Demand for Moral Commitment"
Saccardo & Serra-Garcia, AER

Surface-driven execution of core specifications for 3 baseline groups:
  G1: Choice experiment recommendations (Table 3)
  G2: NoChoice experiment recommendations (Table C.1)
  G3: Choice experiment preferences (Table 2)

Design: Randomized experiment (online experiments)
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import statsmodels.formula.api as smf
import json
import hashlib
import sys
import traceback
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PATHS
# ============================================================
REPO_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search"
PACKAGE_DIR = f"{REPO_DIR}/data/downloads/extracted/180741-V1"
PAPER_ID = "180741-V1"

# ============================================================
# LOAD SURFACE
# ============================================================
with open(f"{PACKAGE_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface = json.load(f)

surface_hash_val = "sha256:" + hashlib.sha256(
    json.dumps(surface, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
).hexdigest()

# ============================================================
# SOFTWARE BLOCK
# ============================================================
software_block = {
    "runner_language": "python",
    "runner_version": sys.version.split()[0],
    "packages": {
        "pyfixest": pf.__version__ if hasattr(pf, '__version__') else "0.40+",
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
}
try:
    import statsmodels
    software_block["packages"]["statsmodels"] = statsmodels.__version__
except:
    pass

# ============================================================
# DESIGN AUDIT BLOCKS (from surface per group)
# ============================================================
bg_map = {bg["baseline_group_id"]: bg for bg in surface["baseline_groups"]}

def design_block_for(group_id):
    da = bg_map[group_id]["design_audit"]
    return {"randomized_experiment": da}

def canonical_inference_for(group_id):
    return bg_map[group_id]["inference_plan"]["canonical"]

# ============================================================
# LOAD AND PREPARE DATA
# ============================================================
# Choice experiments data (for G1 and G3)
choice_raw = pd.read_stata(f"{PACKAGE_DIR}/Data/Clean Data/choice_experiments.dta")
# Apply Stata's drop rule: drop if study!=1 & alphavaluefinal==.
choice = choice_raw[~((choice_raw['study'] != 1) & (choice_raw['alphavaluefinal'].isna()))].copy()

# NoChoice data (for G2)
nc_raw = pd.read_stata(f"{PACKAGE_DIR}/Data/Clean Data/nochoice.dta")
# Drop if alphavaluefinal==.
nochoice = nc_raw[nc_raw['alphavaluefinal'].notna()].copy()

# ============================================================
# RESULT ACCUMULATORS
# ============================================================
results = []
inference_results = []

# ============================================================
# HELPER: RUN A SINGLE OLS SPEC
# ============================================================
def run_ols_spec(spec_id, spec_run_id, spec_tree_path, outcome_var, treatment_var,
                 data, rhs_vars, vcov, sample_desc, controls_desc,
                 baseline_group_id, extra_payload=None):
    """Run a single OLS specification and append to results."""
    row = {
        "paper_id": PAPER_ID,
        "spec_run_id": spec_run_id,
        "spec_id": spec_id,
        "spec_tree_path": spec_tree_path,
        "baseline_group_id": baseline_group_id,
        "outcome_var": outcome_var,
        "treatment_var": treatment_var,
        "sample_desc": sample_desc,
        "fixed_effects": "",
        "controls_desc": controls_desc,
        "cluster_var": "",
    }

    try:
        formula = f"{outcome_var} ~ {' + '.join(rhs_vars)}"
        all_vars = [outcome_var] + rhs_vars
        reg_data = data.dropna(subset=[v for v in all_vars if v in data.columns]).copy()

        model = pf.feols(formula, data=reg_data, vcov=vcov)

        coef_val = float(model.coef().get(treatment_var, np.nan))
        se_val = float(model.se().get(treatment_var, np.nan))
        pval = float(model.pvalue().get(treatment_var, np.nan))
        ci = model.confint()
        if treatment_var in ci.index:
            ci_lower = float(ci.loc[treatment_var].iloc[0])
            ci_upper = float(ci.loc[treatment_var].iloc[1])
        else:
            ci_lower = np.nan
            ci_upper = np.nan
        n_obs = int(model._N)
        r2 = float(model._r2)

        all_coefs = {k: float(v) for k, v in model.coef().items()}
        canon_inf = canonical_inference_for(baseline_group_id)

        payload = {
            "coefficients": all_coefs,
            "inference": {"spec_id": canon_inf["spec_id"], "params": canon_inf.get("params", {})},
            "software": software_block,
            "surface_hash": surface_hash_val,
            "design": design_block_for(baseline_group_id),
        }
        if extra_payload:
            for k, v in extra_payload.items():
                payload[k] = v

        row.update({
            "coefficient": coef_val,
            "std_error": se_val,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": n_obs,
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 1,
            "run_error": "",
        })
    except Exception as e:
        error_msg = str(e).replace("\n", " ")[:240]
        payload = {
            "error": error_msg,
            "error_details": {
                "stage": "estimation",
                "exception_type": type(e).__name__,
                "exception_message": str(e)[:500],
            }
        }
        row.update({
            "coefficient": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": np.nan,
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 0,
            "run_error": error_msg,
        })

    results.append(row)
    return row


def run_logit_probit_spec(spec_id, spec_run_id, spec_tree_path, outcome_var, treatment_var,
                          data, rhs_vars, model_type, sample_desc, controls_desc,
                          baseline_group_id, extra_payload=None):
    """Run a logit or probit specification and append to results."""
    row = {
        "paper_id": PAPER_ID,
        "spec_run_id": spec_run_id,
        "spec_id": spec_id,
        "spec_tree_path": spec_tree_path,
        "baseline_group_id": baseline_group_id,
        "outcome_var": outcome_var,
        "treatment_var": treatment_var,
        "sample_desc": sample_desc,
        "fixed_effects": "",
        "controls_desc": controls_desc,
        "cluster_var": "",
    }

    try:
        formula = f"{outcome_var} ~ {' + '.join(rhs_vars)}"
        all_vars = [outcome_var] + rhs_vars
        reg_data = data.dropna(subset=[v for v in all_vars if v in data.columns]).copy()

        if model_type == "logit":
            model = smf.logit(formula, data=reg_data).fit(disp=0, cov_type="HC3")
        else:
            model = smf.probit(formula, data=reg_data).fit(disp=0, cov_type="HC3")

        coef_val = float(model.params.get(treatment_var, np.nan))
        se_val = float(model.bse.get(treatment_var, np.nan))
        pval = float(model.pvalues.get(treatment_var, np.nan))
        ci = model.conf_int()
        if treatment_var in ci.index:
            ci_lower = float(ci.loc[treatment_var].iloc[0])
            ci_upper = float(ci.loc[treatment_var].iloc[1])
        else:
            ci_lower = np.nan
            ci_upper = np.nan
        n_obs = int(model.nobs)
        r2 = float(model.prsquared)

        all_coefs = {k: float(v) for k, v in model.params.items()}
        canon_inf = canonical_inference_for(baseline_group_id)

        payload = {
            "coefficients": all_coefs,
            "inference": {"spec_id": canon_inf["spec_id"], "params": canon_inf.get("params", {})},
            "software": software_block,
            "surface_hash": surface_hash_val,
            "design": design_block_for(baseline_group_id),
        }
        if extra_payload:
            for k, v in extra_payload.items():
                payload[k] = v

        row.update({
            "coefficient": coef_val,
            "std_error": se_val,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": n_obs,
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 1,
            "run_error": "",
        })
    except Exception as e:
        error_msg = str(e).replace("\n", " ")[:240]
        payload = {
            "error": error_msg,
            "error_details": {
                "stage": "estimation",
                "exception_type": type(e).__name__,
                "exception_message": str(e)[:500],
            }
        }
        row.update({
            "coefficient": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": np.nan,
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 0,
            "run_error": error_msg,
        })

    results.append(row)
    return row


def run_inference_variant(base_row, infer_spec_id, infer_tree_path, data, rhs_vars, vcov_type,
                          baseline_group_id):
    """Re-estimate with a different vcov and write to inference_results."""
    irow = {
        "paper_id": PAPER_ID,
        "inference_run_id": f"{base_row['spec_run_id']}__{infer_spec_id.replace('/', '_')}",
        "spec_run_id": base_row["spec_run_id"],
        "spec_id": infer_spec_id,
        "spec_tree_path": infer_tree_path,
        "baseline_group_id": baseline_group_id,
        "outcome_var": base_row["outcome_var"],
        "treatment_var": base_row["treatment_var"],
        "cluster_var": "",
    }
    try:
        outcome_var = base_row["outcome_var"]
        treatment_var = base_row["treatment_var"]
        formula = f"{outcome_var} ~ {' + '.join(rhs_vars)}"
        all_vars = [outcome_var] + rhs_vars
        reg_data = data.dropna(subset=[v for v in all_vars if v in data.columns]).copy()

        model = pf.feols(formula, data=reg_data, vcov=vcov_type)

        coef_val = float(model.coef().get(treatment_var, np.nan))
        se_val = float(model.se().get(treatment_var, np.nan))
        pval = float(model.pvalue().get(treatment_var, np.nan))
        ci = model.confint()
        if treatment_var in ci.index:
            ci_lower = float(ci.loc[treatment_var].iloc[0])
            ci_upper = float(ci.loc[treatment_var].iloc[1])
        else:
            ci_lower = np.nan
            ci_upper = np.nan
        n_obs = int(model._N)
        r2 = float(model._r2)

        all_coefs = {k: float(v) for k, v in model.coef().items()}
        payload = {
            "coefficients": all_coefs,
            "inference": {"spec_id": infer_spec_id, "params": {}},
            "software": software_block,
            "surface_hash": surface_hash_val,
        }

        irow.update({
            "coefficient": coef_val,
            "std_error": se_val,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": n_obs,
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 1,
            "run_error": "",
        })
    except Exception as e:
        error_msg = str(e).replace("\n", " ")[:240]
        payload = {
            "error": error_msg,
            "error_details": {
                "stage": "inference",
                "exception_type": type(e).__name__,
                "exception_message": str(e)[:500],
            }
        }
        irow.update({
            "coefficient": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": np.nan,
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 0,
            "run_error": error_msg,
        })

    inference_results.append(irow)
    return irow


# ============================================================
# SAMPLE DEFINITIONS
# ============================================================
# G1 & G3 base sample: Highx10==0 & Highx100==0
choice_base = choice[(choice['Highx10'] == 0) & (choice['Highx100'] == 0)].copy()

# G2 base sample: missingalpha==0 (all attentive have this)
nochoice_base = nochoice[nochoice['missingalpha'] == 0].copy()

# ============================================================
# G1 VARIABLES (Table 3: Recommendations in Choice experiment)
# ============================================================
G1_COVARIATES2 = ['professionalsfree', 'seeincentivecostly', 'seequalitycostly',
                  'wave2', 'wave3', 'professionalscloudresearch',
                  'incentiveshigh', 'incentiveleft', 'incentiveshigh_incentiveleft',
                  'age', 'female']

# Structural treatment indicators for G1
G1_STRUCTURAL_COL12 = ['choicebeforenoconflict', 'noconflict']
G1_STRUCTURAL_COL3 = ['choicebeforenoconflict', 'noconflict', 'notgetyourchoice',
                       'choicebeforenotgetyourchoice', 'notgetyourchoicenoconflict']

# Optional controls in G1 (can be dropped in LOO)
G1_OPTIONAL = ['incentiveB', 'professionalsfree', 'seeincentivecostly', 'seequalitycostly',
               'wave2', 'wave3', 'professionalscloudresearch',
               'incentiveshigh', 'incentiveleft', 'incentiveshigh_incentiveleft',
               'age', 'female']

# ============================================================
# G2 VARIABLES (Table C.1: Recommendations in NoChoice experiment)
# ============================================================
G2_CONTROLS_COL12 = ['noconflict', 'incentiveB', 'female', 'age', 'stdalpha']
G2_CONTROLS_COL3 = ['noconflict', 'seeincentivefirst_noconflict', 'incentiveB', 'female', 'age', 'stdalpha']
G2_OPTIONAL = ['incentiveB', 'female', 'age', 'stdalpha']
G2_STRUCTURAL_COL12 = ['noconflict']
G2_STRUCTURAL_COL3 = ['noconflict', 'seeincentivefirst_noconflict']

# ============================================================
# G3 VARIABLES (Table 2: Preferences in Choice experiment)
# ============================================================
G3_COVARIATES2 = ['professionalsfree', 'seeincentivecostly', 'seequalitycostly',
                  'wave2', 'wave3', 'professionalscloudresearch',
                  'incentiveshigh', 'incentiveleft', 'incentiveshigh_incentiveleft',
                  'age', 'female']

G3_STRUCTURAL = ['seequalitycostly']  # Always included as treatment indicator
G3_OPTIONAL = ['professionalsfree', 'wave2', 'wave3', 'professionalscloudresearch',
               'incentiveshigh', 'incentiveleft', 'incentiveshigh_incentiveleft',
               'age', 'female']

print("=" * 60)
print("RUNNING SPECIFICATION SEARCH FOR 180741-V1")
print("=" * 60)

# ============================================================
# STEP 1: BASELINE SPECIFICATIONS
# ============================================================
print("\n--- G1 Baselines (Table 3) ---")

# G1 Baseline: Table 3 Col 1 (getyourchoice==1)
g1_col1_sample = choice_base[choice_base['getyourchoice'] == 1].copy()
g1_col1_rhs = ['choicebefore'] + G1_STRUCTURAL_COL12 + ['incentiveB'] + G1_COVARIATES2
run_ols_spec(
    spec_id="baseline", spec_run_id="G1__baseline__table3_col1",
    spec_tree_path="designs/randomized_experiment.md#baseline",
    outcome_var="recommendincentive", treatment_var="choicebefore",
    data=g1_col1_sample, rhs_vars=g1_col1_rhs, vcov="HC3",
    sample_desc="Highx10==0 & Highx100==0 & getyourchoice==1",
    controls_desc="choicebeforenoconflict noconflict incentiveB professionalsfree seeincentivecostly seequalitycostly wave2 wave3 professionalscloudresearch incentiveshigh incentiveleft incentiveshigh_incentiveleft age female",
    baseline_group_id="G1",
)
print(f"  Table3-Col1: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}, N={results[-1]['n_obs']}")

# G1 Baseline: Table 3 Col 2 (getyourchoice==0)
g1_col2_sample = choice_base[choice_base['getyourchoice'] == 0].copy()
g1_col2_rhs = ['choicebefore'] + G1_STRUCTURAL_COL12 + ['incentiveB'] + G1_COVARIATES2
run_ols_spec(
    spec_id="baseline", spec_run_id="G1__baseline__table3_col2",
    spec_tree_path="designs/randomized_experiment.md#baseline",
    outcome_var="recommendincentive", treatment_var="choicebefore",
    data=g1_col2_sample, rhs_vars=g1_col2_rhs, vcov="HC3",
    sample_desc="Highx10==0 & Highx100==0 & getyourchoice==0",
    controls_desc="choicebeforenoconflict noconflict incentiveB professionalsfree seeincentivecostly seequalitycostly wave2 wave3 professionalscloudresearch incentiveshigh incentiveleft incentiveshigh_incentiveleft age female",
    baseline_group_id="G1",
)
print(f"  Table3-Col2: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}, N={results[-1]['n_obs']}")

# G1 Baseline: Table 3 Col 3 (combined)
g1_col3_rhs = ['choicebefore'] + G1_STRUCTURAL_COL3 + ['incentiveB'] + G1_COVARIATES2
run_ols_spec(
    spec_id="baseline", spec_run_id="G1__baseline__table3_col3",
    spec_tree_path="designs/randomized_experiment.md#baseline",
    outcome_var="recommendincentive", treatment_var="choicebefore",
    data=choice_base, rhs_vars=g1_col3_rhs, vcov="HC3",
    sample_desc="Highx10==0 & Highx100==0",
    controls_desc="choicebeforenoconflict noconflict notgetyourchoice choicebeforenotgetyourchoice notgetyourchoicenoconflict incentiveB professionalsfree seeincentivecostly seequalitycostly wave2 wave3 professionalscloudresearch incentiveshigh incentiveleft incentiveshigh_incentiveleft age female",
    baseline_group_id="G1",
)
print(f"  Table3-Col3: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}, N={results[-1]['n_obs']}")

# ============================================================
print("\n--- G2 Baselines (Table C.1) ---")

# G2 Baseline: Table C.1 Col 1 (conflict==1)
g2_col1_sample = nochoice_base[nochoice_base['conflict'] == 1].copy()
g2_col1_rhs = ['seeincentivefirst'] + G2_CONTROLS_COL12
run_ols_spec(
    spec_id="baseline", spec_run_id="G2__baseline__tablec1_col1",
    spec_tree_path="designs/randomized_experiment.md#baseline",
    outcome_var="recommendincentive", treatment_var="seeincentivefirst",
    data=g2_col1_sample, rhs_vars=g2_col1_rhs, vcov="HC3",
    sample_desc="missingalpha==0 & conflict==1",
    controls_desc="noconflict incentiveB female age stdalpha",
    baseline_group_id="G2",
)
print(f"  TableC1-Col1: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}, N={results[-1]['n_obs']}")

# G2 Baseline: Table C.1 Col 2 (conflict==0)
g2_col2_sample = nochoice_base[nochoice_base['conflict'] == 0].copy()
g2_col2_rhs = ['seeincentivefirst'] + G2_CONTROLS_COL12
run_ols_spec(
    spec_id="baseline", spec_run_id="G2__baseline__tablec1_col2",
    spec_tree_path="designs/randomized_experiment.md#baseline",
    outcome_var="recommendincentive", treatment_var="seeincentivefirst",
    data=g2_col2_sample, rhs_vars=g2_col2_rhs, vcov="HC3",
    sample_desc="missingalpha==0 & conflict==0",
    controls_desc="noconflict incentiveB female age stdalpha",
    baseline_group_id="G2",
)
print(f"  TableC1-Col2: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}, N={results[-1]['n_obs']}")

# G2 Baseline: Table C.1 Col 3 (full sample)
g2_col3_rhs = ['seeincentivefirst'] + G2_CONTROLS_COL3
run_ols_spec(
    spec_id="baseline", spec_run_id="G2__baseline__tablec1_col3",
    spec_tree_path="designs/randomized_experiment.md#baseline",
    outcome_var="recommendincentive", treatment_var="seeincentivefirst",
    data=nochoice_base, rhs_vars=g2_col3_rhs, vcov="HC3",
    sample_desc="missingalpha==0",
    controls_desc="noconflict seeincentivefirst_noconflict incentiveB female age stdalpha",
    baseline_group_id="G2",
)
print(f"  TableC1-Col3: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}, N={results[-1]['n_obs']}")

# ============================================================
print("\n--- G3 Baselines (Table 2) ---")

# G3 Baseline: Table 2 Col 1
g3_col1_rhs = ['seeincentivecostly'] + G3_COVARIATES2
run_ols_spec(
    spec_id="baseline", spec_run_id="G3__baseline__table2_col1",
    spec_tree_path="designs/randomized_experiment.md#baseline",
    outcome_var="choicebefore", treatment_var="seeincentivecostly",
    data=choice_base, rhs_vars=g3_col1_rhs, vcov="HC3",
    sample_desc="Highx10==0 & Highx100==0",
    controls_desc="seequalitycostly professionalsfree wave2 wave3 professionalscloudresearch incentiveshigh incentiveleft incentiveshigh_incentiveleft age female",
    baseline_group_id="G3",
)
print(f"  Table2-Col1: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}, N={results[-1]['n_obs']}")

# G3 Baseline: Table 2 Col 2 (professionals==0, adds stdalpha)
g3_col2_sample = choice_base[choice_base['professionals'] == 0].copy()
g3_col2_rhs = ['seeincentivecostly'] + G3_COVARIATES2 + ['stdalpha']
run_ols_spec(
    spec_id="baseline", spec_run_id="G3__baseline__table2_col2",
    spec_tree_path="designs/randomized_experiment.md#baseline",
    outcome_var="choicebefore", treatment_var="seeincentivecostly",
    data=g3_col2_sample, rhs_vars=g3_col2_rhs, vcov="HC3",
    sample_desc="Highx10==0 & Highx100==0 & professionals==0",
    controls_desc="seequalitycostly professionalsfree stdalpha wave2 wave3 professionalscloudresearch incentiveshigh incentiveleft incentiveshigh_incentiveleft age female",
    baseline_group_id="G3",
)
print(f"  Table2-Col2: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}, N={results[-1]['n_obs']}")

# G3 Baseline: Table 2 Col 3 (adds stdalpha + selfishness interactions)
g3_col3_rhs = ['seeincentivecostly'] + G3_COVARIATES2 + ['stdalpha', 'selfishseeincentivecostly', 'selfishseequalitycostly']
run_ols_spec(
    spec_id="baseline", spec_run_id="G3__baseline__table2_col3",
    spec_tree_path="designs/randomized_experiment.md#baseline",
    outcome_var="choicebefore", treatment_var="seeincentivecostly",
    data=choice_base, rhs_vars=g3_col3_rhs, vcov="HC3",
    sample_desc="Highx10==0 & Highx100==0",
    controls_desc="seequalitycostly professionalsfree stdalpha selfishseeincentivecostly selfishseequalitycostly wave2 wave3 professionalscloudresearch incentiveshigh incentiveleft incentiveshigh_incentiveleft age female",
    baseline_group_id="G3",
)
print(f"  Table2-Col3: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}, N={results[-1]['n_obs']}")


# ============================================================
# STEP 2: DESIGN VARIANTS
# ============================================================
print("\n--- Design Variants ---")

# For each group, run diff-in-means (no covariates) and with-covariates (minimal)
# G1 design variants use Table 3 Col 3 as the base (combined sample)

# G1: diff_in_means
run_ols_spec(
    spec_id="design/randomized_experiment/estimator/diff_in_means",
    spec_run_id="G1__design__diff_in_means",
    spec_tree_path="designs/randomized_experiment.md#diff-in-means",
    outcome_var="recommendincentive", treatment_var="choicebefore",
    data=choice_base, rhs_vars=['choicebefore'], vcov="HC3",
    sample_desc="Highx10==0 & Highx100==0",
    controls_desc="none",
    baseline_group_id="G1",
    extra_payload={"design": design_block_for("G1")},
)
print(f"  G1 diff-in-means: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}")

# G1: with_covariates (structural terms only, no optional controls)
run_ols_spec(
    spec_id="design/randomized_experiment/estimator/with_covariates",
    spec_run_id="G1__design__with_covariates",
    spec_tree_path="designs/randomized_experiment.md#with-covariates",
    outcome_var="recommendincentive", treatment_var="choicebefore",
    data=choice_base,
    rhs_vars=['choicebefore'] + G1_STRUCTURAL_COL3,
    vcov="HC3",
    sample_desc="Highx10==0 & Highx100==0",
    controls_desc="choicebeforenoconflict noconflict notgetyourchoice choicebeforenotgetyourchoice notgetyourchoicenoconflict (structural only)",
    baseline_group_id="G1",
)
print(f"  G1 with-covariates: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}")

# G2: diff_in_means (full sample)
run_ols_spec(
    spec_id="design/randomized_experiment/estimator/diff_in_means",
    spec_run_id="G2__design__diff_in_means",
    spec_tree_path="designs/randomized_experiment.md#diff-in-means",
    outcome_var="recommendincentive", treatment_var="seeincentivefirst",
    data=nochoice_base, rhs_vars=['seeincentivefirst'], vcov="HC3",
    sample_desc="missingalpha==0",
    controls_desc="none",
    baseline_group_id="G2",
)
print(f"  G2 diff-in-means: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}")

# G2: with_covariates (structural + noconflict + interaction)
run_ols_spec(
    spec_id="design/randomized_experiment/estimator/with_covariates",
    spec_run_id="G2__design__with_covariates",
    spec_tree_path="designs/randomized_experiment.md#with-covariates",
    outcome_var="recommendincentive", treatment_var="seeincentivefirst",
    data=nochoice_base,
    rhs_vars=['seeincentivefirst', 'noconflict', 'seeincentivefirst_noconflict'],
    vcov="HC3",
    sample_desc="missingalpha==0",
    controls_desc="noconflict seeincentivefirst_noconflict (structural only)",
    baseline_group_id="G2",
)
print(f"  G2 with-covariates: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}")

# G3: diff_in_means
run_ols_spec(
    spec_id="design/randomized_experiment/estimator/diff_in_means",
    spec_run_id="G3__design__diff_in_means",
    spec_tree_path="designs/randomized_experiment.md#diff-in-means",
    outcome_var="choicebefore", treatment_var="seeincentivecostly",
    data=choice_base, rhs_vars=['seeincentivecostly'], vcov="HC3",
    sample_desc="Highx10==0 & Highx100==0",
    controls_desc="none",
    baseline_group_id="G3",
)
print(f"  G3 diff-in-means: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}")

# G3: with_covariates (structural only: seequalitycostly)
run_ols_spec(
    spec_id="design/randomized_experiment/estimator/with_covariates",
    spec_run_id="G3__design__with_covariates",
    spec_tree_path="designs/randomized_experiment.md#with-covariates",
    outcome_var="choicebefore", treatment_var="seeincentivecostly",
    data=choice_base,
    rhs_vars=['seeincentivecostly', 'seequalitycostly'],
    vcov="HC3",
    sample_desc="Highx10==0 & Highx100==0",
    controls_desc="seequalitycostly (structural only)",
    baseline_group_id="G3",
)
print(f"  G3 with-covariates: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}")


# ============================================================
# STEP 3: RC VARIANTS
# ============================================================
print("\n--- RC Variants ---")

# ============================================================
# G1 RC VARIANTS (based on Table 3 Col 3 as reference baseline)
# ============================================================
print("\n  --- G1 RC ---")

# G1 LOO controls: drop optional controls one at a time from Col 3 specification
g1_base_rhs = ['choicebefore'] + G1_STRUCTURAL_COL3 + ['incentiveB'] + G1_COVARIATES2
g1_loo_controls = ['incentiveB', 'professionalsfree', 'seeincentivecostly', 'seequalitycostly',
                   'wave2', 'wave3', 'professionalscloudresearch',
                   'incentiveshigh', 'incentiveleft', 'incentiveshigh_incentiveleft',
                   'age', 'female']

for ctrl in g1_loo_controls:
    rhs = [v for v in g1_base_rhs if v != ctrl]
    run_ols_spec(
        spec_id=f"rc/controls/loo/{ctrl}",
        spec_run_id=f"G1__rc__loo__{ctrl}",
        spec_tree_path="modules/robustness/controls.md#leave-one-out-controls-loo",
        outcome_var="recommendincentive", treatment_var="choicebefore",
        data=choice_base, rhs_vars=rhs, vcov="HC3",
        sample_desc="Highx10==0 & Highx100==0",
        controls_desc=f"Table3-Col3 minus {ctrl}",
        baseline_group_id="G1",
        extra_payload={"controls": {
            "spec_id": f"rc/controls/loo/{ctrl}",
            "family": "loo",
            "dropped": [ctrl],
            "n_controls": len(rhs) - 1,  # minus treatment
        }},
    )
    print(f"    G1 LOO {ctrl}: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}")

# G1 ADD controls: add stdalpha
run_ols_spec(
    spec_id="rc/controls/add/stdalpha",
    spec_run_id="G1__rc__add__stdalpha",
    spec_tree_path="modules/robustness/controls.md#add-controls",
    outcome_var="recommendincentive", treatment_var="choicebefore",
    data=choice_base, rhs_vars=g1_base_rhs + ['stdalpha'], vcov="HC3",
    sample_desc="Highx10==0 & Highx100==0",
    controls_desc="Table3-Col3 plus stdalpha",
    baseline_group_id="G1",
    extra_payload={"controls": {
        "spec_id": "rc/controls/add/stdalpha",
        "family": "add",
        "added": ["stdalpha"],
        "n_controls": len(g1_base_rhs) + 1 - 1,
    }},
)
print(f"    G1 ADD stdalpha: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}")

# G1 ADD selfishseeincentivecostly
run_ols_spec(
    spec_id="rc/controls/add/selfishseeincentivecostly",
    spec_run_id="G1__rc__add__selfishseeincentivecostly",
    spec_tree_path="modules/robustness/controls.md#add-controls",
    outcome_var="recommendincentive", treatment_var="choicebefore",
    data=choice_base, rhs_vars=g1_base_rhs + ['selfishseeincentivecostly'], vcov="HC3",
    sample_desc="Highx10==0 & Highx100==0",
    controls_desc="Table3-Col3 plus selfishseeincentivecostly",
    baseline_group_id="G1",
    extra_payload={"controls": {
        "spec_id": "rc/controls/add/selfishseeincentivecostly",
        "family": "add",
        "added": ["selfishseeincentivecostly"],
        "n_controls": len(g1_base_rhs) + 1 - 1,
    }},
)
print(f"    G1 ADD selfishseeincentivecostly: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}")

# G1 ADD selfishseequalitycostly
run_ols_spec(
    spec_id="rc/controls/add/selfishseequalitycostly",
    spec_run_id="G1__rc__add__selfishseequalitycostly",
    spec_tree_path="modules/robustness/controls.md#add-controls",
    outcome_var="recommendincentive", treatment_var="choicebefore",
    data=choice_base, rhs_vars=g1_base_rhs + ['selfishseequalitycostly'], vcov="HC3",
    sample_desc="Highx10==0 & Highx100==0",
    controls_desc="Table3-Col3 plus selfishseequalitycostly",
    baseline_group_id="G1",
    extra_payload={"controls": {
        "spec_id": "rc/controls/add/selfishseequalitycostly",
        "family": "add",
        "added": ["selfishseequalitycostly"],
        "n_controls": len(g1_base_rhs) + 1 - 1,
    }},
)
print(f"    G1 ADD selfishseequalitycostly: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}")

# G1 SAMPLE variants
# include_inattentive: use raw data without the alphavaluefinal drop
choice_inattentive = choice_raw[(choice_raw['Highx10'] == 0) & (choice_raw['Highx100'] == 0)].copy()
run_ols_spec(
    spec_id="rc/sample/include_inattentive",
    spec_run_id="G1__rc__sample__include_inattentive",
    spec_tree_path="modules/robustness/sample.md#include-inattentive",
    outcome_var="recommendincentive", treatment_var="choicebefore",
    data=choice_inattentive, rhs_vars=g1_col3_rhs, vcov="HC3",
    sample_desc="Highx10==0 & Highx100==0, including inattentive",
    controls_desc="Table3-Col3 controls",
    baseline_group_id="G1",
    extra_payload={"sample": {
        "spec_id": "rc/sample/include_inattentive",
        "description": "Include inattentive participants (those with alphavaluefinal==.)"
    }},
)
print(f"    G1 include_inattentive: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}, N={results[-1]['n_obs']}")

# include_high_stakes_10x
choice_w10 = choice[(choice['Highx100'] == 0)].copy()  # include 10x but not 100x
run_ols_spec(
    spec_id="rc/sample/include_high_stakes_10x",
    spec_run_id="G1__rc__sample__include_high_stakes_10x",
    spec_tree_path="modules/robustness/sample.md#include-subgroup",
    outcome_var="recommendincentive", treatment_var="choicebefore",
    data=choice_w10, rhs_vars=g1_col3_rhs, vcov="HC3",
    sample_desc="Highx100==0 (includes 10x stakes)",
    controls_desc="Table3-Col3 controls",
    baseline_group_id="G1",
    extra_payload={"sample": {
        "spec_id": "rc/sample/include_high_stakes_10x",
        "description": "Include 10x high-stakes participants"
    }},
)
print(f"    G1 include_high_stakes_10x: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}, N={results[-1]['n_obs']}")

# include_high_stakes_100x
choice_wall = choice.copy()  # include all stakes
run_ols_spec(
    spec_id="rc/sample/include_high_stakes_100x",
    spec_run_id="G1__rc__sample__include_high_stakes_100x",
    spec_tree_path="modules/robustness/sample.md#include-subgroup",
    outcome_var="recommendincentive", treatment_var="choicebefore",
    data=choice_wall, rhs_vars=g1_col3_rhs, vcov="HC3",
    sample_desc="All stakes included",
    controls_desc="Table3-Col3 controls",
    baseline_group_id="G1",
    extra_payload={"sample": {
        "spec_id": "rc/sample/include_high_stakes_100x",
        "description": "Include all high-stakes participants (10x and 100x)"
    }},
)
print(f"    G1 include_high_stakes_100x: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}, N={results[-1]['n_obs']}")

# restrict_choicefree_only: treatment==0 (ChoiceFree non-professionals) plus treatment==1 (ChoiceFree_Professionals)
# Actually, from the data: study-based conditions map to treatment indicators
# ChoiceFree = condition=="ChoiceFree" or condition=="ChoiceFree_Professionals"
choice_free = choice_base[choice_base['condition'].isin(['ChoiceFree', 'ChoiceFree_Professionals'])].copy()
run_ols_spec(
    spec_id="rc/sample/restrict_choicefree_only",
    spec_run_id="G1__rc__sample__restrict_choicefree_only",
    spec_tree_path="modules/robustness/sample.md#restrict-subgroup",
    outcome_var="recommendincentive", treatment_var="choicebefore",
    data=choice_free,
    rhs_vars=['choicebefore'] + G1_STRUCTURAL_COL3 + ['incentiveB'] + ['wave2', 'wave3', 'professionalscloudresearch', 'incentiveshigh', 'incentiveleft', 'incentiveshigh_incentiveleft', 'age', 'female'],
    vcov="HC3",
    sample_desc="Highx10==0 & Highx100==0 & ChoiceFree only",
    controls_desc="Col3 controls minus seeincentivecostly seequalitycostly professionalsfree (collinear in ChoiceFree only)",
    baseline_group_id="G1",
    extra_payload={"sample": {
        "spec_id": "rc/sample/restrict_choicefree_only",
        "description": "Restrict to ChoiceFree conditions only"
    }},
)
print(f"    G1 restrict_choicefree_only: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}, N={results[-1]['n_obs']}")

# restrict_professionals_only
choice_prof = choice_base[choice_base['professionals'] == 1].copy()
run_ols_spec(
    spec_id="rc/sample/restrict_professionals_only",
    spec_run_id="G1__rc__sample__restrict_professionals_only",
    spec_tree_path="modules/robustness/sample.md#restrict-subgroup",
    outcome_var="recommendincentive", treatment_var="choicebefore",
    data=choice_prof,
    rhs_vars=['choicebefore'] + G1_STRUCTURAL_COL3 + ['incentiveB'] + ['wave2', 'wave3', 'incentiveshigh', 'incentiveleft', 'incentiveshigh_incentiveleft', 'age', 'female'],
    vcov="HC3",
    sample_desc="Highx10==0 & Highx100==0 & professionals==1",
    controls_desc="Col3 controls (professionals-only sample)",
    baseline_group_id="G1",
    extra_payload={"sample": {
        "spec_id": "rc/sample/restrict_professionals_only",
        "description": "Restrict to professionals only"
    }},
)
print(f"    G1 restrict_professionals_only: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}, N={results[-1]['n_obs']}")

# restrict_incentiveA_only
choice_incA = choice_base[choice_base['incentiveA'] == 1].copy()
run_ols_spec(
    spec_id="rc/sample/restrict_incentiveA_only",
    spec_run_id="G1__rc__sample__restrict_incentiveA_only",
    spec_tree_path="modules/robustness/sample.md#restrict-subgroup",
    outcome_var="recommendincentive", treatment_var="choicebefore",
    data=choice_incA,
    rhs_vars=['choicebefore'] + G1_STRUCTURAL_COL3 + G1_COVARIATES2,
    vcov="HC3",
    sample_desc="Highx10==0 & Highx100==0 & incentiveA==1",
    controls_desc="Col3 controls minus incentiveB (collinear)",
    baseline_group_id="G1",
    extra_payload={"sample": {
        "spec_id": "rc/sample/restrict_incentiveA_only",
        "description": "Restrict to incentiveA observations"
    }},
)
print(f"    G1 restrict_incentiveA_only: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}, N={results[-1]['n_obs']}")

# restrict_incentiveB_only
choice_incB = choice_base[choice_base['incentiveB'] == 1].copy()
run_ols_spec(
    spec_id="rc/sample/restrict_incentiveB_only",
    spec_run_id="G1__rc__sample__restrict_incentiveB_only",
    spec_tree_path="modules/robustness/sample.md#restrict-subgroup",
    outcome_var="recommendincentive", treatment_var="choicebefore",
    data=choice_incB,
    rhs_vars=['choicebefore'] + G1_STRUCTURAL_COL3 + G1_COVARIATES2,
    vcov="HC3",
    sample_desc="Highx10==0 & Highx100==0 & incentiveB==1",
    controls_desc="Col3 controls minus incentiveB (collinear)",
    baseline_group_id="G1",
    extra_payload={"sample": {
        "spec_id": "rc/sample/restrict_incentiveB_only",
        "description": "Restrict to incentiveB observations"
    }},
)
print(f"    G1 restrict_incentiveB_only: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}, N={results[-1]['n_obs']}")

# G1 OUTCOME variants: probit & logit
run_logit_probit_spec(
    spec_id="rc/outcome/probit",
    spec_run_id="G1__rc__outcome__probit",
    spec_tree_path="modules/robustness/functional_form.md#probit",
    outcome_var="recommendincentive", treatment_var="choicebefore",
    data=choice_base, rhs_vars=g1_col3_rhs, model_type="probit",
    sample_desc="Highx10==0 & Highx100==0",
    controls_desc="Table3-Col3 controls",
    baseline_group_id="G1",
    extra_payload={"functional_form": {
        "spec_id": "rc/outcome/probit",
        "model": "probit",
        "interpretation": "Probit coefficients (not marginal effects); sign and significance comparable to LPM"
    }},
)
print(f"    G1 probit: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}")

run_logit_probit_spec(
    spec_id="rc/outcome/logit",
    spec_run_id="G1__rc__outcome__logit",
    spec_tree_path="modules/robustness/functional_form.md#logit",
    outcome_var="recommendincentive", treatment_var="choicebefore",
    data=choice_base, rhs_vars=g1_col3_rhs, model_type="logit",
    sample_desc="Highx10==0 & Highx100==0",
    controls_desc="Table3-Col3 controls",
    baseline_group_id="G1",
    extra_payload={"functional_form": {
        "spec_id": "rc/outcome/logit",
        "model": "logit",
        "interpretation": "Logit coefficients (not marginal effects); sign and significance comparable to LPM"
    }},
)
print(f"    G1 logit: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}")


# ============================================================
# G2 RC VARIANTS
# ============================================================
print("\n  --- G2 RC ---")

# G2 LOO (based on Col 3 spec as reference)
g2_base_rhs = ['seeincentivefirst'] + G2_CONTROLS_COL3
g2_loo_controls = ['incentiveB', 'female', 'age', 'stdalpha']

for ctrl in g2_loo_controls:
    rhs = [v for v in g2_base_rhs if v != ctrl]
    run_ols_spec(
        spec_id=f"rc/controls/loo/{ctrl}",
        spec_run_id=f"G2__rc__loo__{ctrl}",
        spec_tree_path="modules/robustness/controls.md#leave-one-out-controls-loo",
        outcome_var="recommendincentive", treatment_var="seeincentivefirst",
        data=nochoice_base, rhs_vars=rhs, vcov="HC3",
        sample_desc="missingalpha==0",
        controls_desc=f"TableC1-Col3 minus {ctrl}",
        baseline_group_id="G2",
        extra_payload={"controls": {
            "spec_id": f"rc/controls/loo/{ctrl}",
            "family": "loo",
            "dropped": [ctrl],
            "n_controls": len(rhs) - 1,
        }},
    )
    print(f"    G2 LOO {ctrl}: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}")

# G2 minimal_nodemog: drop female, age, stdalpha
g2_nodemog_rhs = ['seeincentivefirst', 'noconflict', 'seeincentivefirst_noconflict', 'incentiveB']
run_ols_spec(
    spec_id="rc/controls/minimal_nodemog",
    spec_run_id="G2__rc__controls__minimal_nodemog",
    spec_tree_path="modules/robustness/controls.md#minimal-controls",
    outcome_var="recommendincentive", treatment_var="seeincentivefirst",
    data=nochoice_base, rhs_vars=g2_nodemog_rhs, vcov="HC3",
    sample_desc="missingalpha==0",
    controls_desc="noconflict seeincentivefirst_noconflict incentiveB (no demographics)",
    baseline_group_id="G2",
    extra_payload={"controls": {
        "spec_id": "rc/controls/minimal_nodemog",
        "family": "minimal",
        "dropped": ["female", "age", "stdalpha"],
        "n_controls": 3,
    }},
)
print(f"    G2 minimal_nodemog: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}")

# G2 SAMPLE: include_inattentive (use full nochoice without alphavaluefinal drop)
nc_inattentive = nc_raw.copy()
run_ols_spec(
    spec_id="rc/sample/include_inattentive",
    spec_run_id="G2__rc__sample__include_inattentive",
    spec_tree_path="modules/robustness/sample.md#include-inattentive",
    outcome_var="recommendincentive", treatment_var="seeincentivefirst",
    data=nc_inattentive,
    rhs_vars=['seeincentivefirst', 'noconflict', 'seeincentivefirst_noconflict', 'incentiveB', 'female', 'age'],
    vcov="HC3",
    sample_desc="All nochoice participants (including inattentive, no stdalpha)",
    controls_desc="noconflict seeincentivefirst_noconflict incentiveB female age (no stdalpha - missing for inattentive)",
    baseline_group_id="G2",
    extra_payload={"sample": {
        "spec_id": "rc/sample/include_inattentive",
        "description": "Include inattentive participants; stdalpha dropped since it is missing for inattentive"
    }},
)
print(f"    G2 include_inattentive: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}, N={results[-1]['n_obs']}")

# G2 SAMPLE: restrict_conflict_only
g2_conflict = nochoice_base[nochoice_base['conflict'] == 1].copy()
run_ols_spec(
    spec_id="rc/sample/restrict_conflict_only",
    spec_run_id="G2__rc__sample__restrict_conflict_only",
    spec_tree_path="modules/robustness/sample.md#restrict-subgroup",
    outcome_var="recommendincentive", treatment_var="seeincentivefirst",
    data=g2_conflict,
    rhs_vars=['seeincentivefirst', 'noconflict', 'incentiveB', 'female', 'age', 'stdalpha'],
    vcov="HC3",
    sample_desc="missingalpha==0 & conflict==1",
    controls_desc="noconflict incentiveB female age stdalpha",
    baseline_group_id="G2",
    extra_payload={"sample": {
        "spec_id": "rc/sample/restrict_conflict_only",
        "description": "Restrict to conflict subsample"
    }},
)
print(f"    G2 restrict_conflict_only: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}")

# G2 SAMPLE: restrict_noconflict_only
g2_noconflict = nochoice_base[nochoice_base['conflict'] == 0].copy()
run_ols_spec(
    spec_id="rc/sample/restrict_noconflict_only",
    spec_run_id="G2__rc__sample__restrict_noconflict_only",
    spec_tree_path="modules/robustness/sample.md#restrict-subgroup",
    outcome_var="recommendincentive", treatment_var="seeincentivefirst",
    data=g2_noconflict,
    rhs_vars=['seeincentivefirst', 'noconflict', 'incentiveB', 'female', 'age', 'stdalpha'],
    vcov="HC3",
    sample_desc="missingalpha==0 & conflict==0",
    controls_desc="noconflict incentiveB female age stdalpha",
    baseline_group_id="G2",
    extra_payload={"sample": {
        "spec_id": "rc/sample/restrict_noconflict_only",
        "description": "Restrict to no-conflict subsample"
    }},
)
print(f"    G2 restrict_noconflict_only: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}")

# G2 SAMPLE: restrict_incentiveA_only
g2_incA = nochoice_base[nochoice_base['incentiveA'] == 1].copy()
run_ols_spec(
    spec_id="rc/sample/restrict_incentiveA_only",
    spec_run_id="G2__rc__sample__restrict_incentiveA_only",
    spec_tree_path="modules/robustness/sample.md#restrict-subgroup",
    outcome_var="recommendincentive", treatment_var="seeincentivefirst",
    data=g2_incA,
    rhs_vars=['seeincentivefirst', 'noconflict', 'seeincentivefirst_noconflict', 'female', 'age', 'stdalpha'],
    vcov="HC3",
    sample_desc="missingalpha==0 & incentiveA==1",
    controls_desc="noconflict seeincentivefirst_noconflict female age stdalpha",
    baseline_group_id="G2",
    extra_payload={"sample": {
        "spec_id": "rc/sample/restrict_incentiveA_only",
        "description": "Restrict to incentiveA observations"
    }},
)
print(f"    G2 restrict_incentiveA_only: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}")

# G2 SAMPLE: restrict_incentiveB_only
g2_incB = nochoice_base[nochoice_base['incentiveB'] == 1].copy()
run_ols_spec(
    spec_id="rc/sample/restrict_incentiveB_only",
    spec_run_id="G2__rc__sample__restrict_incentiveB_only",
    spec_tree_path="modules/robustness/sample.md#restrict-subgroup",
    outcome_var="recommendincentive", treatment_var="seeincentivefirst",
    data=g2_incB,
    rhs_vars=['seeincentivefirst', 'noconflict', 'seeincentivefirst_noconflict', 'female', 'age', 'stdalpha'],
    vcov="HC3",
    sample_desc="missingalpha==0 & incentiveB==1",
    controls_desc="noconflict seeincentivefirst_noconflict female age stdalpha",
    baseline_group_id="G2",
    extra_payload={"sample": {
        "spec_id": "rc/sample/restrict_incentiveB_only",
        "description": "Restrict to incentiveB observations"
    }},
)
print(f"    G2 restrict_incentiveB_only: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}")

# G2 OUTCOME: probit & logit
run_logit_probit_spec(
    spec_id="rc/outcome/probit",
    spec_run_id="G2__rc__outcome__probit",
    spec_tree_path="modules/robustness/functional_form.md#probit",
    outcome_var="recommendincentive", treatment_var="seeincentivefirst",
    data=nochoice_base, rhs_vars=g2_base_rhs, model_type="probit",
    sample_desc="missingalpha==0",
    controls_desc="TableC1-Col3 controls",
    baseline_group_id="G2",
    extra_payload={"functional_form": {
        "spec_id": "rc/outcome/probit",
        "model": "probit",
        "interpretation": "Probit coefficients (not marginal effects)"
    }},
)
print(f"    G2 probit: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}")

run_logit_probit_spec(
    spec_id="rc/outcome/logit",
    spec_run_id="G2__rc__outcome__logit",
    spec_tree_path="modules/robustness/functional_form.md#logit",
    outcome_var="recommendincentive", treatment_var="seeincentivefirst",
    data=nochoice_base, rhs_vars=g2_base_rhs, model_type="logit",
    sample_desc="missingalpha==0",
    controls_desc="TableC1-Col3 controls",
    baseline_group_id="G2",
    extra_payload={"functional_form": {
        "spec_id": "rc/outcome/logit",
        "model": "logit",
        "interpretation": "Logit coefficients (not marginal effects)"
    }},
)
print(f"    G2 logit: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}")


# ============================================================
# G3 RC VARIANTS
# ============================================================
print("\n  --- G3 RC ---")

# G3 LOO (based on Col 1 spec as reference — the simplest full-sample spec)
g3_base_rhs = ['seeincentivecostly'] + G3_COVARIATES2
g3_loo_controls = ['professionalsfree', 'wave2', 'wave3', 'professionalscloudresearch',
                   'incentiveshigh', 'incentiveleft', 'incentiveshigh_incentiveleft',
                   'age', 'female']

for ctrl in g3_loo_controls:
    rhs = [v for v in g3_base_rhs if v != ctrl]
    run_ols_spec(
        spec_id=f"rc/controls/loo/{ctrl}",
        spec_run_id=f"G3__rc__loo__{ctrl}",
        spec_tree_path="modules/robustness/controls.md#leave-one-out-controls-loo",
        outcome_var="choicebefore", treatment_var="seeincentivecostly",
        data=choice_base, rhs_vars=rhs, vcov="HC3",
        sample_desc="Highx10==0 & Highx100==0",
        controls_desc=f"Table2-Col1 minus {ctrl}",
        baseline_group_id="G3",
        extra_payload={"controls": {
            "spec_id": f"rc/controls/loo/{ctrl}",
            "family": "loo",
            "dropped": [ctrl],
            "n_controls": len(rhs) - 1,
        }},
    )
    print(f"    G3 LOO {ctrl}: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}")

# G3 ADD stdalpha
run_ols_spec(
    spec_id="rc/controls/add/stdalpha",
    spec_run_id="G3__rc__add__stdalpha",
    spec_tree_path="modules/robustness/controls.md#add-controls",
    outcome_var="choicebefore", treatment_var="seeincentivecostly",
    data=choice_base, rhs_vars=g3_base_rhs + ['stdalpha'], vcov="HC3",
    sample_desc="Highx10==0 & Highx100==0",
    controls_desc="Table2-Col1 plus stdalpha",
    baseline_group_id="G3",
    extra_payload={"controls": {
        "spec_id": "rc/controls/add/stdalpha",
        "family": "add",
        "added": ["stdalpha"],
        "n_controls": len(g3_base_rhs),
    }},
)
print(f"    G3 ADD stdalpha: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}")

# G3 ADD selfishseeincentivecostly
run_ols_spec(
    spec_id="rc/controls/add/selfishseeincentivecostly",
    spec_run_id="G3__rc__add__selfishseeincentivecostly",
    spec_tree_path="modules/robustness/controls.md#add-controls",
    outcome_var="choicebefore", treatment_var="seeincentivecostly",
    data=choice_base, rhs_vars=g3_base_rhs + ['selfishseeincentivecostly'], vcov="HC3",
    sample_desc="Highx10==0 & Highx100==0",
    controls_desc="Table2-Col1 plus selfishseeincentivecostly",
    baseline_group_id="G3",
    extra_payload={"controls": {
        "spec_id": "rc/controls/add/selfishseeincentivecostly",
        "family": "add",
        "added": ["selfishseeincentivecostly"],
        "n_controls": len(g3_base_rhs),
    }},
)
print(f"    G3 ADD selfishseeincentivecostly: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}")

# G3 ADD selfishseequalitycostly
run_ols_spec(
    spec_id="rc/controls/add/selfishseequalitycostly",
    spec_run_id="G3__rc__add__selfishseequalitycostly",
    spec_tree_path="modules/robustness/controls.md#add-controls",
    outcome_var="choicebefore", treatment_var="seeincentivecostly",
    data=choice_base, rhs_vars=g3_base_rhs + ['selfishseequalitycostly'], vcov="HC3",
    sample_desc="Highx10==0 & Highx100==0",
    controls_desc="Table2-Col1 plus selfishseequalitycostly",
    baseline_group_id="G3",
    extra_payload={"controls": {
        "spec_id": "rc/controls/add/selfishseequalitycostly",
        "family": "add",
        "added": ["selfishseequalitycostly"],
        "n_controls": len(g3_base_rhs),
    }},
)
print(f"    G3 ADD selfishseequalitycostly: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}")

# G3 SAMPLE variants
# include_inattentive
run_ols_spec(
    spec_id="rc/sample/include_inattentive",
    spec_run_id="G3__rc__sample__include_inattentive",
    spec_tree_path="modules/robustness/sample.md#include-inattentive",
    outcome_var="choicebefore", treatment_var="seeincentivecostly",
    data=choice_inattentive, rhs_vars=g3_base_rhs, vcov="HC3",
    sample_desc="Highx10==0 & Highx100==0, including inattentive",
    controls_desc="Table2-Col1 controls",
    baseline_group_id="G3",
    extra_payload={"sample": {
        "spec_id": "rc/sample/include_inattentive",
        "description": "Include inattentive participants"
    }},
)
print(f"    G3 include_inattentive: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}")

# include_high_stakes_10x
run_ols_spec(
    spec_id="rc/sample/include_high_stakes_10x",
    spec_run_id="G3__rc__sample__include_high_stakes_10x",
    spec_tree_path="modules/robustness/sample.md#include-subgroup",
    outcome_var="choicebefore", treatment_var="seeincentivecostly",
    data=choice_w10, rhs_vars=g3_base_rhs, vcov="HC3",
    sample_desc="Highx100==0 (includes 10x stakes)",
    controls_desc="Table2-Col1 controls",
    baseline_group_id="G3",
    extra_payload={"sample": {
        "spec_id": "rc/sample/include_high_stakes_10x",
        "description": "Include 10x high-stakes participants"
    }},
)
print(f"    G3 include_high_stakes_10x: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}")

# include_high_stakes_100x
run_ols_spec(
    spec_id="rc/sample/include_high_stakes_100x",
    spec_run_id="G3__rc__sample__include_high_stakes_100x",
    spec_tree_path="modules/robustness/sample.md#include-subgroup",
    outcome_var="choicebefore", treatment_var="seeincentivecostly",
    data=choice_wall, rhs_vars=g3_base_rhs, vcov="HC3",
    sample_desc="All stakes included",
    controls_desc="Table2-Col1 controls",
    baseline_group_id="G3",
    extra_payload={"sample": {
        "spec_id": "rc/sample/include_high_stakes_100x",
        "description": "Include all high-stakes participants"
    }},
)
print(f"    G3 include_high_stakes_100x: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}")

# restrict_nonprofessionals_only
choice_nonprof = choice_base[choice_base['professionals'] == 0].copy()
run_ols_spec(
    spec_id="rc/sample/restrict_nonprofessionals_only",
    spec_run_id="G3__rc__sample__restrict_nonprofessionals_only",
    spec_tree_path="modules/robustness/sample.md#restrict-subgroup",
    outcome_var="choicebefore", treatment_var="seeincentivecostly",
    data=choice_nonprof, rhs_vars=g3_base_rhs, vcov="HC3",
    sample_desc="Highx10==0 & Highx100==0 & professionals==0",
    controls_desc="Table2-Col1 controls",
    baseline_group_id="G3",
    extra_payload={"sample": {
        "spec_id": "rc/sample/restrict_nonprofessionals_only",
        "description": "Restrict to non-professionals only"
    }},
)
print(f"    G3 restrict_nonprofessionals_only: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}")

# G3 OUTCOME: probit & logit
run_logit_probit_spec(
    spec_id="rc/outcome/probit",
    spec_run_id="G3__rc__outcome__probit",
    spec_tree_path="modules/robustness/functional_form.md#probit",
    outcome_var="choicebefore", treatment_var="seeincentivecostly",
    data=choice_base, rhs_vars=g3_base_rhs, model_type="probit",
    sample_desc="Highx10==0 & Highx100==0",
    controls_desc="Table2-Col1 controls",
    baseline_group_id="G3",
    extra_payload={"functional_form": {
        "spec_id": "rc/outcome/probit",
        "model": "probit",
        "interpretation": "Probit coefficients (not marginal effects)"
    }},
)
print(f"    G3 probit: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}")

run_logit_probit_spec(
    spec_id="rc/outcome/logit",
    spec_run_id="G3__rc__outcome__logit",
    spec_tree_path="modules/robustness/functional_form.md#logit",
    outcome_var="choicebefore", treatment_var="seeincentivecostly",
    data=choice_base, rhs_vars=g3_base_rhs, model_type="logit",
    sample_desc="Highx10==0 & Highx100==0",
    controls_desc="Table2-Col1 controls",
    baseline_group_id="G3",
    extra_payload={"functional_form": {
        "spec_id": "rc/outcome/logit",
        "model": "logit",
        "interpretation": "Logit coefficients (not marginal effects)"
    }},
)
print(f"    G3 logit: coef={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}")


# ============================================================
# STEP 4: INFERENCE VARIANTS
# ============================================================
print("\n--- Inference Variants ---")

# Run HC1 and HC2 variants for all baseline specs
inference_base_specs = {
    "G1": [
        ("G1__baseline__table3_col1", g1_col1_sample, g1_col1_rhs),
        ("G1__baseline__table3_col2", g1_col2_sample, g1_col2_rhs),
        ("G1__baseline__table3_col3", choice_base, g1_col3_rhs),
    ],
    "G2": [
        ("G2__baseline__tablec1_col1", g2_col1_sample, ['seeincentivefirst'] + G2_CONTROLS_COL12),
        ("G2__baseline__tablec1_col2", g2_col2_sample, ['seeincentivefirst'] + G2_CONTROLS_COL12),
        ("G2__baseline__tablec1_col3", nochoice_base, g2_base_rhs),
    ],
    "G3": [
        ("G3__baseline__table2_col1", choice_base, g3_base_rhs),
        ("G3__baseline__table2_col2", g3_col2_sample, g3_col2_rhs),
        ("G3__baseline__table2_col3", choice_base, g3_col3_rhs),
    ],
}

# Map group to its inference variants
inference_variant_map = {
    "G1": [
        ("infer/se/hc/hc1", "modules/inference/standard_errors.md#hc1", "hetero"),
        ("infer/se/hc/hc2", "modules/inference/standard_errors.md#hc2", "HC2"),
    ],
    "G2": [
        ("infer/se/hc/hc1", "modules/inference/standard_errors.md#hc1", "hetero"),
        ("infer/se/hc/hc2", "modules/inference/standard_errors.md#hc2", "HC2"),
    ],
    "G3": [
        ("infer/se/hc/hc1", "modules/inference/standard_errors.md#hc1", "hetero"),
    ],
}

# Find the matching base row for each baseline spec
base_rows_by_runid = {r["spec_run_id"]: r for r in results}

for group_id, specs in inference_base_specs.items():
    variants = inference_variant_map.get(group_id, [])
    for spec_run_id, data, rhs in specs:
        base_row = base_rows_by_runid.get(spec_run_id)
        if base_row is None or base_row["run_success"] == 0:
            continue
        for infer_sid, infer_path, vcov_val in variants:
            run_inference_variant(
                base_row=base_row,
                infer_spec_id=infer_sid,
                infer_tree_path=infer_path,
                data=data,
                rhs_vars=rhs,
                vcov_type=vcov_val,
                baseline_group_id=group_id,
            )
            print(f"    {spec_run_id} {infer_sid}: se={inference_results[-1]['std_error']:.4f}, p={inference_results[-1]['p_value']:.4f}")


# ============================================================
# WRITE OUTPUTS
# ============================================================
print("\n" + "=" * 60)
print("WRITING OUTPUTS")
print("=" * 60)

# Write specification_results.csv
spec_df = pd.DataFrame(results)
spec_cols = ["paper_id", "spec_run_id", "spec_id", "spec_tree_path", "baseline_group_id",
             "outcome_var", "treatment_var", "coefficient", "std_error", "p_value",
             "ci_lower", "ci_upper", "n_obs", "r_squared",
             "coefficient_vector_json", "sample_desc", "fixed_effects",
             "controls_desc", "cluster_var", "run_success", "run_error"]
spec_df = spec_df[spec_cols]
spec_df.to_csv(f"{PACKAGE_DIR}/specification_results.csv", index=False)
print(f"Wrote specification_results.csv: {len(spec_df)} rows")

# Write inference_results.csv
if inference_results:
    inf_df = pd.DataFrame(inference_results)
    inf_cols = ["paper_id", "inference_run_id", "spec_run_id", "spec_id", "spec_tree_path",
                "baseline_group_id", "outcome_var", "treatment_var",
                "coefficient", "std_error", "p_value",
                "ci_lower", "ci_upper", "n_obs", "r_squared",
                "coefficient_vector_json", "cluster_var", "run_success", "run_error"]
    inf_df = inf_df[inf_cols]
    inf_df.to_csv(f"{PACKAGE_DIR}/inference_results.csv", index=False)
    print(f"Wrote inference_results.csv: {len(inf_df)} rows")

# Summary counts
n_total = len(spec_df)
n_success = spec_df["run_success"].sum()
n_fail = n_total - n_success
n_baselines = len(spec_df[spec_df["spec_id"] == "baseline"])
n_design = len(spec_df[spec_df["spec_id"].str.startswith("design/")])
n_rc = len(spec_df[spec_df["spec_id"].str.startswith("rc/")])
n_infer = len(inf_df) if inference_results else 0

print(f"\nSummary:")
print(f"  Total core specs: {n_total}")
print(f"  Baselines: {n_baselines}")
print(f"  Design variants: {n_design}")
print(f"  RC variants: {n_rc}")
print(f"  Succeeded: {n_success}")
print(f"  Failed: {n_fail}")
print(f"  Inference variants: {n_infer}")

# Per-group counts
for gid in ["G1", "G2", "G3"]:
    g_df = spec_df[spec_df["baseline_group_id"] == gid]
    print(f"  {gid}: {len(g_df)} specs ({g_df['run_success'].sum()} success)")

print("\nDone!")
