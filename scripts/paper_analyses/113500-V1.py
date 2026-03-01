"""
Specification Search Script for Babcock, Recalde, Vesterlund (2017)
"Gender Differences in the Allocation of Low-Promotability Tasks:
 The Role of Backlash"
American Economic Review Papers & Proceedings, 107(5), 131-135.

Paper ID: 113500-V1

Surface-driven execution:
  - G1: decision ~ solicited + female + femaleXsolicited + controls, sample=Control (Table 1, Col 1)
  - G2: decision ~ solicited + female + femaleXsolicited + controls, sample=Backlash (Table 1, Col 2)
  - G3: decision ~ solicited + female + backlash + interactions + controls, pooled (Table 1, Col 3)
  - Baseline estimator: probit (marginal effects)
  - Design variants: LPM (diff-in-means, with covariates)
  - RC: LOO controls, control sets, functional form (LPM/logit), period splits
  - Inference: cluster(session_id) canonical; variants cluster(subject), HC1

Outputs:
  - specification_results.csv (baseline, design/*, rc/* rows)
  - inference_results.csv (infer/* rows)
  - SPECIFICATION_SEARCH.md
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import statsmodels.formula.api as smf
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

PAPER_ID = "113500-V1"
DATA_DIR = "data/downloads/extracted/113500-V1"
DATA_PATH = f"{DATA_DIR}/data_-corr_author_Recalde-/final_data_folder_aer_p-p/data/final_dataset.dta"
OUTPUT_DIR = DATA_DIR

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

# Design audit blocks from surface
G1_DESIGN_AUDIT = surface_obj["baseline_groups"][0]["design_audit"]
G2_DESIGN_AUDIT = surface_obj["baseline_groups"][1]["design_audit"]
G3_DESIGN_AUDIT = surface_obj["baseline_groups"][2]["design_audit"]
G1_INFERENCE_CANONICAL = surface_obj["baseline_groups"][0]["inference_plan"]["canonical"]
G2_INFERENCE_CANONICAL = surface_obj["baseline_groups"][1]["inference_plan"]["canonical"]
G3_INFERENCE_CANONICAL = surface_obj["baseline_groups"][2]["inference_plan"]["canonical"]

# ============================================================
# LOAD AND PREPARE DATA
# ============================================================
df_raw = pd.read_stata(DATA_PATH, convert_categoricals=False)

# Convert float32 to float64 for precision
for col in df_raw.columns:
    if df_raw[col].dtype == np.float32:
        df_raw[col] = df_raw[col].astype(np.float64)

# In the Stata file:
# treatment: 1=Control, 2=Backlash
# female: 0=Male, 1=Female (already numeric 0/1)
# student: 1=Freshman, 2=Sophomore, 3=Junior, 4=Senior (ordinal numeric)
# All other controls are already numeric

# Create derived variables
df_raw['backlash'] = (df_raw['treatment'] == 2).astype(float)
df_raw['femaleXsolicited'] = df_raw['female'] * df_raw['solicited']
df_raw['backlashXsolicited'] = df_raw['backlash'] * df_raw['solicited']
df_raw['femaleXbacklash'] = df_raw['female'] * df_raw['backlash']
df_raw['femaleXbacklashXsol'] = df_raw['female'] * df_raw['backlash'] * df_raw['solicited']

# Filter for green players (invest_group > 0)
df_green = df_raw[df_raw['invest_group'] > 0].copy()

# G1: Control condition (treatment==1)
df_g1 = df_green[df_green['treatment'] == 1].copy()
# G2: Backlash condition (treatment==2)
df_g2 = df_green[df_green['treatment'] == 2].copy()
# G3: Pooled (all green players)
df_g3 = df_green.copy()

print(f"Data loaded: {len(df_raw)} total rows")
print(f"Green players: {len(df_green)} rows")
print(f"G1 (Control): {len(df_g1)} rows, {df_g1['session_id'].nunique()} sessions")
print(f"G2 (Backlash): {len(df_g2)} rows, {df_g2['session_id'].nunique()} sessions")
print(f"G3 (Pooled): {len(df_g3)} rows, {df_g3['session_id'].nunique()} sessions")

# Period subsets
df_g1_first_half = df_g1[df_g1['period'] <= 5].copy()
df_g1_second_half = df_g1[df_g1['period'] > 5].copy()
df_g1_first_period = df_g1[df_g1['period'] == 1].copy()
df_g2_first_half = df_g2[df_g2['period'] <= 5].copy()
df_g2_second_half = df_g2[df_g2['period'] > 5].copy()
df_g2_first_period = df_g2[df_g2['period'] == 1].copy()
df_g3_first_half = df_g3[df_g3['period'] <= 5].copy()
df_g3_second_half = df_g3[df_g3['period'] > 5].copy()
df_g3_first_period = df_g3[df_g3['period'] == 1].copy()

# ============================================================
# VARIABLE DEFINITIONS
# ============================================================
ALL_CONTROLS = ["period", "risk_seeking1", "social1", "age", "non_caucasian", "student", "usborn", "business", "other"]
DEMOGRAPHICS_ONLY = ["age", "non_caucasian", "student", "usborn", "business", "other"]
PREFERENCES_ONLY = ["risk_seeking1", "social1"]

# G1/G2 treatment + interaction terms (mandatory, cannot be dropped)
G1G2_TREATMENT_VARS = ["solicited", "female", "femaleXsolicited"]
# G3 treatment + interaction terms (mandatory)
G3_TREATMENT_VARS = ["solicited", "female", "backlash", "femaleXsolicited",
                     "backlashXsolicited", "femaleXbacklash", "femaleXbacklashXsol"]

# Focal variable for G1/G2 is femaleXsolicited; for G3 is femaleXbacklashXsol
G1G2_FOCAL = "femaleXsolicited"
G3_FOCAL = "femaleXbacklashXsol"

results = []
inference_results = []
spec_run_counter = 0


# ============================================================
# HELPER: Run LPM via pyfixest (for design variants and rc/form/lpm)
# ============================================================
def run_lpm(spec_id, spec_tree_path, baseline_group_id,
            outcome_var, treatment_vars, focal_var, controls, data, vcov,
            sample_desc, controls_desc, cluster_var,
            design_audit, inference_canonical,
            axis_block_name=None, axis_block=None, functional_form=None, notes=""):
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        rhs = treatment_vars + controls
        rhs_str = " + ".join(rhs)
        formula = f"{outcome_var} ~ {rhs_str}"

        # Drop rows with NaN in regression variables
        reg_vars = [outcome_var] + rhs
        if cluster_var and cluster_var != "none":
            reg_vars.append(cluster_var)
        df_reg = data.dropna(subset=[v for v in reg_vars if v in data.columns]).copy()

        m = pf.feols(formula, data=df_reg, vcov=vcov)
        coef_val = float(m.coef().get(focal_var, np.nan))
        se_val = float(m.se().get(focal_var, np.nan))
        pval = float(m.pvalue().get(focal_var, np.nan))
        try:
            ci = m.confint()
            ci_lower = float(ci.loc[focal_var, ci.columns[0]])
            ci_upper = float(ci.loc[focal_var, ci.columns[1]])
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
        if functional_form:
            blocks["functional_form"] = functional_form

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": inference_canonical["spec_id"],
                       "params": inference_canonical["params"]},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"randomized_experiment": design_audit},
            blocks=blocks,
            notes=notes if notes else None,
        )

        results.append({
            "paper_id": PAPER_ID, "spec_run_id": run_id,
            "spec_id": spec_id, "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var, "treatment_var": focal_var,
            "coefficient": coef_val, "std_error": se_val, "p_value": pval,
            "ci_lower": ci_lower, "ci_upper": ci_upper,
            "n_obs": nobs, "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc, "fixed_effects": "none",
            "controls_desc": controls_desc,
            "cluster_var": cluster_var if cluster_var != "none" else "",
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
            "outcome_var": outcome_var, "treatment_var": focal_var,
            "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "n_obs": np.nan, "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc, "fixed_effects": "none",
            "controls_desc": controls_desc,
            "cluster_var": cluster_var if cluster_var != "none" else "",
            "run_success": 0, "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


# ============================================================
# HELPER: Run Probit marginal effects via statsmodels
# ============================================================
def run_probit_me(spec_id, spec_tree_path, baseline_group_id,
                  outcome_var, treatment_vars, focal_var, controls, data,
                  sample_desc, controls_desc, cluster_var,
                  design_audit, inference_canonical,
                  axis_block_name=None, axis_block=None, notes=""):
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        rhs = treatment_vars + controls
        rhs_str = " + ".join(rhs)
        formula = f"{outcome_var} ~ {rhs_str}"

        # Drop rows with NaN in regression variables
        reg_vars = [outcome_var] + rhs
        if cluster_var and cluster_var != "none":
            reg_vars.append(cluster_var)
        df_reg = data.dropna(subset=[v for v in reg_vars if v in data.columns]).copy()

        # Fit probit with clustered SEs
        if cluster_var and cluster_var != "none":
            probit_model = smf.probit(formula, data=df_reg).fit(
                cov_type='cluster', cov_kwds={'groups': df_reg[cluster_var]}, disp=0)
        else:
            probit_model = smf.probit(formula, data=df_reg).fit(cov_type='HC1', disp=0)

        mfx = probit_model.get_margeff(at='overall')
        mfx_frame = mfx.summary_frame()
        var_names = list(mfx_frame.index)

        if focal_var in var_names:
            idx = var_names.index(focal_var)
        else:
            # Focal variable not found in marginal effects -- fallback
            idx = 0

        coef_val = float(mfx.margeff[idx])
        se_val = float(mfx.margeff_se[idx])
        pval = float(mfx.pvalues[idx])
        ci_lower = float(mfx.conf_int()[idx, 0])
        ci_upper = float(mfx.conf_int()[idx, 1])
        nobs = int(probit_model.nobs)
        r2 = float(probit_model.prsquared)
        all_coefs = {var_names[i]: float(mfx.margeff[i]) for i in range(len(var_names))}

        blocks = {}
        if axis_block_name and axis_block:
            blocks[axis_block_name] = axis_block

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": inference_canonical["spec_id"],
                       "params": inference_canonical["params"]},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"randomized_experiment": design_audit},
            blocks=blocks,
            notes=notes if notes else None,
        )

        results.append({
            "paper_id": PAPER_ID, "spec_run_id": run_id,
            "spec_id": spec_id, "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var, "treatment_var": focal_var,
            "coefficient": coef_val, "std_error": se_val, "p_value": pval,
            "ci_lower": ci_lower, "ci_upper": ci_upper,
            "n_obs": nobs, "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc, "fixed_effects": "none",
            "controls_desc": controls_desc,
            "cluster_var": cluster_var if cluster_var != "none" else "",
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
            "outcome_var": outcome_var, "treatment_var": focal_var,
            "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "n_obs": np.nan, "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc, "fixed_effects": "none",
            "controls_desc": controls_desc,
            "cluster_var": cluster_var if cluster_var != "none" else "",
            "run_success": 0, "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


# ============================================================
# HELPER: Run Logit marginal effects via statsmodels
# ============================================================
def run_logit_me(spec_id, spec_tree_path, baseline_group_id,
                 outcome_var, treatment_vars, focal_var, controls, data,
                 sample_desc, controls_desc, cluster_var,
                 design_audit, inference_canonical,
                 axis_block_name=None, axis_block=None, notes=""):
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        rhs = treatment_vars + controls
        rhs_str = " + ".join(rhs)
        formula = f"{outcome_var} ~ {rhs_str}"

        reg_vars = [outcome_var] + rhs
        if cluster_var and cluster_var != "none":
            reg_vars.append(cluster_var)
        df_reg = data.dropna(subset=[v for v in reg_vars if v in data.columns]).copy()

        if cluster_var and cluster_var != "none":
            logit_model = smf.logit(formula, data=df_reg).fit(
                cov_type='cluster', cov_kwds={'groups': df_reg[cluster_var]}, disp=0)
        else:
            logit_model = smf.logit(formula, data=df_reg).fit(cov_type='HC1', disp=0)

        mfx = logit_model.get_margeff(at='overall')
        mfx_frame = mfx.summary_frame()
        var_names = list(mfx_frame.index)

        if focal_var in var_names:
            idx = var_names.index(focal_var)
        else:
            idx = 0

        coef_val = float(mfx.margeff[idx])
        se_val = float(mfx.margeff_se[idx])
        pval = float(mfx.pvalues[idx])
        ci_lower = float(mfx.conf_int()[idx, 0])
        ci_upper = float(mfx.conf_int()[idx, 1])
        nobs = int(logit_model.nobs)
        r2 = float(logit_model.prsquared)
        all_coefs = {var_names[i]: float(mfx.margeff[i]) for i in range(len(var_names))}

        blocks = {}
        if axis_block_name and axis_block:
            blocks[axis_block_name] = axis_block

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": inference_canonical["spec_id"],
                       "params": inference_canonical["params"]},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"randomized_experiment": design_audit},
            blocks=blocks,
            notes=notes if notes else None,
        )
        payload["functional_form"] = {
            "spec_id": "rc/form/estimator/logit",
            "estimator": "logit",
            "interpretation": "Average marginal effects from logit model"
        }

        results.append({
            "paper_id": PAPER_ID, "spec_run_id": run_id,
            "spec_id": spec_id, "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var, "treatment_var": focal_var,
            "coefficient": coef_val, "std_error": se_val, "p_value": pval,
            "ci_lower": ci_lower, "ci_upper": ci_upper,
            "n_obs": nobs, "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc, "fixed_effects": "none",
            "controls_desc": controls_desc,
            "cluster_var": cluster_var if cluster_var != "none" else "",
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
            "outcome_var": outcome_var, "treatment_var": focal_var,
            "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "n_obs": np.nan, "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc, "fixed_effects": "none",
            "controls_desc": controls_desc,
            "cluster_var": cluster_var if cluster_var != "none" else "",
            "run_success": 0, "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


# ============================================================
# HELPER: Run inference variant (LPM only, since that's what
# pyfixest can do easily with different vcov)
# ============================================================
def run_inference_variant(base_run_id, spec_id, spec_tree_path,
                          baseline_group_id, outcome_var, treatment_vars,
                          focal_var, controls, data, vcov,
                          cluster_var, design_audit):
    infer_counter = len(inference_results) + 1
    infer_run_id = f"{PAPER_ID}_infer_{infer_counter:03d}"

    try:
        rhs = treatment_vars + controls
        rhs_str = " + ".join(rhs)
        formula = f"{outcome_var} ~ {rhs_str}"

        reg_vars = [outcome_var] + rhs
        if isinstance(vcov, dict):
            clust = list(vcov.values())[0]
            reg_vars.append(clust)
        df_reg = data.dropna(subset=[v for v in reg_vars if v in data.columns]).copy()

        m = pf.feols(formula, data=df_reg, vcov=vcov)
        coef_val = float(m.coef().get(focal_var, np.nan))
        se_val = float(m.se().get(focal_var, np.nan))
        pval = float(m.pvalue().get(focal_var, np.nan))
        try:
            ci = m.confint()
            ci_lower = float(ci.loc[focal_var, ci.columns[0]])
            ci_upper = float(ci.loc[focal_var, ci.columns[1]])
        except Exception:
            ci_lower, ci_upper = np.nan, np.nan
        nobs = int(m._N)
        try:
            r2 = float(m._r2)
        except Exception:
            r2 = np.nan
        all_coefs = {k: float(v) for k, v in m.coef().items()}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": spec_id, "params": {"vcov": str(vcov), "cluster_var": cluster_var}},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"randomized_experiment": design_audit},
        )
        inference_results.append({
            "paper_id": PAPER_ID,
            "inference_run_id": infer_run_id,
            "spec_run_id": base_run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": focal_var,
            "coefficient": coef_val, "std_error": se_val, "p_value": pval,
            "ci_lower": ci_lower, "ci_upper": ci_upper,
            "n_obs": nobs, "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "cluster_var": cluster_var,
            "run_success": 1, "run_error": ""
        })
    except Exception as e:
        err_msg = str(e)[:240]
        payload = make_failure_payload(error=err_msg,
            error_details=error_details_from_exception(e, stage="inference"),
            software=SW_BLOCK, surface_hash=SURFACE_HASH)
        inference_results.append({
            "paper_id": PAPER_ID,
            "inference_run_id": infer_run_id,
            "spec_run_id": base_run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": focal_var,
            "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "n_obs": np.nan, "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "cluster_var": cluster_var,
            "run_success": 0, "run_error": err_msg
        })


# ============================================================
# HELPER: Run all specs for a given baseline group (G1 or G2)
# These have the same structure: 2-way interaction model
# ============================================================
def run_group_g1g2(group_id, data, design_audit, inference_canonical,
                   sample_desc_base, data_first_half, data_second_half,
                   data_first_period):
    """Run all specs for G1 or G2 (2-way interaction: female x solicited)."""
    treatment_vars = G1G2_TREATMENT_VARS
    focal_var = G1G2_FOCAL
    cluster_var = "session_id"
    vcov_canonical = {"CRV1": "session_id"}

    # === BASELINE ===
    print(f"=== {group_id} BASELINE ===")
    run_id_bl, *_ = run_probit_me(
        "baseline", "designs/randomized_experiment.md#baseline", group_id,
        "decision", treatment_vars, focal_var, ALL_CONTROLS, data,
        sample_desc_base, ", ".join(ALL_CONTROLS), cluster_var,
        design_audit, inference_canonical,
        notes=f"Probit marginal effects, Table 1 {'Col 1 (no-penalty)' if group_id=='G1' else 'Col 2 (backlash)'}"
    )

    # === DESIGN VARIANTS ===
    print(f"=== {group_id} DESIGN VARIANTS ===")

    # design/randomized_experiment/estimator/diff_in_means (LPM, no controls)
    run_id_dim, *_ = run_lpm(
        "design/randomized_experiment/estimator/diff_in_means",
        "designs/randomized_experiment.md#diff-in-means", group_id,
        "decision", treatment_vars, focal_var, [], data, vcov_canonical,
        sample_desc_base, "none (diff-in-means)", cluster_var,
        {**design_audit, "estimator": "diff_in_means"}, inference_canonical,
        notes="Difference-in-means: LPM with no controls"
    )

    # design/randomized_experiment/estimator/with_covariates (LPM + all controls)
    run_id_wc, *_ = run_lpm(
        "design/randomized_experiment/estimator/with_covariates",
        "designs/randomized_experiment.md#with-covariates", group_id,
        "decision", treatment_vars, focal_var, ALL_CONTROLS, data, vcov_canonical,
        sample_desc_base, ", ".join(ALL_CONTROLS), cluster_var,
        {**design_audit, "estimator": "ols_with_covariates"}, inference_canonical,
        notes="LPM with all controls (OLS analog of probit baseline)"
    )

    # === RC: LEAVE-ONE-OUT CONTROL DROPS ===
    print(f"=== {group_id} RC: LOO CONTROL DROPS ===")
    for ctrl in ALL_CONTROLS:
        remaining = [c for c in ALL_CONTROLS if c != ctrl]
        run_probit_me(
            f"rc/controls/loo/drop_{ctrl}",
            "modules/robustness/controls.md#leave-one-out-controls-loo", group_id,
            "decision", treatment_vars, focal_var, remaining, data,
            sample_desc_base, f"all controls minus {ctrl}", cluster_var,
            design_audit, inference_canonical,
            axis_block_name="controls",
            axis_block={"spec_id": f"rc/controls/loo/drop_{ctrl}", "family": "loo",
                        "dropped": [ctrl], "added": [], "n_controls": len(remaining)}
        )

    # === RC: CONTROL SETS ===
    print(f"=== {group_id} RC: CONTROL SETS ===")

    # no_controls
    run_probit_me(
        "rc/controls/sets/no_controls",
        "modules/robustness/controls.md#standard-control-sets", group_id,
        "decision", treatment_vars, focal_var, [], data,
        sample_desc_base, "none", cluster_var,
        design_audit, inference_canonical,
        axis_block_name="controls",
        axis_block={"spec_id": "rc/controls/sets/no_controls", "family": "sets",
                    "dropped": ALL_CONTROLS, "added": [], "n_controls": 0}
    )

    # demographics_only
    run_probit_me(
        "rc/controls/sets/demographics_only",
        "modules/robustness/controls.md#standard-control-sets", group_id,
        "decision", treatment_vars, focal_var, DEMOGRAPHICS_ONLY, data,
        sample_desc_base, ", ".join(DEMOGRAPHICS_ONLY), cluster_var,
        design_audit, inference_canonical,
        axis_block_name="controls",
        axis_block={"spec_id": "rc/controls/sets/demographics_only", "family": "sets",
                    "dropped": [c for c in ALL_CONTROLS if c not in DEMOGRAPHICS_ONLY],
                    "added": [], "n_controls": len(DEMOGRAPHICS_ONLY)}
    )

    # preferences_only
    run_probit_me(
        "rc/controls/sets/preferences_only",
        "modules/robustness/controls.md#standard-control-sets", group_id,
        "decision", treatment_vars, focal_var, PREFERENCES_ONLY, data,
        sample_desc_base, ", ".join(PREFERENCES_ONLY), cluster_var,
        design_audit, inference_canonical,
        axis_block_name="controls",
        axis_block={"spec_id": "rc/controls/sets/preferences_only", "family": "sets",
                    "dropped": [c for c in ALL_CONTROLS if c not in PREFERENCES_ONLY],
                    "added": [], "n_controls": len(PREFERENCES_ONLY)}
    )

    # === RC: FUNCTIONAL FORM ===
    print(f"=== {group_id} RC: FUNCTIONAL FORM ===")

    # LPM (linear probability model)
    run_lpm(
        "rc/form/estimator/lpm",
        "modules/robustness/functional_form.md#estimator-alternatives", group_id,
        "decision", treatment_vars, focal_var, ALL_CONTROLS, data, vcov_canonical,
        sample_desc_base, ", ".join(ALL_CONTROLS), cluster_var,
        design_audit, inference_canonical,
        functional_form={"spec_id": "rc/form/estimator/lpm", "estimator": "lpm",
                         "interpretation": "Linear probability model coefficients; interaction directly interpretable"},
        notes="LPM: interaction coefficient is directly interpretable (no inteff correction needed)"
    )

    # Logit
    run_logit_me(
        "rc/form/estimator/logit",
        "modules/robustness/functional_form.md#estimator-alternatives", group_id,
        "decision", treatment_vars, focal_var, ALL_CONTROLS, data,
        sample_desc_base, ", ".join(ALL_CONTROLS), cluster_var,
        design_audit, inference_canonical,
        axis_block_name="functional_form",
        axis_block={"spec_id": "rc/form/estimator/logit", "estimator": "logit",
                    "interpretation": "Average marginal effects from logit model"},
        notes="Logit marginal effects as alternative to probit"
    )

    # === RC: PERIOD SPLITS ===
    print(f"=== {group_id} RC: PERIOD SPLITS ===")

    # First half (periods 1-5)
    run_probit_me(
        "rc/sample/period/first_half",
        "modules/robustness/sample.md#period-restrictions", group_id,
        "decision", treatment_vars, focal_var, ALL_CONTROLS, data_first_half,
        f"{sample_desc_base}, periods 1-5", ", ".join(ALL_CONTROLS), cluster_var,
        design_audit, inference_canonical,
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/period/first_half", "restriction": "period <= 5",
                    "description": "First half of experiment (periods 1-5)"}
    )

    # Second half (periods 6-10)
    run_probit_me(
        "rc/sample/period/second_half",
        "modules/robustness/sample.md#period-restrictions", group_id,
        "decision", treatment_vars, focal_var, ALL_CONTROLS, data_second_half,
        f"{sample_desc_base}, periods 6-10", ", ".join(ALL_CONTROLS), cluster_var,
        design_audit, inference_canonical,
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/period/second_half", "restriction": "period > 5",
                    "description": "Second half of experiment (periods 6-10)"}
    )

    # First period only
    run_probit_me(
        "rc/sample/period/first_period_only",
        "modules/robustness/sample.md#period-restrictions", group_id,
        "decision", treatment_vars, focal_var,
        [c for c in ALL_CONTROLS if c != "period"],  # period is constant, drop it
        data_first_period,
        f"{sample_desc_base}, period 1 only",
        ", ".join([c for c in ALL_CONTROLS if c != "period"]),
        cluster_var,
        design_audit, inference_canonical,
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/period/first_period_only", "restriction": "period == 1",
                    "description": "First period only (cleanest experimental test, no learning)"}
    )

    # === INFERENCE VARIANTS (on the LPM with_covariates baseline) ===
    print(f"=== {group_id} INFERENCE VARIANTS ===")

    # Cluster at subject level
    run_inference_variant(
        run_id_wc, "infer/se/cluster/subject",
        "modules/inference/standard_errors.md#cluster", group_id,
        "decision", treatment_vars, focal_var, ALL_CONTROLS, data,
        {"CRV1": "unique_subjectid"}, "unique_subjectid", design_audit
    )

    # HC1 robust (no clustering)
    run_inference_variant(
        run_id_wc, "infer/se/hc/hc1",
        "modules/inference/standard_errors.md#robust", group_id,
        "decision", treatment_vars, focal_var, ALL_CONTROLS, data,
        "hetero", "none (HC1)", design_audit
    )

    return run_id_bl, run_id_dim, run_id_wc


# ============================================================
# HELPER: Run all specs for G3 (pooled, triple interaction)
# ============================================================
def run_group_g3():
    """Run all specs for G3 (triple interaction: female x backlash x solicited)."""
    group_id = "G3"
    treatment_vars = G3_TREATMENT_VARS
    focal_var = G3_FOCAL
    cluster_var = "session_id"
    vcov_canonical = {"CRV1": "session_id"}
    design_audit = G3_DESIGN_AUDIT
    inference_canonical = G3_INFERENCE_CANONICAL
    sample_desc_base = "invest_group > 0 (all green players, both treatments)"

    # === BASELINE ===
    print(f"=== {group_id} BASELINE ===")
    run_id_bl, *_ = run_probit_me(
        "baseline", "designs/randomized_experiment.md#baseline", group_id,
        "decision", treatment_vars, focal_var, ALL_CONTROLS, df_g3,
        sample_desc_base, ", ".join(ALL_CONTROLS), cluster_var,
        design_audit, inference_canonical,
        notes="Probit marginal effects, Table 1 Col 3 (pooled triple interaction)"
    )

    # === DESIGN VARIANTS ===
    print(f"=== {group_id} DESIGN VARIANTS ===")

    # diff-in-means (LPM, no controls)
    run_id_dim, *_ = run_lpm(
        "design/randomized_experiment/estimator/diff_in_means",
        "designs/randomized_experiment.md#diff-in-means", group_id,
        "decision", treatment_vars, focal_var, [], df_g3, vcov_canonical,
        sample_desc_base, "none (diff-in-means)", cluster_var,
        {**design_audit, "estimator": "diff_in_means"}, inference_canonical,
        notes="Difference-in-means: LPM with no controls, triple interaction"
    )

    # with_covariates (LPM + all controls)
    run_id_wc, *_ = run_lpm(
        "design/randomized_experiment/estimator/with_covariates",
        "designs/randomized_experiment.md#with-covariates", group_id,
        "decision", treatment_vars, focal_var, ALL_CONTROLS, df_g3, vcov_canonical,
        sample_desc_base, ", ".join(ALL_CONTROLS), cluster_var,
        {**design_audit, "estimator": "ols_with_covariates"}, inference_canonical,
        notes="LPM with all controls (OLS analog of probit baseline)"
    )

    # === RC: LOO CONTROL DROPS ===
    print(f"=== {group_id} RC: LOO CONTROL DROPS ===")
    for ctrl in ALL_CONTROLS:
        remaining = [c for c in ALL_CONTROLS if c != ctrl]
        run_probit_me(
            f"rc/controls/loo/drop_{ctrl}",
            "modules/robustness/controls.md#leave-one-out-controls-loo", group_id,
            "decision", treatment_vars, focal_var, remaining, df_g3,
            sample_desc_base, f"all controls minus {ctrl}", cluster_var,
            design_audit, inference_canonical,
            axis_block_name="controls",
            axis_block={"spec_id": f"rc/controls/loo/drop_{ctrl}", "family": "loo",
                        "dropped": [ctrl], "added": [], "n_controls": len(remaining)}
        )

    # === RC: CONTROL SETS ===
    print(f"=== {group_id} RC: CONTROL SETS ===")

    run_probit_me(
        "rc/controls/sets/no_controls",
        "modules/robustness/controls.md#standard-control-sets", group_id,
        "decision", treatment_vars, focal_var, [], df_g3,
        sample_desc_base, "none", cluster_var,
        design_audit, inference_canonical,
        axis_block_name="controls",
        axis_block={"spec_id": "rc/controls/sets/no_controls", "family": "sets",
                    "dropped": ALL_CONTROLS, "added": [], "n_controls": 0}
    )

    run_probit_me(
        "rc/controls/sets/demographics_only",
        "modules/robustness/controls.md#standard-control-sets", group_id,
        "decision", treatment_vars, focal_var, DEMOGRAPHICS_ONLY, df_g3,
        sample_desc_base, ", ".join(DEMOGRAPHICS_ONLY), cluster_var,
        design_audit, inference_canonical,
        axis_block_name="controls",
        axis_block={"spec_id": "rc/controls/sets/demographics_only", "family": "sets",
                    "dropped": [c for c in ALL_CONTROLS if c not in DEMOGRAPHICS_ONLY],
                    "added": [], "n_controls": len(DEMOGRAPHICS_ONLY)}
    )

    run_probit_me(
        "rc/controls/sets/preferences_only",
        "modules/robustness/controls.md#standard-control-sets", group_id,
        "decision", treatment_vars, focal_var, PREFERENCES_ONLY, df_g3,
        sample_desc_base, ", ".join(PREFERENCES_ONLY), cluster_var,
        design_audit, inference_canonical,
        axis_block_name="controls",
        axis_block={"spec_id": "rc/controls/sets/preferences_only", "family": "sets",
                    "dropped": [c for c in ALL_CONTROLS if c not in PREFERENCES_ONLY],
                    "added": [], "n_controls": len(PREFERENCES_ONLY)}
    )

    # === RC: FUNCTIONAL FORM ===
    print(f"=== {group_id} RC: FUNCTIONAL FORM ===")

    # LPM
    run_lpm(
        "rc/form/estimator/lpm",
        "modules/robustness/functional_form.md#estimator-alternatives", group_id,
        "decision", treatment_vars, focal_var, ALL_CONTROLS, df_g3, vcov_canonical,
        sample_desc_base, ", ".join(ALL_CONTROLS), cluster_var,
        design_audit, inference_canonical,
        functional_form={"spec_id": "rc/form/estimator/lpm", "estimator": "lpm",
                         "interpretation": "Linear probability model; triple interaction directly interpretable"},
        notes="LPM: triple interaction coefficient directly interpretable"
    )

    # Logit
    run_logit_me(
        "rc/form/estimator/logit",
        "modules/robustness/functional_form.md#estimator-alternatives", group_id,
        "decision", treatment_vars, focal_var, ALL_CONTROLS, df_g3,
        sample_desc_base, ", ".join(ALL_CONTROLS), cluster_var,
        design_audit, inference_canonical,
        axis_block_name="functional_form",
        axis_block={"spec_id": "rc/form/estimator/logit", "estimator": "logit",
                    "interpretation": "Average marginal effects from logit model"},
        notes="Logit marginal effects as alternative to probit"
    )

    # === RC: PERIOD SPLITS ===
    print(f"=== {group_id} RC: PERIOD SPLITS ===")

    run_probit_me(
        "rc/sample/period/first_half",
        "modules/robustness/sample.md#period-restrictions", group_id,
        "decision", treatment_vars, focal_var, ALL_CONTROLS, df_g3_first_half,
        f"{sample_desc_base}, periods 1-5", ", ".join(ALL_CONTROLS), cluster_var,
        design_audit, inference_canonical,
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/period/first_half", "restriction": "period <= 5",
                    "description": "First half of experiment (periods 1-5)"}
    )

    run_probit_me(
        "rc/sample/period/second_half",
        "modules/robustness/sample.md#period-restrictions", group_id,
        "decision", treatment_vars, focal_var, ALL_CONTROLS, df_g3_second_half,
        f"{sample_desc_base}, periods 6-10", ", ".join(ALL_CONTROLS), cluster_var,
        design_audit, inference_canonical,
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/period/second_half", "restriction": "period > 5",
                    "description": "Second half of experiment (periods 6-10)"}
    )

    run_probit_me(
        "rc/sample/period/first_period_only",
        "modules/robustness/sample.md#period-restrictions", group_id,
        "decision", treatment_vars, focal_var,
        [c for c in ALL_CONTROLS if c != "period"],
        df_g3_first_period,
        f"{sample_desc_base}, period 1 only",
        ", ".join([c for c in ALL_CONTROLS if c != "period"]),
        cluster_var,
        design_audit, inference_canonical,
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/period/first_period_only", "restriction": "period == 1",
                    "description": "First period only (cleanest experimental test, no learning)"}
    )

    # === INFERENCE VARIANTS (on LPM with_covariates) ===
    print(f"=== {group_id} INFERENCE VARIANTS ===")

    # Cluster at subject
    run_inference_variant(
        run_id_wc, "infer/se/cluster/subject",
        "modules/inference/standard_errors.md#cluster", group_id,
        "decision", treatment_vars, focal_var, ALL_CONTROLS, df_g3,
        {"CRV1": "unique_subjectid"}, "unique_subjectid", design_audit
    )

    # HC1 robust
    run_inference_variant(
        run_id_wc, "infer/se/hc/hc1",
        "modules/inference/standard_errors.md#robust", group_id,
        "decision", treatment_vars, focal_var, ALL_CONTROLS, df_g3,
        "hetero", "none (HC1)", design_audit
    )


# ############################################################
# RUN ALL GROUPS
# ############################################################

# G1: No-penalty condition
print("\n" + "="*60)
print("G1: NO-PENALTY CONDITION (treatment==Control)")
print("="*60)
run_group_g1g2(
    "G1", df_g1, G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    "invest_group > 0 & treatment==Control (green players, no-penalty)",
    df_g1_first_half, df_g1_second_half, df_g1_first_period
)

# G2: Backlash condition
print("\n" + "="*60)
print("G2: BACKLASH CONDITION (treatment==Backlash)")
print("="*60)
run_group_g1g2(
    "G2", df_g2, G2_DESIGN_AUDIT, G2_INFERENCE_CANONICAL,
    "invest_group > 0 & treatment==Backlash (green players, penalty condition)",
    df_g2_first_half, df_g2_second_half, df_g2_first_period
)

# G3: Pooled triple interaction
print("\n" + "="*60)
print("G3: POOLED TRIPLE INTERACTION (both treatments)")
print("="*60)
run_group_g3()


# ############################################################
# WRITE OUTPUTS
# ############################################################
print("\n" + "="*60)
print("WRITING OUTPUTS")
print("="*60)

# specification_results.csv
df_results = pd.DataFrame(results)
df_results.to_csv(f"{OUTPUT_DIR}/specification_results.csv", index=False)
print(f"Wrote {len(df_results)} rows to specification_results.csv")
print(f"  Successful: {df_results['run_success'].sum()}")
print(f"  Failed: {(df_results['run_success'] == 0).sum()}")

# inference_results.csv
df_infer = pd.DataFrame(inference_results)
df_infer.to_csv(f"{OUTPUT_DIR}/inference_results.csv", index=False)
print(f"Wrote {len(df_infer)} rows to inference_results.csv")

# Count by group
for g in ["G1", "G2", "G3"]:
    n_g = len(df_results[df_results['baseline_group_id'] == g])
    n_succ = df_results[(df_results['baseline_group_id'] == g) & (df_results['run_success'] == 1)].shape[0]
    print(f"  {g}: {n_g} total specs, {n_succ} successful")

# SPECIFICATION_SEARCH.md
n_planned_per_group = 1 + 2 + 9 + 3 + 2 + 3  # baseline + design + loo + sets + form + period = 20
n_total_planned = n_planned_per_group * 3
n_inference_planned = 2 * 3  # 2 inference variants per group
n_total_executed = len(df_results)
n_total_success = int(df_results['run_success'].sum())
n_total_failed = n_total_executed - n_total_success

md_content = f"""# Specification Search: {PAPER_ID}

## Paper
Babcock, Recalde, Vesterlund (2017), "Gender Differences in the Allocation of Low-Promotability Tasks: The Role of Backlash," AER P&P.

## Surface Summary
- **Design**: randomized_experiment (lab experiment)
- **Baseline groups**: 3
  - G1: Gender gap in solicitation response, no-penalty condition (Table 1, Col 1)
  - G2: Gender gap in solicitation response, backlash/penalty condition (Table 1, Col 2)
  - G3: Cross-treatment comparison via triple interaction (Table 1, Col 3)
- **Budget**: 55 specs per group (nominal)
- **Seed**: 113500
- **Canonical inference**: Cluster at session_id

## Execution Summary
- **Total specification rows**: {n_total_executed}
- **Successful**: {n_total_success}
- **Failed**: {n_total_failed}
- **Inference variant rows**: {len(df_infer)}

### Per-group breakdown
| Group | Planned | Executed | Successful | Failed |
|-------|---------|----------|------------|--------|
"""

for g in ["G1", "G2", "G3"]:
    n_exec = len(df_results[df_results['baseline_group_id'] == g])
    n_succ = int(df_results[(df_results['baseline_group_id'] == g) & (df_results['run_success'] == 1)].shape[0])
    n_fail = n_exec - n_succ
    md_content += f"| {g} | {n_planned_per_group} | {n_exec} | {n_succ} | {n_fail} |\n"

md_content += f"""
### Spec types executed per group
- 1 baseline (probit marginal effects, cluster session_id)
- 2 design variants (diff-in-means LPM, LPM with covariates)
- 9 LOO control drops (probit, drop each of 9 controls)
- 3 control sets (no controls, demographics only, preferences only)
- 2 functional form variants (LPM, logit)
- 3 period sample splits (first half, second half, first period only)
- **Total: {n_planned_per_group} per group x 3 groups = {n_total_planned}**

### Inference variants (on LPM with_covariates baseline)
- 2 per group x 3 groups = {n_inference_planned} total
  - infer/se/cluster/subject (cluster at unique_subjectid)
  - infer/se/hc/hc1 (robust HC1, no clustering)
- Wild cluster bootstrap (infer/bootstrap/wild_cluster/session) SKIPPED: wildboottest package not available in environment

## Data Preparation Notes
- `treatment` is numeric 1/2 in .dta (1=Control, 2=Backlash)
- `female` is numeric 0/1 in .dta (0=Male, 1=Female)
- `student` is numeric 1-4 in .dta (1=Freshman, 2=Sophomore, 3=Junior, 4=Senior), used as ordinal numeric in regressions
- Constructed variables: backlash, femaleXsolicited, backlashXsolicited, femaleXbacklash, femaleXbacklashXsol
- Data read with convert_categoricals=False to preserve original Stata numeric encoding

## Clustering Bug in Original Code
The Stata do-file defines `local clust_var unique_subjectid` but then uses `cluster(\\`clus_var\\')` (misspelled macro name), so the original published probit estimates are effectively **unclustered**. Our baseline uses cluster(session_id) as specified by the surface (correct design-based choice). The paper's primary inference method is cgmwildboot (wild cluster bootstrap), not the probit clustered SEs.

## Probit Interaction Effects
For probit models, the marginal effect of the interaction term (femaleXsolicited) reported by statsmodels `get_margeff(at='overall')` is the average marginal effect, which differs from the Stata `inteff`-corrected interaction effect. The LPM variant provides a directly interpretable interaction coefficient. See Norton, Wang, Ai (2004) for the distinction.

## Software Stack
- Python {sys.version.split()[0]}
- pyfixest (LPM estimation with clustered SEs)
- statsmodels (probit/logit marginal effects with clustered SEs)
- pandas, numpy
"""

# Add failure details if any
if n_total_failed > 0:
    md_content += "\n## Failed Specifications\n"
    failed = df_results[df_results['run_success'] == 0]
    for _, row in failed.iterrows():
        md_content += f"- `{row['spec_run_id']}` ({row['spec_id']}, {row['baseline_group_id']}): {row['run_error']}\n"

with open(f"{OUTPUT_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write(md_content)
print(f"Wrote SPECIFICATION_SEARCH.md")

print("\n=== DONE ===")
print(f"Total specs: {n_total_executed}")
print(f"Total inference: {len(df_infer)}")
