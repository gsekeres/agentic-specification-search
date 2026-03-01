"""
Specification Search Script for Lopez, Sautmann, Schaner (AEJ Applied)
"Does Patient Demand Contribute to the Overuse of Prescription Drugs?"

Paper ID: 126722-V1

Surface-driven execution:
  - G1: Voucher effects on malaria treatment outcomes (5 outcomes)
  - Randomized experiment: OLS with date FE + manual controls + clinic clustering
  - ~65 specifications total

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
import hashlib
import traceback
import warnings
warnings.filterwarnings('ignore')

# Add scripts dir for utilities
sys.path.insert(0, "scripts")
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash, software_block
)

DATA_DIR = "data/downloads/extracted/126722-V1"
PAPER_ID = "126722-V1"

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

# Load data
df_raw = pd.read_stata(f"{DATA_DIR}/Data/LSS_analysis_datasets_20201108.dta")
df_raw = df_raw[df_raw['dropme'] != 1].copy()

# Convert ALL categorical columns to numeric
# Handle different category encodings
for col in df_raw.columns:
    if hasattr(df_raw[col], 'cat') or str(df_raw[col].dtype) == 'category':
        cats = set(df_raw[col].dropna().astype(str).unique())
        is_na = df_raw[col].isna()
        if cats <= {'Yes', 'No'}:
            # Binary Yes/No -> 1/0
            df_raw[col] = (df_raw[col].astype(str) == 'Yes').astype(float)
            df_raw.loc[is_na, col] = np.nan
        elif cats <= {'Male', 'Female'}:
            # Gender: Male=1, Female=0 (Stata convention for genderpatient)
            df_raw[col] = (df_raw[col].astype(str) == 'Male').astype(float)
            df_raw.loc[is_na, col] = np.nan
        else:
            # Try numeric conversion
            try:
                str_vals = df_raw[col].astype(str)
                str_vals[is_na] = np.nan
                df_raw[col] = pd.to_numeric(str_vals, errors='coerce')
            except:
                pass  # leave as is

# Convert all float32 to float64
for col in df_raw.columns:
    if df_raw[col].dtype == np.float32:
        df_raw[col] = df_raw[col].astype(np.float64)

# Ensure numeric types for key variables
for v in ['num_symptoms', 'daysillness99', 'agepatient', 'MSSpregnancy', 'MSSethnic_bambara',
          'above_med_pos', 'pred_mal_pos', 'cscomnum_OS',
          'symptomsscreening_1', 'symptomsscreening_2', 'symptomsscreening_3',
          'symptomsscreening_4', 'symptomsscreening_5', 'symptomsscreening_6',
          'respondent', 'gender', 'genderpatient', 'under5', 'pregnancy',
          'ethnic_bambara', 'speak_french', 'readwrite_fluent_french', 'prischoolorless',
          'patient_voucher', 'doctor_voucher']:
    if v in df_raw.columns:
        df_raw[v] = pd.to_numeric(df_raw[v], errors='coerce')

# Create clinic cluster string variable for pyfixest
df_raw['clinic_id'] = df_raw['cscomnum_OS'].astype(str)

# Define variable groups
# Date FE dummies (DD1-DD35)
date_fe_vars = [f'DD{i}' for i in range(1, 36)]
date_fe_vars = [v for v in date_fe_vars if v in df_raw.columns]

# Manual covariates (matching Table B10 -- no lasso)
manual_covariates = [
    'num_symptoms',
    'symptomsscreening_1', 'symptomsscreening_2', 'symptomsscreening_3',
    'symptomsscreening_4', 'symptomsscreening_5', 'symptomsscreening_6',
    'daysillness99', 'agepatient', 'under5', 'genderpatient',
    'pregnancy', 'MSSpregnancy',
    'respondent', 'gender', 'ethnic_bambara', 'MSSethnic_bambara',
    'speak_french', 'readwrite_fluent_french', 'prischoolorless'
]
# Verify all exist
manual_covariates = [v for v in manual_covariates if v in df_raw.columns]

# Symptom-only controls (minimal set)
symptom_covariates = [
    'num_symptoms',
    'symptomsscreening_1', 'symptomsscreening_2', 'symptomsscreening_3',
    'symptomsscreening_4', 'symptomsscreening_5', 'symptomsscreening_6',
    'daysillness99'
]
symptom_covariates = [v for v in symptom_covariates if v in df_raw.columns]

# Extended controls (add demographic + HH)
extended_additional = ['agepatient', 'under5', 'genderpatient', 'pregnancy']
extended_additional = [v for v in extended_additional if v in df_raw.columns]

# Outcome variables
outcomes = {
    'RXtreat_sev_simple_mal': 'Prescribed any antimalarial',
    'treat_sev_simple_mal': 'Purchased any antimalarial',
    'RXtreat_severe_mal': 'Prescribed severe malaria treatment',
    'treat_severe_mal': 'Purchased severe malaria treatment',
    'used_vouchers_admin': 'Used voucher (admin)'
}

# Treatment variables -- both entered simultaneously
treatment_vars = ['patient_voucher', 'doctor_voucher']
focal_treatment = 'patient_voucher'

# Handle missing values: recode to sample mean (matching Stata approach)
# The paper explicitly does this: "Missing values are recoded to the sample mean"
# MSSpregnancy and MSSethnic_bambara are already missing-value indicators
for v in manual_covariates:
    if v in df_raw.columns and v not in ['MSSpregnancy', 'MSSethnic_bambara']:
        if df_raw[v].isna().any():
            col_mean = df_raw[v].mean()
            df_raw[v] = df_raw[v].fillna(col_mean)

# Also impute date FE dummies (should be complete but just in case)
for v in date_fe_vars:
    if v in df_raw.columns and df_raw[v].isna().any():
        df_raw[v] = df_raw[v].fillna(0)

design_audit = surface_obj["baseline_groups"][0]["design_audit"]
inference_canonical = surface_obj["baseline_groups"][0]["inference_plan"]["canonical"]

results = []
inference_results = []
spec_run_counter = 0


def run_spec(spec_id, spec_tree_path, baseline_group_id, outcome_var, treatment_var,
             controls, date_fe_included, clinic_fe_included, data, vcov_spec,
             sample_desc, controls_desc, cluster_var="cscomnum_OS",
             axis_block_name=None, axis_block=None, notes="",
             secondary_treatment="doctor_voucher"):
    """Run a single specification and append to results."""
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        # Build formula
        treat_str = f"{treatment_var} + {secondary_treatment}" if secondary_treatment else treatment_var
        controls_str = " + ".join(controls) if controls else ""

        # Build FE part
        fe_parts = []
        if date_fe_included:
            # Include date FE as explicit dummies in formula (no absorb)
            pass  # We'll add date dummies to controls
        if clinic_fe_included:
            fe_parts.append("clinic_id")

        # Add date FE dummies to controls if needed
        all_rhs = [treatment_var]
        if secondary_treatment:
            all_rhs.append(secondary_treatment)
        if controls:
            all_rhs.extend(controls)
        if date_fe_included:
            all_rhs.extend(date_fe_vars)

        rhs_str = " + ".join(all_rhs)

        if fe_parts:
            formula = f"{outcome_var} ~ {rhs_str} | {' + '.join(fe_parts)}"
        else:
            formula = f"{outcome_var} ~ {rhs_str}"

        # Drop rows with NaN in outcome and treatment only
        # Controls have been mean-imputed, so should be complete
        essential_vars = [outcome_var, treatment_var]
        if secondary_treatment:
            essential_vars.append(secondary_treatment)
        data_clean = data.dropna(subset=[v for v in essential_vars if v in data.columns]).copy()

        m = pf.feols(formula, data=data_clean, vcov=vcov_spec)

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

        # Get all coefficients (limit to treatment vars to keep JSON manageable)
        all_coefs = {}
        for k, v in m.coef().items():
            if k in treatment_vars or k in controls[:5]:  # focal + first few controls
                all_coefs[k] = float(v)

        fe_desc_parts = []
        if date_fe_included:
            fe_desc_parts.append("date")
        if clinic_fe_included:
            fe_desc_parts.append("clinic")
        fe_desc = " + ".join(fe_desc_parts) if fe_desc_parts else "none"

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": inference_canonical["spec_id"],
                      "method": "cluster", "cluster_var": "cscomnum_OS"},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"randomized_experiment": design_audit},
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
            "fixed_effects": "",
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
            "run_success": 0,
            "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


def run_inference_variant(base_run_id, spec_id, spec_tree_path, baseline_group_id,
                          outcome_var, treatment_var, controls, date_fe_included,
                          clinic_fe_included, data, vcov_spec, cluster_var_str,
                          secondary_treatment="doctor_voucher"):
    """Run an inference variant for an existing estimate."""
    global spec_run_counter

    try:
        treat_str = f"{treatment_var} + {secondary_treatment}" if secondary_treatment else treatment_var
        all_rhs = [treatment_var]
        if secondary_treatment:
            all_rhs.append(secondary_treatment)
        if controls:
            all_rhs.extend(controls)
        if date_fe_included:
            all_rhs.extend(date_fe_vars)

        rhs_str = " + ".join(all_rhs)

        fe_parts = []
        if clinic_fe_included:
            fe_parts.append("clinic_id")

        if fe_parts:
            formula = f"{outcome_var} ~ {rhs_str} | {' + '.join(fe_parts)}"
        else:
            formula = f"{outcome_var} ~ {rhs_str}"

        essential_vars = [outcome_var, treatment_var]
        if secondary_treatment:
            essential_vars.append(secondary_treatment)
        data_clean = data.dropna(subset=[v for v in essential_vars if v in data.columns]).copy()

        m = pf.feols(formula, data=data_clean, vcov=vcov_spec)

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

        all_coefs = {treatment_var: coef_val}
        if secondary_treatment:
            all_coefs[secondary_treatment] = float(m.coef().get(secondary_treatment, np.nan))

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": spec_id, "method": "robust" if vcov_spec == "hetero" else "cluster",
                      "type": "HC1" if vcov_spec == "hetero" else "CRV1"},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"randomized_experiment": design_audit}
        )

        infer_run_id = f"{PAPER_ID}_infer_{len(inference_results)+1:03d}"
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
            "n_obs": int(m._N),
            "r_squared": float(m._r2) if hasattr(m, '_r2') else np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "cluster_var": cluster_var_str,
            "run_success": 1,
            "run_error": ""
        })

    except Exception as e:
        err_msg = str(e)[:240]
        err_details = error_details_from_exception(e, stage="inference")
        payload = make_failure_payload(
            error=err_msg, error_details=err_details,
            software=SW_BLOCK, surface_hash=SURFACE_HASH
        )
        infer_run_id = f"{PAPER_ID}_infer_{len(inference_results)+1:03d}"
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
            "cluster_var": cluster_var_str,
            "run_success": 0,
            "run_error": err_msg
        })


# ============================================================
# EXECUTE SPECIFICATIONS
# ============================================================

print("=" * 60)
print(f"Running specification search for {PAPER_ID}")
print("=" * 60)

# Canonical vcov: cluster at clinic level
canonical_vcov = {"CRV1": "clinic_id"}

# ===== BASELINE: RXtreat_sev_simple_mal (primary outcome) =====
print("\n--- BASELINE: RXtreat_sev_simple_mal ---")
base_run = run_spec(
    "baseline",
    "designs/randomized_experiment.md#baseline", "G1",
    "RXtreat_sev_simple_mal", "patient_voucher",
    manual_covariates, True, False, df_raw, canonical_vcov,
    "dropme!=1, N~2053", "Manual covariates (Table B10 approach)"
)
print(f"  baseline: coef={base_run[1]:.4f}, se={base_run[2]:.4f}, p={base_run[3]:.4f}, N={base_run[4]}")

# ===== ADDITIONAL BASELINES (other main outcomes) =====
additional_outcomes = {
    'treat_sev_simple_mal': 'baseline__treat_sev_simple_mal',
    'used_vouchers_admin': 'baseline__used_vouchers_admin',
    'RXtreat_severe_mal': 'baseline__RXtreat_severe_mal',
    'treat_severe_mal': 'baseline__treat_severe_mal'
}

for out_var, spec_id in additional_outcomes.items():
    print(f"\n--- {spec_id} ---")
    r = run_spec(
        spec_id,
        "designs/randomized_experiment.md#baseline", "G1",
        out_var, "patient_voucher",
        manual_covariates, True, False, df_raw, canonical_vcov,
        "dropme!=1", f"Manual covariates; outcome={out_var}"
    )
    print(f"  {spec_id}: coef={r[1]:.4f}, se={r[2]:.4f}, p={r[3]:.4f}, N={r[4]}")

# ===== DESIGN VARIANTS =====
# For all 5 outcomes

for out_var, out_label in outcomes.items():
    print(f"\n--- Design variants for {out_var} ---")

    # 1. diff_in_means: date FE only, no additional controls
    r = run_spec(
        f"design/randomized_experiment/estimator/diff_in_means__{out_var}",
        "designs/randomized_experiment.md#a-itt-implementations",
        "G1", out_var, "patient_voucher",
        [], True, False, df_raw, canonical_vcov,
        "dropme!=1", "No additional controls (date FE only)"
    )
    print(f"  diff_in_means: coef={r[1]:.4f}, p={r[3]:.4f}, N={r[4]}")

    # 2. with_covariates: same as baseline (already done for primary)
    if out_var != 'RXtreat_sev_simple_mal':
        # Already done as baseline/additional baseline
        pass

    # 3. strata_fe: add clinic FE
    r = run_spec(
        f"design/randomized_experiment/estimator/strata_fe__{out_var}",
        "designs/randomized_experiment.md#a-itt-implementations",
        "G1", out_var, "patient_voucher",
        manual_covariates, True, True, df_raw, canonical_vcov,
        "dropme!=1", "Manual covariates + clinic FE"
    )
    print(f"  strata_fe: coef={r[1]:.4f}, p={r[3]:.4f}, N={r[4]}")


# ===== RC: CONTROLS LOO =====
# For the primary outcome only (RXtreat_sev_simple_mal)
primary_outcome = 'RXtreat_sev_simple_mal'
loo_candidates = [v for v in manual_covariates if v not in ['MSSpregnancy', 'MSSethnic_bambara']]
# Group symptom dummies -- drop all symptoms together as one LOO
symptom_group = ['symptomsscreening_1', 'symptomsscreening_2', 'symptomsscreening_3',
                 'symptomsscreening_4', 'symptomsscreening_5', 'symptomsscreening_6']
individual_loo = [v for v in loo_candidates if v not in symptom_group]

print(f"\n--- RC: Controls LOO for {primary_outcome} ---")

# Drop individual covariates
for var in individual_loo:
    ctrl = [c for c in manual_covariates if c != var]
    r = run_spec(
        f"rc/controls/loo/drop_{var}",
        "modules/robustness/controls.md#leave-one-out-controls-loo", "G1",
        primary_outcome, "patient_voucher",
        ctrl, True, False, df_raw, canonical_vcov,
        "dropme!=1", f"Manual covariates minus {var}",
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/controls/loo/drop_{var}", "family": "loo",
                   "dropped": [var], "added": [], "n_controls": len(ctrl)}
    )
    print(f"  loo/drop_{var}: coef={r[1]:.4f}, p={r[3]:.4f}")

# Drop all symptom dummies together
ctrl = [c for c in manual_covariates if c not in symptom_group and c != 'num_symptoms']
r = run_spec(
    "rc/controls/loo/drop_symptoms",
    "modules/robustness/controls.md#leave-one-out-controls-loo", "G1",
    primary_outcome, "patient_voucher",
    ctrl, True, False, df_raw, canonical_vcov,
    "dropme!=1", "Manual covariates minus all symptom vars",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_symptoms", "family": "loo",
               "dropped": symptom_group + ['num_symptoms'], "added": [], "n_controls": len(ctrl)}
)
print(f"  loo/drop_symptoms: coef={r[1]:.4f}, p={r[3]:.4f}")


# ===== RC: CONTROL SETS =====
print(f"\n--- RC: Control sets for {primary_outcome} ---")

# No controls
r = run_spec(
    "rc/controls/sets/none",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    primary_outcome, "patient_voucher",
    [], True, False, df_raw, canonical_vcov,
    "dropme!=1", "No additional controls (date FE only)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/none", "family": "sets",
               "dropped": manual_covariates, "added": [], "n_controls": 0, "set_name": "none"}
)
print(f"  sets/none: coef={r[1]:.4f}, p={r[3]:.4f}")

# Minimal: symptoms only
r = run_spec(
    "rc/controls/sets/minimal",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    primary_outcome, "patient_voucher",
    symptom_covariates, True, False, df_raw, canonical_vcov,
    "dropme!=1", "Symptom covariates only",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/minimal", "family": "sets",
               "dropped": [], "added": [], "n_controls": len(symptom_covariates), "set_name": "minimal_symptoms"}
)
print(f"  sets/minimal: coef={r[1]:.4f}, p={r[3]:.4f}")

# Extended: manual + demographic interactions
r = run_spec(
    "rc/controls/sets/extended",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    primary_outcome, "patient_voucher",
    manual_covariates, True, False, df_raw, canonical_vcov,
    "dropme!=1", "Full manual covariate set (= baseline)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/extended", "family": "sets",
               "dropped": [], "added": [], "n_controls": len(manual_covariates), "set_name": "extended"}
)
print(f"  sets/extended: coef={r[1]:.4f}, p={r[3]:.4f}")


# ===== RC: CONTROL PROGRESSION =====
print(f"\n--- RC: Control progression for {primary_outcome} ---")

progression_steps = [
    ("rc/controls/progression/bivariate", [], "Bivariate (date FE only)"),
    ("rc/controls/progression/symptoms_only", symptom_covariates, "Symptoms + illness duration"),
    ("rc/controls/progression/demographics",
     symptom_covariates + ['agepatient', 'under5', 'genderpatient', 'pregnancy', 'MSSpregnancy'],
     "Symptoms + demographics"),
    ("rc/controls/progression/respondent",
     symptom_covariates + ['agepatient', 'under5', 'genderpatient', 'pregnancy', 'MSSpregnancy',
                           'respondent', 'gender'],
     "Symptoms + demographics + respondent"),
    ("rc/controls/progression/full", manual_covariates, "Full manual covariate set"),
]

for spec_id, ctrls, desc in progression_steps:
    ctrls = [c for c in ctrls if c in df_raw.columns]
    r = run_spec(
        spec_id,
        "modules/robustness/controls.md#control-progression-build-up", "G1",
        primary_outcome, "patient_voucher",
        ctrls, True, False, df_raw, canonical_vcov,
        "dropme!=1", desc,
        axis_block_name="controls",
        axis_block={"spec_id": spec_id, "family": "progression",
                   "dropped": [], "added": [], "n_controls": len(ctrls), "set_name": desc}
    )
    print(f"  {spec_id}: coef={r[1]:.4f}, p={r[3]:.4f}")


# ===== RC: CONTROL SUBSET (random draws) =====
print(f"\n--- RC: Control subset random draws for {primary_outcome} ---")

rng = np.random.RandomState(126722)
subset_pool = [v for v in manual_covariates if v not in ['MSSpregnancy', 'MSSethnic_bambara']]
mandatory = []  # No mandatory controls beyond date FE

for draw_i in range(1, 16):
    k = rng.randint(3, len(subset_pool) + 1)
    chosen = list(rng.choice(subset_pool, size=k, replace=False))
    # Add missing indicators if pregnancy or ethnic_bambara included
    if 'pregnancy' in chosen and 'MSSpregnancy' not in chosen:
        chosen.append('MSSpregnancy')
    if 'ethnic_bambara' in chosen and 'MSSethnic_bambara' not in chosen:
        chosen.append('MSSethnic_bambara')

    spec_id = f"rc/controls/subset/random_{draw_i:03d}"
    ctrls = mandatory + chosen
    excluded = [v for v in subset_pool if v not in chosen]
    r = run_spec(
        spec_id,
        "modules/robustness/controls.md#subset-generation-specids", "G1",
        primary_outcome, "patient_voucher",
        ctrls, True, False, df_raw, canonical_vcov,
        "dropme!=1", f"Random subset draw {draw_i} ({len(ctrls)} controls)",
        axis_block_name="controls",
        axis_block={"spec_id": spec_id, "family": "subset", "method": "random",
                   "seed": 126722, "draw_index": draw_i,
                   "pool": subset_pool, "mandatory": mandatory,
                   "included": chosen, "excluded": excluded,
                   "n_controls": len(ctrls)}
    )
    print(f"  subset/random_{draw_i:03d}: coef={r[1]:.4f}, p={r[3]:.4f}, n_controls={len(ctrls)}")


# ===== RC: SAMPLE =====
print(f"\n--- RC: Sample restrictions ---")

# For all outcomes: trimming doesn't make sense for binary 0/1 outcomes
# Instead: restrict to home survey subsample
df_home = df_raw[df_raw['home'] == 1].copy() if 'home' in df_raw.columns else df_raw[df_raw['in_home_survey'] == 1].copy() if 'in_home_survey' in df_raw.columns else None

if df_home is not None and len(df_home) > 50:
    for out_var in ['RXtreat_sev_simple_mal', 'treat_sev_simple_mal']:
        r = run_spec(
            f"rc/sample/restriction/home_survey__{out_var}",
            "modules/robustness/sample.md#subpopulations-and-domain-restrictions", "G1",
            out_var, "patient_voucher",
            manual_covariates, True, False, df_home, canonical_vcov,
            "dropme!=1, home survey subsample", "Manual covariates, home survey only",
            axis_block_name="sample",
            axis_block={"spec_id": f"rc/sample/restriction/home_survey__{out_var}",
                       "axis": "restriction", "rule": "home_survey",
                       "n_obs_before": len(df_raw), "n_obs_after": len(df_home)}
        )
        print(f"  sample/home_survey__{out_var}: coef={r[1]:.4f}, p={r[3]:.4f}, N={r[4]}")
else:
    print("  Home survey subsample not available or too small")
    # Try alternative: patients with valid RDT
    df_rdt = df_raw[df_raw['RDTresult_POS'].notna()].copy() if 'RDTresult_POS' in df_raw.columns else None
    if df_rdt is not None and len(df_rdt) > 50:
        for out_var in ['RXtreat_sev_simple_mal', 'treat_sev_simple_mal']:
            r = run_spec(
                f"rc/sample/restriction/valid_rdt__{out_var}",
                "modules/robustness/sample.md#subpopulations-and-domain-restrictions", "G1",
                out_var, "patient_voucher",
                manual_covariates, True, False, df_rdt, canonical_vcov,
                "dropme!=1, valid RDT result", "Manual covariates, valid RDT subsample",
                axis_block_name="sample",
                axis_block={"spec_id": f"rc/sample/restriction/valid_rdt__{out_var}",
                           "axis": "restriction", "rule": "valid_rdt",
                           "n_obs_before": len(df_raw), "n_obs_after": len(df_rdt)}
            )
            print(f"  sample/valid_rdt__{out_var}: coef={r[1]:.4f}, p={r[3]:.4f}, N={r[4]}")


# ===== RC: FIXED EFFECTS =====
print(f"\n--- RC: FE variations ---")

# Add clinic FE (for primary outcome -- already done in strata_fe design variant)
# Drop date FE
for out_var in ['RXtreat_sev_simple_mal', 'treat_sev_simple_mal']:
    r = run_spec(
        f"rc/fe/drop/date__{out_var}",
        "modules/robustness/fixed_effects.md#dropping-fe-relative-to-baseline", "G1",
        out_var, "patient_voucher",
        manual_covariates, False, False, df_raw, canonical_vcov,
        "dropme!=1", "Manual covariates, no date FE",
        axis_block_name="fixed_effects",
        axis_block={"spec_id": f"rc/fe/drop/date__{out_var}", "family": "drop",
                   "added": [], "dropped": ["date"],
                   "baseline_fe": ["date"], "new_fe": []}
    )
    print(f"  fe/drop/date__{out_var}: coef={r[1]:.4f}, p={r[3]:.4f}, N={r[4]}")

# Add clinic FE
for out_var in ['RXtreat_sev_simple_mal', 'treat_sev_simple_mal']:
    r = run_spec(
        f"rc/fe/add/clinic__{out_var}",
        "modules/robustness/fixed_effects.md#additive-fe-variations-relative-to-baseline", "G1",
        out_var, "patient_voucher",
        manual_covariates, True, True, df_raw, canonical_vcov,
        "dropme!=1", "Manual covariates + date FE + clinic FE",
        axis_block_name="fixed_effects",
        axis_block={"spec_id": f"rc/fe/add/clinic__{out_var}", "family": "add",
                   "added": ["clinic"], "dropped": [],
                   "baseline_fe": ["date"], "new_fe": ["date", "clinic"]}
    )
    print(f"  fe/add/clinic__{out_var}: coef={r[1]:.4f}, p={r[3]:.4f}, N={r[4]}")


# ===== INFERENCE VARIANTS =====
print(f"\n--- Inference variants ---")

# HC1 (no clustering) for all baselines
baseline_run_ids = {}
for i, row in enumerate(results):
    if row['spec_id'] in ['baseline'] + list(additional_outcomes.values()):
        baseline_run_ids[row['outcome_var']] = row['spec_run_id']

for out_var, base_id in baseline_run_ids.items():
    run_inference_variant(
        base_id,
        "infer/se/hc/hc1",
        "modules/inference/standard_errors.md#heteroskedasticity-consistent-hc",
        "G1", out_var, "patient_voucher",
        manual_covariates, True, False, df_raw,
        "hetero", ""
    )
    print(f"  infer/hc1 for {out_var}: done")


# ===== WRITE OUTPUTS =====
print(f"\n{'=' * 60}")
print(f"Writing outputs...")

spec_df = pd.DataFrame(results)
spec_df.to_csv(f"{DATA_DIR}/specification_results.csv", index=False)
print(f"  specification_results.csv: {len(results)} rows")

if inference_results:
    infer_df = pd.DataFrame(inference_results)
    infer_df.to_csv(f"{DATA_DIR}/inference_results.csv", index=False)
    print(f"  inference_results.csv: {len(inference_results)} rows")

# Count successes/failures
n_success = sum(1 for r in results if r['run_success'] == 1)
n_fail = sum(1 for r in results if r['run_success'] == 0)
print(f"\n  Total: {len(results)} specs, {n_success} success, {n_fail} failed")

# Summary of baseline results
print(f"\n--- Baseline Summary ---")
for r in results:
    if r['spec_id'].startswith('baseline'):
        print(f"  {r['spec_id']} ({r['outcome_var']}): coef={r['coefficient']:.4f}, "
              f"se={r['std_error']:.4f}, p={r['p_value']:.4f}, N={r['n_obs']}")

print(f"\nDone.")
