"""
Specification Search Script for DellaVigna, Enikolopov, Mironova, Petrova, and Zhuravskaya (2014)
"Cross-Border Media and Nationalism: Evidence from Serbian Radio in Croatia"
American Economic Journal: Applied Economics, 6(3), 103-132.

Paper ID: 113893-V1

Surface-driven execution:
  - G1: Nazi_share ~ radio1 + controls_long [aweight=people_listed], cluster(Opsina2)
    OLS reduced-form and reduced-form signal strength; 2SLS with s1_1 as instrument.
  - G2: graffiti ~ radio1 + controls_long, cluster(Opsina2)
    Binary outcome (LPM), same control structure.

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

PAPER_ID = "113893-V1"
DATA_DIR = "data/downloads/extracted/113893-V1"
OUTPUT_DIR = DATA_DIR
DATA_PATH = f"{DATA_DIR}/Replication-AEJ/Data_AEJ_Replication.dta"

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

bg1 = surface_obj["baseline_groups"][0]
bg2 = surface_obj["baseline_groups"][1]
design_audit_g1 = bg1["design_audit"]
design_audit_g2 = bg2["design_audit"]
inference_canonical = bg1["inference_plan"]["canonical"]

# ============================================================
# Data Loading and Preparation
# ============================================================

df_raw = pd.read_stata(DATA_PATH)
print(f"Loaded data: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

# Convert float32 to float64 for precision
for col in df_raw.columns:
    if df_raw[col].dtype == np.float32:
        df_raw[col] = df_raw[col].astype(np.float64)

# Variable name mapping: the do-file uses Stata abbreviations
# mon -> monument
# name_of_the_streets_c -> name_of_the_streets_cyrillic
# name_of_the_streets_i -> name_of_the_streets_in_hungarian
# pivo_S -> pivo_Serb
# bliz -> bliz_forest
VARNAME_MAP = {
    "mon": "monument",
    "name_of_the_streets_c": "name_of_the_streets_cyrillic",
    "name_of_the_streets_i": "name_of_the_streets_in_hungarian",
    "pivo_S": "pivo_Serb",
    "bliz": "bliz_forest",
}

# Define control variable lists using ACTUAL names in the data
REGION_DUMMIES = ["r1", "r2", "r3", "r4", "r5"]

CONTROLS_SHORT = [
    "log_distance_full", "logpop", "male_share", "z2", "z3", "z6",
    "Croats", "higher_educ", "ec_active", "disable_share2",
] + REGION_DUMMIES

CONTROLS_LONG_EXTRA = [
    "war", "monument", "name_of_the_streets_cyrillic",
    "name_of_the_streets_in_hungarian", "pivo_Serb", "bliz_forest",
]

CONTROLS_LONG = CONTROLS_SHORT + CONTROLS_LONG_EXTRA

CONTROLS_GEO_ONLY = ["log_distance_full"] + REGION_DUMMIES

# Census controls (excluding geography and region dummies)
CONTROLS_CENSUS = [
    "logpop", "male_share", "z2", "z3", "z6",
    "Croats", "higher_educ", "ec_active", "disable_share2",
]

# War/manual controls
CONTROLS_MANUAL = [
    "war", "monument", "name_of_the_streets_cyrillic",
    "name_of_the_streets_in_hungarian", "pivo_Serb",
]

# Build samples
# G1 sample: radio1 non-missing AND Nazi_share non-missing
df_g1 = df_raw.dropna(subset=["radio1", "Nazi_share"]).copy()
print(f"G1 sample (radio1 & Nazi_share non-missing): {len(df_g1)} rows")

# G2 sample: radio1 non-missing AND graffiti non-missing
df_g2 = df_raw.dropna(subset=["radio1", "graffiti"]).copy()
print(f"G2 sample (radio1 & graffiti non-missing): {len(df_g2)} rows")

# Create signal strength dummies for instrument variants
# Do-file: centile s1 if radio1!=., centile(20(20)80)
# Then s_dum_x = (s1 >= centile_x)
quantiles_s1 = df_g1["s1"].quantile([0.2, 0.4, 0.6, 0.8]).values
for i in range(4):
    df_g1[f"s_dum_{i+1}"] = (df_g1["s1"] >= quantiles_s1[i]).astype(float)
    df_g2[f"s_dum_{i+1}"] = (df_g2["s1"] >= quantiles_s1[i]).astype(float)

print(f"Opsina2 unique in G1: {df_g1['Opsina2'].nunique()}")

# ============================================================
# Accumulators
# ============================================================

results = []
inference_results = []
spec_run_counter = 0


# ============================================================
# Helper: run_spec (OLS / reduced-form with pyfixest)
# ============================================================

def run_spec(spec_id, spec_tree_path, baseline_group_id,
             outcome_var, treatment_var, controls, data,
             vcov, sample_desc, controls_desc,
             cluster_var="Opsina2", weights_var="people_listed",
             fe_formula_str="", fe_desc="none",
             design_audit=None,
             axis_block_name=None, axis_block=None, notes=""):
    """Run a single OLS/RF specification and record results."""
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    if design_audit is None:
        design_audit = design_audit_g1

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

        kwargs = dict(data=data, vcov=vcov)
        if weights_var:
            kwargs["weights"] = weights_var

        m = pf.feols(formula, **kwargs)

        coef_val = float(m.coef().get(treatment_var, np.nan))
        se_val = float(m.se().get(treatment_var, np.nan))
        pval = float(m.pvalue().get(treatment_var, np.nan))

        try:
            ci = m.confint()
            ci_lower = float(ci.loc[treatment_var, ci.columns[0]]) if treatment_var in ci.index else np.nan
            ci_upper = float(ci.loc[treatment_var, ci.columns[1]]) if treatment_var in ci.index else np.nan
        except Exception:
            ci_lower = np.nan
            ci_upper = np.nan

        nobs = int(m._N)
        try:
            r2 = float(m._r2)
        except Exception:
            r2 = np.nan

        all_coefs = {k: float(v) for k, v in m.coef().items()}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": inference_canonical["spec_id"],
                       "method": "cluster", "cluster_vars": ["Opsina2"]},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"instrumental_variables": design_audit},
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
# Helper: run_iv (2SLS via pyfixest)
# ============================================================

def run_iv(spec_id, spec_tree_path, baseline_group_id,
           outcome_var, endogenous_var, instrument_str, controls, data,
           vcov, sample_desc, controls_desc,
           cluster_var="Opsina2", weights_var="people_listed",
           fe_formula_str="", fe_desc="none",
           axis_block_name=None, axis_block=None, notes=""):
    """Run a 2SLS IV specification.
    pyfixest IV syntax: Y ~ exog | FE | endog ~ instrument
    With no FE: Y ~ exog | endog ~ instrument
    """
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        controls_str = " + ".join(controls) if controls else "1"
        if fe_formula_str:
            formula = f"{outcome_var} ~ {controls_str} | {fe_formula_str} | {endogenous_var} ~ {instrument_str}"
        else:
            formula = f"{outcome_var} ~ {controls_str} | {endogenous_var} ~ {instrument_str}"

        kwargs = dict(data=data, vcov=vcov)
        if weights_var:
            kwargs["weights"] = weights_var

        m = pf.feols(formula, **kwargs)

        coef_val = float(m.coef().get(endogenous_var, np.nan))
        se_val = float(m.se().get(endogenous_var, np.nan))
        pval = float(m.pvalue().get(endogenous_var, np.nan))

        try:
            ci = m.confint()
            ci_lower = float(ci.loc[endogenous_var, ci.columns[0]]) if endogenous_var in ci.index else np.nan
            ci_upper = float(ci.loc[endogenous_var, ci.columns[1]]) if endogenous_var in ci.index else np.nan
        except Exception:
            ci_lower = np.nan
            ci_upper = np.nan

        nobs = int(m._N)
        try:
            r2 = float(m._r2)
        except Exception:
            r2 = np.nan

        all_coefs = {k: float(v) for k, v in m.coef().items()}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": inference_canonical["spec_id"],
                       "method": "cluster", "cluster_vars": ["Opsina2"],
                       "estimator": "2sls"},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"instrumental_variables": design_audit_g1},
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
            "treatment_var": endogenous_var,
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
        err_details = error_details_from_exception(e, stage="iv_estimation")
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
            "treatment_var": endogenous_var,
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


# ############################################################
# G1: NAZI_SHARE ~ RADIO1 + CONTROLS
# ############################################################

print("\n" + "=" * 60)
print("G1: Nazi_share (HSP vote share) outcome")
print("=" * 60)

# ============================================================
# BASELINE: Table 3, OLS radio1 with long controls (weighted, clustered)
# ============================================================

print("\nRunning G1 baseline specifications...")

base_run_id, base_coef, base_se, base_pval, base_nobs = run_spec(
    "baseline__table3_ols_long_controls",
    "designs/instrumental_variables.md#baseline", "G1",
    "Nazi_share", "radio1", CONTROLS_LONG, df_g1,
    {"CRV1": "Opsina2"},
    f"radio1 sample, N={len(df_g1)}", "controls_long (21 vars)",
    weights_var="people_listed")

print(f"  Baseline OLS: coef={base_coef:.4f}, se={base_se:.4f}, p={base_pval:.4f}, N={base_nobs}")

# ============================================================
# BASELINE VARIANTS FROM SURFACE
# ============================================================

# baseline__table3_ols_short_controls
run_spec(
    "baseline__table3_ols_short_controls",
    "designs/instrumental_variables.md#baseline", "G1",
    "Nazi_share", "radio1", CONTROLS_SHORT, df_g1,
    {"CRV1": "Opsina2"},
    "radio1 sample", "controls_short (15 vars)",
    weights_var="people_listed",
    axis_block_name="controls",
    axis_block={"spec_id": "baseline__table3_ols_short_controls", "family": "baseline",
                "set_name": "controls_short"})

# baseline__table3_rf_short_controls (signal strength reduced form)
run_spec(
    "baseline__table3_rf_short_controls",
    "designs/instrumental_variables.md#baseline", "G1",
    "Nazi_share", "s1_1", CONTROLS_SHORT, df_g1,
    {"CRV1": "Opsina2"},
    "radio1 sample", "controls_short (15 vars)",
    weights_var="people_listed",
    axis_block_name="controls",
    axis_block={"spec_id": "baseline__table3_rf_short_controls", "family": "baseline",
                "treatment": "s1_1 (signal strength RF)"})

# baseline__table3_rf_geography_only
run_spec(
    "baseline__table3_rf_geography_only",
    "designs/instrumental_variables.md#baseline", "G1",
    "Nazi_share", "s1_1", CONTROLS_GEO_ONLY, df_g1,
    {"CRV1": "Opsina2"},
    "radio1 sample", "geography only (log_distance + region dummies)",
    weights_var="people_listed",
    axis_block_name="controls",
    axis_block={"spec_id": "baseline__table3_rf_geography_only", "family": "baseline",
                "treatment": "s1_1 (signal strength RF)", "set_name": "geography_only"})

# Baseline reduced form with long controls
run_spec(
    "baseline__table3_rf_long_controls",
    "designs/instrumental_variables.md#baseline", "G1",
    "Nazi_share", "s1_1", CONTROLS_LONG, df_g1,
    {"CRV1": "Opsina2"},
    "radio1 sample", "controls_long (21 vars)",
    weights_var="people_listed",
    axis_block_name="controls",
    axis_block={"spec_id": "baseline__table3_rf_long_controls", "family": "baseline",
                "treatment": "s1_1 (signal strength RF)", "set_name": "controls_long"})


# ============================================================
# DESIGN: 2SLS IV specifications
# ============================================================

print("\nRunning design variants (2SLS)...")

# 2SLS with s1_1 as instrument for radio1, long controls
run_iv(
    "design/instrumental_variables/estimator/2sls",
    "designs/instrumental_variables.md#2sls", "G1",
    "Nazi_share", "radio1", "s1_1", CONTROLS_LONG, df_g1,
    {"CRV1": "Opsina2"},
    "radio1 sample", "controls_long (21 vars), 2SLS IV=s1_1",
    weights_var="people_listed",
    axis_block_name="estimation",
    axis_block={"spec_id": "design/instrumental_variables/estimator/2sls",
                "estimator": "2sls", "instrument": "s1_1"})

# 2SLS with short controls
run_iv(
    "design/instrumental_variables/estimator/2sls_short",
    "designs/instrumental_variables.md#2sls", "G1",
    "Nazi_share", "radio1", "s1_1", CONTROLS_SHORT, df_g1,
    {"CRV1": "Opsina2"},
    "radio1 sample", "controls_short (15 vars), 2SLS IV=s1_1",
    weights_var="people_listed",
    axis_block_name="estimation",
    axis_block={"spec_id": "design/instrumental_variables/estimator/2sls_short",
                "estimator": "2sls", "instrument": "s1_1", "controls": "short"})

# Continuous signal strength as treatment (alternative to binary radio1)
run_spec(
    "design/instrumental_variables/instrument/signal_strength_continuous",
    "designs/instrumental_variables.md#instrument", "G1",
    "Nazi_share", "s1", CONTROLS_LONG, df_g1,
    {"CRV1": "Opsina2"},
    "radio1 sample", "controls_long, signal strength s1 (continuous)",
    weights_var="people_listed",
    axis_block_name="estimation",
    axis_block={"spec_id": "design/instrumental_variables/instrument/signal_strength_continuous",
                "treatment": "s1 (raw signal strength)"})

# Signal strength dummies as instruments (overidentified 2SLS)
run_iv(
    "design/instrumental_variables/instrument/signal_dummies",
    "designs/instrumental_variables.md#instrument", "G1",
    "Nazi_share", "radio1", "s_dum_1 + s_dum_2 + s_dum_3 + s_dum_4",
    CONTROLS_LONG, df_g1,
    {"CRV1": "Opsina2"},
    "radio1 sample", "controls_long, 2SLS IV=signal quintile dummies",
    weights_var="people_listed",
    axis_block_name="estimation",
    axis_block={"spec_id": "design/instrumental_variables/instrument/signal_dummies",
                "estimator": "2sls", "instrument": "signal_strength_dummies"})

# radio1 and radio2 together (both dummy indicators of signal availability)
run_spec(
    "design/instrumental_variables/instrument/radio1_and_radio2",
    "designs/instrumental_variables.md#instrument", "G1",
    "Nazi_share", "radio1", ["radio2"] + CONTROLS_LONG, df_g1,
    {"CRV1": "Opsina2"},
    "radio1 sample", "controls_long + radio2",
    weights_var="people_listed",
    axis_block_name="estimation",
    axis_block={"spec_id": "design/instrumental_variables/instrument/radio1_and_radio2",
                "notes": "Include both radio1 and radio2 dummies simultaneously"})


# ============================================================
# RC: CONTROLS LOO (leave-one-out from long controls)
# ============================================================

print("\nRunning controls LOO variants...")

# LOO for the extra controls in long set (war, monument, streets, pivo, bliz)
LOO_MAP = {
    "rc/controls/loo/drop_war": ["war"],
    "rc/controls/loo/drop_mon": ["monument"],
    "rc/controls/loo/drop_name_streets_c": ["name_of_the_streets_cyrillic"],
    "rc/controls/loo/drop_name_streets_i": ["name_of_the_streets_in_hungarian"],
    "rc/controls/loo/drop_pivo_S": ["pivo_Serb"],
    "rc/controls/loo/drop_bliz": ["bliz_forest"],
    "rc/controls/loo/drop_log_distance_full": ["log_distance_full"],
    "rc/controls/loo/drop_higher_educ": ["higher_educ"],
    "rc/controls/loo/drop_ec_active": ["ec_active"],
    "rc/controls/loo/drop_disable_share2": ["disable_share2"],
}

for spec_id, drop_vars in LOO_MAP.items():
    ctrl = [c for c in CONTROLS_LONG if c not in drop_vars]
    run_spec(
        spec_id, "modules/robustness/controls.md#leave-one-out-controls-loo", "G1",
        "Nazi_share", "radio1", ctrl, df_g1,
        {"CRV1": "Opsina2"},
        "radio1 sample", f"controls_long minus {', '.join(drop_vars)}",
        weights_var="people_listed",
        axis_block_name="controls",
        axis_block={"spec_id": spec_id, "family": "loo",
                    "dropped": drop_vars, "n_controls": len(ctrl)})


# ============================================================
# RC: CONTROL SETS
# ============================================================

print("\nRunning control set variants...")

# Geography only
run_spec(
    "rc/controls/sets/geography_only",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "Nazi_share", "radio1", CONTROLS_GEO_ONLY, df_g1,
    {"CRV1": "Opsina2"},
    "radio1 sample", "geography only (log_distance + region)",
    weights_var="people_listed",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/geography_only", "family": "sets",
                "n_controls": len(CONTROLS_GEO_ONLY), "set_name": "geography_only"})

# Controls short
run_spec(
    "rc/controls/sets/controls_short",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "Nazi_share", "radio1", CONTROLS_SHORT, df_g1,
    {"CRV1": "Opsina2"},
    "radio1 sample", "controls_short (census + geography + region, 15 vars)",
    weights_var="people_listed",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/controls_short", "family": "sets",
                "n_controls": len(CONTROLS_SHORT), "set_name": "controls_short"})

# Controls long (same as baseline, for completeness)
run_spec(
    "rc/controls/sets/controls_long",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "Nazi_share", "radio1", CONTROLS_LONG, df_g1,
    {"CRV1": "Opsina2"},
    "radio1 sample", "controls_long (21 vars, same as baseline)",
    weights_var="people_listed",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/controls_long", "family": "sets",
                "n_controls": len(CONTROLS_LONG), "set_name": "controls_long"})

# No controls (bivariate)
run_spec(
    "rc/controls/sets/none",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "Nazi_share", "radio1", [], df_g1,
    {"CRV1": "Opsina2"},
    "radio1 sample", "none (bivariate)",
    weights_var="people_listed",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/none", "family": "sets",
                "n_controls": 0, "set_name": "none"})


# ============================================================
# RC: ADD ADDITIONAL CONTROLS (Table 6 robustness variables)
# ============================================================

print("\nRunning add-control variants (Table 6)...")

# Add radio_hung (Hungarian radio control)
run_spec(
    "rc/controls/add/radio_hung",
    "modules/robustness/controls.md#additional-controls", "G1",
    "Nazi_share", "radio1", CONTROLS_LONG + ["radio_hung"], df_g1,
    {"CRV1": "Opsina2"},
    "radio1 sample", "controls_long + radio_hung (Hungarian radio)",
    weights_var="people_listed",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/add/radio_hung", "family": "add",
                "added": ["radio_hung"]})

# Add eloss5050powerHKR (Croatian radio signal strength)
run_spec(
    "rc/controls/add/eloss5050powerHKR",
    "modules/robustness/controls.md#additional-controls", "G1",
    "Nazi_share", "radio1", CONTROLS_LONG + ["eloss5050powerHKR", "eloss5050powerHR1"], df_g1,
    {"CRV1": "Opsina2"},
    "radio1 sample", "controls_long + Croatian radio signal",
    weights_var="people_listed",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/add/eloss5050powerHKR", "family": "add",
                "added": ["eloss5050powerHKR", "eloss5050powerHR1"]})

# Add floss_1 (free-space loss)
run_spec(
    "rc/controls/add/floss_1",
    "modules/robustness/controls.md#additional-controls", "G1",
    "Nazi_share", "radio1", CONTROLS_LONG + ["floss_1"], df_g1,
    {"CRV1": "Opsina2"},
    "radio1 sample", "controls_long + floss_1 (free-space loss)",
    weights_var="people_listed",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/add/floss_1", "family": "add",
                "added": ["floss_1"]})


# ============================================================
# RC: RANDOM CONTROL SUBSETS
# ============================================================

print("\nRunning random control subset variants...")

rng = np.random.RandomState(113893)
# Pool = non-region controls from long set (region dummies always included)
subset_pool = [c for c in CONTROLS_LONG if c not in REGION_DUMMIES]

for draw_i in range(1, 16):
    k = rng.randint(3, len(subset_pool) + 1)
    chosen = list(rng.choice(subset_pool, size=k, replace=False))
    # Always include region dummies
    full_chosen = chosen + REGION_DUMMIES
    excluded = [v for v in subset_pool if v not in chosen]
    spec_id = f"rc/controls/subset/random_{draw_i:03d}"
    run_spec(
        spec_id, "modules/robustness/controls.md#subset-generation-specids", "G1",
        "Nazi_share", "radio1", full_chosen, df_g1,
        {"CRV1": "Opsina2"},
        "radio1 sample", f"random subset draw {draw_i} ({len(full_chosen)} controls)",
        weights_var="people_listed",
        axis_block_name="controls",
        axis_block={"spec_id": spec_id, "family": "subset", "method": "random",
                    "seed": 113893, "draw_index": draw_i,
                    "included": full_chosen, "excluded": excluded,
                    "n_controls": len(full_chosen)})


# ============================================================
# RC: SAMPLE RESTRICTIONS
# ============================================================

print("\nRunning sample restriction variants...")

# Distance < 75km (Table 7 in paper)
# Note: the full radio1 sample is actually all within 75km (distance max ~40 in sample)
# But let's still include it in case there's variation
df_g1_lt75 = df_g1[df_g1["distance"] < 75].copy()
if len(df_g1_lt75) > 20:
    run_spec(
        "rc/sample/restriction/distance_lt_75km",
        "modules/robustness/sample.md#sample-restrictions", "G1",
        "Nazi_share", "radio1", CONTROLS_SHORT, df_g1_lt75,
        {"CRV1": "Opsina2"},
        f"distance < 75km, N={len(df_g1_lt75)}", "controls_short (Table 7 spec)",
        weights_var="people_listed",
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/restriction/distance_lt_75km",
                    "n_obs": len(df_g1_lt75)})

# Trim outcome at 1st/99th percentile
q01 = df_g1["Nazi_share"].quantile(0.01)
q99 = df_g1["Nazi_share"].quantile(0.99)
df_trim1 = df_g1[(df_g1["Nazi_share"] >= q01) & (df_g1["Nazi_share"] <= q99)].copy()

run_spec(
    "rc/sample/outliers/trim_y_1_99",
    "modules/robustness/sample.md#outliers-and-influential-observations", "G1",
    "Nazi_share", "radio1", CONTROLS_LONG, df_trim1,
    {"CRV1": "Opsina2"},
    f"trim Nazi_share [1%,99%], N={len(df_trim1)}", "controls_long",
    weights_var="people_listed",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_1_99", "axis": "outliers",
                "rule": "trim", "params": {"var": "Nazi_share", "lower_q": 0.01, "upper_q": 0.99},
                "n_obs_before": len(df_g1), "n_obs_after": len(df_trim1)})

# Trim outcome at 5th/95th percentile
q05 = df_g1["Nazi_share"].quantile(0.05)
q95 = df_g1["Nazi_share"].quantile(0.95)
df_trim5 = df_g1[(df_g1["Nazi_share"] >= q05) & (df_g1["Nazi_share"] <= q95)].copy()

run_spec(
    "rc/sample/outliers/trim_y_5_95",
    "modules/robustness/sample.md#outliers-and-influential-observations", "G1",
    "Nazi_share", "radio1", CONTROLS_LONG, df_trim5,
    {"CRV1": "Opsina2"},
    f"trim Nazi_share [5%,95%], N={len(df_trim5)}", "controls_long",
    weights_var="people_listed",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_5_95", "axis": "outliers",
                "rule": "trim", "params": {"var": "Nazi_share", "lower_q": 0.05, "upper_q": 0.95},
                "n_obs_before": len(df_g1), "n_obs_after": len(df_trim5)})


# ============================================================
# RC: CLUSTERING VARIANT (municipality is already baseline)
# ============================================================

print("\nRunning clustering variant...")

run_spec(
    "rc/fe/cluster/municipality",
    "modules/robustness/fixed_effects.md#clustering", "G1",
    "Nazi_share", "radio1", CONTROLS_LONG, df_g1,
    {"CRV1": "Opsina2"},
    "radio1 sample", "controls_long, cluster(Opsina2) [same as baseline]",
    weights_var="people_listed",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/cluster/municipality", "family": "cluster",
                "cluster_var": "Opsina2",
                "notes": "Same as baseline; included for completeness"})


# ============================================================
# RC: UNWEIGHTED
# ============================================================

print("\nRunning unweighted variant...")

run_spec(
    "rc/weights/unweighted",
    "modules/robustness/weights.md#unweighted", "G1",
    "Nazi_share", "radio1", CONTROLS_LONG, df_g1,
    {"CRV1": "Opsina2"},
    "radio1 sample (unweighted)", "controls_long",
    weights_var=None,
    axis_block_name="weights",
    axis_block={"spec_id": "rc/weights/unweighted", "family": "weights",
                "notes": "Unweighted OLS (baseline uses aweight=people_listed)"})


# ============================================================
# RC: REDUCED FORM VARIANTS (with LOO and control sets)
# ============================================================

print("\nRunning reduced-form signal strength variants...")

# RF with geography only
run_spec(
    "rc/rf/controls/geography_only",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "Nazi_share", "s1_1", CONTROLS_GEO_ONLY, df_g1,
    {"CRV1": "Opsina2"},
    "radio1 sample", "RF: s1_1, geography only",
    weights_var="people_listed",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/rf/controls/geography_only", "treatment": "s1_1",
                "set_name": "geography_only"})

# RF with no controls
run_spec(
    "rc/rf/controls/none",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "Nazi_share", "s1_1", [], df_g1,
    {"CRV1": "Opsina2"},
    "radio1 sample", "RF: s1_1, no controls",
    weights_var="people_listed",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/rf/controls/none", "treatment": "s1_1",
                "set_name": "none"})

# RF unweighted
run_spec(
    "rc/rf/weights/unweighted",
    "modules/robustness/weights.md#unweighted", "G1",
    "Nazi_share", "s1_1", CONTROLS_LONG, df_g1,
    {"CRV1": "Opsina2"},
    "radio1 sample (unweighted)", "RF: s1_1, controls_long, unweighted",
    weights_var=None,
    axis_block_name="weights",
    axis_block={"spec_id": "rc/rf/weights/unweighted", "treatment": "s1_1",
                "notes": "Unweighted reduced form"})

# RF with add robustness controls
run_spec(
    "rc/rf/add/radio_hung",
    "modules/robustness/controls.md#additional-controls", "G1",
    "Nazi_share", "s1_1", CONTROLS_LONG + ["hung1"], df_g1,
    {"CRV1": "Opsina2"},
    "radio1 sample", "RF: s1_1, controls_long + hung1 (Hungarian signal)",
    weights_var="people_listed",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/rf/add/radio_hung", "treatment": "s1_1",
                "added": ["hung1"]})

run_spec(
    "rc/rf/add/eloss5050powerHKR",
    "modules/robustness/controls.md#additional-controls", "G1",
    "Nazi_share", "s1_1", CONTROLS_LONG + ["eloss5050powerHKR", "eloss5050powerHR1"], df_g1,
    {"CRV1": "Opsina2"},
    "radio1 sample", "RF: s1_1, controls_long + Croatian radio signal",
    weights_var="people_listed",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/rf/add/eloss5050powerHKR", "treatment": "s1_1",
                "added": ["eloss5050powerHKR", "eloss5050powerHR1"]})

run_spec(
    "rc/rf/add/floss_1",
    "modules/robustness/controls.md#additional-controls", "G1",
    "Nazi_share", "s1_1", CONTROLS_LONG + ["floss_1"], df_g1,
    {"CRV1": "Opsina2"},
    "radio1 sample", "RF: s1_1, controls_long + floss_1",
    weights_var="people_listed",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/rf/add/floss_1", "treatment": "s1_1",
                "added": ["floss_1"]})


# ============================================================
# RC: ALTERNATIVE OUTCOMES (Table 4)
# ============================================================

print("\nRunning alternative outcome variants (Table 4)...")

for alt_outcome in ["hdz_share", "sdp_share", "turnout", "others_share"]:
    run_spec(
        f"rc/outcome/{alt_outcome}/radio1",
        "modules/robustness/outcomes.md#alternative-outcomes", "G1",
        alt_outcome, "radio1", CONTROLS_LONG, df_g1,
        {"CRV1": "Opsina2"},
        "radio1 sample", f"controls_long, outcome={alt_outcome}",
        weights_var="people_listed",
        axis_block_name="estimation",
        axis_block={"spec_id": f"rc/outcome/{alt_outcome}/radio1",
                    "notes": f"Alternative outcome {alt_outcome} (Table 4)"})

    run_spec(
        f"rc/outcome/{alt_outcome}/s1",
        "modules/robustness/outcomes.md#alternative-outcomes", "G1",
        alt_outcome, "s1", CONTROLS_LONG, df_g1,
        {"CRV1": "Opsina2"},
        "radio1 sample", f"controls_long, outcome={alt_outcome}, treatment=s1",
        weights_var="people_listed",
        axis_block_name="estimation",
        axis_block={"spec_id": f"rc/outcome/{alt_outcome}/s1",
                    "notes": f"Alternative outcome {alt_outcome}, signal strength (Table 4)"})


# ############################################################
# G2: GRAFFITI (binary outcome)
# ############################################################

print("\n" + "=" * 60)
print("G2: Graffiti (anti-Serb graffiti) outcome")
print("=" * 60)

# ============================================================
# BASELINE: LPM radio1 on graffiti, long controls
# ============================================================

print("\nRunning G2 baseline (LPM)...")

g2_base_run_id, g2_base_coef, g2_base_se, g2_base_pval, g2_base_nobs = run_spec(
    "baseline__table5_lpm_radio1_controls_long",
    "designs/instrumental_variables.md#baseline", "G2",
    "graffiti", "radio1", CONTROLS_LONG, df_g2,
    {"CRV1": "Opsina2"},
    f"graffiti sample, N={len(df_g2)}", "controls_long (21 vars), LPM",
    weights_var=None,  # Table 5 in do-file uses no weights (local cond="")
    design_audit=design_audit_g2)

print(f"  G2 Baseline LPM: coef={g2_base_coef:.4f}, se={g2_base_se:.4f}, p={g2_base_pval:.4f}, N={g2_base_nobs}")

# ============================================================
# G2: DESIGN VARIANTS
# ============================================================

# LPM with short controls
run_spec(
    "g2/design/lpm/controls_short",
    "designs/instrumental_variables.md#baseline", "G2",
    "graffiti", "radio1", CONTROLS_SHORT, df_g2,
    {"CRV1": "Opsina2"},
    "graffiti sample", "controls_short, LPM",
    weights_var=None, design_audit=design_audit_g2,
    axis_block_name="controls",
    axis_block={"spec_id": "g2/design/lpm/controls_short", "set_name": "controls_short"})

# LPM with geography only
run_spec(
    "g2/design/lpm/geography_only",
    "designs/instrumental_variables.md#baseline", "G2",
    "graffiti", "radio1", CONTROLS_GEO_ONLY, df_g2,
    {"CRV1": "Opsina2"},
    "graffiti sample", "geography only, LPM",
    weights_var=None, design_audit=design_audit_g2,
    axis_block_name="controls",
    axis_block={"spec_id": "g2/design/lpm/geography_only", "set_name": "geography_only"})

# Signal strength on graffiti (reduced form)
run_spec(
    "g2/design/rf/s1_controls_long",
    "designs/instrumental_variables.md#instrument", "G2",
    "graffiti", "s1", CONTROLS_LONG, df_g2,
    {"CRV1": "Opsina2"},
    "graffiti sample", "RF: s1, controls_long",
    weights_var=None, design_audit=design_audit_g2,
    axis_block_name="estimation",
    axis_block={"spec_id": "g2/design/rf/s1_controls_long", "treatment": "s1"})

# Signal strength on graffiti, short controls
run_spec(
    "g2/design/rf/s1_controls_short",
    "designs/instrumental_variables.md#instrument", "G2",
    "graffiti", "s1", CONTROLS_SHORT, df_g2,
    {"CRV1": "Opsina2"},
    "graffiti sample", "RF: s1, controls_short",
    weights_var=None, design_audit=design_audit_g2,
    axis_block_name="estimation",
    axis_block={"spec_id": "g2/design/rf/s1_controls_short", "treatment": "s1",
                "set_name": "controls_short"})

# Signal strength on graffiti, geography only
run_spec(
    "g2/design/rf/s1_geography_only",
    "designs/instrumental_variables.md#instrument", "G2",
    "graffiti", "s1", CONTROLS_GEO_ONLY, df_g2,
    {"CRV1": "Opsina2"},
    "graffiti sample", "RF: s1, geography only",
    weights_var=None, design_audit=design_audit_g2,
    axis_block_name="estimation",
    axis_block={"spec_id": "g2/design/rf/s1_geography_only", "treatment": "s1",
                "set_name": "geography_only"})

# radio1 + radio2 on graffiti
run_spec(
    "g2/design/radio1_and_radio2",
    "designs/instrumental_variables.md#instrument", "G2",
    "graffiti", "radio1", ["radio2"] + CONTROLS_LONG, df_g2,
    {"CRV1": "Opsina2"},
    "graffiti sample", "controls_long + radio2, LPM",
    weights_var=None, design_audit=design_audit_g2,
    axis_block_name="estimation",
    axis_block={"spec_id": "g2/design/radio1_and_radio2",
                "notes": "Both radio1 and radio2"})

# G2 unweighted (baseline is already unweighted for graffiti, try weighted for comparison)
run_spec(
    "g2/rc/weights/weighted",
    "modules/robustness/weights.md#weighted", "G2",
    "graffiti", "radio1", CONTROLS_LONG, df_g2,
    {"CRV1": "Opsina2"},
    "graffiti sample (weighted)", "controls_long, weighted by people_listed",
    weights_var="people_listed", design_audit=design_audit_g2,
    axis_block_name="weights",
    axis_block={"spec_id": "g2/rc/weights/weighted",
                "notes": "Weighted variant (baseline is unweighted)"})

# G2 no controls
run_spec(
    "g2/rc/controls/none",
    "modules/robustness/controls.md#standard-control-sets", "G2",
    "graffiti", "radio1", [], df_g2,
    {"CRV1": "Opsina2"},
    "graffiti sample", "no controls (bivariate)",
    weights_var=None, design_audit=design_audit_g2,
    axis_block_name="controls",
    axis_block={"spec_id": "g2/rc/controls/none", "set_name": "none"})


# ============================================================
# INFERENCE VARIANTS
# ============================================================

print("\nRunning inference variants...")

baseline_run_id_g1 = f"{PAPER_ID}_run_001"
infer_counter = 0


def run_inference_variant(base_run_id, spec_id, spec_tree_path, baseline_group_id,
                          formula_str, data, focal_var, vcov, vcov_desc,
                          weights_var="people_listed"):
    """Re-run baseline spec with different variance-covariance estimator."""
    global infer_counter
    infer_counter += 1
    infer_run_id = f"{PAPER_ID}_infer_{infer_counter:03d}"

    try:
        kwargs = dict(data=data, vcov=vcov)
        if weights_var:
            kwargs["weights"] = weights_var

        m = pf.feols(formula_str, **kwargs)

        coef_val = float(m.coef().get(focal_var, np.nan))
        se_val = float(m.se().get(focal_var, np.nan))
        pval = float(m.pvalue().get(focal_var, np.nan))

        try:
            ci = m.confint()
            ci_lower = float(ci.loc[focal_var, ci.columns[0]]) if focal_var in ci.index else np.nan
            ci_upper = float(ci.loc[focal_var, ci.columns[1]]) if focal_var in ci.index else np.nan
        except Exception:
            ci_lower = np.nan
            ci_upper = np.nan

        nobs = int(m._N)
        try:
            r2 = float(m._r2)
        except Exception:
            r2 = np.nan

        all_coefs = {k: float(v) for k, v in m.coef().items()}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": spec_id, "method": vcov_desc},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"instrumental_variables": design_audit_g1},
        )

        inference_results.append({
            "paper_id": PAPER_ID,
            "inference_run_id": infer_run_id,
            "spec_run_id": base_run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "coefficient": coef_val,
            "std_error": se_val,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": nobs,
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "cluster_var": vcov_desc,
            "run_success": 1,
            "run_error": ""
        })

    except Exception as e:
        err_msg = str(e)[:240]
        err_details = error_details_from_exception(e, stage="inference")
        payload = make_failure_payload(
            error=err_msg,
            error_details=err_details,
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH
        )
        inference_results.append({
            "paper_id": PAPER_ID,
            "inference_run_id": infer_run_id,
            "spec_run_id": base_run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "coefficient": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": np.nan,
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "cluster_var": vcov_desc,
            "run_success": 0,
            "run_error": err_msg
        })


# G1 baseline formula
ctrl_long_str = " + ".join(CONTROLS_LONG)
g1_baseline_formula = f"Nazi_share ~ radio1 + {ctrl_long_str}"

# HC1 robust (no clustering)
run_inference_variant(
    baseline_run_id_g1, "infer/se/hc/hc1",
    "modules/inference/standard_errors.md#heteroskedasticity-robust", "G1",
    g1_baseline_formula, df_g1, "radio1",
    "hetero", "HC1 (robust, no clustering)",
    weights_var="people_listed")

# G2 inference: HC1
g2_baseline_formula = f"graffiti ~ radio1 + {ctrl_long_str}"
g2_base_run_id_actual = [r["spec_run_id"] for r in results
                          if r["spec_id"] == "baseline__table5_lpm_radio1_controls_long"][0]

run_inference_variant(
    g2_base_run_id_actual, "infer/g2/se/hc/hc1",
    "modules/inference/standard_errors.md#heteroskedasticity-robust", "G2",
    g2_baseline_formula, df_g2, "radio1",
    "hetero", "HC1 (robust, no clustering)",
    weights_var=None)


# ============================================================
# FIRST-STAGE DIAGNOSTIC
# ============================================================

print("\nRunning first-stage diagnostic...")

# First stage: radio1 ~ s1_1 + controls_long [aweight=people_listed], cluster(Opsina2)
run_spec(
    "diag/instrumental_variables/first_stage/f_statistic",
    "modules/diagnostics/instrumental_variables.md#first-stage", "G1",
    "radio1", "s1_1", CONTROLS_LONG, df_g1,
    {"CRV1": "Opsina2"},
    "radio1 sample", "First stage: s1_1 -> radio1, controls_long",
    weights_var="people_listed",
    axis_block_name="estimation",
    axis_block={"spec_id": "diag/instrumental_variables/first_stage/f_statistic",
                "notes": "First-stage regression for signal strength predicting radio availability"})


# ============================================================
# WRITE OUTPUTS
# ============================================================

print(f"\nWriting outputs...")
print(f"  Specification specs: {len(results)}")
print(f"  Inference variants: {len(inference_results)}")

# specification_results.csv
spec_df = pd.DataFrame(results)
spec_df.to_csv(f"{OUTPUT_DIR}/specification_results.csv", index=False)

# inference_results.csv
infer_df = pd.DataFrame(inference_results)
infer_df.to_csv(f"{OUTPUT_DIR}/inference_results.csv", index=False)

# Summary stats
successful = spec_df[spec_df['run_success'] == 1]
failed = spec_df[spec_df['run_success'] == 0]

print("\n=== SPECIFICATION RESULTS SUMMARY ===")
print(f"Total rows: {len(spec_df)}")
print(f"Successful: {len(successful)}")
print(f"Failed: {len(failed)}")

if len(successful) > 0:
    base_row = spec_df[spec_df['spec_id'] == 'baseline__table3_ols_long_controls']
    if len(base_row) > 0:
        print(f"\nBaseline coef on radio1: {base_row['coefficient'].values[0]:.6f}")
        print(f"Baseline SE: {base_row['std_error'].values[0]:.6f}")
        print(f"Baseline p-value: {base_row['p_value'].values[0]:.6f}")
        print(f"Baseline N: {base_row['n_obs'].values[0]:.0f}")

    # G1 specs only (radio1 on Nazi_share)
    g1_specs = successful[successful['baseline_group_id'] == 'G1']
    g1_radio1 = g1_specs[g1_specs['treatment_var'] == 'radio1']
    if len(g1_radio1) > 0:
        print(f"\n=== G1 radio1 COEFFICIENT RANGE ===")
        print(f"Min coef: {g1_radio1['coefficient'].min():.6f}")
        print(f"Max coef: {g1_radio1['coefficient'].max():.6f}")
        print(f"Median coef: {g1_radio1['coefficient'].median():.6f}")
        n_sig = (g1_radio1['p_value'] < 0.05).sum()
        print(f"Significant at 5%: {n_sig}/{len(g1_radio1)}")

    print(f"\n=== ALL SPECS COEFFICIENT RANGE ===")
    print(f"Min coef: {successful['coefficient'].min():.6f}")
    print(f"Max coef: {successful['coefficient'].max():.6f}")
    print(f"Median coef: {successful['coefficient'].median():.6f}")


# ============================================================
# SPECIFICATION_SEARCH.md
# ============================================================

print("\nWriting SPECIFICATION_SEARCH.md...")

md_lines = []
md_lines.append("# Specification Search Report: 113893-V1")
md_lines.append("")
md_lines.append("**Paper:** DellaVigna, Enikolopov, Mironova, Petrova, and Zhuravskaya (2014), \"Cross-Border Media and Nationalism: Evidence from Serbian Radio in Croatia\", AEJ: Applied 6(3)")
md_lines.append("")
md_lines.append("## Baseline Specification (G1)")
md_lines.append("")
md_lines.append("- **Design:** IV / reduced-form OLS (cross-sectional)")
md_lines.append("- **Outcome:** Nazi_share (HSP nationalist party vote share)")
md_lines.append("- **Treatment:** radio1 (binary Serbian radio availability)")
md_lines.append("- **Instrument:** s1_1 (signal strength, for 2SLS)")
md_lines.append(f"- **Controls:** {len(CONTROLS_LONG)} controls (census, geographic, war/manual, region dummies)")
md_lines.append("- **Weights:** people_listed (population)")
md_lines.append("- **Clustering:** Opsina2 (municipality)")
md_lines.append("")

base_row = spec_df[spec_df['spec_id'] == 'baseline__table3_ols_long_controls']
if len(base_row) > 0:
    bc = base_row.iloc[0]
    md_lines.append("| Statistic | Value |")
    md_lines.append("|-----------|-------|")
    md_lines.append(f"| Coefficient | {bc['coefficient']:.6f} |")
    md_lines.append(f"| Std. Error | {bc['std_error']:.6f} |")
    md_lines.append(f"| p-value | {bc['p_value']:.6f} |")
    md_lines.append(f"| 95% CI | [{bc['ci_lower']:.6f}, {bc['ci_upper']:.6f}] |")
    md_lines.append(f"| N | {bc['n_obs']:.0f} |")
    md_lines.append(f"| R-squared | {bc['r_squared']:.4f} |")
    md_lines.append("")

md_lines.append("## Baseline Specification (G2)")
md_lines.append("")
md_lines.append("- **Design:** LPM (cross-sectional)")
md_lines.append("- **Outcome:** graffiti (anti-Serb graffiti, binary)")
md_lines.append("- **Treatment:** radio1 (binary Serbian radio availability)")
md_lines.append("- **Weights:** unweighted")
md_lines.append("- **Clustering:** Opsina2 (municipality)")
md_lines.append("")

g2_base_row = spec_df[spec_df['spec_id'] == 'baseline__table5_lpm_radio1_controls_long']
if len(g2_base_row) > 0:
    g2c = g2_base_row.iloc[0]
    md_lines.append("| Statistic | Value |")
    md_lines.append("|-----------|-------|")
    md_lines.append(f"| Coefficient | {g2c['coefficient']:.6f} |")
    md_lines.append(f"| Std. Error | {g2c['std_error']:.6f} |")
    md_lines.append(f"| p-value | {g2c['p_value']:.6f} |")
    md_lines.append(f"| 95% CI | [{g2c['ci_lower']:.6f}, {g2c['ci_upper']:.6f}] |")
    md_lines.append(f"| N | {g2c['n_obs']:.0f} |")
    md_lines.append(f"| R-squared | {g2c['r_squared']:.4f} |")
    md_lines.append("")

md_lines.append("## Specification Counts")
md_lines.append("")
md_lines.append(f"- Total specifications: {len(spec_df)}")
md_lines.append(f"- Successful: {len(successful)}")
md_lines.append(f"- Failed: {len(failed)}")
md_lines.append(f"- Inference variants: {len(infer_df)}")
md_lines.append("")

# Category breakdown
md_lines.append("## Category Breakdown (G1: Nazi_share)")
md_lines.append("")
md_lines.append("| Category | Count | Sig. (p<0.05) | Coef Range |")
md_lines.append("|----------|-------|---------------|------------|")

g1_successful = successful[successful['baseline_group_id'] == 'G1']

categories_g1 = {
    "Baselines": g1_successful[g1_successful['spec_id'].str.startswith('baseline')],
    "Design (2SLS/IV)": g1_successful[g1_successful['spec_id'].str.startswith('design/')],
    "Controls LOO": g1_successful[g1_successful['spec_id'].str.startswith('rc/controls/loo/')],
    "Controls Sets": g1_successful[g1_successful['spec_id'].str.startswith('rc/controls/sets/')],
    "Controls Add": g1_successful[g1_successful['spec_id'].str.startswith('rc/controls/add/')],
    "Controls Subset": g1_successful[g1_successful['spec_id'].str.startswith('rc/controls/subset/')],
    "Sample Restrictions": g1_successful[g1_successful['spec_id'].str.startswith('rc/sample/')],
    "Weights": g1_successful[g1_successful['spec_id'].str.startswith('rc/weights/')],
    "RF Variants": g1_successful[g1_successful['spec_id'].str.startswith('rc/rf/')],
    "Alt. Outcomes": g1_successful[g1_successful['spec_id'].str.startswith('rc/outcome/')],
    "First Stage": g1_successful[g1_successful['spec_id'].str.startswith('diag/')],
}

for cat_name, cat_df in categories_g1.items():
    if len(cat_df) > 0:
        n_sig_cat = (cat_df['p_value'] < 0.05).sum()
        coef_range = f"[{cat_df['coefficient'].min():.4f}, {cat_df['coefficient'].max():.4f}]"
        md_lines.append(f"| {cat_name} | {len(cat_df)} | {n_sig_cat}/{len(cat_df)} | {coef_range} |")

md_lines.append("")

md_lines.append("## Category Breakdown (G2: Graffiti)")
md_lines.append("")
md_lines.append("| Category | Count | Sig. (p<0.05) | Coef Range |")
md_lines.append("|----------|-------|---------------|------------|")

g2_successful = successful[successful['baseline_group_id'] == 'G2']
if len(g2_successful) > 0:
    n_sig_g2 = (g2_successful['p_value'] < 0.05).sum()
    coef_range_g2 = f"[{g2_successful['coefficient'].min():.4f}, {g2_successful['coefficient'].max():.4f}]"
    md_lines.append(f"| All G2 | {len(g2_successful)} | {n_sig_g2}/{len(g2_successful)} | {coef_range_g2} |")

md_lines.append("")

# Inference variants
md_lines.append("## Inference Variants")
md_lines.append("")
if len(infer_df) > 0:
    md_lines.append("| Spec ID | SE | p-value | 95% CI |")
    md_lines.append("|---------|-----|---------|--------|")
    for _, row in infer_df.iterrows():
        if row['run_success'] == 1:
            md_lines.append(f"| {row['spec_id']} | {row['std_error']:.6f} | {row['p_value']:.6f} | [{row['ci_lower']:.6f}, {row['ci_upper']:.6f}] |")
        else:
            md_lines.append(f"| {row['spec_id']} | FAILED | - | - |")

md_lines.append("")

# Overall assessment
md_lines.append("## Overall Assessment")
md_lines.append("")

# Focus on G1 radio1 specs with Nazi_share outcome for robustness assessment
# Exclude alternative outcomes (rc/outcome/*) and first-stage diagnostics from robustness count
g1_radio1_core = g1_successful[
    (g1_successful['treatment_var'] == 'radio1') &
    (g1_successful['outcome_var'] == 'Nazi_share') &
    (~g1_successful['spec_id'].str.startswith('diag/'))
]
if len(g1_radio1_core) > 0:
    n_sig_total = (g1_radio1_core['p_value'] < 0.05).sum()
    pct_sig = n_sig_total / len(g1_radio1_core) * 100
    sign_consistent = ((g1_radio1_core['coefficient'] > 0).sum() == len(g1_radio1_core)) or \
                      ((g1_radio1_core['coefficient'] < 0).sum() == len(g1_radio1_core))
    median_coef = g1_radio1_core['coefficient'].median()
    sign_word = "positive" if median_coef > 0 else "negative"

    md_lines.append(f"### G1: Nazi_share ~ radio1 (core robustness, N={len(g1_radio1_core)} specs)")
    md_lines.append(f"- **Sign consistency:** {'All specifications have the same sign' if sign_consistent else 'Mixed signs across specifications'}")
    md_lines.append(f"- **Significance stability:** {n_sig_total}/{len(g1_radio1_core)} ({pct_sig:.1f}%) specifications significant at 5%")
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
    md_lines.append("")

if len(g2_successful) > 0:
    g2_radio1 = g2_successful[g2_successful['treatment_var'] == 'radio1']
    if len(g2_radio1) > 0:
        n_sig_g2_r1 = (g2_radio1['p_value'] < 0.05).sum()
        pct_sig_g2 = n_sig_g2_r1 / len(g2_radio1) * 100
        sign_consistent_g2 = ((g2_radio1['coefficient'] > 0).sum() == len(g2_radio1)) or \
                              ((g2_radio1['coefficient'] < 0).sum() == len(g2_radio1))
        median_g2 = g2_radio1['coefficient'].median()
        sign_word_g2 = "positive" if median_g2 > 0 else "negative"

        md_lines.append(f"### G2: Graffiti ~ radio1")
        md_lines.append(f"- **Sign consistency:** {'All specifications have the same sign' if sign_consistent_g2 else 'Mixed signs across specifications'}")
        md_lines.append(f"- **Significance stability:** {n_sig_g2_r1}/{len(g2_radio1)} ({pct_sig_g2:.1f}%) specifications significant at 5%")
        md_lines.append(f"- **Direction:** Median coefficient is {sign_word_g2} ({median_g2:.6f})")

        if pct_sig_g2 >= 80 and sign_consistent_g2:
            strength_g2 = "STRONG"
        elif pct_sig_g2 >= 50 and sign_consistent_g2:
            strength_g2 = "MODERATE"
        elif pct_sig_g2 >= 30:
            strength_g2 = "WEAK"
        else:
            strength_g2 = "FRAGILE"

        md_lines.append(f"- **Robustness assessment:** {strength_g2}")

md_lines.append("")
md_lines.append(f"Surface hash: `{SURFACE_HASH}`")
md_lines.append("")

with open(f"{OUTPUT_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write("\n".join(md_lines))

print(f"Wrote SPECIFICATION_SEARCH.md")
print("\nDone!")
