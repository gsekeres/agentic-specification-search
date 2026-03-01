"""
Specification Search Script for Combes, Decreuse, Laounan & Trannoy (2016)
"Can Information Reduce Ethnic Discrimination? Evidence from Airbnb"
American Economic Journal: Applied Economics

Paper ID: 120078-V1

Surface-driven execution:
  - G1: Table 5 Col 1 baseline: log_price ~ minodummy*rev100 with listing FE
  - Panel FE (within estimator), clustered at listing level
  - Focal parameter: coefficient on minodummy x rev100

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
import time
import warnings
import random
import os
warnings.filterwarnings('ignore')

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, "scripts")
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash, software_block
)

DATA_DIR = "data/downloads/extracted/120078-V1"
PAPER_ID = "120078-V1"

with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()
design_audit = surface_obj["baseline_groups"][0]["design_audit"]
inference_canonical = surface_obj["baseline_groups"][0]["inference_plan"]["canonical"]

# ============================================================
# Variable definitions
# ============================================================
missing_cols = [
    'missingcabletv', 'missingwireless', 'missingheating', 'missingac',
    'missingelevator', 'missinghandiaccess', 'missingdoorman', 'missingfireplace',
    'missingwasher', 'missingdryer', 'missingparking', 'missinggym',
    'missingpool', 'missingbuzzer', 'missinghottub', 'missingbreakfast',
    'missingfamily', 'missingevents', 'missingyear', 'missingverified_email',
    'missingverified_phone', 'missingfacebook', 'missingverified_offline',
    'missingsuperhost', 'missingcancel_policy', 'missingnoccur_pro_true'
]

size_vars = ['person_capacity345', 'bedrooms', 'bathrooms']
descrip_gen = ['appart', 'house_loft']
descrip_spe = ['couch', 'airbed', 'sofa', 'futon']
equip_vars = ['cabletv', 'wireless', 'heating', 'ac', 'elevator', 'handiaccess',
              'doorman', 'fireplace', 'washer', 'dryer', 'parking', 'gym', 'pool',
              'buzzer', 'hottub', 'breakfast', 'family', 'events']
rules_vars = ['people', 'extrapeople', 'cancel_policy', 'smoking_allowed', 'pets_allowed']
loueur_vars = ['more_1_flat', 'year2009', 'year2010', 'year2011', 'year2012',
               'year2013', 'year2014', 'year2015', 'superhost', 'verified_email',
               'verified_offline', 'verified_phone', 'facebook']
count_vars = ['count_descrip', 'count_about', 'count_languages', 'count_rules',
              'picture_count', 'noccur_pro_true', 'change_pics']
substantive_controls = (['sharedflat'] + size_vars + descrip_gen + descrip_spe +
                        equip_vars + rules_vars + loueur_vars + count_vars)

# Columns to load
needed_cols = (['log_price', 'minodummy', 'rev100', 'review', 'newid', 'wave',
                'citywaveID', 'hoodcityID', 'Drev100',
                'lastrat7', 'lastrat8', 'lastrat9', 'lastrat10'] +
               substantive_controls + missing_cols)
needed_cols = list(dict.fromkeys(needed_cols))  # deduplicate

# ============================================================
# Load and prepare data
# ============================================================
print("Loading data...")
t0 = time.time()
df_raw = pd.read_stata(f"{DATA_DIR}/data/base_airbnb_AEJ.dta")
# Keep only needed columns to reduce memory
keep_cols = set(needed_cols)
drop_cols = [c for c in df_raw.columns if c not in keep_cols]
df_raw.drop(columns=drop_cols, inplace=True)
for col in df_raw.columns:
    if df_raw[col].dtype == np.float32:
        df_raw[col] = df_raw[col].astype(np.float64)
print(f"  Loaded in {time.time()-t0:.1f}s, shape: {df_raw.shape}")

df_raw = df_raw[df_raw['Drev100'] > 0].copy()

# Create interaction terms
df_raw['mino_x_rev100'] = df_raw['minodummy'] * df_raw['rev100']
df_raw['lastrat7_x_rev100'] = df_raw['lastrat7'] * df_raw['rev100']
df_raw['lastrat8_x_rev100'] = df_raw['lastrat8'] * df_raw['rev100']
df_raw['lastrat9_x_rev100'] = df_raw['lastrat9'] * df_raw['rev100']
df_raw['lastrat10_x_rev100'] = df_raw['lastrat10'] * df_raw['rev100']

# Quadratic terms for Table 5 Col 3
df_raw['mino_x_rev100sq'] = df_raw['minodummy'] * df_raw['rev100'] ** 2
df_raw['lastrat7_x_rev100sq'] = df_raw['lastrat7'] * df_raw['rev100'] ** 2
df_raw['lastrat8_x_rev100sq'] = df_raw['lastrat8'] * df_raw['rev100'] ** 2
df_raw['lastrat9_x_rev100sq'] = df_raw['lastrat9'] * df_raw['rev100'] ** 2
df_raw['lastrat10_x_rev100sq'] = df_raw['lastrat10'] * df_raw['rev100'] ** 2

# Wave x minority interactions (wave 10 = reference)
df_raw['wave_int'] = df_raw['wave'].astype(int)
mino_wave_cols = []
for w in sorted(df_raw['wave_int'].unique()):
    if w != 10:
        colname = f'mino_wave_{w}'
        df_raw[colname] = df_raw['minodummy'] * (df_raw['wave_int'] == w).astype(float)
        mino_wave_cols.append(colname)

df_raw['newid_int'] = df_raw['newid'].astype(int)
df_raw['citywaveID_int'] = df_raw['citywaveID'].astype(int)
df_raw['hoodcityID_int'] = df_raw['hoodcityID'].astype(int)

# Build samples
df_lt40 = df_raw[(df_raw['review'] > 0) & (df_raw['review'] < 40)].copy()
df_lt60 = df_raw[(df_raw['review'] > 0) & (df_raw['review'] < 60)].copy()
df_lt80 = df_raw[(df_raw['review'] > 0) & (df_raw['review'] < 80)].copy()
df_lt100 = df_raw[(df_raw['review'] > 0) & (df_raw['review'] < 100)].copy()
df_gt0 = df_raw[df_raw['review'] > 0].copy()

print(f"Data loaded in {time.time()-t0:.1f}s. Samples: lt40={len(df_lt40)}, lt60={len(df_lt60)}, lt80={len(df_lt80)}")

# ============================================================
# Rating terms and formula builder
# ============================================================
rating_rev_terms = ['lastrat7_x_rev100', 'lastrat8_x_rev100',
                    'lastrat9_x_rev100', 'lastrat10_x_rev100']
quadratic_rev_terms = ['lastrat7_x_rev100sq', 'lastrat8_x_rev100sq',
                       'lastrat9_x_rev100sq', 'lastrat10_x_rev100sq',
                       'mino_x_rev100sq']


def build_formula(controls_list, rating_terms, fe_str, include_missing=True, extra_terms=None):
    rhs_vars = ['mino_x_rev100'] + rating_terms + mino_wave_cols
    if controls_list:
        rhs_vars += controls_list
    if include_missing:
        rhs_vars += missing_cols
    if extra_terms:
        rhs_vars += extra_terms
    return f"log_price ~ {' + '.join(rhs_vars)} | {fe_str}"


# ============================================================
# Runner functions
# ============================================================
results = []
inference_results = []
spec_run_counter = 0


def run_spec(spec_id, spec_tree_path, baseline_group_id, outcome_var, treatment_var,
             formula, data, vcov, sample_desc, fixed_effects_str, controls_desc,
             cluster_var="newid_int", axis_block_name=None, axis_block=None):
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"
    t1 = time.time()

    try:
        m = pf.feols(formula, data=data, vcov=vcov)
        coef_val = float(m.coef().get(treatment_var, np.nan))
        se_val = float(m.se().get(treatment_var, np.nan))
        pval = float(m.pvalue().get(treatment_var, np.nan))
        try:
            ci = m.confint()
            ci_lower = float(ci.loc[treatment_var, ci.columns[0]]) if treatment_var in ci.index else np.nan
            ci_upper = float(ci.loc[treatment_var, ci.columns[1]]) if treatment_var in ci.index else np.nan
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
            inference={"spec_id": inference_canonical["spec_id"],
                       "params": inference_canonical["params"]},
            software=SW_BLOCK, surface_hash=SURFACE_HASH,
            design={"panel_fixed_effects": design_audit},
            axis_block_name=axis_block_name, axis_block=axis_block)
        results.append({
            "paper_id": PAPER_ID, "spec_run_id": run_id, "spec_id": spec_id,
            "spec_tree_path": spec_tree_path, "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var, "treatment_var": treatment_var,
            "coefficient": coef_val, "std_error": se_val, "p_value": pval,
            "ci_lower": ci_lower, "ci_upper": ci_upper,
            "n_obs": nobs, "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc, "fixed_effects": fixed_effects_str,
            "controls_desc": controls_desc, "cluster_var": cluster_var,
            "run_success": 1, "run_error": ""})
        print(f"  [{time.time()-t1:.0f}s] OK: {spec_id} | coef={coef_val:.6f} se={se_val:.6f} p={pval:.4f} N={nobs}")
        return run_id, coef_val, se_val, pval, nobs
    except Exception as e:
        err_msg = str(e)[:240]
        payload = make_failure_payload(
            error=err_msg,
            error_details=error_details_from_exception(e, stage="estimation"),
            software=SW_BLOCK, surface_hash=SURFACE_HASH)
        results.append({
            "paper_id": PAPER_ID, "spec_run_id": run_id, "spec_id": spec_id,
            "spec_tree_path": spec_tree_path, "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var, "treatment_var": treatment_var,
            "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "n_obs": np.nan, "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc, "fixed_effects": fixed_effects_str,
            "controls_desc": controls_desc, "cluster_var": cluster_var,
            "run_success": 0, "run_error": err_msg})
        print(f"  [{time.time()-t1:.0f}s] FAIL: {spec_id} | {err_msg[:80]}")
        return run_id, np.nan, np.nan, np.nan, np.nan


def run_inference_variant(base_run_id, spec_id, spec_tree_path, baseline_group_id,
                          formula, data, vcov, cluster_var=""):
    global spec_run_counter
    spec_run_counter += 1
    infer_run_id = f"{PAPER_ID}_infer_{spec_run_counter:03d}"
    treatment_var = "mino_x_rev100"
    try:
        m = pf.feols(formula, data=data, vcov=vcov)
        coef_val = float(m.coef().get(treatment_var, np.nan))
        se_val = float(m.se().get(treatment_var, np.nan))
        pval = float(m.pvalue().get(treatment_var, np.nan))
        try:
            ci = m.confint()
            ci_lower = float(ci.loc[treatment_var, ci.columns[0]]) if treatment_var in ci.index else np.nan
            ci_upper = float(ci.loc[treatment_var, ci.columns[1]]) if treatment_var in ci.index else np.nan
        except Exception:
            ci_lower, ci_upper = np.nan, np.nan
        nobs = int(m._N)
        try:
            r2 = float(m._r2)
        except Exception:
            r2 = np.nan
        payload_dict = {
            "coefficients": {k: float(v) for k, v in m.coef().items()},
            "inference": {"spec_id": spec_id, "params": {}},
            "software": SW_BLOCK, "surface_hash": SURFACE_HASH}
        inference_results.append({
            "paper_id": PAPER_ID, "inference_run_id": infer_run_id,
            "spec_run_id": base_run_id, "spec_id": spec_id,
            "spec_tree_path": spec_tree_path, "baseline_group_id": baseline_group_id,
            "coefficient": coef_val, "std_error": se_val, "p_value": pval,
            "ci_lower": ci_lower, "ci_upper": ci_upper,
            "n_obs": nobs, "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload_dict),
            "run_success": 1, "run_error": ""})
        print(f"  OK infer: {spec_id} | se={se_val:.6f} p={pval:.4f}")
    except Exception as e:
        err_msg = str(e)[:240]
        payload_dict = {"error": err_msg,
            "error_details": error_details_from_exception(e, stage="inference"),
            "software": SW_BLOCK, "surface_hash": SURFACE_HASH}
        inference_results.append({
            "paper_id": PAPER_ID, "inference_run_id": infer_run_id,
            "spec_run_id": base_run_id, "spec_id": spec_id,
            "spec_tree_path": spec_tree_path, "baseline_group_id": baseline_group_id,
            "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "n_obs": np.nan, "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload_dict),
            "run_success": 0, "run_error": err_msg})
        print(f"  FAIL infer: {spec_id} | {err_msg[:80]}")


# ============================================================
# Standard FE strings
# ============================================================
baseline_fe = "newid_int + citywaveID_int"
baseline_formula = build_formula(substantive_controls, rating_rev_terms, baseline_fe)

# ============================================================
# BASELINE
# ============================================================
print("\n=== BASELINE ===")
base_run_id, _, _, _, _ = run_spec(
    "baseline", "designs/panel_fixed_effects.md#baseline", "G1",
    "log_price", "mino_x_rev100", baseline_formula, df_lt40,
    {"CRV1": "newid_int"},
    "review>0 & review<40 (Table 5 Col 1)", "newid + citywaveID",
    "lesX (all substantive + missing dummies)")

# ============================================================
# ADDITIONAL BASELINES
# ============================================================
print("\n=== ADDITIONAL BASELINES ===")
run_spec("baseline__table5_col2", "designs/panel_fixed_effects.md#baseline", "G1",
         "log_price", "mino_x_rev100", baseline_formula, df_lt60,
         {"CRV1": "newid_int"},
         "review>0 & review<60 (Table 5 Col 2)", "newid + citywaveID",
         "lesX (all substantive + missing dummies)")

quadratic_formula = build_formula(substantive_controls,
                                  rating_rev_terms + quadratic_rev_terms, baseline_fe)
run_spec("baseline__table5_col3", "designs/panel_fixed_effects.md#baseline", "G1",
         "log_price", "mino_x_rev100", quadratic_formula, df_lt80,
         {"CRV1": "newid_int"},
         "review>0 & review<80 with quadratic (Table 5 Col 3)", "newid + citywaveID",
         "lesX + quadratic interactions")

# ============================================================
# RC: CONTROLS LOO
# ============================================================
print("\n=== RC: CONTROLS LOO ===")
loo_vars = ['sharedflat', 'person_capacity345', 'bedrooms', 'bathrooms',
            'superhost', 'verified_email', 'facebook', 'count_descrip',
            'picture_count', 'noccur_pro_true', 'change_pics', 'more_1_flat',
            'cancel_policy']
for var in loo_vars:
    ctrl = [c for c in substantive_controls if c != var]
    formula = build_formula(ctrl, rating_rev_terms, baseline_fe)
    run_spec(f"rc/controls/loo/drop_{var}",
             "modules/robustness/controls.md#leave-one-out-controls-loo", "G1",
             "log_price", "mino_x_rev100", formula, df_lt40,
             {"CRV1": "newid_int"},
             "review>0 & review<40", "newid + citywaveID",
             f"baseline minus {var}",
             axis_block_name="controls",
             axis_block={"spec_id": f"rc/controls/loo/drop_{var}", "family": "loo",
                         "dropped": [var], "added": [], "n_controls": len(ctrl)})

# ============================================================
# RC: CONTROL SETS
# ============================================================
print("\n=== RC: CONTROL SETS ===")

for set_id, ctrl_list, incl_miss, desc in [
    ("none", [], False, "no substantive controls"),
    ("minimal_size", size_vars, False, "size only"),
    ("property_chars", ['sharedflat'] + size_vars + descrip_gen + descrip_spe + equip_vars + rules_vars, True, "property chars"),
    ("host_chars", loueur_vars + count_vars, True, "host/loueur/count"),
    ("full", substantive_controls, True, "full (same as baseline)")
]:
    formula = build_formula(ctrl_list, rating_rev_terms, baseline_fe, include_missing=incl_miss)
    run_spec(f"rc/controls/sets/{set_id}",
             "modules/robustness/controls.md#standard-control-sets", "G1",
             "log_price", "mino_x_rev100", formula, df_lt40,
             {"CRV1": "newid_int"},
             "review>0 & review<40", "newid + citywaveID", desc,
             axis_block_name="controls",
             axis_block={"spec_id": f"rc/controls/sets/{set_id}", "family": "sets",
                         "n_controls": len(ctrl_list), "set_name": set_id})

# ============================================================
# RC: CONTROL PROGRESSION
# ============================================================
print("\n=== RC: CONTROL PROGRESSION ===")
prog_steps = [
    ("bivariate", [], False, "bivariate"),
    ("size_only", size_vars, False, "size only"),
    ("size_amenities", size_vars + descrip_gen + descrip_spe + equip_vars, True, "size+amenities"),
    ("size_amenities_host", size_vars + descrip_gen + descrip_spe + equip_vars + rules_vars + loueur_vars, True, "size+amenities+host"),
    ("full_with_counts", substantive_controls, True, "full")
]
for prog_id, ctrl_list, incl_miss, desc in prog_steps:
    formula = build_formula(ctrl_list, rating_rev_terms, baseline_fe, include_missing=incl_miss)
    run_spec(f"rc/controls/progression/{prog_id}",
             "modules/robustness/controls.md#control-progression", "G1",
             "log_price", "mino_x_rev100", formula, df_lt40,
             {"CRV1": "newid_int"},
             "review>0 & review<40", "newid + citywaveID", desc,
             axis_block_name="controls",
             axis_block={"spec_id": f"rc/controls/progression/{prog_id}", "family": "progression",
                         "n_controls": len(ctrl_list)})

# ============================================================
# RC: CONTROL SUBSETS (random)
# ============================================================
print("\n=== RC: CONTROL SUBSETS ===")
rng = random.Random(120078)
for i in range(1, 16):
    n_pick = rng.randint(5, len(substantive_controls) - 1)
    subset = sorted(rng.sample(substantive_controls, n_pick))
    formula = build_formula(subset, rating_rev_terms, baseline_fe)
    run_spec(f"rc/controls/subset/random_{i:03d}",
             "modules/robustness/controls.md#random-control-subsets", "G1",
             "log_price", "mino_x_rev100", formula, df_lt40,
             {"CRV1": "newid_int"},
             "review>0 & review<40", "newid + citywaveID",
             f"random subset {i}: {n_pick}/{len(substantive_controls)} controls",
             axis_block_name="controls",
             axis_block={"spec_id": f"rc/controls/subset/random_{i:03d}",
                         "family": "subset", "draw_index": i,
                         "included": subset, "n_controls": n_pick})

# ============================================================
# RC: SAMPLE RESTRICTIONS
# ============================================================
print("\n=== RC: SAMPLE RESTRICTIONS ===")
for sid, data, sdesc, desc in [
    ("review_lt_60", df_lt60, "review>0 & review<60", "review upper bound 60"),
    ("review_lt_80", df_lt80, "review>0 & review<80", "review upper bound 80"),
    ("review_lt_100", df_lt100, "review>0 & review<100", "review upper bound 100"),
    ("review_gt_0", df_gt0, "review>0 (no upper bound)", "no review upper bound")
]:
    run_spec(f"rc/sample/restriction/{sid}",
             "modules/robustness/sample.md#sample-restriction", "G1",
             "log_price", "mino_x_rev100", baseline_formula, data,
             {"CRV1": "newid_int"}, sdesc, "newid + citywaveID",
             "lesX (all substantive + missing dummies)",
             axis_block_name="sample",
             axis_block={"spec_id": f"rc/sample/restriction/{sid}",
                         "family": "restriction", "description": desc})

# ============================================================
# RC: SAMPLE OUTLIERS
# ============================================================
print("\n=== RC: SAMPLE OUTLIERS ===")
for trim_lo, trim_hi, sid in [(0.01, 0.99, "trim_y_1_99"), (0.05, 0.95, "trim_y_5_95")]:
    plo = df_lt40['log_price'].quantile(trim_lo)
    phi = df_lt40['log_price'].quantile(trim_hi)
    df_trim = df_lt40[(df_lt40['log_price'] >= plo) & (df_lt40['log_price'] <= phi)].copy()
    run_spec(f"rc/sample/outliers/{sid}",
             "modules/robustness/sample.md#outlier-trimming", "G1",
             "log_price", "mino_x_rev100", baseline_formula, df_trim,
             {"CRV1": "newid_int"},
             f"review>0 & review<40, log_price trimmed [{plo:.3f},{phi:.3f}]",
             "newid + citywaveID", "lesX (all substantive + missing dummies)",
             axis_block_name="sample",
             axis_block={"spec_id": f"rc/sample/outliers/{sid}",
                         "family": "outliers", "trim_lower": trim_lo, "trim_upper": trim_hi})

# ============================================================
# RC: FE VARIATIONS
# ============================================================
print("\n=== RC: FE VARIATIONS ===")
# Drop citywaveID
formula_nociwave = build_formula(substantive_controls, rating_rev_terms, "newid_int")
run_spec("rc/fe/drop/citywaveID",
         "modules/robustness/fixed_effects.md#drop-fixed-effects", "G1",
         "log_price", "mino_x_rev100", formula_nociwave, df_lt40,
         {"CRV1": "newid_int"},
         "review>0 & review<40", "newid only", "lesX",
         axis_block_name="fixed_effects",
         axis_block={"spec_id": "rc/fe/drop/citywaveID", "family": "drop", "dropped": ["citywaveID"]})

# Swap citywaveID -> hoodcityID
formula_hood = build_formula(substantive_controls, rating_rev_terms, "newid_int + hoodcityID_int")
run_spec("rc/fe/swap/hoodcityID_for_citywaveID",
         "modules/robustness/fixed_effects.md#swap-fixed-effects", "G1",
         "log_price", "mino_x_rev100", formula_hood, df_lt40,
         {"CRV1": "newid_int"},
         "review>0 & review<40", "newid + hoodcityID", "lesX",
         axis_block_name="fixed_effects",
         axis_block={"spec_id": "rc/fe/swap/hoodcityID_for_citywaveID",
                     "family": "swap", "dropped": ["citywaveID"], "added": ["hoodcityID"]})

# ============================================================
# RC: FUNCTIONAL FORM
# ============================================================
print("\n=== RC: FUNCTIONAL FORM ===")
run_spec("rc/form/treatment/quadratic_rev100",
         "modules/robustness/functional_form.md#quadratic", "G1",
         "log_price", "mino_x_rev100",
         build_formula(substantive_controls, rating_rev_terms + quadratic_rev_terms, baseline_fe),
         df_lt40, {"CRV1": "newid_int"},
         "review>0 & review<40 + quadratic", "newid + citywaveID",
         "lesX + quadratic interactions",
         axis_block_name="functional_form",
         axis_block={"spec_id": "rc/form/treatment/quadratic_rev100", "family": "quadratic",
                     "interpretation": "Adds (minodummy x rev100^2) for non-linear information effect"})

# ============================================================
# INFERENCE VARIANTS
# ============================================================
print("\n=== INFERENCE VARIANTS ===")
run_inference_variant(base_run_id, "infer/se/hc/hc1",
                      "modules/inference/standard_errors.md#heteroskedasticity-robust", "G1",
                      baseline_formula, df_lt40, "hetero")

run_inference_variant(base_run_id, "infer/se/cluster/hoodcityID",
                      "modules/inference/standard_errors.md#cluster-robust", "G1",
                      baseline_formula, df_lt40, {"CRV1": "hoodcityID_int"},
                      cluster_var="hoodcityID")

# ============================================================
# WRITE OUTPUTS
# ============================================================
print("\n=== WRITING OUTPUTS ===")
df_results = pd.DataFrame(results)
df_results.to_csv(f"{DATA_DIR}/specification_results.csv", index=False)
print(f"Wrote specification_results.csv: {len(df_results)} rows")

df_inference = pd.DataFrame(inference_results)
df_inference.to_csv(f"{DATA_DIR}/inference_results.csv", index=False)
print(f"Wrote inference_results.csv: {len(df_inference)} rows")

n_success = int(df_results['run_success'].sum())
n_fail = len(df_results) - n_success
n_infer_success = int(df_inference['run_success'].sum()) if len(df_inference) > 0 else 0

md_content = f"""# Specification Search: {PAPER_ID}

## Paper
- **Title**: Can Information Reduce Ethnic Discrimination? Evidence from Airbnb
- **Authors**: Combes, Decreuse, Laounan & Trannoy
- **Journal**: AEJ: Applied Economics

## Surface Summary
- **Baseline group**: G1 (Table 5 Col 1)
- **Design**: Panel fixed effects (within estimator)
- **Outcome**: log_price (daily log-price of Airbnb listing)
- **Treatment**: minodummy x rev100 (minority host x review count/100)
- **FE**: newid (listing) + citywaveID (city x wave)
- **Cluster**: newid (listing)
- **Sample**: Drev100 > 0 & review > 0 & review < 40
- **Budget**: max 70 core specs, 15 control subsets
- **Seed**: 120078

## Execution Summary
- **Planned specs**: {len(df_results)} estimate rows + {len(df_inference)} inference rows
- **Successful**: {n_success} estimate + {n_infer_success} inference
- **Failed**: {n_fail} estimate + {len(df_inference) - n_infer_success} inference

### Spec breakdown
| Category | Count |
|----------|-------|
| baseline | 1 |
| baseline (additional) | 2 |
| rc/controls/loo | {len(loo_vars)} |
| rc/controls/sets | 5 |
| rc/controls/progression | {len(prog_steps)} |
| rc/controls/subset | 15 |
| rc/sample/restriction | 4 |
| rc/sample/outliers | 2 |
| rc/fe/drop | 1 |
| rc/fe/swap | 1 |
| rc/form/treatment | 1 |
| **Total estimate rows** | **{len(df_results)}** |
| infer/se/hc | 1 |
| infer/se/cluster | 1 |
| **Total inference rows** | **{len(df_inference)}** |

## Notes
- In the listing FE model, many lesX controls are time-invariant and get absorbed.
  pyfixest silently drops collinear variables, matching Stata xtreg behavior.
- Wave-minority interactions (c.minodummy#ib10.wave) created manually (wave 10 = reference).
- Focal coefficient: mino_x_rev100 = minodummy * rev100.
- Table 5 Col 3 adds quadratic terms.

## Software
- Python {SW_BLOCK.get('runner_version', 'N/A')}
- pyfixest {SW_BLOCK.get('packages', {}).get('pyfixest', 'N/A')}
- pandas {SW_BLOCK.get('packages', {}).get('pandas', 'N/A')}
- numpy {SW_BLOCK.get('packages', {}).get('numpy', 'N/A')}
"""

with open(f"{DATA_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write(md_content)
print("Wrote SPECIFICATION_SEARCH.md")

total_time = time.time() - t0
print(f"\nDone! {len(df_results)} specs + {len(df_inference)} inference variants in {total_time:.0f}s")
