"""
Specification Search Script for Okeke (2021?)
"When a Doctor Falls from the Sky: The Impact of Easing Doctor Supply
Constraints on Mortality"

Paper ID: 181581-V1

Surface-driven execution:
  - G1: mort7 ~ mlp + doctor (Table 4 Col 1 baseline, strata FE, cluster fid)
  - Randomized experiment, ITT
  - ~50+ specifications

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
import random
warnings.filterwarnings('ignore')

sys.path.insert(0, "scripts")
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash, software_block
)

PAPER_ID = "181581-V1"
DATA_DIR = "data/downloads/extracted/181581-V1"
OUT_DIR = DATA_DIR  # outputs go to top-level extracted dir

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

design_audit = surface_obj["baseline_groups"][0]["design_audit"]
inference_canonical = surface_obj["baseline_groups"][0]["inference_plan"]["canonical"]
inference_variants = surface_obj["baseline_groups"][0]["inference_plan"]["variants"]

# ---- Load Data ----
df_raw = pd.read_stata(f"{DATA_DIR}/data/analysis/child.dta", convert_categoricals=False)

# Convert float32 -> float64 for stability
for col in df_raw.columns:
    if df_raw[col].dtype == np.float32:
        df_raw[col] = df_raw[col].astype(np.float64)

# Fix variable name mismatches from the surface
# The surface references 'auton' but the data has 'autonomy'
if 'autonomy' in df_raw.columns and 'auton' not in df_raw.columns:
    df_raw['auton'] = df_raw['autonomy']
# The surface references 'hc_drug' but data has 'hc_drugs'
if 'hc_drugs' in df_raw.columns and 'hc_drug' not in df_raw.columns:
    df_raw['hc_drug'] = df_raw['hc_drugs']
# The surface references 'hc_nopow' but data has 'hc_nopower'
if 'hc_nopower' in df_raw.columns and 'hc_nopow' not in df_raw.columns:
    df_raw['hc_nopow'] = df_raw['hc_nopower']

# qtr is datetime; convert to integer for FE
if df_raw['qtr'].dtype == 'datetime64[ns]':
    df_raw['qtr_int'] = df_raw['qtr'].dt.year * 4 + df_raw['qtr'].dt.quarter
else:
    df_raw['qtr_int'] = df_raw['qtr']

# Ensure strata and fid are numeric for FE absorption
df_raw['strata_int'] = df_raw['strata'].astype('Int64')
df_raw['fid_int'] = df_raw['fid'].astype('Int64')

# magedum and mschool are factor variables in Stata; create dummies
for fv in ['magedum', 'mschool', 'hc_clean', 'hc_cond']:
    if fv in df_raw.columns:
        vals = sorted(df_raw[fv].dropna().unique())
        if len(vals) > 1:
            base = vals[0]  # use first as reference
            for v in vals[1:]:
                df_raw[f'{fv}_{int(v)}'] = (df_raw[fv] == v).astype(float)

# Define control variable groups (matching Stata globals)
magedum_dummies = [c for c in df_raw.columns if c.startswith('magedum_')]
mschool_dummies = [c for c in df_raw.columns if c.startswith('mschool_')]
hc_clean_dummies = [c for c in df_raw.columns if c.startswith('hc_clean_')]
hc_cond_dummies = [c for c in df_raw.columns if c.startswith('hc_cond_')]

cont_ind = ['cct'] + magedum_dummies + ['first', 'hausa'] + mschool_dummies + ['auton', 'car', 'last', 'gest']
cont_base = cont_ind + ['male']
cont_hc = ['hc_deliveries', 'hc_cesarean', 'hc_transfusion'] + hc_clean_dummies
cont_all_extra = ['pastdeath', 'hc_workers', 'hc_open24hrs', 'hc_equipment',
                  'hc_beds', 'hc_lab', 'hc_drug', 'hc_nopow', 'hc_vent'] + hc_cond_dummies
cont_all = cont_base + cont_hc + cont_all_extra

# Full control pool for subset sampling
control_pool = cont_all.copy()

# ---- Globals ----
results = []
inference_results = []
spec_run_counter = 0


def run_spec(spec_id, spec_tree_path, baseline_group_id, outcome_var,
             treatment_var, additional_treatment_vars, controls, fe_vars,
             data, vcov, sample_desc, controls_desc, fe_desc,
             cluster_var="fid", axis_block_name=None, axis_block=None,
             notes=""):
    """Run a single specification and append to results."""
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        # Build formula
        rhs_vars = []
        if additional_treatment_vars:
            rhs_vars.extend(additional_treatment_vars)
        rhs_vars.append(treatment_var)
        if controls:
            rhs_vars.extend(controls)

        rhs = " + ".join(rhs_vars)

        if fe_vars:
            fe_str = " + ".join(fe_vars)
            formula = f"{outcome_var} ~ {rhs} | {fe_str}"
        else:
            formula = f"{outcome_var} ~ {rhs}"

        m = pf.feols(formula, data=data, vcov=vcov)

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
                       "params": inference_canonical["params"]},
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
            "fixed_effects": fe_desc,
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
            "run_success": 0,
            "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


def run_inference_variant(base_run_id, spec_id, spec_tree_path,
                          baseline_group_id, outcome_var, treatment_var,
                          additional_treatment_vars, controls, fe_vars,
                          data, vcov_new, cluster_var_new=""):
    """Re-estimate with alternative inference and write to inference_results."""
    global spec_run_counter
    spec_run_counter += 1
    inf_run_id = f"{PAPER_ID}_infer_{spec_run_counter:03d}"

    try:
        rhs_vars = []
        if additional_treatment_vars:
            rhs_vars.extend(additional_treatment_vars)
        rhs_vars.append(treatment_var)
        if controls:
            rhs_vars.extend(controls)

        rhs = " + ".join(rhs_vars)
        if fe_vars:
            fe_str = " + ".join(fe_vars)
            formula = f"{outcome_var} ~ {rhs} | {fe_str}"
        else:
            formula = f"{outcome_var} ~ {rhs}"

        m = pf.feols(formula, data=data, vcov=vcov_new)

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
            inference={"spec_id": spec_id,
                       "params": {"cluster_var": cluster_var_new} if cluster_var_new else {}},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"randomized_experiment": design_audit},
        )

        inference_results.append({
            "paper_id": PAPER_ID,
            "inference_run_id": inf_run_id,
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
            "run_success": 1,
            "run_error": ""
        })

    except Exception as e:
        err_msg = str(e)[:240]
        err_details = error_details_from_exception(e, stage="inference_variant")
        payload = make_failure_payload(
            error=err_msg, error_details=err_details,
            software=SW_BLOCK, surface_hash=SURFACE_HASH
        )
        inference_results.append({
            "paper_id": PAPER_ID,
            "inference_run_id": inf_run_id,
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
            "run_success": 0,
            "run_error": err_msg
        })


# ============================================================
# DATA PREP
# ============================================================
df = df_raw.copy()

# ============================================================
# BASELINE: Table 4 Col 1 -- mort7 ~ mlp + doctor | strata, cl(fid)
# ============================================================
print("Running baseline...")
baseline_run_id, *_ = run_spec(
    spec_id="baseline",
    spec_tree_path="designs/randomized_experiment.md#baseline",
    baseline_group_id="G1",
    outcome_var="mort7",
    treatment_var="doctor",
    additional_treatment_vars=["mlp"],
    controls=[],
    fe_vars=["strata_int"],
    data=df,
    vcov={"CRV1": "fid_int"},
    sample_desc="Live births, all arms",
    controls_desc="none",
    fe_desc="strata",
    cluster_var="fid"
)

# ============================================================
# ADDITIONAL BASELINE: Table 4 Col 2 -- with basic controls + qtr FE
# ============================================================
print("Running baseline with basic controls (Table 4 Col 2)...")
run_spec(
    spec_id="baseline__basic_controls",
    spec_tree_path="designs/randomized_experiment.md#baseline",
    baseline_group_id="G1",
    outcome_var="mort7",
    treatment_var="doctor",
    additional_treatment_vars=["mlp"],
    controls=cont_base,
    fe_vars=["strata_int", "qtr_int"],
    data=df,
    vcov={"CRV1": "fid_int"},
    sample_desc="Live births, all arms",
    controls_desc="cont_base (cct, magedum, first, hausa, mschool, auton, car, last, gest, male)",
    fe_desc="strata + qtr",
    cluster_var="fid"
)

# ============================================================
# DESIGN: Difference in means (no FE)
# ============================================================
print("Running design/diff_in_means...")
run_spec(
    spec_id="design/randomized_experiment/estimator/diff_in_means",
    spec_tree_path="designs/randomized_experiment.md#diff-in-means",
    baseline_group_id="G1",
    outcome_var="mort7",
    treatment_var="doctor",
    additional_treatment_vars=["mlp"],
    controls=[],
    fe_vars=[],
    data=df,
    vcov={"CRV1": "fid_int"},
    sample_desc="Live births, all arms",
    controls_desc="none",
    fe_desc="none",
    cluster_var="fid"
)

# ============================================================
# RC: CONTROLS SETS
# ============================================================
print("Running rc/controls/sets...")

# rc/controls/sets/none -- already captured by baseline (no controls, strata FE)
# but the surface lists it as separate from baseline
run_spec(
    spec_id="rc/controls/sets/none",
    spec_tree_path="modules/robustness/controls.md#sets",
    baseline_group_id="G1",
    outcome_var="mort7",
    treatment_var="doctor",
    additional_treatment_vars=["mlp"],
    controls=[],
    fe_vars=["strata_int"],
    data=df,
    vcov={"CRV1": "fid_int"},
    sample_desc="Live births, all arms",
    controls_desc="none",
    fe_desc="strata",
    cluster_var="fid",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/none", "family": "sets", "controls": [], "n_controls": 0}
)

# rc/controls/sets/basic -- cont_base + strata + qtr FE
run_spec(
    spec_id="rc/controls/sets/basic",
    spec_tree_path="modules/robustness/controls.md#sets",
    baseline_group_id="G1",
    outcome_var="mort7",
    treatment_var="doctor",
    additional_treatment_vars=["mlp"],
    controls=cont_base,
    fe_vars=["strata_int", "qtr_int"],
    data=df,
    vcov={"CRV1": "fid_int"},
    sample_desc="Live births, all arms",
    controls_desc="cont_base",
    fe_desc="strata + qtr",
    cluster_var="fid",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/basic", "family": "sets",
                "controls": cont_base, "n_controls": len(cont_base)}
)

# rc/controls/sets/extended -- cont_base + cont_hc + strata + qtr FE
run_spec(
    spec_id="rc/controls/sets/extended",
    spec_tree_path="modules/robustness/controls.md#sets",
    baseline_group_id="G1",
    outcome_var="mort7",
    treatment_var="doctor",
    additional_treatment_vars=["mlp"],
    controls=cont_base + cont_hc,
    fe_vars=["strata_int", "qtr_int"],
    data=df,
    vcov={"CRV1": "fid_int"},
    sample_desc="Live births, all arms",
    controls_desc="cont_base + cont_hc",
    fe_desc="strata + qtr",
    cluster_var="fid",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/extended", "family": "sets",
                "controls": cont_base + cont_hc, "n_controls": len(cont_base + cont_hc)}
)

# ============================================================
# RC: CONTROLS PROGRESSION
# ============================================================
print("Running rc/controls/progression...")

# strata_only (no controls, strata FE, no qtr)
run_spec(
    spec_id="rc/controls/progression/strata_only",
    spec_tree_path="modules/robustness/controls.md#progression",
    baseline_group_id="G1",
    outcome_var="mort7",
    treatment_var="doctor",
    additional_treatment_vars=["mlp"],
    controls=[],
    fe_vars=["strata_int"],
    data=df,
    vcov={"CRV1": "fid_int"},
    sample_desc="Live births, all arms",
    controls_desc="none",
    fe_desc="strata",
    cluster_var="fid",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/strata_only", "family": "progression", "step": 1}
)

# strata_qtr
run_spec(
    spec_id="rc/controls/progression/strata_qtr",
    spec_tree_path="modules/robustness/controls.md#progression",
    baseline_group_id="G1",
    outcome_var="mort7",
    treatment_var="doctor",
    additional_treatment_vars=["mlp"],
    controls=[],
    fe_vars=["strata_int", "qtr_int"],
    data=df,
    vcov={"CRV1": "fid_int"},
    sample_desc="Live births, all arms",
    controls_desc="none",
    fe_desc="strata + qtr",
    cluster_var="fid",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/strata_qtr", "family": "progression", "step": 2}
)

# strata_qtr_individual
run_spec(
    spec_id="rc/controls/progression/strata_qtr_individual",
    spec_tree_path="modules/robustness/controls.md#progression",
    baseline_group_id="G1",
    outcome_var="mort7",
    treatment_var="doctor",
    additional_treatment_vars=["mlp"],
    controls=cont_base,
    fe_vars=["strata_int", "qtr_int"],
    data=df,
    vcov={"CRV1": "fid_int"},
    sample_desc="Live births, all arms",
    controls_desc="cont_base",
    fe_desc="strata + qtr",
    cluster_var="fid",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/strata_qtr_individual", "family": "progression", "step": 3}
)

# strata_qtr_individual_hc
run_spec(
    spec_id="rc/controls/progression/strata_qtr_individual_hc",
    spec_tree_path="modules/robustness/controls.md#progression",
    baseline_group_id="G1",
    outcome_var="mort7",
    treatment_var="doctor",
    additional_treatment_vars=["mlp"],
    controls=cont_base + cont_hc,
    fe_vars=["strata_int", "qtr_int"],
    data=df,
    vcov={"CRV1": "fid_int"},
    sample_desc="Live births, all arms",
    controls_desc="cont_base + cont_hc",
    fe_desc="strata + qtr",
    cluster_var="fid",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/strata_qtr_individual_hc", "family": "progression", "step": 4}
)

# ============================================================
# RC: CONTROLS LOO (drop one from extended)
# ============================================================
print("Running rc/controls/loo...")

loo_from_extended = cont_base + cont_hc  # extended set
loo_candidates = {
    "drop_male": "male",
    "drop_first": "first",
    "drop_hausa": "hausa",
    "drop_gest": "gest",
    "drop_car": "car",
    "drop_last": "last",
    "drop_auton": "auton",
    "drop_cct": "cct",
    "drop_hc_deliveries": "hc_deliveries",
    "drop_hc_cesarean": "hc_cesarean",
    "drop_hc_transfusion": "hc_transfusion",
    "drop_hc_clean": None,  # hc_clean dummies
}

for loo_name, drop_var in loo_candidates.items():
    if drop_var is not None:
        # Drop a single variable
        loo_controls = [c for c in loo_from_extended if c != drop_var]
    else:
        # Special: drop hc_clean dummies
        loo_controls = [c for c in loo_from_extended if not c.startswith('hc_clean_')]

    run_spec(
        spec_id=f"rc/controls/loo/{loo_name}",
        spec_tree_path="modules/robustness/controls.md#loo",
        baseline_group_id="G1",
        outcome_var="mort7",
        treatment_var="doctor",
        additional_treatment_vars=["mlp"],
        controls=loo_controls,
        fe_vars=["strata_int", "qtr_int"],
        data=df,
        vcov={"CRV1": "fid_int"},
        sample_desc="Live births, all arms",
        controls_desc=f"extended minus {loo_name.replace('drop_', '')}",
        fe_desc="strata + qtr",
        cluster_var="fid",
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/controls/loo/{loo_name}", "family": "loo",
                    "dropped": [drop_var] if drop_var else [c for c in hc_clean_dummies],
                    "n_controls": len(loo_controls)}
    )

# ============================================================
# RC: CONTROLS SUBSET (random draws)
# ============================================================
print("Running rc/controls/subset...")

rng = random.Random(181581)

# Pool of individual-level controls to sample from
# (all individual + facility controls, excluding dummies that go together)
# For subset sampling, use the named control variables
named_controls = ['cct', 'first', 'hausa', 'auton', 'car', 'last', 'gest', 'male',
                  'hc_deliveries', 'hc_cesarean', 'hc_transfusion',
                  'pastdeath', 'hc_workers', 'hc_open24hrs', 'hc_equipment',
                  'hc_beds', 'hc_lab', 'hc_drug', 'hc_nopow', 'hc_vent']
# Factor variables treated as groups
factor_groups = {
    'magedum': magedum_dummies,
    'mschool': mschool_dummies,
    'hc_clean': hc_clean_dummies,
    'hc_cond': hc_cond_dummies,
}

# Build pool of control "units" (single vars + factor groups)
control_units = named_controls + list(factor_groups.keys())

for draw_idx in range(1, 11):
    # Random number of controls (between 1 and len(control_units))
    n_draw = rng.randint(1, len(control_units))
    drawn_units = rng.sample(control_units, n_draw)

    # Expand factor groups to their dummies
    drawn_vars = []
    for u in drawn_units:
        if u in factor_groups:
            drawn_vars.extend(factor_groups[u])
        else:
            drawn_vars.append(u)

    run_spec(
        spec_id=f"rc/controls/subset/random_{draw_idx:03d}",
        spec_tree_path="modules/robustness/controls.md#subset",
        baseline_group_id="G1",
        outcome_var="mort7",
        treatment_var="doctor",
        additional_treatment_vars=["mlp"],
        controls=drawn_vars,
        fe_vars=["strata_int", "qtr_int"],
        data=df,
        vcov={"CRV1": "fid_int"},
        sample_desc="Live births, all arms",
        controls_desc=f"random subset {draw_idx:03d}: {', '.join(drawn_units)}",
        fe_desc="strata + qtr",
        cluster_var="fid",
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/controls/subset/random_{draw_idx:03d}", "family": "subset",
                    "draw_index": draw_idx, "drawn_units": drawn_units,
                    "expanded_vars": drawn_vars, "n_controls": len(drawn_vars),
                    "seed": 181581}
    )

# ============================================================
# RC: FIXED EFFECTS
# ============================================================
print("Running rc/fe...")

# rc/fe/add/qtr -- add qtr FE to strata-only baseline
run_spec(
    spec_id="rc/fe/add/qtr",
    spec_tree_path="modules/robustness/fixed_effects.md#add",
    baseline_group_id="G1",
    outcome_var="mort7",
    treatment_var="doctor",
    additional_treatment_vars=["mlp"],
    controls=[],
    fe_vars=["strata_int", "qtr_int"],
    data=df,
    vcov={"CRV1": "fid_int"},
    sample_desc="Live births, all arms",
    controls_desc="none",
    fe_desc="strata + qtr",
    cluster_var="fid",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/add/qtr", "action": "add", "added": ["qtr"]}
)

# rc/fe/drop/qtr -- drop qtr FE from basic-controls spec (which has strata + qtr)
run_spec(
    spec_id="rc/fe/drop/qtr",
    spec_tree_path="modules/robustness/fixed_effects.md#drop",
    baseline_group_id="G1",
    outcome_var="mort7",
    treatment_var="doctor",
    additional_treatment_vars=["mlp"],
    controls=cont_base,
    fe_vars=["strata_int"],
    data=df,
    vcov={"CRV1": "fid_int"},
    sample_desc="Live births, all arms",
    controls_desc="cont_base",
    fe_desc="strata only (qtr dropped)",
    cluster_var="fid",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/drop/qtr", "action": "drop", "dropped": ["qtr"]}
)

# ============================================================
# RC: SAMPLE RESTRICTIONS
# ============================================================
print("Running rc/sample...")

# rc/sample/restriction/exclude_multiple_births
df_single = df[df['multiple'] == 0].copy()
run_spec(
    spec_id="rc/sample/restriction/exclude_multiple_births",
    spec_tree_path="modules/robustness/sample.md#restriction",
    baseline_group_id="G1",
    outcome_var="mort7",
    treatment_var="doctor",
    additional_treatment_vars=["mlp"],
    controls=[],
    fe_vars=["strata_int"],
    data=df_single,
    vcov={"CRV1": "fid_int"},
    sample_desc="Singleton births only",
    controls_desc="none",
    fe_desc="strata",
    cluster_var="fid",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/exclude_multiple_births",
                "restriction": "multiple==0", "description": "singleton births only"}
)

# ============================================================
# RC: OUTCOME DEFINITION
# ============================================================
print("Running rc/data/outcome/mort30...")

run_spec(
    spec_id="rc/data/outcome/mort30",
    spec_tree_path="modules/robustness/data_construction.md#outcome",
    baseline_group_id="G1",
    outcome_var="mort30",
    treatment_var="doctor",
    additional_treatment_vars=["mlp"],
    controls=[],
    fe_vars=["strata_int"],
    data=df,
    vcov={"CRV1": "fid_int"},
    sample_desc="Live births, all arms",
    controls_desc="none",
    fe_desc="strata",
    cluster_var="fid",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/outcome/mort30", "outcome_change": "mort30 (30-day mortality)"}
)

# Also run mort30 with basic controls
run_spec(
    spec_id="rc/data/outcome/mort30__basic",
    spec_tree_path="modules/robustness/data_construction.md#outcome",
    baseline_group_id="G1",
    outcome_var="mort30",
    treatment_var="doctor",
    additional_treatment_vars=["mlp"],
    controls=cont_base,
    fe_vars=["strata_int", "qtr_int"],
    data=df,
    vcov={"CRV1": "fid_int"},
    sample_desc="Live births, all arms",
    controls_desc="cont_base",
    fe_desc="strata + qtr",
    cluster_var="fid",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/outcome/mort30", "outcome_change": "mort30 (30-day mortality) + basic controls"}
)

# ============================================================
# RC: TREATMENT DEFINITION
# ============================================================
print("Running rc/data/treatment/doctor_only...")

# Drop mlp from regression, estimate doctor effect vs everything else
run_spec(
    spec_id="rc/data/treatment/doctor_only",
    spec_tree_path="modules/robustness/data_construction.md#treatment",
    baseline_group_id="G1",
    outcome_var="mort7",
    treatment_var="doctor",
    additional_treatment_vars=[],
    controls=[],
    fe_vars=["strata_int"],
    data=df,
    vcov={"CRV1": "fid_int"},
    sample_desc="Live births, all arms (doctor vs control+MLP)",
    controls_desc="none",
    fe_desc="strata",
    cluster_var="fid",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/treatment/doctor_only",
                "treatment_change": "doctor only (mlp omitted, control+MLP pooled as reference)"}
)

# Also with basic controls
run_spec(
    spec_id="rc/data/treatment/doctor_only__basic",
    spec_tree_path="modules/robustness/data_construction.md#treatment",
    baseline_group_id="G1",
    outcome_var="mort7",
    treatment_var="doctor",
    additional_treatment_vars=[],
    controls=cont_base,
    fe_vars=["strata_int", "qtr_int"],
    data=df,
    vcov={"CRV1": "fid_int"},
    sample_desc="Live births, all arms (doctor vs control+MLP)",
    controls_desc="cont_base",
    fe_desc="strata + qtr",
    cluster_var="fid",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/treatment/doctor_only",
                "treatment_change": "doctor only (mlp omitted) + basic controls"}
)

# ============================================================
# RC: Additional cross-combinations for richness
# ============================================================
print("Running cross-combinations...")

# Extended controls + exclude multiple births
run_spec(
    spec_id="rc/sample/restriction/exclude_multiple__extended",
    spec_tree_path="modules/robustness/sample.md#restriction",
    baseline_group_id="G1",
    outcome_var="mort7",
    treatment_var="doctor",
    additional_treatment_vars=["mlp"],
    controls=cont_base + cont_hc,
    fe_vars=["strata_int", "qtr_int"],
    data=df_single,
    vcov={"CRV1": "fid_int"},
    sample_desc="Singleton births, extended controls",
    controls_desc="cont_base + cont_hc",
    fe_desc="strata + qtr",
    cluster_var="fid",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/exclude_multiple_births",
                "restriction": "multiple==0 + extended controls"}
)

# mort30 + exclude multiple
run_spec(
    spec_id="rc/data/outcome/mort30__singleton",
    spec_tree_path="modules/robustness/data_construction.md#outcome",
    baseline_group_id="G1",
    outcome_var="mort30",
    treatment_var="doctor",
    additional_treatment_vars=["mlp"],
    controls=[],
    fe_vars=["strata_int"],
    data=df_single,
    vcov={"CRV1": "fid_int"},
    sample_desc="Singleton births, mort30",
    controls_desc="none",
    fe_desc="strata",
    cluster_var="fid",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/outcome/mort30", "outcome_change": "mort30 + singleton only"}
)

# doctor_only + exclude multiple
run_spec(
    spec_id="rc/data/treatment/doctor_only__singleton",
    spec_tree_path="modules/robustness/data_construction.md#treatment",
    baseline_group_id="G1",
    outcome_var="mort7",
    treatment_var="doctor",
    additional_treatment_vars=[],
    controls=[],
    fe_vars=["strata_int"],
    data=df_single,
    vcov={"CRV1": "fid_int"},
    sample_desc="Singleton births, doctor only",
    controls_desc="none",
    fe_desc="strata",
    cluster_var="fid",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/treatment/doctor_only",
                "treatment_change": "doctor only + singleton"}
)

# Extended controls + mort30
run_spec(
    spec_id="rc/data/outcome/mort30__extended",
    spec_tree_path="modules/robustness/data_construction.md#outcome",
    baseline_group_id="G1",
    outcome_var="mort30",
    treatment_var="doctor",
    additional_treatment_vars=["mlp"],
    controls=cont_base + cont_hc,
    fe_vars=["strata_int", "qtr_int"],
    data=df,
    vcov={"CRV1": "fid_int"},
    sample_desc="Live births, all arms, mort30",
    controls_desc="cont_base + cont_hc",
    fe_desc="strata + qtr",
    cluster_var="fid",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/outcome/mort30", "outcome_change": "mort30 + extended controls"}
)

# Doctor only + basic controls + singleton
run_spec(
    spec_id="rc/data/treatment/doctor_only__basic_singleton",
    spec_tree_path="modules/robustness/data_construction.md#treatment",
    baseline_group_id="G1",
    outcome_var="mort7",
    treatment_var="doctor",
    additional_treatment_vars=[],
    controls=cont_base,
    fe_vars=["strata_int", "qtr_int"],
    data=df_single,
    vcov={"CRV1": "fid_int"},
    sample_desc="Singleton births, doctor only, basic controls",
    controls_desc="cont_base",
    fe_desc="strata + qtr",
    cluster_var="fid",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/treatment/doctor_only",
                "treatment_change": "doctor only + basic controls + singleton"}
)

# Basic controls + singleton
run_spec(
    spec_id="rc/sample/restriction/exclude_multiple__basic",
    spec_tree_path="modules/robustness/sample.md#restriction",
    baseline_group_id="G1",
    outcome_var="mort7",
    treatment_var="doctor",
    additional_treatment_vars=["mlp"],
    controls=cont_base,
    fe_vars=["strata_int", "qtr_int"],
    data=df_single,
    vcov={"CRV1": "fid_int"},
    sample_desc="Singleton births, basic controls",
    controls_desc="cont_base",
    fe_desc="strata + qtr",
    cluster_var="fid",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/exclude_multiple_births",
                "restriction": "multiple==0 + basic controls"}
)

# ============================================================
# INFERENCE VARIANTS on baseline
# ============================================================
print("Running inference variants...")

# HC1 robust SEs
run_inference_variant(
    base_run_id=baseline_run_id,
    spec_id="infer/se/hc/hc1",
    spec_tree_path="modules/inference/standard_errors.md#hc1",
    baseline_group_id="G1",
    outcome_var="mort7",
    treatment_var="doctor",
    additional_treatment_vars=["mlp"],
    controls=[],
    fe_vars=["strata_int"],
    data=df,
    vcov_new="hetero",
    cluster_var_new=""
)

# Cluster at strata level
run_inference_variant(
    base_run_id=baseline_run_id,
    spec_id="infer/se/cluster/strata",
    spec_tree_path="modules/inference/standard_errors.md#cluster",
    baseline_group_id="G1",
    outcome_var="mort7",
    treatment_var="doctor",
    additional_treatment_vars=["mlp"],
    controls=[],
    fe_vars=["strata_int"],
    data=df,
    vcov_new={"CRV1": "strata_int"},
    cluster_var_new="strata"
)

# ============================================================
# SAVE OUTPUTS
# ============================================================
print(f"\nTotal specification rows: {len(results)}")
print(f"Total inference rows: {len(inference_results)}")

# specification_results.csv
df_results = pd.DataFrame(results)
df_results.to_csv(f"{OUT_DIR}/specification_results.csv", index=False)
print(f"Wrote {OUT_DIR}/specification_results.csv")

# inference_results.csv
if inference_results:
    df_infer = pd.DataFrame(inference_results)
    df_infer.to_csv(f"{OUT_DIR}/inference_results.csv", index=False)
    print(f"Wrote {OUT_DIR}/inference_results.csv")

# Count successes/failures
n_success = sum(1 for r in results if r["run_success"] == 1)
n_fail = sum(1 for r in results if r["run_success"] == 0)
n_infer_success = sum(1 for r in inference_results if r["run_success"] == 1)
n_infer_fail = sum(1 for r in inference_results if r["run_success"] == 0)

# ============================================================
# SPECIFICATION_SEARCH.md
# ============================================================
md = f"""# Specification Search Report: {PAPER_ID}

## Paper
"When a Doctor Falls from the Sky: The Impact of Easing Doctor Supply Constraints on Mortality" (Okeke)

## Surface Summary
- **Paper ID**: {PAPER_ID}
- **Baseline groups**: 1 (G1)
- **Design code**: randomized_experiment
- **Baseline outcome**: mort7 (7-day neonatal mortality)
- **Treatment**: doctor (Doctor Arm assignment), with mlp (MLP Arm) included
- **Canonical inference**: cluster SEs at fid (health facility level)
- **Budget**: max 80 core specs, 10 control subset draws
- **Seed**: 181581

## Execution Summary
- **Total specification rows**: {len(results)}
- **Successful**: {n_success}
- **Failed**: {n_fail}
- **Inference variant rows**: {len(inference_results)}
- **Inference successful**: {n_infer_success}
- **Inference failed**: {n_infer_fail}

## Specs Executed

### Baselines (2)
- `baseline`: Table 4 Col 1 (mort7 ~ mlp + doctor | strata, cl(fid))
- `baseline__basic_controls`: Table 4 Col 2 (+ cont_base + qtr FE)

### Design variants (1)
- `design/randomized_experiment/estimator/diff_in_means`: No FE, simple regression

### RC: Control sets (3)
- `rc/controls/sets/none`: No controls
- `rc/controls/sets/basic`: cont_base (10 vars)
- `rc/controls/sets/extended`: cont_base + cont_hc (14+ vars)

### RC: Control progression (4)
- `rc/controls/progression/strata_only`
- `rc/controls/progression/strata_qtr`
- `rc/controls/progression/strata_qtr_individual`
- `rc/controls/progression/strata_qtr_individual_hc`

### RC: LOO controls (12)
- Drop each of: male, first, hausa, gest, car, last, auton, cct, hc_deliveries, hc_cesarean, hc_transfusion, hc_clean

### RC: Subset controls (10)
- 10 random draws from control pool (seed=181581)

### RC: Fixed effects (2)
- `rc/fe/add/qtr`: Add quarter FE to strata-only baseline
- `rc/fe/drop/qtr`: Drop quarter FE from basic-controls spec

### RC: Sample restrictions (4)
- `rc/sample/restriction/exclude_multiple_births`: Singleton births only
- Cross-combinations with extended controls, basic controls, mort30

### RC: Outcome definition (4)
- `rc/data/outcome/mort30`: 30-day mortality (no controls, basic, extended, singleton)

### RC: Treatment definition (4)
- `rc/data/treatment/doctor_only`: Drop mlp from regression (various control combos)

### Inference variants (2)
- `infer/se/hc/hc1`: HC1 robust SEs
- `infer/se/cluster/strata`: Cluster at strata level

## Software Stack
- Python {SW_BLOCK.get('runner_version', 'N/A')}
- pyfixest {SW_BLOCK.get('packages', dict()).get('pyfixest', 'N/A')}
- pandas {SW_BLOCK.get('packages', dict()).get('pandas', 'N/A')}
- numpy {SW_BLOCK.get('packages', dict()).get('numpy', 'N/A')}

## Data Notes
- Used pre-constructed `data/analysis/child.dta` (N={len(df)})
- Variable name fixes: autonomy->auton, hc_drugs->hc_drug, hc_nopower->hc_nopow
- Factor variables (magedum, mschool, hc_clean, hc_cond) expanded to dummies
- qtr (datetime) converted to integer for FE absorption

## Deviations from Surface
- Double-lasso (dsregress, Table 4 Col 4) is not replicated because it requires Stata 17.
  The core surface explicitly excludes this.
- Exploration specs (high_dose, low_dose, doctor_arm_only) were not executed per surface instructions.
"""

with open(f"{OUT_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write(md)
print(f"Wrote {OUT_DIR}/SPECIFICATION_SEARCH.md")

print("\nDone with 181581-V1!")
