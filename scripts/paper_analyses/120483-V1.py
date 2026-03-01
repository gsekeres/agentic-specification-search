"""
Specification Search Script for Malaria Stability and Slavery
Paper ID: 120483-V1

Surface-driven execution:
  - G1: slaveratio ~ MAL + controls | state_g
  - Cross-sectional OLS with state FE
  - Canonical inference: state-clustered SEs (Conley spatial not available in pyfixest)
  - 1860 dataset as primary baseline (Table 1 Col 5)

Outputs:
  - specification_results.csv (baseline, rc/* rows)
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
    error_details_from_exception, surface_hash as compute_surface_hash,
    software_block
)

PAPER_ID = "120483-V1"
DATA_DIR = "data/downloads/extracted/120483-V1"
DTA_DIR = f"{DATA_DIR}/AEJApp-2019-0372/dta"
OUTPUT_DIR = DATA_DIR

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = compute_surface_hash(surface_obj)
SW_BLOCK = software_block()

# Load datasets
df_1860 = pd.read_stata(f"{DTA_DIR}/county_1860.dta")
df_1790 = pd.read_stata(f"{DTA_DIR}/county_1790.dta")

# Convert float32 columns to float64
for d in [df_1860, df_1790]:
    for col in d.columns:
        if d[col].dtype == np.float32:
            d[col] = d[col].astype(np.float64)

# Variable groups from the do file
crop_1790 = ["cotton", "rice", "sugar", "tea", "tobacco", "indigo"]
geo_1790 = ["DISTSEA", "DISTRIV", "prec", "temp", "ELEV", "lat_deg", "long_deg", "lat_long"]
crop_1860 = ["cotton", "coffee", "rice", "sugar", "tea", "tobacco", "indigo"]
geo_1860 = ["ELEV", "prec", "temp", "DISTRIV", "DISTSEA", "lat_deg", "long_deg", "lat_long"]
all_controls_1860 = crop_1860 + geo_1860
all_controls_1790 = crop_1790 + geo_1790

design_audit = surface_obj["baseline_groups"][0]["design_audit"]
canonical_inference = surface_obj["baseline_groups"][0]["inference_plan"]["canonical"]
inference_variants = surface_obj["baseline_groups"][0]["inference_plan"]["variants"]

results = []
inference_results = []
spec_run_counter = 0
inference_run_counter = 0


def run_ols_spec(spec_id, spec_tree_path, outcome_var, treatment_var,
                 controls, fe_list, data, vcov, sample_desc, controls_desc,
                 cluster_var="state_g", fixed_effects_str="state_g",
                 axis_block_name=None, axis_block=None, notes="",
                 weights=None):
    """Run a single OLS specification and record results."""
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        controls_str = " + ".join(controls) if controls else ""
        fe_formula = " + ".join(fe_list) if fe_list else ""

        if controls_str and fe_formula:
            formula = f"{outcome_var} ~ {treatment_var} + {controls_str} | {fe_formula}"
        elif controls_str:
            formula = f"{outcome_var} ~ {treatment_var} + {controls_str}"
        elif fe_formula:
            formula = f"{outcome_var} ~ {treatment_var} | {fe_formula}"
        else:
            formula = f"{outcome_var} ~ {treatment_var}"

        # Drop rows with NaN in regression variables
        reg_vars = [outcome_var, treatment_var] + controls + fe_list
        if weights:
            reg_vars.append(weights)
        reg_vars = [v for v in reg_vars if v in data.columns]
        df_reg = data.dropna(subset=reg_vars).copy()

        kwargs = {"data": df_reg, "vcov": vcov}
        if weights and weights in df_reg.columns:
            kwargs["weights"] = weights

        m = pf.feols(formula, **kwargs)

        coef_val = float(m.coef().get(treatment_var, np.nan))
        se_val = float(m.se().get(treatment_var, np.nan))
        pval = float(m.pvalue().get(treatment_var, np.nan))

        try:
            ci = m.confint()
            ci_lower = float(ci.loc[treatment_var, ci.columns[0]])
            ci_upper = float(ci.loc[treatment_var, ci.columns[1]])
        except Exception:
            ci_lower = np.nan
            ci_upper = np.nan

        n_obs = int(m._N)
        r2 = float(m._r2)

        all_coefs = {k: float(v) for k, v in m.coef().items()}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": "infer/se/cluster/state",
                       "params": {"cluster_var": "state_g"}},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"cross_sectional_ols": {
                "estimator": "ols",
                "fe_structure": fe_list,
                "cluster_vars": [cluster_var] if cluster_var else [],
                "selection_story": design_audit.get("selection_story", "")
            }},
            axis_block_name=axis_block_name,
            axis_block=axis_block,
            notes=notes
        )

        row = {
            "paper_id": PAPER_ID,
            "spec_run_id": run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": "G1",
            "outcome_var": outcome_var,
            "treatment_var": treatment_var,
            "coefficient": coef_val,
            "std_error": se_val,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": n_obs,
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": fixed_effects_str,
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
            "run_success": 1,
            "run_error": ""
        }
        results.append(row)
        return m, run_id

    except Exception as e:
        err_msg = str(e)[:240]
        payload = make_failure_payload(
            error=err_msg,
            error_details=error_details_from_exception(e, stage="estimation"),
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH
        )
        row = {
            "paper_id": PAPER_ID,
            "spec_run_id": run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": "G1",
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
            "fixed_effects": fixed_effects_str,
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
            "run_success": 0,
            "run_error": err_msg
        }
        results.append(row)
        return None, run_id


def add_inference_row(base_run_id, spec_id, spec_tree_path, model_or_data,
                      outcome_var, treatment_var, controls, fe_list, data,
                      vcov, notes=""):
    """Run inference variant for a given base specification."""
    global inference_run_counter
    inference_run_counter += 1
    inf_id = f"{PAPER_ID}_infer_{inference_run_counter:03d}"

    try:
        controls_str = " + ".join(controls) if controls else ""
        fe_formula = " + ".join(fe_list) if fe_list else ""

        if controls_str and fe_formula:
            formula = f"{outcome_var} ~ {treatment_var} + {controls_str} | {fe_formula}"
        elif controls_str:
            formula = f"{outcome_var} ~ {treatment_var} + {controls_str}"
        elif fe_formula:
            formula = f"{outcome_var} ~ {treatment_var} | {fe_formula}"
        else:
            formula = f"{outcome_var} ~ {treatment_var}"

        reg_vars = [outcome_var, treatment_var] + controls + fe_list
        reg_vars = [v for v in reg_vars if v in data.columns]
        df_reg = data.dropna(subset=reg_vars).copy()

        m = pf.feols(formula, data=df_reg, vcov=vcov)

        coef_val = float(m.coef().get(treatment_var, np.nan))
        se_val = float(m.se().get(treatment_var, np.nan))
        pval = float(m.pvalue().get(treatment_var, np.nan))

        try:
            ci = m.confint()
            ci_lower = float(ci.loc[treatment_var, ci.columns[0]])
            ci_upper = float(ci.loc[treatment_var, ci.columns[1]])
        except Exception:
            ci_lower = np.nan
            ci_upper = np.nan

        n_obs = int(m._N)
        r2 = float(m._r2)

        payload = {
            "coefficients": {k: float(v) for k, v in m.coef().items()},
            "inference": {"spec_id": spec_id, "params": {}},
            "software": SW_BLOCK,
            "surface_hash": SURFACE_HASH,
        }

        row = {
            "paper_id": PAPER_ID,
            "inference_run_id": inf_id,
            "spec_run_id": base_run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": "G1",
            "coefficient": coef_val,
            "std_error": se_val,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": n_obs,
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 1,
            "run_error": ""
        }
        inference_results.append(row)

    except Exception as e:
        err_msg = str(e)[:240]
        payload = make_failure_payload(
            error=err_msg,
            error_details=error_details_from_exception(e, stage="inference_recomputation"),
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH
        )
        row = {
            "paper_id": PAPER_ID,
            "inference_run_id": inf_id,
            "spec_run_id": base_run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": "G1",
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
        }
        inference_results.append(row)


# ============================================================================
# BASELINE: Table 1 Col 5 (1860, all states, full controls)
# Note: Paper uses Conley SEs, but pyfixest does not support spatial SEs.
# We use state-clustered SEs as the canonical inference (per surface note).
# ============================================================================

print("=== Running baseline: Table1-Col5 (1860, full controls, state-clustered) ===")
m_baseline, baseline_run_id = run_ols_spec(
    spec_id="baseline__table1_col5",
    spec_tree_path="specification_tree/methods/cross_sectional_ols.md",
    outcome_var="slaveratio",
    treatment_var="MAL",
    controls=all_controls_1860,
    fe_list=["state_g"],
    data=df_1860,
    vcov={"CRV1": "state_g"},
    sample_desc="US counties 1860, all states",
    controls_desc="crop_suitability + geography (full 1860 controls)",
    notes="Baseline specification Table 1 Col 5. State-clustered SEs used as canonical (Conley unavailable in pyfixest)."
)

# Run inference variants for baseline
add_inference_row(baseline_run_id, "infer/se/hc/hc1",
                  "specification_tree/modules/inference/se/hc.md#hc1",
                  None, "slaveratio", "MAL", all_controls_1860, ["state_g"],
                  df_1860, "hetero",
                  notes="HC1 robust SEs for baseline")

# ============================================================================
# LOO CONTROL DROPS (15 specs)
# ============================================================================
print("=== Running LOO control drop specs ===")

loo_controls = [
    ("cotton", "rc/controls/loo/drop_cotton"),
    ("coffee", "rc/controls/loo/drop_coffee"),
    ("rice", "rc/controls/loo/drop_rice"),
    ("sugar", "rc/controls/loo/drop_sugar"),
    ("tea", "rc/controls/loo/drop_tea"),
    ("tobacco", "rc/controls/loo/drop_tobacco"),
    ("indigo", "rc/controls/loo/drop_indigo"),
    ("ELEV", "rc/controls/loo/drop_ELEV"),
    ("prec", "rc/controls/loo/drop_prec"),
    ("temp", "rc/controls/loo/drop_temp"),
    ("DISTRIV", "rc/controls/loo/drop_DISTRIV"),
    ("DISTSEA", "rc/controls/loo/drop_DISTSEA"),
    ("lat_deg", "rc/controls/loo/drop_lat_deg"),
    ("long_deg", "rc/controls/loo/drop_long_deg"),
    ("lat_long", "rc/controls/loo/drop_lat_long"),
]

for drop_var, spec_id in loo_controls:
    ctrl_remaining = [c for c in all_controls_1860 if c != drop_var]
    run_ols_spec(
        spec_id=spec_id,
        spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
        outcome_var="slaveratio",
        treatment_var="MAL",
        controls=ctrl_remaining,
        fe_list=["state_g"],
        data=df_1860,
        vcov={"CRV1": "state_g"},
        sample_desc="US counties 1860, all states",
        controls_desc=f"full 1860 controls minus {drop_var}",
        axis_block_name="controls",
        axis_block={"spec_id": spec_id, "family": "loo",
                    "dropped": [drop_var], "n_controls": len(ctrl_remaining)},
        notes=f"LOO: drop {drop_var}"
    )

# ============================================================================
# CONTROL SETS (none, crop_only, geo_only)
# ============================================================================
print("=== Running control set variants ===")

# No controls
run_ols_spec(
    spec_id="rc/controls/sets/none",
    spec_tree_path="specification_tree/modules/robustness/controls.md#sets",
    outcome_var="slaveratio",
    treatment_var="MAL",
    controls=[],
    fe_list=["state_g"],
    data=df_1860,
    vcov={"CRV1": "state_g"},
    sample_desc="US counties 1860, all states",
    controls_desc="no controls (MAL + state FE only)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/none", "family": "sets",
                "set_name": "none", "n_controls": 0}
)

# Crop only
run_ols_spec(
    spec_id="rc/controls/sets/crop_only",
    spec_tree_path="specification_tree/modules/robustness/controls.md#sets",
    outcome_var="slaveratio",
    treatment_var="MAL",
    controls=crop_1860,
    fe_list=["state_g"],
    data=df_1860,
    vcov={"CRV1": "state_g"},
    sample_desc="US counties 1860, all states",
    controls_desc="crop suitability controls only",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/crop_only", "family": "sets",
                "set_name": "crop_only", "n_controls": len(crop_1860),
                "controls": crop_1860}
)

# Geo only
run_ols_spec(
    spec_id="rc/controls/sets/geo_only",
    spec_tree_path="specification_tree/modules/robustness/controls.md#sets",
    outcome_var="slaveratio",
    treatment_var="MAL",
    controls=geo_1860,
    fe_list=["state_g"],
    data=df_1860,
    vcov={"CRV1": "state_g"},
    sample_desc="US counties 1860, all states",
    controls_desc="geography controls only",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/geo_only", "family": "sets",
                "set_name": "geo_only", "n_controls": len(geo_1860),
                "controls": geo_1860}
)

# ============================================================================
# CONTROL PROGRESSION (crop_suitability, crop_and_geo)
# ============================================================================
print("=== Running control progression specs ===")

run_ols_spec(
    spec_id="rc/controls/progression/crop_suitability",
    spec_tree_path="specification_tree/modules/robustness/controls.md#progression",
    outcome_var="slaveratio",
    treatment_var="MAL",
    controls=crop_1860,
    fe_list=["state_g"],
    data=df_1860,
    vcov={"CRV1": "state_g"},
    sample_desc="US counties 1860, all states",
    controls_desc="progressive: crop suitability only",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/crop_suitability",
                "family": "progression", "step": 1, "n_controls": len(crop_1860)}
)

run_ols_spec(
    spec_id="rc/controls/progression/crop_and_geo",
    spec_tree_path="specification_tree/modules/robustness/controls.md#progression",
    outcome_var="slaveratio",
    treatment_var="MAL",
    controls=crop_1860 + geo_1860,
    fe_list=["state_g"],
    data=df_1860,
    vcov={"CRV1": "state_g"},
    sample_desc="US counties 1860, all states",
    controls_desc="progressive: crop + geography (full)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/crop_and_geo",
                "family": "progression", "step": 2,
                "n_controls": len(crop_1860 + geo_1860)}
)

# ============================================================================
# RANDOM CONTROL SUBSETS (15 draws)
# ============================================================================
print("=== Running random control subset specs ===")

rng = random.Random(120483)
optional_controls = all_controls_1860[:]  # all are optional

for i in range(1, 16):
    # Draw random size between 1 and len(optional)-1
    size = rng.randint(1, len(optional_controls) - 1)
    subset = sorted(rng.sample(optional_controls, size))
    spec_id = f"rc/controls/subset/random_{i:03d}"
    run_ols_spec(
        spec_id=spec_id,
        spec_tree_path="specification_tree/modules/robustness/controls.md#subset",
        outcome_var="slaveratio",
        treatment_var="MAL",
        controls=subset,
        fe_list=["state_g"],
        data=df_1860,
        vcov={"CRV1": "state_g"},
        sample_desc="US counties 1860, all states",
        controls_desc=f"random subset draw {i}: {len(subset)} controls",
        axis_block_name="controls",
        axis_block={"spec_id": spec_id, "family": "subset",
                    "draw_index": i, "n_controls": len(subset),
                    "included": subset, "seed": 120483}
    )

# ============================================================================
# SAMPLE RESTRICTIONS
# ============================================================================
print("=== Running sample restriction specs ===")

# Slave states only (1860)
run_ols_spec(
    spec_id="rc/sample/restrict/slave_states_only",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restrict",
    outcome_var="slaveratio",
    treatment_var="MAL",
    controls=all_controls_1860,
    fe_list=["state_g"],
    data=df_1860[df_1860["slave_state"] == 1].copy(),
    vcov={"CRV1": "state_g"},
    sample_desc="US counties 1860, slave states only",
    controls_desc="full 1860 controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restrict/slave_states_only",
                "restriction": "slave_state==1"}
)

# 1790 data (full controls)
run_ols_spec(
    spec_id="rc/sample/restrict/1790_data",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restrict",
    outcome_var="slaveratio",
    treatment_var="MAL",
    controls=all_controls_1790,
    fe_list=["state_g"],
    data=df_1790,
    vcov={"CRV1": "state_g"},
    sample_desc="US counties 1790, all states",
    controls_desc="full 1790 controls (no coffee)",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restrict/1790_data",
                "restriction": "1790_dataset", "dataset": "county_1790.dta"}
)

# Trim outcome 1/99
print("=== Running outlier trimming specs ===")
p1 = df_1860["slaveratio"].quantile(0.01)
p99 = df_1860["slaveratio"].quantile(0.99)
df_trimmed_1_99 = df_1860[(df_1860["slaveratio"] >= p1) & (df_1860["slaveratio"] <= p99)].copy()

run_ols_spec(
    spec_id="rc/sample/outliers/trim_y_1_99",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    outcome_var="slaveratio",
    treatment_var="MAL",
    controls=all_controls_1860,
    fe_list=["state_g"],
    data=df_trimmed_1_99,
    vcov={"CRV1": "state_g"},
    sample_desc="US counties 1860, trimmed 1-99 pct",
    controls_desc="full 1860 controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_1_99",
                "trim_lower": 0.01, "trim_upper": 0.99}
)

# Trim outcome 5/95
p5 = df_1860["slaveratio"].quantile(0.05)
p95 = df_1860["slaveratio"].quantile(0.95)
df_trimmed_5_95 = df_1860[(df_1860["slaveratio"] >= p5) & (df_1860["slaveratio"] <= p95)].copy()

run_ols_spec(
    spec_id="rc/sample/outliers/trim_y_5_95",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    outcome_var="slaveratio",
    treatment_var="MAL",
    controls=all_controls_1860,
    fe_list=["state_g"],
    data=df_trimmed_5_95,
    vcov={"CRV1": "state_g"},
    sample_desc="US counties 1860, trimmed 5-95 pct",
    controls_desc="full 1860 controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_5_95",
                "trim_lower": 0.05, "trim_upper": 0.95}
)

# ============================================================================
# FE VARIANTS
# ============================================================================
print("=== Running FE variant specs ===")

# Drop state FE
run_ols_spec(
    spec_id="rc/fe/drop/state_g",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#drop",
    outcome_var="slaveratio",
    treatment_var="MAL",
    controls=all_controls_1860,
    fe_list=[],
    data=df_1860,
    vcov="hetero",
    sample_desc="US counties 1860, all states",
    controls_desc="full 1860 controls",
    cluster_var="",
    fixed_effects_str="none",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/drop/state_g", "dropped": ["state_g"]}
)

# ============================================================================
# FUNCTIONAL FORM VARIANTS
# ============================================================================
print("=== Running functional form specs ===")

# asinh(slaveratio)
df_1860_form = df_1860.copy()
df_1860_form["asinh_slaveratio"] = np.arcsinh(df_1860_form["slaveratio"])

run_ols_spec(
    spec_id="rc/form/outcome/asinh",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    outcome_var="asinh_slaveratio",
    treatment_var="MAL",
    controls=all_controls_1860,
    fe_list=["state_g"],
    data=df_1860_form,
    vcov={"CRV1": "state_g"},
    sample_desc="US counties 1860, all states",
    controls_desc="full 1860 controls",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/asinh",
                "transform": "asinh", "original_var": "slaveratio",
                "interpretation": "asinh(slave share); approximate pct change interpretation for small values"}
)

# log1p(slaveratio)
df_1860_form["log1p_slaveratio"] = np.log1p(df_1860_form["slaveratio"])

run_ols_spec(
    spec_id="rc/form/outcome/log1p",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    outcome_var="log1p_slaveratio",
    treatment_var="MAL",
    controls=all_controls_1860,
    fe_list=["state_g"],
    data=df_1860_form,
    vcov={"CRV1": "state_g"},
    sample_desc="US counties 1860, all states",
    controls_desc="full 1860 controls",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/log1p",
                "transform": "log1p", "original_var": "slaveratio",
                "interpretation": "log(1+slave share); approximate pct change interpretation"}
)

# ============================================================================
# ADDITIONAL SPECIFICATIONS: Slave states with control variations
# ============================================================================
print("=== Running slave-states + control variation specs ===")

# Slave states, no controls
run_ols_spec(
    spec_id="rc/sample/restrict/slave_states_no_controls",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restrict",
    outcome_var="slaveratio",
    treatment_var="MAL",
    controls=[],
    fe_list=["state_g"],
    data=df_1860[df_1860["slave_state"] == 1].copy(),
    vcov={"CRV1": "state_g"},
    sample_desc="US counties 1860, slave states only, no controls",
    controls_desc="no controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restrict/slave_states_no_controls",
                "restriction": "slave_state==1, no controls"}
)

# Slave states, crop only
run_ols_spec(
    spec_id="rc/sample/restrict/slave_states_crop_only",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restrict",
    outcome_var="slaveratio",
    treatment_var="MAL",
    controls=crop_1860,
    fe_list=["state_g"],
    data=df_1860[df_1860["slave_state"] == 1].copy(),
    vcov={"CRV1": "state_g"},
    sample_desc="US counties 1860, slave states only, crop controls",
    controls_desc="crop suitability controls only",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restrict/slave_states_crop_only",
                "restriction": "slave_state==1, crop controls only"}
)

# Slave states, geo only
run_ols_spec(
    spec_id="rc/sample/restrict/slave_states_geo_only",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restrict",
    outcome_var="slaveratio",
    treatment_var="MAL",
    controls=geo_1860,
    fe_list=["state_g"],
    data=df_1860[df_1860["slave_state"] == 1].copy(),
    vcov={"CRV1": "state_g"},
    sample_desc="US counties 1860, slave states only, geo controls",
    controls_desc="geography controls only",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restrict/slave_states_geo_only",
                "restriction": "slave_state==1, geo controls only"}
)

# 1790 slave states only (Table 1 Col 6 analog)
if "slave_state" in df_1790.columns:
    run_ols_spec(
        spec_id="rc/sample/restrict/1790_slave_states",
        spec_tree_path="specification_tree/modules/robustness/sample.md#restrict",
        outcome_var="slaveratio",
        treatment_var="MAL",
        controls=all_controls_1790,
        fe_list=["state_g"],
        data=df_1790[df_1790["slave_state"] == 1].copy(),
        vcov={"CRV1": "state_g"},
        sample_desc="US counties 1790, slave states only",
        controls_desc="full 1790 controls",
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/restrict/1790_slave_states",
                    "restriction": "1790, slave_state==1"}
    )

# 1790 no controls
run_ols_spec(
    spec_id="rc/sample/restrict/1790_no_controls",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restrict",
    outcome_var="slaveratio",
    treatment_var="MAL",
    controls=[],
    fe_list=["state_g"],
    data=df_1790,
    vcov={"CRV1": "state_g"},
    sample_desc="US counties 1790, all states, no controls",
    controls_desc="no controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restrict/1790_no_controls",
                "restriction": "1790_dataset, no controls"}
)

# 1790 crop only
run_ols_spec(
    spec_id="rc/sample/restrict/1790_crop_only",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restrict",
    outcome_var="slaveratio",
    treatment_var="MAL",
    controls=crop_1790,
    fe_list=["state_g"],
    data=df_1790,
    vcov={"CRV1": "state_g"},
    sample_desc="US counties 1790, all states, crop controls",
    controls_desc="crop suitability controls only (1790)",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restrict/1790_crop_only",
                "restriction": "1790_dataset, crop only"}
)

# ============================================================================
# ADDITIONAL FUNCTIONAL FORM: asinh and log1p on 1790 data
# ============================================================================
print("=== Running additional functional form specs ===")

df_1790_form = df_1790.copy()
df_1790_form["asinh_slaveratio"] = np.arcsinh(df_1790_form["slaveratio"])
df_1790_form["log1p_slaveratio"] = np.log1p(df_1790_form["slaveratio"])

run_ols_spec(
    spec_id="rc/form/outcome/asinh_1790",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md",
    outcome_var="asinh_slaveratio",
    treatment_var="MAL",
    controls=all_controls_1790,
    fe_list=["state_g"],
    data=df_1790_form,
    vcov={"CRV1": "state_g"},
    sample_desc="US counties 1790, all states",
    controls_desc="full 1790 controls",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/asinh_1790",
                "transform": "asinh", "original_var": "slaveratio",
                "interpretation": "asinh(slave share) on 1790 data"}
)

# ============================================================================
# ADDITIONAL SPECS: 1860 no FE + slave states, 1790 no FE
# ============================================================================
print("=== Running more FE/sample combination specs ===")

# 1860, no FE, no controls (raw bivariate)
run_ols_spec(
    spec_id="rc/fe/drop/state_g_no_controls",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#drop",
    outcome_var="slaveratio",
    treatment_var="MAL",
    controls=[],
    fe_list=[],
    data=df_1860,
    vcov="hetero",
    sample_desc="US counties 1860, all states",
    controls_desc="no controls, no FE",
    cluster_var="",
    fixed_effects_str="none",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/drop/state_g_no_controls",
                "dropped": ["state_g"], "controls": "none"}
)

# 1860 slave states, no FE, full controls
run_ols_spec(
    spec_id="rc/sample/restrict/slave_states_no_fe",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restrict",
    outcome_var="slaveratio",
    treatment_var="MAL",
    controls=all_controls_1860,
    fe_list=[],
    data=df_1860[df_1860["slave_state"] == 1].copy(),
    vcov="hetero",
    sample_desc="US counties 1860, slave states only, no state FE",
    controls_desc="full 1860 controls",
    cluster_var="",
    fixed_effects_str="none",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restrict/slave_states_no_fe",
                "restriction": "slave_state==1, no FE"}
)

# ============================================================================
# INFERENCE VARIANTS for baseline
# ============================================================================
print("=== Running inference variants ===")
# HC1 already added above after baseline

# ============================================================================
# WRITE OUTPUTS
# ============================================================================
print(f"=== Writing outputs ({len(results)} estimate rows, {len(inference_results)} inference rows) ===")

df_results = pd.DataFrame(results)
df_results.to_csv(f"{OUTPUT_DIR}/specification_results.csv", index=False)

df_inference = pd.DataFrame(inference_results)
df_inference.to_csv(f"{OUTPUT_DIR}/inference_results.csv", index=False)

# ============================================================================
# SPECIFICATION_SEARCH.md
# ============================================================================
n_success = df_results["run_success"].sum()
n_fail = len(df_results) - n_success

search_md = f"""# Specification Search Report: {PAPER_ID}

## Paper
- **Title**: Malaria Stability and Slavery
- **Paper ID**: {PAPER_ID}

## Surface Summary
- **Baseline groups**: 1 (G1)
- **Design code**: cross_sectional_ols
- **Baseline spec**: Table 1 Col 5 (1860, all states, full controls)
- **Budget**: max 75 core specs, 15 control subsets
- **Seed**: 120483

## Canonical Inference
- Paper uses Conley spatial SEs (acreg) as primary inference
- **Canonical inference used here**: State-clustered SEs (CRV1) -- Conley spatial SEs not available in pyfixest
- State-clustered SEs are reported as alternative inference in the paper (curly braces)

## Execution Summary
- **Total specifications planned**: {len(results)}
- **Successful**: {n_success}
- **Failed**: {n_fail}
- **Inference variants**: {len(inference_results)}

## Specifications Executed

### Baseline (1 spec)
- `baseline__table1_col5`: 1860 data, full controls, state-clustered SEs

### LOO Control Drops (15 specs)
- Dropped each of the 15 controls in the 1860 specification one at a time

### Control Sets (3 specs)
- `rc/controls/sets/none`: No controls (MAL + state FE only)
- `rc/controls/sets/crop_only`: Crop suitability controls only
- `rc/controls/sets/geo_only`: Geography controls only

### Control Progression (2 specs)
- `rc/controls/progression/crop_suitability`: Add crop controls
- `rc/controls/progression/crop_and_geo`: Add crop + geography (=full)

### Random Control Subsets (15 specs)
- 15 random draws from the control pool, seed=120483

### Sample Restrictions (4 specs)
- `rc/sample/restrict/slave_states_only`: 1860, slave states only
- `rc/sample/restrict/1790_data`: 1790 dataset with 1790 controls
- `rc/sample/outliers/trim_y_1_99`: Trim outcome 1-99 pct
- `rc/sample/outliers/trim_y_5_95`: Trim outcome 5-95 pct

### FE Variants (1 spec)
- `rc/fe/drop/state_g`: Drop state FE, use HC1

### Functional Form (2 specs)
- `rc/form/outcome/asinh`: asinh(slaveratio)
- `rc/form/outcome/log1p`: log1p(slaveratio)

## Inference Results
- `infer/se/hc/hc1`: HC1 for baseline

## Deviations from Surface
- Conley spatial SEs (100km, 250km, 500km) not available in Python/pyfixest. The surface notes that state-clustered SEs are an acceptable canonical fallback.
- Conley inference variants (`infer/se/spatial/conley_100km`, `conley_250km`, `conley_500km`) not implemented.

## Software Stack
- Python {sys.version.split()[0]}
- pyfixest: {SW_BLOCK['packages'].get('pyfixest', 'unknown')}
- pandas: {SW_BLOCK['packages'].get('pandas', 'unknown')}
- numpy: {SW_BLOCK['packages'].get('numpy', 'unknown')}
"""

with open(f"{OUTPUT_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write(search_md)

print(f"Done! {len(results)} specs written to specification_results.csv")
print(f"       {len(inference_results)} inference rows written to inference_results.csv")
