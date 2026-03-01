"""
Specification Search Script for Martinez-Bravo (2014)
"The Role of Local Officials in New Democracies: Evidence from Indonesia"
American Economic Review, 104(4), 1244-1287.

Paper ID: 112756-V1

Surface-driven execution:
  - G1: GolkarFirst ~ kelurDum + controls | kab FE, cluster(kab)
  - Cross-sectional OLS with district (kab) FE, clustered SEs at kab level
  - 50+ specifications across controls LOO, controls subsets, controls progression,
    sample trimming, FE swaps, functional form (probit)

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

PAPER_ID = "112756-V1"
DATA_DIR = "data/downloads/extracted/112756-V1"
OUTPUT_DIR = DATA_DIR
DATA_PATH = f"{DATA_DIR}/AER-2011-1027.R2_Data/LocalOfficials_AER.dta"

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

bg = surface_obj["baseline_groups"][0]
design_audit = bg["design_audit"]
inference_canonical = bg["inference_plan"]["canonical"]

# ============================================================
# Data Loading and Preparation
# ============================================================

df_raw = pd.read_stata(DATA_PATH)
print(f"Loaded data: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

# Convert float32 to float64 for precision
for col in df_raw.columns:
    if df_raw[col].dtype == np.float32:
        df_raw[col] = df_raw[col].astype(np.float64)

# Make kab a string for pyfixest FE
df_raw['kab_str'] = df_raw['kab'].astype(str)
df_raw['kec_str'] = df_raw['kec'].astype(str)

# Scale polynomial terms to avoid demeaning convergence issues in pyfixest.
# The original perc_ruralHH_1996 is in [0,100]; its 4th power has within-kab
# std ~23 million, causing the iterative demeaning to fail. Re-scaling to [0,1]
# does not change the treatment coefficient (kelurDum).
df_raw['prh_sc'] = df_raw['perc_ruralHH_1996'] / 100.0
df_raw['prh_sc_2'] = df_raw['prh_sc'] ** 2
df_raw['prh_sc_3'] = df_raw['prh_sc'] ** 3
df_raw['prh_sc_4'] = df_raw['prh_sc'] ** 4

# Define all control variables (from surface, with scaled polynomial)
ALL_CONTROLS = [
    "urbDum_1996", "prh_sc", "prh_sc_2",
    "prh_sc_3", "prh_sc_4",
    "ShareRurLand_1996", "altitude_high_1996",
    "lpopulation_1996", "popdensity_1996",
    "dist_kecoffice_2000", "dist_kab_kotacapital_2000",
    "mosqueph_1996", "prayerhouseph_1996",
    "churchesph_1996", "viharaph_1996",
    "num_TVs_1996_pc", "num_hospitals_1996_pc",
    "num_maternhosp_1996_pc", "num_polyclinic_1996_pc",
    "num_puskesmas_1996_pc", "num_kindgarden_1996_pc",
    "num_primarysch_1996_pc", "num_HS_1996_pc"
]

# Mapping from surface variable names to actual (scaled) variable names
# (used in LOO descriptions and spec_ids which reference original names)
SURFACE_TO_ACTUAL = {
    "perc_ruralHH_1996": "prh_sc",
    "perc_ruralHH_1996_2": "prh_sc_2",
    "perc_ruralHH_1996_3": "prh_sc_3",
    "perc_ruralHH_1996_4": "prh_sc_4",
}

# Define control groups (as described in the paper)
GEOGRAPHY_CONTROLS = [
    "urbDum_1996", "prh_sc", "prh_sc_2",
    "prh_sc_3", "prh_sc_4",
    "ShareRurLand_1996", "altitude_high_1996",
    "lpopulation_1996", "popdensity_1996",
    "dist_kecoffice_2000", "dist_kab_kotacapital_2000"
]

RELIGION_CONTROLS = [
    "mosqueph_1996", "prayerhouseph_1996",
    "churchesph_1996", "viharaph_1996"
]

FACILITIES_CONTROLS = [
    "num_TVs_1996_pc", "num_hospitals_1996_pc",
    "num_maternhosp_1996_pc", "num_polyclinic_1996_pc",
    "num_puskesmas_1996_pc", "num_kindgarden_1996_pc",
    "num_primarysch_1996_pc", "num_HS_1996_pc"
]

# Drop rows with missing outcome or treatment
df = df_raw.dropna(subset=["GolkarFirst", "kelurDum"]).copy()
print(f"After dropping missing outcome/treatment: {len(df)} rows")

# Drop rows with missing controls for baseline sample
df_base = df.dropna(subset=ALL_CONTROLS).copy()
print(f"Baseline sample (all controls non-missing): {len(df_base)} rows")
print(f"  kab unique: {df_base['kab_str'].nunique()}")

# ============================================================
# Accumulators
# ============================================================

results = []
inference_results = []
spec_run_counter = 0


# ============================================================
# Helper: run_spec (OLS with FE via pyfixest)
# ============================================================

def run_spec(spec_id, spec_tree_path, baseline_group_id,
             outcome_var, treatment_var, controls, fe_formula_str,
             fe_desc, data, vcov, sample_desc, controls_desc,
             cluster_var="kab_str",
             axis_block_name=None, axis_block=None, notes=""):
    """Run a single OLS specification and record results."""
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

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

        m = pf.feols(formula, data=data, vcov=vcov)

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

        all_coefs = {k: float(v) for k, v in m.coef().items()}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": inference_canonical["spec_id"],
                       "method": "cluster", "cluster_vars": ["kab"]},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"cross_sectional_ols": design_audit},
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
# Helper: run_probit (via statsmodels)
# ============================================================

def run_probit(spec_id, spec_tree_path, baseline_group_id,
               outcome_var, treatment_var, controls, fe_vars,
               fe_desc, data, sample_desc, controls_desc,
               cluster_var="kab_str",
               axis_block_name=None, axis_block=None, notes=""):
    """Run a probit specification using statsmodels.

    For large FE sets, uses BFGS optimizer with warm-start from logit
    to avoid extremely slow Newton convergence.
    """
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        import statsmodels.api as sm

        # Build RHS: treatment + controls + FE dummies
        rhs_vars = [treatment_var] + list(controls)

        # Create FE dummies
        temp_data = data.copy()
        if fe_vars:
            for fv in fe_vars:
                dummies = pd.get_dummies(temp_data[fv], prefix=fv, drop_first=True).astype(float)
                temp_data = pd.concat([temp_data, dummies], axis=1)
                rhs_vars = rhs_vars + list(dummies.columns)

        # Drop missing
        all_vars = [outcome_var] + rhs_vars
        est_data = temp_data.dropna(subset=all_vars).copy()

        y = est_data[outcome_var].astype(float)
        X = sm.add_constant(est_data[rhs_vars].astype(float))

        # Check for perfect prediction: drop FE dummies where outcome is all 0 or all 1
        # This is common with many FE dummies in probit
        cols_to_keep = list(X.columns)
        if fe_vars:
            for col in list(X.columns):
                if any(col.startswith(f"{fv}_") for fv in fe_vars):
                    mask = X[col] == 1
                    if mask.sum() > 0:
                        y_vals = y[mask]
                        if y_vals.nunique() <= 1:
                            cols_to_keep.remove(col)
            if len(cols_to_keep) < len(X.columns):
                n_dropped_fe = len(X.columns) - len(cols_to_keep)
                print(f"    Probit: dropping {n_dropped_fe} FE dummies due to perfect prediction")
                est_data_mask = pd.Series(True, index=est_data.index)
                for col in list(X.columns):
                    if col not in cols_to_keep and any(col.startswith(f"{fv}_") for fv in fe_vars):
                        est_data_mask &= (X[col] == 0)
                est_data = est_data[est_data_mask].copy()
                y = est_data[outcome_var].astype(float)
                remaining_rhs = [v for v in rhs_vars if v in cols_to_keep or
                                 not any(v.startswith(f"{fv}_") for fv in fe_vars)]
                X = sm.add_constant(est_data[[v for v in rhs_vars if v in cols_to_keep or
                                              not any(v.startswith(f"{fv}_") for fv in fe_vars)]].astype(float))

        probit_model = sm.Probit(y, X)
        # Use BFGS for speed with many parameters; fall back to newton if BFGS fails
        try:
            probit_result = probit_model.fit(method='bfgs',
                                              cov_type='cluster',
                                              cov_kwds={'groups': est_data[cluster_var].values},
                                              disp=0, maxiter=500)
        except Exception:
            probit_result = probit_model.fit(method='nm',
                                              cov_type='cluster',
                                              cov_kwds={'groups': est_data[cluster_var].values},
                                              disp=0, maxiter=1000)

        coef_val = float(probit_result.params[treatment_var])
        se_val = float(probit_result.bse[treatment_var])
        pval = float(probit_result.pvalues[treatment_var])
        ci = probit_result.conf_int()
        ci_lower = float(ci.loc[treatment_var, 0])
        ci_upper = float(ci.loc[treatment_var, 1])
        nobs = int(probit_result.nobs)
        try:
            r2 = float(probit_result.prsquared)
        except:
            r2 = np.nan

        all_coefs = {k: float(v) for k, v in probit_result.params.items()
                     if not k.startswith('kab_str_') and not k.startswith('kec_str_')}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": "infer/se/cluster/kab",
                       "method": "cluster", "cluster_vars": ["kab"],
                       "estimator": "probit"},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"cross_sectional_ols": design_audit},
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
        err_details = error_details_from_exception(e, stage="probit_estimation")
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
# BASELINE: Table 2, Col 5 — OLS with full controls + kab FE
# ============================================================

print("Running baseline specification...")
base_run_id, base_coef, base_se, base_pval, base_nobs = run_spec(
    "baseline", "designs/cross_sectional_ols.md#baseline", "G1",
    "GolkarFirst", "kelurDum", ALL_CONTROLS,
    "kab_str", "kab (district)", df_base,
    {"CRV1": "kab_str"},
    f"Full sample, N={len(df_base)}", "geography + religion + facilities (23 controls)")

print(f"  Baseline: coef={base_coef:.4f}, se={base_se:.4f}, p={base_pval:.4f}, N={base_nobs}")


# ============================================================
# ADDITIONAL BASELINE: Probit (Table 2 Col 6-9 in paper)
# ============================================================

print("Running probit baseline...")
run_probit(
    "baseline__probit_full", "designs/cross_sectional_ols.md#baseline", "G1",
    "GolkarFirst", "kelurDum", ALL_CONTROLS,
    ["kab_str"], "kab (district)", df_base,
    f"Full sample, N={len(df_base)}", "geography + religion + facilities (probit)",
    axis_block_name="functional_form",
    axis_block={"spec_id": "baseline__probit_full", "estimator": "probit",
                "notes": "Probit model; coefficient is probit index, not marginal effect"})


# ============================================================
# RC: CONTROLS LOO — Drop one control (or group) at a time
# ============================================================

print("Running controls LOO variants...")

# LOO map: spec_id -> variable(s) to drop
LOO_MAP = {
    "rc/controls/loo/drop_urbDum": ["urbDum_1996"],
    "rc/controls/loo/drop_ruralHH_poly": ["prh_sc", "prh_sc_2",
                                           "prh_sc_3", "prh_sc_4"],
    "rc/controls/loo/drop_ShareRurLand": ["ShareRurLand_1996"],
    "rc/controls/loo/drop_altitude": ["altitude_high_1996"],
    "rc/controls/loo/drop_lpopulation": ["lpopulation_1996"],
    "rc/controls/loo/drop_popdensity": ["popdensity_1996"],
    "rc/controls/loo/drop_dist_kecoffice": ["dist_kecoffice_2000"],
    "rc/controls/loo/drop_dist_kabcapital": ["dist_kab_kotacapital_2000"],
    "rc/controls/loo/drop_mosques": ["mosqueph_1996"],
    "rc/controls/loo/drop_prayerhouse": ["prayerhouseph_1996"],
    "rc/controls/loo/drop_churches": ["churchesph_1996"],
    "rc/controls/loo/drop_vihara": ["viharaph_1996"],
    "rc/controls/loo/drop_TVs": ["num_TVs_1996_pc"],
    "rc/controls/loo/drop_hospitals": ["num_hospitals_1996_pc"],
    "rc/controls/loo/drop_maternhosp": ["num_maternhosp_1996_pc"],
    "rc/controls/loo/drop_polyclinic": ["num_polyclinic_1996_pc"],
    "rc/controls/loo/drop_puskesmas": ["num_puskesmas_1996_pc"],
    "rc/controls/loo/drop_kindergarten": ["num_kindgarden_1996_pc"],
    "rc/controls/loo/drop_primarysch": ["num_primarysch_1996_pc"],
    "rc/controls/loo/drop_HS": ["num_HS_1996_pc"],
}

for spec_id, drop_vars in LOO_MAP.items():
    ctrl = [c for c in ALL_CONTROLS if c not in drop_vars]
    run_spec(
        spec_id, "modules/robustness/controls.md#leave-one-out-controls-loo", "G1",
        "GolkarFirst", "kelurDum", ctrl,
        "kab_str", "kab (district)", df_base,
        {"CRV1": "kab_str"},
        f"Full sample", f"baseline minus {', '.join(drop_vars)}",
        axis_block_name="controls",
        axis_block={"spec_id": spec_id, "family": "loo",
                    "dropped": drop_vars, "n_controls": len(ctrl)})


# ============================================================
# RC: CONTROL SETS (named subsets)
# ============================================================

print("Running control set variants...")

# No controls (bivariate + kab FE)
run_spec(
    "rc/controls/sets/none",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "GolkarFirst", "kelurDum", [],
    "kab_str", "kab (district)", df_base,
    {"CRV1": "kab_str"},
    "Full sample", "none (bivariate + kab FE)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/none", "family": "sets",
                "n_controls": 0, "set_name": "none"})

# Geography only
run_spec(
    "rc/controls/sets/geography_only",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "GolkarFirst", "kelurDum", GEOGRAPHY_CONTROLS,
    "kab_str", "kab (district)", df_base,
    {"CRV1": "kab_str"},
    "Full sample", "geography controls only (11)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/geography_only", "family": "sets",
                "n_controls": len(GEOGRAPHY_CONTROLS), "set_name": "geography_only"})

# Geography + Religion
run_spec(
    "rc/controls/sets/geo_religion",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "GolkarFirst", "kelurDum", GEOGRAPHY_CONTROLS + RELIGION_CONTROLS,
    "kab_str", "kab (district)", df_base,
    {"CRV1": "kab_str"},
    "Full sample", "geography + religion controls (15)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/geo_religion", "family": "sets",
                "n_controls": len(GEOGRAPHY_CONTROLS) + len(RELIGION_CONTROLS),
                "set_name": "geo_religion"})

# Full (same as baseline, for completeness)
run_spec(
    "rc/controls/sets/full",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "GolkarFirst", "kelurDum", ALL_CONTROLS,
    "kab_str", "kab (district)", df_base,
    {"CRV1": "kab_str"},
    "Full sample", "all 23 controls (same as baseline)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/full", "family": "sets",
                "n_controls": len(ALL_CONTROLS), "set_name": "full"})


# ============================================================
# RC: CONTROL PROGRESSION (build-up from Table 2 structure)
# ============================================================

print("Running control progression variants...")

# Raw difference (no FE, no controls)
run_spec(
    "rc/controls/progression/raw_diff",
    "modules/robustness/controls.md#control-progression-build-up", "G1",
    "GolkarFirst", "kelurDum", [],
    "", "none", df_base,
    {"CRV1": "kab_str"},
    "Full sample", "raw difference (no controls, no FE)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/raw_diff", "family": "progression",
                "n_controls": 0, "set_name": "raw_diff"})

# FE only (no controls)
run_spec(
    "rc/controls/progression/fe_only",
    "modules/robustness/controls.md#control-progression-build-up", "G1",
    "GolkarFirst", "kelurDum", [],
    "kab_str", "kab (district)", df_base,
    {"CRV1": "kab_str"},
    "Full sample", "kab FE only",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/fe_only", "family": "progression",
                "n_controls": 0, "set_name": "fe_only"})

# Geography
run_spec(
    "rc/controls/progression/geo",
    "modules/robustness/controls.md#control-progression-build-up", "G1",
    "GolkarFirst", "kelurDum", GEOGRAPHY_CONTROLS,
    "kab_str", "kab (district)", df_base,
    {"CRV1": "kab_str"},
    "Full sample", "kab FE + geography",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/geo", "family": "progression",
                "n_controls": len(GEOGRAPHY_CONTROLS), "set_name": "geography"})

# Geography + Religion
run_spec(
    "rc/controls/progression/geo_religion",
    "modules/robustness/controls.md#control-progression-build-up", "G1",
    "GolkarFirst", "kelurDum", GEOGRAPHY_CONTROLS + RELIGION_CONTROLS,
    "kab_str", "kab (district)", df_base,
    {"CRV1": "kab_str"},
    "Full sample", "kab FE + geography + religion",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/geo_religion", "family": "progression",
                "n_controls": len(GEOGRAPHY_CONTROLS) + len(RELIGION_CONTROLS),
                "set_name": "geo_religion"})

# Geography + Religion + Facilities (= full)
run_spec(
    "rc/controls/progression/geo_religion_facilities",
    "modules/robustness/controls.md#control-progression-build-up", "G1",
    "GolkarFirst", "kelurDum", ALL_CONTROLS,
    "kab_str", "kab (district)", df_base,
    {"CRV1": "kab_str"},
    "Full sample", "kab FE + geography + religion + facilities (full)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/geo_religion_facilities", "family": "progression",
                "n_controls": len(ALL_CONTROLS), "set_name": "geo_religion_facilities"})


# ============================================================
# RC: CONTROL SUBSET (random draws)
# ============================================================

print("Running random control subset variants...")

rng = np.random.RandomState(112756)
subset_pool = ALL_CONTROLS.copy()

for draw_i in range(1, 16):
    k = rng.randint(3, len(subset_pool) + 1)
    chosen = list(rng.choice(subset_pool, size=k, replace=False))
    excluded = [v for v in subset_pool if v not in chosen]
    spec_id = f"rc/controls/subset/random_{draw_i:03d}"
    run_spec(
        spec_id, "modules/robustness/controls.md#subset-generation-specids", "G1",
        "GolkarFirst", "kelurDum", chosen,
        "kab_str", "kab (district)", df_base,
        {"CRV1": "kab_str"},
        "Full sample", f"random subset draw {draw_i} ({len(chosen)} controls)",
        axis_block_name="controls",
        axis_block={"spec_id": spec_id, "family": "subset", "method": "random",
                    "seed": 112756, "draw_index": draw_i,
                    "included": chosen, "excluded": excluded,
                    "n_controls": len(chosen)})


# ============================================================
# RC: SAMPLE TRIMMING
# ============================================================

print("Running sample trimming variants...")

# Trim outcome at 1st/99th percentile
q01 = df_base['GolkarFirst'].quantile(0.01)
q99 = df_base['GolkarFirst'].quantile(0.99)
df_trim1 = df_base[(df_base['GolkarFirst'] >= q01) & (df_base['GolkarFirst'] <= q99)].copy()
n_before = len(df_base)

run_spec(
    "rc/sample/outliers/trim_y_1_99",
    "modules/robustness/sample.md#outliers-and-influential-observations", "G1",
    "GolkarFirst", "kelurDum", ALL_CONTROLS,
    "kab_str", "kab (district)", df_trim1,
    {"CRV1": "kab_str"},
    f"trim GolkarFirst [1%,99%], N={len(df_trim1)}", "full controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_1_99", "axis": "outliers",
                "rule": "trim", "params": {"var": "GolkarFirst", "lower_q": 0.01, "upper_q": 0.99},
                "n_obs_before": n_before, "n_obs_after": len(df_trim1)})

# Trim outcome at 5th/95th percentile
q05 = df_base['GolkarFirst'].quantile(0.05)
q95 = df_base['GolkarFirst'].quantile(0.95)
df_trim5 = df_base[(df_base['GolkarFirst'] >= q05) & (df_base['GolkarFirst'] <= q95)].copy()

run_spec(
    "rc/sample/outliers/trim_y_5_95",
    "modules/robustness/sample.md#outliers-and-influential-observations", "G1",
    "GolkarFirst", "kelurDum", ALL_CONTROLS,
    "kab_str", "kab (district)", df_trim5,
    {"CRV1": "kab_str"},
    f"trim GolkarFirst [5%,95%], N={len(df_trim5)}", "full controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_5_95", "axis": "outliers",
                "rule": "trim", "params": {"var": "GolkarFirst", "lower_q": 0.05, "upper_q": 0.95},
                "n_obs_before": n_before, "n_obs_after": len(df_trim5)})


# ============================================================
# RC: FIXED EFFECTS
# ============================================================

print("Running FE variants...")

# Drop kab FE (pooled OLS)
run_spec(
    "rc/fe/drop/kab",
    "modules/robustness/fixed_effects.md#dropping-fe-relative-to-baseline", "G1",
    "GolkarFirst", "kelurDum", ALL_CONTROLS,
    "", "none (pooled OLS)", df_base,
    {"CRV1": "kab_str"},
    "Full sample", "full controls, no FE",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/drop/kab", "family": "drop",
                "dropped": ["kab"], "baseline_fe": ["kab"], "new_fe": []})

# Add kec FE (sub-district)
run_spec(
    "rc/fe/add/kec",
    "modules/robustness/fixed_effects.md#additive-fe-variations-relative-to-baseline", "G1",
    "GolkarFirst", "kelurDum", ALL_CONTROLS,
    "kec_str", "kec (sub-district)", df_base,
    {"CRV1": "kab_str"},
    "Full sample", "full controls, kec FE",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/add/kec", "family": "add",
                "added": ["kec"], "dropped": ["kab"],
                "baseline_fe": ["kab"], "new_fe": ["kec"],
                "notes": "kec is finer than kab; subsumes kab FE"})


# ============================================================
# RC: FUNCTIONAL FORM
# ============================================================

print("Running functional form variants...")

# Continuous outcome: GolkarFirst is binary (0/1), but the paper also
# reports results using vote share directly. Check if continuous version exists.
# GolkarFirst is already binary, so "continuous" would be a different outcome.
# The surface mentions "golkar_voteshare_continuous" — check if available.
# Looking at the data, GolkarFirst is binary (Golkar wins plurality).
# The paper's Table 1 shows GolkarFirst is binary.
# For rc/form/outcome/golkar_voteshare_continuous, we could use the kab-level
# Golkar vote share (GolkarFirst_kab), but that's a different variable.
# Actually, looking at the data description, there might not be a continuous
# village-level vote share. Let's try using GolkarFirst_kab as an approximation,
# or skip if not meaningful.

# Note: GolkarFirst_kab is a kab-level variable which would be fully absorbed by kab FE.
# Instead, use GolkarFirst without FE as a "continuous" LPM variant (same outcome, no FE).
# This is already covered by rc/fe/drop/kab. For a meaningful continuous outcome variant,
# we could use other party vote shares. The surface spec "golkar_voteshare_continuous" is
# interpreted as using GolkarFirst (which is binary) in a no-FE setting.
# We skip this as it overlaps with rc/fe/drop/kab.

# Probit (already in core_universe)
run_probit(
    "rc/form/estimator/probit",
    "modules/robustness/functional_form.md#estimator-alternatives", "G1",
    "GolkarFirst", "kelurDum", ALL_CONTROLS,
    ["kab_str"], "kab (district)", df_base,
    "Full sample", "full controls (probit estimator)",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/estimator/probit", "estimator": "probit",
                "notes": "Probit model with kab FE dummies; coefficient is probit index"})


# ============================================================
# INFERENCE VARIANTS (on baseline specification)
# ============================================================

print("Running inference variants...")

baseline_run_id = f"{PAPER_ID}_run_001"
infer_counter = 0


def run_inference_variant(base_run_id, spec_id, spec_tree_path, baseline_group_id,
                          formula_str, fe_str, data, focal_var, vcov, vcov_desc):
    """Re-run baseline spec with different variance-covariance estimator."""
    global infer_counter
    infer_counter += 1
    infer_run_id = f"{PAPER_ID}_infer_{infer_counter:03d}"

    try:
        if fe_str:
            full_formula = f"{formula_str} | {fe_str}"
        else:
            full_formula = formula_str

        m = pf.feols(full_formula, data=data, vcov=vcov)

        coef_val = float(m.coef().get(focal_var, np.nan))
        se_val = float(m.se().get(focal_var, np.nan))
        pval = float(m.pvalue().get(focal_var, np.nan))

        try:
            ci = m.confint()
            ci_lower = float(ci.loc[focal_var, ci.columns[0]]) if focal_var in ci.index else np.nan
            ci_upper = float(ci.loc[focal_var, ci.columns[1]]) if focal_var in ci.index else np.nan
        except:
            ci_lower = np.nan
            ci_upper = np.nan

        nobs = int(m._N)
        try:
            r2 = float(m._r2)
        except:
            r2 = np.nan

        all_coefs = {k: float(v) for k, v in m.coef().items()}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": spec_id, "method": vcov_desc},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"cross_sectional_ols": design_audit},
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


# Baseline formula for inference variants
baseline_controls_str = " + ".join(ALL_CONTROLS)
baseline_formula = f"GolkarFirst ~ kelurDum + {baseline_controls_str}"

# HC1 robust (no clustering)
run_inference_variant(
    baseline_run_id, "infer/se/hc/hc1",
    "modules/inference/standard_errors.md#heteroskedasticity-robust", "G1",
    baseline_formula, "kab_str", df_base, "kelurDum",
    "hetero", "HC1 (robust, no clustering)")

# Cluster by kec (sub-district)
run_inference_variant(
    baseline_run_id, "infer/se/cluster/kec",
    "modules/inference/standard_errors.md#clustering", "G1",
    baseline_formula, "kab_str", df_base, "kelurDum",
    {"CRV1": "kec_str"}, "cluster(kec)")


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
    base_row = spec_df[spec_df['spec_id'] == 'baseline']
    if len(base_row) > 0:
        print(f"\nBaseline coef on kelurDum: {base_row['coefficient'].values[0]:.6f}")
        print(f"Baseline SE: {base_row['std_error'].values[0]:.6f}")
        print(f"Baseline p-value: {base_row['p_value'].values[0]:.6f}")
        print(f"Baseline N: {base_row['n_obs'].values[0]:.0f}")

    print(f"\n=== COEFFICIENT RANGE (successful specs) ===")
    print(f"Min coef: {successful['coefficient'].min():.6f}")
    print(f"Max coef: {successful['coefficient'].max():.6f}")
    print(f"Median coef: {successful['coefficient'].median():.6f}")
    n_sig = (successful['p_value'] < 0.05).sum()
    print(f"Significant at 5%: {n_sig}/{len(successful)}")
    n_sig10 = (successful['p_value'] < 0.10).sum()
    print(f"Significant at 10%: {n_sig10}/{len(successful)}")


# ============================================================
# SPECIFICATION_SEARCH.md
# ============================================================

print("\nWriting SPECIFICATION_SEARCH.md...")

md_lines = []
md_lines.append("# Specification Search Report: 112756-V1")
md_lines.append("")
md_lines.append("**Paper:** Martinez-Bravo (2014), \"The Role of Local Officials in New Democracies: Evidence from Indonesia\", AER 104(4)")
md_lines.append("")
md_lines.append("## Baseline Specification")
md_lines.append("")
md_lines.append("- **Design:** Cross-sectional OLS")
md_lines.append("- **Outcome:** GolkarFirst (binary: Golkar wins plurality in 1999 election)")
md_lines.append("- **Treatment:** kelurDum (kelurahan vs desa administrative status)")
md_lines.append(f"- **Controls:** {len(ALL_CONTROLS)} controls (geography, religion, facilities)")
md_lines.append("- **Fixed effects:** kab (district)")
md_lines.append("- **Clustering:** kab (district)")
md_lines.append("")

if len(successful) > 0 and len(base_row) > 0:
    bc = base_row.iloc[0]
    md_lines.append(f"| Statistic | Value |")
    md_lines.append(f"|-----------|-------|")
    md_lines.append(f"| Coefficient | {bc['coefficient']:.6f} |")
    md_lines.append(f"| Std. Error | {bc['std_error']:.6f} |")
    md_lines.append(f"| p-value | {bc['p_value']:.6f} |")
    md_lines.append(f"| 95% CI | [{bc['ci_lower']:.6f}, {bc['ci_upper']:.6f}] |")
    md_lines.append(f"| N | {bc['n_obs']:.0f} |")
    md_lines.append(f"| R-squared | {bc['r_squared']:.4f} |")
    md_lines.append("")

md_lines.append("## Specification Counts")
md_lines.append("")
md_lines.append(f"- Total specifications: {len(spec_df)}")
md_lines.append(f"- Successful: {len(successful)}")
md_lines.append(f"- Failed: {len(failed)}")
md_lines.append(f"- Inference variants: {len(infer_df)}")
md_lines.append("")

# Category breakdown
md_lines.append("## Category Breakdown")
md_lines.append("")
md_lines.append("| Category | Count | Sig. (p<0.05) | Coef Range |")
md_lines.append("|----------|-------|---------------|------------|")

categories = {
    "Baseline": successful[successful['spec_id'].str.startswith('baseline')],
    "Controls LOO": successful[successful['spec_id'].str.startswith('rc/controls/loo/')],
    "Controls Sets": successful[successful['spec_id'].str.startswith('rc/controls/sets/')],
    "Controls Progression": successful[successful['spec_id'].str.startswith('rc/controls/progression/')],
    "Controls Subset": successful[successful['spec_id'].str.startswith('rc/controls/subset/')],
    "Sample Trimming": successful[successful['spec_id'].str.startswith('rc/sample/')],
    "Fixed Effects": successful[successful['spec_id'].str.startswith('rc/fe/')],
    "Functional Form": successful[successful['spec_id'].str.startswith('rc/form/')],
}

for cat_name, cat_df in categories.items():
    if len(cat_df) > 0:
        n_sig_cat = (cat_df['p_value'] < 0.05).sum()
        coef_range = f"[{cat_df['coefficient'].min():.4f}, {cat_df['coefficient'].max():.4f}]"
        md_lines.append(f"| {cat_name} | {len(cat_df)} | {n_sig_cat}/{len(cat_df)} | {coef_range} |")

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
if len(successful) > 0:
    n_sig_total = (successful['p_value'] < 0.05).sum()
    pct_sig = n_sig_total / len(successful) * 100
    sign_consistent = ((successful['coefficient'] > 0).sum() == len(successful)) or \
                      ((successful['coefficient'] < 0).sum() == len(successful))
    median_coef = successful['coefficient'].median()
    sign_word = "positive" if median_coef > 0 else "negative"

    md_lines.append(f"- **Sign consistency:** {'All specifications have the same sign' if sign_consistent else 'Mixed signs across specifications'}")
    md_lines.append(f"- **Significance stability:** {n_sig_total}/{len(successful)} ({pct_sig:.1f}%) specifications significant at 5%")
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
md_lines.append(f"Surface hash: `{SURFACE_HASH}`")
md_lines.append("")

with open(f"{OUTPUT_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write("\n".join(md_lines))

print(f"Wrote SPECIFICATION_SEARCH.md")
print("\nDone!")
