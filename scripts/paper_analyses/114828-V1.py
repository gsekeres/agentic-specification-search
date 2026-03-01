"""
Specification Search Script for Grosfeld, Rodnyansky & Zhuravskaya (2013)
"Persistent Antimarket Culture: A Legacy of the Pale of Settlement"
American Economic Journal: Economic Policy, 5(3), 189-226.

Paper ID: 114828-V1

Surface-driven execution:
  - G1: Sharp geographic/spatial RD at Pale of Settlement boundary
  - Running variable: distance to boundary (negative = inside Pale)
  - Primary outcome: prefer_market (also prefer_democracy, selfemp, trust_d)
  - Nonparametric RD (local linear, bw=60, triangular kernel) = primary baseline
  - Parametric control-function specification as robustness
  - Urban respondents in Russia, Ukraine, Latvia within bandwidth

Implementation note:
  The Python rdrobust package fails on this data due to mass points (PSU-level
  distances with 20 obs per PSU). We implement the nonparametric RD as a
  weighted local linear regression following Stata's `rd` command:
    y ~ treat + distance + treat*distance, weighted by triangular kernel,
    restricted to abs(distance) <= bandwidth, clustered at PSU level.

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

PAPER_ID = "114828-V1"
DATA_DIR = "data/downloads/extracted/114828-V1"
OUTPUT_DIR = DATA_DIR
DATA_PATH = f"{DATA_DIR}/replication/Pale_LITS.dta"

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

# Replicate do-file transformations:
# 1. Negate distance (so negative = inside Pale, positive = outside)
df_raw['distance'] = -df_raw['distance']

# 2. Drop far east
df_raw = df_raw[df_raw['lon_east'] <= 75].copy()
print(f"After dropping far east (lon>75): {len(df_raw)} rows")

# 3. Create control function variables
df_raw['in_pale_distance'] = df_raw['in_pale'] * df_raw['distance']
df_raw['distance_rural'] = df_raw['distance'] * df_raw['rural']
df_raw['in_pale_distance_rural'] = df_raw['in_pale_distance'] * df_raw['rural']

# 4. Create in_pale_nonrural and in_pale_rural
df_raw['in_pale_nonrural'] = ((df_raw['in_pale'] == 1) & (df_raw['rural'] == 0)).astype(int)
df_raw['in_pale_rural'] = ((df_raw['in_pale'] == 1) & (df_raw['rural'] == 1)).astype(int)

# 5. Encode genderB numerically (male=1, female=0)
df_raw['genderB_num'] = (df_raw['genderB'] == 'male').astype(int)

# 6. russian_empire variable
df_raw['russian_empire'] = df_raw['russian_empire_1897']

# 7. Convert identifiers to string for clustering
df_raw['psu1_str'] = df_raw['psu1'].astype(str)
df_raw['country_str'] = df_raw['country'].astype(str)

# Define the baseline sample: urban, russian_empire, (Latvia|Russia|Ukraine)
OUTCOMES = ['prefer_market', 'prefer_democracy', 'selfemp', 'trust_d']

# Full sample mask (no bandwidth restriction)
sample_mask_full = (
    (df_raw['rural'] == 0) &
    (df_raw['russian_empire'] == 1) &
    ((df_raw['Latvia'] == 1) | (df_raw['Russia'] == 1) | (df_raw['Ukraine'] == 1))
)

PRIMARY_OUTCOME = 'prefer_market'

# Parametric controls
BASELINE_CONTROLS = [
    'lat_north', 'lon_east', 'plc', 'genderB_num', 'ageB', 'ageBsquared',
    'christian', 'muslim', 'jewish', 'metropolitan', 'Latvia', 'Ukraine', 'l_elev'
]

# Control function terms (parametric RD)
CONTROL_FUNCTION = ['distance', 'in_pale_distance']

# Control subgroups
GEOGRAPHIC_CONTROLS = ['lat_north', 'lon_east', 'l_elev']
DEMOGRAPHIC_CONTROLS = ['genderB_num', 'ageB', 'ageBsquared']
RELIGION_CONTROLS = ['christian', 'muslim', 'jewish']
COUNTRY_CONTROLS = ['Latvia', 'Ukraine']
OTHER_CONTROLS = ['plc', 'metropolitan']

# Extended controls (from do-file Table 4)
EXTENDED_CONTROLS = ['east_psu', 'tmp', 'prc', 'Russia']

# Full parametric controls = BASELINE_CONTROLS + CONTROL_FUNCTION
FULL_PARAMETRIC_CONTROLS = BASELINE_CONTROLS + CONTROL_FUNCTION

# ============================================================
# Accumulators
# ============================================================

results = []
inference_results = []
spec_run_counter = 0


# ============================================================
# Helper: run_rd_nonparametric (weighted local linear regression)
# ============================================================

def run_rd_nonparametric(spec_id, spec_tree_path, baseline_group_id,
                         outcome_var, data, bw, kernel='triangular',
                         p=1, sample_desc="", notes="",
                         cluster_var='psu1_str', vcov_spec=None,
                         axis_block_name=None, axis_block=None):
    """Run nonparametric RD as weighted local linear regression.

    Implements Stata's `rd` command approach:
      - Restrict to |distance| <= bw
      - Apply kernel weights (triangular or uniform)
      - Run: outcome ~ _treat + distance + _treat*distance [+ higher-order terms]
      - _treat = 1(distance < 0) = inside Pale
      - Cluster SEs at PSU level
    """
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        # Restrict to bandwidth
        sub = data[data['distance'].abs() <= bw].copy()
        sub = sub.dropna(subset=[outcome_var, 'distance']).reset_index(drop=True)

        if len(sub) < 20:
            raise ValueError(f"Too few observations ({len(sub)}) for RD")

        # Kernel weights
        if kernel == 'triangular':
            sub['_weight'] = (1.0 - sub['distance'].abs() / bw).clip(lower=0)
        elif kernel == 'uniform':
            sub['_weight'] = 1.0
        else:
            sub['_weight'] = (1.0 - sub['distance'].abs() / bw).clip(lower=0)

        # Treatment indicator
        sub['_treat'] = (sub['distance'] < 0).astype(int)
        sub['_dist_treat'] = sub['distance'] * sub['_treat']

        # Build formula based on polynomial order
        if p == 1:
            formula = f"{outcome_var} ~ _treat + distance + _dist_treat"
        elif p == 2:
            sub['_dist2'] = sub['distance'] ** 2
            sub['_dist2_treat'] = sub['_dist2'] * sub['_treat']
            formula = f"{outcome_var} ~ _treat + distance + _dist_treat + _dist2 + _dist2_treat"
        else:
            formula = f"{outcome_var} ~ _treat + distance + _dist_treat"

        # Variance-covariance
        if vcov_spec is not None:
            vcov = vcov_spec
        elif cluster_var:
            vcov = {"CRV1": cluster_var}
        else:
            vcov = "hetero"

        m = pf.feols(formula, data=sub, vcov=vcov, weights='_weight')

        coef_val = float(m.coef().get('_treat', np.nan))
        se_val = float(m.se().get('_treat', np.nan))
        pval = float(m.pvalue().get('_treat', np.nan))

        try:
            ci = m.confint()
            ci_lower = float(ci.loc['_treat', ci.columns[0]]) if '_treat' in ci.index else np.nan
            ci_upper = float(ci.loc['_treat', ci.columns[1]]) if '_treat' in ci.index else np.nan
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
                       "method": "cluster" if cluster_var else "robust",
                       "cluster_vars": [cluster_var] if cluster_var else []},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"regression_discontinuity": design_audit},
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
            "treatment_var": "in_pale (RD at distance=0)",
            "coefficient": coef_val,
            "std_error": se_val,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": nobs,
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": "none (nonparametric RD)",
            "controls_desc": f"local poly p={p}, kernel={kernel}, bw={bw}",
            "cluster_var": cluster_var if cluster_var else "none",
            "run_success": 1,
            "run_error": ""
        })
        return run_id, coef_val, se_val, pval, nobs

    except Exception as e:
        err_msg = str(e)[:240]
        err_details = error_details_from_exception(e, stage="rd_nonparametric")
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
            "treatment_var": "in_pale (RD at distance=0)",
            "coefficient": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": np.nan,
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": "none (nonparametric RD)",
            "controls_desc": f"local poly p={p}, kernel={kernel}, bw={bw}",
            "cluster_var": cluster_var if cluster_var else "none",
            "run_success": 0,
            "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


# ============================================================
# Helper: run_rd_parametric (OLS with control function via pyfixest)
# ============================================================

def run_rd_parametric(spec_id, spec_tree_path, baseline_group_id,
                      outcome_var, treatment_var, controls, data,
                      vcov, sample_desc, controls_desc,
                      cluster_var="psu1_str",
                      axis_block_name=None, axis_block=None, notes=""):
    """Run parametric RD specification (OLS with control function)."""
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        controls_str = " + ".join(controls) if controls else ""
        if controls_str:
            formula = f"{outcome_var} ~ {treatment_var} + {controls_str}"
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
                       "method": "cluster", "cluster_vars": ["psu1"]},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"regression_discontinuity": design_audit},
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
            "fixed_effects": "none (parametric RD)",
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
            "run_success": 1,
            "run_error": ""
        })
        return run_id, coef_val, se_val, pval, nobs

    except Exception as e:
        err_msg = str(e)[:240]
        err_details = error_details_from_exception(e, stage="rd_parametric")
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
            "fixed_effects": "none (parametric RD)",
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
            "run_success": 0,
            "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


# ============================================================
# Prepare sample subsets
# ============================================================

df_urban = df_raw[sample_mask_full].copy().reset_index(drop=True)
print(f"Urban + russian_empire + 3-country sample: {len(df_urban)} rows")

# Baseline BW=60 sample for parametric specs
df_bw60 = df_urban[df_urban['distance'].abs() <= 60].copy().reset_index(drop=True)
print(f"Baseline BW=60 sample: {len(df_bw60)} rows")
print(f"  in_pale=1: {(df_bw60['in_pale']==1).sum()}, in_pale=0: {(df_bw60['in_pale']==0).sum()}")


# ============================================================
# BASELINE: Nonparametric RD, prefer_market, bw=60, triangular, p=1
# ============================================================

print("\n=== Running baseline specification (nonparametric RD) ===")
base_run_id, base_coef, base_se, base_pval, base_nobs = run_rd_nonparametric(
    "baseline", "designs/regression_discontinuity.md#baseline", "G1",
    "prefer_market", df_urban, bw=60, kernel='triangular', p=1,
    sample_desc="Urban, russian_empire, Latvia|Russia|Ukraine, BW=60",
    cluster_var='psu1_str')

print(f"  Baseline: coef={base_coef:.4f}, se={base_se:.4f}, p={base_pval:.4f}, N={base_nobs}")


# ============================================================
# BASELINE VARIANTS: Other outcomes
# ============================================================

print("\n=== Running baseline spec for other outcomes ===")

for outcome in ['prefer_democracy', 'selfemp', 'trust_d']:
    rid, c, s, p, n = run_rd_nonparametric(
        f"baseline__{outcome}",
        "designs/regression_discontinuity.md#baseline", "G1",
        outcome, df_urban, bw=60, kernel='triangular', p=1,
        sample_desc="Urban, russian_empire, Latvia|Russia|Ukraine, BW=60",
        cluster_var='psu1_str',
        axis_block_name="estimation",
        axis_block={"spec_id": f"baseline__{outcome}", "outcome": outcome})
    print(f"  {outcome}: coef={c:.4f}, se={s:.4f}, p={p:.4f}, N={n}")


# ============================================================
# DESIGN: Bandwidth variants (nonparametric)
# ============================================================

print("\n=== Running bandwidth variants (nonparametric) ===")

BW_VARIANTS = {
    "design/regression_discontinuity/bandwidth/half_baseline": 30,
    "design/regression_discontinuity/bandwidth/double_baseline": 120,
}

for spec_id, bw in BW_VARIANTS.items():
    rid, c, s, p, n = run_rd_nonparametric(
        spec_id, "designs/regression_discontinuity.md#bandwidth", "G1",
        "prefer_market", df_urban, bw=bw, kernel='triangular', p=1,
        sample_desc=f"Urban, russian_empire, 3-country, BW={bw}",
        cluster_var='psu1_str',
        axis_block_name="design",
        axis_block={"spec_id": spec_id, "bandwidth": bw})
    print(f"  BW={bw}: coef={c:.4f}, se={s:.4f}, p={p:.4f}, N={n}")


# ============================================================
# DESIGN: Polynomial order variants
# ============================================================

print("\n=== Running polynomial order variants ===")

# local linear (p=1) is baseline
# local quadratic (p=2)
rid, c, s, p_val, n = run_rd_nonparametric(
    "design/regression_discontinuity/poly/local_quadratic",
    "designs/regression_discontinuity.md#polynomial", "G1",
    "prefer_market", df_urban, bw=60, kernel='triangular', p=2,
    sample_desc="Urban, russian_empire, 3-country, BW=60",
    cluster_var='psu1_str',
    axis_block_name="design",
    axis_block={"spec_id": "design/regression_discontinuity/poly/local_quadratic",
                "poly_order": 2})
print(f"  Local quadratic: coef={c:.4f}, se={s:.4f}, p={p_val:.4f}, N={n}")


# ============================================================
# DESIGN: Kernel variants
# ============================================================

print("\n=== Running kernel variants ===")

# uniform kernel
rid, c, s, p_val, n = run_rd_nonparametric(
    "design/regression_discontinuity/kernel/uniform",
    "designs/regression_discontinuity.md#kernel", "G1",
    "prefer_market", df_urban, bw=60, kernel='uniform', p=1,
    sample_desc="Urban, russian_empire, 3-country, BW=60",
    cluster_var='psu1_str',
    axis_block_name="design",
    axis_block={"spec_id": "design/regression_discontinuity/kernel/uniform",
                "kernel": "uniform"})
print(f"  Uniform kernel: coef={c:.4f}, se={s:.4f}, p={p_val:.4f}, N={n}")


# ============================================================
# RC: Controls LOO (on parametric specification)
# ============================================================

print("\n=== Running controls LOO variants (parametric) ===")

# Full parametric baseline
df_param_bw60 = df_bw60.dropna(subset=FULL_PARAMETRIC_CONTROLS + ['prefer_market']).copy()
print(f"  Parametric BW60 sample (non-missing controls): {len(df_param_bw60)} rows")

param_base_id, pc, ps, pp, pn = run_rd_parametric(
    "baseline__parametric",
    "designs/regression_discontinuity.md#parametric_baseline", "G1",
    "prefer_market", "in_pale_nonrural",
    FULL_PARAMETRIC_CONTROLS,
    df_param_bw60,
    {"CRV1": "psu1_str"},
    "Urban, russian_empire, 3-country, BW=60, parametric",
    f"baseline + control function ({len(FULL_PARAMETRIC_CONTROLS)} vars)",
    cluster_var="psu1_str")
print(f"  Parametric baseline: coef={pc:.4f}, se={ps:.4f}, p={pp:.4f}, N={pn}")

# LOO: Drop individual baseline controls (not control function terms)
LOO_MAP = {
    "rc/controls/loo/drop_lat_north": ["lat_north"],
    "rc/controls/loo/drop_lon_east": ["lon_east"],
    "rc/controls/loo/drop_plc": ["plc"],
    "rc/controls/loo/drop_genderB": ["genderB_num"],
    "rc/controls/loo/drop_ageB": ["ageB", "ageBsquared"],
    "rc/controls/loo/drop_christian": ["christian"],
    "rc/controls/loo/drop_muslim": ["muslim"],
    "rc/controls/loo/drop_jewish": ["jewish"],
    "rc/controls/loo/drop_metropolitan": ["metropolitan"],
    "rc/controls/loo/drop_l_elev": ["l_elev"],
}

for spec_id, drop_vars in LOO_MAP.items():
    ctrl = [c for c in FULL_PARAMETRIC_CONTROLS if c not in drop_vars]
    df_loo = df_bw60.dropna(subset=ctrl + ['prefer_market']).copy()
    run_rd_parametric(
        spec_id, "modules/robustness/controls.md#leave-one-out-controls-loo", "G1",
        "prefer_market", "in_pale_nonrural", ctrl, df_loo,
        {"CRV1": "psu1_str"},
        "Urban, russian_empire, 3-country, BW=60",
        f"parametric minus {', '.join(drop_vars)}",
        axis_block_name="controls",
        axis_block={"spec_id": spec_id, "family": "loo",
                    "dropped": drop_vars, "n_controls": len(ctrl)})


# ============================================================
# RC: Control set variants (parametric)
# ============================================================

print("\n=== Running control set variants (parametric) ===")

# No controls (just treatment + control function)
ctrl_none = CONTROL_FUNCTION
df_tmp = df_bw60.dropna(subset=ctrl_none + ['prefer_market']).copy()
run_rd_parametric(
    "rc/controls/sets/none", "modules/robustness/controls.md#control-sets", "G1",
    "prefer_market", "in_pale_nonrural", ctrl_none, df_tmp,
    {"CRV1": "psu1_str"},
    "Urban, russian_empire, 3-country, BW=60",
    "control function only (no covariates)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/none", "family": "sets", "set_name": "none"})

# Geographic only
ctrl_geo = GEOGRAPHIC_CONTROLS + CONTROL_FUNCTION
df_tmp = df_bw60.dropna(subset=ctrl_geo + ['prefer_market']).copy()
run_rd_parametric(
    "rc/controls/sets/geographic_only", "modules/robustness/controls.md#control-sets", "G1",
    "prefer_market", "in_pale_nonrural", ctrl_geo, df_tmp,
    {"CRV1": "psu1_str"},
    "Urban, russian_empire, 3-country, BW=60",
    f"geographic + control function ({len(ctrl_geo)} vars)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/geographic_only", "family": "sets",
                "set_name": "geographic_only"})

# Demographic only
ctrl_demo = DEMOGRAPHIC_CONTROLS + CONTROL_FUNCTION
df_tmp = df_bw60.dropna(subset=ctrl_demo + ['prefer_market']).copy()
run_rd_parametric(
    "rc/controls/sets/demographic_only", "modules/robustness/controls.md#control-sets", "G1",
    "prefer_market", "in_pale_nonrural", ctrl_demo, df_tmp,
    {"CRV1": "psu1_str"},
    "Urban, russian_empire, 3-country, BW=60",
    f"demographic + control function ({len(ctrl_demo)} vars)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/demographic_only", "family": "sets",
                "set_name": "demographic_only"})

# Full parametric (same as baseline__parametric, for completeness in set)
run_rd_parametric(
    "rc/controls/sets/full_parametric", "modules/robustness/controls.md#control-sets", "G1",
    "prefer_market", "in_pale_nonrural", FULL_PARAMETRIC_CONTROLS, df_param_bw60,
    {"CRV1": "psu1_str"},
    "Urban, russian_empire, 3-country, BW=60",
    f"full baseline + control function ({len(FULL_PARAMETRIC_CONTROLS)} vars)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/full_parametric", "family": "sets",
                "set_name": "full_parametric"})

# Baseline only (all baseline controls + control function)
ctrl_baseline_only = ['plc', 'genderB_num', 'ageB', 'ageBsquared', 'christian', 'muslim',
                      'jewish', 'metropolitan', 'l_elev', 'Latvia', 'Ukraine',
                      'lat_north', 'lon_east'] + CONTROL_FUNCTION
df_tmp = df_bw60.dropna(subset=ctrl_baseline_only + ['prefer_market']).copy()
run_rd_parametric(
    "rc/controls/sets/baseline_only", "modules/robustness/controls.md#control-sets", "G1",
    "prefer_market", "in_pale_nonrural", ctrl_baseline_only, df_tmp,
    {"CRV1": "psu1_str"},
    "Urban, russian_empire, 3-country, BW=60",
    f"baseline controls + control function ({len(ctrl_baseline_only)} vars)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/baseline_only", "family": "sets",
                "set_name": "baseline_only"})

# Extended (baseline + extended controls)
ctrl_extended = FULL_PARAMETRIC_CONTROLS + EXTENDED_CONTROLS
df_tmp = df_bw60.dropna(subset=ctrl_extended + ['prefer_market']).copy()
run_rd_parametric(
    "rc/controls/sets/extended", "modules/robustness/controls.md#control-sets", "G1",
    "prefer_market", "in_pale_nonrural", ctrl_extended, df_tmp,
    {"CRV1": "psu1_str"},
    "Urban, russian_empire, 3-country, BW=60",
    f"full + extended controls ({len(ctrl_extended)} vars)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/extended", "family": "sets",
                "set_name": "extended"})


# ============================================================
# RC: Control progressions (parametric)
# ============================================================

print("\n=== Running control progression variants (parametric) ===")

progressions = {
    "rc/controls/progression/geographic_only": GEOGRAPHIC_CONTROLS + CONTROL_FUNCTION,
    "rc/controls/progression/geographic_demographic":
        GEOGRAPHIC_CONTROLS + DEMOGRAPHIC_CONTROLS + CONTROL_FUNCTION,
    "rc/controls/progression/geographic_demographic_religion":
        GEOGRAPHIC_CONTROLS + DEMOGRAPHIC_CONTROLS + RELIGION_CONTROLS + CONTROL_FUNCTION,
    "rc/controls/progression/full_with_extended":
        FULL_PARAMETRIC_CONTROLS + EXTENDED_CONTROLS,
}

for spec_id, ctrl in progressions.items():
    df_tmp = df_bw60.dropna(subset=ctrl + ['prefer_market']).copy()
    run_rd_parametric(
        spec_id, "modules/robustness/controls.md#control-progression", "G1",
        "prefer_market", "in_pale_nonrural", ctrl, df_tmp,
        {"CRV1": "psu1_str"},
        "Urban, russian_empire, 3-country, BW=60",
        f"progression: {spec_id.split('/')[-1]} ({len(ctrl)} vars)",
        axis_block_name="controls",
        axis_block={"spec_id": spec_id, "family": "progression", "n_controls": len(ctrl)})


# ============================================================
# RC: Additional controls (parametric)
# ============================================================

print("\n=== Running add-control variants (parametric) ===")

for add_var in EXTENDED_CONTROLS:
    spec_id = f"rc/controls/add/{add_var}"
    ctrl = FULL_PARAMETRIC_CONTROLS + [add_var]
    df_tmp = df_bw60.dropna(subset=ctrl + ['prefer_market']).copy()
    run_rd_parametric(
        spec_id, "modules/robustness/controls.md#add-controls", "G1",
        "prefer_market", "in_pale_nonrural", ctrl, df_tmp,
        {"CRV1": "psu1_str"},
        "Urban, russian_empire, 3-country, BW=60",
        f"full + {add_var} ({len(ctrl)} vars)",
        axis_block_name="controls",
        axis_block={"spec_id": spec_id, "family": "add", "added": [add_var]})


# ============================================================
# RC: Additional bandwidth variants (nonparametric)
# ============================================================

print("\n=== Running additional bandwidth variants ===")

ADDITIONAL_BW = {
    "rc/sample/bandwidth/bw_30": 30,
    "rc/sample/bandwidth/bw_90": 90,
    "rc/sample/bandwidth/bw_120": 120,
}

for spec_id, bw in ADDITIONAL_BW.items():
    rid, c, s, p_val, n = run_rd_nonparametric(
        spec_id, "modules/robustness/sample.md#bandwidth", "G1",
        "prefer_market", df_urban, bw=bw, kernel='triangular', p=1,
        sample_desc=f"Urban, russian_empire, 3-country, BW={bw}",
        cluster_var='psu1_str',
        axis_block_name="sample",
        axis_block={"spec_id": spec_id, "bandwidth": bw})
    print(f"  BW={bw}: coef={c:.4f}, se={s:.4f}, p={p_val:.4f}, N={n}")


# ============================================================
# RC: Sample restriction variants (nonparametric)
# ============================================================

print("\n=== Running sample restriction variants ===")

# Urban only is the baseline -- already done

# Rural only
rural_mask = (
    (df_raw['rural'] == 1) &
    (df_raw['russian_empire'] == 1) &
    ((df_raw['Latvia'] == 1) | (df_raw['Russia'] == 1) | (df_raw['Ukraine'] == 1))
)
df_rural = df_raw[rural_mask].copy().reset_index(drop=True)
run_rd_nonparametric(
    "rc/sample/restrict/rural_only",
    "modules/robustness/sample.md#subgroup", "G1",
    "prefer_market", df_rural, bw=60, kernel='triangular', p=1,
    sample_desc=f"Rural only, russian_empire, 3-country, BW=60 (N_full={len(df_rural)})",
    cluster_var='psu1_str',
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restrict/rural_only", "restriction": "rural_only"})

# Russia only
df_russia = df_urban[df_urban['Russia'] == 1].copy().reset_index(drop=True)
run_rd_nonparametric(
    "rc/sample/restrict/russia_only",
    "modules/robustness/sample.md#subgroup", "G1",
    "prefer_market", df_russia, bw=60, kernel='triangular', p=1,
    sample_desc=f"Urban, Russia only, BW=60 (N_full={len(df_russia)})",
    cluster_var='psu1_str',
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restrict/russia_only", "restriction": "russia_only"})

# Ukraine only
df_ukraine = df_urban[df_urban['Ukraine'] == 1].copy().reset_index(drop=True)
run_rd_nonparametric(
    "rc/sample/restrict/ukraine_only",
    "modules/robustness/sample.md#subgroup", "G1",
    "prefer_market", df_ukraine, bw=60, kernel='triangular', p=1,
    sample_desc=f"Urban, Ukraine only, BW=60 (N_full={len(df_ukraine)})",
    cluster_var='psu1_str',
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restrict/ukraine_only", "restriction": "ukraine_only"})

# Latvia only
df_latvia = df_urban[df_urban['Latvia'] == 1].copy().reset_index(drop=True)
run_rd_nonparametric(
    "rc/sample/restrict/latvia_only",
    "modules/robustness/sample.md#subgroup", "G1",
    "prefer_market", df_latvia, bw=60, kernel='triangular', p=1,
    sample_desc=f"Urban, Latvia only, BW=60 (N_full={len(df_latvia)})",
    cluster_var='psu1_str',
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restrict/latvia_only", "restriction": "latvia_only"})

# Russia + Ukraine
df_ru_ua = df_urban[(df_urban['Russia'] == 1) | (df_urban['Ukraine'] == 1)].copy().reset_index(drop=True)
run_rd_nonparametric(
    "rc/sample/restrict/russia_ukraine_only",
    "modules/robustness/sample.md#subgroup", "G1",
    "prefer_market", df_ru_ua, bw=60, kernel='triangular', p=1,
    sample_desc=f"Urban, Russia+Ukraine, BW=60 (N_full={len(df_ru_ua)})",
    cluster_var='psu1_str',
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restrict/russia_ukraine_only",
                "restriction": "russia_ukraine_only"})

# Russian empire (all countries, not just 3)
empire_mask = (df_raw['rural'] == 0) & (df_raw['russian_empire'] == 1)
df_empire = df_raw[empire_mask].copy().reset_index(drop=True)
run_rd_nonparametric(
    "rc/sample/restrict/russian_empire_only",
    "modules/robustness/sample.md#subgroup", "G1",
    "prefer_market", df_empire, bw=60, kernel='triangular', p=1,
    sample_desc=f"Urban, all russian_empire countries, BW=60 (N_full={len(df_empire)})",
    cluster_var='psu1_str',
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restrict/russian_empire_only",
                "restriction": "russian_empire_only"})

# Urban only (explicit, same as baseline for documentation)
run_rd_nonparametric(
    "rc/sample/restrict/urban_only",
    "modules/robustness/sample.md#subgroup", "G1",
    "prefer_market", df_urban, bw=60, kernel='triangular', p=1,
    sample_desc=f"Urban only (same as baseline), BW=60",
    cluster_var='psu1_str',
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restrict/urban_only", "restriction": "urban_only"})


# ============================================================
# RC: Donut hole variants (nonparametric)
# ============================================================

print("\n=== Running donut hole variants ===")

DONUT_VARIANTS = {
    "rc/sample/donut/exclude_5km": 5,
    "rc/sample/donut/exclude_10km": 10,
    "rc/sample/donut/exclude_15km": 15,
    "rc/sample/donut/exclude_20km": 20,
}

for spec_id, donut_km in DONUT_VARIANTS.items():
    df_donut = df_urban[df_urban['distance'].abs() > donut_km].copy().reset_index(drop=True)
    rid, c, s, p_val, n = run_rd_nonparametric(
        spec_id, "modules/robustness/sample.md#donut", "G1",
        "prefer_market", df_donut, bw=60, kernel='triangular', p=1,
        sample_desc=f"Urban, 3-country, BW=60, donut>{donut_km}km (N_avail={len(df_donut)})",
        cluster_var='psu1_str',
        axis_block_name="sample",
        axis_block={"spec_id": spec_id, "donut_km": donut_km})
    print(f"  Donut {donut_km}km: coef={c:.4f}, se={s:.4f}, p={p_val:.4f}, N={n}")


# ============================================================
# RC: Cross outcome x design (nonparametric)
# ============================================================

print("\n=== Running cross outcome x design combinations ===")

for outcome in ['prefer_democracy', 'selfemp', 'trust_d']:
    # half bandwidth
    run_rd_nonparametric(
        f"cross/{outcome}/bw_30",
        "designs/regression_discontinuity.md#cross", "G1",
        outcome, df_urban, bw=30, kernel='triangular', p=1,
        sample_desc=f"Urban, 3-country, BW=30",
        cluster_var='psu1_str',
        axis_block_name="design",
        axis_block={"spec_id": f"cross/{outcome}/bw_30", "outcome": outcome, "bw": 30})

    # double bandwidth
    run_rd_nonparametric(
        f"cross/{outcome}/bw_120",
        "designs/regression_discontinuity.md#cross", "G1",
        outcome, df_urban, bw=120, kernel='triangular', p=1,
        sample_desc=f"Urban, 3-country, BW=120",
        cluster_var='psu1_str',
        axis_block_name="design",
        axis_block={"spec_id": f"cross/{outcome}/bw_120", "outcome": outcome, "bw": 120})

    # uniform kernel
    run_rd_nonparametric(
        f"cross/{outcome}/uniform",
        "designs/regression_discontinuity.md#cross", "G1",
        outcome, df_urban, bw=60, kernel='uniform', p=1,
        sample_desc=f"Urban, 3-country, BW=60",
        cluster_var='psu1_str',
        axis_block_name="design",
        axis_block={"spec_id": f"cross/{outcome}/uniform", "outcome": outcome,
                    "kernel": "uniform"})

    # local quadratic
    run_rd_nonparametric(
        f"cross/{outcome}/local_quadratic",
        "designs/regression_discontinuity.md#cross", "G1",
        outcome, df_urban, bw=60, kernel='triangular', p=2,
        sample_desc=f"Urban, 3-country, BW=60",
        cluster_var='psu1_str',
        axis_block_name="design",
        axis_block={"spec_id": f"cross/{outcome}/local_quadratic", "outcome": outcome,
                    "poly_order": 2})


# ============================================================
# RC: Parametric specification for other outcomes
# ============================================================

print("\n=== Running parametric specs for other outcomes ===")

for outcome in ['prefer_democracy', 'selfemp', 'trust_d']:
    df_tmp = df_bw60.dropna(subset=FULL_PARAMETRIC_CONTROLS + [outcome]).copy()
    run_rd_parametric(
        f"parametric/{outcome}",
        "designs/regression_discontinuity.md#parametric", "G1",
        outcome, "in_pale_nonrural", FULL_PARAMETRIC_CONTROLS, df_tmp,
        {"CRV1": "psu1_str"},
        f"Urban, 3-country, BW=60, parametric",
        f"full baseline + control function ({len(FULL_PARAMETRIC_CONTROLS)} vars)",
        axis_block_name="estimation",
        axis_block={"spec_id": f"parametric/{outcome}", "outcome": outcome,
                    "approach": "parametric"})


# ============================================================
# INFERENCE VARIANTS
# ============================================================

print("\n=== Running inference variants ===")

baseline_run_id = f"{PAPER_ID}_run_001"
infer_counter = 0


def run_inference_rd(base_run_id, spec_id, spec_tree_path, baseline_group_id,
                     outcome_var, data, bw, kernel, p_order, vcov_spec, vcov_desc):
    """Run nonparametric RD with different variance-covariance specification."""
    global infer_counter
    infer_counter += 1
    infer_run_id = f"{PAPER_ID}_infer_{infer_counter:03d}"

    try:
        sub = data[data['distance'].abs() <= bw].copy()
        sub = sub.dropna(subset=[outcome_var, 'distance']).reset_index(drop=True)

        if kernel == 'triangular':
            sub['_weight'] = (1.0 - sub['distance'].abs() / bw).clip(lower=0)
        else:
            sub['_weight'] = 1.0

        sub['_treat'] = (sub['distance'] < 0).astype(int)
        sub['_dist_treat'] = sub['distance'] * sub['_treat']

        formula = f"{outcome_var} ~ _treat + distance + _dist_treat"
        m = pf.feols(formula, data=sub, vcov=vcov_spec, weights='_weight')

        coef_val = float(m.coef().get('_treat', np.nan))
        se_val = float(m.se().get('_treat', np.nan))
        pval = float(m.pvalue().get('_treat', np.nan))

        try:
            ci = m.confint()
            ci_lower = float(ci.loc['_treat', ci.columns[0]]) if '_treat' in ci.index else np.nan
            ci_upper = float(ci.loc['_treat', ci.columns[1]]) if '_treat' in ci.index else np.nan
        except Exception:
            ci_lower = np.nan
            ci_upper = np.nan

        nobs = int(m._N)
        try:
            r2 = float(m._r2)
        except Exception:
            r2 = np.nan

        payload = make_success_payload(
            coefficients={k: float(v) for k, v in m.coef().items()},
            inference={"spec_id": spec_id, "method": vcov_desc},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"regression_discontinuity": design_audit},
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


# HC1 robust (no clustering)
run_inference_rd(
    baseline_run_id, "infer/se/hc/hc1",
    "modules/inference/standard_errors.md#heteroskedasticity-robust", "G1",
    "prefer_market", df_urban, bw=60, kernel='triangular', p_order=1,
    vcov_spec="hetero", vcov_desc="HC1 (robust, no clustering)")

# Cluster by country (very few clusters -- interpretive caveat)
run_inference_rd(
    baseline_run_id, "infer/se/cluster/country",
    "modules/inference/standard_errors.md#clustering", "G1",
    "prefer_market", df_urban, bw=60, kernel='triangular', p_order=1,
    vcov_spec={"CRV1": "country_str"}, vcov_desc="cluster(country)")


# ============================================================
# WRITE OUTPUTS
# ============================================================

print(f"\n=== Writing outputs ===")
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
        print(f"\nBaseline coef (nonparametric RD): {base_row['coefficient'].values[0]:.6f}")
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

if len(failed) > 0:
    print(f"\n=== FAILED SPECS ===")
    for _, row in failed.iterrows():
        print(f"  {row['spec_id']}: {row['run_error'][:100]}")


# ============================================================
# SPECIFICATION_SEARCH.md
# ============================================================

print("\nWriting SPECIFICATION_SEARCH.md...")

md_lines = []
md_lines.append("# Specification Search Report: 114828-V1")
md_lines.append("")
md_lines.append("**Paper:** Grosfeld, Rodnyansky & Zhuravskaya (2013), \"Persistent Antimarket Culture: A Legacy of the Pale of Settlement\", AEJ: Economic Policy 5(3)")
md_lines.append("")
md_lines.append("## Baseline Specification")
md_lines.append("")
md_lines.append("- **Design:** Sharp geographic/spatial RD")
md_lines.append("- **Running variable:** Distance to Pale of Settlement boundary (km)")
md_lines.append("- **Cutoff:** 0 (the boundary itself)")
md_lines.append("- **Outcome:** prefer_market (binary: prefers market economy)")
md_lines.append("- **Treatment:** Inside Pale (distance < 0)")
md_lines.append("- **Method:** Weighted local linear regression (triangular kernel, BW=60km)")
md_lines.append("- **Clustering:** PSU (psu1)")
md_lines.append("- **Sample:** Urban respondents in Russia, Ukraine, Latvia (former Russian Empire)")
md_lines.append("")

if len(successful) > 0:
    base_row = spec_df[spec_df['spec_id'] == 'baseline']
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
    "Baseline (nonparametric)": successful[successful['spec_id'].str.startswith('baseline')],
    "Design (BW/poly/kernel)": successful[successful['spec_id'].str.startswith('design/')],
    "Controls LOO (parametric)": successful[successful['spec_id'].str.startswith('rc/controls/loo/')],
    "Controls Sets (parametric)": successful[successful['spec_id'].str.startswith('rc/controls/sets/')],
    "Controls Progression": successful[successful['spec_id'].str.startswith('rc/controls/progression/')],
    "Controls Add": successful[successful['spec_id'].str.startswith('rc/controls/add/')],
    "Sample/Bandwidth": successful[successful['spec_id'].str.startswith('rc/sample/bandwidth/')],
    "Sample/Restrict": successful[successful['spec_id'].str.startswith('rc/sample/restrict/')],
    "Sample/Donut": successful[successful['spec_id'].str.startswith('rc/sample/donut/')],
    "Cross Outcome x Design": successful[successful['spec_id'].str.startswith('cross/')],
    "Parametric (other outcomes)": successful[successful['spec_id'].str.startswith('parametric/')],
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
    # Focus on prefer_market specs for primary assessment
    pm_specs = successful[successful['outcome_var'] == 'prefer_market']
    if len(pm_specs) > 0:
        n_sig_total = (pm_specs['p_value'] < 0.05).sum()
        pct_sig = n_sig_total / len(pm_specs) * 100
        sign_consistent = ((pm_specs['coefficient'] > 0).sum() == len(pm_specs)) or \
                          ((pm_specs['coefficient'] < 0).sum() == len(pm_specs))
        median_coef = pm_specs['coefficient'].median()
        sign_word = "positive" if median_coef > 0 else "negative"

        md_lines.append(f"### prefer_market (primary outcome, {len(pm_specs)} specifications)")
        md_lines.append(f"- **Sign consistency:** {'All specifications have the same sign' if sign_consistent else 'Mixed signs across specifications'}")
        md_lines.append(f"- **Significance stability:** {n_sig_total}/{len(pm_specs)} ({pct_sig:.1f}%) specifications significant at 5%")
        md_lines.append(f"- **Direction:** Median coefficient is {sign_word} ({median_coef:.6f})")

    # All outcomes together
    all_sig = (successful['p_value'] < 0.05).sum()
    all_pct = all_sig / len(successful) * 100
    md_lines.append("")
    md_lines.append(f"### All outcomes ({len(successful)} specifications)")
    md_lines.append(f"- **Significance stability:** {all_sig}/{len(successful)} ({all_pct:.1f}%) specifications significant at 5%")

    if len(pm_specs) > 0:
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
