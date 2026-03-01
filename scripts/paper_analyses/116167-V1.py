"""
Specification Search Script for Durante, Pinotti & Tesei (2019)
"The Political Legacy of Entertainment TV"
American Economic Review, 109(7), 2497-2530.

Paper ID: 116167-V1

Surface-driven execution:
  - G1: berl ~ signal + signalfree + controls + i.sll2001 | district [w=pop81]
  - Cross-sectional OLS with district FE, clustered SEs at district level
  - Population-weighted (pop81), trimmed sample (xsignal 2.5-97.5 percentile)
  - 50+ specifications across controls LOO, controls progression, controls add,
    sample restrictions, FE swaps, weights, treatment form, matched neighbors,
    year-by-year estimates

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

PAPER_ID = "116167-V1"
DATA_DIR = "data/downloads/extracted/116167-V1"
OUTPUT_DIR = DATA_DIR
DATA_PATH = f"{DATA_DIR}/data_DPT/municipality-level_data.dta"
MATCHED_DATA_PATH = f"{DATA_DIR}/data_DPT/municipality-level-matched-neighbors_data.dta"

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

df_raw = pd.read_stata(DATA_PATH, convert_categoricals=False)
print(f"Loaded data: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

# Convert float32 to float64 for precision
for col in df_raw.columns:
    if df_raw[col].dtype == np.float32:
        df_raw[col] = df_raw[col].astype(np.float64)

# Apply the paper's sample trimming: keep if xsignal>=25 & xsignal<=975
# This trims the top and bottom 2.5% of the signal distribution
df_raw = df_raw[(df_raw['xsignal'] >= 25) & (df_raw['xsignal'] <= 975)].copy()
print(f"After xsignal trimming (2.5-97.5%): {len(df_raw)} rows")

# Make FE variables strings for pyfixest
df_raw['district_str'] = df_raw['district'].astype(str)
df_raw['sll2001_str'] = df_raw['sll2001'].astype(str)
df_raw['prov_str'] = df_raw['prov'].astype(str)

# Create geographic region dummies for sample restrictions
# ISTAT standard: North (1-8 = Piemonte through Emilia-Romagna),
# Center (9-12 = Toscana, Umbria, Marche, Lazio),
# South+Islands (13-20 = Abruzzo through Sardegna)
df_raw['north'] = (df_raw['reg'] <= 8).astype(int)
df_raw['center'] = ((df_raw['reg'] >= 9) & (df_raw['reg'] <= 12)).astype(int)
df_raw['south'] = (df_raw['reg'] >= 13).astype(int)

# Focus on year==1994 for baseline (Table 3)
df_1994 = df_raw[df_raw['year'] == 1994].copy()
print(f"Year==1994 sample: {len(df_1994)} rows")

# Define control variable groups (from Stata dofile)
# "contr" = area area2 altitude altitude2 ruggedness electorate lnincome highschool_college81
# "land"  = area area2 altitude altitude2 ruggedness
LAND_CONTROLS = ["area", "area2", "altitude", "altitude2", "ruggedness"]
SOCIOEC_CONTROLS = ["electorate", "lnincome", "highschool_college81"]
FULL_CONTROLS = LAND_CONTROLS + SOCIOEC_CONTROLS

# Baseline always includes signalfree alongside signal
# The formula is: berl ~ signal + signalfree + controls | district + sll2001 [w=pop81]

# Drop rows with missing outcome, treatment, or key variables for baseline sample
base_vars = ["berl", "signal", "signalfree"] + FULL_CONTROLS + ["district", "sll2001", "pop81"]
df_base = df_1994.dropna(subset=base_vars).copy()
print(f"Baseline sample (all vars non-missing, year=1994): {len(df_base)} rows")
print(f"  District unique: {df_base['district_str'].nunique()}")
print(f"  SLL2001 unique: {df_base['sll2001_str'].nunique()}")

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
             cluster_var="district_str", weights_var="pop81",
             fixef_tol=None,
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

        # Build kwargs
        kwargs = {"data": data, "vcov": vcov}
        if weights_var:
            kwargs["weights"] = weights_var
        if fixef_tol is not None:
            kwargs["fixef_tol"] = fixef_tol

        m = pf.feols(formula, **kwargs)

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
                       "method": "cluster", "cluster_vars": ["district"]},
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
# BASELINE: Table 3, Col 5 — OLS with full controls + district+sll FE, weighted
# ============================================================
# Stata: areg berl signal signalfree $contr i.sll2001 if year==1994 [w=pop81],
#         robust cluster(district) abs(district)

print("=" * 60)
print("Running baseline specification (Table 3, Col 5)...")
print("=" * 60)

base_run_id, base_coef, base_se, base_pval, base_nobs = run_spec(
    "baseline", "designs/cross_sectional_ols.md#baseline", "G1",
    "berl", "signal", ["signalfree"] + FULL_CONTROLS,
    "district_str + sll2001_str", "district + sll2001", df_base,
    {"CRV1": "district_str"},
    f"1994, trimmed xsignal [2.5%,97.5%], N={len(df_base)}",
    "signalfree + land + socioeconomic (8 controls)")

print(f"  Baseline: coef={base_coef:.4f}, se={base_se:.4f}, p={base_pval:.4f}, N={base_nobs}")


# ============================================================
# BASELINE VARIANT: Table 3, Col 4 — land controls only
# ============================================================

print("\nRunning baseline variant (Table 3, Col 4 — land controls only)...")

run_spec(
    "baseline__land_only", "designs/cross_sectional_ols.md#baseline", "G1",
    "berl", "signal", ["signalfree"] + LAND_CONTROLS,
    "district_str + sll2001_str", "district + sll2001", df_base,
    {"CRV1": "district_str"},
    f"1994, trimmed, N={len(df_base)}", "signalfree + land controls (5)",
    axis_block_name="controls",
    axis_block={"spec_id": "baseline__land_only", "family": "baseline_variant",
                "n_controls": 6, "set_name": "land_only"})


# ============================================================
# BASELINE VARIANT: No controls (bivariate with FE)
# ============================================================

print("Running baseline variant (no controls)...")

run_spec(
    "baseline__no_controls", "designs/cross_sectional_ols.md#baseline", "G1",
    "berl", "signal", ["signalfree"],
    "district_str + sll2001_str", "district + sll2001", df_base,
    {"CRV1": "district_str"},
    f"1994, trimmed, N={len(df_base)}", "signalfree only (no other controls)",
    axis_block_name="controls",
    axis_block={"spec_id": "baseline__no_controls", "family": "baseline_variant",
                "n_controls": 1, "set_name": "no_controls"})


# ============================================================
# BASELINE VARIANT: Unweighted (Table 3, Col 6)
# ============================================================

print("Running baseline variant (unweighted)...")

run_spec(
    "baseline__unweighted", "designs/cross_sectional_ols.md#baseline", "G1",
    "berl", "signal", ["signalfree"] + FULL_CONTROLS,
    "district_str + sll2001_str", "district + sll2001", df_base,
    {"CRV1": "district_str"},
    f"1994, trimmed, unweighted, N={len(df_base)}", "full controls (unweighted)",
    weights_var=None,
    axis_block_name="weights",
    axis_block={"spec_id": "baseline__unweighted", "family": "weights",
                "weights": "none"})


# ============================================================
# RC: CONTROLS LOO — Drop one control at a time
# ============================================================

print("\nRunning controls LOO variants...")

LOO_MAP = {
    "rc/controls/loo/drop_signalfree": ["signalfree"],
    "rc/controls/loo/drop_area": ["area", "area2"],
    "rc/controls/loo/drop_altitude": ["altitude", "altitude2"],
    "rc/controls/loo/drop_ruggedness": ["ruggedness"],
    "rc/controls/loo/drop_electorate": ["electorate"],
    "rc/controls/loo/drop_lnincome": ["lnincome"],
    "rc/controls/loo/drop_highschool_college81": ["highschool_college81"],
}

# Full baseline controls including signalfree
BASELINE_CONTROLS = ["signalfree"] + FULL_CONTROLS

for spec_id, drop_vars in LOO_MAP.items():
    ctrl = [c for c in BASELINE_CONTROLS if c not in drop_vars]
    run_spec(
        spec_id, "modules/robustness/controls.md#leave-one-out-controls-loo", "G1",
        "berl", "signal", ctrl,
        "district_str + sll2001_str", "district + sll2001", df_base,
        {"CRV1": "district_str"},
        f"1994, trimmed", f"baseline minus {', '.join(drop_vars)}",
        axis_block_name="controls",
        axis_block={"spec_id": spec_id, "family": "loo",
                    "dropped": drop_vars, "n_controls": len(ctrl)})


# ============================================================
# RC: CONTROL SETS (named subsets)
# ============================================================

print("Running control set variants...")

# No controls (signal only, no signalfree)
run_spec(
    "rc/controls/sets/none",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "berl", "signal", [],
    "district_str + sll2001_str", "district + sll2001", df_base,
    {"CRV1": "district_str"},
    "1994, trimmed", "none (signal only, FE absorb location)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/none", "family": "sets",
                "n_controls": 0, "set_name": "none"})

# Signal + signalfree only
run_spec(
    "rc/controls/sets/signal_only",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "berl", "signal", ["signalfree"],
    "district_str + sll2001_str", "district + sll2001", df_base,
    {"CRV1": "district_str"},
    "1994, trimmed", "signalfree only",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/signal_only", "family": "sets",
                "n_controls": 1, "set_name": "signal_only"})

# Land controls only
run_spec(
    "rc/controls/sets/land_controls",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "berl", "signal", ["signalfree"] + LAND_CONTROLS,
    "district_str + sll2001_str", "district + sll2001", df_base,
    {"CRV1": "district_str"},
    "1994, trimmed", "signalfree + land controls (6)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/land_controls", "family": "sets",
                "n_controls": 6, "set_name": "land_controls"})

# Full controls (same as baseline, for completeness)
run_spec(
    "rc/controls/sets/full_controls",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "berl", "signal", ["signalfree"] + FULL_CONTROLS,
    "district_str + sll2001_str", "district + sll2001", df_base,
    {"CRV1": "district_str"},
    "1994, trimmed", "full controls (same as baseline)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/full_controls", "family": "sets",
                "n_controls": 9, "set_name": "full_controls"})


# ============================================================
# RC: CONTROL PROGRESSION (build-up from Table 3 structure)
# ============================================================

print("Running control progression variants...")

# Bivariate: signal only, no FE
run_spec(
    "rc/controls/progression/bivariate",
    "modules/robustness/controls.md#control-progression-build-up", "G1",
    "berl", "signal", [],
    "", "none", df_base,
    {"CRV1": "district_str"},
    "1994, trimmed", "bivariate (no controls, no FE)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/bivariate", "family": "progression",
                "n_controls": 0, "set_name": "bivariate"})

# Signal + signalfree, no FE (Table 3, Col 2)
run_spec(
    "rc/controls/progression/signal_signalfree",
    "modules/robustness/controls.md#control-progression-build-up", "G1",
    "berl", "signal", ["signalfree"],
    "", "none", df_base,
    {"CRV1": "district_str"},
    "1994, trimmed", "signal + signalfree, no FE",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/signal_signalfree", "family": "progression",
                "n_controls": 1, "set_name": "signal_signalfree"})

# Land controls, no FE (Table 3, Col 3)
run_spec(
    "rc/controls/progression/land",
    "modules/robustness/controls.md#control-progression-build-up", "G1",
    "berl", "signal", ["signalfree"] + LAND_CONTROLS,
    "", "none", df_base,
    {"CRV1": "district_str"},
    "1994, trimmed", "signalfree + land, no FE",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/land", "family": "progression",
                "n_controls": 6, "set_name": "land"})

# Land + sll FE (Table 3, Col 4)
run_spec(
    "rc/controls/progression/land_sll",
    "modules/robustness/controls.md#control-progression-build-up", "G1",
    "berl", "signal", ["signalfree"] + LAND_CONTROLS,
    "district_str + sll2001_str", "district + sll2001", df_base,
    {"CRV1": "district_str"},
    "1994, trimmed", "signalfree + land + FE",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/land_sll", "family": "progression",
                "n_controls": 6, "set_name": "land_sll"})

# Signal + signalfree + land + sll FE (no socioeconomic)
run_spec(
    "rc/controls/progression/signal_signalfree_land_sll",
    "modules/robustness/controls.md#control-progression-build-up", "G1",
    "berl", "signal", ["signalfree"] + LAND_CONTROLS,
    "district_str + sll2001_str", "district + sll2001", df_base,
    {"CRV1": "district_str"},
    "1994, trimmed", "signalfree + land + district+sll FE",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/signal_signalfree_land_sll",
                "family": "progression", "n_controls": 6})

# Full controls + FE (= baseline)
run_spec(
    "rc/controls/progression/full",
    "modules/robustness/controls.md#control-progression-build-up", "G1",
    "berl", "signal", ["signalfree"] + FULL_CONTROLS,
    "district_str + sll2001_str", "district + sll2001", df_base,
    {"CRV1": "district_str"},
    "1994, trimmed", "full controls + FE (= baseline)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/full", "family": "progression",
                "n_controls": 9, "set_name": "full"})


# ============================================================
# RC: ADD CONTROLS (civic81 and others from surface)
# ============================================================

print("Running add-control variants...")

# civic81 (from Table A8 — adds civic capital measure)
add_vars_specs = {
    "rc/controls/add/civic81": ("civic81", "civic capital 1981"),
    "rc/controls/add/dpop7181": ("dpop7181", "pop growth 1971-1981"),
    "rc/controls/add/actrate": ("actrate", "activity rate"),
    "rc/controls/add/emplrate": ("emplrate", "employment rate"),
    "rc/controls/add/firmspop81": ("firmspop81", "firms per capita 1981"),
    "rc/controls/add/workers81": ("workers81", "workers 1981"),
}

for spec_id, (add_var, desc) in add_vars_specs.items():
    # Need to restrict to non-missing for the added variable
    df_add = df_base.dropna(subset=[add_var]).copy()
    run_spec(
        spec_id, "modules/robustness/controls.md#adding-controls", "G1",
        "berl", "signal", ["signalfree"] + FULL_CONTROLS + [add_var],
        "district_str + sll2001_str", "district + sll2001", df_add,
        {"CRV1": "district_str"},
        f"1994, trimmed, N={len(df_add)}", f"full controls + {desc}",
        axis_block_name="controls",
        axis_block={"spec_id": spec_id, "family": "add",
                    "added": [add_var], "n_controls": 10})


# ============================================================
# RC: FIXED EFFECTS
# ============================================================

print("Running FE variants...")

# Drop sll2001 FE (keep only district)
run_spec(
    "rc/fe/drop/sll2001",
    "modules/robustness/fixed_effects.md#dropping-fe-relative-to-baseline", "G1",
    "berl", "signal", ["signalfree"] + FULL_CONTROLS,
    "district_str", "district only (no sll2001)", df_base,
    {"CRV1": "district_str"},
    "1994, trimmed", "full controls, district FE only",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/drop/sll2001", "family": "drop",
                "dropped": ["sll2001"], "baseline_fe": ["district", "sll2001"],
                "new_fe": ["district"]})

# Drop district FE (keep only sll2001)
run_spec(
    "rc/fe/drop/district",
    "modules/robustness/fixed_effects.md#dropping-fe-relative-to-baseline", "G1",
    "berl", "signal", ["signalfree"] + FULL_CONTROLS,
    "sll2001_str", "sll2001 only (no district)", df_base,
    {"CRV1": "district_str"},
    "1994, trimmed", "full controls, sll2001 FE only",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/drop/district", "family": "drop",
                "dropped": ["district"], "baseline_fe": ["district", "sll2001"],
                "new_fe": ["sll2001"]})

# Add province FE instead (coarser than district)
run_spec(
    "rc/fe/add/province",
    "modules/robustness/fixed_effects.md#additive-fe-variations-relative-to-baseline", "G1",
    "berl", "signal", ["signalfree"] + FULL_CONTROLS,
    "prov_str + sll2001_str", "province + sll2001", df_base,
    {"CRV1": "district_str"},
    "1994, trimmed", "full controls, province+sll FE",
    fixef_tol=1e-6,
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/add/province", "family": "swap",
                "added": ["province"], "dropped": ["district"],
                "baseline_fe": ["district", "sll2001"],
                "new_fe": ["province", "sll2001"]})


# ============================================================
# RC: SAMPLE RESTRICTIONS (Table 4 robustness)
# ============================================================

print("Running sample restriction variants...")

# No provincial capitals (Table 4, Col 2)
df_nocap = df_base[df_base['provcapital'] == 0].copy()
run_spec(
    "rc/sample/restrict/no_provincial_capitals",
    "modules/robustness/sample.md#sample-restriction", "G1",
    "berl", "signal", ["signalfree"] + FULL_CONTROLS,
    "district_str + sll2001_str", "district + sll2001", df_nocap,
    {"CRV1": "district_str"},
    f"1994, no provincial capitals, N={len(df_nocap)}", "full controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restrict/no_provincial_capitals",
                "axis": "sample_restriction", "rule": "drop provcapital==1",
                "n_obs_before": len(df_base), "n_obs_after": len(df_nocap)})

# Pop <= 100k (Table 4, Col 3) — pop81 is in thousands
df_pop100 = df_base[df_base['pop81'] <= 100].copy()
run_spec(
    "rc/sample/restrict/pop_le_100k",
    "modules/robustness/sample.md#sample-restriction", "G1",
    "berl", "signal", ["signalfree"] + FULL_CONTROLS,
    "district_str + sll2001_str", "district + sll2001", df_pop100,
    {"CRV1": "district_str"},
    f"1994, pop81<=100k, N={len(df_pop100)}", "full controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restrict/pop_le_100k",
                "axis": "sample_restriction", "rule": "pop81<=100",
                "n_obs_before": len(df_base), "n_obs_after": len(df_pop100)})

# Pop <= 50k (Table 4, Col 4)
df_pop50 = df_base[df_base['pop81'] <= 50].copy()
run_spec(
    "rc/sample/restrict/pop_le_50k",
    "modules/robustness/sample.md#sample-restriction", "G1",
    "berl", "signal", ["signalfree"] + FULL_CONTROLS,
    "district_str + sll2001_str", "district + sll2001", df_pop50,
    {"CRV1": "district_str"},
    f"1994, pop81<=50k, N={len(df_pop50)}", "full controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restrict/pop_le_50k",
                "axis": "sample_restriction", "rule": "pop81<=50",
                "n_obs_before": len(df_base), "n_obs_after": len(df_pop50)})

# Pop <= 10k (Table 4, Col 5)
df_pop10 = df_base[df_base['pop81'] <= 10].copy()
run_spec(
    "rc/sample/restrict/pop_le_10k",
    "modules/robustness/sample.md#sample-restriction", "G1",
    "berl", "signal", ["signalfree"] + FULL_CONTROLS,
    "district_str + sll2001_str", "district + sll2001", df_pop10,
    {"CRV1": "district_str"},
    f"1994, pop81<=10k, N={len(df_pop10)}", "full controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restrict/pop_le_10k",
                "axis": "sample_restriction", "rule": "pop81<=10",
                "n_obs_before": len(df_base), "n_obs_after": len(df_pop10)})

# Pop <= 5k
df_pop5 = df_base[df_base['pop81'] <= 5].copy()
run_spec(
    "rc/sample/restrict/pop_le_5k",
    "modules/robustness/sample.md#sample-restriction", "G1",
    "berl", "signal", ["signalfree"] + FULL_CONTROLS,
    "district_str + sll2001_str", "district + sll2001", df_pop5,
    {"CRV1": "district_str"},
    f"1994, pop81<=5k, N={len(df_pop5)}", "full controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restrict/pop_le_5k",
                "axis": "sample_restriction", "rule": "pop81<=5",
                "n_obs_before": len(df_base), "n_obs_after": len(df_pop5)})

# North only
df_north = df_base[df_base['north'] == 1].copy()
run_spec(
    "rc/sample/restrict/north_only",
    "modules/robustness/sample.md#sample-restriction", "G1",
    "berl", "signal", ["signalfree"] + FULL_CONTROLS,
    "district_str + sll2001_str", "district + sll2001", df_north,
    {"CRV1": "district_str"},
    f"1994, North only, N={len(df_north)}", "full controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restrict/north_only",
                "axis": "sample_restriction", "rule": "north==1",
                "n_obs_before": len(df_base), "n_obs_after": len(df_north)})

# South only
df_south = df_base[df_base['south'] == 1].copy()
run_spec(
    "rc/sample/restrict/south_only",
    "modules/robustness/sample.md#sample-restriction", "G1",
    "berl", "signal", ["signalfree"] + FULL_CONTROLS,
    "district_str + sll2001_str", "district + sll2001", df_south,
    {"CRV1": "district_str"},
    f"1994, South+Islands only, N={len(df_south)}", "full controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restrict/south_only",
                "axis": "sample_restriction", "rule": "south==1",
                "n_obs_before": len(df_base), "n_obs_after": len(df_south)})

# Center only
df_center = df_base[df_base['center'] == 1].copy()
run_spec(
    "rc/sample/restrict/center_only",
    "modules/robustness/sample.md#sample-restriction", "G1",
    "berl", "signal", ["signalfree"] + FULL_CONTROLS,
    "district_str + sll2001_str", "district + sll2001", df_center,
    {"CRV1": "district_str"},
    f"1994, Center only, N={len(df_center)}", "full controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restrict/center_only",
                "axis": "sample_restriction", "rule": "center==1",
                "n_obs_before": len(df_base), "n_obs_after": len(df_center)})


# ============================================================
# RC: SIGNAL TRIMMING (outlier robustness)
# ============================================================

print("Running signal trimming variants...")

# Trim signal at 5th/95th percentile
q05 = df_base['signal'].quantile(0.05)
q95 = df_base['signal'].quantile(0.95)
df_trim_5_95 = df_base[(df_base['signal'] >= q05) & (df_base['signal'] <= q95)].copy()
run_spec(
    "rc/sample/outliers/trim_signal_5_95",
    "modules/robustness/sample.md#outliers-and-influential-observations", "G1",
    "berl", "signal", ["signalfree"] + FULL_CONTROLS,
    "district_str + sll2001_str", "district + sll2001", df_trim_5_95,
    {"CRV1": "district_str"},
    f"1994, signal trimmed [5%,95%], N={len(df_trim_5_95)}", "full controls",
    fixef_tol=1e-6,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_signal_5_95", "axis": "outliers",
                "rule": "trim", "params": {"var": "signal", "lower_q": 0.05, "upper_q": 0.95},
                "n_obs_before": len(df_base), "n_obs_after": len(df_trim_5_95)})

# Trim signal at 10th/90th percentile
q10 = df_base['signal'].quantile(0.10)
q90 = df_base['signal'].quantile(0.90)
df_trim_10_90 = df_base[(df_base['signal'] >= q10) & (df_base['signal'] <= q90)].copy()
run_spec(
    "rc/sample/outliers/trim_signal_10_90",
    "modules/robustness/sample.md#outliers-and-influential-observations", "G1",
    "berl", "signal", ["signalfree"] + FULL_CONTROLS,
    "district_str + sll2001_str", "district + sll2001", df_trim_10_90,
    {"CRV1": "district_str"},
    f"1994, signal trimmed [10%,90%], N={len(df_trim_10_90)}", "full controls",
    fixef_tol=1e-4,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_signal_10_90", "axis": "outliers",
                "rule": "trim", "params": {"var": "signal", "lower_q": 0.10, "upper_q": 0.90},
                "n_obs_before": len(df_base), "n_obs_after": len(df_trim_10_90)})


# ============================================================
# RC: WEIGHTS
# ============================================================

print("Running weight variants...")

# Unweighted
run_spec(
    "rc/weights/unweighted",
    "modules/robustness/weights.md#weight-alternatives", "G1",
    "berl", "signal", ["signalfree"] + FULL_CONTROLS,
    "district_str + sll2001_str", "district + sll2001", df_base,
    {"CRV1": "district_str"},
    "1994, trimmed, unweighted", "full controls (unweighted)",
    weights_var=None,
    axis_block_name="weights",
    axis_block={"spec_id": "rc/weights/unweighted", "family": "weights",
                "weights": "none"})

# pop81 weighted (same as baseline for reference)
run_spec(
    "rc/weights/pop81",
    "modules/robustness/weights.md#weight-alternatives", "G1",
    "berl", "signal", ["signalfree"] + FULL_CONTROLS,
    "district_str + sll2001_str", "district + sll2001", df_base,
    {"CRV1": "district_str"},
    "1994, trimmed, pop81 weights", "full controls (pop81 weights)",
    weights_var="pop81",
    axis_block_name="weights",
    axis_block={"spec_id": "rc/weights/pop81", "family": "weights",
                "weights": "pop81"})


# ============================================================
# RC: TREATMENT FORM — capsignal (capped signal)
# ============================================================

print("Running treatment form variant (capsignal)...")

# Table 3, Col 7: uses capsignal from the UNTRIMMED data
# But we need capsignal in the trimmed sample. It's already available.
run_spec(
    "rc/form/treatment/capsignal",
    "modules/robustness/functional_form.md#treatment-form", "G1",
    "berl", "capsignal", ["signalfree"] + FULL_CONTROLS,
    "district_str + sll2001_str", "district + sll2001", df_base,
    {"CRV1": "district_str"},
    "1994, trimmed, capped signal", "full controls (capsignal instead of signal)",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/treatment/capsignal",
                "family": "treatment_form", "treatment": "capsignal",
                "notes": "Signal capped at 0 (only positive values contribute)"})


# ============================================================
# RC: MATCHED NEIGHBORS (Table 4, Cols 6-8)
# ============================================================

print("Running matched neighbors variants...")

df_mn = pd.read_stata(MATCHED_DATA_PATH, convert_categoricals=False)
for col in df_mn.columns:
    if df_mn[col].dtype == np.float32:
        df_mn[col] = df_mn[col].astype(np.float64)

df_mn['couple_str'] = df_mn['couple'].astype(str)
df_mn['istat_str'] = df_mn['istat'].astype(str)

# Matched neighbors use dumsignal as treatment, couple FE, cluster by couple
# Controls are the full set: area area2 altitude altitude2 ruggedness electorate lnincome highschool_college81
CONTR = ["area", "area2", "altitude", "altitude2", "ruggedness", "electorate", "lnincome", "highschool_college81"]

for dist_label, dist_var in [("distsignal1", "distsignal1"),
                              ("distsignal05", "distsignal05"),
                              ("distsignal025", "distsignal025")]:
    df_mn_sub = df_mn[(df_mn['year'] == 1994) & (df_mn[dist_var] == 1)].copy()
    df_mn_sub = df_mn_sub.dropna(subset=["berl", "dumsignal"] + CONTR + ["couple"]).copy()

    spec_id = f"rc/data/matching/matched_neighbors_{dist_label}"
    run_spec(
        spec_id, "modules/robustness/data_construction.md#matched-neighbors", "G1",
        "berl", "dumsignal", CONTR,
        "couple_str", "couple (matched pair)", df_mn_sub,
        {"CRV1": "couple_str"},
        f"1994, matched neighbors ({dist_label}), N={len(df_mn_sub)}", "full controls",
        cluster_var="couple_str", weights_var=None,
        axis_block_name="data_construction",
        axis_block={"spec_id": spec_id, "family": "matching",
                    "matched_on": dist_label, "treatment": "dumsignal",
                    "fe": "couple", "n_obs": len(df_mn_sub)})


# ============================================================
# RC: YEAR-BY-YEAR ESTIMATES (Table 5, berl across years)
# ============================================================

print("Running year-by-year variants...")

for yr in [1994, 1996, 2001, 2006, 2008, 2013]:
    df_yr = df_raw[df_raw['year'] == yr].copy()
    base_yr_vars = ["berl", "signal", "signalfree"] + FULL_CONTROLS + ["district", "sll2001", "pop81"]
    df_yr = df_yr.dropna(subset=base_yr_vars).copy()

    if len(df_yr) < 50:
        print(f"  Skipping year {yr}: only {len(df_yr)} obs")
        continue

    spec_id = f"rc/sample/restrict/year_{yr}"
    run_spec(
        spec_id, "modules/robustness/sample.md#time-restriction", "G1",
        "berl", "signal", ["signalfree"] + FULL_CONTROLS,
        "district_str + sll2001_str", "district + sll2001", df_yr,
        {"CRV1": "district_str"},
        f"year={yr}, trimmed, N={len(df_yr)}", "full controls",
        axis_block_name="sample",
        axis_block={"spec_id": spec_id, "axis": "time_restriction",
                    "year": yr, "n_obs": len(df_yr)})


# ============================================================
# INFERENCE VARIANTS (on baseline specification)
# ============================================================

print("\nRunning inference variants...")

baseline_run_id = f"{PAPER_ID}_run_001"
infer_counter = 0


def run_inference_variant(base_run_id, spec_id, spec_tree_path, baseline_group_id,
                          formula_str, fe_str, data, focal_var, vcov, vcov_desc,
                          weights_var="pop81"):
    """Re-run baseline spec with different variance-covariance estimator."""
    global infer_counter
    infer_counter += 1
    infer_run_id = f"{PAPER_ID}_infer_{infer_counter:03d}"

    try:
        if fe_str:
            full_formula = f"{formula_str} | {fe_str}"
        else:
            full_formula = formula_str

        kwargs = {"data": data, "vcov": vcov}
        if weights_var:
            kwargs["weights"] = weights_var

        m = pf.feols(full_formula, **kwargs)

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
baseline_controls_str = " + ".join(["signalfree"] + FULL_CONTROLS)
baseline_formula = f"berl ~ signal + {baseline_controls_str}"
baseline_fe = "district_str + sll2001_str"

# HC1 robust (no clustering)
run_inference_variant(
    baseline_run_id, "infer/se/hc/hc1",
    "modules/inference/standard_errors.md#heteroskedasticity-robust", "G1",
    baseline_formula, baseline_fe, df_base, "signal",
    "hetero", "HC1 (robust, no clustering)")

# Cluster by sll2001 (local labor system)
run_inference_variant(
    baseline_run_id, "infer/se/cluster/sll2001",
    "modules/inference/standard_errors.md#clustering", "G1",
    baseline_formula, baseline_fe, df_base, "signal",
    {"CRV1": "sll2001_str"}, "cluster(sll2001)")

# Two-way clustering (district x sll2001)
run_inference_variant(
    baseline_run_id, "infer/se/cluster/two_way_district_sll",
    "modules/inference/standard_errors.md#two-way-clustering", "G1",
    baseline_formula, baseline_fe, df_base, "signal",
    {"CRV1": "district_str + sll2001_str"}, "two-way cluster(district, sll2001)")


# ============================================================
# WRITE OUTPUTS
# ============================================================

print(f"\n{'='*60}")
print(f"Writing outputs...")
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
if len(failed) > 0:
    print("Failed specs:")
    for _, row in failed.iterrows():
        print(f"  {row['spec_id']}: {row['run_error'][:100]}")

if len(successful) > 0:
    base_row = spec_df[spec_df['spec_id'] == 'baseline']
    if len(base_row) > 0:
        print(f"\nBaseline coef on signal: {base_row['coefficient'].values[0]:.6f}")
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
md_lines.append("# Specification Search Report: 116167-V1")
md_lines.append("")
md_lines.append("**Paper:** Durante, Pinotti & Tesei (2019), \"The Political Legacy of Entertainment TV\", AER 109(7)")
md_lines.append("")
md_lines.append("## Baseline Specification")
md_lines.append("")
md_lines.append("- **Design:** Cross-sectional OLS")
md_lines.append("- **Outcome:** berl (Forza Italia vote share in 1994)")
md_lines.append("- **Treatment:** signal (early exposure to Mediaset commercial TV, standardized)")
md_lines.append("- **Controls:** signalfree + area, area2, altitude, altitude2, ruggedness, electorate, lnincome, highschool_college81")
md_lines.append("- **Fixed effects:** district + sll2001 (local labor system)")
md_lines.append("- **Clustering:** district")
md_lines.append("- **Weights:** pop81 (population weights)")
md_lines.append("- **Sample:** Trimmed top/bottom 2.5% of signal distribution")
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
    "Controls Add": successful[successful['spec_id'].str.startswith('rc/controls/add/')],
    "Fixed Effects": successful[successful['spec_id'].str.startswith('rc/fe/')],
    "Sample Restriction": successful[successful['spec_id'].str.startswith('rc/sample/restrict/')],
    "Sample Outliers": successful[successful['spec_id'].str.startswith('rc/sample/outliers/')],
    "Weights": successful[successful['spec_id'].str.startswith('rc/weights/')],
    "Functional Form": successful[successful['spec_id'].str.startswith('rc/form/')],
    "Matched Neighbors": successful[successful['spec_id'].str.startswith('rc/data/matching/')],
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
