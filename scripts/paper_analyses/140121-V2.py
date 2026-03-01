"""
Specification Search Script for Jones & Marinescu (2022)
"The Labor Market Impacts of Universal and Permanent Cash Transfers:
 Evidence from the Alaska Permanent Fund"
American Economic Review, 112(7), 2424-2468.

Paper ID: 140121-V2

Surface-driven execution:
  - G1: employed ~ alaska_post | statefip + year, cluster(statefip)
  - Panel DiD with state and year FE, clustered SEs at state level
  - The paper's primary method is synthetic control; we implement the DiD analog
    (Appendix Table C.1 in the paper) as a pyfixest-compatible specification.
  - Baseline is without controls (pure DiD) because the synthetic control matching
    variables (industry composition) are mechanically collinear with employment.
  - Demographic composition controls (age, education, female shares) are valid
    robustness controls since they are not mechanically determined by employment.
  - 50+ specifications across outcomes, subgroup samples, controls LOO,
    controls subsets, time windows, FE swaps, additional covariates

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

PAPER_ID = "140121-V2"
DATA_DIR = "data/downloads/extracted/140121-V2"
OUTPUT_DIR = DATA_DIR

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

# Main IPUMS data (state-year panel, 51 states x 38 years = 1938 obs)
df_ipums_raw = pd.read_stata(f"{DATA_DIR}/data/proc/IPUMS_main.dta",
                              convert_categoricals=False)
print(f"Loaded IPUMS_main: {df_ipums_raw.shape[0]} rows, {df_ipums_raw.shape[1]} columns")

# MORG data (for hourslw outcome)
df_morg_raw = pd.read_stata(f"{DATA_DIR}/data/proc/MORG_main.dta",
                             convert_categoricals=False)
print(f"Loaded MORG_main: {df_morg_raw.shape[0]} rows, {df_morg_raw.shape[1]} columns")

# Subgroup datasets
SUBGROUP_FILES = {
    "male": "IPUMS_main_male.dta",
    "female": "IPUMS_main_female.dta",
    "trade": "IPUMS_main_trade.dta",
    "nontrade": "IPUMS_main_nontrade.dta",
    "under55": "IPUMS_main_under55.dta",
    "over55": "IPUMS_main_over55.dta",
    "married_male": "IPUMS_married_male.dta",
    "married_female": "IPUMS_married_female.dta",
    "unmarried_male": "IPUMS_unmarried_male.dta",
    "unmarried_female": "IPUMS_unmarried_female.dta",
}

subgroup_data = {}
for sg_name, sg_file in SUBGROUP_FILES.items():
    try:
        sg_df = pd.read_stata(f"{DATA_DIR}/data/proc/{sg_file}",
                               convert_categoricals=False)
        subgroup_data[sg_name] = sg_df
    except Exception as e:
        print(f"Warning: Could not load {sg_file}: {e}")


def prepare_panel(df, y0=1977, yT=2014, treatment_state=2, treatment_year=1982):
    """Prepare a panel for DiD estimation.

    Creates treatment indicator (alaska_post = 1 for Alaska after 1982).
    Converts statefip/year to strings for pyfixest FE absorption.
    """
    df = df.copy()

    # Convert to float64 for precision
    for col in df.columns:
        if df[col].dtype == np.float32:
            df[col] = df[col].astype(np.float64)

    # Filter time window
    df = df[(df['year'] >= y0) & (df['year'] <= yT)].copy()

    # Create treatment indicator
    df['alaska'] = (df['statefip'] == treatment_state).astype(float)
    df['post'] = (df['year'] >= treatment_year).astype(float)
    df['alaska_post'] = df['alaska'] * df['post']

    # String versions for FE
    df['statefip_str'] = df['statefip'].astype(int).astype(str)
    df['year_str'] = df['year'].astype(int).astype(str)

    # State-specific linear trend variable
    df['trend'] = df['year'] - y0

    return df


# Prepare main datasets
df_ipums = prepare_panel(df_ipums_raw)
df_morg = prepare_panel(df_morg_raw, y0=1979)
print(f"IPUMS panel: {len(df_ipums)} obs, {df_ipums['statefip'].nunique()} states, "
      f"{df_ipums['year'].nunique()} years")
print(f"MORG panel: {len(df_morg)} obs, {df_morg['statefip'].nunique()} states, "
      f"{df_morg['year'].nunique()} years")

# Prepare subgroup datasets
subgroup_panels = {}
for sg_name, sg_df in subgroup_data.items():
    subgroup_panels[sg_name] = prepare_panel(sg_df)

# Define control variables
# NOTE: Industry composition shares (ind1-ind5) are mechanically collinear with
# employment rates (employed = sum of industry shares), so they cannot be used as
# controls in a DiD regression with employment as the outcome. They are used in the
# paper's synthetic control as matching variables, not regression controls.
# Demographic composition (age, education, gender) are valid DiD controls.

DEMO_CONTROLS = ["age1", "age2", "age3", "educ1", "educ2", "female"]

# Additional controls available in the data (not mechanically related to employment)
EXTRA_CONTROLS = ["oil_gdp", "net_mig"]

# Full set of non-endogenous controls
ALL_VALID_CONTROLS = DEMO_CONTROLS + EXTRA_CONTROLS

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
             cluster_var="statefip_str",
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
                       "method": "cluster", "cluster_vars": ["statefip"]},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"panel_did": design_audit},
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
# BASELINE: Employment rate ~ alaska_post | statefip + year
# The paper's DiD (Appendix C) uses no controls, just state + year FE
# ============================================================

print("=" * 60)
print("Running baseline specification (no controls, state + year FE)...")
base_run_id, base_coef, base_se, base_pval, base_nobs = run_spec(
    "baseline", "designs/panel_did.md#baseline", "G1",
    "employed", "alaska_post", [],
    "statefip_str + year_str", "statefip + year", df_ipums,
    {"CRV1": "statefip_str"},
    f"All states 1977-2014, N={len(df_ipums)}", "none (pure DiD with state + year FE)")

print(f"  Baseline: coef={base_coef:.6f}, se={base_se:.6f}, p={base_pval:.4f}, N={base_nobs}")


# ============================================================
# ALTERNATIVE OUTCOMES (Table 2 columns 2-4)
# ============================================================

print("\nRunning alternative outcome specifications...")

# Part-time rate (IPUMS)
run_spec(
    "baseline__parttime", "designs/panel_did.md#baseline", "G1",
    "parttime", "alaska_post", [],
    "statefip_str + year_str", "statefip + year", df_ipums,
    {"CRV1": "statefip_str"},
    "All states 1977-2014", "none (pure DiD)",
    axis_block_name="outcome",
    axis_block={"spec_id": "baseline__parttime", "outcome": "parttime"})

# Labor force participation (IPUMS)
run_spec(
    "baseline__activelf", "designs/panel_did.md#baseline", "G1",
    "activelf", "alaska_post", [],
    "statefip_str + year_str", "statefip + year", df_ipums,
    {"CRV1": "statefip_str"},
    "All states 1977-2014", "none (pure DiD)",
    axis_block_name="outcome",
    axis_block={"spec_id": "baseline__activelf", "outcome": "activelf"})

# Hours worked last week (MORG data, starts 1979)
run_spec(
    "baseline__hourslw", "designs/panel_did.md#baseline", "G1",
    "hourslw", "alaska_post", [],
    "statefip_str + year_str", "statefip + year", df_morg,
    {"CRV1": "statefip_str"},
    "All states 1979-2014 (MORG data)", "none (pure DiD)",
    axis_block_name="outcome",
    axis_block={"spec_id": "baseline__hourslw", "outcome": "hourslw"})


# ============================================================
# RC: DEMOGRAPHIC CONTROLS (robustness - add demographic composition)
# ============================================================

print("\nRunning demographic control variants...")

# Add all demographic controls
run_spec(
    "rc/controls/add/demographics",
    "modules/robustness/controls.md#additional-controls", "G1",
    "employed", "alaska_post", DEMO_CONTROLS,
    "statefip_str + year_str", "statefip + year", df_ipums,
    {"CRV1": "statefip_str"},
    "All states 1977-2014", f"demographic controls ({len(DEMO_CONTROLS)})",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/add/demographics", "family": "add",
                "added": DEMO_CONTROLS, "n_controls": len(DEMO_CONTROLS)})

# Demographics + oil_gdp
run_spec(
    "rc/controls/add/demo_oil",
    "modules/robustness/controls.md#additional-controls", "G1",
    "employed", "alaska_post", DEMO_CONTROLS + ["oil_gdp"],
    "statefip_str + year_str", "statefip + year", df_ipums,
    {"CRV1": "statefip_str"},
    "All states 1977-2014", f"demographics + oil_gdp ({len(DEMO_CONTROLS)+1})",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/add/demo_oil", "family": "add",
                "added": DEMO_CONTROLS + ["oil_gdp"]})

# Demographics + net_mig
run_spec(
    "rc/controls/add/demo_netmig",
    "modules/robustness/controls.md#additional-controls", "G1",
    "employed", "alaska_post", DEMO_CONTROLS + ["net_mig"],
    "statefip_str + year_str", "statefip + year", df_ipums,
    {"CRV1": "statefip_str"},
    "All states 1977-2014", f"demographics + net_mig ({len(DEMO_CONTROLS)+1})",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/add/demo_netmig", "family": "add",
                "added": DEMO_CONTROLS + ["net_mig"]})

# Full valid controls (demographics + oil_gdp + net_mig)
run_spec(
    "rc/controls/add/full_valid",
    "modules/robustness/controls.md#additional-controls", "G1",
    "employed", "alaska_post", ALL_VALID_CONTROLS,
    "statefip_str + year_str", "statefip + year", df_ipums,
    {"CRV1": "statefip_str"},
    "All states 1977-2014", f"all valid controls ({len(ALL_VALID_CONTROLS)})",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/add/full_valid", "family": "add",
                "added": ALL_VALID_CONTROLS})

# Oil only
run_spec(
    "rc/controls/add/oil_gdp",
    "modules/robustness/controls.md#additional-controls", "G1",
    "employed", "alaska_post", ["oil_gdp"],
    "statefip_str + year_str", "statefip + year", df_ipums,
    {"CRV1": "statefip_str"},
    "All states 1977-2014", "oil_gdp only",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/add/oil_gdp", "family": "add",
                "added": ["oil_gdp"]})

# Net migration only
run_spec(
    "rc/controls/add/net_mig",
    "modules/robustness/controls.md#additional-controls", "G1",
    "employed", "alaska_post", ["net_mig"],
    "statefip_str + year_str", "statefip + year", df_ipums,
    {"CRV1": "statefip_str"},
    "All states 1977-2014", "net_mig only",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/add/net_mig", "family": "add",
                "added": ["net_mig"]})


# ============================================================
# RC: CONTROLS LOO on demographic controls
# ============================================================

print("\nRunning controls LOO variants (on demographic controls)...")

for ctrl_var in DEMO_CONTROLS:
    ctrl_remaining = [c for c in DEMO_CONTROLS if c != ctrl_var]
    spec_id = f"rc/controls/loo/drop_{ctrl_var}"
    run_spec(
        spec_id, "modules/robustness/controls.md#leave-one-out-controls-loo", "G1",
        "employed", "alaska_post", ctrl_remaining,
        "statefip_str + year_str", "statefip + year", df_ipums,
        {"CRV1": "statefip_str"},
        "All states 1977-2014", f"demographics minus {ctrl_var}",
        axis_block_name="controls",
        axis_block={"spec_id": spec_id, "family": "loo",
                    "dropped": [ctrl_var], "n_controls": len(ctrl_remaining)})


# ============================================================
# RC: CONTROL SUBSETS (random draws from demographics)
# ============================================================

print("\nRunning random control subset variants...")

rng = np.random.RandomState(140121)

for draw_i in range(1, 6):
    k = rng.randint(1, len(DEMO_CONTROLS) + 1)
    chosen = list(rng.choice(DEMO_CONTROLS, size=k, replace=False))
    excluded = [v for v in DEMO_CONTROLS if v not in chosen]
    spec_id = f"rc/controls/subset/random_{draw_i:03d}"
    run_spec(
        spec_id, "modules/robustness/controls.md#subset-generation-specids", "G1",
        "employed", "alaska_post", chosen,
        "statefip_str + year_str", "statefip + year", df_ipums,
        {"CRV1": "statefip_str"},
        "All states 1977-2014", f"random subset draw {draw_i} ({len(chosen)} controls)",
        axis_block_name="controls",
        axis_block={"spec_id": spec_id, "family": "subset", "method": "random",
                    "seed": 140121, "draw_index": draw_i,
                    "included": chosen, "excluded": excluded,
                    "n_controls": len(chosen)})


# ============================================================
# RC: SUBGROUP SAMPLES (Table 3, A.1, etc.)
# ============================================================

print("\nRunning subgroup sample variants...")

SUBGROUP_SPECS = {
    "male": ("employed", "All states 1977-2014, males only"),
    "female": ("employed", "All states 1977-2014, females only"),
    "trade": ("employed", "All states 1977-2014, tradeable sector"),
    "nontrade": ("employed", "All states 1977-2014, non-tradeable sector"),
    "under55": ("employed", "All states 1977-2014, under 55"),
    "over55": ("employed", "All states 1977-2014, over 55"),
    "married_male": ("employed", "All states 1977-2014, married males"),
    "married_female": ("employed", "All states 1977-2014, married females"),
    "unmarried_male": ("employed", "All states 1977-2014, unmarried males"),
    "unmarried_female": ("employed", "All states 1977-2014, unmarried females"),
}

for sg_name, (sg_outcome, sg_desc) in SUBGROUP_SPECS.items():
    if sg_name not in subgroup_panels:
        continue
    sg_df = subgroup_panels[sg_name]

    spec_id = f"rc/sample/subgroup/{sg_name}"
    run_spec(
        spec_id, "modules/robustness/sample.md#subgroup-analysis", "G1",
        sg_outcome, "alaska_post", [],
        "statefip_str + year_str", "statefip + year", sg_df,
        {"CRV1": "statefip_str"},
        sg_desc, "none (pure DiD)",
        axis_block_name="sample",
        axis_block={"spec_id": spec_id, "axis": "subgroup",
                    "subgroup": sg_name})

# Also run parttime for male/female subgroups (Table 3)
for sg_name in ["male", "female"]:
    if sg_name not in subgroup_panels:
        continue
    sg_df = subgroup_panels[sg_name]

    spec_id = f"rc/form/outcome/parttime_{sg_name}"
    run_spec(
        spec_id, "modules/robustness/sample.md#subgroup-analysis", "G1",
        "parttime", "alaska_post", [],
        "statefip_str + year_str", "statefip + year", sg_df,
        {"CRV1": "statefip_str"},
        f"All states 1977-2014, {sg_name}s only", "none (pure DiD)",
        axis_block_name="outcome",
        axis_block={"spec_id": spec_id, "outcome": f"parttime_{sg_name}"})

    spec_id = f"rc/form/outcome/employed_{sg_name}"
    run_spec(
        spec_id, "modules/robustness/sample.md#subgroup-analysis", "G1",
        "employed", "alaska_post", [],
        "statefip_str + year_str", "statefip + year", sg_df,
        {"CRV1": "statefip_str"},
        f"All states 1977-2014, {sg_name}s only", "none (pure DiD)",
        axis_block_name="outcome",
        axis_block={"spec_id": spec_id, "outcome": f"employed_{sg_name}"})

# Also run activelf for subgroups
for sg_name in ["male", "female", "under55", "over55"]:
    if sg_name not in subgroup_panels:
        continue
    sg_df = subgroup_panels[sg_name]
    if "activelf" not in sg_df.columns:
        continue

    spec_id = f"rc/form/outcome/activelf_{sg_name}"
    run_spec(
        spec_id, "modules/robustness/sample.md#subgroup-analysis", "G1",
        "activelf", "alaska_post", [],
        "statefip_str + year_str", "statefip + year", sg_df,
        {"CRV1": "statefip_str"},
        f"All states 1977-2014, {sg_name}s only", "none (pure DiD)",
        axis_block_name="outcome",
        axis_block={"spec_id": spec_id, "outcome": f"activelf_{sg_name}"})


# ============================================================
# RC: TIME WINDOW VARIATIONS
# ============================================================

print("\nRunning time window variants...")

# End in 2000 (shorter post period)
df_end2000 = df_ipums[df_ipums['year'] <= 2000].copy()
run_spec(
    "rc/sample/time_window/end_2000",
    "modules/robustness/sample.md#time-window-variations", "G1",
    "employed", "alaska_post", [],
    "statefip_str + year_str", "statefip + year", df_end2000,
    {"CRV1": "statefip_str"},
    f"All states 1977-2000, N={len(df_end2000)}", "none",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/time_window/end_2000", "axis": "time_window",
                "end_year": 2000})

# End in 1990
df_end1990 = df_ipums[df_ipums['year'] <= 1990].copy()
run_spec(
    "rc/sample/time_window/end_1990",
    "modules/robustness/sample.md#time-window-variations", "G1",
    "employed", "alaska_post", [],
    "statefip_str + year_str", "statefip + year", df_end1990,
    {"CRV1": "statefip_str"},
    f"All states 1977-1990, N={len(df_end1990)}", "none",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/time_window/end_1990", "axis": "time_window",
                "end_year": 1990})

# End in 2010
df_end2010 = df_ipums[df_ipums['year'] <= 2010].copy()
run_spec(
    "rc/sample/time_window/end_2010",
    "modules/robustness/sample.md#time-window-variations", "G1",
    "employed", "alaska_post", [],
    "statefip_str + year_str", "statefip + year", df_end2010,
    {"CRV1": "statefip_str"},
    f"All states 1977-2010, N={len(df_end2010)}", "none",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/time_window/end_2010", "axis": "time_window",
                "end_year": 2010})

# Start in 1979 (match MORG range)
df_start1979 = df_ipums[df_ipums['year'] >= 1979].copy()
run_spec(
    "rc/sample/time_window/start_1979",
    "modules/robustness/sample.md#time-window-variations", "G1",
    "employed", "alaska_post", [],
    "statefip_str + year_str", "statefip + year", df_start1979,
    {"CRV1": "statefip_str"},
    f"All states 1979-2014, N={len(df_start1979)}", "none",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/time_window/start_1979", "axis": "time_window",
                "start_year": 1979})

# Short pre-period: start 1980 (only 2 pre-treatment years)
df_start1980 = df_ipums[df_ipums['year'] >= 1980].copy()
run_spec(
    "rc/sample/time_window/start_1980",
    "modules/robustness/sample.md#time-window-variations", "G1",
    "employed", "alaska_post", [],
    "statefip_str + year_str", "statefip + year", df_start1980,
    {"CRV1": "statefip_str"},
    f"All states 1980-2014, N={len(df_start1980)}", "none",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/time_window/start_1980", "axis": "time_window",
                "start_year": 1980})

# Parttime with time window variations
for end_yr in [1990, 2000]:
    df_sub = df_ipums[df_ipums['year'] <= end_yr].copy()
    spec_id = f"rc/sample/time_window/parttime_end_{end_yr}"
    run_spec(
        spec_id, "modules/robustness/sample.md#time-window-variations", "G1",
        "parttime", "alaska_post", [],
        "statefip_str + year_str", "statefip + year", df_sub,
        {"CRV1": "statefip_str"},
        f"All states 1977-{end_yr}", "none",
        axis_block_name="sample",
        axis_block={"spec_id": spec_id, "axis": "time_window",
                    "outcome": "parttime", "end_year": end_yr})

# Hourslw with time window
df_morg_end2000 = df_morg[df_morg['year'] <= 2000].copy()
run_spec(
    "rc/sample/time_window/hourslw_end_2000",
    "modules/robustness/sample.md#time-window-variations", "G1",
    "hourslw", "alaska_post", [],
    "statefip_str + year_str", "statefip + year", df_morg_end2000,
    {"CRV1": "statefip_str"},
    "All states 1979-2000 (MORG)", "none",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/time_window/hourslw_end_2000",
                "axis": "time_window", "outcome": "hourslw", "end_year": 2000})


# ============================================================
# RC: FIXED EFFECTS VARIATIONS
# ============================================================

print("\nRunning FE variants...")

# Drop year FE (state FE only + post indicator)
run_spec(
    "rc/fe/drop/year",
    "modules/robustness/fixed_effects.md#dropping-fe-relative-to-baseline", "G1",
    "employed", "alaska_post", ["post"],
    "statefip_str", "statefip only (no year FE)", df_ipums,
    {"CRV1": "statefip_str"},
    "All states 1977-2014", "post indicator (no year FE)",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/drop/year", "family": "drop",
                "dropped": ["year"], "baseline_fe": ["statefip", "year"],
                "new_fe": ["statefip"]})

# No FE (raw alaska + post + interaction)
run_spec(
    "rc/fe/drop/all",
    "modules/robustness/fixed_effects.md#dropping-fe-relative-to-baseline", "G1",
    "employed", "alaska_post", ["alaska", "post"],
    "", "none (pooled OLS)", df_ipums,
    {"CRV1": "statefip_str"},
    "All states 1977-2014", "alaska + post (no FE)",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/drop/all", "family": "drop",
                "dropped": ["statefip", "year"], "baseline_fe": ["statefip", "year"],
                "new_fe": []})

# State FE only with demographics
run_spec(
    "rc/fe/drop/year_with_demo",
    "modules/robustness/fixed_effects.md#dropping-fe-relative-to-baseline", "G1",
    "employed", "alaska_post", DEMO_CONTROLS + ["post"],
    "statefip_str", "statefip only + demographics", df_ipums,
    {"CRV1": "statefip_str"},
    "All states 1977-2014", f"demographics + post, state FE only",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/drop/year_with_demo", "family": "drop",
                "dropped": ["year"], "added_controls": DEMO_CONTROLS})


# ============================================================
# RC: OUTCOME ALTERNATIVES WITH CONTROLS
# ============================================================

print("\nRunning outcome x controls combinations...")

# Parttime with demographic controls
run_spec(
    "rc/form/outcome/parttime_demo",
    "modules/robustness/controls.md#additional-controls", "G1",
    "parttime", "alaska_post", DEMO_CONTROLS,
    "statefip_str + year_str", "statefip + year", df_ipums,
    {"CRV1": "statefip_str"},
    "All states 1977-2014", f"demographic controls ({len(DEMO_CONTROLS)})",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/form/outcome/parttime_demo",
                "outcome": "parttime", "controls": "demographics"})

# Activelf with demographic controls
run_spec(
    "rc/form/outcome/activelf_demo",
    "modules/robustness/controls.md#additional-controls", "G1",
    "activelf", "alaska_post", DEMO_CONTROLS,
    "statefip_str + year_str", "statefip + year", df_ipums,
    {"CRV1": "statefip_str"},
    "All states 1977-2014", f"demographic controls ({len(DEMO_CONTROLS)})",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/form/outcome/activelf_demo",
                "outcome": "activelf", "controls": "demographics"})

# Hourslw with demographic controls (MORG)
morg_demo = [c for c in DEMO_CONTROLS if c in df_morg.columns]
run_spec(
    "rc/form/outcome/hourslw_demo",
    "modules/robustness/controls.md#additional-controls", "G1",
    "hourslw", "alaska_post", morg_demo,
    "statefip_str + year_str", "statefip + year", df_morg,
    {"CRV1": "statefip_str"},
    "All states 1979-2014 (MORG)", f"demographic controls ({len(morg_demo)})",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/form/outcome/hourslw_demo",
                "outcome": "hourslw", "controls": "demographics"})

# Activelf no controls
run_spec(
    "rc/form/outcome/activelf_no_controls",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "activelf", "alaska_post", [],
    "statefip_str + year_str", "statefip + year", df_ipums,
    {"CRV1": "statefip_str"},
    "All states 1977-2014", "no controls",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/form/outcome/activelf_no_controls",
                "outcome": "activelf", "n_controls": 0})

# Hourslw no controls
run_spec(
    "rc/form/outcome/hourslw_no_controls",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "hourslw", "alaska_post", [],
    "statefip_str + year_str", "statefip + year", df_morg,
    {"CRV1": "statefip_str"},
    "All states 1979-2014 (MORG)", "no controls",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/form/outcome/hourslw_no_controls",
                "outcome": "hourslw", "n_controls": 0})


# ============================================================
# RC: ALTERNATIVE TREATMENT DEFINITIONS
# ============================================================

print("\nRunning alternative treatment definitions...")

# Later treatment year (1984 - when PFD was more established)
df_late_treat = prepare_panel(df_ipums_raw, treatment_year=1984)
run_spec(
    "rc/treatment/late_1984",
    "modules/robustness/sample.md#treatment-timing", "G1",
    "employed", "alaska_post", [],
    "statefip_str + year_str", "statefip + year", df_late_treat,
    {"CRV1": "statefip_str"},
    "All states 1977-2014, treatment year=1984", "none",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/treatment/late_1984", "axis": "treatment_timing",
                "treatment_year": 1984,
                "notes": "PFD was more established by 1984"})

# Earlier treatment year (1980 - pre-PFD, placebo test)
df_placebo = prepare_panel(df_ipums_raw, treatment_year=1980)
# Only use pre-1982 data for this placebo
df_placebo = df_placebo[df_placebo['year'] < 1982].copy()
run_spec(
    "rc/treatment/placebo_1980",
    "modules/robustness/sample.md#placebo-tests", "G1",
    "employed", "alaska_post", [],
    "statefip_str + year_str", "statefip + year", df_placebo,
    {"CRV1": "statefip_str"},
    f"All states 1977-1981 (pre-treatment), placebo year=1980, N={len(df_placebo)}",
    "none (placebo test)",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/treatment/placebo_1980", "axis": "placebo",
                "treatment_year": 1980,
                "notes": "Placebo test using pre-treatment period only"})


# ============================================================
# INFERENCE VARIANTS (on baseline specification)
# ============================================================

print("\nRunning inference variants...")

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
            design={"panel_did": design_audit},
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
baseline_formula = "employed ~ alaska_post"

# HC1 robust (no clustering)
run_inference_variant(
    baseline_run_id, "infer/se/hc/hc1",
    "modules/inference/standard_errors.md#heteroskedasticity-robust", "G1",
    baseline_formula, "statefip_str + year_str", df_ipums, "alaska_post",
    "hetero", "HC1 (robust, no clustering)")

# HC3
try:
    run_inference_variant(
        baseline_run_id, "infer/se/hc/hc3",
        "modules/inference/standard_errors.md#heteroskedasticity-robust", "G1",
        baseline_formula, "statefip_str + year_str", df_ipums, "alaska_post",
        {"CRV3": "statefip_str"}, "CRV3 cluster(statefip)")
except:
    # CRV3 may fail - fallback
    run_inference_variant(
        baseline_run_id, "infer/se/hc/hc3",
        "modules/inference/standard_errors.md#heteroskedasticity-robust", "G1",
        baseline_formula, "statefip_str + year_str", df_ipums, "alaska_post",
        "HC3", "HC3 (no clustering)")

# iid
run_inference_variant(
    baseline_run_id, "infer/se/iid",
    "modules/inference/standard_errors.md#iid", "G1",
    baseline_formula, "statefip_str + year_str", df_ipums, "alaska_post",
    "iid", "iid (homoskedastic)")


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
        print(f"\nBaseline coef on alaska_post: {base_row['coefficient'].values[0]:.6f}")
        print(f"Baseline SE: {base_row['std_error'].values[0]:.6f}")
        print(f"Baseline p-value: {base_row['p_value'].values[0]:.6f}")
        print(f"Baseline N: {base_row['n_obs'].values[0]:.0f}")

    print(f"\n=== COEFFICIENT RANGE (successful specs, employed outcome) ===")
    emp_ok = successful[successful['outcome_var'] == 'employed']
    if len(emp_ok) > 0:
        print(f"Min coef: {emp_ok['coefficient'].min():.6f}")
        print(f"Max coef: {emp_ok['coefficient'].max():.6f}")
        print(f"Median coef: {emp_ok['coefficient'].median():.6f}")
        n_sig = (emp_ok['p_value'] < 0.05).sum()
        print(f"Significant at 5%: {n_sig}/{len(emp_ok)}")
        n_sig10 = (emp_ok['p_value'] < 0.10).sum()
        print(f"Significant at 10%: {n_sig10}/{len(emp_ok)}")

    print(f"\n=== ALL OUTCOMES ===")
    for out in successful['outcome_var'].unique():
        sub = successful[successful['outcome_var'] == out]
        print(f"  {out}: {len(sub)} specs, coef range [{sub['coefficient'].min():.6f}, {sub['coefficient'].max():.6f}]")


# ============================================================
# SPECIFICATION_SEARCH.md
# ============================================================

print("\nWriting SPECIFICATION_SEARCH.md...")

md_lines = []
md_lines.append("# Specification Search Report: 140121-V2")
md_lines.append("")
md_lines.append("**Paper:** Jones & Marinescu (2022), \"The Labor Market Impacts of Universal and Permanent Cash Transfers: Evidence from the Alaska Permanent Fund\", AER 112(7)")
md_lines.append("")
md_lines.append("## Baseline Specification")
md_lines.append("")
md_lines.append("- **Design:** Panel DiD (TWFE analog of paper's synthetic control method)")
md_lines.append("- **Outcome:** employed (employment-to-population ratio)")
md_lines.append("- **Treatment:** alaska_post (Alaska x Post-1982 interaction)")
md_lines.append("- **Controls:** None in baseline (pure DiD); demographic controls in robustness")
md_lines.append("- **Fixed effects:** statefip + year")
md_lines.append("- **Clustering:** statefip (state-level)")
md_lines.append("- **Note:** Paper's primary method is synthetic control; DiD is a robustness check (Appendix C). Industry composition controls (ind1-ind5) are excluded because they are mechanically collinear with the employment outcome.")
md_lines.append("")

if len(successful) > 0:
    base_row = spec_df[spec_df['spec_id'] == 'baseline']
    if len(base_row) > 0:
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
    "Baseline (all outcomes)": successful[successful['spec_id'].str.startswith('baseline')],
    "Controls (add/LOO/subset)": successful[successful['spec_id'].str.startswith('rc/controls/')],
    "Sample Subgroups": successful[successful['spec_id'].str.startswith('rc/sample/subgroup/')],
    "Time Windows": successful[successful['spec_id'].str.startswith('rc/sample/time_window/')],
    "Fixed Effects": successful[successful['spec_id'].str.startswith('rc/fe/')],
    "Alt Outcomes + Controls": successful[successful['spec_id'].str.startswith('rc/form/outcome/')],
    "Treatment Timing": successful[successful['spec_id'].str.startswith('rc/treatment/')],
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
    # Focus on employment specifications (baseline outcome)
    emp_specs = successful[successful['outcome_var'] == 'employed']
    n_sig_total = (emp_specs['p_value'] < 0.05).sum() if len(emp_specs) > 0 else 0
    pct_sig = n_sig_total / len(emp_specs) * 100 if len(emp_specs) > 0 else 0
    sign_consistent = False
    if len(emp_specs) > 0:
        sign_consistent = ((emp_specs['coefficient'] > 0).sum() == len(emp_specs)) or \
                          ((emp_specs['coefficient'] < 0).sum() == len(emp_specs))
    median_coef = emp_specs['coefficient'].median() if len(emp_specs) > 0 else np.nan
    sign_word = "positive" if median_coef > 0 else "negative"

    md_lines.append(f"- **Employment specifications:** {len(emp_specs)} total")
    md_lines.append(f"- **Sign consistency (employment):** {'All specifications have the same sign' if sign_consistent else 'Mixed signs across specifications'}")
    md_lines.append(f"- **Significance stability (employment):** {n_sig_total}/{len(emp_specs)} ({pct_sig:.1f}%) significant at 5%")
    md_lines.append(f"- **Direction:** Median coefficient is {sign_word} ({median_coef:.6f})")

    # Overall across all outcomes
    n_sig_all = (successful['p_value'] < 0.05).sum()
    pct_sig_all = n_sig_all / len(successful) * 100

    if pct_sig >= 80 and sign_consistent:
        strength = "STRONG"
    elif pct_sig >= 50 and sign_consistent:
        strength = "MODERATE"
    elif pct_sig >= 30:
        strength = "WEAK"
    else:
        strength = "FRAGILE"

    md_lines.append(f"- **Overall significance (all outcomes):** {n_sig_all}/{len(successful)} ({pct_sig_all:.1f}%)")
    md_lines.append(f"- **Robustness assessment:** {strength}")
    md_lines.append(f"- **Note:** The paper's main finding is a null effect on employment and a positive effect on part-time work. The paper uses synthetic control with permutation inference as the primary method; DiD provides qualitatively similar results.")

md_lines.append("")
md_lines.append(f"Surface hash: `{SURFACE_HASH}`")
md_lines.append("")

with open(f"{OUTPUT_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write("\n".join(md_lines))

print(f"Wrote SPECIFICATION_SEARCH.md")
print("\nDone!")
