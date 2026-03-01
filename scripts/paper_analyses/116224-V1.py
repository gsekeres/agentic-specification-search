"""
Specification Search Script for Kuruscu (2006)
"Training and Lifetime Income"
American Economic Review, 96(1), 293-313.

Paper ID: 116224-V1

Surface-driven execution:
  - G1: later_wage_growth ~ earlier_wage_growth, bivariate OLS, HC1 SE
  - Replicate wagegrowthcorrelationlog55yrahead3035.m from Matlab
  - Specification search over: window parameters (yearahead, nobs, initial_age),
    time variable (age vs experience), sample restrictions (outlier trim,
    min obs, schooling caps), functional form (log vs level wages),
    and adding schooling as a control

Outputs:
  - specification_results.csv
  - inference_results.csv
  - SPECIFICATION_SEARCH.md
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
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

PAPER_ID = "116224-V1"
DATA_DIR = "data/downloads/extracted/116224-V1"
OUTPUT_DIR = DATA_DIR
DATA_SUBDIR = f"{DATA_DIR}/data-programs"

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

print("Loading PSID data...")

# Each file: ~5394 rows (individuals) x 26 columns (time periods)
# Tab-separated. 0 = missing in age/wage/school; -1 = missing in exper.
psid_age = np.loadtxt(f"{DATA_SUBDIR}/20030612_3_DATA_psidage.txt")
psid_realwage = np.loadtxt(f"{DATA_SUBDIR}/20030612_3_DATA_psidrealwage.txt")
psid_exper = np.loadtxt(f"{DATA_SUBDIR}/20030612_3_DATA_psidexper.txt")
psid_school = np.loadtxt(f"{DATA_SUBDIR}/20030612_3_DATA_psidschool.txt")

# Take log of real wages (matching Matlab: psidrealwage=log(psidrealwage))
# Replace 0 or negative wages with NaN before logging
psid_realwage_raw = psid_realwage.copy()
psid_realwage[psid_realwage <= 0] = np.nan
psid_logwage = np.log(psid_realwage)

n_individuals, n_periods = psid_age.shape
print(f"Loaded data: {n_individuals} individuals, {n_periods} time periods")


# ============================================================
# Core function: compute individual wage growth slopes
# ============================================================

def compute_wage_growth(time_var_data, wage_data, start_val, nobs,
                        missing_indicator=0, use_log=True):
    """
    For each individual, compute the slope of wage on time_var over the window
    [start_val, start_val + nobs]. Returns growth slope or NaN if insufficient data.

    Parameters:
    - time_var_data: n_individuals x n_periods array (age or experience)
    - wage_data: n_individuals x n_periods array (log wages if use_log=True, levels otherwise)
    - start_val: starting value of the time variable for the window
    - nobs: window length (include periods with time_var in [start_val, start_val+nobs])
    - missing_indicator: value indicating missing data in time_var (0 for age, -1 for exper)
    - use_log: if True, use log wages; if False, use level wages
    """
    n_ind = time_var_data.shape[0]
    slopes = np.full(n_ind, np.nan)

    for k in range(n_ind):
        xx_list = []
        yy_list = []
        for j in range(n_periods):
            t_val = time_var_data[k, j]
            w_val = wage_data[k, j]

            # Check if time_var is valid (not missing)
            if missing_indicator == 0:
                time_valid = (t_val > 0)
            else:
                time_valid = (t_val != missing_indicator)

            # Check if wage is valid
            if use_log:
                wage_valid = np.isfinite(w_val) and w_val != 0
            else:
                wage_valid = (w_val > 0)

            if time_valid and wage_valid:
                if (t_val >= start_val) and (t_val <= start_val + nobs):
                    xx_list.append(t_val)
                    if use_log:
                        yy_list.append(w_val)
                    else:
                        yy_list.append(w_val)

        if len(xx_list) >= 2:  # Need at least 2 obs to compute slope
            xx = np.array(xx_list)
            yy = np.array(yy_list)
            X = np.column_stack([np.ones(len(xx)), xx])
            try:
                b = np.linalg.solve(X.T @ X, X.T @ yy)
                slopes[k] = b[1]  # slope coefficient
            except np.linalg.LinAlgError:
                pass

    return slopes


def compute_wage_growth_reqnobs(time_var_data, wage_data, start_val, nobs,
                                 reqnobs=2, missing_indicator=0, use_log=True):
    """Same as compute_wage_growth but with configurable minimum required observations."""
    n_ind = time_var_data.shape[0]
    slopes = np.full(n_ind, np.nan)

    for k in range(n_ind):
        xx_list = []
        yy_list = []
        for j in range(n_periods):
            t_val = time_var_data[k, j]
            w_val = wage_data[k, j]

            if missing_indicator == 0:
                time_valid = (t_val > 0)
            else:
                time_valid = (t_val != missing_indicator)

            if use_log:
                wage_valid = np.isfinite(w_val) and w_val != 0
            else:
                wage_valid = (w_val > 0)

            if time_valid and wage_valid:
                if (t_val >= start_val) and (t_val <= start_val + nobs):
                    xx_list.append(t_val)
                    yy_list.append(w_val)

        if len(xx_list) >= reqnobs:
            xx = np.array(xx_list)
            yy = np.array(yy_list)
            X = np.column_stack([np.ones(len(xx)), xx])
            try:
                b = np.linalg.solve(X.T @ X, X.T @ yy)
                slopes[k] = b[1]
            except np.linalg.LinAlgError:
                pass

    return slopes


def build_growth_regression_data(time_var_data, wage_data, school_data,
                                  initial_val, nobs, yearahead,
                                  maxpercentchange=10.0,
                                  missing_indicator=0, use_log=True,
                                  reqnobs=2, minschool=100):
    """
    Build a dataset of (earlier_growth, later_growth) pairs, mirroring the
    Matlab code's logic.

    Returns a DataFrame with columns:
      earlier_growth, later_growth, (and optionally avg_school)
    """
    start1 = initial_val
    start2 = start1 + nobs + yearahead

    # Apply schooling filter to wage data if needed
    if minschool < 100:
        # Create a masked version of wage data
        wage_masked = wage_data.copy()
        for k in range(n_individuals):
            for j in range(n_periods):
                s = school_data[k, j]
                if s > 0 and s >= minschool:
                    wage_masked[k, j] = np.nan if use_log else 0
        wage_use = wage_masked
    else:
        wage_use = wage_data

    growth1 = compute_wage_growth_reqnobs(
        time_var_data, wage_use, start1, nobs,
        reqnobs=reqnobs, missing_indicator=missing_indicator, use_log=use_log)
    growth2 = compute_wage_growth_reqnobs(
        time_var_data, wage_use, start2, nobs,
        reqnobs=reqnobs, missing_indicator=missing_indicator, use_log=use_log)

    # Both must be non-NaN
    valid = np.isfinite(growth1) & np.isfinite(growth2)

    # Apply outlier filter (maxpercentchange)
    valid &= (np.abs(growth1) < maxpercentchange) & (np.abs(growth2) < maxpercentchange)

    g1 = growth1[valid]
    g2 = growth2[valid]

    # Compute average schooling for each individual
    avg_school = np.full(n_individuals, np.nan)
    for k in range(n_individuals):
        s_vals = school_data[k, school_data[k, :] > 0]
        if len(s_vals) > 0:
            avg_school[k] = np.mean(s_vals)
    sch = avg_school[valid]

    df = pd.DataFrame({
        'earlier_growth': g1,
        'later_growth': g2,
        'avg_school': sch
    })
    return df


# ============================================================
# Accumulators
# ============================================================

results = []
inference_results = []
spec_run_counter = 0


# ============================================================
# Helper: run_spec (bivariate or multivariate OLS via statsmodels)
# ============================================================

def run_spec(spec_id, spec_tree_path, baseline_group_id,
             outcome_var, treatment_var, controls, data,
             vcov_type, sample_desc, controls_desc,
             axis_block_name=None, axis_block=None, notes=""):
    """Run a single OLS specification and record results."""
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        y = data[outcome_var].values.astype(float)
        rhs_vars = [treatment_var] + list(controls)
        X = sm.add_constant(data[rhs_vars].values.astype(float))
        rhs_names = ['const'] + rhs_vars

        model = sm.OLS(y, X)
        if vcov_type == 'HC1':
            result = model.fit(cov_type='HC1')
        elif vcov_type == 'HC3':
            result = model.fit(cov_type='HC3')
        elif vcov_type == 'HC0':
            result = model.fit(cov_type='HC0')
        else:
            result = model.fit(cov_type='HC1')

        # Map variable names
        param_map = dict(zip(rhs_names, range(len(rhs_names))))
        treat_idx = param_map[treatment_var]

        coef_val = float(result.params[treat_idx])
        se_val = float(result.bse[treat_idx])
        pval = float(result.pvalues[treat_idx])
        ci = result.conf_int()
        ci_lower = float(ci[treat_idx, 0])
        ci_upper = float(ci[treat_idx, 1])
        nobs = int(result.nobs)
        r2 = float(result.rsquared)

        all_coefs = {name: float(result.params[i]) for i, name in enumerate(rhs_names)}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": inference_canonical["spec_id"],
                       "method": "HC1", "cluster_vars": []},
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
            "fixed_effects": "none",
            "controls_desc": controls_desc,
            "cluster_var": "",
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
            "fixed_effects": "none",
            "controls_desc": controls_desc,
            "cluster_var": "",
            "run_success": 0,
            "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


# ============================================================
# Prepare wage data variants
# ============================================================

# Log wages (baseline - already computed)
wage_for_log = psid_logwage.copy()

# Level wages (for functional form variant)
wage_for_level = psid_realwage_raw.copy()

# ============================================================
# BASELINE: Age-based, initial_age=30, nobs=4, yearahead=5, reqnobs=2
# Matching paper's wagegrowthcorrelationlog55yrahead3035.m
# with initialage=29 -> agecount=1 -> startage1=30
# ============================================================

print("=" * 60)
print("BASELINE SPECIFICATION")
print("=" * 60)

# Paper baseline params
INITIAL_AGE = 30  # initialage=29, agecount=1 -> startage1=30
NOBS = 4          # 5-year window (nobs+1 years)
YEARAHEAD = 5     # gap between early and late periods
MAXPCTCHANGE = 10.0
REQNOBS = 2

df_base_age = build_growth_regression_data(
    psid_age, wage_for_log, psid_school,
    initial_val=INITIAL_AGE, nobs=NOBS, yearahead=YEARAHEAD,
    maxpercentchange=MAXPCTCHANGE, missing_indicator=0,
    use_log=True, reqnobs=REQNOBS)

print(f"Baseline age-based sample: {len(df_base_age)} individuals")

base_run_id, base_coef, base_se, base_pval, base_nobs = run_spec(
    "baseline", "designs/cross_sectional_ols.md#baseline", "G1",
    "later_growth", "earlier_growth", [], df_base_age,
    "HC1",
    f"Age-based, init_age={INITIAL_AGE}, nobs={NOBS}, yearahead={YEARAHEAD}, N={len(df_base_age)}",
    "bivariate (no controls)")

print(f"  Baseline (age): coef={base_coef:.4f}, se={base_se:.4f}, p={base_pval:.4f}, N={base_nobs}")


# ============================================================
# BASELINE EXPERIENCE: Experience-based with same parameters
# ============================================================

print("\nRunning experience-based baseline...")

INITIAL_EXPER = 11  # initialexper=10, agecount=1 -> startexper1=11

df_base_exper = build_growth_regression_data(
    psid_exper, wage_for_log, psid_school,
    initial_val=INITIAL_EXPER, nobs=NOBS, yearahead=YEARAHEAD,
    maxpercentchange=MAXPCTCHANGE, missing_indicator=-1,
    use_log=True, reqnobs=REQNOBS)

print(f"Experience-based sample: {len(df_base_exper)} individuals")

run_spec(
    "baseline_experience", "designs/cross_sectional_ols.md#baseline", "G1",
    "later_growth", "earlier_growth", [], df_base_exper,
    "HC1",
    f"Exper-based, init_exper={INITIAL_EXPER}, nobs={NOBS}, yearahead={YEARAHEAD}, N={len(df_base_exper)}",
    "bivariate (no controls)",
    axis_block_name="time_variable",
    axis_block={"spec_id": "baseline_experience", "time_var": "experience"})


# ============================================================
# RC: YEARAHEAD VARIATIONS (age-based)
# ============================================================

print("\n" + "=" * 60)
print("RC: YEARAHEAD VARIATIONS")
print("=" * 60)

for ya in [1, 2, 3, 5, 7, 10, 12, 15]:
    if ya == YEARAHEAD:
        continue  # skip baseline
    spec_id = f"rc/window/yearahead_{ya}"
    try:
        df_ya = build_growth_regression_data(
            psid_age, wage_for_log, psid_school,
            initial_val=INITIAL_AGE, nobs=NOBS, yearahead=ya,
            maxpercentchange=MAXPCTCHANGE, missing_indicator=0,
            use_log=True, reqnobs=REQNOBS)
        if len(df_ya) >= 10:
            rid, c, s, p, n = run_spec(
                spec_id, "modules/robustness/sample.md#window-parameters", "G1",
                "later_growth", "earlier_growth", [], df_ya,
                "HC1",
                f"Age-based, init_age={INITIAL_AGE}, nobs={NOBS}, yearahead={ya}, N={len(df_ya)}",
                "bivariate",
                axis_block_name="window",
                axis_block={"spec_id": spec_id, "yearahead": ya, "nobs": NOBS,
                            "initial_age": INITIAL_AGE})
            print(f"  yearahead={ya}: coef={c:.4f}, p={p:.4f}, N={n}")
        else:
            print(f"  yearahead={ya}: insufficient data ({len(df_ya)} obs)")
    except Exception as e:
        print(f"  yearahead={ya}: ERROR - {e}")


# ============================================================
# RC: YEARAHEAD VARIATIONS (experience-based)
# ============================================================

print("\n" + "=" * 60)
print("RC: YEARAHEAD VARIATIONS (experience)")
print("=" * 60)

for ya in [1, 2, 3, 5, 7, 10, 12, 15]:
    if ya == YEARAHEAD:
        continue
    spec_id = f"rc/window/yearahead_{ya}_exper"
    try:
        df_ya = build_growth_regression_data(
            psid_exper, wage_for_log, psid_school,
            initial_val=INITIAL_EXPER, nobs=NOBS, yearahead=ya,
            maxpercentchange=MAXPCTCHANGE, missing_indicator=-1,
            use_log=True, reqnobs=REQNOBS)
        if len(df_ya) >= 10:
            rid, c, s, p, n = run_spec(
                spec_id, "modules/robustness/sample.md#window-parameters", "G1",
                "later_growth", "earlier_growth", [], df_ya,
                "HC1",
                f"Exper-based, init_exper={INITIAL_EXPER}, nobs={NOBS}, yearahead={ya}, N={len(df_ya)}",
                "bivariate",
                axis_block_name="window",
                axis_block={"spec_id": spec_id, "yearahead": ya, "nobs": NOBS,
                            "initial_exper": INITIAL_EXPER, "time_var": "experience"})
            print(f"  yearahead={ya} (exper): coef={c:.4f}, p={p:.4f}, N={n}")
        else:
            print(f"  yearahead={ya} (exper): insufficient data ({len(df_ya)} obs)")
    except Exception as e:
        print(f"  yearahead={ya} (exper): ERROR - {e}")


# ============================================================
# RC: WINDOW SIZE (nobs) VARIATIONS
# ============================================================

print("\n" + "=" * 60)
print("RC: WINDOW SIZE VARIATIONS")
print("=" * 60)

for nb in [3, 5, 6]:
    if nb == NOBS:
        continue
    spec_id = f"rc/window/nobs_{nb}"
    try:
        df_nb = build_growth_regression_data(
            psid_age, wage_for_log, psid_school,
            initial_val=INITIAL_AGE, nobs=nb, yearahead=YEARAHEAD,
            maxpercentchange=MAXPCTCHANGE, missing_indicator=0,
            use_log=True, reqnobs=REQNOBS)
        if len(df_nb) >= 10:
            rid, c, s, p, n = run_spec(
                spec_id, "modules/robustness/sample.md#window-parameters", "G1",
                "later_growth", "earlier_growth", [], df_nb,
                "HC1",
                f"Age-based, init_age={INITIAL_AGE}, nobs={nb}, yearahead={YEARAHEAD}, N={len(df_nb)}",
                "bivariate",
                axis_block_name="window",
                axis_block={"spec_id": spec_id, "yearahead": YEARAHEAD, "nobs": nb,
                            "initial_age": INITIAL_AGE})
            print(f"  nobs={nb}: coef={c:.4f}, p={p:.4f}, N={n}")
        else:
            print(f"  nobs={nb}: insufficient data ({len(df_nb)} obs)")
    except Exception as e:
        print(f"  nobs={nb}: ERROR - {e}")

# nobs variations with experience
for nb in [3, 5, 6]:
    if nb == NOBS:
        continue
    spec_id = f"rc/window/nobs_{nb}_exper"
    try:
        df_nb = build_growth_regression_data(
            psid_exper, wage_for_log, psid_school,
            initial_val=INITIAL_EXPER, nobs=nb, yearahead=YEARAHEAD,
            maxpercentchange=MAXPCTCHANGE, missing_indicator=-1,
            use_log=True, reqnobs=REQNOBS)
        if len(df_nb) >= 10:
            rid, c, s, p, n = run_spec(
                spec_id, "modules/robustness/sample.md#window-parameters", "G1",
                "later_growth", "earlier_growth", [], df_nb,
                "HC1",
                f"Exper-based, init_exper={INITIAL_EXPER}, nobs={nb}, yearahead={YEARAHEAD}, N={len(df_nb)}",
                "bivariate",
                axis_block_name="window",
                axis_block={"spec_id": spec_id, "yearahead": YEARAHEAD, "nobs": nb,
                            "initial_exper": INITIAL_EXPER, "time_var": "experience"})
            print(f"  nobs={nb} (exper): coef={c:.4f}, p={p:.4f}, N={n}")
        else:
            print(f"  nobs={nb} (exper): insufficient data ({len(df_nb)} obs)")
    except Exception as e:
        print(f"  nobs={nb} (exper): ERROR - {e}")


# ============================================================
# RC: INITIAL AGE VARIATIONS
# ============================================================

print("\n" + "=" * 60)
print("RC: INITIAL AGE VARIATIONS")
print("=" * 60)

for init_a in [30, 31, 32, 33, 34, 35]:
    if init_a == INITIAL_AGE:
        continue
    spec_id = f"rc/age/init_{init_a}"
    try:
        df_ia = build_growth_regression_data(
            psid_age, wage_for_log, psid_school,
            initial_val=init_a, nobs=NOBS, yearahead=YEARAHEAD,
            maxpercentchange=MAXPCTCHANGE, missing_indicator=0,
            use_log=True, reqnobs=REQNOBS)
        if len(df_ia) >= 10:
            rid, c, s, p, n = run_spec(
                spec_id, "modules/robustness/sample.md#window-parameters", "G1",
                "later_growth", "earlier_growth", [], df_ia,
                "HC1",
                f"Age-based, init_age={init_a}, nobs={NOBS}, yearahead={YEARAHEAD}, N={len(df_ia)}",
                "bivariate",
                axis_block_name="initial_age",
                axis_block={"spec_id": spec_id, "initial_age": init_a,
                            "nobs": NOBS, "yearahead": YEARAHEAD})
            print(f"  init_age={init_a}: coef={c:.4f}, p={p:.4f}, N={n}")
        else:
            print(f"  init_age={init_a}: insufficient data ({len(df_ia)} obs)")
    except Exception as e:
        print(f"  init_age={init_a}: ERROR - {e}")

# Initial experience variations
for init_e in [9, 10, 12, 13, 14, 15]:
    if init_e == INITIAL_EXPER:
        continue
    spec_id = f"rc/age/init_exper_{init_e}"
    try:
        df_ie = build_growth_regression_data(
            psid_exper, wage_for_log, psid_school,
            initial_val=init_e, nobs=NOBS, yearahead=YEARAHEAD,
            maxpercentchange=MAXPCTCHANGE, missing_indicator=-1,
            use_log=True, reqnobs=REQNOBS)
        if len(df_ie) >= 10:
            rid, c, s, p, n = run_spec(
                spec_id, "modules/robustness/sample.md#window-parameters", "G1",
                "later_growth", "earlier_growth", [], df_ie,
                "HC1",
                f"Exper-based, init_exper={init_e}, nobs={NOBS}, yearahead={YEARAHEAD}, N={len(df_ie)}",
                "bivariate",
                axis_block_name="initial_age",
                axis_block={"spec_id": spec_id, "initial_exper": init_e,
                            "nobs": NOBS, "yearahead": YEARAHEAD, "time_var": "experience"})
            print(f"  init_exper={init_e}: coef={c:.4f}, p={p:.4f}, N={n}")
        else:
            print(f"  init_exper={init_e}: insufficient data ({len(df_ie)} obs)")
    except Exception as e:
        print(f"  init_exper={init_e}: ERROR - {e}")


# ============================================================
# RC: OUTLIER TRIM VARIATIONS
# ============================================================

print("\n" + "=" * 60)
print("RC: OUTLIER TRIM VARIATIONS")
print("=" * 60)

for mpc in [0.5, 1.0, 2.0, 5.0]:
    spec_id = f"rc/sample/outlier_trim_{mpc}"
    try:
        df_trim = build_growth_regression_data(
            psid_age, wage_for_log, psid_school,
            initial_val=INITIAL_AGE, nobs=NOBS, yearahead=YEARAHEAD,
            maxpercentchange=mpc, missing_indicator=0,
            use_log=True, reqnobs=REQNOBS)
        if len(df_trim) >= 10:
            rid, c, s, p, n = run_spec(
                spec_id, "modules/robustness/sample.md#outliers-and-influential-observations", "G1",
                "later_growth", "earlier_growth", [], df_trim,
                "HC1",
                f"Age-based, maxpctchange={mpc}, N={len(df_trim)}",
                "bivariate",
                axis_block_name="sample",
                axis_block={"spec_id": spec_id, "maxpercentchange": mpc,
                            "n_obs": len(df_trim)})
            print(f"  maxpctchange={mpc}: coef={c:.4f}, p={p:.4f}, N={n}")
        else:
            print(f"  maxpctchange={mpc}: insufficient data ({len(df_trim)} obs)")
    except Exception as e:
        print(f"  maxpctchange={mpc}: ERROR - {e}")


# ============================================================
# RC: MINIMUM REQUIRED OBSERVATIONS VARIATIONS
# ============================================================

print("\n" + "=" * 60)
print("RC: REQNOBS VARIATIONS")
print("=" * 60)

for rn in [3, 4]:
    spec_id = f"rc/sample/reqnobs_{rn}"
    try:
        df_rn = build_growth_regression_data(
            psid_age, wage_for_log, psid_school,
            initial_val=INITIAL_AGE, nobs=NOBS, yearahead=YEARAHEAD,
            maxpercentchange=MAXPCTCHANGE, missing_indicator=0,
            use_log=True, reqnobs=rn)
        if len(df_rn) >= 10:
            rid, c, s, p, n = run_spec(
                spec_id, "modules/robustness/sample.md#window-parameters", "G1",
                "later_growth", "earlier_growth", [], df_rn,
                "HC1",
                f"Age-based, reqnobs={rn}, N={len(df_rn)}",
                "bivariate",
                axis_block_name="sample",
                axis_block={"spec_id": spec_id, "reqnobs": rn,
                            "n_obs": len(df_rn)})
            print(f"  reqnobs={rn}: coef={c:.4f}, p={p:.4f}, N={n}")
        else:
            print(f"  reqnobs={rn}: insufficient data ({len(df_rn)} obs)")
    except Exception as e:
        print(f"  reqnobs={rn}: ERROR - {e}")


# ============================================================
# RC: SCHOOLING RESTRICTION
# ============================================================

print("\n" + "=" * 60)
print("RC: SCHOOLING RESTRICTION")
print("=" * 60)

for max_sch in [12, 16]:
    spec_id = f"rc/sample/school_lt{max_sch}"
    try:
        df_sch = build_growth_regression_data(
            psid_age, wage_for_log, psid_school,
            initial_val=INITIAL_AGE, nobs=NOBS, yearahead=YEARAHEAD,
            maxpercentchange=MAXPCTCHANGE, missing_indicator=0,
            use_log=True, reqnobs=REQNOBS, minschool=max_sch)
        if len(df_sch) >= 10:
            rid, c, s, p, n = run_spec(
                spec_id, "modules/robustness/sample.md#subgroup-analysis", "G1",
                "later_growth", "earlier_growth", [], df_sch,
                "HC1",
                f"Age-based, school<{max_sch}, N={len(df_sch)}",
                "bivariate",
                axis_block_name="sample",
                axis_block={"spec_id": spec_id, "minschool": max_sch,
                            "n_obs": len(df_sch)})
            print(f"  school<{max_sch}: coef={c:.4f}, p={p:.4f}, N={n}")
        else:
            print(f"  school<{max_sch}: insufficient data ({len(df_sch)} obs)")
    except Exception as e:
        print(f"  school<{max_sch}: ERROR - {e}")


# ============================================================
# RC: FUNCTIONAL FORM - Level wages
# ============================================================

print("\n" + "=" * 60)
print("RC: FUNCTIONAL FORM - Level wages")
print("=" * 60)

spec_id = "rc/form/level_wages"
try:
    df_level = build_growth_regression_data(
        psid_age, wage_for_level, psid_school,
        initial_val=INITIAL_AGE, nobs=NOBS, yearahead=YEARAHEAD,
        maxpercentchange=100.0,  # different scale for levels
        missing_indicator=0,
        use_log=False, reqnobs=REQNOBS)
    if len(df_level) >= 10:
        rid, c, s, p, n = run_spec(
            spec_id, "modules/robustness/functional_form.md#outcome-transformations", "G1",
            "later_growth", "earlier_growth", [], df_level,
            "HC1",
            f"Age-based, level wages, N={len(df_level)}",
            "bivariate",
            axis_block_name="functional_form",
            axis_block={"spec_id": spec_id, "wage_transform": "level"})
        print(f"  Level wages: coef={c:.4f}, p={p:.4f}, N={n}")
    else:
        print(f"  Level wages: insufficient data ({len(df_level)} obs)")
except Exception as e:
    print(f"  Level wages: ERROR - {e}")

# Level wages with experience
spec_id = "rc/form/level_wages_exper"
try:
    df_level_e = build_growth_regression_data(
        psid_exper, wage_for_level, psid_school,
        initial_val=INITIAL_EXPER, nobs=NOBS, yearahead=YEARAHEAD,
        maxpercentchange=100.0,
        missing_indicator=-1,
        use_log=False, reqnobs=REQNOBS)
    if len(df_level_e) >= 10:
        rid, c, s, p, n = run_spec(
            spec_id, "modules/robustness/functional_form.md#outcome-transformations", "G1",
            "later_growth", "earlier_growth", [], df_level_e,
            "HC1",
            f"Exper-based, level wages, N={len(df_level_e)}",
            "bivariate",
            axis_block_name="functional_form",
            axis_block={"spec_id": spec_id, "wage_transform": "level",
                        "time_var": "experience"})
        print(f"  Level wages (exper): coef={c:.4f}, p={p:.4f}, N={n}")
    else:
        print(f"  Level wages (exper): insufficient data ({len(df_level_e)} obs)")
except Exception as e:
    print(f"  Level wages (exper): ERROR - {e}")


# ============================================================
# RC: CONTROLS - Add schooling as control
# ============================================================

print("\n" + "=" * 60)
print("RC: ADD SCHOOLING CONTROL")
print("=" * 60)

spec_id = "rc/controls/add_schooling"
try:
    df_sch_ctrl = df_base_age.dropna(subset=['avg_school']).copy()
    if len(df_sch_ctrl) >= 10:
        rid, c, s, p, n = run_spec(
            spec_id, "modules/robustness/controls.md#standard-control-sets", "G1",
            "later_growth", "earlier_growth", ["avg_school"], df_sch_ctrl,
            "HC1",
            f"Age-based, with schooling control, N={len(df_sch_ctrl)}",
            "earlier_growth + avg_school",
            axis_block_name="controls",
            axis_block={"spec_id": spec_id, "added_control": "avg_school"})
        print(f"  With schooling: coef={c:.4f}, p={p:.4f}, N={n}")
    else:
        print(f"  With schooling: insufficient data ({len(df_sch_ctrl)} obs)")
except Exception as e:
    print(f"  With schooling: ERROR - {e}")

# Schooling control with experience
spec_id = "rc/controls/add_schooling_exper"
try:
    df_sch_ctrl_e = df_base_exper.dropna(subset=['avg_school']).copy()
    if len(df_sch_ctrl_e) >= 10:
        rid, c, s, p, n = run_spec(
            spec_id, "modules/robustness/controls.md#standard-control-sets", "G1",
            "later_growth", "earlier_growth", ["avg_school"], df_sch_ctrl_e,
            "HC1",
            f"Exper-based, with schooling control, N={len(df_sch_ctrl_e)}",
            "earlier_growth + avg_school",
            axis_block_name="controls",
            axis_block={"spec_id": spec_id, "added_control": "avg_school",
                        "time_var": "experience"})
        print(f"  With schooling (exper): coef={c:.4f}, p={p:.4f}, N={n}")
    else:
        print(f"  With schooling (exper): insufficient data ({len(df_sch_ctrl_e)} obs)")
except Exception as e:
    print(f"  With schooling (exper): ERROR - {e}")


# ============================================================
# RC: COMBINED GRID - Joint yearahead x initial_age (age-based)
# ============================================================

print("\n" + "=" * 60)
print("RC: COMBINED GRID (yearahead x initial_age)")
print("=" * 60)

grid_combos = [
    (31, 3), (31, 7), (31, 10),
    (32, 3), (32, 7), (32, 10),
    (33, 3), (33, 7),
    (34, 3), (34, 7),
    (35, 3),
]

for init_a, ya in grid_combos:
    spec_id = f"rc/grid/age{init_a}_ya{ya}"
    try:
        df_g = build_growth_regression_data(
            psid_age, wage_for_log, psid_school,
            initial_val=init_a, nobs=NOBS, yearahead=ya,
            maxpercentchange=MAXPCTCHANGE, missing_indicator=0,
            use_log=True, reqnobs=REQNOBS)
        if len(df_g) >= 10:
            rid, c, s, p, n = run_spec(
                spec_id, "modules/robustness/sample.md#window-parameters", "G1",
                "later_growth", "earlier_growth", [], df_g,
                "HC1",
                f"Age-based, init_age={init_a}, nobs={NOBS}, yearahead={ya}, N={len(df_g)}",
                "bivariate",
                axis_block_name="grid",
                axis_block={"spec_id": spec_id, "initial_age": init_a,
                            "yearahead": ya, "nobs": NOBS})
            print(f"  age={init_a}, ya={ya}: coef={c:.4f}, p={p:.4f}, N={n}")
        else:
            print(f"  age={init_a}, ya={ya}: insufficient data ({len(df_g)} obs)")
    except Exception as e:
        print(f"  age={init_a}, ya={ya}: ERROR - {e}")


# ============================================================
# INFERENCE VARIANTS (on baseline specification)
# ============================================================

print("\n" + "=" * 60)
print("INFERENCE VARIANTS")
print("=" * 60)

baseline_run_id = f"{PAPER_ID}_run_001"
infer_counter = 0


def run_inference_variant(base_run_id, spec_id, spec_tree_path, baseline_group_id,
                          data, focal_var, vcov_type, vcov_desc, controls=None):
    """Re-run baseline spec with different variance-covariance estimator."""
    global infer_counter
    infer_counter += 1
    infer_run_id = f"{PAPER_ID}_infer_{infer_counter:03d}"

    try:
        y = data['later_growth'].values.astype(float)
        rhs_vars = [focal_var] + (list(controls) if controls else [])
        X = sm.add_constant(data[rhs_vars].values.astype(float))
        rhs_names = ['const'] + rhs_vars

        model = sm.OLS(y, X)
        result = model.fit(cov_type=vcov_type)

        param_map = dict(zip(rhs_names, range(len(rhs_names))))
        treat_idx = param_map[focal_var]

        coef_val = float(result.params[treat_idx])
        se_val = float(result.bse[treat_idx])
        pval = float(result.pvalues[treat_idx])
        ci = result.conf_int()
        ci_lower = float(ci[treat_idx, 0])
        ci_upper = float(ci[treat_idx, 1])
        nobs = int(result.nobs)
        r2 = float(result.rsquared)

        all_coefs = {name: float(result.params[i]) for i, name in enumerate(rhs_names)}

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
        print(f"  {spec_id}: se={se_val:.4f}, p={pval:.4f}")

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
        print(f"  {spec_id}: FAILED - {err_msg}")


# HC3 robust SEs
run_inference_variant(
    baseline_run_id, "infer/se/hc3",
    "modules/inference/standard_errors.md#heteroskedasticity-robust", "G1",
    df_base_age, "earlier_growth", "HC3", "HC3 (jackknife-bias-corrected)")

# HC0 robust SEs
run_inference_variant(
    baseline_run_id, "infer/se/hc0",
    "modules/inference/standard_errors.md#heteroskedasticity-robust", "G1",
    df_base_age, "earlier_growth", "HC0", "HC0 (White)")

# Homoskedastic SEs (classical OLS)
run_inference_variant(
    baseline_run_id, "infer/se/ols_classical",
    "modules/inference/standard_errors.md#classical", "G1",
    df_base_age, "earlier_growth", "nonrobust", "Classical OLS (homoskedastic)")


# ============================================================
# WRITE OUTPUTS
# ============================================================

print(f"\n{'=' * 60}")
print("WRITING OUTPUTS")
print(f"{'=' * 60}")
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
        print(f"\nBaseline coef on earlier_growth: {base_row['coefficient'].values[0]:.6f}")
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
md_lines.append("# Specification Search Report: 116224-V1")
md_lines.append("")
md_lines.append("**Paper:** Kuruscu (2006), \"Training and Lifetime Income\", AER 96(1)")
md_lines.append("")
md_lines.append("## Baseline Specification")
md_lines.append("")
md_lines.append("- **Design:** Cross-sectional OLS (bivariate)")
md_lines.append("- **Outcome:** Later-career log-wage growth (slope of log real wage on age over 5-year window)")
md_lines.append("- **Treatment:** Earlier-career log-wage growth (slope of log real wage on age over 5-year window)")
md_lines.append("- **Controls:** None (bivariate regression)")
md_lines.append("- **Fixed effects:** None")
md_lines.append("- **SE type:** HC1 (robust)")
md_lines.append("- **Data:** PSID panel, individual-level wage growth computed from within-person regressions")
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
    "Baseline": successful[successful['spec_id'].str.startswith('baseline')],
    "Yearahead Variations": successful[successful['spec_id'].str.startswith('rc/window/yearahead')],
    "Window Size (nobs)": successful[successful['spec_id'].str.startswith('rc/window/nobs')],
    "Initial Age/Exper": successful[successful['spec_id'].str.startswith('rc/age/')],
    "Outlier Trim": successful[successful['spec_id'].str.startswith('rc/sample/outlier')],
    "Req. Observations": successful[successful['spec_id'].str.startswith('rc/sample/reqnobs')],
    "Schooling Restriction": successful[successful['spec_id'].str.startswith('rc/sample/school')],
    "Functional Form": successful[successful['spec_id'].str.startswith('rc/form/')],
    "Controls": successful[successful['spec_id'].str.startswith('rc/controls/')],
    "Grid Combinations": successful[successful['spec_id'].str.startswith('rc/grid/')],
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
