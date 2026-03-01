"""
Specification Search Script for Xiong & Yu (2011)
"The Chinese Warrants Bubble"
American Economic Review, 101(6), 2723-2753.

Paper ID: 112466-V1

Surface-driven execution:
  - G1: Price ~ Turnover + Volatility + Volume + Days_remaining (maturity effect, N=60)
  - G2: Warrant_price ~ Turnover + Volatility + Volume + days_to_exp (daily, N=468)
  - Cross-sectional OLS with Newey-West HAC standard errors
  - 50+ specifications across controls LOO, subsets, progression,
    sample trimming, functional form, and additional controls

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

PAPER_ID = "112466-V1"
DATA_DIR = "data/downloads/extracted/112466-V1"
OUTPUT_DIR = DATA_DIR
DATA_PATH = f"{DATA_DIR}/data/data.xls"

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

# ============================================================
# Data Loading and Preparation
# ============================================================

# Maturity effect data (G1)
df_mat_raw = pd.read_excel(DATA_PATH, 'Maturity effect data')
df_mat_raw.columns = [c.replace(' ', '_') for c in df_mat_raw.columns]
print(f"Loaded maturity effect data: {df_mat_raw.shape[0]} rows, {df_mat_raw.shape[1]} columns")

# Daily data (G2)
df_daily_raw = pd.read_excel(DATA_PATH, 'Daily data')
df_daily_raw.columns = [c.replace(' ', '_') for c in df_daily_raw.columns]
df_daily_raw['date'] = pd.to_datetime(df_daily_raw[['Year', 'Month', 'Day']])
last_date = df_daily_raw['date'].max()
df_daily_raw['days_to_exp'] = (last_date - df_daily_raw['date']).dt.days

# Derived variables for daily data
df_daily_raw['log_price'] = np.log(df_daily_raw['Warrant_price'].clip(lower=0.001))
df_daily_raw['log_turnover'] = np.log(df_daily_raw['Turnover'].clip(lower=0.001))
df_daily_raw['log_volume'] = np.log(df_daily_raw['Volume'].clip(lower=0.001))
df_daily_raw['bubble'] = df_daily_raw['Warrant_price'] - df_daily_raw['Fundamental_upperbound']
df_daily_raw['moneyness'] = df_daily_raw['Stock_price'] / df_daily_raw['Strike_price']

# Derived variables for maturity data
df_mat_raw['log_Price'] = np.log(df_mat_raw['Price'].clip(lower=0.001))
df_mat_raw['Price_sq'] = df_mat_raw['Price'] ** 2

print(f"Loaded daily data: {df_daily_raw.shape[0]} rows, {df_daily_raw.shape[1]} columns")
print(f"  Date range: {df_daily_raw['date'].min().date()} to {df_daily_raw['date'].max().date()}")

# Working copies
df_mat = df_mat_raw.copy()
df_daily = df_daily_raw.copy()

# ============================================================
# Accumulators
# ============================================================

results = []
inference_results = []
spec_run_counter = 0


# ============================================================
# Helper: run_spec (OLS with HAC via statsmodels)
# ============================================================

def run_spec(spec_id, spec_tree_path, baseline_group_id,
             outcome_var, treatment_var, controls, data,
             nw_lags=5, sample_desc="", controls_desc="",
             axis_block_name=None, axis_block=None, notes="",
             design_audit=None):
    """Run a single OLS specification with Newey-West HAC SEs."""
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    if design_audit is None:
        bg = surface_obj["baseline_groups"][0] if baseline_group_id == "G1" else surface_obj["baseline_groups"][1]
        design_audit = bg["design_audit"]

    try:
        y = data[outcome_var].astype(float)
        rhs_vars = [treatment_var] + list(controls)
        X = sm.add_constant(data[rhs_vars].astype(float))

        # Drop missing
        mask = y.notna() & X.notna().all(axis=1)
        y = y[mask]
        X = X[mask]

        m = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': nw_lags})

        coef_val = float(m.params[treatment_var])
        se_val = float(m.bse[treatment_var])
        pval = float(m.pvalues[treatment_var])
        ci = m.conf_int()
        ci_lower = float(ci.loc[treatment_var, 0])
        ci_upper = float(ci.loc[treatment_var, 1])
        nobs = int(m.nobs)
        r2 = float(m.rsquared)

        all_coefs = {k: float(v) for k, v in m.params.items()}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": f"infer/se/hac/nw{nw_lags}",
                       "method": "HAC", "maxlags": nw_lags},
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
            "cluster_var": f"HAC_NW_{nw_lags}",
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
            "cluster_var": f"HAC_NW_{nw_lags}",
            "run_success": 0,
            "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


# ============================================================
# G1: MATURITY EFFECT REGRESSIONS (cross-sectional, N=60)
# ============================================================

print("\n========================================")
print("G1: Maturity Effect Regressions")
print("========================================")

G1_CONTROLS = ["Volatility", "Volume", "Days_remaining"]

# --- BASELINE ---
print("Running G1 baseline specification...")
g1_base_id, g1_coef, g1_se, g1_pval, g1_n = run_spec(
    "baseline", "designs/cross_sectional_ols.md#baseline", "G1",
    "Price", "Turnover", G1_CONTROLS, df_mat,
    nw_lags=5,
    sample_desc=f"Maturity effect data, N={len(df_mat)}",
    controls_desc="Volatility + Volume + Days_remaining (full)")

print(f"  G1 Baseline: coef={g1_coef:.6f}, se={g1_se:.6f}, p={g1_pval:.6f}, N={g1_n}")


# --- CONTROLS LOO ---
print("Running G1 controls LOO...")

G1_LOO_MAP = {
    "rc/controls/loo/drop_Volatility": ["Volatility"],
    "rc/controls/loo/drop_Volume": ["Volume"],
    "rc/controls/loo/drop_Days_remaining": ["Days_remaining"],
}

for spec_id, drop_vars in G1_LOO_MAP.items():
    ctrl = [c for c in G1_CONTROLS if c not in drop_vars]
    run_spec(
        spec_id, "modules/robustness/controls.md#leave-one-out-controls-loo", "G1",
        "Price", "Turnover", ctrl, df_mat,
        nw_lags=5,
        sample_desc="Maturity effect data",
        controls_desc=f"baseline minus {', '.join(drop_vars)}",
        axis_block_name="controls",
        axis_block={"spec_id": spec_id, "family": "loo",
                    "dropped": drop_vars, "n_controls": len(ctrl)})


# --- CONTROL SETS ---
print("Running G1 control sets...")

# No controls (bivariate)
run_spec(
    "rc/controls/sets/none",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "Price", "Turnover", [], df_mat,
    nw_lags=5,
    sample_desc="Maturity effect data",
    controls_desc="none (bivariate)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/none", "family": "sets",
                "n_controls": 0, "set_name": "none"})

# Turnover + Days_remaining only
run_spec(
    "rc/controls/sets/turnover_days",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "Price", "Turnover", ["Days_remaining"], df_mat,
    nw_lags=5,
    sample_desc="Maturity effect data",
    controls_desc="Days_remaining only",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/turnover_days", "family": "sets",
                "n_controls": 1, "set_name": "turnover_days"})

# Turnover + Volatility only
run_spec(
    "rc/controls/sets/turnover_volatility",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "Price", "Turnover", ["Volatility"], df_mat,
    nw_lags=5,
    sample_desc="Maturity effect data",
    controls_desc="Volatility only",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/turnover_volatility", "family": "sets",
                "n_controls": 1, "set_name": "turnover_volatility"})


# --- CONTROL PROGRESSION ---
print("Running G1 control progression...")

run_spec(
    "rc/controls/progression/bivariate",
    "modules/robustness/controls.md#control-progression-build-up", "G1",
    "Price", "Turnover", [], df_mat,
    nw_lags=5,
    sample_desc="Maturity effect data",
    controls_desc="bivariate (Turnover only)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/bivariate", "family": "progression",
                "n_controls": 0, "set_name": "bivariate"})

run_spec(
    "rc/controls/progression/plus_days",
    "modules/robustness/controls.md#control-progression-build-up", "G1",
    "Price", "Turnover", ["Days_remaining"], df_mat,
    nw_lags=5,
    sample_desc="Maturity effect data",
    controls_desc="Turnover + Days_remaining",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/plus_days", "family": "progression",
                "n_controls": 1, "set_name": "plus_days"})

run_spec(
    "rc/controls/progression/plus_volatility",
    "modules/robustness/controls.md#control-progression-build-up", "G1",
    "Price", "Turnover", ["Days_remaining", "Volatility"], df_mat,
    nw_lags=5,
    sample_desc="Maturity effect data",
    controls_desc="Turnover + Days_remaining + Volatility",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/plus_volatility", "family": "progression",
                "n_controls": 2, "set_name": "plus_volatility"})

run_spec(
    "rc/controls/progression/full",
    "modules/robustness/controls.md#control-progression-build-up", "G1",
    "Price", "Turnover", G1_CONTROLS, df_mat,
    nw_lags=5,
    sample_desc="Maturity effect data",
    controls_desc="full (same as baseline)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/full", "family": "progression",
                "n_controls": len(G1_CONTROLS), "set_name": "full"})


# --- RANDOM CONTROL SUBSETS (G1) ---
print("Running G1 random control subsets...")

rng_g1 = np.random.RandomState(112466)
g1_pool = G1_CONTROLS.copy()

for draw_i in range(1, 11):
    k = rng_g1.randint(1, len(g1_pool) + 1)
    chosen = list(rng_g1.choice(g1_pool, size=k, replace=False))
    excluded = [v for v in g1_pool if v not in chosen]
    spec_id = f"rc/controls/subset/random_{draw_i:03d}"
    run_spec(
        spec_id, "modules/robustness/controls.md#subset-generation-specids", "G1",
        "Price", "Turnover", chosen, df_mat,
        nw_lags=5,
        sample_desc="Maturity effect data",
        controls_desc=f"random subset draw {draw_i} ({len(chosen)} controls)",
        axis_block_name="controls",
        axis_block={"spec_id": spec_id, "family": "subset", "method": "random",
                    "seed": 112466, "draw_index": draw_i,
                    "included": chosen, "excluded": excluded,
                    "n_controls": len(chosen)})


# --- SAMPLE TRIMMING (G1) ---
print("Running G1 sample trimming variants...")

# Trim outcome at 1st/99th percentile
q01 = df_mat['Price'].quantile(0.01)
q99 = df_mat['Price'].quantile(0.99)
df_trim1 = df_mat[(df_mat['Price'] >= q01) & (df_mat['Price'] <= q99)].copy()

run_spec(
    "rc/sample/outliers/trim_y_1_99",
    "modules/robustness/sample.md#outliers-and-influential-observations", "G1",
    "Price", "Turnover", G1_CONTROLS, df_trim1,
    nw_lags=5,
    sample_desc=f"trim Price [1%,99%], N={len(df_trim1)}",
    controls_desc="full controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_1_99", "axis": "outliers",
                "rule": "trim", "params": {"var": "Price", "lower_q": 0.01, "upper_q": 0.99},
                "n_obs_before": len(df_mat), "n_obs_after": len(df_trim1)})

# Trim at 5th/95th
q05 = df_mat['Price'].quantile(0.05)
q95 = df_mat['Price'].quantile(0.95)
df_trim5 = df_mat[(df_mat['Price'] >= q05) & (df_mat['Price'] <= q95)].copy()

run_spec(
    "rc/sample/outliers/trim_y_5_95",
    "modules/robustness/sample.md#outliers-and-influential-observations", "G1",
    "Price", "Turnover", G1_CONTROLS, df_trim5,
    nw_lags=5,
    sample_desc=f"trim Price [5%,95%], N={len(df_trim5)}",
    controls_desc="full controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_5_95", "axis": "outliers",
                "rule": "trim", "params": {"var": "Price", "lower_q": 0.05, "upper_q": 0.95},
                "n_obs_before": len(df_mat), "n_obs_after": len(df_trim5)})

# Trim at 10th/90th
q10 = df_mat['Price'].quantile(0.10)
q90 = df_mat['Price'].quantile(0.90)
df_trim10 = df_mat[(df_mat['Price'] >= q10) & (df_mat['Price'] <= q90)].copy()

run_spec(
    "rc/sample/outliers/trim_y_10_90",
    "modules/robustness/sample.md#outliers-and-influential-observations", "G1",
    "Price", "Turnover", G1_CONTROLS, df_trim10,
    nw_lags=5,
    sample_desc=f"trim Price [10%,90%], N={len(df_trim10)}",
    controls_desc="full controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_10_90", "axis": "outliers",
                "rule": "trim", "params": {"var": "Price", "lower_q": 0.10, "upper_q": 0.90},
                "n_obs_before": len(df_mat), "n_obs_after": len(df_trim10)})

# Drop last 5 days (near expiration, extreme behavior)
df_no_last5 = df_mat[df_mat['Days_remaining'] >= 5].copy()
run_spec(
    "rc/sample/drop_last_5_days",
    "modules/robustness/sample.md#sample-restrictions", "G1",
    "Price", "Turnover", G1_CONTROLS, df_no_last5,
    nw_lags=5,
    sample_desc=f"drop last 5 days to maturity, N={len(df_no_last5)}",
    controls_desc="full controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/drop_last_5_days", "axis": "sample_restriction",
                "restriction": "Days_remaining >= 5",
                "n_obs_before": len(df_mat), "n_obs_after": len(df_no_last5)})

# Drop first 10 days (early period)
df_no_first10 = df_mat[df_mat['Days_remaining'] <= 49].copy()
run_spec(
    "rc/sample/drop_first_10_days",
    "modules/robustness/sample.md#sample-restrictions", "G1",
    "Price", "Turnover", G1_CONTROLS, df_no_first10,
    nw_lags=5,
    sample_desc=f"drop first 10 days (high maturity), N={len(df_no_first10)}",
    controls_desc="full controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/drop_first_10_days", "axis": "sample_restriction",
                "restriction": "Days_remaining <= 49",
                "n_obs_before": len(df_mat), "n_obs_after": len(df_no_first10)})


# --- FUNCTIONAL FORM (G1) ---
print("Running G1 functional form variants...")

# Log price outcome
run_spec(
    "rc/form/outcome/log_price",
    "modules/robustness/functional_form.md#outcome-transformations", "G1",
    "log_Price", "Turnover", G1_CONTROLS, df_mat,
    nw_lags=5,
    sample_desc="Maturity effect data",
    controls_desc="full controls, log(Price) outcome",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/log_price", "transformation": "log",
                "outcome_original": "Price", "outcome_transformed": "log_Price"})

# Squared price outcome (sensitivity check)
run_spec(
    "rc/form/outcome/price_sq",
    "modules/robustness/functional_form.md#outcome-transformations", "G1",
    "Price_sq", "Turnover", G1_CONTROLS, df_mat,
    nw_lags=5,
    sample_desc="Maturity effect data",
    controls_desc="full controls, Price^2 outcome",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/price_sq", "transformation": "squared",
                "outcome_original": "Price", "outcome_transformed": "Price_sq"})


# ============================================================
# G2: DAILY TIME-SERIES REGRESSIONS (N=468)
# ============================================================

print("\n========================================")
print("G2: Daily Time-Series Regressions")
print("========================================")

G2_CONTROLS = ["Volatility", "Volume", "days_to_exp"]
G2_EXTENDED = ["Volatility", "Volume", "days_to_exp", "moneyness"]

# --- BASELINE ---
print("Running G2 baseline specification...")
g2_base_id, g2_coef, g2_se, g2_pval, g2_n = run_spec(
    "baseline", "designs/cross_sectional_ols.md#baseline", "G2",
    "Warrant_price", "Turnover", G2_CONTROLS, df_daily,
    nw_lags=5,
    sample_desc=f"Daily data, N={len(df_daily)}",
    controls_desc="Volatility + Volume + days_to_exp (full)")

print(f"  G2 Baseline: coef={g2_coef:.6f}, se={g2_se:.6f}, p={g2_pval:.6f}, N={g2_n}")


# --- CONTROLS LOO (G2) ---
print("Running G2 controls LOO...")

G2_LOO_MAP = {
    "rc/controls/loo/drop_Volatility": ["Volatility"],
    "rc/controls/loo/drop_Volume": ["Volume"],
    "rc/controls/loo/drop_days_to_exp": ["days_to_exp"],
}

for spec_id, drop_vars in G2_LOO_MAP.items():
    ctrl = [c for c in G2_CONTROLS if c not in drop_vars]
    run_spec(
        spec_id, "modules/robustness/controls.md#leave-one-out-controls-loo", "G2",
        "Warrant_price", "Turnover", ctrl, df_daily,
        nw_lags=5,
        sample_desc="Daily data",
        controls_desc=f"baseline minus {', '.join(drop_vars)}",
        axis_block_name="controls",
        axis_block={"spec_id": spec_id, "family": "loo",
                    "dropped": drop_vars, "n_controls": len(ctrl)})


# --- CONTROL SETS (G2) ---
print("Running G2 control sets...")

# Bivariate
run_spec(
    "rc/controls/sets/none",
    "modules/robustness/controls.md#standard-control-sets", "G2",
    "Warrant_price", "Turnover", [], df_daily,
    nw_lags=5,
    sample_desc="Daily data",
    controls_desc="none (bivariate)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/none", "family": "sets",
                "n_controls": 0, "set_name": "none"})

# Turnover + days_to_exp
run_spec(
    "rc/controls/sets/turnover_days",
    "modules/robustness/controls.md#standard-control-sets", "G2",
    "Warrant_price", "Turnover", ["days_to_exp"], df_daily,
    nw_lags=5,
    sample_desc="Daily data",
    controls_desc="days_to_exp only",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/turnover_days", "family": "sets",
                "n_controls": 1, "set_name": "turnover_days"})

# Turnover + Volatility
run_spec(
    "rc/controls/sets/turnover_volatility",
    "modules/robustness/controls.md#standard-control-sets", "G2",
    "Warrant_price", "Turnover", ["Volatility"], df_daily,
    nw_lags=5,
    sample_desc="Daily data",
    controls_desc="Volatility only",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/turnover_volatility", "family": "sets",
                "n_controls": 1, "set_name": "turnover_volatility"})

# Full + moneyness
run_spec(
    "rc/controls/sets/all_plus_moneyness",
    "modules/robustness/controls.md#standard-control-sets", "G2",
    "Warrant_price", "Turnover", G2_EXTENDED, df_daily,
    nw_lags=5,
    sample_desc="Daily data",
    controls_desc="full + moneyness",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/all_plus_moneyness", "family": "sets",
                "n_controls": len(G2_EXTENDED), "set_name": "all_plus_moneyness"})


# --- CONTROL PROGRESSION (G2) ---
print("Running G2 control progression...")

run_spec(
    "rc/controls/progression/bivariate",
    "modules/robustness/controls.md#control-progression-build-up", "G2",
    "Warrant_price", "Turnover", [], df_daily,
    nw_lags=5,
    sample_desc="Daily data",
    controls_desc="bivariate (Turnover only)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/bivariate", "family": "progression",
                "n_controls": 0, "set_name": "bivariate"})

run_spec(
    "rc/controls/progression/plus_days",
    "modules/robustness/controls.md#control-progression-build-up", "G2",
    "Warrant_price", "Turnover", ["days_to_exp"], df_daily,
    nw_lags=5,
    sample_desc="Daily data",
    controls_desc="+ days_to_exp",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/plus_days", "family": "progression",
                "n_controls": 1, "set_name": "plus_days"})

run_spec(
    "rc/controls/progression/plus_volatility",
    "modules/robustness/controls.md#control-progression-build-up", "G2",
    "Warrant_price", "Turnover", ["days_to_exp", "Volatility"], df_daily,
    nw_lags=5,
    sample_desc="Daily data",
    controls_desc="+ days_to_exp + Volatility",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/plus_volatility", "family": "progression",
                "n_controls": 2, "set_name": "plus_volatility"})

run_spec(
    "rc/controls/progression/plus_volume",
    "modules/robustness/controls.md#control-progression-build-up", "G2",
    "Warrant_price", "Turnover", G2_CONTROLS, df_daily,
    nw_lags=5,
    sample_desc="Daily data",
    controls_desc="+ days_to_exp + Volatility + Volume (full)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/plus_volume", "family": "progression",
                "n_controls": len(G2_CONTROLS), "set_name": "plus_volume"})

run_spec(
    "rc/controls/progression/plus_moneyness",
    "modules/robustness/controls.md#control-progression-build-up", "G2",
    "Warrant_price", "Turnover", G2_EXTENDED, df_daily,
    nw_lags=5,
    sample_desc="Daily data",
    controls_desc="+ days_to_exp + Volatility + Volume + moneyness",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/plus_moneyness", "family": "progression",
                "n_controls": len(G2_EXTENDED), "set_name": "plus_moneyness"})


# --- RANDOM CONTROL SUBSETS (G2) ---
print("Running G2 random control subsets...")

rng_g2 = np.random.RandomState(112466 + 1)
g2_pool = G2_EXTENDED.copy()

for draw_i in range(1, 6):
    k = rng_g2.randint(1, len(g2_pool) + 1)
    chosen = list(rng_g2.choice(g2_pool, size=k, replace=False))
    excluded = [v for v in g2_pool if v not in chosen]
    spec_id = f"rc/controls/subset/random_{draw_i:03d}"
    run_spec(
        spec_id, "modules/robustness/controls.md#subset-generation-specids", "G2",
        "Warrant_price", "Turnover", chosen, df_daily,
        nw_lags=5,
        sample_desc="Daily data",
        controls_desc=f"random subset draw {draw_i} ({len(chosen)} controls)",
        axis_block_name="controls",
        axis_block={"spec_id": spec_id, "family": "subset", "method": "random",
                    "seed": 112467, "draw_index": draw_i,
                    "included": chosen, "excluded": excluded,
                    "n_controls": len(chosen)})


# --- SAMPLE TRIMMING (G2) ---
print("Running G2 sample trimming variants...")

# Trim outcome at 1st/99th
q01_d = df_daily['Warrant_price'].quantile(0.01)
q99_d = df_daily['Warrant_price'].quantile(0.99)
df_dtrim1 = df_daily[(df_daily['Warrant_price'] >= q01_d) & (df_daily['Warrant_price'] <= q99_d)].copy()

run_spec(
    "rc/sample/outliers/trim_y_1_99",
    "modules/robustness/sample.md#outliers-and-influential-observations", "G2",
    "Warrant_price", "Turnover", G2_CONTROLS, df_dtrim1,
    nw_lags=5,
    sample_desc=f"trim Warrant_price [1%,99%], N={len(df_dtrim1)}",
    controls_desc="full controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_1_99", "axis": "outliers",
                "rule": "trim", "params": {"var": "Warrant_price", "lower_q": 0.01, "upper_q": 0.99},
                "n_obs_before": len(df_daily), "n_obs_after": len(df_dtrim1)})

# Trim at 5th/95th
q05_d = df_daily['Warrant_price'].quantile(0.05)
q95_d = df_daily['Warrant_price'].quantile(0.95)
df_dtrim5 = df_daily[(df_daily['Warrant_price'] >= q05_d) & (df_daily['Warrant_price'] <= q95_d)].copy()

run_spec(
    "rc/sample/outliers/trim_y_5_95",
    "modules/robustness/sample.md#outliers-and-influential-observations", "G2",
    "Warrant_price", "Turnover", G2_CONTROLS, df_dtrim5,
    nw_lags=5,
    sample_desc=f"trim Warrant_price [5%,95%], N={len(df_dtrim5)}",
    controls_desc="full controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_5_95", "axis": "outliers",
                "rule": "trim", "params": {"var": "Warrant_price", "lower_q": 0.05, "upper_q": 0.95},
                "n_obs_before": len(df_daily), "n_obs_after": len(df_dtrim5)})

# Drop last month (near expiration)
last_month_start = df_daily['date'].max() - pd.Timedelta(days=30)
df_no_last_month = df_daily[df_daily['date'] < last_month_start].copy()

run_spec(
    "rc/sample/drop_last_month",
    "modules/robustness/sample.md#sample-restrictions", "G2",
    "Warrant_price", "Turnover", G2_CONTROLS, df_no_last_month,
    nw_lags=5,
    sample_desc=f"drop last month, N={len(df_no_last_month)}",
    controls_desc="full controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/drop_last_month", "axis": "sample_restriction",
                "restriction": "drop last 30 calendar days",
                "n_obs_before": len(df_daily), "n_obs_after": len(df_no_last_month)})

# Drop first month
first_month_end = df_daily['date'].min() + pd.Timedelta(days=30)
df_no_first_month = df_daily[df_daily['date'] > first_month_end].copy()

run_spec(
    "rc/sample/drop_first_month",
    "modules/robustness/sample.md#sample-restrictions", "G2",
    "Warrant_price", "Turnover", G2_CONTROLS, df_no_first_month,
    nw_lags=5,
    sample_desc=f"drop first month, N={len(df_no_first_month)}",
    controls_desc="full controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/drop_first_month", "axis": "sample_restriction",
                "restriction": "drop first 30 calendar days",
                "n_obs_before": len(df_daily), "n_obs_after": len(df_no_first_month)})

# Zero-fundamental period only (where Fundamental_upperbound < 0.1)
df_zero_fund = df_daily[df_daily['Fundamental_upperbound'] < 0.1].copy()
if len(df_zero_fund) >= 10:
    run_spec(
        "rc/sample/zero_fund_only",
        "modules/robustness/sample.md#sample-restrictions", "G2",
        "Warrant_price", "Turnover", G2_CONTROLS, df_zero_fund,
        nw_lags=5,
        sample_desc=f"zero-fundamental period only, N={len(df_zero_fund)}",
        controls_desc="full controls",
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/zero_fund_only", "axis": "sample_restriction",
                    "restriction": "Fundamental_upperbound < 0.1",
                    "n_obs_before": len(df_daily), "n_obs_after": len(df_zero_fund)})
else:
    print(f"  Skipping zero_fund_only: only {len(df_zero_fund)} obs")


# --- FUNCTIONAL FORM (G2) ---
print("Running G2 functional form variants...")

# Log price outcome
run_spec(
    "rc/form/outcome/log_price",
    "modules/robustness/functional_form.md#outcome-transformations", "G2",
    "log_price", "Turnover", G2_CONTROLS, df_daily,
    nw_lags=5,
    sample_desc="Daily data",
    controls_desc="full controls, log(Warrant_price) outcome",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/log_price", "transformation": "log",
                "outcome_original": "Warrant_price", "outcome_transformed": "log_price"})

# Bubble component as outcome (Warrant_price - Fundamental_upperbound)
run_spec(
    "rc/form/outcome/bubble_component",
    "modules/robustness/functional_form.md#outcome-transformations", "G2",
    "bubble", "Turnover", G2_CONTROLS, df_daily,
    nw_lags=5,
    sample_desc="Daily data",
    controls_desc="full controls, bubble=Warrant_price-Fundamental outcome",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/bubble_component", "transformation": "difference",
                "outcome_original": "Warrant_price", "outcome_transformed": "bubble",
                "notes": "bubble = Warrant_price - Fundamental_upperbound"})


# ============================================================
# ADDITIONAL SPECIFICATIONS: Alternative NW lag lengths
# ============================================================

print("Running alternative Newey-West lag specifications...")

# G1 with NW(10)
run_spec(
    "baseline__nw10", "designs/cross_sectional_ols.md#baseline", "G1",
    "Price", "Turnover", G1_CONTROLS, df_mat,
    nw_lags=10,
    sample_desc="Maturity effect data",
    controls_desc="full controls, NW(10)",
    axis_block_name="estimation",
    axis_block={"spec_id": "baseline__nw10", "nw_lags": 10})

# G1 with HC1 (no autocorrelation correction)
run_spec(
    "baseline__hc1", "designs/cross_sectional_ols.md#baseline", "G1",
    "Price", "Turnover", G1_CONTROLS, df_mat,
    nw_lags=0,
    sample_desc="Maturity effect data",
    controls_desc="full controls, HC1 (no HAC)")

# G2 with NW(10)
run_spec(
    "baseline__nw10", "designs/cross_sectional_ols.md#baseline", "G2",
    "Warrant_price", "Turnover", G2_CONTROLS, df_daily,
    nw_lags=10,
    sample_desc="Daily data",
    controls_desc="full controls, NW(10)",
    axis_block_name="estimation",
    axis_block={"spec_id": "baseline__nw10", "nw_lags": 10})

# G2 with NW(20)
run_spec(
    "baseline__nw20", "designs/cross_sectional_ols.md#baseline", "G2",
    "Warrant_price", "Turnover", G2_CONTROLS, df_daily,
    nw_lags=20,
    sample_desc="Daily data",
    controls_desc="full controls, NW(20)",
    axis_block_name="estimation",
    axis_block={"spec_id": "baseline__nw20", "nw_lags": 20})


# ============================================================
# ADDITIONAL SPECIFICATIONS: Volatility as treatment (G1 & G2)
# ============================================================

print("Running Volatility-as-treatment variants...")

# G1: Price ~ Volatility + Turnover + Volume + Days_remaining
run_spec(
    "alt_treatment/volatility", "designs/cross_sectional_ols.md#baseline", "G1",
    "Price", "Volatility", ["Turnover", "Volume", "Days_remaining"], df_mat,
    nw_lags=5,
    sample_desc="Maturity effect data",
    controls_desc="Turnover + Volume + Days_remaining (Volatility as treatment)")

# G2: Warrant_price ~ Volatility + Turnover + Volume + days_to_exp
run_spec(
    "alt_treatment/volatility", "designs/cross_sectional_ols.md#baseline", "G2",
    "Warrant_price", "Volatility", ["Turnover", "Volume", "days_to_exp"], df_daily,
    nw_lags=5,
    sample_desc="Daily data",
    controls_desc="Turnover + Volume + days_to_exp (Volatility as treatment)")


# ============================================================
# ADDITIONAL SPECIFICATIONS: Volume as treatment
# ============================================================

print("Running Volume-as-treatment variants...")

# G1: Price ~ Volume + Turnover + Volatility + Days_remaining
run_spec(
    "alt_treatment/volume", "designs/cross_sectional_ols.md#baseline", "G1",
    "Price", "Volume", ["Turnover", "Volatility", "Days_remaining"], df_mat,
    nw_lags=5,
    sample_desc="Maturity effect data",
    controls_desc="Turnover + Volatility + Days_remaining (Volume as treatment)")

# G2: Warrant_price ~ Volume + Turnover + Volatility + days_to_exp
run_spec(
    "alt_treatment/volume", "designs/cross_sectional_ols.md#baseline", "G2",
    "Warrant_price", "Volume", ["Turnover", "Volatility", "days_to_exp"], df_daily,
    nw_lags=5,
    sample_desc="Daily data",
    controls_desc="Turnover + Volatility + days_to_exp (Volume as treatment)")


# ============================================================
# INFERENCE VARIANTS
# ============================================================

print("Running inference variants...")

infer_counter = 0
g1_baseline_run_id = f"{PAPER_ID}_run_001"
g2_baseline_run_id = f"{PAPER_ID}_run_{spec_run_counter - len(results) + 1 + len([r for r in results if r['baseline_group_id'] == 'G1']):03d}"
# Find actual G2 baseline run_id
for r in results:
    if r['spec_id'] == 'baseline' and r['baseline_group_id'] == 'G2':
        g2_baseline_run_id = r['spec_run_id']
        break


def run_inference_variant(base_run_id, spec_id, spec_tree_path, baseline_group_id,
                          outcome_var, treatment_var, controls, data, nw_lags, vcov_desc):
    """Re-run baseline spec with different variance-covariance estimator."""
    global infer_counter
    infer_counter += 1
    infer_run_id = f"{PAPER_ID}_infer_{infer_counter:03d}"

    if baseline_group_id == "G1":
        design_audit = surface_obj["baseline_groups"][0]["design_audit"]
    else:
        design_audit = surface_obj["baseline_groups"][1]["design_audit"]

    try:
        y = data[outcome_var].astype(float)
        rhs_vars = [treatment_var] + list(controls)
        X = sm.add_constant(data[rhs_vars].astype(float))
        mask = y.notna() & X.notna().all(axis=1)
        y = y[mask]
        X = X[mask]

        if nw_lags > 0:
            m = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': nw_lags})
        else:
            m = sm.OLS(y, X).fit(cov_type='HC1')

        coef_val = float(m.params[treatment_var])
        se_val = float(m.bse[treatment_var])
        pval = float(m.pvalues[treatment_var])
        ci = m.conf_int()
        ci_lower = float(ci.loc[treatment_var, 0])
        ci_upper = float(ci.loc[treatment_var, 1])
        nobs = int(m.nobs)
        r2 = float(m.rsquared)

        all_coefs = {k: float(v) for k, v in m.params.items()}

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


# G1 inference variants
run_inference_variant(
    g1_baseline_run_id, "infer/se/hac/nw10",
    "modules/inference/standard_errors.md#hac", "G1",
    "Price", "Turnover", G1_CONTROLS, df_mat, 10, "HAC_NW_10")

run_inference_variant(
    g1_baseline_run_id, "infer/se/hc/hc1",
    "modules/inference/standard_errors.md#heteroskedasticity-robust", "G1",
    "Price", "Turnover", G1_CONTROLS, df_mat, 0, "HC1")

# G2 inference variants
run_inference_variant(
    g2_baseline_run_id, "infer/se/hac/nw10",
    "modules/inference/standard_errors.md#hac", "G2",
    "Warrant_price", "Turnover", G2_CONTROLS, df_daily, 10, "HAC_NW_10")

run_inference_variant(
    g2_baseline_run_id, "infer/se/hac/nw20",
    "modules/inference/standard_errors.md#hac", "G2",
    "Warrant_price", "Turnover", G2_CONTROLS, df_daily, 20, "HAC_NW_20")

run_inference_variant(
    g2_baseline_run_id, "infer/se/hc/hc1",
    "modules/inference/standard_errors.md#heteroskedasticity-robust", "G2",
    "Warrant_price", "Turnover", G2_CONTROLS, df_daily, 0, "HC1")


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

for bg_id in ["G1", "G2"]:
    bg_df = successful[successful['baseline_group_id'] == bg_id]
    if len(bg_df) > 0:
        base_row = bg_df[bg_df['spec_id'] == 'baseline']
        if len(base_row) > 0:
            bc = base_row.iloc[0]
            print(f"\n{bg_id} Baseline coef on {bc['treatment_var']}: {bc['coefficient']:.6f}")
            print(f"  SE: {bc['std_error']:.6f}, p-value: {bc['p_value']:.6f}, N: {bc['n_obs']:.0f}")

        print(f"\n{bg_id} Coefficient range (successful):")
        print(f"  Min: {bg_df['coefficient'].min():.6f}")
        print(f"  Max: {bg_df['coefficient'].max():.6f}")
        print(f"  Median: {bg_df['coefficient'].median():.6f}")
        n_sig = (bg_df['p_value'] < 0.05).sum()
        print(f"  Significant at 5%: {n_sig}/{len(bg_df)}")


# ============================================================
# SPECIFICATION_SEARCH.md
# ============================================================

print("\nWriting SPECIFICATION_SEARCH.md...")

md_lines = []
md_lines.append("# Specification Search Report: 112466-V1")
md_lines.append("")
md_lines.append("**Paper:** Xiong & Yu (2011), \"The Chinese Warrants Bubble\", AER 101(6)")
md_lines.append("")
md_lines.append("## Overview")
md_lines.append("")
md_lines.append("This paper studies the bubble in Chinese put warrants (2005-2008), testing whether")
md_lines.append("warrant prices covary with turnover, volatility, and volume as predicted by resale")
md_lines.append("option theory. Two baseline groups are analyzed:")
md_lines.append("")
md_lines.append("- **G1**: Maturity effect regressions (cross-sectional averages by days remaining, N=60)")
md_lines.append("- **G2**: Daily time-series regressions (warrant 38004, N=468)")
md_lines.append("")

for bg_id in ["G1", "G2"]:
    bg_df = successful[successful['baseline_group_id'] == bg_id]
    base_row = bg_df[bg_df['spec_id'] == 'baseline']

    md_lines.append(f"## {bg_id} Baseline Specification")
    md_lines.append("")

    if bg_id == "G1":
        md_lines.append("- **Design:** Cross-sectional OLS")
        md_lines.append("- **Outcome:** Price (average warrant closing price by days to maturity)")
        md_lines.append("- **Treatment:** Turnover (daily trading volume / shares outstanding)")
        md_lines.append("- **Controls:** Volatility, Volume, Days_remaining")
        md_lines.append("- **SE:** Newey-West HAC (5 lags)")
    else:
        md_lines.append("- **Design:** Time-series OLS")
        md_lines.append("- **Outcome:** Warrant_price (daily closing price)")
        md_lines.append("- **Treatment:** Turnover")
        md_lines.append("- **Controls:** Volatility, Volume, days_to_exp")
        md_lines.append("- **SE:** Newey-West HAC (5 lags)")
    md_lines.append("")

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

# Category breakdown by baseline group
for bg_id in ["G1", "G2"]:
    bg_succ = successful[successful['baseline_group_id'] == bg_id]
    md_lines.append(f"## {bg_id} Category Breakdown")
    md_lines.append("")
    md_lines.append("| Category | Count | Sig. (p<0.05) | Coef Range |")
    md_lines.append("|----------|-------|---------------|------------|")

    categories = {
        "Baseline": bg_succ[bg_succ['spec_id'].str.startswith('baseline')],
        "Controls LOO": bg_succ[bg_succ['spec_id'].str.startswith('rc/controls/loo/')],
        "Controls Sets": bg_succ[bg_succ['spec_id'].str.startswith('rc/controls/sets/')],
        "Controls Progression": bg_succ[bg_succ['spec_id'].str.startswith('rc/controls/progression/')],
        "Controls Subset": bg_succ[bg_succ['spec_id'].str.startswith('rc/controls/subset/')],
        "Sample Trimming/Restriction": bg_succ[bg_succ['spec_id'].str.startswith('rc/sample/')],
        "Functional Form": bg_succ[bg_succ['spec_id'].str.startswith('rc/form/')],
        "Alt Treatment": bg_succ[bg_succ['spec_id'].str.startswith('alt_treatment/')],
        "SE Variants": bg_succ[bg_succ['spec_id'].str.startswith('baseline__')],
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
    md_lines.append("| Group | Spec ID | SE | p-value | 95% CI |")
    md_lines.append("|-------|---------|-----|---------|--------|")
    for _, row in infer_df.iterrows():
        if row['run_success'] == 1:
            md_lines.append(f"| {row['baseline_group_id']} | {row['spec_id']} | {row['std_error']:.6f} | {row['p_value']:.6f} | [{row['ci_lower']:.6f}, {row['ci_upper']:.6f}] |")
        else:
            md_lines.append(f"| {row['baseline_group_id']} | {row['spec_id']} | FAILED | - | - |")

md_lines.append("")

# Overall assessment
md_lines.append("## Overall Assessment")
md_lines.append("")

for bg_id in ["G1", "G2"]:
    bg_succ = successful[successful['baseline_group_id'] == bg_id]
    if len(bg_succ) > 0:
        n_sig_total = (bg_succ['p_value'] < 0.05).sum()
        pct_sig = n_sig_total / len(bg_succ) * 100
        sign_consistent = ((bg_succ['coefficient'] > 0).sum() == len(bg_succ)) or \
                          ((bg_succ['coefficient'] < 0).sum() == len(bg_succ))
        median_coef = bg_succ['coefficient'].median()
        sign_word = "positive" if median_coef > 0 else "negative"

        md_lines.append(f"### {bg_id}")
        md_lines.append(f"- **Sign consistency:** {'All specifications have the same sign' if sign_consistent else 'Mixed signs across specifications'}")
        md_lines.append(f"- **Significance stability:** {n_sig_total}/{len(bg_succ)} ({pct_sig:.1f}%) specifications significant at 5%")
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
