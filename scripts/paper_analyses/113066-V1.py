"""
Specification Search Script for Steinwender (2018)
"Real Effects of Information Frictions: When the States and the Kingdom became United"
American Economic Review, 108(3), 657-696.

Paper ID: 113066-V1

Surface-driven execution:
  - G1: diff ~ tele + controls, Newey-West SE (lag 2)
    Effect of transatlantic telegraph on cotton price difference (level)
  - G2: dev2 ~ tele + controls, Newey-West SE (lag 2)
    Effect of transatlantic telegraph on cotton price difference (variance)
  - Daily time-series data, before/after comparison (pre vs post telegraph)
  - 50+ specifications across outcome definitions, controls, sample restrictions,
    and time windows

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

PAPER_ID = "113066-V1"
DATA_DIR = "data/downloads/extracted/113066-V1"
OUTPUT_DIR = DATA_DIR
DATA_PATH = f"{DATA_DIR}/dta/cotton_data.dta"

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

df_raw = pd.read_stata(DATA_PATH, convert_categoricals=False)
print(f"Loaded data: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

# Convert float32 to float64 for precision
for col in df_raw.columns:
    if df_raw[col].dtype == np.float32:
        df_raw[col] = df_raw[col].astype(np.float64)

# ============================================================
# Construct derived variables (following the Stata do-file)
# ============================================================

# Rescale l1nyrec to thousand bales (1 bale = 400 lbs) as in the paper
df_raw['l1nyrec_k'] = df_raw['l1nyrec'] / 400000.0

# Construct freight-adjusted price differences
# difffrctotal = diff - frctotal (net of total freight cost)
df_raw['difffrctotal'] = df_raw['diff'] - df_raw['frctotal']
# diff_forwardfrctotal = diff_forward - frctotal
df_raw['diff_forwardfrctotal'] = df_raw['diff_forward'] - df_raw['frctotal']
# diff net of freight cost only
df_raw['difffreightcost'] = df_raw['diff'] - df_raw['freightcost']

# Construct dev2 (variance proxy) for different outcome variables
# dev2 = N/(N-1) * (var - mean(var|tele))^2
# This is the sample-corrected squared deviation from pre/post period mean

def construct_dev2(df, var_name, sample_mask=None):
    """Construct dev2 (variance proxy) for a variable, by pre/post telegraph period."""
    col_name = f"dev2_{var_name}"
    df[col_name] = np.nan

    if sample_mask is None:
        sample_mask = pd.Series(True, index=df.index)

    for tele_val in [0.0, 1.0]:
        mask = (df['tele'] == tele_val) & sample_mask & df[var_name].notna()
        vals = df.loc[mask, var_name]
        n = len(vals)
        if n > 1:
            mean_val = vals.mean()
            dev2_vals = (n / (n - 1)) * (vals - mean_val) ** 2
            df.loc[mask, col_name] = dev2_vals

    return col_name

# Full sample (ct_t not null) -- for diff
mask_ct_t = df_raw['ct_t'].notna()
construct_dev2(df_raw, 'diff', mask_ct_t)

# No-trade excluded sample (ct_notrade not null) -- for difffrctotal
mask_notrade = df_raw['ct_notrade'].notna()
construct_dev2(df_raw, 'difffrctotal', mask_notrade)

# No-trade excluded -- for diff_forwardfrctotal
construct_dev2(df_raw, 'diff_forwardfrctotal', mask_notrade)

# Additional variance proxies
construct_dev2(df_raw, 'diff', mask_notrade)  # diff variance on notrade sample
# Rename to distinguish
df_raw['dev2_diff_notrade'] = np.nan
for tele_val in [0.0, 1.0]:
    mask = (df_raw['tele'] == tele_val) & mask_notrade & df_raw['diff'].notna()
    vals = df_raw.loc[mask, 'diff']
    n = len(vals)
    if n > 1:
        mean_val = vals.mean()
        df_raw.loc[mask, 'dev2_diff_notrade'] = (n / (n - 1)) * (vals - mean_val) ** 2

# Log absolute diff
df_raw['log_abs_diff'] = np.log(np.abs(df_raw['diff']).clip(lower=1e-10))

# Absolute deviation (from pre/post mean)
df_raw['abs_dev_diff'] = np.nan
for tele_val in [0.0, 1.0]:
    mask = (df_raw['tele'] == tele_val) & mask_ct_t & df_raw['diff'].notna()
    vals = df_raw.loc[mask, 'diff']
    if len(vals) > 0:
        mean_val = vals.mean()
        df_raw.loc[mask, 'abs_dev_diff'] = np.abs(vals - mean_val)

df_raw['abs_dev_difffrctotal'] = np.nan
for tele_val in [0.0, 1.0]:
    mask = (df_raw['tele'] == tele_val) & mask_notrade & df_raw['difffrctotal'].notna()
    vals = df_raw.loc[mask, 'difffrctotal']
    if len(vals) > 0:
        mean_val = vals.mean()
        df_raw.loc[mask, 'abs_dev_difffrctotal'] = np.abs(vals - mean_val)

# Log dev2
df_raw['log_dev2_difffrctotal'] = np.log(df_raw['dev2_difffrctotal'].clip(lower=1e-10))

# Linear detrending: residualize diff on ct_t
mask_detrend = df_raw['tele'].notna() & df_raw['diff'].notna() & df_raw['ct_t'].notna()
df_temp = df_raw.loc[mask_detrend].copy()
X_trend = sm.add_constant(df_temp['ct_t'])
trend_model = sm.OLS(df_temp['diff'], X_trend).fit()
df_raw['diff_detrended'] = np.nan
df_raw.loc[mask_detrend, 'diff_detrended'] = trend_model.resid.values

# Telegraph date: the paper uses tele as the treatment indicator
# Symmetric window: equal number of obs pre/post
n_pre = (df_raw['tele'] == 0).sum()
n_post = (df_raw['tele'] == 1).sum()
min_n = min(n_pre, n_post)

# Find date boundaries for different windows
# Day variable gives us the actual dates
df_tele = df_raw[df_raw['tele'].notna()].copy()
df_tele = df_tele.sort_values('day')

# Telegraph start roughly after row where tele switches from 0 to 1
pre_days = df_tele[df_tele['tele'] == 0]['day']
post_days = df_tele[df_tele['tele'] == 1]['day']
tele_date = post_days.min()  # first post-telegraph day

# Symmetric window: trim to equal pre/post
if n_pre > n_post:
    # Keep only last n_post pre-telegraph obs
    pre_sorted = df_tele[df_tele['tele'] == 0].sort_values('ct_t')
    sym_pre_start = pre_sorted.iloc[-(min_n):]['ct_t'].min()
    mask_symmetric = (df_raw['tele'].notna()) & (
        ((df_raw['tele'] == 0) & (df_raw['ct_t'] >= sym_pre_start)) |
        (df_raw['tele'] == 1)
    )
elif n_post > n_pre:
    post_sorted = df_tele[df_tele['tele'] == 1].sort_values('ct_t')
    sym_post_end = post_sorted.iloc[:min_n]['ct_t'].max()
    mask_symmetric = (df_raw['tele'].notna()) & (
        (df_raw['tele'] == 0) |
        ((df_raw['tele'] == 1) & (df_raw['ct_t'] <= sym_post_end))
    )
else:
    mask_symmetric = df_raw['tele'].notna()

# Narrow window: ~6 months around telegraph
days_range = (df_tele['day'].max() - df_tele['day'].min()).days
narrow_days = pd.Timedelta(days=180)
mask_narrow = (df_raw['tele'].notna()) & (
    (df_raw['day'] >= tele_date - narrow_days) & (df_raw['day'] <= tele_date + narrow_days)
)

# Wide window: full sample (already have it)
mask_wide = df_raw['tele'].notna()

print(f"Full sample (tele non-null): {mask_wide.sum()}")
print(f"No-trade sample (ct_notrade non-null & tele non-null): {(mask_notrade & df_raw['tele'].notna()).sum()}")
print(f"Symmetric window: {mask_symmetric.sum()}")
print(f"Narrow window (180d): {mask_narrow.sum()}")

# ============================================================
# Accumulators
# ============================================================

results = []
inference_results = []
spec_run_counter = 0

# ============================================================
# Helper: run_spec (OLS with Newey-West HAC via statsmodels)
# ============================================================

def run_spec(spec_id, spec_tree_path, baseline_group_id,
             outcome_var, treatment_var, controls, data,
             nw_lags=2, sample_desc="", controls_desc="",
             design_audit=None, axis_block_name=None, axis_block=None,
             notes=""):
    """Run a single OLS specification with Newey-West HAC standard errors."""
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    if design_audit is None:
        design_audit = design_audit_g1

    try:
        # Build estimation data
        all_vars = [outcome_var, treatment_var] + list(controls)
        est_data = data.dropna(subset=all_vars).copy()

        if len(est_data) < 10:
            raise ValueError(f"Too few observations: {len(est_data)}")

        y = est_data[outcome_var].astype(float)
        rhs_vars = [treatment_var] + list(controls)
        X = sm.add_constant(est_data[rhs_vars].astype(float))

        model = sm.OLS(y, X)
        res = model.fit(cov_type='HAC', cov_kwds={'maxlags': nw_lags})

        coef_val = float(res.params[treatment_var])
        se_val = float(res.bse[treatment_var])
        pval = float(res.pvalues[treatment_var])

        # Get confidence interval for treatment var
        ci = res.conf_int()
        ci_lower = float(ci.iloc[list(res.params.index).index(treatment_var), 0])
        ci_upper = float(ci.iloc[list(res.params.index).index(treatment_var), 1])

        nobs = int(res.nobs)
        r2 = float(res.rsquared)

        all_coefs = {k: float(v) for k, v in res.params.items()}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": inference_canonical["spec_id"],
                       "method": "newey_west", "lags": nw_lags},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"time_series_ols": design_audit},
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
            "cluster_var": f"newey_west_{nw_lags}",
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
            "cluster_var": f"newey_west_{nw_lags}",
            "run_success": 0,
            "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


# ============================================================
# Prepare sample dataframes
# ============================================================

# Full sample: tele not null, ct_t not null (for tsset day)
df_full = df_raw[df_raw['tele'].notna() & df_raw['ct_t'].notna()].copy()
print(f"df_full: {len(df_full)} obs")

# No-trade sample: ct_notrade not null (excludes no-trade days)
df_notrade = df_raw[df_raw['tele'].notna() & df_raw['ct_notrade'].notna()].copy()
print(f"df_notrade: {len(df_notrade)} obs")

# Symmetric window
df_symmetric = df_raw[mask_symmetric & df_raw['ct_t'].notna()].copy()
print(f"df_symmetric: {len(df_symmetric)} obs")

# Narrow window
df_narrow = df_raw[mask_narrow & df_raw['ct_t'].notna()].copy()
print(f"df_narrow: {len(df_narrow)} obs")


# ============================================================
# ======== GROUP G1: PRICE LEVEL SPECIFICATIONS ==============
# ============================================================

print("\n=== GROUP G1: Price Level ===")

# ============================================================
# BASELINE G1: Table 2, Col 1 - diff ~ tele, NW(2)
# ============================================================

print("Running G1 baseline specs...")

# Table2-Col1: diff ~ tele (raw, full sample)
run_spec(
    "baseline__table2_col1", "designs/time_series_ols.md#baseline", "G1",
    "diff", "tele", [],
    df_full, nw_lags=2,
    sample_desc=f"Full sample (tsset day), N={len(df_full)}", controls_desc="none")

# Table2-Col2 (baseline): difffrctotal ~ tele + l1nyrec_k, NW(2), notrade sample
base_run_id_g1, base_coef_g1, base_se_g1, base_pval_g1, base_nobs_g1 = run_spec(
    "baseline__table2_col2", "designs/time_series_ols.md#baseline", "G1",
    "difffrctotal", "tele", ["l1nyrec_k"],
    df_notrade, nw_lags=2,
    sample_desc=f"No-trade excluded (tsset ct_notrade), N={len(df_notrade)}",
    controls_desc="l1nyrec (lagged NY receipts, thousand bales)")

print(f"  G1 Baseline (Table2-Col2): coef={base_coef_g1:.4f}, se={base_se_g1:.4f}, p={base_pval_g1:.4f}, N={base_nobs_g1}")

# Table2-Col3: diff_forwardfrctotal ~ tele + l1nyrec_k, NW(2), notrade sample
run_spec(
    "baseline__table2_col3", "designs/time_series_ols.md#baseline", "G1",
    "diff_forwardfrctotal", "tele", ["l1nyrec_k"],
    df_notrade, nw_lags=2,
    sample_desc=f"No-trade excluded, N={len(df_notrade)}",
    controls_desc="l1nyrec (lagged NY receipts)")


# ============================================================
# RC: CONTROLS VARIANTS (G1)
# ============================================================

print("Running G1 control variants...")

# Add l1nyrec to the raw diff specification (full sample)
run_spec(
    "rc/controls/single/add_l1nyrec", "modules/robustness/controls.md#single-control-add", "G1",
    "diff", "tele", ["l1nyrec_k"],
    df_full, nw_lags=2,
    sample_desc="Full sample", controls_desc="add l1nyrec_k",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/single/add_l1nyrec", "family": "single_add",
                "added": ["l1nyrec_k"], "n_controls": 1})

# Drop l1nyrec from baseline (Col2 -> diff raw on notrade sample)
run_spec(
    "rc/controls/loo/drop_l1nyrec", "modules/robustness/controls.md#leave-one-out-controls-loo", "G1",
    "difffrctotal", "tele", [],
    df_notrade, nw_lags=2,
    sample_desc="No-trade excluded", controls_desc="none (drop l1nyrec)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_l1nyrec", "family": "loo",
                "dropped": ["l1nyrec_k"], "n_controls": 0})

# Add freight cost as control instead of adjusting outcome
run_spec(
    "rc/controls/single/add_freightcost", "modules/robustness/controls.md#single-control-add", "G1",
    "diff", "tele", ["freightcost"],
    df_full, nw_lags=2,
    sample_desc="Full sample", controls_desc="freightcost",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/single/add_freightcost", "family": "single_add",
                "added": ["freightcost"], "n_controls": 1})

# Add frctotal as control
run_spec(
    "rc/controls/single/add_frctotal", "modules/robustness/controls.md#single-control-add", "G1",
    "diff", "tele", ["frctotal"],
    df_full, nw_lags=2,
    sample_desc="Full sample", controls_desc="frctotal (total freight cost)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/single/add_frctotal", "family": "single_add",
                "added": ["frctotal"], "n_controls": 1})

# Both freight and l1nyrec
run_spec(
    "rc/controls/joint/freight_l1nyrec", "modules/robustness/controls.md#control-sets", "G1",
    "diff", "tele", ["frctotal", "l1nyrec_k"],
    df_notrade, nw_lags=2,
    sample_desc="No-trade excluded", controls_desc="frctotal + l1nyrec_k",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/joint/freight_l1nyrec", "family": "sets",
                "n_controls": 2})

# difffrctotal with freightcost instead of frctotal subtraction
run_spec(
    "rc/controls/single/add_freightcost_notrade", "modules/robustness/controls.md#single-control-add", "G1",
    "difffrctotal", "tele", ["freightcost", "l1nyrec_k"],
    df_notrade, nw_lags=2,
    sample_desc="No-trade excluded", controls_desc="freightcost + l1nyrec_k",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/single/add_freightcost_notrade", "family": "single_add",
                "added": ["freightcost"], "n_controls": 2})


# ============================================================
# RC: SAMPLE VARIANTS (G1)
# ============================================================

print("Running G1 sample variants...")

# Exclude no-trade days (on diff, which baseline uses full sample)
run_spec(
    "rc/sample/subset/exclude_notrade_days", "modules/robustness/sample.md#sample-restriction", "G1",
    "diff", "tele", [],
    df_notrade, nw_lags=2,
    sample_desc=f"No-trade excluded, N={len(df_notrade)}", controls_desc="none",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subset/exclude_notrade_days", "axis": "subset",
                "rule": "exclude_notrade", "n_obs": len(df_notrade)})

# Include no-trade days for the freight-adjusted diff
run_spec(
    "rc/sample/subset/include_notrade_days", "modules/robustness/sample.md#sample-restriction", "G1",
    "difffrctotal", "tele", ["l1nyrec_k"],
    df_full, nw_lags=2,
    sample_desc=f"Full sample (including no-trade), N={len(df_full)}",
    controls_desc="l1nyrec_k",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subset/include_notrade_days", "axis": "subset",
                "rule": "include_notrade", "n_obs": len(df_full)})

# Symmetric window
run_spec(
    "rc/sample/period/symmetric_window", "modules/robustness/sample.md#time-window", "G1",
    "difffrctotal", "tele", ["l1nyrec_k"],
    df_symmetric[df_symmetric['ct_notrade'].notna()], nw_lags=2,
    sample_desc=f"Symmetric window, N={len(df_symmetric)}", controls_desc="l1nyrec_k",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/period/symmetric_window", "axis": "time_window",
                "n_obs": len(df_symmetric)})

# Narrow window (6 months around telegraph)
run_spec(
    "rc/sample/period/narrow_window", "modules/robustness/sample.md#time-window", "G1",
    "difffrctotal", "tele", ["l1nyrec_k"],
    df_narrow[df_narrow['ct_notrade'].notna()], nw_lags=2,
    sample_desc=f"Narrow window (180d), N={int(mask_narrow.sum())}", controls_desc="l1nyrec_k",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/period/narrow_window", "axis": "time_window",
                "rule": "narrow_180d", "n_obs": int(mask_narrow.sum())})

# Wide window (full sample, including missing tele -> already handled as df_full)
run_spec(
    "rc/sample/period/wide_window", "modules/robustness/sample.md#time-window", "G1",
    "difffrctotal", "tele", ["l1nyrec_k"],
    df_full[df_full['difffrctotal'].notna()], nw_lags=2,
    sample_desc=f"Wide window (full sample)", controls_desc="l1nyrec_k",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/period/wide_window", "axis": "time_window",
                "rule": "wide_full"})

# Outlier trimming: trim diff at 1/99 percentile
q01 = df_notrade['difffrctotal'].quantile(0.01)
q99 = df_notrade['difffrctotal'].quantile(0.99)
df_trim1 = df_notrade[(df_notrade['difffrctotal'] >= q01) & (df_notrade['difffrctotal'] <= q99)].copy()

run_spec(
    "rc/sample/outliers/trim_diff_1_99", "modules/robustness/sample.md#outliers", "G1",
    "difffrctotal", "tele", ["l1nyrec_k"],
    df_trim1, nw_lags=2,
    sample_desc=f"Trim difffrctotal [1%,99%], N={len(df_trim1)}", controls_desc="l1nyrec_k",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_diff_1_99", "axis": "outliers",
                "rule": "trim", "params": {"lower_q": 0.01, "upper_q": 0.99},
                "n_obs_before": len(df_notrade), "n_obs_after": len(df_trim1)})

# Trim at 5/95 percentile
q05 = df_notrade['difffrctotal'].quantile(0.05)
q95 = df_notrade['difffrctotal'].quantile(0.95)
df_trim5 = df_notrade[(df_notrade['difffrctotal'] >= q05) & (df_notrade['difffrctotal'] <= q95)].copy()

run_spec(
    "rc/sample/outliers/trim_diff_5_95", "modules/robustness/sample.md#outliers", "G1",
    "difffrctotal", "tele", ["l1nyrec_k"],
    df_trim5, nw_lags=2,
    sample_desc=f"Trim difffrctotal [5%,95%], N={len(df_trim5)}", controls_desc="l1nyrec_k",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_diff_5_95", "axis": "outliers",
                "rule": "trim", "params": {"lower_q": 0.05, "upper_q": 0.95},
                "n_obs_before": len(df_notrade), "n_obs_after": len(df_trim5)})


# ============================================================
# RC: OUTCOME FORM VARIANTS (G1)
# ============================================================

print("Running G1 outcome form variants...")

# Raw diff (no freight adjustment) - on full sample
run_spec(
    "rc/form/outcome/diff_raw", "modules/robustness/functional_form.md#outcome-transform", "G1",
    "diff", "tele", ["l1nyrec_k"],
    df_notrade, nw_lags=2,
    sample_desc="No-trade excluded", controls_desc="l1nyrec_k",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/diff_raw", "outcome": "diff",
                "notes": "Raw price diff, not net of freight"})

# Freight-adjusted diff (baseline = difffrctotal, already covered)
# diff_net_freight is the same as difffrctotal

# Forward-looking diff net of freight (Table2-Col3 already a baseline)
# Additional: diff_forward without freight adjustment
run_spec(
    "rc/form/outcome/diff_forward_raw", "modules/robustness/functional_form.md#outcome-transform", "G1",
    "diff_forward", "tele", ["l1nyrec_k"],
    df_notrade, nw_lags=2,
    sample_desc="No-trade excluded", controls_desc="l1nyrec_k",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/diff_forward_raw", "outcome": "diff_forward",
                "notes": "Forward-looking price diff, not net of freight"})

# diff net of freightcost only (not frctotal)
run_spec(
    "rc/form/outcome/diff_net_freightcost", "modules/robustness/functional_form.md#outcome-transform", "G1",
    "difffreightcost", "tele", ["l1nyrec_k"],
    df_notrade, nw_lags=2,
    sample_desc="No-trade excluded", controls_desc="l1nyrec_k",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/diff_net_freightcost", "outcome": "difffreightcost",
                "notes": "Price diff net of freight cost only (not insurance)"})

# Log absolute diff
run_spec(
    "rc/form/outcome/log_diff_abs", "modules/robustness/functional_form.md#outcome-transform", "G1",
    "log_abs_diff", "tele", [],
    df_full, nw_lags=2,
    sample_desc="Full sample", controls_desc="none",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/log_diff_abs", "outcome": "log_abs_diff",
                "notes": "Log of absolute price difference"})

# Detrended diff
run_spec(
    "rc/form/outcome/diff_detrended", "modules/robustness/functional_form.md#outcome-transform", "G1",
    "diff_detrended", "tele", [],
    df_full, nw_lags=2,
    sample_desc="Full sample", controls_desc="none (linearly detrended)",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/preprocess/outcome/detrend_linear", "preprocess": "linear_detrend",
                "notes": "Residualized diff on linear time trend"})

# Detrended diff with l1nyrec
run_spec(
    "rc/preprocess/outcome/detrend_linear_ctrl", "modules/robustness/functional_form.md#outcome-transform", "G1",
    "diff_detrended", "tele", ["l1nyrec_k"],
    df_notrade, nw_lags=2,
    sample_desc="No-trade excluded", controls_desc="l1nyrec_k (detrended outcome)",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/preprocess/outcome/detrend_linear_ctrl", "preprocess": "linear_detrend"})


# ============================================================
# RC: JOINT VARIANTS (G1) - outcome x control combos
# ============================================================

print("Running G1 joint variants...")

# diff_raw + no controls + notrade sample
run_spec(
    "rc/joint/outcome_control/diff_raw_nocontrols", "modules/robustness/joint.md#outcome-x-controls", "G1",
    "diff", "tele", [],
    df_notrade, nw_lags=2,
    sample_desc="No-trade excluded", controls_desc="none",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/outcome_control/diff_raw_nocontrols",
                "outcome": "diff", "controls": "none"})

# diff_forward + frctotal control
run_spec(
    "rc/joint/outcome_control/diff_forward_frctotal_ctrl", "modules/robustness/joint.md#outcome-x-controls", "G1",
    "diff_forward", "tele", ["frctotal", "l1nyrec_k"],
    df_notrade, nw_lags=2,
    sample_desc="No-trade excluded", controls_desc="frctotal + l1nyrec_k",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/outcome_control/diff_forward_frctotal_ctrl",
                "outcome": "diff_forward", "controls": "frctotal+l1nyrec_k"})

# difffrctotal + narrow window
run_spec(
    "rc/joint/sample_outcome/difffrctotal_narrow", "modules/robustness/joint.md#sample-x-outcome", "G1",
    "difffrctotal", "tele", ["l1nyrec_k"],
    df_narrow[df_narrow['ct_notrade'].notna()], nw_lags=2,
    sample_desc="Narrow window, no-trade excluded", controls_desc="l1nyrec_k",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/sample_outcome/difffrctotal_narrow",
                "outcome": "difffrctotal", "sample": "narrow_180d"})

# diff_forward + symmetric window
run_spec(
    "rc/joint/sample_outcome/diff_forward_symmetric", "modules/robustness/joint.md#sample-x-outcome", "G1",
    "diff_forwardfrctotal", "tele", ["l1nyrec_k"],
    df_symmetric[df_symmetric['ct_notrade'].notna()], nw_lags=2,
    sample_desc="Symmetric window, no-trade excluded", controls_desc="l1nyrec_k",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/sample_outcome/diff_forward_symmetric",
                "outcome": "diff_forwardfrctotal", "sample": "symmetric"})

# diff_raw + symmetric + l1nyrec
run_spec(
    "rc/joint/sample_outcome/diff_raw_symmetric", "modules/robustness/joint.md#sample-x-outcome", "G1",
    "diff", "tele", ["l1nyrec_k"],
    df_symmetric, nw_lags=2,
    sample_desc="Symmetric window", controls_desc="l1nyrec_k",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/sample_outcome/diff_raw_symmetric",
                "outcome": "diff", "sample": "symmetric"})

# diff_detrended + trim 1/99
df_trim1_full = df_full.copy()
q01f = df_trim1_full['diff'].quantile(0.01)
q99f = df_trim1_full['diff'].quantile(0.99)
df_trim1_full = df_trim1_full[(df_trim1_full['diff'] >= q01f) & (df_trim1_full['diff'] <= q99f)]

run_spec(
    "rc/joint/sample_outcome/diff_detrended_trim", "modules/robustness/joint.md#sample-x-outcome", "G1",
    "diff_detrended", "tele", [],
    df_trim1_full, nw_lags=2,
    sample_desc=f"Full sample, trim diff [1%,99%], N={len(df_trim1_full)}", controls_desc="none (detrended)",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/sample_outcome/diff_detrended_trim",
                "outcome": "diff_detrended", "sample": "trim_1_99"})

# difffrctotal + freightcost control + trim 5/95
run_spec(
    "rc/joint/outcome_control/difffrctotal_trim5_freightctrl",
    "modules/robustness/joint.md#outcome-x-controls", "G1",
    "difffrctotal", "tele", ["freightcost", "l1nyrec_k"],
    df_trim5, nw_lags=2,
    sample_desc=f"Trim [5%,95%], N={len(df_trim5)}", controls_desc="freightcost + l1nyrec_k",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/outcome_control/difffrctotal_trim5_freightctrl",
                "outcome": "difffrctotal", "controls": "freightcost+l1nyrec_k", "sample": "trim_5_95"})

# Additional G1 specs to reach 50+ total

# diff_forwardfrctotal on narrow window
run_spec(
    "rc/joint/sample_outcome/diff_forward_narrow", "modules/robustness/joint.md#sample-x-outcome", "G1",
    "diff_forwardfrctotal", "tele", ["l1nyrec_k"],
    df_narrow[df_narrow['ct_notrade'].notna()], nw_lags=2,
    sample_desc="Narrow window, no-trade excluded", controls_desc="l1nyrec_k",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/sample_outcome/diff_forward_narrow",
                "outcome": "diff_forwardfrctotal", "sample": "narrow_180d"})

# diff_raw on full sample with no controls
run_spec(
    "rc/joint/outcome_control/diff_raw_full_nocontrols", "modules/robustness/joint.md#outcome-x-controls", "G1",
    "diff", "tele", [],
    df_full, nw_lags=2,
    sample_desc="Full sample", controls_desc="none",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/outcome_control/diff_raw_full_nocontrols",
                "outcome": "diff", "controls": "none", "sample": "full"})

# difffreightcost + no controls on full sample
run_spec(
    "rc/joint/outcome_control/difffreight_full_nocontrols", "modules/robustness/joint.md#outcome-x-controls", "G1",
    "difffreightcost", "tele", [],
    df_full, nw_lags=2,
    sample_desc="Full sample", controls_desc="none",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/outcome_control/difffreight_full_nocontrols",
                "outcome": "difffreightcost", "controls": "none"})

# Symmetric window + diff raw (no freight adjustment, no controls)
run_spec(
    "rc/joint/sample_outcome/diff_raw_symmetric_nocontrols", "modules/robustness/joint.md#sample-x-outcome", "G1",
    "diff", "tele", [],
    df_symmetric, nw_lags=2,
    sample_desc="Symmetric window", controls_desc="none",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/sample_outcome/diff_raw_symmetric_nocontrols",
                "outcome": "diff", "sample": "symmetric", "controls": "none"})

# difffrctotal on full sample with no controls (vs baseline which uses notrade)
run_spec(
    "rc/joint/sample_outcome/difffrctotal_full_nocontrols", "modules/robustness/joint.md#sample-x-outcome", "G1",
    "difffrctotal", "tele", [],
    df_full[df_full['difffrctotal'].notna()], nw_lags=2,
    sample_desc="Full sample (including no-trade)", controls_desc="none",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/sample_outcome/difffrctotal_full_nocontrols",
                "outcome": "difffrctotal", "sample": "full", "controls": "none"})

# diff_forwardfrctotal + no controls on notrade sample
run_spec(
    "rc/joint/outcome_control/diff_forward_nocontrols", "modules/robustness/joint.md#outcome-x-controls", "G1",
    "diff_forwardfrctotal", "tele", [],
    df_notrade, nw_lags=2,
    sample_desc="No-trade excluded", controls_desc="none",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/outcome_control/diff_forward_nocontrols",
                "outcome": "diff_forwardfrctotal", "controls": "none"})


# ============================================================
# ======== GROUP G2: VARIANCE SPECIFICATIONS =================
# ============================================================

print("\n=== GROUP G2: Price Variance ===")

# ============================================================
# BASELINE G2: Table 2, Col 4 - dev2(diff) ~ tele, NW(2)
# ============================================================

print("Running G2 baseline specs...")

# Table2-Col4: variance of raw diff
run_spec(
    "baseline__table2_col4_variance", "designs/time_series_ols.md#baseline", "G2",
    "dev2_diff", "tele", [],
    df_full, nw_lags=2,
    sample_desc=f"Full sample, N={len(df_full)}", controls_desc="none",
    design_audit=design_audit_g2)

# Table2-Col5 (baseline): variance of difffrctotal + l1nyrec_k
base_run_id_g2, base_coef_g2, base_se_g2, base_pval_g2, base_nobs_g2 = run_spec(
    "baseline__table2_col5_variance", "designs/time_series_ols.md#baseline", "G2",
    "dev2_difffrctotal", "tele", ["l1nyrec_k"],
    df_notrade, nw_lags=2,
    sample_desc=f"No-trade excluded, N={len(df_notrade)}", controls_desc="l1nyrec_k",
    design_audit=design_audit_g2)

print(f"  G2 Baseline (Table2-Col5): coef={base_coef_g2:.4f}, se={base_se_g2:.4f}, p={base_pval_g2:.4f}, N={base_nobs_g2}")

# Table2-Col6: variance of forward diff net freight
run_spec(
    "baseline__table2_col6_variance", "designs/time_series_ols.md#baseline", "G2",
    "dev2_diff_forwardfrctotal", "tele", ["l1nyrec_k"],
    df_notrade, nw_lags=2,
    sample_desc=f"No-trade excluded", controls_desc="l1nyrec_k",
    design_audit=design_audit_g2)


# ============================================================
# RC: CONTROLS VARIANTS (G2)
# ============================================================

print("Running G2 control variants...")

# Add l1nyrec to raw diff variance
run_spec(
    "rc/controls/single/add_l1nyrec_var", "modules/robustness/controls.md#single-control-add", "G2",
    "dev2_diff", "tele", ["l1nyrec_k"],
    df_full, nw_lags=2,
    sample_desc="Full sample", controls_desc="l1nyrec_k",
    design_audit=design_audit_g2,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/single/add_l1nyrec_var", "family": "single_add",
                "added": ["l1nyrec_k"]})

# Drop l1nyrec from freight-adjusted variance baseline
run_spec(
    "rc/controls/loo/drop_l1nyrec_var", "modules/robustness/controls.md#leave-one-out-controls-loo", "G2",
    "dev2_difffrctotal", "tele", [],
    df_notrade, nw_lags=2,
    sample_desc="No-trade excluded", controls_desc="none (drop l1nyrec)",
    design_audit=design_audit_g2,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_l1nyrec_var", "family": "loo"})


# ============================================================
# RC: SAMPLE VARIANTS (G2)
# ============================================================

print("Running G2 sample variants...")

# Exclude no-trade on raw diff variance
run_spec(
    "rc/sample/subset/exclude_notrade_days_var", "modules/robustness/sample.md#sample-restriction", "G2",
    "dev2_diff_notrade", "tele", [],
    df_notrade, nw_lags=2,
    sample_desc="No-trade excluded", controls_desc="none",
    design_audit=design_audit_g2,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subset/exclude_notrade_days_var", "axis": "subset"})

# Symmetric window for variance
# Need to reconstruct dev2 on the symmetric sample
df_sym_nt = df_symmetric[df_symmetric['ct_notrade'].notna()].copy()
construct_dev2(df_sym_nt, 'difffrctotal')

run_spec(
    "rc/sample/period/symmetric_window_var", "modules/robustness/sample.md#time-window", "G2",
    "dev2_difffrctotal", "tele", ["l1nyrec_k"],
    df_sym_nt, nw_lags=2,
    sample_desc=f"Symmetric window, no-trade excluded, N={len(df_sym_nt)}", controls_desc="l1nyrec_k",
    design_audit=design_audit_g2,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/period/symmetric_window_var", "axis": "time_window"})

# Trim dev2 at 1/99
q01v = df_notrade['dev2_difffrctotal'].quantile(0.01)
q99v = df_notrade['dev2_difffrctotal'].quantile(0.99)
df_trim_var = df_notrade[(df_notrade['dev2_difffrctotal'] >= q01v) & (df_notrade['dev2_difffrctotal'] <= q99v)].copy()

run_spec(
    "rc/sample/outliers/trim_dev2_1_99", "modules/robustness/sample.md#outliers", "G2",
    "dev2_difffrctotal", "tele", ["l1nyrec_k"],
    df_trim_var, nw_lags=2,
    sample_desc=f"Trim dev2 [1%,99%], N={len(df_trim_var)}", controls_desc="l1nyrec_k",
    design_audit=design_audit_g2,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_dev2_1_99", "axis": "outliers",
                "n_obs_before": len(df_notrade), "n_obs_after": len(df_trim_var)})


# ============================================================
# RC: OUTCOME FORM VARIANTS (G2)
# ============================================================

print("Running G2 outcome form variants...")

# Raw diff variance (no freight adjustment)
run_spec(
    "rc/form/outcome/raw_diff_variance", "modules/robustness/functional_form.md#outcome-transform", "G2",
    "dev2_diff", "tele", ["l1nyrec_k"],
    df_notrade, nw_lags=2,
    sample_desc="No-trade excluded", controls_desc="l1nyrec_k",
    design_audit=design_audit_g2,
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/raw_diff_variance", "outcome": "dev2_diff"})

# Freight-adjusted variance (baseline = dev2_difffrctotal, already covered)

# Forward freight variance
run_spec(
    "rc/form/outcome/forward_freight_variance", "modules/robustness/functional_form.md#outcome-transform", "G2",
    "dev2_diff_forwardfrctotal", "tele", ["l1nyrec_k"],
    df_notrade, nw_lags=2,
    sample_desc="No-trade excluded", controls_desc="l1nyrec_k",
    design_audit=design_audit_g2,
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/forward_freight_variance",
                "outcome": "dev2_diff_forwardfrctotal"})

# Log variance
run_spec(
    "rc/form/outcome/log_variance", "modules/robustness/functional_form.md#outcome-transform", "G2",
    "log_dev2_difffrctotal", "tele", ["l1nyrec_k"],
    df_notrade, nw_lags=2,
    sample_desc="No-trade excluded", controls_desc="l1nyrec_k",
    design_audit=design_audit_g2,
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/log_variance", "outcome": "log_dev2_difffrctotal",
                "notes": "Log of variance proxy"})

# Absolute deviation (instead of squared)
run_spec(
    "rc/form/outcome/absolute_deviation", "modules/robustness/functional_form.md#outcome-transform", "G2",
    "abs_dev_difffrctotal", "tele", ["l1nyrec_k"],
    df_notrade, nw_lags=2,
    sample_desc="No-trade excluded", controls_desc="l1nyrec_k",
    design_audit=design_audit_g2,
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/absolute_deviation", "outcome": "abs_dev_difffrctotal",
                "notes": "Absolute deviation from pre/post mean"})

# Absolute deviation of raw diff
run_spec(
    "rc/form/outcome/absolute_deviation_raw", "modules/robustness/functional_form.md#outcome-transform", "G2",
    "abs_dev_diff", "tele", [],
    df_full, nw_lags=2,
    sample_desc="Full sample", controls_desc="none",
    design_audit=design_audit_g2,
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/absolute_deviation_raw", "outcome": "abs_dev_diff"})


# ============================================================
# RC: JOINT VARIANTS (G2) - outcome x sample combos
# ============================================================

print("Running G2 joint variants...")

# Raw diff variance + symmetric window
df_sym_full = df_symmetric.copy()
construct_dev2(df_sym_full, 'diff')

run_spec(
    "rc/joint/outcome_sample/raw_var_symmetric", "modules/robustness/joint.md#outcome-x-sample", "G2",
    "dev2_diff", "tele", [],
    df_sym_full, nw_lags=2,
    sample_desc="Symmetric window", controls_desc="none",
    design_audit=design_audit_g2,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/outcome_sample/raw_var_symmetric",
                "outcome": "dev2_diff", "sample": "symmetric"})

# Forward variance + trim 1/99
q01fv = df_notrade['dev2_diff_forwardfrctotal'].quantile(0.01)
q99fv = df_notrade['dev2_diff_forwardfrctotal'].quantile(0.99)
df_trim_fv = df_notrade[
    (df_notrade['dev2_diff_forwardfrctotal'] >= q01fv) &
    (df_notrade['dev2_diff_forwardfrctotal'] <= q99fv)
].copy()

run_spec(
    "rc/joint/outcome_sample/forward_var_trim", "modules/robustness/joint.md#outcome-x-sample", "G2",
    "dev2_diff_forwardfrctotal", "tele", ["l1nyrec_k"],
    df_trim_fv, nw_lags=2,
    sample_desc=f"Trim forward var [1%,99%], N={len(df_trim_fv)}", controls_desc="l1nyrec_k",
    design_audit=design_audit_g2,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/outcome_sample/forward_var_trim",
                "outcome": "dev2_diff_forwardfrctotal", "sample": "trim_1_99"})

# Absolute deviation + narrow window
df_narrow_nt = df_narrow[df_narrow['ct_notrade'].notna()].copy()
# Reconstruct abs_dev for narrow sample
for tele_val in [0.0, 1.0]:
    mask_t = (df_narrow_nt['tele'] == tele_val) & df_narrow_nt['difffrctotal'].notna()
    vals = df_narrow_nt.loc[mask_t, 'difffrctotal']
    if len(vals) > 0:
        mean_val = vals.mean()
        df_narrow_nt.loc[mask_t, 'abs_dev_difffrctotal'] = np.abs(vals - mean_val)

run_spec(
    "rc/joint/outcome_sample/absdev_narrow", "modules/robustness/joint.md#outcome-x-sample", "G2",
    "abs_dev_difffrctotal", "tele", ["l1nyrec_k"],
    df_narrow_nt, nw_lags=2,
    sample_desc="Narrow window, no-trade excluded", controls_desc="l1nyrec_k",
    design_audit=design_audit_g2,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/outcome_sample/absdev_narrow",
                "outcome": "abs_dev_difffrctotal", "sample": "narrow_180d"})

# log variance + no controls
run_spec(
    "rc/joint/outcome_control/logvar_nocontrol", "modules/robustness/joint.md#outcome-x-controls", "G2",
    "log_dev2_difffrctotal", "tele", [],
    df_notrade, nw_lags=2,
    sample_desc="No-trade excluded", controls_desc="none",
    design_audit=design_audit_g2,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/outcome_control/logvar_nocontrol",
                "outcome": "log_dev2_difffrctotal", "controls": "none"})


# ============================================================
# INFERENCE VARIANTS (on G1 and G2 baseline specifications)
# ============================================================

print("\n=== INFERENCE VARIANTS ===")

infer_counter = 0

def run_inference_variant(base_run_id, spec_id, spec_tree_path, baseline_group_id,
                          outcome_var, treatment_var, controls, data,
                          vcov_type, vcov_params, vcov_desc, design_audit_ref=None):
    """Re-run baseline spec with different variance-covariance estimator."""
    global infer_counter
    infer_counter += 1
    infer_run_id = f"{PAPER_ID}_infer_{infer_counter:03d}"

    if design_audit_ref is None:
        design_audit_ref = design_audit_g1

    try:
        all_vars = [outcome_var, treatment_var] + list(controls)
        est_data = data.dropna(subset=all_vars).copy()

        y = est_data[outcome_var].astype(float)
        rhs_vars = [treatment_var] + list(controls)
        X = sm.add_constant(est_data[rhs_vars].astype(float))

        model = sm.OLS(y, X)

        if vcov_type == 'HAC':
            res = model.fit(cov_type='HAC', cov_kwds={'maxlags': vcov_params.get('lags', 2)})
        elif vcov_type == 'HC1':
            res = model.fit(cov_type='HC1')
        else:
            res = model.fit(cov_type='HAC', cov_kwds={'maxlags': vcov_params.get('lags', 2)})

        coef_val = float(res.params[treatment_var])
        se_val = float(res.bse[treatment_var])
        pval = float(res.pvalues[treatment_var])

        ci = res.conf_int()
        treat_idx = list(res.params.index).index(treatment_var)
        ci_lower = float(ci.iloc[treat_idx, 0])
        ci_upper = float(ci.iloc[treat_idx, 1])

        nobs = int(res.nobs)
        r2 = float(res.rsquared)

        all_coefs = {k: float(v) for k, v in res.params.items()}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": spec_id, "method": vcov_desc},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"time_series_ols": design_audit_ref},
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


# G1 inference variants on baseline (Table2-Col2: difffrctotal ~ tele + l1nyrec_k)
g1_base_run = f"{PAPER_ID}_run_002"  # baseline__table2_col2

# Newey-West 4 lags
run_inference_variant(
    g1_base_run, "infer/se/hac/newey_west_4",
    "modules/inference/standard_errors.md#hac", "G1",
    "difffrctotal", "tele", ["l1nyrec_k"], df_notrade,
    "HAC", {"lags": 4}, "Newey-West HAC (4 lags)")

# Newey-West 8 lags
run_inference_variant(
    g1_base_run, "infer/se/hac/newey_west_8",
    "modules/inference/standard_errors.md#hac", "G1",
    "difffrctotal", "tele", ["l1nyrec_k"], df_notrade,
    "HAC", {"lags": 8}, "Newey-West HAC (8 lags)")

# HC1 robust (no autocorrelation correction)
run_inference_variant(
    g1_base_run, "infer/se/hc/hc1",
    "modules/inference/standard_errors.md#heteroskedasticity-robust", "G1",
    "difffrctotal", "tele", ["l1nyrec_k"], df_notrade,
    "HC1", {}, "HC1 (robust, no autocorrelation)")

# G2 inference variants on baseline (Table2-Col5: dev2_difffrctotal ~ tele + l1nyrec_k)
g2_base_run = f"{PAPER_ID}_run_{spec_run_counter - 11:03d}"  # roughly the col5 run
# Find actual g2 baseline run
g2_base_run = base_run_id_g2

# Newey-West 4 lags
run_inference_variant(
    g2_base_run, "infer/se/hac/newey_west_4_var",
    "modules/inference/standard_errors.md#hac", "G2",
    "dev2_difffrctotal", "tele", ["l1nyrec_k"], df_notrade,
    "HAC", {"lags": 4}, "Newey-West HAC (4 lags)",
    design_audit_ref=design_audit_g2)

# HC1 robust for G2
run_inference_variant(
    g2_base_run, "infer/se/hc/hc1_var",
    "modules/inference/standard_errors.md#heteroskedasticity-robust", "G2",
    "dev2_difffrctotal", "tele", ["l1nyrec_k"], df_notrade,
    "HC1", {}, "HC1 (robust, no autocorrelation)",
    design_audit_ref=design_audit_g2)


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
    # G1 baseline
    base_g1 = spec_df[spec_df['spec_id'] == 'baseline__table2_col2']
    if len(base_g1) > 0:
        print(f"\nG1 Baseline coef on tele: {base_g1['coefficient'].values[0]:.6f}")
        print(f"G1 Baseline SE: {base_g1['std_error'].values[0]:.6f}")
        print(f"G1 Baseline p-value: {base_g1['p_value'].values[0]:.6f}")
        print(f"G1 Baseline N: {base_g1['n_obs'].values[0]:.0f}")

    # G2 baseline
    base_g2 = spec_df[spec_df['spec_id'] == 'baseline__table2_col5_variance']
    if len(base_g2) > 0:
        print(f"\nG2 Baseline coef on tele: {base_g2['coefficient'].values[0]:.6f}")
        print(f"G2 Baseline SE: {base_g2['std_error'].values[0]:.6f}")
        print(f"G2 Baseline p-value: {base_g2['p_value'].values[0]:.6f}")
        print(f"G2 Baseline N: {base_g2['n_obs'].values[0]:.0f}")

    print(f"\n=== COEFFICIENT RANGE (successful specs) ===")

    # G1 range
    g1_specs = successful[successful['baseline_group_id'] == 'G1']
    if len(g1_specs) > 0:
        print(f"G1 (level): min={g1_specs['coefficient'].min():.6f}, max={g1_specs['coefficient'].max():.6f}, median={g1_specs['coefficient'].median():.6f}")
        n_sig_g1 = (g1_specs['p_value'] < 0.05).sum()
        print(f"  Significant at 5%: {n_sig_g1}/{len(g1_specs)}")

    # G2 range
    g2_specs = successful[successful['baseline_group_id'] == 'G2']
    if len(g2_specs) > 0:
        print(f"G2 (variance): min={g2_specs['coefficient'].min():.6f}, max={g2_specs['coefficient'].max():.6f}, median={g2_specs['coefficient'].median():.6f}")
        n_sig_g2 = (g2_specs['p_value'] < 0.05).sum()
        print(f"  Significant at 5%: {n_sig_g2}/{len(g2_specs)}")

if len(failed) > 0:
    print(f"\nFailed specs:")
    for _, row in failed.iterrows():
        print(f"  {row['spec_id']}: {row['run_error']}")


# ============================================================
# SPECIFICATION_SEARCH.md
# ============================================================

print("\nWriting SPECIFICATION_SEARCH.md...")

md_lines = []
md_lines.append("# Specification Search Report: 113066-V1")
md_lines.append("")
md_lines.append("**Paper:** Steinwender (2018), \"Real Effects of Information Frictions: When the States and the Kingdom became United\", AER 108(3)")
md_lines.append("")
md_lines.append("## Design")
md_lines.append("")
md_lines.append("- **Design:** Before-after comparison (pre vs post transatlantic telegraph, 1866)")
md_lines.append("- **Data:** Daily cotton price data, Liverpool and New York, ~1865-1867")
md_lines.append("- **Treatment:** tele (= 1 after telegraph introduction)")
md_lines.append("- **SE:** Newey-West HAC with 2 lags (baseline)")
md_lines.append("")

md_lines.append("## Baseline Group G1: Price Level")
md_lines.append("")
md_lines.append("- **Outcome:** difffrctotal (cotton price difference net of freight cost)")
md_lines.append("- **Controls:** l1nyrec (lagged NY receipts, in thousand bales)")
md_lines.append("- **Sample:** Excludes no-trade days")
md_lines.append("")

base_g1 = spec_df[spec_df['spec_id'] == 'baseline__table2_col2']
if len(base_g1) > 0:
    bc = base_g1.iloc[0]
    md_lines.append(f"| Statistic | Value |")
    md_lines.append(f"|-----------|-------|")
    md_lines.append(f"| Coefficient | {bc['coefficient']:.6f} |")
    md_lines.append(f"| Std. Error | {bc['std_error']:.6f} |")
    md_lines.append(f"| p-value | {bc['p_value']:.6f} |")
    md_lines.append(f"| 95% CI | [{bc['ci_lower']:.6f}, {bc['ci_upper']:.6f}] |")
    md_lines.append(f"| N | {bc['n_obs']:.0f} |")
    md_lines.append(f"| R-squared | {bc['r_squared']:.4f} |")
    md_lines.append("")

md_lines.append("## Baseline Group G2: Price Variance")
md_lines.append("")
md_lines.append("- **Outcome:** dev2_difffrctotal (sample-corrected squared deviation from pre/post mean)")
md_lines.append("- **Controls:** l1nyrec (lagged NY receipts)")
md_lines.append("")

base_g2 = spec_df[spec_df['spec_id'] == 'baseline__table2_col5_variance']
if len(base_g2) > 0:
    bc = base_g2.iloc[0]
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
md_lines.append("### G1: Price Level")
md_lines.append("")
md_lines.append("| Category | Count | Sig. (p<0.05) | Coef Range |")
md_lines.append("|----------|-------|---------------|------------|")

g1_all = successful[successful['baseline_group_id'] == 'G1']
categories_g1 = {
    "Baseline": g1_all[g1_all['spec_id'].str.startswith('baseline')],
    "Controls": g1_all[g1_all['spec_id'].str.startswith('rc/controls/')],
    "Sample": g1_all[g1_all['spec_id'].str.startswith('rc/sample/')],
    "Outcome Form": g1_all[g1_all['spec_id'].str.startswith('rc/form/') | g1_all['spec_id'].str.startswith('rc/preprocess/')],
    "Joint": g1_all[g1_all['spec_id'].str.startswith('rc/joint/')],
}

for cat_name, cat_df in categories_g1.items():
    if len(cat_df) > 0:
        n_sig_cat = (cat_df['p_value'] < 0.05).sum()
        coef_range = f"[{cat_df['coefficient'].min():.4f}, {cat_df['coefficient'].max():.4f}]"
        md_lines.append(f"| {cat_name} | {len(cat_df)} | {n_sig_cat}/{len(cat_df)} | {coef_range} |")

md_lines.append("")
md_lines.append("### G2: Price Variance")
md_lines.append("")
md_lines.append("| Category | Count | Sig. (p<0.05) | Coef Range |")
md_lines.append("|----------|-------|---------------|------------|")

g2_all = successful[successful['baseline_group_id'] == 'G2']
categories_g2 = {
    "Baseline": g2_all[g2_all['spec_id'].str.startswith('baseline')],
    "Controls": g2_all[g2_all['spec_id'].str.startswith('rc/controls/')],
    "Sample": g2_all[g2_all['spec_id'].str.startswith('rc/sample/')],
    "Outcome Form": g2_all[g2_all['spec_id'].str.startswith('rc/form/')],
    "Joint": g2_all[g2_all['spec_id'].str.startswith('rc/joint/')],
}

for cat_name, cat_df in categories_g2.items():
    if len(cat_df) > 0:
        n_sig_cat = (cat_df['p_value'] < 0.05).sum()
        coef_range = f"[{cat_df['coefficient'].min():.4f}, {cat_df['coefficient'].max():.4f}]"
        md_lines.append(f"| {cat_name} | {len(cat_df)} | {n_sig_cat}/{len(cat_df)} | {coef_range} |")

md_lines.append("")

# Inference variants
md_lines.append("## Inference Variants")
md_lines.append("")
if len(infer_df) > 0:
    md_lines.append("| Spec ID | Group | SE | p-value | 95% CI |")
    md_lines.append("|---------|-------|-----|---------|--------|")
    for _, row in infer_df.iterrows():
        if row['run_success'] == 1:
            md_lines.append(f"| {row['spec_id']} | {row['baseline_group_id']} | {row['std_error']:.6f} | {row['p_value']:.6f} | [{row['ci_lower']:.6f}, {row['ci_upper']:.6f}] |")
        else:
            md_lines.append(f"| {row['spec_id']} | {row['baseline_group_id']} | FAILED | - | - |")

md_lines.append("")

# Overall assessment
md_lines.append("## Overall Assessment")
md_lines.append("")

for group_id, group_label, group_specs in [("G1", "Price Level", g1_all), ("G2", "Price Variance", g2_all)]:
    if len(group_specs) > 0:
        n_sig_total = (group_specs['p_value'] < 0.05).sum()
        pct_sig = n_sig_total / len(group_specs) * 100
        median_coef = group_specs['coefficient'].median()

        # For G1, expect negative coefficient (telegraph reduces price gap)
        # For G2, expect negative coefficient (telegraph reduces variance)
        sign_consistent = ((group_specs['coefficient'] < 0).sum() == len(group_specs)) or \
                          ((group_specs['coefficient'] > 0).sum() == len(group_specs))
        sign_word = "negative" if median_coef < 0 else "positive"

        md_lines.append(f"### {group_id}: {group_label}")
        md_lines.append("")
        md_lines.append(f"- **Sign consistency:** {'All specifications have the same sign' if sign_consistent else 'Mixed signs across specifications'}")
        md_lines.append(f"- **Significance stability:** {n_sig_total}/{len(group_specs)} ({pct_sig:.1f}%) specifications significant at 5%")
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
