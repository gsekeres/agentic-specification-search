"""
Specification Search Script for Camacho & Conover (2011)
"Manipulation of Social Program Eligibility"
American Economic Journal: Economic Policy, 3(2), 41-65.

Paper ID: 114759-V1

Surface-driven execution:
  - G1: Density discontinuity in SISBEN poverty score at cutoff=47
  - Sharp RD with local linear regression, triangular kernel, data-driven bandwidth
  - Outcome: share of surveys at each score (collapsed density)
  - Treatment: jump (score <= 47 indicator)
  - Core universe: bandwidth, polynomial, kernel, donut, sample restriction, placebo cutoff variants

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

PAPER_ID = "114759-V1"
DATA_DIR = "data/downloads/extracted/114759-V1"
OUTPUT_DIR = DATA_DIR
DATA_PATH = f"{DATA_DIR}/data/AEJPol-2010-0061_data/sisben_aejep.dta"

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

bg = surface_obj["baseline_groups"][0]
design_audit = bg["design_audit"]
inference_canonical = bg["inference_plan"]["canonical"]

CUTOFF = 47

# ============================================================
# Data Loading and Preparation
# ============================================================
# Replicate the Stata Table 3 data construction:
# 1. Load raw survey data (only needed columns)
# 2. Keep urban (zona==1), SES 1-3, years 1994-2003
# 3. Collapse to score-level counts by year
# 4. Compute percentage at each score (density)

print("Loading raw survey data (this may take a moment due to 18M rows)...")

import pyreadstat
df_raw, meta = pyreadstat.read_dta(
    DATA_PATH,
    usecols=['puntaje', 'zona', 'estrato', 'fencuesta']
)
print(f"Loaded raw data: {len(df_raw)} rows")

# Rename to match do-file
df_raw.rename(columns={'puntaje': 'score', 'zona': 'area', 'estrato': 'ses'}, inplace=True)

# Filter: urban only, SES 1-3
df_raw = df_raw[df_raw['area'] == 1].copy()
df_raw = df_raw[df_raw['ses'] <= 3].copy()

# Extract year from date
df_raw['doi_yr'] = pd.to_datetime(df_raw['fencuesta']).dt.year
df_raw = df_raw[(df_raw['doi_yr'] >= 1994) & (df_raw['doi_yr'] <= 2003)].copy()
print(f"After filtering (urban, ses<=3, 1994-2003): {len(df_raw)} rows")

# Drop missing scores
df_raw = df_raw.dropna(subset=['score']).copy()
df_raw['score'] = df_raw['score'].astype(int)

# Add survey counter
df_raw['survey'] = 1

# ============================================================
# Collapse to score-level density (one row per score per year)
# ============================================================

collapsed = df_raw.groupby(['score', 'doi_yr'])['survey'].sum().reset_index()

# Reshape wide: one column per year
collapsed_wide = collapsed.pivot(index='score', columns='doi_yr', values='survey').fillna(0)
collapsed_wide.columns = [f'survey{int(c)}' for c in collapsed_wide.columns]
collapsed_wide = collapsed_wide.reset_index()

# Compute percentage (proportion) for each year -- same as Stata egen pc()
for yr in range(1994, 2004):
    col = f'survey{yr}'
    pct_col = f'sisses{yr}'
    if col in collapsed_wide.columns:
        total = collapsed_wide[col].sum()
        collapsed_wide[pct_col] = (collapsed_wide[col] / total) * 100.0
    else:
        collapsed_wide[pct_col] = 0.0

# Create RD variables
collapsed_wide['align'] = collapsed_wide['score'] - CUTOFF
collapsed_wide['elig'] = (collapsed_wide['score'] <= CUTOFF).astype(int)
collapsed_wide['jump'] = collapsed_wide['elig']
collapsed_wide['jump_align'] = collapsed_wide['jump'] * collapsed_wide['align']

# Higher-order polynomial terms
for p in range(2, 9):
    collapsed_wide[f'align{p}'] = collapsed_wide['align'] ** p
    collapsed_wide[f'align{p}_jump'] = collapsed_wide[f'align{p}'] * collapsed_wide['jump']

print(f"Collapsed data: {len(collapsed_wide)} score values")
print(f"Score range: [{collapsed_wide['score'].min()}, {collapsed_wide['score'].max()}]")

# Also create a pooled outcome: average across all years
year_cols = [f'sisses{yr}' for yr in range(1994, 2004) if f'sisses{yr}' in collapsed_wide.columns]
collapsed_wide['sisses_pooled'] = collapsed_wide[year_cols].mean(axis=1)

# ============================================================
# Bandwidth computation (Imbens-Kalyanaraman style)
# Replicate the Stata code step-by-step for each year
# ============================================================

def compute_ik_bandwidth(data, outcome_col):
    """Compute Imbens-Kalyanaraman optimal bandwidth.

    Following the Stata do-file:
    Step 1: binsize = 2 * sd(align) * N^(-0.5)
    Step 2: kappa = 3.348
    Pre-side: regress outcome on align + align^2 + align^3 + align^4 for align<=0
              compute sum of squared second derivatives
    Post-side: same for align>0
    Bandwidth = kappa * [(MSE * range) / sum_sq_2nd_deriv]^0.2
    Final: average of pre and post bandwidths
    """
    df = data[['align', outcome_col] + [f'align{p}' for p in range(2, 5)]].dropna().copy()

    N = len(df)
    if N < 10:
        return 10.0  # fallback

    ssd_align = df['align'].std()
    kappa = 3.348

    try:
        # Pre-side (align <= 0)
        pre = df[df['align'] <= 0].copy()
        if len(pre) < 5:
            return 10.0

        from numpy.linalg import lstsq
        X_pre = np.column_stack([
            pre['align'].values,
            pre['align2'].values,
            pre['align3'].values,
            pre['align4'].values
        ])
        X_pre = np.column_stack([np.ones(len(pre)), X_pre])
        y_pre = pre[outcome_col].values

        beta_pre, _, _, _ = lstsq(X_pre, y_pre, rcond=None)
        resid_pre = y_pre - X_pre @ beta_pre
        mse_pre = np.mean(resid_pre**2)

        # Second derivative: 2*b2 + 6*b3*align + 12*b4*align^2
        b2, b3, b4 = beta_pre[2], beta_pre[3], beta_pre[4]
        second_deriv_pre = 2*b2 + 6*b3*pre['align'].values + 12*b4*pre['align2'].values
        pressd = np.sum(second_deriv_pre**2)
        predist = -pre['align'].min()

        prebw = kappa * ((mse_pre * predist) / pressd) ** 0.2

        # Post-side (align > 0)
        post = df[df['align'] > 0].copy()
        if len(post) < 5:
            return prebw

        X_post = np.column_stack([
            post['align'].values,
            post['align2'].values,
            post['align3'].values,
            post['align4'].values
        ])
        X_post = np.column_stack([np.ones(len(post)), X_post])
        y_post = post[outcome_col].values

        beta_post, _, _, _ = lstsq(X_post, y_post, rcond=None)
        resid_post = y_post - X_post @ beta_post
        mse_post = np.mean(resid_post**2)

        b2, b3, b4 = beta_post[2], beta_post[3], beta_post[4]
        second_deriv_post = 2*b2 + 6*b3*post['align'].values + 12*b4*post['align2'].values
        postssd = np.sum(second_deriv_post**2)
        postdist = post['align'].max()

        postbw = kappa * ((mse_post * postdist) / postssd) ** 0.2

        obw = 0.5 * (prebw + postbw)
        return max(obw, 3.0)  # minimum bandwidth of 3

    except Exception as e:
        print(f"  IK bandwidth computation failed: {e}, using fallback bw=10")
        return 10.0


def triangular_kernel(x, h):
    """Triangular kernel weight: max(0, 1 - |x/h|)"""
    norm = 1 - np.abs(x / h)
    return np.maximum(norm, 0)


def uniform_kernel(x, h):
    """Uniform kernel: 1 if |x| <= h, else 0"""
    return (np.abs(x) <= h).astype(float)


def epanechnikov_kernel(x, h):
    """Epanechnikov kernel: 0.75 * (1 - (x/h)^2) for |x/h| <= 1"""
    u = x / h
    w = 0.75 * (1 - u**2)
    w[np.abs(u) > 1] = 0
    return np.maximum(w, 0)


# ============================================================
# Accumulators
# ============================================================

results = []
inference_results = []
spec_run_counter = 0


# ============================================================
# Helper: run_rd_spec
# ============================================================

def run_rd_spec(spec_id, spec_tree_path, baseline_group_id,
                outcome_col, data, bandwidth, kernel_func, poly_order,
                cutoff_val, sample_desc, design_desc, vcov_type="HC1",
                donut=0, axis_block_name=None, axis_block=None, notes=""):
    """Run a single RD specification using weighted local polynomial regression.

    Following the Stata do-file approach:
    - Restrict to observations within bandwidth
    - Apply kernel weights
    - Run WLS: outcome ~ jump + align + jump_align [+ higher poly terms]
    - Report coefficient on 'jump'
    """
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        df = data.copy()
        h = bandwidth

        # Apply donut hole exclusion
        if donut > 0:
            df = df[np.abs(df['align']) > donut].copy()

        # Compute kernel weights
        df['k_weight'] = kernel_func(df['align'].values, h)

        # Keep only observations with positive weight (within bandwidth)
        df = df[df['k_weight'] > 0].copy()

        if len(df) < 5:
            raise ValueError(f"Too few observations within bandwidth h={h:.2f} (N={len(df)})")

        # Build formula based on polynomial order
        treatment_var = "jump"
        controls = ["align", "jump_align"]

        if poly_order >= 2:
            controls.extend(["align2", "align2_jump"])
        if poly_order >= 3:
            controls.extend(["align3", "align3_jump"])

        controls_str = " + ".join(controls)
        formula = f"{outcome_col} ~ {treatment_var} + {controls_str}"

        # Set up vcov
        if vcov_type == "HC1":
            vcov = "hetero"
        elif vcov_type == "cluster_score":
            df['score_str'] = df['score'].astype(str)
            vcov = {"CRV1": "score_str"}
        else:
            vcov = "hetero"

        # Run weighted regression
        m = pf.feols(formula, data=df, vcov=vcov, weights="k_weight")

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
                       "method": vcov_type},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"regression_discontinuity": {
                "cutoff": cutoff_val,
                "bandwidth": float(h),
                "poly_order": poly_order,
                "kernel": kernel_func.__name__.replace("_kernel", ""),
                "donut": donut,
                **design_audit
            }},
            axis_block_name=axis_block_name,
            axis_block=axis_block,
        )

        results.append({
            "paper_id": PAPER_ID,
            "spec_run_id": run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_col,
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
            "controls_desc": design_desc,
            "cluster_var": vcov_type,
            "run_success": 1,
            "run_error": ""
        })
        return run_id, coef_val, se_val, pval, nobs

    except Exception as e:
        err_msg = str(e)[:240]
        err_details = error_details_from_exception(e, stage="rd_estimation")
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
            "outcome_var": outcome_col,
            "treatment_var": "jump",
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
            "controls_desc": design_desc,
            "cluster_var": vcov_type,
            "run_success": 0,
            "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


# ============================================================
# Compute baseline bandwidths for each year and pooled
# ============================================================

print("\nComputing IK bandwidths for each year...")
bandwidths = {}
for yr in range(1994, 2004):
    outcome_col = f'sisses{yr}'
    if outcome_col in collapsed_wide.columns:
        bw = compute_ik_bandwidth(collapsed_wide, outcome_col)
        bandwidths[yr] = bw
        print(f"  {yr}: bandwidth = {bw:.2f}")

# Pooled bandwidth: average of all year bandwidths
bw_pooled = np.mean(list(bandwidths.values()))
bandwidths['pooled'] = bw_pooled
print(f"  Pooled: bandwidth = {bw_pooled:.2f}")


# ============================================================
# BASELINE SPECS: Year-by-year (surface baseline_spec_ids)
# ============================================================

print("\n=== Running Baseline Specifications ===")

baseline_years = {
    "baseline__1997": 1997,
    "baseline__1999": 1999,
    "baseline__2001": 2001,
    "baseline__2003": 2003,
}

for spec_id, yr in baseline_years.items():
    outcome_col = f'sisses{yr}'
    bw = bandwidths.get(yr, bw_pooled)
    print(f"\nRunning {spec_id} (bw={bw:.2f})...")

    rid, coef, se, pval, nobs = run_rd_spec(
        spec_id,
        "designs/regression_discontinuity.md#baseline",
        "G1",
        outcome_col,
        collapsed_wide,
        bw,
        triangular_kernel,
        1,  # local linear
        CUTOFF,
        f"Year {yr}, IK bandwidth={bw:.2f}",
        f"Local linear, triangular kernel, h={bw:.2f}",
    )
    print(f"  coef={coef:.4f}, se={se:.4f}, p={pval:.4f}, N={nobs}" if not np.isnan(coef) else "  FAILED")

# Also run pooled baseline
print("\nRunning pooled baseline...")
rid_pooled, coef_pooled, se_pooled, pval_pooled, nobs_pooled = run_rd_spec(
    "baseline__pooled",
    "designs/regression_discontinuity.md#baseline",
    "G1",
    "sisses_pooled",
    collapsed_wide,
    bw_pooled,
    triangular_kernel,
    1,
    CUTOFF,
    f"Pooled (avg across years), IK bandwidth={bw_pooled:.2f}",
    f"Local linear, triangular kernel, h={bw_pooled:.2f}",
)
print(f"  Pooled: coef={coef_pooled:.4f}, se={se_pooled:.4f}, p={pval_pooled:.4f}, N={nobs_pooled}"
      if not np.isnan(coef_pooled) else "  FAILED")


# ============================================================
# DESIGN VARIANTS: Bandwidth
# ============================================================

print("\n=== Running Design Variants: Bandwidth ===")

# Use 2003 as the reference year (strongest manipulation period)
REF_YEAR = 2003
ref_outcome = f'sisses{REF_YEAR}'
ref_bw = bandwidths.get(REF_YEAR, bw_pooled)

# Half baseline bandwidth
run_rd_spec(
    "design/regression_discontinuity/bandwidth/half_baseline",
    "designs/regression_discontinuity.md#bandwidth",
    "G1", ref_outcome, collapsed_wide,
    ref_bw * 0.5, triangular_kernel, 1, CUTOFF,
    f"Year {REF_YEAR}, h={ref_bw*0.5:.2f} (half baseline)",
    f"Local linear, triangular, h={ref_bw*0.5:.2f}",
    axis_block_name="design",
    axis_block={"spec_id": "design/regression_discontinuity/bandwidth/half_baseline",
                "bandwidth_multiplier": 0.5})

# Double baseline bandwidth
run_rd_spec(
    "design/regression_discontinuity/bandwidth/double_baseline",
    "designs/regression_discontinuity.md#bandwidth",
    "G1", ref_outcome, collapsed_wide,
    ref_bw * 2.0, triangular_kernel, 1, CUTOFF,
    f"Year {REF_YEAR}, h={ref_bw*2.0:.2f} (double baseline)",
    f"Local linear, triangular, h={ref_bw*2.0:.2f}",
    axis_block_name="design",
    axis_block={"spec_id": "design/regression_discontinuity/bandwidth/double_baseline",
                "bandwidth_multiplier": 2.0})


# ============================================================
# DESIGN VARIANTS: Polynomial order
# ============================================================

print("\n=== Running Design Variants: Polynomial Order ===")

# Local linear (p=1) -- already done as baseline, but run on ref year explicitly
run_rd_spec(
    "design/regression_discontinuity/poly/local_linear",
    "designs/regression_discontinuity.md#polynomial",
    "G1", ref_outcome, collapsed_wide,
    ref_bw, triangular_kernel, 1, CUTOFF,
    f"Year {REF_YEAR}, local linear (p=1)",
    f"Local linear (p=1), triangular, h={ref_bw:.2f}",
    axis_block_name="design",
    axis_block={"spec_id": "design/regression_discontinuity/poly/local_linear",
                "poly_order": 1})

# Local quadratic (p=2)
run_rd_spec(
    "design/regression_discontinuity/poly/local_quadratic",
    "designs/regression_discontinuity.md#polynomial",
    "G1", ref_outcome, collapsed_wide,
    ref_bw, triangular_kernel, 2, CUTOFF,
    f"Year {REF_YEAR}, local quadratic (p=2)",
    f"Local quadratic (p=2), triangular, h={ref_bw:.2f}",
    axis_block_name="design",
    axis_block={"spec_id": "design/regression_discontinuity/poly/local_quadratic",
                "poly_order": 2})

# Local cubic (p=3)
run_rd_spec(
    "design/regression_discontinuity/poly/local_cubic",
    "designs/regression_discontinuity.md#polynomial",
    "G1", ref_outcome, collapsed_wide,
    ref_bw, triangular_kernel, 3, CUTOFF,
    f"Year {REF_YEAR}, local cubic (p=3)",
    f"Local cubic (p=3), triangular, h={ref_bw:.2f}",
    axis_block_name="design",
    axis_block={"spec_id": "design/regression_discontinuity/poly/local_cubic",
                "poly_order": 3})


# ============================================================
# DESIGN VARIANTS: Kernel
# ============================================================

print("\n=== Running Design Variants: Kernel ===")

# Triangular (already baseline)
run_rd_spec(
    "design/regression_discontinuity/kernel/triangular",
    "designs/regression_discontinuity.md#kernel",
    "G1", ref_outcome, collapsed_wide,
    ref_bw, triangular_kernel, 1, CUTOFF,
    f"Year {REF_YEAR}, triangular kernel",
    f"Local linear, triangular, h={ref_bw:.2f}",
    axis_block_name="design",
    axis_block={"spec_id": "design/regression_discontinuity/kernel/triangular",
                "kernel": "triangular"})

# Uniform kernel
run_rd_spec(
    "design/regression_discontinuity/kernel/uniform",
    "designs/regression_discontinuity.md#kernel",
    "G1", ref_outcome, collapsed_wide,
    ref_bw, uniform_kernel, 1, CUTOFF,
    f"Year {REF_YEAR}, uniform kernel",
    f"Local linear, uniform, h={ref_bw:.2f}",
    axis_block_name="design",
    axis_block={"spec_id": "design/regression_discontinuity/kernel/uniform",
                "kernel": "uniform"})

# Epanechnikov kernel
run_rd_spec(
    "design/regression_discontinuity/kernel/epanechnikov",
    "designs/regression_discontinuity.md#kernel",
    "G1", ref_outcome, collapsed_wide,
    ref_bw, epanechnikov_kernel, 1, CUTOFF,
    f"Year {REF_YEAR}, Epanechnikov kernel",
    f"Local linear, Epanechnikov, h={ref_bw:.2f}",
    axis_block_name="design",
    axis_block={"spec_id": "design/regression_discontinuity/kernel/epanechnikov",
                "kernel": "epanechnikov"})


# ============================================================
# DESIGN VARIANTS: Procedure (conventional vs robust bias-corrected)
# ============================================================

print("\n=== Running Design Variants: Procedure ===")

# Conventional (already done)
run_rd_spec(
    "design/regression_discontinuity/procedure/conventional",
    "designs/regression_discontinuity.md#procedure",
    "G1", ref_outcome, collapsed_wide,
    ref_bw, triangular_kernel, 1, CUTOFF,
    f"Year {REF_YEAR}, conventional",
    f"Conventional local linear, h={ref_bw:.2f}",
    axis_block_name="design",
    axis_block={"spec_id": "design/regression_discontinuity/procedure/conventional",
                "procedure": "conventional"})

# Robust bias-corrected: use local quadratic with smaller bandwidth
# Following Calonico, Cattaneo, Titiunik (2014) approximation:
# use p+1 polynomial and scale bandwidth
rbc_bw = ref_bw  # same bandwidth but higher polynomial for bias correction
run_rd_spec(
    "design/regression_discontinuity/procedure/robust_bias_corrected",
    "designs/regression_discontinuity.md#procedure",
    "G1", ref_outcome, collapsed_wide,
    rbc_bw, triangular_kernel, 2, CUTOFF,  # p=2 for bias correction of p=1
    f"Year {REF_YEAR}, robust bias-corrected (p=2)",
    f"Bias-corrected (local quadratic), h={rbc_bw:.2f}",
    axis_block_name="design",
    axis_block={"spec_id": "design/regression_discontinuity/procedure/robust_bias_corrected",
                "procedure": "robust_bias_corrected",
                "notes": "Approximated using local quadratic at same bandwidth"})


# ============================================================
# RC: DONUT HOLE EXCLUSIONS
# ============================================================

print("\n=== Running RC: Donut Hole Exclusions ===")

for donut_size in [1, 2, 3]:
    spec_id = f"rc/sample/donut/exclude_{donut_size}"
    run_rd_spec(
        spec_id,
        "modules/robustness/sample.md#donut-hole",
        "G1", ref_outcome, collapsed_wide,
        ref_bw, triangular_kernel, 1, CUTOFF,
        f"Year {REF_YEAR}, donut={donut_size}",
        f"Local linear, triangular, h={ref_bw:.2f}, donut={donut_size}",
        donut=donut_size,
        axis_block_name="sample",
        axis_block={"spec_id": spec_id, "donut": donut_size})


# ============================================================
# RC: BANDWIDTH VARIANTS (multiples of baseline)
# ============================================================

print("\n=== Running RC: Bandwidth Variants ===")

bw_multiples = {
    "rc/sample/bandwidth/bw_50pct": 0.50,
    "rc/sample/bandwidth/bw_75pct": 0.75,
    "rc/sample/bandwidth/bw_125pct": 1.25,
    "rc/sample/bandwidth/bw_150pct": 1.50,
    "rc/sample/bandwidth/bw_200pct": 2.00,
    "rc/sample/bandwidth/bw_300pct": 3.00,
}

for spec_id, mult in bw_multiples.items():
    bw_val = ref_bw * mult
    run_rd_spec(
        spec_id,
        "modules/robustness/sample.md#bandwidth",
        "G1", ref_outcome, collapsed_wide,
        bw_val, triangular_kernel, 1, CUTOFF,
        f"Year {REF_YEAR}, h={bw_val:.2f} ({mult*100:.0f}% of baseline)",
        f"Local linear, triangular, h={bw_val:.2f}",
        axis_block_name="sample",
        axis_block={"spec_id": spec_id, "bandwidth_multiplier": mult,
                    "bandwidth": bw_val})


# ============================================================
# RC: SAMPLE RESTRICTIONS (SES strata)
# ============================================================

print("\n=== Running RC: Sample Restrictions (SES Strata) ===")

# For SES-restricted samples, we need to re-collapse from raw data
def build_collapsed_for_subset(raw_df, subset_mask, subset_desc):
    """Re-collapse raw data for a subset."""
    sub = raw_df[subset_mask].copy()
    if len(sub) == 0:
        return None

    sub['survey'] = 1
    coll = sub.groupby(['score', 'doi_yr'])['survey'].sum().reset_index()
    coll_wide = coll.pivot(index='score', columns='doi_yr', values='survey').fillna(0)
    coll_wide.columns = [f'survey{int(c)}' for c in coll_wide.columns]
    coll_wide = coll_wide.reset_index()

    for yr in range(1994, 2004):
        col = f'survey{yr}'
        pct_col = f'sisses{yr}'
        if col in coll_wide.columns:
            total = coll_wide[col].sum()
            if total > 0:
                coll_wide[pct_col] = (coll_wide[col] / total) * 100.0
            else:
                coll_wide[pct_col] = 0.0

    coll_wide['align'] = coll_wide['score'] - CUTOFF
    coll_wide['elig'] = (coll_wide['score'] <= CUTOFF).astype(int)
    coll_wide['jump'] = coll_wide['elig']
    coll_wide['jump_align'] = coll_wide['jump'] * coll_wide['align']

    for p in range(2, 9):
        coll_wide[f'align{p}'] = coll_wide['align'] ** p
        coll_wide[f'align{p}_jump'] = coll_wide[f'align{p}'] * coll_wide['jump']

    year_cols = [f'sisses{yr}' for yr in range(1994, 2004) if f'sisses{yr}' in coll_wide.columns]
    if year_cols:
        coll_wide['sisses_pooled'] = coll_wide[year_cols].mean(axis=1)

    return coll_wide


# SES 1 only
coll_ses1 = build_collapsed_for_subset(df_raw, df_raw['ses'] == 1, "SES 1 only")
if coll_ses1 is not None and ref_outcome in coll_ses1.columns:
    run_rd_spec(
        "rc/sample/restrict/ses_1_only",
        "modules/robustness/sample.md#subgroup",
        "G1", ref_outcome, coll_ses1,
        ref_bw, triangular_kernel, 1, CUTOFF,
        f"SES 1 only, year {REF_YEAR}",
        f"Local linear, triangular, h={ref_bw:.2f}, SES 1 only",
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/restrict/ses_1_only", "restriction": "ses==1"})

# SES 1-2 only
coll_ses12 = build_collapsed_for_subset(df_raw, df_raw['ses'].isin([1, 2]), "SES 1-2 only")
if coll_ses12 is not None and ref_outcome in coll_ses12.columns:
    run_rd_spec(
        "rc/sample/restrict/ses_1_2_only",
        "modules/robustness/sample.md#subgroup",
        "G1", ref_outcome, coll_ses12,
        ref_bw, triangular_kernel, 1, CUTOFF,
        f"SES 1-2 only, year {REF_YEAR}",
        f"Local linear, triangular, h={ref_bw:.2f}, SES 1-2 only",
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/restrict/ses_1_2_only", "restriction": "ses in [1,2]"})

# SES 2-3 only
coll_ses23 = build_collapsed_for_subset(df_raw, df_raw['ses'].isin([2, 3]), "SES 2-3 only")
if coll_ses23 is not None and ref_outcome in coll_ses23.columns:
    run_rd_spec(
        "rc/sample/restrict/ses_2_3_only",
        "modules/robustness/sample.md#subgroup",
        "G1", ref_outcome, coll_ses23,
        ref_bw, triangular_kernel, 1, CUTOFF,
        f"SES 2-3 only, year {REF_YEAR}",
        f"Local linear, triangular, h={ref_bw:.2f}, SES 2-3 only",
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/restrict/ses_2_3_only", "restriction": "ses in [2,3]"})


# ============================================================
# RC: TIME PERIOD RESTRICTIONS
# ============================================================

print("\n=== Running RC: Time Period Restrictions ===")

# Post-1998 only (when manipulation intensified after algorithm disclosure)
coll_post98 = build_collapsed_for_subset(df_raw, df_raw['doi_yr'] >= 1998, "Post-1998 only")
if coll_post98 is not None:
    # Use pooled post-1998
    post98_cols = [f'sisses{yr}' for yr in range(1998, 2004) if f'sisses{yr}' in coll_post98.columns]
    if post98_cols:
        coll_post98['sisses_post98'] = coll_post98[post98_cols].mean(axis=1)
        run_rd_spec(
            "rc/sample/restrict/post1998_only",
            "modules/robustness/sample.md#subgroup",
            "G1", "sisses_post98", coll_post98,
            bw_pooled, triangular_kernel, 1, CUTOFF,
            "Post-1998 only (pooled)",
            f"Local linear, triangular, h={bw_pooled:.2f}, post-1998",
            axis_block_name="sample",
            axis_block={"spec_id": "rc/sample/restrict/post1998_only",
                        "restriction": "year >= 1998"})

# Pre-1998 only
coll_pre98 = build_collapsed_for_subset(df_raw, df_raw['doi_yr'] < 1998, "Pre-1998 only")
if coll_pre98 is not None:
    pre98_cols = [f'sisses{yr}' for yr in range(1994, 1998) if f'sisses{yr}' in coll_pre98.columns]
    if pre98_cols:
        coll_pre98['sisses_pre98'] = coll_pre98[pre98_cols].mean(axis=1)
        run_rd_spec(
            "rc/sample/restrict/pre1998_only",
            "modules/robustness/sample.md#subgroup",
            "G1", "sisses_pre98", coll_pre98,
            bw_pooled, triangular_kernel, 1, CUTOFF,
            "Pre-1998 only (pooled)",
            f"Local linear, triangular, h={bw_pooled:.2f}, pre-1998",
            axis_block_name="sample",
            axis_block={"spec_id": "rc/sample/restrict/pre1998_only",
                        "restriction": "year < 1998"})


# ============================================================
# RC: INDIVIDUAL YEAR ESTIMATES (from surface)
# ============================================================

print("\n=== Running RC: Individual Year Estimates ===")

for yr in range(1994, 2004):
    outcome_col = f'sisses{yr}'
    if outcome_col not in collapsed_wide.columns:
        continue
    bw_yr = bandwidths.get(yr, bw_pooled)
    spec_id = f"rc/sample/restrict/year_{yr}"

    run_rd_spec(
        spec_id,
        "modules/robustness/sample.md#subgroup",
        "G1", outcome_col, collapsed_wide,
        bw_yr, triangular_kernel, 1, CUTOFF,
        f"Year {yr}, IK bandwidth={bw_yr:.2f}",
        f"Local linear, triangular, h={bw_yr:.2f}",
        axis_block_name="sample",
        axis_block={"spec_id": spec_id, "year": yr})


# ============================================================
# RC: MAY-OCTOBER ONLY (survey timing restriction)
# ============================================================

print("\n=== Running RC: May-October Only ===")

df_raw['doi_month'] = pd.to_datetime(df_raw['fencuesta']).dt.month
coll_mayoct = build_collapsed_for_subset(
    df_raw, (df_raw['doi_month'] >= 5) & (df_raw['doi_month'] <= 10),
    "May-October only"
)
if coll_mayoct is not None and ref_outcome in coll_mayoct.columns:
    run_rd_spec(
        "rc/sample/restrict/may_oct_only",
        "modules/robustness/sample.md#subgroup",
        "G1", ref_outcome, coll_mayoct,
        ref_bw, triangular_kernel, 1, CUTOFF,
        f"May-October surveys only, year {REF_YEAR}",
        f"Local linear, triangular, h={ref_bw:.2f}, May-Oct only",
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/restrict/may_oct_only",
                    "restriction": "month in [5,10]"})


# ============================================================
# RC: DATA CONSTRUCTION VARIANTS (score rounding)
# ============================================================

print("\n=== Running RC: Score Construction Variants ===")

# Floor score (round down to nearest integer -- scores may have decimals)
df_floor = df_raw.copy()
df_floor['score'] = np.floor(df_floor['score']).astype(int)
coll_floor = build_collapsed_for_subset(df_floor, pd.Series(True, index=df_floor.index), "Floor score")
if coll_floor is not None and ref_outcome in coll_floor.columns:
    run_rd_spec(
        "rc/data/score_construction/floor_score",
        "modules/robustness/data_construction.md",
        "G1", ref_outcome, coll_floor,
        ref_bw, triangular_kernel, 1, CUTOFF,
        f"Floor(score), year {REF_YEAR}",
        f"Local linear, triangular, h={ref_bw:.2f}, floor(score)",
        axis_block_name="data_construction",
        axis_block={"spec_id": "rc/data/score_construction/floor_score",
                    "score_transform": "floor"})

# Round score
df_round = df_raw.copy()
df_round['score'] = np.round(df_round['score']).astype(int)
coll_round = build_collapsed_for_subset(df_round, pd.Series(True, index=df_round.index), "Round score")
if coll_round is not None and ref_outcome in coll_round.columns:
    run_rd_spec(
        "rc/data/score_construction/round_score",
        "modules/robustness/data_construction.md",
        "G1", ref_outcome, coll_round,
        ref_bw, triangular_kernel, 1, CUTOFF,
        f"Round(score), year {REF_YEAR}",
        f"Local linear, triangular, h={ref_bw:.2f}, round(score)",
        axis_block_name="data_construction",
        axis_block={"spec_id": "rc/data/score_construction/round_score",
                    "score_transform": "round"})


# ============================================================
# RC: PLACEBO CUTOFFS
# ============================================================

print("\n=== Running RC: Placebo Cutoffs ===")

placebo_cutoffs = {
    "rc/sample/placebo_cutoff/cutoff_35": 35,
    "rc/sample/placebo_cutoff/cutoff_40": 40,
    "rc/sample/placebo_cutoff/cutoff_42": 42,
    "rc/sample/placebo_cutoff/cutoff_44": 44,
    "rc/sample/placebo_cutoff/cutoff_50": 50,
    "rc/sample/placebo_cutoff/cutoff_52": 52,
    "rc/sample/placebo_cutoff/cutoff_55": 55,
    "rc/sample/placebo_cutoff/cutoff_60": 60,
}

for spec_id, placebo_c in placebo_cutoffs.items():
    # Rebuild RD variables with new cutoff
    df_placebo = collapsed_wide.copy()
    df_placebo['align'] = df_placebo['score'] - placebo_c
    df_placebo['jump'] = (df_placebo['score'] <= placebo_c).astype(int)
    df_placebo['jump_align'] = df_placebo['jump'] * df_placebo['align']
    for p in range(2, 9):
        df_placebo[f'align{p}'] = df_placebo['align'] ** p
        df_placebo[f'align{p}_jump'] = df_placebo[f'align{p}'] * df_placebo['jump']

    run_rd_spec(
        spec_id,
        "modules/robustness/sample.md#placebo-cutoff",
        "G1", ref_outcome, df_placebo,
        ref_bw, triangular_kernel, 1, placebo_c,
        f"Placebo cutoff at {placebo_c}, year {REF_YEAR}",
        f"Local linear, triangular, h={ref_bw:.2f}, cutoff={placebo_c}",
        axis_block_name="sample",
        axis_block={"spec_id": spec_id, "placebo_cutoff": placebo_c,
                    "true_cutoff": CUTOFF})


# ============================================================
# INFERENCE VARIANTS
# ============================================================

print("\n=== Running Inference Variants ===")

# Use the first baseline (2003 year) as the reference for inference
baseline_run_id = f"{PAPER_ID}_run_004"  # baseline__2003
infer_counter = 0

def run_inference_rd(base_run_id, spec_id, outcome_col, data, bandwidth,
                     kernel_func, poly_order, cutoff_val, vcov_type, vcov_desc):
    """Re-run baseline RD with different variance-covariance estimator."""
    global infer_counter
    infer_counter += 1
    infer_run_id = f"{PAPER_ID}_infer_{infer_counter:03d}"

    try:
        df = data.copy()
        h = bandwidth
        df['k_weight'] = kernel_func(df['align'].values, h)
        df = df[df['k_weight'] > 0].copy()

        if len(df) < 5:
            raise ValueError(f"Too few observations (N={len(df)})")

        treatment_var = "jump"
        controls = ["align", "jump_align"]
        if poly_order >= 2:
            controls.extend(["align2", "align2_jump"])
        if poly_order >= 3:
            controls.extend(["align3", "align3_jump"])

        controls_str = " + ".join(controls)
        formula = f"{outcome_col} ~ {treatment_var} + {controls_str}"

        if vcov_type == "HC1":
            vcov = "hetero"
        elif vcov_type == "cluster_score":
            df['score_str'] = df['score'].astype(str)
            vcov = {"CRV1": "score_str"}
        elif vcov_type == "iid":
            vcov = "iid"
        else:
            vcov = "hetero"

        m = pf.feols(formula, data=df, vcov=vcov, weights="k_weight")

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
            "spec_tree_path": "modules/inference/standard_errors.md",
            "baseline_group_id": "G1",
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
            "spec_tree_path": "modules/inference/standard_errors.md",
            "baseline_group_id": "G1",
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


# HC1 (canonical -- already in baseline)
run_inference_rd(
    baseline_run_id, "infer/se/hc/hc1",
    ref_outcome, collapsed_wide, ref_bw,
    triangular_kernel, 1, CUTOFF,
    "HC1", "HC1 (robust)")

# Cluster at score level
run_inference_rd(
    baseline_run_id, "infer/se/cluster/score",
    ref_outcome, collapsed_wide, ref_bw,
    triangular_kernel, 1, CUTOFF,
    "cluster_score", "cluster(score)")

# IID (homoskedastic)
run_inference_rd(
    baseline_run_id, "infer/se/iid",
    ref_outcome, collapsed_wide, ref_bw,
    triangular_kernel, 1, CUTOFF,
    "iid", "IID (homoskedastic)")


# ============================================================
# WRITE OUTPUTS
# ============================================================

print(f"\n=== Writing Outputs ===")
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
    print("\nFailed specs:")
    for _, row in failed.iterrows():
        print(f"  {row['spec_id']}: {row['run_error']}")

if len(successful) > 0:
    # Find 2003 baseline
    base_row = spec_df[spec_df['spec_id'] == 'baseline__2003']
    if len(base_row) == 0:
        base_row = spec_df[spec_df['spec_id'] == 'baseline__pooled']

    if len(base_row) > 0:
        print(f"\nBaseline coef on jump: {base_row['coefficient'].values[0]:.6f}")
        print(f"Baseline SE: {base_row['std_error'].values[0]:.6f}")
        print(f"Baseline p-value: {base_row['p_value'].values[0]:.6f}")
        print(f"Baseline N: {base_row['n_obs'].values[0]:.0f}")

    # Exclude placebo cutoffs from main summary (they should be near zero)
    non_placebo = successful[~successful['spec_id'].str.contains('placebo')]

    print(f"\n=== COEFFICIENT RANGE (non-placebo specs) ===")
    print(f"Min coef: {non_placebo['coefficient'].min():.6f}")
    print(f"Max coef: {non_placebo['coefficient'].max():.6f}")
    print(f"Median coef: {non_placebo['coefficient'].median():.6f}")
    n_sig = (non_placebo['p_value'] < 0.05).sum()
    print(f"Significant at 5%: {n_sig}/{len(non_placebo)}")
    n_sig10 = (non_placebo['p_value'] < 0.10).sum()
    print(f"Significant at 10%: {n_sig10}/{len(non_placebo)}")


# ============================================================
# SPECIFICATION_SEARCH.md
# ============================================================

print("\nWriting SPECIFICATION_SEARCH.md...")

md_lines = []
md_lines.append("# Specification Search Report: 114759-V1")
md_lines.append("")
md_lines.append("**Paper:** Camacho & Conover (2011), \"Manipulation of Social Program Eligibility\", AEJ: Economic Policy 3(2)")
md_lines.append("")
md_lines.append("## Baseline Specification")
md_lines.append("")
md_lines.append("- **Design:** Sharp Regression Discontinuity")
md_lines.append("- **Running variable:** SISBEN poverty score (puntaje)")
md_lines.append(f"- **Cutoff:** {CUTOFF}")
md_lines.append("- **Outcome:** Share of surveys at each score (density)")
md_lines.append("- **Treatment:** jump (score <= 47 indicator)")
md_lines.append("- **Estimator:** Local linear regression with triangular kernel")
md_lines.append("- **Bandwidth:** Data-driven (Imbens-Kalyanaraman)")
md_lines.append("- **Inference:** HC1 robust SEs")
md_lines.append("")

if len(successful) > 0:
    base_row = spec_df[spec_df['spec_id'] == 'baseline__2003']
    if len(base_row) == 0:
        base_row = spec_df[spec_df['spec_id'] == 'baseline__pooled']

    if len(base_row) > 0:
        bc = base_row.iloc[0]
        md_lines.append(f"| Statistic | Value |")
        md_lines.append(f"|-----------|-------|")
        md_lines.append(f"| Coefficient (jump) | {bc['coefficient']:.6f} |")
        md_lines.append(f"| Std. Error | {bc['std_error']:.6f} |")
        md_lines.append(f"| p-value | {bc['p_value']:.6f} |")
        md_lines.append(f"| 95% CI | [{bc['ci_lower']:.6f}, {bc['ci_upper']:.6f}] |")
        md_lines.append(f"| N (score bins in BW) | {bc['n_obs']:.0f} |")
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
    "Baseline (year-specific)": successful[successful['spec_id'].str.startswith('baseline__')],
    "Design: Bandwidth": successful[successful['spec_id'].str.contains('bandwidth')],
    "Design: Polynomial": successful[successful['spec_id'].str.contains('poly/')],
    "Design: Kernel": successful[successful['spec_id'].str.contains('kernel/')],
    "Design: Procedure": successful[successful['spec_id'].str.contains('procedure/')],
    "RC: Donut Hole": successful[successful['spec_id'].str.contains('donut/')],
    "RC: Bandwidth Multiples": successful[successful['spec_id'].str.startswith('rc/sample/bandwidth/')],
    "RC: SES Restrictions": successful[successful['spec_id'].str.contains('ses_')],
    "RC: Time Period": successful[successful['spec_id'].str.contains('1998|may_oct|year_')],
    "RC: Score Construction": successful[successful['spec_id'].str.contains('score_construction')],
    "RC: Placebo Cutoffs": successful[successful['spec_id'].str.contains('placebo')],
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
    non_placebo = successful[~successful['spec_id'].str.contains('placebo')]
    placebo = successful[successful['spec_id'].str.contains('placebo')]

    n_sig_total = (non_placebo['p_value'] < 0.05).sum()
    pct_sig = n_sig_total / len(non_placebo) * 100 if len(non_placebo) > 0 else 0
    sign_consistent = ((non_placebo['coefficient'] > 0).sum() == len(non_placebo)) or \
                      ((non_placebo['coefficient'] < 0).sum() == len(non_placebo))
    median_coef = non_placebo['coefficient'].median()
    sign_word = "positive" if median_coef > 0 else "negative"

    md_lines.append(f"- **Sign consistency:** {'All non-placebo specifications have the same sign' if sign_consistent else 'Mixed signs across non-placebo specifications'}")
    md_lines.append(f"- **Significance stability:** {n_sig_total}/{len(non_placebo)} ({pct_sig:.1f}%) non-placebo specifications significant at 5%")
    md_lines.append(f"- **Direction:** Median coefficient is {sign_word} ({median_coef:.6f}), indicating {'upward bunching' if median_coef > 0 else 'downward bunching'} below cutoff")

    if len(placebo) > 0:
        placebo_sig = (placebo['p_value'] < 0.05).sum()
        md_lines.append(f"- **Placebo cutoffs:** {placebo_sig}/{len(placebo)} significant at 5% (lower is better)")

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
