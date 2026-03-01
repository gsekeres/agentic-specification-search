"""
Specification Search Script for Romer & Romer (2017)
"New Evidence on the Aftermath of Financial Crises in Advanced Countries"
American Economic Review, 107(10), 3072-3118.

Paper ID: 113046-V1

Surface-driven execution:
  - G1: Jorda local projection impulse response
  - lnGDP(t+h) ~ CRISIS(t) + CRISIS_lags + lnGDP_lags + country_FE + time_FE
  - Panel of 24 advanced economies, semiannual data 1967-2012
  - 50+ specifications across horizons, outcomes, samples, FE, lag structures,
    weighting, alternative crisis measures, SE variants

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

PAPER_ID = "113046-V1"
DATA_DIR = "data/downloads/extracted/113046-V1"
OUTPUT_DIR = DATA_DIR
EXCEL_PATH = f"{DATA_DIR}/20150320_data/Data-and-Programs/Romer-RomerCrisesData.xlsx"

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

bg = surface_obj["baseline_groups"][0]
design_audit = bg["design_audit"]
inference_canonical = bg["inference_plan"]["canonical"]

# ============================================================
# Data Loading and Panel Construction
# ============================================================

print("Loading data from Excel...")
xlsx = pd.ExcelFile(EXCEL_PATH)

# --- 1. Financial distress (Romer & Romer measure) ---
crisis_raw = xlsx.parse('New Meas. of Financial Distress')
# Date column is like "1967:1", "1967:2"
# 24 country columns

COUNTRIES = [
    'Australia', 'Austria', 'Belgium', 'Canada', 'Denmark', 'Finland',
    'France', 'Germany', 'Greece', 'Iceland', 'Ireland', 'Italy',
    'Japan', 'Luxembourg', 'Netherlands', 'NewZealand', 'Norway',
    'Portugal', 'Spain', 'Sweden', 'Switzerland', 'Turkey',
    'UnitedKingdom', 'UnitedStates'
]
COUNTRY_ID = {c: i+1 for i, c in enumerate(COUNTRIES)}

crisis_long = crisis_raw.melt(id_vars='Date', value_vars=COUNTRIES,
                               var_name='country', value_name='CRISIS')
crisis_long = crisis_long.rename(columns={'Date': 'date'})

# --- 2. Semiannual macro data (GDP, IP, Unemployment stacked) ---
macro_raw = xlsx.parse('Semiannual Macro Data')

# Find block boundaries by searching for label rows
gdp_start = None
ip_start = None
unemp_start = None
for i, row in macro_raw.iterrows():
    if isinstance(row['Date'], str):
        if row['Date'] == 'GDP':
            gdp_start = i
        elif row['Date'] == 'IP':
            ip_start = i
        elif row['Date'] == 'UNEMP':
            unemp_start = i

# Parse each block: skip 2 rows after label (label + blank), read until next label
def parse_macro_block(df, start_idx, end_idx):
    """Extract a macro data block from the stacked semiannual data."""
    block = df.iloc[start_idx+2:end_idx].copy()
    block = block.dropna(subset=['Date'])
    block = block[block['Date'].apply(lambda x: isinstance(x, str) and ':' in str(x))]
    return block

gdp_block = parse_macro_block(macro_raw, gdp_start, ip_start)
ip_block = parse_macro_block(macro_raw, ip_start, unemp_start)
unemp_block = parse_macro_block(macro_raw, unemp_start, len(macro_raw))

# Melt each block to long format
def melt_macro(block, varname):
    long = block.melt(id_vars='Date', value_vars=COUNTRIES,
                      var_name='country', value_name=varname)
    long = long.rename(columns={'Date': 'date'})
    long[varname] = pd.to_numeric(long[varname], errors='coerce')
    return long

gdp_long = melt_macro(gdp_block, 'GDP')
ip_long = melt_macro(ip_block, 'IP')
unemp_long = melt_macro(unemp_block, 'UNEMP')

# --- 3. Merge into panel ---
panel = crisis_long.merge(gdp_long, on=['date', 'country'], how='outer')
panel = panel.merge(ip_long, on=['date', 'country'], how='outer')
panel = panel.merge(unemp_long, on=['date', 'country'], how='outer')

# Create log transformations (following RATS code: LOG(GDP)*100)
panel['lnGDP'] = np.log(panel['GDP']) * 100.0
panel['lnIP'] = np.log(panel['IP']) * 100.0

# Create country_id and period_id for FE
panel['country_id'] = panel['country'].map(COUNTRY_ID)

# Parse date into year and half
def parse_date(d):
    if not isinstance(d, str) or ':' not in d:
        return np.nan, np.nan
    parts = d.split(':')
    return int(parts[0]), int(parts[1])

panel[['year', 'half']] = panel['date'].apply(lambda d: pd.Series(parse_date(d)))
panel = panel.dropna(subset=['year', 'half'])
panel['year'] = panel['year'].astype(int)
panel['half'] = panel['half'].astype(int)

# Create a numeric time index (for sorting and lag construction)
panel['time_idx'] = (panel['year'] - 1960) * 2 + panel['half']

# Sort for proper lag construction
panel = panel.sort_values(['country_id', 'time_idx']).reset_index(drop=True)

# Make FE vars strings for pyfixest
panel['country_str'] = panel['country_id'].astype(str)
panel['time_str'] = panel['time_idx'].astype(str)

print(f"Panel shape: {panel.shape}")
print(f"Countries: {panel['country'].nunique()}")
print(f"Time periods: {panel['time_idx'].nunique()}")
print(f"Date range: {panel['date'].min()} to {panel['date'].max()}")

# --- 4. Construct leads and lags ---
# For each country, compute lags of CRISIS and output, and leads of output (for horizons)

for lag in range(1, 7):
    panel[f'CRISIS_L{lag}'] = panel.groupby('country')['CRISIS'].shift(lag)
    panel[f'lnGDP_L{lag}'] = panel.groupby('country')['lnGDP'].shift(lag)
    panel[f'lnIP_L{lag}'] = panel.groupby('country')['lnIP'].shift(lag)
    panel[f'UNEMP_L{lag}'] = panel.groupby('country')['UNEMP'].shift(lag)

# Leads for dependent variable at horizons h=0,...,10
for h in range(0, 11):
    panel[f'lnGDP_h{h}'] = panel.groupby('country')['lnGDP'].shift(-h)
    panel[f'lnIP_h{h}'] = panel.groupby('country')['lnIP'].shift(-h)
    panel[f'UNEMP_h{h}'] = panel.groupby('country')['UNEMP'].shift(-h)

# --- 5. Alternative crisis chronologies ---
rr_sys = xlsx.parse('Reinhart and Rogoff (Systemic)')
rr_all = xlsx.parse('Reinhart and Rogoff (All)')
imf_sys = xlsx.parse('IMF (Systemic)')
imf_all_data = xlsx.parse('IMF (All)')

for name, alt_df in [('CRISIS_RR_SYS', rr_sys), ('CRISIS_RR_ALL', rr_all),
                      ('CRISIS_IMF_SYS', imf_sys), ('CRISIS_IMF_ALL', imf_all_data)]:
    alt_long = alt_df.melt(id_vars='Date', value_vars=COUNTRIES,
                            var_name='country', value_name=name)
    alt_long = alt_long.rename(columns={'Date': 'date'})
    alt_long[name] = pd.to_numeric(alt_long[name], errors='coerce')
    panel = panel.merge(alt_long, on=['date', 'country'], how='left')

# Construct lags for alternative crisis measures
for alt_name in ['CRISIS_RR_SYS', 'CRISIS_RR_ALL', 'CRISIS_IMF_SYS', 'CRISIS_IMF_ALL']:
    for lag in range(1, 5):
        panel[f'{alt_name}_L{lag}'] = panel.groupby('country')[alt_name].shift(lag)

# --- 6. Sample definitions ---
panel['is_pre_gfc'] = (panel['year'] < 2007).astype(int)
panel['is_gfc'] = ((panel['year'] >= 2007)).astype(int)

print(f"Panel after all transformations: {panel.shape}")

# ============================================================
# Accumulators
# ============================================================

results = []
inference_results = []
spec_run_counter = 0


# ============================================================
# Helper: run_spec (Local Projection OLS with FE via pyfixest)
# ============================================================

def run_spec(spec_id, spec_tree_path, baseline_group_id,
             outcome_var, treatment_var, controls, fe_formula_str,
             fe_desc, data, vcov, sample_desc, controls_desc,
             axis_block_name=None, axis_block=None, notes=""):
    """Run a single Jorda local projection specification and record results."""
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
                       "method": "ols_conventional"},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"local_projection": design_audit},
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
            "fixed_effects": fe_desc,
            "controls_desc": controls_desc,
            "cluster_var": "",
            "run_success": 0,
            "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


# ============================================================
# Define standard control sets
# ============================================================

# Baseline: 4 crisis lags + 4 output lags (output lags depend on outcome)
CRISIS_LAGS_4 = ["CRISIS_L1", "CRISIS_L2", "CRISIS_L3", "CRISIS_L4"]
GDP_LAGS_4 = ["lnGDP_L1", "lnGDP_L2", "lnGDP_L3", "lnGDP_L4"]
IP_LAGS_4 = ["lnIP_L1", "lnIP_L2", "lnIP_L3", "lnIP_L4"]
UNEMP_LAGS_4 = ["UNEMP_L1", "UNEMP_L2", "UNEMP_L3", "UNEMP_L4"]

BASELINE_CONTROLS_GDP = CRISIS_LAGS_4 + GDP_LAGS_4
BASELINE_FE = "country_str + time_str"

# Prepare baseline estimation sample (drop missing on key vars)
est_vars = ['CRISIS', 'lnGDP_h0', 'lnGDP'] + CRISIS_LAGS_4 + GDP_LAGS_4 + \
           ['country_str', 'time_str']
df_base = panel.dropna(subset=est_vars).copy()
print(f"\nBaseline estimation sample: {len(df_base)} obs")
print(f"  Countries: {df_base['country'].nunique()}")
print(f"  Time periods: {df_base['time_str'].nunique()}")


# ============================================================
# BASELINE: GDP at horizon 0 (Table 1, Line 1, Figure 4)
# ============================================================

print("\n=== Running baseline specification (GDP, h=0) ===")
base_run_id, base_coef, base_se, base_pval, base_nobs = run_spec(
    "baseline__GDP_h0", "designs/local_projection.md#baseline", "G1",
    "lnGDP_h0", "CRISIS", BASELINE_CONTROLS_GDP,
    BASELINE_FE, "country + time", df_base,
    "iid",
    "Full sample, 24 countries, 1967-2012", "4 crisis lags + 4 GDP lags")

print(f"  Baseline: coef={base_coef:.4f}, se={base_se:.4f}, p={base_pval:.4f}, N={base_nobs}")


# ============================================================
# RC: ALTERNATIVE OUTCOMES (IP, Unemployment) at h=0
# ============================================================

print("\n=== Running alternative outcome specifications ===")

# IP
ip_est_vars = ['CRISIS', 'lnIP_h0', 'lnIP'] + CRISIS_LAGS_4 + IP_LAGS_4 + \
              ['country_str', 'time_str']
df_ip = panel.dropna(subset=ip_est_vars).copy()
run_spec(
    "rc/outcome/IP_h0", "modules/robustness/outcome.md#alternative-outcomes", "G1",
    "lnIP_h0", "CRISIS", CRISIS_LAGS_4 + IP_LAGS_4,
    BASELINE_FE, "country + time", df_ip,
    "iid",
    "Full sample (IP available)", "4 crisis lags + 4 IP lags",
    axis_block_name="estimation",
    axis_block={"spec_id": "rc/outcome/IP_h0", "outcome": "lnIP", "horizon": 0})

# Unemployment
unemp_est_vars = ['CRISIS', 'UNEMP_h0', 'UNEMP'] + CRISIS_LAGS_4 + UNEMP_LAGS_4 + \
                 ['country_str', 'time_str']
df_unemp = panel.dropna(subset=unemp_est_vars).copy()
run_spec(
    "rc/outcome/UNEMP_h0", "modules/robustness/outcome.md#alternative-outcomes", "G1",
    "UNEMP_h0", "CRISIS", CRISIS_LAGS_4 + UNEMP_LAGS_4,
    BASELINE_FE, "country + time", df_unemp,
    "iid",
    "Full sample (Unemp available)", "4 crisis lags + 4 Unemp lags",
    axis_block_name="estimation",
    axis_block={"spec_id": "rc/outcome/UNEMP_h0", "outcome": "UNEMP", "horizon": 0})


# ============================================================
# RC: HORIZONS h=1,...,10 for GDP (impulse response profile)
# ============================================================

print("\n=== Running horizon specifications (GDP, h=1 to h=10) ===")

for h in [1, 2, 3, 4, 5, 7, 10]:
    outcome_h = f"lnGDP_h{h}"
    h_est_vars = ['CRISIS', outcome_h] + CRISIS_LAGS_4 + GDP_LAGS_4 + \
                 ['country_str', 'time_str']
    df_h = panel.dropna(subset=h_est_vars).copy()
    run_spec(
        f"rc/horizon/GDP_h{h}", "modules/robustness/horizon.md#impulse-response-horizons", "G1",
        outcome_h, "CRISIS", BASELINE_CONTROLS_GDP,
        BASELINE_FE, "country + time", df_h,
        "iid",
        f"Full sample, horizon h={h} ({h/2:.1f} years)", "4 crisis lags + 4 GDP lags",
        axis_block_name="estimation",
        axis_block={"spec_id": f"rc/horizon/GDP_h{h}", "outcome": "lnGDP",
                    "horizon": h, "horizon_years": h/2})


# ============================================================
# RC: SAMPLE RESTRICTIONS
# ============================================================

print("\n=== Running sample restriction specifications ===")

# Pre-GFC sample (before 2007)
df_pregfc = df_base[df_base['is_pre_gfc'] == 1].copy()
run_spec(
    "rc/sample/pre_gfc", "modules/robustness/sample.md#temporal-subsamples", "G1",
    "lnGDP_h0", "CRISIS", BASELINE_CONTROLS_GDP,
    BASELINE_FE, "country + time", df_pregfc,
    "iid",
    "Pre-GFC sample (before 2007)", "4 crisis lags + 4 GDP lags",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/pre_gfc", "restriction": "year < 2007"})

# No Japan
df_nojapan = df_base[df_base['country'] != 'Japan'].copy()
run_spec(
    "rc/sample/no_japan", "modules/robustness/sample.md#leave-one-out-sample", "G1",
    "lnGDP_h0", "CRISIS", BASELINE_CONTROLS_GDP,
    BASELINE_FE, "country + time", df_nojapan,
    "iid",
    "Full sample excluding Japan", "4 crisis lags + 4 GDP lags",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/no_japan", "dropped_country": "Japan"})

# No Greece
df_nogreece = df_base[df_base['country'] != 'Greece'].copy()
run_spec(
    "rc/sample/no_greece", "modules/robustness/sample.md#leave-one-out-sample", "G1",
    "lnGDP_h0", "CRISIS", BASELINE_CONTROLS_GDP,
    BASELINE_FE, "country + time", df_nogreece,
    "iid",
    "Full sample excluding Greece", "4 crisis lags + 4 GDP lags",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/no_greece", "dropped_country": "Greece"})

# Drop one country at a time (24 specifications)
print("  Running drop-one-country specifications...")
for cidx, cname in enumerate(COUNTRIES, 1):
    df_drop = df_base[df_base['country'] != cname].copy()
    run_spec(
        f"rc/sample/drop_country_{cidx:02d}",
        "modules/robustness/sample.md#leave-one-out-sample", "G1",
        "lnGDP_h0", "CRISIS", BASELINE_CONTROLS_GDP,
        BASELINE_FE, "country + time", df_drop,
        "iid",
        f"Full sample excluding {cname}", "4 crisis lags + 4 GDP lags",
        axis_block_name="sample",
        axis_block={"spec_id": f"rc/sample/drop_country_{cidx:02d}",
                    "dropped_country": cname, "country_id": cidx})


# ============================================================
# RC: FIXED EFFECTS STRUCTURE
# ============================================================

print("\n=== Running FE variant specifications ===")

# Country FE only (no time FE)
run_spec(
    "rc/fe/country_only", "modules/robustness/fixed_effects.md#dropping-fe", "G1",
    "lnGDP_h0", "CRISIS", BASELINE_CONTROLS_GDP,
    "country_str", "country only (no time FE)", df_base,
    "iid",
    "Full sample", "4 crisis lags + 4 GDP lags",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/country_only", "fe": ["country"],
                "dropped": ["time"]})

# Time FE only (no country FE)
run_spec(
    "rc/fe/time_only", "modules/robustness/fixed_effects.md#dropping-fe", "G1",
    "lnGDP_h0", "CRISIS", BASELINE_CONTROLS_GDP,
    "time_str", "time only (no country FE)", df_base,
    "iid",
    "Full sample", "4 crisis lags + 4 GDP lags",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/time_only", "fe": ["time"],
                "dropped": ["country"]})

# No FE (pooled OLS with constant)
run_spec(
    "rc/fe/no_fe", "modules/robustness/fixed_effects.md#dropping-fe", "G1",
    "lnGDP_h0", "CRISIS", BASELINE_CONTROLS_GDP,
    "", "none (pooled OLS)", df_base,
    "iid",
    "Full sample", "4 crisis lags + 4 GDP lags",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/no_fe", "fe": [],
                "dropped": ["country", "time"]})


# ============================================================
# RC: LAG STRUCTURE VARIATIONS
# ============================================================

print("\n=== Running lag structure specifications ===")

lag_configs = [
    ("rc/controls/lags/crisis_2_output_4", 2, 4),
    ("rc/controls/lags/crisis_6_output_4", 6, 4),
    ("rc/controls/lags/crisis_4_output_2", 4, 2),
    ("rc/controls/lags/crisis_4_output_6", 4, 6),
    ("rc/controls/lags/crisis_2_output_2", 2, 2),
]

for spec_id, n_crisis_lags, n_gdp_lags in lag_configs:
    crisis_lag_vars = [f"CRISIS_L{i}" for i in range(1, n_crisis_lags + 1)]
    gdp_lag_vars = [f"lnGDP_L{i}" for i in range(1, n_gdp_lags + 1)]
    ctrl = crisis_lag_vars + gdp_lag_vars

    # Need these lags to exist
    lag_est_vars = ['CRISIS', 'lnGDP_h0'] + ctrl + ['country_str', 'time_str']
    df_lags = panel.dropna(subset=lag_est_vars).copy()

    run_spec(
        spec_id, "modules/robustness/controls.md#lag-structure", "G1",
        "lnGDP_h0", "CRISIS", ctrl,
        BASELINE_FE, "country + time", df_lags,
        "iid",
        "Full sample", f"{n_crisis_lags} crisis lags + {n_gdp_lags} GDP lags",
        axis_block_name="controls",
        axis_block={"spec_id": spec_id, "n_crisis_lags": n_crisis_lags,
                    "n_gdp_lags": n_gdp_lags})


# ============================================================
# RC: WLS (Weighted Least Squares)
# ============================================================

print("\n=== Running WLS specification ===")

# Following the RATS code: compute country-specific residual variances from OLS,
# then use inverse variance as weights.
try:
    # First run OLS to get residuals
    ols_formula = f"lnGDP_h0 ~ CRISIS + {' + '.join(BASELINE_CONTROLS_GDP)} | {BASELINE_FE}"
    m_ols = pf.feols(ols_formula, data=df_base, vcov="iid")
    resids = m_ols.resid()

    # Compute country-specific variance
    df_base_wls = df_base.copy()
    df_base_wls['_resid'] = resids
    country_var = df_base_wls.groupby('country')['_resid'].var()

    # Inverse variance weights
    df_base_wls['_wt'] = df_base_wls['country'].map(lambda c: 1.0 / country_var.get(c, np.nan))
    df_base_wls = df_base_wls.dropna(subset=['_wt'])

    # pyfixest supports weights via `weights` argument
    wls_formula = f"lnGDP_h0 ~ CRISIS + {' + '.join(BASELINE_CONTROLS_GDP)} | {BASELINE_FE}"
    m_wls = pf.feols(wls_formula, data=df_base_wls, vcov="iid", weights="_wt")

    coef_val = float(m_wls.coef().get("CRISIS", np.nan))
    se_val = float(m_wls.se().get("CRISIS", np.nan))
    pval_val = float(m_wls.pvalue().get("CRISIS", np.nan))

    try:
        ci = m_wls.confint()
        ci_lower = float(ci.loc["CRISIS", ci.columns[0]])
        ci_upper = float(ci.loc["CRISIS", ci.columns[1]])
    except:
        ci_lower = ci_upper = np.nan

    nobs = int(m_wls._N)
    try:
        r2 = float(m_wls._r2)
    except:
        r2 = np.nan

    all_coefs = {k: float(v) for k, v in m_wls.coef().items()}
    payload = make_success_payload(
        coefficients=all_coefs,
        inference={"spec_id": inference_canonical["spec_id"], "method": "wls"},
        software=SW_BLOCK,
        surface_hash=SURFACE_HASH,
        design={"local_projection": design_audit},
        axis_block_name="weights",
        axis_block={"spec_id": "rc/weights/wls", "method": "wls",
                    "notes": "Inverse country-specific residual variance weights"},
    )

    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"
    results.append({
        "paper_id": PAPER_ID,
        "spec_run_id": run_id,
        "spec_id": "rc/weights/wls",
        "spec_tree_path": "modules/robustness/weights.md#wls",
        "baseline_group_id": "G1",
        "outcome_var": "lnGDP_h0",
        "treatment_var": "CRISIS",
        "coefficient": coef_val,
        "std_error": se_val,
        "p_value": pval_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_obs": nobs,
        "r_squared": r2,
        "coefficient_vector_json": json.dumps(payload),
        "sample_desc": "Full sample, WLS weights",
        "fixed_effects": "country + time",
        "controls_desc": "4 crisis lags + 4 GDP lags",
        "cluster_var": "",
        "run_success": 1,
        "run_error": ""
    })
    print(f"  WLS: coef={coef_val:.4f}, se={se_val:.4f}, p={pval_val:.4f}, N={nobs}")

except Exception as e:
    err_msg = str(e)[:240]
    err_details = error_details_from_exception(e, stage="wls_estimation")
    payload = make_failure_payload(
        error=err_msg, error_details=err_details,
        software=SW_BLOCK, surface_hash=SURFACE_HASH
    )
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"
    results.append({
        "paper_id": PAPER_ID,
        "spec_run_id": run_id,
        "spec_id": "rc/weights/wls",
        "spec_tree_path": "modules/robustness/weights.md#wls",
        "baseline_group_id": "G1",
        "outcome_var": "lnGDP_h0",
        "treatment_var": "CRISIS",
        "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
        "ci_lower": np.nan, "ci_upper": np.nan,
        "n_obs": np.nan, "r_squared": np.nan,
        "coefficient_vector_json": json.dumps(payload),
        "sample_desc": "Full sample, WLS weights",
        "fixed_effects": "country + time",
        "controls_desc": "4 crisis lags + 4 GDP lags",
        "cluster_var": "",
        "run_success": 0,
        "run_error": err_msg
    })
    print(f"  WLS FAILED: {err_msg}")


# ============================================================
# RC: ALTERNATIVE CRISIS MEASURES
# ============================================================

print("\n=== Running alternative crisis measure specifications ===")

alt_crisis_configs = [
    ("rc/treatment/rr_systemic", "CRISIS_RR_SYS", "Reinhart & Rogoff systemic"),
    ("rc/treatment/rr_all", "CRISIS_RR_ALL", "Reinhart & Rogoff all crises"),
    ("rc/treatment/imf_systemic", "CRISIS_IMF_SYS", "IMF systemic"),
    ("rc/treatment/imf_all", "CRISIS_IMF_ALL", "IMF all crises"),
]

for spec_id, alt_crisis_var, desc in alt_crisis_configs:
    alt_crisis_lags = [f"{alt_crisis_var}_L{i}" for i in range(1, 5)]
    alt_controls = alt_crisis_lags + GDP_LAGS_4
    alt_est_vars = [alt_crisis_var, 'lnGDP_h0'] + alt_controls + ['country_str', 'time_str']
    df_alt = panel.dropna(subset=alt_est_vars).copy()

    if len(df_alt) > 50:  # need reasonable sample
        run_spec(
            spec_id, "modules/robustness/treatment.md#alternative-measures", "G1",
            "lnGDP_h0", alt_crisis_var, alt_controls,
            BASELINE_FE, "country + time", df_alt,
            "iid",
            f"Full sample ({desc})", f"4 {desc} lags + 4 GDP lags",
            axis_block_name="estimation",
            axis_block={"spec_id": spec_id, "crisis_measure": alt_crisis_var,
                        "description": desc})
    else:
        print(f"  Skipping {spec_id}: only {len(df_alt)} obs")


# ============================================================
# INFERENCE VARIANTS (on baseline specification)
# ============================================================

print("\n=== Running inference variants ===")

infer_counter = 0
baseline_run_id = f"{PAPER_ID}_run_001"


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
            design={"local_projection": design_audit},
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
baseline_controls_str = " + ".join(BASELINE_CONTROLS_GDP)
baseline_formula = f"lnGDP_h0 ~ CRISIS + {baseline_controls_str}"

# HC1 robust (heteroskedasticity-consistent)
run_inference_variant(
    baseline_run_id, "infer/se/hc/robust",
    "modules/inference/standard_errors.md#heteroskedasticity-robust", "G1",
    baseline_formula, BASELINE_FE, df_base, "CRISIS",
    "hetero", "HC1 (robust)")

# Clustered by country
run_inference_variant(
    baseline_run_id, "infer/se/cluster/country",
    "modules/inference/standard_errors.md#clustering", "G1",
    baseline_formula, BASELINE_FE, df_base, "CRISIS",
    {"CRV1": "country_str"}, "cluster(country)")

# Clustered by time period
run_inference_variant(
    baseline_run_id, "infer/se/cluster/time",
    "modules/inference/standard_errors.md#clustering", "G1",
    baseline_formula, BASELINE_FE, df_base, "CRISIS",
    {"CRV1": "time_str"}, "cluster(time)")


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
    base_row = spec_df[spec_df['spec_id'] == 'baseline__GDP_h0']
    if len(base_row) > 0:
        print(f"\nBaseline coef on CRISIS: {base_row['coefficient'].values[0]:.6f}")
        print(f"Baseline SE: {base_row['std_error'].values[0]:.6f}")
        print(f"Baseline p-value: {base_row['p_value'].values[0]:.6f}")
        print(f"Baseline N: {base_row['n_obs'].values[0]:.0f}")

    print(f"\n=== COEFFICIENT RANGE (successful specs, same treatment) ===")
    # Only baseline crisis measure specs for coefficient range
    crisis_specs = successful[successful['treatment_var'] == 'CRISIS']
    if len(crisis_specs) > 0:
        print(f"Min coef: {crisis_specs['coefficient'].min():.6f}")
        print(f"Max coef: {crisis_specs['coefficient'].max():.6f}")
        print(f"Median coef: {crisis_specs['coefficient'].median():.6f}")
        n_sig = (crisis_specs['p_value'] < 0.05).sum()
        print(f"Significant at 5%: {n_sig}/{len(crisis_specs)}")
        n_sig10 = (crisis_specs['p_value'] < 0.10).sum()
        print(f"Significant at 10%: {n_sig10}/{len(crisis_specs)}")


# ============================================================
# SPECIFICATION_SEARCH.md
# ============================================================

print("\nWriting SPECIFICATION_SEARCH.md...")

md_lines = []
md_lines.append("# Specification Search Report: 113046-V1")
md_lines.append("")
md_lines.append("**Paper:** Romer & Romer (2017), \"New Evidence on the Aftermath of Financial Crises in Advanced Countries\", AER 107(10)")
md_lines.append("")
md_lines.append("## Baseline Specification")
md_lines.append("")
md_lines.append("- **Design:** Jorda local projection impulse response")
md_lines.append("- **Outcome:** log(GDP) x 100 at horizon h=0")
md_lines.append("- **Treatment:** CRISIS (new semi-annual measure of financial distress, 0-15 scale)")
md_lines.append("- **Controls:** 4 crisis lags + 4 GDP lags")
md_lines.append("- **Fixed effects:** Country + Time (semiannual)")
md_lines.append("- **Standard errors:** Conventional OLS")
md_lines.append("- **Panel:** 24 OECD countries, semiannual 1967-2012")
md_lines.append("")

if len(successful) > 0:
    base_row = spec_df[spec_df['spec_id'] == 'baseline__GDP_h0']
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
    "Alt. Outcomes": successful[successful['spec_id'].str.startswith('rc/outcome/')],
    "Horizons": successful[successful['spec_id'].str.startswith('rc/horizon/')],
    "Sample (Pre-GFC/No-country)": successful[successful['spec_id'].str.startswith('rc/sample/')],
    "Fixed Effects": successful[successful['spec_id'].str.startswith('rc/fe/')],
    "Lag Structure": successful[successful['spec_id'].str.startswith('rc/controls/lags/')],
    "WLS": successful[successful['spec_id'].str.startswith('rc/weights/')],
    "Alt. Crisis Measures": successful[successful['spec_id'].str.startswith('rc/treatment/')],
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
    # Focus on CRISIS-measure specs for sign consistency
    crisis_specs = successful[successful['treatment_var'] == 'CRISIS']
    # Exclude unemployment (opposite sign expected)
    gdp_ip_specs = crisis_specs[~crisis_specs['outcome_var'].str.startswith('UNEMP')]

    if len(gdp_ip_specs) > 0:
        n_sig_total = (gdp_ip_specs['p_value'] < 0.05).sum()
        pct_sig = n_sig_total / len(gdp_ip_specs) * 100
        sign_consistent = ((gdp_ip_specs['coefficient'] > 0).sum() == len(gdp_ip_specs)) or \
                          ((gdp_ip_specs['coefficient'] < 0).sum() == len(gdp_ip_specs))
        median_coef = gdp_ip_specs['coefficient'].median()
        sign_word = "positive" if median_coef > 0 else "negative"

        md_lines.append(f"- **Sign consistency (GDP/IP specs):** {'All specifications have the same sign' if sign_consistent else 'Mixed signs across specifications'}")
        md_lines.append(f"- **Significance stability:** {n_sig_total}/{len(gdp_ip_specs)} ({pct_sig:.1f}%) GDP/IP specifications significant at 5%")
        md_lines.append(f"- **Direction:** Median coefficient is {sign_word} ({median_coef:.4f})")
        md_lines.append(f"- **Interpretation:** Financial distress {'reduces' if median_coef < 0 else 'increases'} log GDP")

        if pct_sig >= 80 and sign_consistent:
            strength = "STRONG"
        elif pct_sig >= 50 and sign_consistent:
            strength = "MODERATE"
        elif pct_sig >= 30:
            strength = "WEAK"
        else:
            strength = "FRAGILE"

        md_lines.append(f"- **Robustness assessment:** {strength}")

    # Note about total specs
    md_lines.append(f"\n- **Total specifications (all treatment vars):** {len(successful)}")
    n_sig_all = (successful['p_value'] < 0.05).sum()
    md_lines.append(f"- **Significant at 5% (all):** {n_sig_all}/{len(successful)}")

md_lines.append("")
md_lines.append(f"Surface hash: `{SURFACE_HASH}`")
md_lines.append("")

with open(f"{OUTPUT_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write("\n".join(md_lines))

print(f"Wrote SPECIFICATION_SEARCH.md")
print("\nDone!")
