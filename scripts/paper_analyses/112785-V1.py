"""
Specification Search Script for Hlatshwayo & Spence (2014)
"Demand and Defective Growth Patterns: The Role of the Tradable
and Non-Tradable Sectors in an Open Economy"
American Economic Review P&P, 104(5), 132-137.

Paper ID: 112785-V1

Surface-driven execution:
  - G1: va_growth ~ nontradable | year FE, cluster(industry)
  - Panel OLS with year FE (not industry FE, since nontradable is time-invariant)
  - Clustered SEs at industry level
  - 50+ specifications across controls, sample periods, FE swaps,
    outcome measures (VA growth, emp growth, levels, logs),
    treatment coding, weighting, and interactions

The paper's main claim: nontradable sectors grew faster than tradable
sectors in both employment and value added over 1990-2012 in the US.

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

PAPER_ID = "112785-V1"
DATA_DIR = "data/downloads/extracted/112785-V1"
OUTPUT_DIR = DATA_DIR
DATA_PATH = f"{DATA_DIR}/P2014_1103_data/2014_1103_data.xls"

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

print("Loading and constructing panel data from Excel...")

# --- Value Added Panel (from Figure 2 sheet) ---
df_f2 = pd.read_excel(DATA_PATH, sheet_name='Figure 2')
years_va = list(range(1990, 2013))  # 1990-2012

va_records = []

# Nontradable industries: rows 0-14 in Figure 2
for i in range(0, 15):
    row = df_f2.iloc[i]
    industry = str(row.iloc[2]).strip()
    sector = str(row.iloc[3]).strip()
    if sector not in ['Tradable', 'Nontradable']:
        continue
    for j, year in enumerate(years_va):
        val = row.iloc[4 + j]
        try:
            val = float(val)
        except:
            val = np.nan
        va_records.append({'industry': industry, 'sector': sector,
                           'year': year, 'va': val})

# Tradable industries: rows 18-36 in Figure 2
for i in range(18, 37):
    row = df_f2.iloc[i]
    industry = str(row.iloc[2]).strip()
    sector = str(row.iloc[3]).strip()
    if sector not in ['Tradable', 'Nontradable']:
        continue
    for j, year in enumerate(years_va):
        val = row.iloc[4 + j]
        try:
            val = float(val)
        except:
            val = np.nan
        va_records.append({'industry': industry, 'sector': sector,
                           'year': year, 'va': val})

df_va = pd.DataFrame(va_records).dropna(subset=['va'])

# --- Employment Panel (from Employment Data sheet) ---
df_emp_raw = pd.read_excel(DATA_PATH, sheet_name='Employment Data')
years_emp = list(range(1990, 2014))  # 1990-2013

emp_records = []
for i in range(len(df_emp_raw)):
    row = df_emp_raw.iloc[i]
    industry = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ''
    sector = str(row.iloc[3]).strip() if pd.notna(row.iloc[3]) else ''
    if sector in ['Tradable', 'Nontradable'] and industry:
        for j, year in enumerate(years_emp):
            val = row.iloc[5 + j]
            try:
                val = float(val)
            except:
                val = np.nan
            emp_records.append({'industry': industry, 'sector': sector,
                                'year': year, 'emp': val})

df_emp = pd.DataFrame(emp_records).dropna(subset=['emp'])

# --- Create unique industry identifiers ---
# Some industries appear in both tradable and nontradable (split industries)
df_va['industry_id'] = df_va['industry'] + '_' + df_va['sector']
df_emp['industry_id'] = df_emp['industry'] + '_' + df_emp['sector']

# --- Merge VA and employment panels ---
df_panel = pd.merge(df_va, df_emp[['industry_id', 'year', 'emp']],
                     on=['industry_id', 'year'], how='outer')

df_panel = df_panel.sort_values(['industry_id', 'year']).reset_index(drop=True)

# Nontradable dummy
df_panel['nontradable'] = (df_panel['sector'] == 'Nontradable').astype(int)
# Tradable dummy (reverse coding)
df_panel['tradable'] = (df_panel['sector'] == 'Tradable').astype(int)

# Year trend (centered at 2000)
df_panel['year_trend'] = df_panel['year'] - 2000

# Crisis dummy (2008-2009)
df_panel['crisis'] = ((df_panel['year'] >= 2008) & (df_panel['year'] <= 2009)).astype(int)

# Post-crisis dummy
df_panel['post_crisis'] = (df_panel['year'] >= 2010).astype(int)

# Post-2000 dummy
df_panel['post2000'] = (df_panel['year'] > 2000).astype(int)

# Log transforms
df_panel['log_va'] = np.log(df_panel['va'].clip(lower=0.001))
df_panel['log_emp'] = np.log(df_panel['emp'].clip(lower=0.001))

# Growth rates (annual log-change within industry)
df_panel['va_growth'] = df_panel.groupby('industry_id')['log_va'].diff()
df_panel['emp_growth'] = df_panel.groupby('industry_id')['log_emp'].diff()

# Level changes
df_panel['va_change'] = df_panel.groupby('industry_id')['va'].diff()
df_panel['emp_change'] = df_panel.groupby('industry_id')['emp'].diff()

# Base-period (1990) values for weighting
base_va = df_panel[df_panel['year'] == 1990][['industry_id', 'va']].rename(
    columns={'va': 'base_va'})
base_emp = df_panel[df_panel['year'] == 1990][['industry_id', 'emp']].rename(
    columns={'emp': 'base_emp'})
df_panel = df_panel.merge(base_va, on='industry_id', how='left')
df_panel = df_panel.merge(base_emp, on='industry_id', how='left')

# Industry size control (log of base VA)
df_panel['log_base_va'] = np.log(df_panel['base_va'].clip(lower=0.001))
df_panel['log_base_emp'] = np.log(df_panel['base_emp'].clip(lower=0.001))

# Interaction terms
df_panel['nt_trend'] = df_panel['nontradable'] * df_panel['year_trend']
df_panel['nt_post2000'] = df_panel['nontradable'] * df_panel['post2000']
df_panel['nt_crisis'] = df_panel['nontradable'] * df_panel['crisis']

# Ensure string columns for FE
df_panel['industry_id'] = df_panel['industry_id'].astype(str)
df_panel['year_str'] = df_panel['year'].astype(str)
df_panel['sector'] = df_panel['sector'].astype(str)

print(f"Panel constructed: {df_panel.shape[0]} obs, {df_panel['industry_id'].nunique()} industries")
print(f"  Nontradable: {df_panel[df_panel['nontradable']==1]['industry_id'].nunique()} industries")
print(f"  Tradable: {df_panel[df_panel['tradable']==1]['industry_id'].nunique()} industries")
print(f"  Years: {sorted(df_panel['year'].unique())}")

# Baseline sample: va_growth non-missing
df_base = df_panel.dropna(subset=['va_growth', 'nontradable']).copy()
print(f"Baseline sample (va_growth non-missing): {len(df_base)} obs")

# ============================================================
# Define controls
# ============================================================

# Note: with year FE as baseline, controls are cross-sectional characteristics
ALL_CONTROLS = ["log_base_va", "crisis", "post_crisis"]

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
             cluster_var="industry_id",
             axis_block_name=None, axis_block=None, notes="",
             weights_var=None):
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

        kwargs = {"fml": formula, "data": data, "vcov": vcov}
        if weights_var and weights_var in data.columns:
            kwargs["weights"] = weights_var

        m = pf.feols(**kwargs)

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
                       "method": "cluster", "cluster_vars": ["industry"]},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"panel_ols": design_audit},
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
# BASELINE: va_growth ~ nontradable | year FE, cluster(industry)
# ============================================================

print("\n=== Running baseline specification ===")
base_run_id, base_coef, base_se, base_pval, base_nobs = run_spec(
    "baseline", "designs/panel_ols.md#baseline", "G1",
    "va_growth", "nontradable", [],
    "year_str", "year", df_base,
    {"CRV1": "industry_id"},
    f"Full sample 1991-2012, N={len(df_base)}", "year FE (baseline)")

print(f"  Baseline: coef={base_coef:.6f}, se={base_se:.6f}, p={base_pval:.4f}, N={base_nobs}")


# ============================================================
# ADDITIONAL BASELINES: Alternative outcomes
# ============================================================

print("\n=== Running alternative baseline outcomes ===")

# Employment growth
df_emp_base = df_panel.dropna(subset=['emp_growth', 'nontradable']).copy()
run_spec(
    "baseline__emp_growth", "designs/panel_ols.md#baseline", "G1",
    "emp_growth", "nontradable", [],
    "year_str", "year", df_emp_base,
    {"CRV1": "industry_id"},
    f"Employment sample, N={len(df_emp_base)}", "year FE",
    axis_block_name="functional_form",
    axis_block={"spec_id": "baseline__emp_growth", "outcome": "emp_growth"})

# VA level
df_va_level = df_panel.dropna(subset=['va', 'nontradable']).copy()
run_spec(
    "baseline__va_level", "designs/panel_ols.md#baseline", "G1",
    "va", "nontradable", [],
    "year_str", "year", df_va_level,
    {"CRV1": "industry_id"},
    f"VA level sample, N={len(df_va_level)}", "year FE",
    axis_block_name="functional_form",
    axis_block={"spec_id": "baseline__va_level", "outcome": "va_level"})


# ============================================================
# CONTROLS SETS: Add controls to baseline
# ============================================================

print("\n=== Running controls sets ===")

# No FE, no controls (raw pooled comparison)
run_spec(
    "rc/controls/sets/none_no_fe", "designs/panel_ols.md#controls/sets", "G1",
    "va_growth", "nontradable", [],
    "", "none", df_base,
    "hetero",
    f"No FE no controls, N={len(df_base)}", "none")

# Year trend instead of year FE
run_spec(
    "rc/controls/sets/year_trend", "designs/panel_ols.md#controls/sets", "G1",
    "va_growth", "nontradable", ["year_trend"],
    "", "none", df_base,
    {"CRV1": "industry_id"},
    f"Year trend, N={len(df_base)}", "year_trend (no FE)")

# Year trend + size
df_ts = df_base.dropna(subset=["log_base_va"]).copy()
run_spec(
    "rc/controls/sets/year_trend_size", "designs/panel_ols.md#controls/sets", "G1",
    "va_growth", "nontradable", ["year_trend", "log_base_va"],
    "", "none", df_ts,
    {"CRV1": "industry_id"},
    f"Year trend+size, N={len(df_ts)}", "year_trend + log_base_va")

# Year FE + size
run_spec(
    "rc/controls/sets/year_fe_size", "designs/panel_ols.md#controls/sets", "G1",
    "va_growth", "nontradable", ["log_base_va"],
    "year_str", "year", df_ts,
    {"CRV1": "industry_id"},
    f"Year FE+size, N={len(df_ts)}", "log_base_va + year FE")

# Full controls (size + crisis dummies) with year FE
df_full = df_base.dropna(subset=ALL_CONTROLS).copy()
run_spec(
    "rc/controls/sets/full", "designs/panel_ols.md#controls/sets", "G1",
    "va_growth", "nontradable", ALL_CONTROLS,
    "year_str", "year", df_full,
    {"CRV1": "industry_id"},
    f"Full controls + year FE, N={len(df_full)}",
    "log_base_va + crisis + post_crisis + year FE")


# ============================================================
# CONTROLS LOO: Drop each control from full set
# ============================================================

print("\n=== Running controls LOO ===")

for ctrl in ALL_CONTROLS:
    remaining = [c for c in ALL_CONTROLS if c != ctrl]
    df_loo = df_full.copy()
    run_spec(
        f"rc/controls/loo/drop_{ctrl}",
        "designs/panel_ols.md#controls/loo", "G1",
        "va_growth", "nontradable", remaining,
        "year_str", "year", df_loo,
        {"CRV1": "industry_id"},
        f"Drop {ctrl}, N={len(df_loo)}", f"full minus {ctrl}",
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/controls/loo/drop_{ctrl}", "dropped": ctrl})


# ============================================================
# CONTROLS PROGRESSION: Build up incrementally
# ============================================================

print("\n=== Running controls progression ===")

progressions = [
    ("rc/controls/progression/no_controls", [], "", "none", "no FE, no controls"),
    ("rc/controls/progression/yearfe_only", [], "year_str", "year", "year FE only"),
    ("rc/controls/progression/yearfe_size", ["log_base_va"], "year_str", "year",
     "year FE + base size"),
    ("rc/controls/progression/yearfe_crisis", ["crisis", "post_crisis"], "year_str",
     "year", "year FE + crisis dummies"),
    ("rc/controls/progression/yearfe_crisis_size", ALL_CONTROLS, "year_str", "year",
     "year FE + crisis + base size"),
]

for spec_id, ctrls, fe, fe_desc, desc in progressions:
    data_prog = df_base.dropna(subset=ctrls).copy() if ctrls else df_base.copy()
    vcov = {"CRV1": "industry_id"} if fe else "hetero"
    run_spec(
        spec_id, "designs/panel_ols.md#controls/progression", "G1",
        "va_growth", "nontradable", ctrls,
        fe, fe_desc, data_prog,
        vcov,
        f"Progression: {desc}, N={len(data_prog)}", desc)


# ============================================================
# CONTROLS SUBSETS: Random subsets
# ============================================================

print("\n=== Running controls subsets ===")

rng = np.random.default_rng(112785)
n_subsets = 10

for k in range(1, n_subsets + 1):
    n_draw = rng.integers(1, len(ALL_CONTROLS) + 1)
    subset = list(rng.choice(ALL_CONTROLS, size=n_draw, replace=False))
    data_sub = df_base.dropna(subset=subset).copy()
    run_spec(
        f"rc/controls/subset/random_{k:03d}",
        "designs/panel_ols.md#controls/subset", "G1",
        "va_growth", "nontradable", subset,
        "year_str", "year", data_sub,
        {"CRV1": "industry_id"},
        f"Random subset {k}, N={len(data_sub)}", f"subset: {', '.join(subset)}",
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/controls/subset/random_{k:03d}",
                    "controls": subset})


# ============================================================
# SAMPLE PERIODS: Different time windows
# ============================================================

print("\n=== Running sample period variations ===")

period_specs = [
    ("rc/sample/period/pre2000", lambda d: d[d['year'] <= 2000],
     "1991-2000"),
    ("rc/sample/period/post2000", lambda d: d[d['year'] > 2000],
     "2001-2012"),
    ("rc/sample/period/pre_crisis_1990_2007",
     lambda d: d[d['year'] <= 2007], "1991-2007"),
    ("rc/sample/period/post_crisis_2009_2012",
     lambda d: d[d['year'] >= 2009], "2009-2012"),
    ("rc/sample/period/1990_2000",
     lambda d: d[(d['year'] >= 1991) & (d['year'] <= 2000)], "1991-2000"),
    ("rc/sample/period/2000_2007",
     lambda d: d[(d['year'] >= 2000) & (d['year'] <= 2007)], "2000-2007"),
    ("rc/sample/period/2007_2012",
     lambda d: d[(d['year'] >= 2007) & (d['year'] <= 2012)], "2007-2012"),
]

for spec_id, filter_fn, period_desc in period_specs:
    data_period = filter_fn(df_base).copy()
    if len(data_period) > 10:
        run_spec(
            spec_id, "designs/panel_ols.md#sample/period", "G1",
            "va_growth", "nontradable", [],
            "year_str", "year", data_period,
            {"CRV1": "industry_id"},
            f"Period {period_desc}, N={len(data_period)}", "year FE",
            axis_block_name="sample",
            axis_block={"spec_id": spec_id, "period": period_desc})


# ============================================================
# SAMPLE OUTLIERS: Drop large/unusual industries
# ============================================================

print("\n=== Running sample outlier variations ===")

outlier_specs = [
    ("rc/sample/outliers/drop_government",
     lambda d: d[~d['industry'].str.contains('Government', na=False)],
     "drop Government"),
    ("rc/sample/outliers/drop_realestate",
     lambda d: d[~d['industry'].str.contains('Real estate', na=False)],
     "drop Real estate"),
    ("rc/sample/outliers/drop_finance",
     lambda d: d[~d['industry'].str.contains('Finance', na=False)],
     "drop Finance & Insurance"),
    ("rc/sample/outliers/drop_government_realestate",
     lambda d: d[~d['industry'].str.contains('Government|Real estate', na=False)],
     "drop Government + Real estate"),
    ("rc/sample/outliers/drop_mining",
     lambda d: d[~d['industry'].str.contains('Mining', na=False)],
     "drop Mining"),
    ("rc/sample/outliers/drop_agriculture",
     lambda d: d[~d['industry'].str.contains('Agriculture', na=False)],
     "drop Agriculture"),
]

for spec_id, filter_fn, desc in outlier_specs:
    data_out = filter_fn(df_base).copy()
    if len(data_out) > 10:
        run_spec(
            spec_id, "designs/panel_ols.md#sample/outliers", "G1",
            "va_growth", "nontradable", [],
            "year_str", "year", data_out,
            {"CRV1": "industry_id"},
            f"{desc}, N={len(data_out)}", "year FE",
            axis_block_name="sample",
            axis_block={"spec_id": spec_id, "dropped": desc})


# ============================================================
# FIXED EFFECTS: Vary FE structure
# ============================================================

print("\n=== Running FE variations ===")

# Drop all FE (pooled OLS)
run_spec(
    "rc/fe/drop/all", "designs/panel_ols.md#fe/drop", "G1",
    "va_growth", "nontradable", [],
    "", "none", df_base,
    "hetero",
    f"No FE, N={len(df_base)}", "none",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/drop/all", "fe": "none"})

# Year trend instead of year FE
run_spec(
    "rc/fe/swap/trend_for_yearfe", "designs/panel_ols.md#fe/swap", "G1",
    "va_growth", "nontradable", ["year_trend"],
    "", "none", df_base,
    {"CRV1": "industry_id"},
    f"Trend for year FE, N={len(df_base)}", "year_trend replaces year FE",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/swap/trend_for_yearfe", "fe": "year_trend"})

# Industry FE with interaction (nontradable is absorbed, but nt_trend captures
# differential trend)
run_spec(
    "rc/fe/add/industry_nt_trend", "designs/panel_ols.md#fe/add", "G1",
    "va_growth", "nt_trend", ["year_trend"],
    "industry_id", "industry", df_base,
    {"CRV1": "industry_id"},
    f"Industry FE + NT*trend, N={len(df_base)}",
    "year_trend (nt absorbed, nt_trend is focal)",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/add/industry_nt_trend",
                "fe": "industry", "notes": "nontradable absorbed; nt_trend is focal"})

# Industry + Year FE with interaction
run_spec(
    "rc/fe/add/industry_year_nt_trend", "designs/panel_ols.md#fe/add", "G1",
    "va_growth", "nt_trend", [],
    "industry_id + year_str", "industry + year", df_base,
    {"CRV1": "industry_id"},
    f"Industry+Year FE + NT*trend, N={len(df_base)}",
    "none (nt and year absorbed; nt_trend is focal)",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/add/industry_year_nt_trend",
                "fe": "industry + year",
                "notes": "nontradable and year absorbed; nt_trend is focal"})

# Quadratic trend
df_base['year_trend_sq'] = df_base['year_trend'] ** 2
run_spec(
    "rc/fe/swap/quadratic_trend", "designs/panel_ols.md#fe/swap", "G1",
    "va_growth", "nontradable", ["year_trend", "year_trend_sq"],
    "", "none", df_base,
    {"CRV1": "industry_id"},
    f"Quadratic trend, N={len(df_base)}", "year_trend + year_trend^2",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/swap/quadratic_trend", "fe": "quadratic trend"})


# ============================================================
# FUNCTIONAL FORM: Alternative outcomes
# ============================================================

print("\n=== Running outcome/functional form variations ===")

outcome_specs = [
    ("rc/form/outcome/va_level", "va",
     df_panel.dropna(subset=['va', 'nontradable']).copy(),
     "VA level (billions)"),
    ("rc/form/outcome/emp_growth", "emp_growth",
     df_panel.dropna(subset=['emp_growth', 'nontradable']).copy(),
     "Employment growth (log-change)"),
    ("rc/form/outcome/emp_level", "emp",
     df_panel.dropna(subset=['emp', 'nontradable']).copy(),
     "Employment level (thousands)"),
    ("rc/form/outcome/log_va", "log_va",
     df_panel.dropna(subset=['log_va', 'nontradable']).copy(),
     "Log VA"),
    ("rc/form/outcome/log_emp", "log_emp",
     df_panel.dropna(subset=['log_emp', 'nontradable']).copy(),
     "Log employment"),
    ("rc/form/outcome/va_change", "va_change",
     df_panel.dropna(subset=['va_change', 'nontradable']).copy(),
     "VA level change"),
    ("rc/form/outcome/emp_change", "emp_change",
     df_panel.dropna(subset=['emp_change', 'nontradable']).copy(),
     "Employment level change"),
]

for spec_id, outcome, data_out, desc in outcome_specs:
    if len(data_out) > 10:
        run_spec(
            spec_id, "designs/panel_ols.md#form/outcome", "G1",
            outcome, "nontradable", [],
            "year_str", "year", data_out,
            {"CRV1": "industry_id"},
            f"{desc}, N={len(data_out)}", "year FE",
            axis_block_name="functional_form",
            axis_block={"spec_id": spec_id, "outcome": outcome})


# ============================================================
# TREATMENT CODING: Tradable dummy (reverse)
# ============================================================

print("\n=== Running treatment coding variations ===")

run_spec(
    "rc/form/treatment/tradable_dummy", "designs/panel_ols.md#form/treatment", "G1",
    "va_growth", "tradable", [],
    "year_str", "year", df_base,
    {"CRV1": "industry_id"},
    f"Tradable dummy, N={len(df_base)}", "year FE",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/treatment/tradable_dummy",
                "treatment": "tradable (reverse coding)"})


# ============================================================
# WEIGHTED REGRESSIONS
# ============================================================

print("\n=== Running weighted regressions ===")

# Weighted by base VA
df_wva = df_base.dropna(subset=['base_va']).copy()
df_wva = df_wva[df_wva['base_va'] > 0].copy()
run_spec(
    "rc/form/estimator/weighted_by_base_va",
    "designs/panel_ols.md#form/estimator", "G1",
    "va_growth", "nontradable", [],
    "year_str", "year", df_wva,
    {"CRV1": "industry_id"},
    f"Weighted by base VA, N={len(df_wva)}", "year FE",
    axis_block_name="estimation",
    axis_block={"spec_id": "rc/form/estimator/weighted_by_base_va",
                "weights": "base_va"},
    weights_var="base_va")

# Weighted by base employment
df_wemp = df_base.dropna(subset=['base_emp']).copy()
df_wemp = df_wemp[df_wemp['base_emp'] > 0].copy()
run_spec(
    "rc/form/estimator/weighted_by_base_emp",
    "designs/panel_ols.md#form/estimator", "G1",
    "va_growth", "nontradable", [],
    "year_str", "year", df_wemp,
    {"CRV1": "industry_id"},
    f"Weighted by base emp, N={len(df_wemp)}", "year FE",
    axis_block_name="estimation",
    axis_block={"spec_id": "rc/form/estimator/weighted_by_base_emp",
                "weights": "base_emp"},
    weights_var="base_emp")


# ============================================================
# INTERACTION SPECIFICATIONS
# ============================================================

print("\n=== Running interaction specifications ===")

# NT x post-2000 (structural break in differential growth)
run_spec(
    "rc/form/interaction/nt_x_post2000",
    "designs/panel_ols.md#form/interaction", "G1",
    "va_growth", "nontradable", ["post2000", "nt_post2000"],
    "year_str", "year", df_base,
    {"CRV1": "industry_id"},
    f"NT x post-2000 interaction, N={len(df_base)}",
    "post2000 + NT*post2000 + year FE",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/interaction/nt_x_post2000",
                "interaction": "nontradable x post2000"})

# NT x crisis
run_spec(
    "rc/form/interaction/nt_x_crisis",
    "designs/panel_ols.md#form/interaction", "G1",
    "va_growth", "nontradable", ["crisis", "nt_crisis"],
    "year_str", "year", df_base,
    {"CRV1": "industry_id"},
    f"NT x crisis interaction, N={len(df_base)}",
    "crisis + NT*crisis + year FE",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/interaction/nt_x_crisis",
                "interaction": "nontradable x crisis"})

# NT x trend (differential trend, industry FE absorbs level)
run_spec(
    "rc/form/interaction/nt_trend_industry_fe",
    "designs/panel_ols.md#form/interaction", "G1",
    "va_growth", "nt_trend", ["year_trend"],
    "industry_id", "industry", df_base,
    {"CRV1": "industry_id"},
    f"NT*trend with industry FE, N={len(df_base)}",
    "year_trend + industry FE (nt absorbed)",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/interaction/nt_trend_industry_fe",
                "interaction": "nontradable x year_trend",
                "notes": "industry FE absorbs nontradable level"})


# ============================================================
# COMBINED VARIATIONS: Full controls with different outcomes
# ============================================================

print("\n=== Running combined variations ===")

combined_specs = [
    ("rc/combined/emp_growth_full", "emp_growth",
     df_panel.dropna(subset=['emp_growth'] + ALL_CONTROLS).copy(),
     "Employment growth + full controls"),
    ("rc/combined/va_level_full", "va",
     df_panel.dropna(subset=['va'] + ALL_CONTROLS).copy(),
     "VA level + full controls"),
    ("rc/combined/log_va_full", "log_va",
     df_panel.dropna(subset=['log_va'] + ALL_CONTROLS).copy(),
     "Log VA + full controls"),
    ("rc/combined/va_change_full", "va_change",
     df_panel.dropna(subset=['va_change'] + ALL_CONTROLS).copy(),
     "VA change + full controls"),
]

for spec_id, outcome, data_comb, desc in combined_specs:
    if len(data_comb) > 10:
        run_spec(
            spec_id, "designs/panel_ols.md#combined", "G1",
            outcome, "nontradable", ALL_CONTROLS,
            "year_str", "year", data_comb,
            {"CRV1": "industry_id"},
            f"{desc}, N={len(data_comb)}", "full controls + year FE")


# ============================================================
# COMBINED: Outcome x Period
# ============================================================

print("\n=== Running outcome x period combinations ===")

for outcome, out_desc, data_src in [
    ("va_growth", "VA growth", df_base),
    ("emp_growth", "Emp growth",
     df_panel.dropna(subset=['emp_growth', 'nontradable']).copy()),
]:
    for period_name, filter_fn, period_desc in [
        ("pre2000", lambda d: d[d['year'] <= 2000], "1991-2000"),
        ("post2000", lambda d: d[d['year'] > 2000], "2001-2012"),
        ("pre_crisis", lambda d: d[d['year'] <= 2007], "1991-2007"),
    ]:
        data_op = filter_fn(data_src).copy()
        if len(data_op) > 10:
            run_spec(
                f"rc/combined/{outcome}_{period_name}",
                "designs/panel_ols.md#combined", "G1",
                outcome, "nontradable", [],
                "year_str", "year", data_op,
                {"CRV1": "industry_id"},
                f"{out_desc} {period_desc}, N={len(data_op)}", "year FE",
                axis_block_name="sample",
                axis_block={"spec_id": f"rc/combined/{outcome}_{period_name}",
                            "outcome": outcome, "period": period_desc})


# ============================================================
# ADDITIONAL: Sector-aggregated regressions (Table 1 style)
# ============================================================

print("\n=== Running sector-aggregated specifications ===")

# Aggregate to sector-year level (total NT vs T per year)
sector_year = df_panel.groupby(['sector', 'year']).agg(
    total_va=('va', 'sum'),
    total_emp=('emp', 'sum'),
    n_industries=('industry_id', 'nunique')
).reset_index()

sector_year['nontradable'] = (sector_year['sector'] == 'Nontradable').astype(int)
sector_year['year_str'] = sector_year['year'].astype(str)
sector_year['year_trend'] = sector_year['year'] - 2000
sector_year['log_total_va'] = np.log(sector_year['total_va'].clip(lower=0.001))
sector_year['log_total_emp'] = np.log(sector_year['total_emp'].clip(lower=0.001))
sector_year['va_growth_agg'] = sector_year.groupby('sector')['log_total_va'].diff()
sector_year['emp_growth_agg'] = sector_year.groupby('sector')['log_total_emp'].diff()

sy_base = sector_year.dropna(subset=['va_growth_agg']).copy()

if len(sy_base) > 5:
    run_spec(
        "rc/form/aggregation/sector_year_va",
        "designs/panel_ols.md#form/aggregation", "G1",
        "va_growth_agg", "nontradable", ["year_trend"],
        "", "none", sy_base,
        "hetero",
        f"Sector-year aggregated VA growth, N={len(sy_base)}",
        "year_trend (sector-year level)",
        cluster_var="sector",
        axis_block_name="functional_form",
        axis_block={"spec_id": "rc/form/aggregation/sector_year_va",
                    "aggregation": "sector-year totals"})

sy_emp = sector_year.dropna(subset=['emp_growth_agg']).copy()
if len(sy_emp) > 5:
    run_spec(
        "rc/form/aggregation/sector_year_emp",
        "designs/panel_ols.md#form/aggregation", "G1",
        "emp_growth_agg", "nontradable", ["year_trend"],
        "", "none", sy_emp,
        "hetero",
        f"Sector-year aggregated emp growth, N={len(sy_emp)}",
        "year_trend (sector-year level)",
        cluster_var="sector",
        axis_block_name="functional_form",
        axis_block={"spec_id": "rc/form/aggregation/sector_year_emp",
                    "aggregation": "sector-year totals"})


# ============================================================
# INFERENCE VARIANTS (on baseline spec)
# ============================================================

print("\n=== Running inference variants ===")

infer_run_counter = 0

def run_inference_variant(infer_spec_id, vcov_spec, vcov_desc):
    """Run baseline spec with alternative inference."""
    global infer_run_counter
    infer_run_counter += 1
    run_id = f"{PAPER_ID}_infer_{infer_run_counter:03d}"

    try:
        formula = "va_growth ~ nontradable | year_str"
        m = pf.feols(formula, data=df_base, vcov=vcov_spec)

        coef_val = float(m.coef().get("nontradable", np.nan))
        se_val = float(m.se().get("nontradable", np.nan))
        pval = float(m.pvalue().get("nontradable", np.nan))

        try:
            ci = m.confint()
            ci_lower = float(ci.loc["nontradable", ci.columns[0]])
            ci_upper = float(ci.loc["nontradable", ci.columns[1]])
        except:
            ci_lower = np.nan
            ci_upper = np.nan

        nobs = int(m._N)

        inference_results.append({
            "paper_id": PAPER_ID,
            "spec_run_id": run_id,
            "spec_id": infer_spec_id,
            "baseline_group_id": "G1",
            "outcome_var": "va_growth",
            "treatment_var": "nontradable",
            "coefficient": coef_val,
            "std_error": se_val,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": nobs,
            "vcov_desc": vcov_desc,
            "run_success": 1,
            "run_error": ""
        })

    except Exception as e:
        err_msg = str(e)[:240]
        inference_results.append({
            "paper_id": PAPER_ID,
            "spec_run_id": run_id,
            "spec_id": infer_spec_id,
            "baseline_group_id": "G1",
            "outcome_var": "va_growth",
            "treatment_var": "nontradable",
            "coefficient": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": np.nan,
            "vcov_desc": vcov_desc,
            "run_success": 0,
            "run_error": err_msg
        })

# Canonical: cluster by industry
run_inference_variant("infer/se/cluster/industry",
                      {"CRV1": "industry_id"}, "cluster(industry)")

# HC1 robust
run_inference_variant("infer/se/hc/hc1", "hetero", "HC1 robust")

# HC3 robust
run_inference_variant("infer/se/hc/hc3", "HC3", "HC3 robust")

# iid
run_inference_variant("infer/se/iid", "iid", "iid (no correction)")


# ============================================================
# Save Results
# ============================================================

print("\n=== Saving results ===")

spec_df = pd.DataFrame(results)
spec_df.to_csv(f"{OUTPUT_DIR}/specification_results.csv", index=False)
print(f"Saved specification_results.csv: {len(spec_df)} rows")

infer_df = pd.DataFrame(inference_results)
infer_df.to_csv(f"{OUTPUT_DIR}/inference_results.csv", index=False)
print(f"Saved inference_results.csv: {len(infer_df)} rows")

# Summary stats
successful = spec_df[spec_df['run_success'] == 1]
failed = spec_df[spec_df['run_success'] == 0]
print(f"\nSuccessful: {len(successful)}, Failed: {len(failed)}")

if len(successful) > 0:
    n_sig = (successful['p_value'] < 0.05).sum()
    print(f"Significant at 5%: {n_sig}/{len(successful)} ({100*n_sig/len(successful):.1f}%)")
    print(f"Coefficient range: [{successful['coefficient'].min():.6f}, {successful['coefficient'].max():.6f}]")
    print(f"Median coefficient: {successful['coefficient'].median():.6f}")


# ============================================================
# Generate SPECIFICATION_SEARCH.md
# ============================================================

print("\n=== Generating SPECIFICATION_SEARCH.md ===")

md_lines = []
md_lines.append("# Specification Search Report")
md_lines.append("")
md_lines.append(f"**Paper ID:** {PAPER_ID}")
md_lines.append(f"**Paper:** Hlatshwayo & Spence (2014) - Demand and Defective Growth Patterns: The Role of the Tradable and Non-Tradable Sectors in an Open Economy")
md_lines.append("")
md_lines.append("## Design Summary")
md_lines.append("")
md_lines.append("- **Design:** Panel OLS (industry-year panel, 1990-2012)")
md_lines.append("- **Outcome:** VA growth (annual log-change in real value added)")
md_lines.append("- **Treatment:** nontradable (1=nontradable sector, 0=tradable)")
md_lines.append("- **Controls:** base-period size, crisis dummies")
md_lines.append("- **Fixed effects:** year (baseline); industry FE used with interaction terms")
md_lines.append("- **Clustering:** industry")
md_lines.append("- **Note:** Nontradable is time-invariant, so industry FE absorbs it. Baseline uses year FE to capture cross-sectional growth differential.")
md_lines.append("")

base_row = successful[successful['spec_id'] == 'baseline']
if len(base_row) > 0:
    bc = base_row.iloc[0]
    md_lines.append("## Baseline Results")
    md_lines.append("")
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
    "Sample Period": successful[successful['spec_id'].str.startswith('rc/sample/period/')],
    "Sample Outliers": successful[successful['spec_id'].str.startswith('rc/sample/outliers/')],
    "Fixed Effects": successful[successful['spec_id'].str.startswith('rc/fe/')],
    "Functional Form": successful[successful['spec_id'].str.startswith('rc/form/')],
    "Interactions": successful[successful['spec_id'].str.contains('interaction')],
    "Combined": successful[successful['spec_id'].str.startswith('rc/combined/')],
    "Aggregation": successful[successful['spec_id'].str.contains('aggregation')],
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
