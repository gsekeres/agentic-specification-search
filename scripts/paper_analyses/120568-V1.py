"""
Specification Search Script for Pries & Rogerson (2020)
"Declining Worker Turnover: the Role of Short Duration Employment Spells"
American Economic Journal: Macroeconomics, 12(1), 58-98.

Paper ID: 120568-V1

Surface-driven execution:
  - G1: oneqsepsrate ~ time_trend | state FE, cluster(state)
  - Panel time-series OLS with state FE, clustered SEs at state level
  - 50+ specifications across outcome variables, controls (quarter FE, quadratic trend),
    sample restrictions, panel disaggregations (sex-age, industry, firm size),
    functional form (log, first-difference), FE swaps, inference variants

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

PAPER_ID = "120568-V1"
DATA_DIR = "data/downloads/extracted/120568-V1"
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

STATELIST = ["ca","co","ct","fl","ga","hi","id","il","in","ks","la","md","me",
             "mn","mo","mt","nc","nd","nj","nm","nv","pa","ri","sc","sd","tn",
             "tx","va","wa","wv"]

STATE_LABELS = {
    6:"CA", 8:"CO", 9:"CT", 12:"FL", 13:"GA", 15:"HI", 16:"ID", 17:"IL",
    18:"IN", 20:"KS", 22:"LA", 24:"MD", 23:"ME", 27:"MN", 29:"MO", 30:"MT",
    37:"NC", 38:"ND", 34:"NJ", 35:"NM", 32:"NV", 42:"PA", 44:"RI", 45:"SC",
    46:"SD", 47:"TN", 48:"TX", 51:"VA", 53:"WA", 54:"WV"
}


def load_geography_panel():
    """Load state-level panel from raw QWI CSV files (replicating shortjobs_geography.do)."""
    frames = []
    for st in STATELIST:
        fpath = f"{DATA_DIR}/data/qwi_{st}_sa_f_gs_ns_op_u.csv"
        df = pd.read_csv(fpath, usecols=['geography','industry','sex','agegrp','year','quarter',
                                          'Emp','EmpEnd','EmpTotal','HirA','HirAEnd','HirAS','Sep','SepBeg'])
        # Keep aggregated sex/age, all-industry (like geography.do)
        df = df[(df['sex'] == 0) & (df['agegrp'] == 'A00') & (df['industry'] == '00')]
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.rename(columns={
        'Emp': 'emp', 'EmpEnd': 'empend', 'EmpTotal': 'emptotal',
        'HirA': 'hira', 'HirAEnd': 'hiraend', 'HirAS': 'hiras',
        'Sep': 'sep', 'SepBeg': 'sepbeg'
    })
    return panel


def load_sex_age_panel():
    """Load sex-age disaggregated panel (replicating shortjobs_sex_age.do)."""
    frames = []
    for st in STATELIST:
        fpath = f"{DATA_DIR}/data/qwi_{st}_sa_f_gs_ns_op_u.csv"
        df = pd.read_csv(fpath, usecols=['geography','industry','sex','agegrp','year','quarter',
                                          'Emp','EmpEnd','EmpTotal','HirA','HirAEnd','HirAS','Sep','SepBeg'])
        # Drop aggregated sex/age, keep all-industry
        df = df[(df['sex'] != 0) & (df['agegrp'] != 'A00') & (df['industry'] == '00')]
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.rename(columns={
        'Emp': 'emp', 'EmpEnd': 'empend', 'EmpTotal': 'emptotal',
        'HirA': 'hira', 'HirAEnd': 'hiraend', 'HirAS': 'hiras',
        'Sep': 'sep', 'SepBeg': 'sepbeg'
    })
    return panel


def load_industry_panel():
    """Load industry disaggregated panel (replicating shortjobs_industry.do)."""
    frames = []
    for st in STATELIST:
        fpath = f"{DATA_DIR}/data/qwi_{st}_sa_f_gs_ns_op_u.csv"
        df = pd.read_csv(fpath, usecols=['geography','industry','sex','agegrp','year','quarter',
                                          'Emp','EmpEnd','EmpTotal','HirA','HirAEnd','HirAS','Sep','SepBeg'])
        # Keep aggregated sex/age, drop "00" (all) and "92" (public admin)
        df = df[(df['sex'] == 0) & (df['agegrp'] == 'A00') & (~df['industry'].isin(['00', '92']))]
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.rename(columns={
        'Emp': 'emp', 'EmpEnd': 'empend', 'EmpTotal': 'emptotal',
        'HirA': 'hira', 'HirAEnd': 'hiraend', 'HirAS': 'hiras',
        'Sep': 'sep', 'SepBeg': 'sepbeg'
    })
    return panel


def compute_rates(df, group_col):
    """Compute labor market flow rates from raw QWI variables, within panel groups."""
    df = df.sort_values([group_col, 'time']).copy()
    df['allemptot'] = df['emptotal']

    # One-quarter rates
    df['oneqsepsrate'] = (df['hira'] - df['hiraend']) / df['allemptot']
    df['oneqhazrate'] = (df['hira'] - df['hiraend']) / df['hira']
    df['hirerate'] = df['hira'] / df['allemptot']
    df['sepsrate'] = df['sep'] / df['allemptot']
    df['twoplussepsrate'] = df['sepbeg'] / df['allemptot']
    df['twoplushazrate'] = df['sepbeg'] / (df['allemptot'] - df['hira'])

    # Two-quarter rates (need lag)
    df['L_hiraend'] = df.groupby(group_col)['hiraend'].shift(1)
    df['twoqsepsrate'] = (df['L_hiraend'] - df['hiras']) / df['allemptot']
    df['twoqhazrate'] = (df['L_hiraend'] - df['hiras']) / df['L_hiraend']

    # Three-plus quarter rates
    df['threeplussepsrate'] = (df['sep'] - (df['L_hiraend'] - df['hiras']) - (df['hira'] - df['hiraend'])) / df['allemptot']
    df['threeplushazrate'] = (df['sep'] - (df['L_hiraend'] - df['hiras']) - (df['hira'] - df['hiraend'])) / (df['allemptot'] - df['L_hiraend'] - df['hira'])

    return df


def prepare_panel(raw_df, group_col, time_filter_end=2018*4):
    """Prepare panel: create time, filter, compute rates, add trend."""
    raw_df['time'] = raw_df['year'] * 4 + raw_df['quarter']
    raw_df = compute_rates(raw_df, group_col)

    # Time filter: 1999Q1 through end
    raw_df = raw_df[(raw_df['time'] > 1998*4+4) & (raw_df['time'] <= time_filter_end)].copy()

    time_min = raw_df['time'].min()
    raw_df['time_trend'] = raw_df['time'] - time_min
    raw_df['time_trend_sq'] = raw_df['time_trend'] ** 2
    raw_df['date'] = raw_df['year'] + (raw_df['quarter'] - 1) / 4.0
    raw_df['quarter_str'] = raw_df['quarter'].astype(str)

    return raw_df


print("Loading geography (state-level) panel...")
geo_raw = load_geography_panel()
geo_raw['time'] = geo_raw['year'] * 4 + geo_raw['quarter']
geo_df = prepare_panel(geo_raw, 'geography')
geo_df['state_str'] = geo_df['geography'].astype(str)
print(f"  Geography panel: {len(geo_df)} rows, {geo_df['geography'].nunique()} states")

# Log outcome
geo_df['log_oneqsepsrate'] = np.log(geo_df['oneqsepsrate'].clip(lower=1e-10))

# First-difference outcome
geo_df = geo_df.sort_values(['geography', 'time'])
geo_df['fd_oneqsepsrate'] = geo_df.groupby('geography')['oneqsepsrate'].diff()

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
             cluster_var="state_str",
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

        est_data = data.dropna(subset=[outcome_var, treatment_var]).copy()
        m = pf.feols(formula, data=est_data, vcov=vcov)

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
                       "method": "cluster", "cluster_vars": ["state"]},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"panel_time_series_ols": design_audit},
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
# BASELINE: oneqsepsrate ~ time_trend | state FE, cluster(state)
# ============================================================

print("Running baseline specification...")
base_run_id, base_coef, base_se, base_pval, base_nobs = run_spec(
    "baseline", "designs/panel_time_series_ols.md#baseline", "G1",
    "oneqsepsrate", "time_trend", [],
    "state_str", "state", geo_df,
    {"CRV1": "state_str"},
    f"30 states, 1999Q1-2017Q4", "none (linear trend + state FE)")

print(f"  Baseline: coef={base_coef:.8f}, se={base_se:.8f}, p={base_pval:.6f}, N={base_nobs}")


# ============================================================
# RC: ALTERNATIVE OUTCOMES
# ============================================================

print("Running alternative outcome specifications...")

OUTCOME_SPECS = [
    ("rc/outcome/oneqhazrate", "oneqhazrate", "one-quarter hazard rate"),
    ("rc/outcome/hirerate", "hirerate", "hire rate"),
    ("rc/outcome/sepsrate", "sepsrate", "overall separation rate"),
    ("rc/outcome/twoqsepsrate", "twoqsepsrate", "two-quarter separation rate"),
    ("rc/outcome/threeplussepsrate", "threeplussepsrate", "three-plus quarter separation rate"),
    ("rc/outcome/twoplussepsrate", "twoplussepsrate", "two-plus quarter separation rate"),
]

for spec_id, outcome, desc in OUTCOME_SPECS:
    run_spec(
        spec_id, "modules/robustness/outcome.md#alternative-outcomes", "G1",
        outcome, "time_trend", [],
        "state_str", "state", geo_df,
        {"CRV1": "state_str"},
        "30 states, 1999Q1-2017Q4", f"outcome: {desc}",
        axis_block_name="outcome",
        axis_block={"spec_id": spec_id, "outcome_var": outcome, "desc": desc})


# ============================================================
# RC: CONTROLS — Add quarter FE (seasonal adjustment)
# ============================================================

print("Running controls variants...")

run_spec(
    "rc/controls/add_quarter_fe",
    "modules/robustness/controls.md#add-controls", "G1",
    "oneqsepsrate", "time_trend", [],
    "state_str + quarter_str", "state + quarter", geo_df,
    {"CRV1": "state_str"},
    "30 states, 1999Q1-2017Q4", "quarter FE for seasonal adjustment",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/add_quarter_fe", "family": "add",
                "added": ["quarter_fe"], "n_controls": 0})

# Quadratic time trend
run_spec(
    "rc/controls/quadratic_trend",
    "modules/robustness/controls.md#control-progression-build-up", "G1",
    "oneqsepsrate", "time_trend", ["time_trend_sq"],
    "state_str", "state", geo_df,
    {"CRV1": "state_str"},
    "30 states, 1999Q1-2017Q4", "quadratic time trend",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/quadratic_trend", "family": "progression",
                "n_controls": 1, "set_name": "quadratic"})

# Quadratic + quarter FE
run_spec(
    "rc/controls/quadratic_plus_quarter",
    "modules/robustness/controls.md#control-progression-build-up", "G1",
    "oneqsepsrate", "time_trend", ["time_trend_sq"],
    "state_str + quarter_str", "state + quarter", geo_df,
    {"CRV1": "state_str"},
    "30 states, 1999Q1-2017Q4", "quadratic trend + quarter FE",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/quadratic_plus_quarter", "family": "progression",
                "n_controls": 1, "set_name": "quadratic_plus_quarter"})


# ============================================================
# RC: CONTROLS + ALTERNATIVE OUTCOMES (cross outcomes x controls)
# ============================================================

print("Running outcomes x controls combinations...")

for spec_id_base, outcome, desc in OUTCOME_SPECS:
    # With quarter FE
    run_spec(
        f"{spec_id_base}_quarter_fe",
        "modules/robustness/outcome.md#alternative-outcomes", "G1",
        outcome, "time_trend", [],
        "state_str + quarter_str", "state + quarter", geo_df,
        {"CRV1": "state_str"},
        "30 states, 1999Q1-2017Q4", f"{desc} + quarter FE",
        axis_block_name="outcome",
        axis_block={"spec_id": f"{spec_id_base}_quarter_fe", "outcome_var": outcome,
                    "desc": f"{desc} + quarter FE"})

    # With quadratic trend
    run_spec(
        f"{spec_id_base}_quad",
        "modules/robustness/outcome.md#alternative-outcomes", "G1",
        outcome, "time_trend", ["time_trend_sq"],
        "state_str", "state", geo_df,
        {"CRV1": "state_str"},
        "30 states, 1999Q1-2017Q4", f"{desc} + quadratic trend",
        axis_block_name="outcome",
        axis_block={"spec_id": f"{spec_id_base}_quad", "outcome_var": outcome,
                    "desc": f"{desc} + quadratic"})


# ============================================================
# RC: FIXED EFFECTS
# ============================================================

print("Running FE variants...")

# Drop state FE (pooled OLS)
run_spec(
    "rc/fe/drop_state",
    "modules/robustness/fixed_effects.md#dropping-fe-relative-to-baseline", "G1",
    "oneqsepsrate", "time_trend", [],
    "", "none (pooled OLS)", geo_df,
    {"CRV1": "state_str"},
    "30 states, 1999Q1-2017Q4", "no FE (pooled)",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/drop_state", "family": "drop",
                "dropped": ["state"], "baseline_fe": ["state"], "new_fe": []})

# Add quarter FE (state + quarter)
run_spec(
    "rc/fe/add_quarter",
    "modules/robustness/fixed_effects.md#additive-fe-variations-relative-to-baseline", "G1",
    "oneqsepsrate", "time_trend", [],
    "state_str + quarter_str", "state + quarter", geo_df,
    {"CRV1": "state_str"},
    "30 states, 1999Q1-2017Q4", "state + quarter FE",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/add_quarter", "family": "add",
                "added": ["quarter"], "baseline_fe": ["state"], "new_fe": ["state", "quarter"]})


# ============================================================
# RC: SAMPLE RESTRICTIONS
# ============================================================

print("Running sample restriction variants...")

# Exclude Great Recession period (2007Q4 - 2010Q2)
df_no_gr = geo_df[~((geo_df['year'] >= 2008) & (geo_df['year'] <= 2009))].copy()
run_spec(
    "rc/sample/exclude_great_recession",
    "modules/robustness/sample.md#sample-restrictions", "G1",
    "oneqsepsrate", "time_trend", [],
    "state_str", "state", df_no_gr,
    {"CRV1": "state_str"},
    f"Exclude 2008-2009 (Great Recession), N={len(df_no_gr)}", "none",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/exclude_great_recession", "axis": "time_restriction",
                "rule": "exclude", "params": {"exclude_years": [2008, 2009]}})

# Pre-2008 only
df_pre = geo_df[geo_df['year'] < 2008].copy()
run_spec(
    "rc/sample/pre_2008",
    "modules/robustness/sample.md#sample-restrictions", "G1",
    "oneqsepsrate", "time_trend", [],
    "state_str", "state", df_pre,
    {"CRV1": "state_str"},
    f"Pre-2008 only, N={len(df_pre)}", "none",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/pre_2008", "axis": "time_restriction",
                "rule": "before", "params": {"cutoff_year": 2008}})

# Post-2008
df_post = geo_df[geo_df['year'] >= 2008].copy()
# Need new time_trend for this subsample
df_post['time_trend'] = df_post['time'] - df_post['time'].min()
df_post['time_trend_sq'] = df_post['time_trend'] ** 2
run_spec(
    "rc/sample/post_2008",
    "modules/robustness/sample.md#sample-restrictions", "G1",
    "oneqsepsrate", "time_trend", [],
    "state_str", "state", df_post,
    {"CRV1": "state_str"},
    f"Post-2008 only, N={len(df_post)}", "none",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/post_2008", "axis": "time_restriction",
                "rule": "after", "params": {"cutoff_year": 2008}})

# Pre-2005 (early period)
df_early = geo_df[geo_df['year'] < 2005].copy()
run_spec(
    "rc/sample/pre_2005",
    "modules/robustness/sample.md#sample-restrictions", "G1",
    "oneqsepsrate", "time_trend", [],
    "state_str", "state", df_early,
    {"CRV1": "state_str"},
    f"Pre-2005 only, N={len(df_early)}", "none",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/pre_2005", "axis": "time_restriction",
                "rule": "before", "params": {"cutoff_year": 2005}})

# Post-2010 (recovery)
df_recovery = geo_df[geo_df['year'] >= 2010].copy()
df_recovery['time_trend'] = df_recovery['time'] - df_recovery['time'].min()
run_spec(
    "rc/sample/post_2010",
    "modules/robustness/sample.md#sample-restrictions", "G1",
    "oneqsepsrate", "time_trend", [],
    "state_str", "state", df_recovery,
    {"CRV1": "state_str"},
    f"Post-2010 only, N={len(df_recovery)}", "none",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/post_2010", "axis": "time_restriction",
                "rule": "after", "params": {"cutoff_year": 2010}})

# Trim outcome at 1st/99th percentile
q01 = geo_df['oneqsepsrate'].quantile(0.01)
q99 = geo_df['oneqsepsrate'].quantile(0.99)
df_trim1 = geo_df[(geo_df['oneqsepsrate'] >= q01) & (geo_df['oneqsepsrate'] <= q99)].copy()
run_spec(
    "rc/sample/trim_rate_1_99",
    "modules/robustness/sample.md#outliers-and-influential-observations", "G1",
    "oneqsepsrate", "time_trend", [],
    "state_str", "state", df_trim1,
    {"CRV1": "state_str"},
    f"Trim oneqsepsrate [1%,99%], N={len(df_trim1)}", "none",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/trim_rate_1_99", "axis": "outliers",
                "rule": "trim", "params": {"lower_q": 0.01, "upper_q": 0.99}})

# Trim outcome at 5th/95th
q05 = geo_df['oneqsepsrate'].quantile(0.05)
q95 = geo_df['oneqsepsrate'].quantile(0.95)
df_trim5 = geo_df[(geo_df['oneqsepsrate'] >= q05) & (geo_df['oneqsepsrate'] <= q95)].copy()
run_spec(
    "rc/sample/trim_rate_5_95",
    "modules/robustness/sample.md#outliers-and-influential-observations", "G1",
    "oneqsepsrate", "time_trend", [],
    "state_str", "state", df_trim5,
    {"CRV1": "state_str"},
    f"Trim oneqsepsrate [5%,95%], N={len(df_trim5)}", "none",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/trim_rate_5_95", "axis": "outliers",
                "rule": "trim", "params": {"lower_q": 0.05, "upper_q": 0.95}})


# ============================================================
# RC: STATE LOO — Drop one state at a time
# ============================================================

print("Running state LOO variants...")

for state_fips, state_label in STATE_LABELS.items():
    df_loo = geo_df[geo_df['geography'] != state_fips].copy()
    run_spec(
        f"rc/sample/loo_state/{state_label}",
        "modules/robustness/sample.md#leave-one-out-sample", "G1",
        "oneqsepsrate", "time_trend", [],
        "state_str", "state", df_loo,
        {"CRV1": "state_str"},
        f"Drop {state_label}, N={len(df_loo)}", "none",
        axis_block_name="sample",
        axis_block={"spec_id": f"rc/sample/loo_state/{state_label}", "axis": "loo_state",
                    "dropped_state": state_label, "dropped_fips": state_fips})


# ============================================================
# RC: FUNCTIONAL FORM
# ============================================================

print("Running functional form variants...")

# Log outcome
run_spec(
    "rc/form/log_outcome",
    "modules/robustness/functional_form.md#outcome-transformations", "G1",
    "log_oneqsepsrate", "time_trend", [],
    "state_str", "state", geo_df,
    {"CRV1": "state_str"},
    "30 states, 1999Q1-2017Q4", "log(oneqsepsrate)",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/log_outcome", "transformation": "log"})

# First-difference
fd_data = geo_df.dropna(subset=['fd_oneqsepsrate']).copy()
run_spec(
    "rc/form/first_difference",
    "modules/robustness/functional_form.md#first-differences", "G1",
    "fd_oneqsepsrate", "time_trend", [],
    "state_str", "state", fd_data,
    {"CRV1": "state_str"},
    "30 states, first-differenced", "FD(oneqsepsrate)",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/first_difference", "transformation": "first_difference"})


# ============================================================
# RC: PANEL DISAGGREGATION — Sex-Age cells
# ============================================================

print("Loading and running sex-age panel...")

try:
    sa_raw = load_sex_age_panel()
    sa_raw['time'] = sa_raw['year'] * 4 + sa_raw['quarter']

    # Aggregate over geography within sex-age-quarter cells (like the do-file)
    sa_agg = sa_raw.groupby(['sex', 'agegrp', 'year', 'quarter', 'time']).agg({
        'emp': 'sum', 'empend': 'sum', 'emptotal': 'sum',
        'hira': 'sum', 'hiraend': 'sum', 'hiras': 'sum',
        'sep': 'sum', 'sepbeg': 'sum'
    }).reset_index()

    sa_agg['cellid'] = sa_agg['sex'].astype(str) + '_' + sa_agg['agegrp'].astype(str)
    sa_panel = prepare_panel(sa_agg, 'cellid')
    sa_panel['cell_str'] = sa_panel['cellid']

    run_spec(
        "rc/panel/sex_age",
        "modules/robustness/sample.md#disaggregation", "G1",
        "oneqsepsrate", "time_trend", [],
        "cell_str", "sex-age cell", sa_panel,
        {"CRV1": "cell_str"},
        f"Sex-age cells, N={len(sa_panel.dropna(subset=['oneqsepsrate']))}", "sex-age panel",
        cluster_var="cell_str",
        axis_block_name="panel_disaggregation",
        axis_block={"spec_id": "rc/panel/sex_age", "disaggregation": "sex_age"})

    # Also with quarter FE
    run_spec(
        "rc/panel/sex_age_quarter_fe",
        "modules/robustness/sample.md#disaggregation", "G1",
        "oneqsepsrate", "time_trend", [],
        "cell_str + quarter_str", "sex-age cell + quarter", sa_panel,
        {"CRV1": "cell_str"},
        f"Sex-age cells + quarter FE, N={len(sa_panel.dropna(subset=['oneqsepsrate']))}", "sex-age panel + quarter FE",
        cluster_var="cell_str",
        axis_block_name="panel_disaggregation",
        axis_block={"spec_id": "rc/panel/sex_age_quarter_fe", "disaggregation": "sex_age",
                    "extra_fe": "quarter"})

except Exception as e:
    print(f"  Sex-age panel failed: {e}")


# ============================================================
# RC: PANEL DISAGGREGATION — Industry cells
# ============================================================

print("Loading and running industry panel...")

try:
    ind_raw = load_industry_panel()
    ind_raw['time'] = ind_raw['year'] * 4 + ind_raw['quarter']

    # Aggregate over geography within industry-quarter cells
    ind_agg = ind_raw.groupby(['industry', 'year', 'quarter', 'time']).agg({
        'emp': 'sum', 'empend': 'sum', 'emptotal': 'sum',
        'hira': 'sum', 'hiraend': 'sum', 'hiras': 'sum',
        'sep': 'sum', 'sepbeg': 'sum'
    }).reset_index()

    ind_panel = prepare_panel(ind_agg, 'industry')
    ind_panel['ind_str'] = ind_panel['industry'].astype(str)

    run_spec(
        "rc/panel/industry",
        "modules/robustness/sample.md#disaggregation", "G1",
        "oneqsepsrate", "time_trend", [],
        "ind_str", "industry", ind_panel,
        {"CRV1": "ind_str"},
        f"Industry cells, N={len(ind_panel.dropna(subset=['oneqsepsrate']))}", "industry panel",
        cluster_var="ind_str",
        axis_block_name="panel_disaggregation",
        axis_block={"spec_id": "rc/panel/industry", "disaggregation": "industry"})

    # Alternative outcomes in industry panel
    for spec_id_base, outcome, desc in [("rc/panel/industry_hirerate", "hirerate", "hire rate"),
                                         ("rc/panel/industry_sepsrate", "sepsrate", "separation rate")]:
        run_spec(
            spec_id_base,
            "modules/robustness/sample.md#disaggregation", "G1",
            outcome, "time_trend", [],
            "ind_str", "industry", ind_panel,
            {"CRV1": "ind_str"},
            f"Industry cells, outcome={desc}", f"industry panel, {desc}",
            cluster_var="ind_str",
            axis_block_name="panel_disaggregation",
            axis_block={"spec_id": spec_id_base, "disaggregation": "industry",
                        "outcome_var": outcome})

except Exception as e:
    print(f"  Industry panel failed: {e}")


# ============================================================
# RC: WEIGHTING — Population-weighted regression
# ============================================================

print("Running weighted regression variants...")

# Weight by total employment
geo_df['weight_emptot'] = geo_df['emptotal']
try:
    m_wt = pf.feols("oneqsepsrate ~ time_trend | state_str",
                     data=geo_df.dropna(subset=['oneqsepsrate','weight_emptot']),
                     vcov={"CRV1": "state_str"},
                     weights="weight_emptot")
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"
    coef_val = float(m_wt.coef()['time_trend'])
    se_val = float(m_wt.se()['time_trend'])
    pval = float(m_wt.pvalue()['time_trend'])
    try:
        ci = m_wt.confint()
        ci_lower = float(ci.loc['time_trend', ci.columns[0]])
        ci_upper = float(ci.loc['time_trend', ci.columns[1]])
    except:
        ci_lower = np.nan
        ci_upper = np.nan
    nobs = int(m_wt._N)
    r2 = float(m_wt._r2) if hasattr(m_wt, '_r2') else np.nan

    payload = make_success_payload(
        coefficients={k: float(v) for k, v in m_wt.coef().items()},
        inference={"spec_id": inference_canonical["spec_id"], "method": "cluster", "cluster_vars": ["state"]},
        software=SW_BLOCK, surface_hash=SURFACE_HASH,
        design={"panel_time_series_ols": design_audit},
        axis_block_name="weights",
        axis_block={"spec_id": "rc/weight/employment", "weight_var": "emptotal"})

    results.append({
        "paper_id": PAPER_ID, "spec_run_id": run_id,
        "spec_id": "rc/weight/employment",
        "spec_tree_path": "modules/robustness/weights.md",
        "baseline_group_id": "G1",
        "outcome_var": "oneqsepsrate", "treatment_var": "time_trend",
        "coefficient": coef_val, "std_error": se_val, "p_value": pval,
        "ci_lower": ci_lower, "ci_upper": ci_upper,
        "n_obs": nobs, "r_squared": r2,
        "coefficient_vector_json": json.dumps(payload),
        "sample_desc": "30 states, employment-weighted",
        "fixed_effects": "state", "controls_desc": "employment-weighted",
        "cluster_var": "state_str", "run_success": 1, "run_error": ""
    })
    print(f"  Weighted: coef={coef_val:.8f}, N={nobs}")
except Exception as e:
    print(f"  Weighted regression failed: {e}")


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

        est_data = data.dropna(subset=['oneqsepsrate', 'time_trend']).copy()
        m = pf.feols(full_formula, data=est_data, vcov=vcov)

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
            design={"panel_time_series_ols": design_audit},
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
baseline_formula = "oneqsepsrate ~ time_trend"

# HC1 robust (no clustering)
run_inference_variant(
    baseline_run_id, "infer/se/hc/hc1",
    "modules/inference/standard_errors.md#heteroskedasticity-robust", "G1",
    baseline_formula, "state_str", geo_df, "time_trend",
    "hetero", "HC1 (robust, no clustering)")

# Two-way clustering by state and quarter
run_inference_variant(
    baseline_run_id, "infer/se/twoway/state_quarter",
    "modules/inference/standard_errors.md#twoway-clustering", "G1",
    baseline_formula, "state_str", geo_df, "time_trend",
    {"CRV1": "state_str + quarter_str"}, "twoway(state, quarter)")

# Newey-West HAC (via iid -- fallback)
run_inference_variant(
    baseline_run_id, "infer/se/iid",
    "modules/inference/standard_errors.md#iid", "G1",
    baseline_formula, "state_str", geo_df, "time_trend",
    "iid", "iid (no clustering, no robust)")


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
        print(f"\nBaseline coef on time_trend: {base_row['coefficient'].values[0]:.8f}")
        print(f"Baseline SE: {base_row['std_error'].values[0]:.8f}")
        print(f"Baseline p-value: {base_row['p_value'].values[0]:.6f}")
        print(f"Baseline N: {base_row['n_obs'].values[0]:.0f}")

    print(f"\n=== COEFFICIENT RANGE (successful specs) ===")
    print(f"Min coef: {successful['coefficient'].min():.8f}")
    print(f"Max coef: {successful['coefficient'].max():.8f}")
    print(f"Median coef: {successful['coefficient'].median():.8f}")
    n_sig = (successful['p_value'] < 0.05).sum()
    print(f"Significant at 5%: {n_sig}/{len(successful)}")
    n_sig10 = (successful['p_value'] < 0.10).sum()
    print(f"Significant at 10%: {n_sig10}/{len(successful)}")


# ============================================================
# SPECIFICATION_SEARCH.md
# ============================================================

print("\nWriting SPECIFICATION_SEARCH.md...")

md_lines = []
md_lines.append("# Specification Search Report: 120568-V1")
md_lines.append("")
md_lines.append("**Paper:** Pries & Rogerson (2020), \"Declining Worker Turnover: the Role of Short Duration Employment Spells\", AEJ: Macroeconomics 12(1)")
md_lines.append("")
md_lines.append("## Baseline Specification")
md_lines.append("")
md_lines.append("- **Design:** Panel time-series OLS")
md_lines.append("- **Outcome:** oneqsepsrate (one-quarter separation rate)")
md_lines.append("- **Treatment:** time_trend (linear quarterly trend, 1999Q1-2017Q4)")
md_lines.append("- **Controls:** None (linear trend only)")
md_lines.append("- **Fixed effects:** state (30 US states)")
md_lines.append("- **Clustering:** state")
md_lines.append("- **Data:** QWI quarterly data for 30 states, private sector")
md_lines.append("")

if len(successful) > 0 and len(base_row) > 0:
    bc = base_row.iloc[0]
    md_lines.append(f"| Statistic | Value |")
    md_lines.append(f"|-----------|-------|")
    md_lines.append(f"| Coefficient | {bc['coefficient']:.8f} |")
    md_lines.append(f"| Std. Error | {bc['std_error']:.8f} |")
    md_lines.append(f"| p-value | {bc['p_value']:.6f} |")
    md_lines.append(f"| 95% CI | [{bc['ci_lower']:.8f}, {bc['ci_upper']:.8f}] |")
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
    "Controls/FE": successful[successful['spec_id'].str.match(r'^rc/(controls|fe)/')],
    "Sample Restrictions": successful[successful['spec_id'].str.startswith('rc/sample/')],
    "Panel Disaggregation": successful[successful['spec_id'].str.startswith('rc/panel/')],
    "Functional Form": successful[successful['spec_id'].str.startswith('rc/form/')],
    "Weighting": successful[successful['spec_id'].str.startswith('rc/weight/')],
}

for cat_name, cat_df in categories.items():
    if len(cat_df) > 0:
        n_sig_cat = (cat_df['p_value'] < 0.05).sum()
        coef_range = f"[{cat_df['coefficient'].min():.6f}, {cat_df['coefficient'].max():.6f}]"
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
            md_lines.append(f"| {row['spec_id']} | {row['std_error']:.8f} | {row['p_value']:.6f} | [{row['ci_lower']:.8f}, {row['ci_upper']:.8f}] |")
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
    md_lines.append(f"- **Direction:** Median coefficient is {sign_word} ({median_coef:.8f})")

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
