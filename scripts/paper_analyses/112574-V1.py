"""
Specification Search Script for Kuziemko & Werker (2006)
"How Much Is a Seat on the Security Council Worth? Foreign Aid and Bribery at the United Nations"
Journal of Development Economics, 2006.

Paper ID: 112574-V1

Surface-driven execution:
  - G1: oda ~ unvotes + i_elecex + p_unvotes_elecex (Main Table, Col IV baseline)
  - Panel FE (within-transformation) with pair + year/donor-year FE
  - 3-way clustering (Cameron-Gelbach-Miller) on donor, recipient, year
  - 50+ specifications across FE structure, controls, sample, outcome form, treatment decomposition

Outputs:
  - specification_results.csv (baseline, rc/* rows)
  - inference_results.csv (infer/* rows)
  - exploration_results.csv (explore/* rows)
  - SPECIFICATION_SEARCH.md
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import sys
import warnings
from scipy.stats import norm

warnings.filterwarnings('ignore')

# Add scripts dir for utilities
sys.path.insert(0, "scripts")
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash, software_block
)

PAPER_ID = "112574-V1"
DATA_DIR = "data/downloads/extracted/112574-V1"
OUTPUT_DIR = DATA_DIR  # outputs go to top-level package directory
DATA_PATH = f"{DATA_DIR}/data_analysis/data/111102_oda_final_data_big5_commit_080107_unvotes_term.dta"

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

design_audit = surface_obj["baseline_groups"][0]["design_audit"]
inference_canonical = surface_obj["baseline_groups"][0]["inference_plan"]["canonical"]

# ============================================================
# Data Loading and Preparation
# ============================================================

df_raw = pd.read_stata(DATA_PATH)

# Convert float32 to float64 to avoid precision issues
for col in df_raw.columns:
    if df_raw[col].dtype == np.float32:
        df_raw[col] = df_raw[col].astype(np.float64)

# Create pair and grouping variables
df_raw['pair'] = df_raw['wbcode_donor'] + '_' + df_raw['wbcode_recipient']
df_raw['year'] = df_raw['year'].astype(int)
df_raw['d'] = df_raw['wbcode_donor']
df_raw['r'] = df_raw['wbcode_recipient']
df_raw['donor_year'] = df_raw['d'] + '_' + df_raw['year'].astype(str)

# Create clustering interaction variables
df_raw['dr'] = df_raw['d'] + '_' + df_raw['r']  # same as pair
df_raw['dy'] = df_raw['d'] + '_' + df_raw['year'].astype(str)
df_raw['ry'] = df_raw['r'] + '_' + df_raw['year'].astype(str)


# ============================================================
# estsample function: drop missing + drop pairs with gaps
# ============================================================

def estsample(data, varlist):
    """Replicate the R estsample: drop NA in varlist, then drop pairs with gaps."""
    sub = data.dropna(subset=varlist).copy()
    gstats = sub.groupby('pair')['year'].agg(['min', 'max', 'count']).reset_index()
    gstats['expected'] = gstats['max'] - gstats['min'] + 1
    balanced = gstats[gstats['count'] == gstats['expected']]['pair']
    return sub[sub['pair'].isin(balanced)].copy()


# ============================================================
# Prepare samples
# ============================================================

# Base sample (no controls needed)
base_vars = ['unvotes', 'i_elecex', 'p_unvotes_elecex', 'unvotes_rt', 'unvotes_resid',
             'p_unvotes_rt_elecex', 'p_unvotes_resid_elecex', 'oda']
data_base = estsample(df_raw, base_vars)

# Controls sample (macro controls available)
ctrl_vars = base_vars + ['gdp2000', 'pop', 'gdp2000_donor', 'pop_donor']
data_ctrl = estsample(df_raw, ctrl_vars)

# EIEC competitiveness sample
eiec_vars = ['unvotes', 'i_elecex', 'p_unvotes_elecex', 'oda', 'i_far_eiec']
data_eiec = estsample(df_raw, eiec_vars)
# Drop pairs with only one observation (as in R code)
pair_counts = data_eiec.groupby('pair')['year'].transform('count')
data_eiec = data_eiec[pair_counts > 1].copy()

# PCT competitiveness sample (more missing)
pct_vars = ['unvotes', 'i_elecex', 'p_unvotes_elecex', 'oda', 'i_far_pct']
data_pct = estsample(df_raw, pct_vars)
pair_counts = data_pct.groupby('pair')['year'].transform('count')
data_pct = data_pct[pair_counts > 1].copy()

# Early/late election sample
df_raw['dateexec_num'] = pd.to_numeric(df_raw['dateexec'], errors='coerce')
df_raw.loc[df_raw['dateexec_num'] == 13, 'dateexec_num'] = np.nan
df_raw['elecearly4'] = (df_raw['dateexec_num'] < 4).astype(float)
df_raw['i_earlyelec'] = df_raw['i_elecex'] * df_raw['elecearly4']
df_raw.loc[df_raw['i_elecex'] == 0, 'i_earlyelec'] = 0
df_raw['i_lateelec'] = df_raw['i_elecex'] * (1 - df_raw['elecearly4'])
df_raw.loc[df_raw['i_elecex'] == 0, 'i_lateelec'] = 0
df_raw['p_unvotes_earlyelec'] = df_raw['i_earlyelec'] * df_raw['unvotes']
df_raw['p_unvotes_lateelec'] = df_raw['i_lateelec'] * df_raw['unvotes']
# Flag series missing a time for their election
df_raw['time_missing'] = ((df_raw['dateexec_num'].isna()) & (df_raw['i_elecex'] == 1)).astype(float)
df_raw['anytimemissing'] = df_raw.groupby('pair')['time_missing'].transform('max')

time_vars = ['unvotes', 'i_earlyelec', 'i_lateelec', 'p_unvotes_earlyelec', 'p_unvotes_lateelec', 'oda']
data_time = estsample(df_raw[df_raw['anytimemissing'] == 0], time_vars)


# ============================================================
# Create competitiveness interaction variables
# ============================================================

def add_eiec_vars(data):
    """Add EIEC competitiveness interactions."""
    data = data.copy()
    data['noncomp'] = data['i_far_eiec']
    data['i_elecex_comp'] = data['i_elecex'] * (1 - data['noncomp'])
    data['i_elecex_noncomp'] = data['i_elecex'] * data['noncomp']
    data['p_unvotes_elecex_comp'] = data['unvotes'] * data['i_elecex_comp']
    data['p_unvotes_elecex_noncomp'] = data['unvotes'] * data['i_elecex_noncomp']
    data['p_unvotes_noncomp'] = data['unvotes'] * data['noncomp']
    return data


def add_pct_vars(data):
    """Add PCT competitiveness interactions."""
    data = data.copy()
    data['noncomp'] = data['i_far_pct']
    data['i_elecex_comp'] = data['i_elecex'] * (1 - data['noncomp'])
    data['i_elecex_noncomp'] = data['i_elecex'] * data['noncomp']
    data['p_unvotes_elecex_comp'] = data['unvotes'] * data['i_elecex_comp']
    data['p_unvotes_elecex_noncomp'] = data['unvotes'] * data['i_elecex_noncomp']
    data['p_unvotes_noncomp'] = data['unvotes'] * data['noncomp']
    return data


data_eiec = add_eiec_vars(data_eiec)
data_pct = add_pct_vars(data_pct)

# EIEC + controls sample
eiec_ctrl_vars = eiec_vars + ['gdp2000', 'pop', 'gdp2000_donor', 'pop_donor']
data_eiec_ctrl = estsample(df_raw, eiec_ctrl_vars)
pair_counts = data_eiec_ctrl.groupby('pair')['year'].transform('count')
data_eiec_ctrl = data_eiec_ctrl[pair_counts > 1].copy()
data_eiec_ctrl = add_eiec_vars(data_eiec_ctrl)

# PCT + controls sample
pct_ctrl_vars = pct_vars + ['gdp2000', 'pop', 'gdp2000_donor', 'pop_donor']
data_pct_ctrl = estsample(df_raw, pct_ctrl_vars)
pair_counts = data_pct_ctrl.groupby('pair')['year'].transform('count')
data_pct_ctrl = data_pct_ctrl[pair_counts > 1].copy()
data_pct_ctrl = add_pct_vars(data_pct_ctrl)

# Sample restrictions
data_no_big3 = data_base[~data_base['r'].isin(['EGY', 'IDN', 'IND'])].copy()
data_no_big5 = data_base[~data_base['r'].isin(['EGY', 'IDN', 'IND', 'ISR', 'CHN'])].copy()
data_us_only = data_base[data_base['d'] == 'USA'].copy()

data_no_big3_ctrl = data_ctrl[~data_ctrl['r'].isin(['EGY', 'IDN', 'IND'])].copy()
data_no_big5_ctrl = data_ctrl[~data_ctrl['r'].isin(['EGY', 'IDN', 'IND', 'ISR', 'CHN'])].copy()

# Log ODA datasets
def make_log_oda(data, log_controls=False):
    """Create log ODA version of data."""
    d = data.copy()
    d['oda_orig'] = d['oda']
    d['oda'] = np.log(d['oda'].clip(lower=1e-6))
    d.loc[d['oda'].isna() | np.isinf(d['oda']), 'oda'] = np.log(1/1000000)
    if log_controls and all(v in d.columns for v in ['pop', 'gdp2000', 'pop_donor', 'gdp2000_donor']):
        for v in ['pop', 'gdp2000', 'pop_donor', 'gdp2000_donor']:
            d[v] = np.log(d[v].clip(lower=1e-10))
    return d


data_log_base = make_log_oda(data_base)
data_log_ctrl = make_log_oda(data_ctrl, log_controls=True)
data_log_no_big3 = make_log_oda(data_no_big3)
data_log_no_big5 = make_log_oda(data_no_big5)


# ============================================================
# 3-Way Clustering Implementation (Cameron-Gelbach-Miller 2006)
# ============================================================

def compute_3way_cluster_se(data, formula, fe_formula, focal_var):
    """
    Compute 3-way clustered SEs using CGM (2006) formula:
    V_3way = V_d + V_r + V_y - V_dr - V_dy - V_ry + V_dry
    with D/(D-1) inflation factor for each component.

    Falls back to 2-way (recipient, year) clustering if only 1 donor.

    Returns: (model, se_3way, pval_3way) for the focal_var
    """
    full_formula = f"{formula} | {fe_formula}" if fe_formula else formula

    nd = data['d'].nunique()

    if nd <= 1:
        # Cannot do 3-way clustering with 1 donor; fall back to 2-way (recipient, year)
        m_r = pf.feols(full_formula, data=data, vcov={'CRV1': 'r'})
        m_y = pf.feols(full_formula, data=data, vcov={'CRV1': 'year'})
        m_ry = pf.feols(full_formula, data=data, vcov={'CRV1': 'ry'})

        nr = data['r'].nunique()
        ny = data['year'].nunique()
        nry = data['ry'].nunique()

        def v(model, var):
            return model.se()[var] ** 2

        var_2way = (
            (nr / (nr - 1)) * v(m_r, focal_var)
            + (ny / (ny - 1)) * v(m_y, focal_var)
            - (nry / (nry - 1)) * v(m_ry, focal_var)
        )
        if var_2way < 0:
            var_2way = 0.0
        se = np.sqrt(var_2way)
        coef = float(m_r.coef()[focal_var])
        pval = 2 * (1 - norm.cdf(abs(coef / se))) if se > 0 else np.nan
        return m_r, se, pval

    # Fit model with each cluster level
    m_d = pf.feols(full_formula, data=data, vcov={'CRV1': 'd'})
    m_r = pf.feols(full_formula, data=data, vcov={'CRV1': 'r'})
    m_y = pf.feols(full_formula, data=data, vcov={'CRV1': 'year'})
    m_dr = pf.feols(full_formula, data=data, vcov={'CRV1': 'pair'})
    m_dy = pf.feols(full_formula, data=data, vcov={'CRV1': 'dy'})
    m_ry = pf.feols(full_formula, data=data, vcov={'CRV1': 'ry'})
    m_het = pf.feols(full_formula, data=data, vcov='hetero')

    # Get variance for focal var from each model
    def v(model, var):
        return model.se()[var] ** 2

    nr = data['r'].nunique()
    ny = data['year'].nunique()
    ndr = data['pair'].nunique()
    ndy = data['dy'].nunique()
    nry = data['ry'].nunique()
    n = len(data)

    # CGM formula with inflation factors
    var_3way = (
        (nd / (nd - 1)) * v(m_d, focal_var)
        + (nr / (nr - 1)) * v(m_r, focal_var)
        + (ny / (ny - 1)) * v(m_y, focal_var)
        - (ndr / (ndr - 1)) * v(m_dr, focal_var)
        - (ndy / (ndy - 1)) * v(m_dy, focal_var)
        - (nry / (nry - 1)) * v(m_ry, focal_var)
        + (n / (n - 1)) * v(m_het, focal_var)
    )

    # Fix negative variances
    if var_3way < 0:
        var_3way = 0.0

    se_3way = np.sqrt(var_3way)
    coef = float(m_d.coef()[focal_var])
    if se_3way > 0:
        t_stat = coef / se_3way
        pval = 2 * (1 - norm.cdf(abs(t_stat)))
    else:
        pval = np.nan

    return m_d, se_3way, pval


# ============================================================
# Spec Runner
# ============================================================

results = []
inference_results = []
exploration_results = []
spec_run_counter = 0
infer_run_counter = 0
explore_run_counter = 0


def run_spec(spec_id, spec_tree_path, baseline_group_id, outcome_var, treatment_var,
             formula, fe_formula, data, focal_var,
             sample_desc, fixed_effects_str, controls_desc, cluster_var="donor, recipient, year",
             axis_block_name=None, axis_block=None, design_override=None):
    """Run a single specification with 3-way clustered SEs."""
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        full_formula = f"{formula} | {fe_formula}" if fe_formula else formula

        # Fit model with hetero SE first (for coefficient extraction)
        m = pf.feols(full_formula, data=data, vcov='hetero')

        coef_val = float(m.coef().get(focal_var, np.nan))
        nobs = int(m._N)
        try:
            r2 = float(m._r2)
        except Exception:
            r2 = np.nan

        all_coefs = {k: float(v) for k, v in m.coef().items()}

        # Compute 3-way clustered SE for the focal variable
        _, se_3way, pval_3way = compute_3way_cluster_se(data, formula, fe_formula, focal_var)

        # CI from normal distribution
        if se_3way > 0 and not np.isnan(se_3way):
            ci_lower = coef_val - 1.96 * se_3way
            ci_upper = coef_val + 1.96 * se_3way
        else:
            ci_lower = np.nan
            ci_upper = np.nan

        design_block = design_override or {"panel_fixed_effects": design_audit}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": inference_canonical["spec_id"],
                       "params": inference_canonical["params"],
                       "method": "CGM_3way_cluster",
                       "cluster_vars": ["donor", "recipient", "year"]},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design=design_block,
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
            "std_error": se_3way,
            "p_value": pval_3way,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": nobs,
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": fixed_effects_str,
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
            "run_success": 1,
            "run_error": ""
        })
        return run_id, coef_val, se_3way, pval_3way, nobs

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
            "fixed_effects": fixed_effects_str,
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
            "run_success": 0,
            "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


def run_explore(spec_id, spec_tree_path, baseline_group_id, outcome_var, treatment_var,
                formula, fe_formula, data, focal_var,
                sample_desc, fixed_effects_str, controls_desc, cluster_var="donor, recipient, year",
                axis_block_name=None, axis_block=None, extra_coefs_to_report=None):
    """Run an exploration specification (explore/*) with 3-way clustered SEs."""
    global explore_run_counter
    explore_run_counter += 1
    run_id = f"{PAPER_ID}_explore_{explore_run_counter:03d}"

    try:
        full_formula = f"{formula} | {fe_formula}" if fe_formula else formula
        m = pf.feols(full_formula, data=data, vcov='hetero')

        coef_val = float(m.coef().get(focal_var, np.nan))
        nobs = int(m._N)
        try:
            r2 = float(m._r2)
        except Exception:
            r2 = np.nan

        all_coefs = {k: float(v) for k, v in m.coef().items()}

        _, se_3way, pval_3way = compute_3way_cluster_se(data, formula, fe_formula, focal_var)

        explore_block = {
            "spec_id": spec_id,
            "focal_var": focal_var,
            "outcome_var": outcome_var,
            "treatment_var": treatment_var,
            "formula": formula,
            "fe": fe_formula,
            "sample_desc": sample_desc,
            "coefficient": coef_val,
            "std_error": se_3way,
            "p_value": pval_3way,
            "n_obs": nobs,
            "r_squared": r2,
        }
        if extra_coefs_to_report:
            for evar in extra_coefs_to_report:
                if evar in m.coef():
                    explore_block[f"coef_{evar}"] = float(m.coef()[evar])

        payload = {
            "exploration": explore_block,
            "coefficients": all_coefs,
            "software": SW_BLOCK,
            "surface_hash": SURFACE_HASH,
        }
        if axis_block_name and axis_block:
            payload[axis_block_name] = axis_block

        exploration_results.append({
            "paper_id": PAPER_ID,
            "exploration_run_id": run_id,
            "spec_run_id": "",
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "exploration_json": json.dumps(payload),
            "run_success": 1,
            "run_error": ""
        })
        return run_id

    except Exception as e:
        err_msg = str(e)[:240]
        err_details = error_details_from_exception(e, stage="estimation")
        payload = {
            "error": err_msg,
            "error_details": err_details,
            "software": SW_BLOCK,
            "surface_hash": SURFACE_HASH,
        }
        exploration_results.append({
            "paper_id": PAPER_ID,
            "exploration_run_id": run_id,
            "spec_run_id": "",
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "exploration_json": json.dumps(payload),
            "run_success": 0,
            "run_error": err_msg
        })
        return run_id


def run_inference_variant(base_run_id, spec_id, spec_tree_path, baseline_group_id,
                          formula, fe_formula, data, focal_var,
                          vcov_spec, cluster_var_desc):
    """Run an inference variant on a given spec."""
    global infer_run_counter
    infer_run_counter += 1
    run_id = f"{PAPER_ID}_infer_{infer_run_counter:03d}"

    try:
        full_formula = f"{formula} | {fe_formula}" if fe_formula else formula
        m = pf.feols(full_formula, data=data, vcov=vcov_spec)

        coef_val = float(m.coef().get(focal_var, np.nan))
        se_val = float(m.se().get(focal_var, np.nan))
        pval = float(m.pvalue().get(focal_var, np.nan))
        nobs = int(m._N)
        try:
            r2 = float(m._r2)
        except Exception:
            r2 = np.nan
        try:
            ci = m.confint()
            ci_lower = float(ci.loc[focal_var, ci.columns[0]]) if focal_var in ci.index else np.nan
            ci_upper = float(ci.loc[focal_var, ci.columns[1]]) if focal_var in ci.index else np.nan
        except Exception:
            ci_lower = np.nan
            ci_upper = np.nan

        all_coefs = {k: float(v) for k, v in m.coef().items()}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": spec_id, "params": {"cluster_var": cluster_var_desc}},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"panel_fixed_effects": design_audit},
        )

        inference_results.append({
            "paper_id": PAPER_ID,
            "inference_run_id": run_id,
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
            "cluster_var": cluster_var_desc,
            "run_success": 1,
            "run_error": ""
        })
        return run_id

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
            "inference_run_id": run_id,
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
            "cluster_var": cluster_var_desc,
            "run_success": 0,
            "run_error": err_msg
        })
        return run_id


# ============================================================
# BASELINE SPECIFICATIONS (Main Table Cols IV, V, VI, VII, VIII, IX)
# ============================================================

print("Running baseline specifications...")

# Col IV: pair + year FE, no controls (PRIMARY BASELINE)
base_run_id, _, _, _, _ = run_spec(
    "baseline", "designs/panel_fixed_effects.md#baseline", "G1",
    "oda", "p_unvotes_elecex",
    "oda ~ unvotes + i_elecex + p_unvotes_elecex", "pair + year",
    data_base, "p_unvotes_elecex",
    f"Big 5 donors x all recipients, 1975-2004, N={len(data_base)}", "pair + year", "none"
)

# Col V: pair + donor_year FE
run_spec(
    "baseline__main_colV", "designs/panel_fixed_effects.md#baseline", "G1",
    "oda", "p_unvotes_elecex",
    "oda ~ unvotes + i_elecex + p_unvotes_elecex", "pair + donor_year",
    data_base, "p_unvotes_elecex",
    f"Big 5, N={len(data_base)}", "pair + donor_year", "none"
)

# Col VI: pair FE + macro controls
run_spec(
    "baseline__main_colVI", "designs/panel_fixed_effects.md#baseline", "G1",
    "oda", "p_unvotes_elecex",
    "oda ~ unvotes + i_elecex + p_unvotes_elecex + pop + gdp2000 + pop_donor + gdp2000_donor", "pair",
    data_ctrl, "p_unvotes_elecex",
    f"Big 5 w/ controls, N={len(data_ctrl)}", "pair", "pop + gdp2000 + pop_donor + gdp2000_donor"
)

# Col VII: decomposed UN votes, pair + year FE
run_spec(
    "baseline__main_colVII", "designs/panel_fixed_effects.md#baseline", "G1",
    "oda", "p_unvotes_rt_elecex",
    "oda ~ unvotes_rt + unvotes_resid + i_elecex + p_unvotes_rt_elecex + p_unvotes_resid_elecex",
    "pair + year", data_base, "p_unvotes_rt_elecex",
    f"Big 5, decomposed UN votes, N={len(data_base)}", "pair + year", "none"
)

# Col VIII: decomposed UN votes, pair + donor_year FE
run_spec(
    "baseline__main_colVIII", "designs/panel_fixed_effects.md#baseline", "G1",
    "oda", "p_unvotes_rt_elecex",
    "oda ~ unvotes_rt + unvotes_resid + i_elecex + p_unvotes_rt_elecex + p_unvotes_resid_elecex",
    "pair + donor_year", data_base, "p_unvotes_rt_elecex",
    f"Big 5, decomposed, N={len(data_base)}", "pair + donor_year", "none"
)

# Col IX: decomposed UN votes, pair FE + controls
run_spec(
    "baseline__main_colIX", "designs/panel_fixed_effects.md#baseline", "G1",
    "oda", "p_unvotes_rt_elecex",
    "oda ~ unvotes_rt + unvotes_resid + i_elecex + p_unvotes_rt_elecex + p_unvotes_resid_elecex + pop + gdp2000 + pop_donor + gdp2000_donor",
    "pair", data_ctrl, "p_unvotes_rt_elecex",
    f"Big 5, decomposed w/ controls, N={len(data_ctrl)}", "pair",
    "pop + gdp2000 + pop_donor + gdp2000_donor"
)


# ============================================================
# RC: FE STRUCTURE VARIANTS
# ============================================================

print("Running FE structure variants...")

# rc/fe/pair_plus_year (same as baseline, but explicit as RC for cross-product)
run_spec(
    "rc/fe/pair_plus_year", "modules/robustness/fixed_effects.md#fe-structure", "G1",
    "oda", "p_unvotes_elecex",
    "oda ~ unvotes + i_elecex + p_unvotes_elecex", "pair + year",
    data_base, "p_unvotes_elecex",
    f"Big 5, N={len(data_base)}", "pair + year", "none",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/pair_plus_year", "fe_structure": ["pair", "year"]}
)

# rc/fe/pair_plus_donor_year
run_spec(
    "rc/fe/pair_plus_donor_year", "modules/robustness/fixed_effects.md#fe-structure", "G1",
    "oda", "p_unvotes_elecex",
    "oda ~ unvotes + i_elecex + p_unvotes_elecex", "pair + donor_year",
    data_base, "p_unvotes_elecex",
    f"Big 5, N={len(data_base)}", "pair + donor_year", "none",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/pair_plus_donor_year", "fe_structure": ["pair", "donor_year"]}
)

# rc/fe/pair_only_with_controls
run_spec(
    "rc/fe/pair_only_with_controls", "modules/robustness/fixed_effects.md#fe-structure", "G1",
    "oda", "p_unvotes_elecex",
    "oda ~ unvotes + i_elecex + p_unvotes_elecex + pop + gdp2000 + pop_donor + gdp2000_donor", "pair",
    data_ctrl, "p_unvotes_elecex",
    f"Big 5 w/ controls, N={len(data_ctrl)}", "pair",
    "pop + gdp2000 + pop_donor + gdp2000_donor",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/pair_only_with_controls", "fe_structure": ["pair"],
                "notes": "No time FE; time variation captured by macro controls"}
)

# rc/fe/pair_plus_recipient_year (novel robustness check)
# Need recipient_year FE variable
data_base_ry = data_base.copy()
data_base_ry['recipient_year'] = data_base_ry['r'] + '_' + data_base_ry['year'].astype(str)
run_spec(
    "rc/fe/pair_plus_recipient_year", "modules/robustness/fixed_effects.md#fe-structure", "G1",
    "oda", "p_unvotes_elecex",
    "oda ~ unvotes + i_elecex + p_unvotes_elecex", "pair + recipient_year",
    data_base_ry, "p_unvotes_elecex",
    f"Big 5, N={len(data_base_ry)}", "pair + recipient_year", "none",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/pair_plus_recipient_year", "fe_structure": ["pair", "recipient_year"],
                "notes": "Absorbs all recipient-year variation; stronger than baseline"}
)


# ============================================================
# RC: CONTROLS
# ============================================================

print("Running controls variants...")

# rc/controls/add_macro_block (pair + year FE + controls)
run_spec(
    "rc/controls/add_macro_block", "modules/robustness/controls.md#add-controls", "G1",
    "oda", "p_unvotes_elecex",
    "oda ~ unvotes + i_elecex + p_unvotes_elecex + pop + gdp2000 + pop_donor + gdp2000_donor", "pair + year",
    data_ctrl, "p_unvotes_elecex",
    f"Big 5 w/ controls, N={len(data_ctrl)}", "pair + year",
    "pop + gdp2000 + pop_donor + gdp2000_donor",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/add_macro_block", "family": "add_block",
                "added": ["pop", "gdp2000", "pop_donor", "gdp2000_donor"]}
)

# LOO controls (from control sample with pair FE only)
for ctrl_name in ['pop', 'gdp2000', 'pop_donor', 'gdp2000_donor']:
    kept = [c for c in ['pop', 'gdp2000', 'pop_donor', 'gdp2000_donor'] if c != ctrl_name]
    ctrl_str = " + ".join(kept)
    run_spec(
        f"rc/controls/loo/drop_{ctrl_name}", "modules/robustness/controls.md#leave-one-out-controls-loo", "G1",
        "oda", "p_unvotes_elecex",
        f"oda ~ unvotes + i_elecex + p_unvotes_elecex + {ctrl_str}", "pair",
        data_ctrl, "p_unvotes_elecex",
        f"Big 5, LOO drop {ctrl_name}, N={len(data_ctrl)}", "pair", ctrl_str,
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/controls/loo/drop_{ctrl_name}", "family": "loo",
                    "dropped": [ctrl_name], "n_controls": 3}
    )


# ============================================================
# RC: SAMPLE RESTRICTIONS
# ============================================================

print("Running sample restriction variants...")

# Drop Big 3 (pair + year FE)
run_spec(
    "rc/sample/subset/drop_big3_recipients", "modules/robustness/sample.md#sample-restrictions", "G1",
    "oda", "p_unvotes_elecex",
    "oda ~ unvotes + i_elecex + p_unvotes_elecex", "pair + year",
    data_no_big3, "p_unvotes_elecex",
    f"Big 5 excl EGY/IDN/IND, N={len(data_no_big3)}", "pair + year", "none",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subset/drop_big3_recipients", "family": "subset",
                "dropped_recipients": ["EGY", "IDN", "IND"]}
)

# Drop Big 5
run_spec(
    "rc/sample/subset/drop_big5_recipients", "modules/robustness/sample.md#sample-restrictions", "G1",
    "oda", "p_unvotes_elecex",
    "oda ~ unvotes + i_elecex + p_unvotes_elecex", "pair + year",
    data_no_big5, "p_unvotes_elecex",
    f"Big 5 excl EGY/IDN/IND/ISR/CHN, N={len(data_no_big5)}", "pair + year", "none",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subset/drop_big5_recipients", "family": "subset",
                "dropped_recipients": ["EGY", "IDN", "IND", "ISR", "CHN"]}
)

# US only
run_spec(
    "rc/sample/subset/us_only", "modules/robustness/sample.md#sample-restrictions", "G1",
    "oda", "p_unvotes_elecex",
    "oda ~ unvotes + i_elecex + p_unvotes_elecex", "pair + year",
    data_us_only, "p_unvotes_elecex",
    f"US only, N={len(data_us_only)}", "pair + year", "none",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subset/us_only", "family": "subset",
                "restriction": "US donor only"}
)

# Balanced panel (the estsample function already enforces this, so this is same as baseline)
run_spec(
    "rc/sample/subset/balanced_panel", "modules/robustness/sample.md#sample-restrictions", "G1",
    "oda", "p_unvotes_elecex",
    "oda ~ unvotes + i_elecex + p_unvotes_elecex", "pair + year",
    data_base, "p_unvotes_elecex",
    f"Balanced panel (already enforced), N={len(data_base)}", "pair + year", "none",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subset/balanced_panel", "family": "balanced",
                "notes": "estsample already enforces balanced panel; identical to baseline"}
)


# ============================================================
# RC: FUNCTIONAL FORM (Log ODA)
# ============================================================

print("Running functional form variants...")

# Log ODA, pair + year FE
run_spec(
    "rc/form/outcome/log_oda", "modules/robustness/functional_form.md#outcome-transformation", "G1",
    "oda", "p_unvotes_elecex",
    "oda ~ unvotes + i_elecex + p_unvotes_elecex", "pair + year",
    data_log_base, "p_unvotes_elecex",
    f"Log ODA, pair+year FE, N={len(data_log_base)}", "pair + year", "none",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/log_oda", "transform": "log",
                "interpretation": "semi-elasticity: % change in ODA per unit change in UN alignment * election",
                "units": "log(ODA in USD thousands 2000 constant)"}
)

# Log ODA with log controls, pair FE only
run_spec(
    "rc/form/outcome/log_oda_with_log_controls",
    "modules/robustness/functional_form.md#outcome-transformation", "G1",
    "oda", "p_unvotes_elecex",
    "oda ~ unvotes + i_elecex + p_unvotes_elecex + pop + gdp2000 + pop_donor + gdp2000_donor", "pair",
    data_log_ctrl, "p_unvotes_elecex",
    f"Log ODA + log controls, pair FE, N={len(data_log_ctrl)}", "pair",
    "log(pop) + log(gdp2000) + log(pop_donor) + log(gdp2000_donor)",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/log_oda_with_log_controls", "transform": "log",
                "interpretation": "semi-elasticity with log macro controls",
                "units": "log(ODA), log macro controls"}
)


# ============================================================
# RC: JOINT VARIATIONS
# ============================================================

print("Running joint specifications...")

# Log ODA + pair+year FE
run_spec(
    "rc/joint/log_oda_pair_year", "modules/robustness/joint.md#joint-variations", "G1",
    "oda", "p_unvotes_elecex",
    "oda ~ unvotes + i_elecex + p_unvotes_elecex", "pair + year",
    data_log_base, "p_unvotes_elecex",
    f"Log ODA, pair+year, N={len(data_log_base)}", "pair + year", "none",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/log_oda_pair_year",
                "axes_changed": ["functional_form", "fixed_effects"],
                "details": {"outcome": "log(oda)", "fe": "pair+year"}}
)

# Log ODA + pair + controls
run_spec(
    "rc/joint/log_oda_pair_controls", "modules/robustness/joint.md#joint-variations", "G1",
    "oda", "p_unvotes_elecex",
    "oda ~ unvotes + i_elecex + p_unvotes_elecex + pop + gdp2000 + pop_donor + gdp2000_donor", "pair",
    data_log_ctrl, "p_unvotes_elecex",
    f"Log ODA + log controls, pair FE, N={len(data_log_ctrl)}", "pair",
    "log(pop) + log(gdp2000) + log(pop_donor) + log(gdp2000_donor)",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/log_oda_pair_controls",
                "axes_changed": ["functional_form", "controls", "fixed_effects"],
                "details": {"outcome": "log(oda)", "controls": "log macro block", "fe": "pair"}}
)

# Log ODA + donor-year FE
run_spec(
    "rc/joint/log_oda_donor_year", "modules/robustness/joint.md#joint-variations", "G1",
    "oda", "p_unvotes_elecex",
    "oda ~ unvotes + i_elecex + p_unvotes_elecex", "pair + donor_year",
    data_log_base, "p_unvotes_elecex",
    f"Log ODA, pair+donor_year, N={len(data_log_base)}", "pair + donor_year", "none",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/log_oda_donor_year",
                "axes_changed": ["functional_form", "fixed_effects"],
                "details": {"outcome": "log(oda)", "fe": "pair+donor_year"}}
)

# Drop Big 3 + pair+year
run_spec(
    "rc/joint/drop_big3_pair_year", "modules/robustness/joint.md#joint-variations", "G1",
    "oda", "p_unvotes_elecex",
    "oda ~ unvotes + i_elecex + p_unvotes_elecex", "pair + year",
    data_no_big3, "p_unvotes_elecex",
    f"Excl EGY/IDN/IND, pair+year, N={len(data_no_big3)}", "pair + year", "none",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/drop_big3_pair_year",
                "axes_changed": ["sample"],
                "details": {"dropped_recipients": ["EGY", "IDN", "IND"], "fe": "pair+year"}}
)

# Drop Big 5 + pair+year
run_spec(
    "rc/joint/drop_big5_pair_year", "modules/robustness/joint.md#joint-variations", "G1",
    "oda", "p_unvotes_elecex",
    "oda ~ unvotes + i_elecex + p_unvotes_elecex", "pair + year",
    data_no_big5, "p_unvotes_elecex",
    f"Excl EGY/IDN/IND/ISR/CHN, pair+year, N={len(data_no_big5)}", "pair + year", "none",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/drop_big5_pair_year",
                "axes_changed": ["sample"],
                "details": {"dropped_recipients": ["EGY", "IDN", "IND", "ISR", "CHN"], "fe": "pair+year"}}
)

# Drop Big 3 + donor-year FE
run_spec(
    "rc/joint/drop_big3_donor_year", "modules/robustness/joint.md#joint-variations", "G1",
    "oda", "p_unvotes_elecex",
    "oda ~ unvotes + i_elecex + p_unvotes_elecex", "pair + donor_year",
    data_no_big3, "p_unvotes_elecex",
    f"Excl EGY/IDN/IND, donor-year FE, N={len(data_no_big3)}", "pair + donor_year", "none",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/drop_big3_donor_year",
                "axes_changed": ["sample", "fixed_effects"],
                "details": {"dropped_recipients": ["EGY", "IDN", "IND"], "fe": "pair+donor_year"}}
)

# Drop Big 5 + controls
run_spec(
    "rc/joint/drop_big5_controls", "modules/robustness/joint.md#joint-variations", "G1",
    "oda", "p_unvotes_elecex",
    "oda ~ unvotes + i_elecex + p_unvotes_elecex + pop + gdp2000 + pop_donor + gdp2000_donor", "pair",
    data_no_big5_ctrl, "p_unvotes_elecex",
    f"Excl Big 5 recipients, pair FE + controls, N={len(data_no_big5_ctrl)}", "pair",
    "pop + gdp2000 + pop_donor + gdp2000_donor",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/drop_big5_controls",
                "axes_changed": ["sample", "controls"],
                "details": {"dropped_recipients": ["EGY", "IDN", "IND", "ISR", "CHN"],
                            "controls": "macro block", "fe": "pair"}}
)

# Balanced panel + pair+year (same as balanced_panel since estsample already enforces it)
run_spec(
    "rc/joint/balanced_panel_pair_year", "modules/robustness/joint.md#joint-variations", "G1",
    "oda", "p_unvotes_elecex",
    "oda ~ unvotes + i_elecex + p_unvotes_elecex", "pair + year",
    data_base, "p_unvotes_elecex",
    f"Balanced panel, pair+year, N={len(data_base)}", "pair + year", "none",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/balanced_panel_pair_year",
                "axes_changed": ["sample"],
                "details": {"sample": "balanced panel (enforced by estsample)", "fe": "pair+year"}}
)

# US only + pair+year
run_spec(
    "rc/joint/us_only_pair_year", "modules/robustness/joint.md#joint-variations", "G1",
    "oda", "p_unvotes_elecex",
    "oda ~ unvotes + i_elecex + p_unvotes_elecex", "pair + year",
    data_us_only, "p_unvotes_elecex",
    f"US only, pair+year, N={len(data_us_only)}", "pair + year", "none",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/us_only_pair_year",
                "axes_changed": ["sample"],
                "details": {"restriction": "US donor only", "fe": "pair+year"}}
)

# Log ODA + Drop Big 3
run_spec(
    "rc/joint/log_oda_drop_big3", "modules/robustness/joint.md#joint-variations", "G1",
    "oda", "p_unvotes_elecex",
    "oda ~ unvotes + i_elecex + p_unvotes_elecex", "pair + year",
    data_log_no_big3, "p_unvotes_elecex",
    f"Log ODA, excl Big 3, N={len(data_log_no_big3)}", "pair + year", "none",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/log_oda_drop_big3",
                "axes_changed": ["functional_form", "sample"],
                "details": {"outcome": "log(oda)", "dropped_recipients": ["EGY", "IDN", "IND"]}}
)

# Log ODA + Drop Big 5
run_spec(
    "rc/joint/log_oda_drop_big5", "modules/robustness/joint.md#joint-variations", "G1",
    "oda", "p_unvotes_elecex",
    "oda ~ unvotes + i_elecex + p_unvotes_elecex", "pair + year",
    data_log_no_big5, "p_unvotes_elecex",
    f"Log ODA, excl Big 5, N={len(data_log_no_big5)}", "pair + year", "none",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/log_oda_drop_big5",
                "axes_changed": ["functional_form", "sample"],
                "details": {"outcome": "log(oda)", "dropped_recipients": ["EGY", "IDN", "IND", "ISR", "CHN"]}}
)

# Drop Big 5 + donor-year FE
run_spec(
    "rc/joint/drop_big5_donor_year", "modules/robustness/joint.md#joint-variations", "G1",
    "oda", "p_unvotes_elecex",
    "oda ~ unvotes + i_elecex + p_unvotes_elecex", "pair + donor_year",
    data_no_big5, "p_unvotes_elecex",
    f"Excl Big 5, donor-year FE, N={len(data_no_big5)}", "pair + donor_year", "none",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/drop_big5_donor_year",
                "axes_changed": ["sample", "fixed_effects"],
                "details": {"dropped_recipients": ["EGY", "IDN", "IND", "ISR", "CHN"], "fe": "pair+donor_year"}}
)

# Drop Big 3 + controls
run_spec(
    "rc/joint/drop_big3_controls", "modules/robustness/joint.md#joint-variations", "G1",
    "oda", "p_unvotes_elecex",
    "oda ~ unvotes + i_elecex + p_unvotes_elecex + pop + gdp2000 + pop_donor + gdp2000_donor", "pair",
    data_no_big3_ctrl, "p_unvotes_elecex",
    f"Excl Big 3, pair FE + controls, N={len(data_no_big3_ctrl)}", "pair",
    "pop + gdp2000 + pop_donor + gdp2000_donor",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/drop_big3_controls",
                "axes_changed": ["sample", "controls"],
                "details": {"dropped_recipients": ["EGY", "IDN", "IND"],
                            "controls": "macro block", "fe": "pair"}}
)

# Macro controls + donor-year FE (double absorption)
run_spec(
    "rc/joint/controls_donor_year", "modules/robustness/joint.md#joint-variations", "G1",
    "oda", "p_unvotes_elecex",
    "oda ~ unvotes + i_elecex + p_unvotes_elecex + pop + gdp2000 + pop_donor + gdp2000_donor",
    "pair + donor_year",
    data_ctrl, "p_unvotes_elecex",
    f"Controls + donor-year FE, N={len(data_ctrl)}", "pair + donor_year",
    "pop + gdp2000 + pop_donor + gdp2000_donor",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/controls_donor_year",
                "axes_changed": ["controls", "fixed_effects"],
                "details": {"controls": "macro block", "fe": "pair+donor_year"}}
)

# Log ODA + log controls + year FE
run_spec(
    "rc/joint/log_oda_log_controls_year", "modules/robustness/joint.md#joint-variations", "G1",
    "oda", "p_unvotes_elecex",
    "oda ~ unvotes + i_elecex + p_unvotes_elecex + pop + gdp2000 + pop_donor + gdp2000_donor",
    "pair + year",
    data_log_ctrl, "p_unvotes_elecex",
    f"Log ODA + log controls + year FE, N={len(data_log_ctrl)}", "pair + year",
    "log(pop) + log(gdp2000) + log(pop_donor) + log(gdp2000_donor)",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/log_oda_log_controls_year",
                "axes_changed": ["functional_form", "controls", "fixed_effects"],
                "details": {"outcome": "log(oda)", "controls": "log macro block", "fe": "pair+year"}}
)

# US-only + donor-year FE (just pair FE since 1 donor; add year FE)
run_spec(
    "rc/joint/us_only_donor_year", "modules/robustness/joint.md#joint-variations", "G1",
    "oda", "p_unvotes_elecex",
    "oda ~ unvotes + i_elecex + p_unvotes_elecex", "pair + year",
    data_us_only, "p_unvotes_elecex",
    f"US only, pair+year, N={len(data_us_only)}", "pair + year", "none",
    cluster_var="recipient, year (2-way; 1 donor)",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/us_only_donor_year",
                "axes_changed": ["sample"],
                "details": {"restriction": "US donor only", "fe": "pair+year",
                            "notes": "2-way clustering (recipient, year) since only 1 donor"}}
)

# Drop Big 3 + recipient-year FE
data_no_big3_ry = data_no_big3.copy()
data_no_big3_ry['recipient_year'] = data_no_big3_ry['r'] + '_' + data_no_big3_ry['year'].astype(str)
run_spec(
    "rc/joint/drop_big3_recip_year", "modules/robustness/joint.md#joint-variations", "G1",
    "oda", "p_unvotes_elecex",
    "oda ~ unvotes + i_elecex + p_unvotes_elecex", "pair + recipient_year",
    data_no_big3_ry, "p_unvotes_elecex",
    f"Excl Big 3, recipient-year FE, N={len(data_no_big3_ry)}", "pair + recipient_year", "none",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/drop_big3_recip_year",
                "axes_changed": ["sample", "fixed_effects"],
                "details": {"dropped_recipients": ["EGY", "IDN", "IND"], "fe": "pair+recipient_year"}}
)


# ============================================================
# RC: Additional Sample x FE and Sample x Controls combinations
# ============================================================

print("Running additional joint specifications...")

# Drop Big 3 + donor-year FE + controls  (NOTE: already have drop_big3_donor_year above)
# Let's do: drop_big3 + pair + year + controls
run_spec(
    "rc/joint/drop_big3_year_controls", "modules/robustness/joint.md#joint-variations", "G1",
    "oda", "p_unvotes_elecex",
    "oda ~ unvotes + i_elecex + p_unvotes_elecex + pop + gdp2000 + pop_donor + gdp2000_donor",
    "pair + year",
    data_no_big3_ctrl, "p_unvotes_elecex",
    f"Excl Big 3, year FE + controls, N={len(data_no_big3_ctrl)}", "pair + year",
    "pop + gdp2000 + pop_donor + gdp2000_donor",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/drop_big3_year_controls",
                "axes_changed": ["sample", "controls", "fixed_effects"],
                "details": {"dropped_recipients": ["EGY", "IDN", "IND"],
                            "controls": "macro block", "fe": "pair+year"}}
)

# Drop Big 5 + controls + year FE
run_spec(
    "rc/joint/drop_big5_year_controls", "modules/robustness/joint.md#joint-variations", "G1",
    "oda", "p_unvotes_elecex",
    "oda ~ unvotes + i_elecex + p_unvotes_elecex + pop + gdp2000 + pop_donor + gdp2000_donor",
    "pair + year",
    data_no_big5_ctrl, "p_unvotes_elecex",
    f"Excl Big 5, year FE + controls, N={len(data_no_big5_ctrl)}", "pair + year",
    "pop + gdp2000 + pop_donor + gdp2000_donor",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/drop_big5_year_controls",
                "axes_changed": ["sample", "controls", "fixed_effects"],
                "details": {"dropped_recipients": ["EGY", "IDN", "IND", "ISR", "CHN"],
                            "controls": "macro block", "fe": "pair+year"}}
)

# Log ODA + donor-year + controls  (log controls + donor-year FE)
run_spec(
    "rc/joint/log_oda_log_controls_donor_year", "modules/robustness/joint.md#joint-variations", "G1",
    "oda", "p_unvotes_elecex",
    "oda ~ unvotes + i_elecex + p_unvotes_elecex + pop + gdp2000 + pop_donor + gdp2000_donor",
    "pair + donor_year",
    data_log_ctrl, "p_unvotes_elecex",
    f"Log ODA + log controls + donor-year FE, N={len(data_log_ctrl)}", "pair + donor_year",
    "log(pop) + log(gdp2000) + log(pop_donor) + log(gdp2000_donor)",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/log_oda_log_controls_donor_year",
                "axes_changed": ["functional_form", "controls", "fixed_effects"],
                "details": {"outcome": "log(oda)", "controls": "log macro block", "fe": "pair+donor_year"}}
)

# LOO controls + year FE (drop each control, pair + year)
loo_controls = ['pop', 'gdp2000', 'pop_donor', 'gdp2000_donor']
loo_short = {'pop': 'pop', 'gdp2000': 'gdp', 'pop_donor': 'popd', 'gdp2000_donor': 'gdpd'}
for ctrl_name in loo_controls:
    kept = [c for c in loo_controls if c != ctrl_name]
    ctrl_str = " + ".join(kept)
    sid = f"rc/joint/loo_{loo_short[ctrl_name]}_yearfe"
    run_spec(
        sid,
        "modules/robustness/joint.md#joint-variations", "G1",
        "oda", "p_unvotes_elecex",
        f"oda ~ unvotes + i_elecex + p_unvotes_elecex + {ctrl_str}", "pair + year",
        data_ctrl, "p_unvotes_elecex",
        f"LOO drop {ctrl_name} + year FE, N={len(data_ctrl)}", "pair + year", ctrl_str,
        axis_block_name="joint",
        axis_block={"spec_id": sid,
                    "axes_changed": ["controls", "fixed_effects"],
                    "details": {"dropped": [ctrl_name], "fe": "pair+year"}}
    )

# LOO controls + donor-year FE (drop each control, pair + donor_year)
for ctrl_name in loo_controls:
    kept = [c for c in loo_controls if c != ctrl_name]
    ctrl_str = " + ".join(kept)
    sid = f"rc/joint/loo_{loo_short[ctrl_name]}_dyfe"
    run_spec(
        sid,
        "modules/robustness/joint.md#joint-variations", "G1",
        "oda", "p_unvotes_elecex",
        f"oda ~ unvotes + i_elecex + p_unvotes_elecex + {ctrl_str}", "pair + donor_year",
        data_ctrl, "p_unvotes_elecex",
        f"LOO drop {ctrl_name} + donor-year FE, N={len(data_ctrl)}", "pair + donor_year", ctrl_str,
        axis_block_name="joint",
        axis_block={"spec_id": sid,
                    "axes_changed": ["controls", "fixed_effects"],
                    "details": {"dropped": [ctrl_name], "fe": "pair+donor_year"}}
    )


# Log ODA + recipient-year FE
data_log_base_ry = data_log_base.copy()
data_log_base_ry['recipient_year'] = data_log_base_ry['r'] + '_' + data_log_base_ry['year'].astype(str)
run_spec(
    "rc/joint/log_oda_recip_year", "modules/robustness/joint.md#joint-variations", "G1",
    "oda", "p_unvotes_elecex",
    "oda ~ unvotes + i_elecex + p_unvotes_elecex", "pair + recipient_year",
    data_log_base_ry, "p_unvotes_elecex",
    f"Log ODA, recip-year FE, N={len(data_log_base_ry)}", "pair + recipient_year", "none",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/log_oda_recip_year",
                "axes_changed": ["functional_form", "fixed_effects"],
                "details": {"outcome": "log(oda)", "fe": "pair+recipient_year"}}
)

# Drop Big 5 + recipient-year FE
data_no_big5_ry = data_no_big5.copy()
data_no_big5_ry['recipient_year'] = data_no_big5_ry['r'] + '_' + data_no_big5_ry['year'].astype(str)
run_spec(
    "rc/joint/drop_big5_recip_year", "modules/robustness/joint.md#joint-variations", "G1",
    "oda", "p_unvotes_elecex",
    "oda ~ unvotes + i_elecex + p_unvotes_elecex", "pair + recipient_year",
    data_no_big5_ry, "p_unvotes_elecex",
    f"Excl Big 5, recip-year FE, N={len(data_no_big5_ry)}", "pair + recipient_year", "none",
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/drop_big5_recip_year",
                "axes_changed": ["sample", "fixed_effects"],
                "details": {"dropped_recipients": ["EGY", "IDN", "IND", "ISR", "CHN"],
                            "fe": "pair+recipient_year"}}
)


# ============================================================
# EXPLORE: TREATMENT DECOMPOSITION
# ============================================================

print("Running exploration specifications (treatment decompositions)...")

# Decomposed UN votes (pair + year FE)
run_explore(
    "explore/treatment/decompose_unvotes_rt_resid",
    "modules/robustness/controls.md#treatment-decomposition", "G1",
    "oda", "p_unvotes_rt_elecex",
    "oda ~ unvotes_rt + unvotes_resid + i_elecex + p_unvotes_rt_elecex + p_unvotes_resid_elecex",
    "pair + year", data_base, "p_unvotes_rt_elecex",
    f"Decomposed UN votes, pair+year, N={len(data_base)}", "pair + year", "none",
    extra_coefs_to_report=["p_unvotes_resid_elecex"]
)

# Competitive election split (EIEC), pair + year FE
run_explore(
    "explore/treatment/competitive_election_split_eiec",
    "modules/robustness/controls.md#treatment-decomposition", "G1",
    "oda", "p_unvotes_elecex_comp",
    "oda ~ unvotes + noncomp + p_unvotes_noncomp + i_elecex_comp + p_unvotes_elecex_comp + i_elecex_noncomp + p_unvotes_elecex_noncomp",
    "pair + year", data_eiec, "p_unvotes_elecex_comp",
    f"EIEC competitive split, pair+year, N={len(data_eiec)}", "pair + year", "none",
    extra_coefs_to_report=["p_unvotes_elecex_noncomp"]
)

# Competitive election split (PCT), pair + year FE
run_explore(
    "explore/treatment/competitive_election_split_pct",
    "modules/robustness/controls.md#treatment-decomposition", "G1",
    "oda", "p_unvotes_elecex_comp",
    "oda ~ unvotes + noncomp + p_unvotes_noncomp + i_elecex_comp + p_unvotes_elecex_comp + i_elecex_noncomp + p_unvotes_elecex_noncomp",
    "pair + year", data_pct, "p_unvotes_elecex_comp",
    f"PCT competitive split, pair+year, N={len(data_pct)}", "pair + year", "none",
    extra_coefs_to_report=["p_unvotes_elecex_noncomp"]
)

# Early vs late election
run_explore(
    "explore/treatment/early_vs_late_election",
    "modules/robustness/controls.md#treatment-decomposition", "G1",
    "oda", "p_unvotes_lateelec",
    "oda ~ unvotes + i_earlyelec + p_unvotes_earlyelec + i_lateelec + p_unvotes_lateelec",
    "pair + year", data_time, "p_unvotes_lateelec",
    f"Early/late election, pair+year, N={len(data_time)}", "pair + year", "none",
    extra_coefs_to_report=["p_unvotes_earlyelec"]
)


# ============================================================
# EXPLORE: JOINT TREATMENT DECOMPOSITION x FE
# ============================================================

print("Running joint explore specifications...")

# Decomposed UN votes + donor_year FE
run_explore(
    "explore/joint/decompose_unvotes_donor_year",
    "modules/robustness/joint.md#joint-variations", "G1",
    "oda", "p_unvotes_rt_elecex",
    "oda ~ unvotes_rt + unvotes_resid + i_elecex + p_unvotes_rt_elecex + p_unvotes_resid_elecex",
    "pair + donor_year", data_base, "p_unvotes_rt_elecex",
    f"Decomposed, donor_year FE, N={len(data_base)}", "pair + donor_year", "none",
    extra_coefs_to_report=["p_unvotes_resid_elecex"]
)

# Decomposed UN votes + controls (pair FE only)
run_explore(
    "explore/joint/decompose_unvotes_controls",
    "modules/robustness/joint.md#joint-variations", "G1",
    "oda", "p_unvotes_rt_elecex",
    "oda ~ unvotes_rt + unvotes_resid + i_elecex + p_unvotes_rt_elecex + p_unvotes_resid_elecex + pop + gdp2000 + pop_donor + gdp2000_donor",
    "pair", data_ctrl, "p_unvotes_rt_elecex",
    f"Decomposed + controls, pair FE, N={len(data_ctrl)}", "pair",
    "pop + gdp2000 + pop_donor + gdp2000_donor",
    extra_coefs_to_report=["p_unvotes_resid_elecex"]
)

# Decomposed UN votes + pair+year FE (explicit)
run_explore(
    "explore/joint/decompose_unvotes_pair_year",
    "modules/robustness/joint.md#joint-variations", "G1",
    "oda", "p_unvotes_rt_elecex",
    "oda ~ unvotes_rt + unvotes_resid + i_elecex + p_unvotes_rt_elecex + p_unvotes_resid_elecex",
    "pair + year", data_base, "p_unvotes_rt_elecex",
    f"Decomposed, pair+year, N={len(data_base)}", "pair + year", "none",
    extra_coefs_to_report=["p_unvotes_resid_elecex"]
)

# EIEC competitive + donor_year FE
run_explore(
    "explore/joint/competitive_eiec_donor_year",
    "modules/robustness/joint.md#joint-variations", "G1",
    "oda", "p_unvotes_elecex_comp",
    "oda ~ unvotes + noncomp + p_unvotes_noncomp + i_elecex_comp + p_unvotes_elecex_comp + i_elecex_noncomp + p_unvotes_elecex_noncomp",
    "pair + donor_year", data_eiec, "p_unvotes_elecex_comp",
    f"EIEC competitive, donor_year FE, N={len(data_eiec)}", "pair + donor_year", "none",
    extra_coefs_to_report=["p_unvotes_elecex_noncomp"]
)

# EIEC competitive + pair_year FE
run_explore(
    "explore/joint/competitive_eiec_pair_year",
    "modules/robustness/joint.md#joint-variations", "G1",
    "oda", "p_unvotes_elecex_comp",
    "oda ~ unvotes + noncomp + p_unvotes_noncomp + i_elecex_comp + p_unvotes_elecex_comp + i_elecex_noncomp + p_unvotes_elecex_noncomp",
    "pair + year", data_eiec, "p_unvotes_elecex_comp",
    f"EIEC competitive, pair+year, N={len(data_eiec)}", "pair + year", "none",
    extra_coefs_to_report=["p_unvotes_elecex_noncomp"]
)

# EIEC competitive + controls (pair FE)
run_explore(
    "explore/joint/competitive_eiec_controls",
    "modules/robustness/joint.md#joint-variations", "G1",
    "oda", "p_unvotes_elecex_comp",
    "oda ~ unvotes + noncomp + p_unvotes_noncomp + i_elecex_comp + p_unvotes_elecex_comp + i_elecex_noncomp + p_unvotes_elecex_noncomp + pop + gdp2000 + pop_donor + gdp2000_donor",
    "pair", data_eiec_ctrl, "p_unvotes_elecex_comp",
    f"EIEC competitive + controls, pair FE, N={len(data_eiec_ctrl)}", "pair",
    "pop + gdp2000 + pop_donor + gdp2000_donor",
    extra_coefs_to_report=["p_unvotes_elecex_noncomp"]
)

# PCT competitive + pair_year FE
run_explore(
    "explore/joint/competitive_pct_pair_year",
    "modules/robustness/joint.md#joint-variations", "G1",
    "oda", "p_unvotes_elecex_comp",
    "oda ~ unvotes + noncomp + p_unvotes_noncomp + i_elecex_comp + p_unvotes_elecex_comp + i_elecex_noncomp + p_unvotes_elecex_noncomp",
    "pair + year", data_pct, "p_unvotes_elecex_comp",
    f"PCT competitive, pair+year, N={len(data_pct)}", "pair + year", "none",
    extra_coefs_to_report=["p_unvotes_elecex_noncomp"]
)

# PCT competitive + donor_year FE
run_explore(
    "explore/joint/competitive_pct_donor_year",
    "modules/robustness/joint.md#joint-variations", "G1",
    "oda", "p_unvotes_elecex_comp",
    "oda ~ unvotes + noncomp + p_unvotes_noncomp + i_elecex_comp + p_unvotes_elecex_comp + i_elecex_noncomp + p_unvotes_elecex_noncomp",
    "pair + donor_year", data_pct, "p_unvotes_elecex_comp",
    f"PCT competitive, donor_year FE, N={len(data_pct)}", "pair + donor_year", "none",
    extra_coefs_to_report=["p_unvotes_elecex_noncomp"]
)

# PCT competitive + controls
run_explore(
    "explore/joint/competitive_pct_controls",
    "modules/robustness/joint.md#joint-variations", "G1",
    "oda", "p_unvotes_elecex_comp",
    "oda ~ unvotes + noncomp + p_unvotes_noncomp + i_elecex_comp + p_unvotes_elecex_comp + i_elecex_noncomp + p_unvotes_elecex_noncomp + pop + gdp2000 + pop_donor + gdp2000_donor",
    "pair", data_pct_ctrl, "p_unvotes_elecex_comp",
    f"PCT competitive + controls, pair FE, N={len(data_pct_ctrl)}", "pair",
    "pop + gdp2000 + pop_donor + gdp2000_donor",
    extra_coefs_to_report=["p_unvotes_elecex_noncomp"]
)

# Early/late election + pair+year (already done above, skip duplicate)

# Log ODA + competitive EIEC
run_explore(
    "explore/joint/log_oda_competitive_eiec",
    "modules/robustness/joint.md#joint-variations", "G1",
    "oda", "p_unvotes_elecex_comp",
    "oda ~ unvotes + noncomp + p_unvotes_noncomp + i_elecex_comp + p_unvotes_elecex_comp + i_elecex_noncomp + p_unvotes_elecex_noncomp",
    "pair + year", make_log_oda(data_eiec), "p_unvotes_elecex_comp",
    f"Log ODA, EIEC competitive, pair+year, N={len(data_eiec)}", "pair + year", "none",
    extra_coefs_to_report=["p_unvotes_elecex_noncomp"]
)

# Log ODA + competitive PCT
run_explore(
    "explore/joint/log_oda_competitive_pct",
    "modules/robustness/joint.md#joint-variations", "G1",
    "oda", "p_unvotes_elecex_comp",
    "oda ~ unvotes + noncomp + p_unvotes_noncomp + i_elecex_comp + p_unvotes_elecex_comp + i_elecex_noncomp + p_unvotes_elecex_noncomp",
    "pair + year", make_log_oda(data_pct), "p_unvotes_elecex_comp",
    f"Log ODA, PCT competitive, pair+year, N={len(data_pct)}", "pair + year", "none",
    extra_coefs_to_report=["p_unvotes_elecex_noncomp"]
)

# Log ODA + decomposed UN votes
run_explore(
    "explore/joint/log_oda_decompose_unvotes",
    "modules/robustness/joint.md#joint-variations", "G1",
    "oda", "p_unvotes_rt_elecex",
    "oda ~ unvotes_rt + unvotes_resid + i_elecex + p_unvotes_rt_elecex + p_unvotes_resid_elecex",
    "pair + year", data_log_base, "p_unvotes_rt_elecex",
    f"Log ODA, decomposed UN votes, pair+year, N={len(data_log_base)}", "pair + year", "none",
    extra_coefs_to_report=["p_unvotes_resid_elecex"]
)

# Early/late election + pair+year (already above)
run_explore(
    "explore/joint/early_late_pair_year",
    "modules/robustness/joint.md#joint-variations", "G1",
    "oda", "p_unvotes_lateelec",
    "oda ~ unvotes + i_earlyelec + p_unvotes_earlyelec + i_lateelec + p_unvotes_lateelec",
    "pair + year", data_time, "p_unvotes_lateelec",
    f"Early/late election, pair+year, N={len(data_time)}", "pair + year", "none",
    extra_coefs_to_report=["p_unvotes_earlyelec"]
)


# ============================================================
# INFERENCE VARIANTS (on baseline spec)
# ============================================================

print("Running inference variants on baseline...")

baseline_formula = "oda ~ unvotes + i_elecex + p_unvotes_elecex"
baseline_fe = "pair + year"
focal = "p_unvotes_elecex"

# Cluster at pair level
run_inference_variant(
    base_run_id, "infer/se/cluster/pair",
    "modules/inference/standard_errors.md#clustering", "G1",
    baseline_formula, baseline_fe, data_base, focal,
    {"CRV1": "pair"}, "pair"
)

# Cluster at recipient level
run_inference_variant(
    base_run_id, "infer/se/cluster/recipient",
    "modules/inference/standard_errors.md#clustering", "G1",
    baseline_formula, baseline_fe, data_base, focal,
    {"CRV1": "r"}, "recipient"
)

# Two-way cluster on donor and recipient
# pyfixest supports two-way clustering
run_inference_variant(
    base_run_id, "infer/se/cluster/twoway_donor_recipient",
    "modules/inference/standard_errors.md#two-way-clustering", "G1",
    baseline_formula, baseline_fe, data_base, focal,
    {"CRV1": "d+r"}, "donor+recipient"
)

# HC1 robust
run_inference_variant(
    base_run_id, "infer/se/hc/hc1",
    "modules/inference/standard_errors.md#heteroskedasticity-robust", "G1",
    baseline_formula, baseline_fe, data_base, focal,
    "hetero", "none (HC1)"
)


# ============================================================
# Write Output Files
# ============================================================

print(f"\nWriting outputs...")
print(f"  Estimate specs: {len(results)}")
print(f"  Inference variants: {len(inference_results)}")
print(f"  Exploration specs: {len(exploration_results)}")

# specification_results.csv
df_results = pd.DataFrame(results)
df_results.to_csv(f"{OUTPUT_DIR}/specification_results.csv", index=False)

# inference_results.csv
df_infer = pd.DataFrame(inference_results)
df_infer.to_csv(f"{OUTPUT_DIR}/inference_results.csv", index=False)

# exploration_results.csv
df_explore = pd.DataFrame(exploration_results)
df_explore.to_csv(f"{OUTPUT_DIR}/exploration_results.csv", index=False)

# Print summary
print("\n=== SPECIFICATION RESULTS SUMMARY ===")
print(f"Total rows: {len(df_results)}")
print(f"Successful: {df_results['run_success'].sum()}")
print(f"Failed: {(df_results['run_success'] == 0).sum()}")
print(f"\nBaseline coef on p_unvotes_elecex: {df_results.loc[df_results['spec_id']=='baseline', 'coefficient'].values[0]:.4f}")
print(f"Baseline 3-way SE: {df_results.loc[df_results['spec_id']=='baseline', 'std_error'].values[0]:.4f}")
print(f"Baseline p-value: {df_results.loc[df_results['spec_id']=='baseline', 'p_value'].values[0]:.4f}")

print("\n=== COEFFICIENT RANGE (successful core specs) ===")
successful = df_results[df_results['run_success'] == 1]
print(f"Min coef: {successful['coefficient'].min():.4f}")
print(f"Max coef: {successful['coefficient'].max():.4f}")
print(f"Median coef: {successful['coefficient'].median():.4f}")
n_sig = (successful['p_value'] < 0.05).sum()
print(f"Significant at 5%: {n_sig}/{len(successful)}")

print("\nDone!")
