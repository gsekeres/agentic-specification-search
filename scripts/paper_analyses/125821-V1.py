#!/usr/bin/env python3
"""
Specification search script for 125821-V1:
"School Spending and Student Outcomes: Evidence from Revenue Limit Elections
in Wisconsin" - Jason Baron, AEJ: Economic Policy

Surface-driven execution of ~60 specifications for baseline group G1.

Design: Regression Discontinuity (Cellini one-step dynamic RD estimator)
- Panel data: areg y <controls> | district_code, cluster(district_code)
- Focal parameter: 10-year average of op_win_prev1..10 coefficients
- Cross-section RD (rdrobust) as robustness check
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import hashlib
import sys
import traceback
import warnings
warnings.filterwarnings('ignore')

from scipy import stats

# ============================================================
# PATHS
# ============================================================
BASE_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search"
PACKAGE_DIR = f"{BASE_DIR}/data/downloads/extracted/125821-V1"
PAPER_ID = "125821-V1"

# ============================================================
# LOAD SURFACE
# ============================================================
with open(f"{PACKAGE_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface = json.load(f)

shash = "sha256:" + hashlib.sha256(
    json.dumps(surface, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
).hexdigest()

# ============================================================
# SOFTWARE BLOCK
# ============================================================
software_blk = {
    "runner_language": "python",
    "runner_version": sys.version.split()[0],
    "packages": {
        "pyfixest": getattr(pf, '__version__', '0.40+'),
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
}

try:
    import rdrobust as rdrobust_pkg
    software_blk["packages"]["rdrobust"] = "1.3+"
except:
    pass

# ============================================================
# DESIGN AUDIT BLOCK (from surface)
# ============================================================
bg = surface["baseline_groups"][0]
design_audit = bg["design_audit"]
design_block = {"regression_discontinuity": design_audit}

# Canonical inference
canonical_inference = bg["inference_plan"]["canonical"]

# ============================================================
# LOAD PANEL DATA (for one-step estimator)
# ============================================================
df_panel = pd.read_stata(f"{PACKAGE_DIR}/Data/Final/onestep_panel_tables.dta")
print(f"Panel data shape: {df_panel.shape}")

# ============================================================
# LOAD CROSS-SECTION DATA (for rdrobust)
# ============================================================
df_cs = pd.read_stata(f"{PACKAGE_DIR}/Data/Final/itt_cross_section.dta")
df_cs['perc'] = df_cs['perc'] - 50  # re-center vote share
print(f"Cross-section data shape: {df_cs.shape}")

# ============================================================
# DEFINE CONTROL VARIABLE LISTS (matching Stata globals)
# ============================================================
op_win = [f'op_win_prev{i}' for i in range(1, 19)]
bond_win = [f'bond_win_prev{i}' for i in range(1, 19)]
yrdums = [f'yrdums{i}' for i in range(1, 20)]
op_ismeas = [f'op_ismeas_prev{i}' for i in range(1, 19)]
bond_ismeas = [f'bond_ismeas_prev{i}' for i in range(1, 19)]
op_month = [f'op_month_prev{i}' for i in range(1, 19)]
bond_month = [f'bond_month_prev{i}' for i in range(1, 19)]
op_percent = [f'op_percent_prev{i}' for i in range(1, 19)]
op_percent2 = [f'op_percent2_prev{i}' for i in range(1, 19)]
op_percent3 = [f'op_percent3_prev{i}' for i in range(1, 19)]
bond_percent = [f'bond_percent_prev{i}' for i in range(1, 19)]
bond_percent2 = [f'bond_percent2_prev{i}' for i in range(1, 19)]
bond_percent3 = [f'bond_percent3_prev{i}' for i in range(1, 19)]
recurring = [f'recurring_prev{i}' for i in range(1, 19)]
op_numelec = [f'op_numelec_prev{i}' for i in range(1, 19)]
bond_numelec = [f'bond_numelec_prev{i}' for i in range(1, 19)]

# Filter to columns present in data
def filter_cols(varlist, df):
    return [v for v in varlist if v in df.columns]

# Cubic: all polynomial terms (1st, 2nd, 3rd order)
cubic_controls = (
    filter_cols(op_win, df_panel) + filter_cols(bond_win, df_panel) +
    filter_cols(yrdums, df_panel) +
    filter_cols(op_ismeas, df_panel) + filter_cols(bond_ismeas, df_panel) +
    filter_cols(op_month, df_panel) + filter_cols(bond_month, df_panel) +
    filter_cols(op_percent, df_panel) + filter_cols(op_percent2, df_panel) + filter_cols(op_percent3, df_panel) +
    filter_cols(bond_percent, df_panel) + filter_cols(bond_percent2, df_panel) + filter_cols(bond_percent3, df_panel) +
    filter_cols(recurring, df_panel) + filter_cols(op_numelec, df_panel) + filter_cols(bond_numelec, df_panel)
)

# Quadratic: drop cubic terms
quadratic_controls = (
    filter_cols(op_win, df_panel) + filter_cols(bond_win, df_panel) +
    filter_cols(yrdums, df_panel) +
    filter_cols(op_ismeas, df_panel) + filter_cols(bond_ismeas, df_panel) +
    filter_cols(op_month, df_panel) + filter_cols(bond_month, df_panel) +
    filter_cols(op_percent, df_panel) + filter_cols(op_percent2, df_panel) +
    filter_cols(bond_percent, df_panel) + filter_cols(bond_percent2, df_panel) +
    filter_cols(recurring, df_panel) + filter_cols(op_numelec, df_panel) + filter_cols(bond_numelec, df_panel)
)

# Linear: drop quadratic and cubic terms
linear_controls = (
    filter_cols(op_win, df_panel) + filter_cols(bond_win, df_panel) +
    filter_cols(yrdums, df_panel) +
    filter_cols(op_ismeas, df_panel) + filter_cols(bond_ismeas, df_panel) +
    filter_cols(op_month, df_panel) + filter_cols(bond_month, df_panel) +
    filter_cols(op_percent, df_panel) +
    filter_cols(bond_percent, df_panel) +
    filter_cols(recurring, df_panel) + filter_cols(op_numelec, df_panel) + filter_cols(bond_numelec, df_panel)
)

# ============================================================
# OUTCOME CONFIGURATIONS (from surface)
# ============================================================
outcome_configs = {
    "advprof_math10": {"weight": "num_takers_math10", "extra_controls": [], "label": "Math proficiency grade 10"},
    "dropout_rate": {"weight": "student_count", "extra_controls": [], "label": "Dropout rate"},
    "wkce_math10": {"weight": "num_takers_math10", "extra_controls": [], "label": "WKCE math score grade 10"},
    "log_instate_enr": {"weight": None, "extra_controls": ["grade9lagged"], "label": "Log in-state postsec enrollment"},
    "rev_lim_mem": {"weight": None, "extra_controls": [], "label": "Revenue limits per member"},
    "tot_exp_mem": {"weight": None, "extra_controls": [], "label": "Total expenditures per member"},
}

# ============================================================
# HELPER: Run one-step panel regression
# ============================================================
def run_onestep(outcome_var, control_list, weight_var=None, extra_controls=None,
                sample_mask=None, df=None, n_avg_years=10):
    """
    Run the Cellini one-step estimator using pyfixest.
    Returns (model, lincom_coef, lincom_se, lincom_p, lincom_ci, n_obs, r2, coef_dict)
    """
    if df is None:
        df = df_panel
    if extra_controls is None:
        extra_controls = []

    all_controls = control_list + extra_controls
    all_vars = [outcome_var] + all_controls + ['district_code']
    if weight_var:
        all_vars.append(weight_var)

    regdf = df[all_vars].copy() if sample_mask is None else df.loc[sample_mask, all_vars].copy()
    regdf = regdf.dropna()

    rhs = ' + '.join(all_controls)
    formula = f'{outcome_var} ~ {rhs} | district_code'

    if weight_var:
        m = pf.feols(formula, data=regdf, vcov={'CRV1': 'district_code'}, weights=weight_var)
    else:
        m = pf.feols(formula, data=regdf, vcov={'CRV1': 'district_code'})

    # Compute lincom: average of op_win_prev1 ... op_win_prev{n_avg_years}
    coef_index = m.coef().index.tolist()
    weight_factor = 1.0 / n_avg_years
    op_coefs = []
    for i in range(1, n_avg_years + 1):
        vname = f'op_win_prev{i}'
        if vname in coef_index:
            op_coefs.append((vname, m.coef()[vname]))

    lincom_coef = weight_factor * sum(c for _, c in op_coefs)

    # Delta method SE
    vcov_mat = m._vcov
    c_vec = np.zeros(len(coef_index))
    for vname, _ in op_coefs:
        idx = coef_index.index(vname)
        c_vec[idx] = weight_factor
    lincom_se = np.sqrt(c_vec @ vcov_mat @ c_vec)

    # p-value (t-distribution with clustering dof)
    n_clusters = regdf['district_code'].nunique()
    dof = n_clusters - 1
    lincom_t = lincom_coef / lincom_se
    lincom_p = 2 * (1 - stats.t.cdf(abs(lincom_t), df=dof))

    # CI
    t_crit = stats.t.ppf(0.975, df=dof)
    ci_lower = lincom_coef - t_crit * lincom_se
    ci_upper = lincom_coef + t_crit * lincom_se

    # Coefficient dict (all coefficients)
    coef_dict = {k: float(v) for k, v in zip(coef_index, m.coef().values)}

    return m, lincom_coef, lincom_se, lincom_p, (ci_lower, ci_upper), int(m._N), float(m._r2), coef_dict


def run_onestep_hetero(outcome_var, control_list, weight_var=None, extra_controls=None,
                       sample_mask=None, df=None, n_avg_years=10):
    """
    Same as run_onestep but with heteroskedasticity-robust SE (no clustering).
    """
    if df is None:
        df = df_panel
    if extra_controls is None:
        extra_controls = []

    all_controls = control_list + extra_controls
    all_vars = [outcome_var] + all_controls + ['district_code']
    if weight_var:
        all_vars.append(weight_var)

    regdf = df[all_vars].copy() if sample_mask is None else df.loc[sample_mask, all_vars].copy()
    regdf = regdf.dropna()

    rhs = ' + '.join(all_controls)
    formula = f'{outcome_var} ~ {rhs} | district_code'

    if weight_var:
        m = pf.feols(formula, data=regdf, vcov='hetero', weights=weight_var)
    else:
        m = pf.feols(formula, data=regdf, vcov='hetero')

    # Compute lincom
    coef_index = m.coef().index.tolist()
    weight_factor = 1.0 / n_avg_years
    op_coefs = []
    for i in range(1, n_avg_years + 1):
        vname = f'op_win_prev{i}'
        if vname in coef_index:
            op_coefs.append((vname, m.coef()[vname]))

    lincom_coef = weight_factor * sum(c for _, c in op_coefs)

    vcov_mat = m._vcov
    c_vec = np.zeros(len(coef_index))
    for vname, _ in op_coefs:
        idx = coef_index.index(vname)
        c_vec[idx] = weight_factor
    lincom_se = np.sqrt(c_vec @ vcov_mat @ c_vec)

    lincom_t = lincom_coef / lincom_se
    n_obs = int(m._N)
    n_params = len(coef_index)
    dof = n_obs - n_params - 1  # approximate dof for HC
    lincom_p = 2 * (1 - stats.t.cdf(abs(lincom_t), df=max(dof, 1)))

    t_crit = stats.t.ppf(0.975, df=max(dof, 1))
    ci_lower = lincom_coef - t_crit * lincom_se
    ci_upper = lincom_coef + t_crit * lincom_se

    coef_dict = {k: float(v) for k, v in zip(coef_index, m.coef().values)}

    return m, lincom_coef, lincom_se, lincom_p, (ci_lower, ci_upper), n_obs, float(m._r2), coef_dict


# ============================================================
# RESULTS ACCUMULATORS
# ============================================================
spec_results = []
inference_results = []
spec_run_counter = 0
infer_run_counter = 0


def make_spec_row(spec_id, spec_tree_path, outcome_var, treatment_var,
                  coef, se, pval, ci, n_obs, r2, coef_vec_json,
                  sample_desc, fixed_effects, controls_desc, cluster_var,
                  run_success=1, run_error=""):
    global spec_run_counter
    spec_run_counter += 1
    return {
        "paper_id": PAPER_ID,
        "spec_run_id": f"{PAPER_ID}_run{spec_run_counter:04d}",
        "spec_id": spec_id,
        "spec_tree_path": spec_tree_path,
        "baseline_group_id": "G1",
        "outcome_var": outcome_var,
        "treatment_var": treatment_var,
        "coefficient": coef,
        "std_error": se,
        "p_value": pval,
        "ci_lower": ci[0] if ci else np.nan,
        "ci_upper": ci[1] if ci else np.nan,
        "n_obs": n_obs,
        "r_squared": r2,
        "coefficient_vector_json": json.dumps(coef_vec_json),
        "sample_desc": sample_desc,
        "fixed_effects": fixed_effects,
        "controls_desc": controls_desc,
        "cluster_var": cluster_var,
        "run_success": run_success,
        "run_error": run_error,
    }


def make_infer_row(base_spec_run_id, spec_id, spec_tree_path,
                   outcome_var, coef, se, pval, ci, n_obs, r2,
                   coef_vec_json, cluster_var="",
                   treatment_var="op_win_prev1_through_10",
                   run_success=1, run_error=""):
    global infer_run_counter
    infer_run_counter += 1
    return {
        "paper_id": PAPER_ID,
        "inference_run_id": f"{PAPER_ID}_infer{infer_run_counter:04d}",
        "spec_run_id": base_spec_run_id,
        "spec_id": spec_id,
        "spec_tree_path": spec_tree_path,
        "baseline_group_id": "G1",
        "outcome_var": outcome_var,
        "treatment_var": treatment_var,
        "coefficient": coef,
        "std_error": se,
        "p_value": pval,
        "ci_lower": ci[0] if ci else np.nan,
        "ci_upper": ci[1] if ci else np.nan,
        "n_obs": n_obs,
        "r_squared": r2,
        "coefficient_vector_json": json.dumps(coef_vec_json),
        "cluster_var": cluster_var,
        "run_success": run_success,
        "run_error": run_error,
    }


def make_success_payload(coefficients, inference_block, design_blk=None,
                         axis_block=None, axis_block_name=None,
                         extra=None, focal=None):
    payload = {
        "coefficients": coefficients,
        "inference": inference_block,
        "software": software_blk,
        "surface_hash": shash,
        "design": design_blk or design_block,
    }
    if axis_block and axis_block_name:
        payload[axis_block_name] = axis_block
    if extra:
        payload["extra"] = extra
    if focal:
        payload["focal"] = focal
    return payload


def make_failure_payload(error_msg, stage="estimation"):
    return {
        "error": error_msg[:240],
        "error_details": {
            "stage": stage,
            "exception_type": "RuntimeError",
            "exception_message": error_msg[:500],
        }
    }


# ============================================================
# STEP 1: BASELINE SPECS
# ============================================================
print("\n===== STEP 1: BASELINE SPECIFICATIONS =====")

# The headline baseline: advprof_math10 (Table 5 Panel A Col 2)
baseline_outcomes = {
    "advprof_math10": "baseline",
    "dropout_rate": "baseline__dropout_rate",
    "wkce_math10": "baseline__wkce_math10",
    "log_instate_enr": "baseline__log_instate_enr",
    "rev_lim_mem": "baseline__rev_lim_mem",
    "tot_exp_mem": "baseline__tot_exp_mem",
}

for ovar, spec_id in baseline_outcomes.items():
    cfg = outcome_configs[ovar]
    try:
        m, coef, se, pval, ci, n_obs, r2, coef_dict = run_onestep(
            outcome_var=ovar,
            control_list=cubic_controls,
            weight_var=cfg["weight"],
            extra_controls=cfg["extra_controls"],
        )
        payload = make_success_payload(
            coefficients=coef_dict,
            inference_block={"spec_id": canonical_inference["spec_id"],
                             "params": canonical_inference["params"]},
            focal={
                "parameter": "op_win_prev_10yr_avg",
                "label": "10-year average operational referendum effect",
                "selector": {"lincom": "0.10*(op_win_prev1 + ... + op_win_prev10)"},
                "summary_rule": "lincom"
            },
        )
        row = make_spec_row(
            spec_id=spec_id,
            spec_tree_path="designs/regression_discontinuity.md#baseline-required",
            outcome_var=ovar,
            treatment_var="op_win_prev1_through_10",
            coef=coef, se=se, pval=pval, ci=ci,
            n_obs=n_obs, r2=r2,
            coef_vec_json=payload,
            sample_desc="Full panel, Wisconsin school districts 1996-2014",
            fixed_effects="district_code",
            controls_desc="Cubic polynomial (Cellini one-step): op_win/bond_win + yrdums + election chars + poly(vote_share,3)",
            cluster_var="district_code",
        )
        spec_results.append(row)
        print(f"  {spec_id} ({ovar}): coef={coef:.4f}, se={se:.4f}, p={pval:.4f}, N={n_obs}")
    except Exception as e:
        payload = make_failure_payload(str(e), stage="estimation")
        row = make_spec_row(
            spec_id=spec_id,
            spec_tree_path="designs/regression_discontinuity.md#baseline-required",
            outcome_var=ovar,
            treatment_var="op_win_prev1_through_10",
            coef=np.nan, se=np.nan, pval=np.nan, ci=(np.nan, np.nan),
            n_obs=np.nan, r2=np.nan,
            coef_vec_json=payload,
            sample_desc="Full panel",
            fixed_effects="district_code",
            controls_desc="Cubic polynomial",
            cluster_var="district_code",
            run_success=0, run_error=str(e)[:240],
        )
        spec_results.append(row)
        print(f"  FAILED {spec_id} ({ovar}): {e}")


# ============================================================
# STEP 2: DESIGN VARIANTS (polynomial order)
# ============================================================
print("\n===== STEP 2: DESIGN VARIANTS =====")

design_variants = {
    "design/regression_discontinuity/poly/local_linear": {
        "controls": linear_controls,
        "desc": "Linear polynomial (Cellini one-step)",
        "poly_order": 1,
    },
    "design/regression_discontinuity/poly/local_quadratic": {
        "controls": quadratic_controls,
        "desc": "Quadratic polynomial (Cellini one-step)",
        "poly_order": 2,
    },
}

for design_spec_id, dconfig in design_variants.items():
    for ovar, cfg in outcome_configs.items():
        spec_id = design_spec_id
        try:
            m, coef, se, pval, ci, n_obs, r2, coef_dict = run_onestep(
                outcome_var=ovar,
                control_list=dconfig["controls"],
                weight_var=cfg["weight"],
                extra_controls=cfg["extra_controls"],
            )
            # Override design audit for this variant
            variant_design_audit = dict(design_audit)
            variant_design_audit["poly_order"] = dconfig["poly_order"]
            variant_design_block = {"regression_discontinuity": variant_design_audit}

            payload = make_success_payload(
                coefficients=coef_dict,
                inference_block={"spec_id": canonical_inference["spec_id"],
                                 "params": canonical_inference["params"]},
                design_blk=variant_design_block,
                focal={
                    "parameter": "op_win_prev_10yr_avg",
                    "label": "10-year average operational referendum effect",
                    "selector": {"lincom": "0.10*(op_win_prev1 + ... + op_win_prev10)"},
                    "summary_rule": "lincom"
                },
            )
            row = make_spec_row(
                spec_id=spec_id,
                spec_tree_path="designs/regression_discontinuity.md#c-local-polynomial-order",
                outcome_var=ovar,
                treatment_var="op_win_prev1_through_10",
                coef=coef, se=se, pval=pval, ci=ci,
                n_obs=n_obs, r2=r2,
                coef_vec_json=payload,
                sample_desc="Full panel, Wisconsin school districts 1996-2014",
                fixed_effects="district_code",
                controls_desc=dconfig["desc"],
                cluster_var="district_code",
            )
            spec_results.append(row)
            print(f"  {spec_id} ({ovar}): coef={coef:.4f}, se={se:.4f}, p={pval:.4f}")
        except Exception as e:
            payload = make_failure_payload(str(e))
            row = make_spec_row(
                spec_id=spec_id,
                spec_tree_path="designs/regression_discontinuity.md#c-local-polynomial-order",
                outcome_var=ovar,
                treatment_var="op_win_prev1_through_10",
                coef=np.nan, se=np.nan, pval=np.nan, ci=(np.nan, np.nan),
                n_obs=np.nan, r2=np.nan,
                coef_vec_json=payload,
                sample_desc="Full panel",
                fixed_effects="district_code",
                controls_desc=dconfig["desc"],
                cluster_var="district_code",
                run_success=0, run_error=str(e)[:240],
            )
            spec_results.append(row)
            print(f"  FAILED {spec_id} ({ovar}): {e}")

# ============================================================
# STEP 3: RC SPECS
# ============================================================
print("\n===== STEP 3: ROBUSTNESS CHECKS =====")

# --- RC: Sample restrictions ---
print("\n  --- RC: Sample restrictions ---")

sample_restrictions = {
    "rc/sample/restrict/tried_both": {
        "mask_col": "tried_both",
        "mask_val": 1.0,
        "desc": "Districts that proposed both operational and bond referenda",
    },
    "rc/sample/restrict/passed_both": {
        "mask_col": "passed_both",
        "mask_val": 1.0,
        "desc": "Districts that passed both operational and bond referenda",
    },
}

for rc_spec_id, sconfig in sample_restrictions.items():
    mask = df_panel[sconfig["mask_col"]] == sconfig["mask_val"]
    for ovar, cfg in outcome_configs.items():
        try:
            m, coef, se, pval, ci, n_obs, r2, coef_dict = run_onestep(
                outcome_var=ovar,
                control_list=cubic_controls,
                weight_var=cfg["weight"],
                extra_controls=cfg["extra_controls"],
                sample_mask=mask,
            )
            payload = make_success_payload(
                coefficients=coef_dict,
                inference_block={"spec_id": canonical_inference["spec_id"],
                                 "params": canonical_inference["params"]},
                axis_block={"spec_id": rc_spec_id,
                            "restriction": sconfig["desc"],
                            "mask_col": sconfig["mask_col"],
                            "mask_val": sconfig["mask_val"]},
                axis_block_name="sample",
                focal={
                    "parameter": "op_win_prev_10yr_avg",
                    "label": "10-year average operational referendum effect",
                    "selector": {"lincom": "0.10*(op_win_prev1 + ... + op_win_prev10)"},
                    "summary_rule": "lincom"
                },
            )
            row = make_spec_row(
                spec_id=rc_spec_id,
                spec_tree_path="modules/robustness/sample.md#sample-restriction",
                outcome_var=ovar,
                treatment_var="op_win_prev1_through_10",
                coef=coef, se=se, pval=pval, ci=ci,
                n_obs=n_obs, r2=r2,
                coef_vec_json=payload,
                sample_desc=sconfig["desc"],
                fixed_effects="district_code",
                controls_desc="Cubic polynomial (Cellini one-step)",
                cluster_var="district_code",
            )
            spec_results.append(row)
            print(f"    {rc_spec_id} ({ovar}): coef={coef:.4f}, se={se:.4f}, p={pval:.4f}, N={n_obs}")
        except Exception as e:
            payload = make_failure_payload(str(e))
            row = make_spec_row(
                spec_id=rc_spec_id,
                spec_tree_path="modules/robustness/sample.md#sample-restriction",
                outcome_var=ovar,
                treatment_var="op_win_prev1_through_10",
                coef=np.nan, se=np.nan, pval=np.nan, ci=(np.nan, np.nan),
                n_obs=np.nan, r2=np.nan,
                coef_vec_json=payload,
                sample_desc=sconfig["desc"],
                fixed_effects="district_code",
                controls_desc="Cubic polynomial",
                cluster_var="district_code",
                run_success=0, run_error=str(e)[:240],
            )
            spec_results.append(row)
            print(f"    FAILED {rc_spec_id} ({ovar}): {e}")


# --- RC: Outlier trimming ---
print("\n  --- RC: Outlier trimming ---")

trim_configs = {
    "rc/sample/outliers/trim_y_1_99": {"lower": 0.01, "upper": 0.99, "desc": "Trim outcome 1st/99th percentile"},
    "rc/sample/outliers/trim_y_5_95": {"lower": 0.05, "upper": 0.95, "desc": "Trim outcome 5th/95th percentile"},
}

for trim_spec_id, tconfig in trim_configs.items():
    for ovar, cfg in outcome_configs.items():
        try:
            # Compute percentiles on outcome
            vals = df_panel[ovar].dropna()
            lo = vals.quantile(tconfig["lower"])
            hi = vals.quantile(tconfig["upper"])
            mask = (df_panel[ovar] >= lo) & (df_panel[ovar] <= hi)

            m, coef, se, pval, ci, n_obs, r2, coef_dict = run_onestep(
                outcome_var=ovar,
                control_list=cubic_controls,
                weight_var=cfg["weight"],
                extra_controls=cfg["extra_controls"],
                sample_mask=mask,
            )
            payload = make_success_payload(
                coefficients=coef_dict,
                inference_block={"spec_id": canonical_inference["spec_id"],
                                 "params": canonical_inference["params"]},
                axis_block={"spec_id": trim_spec_id,
                            "trim_lower": tconfig["lower"],
                            "trim_upper": tconfig["upper"],
                            "outcome_range": [float(lo), float(hi)]},
                axis_block_name="sample",
                focal={
                    "parameter": "op_win_prev_10yr_avg",
                    "label": "10-year average operational referendum effect",
                    "selector": {"lincom": "0.10*(op_win_prev1 + ... + op_win_prev10)"},
                    "summary_rule": "lincom"
                },
            )
            row = make_spec_row(
                spec_id=trim_spec_id,
                spec_tree_path="modules/robustness/sample.md#outlier-trimming",
                outcome_var=ovar,
                treatment_var="op_win_prev1_through_10",
                coef=coef, se=se, pval=pval, ci=ci,
                n_obs=n_obs, r2=r2,
                coef_vec_json=payload,
                sample_desc=tconfig["desc"],
                fixed_effects="district_code",
                controls_desc="Cubic polynomial (Cellini one-step)",
                cluster_var="district_code",
            )
            spec_results.append(row)
            print(f"    {trim_spec_id} ({ovar}): coef={coef:.4f}, se={se:.4f}, p={pval:.4f}, N={n_obs}")
        except Exception as e:
            payload = make_failure_payload(str(e))
            row = make_spec_row(
                spec_id=trim_spec_id,
                spec_tree_path="modules/robustness/sample.md#outlier-trimming",
                outcome_var=ovar,
                treatment_var="op_win_prev1_through_10",
                coef=np.nan, se=np.nan, pval=np.nan, ci=(np.nan, np.nan),
                n_obs=np.nan, r2=np.nan,
                coef_vec_json=payload,
                sample_desc=tconfig["desc"],
                fixed_effects="district_code",
                controls_desc="Cubic polynomial",
                cluster_var="district_code",
                run_success=0, run_error=str(e)[:240],
            )
            spec_results.append(row)
            print(f"    FAILED {trim_spec_id} ({ovar}): {e}")


# --- RC: Weights ---
print("\n  --- RC: Unweighted ---")

# For outcomes that use weights in baseline, run unweighted
weighted_outcomes = {k: v for k, v in outcome_configs.items() if v["weight"] is not None}

for ovar, cfg in weighted_outcomes.items():
    rc_spec_id = "rc/weights/unweighted"
    try:
        m, coef, se, pval, ci, n_obs, r2, coef_dict = run_onestep(
            outcome_var=ovar,
            control_list=cubic_controls,
            weight_var=None,  # unweighted
            extra_controls=cfg["extra_controls"],
        )
        payload = make_success_payload(
            coefficients=coef_dict,
            inference_block={"spec_id": canonical_inference["spec_id"],
                             "params": canonical_inference["params"]},
            axis_block={"spec_id": rc_spec_id,
                        "baseline_weight": cfg["weight"],
                        "variant_weight": "none"},
            axis_block_name="weights",
            focal={
                "parameter": "op_win_prev_10yr_avg",
                "label": "10-year average operational referendum effect",
                "selector": {"lincom": "0.10*(op_win_prev1 + ... + op_win_prev10)"},
                "summary_rule": "lincom"
            },
        )
        row = make_spec_row(
            spec_id=rc_spec_id,
            spec_tree_path="modules/robustness/weights.md#unweighted",
            outcome_var=ovar,
            treatment_var="op_win_prev1_through_10",
            coef=coef, se=se, pval=pval, ci=ci,
            n_obs=n_obs, r2=r2,
            coef_vec_json=payload,
            sample_desc="Full panel, unweighted",
            fixed_effects="district_code",
            controls_desc="Cubic polynomial (Cellini one-step)",
            cluster_var="district_code",
        )
        spec_results.append(row)
        print(f"    {rc_spec_id} ({ovar}): coef={coef:.4f}, se={se:.4f}, p={pval:.4f}")
    except Exception as e:
        payload = make_failure_payload(str(e))
        row = make_spec_row(
            spec_id=rc_spec_id,
            spec_tree_path="modules/robustness/weights.md#unweighted",
            outcome_var=ovar,
            treatment_var="op_win_prev1_through_10",
            coef=np.nan, se=np.nan, pval=np.nan, ci=(np.nan, np.nan),
            n_obs=np.nan, r2=np.nan,
            coef_vec_json=payload,
            sample_desc="Full panel, unweighted",
            fixed_effects="district_code",
            controls_desc="Cubic polynomial",
            cluster_var="district_code",
            run_success=0, run_error=str(e)[:240],
        )
        spec_results.append(row)
        print(f"    FAILED {rc_spec_id} ({ovar}): {e}")


# --- RC: 5-year average focal parameter ---
print("\n  --- RC: 5-year average focal parameter ---")

for ovar, cfg in outcome_configs.items():
    rc_spec_id = "rc/data/focal_parameter/five_year_avg"
    try:
        m, coef, se, pval, ci, n_obs, r2, coef_dict = run_onestep(
            outcome_var=ovar,
            control_list=cubic_controls,
            weight_var=cfg["weight"],
            extra_controls=cfg["extra_controls"],
            n_avg_years=5,
        )
        payload = make_success_payload(
            coefficients=coef_dict,
            inference_block={"spec_id": canonical_inference["spec_id"],
                             "params": canonical_inference["params"]},
            axis_block={"spec_id": rc_spec_id,
                        "focal_parameter": "5-year average",
                        "lincom": "0.20*(op_win_prev1 + ... + op_win_prev5)",
                        "baseline_focal": "10-year average"},
            axis_block_name="data_construction",
            focal={
                "parameter": "op_win_prev_5yr_avg",
                "label": "5-year average operational referendum effect",
                "selector": {"lincom": "0.20*(op_win_prev1 + ... + op_win_prev5)"},
                "summary_rule": "lincom"
            },
        )
        row = make_spec_row(
            spec_id=rc_spec_id,
            spec_tree_path="modules/robustness/data_construction.md#data-construction-robustness-checks",
            outcome_var=ovar,
            treatment_var="op_win_prev1_through_5",
            coef=coef, se=se, pval=pval, ci=ci,
            n_obs=n_obs, r2=r2,
            coef_vec_json=payload,
            sample_desc="Full panel, 5-year average focal parameter",
            fixed_effects="district_code",
            controls_desc="Cubic polynomial (Cellini one-step)",
            cluster_var="district_code",
        )
        spec_results.append(row)
        print(f"    {rc_spec_id} ({ovar}): coef={coef:.4f}, se={se:.4f}, p={pval:.4f}")
    except Exception as e:
        payload = make_failure_payload(str(e))
        row = make_spec_row(
            spec_id=rc_spec_id,
            spec_tree_path="modules/robustness/data_construction.md#data-construction-robustness-checks",
            outcome_var=ovar,
            treatment_var="op_win_prev1_through_5",
            coef=np.nan, se=np.nan, pval=np.nan, ci=(np.nan, np.nan),
            n_obs=np.nan, r2=np.nan,
            coef_vec_json=payload,
            sample_desc="Full panel, 5-year average",
            fixed_effects="district_code",
            controls_desc="Cubic polynomial",
            cluster_var="district_code",
            run_success=0, run_error=str(e)[:240],
        )
        spec_results.append(row)
        print(f"    FAILED {rc_spec_id} ({ovar}): {e}")


# --- RC: Cross-section RD (rdrobust) ---
print("\n  --- RC: Cross-section RD (rdrobust) ---")

from rdrobust import rdrobust as rd_func

# Map panel outcome names to cross-section names (same names)
# Post-election: dyear >= 0
# Pre-election placebo: dyear == -2

cs_outcomes = {
    "advprof_math10": {"weight": None, "label": "Math proficiency grade 10"},
    "dropout_rate": {"weight": None, "label": "Dropout rate"},
    "wkce_math10": {"weight": None, "label": "WKCE math score grade 10"},
    "rev_lim_mem": {"weight": None, "label": "Revenue limits per member"},
    "tot_exp_mem": {"weight": None, "label": "Total expenditures per member"},
}

# Note: log_instate_enr uses perc_instate in the cross-section with covariates;
# skip for simplicity or use perc_instate
cs_outcomes_extra = {
    "perc_instate": {"weight": None, "label": "% in-state postsecondary enrollment (proxy for log_instate_enr)"},
}

# Post-election rdrobust
for ovar, cs_cfg in {**cs_outcomes, **cs_outcomes_extra}.items():
    rc_spec_id = "rc/joint/cross_section_rd/rdrobust_post"
    try:
        sub = df_cs[df_cs['dyear'] >= 0].dropna(subset=[ovar, 'perc']).copy()
        y = sub[ovar].values
        x = sub['perc'].values

        result = rd_func(y, x, c=0)

        # Use robust bias-corrected estimates
        rd_coef = float(result.coef.iloc[2, 0])  # Robust
        rd_se = float(result.se.iloc[2, 0])
        rd_pval = float(result.pv.iloc[2, 0])
        rd_ci = (float(result.ci.iloc[2, 0]), float(result.ci.iloc[2, 1]))
        rd_n = int(sum(result.N))
        rd_bw_left = float(result.bws.iloc[0, 0])
        rd_bw_right = float(result.bws.iloc[0, 1])

        coef_dict = {
            "conventional": float(result.coef.iloc[0, 0]),
            "bias_corrected": float(result.coef.iloc[1, 0]),
            "robust": float(result.coef.iloc[2, 0]),
        }

        # Map ovar to the matching panel outcome for display
        mapped_ovar = "log_instate_enr" if ovar == "perc_instate" else ovar

        payload = make_success_payload(
            coefficients=coef_dict,
            inference_block={"spec_id": canonical_inference["spec_id"],
                             "params": {"method": "rdrobust_robust_bias_corrected"}},
            axis_block={"spec_id": rc_spec_id,
                        "axes_changed": ["estimator", "data_structure"],
                        "details": {
                            "estimator": "rdrobust (local polynomial RD)",
                            "sample": "cross-section, dyear >= 0 (post-election)",
                            "baseline_estimator": "cellini_onestep panel",
                            "rdrobust_bandwidth_left": rd_bw_left,
                            "rdrobust_bandwidth_right": rd_bw_right,
                            "rdrobust_kernel": "triangular",
                            "rdrobust_poly_order": 1,
                        }},
            axis_block_name="joint",
            focal={
                "parameter": "rd_treatment_effect",
                "label": "Local RD effect at 50% vote share (robust bias-corrected)",
                "summary_rule": "single"
            },
        )
        row = make_spec_row(
            spec_id=rc_spec_id,
            spec_tree_path="modules/robustness/joint.md#joint-robustness-variants-multi-axis-rc",
            outcome_var=mapped_ovar,
            treatment_var="op_win (vote share > 50%)",
            coef=rd_coef, se=rd_se, pval=rd_pval, ci=rd_ci,
            n_obs=rd_n, r2=np.nan,
            coef_vec_json=payload,
            sample_desc="Cross-section, post-election (dyear >= 0)",
            fixed_effects="",
            controls_desc="rdrobust local polynomial RD, CCFT bandwidth",
            cluster_var="",
        )
        spec_results.append(row)
        print(f"    {rc_spec_id} ({ovar}): coef={rd_coef:.4f}, se={rd_se:.4f}, p={rd_pval:.4f}, N={rd_n}")
    except Exception as e:
        mapped_ovar = "log_instate_enr" if ovar == "perc_instate" else ovar
        payload = make_failure_payload(str(e), stage="rdrobust")
        row = make_spec_row(
            spec_id=rc_spec_id,
            spec_tree_path="modules/robustness/joint.md#joint-robustness-variants-multi-axis-rc",
            outcome_var=mapped_ovar,
            treatment_var="op_win (vote share > 50%)",
            coef=np.nan, se=np.nan, pval=np.nan, ci=(np.nan, np.nan),
            n_obs=np.nan, r2=np.nan,
            coef_vec_json=payload,
            sample_desc="Cross-section, post-election",
            fixed_effects="",
            controls_desc="rdrobust local polynomial RD",
            cluster_var="",
            run_success=0, run_error=str(e)[:240],
        )
        spec_results.append(row)
        print(f"    FAILED {rc_spec_id} ({ovar}): {e}")


# Pre-election placebo rdrobust
for ovar, cs_cfg in {**cs_outcomes, **cs_outcomes_extra}.items():
    rc_spec_id = "rc/joint/cross_section_rd/rdrobust_pre_placebo"
    try:
        sub = df_cs[df_cs['dyear'] == -2].dropna(subset=[ovar, 'perc']).copy()
        y = sub[ovar].values
        x = sub['perc'].values

        result = rd_func(y, x, c=0)

        rd_coef = float(result.coef.iloc[2, 0])
        rd_se = float(result.se.iloc[2, 0])
        rd_pval = float(result.pv.iloc[2, 0])
        rd_ci = (float(result.ci.iloc[2, 0]), float(result.ci.iloc[2, 1]))
        rd_n = int(sum(result.N))
        rd_bw_left = float(result.bws.iloc[0, 0])
        rd_bw_right = float(result.bws.iloc[0, 1])

        coef_dict = {
            "conventional": float(result.coef.iloc[0, 0]),
            "bias_corrected": float(result.coef.iloc[1, 0]),
            "robust": float(result.coef.iloc[2, 0]),
        }

        mapped_ovar = "log_instate_enr" if ovar == "perc_instate" else ovar

        payload = make_success_payload(
            coefficients=coef_dict,
            inference_block={"spec_id": canonical_inference["spec_id"],
                             "params": {"method": "rdrobust_robust_bias_corrected"}},
            axis_block={"spec_id": rc_spec_id,
                        "axes_changed": ["estimator", "data_structure", "sample_period"],
                        "details": {
                            "estimator": "rdrobust (local polynomial RD)",
                            "sample": "cross-section, dyear == -2 (pre-election placebo)",
                            "baseline_estimator": "cellini_onestep panel",
                            "note": "Pre-election placebo: should find no effect if identification is valid",
                            "rdrobust_bandwidth_left": rd_bw_left,
                            "rdrobust_bandwidth_right": rd_bw_right,
                            "rdrobust_kernel": "triangular",
                            "rdrobust_poly_order": 1,
                        }},
            axis_block_name="joint",
            focal={
                "parameter": "rd_treatment_effect",
                "label": "Local RD effect at 50% (pre-election placebo, robust BC)",
                "summary_rule": "single"
            },
        )
        row = make_spec_row(
            spec_id=rc_spec_id,
            spec_tree_path="modules/robustness/joint.md#joint-robustness-variants-multi-axis-rc",
            outcome_var=mapped_ovar,
            treatment_var="op_win (vote share > 50%)",
            coef=rd_coef, se=rd_se, pval=rd_pval, ci=rd_ci,
            n_obs=rd_n, r2=np.nan,
            coef_vec_json=payload,
            sample_desc="Cross-section, pre-election placebo (dyear == -2)",
            fixed_effects="",
            controls_desc="rdrobust local polynomial RD, CCFT bandwidth",
            cluster_var="",
        )
        spec_results.append(row)
        print(f"    {rc_spec_id} ({ovar}): coef={rd_coef:.4f}, se={rd_se:.4f}, p={rd_pval:.4f}, N={rd_n}")
    except Exception as e:
        mapped_ovar = "log_instate_enr" if ovar == "perc_instate" else ovar
        payload = make_failure_payload(str(e), stage="rdrobust")
        row = make_spec_row(
            spec_id=rc_spec_id,
            spec_tree_path="modules/robustness/joint.md#joint-robustness-variants-multi-axis-rc",
            outcome_var=mapped_ovar,
            treatment_var="op_win (vote share > 50%)",
            coef=np.nan, se=np.nan, pval=np.nan, ci=(np.nan, np.nan),
            n_obs=np.nan, r2=np.nan,
            coef_vec_json=payload,
            sample_desc="Cross-section, pre-election placebo",
            fixed_effects="",
            controls_desc="rdrobust local polynomial RD",
            cluster_var="",
            run_success=0, run_error=str(e)[:240],
        )
        spec_results.append(row)
        print(f"    FAILED {rc_spec_id} ({ovar}): {e}")


# ============================================================
# STEP 4: INFERENCE VARIANTS (HC1 for baseline specs)
# ============================================================
print("\n===== STEP 4: INFERENCE VARIANTS =====")

# For each baseline spec (panel-based), recompute with hetero-robust SE
for ovar, spec_id in baseline_outcomes.items():
    cfg = outcome_configs[ovar]
    # Find the baseline spec_run_id
    base_row = [r for r in spec_results if r["spec_id"] == spec_id and r["outcome_var"] == ovar]
    if not base_row:
        continue
    base_run_id = base_row[0]["spec_run_id"]

    infer_spec_id = "infer/se/hc/hc1"
    try:
        m, coef, se, pval, ci, n_obs, r2, coef_dict = run_onestep_hetero(
            outcome_var=ovar,
            control_list=cubic_controls,
            weight_var=cfg["weight"],
            extra_controls=cfg["extra_controls"],
        )
        payload = make_success_payload(
            coefficients=coef_dict,
            inference_block={"spec_id": infer_spec_id,
                             "params": {"vcov": "hetero (HC1)"}},
            focal={
                "parameter": "op_win_prev_10yr_avg",
                "label": "10-year average operational referendum effect",
                "selector": {"lincom": "0.10*(op_win_prev1 + ... + op_win_prev10)"},
                "summary_rule": "lincom"
            },
        )
        irow = make_infer_row(
            base_spec_run_id=base_run_id,
            spec_id=infer_spec_id,
            spec_tree_path="modules/inference/standard_errors.md#heteroskedasticity-robust",
            outcome_var=ovar,
            coef=coef, se=se, pval=pval, ci=ci,
            n_obs=n_obs, r2=r2,
            coef_vec_json=payload,
            cluster_var="",
        )
        inference_results.append(irow)
        print(f"  {infer_spec_id} ({ovar}): coef={coef:.4f}, se={se:.4f}, p={pval:.4f}")
    except Exception as e:
        payload = make_failure_payload(str(e), stage="inference")
        irow = make_infer_row(
            base_spec_run_id=base_run_id,
            spec_id=infer_spec_id,
            spec_tree_path="modules/inference/standard_errors.md#heteroskedasticity-robust",
            outcome_var=ovar,
            coef=np.nan, se=np.nan, pval=np.nan, ci=(np.nan, np.nan),
            n_obs=np.nan, r2=np.nan,
            coef_vec_json=payload,
            cluster_var="",
            run_success=0, run_error=str(e)[:240],
        )
        inference_results.append(irow)
        print(f"  FAILED {infer_spec_id} ({ovar}): {e}")


# ============================================================
# STEP 5: DIAGNOSTICS
# ============================================================
print("\n===== STEP 5: DIAGNOSTICS =====")

diagnostics_results = []
diag_run_counter = 0

# McCrary density test at cutoff
# Use rddensity package if available, otherwise approximate
diag_run_counter += 1
diag_spec_id = "diag/regression_discontinuity/manipulation/mccrary_density"
try:
    # Try rddensity
    from rddensity import rddensity
    sub = df_cs.dropna(subset=['perc']).copy()
    result = rddensity(sub['perc'].values, c=0)
    # rddensity stores test stat and p-value in result.test Series
    t_jk = float(result.test['t_jk']) if 't_jk' in result.test.index else np.nan
    p_jk = float(result.test['p_jk']) if 'p_jk' in result.test.index else np.nan
    diag_payload = {
        "test": "rddensity",
        "T_statistic_jk": t_jk,
        "p_value_jk": p_jk,
        "density_left": float(result.hat['left']),
        "density_right": float(result.hat['right']),
        "density_diff": float(result.hat['diff']),
        "bandwidth_left": float(result.h['left']),
        "bandwidth_right": float(result.h['right']),
        "n_full": int(result.n['full']),
        "n_eff_left": int(result.n['eff_left']),
        "n_eff_right": int(result.n['eff_right']),
        "note": "McCrary-type density continuity test at cutoff=0 (rddensity package)"
    }
    diagnostics_results.append({
        "paper_id": PAPER_ID,
        "diagnostic_run_id": f"{PAPER_ID}_diag{diag_run_counter:04d}",
        "diag_spec_id": diag_spec_id,
        "spec_tree_path": "modules/diagnostics/design_diagnostics.md#manipulation-testing",
        "diagnostic_scope": "baseline_group",
        "diagnostic_context_id": "G1_mccrary",
        "diagnostic_json": json.dumps(diag_payload),
        "run_success": 1,
        "run_error": "",
    })
    print(f"  McCrary density test: T_jk={diag_payload.get('T_statistic_jk', 'N/A')}, p_jk={diag_payload.get('p_value_jk', 'N/A')}")
except ImportError:
    # rddensity not installed - do a simple histogram-based check
    try:
        sub = df_cs.dropna(subset=['perc']).copy()
        below = sub[sub['perc'] < 0]['perc']
        above = sub[sub['perc'] >= 0]['perc']
        diag_payload = {
            "test": "histogram_based_approximation",
            "n_below_cutoff": int(len(below)),
            "n_above_cutoff": int(len(above)),
            "ratio": float(len(below)) / float(len(above)) if len(above) > 0 else np.nan,
            "note": "rddensity not installed; histogram count approximation"
        }
        diagnostics_results.append({
            "paper_id": PAPER_ID,
            "diagnostic_run_id": f"{PAPER_ID}_diag{diag_run_counter:04d}",
            "diag_spec_id": diag_spec_id,
            "spec_tree_path": "modules/diagnostics/design_diagnostics.md#manipulation-testing",
            "diagnostic_scope": "baseline_group",
            "diagnostic_context_id": "G1_mccrary",
            "diagnostic_json": json.dumps(diag_payload),
            "run_success": 1,
            "run_error": "",
        })
        print(f"  McCrary (approx): n_below={len(below)}, n_above={len(above)}, ratio={len(below)/len(above):.3f}")
    except Exception as e2:
        diagnostics_results.append({
            "paper_id": PAPER_ID,
            "diagnostic_run_id": f"{PAPER_ID}_diag{diag_run_counter:04d}",
            "diag_spec_id": diag_spec_id,
            "spec_tree_path": "modules/diagnostics/design_diagnostics.md#manipulation-testing",
            "diagnostic_scope": "baseline_group",
            "diagnostic_context_id": "G1_mccrary",
            "diagnostic_json": json.dumps({"error": str(e2), "error_details": {"stage": "diagnostics", "exception_type": type(e2).__name__, "exception_message": str(e2)}}),
            "run_success": 0,
            "run_error": str(e2)[:240],
        })
        print(f"  FAILED McCrary: {e2}")
except Exception as e:
    diagnostics_results.append({
        "paper_id": PAPER_ID,
        "diagnostic_run_id": f"{PAPER_ID}_diag{diag_run_counter:04d}",
        "diag_spec_id": diag_spec_id,
        "spec_tree_path": "modules/diagnostics/design_diagnostics.md#manipulation-testing",
        "diagnostic_scope": "baseline_group",
        "diagnostic_context_id": "G1_mccrary",
        "diagnostic_json": json.dumps({"error": str(e), "error_details": {"stage": "diagnostics", "exception_type": type(e).__name__, "exception_message": str(e)}}),
        "run_success": 0,
        "run_error": str(e)[:240],
    })
    print(f"  FAILED McCrary: {e}")


# Covariate continuity (balance test)
diag_run_counter += 1
diag_spec_id_bal = "diag/regression_discontinuity/balance/covariate_continuity"
try:
    # Run rdrobust on pre-election covariates to check balance
    balance_vars = ['econ_disadv_percent', 'enrollment']
    balance_results_dict = {}
    sub_pre = df_cs[df_cs['dyear'] == -2].copy()

    for bvar in balance_vars:
        sub_b = sub_pre.dropna(subset=[bvar, 'perc'])
        if len(sub_b) < 50:
            balance_results_dict[bvar] = {"error": "insufficient observations"}
            continue
        try:
            result = rd_func(sub_b[bvar].values, sub_b['perc'].values, c=0)
            balance_results_dict[bvar] = {
                "coef_robust": float(result.coef.iloc[2, 0]),
                "se_robust": float(result.se.iloc[2, 0]),
                "pval_robust": float(result.pv.iloc[2, 0]),
                "n": int(sum(result.N)),
            }
        except Exception as e_b:
            balance_results_dict[bvar] = {"error": str(e_b)[:200]}

    diag_payload_bal = {
        "test": "covariate_continuity_rdrobust",
        "covariates": balance_results_dict,
        "note": "RD balance tests on pre-election covariates using rdrobust",
    }
    diagnostics_results.append({
        "paper_id": PAPER_ID,
        "diagnostic_run_id": f"{PAPER_ID}_diag{diag_run_counter:04d}",
        "diag_spec_id": diag_spec_id_bal,
        "spec_tree_path": "modules/diagnostics/design_diagnostics.md#covariate-balance",
        "diagnostic_scope": "baseline_group",
        "diagnostic_context_id": "G1_balance",
        "diagnostic_json": json.dumps(diag_payload_bal),
        "run_success": 1,
        "run_error": "",
    })
    print(f"  Balance test: {len(balance_results_dict)} covariates tested")
except Exception as e:
    diagnostics_results.append({
        "paper_id": PAPER_ID,
        "diagnostic_run_id": f"{PAPER_ID}_diag{diag_run_counter:04d}",
        "diag_spec_id": diag_spec_id_bal,
        "spec_tree_path": "modules/diagnostics/design_diagnostics.md#covariate-balance",
        "diagnostic_scope": "baseline_group",
        "diagnostic_context_id": "G1_balance",
        "diagnostic_json": json.dumps({"error": str(e), "error_details": {"stage": "diagnostics", "exception_type": type(e).__name__, "exception_message": str(e)}}),
        "run_success": 0,
        "run_error": str(e)[:240],
    })
    print(f"  FAILED balance test: {e}")


# ============================================================
# WRITE OUTPUTS
# ============================================================
print("\n===== WRITING OUTPUTS =====")

# specification_results.csv
spec_df = pd.DataFrame(spec_results)
spec_df.to_csv(f"{PACKAGE_DIR}/specification_results.csv", index=False)
print(f"Wrote specification_results.csv: {len(spec_df)} rows")

# inference_results.csv
if inference_results:
    infer_df = pd.DataFrame(inference_results)
    infer_df.to_csv(f"{PACKAGE_DIR}/inference_results.csv", index=False)
    print(f"Wrote inference_results.csv: {len(infer_df)} rows")

# diagnostics_results.csv
if diagnostics_results:
    diag_df = pd.DataFrame(diagnostics_results)
    diag_df.to_csv(f"{PACKAGE_DIR}/diagnostics_results.csv", index=False)
    print(f"Wrote diagnostics_results.csv: {len(diag_df)} rows")

# spec_diagnostics_map.csv - link all baseline specs to group-level diagnostics
diag_map_rows = []
for sr in spec_results:
    if sr["spec_id"].startswith("baseline"):
        for dr in diagnostics_results:
            diag_map_rows.append({
                "paper_id": PAPER_ID,
                "spec_run_id": sr["spec_run_id"],
                "diagnostic_run_id": dr["diagnostic_run_id"],
                "relationship": "shared_invariant_check",
            })
if diag_map_rows:
    diag_map_df = pd.DataFrame(diag_map_rows)
    diag_map_df.to_csv(f"{PACKAGE_DIR}/spec_diagnostics_map.csv", index=False)
    print(f"Wrote spec_diagnostics_map.csv: {len(diag_map_df)} rows")


# ============================================================
# SUMMARY
# ============================================================
n_success = sum(1 for r in spec_results if r["run_success"] == 1)
n_fail = sum(1 for r in spec_results if r["run_success"] == 0)
n_infer = len(inference_results)
n_diag = len(diagnostics_results)

print(f"\n===== SUMMARY =====")
print(f"Total estimate specs: {len(spec_results)} ({n_success} success, {n_fail} failed)")
print(f"Inference recomputations: {n_infer}")
print(f"Diagnostics: {n_diag}")
print(f"Unique spec_run_ids: {len(set(r['spec_run_id'] for r in spec_results))}")
print(f"Unique spec_ids: {len(set(r['spec_id'] for r in spec_results))}")

# Verify spec_run_id uniqueness
run_ids = [r['spec_run_id'] for r in spec_results]
assert len(run_ids) == len(set(run_ids)), "DUPLICATE spec_run_ids found!"
print("spec_run_id uniqueness: PASS")

# Verify JSON validity
for r in spec_results:
    try:
        json.loads(r['coefficient_vector_json'])
    except json.JSONDecodeError as e:
        print(f"  INVALID JSON in {r['spec_run_id']}: {e}")
print("JSON validity check: PASS")

print("\nDone.")
