"""
Specification Search Script for 113517-V1
"The Relative Power of Employment-to-Employment Reallocation
and Unemployment Exits in Predicting Wage Growth"
Moscarini & Postel-Vinay, AER P&P 2017

Surface-driven specification search using the approved SPECIFICATION_SURFACE.json.
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import warnings
import time
import gc
import sys
import hashlib
import os

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

# ==== Paths ====
REPO_ROOT = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search'
PKG_DIR = f'{REPO_ROOT}/data/downloads/extracted/113517-V1'
DATA_DIR = f'{PKG_DIR}/Codes-and-data'
SURFACE_PATH = f'{PKG_DIR}/SPECIFICATION_SURFACE.json'

# ==== Surface hash ====
with open(SURFACE_PATH, 'rb') as f:
    SURFACE_HASH = f'sha256:{hashlib.sha256(f.read()).hexdigest()}'

# ==== Software block ====
SOFTWARE = {
    "runner_language": "python",
    "runner_version": "3.12.7",
    "packages": {"pyfixest": "0.40.1", "pandas": "2.2.3", "numpy": str(np.__version__)}
}

# ==== Design audit (from surface) ====
DESIGN_AUDIT = {
    "cross_sectional_ols": {
        "estimator": "two_stage_areg",
        "first_stage": "areg transition_rate ~ covariates [w=wgt], a(mkt_t)",
        "second_stage": "areg xdw ~ predicted_flows + year_month [w=wgt], a(mkt)",
        "panel_unit": "mkt (sex x race x agegroup x education)",
        "panel_time": "year_month",
        "fe_structure": ["mkt (second stage)", "mkt_t (first stage)"],
        "weights": "wgt (analytic weights from SIPP)"
    }
}

CANONICAL_INFERENCE = {"spec_id": "infer/se/default/iid", "params": {"vcov": "iid"}}

PAPER_ID = "113517-V1"
BASELINE_GROUP_ID = "G1"

# ==== Load data ====
print("Loading data...")
t0 = time.time()
df = pd.read_parquet(f'{DATA_DIR}/preprocessed.parquet')
print(f"Loaded in {time.time()-t0:.1f}s, shape: {df.shape}")

# Prepare types
for v in ['lagstate', 'laguni', 'lagsiz', 'lagocc', 'lagind', 'lagpub', 'mkt_t', 'mkt']:
    df[v] = df[v].astype('Int32')

df['ym_num'] = df['year_month_num'].astype('float32')

# Create wage changes
for dv in ['logern_nom', 'logern', 'loghwr_nom', 'loghwr']:
    df[f'd{dv}'] = df[dv] - df[f'lag{dv}']

# Eligibility for hourly wage
df['EZeligible_hw'] = ((df['EZeligible'] == 1) & (df['lagphr'] == 1)).astype('int8')
df['DWeligible_hw'] = ((df['DWeligible'] == 1) & (df['lagphr'] == 1)).astype('int8')

# Drop unneeded columns
drop_now = [c for c in df.columns if c in [
    'year_month', 'year_month_num', 'loghrs', 'lagphr',
    'clw', 'siz', 'ind', 'occ', 'phr', 'married', 'state', 'uni',
    'sex', 'race', 'education', 'agegroup', 'panel_id',
]]
df.drop(columns=drop_now, inplace=True)
gc.collect()
print(f"Shape after cleanup: {df.shape}")

# ==== Control variable formulas ====
e_controls = "C(lagstate) + C(laguni) + C(lagsiz) + C(lagocc) + C(lagind) + C(lagpub)"
u_controls = "C(lagstate)"


# ==== Helper functions ====
def run_first_stage(data, depv, rhs_formula, elig_col):
    all_vars = [depv, 'wgt']
    for term in rhs_formula.split('+'):
        term = term.strip()
        if term.startswith('C('):
            all_vars.append(term[2:-1])
        else:
            all_vars.append(term)
    all_vars.append('mkt_t')
    mask = (data[elig_col] == 1) & (data['wgt'] > 0)
    for v in all_vars:
        mask = mask & data[v].notna()
    sub = data.loc[mask, all_vars]
    formula = f"{depv} ~ {rhs_formula} | mkt_t"
    m = pf.feols(formula, data=sub, weights='wgt')
    fe = m.fixef()
    fe_key = [k for k in fe.keys() if 'mkt_t' in k][0]
    fe_dict = fe[fe_key]
    del m, sub
    gc.collect()
    return fe_dict


def map_fe_all(df, fe_dict, colname):
    fe_map = {int(k): v for k, v in fe_dict.items()}
    df[colname] = df['mkt_t'].map(fe_map)
    return df


def make_coef_json(coef_dict, extra_blocks=None):
    payload = {
        "coefficients": {k: float(v) for k, v in coef_dict.items()},
        "inference": CANONICAL_INFERENCE,
        "software": SOFTWARE,
        "surface_hash": SURFACE_HASH,
        "design": DESIGN_AUDIT
    }
    if extra_blocks:
        payload.update(extra_blocks)
    return json.dumps(payload)


def make_failure_json(error_msg, stage="estimation"):
    return json.dumps({
        "error": error_msg,
        "error_details": {
            "stage": stage,
            "exception_type": "RuntimeError",
            "exception_message": error_msg
        }
    })


def run_second_stage(data, formula, focal_var, weights_col='wgt', sample_mask=None):
    lhs = formula.split('~')[0].strip()
    rhs_and_fe = formula.split('~')[1]
    if '|' in rhs_and_fe:
        rhs_part = rhs_and_fe.split('|')[0].strip()
        fe_part = rhs_and_fe.split('|')[1].strip()
    else:
        rhs_part = rhs_and_fe.strip()
        fe_part = None
    rhs_vars = [v.strip() for v in rhs_part.split('+')]
    all_needed = [lhs] + rhs_vars
    if fe_part:
        for fv in fe_part.split('+'):
            all_needed.append(fv.strip())
    if weights_col:
        all_needed.append(weights_col)
    mask = pd.Series(True, index=data.index)
    if weights_col:
        mask = mask & (data[weights_col] > 0)
    for v in all_needed:
        if v in data.columns:
            mask = mask & data[v].notna()
    if sample_mask is not None:
        mask = mask & sample_mask
    sub = data.loc[mask]
    if weights_col:
        m = pf.feols(formula, data=sub, weights=weights_col)
    else:
        m = pf.feols(formula, data=sub)
    coef = m.coef()
    se = m.se()
    pvals = m.pvalue()
    ci = m.confint()
    n_obs = int(m._N)
    r2 = float(m._r2)
    ci_lower = float(ci.loc[focal_var, '2.5%']) if focal_var in ci.index else np.nan
    ci_upper = float(ci.loc[focal_var, '97.5%']) if focal_var in ci.index else np.nan
    coef_dict = {k: float(v) for k, v in coef.items()}
    del m, sub
    gc.collect()
    return {
        'coef': float(coef[focal_var]), 'se': float(se[focal_var]),
        'pval': float(pvals[focal_var]),
        'ci_lower': ci_lower, 'ci_upper': ci_upper,
        'n_obs': n_obs, 'r2': r2, 'coef_dict': coef_dict
    }


# ==== Output storage ====
all_results = []
inference_results = []
spec_run_counter = 0


def add_result(spec_id, spec_tree_path, outcome_var, treatment_var, res,
               controls_desc, fixed_effects, sample_desc, cluster_var="",
               extra_json_blocks=None):
    global spec_run_counter
    spec_run_counter += 1
    all_results.append({
        'paper_id': PAPER_ID,
        'spec_run_id': f'{PAPER_ID}_R{spec_run_counter:03d}',
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'baseline_group_id': BASELINE_GROUP_ID,
        'outcome_var': outcome_var,
        'treatment_var': treatment_var,
        'coefficient': res['coef'],
        'std_error': res['se'],
        'p_value': res['pval'],
        'ci_lower': res['ci_lower'],
        'ci_upper': res['ci_upper'],
        'n_obs': res['n_obs'],
        'r_squared': res['r2'],
        'coefficient_vector_json': make_coef_json(res['coef_dict'], extra_json_blocks),
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'run_success': 1,
        'run_error': ''
    })
    return f'{PAPER_ID}_R{spec_run_counter:03d}'


def add_failure(spec_id, spec_tree_path, outcome_var, treatment_var, error_msg,
                controls_desc="", fixed_effects="", sample_desc=""):
    global spec_run_counter
    spec_run_counter += 1
    all_results.append({
        'paper_id': PAPER_ID,
        'spec_run_id': f'{PAPER_ID}_R{spec_run_counter:03d}',
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'baseline_group_id': BASELINE_GROUP_ID,
        'outcome_var': outcome_var,
        'treatment_var': treatment_var,
        'coefficient': np.nan, 'std_error': np.nan, 'p_value': np.nan,
        'ci_lower': np.nan, 'ci_upper': np.nan,
        'n_obs': np.nan, 'r_squared': np.nan,
        'coefficient_vector_json': make_failure_json(error_msg),
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': '',
        'run_success': 0,
        'run_error': error_msg
    })


def add_inference(base_run_id, infer_spec_id, infer_tree_path,
                  outcome_var, treatment_var, res, cluster_var=""):
    payload = {
        "coefficients": {k: float(v) for k, v in res['coef_dict'].items()},
        "inference": {"spec_id": infer_spec_id, "params": {"cluster_var": cluster_var} if cluster_var else {}},
        "software": SOFTWARE,
        "surface_hash": SURFACE_HASH
    }
    inference_results.append({
        'paper_id': PAPER_ID,
        'inference_run_id': f'{PAPER_ID}_I{len(inference_results)+1:03d}',
        'spec_run_id': base_run_id,
        'spec_id': infer_spec_id,
        'spec_tree_path': infer_tree_path,
        'baseline_group_id': BASELINE_GROUP_ID,
        'outcome_var': outcome_var,
        'treatment_var': treatment_var,
        'coefficient': res['coef'], 'std_error': res['se'], 'p_value': res['pval'],
        'ci_lower': res['ci_lower'], 'ci_upper': res['ci_upper'],
        'n_obs': res['n_obs'], 'r_squared': res['r2'],
        'coefficient_vector_json': json.dumps(payload),
        'cluster_var': cluster_var,
        'run_success': 1, 'run_error': ''
    })


# ==== FIRST STAGE ====
print("\n--- Shared First-Stage Regressions ---")
fe_ue = run_first_stage(df, 'uetrans_i', u_controls, 'UZeligible')
fe_ne = run_first_stage(df, 'netrans_i', u_controls, 'NZeligible')
fe_ur = run_first_stage(df, 'unm', u_controls, 'UReligible')
fe_eu_earn = run_first_stage(df, 'eutrans_i', e_controls, 'EZeligible')
fe_en_earn = run_first_stage(df, 'entrans_i', e_controls, 'EZeligible')
fe_eu_hw = run_first_stage(df, 'eutrans_i', e_controls, 'EZeligible_hw')
fe_en_hw = run_first_stage(df, 'entrans_i', e_controls, 'EZeligible_hw')
gc.collect()
print(f"  First-stage done at {time.time()-t0:.1f}s")

median_ym = df['ym_num'].median()

# ==== MAIN LOOP ====
depvarlist = ['logern_nom', 'logern', 'loghwr_nom', 'loghwr']
depvar_labels = {
    'logern_nom': 'Log Nominal Earnings', 'logern': 'Log Real Earnings',
    'loghwr_nom': 'Log Nominal Hourly Wage', 'loghwr': 'Log Real Hourly Wage'
}

for depvar in depvarlist:
    lagdepvar = f'lag{depvar}'
    dvar = f'd{depvar}'
    xdvar = f'xd{depvar}'
    ez_col = 'EZeligible_hw' if depvar in ['loghwr', 'loghwr_nom'] else 'EZeligible'
    dw_col = 'DWeligible_hw' if depvar in ['loghwr', 'loghwr_nom'] else 'DWeligible'
    dv_label = depvar_labels[depvar]
    base_sample = f'{dv_label}, all workers'

    print(f"\n{'='*60}")
    print(f"Processing: {depvar}")
    print(f"{'='*60}")

    fe_ee = run_first_stage(df, 'eetrans_i', f'{lagdepvar} + {e_controls}', ez_col)
    fe_dw = run_first_stage(df, dvar, f'eetrans_i + {e_controls}', dw_col)
    fe_eu = fe_eu_hw if depvar in ['loghwr', 'loghwr_nom'] else fe_eu_earn
    fe_en = fe_en_hw if depvar in ['loghwr', 'loghwr_nom'] else fe_en_earn

    map_fe_all(df, fe_ee, 'xee')
    map_fe_all(df, fe_ue, 'xue')
    map_fe_all(df, fe_ne, 'xne')
    map_fe_all(df, fe_eu, 'xeu')
    map_fe_all(df, fe_en, 'xen')
    map_fe_all(df, fe_ur, 'xur')
    map_fe_all(df, fe_dw, xdvar)
    df.loc[df[dw_col] != 1, xdvar] = np.nan
    df['xnue'] = df['xue'] + df['xne']
    df['xenu'] = df['xen'] + df['xeu']
    df['xee_i'] = df['xee'] * df['eetrans_i']

    baseline_formula = f"{xdvar} ~ xee + xue + xur + xne + xen + xeu + ym_num | mkt"

    # ==== BASELINE ====
    base_spec_id = "baseline" if depvar == 'logern_nom' else f"baseline__{depvar}_spec6"
    try:
        res = run_second_stage(df, baseline_formula, 'xee')
        run_id = add_result(
            base_spec_id, "designs/cross_sectional_ols.md#baseline",
            xdvar, 'xee', res,
            "xue + xur + xne + xen + xeu + ym_num", "mkt", base_sample
        )
        print(f"  {base_spec_id}: coef={res['coef']:.6f}, se={res['se']:.6f}, N={res['n_obs']}")

        # Inference variants for baseline
        sub_mask = (df['wgt'] > 0)
        for v in [xdvar, 'xee', 'xue', 'xur', 'xne', 'xen', 'xeu', 'ym_num', 'mkt']:
            sub_mask = sub_mask & df[v].notna()
        sub = df.loc[sub_mask]

        try:
            m_hc = pf.feols(baseline_formula, data=sub, weights='wgt', vcov='hetero')
            res_hc = {
                'coef': float(m_hc.coef()['xee']), 'se': float(m_hc.se()['xee']),
                'pval': float(m_hc.pvalue()['xee']),
                'ci_lower': float(m_hc.confint().loc['xee', '2.5%']),
                'ci_upper': float(m_hc.confint().loc['xee', '97.5%']),
                'n_obs': int(m_hc._N), 'r2': float(m_hc._r2),
                'coef_dict': {k: float(v) for k, v in m_hc.coef().items()}
            }
            add_inference(run_id, "infer/se/hc/hc1",
                         "modules/inference/standard_errors.md#heteroskedasticity-robust",
                         xdvar, 'xee', res_hc)
            del m_hc
        except Exception as e:
            print(f"    HC1 failed: {e}")

        try:
            m_cl = pf.feols(baseline_formula, data=sub, weights='wgt', vcov={"CRV1": "mkt"})
            res_cl = {
                'coef': float(m_cl.coef()['xee']), 'se': float(m_cl.se()['xee']),
                'pval': float(m_cl.pvalue()['xee']),
                'ci_lower': float(m_cl.confint().loc['xee', '2.5%']),
                'ci_upper': float(m_cl.confint().loc['xee', '97.5%']),
                'n_obs': int(m_cl._N), 'r2': float(m_cl._r2),
                'coef_dict': {k: float(v) for k, v in m_cl.coef().items()}
            }
            add_inference(run_id, "infer/se/cluster/mkt",
                         "modules/inference/standard_errors.md#clustering",
                         xdvar, 'xee', res_cl, cluster_var="mkt")
            del m_cl
        except Exception as e:
            print(f"    Cluster failed: {e}")
        del sub

    except Exception as e:
        add_failure(base_spec_id, "designs/cross_sectional_ols.md#baseline",
                   xdvar, 'xee', str(e), sample_desc=base_sample)

    # ==== RC specs only for primary depvar (logern_nom) ====
    if depvar != 'logern_nom':
        for col in ['xee', 'xue', 'xne', 'xeu', 'xen', 'xur', xdvar, 'xnue', 'xenu', 'xee_i']:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
        gc.collect()
        continue

    # ==== CONTROLS PROGRESSION ====
    print("\n  Controls progression...")
    for sid, formula, ctrl, desc in [
        ("rc/controls/progression/ee_only", f"{xdvar} ~ xee + ym_num | mkt", "ym_num", "EE only"),
        ("rc/controls/progression/ee_ue", f"{xdvar} ~ xee + xue + ym_num | mkt", "xue + ym_num", "EE+UE"),
        ("rc/controls/progression/ee_ue_ur", f"{xdvar} ~ xee + xue + xur + ym_num | mkt", "xue + xur + ym_num", "EE+UE+UR"),
        ("rc/controls/progression/grouped_flows", f"{xdvar} ~ xee + xur + xnue + xenu + ym_num | mkt", "xur + xnue + xenu + ym_num", "grouped"),
    ]:
        try:
            res = run_second_stage(df, formula, 'xee')
            add_result(sid, "modules/robustness/controls.md#d-control-progression-build-up",
                      xdvar, 'xee', res, ctrl, "mkt", base_sample,
                      extra_json_blocks={"controls": {"spec_id": sid, "family": "progression", "set_name": desc, "n_controls": len(ctrl.split('+'))}})
            print(f"    {sid}: coef={res['coef']:.6f}")
        except Exception as e:
            add_failure(sid, "modules/robustness/controls.md#d-control-progression-build-up", xdvar, 'xee', str(e), sample_desc=base_sample)

    # ==== LOO ====
    print("\n  Leave-one-out...")
    for sid, dropped, formula in [
        ("rc/controls/loo/drop_xue", "xue", f"{xdvar} ~ xee + xur + xne + xen + xeu + ym_num | mkt"),
        ("rc/controls/loo/drop_xur", "xur", f"{xdvar} ~ xee + xue + xne + xen + xeu + ym_num | mkt"),
        ("rc/controls/loo/drop_xne", "xne", f"{xdvar} ~ xee + xue + xur + xen + xeu + ym_num | mkt"),
        ("rc/controls/loo/drop_xen", "xen", f"{xdvar} ~ xee + xue + xur + xne + xeu + ym_num | mkt"),
        ("rc/controls/loo/drop_xeu", "xeu", f"{xdvar} ~ xee + xue + xur + xne + xen + ym_num | mkt"),
    ]:
        try:
            res = run_second_stage(df, formula, 'xee')
            remaining = [v for v in ['xue','xur','xne','xen','xeu','ym_num'] if v != dropped]
            add_result(sid, "modules/robustness/controls.md#b-leave-one-out-controls-loo",
                      xdvar, 'xee', res, " + ".join(remaining), "mkt", base_sample,
                      extra_json_blocks={"controls": {"spec_id": sid, "family": "loo", "dropped": [dropped], "n_controls": 5}})
            print(f"    {sid}: coef={res['coef']:.6f}")
        except Exception as e:
            add_failure(sid, "modules/robustness/controls.md#b-leave-one-out-controls-loo", xdvar, 'xee', str(e), sample_desc=base_sample)

    # ==== BIVARIATE (none) ====
    try:
        res = run_second_stage(df, f"{xdvar} ~ xee + ym_num | mkt", 'xee')
        add_result("rc/controls/sets/none", "modules/robustness/controls.md#a-standard-control-sets",
                  xdvar, 'xee', res, "ym_num", "mkt", base_sample,
                  extra_json_blocks={"controls": {"spec_id": "rc/controls/sets/none", "family": "sets", "set_name": "bivariate", "n_controls": 1}})
    except Exception as e:
        add_failure("rc/controls/sets/none", "modules/robustness/controls.md#a-standard-control-sets", xdvar, 'xee', str(e))

    # ==== MINIMAL ====
    try:
        res = run_second_stage(df, f"{xdvar} ~ xee + xue + ym_num | mkt", 'xee')
        add_result("rc/controls/sets/minimal", "modules/robustness/controls.md#a-standard-control-sets",
                  xdvar, 'xee', res, "xue + ym_num", "mkt", base_sample,
                  extra_json_blocks={"controls": {"spec_id": "rc/controls/sets/minimal", "family": "sets", "set_name": "EE+UE", "n_controls": 2}})
    except Exception as e:
        add_failure("rc/controls/sets/minimal", "modules/robustness/controls.md#a-standard-control-sets", xdvar, 'xee', str(e))

    # ==== SAMPLE RESTRICTIONS ====
    print("\n  Sample restrictions...")

    # Job stayers
    try:
        stayer = (df['eetrans_i'] == 0) & (df['lagemp'] > 0)
        res = run_second_stage(df, baseline_formula, 'xee', sample_mask=stayer)
        add_result("rc/sample/subpop/job_stayers", "modules/robustness/sample.md#d-data-quality-and-eligibility-filters",
                  xdvar, 'xee', res, "xue + xur + xne + xen + xeu + ym_num", "mkt",
                  f"{dv_label}, job stayers",
                  extra_json_blocks={"sample": {"spec_id": "rc/sample/subpop/job_stayers", "axis": "subpop", "rule": "filter", "params": {"condition": "eetrans_i==0 & lagemp>0"}}})
        print(f"    job_stayers: coef={res['coef']:.6f}, N={res['n_obs']}")
    except Exception as e:
        add_failure("rc/sample/subpop/job_stayers", "modules/robustness/sample.md", xdvar, 'xee', str(e))

    # Early/late half
    for sid, mask_fn, desc in [
        ("rc/sample/time/early_half", df['ym_num'] <= median_ym, f"early half (ym<={median_ym})"),
        ("rc/sample/time/late_half", df['ym_num'] > median_ym, f"late half (ym>{median_ym})"),
    ]:
        try:
            res = run_second_stage(df, baseline_formula, 'xee', sample_mask=mask_fn)
            add_result(sid, "modules/robustness/sample.md#a-time-period-restrictions",
                      xdvar, 'xee', res, "xue + xur + xne + xen + xeu + ym_num", "mkt",
                      f"{dv_label}, {desc}",
                      extra_json_blocks={"sample": {"spec_id": sid, "axis": "time", "rule": "filter"}})
            print(f"    {sid}: coef={res['coef']:.6f}, N={res['n_obs']}")
        except Exception as e:
            add_failure(sid, "modules/robustness/sample.md#a-time-period-restrictions", xdvar, 'xee', str(e))

    # Outlier trims
    for sid, lo_q, hi_q in [
        ("rc/sample/outliers/trim_y_1_99", 0.01, 0.99),
        ("rc/sample/outliers/trim_y_5_95", 0.05, 0.95),
    ]:
        try:
            lo = df[xdvar].quantile(lo_q)
            hi = df[xdvar].quantile(hi_q)
            trim = (df[xdvar] >= lo) & (df[xdvar] <= hi)
            res = run_second_stage(df, baseline_formula, 'xee', sample_mask=trim)
            add_result(sid, "modules/robustness/sample.md#b-outliers-and-influential-observations",
                      xdvar, 'xee', res, "xue + xur + xne + xen + xeu + ym_num", "mkt",
                      f"{dv_label}, trimmed [{lo_q*100:.0f}%,{hi_q*100:.0f}%]",
                      extra_json_blocks={"sample": {"spec_id": sid, "axis": "outliers", "rule": "trim", "params": {"var": xdvar, "lower_q": lo_q, "upper_q": hi_q}, "n_obs_after": res['n_obs']}})
            print(f"    {sid}: coef={res['coef']:.6f}, N={res['n_obs']}")
        except Exception as e:
            add_failure(sid, "modules/robustness/sample.md#b-outliers-and-influential-observations", xdvar, 'xee', str(e))

    # ==== EE INTERACTION ====
    print("\n  EE interaction...")
    try:
        res = run_second_stage(df, f"{xdvar} ~ xee + xee_i + xue + xur + xne + xen + xeu + ym_num | mkt", 'xee')
        add_result("rc/form/model/xee_interaction", "modules/robustness/functional_form.md#d-interactions-as-functional-form-robustness",
                  xdvar, 'xee', res, "xee_i + xue + xur + xne + xen + xeu + ym_num", "mkt", base_sample,
                  extra_json_blocks={"functional_form": {"spec_id": "rc/form/model/xee_interaction", "outcome_transform": "level", "treatment_transform": "level", "interpretation": "xee coef for job stayers; xee_i is additional effect for job changers"}})
        print(f"    xee_interaction: coef={res['coef']:.6f}")
    except Exception as e:
        add_failure("rc/form/model/xee_interaction", "modules/robustness/functional_form.md", xdvar, 'xee', str(e))

    # ==== UNWEIGHTED ====
    print("\n  Unweighted...")
    try:
        res = run_second_stage(df, baseline_formula, 'xee', weights_col=None)
        add_result("rc/weights/main/unweighted", "modules/robustness/weights.md#a-main-weight-choices",
                  xdvar, 'xee', res, "xue + xur + xne + xen + xeu + ym_num", "mkt",
                  f"{dv_label}, unweighted",
                  extra_json_blocks={"weights": {"spec_id": "rc/weights/main/unweighted", "weight_var": "none", "family": "main"}})
        print(f"    unweighted: coef={res['coef']:.6f}")
    except Exception as e:
        add_failure("rc/weights/main/unweighted", "modules/robustness/weights.md", xdvar, 'xee', str(e))

    # ==== FE VARIATIONS ====
    print("\n  FE variations...")

    # Drop year_month trend
    try:
        res = run_second_stage(df, f"{xdvar} ~ xee + xue + xur + xne + xen + xeu | mkt", 'xee')
        add_result("rc/fe/drop/year_month_trend", "modules/robustness/fixed_effects.md#b-dropping-fe-relative-to-baseline",
                  xdvar, 'xee', res, "xue + xur + xne + xen + xeu", "mkt", base_sample,
                  extra_json_blocks={"fixed_effects": {"spec_id": "rc/fe/drop/year_month_trend", "family": "drop", "dropped": ["year_month_trend"], "baseline_fe": ["mkt"], "new_fe": ["mkt"]}})
        print(f"    drop_ym_trend: coef={res['coef']:.6f}")
    except Exception as e:
        add_failure("rc/fe/drop/year_month_trend", "modules/robustness/fixed_effects.md", xdvar, 'xee', str(e))

    # Add year_month FE
    try:
        df['ym_cat'] = df['ym_num'].astype('Int32')
        res = run_second_stage(df, f"{xdvar} ~ xee + xue + xur + xne + xen + xeu | mkt + ym_cat", 'xee')
        add_result("rc/fe/add/year_month_fe", "modules/robustness/fixed_effects.md#a-additive-fe-variations-relative-to-baseline",
                  xdvar, 'xee', res, "xue + xur + xne + xen + xeu", "mkt + year_month", base_sample,
                  extra_json_blocks={"fixed_effects": {"spec_id": "rc/fe/add/year_month_fe", "family": "add", "added": ["year_month"], "dropped": ["year_month_trend"], "baseline_fe": ["mkt"], "new_fe": ["mkt", "year_month"]}})
        print(f"    add_ym_fe: coef={res['coef']:.6f}")
    except Exception as e:
        add_failure("rc/fe/add/year_month_fe", "modules/robustness/fixed_effects.md", xdvar, 'xee', str(e))

    # ==== ADDITIONAL CONTROL SUBSETS ====
    print("\n  Additional control subsets...")

    for sid, formula, ctrl_desc in [
        ("rc/controls/subset/ee_ur", f"{xdvar} ~ xee + xur + ym_num | mkt", "xur + ym_num"),
        ("rc/controls/subset/inflows", f"{xdvar} ~ xee + xue + xne + ym_num | mkt", "xue + xne + ym_num"),
        ("rc/controls/subset/outflows", f"{xdvar} ~ xee + xeu + xen + ym_num | mkt", "xeu + xen + ym_num"),
        ("rc/controls/subset/no_ne_en", f"{xdvar} ~ xee + xue + xur + xeu + ym_num | mkt", "xue + xur + xeu + ym_num"),
        ("rc/controls/subset/no_ur_eu", f"{xdvar} ~ xee + xue + xne + xen + ym_num | mkt", "xue + xne + xen + ym_num"),
    ]:
        try:
            res = run_second_stage(df, formula, 'xee')
            add_result(sid, "modules/robustness/controls.md#e-high-dimensional-control-set-search",
                      xdvar, 'xee', res, ctrl_desc, "mkt", base_sample,
                      extra_json_blocks={"controls": {"spec_id": sid, "family": "subset", "n_controls": len(ctrl_desc.split('+'))}})
            print(f"    {sid}: coef={res['coef']:.6f}")
        except Exception as e:
            add_failure(sid, "modules/robustness/controls.md", xdvar, 'xee', str(e))

    # ==== JOINT SPECS ====
    print("\n  Joint spec variations...")

    # Stayers + unweighted
    try:
        stayer = (df['eetrans_i'] == 0) & (df['lagemp'] > 0)
        res = run_second_stage(df, baseline_formula, 'xee', weights_col=None, sample_mask=stayer)
        add_result("rc/joint/stayers_unweighted", "modules/robustness/joint.md",
                  xdvar, 'xee', res, "xue + xur + xne + xen + xeu + ym_num", "mkt",
                  f"{dv_label}, stayers, unweighted",
                  extra_json_blocks={"joint": {"axes_changed": ["sample", "weights"], "details": {"sample": "job stayers", "weights": "unweighted"}}})
        print(f"    stayers_unweighted: coef={res['coef']:.6f}")
    except Exception as e:
        add_failure("rc/joint/stayers_unweighted", "modules/robustness/joint.md", xdvar, 'xee', str(e))

    # EE only + early/late
    for sid, period_mask, desc in [
        ("rc/joint/ee_only_early", df['ym_num'] <= median_ym, "early half"),
        ("rc/joint/ee_only_late", df['ym_num'] > median_ym, "late half"),
    ]:
        try:
            res = run_second_stage(df, f"{xdvar} ~ xee + ym_num | mkt", 'xee', sample_mask=period_mask)
            add_result(sid, "modules/robustness/joint.md",
                      xdvar, 'xee', res, "ym_num", "mkt", f"{dv_label}, EE only, {desc}",
                      extra_json_blocks={"joint": {"axes_changed": ["controls", "sample"], "details": {"controls": "EE only", "sample": desc}}})
            print(f"    {sid}: coef={res['coef']:.6f}")
        except Exception as e:
            add_failure(sid, "modules/robustness/joint.md", xdvar, 'xee', str(e))

    # All flows + early/late
    for sid, period_mask, desc in [
        ("rc/joint/all_flows_early", df['ym_num'] <= median_ym, "early half"),
        ("rc/joint/all_flows_late", df['ym_num'] > median_ym, "late half"),
    ]:
        try:
            res = run_second_stage(df, baseline_formula, 'xee', sample_mask=period_mask)
            add_result(sid, "modules/robustness/joint.md",
                      xdvar, 'xee', res, "xue + xur + xne + xen + xeu + ym_num", "mkt",
                      f"{dv_label}, all flows, {desc}",
                      extra_json_blocks={"joint": {"axes_changed": ["sample"], "details": {"controls": "all flows", "sample": desc}}})
            print(f"    {sid}: coef={res['coef']:.6f}")
        except Exception as e:
            add_failure(sid, "modules/robustness/joint.md", xdvar, 'xee', str(e))

    # EE+UE+UR unweighted
    try:
        res = run_second_stage(df, f"{xdvar} ~ xee + xue + xur + ym_num | mkt", 'xee', weights_col=None)
        add_result("rc/joint/ee_ue_ur_unweighted", "modules/robustness/joint.md",
                  xdvar, 'xee', res, "xue + xur + ym_num", "mkt", f"{dv_label}, EE+UE+UR, unweighted",
                  extra_json_blocks={"joint": {"axes_changed": ["controls", "weights"], "details": {"controls": "EE+UE+UR", "weights": "unweighted"}}})
        print(f"    ee_ue_ur_unweighted: coef={res['coef']:.6f}")
    except Exception as e:
        add_failure("rc/joint/ee_ue_ur_unweighted", "modules/robustness/joint.md", xdvar, 'xee', str(e))

    # All flows + trim + ym_fe
    try:
        lo = df[xdvar].quantile(0.01)
        hi = df[xdvar].quantile(0.99)
        trim = (df[xdvar] >= lo) & (df[xdvar] <= hi)
        res = run_second_stage(df, f"{xdvar} ~ xee + xue + xur + xne + xen + xeu | mkt + ym_cat", 'xee', sample_mask=trim)
        add_result("rc/joint/all_flows_trim_ym_fe", "modules/robustness/joint.md",
                  xdvar, 'xee', res, "xue + xur + xne + xen + xeu", "mkt + year_month",
                  f"{dv_label}, trimmed 1/99, ym FE",
                  extra_json_blocks={"joint": {"axes_changed": ["sample", "fixed_effects"], "details": {"sample": "trim 1/99", "fe": "mkt + year_month"}}})
        print(f"    all_flows_trim_ym_fe: coef={res['coef']:.6f}")
    except Exception as e:
        add_failure("rc/joint/all_flows_trim_ym_fe", "modules/robustness/joint.md", xdvar, 'xee', str(e))

    # ==== ALTERNATIVE FOCAL VARIABLES ====
    print("\n  Alternative focal variables...")
    for sid, formula, focal, desc in [
        ("rc/controls/progression/ue_only", f"{xdvar} ~ xue + ym_num | mkt", "xue", "UE only"),
        ("rc/controls/progression/ur_only", f"{xdvar} ~ xur + ym_num | mkt", "xur", "UR only"),
        ("rc/controls/progression/all_flows_focal_xue", baseline_formula, "xue", "all flows, focal=xue"),
        ("rc/controls/progression/all_flows_focal_xur", baseline_formula, "xur", "all flows, focal=xur"),
    ]:
        try:
            res = run_second_stage(df, formula, focal)
            add_result(sid, "modules/robustness/controls.md#d-control-progression-build-up",
                      xdvar, focal, res, desc, "mkt", base_sample,
                      extra_json_blocks={"controls": {"spec_id": sid, "family": "progression", "set_name": desc}})
            print(f"    {sid}: coef({focal})={res['coef']:.6f}")
        except Exception as e:
            add_failure(sid, "modules/robustness/controls.md", xdvar, focal, str(e))

    # Cleanup
    for col in ['xee', 'xue', 'xne', 'xeu', 'xen', 'xur', xdvar, 'xnue', 'xenu', 'xee_i', 'ym_cat']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    gc.collect()
    print(f"  Finished {depvar} at {time.time()-t0:.1f}s")


# ==== Write outputs ====
print(f"\n{'='*60}")
print("Writing outputs...")

results_df = pd.DataFrame(all_results)
results_df.to_csv(f'{PKG_DIR}/specification_results.csv', index=False)
n_success = int(results_df['run_success'].sum())
n_fail = int((results_df['run_success'] == 0).sum())
print(f"specification_results.csv: {len(results_df)} rows ({n_success} success, {n_fail} failed)")

if inference_results:
    infer_df = pd.DataFrame(inference_results)
    infer_df.to_csv(f'{PKG_DIR}/inference_results.csv', index=False)
    print(f"inference_results.csv: {len(infer_df)} rows")

# SPECIFICATION_SEARCH.md
search_md = f"""# Specification Search Log: {PAPER_ID}

## Surface Summary
- **Paper**: Moscarini & Postel-Vinay (2017), AER P&P
- **Baseline groups**: 1 (G1) with 4 outcome variants
- **Design**: cross_sectional_ols (two-stage areg)
- **Canonical inference**: IID SE (paper default)
- **Surface hash**: {SURFACE_HASH}

## Execution Summary
- **Total core specs**: {len(results_df)}
- **Successful**: {n_success}
- **Failed**: {n_fail}
- **Inference variants**: {len(inference_results)}

### By spec_id namespace
- baseline: {len(results_df[results_df['spec_id'].str.startswith('baseline')])}
- rc/controls/*: {len(results_df[results_df['spec_id'].str.startswith('rc/controls/')])}
- rc/sample/*: {len(results_df[results_df['spec_id'].str.startswith('rc/sample/')])}
- rc/form/*: {len(results_df[results_df['spec_id'].str.startswith('rc/form/')])}
- rc/weights/*: {len(results_df[results_df['spec_id'].str.startswith('rc/weights/')])}
- rc/fe/*: {len(results_df[results_df['spec_id'].str.startswith('rc/fe/')])}
- rc/joint/*: {len(results_df[results_df['spec_id'].str.startswith('rc/joint/')])}

## Software
- Python {SOFTWARE['runner_version']}
- pyfixest {SOFTWARE['packages']['pyfixest']}
- pandas {SOFTWARE['packages']['pandas']}
- numpy {SOFTWARE['packages']['numpy']}

## Runtime
- Total: {time.time()-t0:.0f}s
"""
with open(f'{PKG_DIR}/SPECIFICATION_SEARCH.md', 'w') as f:
    f.write(search_md)
print("Wrote SPECIFICATION_SEARCH.md")

print(f"\nTotal time: {time.time()-t0:.1f}s")
print("Done.")
