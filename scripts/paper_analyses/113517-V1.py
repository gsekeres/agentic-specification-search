"""
Specification Search Script for Moscarini & Postel-Vinay (2017)
"The Relative Power of Employment-to-Employment Reallocation and
Unemployment Exits in Predicting Wage Growth"
AER Papers & Proceedings, 107(5), 203-207.

Paper ID: 113517-V1

Surface-driven execution:
  - G1: xdlogern_nom ~ xee (log nominal earnings)
  - G2: xdlogern ~ xee (log real earnings)
  - G3: xdloghwr_nom ~ xee (log nominal hourly wage)
  - G4: xdloghwr ~ xee (log real hourly wage)
  - Cross-sectional OLS with absorbed market FE, classical SE
  - Two-stage procedure: first stage extracts market*time FE, second stage runs OLS

Outputs:
  - specification_results.csv (baseline, design/*, rc/*)
  - inference_results.csv (infer/* variants)
  - SPECIFICATION_SEARCH.md
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import os
import warnings
import time
import gc
import sys

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

# =============================================================================
# Configuration
# =============================================================================
PAPER_ID = "113517-V1"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PACKAGE_DIR = os.path.join(REPO_ROOT, "data", "downloads", "extracted", PAPER_ID)
DATA_DIR = os.path.join(PACKAGE_DIR, "Codes-and-data")
SEED = 113517

t0 = time.time()

# =============================================================================
# Load data
# =============================================================================
print("Loading preprocessed data...")
df = pd.read_parquet(os.path.join(DATA_DIR, 'preprocessed.parquet'))
print(f"Loaded in {time.time()-t0:.1f}s, shape: {df.shape}")

# Prepare data types
for v in ['lagstate', 'laguni', 'lagsiz', 'lagocc', 'lagind', 'lagpub', 'mkt_t', 'mkt']:
    df[v] = df[v].astype('Int32')

df['ym_num'] = df['year_month_num'].astype('float32')

# Create wage changes
for dv in ['logern_nom', 'logern', 'loghwr_nom', 'loghwr']:
    df[f'd{dv}'] = df[dv] - df[f'lag{dv}']

# Create hourly-wage-adjusted eligibility
df['EZeligible_hw'] = ((df['EZeligible'] == 1) & (df['lagphr'] == 1)).astype('int8')
df['DWeligible_hw'] = ((df['DWeligible'] == 1) & (df['lagphr'] == 1)).astype('int8')

# Drop unnecessary columns to save memory
drop_now = [c for c in df.columns if c in [
    'year_month', 'year_month_num', 'loghrs', 'lagphr',
    'clw', 'siz', 'ind', 'occ', 'phr', 'married', 'state', 'uni',
    'sex', 'race', 'education', 'agegroup', 'panel_id',
    'lagmarried', 'laglogern_raw', 'laglogern_raw_nom',
    'laglogern_sm', 'laglogern_sm_nom',
]]
df.drop(columns=drop_now, inplace=True)
gc.collect()
print(f"Shape after cleanup: {df.shape}")
print(f"Memory: {df.memory_usage(deep=True).sum()/1e9:.2f} GB")

# =============================================================================
# Control formulas for first stage
# =============================================================================
e_controls = "C(lagstate) + C(laguni) + C(lagsiz) + C(lagocc) + C(lagind) + C(lagpub)"
u_controls = "C(lagstate)"


# =============================================================================
# First-stage helper
# =============================================================================
def run_first_stage(data, depv, rhs_formula, elig_col):
    """Run first-stage areg and return FE dict."""
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


def map_fe_all(data, fe_dict, colname):
    """Map FE values to ALL observations by mkt_t"""
    fe_map = {int(k): v for k, v in fe_dict.items()}
    data[colname] = data['mkt_t'].map(fe_map)
    return data


# =============================================================================
# Run shared first-stage regressions (invariant across depvars)
# =============================================================================
print("\n--- Shared First-Stage Regressions ---")

print("  UE transition rate...")
fe_ue = run_first_stage(df, 'uetrans_i', u_controls, 'UZeligible')

print("  NE transition rate...")
fe_ne = run_first_stage(df, 'netrans_i', u_controls, 'NZeligible')

print("  Unemployment rate...")
fe_ur = run_first_stage(df, 'unm', u_controls, 'UReligible')

print("  EU transition (earnings sample)...")
fe_eu_earn = run_first_stage(df, 'eutrans_i', e_controls, 'EZeligible')

print("  EN transition (earnings sample)...")
fe_en_earn = run_first_stage(df, 'entrans_i', e_controls, 'EZeligible')

print("  EU transition (hourly wage sample)...")
fe_eu_hw = run_first_stage(df, 'eutrans_i', e_controls, 'EZeligible_hw')

print("  EN transition (hourly wage sample)...")
fe_en_hw = run_first_stage(df, 'entrans_i', e_controls, 'EZeligible_hw')

gc.collect()
print(f"  Shared first-stage done in {time.time()-t0:.1f}s")

# =============================================================================
# Output storage
# =============================================================================
spec_results = []
infer_results = []
run_counter = 0
infer_counter = 0


def run_second_stage(data, formula, focal_var, weights_col, sample_filter=None,
                     spec_id=None, spec_tree_path=None, baseline_group_id=None,
                     outcome_var=None, controls_desc="", sample_desc="",
                     fe_desc="mkt (sex x race x agegroup x education)",
                     extra_json=None):
    """Run a second-stage regression and record the result."""
    global run_counter
    run_counter += 1
    spec_run_id = f"{PAPER_ID}_run{run_counter:04d}"

    try:
        lhs = formula.split('~')[0].strip()
        rhs_part = formula.split('~')[1].split('|')[0].strip()
        fe_part = formula.split('|')[1].strip() if '|' in formula else None
        rhs_vars = [v.strip() for v in rhs_part.split('+')]
        all_needed = [lhs] + rhs_vars + ([fe_part] if fe_part else [])
        if weights_col:
            all_needed.append(weights_col)

        mask = pd.Series(True, index=data.index)
        if weights_col:
            mask = mask & (data[weights_col] > 0)
        for v in all_needed:
            mask = mask & data[v].notna()
        if sample_filter is not None:
            mask = mask & sample_filter(data)
        sub = data.loc[mask]

        if weights_col:
            m = pf.feols(formula, data=sub, weights=weights_col)
        else:
            m = pf.feols(formula, data=sub)

        coef = float(m.coef()[focal_var])
        se_val = float(m.se()[focal_var])
        pval = float(m.pvalue()[focal_var])
        nobs = int(m._N)
        r2 = float(m._r2)
        ci = m.confint()
        ci_lower = float(ci.loc[focal_var, '2.5%']) if focal_var in ci.index else coef - 1.96 * se_val
        ci_upper = float(ci.loc[focal_var, '97.5%']) if focal_var in ci.index else coef + 1.96 * se_val

        coef_dict = {k: float(v) for k, v in m.coef().items()}
        if extra_json:
            coef_dict.update(extra_json)

        del m, sub
        gc.collect()

        row = {
            "paper_id": PAPER_ID,
            "spec_run_id": spec_run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var or lhs,
            "treatment_var": focal_var,
            "coefficient": round(coef, 10),
            "std_error": round(se_val, 10),
            "p_value": round(pval, 10),
            "ci_lower": round(ci_lower, 10),
            "ci_upper": round(ci_upper, 10),
            "n_obs": nobs,
            "r_squared": round(r2, 8),
            "coefficient_vector_json": json.dumps(coef_dict),
            "sample_desc": sample_desc,
            "fixed_effects": fe_desc,
            "controls_desc": controls_desc,
            "cluster_var": "",
            "run_success": 1,
            "run_error": "",
        }
        spec_results.append(row)
        print(f"  {spec_run_id}: {spec_id} | coef={coef:.6f} se={se_val:.6f} p={pval:.4f} N={nobs}")
        return row

    except Exception as e:
        row = {
            "paper_id": PAPER_ID,
            "spec_run_id": spec_run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var or "",
            "treatment_var": focal_var,
            "coefficient": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": 0,
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps({"error": str(e)}),
            "sample_desc": sample_desc,
            "fixed_effects": fe_desc,
            "controls_desc": controls_desc,
            "cluster_var": "",
            "run_success": 0,
            "run_error": str(e),
        }
        spec_results.append(row)
        print(f"  {spec_run_id}: {spec_id} | FAILED: {e}")
        return row


def run_infer_variant(base_row, data, formula, focal_var, weights_col,
                      sample_filter=None, vcov_arg=None, cluster_var_name="",
                      infer_spec_id=None, spec_tree_path=None):
    """Run inference variant recomputation."""
    global infer_counter
    infer_counter += 1
    inference_run_id = f"{PAPER_ID}_infer{infer_counter:04d}"

    try:
        lhs = formula.split('~')[0].strip()
        rhs_part = formula.split('~')[1].split('|')[0].strip()
        fe_part = formula.split('|')[1].strip() if '|' in formula else None
        rhs_vars = [v.strip() for v in rhs_part.split('+')]
        all_needed = [lhs] + rhs_vars + ([fe_part] if fe_part else [])
        if weights_col:
            all_needed.append(weights_col)

        mask = pd.Series(True, index=data.index)
        if weights_col:
            mask = mask & (data[weights_col] > 0)
        for v in all_needed:
            mask = mask & data[v].notna()
        if sample_filter is not None:
            mask = mask & sample_filter(data)
        sub = data.loc[mask]

        if weights_col:
            m = pf.feols(formula, data=sub, weights=weights_col, vcov=vcov_arg)
        else:
            m = pf.feols(formula, data=sub, vcov=vcov_arg)

        coef = float(m.coef()[focal_var])
        se_val = float(m.se()[focal_var])
        pval = float(m.pvalue()[focal_var])
        nobs = int(m._N)
        r2 = float(m._r2)
        ci = m.confint()
        ci_lower = float(ci.loc[focal_var, '2.5%']) if focal_var in ci.index else coef - 1.96 * se_val
        ci_upper = float(ci.loc[focal_var, '97.5%']) if focal_var in ci.index else coef + 1.96 * se_val

        coef_dict = {k: float(v) for k, v in m.coef().items()}
        coef_dict["inference"] = {
            "spec_id": infer_spec_id,
            "method": "cluster" if cluster_var_name else "hc",
            "cluster_var": cluster_var_name,
        }

        del m, sub
        gc.collect()

        row = {
            "paper_id": PAPER_ID,
            "inference_run_id": inference_run_id,
            "spec_run_id": base_row["spec_run_id"],
            "spec_id": infer_spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": base_row["baseline_group_id"],
            "outcome_var": base_row["outcome_var"],
            "treatment_var": focal_var,
            "coefficient": round(coef, 10),
            "std_error": round(se_val, 10),
            "p_value": round(pval, 10),
            "ci_lower": round(ci_lower, 10),
            "ci_upper": round(ci_upper, 10),
            "n_obs": nobs,
            "r_squared": round(r2, 8),
            "coefficient_vector_json": json.dumps(coef_dict),
            "cluster_var": cluster_var_name,
            "run_success": 1,
            "run_error": "",
        }
        infer_results.append(row)
        print(f"  {inference_run_id}: {infer_spec_id} (base={base_row['spec_run_id']}) | se={se_val:.6f} p={pval:.4f}")
        return row

    except Exception as e:
        row = {
            "paper_id": PAPER_ID,
            "inference_run_id": inference_run_id,
            "spec_run_id": base_row["spec_run_id"],
            "spec_id": infer_spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": base_row["baseline_group_id"],
            "outcome_var": base_row["outcome_var"],
            "treatment_var": focal_var,
            "coefficient": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": 0,
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps({"error": str(e)}),
            "cluster_var": cluster_var_name,
            "run_success": 0,
            "run_error": str(e),
        }
        infer_results.append(row)
        print(f"  {inference_run_id}: {infer_spec_id} | FAILED: {e}")
        return row


# =============================================================================
# MAIN LOOP: Process each dependent variable
# =============================================================================

depvar_config = [
    ("logern_nom", "G1", "EZeligible", "DWeligible",
     "Log Nominal Earnings, all employed"),
    ("logern", "G2", "EZeligible", "DWeligible",
     "Log Real Earnings, all employed"),
    ("loghwr_nom", "G3", "EZeligible_hw", "DWeligible_hw",
     "Log Nominal Hourly Wage, hourly workers"),
    ("loghwr", "G4", "EZeligible_hw", "DWeligible_hw",
     "Log Real Hourly Wage, hourly workers"),
]

for depvar, group_id, ez_col, dw_col, sample_label in depvar_config:
    lagdepvar = f'lag{depvar}'
    dvar = f'd{depvar}'
    xdvar = f'xd{depvar}'

    print(f"\n{'='*70}")
    print(f"BASELINE GROUP {group_id}: {depvar}")
    print(f"{'='*70}")

    # --- First stage: EE and DW (depvar-specific) ---
    print("  First stage: EE transition...")
    fe_ee = run_first_stage(df, 'eetrans_i', f'{lagdepvar} + {e_controls}', ez_col)

    print(f"  First stage: Wage growth ({dvar})...")
    fe_dw = run_first_stage(df, dvar, f'eetrans_i + {e_controls}', dw_col)

    # Select EU/EN FE based on depvar type
    fe_eu = fe_eu_hw if depvar in ['loghwr', 'loghwr_nom'] else fe_eu_earn
    fe_en = fe_en_hw if depvar in ['loghwr', 'loghwr_nom'] else fe_en_earn

    # Map FE to all observations
    map_fe_all(df, fe_ee, 'xee')
    map_fe_all(df, fe_ue, 'xue')
    map_fe_all(df, fe_ne, 'xne')
    map_fe_all(df, fe_eu, 'xeu')
    map_fe_all(df, fe_en, 'xen')
    map_fe_all(df, fe_ur, 'xur')
    map_fe_all(df, fe_dw, xdvar)

    # Restrict dependent variable to eligible obs
    df.loc[df[dw_col] != 1, xdvar] = np.nan

    # Composite variables
    df['xnue'] = df['xue'] + df['xne']
    df['xenu'] = df['xen'] + df['xeu']
    df['xee_i'] = df['xee'] * df['eetrans_i']

    # Define the baseline formula
    baseline_formula = f"{xdvar} ~ xee + xue + xur + xne + xen + xeu + ym_num | mkt"

    # =========================================================================
    # BASELINE: All flows (spec 6)
    # =========================================================================
    print(f"\n--- Baseline (All flows) ---")
    baseline_row = run_second_stage(
        df, baseline_formula, "xee", "wgt",
        spec_id="baseline",
        spec_tree_path="designs/cross_sectional_ols.md#baseline",
        baseline_group_id=group_id,
        outcome_var=xdvar,
        controls_desc="xue + xur + xne + xen + xeu + ym_num",
        sample_desc=sample_label,
    )

    # =========================================================================
    # CONTROL PROGRESSIONS
    # =========================================================================
    print(f"\n--- Control Progressions ---")

    # EE only (spec 1)
    run_second_stage(
        df, f"{xdvar} ~ xee + ym_num | mkt", "xee", "wgt",
        spec_id="rc/controls/progression/ee_only",
        spec_tree_path="modules/robustness/controls.md#control-progression-build-up",
        baseline_group_id=group_id,
        outcome_var=xdvar,
        controls_desc="ym_num",
        sample_desc=sample_label,
        extra_json={"controls": {"spec_id": "rc/controls/progression/ee_only",
                                 "family": "progression", "n_controls": 1}},
    )

    # EE + UE (spec 4)
    run_second_stage(
        df, f"{xdvar} ~ xee + xue + ym_num | mkt", "xee", "wgt",
        spec_id="rc/controls/progression/ee_ue",
        spec_tree_path="modules/robustness/controls.md#control-progression-build-up",
        baseline_group_id=group_id,
        outcome_var=xdvar,
        controls_desc="xue + ym_num",
        sample_desc=sample_label,
        extra_json={"controls": {"spec_id": "rc/controls/progression/ee_ue",
                                 "family": "progression", "n_controls": 2}},
    )

    # EE + UE + UR (spec 5)
    run_second_stage(
        df, f"{xdvar} ~ xee + xue + xur + ym_num | mkt", "xee", "wgt",
        spec_id="rc/controls/progression/ee_ue_ur",
        spec_tree_path="modules/robustness/controls.md#control-progression-build-up",
        baseline_group_id=group_id,
        outcome_var=xdvar,
        controls_desc="xue + xur + ym_num",
        sample_desc=sample_label,
        extra_json={"controls": {"spec_id": "rc/controls/progression/ee_ue_ur",
                                 "family": "progression", "n_controls": 3}},
    )

    # Grouped flows (spec 7)
    run_second_stage(
        df, f"{xdvar} ~ xee + xur + xnue + xenu + ym_num | mkt", "xee", "wgt",
        spec_id="rc/controls/progression/grouped_flows",
        spec_tree_path="modules/robustness/controls.md#control-progression-build-up",
        baseline_group_id=group_id,
        outcome_var=xdvar,
        controls_desc="xur + xnue + xenu + ym_num",
        sample_desc=sample_label,
        extra_json={"controls": {"spec_id": "rc/controls/progression/grouped_flows",
                                 "family": "progression", "n_controls": 4}},
    )

    # =========================================================================
    # LEAVE-ONE-OUT FROM BASELINE
    # =========================================================================
    print(f"\n--- Leave-One-Out from baseline ---")

    loo_specs = [
        ("xue", "rc/controls/loo/drop_xue",
         f"{xdvar} ~ xee + xur + xne + xen + xeu + ym_num | mkt",
         "xur + xne + xen + xeu + ym_num"),
        ("xur", "rc/controls/loo/drop_xur",
         f"{xdvar} ~ xee + xue + xne + xen + xeu + ym_num | mkt",
         "xue + xne + xen + xeu + ym_num"),
        ("xne", "rc/controls/loo/drop_xne",
         f"{xdvar} ~ xee + xue + xur + xen + xeu + ym_num | mkt",
         "xue + xur + xen + xeu + ym_num"),
        ("xen", "rc/controls/loo/drop_xen",
         f"{xdvar} ~ xee + xue + xur + xne + xeu + ym_num | mkt",
         "xue + xur + xne + xeu + ym_num"),
        ("xeu", "rc/controls/loo/drop_xeu",
         f"{xdvar} ~ xee + xue + xur + xne + xen + ym_num | mkt",
         "xue + xur + xne + xen + ym_num"),
        ("ym_num", "rc/controls/loo/drop_ym_num",
         f"{xdvar} ~ xee + xue + xur + xne + xen + xeu | mkt",
         "xue + xur + xne + xen + xeu"),
    ]

    for dropped_var, loo_spec_id, loo_formula, loo_controls in loo_specs:
        run_second_stage(
            df, loo_formula, "xee", "wgt",
            spec_id=loo_spec_id,
            spec_tree_path="modules/robustness/controls.md#leave-one-out-controls-loo",
            baseline_group_id=group_id,
            outcome_var=xdvar,
            controls_desc=loo_controls,
            sample_desc=sample_label,
            extra_json={"controls": {"spec_id": loo_spec_id, "family": "loo",
                                     "dropped": [dropped_var], "n_controls": 5}},
        )

    # =========================================================================
    # SAMPLE: JOB STAYERS
    # =========================================================================
    print(f"\n--- Job stayers subsample ---")
    run_second_stage(
        df, baseline_formula, "xee", "wgt",
        sample_filter=lambda d: (d['eetrans_i'] == 0) & (d['lagemp'] > 0),
        spec_id="rc/sample/subpop/job_stayers",
        spec_tree_path="modules/robustness/sample.md#data-quality-and-eligibility-filters",
        baseline_group_id=group_id,
        outcome_var=xdvar,
        controls_desc="xue + xur + xne + xen + xeu + ym_num",
        sample_desc=f"{sample_label}, job stayers (eetrans_i==0 & lagemp>0)",
        extra_json={"sample": {"spec_id": "rc/sample/subpop/job_stayers",
                               "axis": "subpop", "rule": "eetrans_i==0 & lagemp>0"}},
    )

    # =========================================================================
    # FUNCTIONAL FORM: EE interaction
    # =========================================================================
    print(f"\n--- EE interaction ---")
    run_second_stage(
        df, f"{xdvar} ~ xee + xee_i + xue + xur + xne + xen + xeu + ym_num | mkt",
        "xee", "wgt",
        spec_id="rc/form/model/ee_interaction",
        spec_tree_path="modules/robustness/functional_form.md#interactions-as-functional-form-robustness",
        baseline_group_id=group_id,
        outcome_var=xdvar,
        controls_desc="xee_i + xue + xur + xne + xen + xeu + ym_num",
        sample_desc=sample_label,
        extra_json={"functional_form": {"spec_id": "rc/form/model/ee_interaction",
                                        "model_terms": ["xee", "xee_i", "xue", "xur",
                                                        "xne", "xen", "xeu", "ym_num"]}},
    )

    # =========================================================================
    # WEIGHTS: UNWEIGHTED
    # =========================================================================
    print(f"\n--- Unweighted ---")
    run_second_stage(
        df, baseline_formula, "xee", None,
        spec_id="rc/weights/main/unweighted",
        spec_tree_path="modules/robustness/weights.md#main-weight-choices",
        baseline_group_id=group_id,
        outcome_var=xdvar,
        controls_desc="xue + xur + xne + xen + xeu + ym_num",
        sample_desc=f"{sample_label}, unweighted",
        extra_json={"weights": {"spec_id": "rc/weights/main/unweighted",
                                "weight_var": "none", "family": "main"}},
    )

    # =========================================================================
    # FE: DROP MARKET FE (pooled OLS)
    # =========================================================================
    print(f"\n--- Drop market FE (pooled OLS) ---")
    run_second_stage(
        df, f"{xdvar} ~ xee + xue + xur + xne + xen + xeu + ym_num",
        "xee", "wgt",
        spec_id="rc/fe/drop/mkt",
        spec_tree_path="modules/robustness/fixed_effects.md#dropping-fe-relative-to-baseline",
        baseline_group_id=group_id,
        outcome_var=xdvar,
        controls_desc="xue + xur + xne + xen + xeu + ym_num",
        sample_desc=sample_label,
        fe_desc="none",
        extra_json={"fixed_effects": {"spec_id": "rc/fe/drop/mkt", "family": "drop",
                                      "dropped": ["mkt"], "baseline_fe": ["mkt"],
                                      "new_fe": []}},
    )

    # =========================================================================
    # ADDITIONAL RC SPECS (for 50+ target)
    # =========================================================================

    # Controls: UE only and UR only (alternative single-flow focal specs)
    print(f"\n--- Additional single-flow specs ---")
    run_second_stage(
        df, f"{xdvar} ~ xee + xue | mkt", "xee", "wgt",
        spec_id="rc/controls/sets/minimal",
        spec_tree_path="modules/robustness/controls.md#standard-control-sets",
        baseline_group_id=group_id,
        outcome_var=xdvar,
        controls_desc="xue (no time trend)",
        sample_desc=sample_label,
        extra_json={"controls": {"spec_id": "rc/controls/sets/minimal",
                                 "family": "sets", "n_controls": 1}},
    )

    # EE + UR only (no UE)
    run_second_stage(
        df, f"{xdvar} ~ xee + xur + ym_num | mkt", "xee", "wgt",
        spec_id="rc/controls/progression/ee_ur",
        spec_tree_path="modules/robustness/controls.md#control-progression-build-up",
        baseline_group_id=group_id,
        outcome_var=xdvar,
        controls_desc="xur + ym_num",
        sample_desc=sample_label,
        extra_json={"controls": {"spec_id": "rc/controls/progression/ee_ur",
                                 "family": "progression", "n_controls": 2}},
    )

    # =========================================================================
    # INFERENCE VARIANTS (for baseline row)
    # =========================================================================
    print(f"\n--- Inference variants ---")

    # HC1 (robust)
    run_infer_variant(
        baseline_row, df, baseline_formula, "xee", "wgt",
        vcov_arg="hetero",
        infer_spec_id="infer/se/hc/hc1",
        spec_tree_path="modules/inference/standard_errors.md#heteroskedasticity-robust-se",
    )

    # Cluster by market
    run_infer_variant(
        baseline_row, df, baseline_formula, "xee", "wgt",
        vcov_arg={"CRV1": "mkt"},
        cluster_var_name="mkt",
        infer_spec_id="infer/se/cluster/mkt",
        spec_tree_path="modules/inference/standard_errors.md#single-level-clustering",
    )

    # ----- Cleanup -----
    for col in ['xee', 'xue', 'xne', 'xeu', 'xen', 'xur', xdvar, 'xnue', 'xenu', 'xee_i']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    gc.collect()
    print(f"  Finished {group_id} ({depvar}) at {time.time()-t0:.1f}s")


# =============================================================================
# Write outputs
# =============================================================================
print(f"\n{'='*70}")
print("WRITING OUTPUTS")
print(f"{'='*70}")

# specification_results.csv
spec_df = pd.DataFrame(spec_results)
spec_path = os.path.join(PACKAGE_DIR, "specification_results.csv")
spec_df.to_csv(spec_path, index=False)
print(f"Wrote {len(spec_df)} rows to specification_results.csv")

# inference_results.csv
infer_df = pd.DataFrame(infer_results)
infer_path = os.path.join(PACKAGE_DIR, "inference_results.csv")
infer_df.to_csv(infer_path, index=False)
print(f"Wrote {len(infer_df)} rows to inference_results.csv")

# Count spec types
n_baseline = len(spec_df[spec_df['spec_id'] == 'baseline'])
n_progression = len(spec_df[spec_df['spec_id'].str.startswith('rc/controls/progression/')])
n_loo = len(spec_df[spec_df['spec_id'].str.startswith('rc/controls/loo/')])
n_sets = len(spec_df[spec_df['spec_id'].str.startswith('rc/controls/sets/')])
n_sample = len(spec_df[spec_df['spec_id'].str.startswith('rc/sample/')])
n_form = len(spec_df[spec_df['spec_id'].str.startswith('rc/form/')])
n_weights = len(spec_df[spec_df['spec_id'].str.startswith('rc/weights/')])
n_fe = len(spec_df[spec_df['spec_id'].str.startswith('rc/fe/')])
n_failed = len(spec_df[spec_df['run_success'] == 0])

# SPECIFICATION_SEARCH.md
search_md = f"""# Specification Search Log: {PAPER_ID}

## Surface Summary
- **Paper**: Moscarini & Postel-Vinay (2017), "The Relative Power of Employment-to-Employment Reallocation and Unemployment Exits in Predicting Wage Growth", AER P&P
- **Baseline groups**: 4 (G1: log nom earnings, G2: log real earnings, G3: log nom hourly wage, G4: log real hourly wage)
- **Design**: cross_sectional_ols (two-stage procedure with absorbed market FE)
- **Canonical inference**: Classical (IID) SE (paper default -- areg without robust/cluster)
- **Seed**: {SEED}

## Execution Summary

### Planned vs Executed
- **Total core specs**: {len(spec_df)} (4 groups x {len(spec_df)//4} per group)
- **Inference variants**: {len(infer_df)} (2 per group: HC1, cluster by mkt)
- **Failed specs**: {n_failed}

### Specs by Type
- baseline: {n_baseline}
- rc/controls/progression/*: {n_progression}
- rc/controls/loo/*: {n_loo}
- rc/controls/sets/*: {n_sets}
- rc/sample/*: {n_sample}
- rc/form/*: {n_form}
- rc/weights/*: {n_weights}
- rc/fe/*: {n_fe}

### Deviations
- Added rc/controls/sets/minimal and rc/controls/progression/ee_ur beyond the original surface to reach 50+ specs.

## Software
- Python 3.x
- pyfixest (feols with absorbed FE)
- pandas, numpy

## Notes
- Two-stage procedure: first stage extracts market*time FE from individual-level regressions, second stage regresses predicted values on each other with market FE absorbed.
- First-stage controls are invariant across all specifications.
- Data has ~6M observations; memory-efficient approach used.
- Total runtime: ~{time.time()-t0:.0f}s
"""

search_path = os.path.join(PACKAGE_DIR, "SPECIFICATION_SEARCH.md")
with open(search_path, 'w') as f:
    f.write(search_md)
print(f"Wrote SPECIFICATION_SEARCH.md")

print(f"\nTotal time: {time.time()-t0:.1f}s")
print("Done.")
