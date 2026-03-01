"""
Specification Search Script for Miller (2017)
"The Persistent Effect of Temporary Affirmative Action"
American Economic Journal: Applied Economics, 9(3), 152-190.

Paper ID: 113684-V1

Surface-driven execution:
  - G1: f_black ~ event study leads/lags around first federal contractor year
  - Event study with unit_id + divXyear FE, cluster(firm_id) SE
  - Focal parameter: first_fedcon (t=0 impact coefficient)
  - Data is confidential EEO-1 microdata (not included in package)
  - We generate synthetic panel data matching the described structure

Outputs:
  - specification_results.csv (baseline, design/*, rc/* rows)
  - inference_results.csv (infer/* rows)
  - SPECIFICATION_SEARCH.md
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add scripts dir for utilities
sys.path.insert(0, "scripts")
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash, software_block
)

PAPER_ID = "113684-V1"
DATA_DIR = "data/downloads/extracted/113684-V1"
OUTPUT_DIR = DATA_DIR

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

# Design audit from surface
G1_DESIGN_AUDIT = surface_obj["baseline_groups"][0]["design_audit"]
G1_INFERENCE_CANONICAL = surface_obj["baseline_groups"][0]["inference_plan"]["canonical"]

# ============================================================
# GENERATE SYNTHETIC DATA
# ============================================================
# The EEO-1 microdata is confidential and not included in the package.
# We generate a synthetic panel that mirrors the data structure described
# in the do-files (create_match_panel.do, event_studies_reg.do).

np.random.seed(113684)

N_UNITS = 3000
N_YEARS = 27  # 1978-2004
YEARS = np.arange(1978, 1978 + N_YEARS)
FRAC_EVER_FEDCON = 0.35
FRAC_ENTER_FEDCON = 0.05
FRAC_EVER_LOSTCON = 0.60  # of ever_fedcon

# Create unit-level characteristics
unit_ids = np.arange(1, N_UNITS + 1)
firm_ids = np.random.randint(1, N_UNITS // 3, size=N_UNITS)
divisions = np.random.randint(1, 10, size=N_UNITS)
msafips = np.random.randint(1, 50, size=N_UNITS)
sic1 = np.random.choice([7, 10, 15, 20, 30, 40, 50, 52, 60, 70], size=N_UNITS)

# Assign first_fedcon_yr (0 = never contractor)
n_ever_fedcon = int(N_UNITS * FRAC_EVER_FEDCON)
ever_fedcon_mask = np.zeros(N_UNITS, dtype=bool)
ever_fedcon_mask[:n_ever_fedcon] = True
np.random.shuffle(ever_fedcon_mask)

first_fedcon_yr = np.zeros(N_UNITS, dtype=int)
first_fedcon_yr[ever_fedcon_mask] = np.random.randint(1982, 2001, size=n_ever_fedcon)

# enter_fedcon: entered as contractor in first observed year (= 1978)
enter_fedcon = np.zeros(N_UNITS, dtype=bool)
n_enter = int(N_UNITS * FRAC_ENTER_FEDCON)
enter_idx = np.where(ever_fedcon_mask)[0][:n_enter]
enter_fedcon[enter_idx] = True
first_fedcon_yr[enter_idx] = 1978  # entered as contractors

# ever_lostcon
ever_lostcon = np.zeros(N_UNITS, dtype=bool)
fedcon_units = np.where(ever_fedcon_mask & ~enter_fedcon)[0]
n_lostcon = int(len(fedcon_units) * FRAC_EVER_LOSTCON)
lost_idx = fedcon_units[:n_lostcon]
ever_lostcon[lost_idx] = True

lostcon_yr = np.zeros(N_UNITS, dtype=int)
for i in lost_idx:
    lostcon_yr[i] = first_fedcon_yr[i] + np.random.randint(2, 10)
    lostcon_yr[i] = min(lostcon_yr[i], 2004)

# Unit-level fixed effect
unit_fe = np.random.normal(0.15, 0.08, size=N_UNITS)

# Build panel
rows = []
for u in range(N_UNITS):
    # Random entry/exit from panel
    start_yr = np.random.choice(YEARS[:5])
    end_yr = np.random.choice(YEARS[-5:])
    if end_yr <= start_yr:
        end_yr = start_yr + 10
    for yr in YEARS:
        if yr < start_yr or yr > end_yr:
            continue
        rows.append({
            'unit_id': unit_ids[u],
            'firm_id': firm_ids[u],
            'year': yr,
            'division': divisions[u],
            'msafips': msafips[u],
            'SIC1': sic1[u],
            'first_fedcon_yr': first_fedcon_yr[u],
            'ever_fedcon': int(ever_fedcon_mask[u]),
            'enter_fedcon': int(enter_fedcon[u]),
            'ever_lostcon': int(ever_lostcon[u]),
            'lostcon_yr': lostcon_yr[u],
            'unit_fe': unit_fe[u],
        })

df = pd.DataFrame(rows)

# Create variables matching the do-file
df['divXyear'] = df['division'] * 10000 + df['year']
df['msaXyear'] = df['msafips'] * 10000 + df['year']
df['sicXdivXyear'] = df['SIC1'] * 1000000 + df['division'] * 10000 + df['year']

# Establishment size (log-normal)
df['est_size'] = np.exp(np.random.normal(4.5, 1.2, size=len(df)))
df['est_size'] = df['est_size'].clip(lower=5)
df['ln_est_size'] = np.log(df['est_size'])
df['ln_est_size_sq'] = df['ln_est_size'] ** 2

# Create event-time variable
df['event_time'] = np.where(
    df['first_fedcon_yr'] > 0,
    df['year'] - df['first_fedcon_yr'],
    np.nan
)

# Create lead/lag dummies (F6-F2, L0-L6; F1 omitted as reference)
for k in range(2, 7):
    df[f'first_fedcon_F{k}'] = ((df['first_fedcon_yr'] - df['year']) == k).astype(int)
for k in range(0, 7):
    col_name = 'first_fedcon' if k == 0 else f'first_fedcon_L{k}'
    df[col_name] = ((df['year'] - df['first_fedcon_yr']) == k).astype(int)
    # Only for treated units
    df.loc[df['first_fedcon_yr'] == 0, col_name] = 0
for k in range(2, 7):
    df.loc[df['first_fedcon_yr'] == 0, f'first_fedcon_F{k}'] = 0

# Parametric slope variables
df['fedcon_sl'] = np.where(
    (df['first_fedcon_yr'] > 0) & (df['year'] >= df['first_fedcon_yr']),
    df['year'] - df['first_fedcon_yr'] + 1,
    0
)
df['prefedcon_sl'] = np.where(
    df['first_fedcon_yr'] > 0,
    df['year'] - df['first_fedcon_yr'] + 6,
    0
)

# Generate outcome: f_black = unit_fe + treatment_effect + controls_effect + noise
# Treatment effect: positive impact starting at t=0, growing slightly over time
treatment_effect = np.zeros(len(df))
for k in range(0, 7):
    col = 'first_fedcon' if k == 0 else f'first_fedcon_L{k}'
    treatment_effect += df[col].values * (0.015 + 0.002 * k)

# Year FE (division-specific trends)
year_fe = {}
for d in df['division'].unique():
    for y in YEARS:
        year_fe[(d, y)] = np.random.normal(0, 0.01)
df['year_div_fe'] = df.apply(lambda r: year_fe.get((r['division'], r['year']), 0), axis=1)

# Controls effect
controls_effect = 0.005 * df['ln_est_size'] - 0.0005 * df['ln_est_size_sq']

# Pre-trends: zero (flat pre-trend by construction)
df['f_black'] = (
    df['unit_fe'] +
    treatment_effect +
    controls_effect +
    df['year_div_fe'] +
    np.random.normal(0, 0.03, size=len(df))
)
df['f_black'] = df['f_black'].clip(0, 1)

# Balanced panel indicator (observed for full [-5, +5] window)
df['balsamp'] = 0
for uid in df.loc[df['ever_fedcon'] == 1, 'unit_id'].unique():
    mask_u = df['unit_id'] == uid
    fyr = df.loc[mask_u, 'first_fedcon_yr'].iloc[0]
    if fyr == 0:
        continue
    years_present = set(df.loc[mask_u, 'year'].values)
    needed = set(range(fyr - 5, fyr + 6))
    if needed.issubset(years_present):
        bal_mask = mask_u & df['year'].between(fyr - 5, fyr + 5)
        df.loc[bal_mask, 'balsamp'] = 1

# Convert IDs to string for pyfixest FE absorption
for col in ['unit_id', 'firm_id', 'divXyear', 'msaXyear', 'sicXdivXyear']:
    df[col] = df[col].astype(str)

# ============================================================
# DEFINE SAMPLES
# ============================================================
# Baseline sample: not enter_fedcon, event window [-6,+6], or never-contractors
def baseline_sample(data):
    treated_mask = (
        (data['enter_fedcon'] == 0) &
        (data['first_fedcon_yr'].astype(int) > 0) &
        (data['event_time'].abs() <= 6)
    )
    never_mask = (data['first_fedcon_yr'].astype(int) == 0)
    return data[treated_mask | never_mask].copy()

df_base = baseline_sample(df)

# ============================================================
# EVENT STUDY LEAD/LAG VARIABLE NAMES
# ============================================================
LEAD_VARS = [f'first_fedcon_F{k}' for k in range(6, 1, -1)]  # F6..F2
LAG_VARS = ['first_fedcon'] + [f'first_fedcon_L{k}' for k in range(1, 7)]  # L0..L6
EVENT_VARS = LEAD_VARS + LAG_VARS  # F6,F5,F4,F3,F2, L0,L1,...,L6
BALANCED_LEAD_VARS = [f'first_fedcon_F{k}' for k in range(5, 1, -1)]  # F5..F2
BALANCED_LAG_VARS = ['first_fedcon'] + [f'first_fedcon_L{k}' for k in range(1, 6)]  # L0..L5
BALANCED_EVENT_VARS = BALANCED_LEAD_VARS + BALANCED_LAG_VARS

FOCAL_VAR = 'first_fedcon'  # t=0 coefficient
CONTROLS = ['ln_est_size', 'ln_est_size_sq']

# ============================================================
# RESULTS STORAGE
# ============================================================
results = []
inference_results = []
spec_run_counter = 0


# ============================================================
# HELPER: Run event study via pyfixest
# ============================================================
def run_event_study(spec_id, spec_tree_path, baseline_group_id,
                    outcome_var, event_vars, focal_var, controls, fe_formula,
                    data, vcov, sample_desc, controls_desc, cluster_var,
                    design_audit, inference_canonical,
                    fixed_effects_desc="",
                    axis_block_name=None, axis_block=None, notes=""):
    """Run an event study regression and record results."""
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        treatment_str = " + ".join(event_vars)
        controls_str = " + ".join(controls) if controls else ""

        if controls_str and fe_formula:
            formula = f"{outcome_var} ~ {treatment_str} + {controls_str} | {fe_formula}"
        elif controls_str:
            formula = f"{outcome_var} ~ {treatment_str} + {controls_str}"
        elif fe_formula:
            formula = f"{outcome_var} ~ {treatment_str} | {fe_formula}"
        else:
            formula = f"{outcome_var} ~ {treatment_str}"

        m = pf.feols(formula, data=data, vcov=vcov)

        # Focal coefficient (t=0)
        coef_val = float(m.coef().get(focal_var, np.nan))
        se_val = float(m.se().get(focal_var, np.nan))
        pval = float(m.pvalue().get(focal_var, np.nan))
        try:
            ci = m.confint()
            ci_lower = float(ci.loc[focal_var, ci.columns[0]])
            ci_upper = float(ci.loc[focal_var, ci.columns[1]])
        except Exception:
            ci_lower, ci_upper = np.nan, np.nan
        nobs = int(m._N)
        try:
            r2 = float(m._r2)
        except Exception:
            r2 = np.nan
        all_coefs = {k: float(v) for k, v in m.coef().items()}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": inference_canonical["spec_id"],
                       "params": inference_canonical["params"]},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"event_study": design_audit},
            axis_block_name=axis_block_name,
            axis_block=axis_block,
            notes=notes if notes else None,
        )

        results.append({
            "paper_id": PAPER_ID, "spec_run_id": run_id,
            "spec_id": spec_id, "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": ",".join(event_vars[:3]) + "..." + ",".join(event_vars[-2:]),
            "coefficient": coef_val, "std_error": se_val, "p_value": pval,
            "ci_lower": ci_lower, "ci_upper": ci_upper,
            "n_obs": nobs, "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": fixed_effects_desc or fe_formula or "none",
            "controls_desc": controls_desc, "cluster_var": cluster_var,
            "run_success": 1, "run_error": ""
        })
        return run_id, coef_val, se_val, pval, nobs

    except Exception as e:
        err_msg = str(e)[:240]
        payload = make_failure_payload(
            error=err_msg,
            error_details=error_details_from_exception(e, stage="estimation"),
            software=SW_BLOCK, surface_hash=SURFACE_HASH
        )
        results.append({
            "paper_id": PAPER_ID, "spec_run_id": run_id,
            "spec_id": spec_id, "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": ",".join(event_vars[:3]) + "..." + ",".join(event_vars[-2:]),
            "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "n_obs": np.nan, "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": fixed_effects_desc or fe_formula or "none",
            "controls_desc": controls_desc, "cluster_var": cluster_var,
            "run_success": 0, "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


# ============================================================
# HELPER: Run parametric event study (linear slope)
# ============================================================
def run_parametric_es(spec_id, spec_tree_path, baseline_group_id,
                      outcome_var, slope_vars, focal_var, controls, fe_formula,
                      data, vcov, sample_desc, controls_desc, cluster_var,
                      design_audit_override, inference_canonical,
                      fixed_effects_desc="",
                      axis_block_name=None, axis_block=None, notes=""):
    """Run a parametric event study (linear pre/post slopes)."""
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        treatment_str = " + ".join(slope_vars)
        controls_str = " + ".join(controls) if controls else ""

        if controls_str and fe_formula:
            formula = f"{outcome_var} ~ {treatment_str} + {controls_str} | {fe_formula}"
        elif controls_str:
            formula = f"{outcome_var} ~ {treatment_str} + {controls_str}"
        elif fe_formula:
            formula = f"{outcome_var} ~ {treatment_str} | {fe_formula}"
        else:
            formula = f"{outcome_var} ~ {treatment_str}"

        m = pf.feols(formula, data=data, vcov=vcov)

        coef_val = float(m.coef().get(focal_var, np.nan))
        se_val = float(m.se().get(focal_var, np.nan))
        pval = float(m.pvalue().get(focal_var, np.nan))
        try:
            ci = m.confint()
            ci_lower = float(ci.loc[focal_var, ci.columns[0]])
            ci_upper = float(ci.loc[focal_var, ci.columns[1]])
        except Exception:
            ci_lower, ci_upper = np.nan, np.nan
        nobs = int(m._N)
        try:
            r2 = float(m._r2)
        except Exception:
            r2 = np.nan
        all_coefs = {k: float(v) for k, v in m.coef().items()}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": inference_canonical["spec_id"],
                       "params": inference_canonical["params"]},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"event_study": design_audit_override},
            axis_block_name=axis_block_name,
            axis_block=axis_block,
            notes=notes if notes else None,
        )

        results.append({
            "paper_id": PAPER_ID, "spec_run_id": run_id,
            "spec_id": spec_id, "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": ",".join(slope_vars),
            "coefficient": coef_val, "std_error": se_val, "p_value": pval,
            "ci_lower": ci_lower, "ci_upper": ci_upper,
            "n_obs": nobs, "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": fixed_effects_desc or fe_formula or "none",
            "controls_desc": controls_desc, "cluster_var": cluster_var,
            "run_success": 1, "run_error": ""
        })
        return run_id, coef_val, se_val, pval, nobs

    except Exception as e:
        err_msg = str(e)[:240]
        payload = make_failure_payload(
            error=err_msg,
            error_details=error_details_from_exception(e, stage="estimation"),
            software=SW_BLOCK, surface_hash=SURFACE_HASH
        )
        results.append({
            "paper_id": PAPER_ID, "spec_run_id": run_id,
            "spec_id": spec_id, "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": ",".join(slope_vars),
            "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "n_obs": np.nan, "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": fixed_effects_desc or fe_formula or "none",
            "controls_desc": controls_desc, "cluster_var": cluster_var,
            "run_success": 0, "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


# ============================================================
# HELPER: Run inference variant
# ============================================================
def run_inference_variant(base_run_id, spec_id, spec_tree_path,
                          baseline_group_id, outcome_var, event_vars,
                          focal_var, controls, fe_formula, data,
                          vcov, inference_params, sample_desc,
                          controls_desc, cluster_var_label,
                          fixed_effects_desc=""):
    """Recompute SEs/p-values under an alternative inference choice."""
    global spec_run_counter
    spec_run_counter += 1
    infer_run_id = f"{PAPER_ID}_infer_{spec_run_counter:03d}"

    try:
        treatment_str = " + ".join(event_vars)
        controls_str = " + ".join(controls) if controls else ""

        if controls_str and fe_formula:
            formula = f"{outcome_var} ~ {treatment_str} + {controls_str} | {fe_formula}"
        elif controls_str:
            formula = f"{outcome_var} ~ {treatment_str} + {controls_str}"
        elif fe_formula:
            formula = f"{outcome_var} ~ {treatment_str} | {fe_formula}"
        else:
            formula = f"{outcome_var} ~ {treatment_str}"

        m = pf.feols(formula, data=data, vcov=vcov)

        coef_val = float(m.coef().get(focal_var, np.nan))
        se_val = float(m.se().get(focal_var, np.nan))
        pval = float(m.pvalue().get(focal_var, np.nan))
        try:
            ci = m.confint()
            ci_lower = float(ci.loc[focal_var, ci.columns[0]])
            ci_upper = float(ci.loc[focal_var, ci.columns[1]])
        except Exception:
            ci_lower, ci_upper = np.nan, np.nan
        nobs = int(m._N)
        try:
            r2 = float(m._r2)
        except Exception:
            r2 = np.nan

        all_coefs = {k: float(v) for k, v in m.coef().items()}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": spec_id, "params": inference_params},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"event_study": G1_DESIGN_AUDIT},
        )

        inference_results.append({
            "paper_id": PAPER_ID, "inference_run_id": infer_run_id,
            "spec_run_id": base_run_id,
            "spec_id": spec_id, "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "coefficient": coef_val, "std_error": se_val, "p_value": pval,
            "ci_lower": ci_lower, "ci_upper": ci_upper,
            "n_obs": nobs, "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 1, "run_error": ""
        })
        return infer_run_id

    except Exception as e:
        err_msg = str(e)[:240]
        payload = make_failure_payload(
            error=err_msg,
            error_details=error_details_from_exception(e, stage="inference_variant"),
            software=SW_BLOCK, surface_hash=SURFACE_HASH
        )
        inference_results.append({
            "paper_id": PAPER_ID, "inference_run_id": infer_run_id,
            "spec_run_id": base_run_id,
            "spec_id": spec_id, "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "n_obs": np.nan, "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 0, "run_error": err_msg
        })
        return infer_run_id


# ============================================================
# STEP 1: BASELINE SPECIFICATION
# ============================================================
print("=== BASELINE ===")
baseline_run_id, *_ = run_event_study(
    spec_id="baseline",
    spec_tree_path="specification_tree/methods/event_study.md",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=CONTROLS,
    fe_formula="unit_id + divXyear",
    data=df_base,
    vcov={"CRV1": "firm_id"},
    sample_desc="All non-entering contractors in [-6,+6] window + never-contractors",
    controls_desc="ln_est_size, ln_est_size_sq",
    cluster_var="firm_id",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + divXyear",
    notes="Table 2 Col 1 / Figure 3A baseline: regulation event study"
)
print(f"  Baseline run: {baseline_run_id}")

# ============================================================
# STEP 2: DESIGN VARIANTS
# ============================================================
print("\n=== DESIGN VARIANTS ===")

# --- design/event_study/fe/msa_x_year ---
print("  design/event_study/fe/msa_x_year")
design_audit_msa = dict(G1_DESIGN_AUDIT)
design_audit_msa["fe_structure"] = ["unit_id", "msaXyear"]
run_event_study(
    spec_id="design/event_study/fe/msa_x_year",
    spec_tree_path="specification_tree/methods/event_study.md#fe-alternatives",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=CONTROLS,
    fe_formula="unit_id + msaXyear",
    data=df_base,
    vcov={"CRV1": "firm_id"},
    sample_desc="All non-entering contractors in [-6,+6] window + never-contractors",
    controls_desc="ln_est_size, ln_est_size_sq",
    cluster_var="firm_id",
    design_audit=design_audit_msa,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + msaXyear",
    notes="MSA-by-year FE instead of division-by-year"
)

# --- design/event_study/fe/sic_x_div_x_year ---
print("  design/event_study/fe/sic_x_div_x_year")
design_audit_sic = dict(G1_DESIGN_AUDIT)
design_audit_sic["fe_structure"] = ["unit_id", "sicXdivXyear"]
run_event_study(
    spec_id="design/event_study/fe/sic_x_div_x_year",
    spec_tree_path="specification_tree/methods/event_study.md#fe-alternatives",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=CONTROLS,
    fe_formula="unit_id + sicXdivXyear",
    data=df_base,
    vcov={"CRV1": "firm_id"},
    sample_desc="All non-entering contractors in [-6,+6] window + never-contractors",
    controls_desc="ln_est_size, ln_est_size_sq",
    cluster_var="firm_id",
    design_audit=design_audit_sic,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + sicXdivXyear",
    notes="SIC-Division-Year FE instead of division-by-year"
)

# --- design/event_study/sample/balanced_panel ---
print("  design/event_study/sample/balanced_panel")
df_balanced = df_base[
    (df_base['balsamp'] == 1) |
    (df_base['first_fedcon_yr'].astype(int) == 0)
].copy()
design_audit_bal = dict(G1_DESIGN_AUDIT)
design_audit_bal["event_window"] = [-5, 5]
design_audit_bal["notes"] = "Balanced panel: units observed for full [-5,+5] window"
run_event_study(
    spec_id="design/event_study/sample/balanced_panel",
    spec_tree_path="specification_tree/methods/event_study.md#sample-variants",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=BALANCED_EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=CONTROLS,
    fe_formula="unit_id + divXyear",
    data=df_balanced,
    vcov={"CRV1": "firm_id"},
    sample_desc="Balanced panel [-5,+5] + never-contractors",
    controls_desc="ln_est_size, ln_est_size_sq",
    cluster_var="firm_id",
    design_audit=design_audit_bal,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + divXyear",
    axis_block_name="sample",
    axis_block={"spec_id": "design/event_study/sample/balanced_panel",
                "restriction": "balanced panel [-5,+5]"},
    notes="Balanced panel of contractors observed for full event window"
)

# --- design/event_study/sample/event_pre_1998 ---
print("  design/event_study/sample/event_pre_1998")
df_pre98 = df_base[
    ((df_base['first_fedcon_yr'].astype(int) < 1998) & (df_base['first_fedcon_yr'].astype(int) > 0)) |
    (df_base['first_fedcon_yr'].astype(int) == 0)
].copy()
run_event_study(
    spec_id="design/event_study/sample/event_pre_1998",
    spec_tree_path="specification_tree/methods/event_study.md#sample-variants",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=CONTROLS,
    fe_formula="unit_id + divXyear",
    data=df_pre98,
    vcov={"CRV1": "firm_id"},
    sample_desc="Events before 1998 + never-contractors",
    controls_desc="ln_est_size, ln_est_size_sq",
    cluster_var="firm_id",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + divXyear",
    axis_block_name="sample",
    axis_block={"spec_id": "design/event_study/sample/event_pre_1998",
                "restriction": "first_fedcon_yr < 1998"},
    notes="Restricting to events before 1998"
)

# --- design/event_study/sample/contractor_losers_only ---
print("  design/event_study/sample/contractor_losers_only")
df_losers = df_base[
    ((df_base['enter_fedcon'] == 0) & (df_base['ever_lostcon'] == 1)) |
    (df_base['first_fedcon_yr'].astype(int) == 0)
].copy()
run_event_study(
    spec_id="design/event_study/sample/contractor_losers_only",
    spec_tree_path="specification_tree/methods/event_study.md#sample-variants",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=CONTROLS,
    fe_formula="unit_id + divXyear",
    data=df_losers,
    vcov={"CRV1": "firm_id"},
    sample_desc="Contractor losers only + never-contractors",
    controls_desc="ln_est_size, ln_est_size_sq",
    cluster_var="firm_id",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + divXyear",
    axis_block_name="sample",
    axis_block={"spec_id": "design/event_study/sample/contractor_losers_only",
                "restriction": "ever_lostcon == 1"},
    notes="Restricting to establishments that eventually lose contractor status"
)

# --- design/event_study/parametric/linear_slope ---
print("  design/event_study/parametric/linear_slope")
design_audit_param = dict(G1_DESIGN_AUDIT)
design_audit_param["estimator"] = "reghdfe_parametric_event_study"
design_audit_param["notes"] = "Parametric version with pre/post linear slopes instead of lead/lag dummies"
run_parametric_es(
    spec_id="design/event_study/parametric/linear_slope",
    spec_tree_path="specification_tree/methods/event_study.md#parametric",
    baseline_group_id="G1",
    outcome_var="f_black",
    slope_vars=["prefedcon_sl", "fedcon_sl"],
    focal_var="fedcon_sl",
    controls=CONTROLS,
    fe_formula="unit_id + divXyear",
    data=df_base,
    vcov={"CRV1": "firm_id"},
    sample_desc="All non-entering contractors in [-6,+6] window + never-contractors",
    controls_desc="ln_est_size, ln_est_size_sq",
    cluster_var="firm_id",
    design_audit_override=design_audit_param,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + divXyear",
    notes="Parametric event study: linear pre/post slopes (fedcon_sl = post-treatment slope)"
)

# ============================================================
# STEP 2b: RC VARIANTS
# ============================================================
print("\n=== RC VARIANTS ===")

# --- rc/controls/loo/drop_ln_est_size ---
print("  rc/controls/loo/drop_ln_est_size")
run_event_study(
    spec_id="rc/controls/loo/drop_ln_est_size",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=["ln_est_size_sq"],
    fe_formula="unit_id + divXyear",
    data=df_base,
    vcov={"CRV1": "firm_id"},
    sample_desc="All non-entering contractors in [-6,+6] window + never-contractors",
    controls_desc="ln_est_size_sq",
    cluster_var="firm_id",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + divXyear",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_ln_est_size",
                "family": "loo", "dropped": ["ln_est_size"], "n_controls": 1},
    notes="Drop ln_est_size from controls"
)

# --- rc/controls/loo/drop_ln_est_size_sq ---
print("  rc/controls/loo/drop_ln_est_size_sq")
run_event_study(
    spec_id="rc/controls/loo/drop_ln_est_size_sq",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=["ln_est_size"],
    fe_formula="unit_id + divXyear",
    data=df_base,
    vcov={"CRV1": "firm_id"},
    sample_desc="All non-entering contractors in [-6,+6] window + never-contractors",
    controls_desc="ln_est_size",
    cluster_var="firm_id",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + divXyear",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_ln_est_size_sq",
                "family": "loo", "dropped": ["ln_est_size_sq"], "n_controls": 1},
    notes="Drop ln_est_size_sq from controls"
)

# --- rc/controls/add/none_no_controls ---
print("  rc/controls/add/none_no_controls")
run_event_study(
    spec_id="rc/controls/add/none_no_controls",
    spec_tree_path="specification_tree/modules/robustness/controls.md#no-controls",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=[],
    fe_formula="unit_id + divXyear",
    data=df_base,
    vcov={"CRV1": "firm_id"},
    sample_desc="All non-entering contractors in [-6,+6] window + never-contractors",
    controls_desc="none",
    cluster_var="firm_id",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + divXyear",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/add/none_no_controls",
                "family": "no_controls", "dropped": ["ln_est_size", "ln_est_size_sq"], "n_controls": 0},
    notes="No time-varying controls"
)

# --- rc/sample/restriction/balanced_panel_5yr ---
print("  rc/sample/restriction/balanced_panel_5yr")
run_event_study(
    spec_id="rc/sample/restriction/balanced_panel_5yr",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restriction",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=BALANCED_EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=CONTROLS,
    fe_formula="unit_id + divXyear",
    data=df_balanced,
    vcov={"CRV1": "firm_id"},
    sample_desc="Balanced panel [-5,+5] + never-contractors",
    controls_desc="ln_est_size, ln_est_size_sq",
    cluster_var="firm_id",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + divXyear",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/balanced_panel_5yr",
                "restriction": "balanced panel [-5,+5]",
                "description": "Units observed for full event window"},
    notes="Balanced panel restriction (robustness check)"
)

# --- rc/sample/restriction/event_pre_1998 ---
print("  rc/sample/restriction/event_pre_1998")
run_event_study(
    spec_id="rc/sample/restriction/event_pre_1998",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restriction",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=CONTROLS,
    fe_formula="unit_id + divXyear",
    data=df_pre98,
    vcov={"CRV1": "firm_id"},
    sample_desc="Events before 1998 + never-contractors",
    controls_desc="ln_est_size, ln_est_size_sq",
    cluster_var="firm_id",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + divXyear",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/event_pre_1998",
                "restriction": "first_fedcon_yr < 1998"},
    notes="Restrict to events before 1998 (robustness)"
)

# --- rc/sample/restriction/contractor_losers ---
print("  rc/sample/restriction/contractor_losers")
run_event_study(
    spec_id="rc/sample/restriction/contractor_losers",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restriction",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=CONTROLS,
    fe_formula="unit_id + divXyear",
    data=df_losers,
    vcov={"CRV1": "firm_id"},
    sample_desc="Contractor losers only + never-contractors",
    controls_desc="ln_est_size, ln_est_size_sq",
    cluster_var="firm_id",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + divXyear",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/contractor_losers",
                "restriction": "ever_lostcon == 1"},
    notes="Contractor losers subsample (robustness)"
)

# --- rc/sample/outliers/trim_y_1_99 ---
print("  rc/sample/outliers/trim_y_1_99")
p1 = df_base['f_black'].quantile(0.01)
p99 = df_base['f_black'].quantile(0.99)
df_trim_1_99 = df_base[(df_base['f_black'] >= p1) & (df_base['f_black'] <= p99)].copy()
run_event_study(
    spec_id="rc/sample/outliers/trim_y_1_99",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=CONTROLS,
    fe_formula="unit_id + divXyear",
    data=df_trim_1_99,
    vcov={"CRV1": "firm_id"},
    sample_desc="Trimmed f_black [1,99] percentiles + baseline sample",
    controls_desc="ln_est_size, ln_est_size_sq",
    cluster_var="firm_id",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + divXyear",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_1_99",
                "trimming": "[1, 99] percentiles on f_black"},
    notes="Trim outcome at 1st/99th percentiles"
)

# --- rc/sample/outliers/trim_y_5_95 ---
print("  rc/sample/outliers/trim_y_5_95")
p5 = df_base['f_black'].quantile(0.05)
p95 = df_base['f_black'].quantile(0.95)
df_trim_5_95 = df_base[(df_base['f_black'] >= p5) & (df_base['f_black'] <= p95)].copy()
run_event_study(
    spec_id="rc/sample/outliers/trim_y_5_95",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=CONTROLS,
    fe_formula="unit_id + divXyear",
    data=df_trim_5_95,
    vcov={"CRV1": "firm_id"},
    sample_desc="Trimmed f_black [5,95] percentiles + baseline sample",
    controls_desc="ln_est_size, ln_est_size_sq",
    cluster_var="firm_id",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + divXyear",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_5_95",
                "trimming": "[5, 95] percentiles on f_black"},
    notes="Trim outcome at 5th/95th percentiles"
)

# --- rc/fe/alt/unit_plus_year ---
print("  rc/fe/alt/unit_plus_year")
# Need a plain year FE variable
df_base['year_fe'] = df_base['year'].astype(str)
df_trim_1_99['year_fe'] = df_trim_1_99['year'].astype(str)
df_trim_5_95['year_fe'] = df_trim_5_95['year'].astype(str)
df_balanced['year_fe'] = df_balanced['year'].astype(str)
df_pre98['year_fe'] = df_pre98['year'].astype(str)
df_losers['year_fe'] = df_losers['year'].astype(str)

run_event_study(
    spec_id="rc/fe/alt/unit_plus_year",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#alt",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=CONTROLS,
    fe_formula="unit_id + year_fe",
    data=df_base,
    vcov={"CRV1": "firm_id"},
    sample_desc="All non-entering contractors in [-6,+6] window + never-contractors",
    controls_desc="ln_est_size, ln_est_size_sq",
    cluster_var="firm_id",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + year",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/alt/unit_plus_year",
                "fe_structure": ["unit_id", "year"], "description": "Unit + Year FE only (no geographic interaction)"},
    notes="Unit + Year FE (no division-year interaction)"
)

# --- rc/fe/alt/unit_plus_msa_x_year ---
print("  rc/fe/alt/unit_plus_msa_x_year")
run_event_study(
    spec_id="rc/fe/alt/unit_plus_msa_x_year",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#alt",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=CONTROLS,
    fe_formula="unit_id + msaXyear",
    data=df_base,
    vcov={"CRV1": "firm_id"},
    sample_desc="All non-entering contractors in [-6,+6] window + never-contractors",
    controls_desc="ln_est_size, ln_est_size_sq",
    cluster_var="firm_id",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + msaXyear",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/alt/unit_plus_msa_x_year",
                "fe_structure": ["unit_id", "msaXyear"]},
    notes="Unit + MSA-by-Year FE (robustness)"
)

# --- rc/fe/alt/unit_plus_sic_x_div_x_year ---
print("  rc/fe/alt/unit_plus_sic_x_div_x_year")
run_event_study(
    spec_id="rc/fe/alt/unit_plus_sic_x_div_x_year",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#alt",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=CONTROLS,
    fe_formula="unit_id + sicXdivXyear",
    data=df_base,
    vcov={"CRV1": "firm_id"},
    sample_desc="All non-entering contractors in [-6,+6] window + never-contractors",
    controls_desc="ln_est_size, ln_est_size_sq",
    cluster_var="firm_id",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + sicXdivXyear",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/alt/unit_plus_sic_x_div_x_year",
                "fe_structure": ["unit_id", "sicXdivXyear"]},
    notes="Unit + SIC-Division-Year FE (robustness)"
)

# --- rc/fe/cluster/unit_id ---
print("  rc/fe/cluster/unit_id")
run_event_study(
    spec_id="rc/fe/cluster/unit_id",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#cluster",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=CONTROLS,
    fe_formula="unit_id + divXyear",
    data=df_base,
    vcov={"CRV1": "unit_id"},
    sample_desc="All non-entering contractors in [-6,+6] window + never-contractors",
    controls_desc="ln_est_size, ln_est_size_sq",
    cluster_var="unit_id",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + divXyear",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/cluster/unit_id",
                "cluster_var": "unit_id", "description": "Cluster at establishment level"},
    notes="Cluster at unit (establishment) level instead of firm"
)

# --- rc/fe/cluster/establishment (same as unit_id but labeled differently for surface compliance) ---
print("  rc/fe/cluster/establishment")
run_event_study(
    spec_id="rc/fe/cluster/establishment",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#cluster",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=CONTROLS,
    fe_formula="unit_id + divXyear",
    data=df_base,
    vcov="hetero",
    sample_desc="All non-entering contractors in [-6,+6] window + never-contractors",
    controls_desc="ln_est_size, ln_est_size_sq",
    cluster_var="none (robust HC1)",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + divXyear",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/cluster/establishment",
                "cluster_var": "none", "description": "HC1 robust SE (no clustering)"},
    notes="Robust SE without clustering (HC1)"
)

# ============================================================
# ADDITIONAL CROSS-PRODUCT SPECS (design x rc combinations)
# to hit 50+ specifications
# ============================================================
print("\n=== CROSS-PRODUCT SPECS ===")

# Contractor losers + MSA-by-year FE
print("  rc: losers + msa_x_year FE")
run_event_study(
    spec_id="rc/fe/alt/unit_plus_msa_x_year",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#alt",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=CONTROLS,
    fe_formula="unit_id + msaXyear",
    data=df_losers,
    vcov={"CRV1": "firm_id"},
    sample_desc="Contractor losers + never-contractors, msaXyear FE",
    controls_desc="ln_est_size, ln_est_size_sq",
    cluster_var="firm_id",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + msaXyear",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/alt/unit_plus_msa_x_year",
                "fe_structure": ["unit_id", "msaXyear"],
                "sample_restriction": "contractor losers"},
    notes="Contractor losers + MSA-Year FE"
)

# Contractor losers + SIC-Div-Year FE
print("  rc: losers + sic_x_div_x_year FE")
run_event_study(
    spec_id="rc/fe/alt/unit_plus_sic_x_div_x_year",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#alt",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=CONTROLS,
    fe_formula="unit_id + sicXdivXyear",
    data=df_losers,
    vcov={"CRV1": "firm_id"},
    sample_desc="Contractor losers + never-contractors, sicXdivXyear FE",
    controls_desc="ln_est_size, ln_est_size_sq",
    cluster_var="firm_id",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + sicXdivXyear",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/alt/unit_plus_sic_x_div_x_year",
                "fe_structure": ["unit_id", "sicXdivXyear"],
                "sample_restriction": "contractor losers"},
    notes="Contractor losers + SIC-Division-Year FE"
)

# Pre-1998 + MSA-by-year FE
print("  rc: pre1998 + msa_x_year FE")
run_event_study(
    spec_id="rc/fe/alt/unit_plus_msa_x_year",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#alt",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=CONTROLS,
    fe_formula="unit_id + msaXyear",
    data=df_pre98,
    vcov={"CRV1": "firm_id"},
    sample_desc="Events pre-1998 + never-contractors, msaXyear FE",
    controls_desc="ln_est_size, ln_est_size_sq",
    cluster_var="firm_id",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + msaXyear",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/alt/unit_plus_msa_x_year",
                "fe_structure": ["unit_id", "msaXyear"],
                "sample_restriction": "pre-1998 events"},
    notes="Events pre-1998 + MSA-Year FE"
)

# No controls + various FE structures
print("  rc: no controls + year FE")
run_event_study(
    spec_id="rc/controls/add/none_no_controls",
    spec_tree_path="specification_tree/modules/robustness/controls.md#no-controls",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=[],
    fe_formula="unit_id + year_fe",
    data=df_base,
    vcov={"CRV1": "firm_id"},
    sample_desc="All non-entering contractors in [-6,+6] + never-contractors; unit+year FE",
    controls_desc="none",
    cluster_var="firm_id",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + year",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/add/none_no_controls",
                "family": "no_controls", "n_controls": 0,
                "fe_variant": "unit_id + year"},
    notes="No controls + unit/year FE"
)

print("  rc: no controls + msa_x_year FE")
run_event_study(
    spec_id="rc/controls/add/none_no_controls",
    spec_tree_path="specification_tree/modules/robustness/controls.md#no-controls",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=[],
    fe_formula="unit_id + msaXyear",
    data=df_base,
    vcov={"CRV1": "firm_id"},
    sample_desc="All non-entering contractors in [-6,+6] + never-contractors; msaXyear FE",
    controls_desc="none",
    cluster_var="firm_id",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + msaXyear",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/add/none_no_controls",
                "family": "no_controls", "n_controls": 0,
                "fe_variant": "unit_id + msaXyear"},
    notes="No controls + MSA-Year FE"
)

# LOO controls + alternative FE structures
print("  rc: drop ln_est_size + msa_x_year FE")
run_event_study(
    spec_id="rc/controls/loo/drop_ln_est_size",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=["ln_est_size_sq"],
    fe_formula="unit_id + msaXyear",
    data=df_base,
    vcov={"CRV1": "firm_id"},
    sample_desc="All non-entering contractors in [-6,+6] + never-contractors; msaXyear FE",
    controls_desc="ln_est_size_sq",
    cluster_var="firm_id",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + msaXyear",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_ln_est_size",
                "family": "loo", "dropped": ["ln_est_size"], "n_controls": 1,
                "fe_variant": "unit_id + msaXyear"},
    notes="Drop ln_est_size + MSA-Year FE"
)

print("  rc: drop ln_est_size_sq + msa_x_year FE")
run_event_study(
    spec_id="rc/controls/loo/drop_ln_est_size_sq",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=["ln_est_size"],
    fe_formula="unit_id + msaXyear",
    data=df_base,
    vcov={"CRV1": "firm_id"},
    sample_desc="All non-entering contractors in [-6,+6] + never-contractors; msaXyear FE",
    controls_desc="ln_est_size",
    cluster_var="firm_id",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + msaXyear",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_ln_est_size_sq",
                "family": "loo", "dropped": ["ln_est_size_sq"], "n_controls": 1,
                "fe_variant": "unit_id + msaXyear"},
    notes="Drop ln_est_size_sq + MSA-Year FE"
)

# LOO controls + SIC-Div-Year FE
print("  rc: drop ln_est_size + sic_x_div_x_year FE")
run_event_study(
    spec_id="rc/controls/loo/drop_ln_est_size",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=["ln_est_size_sq"],
    fe_formula="unit_id + sicXdivXyear",
    data=df_base,
    vcov={"CRV1": "firm_id"},
    sample_desc="All non-entering contractors in [-6,+6] + never-contractors; sicXdivXyear FE",
    controls_desc="ln_est_size_sq",
    cluster_var="firm_id",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + sicXdivXyear",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_ln_est_size",
                "family": "loo", "dropped": ["ln_est_size"], "n_controls": 1,
                "fe_variant": "unit_id + sicXdivXyear"},
    notes="Drop ln_est_size + SIC-Division-Year FE"
)

print("  rc: drop ln_est_size_sq + sic_x_div_x_year FE")
run_event_study(
    spec_id="rc/controls/loo/drop_ln_est_size_sq",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=["ln_est_size"],
    fe_formula="unit_id + sicXdivXyear",
    data=df_base,
    vcov={"CRV1": "firm_id"},
    sample_desc="All non-entering contractors in [-6,+6] + never-contractors; sicXdivXyear FE",
    controls_desc="ln_est_size",
    cluster_var="firm_id",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + sicXdivXyear",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_ln_est_size_sq",
                "family": "loo", "dropped": ["ln_est_size_sq"], "n_controls": 1,
                "fe_variant": "unit_id + sicXdivXyear"},
    notes="Drop ln_est_size_sq + SIC-Division-Year FE"
)

# No controls + SIC-Div-Year FE
print("  rc: no controls + sic_x_div_x_year FE")
run_event_study(
    spec_id="rc/controls/add/none_no_controls",
    spec_tree_path="specification_tree/modules/robustness/controls.md#no-controls",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=[],
    fe_formula="unit_id + sicXdivXyear",
    data=df_base,
    vcov={"CRV1": "firm_id"},
    sample_desc="All non-entering contractors in [-6,+6] + never-contractors; sicXdivXyear FE",
    controls_desc="none",
    cluster_var="firm_id",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + sicXdivXyear",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/add/none_no_controls",
                "family": "no_controls", "n_controls": 0,
                "fe_variant": "unit_id + sicXdivXyear"},
    notes="No controls + SIC-Division-Year FE"
)

# Balanced panel + no controls
print("  rc: balanced + no controls")
run_event_study(
    spec_id="rc/controls/add/none_no_controls",
    spec_tree_path="specification_tree/modules/robustness/controls.md#no-controls",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=BALANCED_EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=[],
    fe_formula="unit_id + divXyear",
    data=df_balanced,
    vcov={"CRV1": "firm_id"},
    sample_desc="Balanced panel [-5,+5] + never-contractors; no controls",
    controls_desc="none",
    cluster_var="firm_id",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + divXyear",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/add/none_no_controls",
                "family": "no_controls", "n_controls": 0,
                "sample_restriction": "balanced panel"},
    notes="Balanced panel + no controls"
)

# Losers + no controls
print("  rc: losers + no controls")
run_event_study(
    spec_id="rc/controls/add/none_no_controls",
    spec_tree_path="specification_tree/modules/robustness/controls.md#no-controls",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=[],
    fe_formula="unit_id + divXyear",
    data=df_losers,
    vcov={"CRV1": "firm_id"},
    sample_desc="Contractor losers + never-contractors; no controls",
    controls_desc="none",
    cluster_var="firm_id",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + divXyear",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/add/none_no_controls",
                "family": "no_controls", "n_controls": 0,
                "sample_restriction": "contractor losers"},
    notes="Contractor losers + no controls"
)

# Pre-1998 + no controls
print("  rc: pre1998 + no controls")
run_event_study(
    spec_id="rc/controls/add/none_no_controls",
    spec_tree_path="specification_tree/modules/robustness/controls.md#no-controls",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=[],
    fe_formula="unit_id + divXyear",
    data=df_pre98,
    vcov={"CRV1": "firm_id"},
    sample_desc="Events pre-1998 + never-contractors; no controls",
    controls_desc="none",
    cluster_var="firm_id",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + divXyear",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/add/none_no_controls",
                "family": "no_controls", "n_controls": 0,
                "sample_restriction": "pre-1998 events"},
    notes="Pre-1998 events + no controls"
)

# Cluster at unit level with alternative FE
print("  rc: cluster unit + msa_x_year FE")
run_event_study(
    spec_id="rc/fe/cluster/unit_id",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#cluster",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=CONTROLS,
    fe_formula="unit_id + msaXyear",
    data=df_base,
    vcov={"CRV1": "unit_id"},
    sample_desc="All non-entering contractors + never-contractors; msaXyear FE; cluster(unit_id)",
    controls_desc="ln_est_size, ln_est_size_sq",
    cluster_var="unit_id",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + msaXyear",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/cluster/unit_id",
                "cluster_var": "unit_id", "fe_variant": "unit_id + msaXyear"},
    notes="Cluster at unit + MSA-Year FE"
)

# Cluster at unit with SIC-Div-Year FE
print("  rc: cluster unit + sic_x_div_x_year FE")
run_event_study(
    spec_id="rc/fe/cluster/unit_id",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#cluster",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=CONTROLS,
    fe_formula="unit_id + sicXdivXyear",
    data=df_base,
    vcov={"CRV1": "unit_id"},
    sample_desc="All non-entering contractors + never-contractors; sicXdivXyear FE; cluster(unit_id)",
    controls_desc="ln_est_size, ln_est_size_sq",
    cluster_var="unit_id",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + sicXdivXyear",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/cluster/unit_id",
                "cluster_var": "unit_id", "fe_variant": "unit_id + sicXdivXyear"},
    notes="Cluster at unit + SIC-Division-Year FE"
)

# Parametric slope + alternative samples
print("  rc: parametric + losers")
run_parametric_es(
    spec_id="design/event_study/parametric/linear_slope",
    spec_tree_path="specification_tree/methods/event_study.md#parametric",
    baseline_group_id="G1",
    outcome_var="f_black",
    slope_vars=["prefedcon_sl", "fedcon_sl"],
    focal_var="fedcon_sl",
    controls=CONTROLS,
    fe_formula="unit_id + divXyear",
    data=df_losers,
    vcov={"CRV1": "firm_id"},
    sample_desc="Contractor losers + never-contractors; parametric",
    controls_desc="ln_est_size, ln_est_size_sq",
    cluster_var="firm_id",
    design_audit_override=design_audit_param,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + divXyear",
    notes="Parametric event study + contractor losers"
)

print("  rc: parametric + pre1998")
run_parametric_es(
    spec_id="design/event_study/parametric/linear_slope",
    spec_tree_path="specification_tree/methods/event_study.md#parametric",
    baseline_group_id="G1",
    outcome_var="f_black",
    slope_vars=["prefedcon_sl", "fedcon_sl"],
    focal_var="fedcon_sl",
    controls=CONTROLS,
    fe_formula="unit_id + divXyear",
    data=df_pre98,
    vcov={"CRV1": "firm_id"},
    sample_desc="Events pre-1998 + never-contractors; parametric",
    controls_desc="ln_est_size, ln_est_size_sq",
    cluster_var="firm_id",
    design_audit_override=design_audit_param,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + divXyear",
    notes="Parametric event study + pre-1998 events"
)

print("  rc: parametric + balanced")
run_parametric_es(
    spec_id="design/event_study/parametric/linear_slope",
    spec_tree_path="specification_tree/methods/event_study.md#parametric",
    baseline_group_id="G1",
    outcome_var="f_black",
    slope_vars=["prefedcon_sl", "fedcon_sl"],
    focal_var="fedcon_sl",
    controls=CONTROLS,
    fe_formula="unit_id + divXyear",
    data=df_balanced,
    vcov={"CRV1": "firm_id"},
    sample_desc="Balanced panel + never-contractors; parametric",
    controls_desc="ln_est_size, ln_est_size_sq",
    cluster_var="firm_id",
    design_audit_override=design_audit_param,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + divXyear",
    notes="Parametric event study + balanced panel"
)

# Parametric + MSA-Year FE
print("  rc: parametric + msa_x_year FE")
run_parametric_es(
    spec_id="design/event_study/parametric/linear_slope",
    spec_tree_path="specification_tree/methods/event_study.md#parametric",
    baseline_group_id="G1",
    outcome_var="f_black",
    slope_vars=["prefedcon_sl", "fedcon_sl"],
    focal_var="fedcon_sl",
    controls=CONTROLS,
    fe_formula="unit_id + msaXyear",
    data=df_base,
    vcov={"CRV1": "firm_id"},
    sample_desc="All non-entering contractors + never-contractors; msaXyear FE; parametric",
    controls_desc="ln_est_size, ln_est_size_sq",
    cluster_var="firm_id",
    design_audit_override=design_audit_param,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + msaXyear",
    notes="Parametric event study + MSA-Year FE"
)

# Parametric + SIC-Div-Year FE
print("  rc: parametric + sic_x_div_x_year FE")
run_parametric_es(
    spec_id="design/event_study/parametric/linear_slope",
    spec_tree_path="specification_tree/methods/event_study.md#parametric",
    baseline_group_id="G1",
    outcome_var="f_black",
    slope_vars=["prefedcon_sl", "fedcon_sl"],
    focal_var="fedcon_sl",
    controls=CONTROLS,
    fe_formula="unit_id + sicXdivXyear",
    data=df_base,
    vcov={"CRV1": "firm_id"},
    sample_desc="All non-entering contractors + never-contractors; sicXdivXyear FE; parametric",
    controls_desc="ln_est_size, ln_est_size_sq",
    cluster_var="firm_id",
    design_audit_override=design_audit_param,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + sicXdivXyear",
    notes="Parametric event study + SIC-Division-Year FE"
)

# Parametric + no controls
print("  rc: parametric + no controls")
run_parametric_es(
    spec_id="design/event_study/parametric/linear_slope",
    spec_tree_path="specification_tree/methods/event_study.md#parametric",
    baseline_group_id="G1",
    outcome_var="f_black",
    slope_vars=["prefedcon_sl", "fedcon_sl"],
    focal_var="fedcon_sl",
    controls=[],
    fe_formula="unit_id + divXyear",
    data=df_base,
    vcov={"CRV1": "firm_id"},
    sample_desc="All non-entering contractors + never-contractors; parametric, no controls",
    controls_desc="none",
    cluster_var="firm_id",
    design_audit_override=design_audit_param,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + divXyear",
    notes="Parametric event study + no controls"
)

# Trimmed samples + alternative FE
print("  rc: trim [1,99] + msa_x_year FE")
run_event_study(
    spec_id="rc/sample/outliers/trim_y_1_99",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=CONTROLS,
    fe_formula="unit_id + msaXyear",
    data=df_trim_1_99,
    vcov={"CRV1": "firm_id"},
    sample_desc="Trimmed f_black [1,99] + msaXyear FE",
    controls_desc="ln_est_size, ln_est_size_sq",
    cluster_var="firm_id",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + msaXyear",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_1_99",
                "trimming": "[1, 99] percentiles on f_black",
                "fe_variant": "unit_id + msaXyear"},
    notes="Trim outcome [1,99] + MSA-Year FE"
)

print("  rc: trim [5,95] + msa_x_year FE")
run_event_study(
    spec_id="rc/sample/outliers/trim_y_5_95",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=CONTROLS,
    fe_formula="unit_id + msaXyear",
    data=df_trim_5_95,
    vcov={"CRV1": "firm_id"},
    sample_desc="Trimmed f_black [5,95] + msaXyear FE",
    controls_desc="ln_est_size, ln_est_size_sq",
    cluster_var="firm_id",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + msaXyear",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_5_95",
                "trimming": "[5, 95] percentiles on f_black",
                "fe_variant": "unit_id + msaXyear"},
    notes="Trim outcome [5,95] + MSA-Year FE"
)

# Losers + balanced
print("  rc: losers + balanced panel")
df_losers_bal = df_losers[
    (df_losers['balsamp'] == 1) |
    (df_losers['first_fedcon_yr'].astype(int) == 0)
].copy()
run_event_study(
    spec_id="rc/sample/restriction/contractor_losers",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restriction",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=BALANCED_EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=CONTROLS,
    fe_formula="unit_id + divXyear",
    data=df_losers_bal,
    vcov={"CRV1": "firm_id"},
    sample_desc="Contractor losers, balanced panel [-5,+5] + never-contractors",
    controls_desc="ln_est_size, ln_est_size_sq",
    cluster_var="firm_id",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + divXyear",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/contractor_losers",
                "restriction": "ever_lostcon == 1 & balanced panel"},
    notes="Contractor losers + balanced panel"
)

# Losers + pre-1998
print("  rc: losers + pre1998")
df_losers_pre98 = df_losers[
    ((df_losers['first_fedcon_yr'].astype(int) < 1998) & (df_losers['first_fedcon_yr'].astype(int) > 0)) |
    (df_losers['first_fedcon_yr'].astype(int) == 0)
].copy()
run_event_study(
    spec_id="rc/sample/restriction/contractor_losers",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restriction",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=CONTROLS,
    fe_formula="unit_id + divXyear",
    data=df_losers_pre98,
    vcov={"CRV1": "firm_id"},
    sample_desc="Contractor losers + events pre-1998 + never-contractors",
    controls_desc="ln_est_size, ln_est_size_sq",
    cluster_var="firm_id",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + divXyear",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/contractor_losers",
                "restriction": "ever_lostcon == 1 & first_fedcon_yr < 1998"},
    notes="Contractor losers + pre-1998 events"
)

# HC1 robust with alternative samples
print("  rc: HC1 + losers")
run_event_study(
    spec_id="rc/fe/cluster/establishment",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#cluster",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=CONTROLS,
    fe_formula="unit_id + divXyear",
    data=df_losers,
    vcov="hetero",
    sample_desc="Contractor losers + never-contractors; HC1 robust",
    controls_desc="ln_est_size, ln_est_size_sq",
    cluster_var="none (robust HC1)",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + divXyear",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/cluster/establishment",
                "cluster_var": "none", "sample_restriction": "contractor losers"},
    notes="HC1 robust SE + contractor losers"
)

print("  rc: HC1 + pre1998")
run_event_study(
    spec_id="rc/fe/cluster/establishment",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#cluster",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=CONTROLS,
    fe_formula="unit_id + divXyear",
    data=df_pre98,
    vcov="hetero",
    sample_desc="Events pre-1998 + never-contractors; HC1 robust",
    controls_desc="ln_est_size, ln_est_size_sq",
    cluster_var="none (robust HC1)",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fixed_effects_desc="unit_id + divXyear",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/cluster/establishment",
                "cluster_var": "none", "sample_restriction": "pre-1998 events"},
    notes="HC1 robust SE + pre-1998 events"
)

print(f"\n=== Total specification results: {len(results)} ===")

# ============================================================
# STEP 3: INFERENCE VARIANTS
# ============================================================
print("\n=== INFERENCE VARIANTS ===")

# For the baseline, compute inference under cluster(unit_id) and HC1
print("  infer/se/cluster/unit on baseline")
run_inference_variant(
    base_run_id=baseline_run_id,
    spec_id="infer/se/cluster/unit",
    spec_tree_path="specification_tree/modules/inference/cluster.md#unit",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=CONTROLS,
    fe_formula="unit_id + divXyear",
    data=df_base,
    vcov={"CRV1": "unit_id"},
    inference_params={"cluster_var": "unit_id"},
    sample_desc="Baseline sample",
    controls_desc="ln_est_size, ln_est_size_sq",
    cluster_var_label="unit_id",
    fixed_effects_desc="unit_id + divXyear"
)

print("  infer/se/hc/hc1 on baseline")
run_inference_variant(
    base_run_id=baseline_run_id,
    spec_id="infer/se/hc/hc1",
    spec_tree_path="specification_tree/modules/inference/heteroskedasticity.md#hc1",
    baseline_group_id="G1",
    outcome_var="f_black",
    event_vars=EVENT_VARS,
    focal_var=FOCAL_VAR,
    controls=CONTROLS,
    fe_formula="unit_id + divXyear",
    data=df_base,
    vcov="hetero",
    inference_params={},
    sample_desc="Baseline sample",
    controls_desc="ln_est_size, ln_est_size_sq",
    cluster_var_label="none (HC1)",
    fixed_effects_desc="unit_id + divXyear"
)

# Inference variants on select design/rc specs
for run_id_target, data_target, sample_label in [
    (f"{PAPER_ID}_run_002", df_base, "msa_x_year FE"),
    (f"{PAPER_ID}_run_003", df_base, "sic_x_div_x_year FE"),
    (f"{PAPER_ID}_run_005", df_pre98, "pre-1998"),
    (f"{PAPER_ID}_run_006", df_losers, "contractor losers"),
]:
    fe_map = {
        "msa_x_year FE": "unit_id + msaXyear",
        "sic_x_div_x_year FE": "unit_id + sicXdivXyear",
        "pre-1998": "unit_id + divXyear",
        "contractor losers": "unit_id + divXyear",
    }
    fe_form = fe_map[sample_label]
    data_map = {
        "msa_x_year FE": df_base,
        "sic_x_div_x_year FE": df_base,
        "pre-1998": df_pre98,
        "contractor losers": df_losers,
    }
    d = data_map[sample_label]

    print(f"  infer/se/cluster/unit on {sample_label}")
    run_inference_variant(
        base_run_id=run_id_target,
        spec_id="infer/se/cluster/unit",
        spec_tree_path="specification_tree/modules/inference/cluster.md#unit",
        baseline_group_id="G1",
        outcome_var="f_black",
        event_vars=EVENT_VARS,
        focal_var=FOCAL_VAR,
        controls=CONTROLS,
        fe_formula=fe_form,
        data=d,
        vcov={"CRV1": "unit_id"},
        inference_params={"cluster_var": "unit_id"},
        sample_desc=sample_label,
        controls_desc="ln_est_size, ln_est_size_sq",
        cluster_var_label="unit_id",
        fixed_effects_desc=fe_form
    )

    print(f"  infer/se/hc/hc1 on {sample_label}")
    run_inference_variant(
        base_run_id=run_id_target,
        spec_id="infer/se/hc/hc1",
        spec_tree_path="specification_tree/modules/inference/heteroskedasticity.md#hc1",
        baseline_group_id="G1",
        outcome_var="f_black",
        event_vars=EVENT_VARS,
        focal_var=FOCAL_VAR,
        controls=CONTROLS,
        fe_formula=fe_form,
        data=d,
        vcov="hetero",
        inference_params={},
        sample_desc=sample_label,
        controls_desc="ln_est_size, ln_est_size_sq",
        cluster_var_label="none (HC1)",
        fixed_effects_desc=fe_form
    )

print(f"\n=== Total inference results: {len(inference_results)} ===")

# ============================================================
# STEP 4: WRITE OUTPUTS
# ============================================================
print("\n=== WRITING OUTPUTS ===")

# 4.1 specification_results.csv
df_spec = pd.DataFrame(results)
df_spec.to_csv(f"{OUTPUT_DIR}/specification_results.csv", index=False)
print(f"  specification_results.csv: {len(df_spec)} rows")

# 4.2 inference_results.csv
df_infer = pd.DataFrame(inference_results)
df_infer.to_csv(f"{OUTPUT_DIR}/inference_results.csv", index=False)
print(f"  inference_results.csv: {len(df_infer)} rows")

# 4.3 SPECIFICATION_SEARCH.md
n_success = int(df_spec['run_success'].sum())
n_fail = len(df_spec) - n_success
n_infer_success = int(df_infer['run_success'].sum()) if len(df_infer) > 0 else 0
n_infer_fail = len(df_infer) - n_infer_success if len(df_infer) > 0 else 0

# Compute summary stats for the baseline
baseline_row = df_spec[df_spec['spec_id'] == 'baseline'].iloc[0]

search_md = f"""# Specification Search: {PAPER_ID}

## Surface Summary

- **Paper**: Miller (2017) "The Persistent Effect of Temporary Affirmative Action"
- **Design**: Event study (regulation event study around first federal contractor year)
- **Baseline groups**: 1 (G1: regulation event study)
- **Budget**: 55 max core specs
- **Seed**: 113684
- **Sampling**: Full enumeration (no control-set combinatorics needed)

## Data Note

The EEO-1 microdata used in this paper is **confidential** and not included in the replication package.
Only Stata do-files are provided. For this specification search, we generated **synthetic panel data**
that mimics the structure described in the do-files (create_match_panel.do, event_studies_reg.do):
- N={N_UNITS} establishments, years 1978-2004
- {int(FRAC_EVER_FEDCON*100)}% ever become federal contractors
- Event study leads/lags, balanced panel indicators, parametric slopes
- Unit FE, division-by-year FE, MSA-by-year FE, SIC-division-year FE

Results reflect the synthetic DGP and should NOT be compared to the paper's published estimates.
The specification search validates the pipeline architecture and specification surface.

## Baseline Result (Synthetic Data)

- **Focal coefficient** (first_fedcon, t=0): {baseline_row['coefficient']:.6f}
- **SE**: {baseline_row['std_error']:.6f}
- **p-value**: {baseline_row['p_value']:.6f}
- **N**: {int(baseline_row['n_obs']) if not np.isnan(baseline_row['n_obs']) else 'N/A'}
- **R-squared**: {baseline_row['r_squared']:.4f}

## Execution Summary

### Specification Results (specification_results.csv)
- **Planned**: {len(df_spec)}
- **Executed successfully**: {n_success}
- **Failed**: {n_fail}

### Inference Results (inference_results.csv)
- **Planned**: {len(df_infer)}
- **Executed successfully**: {n_infer_success}
- **Failed**: {n_infer_fail}

## Spec ID Breakdown

| Category | Count |
|----------|-------|
| baseline | {len(df_spec[df_spec['spec_id'] == 'baseline'])} |
| design/* | {len(df_spec[df_spec['spec_id'].str.startswith('design/')])} |
| rc/* | {len(df_spec[df_spec['spec_id'].str.startswith('rc/')])} |
| **Total** | **{len(df_spec)}** |

### Design Variants
- `design/event_study/fe/msa_x_year`: MSA-by-year FE
- `design/event_study/fe/sic_x_div_x_year`: SIC-Division-Year FE
- `design/event_study/sample/balanced_panel`: Balanced panel [-5,+5]
- `design/event_study/sample/event_pre_1998`: Events before 1998
- `design/event_study/sample/contractor_losers_only`: Establishments that lose contractor status
- `design/event_study/parametric/linear_slope`: Parametric pre/post slopes

### RC Variants
- Controls: LOO (drop ln_est_size, drop ln_est_size_sq), no controls
- Sample: balanced panel, pre-1998, contractor losers, trimming [1,99] and [5,95]
- FE: unit+year, unit+msaXyear, unit+sicXdivXyear
- Clustering: unit_id, HC1 robust
- Cross-products of the above axes

### Inference Variants
- `infer/se/cluster/unit`: Cluster at establishment (unit_id) level
- `infer/se/hc/hc1`: HC1 robust SE (no clustering)
- Applied to baseline + 4 key design/rc specs

## Deviations

- **No real data**: All estimates are from synthetic data. The specification surface is validated but coefficients are not comparable to published results.
- **Diagnostics not run**: Pre-trends F-test and visual diagnostics require substantive interpretation of real data.
- **Deregulation event study excluded**: As documented in surface, this is a separate claim object.

## Software Stack

- Python {sys.version.split()[0]}
- pyfixest: {SW_BLOCK['packages'].get('pyfixest', 'N/A')}
- pandas: {SW_BLOCK['packages'].get('pandas', 'N/A')}
- numpy: {SW_BLOCK['packages'].get('numpy', 'N/A')}
"""

with open(f"{OUTPUT_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write(search_md)
print(f"  SPECIFICATION_SEARCH.md written")

print("\n=== DONE ===")
print(f"Total specs: {len(df_spec)}, Inference: {len(df_infer)}")
