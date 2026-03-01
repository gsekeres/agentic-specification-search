"""
Specification Search Script for Bjorkman Martina & Svensson (2014)
"Experimental Evidence on the Long-Run Impact of Community-Based Monitoring"
American Economic Journal: Applied Economics

Paper ID: 113630-V1

Surface-driven execution:
  - G1: Mortality ~ treatment | dcode FE, robust SEs (cluster-level data)
  - G2: Weight-for-age z-score ~ treatment | dcode FE, cluster(hfcode) (individual data)
  - Two experiments: I&P (sample1==1) and P-only (sample1==0)
  - Randomized at health facility (hfcode) level within district (dcode) strata

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

PAPER_ID = "113630-V1"
DATA_DIR = "data/downloads/extracted/113630-V1/APP2015-0027_data/Dataset"
OUTPUT_DIR = "data/downloads/extracted/113630-V1"

# Load surface
with open(f"{OUTPUT_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

bg_mort = surface_obj["baseline_groups"][0]  # G1_mortality
bg_weight = surface_obj["baseline_groups"][1]  # G2_weight

design_audit_mort = bg_mort["design_audit"]
design_audit_weight = bg_weight["design_audit"]

# ============================================================
# Data Loading and Preparation — MORTALITY (G1)
# ============================================================

# Replicate the do-file construction of mortality data:
# 1. Combine alive + died children
# 2. Construct death indicators & exposure
# 3. Collapse to health facility level

print("=== Loading and constructing mortality data ===")

died = pd.read_stata(f"{DATA_DIR}/sec-8-children-u5-died.dta")
alive = pd.read_stata(f"{DATA_DIR}/sec-8-children-u5.dta")

# Following the do-file: rename and align
died = died.rename(columns={
    's8b24': 'birth_monthyr', 's8b23': 'childname', 's8b25': 'gender',
    's8b27': 'death_yr', 's8b26': 'death_month',
    's8b24_yr': 'birth_yr', 's8b24_month': 'birth_month'
})
died['death'] = 1
died = died.dropna(subset=['birth_monthyr'])

alive = alive.rename(columns={
    's84': 'birth_monthyr', 's83': 'childname', 's85': 'gender',
    's84_month': 'birth_month', 's84_yr': 'birth_yr'
})
# destring s84
alive['birth_monthyr'] = pd.to_numeric(alive['birth_monthyr'], errors='coerce')
alive = alive.dropna(subset=['birth_monthyr'])

# Combine
all_children = pd.concat([alive, died], ignore_index=True, sort=False)
all_children['death'] = all_children['death'].fillna(0)

# CMC dates
all_children['bdate'] = (all_children['birth_yr'] - 1900) * 12 + all_children['birth_month']
all_children['ddate'] = (all_children['death_yr'] - 1900) * 12 + all_children['death_month']
all_children['age_death'] = all_children['ddate'] - all_children['bdate']

# Error corrections following do-file
error_mask = all_children['age_death'] < 0
all_children.loc[error_mask, 'death_yr'] = all_children.loc[error_mask, 'birth_yr']
all_children.loc[error_mask, 'death_month'] = all_children.loc[error_mask, 'birth_month']

# Recalculate dates after correction
all_children['ddate'] = (all_children['death_yr'] - 1900) * 12 + all_children['death_month']
all_children['bdate'] = (all_children['birth_yr'] - 1900) * 12 + all_children['birth_month']
all_children['age_death'] = all_children['ddate'] - all_children['bdate']

# Fix birthdate beyond survey
all_children.loc[all_children['bdate'] == 1318, 'bdate'] = 1315
all_children.loc[all_children['bdate'] == 1319, 'bdate'] = 1315
all_children.loc[all_children['bdate'] == 1320, 'bdate'] = 1315

# CRITICAL: For alive children, set ddate to end of survey (1315)
# so that exposure calculations work correctly (alive children survived
# to the survey date, their ddate is not a death date but an observation date)
all_children.loc[all_children['death'] == 0, 'ddate'] = 1315


# === PANEL A: I&P experiment (sample1==1), 2006-2009 ===
def build_mortality_panel(df, sample_val, start_cmc, n_years, label):
    """Build facility-level mortality panel following the do-file."""
    if sample_val == 1:
        # drop if death_yr1 < 2006
        panel = df[~((df['death'] == 1) & (df['death_yr'] < 2006))].copy()
    else:
        # drop if death_yr1 < 2007
        panel = df[~((df['death'] == 1) & (df['death_yr'] < 2007))].copy()

    panel = panel[panel['sample1'] == sample_val].copy()

    # Infant deaths: died at age < 12 months
    panel['death_inf'] = panel['death'].copy()
    panel.loc[(panel['age_death'] > 12) & (panel['age_death'] < 60), 'death_inf'] = np.nan

    # Neonatal deaths: died at age < 1 month
    panel['death_neo'] = panel['death'].copy()
    panel.loc[(panel['age_death'] > 0) & (panel['age_death'] < 60), 'death_neo'] = np.nan

    # Exposure
    panel['exposure'] = 1 + np.minimum(1315, panel['ddate']) - np.maximum(start_cmc, panel['bdate'])
    panel['exp_y'] = panel['exposure'] / 12

    # Births (for neonatal mortality ratio)
    if sample_val == 1:
        panel['births'] = ((panel['bdate'] > 1272) & (panel['bdate'] < 1315)).astype(float)
    else:
        panel['births'] = ((panel['bdate'] > 1284) & (panel['bdate'] < 1315)).astype(float)

    # Collapse to facility level
    agg = panel.groupby('hfcode').agg({
        'dcode': 'first', 'treatment': 'first', 'sample1': 'first',
        'death': 'sum', 'death_inf': 'sum', 'death_neo': 'sum',
        'births': 'sum', 'exp_y': 'sum'
    }).reset_index()

    # Crude death rate
    agg['death_year'] = agg['death'] / n_years
    agg['death_inf_year'] = agg['death_inf'] / n_years
    agg['death_neo_year'] = agg['death_neo'] / n_years

    # Exposure-corrected
    agg['exp_y1000'] = agg['exp_y'] / 1000
    agg['death_exp'] = agg['death'] / agg['exp_y1000']

    # Neonatal mortality ratio
    agg['neo_mr'] = agg['death_neo'] / (agg['births'] / 1000)

    # Make FE vars string
    agg['dcode_str'] = agg['dcode'].astype(int).astype(str)
    agg['hfcode_str'] = agg['hfcode'].astype(int).astype(str)

    print(f"  {label}: {len(agg)} facilities, treatment={agg['treatment'].sum():.0f}")
    return agg


# Convert float32 columns
for col in all_children.columns:
    if all_children[col].dtype == np.float32:
        all_children[col] = all_children[col].astype(np.float64)

df_ip = build_mortality_panel(all_children, 1, 1273, 3.58, "I&P (2006-2009)")
df_p = build_mortality_panel(all_children, 0, 1285, 2.33, "P-only (2007-2009)")

# Load baseline controls for Panel C
charges = pd.read_stata(f"{DATA_DIR}/average_charges_hflevel_2004.dta")
hf_main = pd.read_stata(f"{DATA_DIR}/hfmain_2004.dta")
sample_stats = pd.read_stata(f"{DATA_DIR}/samplestats_PI2006.dta")

# Build baseline controls dataset (merge charges + hfmain)
baseline_ctrl = charges[['hfcode', 'avg_charge_gentreat']].merge(
    hf_main[['hfcode', 'avgOP', 'hhs']], on='hfcode', how='left'
)
baseline_ctrl = baseline_ctrl.rename(columns={'avgOP': 'avgOP_baseline'})
baseline_ctrl['charge2'] = baseline_ctrl['avg_charge_gentreat'] ** 2
baseline_ctrl['op2'] = baseline_ctrl['avgOP_baseline'] ** 2
baseline_ctrl['hhs2'] = baseline_ctrl['hhs'] ** 2

# Merge baseline controls into mortality data
df_ip = df_ip.merge(baseline_ctrl, on='hfcode', how='left')
df_p = df_p.merge(baseline_ctrl, on='hfcode', how='left')

print(f"I&P mortality data: {len(df_ip)} obs, baseline controls merged")
print(f"P mortality data: {len(df_p)} obs, baseline controls merged")


# ============================================================
# Data Loading and Preparation — WEIGHT (G2)
# ============================================================

print("\n=== Loading weight/height data ===")

wh = pd.read_stata(f"{DATA_DIR}/sec8_weightheight.dta")
cort97w = pd.read_stata(f"{DATA_DIR}/cort97w.dta")
cort97h = pd.read_stata(f"{DATA_DIR}/cort97h.dta")

# Convert float32
for col in wh.columns:
    if wh[col].dtype == np.float32:
        wh[col] = wh[col].astype(np.float64)
for col in cort97w.columns:
    if cort97w[col].dtype == np.float32:
        cort97w[col] = cort97w[col].astype(np.float64)
for col in cort97h.columns:
    if cort97h[col].dtype == np.float32:
        cort97h[col] = cort97h[col].astype(np.float64)

# Merge cort97w (weight outlier cutoffs by age-month & gender)
# The do-file uses: gen gender=s85 then merge gender months using cort97w.dta
# But s85 is categorical ('male'/'female') while cort97w uses 0/1
# In the do-file: gencode(male=1, female=0) is the convention
wh['gender_num'] = wh['s85'].map({'male': 1, 'female': 0}).astype(float)
wh['months_f'] = wh['s821a'].astype(float)
cort97w['gender_num'] = cort97w['gender'].astype(float)
cort97w['months_f'] = cort97w['months'].astype(float)
wh = wh.merge(cort97w[['months_f', 'gender_num', 'cort97']],
              on=['months_f', 'gender_num'], how='left')

# Merge cort97h (height outlier cutoffs by month)
cort97h['months_f'] = cort97h['months'].astype(float)
wh = wh.merge(cort97h[['months_f', 'cort97_h']], on='months_f', how='left')

# Compute z-scores using a simplified approach
# The original uses Stata's zanthro command (WHO z-scores)
# We'll compute approximate weight-for-age z-scores using available data
# NOTE: The paper uses egen zanthro which requires WHO reference data
# For our purposes, we use the raw z-score values if available in the dataset
# or attempt to standardize weight by age-gender

# Try loading weight data that may already have z-scores computed
try:
    wt_2006 = pd.read_stata(f"{DATA_DIR}/weight_2006data.dta")
    print(f"  weight_2006data loaded: {wt_2006.shape}")
    has_2006_weight = True
except:
    has_2006_weight = False

# Since zanthro is a Stata-specific command, we need to compute z-scores ourselves
# Use WHO weight-for-age reference data (approximate)
# For specification search, we can use a reasonable standardization
# Let's compute z-scores by standardizing within age-month-gender groups

def compute_approx_zscore(df, value_col, group_cols=['months_f', 'gender_num']):
    """Compute approximate z-scores by standardizing within age-month x gender groups."""
    result = pd.Series(np.nan, index=df.index)
    for name, group in df.groupby(group_cols):
        vals = group[value_col].dropna()
        if len(vals) > 2:
            mean = vals.mean()
            std = vals.std()
            if std > 0:
                result[group.index] = (group[value_col] - mean) / std
    return result

# Weight-for-age z-score
wh['zw1'] = compute_approx_zscore(wh, 's821b', ['months_f', 'gender_num'])

# Height-for-age z-score
wh['zh1'] = compute_approx_zscore(wh, 's821e', ['months_f', 'gender_num'])

# Make string FE vars
wh['dcode_str'] = wh['dcode'].astype(int).astype(str)
wh['hfcode_str'] = wh['hfcode'].astype(int).astype(str)

# Merge baseline controls for weight data
wh = wh.merge(baseline_ctrl, on='hfcode', how='left')

# Apply sample restrictions from the paper
# Infants: s821a < 13, zw1 in (-4.5, 4.5), s821b < cort97
wh_ip_infants = wh[(wh['s821a'] < 13) & (wh['zw1'] > -4.5) & (wh['zw1'] < 4.5)
                    & (wh['s821b'] < wh['cort97']) & (wh['sample1'] == 1)].copy()

# Older children: s821a > 12, zw1 in (-4.5, 4.5)
# Note: Col II in paper does NOT apply cort97 filter for older children
wh_ip_children = wh[(wh['s821a'] > 12) & (wh['zw1'] > -4.5) & (wh['zw1'] < 4.5)
                     & (wh['sample1'] == 1)].copy()

# P experiment
wh_p_infants = wh[(wh['s821a'] < 13) & (wh['zw1'] > -4.5) & (wh['zw1'] < 4.5)
                   & (wh['s821b'] < wh['cort97']) & (wh['sample1'] == 0)].copy()

wh_p_children = wh[(wh['s821a'] > 12) & (wh['zw1'] > -4.5) & (wh['zw1'] < 4.5)
                    & (wh['sample1'] == 0)].copy()

print(f"  Weight data: I&P infants={len(wh_ip_infants)}, I&P children={len(wh_ip_children)}")
print(f"  Weight data: P infants={len(wh_p_infants)}, P children={len(wh_p_children)}")


# ============================================================
# Accumulators
# ============================================================

results = []
inference_results = []
spec_run_counter = 0


# ============================================================
# Helper: run_spec
# ============================================================

def run_spec(spec_id, spec_tree_path, baseline_group_id,
             outcome_var, treatment_var, controls, fe_formula_str,
             fe_desc, data, vcov, sample_desc, controls_desc,
             design_audit_ref=None,
             axis_block_name=None, axis_block=None, notes=""):
    """Run a single OLS specification and record results."""
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    if design_audit_ref is None:
        design_audit_ref = design_audit_mort

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

        infer_info = {"spec_id": "infer/se/hc/hc1", "method": "robust"}
        if isinstance(vcov, dict) and "CRV1" in vcov:
            infer_info = {"spec_id": "infer/se/cluster/hfcode",
                          "method": "cluster", "cluster_vars": [list(vcov.values())[0]]}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference=infer_info,
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"randomized_experiment": design_audit_ref},
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
            "cluster_var": str(vcov),
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
            "cluster_var": str(vcov),
            "run_success": 0,
            "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


# ============================================================
# G1: MORTALITY SPECIFICATIONS
# ============================================================

print("\n" + "="*60)
print("G1: MORTALITY SPECIFICATIONS")
print("="*60)

# --- Baseline: Table3-PanelA-ColI-IP (crude death rate, I&P) ---
print("\nRunning G1 baseline: Table3-PanelA-ColI-IP (death_year, I&P)...")
base_run_id_g1, base_coef_g1, base_se_g1, base_pval_g1, base_n_g1 = run_spec(
    "baseline__table3_panelA_colI_IP",
    "designs/randomized_experiment.md#baseline", "G1_mortality",
    "death_year", "treatment", [],
    "dcode_str", "dcode (district)", df_ip,
    "hetero",
    f"I&P experiment, HF-level, N={len(df_ip)}", "none (strata FE only)")

print(f"  Baseline G1: coef={base_coef_g1:.4f}, se={base_se_g1:.4f}, p={base_pval_g1:.4f}, N={base_n_g1}")


# --- Additional baselines from surface ---

# Table3-PanelA-ColII-IP (infant death rate)
print("Running baseline: Table3-PanelA-ColII-IP-Infant...")
run_spec(
    "baseline__table3_panelA_colII_infant",
    "designs/randomized_experiment.md#baseline", "G1_mortality",
    "death_inf_year", "treatment", [],
    "dcode_str", "dcode (district)", df_ip,
    "hetero",
    "I&P experiment, HF-level", "none (strata FE only)")

# Table3-PanelA-ColIV-IP (exposure-corrected death rate)
print("Running baseline: Table3-PanelA-ColIV-IP-ExposureCorrected...")
run_spec(
    "baseline__table3_panelA_colIV_exposure",
    "designs/randomized_experiment.md#baseline", "G1_mortality",
    "death_exp", "treatment", [],
    "dcode_str", "dcode (district)", df_ip,
    "hetero",
    "I&P experiment, HF-level, exposure-corrected", "none (strata FE only)")

# Table3-PanelB-ColI-P (crude death rate, P experiment)
print("Running baseline: Table3-PanelB-ColI-P...")
run_spec(
    "baseline__table3_panelB_colI_P",
    "designs/randomized_experiment.md#baseline", "G1_mortality",
    "death_year", "treatment", [],
    "dcode_str", "dcode (district)", df_p,
    "hetero",
    "P experiment, HF-level", "none (strata FE only)")

# Table3-PanelB-ColII-P (infant death rate, P experiment)
print("Running baseline: Table3-PanelB-ColII-P-Infant...")
run_spec(
    "baseline__table3_panelB_colII_P_infant",
    "designs/randomized_experiment.md#baseline", "G1_mortality",
    "death_inf_year", "treatment", [],
    "dcode_str", "dcode (district)", df_p,
    "hetero",
    "P experiment, HF-level", "none (strata FE only)")


# --- Design variants ---

# Diff-in-means (no FE)
print("\nRunning design variant: diff-in-means (no FE)...")
run_spec(
    "design/randomized_experiment/estimator/diff_in_means",
    "designs/randomized_experiment.md#diff-in-means", "G1_mortality",
    "death_year", "treatment", [],
    "", "none (pooled diff-in-means)", df_ip,
    "hetero",
    "I&P experiment, HF-level, no FE", "none")

# Diff-in-means for P experiment
run_spec(
    "design/randomized_experiment/estimator/diff_in_means_P",
    "designs/randomized_experiment.md#diff-in-means", "G1_mortality",
    "death_year", "treatment", [],
    "", "none (pooled diff-in-means)", df_p,
    "hetero",
    "P experiment, HF-level, no FE", "none")


# --- RC: Controls variants ---

BASELINE_CONTROLS = ['avg_charge_gentreat', 'avgOP_baseline', 'hhs']
BASELINE_CONTROLS_SQ = ['avg_charge_gentreat', 'avgOP_baseline', 'hhs',
                         'charge2', 'op2', 'hhs2']

# Single control additions
print("\nRunning control addition variants...")
for ctrl_var in BASELINE_CONTROLS:
    run_spec(
        f"rc/controls/single/add_{ctrl_var}",
        "modules/robustness/controls.md#single-control-addition", "G1_mortality",
        "death_year", "treatment", [ctrl_var],
        "dcode_str", "dcode (district)", df_ip,
        "hetero",
        "I&P experiment, HF-level", f"add {ctrl_var}")

# Baseline controls set (3 vars)
run_spec(
    "rc/controls/sets/baseline_controls",
    "modules/robustness/controls.md#standard-control-sets", "G1_mortality",
    "death_year", "treatment", BASELINE_CONTROLS,
    "dcode_str", "dcode (district)", df_ip,
    "hetero",
    "I&P experiment, HF-level", f"baseline controls ({len(BASELINE_CONTROLS)} vars)")

# Baseline controls with squares (6 vars)
run_spec(
    "rc/controls/sets/baseline_controls_with_squares",
    "modules/robustness/controls.md#standard-control-sets", "G1_mortality",
    "death_year", "treatment", BASELINE_CONTROLS_SQ,
    "dcode_str", "dcode (district)", df_ip,
    "hetero",
    "I&P experiment, HF-level", f"baseline controls + squares ({len(BASELINE_CONTROLS_SQ)} vars)")


# --- RC: Sample restrictions ---

print("\nRunning sample restriction variants...")

# I&P experiment only (already the baseline for G1)
# P experiment only
run_spec(
    "rc/sample/restriction/p_experiment_only",
    "modules/robustness/sample.md#sample-restriction", "G1_mortality",
    "death_year", "treatment", [],
    "dcode_str", "dcode (district)", df_p,
    "hetero",
    "P experiment only", "none (strata FE only)")

# Post-2007 restriction for I&P experiment
# Already covered by P experiment which starts 2007
# For I&P, restrict to births after 2007
ip_post07 = build_mortality_panel(all_children, 1, 1285, 2.33, "I&P (2007-2009)")
ip_post07 = ip_post07.merge(baseline_ctrl, on='hfcode', how='left')
run_spec(
    "rc/sample/time/post_2007",
    "modules/robustness/sample.md#time-restriction", "G1_mortality",
    "death_year", "treatment", [],
    "dcode_str", "dcode (district)", ip_post07,
    "hetero",
    "I&P experiment, post-2007 only", "none (strata FE only)")

# Trim death rate at 1st/99th percentile
q01 = df_ip['death_year'].quantile(0.01)
q99 = df_ip['death_year'].quantile(0.99)
df_ip_trim = df_ip[(df_ip['death_year'] >= q01) & (df_ip['death_year'] <= q99)].copy()
run_spec(
    "rc/sample/outliers/trim_death_rate_1_99",
    "modules/robustness/sample.md#outliers", "G1_mortality",
    "death_year", "treatment", [],
    "dcode_str", "dcode (district)", df_ip_trim,
    "hetero",
    f"I&P experiment, trim death_year [1%,99%], N={len(df_ip_trim)}", "none (strata FE only)",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_death_rate_1_99",
                "n_obs_before": len(df_ip), "n_obs_after": len(df_ip_trim)})


# --- RC: Outcome form variants ---

print("\nRunning outcome form variants...")

# Neonatal mortality ratio
run_spec(
    "rc/form/outcome/neonatal_mortality_ratio",
    "modules/robustness/functional_form.md#outcome-transform", "G1_mortality",
    "neo_mr", "treatment", [],
    "dcode_str", "dcode (district)", df_ip.dropna(subset=['neo_mr']),
    "hetero",
    "I&P experiment, neonatal mortality ratio", "none (strata FE only)")

# Neonatal mortality ratio for P experiment
run_spec(
    "rc/form/outcome/neonatal_mortality_ratio_P",
    "modules/robustness/functional_form.md#outcome-transform", "G1_mortality",
    "neo_mr", "treatment", [],
    "dcode_str", "dcode (district)", df_p.dropna(subset=['neo_mr']),
    "hetero",
    "P experiment, neonatal mortality ratio", "none (strata FE only)")

# Infant death rate for P experiment
run_spec(
    "rc/form/outcome/infant_death_rate_P",
    "modules/robustness/functional_form.md#outcome-transform", "G1_mortality",
    "death_inf_year", "treatment", [],
    "dcode_str", "dcode (district)", df_p,
    "hetero",
    "P experiment, infant death rate", "none (strata FE only)")

# Exposure-corrected for P experiment
run_spec(
    "rc/form/outcome/exposure_corrected_P",
    "modules/robustness/functional_form.md#outcome-transform", "G1_mortality",
    "death_exp", "treatment", [],
    "dcode_str", "dcode (district)", df_p,
    "hetero",
    "P experiment, exposure-corrected", "none (strata FE only)")

# Neonatal death rate (crude)
run_spec(
    "rc/form/outcome/neonatal_death_rate_crude",
    "modules/robustness/functional_form.md#outcome-transform", "G1_mortality",
    "death_neo_year", "treatment", [],
    "dcode_str", "dcode (district)", df_ip,
    "hetero",
    "I&P experiment, neonatal death rate (crude)", "none (strata FE only)")


# --- RC: Controls + outcome combinations ---

print("\nRunning controls + outcome combinations...")

# Infant death + baseline controls
run_spec(
    "rc/controls/sets/baseline_controls_infant",
    "modules/robustness/controls.md#combined", "G1_mortality",
    "death_inf_year", "treatment", BASELINE_CONTROLS,
    "dcode_str", "dcode (district)", df_ip,
    "hetero",
    "I&P experiment, infant death rate + controls", f"baseline controls ({len(BASELINE_CONTROLS)} vars)")

# Exposure-corrected + baseline controls
run_spec(
    "rc/controls/sets/baseline_controls_exposure",
    "modules/robustness/controls.md#combined", "G1_mortality",
    "death_exp", "treatment", BASELINE_CONTROLS,
    "dcode_str", "dcode (district)", df_ip,
    "hetero",
    "I&P experiment, exposure-corrected + controls", f"baseline controls ({len(BASELINE_CONTROLS)} vars)")

# Crude death + baseline controls with squares
run_spec(
    "rc/controls/sets/baseline_sq_crude",
    "modules/robustness/controls.md#combined", "G1_mortality",
    "death_year", "treatment", BASELINE_CONTROLS_SQ,
    "dcode_str", "dcode (district)", df_ip,
    "hetero",
    "I&P experiment, crude death + full controls", f"baseline controls + squares ({len(BASELINE_CONTROLS_SQ)} vars)")

# NOTE: P experiment HF codes (3001-3030) do not appear in the baseline controls
# dataset (which only covers I&P HF codes). Therefore P + controls specs are not
# possible and are omitted.


# --- RC: Control subsets (exhaustive enumeration for 3 controls) ---

print("\nRunning control subset variants...")

import itertools
for r in range(1, len(BASELINE_CONTROLS) + 1):
    for combo in itertools.combinations(BASELINE_CONTROLS, r):
        ctrl_list = list(combo)
        ctrl_name = "+".join(ctrl_list)
        run_spec(
            f"rc/controls/subset/{ctrl_name}",
            "modules/robustness/controls.md#subset-generation", "G1_mortality",
            "death_year", "treatment", ctrl_list,
            "dcode_str", "dcode (district)", df_ip,
            "hetero",
            "I&P experiment", f"subset: {ctrl_name}")


# ============================================================
# G2: WEIGHT SPECIFICATIONS
# ============================================================

print("\n" + "="*60)
print("G2: WEIGHT SPECIFICATIONS")
print("="*60)

# --- Baseline: Table5-ColI-IP-Infants ---
print("\nRunning G2 baseline: Table5-ColI-IP-Infants (zw1, I&P infants)...")
base_run_id_g2, base_coef_g2, base_se_g2, base_pval_g2, base_n_g2 = run_spec(
    "baseline__table5_colI_IP_infants",
    "designs/randomized_experiment.md#baseline", "G2_weight",
    "zw1", "treatment", [],
    "dcode_str", "dcode (district)", wh_ip_infants,
    {"CRV1": "hfcode_str"},
    f"I&P experiment, infants 0-12mo, N={len(wh_ip_infants)}", "none (strata FE only)",
    design_audit_ref=design_audit_weight)

print(f"  Baseline G2: coef={base_coef_g2:.4f}, se={base_se_g2:.4f}, p={base_pval_g2:.4f}, N={base_n_g2}")


# --- Additional baselines ---

# Table5-ColII-IP-Children (13-59 months)
print("Running baseline: Table5-ColII-IP-Children...")
run_spec(
    "baseline__table5_colII_children",
    "designs/randomized_experiment.md#baseline", "G2_weight",
    "zw1", "treatment", [],
    "dcode_str", "dcode (district)", wh_ip_children,
    {"CRV1": "hfcode_str"},
    "I&P experiment, children 13-59mo", "none (strata FE only)",
    design_audit_ref=design_audit_weight)

# Table5-ColI-P-Infants (P experiment)
print("Running baseline: Table5-ColI-P-Infants...")
run_spec(
    "baseline__table5_colI_P_infants",
    "designs/randomized_experiment.md#baseline", "G2_weight",
    "zw1", "treatment", [],
    "dcode_str", "dcode (district)", wh_p_infants,
    {"CRV1": "hfcode_str"},
    "P experiment, infants 0-12mo", "none (strata FE only)",
    design_audit_ref=design_audit_weight)

# Table5-ColII-P-Children
print("Running baseline: Table5-ColII-P-Children...")
run_spec(
    "baseline__table5_colII_P_children",
    "designs/randomized_experiment.md#baseline", "G2_weight",
    "zw1", "treatment", [],
    "dcode_str", "dcode (district)", wh_p_children,
    {"CRV1": "hfcode_str"},
    "P experiment, children 13-59mo", "none (strata FE only)",
    design_audit_ref=design_audit_weight)


# --- Design variant: diff-in-means ---
print("\nRunning G2 design variant: diff-in-means...")
run_spec(
    "design/randomized_experiment/estimator/diff_in_means_weight",
    "designs/randomized_experiment.md#diff-in-means", "G2_weight",
    "zw1", "treatment", [],
    "", "none (diff-in-means)", wh_ip_infants,
    {"CRV1": "hfcode_str"},
    "I&P infants, no FE", "none",
    design_audit_ref=design_audit_weight)


# --- RC: Sample restrictions for weight ---

print("\nRunning G2 sample restriction variants...")

# Tighter z-score trimming (-3, 3)
wh_ip_infants_trim3 = wh[(wh['s821a'] < 13) & (wh['zw1'] > -3) & (wh['zw1'] < 3)
                          & (wh['s821b'] < wh['cort97']) & (wh['sample1'] == 1)].copy()
run_spec(
    "rc/sample/outliers/trim_zscore_3_infants",
    "modules/robustness/sample.md#outliers", "G2_weight",
    "zw1", "treatment", [],
    "dcode_str", "dcode (district)", wh_ip_infants_trim3,
    {"CRV1": "hfcode_str"},
    f"I&P infants, trim z in (-3,3), N={len(wh_ip_infants_trim3)}", "none",
    design_audit_ref=design_audit_weight)

wh_ip_children_trim3 = wh[(wh['s821a'] > 12) & (wh['zw1'] > -3) & (wh['zw1'] < 3)
                           & (wh['sample1'] == 1)].copy()
run_spec(
    "rc/sample/outliers/trim_zscore_3_children",
    "modules/robustness/sample.md#outliers", "G2_weight",
    "zw1", "treatment", [],
    "dcode_str", "dcode (district)", wh_ip_children_trim3,
    {"CRV1": "hfcode_str"},
    f"I&P children, trim z in (-3,3), N={len(wh_ip_children_trim3)}", "none",
    design_audit_ref=design_audit_weight)

# NOTE: cort97 weight cutoffs are only defined for months 0-12.
# For older children (>12mo), the do-file does NOT apply the cort97 filter.
# Instead, as a robustness check, we can apply a percentile-based weight filter.
q01_w = wh_ip_children['s821b'].quantile(0.01)
q99_w = wh_ip_children['s821b'].quantile(0.99)
wh_ip_children_trimw = wh_ip_children[(wh_ip_children['s821b'] >= q01_w) &
                                       (wh_ip_children['s821b'] <= q99_w)].copy()
run_spec(
    "rc/sample/outliers/trim_weight_1_99_children",
    "modules/robustness/sample.md#outliers", "G2_weight",
    "zw1", "treatment", [],
    "dcode_str", "dcode (district)", wh_ip_children_trimw,
    {"CRV1": "hfcode_str"},
    f"I&P children, trim weight [1%,99%], N={len(wh_ip_children_trimw)}", "none",
    design_audit_ref=design_audit_weight)

# All children combined (infants + older)
wh_ip_all = wh[(wh['zw1'] > -4.5) & (wh['zw1'] < 4.5)
                & (wh['sample1'] == 1)].copy()
run_spec(
    "rc/sample/restriction/all_ages_IP",
    "modules/robustness/sample.md#sample-restriction", "G2_weight",
    "zw1", "treatment", [],
    "dcode_str", "dcode (district)", wh_ip_all,
    {"CRV1": "hfcode_str"},
    f"I&P experiment, all ages, N={len(wh_ip_all)}", "none (strata FE only)",
    design_audit_ref=design_audit_weight)

wh_p_all = wh[(wh['zw1'] > -4.5) & (wh['zw1'] < 4.5)
               & (wh['sample1'] == 0)].copy()
run_spec(
    "rc/sample/restriction/all_ages_P",
    "modules/robustness/sample.md#sample-restriction", "G2_weight",
    "zw1", "treatment", [],
    "dcode_str", "dcode (district)", wh_p_all,
    {"CRV1": "hfcode_str"},
    f"P experiment, all ages, N={len(wh_p_all)}", "none (strata FE only)",
    design_audit_ref=design_audit_weight)


# --- RC: Outcome form (height-for-age z-score) ---

print("\nRunning G2 outcome form variants (height-for-age)...")

# Height-for-age z-score for infants (I&P)
wh_ip_infants_h = wh[(wh['s821a'] < 13) & (wh['zh1'] > -4.5) & (wh['zh1'] < 4.5)
                      & (wh['s821e'] < wh['cort97_h']) & (wh['sample1'] == 1)].dropna(subset=['zh1']).copy()
run_spec(
    "rc/form/outcome/height_for_age_infants_IP",
    "modules/robustness/functional_form.md#outcome-transform", "G2_weight",
    "zh1", "treatment", [],
    "dcode_str", "dcode (district)", wh_ip_infants_h,
    {"CRV1": "hfcode_str"},
    f"I&P infants, height-for-age z, N={len(wh_ip_infants_h)}", "none",
    design_audit_ref=design_audit_weight)

# Height-for-age z-score for children (I&P)
wh_ip_children_h = wh[(wh['s821a'] > 12) & (wh['zh1'] > -4.5) & (wh['zh1'] < 4.5)
                       & (wh['s821e'] < wh['cort97_h']) & (wh['sample1'] == 1)].dropna(subset=['zh1']).copy()
run_spec(
    "rc/form/outcome/height_for_age_children_IP",
    "modules/robustness/functional_form.md#outcome-transform", "G2_weight",
    "zh1", "treatment", [],
    "dcode_str", "dcode (district)", wh_ip_children_h,
    {"CRV1": "hfcode_str"},
    f"I&P children, height-for-age z, N={len(wh_ip_children_h)}", "none",
    design_audit_ref=design_audit_weight)

# Height-for-age, P experiment infants
wh_p_infants_h = wh[(wh['s821a'] < 13) & (wh['zh1'] > -4.5) & (wh['zh1'] < 4.5)
                     & (wh['s821e'] < wh['cort97_h']) & (wh['sample1'] == 0)].dropna(subset=['zh1']).copy()
run_spec(
    "rc/form/outcome/height_for_age_infants_P",
    "modules/robustness/functional_form.md#outcome-transform", "G2_weight",
    "zh1", "treatment", [],
    "dcode_str", "dcode (district)", wh_p_infants_h,
    {"CRV1": "hfcode_str"},
    f"P infants, height-for-age z, N={len(wh_p_infants_h)}", "none",
    design_audit_ref=design_audit_weight)

# Height-for-age, P experiment children
wh_p_children_h = wh[(wh['s821a'] > 12) & (wh['zh1'] > -4.5) & (wh['zh1'] < 4.5)
                      & (wh['s821e'] < wh['cort97_h']) & (wh['sample1'] == 0)].dropna(subset=['zh1']).copy()
run_spec(
    "rc/form/outcome/height_for_age_children_P",
    "modules/robustness/functional_form.md#outcome-transform", "G2_weight",
    "zh1", "treatment", [],
    "dcode_str", "dcode (district)", wh_p_children_h,
    {"CRV1": "hfcode_str"},
    f"P children, height-for-age z, N={len(wh_p_children_h)}", "none",
    design_audit_ref=design_audit_weight)


# --- RC: Controls for weight ---

print("\nRunning G2 control variants...")

# Add baseline facility controls to weight regressions
WEIGHT_CONTROLS = ['avg_charge_gentreat', 'avgOP_baseline', 'hhs',
                   'charge2', 'op2', 'hhs2']

run_spec(
    "rc/controls/sets/baseline_facility_infants_IP",
    "modules/robustness/controls.md#standard-control-sets", "G2_weight",
    "zw1", "treatment", WEIGHT_CONTROLS,
    "dcode_str", "dcode (district)", wh_ip_infants.dropna(subset=WEIGHT_CONTROLS),
    {"CRV1": "hfcode_str"},
    "I&P infants + facility controls", f"baseline facility controls ({len(WEIGHT_CONTROLS)} vars)",
    design_audit_ref=design_audit_weight)

run_spec(
    "rc/controls/sets/baseline_facility_children_IP",
    "modules/robustness/controls.md#standard-control-sets", "G2_weight",
    "zw1", "treatment", WEIGHT_CONTROLS,
    "dcode_str", "dcode (district)", wh_ip_children.dropna(subset=WEIGHT_CONTROLS),
    {"CRV1": "hfcode_str"},
    "I&P children + facility controls", f"baseline facility controls ({len(WEIGHT_CONTROLS)} vars)",
    design_audit_ref=design_audit_weight)

# NOTE: P experiment HF codes (3001-3030) do not appear in the baseline controls
# dataset. P + controls specs are not possible and are omitted.

# Partial controls: just the 3 linear baseline controls
WEIGHT_CONTROLS_LINEAR = ['avg_charge_gentreat', 'avgOP_baseline', 'hhs']

run_spec(
    "rc/controls/sets/linear_facility_infants_IP",
    "modules/robustness/controls.md#standard-control-sets", "G2_weight",
    "zw1", "treatment", WEIGHT_CONTROLS_LINEAR,
    "dcode_str", "dcode (district)", wh_ip_infants.dropna(subset=WEIGHT_CONTROLS_LINEAR),
    {"CRV1": "hfcode_str"},
    "I&P infants + linear facility controls", f"linear controls ({len(WEIGHT_CONTROLS_LINEAR)} vars)",
    design_audit_ref=design_audit_weight)

run_spec(
    "rc/controls/sets/linear_facility_children_IP",
    "modules/robustness/controls.md#standard-control-sets", "G2_weight",
    "zw1", "treatment", WEIGHT_CONTROLS_LINEAR,
    "dcode_str", "dcode (district)", wh_ip_children.dropna(subset=WEIGHT_CONTROLS_LINEAR),
    {"CRV1": "hfcode_str"},
    "I&P children + linear facility controls", f"linear controls ({len(WEIGHT_CONTROLS_LINEAR)} vars)",
    design_audit_ref=design_audit_weight)


# ============================================================
# INFERENCE VARIANTS
# ============================================================

print("\n" + "="*60)
print("INFERENCE VARIANTS")
print("="*60)

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
            design={"randomized_experiment": design_audit_mort},
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


# G1 inference variants (mortality, baseline is robust SEs)
baseline_formula_g1 = "death_year ~ treatment"

# Cluster at hfcode (redundant for cluster-level data, but a check)
run_inference_variant(
    base_run_id_g1, "infer/se/cluster/hfcode",
    "modules/inference/standard_errors.md#clustering", "G1_mortality",
    baseline_formula_g1, "dcode_str", df_ip, "treatment",
    {"CRV1": "hfcode_str"}, "cluster(hfcode)")

# Cluster at dcode (district)
run_inference_variant(
    base_run_id_g1, "infer/se/cluster/dcode",
    "modules/inference/standard_errors.md#clustering", "G1_mortality",
    baseline_formula_g1, "dcode_str", df_ip, "treatment",
    {"CRV1": "dcode_str"}, "cluster(dcode)")


# G2 inference variants (weight, baseline is cluster(hfcode))
baseline_formula_g2 = "zw1 ~ treatment"

# HC1 robust (no clustering)
run_inference_variant(
    base_run_id_g2, "infer/se/hc/hc1",
    "modules/inference/standard_errors.md#robust", "G2_weight",
    baseline_formula_g2, "dcode_str", wh_ip_infants, "treatment",
    "hetero", "HC1 (robust, no clustering)")

# Cluster at dcode
run_inference_variant(
    base_run_id_g2, "infer/se/cluster/dcode_weight",
    "modules/inference/standard_errors.md#clustering", "G2_weight",
    baseline_formula_g2, "dcode_str", wh_ip_infants, "treatment",
    {"CRV1": "dcode_str"}, "cluster(dcode)")


# ============================================================
# WRITE OUTPUTS
# ============================================================

print(f"\n{'='*60}")
print("Writing outputs...")
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
    g1_base = spec_df[spec_df['spec_id'] == 'baseline__table3_panelA_colI_IP']
    if len(g1_base) > 0:
        bc = g1_base.iloc[0]
        print(f"\nG1 Baseline (death_year, I&P):")
        print(f"  coef={bc['coefficient']:.6f}, se={bc['std_error']:.6f}, p={bc['p_value']:.6f}, N={bc['n_obs']:.0f}")

    # G2 baseline
    g2_base = spec_df[spec_df['spec_id'] == 'baseline__table5_colI_IP_infants']
    if len(g2_base) > 0:
        bc2 = g2_base.iloc[0]
        print(f"\nG2 Baseline (zw1, I&P infants):")
        print(f"  coef={bc2['coefficient']:.6f}, se={bc2['std_error']:.6f}, p={bc2['p_value']:.6f}, N={bc2['n_obs']:.0f}")

    print(f"\n=== COEFFICIENT RANGE (all successful specs) ===")
    print(f"Min coef: {successful['coefficient'].min():.6f}")
    print(f"Max coef: {successful['coefficient'].max():.6f}")
    print(f"Median coef: {successful['coefficient'].median():.6f}")

    # Breakdown by group
    g1_specs = successful[successful['baseline_group_id'] == 'G1_mortality']
    g2_specs = successful[successful['baseline_group_id'] == 'G2_weight']

    if len(g1_specs) > 0:
        n_sig_g1 = (g1_specs['p_value'] < 0.05).sum()
        print(f"\nG1 (mortality): {len(g1_specs)} specs, {n_sig_g1} sig at 5%")
        print(f"  Coef range: [{g1_specs['coefficient'].min():.4f}, {g1_specs['coefficient'].max():.4f}]")

    if len(g2_specs) > 0:
        n_sig_g2 = (g2_specs['p_value'] < 0.05).sum()
        print(f"\nG2 (weight): {len(g2_specs)} specs, {n_sig_g2} sig at 5%")
        print(f"  Coef range: [{g2_specs['coefficient'].min():.4f}, {g2_specs['coefficient'].max():.4f}]")


# ============================================================
# SPECIFICATION_SEARCH.md
# ============================================================

print("\nWriting SPECIFICATION_SEARCH.md...")

md_lines = []
md_lines.append("# Specification Search Report: 113630-V1")
md_lines.append("")
md_lines.append("**Paper:** Bjorkman Martina & Svensson (2014), \"Experimental Evidence on the Long-Run Impact of Community-Based Monitoring\", AEJ: Applied Economics")
md_lines.append("")
md_lines.append("## Study Design")
md_lines.append("")
md_lines.append("- **Design:** Cluster-randomized controlled trial (2 experiments)")
md_lines.append("- **Randomization unit:** Health facility (hfcode)")
md_lines.append("- **Strata:** District (dcode)")
md_lines.append("- **Experiment 1 (I&P):** Information & Participation, sample1==1, 2004-2009")
md_lines.append("- **Experiment 2 (P):** Participation only, sample1==0, 2006-2009")
md_lines.append("")

md_lines.append("## G1: Child Mortality")
md_lines.append("")
md_lines.append("### Baseline (Table 3, Panel A, Col I)")
md_lines.append("")
md_lines.append("- **Outcome:** Crude under-5 death rate (annualized)")
md_lines.append("- **Treatment:** Community monitoring (I&P)")
md_lines.append("- **FE:** District (dcode)")
md_lines.append("- **SEs:** Robust (HC1) — data at cluster level")
md_lines.append("")

if len(g1_base) > 0:
    bc = g1_base.iloc[0]
    md_lines.append(f"| Statistic | Value |")
    md_lines.append(f"|-----------|-------|")
    md_lines.append(f"| Coefficient | {bc['coefficient']:.6f} |")
    md_lines.append(f"| Std. Error | {bc['std_error']:.6f} |")
    md_lines.append(f"| p-value | {bc['p_value']:.6f} |")
    md_lines.append(f"| 95% CI | [{bc['ci_lower']:.6f}, {bc['ci_upper']:.6f}] |")
    md_lines.append(f"| N | {bc['n_obs']:.0f} |")
    md_lines.append(f"| R-squared | {bc['r_squared']:.4f} |")
    md_lines.append("")

md_lines.append("## G2: Weight-for-Age Z-Score")
md_lines.append("")
md_lines.append("### Baseline (Table 5, Col I)")
md_lines.append("")
md_lines.append("- **Outcome:** Weight-for-age z-score (zw1)")
md_lines.append("- **Treatment:** Community monitoring (I&P)")
md_lines.append("- **Sample:** Infants 0-12 months, z in (-4.5, 4.5), weight < cort97")
md_lines.append("- **FE:** District (dcode)")
md_lines.append("- **SEs:** Clustered at health facility")
md_lines.append("")

if len(g2_base) > 0:
    bc2 = g2_base.iloc[0]
    md_lines.append(f"| Statistic | Value |")
    md_lines.append(f"|-----------|-------|")
    md_lines.append(f"| Coefficient | {bc2['coefficient']:.6f} |")
    md_lines.append(f"| Std. Error | {bc2['std_error']:.6f} |")
    md_lines.append(f"| p-value | {bc2['p_value']:.6f} |")
    md_lines.append(f"| 95% CI | [{bc2['ci_lower']:.6f}, {bc2['ci_upper']:.6f}] |")
    md_lines.append(f"| N | {bc2['n_obs']:.0f} |")
    md_lines.append(f"| R-squared | {bc2['r_squared']:.4f} |")
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
md_lines.append("| Category | Group | Count | Sig. (p<0.05) | Coef Range |")
md_lines.append("|----------|-------|-------|---------------|------------|")

categories = {
    ("Baseline", "G1"): successful[(successful['spec_id'].str.startswith('baseline')) & (successful['baseline_group_id'] == 'G1_mortality')],
    ("Baseline", "G2"): successful[(successful['spec_id'].str.startswith('baseline')) & (successful['baseline_group_id'] == 'G2_weight')],
    ("Design variants", "G1"): successful[(successful['spec_id'].str.startswith('design/')) & (successful['baseline_group_id'] == 'G1_mortality')],
    ("Design variants", "G2"): successful[(successful['spec_id'].str.startswith('design/')) & (successful['baseline_group_id'] == 'G2_weight')],
    ("Controls", "G1"): successful[(successful['spec_id'].str.startswith('rc/controls/')) & (successful['baseline_group_id'] == 'G1_mortality')],
    ("Controls", "G2"): successful[(successful['spec_id'].str.startswith('rc/controls/')) & (successful['baseline_group_id'] == 'G2_weight')],
    ("Sample", "G1"): successful[(successful['spec_id'].str.startswith('rc/sample/')) & (successful['baseline_group_id'] == 'G1_mortality')],
    ("Sample", "G2"): successful[(successful['spec_id'].str.startswith('rc/sample/')) & (successful['baseline_group_id'] == 'G2_weight')],
    ("Outcome form", "G1"): successful[(successful['spec_id'].str.startswith('rc/form/')) & (successful['baseline_group_id'] == 'G1_mortality')],
    ("Outcome form", "G2"): successful[(successful['spec_id'].str.startswith('rc/form/')) & (successful['baseline_group_id'] == 'G2_weight')],
}

for (cat_name, group), cat_df in categories.items():
    if len(cat_df) > 0:
        n_sig_cat = (cat_df['p_value'] < 0.05).sum()
        coef_range = f"[{cat_df['coefficient'].min():.4f}, {cat_df['coefficient'].max():.4f}]"
        md_lines.append(f"| {cat_name} | {group} | {len(cat_df)} | {n_sig_cat}/{len(cat_df)} | {coef_range} |")

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

for group_name, group_id in [("G1 (Mortality)", "G1_mortality"), ("G2 (Weight)", "G2_weight")]:
    grp = successful[successful['baseline_group_id'] == group_id]
    if len(grp) > 0:
        n_sig_total = (grp['p_value'] < 0.05).sum()
        pct_sig = n_sig_total / len(grp) * 100
        sign_consistent = ((grp['coefficient'] > 0).sum() == len(grp)) or \
                          ((grp['coefficient'] < 0).sum() == len(grp))
        median_coef = grp['coefficient'].median()
        sign_word = "positive" if median_coef > 0 else "negative"

        if pct_sig >= 80 and sign_consistent:
            strength = "STRONG"
        elif pct_sig >= 50 and sign_consistent:
            strength = "MODERATE"
        elif pct_sig >= 30:
            strength = "WEAK"
        else:
            strength = "FRAGILE"

        md_lines.append(f"### {group_name}")
        md_lines.append(f"- **Sign consistency:** {'All specs same sign' if sign_consistent else 'Mixed signs'}")
        md_lines.append(f"- **Significance stability:** {n_sig_total}/{len(grp)} ({pct_sig:.1f}%) significant at 5%")
        md_lines.append(f"- **Direction:** Median coefficient is {sign_word} ({median_coef:.6f})")
        md_lines.append(f"- **Robustness assessment:** {strength}")
        md_lines.append("")

md_lines.append(f"Surface hash: `{SURFACE_HASH}`")
md_lines.append("")

with open(f"{OUTPUT_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write("\n".join(md_lines))

print(f"Wrote SPECIFICATION_SEARCH.md")
print("\nDone!")
