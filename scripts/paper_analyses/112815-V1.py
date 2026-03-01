"""
Specification Search Script for Hoxby (2014)
"The Economics of Online Postsecondary Education: MOOCs, Nonselective Education,
and Highly Selective Education"
American Economic Review Papers & Proceedings, 104(5), 528-533.

Paper ID: 112815-V1

DATA CONSTRAINT NOTE:
The paper's main regressions require restricted-access BPS 2004/2009 data (NCES license).
Only two supplementary public datasets are provided:
  - ipeds_asc_variables.dta (institution-level IPEDS/ASC financial and selectivity data)
  - bps200409_course_codes_titles_definitions.dta (course code lookups, no analysis vars)

The specification surface defines a baseline of incres09 ~ cert* + instid* (earnings on
certificates and institution indicators) but that data is unavailable.

APPROACH:
Since the paper's IPEDS-based analysis characterizes NSPE vs non-NSPE institutions
using financial variables (tuition revenue share, instructional expenditure, sticker price),
we construct regressions with nspe as treatment and institution financial characteristics
as outcomes. This matches the second part of the do-file analysis and the paper's
descriptive contribution.

The paper explicitly states "No causal interpretation" for all regressions.

Surface-driven execution:
  - G1: Institution financial characteristics ~ nspe indicator + controls
  - Cross-sectional OLS, institution-level, HC1 robust SEs
  - Specifications across: outcomes, controls, samples, functional form

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

PAPER_ID = "112815-V1"
DATA_DIR = "data/downloads/extracted/112815-V1"
OUTPUT_DIR = DATA_DIR
DATA_PATH = f"{DATA_DIR}/P2014_1147_data/ipeds_asc_variables.dta"

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

df_raw = pd.read_stata(DATA_PATH)
print(f"Loaded data: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

# Convert float32 to float64 for precision
for col in df_raw.columns:
    if df_raw[col].dtype == np.float32:
        df_raw[col] = df_raw[col].astype(np.float64)

# Replicate the do-file's variable construction
# satoractpct = sum of SAT and ACT percentiles
df_raw['medSAT'] = df_raw['medSATpct']
df_raw['satoractpct'] = df_raw['satpct'].fillna(0) + df_raw['actpct'].fillna(0)

# ---- NSPE definition (from do file) ----
# Carnegie codes 32, 33, 40 in Stata numeric coding correspond to:
# Associates Colleges, Baccalaureate/Associates Colleges, and Baccalaureate Colleges--General
# (The do file uses: carnegie==32 | carnegie==33 | carnegie==40)
carnegie_nspe_cats = [
    'Associates Colleges',
    'Baccalaureate/Associates Colleges',
    'Baccalaureate Colleges--General'
]

# ccugprof exclusion codes: -2, 8, 10, 11, 12, 13 in Stata mapping:
# -2 -> Not applicable, not in Carnegie universe
# 8 -> Higher part-time four-year
# 10 -> Medium full-time four-year, selective, lower transfer-in
# 11 -> Medium full-time four-year, selective, higher transfer-in
# 12 -> Full-time four-year, inclusive
# 13 -> Full-time four-year, selective, lower transfer-in
ccugprof_exclude = [
    'Not applicable, not in Carnegie universe (not accredited or nondegree-granting)',
    'Higher part-time four-year',
    'Medium full-time four-year, selective, lower transfer-in',
    'Medium full-time four-year, selective, higher transfer-in',
    'Full-time four-year, inclusive',
    'Full-time four-year, selective, lower transfer-in'
]

# Step 1: temp=1 if carnegie in NSPE categories
df_raw['temp'] = np.where(df_raw['carnegie'].isin(carnegie_nspe_cats), 1, 0)
# Step 2: replace temp=. if medSAT>=40 & medSAT~=.
df_raw.loc[(df_raw['medSAT'] >= 40) & (df_raw['medSAT'].notna()), 'temp'] = np.nan
# Step 3: replace temp=. if ccugprof in excluded categories
df_raw.loc[df_raw['ccugprof'].isin(ccugprof_exclude), 'temp'] = np.nan
# Step 4: nspe=1 if temp==1 & (medSAT<=25 | satoractpct<=25)
df_raw['nspe'] = 0
df_raw.loc[
    (df_raw['temp'] == 1) &
    ((df_raw['medSAT'] <= 25) | (df_raw['satoractpct'] <= 25)),
    'nspe'
] = 1

# ---- Construct financial variables (from do file, IPEDS section) ----
df_raw['tuitionrevshare'] = df_raw['tuition03'] / df_raw['tot_rev_wo_aux']
df_raw['tuitapprev'] = (df_raw['tuition03'].fillna(0) + df_raw['state03'].fillna(0) +
                         df_raw['local03'].fillna(0) + df_raw['federal03'].fillna(0))
df_raw['tuitapprevshare'] = df_raw['tuitapprev'] / df_raw['tot_rev_wo_aux']
df_raw['investrevshare'] = df_raw['investment01'] / df_raw['tot_rev_wo_aux']

# Instructional expenditure per FTE (core educational cost)
df_raw['instruct_per_fte'] = df_raw['instruction01'] / df_raw['fte_count']
# Total educational expenditure per FTE
df_raw['totalexp_per_fte'] = (
    (df_raw['instruction01'].fillna(0) + df_raw['acadsupp01'].fillna(0) +
     df_raw['studserv01'].fillna(0) + df_raw['instsupp01'].fillna(0))
    / df_raw['fte_count']
)
# Log versions
df_raw['log_instruct_per_fte'] = np.log(df_raw['instruct_per_fte'].clip(lower=1))
df_raw['log_totalexp_per_fte'] = np.log(df_raw['totalexp_per_fte'].clip(lower=1))
df_raw['log_fte'] = np.log(df_raw['fte_count'].clip(lower=1))
df_raw['log_tuition03'] = np.log(df_raw['tuition03'].clip(lower=1))

# Sector dummies for controls
# 1=pub4yr, 2=priv4yr, 3=profitpriv4yr, 4=pub2yr, 5=priv2yr, 6=profitpriv2yr,
# 7=pub<2yr, 8=priv<2yr, 9=profitpriv<2yr
df_raw['public'] = df_raw['sector'].isin([1, 4, 7]).astype(int)
df_raw['forprofit'] = df_raw['sector'].isin([3, 6, 9]).astype(int)
df_raw['fouryear'] = df_raw['sector'].isin([1, 2, 3]).astype(int)
df_raw['twoyear'] = df_raw['sector'].isin([4, 5, 6]).astype(int)

# Carnegie group dummies for controls
df_raw['is_associates'] = (df_raw['carnegie'] == 'Associates Colleges').astype(int)

# Sector as string for FE
df_raw['sector_str'] = df_raw['sector'].astype(str)
# Carnegie as string for FE
df_raw['carnegie_str'] = df_raw['carnegie'].astype(str)

print(f"NSPE institutions: {df_raw['nspe'].sum()}")
print(f"Non-NSPE institutions: {(df_raw['nspe'] == 0).sum()}")

# ---- Define analysis samples ----
# Full sample: all institutions with non-missing key variables
CORE_VARS = ['nspe', 'tuitionrevshare', 'fte_count', 'sector']
df = df_raw.dropna(subset=CORE_VARS).copy()
# Drop extreme outliers in tuitionrevshare (>2 or <0 are implausible shares)
df = df[(df['tuitionrevshare'] >= 0) & (df['tuitionrevshare'] <= 2)].copy()
print(f"Analysis sample (core vars non-missing, tuitionrevshare in [0,2]): {len(df)}")
print(f"  NSPE in sample: {df['nspe'].sum()}")

# Control variable lists
SECTOR_CONTROLS = ['public', 'forprofit']
SIZE_CONTROLS = ['log_fte']
FULL_CONTROLS = ['public', 'forprofit', 'fouryear', 'log_fte']

# ============================================================
# Accumulators
# ============================================================

results = []
inference_results = []
spec_run_counter = 0


# ============================================================
# Helper: run_spec (OLS via pyfixest)
# ============================================================

def run_spec(spec_id, spec_tree_path, baseline_group_id,
             outcome_var, treatment_var, controls, fe_formula_str,
             fe_desc, data, vcov, sample_desc, controls_desc,
             cluster_var=None,
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

        m = pf.feols(formula, data=data, vcov=vcov)

        coef_val = float(m.coef().get(treatment_var, np.nan))
        se_val = float(m.se().get(treatment_var, np.nan))
        pval = float(m.pvalue().get(treatment_var, np.nan))

        try:
            ci = m.confint()
            ci_lower = float(ci.loc[treatment_var, ci.columns[0]]) if treatment_var in ci.index else np.nan
            ci_upper = float(ci.loc[treatment_var, ci.columns[1]]) if treatment_var in ci.index else np.nan
        except Exception:
            ci_lower = np.nan
            ci_upper = np.nan

        nobs = int(m._N)
        try:
            r2 = float(m._r2)
        except Exception:
            r2 = np.nan

        all_coefs = {k: float(v) for k, v in m.coef().items()}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": inference_canonical["spec_id"],
                       "method": "HC1" if vcov == "hetero" else str(vcov)},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"cross_sectional_ols": design_audit},
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
            "cluster_var": cluster_var or "",
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
            "cluster_var": cluster_var or "",
            "run_success": 0,
            "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


# ============================================================
# BASELINE: tuitionrevshare ~ nspe + sector controls, HC1
# ============================================================
# The paper characterizes how NSPE institutions' revenue structure differs.
# Baseline: tuition revenue share regressed on NSPE indicator with sector controls.

print("\n=== Running baseline specification ===")
base_run_id, base_coef, base_se, base_pval, base_nobs = run_spec(
    "baseline__nspe_tuitionrevshare",
    "designs/cross_sectional_ols.md#baseline", "G1",
    "tuitionrevshare", "nspe", SECTOR_CONTROLS,
    "", "none", df,
    "hetero",
    f"IPEDS institutions with non-missing core vars, N={len(df)}",
    "sector controls (public, forprofit)",
    notes="NSPE effect on tuition revenue share, HC1 robust SEs")

print(f"  Baseline: coef={base_coef:.4f}, se={base_se:.4f}, p={base_pval:.4f}, N={base_nobs}")


# ============================================================
# ADDITIONAL BASELINES: Different outcomes from the paper
# ============================================================

print("\n=== Running additional baseline outcomes ===")

# Investment revenue share
run_spec(
    "baseline__nspe_investrevshare",
    "designs/cross_sectional_ols.md#baseline", "G1",
    "investrevshare", "nspe", SECTOR_CONTROLS,
    "", "none",
    df.dropna(subset=['investrevshare']),
    "hetero",
    "Institutions with non-missing investrevshare",
    "sector controls (public, forprofit)")

# Log instructional expenditure per FTE
run_spec(
    "baseline__nspe_log_instruct_per_fte",
    "designs/cross_sectional_ols.md#baseline", "G1",
    "log_instruct_per_fte", "nspe", SECTOR_CONTROLS,
    "", "none",
    df.dropna(subset=['log_instruct_per_fte']),
    "hetero",
    "Institutions with non-missing instructional exp",
    "sector controls (public, forprofit)")

# Log total educational expenditure per FTE
run_spec(
    "baseline__nspe_log_totalexp_per_fte",
    "designs/cross_sectional_ols.md#baseline", "G1",
    "log_totalexp_per_fte", "nspe", SECTOR_CONTROLS,
    "", "none",
    df.dropna(subset=['log_totalexp_per_fte']),
    "hetero",
    "Institutions with non-missing total exp",
    "sector controls (public, forprofit)")

# Sticker share (sticker price / core cost)
df_sticker = df.dropna(subset=['sticker_share']).copy()
df_sticker = df_sticker[(df_sticker['sticker_share'] > 0) & (df_sticker['sticker_share'] < 10)]
run_spec(
    "baseline__nspe_sticker_share",
    "designs/cross_sectional_ols.md#baseline", "G1",
    "sticker_share", "nspe", SECTOR_CONTROLS,
    "", "none",
    df_sticker,
    "hetero",
    "Institutions with sticker_share in (0,10)",
    "sector controls (public, forprofit)")

# Log FTE enrollment
run_spec(
    "baseline__nspe_log_fte",
    "designs/cross_sectional_ols.md#baseline", "G1",
    "log_fte", "nspe", SECTOR_CONTROLS,
    "", "none",
    df.dropna(subset=['log_fte']),
    "hetero",
    "Institutions with non-missing FTE",
    "sector controls (public, forprofit)")

# Log tuition revenue
run_spec(
    "baseline__nspe_log_tuition",
    "designs/cross_sectional_ols.md#baseline", "G1",
    "log_tuition03", "nspe", SECTOR_CONTROLS,
    "", "none",
    df.dropna(subset=['log_tuition03']),
    "hetero",
    "Institutions with non-missing tuition",
    "sector controls (public, forprofit)")


# ============================================================
# RC: CONTROLS VARIANTS — tuitionrevshare outcome
# ============================================================

print("\n=== Running controls variants (tuitionrevshare) ===")

# No controls (bivariate)
run_spec(
    "rc/controls/sets/none",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "tuitionrevshare", "nspe", [],
    "", "none", df,
    "hetero",
    "Full sample", "none (bivariate)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/none", "family": "sets",
                "n_controls": 0, "set_name": "none"})

# Size control only
run_spec(
    "rc/controls/sets/size_only",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "tuitionrevshare", "nspe", SIZE_CONTROLS,
    "", "none", df,
    "hetero",
    "Full sample", "size control (log_fte)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/size_only", "family": "sets",
                "n_controls": 1, "set_name": "size_only"})

# Full controls (sector + size)
run_spec(
    "rc/controls/sets/full",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "tuitionrevshare", "nspe", FULL_CONTROLS,
    "", "none", df,
    "hetero",
    "Full sample", "full controls (sector + fouryear + log_fte)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/full", "family": "sets",
                "n_controls": len(FULL_CONTROLS), "set_name": "full"})

# Sector controls only (baseline, for explicit listing)
run_spec(
    "rc/controls/sets/sector_only",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "tuitionrevshare", "nspe", SECTOR_CONTROLS,
    "", "none", df,
    "hetero",
    "Full sample", "sector controls (public, forprofit)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/sector_only", "family": "sets",
                "n_controls": len(SECTOR_CONTROLS), "set_name": "sector_only"})


# ============================================================
# RC: CONTROLS LOO — drop one control at a time from FULL_CONTROLS
# ============================================================

print("\n=== Running controls LOO variants ===")

for ctrl_to_drop in FULL_CONTROLS:
    ctrl_remaining = [c for c in FULL_CONTROLS if c != ctrl_to_drop]
    spec_id = f"rc/controls/loo/drop_{ctrl_to_drop}"
    run_spec(
        spec_id, "modules/robustness/controls.md#leave-one-out-controls-loo", "G1",
        "tuitionrevshare", "nspe", ctrl_remaining,
        "", "none", df,
        "hetero",
        "Full sample", f"full minus {ctrl_to_drop}",
        axis_block_name="controls",
        axis_block={"spec_id": spec_id, "family": "loo",
                    "dropped": [ctrl_to_drop], "n_controls": len(ctrl_remaining)})


# ============================================================
# RC: SAMPLE VARIANTS — institution type subsamples
# ============================================================

print("\n=== Running sample variants ===")

# NSPE only sample (within-NSPE variation — all nspe==1, so nspe drops)
# Instead, run the opposite: restrict to non-NSPE and check if excluding
# matters. Actually, the surface says split by institution type.
# For the surface-defined specs:

# rc/sample/institution_type/nspe_only — all NSPE institutions
# Since all nspe==1, treatment var has no variation. Instead, we use
# the NSPE sample for financial outcome regressions with different treatment.
# Use 'public' as treatment within NSPE sample.
df_nspe = df[df['nspe'] == 1].copy()
if len(df_nspe) > 30:
    run_spec(
        "rc/sample/institution_type/nspe_only",
        "modules/robustness/sample.md#subgroup-analyses", "G1",
        "tuitionrevshare", "public", ['forprofit', 'log_fte'],
        "", "none", df_nspe,
        "hetero",
        f"NSPE institutions only, N={len(df_nspe)}",
        "forprofit + log_fte",
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/institution_type/nspe_only",
                    "axis": "institution_type",
                    "sample": "nspe_only", "treatment": "public",
                    "notes": "Within NSPE: effect of public vs private on tuition rev share"})

# HSPE only — need to define HSPE
# HSPE: barrons08==1 & medSAT>=95 — we don't have barrons08, so approximate
# using medSAT>=95 only
df_hspe = df[df['medSAT'] >= 95].copy()
if len(df_hspe) > 10:
    run_spec(
        "rc/sample/institution_type/hspe_only",
        "modules/robustness/sample.md#subgroup-analyses", "G1",
        "tuitionrevshare", "public", ['log_fte'],
        "", "none", df_hspe,
        "hetero",
        f"HSPE-approx institutions (medSAT>=95), N={len(df_hspe)}",
        "log_fte",
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/institution_type/hspe_only",
                    "axis": "institution_type",
                    "sample": "hspe_approx_medSAT_ge95",
                    "notes": "Approximate HSPE (medSAT>=95, no Barrons data)"})

# Public institutions only
df_pub = df[df['public'] == 1].copy()
if len(df_pub) > 30:
    run_spec(
        "rc/sample/sector/public_only",
        "modules/robustness/sample.md#subgroup-analyses", "G1",
        "tuitionrevshare", "nspe", ['log_fte'],
        "", "none", df_pub,
        "hetero",
        f"Public institutions only, N={len(df_pub)}",
        "log_fte",
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/sector/public_only",
                    "axis": "sector", "sample": "public_only"})

# Private not-for-profit only
df_privnfp = df[df['sector'].isin([2, 5, 8])].copy()
if len(df_privnfp) > 30:
    run_spec(
        "rc/sample/sector/private_nfp_only",
        "modules/robustness/sample.md#subgroup-analyses", "G1",
        "tuitionrevshare", "nspe", ['fouryear', 'log_fte'],
        "", "none", df_privnfp,
        "hetero",
        f"Private not-for-profit only, N={len(df_privnfp)}",
        "fouryear + log_fte",
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/sector/private_nfp_only",
                    "axis": "sector", "sample": "private_nfp"})

# For-profit only
df_fp = df[df['forprofit'] == 1].copy()
if len(df_fp) > 30:
    run_spec(
        "rc/sample/sector/forprofit_only",
        "modules/robustness/sample.md#subgroup-analyses", "G1",
        "tuitionrevshare", "nspe", ['log_fte'],
        "", "none", df_fp,
        "hetero",
        f"For-profit institutions only, N={len(df_fp)}",
        "log_fte",
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/sector/forprofit_only",
                    "axis": "sector", "sample": "forprofit_only"})

# Four-year institutions only
df_4yr = df[df['fouryear'] == 1].copy()
if len(df_4yr) > 30:
    run_spec(
        "rc/sample/level/fouryear_only",
        "modules/robustness/sample.md#subgroup-analyses", "G1",
        "tuitionrevshare", "nspe", SECTOR_CONTROLS + ['log_fte'],
        "", "none", df_4yr,
        "hetero",
        f"Four-year institutions only, N={len(df_4yr)}",
        "sector controls + log_fte",
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/level/fouryear_only",
                    "axis": "institution_level", "sample": "fouryear_only"})

# Two-year institutions only
df_2yr = df[df['twoyear'] == 1].copy()
if len(df_2yr) > 30:
    run_spec(
        "rc/sample/level/twoyear_only",
        "modules/robustness/sample.md#subgroup-analyses", "G1",
        "tuitionrevshare", "nspe", SECTOR_CONTROLS + ['log_fte'],
        "", "none", df_2yr,
        "hetero",
        f"Two-year institutions only, N={len(df_2yr)}",
        "sector controls + log_fte",
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/level/twoyear_only",
                    "axis": "institution_level", "sample": "twoyear_only"})


# ============================================================
# RC: SAMPLE TRIMMING
# ============================================================

print("\n=== Running sample trimming variants ===")

# Trim tuitionrevshare at 1st/99th percentile
q01 = df['tuitionrevshare'].quantile(0.01)
q99 = df['tuitionrevshare'].quantile(0.99)
df_trim1 = df[(df['tuitionrevshare'] >= q01) & (df['tuitionrevshare'] <= q99)].copy()
n_before = len(df)

run_spec(
    "rc/sample/outliers/trim_y_1_99",
    "modules/robustness/sample.md#outliers-and-influential-observations", "G1",
    "tuitionrevshare", "nspe", SECTOR_CONTROLS,
    "", "none", df_trim1,
    "hetero",
    f"Trim tuitionrevshare [1%,99%], N={len(df_trim1)}", "sector controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_1_99", "axis": "outliers",
                "rule": "trim", "params": {"var": "tuitionrevshare", "lower_q": 0.01, "upper_q": 0.99},
                "n_obs_before": n_before, "n_obs_after": len(df_trim1)})

# Trim at 5th/95th percentile
q05 = df['tuitionrevshare'].quantile(0.05)
q95 = df['tuitionrevshare'].quantile(0.95)
df_trim5 = df[(df['tuitionrevshare'] >= q05) & (df['tuitionrevshare'] <= q95)].copy()

run_spec(
    "rc/sample/outliers/trim_y_5_95",
    "modules/robustness/sample.md#outliers-and-influential-observations", "G1",
    "tuitionrevshare", "nspe", SECTOR_CONTROLS,
    "", "none", df_trim5,
    "hetero",
    f"Trim tuitionrevshare [5%,95%], N={len(df_trim5)}", "sector controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_5_95", "axis": "outliers",
                "rule": "trim", "params": {"var": "tuitionrevshare", "lower_q": 0.05, "upper_q": 0.95},
                "n_obs_before": n_before, "n_obs_after": len(df_trim5)})

# Trim by FTE size (drop very small institutions)
df_fte_min = df[df['fte_count'] >= 100].copy()
run_spec(
    "rc/sample/outliers/fte_ge_100",
    "modules/robustness/sample.md#outliers-and-influential-observations", "G1",
    "tuitionrevshare", "nspe", SECTOR_CONTROLS,
    "", "none", df_fte_min,
    "hetero",
    f"FTE >= 100, N={len(df_fte_min)}", "sector controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/fte_ge_100", "axis": "outliers",
                "rule": "min_threshold", "params": {"var": "fte_count", "min": 100},
                "n_obs_before": n_before, "n_obs_after": len(df_fte_min)})

# Drop very large institutions
df_fte_max = df[df['fte_count'] <= 50000].copy()
run_spec(
    "rc/sample/outliers/fte_le_50k",
    "modules/robustness/sample.md#outliers-and-influential-observations", "G1",
    "tuitionrevshare", "nspe", SECTOR_CONTROLS,
    "", "none", df_fte_max,
    "hetero",
    f"FTE <= 50000, N={len(df_fte_max)}", "sector controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/fte_le_50k", "axis": "outliers",
                "rule": "max_threshold", "params": {"var": "fte_count", "max": 50000},
                "n_obs_before": n_before, "n_obs_after": len(df_fte_max)})


# ============================================================
# RC: CONTROLS from surface (cert_only, instid_only, no_institution_fe)
# Surface-defined specs adapted to available data
# ============================================================

print("\n=== Running surface-defined RC specs ===")

# rc/controls/sets/no_institution_fe — run without any FE or institution controls
# (Already covered by bivariate, but include for surface compliance)
run_spec(
    "rc/controls/sets/no_institution_fe",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "tuitionrevshare", "nspe", [],
    "", "none", df,
    "hetero",
    "Full sample", "no institution controls (bivariate, per surface)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/no_institution_fe", "family": "sets",
                "n_controls": 0,
                "notes": "Surface-defined: no institution FE"})

# rc/controls/sets/cert_only — certificate indicators only (unavailable)
# Adapted: nspe indicator only (no institution-level controls)
# This is effectively the bivariate specification; include for traceability
run_spec(
    "rc/controls/sets/cert_only_adapted",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "tuitionrevshare", "nspe", [],
    "", "none", df,
    "hetero",
    "Full sample", "nspe only (adapted from cert_only, no controls)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/cert_only_adapted", "family": "sets",
                "n_controls": 0,
                "notes": "Adapted: surface cert_only -> nspe only"})

# rc/controls/sets/instid_only — institution indicators only (unavailable as treatment)
# Adapted: use sector FE as proxy for institution grouping
run_spec(
    "rc/controls/sets/instid_only_adapted",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "tuitionrevshare", "nspe", [],
    "sector_str", "sector FE", df,
    "hetero",
    "Full sample", "sector FE (adapted from instid_only)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/instid_only_adapted", "family": "sets",
                "n_controls": 0,
                "notes": "Adapted: surface instid_only -> sector FE"})


# ============================================================
# RC: FIXED EFFECTS variants
# ============================================================

print("\n=== Running FE variants ===")

# Sector FE
run_spec(
    "rc/fe/add/sector",
    "modules/robustness/fixed_effects.md#additive-fe-variations-relative-to-baseline", "G1",
    "tuitionrevshare", "nspe", ['log_fte'],
    "sector_str", "sector FE", df,
    "hetero",
    "Full sample", "log_fte + sector FE",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/add/sector", "family": "add",
                "added": ["sector"], "baseline_fe": [], "new_fe": ["sector"]})

# Carnegie FE (coarser institutional classification)
run_spec(
    "rc/fe/add/carnegie",
    "modules/robustness/fixed_effects.md#additive-fe-variations-relative-to-baseline", "G1",
    "tuitionrevshare", "nspe", ['log_fte'],
    "carnegie_str", "carnegie FE", df,
    "hetero",
    "Full sample", "log_fte + carnegie FE",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/add/carnegie", "family": "add",
                "added": ["carnegie"], "baseline_fe": [], "new_fe": ["carnegie"]})

# Sector + Carnegie interaction FE
# Note: nspe is defined from carnegie, so this may absorb the treatment
# Let's try sector FE only with full controls
run_spec(
    "rc/fe/add/sector_full_ctrl",
    "modules/robustness/fixed_effects.md#additive-fe-variations-relative-to-baseline", "G1",
    "tuitionrevshare", "nspe", FULL_CONTROLS,
    "sector_str", "sector FE", df,
    "hetero",
    "Full sample", "full controls + sector FE",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/add/sector_full_ctrl", "family": "add",
                "added": ["sector"], "baseline_fe": [], "new_fe": ["sector"],
                "notes": "Sector FE absorbs public/forprofit/fouryear dummies partially"})


# ============================================================
# RC: OUTCOME VARIANTS with full controls
# Each outcome × control combination is a distinct specification
# ============================================================

print("\n=== Running outcome variants with different control sets ===")

OUTCOME_VARS = {
    "investrevshare": ("investrevshare", "Investment revenue share"),
    "log_instruct_per_fte": ("log_instruct_per_fte", "Log instructional exp per FTE"),
    "log_totalexp_per_fte": ("log_totalexp_per_fte", "Log total educational exp per FTE"),
    "sticker_share": ("sticker_share", "Sticker price / core cost"),
    "log_fte": ("log_fte", "Log FTE enrollment"),
    "log_tuition": ("log_tuition03", "Log tuition revenue"),
}

for outcome_key, (outcome_var, outcome_desc) in OUTCOME_VARS.items():
    # Full controls version
    if outcome_var == "sticker_share":
        sample = df_sticker.copy()
        sample_desc = "Institutions with sticker_share in (0,10)"
    else:
        sample = df.dropna(subset=[outcome_var]).copy()
        sample_desc = f"Institutions with non-missing {outcome_var}"

    # With full controls
    run_spec(
        f"rc/outcome/{outcome_key}/full_controls",
        "modules/robustness/controls.md#standard-control-sets", "G1",
        outcome_var, "nspe", FULL_CONTROLS,
        "", "none", sample,
        "hetero",
        sample_desc, "full controls (sector + fouryear + log_fte)" if outcome_key != "log_fte" else "sector controls only",
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/outcome/{outcome_key}/full_controls",
                    "family": "outcome_x_controls",
                    "outcome": outcome_var})

    # Without controls (bivariate)
    run_spec(
        f"rc/outcome/{outcome_key}/bivariate",
        "modules/robustness/controls.md#standard-control-sets", "G1",
        outcome_var, "nspe", [],
        "", "none", sample,
        "hetero",
        sample_desc, "none (bivariate)",
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/outcome/{outcome_key}/bivariate",
                    "family": "outcome_x_controls",
                    "outcome": outcome_var})

    # With sector FE
    run_spec(
        f"rc/outcome/{outcome_key}/sector_fe",
        "modules/robustness/fixed_effects.md#additive-fe-variations-relative-to-baseline", "G1",
        outcome_var, "nspe", ['log_fte'] if outcome_key != "log_fte" else [],
        "sector_str", "sector FE", sample,
        "hetero",
        sample_desc, ("log_fte + sector FE" if outcome_key != "log_fte" else "sector FE only"),
        axis_block_name="fixed_effects",
        axis_block={"spec_id": f"rc/outcome/{outcome_key}/sector_fe",
                    "family": "outcome_x_fe",
                    "outcome": outcome_var})


# ============================================================
# RC: FUNCTIONAL FORM — WLS weighted by FTE
# ============================================================

print("\n=== Running WLS (FTE-weighted) variants ===")

# The do file uses [aw=fte_count] for IPEDS-level statistics
# WLS with tuitionrevshare
df_wls = df.dropna(subset=['fte_count']).copy()
df_wls = df_wls[df_wls['fte_count'] > 0].copy()
df_wls['wt'] = df_wls['fte_count']

run_spec(
    "rc/form/weights/fte_wls_baseline",
    "modules/robustness/functional_form.md#weighting", "G1",
    "tuitionrevshare", "nspe", SECTOR_CONTROLS,
    "", "none", df_wls,
    {"CRV1": "sector_str"},
    f"FTE-weighted, N={len(df_wls)}", "sector controls, WLS by FTE",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/weights/fte_wls_baseline",
                "weights": "fte_count", "estimator": "WLS"})

# WLS with full controls
run_spec(
    "rc/form/weights/fte_wls_full",
    "modules/robustness/functional_form.md#weighting", "G1",
    "tuitionrevshare", "nspe", FULL_CONTROLS,
    "", "none", df_wls,
    {"CRV1": "sector_str"},
    f"FTE-weighted, N={len(df_wls)}", "full controls, WLS by FTE",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/weights/fte_wls_full",
                "weights": "fte_count", "estimator": "WLS"})

# WLS for other outcomes
for outcome_key in ["investrevshare", "log_instruct_per_fte", "sticker_share"]:
    outcome_var = OUTCOME_VARS[outcome_key][0]
    if outcome_key == "sticker_share":
        sample_wls = df_sticker.dropna(subset=['fte_count']).copy()
        sample_wls = sample_wls[sample_wls['fte_count'] > 0].copy()
    else:
        sample_wls = df_wls.dropna(subset=[outcome_var]).copy()

    run_spec(
        f"rc/form/weights/fte_wls_{outcome_key}",
        "modules/robustness/functional_form.md#weighting", "G1",
        outcome_var, "nspe", SECTOR_CONTROLS,
        "", "none", sample_wls,
        {"CRV1": "sector_str"},
        f"FTE-weighted, N={len(sample_wls)}", "sector controls, WLS by FTE",
        axis_block_name="functional_form",
        axis_block={"spec_id": f"rc/form/weights/fte_wls_{outcome_key}",
                    "weights": "fte_count", "estimator": "WLS"})


# ============================================================
# RC: CONTROL PROGRESSION (build-up)
# ============================================================

print("\n=== Running control progression variants ===")

progression_specs = [
    ("rc/controls/progression/raw_diff", [], "", "none", "raw difference (no controls, no FE)"),
    ("rc/controls/progression/public_only", ["public"], "", "none", "public dummy only"),
    ("rc/controls/progression/sector", SECTOR_CONTROLS, "", "none", "sector controls"),
    ("rc/controls/progression/sector_size", SECTOR_CONTROLS + ["log_fte"], "", "none", "sector + size"),
    ("rc/controls/progression/full", FULL_CONTROLS, "", "none", "full controls"),
    ("rc/controls/progression/full_sector_fe", FULL_CONTROLS, "sector_str", "sector FE", "full controls + sector FE"),
]

for spec_id, ctrl, fe_str, fe_desc, ctrl_desc in progression_specs:
    run_spec(
        spec_id,
        "modules/robustness/controls.md#control-progression-build-up", "G1",
        "tuitionrevshare", "nspe", ctrl,
        fe_str, fe_desc, df,
        "hetero",
        "Full sample", ctrl_desc,
        axis_block_name="controls",
        axis_block={"spec_id": spec_id, "family": "progression",
                    "n_controls": len(ctrl), "set_name": spec_id.split("/")[-1]})


# ============================================================
# RC: RANDOM CONTROL SUBSETS
# ============================================================

print("\n=== Running random control subsets ===")

rng = np.random.RandomState(112815)
extended_controls = ['public', 'forprofit', 'fouryear', 'twoyear', 'log_fte', 'is_associates']
# Remove twoyear if fouryear present (collinear with sector dummies)

for draw_i in range(1, 11):
    k = rng.randint(1, len(extended_controls) + 1)
    chosen = list(rng.choice(extended_controls, size=k, replace=False))
    # Check for collinearity: if both fouryear and twoyear, drop twoyear
    if 'fouryear' in chosen and 'twoyear' in chosen:
        chosen.remove('twoyear')
    if 'public' in chosen and 'forprofit' in chosen and 'fouryear' in chosen and 'twoyear' in chosen:
        chosen.remove('twoyear')
    excluded = [v for v in extended_controls if v not in chosen]
    spec_id = f"rc/controls/subset/random_{draw_i:03d}"
    run_spec(
        spec_id, "modules/robustness/controls.md#subset-generation-specids", "G1",
        "tuitionrevshare", "nspe", chosen,
        "", "none", df,
        "hetero",
        "Full sample", f"random subset draw {draw_i} ({len(chosen)} controls)",
        axis_block_name="controls",
        axis_block={"spec_id": spec_id, "family": "subset", "method": "random",
                    "seed": 112815, "draw_index": draw_i,
                    "included": chosen, "excluded": excluded,
                    "n_controls": len(chosen)})


# ============================================================
# INFERENCE VARIANTS (on baseline specification)
# ============================================================

print("\n=== Running inference variants ===")

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

        m = pf.feols(full_formula, data=data, vcov=vcov)

        coef_val = float(m.coef().get(focal_var, np.nan))
        se_val = float(m.se().get(focal_var, np.nan))
        pval = float(m.pvalue().get(focal_var, np.nan))

        try:
            ci = m.confint()
            ci_lower = float(ci.loc[focal_var, ci.columns[0]]) if focal_var in ci.index else np.nan
            ci_upper = float(ci.loc[focal_var, ci.columns[1]]) if focal_var in ci.index else np.nan
        except Exception:
            ci_lower = np.nan
            ci_upper = np.nan

        nobs = int(m._N)
        try:
            r2 = float(m._r2)
        except Exception:
            r2 = np.nan

        all_coefs = {k: float(v) for k, v in m.coef().items()}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": spec_id, "method": vcov_desc},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"cross_sectional_ols": design_audit},
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
baseline_formula = "tuitionrevshare ~ nspe + public + forprofit"

# HC1 robust (default, same as baseline — for explicit listing)
run_inference_variant(
    baseline_run_id, "infer/se/hc/hc1",
    "modules/inference/standard_errors.md#heteroskedasticity-robust", "G1",
    baseline_formula, "", df, "nspe",
    "hetero", "HC1 (heteroskedasticity-robust)")

# HC3 (small-sample corrected)
run_inference_variant(
    baseline_run_id, "infer/se/hc/hc3",
    "modules/inference/standard_errors.md#heteroskedasticity-robust", "G1",
    baseline_formula, "", df, "nspe",
    {"CRV3": "sector_str"}, "HC3 / CRV3 by sector")

# Cluster by sector
run_inference_variant(
    baseline_run_id, "infer/se/cluster/sector",
    "modules/inference/standard_errors.md#clustering", "G1",
    baseline_formula, "", df, "nspe",
    {"CRV1": "sector_str"}, "cluster(sector)")

# Cluster by Carnegie classification
run_inference_variant(
    baseline_run_id, "infer/se/cluster/carnegie",
    "modules/inference/standard_errors.md#clustering", "G1",
    baseline_formula, "", df, "nspe",
    {"CRV1": "carnegie_str"}, "cluster(carnegie)")

# IID (homoskedastic, no correction)
run_inference_variant(
    baseline_run_id, "infer/se/iid",
    "modules/inference/standard_errors.md#iid", "G1",
    baseline_formula, "", df, "nspe",
    "iid", "IID (homoskedastic)")


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
    base_row = spec_df[spec_df['spec_id'] == 'baseline__nspe_tuitionrevshare']
    if len(base_row) > 0:
        print(f"\nBaseline coef on nspe (tuitionrevshare): {base_row['coefficient'].values[0]:.6f}")
        print(f"Baseline SE: {base_row['std_error'].values[0]:.6f}")
        print(f"Baseline p-value: {base_row['p_value'].values[0]:.6f}")
        print(f"Baseline N: {base_row['n_obs'].values[0]:.0f}")

    # For tuitionrevshare outcome specs
    tuit_specs = successful[successful['outcome_var'] == 'tuitionrevshare']
    if len(tuit_specs) > 0:
        print(f"\n=== TUITIONREVSHARE COEFFICIENT RANGE (successful specs) ===")
        print(f"Min coef: {tuit_specs['coefficient'].min():.6f}")
        print(f"Max coef: {tuit_specs['coefficient'].max():.6f}")
        print(f"Median coef: {tuit_specs['coefficient'].median():.6f}")
        n_sig = (tuit_specs['p_value'] < 0.05).sum()
        print(f"Significant at 5%: {n_sig}/{len(tuit_specs)}")

    print(f"\n=== ALL SPECS COEFFICIENT RANGE ===")
    print(f"Min coef: {successful['coefficient'].min():.6f}")
    print(f"Max coef: {successful['coefficient'].max():.6f}")
    print(f"Median coef: {successful['coefficient'].median():.6f}")
    n_sig = (successful['p_value'] < 0.05).sum()
    print(f"Significant at 5%: {n_sig}/{len(successful)}")
    n_sig10 = (successful['p_value'] < 0.10).sum()
    print(f"Significant at 10%: {n_sig10}/{len(successful)}")

if len(failed) > 0:
    print(f"\n=== FAILED SPECIFICATIONS ===")
    for _, row in failed.iterrows():
        print(f"  {row['spec_id']}: {row['run_error'][:100]}")


# ============================================================
# SPECIFICATION_SEARCH.md
# ============================================================

print("\nWriting SPECIFICATION_SEARCH.md...")

md_lines = []
md_lines.append("# Specification Search Report: 112815-V1")
md_lines.append("")
md_lines.append("**Paper:** Hoxby (2014), \"The Economics of Online Postsecondary Education: MOOCs, Nonselective Education, and Highly Selective Education\", AER P&P 104(5)")
md_lines.append("")
md_lines.append("## Important Note")
md_lines.append("")
md_lines.append("This paper is primarily descriptive with no causal claims. The original regressions require")
md_lines.append("restricted-access BPS 2004/2009 data (NCES license) which is not included in the package.")
md_lines.append("The specifications below use the available IPEDS institution-level data to characterize how")
md_lines.append("NSPE (nonselective postsecondary education) institutions differ in financial structure from")
md_lines.append("other institutions, matching the paper's second-part analysis.")
md_lines.append("")
md_lines.append("## Baseline Specification")
md_lines.append("")
md_lines.append("- **Design:** Cross-sectional OLS (descriptive, no causal interpretation)")
md_lines.append("- **Outcome:** tuitionrevshare (tuition revenue share)")
md_lines.append("- **Treatment:** nspe (NSPE institution indicator)")
md_lines.append("- **Controls:** public, forprofit (sector dummies)")
md_lines.append("- **Fixed effects:** none")
md_lines.append("- **Standard errors:** HC1 (heteroskedasticity-robust)")
md_lines.append("")

if len(successful) > 0:
    base_row = spec_df[spec_df['spec_id'] == 'baseline__nspe_tuitionrevshare']
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
    "Baselines": successful[successful['spec_id'].str.startswith('baseline')],
    "Controls Sets": successful[successful['spec_id'].str.startswith('rc/controls/sets/')],
    "Controls LOO": successful[successful['spec_id'].str.startswith('rc/controls/loo/')],
    "Controls Progression": successful[successful['spec_id'].str.startswith('rc/controls/progression/')],
    "Controls Subset": successful[successful['spec_id'].str.startswith('rc/controls/subset/')],
    "Sample Variants": successful[successful['spec_id'].str.startswith('rc/sample/')],
    "Outcome Variants": successful[successful['spec_id'].str.startswith('rc/outcome/')],
    "Fixed Effects": successful[successful['spec_id'].str.startswith('rc/fe/')],
    "Functional Form (WLS)": successful[successful['spec_id'].str.startswith('rc/form/')],
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
    # Focus on tuitionrevshare specs for primary assessment
    tuit_specs = successful[successful['outcome_var'] == 'tuitionrevshare']
    if len(tuit_specs) > 0:
        n_sig_total = (tuit_specs['p_value'] < 0.05).sum()
        pct_sig = n_sig_total / len(tuit_specs) * 100
        sign_consistent = ((tuit_specs['coefficient'] > 0).sum() == len(tuit_specs)) or \
                          ((tuit_specs['coefficient'] < 0).sum() == len(tuit_specs))
        median_coef = tuit_specs['coefficient'].median()
        sign_word = "positive" if median_coef > 0 else "negative"

        md_lines.append(f"### Primary outcome (tuitionrevshare)")
        md_lines.append(f"- **Sign consistency:** {'All specifications have the same sign' if sign_consistent else 'Mixed signs across specifications'}")
        md_lines.append(f"- **Significance stability:** {n_sig_total}/{len(tuit_specs)} ({pct_sig:.1f}%) specifications significant at 5%")
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
    else:
        strength = "N/A"

    md_lines.append("")
    md_lines.append("### Caveats")
    md_lines.append("- This paper is explicitly descriptive with no causal claims")
    md_lines.append("- The original regressions (incres09 ~ cert* + instid*) require restricted BPS data")
    md_lines.append("- These specifications use available IPEDS data as an adapted specification search")
    md_lines.append("- The NSPE indicator is partly collinear with Carnegie classification by construction")

md_lines.append("")
md_lines.append(f"Surface hash: `{SURFACE_HASH}`")
md_lines.append("")

with open(f"{OUTPUT_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write("\n".join(md_lines))

print(f"Wrote SPECIFICATION_SEARCH.md")
print("\nDone!")
