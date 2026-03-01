"""
Specification Search Script for 138401-V1
"Measles, the MMR Vaccine, and Adult Labor Market Outcomes"

Continuous DiD design:
  - Treatment: M12_exp_rate = (avg 12-year pre-vaccine measles incidence rate) x (years exposure) / 100,000
  - Outcome: ln_cpi_income (focal), plus cpi_incwage, cpi_incwage_no0, poverty100, employed, hrs_worked
  - FE: bpl + birthyr + year + ageblackfemale + bpl_black + bpl_female + bpl_black_female
  - Cluster: bplcohort (birth state x birth year)
  - Sample: native-born, black/white, age 25-59, ACS 2000-2017, born 1941-1980

OPTIMIZED VERSION: selective column loading, chunked filtering, dtype downcasting.
Full dataset (no subsampling). Per-spec timeouts protect against runaway computations.

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
import time
import signal
import gc

warnings.filterwarnings('ignore')

sys.path.insert(0, "scripts")
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash, software_block
)

PAPER_ID = "138401-V1"
DATA_DIR = "data/downloads/extracted/138401-V1"
OUTPUT_DIR = DATA_DIR

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

G1_DESIGN_AUDIT = surface_obj["baseline_groups"][0]["design_audit"]
G1_INFERENCE_CANONICAL = surface_obj["baseline_groups"][0]["inference_plan"]["canonical"]

# ============================================================
# TIMEOUT MECHANISM
# ============================================================
class SpecTimeout(Exception):
    pass

def timeout_handler(signum, frame):
    raise SpecTimeout("Specification timed out")

SPEC_TIMEOUT_SECONDS = 600  # 10 min per spec (full dataset is slow)

# ============================================================
# DATA LOADING & CLEANING (OPTIMIZED)
# ============================================================
print("=" * 60)
print("LOADING DATA (optimized)")
print("=" * 60)
t_start = time.time()

# Only load columns we actually need from the 1.8GB file
NEEDED_COLS = [
    'year', 'cpi99', 'statefip', 'sex', 'age', 'birthyr',
    'race', 'bpl', 'empstat', 'uhrswork', 'incwage', 'poverty'
]

print(f"Loading ACS data (columns: {len(NEEDED_COLS)} of 30)...")
t0 = time.time()

# Read in chunks, filter immediately, keep only needed columns
chunks = []
reader = pd.read_stata(
    f"{DATA_DIR}/raw_data/longrun_20002017_acs.dta",
    convert_categoricals=False,
    columns=NEEDED_COLS,
    chunksize=1_000_000
)

for i, chunk in enumerate(reader):
    # Apply sample filters immediately to reduce memory
    mask = (
        (chunk['age'] > 25) & (chunk['age'] < 60) &
        (chunk['bpl'] < 57) &
        ((chunk['race'] == 1) | (chunk['race'] == 2))
    )
    filtered = chunk.loc[mask].copy()
    if len(filtered) > 0:
        chunks.append(filtered)
    if (i + 1) % 5 == 0:
        print(f"  chunk {i+1}: {sum(len(c) for c in chunks):,} rows kept so far")

df = pd.concat(chunks, ignore_index=True)
del chunks
gc.collect()
print(f"  Filtered ACS data: {len(df):,} rows loaded in {time.time()-t0:.1f}s")

# --- Downcast integer dtypes to reduce memory ---
# NOTE: Keep incwage/cpi99 as float64 -- pyfixest demeaning requires float64
print("Downcasting dtypes...")
df['year'] = df['year'].astype(np.int16)
df['age'] = df['age'].astype(np.int8)
df['birthyr'] = df['birthyr'].astype(np.int16)
df['sex'] = df['sex'].astype(np.int8)
df['race'] = df['race'].astype(np.int8)
df['bpl'] = df['bpl'].astype(np.int8)
df['empstat'] = df['empstat'].astype(np.int8)
df['statefip'] = df['statefip'].astype(np.int8)
df['poverty'] = df['poverty'].astype(np.int16)
# incwage, cpi99, uhrswork stay as float64 for regression accuracy

# --- Create variables (matching acs_cleaning.do) ---
print("Creating variables...")

# Race dummies
df['white'] = (df['race'] == 1).astype(np.int8)
df['black'] = (df['race'] == 2).astype(np.int8)

# Exposure variable
df['exposure'] = np.int8(0)
for yr in range(1949, 1964):
    df.loc[df['birthyr'] == yr, 'exposure'] = np.int8(yr - 1948)
df.loc[df['birthyr'] > 1963, 'exposure'] = np.int8(16)

# Female identifier
df['female'] = (df['sex'] == 2).astype(np.int8)

# Interaction variables for FE (using group IDs)
df['ageblackfemale'] = df.groupby(['age', 'black', 'female']).ngroup().astype(np.int32)
# bpl_black, bpl_female, bpl_black_female as multiplicative interactions (matches Stata)
df['bpl_black'] = (df['bpl'].astype(np.int32) * df['black'].astype(np.int32)).astype(np.int32)
df['bpl_female'] = (df['bpl'].astype(np.int32) * df['female'].astype(np.int32)).astype(np.int32)
df['bpl_black_female'] = (df['bpl'].astype(np.int32) * df['black'].astype(np.int32) * df['female'].astype(np.int32)).astype(np.int32)

# Outcome variables (float64 for pyfixest compatibility)
df['cpi_incwage'] = df['incwage'] * df['cpi99'] * 1.507
df['ln_cpi_income'] = np.log(df['cpi_incwage'].replace(0, np.nan))
df['poverty100'] = np.where(df['poverty'] == 0, np.nan,
                            (df['poverty'] < 101).astype(float))
df['hrs_worked'] = df['uhrswork'].astype(np.float64)
df['employed'] = np.where(df['empstat'] == 3, np.nan,
                          (df['empstat'] == 1).astype(float))
df['cpi_incwage_no0'] = df['cpi_incwage'].replace(0, np.nan)

# Cluster variable: bplcohort
df['bplcohort'] = df.groupby(['bpl', 'birthyr']).ngroup().astype(np.int32)

# Drop columns no longer needed
df.drop(columns=['race', 'sex', 'uhrswork', 'incwage', 'cpi99', 'poverty'], inplace=True)
gc.collect()

print(f"  After cleaning: {len(df):,} rows, {len(df.columns)} columns")
mem_mb = df.memory_usage(deep=True).sum() / 1e6
print(f"  Memory usage: {mem_mb:.0f} MB")

# --- Load and merge rates data ---
print("Loading rates data...")
rates = pd.read_stata(f"{DATA_DIR}/raw_data/case_counts_population.dta", convert_categoricals=False)

# Compute measles rates
rates_pivot = rates.pivot_table(
    index=['state', 'statefip', 'bpl_region4', 'bpl_region9'],
    columns='year', values=['measles', 'population']
)
for yr in range(1952, 1964):
    m = rates_pivot[('measles', yr)]
    p = rates_pivot[('population', yr)]
    rates_pivot[('measles_rate', yr)] = (m / p) * 100000

# Compute average pre-vaccine measles rates (averaging windows 2-12)
rate_df = pd.DataFrame(index=rates_pivot.index)
for window in range(2, 13):
    start_yr = 1964 - window
    years = list(range(start_yr, 1964))
    avg = np.nanmean([rates_pivot[('measles_rate', yr)] for yr in years], axis=0)
    rate_df[f'avg_{window}yr_measles_rate'] = avg

rate_df = rate_df.reset_index()
rate_df.rename(columns={'statefip': 'bpl'}, inplace=True)

# Merge rates into main data
merge_cols = ['bpl'] + [f'avg_{w}yr_measles_rate' for w in range(2, 13)]
df = df.merge(rate_df[merge_cols], on='bpl', how='left')

# Create M_exp_rate variables (float64 for pyfixest)
for w in range(2, 13):
    df[f'M{w}_exp_rate'] = (df[f'avg_{w}yr_measles_rate'] * df['exposure']) / 100000

# Merge region info for clustering/FE variants
region_info = rates[['statefip', 'bpl_region4', 'bpl_region9']].drop_duplicates()
region_info.rename(columns={'statefip': 'bpl'}, inplace=True)
# Encode regions as string categories for pyfixest
df = df.merge(region_info, on='bpl', how='left', suffixes=('', '_rate'))
if 'bpl_region4_rate' in df.columns:
    df['bpl_region4'] = df['bpl_region4'].fillna(df['bpl_region4_rate'])
    df['bpl_region9'] = df['bpl_region9'].fillna(df['bpl_region9_rate'])
    df.drop(columns=['bpl_region4_rate', 'bpl_region9_rate'], inplace=True, errors='ignore')

# Additional clustering variables
df['bplexposure'] = df.groupby(['bpl', 'exposure']).ngroup().astype(np.int32)
df['stateexposure'] = df.groupby(['statefip', 'exposure']).ngroup().astype(np.int32)
df['statecohort'] = df.groupby(['statefip', 'birthyr']).ngroup().astype(np.int32)

# Additional FE variables
df['breg4_byear'] = df.groupby(['bpl_region4', 'birthyr']).ngroup().astype(np.int32)
df['breg9_byear'] = df.groupby(['bpl_region9', 'birthyr']).ngroup().astype(np.int32)

# State-specific linear cohort trend
df['cohort'] = (df['birthyr'] - 1940).astype(np.int8)

# Drop avg_*yr_measles_rate columns -- no longer needed
avg_rate_cols = [f'avg_{w}yr_measles_rate' for w in range(2, 13)]
df.drop(columns=avg_rate_cols, inplace=True, errors='ignore')
gc.collect()

print(f"Full data: {len(df):,} rows, {len(df.columns)} columns")
mem_mb = df.memory_usage(deep=True).sum() / 1e6
print(f"Memory usage: {mem_mb:.0f} MB")
print(f"Birth years: {df['birthyr'].min()} - {df['birthyr'].max()}")
print(f"Exposure: {df['exposure'].min()} - {df['exposure'].max()}")

# No subsampling -- use full dataset
print(f"\nUsing full dataset (no subsampling)")
print(f"Total loading time: {time.time()-t_start:.1f}s")

# ============================================================
# RESULTS STORAGE
# ============================================================
results = []
inference_results = []
spec_run_counter = 0

# ============================================================
# BASELINE FE AND CONTROLS
# ============================================================
BASELINE_FE = "bpl + birthyr + year + ageblackfemale + bpl_black + bpl_female + bpl_black_female"
BASELINE_CONTROLS = ["black", "female"]
BASELINE_VCOV = {"CRV1": "bplcohort"}
BASELINE_CLUSTER = "bplcohort"


# ============================================================
# HELPER: Run regression via pyfixest (with timeout)
# ============================================================
def run_reg(spec_id, spec_tree_path, baseline_group_id,
            outcome_var, treatment_var, controls, fe_formula,
            data, vcov, sample_desc, controls_desc, cluster_var,
            design_audit, inference_canonical,
            axis_block_name=None, axis_block=None,
            extra_block=None, notes="", return_model=False):
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"
    t_spec = time.time()

    try:
        # Set timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(SPEC_TIMEOUT_SECONDS)

        controls_str = " + ".join(controls) if controls else ""
        if controls_str and fe_formula:
            formula = f"{outcome_var} ~ {treatment_var} + {controls_str} | {fe_formula}"
        elif controls_str:
            formula = f"{outcome_var} ~ {treatment_var} + {controls_str}"
        elif fe_formula:
            formula = f"{outcome_var} ~ {treatment_var} | {fe_formula}"
        else:
            formula = f"{outcome_var} ~ {treatment_var}"

        m = pf.feols(formula, data=data, vcov=vcov)

        # Cancel timeout
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

        coef_val = float(m.coef().get(treatment_var, np.nan))
        se_val = float(m.se().get(treatment_var, np.nan))
        pval = float(m.pvalue().get(treatment_var, np.nan))
        try:
            ci = m.confint()
            ci_lower = float(ci.loc[treatment_var, ci.columns[0]])
            ci_upper = float(ci.loc[treatment_var, ci.columns[1]])
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
            design={"difference_in_differences": design_audit},
            axis_block_name=axis_block_name,
            axis_block=axis_block,
            extra=extra_block,
            notes=notes if notes else None,
        )

        results.append({
            "paper_id": PAPER_ID, "spec_run_id": run_id,
            "spec_id": spec_id, "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var, "treatment_var": treatment_var,
            "coefficient": coef_val, "std_error": se_val, "p_value": pval,
            "ci_lower": ci_lower, "ci_upper": ci_upper,
            "n_obs": nobs, "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc, "fixed_effects": fe_formula or "none",
            "controls_desc": controls_desc, "cluster_var": cluster_var,
            "run_success": 1, "run_error": ""
        })
        elapsed = time.time() - t_spec
        print(f"    [{elapsed:.1f}s] coef={coef_val:.6f}, se={se_val:.6f}, p={pval:.4f}, N={nobs:,}")
        if return_model:
            return run_id, coef_val, se_val, pval, nobs, m
        return run_id, coef_val, se_val, pval, nobs

    except (SpecTimeout, Exception) as e:
        # Cancel timeout on error
        signal.alarm(0)
        try:
            signal.signal(signal.SIGALRM, old_handler)
        except Exception:
            pass

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
            "outcome_var": outcome_var, "treatment_var": treatment_var,
            "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "n_obs": np.nan, "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc, "fixed_effects": fe_formula or "none",
            "controls_desc": controls_desc, "cluster_var": cluster_var,
            "run_success": 0, "run_error": err_msg
        })
        elapsed = time.time() - t_spec
        print(f"    [{elapsed:.1f}s] FAILED: {err_msg[:80]}")
        if return_model:
            return run_id, np.nan, np.nan, np.nan, np.nan, None
        return run_id, np.nan, np.nan, np.nan, np.nan


# ============================================================
# HELPER: Run inference variant using cached model .vcov() method
# ============================================================
def run_inference_variant_from_model(base_model, base_run_id, spec_id, spec_tree_path,
                                     baseline_group_id, treatment_var,
                                     vcov, cluster_var_label, notes=""):
    """Re-compute SEs on an already-fitted model using .vcov() -- avoids re-demeaning."""
    global spec_run_counter
    spec_run_counter += 1
    inf_run_id = f"{PAPER_ID}_infer_{spec_run_counter:03d}"
    t_spec = time.time()

    try:
        # Use .vcov() to re-compute standard errors without re-fitting
        m = base_model.vcov(vcov)

        coef_val = float(m.coef().get(treatment_var, np.nan))
        se_val = float(m.se().get(treatment_var, np.nan))
        pval = float(m.pvalue().get(treatment_var, np.nan))
        try:
            ci = m.confint()
            ci_lower = float(ci.loc[treatment_var, ci.columns[0]])
            ci_upper = float(ci.loc[treatment_var, ci.columns[1]])
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
            inference={"spec_id": spec_id, "params": {"cluster_var": cluster_var_label}},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"difference_in_differences": G1_DESIGN_AUDIT},
            notes=notes if notes else None,
        )

        inference_results.append({
            "paper_id": PAPER_ID, "inference_run_id": inf_run_id,
            "spec_run_id": base_run_id, "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "coefficient": coef_val, "std_error": se_val, "p_value": pval,
            "ci_lower": ci_lower, "ci_upper": ci_upper,
            "n_obs": nobs, "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 1, "run_error": ""
        })
        elapsed = time.time() - t_spec
        print(f"    [{elapsed:.1f}s] se={se_val:.6f}, p={pval:.4f}")

    except Exception as e:
        err_msg = str(e)[:240]
        payload = make_failure_payload(
            error=err_msg,
            error_details=error_details_from_exception(e, stage="inference"),
            software=SW_BLOCK, surface_hash=SURFACE_HASH
        )
        inference_results.append({
            "paper_id": PAPER_ID, "inference_run_id": inf_run_id,
            "spec_run_id": base_run_id, "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "n_obs": np.nan, "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 0, "run_error": err_msg
        })
        elapsed = time.time() - t_spec
        print(f"    [{elapsed:.1f}s] FAILED: {err_msg[:80]}")


# ============================================================
# BASELINE SPECS (G1)
# ============================================================
print("\n" + "=" * 60)
print("BASELINE SPECS")
print("=" * 60)

baseline_result = run_reg(
    spec_id="baseline__ln_cpi_income",
    spec_tree_path="specification_tree/designs/difference_in_differences.md",
    baseline_group_id="G1",
    outcome_var="ln_cpi_income", treatment_var="M12_exp_rate",
    controls=BASELINE_CONTROLS, fe_formula=BASELINE_FE,
    data=df, vcov=BASELINE_VCOV,
    sample_desc="Native-born black/white, age 26-59, ACS 2000-2017",
    controls_desc="black, female",
    cluster_var=BASELINE_CLUSTER,
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    notes="Table 2 Col 3: log CPI-adjusted income (focal baseline)",
    return_model=True
)
baseline_run_id = baseline_result[0]
baseline_model = baseline_result[5]  # cached for inference variants
print(f"  baseline__ln_cpi_income: run_id={baseline_run_id}")

other_baselines = [
    ("baseline__cpi_incwage", "cpi_incwage", "Table 2 Col 1: CPI-adjusted wage income"),
    ("baseline__cpi_incwage_no0", "cpi_incwage_no0", "Table 2 Col 2: CPI wage income, excl zeros"),
    ("baseline__poverty100", "poverty100", "Table 2 Col 4: Poverty status (<100% FPL)"),
    ("baseline__employed", "employed", "Table 2 Col 5: Employment status"),
    ("baseline__hrs_worked", "hrs_worked", "Table 2 Col 6: Hours worked per week"),
]

baseline_run_ids = {"ln_cpi_income": baseline_run_id}
for spec_id, outcome, note in other_baselines:
    rid, *_ = run_reg(
        spec_id=spec_id,
        spec_tree_path="specification_tree/designs/difference_in_differences.md",
        baseline_group_id="G1",
        outcome_var=outcome, treatment_var="M12_exp_rate",
        controls=BASELINE_CONTROLS, fe_formula=BASELINE_FE,
        data=df, vcov=BASELINE_VCOV,
        sample_desc="Native-born black/white, age 26-59, ACS 2000-2017",
        controls_desc="black, female",
        cluster_var=BASELINE_CLUSTER,
        design_audit=G1_DESIGN_AUDIT,
        inference_canonical=G1_INFERENCE_CANONICAL,
        notes=note
    )
    baseline_run_ids[outcome] = rid
    print(f"  {spec_id}: run_id={rid}")


# ============================================================
# DESIGN SPEC: TWFE
# ============================================================
print("\n" + "=" * 60)
print("DESIGN SPECS")
print("=" * 60)

rid, *_ = run_reg(
    spec_id="design/difference_in_differences/estimator/twfe",
    spec_tree_path="specification_tree/designs/difference_in_differences.md#twfe",
    baseline_group_id="G1",
    outcome_var="ln_cpi_income", treatment_var="M12_exp_rate",
    controls=BASELINE_CONTROLS, fe_formula=BASELINE_FE,
    data=df, vcov=BASELINE_VCOV,
    sample_desc="Native-born black/white, age 26-59, ACS 2000-2017",
    controls_desc="black, female",
    cluster_var=BASELINE_CLUSTER,
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    notes="TWFE design spec (identical to baseline)"
)
print(f"  design/twfe: run_id={rid}")


# ============================================================
# RC/CONTROLS SPECS
# ============================================================
print("\n" + "=" * 60)
print("RC/CONTROLS SPECS")
print("=" * 60)

# LOO: drop black
rid, *_ = run_reg(
    spec_id="rc/controls/loo/drop_black",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G1",
    outcome_var="ln_cpi_income", treatment_var="M12_exp_rate",
    controls=["female"], fe_formula=BASELINE_FE,
    data=df, vcov=BASELINE_VCOV,
    sample_desc="Native-born black/white, age 26-59, ACS 2000-2017",
    controls_desc="female (dropped: black)",
    cluster_var=BASELINE_CLUSTER,
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_black", "family": "loo",
                "dropped": ["black"], "n_controls": 1},
    notes="LOO: drop black level control (FE interactions retained)"
)
print(f"  rc/controls/loo/drop_black: run_id={rid}")

# LOO: drop female
rid, *_ = run_reg(
    spec_id="rc/controls/loo/drop_female",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G1",
    outcome_var="ln_cpi_income", treatment_var="M12_exp_rate",
    controls=["black"], fe_formula=BASELINE_FE,
    data=df, vcov=BASELINE_VCOV,
    sample_desc="Native-born black/white, age 26-59, ACS 2000-2017",
    controls_desc="black (dropped: female)",
    cluster_var=BASELINE_CLUSTER,
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_female", "family": "loo",
                "dropped": ["female"], "n_controls": 1},
    notes="LOO: drop female level control (FE interactions retained)"
)
print(f"  rc/controls/loo/drop_female: run_id={rid}")

# No level controls
rid, *_ = run_reg(
    spec_id="rc/controls/sets/no_level_controls",
    spec_tree_path="specification_tree/modules/robustness/controls.md#sets",
    baseline_group_id="G1",
    outcome_var="ln_cpi_income", treatment_var="M12_exp_rate",
    controls=[], fe_formula=BASELINE_FE,
    data=df, vcov=BASELINE_VCOV,
    sample_desc="Native-born black/white, age 26-59, ACS 2000-2017",
    controls_desc="none (all level controls dropped)",
    cluster_var=BASELINE_CLUSTER,
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/no_level_controls", "family": "sets",
                "dropped": ["black", "female"], "n_controls": 0},
    notes="No level controls (FE interactions retained)"
)
print(f"  rc/controls/sets/no_level_controls: run_id={rid}")


# ============================================================
# RC/SAMPLE SPECS
# ============================================================
print("\n" + "=" * 60)
print("RC/SAMPLE SPECS")
print("=" * 60)

# exclude_partial_exposure: keep only exposure==0 or exposure==16
df_partial = df[(df['exposure'] == 0) | (df['exposure'] == 16)]
rid, *_ = run_reg(
    spec_id="rc/sample/restriction/exclude_partial_exposure",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restriction",
    baseline_group_id="G1",
    outcome_var="ln_cpi_income", treatment_var="M12_exp_rate",
    controls=BASELINE_CONTROLS, fe_formula=BASELINE_FE,
    data=df_partial, vcov=BASELINE_VCOV,
    sample_desc="Exposure 0 or 16 only (Table 3)",
    controls_desc="black, female",
    cluster_var=BASELINE_CLUSTER,
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/exclude_partial_exposure",
                "restriction": "exposure==0 or exposure==16"},
    notes="Table 3: exclude partial exposure cohorts"
)
print(f"  exclude_partial_exposure: run_id={rid}")
del df_partial

# narrow_cohort_window_1941_1971
df_narrow = df[df['birthyr'] < 1972]
rid, *_ = run_reg(
    spec_id="rc/sample/restriction/narrow_cohort_window_1941_1971",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restriction",
    baseline_group_id="G1",
    outcome_var="ln_cpi_income", treatment_var="M12_exp_rate",
    controls=BASELINE_CONTROLS, fe_formula=BASELINE_FE,
    data=df_narrow, vcov=BASELINE_VCOV,
    sample_desc="Birth years 1941-1971 (Table 3)",
    controls_desc="black, female",
    cluster_var=BASELINE_CLUSTER,
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/narrow_cohort_window_1941_1971",
                "restriction": "birthyr < 1972"},
    notes="Table 3: narrow cohort window 1941-1971"
)
print(f"  narrow_cohort_window_1941_1971: run_id={rid}")
del df_narrow

# narrow_cohort_window_1945_1975
df_narrow2 = df[(df['birthyr'] >= 1945) & (df['birthyr'] <= 1975)]
rid, *_ = run_reg(
    spec_id="rc/sample/restriction/narrow_cohort_window_1945_1975",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restriction",
    baseline_group_id="G1",
    outcome_var="ln_cpi_income", treatment_var="M12_exp_rate",
    controls=BASELINE_CONTROLS, fe_formula=BASELINE_FE,
    data=df_narrow2, vcov=BASELINE_VCOV,
    sample_desc="Birth years 1945-1975",
    controls_desc="black, female",
    cluster_var=BASELINE_CLUSTER,
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/narrow_cohort_window_1945_1975",
                "restriction": "birthyr >= 1945 & birthyr <= 1975"},
    notes="Symmetric narrow window around vaccine introduction"
)
print(f"  narrow_cohort_window_1945_1975: run_id={rid}")
del df_narrow2

# men_only (with adjusted FE: drop bpl_female, bpl_black_female)
df_men = df[df['female'] == 0]
men_fe = "bpl + birthyr + year + ageblackfemale + bpl_black"
rid, *_ = run_reg(
    spec_id="rc/sample/restriction/men_only",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restriction",
    baseline_group_id="G1",
    outcome_var="ln_cpi_income", treatment_var="M12_exp_rate",
    controls=["black"], fe_formula=men_fe,
    data=df_men, vcov=BASELINE_VCOV,
    sample_desc="Men only",
    controls_desc="black",
    cluster_var=BASELINE_CLUSTER,
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/men_only",
                "restriction": "female==0",
                "fe_adjustment": "dropped bpl_female, bpl_black_female"},
    notes="Men only subsample, drop female-interaction FE"
)
print(f"  men_only: run_id={rid}")
del df_men

# women_only (drop bpl_female, bpl_black_female FE)
df_women = df[df['female'] == 1]
women_fe = "bpl + birthyr + year + ageblackfemale + bpl_black"
rid, *_ = run_reg(
    spec_id="rc/sample/restriction/women_only",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restriction",
    baseline_group_id="G1",
    outcome_var="ln_cpi_income", treatment_var="M12_exp_rate",
    controls=["black"], fe_formula=women_fe,
    data=df_women, vcov=BASELINE_VCOV,
    sample_desc="Women only",
    controls_desc="black",
    cluster_var=BASELINE_CLUSTER,
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/women_only",
                "restriction": "female==1",
                "fe_adjustment": "dropped bpl_female, bpl_black_female"},
    notes="Women only subsample, drop female-interaction FE"
)
print(f"  women_only: run_id={rid}")
del df_women

# white_only (drop bpl_black, bpl_black_female FE)
df_white = df[df['white'] == 1]
white_fe = "bpl + birthyr + year + ageblackfemale + bpl_female"
rid, *_ = run_reg(
    spec_id="rc/sample/restriction/white_only",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restriction",
    baseline_group_id="G1",
    outcome_var="ln_cpi_income", treatment_var="M12_exp_rate",
    controls=["female"], fe_formula=white_fe,
    data=df_white, vcov=BASELINE_VCOV,
    sample_desc="White only",
    controls_desc="female",
    cluster_var=BASELINE_CLUSTER,
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/white_only",
                "restriction": "white==1",
                "fe_adjustment": "dropped bpl_black, bpl_black_female"},
    notes="White only subsample, drop black-interaction FE"
)
print(f"  white_only: run_id={rid}")
del df_white

# black_only (drop bpl_black, bpl_black_female FE)
df_black = df[df['black'] == 1]
black_fe = "bpl + birthyr + year + ageblackfemale + bpl_female"
rid, *_ = run_reg(
    spec_id="rc/sample/restriction/black_only",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restriction",
    baseline_group_id="G1",
    outcome_var="ln_cpi_income", treatment_var="M12_exp_rate",
    controls=["female"], fe_formula=black_fe,
    data=df_black, vcov=BASELINE_VCOV,
    sample_desc="Black only",
    controls_desc="female",
    cluster_var=BASELINE_CLUSTER,
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/black_only",
                "restriction": "black==1",
                "fe_adjustment": "dropped bpl_black, bpl_black_female"},
    notes="Black only subsample, drop black-interaction FE"
)
print(f"  black_only: run_id={rid}")
del df_black

# age_30_55
df_age = df[(df['age'] >= 30) & (df['age'] <= 55)]
rid, *_ = run_reg(
    spec_id="rc/sample/restriction/age_30_55",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restriction",
    baseline_group_id="G1",
    outcome_var="ln_cpi_income", treatment_var="M12_exp_rate",
    controls=BASELINE_CONTROLS, fe_formula=BASELINE_FE,
    data=df_age, vcov=BASELINE_VCOV,
    sample_desc="Age 30-55 (tighter age window)",
    controls_desc="black, female",
    cluster_var=BASELINE_CLUSTER,
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/age_30_55",
                "restriction": "age >= 30 & age <= 55"},
    notes="Tighter age window: 30-55"
)
print(f"  age_30_55: run_id={rid}")
del df_age

# exclude_dc (bpl != 11, DC FIPS code)
df_nodc = df[df['bpl'] != 11]
rid, *_ = run_reg(
    spec_id="rc/sample/restriction/exclude_dc",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restriction",
    baseline_group_id="G1",
    outcome_var="ln_cpi_income", treatment_var="M12_exp_rate",
    controls=BASELINE_CONTROLS, fe_formula=BASELINE_FE,
    data=df_nodc, vcov=BASELINE_VCOV,
    sample_desc="Exclude DC (bpl != 11)",
    controls_desc="black, female",
    cluster_var=BASELINE_CLUSTER,
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/exclude_dc",
                "restriction": "bpl != 11 (DC)"},
    notes="Exclude District of Columbia"
)
print(f"  exclude_dc: run_id={rid}")
del df_nodc

# post_2005_acs_only
df_post05 = df[df['year'] >= 2005]
rid, *_ = run_reg(
    spec_id="rc/sample/restriction/post_2005_acs_only",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restriction",
    baseline_group_id="G1",
    outcome_var="ln_cpi_income", treatment_var="M12_exp_rate",
    controls=BASELINE_CONTROLS, fe_formula=BASELINE_FE,
    data=df_post05, vcov=BASELINE_VCOV,
    sample_desc="ACS 2005-2017 only",
    controls_desc="black, female",
    cluster_var=BASELINE_CLUSTER,
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/post_2005_acs_only",
                "restriction": "year >= 2005"},
    notes="Post-2005 ACS only"
)
print(f"  post_2005_acs_only: run_id={rid}")
del df_post05

# pre_2010_acs_only
df_pre10 = df[df['year'] <= 2010]
rid, *_ = run_reg(
    spec_id="rc/sample/restriction/pre_2010_acs_only",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restriction",
    baseline_group_id="G1",
    outcome_var="ln_cpi_income", treatment_var="M12_exp_rate",
    controls=BASELINE_CONTROLS, fe_formula=BASELINE_FE,
    data=df_pre10, vcov=BASELINE_VCOV,
    sample_desc="ACS 2000-2010 only",
    controls_desc="black, female",
    cluster_var=BASELINE_CLUSTER,
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/pre_2010_acs_only",
                "restriction": "year <= 2010"},
    notes="Pre-2010 ACS only"
)
print(f"  pre_2010_acs_only: run_id={rid}")
del df_pre10

# employed_only (empstat == 1)
df_emp = df[df['employed'] == 1]
rid, *_ = run_reg(
    spec_id="rc/sample/restriction/employed_only",
    spec_tree_path="specification_tree/modules/robustness/sample.md#restriction",
    baseline_group_id="G1",
    outcome_var="ln_cpi_income", treatment_var="M12_exp_rate",
    controls=BASELINE_CONTROLS, fe_formula=BASELINE_FE,
    data=df_emp, vcov=BASELINE_VCOV,
    sample_desc="Employed only",
    controls_desc="black, female",
    cluster_var=BASELINE_CLUSTER,
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/employed_only",
                "restriction": "employed==1"},
    notes="Employed only subsample"
)
print(f"  employed_only: run_id={rid}")
del df_emp

gc.collect()


# ============================================================
# RC/FE SPECS
# ============================================================
print("\n" + "=" * 60)
print("RC/FE SPECS")
print("=" * 60)

# Add breg4_byear (Census region 4 x birth year)
fe_breg4 = BASELINE_FE + " + breg4_byear"
rid, *_ = run_reg(
    spec_id="rc/fe/add/breg4_byear",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#add",
    baseline_group_id="G1",
    outcome_var="ln_cpi_income", treatment_var="M12_exp_rate",
    controls=BASELINE_CONTROLS, fe_formula=fe_breg4,
    data=df, vcov=BASELINE_VCOV,
    sample_desc="Native-born black/white, age 26-59, ACS 2000-2017",
    controls_desc="black, female",
    cluster_var=BASELINE_CLUSTER,
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/add/breg4_byear", "action": "add",
                "added": ["breg4_byear"]},
    notes="Appendix Table 4: add Census region(4) x birth year FE"
)
print(f"  rc/fe/add/breg4_byear: run_id={rid}")

# Add breg9_byear (Census division 9 x birth year)
fe_breg9 = BASELINE_FE + " + breg9_byear"
rid, *_ = run_reg(
    spec_id="rc/fe/add/breg9_byear",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#add",
    baseline_group_id="G1",
    outcome_var="ln_cpi_income", treatment_var="M12_exp_rate",
    controls=BASELINE_CONTROLS, fe_formula=fe_breg9,
    data=df, vcov=BASELINE_VCOV,
    sample_desc="Native-born black/white, age 26-59, ACS 2000-2017",
    controls_desc="black, female",
    cluster_var=BASELINE_CLUSTER,
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/add/breg9_byear", "action": "add",
                "added": ["breg9_byear"]},
    notes="Table 4 / Appendix Table 4: add Census division(9) x birth year FE"
)
print(f"  rc/fe/add/breg9_byear: run_id={rid}")

# Add bpl_cohort_trend (state-specific linear cohort trends)
rid, *_ = run_reg(
    spec_id="rc/fe/add/bpl_cohort_trend",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#add",
    baseline_group_id="G1",
    outcome_var="ln_cpi_income", treatment_var="M12_exp_rate",
    controls=BASELINE_CONTROLS + ["i(bpl, cohort)"], fe_formula=BASELINE_FE,
    data=df, vcov=BASELINE_VCOV,
    sample_desc="Native-born black/white, age 26-59, ACS 2000-2017",
    controls_desc="black, female, bpl-specific linear cohort trends",
    cluster_var=BASELINE_CLUSTER,
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/add/bpl_cohort_trend", "action": "add",
                "added": ["bpl_cohort_trend (bpl x linear cohort)"]},
    notes="Appendix Table 4: add birth-state-specific linear cohort trends"
)
print(f"  rc/fe/add/bpl_cohort_trend: run_id={rid}")

# Add mean_reversion_control (Table 4)
df['precohort'] = (df['exposure'] == 0).astype(np.int8)
mean_by_bpl_exp = df.groupby(['bpl', 'exposure'])['ln_cpi_income'].transform('mean')
df['pm_ln_cpi_income'] = mean_by_bpl_exp * df['precohort']  # float64

rid, *_ = run_reg(
    spec_id="rc/fe/add/mean_reversion_control",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#add",
    baseline_group_id="G1",
    outcome_var="ln_cpi_income", treatment_var="M12_exp_rate",
    controls=BASELINE_CONTROLS + ["pm_ln_cpi_income"], fe_formula=BASELINE_FE,
    data=df, vcov=BASELINE_VCOV,
    sample_desc="Native-born black/white, age 26-59, ACS 2000-2017",
    controls_desc="black, female, mean_reversion_control (pre-cohort mean x precohort)",
    cluster_var=BASELINE_CLUSTER,
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/add/mean_reversion_control", "action": "add",
                "added": ["pm_ln_cpi_income"]},
    notes="Table 4: mean reversion control (pre-cohort avg outcome x precohort dummy)"
)
print(f"  rc/fe/add/mean_reversion_control: run_id={rid}")

# Drop bpl_black FE
fe_drop_bpl_black = "bpl + birthyr + year + ageblackfemale + bpl_female + bpl_black_female"
rid, *_ = run_reg(
    spec_id="rc/fe/drop/bpl_black",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#drop",
    baseline_group_id="G1",
    outcome_var="ln_cpi_income", treatment_var="M12_exp_rate",
    controls=BASELINE_CONTROLS, fe_formula=fe_drop_bpl_black,
    data=df, vcov=BASELINE_VCOV,
    sample_desc="Native-born black/white, age 26-59, ACS 2000-2017",
    controls_desc="black, female",
    cluster_var=BASELINE_CLUSTER,
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/drop/bpl_black", "action": "drop",
                "dropped": ["bpl_black"]},
    notes="Drop bpl x black FE interaction"
)
print(f"  rc/fe/drop/bpl_black: run_id={rid}")

# Drop bpl_female FE
fe_drop_bpl_female = "bpl + birthyr + year + ageblackfemale + bpl_black + bpl_black_female"
rid, *_ = run_reg(
    spec_id="rc/fe/drop/bpl_female",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#drop",
    baseline_group_id="G1",
    outcome_var="ln_cpi_income", treatment_var="M12_exp_rate",
    controls=BASELINE_CONTROLS, fe_formula=fe_drop_bpl_female,
    data=df, vcov=BASELINE_VCOV,
    sample_desc="Native-born black/white, age 26-59, ACS 2000-2017",
    controls_desc="black, female",
    cluster_var=BASELINE_CLUSTER,
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/drop/bpl_female", "action": "drop",
                "dropped": ["bpl_female"]},
    notes="Drop bpl x female FE interaction"
)
print(f"  rc/fe/drop/bpl_female: run_id={rid}")

# Drop bpl_black_female FE
fe_drop_bpl_bf = "bpl + birthyr + year + ageblackfemale + bpl_black + bpl_female"
rid, *_ = run_reg(
    spec_id="rc/fe/drop/bpl_black_female",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#drop",
    baseline_group_id="G1",
    outcome_var="ln_cpi_income", treatment_var="M12_exp_rate",
    controls=BASELINE_CONTROLS, fe_formula=fe_drop_bpl_bf,
    data=df, vcov=BASELINE_VCOV,
    sample_desc="Native-born black/white, age 26-59, ACS 2000-2017",
    controls_desc="black, female",
    cluster_var=BASELINE_CLUSTER,
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/drop/bpl_black_female", "action": "drop",
                "dropped": ["bpl_black_female"]},
    notes="Drop bpl x black x female triple interaction FE"
)
print(f"  rc/fe/drop/bpl_black_female: run_id={rid}")

# Drop ageblackfemale FE
fe_drop_abf = "bpl + birthyr + year + bpl_black + bpl_female + bpl_black_female"
rid, *_ = run_reg(
    spec_id="rc/fe/drop/ageblackfemale",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#drop",
    baseline_group_id="G1",
    outcome_var="ln_cpi_income", treatment_var="M12_exp_rate",
    controls=BASELINE_CONTROLS, fe_formula=fe_drop_abf,
    data=df, vcov=BASELINE_VCOV,
    sample_desc="Native-born black/white, age 26-59, ACS 2000-2017",
    controls_desc="black, female",
    cluster_var=BASELINE_CLUSTER,
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/drop/ageblackfemale", "action": "drop",
                "dropped": ["ageblackfemale"]},
    notes="Drop age x black x female FE interaction"
)
print(f"  rc/fe/drop/ageblackfemale: run_id={rid}")

# Simplify: no demographic interactions (core DiD FE only)
fe_simple = "bpl + birthyr + year"
rid, *_ = run_reg(
    spec_id="rc/fe/simplify/no_demographic_interactions",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#simplify",
    baseline_group_id="G1",
    outcome_var="ln_cpi_income", treatment_var="M12_exp_rate",
    controls=BASELINE_CONTROLS, fe_formula=fe_simple,
    data=df, vcov=BASELINE_VCOV,
    sample_desc="Native-born black/white, age 26-59, ACS 2000-2017",
    controls_desc="black, female",
    cluster_var=BASELINE_CLUSTER,
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/simplify/no_demographic_interactions", "action": "simplify",
                "dropped": ["ageblackfemale", "bpl_black", "bpl_female", "bpl_black_female"]},
    notes="Core DiD FE only (bpl, birthyr, year) without demographic interactions"
)
print(f"  rc/fe/simplify/no_demographic_interactions: run_id={rid}")


# ============================================================
# RC/DATA/TREATMENT_CONSTRUCTION SPECS (M2-M11)
# ============================================================
print("\n" + "=" * 60)
print("RC/DATA/TREATMENT_CONSTRUCTION SPECS")
print("=" * 60)

for w in range(2, 12):
    tvar = f"M{w}_exp_rate"
    rid, *_ = run_reg(
        spec_id=f"rc/data/treatment_construction/M{w}_exp_rate",
        spec_tree_path="specification_tree/modules/robustness/data_construction.md",
        baseline_group_id="G1",
        outcome_var="ln_cpi_income", treatment_var=tvar,
        controls=BASELINE_CONTROLS, fe_formula=BASELINE_FE,
        data=df, vcov=BASELINE_VCOV,
        sample_desc="Native-born black/white, age 26-59, ACS 2000-2017",
        controls_desc="black, female",
        cluster_var=BASELINE_CLUSTER,
        design_audit=G1_DESIGN_AUDIT,
        inference_canonical=G1_INFERENCE_CANONICAL,
        axis_block_name="data_construction",
        axis_block={"spec_id": f"rc/data/treatment_construction/M{w}_exp_rate",
                    "treatment_construction": f"avg_{w}yr_measles_rate * exposure / 100000",
                    "averaging_window": w},
        notes=f"Appendix Table 2: {w}-year pre-vaccine measles rate averaging window"
    )
    print(f"  M{w}_exp_rate: run_id={rid}")


# ============================================================
# RC/FORM/OUTCOME SPECS
# ============================================================
print("\n" + "=" * 60)
print("RC/FORM/OUTCOME SPECS")
print("=" * 60)

# level_income: cpi_incwage as outcome instead of log
rid, *_ = run_reg(
    spec_id="rc/form/outcome/level_income",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md#outcome",
    baseline_group_id="G1",
    outcome_var="cpi_incwage", treatment_var="M12_exp_rate",
    controls=BASELINE_CONTROLS, fe_formula=BASELINE_FE,
    data=df, vcov=BASELINE_VCOV,
    sample_desc="Native-born black/white, age 26-59, ACS 2000-2017",
    controls_desc="black, female",
    cluster_var=BASELINE_CLUSTER,
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/level_income",
                "interpretation": "Level CPI-adjusted income instead of log",
                "outcome_transform": "level"},
    notes="Functional form: level income instead of log"
)
print(f"  rc/form/outcome/level_income: run_id={rid}")

# level_income_no_zeros
rid, *_ = run_reg(
    spec_id="rc/form/outcome/level_income_no_zeros",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md#outcome",
    baseline_group_id="G1",
    outcome_var="cpi_incwage_no0", treatment_var="M12_exp_rate",
    controls=BASELINE_CONTROLS, fe_formula=BASELINE_FE,
    data=df, vcov=BASELINE_VCOV,
    sample_desc="Native-born black/white, age 26-59, ACS 2000-2017",
    controls_desc="black, female",
    cluster_var=BASELINE_CLUSTER,
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/level_income_no_zeros",
                "interpretation": "Level CPI-adjusted income excluding zeros",
                "outcome_transform": "level, zeros excluded"},
    notes="Functional form: level income excluding zeros"
)
print(f"  rc/form/outcome/level_income_no_zeros: run_id={rid}")


# ============================================================
# RC/JOINT SPECS
# ============================================================
print("\n" + "=" * 60)
print("RC/JOINT SPECS")
print("=" * 60)

# narrow_window_breg9: birthyr < 1972 + breg9_byear FE
df_nw = df[df['birthyr'] < 1972]
rid, *_ = run_reg(
    spec_id="rc/joint/sample_and_fe/narrow_window_breg9",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="ln_cpi_income", treatment_var="M12_exp_rate",
    controls=BASELINE_CONTROLS, fe_formula=fe_breg9,
    data=df_nw, vcov=BASELINE_VCOV,
    sample_desc="Birth years 1941-1971 + breg9_byear FE",
    controls_desc="black, female",
    cluster_var=BASELINE_CLUSTER,
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/sample_and_fe/narrow_window_breg9",
                "components": ["rc/sample/restriction/narrow_cohort_window_1941_1971",
                              "rc/fe/add/breg9_byear"]},
    notes="Joint: narrow window (1941-1971) + division x birth year FE"
)
print(f"  narrow_window_breg9: run_id={rid}")
del df_nw

# exclude_partial_breg9: (exposure 0 or 16) + breg9_byear FE
df_ep = df[(df['exposure'] == 0) | (df['exposure'] == 16)]
rid, *_ = run_reg(
    spec_id="rc/joint/sample_and_fe/exclude_partial_breg9",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="ln_cpi_income", treatment_var="M12_exp_rate",
    controls=BASELINE_CONTROLS, fe_formula=fe_breg9,
    data=df_ep, vcov=BASELINE_VCOV,
    sample_desc="Exposure 0 or 16 only + breg9_byear FE",
    controls_desc="black, female",
    cluster_var=BASELINE_CLUSTER,
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/sample_and_fe/exclude_partial_breg9",
                "components": ["rc/sample/restriction/exclude_partial_exposure",
                              "rc/fe/add/breg9_byear"]},
    notes="Joint: exclude partial exposure + division x birth year FE"
)
print(f"  exclude_partial_breg9: run_id={rid}")
del df_ep

# white_men_only
wm_fe = "bpl + birthyr + year + ageblackfemale"
df_wm = df[(df['white'] == 1) & (df['female'] == 0)]
rid, *_ = run_reg(
    spec_id="rc/joint/sample_and_fe/white_men_only",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="ln_cpi_income", treatment_var="M12_exp_rate",
    controls=[], fe_formula=wm_fe,
    data=df_wm, vcov=BASELINE_VCOV,
    sample_desc="White men only",
    controls_desc="none",
    cluster_var=BASELINE_CLUSTER,
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/sample_and_fe/white_men_only",
                "components": ["rc/sample/restriction/white_only",
                              "rc/sample/restriction/men_only"],
                "fe_adjustment": "dropped black, female, bpl_black, bpl_female, bpl_black_female"},
    notes="Joint: white men only, all demographic interaction FE dropped"
)
print(f"  white_men_only: run_id={rid}")
del df_wm

# black_men_only
df_bm = df[(df['black'] == 1) & (df['female'] == 0)]
rid, *_ = run_reg(
    spec_id="rc/joint/sample_and_fe/black_men_only",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="ln_cpi_income", treatment_var="M12_exp_rate",
    controls=[], fe_formula=wm_fe,
    data=df_bm, vcov=BASELINE_VCOV,
    sample_desc="Black men only",
    controls_desc="none",
    cluster_var=BASELINE_CLUSTER,
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/sample_and_fe/black_men_only",
                "components": ["rc/sample/restriction/black_only",
                              "rc/sample/restriction/men_only"],
                "fe_adjustment": "dropped black, female, bpl_black, bpl_female, bpl_black_female"},
    notes="Joint: black men only, all demographic interaction FE dropped"
)
print(f"  black_men_only: run_id={rid}")
del df_bm

# white_women_only
df_ww = df[(df['white'] == 1) & (df['female'] == 1)]
rid, *_ = run_reg(
    spec_id="rc/joint/sample_and_fe/white_women_only",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="ln_cpi_income", treatment_var="M12_exp_rate",
    controls=[], fe_formula=wm_fe,
    data=df_ww, vcov=BASELINE_VCOV,
    sample_desc="White women only",
    controls_desc="none",
    cluster_var=BASELINE_CLUSTER,
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/sample_and_fe/white_women_only",
                "components": ["rc/sample/restriction/white_only",
                              "rc/sample/restriction/women_only"],
                "fe_adjustment": "dropped black, female, bpl_black, bpl_female, bpl_black_female"},
    notes="Joint: white women only, all demographic interaction FE dropped"
)
print(f"  white_women_only: run_id={rid}")
del df_ww

# narrow_window_breg4: birthyr < 1972 + breg4_byear FE
df_nw4 = df[df['birthyr'] < 1972]
rid, *_ = run_reg(
    spec_id="rc/joint/sample_and_fe/narrow_window_breg4",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="ln_cpi_income", treatment_var="M12_exp_rate",
    controls=BASELINE_CONTROLS, fe_formula=fe_breg4,
    data=df_nw4, vcov=BASELINE_VCOV,
    sample_desc="Birth years 1941-1971 + breg4_byear FE",
    controls_desc="black, female",
    cluster_var=BASELINE_CLUSTER,
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="joint",
    axis_block={"spec_id": "rc/joint/sample_and_fe/narrow_window_breg4",
                "components": ["rc/sample/restriction/narrow_cohort_window_1941_1971",
                              "rc/fe/add/breg4_byear"]},
    notes="Joint: narrow window (1941-1971) + region x birth year FE"
)
print(f"  narrow_window_breg4: run_id={rid}")
del df_nw4

gc.collect()


# ============================================================
# INFERENCE VARIANTS (for focal baseline only)
# ============================================================
print("\n" + "=" * 60)
print("INFERENCE VARIANTS")
print("=" * 60)

inference_variants = [
    ("infer/se/cluster/bpl", {"CRV1": "bpl"}, "bpl",
     "specification_tree/modules/inference/clustering.md"),
    ("infer/se/cluster/bplexposure", {"CRV1": "bplexposure"}, "bplexposure",
     "specification_tree/modules/inference/clustering.md"),
    ("infer/se/cluster/bpl_region4", {"CRV1": "bpl_region4"}, "bpl_region4",
     "specification_tree/modules/inference/clustering.md"),
    ("infer/se/cluster/bpl_region9", {"CRV1": "bpl_region9"}, "bpl_region9",
     "specification_tree/modules/inference/clustering.md"),
    ("infer/se/cluster/stateexposure", {"CRV1": "stateexposure"}, "stateexposure",
     "specification_tree/modules/inference/clustering.md"),
    ("infer/se/cluster/statecohort", {"CRV1": "statecohort"}, "statecohort",
     "specification_tree/modules/inference/clustering.md"),
    ("infer/se/cluster/statefip", {"CRV1": "statefip"}, "statefip",
     "specification_tree/modules/inference/clustering.md"),
    ("infer/se/cluster/birthyr", {"CRV1": "birthyr"}, "birthyr",
     "specification_tree/modules/inference/clustering.md"),
    ("infer/se/cluster/exposure", {"CRV1": "exposure"}, "exposure",
     "specification_tree/modules/inference/clustering.md"),
    ("infer/se/hc/hc1", "hetero", "HC1",
     "specification_tree/modules/inference/robust_se.md"),
]

if baseline_model is not None:
    for spec_id, vcov, cluster_label, tree_path in inference_variants:
        run_inference_variant_from_model(
            base_model=baseline_model,
            base_run_id=baseline_run_id,
            spec_id=spec_id,
            spec_tree_path=tree_path,
            baseline_group_id="G1",
            treatment_var="M12_exp_rate",
            vcov=vcov,
            cluster_var_label=cluster_label,
            notes=f"Inference variant: {cluster_label}"
        )
        print(f"  {spec_id}: done")
else:
    print("  SKIPPED: baseline model failed, cannot run inference variants")


# ============================================================
# WRITE OUTPUTS
# ============================================================
print("\n" + "=" * 60)
print("WRITING OUTPUTS")
print("=" * 60)

# specification_results.csv
df_results = pd.DataFrame(results)
df_results.to_csv(f"{OUTPUT_DIR}/specification_results.csv", index=False)
print(f"  specification_results.csv: {len(df_results)} rows")

# inference_results.csv
df_infer = pd.DataFrame(inference_results)
df_infer.to_csv(f"{OUTPUT_DIR}/inference_results.csv", index=False)
print(f"  inference_results.csv: {len(df_infer)} rows")

# Count successes/failures
n_success = int(df_results['run_success'].sum())
n_fail = len(df_results) - n_success
n_infer_success = int(df_infer['run_success'].sum()) if len(df_infer) > 0 else 0
n_infer_fail = len(df_infer) - n_infer_success if len(df_infer) > 0 else 0

# SPECIFICATION_SEARCH.md
search_md = f"""# Specification Search: {PAPER_ID}

## Surface Summary

- **Paper**: Measles, the MMR Vaccine, and Adult Labor Market Outcomes
- **Design**: Continuous difference-in-differences (TWFE)
- **Baseline groups**: 1 (G1)
- **Focal outcome**: ln_cpi_income (log CPI-adjusted wage income)
- **Treatment**: M12_exp_rate (12-year pre-vaccine measles rate x exposure / 100,000)
- **Fixed effects**: bpl, birthyr, year, ageblackfemale, bpl_black, bpl_female, bpl_black_female
- **Clustering**: bplcohort (birth state x birth year)
- **Budget**: 80 max core specs
- **Seed**: 138401

## Execution Summary

### Specification Results (specification_results.csv)
- **Total specs planned**: {len(df_results)}
- **Successful**: {n_success}
- **Failed**: {n_fail}

| Category | Count |
|---|---|
| Baseline specs (6 outcomes) | 6 |
| Design (TWFE) | 1 |
| RC/controls/loo | 2 |
| RC/controls/sets | 1 |
| RC/sample/restriction | 12 |
| RC/fe/add | 4 |
| RC/fe/drop | 4 |
| RC/fe/simplify | 1 |
| RC/data/treatment_construction (M2-M11) | 10 |
| RC/form/outcome | 2 |
| RC/joint/sample_and_fe | 6 |
| **Total** | **{len(df_results)}** |

### Inference Results (inference_results.csv)
- **Total inference variants**: {len(df_infer)}
- **Successful**: {n_infer_success}
- **Failed**: {n_infer_fail}

Inference variants run on the focal baseline (ln_cpi_income):
- bpl (birth state)
- bplexposure (birth state x exposure)
- bpl_region4 (Census region, 4 clusters)
- bpl_region9 (Census division, 9 clusters)
- stateexposure (state of residence x exposure)
- statecohort (state of residence x birth year)
- statefip (state of residence)
- birthyr (birth year)
- exposure (exposure level)
- HC1 (heteroskedasticity-robust, no clustering)

## Deviations and Notes

- Data cleaning replicates acs_cleaning.do: age 26-59 (age>25 & age<60), native-born (bpl<57), black/white only
- All ACS years 2000-2017 pooled
- The Stata code uses `reg ... , robust cluster()` which is equivalent to CRV1 clustering in pyfixest
- Birth-state-specific linear cohort trends implemented via pyfixest `i(bpl, cohort)` interaction syntax
- Mean reversion control: pre-cohort average of ln_cpi_income x precohort indicator, following Table 4 Stata code
- Demographic subsample specs (men_only, women_only, white_only, black_only, joint white_men/black_men/white_women) correctly drop collinear demographic FE interactions
- Treatment construction variants M2-M11 use different pre-vaccine averaging windows (Appendix Table 2)
- OPTIMIZATION: selective column loading (12 of 30 columns), chunked row filtering during load
- Full dataset used (no subsampling)
- Per-specification timeout of {SPEC_TIMEOUT_SECONDS}s to avoid runaway computations

## Software Stack
- Python {sys.version.split()[0]}
- pyfixest: {SW_BLOCK['packages'].get('pyfixest', 'N/A')}
- pandas: {SW_BLOCK['packages'].get('pandas', 'N/A')}
- numpy: {SW_BLOCK['packages'].get('numpy', 'N/A')}
"""

# Add failure details if any
if n_fail > 0:
    search_md += "\n## Failed Specifications\n\n"
    for _, row in df_results[df_results['run_success'] == 0].iterrows():
        search_md += f"- **{row['spec_id']}** ({row['spec_run_id']}): {row['run_error']}\n"

if n_infer_fail > 0:
    search_md += "\n## Failed Inference Variants\n\n"
    for _, row in df_infer[df_infer['run_success'] == 0].iterrows():
        search_md += f"- **{row['spec_id']}** ({row['inference_run_id']}): {row['run_error']}\n"

with open(f"{OUTPUT_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write(search_md)
print(f"  SPECIFICATION_SEARCH.md written")

total_time = time.time() - t_start
print(f"\nDone! {len(df_results)} specs + {len(df_infer)} inference variants in {total_time:.0f}s ({total_time/60:.1f}min)")
