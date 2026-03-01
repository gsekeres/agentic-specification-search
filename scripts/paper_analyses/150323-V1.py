#!/usr/bin/env python3
"""
Specification Search Script for Akhtari, Moreira & Trucco (2022)
"Political Turnover, Bureaucratic Turnover, and the Quality of Public Services"
American Economic Review, 112(2), 442-493.

Paper ID: 150323-V1

Surface-driven execution:
  - G1: Test scores ~ political turnover (sharp RD, student-level)
  - G2: Municipal personnel replacement ~ political turnover (sharp RD, municipality-level)
  - G3: Headmaster/teacher replacement ~ political turnover (sharp RD, school-level)
  - All groups: piecewise-linear RD within data-driven bandwidth, cluster(COD_MUNICIPIO)

MEMORY OPTIMIZATIONS:
  - Student-level data (11.5M rows): full dataset, no subsampling
  - Selective column loading for all .dta files
  - Float32 -> Float64 conversion only for regression variables
  - Free data after each group completes

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
import gc
import warnings
warnings.filterwarnings('ignore')

os.environ['PYTHONUNBUFFERED'] = '1'

def log(msg):
    print(msg, flush=True)

sys.path.insert(0, "scripts")
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash, software_block
)

PAPER_ID = "150323-V1"
DATA_DIR = "data/downloads/extracted/150323-V1"
MAIN_DATA_DIR = f"{DATA_DIR}/Data and Code/Data/Main Data"
OUTPUT_DIR = DATA_DIR

SUBSAMPLE_FRAC = 1.0  # Full dataset, no subsampling
SUBSAMPLE_SEED = 150323

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

G1_DESIGN_AUDIT = surface_obj["baseline_groups"][0]["design_audit"]
G2_DESIGN_AUDIT = surface_obj["baseline_groups"][1]["design_audit"]
G3_DESIGN_AUDIT = surface_obj["baseline_groups"][2]["design_audit"]

G1_INFERENCE = surface_obj["baseline_groups"][0]["inference_plan"]["canonical"]
G2_INFERENCE = surface_obj["baseline_groups"][1]["inference_plan"]["canonical"]
G3_INFERENCE = surface_obj["baseline_groups"][2]["inference_plan"]["canonical"]

CONTROLS_SCH = [
    "urban_schl", "waterpblcnetwork_schl", "sewerpblcnetwork_schl",
    "trashcollect_schl", "eqpinternet_schl", "eqpinternet_schl_miss"
]
CONTROLS_STUD = [
    "female_SPB", "female_SPB_miss", "white_SPB", "white_SPB_miss",
    "mom_reads_SPB", "mom_reads_SPB_miss"
]

spec_results = []
inference_results = []
spec_counter = 0
infer_counter = 0


def next_spec_id():
    global spec_counter
    spec_counter += 1
    return f"{PAPER_ID}_spec_{spec_counter:03d}"


def next_infer_id():
    global infer_counter
    infer_counter += 1
    return f"{PAPER_ID}_infer_{infer_counter:03d}"


BW_SUBSAMPLE_N = 100_000  # subsample for bandwidth selection only (rdrobust is ~O(n^2))

def compute_optimal_bw(y, x, cluster=None, covs=None, kernel="uniform"):
    """Compute optimal RD bandwidth using rdrobust.

    Uses a random subsample for bandwidth selection to keep rdrobust tractable
    on large datasets. Regressions still run on full data within the bandwidth.
    """
    from rdrobust import rdbwselect
    # Subsample for bandwidth selection if data is large
    n = len(y)
    if n > BW_SUBSAMPLE_N:
        rng = np.random.default_rng(SUBSAMPLE_SEED)
        idx = rng.choice(n, size=BW_SUBSAMPLE_N, replace=False)
        idx.sort()
        y = y.iloc[idx].reset_index(drop=True)
        x = x.iloc[idx].reset_index(drop=True)
        if cluster is not None:
            cluster = cluster.iloc[idx].reset_index(drop=True)
        if covs is not None:
            covs = covs.iloc[idx].reset_index(drop=True)
        log(f"  (bw selection subsampled to {BW_SUBSAMPLE_N:,} of {n:,} obs)")
    kwargs = dict(kernel=kernel, masspoints="off")
    if cluster is not None:
        kwargs["cluster"] = cluster
    if covs is not None:
        kwargs["covs"] = covs
    result = rdbwselect(y, x, **kwargs)
    bw = result.bws.iloc[0, 0]
    return bw


def run_rd_ols(df, outcome_var, controls, bw, cluster_var="COD_MUNICIPIO",
               treatment_var="pX_dummy", running_var="pX", interaction_var="pX_pD"):
    """Run a sharp RD OLS regression within bandwidth."""
    reg_df = df[df[running_var].abs() <= bw].copy()
    rhs_vars = [treatment_var, running_var, interaction_var] + controls
    all_vars = [outcome_var] + rhs_vars + [cluster_var]
    reg_df = reg_df.dropna(subset=[v for v in all_vars if v in reg_df.columns])
    formula = f"{outcome_var} ~ " + " + ".join(rhs_vars)
    model = pf.feols(formula, data=reg_df, vcov={"CRV1": cluster_var})
    return model


def extract_results(model, treatment_var="pX_dummy"):
    """Extract coefficient, SE, p-value, CI, N, R2 from a pyfixest model."""
    coef = float(model.coef()[treatment_var])
    se = float(model.se()[treatment_var])
    pval = float(model.pvalue()[treatment_var])
    ci = model.confint()
    ci_low = float(ci.loc[treatment_var].iloc[0])
    ci_high = float(ci.loc[treatment_var].iloc[1])
    n_obs = int(model._N)
    r2 = float(model._r2)
    coef_dict = {k: float(v) for k, v in model.coef().items()}
    return coef, se, pval, ci_low, ci_high, n_obs, r2, coef_dict


def make_row(spec_run_id, spec_id, spec_tree_path, baseline_group_id,
             outcome_var, treatment_var, coef, se, pval, ci_low, ci_high,
             n_obs, r2, payload_json, sample_desc, fixed_effects, controls_desc,
             cluster_var, run_success=1, run_error=""):
    return {
        "paper_id": PAPER_ID,
        "spec_run_id": spec_run_id,
        "spec_id": spec_id,
        "spec_tree_path": spec_tree_path,
        "baseline_group_id": baseline_group_id,
        "outcome_var": outcome_var,
        "treatment_var": treatment_var,
        "coefficient": coef,
        "std_error": se,
        "p_value": pval,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "n_obs": n_obs,
        "r_squared": r2,
        "coefficient_vector_json": json.dumps(payload_json),
        "sample_desc": sample_desc,
        "fixed_effects": fixed_effects,
        "controls_desc": controls_desc,
        "cluster_var": cluster_var,
        "run_success": run_success,
        "run_error": run_error,
    }


def make_fail_row(spec_run_id, spec_id, spec_tree_path, baseline_group_id,
                  outcome_var, treatment_var, error_msg, error_det,
                  sample_desc="", controls_desc="", cluster_var="COD_MUNICIPIO"):
    payload = make_failure_payload(
        error=error_msg, error_details=error_det,
        software=SW_BLOCK, surface_hash=SURFACE_HASH,
    )
    return make_row(
        spec_run_id, spec_id, spec_tree_path, baseline_group_id,
        outcome_var, treatment_var,
        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        payload, sample_desc, "", controls_desc, cluster_var,
        run_success=0, run_error=error_msg,
    )


def run_spec(df, outcome_var, controls, bw, spec_id, spec_tree_path,
             baseline_group_id, design_audit, inference_canonical,
             sample_desc="", controls_desc="", cluster_var="COD_MUNICIPIO",
             treatment_var="pX_dummy", running_var="pX", interaction_var="pX_pD",
             extra_design_overrides=None, axis_blocks=None):
    """Run a single RD specification and append to spec_results."""
    run_id = next_spec_id()
    try:
        model = run_rd_ols(df, outcome_var, controls, bw,
                           cluster_var=cluster_var, treatment_var=treatment_var,
                           running_var=running_var, interaction_var=interaction_var)
        coef, se, pval, ci_low, ci_high, n_obs, r2, coef_dict = extract_results(model, treatment_var)

        design_rd = dict(design_audit)
        if extra_design_overrides:
            design_rd.update(extra_design_overrides)

        payload = make_success_payload(
            coefficients=coef_dict,
            inference={"spec_id": inference_canonical["spec_id"],
                       "params": inference_canonical.get("params", {})},
            software=SW_BLOCK, surface_hash=SURFACE_HASH,
            design={"regression_discontinuity": design_rd},
        )
        row = make_row(run_id, spec_id, spec_tree_path, baseline_group_id,
                       outcome_var, treatment_var, coef, se, pval, ci_low, ci_high,
                       n_obs, r2, payload, sample_desc, "", controls_desc, cluster_var)
        spec_results.append(row)
        return model, run_id
    except Exception as e:
        err_msg = str(e)[:240]
        err_det = error_details_from_exception(e, stage="estimation")
        row = make_fail_row(run_id, spec_id, spec_tree_path, baseline_group_id,
                            outcome_var, treatment_var, err_msg, err_det,
                            sample_desc=sample_desc, controls_desc=controls_desc)
        spec_results.append(row)
        return None, run_id


def run_inference_reestimate(df, outcome_var, controls, bw, base_run_id,
                             spec_id, spec_tree_path, baseline_group_id,
                             vcov_spec, treatment_var="pX_dummy",
                             running_var="pX", interaction_var="pX_pD"):
    """Re-estimate the baseline spec with a different variance estimator."""
    infer_id = next_infer_id()
    try:
        reg_df = df[df[running_var].abs() <= bw].copy()
        rhs_vars = [treatment_var, running_var, interaction_var] + controls
        all_vars = [outcome_var] + rhs_vars
        if isinstance(vcov_spec, dict):
            clust_var = list(vcov_spec.values())[0]
            all_vars.append(clust_var)
        reg_df = reg_df.dropna(subset=[v for v in all_vars if v in reg_df.columns])
        formula = f"{outcome_var} ~ " + " + ".join(rhs_vars)
        model = pf.feols(formula, data=reg_df, vcov=vcov_spec)
        coef, se, pval, ci_low, ci_high, n_obs, r2, coef_dict = extract_results(model, treatment_var)
        payload = {
            "coefficients": coef_dict,
            "inference": {"spec_id": spec_id, "vcov": str(vcov_spec)},
            "software": SW_BLOCK, "surface_hash": SURFACE_HASH,
        }
        row = {
            "paper_id": PAPER_ID, "inference_run_id": infer_id,
            "spec_run_id": base_run_id, "spec_id": spec_id,
            "spec_tree_path": spec_tree_path, "baseline_group_id": baseline_group_id,
            "coefficient": coef, "std_error": se, "p_value": pval,
            "ci_lower": ci_low, "ci_upper": ci_high,
            "n_obs": n_obs, "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 1, "run_error": "",
        }
        inference_results.append(row)
        return infer_id
    except Exception as e:
        err_msg = str(e)[:240]
        payload = make_failure_payload(
            error=err_msg, error_details=error_details_from_exception(e, stage="inference"),
            software=SW_BLOCK, surface_hash=SURFACE_HASH,
        )
        row = {
            "paper_id": PAPER_ID, "inference_run_id": infer_id,
            "spec_run_id": base_run_id, "spec_id": spec_id,
            "spec_tree_path": spec_tree_path, "baseline_group_id": baseline_group_id,
            "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "n_obs": np.nan, "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 0, "run_error": err_msg,
        }
        inference_results.append(row)
        return infer_id


# =====================================================
# SAMPLE PREPARATION UTILITIES
# =====================================================
def apply_sample_restrictions(df, add_year_dummy=True):
    """Apply paper's sample restrictions: years 2009/2013, drop supplement/large."""
    df = df[(df["year"] == 2009) | (df["year"] == 2013)].copy()
    df = df[~((df["year"] == 2009) & (df["supplement_2008"] == 1))]
    df = df[~((df["year"] == 2013) & (df["supplement_2012"] == 1))]
    df = df[~((df["year"] == 2009) & (df["population_large"] == 1))]
    df = df[~((df["year"] == 2013) & (df["population_large"] == 1))]
    if add_year_dummy:
        df["year_dummy"] = (df["year"] == 2013).astype(np.float32)
    return df


# =====================================================
# LOAD DATA (with memory optimizations)
# =====================================================

# --- School-level municipal (172MB -> selective columns) ---
log("Loading school-level municipal data (selective columns)...")
schl_munic_cols = [
    "year", "COD_MUNICIPIO", "PK_COD_ENTIDADE",
    "pX", "pX_dummy", "pX_pD",
    "supplement_2008", "supplement_2012", "population_large",
    "IN_CEDocentes",
    "urban_schl", "waterpblcnetwork_schl", "sewerpblcnetwork_schl",
    "trashcollect_schl", "eqpinternet_schl", "eqpinternet_schl_miss",
    "both_score_4_std", "both_score_8_std", "both_score_4_baseline",
    "expthisschl_lessthan2_DPB",
    "newtchr", "lefttchr",
]
schl_munic = pd.read_stata(
    f"{MAIN_DATA_DIR}/s_MainData_SchlLevel2007_2013_MunicSchools.dta",
    columns=schl_munic_cols,
)
log(f"  School-level munic: {schl_munic.shape} ({schl_munic.memory_usage(deep=True).sum()/1e6:.1f} MB)")

# --- School-level non-municipal (95MB -> selective columns) ---
log("Loading school-level non-municipal data (selective columns)...")
schl_nonmunic_cols = [
    "year", "COD_MUNICIPIO", "PK_COD_ENTIDADE",
    "pX", "pX_dummy", "pX_pD",
    "supplement_2008", "supplement_2012", "population_large",
    "IN_CEDocentes",
    "urban_schl", "waterpblcnetwork_schl", "sewerpblcnetwork_schl",
    "trashcollect_schl", "eqpinternet_schl", "eqpinternet_schl_miss",
    "both_score_4_std", "both_score_4_baseline",
    "expthisschl_lessthan2_DPB",
]
schl_nonmunic = pd.read_stata(
    f"{MAIN_DATA_DIR}/s_MainData_SchlLevel2007_2013_NonMunicSchools.dta",
    columns=schl_nonmunic_cols,
)
log(f"  School-level non-munic: {schl_nonmunic.shape} ({schl_nonmunic.memory_usage(deep=True).sum()/1e6:.1f} MB)")

# --- RAIS data (5.7MB, small) ---
log("Loading RAIS data...")
rais = pd.read_stata(f"{MAIN_DATA_DIR}/s_RaisBR.dta")
log(f"  RAIS: {rais.shape}")

# --- Student-level municipal (2.6GB -> selective columns + 15% subsample) ---
log(f"Loading student-level municipal data ({SUBSAMPLE_FRAC*100:.0f}% subsample, seed={SUBSAMPLE_SEED})...")
stdt_cols = [
    "year", "COD_MUNICIPIO", "PK_COD_ENTIDADE",
    "pX", "pX_dummy", "pX_pD",
    "supplement_2008", "supplement_2012", "population_large",
    "both_score_indiv_4_stdComb", "both_score_indiv_8_stdComb",
    "both_score_indiv_4_std08", "both_score_indiv_4_std12",
    "both_score_4_baseline", "both_score_8_baseline",
    "urban_schl", "waterpblcnetwork_schl", "sewerpblcnetwork_schl",
    "trashcollect_schl", "eqpinternet_schl", "eqpinternet_schl_miss",
    "female_SPB", "female_SPB_miss", "white_SPB", "white_SPB_miss",
    "mom_reads_SPB", "mom_reads_SPB_miss",
]

# Load in chunks and subsample to keep memory low
log("  Reading in chunks and subsampling...")
chunks = []
reader = pd.read_stata(
    f"{MAIN_DATA_DIR}/s_MainData_StdtLevel2007_2013_MunicSchools.dta",
    columns=stdt_cols,
    chunksize=500_000,
)
rng_sub = np.random.RandomState(SUBSAMPLE_SEED)
chunk_idx = 0
for chunk in reader:
    chunk_idx += 1
    mask = rng_sub.random(len(chunk)) < SUBSAMPLE_FRAC
    chunks.append(chunk[mask])
    if chunk_idx % 5 == 0:
        log(f"    Processed {chunk_idx} chunks, kept {sum(len(c) for c in chunks)} rows so far...")
    del chunk, mask
    gc.collect()

stdt_munic = pd.concat(chunks, ignore_index=True)
del chunks
gc.collect()
log(f"  Student-level munic subsample: {stdt_munic.shape} ({stdt_munic.memory_usage(deep=True).sum()/1e6:.1f} MB)")


# =====================================================
# SAMPLE PREPARATION
# =====================================================
log("Preparing G1 student-level data...")
g1_munic = apply_sample_restrictions(stdt_munic)
g1_munic = g1_munic[g1_munic["urban_schl"].notna()].copy()
log(f"  G1 munic sample: {g1_munic.shape}")
del stdt_munic
gc.collect()

log("Preparing G1 nonmunic placebo (school-level)...")
g1_nonmunic_schl = apply_sample_restrictions(schl_nonmunic)
g1_nonmunic_schl = g1_nonmunic_schl[g1_nonmunic_schl["urban_schl"].notna()].copy()
log(f"  G1 nonmunic (school-level): {g1_nonmunic_schl.shape}")


# G2: Municipality-level personnel
log("Preparing G2 municipality-level data...")
def prepare_g2():
    df = schl_munic.copy()
    cnt = df.groupby("PK_COD_ENTIDADE")["IN_CEDocentes"].transform("count")
    df = df[cnt == 4].copy()
    df["munic_id"] = df["PK_COD_ENTIDADE"].astype(str).str[:6].astype(float)

    rais_cols_needed = ["munic_id"]
    for suffix in ["09", "10", "13", "14"]:
        for prefix in ["SHhired_Mun_", "SHfired_Mun_", "SHNet_Mun_",
                        "Qhired_Mun_", "Qfired_Mun_", "QNet_Mun_"]:
            col = f"{prefix}{suffix}"
            if col in rais.columns:
                rais_cols_needed.append(col)
    rais_sub = rais[rais_cols_needed].drop_duplicates()
    df = df.merge(rais_sub, on="munic_id", how="left")

    for var in ["hired", "fired", "Net"]:
        df[f"SH{var}_Mun_lead"] = np.nan
        if f"SH{var}_Mun_09" in df.columns:
            df.loc[df["year"] == 2009, f"SH{var}_Mun_lead"] = df.loc[df["year"] == 2009, f"SH{var}_Mun_09"].values
        if f"SH{var}_Mun_13" in df.columns:
            df.loc[df["year"] == 2013, f"SH{var}_Mun_lead"] = df.loc[df["year"] == 2013, f"SH{var}_Mun_13"].values
        df[f"SH{var}_Mun_after"] = np.nan
        if f"SH{var}_Mun_10" in df.columns:
            df.loc[df["year"] == 2009, f"SH{var}_Mun_after"] = df.loc[df["year"] == 2009, f"SH{var}_Mun_10"].values
        if f"SH{var}_Mun_14" in df.columns:
            df.loc[df["year"] == 2013, f"SH{var}_Mun_after"] = df.loc[df["year"] == 2013, f"SH{var}_Mun_14"].values

    df = df.sort_values(["COD_MUNICIPIO", "year"])
    df = df[~df.duplicated(subset=["COD_MUNICIPIO", "year"])].copy()
    df["year_dummy"] = (df["year"] >= 2013).astype(float)
    df["election_cycle"] = np.where(df["year"] <= 2012, 1, 2)
    for var in ["pX_dummy", "pX", "pX_pD", "supplement_2008", "supplement_2012", "population_large"]:
        if var in df.columns:
            df[var] = df.groupby(["COD_MUNICIPIO", "election_cycle"])[var].transform("min")

    df["sample"] = 0
    for yr_suffix in ["09", "13"]:
        mask = True
        for prefix in ["Qfired_Mun_", "Qhired_Mun_", "QNet_Mun_",
                        "SHfired_Mun_", "SHhired_Mun_", "SHNet_Mun_"]:
            col = f"{prefix}{yr_suffix}"
            if col in df.columns:
                mask = mask & df[col].notna()
        df.loc[mask, "sample"] = 1
    df.loc[df["SHfired_Mun_lead"].isna() & df["SHhired_Mun_lead"].isna() & df["SHNet_Mun_lead"].isna(), "sample"] = 0

    df = df[(df["year"] == 2009) | (df["year"] == 2013)].copy()
    df = df[~((df["year"] == 2009) & (df["supplement_2008"] == 1))]
    df = df[~((df["year"] == 2013) & (df["supplement_2012"] == 1))]
    df = df[~((df["year"] == 2009) & (df["population_large"] == 1))]
    df = df[~((df["year"] == 2013) & (df["population_large"] == 1))]
    return df

g2_data = prepare_g2()
g2_sample = g2_data[g2_data["sample"] == 1].copy()
log(f"  G2 shape: {g2_data.shape}, sample==1: {len(g2_sample)}")
del g2_data
gc.collect()


# G3: School-level headmaster/teacher
log("Preparing G3 school-level data...")
def prepare_schl(df, require_docentes=False):
    df = df.copy()
    if require_docentes:
        cnt = df.groupby("PK_COD_ENTIDADE")["IN_CEDocentes"].transform("count")
        df = df[cnt == 4].copy()
    return apply_sample_restrictions(df)

g3_munic = prepare_schl(schl_munic)
g3_munic_f = g3_munic[g3_munic["urban_schl"].notna()].copy()
log(f"  G3 munic: {g3_munic_f.shape}")

g3_nonmunic = prepare_schl(schl_nonmunic)
g3_nonmunic_f = g3_nonmunic[g3_nonmunic["urban_schl"].notna()].copy()

g3_munic_tchr = prepare_schl(schl_munic, require_docentes=True)
g3_munic_tchr_f = g3_munic_tchr[g3_munic_tchr["urban_schl"].notna()].copy()

# Free raw data
del schl_munic, schl_nonmunic, rais, g3_munic, g3_nonmunic, g3_munic_tchr
gc.collect()
log("Data preparation complete, raw data freed.")


# =====================================================
# G1: TEST SCORES
# =====================================================
log("\n" + "="*60)
log("GROUP G1: TEST SCORES (full student-level data)")
log("="*60)

log("Computing optimal bandwidths...")
g1_bw4 = compute_optimal_bw(
    g1_munic["both_score_indiv_4_stdComb"].dropna(),
    g1_munic.loc[g1_munic["both_score_indiv_4_stdComb"].notna(), "pX"],
    cluster=g1_munic.loc[g1_munic["both_score_indiv_4_stdComb"].notna(), "COD_MUNICIPIO"],
    covs=g1_munic.loc[g1_munic["both_score_indiv_4_stdComb"].notna(), ["both_score_4_baseline"]],
    kernel="uniform",
)
log(f"  G1 optimal BW 4th: {g1_bw4:.5f}")

g1_bw8 = compute_optimal_bw(
    g1_munic["both_score_indiv_8_stdComb"].dropna(),
    g1_munic.loc[g1_munic["both_score_indiv_8_stdComb"].notna(), "pX"],
    cluster=g1_munic.loc[g1_munic["both_score_indiv_8_stdComb"].notna(), "COD_MUNICIPIO"],
    covs=g1_munic.loc[g1_munic["both_score_indiv_8_stdComb"].notna(), ["both_score_8_baseline"]],
    kernel="uniform",
)
log(f"  G1 optimal BW 8th: {g1_bw8:.5f}")

base4 = ["pX", "pX_pD", "both_score_4_baseline"]

def g1_run(spec_id, tree, bw, controls, outcome="both_score_indiv_4_stdComb",
           data=None, sdesc="", cdesc="", dov=None, ablocks=None):
    if data is None:
        data = g1_munic
    return run_spec(data, outcome, controls, bw, spec_id, tree,
                    "G1", G1_DESIGN_AUDIT, G1_INFERENCE,
                    sample_desc=sdesc, controls_desc=cdesc,
                    extra_design_overrides=dov, axis_blocks=ablocks)

# --- G1 Baselines ---
log("G1 baselines...")
_, g1_base_id = g1_run("baseline",
    "specification_tree/designs/regression_discontinuity.md",
    g1_bw4, base4, sdesc="Munic, 4th, stacked, optimal bw (subsample)", cdesc="pX pX_pD baseline_score")

g1_run("baseline__4th_grade_with_controls",
    "specification_tree/designs/regression_discontinuity.md",
    g1_bw4, base4 + ["year_dummy"] + CONTROLS_SCH + CONTROLS_STUD,
    sdesc="Munic, 4th, all controls (subsample)", cdesc="pX pX_pD baseline year_dummy sch stud")

g1_run("baseline__8th_grade_no_controls",
    "specification_tree/designs/regression_discontinuity.md",
    g1_bw8, ["pX", "pX_pD", "both_score_8_baseline"],
    outcome="both_score_indiv_8_stdComb",
    sdesc="Munic, 8th, optimal bw (subsample)", cdesc="pX pX_pD baseline_score_8")

g1_run("baseline__8th_grade_with_controls",
    "specification_tree/designs/regression_discontinuity.md",
    g1_bw8, ["pX", "pX_pD", "both_score_8_baseline", "year_dummy"] + CONTROLS_SCH + CONTROLS_STUD,
    outcome="both_score_indiv_8_stdComb",
    sdesc="Munic, 8th, all controls (subsample)", cdesc="pX pX_pD baseline_8 year_dummy sch stud")

# --- G1 Design Variants ---
log("G1 design variants...")

g1_run("design/regression_discontinuity/bandwidth/half_baseline",
    "specification_tree/designs/regression_discontinuity.md#bandwidth",
    g1_bw4/2, base4, sdesc="Munic, 4th, half bw", cdesc="pX pX_pD baseline",
    dov={"bandwidth": f"half ({g1_bw4/2:.5f})"})

g1_run("design/regression_discontinuity/bandwidth/double_baseline",
    "specification_tree/designs/regression_discontinuity.md#bandwidth",
    g1_bw4*2, base4, sdesc="Munic, 4th, double bw", cdesc="pX pX_pD baseline",
    dov={"bandwidth": f"double ({g1_bw4*2:.5f})"})

g1_run("design/regression_discontinuity/bandwidth/fixed_small",
    "specification_tree/designs/regression_discontinuity.md#bandwidth",
    0.07, base4, sdesc="Munic, 4th, bw=0.07", cdesc="pX pX_pD baseline",
    dov={"bandwidth": "0.07"})

g1_run("design/regression_discontinuity/bandwidth/fixed_large",
    "specification_tree/designs/regression_discontinuity.md#bandwidth",
    0.11, base4, sdesc="Munic, 4th, bw=0.11", cdesc="pX pX_pD baseline",
    dov={"bandwidth": "0.11"})

# Quadratic polynomial
g1_munic["pX_sq"] = g1_munic["pX"] ** 2
g1_munic["pX_sq_pD"] = g1_munic["pX_sq"] * g1_munic["pX_dummy"]
g1_run("design/regression_discontinuity/poly/local_quadratic",
    "specification_tree/designs/regression_discontinuity.md#polynomial",
    g1_bw4, base4 + ["pX_sq", "pX_sq_pD"],
    sdesc="Munic, 4th, quadratic", cdesc="pX pX_pD pX_sq pX_sq_pD baseline",
    dov={"poly_order": 2})

# Triangular kernel
log("G1 triangular kernel...")
try:
    g1_bw4_tri = compute_optimal_bw(
        g1_munic["both_score_indiv_4_stdComb"].dropna(),
        g1_munic.loc[g1_munic["both_score_indiv_4_stdComb"].notna(), "pX"],
        cluster=g1_munic.loc[g1_munic["both_score_indiv_4_stdComb"].notna(), "COD_MUNICIPIO"],
        covs=g1_munic.loc[g1_munic["both_score_indiv_4_stdComb"].notna(), ["both_score_4_baseline"]],
        kernel="triangular",
    )
    g1_munic["tri_weight"] = np.where(g1_munic["pX"].abs() <= g1_bw4_tri,
                                       1 - g1_munic["pX"].abs()/g1_bw4_tri, 0)
    reg_df = g1_munic[(g1_munic["pX"].abs() <= g1_bw4_tri) & (g1_munic["tri_weight"] > 0)].copy()
    rhs = ["pX_dummy", "pX", "pX_pD", "both_score_4_baseline"]
    reg_df = reg_df.dropna(subset=["both_score_indiv_4_stdComb"] + rhs + ["COD_MUNICIPIO"])
    model_tri = pf.feols("both_score_indiv_4_stdComb ~ " + " + ".join(rhs),
                          data=reg_df, vcov={"CRV1": "COD_MUNICIPIO"}, weights="tri_weight")
    coef, se, pval, ci_low, ci_high, n_obs, r2, coef_dict = extract_results(model_tri)
    run_id = next_spec_id()
    payload = make_success_payload(
        coefficients=coef_dict,
        inference={"spec_id": G1_INFERENCE["spec_id"], "params": G1_INFERENCE.get("params", {})},
        software=SW_BLOCK, surface_hash=SURFACE_HASH,
        design={"regression_discontinuity": dict(G1_DESIGN_AUDIT, kernel="triangular", bandwidth=f"{g1_bw4_tri:.5f}")},
    )
    spec_results.append(make_row(run_id, "design/regression_discontinuity/kernel/triangular",
        "specification_tree/designs/regression_discontinuity.md#kernel",
        "G1", "both_score_indiv_4_stdComb", "pX_dummy",
        coef, se, pval, ci_low, ci_high, n_obs, r2, payload,
        f"Munic, 4th, triangular bw={g1_bw4_tri:.5f} (subsample)", "", "pX pX_pD baseline", "COD_MUNICIPIO"))
except Exception as e:
    run_id = next_spec_id()
    spec_results.append(make_fail_row(run_id, "design/regression_discontinuity/kernel/triangular",
        "specification_tree/designs/regression_discontinuity.md#kernel",
        "G1", "both_score_indiv_4_stdComb", "pX_dummy",
        str(e)[:240], error_details_from_exception(e, stage="estimation")))

# --- G1 RC Variants ---
log("G1 RC variants (controls)...")

g1_run("rc/controls/sets/no_controls",
    "specification_tree/modules/robustness/controls.md#sets",
    g1_bw4, ["pX", "pX_pD"], sdesc="Munic, 4th, no controls", cdesc="pX pX_pD",
    ablocks={"controls": {"spec_id": "rc/controls/sets/no_controls", "family": "sets", "n_controls": 0}})

g1_run("rc/controls/sets/with_school_controls",
    "specification_tree/modules/robustness/controls.md#sets",
    g1_bw4, base4 + CONTROLS_SCH, sdesc="Munic, 4th, school controls", cdesc="pX pX_pD baseline sch",
    ablocks={"controls": {"spec_id": "rc/controls/sets/with_school_controls", "family": "sets"}})

g1_run("rc/controls/sets/with_student_controls",
    "specification_tree/modules/robustness/controls.md#sets",
    g1_bw4, base4 + CONTROLS_STUD, sdesc="Munic, 4th, student controls", cdesc="pX pX_pD baseline stud",
    ablocks={"controls": {"spec_id": "rc/controls/sets/with_student_controls", "family": "sets"}})

g1_run("rc/controls/sets/with_all_controls",
    "specification_tree/modules/robustness/controls.md#sets",
    g1_bw4, base4 + ["year_dummy"] + CONTROLS_SCH + CONTROLS_STUD,
    sdesc="Munic, 4th, all controls", cdesc="pX pX_pD baseline year sch stud",
    ablocks={"controls": {"spec_id": "rc/controls/sets/with_all_controls", "family": "sets"}})

g1_run("rc/controls/add/year_dummy",
    "specification_tree/modules/robustness/controls.md#add",
    g1_bw4, base4 + ["year_dummy"], sdesc="Munic, 4th, +year_dummy", cdesc="pX pX_pD baseline year_dummy",
    ablocks={"controls": {"spec_id": "rc/controls/add/year_dummy", "family": "add", "added": ["year_dummy"]}})

g1_run("rc/controls/loo/drop_baseline_score",
    "specification_tree/modules/robustness/controls.md#loo",
    g1_bw4, ["pX", "pX_pD"], sdesc="Munic, 4th, no baseline score", cdesc="pX pX_pD",
    ablocks={"controls": {"spec_id": "rc/controls/loo/drop_baseline_score", "family": "loo", "dropped": ["both_score_4_baseline"]}})

# Subgroups
log("G1 subgroups...")
g1_run("rc/sample/subgroup/2008_cycle_only",
    "specification_tree/modules/robustness/sample.md#subgroup",
    g1_bw4, base4, data=g1_munic[g1_munic["year"]==2009].copy(),
    sdesc="Munic, 4th, 2008 cycle", cdesc="pX pX_pD baseline",
    ablocks={"sample": {"spec_id": "rc/sample/subgroup/2008_cycle_only", "restriction": "year==2009"}})

g1_run("rc/sample/subgroup/2012_cycle_only",
    "specification_tree/modules/robustness/sample.md#subgroup",
    g1_bw4, base4, data=g1_munic[g1_munic["year"]==2013].copy(),
    sdesc="Munic, 4th, 2012 cycle", cdesc="pX pX_pD baseline",
    ablocks={"sample": {"spec_id": "rc/sample/subgroup/2012_cycle_only", "restriction": "year==2013"}})

g1_run("rc/sample/subgroup/urban_schools_only",
    "specification_tree/modules/robustness/sample.md#subgroup",
    g1_bw4, base4, data=g1_munic[g1_munic["urban_schl"]==1].copy(),
    sdesc="Munic, 4th, urban only", cdesc="pX pX_pD baseline",
    ablocks={"sample": {"spec_id": "rc/sample/subgroup/urban_schools_only"}})

g1_run("rc/sample/subgroup/rural_schools_only",
    "specification_tree/modules/robustness/sample.md#subgroup",
    g1_bw4, base4, data=g1_munic[g1_munic["urban_schl"]==0].copy(),
    sdesc="Munic, 4th, rural only", cdesc="pX pX_pD baseline",
    ablocks={"sample": {"spec_id": "rc/sample/subgroup/rural_schools_only"}})

# Nonmunic placebo (school-level data)
log("G1 nonmunic placebo (school-level)...")
try:
    bw_nm = compute_optimal_bw(
        g1_nonmunic_schl["both_score_4_std"].dropna(),
        g1_nonmunic_schl.loc[g1_nonmunic_schl["both_score_4_std"].notna(), "pX"],
        cluster=g1_nonmunic_schl.loc[g1_nonmunic_schl["both_score_4_std"].notna(), "COD_MUNICIPIO"],
        covs=g1_nonmunic_schl.loc[g1_nonmunic_schl["both_score_4_std"].notna(), ["both_score_4_baseline"]],
        kernel="uniform",
    )
except Exception:
    bw_nm = g1_bw4
g1_run("rc/sample/subgroup/nonmunic_schools",
    "specification_tree/modules/robustness/sample.md#subgroup",
    bw_nm, ["pX", "pX_pD", "both_score_4_baseline"],
    outcome="both_score_4_std", data=g1_nonmunic_schl,
    sdesc="NonMunic, 4th, school-level, placebo", cdesc="pX pX_pD baseline",
    ablocks={"sample": {"spec_id": "rc/sample/subgroup/nonmunic_schools", "restriction": "non-municipal schools (school-level)"}})

# Donut holes
log("G1 donut and trim variants...")
g1_run("rc/sample/restriction/donut_1pct",
    "specification_tree/modules/robustness/sample.md#restriction",
    g1_bw4, base4, data=g1_munic[g1_munic["pX"].abs()>=0.01].copy(),
    sdesc="Munic, 4th, donut 1pct", cdesc="pX pX_pD baseline",
    ablocks={"sample": {"spec_id": "rc/sample/restriction/donut_1pct", "restriction": "abs(pX)>=0.01"}})

g1_run("rc/sample/restriction/donut_2pct",
    "specification_tree/modules/robustness/sample.md#restriction",
    g1_bw4, base4, data=g1_munic[g1_munic["pX"].abs()>=0.02].copy(),
    sdesc="Munic, 4th, donut 2pct", cdesc="pX pX_pD baseline",
    ablocks={"sample": {"spec_id": "rc/sample/restriction/donut_2pct", "restriction": "abs(pX)>=0.02"}})

g1_run("rc/sample/outliers/trim_pX_narrow",
    "specification_tree/modules/robustness/sample.md#outliers",
    min(g1_bw4, 0.05), base4, sdesc="Munic, 4th, narrow trim", cdesc="pX pX_pD baseline",
    ablocks={"sample": {"spec_id": "rc/sample/outliers/trim_pX_narrow"}})

# Outcome variants
log("G1 outcome variants...")
for ov_spec, ov_label in [("rc/form/outcome/math_score_only", "math"), ("rc/form/outcome/port_score_only", "port")]:
    run_id = next_spec_id()
    err_msg = "Student data has only combined scores; separate math/Portuguese unavailable"
    spec_results.append(make_fail_row(run_id, ov_spec,
        "specification_tree/modules/robustness/functional_form.md#outcome",
        "G1", "both_score_indiv_4_stdComb", "pX_dummy", err_msg,
        {"stage": "data_check", "exception_type": "DataNotAvailable", "exception_message": err_msg}))

# Unstandardized: use year-specific std (std08/std12) as alternative
g1_munic["both_score_indiv_4_unstd"] = np.where(
    g1_munic["year"]==2009, g1_munic["both_score_indiv_4_std08"], g1_munic["both_score_indiv_4_std12"])
g1_run("rc/form/outcome/unstandardized_score",
    "specification_tree/modules/robustness/functional_form.md#outcome",
    g1_bw4, ["pX", "pX_pD", "both_score_4_baseline"],
    outcome="both_score_indiv_4_unstd",
    sdesc="Munic, 4th, year-specific std", cdesc="pX pX_pD baseline",
    ablocks={"functional_form": {"spec_id": "rc/form/outcome/unstandardized_score",
                                  "interpretation": "Year-specific standardized instead of combined"}})

# mserd bandwidth selector
log("G1 mserd bandwidth...")
try:
    from rdrobust import rdbwselect
    y_bw = g1_munic["both_score_indiv_4_stdComb"].dropna()
    mask_bw = g1_munic["both_score_indiv_4_stdComb"].notna()
    x_bw = g1_munic.loc[mask_bw, "pX"]
    cl_bw = g1_munic.loc[mask_bw, "COD_MUNICIPIO"]
    cv_bw = g1_munic.loc[mask_bw, ["both_score_4_baseline"]]
    # Subsample for tractability
    if len(y_bw) > BW_SUBSAMPLE_N:
        rng_bw = np.random.default_rng(SUBSAMPLE_SEED + 1)
        idx_bw = rng_bw.choice(len(y_bw), size=min(BW_SUBSAMPLE_N, len(y_bw)), replace=False)
        idx_bw.sort()
        y_bw = y_bw.iloc[idx_bw].reset_index(drop=True)
        x_bw = x_bw.iloc[idx_bw].reset_index(drop=True)
        cl_bw = cl_bw.iloc[idx_bw].reset_index(drop=True)
        cv_bw = cv_bw.iloc[idx_bw].reset_index(drop=True)
    res_bw = rdbwselect(y_bw, x_bw, kernel="uniform",
                        cluster=cl_bw, covs=cv_bw,
                        bwselect="mserd", masspoints="off")
    bw_mserd = res_bw.bws.iloc[0, 0]
    g1_run("rc/bandwidth/optimal_mserd",
        "specification_tree/designs/regression_discontinuity.md#bandwidth",
        bw_mserd, base4, sdesc=f"Munic, 4th, mserd bw={bw_mserd:.5f}", cdesc="pX pX_pD baseline",
        dov={"bandwidth_rule": "mserd"})
except Exception as e:
    run_id = next_spec_id()
    spec_results.append(make_fail_row(run_id, "rc/bandwidth/optimal_mserd",
        "specification_tree/designs/regression_discontinuity.md#bandwidth",
        "G1", "both_score_indiv_4_stdComb", "pX_dummy",
        str(e)[:240], error_details_from_exception(e, stage="bandwidth_selection")))

# Cross-product: 8th grade at fixed bandwidths
log("G1 cross-product specs...")
for bw_val, bw_lbl in [(0.07, "bw07"), (0.11, "bw11")]:
    g1_run(f"rc/bandwidth_x_grade/8th_{bw_lbl}_no_controls",
        "specification_tree/designs/regression_discontinuity.md#bandwidth",
        bw_val, ["pX", "pX_pD", "both_score_8_baseline"],
        outcome="both_score_indiv_8_stdComb",
        sdesc=f"Munic, 8th, bw={bw_val}", cdesc="pX pX_pD baseline_8",
        dov={"bandwidth": str(bw_val)})
    g1_run(f"rc/bandwidth_x_grade/8th_{bw_lbl}_all_controls",
        "specification_tree/designs/regression_discontinuity.md#bandwidth",
        bw_val, ["pX", "pX_pD", "both_score_8_baseline", "year_dummy"] + CONTROLS_SCH + CONTROLS_STUD,
        outcome="both_score_indiv_8_stdComb",
        sdesc=f"Munic, 8th, bw={bw_val}, all ctrl", cdesc="pX pX_pD baseline_8 year sch stud",
        dov={"bandwidth": str(bw_val)})

for bw_val, bw_lbl in [(0.07, "bw07"), (0.11, "bw11")]:
    g1_run(f"rc/bandwidth_x_controls/4th_{bw_lbl}_all_controls",
        "specification_tree/designs/regression_discontinuity.md#bandwidth",
        bw_val, base4 + ["year_dummy"] + CONTROLS_SCH + CONTROLS_STUD,
        sdesc=f"Munic, 4th, bw={bw_val}, all ctrl", cdesc="pX pX_pD baseline year sch stud",
        dov={"bandwidth": str(bw_val)})

# G1 INFERENCE VARIANTS
log("G1 inference variants...")
run_inference_reestimate(
    g1_munic, "both_score_indiv_4_stdComb", base4, g1_bw4, g1_base_id,
    "infer/se/hc/hc1", "specification_tree/modules/inference/se.md#hc", "G1", "hetero")

g1_munic["state"] = g1_munic["COD_MUNICIPIO"].astype(str).str[:2]
run_inference_reestimate(
    g1_munic, "both_score_indiv_4_stdComb", base4, g1_bw4, g1_base_id,
    "infer/se/cluster/state", "specification_tree/modules/inference/se.md#cluster", "G1", {"CRV1": "state"})

g1_count = sum(1 for r in spec_results if r["baseline_group_id"] == "G1")
log(f"  G1 complete: {g1_count} specs")

# Free G1 data
del g1_munic, g1_nonmunic_schl
gc.collect()


# =====================================================
# G2: MUNICIPAL PERSONNEL
# =====================================================
log("\n" + "="*60)
log("GROUP G2: MUNICIPAL PERSONNEL")
log("="*60)

log("Computing G2 bandwidths...")
g2_bw = compute_optimal_bw(
    g2_sample["SHhired_Mun_lead"].dropna(),
    g2_sample.loc[g2_sample["SHhired_Mun_lead"].notna(), "pX"],
    cluster=g2_sample.loc[g2_sample["SHhired_Mun_lead"].notna(), "COD_MUNICIPIO"],
    kernel="uniform",
)
log(f"  G2 optimal BW (hired): {g2_bw:.5f}")

g2_ctrl = ["pX", "pX_pD", "year_dummy"]

def g2_run(spec_id, tree, bw, outcome="SHhired_Mun_lead", controls=None,
           data=None, sdesc="", cdesc="", dov=None, ablocks=None):
    if data is None: data = g2_sample
    if controls is None: controls = g2_ctrl
    return run_spec(data, outcome, controls, bw, spec_id, tree,
                    "G2", G2_DESIGN_AUDIT, G2_INFERENCE,
                    sample_desc=sdesc, controls_desc=cdesc,
                    extra_design_overrides=dov, axis_blocks=ablocks)

log("G2 baselines...")
_, g2_base_id = g2_run("baseline",
    "specification_tree/designs/regression_discontinuity.md",
    g2_bw, sdesc="Munic personnel, hired, stacked, optimal bw", cdesc="pX pX_pD year_dummy")

# Additional baselines
for outcome, label, bw_out in [
    ("SHfired_Mun_lead", "fired", None),
    ("SHhired_Mun_after", "hired_after", None),
    ("SHfired_Mun_after", "fired_after", None),
]:
    try:
        bw_out = compute_optimal_bw(
            g2_sample[outcome].dropna(),
            g2_sample.loc[g2_sample[outcome].notna(), "pX"],
            cluster=g2_sample.loc[g2_sample[outcome].notna(), "COD_MUNICIPIO"],
            kernel="uniform",
        )
    except Exception:
        bw_out = g2_bw
    g2_run(f"baseline__personnel_{label}",
        "specification_tree/designs/regression_discontinuity.md",
        bw_out, outcome=outcome, sdesc=f"Munic personnel {label}, stacked", cdesc="pX pX_pD year_dummy")

# Design variants
log("G2 design variants...")
for sid, bw in [
    ("design/regression_discontinuity/bandwidth/half_baseline", g2_bw/2),
    ("design/regression_discontinuity/bandwidth/double_baseline", g2_bw*2),
    ("design/regression_discontinuity/bandwidth/fixed_small", 0.07),
    ("design/regression_discontinuity/bandwidth/fixed_large", 0.11),
]:
    g2_run(sid, "specification_tree/designs/regression_discontinuity.md#bandwidth",
        bw, sdesc=f"Personnel, bw={bw:.4f}", cdesc="pX pX_pD year_dummy",
        dov={"bandwidth": f"{bw:.4f}"})

# RC variants
log("G2 RC variants...")
g2_run("rc/bandwidth/fixed_07", "specification_tree/designs/regression_discontinuity.md#bandwidth",
    0.07, sdesc="Personnel, bw=0.07", cdesc="pX pX_pD year_dummy")
g2_run("rc/bandwidth/fixed_11", "specification_tree/designs/regression_discontinuity.md#bandwidth",
    0.11, sdesc="Personnel, bw=0.11", cdesc="pX pX_pD year_dummy")
g2_run("rc/bandwidth/half_optimal", "specification_tree/designs/regression_discontinuity.md#bandwidth",
    g2_bw/2, sdesc="Personnel, half bw", cdesc="pX pX_pD year_dummy")
g2_run("rc/bandwidth/double_optimal", "specification_tree/designs/regression_discontinuity.md#bandwidth",
    g2_bw*2, sdesc="Personnel, double bw", cdesc="pX pX_pD year_dummy")

g2_run("rc/sample/subgroup/2008_cycle_only",
    "specification_tree/modules/robustness/sample.md#subgroup",
    g2_bw, controls=["pX", "pX_pD"], data=g2_sample[g2_sample["year"]==2009].copy(),
    sdesc="Personnel, 2008 cycle", cdesc="pX pX_pD",
    ablocks={"sample": {"spec_id": "rc/sample/subgroup/2008_cycle_only"}})

g2_run("rc/sample/subgroup/2012_cycle_only",
    "specification_tree/modules/robustness/sample.md#subgroup",
    g2_bw, controls=["pX", "pX_pD"], data=g2_sample[g2_sample["year"]==2013].copy(),
    sdesc="Personnel, 2012 cycle", cdesc="pX pX_pD",
    ablocks={"sample": {"spec_id": "rc/sample/subgroup/2012_cycle_only"}})

g2_run("rc/sample/restriction/donut_1pct",
    "specification_tree/modules/robustness/sample.md#restriction",
    g2_bw, data=g2_sample[g2_sample["pX"].abs()>=0.01].copy(),
    sdesc="Personnel, donut 1pct", cdesc="pX pX_pD year_dummy",
    ablocks={"sample": {"spec_id": "rc/sample/restriction/donut_1pct"}})

g2_run("rc/sample/restriction/donut_2pct",
    "specification_tree/modules/robustness/sample.md#restriction",
    g2_bw, data=g2_sample[g2_sample["pX"].abs()>=0.02].copy(),
    sdesc="Personnel, donut 2pct", cdesc="pX pX_pD year_dummy",
    ablocks={"sample": {"spec_id": "rc/sample/restriction/donut_2pct"}})

g2_sample = g2_sample.copy()
g2_sample["pX_sq"] = g2_sample["pX"]**2
g2_sample["pX_sq_pD"] = g2_sample["pX_sq"] * g2_sample["pX_dummy"]
g2_run("rc/poly/quadratic",
    "specification_tree/designs/regression_discontinuity.md#polynomial",
    g2_bw, controls=["pX", "pX_pD", "pX_sq", "pX_sq_pD", "year_dummy"],
    sdesc="Personnel, quadratic", cdesc="pX pX_pD pX_sq pX_sq_pD year_dummy",
    dov={"poly_order": 2})

g2_run("rc/form/outcome/net_personnel_change",
    "specification_tree/modules/robustness/functional_form.md#outcome",
    g2_bw, outcome="SHNet_Mun_lead", sdesc="Personnel, net change", cdesc="pX pX_pD year_dummy",
    ablocks={"functional_form": {"spec_id": "rc/form/outcome/net_personnel_change",
                                  "interpretation": "Net personnel share (hired-fired)"}})

g2_sample["log_SHhired_Mun_lead"] = np.log(g2_sample["SHhired_Mun_lead"].clip(lower=1e-6))
g2_run("rc/form/outcome/log_new_hires",
    "specification_tree/modules/robustness/functional_form.md#outcome",
    g2_bw, outcome="log_SHhired_Mun_lead", sdesc="Personnel, log hires", cdesc="pX pX_pD year_dummy",
    ablocks={"functional_form": {"spec_id": "rc/form/outcome/log_new_hires",
                                  "interpretation": "Log of new personnel share"}})

# Fired at fixed bw
for bw_val, bw_lbl in [(0.07, "bw07"), (0.11, "bw11")]:
    g2_run(f"rc/fired_x_bw/{bw_lbl}",
        "specification_tree/designs/regression_discontinuity.md#bandwidth",
        bw_val, outcome="SHfired_Mun_lead",
        sdesc=f"Personnel fired, bw={bw_val}", cdesc="pX pX_pD year_dummy")

g2_count = sum(1 for r in spec_results if r["baseline_group_id"] == "G2")
log(f"  G2 complete: {g2_count} specs")

# Free G2 data
del g2_sample
gc.collect()


# =====================================================
# G3: HEADMASTER / TEACHER
# =====================================================
log("\n" + "="*60)
log("GROUP G3: HEADMASTER / TEACHER")
log("="*60)

log("Computing G3 bandwidths...")
g3_bw = compute_optimal_bw(
    g3_munic_f["expthisschl_lessthan2_DPB"].dropna(),
    g3_munic_f.loc[g3_munic_f["expthisschl_lessthan2_DPB"].notna(), "pX"],
    cluster=g3_munic_f.loc[g3_munic_f["expthisschl_lessthan2_DPB"].notna(), "COD_MUNICIPIO"],
    kernel="uniform",
)
log(f"  G3 optimal BW (headmaster): {g3_bw:.5f}")

def g3_run(spec_id, tree, bw, outcome="expthisschl_lessthan2_DPB", controls=None,
           data=None, sdesc="", cdesc="", dov=None, ablocks=None):
    if data is None: data = g3_munic_f
    if controls is None: controls = ["pX", "pX_pD"]
    return run_spec(data, outcome, controls, bw, spec_id, tree,
                    "G3", G3_DESIGN_AUDIT, G3_INFERENCE,
                    sample_desc=sdesc, controls_desc=cdesc,
                    extra_design_overrides=dov, axis_blocks=ablocks)

log("G3 baselines...")
_, g3_base_id = g3_run("baseline",
    "specification_tree/designs/regression_discontinuity.md",
    g3_bw, sdesc="Munic, headmaster, stacked, optimal bw", cdesc="pX pX_pD")

g3_run("baseline__headmaster_with_controls",
    "specification_tree/designs/regression_discontinuity.md",
    g3_bw, controls=["pX", "pX_pD", "year_dummy"] + CONTROLS_SCH,
    sdesc="Munic, headmaster, controls", cdesc="pX pX_pD year_dummy sch")

try:
    g3_bw_nm = compute_optimal_bw(
        g3_nonmunic_f["expthisschl_lessthan2_DPB"].dropna(),
        g3_nonmunic_f.loc[g3_nonmunic_f["expthisschl_lessthan2_DPB"].notna(), "pX"],
        cluster=g3_nonmunic_f.loc[g3_nonmunic_f["expthisschl_lessthan2_DPB"].notna(), "COD_MUNICIPIO"],
        kernel="uniform",
    )
except Exception:
    g3_bw_nm = g3_bw

g3_run("baseline__headmaster_nonmunic",
    "specification_tree/designs/regression_discontinuity.md",
    g3_bw_nm, data=g3_nonmunic_f,
    sdesc="NonMunic, headmaster, optimal bw", cdesc="pX pX_pD")

# RC variants
log("G3 RC variants...")
g3_run("rc/controls/sets/with_school_controls_year",
    "specification_tree/modules/robustness/controls.md#sets",
    g3_bw, controls=["pX", "pX_pD", "year_dummy"] + CONTROLS_SCH,
    sdesc="Munic, headmaster, school ctrl+year", cdesc="pX pX_pD year_dummy sch",
    ablocks={"controls": {"spec_id": "rc/controls/sets/with_school_controls_year", "family": "sets"}})

for bw_val, sid in [(0.07, "rc/bandwidth/fixed_07"), (0.11, "rc/bandwidth/fixed_11"),
                     (g3_bw/2, "rc/bandwidth/half_optimal"), (g3_bw*2, "rc/bandwidth/double_optimal")]:
    g3_run(sid, "specification_tree/designs/regression_discontinuity.md#bandwidth",
        bw_val, sdesc=f"Munic, headmaster, bw={bw_val:.4f}", cdesc="pX pX_pD",
        dov={"bandwidth": f"{bw_val:.4f}"})

g3_run("rc/sample/subgroup/2008_cycle_only",
    "specification_tree/modules/robustness/sample.md#subgroup",
    g3_bw, data=g3_munic_f[g3_munic_f["year"]==2009].copy(),
    sdesc="Munic, headmaster, 2008 cycle", cdesc="pX pX_pD",
    ablocks={"sample": {"spec_id": "rc/sample/subgroup/2008_cycle_only"}})

g3_run("rc/sample/subgroup/2012_cycle_only",
    "specification_tree/modules/robustness/sample.md#subgroup",
    g3_bw, data=g3_munic_f[g3_munic_f["year"]==2013].copy(),
    sdesc="Munic, headmaster, 2012 cycle", cdesc="pX pX_pD",
    ablocks={"sample": {"spec_id": "rc/sample/subgroup/2012_cycle_only"}})

g3_run("rc/sample/subgroup/nonmunic_schools",
    "specification_tree/modules/robustness/sample.md#subgroup",
    g3_bw_nm, data=g3_nonmunic_f,
    sdesc="NonMunic, headmaster", cdesc="pX pX_pD",
    ablocks={"sample": {"spec_id": "rc/sample/subgroup/nonmunic_schools"}})

g3_run("rc/sample/restriction/donut_1pct",
    "specification_tree/modules/robustness/sample.md#restriction",
    g3_bw, data=g3_munic_f[g3_munic_f["pX"].abs()>=0.01].copy(),
    sdesc="Munic, headmaster, donut 1pct", cdesc="pX pX_pD",
    ablocks={"sample": {"spec_id": "rc/sample/restriction/donut_1pct"}})

g3_munic_f = g3_munic_f.copy()
g3_munic_f["pX_sq"] = g3_munic_f["pX"]**2
g3_munic_f["pX_sq_pD"] = g3_munic_f["pX_sq"] * g3_munic_f["pX_dummy"]
g3_run("rc/poly/quadratic",
    "specification_tree/designs/regression_discontinuity.md#polynomial",
    g3_bw, controls=["pX", "pX_pD", "pX_sq", "pX_sq_pD"],
    sdesc="Munic, headmaster, quadratic", cdesc="pX pX_pD pX_sq pX_sq_pD",
    dov={"poly_order": 2})

# Teacher outcomes
log("G3 teacher outcomes...")
for tchr_var, tchr_label in [("newtchr", "new_teacher_share"), ("lefttchr", "left_teacher_share")]:
    try:
        bw_t = compute_optimal_bw(
            g3_munic_tchr_f[tchr_var].dropna(),
            g3_munic_tchr_f.loc[g3_munic_tchr_f[tchr_var].notna(), "pX"],
            cluster=g3_munic_tchr_f.loc[g3_munic_tchr_f[tchr_var].notna(), "COD_MUNICIPIO"],
            kernel="uniform",
        )
    except Exception:
        bw_t = g3_bw
    g3_run(f"rc/form/outcome/{tchr_label}",
        "specification_tree/modules/robustness/functional_form.md#outcome",
        bw_t, outcome=tchr_var, data=g3_munic_tchr_f,
        sdesc=f"Munic, {tchr_label}, docentes matched", cdesc="pX pX_pD",
        ablocks={"functional_form": {"spec_id": f"rc/form/outcome/{tchr_label}",
                                      "interpretation": f"Share of teachers: {tchr_label}"}})

# Cross-product: headmaster at fixed bw with controls
for bw_val, bw_lbl in [(0.07, "bw07"), (0.11, "bw11")]:
    g3_run(f"rc/headmaster_bw_controls/{bw_lbl}_no_controls",
        "specification_tree/designs/regression_discontinuity.md#bandwidth",
        bw_val, sdesc=f"Munic, headmaster, bw={bw_val}", cdesc="pX pX_pD",
        dov={"bandwidth": str(bw_val)})
    g3_run(f"rc/headmaster_bw_controls/{bw_lbl}_with_controls",
        "specification_tree/designs/regression_discontinuity.md#bandwidth",
        bw_val, controls=["pX", "pX_pD", "year_dummy"] + CONTROLS_SCH,
        sdesc=f"Munic, headmaster, bw={bw_val}, ctrl", cdesc="pX pX_pD year_dummy sch",
        dov={"bandwidth": str(bw_val)})

g3_count = sum(1 for r in spec_results if r["baseline_group_id"] == "G3")
log(f"  G3 complete: {g3_count} specs")


# =====================================================
# WRITE OUTPUTS
# =====================================================
log("\n" + "="*60)
log("WRITING OUTPUTS")
log("="*60)

spec_df = pd.DataFrame(spec_results)
spec_df.to_csv(f"{OUTPUT_DIR}/specification_results.csv", index=False)
log(f"  specification_results.csv: {len(spec_df)} rows")
log(f"    Successes: {(spec_df['run_success']==1).sum()}")
log(f"    Failures: {(spec_df['run_success']==0).sum()}")
log(f"    By group: {spec_df.groupby('baseline_group_id').size().to_dict()}")

infer_df = pd.DataFrame(inference_results)
infer_df.to_csv(f"{OUTPUT_DIR}/inference_results.csv", index=False)
log(f"  inference_results.csv: {len(infer_df)} rows")

# SPECIFICATION_SEARCH.md
n_planned = len(spec_df)
n_success = int((spec_df["run_success"]==1).sum())
n_failed = int((spec_df["run_success"]==0).sum())

g1_total = len(spec_df[spec_df["baseline_group_id"]=="G1"])
g1_ok = int((spec_df[spec_df["baseline_group_id"]=="G1"]["run_success"]==1).sum())
g1_fail = g1_total - g1_ok
g2_total = len(spec_df[spec_df["baseline_group_id"]=="G2"])
g2_ok = int((spec_df[spec_df["baseline_group_id"]=="G2"]["run_success"]==1).sum())
g2_fail = g2_total - g2_ok
g3_total = len(spec_df[spec_df["baseline_group_id"]=="G3"])
g3_ok = int((spec_df[spec_df["baseline_group_id"]=="G3"]["run_success"]==1).sum())
g3_fail = g3_total - g3_ok

# Coefficient summaries by group
def coef_summary(group_id):
    g = spec_df[(spec_df["baseline_group_id"]==group_id) & (spec_df["run_success"]==1)]
    if len(g) == 0:
        return "No successful specs"
    base = g[g["spec_id"].str.startswith("baseline")]
    base_coef = f"{base['coefficient'].values[0]:.4f}" if len(base) > 0 else "N/A"
    n_sig = int((g["p_value"] < 0.05).sum())
    return (f"Baseline coef: {base_coef}, range: [{g['coefficient'].min():.4f}, {g['coefficient'].max():.4f}], "
            f"sig(5%): {n_sig}/{len(g)}")

md = f"""# Specification Search: {PAPER_ID}

## Paper
Akhtari, Moreira & Trucco (2022), "Political Turnover, Bureaucratic Turnover, and the Quality of Public Services", AER.

## Surface Summary
- **Baseline groups**: 3 (G1: test scores, G2: municipal personnel, G3: headmaster/teacher)
- **Design**: Sharp regression discontinuity (piecewise-linear within bandwidth)
- **Running variable**: pX (incumbent party vote margin), cutoff at 0
- **Budgets**: G1=60, G2=30, G3=25 (115 target)
- **Seed**: {SUBSAMPLE_SEED}

## Memory Optimization
- **Student-level data (G1)**: Full dataset (11.5M rows), no subsampling.
  Bandwidth selection via rdrobust uses a {BW_SUBSAMPLE_N:,}-obs subsample for tractability.
- **School-level and RAIS data (G2, G3)**: Full sample, selective column loading.
- **Non-municipal student data**: Skipped; school-level aggregates used for placebo tests.

## Execution Summary
- **Total specifications planned**: {n_planned}
- **Successfully executed**: {n_success}
- **Failed**: {n_failed}
- **Inference variants**: {len(infer_df)}

### By Group
| Group | Total | Success | Failed |
|-------|-------|---------|--------|
| G1 | {g1_total} | {g1_ok} | {g1_fail} |
| G2 | {g2_total} | {g2_ok} | {g2_fail} |
| G3 | {g3_total} | {g3_ok} | {g3_fail} |

### Coefficient Summaries
- **G1 (test scores)**: {coef_summary("G1")}
- **G2 (personnel)**: {coef_summary("G2")}
- **G3 (headmaster)**: {coef_summary("G3")}

## Optimal Bandwidths
- G1 (4th grade combined): {g1_bw4:.5f}
- G1 (8th grade combined): {g1_bw8:.5f}
- G2 (SHhired_Mun_lead): {g2_bw:.5f}
- G3 (headmaster): {g3_bw:.5f}

## Deviations and Notes
- **Bandwidth subsampling**: rdrobust bandwidth selection uses a 100K-obs subsample (masspoints=off) for computational tractability. Regressions run on full data within the selected bandwidth.
- **Math/Portuguese scores separately (G1)**: Not available in the student-level data. Only combined scores (both_score_indiv_4_stdComb). Specs rc/form/outcome/math_score_only and port_score_only marked as failed.
- **Unstandardized scores (G1)**: Used year-specific standardized scores (std08/std12) as alternative to combined standardization (stdComb).
- **Triangular kernel (G1)**: Implemented via weighted OLS with triangular kernel weights and re-optimized bandwidth.
- **G1 nonmunic placebo**: Used school-level aggregated scores (both_score_4_std) instead of student-level to avoid loading 2.3GB non-municipal student file.
- **G2 after outcomes**: Created from RAIS _10/_14 year suffixes for t+2 personnel outcomes.
- **Teacher outcomes (G3)**: Restricted to schools matched to DOCENTES census in all 4 years (count==4).
- **G2 bandwidth/rc overlap**: Some bandwidth specs appear in both design/ and rc/ prefixes; these are kept as the surface specifies both.

## Software Stack
- Python {SW_BLOCK.get('runner_version', 'unknown')}
- pyfixest {SW_BLOCK.get('packages', dict()).get('pyfixest', 'unknown')}
- pandas {SW_BLOCK.get('packages', dict()).get('pandas', 'unknown')}
- numpy {SW_BLOCK.get('packages', dict()).get('numpy', 'unknown')}
- rdrobust {SW_BLOCK.get('packages', dict()).get('rdrobust', 'unknown')}

Surface hash: `{SURFACE_HASH}`
"""

with open(f"{OUTPUT_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write(md)
log("  SPECIFICATION_SEARCH.md written")
log("\nDone!")
