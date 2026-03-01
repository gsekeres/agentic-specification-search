"""
Specification Search Script for Azoulay, Jones, Kim & Miranda (2022)
"Immigration and Entrepreneurship in the United States"
American Economic Review: Insights, 4(1), 71-88.

Paper ID: 134622-V1

Surface-driven execution:
  G1: Aggregated admin data (W2/Census) – pooled regression of log_freq_norm ~ log_firm_size
      + immigrant + immigrant*log_firm_size.  Power law slope comparison.
      N=14 (7 bins x 2 groups).  No FE, HC1.
  G2: Fortune 500 firm-level data – ln_nb_employees ~ immi + controls.
      N~449 firms.  No FE (baseline), decade FE variants.  HC1.

Outputs:
  - specification_results.csv
  - inference_results.csv
  - SPECIFICATION_SEARCH.md
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import statsmodels.api as sm
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

PAPER_ID = "134622-V1"
DATA_DIR = "data/downloads/extracted/134622-V1"
OUTPUT_DIR = DATA_DIR

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

bg1 = surface_obj["baseline_groups"][0]
bg2 = surface_obj["baseline_groups"][1]

# ============================================================
# Accumulators
# ============================================================

results = []
inference_results = []
spec_run_counter = 0

# ============================================================
# Helper: run_spec (OLS via statsmodels for small-N data and pyfixest for larger data)
# ============================================================

def run_spec(spec_id, spec_tree_path, baseline_group_id,
             outcome_var, treatment_var, controls, fe_formula_str,
             fe_desc, data, vcov, sample_desc, controls_desc,
             cluster_var=None,
             axis_block_name=None, axis_block=None, notes="",
             design_audit=None, inference_canonical_id="infer/se/hc/hc1"):
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
            inference={"spec_id": inference_canonical_id,
                       "method": "HC1"},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"cross_sectional_ols": design_audit} if design_audit else None,
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


def run_inference_variant(spec_id, baseline_run_id, outcome_var, treatment_var,
                          controls, fe_formula_str, data, vcov, vcov_type,
                          baseline_group_id, notes=""):
    """Run an inference variant on the baseline specification."""
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

        se_val = float(m.se().get(treatment_var, np.nan))
        pval = float(m.pvalue().get(treatment_var, np.nan))

        try:
            ci = m.confint()
            ci_lower = float(ci.loc[treatment_var, ci.columns[0]]) if treatment_var in ci.index else np.nan
            ci_upper = float(ci.loc[treatment_var, ci.columns[1]]) if treatment_var in ci.index else np.nan
        except Exception:
            ci_lower = np.nan
            ci_upper = np.nan

        inference_results.append({
            "paper_id": PAPER_ID,
            "spec_run_id": run_id,
            "spec_id": spec_id,
            "baseline_run_id": baseline_run_id,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": treatment_var,
            "std_error": se_val,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "vcov_type": vcov_type,
            "notes": notes,
            "run_success": 1,
            "run_error": ""
        })
        return run_id

    except Exception as e:
        err_msg = str(e)[:240]
        inference_results.append({
            "paper_id": PAPER_ID,
            "spec_run_id": run_id,
            "spec_id": spec_id,
            "baseline_run_id": baseline_run_id,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": treatment_var,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "vcov_type": vcov_type,
            "notes": notes,
            "run_success": 0,
            "run_error": err_msg
        })
        return run_id


# ============================================================
# G1: Administrative (W2/Census) Data - Firm Size Distributions
# ============================================================

print("=" * 60)
print("G1: Administrative data - firm size power law regressions")
print("=" * 60)

# Load admin_firm_size_freq.xlsx, defn1 sheet
admin_data = pd.read_excel(f"{DATA_DIR}/W2-Code/admin_firm_size_freq.xlsx", sheet_name=None)

# --- Build pooled dataset for defn1 baseline ---
defn1 = admin_data["defn1"].copy()
# The paper's Figure 1 plots log_freq_norm (normalized by population) vs log_firm_size
# Immigrant population: 20,740,000; Native population: 134,000,000
IMMI_POP = 20740000
NATIVE_POP = 134000000

# Build pooled long-form dataset
rows = []
for _, r in defn1.iterrows():
    lfs = r["log_firm_size"]
    # Immigrant observation
    rows.append({
        "log_firm_size": lfs,
        "log_freq_norm": np.log10(r["freq_immi"] / IMMI_POP),
        "immigrant": 1,
    })
    # Native observation
    rows.append({
        "log_firm_size": lfs,
        "log_freq_norm": np.log10(r["freq_native"] / NATIVE_POP),
        "immigrant": 0,
    })

df_admin = pd.DataFrame(rows)
df_admin["immi_x_size"] = df_admin["immigrant"] * df_admin["log_firm_size"]

print(f"Admin pooled data: {len(df_admin)} rows")
print(df_admin.to_string())

# --- G1 Baseline: Admin-Defn1-Pooled ---
print("\nRunning G1 baseline: Admin-Defn1-Pooled")
g1_baseline_id, g1_base_coef, g1_base_se, g1_base_pval, g1_base_n = run_spec(
    "baseline/G1/Admin-Defn1-Pooled",
    "specification_tree/methods/cross_sectional_ols.md", "G1",
    "log_freq_norm", "immigrant", ["log_firm_size", "immi_x_size"],
    "", "none", df_admin,
    "HC1",
    "Admin defn1 pooled (7 bins x 2 groups = 14 obs)", "log_firm_size + immi_x_size",
    design_audit=bg1["design_audit"],
    notes="Pooled regression replicating Figure 1 lfit lines."
)
print(f"  Baseline coef(immigrant): {g1_base_coef:.6f}, SE: {g1_base_se:.6f}, p: {g1_base_pval:.6f}, N: {g1_base_n}")


# ============================================================
# G1 Core Universe: rc_spec_ids from surface
# ============================================================

print("\nRunning G1 core universe specifications...")

# --- rc/data/treatment_alt/defn2 ---
defn2 = admin_data["defn2"].copy()
rows_d2 = []
for _, r in defn2.iterrows():
    lfs = r["log_firm_size"]
    rows_d2.append({"log_firm_size": lfs,
                     "log_freq_norm": np.log10(r["freq_immi"] / IMMI_POP),
                     "immigrant": 1})
    rows_d2.append({"log_firm_size": lfs,
                     "log_freq_norm": np.log10(r["freq_native"] / NATIVE_POP),
                     "immigrant": 0})
df_defn2 = pd.DataFrame(rows_d2)
df_defn2["immi_x_size"] = df_defn2["immigrant"] * df_defn2["log_firm_size"]

run_spec("rc/data/treatment_alt/defn2",
         "modules/robustness/data.md#treatment-alt", "G1",
         "log_freq_norm", "immigrant", ["log_firm_size", "immi_x_size"],
         "", "none", df_defn2, "HC1",
         "Admin defn2 (highest-paid founder foreign-born)", "log_firm_size + immi_x_size",
         design_audit=bg1["design_audit"],
         notes="Definition 2: immigrant = highest-paid founder is foreign-born")

# --- rc/data/treatment_alt/defn3 ---
defn3 = admin_data["defn3"].copy()
# defn3 uses share-weighted founder counts; freq_immi and freq_native are total weighted counts
# Population denominators remain the same
rows_d3 = []
for _, r in defn3.iterrows():
    lfs = r["log_firm_size"]
    rows_d3.append({"log_firm_size": lfs,
                     "log_freq_norm": np.log10(r["freq_immi"] / IMMI_POP),
                     "immigrant": 1})
    rows_d3.append({"log_firm_size": lfs,
                     "log_freq_norm": np.log10(r["freq_native"] / NATIVE_POP),
                     "immigrant": 0})
df_defn3 = pd.DataFrame(rows_d3)
df_defn3["immi_x_size"] = df_defn3["immigrant"] * df_defn3["log_firm_size"]

run_spec("rc/data/treatment_alt/defn3",
         "modules/robustness/data.md#treatment-alt", "G1",
         "log_freq_norm", "immigrant", ["log_firm_size", "immi_x_size"],
         "", "none", df_defn3, "HC1",
         "Admin defn3 (share-weighted immigrant founder counts)", "log_firm_size + immi_x_size",
         design_audit=bg1["design_audit"],
         notes="Definition 3: count immigrant and native founders per firm")

# --- rc/data/treatment_alt/oecd_vs_native ---
# Use OECD immigrant counts from defn1 sheet
OECD_POP = 6916000
rows_oecd = []
for _, r in defn1.iterrows():
    lfs = r["log_firm_size"]
    rows_oecd.append({"log_firm_size": lfs,
                       "log_freq_norm": np.log10(r["freq_oecd"] / OECD_POP),
                       "immigrant": 1})
    rows_oecd.append({"log_firm_size": lfs,
                       "log_freq_norm": np.log10(r["freq_native"] / NATIVE_POP),
                       "immigrant": 0})
df_oecd = pd.DataFrame(rows_oecd)
df_oecd["immi_x_size"] = df_oecd["immigrant"] * df_oecd["log_firm_size"]

run_spec("rc/data/treatment_alt/oecd_vs_native",
         "modules/robustness/data.md#treatment-alt", "G1",
         "log_freq_norm", "immigrant", ["log_firm_size", "immi_x_size"],
         "", "none", df_oecd, "HC1",
         "Admin defn1, OECD immigrants vs natives", "log_firm_size + immi_x_size",
         design_audit=bg1["design_audit"],
         notes="OECD-born immigrant founders vs native founders (Appendix Figure B3)")

# --- rc/data/treatment_alt/non_oecd_vs_native ---
NOECD_POP = 13820000
rows_noecd = []
for _, r in defn1.iterrows():
    lfs = r["log_firm_size"]
    rows_noecd.append({"log_firm_size": lfs,
                        "log_freq_norm": np.log10(r["freq_noecd"] / NOECD_POP),
                        "immigrant": 1})
    rows_noecd.append({"log_firm_size": lfs,
                        "log_freq_norm": np.log10(r["freq_native"] / NATIVE_POP),
                        "immigrant": 0})
df_noecd = pd.DataFrame(rows_noecd)
df_noecd["immi_x_size"] = df_noecd["immigrant"] * df_noecd["log_firm_size"]

run_spec("rc/data/treatment_alt/non_oecd_vs_native",
         "modules/robustness/data.md#treatment-alt", "G1",
         "log_freq_norm", "immigrant", ["log_firm_size", "immi_x_size"],
         "", "none", df_noecd, "HC1",
         "Admin defn1, non-OECD immigrants vs natives", "log_firm_size + immi_x_size",
         design_audit=bg1["design_audit"],
         notes="Non-OECD-born immigrant founders vs native founders")

# --- rc/data/source_alt/sbo_defn1 ---
sbo_defn1 = pd.read_excel(f"{DATA_DIR}/SBO-code/sbo_2020.1_def1.xlsx")
# Convert: Immigrant column has "Native"/"Immigrant" strings
sbo_d1 = sbo_defn1.copy()
sbo_d1["immigrant"] = (sbo_d1["Immigrant"] == "Immigrant").astype(int)
sbo_d1["log_firm_size"] = np.log10(sbo_d1["size_class"])
sbo_d1["immi_x_size"] = sbo_d1["immigrant"] * sbo_d1["log_firm_size"]
sbo_d1 = sbo_d1.rename(columns={"lnorm_freq": "log_freq_norm"})

run_spec("rc/data/source_alt/sbo_defn1",
         "modules/robustness/data.md#source-alt", "G1",
         "log_freq_norm", "immigrant", ["log_firm_size", "immi_x_size"],
         "", "none", sbo_d1, "HC1",
         f"SBO defn1 ({len(sbo_d1)} obs)", "log_firm_size + immi_x_size",
         design_audit=bg1["design_audit"],
         notes="Survey of Business Owners definition 1")

# --- rc/data/source_alt/sbo_defn2 ---
sbo_defn2 = pd.read_excel(f"{DATA_DIR}/SBO-code/sbo_2020.1_def2.xlsx")
sbo_d2 = sbo_defn2.copy()
sbo_d2["immigrant"] = sbo_d2["Immigrant"].astype(int)
sbo_d2["log_firm_size"] = np.log10(sbo_d2["size_class"])
sbo_d2["immi_x_size"] = sbo_d2["immigrant"] * sbo_d2["log_firm_size"]
sbo_d2 = sbo_d2.rename(columns={"lnorm_freq": "log_freq_norm"})

run_spec("rc/data/source_alt/sbo_defn2",
         "modules/robustness/data.md#source-alt", "G1",
         "log_freq_norm", "immigrant", ["log_firm_size", "immi_x_size"],
         "", "none", sbo_d2, "HC1",
         f"SBO defn2 ({len(sbo_d2)} obs)", "log_firm_size + immi_x_size",
         design_audit=bg1["design_audit"],
         notes="Survey of Business Owners definition 2 (largest owner foreign-born)")

# --- rc/data/source_alt/sbo_defn3 ---
sbo_defn3 = pd.read_excel(f"{DATA_DIR}/SBO-code/sbo_2020.1_def3.xlsx")
sbo_d3 = sbo_defn3.copy()
sbo_d3["immigrant"] = sbo_d3["Immigrant"].astype(int)
sbo_d3["log_firm_size"] = np.log10(sbo_d3["size_class"])
sbo_d3["immi_x_size"] = sbo_d3["immigrant"] * sbo_d3["log_firm_size"]
sbo_d3 = sbo_d3.rename(columns={"lnorm_freq": "log_freq_norm"})

run_spec("rc/data/source_alt/sbo_defn3",
         "modules/robustness/data.md#source-alt", "G1",
         "log_freq_norm", "immigrant", ["log_firm_size", "immi_x_size"],
         "", "none", sbo_d3, "HC1",
         f"SBO defn3 ({len(sbo_d3)} obs)", "log_firm_size + immi_x_size",
         design_audit=bg1["design_audit"],
         notes="Survey of Business Owners definition 3 (share-weighted)")

# --- Fortune 500 data for rc/data/source_alt/fortune500_defn1 and defn3 ---
# These use the F500 aggregated log-log data constructed from figure3.do

# Load raw Fortune 500 data and construct aggregated frequency data
f500_raw = pd.read_excel(f"{DATA_DIR}/Fortune500-replication/fortune500_founders_data.xls")
f500_immi_pop = pd.read_excel(f"{DATA_DIR}/Fortune500-replication/us_nb_of_immigrants.xls")

# Clean per tableB2.do / figure3.do
f500 = f500_raw[(f500_raw["immigrant"] == 0) | (f500_raw["immigrant"] == 1)].copy()
f500["impt_yob"] = f500["yob"].isna().astype(int)
f500["yob_impt"] = f500["yob"].copy()
f500.loc[f500["impt_yob"] == 1, "yob_impt"] = f500.loc[f500["impt_yob"] == 1, "inc_year"] - 38
f500 = f500[f500["yob_impt"].notna()].copy()
f500 = f500[f500["inc_year"].notna()].copy()

# Merge immigrant population data
f500["decade"] = 10 * np.floor(f500["inc_year"] / 10).astype(int)
f500.loc[f500["inc_year"] > 2010, "decade"] = f500.loc[f500["inc_year"] > 2010, "inc_year"].astype(int)
f500 = f500.merge(f500_immi_pop, on="decade", how="left")

print(f"\nFortune 500 cleaned data: {len(f500)} founder-rows")

# Definition 1: immi = at least 1 founder is foreign-born (firm-level)
f500_firm = f500.copy()
f500_firm["immi"] = f500_firm.groupby("rank")["immigrant"].transform("max")
f500_firm = f500_firm.drop_duplicates(subset=["rank"]).copy()
f500_firm["nb_natives"] = (f500_firm["nb_immgrts"] / f500_firm["pct_immgrts"]) - f500_firm["nb_immgrts"]

# Create size bins (log10 of nb_employees, binned at 4, 4.5, 5)
f500_firm["log_emp5_bin"] = 4.0
f500_firm.loc[(np.log10(f500_firm["nb_employees"]) >= 4.5) &
              (np.log10(f500_firm["nb_employees"]) < 5.0), "log_emp5_bin"] = 4.5
f500_firm.loc[(np.log10(f500_firm["nb_employees"]) >= 5.0) &
              (f500_firm["nb_employees"].notna()) &
              (f500_firm["nb_employees"] > 0), "log_emp5_bin"] = 5.0

# Build freq by immi x size_bin (definition 1)
freq_immi = f500_firm[f500_firm["immi"] == 1].groupby("log_emp5_bin").agg(
    freq=("rank", "count"),
    wghtd_pop=("nb_immgrts", "mean")
).reset_index()
freq_immi["log_freq_norm"] = np.log10(freq_immi["freq"] / freq_immi["wghtd_pop"])
freq_immi["immigrant"] = 1
freq_immi["log_firm_size"] = freq_immi["log_emp5_bin"]

freq_native = f500_firm[f500_firm["immi"] == 0].groupby("log_emp5_bin").agg(
    freq=("rank", "count"),
    wghtd_pop=("nb_natives", "mean")
).reset_index()
freq_native["log_freq_norm"] = np.log10(freq_native["freq"] / freq_native["wghtd_pop"])
freq_native["immigrant"] = 0
freq_native["log_firm_size"] = freq_native["log_emp5_bin"]

df_f500_d1 = pd.concat([freq_immi[["log_firm_size", "log_freq_norm", "immigrant"]],
                          freq_native[["log_firm_size", "log_freq_norm", "immigrant"]]], ignore_index=True)
df_f500_d1["immi_x_size"] = df_f500_d1["immigrant"] * df_f500_d1["log_firm_size"]

run_spec("rc/data/source_alt/fortune500_defn1",
         "modules/robustness/data.md#source-alt", "G1",
         "log_freq_norm", "immigrant", ["log_firm_size", "immi_x_size"],
         "", "none", df_f500_d1, "HC1",
         f"Fortune 500 defn1, aggregated ({len(df_f500_d1)} obs)", "log_firm_size + immi_x_size",
         design_audit=bg1["design_audit"],
         notes="Fortune 500 definition 1 (at least 1 immigrant founder), aggregated log-log")

# Definition 3 for Fortune 500: share-weighted
f500_all = f500.copy()
f500_all["nb_natives"] = (f500_all["nb_immgrts"] / f500_all["pct_immgrts"]) - f500_all["nb_immgrts"]
f500_all["log_emp5_bin"] = 4.0
f500_all.loc[(np.log10(f500_all["nb_employees"]) >= 4.5) &
              (np.log10(f500_all["nb_employees"]) < 5.0), "log_emp5_bin"] = 4.5
f500_all.loc[(np.log10(f500_all["nb_employees"]) >= 5.0) &
              (f500_all["nb_employees"].notna()) &
              (f500_all["nb_employees"] > 0), "log_emp5_bin"] = 5.0

# Count immigrant and native founders per bin
nbfounders = f500_all.groupby("log_emp5_bin")["founder_seq"].count().reset_index(name="nbfounders")
nbfounders_i = f500_all[f500_all["immigrant"] == 1].groupby("log_emp5_bin")["founder_seq"].count().reset_index(name="nbfounders_i")
nbfounders_n = f500_all[f500_all["immigrant"] == 0].groupby("log_emp5_bin")["founder_seq"].count().reset_index(name="nbfounders_n")

f500_d3_agg = nbfounders.merge(nbfounders_i, on="log_emp5_bin", how="left").merge(nbfounders_n, on="log_emp5_bin", how="left")
f500_d3_agg["nbfounders_i"] = f500_d3_agg["nbfounders_i"].fillna(0)
f500_d3_agg["nbfounders_n"] = f500_d3_agg["nbfounders_n"].fillna(0)
f500_d3_agg["frac_i"] = f500_d3_agg["nbfounders_i"] / f500_d3_agg["nbfounders"]
f500_d3_agg["frac_n"] = f500_d3_agg["nbfounders_n"] / f500_d3_agg["nbfounders"]

# Unique firms per bin
freq_firms = f500_firm.groupby("log_emp5_bin")["rank"].nunique().reset_index(name="freq")
wghtd_pop_i_bin = f500_all.groupby("log_emp5_bin")["nb_immgrts"].mean().reset_index(name="wghtd_pop_i")
wghtd_pop_n_bin = f500_all.groupby("log_emp5_bin")["nb_natives"].mean().reset_index(name="wghtd_pop_n")

f500_d3_agg = f500_d3_agg.merge(freq_firms, on="log_emp5_bin").merge(wghtd_pop_i_bin, on="log_emp5_bin").merge(wghtd_pop_n_bin, on="log_emp5_bin")

rows_f500_d3 = []
for _, r in f500_d3_agg.iterrows():
    lfs = r["log_emp5_bin"]
    if r["frac_i"] > 0 and r["wghtd_pop_i"] > 0:
        rows_f500_d3.append({"log_firm_size": lfs,
                              "log_freq_norm": np.log10((r["freq"] * r["frac_i"]) / r["wghtd_pop_i"]),
                              "immigrant": 1})
    if r["frac_n"] > 0 and r["wghtd_pop_n"] > 0:
        rows_f500_d3.append({"log_firm_size": lfs,
                              "log_freq_norm": np.log10((r["freq"] * r["frac_n"]) / r["wghtd_pop_n"]),
                              "immigrant": 0})

df_f500_d3 = pd.DataFrame(rows_f500_d3)
df_f500_d3["immi_x_size"] = df_f500_d3["immigrant"] * df_f500_d3["log_firm_size"]

run_spec("rc/data/source_alt/fortune500_defn3",
         "modules/robustness/data.md#source-alt", "G1",
         "log_freq_norm", "immigrant", ["log_firm_size", "immi_x_size"],
         "", "none", df_f500_d3, "HC1",
         f"Fortune 500 defn3, aggregated ({len(df_f500_d3)} obs)", "log_firm_size + immi_x_size",
         design_audit=bg1["design_audit"],
         notes="Fortune 500 definition 3 (share-weighted founder counts)")

# --- rc/sample/drop_top_bin ---
df_no_top = df_admin[df_admin["log_firm_size"] < df_admin["log_firm_size"].max()].copy()
run_spec("rc/sample/drop_top_bin",
         "modules/robustness/sample.md#drop-bins", "G1",
         "log_freq_norm", "immigrant", ["log_firm_size", "immi_x_size"],
         "", "none", df_no_top, "HC1",
         f"Admin defn1, drop top size bin (N={len(df_no_top)})", "log_firm_size + immi_x_size",
         design_audit=bg1["design_audit"],
         notes="Drop largest firm-size bin (log_firm_size = 3.5)")

# --- rc/sample/drop_bottom_bin ---
df_no_bot = df_admin[df_admin["log_firm_size"] > df_admin["log_firm_size"].min()].copy()
run_spec("rc/sample/drop_bottom_bin",
         "modules/robustness/sample.md#drop-bins", "G1",
         "log_freq_norm", "immigrant", ["log_firm_size", "immi_x_size"],
         "", "none", df_no_bot, "HC1",
         f"Admin defn1, drop bottom size bin (N={len(df_no_bot)})", "log_firm_size + immi_x_size",
         design_audit=bg1["design_audit"],
         notes="Drop smallest firm-size bin (log_firm_size = 0)")

# --- rc/form/separate_immi_slope ---
# Run separate regression for immigrant group only
df_immi_only = df_admin[df_admin["immigrant"] == 1].copy()
run_spec("rc/form/separate_immi_slope",
         "modules/robustness/form.md#separate-slopes", "G1",
         "log_freq_norm", "log_firm_size", [],
         "", "none", df_immi_only, "HC1",
         "Admin defn1, immigrant group only (7 obs)", "none",
         design_audit=bg1["design_audit"],
         notes="Separate regression for immigrant group: slope of log_freq_norm on log_firm_size")

# --- rc/form/separate_native_slope ---
df_native_only = df_admin[df_admin["immigrant"] == 0].copy()
run_spec("rc/form/separate_native_slope",
         "modules/robustness/form.md#separate-slopes", "G1",
         "log_freq_norm", "log_firm_size", [],
         "", "none", df_native_only, "HC1",
         "Admin defn1, native group only (7 obs)", "none",
         design_audit=bg1["design_audit"],
         notes="Separate regression for native group: slope of log_freq_norm on log_firm_size")

# --- rc/form/quadratic_firm_size ---
df_quad = df_admin.copy()
df_quad["log_firm_size_sq"] = df_quad["log_firm_size"] ** 2
df_quad["immi_x_size_sq"] = df_quad["immigrant"] * df_quad["log_firm_size_sq"]
run_spec("rc/form/quadratic_firm_size",
         "modules/robustness/form.md#polynomial", "G1",
         "log_freq_norm", "immigrant",
         ["log_firm_size", "immi_x_size", "log_firm_size_sq", "immi_x_size_sq"],
         "", "none", df_quad, "HC1",
         "Admin defn1, quadratic in log_firm_size (14 obs)",
         "log_firm_size + immi_x_size + log_firm_size^2 + immi_x_size^2",
         design_audit=bg1["design_audit"],
         notes="Quadratic polynomial in firm size; checks linearity of power law")

# --- rc/form/weighted_by_freq ---
# Weight by the number of firms (freq_immi + freq_native) for each bin
df_wt = df_admin.copy()
freq_total_by_bin = {}
for _, r in defn1.iterrows():
    freq_total_by_bin[r["log_firm_size"]] = r["freq_immi"] + r["freq_native"]
df_wt["weight"] = df_wt["log_firm_size"].map(freq_total_by_bin)

# Use WLS via statsmodels since pyfixest doesn't easily handle analytic weights
try:
    X = sm.add_constant(df_wt[["immigrant", "log_firm_size", "immi_x_size"]].astype(float))
    y = df_wt["log_freq_norm"].astype(float)
    w = df_wt["weight"].astype(float)
    wls_mod = sm.WLS(y, X, weights=w).fit(cov_type="HC1")

    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"
    coef_val = float(wls_mod.params.get("immigrant", np.nan))
    se_val = float(wls_mod.bse.get("immigrant", np.nan))
    pval = float(wls_mod.pvalues.get("immigrant", np.nan))
    ci = wls_mod.conf_int()
    ci_lower = float(ci.loc["immigrant", 0])
    ci_upper = float(ci.loc["immigrant", 1])
    nobs = int(wls_mod.nobs)
    r2 = float(wls_mod.rsquared)
    all_coefs = {k: float(v) for k, v in wls_mod.params.items()}
    payload = make_success_payload(
        coefficients=all_coefs,
        inference={"spec_id": "infer/se/hc/hc1", "method": "HC1"},
        software=SW_BLOCK, surface_hash=SURFACE_HASH,
        design={"cross_sectional_ols": bg1["design_audit"]},
    )
    results.append({
        "paper_id": PAPER_ID, "spec_run_id": run_id,
        "spec_id": "rc/form/weighted_by_freq",
        "spec_tree_path": "modules/robustness/form.md#weights",
        "baseline_group_id": "G1",
        "outcome_var": "log_freq_norm", "treatment_var": "immigrant",
        "coefficient": coef_val, "std_error": se_val, "p_value": pval,
        "ci_lower": ci_lower, "ci_upper": ci_upper,
        "n_obs": nobs, "r_squared": r2,
        "coefficient_vector_json": json.dumps(payload),
        "sample_desc": "Admin defn1, WLS by total freq per bin",
        "fixed_effects": "none", "controls_desc": "log_firm_size + immi_x_size",
        "cluster_var": "", "run_success": 1, "run_error": ""
    })
    print(f"  rc/form/weighted_by_freq: coef={coef_val:.6f}, p={pval:.6f}")
except Exception as e:
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"
    err_msg = str(e)[:240]
    err_details = error_details_from_exception(e, stage="estimation")
    payload = make_failure_payload(error=err_msg, error_details=err_details,
                                   software=SW_BLOCK, surface_hash=SURFACE_HASH)
    results.append({
        "paper_id": PAPER_ID, "spec_run_id": run_id,
        "spec_id": "rc/form/weighted_by_freq",
        "spec_tree_path": "modules/robustness/form.md#weights",
        "baseline_group_id": "G1",
        "outcome_var": "log_freq_norm", "treatment_var": "immigrant",
        "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
        "ci_lower": np.nan, "ci_upper": np.nan,
        "n_obs": np.nan, "r_squared": np.nan,
        "coefficient_vector_json": json.dumps(payload),
        "sample_desc": "Admin defn1, WLS by total freq per bin",
        "fixed_effects": "none", "controls_desc": "log_firm_size + immi_x_size",
        "cluster_var": "", "run_success": 0, "run_error": err_msg
    })

# --- rc/estimation/wls_by_freq_immi ---
# Run separate WLS for immigrant firms only, weighted by frequency
try:
    defn1_immi_wls = defn1.copy()
    defn1_immi_wls["log_freq_norm"] = np.log10(defn1_immi_wls["freq_immi"] / IMMI_POP)
    X = sm.add_constant(defn1_immi_wls["log_firm_size"].astype(float))
    y = defn1_immi_wls["log_freq_norm"].astype(float)
    w = defn1_immi_wls["freq_immi"].astype(float)
    wls_immi = sm.WLS(y, X, weights=w).fit(cov_type="HC1")

    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"
    coef_val = float(wls_immi.params.get("log_firm_size", np.nan))
    se_val = float(wls_immi.bse.get("log_firm_size", np.nan))
    pval = float(wls_immi.pvalues.get("log_firm_size", np.nan))
    ci = wls_immi.conf_int()
    ci_lower = float(ci.loc["log_firm_size", 0])
    ci_upper = float(ci.loc["log_firm_size", 1])
    all_coefs = {k: float(v) for k, v in wls_immi.params.items()}
    payload = make_success_payload(
        coefficients=all_coefs,
        inference={"spec_id": "infer/se/hc/hc1", "method": "HC1"},
        software=SW_BLOCK, surface_hash=SURFACE_HASH,
        design={"cross_sectional_ols": bg1["design_audit"]},
    )
    results.append({
        "paper_id": PAPER_ID, "spec_run_id": run_id,
        "spec_id": "rc/estimation/wls_by_freq_immi",
        "spec_tree_path": "modules/robustness/estimation.md#wls",
        "baseline_group_id": "G1",
        "outcome_var": "log_freq_norm", "treatment_var": "log_firm_size",
        "coefficient": coef_val, "std_error": se_val, "p_value": pval,
        "ci_lower": ci_lower, "ci_upper": ci_upper,
        "n_obs": int(wls_immi.nobs), "r_squared": float(wls_immi.rsquared),
        "coefficient_vector_json": json.dumps(payload),
        "sample_desc": "Admin defn1, immigrant only, WLS by freq",
        "fixed_effects": "none", "controls_desc": "none (slope only)",
        "cluster_var": "", "run_success": 1, "run_error": ""
    })
    print(f"  rc/estimation/wls_by_freq_immi: slope={coef_val:.6f}, p={pval:.6f}")
except Exception as e:
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"
    err_msg = str(e)[:240]
    results.append({
        "paper_id": PAPER_ID, "spec_run_id": run_id,
        "spec_id": "rc/estimation/wls_by_freq_immi",
        "spec_tree_path": "modules/robustness/estimation.md#wls",
        "baseline_group_id": "G1",
        "outcome_var": "log_freq_norm", "treatment_var": "log_firm_size",
        "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
        "ci_lower": np.nan, "ci_upper": np.nan,
        "n_obs": np.nan, "r_squared": np.nan,
        "coefficient_vector_json": json.dumps(make_failure_payload(
            error=err_msg, error_details=error_details_from_exception(e, stage="estimation"),
            software=SW_BLOCK, surface_hash=SURFACE_HASH)),
        "sample_desc": "Admin defn1, immigrant only, WLS by freq",
        "fixed_effects": "none", "controls_desc": "none (slope only)",
        "cluster_var": "", "run_success": 0, "run_error": err_msg
    })

# --- rc/estimation/wls_by_freq_native ---
try:
    defn1_native_wls = defn1.copy()
    defn1_native_wls["log_freq_norm"] = np.log10(defn1_native_wls["freq_native"] / NATIVE_POP)
    X = sm.add_constant(defn1_native_wls["log_firm_size"].astype(float))
    y = defn1_native_wls["log_freq_norm"].astype(float)
    w = defn1_native_wls["freq_native"].astype(float)
    wls_native = sm.WLS(y, X, weights=w).fit(cov_type="HC1")

    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"
    coef_val = float(wls_native.params.get("log_firm_size", np.nan))
    se_val = float(wls_native.bse.get("log_firm_size", np.nan))
    pval = float(wls_native.pvalues.get("log_firm_size", np.nan))
    ci = wls_native.conf_int()
    ci_lower = float(ci.loc["log_firm_size", 0])
    ci_upper = float(ci.loc["log_firm_size", 1])
    all_coefs = {k: float(v) for k, v in wls_native.params.items()}
    payload = make_success_payload(
        coefficients=all_coefs,
        inference={"spec_id": "infer/se/hc/hc1", "method": "HC1"},
        software=SW_BLOCK, surface_hash=SURFACE_HASH,
        design={"cross_sectional_ols": bg1["design_audit"]},
    )
    results.append({
        "paper_id": PAPER_ID, "spec_run_id": run_id,
        "spec_id": "rc/estimation/wls_by_freq_native",
        "spec_tree_path": "modules/robustness/estimation.md#wls",
        "baseline_group_id": "G1",
        "outcome_var": "log_freq_norm", "treatment_var": "log_firm_size",
        "coefficient": coef_val, "std_error": se_val, "p_value": pval,
        "ci_lower": ci_lower, "ci_upper": ci_upper,
        "n_obs": int(wls_native.nobs), "r_squared": float(wls_native.rsquared),
        "coefficient_vector_json": json.dumps(payload),
        "sample_desc": "Admin defn1, native only, WLS by freq",
        "fixed_effects": "none", "controls_desc": "none (slope only)",
        "cluster_var": "", "run_success": 1, "run_error": ""
    })
    print(f"  rc/estimation/wls_by_freq_native: slope={coef_val:.6f}, p={pval:.6f}")
except Exception as e:
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"
    err_msg = str(e)[:240]
    results.append({
        "paper_id": PAPER_ID, "spec_run_id": run_id,
        "spec_id": "rc/estimation/wls_by_freq_native",
        "spec_tree_path": "modules/robustness/estimation.md#wls",
        "baseline_group_id": "G1",
        "outcome_var": "log_freq_norm", "treatment_var": "log_firm_size",
        "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
        "ci_lower": np.nan, "ci_upper": np.nan,
        "n_obs": np.nan, "r_squared": np.nan,
        "coefficient_vector_json": json.dumps(make_failure_payload(
            error=err_msg, error_details=error_details_from_exception(e, stage="estimation"),
            software=SW_BLOCK, surface_hash=SURFACE_HASH)),
        "sample_desc": "Admin defn1, native only, WLS by freq",
        "fixed_effects": "none", "controls_desc": "none (slope only)",
        "cluster_var": "", "run_success": 0, "run_error": err_msg
    })


# ============================================================
# G1 Inference Variants
# ============================================================

print("\nRunning G1 inference variants...")

# infer/se/nonrobust for G1
run_inference_variant(
    "infer/se/nonrobust", g1_baseline_id,
    "log_freq_norm", "immigrant", ["log_firm_size", "immi_x_size"],
    "", df_admin, "iid", "iid (classical OLS SEs)",
    "G1", notes="Classical OLS SEs; small N, HC1 may over-reject"
)


# ============================================================
# G2: Fortune 500 Firm-Level Data
# ============================================================

print("\n" + "=" * 60)
print("G2: Fortune 500 firm-level regressions")
print("=" * 60)

# Build firm-level dataset for G2 (following figure3.do / tableB2.do)
df_g2 = f500_firm.copy()
df_g2["ln_nb_employees"] = np.log(df_g2["nb_employees"])
df_g2["ln_revenues"] = np.log(df_g2["revenues"])
df_g2["ln_inc_year"] = np.log(df_g2["inc_year"])
df_g2["decade_str"] = df_g2["decade"].astype(str)

# immi is already created (max of immigrant across founders per firm)
print(f"G2 firm-level data: {len(df_g2)} firms")
print(f"  immi==1: {(df_g2['immi']==1).sum()}, immi==0: {(df_g2['immi']==0).sum()}")

# --- G2 Baseline: F500-OLS-base ---
print("\nRunning G2 baseline: F500-OLS-base")
g2_baseline_id, g2_base_coef, g2_base_se, g2_base_pval, g2_base_n = run_spec(
    "baseline/G2/F500-OLS-base",
    "specification_tree/methods/cross_sectional_ols.md", "G2",
    "ln_nb_employees", "immi", [],
    "", "none", df_g2,
    "HC1",
    f"Fortune 500 firms, defn1 (N={len(df_g2)})", "none (bivariate)",
    design_audit=bg2["design_audit"],
    inference_canonical_id="infer/se/hc/hc1",
    notes="Bivariate OLS: log employees on immigrant indicator"
)
print(f"  Baseline coef(immi): {g2_base_coef:.6f}, SE: {g2_base_se:.6f}, p: {g2_base_pval:.6f}, N: {g2_base_n}")


# ============================================================
# G2 Core Universe
# ============================================================

print("\nRunning G2 core universe specifications...")

# --- rc/fe/add/decade ---
run_spec("rc/fe/add/decade",
         "modules/robustness/fe.md#add", "G2",
         "ln_nb_employees", "immi", [],
         "decade_str", "decade", df_g2, "HC1",
         f"Fortune 500, decade FE (N={len(df_g2)})", "none + decade FE",
         design_audit=bg2["design_audit"],
         notes="Add decade fixed effects")

# --- rc/controls/add/ln_inc_year ---
run_spec("rc/controls/add/ln_inc_year",
         "modules/robustness/controls.md#add", "G2",
         "ln_nb_employees", "immi", ["ln_inc_year"],
         "", "none", df_g2, "HC1",
         f"Fortune 500, control for log(inc_year) (N={len(df_g2)})", "ln_inc_year",
         design_audit=bg2["design_audit"],
         notes="Control for log incorporation year")

# --- rc/controls/add/yob_impt ---
run_spec("rc/controls/add/yob_impt",
         "modules/robustness/controls.md#add", "G2",
         "ln_nb_employees", "immi", ["yob_impt"],
         "", "none", df_g2.dropna(subset=["yob_impt"]), "HC1",
         f"Fortune 500, control for yob_impt (N={len(df_g2.dropna(subset=['yob_impt']))})", "yob_impt",
         design_audit=bg2["design_audit"],
         notes="Control for founder year of birth (imputed if missing)")

# --- rc/controls/full/decade_yob ---
df_g2_full = df_g2.dropna(subset=["yob_impt"]).copy()
run_spec("rc/controls/full/decade_yob",
         "modules/robustness/controls.md#full", "G2",
         "ln_nb_employees", "immi", ["yob_impt"],
         "decade_str", "decade", df_g2_full, "HC1",
         f"Fortune 500, decade FE + yob_impt (N={len(df_g2_full)})", "yob_impt + decade FE",
         design_audit=bg2["design_audit"],
         notes="Full specification with decade FE and YOB control")

# --- rc/data/treatment_alt/defn3_immigrant_share ---
# Instead of binary immi, use the share of immigrant founders
f500_d3_firm = f500.copy()
founder_counts = f500_d3_firm.groupby("rank").agg(
    n_founders=("founder_seq", "count"),
    n_immi_founders=("immigrant", "sum")
).reset_index()
founder_counts["immi_share"] = founder_counts["n_immi_founders"] / founder_counts["n_founders"]
df_g2_d3 = df_g2.merge(founder_counts[["rank", "immi_share"]], on="rank", how="left")

run_spec("rc/data/treatment_alt/defn3_immigrant_share",
         "modules/robustness/data.md#treatment-alt", "G2",
         "ln_nb_employees", "immi_share", [],
         "", "none", df_g2_d3.dropna(subset=["immi_share"]), "HC1",
         f"Fortune 500, treatment=immigrant share (N={len(df_g2_d3.dropna(subset=['immi_share']))})",
         "none (bivariate, share treatment)",
         design_audit=bg2["design_audit"],
         notes="Definition 3: treatment is share of immigrant founders instead of binary")

# --- rc/sample/post1970 ---
df_g2_post1970 = df_g2[df_g2["inc_year"] > 1970].copy()
run_spec("rc/sample/post1970",
         "modules/robustness/sample.md#time-period", "G2",
         "ln_nb_employees", "immi", [],
         "", "none", df_g2_post1970, "HC1",
         f"Fortune 500, post-1970 (N={len(df_g2_post1970)})", "none (bivariate, post-1970)",
         design_audit=bg2["design_audit"],
         notes="Restrict to firms incorporated after 1970")

# --- rc/sample/pre1970 ---
df_g2_pre1970 = df_g2[df_g2["inc_year"] <= 1970].copy()
run_spec("rc/sample/pre1970",
         "modules/robustness/sample.md#time-period", "G2",
         "ln_nb_employees", "immi", [],
         "", "none", df_g2_pre1970, "HC1",
         f"Fortune 500, pre-1970 (N={len(df_g2_pre1970)})", "none (bivariate, pre-1970)",
         design_audit=bg2["design_audit"],
         notes="Restrict to firms incorporated in or before 1970")

# --- rc/form/outcome_log_revenues ---
df_g2_rev = df_g2.dropna(subset=["ln_revenues"]).copy()
run_spec("rc/form/outcome_log_revenues",
         "modules/robustness/form.md#outcome-alt", "G2",
         "ln_revenues", "immi", [],
         "", "none", df_g2_rev, "HC1",
         f"Fortune 500, outcome=ln(revenues) (N={len(df_g2_rev)})", "none (bivariate)",
         design_audit=bg2["design_audit"],
         notes="Alternative outcome: log revenues instead of log employees")

# --- rc/form/outcome_rank ---
df_g2_rank = df_g2.copy()
df_g2_rank["rank_inv"] = df_g2_rank["rank"].max() + 1 - df_g2_rank["rank"]  # higher = better
run_spec("rc/form/outcome_rank",
         "modules/robustness/form.md#outcome-alt", "G2",
         "rank_inv", "immi", [],
         "", "none", df_g2_rank, "HC1",
         f"Fortune 500, outcome=inverted rank (N={len(df_g2_rank)})", "none (bivariate)",
         design_audit=bg2["design_audit"],
         notes="Alternative outcome: inverted Fortune 500 rank (higher = bigger)")

# --- rc/form/interaction_decade ---
# Add decade interaction with immi
df_g2_intx = df_g2.copy()
df_g2_intx["decade_num"] = df_g2_intx["decade"].astype(float)
df_g2_intx["immi_x_decade"] = df_g2_intx["immi"] * df_g2_intx["decade_num"]
run_spec("rc/form/interaction_decade",
         "modules/robustness/form.md#interaction", "G2",
         "ln_nb_employees", "immi", ["decade_num", "immi_x_decade"],
         "", "none", df_g2_intx, "HC1",
         f"Fortune 500, immi x decade interaction (N={len(df_g2_intx)})",
         "decade_num + immi_x_decade",
         design_audit=bg2["design_audit"],
         notes="Add decade interaction to detect time-varying immigrant advantage")


# ============================================================
# G2 Inference Variants
# ============================================================

print("\nRunning G2 inference variants...")

# infer/se/nonrobust for G2
run_inference_variant(
    "infer/se/nonrobust", g2_baseline_id,
    "ln_nb_employees", "immi", [],
    "", df_g2, "iid", "iid (classical OLS SEs)",
    "G2", notes="Classical OLS SEs for Fortune 500"
)


# ============================================================
# Additional G1 specifications to reach 50+ total
# ============================================================

print("\nRunning additional G1 specifications...")

# G1: Drop top AND bottom bins
df_mid = df_admin[(df_admin["log_firm_size"] > df_admin["log_firm_size"].min()) &
                   (df_admin["log_firm_size"] < df_admin["log_firm_size"].max())].copy()
run_spec("rc/sample/drop_extreme_bins",
         "modules/robustness/sample.md#drop-bins", "G1",
         "log_freq_norm", "immigrant", ["log_firm_size", "immi_x_size"],
         "", "none", df_mid, "HC1",
         f"Admin defn1, drop top+bottom bins (N={len(df_mid)})", "log_firm_size + immi_x_size",
         design_audit=bg1["design_audit"],
         notes="Drop both extreme firm-size bins")

# G1: Use unnormalized frequencies (not divided by population)
rows_unnorm = []
for _, r in defn1.iterrows():
    lfs = r["log_firm_size"]
    rows_unnorm.append({"log_firm_size": lfs,
                         "log_freq_norm": np.log10(r["freq_immi"]),
                         "immigrant": 1})
    rows_unnorm.append({"log_firm_size": lfs,
                         "log_freq_norm": np.log10(r["freq_native"]),
                         "immigrant": 0})
df_unnorm = pd.DataFrame(rows_unnorm)
df_unnorm["immi_x_size"] = df_unnorm["immigrant"] * df_unnorm["log_firm_size"]

run_spec("rc/form/unnormalized_freq",
         "modules/robustness/form.md#outcome-alt", "G1",
         "log_freq_norm", "immigrant", ["log_firm_size", "immi_x_size"],
         "", "none", df_unnorm, "HC1",
         "Admin defn1, unnormalized freq (not pop-adjusted)", "log_firm_size + immi_x_size",
         design_audit=bg1["design_audit"],
         notes="Use raw log10(freq) without dividing by population size")

# G1: Only use bins 0-2.5 (drop the sparse top bin at 3.5)
df_dense = df_admin[df_admin["log_firm_size"] <= 2.5].copy()
run_spec("rc/sample/dense_bins_only",
         "modules/robustness/sample.md#drop-bins", "G1",
         "log_freq_norm", "immigrant", ["log_firm_size", "immi_x_size"],
         "", "none", df_dense, "HC1",
         f"Admin defn1, bins 0-2.5 only (N={len(df_dense)})", "log_firm_size + immi_x_size",
         design_audit=bg1["design_audit"],
         notes="Drop sparse top bin (3.5) using only dense bins")

# SBO variants: defn1 with separate slopes
sbo_immi = sbo_d1[sbo_d1["immigrant"] == 1].copy()
sbo_native = sbo_d1[sbo_d1["immigrant"] == 0].copy()

run_spec("rc/data/source_alt/sbo_defn1_immi_slope",
         "modules/robustness/data.md#source-alt", "G1",
         "log_freq_norm", "log_firm_size", [],
         "", "none", sbo_immi, "HC1",
         f"SBO defn1, immigrant group only ({len(sbo_immi)} obs)", "none (slope only)",
         design_audit=bg1["design_audit"],
         notes="SBO defn1, immigrant group slope")

run_spec("rc/data/source_alt/sbo_defn1_native_slope",
         "modules/robustness/data.md#source-alt", "G1",
         "log_freq_norm", "log_firm_size", [],
         "", "none", sbo_native, "HC1",
         f"SBO defn1, native group only ({len(sbo_native)} obs)", "none (slope only)",
         design_audit=bg1["design_audit"],
         notes="SBO defn1, native group slope")

# G1: defn2 separate slopes
rows_d2_immi = []
rows_d2_native = []
for _, r in defn2.iterrows():
    lfs = r["log_firm_size"]
    rows_d2_immi.append({"log_firm_size": lfs,
                          "log_freq_norm": np.log10(r["freq_immi"] / IMMI_POP)})
    rows_d2_native.append({"log_firm_size": lfs,
                             "log_freq_norm": np.log10(r["freq_native"] / NATIVE_POP)})
df_d2_immi = pd.DataFrame(rows_d2_immi)
df_d2_native = pd.DataFrame(rows_d2_native)

run_spec("rc/data/treatment_alt/defn2_immi_slope",
         "modules/robustness/data.md#treatment-alt", "G1",
         "log_freq_norm", "log_firm_size", [],
         "", "none", df_d2_immi, "HC1",
         "Admin defn2, immigrant group only (7 obs)", "none (slope only)",
         design_audit=bg1["design_audit"],
         notes="Defn2 immigrant separate slope")

run_spec("rc/data/treatment_alt/defn2_native_slope",
         "modules/robustness/data.md#treatment-alt", "G1",
         "log_freq_norm", "log_firm_size", [],
         "", "none", df_d2_native, "HC1",
         "Admin defn2, native group only (7 obs)", "none (slope only)",
         design_audit=bg1["design_audit"],
         notes="Defn2 native separate slope")

# G1: OECD-only separate slope
rows_oecd_only = []
for _, r in defn1.iterrows():
    rows_oecd_only.append({"log_firm_size": r["log_firm_size"],
                            "log_freq_norm": np.log10(r["freq_oecd"] / OECD_POP)})
df_oecd_only = pd.DataFrame(rows_oecd_only)

run_spec("rc/data/treatment_alt/oecd_slope",
         "modules/robustness/data.md#treatment-alt", "G1",
         "log_freq_norm", "log_firm_size", [],
         "", "none", df_oecd_only, "HC1",
         "Admin defn1, OECD immigrant only (7 obs)", "none (slope only)",
         design_audit=bg1["design_audit"],
         notes="OECD immigrant group separate slope")

# G1: non-OECD-only separate slope
rows_noecd_only = []
for _, r in defn1.iterrows():
    rows_noecd_only.append({"log_firm_size": r["log_firm_size"],
                             "log_freq_norm": np.log10(r["freq_noecd"] / NOECD_POP)})
df_noecd_only = pd.DataFrame(rows_noecd_only)

run_spec("rc/data/treatment_alt/noecd_slope",
         "modules/robustness/data.md#treatment-alt", "G1",
         "log_freq_norm", "log_firm_size", [],
         "", "none", df_noecd_only, "HC1",
         "Admin defn1, non-OECD immigrant only (7 obs)", "none (slope only)",
         design_audit=bg1["design_audit"],
         notes="Non-OECD immigrant group separate slope")

# Additional G2 specifications
print("\nRunning additional G2 specifications...")

# G2: bivariate with decade FE + ln_inc_year
run_spec("rc/controls/add/decade_ln_inc",
         "modules/robustness/controls.md#add", "G2",
         "ln_nb_employees", "immi", ["ln_inc_year"],
         "decade_str", "decade", df_g2, "HC1",
         f"Fortune 500, decade FE + ln(inc_year) (N={len(df_g2)})", "ln_inc_year + decade FE",
         design_audit=bg2["design_audit"],
         notes="Decade FE plus log incorporation year control")

# G2: outcome log revenues with decade FE
df_g2_rev_fe = df_g2.dropna(subset=["ln_revenues"]).copy()
run_spec("rc/form/outcome_log_revenues_decade_fe",
         "modules/robustness/form.md#outcome-alt", "G2",
         "ln_revenues", "immi", [],
         "decade_str", "decade", df_g2_rev_fe, "HC1",
         f"Fortune 500, outcome=ln(revenues) + decade FE (N={len(df_g2_rev_fe)})", "decade FE",
         design_audit=bg2["design_audit"],
         notes="Log revenues with decade FE")

# G2: exclude ambiguous firms
df_g2_clean = df_g2[df_g2["ambiguity_code"].isna()].copy()
run_spec("rc/sample/exclude_ambiguous",
         "modules/robustness/sample.md#filter", "G2",
         "ln_nb_employees", "immi", [],
         "", "none", df_g2_clean, "HC1",
         f"Fortune 500, exclude ambiguous (N={len(df_g2_clean)})", "none (bivariate, clean sample)",
         design_audit=bg2["design_audit"],
         notes="Exclude firms with non-missing ambiguity_code (not classic entrepreneurial founding)")

# G2: exclude ambiguous + decade FE
run_spec("rc/sample/exclude_ambiguous_decade_fe",
         "modules/robustness/sample.md#filter", "G2",
         "ln_nb_employees", "immi", [],
         "decade_str", "decade", df_g2_clean, "HC1",
         f"Fortune 500, exclude ambiguous + decade FE (N={len(df_g2_clean)})", "decade FE",
         design_audit=bg2["design_audit"],
         notes="Exclude ambiguous firms + decade FE")

# G2: top 250 firms only
df_g2_top250 = df_g2[df_g2["rank"] <= 250].copy()
run_spec("rc/sample/top250",
         "modules/robustness/sample.md#filter", "G2",
         "ln_nb_employees", "immi", [],
         "", "none", df_g2_top250, "HC1",
         f"Fortune 500, top 250 only (N={len(df_g2_top250)})", "none (bivariate)",
         design_audit=bg2["design_audit"],
         notes="Restrict to top 250 Fortune 500 firms")

# G2: bottom 250 firms only
df_g2_bot250 = df_g2[df_g2["rank"] > 250].copy()
run_spec("rc/sample/bottom250",
         "modules/robustness/sample.md#filter", "G2",
         "ln_nb_employees", "immi", [],
         "", "none", df_g2_bot250, "HC1",
         f"Fortune 500, bottom 250 only (N={len(df_g2_bot250)})", "none (bivariate)",
         design_audit=bg2["design_audit"],
         notes="Restrict to bottom 250 Fortune 500 firms")

# G2: log(nb_employees) with yob_impt + decade FE
run_spec("rc/controls/full/decade_yob_post1970",
         "modules/robustness/controls.md#full", "G2",
         "ln_nb_employees", "immi", ["yob_impt"],
         "decade_str", "decade",
         df_g2_full[df_g2_full["inc_year"] > 1970].copy(), "HC1",
         f"Fortune 500, decade FE + yob, post-1970 (N={len(df_g2_full[df_g2_full['inc_year'] > 1970])})",
         "yob_impt + decade FE (post-1970)",
         design_audit=bg2["design_audit"],
         notes="Full spec restricted to post-1970 firms")

# G2: level outcome (nb_employees, not log)
run_spec("rc/form/outcome_level_employees",
         "modules/robustness/form.md#outcome-alt", "G2",
         "nb_employees", "immi", [],
         "", "none", df_g2, "HC1",
         f"Fortune 500, outcome=nb_employees (level) (N={len(df_g2)})", "none (bivariate)",
         design_audit=bg2["design_audit"],
         notes="Level outcome: nb_employees instead of log")

# G2: post-1970 with decade FE
run_spec("rc/sample/post1970_decade_fe",
         "modules/robustness/sample.md#time-period", "G2",
         "ln_nb_employees", "immi", [],
         "decade_str", "decade", df_g2_post1970, "HC1",
         f"Fortune 500, post-1970 + decade FE (N={len(df_g2_post1970)})", "decade FE (post-1970)",
         design_audit=bg2["design_audit"],
         notes="Post-1970 with decade FE")

# G2: pre-1970 with decade FE
run_spec("rc/sample/pre1970_decade_fe",
         "modules/robustness/sample.md#time-period", "G2",
         "ln_nb_employees", "immi", [],
         "decade_str", "decade", df_g2_pre1970, "HC1",
         f"Fortune 500, pre-1970 + decade FE (N={len(df_g2_pre1970)})", "decade FE (pre-1970)",
         design_audit=bg2["design_audit"],
         notes="Pre-1970 with decade FE")

# G2: log revenues, post-1970
df_g2_rev_post = df_g2_rev[df_g2_rev["inc_year"] > 1970].copy()
run_spec("rc/form/outcome_log_revenues_post1970",
         "modules/robustness/form.md#outcome-alt", "G2",
         "ln_revenues", "immi", [],
         "", "none", df_g2_rev_post, "HC1",
         f"Fortune 500, ln(revenues), post-1970 (N={len(df_g2_rev_post)})", "none (bivariate, post-1970)",
         design_audit=bg2["design_audit"],
         notes="Log revenues, restricted to post-1970 firms")

# G2 inference: HC1 for decade FE spec
run_inference_variant(
    "infer/se/hc/hc1_decade_fe", g2_baseline_id,
    "ln_nb_employees", "immi", [],
    "decade_str", df_g2, "HC1", "HC1 with decade FE",
    "G2", notes="HC1 with decade FE added"
)

# G2 inference: nonrobust for decade FE
run_inference_variant(
    "infer/se/nonrobust_decade_fe", g2_baseline_id,
    "ln_nb_employees", "immi", [],
    "decade_str", df_g2, "iid", "iid with decade FE",
    "G2", notes="Classical OLS SEs with decade FE"
)

# G2: log revenues with controls
df_g2_rev_ctrl = df_g2.dropna(subset=["ln_revenues", "yob_impt"]).copy()
run_spec("rc/form/outcome_log_revenues_full",
         "modules/robustness/form.md#outcome-alt", "G2",
         "ln_revenues", "immi", ["yob_impt"],
         "decade_str", "decade", df_g2_rev_ctrl, "HC1",
         f"Fortune 500, ln(revenues) + yob + decade FE (N={len(df_g2_rev_ctrl)})",
         "yob_impt + decade FE",
         design_audit=bg2["design_audit"],
         notes="Log revenues with full controls")

# G2: exclude very old firms (pre-1900)
df_g2_modern = df_g2[df_g2["inc_year"] >= 1900].copy()
run_spec("rc/sample/post1900",
         "modules/robustness/sample.md#time-period", "G2",
         "ln_nb_employees", "immi", [],
         "", "none", df_g2_modern, "HC1",
         f"Fortune 500, post-1900 only (N={len(df_g2_modern)})", "none (bivariate, post-1900)",
         design_audit=bg2["design_audit"],
         notes="Exclude firms incorporated before 1900")

# G2: inverted rank with decade FE
run_spec("rc/form/outcome_rank_decade_fe",
         "modules/robustness/form.md#outcome-alt", "G2",
         "rank_inv", "immi", [],
         "decade_str", "decade", df_g2_rank, "HC1",
         f"Fortune 500, inverted rank + decade FE (N={len(df_g2_rank)})", "decade FE",
         design_audit=bg2["design_audit"],
         notes="Inverted rank with decade FE")


# ============================================================
# Save results
# ============================================================

print("\n" + "=" * 60)
print("Saving results...")
print("=" * 60)

spec_df = pd.DataFrame(results)
spec_df.to_csv(f"{OUTPUT_DIR}/specification_results.csv", index=False)
print(f"Wrote specification_results.csv: {len(spec_df)} rows")

infer_df = pd.DataFrame(inference_results)
infer_df.to_csv(f"{OUTPUT_DIR}/inference_results.csv", index=False)
print(f"Wrote inference_results.csv: {len(infer_df)} rows")

# ============================================================
# Summary statistics
# ============================================================

successful = spec_df[spec_df['run_success'] == 1]
failed = spec_df[spec_df['run_success'] == 0]

print("\n=== SPECIFICATION RESULTS SUMMARY ===")
print(f"Total rows: {len(spec_df)}")
print(f"Successful: {len(successful)}")
print(f"Failed: {len(failed)}")

for gid in ["G1", "G2"]:
    g_df = successful[successful['baseline_group_id'] == gid]
    base_row = spec_df[spec_df['spec_id'].str.startswith(f'baseline/{gid}/')]
    print(f"\n--- {gid} ---")
    if len(base_row) > 0:
        bc = base_row.iloc[0]
        print(f"  Baseline coef: {bc['coefficient']:.6f}, SE: {bc['std_error']:.6f}, p: {bc['p_value']:.6f}, N: {bc['n_obs']:.0f}")
    if len(g_df) > 0:
        print(f"  Total specs: {len(g_df)}")
        n_sig = (g_df['p_value'] < 0.05).sum()
        print(f"  Significant at 5%: {n_sig}/{len(g_df)}")
        print(f"  Coef range: [{g_df['coefficient'].min():.6f}, {g_df['coefficient'].max():.6f}]")


# ============================================================
# SPECIFICATION_SEARCH.md
# ============================================================

print("\nWriting SPECIFICATION_SEARCH.md...")

md_lines = []
md_lines.append("# Specification Search Report: 134622-V1")
md_lines.append("")
md_lines.append("**Paper:** Azoulay, Jones, Kim & Miranda (2022), \"Immigration and Entrepreneurship in the United States\", AER: Insights 4(1)")
md_lines.append("")
md_lines.append("## Baseline Specifications")
md_lines.append("")

# G1 baseline
md_lines.append("### G1: Administrative Data (Firm Size Power Law)")
md_lines.append("")
md_lines.append("- **Design:** Cross-sectional OLS (aggregated log-log regression)")
md_lines.append("- **Outcome:** log_freq_norm (log10 of population-normalized firm count)")
md_lines.append("- **Treatment:** immigrant (indicator)")
md_lines.append("- **Controls:** log_firm_size, immi_x_size (interaction)")
md_lines.append("- **Fixed effects:** None")
md_lines.append("- **SEs:** HC1")
md_lines.append("")

g1_base_row = spec_df[spec_df['spec_id'] == 'baseline/G1/Admin-Defn1-Pooled']
if len(g1_base_row) > 0:
    bc = g1_base_row.iloc[0]
    md_lines.append(f"| Statistic | Value |")
    md_lines.append(f"|-----------|-------|")
    md_lines.append(f"| Coefficient (immigrant) | {bc['coefficient']:.6f} |")
    md_lines.append(f"| Std. Error | {bc['std_error']:.6f} |")
    md_lines.append(f"| p-value | {bc['p_value']:.6f} |")
    md_lines.append(f"| 95% CI | [{bc['ci_lower']:.6f}, {bc['ci_upper']:.6f}] |")
    md_lines.append(f"| N | {bc['n_obs']:.0f} |")
    md_lines.append(f"| R-squared | {bc['r_squared']:.4f} |")
    md_lines.append("")

# G2 baseline
md_lines.append("### G2: Fortune 500 Firm-Level")
md_lines.append("")
md_lines.append("- **Design:** Cross-sectional OLS")
md_lines.append("- **Outcome:** ln_nb_employees (log of employee count)")
md_lines.append("- **Treatment:** immi (at least 1 foreign-born founder)")
md_lines.append("- **Controls:** None (bivariate baseline)")
md_lines.append("- **Fixed effects:** None")
md_lines.append("- **SEs:** HC1")
md_lines.append("")

g2_base_row = spec_df[spec_df['spec_id'] == 'baseline/G2/F500-OLS-base']
if len(g2_base_row) > 0:
    bc = g2_base_row.iloc[0]
    md_lines.append(f"| Statistic | Value |")
    md_lines.append(f"|-----------|-------|")
    md_lines.append(f"| Coefficient (immi) | {bc['coefficient']:.6f} |")
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

# Category breakdown by group
for gid, gname in [("G1", "Admin Data"), ("G2", "Fortune 500")]:
    g_succ = successful[successful['baseline_group_id'] == gid]
    md_lines.append(f"### {gid}: {gname}")
    md_lines.append("")
    md_lines.append("| Category | Count | Sig. (p<0.05) | Coef Range |")
    md_lines.append("|----------|-------|---------------|------------|")

    categories = {
        "Baseline": g_succ[g_succ['spec_id'].str.startswith('baseline/')],
        "Treatment Alternatives": g_succ[g_succ['spec_id'].str.contains('treatment_alt|defn')],
        "Source Alternatives": g_succ[g_succ['spec_id'].str.contains('source_alt')],
        "Sample Variants": g_succ[g_succ['spec_id'].str.contains('sample/')],
        "Functional Form": g_succ[g_succ['spec_id'].str.contains('form/')],
        "Controls": g_succ[g_succ['spec_id'].str.contains('controls/')],
        "FE Variants": g_succ[g_succ['spec_id'].str.contains('fe/')],
        "Estimation": g_succ[g_succ['spec_id'].str.contains('estimation/')],
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

for gid, treatment_var in [("G1", "immigrant"), ("G2", "immi")]:
    g_succ = successful[successful['baseline_group_id'] == gid]
    if len(g_succ) > 0:
        md_lines.append(f"### {gid}")
        n_sig_total = (g_succ['p_value'] < 0.05).sum()
        pct_sig = n_sig_total / len(g_succ) * 100
        sign_consistent = ((g_succ['coefficient'] > 0).sum() == len(g_succ)) or \
                          ((g_succ['coefficient'] < 0).sum() == len(g_succ))
        median_coef = g_succ['coefficient'].median()
        sign_word = "positive" if median_coef > 0 else "negative"

        md_lines.append(f"- **Sign consistency:** {'All specifications have the same sign' if sign_consistent else 'Mixed signs across specifications'}")
        md_lines.append(f"- **Significance stability:** {n_sig_total}/{len(g_succ)} ({pct_sig:.1f}%) specifications significant at 5%")
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
        md_lines.append("")

md_lines.append(f"Surface hash: `{SURFACE_HASH}`")
md_lines.append("")

with open(f"{OUTPUT_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write("\n".join(md_lines))

print("Wrote SPECIFICATION_SEARCH.md")
print("\nDone!")
