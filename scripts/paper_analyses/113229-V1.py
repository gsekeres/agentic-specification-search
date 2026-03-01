"""
Specification Search Script for Bohnet, Greig, Herrmann & Zeckhauser (2008)
"Betrayal Aversion: Evidence from Brazil, China, Oman, Switzerland, Turkey,
and the United States"
American Economic Review, 98(1), 294-310.

Paper ID: 113229-V1

Surface-driven execution:
  - G1: map ~ tg + dp (OLS clustered by session)
  - Randomized experiment (lab): subjects assigned to TG, RDG, or DP game type
  - Focal coefficient: tg (betrayal aversion = TG MAP premium over RDG)
  - Sample: movers only (mover==1)

Outputs:
  - specification_results.csv (baseline, design/*, rc/* rows)
  - inference_results.csv (infer/* rows)
  - SPECIFICATION_SEARCH.md
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import statsmodels.formula.api as smf
import statsmodels.api as sm
import json
import sys
import warnings
import random
from itertools import combinations

warnings.filterwarnings('ignore')

# Add scripts dir for utilities
sys.path.insert(0, "scripts")
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash, software_block
)

PAPER_ID = "113229-V1"
DATA_DIR = "data/downloads/extracted/113229-V1"
OUTPUT_DIR = DATA_DIR

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

# Design audit block from surface
G1_DESIGN_AUDIT = surface_obj["baseline_groups"][0]["design_audit"]
G1_INFERENCE_CANONICAL = surface_obj["baseline_groups"][0]["inference_plan"]["canonical"]

# ============================================================
# LOAD DATA
# ============================================================
df_raw = pd.read_stata(f"{DATA_DIR}/betrayal-data-file.dta")
for col in df_raw.columns:
    if df_raw[col].dtype == np.float32:
        df_raw[col] = df_raw[col].astype(np.float64)

# Movers sample (mover==1) -- this is the analysis sample
df = df_raw[df_raw['mover'] == 1].copy()
print(f"Movers sample: N={len(df)}")
print(f"  TG: {(df['tg']==1).sum()}, DP: {(df['dp']==1).sum()}, RDG: {((df['tg']==0)&(df['dp']==0)).sum()}")
print(f"  Unique sessions: {df['session'].nunique()}")

results = []
inference_results = []
spec_run_counter = 0

# ============================================================
# CONTROL VARIABLE DEFINITIONS
# ============================================================
# Non-interaction controls (9)
demographics = ["female", "age", "income", "econmajor"]
country_dummies = ["brazil", "china", "oman", "swiss", "turk"]
# Note: us is the omitted country category (6 dummies for 6 countries, one omitted)
all_non_interaction_controls = demographics + country_dummies

# Interaction controls (from Table 2 Col 3)
interaction_controls = ["female_tg", "female_dp", "oman_tg", "oman_dp"]

# Full control set for Col 2 and Col 3
controls_col2 = all_non_interaction_controls  # 9 controls
controls_col3 = all_non_interaction_controls + interaction_controls  # 13 controls


# ============================================================
# HELPER: Run OLS via pyfixest
# ============================================================
def run_ols(spec_id, spec_tree_path, baseline_group_id,
            outcome_var, treatment_var, controls, fe_formula,
            data, vcov, sample_desc, controls_desc, cluster_var,
            design_audit, inference_canonical,
            axis_block_name=None, axis_block=None, notes=""):
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
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
        # Focal coefficient is the first variable in treatment_var (tg)
        focal_var = treatment_var.split(" + ")[0].strip()
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
            design={"randomized_experiment": design_audit},
            axis_block_name=axis_block_name,
            axis_block=axis_block,
            notes=notes if notes else None,
        )

        results.append({
            "paper_id": PAPER_ID, "spec_run_id": run_id,
            "spec_id": spec_id, "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var, "treatment_var": focal_var,
            "coefficient": coef_val, "std_error": se_val, "p_value": pval,
            "ci_lower": ci_lower, "ci_upper": ci_upper,
            "n_obs": nobs, "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc, "fixed_effects": fe_formula or "none",
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
            "outcome_var": outcome_var, "treatment_var": treatment_var.split(" + ")[0].strip(),
            "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "n_obs": np.nan, "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc, "fixed_effects": fe_formula or "none",
            "controls_desc": controls_desc, "cluster_var": cluster_var,
            "run_success": 0, "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


# ============================================================
# HELPER: Run inference variant (recompute SE/p-value under alternative inference)
# ============================================================
def run_inference_variant(base_run_id, infer_spec_id, spec_tree_path,
                          baseline_group_id, outcome_var, treatment_var,
                          controls, fe_formula, data, vcov_alt, cluster_var_alt,
                          design_audit):
    global spec_run_counter
    spec_run_counter += 1
    infer_run_id = f"{PAPER_ID}_infer_{spec_run_counter:03d}"

    try:
        controls_str = " + ".join(controls) if controls else ""
        if controls_str and fe_formula:
            formula = f"{outcome_var} ~ {treatment_var} + {controls_str} | {fe_formula}"
        elif controls_str:
            formula = f"{outcome_var} ~ {treatment_var} + {controls_str}"
        elif fe_formula:
            formula = f"{outcome_var} ~ {treatment_var} | {fe_formula}"
        else:
            formula = f"{outcome_var} ~ {treatment_var}"

        m = pf.feols(formula, data=data, vcov=vcov_alt)
        focal_var = treatment_var.split(" + ")[0].strip()
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
            inference={"spec_id": infer_spec_id, "params": {}},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"randomized_experiment": design_audit},
        )

        inference_results.append({
            "paper_id": PAPER_ID,
            "inference_run_id": infer_run_id,
            "spec_run_id": base_run_id,
            "spec_id": infer_spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": focal_var,
            "coefficient": coef_val, "std_error": se_val, "p_value": pval,
            "ci_lower": ci_lower, "ci_upper": ci_upper,
            "n_obs": nobs, "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "cluster_var": cluster_var_alt if isinstance(cluster_var_alt, str) else "",
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
            "paper_id": PAPER_ID,
            "inference_run_id": infer_run_id,
            "spec_run_id": base_run_id,
            "spec_id": infer_spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": treatment_var.split(" + ")[0].strip(),
            "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "n_obs": np.nan, "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "cluster_var": cluster_var_alt if isinstance(cluster_var_alt, str) else "",
            "run_success": 0, "run_error": err_msg
        })
        return infer_run_id


# ============================================================
# VCOV setup (canonical: cluster by session)
# ============================================================
canonical_vcov = {"CRV1": "session"}
treatment_formula = "tg + dp"  # Both dummies, focal is tg

# ############################################################
# BASELINES (Table 2, Columns 1-3)
# ############################################################
print("=== BASELINES ===")

# Baseline: Table 2 Col 1 -- no controls
run_id_bl_col1, *_ = run_ols(
    "baseline", "specification_tree/designs/randomized_experiment.md#baseline", "G1",
    "map", treatment_formula, [], "",
    df, canonical_vcov,
    "movers (mover==1), all countries, N=494", "none", "session",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    notes="Table 2 Col 1: map ~ tg + dp, cl(session). No controls."
)

# Baseline: Table 2 Col 2 -- with controls
run_id_bl_col2, *_ = run_ols(
    "baseline__table2_col2", "specification_tree/designs/randomized_experiment.md#baseline", "G1",
    "map", treatment_formula, controls_col2, "",
    df, canonical_vcov,
    "movers (mover==1), all countries", ", ".join(controls_col2), "session",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    notes="Table 2 Col 2: map ~ tg + dp + demographics + country, cl(session)."
)

# Baseline: Table 2 Col 3 -- with controls + interactions
run_id_bl_col3, *_ = run_ols(
    "baseline__table2_col3", "specification_tree/designs/randomized_experiment.md#baseline", "G1",
    "map", treatment_formula, controls_col3, "",
    df, canonical_vcov,
    "movers (mover==1), all countries", ", ".join(controls_col3), "session",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    notes="Table 2 Col 3: map ~ tg + dp + demographics + country + interactions, cl(session)."
)

# ############################################################
# DESIGN VARIANTS
# ############################################################
print("=== DESIGN VARIANTS ===")

# design/randomized_experiment/estimator/diff_in_means
run_ols(
    "design/randomized_experiment/estimator/diff_in_means",
    "specification_tree/designs/randomized_experiment.md#estimator-variants", "G1",
    "map", treatment_formula, [], "",
    df, "hetero",
    "movers, all countries", "none (robust SE, no clustering)", "",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    notes="Diff-in-means: map ~ tg + dp, HC1 robust SE (no clustering)."
)

# design/randomized_experiment/estimator/with_covariates
run_ols(
    "design/randomized_experiment/estimator/with_covariates",
    "specification_tree/designs/randomized_experiment.md#estimator-variants", "G1",
    "map", treatment_formula, controls_col2, "",
    df, canonical_vcov,
    "movers, all countries", ", ".join(controls_col2), "session",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    notes="Design variant: OLS with pre-treatment covariates, cl(session)."
)

# ############################################################
# RC: CONTROLS AXIS -- Single additions (surface lists 4 demographics only)
# ############################################################
print("=== RC: CONTROLS (single additions) ===")

# Surface rc_spec_ids: rc/controls/single/add_female, add_age, add_income, add_econmajor
single_controls_in_surface = ["female", "age", "income", "econmajor"]

for ctrl in single_controls_in_surface:
    run_ols(
        f"rc/controls/single/add_{ctrl}",
        "modules/robustness/controls.md#single-control-addition", "G1",
        "map", treatment_formula, [ctrl], "",
        df, canonical_vcov,
        "movers, all countries", ctrl, "session",
        G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/controls/single/add_{ctrl}", "family": "single",
                    "added": [ctrl], "dropped": [], "n_controls": 1}
    )

# ############################################################
# RC: CONTROLS AXIS -- Standard sets
# ############################################################
print("=== RC: CONTROLS (standard sets) ===")

# Demographics only
run_ols(
    "rc/controls/sets/demographics",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "map", treatment_formula, demographics, "",
    df, canonical_vcov,
    "movers, all countries", ", ".join(demographics), "session",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/demographics", "family": "sets",
                "added": demographics, "dropped": country_dummies, "n_controls": 4,
                "set_name": "demographics"}
)

# Country dummies only
run_ols(
    "rc/controls/sets/country_dummies",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "map", treatment_formula, country_dummies, "",
    df, canonical_vcov,
    "movers, all countries", ", ".join(country_dummies), "session",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/country_dummies", "family": "sets",
                "added": country_dummies, "dropped": demographics, "n_controls": 5,
                "set_name": "country_dummies"}
)

# Demographics + country (= Col 2 controls)
run_ols(
    "rc/controls/sets/demographics_plus_country",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "map", treatment_formula, all_non_interaction_controls, "",
    df, canonical_vcov,
    "movers, all countries", ", ".join(all_non_interaction_controls), "session",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/demographics_plus_country", "family": "sets",
                "added": all_non_interaction_controls, "dropped": [], "n_controls": 9,
                "set_name": "demographics_plus_country"}
)

# ############################################################
# RC: CONTROLS AXIS -- Progression (build-up)
# Surface: rc/controls/progression/* (wildcard)
# ############################################################
print("=== RC: CONTROLS (progression) ===")

progression_sets = [
    ("female_only", ["female"]),
    ("female_age", ["female", "age"]),
    ("female_age_income", ["female", "age", "income"]),
    ("demographics_all", demographics),
    ("demographics_plus_brazil", demographics + ["brazil"]),
    ("demographics_plus_brazil_china", demographics + ["brazil", "china"]),
    ("demographics_plus_brazil_china_oman", demographics + ["brazil", "china", "oman"]),
    ("demographics_plus_br_cn_om_sw", demographics + ["brazil", "china", "oman", "swiss"]),
    ("full_non_interaction", all_non_interaction_controls),
]

for prog_name, prog_ctrls in progression_sets:
    run_ols(
        f"rc/controls/progression/{prog_name}",
        "modules/robustness/controls.md#progressive-control-addition", "G1",
        "map", treatment_formula, prog_ctrls, "",
        df, canonical_vcov,
        "movers, all countries", ", ".join(prog_ctrls), "session",
        G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/controls/progression/{prog_name}", "family": "progression",
                    "added": prog_ctrls, "dropped": [], "n_controls": len(prog_ctrls),
                    "progression_step": prog_name}
    )

# ############################################################
# RC: CONTROLS AXIS -- Random subsets (seed=113229)
# Surface: rc/controls/subset/random_* (wildcard)
# ############################################################
print("=== RC: CONTROLS (random subsets) ===")

rng = random.Random(113229)
all_possible_subsets = []
for size in range(1, len(all_non_interaction_controls)):
    for combo in combinations(all_non_interaction_controls, size):
        all_possible_subsets.append(list(combo))

# Sample 15 random subsets
sampled_subsets = rng.sample(all_possible_subsets, min(15, len(all_possible_subsets)))

for idx, subset in enumerate(sampled_subsets):
    subset_name = f"random_{idx+1:02d}"
    run_ols(
        f"rc/controls/subset/{subset_name}",
        "modules/robustness/controls.md#random-control-subsets", "G1",
        "map", treatment_formula, subset, "",
        df, canonical_vcov,
        "movers, all countries", ", ".join(subset), "session",
        G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/controls/subset/{subset_name}", "family": "random_subset",
                    "added": subset, "dropped": [c for c in all_non_interaction_controls if c not in subset],
                    "n_controls": len(subset), "seed": 113229, "draw_index": idx}
    )

# ############################################################
# RC: SAMPLE RESTRICTIONS -- Drop one country at a time
# Surface: rc/sample/restriction/drop_country_brazil, ..., drop_country_us
# ############################################################
print("=== RC: SAMPLE (drop country) ===")

country_drop_map = {
    "brazil": "brazil",
    "china": "china",
    "oman": "oman",
    "switzerland": "swiss",
    "turkey": "turk",
    "us": "us"
}

for country_label, country_var in country_drop_map.items():
    df_sub = df[df[country_var] != 1].copy()
    run_ols(
        f"rc/sample/restriction/drop_country_{country_label}",
        "modules/robustness/sample.md#subsample-restrictions", "G1",
        "map", treatment_formula, controls_col2, "",
        df_sub, canonical_vcov,
        f"movers, drop {country_label} (N={len(df_sub)})",
        ", ".join(controls_col2), "session",
        G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
        axis_block_name="sample",
        axis_block={"spec_id": f"rc/sample/restriction/drop_country_{country_label}",
                    "axis": "subsample", "restriction": f"{country_var}!=1",
                    "n_obs_before": len(df), "n_obs_after": len(df_sub)}
    )

# ############################################################
# RC: SAMPLE RESTRICTIONS -- Gender subsamples
# Surface: rc/sample/restriction/women_only, rc/sample/restriction/men_only
# ############################################################
print("=== RC: SAMPLE (gender) ===")

# Women only
df_women = df[df['female'] == 1].copy()
run_ols(
    "rc/sample/restriction/women_only",
    "modules/robustness/sample.md#subsample-restrictions", "G1",
    "map", treatment_formula, [c for c in controls_col2 if c != "female"], "",
    df_women, canonical_vcov,
    f"movers, women only (N={len(df_women)})",
    ", ".join([c for c in controls_col2 if c != "female"]), "session",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/women_only", "axis": "subsample",
                "restriction": "female==1", "n_obs_before": len(df), "n_obs_after": len(df_women)}
)

# Men only
df_men = df[df['female'] == 0].copy()
run_ols(
    "rc/sample/restriction/men_only",
    "modules/robustness/sample.md#subsample-restrictions", "G1",
    "map", treatment_formula, [c for c in controls_col2 if c != "female"], "",
    df_men, canonical_vcov,
    f"movers, men only (N={len(df_men)})",
    ", ".join([c for c in controls_col2 if c != "female"]), "session",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/men_only", "axis": "subsample",
                "restriction": "female==0", "n_obs_before": len(df), "n_obs_after": len(df_men)}
)

# ############################################################
# RC: SAMPLE -- Outlier trimming
# Surface: rc/sample/outliers/trim_map_1_99, rc/sample/outliers/trim_map_5_95
# ############################################################
print("=== RC: SAMPLE (outlier trimming) ===")

# Trim MAP at 1st/99th percentiles
q01 = df['map'].quantile(0.01)
q99 = df['map'].quantile(0.99)
df_trim_1_99 = df[(df['map'] >= q01) & (df['map'] <= q99)].copy()
run_ols(
    "rc/sample/outliers/trim_map_1_99",
    "modules/robustness/sample.md#outliers-and-influential-observations", "G1",
    "map", treatment_formula, controls_col2, "",
    df_trim_1_99, canonical_vcov,
    f"movers, MAP in [{q01:.3f}, {q99:.3f}] (N={len(df_trim_1_99)})",
    ", ".join(controls_col2), "session",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_map_1_99", "axis": "outliers",
                "rule": "trim", "params": {"var": "map", "lower_q": 0.01, "upper_q": 0.99},
                "n_obs_before": len(df), "n_obs_after": len(df_trim_1_99)}
)

# Trim MAP at 5th/95th percentiles
q05 = df['map'].quantile(0.05)
q95 = df['map'].quantile(0.95)
df_trim_5_95 = df[(df['map'] >= q05) & (df['map'] <= q95)].copy()
run_ols(
    "rc/sample/outliers/trim_map_5_95",
    "modules/robustness/sample.md#outliers-and-influential-observations", "G1",
    "map", treatment_formula, controls_col2, "",
    df_trim_5_95, canonical_vcov,
    f"movers, MAP in [{q05:.3f}, {q95:.3f}] (N={len(df_trim_5_95)})",
    ", ".join(controls_col2), "session",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_map_5_95", "axis": "outliers",
                "rule": "trim", "params": {"var": "map", "lower_q": 0.05, "upper_q": 0.95},
                "n_obs_before": len(df), "n_obs_after": len(df_trim_5_95)}
)

# ############################################################
# RC: PREPROCESSING -- Complete cases only
# Surface: rc/preprocess/missing/complete_cases_only
# ############################################################
print("=== RC: PREPROCESS (complete cases) ===")

all_vars = ["map", "tg", "dp", "session"] + controls_col2
df_complete = df.dropna(subset=all_vars).copy()
run_ols(
    "rc/preprocess/missing/complete_cases_only",
    "modules/robustness/preprocessing.md#missing-data-handling", "G1",
    "map", treatment_formula, controls_col2, "",
    df_complete, canonical_vcov,
    f"movers, complete cases only (N={len(df_complete)})",
    ", ".join(controls_col2), "session",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="preprocess",
    axis_block={"spec_id": "rc/preprocess/missing/complete_cases_only",
                "action": "complete_cases", "vars_checked": all_vars,
                "n_obs_before": len(df), "n_obs_after": len(df_complete)}
)

# ############################################################
# RC: FUNCTIONAL FORM -- Logit(MAP)
# Surface: rc/form/outcome/logit_map
# ############################################################
print("=== RC: FUNCTIONAL FORM ===")

# Logit(MAP): transform bounded [0,1] outcome
df_logit = df.copy()
map_clipped = df_logit['map'].clip(0.001, 0.999)
df_logit['logit_map'] = np.log(map_clipped / (1 - map_clipped))

run_ols(
    "rc/form/outcome/logit_map",
    "modules/robustness/functional_form.md#outcome-transformations", "G1",
    "logit_map", treatment_formula, controls_col2, "",
    df_logit, canonical_vcov,
    "movers, logit(MAP) outcome", ", ".join(controls_col2), "session",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/logit_map",
                "outcome_transform": "logit",
                "treatment_transform": "level",
                "interpretation": "Logit-transformed MAP = log(MAP/(1-MAP)). Accounts for bounded [0,1] nature of outcome. Coefficients represent log-odds change."}
)

# ############################################################
# RC: TREATMENT ISOLATION
# Surface: rc/form/treatment/tg_only_vs_rdg, rc/form/treatment/dp_only_vs_rdg
# ############################################################
print("=== RC: TREATMENT ISOLATION ===")

# TG vs RDG only (drop DP observations, type==1)
df_tg_rdg = df[df['dp'] != 1].copy()
run_ols(
    "rc/form/treatment/tg_only_vs_rdg",
    "modules/robustness/functional_form.md#treatment-transformations", "G1",
    "map", "tg", controls_col2, "",
    df_tg_rdg, canonical_vcov,
    f"movers, TG vs RDG only (N={len(df_tg_rdg)})",
    ", ".join(controls_col2), "session",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/treatment/tg_only_vs_rdg",
                "treatment_transform": "binary_tg_vs_rdg",
                "outcome_transform": "level",
                "interpretation": "Pairwise comparison: TG vs RDG only. DP observations excluded. Isolates betrayal aversion from strategic uncertainty."}
)

# DP vs RDG only (drop TG observations)
df_dp_rdg = df[df['tg'] != 1].copy()
run_ols(
    "rc/form/treatment/dp_only_vs_rdg",
    "modules/robustness/functional_form.md#treatment-transformations", "G1",
    "map", "dp", controls_col2, "",
    df_dp_rdg, canonical_vcov,
    f"movers, DP vs RDG only (N={len(df_dp_rdg)})",
    ", ".join(controls_col2), "session",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/treatment/dp_only_vs_rdg",
                "treatment_transform": "binary_dp_vs_rdg",
                "outcome_transform": "level",
                "interpretation": "Pairwise comparison: DP vs RDG only. TG observations excluded. Tests strategic uncertainty aversion."}
)

# ############################################################
# INFERENCE VARIANTS
# ############################################################
print("=== INFERENCE VARIANTS ===")

# For baseline Col 1 (no controls) -- HC1, HC3
run_inference_variant(
    run_id_bl_col1, "infer/se/hc/hc1",
    "modules/inference/standard_errors.md#heteroskedasticity-robust",
    "G1", "map", treatment_formula, [], "", df,
    "hetero", "", G1_DESIGN_AUDIT
)
run_inference_variant(
    run_id_bl_col1, "infer/se/hc/hc3",
    "modules/inference/standard_errors.md#heteroskedasticity-robust",
    "G1", "map", treatment_formula, [], "", df,
    {"CRV3": "session"}, "session", G1_DESIGN_AUDIT
)

# For baseline Col 2 (with controls) -- HC1, HC3
run_inference_variant(
    run_id_bl_col2, "infer/se/hc/hc1",
    "modules/inference/standard_errors.md#heteroskedasticity-robust",
    "G1", "map", treatment_formula, controls_col2, "", df,
    "hetero", "", G1_DESIGN_AUDIT
)
run_inference_variant(
    run_id_bl_col2, "infer/se/hc/hc3",
    "modules/inference/standard_errors.md#heteroskedasticity-robust",
    "G1", "map", treatment_formula, controls_col2, "", df,
    {"CRV3": "session"}, "session", G1_DESIGN_AUDIT
)

# For baseline Col 3 (with interactions) -- HC1, HC3
run_inference_variant(
    run_id_bl_col3, "infer/se/hc/hc1",
    "modules/inference/standard_errors.md#heteroskedasticity-robust",
    "G1", "map", treatment_formula, controls_col3, "", df,
    "hetero", "", G1_DESIGN_AUDIT
)
run_inference_variant(
    run_id_bl_col3, "infer/se/hc/hc3",
    "modules/inference/standard_errors.md#heteroskedasticity-robust",
    "G1", "map", treatment_formula, controls_col3, "", df,
    {"CRV3": "session"}, "session", G1_DESIGN_AUDIT
)


# ############################################################
# SAVE OUTPUTS
# ############################################################
print(f"\n=== SAVING OUTPUTS ===")
print(f"Total spec results: {len(results)}")
print(f"Total inference results: {len(inference_results)}")

# specification_results.csv
df_results = pd.DataFrame(results)
df_results.to_csv(f"{OUTPUT_DIR}/specification_results.csv", index=False)
print(f"Wrote {OUTPUT_DIR}/specification_results.csv ({len(df_results)} rows)")

# inference_results.csv
df_infer = pd.DataFrame(inference_results)
df_infer.to_csv(f"{OUTPUT_DIR}/inference_results.csv", index=False)
print(f"Wrote {OUTPUT_DIR}/inference_results.csv ({len(df_infer)} rows)")

# Summary stats
n_success = df_results['run_success'].sum()
n_fail = len(df_results) - n_success
print(f"\nSpec results: {n_success} success, {n_fail} failed")
n_infer_success = df_infer['run_success'].sum()
n_infer_fail = len(df_infer) - n_infer_success
print(f"Inference results: {n_infer_success} success, {n_infer_fail} failed")

# Print baseline results
print("\n=== BASELINE RESULTS ===")
for _, row in df_results[df_results['spec_id'].str.startswith('baseline')].iterrows():
    print(f"  {row['spec_id']}: coef={row['coefficient']:.4f}, se={row['std_error']:.4f}, "
          f"p={row['p_value']:.4f}, N={row['n_obs']:.0f}")

# ############################################################
# SPECIFICATION_SEARCH.md
# ############################################################
md_content = f"""# Specification Search: {PAPER_ID}

## Paper
- **Title**: Betrayal Aversion: Evidence from Brazil, China, Oman, Switzerland, Turkey, and the United States
- **Authors**: Bohnet, Greig, Herrmann & Zeckhauser (AER 2008)
- **Design**: Randomized experiment (lab)
- **Paper ID**: {PAPER_ID}

## Surface Summary
- **Baseline groups**: 1 (G1: Betrayal Aversion)
- **Budget**: max 60 core specs, 15 control subsets
- **Seed**: 113229
- **Canonical inference**: Cluster SE at session level (CRV1)

## Baseline Group G1
- **Claim**: TG MAP > RDG MAP (betrayal aversion)
- **Outcome**: map (minimum acceptable probability)
- **Treatment**: tg (Trust Game dummy), with dp (Dictator-Principal dummy); RDG is omitted
- **Sample**: movers only (mover==1), N=494
- **Baseline specs**: Table 2 Cols 1-3

## Execution Summary

### Counts
- **Planned core specs**: {len(df_results)}
- **Executed successfully**: {n_success}
- **Failed**: {n_fail}
- **Inference variants**: {len(df_infer)} ({n_infer_success} success, {n_infer_fail} failed)

### Spec Breakdown
| Category | Count |
|----------|-------|
| Baselines (Table 2 Cols 1-3) | 3 |
| Design variants | 2 |
| Controls: single additions | 4 |
| Controls: standard sets | 3 |
| Controls: progression | {len(progression_sets)} |
| Controls: random subsets | 15 |
| Sample: drop country (with controls) | 6 |
| Sample: gender subsamples | 2 |
| Sample: outlier trimming | 2 |
| Preprocessing: complete cases | 1 |
| Functional form: logit MAP | 1 |
| Treatment isolation: TG vs RDG, DP vs RDG | 2 |
| **Total core specs** | **{len(df_results)}** |

### Baseline Results
"""

for _, row in df_results[df_results['spec_id'].str.startswith('baseline')].iterrows():
    md_content += f"- **{row['spec_id']}**: coef={row['coefficient']:.4f}, se={row['std_error']:.4f}, p={row['p_value']:.4f}, N={row['n_obs']:.0f}, R2={row['r_squared']:.4f}\n"

md_content += f"""
### Inference Variants
- HC1 (heteroskedasticity-robust, no clustering) for all 3 baselines
- HC3/CRV3 (jackknife) for all 3 baselines
- Total: {len(df_infer)} inference recomputations

### Deviations from Surface
- None. All planned axes executed.
- Wild cluster bootstrap not implemented (package not available; noted in surface review as optional).

### Software Stack
- Python {SW_BLOCK['runner_version']}
- pyfixest {SW_BLOCK['packages'].get('pyfixest', 'N/A')}
- pandas {SW_BLOCK['packages'].get('pandas', 'N/A')}
- numpy {SW_BLOCK['packages'].get('numpy', 'N/A')}
- statsmodels {SW_BLOCK['packages'].get('statsmodels', 'N/A')}
"""

with open(f"{OUTPUT_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write(md_content)
print(f"Wrote {OUTPUT_DIR}/SPECIFICATION_SEARCH.md")

print("\n=== DONE ===")
