"""
Specification Search Script for Goni (2023)
"Assortative Matching at the Top of the Distribution:
 Evidence from the World's Most Exclusive Marriage Market"
American Economic Review, 113(2), 252-280.

Paper ID: 140921-V1

Surface-driven execution:
  - G1: cOut ~ syntheticT (probit marginal effects baseline; OLS design alternative)
  - G2: fmissmatch ~ syntheticT (OLS baseline)
  - Cross-sectional OLS with cluster(byear) SE
  - Strict adherence to surface core_universe spec_ids

Outputs:
  - specification_results.csv (baseline, design/*, rc/* rows)
  - inference_results.csv (infer/* rows)
  - SPECIFICATION_SEARCH.md
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import statsmodels.formula.api as smf
import json
import sys
import warnings
warnings.filterwarnings('ignore')

# Add scripts dir for utilities
sys.path.insert(0, "scripts")
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash, software_block
)

PAPER_ID = "140921-V1"
DATA_DIR = "data/downloads/extracted/140921-V1"
OUTPUT_DIR = DATA_DIR

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

# Design audit blocks from surface (to be copied verbatim into rc/* rows)
G1_DESIGN_AUDIT = surface_obj["baseline_groups"][0]["design_audit"]
G2_DESIGN_AUDIT = surface_obj["baseline_groups"][1]["design_audit"]
G1_INFERENCE_CANONICAL = surface_obj["baseline_groups"][0]["inference_plan"]["canonical"]
G2_INFERENCE_CANONICAL = surface_obj["baseline_groups"][1]["inference_plan"]["canonical"]

# ============================================================
# LOAD DATA
# ============================================================
df_raw = pd.read_stata(f"{DATA_DIR}/data/final-data.dta")
for col in df_raw.columns:
    if df_raw[col].dtype == np.float32:
        df_raw[col] = df_raw[col].astype(np.float64)

# G1 sample: base_sample==1
df_g1 = df_raw[df_raw['base_sample'] == 1].copy()
# G2 sample: base_sample==1 & fmissmatch not null
df_g2 = df_g1[df_g1['fmissmatch'].notna()].copy()

results = []
inference_results = []
spec_run_counter = 0


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
            design={"cross_sectional_ols": design_audit},
            axis_block_name=axis_block_name,
            axis_block=axis_block,
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
            "outcome_var": outcome_var, "treatment_var": treatment_var,
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
# HELPER: Run Probit ME via statsmodels
# ============================================================
def run_probit_me(spec_id, spec_tree_path, baseline_group_id,
                  outcome_var, treatment_var, controls,
                  data, sample_desc, controls_desc, cluster_var,
                  design_audit, inference_canonical,
                  axis_block_name=None, axis_block=None, notes=""):
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        controls_str = " + ".join(controls) if controls else ""
        formula = f"{outcome_var} ~ {treatment_var} + {controls_str}" if controls_str else f"{outcome_var} ~ {treatment_var}"

        reg_vars = [outcome_var, treatment_var] + controls + [cluster_var]
        df_reg = data.dropna(subset=[v for v in reg_vars if v in data.columns]).copy()

        probit_model = smf.probit(formula, data=df_reg).fit(
            cov_type='cluster', cov_kwds={'groups': df_reg[cluster_var]}, disp=0)
        mfx = probit_model.get_margeff(at='overall')

        var_names = list(mfx.summary_frame().index)
        idx = var_names.index(treatment_var) if treatment_var in var_names else 0

        coef_val = float(mfx.margeff[idx])
        se_val = float(mfx.margeff_se[idx])
        pval = float(mfx.pvalues[idx])
        ci_lower = float(mfx.conf_int()[idx, 0])
        ci_upper = float(mfx.conf_int()[idx, 1])
        nobs = int(probit_model.nobs)
        r2 = float(probit_model.prsquared)

        all_coefs = {var_names[i]: float(mfx.margeff[i]) for i in range(len(var_names))}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": inference_canonical["spec_id"],
                       "params": inference_canonical["params"]},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"cross_sectional_ols": design_audit},
            axis_block_name=axis_block_name,
            axis_block=axis_block,
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
            "sample_desc": sample_desc, "fixed_effects": "none",
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
            "outcome_var": outcome_var, "treatment_var": treatment_var,
            "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "n_obs": np.nan, "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc, "fixed_effects": "none",
            "controls_desc": controls_desc, "cluster_var": cluster_var,
            "run_success": 0, "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


# ============================================================
# HELPER: Run inference variant
# ============================================================
def run_inference_variant(base_run_id, spec_id, spec_tree_path,
                          baseline_group_id, outcome_var, treatment_var,
                          controls, fe_formula, data, vcov,
                          cluster_var, design_audit):
    infer_counter = len(inference_results) + 1
    infer_run_id = f"{PAPER_ID}_infer_{infer_counter:03d}"

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
            inference={"spec_id": spec_id, "params": {"vcov": str(vcov), "cluster_var": cluster_var}},
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
            "outcome_var": outcome_var,
            "treatment_var": treatment_var,
            "coefficient": coef_val, "std_error": se_val, "p_value": pval,
            "ci_lower": ci_lower, "ci_upper": ci_upper,
            "n_obs": nobs, "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "cluster_var": cluster_var,
            "run_success": 1, "run_error": ""
        })
    except Exception as e:
        err_msg = str(e)[:240]
        payload = make_failure_payload(error=err_msg,
            error_details=error_details_from_exception(e, stage="inference"),
            software=SW_BLOCK, surface_hash=SURFACE_HASH)
        inference_results.append({
            "paper_id": PAPER_ID,
            "inference_run_id": infer_run_id,
            "spec_run_id": base_run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": treatment_var,
            "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "n_obs": np.nan, "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "cluster_var": cluster_var,
            "run_success": 0, "run_error": err_msg
        })


# ============================================================
# Prepare derived datasets
# ============================================================

# Treatment: binary top quintile
pct80 = df_g1['syntheticT'].quantile(0.80)
df_g1['treatment_binary'] = (df_g1['syntheticT'] > pct80).astype(float)

# Outcome: log(fmissmatch+1)
df_g2['log_fmissmatch'] = np.log1p(df_g2['fmissmatch'])

# Sample restrictions for G1
q05_t = df_g1['syntheticT'].quantile(0.05)
q95_t = df_g1['syntheticT'].quantile(0.95)
df_g1_trim_t = df_g1[(df_g1['syntheticT'] >= q05_t) & (df_g1['syntheticT'] <= q95_t)].copy()
q10_t = df_g1['syntheticT'].quantile(0.10)
q90_t = df_g1['syntheticT'].quantile(0.90)
df_g1_trim_t_1090 = df_g1[(df_g1['syntheticT'] >= q10_t) & (df_g1['syntheticT'] <= q90_t)].copy()
df_g1_no_mourn = df_g1[df_g1['mourn'] == 0].copy()
df_g1_age1830 = df_g1[(df_g1['age1861'] >= 18) & (df_g1['age1861'] <= 30)].copy()
df_g1_age1833 = df_g1[(df_g1['age1861'] >= 18) & (df_g1['age1861'] <= 33)].copy()
df_g1_age1632 = df_g1[(df_g1['age1861'] >= 16) & (df_g1['age1861'] <= 32)].copy()

# Additional treatment variables for G1
pct50 = df_g1['syntheticT'].quantile(0.50)
df_g1['treatment_binary_median'] = (df_g1['syntheticT'] > pct50).astype(float)
df_g1['syntheticT_sq'] = df_g1['syntheticT'] ** 2

# Sample restrictions for G2
q01_m = df_g2['fmissmatch'].quantile(0.01)
q99_m = df_g2['fmissmatch'].quantile(0.99)
df_g2_trim1 = df_g2[(df_g2['fmissmatch'] >= q01_m) & (df_g2['fmissmatch'] <= q99_m)].copy()
q05_m = df_g2['fmissmatch'].quantile(0.05)
q95_m = df_g2['fmissmatch'].quantile(0.95)
df_g2_trim5 = df_g2[(df_g2['fmissmatch'] >= q05_m) & (df_g2['fmissmatch'] <= q95_m)].copy()
df_g2_age1830 = df_g2[(df_g2['age1861'] >= 18) & (df_g2['age1861'] <= 30)].copy()
df_g2_age1833 = df_g2[(df_g2['age1861'] >= 18) & (df_g2['age1861'] <= 33)].copy()
df_g2_age1632 = df_g2[(df_g2['age1861'] >= 16) & (df_g2['age1861'] <= 32)].copy()
df_g2_no_mourn = df_g2[df_g2['mourn'] == 0].copy()
df_g2['asinh_fmissmatch'] = np.arcsinh(df_g2['fmissmatch'])
df_g2['syntheticT_sq'] = df_g2['syntheticT'] ** 2
df_g2['treatment_binary'] = (df_g2['syntheticT'] > pct80).astype(float)
q05_t_g2 = df_g2['syntheticT'].quantile(0.05)
q95_t_g2 = df_g2['syntheticT'].quantile(0.95)
df_g2_trim_t = df_g2[(df_g2['syntheticT'] >= q05_t_g2) & (df_g2['syntheticT'] <= q95_t_g2)].copy()


# Common control sets
g1_controls_a = ["pr4", "biorder", "hengpee"]
g1_controls_b = ["pr4", "biorder", "hengpee", "distlondon"]
g2_controls_a = ["pr4", "biorder", "hengpee"]
g2_controls_b = ["pr4", "biorder", "hengpee", "distlondon"]
g1_vcov = {"CRV1": "byear"}
g2_vcov = {"CRV1": "byear"}


# ############################################################
# G1: Effect on probability of marrying a commoner
# (Probit marginal effects is baseline estimator)
# ############################################################

print("=== G1 BASELINE SPECS ===")

# Baseline: Panel A (probit ME)
run_id_g1_bl_a, *_ = run_probit_me(
    "baseline", "designs/cross_sectional_ols.md#baseline", "G1",
    "cOut", "syntheticT", g1_controls_a,
    df_g1, "base_sample==1 (N=644)", "pr4, biorder, hengpee", "byear",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    notes="Probit marginal effects, Panel A, Table 2 Col 1"
)

# Additional baseline: Panel B (probit ME) - in surface core_universe
run_id_g1_bl_b, *_ = run_probit_me(
    "baseline__table2_panelb_col1", "designs/cross_sectional_ols.md#baseline", "G1",
    "cOut", "syntheticT", g1_controls_b,
    df_g1, "base_sample==1, distlondon non-missing (N~484)", "pr4, biorder, hengpee, distlondon", "byear",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    notes="Probit marginal effects, Panel B, Table 2 Col 1"
)

# Design alternative: OLS/LPM - in surface core_universe
# For design/* rows, override the design-defining parameter changed
print("=== G1 DESIGN ALTERNATIVES ===")
g1_design_ols = {**G1_DESIGN_AUDIT, "estimator": "ols"}

# OLS Panel B controls (primary design alternative)
run_id_g1_ols, *_ = run_ols(
    "design/cross_sectional_ols/estimator/ols",
    "designs/cross_sectional_ols.md#estimator-alternatives", "G1",
    "cOut", "syntheticT", g1_controls_b, "",
    df_g1, g1_vcov,
    "base_sample==1 (N~484)", "pr4, biorder, hengpee, distlondon", "byear",
    g1_design_ols, G1_INFERENCE_CANONICAL,
    notes="OLS/LPM analog of probit baseline (Panel B controls)"
)

# OLS Panel A controls
run_ols(
    "design/cross_sectional_ols/estimator/ols__panela",
    "designs/cross_sectional_ols.md#estimator-alternatives", "G1",
    "cOut", "syntheticT", g1_controls_a, "",
    df_g1, g1_vcov,
    "base_sample==1 (N=644)", "pr4, biorder, hengpee", "byear",
    g1_design_ols, G1_INFERENCE_CANONICAL,
    notes="OLS/LPM with Panel A controls"
)

# ---- G1 RC specs (all use probit since that's the baseline estimator) ----
# For rc/* rows, design_audit = G1_DESIGN_AUDIT verbatim
print("=== G1 RC SPECS ===")

# rc/controls/loo/drop_pr4
run_probit_me(
    "rc/controls/loo/drop_pr4",
    "modules/robustness/controls.md#leave-one-out-controls-loo", "G1",
    "cOut", "syntheticT", [c for c in g1_controls_b if c != "pr4"],
    df_g1, "base_sample==1", "Panel B minus pr4", "byear",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_pr4", "family": "loo",
                "dropped": ["pr4"], "added": [], "n_controls": 3}
)

# rc/controls/loo/drop_biorder
run_probit_me(
    "rc/controls/loo/drop_biorder",
    "modules/robustness/controls.md#leave-one-out-controls-loo", "G1",
    "cOut", "syntheticT", [c for c in g1_controls_b if c != "biorder"],
    df_g1, "base_sample==1", "Panel B minus biorder", "byear",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_biorder", "family": "loo",
                "dropped": ["biorder"], "added": [], "n_controls": 3}
)

# rc/controls/loo/drop_hengpee
run_probit_me(
    "rc/controls/loo/drop_hengpee",
    "modules/robustness/controls.md#leave-one-out-controls-loo", "G1",
    "cOut", "syntheticT", [c for c in g1_controls_b if c != "hengpee"],
    df_g1, "base_sample==1", "Panel B minus hengpee", "byear",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_hengpee", "family": "loo",
                "dropped": ["hengpee"], "added": [], "n_controls": 3}
)

# rc/controls/loo/drop_distlondon
run_probit_me(
    "rc/controls/loo/drop_distlondon",
    "modules/robustness/controls.md#leave-one-out-controls-loo", "G1",
    "cOut", "syntheticT", [c for c in g1_controls_b if c != "distlondon"],
    df_g1, "base_sample==1 (N=644, no distlondon restriction)", "Panel B minus distlondon = Panel A controls", "byear",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_distlondon", "family": "loo",
                "dropped": ["distlondon"], "added": [], "n_controls": 3}
)

# rc/controls/sets/none
run_probit_me(
    "rc/controls/sets/none",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "cOut", "syntheticT", [],
    df_g1, "base_sample==1 (N=644)", "none (bivariate)", "byear",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/none", "family": "sets",
                "dropped": g1_controls_b, "added": [], "n_controls": 0, "set_name": "none"}
)

# rc/controls/sets/minimal_no_distlondon
run_probit_me(
    "rc/controls/sets/minimal_no_distlondon",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "cOut", "syntheticT", g1_controls_a,
    df_g1, "base_sample==1 (N=644)", "pr4, biorder, hengpee (Panel A set)", "byear",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/minimal_no_distlondon", "family": "sets",
                "dropped": ["distlondon"], "added": [], "n_controls": 3,
                "set_name": "minimal_no_distlondon"}
)

# rc/controls/sets/full_with_distlondon
run_probit_me(
    "rc/controls/sets/full_with_distlondon",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "cOut", "syntheticT", g1_controls_b,
    df_g1, "base_sample==1, distlondon non-missing (N~484)", "pr4, biorder, hengpee, distlondon", "byear",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/full_with_distlondon", "family": "sets",
                "dropped": [], "added": ["distlondon"], "n_controls": 4,
                "set_name": "full_with_distlondon"}
)

# rc/sample/outliers/trim_syntheticT_5_95
# Note: probit on trimmed sample
run_probit_me(
    "rc/sample/outliers/trim_syntheticT_5_95",
    "modules/robustness/sample.md#outliers-and-influential-observations", "G1",
    "cOut", "syntheticT", g1_controls_b,
    df_g1_trim_t,
    f"syntheticT in [{q05_t:.2f}, {q95_t:.2f}] (N~{len(df_g1_trim_t)})",
    "pr4, biorder, hengpee, distlondon", "byear",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_syntheticT_5_95", "axis": "outliers",
                "rule": "trim", "params": {"var": "syntheticT", "lower_q": 0.05, "upper_q": 0.95},
                "n_obs_before": len(df_g1), "n_obs_after": len(df_g1_trim_t)}
)

# rc/sample/restriction/drop_mourn_cohort
run_probit_me(
    "rc/sample/restriction/drop_mourn_cohort",
    "modules/robustness/sample.md#subsample-restrictions", "G1",
    "cOut", "syntheticT", g1_controls_b,
    df_g1_no_mourn,
    f"mourn==0 (N~{len(df_g1_no_mourn)})",
    "pr4, biorder, hengpee, distlondon", "byear",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/drop_mourn_cohort", "axis": "subsample",
                "restriction": "mourn==0", "n_obs_before": len(df_g1), "n_obs_after": len(df_g1_no_mourn)}
)

# rc/sample/restriction/age_18_30
run_probit_me(
    "rc/sample/restriction/age_18_30",
    "modules/robustness/sample.md#subsample-restrictions", "G1",
    "cOut", "syntheticT", g1_controls_b,
    df_g1_age1830,
    f"age1861 in [18,30] (N~{len(df_g1_age1830)})",
    "pr4, biorder, hengpee, distlondon", "byear",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/age_18_30", "axis": "subsample",
                "restriction": "age1861 in [18,30]", "n_obs_before": len(df_g1), "n_obs_after": len(df_g1_age1830)}
)

# rc/sample/restriction/age_18_33
run_probit_me(
    "rc/sample/restriction/age_18_33",
    "modules/robustness/sample.md#subsample-restrictions", "G1",
    "cOut", "syntheticT", g1_controls_b,
    df_g1_age1833,
    f"age1861 in [18,33] (N~{len(df_g1_age1833)})",
    "pr4, biorder, hengpee, distlondon", "byear",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/age_18_33", "axis": "subsample",
                "restriction": "age1861 in [18,33]", "n_obs_before": len(df_g1), "n_obs_after": len(df_g1_age1833)}
)

# rc/form/treatment/treatment_binary_top_quintile
# Probit with binary treatment
run_probit_me(
    "rc/form/treatment/treatment_binary_top_quintile",
    "modules/robustness/functional_form.md#treatment-transformations", "G1",
    "cOut", "treatment_binary", g1_controls_b,
    df_g1, "base_sample==1", "pr4, biorder, hengpee, distlondon", "byear",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/treatment/treatment_binary_top_quintile",
                "treatment_transform": "binarize_top_quintile",
                "outcome_transform": "level",
                "threshold": float(pct80), "direction": "above",
                "units": "syntheticT percentile",
                "interpretation": f"Probit ME with binary treatment: syntheticT > {pct80:.2f} (80th pctile)."}
)

# rc/fe/add/byear
# NOTE: Probit with many FE is problematic (incidental parameters).
# Use OLS/LPM for this spec (standard practice for FE with binary outcomes).
# The design_audit must still be G1_DESIGN_AUDIT verbatim per contract.
run_ols(
    "rc/fe/add/byear",
    "modules/robustness/fixed_effects.md#additive-fe-variations-relative-to-baseline", "G1",
    "cOut", "syntheticT", g1_controls_b, "byear",
    df_g1, "hetero",
    "base_sample==1", "pr4, biorder, hengpee, distlondon", "",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/add/byear", "family": "add",
                "added": ["byear"], "dropped": [],
                "baseline_fe": [], "new_fe": ["byear"]},
    notes="OLS/LPM used for FE spec (probit with many FE has incidental parameters problem). Birth-year FE absorb most syntheticT variation."
)

# rc/sample/outliers/trim_syntheticT_10_90
run_probit_me(
    "rc/sample/outliers/trim_syntheticT_10_90",
    "modules/robustness/sample.md#outliers-and-influential-observations", "G1",
    "cOut", "syntheticT", g1_controls_b,
    df_g1_trim_t_1090,
    f"syntheticT in [{q10_t:.2f}, {q90_t:.2f}] (N~{len(df_g1_trim_t_1090)})",
    "pr4, biorder, hengpee, distlondon", "byear",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_syntheticT_10_90", "axis": "outliers",
                "rule": "trim", "params": {"var": "syntheticT", "lower_q": 0.10, "upper_q": 0.90},
                "n_obs_before": len(df_g1), "n_obs_after": len(df_g1_trim_t_1090)}
)

# rc/sample/restriction/age_16_32
run_probit_me(
    "rc/sample/restriction/age_16_32",
    "modules/robustness/sample.md#subsample-restrictions", "G1",
    "cOut", "syntheticT", g1_controls_b,
    df_g1_age1632,
    f"age1861 in [16,32] (N~{len(df_g1_age1632)})",
    "pr4, biorder, hengpee, distlondon", "byear",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/age_16_32", "axis": "subsample",
                "restriction": "age1861 in [16,32]", "n_obs_before": len(df_g1), "n_obs_after": len(df_g1_age1632)}
)

# rc/form/treatment/treatment_binary_median
run_probit_me(
    "rc/form/treatment/treatment_binary_median",
    "modules/robustness/functional_form.md#treatment-transformations", "G1",
    "cOut", "treatment_binary_median", g1_controls_b,
    df_g1, "base_sample==1", "pr4, biorder, hengpee, distlondon", "byear",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/treatment/treatment_binary_median",
                "treatment_transform": "binarize_median",
                "outcome_transform": "level",
                "threshold": float(pct50), "direction": "above",
                "units": "syntheticT median",
                "interpretation": f"Probit ME with binary treatment: syntheticT > {pct50:.2f} (median split)."}
)

# rc/form/treatment/quadratic_syntheticT
# Use OLS since probit with quadratic + controls may not converge
run_ols(
    "rc/form/treatment/quadratic_syntheticT",
    "modules/robustness/functional_form.md#treatment-transformations", "G1",
    "cOut", "syntheticT", g1_controls_b + ["syntheticT_sq"], "",
    df_g1, g1_vcov,
    "base_sample==1 (OLS with quadratic)", "pr4, biorder, hengpee, distlondon, syntheticT^2", "byear",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/treatment/quadratic_syntheticT",
                "treatment_transform": "quadratic",
                "outcome_transform": "level",
                "interpretation": "OLS/LPM with quadratic term. Tests for nonlinear dose-response."}
)

# rc/form/outcome/mheir (alternative binary outcome from same Table 2)
run_probit_me(
    "rc/form/outcome/mheir",
    "modules/robustness/functional_form.md#outcome-transformations", "G1",
    "mheir", "syntheticT", g1_controls_b,
    df_g1, "base_sample==1", "pr4, biorder, hengpee, distlondon", "byear",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/mheir",
                "outcome_transform": "alternative_binary",
                "treatment_transform": "level",
                "interpretation": "Married an heir (binary). Alternative marriage outcome from same Table 2."}
)

# rc/form/outcome/fdown (married down in landholding rank)
run_probit_me(
    "rc/form/outcome/fdown",
    "modules/robustness/functional_form.md#outcome-transformations", "G1",
    "fdown", "syntheticT", g1_controls_b,
    df_g1, "base_sample==1", "pr4, biorder, hengpee, distlondon", "byear",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/fdown",
                "outcome_transform": "alternative_binary",
                "treatment_transform": "level",
                "interpretation": "Married down in landholding (binary). Alternative outcome from Table 2."}
)

# rc/sample/outliers/trim_syntheticT_1_99
q01_t = df_g1['syntheticT'].quantile(0.01)
q99_t = df_g1['syntheticT'].quantile(0.99)
df_g1_trim_t_0199 = df_g1[(df_g1['syntheticT'] >= q01_t) & (df_g1['syntheticT'] <= q99_t)].copy()
run_probit_me(
    "rc/sample/outliers/trim_syntheticT_1_99",
    "modules/robustness/sample.md#outliers-and-influential-observations", "G1",
    "cOut", "syntheticT", g1_controls_b,
    df_g1_trim_t_0199,
    f"syntheticT in [{q01_t:.2f}, {q99_t:.2f}] (N~{len(df_g1_trim_t_0199)})",
    "pr4, biorder, hengpee, distlondon", "byear",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_syntheticT_1_99", "axis": "outliers",
                "rule": "trim", "params": {"var": "syntheticT", "lower_q": 0.01, "upper_q": 0.99},
                "n_obs_before": len(df_g1), "n_obs_after": len(df_g1_trim_t_0199)}
)

# rc/sample/restriction/age_20_35 (G1)
df_g1_age2035 = df_g1[(df_g1['age1861'] >= 20) & (df_g1['age1861'] <= 35)].copy()
run_probit_me(
    "rc/sample/restriction/age_20_35",
    "modules/robustness/sample.md#subsample-restrictions", "G1",
    "cOut", "syntheticT", g1_controls_b,
    df_g1_age2035,
    f"age1861 in [20,35] (N~{len(df_g1_age2035)})",
    "pr4, biorder, hengpee, distlondon", "byear",
    G1_DESIGN_AUDIT, G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/age_20_35", "axis": "subsample",
                "restriction": "age1861 in [20,35]", "n_obs_before": len(df_g1), "n_obs_after": len(df_g1_age2035)}
)


# ############################################################
# G2: Effect on wealth sorting (mismatch) -- OLS baseline
# ############################################################

print("=== G2 BASELINE SPECS ===")

# Baseline: Panel A
run_id_g2_bl_a, *_ = run_ols(
    "baseline", "designs/cross_sectional_ols.md#baseline", "G2",
    "fmissmatch", "syntheticT", g2_controls_a, "",
    df_g2, g2_vcov,
    "base_sample==1, fmissmatch non-missing (N=324)", "pr4, biorder, hengpee", "byear",
    G2_DESIGN_AUDIT, G2_INFERENCE_CANONICAL,
    notes="OLS, Table 2, Panel A, Col 3"
)

# Additional baseline: Panel B - in surface core_universe
run_id_g2_bl_b, *_ = run_ols(
    "baseline__table2_panelb_col3", "designs/cross_sectional_ols.md#baseline", "G2",
    "fmissmatch", "syntheticT", g2_controls_b, "",
    df_g2, g2_vcov,
    "base_sample==1, fmissmatch+distlondon non-missing (N~260)", "pr4, biorder, hengpee, distlondon", "byear",
    G2_DESIGN_AUDIT, G2_INFERENCE_CANONICAL,
    notes="OLS, Table 2, Panel B, Col 3"
)

# ---- G2 RC specs (OLS, design_audit = G2_DESIGN_AUDIT verbatim) ----
print("=== G2 RC SPECS ===")

for var in g2_controls_b:
    ctrl = [c for c in g2_controls_b if c != var]
    run_ols(
        f"rc/controls/loo/drop_{var}",
        "modules/robustness/controls.md#leave-one-out-controls-loo", "G2",
        "fmissmatch", "syntheticT", ctrl, "",
        df_g2, g2_vcov,
        "G2 sample", f"Panel B minus {var}", "byear",
        G2_DESIGN_AUDIT, G2_INFERENCE_CANONICAL,
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/controls/loo/drop_{var}", "family": "loo",
                    "dropped": [var], "added": [], "n_controls": len(ctrl)}
    )

# rc/controls/sets/none
run_ols(
    "rc/controls/sets/none",
    "modules/robustness/controls.md#standard-control-sets", "G2",
    "fmissmatch", "syntheticT", [], "",
    df_g2, g2_vcov,
    "G2 sample (N=324)", "none (bivariate)", "byear",
    G2_DESIGN_AUDIT, G2_INFERENCE_CANONICAL,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/none", "family": "sets",
                "dropped": g2_controls_b, "added": [], "n_controls": 0, "set_name": "none"}
)

# rc/controls/sets/full_with_distlondon
run_ols(
    "rc/controls/sets/full_with_distlondon",
    "modules/robustness/controls.md#standard-control-sets", "G2",
    "fmissmatch", "syntheticT", g2_controls_b, "",
    df_g2, g2_vcov,
    "G2 sample with distlondon (N~260)", "pr4, biorder, hengpee, distlondon", "byear",
    G2_DESIGN_AUDIT, G2_INFERENCE_CANONICAL,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/full_with_distlondon", "family": "sets",
                "dropped": [], "added": ["distlondon"], "n_controls": 4,
                "set_name": "full_with_distlondon"}
)

# rc/sample/outliers/trim_fmissmatch_1_99
run_ols(
    "rc/sample/outliers/trim_fmissmatch_1_99",
    "modules/robustness/sample.md#outliers-and-influential-observations", "G2",
    "fmissmatch", "syntheticT", g2_controls_b, "",
    df_g2_trim1, g2_vcov,
    f"fmissmatch in [{q01_m:.0f}, {q99_m:.0f}] (N={len(df_g2_trim1)})",
    "pr4, biorder, hengpee, distlondon", "byear",
    G2_DESIGN_AUDIT, G2_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_fmissmatch_1_99", "axis": "outliers",
                "rule": "trim", "params": {"var": "fmissmatch", "lower_q": 0.01, "upper_q": 0.99},
                "n_obs_before": len(df_g2), "n_obs_after": len(df_g2_trim1)}
)

# rc/sample/outliers/trim_fmissmatch_5_95
run_ols(
    "rc/sample/outliers/trim_fmissmatch_5_95",
    "modules/robustness/sample.md#outliers-and-influential-observations", "G2",
    "fmissmatch", "syntheticT", g2_controls_b, "",
    df_g2_trim5, g2_vcov,
    f"fmissmatch in [{q05_m:.0f}, {q95_m:.0f}] (N={len(df_g2_trim5)})",
    "pr4, biorder, hengpee, distlondon", "byear",
    G2_DESIGN_AUDIT, G2_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_fmissmatch_5_95", "axis": "outliers",
                "rule": "trim", "params": {"var": "fmissmatch", "lower_q": 0.05, "upper_q": 0.95},
                "n_obs_before": len(df_g2), "n_obs_after": len(df_g2_trim5)}
)

# rc/sample/restriction/age_18_30
run_ols(
    "rc/sample/restriction/age_18_30",
    "modules/robustness/sample.md#subsample-restrictions", "G2",
    "fmissmatch", "syntheticT", g2_controls_b, "",
    df_g2_age1830, g2_vcov,
    f"G2 sample, age 18-30 (N={len(df_g2_age1830)})",
    "pr4, biorder, hengpee, distlondon", "byear",
    G2_DESIGN_AUDIT, G2_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/age_18_30", "axis": "subsample",
                "restriction": "age1861 in [18,30]", "n_obs_before": len(df_g2), "n_obs_after": len(df_g2_age1830)}
)

# rc/form/outcome/fmissmatch2_signed
run_ols(
    "rc/form/outcome/fmissmatch2_signed",
    "modules/robustness/functional_form.md#outcome-transformations", "G2",
    "fmissmatch2", "syntheticT", g2_controls_b, "",
    df_g2, g2_vcov,
    "G2 sample", "pr4, biorder, hengpee, distlondon", "byear",
    G2_DESIGN_AUDIT, G2_INFERENCE_CANONICAL,
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/fmissmatch2_signed",
                "outcome_transform": "signed_mismatch",
                "treatment_transform": "level",
                "interpretation": "Signed mismatch (husband - wife landholding percentile rank). Positive = husband wealthier."}
)

# rc/form/outcome/log_fmissmatch
run_ols(
    "rc/form/outcome/log_fmissmatch",
    "modules/robustness/functional_form.md#outcome-transformations", "G2",
    "log_fmissmatch", "syntheticT", g2_controls_b, "",
    df_g2, g2_vcov,
    "G2 sample", "pr4, biorder, hengpee, distlondon", "byear",
    G2_DESIGN_AUDIT, G2_INFERENCE_CANONICAL,
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/log_fmissmatch",
                "outcome_transform": "log1p",
                "treatment_transform": "level",
                "interpretation": "Log(1 + absolute mismatch). Reduces influence of extreme values."}
)

# rc/fe/add/byear
run_ols(
    "rc/fe/add/byear",
    "modules/robustness/fixed_effects.md#additive-fe-variations-relative-to-baseline", "G2",
    "fmissmatch", "syntheticT", g2_controls_b, "byear",
    df_g2, "hetero",
    "G2 sample", "pr4, biorder, hengpee, distlondon", "",
    G2_DESIGN_AUDIT, G2_INFERENCE_CANONICAL,
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/add/byear", "family": "add",
                "added": ["byear"], "dropped": [],
                "baseline_fe": [], "new_fe": ["byear"]},
    notes="Birth-year FE absorb most syntheticT variation."
)

# rc/controls/sets/minimal_no_distlondon (G2)
run_ols(
    "rc/controls/sets/minimal_no_distlondon",
    "modules/robustness/controls.md#standard-control-sets", "G2",
    "fmissmatch", "syntheticT", g2_controls_a, "",
    df_g2, g2_vcov,
    "G2 sample (N=324)", "pr4, biorder, hengpee (Panel A controls)", "byear",
    G2_DESIGN_AUDIT, G2_INFERENCE_CANONICAL,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/minimal_no_distlondon", "family": "sets",
                "dropped": ["distlondon"], "added": [], "n_controls": 3,
                "set_name": "minimal_no_distlondon"}
)

# rc/sample/outliers/trim_syntheticT_5_95 (G2)
run_ols(
    "rc/sample/outliers/trim_syntheticT_5_95",
    "modules/robustness/sample.md#outliers-and-influential-observations", "G2",
    "fmissmatch", "syntheticT", g2_controls_b, "",
    df_g2_trim_t, g2_vcov,
    f"G2 sample, syntheticT trimmed (N={len(df_g2_trim_t)})",
    "pr4, biorder, hengpee, distlondon", "byear",
    G2_DESIGN_AUDIT, G2_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_syntheticT_5_95", "axis": "outliers",
                "rule": "trim", "params": {"var": "syntheticT", "lower_q": 0.05, "upper_q": 0.95},
                "n_obs_before": len(df_g2), "n_obs_after": len(df_g2_trim_t)}
)

# rc/sample/restriction/drop_mourn_cohort (G2)
run_ols(
    "rc/sample/restriction/drop_mourn_cohort",
    "modules/robustness/sample.md#subsample-restrictions", "G2",
    "fmissmatch", "syntheticT", g2_controls_b, "",
    df_g2_no_mourn, g2_vcov,
    f"G2 sample, mourn==0 (N={len(df_g2_no_mourn)})",
    "pr4, biorder, hengpee, distlondon", "byear",
    G2_DESIGN_AUDIT, G2_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/drop_mourn_cohort", "axis": "subsample",
                "restriction": "mourn==0", "n_obs_before": len(df_g2), "n_obs_after": len(df_g2_no_mourn)}
)

# rc/sample/restriction/age_18_33 (G2)
run_ols(
    "rc/sample/restriction/age_18_33",
    "modules/robustness/sample.md#subsample-restrictions", "G2",
    "fmissmatch", "syntheticT", g2_controls_b, "",
    df_g2_age1833, g2_vcov,
    f"G2 sample, age 18-33 (N={len(df_g2_age1833)})",
    "pr4, biorder, hengpee, distlondon", "byear",
    G2_DESIGN_AUDIT, G2_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/age_18_33", "axis": "subsample",
                "restriction": "age1861 in [18,33]", "n_obs_before": len(df_g2), "n_obs_after": len(df_g2_age1833)}
)

# rc/sample/restriction/age_16_32 (G2)
run_ols(
    "rc/sample/restriction/age_16_32",
    "modules/robustness/sample.md#subsample-restrictions", "G2",
    "fmissmatch", "syntheticT", g2_controls_b, "",
    df_g2_age1632, g2_vcov,
    f"G2 sample, age 16-32 (N={len(df_g2_age1632)})",
    "pr4, biorder, hengpee, distlondon", "byear",
    G2_DESIGN_AUDIT, G2_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/age_16_32", "axis": "subsample",
                "restriction": "age1861 in [16,32]", "n_obs_before": len(df_g2), "n_obs_after": len(df_g2_age1632)}
)

# rc/form/outcome/asinh_fmissmatch (G2)
run_ols(
    "rc/form/outcome/asinh_fmissmatch",
    "modules/robustness/functional_form.md#outcome-transformations", "G2",
    "asinh_fmissmatch", "syntheticT", g2_controls_b, "",
    df_g2, g2_vcov,
    "G2 sample", "pr4, biorder, hengpee, distlondon", "byear",
    G2_DESIGN_AUDIT, G2_INFERENCE_CANONICAL,
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/asinh_fmissmatch",
                "outcome_transform": "asinh",
                "treatment_transform": "level",
                "interpretation": "Inverse hyperbolic sine of absolute mismatch. Similar to log for large values, handles zeros."}
)

# rc/form/treatment/treatment_binary_top_quintile (G2)
run_ols(
    "rc/form/treatment/treatment_binary_top_quintile",
    "modules/robustness/functional_form.md#treatment-transformations", "G2",
    "fmissmatch", "treatment_binary", g2_controls_b, "",
    df_g2, g2_vcov,
    "G2 sample", "pr4, biorder, hengpee, distlondon", "byear",
    G2_DESIGN_AUDIT, G2_INFERENCE_CANONICAL,
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/treatment/treatment_binary_top_quintile",
                "treatment_transform": "binarize_top_quintile",
                "outcome_transform": "level",
                "threshold": float(pct80), "direction": "above",
                "units": "syntheticT percentile",
                "interpretation": f"Binary treatment: syntheticT > {pct80:.2f} (80th pctile) for mismatch outcome."}
)

# rc/form/treatment/quadratic_syntheticT (G2)
run_ols(
    "rc/form/treatment/quadratic_syntheticT",
    "modules/robustness/functional_form.md#treatment-transformations", "G2",
    "fmissmatch", "syntheticT", g2_controls_b + ["syntheticT_sq"], "",
    df_g2, g2_vcov,
    "G2 sample", "pr4, biorder, hengpee, distlondon, syntheticT^2", "byear",
    G2_DESIGN_AUDIT, G2_INFERENCE_CANONICAL,
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/treatment/quadratic_syntheticT",
                "treatment_transform": "quadratic",
                "outcome_transform": "level",
                "interpretation": "OLS with quadratic syntheticT for mismatch."}
)

# rc/form/outcome/fmissmatch_sq (G2: squared mismatch)
df_g2['fmissmatch_sq'] = df_g2['fmissmatch'] ** 2
run_ols(
    "rc/form/outcome/fmissmatch_sq",
    "modules/robustness/functional_form.md#outcome-transformations", "G2",
    "fmissmatch_sq", "syntheticT", g2_controls_b, "",
    df_g2, g2_vcov,
    "G2 sample", "pr4, biorder, hengpee, distlondon", "byear",
    G2_DESIGN_AUDIT, G2_INFERENCE_CANONICAL,
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/fmissmatch_sq",
                "outcome_transform": "squared",
                "treatment_transform": "level",
                "interpretation": "Squared mismatch. Emphasizes extreme mismatches, sensitive to tail behavior."}
)

# rc/sample/outliers/trim_syntheticT_10_90 (G2)
q10_t_g2 = df_g2['syntheticT'].quantile(0.10)
q90_t_g2 = df_g2['syntheticT'].quantile(0.90)
df_g2_trim_t_1090 = df_g2[(df_g2['syntheticT'] >= q10_t_g2) & (df_g2['syntheticT'] <= q90_t_g2)].copy()
run_ols(
    "rc/sample/outliers/trim_syntheticT_10_90",
    "modules/robustness/sample.md#outliers-and-influential-observations", "G2",
    "fmissmatch", "syntheticT", g2_controls_b, "",
    df_g2_trim_t_1090, g2_vcov,
    f"G2 sample, syntheticT 10-90th (N={len(df_g2_trim_t_1090)})",
    "pr4, biorder, hengpee, distlondon", "byear",
    G2_DESIGN_AUDIT, G2_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_syntheticT_10_90", "axis": "outliers",
                "rule": "trim", "params": {"var": "syntheticT", "lower_q": 0.10, "upper_q": 0.90},
                "n_obs_before": len(df_g2), "n_obs_after": len(df_g2_trim_t_1090)}
)

# rc/sample/restriction/age_20_35 (G2)
df_g2_age2035 = df_g2[(df_g2['age1861'] >= 20) & (df_g2['age1861'] <= 35)].copy()
run_ols(
    "rc/sample/restriction/age_20_35",
    "modules/robustness/sample.md#subsample-restrictions", "G2",
    "fmissmatch", "syntheticT", g2_controls_b, "",
    df_g2_age2035, g2_vcov,
    f"G2 sample, age 20-35 (N={len(df_g2_age2035)})",
    "pr4, biorder, hengpee, distlondon", "byear",
    G2_DESIGN_AUDIT, G2_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/age_20_35", "axis": "subsample",
                "restriction": "age1861 in [20,35]", "n_obs_before": len(df_g2), "n_obs_after": len(df_g2_age2035)}
)


# ############################################################
# INFERENCE VARIANTS
# ############################################################

print("=== INFERENCE VARIANTS ===")

# G1 baseline (Panel A probit -> recompute with OLS HC1)
run_inference_variant(
    run_id_g1_bl_a, "infer/se/hc/hc1",
    "modules/inference/standard_errors.md#heteroskedasticity-robust",
    "G1", "cOut", "syntheticT", g1_controls_a, "", df_g1,
    "hetero", "", G1_DESIGN_AUDIT
)

# G1 baseline Panel B (probit -> OLS HC1)
run_inference_variant(
    run_id_g1_bl_b, "infer/se/hc/hc1",
    "modules/inference/standard_errors.md#heteroskedasticity-robust",
    "G1", "cOut", "syntheticT", g1_controls_b, "", df_g1,
    "hetero", "", G1_DESIGN_AUDIT
)

# G1 OLS design alternative HC1
run_inference_variant(
    run_id_g1_ols, "infer/se/hc/hc1",
    "modules/inference/standard_errors.md#heteroskedasticity-robust",
    "G1", "cOut", "syntheticT", g1_controls_b, "", df_g1,
    "hetero", "", G1_DESIGN_AUDIT
)

# G2 baseline Panel A HC1
run_inference_variant(
    run_id_g2_bl_a, "infer/se/hc/hc1",
    "modules/inference/standard_errors.md#heteroskedasticity-robust",
    "G2", "fmissmatch", "syntheticT", g2_controls_a, "", df_g2,
    "hetero", "", G2_DESIGN_AUDIT
)

# G2 baseline Panel B HC1
run_inference_variant(
    run_id_g2_bl_b, "infer/se/hc/hc1",
    "modules/inference/standard_errors.md#heteroskedasticity-robust",
    "G2", "fmissmatch", "syntheticT", g2_controls_b, "", df_g2,
    "hetero", "", G2_DESIGN_AUDIT
)


# ############################################################
# WRITE OUTPUTS
# ############################################################

print(f"\n=== WRITING OUTPUTS ===")
print(f"Total specification results: {len(results)}")
print(f"Total inference results: {len(inference_results)}")

spec_df = pd.DataFrame(results)
spec_df.to_csv(f"{OUTPUT_DIR}/specification_results.csv", index=False)
print(f"Wrote {OUTPUT_DIR}/specification_results.csv")

if inference_results:
    infer_df = pd.DataFrame(inference_results)
    infer_df.to_csv(f"{OUTPUT_DIR}/inference_results.csv", index=False)
    print(f"Wrote {OUTPUT_DIR}/inference_results.csv")

# Print summary
n_success = spec_df['run_success'].sum()
n_fail = len(spec_df) - n_success
g1_count = (spec_df['baseline_group_id'] == 'G1').sum()
g2_count = (spec_df['baseline_group_id'] == 'G2').sum()
print(f"\nG1 specs: {g1_count}, G2 specs: {g2_count}")
print(f"Successful: {n_success}, Failed: {n_fail}")

for _, row in spec_df[spec_df['spec_id'].str.startswith('baseline')].iterrows():
    print(f"  {row['spec_id']} [{row['baseline_group_id']}]: "
          f"coef={row['coefficient']:.6f}, se={row['std_error']:.6f}, "
          f"p={row['p_value']:.4f}, N={row['n_obs']}")
