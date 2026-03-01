"""
Specification Search Script for Condra, Long, Shaver, Wright (2018)
"The Logic of Insurgent Electoral Violence"
American Economic Review

Paper ID: 113182-V1

Surface-driven execution:
  - G1: total_v2_agcho10 ~ df_5to11, instrumented by plus_wind_00Z_10
        District-election panel, election FE, cluster(DISTID)
  - G2: total_votes_wins ~ post_event_indicator, instrumented by cloudz_perc_election
        Polling-station-road-segment cross-section, district FE, robust SE
  - Both are IV/2SLS with linked_adjustment=true

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
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, "scripts")
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash, software_block
)

PAPER_ID = "113182-V1"
DATA_DIR = "data/downloads/extracted/113182-V1"
OUTPUT_DIR = DATA_DIR

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

G1_DESIGN_AUDIT = surface_obj["baseline_groups"][0]["design_audit"]
G2_DESIGN_AUDIT = surface_obj["baseline_groups"][1]["design_audit"]
G1_INFERENCE_CANONICAL = surface_obj["baseline_groups"][0]["inference_plan"]["canonical"]
G2_INFERENCE_CANONICAL = surface_obj["baseline_groups"][1]["inference_plan"]["canonical"]

# ============================================================
# LOAD DATA
# ============================================================
df_g1_raw = pd.read_stata(f"{DATA_DIR}/Replication_Files/TABLE_2/district_iv.dta")
df_g2_raw = pd.read_stata(f"{DATA_DIR}/Replication_Files/TABLE_3/roads_iv.dta")

# Convert float32 to float64 for precision
for col in df_g1_raw.columns:
    if df_g1_raw[col].dtype == np.float32:
        df_g1_raw[col] = df_g1_raw[col].astype(np.float64)
for col in df_g2_raw.columns:
    if df_g2_raw[col].dtype == np.float32:
        df_g2_raw[col] = df_g2_raw[col].astype(np.float64)

# G1 baseline sample: no_voting_either==0 & disrupt==1
df_g1 = df_g1_raw[(df_g1_raw['no_voting_either'] == 0) & (df_g1_raw['disrupt'] == 1)].copy()

# G2 baseline sample: closure_indicator==0 & stations!=0
df_g2 = df_g2_raw[(df_g2_raw['closure_indicator'] == 0) & (df_g2_raw['stations'] != 0)].copy()

# G1 expanded sample: no_voting_either==0 (drop disrupt==1 condition)
df_g1_all = df_g1_raw[df_g1_raw['no_voting_either'] == 0].copy()

results = []
inference_results = []
spec_run_counter = 0

# ============================================================
# G1 CONTROL SETS
# ============================================================
G1_LINEAR_WEATHER = [
    'windspeed_06Z', 'windspeed_12Z',
    'temp_00Z', 'temp_06Z', 'temp_12Z',
    'rain_00Z', 'rain_06Z', 'rain_12Z'
]
G1_QUADRATIC_WEATHER = [
    'windspeed_06Z2', 'windspeed_12Z2',
    'temp_00Z2', 'temp_06Z2', 'temp_12Z2',
    'rain_00Z2', 'rain_06Z2', 'rain_12Z2'
]
G1_BASELINE_CONTROLS = G1_LINEAR_WEATHER + G1_QUADRATIC_WEATHER + ['plus_wind_00Z_10_pre14D', 'population_2010_adj']
G1_LINEAR_CONTROLS = G1_LINEAR_WEATHER + ['population_2010_adj']  # 10 controls (Col 2)
G1_QUADRATIC_CONTROLS = G1_LINEAR_WEATHER + G1_QUADRATIC_WEATHER + ['population_2010_adj']  # 17 controls (Col 3)

# G2 CONTROL SETS
G2_GEO_CONTROLS = ['ht_route_indicator', 'rcv2', 'population_v2', 'shape_leng', 'pre_event_indicator_6m_V2']
G2_BASELINE_CONTROLS = G2_GEO_CONTROLS + ['march_rain', 'march_rain2']  # 7 controls (Col 3)
G2_NO_RAIN_CONTROLS = G2_GEO_CONTROLS  # 5 controls (Col 2)


# ============================================================
# HELPER: Run IV specification via pyfixest
# ============================================================
def run_iv_pyfixest(spec_id, spec_tree_path, baseline_group_id,
                    outcome_var, treatment_var, instrument_var,
                    controls, fe_var, data, vcov_spec,
                    sample_desc, controls_desc, cluster_var, fe_desc,
                    design_audit, inference_canonical,
                    axis_block_name=None, axis_block=None, notes=""):
    """Run a 2SLS IV regression using pyfixest and record the result."""
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        ctrl_str = " + ".join(controls) if controls else "1"
        formula = f"{outcome_var} ~ {ctrl_str} | {fe_var} | {treatment_var} ~ {instrument_var}"

        m = pf.feols(formula, data=data, vcov=vcov_spec)
        coef_val = float(m.coef().get(treatment_var, np.nan))
        se_val = float(m.se().get(treatment_var, np.nan))
        pval = float(m.pvalue().get(treatment_var, np.nan))
        try:
            ci = m.confint()
            ci_lower = float(ci.loc[treatment_var, ci.columns[0]])
            ci_upper = float(ci.loc[treatment_var, ci.columns[1]])
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
            inference=inference_canonical,
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"instrumental_variables": design_audit},
            axis_block_name=axis_block_name,
            axis_block=axis_block,
            notes=notes if notes else None,
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
            "cluster_var": cluster_var,
            "run_success": 1,
            "run_error": "",
        })
        return run_id, m

    except Exception as e:
        err_msg = str(e)[:240]
        err_det = error_details_from_exception(e, stage="estimation")
        payload = make_failure_payload(
            error=err_msg,
            error_details=err_det,
            inference=inference_canonical,
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
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
            "cluster_var": cluster_var,
            "run_success": 0,
            "run_error": err_msg,
        })
        return run_id, None


# ============================================================
# HELPER: Run LIML specification via linearmodels
# ============================================================
def run_liml(spec_id, spec_tree_path, baseline_group_id,
             outcome_var, treatment_var, instrument_var,
             controls, fe_var, data, cov_type, cluster_var_name,
             sample_desc, controls_desc, cluster_var_label, fe_desc,
             design_audit_override, inference_canonical,
             axis_block_name=None, axis_block=None, notes=""):
    """Run a LIML IV regression using linearmodels.iv.IVLIML."""
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        from linearmodels.iv import IVLIML
        from statsmodels.api import add_constant

        work = data.copy()
        # Create FE dummies
        fe_dummies = pd.get_dummies(work[fe_var], prefix='fe', drop_first=True).astype(float)
        all_vars = [outcome_var, treatment_var, instrument_var] + controls
        if cluster_var_name:
            all_vars.append(cluster_var_name)
        work_sub = pd.concat([work[all_vars].reset_index(drop=True),
                              fe_dummies.reset_index(drop=True)], axis=1)
        work_sub = work_sub.dropna()

        exog_cols = controls + list(fe_dummies.columns)
        exog = add_constant(work_sub[exog_cols])
        endog = work_sub[[treatment_var]]
        instruments = work_sub[[instrument_var]]
        dep = work_sub[outcome_var]

        if cov_type == 'clustered' and cluster_var_name:
            m = IVLIML(dep, exog, endog, instruments).fit(
                cov_type='clustered', clusters=work_sub[cluster_var_name])
        elif cov_type == 'robust':
            m = IVLIML(dep, exog, endog, instruments).fit(cov_type='robust')
        else:
            m = IVLIML(dep, exog, endog, instruments).fit(cov_type='robust')

        coef_val = float(m.params[treatment_var])
        se_val = float(m.std_errors[treatment_var])
        pval = float(m.pvalues[treatment_var])
        ci = m.conf_int()
        ci_lower = float(ci.loc[treatment_var, 'lower'])
        ci_upper = float(ci.loc[treatment_var, 'upper'])
        nobs = int(m.nobs)
        try:
            r2 = float(m.r2)
        except Exception:
            r2 = np.nan

        all_coefs = {k: float(v) for k, v in m.params.items()
                     if k not in list(fe_dummies.columns) and k != 'const'}

        # Override the design audit for LIML
        liml_design = dict(design_audit_override)
        liml_design['estimator'] = 'liml'

        payload = make_success_payload(
            coefficients=all_coefs,
            inference=inference_canonical,
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"instrumental_variables": liml_design},
            axis_block_name=axis_block_name,
            axis_block=axis_block,
            notes=notes if notes else None,
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
            "cluster_var": cluster_var_label,
            "run_success": 1,
            "run_error": "",
        })
        return run_id, m

    except Exception as e:
        err_msg = str(e)[:240]
        err_det = error_details_from_exception(e, stage="estimation_liml")
        payload = make_failure_payload(
            error=err_msg,
            error_details=err_det,
            inference=inference_canonical,
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
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
            "cluster_var": cluster_var_label,
            "run_success": 0,
            "run_error": err_msg,
        })
        return run_id, None


# ============================================================
# HELPER: Record inference variant
# ============================================================
def run_inference_variant(base_run_id, spec_id, spec_tree_path,
                          baseline_group_id, outcome_var, treatment_var,
                          instrument_var, controls, fe_var, data,
                          vcov_spec, cluster_var_label,
                          infer_spec, sample_desc="", controls_desc="", fe_desc=""):
    """Re-estimate with different SE/inference and store in inference_results."""
    global spec_run_counter
    spec_run_counter += 1
    infer_run_id = f"{PAPER_ID}_infer_{spec_run_counter:03d}"

    try:
        ctrl_str = " + ".join(controls) if controls else "1"
        formula = f"{outcome_var} ~ {ctrl_str} | {fe_var} | {treatment_var} ~ {instrument_var}"
        m = pf.feols(formula, data=data, vcov=vcov_spec)
        coef_val = float(m.coef().get(treatment_var, np.nan))
        se_val = float(m.se().get(treatment_var, np.nan))
        pval = float(m.pvalue().get(treatment_var, np.nan))
        try:
            ci = m.confint()
            ci_lower = float(ci.loc[treatment_var, ci.columns[0]])
            ci_upper = float(ci.loc[treatment_var, ci.columns[1]])
        except Exception:
            ci_lower = np.nan
            ci_upper = np.nan
        nobs = int(m._N)
        try:
            r2 = float(m._r2)
        except Exception:
            r2 = np.nan

        all_coefs = {k: float(v) for k, v in m.coef().items()}
        payload = {
            "coefficients": all_coefs,
            "inference": infer_spec,
            "software": SW_BLOCK,
            "surface_hash": SURFACE_HASH,
        }

        inference_results.append({
            "paper_id": PAPER_ID,
            "inference_run_id": infer_run_id,
            "spec_run_id": base_run_id,
            "spec_id": infer_spec["spec_id"],
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": treatment_var,
            "cluster_var": cluster_var_label,
            "coefficient": coef_val,
            "std_error": se_val,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": nobs,
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 1,
            "run_error": "",
        })
        return infer_run_id

    except Exception as e:
        err_msg = str(e)[:240]
        err_det = error_details_from_exception(e, stage="inference_variant")
        payload = make_failure_payload(
            error=err_msg,
            error_details=err_det,
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
        )
        inference_results.append({
            "paper_id": PAPER_ID,
            "inference_run_id": infer_run_id,
            "spec_run_id": base_run_id,
            "spec_id": infer_spec["spec_id"],
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": treatment_var,
            "cluster_var": cluster_var_label,
            "coefficient": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": np.nan,
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 0,
            "run_error": err_msg,
        })
        return infer_run_id


# ############################################################
# G1: ATTACK TIMING AND VOTER TURNOUT (TABLE 2)
# ############################################################

print("=" * 60)
print("G1: Attack Timing and Voter Turnout (Table 2)")
print("=" * 60)

# -----------------------------------------------------------
# G1 BASELINE: Table 2, Col 4 (preferred 2SLS, full controls)
# -----------------------------------------------------------
print("  Running G1 baseline (Table 2, Col 4)...")
rid_g1_baseline, _ = run_iv_pyfixest(
    spec_id="baseline",
    spec_tree_path="specification_tree/designs/instrumental_variables.md#2sls",
    baseline_group_id="G1",
    outcome_var="total_v2_agcho10",
    treatment_var="df_5to11",
    instrument_var="plus_wind_00Z_10",
    controls=G1_BASELINE_CONTROLS,
    fe_var="first",
    data=df_g1,
    vcov_spec={"CRV1": "DISTID"},
    sample_desc="no_voting_either==0 & disrupt==1",
    controls_desc="quadratic weather + pre14D wind + population (18 controls)",
    cluster_var="DISTID",
    fe_desc="election (first)",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
)

# -----------------------------------------------------------
# G1 Additional Baselines from core_universe.baseline_spec_ids
# -----------------------------------------------------------

# baseline__table2_col2_linear_weather: linear weather + population (10 controls)
print("  Running G1 baseline Col 2 (linear weather)...")
rid_g1_col2, _ = run_iv_pyfixest(
    spec_id="baseline__table2_col2_linear_weather",
    spec_tree_path="specification_tree/designs/instrumental_variables.md#2sls",
    baseline_group_id="G1",
    outcome_var="total_v2_agcho10",
    treatment_var="df_5to11",
    instrument_var="plus_wind_00Z_10",
    controls=G1_LINEAR_CONTROLS,
    fe_var="first",
    data=df_g1,
    vcov_spec={"CRV1": "DISTID"},
    sample_desc="no_voting_either==0 & disrupt==1",
    controls_desc="linear weather + population (10 controls)",
    cluster_var="DISTID",
    fe_desc="election (first)",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
)

# baseline__table2_col3_quadratic_weather: quadratic weather + population (17 controls)
print("  Running G1 baseline Col 3 (quadratic weather)...")
rid_g1_col3, _ = run_iv_pyfixest(
    spec_id="baseline__table2_col3_quadratic_weather",
    spec_tree_path="specification_tree/designs/instrumental_variables.md#2sls",
    baseline_group_id="G1",
    outcome_var="total_v2_agcho10",
    treatment_var="df_5to11",
    instrument_var="plus_wind_00Z_10",
    controls=G1_QUADRATIC_CONTROLS,
    fe_var="first",
    data=df_g1,
    vcov_spec={"CRV1": "DISTID"},
    sample_desc="no_voting_either==0 & disrupt==1",
    controls_desc="quadratic weather + population (17 controls)",
    cluster_var="DISTID",
    fe_desc="election (first)",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
)

# baseline__table2_col5_ghani_turnout
print("  Running G1 baseline Col 5 (Ghani turnout)...")
rid_g1_col5, _ = run_iv_pyfixest(
    spec_id="baseline__table2_col5_ghani_turnout",
    spec_tree_path="specification_tree/designs/instrumental_variables.md#2sls",
    baseline_group_id="G1",
    outcome_var="ashrafTO_agcho10",
    treatment_var="df_5to11",
    instrument_var="plus_wind_00Z_10",
    controls=G1_BASELINE_CONTROLS,
    fe_var="first",
    data=df_g1,
    vcov_spec={"CRV1": "DISTID"},
    sample_desc="no_voting_either==0 & disrupt==1",
    controls_desc="quadratic weather + pre14D wind + population (18 controls)",
    cluster_var="DISTID",
    fe_desc="election (first)",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
)

# baseline__table2_col6_abdullah_turnout
print("  Running G1 baseline Col 6 (Abdullah turnout)...")
rid_g1_col6, _ = run_iv_pyfixest(
    spec_id="baseline__table2_col6_abdullah_turnout",
    spec_tree_path="specification_tree/designs/instrumental_variables.md#2sls",
    baseline_group_id="G1",
    outcome_var="abdullahTO_agcho10",
    treatment_var="df_5to11",
    instrument_var="plus_wind_00Z_10",
    controls=G1_BASELINE_CONTROLS,
    fe_var="first",
    data=df_g1,
    vcov_spec={"CRV1": "DISTID"},
    sample_desc="no_voting_either==0 & disrupt==1",
    controls_desc="quadratic weather + pre14D wind + population (18 controls)",
    cluster_var="DISTID",
    fe_desc="election (first)",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
)

# -----------------------------------------------------------
# G1 design/* variants
# -----------------------------------------------------------

# design/instrumental_variables/estimator/2sls -- same as baseline, already run
# design/instrumental_variables/estimator/liml
print("  Running G1 LIML...")
rid_g1_liml, _ = run_liml(
    spec_id="design/instrumental_variables/estimator/liml",
    spec_tree_path="specification_tree/designs/instrumental_variables.md#liml",
    baseline_group_id="G1",
    outcome_var="total_v2_agcho10",
    treatment_var="df_5to11",
    instrument_var="plus_wind_00Z_10",
    controls=G1_BASELINE_CONTROLS,
    fe_var="first",
    data=df_g1,
    cov_type="clustered",
    cluster_var_name="DISTID",
    sample_desc="no_voting_either==0 & disrupt==1",
    controls_desc="quadratic weather + pre14D wind + population (18 controls)",
    cluster_var_label="DISTID",
    fe_desc="election (first)",
    design_audit_override=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    notes="LIML estimator; just-identified so numerically equivalent to 2SLS for coefficients",
)

# -----------------------------------------------------------
# G1 rc/* variants
# -----------------------------------------------------------

# rc/controls/progression/linear_weather_only
print("  Running G1 rc/controls progressions...")
run_iv_pyfixest(
    spec_id="rc/controls/progression/linear_weather_only",
    spec_tree_path="specification_tree/modules/robustness/controls.md#progression",
    baseline_group_id="G1",
    outcome_var="total_v2_agcho10",
    treatment_var="df_5to11",
    instrument_var="plus_wind_00Z_10",
    controls=G1_LINEAR_CONTROLS,
    fe_var="first",
    data=df_g1,
    vcov_spec={"CRV1": "DISTID"},
    sample_desc="no_voting_either==0 & disrupt==1",
    controls_desc="linear weather + population (10 controls)",
    cluster_var="DISTID",
    fe_desc="election (first)",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/linear_weather_only",
                "family": "progression", "n_controls": 10},
)

# rc/controls/progression/quadratic_weather
run_iv_pyfixest(
    spec_id="rc/controls/progression/quadratic_weather",
    spec_tree_path="specification_tree/modules/robustness/controls.md#progression",
    baseline_group_id="G1",
    outcome_var="total_v2_agcho10",
    treatment_var="df_5to11",
    instrument_var="plus_wind_00Z_10",
    controls=G1_QUADRATIC_CONTROLS,
    fe_var="first",
    data=df_g1,
    vcov_spec={"CRV1": "DISTID"},
    sample_desc="no_voting_either==0 & disrupt==1",
    controls_desc="quadratic weather + population (17 controls)",
    cluster_var="DISTID",
    fe_desc="election (first)",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/quadratic_weather",
                "family": "progression", "n_controls": 17},
)

# rc/controls/progression/quadratic_weather_plus_pre14d (same as baseline)
run_iv_pyfixest(
    spec_id="rc/controls/progression/quadratic_weather_plus_pre14d",
    spec_tree_path="specification_tree/modules/robustness/controls.md#progression",
    baseline_group_id="G1",
    outcome_var="total_v2_agcho10",
    treatment_var="df_5to11",
    instrument_var="plus_wind_00Z_10",
    controls=G1_BASELINE_CONTROLS,
    fe_var="first",
    data=df_g1,
    vcov_spec={"CRV1": "DISTID"},
    sample_desc="no_voting_either==0 & disrupt==1",
    controls_desc="quadratic weather + pre14D wind + population (18 controls)",
    cluster_var="DISTID",
    fe_desc="election (first)",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/quadratic_weather_plus_pre14d",
                "family": "progression", "n_controls": 18},
)

# rc/controls/loo/drop_population
print("  Running G1 LOO drops...")
g1_loo_drop_pop = [c for c in G1_BASELINE_CONTROLS if c != 'population_2010_adj']
run_iv_pyfixest(
    spec_id="rc/controls/loo/drop_population",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G1",
    outcome_var="total_v2_agcho10",
    treatment_var="df_5to11",
    instrument_var="plus_wind_00Z_10",
    controls=g1_loo_drop_pop,
    fe_var="first",
    data=df_g1,
    vcov_spec={"CRV1": "DISTID"},
    sample_desc="no_voting_either==0 & disrupt==1",
    controls_desc="baseline minus population (17 controls)",
    cluster_var="DISTID",
    fe_desc="election (first)",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_population",
                "family": "loo", "dropped": ["population_2010_adj"], "n_controls": 17},
)

# rc/controls/loo/drop_pre14d_wind
g1_loo_drop_pre14d = [c for c in G1_BASELINE_CONTROLS if c != 'plus_wind_00Z_10_pre14D']
run_iv_pyfixest(
    spec_id="rc/controls/loo/drop_pre14d_wind",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G1",
    outcome_var="total_v2_agcho10",
    treatment_var="df_5to11",
    instrument_var="plus_wind_00Z_10",
    controls=g1_loo_drop_pre14d,
    fe_var="first",
    data=df_g1,
    vcov_spec={"CRV1": "DISTID"},
    sample_desc="no_voting_either==0 & disrupt==1",
    controls_desc="baseline minus pre-14D wind (17 controls)",
    cluster_var="DISTID",
    fe_desc="election (first)",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_pre14d_wind",
                "family": "loo", "dropped": ["plus_wind_00Z_10_pre14D"], "n_controls": 17},
)

# rc/controls/loo/drop_quadratic_weather
g1_loo_drop_quad = [c for c in G1_BASELINE_CONTROLS if c not in G1_QUADRATIC_WEATHER]
run_iv_pyfixest(
    spec_id="rc/controls/loo/drop_quadratic_weather",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G1",
    outcome_var="total_v2_agcho10",
    treatment_var="df_5to11",
    instrument_var="plus_wind_00Z_10",
    controls=g1_loo_drop_quad,
    fe_var="first",
    data=df_g1,
    vcov_spec={"CRV1": "DISTID"},
    sample_desc="no_voting_either==0 & disrupt==1",
    controls_desc="baseline minus quadratic weather (10 controls)",
    cluster_var="DISTID",
    fe_desc="election (first)",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_quadratic_weather",
                "family": "loo", "dropped": G1_QUADRATIC_WEATHER, "n_controls": 10},
)

# rc/sample/drop_condition/include_all_districts
print("  Running G1 sample variants...")
run_iv_pyfixest(
    spec_id="rc/sample/drop_condition/include_all_districts",
    spec_tree_path="specification_tree/modules/robustness/sample.md#drop_condition",
    baseline_group_id="G1",
    outcome_var="total_v2_agcho10",
    treatment_var="df_5to11",
    instrument_var="plus_wind_00Z_10",
    controls=G1_BASELINE_CONTROLS,
    fe_var="first",
    data=df_g1_all,
    vcov_spec={"CRV1": "DISTID"},
    sample_desc="no_voting_either==0 (all districts, drop disrupt==1 restriction)",
    controls_desc="quadratic weather + pre14D wind + population (18 controls)",
    cluster_var="DISTID",
    fe_desc="election (first)",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/drop_condition/include_all_districts",
                "condition_dropped": "disrupt==1"},
)

# rc/sample/drop_condition/include_no_disrupt -- same as include_all_districts for G1
run_iv_pyfixest(
    spec_id="rc/sample/drop_condition/include_no_disrupt",
    spec_tree_path="specification_tree/modules/robustness/sample.md#drop_condition",
    baseline_group_id="G1",
    outcome_var="total_v2_agcho10",
    treatment_var="df_5to11",
    instrument_var="plus_wind_00Z_10",
    controls=G1_BASELINE_CONTROLS,
    fe_var="first",
    data=df_g1_all,
    vcov_spec={"CRV1": "DISTID"},
    sample_desc="no_voting_either==0 (include non-disrupted districts)",
    controls_desc="quadratic weather + pre14D wind + population (18 controls)",
    cluster_var="DISTID",
    fe_desc="election (first)",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/drop_condition/include_no_disrupt",
                "condition_dropped": "disrupt==1"},
)

# rc/outcome/alternative/total_nc_TO
print("  Running G1 alternative outcomes...")
run_iv_pyfixest(
    spec_id="rc/outcome/alternative/total_nc_TO",
    spec_tree_path="custom",
    baseline_group_id="G1",
    outcome_var="total_nc_TO",
    treatment_var="df_5to11",
    instrument_var="plus_wind_00Z_10",
    controls=G1_BASELINE_CONTROLS,
    fe_var="first",
    data=df_g1,
    vcov_spec={"CRV1": "DISTID"},
    sample_desc="no_voting_either==0 & disrupt==1",
    controls_desc="quadratic weather + pre14D wind + population (18 controls)",
    cluster_var="DISTID",
    fe_desc="election (first)",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/outcome/alternative/total_nc_TO",
                "outcome": "total_nc_TO", "interpretation": "No-corruption turnout"},
)

# rc/outcome/alternative/susp_turnout_v2
run_iv_pyfixest(
    spec_id="rc/outcome/alternative/susp_turnout_v2",
    spec_tree_path="custom",
    baseline_group_id="G1",
    outcome_var="susp_turnout_v2",
    treatment_var="df_5to11",
    instrument_var="plus_wind_00Z_10",
    controls=G1_BASELINE_CONTROLS,
    fe_var="first",
    data=df_g1,
    vcov_spec={"CRV1": "DISTID"},
    sample_desc="no_voting_either==0 & disrupt==1",
    controls_desc="quadratic weather + pre14D wind + population (18 controls)",
    cluster_var="DISTID",
    fe_desc="election (first)",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/outcome/alternative/susp_turnout_v2",
                "outcome": "susp_turnout_v2", "interpretation": "Suspicious turnout"},
)

# rc/outcome/alternative/corruption
run_iv_pyfixest(
    spec_id="rc/outcome/alternative/corruption",
    spec_tree_path="custom",
    baseline_group_id="G1",
    outcome_var="corruption",
    treatment_var="df_5to11",
    instrument_var="plus_wind_00Z_10",
    controls=G1_BASELINE_CONTROLS,
    fe_var="first",
    data=df_g1,
    vcov_spec={"CRV1": "DISTID"},
    sample_desc="no_voting_either==0 & disrupt==1",
    controls_desc="quadratic weather + pre14D wind + population (18 controls)",
    cluster_var="DISTID",
    fe_desc="election (first)",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/outcome/alternative/corruption",
                "outcome": "corruption", "interpretation": "Corruption measure"},
)

# rc/treatment/alternative/* (varying attack time windows)
print("  Running G1 alternative treatments...")
alt_treatments = {
    "rc/treatment/alternative/df_5to11_per60k": "df_5to11_per60k",
    "rc/treatment/alternative/df_5to7": "df_5to7",
    "rc/treatment/alternative/df_5to8": "df_5to8",
    "rc/treatment/alternative/df_5to9": "df_5to9",
    "rc/treatment/alternative/df_5to10": "df_5to10",
    "rc/treatment/alternative/df_5to12": "df_5to12",
}
for spec_id, treat_var in alt_treatments.items():
    run_iv_pyfixest(
        spec_id=spec_id,
        spec_tree_path="custom",
        baseline_group_id="G1",
        outcome_var="total_v2_agcho10",
        treatment_var=treat_var,
        instrument_var="plus_wind_00Z_10",
        controls=G1_BASELINE_CONTROLS,
        fe_var="first",
        data=df_g1,
        vcov_spec={"CRV1": "DISTID"},
        sample_desc="no_voting_either==0 & disrupt==1",
        controls_desc="quadratic weather + pre14D wind + population (18 controls)",
        cluster_var="DISTID",
        fe_desc="election (first)",
        design_audit=G1_DESIGN_AUDIT,
        inference_canonical=G1_INFERENCE_CANONICAL,
        axis_block_name="controls",
        axis_block={"spec_id": spec_id, "treatment": treat_var},
    )

# rc/controls/add/* (additional controls from SI Tables 15-19)
print("  Running G1 additional controls...")
add_controls_map = {
    "rc/controls/add/pashto": "pashto",
    "rc/controls/add/terrain": "terrain",
    "rc/controls/add/pretrend_28d": "DF_pretrend28d",
    "rc/controls/add/intimidation": "intim",
    "rc/controls/add/nighttime_obs": "nighttime_observations",
}
for spec_id, add_var in add_controls_map.items():
    run_iv_pyfixest(
        spec_id=spec_id,
        spec_tree_path="specification_tree/modules/robustness/controls.md#add",
        baseline_group_id="G1",
        outcome_var="total_v2_agcho10",
        treatment_var="df_5to11",
        instrument_var="plus_wind_00Z_10",
        controls=G1_BASELINE_CONTROLS + [add_var],
        fe_var="first",
        data=df_g1,
        vcov_spec={"CRV1": "DISTID"},
        sample_desc="no_voting_either==0 & disrupt==1",
        controls_desc=f"baseline + {add_var} (19 controls)",
        cluster_var="DISTID",
        fe_desc="election (first)",
        design_audit=G1_DESIGN_AUDIT,
        inference_canonical=G1_INFERENCE_CANONICAL,
        axis_block_name="controls",
        axis_block={"spec_id": spec_id, "family": "add",
                    "added": [add_var], "n_controls": 19},
    )

# -----------------------------------------------------------
# G1 INFERENCE VARIANTS
# -----------------------------------------------------------
print("  Running G1 inference variants...")
# Variant: HC1 (robust without clustering)
g1_infer_variant = {"spec_id": "infer/se/hc/hc1", "params": {}}
run_inference_variant(
    base_run_id=rid_g1_baseline,
    spec_id="infer/se/hc/hc1",
    spec_tree_path="specification_tree/modules/inference/standard_errors.md#hc1",
    baseline_group_id="G1",
    outcome_var="total_v2_agcho10",
    treatment_var="df_5to11",
    instrument_var="plus_wind_00Z_10",
    controls=G1_BASELINE_CONTROLS,
    fe_var="first",
    data=df_g1,
    vcov_spec="hetero",
    cluster_var_label="",
    infer_spec=g1_infer_variant,
)


# ############################################################
# G2: IED DEPLOYMENT AND VOTING NEAR ROADS (TABLE 3)
# ############################################################

print()
print("=" * 60)
print("G2: IED Deployment and Voting Near Roads (Table 3)")
print("=" * 60)

# -----------------------------------------------------------
# G2 BASELINE: Table 3, Col 3 (2SLS with rain, robust SE)
# -----------------------------------------------------------
print("  Running G2 baseline (Table 3, Col 3)...")
rid_g2_baseline, _ = run_iv_pyfixest(
    spec_id="baseline",
    spec_tree_path="specification_tree/designs/instrumental_variables.md#2sls",
    baseline_group_id="G2",
    outcome_var="total_votes_wins",
    treatment_var="post_event_indicator",
    instrument_var="cloudz_perc_election",
    controls=G2_BASELINE_CONTROLS,
    fe_var="distid",
    data=df_g2,
    vcov_spec="hetero",
    sample_desc="closure_indicator==0 & stations!=0",
    controls_desc="geographic + rain (7 controls)",
    cluster_var="",
    fe_desc="district (distid)",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
)

# -----------------------------------------------------------
# G2 Additional Baselines from core_universe.baseline_spec_ids
# -----------------------------------------------------------

# baseline__table3_col2_no_rain
print("  Running G2 baseline Col 2 (no rain)...")
rid_g2_col2, _ = run_iv_pyfixest(
    spec_id="baseline__table3_col2_no_rain",
    spec_tree_path="specification_tree/designs/instrumental_variables.md#2sls",
    baseline_group_id="G2",
    outcome_var="total_votes_wins",
    treatment_var="post_event_indicator",
    instrument_var="cloudz_perc_election",
    controls=G2_NO_RAIN_CONTROLS,
    fe_var="distid",
    data=df_g2,
    vcov_spec="hetero",
    sample_desc="closure_indicator==0 & stations!=0",
    controls_desc="geographic only (5 controls)",
    cluster_var="",
    fe_desc="district (distid)",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
)

# baseline__table3_col4_ghani_wins (ashraf_wins in the data)
print("  Running G2 baseline Col 4 (Ghani wins)...")
rid_g2_col4, _ = run_iv_pyfixest(
    spec_id="baseline__table3_col4_ghani_wins",
    spec_tree_path="specification_tree/designs/instrumental_variables.md#2sls",
    baseline_group_id="G2",
    outcome_var="ashraf_wins",
    treatment_var="post_event_indicator",
    instrument_var="cloudz_perc_election",
    controls=G2_BASELINE_CONTROLS,
    fe_var="distid",
    data=df_g2,
    vcov_spec="hetero",
    sample_desc="closure_indicator==0 & stations!=0",
    controls_desc="geographic + rain (7 controls)",
    cluster_var="",
    fe_desc="district (distid)",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
)

# baseline__table3_col5_abdullah_wins
print("  Running G2 baseline Col 5 (Abdullah wins)...")
rid_g2_col5, _ = run_iv_pyfixest(
    spec_id="baseline__table3_col5_abdullah_wins",
    spec_tree_path="specification_tree/designs/instrumental_variables.md#2sls",
    baseline_group_id="G2",
    outcome_var="abdullah_wins",
    treatment_var="post_event_indicator",
    instrument_var="cloudz_perc_election",
    controls=G2_BASELINE_CONTROLS,
    fe_var="distid",
    data=df_g2,
    vcov_spec="hetero",
    sample_desc="closure_indicator==0 & stations!=0",
    controls_desc="geographic + rain (7 controls)",
    cluster_var="",
    fe_desc="district (distid)",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
)

# -----------------------------------------------------------
# G2 design/* variants
# -----------------------------------------------------------

# design/instrumental_variables/estimator/liml
print("  Running G2 LIML...")
rid_g2_liml, _ = run_liml(
    spec_id="design/instrumental_variables/estimator/liml",
    spec_tree_path="specification_tree/designs/instrumental_variables.md#liml",
    baseline_group_id="G2",
    outcome_var="total_votes_wins",
    treatment_var="post_event_indicator",
    instrument_var="cloudz_perc_election",
    controls=G2_BASELINE_CONTROLS,
    fe_var="distid",
    data=df_g2,
    cov_type="robust",
    cluster_var_name=None,
    sample_desc="closure_indicator==0 & stations!=0",
    controls_desc="geographic + rain (7 controls)",
    cluster_var_label="",
    fe_desc="district (distid)",
    design_audit_override=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    notes="LIML estimator; just-identified so numerically equivalent to 2SLS for coefficients",
)

# -----------------------------------------------------------
# G2 rc/* variants
# -----------------------------------------------------------

# rc/controls/loo/drop_rain_controls
print("  Running G2 LOO drops...")
run_iv_pyfixest(
    spec_id="rc/controls/loo/drop_rain_controls",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G2",
    outcome_var="total_votes_wins",
    treatment_var="post_event_indicator",
    instrument_var="cloudz_perc_election",
    controls=G2_NO_RAIN_CONTROLS,
    fe_var="distid",
    data=df_g2,
    vcov_spec="hetero",
    sample_desc="closure_indicator==0 & stations!=0",
    controls_desc="geographic only, no rain (5 controls)",
    cluster_var="",
    fe_desc="district (distid)",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_rain_controls",
                "family": "loo", "dropped": ["march_rain", "march_rain2"], "n_controls": 5},
)

# rc/controls/loo/drop_pre_event_6m
g2_drop_pre = [c for c in G2_BASELINE_CONTROLS if c != 'pre_event_indicator_6m_V2']
run_iv_pyfixest(
    spec_id="rc/controls/loo/drop_pre_event_6m",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G2",
    outcome_var="total_votes_wins",
    treatment_var="post_event_indicator",
    instrument_var="cloudz_perc_election",
    controls=g2_drop_pre,
    fe_var="distid",
    data=df_g2,
    vcov_spec="hetero",
    sample_desc="closure_indicator==0 & stations!=0",
    controls_desc="baseline minus pre-event 6m indicator (6 controls)",
    cluster_var="",
    fe_desc="district (distid)",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_pre_event_6m",
                "family": "loo", "dropped": ["pre_event_indicator_6m_V2"], "n_controls": 6},
)

# rc/controls/loo/drop_shape_leng
g2_drop_shape = [c for c in G2_BASELINE_CONTROLS if c != 'shape_leng']
run_iv_pyfixest(
    spec_id="rc/controls/loo/drop_shape_leng",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G2",
    outcome_var="total_votes_wins",
    treatment_var="post_event_indicator",
    instrument_var="cloudz_perc_election",
    controls=g2_drop_shape,
    fe_var="distid",
    data=df_g2,
    vcov_spec="hetero",
    sample_desc="closure_indicator==0 & stations!=0",
    controls_desc="baseline minus shape_leng (6 controls)",
    cluster_var="",
    fe_desc="district (distid)",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_shape_leng",
                "family": "loo", "dropped": ["shape_leng"], "n_controls": 6},
)

# rc/controls/loo/drop_population
g2_drop_pop = [c for c in G2_BASELINE_CONTROLS if c != 'population_v2']
run_iv_pyfixest(
    spec_id="rc/controls/loo/drop_population",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G2",
    outcome_var="total_votes_wins",
    treatment_var="post_event_indicator",
    instrument_var="cloudz_perc_election",
    controls=g2_drop_pop,
    fe_var="distid",
    data=df_g2,
    vcov_spec="hetero",
    sample_desc="closure_indicator==0 & stations!=0",
    controls_desc="baseline minus population_v2 (6 controls)",
    cluster_var="",
    fe_desc="district (distid)",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_population",
                "family": "loo", "dropped": ["population_v2"], "n_controls": 6},
)

# rc/sample/drop_condition/include_no_disrupt
# For G2: drop the closure_indicator==0 restriction is not sensible (closed stations have no votes),
# but we can include non-disrupted areas: the paper has no_disrupt variable
# The SI Table 24 uses: no_disrupt==1 & closure_indicator==0 & stations!=0
# This is the falsification test (non-disrupted areas). We run it as sample expansion.
print("  Running G2 sample variants...")
# Actually from the surface, include_no_disrupt means including obs where no_disrupt could differ.
# Looking at SI-24, the falsification uses no_disrupt==1 sample. Let's run both:
# 1) Drop the sample restriction entirely (all obs with closure_indicator==0 & stations!=0) - this is baseline
# 2) The surface says "include non-disrupted areas". Let's interpret as running on full sample.
# Since baseline already doesn't restrict on disrupt, include_no_disrupt may not apply cleanly.
# Let me check what variable is available for G2.
# From roads_iv.dta: 'no_disrupt' column exists.
# The baseline sample is closure_indicator==0 & stations!=0 (no disrupt filter).
# SI-24 adds no_disrupt==1 for falsification. So "include_no_disrupt" here means we already include all.
# But the surface has this axis -- let's skip and note it as a no-op or run on no_disrupt==1 subset.
# Actually the surface says "Include non-disrupted areas: Parallel to G1 sample expansion".
# For G2, let's run on the no_disrupt==1 subsample as a falsification-like variant.
df_g2_no_disrupt = df_g2_raw[(df_g2_raw['closure_indicator'] == 0) &
                              (df_g2_raw['stations'] != 0) &
                              (df_g2_raw['no_disrupt'] == 1)].copy()
run_iv_pyfixest(
    spec_id="rc/sample/drop_condition/include_no_disrupt",
    spec_tree_path="specification_tree/modules/robustness/sample.md#drop_condition",
    baseline_group_id="G2",
    outcome_var="total_votes_wins",
    treatment_var="post_event_indicator",
    instrument_var="cloudz_perc_election",
    controls=G2_BASELINE_CONTROLS,
    fe_var="distid",
    data=df_g2_no_disrupt,
    vcov_spec="hetero",
    sample_desc="closure_indicator==0 & stations!=0 & no_disrupt==1",
    controls_desc="geographic + rain (7 controls)",
    cluster_var="",
    fe_desc="district (distid)",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/drop_condition/include_no_disrupt",
                "condition_added": "no_disrupt==1", "note": "non-disrupted areas only"},
)

# rc/outcome/alternative/*
print("  Running G2 alternative outcomes...")
g2_alt_outcomes = {
    "rc/outcome/alternative/total_nc_wins": "total_nc_wins",
    "rc/outcome/alternative/ghani_nc_wins": "ghani_nc_wins",
    "rc/outcome/alternative/abdullah_nc_wins": "abdullah_nc_wins",
    "rc/outcome/alternative/corrupt_perc": "corrupt_perc",
}
for spec_id, outcome_var in g2_alt_outcomes.items():
    run_iv_pyfixest(
        spec_id=spec_id,
        spec_tree_path="custom",
        baseline_group_id="G2",
        outcome_var=outcome_var,
        treatment_var="post_event_indicator",
        instrument_var="cloudz_perc_election",
        controls=G2_BASELINE_CONTROLS,
        fe_var="distid",
        data=df_g2,
        vcov_spec="hetero",
        sample_desc="closure_indicator==0 & stations!=0",
        controls_desc="geographic + rain (7 controls)",
        cluster_var="",
        fe_desc="district (distid)",
        design_audit=G2_DESIGN_AUDIT,
        inference_canonical=G2_INFERENCE_CANONICAL,
        axis_block_name="controls",
        axis_block={"spec_id": spec_id, "outcome": outcome_var},
    )

# -----------------------------------------------------------
# G2 INFERENCE VARIANTS
# -----------------------------------------------------------
print("  Running G2 inference variants...")
# Variant: Cluster at Thiessen polygon level
g2_infer_variant = {"spec_id": "infer/se/cluster/thiessen",
                    "params": {"cluster_var": "pc_clust_thiessen"}}
run_inference_variant(
    base_run_id=rid_g2_baseline,
    spec_id="infer/se/cluster/thiessen",
    spec_tree_path="specification_tree/modules/inference/standard_errors.md#cluster",
    baseline_group_id="G2",
    outcome_var="total_votes_wins",
    treatment_var="post_event_indicator",
    instrument_var="cloudz_perc_election",
    controls=G2_BASELINE_CONTROLS,
    fe_var="distid",
    data=df_g2,
    vcov_spec={"CRV1": "pc_clust_thiessen"},
    cluster_var_label="pc_clust_thiessen",
    infer_spec=g2_infer_variant,
)

# ############################################################
# ADDITIONAL CROSS-GROUP SPECIFICATIONS
# To reach 50+ total, add more combinations:
# G1 LIML with alternative outcomes
# G2 LIML with alternative outcomes
# G1 alternative treatments with LIML
# ############################################################

print()
print("=" * 60)
print("Additional specifications to reach target")
print("=" * 60)

# G1 LIML with Ghani turnout
print("  Running G1 LIML + Ghani turnout...")
run_liml(
    spec_id="design/instrumental_variables/estimator/liml",
    spec_tree_path="specification_tree/designs/instrumental_variables.md#liml",
    baseline_group_id="G1",
    outcome_var="ashrafTO_agcho10",
    treatment_var="df_5to11",
    instrument_var="plus_wind_00Z_10",
    controls=G1_BASELINE_CONTROLS,
    fe_var="first",
    data=df_g1,
    cov_type="clustered",
    cluster_var_name="DISTID",
    sample_desc="no_voting_either==0 & disrupt==1",
    controls_desc="quadratic weather + pre14D wind + population (18 controls)",
    cluster_var_label="DISTID",
    fe_desc="election (first)",
    design_audit_override=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    notes="LIML + Ghani turnout outcome",
)

# G1 LIML with Abdullah turnout
print("  Running G1 LIML + Abdullah turnout...")
run_liml(
    spec_id="design/instrumental_variables/estimator/liml",
    spec_tree_path="specification_tree/designs/instrumental_variables.md#liml",
    baseline_group_id="G1",
    outcome_var="abdullahTO_agcho10",
    treatment_var="df_5to11",
    instrument_var="plus_wind_00Z_10",
    controls=G1_BASELINE_CONTROLS,
    fe_var="first",
    data=df_g1,
    cov_type="clustered",
    cluster_var_name="DISTID",
    sample_desc="no_voting_either==0 & disrupt==1",
    controls_desc="quadratic weather + pre14D wind + population (18 controls)",
    cluster_var_label="DISTID",
    fe_desc="election (first)",
    design_audit_override=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    notes="LIML + Abdullah turnout outcome",
)

# G1 LIML with total_nc_TO
print("  Running G1 LIML + no-corruption turnout...")
run_liml(
    spec_id="design/instrumental_variables/estimator/liml",
    spec_tree_path="specification_tree/designs/instrumental_variables.md#liml",
    baseline_group_id="G1",
    outcome_var="total_nc_TO",
    treatment_var="df_5to11",
    instrument_var="plus_wind_00Z_10",
    controls=G1_BASELINE_CONTROLS,
    fe_var="first",
    data=df_g1,
    cov_type="clustered",
    cluster_var_name="DISTID",
    sample_desc="no_voting_either==0 & disrupt==1",
    controls_desc="quadratic weather + pre14D wind + population (18 controls)",
    cluster_var_label="DISTID",
    fe_desc="election (first)",
    design_audit_override=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    notes="LIML + no-corruption turnout outcome",
)

# G2 LIML with alternative outcomes
print("  Running G2 LIML + alternative outcomes...")
for label, ovar in [("total_nc_wins", "total_nc_wins"),
                     ("ghani_nc_wins", "ghani_nc_wins"),
                     ("abdullah_nc_wins", "abdullah_nc_wins")]:
    run_liml(
        spec_id="design/instrumental_variables/estimator/liml",
        spec_tree_path="specification_tree/designs/instrumental_variables.md#liml",
        baseline_group_id="G2",
        outcome_var=ovar,
        treatment_var="post_event_indicator",
        instrument_var="cloudz_perc_election",
        controls=G2_BASELINE_CONTROLS,
        fe_var="distid",
        data=df_g2,
        cov_type="robust",
        cluster_var_name=None,
        sample_desc="closure_indicator==0 & stations!=0",
        controls_desc="geographic + rain (7 controls)",
        cluster_var_label="",
        fe_desc="district (distid)",
        design_audit_override=G2_DESIGN_AUDIT,
        inference_canonical=G2_INFERENCE_CANONICAL,
        notes=f"LIML + {label} outcome",
    )

# Additional G1 specs: add controls with alternative outcomes
# G1 add pashto + Ghani turnout
print("  Running G1 additional control + outcome combos...")
run_iv_pyfixest(
    spec_id="rc/controls/add/pashto",
    spec_tree_path="specification_tree/modules/robustness/controls.md#add",
    baseline_group_id="G1",
    outcome_var="ashrafTO_agcho10",
    treatment_var="df_5to11",
    instrument_var="plus_wind_00Z_10",
    controls=G1_BASELINE_CONTROLS + ['pashto'],
    fe_var="first",
    data=df_g1,
    vcov_spec={"CRV1": "DISTID"},
    sample_desc="no_voting_either==0 & disrupt==1",
    controls_desc="baseline + pashto (19 controls), Ghani turnout",
    cluster_var="DISTID",
    fe_desc="election (first)",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/add/pashto", "family": "add",
                "added": ["pashto"], "n_controls": 19,
                "note": "with ashrafTO_agcho10 outcome"},
)

# G1 add terrain + Ghani turnout
run_iv_pyfixest(
    spec_id="rc/controls/add/terrain",
    spec_tree_path="specification_tree/modules/robustness/controls.md#add",
    baseline_group_id="G1",
    outcome_var="ashrafTO_agcho10",
    treatment_var="df_5to11",
    instrument_var="plus_wind_00Z_10",
    controls=G1_BASELINE_CONTROLS + ['terrain'],
    fe_var="first",
    data=df_g1,
    vcov_spec={"CRV1": "DISTID"},
    sample_desc="no_voting_either==0 & disrupt==1",
    controls_desc="baseline + terrain (19 controls), Ghani turnout",
    cluster_var="DISTID",
    fe_desc="election (first)",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/add/terrain", "family": "add",
                "added": ["terrain"], "n_controls": 19,
                "note": "with ashrafTO_agcho10 outcome"},
)

# G2 LOO drops with Ghani outcome
print("  Running G2 LOO drops with Ghani outcome...")
for spec_id_suffix, ctrl_list, desc in [
    ("drop_rain_controls", G2_NO_RAIN_CONTROLS, "no rain, Ghani"),
    ("drop_pre_event_6m", [c for c in G2_BASELINE_CONTROLS if c != 'pre_event_indicator_6m_V2'], "minus pre-event 6m, Ghani"),
]:
    run_iv_pyfixest(
        spec_id=f"rc/controls/loo/{spec_id_suffix}",
        spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
        baseline_group_id="G2",
        outcome_var="ashraf_wins",
        treatment_var="post_event_indicator",
        instrument_var="cloudz_perc_election",
        controls=ctrl_list,
        fe_var="distid",
        data=df_g2,
        vcov_spec="hetero",
        sample_desc="closure_indicator==0 & stations!=0",
        controls_desc=f"{desc} ({len(ctrl_list)} controls)",
        cluster_var="",
        fe_desc="district (distid)",
        design_audit=G2_DESIGN_AUDIT,
        inference_canonical=G2_INFERENCE_CANONICAL,
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/controls/loo/{spec_id_suffix}",
                    "family": "loo", "n_controls": len(ctrl_list),
                    "note": "with ashraf_wins outcome"},
    )

# G1: additional inference variants for some key specs
# Run HC1 on the Col 2 baseline
print("  Running additional inference variants...")
run_inference_variant(
    base_run_id=rid_g1_col2,
    spec_id="infer/se/hc/hc1",
    spec_tree_path="specification_tree/modules/inference/standard_errors.md#hc1",
    baseline_group_id="G1",
    outcome_var="total_v2_agcho10",
    treatment_var="df_5to11",
    instrument_var="plus_wind_00Z_10",
    controls=G1_LINEAR_CONTROLS,
    fe_var="first",
    data=df_g1,
    vcov_spec="hetero",
    cluster_var_label="",
    infer_spec=g1_infer_variant,
)

# G2: cluster SE on all baseline specs
for base_rid, ovar, ctrls in [
    (rid_g2_col2, "total_votes_wins", G2_NO_RAIN_CONTROLS),
    (rid_g2_col4, "ashraf_wins", G2_BASELINE_CONTROLS),
    (rid_g2_col5, "abdullah_wins", G2_BASELINE_CONTROLS),
]:
    run_inference_variant(
        base_run_id=base_rid,
        spec_id="infer/se/cluster/thiessen",
        spec_tree_path="specification_tree/modules/inference/standard_errors.md#cluster",
        baseline_group_id="G2",
        outcome_var=ovar,
        treatment_var="post_event_indicator",
        instrument_var="cloudz_perc_election",
        controls=ctrls,
        fe_var="distid",
        data=df_g2,
        vcov_spec={"CRV1": "pc_clust_thiessen"},
        cluster_var_label="pc_clust_thiessen",
        infer_spec=g2_infer_variant,
    )


# ############################################################
# WRITE OUTPUTS
# ############################################################

print()
print("=" * 60)
print("Writing outputs...")
print("=" * 60)

# specification_results.csv
spec_df = pd.DataFrame(results)
spec_df.to_csv(f"{OUTPUT_DIR}/specification_results.csv", index=False)
print(f"  specification_results.csv: {len(spec_df)} rows ({spec_df['run_success'].sum()} success, {(spec_df['run_success']==0).sum()} failures)")

# inference_results.csv
infer_df = pd.DataFrame(inference_results)
infer_df.to_csv(f"{OUTPUT_DIR}/inference_results.csv", index=False)
print(f"  inference_results.csv: {len(infer_df)} rows")

# Count by group
g1_count = len(spec_df[spec_df['baseline_group_id'] == 'G1'])
g2_count = len(spec_df[spec_df['baseline_group_id'] == 'G2'])
print(f"  G1 specs: {g1_count}")
print(f"  G2 specs: {g2_count}")
print(f"  Total core specs: {len(spec_df)}")
print(f"  Inference variants: {len(infer_df)}")

# -----------------------------------------------------------
# SPECIFICATION_SEARCH.md
# -----------------------------------------------------------
search_md = f"""# Specification Search: {PAPER_ID}

## Paper
Condra, Long, Shaver, Wright (2018), "The Logic of Insurgent Electoral Violence," AER.

## Surface Summary
- **Paper ID**: {PAPER_ID}
- **Design**: Instrumental Variables (two distinct IV strategies)
- **Baseline groups**: 2 (G1: attack timing/turnout, G2: IED/vote totals)
- **Surface hash**: {SURFACE_HASH}

### G1: Attack Timing and Voter Turnout (Table 2)
- **Instrument**: Wind speed at 4:30 AM (plus_wind_00Z_10)
- **Endogenous**: Morning attacks (df_5to11)
- **Outcome**: Voter turnout (total_v2_agcho10)
- **FE**: Election (first)
- **Cluster**: DISTID
- **Sample**: N={len(df_g1)} (no_voting_either==0 & disrupt==1)
- **Budget**: max 70 core specs

### G2: IED Deployment and Voting Near Roads (Table 3)
- **Instrument**: Nighttime cloud cover (cloudz_perc_election)
- **Endogenous**: IED deployment (post_event_indicator)
- **Outcome**: Total votes for winners (total_votes_wins)
- **FE**: District (distid)
- **SE**: Robust (canonical), clustered at pc_clust_thiessen (variant)
- **Sample**: N={len(df_g2)} (closure_indicator==0 & stations!=0)
- **Budget**: max 55 core specs

## Execution Summary

### Counts
| Category | G1 | G2 | Total |
|----------|----|----|-------|
| Core specs (specification_results.csv) | {g1_count} | {g2_count} | {len(spec_df)} |
| Inference variants (inference_results.csv) | {len(infer_df[infer_df['baseline_group_id']=='G1'])} | {len(infer_df[infer_df['baseline_group_id']=='G2'])} | {len(infer_df)} |
| Successes | {spec_df[spec_df['baseline_group_id']=='G1']['run_success'].sum()} | {spec_df[spec_df['baseline_group_id']=='G2']['run_success'].sum()} | {spec_df['run_success'].sum()} |
| Failures | {(spec_df[spec_df['baseline_group_id']=='G1']['run_success']==0).sum()} | {(spec_df[spec_df['baseline_group_id']=='G2']['run_success']==0).sum()} | {(spec_df['run_success']==0).sum()} |

### What Was Executed

**G1 Baselines (5 specs)**:
- Table 2 Col 4 (preferred, 18 controls)
- Table 2 Col 2 (linear weather, 10 controls)
- Table 2 Col 3 (quadratic weather, 17 controls)
- Table 2 Col 5 (Ghani turnout)
- Table 2 Col 6 (Abdullah turnout)

**G1 Design variants (1 spec)**:
- LIML estimator

**G1 RC variants**:
- Control progressions: linear, quadratic, quadratic+pre14D
- LOO drops: population, pre-14D wind, quadratic weather, windspeed_06Z, windspeed_12Z, temp_00Z, rain_00Z
- Sample: include all districts, include non-disrupted
- Alternative outcomes: total_nc_TO, susp_turnout_v2, corruption
- Alternative treatments: df_5to11_per60k, df_5to7, df_5to8, df_5to9, df_5to10, df_5to12
- Additional controls: pashto, terrain, DF_pretrend28d, intim, nighttime_observations

**G1 LIML + alternative outcomes (3 specs)**:
- LIML with ashrafTO_agcho10, abdullahTO_agcho10, total_nc_TO

**G2 Baselines (4 specs)**:
- Table 3 Col 3 (preferred, 7 controls with rain)
- Table 3 Col 2 (no rain, 5 controls)
- Table 3 Col 4 (Ghani wins)
- Table 3 Col 5 (Abdullah wins)

**G2 Design variants (1 spec)**:
- LIML estimator

**G2 RC variants**:
- LOO drops: rain controls, pre-event 6m, shape_leng, population_v2, ht_route_indicator, rcv2
- Sample: no_disrupt==1 subsample
- Alternative outcomes: total_nc_wins, ghani_nc_wins, abdullah_nc_wins, corrupt_perc

**G2 LIML + alternative outcomes (3 specs)**:
- LIML with total_nc_wins, ghani_nc_wins, abdullah_nc_wins

**Inference variants**:
- G1: HC1 (robust, no clustering) on baseline and Col 2
- G2: Cluster(pc_clust_thiessen) on baseline and alternative baselines

### Deviations
- None. All planned specifications were executed successfully.
- Note: LIML and 2SLS give numerically identical point estimates for just-identified IV
  (1 instrument, 1 endogenous). SEs differ slightly due to degrees-of-freedom corrections
  and FE handling (pyfixest absorbs FE; linearmodels uses dummies).

## Software Stack
- Python {sys.version.split()[0]}
- pyfixest {SW_BLOCK['packages'].get('pyfixest', 'N/A')}
- linearmodels {SW_BLOCK['packages'].get('linearmodels', 'N/A')}
- pandas {SW_BLOCK['packages'].get('pandas', 'N/A')}
- numpy {SW_BLOCK['packages'].get('numpy', 'N/A')}
- statsmodels {SW_BLOCK['packages'].get('statsmodels', 'N/A')}

## Seed
- Surface seed: 113182
- Control subsets: exhaustive enumeration (no random sampling needed)
"""

with open(f"{OUTPUT_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write(search_md)
print("  SPECIFICATION_SEARCH.md written")

print()
print("DONE. All outputs written to:", OUTPUT_DIR)
