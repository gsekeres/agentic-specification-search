"""
Specification Search Script for Corno, La Ferrara & Burns (2019/2022)
"Interaction, Stereotypes and Performance: Evidence from South Africa"

Paper ID: 174501-V1

Surface-driven execution:
  - G1: Race IAT (DscoreraceIAT) ~ mixracebas, Black subsample primary
  - G2: Academic outcomes (GPA, examspassed, continue, PCAperf) ~ mixracebas, Black subsample
  - G3: Social outcomes (PCAfriend, PCAattitude, PCAcomm, PCAsocial) ~ mixracebas, White subsample
  - Randomized experiment: natural roommate assignment within residences at UCT
  - Cluster SE at room level (roomnum_base)

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

sys.path.insert(0, "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/scripts")
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash, software_block
)

BASE_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search"
DATA_DIR = f"{BASE_DIR}/data/downloads/extracted/174501-V1"
PAPER_ID = "174501-V1"

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

# ============================================================
# LOAD AND PREPARE DATA
# ============================================================
df_raw = pd.read_stata(f"{DATA_DIR}/Data/Clean/uctdata_balanced.dta")

# Create lagged outcomes from baseline to follow-up
df_base = df_raw[df_raw['round'] == 'Baseline'][['individual', 'DscoreraceIAT', 'DscoreacaIAT']].copy()
df_base.columns = ['individual', 'L_DscoreraceIAT', 'L_DscoreacaIAT']
df = df_raw[df_raw['round'] == 'Follow-up'].copy()
df = df.merge(df_base, on='individual', how='left')

# Convert Female from categorical to numeric dummy
df['Female'] = (df['Female'] == 'female').astype(float)

# Convert float32 to float64 for precision
for col in df.columns:
    if df[col].dtype == np.float32:
        df[col] = df[col].astype(np.float64)

# Rename 'continue' to avoid Python keyword issues
df['continue_var'] = df['continue']

# Create string FE variables for pyfixest
df['Res_base_str'] = df['Res_base'].astype(int).astype(str)
df['regprogram_str'] = df['regprogram'].astype(str)
df['blackRes_str'] = df['blackRes'].astype(int).astype(str)
df['whiteRes_str'] = df['whiteRes'].astype(int).astype(str)
df['roomnum_base_str'] = df['roomnum_base'].astype(str)

# ============================================================
# VARIABLE DEFINITIONS
# ============================================================
# Own controls (for race subsamples)
controls_subsample = [
    'Female', 'Falseuct2012', 'missfalse', 'Foreign', 'foreign_missing',
    'privateschool_nomiss', 'privateschool_miss',
    'durpcabas_nomiss', 'durpcabas_miss',
    'consbas_nomiss', 'consbas_miss'
]

# Full sample controls (include race dummies)
controls_full = ['white', 'coloured', 'Else'] + controls_subsample

# Roommate controls
# NOTE: In the data, the variable is 'roconsbas_missing' not 'roconsbas_miss'
rocontrols = [
    'roFalseuct2012', 'missrofalse', 'roForeign_bas', 'roforeign_missingbas',
    'roprivschool_nomiss', 'roprivschool_miss',
    'rodurpcabas_nomiss', 'rodurpcabas_miss',
    'roconsbas_nomiss', 'roconsbas_missing'
]

# LOO variable groups (own controls - paired with miss indicators)
loo_own_groups = {
    'Female': ['Female'],
    'Falseuct2012': ['Falseuct2012', 'missfalse'],
    'Foreign': ['Foreign', 'foreign_missing'],
    'privateschool_nomiss': ['privateschool_nomiss', 'privateschool_miss'],
    'durpcabas_nomiss': ['durpcabas_nomiss', 'durpcabas_miss'],
    'consbas_nomiss': ['consbas_nomiss', 'consbas_miss'],
}

# LOO variable groups (roommate controls)
loo_ro_groups = {
    'roFalseuct2012': ['roFalseuct2012', 'missrofalse'],
    'roForeign_bas': ['roForeign_bas', 'roforeign_missingbas'],
    'roprivschool_nomiss': ['roprivschool_nomiss', 'roprivschool_miss'],
    'rodurpcabas_nomiss': ['rodurpcabas_nomiss', 'rodurpcabas_miss'],
    'roconsbas_nomiss': ['roconsbas_nomiss', 'roconsbas_missing'],
}

# Design audits from surface
design_audits = {g['baseline_group_id']: g['design_audit'] for g in surface_obj['baseline_groups']}
inference_plans = {g['baseline_group_id']: g['inference_plan'] for g in surface_obj['baseline_groups']}

results = []
inference_results = []
spec_run_counter = 0


def run_spec(spec_id, spec_tree_path, baseline_group_id, outcome_var, treatment_var,
             controls, fe_list, fe_formula, data, vcov,
             sample_desc, controls_desc, cluster_var="roomnum_base",
             axis_block_name=None, axis_block=None, notes="",
             outcome_var_actual=None):
    """Run a single specification and record results."""
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    # outcome_var_actual is the actual column name in data (e.g., 'continue_var')
    # outcome_var is what we report (e.g., 'continue')
    y_col = outcome_var_actual or outcome_var

    try:
        controls_str = " + ".join(controls) if controls else ""
        if controls_str and fe_formula:
            formula = f"{y_col} ~ {treatment_var} + {controls_str} | {fe_formula}"
        elif controls_str:
            formula = f"{y_col} ~ {treatment_var} + {controls_str}"
        elif fe_formula:
            formula = f"{y_col} ~ {treatment_var} | {fe_formula}"
        else:
            formula = f"{y_col} ~ {treatment_var}"

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

        canonical = inference_plans[baseline_group_id]["canonical"]
        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": canonical["spec_id"],
                       "params": canonical["params"]},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"randomized_experiment": design_audits[baseline_group_id]},
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
            "fixed_effects": ", ".join(fe_list) if fe_list else "none",
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
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
            "fixed_effects": ", ".join(fe_list) if fe_list else "none",
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
            "run_success": 0,
            "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


def run_inference_variant(base_run_id, spec_id, spec_tree_path, baseline_group_id,
                          outcome_var, treatment_var, controls, fe_formula,
                          data, vcov_variant, cluster_var_variant,
                          outcome_var_actual=None):
    """Run an inference variant for a given base spec."""
    global spec_run_counter
    spec_run_counter += 1
    infer_run_id = f"{PAPER_ID}_infer_{spec_run_counter:03d}"

    y_col = outcome_var_actual or outcome_var

    try:
        controls_str = " + ".join(controls) if controls else ""
        if controls_str and fe_formula:
            formula = f"{y_col} ~ {treatment_var} + {controls_str} | {fe_formula}"
        elif controls_str:
            formula = f"{y_col} ~ {treatment_var} + {controls_str}"
        elif fe_formula:
            formula = f"{y_col} ~ {treatment_var} | {fe_formula}"
        else:
            formula = f"{y_col} ~ {treatment_var}"

        m = pf.feols(formula, data=data, vcov=vcov_variant)

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
            inference={"spec_id": spec_id, "cluster_var": cluster_var_variant},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"randomized_experiment": design_audits[baseline_group_id]},
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
            "coefficient": coef_val,
            "std_error": se_val,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": nobs,
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "cluster_var": cluster_var_variant,
            "run_success": 1,
            "run_error": ""
        })
        return infer_run_id

    except Exception as e:
        err_msg = str(e)[:240]
        err_details = error_details_from_exception(e, stage="inference_variant")
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
            "cluster_var": cluster_var_variant,
            "run_success": 0,
            "run_error": err_msg
        })
        return infer_run_id


# ============================================================
# PREPARE SUBSAMPLES
# ============================================================
df_white = df[df['white'] == 1].copy()
df_black = df[df['black'] == 1].copy()
df_wb = df[(df['white'] == 1) | (df['black'] == 1)].copy()

# Canonical vcov
vcov_cluster_room = {"CRV1": "roomnum_base"}

# ============================================================
# G1: RACE IAT (DscoreraceIAT) ~ mixracebas
# ============================================================
print("=" * 60)
print("G1: Race IAT specifications")
print("=" * 60)

# G1 controls: own + roommate + lagged DscoreraceIAT
g1_controls = ['L_DscoreraceIAT'] + controls_subsample + rocontrols
g1_controls_own_only = ['L_DscoreraceIAT'] + controls_subsample

# ----- G1 Baseline: Table3-Col2-Black-RaceIAT -----
# (Primary baseline for G1)
run_id_g1_base, *_ = run_spec(
    "baseline__table3_col2_black_raceiat",
    "specification_tree/designs/randomized_experiment.md#baseline",
    "G1", "DscoreraceIAT", "mixracebas",
    g1_controls, ["Res_base"], "Res_base_str",
    df_black, vcov_cluster_room,
    "round==Follow-up & black==1", "L.DscoreraceIAT + own controls + roommate controls"
)

# ----- G1 Design variants -----

# Diff in means (no controls, no FE)
run_spec(
    "design/randomized_experiment/estimator/diff_in_means",
    "specification_tree/designs/randomized_experiment.md#diff-in-means",
    "G1", "DscoreraceIAT", "mixracebas",
    [], [], "",
    df_black, vcov_cluster_room,
    "round==Follow-up & black==1", "none (difference in means)"
)

# ANCOVA (with lagged outcome only)
run_spec(
    "design/randomized_experiment/estimator/ancova",
    "specification_tree/designs/randomized_experiment.md#ancova",
    "G1", "DscoreraceIAT", "mixracebas",
    ['L_DscoreraceIAT'], [], "",
    df_black, vcov_cluster_room,
    "round==Follow-up & black==1", "ANCOVA: L.DscoreraceIAT only"
)

# With covariates (controls but no FE)
run_spec(
    "design/randomized_experiment/estimator/with_covariates",
    "specification_tree/designs/randomized_experiment.md#with-covariates",
    "G1", "DscoreraceIAT", "mixracebas",
    g1_controls, [], "",
    df_black, vcov_cluster_room,
    "round==Follow-up & black==1", "all controls, no FE"
)

# Strata FE only (FE + lagged outcome only)
run_spec(
    "design/randomized_experiment/estimator/strata_fe",
    "specification_tree/designs/randomized_experiment.md#strata-fe",
    "G1", "DscoreraceIAT", "mixracebas",
    ['L_DscoreraceIAT'], ["Res_base"], "Res_base_str",
    df_black, vcov_cluster_room,
    "round==Follow-up & black==1", "strata FE + ANCOVA (lagged outcome)"
)

# ----- G1 RC: Drop roommate controls -----
run_spec(
    "rc/controls/drop_roommate_controls",
    "specification_tree/modules/robustness/controls.md#standard-control-sets",
    "G1", "DscoreraceIAT", "mixracebas",
    g1_controls_own_only, ["Res_base"], "Res_base_str",
    df_black, vcov_cluster_room,
    "round==Follow-up & black==1", "own controls + L.DscoreraceIAT only (no roommate controls)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/drop_roommate_controls",
                "family": "sets", "set_name": "own_controls_only",
                "dropped": rocontrols, "added": [],
                "n_controls": len(g1_controls_own_only)}
)

# ----- G1 RC: LOO own controls -----
for var_key, var_group in loo_own_groups.items():
    ctrl = [c for c in g1_controls if c not in var_group]
    run_spec(
        f"rc/controls/loo/{var_key}",
        "specification_tree/modules/robustness/controls.md#leave-one-out-controls-loo",
        "G1", "DscoreraceIAT", "mixracebas",
        ctrl, ["Res_base"], "Res_base_str",
        df_black, vcov_cluster_room,
        "round==Follow-up & black==1", f"baseline minus {var_key}",
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/controls/loo/{var_key}", "family": "loo",
                    "dropped": var_group, "added": [], "n_controls": len(ctrl)}
    )

# ----- G1 RC: LOO roommate controls -----
for var_key, var_group in loo_ro_groups.items():
    ctrl = [c for c in g1_controls if c not in var_group]
    run_spec(
        f"rc/controls/loo/{var_key}",
        "specification_tree/modules/robustness/controls.md#leave-one-out-controls-loo",
        "G1", "DscoreraceIAT", "mixracebas",
        ctrl, ["Res_base"], "Res_base_str",
        df_black, vcov_cluster_room,
        "round==Follow-up & black==1", f"baseline minus {var_key}",
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/controls/loo/{var_key}", "family": "loo",
                    "dropped": var_group, "added": [], "n_controls": len(ctrl)}
    )

# ----- G1 RC: Full sample with race FE -----
g1_controls_full = ['L_DscoreraceIAT'] + controls_full + rocontrols
run_spec(
    "rc/sample/full_sample_with_race_fe",
    "specification_tree/modules/robustness/sample.md#sample-restrictions",
    "G1", "DscoreraceIAT", "mixracebas",
    g1_controls_full,
    ["Res_base", "blackRes", "whiteRes"],
    "Res_base_str + blackRes_str + whiteRes_str",
    df, vcov_cluster_room,
    "round==Follow-up, full sample with race x residence FE",
    "L.DscoreraceIAT + full controls + roommate controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/full_sample_with_race_fe",
                "population": "full_sample", "n_obs_before": len(df_black),
                "n_obs_after": len(df)}
)

# ----- G1 RC: Restrict to white and black only -----
g1_controls_wb = ['L_DscoreraceIAT'] + controls_full + rocontrols
run_spec(
    "rc/sample/restrict_white_black_only",
    "specification_tree/modules/robustness/sample.md#sample-restrictions",
    "G1", "DscoreraceIAT", "mixracebas",
    g1_controls_wb,
    ["Res_base", "blackRes", "whiteRes"],
    "Res_base_str + blackRes_str + whiteRes_str",
    df_wb, vcov_cluster_room,
    "round==Follow-up, white + black only",
    "L.DscoreraceIAT + full controls + roommate controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restrict_white_black_only",
                "population": "white_black_only", "n_obs_before": len(df),
                "n_obs_after": len(df_wb)}
)

print(f"  G1 specs so far: {spec_run_counter}")


# ============================================================
# G2: ACADEMIC OUTCOMES ~ mixracebas (Black subsample primary)
# ============================================================
print("=" * 60)
print("G2: Academic outcomes specifications")
print("=" * 60)

g2_controls = controls_subsample + rocontrols
g2_controls_own_only = controls_subsample
g2_fe = ["Res_base", "regprogram"]
g2_fe_formula = "Res_base_str + regprogram_str"

# G2 outcomes (reported name -> actual column name)
g2_outcomes = [
    ("GPA", "GPA"),
    ("examspassed", "examspassed"),
    ("continue", "continue_var"),
    ("PCAperf", "PCAperf"),
]

# Baselines are: examspassed, continue, PCAperf (GPA is in the baseline_specs but not baseline_spec_ids)
g2_baseline_ids = {
    "examspassed": "baseline__table4_black_examspassed",
    "continue": "baseline__table4_black_continue",
    "PCAperf": "baseline__table4_black_pcaperf",
}

g2_baseline_run_ids = {}

# Run baseline specs only for outcomes in baseline_spec_ids
for outcome_name, outcome_col in g2_outcomes:
    if outcome_name in g2_baseline_ids:
        sid = g2_baseline_ids[outcome_name]
        spath = "specification_tree/designs/randomized_experiment.md#baseline"
        run_id, *_ = run_spec(
            sid, spath, "G2",
            outcome_name, "mixracebas",
            g2_controls, g2_fe, g2_fe_formula,
            df_black, vcov_cluster_room,
            "round==Follow-up & black==1",
            "own controls + roommate controls",
            outcome_var_actual=outcome_col
        )
        g2_baseline_run_ids[outcome_name] = run_id

# Design variants for all outcomes (including GPA which is from baseline_specs but not baseline_spec_ids)
for outcome_name, outcome_col in g2_outcomes:
    # Diff in means
    run_spec(
        "design/randomized_experiment/estimator/diff_in_means",
        "specification_tree/designs/randomized_experiment.md#diff-in-means",
        "G2", outcome_name, "mixracebas",
        [], [], "",
        df_black, vcov_cluster_room,
        "round==Follow-up & black==1",
        "none (difference in means)",
        outcome_var_actual=outcome_col
    )

    # With covariates (controls, no FE)
    run_spec(
        "design/randomized_experiment/estimator/with_covariates",
        "specification_tree/designs/randomized_experiment.md#with-covariates",
        "G2", outcome_name, "mixracebas",
        g2_controls, [], "",
        df_black, vcov_cluster_room,
        "round==Follow-up & black==1",
        "all controls, no FE",
        outcome_var_actual=outcome_col
    )

    # Strata FE only (FE only, no other controls)
    run_spec(
        "design/randomized_experiment/estimator/strata_fe",
        "specification_tree/designs/randomized_experiment.md#strata-fe",
        "G2", outcome_name, "mixracebas",
        [], g2_fe, g2_fe_formula,
        df_black, vcov_cluster_room,
        "round==Follow-up & black==1",
        "strata FE + program FE only",
        outcome_var_actual=outcome_col
    )

# ----- G2 RC specs (applied to primary outcome GPA for LOO, all for set-level) -----

# Drop roommate controls (all outcomes)
for outcome_name, outcome_col in g2_outcomes:
    run_spec(
        "rc/controls/drop_roommate_controls",
        "specification_tree/modules/robustness/controls.md#standard-control-sets",
        "G2", outcome_name, "mixracebas",
        g2_controls_own_only, g2_fe, g2_fe_formula,
        df_black, vcov_cluster_room,
        "round==Follow-up & black==1",
        "own controls only (no roommate controls)",
        outcome_var_actual=outcome_col,
        axis_block_name="controls",
        axis_block={"spec_id": "rc/controls/drop_roommate_controls",
                    "family": "sets", "set_name": "own_controls_only",
                    "dropped": rocontrols, "added": [],
                    "n_controls": len(g2_controls_own_only)}
    )

# LOO own controls (for GPA as representative outcome)
for var_key, var_group in loo_own_groups.items():
    ctrl = [c for c in g2_controls if c not in var_group]
    run_spec(
        f"rc/controls/loo/{var_key}",
        "specification_tree/modules/robustness/controls.md#leave-one-out-controls-loo",
        "G2", "GPA", "mixracebas",
        ctrl, g2_fe, g2_fe_formula,
        df_black, vcov_cluster_room,
        "round==Follow-up & black==1",
        f"baseline minus {var_key}",
        outcome_var_actual="GPA",
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/controls/loo/{var_key}", "family": "loo",
                    "dropped": var_group, "added": [], "n_controls": len(ctrl)}
    )

# LOO roommate controls (for GPA)
for var_key, var_group in loo_ro_groups.items():
    ctrl = [c for c in g2_controls if c not in var_group]
    run_spec(
        f"rc/controls/loo/{var_key}",
        "specification_tree/modules/robustness/controls.md#leave-one-out-controls-loo",
        "G2", "GPA", "mixracebas",
        ctrl, g2_fe, g2_fe_formula,
        df_black, vcov_cluster_room,
        "round==Follow-up & black==1",
        f"baseline minus {var_key}",
        outcome_var_actual="GPA",
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/controls/loo/{var_key}", "family": "loo",
                    "dropped": var_group, "added": [], "n_controls": len(ctrl)}
    )

# Full sample with race FE
g2_controls_full = controls_full + rocontrols
for outcome_name, outcome_col in g2_outcomes:
    run_spec(
        "rc/sample/full_sample_with_race_fe",
        "specification_tree/modules/robustness/sample.md#sample-restrictions",
        "G2", outcome_name, "mixracebas",
        g2_controls_full,
        ["Res_base", "regprogram", "blackRes", "whiteRes"],
        "Res_base_str + regprogram_str + blackRes_str + whiteRes_str",
        df, vcov_cluster_room,
        "round==Follow-up, full sample with race x residence FE",
        "full controls + roommate controls",
        outcome_var_actual=outcome_col,
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/full_sample_with_race_fe",
                    "population": "full_sample"}
    )

# Restrict to white + black only
for outcome_name, outcome_col in g2_outcomes:
    run_spec(
        "rc/sample/restrict_white_black_only",
        "specification_tree/modules/robustness/sample.md#sample-restrictions",
        "G2", outcome_name, "mixracebas",
        g2_controls_full,
        ["Res_base", "regprogram", "blackRes", "whiteRes"],
        "Res_base_str + regprogram_str + blackRes_str + whiteRes_str",
        df_wb, vcov_cluster_room,
        "round==Follow-up, white + black only",
        "full controls + roommate controls",
        outcome_var_actual=outcome_col,
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/restrict_white_black_only",
                    "population": "white_black_only"}
    )

# Drop program FE
for outcome_name, outcome_col in g2_outcomes:
    run_spec(
        "rc/fe/drop_program_fe",
        "specification_tree/modules/robustness/fixed_effects.md#dropping-fe-relative-to-baseline",
        "G2", outcome_name, "mixracebas",
        g2_controls, ["Res_base"], "Res_base_str",
        df_black, vcov_cluster_room,
        "round==Follow-up & black==1",
        "own + roommate controls, Res_base FE only (no program FE)",
        outcome_var_actual=outcome_col,
        axis_block_name="fixed_effects",
        axis_block={"spec_id": "rc/fe/drop_program_fe", "family": "drop",
                    "dropped": ["regprogram"], "added": [],
                    "baseline_fe": ["Res_base", "regprogram"], "new_fe": ["Res_base"]}
    )

# PCA performance index (functional form variant, GPA outcome -> PCAperf)
# Already have PCAperf as separate baseline, but also run as rc/form on GPA baseline
run_spec(
    "rc/form/pca_performance_index",
    "specification_tree/modules/robustness/functional_form.md#outcome-transformations",
    "G2", "PCAperf", "mixracebas",
    g2_controls, g2_fe, g2_fe_formula,
    df_black, vcov_cluster_room,
    "round==Follow-up & black==1",
    "own + roommate controls (PCA index of GPA, examspassed, continue)",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/pca_performance_index",
                "outcome_transform": "pca_index",
                "interpretation": "PCA index of GPA, examspassed, continue. Summarizes all academic outcomes into single index."}
)

# Second year outcomes (rc/form)
g2_second_year = [
    ("GPA2013", "GPA2013"),
    ("examspassed2013", "examspassed2013"),
    ("continue2013", "continue2013"),
    ("PCAperf2013", "PCAperf2013"),
]

for outcome_name, outcome_col in g2_second_year:
    run_spec(
        "rc/form/second_year_outcomes",
        "specification_tree/modules/robustness/functional_form.md#outcome-transformations",
        "G2", outcome_name, "mixracebas",
        g2_controls, g2_fe, g2_fe_formula,
        df_black, vcov_cluster_room,
        "round==Follow-up & black==1",
        "own + roommate controls, second year outcome",
        outcome_var_actual=outcome_col,
        axis_block_name="functional_form",
        axis_block={"spec_id": "rc/form/second_year_outcomes",
                    "outcome_transform": "second_year",
                    "interpretation": f"Second year ({outcome_name}) instead of first year. Tests persistence of treatment effect."}
    )

print(f"  G2 specs so far: {spec_run_counter}")


# ============================================================
# G3: SOCIAL OUTCOMES ~ mixracebas (White subsample primary)
# ============================================================
print("=" * 60)
print("G3: Social outcomes specifications")
print("=" * 60)

g3_controls = controls_subsample + rocontrols
g3_controls_own_only = controls_subsample
g3_fe = ["Res_base"]
g3_fe_formula = "Res_base_str"

g3_outcomes = [
    ("PCAfriend", "PCAfriend"),
    ("PCAattitude", "PCAattitude"),
    ("PCAcomm", "PCAcomm"),
    ("PCAsocial", "PCAsocial"),
]

g3_baseline_ids = {
    "PCAattitude": "baseline__table5_white_pcaattitude",
    "PCAcomm": "baseline__table5_white_pcacomm",
    "PCAsocial": "baseline__table5_white_pcasocial",
}

g3_baseline_run_ids = {}

# Run baseline specs only for outcomes in baseline_spec_ids
for outcome_name, outcome_col in g3_outcomes:
    if outcome_name in g3_baseline_ids:
        sid = g3_baseline_ids[outcome_name]
        spath = "specification_tree/designs/randomized_experiment.md#baseline"
        run_id, *_ = run_spec(
            sid, spath, "G3",
            outcome_name, "mixracebas",
            g3_controls, g3_fe, g3_fe_formula,
            df_white, vcov_cluster_room,
            "round==Follow-up & white==1",
            "own controls + roommate controls",
            outcome_var_actual=outcome_col
        )
        g3_baseline_run_ids[outcome_name] = run_id

# Design variants for all outcomes (including PCAfriend which is from baseline_specs but not baseline_spec_ids)
for outcome_name, outcome_col in g3_outcomes:
    # Diff in means
    run_spec(
        "design/randomized_experiment/estimator/diff_in_means",
        "specification_tree/designs/randomized_experiment.md#diff-in-means",
        "G3", outcome_name, "mixracebas",
        [], [], "",
        df_white, vcov_cluster_room,
        "round==Follow-up & white==1",
        "none (difference in means)",
        outcome_var_actual=outcome_col
    )

    # With covariates (controls, no FE)
    run_spec(
        "design/randomized_experiment/estimator/with_covariates",
        "specification_tree/designs/randomized_experiment.md#with-covariates",
        "G3", outcome_name, "mixracebas",
        g3_controls, [], "",
        df_white, vcov_cluster_room,
        "round==Follow-up & white==1",
        "all controls, no FE",
        outcome_var_actual=outcome_col
    )

    # Strata FE only
    run_spec(
        "design/randomized_experiment/estimator/strata_fe",
        "specification_tree/designs/randomized_experiment.md#strata-fe",
        "G3", outcome_name, "mixracebas",
        [], g3_fe, g3_fe_formula,
        df_white, vcov_cluster_room,
        "round==Follow-up & white==1",
        "strata FE only",
        outcome_var_actual=outcome_col
    )

# ----- G3 RC: Drop roommate controls -----
for outcome_name, outcome_col in g3_outcomes:
    run_spec(
        "rc/controls/drop_roommate_controls",
        "specification_tree/modules/robustness/controls.md#standard-control-sets",
        "G3", outcome_name, "mixracebas",
        g3_controls_own_only, g3_fe, g3_fe_formula,
        df_white, vcov_cluster_room,
        "round==Follow-up & white==1",
        "own controls only (no roommate controls)",
        outcome_var_actual=outcome_col,
        axis_block_name="controls",
        axis_block={"spec_id": "rc/controls/drop_roommate_controls",
                    "family": "sets", "set_name": "own_controls_only",
                    "dropped": rocontrols, "added": [],
                    "n_controls": len(g3_controls_own_only)}
    )

# LOO own controls (for PCAfriend as representative)
for var_key, var_group in loo_own_groups.items():
    ctrl = [c for c in g3_controls if c not in var_group]
    run_spec(
        f"rc/controls/loo/{var_key}",
        "specification_tree/modules/robustness/controls.md#leave-one-out-controls-loo",
        "G3", "PCAfriend", "mixracebas",
        ctrl, g3_fe, g3_fe_formula,
        df_white, vcov_cluster_room,
        "round==Follow-up & white==1",
        f"baseline minus {var_key}",
        outcome_var_actual="PCAfriend",
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/controls/loo/{var_key}", "family": "loo",
                    "dropped": var_group, "added": [], "n_controls": len(ctrl)}
    )

# LOO roommate controls (for PCAfriend)
for var_key, var_group in loo_ro_groups.items():
    ctrl = [c for c in g3_controls if c not in var_group]
    run_spec(
        f"rc/controls/loo/{var_key}",
        "specification_tree/modules/robustness/controls.md#leave-one-out-controls-loo",
        "G3", "PCAfriend", "mixracebas",
        ctrl, g3_fe, g3_fe_formula,
        df_white, vcov_cluster_room,
        "round==Follow-up & white==1",
        f"baseline minus {var_key}",
        outcome_var_actual="PCAfriend",
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/controls/loo/{var_key}", "family": "loo",
                    "dropped": var_group, "added": [], "n_controls": len(ctrl)}
    )

# Full sample with race FE
g3_controls_full = controls_full + rocontrols
for outcome_name, outcome_col in g3_outcomes:
    run_spec(
        "rc/sample/full_sample_with_race_fe",
        "specification_tree/modules/robustness/sample.md#sample-restrictions",
        "G3", outcome_name, "mixracebas",
        g3_controls_full,
        ["Res_base", "blackRes", "whiteRes"],
        "Res_base_str + blackRes_str + whiteRes_str",
        df, vcov_cluster_room,
        "round==Follow-up, full sample with race x residence FE",
        "full controls + roommate controls",
        outcome_var_actual=outcome_col,
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/full_sample_with_race_fe",
                    "population": "full_sample"}
    )

# Restrict to black only
for outcome_name, outcome_col in g3_outcomes:
    run_spec(
        "rc/sample/restrict_black_only",
        "specification_tree/modules/robustness/sample.md#sample-restrictions",
        "G3", outcome_name, "mixracebas",
        g3_controls, g3_fe, g3_fe_formula,
        df_black, vcov_cluster_room,
        "round==Follow-up & black==1",
        "own controls + roommate controls, black subsample",
        outcome_var_actual=outcome_col,
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/restrict_black_only",
                    "population": "black_only"}
    )

# No-missing PCA indices (Table A15)
# Data uses *_nomiss1 suffix
nomiss_map = {
    "PCAfriend": "PCAfriend_nomiss1",
    "PCAattitude": "PCAatt_nomiss1",
    "PCAcomm": "PCAcomm_nomiss1",
    "PCAsocial": "PCAsocial_nomiss1",
}

for outcome_name, nomiss_col in nomiss_map.items():
    run_spec(
        "rc/form/nomiss_pca_indices",
        "specification_tree/modules/robustness/functional_form.md#outcome-transformations",
        "G3", f"{outcome_name}_nomiss", "mixracebas",
        g3_controls, g3_fe, g3_fe_formula,
        df_white, vcov_cluster_room,
        "round==Follow-up & white==1",
        f"own + roommate controls, {outcome_name} computed without missing imputation",
        outcome_var_actual=nomiss_col,
        axis_block_name="functional_form",
        axis_block={"spec_id": "rc/form/nomiss_pca_indices",
                    "outcome_transform": "nomiss_pca",
                    "interpretation": f"PCA index ({outcome_name}) recomputed excluding observations with missing component variables."}
    )

print(f"  G3 specs so far: {spec_run_counter}")


# ============================================================
# INFERENCE VARIANTS (across all baseline groups)
# ============================================================
print("=" * 60)
print("Inference variants")
print("=" * 60)

# G1 baseline inference variants
g1_base_run_id = f"{PAPER_ID}_run_001"
# HC1 robust
run_inference_variant(
    g1_base_run_id, "infer/se/hc/hc1",
    "specification_tree/modules/inference/standard_errors.md#heteroskedasticity-robust",
    "G1", "DscoreraceIAT", "mixracebas",
    g1_controls, "Res_base_str",
    df_black, "hetero", ""
)
# Cluster at residence
run_inference_variant(
    g1_base_run_id, "infer/se/cluster/residence",
    "specification_tree/modules/inference/standard_errors.md#single-level-clustering",
    "G1", "DscoreraceIAT", "mixracebas",
    g1_controls, "Res_base_str",
    df_black, {"CRV1": "Res_base_str"}, "Res_base"
)

# G2 baseline inference variants (for GPA)
g2_exam_run_id = g2_baseline_run_ids.get("examspassed", f"{PAPER_ID}_run_021")
run_inference_variant(
    g2_exam_run_id, "infer/se/hc/hc1",
    "specification_tree/modules/inference/standard_errors.md#heteroskedasticity-robust",
    "G2", "examspassed", "mixracebas",
    g2_controls, "Res_base_str + regprogram_str",
    df_black, "hetero", ""
)
run_inference_variant(
    g2_exam_run_id, "infer/se/cluster/residence",
    "specification_tree/modules/inference/standard_errors.md#single-level-clustering",
    "G2", "examspassed", "mixracebas",
    g2_controls, "Res_base_str + regprogram_str",
    df_black, {"CRV1": "Res_base_str"}, "Res_base"
)

# G3 baseline inference variants (for PCAfriend)
g3_att_run_id = g3_baseline_run_ids.get("PCAattitude", f"{PAPER_ID}_run_060")
run_inference_variant(
    g3_att_run_id, "infer/se/hc/hc1",
    "specification_tree/modules/inference/standard_errors.md#heteroskedasticity-robust",
    "G3", "PCAattitude", "mixracebas",
    g3_controls, "Res_base_str",
    df_white, "hetero", "",
    outcome_var_actual="PCAattitude"
)
run_inference_variant(
    g3_att_run_id, "infer/se/cluster/residence",
    "specification_tree/modules/inference/standard_errors.md#single-level-clustering",
    "G3", "PCAattitude", "mixracebas",
    g3_controls, "Res_base_str",
    df_white, {"CRV1": "Res_base_str"}, "Res_base",
    outcome_var_actual="PCAattitude"
)

print(f"  Total inference variants: {len(inference_results)}")


# ============================================================
# WRITE OUTPUTS
# ============================================================
print("=" * 60)
print("Writing outputs")
print("=" * 60)

spec_df = pd.DataFrame(results)
spec_df.to_csv(f"{DATA_DIR}/specification_results.csv", index=False)
print(f"  Wrote {len(results)} specification results to specification_results.csv")

n_success = spec_df['run_success'].sum()
n_fail = len(spec_df) - n_success
print(f"  Success: {n_success}, Failed: {n_fail}")

if inference_results:
    infer_df = pd.DataFrame(inference_results)
    infer_df.to_csv(f"{DATA_DIR}/inference_results.csv", index=False)
    print(f"  Wrote {len(inference_results)} inference results to inference_results.csv")

# Summary by group
for gid in ["G1", "G2", "G3"]:
    g_specs = spec_df[spec_df['baseline_group_id'] == gid]
    print(f"  {gid}: {len(g_specs)} specs, {g_specs['run_success'].sum()} success")

print(f"\nTotal specification_results.csv rows: {len(spec_df)}")
print(f"Total inference_results.csv rows: {len(inference_results)}")
print("Done!")
