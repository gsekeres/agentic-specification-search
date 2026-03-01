#!/usr/bin/env python3
"""
Specification search script for 149262-V2:
Mixed-ability seating field experiment in Chinese elementary schools.

Surface-driven execution of ~58 core specifications across two baseline groups:
  G1: Academic performance, lower track (hsco==0)
  G2: Academic performance, upper track (hsco==1)

Outputs:
  - specification_results.csv (baseline, design/*, rc/* rows)
  - inference_results.csv (infer/* rows)
  - SPECIFICATION_SEARCH.md
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import hashlib
import sys
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PATHS
# ============================================================
BASE = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search"
PACKAGE_DIR = f"{BASE}/data/downloads/extracted/149262-V2"
DATA_DIR = f"{PACKAGE_DIR}/Stata-program"
PAPER_ID = "149262-V2"

# ============================================================
# LOAD SURFACE
# ============================================================
with open(f"{PACKAGE_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface = json.load(f)

s_hash = "sha256:" + hashlib.sha256(
    json.dumps(surface, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
).hexdigest()

# ============================================================
# SOFTWARE BLOCK
# ============================================================
sw_block = {
    "runner_language": "python",
    "runner_version": sys.version.split()[0],
    "packages": {
        "pyfixest": pf.__version__ if hasattr(pf, '__version__') else "0.40+",
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
}

# ============================================================
# DESIGN AUDIT (shared by both baseline groups)
# ============================================================
bg_g1 = surface["baseline_groups"][0]
bg_g2 = surface["baseline_groups"][1]
design_audit = bg_g1["design_audit"]
design_block = {"randomized_experiment": design_audit}

# ============================================================
# CANONICAL INFERENCE
# ============================================================
canonical_inference = {
    "spec_id": "infer/se/cluster/class1",
    "params": {"cluster_var": "class1"}
}

# ============================================================
# LOAD DATA
# ============================================================
df_full = pd.read_stata(f"{DATA_DIR}/final.dta", convert_categoricals=False)

# Ensure numeric types
for col in df_full.columns:
    if df_full[col].dtype == np.float32:
        df_full[col] = df_full[col].astype(np.float64)

# Create pooled treatment variable
df_full['any_treatment'] = ((df_full['treat1'] == 1) | (df_full['treat2'] == 1)).astype(float)

# Ensure FE/cluster vars are int
df_full['grade1'] = df_full['grade1'].astype(int)
df_full['class1'] = df_full['class1'].astype(int)
df_full['school'] = df_full['school'].astype(int)

# Track subsets
df_lower = df_full[df_full['hsco'] == 0].copy()
df_upper = df_full[df_full['hsco'] == 1].copy()

# ============================================================
# CONTROL VARIABLE LISTS
# ============================================================
ALL_CONTROLS = ["gender", "age", "height", "hukou_rd1", "nationality_rd1",
                "health", "sib", "fa_eduy", "mo_eduy", "pc", "car"]
DEMOGRAPHICS = ["gender", "age", "height", "health"]
FAMILY_SES = ["fa_eduy", "mo_eduy", "pc", "car", "sib"]

# ============================================================
# RESULT STORAGE
# ============================================================
spec_results = []
inference_results = []
run_counter = 0
infer_counter = 0


def next_run_id():
    global run_counter
    run_counter += 1
    return f"{PAPER_ID}_spec_{run_counter:04d}"


def next_infer_id():
    global infer_counter
    infer_counter += 1
    return f"{PAPER_ID}_infer_{infer_counter:04d}"


def run_regression(outcome_var, treatment_var, baseline_outcome_var, controls,
                   fixed_effects, data, vcov, additional_treatment_var=None,
                   focal_var=None):
    """Run a pyfixest regression and return coefficient info."""
    if focal_var is None:
        focal_var = treatment_var

    rhs_vars = []
    if baseline_outcome_var and baseline_outcome_var in data.columns:
        rhs_vars.append(baseline_outcome_var)
    rhs_vars.append(treatment_var)
    if additional_treatment_var and additional_treatment_var in data.columns:
        rhs_vars.append(additional_treatment_var)
    rhs_vars.extend([c for c in controls if c in data.columns])

    rhs = " + ".join(rhs_vars)

    if fixed_effects:
        fe_str = " + ".join(fixed_effects)
        formula = f"{outcome_var} ~ {rhs} | {fe_str}"
    else:
        formula = f"{outcome_var} ~ {rhs}"

    if isinstance(vcov, dict):
        vcov_arg = vcov
    elif vcov == "hetero":
        vcov_arg = "hetero"
    else:
        vcov_arg = {"CRV1": vcov}

    all_vars = [outcome_var] + rhs_vars
    if fixed_effects:
        all_vars += fixed_effects
    if isinstance(vcov, str) and vcov not in ("hetero",):
        all_vars.append(vcov)
    elif isinstance(vcov, dict):
        for v in vcov.values():
            if isinstance(v, str):
                all_vars.append(v)

    use_data = data.dropna(subset=[v for v in all_vars if v in data.columns]).copy()

    model = pf.feols(formula, data=use_data, vcov=vcov_arg)

    coef = float(model.coef().get(focal_var, np.nan))
    se = float(model.se().get(focal_var, np.nan))
    pval = float(model.pvalue().get(focal_var, np.nan))
    ci = model.confint()
    ci_lo = float(ci.loc[focal_var, ci.columns[0]]) if focal_var in ci.index else np.nan
    ci_hi = float(ci.loc[focal_var, ci.columns[1]]) if focal_var in ci.index else np.nan
    nobs = int(model._N)
    r2 = float(model._r2)

    coefficients = {k: float(v) for k, v in model.coef().items()}

    return {
        "coefficient": coef,
        "std_error": se,
        "p_value": pval,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "n_obs": nobs,
        "r_squared": r2,
        "coefficients": coefficients,
        "formula": formula,
    }


def make_payload(coefficients, extra_blocks=None):
    """Make a contract-compliant coefficient_vector_json."""
    payload = {
        "coefficients": coefficients,
        "inference": canonical_inference,
        "software": sw_block,
        "surface_hash": s_hash,
        "design": design_block,
    }
    if extra_blocks:
        payload.update(extra_blocks)
    return payload


def add_result(spec_id, spec_tree_path, baseline_group_id, outcome_var, treatment_var,
               result, sample_desc, fixed_effects_desc, controls_desc, cluster_var,
               extra_payload_blocks=None):
    """Add a successful result row."""
    run_id = next_run_id()
    payload = make_payload(result["coefficients"], extra_payload_blocks)

    spec_results.append({
        "paper_id": PAPER_ID,
        "spec_run_id": run_id,
        "spec_id": spec_id,
        "spec_tree_path": spec_tree_path,
        "baseline_group_id": baseline_group_id,
        "outcome_var": outcome_var,
        "treatment_var": treatment_var,
        "coefficient": result["coefficient"],
        "std_error": result["std_error"],
        "p_value": result["p_value"],
        "ci_lower": result["ci_lower"],
        "ci_upper": result["ci_upper"],
        "n_obs": result["n_obs"],
        "r_squared": result["r_squared"],
        "coefficient_vector_json": json.dumps(payload),
        "sample_desc": sample_desc,
        "fixed_effects": fixed_effects_desc,
        "controls_desc": controls_desc,
        "cluster_var": cluster_var,
        "run_success": 1,
        "run_error": "",
    })
    return run_id


def add_failure(spec_id, spec_tree_path, baseline_group_id, outcome_var, treatment_var,
                error_msg, stage, sample_desc, fixed_effects_desc, controls_desc, cluster_var):
    """Add a failed result row."""
    run_id = next_run_id()
    payload = {
        "error": error_msg,
        "error_details": {
            "stage": stage,
            "exception_type": "RuntimeError",
            "exception_message": error_msg,
        }
    }
    spec_results.append({
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
        "fixed_effects": fixed_effects_desc,
        "controls_desc": controls_desc,
        "cluster_var": cluster_var,
        "run_success": 0,
        "run_error": error_msg,
    })
    return run_id


def safe_run(spec_id, spec_tree_path, baseline_group_id, outcome_var, treatment_var,
             baseline_outcome_var, controls, fixed_effects, data, vcov_str,
             sample_desc, fe_desc, controls_desc, cluster_var,
             additional_treatment_var=None, focal_var=None,
             extra_payload_blocks=None):
    """Safely run a regression and record result or failure."""
    try:
        if isinstance(vcov_str, str) and vcov_str.startswith("cluster("):
            cluster = vcov_str.replace("cluster(", "").replace(")", "")
            vcov = {"CRV1": cluster}
        elif vcov_str == "hetero":
            vcov = "hetero"
        else:
            vcov = {"CRV1": vcov_str}

        result = run_regression(
            outcome_var=outcome_var,
            treatment_var=treatment_var,
            baseline_outcome_var=baseline_outcome_var,
            controls=controls,
            fixed_effects=fixed_effects,
            data=data,
            vcov=vcov,
            additional_treatment_var=additional_treatment_var,
            focal_var=focal_var,
        )
        run_id = add_result(
            spec_id=spec_id,
            spec_tree_path=spec_tree_path,
            baseline_group_id=baseline_group_id,
            outcome_var=outcome_var,
            treatment_var=treatment_var,
            result=result,
            sample_desc=sample_desc,
            fixed_effects_desc=fe_desc,
            controls_desc=controls_desc,
            cluster_var=cluster_var,
            extra_payload_blocks=extra_payload_blocks,
        )
        return run_id, result
    except Exception as e:
        error_msg = str(e)[:240]
        run_id = add_failure(
            spec_id=spec_id,
            spec_tree_path=spec_tree_path,
            baseline_group_id=baseline_group_id,
            outcome_var=outcome_var,
            treatment_var=treatment_var,
            error_msg=error_msg,
            stage="estimation",
            sample_desc=sample_desc,
            fixed_effects_desc=fe_desc,
            controls_desc=controls_desc,
            cluster_var=cluster_var,
        )
        return run_id, None


def run_group_specs(group_id, data, sample_desc, track_label):
    """Run the full set of specs for one baseline group (lower or upper track)."""
    baseline_run_ids = {}

    # --- Primary baseline: Table3, full controls ---
    print(f"\n--- Baseline: {track_label} ave3 full controls ---")
    rid, res = safe_run(
        "baseline", "designs/randomized_experiment.md#ancova", group_id,
        "ave3", "treat1", "ave1", ALL_CONTROLS, ["grade1"], data,
        "cluster(class1)", sample_desc, "grade1",
        "gender age height hukou_rd1 nationality_rd1 health sib fa_eduy mo_eduy pc car",
        "class1", additional_treatment_var="treat2")
    if res: print(f"  coef={res['coefficient']:.4f}, se={res['std_error']:.4f}, p={res['p_value']:.4f}, N={res['n_obs']}")
    baseline_run_ids["primary"] = rid

    # --- Additional baseline: no controls ---
    rid, res = safe_run(
        f"baseline__table3_p{'A' if group_id=='G1' else 'B'}_col1_nocontrols",
        "designs/randomized_experiment.md#ancova", group_id,
        "ave3", "treat1", "ave1", [], [], data,
        "cluster(class1)", sample_desc, "", "none (minimal)", "class1",
        additional_treatment_var="treat2")
    if res: print(f"  nocontrols: coef={res['coefficient']:.4f}, N={res['n_obs']}")

    # --- Additional baseline: Chinese ---
    rid, res = safe_run(
        f"baseline__table3_p{'A' if group_id=='G1' else 'B'}_col4_chn",
        "designs/randomized_experiment.md#ancova", group_id,
        "tchn3", "treat1", "tchn1", ALL_CONTROLS, ["grade1"], data,
        "cluster(class1)", sample_desc, "grade1", "full controls", "class1",
        additional_treatment_var="treat2")
    if res: print(f"  chn: coef={res['coefficient']:.4f}")

    # --- Additional baseline: Math ---
    rid, res = safe_run(
        f"baseline__table3_p{'A' if group_id=='G1' else 'B'}_col6_math",
        "designs/randomized_experiment.md#ancova", group_id,
        "tmath3", "treat1", "tmath1", ALL_CONTROLS, ["grade1"], data,
        "cluster(class1)", sample_desc, "grade1", "full controls", "class1",
        additional_treatment_var="treat2")
    if res: print(f"  math: coef={res['coefficient']:.4f}")

    # --- Design variants ---
    print(f"\n--- {group_id} Design Variants ---")

    # diff_in_means
    rid, res = safe_run(
        "design/randomized_experiment/estimator/diff_in_means",
        "designs/randomized_experiment.md#difference-in-means", group_id,
        "ave3", "treat1", None, [], [], data,
        "cluster(class1)", sample_desc, "", "none (diff-in-means)", "class1",
        additional_treatment_var="treat2",
        extra_payload_blocks={"design": {"randomized_experiment": {**design_audit, "estimator": "diff_in_means"}}})
    if res: print(f"  diff_in_means: coef={res['coefficient']:.4f}")

    # ancova
    rid, res = safe_run(
        "design/randomized_experiment/estimator/ancova",
        "designs/randomized_experiment.md#ancova", group_id,
        "ave3", "treat1", "ave1", [], [], data,
        "cluster(class1)", sample_desc, "", "baseline only (ANCOVA)", "class1",
        additional_treatment_var="treat2",
        extra_payload_blocks={"design": {"randomized_experiment": {**design_audit, "estimator": "ancova"}}})
    if res: print(f"  ancova: coef={res['coefficient']:.4f}")

    # strata_fe
    rid, res = safe_run(
        "design/randomized_experiment/estimator/strata_fe",
        "designs/randomized_experiment.md#strata-fixed-effects", group_id,
        "ave3", "treat1", "ave1", [], ["grade1"], data,
        "cluster(class1)", sample_desc, "grade1", "ANCOVA + strata FE", "class1",
        additional_treatment_var="treat2",
        extra_payload_blocks={"design": {"randomized_experiment": {**design_audit, "estimator": "strata_fe"}}})
    if res: print(f"  strata_fe: coef={res['coefficient']:.4f}")

    # --- RC/Controls LOO ---
    print(f"\n--- {group_id} RC/Controls LOO ---")
    for ctrl in ALL_CONTROLS:
        remaining = [c for c in ALL_CONTROLS if c != ctrl]
        rid, res = safe_run(
            f"rc/controls/loo/drop_{ctrl}",
            "modules/robustness/controls.md#leave-one-out-controls-loo", group_id,
            "ave3", "treat1", "ave1", remaining, ["grade1"], data,
            "cluster(class1)", sample_desc, "grade1", f"full minus {ctrl}", "class1",
            additional_treatment_var="treat2",
            extra_payload_blocks={"controls": {"spec_id": f"rc/controls/loo/drop_{ctrl}", "family": "loo", "dropped": [ctrl], "n_controls": len(remaining)}})
        if res: print(f"  drop_{ctrl}: coef={res['coefficient']:.4f}")

    # --- RC/Controls Sets ---
    print(f"\n--- {group_id} RC/Controls Sets ---")

    # none
    rid, res = safe_run(
        "rc/controls/sets/none", "modules/robustness/controls.md#control-sets", group_id,
        "ave3", "treat1", "ave1", [], ["grade1"], data,
        "cluster(class1)", sample_desc, "grade1", "no controls (grade FE only)", "class1",
        additional_treatment_var="treat2",
        extra_payload_blocks={"controls": {"spec_id": "rc/controls/sets/none", "family": "sets", "included": [], "n_controls": 0}})
    if res: print(f"  none: coef={res['coefficient']:.4f}")

    # demographics
    rid, res = safe_run(
        "rc/controls/sets/demographics", "modules/robustness/controls.md#control-sets", group_id,
        "ave3", "treat1", "ave1", DEMOGRAPHICS, ["grade1"], data,
        "cluster(class1)", sample_desc, "grade1", "demographics: gender age height health", "class1",
        additional_treatment_var="treat2",
        extra_payload_blocks={"controls": {"spec_id": "rc/controls/sets/demographics", "family": "sets", "included": DEMOGRAPHICS, "n_controls": len(DEMOGRAPHICS)}})
    if res: print(f"  demographics: coef={res['coefficient']:.4f}")

    # family_ses
    rid, res = safe_run(
        "rc/controls/sets/family_ses", "modules/robustness/controls.md#control-sets", group_id,
        "ave3", "treat1", "ave1", FAMILY_SES, ["grade1"], data,
        "cluster(class1)", sample_desc, "grade1", "family SES: fa_eduy mo_eduy pc car sib", "class1",
        additional_treatment_var="treat2",
        extra_payload_blocks={"controls": {"spec_id": "rc/controls/sets/family_ses", "family": "sets", "included": FAMILY_SES, "n_controls": len(FAMILY_SES)}})
    if res: print(f"  family_ses: coef={res['coefficient']:.4f}")

    # --- RC/Controls Progression ---
    print(f"\n--- {group_id} RC/Controls Progression ---")

    prog_specs = [
        ("rc/controls/progression/bivariate", None, [], [], "bivariate"),
        ("rc/controls/progression/baseline_only", "ave1", [], [], "baseline only"),
        ("rc/controls/progression/demographics", "ave1", DEMOGRAPHICS, [], "demographics"),
        ("rc/controls/progression/full", "ave1", ALL_CONTROLS, ["grade1"], "full"),
    ]
    for sid, bl, ctrls, fes, desc in prog_specs:
        rid, res = safe_run(
            sid, "modules/robustness/controls.md#control-progression", group_id,
            "ave3", "treat1", bl, ctrls, fes, data,
            "cluster(class1)", sample_desc, " ".join(fes) if fes else "", desc, "class1",
            additional_treatment_var="treat2",
            extra_payload_blocks={"controls": {"spec_id": sid, "family": "progression", "stage": desc, "n_controls": len(ctrls)}})
        if res: print(f"  {desc}: coef={res['coefficient']:.4f}")

    # --- RC/Sample ---
    print(f"\n--- {group_id} RC/Sample ---")

    # pooled tracks
    rid, res = safe_run(
        "rc/sample/pooled_tracks", "modules/robustness/sample.md#pooled-sample", group_id,
        "ave3", "treat1", "ave1", ALL_CONTROLS, ["grade1"], df_full,
        "cluster(class1)", "pooled tracks", "grade1", "full controls", "class1",
        additional_treatment_var="treat2",
        extra_payload_blocks={"sample": {"spec_id": "rc/sample/pooled_tracks", "description": "pooled lower and upper tracks"}})
    if res: print(f"  pooled_tracks: coef={res['coefficient']:.4f}, N={res['n_obs']}")

    # exclude noncompliers
    data_comply = data[data['noncomplier'] != 1].copy()
    rid, res = safe_run(
        "rc/sample/exclude_noncompliers", "modules/robustness/sample.md#exclude-noncompliers", group_id,
        "ave3", "treat1", "ave1", ALL_CONTROLS, ["grade1"], data_comply,
        "cluster(class1)", f"{sample_desc}, excl noncompliers", "grade1", "full controls", "class1",
        additional_treatment_var="treat2",
        extra_payload_blocks={"sample": {"spec_id": "rc/sample/exclude_noncompliers", "description": "exclude noncomplier==1"}})
    if res: print(f"  excl_noncompliers: coef={res['coefficient']:.4f}, N={res['n_obs']}")

    # --- RC/FE ---
    print(f"\n--- {group_id} RC/FE ---")

    # drop grade1
    rid, res = safe_run(
        "rc/fe/drop/grade1", "modules/robustness/fixed_effects.md#drop-fixed-effects", group_id,
        "ave3", "treat1", "ave1", ALL_CONTROLS, [], data,
        "cluster(class1)", sample_desc, "none", "full controls, no FE", "class1",
        additional_treatment_var="treat2",
        extra_payload_blocks={"fixed_effects": {"spec_id": "rc/fe/drop/grade1", "action": "drop", "dropped": ["grade1"]}})
    if res: print(f"  drop_grade1: coef={res['coefficient']:.4f}")

    # add school FE
    rid, res = safe_run(
        "rc/fe/add/school", "modules/robustness/fixed_effects.md#add-fixed-effects", group_id,
        "ave3", "treat1", "ave1", ALL_CONTROLS, ["school"], data,
        "cluster(class1)", sample_desc, "school", "full controls, school FE", "class1",
        additional_treatment_var="treat2",
        extra_payload_blocks={"fixed_effects": {"spec_id": "rc/fe/add/school", "action": "replace", "added": ["school"], "dropped": ["grade1"]}})
    if res: print(f"  add_school: coef={res['coefficient']:.4f}")

    return baseline_run_ids


# ============================================================
# RUN G1: LOWER TRACK
# ============================================================
print("=" * 60)
print("G1: Academic Performance, Lower Track")
print("=" * 60)
g1_baselines = run_group_specs("G1", df_lower, "hsco==0 (lower track)", "Lower")

# ============================================================
# RUN G2: UPPER TRACK
# ============================================================
print("\n" + "=" * 60)
print("G2: Academic Performance, Upper Track")
print("=" * 60)
g2_baselines = run_group_specs("G2", df_upper, "hsco==1 (upper track)", "Upper")


# ############################################################
# INFERENCE VARIANTS
# ############################################################
print("\n" + "=" * 60)
print("INFERENCE VARIANTS")
print("=" * 60)


def run_inference_variant(base_spec_run_id, spec_id, spec_tree_path, baseline_group_id,
                          outcome_var, treatment_var, baseline_outcome_var, controls,
                          fixed_effects, data, vcov_arg, cluster_var_desc,
                          additional_treatment_var=None):
    """Run an inference variant and record it."""
    try:
        rhs_vars = []
        if baseline_outcome_var and baseline_outcome_var in data.columns:
            rhs_vars.append(baseline_outcome_var)
        rhs_vars.append(treatment_var)
        if additional_treatment_var and additional_treatment_var in data.columns:
            rhs_vars.append(additional_treatment_var)
        rhs_vars.extend([c for c in controls if c in data.columns])

        rhs = " + ".join(rhs_vars)
        if fixed_effects:
            formula = f"{outcome_var} ~ {rhs} | {' + '.join(fixed_effects)}"
        else:
            formula = f"{outcome_var} ~ {rhs}"

        all_vars = [outcome_var] + rhs_vars + (fixed_effects or [])
        use_data = data.dropna(subset=[v for v in all_vars if v in data.columns]).copy()

        model = pf.feols(formula, data=use_data, vcov=vcov_arg)

        focal = treatment_var
        coef = float(model.coef().get(focal, np.nan))
        se = float(model.se().get(focal, np.nan))
        pval = float(model.pvalue().get(focal, np.nan))
        ci = model.confint()
        ci_lo = float(ci.loc[focal, ci.columns[0]]) if focal in ci.index else np.nan
        ci_hi = float(ci.loc[focal, ci.columns[1]]) if focal in ci.index else np.nan
        nobs = int(model._N)
        r2 = float(model._r2)
        coefficients = {k: float(v) for k, v in model.coef().items()}

        infer_id = next_infer_id()
        payload = {
            "coefficients": coefficients,
            "inference": {"spec_id": spec_id, "params": {}},
            "software": sw_block,
            "surface_hash": s_hash,
            "design": design_block,
        }

        inference_results.append({
            "paper_id": PAPER_ID,
            "inference_run_id": infer_id,
            "spec_run_id": base_spec_run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": treatment_var,
            "coefficient": coef,
            "std_error": se,
            "p_value": pval,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "n_obs": nobs,
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "cluster_var": cluster_var_desc,
            "run_success": 1,
            "run_error": "",
        })
        return infer_id
    except Exception as e:
        infer_id = next_infer_id()
        error_msg = str(e)[:240]
        payload = {
            "error": error_msg,
            "error_details": {"stage": "inference", "exception_type": type(e).__name__, "exception_message": error_msg}
        }
        inference_results.append({
            "paper_id": PAPER_ID,
            "inference_run_id": infer_id,
            "spec_run_id": base_spec_run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": treatment_var,
            "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "n_obs": np.nan, "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "cluster_var": cluster_var_desc,
            "run_success": 0,
            "run_error": error_msg,
        })
        return infer_id


# G1 inference variants
for gid, gdata, base_rid in [("G1", df_lower, g1_baselines["primary"]),
                               ("G2", df_upper, g2_baselines["primary"])]:
    print(f"\n--- {gid} Inference Variants ---")

    # HC1 robust
    iid = run_inference_variant(
        base_rid, "infer/se/hc/hc1",
        "modules/inference/standard_errors.md#heteroskedasticity-robust", gid,
        "ave3", "treat1", "ave1", ALL_CONTROLS, ["grade1"], gdata,
        "hetero", "", additional_treatment_var="treat2")
    print(f"  HC1: {iid}")

    # Cluster at grade1
    iid = run_inference_variant(
        base_rid, "infer/se/cluster/grade1",
        "modules/inference/standard_errors.md#cluster-robust", gid,
        "ave3", "treat1", "ave1", ALL_CONTROLS, ["grade1"], gdata,
        {"CRV1": "grade1"}, "grade1", additional_treatment_var="treat2")
    print(f"  cluster_grade1: {iid}")


# ############################################################
# WRITE OUTPUTS
# ############################################################
print("\n" + "=" * 60)
print("WRITING OUTPUTS")
print("=" * 60)

df_specs = pd.DataFrame(spec_results)
df_specs.to_csv(f"{PACKAGE_DIR}/specification_results.csv", index=False)
print(f"  specification_results.csv: {len(df_specs)} rows")
print(f"    G1 rows: {len(df_specs[df_specs['baseline_group_id']=='G1'])}")
print(f"    G2 rows: {len(df_specs[df_specs['baseline_group_id']=='G2'])}")
print(f"    Successes: {df_specs['run_success'].sum()}")
print(f"    Failures: {(df_specs['run_success']==0).sum()}")

df_infer = pd.DataFrame(inference_results)
df_infer.to_csv(f"{PACKAGE_DIR}/inference_results.csv", index=False)
print(f"\n  inference_results.csv: {len(df_infer)} rows")

assert df_specs['spec_run_id'].is_unique, "spec_run_id not unique!"
assert df_infer['inference_run_id'].is_unique, "inference_run_id not unique!"

print("\nDone!")
