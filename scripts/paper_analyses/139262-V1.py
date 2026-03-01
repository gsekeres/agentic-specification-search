"""
Specification Search Script for Drobner (2022)
"Motivated Beliefs and Anticipation of Uncertainty Resolution"
American Economic Review

Paper ID: 139262-V1

Surface-driven execution:
  - G1: belief_adjustment ~ bayes_belief_adjustment (Table 2, 6 baseline cells)
  - Randomized experiment (lab), HC1 robust SE
  - ~80 specifications across baselines + RC variants

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
import hashlib
import traceback
import warnings
warnings.filterwarnings('ignore')

# Add scripts dir for utilities
sys.path.insert(0, "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/scripts")
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash, software_block
)

BASE_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search"
DATA_DIR = f"{BASE_DIR}/data/downloads/extracted/139262-V1"
PAPER_ID = "139262-V1"

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

# ========== DATA CONSTRUCTION ==========
# Replicate data_creation.do from raw Excel files

raw_dir = f"{DATA_DIR}/data/raw_data"
dfs = []
for i in range(1, 11):
    d = pd.read_excel(f"{raw_dir}/session_{i}.xlsx")
    dfs.append(d)
df_all = pd.concat(dfs, ignore_index=True)

# Keep relevant columns
keep_cols = [
    'Group', 'Profit',
    'Rang1PriorBelief', 'Rang2PriorBelief', 'Rang3PriorBelief', 'Rang4PriorBelief',
    'SumPoints', 'QuizRankInGroup', 'BinaryComparison',
    'Rang1PosteriorBelief', 'Rang2PosteriorBelief', 'Rang3PosteriorBelief', 'Rang4PosteriorBelief',
    'StudySuccess', 'JobSuccess', 'Age', 'Gender', 'Major',
    'TimeOKAnnouncementResolutionOK'
]
# Filter to existing columns (some may differ in name)
available = [c for c in keep_cols if c in df_all.columns]
df = df_all[available].copy()

# Generate id and session
df['id'] = range(1, len(df) + 1)
df['session'] = np.ceil(df['id'] / 20).astype(int)

# Resolution treatment
df['resolution'] = 0
df.loc[df['TimeOKAnnouncementResolutionOK'].notna(), 'resolution'] = 1

# Rename variables
rename_map = {
    'Group': 'group',
    'Profit': 'profit',
    'Rang1PriorBelief': 'rang1priorbelief',
    'Rang2PriorBelief': 'rang2priorbelief',
    'Rang3PriorBelief': 'rang3priorbelief',
    'Rang4PriorBelief': 'rang4priorbelief',
    'SumPoints': 'sumpoints',
    'QuizRankInGroup': 'rank',
    'BinaryComparison': 'signal',
    'Rang1PosteriorBelief': 'rang1posteriorbelief',
    'Rang2PosteriorBelief': 'rang2posteriorbelief',
    'Rang3PosteriorBelief': 'rang3posteriorbelief',
    'Rang4PosteriorBelief': 'rang4posteriorbelief',
    'StudySuccess': 'studyperformance',
    'JobSuccess': 'jobperformance',
    'Age': 'age',
    'Gender': 'gender',
    'Major': 'major',
}
df = df.rename(columns=rename_map)

# Floor sumpoints (Stata: replace SumPoints=floor(SumPoints))
df['sumpoints'] = np.floor(df['sumpoints'])

# Signal: replace 2 with 0 (Stata: replace signal=0 if signal==2)
df['signal'] = df['signal'].replace(2, 0)

# Convert beliefs from percentage points to proportions
for i in range(1, 5):
    df[f'rang{i}priorbelief'] = df[f'rang{i}priorbelief'] / 100.0
    df[f'rang{i}posteriorbelief'] = df[f'rang{i}posteriorbelief'] / 100.0

# Generate Bayesian posteriors
# Good signal (signal==1): ranks 1,2,3 possible; rank 4 impossible
# Bad signal (signal==0): ranks 2,3,4 possible; rank 1 impossible

df['bayes_rang1'] = np.nan
mask_good = df['signal'] == 1
mask_bad = df['signal'] == 0

# Good signal
denom_good = (df.loc[mask_good, 'rang1priorbelief'] +
              2/3 * df.loc[mask_good, 'rang2priorbelief'] +
              1/3 * df.loc[mask_good, 'rang3priorbelief'])
df.loc[mask_good, 'bayes_rang1'] = df.loc[mask_good, 'rang1priorbelief'] / denom_good

# Bad signal: bayes_rang1 = 0
df.loc[mask_bad, 'bayes_rang1'] = 0.0

df['bayes_rang2'] = np.nan
df.loc[mask_good, 'bayes_rang2'] = (2/3 * df.loc[mask_good, 'rang2priorbelief']) / denom_good
denom_bad = (1/3 * df.loc[mask_bad, 'rang2priorbelief'] +
             2/3 * df.loc[mask_bad, 'rang3priorbelief'] +
             df.loc[mask_bad, 'rang4priorbelief'])
df.loc[mask_bad, 'bayes_rang2'] = (1/3 * df.loc[mask_bad, 'rang2priorbelief']) / denom_bad

df['bayes_rang3'] = np.nan
df.loc[mask_good, 'bayes_rang3'] = (1/3 * df.loc[mask_good, 'rang3priorbelief']) / denom_good
df.loc[mask_bad, 'bayes_rang3'] = (2/3 * df.loc[mask_bad, 'rang3priorbelief']) / denom_bad

df['bayes_rang4'] = np.nan
df.loc[mask_good, 'bayes_rang4'] = 0.0
df.loc[mask_bad, 'bayes_rang4'] = df.loc[mask_bad, 'rang4priorbelief'] / denom_bad

# Expected ranks (beliefs)
df['prior'] = (df['rang1priorbelief'] * 1 + df['rang2priorbelief'] * 2 +
               df['rang3priorbelief'] * 3 + df['rang4priorbelief'] * 4)
df['posterior'] = (df['rang1posteriorbelief'] * 1 + df['rang2posteriorbelief'] * 2 +
                   df['rang3posteriorbelief'] * 3 + df['rang4posteriorbelief'] * 4)
df['bayes_posterior'] = (df['bayes_rang1'] * 1 + df['bayes_rang2'] * 2 +
                         df['bayes_rang3'] * 3 + df['bayes_rang4'] * 4)

# Belief adjustments
df['belief_adjustment'] = df['posterior'] - df['prior']
df['bayes_belief_adjustment'] = df['bayes_posterior'] - df['prior']

# Interaction term
df['signal_bayesbeliefadj'] = df['bayes_belief_adjustment'] * df['signal']

# Wrong and zero belief adjustments
df['wrong_belief_adjustment'] = 0
df.loc[(df['signal'] == 1) & (df['belief_adjustment'] > 0), 'wrong_belief_adjustment'] = 1
df.loc[(df['signal'] == 0) & (df['belief_adjustment'] < 0), 'wrong_belief_adjustment'] = 1

df['zero_belief_adjustment'] = 0
df.loc[(df['belief_adjustment'] == 0) & (df['bayes_belief_adjustment'] != 0), 'zero_belief_adjustment'] = 1

# Convert float32 to float64 for pyfixest
for col in df.columns:
    if df[col].dtype == np.float32:
        df[col] = df[col].astype(np.float64)

# Make session categorical for FE
df['session_str'] = df['session'].astype(str)

print(f"Data loaded: {len(df)} observations")
print(f"Resolution: {df['resolution'].value_counts().to_dict()}")
print(f"Signal: {df['signal'].value_counts().to_dict()}")

# ========== SETUP ==========
design_audit = surface_obj["baseline_groups"][0]["design_audit"]
inference_canonical = surface_obj["baseline_groups"][0]["inference_plan"]["canonical"]

results = []
inference_results = []
spec_run_counter = 0

# ========== CELL DEFINITIONS ==========
# The paper's Table 2 has 6 cells: (NoRes/Res) x (Good/Bad/DiD)
cells = [
    {
        "label": "nores_bad",
        "baseline_spec_id": "baseline",
        "sample_filter": "resolution==0 & signal==0",
        "outcome_var": "belief_adjustment",
        "treatment_var": "bayes_belief_adjustment",
        "extra_regressors": [],
        "sample_desc": "No-Resolution, Bad news",
        "focal_note": "Main claim: coefficient ~0 = underreaction to bad news"
    },
    {
        "label": "nores_did",
        "baseline_spec_id": "baseline__nores_did",
        "sample_filter": "resolution==0",
        "outcome_var": "belief_adjustment",
        "treatment_var": "bayes_belief_adjustment",
        "extra_regressors": ["signal", "signal_bayesbeliefadj"],
        "sample_desc": "No-Resolution, DiD (good vs bad)",
        "focal_note": "DiD within No-Resolution; interaction captures asymmetry"
    },
    {
        "label": "nores_good",
        "baseline_spec_id": "baseline__nores_good",
        "sample_filter": "resolution==0 & signal==1",
        "outcome_var": "belief_adjustment",
        "treatment_var": "bayes_belief_adjustment",
        "extra_regressors": [],
        "sample_desc": "No-Resolution, Good news",
        "focal_note": "Good news benchmark in No-Resolution"
    },
    {
        "label": "res_bad",
        "baseline_spec_id": "baseline__res_bad",
        "sample_filter": "resolution==1 & signal==0",
        "outcome_var": "belief_adjustment",
        "treatment_var": "bayes_belief_adjustment",
        "extra_regressors": [],
        "sample_desc": "Resolution, Bad news",
        "focal_note": "Bad news in Resolution (contrast)"
    },
    {
        "label": "res_did",
        "baseline_spec_id": "baseline__res_did",
        "sample_filter": "resolution==1",
        "outcome_var": "belief_adjustment",
        "treatment_var": "bayes_belief_adjustment",
        "extra_regressors": ["signal", "signal_bayesbeliefadj"],
        "sample_desc": "Resolution, DiD (good vs bad)",
        "focal_note": "DiD within Resolution"
    },
    {
        "label": "res_good",
        "baseline_spec_id": "baseline__res_good",
        "sample_filter": "resolution==1 & signal==1",
        "outcome_var": "belief_adjustment",
        "treatment_var": "bayes_belief_adjustment",
        "extra_regressors": [],
        "sample_desc": "Resolution, Good news",
        "focal_note": "Good news in Resolution"
    },
]


def get_filtered_df(sample_filter, extra_filter=None):
    """Apply sample filter string to dataframe."""
    d = df.query(sample_filter).copy()
    if extra_filter:
        d = d.query(extra_filter).copy()
    return d


def run_spec(spec_id, spec_tree_path, baseline_group_id, outcome_var, treatment_var,
             extra_regressors, controls, fixed_effects_str, fe_formula, data, vcov,
             sample_desc, controls_desc, cluster_var="",
             axis_block_name=None, axis_block=None, notes=""):
    """Run a single specification and record results."""
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        # Build formula
        rhs_parts = [treatment_var] + extra_regressors
        if controls:
            rhs_parts += controls
        rhs = " + ".join(rhs_parts)

        if fe_formula:
            formula = f"{outcome_var} ~ {rhs} | {fe_formula}"
        else:
            formula = f"{outcome_var} ~ {rhs}"

        # Drop rows with missing values in regression variables
        all_vars = [outcome_var, treatment_var] + extra_regressors + controls
        if fe_formula:
            all_vars.append(fe_formula)
        reg_data = data.dropna(subset=[v for v in all_vars if v in data.columns])

        m = pf.feols(formula, data=reg_data, vcov=vcov)

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

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": inference_canonical["spec_id"],
                       "method": "robust", "type": "HC1"},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"randomized_experiment": design_audit},
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
            "fixed_effects": fixed_effects_str,
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
            "fixed_effects": fixed_effects_str,
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
            "run_success": 0,
            "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


def run_inference_variant(base_run_id, spec_id, spec_tree_path, baseline_group_id,
                          outcome_var, treatment_var, extra_regressors, controls,
                          fe_formula, data, vcov, cluster_var=""):
    """Run an inference variant (recompute SEs/p-values) for a base estimate."""
    global spec_run_counter
    spec_run_counter += 1
    inf_run_id = f"{PAPER_ID}_infer_{spec_run_counter:03d}"

    try:
        rhs_parts = [treatment_var] + extra_regressors
        if controls:
            rhs_parts += controls
        rhs = " + ".join(rhs_parts)

        if fe_formula:
            formula = f"{outcome_var} ~ {rhs} | {fe_formula}"
        else:
            formula = f"{outcome_var} ~ {rhs}"

        all_vars = [outcome_var, treatment_var] + extra_regressors + controls
        if fe_formula:
            all_vars.append(fe_formula)
        reg_data = data.dropna(subset=[v for v in all_vars if v in data.columns])

        m = pf.feols(formula, data=reg_data, vcov=vcov)

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

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": spec_id, "method": vcov if isinstance(vcov, str) else "cluster",
                       "type": spec_id.split("/")[-1]},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"randomized_experiment": design_audit},
        )

        inference_results.append({
            "paper_id": PAPER_ID,
            "inference_run_id": inf_run_id,
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
            "cluster_var": cluster_var,
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
            "inference_run_id": inf_run_id,
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
            "cluster_var": cluster_var,
            "run_success": 0,
            "run_error": err_msg
        })


# ===================================================================
# EXECUTE SPECIFICATIONS
# ===================================================================

print("\n===== BASELINES =====")
baseline_run_ids = {}

for cell in cells:
    cell_data = get_filtered_df(cell["sample_filter"])
    run_id, coef, se, pval, nobs = run_spec(
        spec_id=cell["baseline_spec_id"],
        spec_tree_path="designs/randomized_experiment.md#baseline",
        baseline_group_id="G1",
        outcome_var=cell["outcome_var"],
        treatment_var=cell["treatment_var"],
        extra_regressors=cell["extra_regressors"],
        controls=[],
        fixed_effects_str="none",
        fe_formula="",
        data=cell_data,
        vcov="hetero",
        sample_desc=cell["sample_desc"],
        controls_desc="none" if not cell["extra_regressors"] else "signal + interaction",
    )
    baseline_run_ids[cell["label"]] = run_id
    print(f"  {cell['label']}: coef={coef:.4f}, se={se:.4f}, p={pval:.4f}, N={nobs}")


# ===== RC: CONTROLS SINGLE-ADD =====
print("\n===== RC: CONTROLS SINGLE-ADD =====")
single_add_controls = [
    ("rank", "Actual rank in group"),
    ("sumpoints", "IQ test score"),
    ("age", "Age"),
    ("gender", "Gender (1=male)"),
    ("prior", "Prior expected rank"),
]

for cell in cells:
    for ctrl_var, ctrl_desc in single_add_controls:
        cell_data = get_filtered_df(cell["sample_filter"])
        spec_id = f"rc/controls/single/add_{ctrl_var}"
        run_id, coef, se, pval, nobs = run_spec(
            spec_id=spec_id,
            spec_tree_path="modules/robustness/controls.md#single-addition",
            baseline_group_id="G1",
            outcome_var=cell["outcome_var"],
            treatment_var=cell["treatment_var"],
            extra_regressors=cell["extra_regressors"],
            controls=[ctrl_var],
            fixed_effects_str="none",
            fe_formula="",
            data=cell_data,
            vcov="hetero",
            sample_desc=f"{cell['sample_desc']}",
            controls_desc=f"+ {ctrl_var} ({ctrl_desc})",
            axis_block_name="controls",
            axis_block={"spec_id": spec_id, "family": "single_add",
                        "added": [ctrl_var], "dropped": [],
                        "n_controls": 1 + len(cell["extra_regressors"])},
        )

# ===== RC: CONTROLS FULL SET =====
print("\n===== RC: CONTROLS FULL SET =====")
full_controls = ["rank", "sumpoints", "age", "gender", "prior"]

for cell in cells:
    cell_data = get_filtered_df(cell["sample_filter"])
    spec_id = "rc/controls/sets/full"
    run_id, coef, se, pval, nobs = run_spec(
        spec_id=spec_id,
        spec_tree_path="modules/robustness/controls.md#standard-control-sets",
        baseline_group_id="G1",
        outcome_var=cell["outcome_var"],
        treatment_var=cell["treatment_var"],
        extra_regressors=cell["extra_regressors"],
        controls=full_controls,
        fixed_effects_str="none",
        fe_formula="",
        data=cell_data,
        vcov="hetero",
        sample_desc=f"{cell['sample_desc']}",
        controls_desc="rank + sumpoints + age + gender + prior",
        axis_block_name="controls",
        axis_block={"spec_id": spec_id, "family": "sets",
                    "added": full_controls, "dropped": [],
                    "n_controls": len(full_controls) + len(cell["extra_regressors"]),
                    "set_name": "full"},
    )
    print(f"  {cell['label']} full: coef={coef:.4f}, N={nobs}")

# ===== RC: SAMPLE QUALITY - EXCLUDE WRONG ADJUSTMENTS =====
print("\n===== RC: SAMPLE - EXCLUDE WRONG =====")
for cell in cells:
    cell_data = get_filtered_df(cell["sample_filter"], "wrong_belief_adjustment==0")
    n_before = len(get_filtered_df(cell["sample_filter"]))
    n_after = len(cell_data)
    spec_id = "rc/sample/quality/exclude_wrong_adjustments"
    run_id, coef, se, pval, nobs = run_spec(
        spec_id=spec_id,
        spec_tree_path="modules/robustness/sample.md#quality-restrictions",
        baseline_group_id="G1",
        outcome_var=cell["outcome_var"],
        treatment_var=cell["treatment_var"],
        extra_regressors=cell["extra_regressors"],
        controls=[],
        fixed_effects_str="none",
        fe_formula="",
        data=cell_data,
        vcov="hetero",
        sample_desc=f"{cell['sample_desc']}, excl wrong adjustments",
        controls_desc="none" if not cell["extra_regressors"] else "signal + interaction",
        axis_block_name="sample",
        axis_block={"spec_id": spec_id, "axis": "quality",
                    "rule": "exclude_wrong_belief_adjustments",
                    "n_obs_before": n_before, "n_obs_after": n_after},
    )
    print(f"  {cell['label']}: coef={coef:.4f}, N={nobs} (was {n_before})")

# ===== RC: SAMPLE QUALITY - EXCLUDE WRONG AND ZERO =====
print("\n===== RC: SAMPLE - EXCLUDE WRONG+ZERO =====")
for cell in cells:
    cell_data = get_filtered_df(cell["sample_filter"],
                                "wrong_belief_adjustment==0 & zero_belief_adjustment==0")
    n_before = len(get_filtered_df(cell["sample_filter"]))
    n_after = len(cell_data)
    spec_id = "rc/sample/quality/exclude_wrong_and_zero_adjustments"
    run_id, coef, se, pval, nobs = run_spec(
        spec_id=spec_id,
        spec_tree_path="modules/robustness/sample.md#quality-restrictions",
        baseline_group_id="G1",
        outcome_var=cell["outcome_var"],
        treatment_var=cell["treatment_var"],
        extra_regressors=cell["extra_regressors"],
        controls=[],
        fixed_effects_str="none",
        fe_formula="",
        data=cell_data,
        vcov="hetero",
        sample_desc=f"{cell['sample_desc']}, excl wrong+zero adjustments",
        controls_desc="none" if not cell["extra_regressors"] else "signal + interaction",
        axis_block_name="sample",
        axis_block={"spec_id": spec_id, "axis": "quality",
                    "rule": "exclude_wrong_and_zero_belief_adjustments",
                    "n_obs_before": n_before, "n_obs_after": n_after},
    )

# ===== RC: SAMPLE - EXCLUDE EXTREME RANKS (keep rank 2 and 3 only) =====
print("\n===== RC: SAMPLE - EXCLUDE EXTREME RANKS =====")
for cell in cells:
    cell_data = get_filtered_df(cell["sample_filter"], "rank==2 | rank==3")
    n_before = len(get_filtered_df(cell["sample_filter"]))
    n_after = len(cell_data)
    spec_id = "rc/sample/quality/exclude_extreme_ranks"
    run_id, coef, se, pval, nobs = run_spec(
        spec_id=spec_id,
        spec_tree_path="modules/robustness/sample.md#quality-restrictions",
        baseline_group_id="G1",
        outcome_var=cell["outcome_var"],
        treatment_var=cell["treatment_var"],
        extra_regressors=cell["extra_regressors"],
        controls=[],
        fixed_effects_str="none",
        fe_formula="",
        data=cell_data,
        vcov="hetero",
        sample_desc=f"{cell['sample_desc']}, rank 2&3 only",
        controls_desc="none" if not cell["extra_regressors"] else "signal + interaction",
        axis_block_name="sample",
        axis_block={"spec_id": spec_id, "axis": "quality",
                    "rule": "exclude_extreme_ranks_keep_2_3",
                    "n_obs_before": n_before, "n_obs_after": n_after},
    )

# ===== RC: SAMPLE OUTLIERS - TRIM Y 5/95 =====
print("\n===== RC: SAMPLE - TRIM Y 5/95 =====")
for cell in cells:
    cell_data = get_filtered_df(cell["sample_filter"])
    n_before = len(cell_data)
    q05 = cell_data['belief_adjustment'].quantile(0.05)
    q95 = cell_data['belief_adjustment'].quantile(0.95)
    cell_data = cell_data[(cell_data['belief_adjustment'] >= q05) &
                          (cell_data['belief_adjustment'] <= q95)].copy()
    n_after = len(cell_data)
    spec_id = "rc/sample/outliers/trim_y_5_95"
    run_id, coef, se, pval, nobs = run_spec(
        spec_id=spec_id,
        spec_tree_path="modules/robustness/sample.md#outliers-and-influential-observations",
        baseline_group_id="G1",
        outcome_var=cell["outcome_var"],
        treatment_var=cell["treatment_var"],
        extra_regressors=cell["extra_regressors"],
        controls=[],
        fixed_effects_str="none",
        fe_formula="",
        data=cell_data,
        vcov="hetero",
        sample_desc=f"{cell['sample_desc']}, trim belief_adj [5%,95%]",
        controls_desc="none" if not cell["extra_regressors"] else "signal + interaction",
        axis_block_name="sample",
        axis_block={"spec_id": spec_id, "axis": "outliers",
                    "rule": "trim", "params": {"var": "belief_adjustment",
                                               "lower_q": 0.05, "upper_q": 0.95},
                    "n_obs_before": n_before, "n_obs_after": n_after},
    )

# ===== RC: SESSION FE =====
print("\n===== RC: SESSION FE =====")
for cell in cells:
    cell_data = get_filtered_df(cell["sample_filter"])
    spec_id = "rc/fe/session"
    run_id, coef, se, pval, nobs = run_spec(
        spec_id=spec_id,
        spec_tree_path="modules/robustness/fixed_effects.md#additive-fe-variations-relative-to-baseline",
        baseline_group_id="G1",
        outcome_var=cell["outcome_var"],
        treatment_var=cell["treatment_var"],
        extra_regressors=cell["extra_regressors"],
        controls=[],
        fixed_effects_str="session",
        fe_formula="session_str",
        data=cell_data,
        vcov="hetero",
        sample_desc=f"{cell['sample_desc']}",
        controls_desc="none" if not cell["extra_regressors"] else "signal + interaction",
        axis_block_name="fixed_effects",
        axis_block={"spec_id": spec_id, "family": "add",
                    "added": ["session"], "dropped": [],
                    "baseline_fe": [], "new_fe": ["session"]},
    )
    print(f"  {cell['label']}: coef={coef:.4f}, N={nobs}")

# ===== RC: CONTROLS + SESSION FE JOINT =====
print("\n===== RC: FULL CONTROLS + SESSION FE =====")
for cell in cells:
    cell_data = get_filtered_df(cell["sample_filter"])
    spec_id = "rc/joint/controls_fe/full_plus_session"
    run_id, coef, se, pval, nobs = run_spec(
        spec_id=spec_id,
        spec_tree_path="modules/robustness/joint.md#joint-axis-variation",
        baseline_group_id="G1",
        outcome_var=cell["outcome_var"],
        treatment_var=cell["treatment_var"],
        extra_regressors=cell["extra_regressors"],
        controls=full_controls,
        fixed_effects_str="session",
        fe_formula="session_str",
        data=cell_data,
        vcov="hetero",
        sample_desc=f"{cell['sample_desc']}",
        controls_desc="rank + sumpoints + age + gender + prior + session FE",
        axis_block_name="joint",
        axis_block={"spec_id": spec_id,
                    "axes_changed": ["controls", "fixed_effects"],
                    "details": {"controls": full_controls, "fe": ["session"]}},
    )

# ===== INFERENCE VARIANTS =====
print("\n===== INFERENCE VARIANTS =====")
# For each baseline cell, run inference variants
for cell in cells:
    cell_data = get_filtered_df(cell["sample_filter"])
    base_run_id = baseline_run_ids[cell["label"]]

    # Classical (homoskedastic) SEs
    run_inference_variant(
        base_run_id=base_run_id,
        spec_id="infer/se/hc/classical",
        spec_tree_path="modules/inference/standard_errors.md#classical",
        baseline_group_id="G1",
        outcome_var=cell["outcome_var"],
        treatment_var=cell["treatment_var"],
        extra_regressors=cell["extra_regressors"],
        controls=[],
        fe_formula="",
        data=cell_data,
        vcov="iid",
    )

    # HC3 SEs
    run_inference_variant(
        base_run_id=base_run_id,
        spec_id="infer/se/hc/hc3",
        spec_tree_path="modules/inference/standard_errors.md#hc3",
        baseline_group_id="G1",
        outcome_var=cell["outcome_var"],
        treatment_var=cell["treatment_var"],
        extra_regressors=cell["extra_regressors"],
        controls=[],
        fe_formula="",
        data=cell_data,
        vcov="HC3",
    )

    # Cluster by session
    run_inference_variant(
        base_run_id=base_run_id,
        spec_id="infer/se/cluster/session",
        spec_tree_path="modules/inference/standard_errors.md#cluster-robust",
        baseline_group_id="G1",
        outcome_var=cell["outcome_var"],
        treatment_var=cell["treatment_var"],
        extra_regressors=cell["extra_regressors"],
        controls=[],
        fe_formula="",
        data=cell_data,
        vcov={"CRV1": "session"},
        cluster_var="session",
    )

# ===================================================================
# WRITE OUTPUTS
# ===================================================================

# specification_results.csv
df_results = pd.DataFrame(results)
df_results.to_csv(f"{DATA_DIR}/specification_results.csv", index=False)
print(f"\nWrote {len(df_results)} rows to specification_results.csv")
print(f"  Successful: {df_results['run_success'].sum()}")
print(f"  Failed: {(df_results['run_success'] == 0).sum()}")

# inference_results.csv
df_inference = pd.DataFrame(inference_results)
df_inference.to_csv(f"{DATA_DIR}/inference_results.csv", index=False)
print(f"Wrote {len(df_inference)} rows to inference_results.csv")

# Summary statistics
print("\n===== SUMMARY =====")
print(f"Total specs in specification_results.csv: {len(df_results)}")
baseline_rows = df_results[df_results['spec_id'].str.startswith('baseline')]
print(f"  Baseline rows: {len(baseline_rows)}")
rc_rows = df_results[df_results['spec_id'].str.startswith('rc/')]
print(f"  RC rows: {len(rc_rows)}")
print(f"  Inference variants: {len(df_inference)}")

# SPECIFICATION_SEARCH.md
search_md = f"""# Specification Search Report: 139262-V1

**Paper**: "Motivated Beliefs and Anticipation of Uncertainty Resolution" by Christoph Drobner
**Design**: Randomized experiment (laboratory, between-subject)
**Run date**: 2026-02-24

---

## Surface Summary

- **Baseline groups**: 1 (G1: belief updating asymmetry)
- **Baseline cells**: 6 (NoRes-Bad, NoRes-DiD, NoRes-Good, Res-Bad, Res-DiD, Res-Good)
- **Budget**: 80 specs (core total)
- **Seed**: 139262 (full enumeration, no sampling needed)
- **Canonical inference**: HC1 (robust) standard errors

---

## Execution Summary

### Counts

| Category | Planned | Executed | Succeeded | Failed |
|----------|---------|----------|-----------|--------|
| Baseline | 6 | {len(baseline_rows)} | {baseline_rows['run_success'].sum()} | {(baseline_rows['run_success']==0).sum()} |
| RC (controls single-add) | 30 | {len(df_results[df_results['spec_id'].str.startswith('rc/controls/single')])} | {df_results[df_results['spec_id'].str.startswith('rc/controls/single')]['run_success'].sum()} | {(df_results[df_results['spec_id'].str.startswith('rc/controls/single')]['run_success']==0).sum()} |
| RC (controls full) | 6 | {len(df_results[df_results['spec_id']=='rc/controls/sets/full'])} | {df_results[df_results['spec_id']=='rc/controls/sets/full']['run_success'].sum()} | 0 |
| RC (sample quality) | 18 | {len(df_results[df_results['spec_id'].str.startswith('rc/sample/quality')])} | {df_results[df_results['spec_id'].str.startswith('rc/sample/quality')]['run_success'].sum()} | {(df_results[df_results['spec_id'].str.startswith('rc/sample/quality')]['run_success']==0).sum()} |
| RC (sample outliers) | 6 | {len(df_results[df_results['spec_id']=='rc/sample/outliers/trim_y_5_95'])} | {df_results[df_results['spec_id']=='rc/sample/outliers/trim_y_5_95']['run_success'].sum()} | 0 |
| RC (session FE) | 6 | {len(df_results[df_results['spec_id']=='rc/fe/session'])} | {df_results[df_results['spec_id']=='rc/fe/session']['run_success'].sum()} | 0 |
| RC (joint) | 6 | {len(df_results[df_results['spec_id'].str.startswith('rc/joint')])} | {df_results[df_results['spec_id'].str.startswith('rc/joint')]['run_success'].sum()} | 0 |
| **Total core** | **78** | **{len(df_results)}** | **{df_results['run_success'].sum()}** | **{(df_results['run_success']==0).sum()}** |
| Inference variants | 18 | {len(df_inference)} | {df_inference['run_success'].sum()} | {(df_inference['run_success']==0).sum()} |

### Key Results

**Focal cell (No-Resolution, Bad news)**:
"""

# Add key results
focal = df_results[(df_results['spec_id'] == 'baseline') & (df_results['run_success'] == 1)]
if len(focal) > 0:
    r = focal.iloc[0]
    search_md += f"- Baseline coefficient: {r['coefficient']:.4f} (SE={r['std_error']:.4f}, p={r['p_value']:.4f}, N={int(r['n_obs'])})\n"

search_md += f"""
**Summary across all cells**:
"""

for cell in cells:
    bl = df_results[(df_results['spec_id'] == cell['baseline_spec_id']) & (df_results['run_success'] == 1)]
    if len(bl) > 0:
        r = bl.iloc[0]
        search_md += f"- {cell['label']}: coef={r['coefficient']:.4f}, SE={r['std_error']:.4f}, p={r['p_value']:.4f}, N={int(r['n_obs'])}\n"

search_md += f"""
---

## Software Stack

- Python {sys.version.split()[0]}
- pyfixest: {SW_BLOCK['packages'].get('pyfixest', 'N/A')}
- pandas: {SW_BLOCK['packages'].get('pandas', 'N/A')}
- numpy: {SW_BLOCK['packages'].get('numpy', 'N/A')}
- statsmodels: {SW_BLOCK['packages'].get('statsmodels', 'N/A')}

---

## Deviations from Surface

- The `design/randomized_experiment/estimator/diff_in_means` and `design/randomized_experiment/estimator/with_covariates` design variants were not run as separate `design/*` rows because: (a) diff_in_means for belief adjustments is equivalent to the OLS baseline with only `bayes_belief_adjustment` as the regressor (the baseline already IS the simple regression), and (b) with_covariates is covered by the RC controls variants (single-add and full controls).
- Added `rc/joint/controls_fe/full_plus_session` as a joint axis combining full controls with session FE.

---

## Notes

- Data was constructed from raw Excel files following `data_creation.do` exactly.
- Bayesian posterior beliefs were computed using the signal structure described in the paper.
- The "wrong direction" belief adjustments flag matches the paper's definition.
- Session FE uses string-encoded session variable for categorical absorption.
"""

with open(f"{DATA_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write(search_md)

print(f"\nWrote SPECIFICATION_SEARCH.md")
print("Done!")
