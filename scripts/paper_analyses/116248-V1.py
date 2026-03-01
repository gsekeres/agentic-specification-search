"""
Specification Search Script for Costa-Gomes & Crawford (2006)
"Cognition and Behavior in Two-Person Guessing Games: An Experimental Study"
American Economic Review, 96(5), 1737-1768.

Paper ID: 116248-V1

Surface-driven execution:
  - G1: abs_dev_eq ~ comply_l1 + game_controls | subject_FE + game_FE, cluster(subject)
  - Panel OLS at the subject x game level (88 subjects, 16 games = 1,408 obs)
  - The paper's main finding: subjects are best described as L1 (Level-1) types,
    and L1 compliance predicts lower deviation from equilibrium predictions
  - 50+ specifications across type indicators, outcome measures, control sets,
    sample restrictions, FE swaps, and functional form

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

PAPER_ID = "116248-V1"
DATA_DIR = "data/downloads/extracted/116248-V1"
OUTPUT_DIR = DATA_DIR

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

bg = surface_obj["baseline_groups"][0]
design_audit = bg["design_audit"]
inference_canonical = bg["inference_plan"]["canonical"]

# ============================================================
# Data Loading and Panel Construction
# ============================================================

# SubjectsGuesses: 88 rows x 17 cols (col0=ID, cols1-16=guesses for 16 games in master order)
subj_guesses = pd.read_csv(f"{DATA_DIR}/SubjectsGuesses.xls", sep='\t', header=None)

# TypesGuesses: 16 rows x 7 cols (col0=game_id, cols1-6=Eq,L1,L2,L3,D1,D2 predicted guesses)
type_guesses = pd.read_csv(f"{DATA_DIR}/TypesGuesses.xls", sep='\t', header=None)

# DominanceRounds: bounds after iterated deletion of dominated strategies
dom_rounds = pd.read_csv(f"{DATA_DIR}/DominanceRounds.xls", sep='\t', header=None)

# Games metadata from the GAUSS code (lower limit, upper limit, target for each player)
# Games in master order: 1,3,5,7,9,11,13,15,2,4,6,8,10,12,14,16
gam_raw = np.array([
    [100, 500, 0.7, 100, 900, 0.5],
    [100, 900, 0.5, 300, 500, 0.7],
    [300, 500, 1.5, 300, 900, 1.3],
    [300, 900, 1.3, 300, 900, 1.3],
    [100, 900, 0.5, 100, 500, 1.5],
    [300, 900, 0.7, 100, 900, 1.3],
    [300, 500, 0.7, 100, 900, 1.5],
    [100, 500, 0.7, 100, 500, 1.5],
    [100, 900, 0.5, 100, 500, 0.7],
    [300, 500, 0.7, 100, 900, 0.5],
    [300, 900, 1.3, 300, 500, 1.5],
    [300, 900, 1.3, 300, 900, 1.3],
    [100, 500, 1.5, 100, 900, 0.5],
    [100, 900, 1.3, 300, 900, 0.7],
    [100, 900, 1.5, 300, 500, 0.7],
    [100, 500, 1.5, 100, 500, 0.7],
])

game_ids = type_guesses[0].astype(int).tolist()  # game IDs in master order

# Build panel
rows = []
for subj_idx in range(len(subj_guesses)):
    subj_id_raw = subj_guesses.iloc[subj_idx, 0]
    sid_str = f"{subj_id_raw:.4f}"
    session = int(sid_str.split('.')[1][:2])
    subj_num = int(sid_str.split('.')[1][2:])
    subj_id_int = session * 100 + subj_num  # e.g., 101, 102, etc.

    # First 71 subjects are Baseline, last 17 are Open Boxes
    is_baseline = 1 if subj_idx < 71 else 0

    for game_pos in range(16):
        game_id = game_ids[game_pos]
        guess = float(subj_guesses.iloc[subj_idx, game_pos + 1])

        # Type predictions for this game
        eq_pred = float(type_guesses.iloc[game_pos, 1])
        l1_pred = float(type_guesses.iloc[game_pos, 2])
        l2_pred = float(type_guesses.iloc[game_pos, 3])
        l3_pred = float(type_guesses.iloc[game_pos, 4])
        d1_pred = float(type_guesses.iloc[game_pos, 5])
        d2_pred = float(type_guesses.iloc[game_pos, 6])

        # Game characteristics
        own_lower = gam_raw[game_pos, 0]
        own_upper = gam_raw[game_pos, 1]
        own_target = gam_raw[game_pos, 2]
        opp_lower = gam_raw[game_pos, 3]
        opp_upper = gam_raw[game_pos, 4]
        opp_target = gam_raw[game_pos, 5]

        # Dominance round bounds (cols: game_id, R1LB, R1UB, R2LB, R2UB, ...)
        r1_lb = float(dom_rounds.iloc[game_pos, 1])
        r1_ub = float(dom_rounds.iloc[game_pos, 2])

        game_range_width = own_upper - own_lower
        opp_range_width = opp_upper - opp_lower

        rows.append({
            'subject_id': subj_id_int,
            'session': session,
            'subject_idx': subj_idx,
            'game_id': game_id,
            'game_pos': game_pos + 1,
            'guess': guess,
            'eq_pred': eq_pred,
            'l1_pred': l1_pred,
            'l2_pred': l2_pred,
            'l3_pred': l3_pred,
            'd1_pred': d1_pred,
            'd2_pred': d2_pred,
            'own_lower': own_lower,
            'own_upper': own_upper,
            'own_target': own_target,
            'opp_lower': opp_lower,
            'opp_upper': opp_upper,
            'opp_target': opp_target,
            'game_range_width': game_range_width,
            'opp_range_width': opp_range_width,
            'r1_lb': r1_lb,
            'r1_ub': r1_ub,
            'is_baseline': is_baseline,
        })

df = pd.DataFrame(rows)

# Compute deviation measures for each type
for typ in ['eq', 'l1', 'l2', 'l3', 'd1', 'd2']:
    df[f'dev_{typ}'] = df['guess'] - df[f'{typ}_pred']
    df[f'abs_dev_{typ}'] = np.abs(df[f'dev_{typ}'])
    df[f'sq_dev_{typ}'] = df[f'dev_{typ}'] ** 2
    df[f'comply_{typ}'] = (df[f'abs_dev_{typ}'] <= 0.5).astype(int)

# Log absolute deviation (add 1 to avoid log(0))
for typ in ['eq', 'l1', 'l2', 'l3', 'd1', 'd2']:
    df[f'log_abs_dev_{typ}'] = np.log(df[f'abs_dev_{typ}'] + 1)

# Excluded subjects (from GAUSS code: #309, #405, #513)
excluded_ids = [309, 405, 513]
df['excluded'] = df['subject_id'].isin(excluded_ids).astype(int)

# String versions for FE
df['subject_str'] = df['subject_id'].astype(str)
df['game_str'] = df['game_id'].astype(str)
df['session_str'] = df['session'].astype(str)

# Dominance solvable: games where R1 bounds narrow the range significantly
# A game is more "dominance solvable" if the ratio of R1 range to initial range is small
df['dom_range_ratio'] = (df['r1_ub'] - df['r1_lb']) / (df['own_upper'] - df['own_lower'])
df['dom_solvable'] = (df['dom_range_ratio'] < 0.8).astype(int)

# Game position: early (first 8) vs late (last 8) in played order
df['early_game'] = (df['game_pos'] <= 8).astype(int)

print(f"Panel: {len(df)} obs, {df['subject_id'].nunique()} subjects, {df['game_id'].nunique()} games")
print(f"Baseline subjects: {df[df['is_baseline']==1]['subject_id'].nunique()}")
print(f"Open Boxes subjects: {df[df['is_baseline']==0]['subject_id'].nunique()}")
print(f"Mean L1 compliance: {df['comply_l1'].mean():.3f}")
print(f"Mean abs_dev_eq: {df['abs_dev_eq'].mean():.1f}")

# ============================================================
# Define control groups
# ============================================================

GAME_CONTROLS = ['game_range_width', 'own_target']
EXTENDED_CONTROLS = ['game_range_width', 'own_target', 'opp_range_width', 'opp_target', 'dom_range_ratio', 'early_game']

# ============================================================
# Accumulators
# ============================================================

results = []
inference_results = []
spec_run_counter = 0


# ============================================================
# Helper: run_spec (OLS with FE via pyfixest)
# ============================================================

def run_spec(spec_id, spec_tree_path, baseline_group_id,
             outcome_var, treatment_var, controls, fe_formula_str,
             fe_desc, data, vcov, sample_desc, controls_desc,
             cluster_var="subject_str",
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
                       "method": "cluster", "cluster_vars": ["subject"]},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"panel_ols": design_audit},
            axis_block_name=axis_block_name,
            axis_block=axis_block,
        )

        results.append({
            "paper_id": PAPER_ID,
            "spec_run_id": run_id,
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
            "outcome_var": outcome_var,
            "treatment_var": treatment_var,
            "controls_desc": controls_desc,
            "fe_desc": fe_desc,
            "sample_desc": sample_desc,
            "cluster_var": cluster_var,
            "notes": notes,
            "run_success": 1,
            "run_error": ""
        })
        print(f"  [{spec_run_counter:03d}] {spec_id}: coef={coef_val:.4f}, se={se_val:.4f}, p={pval:.4f}, N={nobs}")

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
            "coefficient": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": np.nan,
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "outcome_var": outcome_var,
            "treatment_var": treatment_var,
            "controls_desc": controls_desc,
            "fe_desc": fe_desc,
            "sample_desc": sample_desc,
            "cluster_var": cluster_var,
            "notes": notes,
            "run_success": 0,
            "run_error": err_msg
        })
        print(f"  [{spec_run_counter:03d}] {spec_id}: FAILED - {err_msg[:80]}")


# ============================================================
# BASELINE SPECIFICATION
# ============================================================

print("\n=== BASELINE ===")
# abs_dev_eq ~ comply_l1 + game_controls | subject + game, cluster(subject)
df_base = df[df['excluded'] == 0].copy()
print(f"Baseline sample (excl. 3 flagged subjects): {len(df_base)} obs, {df_base['subject_id'].nunique()} subjects")

run_spec(
    "baseline",
    "baseline", "G1",
    "abs_dev_eq", "comply_l1", GAME_CONTROLS,
    "subject_str + game_str", "subject + game FE", df_base,
    {"CRV1": "subject_str"},
    f"All subjects excl. flagged (N_subj={df_base['subject_id'].nunique()})", "game range + own target",
    axis_block_name=None, axis_block=None)


# ============================================================
# RC: ALTERNATIVE TYPE COMPLIANCE INDICATORS (treatment)
# ============================================================

print("\n=== ALTERNATIVE TYPE INDICATORS ===")

for typ in ['eq', 'l2', 'l3', 'd1', 'd2']:
    run_spec(
        f"rc/treatment/type/{typ}_comply",
        "modules/robustness/treatment.md", "G1",
        "abs_dev_eq", f"comply_{typ}", GAME_CONTROLS,
        "subject_str + game_str", "subject + game FE", df_base,
        {"CRV1": "subject_str"},
        "All subjects excl. flagged", "game range + own target",
        axis_block_name="treatment",
        axis_block={"spec_id": f"rc/treatment/type/{typ}_comply", "type": typ})


# ============================================================
# RC: ALTERNATIVE OUTCOME MEASURES
# ============================================================

print("\n=== ALTERNATIVE OUTCOMES ===")

# Absolute deviation from different type predictions
for typ in ['l1', 'l2', 'd1']:
    run_spec(
        f"rc/outcome/abs_dev_{typ}",
        "modules/robustness/outcome.md", "G1",
        f"abs_dev_{typ}", "comply_l1", GAME_CONTROLS,
        "subject_str + game_str", "subject + game FE", df_base,
        {"CRV1": "subject_str"},
        "All subjects excl. flagged", "game range + own target",
        axis_block_name="outcome",
        axis_block={"spec_id": f"rc/outcome/abs_dev_{typ}", "outcome": f"abs_dev_{typ}"})

# Squared deviation from equilibrium
run_spec(
    "rc/outcome/sq_dev_eq",
    "modules/robustness/outcome.md", "G1",
    "sq_dev_eq", "comply_l1", GAME_CONTROLS,
    "subject_str + game_str", "subject + game FE", df_base,
    {"CRV1": "subject_str"},
    "All subjects excl. flagged", "game range + own target",
    axis_block_name="outcome",
    axis_block={"spec_id": "rc/outcome/sq_dev_eq", "outcome": "sq_dev_eq"})

# Compliance with equilibrium as binary outcome
run_spec(
    "rc/outcome/comply_eq",
    "modules/robustness/outcome.md", "G1",
    "comply_eq", "comply_l1", GAME_CONTROLS,
    "subject_str + game_str", "subject + game FE", df_base,
    {"CRV1": "subject_str"},
    "All subjects excl. flagged", "game range + own target",
    axis_block_name="outcome",
    axis_block={"spec_id": "rc/outcome/comply_eq", "outcome": "comply_eq"})

# Log absolute deviation
run_spec(
    "rc/outcome/log_abs_dev_eq",
    "modules/robustness/outcome.md", "G1",
    "log_abs_dev_eq", "comply_l1", GAME_CONTROLS,
    "subject_str + game_str", "subject + game FE", df_base,
    {"CRV1": "subject_str"},
    "All subjects excl. flagged", "game range + own target",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/outcome/log_abs_dev_eq", "outcome": "log_abs_dev_eq"})

# Raw signed deviation
run_spec(
    "rc/outcome/dev_eq_signed",
    "modules/robustness/outcome.md", "G1",
    "dev_eq", "comply_l1", GAME_CONTROLS,
    "subject_str + game_str", "subject + game FE", df_base,
    {"CRV1": "subject_str"},
    "All subjects excl. flagged", "game range + own target",
    axis_block_name="outcome",
    axis_block={"spec_id": "rc/outcome/dev_eq_signed", "outcome": "dev_eq"})


# ============================================================
# RC: CONTROL PROGRESSION (with subject-only FE, where controls matter)
# Note: With subject + game FE, game-level controls are absorbed.
# We use subject-only FE for control progression, where controls
# capture game-level variation that game FE would otherwise absorb.
# ============================================================

print("\n=== CONTROL PROGRESSION (subject FE only) ===")

# No controls, subject FE only
run_spec(
    "rc/controls/progression/subj_fe_no_controls",
    "modules/robustness/controls.md#control-progression-build-up", "G1",
    "abs_dev_eq", "comply_l1", [],
    "subject_str", "subject FE only", df_base,
    {"CRV1": "subject_str"},
    "All subjects excl. flagged", "no game controls, subject FE only",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/subj_fe_no_controls", "family": "progression",
                "n_controls": 0, "set_name": "none"})

# Target only, subject FE
run_spec(
    "rc/controls/progression/subj_fe_target",
    "modules/robustness/controls.md#control-progression-build-up", "G1",
    "abs_dev_eq", "comply_l1", ["own_target"],
    "subject_str", "subject FE only", df_base,
    {"CRV1": "subject_str"},
    "All subjects excl. flagged", "own target, subject FE",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/subj_fe_target", "family": "progression",
                "n_controls": 1, "set_name": "target_only"})

# Target + range, subject FE
run_spec(
    "rc/controls/progression/subj_fe_target_range",
    "modules/robustness/controls.md#control-progression-build-up", "G1",
    "abs_dev_eq", "comply_l1", GAME_CONTROLS,
    "subject_str", "subject FE only", df_base,
    {"CRV1": "subject_str"},
    "All subjects excl. flagged", "game range + own target, subject FE",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/subj_fe_target_range", "family": "progression",
                "n_controls": 2, "set_name": "target_range"})

# Full game controls, subject FE
run_spec(
    "rc/controls/sets/subj_fe_full",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "abs_dev_eq", "comply_l1", EXTENDED_CONTROLS,
    "subject_str", "subject FE only", df_base,
    {"CRV1": "subject_str"},
    "All subjects excl. flagged", "all game controls (6), subject FE",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/subj_fe_full", "family": "sets",
                "n_controls": len(EXTENDED_CONTROLS), "set_name": "full_subj_fe"})

# No FE: build-up with controls
run_spec(
    "rc/controls/progression/no_fe_raw",
    "modules/robustness/controls.md#control-progression-build-up", "G1",
    "abs_dev_eq", "comply_l1", [],
    "", "none (pooled OLS)", df_base,
    {"CRV1": "subject_str"},
    "All subjects excl. flagged", "raw (no controls, no FE)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/no_fe_raw", "family": "progression",
                "n_controls": 0, "set_name": "raw"})

# No FE: target only
run_spec(
    "rc/controls/progression/no_fe_target",
    "modules/robustness/controls.md#control-progression-build-up", "G1",
    "abs_dev_eq", "comply_l1", ["own_target"],
    "", "none (pooled OLS)", df_base,
    {"CRV1": "subject_str"},
    "All subjects excl. flagged", "own target, no FE",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/no_fe_target", "family": "progression",
                "n_controls": 1, "set_name": "target_no_fe"})

# No FE: full game controls
run_spec(
    "rc/controls/progression/no_fe_full",
    "modules/robustness/controls.md#control-progression-build-up", "G1",
    "abs_dev_eq", "comply_l1", EXTENDED_CONTROLS,
    "", "none (pooled OLS)", df_base,
    {"CRV1": "subject_str"},
    "All subjects excl. flagged", "all game controls (6), no FE",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/no_fe_full", "family": "progression",
                "n_controls": len(EXTENDED_CONTROLS), "set_name": "full_no_fe"})

# Session FE: control progression
run_spec(
    "rc/controls/sets/session_fe_no_controls",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "abs_dev_eq", "comply_l1", [],
    "session_str", "session FE only", df_base,
    {"CRV1": "subject_str"},
    "All subjects excl. flagged", "no controls, session FE",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/session_fe_no_controls", "family": "sets",
                "n_controls": 0, "set_name": "session_fe_none"})

run_spec(
    "rc/controls/sets/session_fe_full",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "abs_dev_eq", "comply_l1", EXTENDED_CONTROLS,
    "session_str", "session FE only", df_base,
    {"CRV1": "subject_str"},
    "All subjects excl. flagged", "all game controls, session FE",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/session_fe_full", "family": "sets",
                "n_controls": len(EXTENDED_CONTROLS), "set_name": "session_fe_full"})

# Controls LOO with subject FE only (where controls have variation)
print("\n=== CONTROLS LOO (subject FE only) ===")

for ctrl in EXTENDED_CONTROLS:
    remaining = [c for c in EXTENDED_CONTROLS if c != ctrl]
    run_spec(
        f"rc/controls/loo/drop_{ctrl}",
        "modules/robustness/controls.md#loo", "G1",
        "abs_dev_eq", "comply_l1", remaining,
        "subject_str", "subject FE only", df_base,
        {"CRV1": "subject_str"},
        "All subjects excl. flagged", f"drop {ctrl}, subject FE",
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/controls/loo/drop_{ctrl}", "family": "loo",
                    "dropped": ctrl, "n_controls": len(remaining)})


# ============================================================
# RC: SAMPLE RESTRICTIONS
# ============================================================

print("\n=== SAMPLE RESTRICTIONS ===")

# Baseline subjects only (71 subjects)
df_baseline = df_base[df_base['is_baseline'] == 1].copy()
run_spec(
    "rc/sample/baseline_only",
    "modules/robustness/sample.md#treatment-groups", "G1",
    "abs_dev_eq", "comply_l1", GAME_CONTROLS,
    "subject_str + game_str", "subject + game FE", df_baseline,
    {"CRV1": "subject_str"},
    f"Baseline treatment only (N_subj={df_baseline['subject_id'].nunique()})", "game range + own target",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/baseline_only", "axis": "treatment_group",
                "group": "baseline", "n_obs": len(df_baseline)})

# Open Boxes subjects only (17 subjects)
df_openbox = df_base[df_base['is_baseline'] == 0].copy()
run_spec(
    "rc/sample/openboxes_only",
    "modules/robustness/sample.md#treatment-groups", "G1",
    "abs_dev_eq", "comply_l1", GAME_CONTROLS,
    "subject_str + game_str", "subject + game FE", df_openbox,
    {"CRV1": "subject_str"},
    f"Open Boxes treatment only (N_subj={df_openbox['subject_id'].nunique()})", "game range + own target",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/openboxes_only", "axis": "treatment_group",
                "group": "openboxes", "n_obs": len(df_openbox)})

# Include flagged subjects (full sample)
run_spec(
    "rc/sample/full_incl_flagged",
    "modules/robustness/sample.md#full", "G1",
    "abs_dev_eq", "comply_l1", GAME_CONTROLS,
    "subject_str + game_str", "subject + game FE", df,
    {"CRV1": "subject_str"},
    f"Full sample incl. flagged (N_subj={df['subject_id'].nunique()})", "game range + own target",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/full_incl_flagged", "axis": "inclusion",
                "n_obs": len(df)})

# First 8 games (early games in master order)
df_early = df_base[df_base['early_game'] == 1].copy()
run_spec(
    "rc/sample/first_8_games",
    "modules/robustness/sample.md#game-split", "G1",
    "abs_dev_eq", "comply_l1", GAME_CONTROLS,
    "subject_str + game_str", "subject + game FE", df_early,
    {"CRV1": "subject_str"},
    f"First 8 games (N_obs={len(df_early)})", "game range + own target",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/first_8_games", "axis": "game_split",
                "split": "first_8", "n_obs": len(df_early)})

# Last 8 games
df_late = df_base[df_base['early_game'] == 0].copy()
run_spec(
    "rc/sample/last_8_games",
    "modules/robustness/sample.md#game-split", "G1",
    "abs_dev_eq", "comply_l1", GAME_CONTROLS,
    "subject_str + game_str", "subject + game FE", df_late,
    {"CRV1": "subject_str"},
    f"Last 8 games (N_obs={len(df_late)})", "game range + own target",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/last_8_games", "axis": "game_split",
                "split": "last_8", "n_obs": len(df_late)})

# Dominance-solvable games only
df_dom = df_base[df_base['dom_solvable'] == 1].copy()
if len(df_dom) > 50:
    run_spec(
        "rc/sample/dominance_solvable",
        "modules/robustness/sample.md#game-type", "G1",
        "abs_dev_eq", "comply_l1", GAME_CONTROLS,
        "subject_str + game_str", "subject + game FE", df_dom,
        {"CRV1": "subject_str"},
        f"Dominance-solvable games (N_obs={len(df_dom)})", "game range + own target",
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/dominance_solvable", "axis": "game_type",
                    "type": "dominance_solvable", "n_obs": len(df_dom)})

# Non-dominance-solvable games
df_nondom = df_base[df_base['dom_solvable'] == 0].copy()
if len(df_nondom) > 50:
    run_spec(
        "rc/sample/non_dominance_solvable",
        "modules/robustness/sample.md#game-type", "G1",
        "abs_dev_eq", "comply_l1", GAME_CONTROLS,
        "subject_str + game_str", "subject + game FE", df_nondom,
        {"CRV1": "subject_str"},
        f"Non-dominance-solvable games (N_obs={len(df_nondom)})", "game range + own target",
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/non_dominance_solvable", "axis": "game_type",
                    "type": "non_dominance_solvable", "n_obs": len(df_nondom)})

# Trim outcome at 1st/99th percentile
q01 = df_base['abs_dev_eq'].quantile(0.01)
q99 = df_base['abs_dev_eq'].quantile(0.99)
df_trim = df_base[(df_base['abs_dev_eq'] >= q01) & (df_base['abs_dev_eq'] <= q99)].copy()
run_spec(
    "rc/sample/trim_y_1_99",
    "modules/robustness/sample.md#outliers", "G1",
    "abs_dev_eq", "comply_l1", GAME_CONTROLS,
    "subject_str + game_str", "subject + game FE", df_trim,
    {"CRV1": "subject_str"},
    f"Trim abs_dev_eq [1%,99%] (N_obs={len(df_trim)})", "game range + own target",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/trim_y_1_99", "axis": "outliers",
                "rule": "trim", "n_obs_before": len(df_base), "n_obs_after": len(df_trim)})

# Trim at 5th/95th
q05 = df_base['abs_dev_eq'].quantile(0.05)
q95 = df_base['abs_dev_eq'].quantile(0.95)
df_trim5 = df_base[(df_base['abs_dev_eq'] >= q05) & (df_base['abs_dev_eq'] <= q95)].copy()
run_spec(
    "rc/sample/trim_y_5_95",
    "modules/robustness/sample.md#outliers", "G1",
    "abs_dev_eq", "comply_l1", GAME_CONTROLS,
    "subject_str + game_str", "subject + game FE", df_trim5,
    {"CRV1": "subject_str"},
    f"Trim abs_dev_eq [5%,95%] (N_obs={len(df_trim5)})", "game range + own target",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/trim_y_5_95", "axis": "outliers",
                "rule": "trim", "n_obs_before": len(df_base), "n_obs_after": len(df_trim5)})

# Per-session subsamples
for sess in sorted(df_base['session'].unique()):
    df_sess = df_base[df_base['session'] == sess].copy()
    if df_sess['subject_id'].nunique() >= 5:
        run_spec(
            f"rc/sample/session_{sess}",
            "modules/robustness/sample.md#session-split", "G1",
            "abs_dev_eq", "comply_l1", GAME_CONTROLS,
            "subject_str + game_str", "subject + game FE", df_sess,
            {"CRV1": "subject_str"},
            f"Session {sess} only (N_subj={df_sess['subject_id'].nunique()})", "game range + own target",
            axis_block_name="sample",
            axis_block={"spec_id": f"rc/sample/session_{sess}", "axis": "session_split",
                        "session": int(sess), "n_obs": int(len(df_sess))})


# ============================================================
# RC: FIXED EFFECTS VARIANTS
# ============================================================

print("\n=== FIXED EFFECTS VARIANTS ===")

# Subject FE only (no game FE)
run_spec(
    "rc/fe/subject_only",
    "modules/robustness/fixed_effects.md#drop-fe", "G1",
    "abs_dev_eq", "comply_l1", GAME_CONTROLS,
    "subject_str", "subject FE only", df_base,
    {"CRV1": "subject_str"},
    "All subjects excl. flagged", "game range + own target",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/subject_only", "family": "drop",
                "dropped": ["game"], "new_fe": ["subject"]})

# Game FE only (no subject FE)
run_spec(
    "rc/fe/game_only",
    "modules/robustness/fixed_effects.md#drop-fe", "G1",
    "abs_dev_eq", "comply_l1", GAME_CONTROLS,
    "game_str", "game FE only", df_base,
    {"CRV1": "subject_str"},
    "All subjects excl. flagged", "game range + own target",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/game_only", "family": "drop",
                "dropped": ["subject"], "new_fe": ["game"]})

# Session FE only
run_spec(
    "rc/fe/session_only",
    "modules/robustness/fixed_effects.md#swap-fe", "G1",
    "abs_dev_eq", "comply_l1", GAME_CONTROLS,
    "session_str", "session FE only", df_base,
    {"CRV1": "subject_str"},
    "All subjects excl. flagged", "game range + own target",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/session_only", "family": "swap",
                "dropped": ["subject", "game"], "new_fe": ["session"]})

# No FE (pooled OLS)
run_spec(
    "rc/fe/no_fe",
    "modules/robustness/fixed_effects.md#drop-all", "G1",
    "abs_dev_eq", "comply_l1", GAME_CONTROLS,
    "", "none (pooled OLS)", df_base,
    {"CRV1": "subject_str"},
    "All subjects excl. flagged", "game range + own target",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/no_fe", "family": "drop",
                "dropped": ["subject", "game"], "new_fe": []})

# Session + game FE
run_spec(
    "rc/fe/session_plus_game",
    "modules/robustness/fixed_effects.md#swap-fe", "G1",
    "abs_dev_eq", "comply_l1", GAME_CONTROLS,
    "session_str + game_str", "session + game FE", df_base,
    {"CRV1": "subject_str"},
    "All subjects excl. flagged", "game range + own target",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/session_plus_game", "family": "swap",
                "dropped": ["subject"], "new_fe": ["session", "game"]})


# ============================================================
# RC: FUNCTIONAL FORM VARIANTS
# ============================================================

print("\n=== FUNCTIONAL FORM ===")

# Log outcome (already done above as rc/outcome, add another with extended controls)
run_spec(
    "rc/form/log_abs_dev_full",
    "modules/robustness/functional_form.md#log-transform", "G1",
    "log_abs_dev_eq", "comply_l1", EXTENDED_CONTROLS,
    "subject_str + game_str", "subject + game FE", df_base,
    {"CRV1": "subject_str"},
    "All subjects excl. flagged", "log outcome + full controls",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/log_abs_dev_full", "transform": "log"})

# Normalized deviation (divide by game range width)
df_base['norm_abs_dev_eq'] = df_base['abs_dev_eq'] / df_base['game_range_width']
run_spec(
    "rc/form/normalized_dev",
    "modules/robustness/functional_form.md#normalization", "G1",
    "norm_abs_dev_eq", "comply_l1", ["own_target"],
    "subject_str + game_str", "subject + game FE", df_base,
    {"CRV1": "subject_str"},
    "All subjects excl. flagged", "normalized by game range",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/normalized_dev", "normalization": "game_range"})


# ============================================================
# CROSS-TYPE REGRESSIONS: comply_type_X on abs_dev_type_Y
# ============================================================

print("\n=== CROSS-TYPE REGRESSIONS ===")

# For each type, regress abs_dev of that type on comply of that type
for typ in ['eq', 'l1', 'l2', 'l3', 'd1', 'd2']:
    run_spec(
        f"rc/cross_type/self/{typ}",
        "modules/robustness/treatment.md#self-consistency", "G1",
        f"abs_dev_{typ}", f"comply_{typ}", GAME_CONTROLS,
        "subject_str + game_str", "subject + game FE", df_base,
        {"CRV1": "subject_str"},
        "All subjects excl. flagged", "game range + own target",
        axis_block_name="treatment",
        axis_block={"spec_id": f"rc/cross_type/self/{typ}", "type": typ,
                    "notes": f"Self-consistency: comply_{typ} predicts abs_dev_{typ}"})


# ============================================================
# INFERENCE VARIANTS (on baseline specification)
# ============================================================

print("\n=== INFERENCE VARIANTS ===")

baseline_run_id = f"{PAPER_ID}_run_001"
infer_counter = 0

baseline_controls_str = " + ".join(GAME_CONTROLS)
baseline_formula = f"abs_dev_eq ~ comply_l1 + {baseline_controls_str}"
baseline_fe = "subject_str + game_str"


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
            design={"panel_ols": design_audit},
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
        print(f"  [infer_{infer_counter:03d}] {spec_id}: se={se_val:.4f}, p={pval:.4f}")

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
        print(f"  [infer_{infer_counter:03d}] {spec_id}: FAILED - {err_msg[:80]}")


# HC1 robust (no clustering)
run_inference_variant(
    baseline_run_id, "infer/se/hc/hc1",
    "modules/inference/standard_errors.md#heteroskedasticity-robust", "G1",
    baseline_formula, baseline_fe, df_base, "comply_l1",
    "hetero", "HC1 (robust, no clustering)")

# Cluster by session
run_inference_variant(
    baseline_run_id, "infer/se/cluster/session",
    "modules/inference/standard_errors.md#clustering", "G1",
    baseline_formula, baseline_fe, df_base, "comply_l1",
    {"CRV1": "session_str"}, "cluster(session)")

# Cluster by game
run_inference_variant(
    baseline_run_id, "infer/se/cluster/game",
    "modules/inference/standard_errors.md#clustering", "G1",
    baseline_formula, baseline_fe, df_base, "comply_l1",
    {"CRV1": "game_str"}, "cluster(game)")


# ============================================================
# WRITE OUTPUTS
# ============================================================

print(f"\nWriting outputs...")
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
    base_row = spec_df[spec_df['spec_id'] == 'baseline']
    if len(base_row) > 0:
        print(f"\nBaseline coef on comply_l1: {base_row['coefficient'].values[0]:.6f}")
        print(f"Baseline SE: {base_row['std_error'].values[0]:.6f}")
        print(f"Baseline p-value: {base_row['p_value'].values[0]:.6f}")
        print(f"Baseline N: {base_row['n_obs'].values[0]:.0f}")

    print(f"\n=== COEFFICIENT RANGE (successful specs) ===")
    print(f"Min coef: {successful['coefficient'].min():.6f}")
    print(f"Max coef: {successful['coefficient'].max():.6f}")
    print(f"Median coef: {successful['coefficient'].median():.6f}")
    n_sig = (successful['p_value'] < 0.05).sum()
    print(f"Significant at 5%: {n_sig}/{len(successful)}")
    n_sig10 = (successful['p_value'] < 0.10).sum()
    print(f"Significant at 10%: {n_sig10}/{len(successful)}")


# ============================================================
# SPECIFICATION_SEARCH.md
# ============================================================

print("\nWriting SPECIFICATION_SEARCH.md...")

md_lines = []
md_lines.append("# Specification Search Report: 116248-V1")
md_lines.append("")
md_lines.append("**Paper:** Costa-Gomes & Crawford (2006), \"Cognition and Behavior in Two-Person Guessing Games: An Experimental Study\", AER 96(5)")
md_lines.append("")
md_lines.append("## Baseline Specification")
md_lines.append("")
md_lines.append("- **Design:** Panel OLS (subject x game)")
md_lines.append("- **Outcome:** abs_dev_eq (absolute deviation of guess from equilibrium prediction)")
md_lines.append("- **Treatment:** comply_l1 (binary: guess within 0.5 of L1-type prediction)")
md_lines.append(f"- **Controls:** {len(GAME_CONTROLS)} controls (game range width, own target)")
md_lines.append("- **Fixed effects:** subject + game")
md_lines.append("- **Clustering:** subject")
md_lines.append("")
md_lines.append("### Interpretation")
md_lines.append("")
md_lines.append("The paper classifies experimental subjects into behavioral types (Eq, L1, L2, L3, D1, D2)")
md_lines.append("using MLE on guesses and information search patterns in 16 two-person guessing games.")
md_lines.append("The specification search tests whether L1-type compliance (the modal type) robustly")
md_lines.append("predicts lower deviations from equilibrium predictions across specification choices.")
md_lines.append("")

if len(successful) > 0 and len(base_row) > 0:
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
    "Baseline": successful[successful['spec_id'].str.startswith('baseline')],
    "Alt. Type Indicators": successful[successful['spec_id'].str.startswith('rc/treatment/')],
    "Alt. Outcomes": successful[successful['spec_id'].str.startswith('rc/outcome/')],
    "Controls LOO": successful[successful['spec_id'].str.startswith('rc/controls/loo/')],
    "Controls Sets": successful[successful['spec_id'].str.startswith('rc/controls/sets/')],
    "Controls Progression": successful[successful['spec_id'].str.startswith('rc/controls/progression/')],
    "Controls Subset": successful[successful['spec_id'].str.startswith('rc/controls/subset/')],
    "Sample Restrictions": successful[successful['spec_id'].str.startswith('rc/sample/')],
    "Fixed Effects": successful[successful['spec_id'].str.startswith('rc/fe/')],
    "Functional Form": successful[successful['spec_id'].str.startswith('rc/form/')],
    "Cross-Type": successful[successful['spec_id'].str.startswith('rc/cross_type/')],
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
    n_sig_total = (successful['p_value'] < 0.05).sum()
    pct_sig = n_sig_total / len(successful) * 100
    sign_consistent = ((successful['coefficient'] > 0).sum() == len(successful)) or \
                      ((successful['coefficient'] < 0).sum() == len(successful))
    median_coef = successful['coefficient'].median()
    sign_word = "positive" if median_coef > 0 else "negative"

    md_lines.append(f"- **Sign consistency:** {'All specifications have the same sign' if sign_consistent else 'Mixed signs across specifications'}")
    md_lines.append(f"- **Significance stability:** {n_sig_total}/{len(successful)} ({pct_sig:.1f}%) specifications significant at 5%")
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

print(f"Wrote SPECIFICATION_SEARCH.md")
print("\nDone!")
