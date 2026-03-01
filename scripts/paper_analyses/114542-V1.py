"""
Specification Search Script for Cattaneo, Galiani, Gertler, Martinez & Titiunik (2009)
"Housing, Health and Happiness"
American Economic Journal: Economic Policy, 1(1), 75-105.

Paper ID: 114542-V1

Surface-driven execution:
  - G1: S_shcementfloor ~ dpisofirme (cement floor coverage, household-level)
  - G2: S_satisfloor ~ dpisofirme (satisfaction/mental health, household-level)
  - G3: S_parcount ~ dpisofirme (child health, individual-level)
  - Randomized experiment with cluster-robust SEs at census block level

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
sys.path.insert(0, "scripts")
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash, software_block
)

DATA_DIR = "data/downloads/extracted/114542-V1"
PAPER_ID = "114542-V1"

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

# ============================================================
# Load and prepare HOUSEHOLD data
# ============================================================
hh_raw = pd.read_stata(f"{DATA_DIR}/PisoFirme_AEJPol-20070024_household.dta",
                        convert_categoricals=False)
for col in hh_raw.columns:
    if hh_raw[col].dtype == np.float32:
        hh_raw[col] = hh_raw[col].astype(np.float64)

# Filter to valid cluster obs (matching Stata: idcluster != .)
hh = hh_raw[hh_raw['idcluster'].notna()].copy()

# ----- Missing value imputation (exactly as in Stata do-file) -----
# HH_demog1
HH_demog1 = ['S_HHpeople', 'S_headage', 'S_spouseage', 'S_headeduc', 'S_spouseeduc']
HH_demog2 = ['S_dem1', 'S_dem2', 'S_dem3', 'S_dem4', 'S_dem5', 'S_dem6', 'S_dem7', 'S_dem8']
HH_health = ['S_waterland', 'S_waterhouse', 'S_electricity', 'S_hasanimals',
             'S_animalsinside', 'S_garbage', 'S_washhands']
HH_econ = ['S_incomepc', 'S_assetspc']
HH_social = ['S_cashtransfers', 'S_milkprogram', 'S_foodprogram', 'S_seguropopular']

# Create missingness dummies and replace missing with 0
miss_groups_hh = {
    'HH_demog1': HH_demog1,
    'HH_demog2': HH_demog2,
    'HH_health': HH_health,
    'HH_econ': HH_econ,
}

dmiss_hh_cols = {}
for group_name, group_vars in miss_groups_hh.items():
    for idx, var in enumerate(group_vars, 1):
        dcol = f"dmiss_{group_name}_{idx}"
        hh[dcol] = (hh[var].isna()).astype(float)
        hh[var] = hh[var].fillna(0)
        dmiss_hh_cols.setdefault(group_name, []).append(dcol)

# Special: S_cashtransfers missingness
hh['dmiss_S_cashtransfers'] = (hh['S_cashtransfers'].isna()).astype(float)
hh['S_cashtransfers'] = hh['S_cashtransfers'].fillna(0)

# Define the 4 household models (matching Stata exactly)
HHmodel_1_vars = []
HHmodel_2_vars = HH_demog1 + dmiss_hh_cols['HH_demog1'] + HH_demog2 + dmiss_hh_cols['HH_demog2'] + HH_health + dmiss_hh_cols['HH_health']
HHmodel_3_vars = HHmodel_2_vars + HH_social + ['dmiss_S_cashtransfers']
HHmodel_4_vars = HHmodel_3_vars + HH_econ + dmiss_hh_cols['HH_econ']

HH_MODELS = {
    1: HHmodel_1_vars,
    2: HHmodel_2_vars,
    3: HHmodel_3_vars,
    4: HHmodel_4_vars,
}

# ============================================================
# Load and prepare INDIVIDUAL data
# ============================================================
ind_raw = pd.read_stata(f"{DATA_DIR}/PisoFirme_AEJPol-20070024_individual.dta",
                         convert_categoricals=False)
for col in ind_raw.columns:
    if ind_raw[col].dtype == np.float32:
        ind_raw[col] = ind_raw[col].astype(np.float64)

ind = ind_raw[ind_raw['idcluster'].notna()].copy()

# Individual-level variable groups
CH_demog = ['S_HHpeople', 'S_rooms', 'S_age', 'S_gender', 'S_childma',
            'S_childmaage', 'S_childmaeduc', 'S_childpa', 'S_childpaage', 'S_childpaeduc']
IND_HH_health = ['S_waterland', 'S_waterhouse', 'S_electricity', 'S_hasanimals',
                  'S_animalsinside', 'S_garbage', 'S_washhands']
IND_HH_social = ['S_cashtransfers', 'S_milkprogram', 'S_foodprogram', 'S_seguropopular']
IND_HH_econ = ['S_incomepc', 'S_assetspc']

# dtriage columns
dtriage_cols = [c for c in ind.columns if c.startswith('dtriage')]

# Missing value imputation for individual data
miss_groups_ind = {
    'CH_demog': CH_demog,
    'HH_health': IND_HH_health,
    'HH_econ': IND_HH_econ,
}

dmiss_ind_cols = {}
for group_name, group_vars in miss_groups_ind.items():
    for idx, var in enumerate(group_vars, 1):
        dcol = f"dmiss_{group_name}_{idx}"
        ind[dcol] = (ind[var].isna()).astype(float)
        ind[var] = ind[var].fillna(0)
        dmiss_ind_cols.setdefault(group_name, []).append(dcol)

# Special: S_cashtransfers
ind['dmiss_S_cashtransfers'] = (ind['S_cashtransfers'].isna()).astype(float)
ind['S_cashtransfers'] = ind['S_cashtransfers'].fillna(0)

# Define the 4 individual models (matching Stata exactly)
INmodel_1_vars = []
INmodel_2_vars = CH_demog + dmiss_ind_cols['CH_demog'] + dtriage_cols + IND_HH_health + dmiss_ind_cols['HH_health']
INmodel_3_vars = INmodel_2_vars + IND_HH_social + ['dmiss_S_cashtransfers']
INmodel_4_vars = INmodel_3_vars + IND_HH_econ + dmiss_ind_cols['HH_econ']

IN_MODELS = {
    1: INmodel_1_vars,
    2: INmodel_2_vars,
    3: INmodel_3_vars,
    4: INmodel_4_vars,
}

# ============================================================
# Helper: Run a single specification
# ============================================================
results = []
inference_results = []
spec_run_counter = 0


def run_spec(spec_id, spec_tree_path, baseline_group_id, outcome_var, treatment_var,
             controls, data, vcov, sample_desc, controls_desc, cluster_var="idcluster",
             fixed_effects_str="", fe_formula="",
             axis_block_name=None, axis_block=None, notes="",
             design_audit_override=None):
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    # Get design audit for this baseline group
    bg = [g for g in surface_obj["baseline_groups"] if g["baseline_group_id"] == baseline_group_id][0]
    da = design_audit_override or bg["design_audit"]
    infer_canonical = bg["inference_plan"]["canonical"]

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
            ci_lower = float(ci.loc[treatment_var, ci.columns[0]]) if treatment_var in ci.index else np.nan
            ci_upper = float(ci.loc[treatment_var, ci.columns[1]]) if treatment_var in ci.index else np.nan
        except Exception:
            ci_lower = np.nan
            ci_upper = np.nan

        n_obs = int(m._N)
        r2 = float(m._r2)

        # Build coefficient dict
        coef_dict = {k: float(v) for k, v in m.coef().items()}

        payload = make_success_payload(
            coefficients=coef_dict,
            inference={"spec_id": infer_canonical["spec_id"], "params": infer_canonical["params"]},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={bg["design_code"]: da},
            axis_block_name=axis_block_name,
            axis_block=axis_block,
        )

        row = {
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
            "n_obs": n_obs,
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": fixed_effects_str,
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
            "run_success": 1,
            "run_error": "",
        }
        results.append(row)
        return row

    except Exception as e:
        err_str = str(e)[:200]
        ed = error_details_from_exception(e, stage="estimation")
        payload = make_failure_payload(
            error=err_str,
            error_details=ed,
            inference={"spec_id": infer_canonical["spec_id"], "params": infer_canonical["params"]},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
        )
        row = {
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
            "run_error": err_str,
        }
        results.append(row)
        return row


def run_inference_variant(base_row, infer_spec_id, infer_params, data, controls, fe_formula=""):
    """Re-estimate under alternative inference choice."""
    global spec_run_counter
    spec_run_counter += 1
    infer_run_id = f"{PAPER_ID}_infer_{spec_run_counter:03d}"

    bg = [g for g in surface_obj["baseline_groups"]
          if g["baseline_group_id"] == base_row["baseline_group_id"]][0]

    try:
        outcome_var = base_row["outcome_var"]
        treatment_var = base_row["treatment_var"]
        controls_str = " + ".join(controls) if controls else ""

        if controls_str and fe_formula:
            formula = f"{outcome_var} ~ {treatment_var} + {controls_str} | {fe_formula}"
        elif controls_str:
            formula = f"{outcome_var} ~ {treatment_var} + {controls_str}"
        elif fe_formula:
            formula = f"{outcome_var} ~ {treatment_var} | {fe_formula}"
        else:
            formula = f"{outcome_var} ~ {treatment_var}"

        # Determine vcov and tree path for this variant
        if "cluster" in infer_spec_id:
            cvar = infer_params.get("cluster_var", "idcluster")
            vcov = {"CRV1": cvar}
            cluster_str = cvar
            tree_path = "modules/inference/standard_errors.md#cluster-robust-standard-errors"
        else:
            vcov = "hetero"
            cluster_str = ""
            tree_path = "modules/inference/standard_errors.md#heteroskedasticity-robust-hc"

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

        n_obs = int(m._N)
        r2 = float(m._r2)

        coef_dict = {k: float(v) for k, v in m.coef().items()}
        payload = make_success_payload(
            coefficients=coef_dict,
            inference={"spec_id": infer_spec_id, "params": infer_params},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={bg["design_code"]: bg["design_audit"]},
        )

        irow = {
            "paper_id": PAPER_ID,
            "inference_run_id": infer_run_id,
            "spec_run_id": base_row["spec_run_id"],
            "spec_id": infer_spec_id,
            "spec_tree_path": tree_path,
            "baseline_group_id": base_row["baseline_group_id"],
            "outcome_var": outcome_var,
            "treatment_var": treatment_var,
            "coefficient": coef_val,
            "std_error": se_val,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": n_obs,
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "cluster_var": cluster_str,
            "run_success": 1,
            "run_error": "",
        }
        inference_results.append(irow)

    except Exception as e:
        err_str = str(e)[:200]
        ed = error_details_from_exception(e, stage="inference_variant")
        payload = make_failure_payload(
            error=err_str,
            error_details=ed,
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
        )
        irow = {
            "paper_id": PAPER_ID,
            "inference_run_id": infer_run_id,
            "spec_run_id": base_row["spec_run_id"],
            "spec_id": infer_spec_id,
            "spec_tree_path": tree_path,
            "baseline_group_id": base_row["baseline_group_id"],
            "outcome_var": base_row["outcome_var"],
            "treatment_var": base_row["treatment_var"],
            "coefficient": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": np.nan,
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "cluster_var": "",
            "run_success": 0,
            "run_error": err_str,
        }
        inference_results.append(irow)


# ============================================================
# G1: Cement Floor Coverage (Household-Level)
# ============================================================
print("=" * 60)
print("G1: Cement Floor Coverage")
print("=" * 60)

G1_outcomes = ['S_shcementfloor', 'S_cementfloorkit', 'S_cementfloordin',
               'S_cementfloorbat', 'S_cementfloorbed']
G1_primary = 'S_shcementfloor'
G1_vcov = {"CRV1": "idcluster"}

# --- Baselines: Model 1 for primary outcome ---
base_row_g1 = run_spec(
    spec_id="baseline",
    spec_tree_path="designs/randomized_experiment.md#baseline",
    baseline_group_id="G1",
    outcome_var=G1_primary,
    treatment_var="dpisofirme",
    controls=[],
    data=hh,
    vcov=G1_vcov,
    sample_desc="Full household sample",
    controls_desc="No controls (Model 1)",
)
print(f"  baseline: coef={base_row_g1['coefficient']:.4f}, se={base_row_g1['std_error']:.4f}, N={base_row_g1['n_obs']}")

# Additional baselines: Models 2-4 for primary outcome
for m_num in [2, 3, 4]:
    r = run_spec(
        spec_id=f"baseline__S_shcementfloor_m{m_num}",
        spec_tree_path="designs/randomized_experiment.md#baseline",
        baseline_group_id="G1",
        outcome_var=G1_primary,
        treatment_var="dpisofirme",
        controls=HH_MODELS[m_num],
        data=hh,
        vcov=G1_vcov,
        sample_desc="Full household sample",
        controls_desc=f"Model {m_num} controls",
    )
    print(f"  baseline_m{m_num}: coef={r['coefficient']:.4f}, se={r['std_error']:.4f}, N={r['n_obs']}")

# --- Design variant: diff-in-means (same as Model 1, explicitly labeled) ---
run_spec(
    spec_id="design/randomized_experiment/estimator/diff_in_means",
    spec_tree_path="designs/randomized_experiment.md#a-itt-implementations",
    baseline_group_id="G1",
    outcome_var=G1_primary,
    treatment_var="dpisofirme",
    controls=[],
    data=hh,
    vcov=G1_vcov,
    sample_desc="Full household sample",
    controls_desc="No controls (difference in means)",
)

# --- RC: Control sets (Models 2-4 for primary outcome as RC) ---
for m_num in [2, 3, 4]:
    run_spec(
        spec_id=f"rc/controls/sets/model{m_num}",
        spec_tree_path="modules/robustness/controls.md#predefined-control-sets",
        baseline_group_id="G1",
        outcome_var=G1_primary,
        treatment_var="dpisofirme",
        controls=HH_MODELS[m_num],
        data=hh,
        vcov=G1_vcov,
        sample_desc="Full household sample",
        controls_desc=f"Model {m_num} control set",
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/controls/sets/model{m_num}",
                    "family": "sets", "set_name": f"model{m_num}",
                    "n_controls": len(HH_MODELS[m_num])},
    )

# --- RC: Leave-one-out from Model 4 ---
loo_vars_g1 = ['S_HHpeople', 'S_headage', 'S_spouseage', 'S_headeduc', 'S_spouseeduc',
               'S_waterland', 'S_waterhouse', 'S_electricity', 'S_hasanimals',
               'S_animalsinside', 'S_garbage', 'S_washhands', 'S_incomepc', 'S_assetspc',
               'S_cashtransfers']

for drop_var in loo_vars_g1:
    # Drop from Model 4; also drop associated missingness dummy if present
    loo_controls = [v for v in HHmodel_4_vars if v != drop_var]
    # Also remove the corresponding dmiss column if this is a substantive var
    for group_name, group_vars in miss_groups_hh.items():
        if drop_var in group_vars:
            idx = group_vars.index(drop_var) + 1
            dmiss_col = f"dmiss_{group_name}_{idx}"
            loo_controls = [v for v in loo_controls if v != dmiss_col]

    run_spec(
        spec_id=f"rc/controls/loo/drop_{drop_var}",
        spec_tree_path="modules/robustness/controls.md#leave-one-out-controls-loo",
        baseline_group_id="G1",
        outcome_var=G1_primary,
        treatment_var="dpisofirme",
        controls=loo_controls,
        data=hh,
        vcov=G1_vcov,
        sample_desc="Full household sample",
        controls_desc=f"Model 4 minus {drop_var}",
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/controls/loo/drop_{drop_var}",
                    "family": "loo", "dropped": [drop_var],
                    "n_controls": len(loo_controls)},
    )

# --- RC: Random control subsets (G1 only) ---
rng = np.random.RandomState(114542)
optional_controls_g1 = HH_demog1 + HH_demog2 + HH_health + HH_social + HH_econ
# Also need their missingness dummies
all_dmiss_hh = []
for gn in ['HH_demog1', 'HH_demog2', 'HH_health', 'HH_econ']:
    all_dmiss_hh.extend(dmiss_hh_cols[gn])
all_dmiss_hh.append('dmiss_S_cashtransfers')

for draw_idx in range(1, 11):
    # Draw a random subset size between 5 and 20
    subset_size = rng.randint(5, min(20, len(optional_controls_g1)) + 1)
    chosen_idx = rng.choice(len(optional_controls_g1), size=subset_size, replace=False)
    chosen_vars = [optional_controls_g1[i] for i in chosen_idx]

    # Add corresponding missingness dummies
    chosen_with_dmiss = list(chosen_vars)
    for var in chosen_vars:
        for group_name, group_vars in miss_groups_hh.items():
            if var in group_vars:
                idx_in_group = group_vars.index(var) + 1
                dcol = f"dmiss_{group_name}_{idx_in_group}"
                if dcol not in chosen_with_dmiss:
                    chosen_with_dmiss.append(dcol)
        if var == 'S_cashtransfers' and 'dmiss_S_cashtransfers' not in chosen_with_dmiss:
            chosen_with_dmiss.append('dmiss_S_cashtransfers')

    run_spec(
        spec_id=f"rc/controls/subset/random_{draw_idx:03d}",
        spec_tree_path="modules/robustness/controls.md#random-control-subsets",
        baseline_group_id="G1",
        outcome_var=G1_primary,
        treatment_var="dpisofirme",
        controls=chosen_with_dmiss,
        data=hh,
        vcov=G1_vcov,
        sample_desc="Full household sample",
        controls_desc=f"Random subset draw {draw_idx} ({len(chosen_vars)} substantive controls)",
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/controls/subset/random_{draw_idx:03d}",
                    "family": "subset", "draw_index": draw_idx,
                    "seed": 114542, "chosen": chosen_vars,
                    "n_controls": len(chosen_with_dmiss)},
    )

# --- RC: Sample trimming for G1 ---
for trim_lo, trim_hi, label in [(1, 99, "trim_y_1_99"), (5, 95, "trim_y_5_95")]:
    lo_pct = np.nanpercentile(hh[G1_primary], trim_lo)
    hi_pct = np.nanpercentile(hh[G1_primary], trim_hi)
    trimmed = hh[(hh[G1_primary] >= lo_pct) & (hh[G1_primary] <= hi_pct)]
    run_spec(
        spec_id=f"rc/sample/outliers/{label}",
        spec_tree_path="modules/robustness/sample.md#outlier-trimming",
        baseline_group_id="G1",
        outcome_var=G1_primary,
        treatment_var="dpisofirme",
        controls=[],
        data=trimmed,
        vcov=G1_vcov,
        sample_desc=f"Trimmed {trim_lo}/{trim_hi} pctile on {G1_primary}",
        controls_desc="No controls (Model 1)",
        axis_block_name="sample",
        axis_block={"spec_id": f"rc/sample/outliers/{label}",
                    "family": "outliers", "lo_pct": trim_lo, "hi_pct": trim_hi},
    )

# --- Inference variants for G1 baseline ---
# HC1 (no clustering)
run_inference_variant(base_row_g1, "infer/se/hc/hc1", {}, hh, [], "")
# Cluster at municipality
run_inference_variant(base_row_g1, "infer/se/cluster/idmun", {"cluster_var": "idmun"}, hh, [], "")


# ============================================================
# G2: Satisfaction and Mental Health (Household-Level)
# ============================================================
print("=" * 60)
print("G2: Satisfaction and Mental Health")
print("=" * 60)

G2_outcomes = {
    'S_satisfloor': 'Satisfaction with floor quality',
    'S_satishouse': 'Satisfaction with house quality',
    'S_satislife': 'Satisfaction with quality of life',
    'S_cesds': 'Depression Scale (CES-D)',
    'S_pss': 'Perceived Stress Scale',
}

# Baseline for primary outcome (S_satisfloor, Model 1)
base_row_g2 = run_spec(
    spec_id="baseline",
    spec_tree_path="designs/randomized_experiment.md#baseline",
    baseline_group_id="G2",
    outcome_var="S_satisfloor",
    treatment_var="dpisofirme",
    controls=[],
    data=hh,
    vcov=G1_vcov,
    sample_desc="Full household sample",
    controls_desc="No controls (Model 1)",
)
print(f"  baseline S_satisfloor: coef={base_row_g2['coefficient']:.4f}, se={base_row_g2['std_error']:.4f}")

# Additional baselines for other G2 outcomes
for outvar in ['S_satishouse', 'S_satislife', 'S_cesds', 'S_pss']:
    r = run_spec(
        spec_id=f"baseline__{outvar}",
        spec_tree_path="designs/randomized_experiment.md#baseline",
        baseline_group_id="G2",
        outcome_var=outvar,
        treatment_var="dpisofirme",
        controls=[],
        data=hh,
        vcov=G1_vcov,
        sample_desc="Full household sample",
        controls_desc="No controls (Model 1)",
    )
    print(f"  baseline {outvar}: coef={r['coefficient']:.4f}, se={r['std_error']:.4f}")

# RC: Control sets for each G2 outcome
for outvar in G2_outcomes:
    for m_num in [2, 3, 4]:
        run_spec(
            spec_id=f"rc/controls/sets/model{m_num}",
            spec_tree_path="modules/robustness/controls.md#predefined-control-sets",
            baseline_group_id="G2",
            outcome_var=outvar,
            treatment_var="dpisofirme",
            controls=HH_MODELS[m_num],
            data=hh,
            vcov=G1_vcov,
            sample_desc="Full household sample",
            controls_desc=f"{outvar} with Model {m_num} controls",
            axis_block_name="controls",
            axis_block={"spec_id": f"rc/controls/sets/model{m_num}",
                        "family": "sets", "set_name": f"model{m_num}",
                        "n_controls": len(HH_MODELS[m_num])},
        )

# RC: Leave-one-out for G2 (selected vars, primary outcome S_satisfloor only)
loo_vars_g2 = ['S_HHpeople', 'S_headage', 'S_waterland', 'S_hasanimals',
               'S_incomepc', 'S_cashtransfers']
for drop_var in loo_vars_g2:
    loo_controls = [v for v in HHmodel_4_vars if v != drop_var]
    for group_name, group_vars in miss_groups_hh.items():
        if drop_var in group_vars:
            idx = group_vars.index(drop_var) + 1
            dmiss_col = f"dmiss_{group_name}_{idx}"
            loo_controls = [v for v in loo_controls if v != dmiss_col]

    run_spec(
        spec_id=f"rc/controls/loo/drop_{drop_var}",
        spec_tree_path="modules/robustness/controls.md#leave-one-out-controls-loo",
        baseline_group_id="G2",
        outcome_var="S_satisfloor",
        treatment_var="dpisofirme",
        controls=loo_controls,
        data=hh,
        vcov=G1_vcov,
        sample_desc="Full household sample",
        controls_desc=f"Model 4 minus {drop_var}",
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/controls/loo/drop_{drop_var}",
                    "family": "loo", "dropped": [drop_var],
                    "n_controls": len(loo_controls)},
    )

# RC: Sample trimming for G2
for outvar in ['S_satisfloor', 'S_cesds']:
    for trim_lo, trim_hi, label in [(1, 99, "trim_y_1_99"), (5, 95, "trim_y_5_95")]:
        lo_pct = np.nanpercentile(hh[outvar], trim_lo)
        hi_pct = np.nanpercentile(hh[outvar], trim_hi)
        trimmed = hh[(hh[outvar] >= lo_pct) & (hh[outvar] <= hi_pct)]
        run_spec(
            spec_id=f"rc/sample/outliers/{label}",
            spec_tree_path="modules/robustness/sample.md#outlier-trimming",
            baseline_group_id="G2",
            outcome_var=outvar,
            treatment_var="dpisofirme",
            controls=[],
            data=trimmed,
            vcov=G1_vcov,
            sample_desc=f"Trimmed {trim_lo}/{trim_hi} pctile on {outvar}",
            controls_desc="No controls (Model 1)",
            axis_block_name="sample",
            axis_block={"spec_id": f"rc/sample/outliers/{label}",
                        "family": "outliers", "lo_pct": trim_lo, "hi_pct": trim_hi},
        )

# Inference variants for G2 baseline
run_inference_variant(base_row_g2, "infer/se/hc/hc1", {}, hh, [], "")


# ============================================================
# G3: Child Health (Individual-Level)
# ============================================================
print("=" * 60)
print("G3: Child Health")
print("=" * 60)

G3_outcomes = {
    'S_parcount': 'Parasite count',
    'S_diarrhea': 'Diarrhea',
    'S_anemia': 'Anemia',
    'S_haz': 'Height-for-age z-score',
    'S_whz': 'Weight-for-height z-score',
}

# Baseline for primary outcome (S_parcount, Model 1)
base_row_g3 = run_spec(
    spec_id="baseline",
    spec_tree_path="designs/randomized_experiment.md#baseline",
    baseline_group_id="G3",
    outcome_var="S_parcount",
    treatment_var="dpisofirme",
    controls=[],
    data=ind,
    vcov={"CRV1": "idcluster"},
    sample_desc="Full individual sample",
    controls_desc="No controls (Model 1)",
)
print(f"  baseline S_parcount: coef={base_row_g3['coefficient']:.4f}, se={base_row_g3['std_error']:.4f}, N={base_row_g3['n_obs']}")

# Additional baselines for other G3 outcomes
for outvar in ['S_diarrhea', 'S_anemia', 'S_haz', 'S_whz']:
    r = run_spec(
        spec_id=f"baseline__{outvar}",
        spec_tree_path="designs/randomized_experiment.md#baseline",
        baseline_group_id="G3",
        outcome_var=outvar,
        treatment_var="dpisofirme",
        controls=[],
        data=ind,
        vcov={"CRV1": "idcluster"},
        sample_desc="Full individual sample",
        controls_desc="No controls (Model 1)",
    )
    print(f"  baseline {outvar}: coef={r['coefficient']:.4f}, se={r['std_error']:.4f}, N={r['n_obs']}")

# RC: Control sets for each G3 outcome
for outvar in G3_outcomes:
    for m_num in [2, 3, 4]:
        run_spec(
            spec_id=f"rc/controls/sets/model{m_num}",
            spec_tree_path="modules/robustness/controls.md#predefined-control-sets",
            baseline_group_id="G3",
            outcome_var=outvar,
            treatment_var="dpisofirme",
            controls=IN_MODELS[m_num],
            data=ind,
            vcov={"CRV1": "idcluster"},
            sample_desc="Full individual sample",
            controls_desc=f"{outvar} with Model {m_num} controls",
            axis_block_name="controls",
            axis_block={"spec_id": f"rc/controls/sets/model{m_num}",
                        "family": "sets", "set_name": f"model{m_num}",
                        "n_controls": len(IN_MODELS[m_num])},
        )

# RC: Leave-one-out for G3 (selected vars, primary outcome S_parcount)
loo_vars_g3 = ['S_age', 'S_gender', 'S_childmaeduc', 'S_waterland',
               'S_hasanimals', 'S_incomepc']
for drop_var in loo_vars_g3:
    loo_controls = [v for v in INmodel_4_vars if v != drop_var]
    for group_name, group_vars in miss_groups_ind.items():
        if drop_var in group_vars:
            idx = group_vars.index(drop_var) + 1
            dmiss_col = f"dmiss_{group_name}_{idx}"
            loo_controls = [v for v in loo_controls if v != dmiss_col]

    run_spec(
        spec_id=f"rc/controls/loo/drop_{drop_var}",
        spec_tree_path="modules/robustness/controls.md#leave-one-out-controls-loo",
        baseline_group_id="G3",
        outcome_var="S_parcount",
        treatment_var="dpisofirme",
        controls=loo_controls,
        data=ind,
        vcov={"CRV1": "idcluster"},
        sample_desc="Full individual sample",
        controls_desc=f"Model 4 minus {drop_var}",
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/controls/loo/drop_{drop_var}",
                    "family": "loo", "dropped": [drop_var],
                    "n_controls": len(loo_controls)},
    )

# RC: Sample trimming for G3
for outvar in ['S_parcount', 'S_diarrhea']:
    for trim_lo, trim_hi, label in [(1, 99, "trim_y_1_99"), (5, 95, "trim_y_5_95")]:
        lo_pct = np.nanpercentile(ind[outvar], trim_lo)
        hi_pct = np.nanpercentile(ind[outvar], trim_hi)
        trimmed = ind[(ind[outvar] >= lo_pct) & (ind[outvar] <= hi_pct)]
        run_spec(
            spec_id=f"rc/sample/outliers/{label}",
            spec_tree_path="modules/robustness/sample.md#outlier-trimming",
            baseline_group_id="G3",
            outcome_var=outvar,
            treatment_var="dpisofirme",
            controls=[],
            data=trimmed,
            vcov={"CRV1": "idcluster"},
            sample_desc=f"Trimmed {trim_lo}/{trim_hi} pctile on {outvar}",
            controls_desc="No controls (Model 1)",
            axis_block_name="sample",
            axis_block={"spec_id": f"rc/sample/outliers/{label}",
                        "family": "outliers", "lo_pct": trim_lo, "hi_pct": trim_hi},
        )

# Inference variants for G3 baseline
run_inference_variant(base_row_g3, "infer/se/hc/hc1", {}, ind, [], "")


# ============================================================
# Write outputs
# ============================================================
print("=" * 60)
print("Writing outputs")
print("=" * 60)

# specification_results.csv
df_results = pd.DataFrame(results)
df_results.to_csv(f"{DATA_DIR}/specification_results.csv", index=False)
print(f"  specification_results.csv: {len(df_results)} rows")
print(f"    run_success=1: {(df_results['run_success']==1).sum()}")
print(f"    run_success=0: {(df_results['run_success']==0).sum()}")

# inference_results.csv
df_inference = pd.DataFrame(inference_results)
df_inference.to_csv(f"{DATA_DIR}/inference_results.csv", index=False)
print(f"  inference_results.csv: {len(df_inference)} rows")

# Summary stats
for gid in ['G1', 'G2', 'G3']:
    g_rows = df_results[df_results['baseline_group_id'] == gid]
    print(f"\n  {gid}: {len(g_rows)} specs")
    base_rows = g_rows[g_rows['spec_id'].str.startswith('baseline')]
    rc_rows = g_rows[g_rows['spec_id'].str.startswith('rc/')]
    design_rows = g_rows[g_rows['spec_id'].str.startswith('design/')]
    print(f"    baselines: {len(base_rows)}")
    print(f"    design: {len(design_rows)}")
    print(f"    rc: {len(rc_rows)}")

total_specs = len(df_results)
print(f"\nTotal specifications: {total_specs}")
print(f"Total inference variants: {len(df_inference)}")


# ============================================================
# Write SPECIFICATION_SEARCH.md
# ============================================================
search_md = f"""# Specification Search Log: 114542-V1

**Paper**: Cattaneo, Galiani, Gertler, Martinez & Titiunik (2009). "Housing, Health and Happiness."
**Date**: 2026-02-24

---

## Surface Summary

- **Paper ID**: 114542-V1
- **Design**: Randomized Experiment (Piso Firme program)
- **Baseline groups**: 3 (G1: cement floors, G2: satisfaction/mental health, G3: child health)
- **Surface hash**: {SURFACE_HASH}
- **Seed**: 114542

## Execution Summary

### Total Counts
- **Planned specifications**: {total_specs}
- **Executed successfully**: {(df_results['run_success']==1).sum()}
- **Failed**: {(df_results['run_success']==0).sum()}
- **Inference variants**: {len(df_inference)}

### By Baseline Group

#### G1: Cement Floor Coverage (Household-Level)
- Primary outcome: S_shcementfloor
- Baselines: {len(df_results[(df_results['baseline_group_id']=='G1') & df_results['spec_id'].str.startswith('baseline')])}
- Design variants: {len(df_results[(df_results['baseline_group_id']=='G1') & df_results['spec_id'].str.startswith('design/')])}
- RC specs: {len(df_results[(df_results['baseline_group_id']=='G1') & df_results['spec_id'].str.startswith('rc/')])}
- Inference variants: {len([r for r in inference_results if r['baseline_group_id']=='G1'])}

#### G2: Satisfaction and Mental Health (Household-Level)
- Primary outcome: S_satisfloor
- Additional outcomes: S_satishouse, S_satislife, S_cesds, S_pss
- Baselines: {len(df_results[(df_results['baseline_group_id']=='G2') & df_results['spec_id'].str.startswith('baseline')])}
- RC specs: {len(df_results[(df_results['baseline_group_id']=='G2') & df_results['spec_id'].str.startswith('rc/')])}
- Inference variants: {len([r for r in inference_results if r['baseline_group_id']=='G2'])}

#### G3: Child Health (Individual-Level)
- Primary outcome: S_parcount
- Additional outcomes: S_diarrhea, S_anemia, S_haz, S_whz
- Baselines: {len(df_results[(df_results['baseline_group_id']=='G3') & df_results['spec_id'].str.startswith('baseline')])}
- RC specs: {len(df_results[(df_results['baseline_group_id']=='G3') & df_results['spec_id'].str.startswith('rc/')])}
- Inference variants: {len([r for r in inference_results if r['baseline_group_id']=='G3'])}

## Deviations and Notes

1. **Missing value imputation**: Exactly replicates Stata code -- missing control values replaced with 0 plus indicator dummies for missingness.
2. **Clustering**: All specifications use CRV1 at `idcluster` (census block) level, matching the paper's `cl(idcluster)`.
3. **dtriage dummies**: Individual-level regressions (G3) include trimester-age-gender dummy variables as in the original Stata code.
4. **Cognitive outcomes excluded**: S_mccdts (N=601) and S_pbdypct (N=1,589) have very high missingness and were excluded from the specification surface.
5. **idmun clustering**: Only computed for G1 baseline as an inference variant. Note that with only ~2 municipalities, inference is unreliable.

## Software Stack

- Python {SW_BLOCK['runner_version']}
- pyfixest {SW_BLOCK['packages'].get('pyfixest', 'N/A')}
- pandas {SW_BLOCK['packages'].get('pandas', 'N/A')}
- numpy {SW_BLOCK['packages'].get('numpy', 'N/A')}
"""

with open(f"{DATA_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write(search_md)

print("\nDone. All outputs written.")
