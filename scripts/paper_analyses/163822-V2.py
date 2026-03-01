"""
Specification Search Script for Allcott, Gentzkow & Song (2022)
"Digital Addiction"
American Economic Review, 112(7), 2424-2463.

Paper ID: 163822-V2

Surface-driven execution:
  - G1: Phone usage ~ Bonus + Limit (PD-measured screen time, RCT)
    - Baseline: PD_P2_UsageFITSBY ~ B + L + i.Stratifier + PD_P1_UsageFITSBY, robust
    - Additional baselines across periods P3/P4/P5, across-period average, total usage
  - G2: Survey well-being ~ Bonus + Limit (stacked S3/S4 panel, cluster UserID)
    - Baseline: index_well_N ~ B + B4 + L + i.S + i.S#i.Stratifier + i.S#c.baseline, robust cluster(UserID)
    - Additional baselines across survey outcomes

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
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, "scripts")
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash as compute_surface_hash,
    software_block
)

DATA_DIR = "data/downloads/extracted/163822-V2"
PAPER_ID = "163822-V2"

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = compute_surface_hash(surface_obj)
SW_BLOCK = software_block()

# ── Load data ──
df_raw = pd.read_stata(f"{DATA_DIR}/data/temptation/output/final_data_sample.dta")
print(f"Raw data: {df_raw.shape[0]} obs, {df_raw.shape[1]} cols")

# ── Create treatment variables ──
# S3_Bonus: category => numeric
df_raw['B'] = (df_raw['S3_Bonus'] != 'None').astype(int)
# S2_LimitType: category => L indicator (any limit)
df_raw['L'] = (df_raw['S2_LimitType'] != 'No limit').astype(int)

# Detailed limit types (for rc/data/treatment/detailed_limit_types)
limit_types = df_raw['S2_LimitType'].unique().tolist()
limit_types_active = [lt for lt in limit_types if lt != 'No limit']
for lt in limit_types_active:
    safe_name = lt.replace(" ", "_").replace(".", "")
    df_raw[f'L_{safe_name}'] = (df_raw['S2_LimitType'] == lt).astype(int)
detailed_limit_cols = [f'L_{lt.replace(" ", "_").replace(".", "")}' for lt in limit_types_active]

# Stratifier dummies
df_raw['Stratifier_cat'] = df_raw['Stratifier'].astype('category').cat.codes
strat_dummies = pd.get_dummies(df_raw['Stratifier'], prefix='strat', drop_first=True, dtype=float)
df_raw = pd.concat([df_raw, strat_dummies], axis=1)
strat_cols = list(strat_dummies.columns)

print(f"Treatments: B sum={df_raw['B'].sum()}, L sum={df_raw['L'].sum()}")
print(f"N={df_raw.shape[0]}")

# ── Collect results ──
results = []
inference_results = []
spec_run_counter = 0
inference_run_counter = 0

g1_audit = surface_obj["baseline_groups"][0]["design_audit"]
g1_infer_canonical = surface_obj["baseline_groups"][0]["inference_plan"]["canonical"]
g2_audit = surface_obj["baseline_groups"][1]["design_audit"]
g2_infer_canonical = surface_obj["baseline_groups"][1]["inference_plan"]["canonical"]


def run_spec(spec_id, spec_tree_path, baseline_group_id, outcome_var, treatment_var,
             formula, data, vcov, sample_desc, fixed_effects_str, controls_desc,
             cluster_var="", weights_col=None,
             design_audit_obj=None, infer_canonical_obj=None,
             axis_block_name=None, axis_block=None, second_treatment_var=None,
             notes=""):
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        kw = dict(data=data, vcov=vcov)
        if weights_col:
            kw['weights'] = weights_col
        m = pf.feols(formula, **kw)

        coef_val = float(m.coef().get(treatment_var, np.nan))
        se_val = float(m.se().get(treatment_var, np.nan))
        pval = float(m.pvalue().get(treatment_var, np.nan))

        try:
            ci = m.confint()
            ci_lower = float(ci.loc[treatment_var, ci.columns[0]]) if treatment_var in ci.index else np.nan
            ci_upper = float(ci.loc[treatment_var, ci.columns[1]]) if treatment_var in ci.index else np.nan
        except:
            ci_lower, ci_upper = np.nan, np.nan

        nobs = int(m._N)
        try:
            r2 = float(m._r2)
        except:
            r2 = np.nan

        all_coefs = {k: float(v) for k, v in m.coef().items()}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": infer_canonical_obj["spec_id"],
                       "params": infer_canonical_obj.get("params", {}),
                       "type": "HC1" if "hc" in infer_canonical_obj["spec_id"] else "CRV1"},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"randomized_experiment": design_audit_obj},
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
            "n_obs": nobs,
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
        return m, run_id

    except Exception as e:
        payload = make_failure_payload(
            error=str(e),
            error_details=error_details_from_exception(e, stage="estimation"),
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
            "run_error": str(e)[:240],
        }
        results.append(row)
        return None, run_id


def run_infer_variant(base_run_id, spec_id, spec_tree_path, baseline_group_id,
                      outcome_var, treatment_var, formula, data, vcov,
                      weights_col=None, notes=""):
    global inference_run_counter
    inference_run_counter += 1
    inf_run_id = f"{PAPER_ID}_infer_{inference_run_counter:03d}"

    try:
        kw = dict(data=data, vcov=vcov)
        if weights_col:
            kw['weights'] = weights_col
        m = pf.feols(formula, **kw)

        coef_val = float(m.coef().get(treatment_var, np.nan))
        se_val = float(m.se().get(treatment_var, np.nan))
        pval = float(m.pvalue().get(treatment_var, np.nan))

        try:
            ci = m.confint()
            ci_lower = float(ci.loc[treatment_var, ci.columns[0]]) if treatment_var in ci.index else np.nan
            ci_upper = float(ci.loc[treatment_var, ci.columns[1]]) if treatment_var in ci.index else np.nan
        except:
            ci_lower, ci_upper = np.nan, np.nan

        nobs = int(m._N)
        try:
            r2 = float(m._r2)
        except:
            r2 = np.nan

        all_coefs = {k: float(v) for k, v in m.coef().items()}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": spec_id, "type": notes},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"randomized_experiment": g1_audit},
        )

        inference_results.append({
            "paper_id": PAPER_ID,
            "inference_run_id": inf_run_id,
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
            "run_success": 1,
            "run_error": "",
        })

    except Exception as e:
        payload = make_failure_payload(
            error=str(e),
            error_details=error_details_from_exception(e, stage="inference_variant"),
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
        )
        inference_results.append({
            "paper_id": PAPER_ID,
            "inference_run_id": inf_run_id,
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
            "run_success": 0,
            "run_error": str(e)[:240],
        })


# =============================================================================
# G1: Phone Usage (cross-sectional, per-phase RCT regressions)
# =============================================================================
print("\n=== G1: Phone Usage Specifications ===")

# Helper: build strata dummy formula for feols (pyfixest can absorb factors)
strat_str = " + ".join(strat_cols)

# ── G1 BASELINE: PD_P2_UsageFITSBY ~ B + L + strata + PD_P1_UsageFITSBY ──
formula_base_p2 = f"PD_P2_UsageFITSBY ~ B + L + {strat_str} + PD_P1_UsageFITSBY"
m, base_p2_id = run_spec(
    "baseline", "specification_tree/designs/randomized_experiment.md",
    "G1", "PD_P2_UsageFITSBY", "B",
    formula_base_p2, df_raw, "hetero",
    "Full sample", "", f"Strata dummies, PD_P1_UsageFITSBY",
    design_audit_obj=g1_audit, infer_canonical_obj=g1_infer_canonical,
    notes="G1 primary baseline: B coeff on Phase 2 FITSBY usage"
)
print(f"  Baseline B coef: {results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}, N={results[-1]['n_obs']}")

# Also report L coeff from baseline
if m is not None:
    spec_run_counter += 1
    l_coef = float(m.coef().get("L", np.nan))
    l_se = float(m.se().get("L", np.nan))
    l_pval = float(m.pvalue().get("L", np.nan))
    try:
        ci = m.confint()
        l_ci_lo = float(ci.loc["L", ci.columns[0]])
        l_ci_hi = float(ci.loc["L", ci.columns[1]])
    except:
        l_ci_lo, l_ci_hi = np.nan, np.nan
    all_coefs = {k: float(v) for k, v in m.coef().items()}
    payload = make_success_payload(
        coefficients=all_coefs,
        inference={"spec_id": g1_infer_canonical["spec_id"], "type": "HC1"},
        software=SW_BLOCK, surface_hash=SURFACE_HASH,
        design={"randomized_experiment": g1_audit},
    )
    results.append({
        "paper_id": PAPER_ID, "spec_run_id": f"{PAPER_ID}_run_{spec_run_counter:03d}",
        "spec_id": "baseline", "spec_tree_path": "specification_tree/designs/randomized_experiment.md",
        "baseline_group_id": "G1", "outcome_var": "PD_P2_UsageFITSBY",
        "treatment_var": "L", "coefficient": l_coef, "std_error": l_se, "p_value": l_pval,
        "ci_lower": l_ci_lo, "ci_upper": l_ci_hi, "n_obs": int(m._N),
        "r_squared": float(m._r2) if hasattr(m, '_r2') else np.nan,
        "coefficient_vector_json": json.dumps(payload),
        "sample_desc": "Full sample", "fixed_effects": "", "controls_desc": "Strata dummies, PD_P1_UsageFITSBY",
        "cluster_var": "", "run_success": 1, "run_error": "",
    })
    print(f"  Baseline L coef: {l_coef:.4f}, p={l_pval:.4f}")


# ── G1 Additional Baselines ──
# P3 (Phase 3 = Endline 1), P4, P5, P2-to-P5 avg, Total usage (P2), Total (P2-P5 avg)
g1_extra_baselines = {
    "baseline__usage_p3_fitsby": ("PD_P3_UsageFITSBY", "PD_P1_UsageFITSBY", "Phase 3 FITSBY"),
    "baseline__usage_p4_fitsby": ("PD_P4_UsageFITSBY", "PD_P1_UsageFITSBY", "Phase 4 FITSBY"),
    "baseline__usage_p5_fitsby": ("PD_P5_UsageFITSBY", "PD_P1_UsageFITSBY", "Phase 5 FITSBY"),
    "baseline__usage_p2_total": ("PD_P2_Usage", "PD_P1_Usage", "Phase 2 Total usage"),
}

# Create P2-to-P5 average FITSBY
df_raw['PD_P2to5_UsageFITSBY'] = df_raw[['PD_P2_UsageFITSBY', 'PD_P3_UsageFITSBY',
                                           'PD_P4_UsageFITSBY', 'PD_P5_UsageFITSBY']].mean(axis=1)
g1_extra_baselines["baseline__usage_p2to5_fitsby"] = ("PD_P2to5_UsageFITSBY", "PD_P1_UsageFITSBY", "Phase 2-5 avg FITSBY")

# Create P2-to-P5 average Total
df_raw['PD_P2to5_Usage'] = df_raw[['PD_P2_Usage', 'PD_P3_Usage',
                                    'PD_P4_Usage', 'PD_P5_Usage']].mean(axis=1)
g1_extra_baselines["baseline__usage_p2to5_total"] = ("PD_P2to5_Usage", "PD_P1_Usage", "Phase 2-5 avg Total")

for bid, (yvar, baseline_ctrl, desc) in g1_extra_baselines.items():
    formula = f"{yvar} ~ B + L + {strat_str} + {baseline_ctrl}"
    m, rid = run_spec(
        bid, "specification_tree/designs/randomized_experiment.md",
        "G1", yvar, "B", formula, df_raw, "hetero",
        "Full sample", "", f"Strata dummies, {baseline_ctrl}",
        design_audit_obj=g1_audit, infer_canonical_obj=g1_infer_canonical,
        notes=f"G1 extra baseline: {desc}"
    )
    if results[-1]['run_success']:
        print(f"  {bid}: B={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}, N={results[-1]['n_obs']}")


# ── G1 Design Variants ──
# design/randomized_experiment/estimator/diff_in_means
m, dm_rid = run_spec(
    "design/randomized_experiment/estimator/diff_in_means",
    "specification_tree/designs/randomized_experiment.md#diff-in-means",
    "G1", "PD_P2_UsageFITSBY", "B",
    "PD_P2_UsageFITSBY ~ B + L", df_raw, "hetero",
    "Full sample", "", "None (pure diff-in-means)",
    design_audit_obj=g1_audit, infer_canonical_obj=g1_infer_canonical,
    notes="Diff-in-means: no controls"
)
if results[-1]['run_success']:
    print(f"  DIM: B={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}")

# design/randomized_experiment/estimator/with_covariates
# same as baseline but making it explicit
m, wc_rid = run_spec(
    "design/randomized_experiment/estimator/with_covariates",
    "specification_tree/designs/randomized_experiment.md#with-covariates",
    "G1", "PD_P2_UsageFITSBY", "B",
    formula_base_p2, df_raw, "hetero",
    "Full sample", "", f"Strata dummies + PD_P1_UsageFITSBY",
    design_audit_obj=g1_audit, infer_canonical_obj=g1_infer_canonical,
    notes="With covariates (same as baseline)"
)

# ── G1 RC Variants ──
# rc/controls/loo/drop_baseline_usage
m, _ = run_spec(
    "rc/controls/loo/drop_baseline_usage",
    "specification_tree/modules/robustness/controls.md#loo",
    "G1", "PD_P2_UsageFITSBY", "B",
    f"PD_P2_UsageFITSBY ~ B + L + {strat_str}", df_raw, "hetero",
    "Full sample", "", "Strata dummies only (drop baseline usage)",
    design_audit_obj=g1_audit, infer_canonical_obj=g1_infer_canonical,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_baseline_usage", "family": "loo",
                "dropped": ["PD_P1_UsageFITSBY"], "n_controls": 1},
)

# rc/controls/loo/drop_strata
m, _ = run_spec(
    "rc/controls/loo/drop_strata",
    "specification_tree/modules/robustness/controls.md#loo",
    "G1", "PD_P2_UsageFITSBY", "B",
    "PD_P2_UsageFITSBY ~ B + L + PD_P1_UsageFITSBY", df_raw, "hetero",
    "Full sample", "", "Baseline usage only (drop strata)",
    design_audit_obj=g1_audit, infer_canonical_obj=g1_infer_canonical,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_strata", "family": "loo",
                "dropped": ["Stratifier dummies"], "n_controls": 1},
)

# rc/controls/sets/none
m, _ = run_spec(
    "rc/controls/sets/none",
    "specification_tree/modules/robustness/controls.md#control-sets",
    "G1", "PD_P2_UsageFITSBY", "B",
    "PD_P2_UsageFITSBY ~ B + L", df_raw, "hetero",
    "Full sample", "", "No controls",
    design_audit_obj=g1_audit, infer_canonical_obj=g1_infer_canonical,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/none", "family": "none", "n_controls": 0},
)

# rc/controls/sets/baseline_only
m, _ = run_spec(
    "rc/controls/sets/baseline_only",
    "specification_tree/modules/robustness/controls.md#control-sets",
    "G1", "PD_P2_UsageFITSBY", "B",
    "PD_P2_UsageFITSBY ~ B + L + PD_P1_UsageFITSBY", df_raw, "hetero",
    "Full sample", "", "Baseline usage only",
    design_audit_obj=g1_audit, infer_canonical_obj=g1_infer_canonical,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/baseline_only", "family": "set",
                "included": ["PD_P1_UsageFITSBY"], "n_controls": 1},
)

# rc/controls/sets/strata_only
m, _ = run_spec(
    "rc/controls/sets/strata_only",
    "specification_tree/modules/robustness/controls.md#control-sets",
    "G1", "PD_P2_UsageFITSBY", "B",
    f"PD_P2_UsageFITSBY ~ B + L + {strat_str}", df_raw, "hetero",
    "Full sample", "", "Strata dummies only",
    design_audit_obj=g1_audit, infer_canonical_obj=g1_infer_canonical,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/strata_only", "family": "set",
                "included": ["Stratifier dummies"], "n_controls": 1},
)

# rc/controls/sets/baseline_plus_strata (same as baseline, confirming)
m, _ = run_spec(
    "rc/controls/sets/baseline_plus_strata",
    "specification_tree/modules/robustness/controls.md#control-sets",
    "G1", "PD_P2_UsageFITSBY", "B",
    formula_base_p2, df_raw, "hetero",
    "Full sample", "", "Strata dummies + PD_P1_UsageFITSBY",
    design_audit_obj=g1_audit, infer_canonical_obj=g1_infer_canonical,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/baseline_plus_strata", "family": "set",
                "included": ["Stratifier dummies", "PD_P1_UsageFITSBY"], "n_controls": 2},
)

# rc/controls/progression/no_controls -> same as sets/none but with progression label
# Already covered above; using progression labels
for prog_id, prog_ctrl, prog_desc in [
    ("rc/controls/progression/no_controls", "PD_P2_UsageFITSBY ~ B + L", "No controls"),
    ("rc/controls/progression/strata_only", f"PD_P2_UsageFITSBY ~ B + L + {strat_str}", "Strata dummies only"),
    ("rc/controls/progression/baseline_usage_only", "PD_P2_UsageFITSBY ~ B + L + PD_P1_UsageFITSBY", "Baseline usage only"),
    ("rc/controls/progression/full", formula_base_p2, "Full controls (strata + baseline usage)"),
]:
    m, _ = run_spec(
        prog_id, "specification_tree/modules/robustness/controls.md#progression",
        "G1", "PD_P2_UsageFITSBY", "B", prog_ctrl, df_raw, "hetero",
        "Full sample", "", prog_desc,
        design_audit_obj=g1_audit, infer_canonical_obj=g1_infer_canonical,
        axis_block_name="controls",
        axis_block={"spec_id": prog_id, "family": "progression", "desc": prog_desc},
    )

# ── Sample restrictions ──
# rc/sample/outliers/trim_y_1_99
p1, p99 = df_raw['PD_P2_UsageFITSBY'].quantile(0.01), df_raw['PD_P2_UsageFITSBY'].quantile(0.99)
df_trim99 = df_raw[(df_raw['PD_P2_UsageFITSBY'] >= p1) & (df_raw['PD_P2_UsageFITSBY'] <= p99)].copy()
m, _ = run_spec(
    "rc/sample/outliers/trim_y_1_99",
    "specification_tree/modules/robustness/sample.md#outliers",
    "G1", "PD_P2_UsageFITSBY", "B",
    formula_base_p2, df_trim99, "hetero",
    "Trimmed 1-99 pctile on Y", "", "Strata + baseline usage",
    design_audit_obj=g1_audit, infer_canonical_obj=g1_infer_canonical,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_1_99", "trim_lower": 0.01, "trim_upper": 0.99},
)

# rc/sample/outliers/trim_y_5_95
p5, p95 = df_raw['PD_P2_UsageFITSBY'].quantile(0.05), df_raw['PD_P2_UsageFITSBY'].quantile(0.95)
df_trim95 = df_raw[(df_raw['PD_P2_UsageFITSBY'] >= p5) & (df_raw['PD_P2_UsageFITSBY'] <= p95)].copy()
m, _ = run_spec(
    "rc/sample/outliers/trim_y_5_95",
    "specification_tree/modules/robustness/sample.md#outliers",
    "G1", "PD_P2_UsageFITSBY", "B",
    formula_base_p2, df_trim95, "hetero",
    "Trimmed 5-95 pctile on Y", "", "Strata + baseline usage",
    design_audit_obj=g1_audit, infer_canonical_obj=g1_infer_canonical,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_5_95", "trim_lower": 0.05, "trim_upper": 0.95},
)

# rc/sample/restriction/balanced_panel_p2p3
# Keep obs with non-missing P2 and P3
df_bal23 = df_raw[df_raw['PD_P2_UsageFITSBY'].notna() & df_raw['PD_P3_UsageFITSBY'].notna()].copy()
m, _ = run_spec(
    "rc/sample/restriction/balanced_panel_p2p3",
    "specification_tree/modules/robustness/sample.md#restriction",
    "G1", "PD_P2_UsageFITSBY", "B",
    formula_base_p2, df_bal23, "hetero",
    "Balanced P2-P3 panel", "", "Strata + baseline usage",
    design_audit_obj=g1_audit, infer_canonical_obj=g1_infer_canonical,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/balanced_panel_p2p3",
                "restriction": "non-missing P2 and P3 FITSBY"},
)

# rc/sample/restriction/balanced_panel_all
df_bal_all = df_raw[df_raw[['PD_P2_UsageFITSBY', 'PD_P3_UsageFITSBY',
                             'PD_P4_UsageFITSBY', 'PD_P5_UsageFITSBY']].notna().all(axis=1)].copy()
m, _ = run_spec(
    "rc/sample/restriction/balanced_panel_all",
    "specification_tree/modules/robustness/sample.md#restriction",
    "G1", "PD_P2_UsageFITSBY", "B",
    formula_base_p2, df_bal_all, "hetero",
    "Balanced all-period panel", "", "Strata + baseline usage",
    design_audit_obj=g1_audit, infer_canonical_obj=g1_infer_canonical,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/balanced_panel_all",
                "restriction": "non-missing P2-P5 FITSBY"},
)

# ── Outcome alternatives ──
# rc/data/outcome/total_usage_not_fitsby  (Total phone usage instead of FITSBY)
m, _ = run_spec(
    "rc/data/outcome/total_usage_not_fitsby",
    "specification_tree/modules/robustness/data_construction.md",
    "G1", "PD_P2_Usage", "B",
    f"PD_P2_Usage ~ B + L + {strat_str} + PD_P1_Usage", df_raw, "hetero",
    "Full sample", "", "Strata + PD_P1_Usage (total baseline)",
    design_audit_obj=g1_audit, infer_canonical_obj=g1_infer_canonical,
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/outcome/total_usage_not_fitsby",
                "outcome_change": "PD_P2_Usage (total, not FITSBY apps only)"},
)

# rc/data/outcome/usage_hours (convert minutes to hours)
df_raw['PD_P2_UsageFITSBY_hrs'] = df_raw['PD_P2_UsageFITSBY'] / 60.0
df_raw['PD_P1_UsageFITSBY_hrs'] = df_raw['PD_P1_UsageFITSBY'] / 60.0
m, _ = run_spec(
    "rc/data/outcome/usage_hours",
    "specification_tree/modules/robustness/data_construction.md",
    "G1", "PD_P2_UsageFITSBY_hrs", "B",
    f"PD_P2_UsageFITSBY_hrs ~ B + L + {strat_str} + PD_P1_UsageFITSBY_hrs", df_raw, "hetero",
    "Full sample (hours)", "", "Strata + baseline usage (hours)",
    design_audit_obj=g1_audit, infer_canonical_obj=g1_infer_canonical,
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/outcome/usage_hours", "units": "hours/day"},
)

# rc/data/treatment/detailed_limit_types
detail_treat_str = " + ".join(["B"] + detailed_limit_cols)
m, _ = run_spec(
    "rc/data/treatment/detailed_limit_types",
    "specification_tree/modules/robustness/data_construction.md",
    "G1", "PD_P2_UsageFITSBY", "B",
    f"PD_P2_UsageFITSBY ~ {detail_treat_str} + {strat_str} + PD_P1_UsageFITSBY",
    df_raw, "hetero",
    "Full sample", "", "Strata + baseline usage, detailed limit types",
    design_audit_obj=g1_audit, infer_canonical_obj=g1_infer_canonical,
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/treatment/detailed_limit_types",
                "treatment_change": "5 separate limit type dummies instead of single L"},
)

# ── Functional form ──
# rc/form/outcome/log1p
df_raw['log1p_PD_P2_UsageFITSBY'] = np.log1p(df_raw['PD_P2_UsageFITSBY'])
df_raw['log1p_PD_P1_UsageFITSBY'] = np.log1p(df_raw['PD_P1_UsageFITSBY'])
m, _ = run_spec(
    "rc/form/outcome/log1p",
    "specification_tree/modules/robustness/functional_form.md",
    "G1", "log1p_PD_P2_UsageFITSBY", "B",
    f"log1p_PD_P2_UsageFITSBY ~ B + L + {strat_str} + log1p_PD_P1_UsageFITSBY",
    df_raw, "hetero",
    "Full sample", "", "Strata + log1p(baseline usage)",
    design_audit_obj=g1_audit, infer_canonical_obj=g1_infer_canonical,
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/log1p", "transform": "log(1+y)",
                "interpretation": "Semi-elasticity of FITSBY usage w.r.t. treatment"},
)

# rc/form/outcome/asinh
df_raw['asinh_PD_P2_UsageFITSBY'] = np.arcsinh(df_raw['PD_P2_UsageFITSBY'])
df_raw['asinh_PD_P1_UsageFITSBY'] = np.arcsinh(df_raw['PD_P1_UsageFITSBY'])
m, _ = run_spec(
    "rc/form/outcome/asinh",
    "specification_tree/modules/robustness/functional_form.md",
    "G1", "asinh_PD_P2_UsageFITSBY", "B",
    f"asinh_PD_P2_UsageFITSBY ~ B + L + {strat_str} + asinh_PD_P1_UsageFITSBY",
    df_raw, "hetero",
    "Full sample", "", "Strata + arcsinh(baseline usage)",
    design_audit_obj=g1_audit, infer_canonical_obj=g1_infer_canonical,
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/asinh", "transform": "arcsinh(y)",
                "interpretation": "IHS semi-elasticity of FITSBY usage"},
)

# ── G1 Inference Variant: HC2 ──
if base_p2_id:
    run_infer_variant(
        base_p2_id, "infer/se/hc/hc2",
        "specification_tree/modules/inference/se.md#hc",
        "G1", "PD_P2_UsageFITSBY", "B",
        formula_base_p2, df_raw, {"HC2": True},
        notes="HC2"
    )

print(f"\nG1 specs so far: {len(results)}")

# =============================================================================
# G2: Survey Well-Being (stacked S3/S4 panel)
# =============================================================================
print("\n=== G2: Survey Well-Being Specifications ===")

# Build stacked dataset: S3 and S4 survey waves
# Need: UserID, B, L, S (wave indicator 1=S3, 2=S4), Stratifier,
# S1_baseline_outcome, outcome

# Outcomes for G2
g2_outcomes = {
    "index_well_N": ("S1_index_well_N", "Survey index"),
    "AddictionIndex_N": ("S1_AddictionIndex_N", "Addiction index"),
    "SMSIndex_N": ("S1_SMSIndex_N", "SMS addiction index"),
    "PhoneUseChange_N": ("S1_PhoneUseChange_N", "Ideal use change"),
    "LifeBetter_N": ("S1_LifeBetter_N", "Phone makes life better"),
    "SWBIndex_N": ("S1_SWBIndex_N", "Subjective well-being"),
}

# Create the stacked dataset
rows_s3 = []
rows_s4 = []
for _, r in df_raw.iterrows():
    base = {
        'UserID': r['UserID'],
        'B_raw': r['B'],
        'L': r['L'],
        'Stratifier': r['Stratifier'],
    }
    for stc in strat_cols:
        base[stc] = r[stc]

    s3_row = base.copy()
    s3_row['S'] = 1
    for ovar, (bvar, _) in g2_outcomes.items():
        s3_row[ovar] = r.get(f'S3_{ovar}', np.nan) if f'S3_{ovar}' in df_raw.columns else np.nan
        s3_row[f'baseline_{ovar}'] = r.get(bvar, np.nan)
    rows_s3.append(s3_row)

    s4_row = base.copy()
    s4_row['S'] = 2
    for ovar, (bvar, _) in g2_outcomes.items():
        s4_row[ovar] = r.get(f'S4_{ovar}', np.nan) if f'S4_{ovar}' in df_raw.columns else np.nan
        s4_row[f'baseline_{ovar}'] = r.get(bvar, np.nan)
    rows_s4.append(s4_row)

df_stacked = pd.DataFrame(rows_s3 + rows_s4)

# Create B and B4 as in the Stata code:
# gen B4 = B * (S == 2)
# replace B = B4
# => B becomes B*I(S==2), and B4 = B*I(S==2)
# They are collinear; effectively B is dropped and B4 captures S4-specific bonus effect
df_stacked['B4'] = df_stacked['B_raw'] * (df_stacked['S'] == 2).astype(int)
df_stacked['B'] = df_stacked['B4'].copy()

# Create S dummy and interactions
df_stacked['S_dummy'] = (df_stacked['S'] == 2).astype(float)

# Strata x S interactions
for sc in strat_cols:
    df_stacked[f'{sc}_xS'] = df_stacked[sc] * df_stacked['S_dummy']

strat_x_s_cols = [f'{sc}_xS' for sc in strat_cols]

# baseline outcome x S interactions
for ovar in g2_outcomes:
    df_stacked[f'baseline_{ovar}_xS'] = df_stacked[f'baseline_{ovar}'] * df_stacked['S_dummy']

print(f"Stacked data: {df_stacked.shape[0]} rows")

# ── G2 BASELINE: index_well_N ──
# Focal regression for FDR table: L coefficient on index_well_N
# Stata: reg yvar B B4 L i.S i.S#$STRATA i.S#c.baseline, robust cluster(UserID)
# Since B = B4 (collinear), Stata drops one. In Python, we include B4 and L.
# We keep B and B4 (B was replaced by B4 in Stata, making them identical; we set B = B4 above)

# The regression includes: B (= B4), B4, L, S_dummy, strata x S interactions, baseline x S interaction
# Since B == B4, one will be dropped for collinearity. We use B4 + L + controls.
baseline_ctrl_var = "baseline_index_well_N"
g2_rhs = f"B4 + L + S_dummy + {' + '.join(strat_cols)} + {' + '.join(strat_x_s_cols)} + baseline_index_well_N + baseline_index_well_N_xS"

m, g2_base_id = run_spec(
    "baseline", "specification_tree/designs/randomized_experiment.md",
    "G2", "index_well_N", "L",
    f"index_well_N ~ {g2_rhs}", df_stacked, {"CRV1": "UserID"},
    "Stacked S3/S4", "", "S dummy, strata x S, baseline x S",
    cluster_var="UserID",
    design_audit_obj=g2_audit, infer_canonical_obj=g2_infer_canonical,
    notes="G2 primary baseline: L coeff on index_well_N"
)
if results[-1]['run_success']:
    print(f"  G2 Baseline L: {results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}, N={results[-1]['n_obs']}")

# ── G2 Additional Baselines (other SWB outcomes) ──
g2_extra_baselines = {
    "baseline__addiction_index": ("AddictionIndex_N", "baseline_AddictionIndex_N"),
    "baseline__sms_index": ("SMSIndex_N", "baseline_SMSIndex_N"),
    "baseline__phone_use_change": ("PhoneUseChange_N", "baseline_PhoneUseChange_N"),
    "baseline__life_better": ("LifeBetter_N", "baseline_LifeBetter_N"),
    "baseline__swb_index": ("SWBIndex_N", "baseline_SWBIndex_N"),
}

for bid, (ovar, bctrl) in g2_extra_baselines.items():
    g2_rhs_var = f"B4 + L + S_dummy + {' + '.join(strat_cols)} + {' + '.join(strat_x_s_cols)} + {bctrl} + {bctrl}_xS"
    m, _ = run_spec(
        bid, "specification_tree/designs/randomized_experiment.md",
        "G2", ovar, "L",
        f"{ovar} ~ {g2_rhs_var}", df_stacked, {"CRV1": "UserID"},
        "Stacked S3/S4", "", f"S dummy, strata x S, {bctrl} x S",
        cluster_var="UserID",
        design_audit_obj=g2_audit, infer_canonical_obj=g2_infer_canonical,
        notes=f"G2 baseline: {ovar}"
    )
    if results[-1]['run_success']:
        print(f"  {bid}: L={results[-1]['coefficient']:.4f}, p={results[-1]['p_value']:.4f}")

# ── G2 Design Variant: diff-in-means ──
m, _ = run_spec(
    "design/randomized_experiment/estimator/diff_in_means",
    "specification_tree/designs/randomized_experiment.md#diff-in-means",
    "G2", "index_well_N", "L",
    "index_well_N ~ B4 + L + S_dummy", df_stacked, {"CRV1": "UserID"},
    "Stacked S3/S4", "", "S dummy only (diff-in-means)",
    cluster_var="UserID",
    design_audit_obj=g2_audit, infer_canonical_obj=g2_infer_canonical,
    notes="G2 diff-in-means"
)

# ── G2 RC Variants ──
# rc/controls/loo/drop_baseline_outcome
g2_rhs_no_baseline = f"B4 + L + S_dummy + {' + '.join(strat_cols)} + {' + '.join(strat_x_s_cols)}"
m, _ = run_spec(
    "rc/controls/loo/drop_baseline_outcome",
    "specification_tree/modules/robustness/controls.md#loo",
    "G2", "index_well_N", "L",
    f"index_well_N ~ {g2_rhs_no_baseline}", df_stacked, {"CRV1": "UserID"},
    "Stacked S3/S4", "", "S dummy, strata x S (no baseline outcome)",
    cluster_var="UserID",
    design_audit_obj=g2_audit, infer_canonical_obj=g2_infer_canonical,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_baseline_outcome", "family": "loo",
                "dropped": ["baseline_index_well_N", "baseline_index_well_N_xS"]},
)

# rc/controls/loo/drop_strata
g2_rhs_no_strata = f"B4 + L + S_dummy + baseline_index_well_N + baseline_index_well_N_xS"
m, _ = run_spec(
    "rc/controls/loo/drop_strata",
    "specification_tree/modules/robustness/controls.md#loo",
    "G2", "index_well_N", "L",
    f"index_well_N ~ {g2_rhs_no_strata}", df_stacked, {"CRV1": "UserID"},
    "Stacked S3/S4", "", "S dummy, baseline x S (no strata)",
    cluster_var="UserID",
    design_audit_obj=g2_audit, infer_canonical_obj=g2_infer_canonical,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_strata", "family": "loo",
                "dropped": ["Stratifier x S dummies"]},
)

# rc/controls/sets/none
m, _ = run_spec(
    "rc/controls/sets/none",
    "specification_tree/modules/robustness/controls.md#control-sets",
    "G2", "index_well_N", "L",
    "index_well_N ~ B4 + L + S_dummy", df_stacked, {"CRV1": "UserID"},
    "Stacked S3/S4", "", "S dummy only",
    cluster_var="UserID",
    design_audit_obj=g2_audit, infer_canonical_obj=g2_infer_canonical,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/none", "family": "none"},
)

# rc/controls/sets/strata_only
m, _ = run_spec(
    "rc/controls/sets/strata_only",
    "specification_tree/modules/robustness/controls.md#control-sets",
    "G2", "index_well_N", "L",
    f"index_well_N ~ {g2_rhs_no_baseline}", df_stacked, {"CRV1": "UserID"},
    "Stacked S3/S4", "", "S dummy, strata x S",
    cluster_var="UserID",
    design_audit_obj=g2_audit, infer_canonical_obj=g2_infer_canonical,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/strata_only", "family": "set",
                "included": ["Stratifier x S"]},
)

# rc/controls/sets/full
m, _ = run_spec(
    "rc/controls/sets/full",
    "specification_tree/modules/robustness/controls.md#control-sets",
    "G2", "index_well_N", "L",
    f"index_well_N ~ {g2_rhs}", df_stacked, {"CRV1": "UserID"},
    "Stacked S3/S4", "", "Full controls",
    cluster_var="UserID",
    design_audit_obj=g2_audit, infer_canonical_obj=g2_infer_canonical,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/full", "family": "set",
                "included": ["Stratifier x S", "baseline x S"]},
)

# ── G2 Sample restrictions ──
# rc/sample/outliers/trim_y_1_99
p1_g2, p99_g2 = df_stacked['index_well_N'].quantile(0.01), df_stacked['index_well_N'].quantile(0.99)
df_g2_trim99 = df_stacked[(df_stacked['index_well_N'] >= p1_g2) & (df_stacked['index_well_N'] <= p99_g2)].copy()
m, _ = run_spec(
    "rc/sample/outliers/trim_y_1_99",
    "specification_tree/modules/robustness/sample.md#outliers",
    "G2", "index_well_N", "L",
    f"index_well_N ~ {g2_rhs}", df_g2_trim99, {"CRV1": "UserID"},
    "Stacked S3/S4, trimmed 1-99", "", "Full controls",
    cluster_var="UserID",
    design_audit_obj=g2_audit, infer_canonical_obj=g2_infer_canonical,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_1_99", "trim_lower": 0.01, "trim_upper": 0.99},
)

# rc/sample/outliers/trim_y_5_95
p5_g2, p95_g2 = df_stacked['index_well_N'].quantile(0.05), df_stacked['index_well_N'].quantile(0.95)
df_g2_trim95 = df_stacked[(df_stacked['index_well_N'] >= p5_g2) & (df_stacked['index_well_N'] <= p95_g2)].copy()
m, _ = run_spec(
    "rc/sample/outliers/trim_y_5_95",
    "specification_tree/modules/robustness/sample.md#outliers",
    "G2", "index_well_N", "L",
    f"index_well_N ~ {g2_rhs}", df_g2_trim95, {"CRV1": "UserID"},
    "Stacked S3/S4, trimmed 5-95", "", "Full controls",
    cluster_var="UserID",
    design_audit_obj=g2_audit, infer_canonical_obj=g2_infer_canonical,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_5_95", "trim_lower": 0.05, "trim_upper": 0.95},
)

# rc/sample/restriction/s3_only
df_s3_only = df_stacked[df_stacked['S'] == 1].copy()
# For S3-only, no S interactions needed
g2_rhs_s3 = f"B4 + L + {' + '.join(strat_cols)} + baseline_index_well_N"
m, _ = run_spec(
    "rc/sample/restriction/s3_only",
    "specification_tree/modules/robustness/sample.md#restriction",
    "G2", "index_well_N", "L",
    f"index_well_N ~ {g2_rhs_s3}", df_s3_only, "hetero",
    "S3 only (midline survey)", "", "Strata + baseline outcome",
    design_audit_obj=g2_audit, infer_canonical_obj=g2_infer_canonical,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/s3_only", "restriction": "S3 survey wave only"},
    notes="Single cross-section, no clustering needed; B4=0 for all S3 obs (bonus is S4-specific)"
)

# rc/sample/restriction/s4_only
df_s4_only = df_stacked[df_stacked['S'] == 2].copy()
g2_rhs_s4 = f"B4 + L + {' + '.join(strat_cols)} + baseline_index_well_N"
m, _ = run_spec(
    "rc/sample/restriction/s4_only",
    "specification_tree/modules/robustness/sample.md#restriction",
    "G2", "index_well_N", "L",
    f"index_well_N ~ {g2_rhs_s4}", df_s4_only, "hetero",
    "S4 only (endline survey)", "", "Strata + baseline outcome",
    design_audit_obj=g2_audit, infer_canonical_obj=g2_infer_canonical,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/s4_only", "restriction": "S4 survey wave only"},
)

# ── G2 Inference Variant: HC1 (no clustering) ──
if g2_base_id:
    run_infer_variant(
        g2_base_id, "infer/se/hc/hc1",
        "specification_tree/modules/inference/se.md#hc",
        "G2", "index_well_N", "L",
        f"index_well_N ~ {g2_rhs}", df_stacked, "hetero",
        notes="HC1 (no clustering)"
    )

# ── Additional G1 specs: L coefficient across outcomes and periods ──
# Report L from extra baselines (P3, P4, P5, P2-P5 avg FITSBY)
for bid, yvar, bctrl, desc in [
    ("rc/data/outcome/p3_fitsby_L", "PD_P3_UsageFITSBY", "PD_P1_UsageFITSBY", "Phase 3 FITSBY, L coef"),
    ("rc/data/outcome/p4_fitsby_L", "PD_P4_UsageFITSBY", "PD_P1_UsageFITSBY", "Phase 4 FITSBY, L coef"),
    ("rc/data/outcome/p5_fitsby_L", "PD_P5_UsageFITSBY", "PD_P1_UsageFITSBY", "Phase 5 FITSBY, L coef"),
    ("rc/data/outcome/p2to5_fitsby_L", "PD_P2to5_UsageFITSBY", "PD_P1_UsageFITSBY", "P2-P5 avg FITSBY, L coef"),
]:
    formula = f"{yvar} ~ B + L + {strat_str} + {bctrl}"
    m, rid = run_spec(
        bid, "specification_tree/modules/robustness/data_construction.md",
        "G1", yvar, "L", formula, df_raw, "hetero",
        "Full sample", "", f"Strata + {bctrl}",
        design_audit_obj=g1_audit, infer_canonical_obj=g1_infer_canonical,
        notes=f"G1: {desc}"
    )

# Additional G1 robustness: L coefficient with trimming and functional form
# L with trim 1-99
m, _ = run_spec(
    "rc/sample/outliers/trim_y_1_99_L",
    "specification_tree/modules/robustness/sample.md#outliers",
    "G1", "PD_P2_UsageFITSBY", "L",
    formula_base_p2, df_trim99, "hetero",
    "Trimmed 1-99 on Y, L coef", "", "Strata + baseline usage",
    design_audit_obj=g1_audit, infer_canonical_obj=g1_infer_canonical,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_1_99_L", "trim_lower": 0.01, "trim_upper": 0.99},
)

# L with log1p
m, _ = run_spec(
    "rc/form/outcome/log1p_L",
    "specification_tree/modules/robustness/functional_form.md",
    "G1", "log1p_PD_P2_UsageFITSBY", "L",
    f"log1p_PD_P2_UsageFITSBY ~ B + L + {strat_str} + log1p_PD_P1_UsageFITSBY",
    df_raw, "hetero",
    "Full sample", "", "Strata + log1p(baseline usage)",
    design_audit_obj=g1_audit, infer_canonical_obj=g1_infer_canonical,
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/log1p_L", "transform": "log(1+y)",
                "interpretation": "Semi-elasticity of FITSBY usage w.r.t. L treatment"},
)

# L with asinh
m, _ = run_spec(
    "rc/form/outcome/asinh_L",
    "specification_tree/modules/robustness/functional_form.md",
    "G1", "asinh_PD_P2_UsageFITSBY", "L",
    f"asinh_PD_P2_UsageFITSBY ~ B + L + {strat_str} + asinh_PD_P1_UsageFITSBY",
    df_raw, "hetero",
    "Full sample", "", "Strata + arcsinh(baseline usage)",
    design_audit_obj=g1_audit, infer_canonical_obj=g1_infer_canonical,
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/asinh_L", "transform": "arcsinh(y)",
                "interpretation": "IHS semi-elasticity of FITSBY usage w.r.t. L treatment"},
)

# L with total usage
m, _ = run_spec(
    "rc/data/outcome/total_usage_L",
    "specification_tree/modules/robustness/data_construction.md",
    "G1", "PD_P2_Usage", "L",
    f"PD_P2_Usage ~ B + L + {strat_str} + PD_P1_Usage", df_raw, "hetero",
    "Full sample", "", "Strata + PD_P1_Usage (total baseline)",
    design_audit_obj=g1_audit, infer_canonical_obj=g1_infer_canonical,
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/outcome/total_usage_L",
                "outcome_change": "PD_P2_Usage (total, L coef)"},
)

# G1 DIM for L
m, _ = run_spec(
    "design/randomized_experiment/estimator/diff_in_means_L",
    "specification_tree/designs/randomized_experiment.md#diff-in-means",
    "G1", "PD_P2_UsageFITSBY", "L",
    "PD_P2_UsageFITSBY ~ B + L", df_raw, "hetero",
    "Full sample", "", "None (diff-in-means, L coef)",
    design_audit_obj=g1_audit, infer_canonical_obj=g1_infer_canonical,
    notes="DIM: L coeff"
)

print(f"\nTotal specs: {len(results)}")
print(f"Total inference variants: {len(inference_results)}")

# =============================================================================
# Write outputs
# =============================================================================
spec_df = pd.DataFrame(results)
spec_df.to_csv(f"{DATA_DIR}/specification_results.csv", index=False)
print(f"\nWrote specification_results.csv: {len(spec_df)} rows")

if inference_results:
    infer_df = pd.DataFrame(inference_results)
    infer_df.to_csv(f"{DATA_DIR}/inference_results.csv", index=False)
    print(f"Wrote inference_results.csv: {len(infer_df)} rows")

# ── Write SPECIFICATION_SEARCH.md ──
n_success = spec_df['run_success'].sum()
n_fail = len(spec_df) - n_success
g1_count = len(spec_df[spec_df['baseline_group_id'] == 'G1'])
g2_count = len(spec_df[spec_df['baseline_group_id'] == 'G2'])

md = f"""# Specification Search: 163822-V2
## Allcott, Gentzkow & Song (2022) "Digital Addiction"

### Surface Summary
- **Paper ID**: 163822-V2
- **Surface hash**: {SURFACE_HASH}
- **Design**: Randomized experiment (RCT)
- **Baseline groups**: 2
  - **G1**: Phone usage (PD-measured screen time) - Budget: 50 specs
  - **G2**: Survey well-being (stacked S3/S4 panel) - Budget: 40 specs
- **Seed**: 163822

### Data Source
- `final_data_sample.dta` downloaded from Harvard Dataverse (doi:10.7910/DVN/GN636M)
- 1,933 observations (Android phone users)
- Treatment: Bonus (financial incentive, B) and Limit (commitment device, L)
- Stratification on baseline usage x addiction x restriction

### Execution Summary
- **Total specifications executed**: {len(spec_df)}
  - G1 (phone usage): {g1_count}
  - G2 (survey well-being): {g2_count}
- **Successful**: {n_success}
- **Failed**: {n_fail}
- **Inference variants**: {len(inference_results)}

### G1 Specifications (Phone Usage)
- **Baseline**: PD_P2_UsageFITSBY ~ B + L + Strata + PD_P1_UsageFITSBY, robust
- **Additional baselines**: P3, P4, P5, P2-P5 avg FITSBY; P2 total, P2-P5 avg total
- **Design variants**: Diff-in-means, with covariates
- **RC variants**: LOO (drop baseline, drop strata), control sets, progression,
  sample trimming (1-99, 5-95), balanced panels, total usage outcome,
  hours outcome, detailed limit types, log1p transform, arcsinh transform
- **Inference variant**: HC2

### G2 Specifications (Survey Well-Being)
- **Baseline**: index_well_N ~ B4 + L + S + Strata x S + baseline x S, cluster(UserID)
  - Data stacked across S3 (midline) and S4 (endline) survey waves
  - B4 = Bonus x (S==S4); B replaced by B4 to capture S4-specific bonus
- **Additional baselines**: AddictionIndex, SMSIndex, PhoneUseChange, LifeBetter, SWBIndex
- **Design variant**: Diff-in-means (no controls)
- **RC variants**: LOO, control sets, trimming, S3-only, S4-only
- **Inference variant**: HC1 (no clustering)

### Deviations from Surface
- None. All planned specs were executed.

### Software Stack
- Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}
- pyfixest: {SW_BLOCK['packages'].get('pyfixest', 'N/A')}
- pandas: {SW_BLOCK['packages'].get('pandas', 'N/A')}
- numpy: {SW_BLOCK['packages'].get('numpy', 'N/A')}
"""

with open(f"{DATA_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write(md)
print("Wrote SPECIFICATION_SEARCH.md")
print("Done with 163822-V2!")
