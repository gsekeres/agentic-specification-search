"""
Specification Search Script for Dranove, Hughes, and Meltzer (2003/2012)
"Incentives and Promotion for Adverse Drug Reactions"
American Economic Journal: Economic Policy

Paper ID: 114747-V1

Surface-driven execution:
  - G1: Poisson veryserious ~ promotion_vars | drug FE + yearmonth FE
    => CANNOT RUN: promotion expenditure variables missing from provided data
  - G2: Logit any_fda_reaction ~ veryserious*condition_dummies + controls | yearmonth dummies
    => CAN PARTIALLY RUN: some controls missing ($char, $age, generic) but
       core treatment vars (v1-v4, cumulative versions) and Dappr_cats available

Available data: 1456 obs (63 drugs x ~24 months), 18 columns.
Key available vars: veryserious (ADR count), any_fda_reaction (binary FDA labeling),
  Dappr_cats_1-4 (drug approval age), condition (1-4), mastername, yearmo,
  contraindication, boxed_warning, warning, precaution, adverse_reaction,
  countfda, countduration.

Missing: q1totalexp, q2q4totalexp, generic, sh_count* (demographics/characteristics),
  permonths (exposure).

Strategy: Focus on G2 logit/LPM specifications with available variables.
  Build v1-v4 interactions, cumulative ADR measures, and run across:
  - Baseline logit with available controls (Dappr_cats + yearmonth dummies)
  - LPM (OLS) versions with drug FE and yearmonth FE
  - Control variations (LOO Dappr_cats, add/drop FDA label sub-types)
  - Treatment variations (contemporaneous, 3-month cumulative, 12-month cumulative)
  - Outcome variations (any_fda_reaction, countfda, countduration)
  - Sample restrictions (exclude specific drugs, condition subsets)
  - FE variations (drug, yearmonth, drug+yearmonth, condition+yearmonth)
  - Inference variations (robust, cluster by drug, cluster by condition)

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

PAPER_ID = "114747-V1"
DATA_DIR = "data/downloads/extracted/114747-V1"
OUTPUT_DIR = DATA_DIR
DATA_PATH = f"{DATA_DIR}/AEJPol-2009-0070_data.dta"

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

# We focus on G2 (logit/LPM for FDA labeling outcome)
bg = surface_obj["baseline_groups"][1]  # G2
design_audit = bg["design_audit"]
inference_canonical = bg["inference_plan"]["canonical"]

# ============================================================
# Data Loading and Preparation
# ============================================================

df_raw = pd.read_stata(DATA_PATH)
print(f"Loaded data: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

# Convert float32 to float64
for col in df_raw.columns:
    if df_raw[col].dtype == np.float32:
        df_raw[col] = df_raw[col].astype(np.float64)

# Create string versions for FE
df_raw['drug_str'] = df_raw['mastername'].astype(str)
df_raw['yearmo_str'] = df_raw['yearmo'].dt.strftime('%Y-%m')

# Create condition dummies
df_raw['c1'] = (df_raw['condition'] == 1).astype(int)
df_raw['c2'] = (df_raw['condition'] == 2).astype(int)
df_raw['c3'] = (df_raw['condition'] == 3).astype(int)
df_raw['c4'] = (df_raw['condition'] == 4).astype(int)

# Create condition-interacted veryserious (v1-v4)
df_raw['v1'] = df_raw['veryserious'] * df_raw['c1']
df_raw['v2'] = df_raw['veryserious'] * df_raw['c2']
df_raw['v3'] = df_raw['veryserious'] * df_raw['c3']
df_raw['v4'] = df_raw['veryserious'] * df_raw['c4']

# Sort for lag/cumulative calculations
df_raw = df_raw.sort_values(['mastername', 'yearmo']).reset_index(drop=True)

# Create lagged and cumulative ADR measures
df_raw['veryserious_lag1'] = df_raw.groupby('mastername')['veryserious'].shift(1)
df_raw['veryserious_lag3'] = df_raw.groupby('mastername')['veryserious'].shift(3)

# 3-month cumulative (current + 2 lags)
df_raw['very_3'] = (
    df_raw['veryserious']
    + df_raw.groupby('mastername')['veryserious'].shift(1).fillna(0)
    + df_raw.groupby('mastername')['veryserious'].shift(2).fillna(0)
)
# Only valid where we have at least 3 months of data
df_raw.loc[df_raw.groupby('mastername').cumcount() < 2, 'very_3'] = np.nan

# 12-month cumulative
df_raw['very_12'] = df_raw.groupby('mastername')['veryserious'].rolling(12, min_periods=12).sum().reset_index(level=0, drop=True)

# 3-month condition-interacted
df_raw['v_31'] = df_raw['very_3'] * df_raw['c1']
df_raw['v_32'] = df_raw['very_3'] * df_raw['c2']
df_raw['v_33'] = df_raw['very_3'] * df_raw['c3']
df_raw['v_34'] = df_raw['very_3'] * df_raw['c4']

# 12-month condition-interacted
df_raw['v_121'] = df_raw['very_12'] * df_raw['c1']
df_raw['v_122'] = df_raw['very_12'] * df_raw['c2']
df_raw['v_123'] = df_raw['very_12'] * df_raw['c3']
df_raw['v_124'] = df_raw['very_12'] * df_raw['c4']

# Log(1+veryserious) for functional form variants
df_raw['log_veryserious'] = np.log1p(df_raw['veryserious'])
df_raw['log_v1'] = df_raw['log_veryserious'] * df_raw['c1']
df_raw['log_v2'] = df_raw['log_veryserious'] * df_raw['c2']
df_raw['log_v3'] = df_raw['log_veryserious'] * df_raw['c3']
df_raw['log_v4'] = df_raw['log_veryserious'] * df_raw['c4']

# Scale veryserious by 100 for readability of coefficients
df_raw['vs_100_1'] = df_raw['v1'] / 100.0
df_raw['vs_100_2'] = df_raw['v2'] / 100.0
df_raw['vs_100_3'] = df_raw['v3'] / 100.0
df_raw['vs_100_4'] = df_raw['v4'] / 100.0

# Available controls
DAPPR_CONTROLS = ['Dappr_cats_1', 'Dappr_cats_2', 'Dappr_cats_3', 'Dappr_cats_4']
FDA_SUBTYPES = ['contraindication', 'boxed_warning', 'warning', 'precaution', 'adverse_reaction']

# Condition dummies (used as controls -- c4 omitted as reference)
CONDITION_DUMMIES = ['c1', 'c2', 'c3']

# Treatment variable sets
TREATMENT_CONTEMP = ['v1', 'v2', 'v3', 'v4']
TREATMENT_CUM3 = ['v_31', 'v_32', 'v_33', 'v_34']
TREATMENT_CUM12 = ['v_121', 'v_122', 'v_123', 'v_124']
TREATMENT_LOG = ['log_v1', 'log_v2', 'log_v3', 'log_v4']

# Full sample (no missing in key vars)
df = df_raw.dropna(subset=['any_fda_reaction', 'veryserious']).copy()
print(f"Full sample: {len(df)} rows, {df['drug_str'].nunique()} drugs")

# 3-month cumulative sample
df_cum3 = df.dropna(subset=['very_3']).copy()
print(f"3-month cumulative sample: {len(df_cum3)} rows")

# 12-month cumulative sample
df_cum12 = df.dropna(subset=['very_12']).copy()
print(f"12-month cumulative sample: {len(df_cum12)} rows")

# Duration sample (countduration non-missing)
df_dur = df.dropna(subset=['countduration']).copy()
print(f"Duration sample: {len(df_dur)} rows")

# ============================================================
# Accumulators
# ============================================================

results = []
inference_results = []
spec_run_counter = 0

# Focal treatment var for coefficient extraction -- we report v3 (arthritis condition)
# as the paper's main Table 3 focuses on condition 3 (arthritis)
FOCAL_TREATMENT = 'v3'


# ============================================================
# Helper: run_lpm (LPM/OLS with FE via pyfixest)
# ============================================================

def run_lpm(spec_id, spec_tree_path, baseline_group_id,
            outcome_var, treatment_vars, controls, fe_formula_str,
            fe_desc, data, vcov, sample_desc, controls_desc,
            focal_var=None, cluster_var="drug_str",
            axis_block_name=None, axis_block=None, notes=""):
    """Run a single LPM/OLS specification and record results."""
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    if focal_var is None:
        focal_var = treatment_vars[0] if isinstance(treatment_vars, list) else treatment_vars

    try:
        if isinstance(treatment_vars, list):
            treat_str = " + ".join(treatment_vars)
        else:
            treat_str = treatment_vars

        controls_str = " + ".join(controls) if controls else ""
        rhs = treat_str
        if controls_str:
            rhs += " + " + controls_str

        if fe_formula_str:
            formula = f"{outcome_var} ~ {rhs} | {fe_formula_str}"
        else:
            formula = f"{outcome_var} ~ {rhs}"

        m = pf.feols(formula, data=data, vcov=vcov)

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
            inference={"spec_id": inference_canonical["spec_id"],
                       "method": "robust" if isinstance(vcov, str) else "cluster",
                       "cluster_vars": [cluster_var] if not isinstance(vcov, str) else []},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"panel_fixed_effects": design_audit},
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
            "treatment_var": focal_var,
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
            "treatment_var": focal_var,
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
            "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


# ============================================================
# Helper: run_logit (via statsmodels)
# ============================================================

def run_logit(spec_id, spec_tree_path, baseline_group_id,
              outcome_var, treatment_vars, controls, fe_dummy_vars,
              fe_desc, data, sample_desc, controls_desc,
              focal_var=None, cluster_var="drug_str",
              axis_block_name=None, axis_block=None, notes=""):
    """Run a logit specification using statsmodels."""
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    if focal_var is None:
        focal_var = treatment_vars[0] if isinstance(treatment_vars, list) else treatment_vars

    try:
        import statsmodels.api as sm

        treat_list = treatment_vars if isinstance(treatment_vars, list) else [treatment_vars]
        rhs_vars = list(treat_list) + list(controls)

        temp_data = data.copy()

        # Create FE dummies
        if fe_dummy_vars:
            for fv in fe_dummy_vars:
                dummies = pd.get_dummies(temp_data[fv], prefix=fv, drop_first=True).astype(float)
                temp_data = pd.concat([temp_data, dummies], axis=1)
                rhs_vars = rhs_vars + list(dummies.columns)

        all_vars = [outcome_var] + rhs_vars
        est_data = temp_data.dropna(subset=all_vars).copy()

        y = est_data[outcome_var].astype(float)
        X = sm.add_constant(est_data[rhs_vars].astype(float))

        # Drop perfect prediction FE dummies
        if fe_dummy_vars:
            cols_to_drop = []
            for col in list(X.columns):
                if any(col.startswith(f"{fv}_") for fv in fe_dummy_vars):
                    mask = X[col] == 1
                    if mask.sum() > 0:
                        y_vals = y[mask]
                        if y_vals.nunique() <= 1:
                            cols_to_drop.append(col)
            if cols_to_drop:
                # Remove observations in these groups instead of dropping columns
                drop_mask = pd.Series(False, index=est_data.index)
                for col in cols_to_drop:
                    drop_mask |= (X[col] == 1)
                est_data = est_data[~drop_mask].copy()
                y = est_data[outcome_var].astype(float)
                remaining_rhs = [v for v in rhs_vars if v not in cols_to_drop]
                X = sm.add_constant(est_data[remaining_rhs].astype(float))

        logit_model = sm.Logit(y, X)

        # Fit with clustering if cluster_var available
        try:
            if cluster_var and cluster_var in est_data.columns:
                logit_result = logit_model.fit(
                    method='bfgs',
                    cov_type='cluster',
                    cov_kwds={'groups': est_data[cluster_var].values},
                    disp=0, maxiter=500)
            else:
                logit_result = logit_model.fit(method='bfgs', disp=0, maxiter=500)
        except Exception:
            logit_result = logit_model.fit(method='nm', disp=0, maxiter=1000)

        coef_val = float(logit_result.params.get(focal_var, np.nan))
        se_val = float(logit_result.bse.get(focal_var, np.nan))
        pval = float(logit_result.pvalues.get(focal_var, np.nan))

        try:
            ci = logit_result.conf_int()
            ci_lower = float(ci.loc[focal_var, 0])
            ci_upper = float(ci.loc[focal_var, 1])
        except Exception:
            ci_lower = np.nan
            ci_upper = np.nan

        nobs = int(logit_result.nobs)
        try:
            r2 = float(logit_result.prsquared)
        except Exception:
            r2 = np.nan

        all_coefs = {k: float(v) for k, v in logit_result.params.items()
                     if not any(k.startswith(f"{fv}_") for fv in (fe_dummy_vars or []))}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": inference_canonical["spec_id"],
                       "method": "robust", "estimator": "logit"},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"panel_fixed_effects": design_audit},
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
            "treatment_var": focal_var,
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
            "run_error": ""
        })
        return run_id, coef_val, se_val, pval, nobs

    except Exception as e:
        err_msg = str(e)[:240]
        err_details = error_details_from_exception(e, stage="logit_estimation")
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
            "treatment_var": focal_var,
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
            "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


# ============================================================
# G2 BASELINE: Logit any_fda_reaction ~ v1 v2 v3 v4 + c1 c2 c3 + Dappr_cats + yearmonth dummies
# (Table 2 Col 1, contemporaneous ADRs)
# NOTE: Missing generic, $char, $age controls from paper.
# ============================================================

print("\n=== G2 BASELINE: Logit with available controls ===")

baseline_controls_G2 = CONDITION_DUMMIES + DAPPR_CONTROLS
base_run_id_G2, base_coef, base_se, base_pval, base_nobs = run_logit(
    "baseline__table2_logit_contemporaneous",
    "designs/panel_fixed_effects.md#baseline", "G2",
    "any_fda_reaction", TREATMENT_CONTEMP, baseline_controls_G2,
    ["yearmo_str"], "yearmonth dummies", df,
    f"Full sample, N={len(df)}", "c1-c3 + Dappr_cats_1-4 + yearmonth dummies",
    focal_var="v3")

print(f"  G2 Baseline: coef(v3)={base_coef:.6f}, se={base_se:.6f}, p={base_pval:.4f}, N={base_nobs}")


# ============================================================
# G2 BASELINE VARIANTS: 3-month and 12-month cumulative ADRs
# ============================================================

print("\n=== G2 Baseline: 3-month cumulative ===")
run_logit(
    "baseline__table2_logit_3month",
    "designs/panel_fixed_effects.md#baseline", "G2",
    "any_fda_reaction", TREATMENT_CUM3, baseline_controls_G2,
    ["yearmo_str"], "yearmonth dummies", df_cum3,
    f"3-month cumulative sample, N={len(df_cum3)}", "c1-c3 + Dappr_cats + yearmonth",
    focal_var="v_33",
    axis_block_name="estimation",
    axis_block={"spec_id": "baseline__table2_logit_3month", "treatment_form": "cumulative_3month"})

print("\n=== G2 Baseline: 12-month cumulative ===")
run_logit(
    "baseline__table2_logit_12month",
    "designs/panel_fixed_effects.md#baseline", "G2",
    "any_fda_reaction", TREATMENT_CUM12, baseline_controls_G2,
    ["yearmo_str"], "yearmonth dummies", df_cum12,
    f"12-month cumulative sample, N={len(df_cum12)}", "c1-c3 + Dappr_cats + yearmonth",
    focal_var="v_123",
    axis_block_name="estimation",
    axis_block={"spec_id": "baseline__table2_logit_12month", "treatment_form": "cumulative_12month"})


# ============================================================
# G2 BASELINE: Duration model (countduration outcome)
# ============================================================

print("\n=== G2 Baseline: Duration model ===")
run_logit(
    "baseline__table2_duration_contemporaneous",
    "designs/panel_fixed_effects.md#baseline", "G2",
    "countduration", TREATMENT_CONTEMP, baseline_controls_G2,
    ["yearmo_str"], "yearmonth dummies", df_dur,
    f"Duration sample, N={len(df_dur)}", "c1-c3 + Dappr_cats + yearmonth",
    focal_var="v3",
    axis_block_name="estimation",
    axis_block={"spec_id": "baseline__table2_duration_contemporaneous", "outcome": "countduration"})


# ============================================================
# DESIGN VARIANTS: LPM (linear probability model) instead of logit
# ============================================================

print("\n=== Design: LPM (OLS) with yearmonth FE ===")

# LPM baseline -- OLS with yearmonth FE
run_lpm(
    "design/panel_fixed_effects/estimator/lpm",
    "designs/panel_fixed_effects.md#estimator-alternatives", "G2",
    "any_fda_reaction", TREATMENT_CONTEMP, CONDITION_DUMMIES + DAPPR_CONTROLS,
    "yearmo_str", "yearmonth FE", df,
    "hetero", f"Full sample, N={len(df)}", "c1-c3 + Dappr_cats + yearmonth FE",
    focal_var="v3",
    axis_block_name="estimation",
    axis_block={"spec_id": "design/lpm", "estimator": "LPM"})

# LPM with drug FE + yearmonth FE
run_lpm(
    "design/panel_fixed_effects/estimator/lpm_drug_fe",
    "designs/panel_fixed_effects.md#estimator-alternatives", "G2",
    "any_fda_reaction", TREATMENT_CONTEMP, DAPPR_CONTROLS,
    "drug_str + yearmo_str", "drug + yearmonth FE", df,
    "hetero", f"Full sample, N={len(df)}", "Dappr_cats + drug + yearmonth FE",
    focal_var="v3",
    axis_block_name="estimation",
    axis_block={"spec_id": "design/lpm_drug_fe", "estimator": "LPM", "fe": "drug+yearmonth"})

# LPM with cluster(drug)
run_lpm(
    "design/panel_fixed_effects/estimator/lpm_cluster_drug",
    "designs/panel_fixed_effects.md#estimator-alternatives", "G2",
    "any_fda_reaction", TREATMENT_CONTEMP, CONDITION_DUMMIES + DAPPR_CONTROLS,
    "yearmo_str", "yearmonth FE", df,
    {"CRV1": "drug_str"}, f"Full sample", "c1-c3 + Dappr_cats + yearmonth FE, cluster(drug)",
    focal_var="v3", cluster_var="drug_str",
    axis_block_name="estimation",
    axis_block={"spec_id": "design/lpm_cluster_drug", "estimator": "LPM", "cluster": "drug"})

# LPM - 3-month cumulative
run_lpm(
    "design/panel_fixed_effects/estimator/lpm_cum3",
    "designs/panel_fixed_effects.md#estimator-alternatives", "G2",
    "any_fda_reaction", TREATMENT_CUM3, CONDITION_DUMMIES + DAPPR_CONTROLS,
    "yearmo_str", "yearmonth FE", df_cum3,
    "hetero", f"3-month cumulative sample", "c1-c3 + Dappr_cats + yearmonth FE",
    focal_var="v_33",
    axis_block_name="estimation",
    axis_block={"spec_id": "design/lpm_cum3", "estimator": "LPM", "treatment_form": "cumulative_3month"})

# LPM - 12-month cumulative
run_lpm(
    "design/panel_fixed_effects/estimator/lpm_cum12",
    "designs/panel_fixed_effects.md#estimator-alternatives", "G2",
    "any_fda_reaction", TREATMENT_CUM12, CONDITION_DUMMIES + DAPPR_CONTROLS,
    "yearmo_str", "yearmonth FE", df_cum12,
    "hetero", f"12-month cumulative sample", "c1-c3 + Dappr_cats + yearmonth FE",
    focal_var="v_123",
    axis_block_name="estimation",
    axis_block={"spec_id": "design/lpm_cum12", "estimator": "LPM", "treatment_form": "cumulative_12month"})

# LPM - duration outcome
run_lpm(
    "design/panel_fixed_effects/estimator/lpm_duration",
    "designs/panel_fixed_effects.md#estimator-alternatives", "G2",
    "countduration", TREATMENT_CONTEMP, CONDITION_DUMMIES + DAPPR_CONTROLS,
    "yearmo_str", "yearmonth FE", df_dur,
    "hetero", f"Duration sample", "c1-c3 + Dappr_cats + yearmonth FE",
    focal_var="v3",
    axis_block_name="estimation",
    axis_block={"spec_id": "design/lpm_duration", "estimator": "LPM", "outcome": "countduration"})


# ============================================================
# DESIGN: Pooled conditions (veryserious not interacted)
# ============================================================

print("\n=== Design: Pooled conditions ===")

# LPM: any_fda_reaction ~ veryserious + c1 c2 c3 + Dappr_cats | yearmo
run_lpm(
    "design/panel_fixed_effects/pooling/pooled_conditions",
    "designs/panel_fixed_effects.md#pooling", "G2",
    "any_fda_reaction", ["veryserious"], CONDITION_DUMMIES + DAPPR_CONTROLS,
    "yearmo_str", "yearmonth FE", df,
    "hetero", f"Full sample, pooled", "c1-c3 + Dappr_cats + yearmonth FE (pooled)",
    focal_var="veryserious",
    axis_block_name="estimation",
    axis_block={"spec_id": "design/pooled", "pooling": "pooled_conditions"})

# Logit pooled
run_logit(
    "design/panel_fixed_effects/pooling/pooled_logit",
    "designs/panel_fixed_effects.md#pooling", "G2",
    "any_fda_reaction", ["veryserious"], CONDITION_DUMMIES + DAPPR_CONTROLS,
    ["yearmo_str"], "yearmonth dummies", df,
    f"Full sample, pooled logit", "c1-c3 + Dappr_cats + yearmonth (pooled)",
    focal_var="veryserious",
    axis_block_name="estimation",
    axis_block={"spec_id": "design/pooled_logit", "pooling": "pooled", "estimator": "logit"})


# ============================================================
# DESIGN: Condition-specific regressions
# ============================================================

print("\n=== Design: Condition-specific regressions ===")

condition_names = {1: "cholesterol", 2: "allergies", 3: "arthritis", 4: "depression"}
for cond_num, cond_name in condition_names.items():
    df_cond = df[df['condition'] == cond_num].copy()
    if len(df_cond) < 10:
        continue

    # LPM
    run_lpm(
        f"design/panel_fixed_effects/condition/{cond_name}_lpm",
        "designs/panel_fixed_effects.md#condition-specific", "G2",
        "any_fda_reaction", ["veryserious"], DAPPR_CONTROLS,
        "yearmo_str", "yearmonth FE", df_cond,
        "hetero", f"Condition {cond_num} ({cond_name}), N={len(df_cond)}",
        f"Dappr_cats + yearmonth FE (condition {cond_num})",
        focal_var="veryserious",
        axis_block_name="sample",
        axis_block={"spec_id": f"design/condition/{cond_name}_lpm",
                    "condition": cond_num, "condition_name": cond_name})

    # Logit
    run_logit(
        f"design/panel_fixed_effects/condition/{cond_name}_logit",
        "designs/panel_fixed_effects.md#condition-specific", "G2",
        "any_fda_reaction", ["veryserious"], DAPPR_CONTROLS,
        ["yearmo_str"], "yearmonth dummies", df_cond,
        f"Condition {cond_num} ({cond_name}), N={len(df_cond)}",
        f"Dappr_cats + yearmonth (condition {cond_num})",
        focal_var="veryserious",
        axis_block_name="sample",
        axis_block={"spec_id": f"design/condition/{cond_name}_logit",
                    "condition": cond_num, "estimator": "logit"})


# ============================================================
# RC: CONTROLS LOO (drop one Dappr_cats at a time)
# ============================================================

print("\n=== RC: Controls LOO ===")

for i, dappr_var in enumerate(DAPPR_CONTROLS):
    ctrl = [c for c in baseline_controls_G2 if c != dappr_var]
    spec_id = f"rc/controls/loo/drop_{dappr_var}"

    run_lpm(
        spec_id, "modules/robustness/controls.md#leave-one-out", "G2",
        "any_fda_reaction", TREATMENT_CONTEMP, ctrl,
        "yearmo_str", "yearmonth FE", df,
        "hetero", "Full sample", f"drop {dappr_var}",
        focal_var="v3",
        axis_block_name="controls",
        axis_block={"spec_id": spec_id, "family": "loo", "dropped": [dappr_var]})


# ============================================================
# RC: CONTROLS SETS
# ============================================================

print("\n=== RC: Control sets ===")

# No controls (bivariate + yearmonth FE)
run_lpm(
    "rc/controls/sets/none",
    "modules/robustness/controls.md#control-sets", "G2",
    "any_fda_reaction", TREATMENT_CONTEMP, [],
    "yearmo_str", "yearmonth FE", df,
    "hetero", "Full sample", "no controls (treatment + yearmonth FE only)",
    focal_var="v3",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/none", "family": "sets", "n_controls": 0})

# Condition dummies only
run_lpm(
    "rc/controls/sets/condition_dummies_only",
    "modules/robustness/controls.md#control-sets", "G2",
    "any_fda_reaction", TREATMENT_CONTEMP, CONDITION_DUMMIES,
    "yearmo_str", "yearmonth FE", df,
    "hetero", "Full sample", "condition dummies only",
    focal_var="v3",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/condition_dummies_only",
                "family": "sets", "n_controls": 3})

# Dappr_cats only (no condition dummies -- absorbed by treatment interactions)
run_lpm(
    "rc/controls/sets/dappr_only",
    "modules/robustness/controls.md#control-sets", "G2",
    "any_fda_reaction", TREATMENT_CONTEMP, DAPPR_CONTROLS,
    "yearmo_str", "yearmonth FE", df,
    "hetero", "Full sample", "Dappr_cats only",
    focal_var="v3",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/dappr_only",
                "family": "sets", "n_controls": 4})

# Full available (condition + Dappr_cats = baseline)
run_lpm(
    "rc/controls/sets/full",
    "modules/robustness/controls.md#control-sets", "G2",
    "any_fda_reaction", TREATMENT_CONTEMP, CONDITION_DUMMIES + DAPPR_CONTROLS,
    "yearmo_str", "yearmonth FE", df,
    "hetero", "Full sample", "full available (c1-c3 + Dappr_cats)",
    focal_var="v3",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/full",
                "family": "sets", "n_controls": 7})


# ============================================================
# RC: CONTROLS -- Block drops
# ============================================================

print("\n=== RC: Controls block drops ===")

# Drop all Dappr_cats
run_lpm(
    "rc/controls/block/drop_dapp",
    "modules/robustness/controls.md#block-drops", "G2",
    "any_fda_reaction", TREATMENT_CONTEMP, CONDITION_DUMMIES,
    "yearmo_str", "yearmonth FE", df,
    "hetero", "Full sample", "drop all Dappr_cats",
    focal_var="v3",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/block/drop_dapp", "family": "block",
                "dropped_block": "dappr_cats"})

# Drop condition dummies (leave Dappr_cats)
run_lpm(
    "rc/controls/block/drop_condition",
    "modules/robustness/controls.md#block-drops", "G2",
    "any_fda_reaction", TREATMENT_CONTEMP, DAPPR_CONTROLS,
    "yearmo_str", "yearmonth FE", df,
    "hetero", "Full sample", "drop condition dummies",
    focal_var="v3",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/block/drop_condition", "family": "block",
                "dropped_block": "condition_dummies"})


# ============================================================
# RC: SAMPLE RESTRICTIONS
# ============================================================

print("\n=== RC: Sample restrictions ===")

# Exclude Rofecoxib (Vioxx -- withdrawn 2004, high-profile drug)
df_no_rofecoxib = df[df['mastername'] != 'Rofecoxib'].copy()
run_lpm(
    "rc/sample/restriction/exclude_rofecoxib",
    "modules/robustness/sample.md#sample-restrictions", "G2",
    "any_fda_reaction", TREATMENT_CONTEMP, CONDITION_DUMMIES + DAPPR_CONTROLS,
    "yearmo_str", "yearmonth FE", df_no_rofecoxib,
    "hetero", f"Exclude Rofecoxib, N={len(df_no_rofecoxib)}",
    "c1-c3 + Dappr_cats + yearmonth FE",
    focal_var="v3",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/exclude_rofecoxib",
                "excluded_drug": "Rofecoxib"})

# Exclude Valdecoxib (another COX-2 inhibitor, withdrawn 2005)
df_no_valdecoxib = df[df['mastername'] != 'Valdecoxib'].copy()
run_lpm(
    "rc/sample/restriction/exclude_valdecoxib",
    "modules/robustness/sample.md#sample-restrictions", "G2",
    "any_fda_reaction", TREATMENT_CONTEMP, CONDITION_DUMMIES + DAPPR_CONTROLS,
    "yearmo_str", "yearmonth FE", df_no_valdecoxib,
    "hetero", f"Exclude Valdecoxib, N={len(df_no_valdecoxib)}",
    "c1-c3 + Dappr_cats + yearmonth FE",
    focal_var="v3",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/exclude_valdecoxib",
                "excluded_drug": "Valdecoxib"})

# Exclude both COX-2 inhibitors
df_no_cox2 = df[~df['mastername'].isin(['Rofecoxib', 'Valdecoxib'])].copy()
run_lpm(
    "rc/sample/restriction/exclude_cox2",
    "modules/robustness/sample.md#sample-restrictions", "G2",
    "any_fda_reaction", TREATMENT_CONTEMP, CONDITION_DUMMIES + DAPPR_CONTROLS,
    "yearmo_str", "yearmonth FE", df_no_cox2,
    "hetero", f"Exclude COX-2 inhibitors, N={len(df_no_cox2)}",
    "c1-c3 + Dappr_cats + yearmonth FE",
    focal_var="v3",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/exclude_cox2",
                "excluded_drugs": ["Rofecoxib", "Valdecoxib"]})

# Trim veryserious at 1st/99th percentile
q01 = df['veryserious'].quantile(0.01)
q99 = df['veryserious'].quantile(0.99)
df_trim = df[(df['veryserious'] >= q01) & (df['veryserious'] <= q99)].copy()
# Re-create interaction vars for trimmed sample
df_trim['v1'] = df_trim['veryserious'] * df_trim['c1']
df_trim['v2'] = df_trim['veryserious'] * df_trim['c2']
df_trim['v3'] = df_trim['veryserious'] * df_trim['c3']
df_trim['v4'] = df_trim['veryserious'] * df_trim['c4']

run_lpm(
    "rc/sample/outliers/trim_y_1_99",
    "modules/robustness/sample.md#outliers", "G2",
    "any_fda_reaction", TREATMENT_CONTEMP, CONDITION_DUMMIES + DAPPR_CONTROLS,
    "yearmo_str", "yearmonth FE", df_trim,
    "hetero", f"Trim veryserious [1%,99%], N={len(df_trim)}",
    "c1-c3 + Dappr_cats + yearmonth FE",
    focal_var="v3",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_1_99",
                "trim_var": "veryserious", "lower_q": 0.01, "upper_q": 0.99})

# Trim at 5th/95th
q05 = df['veryserious'].quantile(0.05)
q95 = df['veryserious'].quantile(0.95)
df_trim5 = df[(df['veryserious'] >= q05) & (df['veryserious'] <= q95)].copy()
df_trim5['v1'] = df_trim5['veryserious'] * df_trim5['c1']
df_trim5['v2'] = df_trim5['veryserious'] * df_trim5['c2']
df_trim5['v3'] = df_trim5['veryserious'] * df_trim5['c3']
df_trim5['v4'] = df_trim5['veryserious'] * df_trim5['c4']

run_lpm(
    "rc/sample/outliers/trim_y_5_95",
    "modules/robustness/sample.md#outliers", "G2",
    "any_fda_reaction", TREATMENT_CONTEMP, CONDITION_DUMMIES + DAPPR_CONTROLS,
    "yearmo_str", "yearmonth FE", df_trim5,
    "hetero", f"Trim veryserious [5%,95%], N={len(df_trim5)}",
    "c1-c3 + Dappr_cats + yearmonth FE",
    focal_var="v3",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_5_95",
                "trim_var": "veryserious", "lower_q": 0.05, "upper_q": 0.95})

# Year 2003 only
df_2003 = df[df['year'] == 2003].copy()
run_lpm(
    "rc/sample/restriction/year_2003",
    "modules/robustness/sample.md#sample-restrictions", "G2",
    "any_fda_reaction", TREATMENT_CONTEMP, CONDITION_DUMMIES + DAPPR_CONTROLS,
    "yearmo_str", "yearmonth FE", df_2003,
    "hetero", f"Year 2003 only, N={len(df_2003)}",
    "c1-c3 + Dappr_cats + yearmonth FE",
    focal_var="v3",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/year_2003", "year": 2003})

# Year 2004 only
df_2004 = df[df['year'] == 2004].copy()
run_lpm(
    "rc/sample/restriction/year_2004",
    "modules/robustness/sample.md#sample-restrictions", "G2",
    "any_fda_reaction", TREATMENT_CONTEMP, CONDITION_DUMMIES + DAPPR_CONTROLS,
    "yearmo_str", "yearmonth FE", df_2004,
    "hetero", f"Year 2004 only, N={len(df_2004)}",
    "c1-c3 + Dappr_cats + yearmonth FE",
    focal_var="v3",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/year_2004", "year": 2004})


# ============================================================
# RC: FUNCTIONAL FORM -- log treatment
# ============================================================

print("\n=== RC: Functional form ===")

# Log(1+veryserious) interactions
run_lpm(
    "rc/form/outcome/log_treatment",
    "modules/robustness/functional_form.md#treatment-transformations", "G2",
    "any_fda_reaction", TREATMENT_LOG, CONDITION_DUMMIES + DAPPR_CONTROLS,
    "yearmo_str", "yearmonth FE", df,
    "hetero", "Full sample, log(1+veryserious) treatment",
    "c1-c3 + Dappr_cats + yearmonth FE, log treatment",
    focal_var="log_v3",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/log_treatment", "transform": "log1p"})

# Logit with log treatment
run_logit(
    "rc/form/outcome/log_treatment_logit",
    "modules/robustness/functional_form.md#treatment-transformations", "G2",
    "any_fda_reaction", TREATMENT_LOG, baseline_controls_G2,
    ["yearmo_str"], "yearmonth dummies", df,
    "Full sample, log(1+veryserious) logit", "c1-c3 + Dappr_cats + yearmonth, log treatment",
    focal_var="log_v3",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/log_treatment_logit", "transform": "log1p", "estimator": "logit"})

# Scaled veryserious/100
run_lpm(
    "rc/form/outcome/scaled_treatment",
    "modules/robustness/functional_form.md#treatment-transformations", "G2",
    "any_fda_reaction", ['vs_100_1', 'vs_100_2', 'vs_100_3', 'vs_100_4'],
    CONDITION_DUMMIES + DAPPR_CONTROLS,
    "yearmo_str", "yearmonth FE", df,
    "hetero", "Full sample, veryserious/100",
    "c1-c3 + Dappr_cats + yearmonth FE, scaled treatment",
    focal_var="vs_100_3",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/scaled_treatment", "transform": "divide_100"})


# ============================================================
# RC: OUTCOME -- countfda (count of FDA changes)
# ============================================================

print("\n=== RC: Outcome -- countfda ===")

run_lpm(
    "rc/form/outcome/countfda_lpm",
    "modules/robustness/functional_form.md#outcome-alternatives", "G2",
    "countfda", TREATMENT_CONTEMP, CONDITION_DUMMIES + DAPPR_CONTROLS,
    "yearmo_str", "yearmonth FE", df,
    "hetero", "Full sample, countfda outcome",
    "c1-c3 + Dappr_cats + yearmonth FE",
    focal_var="v3",
    axis_block_name="estimation",
    axis_block={"spec_id": "rc/form/outcome/countfda", "outcome": "countfda"})

# With drug FE
run_lpm(
    "rc/form/outcome/countfda_drug_fe",
    "modules/robustness/functional_form.md#outcome-alternatives", "G2",
    "countfda", TREATMENT_CONTEMP, DAPPR_CONTROLS,
    "drug_str + yearmo_str", "drug + yearmonth FE", df,
    "hetero", "Full sample, countfda + drug FE",
    "Dappr_cats + drug + yearmonth FE",
    focal_var="v3",
    axis_block_name="estimation",
    axis_block={"spec_id": "rc/form/outcome/countfda_drug_fe",
                "outcome": "countfda", "fe": "drug+yearmonth"})


# ============================================================
# RC: OUTCOME -- specific FDA label sub-types
# ============================================================

print("\n=== RC: Outcome -- FDA subtypes ===")

for fda_type in ['warning', 'precaution', 'adverse_reaction']:
    # Only run if there's enough variation
    if df[fda_type].sum() >= 5:
        run_lpm(
            f"rc/form/outcome/{fda_type}_lpm",
            "modules/robustness/functional_form.md#outcome-alternatives", "G2",
            fda_type, TREATMENT_CONTEMP, CONDITION_DUMMIES + DAPPR_CONTROLS,
            "yearmo_str", "yearmonth FE", df,
            "hetero", f"Full sample, {fda_type} outcome",
            "c1-c3 + Dappr_cats + yearmonth FE",
            focal_var="v3",
            axis_block_name="estimation",
            axis_block={"spec_id": f"rc/form/outcome/{fda_type}",
                        "outcome": fda_type})


# ============================================================
# RC: FE VARIATIONS
# ============================================================

print("\n=== RC: FE variations ===")

# No FE (pooled)
run_lpm(
    "rc/fe/drop/yearmonth",
    "modules/robustness/fixed_effects.md#dropping-fe", "G2",
    "any_fda_reaction", TREATMENT_CONTEMP, CONDITION_DUMMIES + DAPPR_CONTROLS,
    "", "none (pooled)", df,
    "hetero", "Full sample, no FE",
    "c1-c3 + Dappr_cats, no FE",
    focal_var="v3",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/drop/yearmonth", "fe": "none"})

# Drug FE only (no yearmonth)
run_lpm(
    "rc/fe/alt/drug_only",
    "modules/robustness/fixed_effects.md#alternative-fe", "G2",
    "any_fda_reaction", TREATMENT_CONTEMP, DAPPR_CONTROLS,
    "drug_str", "drug FE", df,
    "hetero", "Full sample, drug FE only",
    "Dappr_cats + drug FE",
    focal_var="v3",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/alt/drug_only", "fe": "drug"})

# Drug + yearmonth FE
run_lpm(
    "rc/fe/alt/drug_plus_yearmonth",
    "modules/robustness/fixed_effects.md#alternative-fe", "G2",
    "any_fda_reaction", TREATMENT_CONTEMP, DAPPR_CONTROLS,
    "drug_str + yearmo_str", "drug + yearmonth FE", df,
    "hetero", "Full sample, drug + yearmonth FE",
    "Dappr_cats + drug + yearmonth FE",
    focal_var="v3",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/alt/drug_plus_yearmonth", "fe": "drug+yearmonth"})

# Year FE only (coarser time)
df['year_str'] = df['year'].astype(str)
run_lpm(
    "rc/fe/alt/year_only",
    "modules/robustness/fixed_effects.md#alternative-fe", "G2",
    "any_fda_reaction", TREATMENT_CONTEMP, CONDITION_DUMMIES + DAPPR_CONTROLS,
    "year_str", "year FE", df,
    "hetero", "Full sample, year FE only",
    "c1-c3 + Dappr_cats + year FE",
    focal_var="v3",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/alt/year_only", "fe": "year"})

# Condition + yearmonth FE
df['condition_str'] = df['condition'].astype(str)
run_lpm(
    "rc/fe/alt/condition_plus_yearmonth",
    "modules/robustness/fixed_effects.md#alternative-fe", "G2",
    "any_fda_reaction", TREATMENT_CONTEMP, DAPPR_CONTROLS,
    "condition_str + yearmo_str", "condition + yearmonth FE", df,
    "hetero", "Full sample, condition + yearmonth FE",
    "Dappr_cats + condition + yearmonth FE",
    focal_var="v3",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/alt/condition_plus_yearmonth",
                "fe": "condition+yearmonth"})


# ============================================================
# RC: RANDOM CONTROL SUBSETS (randomized draws)
# ============================================================

print("\n=== RC: Random control subsets ===")

rng = np.random.RandomState(114747)
all_avail_controls = CONDITION_DUMMIES + DAPPR_CONTROLS  # 7 controls total

for draw_i in range(1, 9):
    k = rng.randint(1, len(all_avail_controls) + 1)
    chosen = list(rng.choice(all_avail_controls, size=k, replace=False))
    excluded = [v for v in all_avail_controls if v not in chosen]
    spec_id = f"rc/controls/subset/random_{draw_i:03d}"

    run_lpm(
        spec_id, "modules/robustness/controls.md#subset-generation", "G2",
        "any_fda_reaction", TREATMENT_CONTEMP, chosen,
        "yearmo_str", "yearmonth FE", df,
        "hetero", "Full sample", f"random subset draw {draw_i} ({len(chosen)} controls)",
        focal_var="v3",
        axis_block_name="controls",
        axis_block={"spec_id": spec_id, "family": "subset",
                    "seed": 114747, "draw_index": draw_i,
                    "included": chosen, "excluded": excluded,
                    "n_controls": len(chosen)})


# ============================================================
# RC: TREATMENT FORM -- Lagged ADRs
# ============================================================

print("\n=== RC: Lagged treatment ===")

# 1-month lagged veryserious
df_lag1 = df.dropna(subset=['veryserious_lag1']).copy()
df_lag1['vlag1_1'] = df_lag1['veryserious_lag1'] * df_lag1['c1']
df_lag1['vlag1_2'] = df_lag1['veryserious_lag1'] * df_lag1['c2']
df_lag1['vlag1_3'] = df_lag1['veryserious_lag1'] * df_lag1['c3']
df_lag1['vlag1_4'] = df_lag1['veryserious_lag1'] * df_lag1['c4']

run_lpm(
    "rc/form/treatment/lagged_1month",
    "modules/robustness/functional_form.md#lagged-treatment", "G2",
    "any_fda_reaction", ['vlag1_1', 'vlag1_2', 'vlag1_3', 'vlag1_4'],
    CONDITION_DUMMIES + DAPPR_CONTROLS,
    "yearmo_str", "yearmonth FE", df_lag1,
    "hetero", f"Lagged 1-month, N={len(df_lag1)}",
    "c1-c3 + Dappr_cats + yearmonth FE, lagged 1m treatment",
    focal_var="vlag1_3",
    axis_block_name="estimation",
    axis_block={"spec_id": "rc/form/treatment/lagged_1month", "lag_months": 1})

# 3-month lagged
df_lag3 = df.dropna(subset=['veryserious_lag3']).copy()
df_lag3['vlag3_1'] = df_lag3['veryserious_lag3'] * df_lag3['c1']
df_lag3['vlag3_2'] = df_lag3['veryserious_lag3'] * df_lag3['c2']
df_lag3['vlag3_3'] = df_lag3['veryserious_lag3'] * df_lag3['c3']
df_lag3['vlag3_4'] = df_lag3['veryserious_lag3'] * df_lag3['c4']

run_lpm(
    "rc/form/treatment/lagged_3month",
    "modules/robustness/functional_form.md#lagged-treatment", "G2",
    "any_fda_reaction", ['vlag3_1', 'vlag3_2', 'vlag3_3', 'vlag3_4'],
    CONDITION_DUMMIES + DAPPR_CONTROLS,
    "yearmo_str", "yearmonth FE", df_lag3,
    "hetero", f"Lagged 3-month, N={len(df_lag3)}",
    "c1-c3 + Dappr_cats + yearmonth FE, lagged 3m treatment",
    focal_var="vlag3_3",
    axis_block_name="estimation",
    axis_block={"spec_id": "rc/form/treatment/lagged_3month", "lag_months": 3})

# Cumulative 3-month as treatment
run_lpm(
    "rc/form/treatment/cumulative_3month",
    "modules/robustness/functional_form.md#cumulative-treatment", "G2",
    "any_fda_reaction", TREATMENT_CUM3, CONDITION_DUMMIES + DAPPR_CONTROLS,
    "yearmo_str", "yearmonth FE", df_cum3,
    "hetero", f"3-month cumulative, N={len(df_cum3)}",
    "c1-c3 + Dappr_cats + yearmonth FE",
    focal_var="v_33",
    axis_block_name="estimation",
    axis_block={"spec_id": "rc/form/treatment/cumulative_3month", "treatment_form": "cumulative_3month"})

# Cumulative 12-month as treatment
run_lpm(
    "rc/form/treatment/cumulative_12month",
    "modules/robustness/functional_form.md#cumulative-treatment", "G2",
    "any_fda_reaction", TREATMENT_CUM12, CONDITION_DUMMIES + DAPPR_CONTROLS,
    "yearmo_str", "yearmonth FE", df_cum12,
    "hetero", f"12-month cumulative, N={len(df_cum12)}",
    "c1-c3 + Dappr_cats + yearmonth FE",
    focal_var="v_123",
    axis_block_name="estimation",
    axis_block={"spec_id": "rc/form/treatment/cumulative_12month", "treatment_form": "cumulative_12month"})


# ============================================================
# G2 ADDITIONAL: Logit with various sub-specifications
# ============================================================

print("\n=== G2 Additional logit specs ===")

# Logit with drug FE (conditional logit approximation)
run_logit(
    "design/panel_fixed_effects/estimator/logit_drug_fe",
    "designs/panel_fixed_effects.md#estimator-alternatives", "G2",
    "any_fda_reaction", TREATMENT_CONTEMP, DAPPR_CONTROLS,
    ["drug_str", "yearmo_str"], "drug + yearmonth dummies", df,
    f"Full sample, logit with drug + yearmonth dummies",
    "Dappr_cats + drug + yearmonth dummies",
    focal_var="v3",
    axis_block_name="estimation",
    axis_block={"spec_id": "design/logit_drug_fe", "estimator": "logit", "fe": "drug+yearmonth"})

# Logit exclude Rofecoxib
run_logit(
    "rc/sample/logit_exclude_rofecoxib",
    "modules/robustness/sample.md#sample-restrictions", "G2",
    "any_fda_reaction", TREATMENT_CONTEMP, baseline_controls_G2,
    ["yearmo_str"], "yearmonth dummies", df_no_rofecoxib,
    f"Logit exclude Rofecoxib, N={len(df_no_rofecoxib)}",
    "c1-c3 + Dappr_cats + yearmonth",
    focal_var="v3",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/logit_exclude_rofecoxib",
                "excluded_drug": "Rofecoxib", "estimator": "logit"})

# Logit no controls
run_logit(
    "rc/controls/logit_no_controls",
    "modules/robustness/controls.md#control-sets", "G2",
    "any_fda_reaction", TREATMENT_CONTEMP, [],
    ["yearmo_str"], "yearmonth dummies", df,
    "Full sample, logit no controls", "yearmonth only",
    focal_var="v3",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/logit_no_controls", "estimator": "logit", "n_controls": 0})

# Logit condition dummies only
run_logit(
    "rc/controls/logit_condition_only",
    "modules/robustness/controls.md#control-sets", "G2",
    "any_fda_reaction", TREATMENT_CONTEMP, CONDITION_DUMMIES,
    ["yearmo_str"], "yearmonth dummies", df,
    "Full sample, logit condition dummies only", "c1-c3 + yearmonth",
    focal_var="v3",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/logit_condition_only", "estimator": "logit", "n_controls": 3})


# ============================================================
# INFERENCE VARIANTS (on LPM baseline)
# ============================================================

print("\n=== Inference variants ===")

# Record the LPM baseline for inference reference
baseline_lpm_controls = " + ".join(TREATMENT_CONTEMP + CONDITION_DUMMIES + DAPPR_CONTROLS)
baseline_lpm_formula = f"any_fda_reaction ~ {baseline_lpm_controls}"
baseline_lpm_fe = "yearmo_str"

infer_counter = 0
# Find the baseline LPM run_id -- it's the first LPM spec
lpm_baseline_run_id = None
for r in results:
    if r['spec_id'] == 'design/panel_fixed_effects/estimator/lpm' and r['run_success'] == 1:
        lpm_baseline_run_id = r['spec_run_id']
        break

if lpm_baseline_run_id is None:
    # Use the logit baseline as fallback
    lpm_baseline_run_id = f"{PAPER_ID}_run_001"


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
            design={"panel_fixed_effects": design_audit},
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


# HC1 robust (no clustering)
run_inference_variant(
    lpm_baseline_run_id, "infer/se/robust/sandwich",
    "modules/inference/standard_errors.md#heteroskedasticity-robust", "G2",
    baseline_lpm_formula, baseline_lpm_fe, df, "v3",
    "hetero", "HC1 (robust, no clustering)")

# Cluster by drug
run_inference_variant(
    lpm_baseline_run_id, "infer/se/cluster/drug",
    "modules/inference/standard_errors.md#clustering", "G2",
    baseline_lpm_formula, baseline_lpm_fe, df, "v3",
    {"CRV1": "drug_str"}, "cluster(drug)")

# Cluster by condition (4 clusters -- likely too few but matches surface)
run_inference_variant(
    lpm_baseline_run_id, "infer/se/cluster/condition",
    "modules/inference/standard_errors.md#clustering", "G2",
    baseline_lpm_formula, baseline_lpm_fe, df, "v3",
    {"CRV1": "condition_str"}, "cluster(condition)")

# iid (homoskedastic)
run_inference_variant(
    lpm_baseline_run_id, "infer/se/iid",
    "modules/inference/standard_errors.md#iid", "G2",
    baseline_lpm_formula, baseline_lpm_fe, df, "v3",
    "iid", "iid (homoskedastic)")


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

if len(failed) > 0:
    print("\n--- Failed specs ---")
    for _, row in failed.iterrows():
        print(f"  {row['spec_id']}: {row['run_error'][:100]}")

if len(successful) > 0:
    base_row = spec_df[spec_df['spec_id'] == 'baseline__table2_logit_contemporaneous']
    if len(base_row) > 0:
        print(f"\nBaseline coef on v3: {base_row['coefficient'].values[0]:.6f}")
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
md_lines.append("# Specification Search Report: 114747-V1")
md_lines.append("")
md_lines.append("**Paper:** Dranove, Hughes, and Meltzer (2003/2012), \"Incentives and Promotion for Adverse Drug Reactions\", AEJ: Economic Policy")
md_lines.append("")
md_lines.append("## Data Limitations")
md_lines.append("")
md_lines.append("The provided .dta file is missing key treatment variables (promotion/advertising expenditures: q1totalexp, q2q4totalexp, etc.) and many controls (generic, sh_count* demographics, age-gender shares, permonths exposure). This means **G1 (Poisson ADR ~ promotion) cannot be executed**. Analysis focuses on **G2 (FDA labeling changes ~ ADR counts)** using available variables.")
md_lines.append("")
md_lines.append("## Baseline Specification (G2)")
md_lines.append("")
md_lines.append("- **Design:** Logit / LPM (Linear Probability Model)")
md_lines.append("- **Outcome:** any_fda_reaction (binary: any FDA labeling change)")
md_lines.append("- **Treatment:** v1-v4 (veryserious ADR count interacted with condition dummies)")
md_lines.append("- **Focal coefficient:** v3 (arthritis condition, matching Table 3 focus)")
md_lines.append("- **Controls:** c1-c3 (condition dummies) + Dappr_cats_1-4 (drug approval age)")
md_lines.append("- **Fixed effects:** Year-month dummies")
md_lines.append("- **Note:** Missing controls (generic, $char, $age) reduce precision vs. paper's specification")
md_lines.append("")

if len(successful) > 0:
    base_row = spec_df[spec_df['spec_id'] == 'baseline__table2_logit_contemporaneous']
    if len(base_row) > 0:
        bc = base_row.iloc[0]
        md_lines.append(f"| Statistic | Value |")
        md_lines.append(f"|-----------|-------|")
        md_lines.append(f"| Coefficient (v3) | {bc['coefficient']:.6f} |")
        md_lines.append(f"| Std. Error | {bc['std_error']:.6f} |")
        md_lines.append(f"| p-value | {bc['p_value']:.6f} |")
        md_lines.append(f"| 95% CI | [{bc['ci_lower']:.6f}, {bc['ci_upper']:.6f}] |")
        md_lines.append(f"| N | {bc['n_obs']:.0f} |")
        if not np.isnan(bc['r_squared']):
            md_lines.append(f"| Pseudo R-squared | {bc['r_squared']:.4f} |")
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
    "Baseline (logit)": successful[successful['spec_id'].str.startswith('baseline__')],
    "Design/Estimator": successful[successful['spec_id'].str.startswith('design/')],
    "Controls LOO": successful[successful['spec_id'].str.startswith('rc/controls/loo/')],
    "Controls Sets": successful[successful['spec_id'].str.startswith('rc/controls/sets/')],
    "Controls Block": successful[successful['spec_id'].str.startswith('rc/controls/block/')],
    "Controls Subset": successful[successful['spec_id'].str.startswith('rc/controls/subset/')],
    "Controls Logit": successful[successful['spec_id'].str.startswith('rc/controls/logit')],
    "Sample Restrictions": successful[successful['spec_id'].str.startswith('rc/sample/')],
    "Functional Form": successful[successful['spec_id'].str.startswith('rc/form/')],
    "Fixed Effects": successful[successful['spec_id'].str.startswith('rc/fe/')],
}

for cat_name, cat_df in categories.items():
    if len(cat_df) > 0:
        n_sig_cat = (cat_df['p_value'] < 0.05).sum()
        coef_range = f"[{cat_df['coefficient'].min():.6f}, {cat_df['coefficient'].max():.6f}]"
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
    md_lines.append("**Note:** This assessment is limited by missing data. The paper's G1 (promotion -> ADR) specifications cannot be run because promotion expenditure variables are not in the provided dataset. G2 results (ADR -> FDA labeling) are estimated with fewer controls than the paper, which may affect inference.")

md_lines.append("")
md_lines.append(f"Surface hash: `{SURFACE_HASH}`")
md_lines.append("")

with open(f"{OUTPUT_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write("\n".join(md_lines))

print(f"Wrote SPECIFICATION_SEARCH.md")
print("\nDone!")
