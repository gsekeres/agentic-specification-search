"""
Specification Search Script for Deming (2014)
"Using School Choice Lotteries to Test Measures of School Effectiveness"
American Economic Review Papers & Proceedings, 104(5), 406-411.

Paper ID: 112805-V1

Surface-driven execution:
  - G1: testz2003 ~ VA (IV: lott_VA) + lagged_scores | lottery_FE
  - IV regression: school VAM instrumented by lottery assignment
  - Specification grid: 2 VAM models x 3 estimation methods x 3 sample windows
    x 2 counterfactual schools = 36 core VAM variants
  - Plus: controls variants, outcome variants, sample splits

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

from sklearn.linear_model import LinearRegression

# Add scripts dir for utilities
sys.path.insert(0, "scripts")
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash, software_block
)

PAPER_ID = "112805-V1"
DATA_DIR = "data/downloads/extracted/112805-V1"
OUTPUT_DIR = DATA_DIR
DATA_PATH = f"{DATA_DIR}/Deming_AERPandP_datafolder/cms_VAManalysis.dta"

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

bg = surface_obj["baseline_groups"][0]
design_audit = bg["design_audit"]
inference_canonical = bg["inference_plan"]["canonical"]


# ============================================================
# Data Loading and Preparation
# ============================================================

print("Loading data...")
df_raw = pd.read_stata(DATA_PATH)
print(f"Loaded data: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

# Convert float32 to float64 for precision
for col in df_raw.columns:
    if df_raw[col].dtype == np.float32:
        df_raw[col] = df_raw[col].astype(np.float64)

# Build sample indicators (following do-file)
df_raw['sample'] = ((df_raw['future_grd'] >= 4) & (df_raw['future_grd'] <= 8) &
                     (df_raw['miss_02'] == 0)).astype(int)
df_raw['onmargin_sample'] = ((df_raw['onmargin'] == 1) & (df_raw['sample'] == 1)).astype(int)

# Build year_g variables (following do-file: year when student was in grade g
# with grade g-1 the prior year)
for g in range(4, 9):
    df_raw[f'year_{g}'] = np.nan
    f = g - 1
    for y in range(1998, 2005):
        x = y - 1
        gcol = f'grade{y}'
        gcol_prev = f'grade{x}'
        if gcol in df_raw.columns and gcol_prev in df_raw.columns:
            mask = (df_raw[gcol] == g) & (df_raw[gcol_prev] == f)
            df_raw.loc[mask, f'year_{g}'] = y

# Build test_g (average of math and read raw scores)
for g in range(3, 9):
    df_raw[f'test_{g}'] = df_raw[[f'math_{g}', f'read_{g}']].mean(axis=1)

# Build lagged score imputation variables (for VAM construction)
for lag in range(3, 8):
    for x in ['math', 'read']:
        df_raw[f'{x}_{lag}_imp'] = df_raw[f'{x}_{lag}'].copy()
        df_raw[f'{x}_{lag}_miss'] = df_raw[f'{x}_{lag}'].isna().astype(float)
        df_raw.loc[df_raw[f'{x}_{lag}_miss'] == 1, f'{x}_{lag}_imp'] = 0.0
        df_raw[f'{x}_{lag}_imp_sq'] = df_raw[f'{x}_{lag}_imp'] ** 2
        df_raw[f'{x}_{lag}_imp_cub'] = df_raw[f'{x}_{lag}_imp'] ** 3

# Build z-score imputation variables for 2002
for y in [2002]:
    for x in ['math', 'read']:
        df_raw[f'{x}_{y}_imp'] = df_raw[f'{x}z{y}'].copy()
        df_raw[f'{x}_{y}_miss'] = df_raw[f'{x}z{y}'].isna().astype(float)
        df_raw.loc[df_raw[f'{x}_{y}_miss'] == 1, f'{x}_{y}_imp'] = 0.0
        df_raw[f'{x}_{y}_imp_sq'] = df_raw[f'{x}_{y}_imp'] ** 2
        df_raw[f'{x}_{y}_imp_cub'] = df_raw[f'{x}_{y}_imp'] ** 3

# Build testz variables (average of math and reading z-scores)
for y in range(1998, 2005):
    df_raw[f'testz{y}'] = df_raw[[f'mathz{y}', f'readz{y}']].mean(axis=1)

df_raw = df_raw.copy()  # defragment

print(f"Analysis sample (onmargin==1): {df_raw['onmargin_sample'].sum()} students")
print(f"Non-lottery sample for VAM: {(df_raw['onmargin_sample'] != 1).sum()} students")


# ============================================================
# VAM Construction
# ============================================================
# Following the do-file: estimate VAMs on non-lottery students, then merge
# school-level VAMs back to the full dataset.
#
# Model 1 (levels): regress test_g on nothing (just intercept)
# Model 2 (gains): regress test_g on lagged scores (imputed, with polynomials)
# Estimation methods:
#   ar = average residual (OLS, then average residuals by school)
#   mix = mixed effects (school random effect) -- we approximate with OLS + shrinkage
#   FE = school fixed effects
# Training windows: 02 (year==2002 only), 2yr (2001-2002), all (<=2002)

print("\nConstructing VAMs...")

# School-level VAM storage: keyed by (model, estimation, window, grade) -> dict of school -> VAM
vam_store = {}

for g in range(4, 9):
    lag = g - 1
    # Define model controls
    model1_controls = []  # levels only - no controls
    model2_controls = [
        f'math_{lag}_imp', f'read_{lag}_imp',
        f'math_{lag}_imp_sq', f'read_{lag}_imp_sq',
        f'math_{lag}_imp_cub', f'read_{lag}_imp_cub',
        f'math_{lag}_miss', f'read_{lag}_miss'
    ]

    # Define windows
    windows = {
        '02': lambda yr: yr == 2002,
        '2yr': lambda yr: (yr >= 2001) & (yr <= 2002),
        'all': lambda yr: yr <= 2002
    }

    for model_num, (model_name, controls) in enumerate(
        [('mod1', model1_controls), ('mod2', model2_controls)], start=1):
        for window_name, window_fn in windows.items():
            # Base data: non-lottery students in the window
            base_mask = (
                (df_raw['onmargin_sample'] != 1) &
                window_fn(df_raw[f'year_{g}']) &
                df_raw[f'school_{g}'].notna() &
                df_raw[f'test_{g}'].notna()
            )
            if controls:
                for c in controls:
                    base_mask = base_mask & df_raw[c].notna()

            sub = df_raw[base_mask].copy()

            if len(sub) < 10:
                continue

            y_vals = sub[f'test_{g}'].values

            # === Average Residual (ar) ===
            if controls:
                X = sub[controls].values
                reg = LinearRegression().fit(X, y_vals)
                resid = y_vals - reg.predict(X)
            else:
                resid = y_vals - y_vals.mean()

            sub['_resid'] = resid
            school_vam = sub.groupby(f'school_{g}')['_resid'].mean().to_dict()
            vam_store[(model_name, 'ar', window_name, g)] = school_vam

            # === School Fixed Effects (FE) ===
            # Use pyfixest to get school FE estimates
            try:
                sub_fe = sub[[f'test_{g}', f'school_{g}'] + controls].dropna().copy()
                sub_fe[f'school_{g}_str'] = sub_fe[f'school_{g}'].astype(int).astype(str)
                if controls:
                    formula_fe = f"test_{g} ~ " + " + ".join(controls) + f" | school_{g}_str"
                else:
                    formula_fe = f"test_{g} ~ 1 | school_{g}_str"
                m_fe = pf.feols(formula_fe, data=sub_fe)
                fe_vals = m_fe.fixef()
                if isinstance(fe_vals, dict):
                    # Get the school FE
                    fe_key = [k for k in fe_vals.keys() if f'school_{g}_str' in k]
                    if fe_key:
                        fe_dict = fe_vals[fe_key[0]]
                        # Convert string keys back to float for mapping
                        school_fe_vam = {float(k): v for k, v in fe_dict.items()}
                        vam_store[(model_name, 'FE', window_name, g)] = school_fe_vam
                    else:
                        # Try first key
                        first_key = list(fe_vals.keys())[0]
                        fe_dict = fe_vals[first_key]
                        school_fe_vam = {float(k): v for k, v in fe_dict.items()}
                        vam_store[(model_name, 'FE', window_name, g)] = school_fe_vam
            except Exception as e:
                print(f"  Warning: FE VAM failed for {model_name}/{window_name}/grade{g}: {e}")

            # === Mixed Effects (mix) ===
            # Approximate with shrinkage: shrink school means toward grand mean
            # using the reliability ratio formula: lambda = var_between / (var_between + var_within/n_j)
            try:
                school_groups = sub.groupby(f'school_{g}')['_resid']
                school_means = school_groups.mean()
                school_counts = school_groups.count()
                grand_mean = resid.mean()

                # Variance decomposition
                var_total = np.var(resid)
                var_within = sub.groupby(f'school_{g}')['_resid'].var().mean()
                var_between = max(var_total - var_within, 0.001)

                # Shrinkage: mix_vam_j = lambda_j * school_mean_j + (1-lambda_j) * grand_mean
                shrunk = {}
                for school, mean_val in school_means.items():
                    n_j = school_counts[school]
                    lam = var_between / (var_between + var_within / max(n_j, 1))
                    shrunk[school] = lam * mean_val + (1 - lam) * grand_mean
                vam_store[(model_name, 'mix', window_name, g)] = shrunk
            except Exception as e:
                print(f"  Warning: mix VAM failed for {model_name}/{window_name}/grade{g}: {e}")

print(f"Constructed {len(vam_store)} VAM variants")


# ============================================================
# Merge VAMs to analysis data
# ============================================================

# School variable mappings:
# as = schl_d20 (assigned school, day 20 enrollment)
# ch1 = choice1_schl (first choice)
# hm = home0203 (neighborhood school)
# ch2 = choice2_schl, ch3 = choice3_schl (for weighted alternative)

SCHOOL_MAP = {
    'as': 'schl_d20',
    'ch1': 'choice1_schl',
    'hm': 'home0203',
    'ch2': 'choice2_schl',
    'ch3': 'choice3_schl',
}

# Build analysis dataframe with VAM columns
analysis = df_raw[df_raw['onmargin_sample'] == 1].copy()
print(f"\nMerging VAMs to analysis sample (N={len(analysis)})...")

for (model_name, est_method, window, grade), school_vam in vam_store.items():
    for schl_label, schl_col in SCHOOL_MAP.items():
        col_name = f'{schl_label}_{model_name}{est_method}_{window}_test'
        # Map school VAM to students based on their school assignment and grade
        mask = analysis['future_grd'] == grade
        if mask.sum() > 0:
            if col_name not in analysis.columns:
                analysis[col_name] = np.nan
            analysis.loc[mask, col_name] = (
                analysis.loc[mask, schl_col].map(
                    lambda s, sv=school_vam: sv.get(s, np.nan)
                )
            )

# Construct weighted alternative counterfactual
# VAMalt = (margin2)*(VAMch2) + (1-margin2)*[(margin3*VAMch3) + (1-margin3)*VAMhome]
for model_name in ['mod1', 'mod2']:
    for est_method in ['ar', 'mix', 'FE']:
        for window in ['02', '2yr', 'all']:
            ch2_col = f'ch2_{model_name}{est_method}_{window}_test'
            ch3_col = f'ch3_{model_name}{est_method}_{window}_test'
            hm_col = f'hm_{model_name}{est_method}_{window}_test'
            alt_col = f'alt_{model_name}{est_method}_{window}_test'

            if ch2_col in analysis.columns and ch3_col in analysis.columns and hm_col in analysis.columns:
                analysis[alt_col] = (
                    analysis['margin2'] * analysis[ch2_col] +
                    (1 - analysis['margin2']) * (
                        analysis['margin3'] * analysis[ch3_col] +
                        (1 - analysis['margin3']) * analysis[hm_col]
                    )
                )

# Construct VA (assigned school VAM) and lott_VA (lottery-determined instrument)
# VA = as_{model}{est}_{window}_test
# lott_VA = hm_{model}{est}_{window}_test if lottery==0
#           ch1_{model}{est}_{window}_test if lottery==1

# Also prepare the IV second-stage controls
# Controls: math_2002_imp, read_2002_imp, math_2002_imp_sq, math_2002_imp_cub,
#           read_2002_imp_sq, read_2002_imp_cub, math_2002_miss, read_2002_miss

IV_CONTROLS = [
    'math_2002_imp', 'read_2002_imp',
    'math_2002_imp_sq', 'math_2002_imp_cub',
    'read_2002_imp_sq', 'read_2002_imp_cub',
    'math_2002_miss', 'read_2002_miss'
]

# Make lottery_FE a string for pyfixest
analysis['lottery_FE_str'] = analysis['lottery_FE'].astype(int).astype(str)

print(f"VAM columns created: {len([c for c in analysis.columns if 'mod' in c])}")


# ============================================================
# Accumulators
# ============================================================

results = []
inference_results = []
spec_run_counter = 0


# ============================================================
# Helper: run_iv_spec (IV with FE via pyfixest)
# ============================================================

def run_iv_spec(spec_id, spec_tree_path, baseline_group_id,
                outcome_var, treatment_var, instrument_var, controls,
                fe_formula_str, fe_desc, data, vcov,
                sample_desc, controls_desc, counterfactual_desc,
                cluster_var="lottery_FE_str",
                axis_block_name=None, axis_block=None, notes=""):
    """Run a single IV specification and record results."""
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        controls_str = " + ".join(controls) if controls else "1"
        # pyfixest IV syntax: Y ~ controls | FE | endog ~ instrument
        if fe_formula_str:
            formula = f"{outcome_var} ~ {controls_str} | {fe_formula_str} | {treatment_var} ~ {instrument_var}"
        else:
            formula = f"{outcome_var} ~ {controls_str} | {treatment_var} ~ {instrument_var}"

        m = pf.feols(formula, data=data, vcov=vcov)

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
                       "method": "cluster", "cluster_vars": ["lottery_FE"]},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"iv_fe": design_audit},
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
            "fixed_effects": fe_desc,
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
            "run_success": 1,
            "run_error": ""
        })
        return run_id, coef_val, se_val, pval, nobs

    except Exception as e:
        err_msg = str(e)[:240]
        err_details = error_details_from_exception(e, stage="iv_estimation")
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
            "fixed_effects": fe_desc,
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
            "run_success": 0,
            "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


# ============================================================
# Helper: prepare IV data for a given VAM variant
# ============================================================

def prepare_iv_data(df, model, est, window, counterfactual='hm'):
    """Create VA (endogenous) and lott_VA (instrument) columns for a given VAM variant.

    Parameters:
    -----------
    df : DataFrame (analysis sample)
    model : str ('mod1' or 'mod2')
    est : str ('ar', 'mix', or 'FE')
    window : str ('02', '2yr', or 'all')
    counterfactual : str ('hm' for home school, 'alt' for weighted alternative)

    Returns:
    --------
    DataFrame with VA and lott_VA columns, or None if data insufficient
    """
    as_col = f'as_{model}{est}_{window}_test'
    cf_col = f'{counterfactual}_{model}{est}_{window}_test'
    ch1_col = f'ch1_{model}{est}_{window}_test'

    needed = [as_col]
    if counterfactual == 'hm':
        needed.extend([cf_col, ch1_col])
    elif counterfactual == 'alt':
        needed.extend([cf_col, ch1_col])

    # Check all columns exist
    for c in needed:
        if c not in df.columns:
            return None

    temp = df.copy()
    temp['VA'] = temp[as_col]

    # Instrument: counterfactual VAM for lottery losers, first-choice VAM for winners
    temp['lott_VA'] = np.where(
        temp['lottery'] == 0,
        temp[cf_col],
        temp[ch1_col]
    )

    return temp


# ============================================================
# BASELINE: Table 1 — Model 2, AR, all pre-lottery years, home school CF
# ============================================================

print("\n" + "="*60)
print("BASELINE SPECIFICATION")
print("="*60)

baseline_data = prepare_iv_data(analysis, 'mod2', 'ar', 'all', 'hm')

if baseline_data is not None:
    # Drop rows with missing outcome, VA, instrument, or controls
    needed_cols = ['testz2003', 'VA', 'lott_VA', 'lottery_FE_str'] + IV_CONTROLS
    baseline_sample = baseline_data.dropna(subset=needed_cols).copy()
    print(f"Baseline sample: {len(baseline_sample)} students")

    base_run_id, base_coef, base_se, base_pval, base_nobs = run_iv_spec(
        "baseline", "designs/iv_fe.md#baseline", "G1",
        "testz2003", "VA", "lott_VA", IV_CONTROLS,
        "lottery_FE_str", "lottery_FE", baseline_sample,
        {"CRV1": "lottery_FE_str"},
        f"onmargin==1, N={len(baseline_sample)}", "lagged scores (imputed, with polynomials)",
        "home school counterfactual",
        notes="Table 1: Model 2, AR, all pre-lottery years")

    print(f"  Baseline: coef={base_coef:.4f}, se={base_se:.4f}, p={base_pval:.4f}, N={base_nobs}")
else:
    print("WARNING: Could not construct baseline VAM!")
    base_run_id = None
    base_coef = base_se = base_pval = base_nobs = np.nan


# ============================================================
# CORE UNIVERSE: VAM Grid (model x estimation x window x counterfactual)
# ============================================================

print("\n" + "="*60)
print("CORE UNIVERSE: VAM Grid Specifications")
print("="*60)

VAM_MODELS = ['mod1', 'mod2']
VAM_EST_METHODS = ['ar', 'mix', 'FE']
VAM_WINDOWS = ['02', '2yr', 'all']
COUNTERFACTUALS = ['hm', 'alt']

model_labels = {'mod1': 'Model1_levels', 'mod2': 'Model2_gains'}
est_labels = {'ar': 'avg_residual', 'mix': 'mixed_effects', 'FE': 'school_FE'}
window_labels = {'02': 'single_year_02', '2yr': '2_years_01_02', 'all': 'all_pre_lottery'}
cf_labels = {'hm': 'home_school', 'alt': 'weighted_alternative'}

vam_grid_count = 0
for model in VAM_MODELS:
    for est in VAM_EST_METHODS:
        for window in VAM_WINDOWS:
            for cf in COUNTERFACTUALS:
                spec_id = f"rc/data/vam/{model}_{est}_{window}_{cf}"
                spec_label = f"{model_labels[model]}, {est_labels[est]}, {window_labels[window]}, {cf_labels[cf]}"

                iv_data = prepare_iv_data(analysis, model, est, window, cf)
                if iv_data is None:
                    continue

                needed_cols = ['testz2003', 'VA', 'lott_VA', 'lottery_FE_str'] + IV_CONTROLS
                sample = iv_data.dropna(subset=needed_cols).copy()

                if len(sample) < 30:
                    print(f"  Skipping {spec_id}: only {len(sample)} obs")
                    continue

                vam_grid_count += 1
                run_iv_spec(
                    spec_id, "designs/iv_fe.md#vam_grid", "G1",
                    "testz2003", "VA", "lott_VA", IV_CONTROLS,
                    "lottery_FE_str", "lottery_FE", sample,
                    {"CRV1": "lottery_FE_str"},
                    f"onmargin==1, N={len(sample)}", "lagged scores (imputed)",
                    cf_labels[cf],
                    axis_block_name="vam_variant",
                    axis_block={"spec_id": spec_id, "model": model,
                                "estimation": est, "window": window,
                                "counterfactual": cf, "label": spec_label})

print(f"Ran {vam_grid_count} VAM grid specifications")


# ============================================================
# ADDITIONAL BASELINE SPECS (Model 1 levels, Model 2 mix/FE)
# ============================================================

print("\n" + "="*60)
print("ADDITIONAL BASELINES")
print("="*60)

# Baseline: Model 1, levels, all, home school
for model, est, window in [('mod1', 'ar', 'all'), ('mod2', 'mix', 'all'), ('mod2', 'FE', 'all')]:
    spec_id = f"baseline__{model}_{est}"
    iv_data = prepare_iv_data(analysis, model, est, window, 'hm')
    if iv_data is not None:
        needed = ['testz2003', 'VA', 'lott_VA', 'lottery_FE_str'] + IV_CONTROLS
        sample = iv_data.dropna(subset=needed).copy()
        if len(sample) >= 30:
            run_iv_spec(
                spec_id, "designs/iv_fe.md#baseline", "G1",
                "testz2003", "VA", "lott_VA", IV_CONTROLS,
                "lottery_FE_str", "lottery_FE", sample,
                {"CRV1": "lottery_FE_str"},
                f"onmargin==1, N={len(sample)}", "lagged scores (imputed)",
                "home school counterfactual",
                notes=f"Additional baseline: {model_labels[model]}, {est_labels[est]}")


# ============================================================
# DESIGN: Diff-in-means (no controls, just FE + IV)
# ============================================================

print("\n" + "="*60)
print("DESIGN: Diff-in-means (no controls)")
print("="*60)

# Use baseline VAM (mod2, ar, all, hm) but drop all second-stage controls
iv_data = prepare_iv_data(analysis, 'mod2', 'ar', 'all', 'hm')
if iv_data is not None:
    needed = ['testz2003', 'VA', 'lott_VA', 'lottery_FE_str']
    sample = iv_data.dropna(subset=needed).copy()
    if len(sample) >= 30:
        run_iv_spec(
            "design/randomized_experiment/estimator/diff_in_means",
            "designs/iv_fe.md#design_variants", "G1",
            "testz2003", "VA", "lott_VA", [],
            "lottery_FE_str", "lottery_FE", sample,
            {"CRV1": "lottery_FE_str"},
            f"onmargin==1, N={len(sample)}", "no controls (diff-in-means style IV)",
            "home school counterfactual",
            axis_block_name="design",
            axis_block={"spec_id": "design/diff_in_means", "n_controls": 0})


# ============================================================
# DESIGN: With covariates (same as baseline, for completeness)
# ============================================================

iv_data = prepare_iv_data(analysis, 'mod2', 'ar', 'all', 'hm')
if iv_data is not None:
    needed = ['testz2003', 'VA', 'lott_VA', 'lottery_FE_str'] + IV_CONTROLS
    sample = iv_data.dropna(subset=needed).copy()
    if len(sample) >= 30:
        run_iv_spec(
            "design/randomized_experiment/estimator/with_covariates",
            "designs/iv_fe.md#design_variants", "G1",
            "testz2003", "VA", "lott_VA", IV_CONTROLS,
            "lottery_FE_str", "lottery_FE", sample,
            {"CRV1": "lottery_FE_str"},
            f"onmargin==1, N={len(sample)}", "lagged scores (standard controls)",
            "home school counterfactual",
            axis_block_name="design",
            axis_block={"spec_id": "design/with_covariates", "n_controls": len(IV_CONTROLS)})


# ============================================================
# RC: CONTROLS LOO — Drop lagged scores
# ============================================================

print("\n" + "="*60)
print("CONTROLS VARIANTS")
print("="*60)

# Drop all lagged scores (keep missing indicators only)
iv_data = prepare_iv_data(analysis, 'mod2', 'ar', 'all', 'hm')
if iv_data is not None:
    CONTROLS_NO_SCORES = ['math_2002_miss', 'read_2002_miss']
    needed = ['testz2003', 'VA', 'lott_VA', 'lottery_FE_str'] + CONTROLS_NO_SCORES
    sample = iv_data.dropna(subset=needed).copy()
    if len(sample) >= 30:
        run_iv_spec(
            "rc/controls/loo/drop_lagged_scores",
            "modules/robustness/controls.md#leave-one-out-controls-loo", "G1",
            "testz2003", "VA", "lott_VA", CONTROLS_NO_SCORES,
            "lottery_FE_str", "lottery_FE", sample,
            {"CRV1": "lottery_FE_str"},
            f"onmargin==1, N={len(sample)}", "missing indicators only (drop lagged scores)",
            "home school counterfactual",
            axis_block_name="controls",
            axis_block={"spec_id": "rc/controls/loo/drop_lagged_scores", "family": "loo",
                        "dropped": ["math_2002_imp", "read_2002_imp", "polynomials"]})

# Drop polynomials (keep linear scores + missing)
CONTROLS_LINEAR = ['math_2002_imp', 'read_2002_imp', 'math_2002_miss', 'read_2002_miss']
iv_data = prepare_iv_data(analysis, 'mod2', 'ar', 'all', 'hm')
if iv_data is not None:
    needed = ['testz2003', 'VA', 'lott_VA', 'lottery_FE_str'] + CONTROLS_LINEAR
    sample = iv_data.dropna(subset=needed).copy()
    if len(sample) >= 30:
        run_iv_spec(
            "rc/controls/loo/drop_polynomials",
            "modules/robustness/controls.md#leave-one-out-controls-loo", "G1",
            "testz2003", "VA", "lott_VA", CONTROLS_LINEAR,
            "lottery_FE_str", "lottery_FE", sample,
            {"CRV1": "lottery_FE_str"},
            f"onmargin==1, N={len(sample)}", "linear lagged scores + missing indicators (no polynomials)",
            "home school counterfactual",
            axis_block_name="controls",
            axis_block={"spec_id": "rc/controls/loo/drop_polynomials", "family": "loo",
                        "dropped": ["polynomials"]})

# No controls at all
iv_data = prepare_iv_data(analysis, 'mod2', 'ar', 'all', 'hm')
if iv_data is not None:
    needed = ['testz2003', 'VA', 'lott_VA', 'lottery_FE_str']
    sample = iv_data.dropna(subset=needed).copy()
    if len(sample) >= 30:
        run_iv_spec(
            "rc/controls/sets/no_controls",
            "modules/robustness/controls.md#standard-control-sets", "G1",
            "testz2003", "VA", "lott_VA", [],
            "lottery_FE_str", "lottery_FE", sample,
            {"CRV1": "lottery_FE_str"},
            f"onmargin==1, N={len(sample)}", "no controls (IV + FE only)",
            "home school counterfactual",
            axis_block_name="controls",
            axis_block={"spec_id": "rc/controls/sets/no_controls", "family": "sets",
                        "n_controls": 0})


# ============================================================
# RC: SAMPLE — Grade subsets
# ============================================================

print("\n" + "="*60)
print("SAMPLE VARIANTS: Grade subsets")
print("="*60)

# Grades 4-5 (elementary)
iv_data = prepare_iv_data(analysis, 'mod2', 'ar', 'all', 'hm')
if iv_data is not None:
    sample_45 = iv_data[iv_data['future_grd'].isin([4, 5])].copy()
    needed = ['testz2003', 'VA', 'lott_VA', 'lottery_FE_str'] + IV_CONTROLS
    sample_45 = sample_45.dropna(subset=needed).copy()
    if len(sample_45) >= 20:
        run_iv_spec(
            "rc/sample/grade/grade_4_5",
            "modules/robustness/sample.md#subgroup-analysis", "G1",
            "testz2003", "VA", "lott_VA", IV_CONTROLS,
            "lottery_FE_str", "lottery_FE", sample_45,
            {"CRV1": "lottery_FE_str"},
            f"grades 4-5 (elementary), N={len(sample_45)}", "lagged scores (imputed)",
            "home school counterfactual",
            axis_block_name="sample",
            axis_block={"spec_id": "rc/sample/grade/grade_4_5", "axis": "grade",
                        "grades": [4, 5]})

# Grades 6-7-8 (middle school)
if iv_data is not None:
    sample_678 = iv_data[iv_data['future_grd'].isin([6, 7, 8])].copy()
    needed = ['testz2003', 'VA', 'lott_VA', 'lottery_FE_str'] + IV_CONTROLS
    sample_678 = sample_678.dropna(subset=needed).copy()
    if len(sample_678) >= 20:
        run_iv_spec(
            "rc/sample/grade/grade_6_7_8",
            "modules/robustness/sample.md#subgroup-analysis", "G1",
            "testz2003", "VA", "lott_VA", IV_CONTROLS,
            "lottery_FE_str", "lottery_FE", sample_678,
            {"CRV1": "lottery_FE_str"},
            f"grades 6-8 (middle school), N={len(sample_678)}", "lagged scores (imputed)",
            "home school counterfactual",
            axis_block_name="sample",
            axis_block={"spec_id": "rc/sample/grade/grade_6_7_8", "axis": "grade",
                        "grades": [6, 7, 8]})


# ============================================================
# RC: FORM — Math-only and Reading-only outcomes
# ============================================================

print("\n" + "="*60)
print("OUTCOME VARIANTS: Math-only, Reading-only")
print("="*60)

for outcome_label, outcome_var in [('math_only', 'mathz2003'), ('read_only', 'readz2003')]:
    iv_data = prepare_iv_data(analysis, 'mod2', 'ar', 'all', 'hm')
    if iv_data is not None:
        needed = [outcome_var, 'VA', 'lott_VA', 'lottery_FE_str'] + IV_CONTROLS
        sample = iv_data.dropna(subset=needed).copy()
        if len(sample) >= 30:
            run_iv_spec(
                f"rc/form/outcome/{outcome_label}",
                "modules/robustness/functional_form.md#outcome-transformations", "G1",
                outcome_var, "VA", "lott_VA", IV_CONTROLS,
                "lottery_FE_str", "lottery_FE", sample,
                {"CRV1": "lottery_FE_str"},
                f"onmargin==1, N={len(sample)}", "lagged scores (imputed)",
                "home school counterfactual",
                axis_block_name="functional_form",
                axis_block={"spec_id": f"rc/form/outcome/{outcome_label}",
                            "outcome": outcome_var})


# ============================================================
# RC: Additional VAM-specific variants from rc_spec_ids
# ============================================================

print("\n" + "="*60)
print("RC: Counterfactual variants (home_school and weighted_alt)")
print("="*60)

# rc/data/counterfactual/home_school — already covered by baseline (hm)
# rc/data/counterfactual/weighted_alt — use alt counterfactual with baseline model
iv_data = prepare_iv_data(analysis, 'mod2', 'ar', 'all', 'alt')
if iv_data is not None:
    needed = ['testz2003', 'VA', 'lott_VA', 'lottery_FE_str'] + IV_CONTROLS
    sample = iv_data.dropna(subset=needed).copy()
    if len(sample) >= 30:
        run_iv_spec(
            "rc/data/counterfactual/weighted_alt",
            "designs/iv_fe.md#counterfactual", "G1",
            "testz2003", "VA", "lott_VA", IV_CONTROLS,
            "lottery_FE_str", "lottery_FE", sample,
            {"CRV1": "lottery_FE_str"},
            f"onmargin==1, N={len(sample)}", "lagged scores (imputed)",
            "weighted alternative counterfactual",
            axis_block_name="counterfactual",
            axis_block={"spec_id": "rc/data/counterfactual/weighted_alt",
                        "counterfactual": "weighted_alternative"})

iv_data = prepare_iv_data(analysis, 'mod2', 'ar', 'all', 'hm')
if iv_data is not None:
    needed = ['testz2003', 'VA', 'lott_VA', 'lottery_FE_str'] + IV_CONTROLS
    sample = iv_data.dropna(subset=needed).copy()
    if len(sample) >= 30:
        run_iv_spec(
            "rc/data/counterfactual/home_school",
            "designs/iv_fe.md#counterfactual", "G1",
            "testz2003", "VA", "lott_VA", IV_CONTROLS,
            "lottery_FE_str", "lottery_FE", sample,
            {"CRV1": "lottery_FE_str"},
            f"onmargin==1, N={len(sample)}", "lagged scores (imputed)",
            "home school counterfactual",
            axis_block_name="counterfactual",
            axis_block={"spec_id": "rc/data/counterfactual/home_school",
                        "counterfactual": "home_school"})


# ============================================================
# INFERENCE VARIANTS (on baseline specification)
# ============================================================

print("\n" + "="*60)
print("INFERENCE VARIANTS")
print("="*60)

baseline_run_id_ref = f"{PAPER_ID}_run_001"
infer_counter = 0


def run_inference_variant(base_run_id, spec_id, spec_tree_path, baseline_group_id,
                          outcome_var, treatment_var, instrument_var, controls,
                          fe_str, data, vcov, vcov_desc):
    """Re-run baseline IV spec with different variance-covariance estimator."""
    global infer_counter
    infer_counter += 1
    infer_run_id = f"{PAPER_ID}_infer_{infer_counter:03d}"

    try:
        controls_str = " + ".join(controls) if controls else "1"
        if fe_str:
            formula = f"{outcome_var} ~ {controls_str} | {fe_str} | {treatment_var} ~ {instrument_var}"
        else:
            formula = f"{outcome_var} ~ {controls_str} | {treatment_var} ~ {instrument_var}"

        m = pf.feols(formula, data=data, vcov=vcov)

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
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(make_success_payload(
                coefficients={treatment_var: coef_val},
                inference={"spec_id": spec_id, "method": vcov_desc},
                software=SW_BLOCK,
                surface_hash=SURFACE_HASH,
                design={"iv_fe": design_audit},
            )),
            "cluster_var": vcov_desc,
            "run_success": 1,
            "run_error": ""
        })

    except Exception as e:
        err_msg = str(e)[:240]
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
            "coefficient_vector_json": json.dumps(make_failure_payload(
                error=err_msg,
                error_details=error_details_from_exception(e, stage="inference"),
                software=SW_BLOCK,
                surface_hash=SURFACE_HASH
            )),
            "cluster_var": vcov_desc,
            "run_success": 0,
            "run_error": err_msg
        })


# Analytic clustered SEs at lottery_FE (no bootstrap)
if baseline_data is not None:
    needed_cols = ['testz2003', 'VA', 'lott_VA', 'lottery_FE_str'] + IV_CONTROLS
    infer_sample = baseline_data.dropna(subset=needed_cols).copy()

    run_inference_variant(
        baseline_run_id_ref, "infer/se/cluster/lottery_FE",
        "modules/inference/standard_errors.md#clustering", "G1",
        "testz2003", "VA", "lott_VA", IV_CONTROLS,
        "lottery_FE_str", infer_sample,
        {"CRV1": "lottery_FE_str"}, "cluster(lottery_FE)")

    # HC1 robust (no clustering)
    run_inference_variant(
        baseline_run_id_ref, "infer/se/hc/hc1",
        "modules/inference/standard_errors.md#heteroskedasticity-robust", "G1",
        "testz2003", "VA", "lott_VA", IV_CONTROLS,
        "lottery_FE_str", infer_sample,
        "hetero", "HC1 (robust, no clustering)")

    # iid (homoskedastic)
    run_inference_variant(
        baseline_run_id_ref, "infer/se/iid",
        "modules/inference/standard_errors.md#homoskedastic", "G1",
        "testz2003", "VA", "lott_VA", IV_CONTROLS,
        "lottery_FE_str", infer_sample,
        "iid", "iid (homoskedastic)")


# ============================================================
# WRITE OUTPUTS
# ============================================================

print("\n" + "="*60)
print("WRITING OUTPUTS")
print("="*60)

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
        print(f"\nBaseline coef on VA: {base_row['coefficient'].values[0]:.6f}")
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

if len(failed) > 0:
    print(f"\n=== FAILED SPECIFICATIONS ===")
    for _, row in failed.iterrows():
        print(f"  {row['spec_id']}: {row['run_error'][:100]}")


# ============================================================
# SPECIFICATION_SEARCH.md
# ============================================================

print("\nWriting SPECIFICATION_SEARCH.md...")

md_lines = []
md_lines.append("# Specification Search Report: 112805-V1")
md_lines.append("")
md_lines.append("**Paper:** Deming (2014), \"Using School Choice Lotteries to Test Measures of School Effectiveness\", AER P&P 104(5)")
md_lines.append("")
md_lines.append("## Baseline Specification")
md_lines.append("")
md_lines.append("- **Design:** IV with lottery FE (randomized experiment)")
md_lines.append("- **Outcome:** testz2003 (average of math and reading z-scores)")
md_lines.append("- **Treatment (endogenous):** VA (school value-added measure)")
md_lines.append("- **Instrument:** lott_VA (lottery-determined school VAM)")
md_lines.append("- **VAM Model:** Model 2 (gains/lagged scores), average residual, all pre-lottery years")
md_lines.append("- **Controls:** 8 lagged score controls (imputed, with polynomials)")
md_lines.append("- **Fixed effects:** lottery_FE")
md_lines.append("- **Clustering:** lottery_FE")
md_lines.append("")

if len(successful) > 0:
    base_row = spec_df[spec_df['spec_id'] == 'baseline']
    if len(base_row) > 0:
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
    "VAM Grid": successful[successful['spec_id'].str.startswith('rc/data/vam/')],
    "Design Variants": successful[successful['spec_id'].str.startswith('design/')],
    "Controls Variants": successful[successful['spec_id'].str.startswith('rc/controls/')],
    "Sample (Grade)": successful[successful['spec_id'].str.startswith('rc/sample/')],
    "Outcome Variants": successful[successful['spec_id'].str.startswith('rc/form/')],
    "Counterfactual": successful[successful['spec_id'].str.startswith('rc/data/counterfactual/')],
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
    median_coef = successful['coefficient'].median()
    sign_word = "positive" if median_coef > 0 else "negative"

    # For this paper, the claim is that VAM coefficient = 1 (unbiased).
    # Positive coefficient close to 1 supports the claim.
    close_to_1 = ((successful['coefficient'] > 0.5) & (successful['coefficient'] < 2.0)).sum()
    pct_close = close_to_1 / len(successful) * 100

    md_lines.append(f"- **Coefficient direction:** Median coefficient is {sign_word} ({median_coef:.4f})")
    md_lines.append(f"- **Significance stability:** {n_sig_total}/{len(successful)} ({pct_sig:.1f}%) specifications significant at 5%")
    md_lines.append(f"- **VAM validity (coef near 1):** {close_to_1}/{len(successful)} ({pct_close:.1f}%) specifications have coefficient in [0.5, 2.0]")
    md_lines.append(f"- **Note:** The paper's claim is that the VAM coefficient equals 1 (unbiased), not just significance")

    if pct_close >= 80:
        strength = "STRONG"
    elif pct_close >= 50:
        strength = "MODERATE"
    elif pct_close >= 30:
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
