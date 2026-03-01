"""
Specification Search Script for Ogaki & Zhang (2001)
"Decreasing Relative Risk Aversion and Tests of Risk Sharing"
Econometrica, 69(2), 515-526.

Paper ID: 112498-V1

Surface-driven execution:
  - G1: ch_r_totexp ~ ch_r_nonlabinc + controls (OLS / IV)
  - Panel data from ICRISAT village-level studies (VLS), India, 1975-1984
  - Standard test of efficient risk sharing: under full insurance,
    idiosyncratic income changes should not predict consumption changes
  - 50+ specifications across villages, controls, outcomes, samples, FE, IV

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

PAPER_ID = "112498-V1"
DATA_DIR = "data/downloads/extracted/112498-V1"
STATA_DIR = f"{DATA_DIR}/DataStataAndFortranFilesForPublication/StataFilesForPublication"
OUTPUT_DIR = DATA_DIR
DATA_PATH = f"{STATA_DIR}/data.cp.month.inst.dta"
RAIN_PATH = f"{STATA_DIR}/ICRISAT/rain/RAIN.RAW"

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

def prepare_village_data(df_raw, villcode, rain_df, min_obs=80):
    """
    Prepare analysis dataset for a single village following StandardTest.do.

    Steps:
    1. Generate leisure and w_leisure variables
    2. Generate transfer variables
    3. Keep only specified village
    4. Create real per-capita variables using price indices
    5. Generate change variables
    6. Drop households with missing observations
    7. Merge rainfall data
    """
    df = df_raw.copy()
    wagevar = 'thh_yrw'

    # Generate leisure variables
    df['leisure'] = (26*(24-8) - df['thh_yrhr']/df['wt_nhh'])
    df['w_leisure'] = (26*(24-8) - df['thh_yrhr']/df['wt_nhh']) * df[wagevar] / (df['wt_nhh']*8)
    df.loc[(df['w_leisure'] < 0) & (df['w_leisure'].notna()), 'w_leisure'] = 0

    # Generate transfer variables
    bought_cols = ['mv_bought640pres','mv_bought64pres','mv_bought6401','mv_bought6402',
        'mv_bought6403','mv_bought6405','mv_bought6406','mv_bought6407','mv_bought6408',
        'mv_bought640KC','mv_bought640V','mv_bought640W','mv_bought640X','mv_bought640Y',
        'mv_bought640Z','mv_bought641','mv_bought642','mv_bought643','mv_bought645',
        'mv_bought646','mv_bought647','mv_bought648','mv_bought64A','mv_bought64KC','mv_bought64V']

    received_cols = ['mv_received640rres','mv_received64rres','mv_received6401','mv_received6403',
        'mv_received6404','mv_received6405','mv_received6406','mv_received6407','mv_received6408',
        'mv_received640A','mv_received640KC','mv_received640V','mv_received640W','mv_received640Z',
        'mv_received641','mv_received642','mv_received643','mv_received645','mv_received646',
        'mv_received647','mv_received648','mv_received649','mv_received64A','mv_received64KC',
        'mv_received64V']

    df['transfout'] = df[bought_cols].sum(axis=1, min_count=1)
    df['transfin'] = df[received_cols].sum(axis=1, min_count=1)

    # Keep only specified village
    df = df[df['villcode'] == villcode].copy()

    # Replace missing grain/food/ndur with 0
    for v in ['grain', 'food', 'ndur']:
        df.loc[df[v].isna(), v] = 0

    # Replace missing profit/exlabva/exanva with 0
    for v in ['profit', 'exlabva', 'exanva']:
        df.loc[df[v].isna(), v] = 0

    # Generate price index base values (first observation in time order)
    df = df.sort_values(['villcode','yearcode','n_month','hh_num']).reset_index(drop=True)
    pi_totexp75 = df['pi_totexp'].iloc[0]
    pi_grain75 = df['pi_grain'].iloc[0]
    pi_food75 = df['pi_food'].iloc[0]
    pi_ndur75 = df['pi_ndur'].iloc[0]

    # Handle missing pi_totexp with year-month mean
    mpi = df.groupby('yearcode')['pi_totexp'].transform('mean')
    df.loc[df['pi_totexp'].isna(), 'pi_totexp'] = mpi[df['pi_totexp'].isna()]

    # Generate real variables
    df['r_wage'] = df[wagevar] * pi_totexp75 / df['pi_totexp']

    df['r_totexp'] = (df['totexp'] * pi_totexp75 / (df['pi_totexp'] * df['wt_nhh']) +
                      df['w_leisure'] * pi_totexp75 / df['pi_totexp'])

    df['r_totexpnoleis'] = df['totexp'] * pi_totexp75 / (df['pi_totexp'] * df['wt_nhh'])

    df['r_grain'] = df['grain'] * pi_grain75 / (df['pi_grain'] * df['wt_nhh'])
    df.loc[df['pi_grain'].isna(), 'r_grain'] = np.nan

    df['r_food'] = df['food'] * pi_food75 / (df['pi_food'] * df['wt_nhh'])

    df['r_ndur'] = df['ndur'] * pi_ndur75 / (df['pi_ndur'] * df['wt_nhh'])

    # Generate pi_grfd75 if possible
    if 'pi_grfd' in df.columns and df['pi_grfd'].notna().any():
        pi_grfd75 = df.sort_values(['yearcode','n_month'])['pi_grfd'].dropna().iloc[0]
        df['r_grfd'] = df['grfd'] * pi_grfd75 / (df['pi_grfd'] * df['wt_nhh'])

    # Real non-labor income, profit, transfers
    df['r_nonlabinc_raw'] = df['nonlabinc'] * pi_totexp75 / (df['pi_totexp'] * df['wt_nhh'])
    df['r_profit'] = df['profit'] * pi_totexp75 / (df['pi_totexp'] * df['wt_nhh'])
    df['r_transfin'] = df['transfin'] * pi_totexp75 / (df['pi_totexp'] * df['wt_nhh'])
    df['r_transfout'] = df['transfout'] * pi_totexp75 / (df['pi_totexp'] * df['wt_nhh'])

    # Combined non-labor income (from StandardTest.do line 367)
    df['r_totnonlabinc'] = df['r_nonlabinc_raw'] + df['r_profit']
    df['r_nonlabinc'] = df['r_totnonlabinc']

    # Real total income per capita
    df['r_tot_dist'] = -1 * ((df['credit_used'] + df['exlabva'] + df['exanva'] +
        df['durables_used'].fillna(0) + df['taxes'].fillna(0) + df['grain_notmilled'].fillna(0)) *
        pi_totexp75 / (df['pi_totexp'] * df['wt_nhh']) +
        (df['transfin'] - df['transfout'] + df['mv_received640Q'].fillna(0) +
         df.get('mv_received64Q', pd.Series(0, index=df.index)).fillna(0) -
         df['mv_bought640Q'].fillna(0) - df['mv_bought64Q'].fillna(0)) *
        pi_totexp75 / (df['pi_totexp'] * df['wt_nhh']))
    df['r_tot_inc_pc'] = df['r_totexp'] + df['r_tot_dist']

    # Real total daily wage
    df['r_tot_daily_wage'] = df['tot_daily_wage'] * pi_totexp75 / df['pi_totexp'] if 'tot_daily_wage' in df.columns else np.nan
    df['r_tot_daily_wage_pc'] = df['r_tot_daily_wage'] / df['wt_nhh'] if 'r_tot_daily_wage' in df.columns else np.nan

    # Trim top/bottom 1% of r_totexp
    p1 = df['r_totexp'].quantile(0.01)
    p99 = df['r_totexp'].quantile(0.99)
    df.loc[df['r_totexp'] < p1, 'r_totexp'] = np.nan
    df.loc[df['r_totexp'] > p99, 'r_totexp'] = np.nan

    # Village average consumption
    df['r_totexp_vill_av'] = df.groupby(['yearcode','n_month'])['r_totexp'].transform('mean')

    # Caste average consumption
    df['r_totexp_caste_av'] = df.groupby(['castrk_b','yearcode','n_month'])['r_totexp'].transform('mean')

    # Drop specific households with missing data
    if villcode == 'C':
        for hh in [47, 32, 57]:
            df = df[~(df['hh_num'] == hh)].copy()
    elif villcode == 'E':
        for hh in [7, 9]:
            df = df[~(df['hh_num'] == hh)].copy()

    # Drop early 1975 observations
    df = df[~((df['yearcode'] == 75) & (df['n_month'] < 7))].copy()

    # Sort by household and time
    df = df.sort_values(['villcode','hh_num','yearcode','n_month']).reset_index(drop=True)

    # Generate lagged variables
    df['l_r_totexp'] = df.groupby(['villcode','hh_num'])['r_totexp'].shift(1)
    df['ch_r_totexp'] = df['r_totexp'] - df['l_r_totexp']

    df['l_r_totexp_vill'] = df.groupby(['villcode','hh_num'])['r_totexp_vill_av'].shift(1)
    df['ch_r_totexp_vill'] = df['r_totexp_vill_av'] - df['l_r_totexp_vill']

    df['l_r_totexp_caste'] = df.groupby(['villcode','hh_num'])['r_totexp_caste_av'].shift(1)
    df['ch_r_totexp_caste'] = df['r_totexp_caste_av'] - df['l_r_totexp_caste']

    df['l_r_wage'] = df.groupby(['villcode','hh_num'])['r_wage'].shift(1)

    df['l_r_nonlabinc'] = df.groupby(['villcode','hh_num'])['r_nonlabinc'].shift(1)
    df['ch_r_nonlabinc'] = df['r_nonlabinc'] - df['l_r_nonlabinc']

    # Change in food, grain, ndur, totexpnoleis (for alternative outcomes)
    for var in ['r_food', 'r_grain', 'r_ndur', 'r_totexpnoleis', 'r_tot_inc_pc']:
        l_var = f'l_{var}'
        ch_var = f'ch_{var}'
        df[l_var] = df.groupby(['villcode','hh_num'])[var].shift(1)
        df[ch_var] = df[var] - df[l_var]

    # Change in wage (alternative treatment)
    df['l_r_wage_for_ch'] = df.groupby(['villcode','hh_num'])['r_wage'].shift(1)
    df['ch_r_wage'] = df['r_wage'] - df['l_r_wage_for_ch']

    # Count valid observations per household
    df['valid'] = (~df['r_totexp'].isna()) & (~df['r_wage'].isna()) & \
                  (df['r_wage'] != 0) & (~df['l_r_totexp'].isna())
    nobs_per_hh = df.groupby(['villcode','hh_num'])['valid'].sum()

    # Keep households with >= min_obs observations
    valid_hhs = set(nobs_per_hh[nobs_per_hh >= min_obs].index)
    df['hh_key'] = list(zip(df['villcode'], df['hh_num']))
    df = df[df['hh_key'].isin(valid_hhs)].copy()

    # Merge rainfall data
    rain_map = {1: 'A', 3: 'C', 5: 'E'}
    vill_rain_code = {'A': 1, 'C': 3, 'E': 5}[villcode]
    rain_vill = rain_df[rain_df['village'] == vill_rain_code][['yearcode','n_month','rain']].copy()
    df = df.merge(rain_vill, on=['yearcode','n_month'], how='left')
    df['rain'] = df['rain'].fillna(0)

    # Create household and year-month identifiers for FE
    df['hh_str'] = df['hh_num'].astype(int).astype(str)
    df['year_str'] = df['yearcode'].astype(int).astype(str)
    df['yearmonth'] = (df['yearcode'].astype(int).astype(str) + '_' +
                       df['n_month'].astype(int).astype(str))

    # Year dummies
    for y in range(75, 85):
        df[f'd{y}'] = (df['yearcode'] == y).astype(float)

    return df


print("Loading raw data...")
df_raw = pd.read_stata(DATA_PATH)

# Convert float32 to float64 for precision
for col in df_raw.columns:
    if df_raw[col].dtype == np.float32:
        df_raw[col] = df_raw[col].astype(np.float64)

print(f"Loaded data: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

# Load rainfall data
rain_cols = ['village', 'day', 'month', 'year', 'dayssinceJan1', 'mm_rain']
rain_raw = pd.read_csv(RAIN_PATH, sep=r'\s+', header=None, names=rain_cols)
rain_agg = rain_raw.groupby(['village','year','month'])['mm_rain'].sum().reset_index()
rain_agg.columns = ['village','yearcode','n_month','rain']

# Prepare village E (main analysis in StandardTest.do)
df_E = prepare_village_data(df_raw, 'E', rain_agg, min_obs=80)
print(f"Village E: {len(df_E)} rows, {df_E['hh_num'].nunique()} households")

# Also prepare A and C for village-level robustness
df_A = prepare_village_data(df_raw, 'A', rain_agg, min_obs=80)
print(f"Village A: {len(df_A)} rows, {df_A['hh_num'].nunique()} households")

df_C = prepare_village_data(df_raw, 'C', rain_agg, min_obs=80)
print(f"Village C: {len(df_C)} rows, {df_C['hh_num'].nunique()} households")

# Pool all three villages
df_all = pd.concat([df_A, df_C, df_E], ignore_index=True)
df_all['vill_str'] = df_all['villcode']
print(f"All villages: {len(df_all)} rows, {df_all['hh_num'].nunique()} households")

# Define controls
BASELINE_CONTROLS = ['ch_r_totexp_vill', 'l_r_wage', 'wt_nhh', 'intfreq']

# Additional controls that can be added
EXTRA_CONTROLS = ['hhsize', 'n_infant', 'm_adultage', 'castrk_b', 'rain', 'l_r_totexp']

# Baseline estimation sample: drop rows with missing baseline vars
baseline_vars = ['ch_r_totexp', 'ch_r_nonlabinc'] + BASELINE_CONTROLS
df_base = df_E.dropna(subset=baseline_vars).copy()
print(f"Baseline sample (E, all vars non-missing): {len(df_base)} rows")


# ============================================================
# Accumulators
# ============================================================

results = []
inference_results = []
spec_run_counter = 0


# ============================================================
# Helper: run_spec (OLS with optional FE via pyfixest)
# ============================================================

def run_spec(spec_id, spec_tree_path, baseline_group_id,
             outcome_var, treatment_var, controls, fe_formula_str,
             fe_desc, data, vcov, sample_desc, controls_desc,
             cluster_var="",
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
                       "method": "iid" if vcov == "iid" else str(vcov)},
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
            "fixed_effects": fe_desc,
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
            "run_success": 0,
            "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


# ============================================================
# Helper: run_iv (IV regression via pyfixest)
# ============================================================

def run_iv(spec_id, spec_tree_path, baseline_group_id,
           outcome_var, treatment_var, controls, instruments,
           fe_formula_str, fe_desc, data, vcov, sample_desc, controls_desc,
           cluster_var="",
           axis_block_name=None, axis_block=None, notes=""):
    """Run a single IV specification and record results."""
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        exog_str = " + ".join(controls) if controls else "1"
        inst_str = " + ".join(instruments)

        if fe_formula_str:
            formula = f"{outcome_var} ~ {exog_str} | {fe_formula_str} | {treatment_var} ~ {inst_str}"
        else:
            formula = f"{outcome_var} ~ {exog_str} | {treatment_var} ~ {inst_str}"

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
                       "method": "iv", "instruments": instruments},
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
# BASELINE: StandardTest.do line 392 — OLS, village E
# reg ch_r_totexp ch_r_totexp_vill l_r_wage wt_nhh intfreq ch_r_nonlabinc
# ============================================================

print("\n=== Running baseline specification ===")
base_run_id, base_coef, base_se, base_pval, base_nobs = run_spec(
    "baseline", "designs/panel_fixed_effects.md#baseline", "G1",
    "ch_r_totexp", "ch_r_nonlabinc", BASELINE_CONTROLS,
    "", "none", df_base,
    "iid",
    f"Village E, N={len(df_base)}", "ch_r_totexp_vill + l_r_wage + wt_nhh + intfreq")

print(f"  Baseline: coef={base_coef:.6f}, se={base_se:.6f}, p={base_pval:.6f}, N={base_nobs}")


# ============================================================
# BASELINE VARIANT: IV (StandardTest.do line 396)
# ivreg ch_r_totexp ch_r_totexp_vill l_r_wage wt_nhh intfreq
#   (ch_r_nonlabinc = wt_nhh intfreq hhsize n_infant rain l_r_totexp l_r_wage)
# ============================================================

print("\nRunning IV baseline...")

# Prepare IV data (needs all instruments to be non-missing)
iv_instruments = ['wt_nhh', 'intfreq', 'hhsize', 'n_infant', 'rain', 'l_r_totexp', 'l_r_wage']
iv_controls = ['ch_r_totexp_vill', 'l_r_wage', 'wt_nhh', 'intfreq']
iv_vars = ['ch_r_totexp', 'ch_r_nonlabinc'] + iv_controls + ['hhsize', 'n_infant', 'rain', 'l_r_totexp']
df_iv = df_E.dropna(subset=iv_vars).copy()

# For pyfixest IV, instruments must be excluded from exogenous regressors
# The exogenous vars are: ch_r_totexp_vill, l_r_wage, wt_nhh, intfreq
# The endogenous var is: ch_r_nonlabinc
# Instruments include: wt_nhh, intfreq, hhsize, n_infant, rain, l_r_totexp, l_r_wage
# Excluded instruments (not in exogenous): hhsize, n_infant, rain, l_r_totexp
# Note: wt_nhh, intfreq, l_r_wage appear in both exogenous and instruments

run_iv(
    "baseline__iv", "designs/panel_fixed_effects.md#baseline", "G1",
    "ch_r_totexp", "ch_r_nonlabinc", iv_controls,
    ['hhsize', 'n_infant', 'rain', 'l_r_totexp'],
    "", "none", df_iv,
    "iid",
    f"Village E (IV), N={len(df_iv)}", "IV: ch_r_nonlabinc instrumented",
    axis_block_name="estimation",
    axis_block={"spec_id": "baseline__iv", "estimator": "iv",
                "instruments": iv_instruments})


# ============================================================
# BASELINE VARIANTS: Other villages (OLS)
# ============================================================

print("\nRunning other village baselines...")

# Village A
df_A_base = df_A.dropna(subset=baseline_vars).copy()
if len(df_A_base) > 10:
    run_spec(
        "baseline__ols_villA", "designs/panel_fixed_effects.md#baseline", "G1",
        "ch_r_totexp", "ch_r_nonlabinc", BASELINE_CONTROLS,
        "", "none", df_A_base,
        "iid",
        f"Village A (Aurepalle), N={len(df_A_base)}", "baseline controls",
        axis_block_name="sample",
        axis_block={"spec_id": "baseline__ols_villA", "village": "A"})

# Village C
df_C_base = df_C.dropna(subset=baseline_vars).copy()
if len(df_C_base) > 10:
    run_spec(
        "baseline__ols_villC", "designs/panel_fixed_effects.md#baseline", "G1",
        "ch_r_totexp", "ch_r_nonlabinc", BASELINE_CONTROLS,
        "", "none", df_C_base,
        "iid",
        f"Village C (Shirapur), N={len(df_C_base)}", "baseline controls",
        axis_block_name="sample",
        axis_block={"spec_id": "baseline__ols_villC", "village": "C"})


# ============================================================
# BASELINE VARIANT: By caste (StandardTest.do line 394)
# ============================================================

print("\nRunning by-caste regressions...")

caste_groups = df_base['castrk_b'].value_counts()
for caste_val, n_caste in caste_groups.items():
    if n_caste >= 50:  # need enough obs for regression
        df_caste = df_base[df_base['castrk_b'] == caste_val].copy()
        caste_label = f"caste_{int(caste_val)}"
        run_spec(
            f"baseline__by_caste/{caste_label}",
            "designs/panel_fixed_effects.md#baseline", "G1",
            "ch_r_totexp", "ch_r_nonlabinc", BASELINE_CONTROLS,
            "", "none", df_caste,
            "iid",
            f"Village E, caste={int(caste_val)}, N={len(df_caste)}", "baseline controls",
            axis_block_name="sample",
            axis_block={"spec_id": f"baseline__by_caste/{caste_label}",
                        "family": "subgroup", "caste": int(caste_val)})


# ============================================================
# RC: CONTROLS LOO - Drop one control at a time
# ============================================================

print("\nRunning controls LOO variants...")

LOO_MAP = {
    "rc/controls/loo/drop_ch_r_totexp_vill": ["ch_r_totexp_vill"],
    "rc/controls/loo/drop_l_r_wage": ["l_r_wage"],
    "rc/controls/loo/drop_wt_nhh": ["wt_nhh"],
    "rc/controls/loo/drop_intfreq": ["intfreq"],
}

for spec_id, drop_vars in LOO_MAP.items():
    ctrl = [c for c in BASELINE_CONTROLS if c not in drop_vars]
    run_spec(
        spec_id, "modules/robustness/controls.md#leave-one-out-controls-loo", "G1",
        "ch_r_totexp", "ch_r_nonlabinc", ctrl,
        "", "none", df_base,
        "iid",
        "Village E", f"baseline minus {', '.join(drop_vars)}",
        axis_block_name="controls",
        axis_block={"spec_id": spec_id, "family": "loo",
                    "dropped": drop_vars, "n_controls": len(ctrl)})


# ============================================================
# RC: CONTROL SETS
# ============================================================

print("\nRunning control set variants...")

# No controls (bivariate)
run_spec(
    "rc/controls/sets/none",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "ch_r_totexp", "ch_r_nonlabinc", [],
    "", "none", df_base,
    "iid",
    "Village E", "none (bivariate)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/none", "family": "sets",
                "n_controls": 0, "set_name": "none"})

# Village average only
run_spec(
    "rc/controls/sets/vill_avg_only",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "ch_r_totexp", "ch_r_nonlabinc", ["ch_r_totexp_vill"],
    "", "none", df_base,
    "iid",
    "Village E", "village avg consumption change only",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/vill_avg_only", "family": "sets",
                "n_controls": 1, "set_name": "vill_avg_only"})

# Village avg + wage
run_spec(
    "rc/controls/sets/vill_avg_plus_wage",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "ch_r_totexp", "ch_r_nonlabinc", ["ch_r_totexp_vill", "l_r_wage"],
    "", "none", df_base,
    "iid",
    "Village E", "village avg + lagged wage",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/vill_avg_plus_wage", "family": "sets",
                "n_controls": 2, "set_name": "vill_avg_plus_wage"})


# ============================================================
# RC: CONTROL PROGRESSION (build-up)
# ============================================================

print("\nRunning control progression variants...")

progressions = [
    ("rc/controls/progression/bivariate", [], "bivariate"),
    ("rc/controls/progression/add_vill_avg", ["ch_r_totexp_vill"], "add village avg"),
    ("rc/controls/progression/add_wage", ["ch_r_totexp_vill", "l_r_wage"], "add wage"),
    ("rc/controls/progression/add_hhsize", ["ch_r_totexp_vill", "l_r_wage", "wt_nhh"],
     "add hh size weight"),
    ("rc/controls/progression/full", BASELINE_CONTROLS, "full baseline controls"),
]

for spec_id, ctrl, desc in progressions:
    run_spec(
        spec_id, "modules/robustness/controls.md#control-progression-build-up", "G1",
        "ch_r_totexp", "ch_r_nonlabinc", ctrl,
        "", "none", df_base,
        "iid",
        "Village E", desc,
        axis_block_name="controls",
        axis_block={"spec_id": spec_id, "family": "progression",
                    "n_controls": len(ctrl), "set_name": desc})


# ============================================================
# RC: ADD EXTRA CONTROLS
# ============================================================

print("\nRunning extra control variants...")

extra_additions = [
    ("rc/controls/add/hhsize", ["hhsize"]),
    ("rc/controls/add/n_infant", ["n_infant"]),
    ("rc/controls/add/m_adultage", ["m_adultage"]),
    ("rc/controls/add/castrk_b", ["castrk_b"]),
    ("rc/controls/add/rain", ["rain"]),
    ("rc/controls/add/l_r_totexp", ["l_r_totexp"]),
    ("rc/controls/add/hhsize_n_infant_m_adultage", ["hhsize", "n_infant", "m_adultage"]),
]

for spec_id, add_vars in extra_additions:
    ctrl = BASELINE_CONTROLS + add_vars
    df_ext = df_E.dropna(subset=['ch_r_totexp', 'ch_r_nonlabinc'] + ctrl).copy()
    if len(df_ext) >= 50:
        run_spec(
            spec_id, "modules/robustness/controls.md#additional-controls", "G1",
            "ch_r_totexp", "ch_r_nonlabinc", ctrl,
            "", "none", df_ext,
            "iid",
            f"Village E, N={len(df_ext)}", f"baseline + {', '.join(add_vars)}",
            axis_block_name="controls",
            axis_block={"spec_id": spec_id, "family": "add",
                        "added": add_vars, "n_controls": len(ctrl)})


# ============================================================
# RC: RANDOM CONTROL SUBSETS
# ============================================================

print("\nRunning random control subset variants...")

rng = np.random.RandomState(112498)
all_available_controls = BASELINE_CONTROLS + EXTRA_CONTROLS

for draw_i in range(1, 11):
    k = rng.randint(2, len(all_available_controls) + 1)
    chosen = list(rng.choice(all_available_controls, size=k, replace=False))
    excluded = [v for v in all_available_controls if v not in chosen]
    spec_id = f"rc/controls/subset/random_{draw_i:03d}"
    df_sub = df_E.dropna(subset=['ch_r_totexp', 'ch_r_nonlabinc'] + chosen).copy()
    if len(df_sub) >= 50:
        run_spec(
            spec_id, "modules/robustness/controls.md#subset-generation-specids", "G1",
            "ch_r_totexp", "ch_r_nonlabinc", chosen,
            "", "none", df_sub,
            "iid",
            f"Village E, N={len(df_sub)}", f"random subset draw {draw_i} ({len(chosen)} controls)",
            axis_block_name="controls",
            axis_block={"spec_id": spec_id, "family": "subset", "method": "random",
                        "seed": 112498, "draw_index": draw_i,
                        "included": chosen, "excluded": excluded,
                        "n_controls": len(chosen)})


# ============================================================
# RC: SAMPLE TRIMMING
# ============================================================

print("\nRunning sample trimming variants...")

# Trim outcome at 1st/99th percentile
q01 = df_base['ch_r_totexp'].quantile(0.01)
q99 = df_base['ch_r_totexp'].quantile(0.99)
df_trim1 = df_base[(df_base['ch_r_totexp'] >= q01) & (df_base['ch_r_totexp'] <= q99)].copy()
n_before = len(df_base)

run_spec(
    "rc/sample/outliers/trim_y_1_99",
    "modules/robustness/sample.md#outliers-and-influential-observations", "G1",
    "ch_r_totexp", "ch_r_nonlabinc", BASELINE_CONTROLS,
    "", "none", df_trim1,
    "iid",
    f"trim ch_r_totexp [1%,99%], N={len(df_trim1)}", "full baseline controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_1_99", "axis": "outliers",
                "rule": "trim", "params": {"var": "ch_r_totexp", "lower_q": 0.01, "upper_q": 0.99},
                "n_obs_before": n_before, "n_obs_after": len(df_trim1)})

# Trim outcome at 5th/95th percentile
q05 = df_base['ch_r_totexp'].quantile(0.05)
q95 = df_base['ch_r_totexp'].quantile(0.95)
df_trim5 = df_base[(df_base['ch_r_totexp'] >= q05) & (df_base['ch_r_totexp'] <= q95)].copy()

run_spec(
    "rc/sample/outliers/trim_y_5_95",
    "modules/robustness/sample.md#outliers-and-influential-observations", "G1",
    "ch_r_totexp", "ch_r_nonlabinc", BASELINE_CONTROLS,
    "", "none", df_trim5,
    "iid",
    f"trim ch_r_totexp [5%,95%], N={len(df_trim5)}", "full baseline controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_5_95", "axis": "outliers",
                "rule": "trim", "params": {"var": "ch_r_totexp", "lower_q": 0.05, "upper_q": 0.95},
                "n_obs_before": n_before, "n_obs_after": len(df_trim5)})


# ============================================================
# RC: VILLAGE SAMPLE VARIANTS
# ============================================================

print("\nRunning village sample variants...")

# All three villages pooled
df_all_base = df_all.dropna(subset=baseline_vars).copy()
if len(df_all_base) > 50:
    run_spec(
        "rc/sample/village/all_three_villages",
        "modules/robustness/sample.md#sample-restrictions", "G1",
        "ch_r_totexp", "ch_r_nonlabinc", BASELINE_CONTROLS,
        "", "none", df_all_base,
        "iid",
        f"All 3 villages pooled, N={len(df_all_base)}", "baseline controls",
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/village/all_three_villages",
                    "axis": "village", "villages": ["A", "C", "E"]})

# Village A only
if len(df_A_base) > 50:
    run_spec(
        "rc/sample/village/A_only",
        "modules/robustness/sample.md#sample-restrictions", "G1",
        "ch_r_totexp", "ch_r_nonlabinc", BASELINE_CONTROLS,
        "", "none", df_A_base,
        "iid",
        f"Village A (Aurepalle), N={len(df_A_base)}", "baseline controls",
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/village/A_only", "axis": "village", "village": "A"})

# Village C only
if len(df_C_base) > 50:
    run_spec(
        "rc/sample/village/C_only",
        "modules/robustness/sample.md#sample-restrictions", "G1",
        "ch_r_totexp", "ch_r_nonlabinc", BASELINE_CONTROLS,
        "", "none", df_C_base,
        "iid",
        f"Village C (Shirapur), N={len(df_C_base)}", "baseline controls",
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/village/C_only", "axis": "village", "village": "C"})


# ============================================================
# RC: MINIMUM OBSERVATIONS THRESHOLD
# ============================================================

print("\nRunning min obs variants...")

for min_threshold in [60, 100]:
    df_min = prepare_village_data(df_raw, 'E', rain_agg, min_obs=min_threshold)
    df_min_base = df_min.dropna(subset=baseline_vars).copy()
    if len(df_min_base) > 50:
        run_spec(
            f"rc/sample/min_obs/{min_threshold}",
            "modules/robustness/sample.md#sample-restrictions", "G1",
            "ch_r_totexp", "ch_r_nonlabinc", BASELINE_CONTROLS,
            "", "none", df_min_base,
            "iid",
            f"Village E, min_obs>={min_threshold}, N={len(df_min_base)}", "baseline controls",
            axis_block_name="sample",
            axis_block={"spec_id": f"rc/sample/min_obs/{min_threshold}",
                        "axis": "min_obs", "threshold": min_threshold,
                        "n_hh": df_min_base['hh_num'].nunique()})


# ============================================================
# RC: FIXED EFFECTS
# ============================================================

print("\nRunning FE variants...")

# Household FE
run_spec(
    "rc/fe/add/hh",
    "modules/robustness/fixed_effects.md#additive-fe-variations-relative-to-baseline", "G1",
    "ch_r_totexp", "ch_r_nonlabinc", BASELINE_CONTROLS,
    "hh_str", "household FE", df_base,
    "iid",
    "Village E", "baseline controls + household FE",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/add/hh", "family": "add",
                "added": ["hh"], "baseline_fe": [], "new_fe": ["hh"]})

# Year FE
run_spec(
    "rc/fe/add/year",
    "modules/robustness/fixed_effects.md#additive-fe-variations-relative-to-baseline", "G1",
    "ch_r_totexp", "ch_r_nonlabinc", BASELINE_CONTROLS,
    "year_str", "year FE", df_base,
    "iid",
    "Village E", "baseline controls + year FE",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/add/year", "family": "add",
                "added": ["year"], "baseline_fe": [], "new_fe": ["year"]})

# Household + Year FE
run_spec(
    "rc/fe/add/hh_year",
    "modules/robustness/fixed_effects.md#additive-fe-variations-relative-to-baseline", "G1",
    "ch_r_totexp", "ch_r_nonlabinc", BASELINE_CONTROLS,
    "hh_str + year_str", "household + year FE", df_base,
    "iid",
    "Village E", "baseline controls + household + year FE",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/add/hh_year", "family": "add",
                "added": ["hh", "year"], "baseline_fe": [], "new_fe": ["hh", "year"]})


# ============================================================
# RC: ALTERNATIVE OUTCOMES
# ============================================================

print("\nRunning alternative outcome variants...")

outcome_variants = [
    ("rc/form/outcome/ch_r_food", "ch_r_food", "change in real food consumption"),
    ("rc/form/outcome/ch_r_grain", "ch_r_grain", "change in real grain consumption"),
    ("rc/form/outcome/ch_r_ndur", "ch_r_ndur", "change in real non-durable consumption"),
    ("rc/form/outcome/ch_r_totexpnoleis", "ch_r_totexpnoleis",
     "change in real consumption without leisure"),
]

for spec_id, outcome, desc in outcome_variants:
    df_out = df_E.dropna(subset=[outcome, 'ch_r_nonlabinc'] + BASELINE_CONTROLS).copy()
    if len(df_out) >= 50:
        run_spec(
            spec_id, "modules/robustness/functional_form.md#outcome-alternatives", "G1",
            outcome, "ch_r_nonlabinc", BASELINE_CONTROLS,
            "", "none", df_out,
            "iid",
            f"Village E, N={len(df_out)}", f"{desc}",
            axis_block_name="functional_form",
            axis_block={"spec_id": spec_id, "family": "outcome",
                        "outcome": outcome, "desc": desc})


# ============================================================
# RC: ALTERNATIVE TREATMENTS
# ============================================================

print("\nRunning alternative treatment variants...")

# Change in total income per capita
df_ti = df_E.dropna(subset=['ch_r_totexp', 'ch_r_tot_inc_pc'] + BASELINE_CONTROLS).copy()
if len(df_ti) >= 50:
    run_spec(
        "rc/form/treatment/ch_r_tot_inc_pc",
        "modules/robustness/functional_form.md#treatment-alternatives", "G1",
        "ch_r_totexp", "ch_r_tot_inc_pc", BASELINE_CONTROLS,
        "", "none", df_ti,
        "iid",
        f"Village E, N={len(df_ti)}", "treatment = change in total income pc",
        axis_block_name="functional_form",
        axis_block={"spec_id": "rc/form/treatment/ch_r_tot_inc_pc",
                    "family": "treatment", "treatment": "ch_r_tot_inc_pc"})

# Change in wage
df_tw = df_E.dropna(subset=['ch_r_totexp', 'ch_r_wage'] + BASELINE_CONTROLS).copy()
if len(df_tw) >= 50:
    run_spec(
        "rc/form/treatment/ch_r_wage",
        "modules/robustness/functional_form.md#treatment-alternatives", "G1",
        "ch_r_totexp", "ch_r_wage", BASELINE_CONTROLS,
        "", "none", df_tw,
        "iid",
        f"Village E, N={len(df_tw)}", "treatment = change in real wage",
        axis_block_name="functional_form",
        axis_block={"spec_id": "rc/form/treatment/ch_r_wage",
                    "family": "treatment", "treatment": "ch_r_wage"})


# ============================================================
# RC: IV with different instrument sets
# ============================================================

print("\nRunning IV variants...")

# IV with year dummies as additional instruments (StandardTest.do line 403)
year_dummies = [f'd{y}' for y in range(75, 85)]
iv_vars_ext = iv_vars + year_dummies + ['stockvalCG', 'stockvalFE', 'stockvalIF', 'savings', 'n_month']
df_iv_ext = df_E.dropna(subset=iv_vars_ext).copy()
if len(df_iv_ext) >= 50:
    extra_inst = ['hhsize', 'n_infant', 'rain', 'l_r_totexp'] + year_dummies + \
                 ['n_month', 'stockvalCG', 'stockvalFE', 'stockvalIF', 'savings']
    run_iv(
        "rc/iv/extended_instruments",
        "modules/robustness/functional_form.md#estimator-alternatives", "G1",
        "ch_r_totexp", "ch_r_nonlabinc", iv_controls,
        extra_inst,
        "", "none", df_iv_ext,
        "iid",
        f"Village E (IV extended), N={len(df_iv_ext)}",
        "IV with year dummies + stocks + savings instruments",
        axis_block_name="estimation",
        axis_block={"spec_id": "rc/iv/extended_instruments", "estimator": "iv",
                    "instruments": extra_inst})

# IV with minimal instruments (just rainfall + lagged consumption)
df_iv_min = df_E.dropna(subset=['ch_r_totexp', 'ch_r_nonlabinc'] + iv_controls +
                        ['rain', 'l_r_totexp']).copy()
if len(df_iv_min) >= 50:
    run_iv(
        "rc/iv/minimal_instruments",
        "modules/robustness/functional_form.md#estimator-alternatives", "G1",
        "ch_r_totexp", "ch_r_nonlabinc", iv_controls,
        ['rain', 'l_r_totexp'],
        "", "none", df_iv_min,
        "iid",
        f"Village E (IV minimal), N={len(df_iv_min)}",
        "IV with rain + lagged consumption only",
        axis_block_name="estimation",
        axis_block={"spec_id": "rc/iv/minimal_instruments", "estimator": "iv",
                    "instruments": ['rain', 'l_r_totexp']})

# IV on all three villages
df_all_iv = df_all.dropna(subset=iv_vars).copy()
if len(df_all_iv) >= 50:
    run_iv(
        "rc/iv/all_villages",
        "modules/robustness/functional_form.md#estimator-alternatives", "G1",
        "ch_r_totexp", "ch_r_nonlabinc", iv_controls,
        ['hhsize', 'n_infant', 'rain', 'l_r_totexp'],
        "", "none", df_all_iv,
        "iid",
        f"All villages (IV), N={len(df_all_iv)}",
        "IV on pooled 3-village sample",
        axis_block_name="estimation",
        axis_block={"spec_id": "rc/iv/all_villages", "estimator": "iv",
                    "villages": ["A", "C", "E"]})


# ============================================================
# RC: CASTE AVERAGE CONSUMPTION instead of village average
# ============================================================

print("\nRunning caste avg consumption variant...")

df_caste_ctrl = df_E.dropna(subset=['ch_r_totexp', 'ch_r_nonlabinc',
                                     'ch_r_totexp_caste', 'l_r_wage', 'wt_nhh', 'intfreq']).copy()
if len(df_caste_ctrl) >= 50:
    run_spec(
        "rc/controls/alt/caste_avg_instead_of_vill_avg",
        "modules/robustness/controls.md#additional-controls", "G1",
        "ch_r_totexp", "ch_r_nonlabinc",
        ["ch_r_totexp_caste", "l_r_wage", "wt_nhh", "intfreq"],
        "", "none", df_caste_ctrl,
        "iid",
        f"Village E, N={len(df_caste_ctrl)}",
        "caste avg consumption change instead of village avg",
        axis_block_name="controls",
        axis_block={"spec_id": "rc/controls/alt/caste_avg_instead_of_vill_avg",
                    "family": "alternative", "notes": "Use caste-level avg consumption instead of village-level"})


# ============================================================
# RC: POOLED VILLAGES WITH VILLAGE FE
# ============================================================

print("\nRunning pooled with village FE...")

if len(df_all_base) > 50:
    run_spec(
        "rc/fe/add/village",
        "modules/robustness/fixed_effects.md#additive-fe-variations-relative-to-baseline", "G1",
        "ch_r_totexp", "ch_r_nonlabinc", BASELINE_CONTROLS,
        "vill_str", "village FE", df_all_base,
        "iid",
        f"All 3 villages, village FE, N={len(df_all_base)}", "baseline controls + village FE",
        axis_block_name="fixed_effects",
        axis_block={"spec_id": "rc/fe/add/village", "family": "add",
                    "added": ["village"], "baseline_fe": [], "new_fe": ["village"]})


# ============================================================
# INFERENCE VARIANTS (on baseline specification)
# ============================================================

print("\nRunning inference variants...")

baseline_run_id = f"{PAPER_ID}_run_001"
infer_counter = 0

baseline_controls_str = " + ".join(BASELINE_CONTROLS)
baseline_formula = f"ch_r_totexp ~ ch_r_nonlabinc + {baseline_controls_str}"


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


# HC1 robust SEs
run_inference_variant(
    baseline_run_id, "infer/se/hc/hc1",
    "modules/inference/standard_errors.md#heteroskedasticity-robust", "G1",
    baseline_formula, "", df_base, "ch_r_nonlabinc",
    "hetero", "HC1 (robust, no clustering)")

# Cluster by household
run_inference_variant(
    baseline_run_id, "infer/se/cluster/hh",
    "modules/inference/standard_errors.md#clustering", "G1",
    baseline_formula, "", df_base, "ch_r_nonlabinc",
    {"CRV1": "hh_str"}, "cluster(hh)")

# Cluster by year-month
run_inference_variant(
    baseline_run_id, "infer/se/cluster/yearmonth",
    "modules/inference/standard_errors.md#clustering", "G1",
    baseline_formula, "", df_base, "ch_r_nonlabinc",
    {"CRV1": "yearmonth"}, "cluster(yearmonth)")


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
        print(f"\nBaseline coef on ch_r_nonlabinc: {base_row['coefficient'].values[0]:.6f}")
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
md_lines.append("# Specification Search Report: 112498-V1")
md_lines.append("")
md_lines.append("**Paper:** Ogaki & Zhang (2001), \"Decreasing Relative Risk Aversion and Tests of Risk Sharing\", Econometrica 69(2)")
md_lines.append("")
md_lines.append("## Baseline Specification")
md_lines.append("")
md_lines.append("- **Design:** Panel OLS (standard risk-sharing test)")
md_lines.append("- **Outcome:** ch_r_totexp (change in real per-capita total consumption)")
md_lines.append("- **Treatment:** ch_r_nonlabinc (change in real per-capita non-labor income)")
md_lines.append("- **Controls:** ch_r_totexp_vill (village avg consumption change), l_r_wage, wt_nhh, intfreq")
md_lines.append("- **Fixed effects:** none (baseline)")
md_lines.append("- **Sample:** Village E (Kanzara), households with 80+ monthly observations, 1975-1984")
md_lines.append("- **Hypothesis:** Under full risk sharing, coef on ch_r_nonlabinc = 0; positive coef rejects full insurance")
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
    "Baseline (OLS)": successful[successful['spec_id'] == 'baseline'],
    "Baseline (IV)": successful[successful['spec_id'].str.contains('iv', case=False) & successful['spec_id'].str.startswith('baseline')],
    "Village variants": successful[successful['spec_id'].str.contains('vill', case=False) & ~successful['spec_id'].str.startswith('baseline')],
    "By-caste": successful[successful['spec_id'].str.contains('caste') & successful['spec_id'].str.startswith('baseline')],
    "Controls LOO": successful[successful['spec_id'].str.startswith('rc/controls/loo/')],
    "Controls Sets": successful[successful['spec_id'].str.startswith('rc/controls/sets/')],
    "Controls Progression": successful[successful['spec_id'].str.startswith('rc/controls/progression/')],
    "Controls Add": successful[successful['spec_id'].str.startswith('rc/controls/add/')],
    "Controls Subset": successful[successful['spec_id'].str.startswith('rc/controls/subset/')],
    "Controls Alt": successful[successful['spec_id'].str.startswith('rc/controls/alt/')],
    "Sample Trimming": successful[successful['spec_id'].str.startswith('rc/sample/outliers/')],
    "Sample Village": successful[successful['spec_id'].str.startswith('rc/sample/village/')],
    "Sample Min-obs": successful[successful['spec_id'].str.startswith('rc/sample/min_obs/')],
    "Fixed Effects": successful[successful['spec_id'].str.startswith('rc/fe/')],
    "Alt Outcomes": successful[successful['spec_id'].str.startswith('rc/form/outcome/')],
    "Alt Treatments": successful[successful['spec_id'].str.startswith('rc/form/treatment/')],
    "IV variants": successful[successful['spec_id'].str.startswith('rc/iv/')],
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
