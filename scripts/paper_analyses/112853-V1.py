"""
Specification Search Script for Manuelli & Seshadri (2014)
"Human Capital and the Wealth of Nations"
American Economic Review, 104(11), 3752-3777.

Paper ID: 112853-V1

Surface-driven execution:
  - G1: log(av_rel_y) ~ av_sch + controls, HC1 robust SEs
  - Cross-country OLS development accounting
  - 50+ specifications across controls LOO, controls subsets, controls progression,
    sample trimming, subgroup analysis, outcome variants, functional form variants

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

PAPER_ID = "112853-V1"
DATA_DIR = "data/downloads/extracted/112853-V1"
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
# Data Loading and Preparation
# ============================================================

BASE_DATA = f"{DATA_DIR}/data/Creating_WorldDistributionTable"

# --- PWT 8.0 ---
pwt = pd.read_stata(f"{BASE_DATA}/pwt80.dta")
pwt = pwt.dropna(subset=['rgdpna', 'emp'])
pwt = pwt[['country', 'year', 'rgdpna', 'emp', 'pl_i', 'pl_c', 'pop']].copy()
pwt['y'] = pwt['rgdpna'] / pwt['emp']
pwt['y_per_capita'] = pwt['rgdpna'] / pwt['pop']
pwt['pk'] = pwt['pl_i'] / pwt['pl_c']
pwt.loc[pwt['pl_i'].isna() | pwt['pl_c'].isna(), 'pk'] = np.nan
pwt = pwt[(pwt['year'] >= 2003) & (pwt['year'] <= 2007)].copy()
pwt.loc[pwt['country'] == 'China', 'country'] = 'China Version 1'
pwt.loc[pwt['country'] == 'Trinidad & Tobago', 'country'] = 'Trinidad &Tobago'
pwt['pop_max'] = pwt.groupby('country')['pop'].transform('max')
pwt = pwt[pwt['pop_max'] >= 1]
oil_countries = ['Qatar', 'Kuwait', 'Equatorial Guinea', 'United Arab Emirates',
                 'Norway', 'Brunei', 'Bahrain', 'Singapore']
pwt = pwt[~pwt['country'].isin(oil_countries)]

us_refs = {2003: 84367.45, 2004: 86350.95, 2005: 87483.43, 2006: 88191.02, 2007: 88931.5}
us_pk = {2003: 0.8686909, 2004: 0.884138, 2005: 0.9240157, 2006: 0.9485181, 2007: 0.965291}
pwt['rel_y'] = pwt.apply(lambda r: r['y'] / us_refs.get(int(r['year']), np.nan), axis=1)
pwt['rel_pk'] = pwt.apply(
    lambda r: r['pk'] / us_pk.get(int(r['year']), np.nan) if pd.notna(r['pk']) else np.nan, axis=1)

agg = pwt.groupby('country').agg(
    av_y=('y', 'mean'), av_rel_y=('rel_y', 'mean'), av_rel_pk=('rel_pk', 'mean'),
    av_y_per_capita=('y_per_capita', 'mean'), pop_total=('pop', 'mean')
).reset_index()

# --- Barro-Lee ---
bl = pd.read_stata(f"{BASE_DATA}/Barro_Lee_age25more.dta")
bl = bl[bl['year'] == 2000]
bl_avg = bl.groupby('BLcode').agg(av_sch=('yr_sch', 'mean')).reset_index()
bl_first = bl.drop_duplicates(subset='BLcode', keep='first')[['BLcode', 'country']].copy()
bl_merged = bl_first.merge(bl_avg, on='BLcode')
name_map_bl = {
    'Brunei Darussalam': 'Brunei', 'China': 'China Version 1',
    'China, Hong Kong Special Administrative Region': 'Hong Kong',
    'China, Macao Special Administrative Region': 'Macao',
    'Congo': 'Congo, Republic of',
    'Democratic Republic of the Congo': 'Congo, Dem. Rep.',
    'Dominican Rep.': 'Dominican Republic', 'Gambia': 'Gambia, The',
    'Iran (Islamic Republic of)': 'Iran',
    "Lao People's Democratic Republic": 'Laos',
    'Libyan Arab Jamahiriya': 'Libya', 'Republic of Korea': 'Korea, Republic of',
    'Republic of Moldova': 'Moldova', 'Russian Federation': 'Russia',
    'Slovakia': 'Slovak Republic', 'Syrian Arab Republic': 'Syria',
    'Trinidad and Tobago': 'Trinidad &Tobago', 'USA': 'United States',
    'United Republic of Tanzania': 'Tanzania', 'Viet Nam': 'Vietnam'
}
bl_merged['country'] = bl_merged['country'].replace(name_map_bl)
combined = agg.merge(bl_merged[['country', 'av_sch']], on='country', how='inner')

# --- Life expectancy ---
le = pd.read_csv(f"{BASE_DATA}/World_databank_life_expectancy.csv",
                 header=None, encoding='latin-1', sep=None, engine='python')
le = le.iloc[1:]  # skip header
le.columns = range(le.shape[1])
le.rename(columns={0: 'country'}, inplace=True)
# Cols 4..15 = T2000..T2011; we want T2003..T2007 = cols 7..11
for c in [7, 8, 9, 10, 11]:
    le[c] = pd.to_numeric(le[c], errors='coerce')
le['av_T'] = le[[7, 8, 9, 10, 11]].mean(axis=1)
le = le.dropna(subset=['av_T'])
le_name_map = {
    'Russian Federation': 'Russia', 'Macedonia, FYR': 'Macedonia',
    'Brunei Darussalam': 'Brunei', "Yemen, Rep.": 'Yemen',
    'Hong Kong SAR, China': 'Hong Kong', 'Iran, Islamic Rep.': 'Iran',
    'Venezuela, RB': 'Venezuela', 'Macao SAR, China': 'Macao',
    'Egypt, Arab Rep.': 'Egypt', 'Kyrgyz Republic': 'Kyrgyzstan',
    'Korea, Rep.': 'Korea, Republic of', 'China': 'China Version 1',
    'St. Vincent and the Grenadines': 'St.Vincent & Grenadines',
    'Trinidad and Tobago': 'Trinidad &Tobago',
    'Syrian Arab Republic': 'Syria', 'Bahamas, The': 'Bahamas',
    'Congo, Rep.': 'Congo, Republic of',
    "Cote d'Ivoire": 'Cote d`Ivoire'
}
le['country'] = le['country'].replace(le_name_map)
le = le[['country', 'av_T']].drop_duplicates(subset='country', keep='first')
combined = combined.merge(le, on='country', how='inner')

# --- IMR ---
imr = pd.read_csv(f"{BASE_DATA}/infant_mortality_rate.csv", header=None, encoding='latin-1')
imr = imr[[0, 1]].rename(columns={0: 'country', 1: 'IMR'})
imr_name_map = {
    'Congo, Democratic Republic of': 'Congo, Dem. Rep.',
    'China': 'China Version 1',
    'Central Aftrican Republic': 'Central African Republic',
    'Saint Lucia': 'St. Lucia',
    'Saint Vincent and the Grenadines': 'St.Vincent & Grenadine',
    "Cote d'Ivoire": 'Cote d`Ivoire'
}
imr['country'] = imr['country'].replace(imr_name_map)
imr = imr[~((imr['IMR'] >= 10.3) & (imr['country'] == 'Netherlands'))]
combined = combined.merge(imr[['country', 'IMR']], on='country', how='inner')

# --- UNESCO education expenditure ---
unesco = pd.read_csv(f"{BASE_DATA}/UNESCO_education_expense.csv",
                     header=None, encoding='latin-1')
unesco_name_map = {
    'United Republic of Tanzania': 'Tanzania',
    'Bolivia (Plurinational State of)': 'Bolivia',
    'Brunei Darussalam': 'Brunei',
    'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom',
    'China, Hong Kong Special Administrative Region': 'Hong Kong',
    'Republic of Moldova': 'Moldova', 'Iran (Islamic Republic of)': 'Iran',
    'United States of America': 'United States',
    'The former Yugoslav Republic of Macedonia': 'Macedonia',
    'Viet Nam': 'Vietnam', 'Venezuela (Bolivarian Republic of)': 'Venezuela',
    "Lao People's Democratic Republic": 'Laos',
    'Congo': 'Congo, Republic of',
    'Saint Vincent and the Grenadines': 'St.Vincent & Grenadines',
    'Democratic Republic of the Congo': 'Congo, Dem. Rep.',
    'Gambia': 'Gambia, The', 'Russian Federation': 'Russia',
    'China, Macao Special Administrative Region': 'Macao',
    'Trinidad and Tobago': 'Trinidad &Tobago',
    'Syrian Arab Republic': 'Syria', 'Slovakia': 'Slovak Republic',
    'Republic of Korea': 'Korea, Republic of',
}
unesco[0] = unesco[0].replace(unesco_name_map)
unesco.loc[unesco[0].str.contains("Ivoire", na=False), 0] = 'Cote d`Ivoire'
for c in [5, 6, 7, 8, 9]:
    unesco[c] = pd.to_numeric(unesco[c], errors='coerce')
    unesco.loc[unesco[c] == -9999, c] = np.nan
unesco['av_x_s'] = unesco[[5, 6, 7, 8, 9]].mean(axis=1)
unesco = unesco.dropna(subset=['av_x_s'])
unesco = unesco[[0, 'av_x_s']].rename(columns={0: 'country'}).drop_duplicates(
    subset='country', keep='first')
combined = combined.merge(unesco, on='country', how='inner')

# --- TFR ---
tfr = pd.read_csv(f"{BASE_DATA}/TFR.csv", header=None)
tfr = tfr.iloc[1:]
tfr_name_map = {
    'Macedonia, FYR': 'Macedonia', 'Syrian Arab Republic': 'Syria',
    'Hong Kong SAR, China': 'Hong Kong', 'Russian Federation': 'Russia',
    'Egypt, Arab Rep.': 'Egypt', 'Bahamas, The': 'Bahamas',
    'Korea, Rep.': 'Korea, Republic of',
    'St. Vincent and the Grenadines': 'St.Vincent & Grenadines',
    'Yemen, Rep.': 'Yemen', 'China': 'China Version 1',
    'Trinidad and Tobago': 'Trinidad &Tobago',
    'Macao SAR, China': 'Macao', 'Brunei Darussalam': 'Brunei',
    'Lao PDR': 'Laos', 'Congo, Rep.': 'Congo, Republic of',
    'Iran, Islamic Rep.': 'Iran', 'Venezuela, RB': 'Venezuela',
    "Cote d'Ivoire": 'Cote d`Ivoire'
}
tfr[0] = tfr[0].replace(tfr_name_map)
for c in [4, 5, 6, 7, 8]:
    tfr[c] = pd.to_numeric(tfr[c], errors='coerce')
tfr['av_TFR'] = tfr[[4, 5, 6, 7, 8]].mean(axis=1)
tfr = tfr[[0, 'av_TFR']].rename(columns={0: 'country'}).drop_duplicates(
    subset='country', keep='first')
combined = combined.merge(tfr, on='country', how='inner')

# Adjust TFR for infant mortality (matching do file)
combined.loc[combined['IMR'].notna(), 'av_TFR'] = (
    combined['av_TFR'] * (1 - combined['IMR'] / 1000))

# Drop missing on key vars
key_vars = ['av_rel_y', 'av_rel_pk', 'av_sch', 'av_T', 'av_x_s', 'av_TFR']
df = combined.dropna(subset=key_vars).copy()

# Create derived variables
df['log_av_rel_y'] = np.log(df['av_rel_y'])
df['log_av_y_per_capita'] = np.log(df['av_y_per_capita'])
df['log_av_y'] = np.log(df['av_y'])
df['log_av_sch'] = np.log(df['av_sch'].clip(lower=0.1))
df['av_sch_sq'] = df['av_sch'] ** 2

# Decile assignment
df = df.sort_values('av_rel_y')
df['pct_nomiss'] = pd.qcut(df['av_rel_y'], 10, labels=False) + 1
df['pct_str'] = df['pct_nomiss'].astype(str)

# Convert float32 to float64
for col in df.columns:
    if df[col].dtype == np.float32:
        df[col] = df[col].astype(np.float64)

print(f"Loaded data: {df.shape[0]} countries, {df.shape[1]} columns")
print(f"Key vars: log_av_rel_y range [{df['log_av_rel_y'].min():.2f}, {df['log_av_rel_y'].max():.2f}]")
print(f"av_sch range [{df['av_sch'].min():.2f}, {df['av_sch'].max():.2f}]")

# Define control groups
ALL_CONTROLS = ["av_T", "av_TFR", "av_x_s", "av_rel_pk", "IMR"]
DEMOGRAPHIC_CONTROLS = ["av_T", "av_TFR", "IMR"]
ECONOMIC_CONTROLS = ["av_x_s", "av_rel_pk"]

# ============================================================
# Accumulators
# ============================================================

results = []
inference_results = []
spec_run_counter = 0


# ============================================================
# Helper: run_spec (OLS via pyfixest)
# ============================================================

def run_spec(spec_id, spec_tree_path, baseline_group_id,
             outcome_var, treatment_var, controls, fe_formula_str,
             fe_desc, data, vcov, sample_desc, controls_desc,
             cluster_var=None,
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
                       "method": "HC1"},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"cross_sectional_ols": design_audit},
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
            "cluster_var": cluster_var or "",
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
            "cluster_var": cluster_var or "",
            "run_success": 0,
            "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


# ============================================================
# Helper: run_statsmodels_spec (for WLS, quantile, robust regression)
# ============================================================

def run_sm_spec(spec_id, spec_tree_path, baseline_group_id,
                outcome_var, treatment_var, controls, data,
                sample_desc, controls_desc, estimator_type="wls",
                weights_var=None,
                axis_block_name=None, axis_block=None, notes=""):
    """Run a specification using statsmodels (for WLS, quantile, robust reg)."""
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        import statsmodels.api as sm

        rhs_vars = [treatment_var] + list(controls)
        est_data = data.dropna(subset=[outcome_var] + rhs_vars).copy()

        y = est_data[outcome_var].astype(float)
        X = sm.add_constant(est_data[rhs_vars].astype(float))

        if estimator_type == "wls":
            w = est_data[weights_var].astype(float) if weights_var else None
            model = sm.WLS(y, X, weights=w)
            result = model.fit(cov_type='HC1')
        elif estimator_type == "median":
            model = sm.QuantReg(y, X)
            result = model.fit(q=0.5)
        elif estimator_type == "robust":
            model = sm.RLM(y, X)
            result = model.fit()
        else:
            raise ValueError(f"Unknown estimator: {estimator_type}")

        coef_val = float(result.params[treatment_var])
        se_val = float(result.bse[treatment_var])
        pval = float(result.pvalues[treatment_var])
        ci = result.conf_int()
        ci_lower = float(ci.loc[treatment_var, 0])
        ci_upper = float(ci.loc[treatment_var, 1])
        nobs = int(result.nobs)
        try:
            r2 = float(result.rsquared) if hasattr(result, 'rsquared') else np.nan
        except Exception:
            r2 = np.nan

        all_coefs = {k: float(v) for k, v in result.params.items()
                     if k != 'const'}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": f"infer/se/{estimator_type}",
                       "method": estimator_type,
                       "estimator": estimator_type},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"cross_sectional_ols": design_audit},
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
            "fixed_effects": "none",
            "controls_desc": controls_desc,
            "cluster_var": "",
            "run_success": 1,
            "run_error": ""
        })
        return run_id, coef_val, se_val, pval, nobs

    except Exception as e:
        err_msg = str(e)[:240]
        err_details = error_details_from_exception(e, stage=f"{estimator_type}_estimation")
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
            "fixed_effects": "none",
            "controls_desc": controls_desc,
            "cluster_var": "",
            "run_success": 0,
            "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


# ============================================================
# BASELINE: Cross-country OLS with full controls
# ============================================================

print("Running baseline specification...")
base_run_id, base_coef, base_se, base_pval, base_nobs = run_spec(
    "baseline", "designs/cross_sectional_ols.md#baseline", "G1",
    "log_av_rel_y", "av_sch", ALL_CONTROLS,
    "", "none", df,
    "hetero",
    f"Full sample, N={len(df)}", "demographics + economic (5 controls)")

print(f"  Baseline: coef={base_coef:.4f}, se={base_se:.4f}, p={base_pval:.4f}, N={base_nobs}")


# ============================================================
# RC: CONTROLS LOO — Drop one control at a time
# ============================================================

print("Running controls LOO variants...")

LOO_MAP = {
    "rc/controls/loo/drop_av_T": ["av_T"],
    "rc/controls/loo/drop_av_TFR": ["av_TFR"],
    "rc/controls/loo/drop_av_x_s": ["av_x_s"],
    "rc/controls/loo/drop_av_rel_pk": ["av_rel_pk"],
    "rc/controls/loo/drop_IMR": ["IMR"],
}

for spec_id, drop_vars in LOO_MAP.items():
    ctrl = [c for c in ALL_CONTROLS if c not in drop_vars]
    run_spec(
        spec_id, "modules/robustness/controls.md#leave-one-out-controls-loo", "G1",
        "log_av_rel_y", "av_sch", ctrl,
        "", "none", df,
        "hetero",
        "Full sample", f"baseline minus {', '.join(drop_vars)}",
        axis_block_name="controls",
        axis_block={"spec_id": spec_id, "family": "loo",
                    "dropped": drop_vars, "n_controls": len(ctrl)})


# ============================================================
# RC: CONTROL SETS (named subsets)
# ============================================================

print("Running control set variants...")

# No controls (bivariate)
run_spec(
    "rc/controls/sets/none",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "log_av_rel_y", "av_sch", [],
    "", "none", df,
    "hetero",
    "Full sample", "none (bivariate)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/none", "family": "sets",
                "n_controls": 0, "set_name": "none"})

# Demographics only
run_spec(
    "rc/controls/sets/demographics_only",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "log_av_rel_y", "av_sch", DEMOGRAPHIC_CONTROLS,
    "", "none", df,
    "hetero",
    "Full sample", "demographics only (av_T, av_TFR, IMR)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/demographics_only", "family": "sets",
                "n_controls": len(DEMOGRAPHIC_CONTROLS), "set_name": "demographics_only"})

# Economic only
run_spec(
    "rc/controls/sets/economic_only",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "log_av_rel_y", "av_sch", ECONOMIC_CONTROLS,
    "", "none", df,
    "hetero",
    "Full sample", "economic only (av_x_s, av_rel_pk)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/economic_only", "family": "sets",
                "n_controls": len(ECONOMIC_CONTROLS), "set_name": "economic_only"})

# Full
run_spec(
    "rc/controls/sets/full",
    "modules/robustness/controls.md#standard-control-sets", "G1",
    "log_av_rel_y", "av_sch", ALL_CONTROLS,
    "", "none", df,
    "hetero",
    "Full sample", "all 5 controls (same as baseline)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/full", "family": "sets",
                "n_controls": len(ALL_CONTROLS), "set_name": "full"})


# ============================================================
# RC: CONTROL PROGRESSION (build-up)
# ============================================================

print("Running control progression variants...")

# Bivariate
run_spec(
    "rc/controls/progression/bivariate",
    "modules/robustness/controls.md#control-progression-build-up", "G1",
    "log_av_rel_y", "av_sch", [],
    "", "none", df,
    "hetero",
    "Full sample", "bivariate (schooling only)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/bivariate", "family": "progression",
                "n_controls": 0, "set_name": "bivariate"})

# + life expectancy
run_spec(
    "rc/controls/progression/plus_lifeexp",
    "modules/robustness/controls.md#control-progression-build-up", "G1",
    "log_av_rel_y", "av_sch", ["av_T"],
    "", "none", df,
    "hetero",
    "Full sample", "schooling + life expectancy",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/plus_lifeexp", "family": "progression",
                "n_controls": 1, "set_name": "plus_lifeexp"})

# + life expectancy + TFR
run_spec(
    "rc/controls/progression/plus_lifeexp_tfr",
    "modules/robustness/controls.md#control-progression-build-up", "G1",
    "log_av_rel_y", "av_sch", ["av_T", "av_TFR"],
    "", "none", df,
    "hetero",
    "Full sample", "schooling + life exp + TFR",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/plus_lifeexp_tfr", "family": "progression",
                "n_controls": 2, "set_name": "plus_lifeexp_tfr"})

# + life expectancy + TFR + ed expenditure
run_spec(
    "rc/controls/progression/plus_lifeexp_tfr_edexp",
    "modules/robustness/controls.md#control-progression-build-up", "G1",
    "log_av_rel_y", "av_sch", ["av_T", "av_TFR", "av_x_s"],
    "", "none", df,
    "hetero",
    "Full sample", "schooling + life exp + TFR + ed exp",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/plus_lifeexp_tfr_edexp", "family": "progression",
                "n_controls": 3, "set_name": "plus_lifeexp_tfr_edexp"})

# Full
run_spec(
    "rc/controls/progression/full",
    "modules/robustness/controls.md#control-progression-build-up", "G1",
    "log_av_rel_y", "av_sch", ALL_CONTROLS,
    "", "none", df,
    "hetero",
    "Full sample", "all 5 controls (full progression)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/progression/full", "family": "progression",
                "n_controls": 5, "set_name": "full"})


# ============================================================
# RC: CONTROL SUBSET (random draws)
# ============================================================

print("Running random control subset variants...")

rng = np.random.RandomState(112853)
subset_pool = ALL_CONTROLS.copy()

for draw_i in range(1, 16):
    k = rng.randint(1, len(subset_pool) + 1)
    chosen = list(rng.choice(subset_pool, size=k, replace=False))
    excluded = [v for v in subset_pool if v not in chosen]
    spec_id = f"rc/controls/subset/random_{draw_i:03d}"
    run_spec(
        spec_id, "modules/robustness/controls.md#subset-generation-specids", "G1",
        "log_av_rel_y", "av_sch", chosen,
        "", "none", df,
        "hetero",
        "Full sample", f"random subset draw {draw_i} ({len(chosen)} controls)",
        axis_block_name="controls",
        axis_block={"spec_id": spec_id, "family": "subset", "method": "random",
                    "seed": 112853, "draw_index": draw_i,
                    "included": chosen, "excluded": excluded,
                    "n_controls": len(chosen)})


# ============================================================
# RC: SAMPLE TRIMMING
# ============================================================

print("Running sample trimming variants...")

n_before = len(df)

# Trim outcome at 1st/99th percentile
q01 = df['log_av_rel_y'].quantile(0.01)
q99 = df['log_av_rel_y'].quantile(0.99)
df_trim1 = df[(df['log_av_rel_y'] >= q01) & (df['log_av_rel_y'] <= q99)].copy()
run_spec(
    "rc/sample/outliers/trim_y_1_99",
    "modules/robustness/sample.md#outliers-and-influential-observations", "G1",
    "log_av_rel_y", "av_sch", ALL_CONTROLS,
    "", "none", df_trim1,
    "hetero",
    f"trim log_av_rel_y [1%,99%], N={len(df_trim1)}", "full controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_1_99", "axis": "outliers",
                "rule": "trim", "params": {"var": "log_av_rel_y", "lower_q": 0.01, "upper_q": 0.99},
                "n_obs_before": n_before, "n_obs_after": len(df_trim1)})

# Trim at 5th/95th
q05 = df['log_av_rel_y'].quantile(0.05)
q95 = df['log_av_rel_y'].quantile(0.95)
df_trim5 = df[(df['log_av_rel_y'] >= q05) & (df['log_av_rel_y'] <= q95)].copy()
run_spec(
    "rc/sample/outliers/trim_y_5_95",
    "modules/robustness/sample.md#outliers-and-influential-observations", "G1",
    "log_av_rel_y", "av_sch", ALL_CONTROLS,
    "", "none", df_trim5,
    "hetero",
    f"trim log_av_rel_y [5%,95%], N={len(df_trim5)}", "full controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_5_95", "axis": "outliers",
                "rule": "trim", "params": {"var": "log_av_rel_y", "lower_q": 0.05, "upper_q": 0.95},
                "n_obs_before": n_before, "n_obs_after": len(df_trim5)})

# Trim at 10th/90th
q10 = df['log_av_rel_y'].quantile(0.10)
q90 = df['log_av_rel_y'].quantile(0.90)
df_trim10 = df[(df['log_av_rel_y'] >= q10) & (df['log_av_rel_y'] <= q90)].copy()
run_spec(
    "rc/sample/outliers/trim_y_10_90",
    "modules/robustness/sample.md#outliers-and-influential-observations", "G1",
    "log_av_rel_y", "av_sch", ALL_CONTROLS,
    "", "none", df_trim10,
    "hetero",
    f"trim log_av_rel_y [10%,90%], N={len(df_trim10)}", "full controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_y_10_90", "axis": "outliers",
                "rule": "trim", "params": {"var": "log_av_rel_y", "lower_q": 0.10, "upper_q": 0.90},
                "n_obs_before": n_before, "n_obs_after": len(df_trim10)})


# ============================================================
# RC: SUBGROUP ANALYSIS
# ============================================================

print("Running subgroup variants...")

# Drop poorest decile
df_no_d1 = df[df['pct_nomiss'] > 1].copy()
run_spec(
    "rc/sample/subgroup/drop_decile_1",
    "modules/robustness/sample.md#subgroup-analysis", "G1",
    "log_av_rel_y", "av_sch", ALL_CONTROLS,
    "", "none", df_no_d1,
    "hetero",
    f"Drop decile 1 (poorest), N={len(df_no_d1)}", "full controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subgroup/drop_decile_1", "axis": "subgroup",
                "rule": "drop_decile_1"})

# Drop richest decile
df_no_d10 = df[df['pct_nomiss'] < 10].copy()
run_spec(
    "rc/sample/subgroup/drop_decile_10",
    "modules/robustness/sample.md#subgroup-analysis", "G1",
    "log_av_rel_y", "av_sch", ALL_CONTROLS,
    "", "none", df_no_d10,
    "hetero",
    f"Drop decile 10 (richest), N={len(df_no_d10)}", "full controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subgroup/drop_decile_10", "axis": "subgroup",
                "rule": "drop_decile_10"})

# Drop countries adjacent to oil exporters (high rel_pk)
df_no_oil = df[df['av_rel_pk'] < df['av_rel_pk'].quantile(0.90)].copy()
run_spec(
    "rc/sample/subgroup/drop_oil_adjacent",
    "modules/robustness/sample.md#subgroup-analysis", "G1",
    "log_av_rel_y", "av_sch", ALL_CONTROLS,
    "", "none", df_no_oil,
    "hetero",
    f"Drop high rel_pk (>90th pctile), N={len(df_no_oil)}", "full controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subgroup/drop_oil_adjacent", "axis": "subgroup",
                "rule": "drop_high_relpk"})

# Drop small population countries
df_big = df[df['pop_total'] >= df['pop_total'].quantile(0.25)].copy()
run_spec(
    "rc/sample/subgroup/drop_small_pop",
    "modules/robustness/sample.md#subgroup-analysis", "G1",
    "log_av_rel_y", "av_sch", ALL_CONTROLS,
    "", "none", df_big,
    "hetero",
    f"Drop small pop (<25th pctile), N={len(df_big)}", "full controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subgroup/drop_small_pop", "axis": "subgroup",
                "rule": "drop_small_pop"})

# Above-median income
med_y = df['av_rel_y'].median()
df_above = df[df['av_rel_y'] >= med_y].copy()
run_spec(
    "rc/sample/subgroup/above_median_income",
    "modules/robustness/sample.md#subgroup-analysis", "G1",
    "log_av_rel_y", "av_sch", ALL_CONTROLS,
    "", "none", df_above,
    "hetero",
    f"Above median income, N={len(df_above)}", "full controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subgroup/above_median_income", "axis": "subgroup",
                "rule": "above_median_income"})

# Below-median income
df_below = df[df['av_rel_y'] < med_y].copy()
run_spec(
    "rc/sample/subgroup/below_median_income",
    "modules/robustness/sample.md#subgroup-analysis", "G1",
    "log_av_rel_y", "av_sch", ALL_CONTROLS,
    "", "none", df_below,
    "hetero",
    f"Below median income, N={len(df_below)}", "full controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subgroup/below_median_income", "axis": "subgroup",
                "rule": "below_median_income"})


# ============================================================
# RC: OUTCOME / FUNCTIONAL FORM VARIANTS
# ============================================================

print("Running outcome and functional form variants...")

# Log GDP per capita (instead of per worker)
run_spec(
    "rc/form/outcome/log_av_y_per_capita",
    "modules/robustness/functional_form.md#outcome-alternatives", "G1",
    "log_av_y_per_capita", "av_sch", ALL_CONTROLS,
    "", "none", df,
    "hetero",
    "Full sample", "full controls (outcome: log GDP per capita)",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/log_av_y_per_capita",
                "outcome": "log_av_y_per_capita",
                "notes": "GDP per capita instead of per worker"})

# Log output per worker (absolute, not relative to US)
run_spec(
    "rc/form/outcome/log_av_y",
    "modules/robustness/functional_form.md#outcome-alternatives", "G1",
    "log_av_y", "av_sch", ALL_CONTROLS,
    "", "none", df,
    "hetero",
    "Full sample", "full controls (outcome: log output per worker)",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/log_av_y",
                "outcome": "log_av_y",
                "notes": "Absolute output per worker instead of relative to US"})

# Level outcome (not log)
run_spec(
    "rc/form/outcome/av_rel_y_levels",
    "modules/robustness/functional_form.md#outcome-alternatives", "G1",
    "av_rel_y", "av_sch", ALL_CONTROLS,
    "", "none", df,
    "hetero",
    "Full sample", "full controls (outcome: levels, not log)",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/outcome/av_rel_y_levels",
                "outcome": "av_rel_y",
                "notes": "Level outcome (relative Y) instead of log"})

# Treatment: schooling squared (quadratic)
run_spec(
    "rc/form/treatment/av_sch_squared",
    "modules/robustness/functional_form.md#treatment-alternatives", "G1",
    "log_av_rel_y", "av_sch", ALL_CONTROLS + ["av_sch_sq"],
    "", "none", df,
    "hetero",
    "Full sample", "full controls + schooling squared",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/treatment/av_sch_squared",
                "treatment": "av_sch + av_sch_sq",
                "notes": "Quadratic schooling term added"})

# Treatment: log schooling
run_spec(
    "rc/form/treatment/log_av_sch",
    "modules/robustness/functional_form.md#treatment-alternatives", "G1",
    "log_av_rel_y", "log_av_sch", ALL_CONTROLS,
    "", "none", df,
    "hetero",
    "Full sample", "full controls (treatment: log schooling)",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/treatment/log_av_sch",
                "treatment": "log_av_sch",
                "notes": "Log-log specification (elasticity interpretation)"})


# ============================================================
# RC: ESTIMATOR VARIANTS
# ============================================================

print("Running estimator variants...")

# WLS weighted by population
run_sm_spec(
    "rc/form/estimator/wls_pop",
    "modules/robustness/functional_form.md#estimator-alternatives", "G1",
    "log_av_rel_y", "av_sch", ALL_CONTROLS, df,
    "Full sample", "full controls (WLS, pop-weighted)",
    estimator_type="wls", weights_var="pop_total",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/estimator/wls_pop",
                "estimator": "wls", "weights": "pop_total",
                "notes": "Population-weighted least squares"})

# Median regression (LAD)
run_sm_spec(
    "rc/form/estimator/median_regression",
    "modules/robustness/functional_form.md#estimator-alternatives", "G1",
    "log_av_rel_y", "av_sch", ALL_CONTROLS, df,
    "Full sample", "full controls (median regression)",
    estimator_type="median",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/estimator/median_regression",
                "estimator": "quantile_regression", "quantile": 0.5,
                "notes": "Median regression (LAD), robust to outliers"})

# Robust regression (M-estimation)
run_sm_spec(
    "rc/form/estimator/robust_regression",
    "modules/robustness/functional_form.md#estimator-alternatives", "G1",
    "log_av_rel_y", "av_sch", ALL_CONTROLS, df,
    "Full sample", "full controls (robust regression)",
    estimator_type="robust",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/estimator/robust_regression",
                "estimator": "robust_rlm",
                "notes": "Robust regression (Huber M-estimator)"})


# ============================================================
# RC: ADDITIONAL SAMPLE VARIANTS
# ============================================================

print("Running additional sample variants...")

# Drop both extremes (deciles 1 and 10)
df_mid = df[(df['pct_nomiss'] > 1) & (df['pct_nomiss'] < 10)].copy()
run_spec(
    "rc/sample/subgroup/drop_extremes",
    "modules/robustness/sample.md#subgroup-analysis", "G1",
    "log_av_rel_y", "av_sch", ALL_CONTROLS,
    "", "none", df_mid,
    "hetero",
    f"Drop deciles 1 and 10, N={len(df_mid)}", "full controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subgroup/drop_extremes", "axis": "subgroup",
                "rule": "drop_extremes"})

# Africa subsample (using low life expectancy as proxy)
df_low_le = df[df['av_T'] < 65].copy()
if len(df_low_le) >= 10:
    run_spec(
        "rc/sample/subgroup/low_life_expectancy",
        "modules/robustness/sample.md#subgroup-analysis", "G1",
        "log_av_rel_y", "av_sch", [c for c in ALL_CONTROLS if c != 'av_T'],
        "", "none", df_low_le,
        "hetero",
        f"Low life expectancy (<65), N={len(df_low_le)}", "controls excl. life exp",
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/subgroup/low_life_expectancy", "axis": "subgroup",
                    "rule": "low_life_expectancy"})

# High life expectancy subsample
df_high_le = df[df['av_T'] >= 65].copy()
if len(df_high_le) >= 10:
    run_spec(
        "rc/sample/subgroup/high_life_expectancy",
        "modules/robustness/sample.md#subgroup-analysis", "G1",
        "log_av_rel_y", "av_sch", [c for c in ALL_CONTROLS if c != 'av_T'],
        "", "none", df_high_le,
        "hetero",
        f"High life expectancy (>=65), N={len(df_high_le)}", "controls excl. life exp",
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/subgroup/high_life_expectancy", "axis": "subgroup",
                    "rule": "high_life_expectancy"})

# Drop high-IMR countries
df_low_imr = df[df['IMR'] < df['IMR'].median()].copy()
run_spec(
    "rc/sample/subgroup/low_imr",
    "modules/robustness/sample.md#subgroup-analysis", "G1",
    "log_av_rel_y", "av_sch", ALL_CONTROLS,
    "", "none", df_low_imr,
    "hetero",
    f"Low IMR (<median), N={len(df_low_imr)}", "full controls",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subgroup/low_imr", "axis": "subgroup",
                "rule": "below_median_imr"})

# Interaction: schooling x life expectancy
df['av_sch_x_av_T'] = df['av_sch'] * df['av_T']
run_spec(
    "rc/form/treatment/interaction_sch_T",
    "modules/robustness/functional_form.md#treatment-alternatives", "G1",
    "log_av_rel_y", "av_sch", ALL_CONTROLS + ["av_sch_x_av_T"],
    "", "none", df,
    "hetero",
    "Full sample", "full controls + schooling x life expectancy interaction",
    axis_block_name="functional_form",
    axis_block={"spec_id": "rc/form/treatment/interaction_sch_T",
                "treatment": "av_sch + av_sch_x_av_T",
                "notes": "Interaction of schooling and life expectancy"})


# ============================================================
# INFERENCE VARIANTS (on baseline specification)
# ============================================================

print("Running inference variants...")

baseline_run_id = f"{PAPER_ID}_run_001"
infer_counter = 0


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
            design={"cross_sectional_ols": design_audit},
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


# Baseline formula for inference variants
baseline_controls_str = " + ".join(ALL_CONTROLS)
baseline_formula = f"log_av_rel_y ~ av_sch + {baseline_controls_str}"

# HC3 robust SEs
run_inference_variant(
    baseline_run_id, "infer/se/hc/hc3",
    "modules/inference/standard_errors.md#heteroskedasticity-robust", "G1",
    baseline_formula, "", df, "av_sch",
    {"CRV1": "pct_str"}, "HC3 (small-sample corrected)")

# Classical OLS SEs
run_inference_variant(
    baseline_run_id, "infer/se/ols",
    "modules/inference/standard_errors.md#homoskedastic", "G1",
    baseline_formula, "", df, "av_sch",
    "iid", "OLS (classical, homoskedastic)")


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
        print(f"\nBaseline coef on av_sch: {base_row['coefficient'].values[0]:.6f}")
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
md_lines.append("# Specification Search Report: 112853-V1")
md_lines.append("")
md_lines.append("**Paper:** Manuelli & Seshadri (2014), \"Human Capital and the Wealth of Nations\", AER 104(11)")
md_lines.append("")
md_lines.append("## Baseline Specification")
md_lines.append("")
md_lines.append("- **Design:** Cross-sectional OLS (cross-country development accounting)")
md_lines.append("- **Outcome:** log(output per worker relative to US) [log_av_rel_y]")
md_lines.append("- **Treatment:** Average years of schooling (Barro-Lee) [av_sch]")
md_lines.append(f"- **Controls:** {len(ALL_CONTROLS)} controls (life expectancy, TFR, education expenditure, relative capital price, infant mortality)")
md_lines.append("- **Fixed effects:** None")
md_lines.append("- **Standard errors:** HC1 robust")
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
    "Controls LOO": successful[successful['spec_id'].str.startswith('rc/controls/loo/')],
    "Controls Sets": successful[successful['spec_id'].str.startswith('rc/controls/sets/')],
    "Controls Progression": successful[successful['spec_id'].str.startswith('rc/controls/progression/')],
    "Controls Subset": successful[successful['spec_id'].str.startswith('rc/controls/subset/')],
    "Sample Trimming": successful[successful['spec_id'].str.startswith('rc/sample/outliers/')],
    "Subgroup Analysis": successful[successful['spec_id'].str.startswith('rc/sample/subgroup/')],
    "Outcome Variants": successful[successful['spec_id'].str.startswith('rc/form/outcome/')],
    "Treatment Variants": successful[successful['spec_id'].str.startswith('rc/form/treatment/')],
    "Estimator Variants": successful[successful['spec_id'].str.startswith('rc/form/estimator/')],
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
    # Focus on same-outcome specs (log_av_rel_y with av_sch treatment)
    same_outcome = successful[
        (successful['outcome_var'] == 'log_av_rel_y') &
        (successful['treatment_var'] == 'av_sch')
    ]
    n_sig_total = (same_outcome['p_value'] < 0.05).sum() if len(same_outcome) > 0 else 0
    pct_sig = n_sig_total / len(same_outcome) * 100 if len(same_outcome) > 0 else 0
    # Check sign consistency among significant specs (interaction terms can flip sign mechanically)
    sig_specs = same_outcome[same_outcome['p_value'] < 0.10]
    sign_consistent_sig = (
        ((sig_specs['coefficient'] > 0).sum() == len(sig_specs)) or
        ((sig_specs['coefficient'] < 0).sum() == len(sig_specs))
    ) if len(sig_specs) > 0 else False
    sign_consistent_all = (
        ((same_outcome['coefficient'] > 0).sum() == len(same_outcome)) or
        ((same_outcome['coefficient'] < 0).sum() == len(same_outcome))
    ) if len(same_outcome) > 0 else False
    median_coef = same_outcome['coefficient'].median() if len(same_outcome) > 0 else np.nan
    sign_word = "positive" if median_coef > 0 else "negative"

    n_pos = (same_outcome['coefficient'] > 0).sum()
    n_neg = (same_outcome['coefficient'] < 0).sum()
    md_lines.append(f"- **Same-outcome specifications:** {len(same_outcome)} (log_av_rel_y ~ av_sch)")
    md_lines.append(f"- **Sign consistency:** {n_pos} positive, {n_neg} negative{'  (all significant specs same sign)' if sign_consistent_sig and not sign_consistent_all else ''}")
    md_lines.append(f"- **Significance stability:** {n_sig_total}/{len(same_outcome)} ({pct_sig:.1f}%) specifications significant at 5%")
    md_lines.append(f"- **Direction:** Median coefficient is {sign_word} ({median_coef:.6f})")

    # Use sign consistency among significant specs for strength assessment
    sign_ok = sign_consistent_sig
    if pct_sig >= 80 and sign_ok:
        strength = "STRONG"
    elif pct_sig >= 50 and sign_ok:
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
