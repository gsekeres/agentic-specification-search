"""
Specification Search Script for Charles, Hurst, & Notowidigdo (2018)
"Housing Booms and Busts, Labor Market Outcomes, and College Attendance"
American Economic Review

Paper ID: 113109-V1

Surface-driven execution:
  - G1: d_emp_18_25_le ~ hp_growth_real_00_06 + controls, instrumented by iv (Saiz)
        Cross-sectional IV at MSA level, weighted by msa_total_emp_all_2000, cluster(statefip)
  - G2: d_any_18_25_a1 ~ housing_demand_shock + controls, instrumented by iv (Saiz)
        Cross-sectional IV at MSA level, weighted by pop_18_33_00, cluster(statefip)

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
import itertools
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, "scripts")
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash, software_block
)

PAPER_ID = "113109-V1"
DATA_DIR = "data/downloads/extracted/113109-V1"
EXTRACT_DIR = f"{DATA_DIR}/chn_housing_booms_and_college"
OUTPUT_DIR = DATA_DIR

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

G1_DESIGN_AUDIT = surface_obj["baseline_groups"][0]["design_audit"]
G2_DESIGN_AUDIT = surface_obj["baseline_groups"][1]["design_audit"]
G1_INFERENCE = surface_obj["baseline_groups"][0]["inference_plan"]["canonical"]
G2_INFERENCE = surface_obj["baseline_groups"][1]["inference_plan"]["canonical"]

# ============================================================
# DATA LOADING AND PREPARATION
# ============================================================

print("=== Loading data ===")

# ---------- G1: Employment outcomes (Table 1) ----------
# Replicate main.do data build:
# 1. analysis_sample_FINAL_v2_main_same.dta (employment outcomes + MSA characteristics)
# 2. merge share_foreign, units_growth, unemp_manuf, routine_share, controls (statefip)
# 3. hp_growth_real_00_06 = deltaP + units_growth
# 4. Weight: msa_total_emp_all_2000, Cluster: statefip

# Use pyreadstat to get numeric metarea codes (pandas read_stata converts to string labels)
import pyreadstat
df_emp_raw, _meta = pyreadstat.read_dta(f"{EXTRACT_DIR}/chn/analysis_sample_FINAL_v2_main_same.dta")
# metarea is now numeric (float64); create string version from labels for human-readable merges
metarea_labels = _meta.variable_value_labels.get('metarea', {})
df_emp_raw['metarea_str'] = df_emp_raw['metarea'].map(metarea_labels).fillna(df_emp_raw['metarea'].astype(str))
# Also keep numeric metarea for merging with helper files that use numeric codes
df_emp_raw['metarea_num'] = df_emp_raw['metarea']

# Merge main_data (has IV, controls, housing_demand_shock, units_growth, deltaP)
# main_data also uses labeled metarea; read with pyreadstat for consistent merge
md_raw, _md_meta = pyreadstat.read_dta(f"{EXTRACT_DIR}/chn/main_data.dta")
md_labels = _md_meta.variable_value_labels.get('metarea', {})
md_raw['metarea_str'] = md_raw['metarea'].map(md_labels).fillna(md_raw['metarea'].astype(str))

md = md_raw.copy()

# Drop overlapping columns from analysis_sample that main_data will provide
# (main_data has the correctly constructed versions from main.do)
md_merge_cols = ['metarea_str', 'iv', 'iv2_log', 't_log', 'units_growth', 'units_growth_alt',
                 'housing_demand_shock', 'housing_demand_shock_alt',
                 'pop_prev', 'female_employed_share_2000', 'college_share_2000',
                 'wgt', 'elasticity', 'reg4']
overlap_cols = [c for c in md_merge_cols if c != 'metarea_str' and c in df_emp_raw.columns]
if overlap_cols:
    df_emp_raw = df_emp_raw.drop(columns=overlap_cols)

df_emp = df_emp_raw.merge(
    md[md_merge_cols],
    on='metarea_str', how='inner')

# Merge share_foreign (uses numeric metarea)
sf_raw, _ = pyreadstat.read_dta(f"{EXTRACT_DIR}/chn/share_foreign.dta")
sf_raw['metarea_num'] = sf_raw['metarea']
df_emp = df_emp.merge(sf_raw[['metarea_num', 'share_foreign_18_55_2000']], on='metarea_num', how='left')

# Merge unemp_manuf (uses numeric metarea)
um_raw, _ = pyreadstat.read_dta(f"{EXTRACT_DIR}/chn/unemp_manuf.dta")
um_raw['metarea_num'] = um_raw['metarea']
df_emp = df_emp.merge(um_raw[['metarea_num', 'unemp_18_55_2000', 'manuf_18_55_2000']], on='metarea_num', how='left')

# Merge routine_share (uses numeric metarea)
rs_raw, _ = pyreadstat.read_dta(f"{EXTRACT_DIR}/chn/routine_share_2000.dta")
rs_raw['metarea_num'] = rs_raw['metarea']
df_emp = df_emp.merge(rs_raw[['metarea_num', 'routine_share_2000']], on='metarea_num', how='left')

# Note: iv2_poly3, iv2_log, t_log etc. are already in the analysis_sample from Stata
# (iv2_poly3 survived the overlap drop since it's not in md_merge_cols).

# Merge controls.dta for statefip (with the correct state assignments from main.do)
ctrl_raw, _ = pyreadstat.read_dta(f"{EXTRACT_DIR}/chn/controls.dta")
ctrl_raw['metarea_num'] = ctrl_raw['metarea']
# Use statefip from controls.dta (already has the fixups from main.do)
df_emp = df_emp.merge(ctrl_raw[['metarea_num', 'statefip']], on='metarea_num', how='left',
                       suffixes=('_orig', ''))
# If statefip came from both, prefer the controls.dta version
if 'statefip_orig' in df_emp.columns:
    df_emp['statefip'] = df_emp['statefip'].fillna(df_emp['statefip_orig'])
    df_emp.drop(columns=['statefip_orig'], inplace=True)

# Reconstruct hp_growth_real_00_06 = deltaP + units_growth (as in main.do line 244/338)
df_emp['hp_growth_real_00_06'] = df_emp['deltaP'] + df_emp['units_growth']

# Region dummies (from the analysis_sample 'region' column)
df_emp['region_str'] = df_emp['region'].astype(str)

# Convert float32 to float64 for precision
for col in df_emp.columns:
    if df_emp[col].dtype == np.float32:
        df_emp[col] = df_emp[col].astype(np.float64)

df_emp['statefip_str'] = df_emp['statefip'].astype(int).astype(str)

# G1 baseline sample: require non-missing key variables
G1_BASELINE_CONTROLS = ['college_share_2000', 'female_employed_share_2000', 'pop_prev', 'share_foreign_18_55_2000']
G1_EXTRA_CONTROLS = ['manuf_18_55_2000', 'routine_share_2000', 'unemp_18_55_2000']

g1_required = ['d_emp_18_25_le', 'hp_growth_real_00_06', 'iv',
               'msa_total_emp_all_2000', 'statefip_str'] + G1_BASELINE_CONTROLS
df_g1 = df_emp.dropna(subset=g1_required).copy()
print(f"G1 employment sample: {len(df_g1)} MSAs, {df_g1['statefip_str'].nunique()} states")

# Create region FE dummies for robustness specs
region_dummies = pd.get_dummies(df_g1['region'], prefix='region', drop_first=True).astype(float)
for c in region_dummies.columns:
    df_g1[c] = region_dummies[c].values

REGION_DUMMY_COLS = list(region_dummies.columns)


# ---------- G2: Education outcomes (Table 3) ----------
# Replicate table3_educ.do data build

df_edu_raw, _edu_meta = pyreadstat.read_dta(f"{EXTRACT_DIR}/chn/msa_education_2000_2013_same.dta")
edu_labels = _edu_meta.variable_value_labels.get('metarea', {})
df_edu_raw['metarea_str'] = df_edu_raw['metarea'].map(edu_labels).fillna(df_edu_raw['metarea'].astype(str))
df_edu_raw['metarea_num'] = df_edu_raw['metarea']

# Merge main_data
df_edu = df_edu_raw.merge(
    md[['metarea_str', 'iv', 'iv2_log', 't_log', 'housing_demand_shock',
        'pop_prev', 'female_employed_share_2000', 'college_share_2000',
        'wgt', 'elasticity', 'units_growth', 'deltaP']],
    on='metarea_str', how='inner')

# Merge share_foreign (numeric metarea)
df_edu = df_edu.merge(sf_raw[['metarea_num', 'share_foreign_18_55_2000']], on='metarea_num', how='inner')

# Merge controls.dta for statefip (numeric metarea)
df_edu = df_edu.merge(ctrl_raw[['metarea_num', 'statefip']], on='metarea_num', how='inner')

# Merge unemp_manuf and routine_share for robustness (numeric metarea)
df_edu = df_edu.merge(um_raw[['metarea_num', 'unemp_18_55_2000', 'manuf_18_55_2000']], on='metarea_num', how='left')
df_edu = df_edu.merge(rs_raw[['metarea_num', 'routine_share_2000']], on='metarea_num', how='left')

# Merge iv2_poly3 from analysis_sample (numeric metarea)
iv2p3_lookup = df_emp_raw[['metarea_num', 'iv2_poly3']].drop_duplicates(subset='metarea_num')
df_edu = df_edu.merge(iv2p3_lookup, on='metarea_num', how='left')

# Filter to non-missing housing demand shock
df_edu = df_edu[df_edu['housing_demand_shock'].notna()].copy()

# Create education outcome variables (following table3_educ.do)
for yr in ['2000', '2007']:
    # Both sexes (type='_' in Stata)
    df_edu[f'any_18_25_{yr}'] = df_edu[f'any_college_22_25_{yr}'] + df_edu[f'any_college_18_21_{yr}']
    df_edu[f'bachelor_18_25_{yr}'] = df_edu[f'bachelor_22_25_{yr}'] + df_edu[f'bachelor_18_21_{yr}']
    df_edu[f'pop_18_25_{yr}'] = df_edu[f'pop_22_25_{yr}'] + df_edu[f'pop_18_21_{yr}']
    df_edu[f'any_26_33_{yr}'] = df_edu[f'any_college_26_29_{yr}'] + df_edu[f'any_college_30_33_{yr}']
    df_edu[f'bachelor_26_33_{yr}'] = df_edu[f'bachelor_26_29_{yr}'] + df_edu[f'bachelor_30_33_{yr}']
    df_edu[f'pop_26_33_{yr}'] = df_edu[f'pop_26_29_{yr}'] + df_edu[f'pop_30_33_{yr}']
    # Associate degree
    df_edu[f'assoc_18_25_{yr}'] = df_edu[f'associate_22_25_{yr}'] + df_edu[f'associate_18_21_{yr}']

# Create long-difference outcomes (2000 -> 2007)
df_edu['d_any_18_25_a1'] = df_edu['any_18_25_2007'] / df_edu['pop_18_25_2007'] - \
                             df_edu['any_18_25_2000'] / df_edu['pop_18_25_2000']
df_edu['d_bachelor_18_25_a1'] = df_edu['bachelor_18_25_2007'] / df_edu['pop_18_25_2007'] - \
                                  df_edu['bachelor_18_25_2000'] / df_edu['pop_18_25_2000']
df_edu['d_any_26_33_a1'] = df_edu['any_26_33_2007'] / df_edu['pop_26_33_2007'] - \
                              df_edu['any_26_33_2000'] / df_edu['pop_26_33_2000']
df_edu['d_bachelor_26_33_a1'] = df_edu['bachelor_26_33_2007'] / df_edu['pop_26_33_2007'] - \
                                   df_edu['bachelor_26_33_2000'] / df_edu['pop_26_33_2000']
# Associate degree outcome
df_edu['d_assoc_18_25_a1'] = df_edu['assoc_18_25_2007'] / df_edu['pop_18_25_2007'] - \
                                df_edu['assoc_18_25_2000'] / df_edu['pop_18_25_2000']

# Weight for education: wgt = exp(pop_prev) [as in table3_educ.do line 125]
df_edu['wgt'] = np.exp(df_edu['pop_prev'])

# Convert float32 to float64
for col in df_edu.columns:
    if df_edu[col].dtype == np.float32:
        df_edu[col] = df_edu[col].astype(np.float64)

df_edu['statefip_str'] = df_edu['statefip'].astype(int).astype(str)

# G2 baseline sample
g2_required = ['d_any_18_25_a1', 'housing_demand_shock', 'iv',
               'wgt', 'statefip_str'] + G1_BASELINE_CONTROLS
df_g2 = df_edu.dropna(subset=g2_required).copy()
print(f"G2 education sample: {len(df_g2)} MSAs, {df_g2['statefip_str'].nunique()} states")

# Region dummies for G2
region_dummies_g2 = pd.get_dummies(df_g2['statefip'].map(
    dict(zip(df_g1['statefip'], df_g1['region'])) if 'region' in df_g1.columns else {}),
    prefix='region', drop_first=True).astype(float)
# Use a simpler approach: merge region from main_data
if 'region' not in df_g2.columns:
    # Get region from the xwalk or analysis sample
    xw = pd.read_stata(f"{EXTRACT_DIR}/chn/xwalk_metarea_to_state_and_region.dta")
    xw['metarea_str'] = xw['metarea'].astype(str)
    df_g2 = df_g2.merge(xw[['metarea_str', 'region']], on='metarea_str', how='left')

# Create region dummies for G2
if 'region' in df_g2.columns and df_g2['region'].notna().sum() > 0:
    rd2 = pd.get_dummies(df_g2['region'], prefix='region', drop_first=True).astype(float)
    for c in rd2.columns:
        df_g2[c] = rd2[c].values
    REGION_DUMMY_COLS_G2 = list(rd2.columns)
else:
    REGION_DUMMY_COLS_G2 = []


# ============================================================
# Accumulators
# ============================================================

results = []
inference_results = []
spec_run_counter = 0


# ============================================================
# HELPER: run_iv (2SLS via pyfixest)
# ============================================================

def run_iv(spec_id, spec_tree_path, baseline_group_id,
           outcome_var, treatment_var, instrument_var,
           controls, data, vcov_spec,
           weight_var, sample_desc, controls_desc, cluster_var_label,
           design_audit, inference_canonical,
           axis_block_name=None, axis_block=None, notes=""):
    """Run a 2SLS IV regression (cross-sectional, no FE) using pyfixest."""
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        ctrl_str = " + ".join(controls) if controls else "1"
        # pyfixest IV syntax: Y ~ exog_controls | FE | endog ~ instrument
        # No FE: use 0
        formula = f"{outcome_var} ~ {ctrl_str} | 0 | {treatment_var} ~ {instrument_var}"

        kwargs = dict(data=data, vcov=vcov_spec)
        if weight_var:
            kwargs['weights'] = weight_var

        m = pf.feols(formula, **kwargs)
        coef_val = float(m.coef().get(treatment_var, np.nan))
        se_val = float(m.se().get(treatment_var, np.nan))
        pval = float(m.pvalue().get(treatment_var, np.nan))

        try:
            ci = m.confint()
            ci_lower = float(ci.loc[treatment_var, ci.columns[0]])
            ci_upper = float(ci.loc[treatment_var, ci.columns[1]])
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
            inference=inference_canonical,
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"instrumental_variables": design_audit},
            axis_block_name=axis_block_name,
            axis_block=axis_block,
            notes=notes if notes else None,
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
            "cluster_var": cluster_var_label,
            "run_success": 1,
            "run_error": "",
        })
        return run_id, coef_val, se_val, pval, nobs

    except Exception as e:
        err_msg = str(e)[:240]
        err_det = error_details_from_exception(e, stage="estimation")
        payload = make_failure_payload(
            error=err_msg,
            error_details=err_det,
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
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
            "cluster_var": cluster_var_label,
            "run_success": 0,
            "run_error": err_msg,
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


# ============================================================
# HELPER: run_liml (via linearmodels)
# ============================================================

def run_liml(spec_id, spec_tree_path, baseline_group_id,
             outcome_var, treatment_var, instrument_var,
             controls, data, cluster_var_name, weight_var,
             sample_desc, controls_desc, cluster_var_label,
             design_audit, inference_canonical,
             axis_block_name=None, axis_block=None, notes=""):
    """Run a LIML IV regression using linearmodels."""
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        from linearmodels.iv import IVLIML
        import statsmodels.api as sm

        work = data.copy()
        all_vars = [outcome_var, treatment_var, instrument_var] + controls
        if cluster_var_name:
            all_vars.append(cluster_var_name)
        if weight_var:
            all_vars.append(weight_var)
        work = work.dropna(subset=all_vars)

        dep = work[outcome_var]
        exog = sm.add_constant(work[controls]) if controls else sm.add_constant(pd.DataFrame(index=work.index))
        endog = work[[treatment_var]]
        instruments = work[[instrument_var]]

        if weight_var:
            w = work[weight_var]
            # IVLIML does not directly support aweights like pyfixest;
            # we use WLS-style pre-multiplication
            wt_sqrt = np.sqrt(w / w.mean())
            dep = dep * wt_sqrt
            exog = exog.multiply(wt_sqrt, axis=0)
            endog = endog.multiply(wt_sqrt, axis=0)
            instruments = instruments.multiply(wt_sqrt, axis=0)

        model = IVLIML(dep, exog, endog, instruments)
        if cluster_var_name:
            fit = model.fit(cov_type='clustered', clusters=work[cluster_var_name])
        else:
            fit = model.fit(cov_type='robust')

        coef_val = float(fit.params[treatment_var])
        se_val = float(fit.std_errors[treatment_var])
        pval = float(fit.pvalues[treatment_var])
        ci = fit.conf_int()
        ci_lower = float(ci.loc[treatment_var, 'lower'])
        ci_upper = float(ci.loc[treatment_var, 'upper'])
        nobs = int(fit.nobs)
        try:
            r2 = float(fit.r2)
        except:
            r2 = np.nan

        all_coefs = {k: float(v) for k, v in fit.params.items()
                     if not k.startswith('fe_')}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference=inference_canonical,
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"instrumental_variables": design_audit},
            axis_block_name=axis_block_name,
            axis_block=axis_block,
            notes=notes if notes else None,
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
            "cluster_var": cluster_var_label,
            "run_success": 1,
            "run_error": "",
        })
        return run_id, coef_val, se_val, pval, nobs

    except Exception as e:
        err_msg = str(e)[:240]
        err_det = error_details_from_exception(e, stage="liml_estimation")
        payload = make_failure_payload(
            error=err_msg,
            error_details=err_det,
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
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
            "cluster_var": cluster_var_label,
            "run_success": 0,
            "run_error": err_msg,
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


# ============================================================
# G1 BASELINE: Table 1 IV - Employment
# ============================================================

print("\n=== G1: Employment Specifications ===")

G1_TREATMENT = "hp_growth_real_00_06"
G1_INSTRUMENT = "iv"
G1_OUTCOME = "d_emp_18_25_le"
G1_WEIGHT = "msa_total_emp_all_2000"
G1_VCOV = {"CRV1": "statefip_str"}

baseline_g1_run_id, *_ = run_iv(
    "baseline__table1_iv_employment",
    "specification_tree/methods/instrumental_variables.md",
    "G1", G1_OUTCOME, G1_TREATMENT, G1_INSTRUMENT,
    G1_BASELINE_CONTROLS, df_g1, G1_VCOV,
    G1_WEIGHT, "Full MSA sample (N=275)", "baseline controls (4)", "cluster(statefip)",
    G1_DESIGN_AUDIT, G1_INFERENCE)
print(f"  Baseline G1: coef={_[0]:.6f}, se={_[1]:.6f}, p={_[2]:.6f}, N={_[3]}")


# ============================================================
# G1: DESIGN VARIANTS
# ============================================================

print("Running G1 design variants...")

# LIML estimator
run_liml(
    "design/instrumental_variables/estimator/liml",
    "specification_tree/methods/instrumental_variables.md#liml",
    "G1", G1_OUTCOME, G1_TREATMENT, G1_INSTRUMENT,
    G1_BASELINE_CONTROLS, df_g1, "statefip_str", G1_WEIGHT,
    "Full MSA sample", "baseline controls (LIML)", "cluster(statefip)",
    G1_DESIGN_AUDIT, G1_INFERENCE,
    axis_block_name="estimation",
    axis_block={"estimator": "liml", "notes": "LIML for robustness to weak instruments"})


# ============================================================
# G1: CONTROLS LOO (Leave-one-out)
# ============================================================

print("Running G1 controls LOO...")

loo_map = {
    "drop_college_share": "college_share_2000",
    "drop_female_employed": "female_employed_share_2000",
    "drop_pop_prev": "pop_prev",
    "drop_share_foreign": "share_foreign_18_55_2000",
}

for loo_name, drop_var in loo_map.items():
    loo_controls = [c for c in G1_BASELINE_CONTROLS if c != drop_var]
    run_iv(
        f"rc/controls/loo/{loo_name}",
        "specification_tree/modules/robustness/controls.md#leave-one-out",
        "G1", G1_OUTCOME, G1_TREATMENT, G1_INSTRUMENT,
        loo_controls, df_g1, G1_VCOV,
        G1_WEIGHT, "Full MSA sample", f"baseline minus {drop_var}", "cluster(statefip)",
        G1_DESIGN_AUDIT, G1_INFERENCE,
        axis_block_name="controls",
        axis_block={"family": "loo", "dropped": [drop_var]})


# ============================================================
# G1: CONTROLS SETS
# ============================================================

print("Running G1 controls sets...")

# No controls
run_iv(
    "rc/controls/sets/no_controls",
    "specification_tree/modules/robustness/controls.md#controls-sets",
    "G1", G1_OUTCOME, G1_TREATMENT, G1_INSTRUMENT,
    [], df_g1, G1_VCOV,
    G1_WEIGHT, "Full MSA sample", "no controls", "cluster(statefip)",
    G1_DESIGN_AUDIT, G1_INFERENCE,
    axis_block_name="controls",
    axis_block={"family": "sets", "set_name": "no_controls"})

# Baseline + region FE (as control dummies)
run_iv(
    "rc/controls/sets/baseline_plus_region_fe",
    "specification_tree/modules/robustness/controls.md#controls-sets",
    "G1", G1_OUTCOME, G1_TREATMENT, G1_INSTRUMENT,
    G1_BASELINE_CONTROLS + REGION_DUMMY_COLS, df_g1, G1_VCOV,
    G1_WEIGHT, "Full MSA sample", "baseline + region dummies", "cluster(statefip)",
    G1_DESIGN_AUDIT, G1_INFERENCE,
    axis_block_name="controls",
    axis_block={"family": "sets", "set_name": "baseline_plus_region_fe"})

# Baseline + manuf
g1_extra_sample = df_g1.dropna(subset=['manuf_18_55_2000']).copy()
run_iv(
    "rc/controls/sets/baseline_plus_manuf",
    "specification_tree/modules/robustness/controls.md#controls-sets",
    "G1", G1_OUTCOME, G1_TREATMENT, G1_INSTRUMENT,
    G1_BASELINE_CONTROLS + ['manuf_18_55_2000'], g1_extra_sample, G1_VCOV,
    G1_WEIGHT, "MSAs with manuf data", "baseline + manuf_18_55_2000", "cluster(statefip)",
    G1_DESIGN_AUDIT, G1_INFERENCE,
    axis_block_name="controls",
    axis_block={"family": "sets", "set_name": "baseline_plus_manuf"})

# Baseline + routine
g1_routine_sample = df_g1.dropna(subset=['routine_share_2000']).copy()
run_iv(
    "rc/controls/sets/baseline_plus_routine",
    "specification_tree/modules/robustness/controls.md#controls-sets",
    "G1", G1_OUTCOME, G1_TREATMENT, G1_INSTRUMENT,
    G1_BASELINE_CONTROLS + ['routine_share_2000'], g1_routine_sample, G1_VCOV,
    G1_WEIGHT, "MSAs with routine data", "baseline + routine_share_2000", "cluster(statefip)",
    G1_DESIGN_AUDIT, G1_INFERENCE,
    axis_block_name="controls",
    axis_block={"family": "sets", "set_name": "baseline_plus_routine"})

# Baseline + unemp
g1_unemp_sample = df_g1.dropna(subset=['unemp_18_55_2000']).copy()
run_iv(
    "rc/controls/sets/baseline_plus_unemp",
    "specification_tree/modules/robustness/controls.md#controls-sets",
    "G1", G1_OUTCOME, G1_TREATMENT, G1_INSTRUMENT,
    G1_BASELINE_CONTROLS + ['unemp_18_55_2000'], g1_unemp_sample, G1_VCOV,
    G1_WEIGHT, "MSAs with unemp data", "baseline + unemp_18_55_2000", "cluster(statefip)",
    G1_DESIGN_AUDIT, G1_INFERENCE,
    axis_block_name="controls",
    axis_block={"family": "sets", "set_name": "baseline_plus_unemp"})

# Baseline + all extra (manuf + routine + unemp)
g1_all_extra_sample = df_g1.dropna(subset=G1_EXTRA_CONTROLS).copy()
run_iv(
    "rc/controls/sets/baseline_plus_all_extra",
    "specification_tree/modules/robustness/controls.md#controls-sets",
    "G1", G1_OUTCOME, G1_TREATMENT, G1_INSTRUMENT,
    G1_BASELINE_CONTROLS + G1_EXTRA_CONTROLS, g1_all_extra_sample, G1_VCOV,
    G1_WEIGHT, "MSAs with all extra controls", "baseline + manuf + routine + unemp", "cluster(statefip)",
    G1_DESIGN_AUDIT, G1_INFERENCE,
    axis_block_name="controls",
    axis_block={"family": "sets", "set_name": "baseline_plus_all_extra"})

# Region FE only
run_iv(
    "rc/controls/sets/region_fe_only",
    "specification_tree/modules/robustness/controls.md#controls-sets",
    "G1", G1_OUTCOME, G1_TREATMENT, G1_INSTRUMENT,
    REGION_DUMMY_COLS, df_g1, G1_VCOV,
    G1_WEIGHT, "Full MSA sample", "region dummies only", "cluster(statefip)",
    G1_DESIGN_AUDIT, G1_INFERENCE,
    axis_block_name="controls",
    axis_block={"family": "sets", "set_name": "region_fe_only"})

# Full with region FE (baseline + all extra + region)
run_iv(
    "rc/controls/sets/full_with_region_fe",
    "specification_tree/modules/robustness/controls.md#controls-sets",
    "G1", G1_OUTCOME, G1_TREATMENT, G1_INSTRUMENT,
    G1_BASELINE_CONTROLS + G1_EXTRA_CONTROLS + REGION_DUMMY_COLS, g1_all_extra_sample, G1_VCOV,
    G1_WEIGHT, "MSAs with all extra controls", "baseline + all extra + region dummies", "cluster(statefip)",
    G1_DESIGN_AUDIT, G1_INFERENCE,
    axis_block_name="controls",
    axis_block={"family": "sets", "set_name": "full_with_region_fe"})


# ============================================================
# G1: RANDOM CONTROL SUBSETS
# ============================================================

print("Running G1 random control subsets...")

np.random.seed(113109)
all_possible_controls = G1_BASELINE_CONTROLS + G1_EXTRA_CONTROLS
# Generate random subsets of size 2-5
random_subset_count = 0
for _ in range(20):
    k = np.random.randint(2, min(6, len(all_possible_controls) + 1))
    subset = sorted(np.random.choice(all_possible_controls, size=k, replace=False).tolist())
    subset_name = "_".join([c[:8] for c in subset])
    random_subset_count += 1
    # Ensure sample has non-missing for selected controls
    sub_sample = df_g1.dropna(subset=subset).copy()
    if len(sub_sample) < 50:
        continue
    run_iv(
        f"rc/controls/subset/random_{random_subset_count:02d}",
        "specification_tree/modules/robustness/controls.md#random-subsets",
        "G1", G1_OUTCOME, G1_TREATMENT, G1_INSTRUMENT,
        subset, sub_sample, G1_VCOV,
        G1_WEIGHT, f"MSAs with non-missing subset controls (N={len(sub_sample)})",
        f"random subset: {', '.join(subset)}", "cluster(statefip)",
        G1_DESIGN_AUDIT, G1_INFERENCE,
        axis_block_name="controls",
        axis_block={"family": "subset", "controls": subset})


# ============================================================
# G1: SAMPLE VARIANTS
# ============================================================

print("Running G1 sample variants...")

# Trim outcome 1%/99%
q_low_y = df_g1[G1_OUTCOME].quantile(0.01)
q_high_y = df_g1[G1_OUTCOME].quantile(0.99)
df_trim_y = df_g1[(df_g1[G1_OUTCOME] >= q_low_y) & (df_g1[G1_OUTCOME] <= q_high_y)].copy()
run_iv(
    "rc/sample/outliers/trim_y_1_99",
    "specification_tree/modules/robustness/sample.md#trimming",
    "G1", G1_OUTCOME, G1_TREATMENT, G1_INSTRUMENT,
    G1_BASELINE_CONTROLS, df_trim_y, G1_VCOV,
    G1_WEIGHT, f"Trim outcome 1/99 (N={len(df_trim_y)})", "baseline controls", "cluster(statefip)",
    G1_DESIGN_AUDIT, G1_INFERENCE,
    axis_block_name="sample",
    axis_block={"family": "outliers", "trim": "y_1_99"})

# Trim outcome 5%/95%
q_low_y5 = df_g1[G1_OUTCOME].quantile(0.05)
q_high_y5 = df_g1[G1_OUTCOME].quantile(0.95)
df_trim_y5 = df_g1[(df_g1[G1_OUTCOME] >= q_low_y5) & (df_g1[G1_OUTCOME] <= q_high_y5)].copy()
run_iv(
    "rc/sample/outliers/trim_y_5_95",
    "specification_tree/modules/robustness/sample.md#trimming",
    "G1", G1_OUTCOME, G1_TREATMENT, G1_INSTRUMENT,
    G1_BASELINE_CONTROLS, df_trim_y5, G1_VCOV,
    G1_WEIGHT, f"Trim outcome 5/95 (N={len(df_trim_y5)})", "baseline controls", "cluster(statefip)",
    G1_DESIGN_AUDIT, G1_INFERENCE,
    axis_block_name="sample",
    axis_block={"family": "outliers", "trim": "y_5_95"})

# Trim treatment 1%/99%
q_low_x = df_g1[G1_TREATMENT].quantile(0.01)
q_high_x = df_g1[G1_TREATMENT].quantile(0.99)
df_trim_x = df_g1[(df_g1[G1_TREATMENT] >= q_low_x) & (df_g1[G1_TREATMENT] <= q_high_x)].copy()
run_iv(
    "rc/sample/outliers/trim_x_1_99",
    "specification_tree/modules/robustness/sample.md#trimming",
    "G1", G1_OUTCOME, G1_TREATMENT, G1_INSTRUMENT,
    G1_BASELINE_CONTROLS, df_trim_x, G1_VCOV,
    G1_WEIGHT, f"Trim treatment 1/99 (N={len(df_trim_x)})", "baseline controls", "cluster(statefip)",
    G1_DESIGN_AUDIT, G1_INFERENCE,
    axis_block_name="sample",
    axis_block={"family": "outliers", "trim": "x_1_99"})

# Drop extreme elasticity MSAs (top/bottom 10% of Saiz elasticity)
if df_g1['elasticity'].notna().sum() > 0:
    q10_e = df_g1['elasticity'].quantile(0.10)
    q90_e = df_g1['elasticity'].quantile(0.90)
    df_no_extreme = df_g1[(df_g1['elasticity'] >= q10_e) & (df_g1['elasticity'] <= q90_e)].copy()
    run_iv(
        "rc/sample/subset/drop_extreme_elasticity",
        "specification_tree/modules/robustness/sample.md#subsets",
        "G1", G1_OUTCOME, G1_TREATMENT, G1_INSTRUMENT,
        G1_BASELINE_CONTROLS, df_no_extreme, G1_VCOV,
        G1_WEIGHT, f"Drop extreme elasticity (N={len(df_no_extreme)})", "baseline controls", "cluster(statefip)",
        G1_DESIGN_AUDIT, G1_INFERENCE,
        axis_block_name="sample",
        axis_block={"family": "subset", "rule": "drop_extreme_elasticity_10_90"})

# Drop small MSAs (bottom quartile by employment)
q25_emp = df_g1['msa_total_emp_all_2000'].quantile(0.25)
df_large = df_g1[df_g1['msa_total_emp_all_2000'] > q25_emp].copy()
run_iv(
    "rc/sample/subset/drop_small_msas",
    "specification_tree/modules/robustness/sample.md#subsets",
    "G1", G1_OUTCOME, G1_TREATMENT, G1_INSTRUMENT,
    G1_BASELINE_CONTROLS, df_large, G1_VCOV,
    G1_WEIGHT, f"Drop small MSAs (N={len(df_large)})", "baseline controls", "cluster(statefip)",
    G1_DESIGN_AUDIT, G1_INFERENCE,
    axis_block_name="sample",
    axis_block={"family": "subset", "rule": "drop_bottom_quartile_employment"})


# ============================================================
# G1: TREATMENT FORM VARIANTS
# ============================================================

print("Running G1 treatment form variants...")

# Use deltaP only (price growth only, not deltaP + units_growth)
df_g1['deltaP_only'] = df_g1['deltaP']
run_iv(
    "rc/form/treatment/deltaP_only",
    "specification_tree/modules/robustness/functional_form.md#treatment",
    "G1", G1_OUTCOME, "deltaP_only", G1_INSTRUMENT,
    G1_BASELINE_CONTROLS, df_g1, G1_VCOV,
    G1_WEIGHT, "Full MSA sample", "baseline controls, deltaP only as treatment", "cluster(statefip)",
    G1_DESIGN_AUDIT, G1_INFERENCE,
    axis_block_name="functional_form",
    axis_block={"family": "treatment", "treatment": "deltaP_only"})

# Use deltaP scaled by std dev
df_g1['deltaP_scaled'] = df_g1['deltaP'] / df_g1['deltaP'].std()
run_iv(
    "rc/form/treatment/deltaP_scaled",
    "specification_tree/modules/robustness/functional_form.md#treatment",
    "G1", G1_OUTCOME, "deltaP_scaled", G1_INSTRUMENT,
    G1_BASELINE_CONTROLS, df_g1, G1_VCOV,
    G1_WEIGHT, "Full MSA sample", "baseline controls, deltaP scaled by SD", "cluster(statefip)",
    G1_DESIGN_AUDIT, G1_INFERENCE,
    axis_block_name="functional_form",
    axis_block={"family": "treatment", "treatment": "deltaP_scaled"})


# ============================================================
# G1: INSTRUMENT VARIANTS
# ============================================================

print("Running G1 instrument variants...")

# iv_sig: use sign of iv (binary instrument)
df_g1['iv_sig'] = (df_g1['iv'] > df_g1['iv'].median()).astype(float)
run_iv(
    "rc/data/instrument/iv_sig",
    "specification_tree/modules/robustness/instrument.md#alternatives",
    "G1", G1_OUTCOME, G1_TREATMENT, "iv_sig",
    G1_BASELINE_CONTROLS, df_g1, G1_VCOV,
    G1_WEIGHT, "Full MSA sample", "baseline controls, binary IV (above median)", "cluster(statefip)",
    G1_DESIGN_AUDIT, G1_INFERENCE,
    axis_block_name="data_construction",
    axis_block={"family": "instrument", "instrument": "iv_sig (above median)"})

# iv_sig2: binary at 2/3 percentile
p67 = df_g1['iv'].quantile(0.667)
df_g1['iv_sig2'] = (df_g1['iv'] > p67).astype(float)
run_iv(
    "rc/data/instrument/iv_sig2",
    "specification_tree/modules/robustness/instrument.md#alternatives",
    "G1", G1_OUTCOME, G1_TREATMENT, "iv_sig2",
    G1_BASELINE_CONTROLS, df_g1, G1_VCOV,
    G1_WEIGHT, "Full MSA sample", "baseline controls, binary IV (top 1/3)", "cluster(statefip)",
    G1_DESIGN_AUDIT, G1_INFERENCE,
    axis_block_name="data_construction",
    axis_block={"family": "instrument", "instrument": "iv_sig2 (top 1/3)"})

# iv2_poly3: Polynomial in Saiz elasticity from FHFA
df_g1_iv2p3 = df_g1.dropna(subset=['iv2_poly3']).copy()
if len(df_g1_iv2p3) >= 50:
    run_iv(
        "rc/data/instrument/iv2_poly3",
        "specification_tree/modules/robustness/instrument.md#alternatives",
        "G1", G1_OUTCOME, G1_TREATMENT, "iv2_poly3",
        G1_BASELINE_CONTROLS, df_g1_iv2p3, G1_VCOV,
        G1_WEIGHT, f"MSAs with iv2_poly3 (N={len(df_g1_iv2p3)})",
        "baseline controls, polynomial IV", "cluster(statefip)",
        G1_DESIGN_AUDIT, G1_INFERENCE,
        axis_block_name="data_construction",
        axis_block={"family": "instrument", "instrument": "iv2_poly3"})

# iv2_log: log version of IV
df_g1_iv2log = df_g1.dropna(subset=['iv2_log']).copy()
if len(df_g1_iv2log) >= 50:
    run_iv(
        "rc/data/instrument/price_rent_ratio",
        "specification_tree/modules/robustness/instrument.md#alternatives",
        "G1", G1_OUTCOME, G1_TREATMENT, "iv2_log",
        G1_BASELINE_CONTROLS, df_g1_iv2log, G1_VCOV,
        G1_WEIGHT, f"MSAs with iv2_log (N={len(df_g1_iv2log)})",
        "baseline controls, log IV", "cluster(statefip)",
        G1_DESIGN_AUDIT, G1_INFERENCE,
        axis_block_name="data_construction",
        axis_block={"family": "instrument", "instrument": "iv2_log"})


# ============================================================
# G1: WEIGHT VARIANTS
# ============================================================

print("Running G1 weight variants...")

# Unweighted
run_iv(
    "rc/weights/unweighted",
    "specification_tree/modules/robustness/weights.md#unweighted",
    "G1", G1_OUTCOME, G1_TREATMENT, G1_INSTRUMENT,
    G1_BASELINE_CONTROLS, df_g1, G1_VCOV,
    None, "Full MSA sample (unweighted)", "baseline controls", "cluster(statefip)",
    G1_DESIGN_AUDIT, G1_INFERENCE,
    axis_block_name="weights",
    axis_block={"family": "unweighted"})

# Population weights instead of employment weights
df_g1['pop_weight'] = np.exp(df_g1['pop_prev'])
run_iv(
    "rc/weights/pop_weights",
    "specification_tree/modules/robustness/weights.md#alternatives",
    "G1", G1_OUTCOME, G1_TREATMENT, G1_INSTRUMENT,
    G1_BASELINE_CONTROLS, df_g1, G1_VCOV,
    "pop_weight", "Full MSA sample (pop weights)", "baseline controls", "cluster(statefip)",
    G1_DESIGN_AUDIT, G1_INFERENCE,
    axis_block_name="weights",
    axis_block={"family": "pop_weights"})


# ============================================================
# G1: JOINT VARIANTS (controls + instrument)
# ============================================================

print("Running G1 joint variants...")

# No controls + binary IV
run_iv(
    "rc/joint/controls_instrument/no_ctrl_iv_sig",
    "specification_tree/modules/robustness/joint.md",
    "G1", G1_OUTCOME, G1_TREATMENT, "iv_sig",
    [], df_g1, G1_VCOV,
    G1_WEIGHT, "Full MSA sample", "no controls, binary IV", "cluster(statefip)",
    G1_DESIGN_AUDIT, G1_INFERENCE,
    axis_block_name="joint",
    axis_block={"controls": "none", "instrument": "iv_sig"})

# Full controls + binary IV
run_iv(
    "rc/joint/controls_instrument/full_ctrl_iv_sig",
    "specification_tree/modules/robustness/joint.md",
    "G1", G1_OUTCOME, G1_TREATMENT, "iv_sig",
    G1_BASELINE_CONTROLS + REGION_DUMMY_COLS, df_g1, G1_VCOV,
    G1_WEIGHT, "Full MSA sample", "baseline+region, binary IV", "cluster(statefip)",
    G1_DESIGN_AUDIT, G1_INFERENCE,
    axis_block_name="joint",
    axis_block={"controls": "baseline_plus_region", "instrument": "iv_sig"})

# Baseline controls + iv2_poly3
if len(df_g1_iv2p3) >= 50:
    run_iv(
        "rc/joint/controls_instrument/baseline_iv2_poly3",
        "specification_tree/modules/robustness/joint.md",
        "G1", G1_OUTCOME, G1_TREATMENT, "iv2_poly3",
        G1_BASELINE_CONTROLS + REGION_DUMMY_COLS, df_g1_iv2p3, G1_VCOV,
        G1_WEIGHT, f"MSAs with iv2_poly3 (N={len(df_g1_iv2p3)})",
        "baseline+region, polynomial IV", "cluster(statefip)",
        G1_DESIGN_AUDIT, G1_INFERENCE,
        axis_block_name="joint",
        axis_block={"controls": "baseline_plus_region", "instrument": "iv2_poly3"})

# Joint: controls + sample
# Trim outcome + full controls
run_iv(
    "rc/joint/controls_sample/trim_y_full_ctrl",
    "specification_tree/modules/robustness/joint.md",
    "G1", G1_OUTCOME, G1_TREATMENT, G1_INSTRUMENT,
    G1_BASELINE_CONTROLS + REGION_DUMMY_COLS,
    df_trim_y, G1_VCOV,
    G1_WEIGHT, f"Trim outcome 1/99, full controls (N={len(df_trim_y)})",
    "baseline+region, trimmed", "cluster(statefip)",
    G1_DESIGN_AUDIT, G1_INFERENCE,
    axis_block_name="joint",
    axis_block={"controls": "baseline_plus_region", "sample": "trim_y_1_99"})

# Drop small MSAs + no controls
run_iv(
    "rc/joint/controls_sample/large_msas_no_ctrl",
    "specification_tree/modules/robustness/joint.md",
    "G1", G1_OUTCOME, G1_TREATMENT, G1_INSTRUMENT,
    [], df_large, G1_VCOV,
    G1_WEIGHT, f"Large MSAs, no controls (N={len(df_large)})",
    "no controls, large MSAs", "cluster(statefip)",
    G1_DESIGN_AUDIT, G1_INFERENCE,
    axis_block_name="joint",
    axis_block={"controls": "none", "sample": "large_msas"})


# ============================================================
# G2 BASELINE: Table 3 IV - College Enrollment (any college, 18-25)
# ============================================================

print("\n=== G2: Education Specifications ===")

G2_TREATMENT = "housing_demand_shock"
G2_INSTRUMENT = "iv"
G2_OUTCOME_ANY = "d_any_18_25_a1"
G2_OUTCOME_BACH = "d_bachelor_18_25_a1"
G2_WEIGHT = "wgt"
G2_VCOV = {"CRV1": "statefip_str"}

# Baseline: any college 18-25
baseline_g2_run_id, *_ = run_iv(
    "baseline__table3_iv_any_college_18_25",
    "specification_tree/methods/instrumental_variables.md",
    "G2", G2_OUTCOME_ANY, G2_TREATMENT, G2_INSTRUMENT,
    G1_BASELINE_CONTROLS, df_g2, G2_VCOV,
    G2_WEIGHT, "Full MSA sample (education)", "baseline controls (4)", "cluster(statefip)",
    G2_DESIGN_AUDIT, G2_INFERENCE)
print(f"  Baseline G2 (any college): coef={_[0]:.6f}, se={_[1]:.6f}, p={_[2]:.6f}, N={_[3]}")

# Additional baseline: bachelor's 18-25
run_iv(
    "baseline__table3_bachelor_18_25",
    "specification_tree/methods/instrumental_variables.md",
    "G2", G2_OUTCOME_BACH, G2_TREATMENT, G2_INSTRUMENT,
    G1_BASELINE_CONTROLS, df_g2, G2_VCOV,
    G2_WEIGHT, "Full MSA sample (education)", "baseline controls (4)", "cluster(statefip)",
    G2_DESIGN_AUDIT, G2_INFERENCE)


# ============================================================
# G2: DESIGN VARIANTS
# ============================================================

print("Running G2 design variants...")

# LIML
run_liml(
    "design/instrumental_variables/estimator/liml",
    "specification_tree/methods/instrumental_variables.md#liml",
    "G2", G2_OUTCOME_ANY, G2_TREATMENT, G2_INSTRUMENT,
    G1_BASELINE_CONTROLS, df_g2, "statefip_str", G2_WEIGHT,
    "Full MSA sample (education)", "baseline controls (LIML)", "cluster(statefip)",
    G2_DESIGN_AUDIT, G2_INFERENCE,
    axis_block_name="estimation",
    axis_block={"estimator": "liml"})


# ============================================================
# G2: CONTROLS LOO
# ============================================================

print("Running G2 controls LOO...")

for loo_name, drop_var in loo_map.items():
    loo_controls = [c for c in G1_BASELINE_CONTROLS if c != drop_var]
    run_iv(
        f"rc/controls/loo/{loo_name}",
        "specification_tree/modules/robustness/controls.md#leave-one-out",
        "G2", G2_OUTCOME_ANY, G2_TREATMENT, G2_INSTRUMENT,
        loo_controls, df_g2, G2_VCOV,
        G2_WEIGHT, "Full MSA sample (education)", f"baseline minus {drop_var}", "cluster(statefip)",
        G2_DESIGN_AUDIT, G2_INFERENCE,
        axis_block_name="controls",
        axis_block={"family": "loo", "dropped": [drop_var]})


# ============================================================
# G2: CONTROLS SETS
# ============================================================

print("Running G2 controls sets...")

# No controls
run_iv(
    "rc/controls/sets/no_controls",
    "specification_tree/modules/robustness/controls.md#controls-sets",
    "G2", G2_OUTCOME_ANY, G2_TREATMENT, G2_INSTRUMENT,
    [], df_g2, G2_VCOV,
    G2_WEIGHT, "Full MSA sample (education)", "no controls", "cluster(statefip)",
    G2_DESIGN_AUDIT, G2_INFERENCE,
    axis_block_name="controls",
    axis_block={"family": "sets", "set_name": "no_controls"})

# Baseline + region FE
if REGION_DUMMY_COLS_G2:
    run_iv(
        "rc/controls/sets/baseline_plus_region_fe",
        "specification_tree/modules/robustness/controls.md#controls-sets",
        "G2", G2_OUTCOME_ANY, G2_TREATMENT, G2_INSTRUMENT,
        G1_BASELINE_CONTROLS + REGION_DUMMY_COLS_G2, df_g2, G2_VCOV,
        G2_WEIGHT, "Full MSA sample (education)", "baseline + region dummies", "cluster(statefip)",
        G2_DESIGN_AUDIT, G2_INFERENCE,
        axis_block_name="controls",
        axis_block={"family": "sets", "set_name": "baseline_plus_region_fe"})

# Full with region FE
g2_extra_sample = df_g2.dropna(subset=G1_EXTRA_CONTROLS).copy()
if len(g2_extra_sample) >= 50 and REGION_DUMMY_COLS_G2:
    run_iv(
        "rc/controls/sets/full_with_region_fe",
        "specification_tree/modules/robustness/controls.md#controls-sets",
        "G2", G2_OUTCOME_ANY, G2_TREATMENT, G2_INSTRUMENT,
        G1_BASELINE_CONTROLS + G1_EXTRA_CONTROLS + REGION_DUMMY_COLS_G2, g2_extra_sample, G2_VCOV,
        G2_WEIGHT, "MSAs with all extra controls (education)",
        "baseline + all extra + region", "cluster(statefip)",
        G2_DESIGN_AUDIT, G2_INFERENCE,
        axis_block_name="controls",
        axis_block={"family": "sets", "set_name": "full_with_region_fe"})


# ============================================================
# G2: RANDOM CONTROL SUBSETS
# ============================================================

print("Running G2 random control subsets...")

np.random.seed(113109 + 1)
random_subset_count_g2 = 0
for _ in range(15):
    k = np.random.randint(2, min(6, len(all_possible_controls) + 1))
    subset = sorted(np.random.choice(all_possible_controls, size=k, replace=False).tolist())
    random_subset_count_g2 += 1
    sub_sample = df_g2.dropna(subset=subset).copy()
    if len(sub_sample) < 50:
        continue
    run_iv(
        f"rc/controls/subset/random_{random_subset_count_g2:02d}",
        "specification_tree/modules/robustness/controls.md#random-subsets",
        "G2", G2_OUTCOME_ANY, G2_TREATMENT, G2_INSTRUMENT,
        subset, sub_sample, G2_VCOV,
        G2_WEIGHT, f"MSAs with non-missing subset controls (N={len(sub_sample)})",
        f"random subset: {', '.join(subset)}", "cluster(statefip)",
        G2_DESIGN_AUDIT, G2_INFERENCE,
        axis_block_name="controls",
        axis_block={"family": "subset", "controls": subset})


# ============================================================
# G2: SAMPLE VARIANTS
# ============================================================

print("Running G2 sample variants...")

# Trim outcome 1%/99%
q_low_y_g2 = df_g2[G2_OUTCOME_ANY].quantile(0.01)
q_high_y_g2 = df_g2[G2_OUTCOME_ANY].quantile(0.99)
df_trim_y_g2 = df_g2[(df_g2[G2_OUTCOME_ANY] >= q_low_y_g2) & (df_g2[G2_OUTCOME_ANY] <= q_high_y_g2)].copy()
run_iv(
    "rc/sample/outliers/trim_y_1_99",
    "specification_tree/modules/robustness/sample.md#trimming",
    "G2", G2_OUTCOME_ANY, G2_TREATMENT, G2_INSTRUMENT,
    G1_BASELINE_CONTROLS, df_trim_y_g2, G2_VCOV,
    G2_WEIGHT, f"Trim outcome 1/99 (N={len(df_trim_y_g2)})", "baseline controls", "cluster(statefip)",
    G2_DESIGN_AUDIT, G2_INFERENCE,
    axis_block_name="sample",
    axis_block={"family": "outliers", "trim": "y_1_99"})

# Drop extreme elasticity
if df_g2['elasticity'].notna().sum() > 0:
    q10_e_g2 = df_g2['elasticity'].quantile(0.10)
    q90_e_g2 = df_g2['elasticity'].quantile(0.90)
    df_no_extreme_g2 = df_g2[(df_g2['elasticity'] >= q10_e_g2) & (df_g2['elasticity'] <= q90_e_g2)].copy()
    run_iv(
        "rc/sample/subset/drop_extreme_elasticity",
        "specification_tree/modules/robustness/sample.md#subsets",
        "G2", G2_OUTCOME_ANY, G2_TREATMENT, G2_INSTRUMENT,
        G1_BASELINE_CONTROLS, df_no_extreme_g2, G2_VCOV,
        G2_WEIGHT, f"Drop extreme elasticity (N={len(df_no_extreme_g2)})",
        "baseline controls", "cluster(statefip)",
        G2_DESIGN_AUDIT, G2_INFERENCE,
        axis_block_name="sample",
        axis_block={"family": "subset", "rule": "drop_extreme_elasticity_10_90"})


# ============================================================
# G2: OUTCOME VARIANTS
# ============================================================

print("Running G2 outcome variants...")

# Associate degree outcome
run_iv(
    "rc/form/outcome/associate_degree",
    "specification_tree/modules/robustness/functional_form.md#outcome",
    "G2", "d_assoc_18_25_a1", G2_TREATMENT, G2_INSTRUMENT,
    G1_BASELINE_CONTROLS, df_g2, G2_VCOV,
    G2_WEIGHT, "Full MSA sample (education)", "baseline controls, associate degree outcome", "cluster(statefip)",
    G2_DESIGN_AUDIT, G2_INFERENCE,
    axis_block_name="functional_form",
    axis_block={"family": "outcome", "outcome": "associate_degree_18_25"})

# Older age group: 26-33
run_iv(
    "rc/form/outcome/older_age_group_26_33",
    "specification_tree/modules/robustness/functional_form.md#outcome",
    "G2", "d_any_26_33_a1", G2_TREATMENT, G2_INSTRUMENT,
    G1_BASELINE_CONTROLS, df_g2, G2_VCOV,
    G2_WEIGHT, "Full MSA sample (education)", "baseline controls, any college 26-33", "cluster(statefip)",
    G2_DESIGN_AUDIT, G2_INFERENCE,
    axis_block_name="functional_form",
    axis_block={"family": "outcome", "outcome": "any_college_26_33"})

# Bachelor 26-33
run_iv(
    "rc/form/outcome/bachelor_26_33",
    "specification_tree/modules/robustness/functional_form.md#outcome",
    "G2", "d_bachelor_26_33_a1", G2_TREATMENT, G2_INSTRUMENT,
    G1_BASELINE_CONTROLS, df_g2, G2_VCOV,
    G2_WEIGHT, "Full MSA sample (education)", "baseline controls, bachelor 26-33", "cluster(statefip)",
    G2_DESIGN_AUDIT, G2_INFERENCE,
    axis_block_name="functional_form",
    axis_block={"family": "outcome", "outcome": "bachelor_26_33"})


# ============================================================
# G2: INSTRUMENT VARIANTS
# ============================================================

print("Running G2 instrument variants...")

# Binary IV (above median)
df_g2['iv_sig'] = (df_g2['iv'] > df_g2['iv'].median()).astype(float)
run_iv(
    "rc/data/instrument/iv_sig",
    "specification_tree/modules/robustness/instrument.md#alternatives",
    "G2", G2_OUTCOME_ANY, G2_TREATMENT, "iv_sig",
    G1_BASELINE_CONTROLS, df_g2, G2_VCOV,
    G2_WEIGHT, "Full MSA sample (education)", "baseline controls, binary IV", "cluster(statefip)",
    G2_DESIGN_AUDIT, G2_INFERENCE,
    axis_block_name="data_construction",
    axis_block={"family": "instrument", "instrument": "iv_sig"})

# Binary at 2/3
p67_g2 = df_g2['iv'].quantile(0.667)
df_g2['iv_sig2'] = (df_g2['iv'] > p67_g2).astype(float)
run_iv(
    "rc/data/instrument/iv_sig2",
    "specification_tree/modules/robustness/instrument.md#alternatives",
    "G2", G2_OUTCOME_ANY, G2_TREATMENT, "iv_sig2",
    G1_BASELINE_CONTROLS, df_g2, G2_VCOV,
    G2_WEIGHT, "Full MSA sample (education)", "baseline controls, binary IV top 1/3", "cluster(statefip)",
    G2_DESIGN_AUDIT, G2_INFERENCE,
    axis_block_name="data_construction",
    axis_block={"family": "instrument", "instrument": "iv_sig2"})

# iv2_poly3
df_g2_iv2p3 = df_g2.dropna(subset=['iv2_poly3']).copy()
if len(df_g2_iv2p3) >= 50:
    run_iv(
        "rc/data/instrument/iv2_poly3",
        "specification_tree/modules/robustness/instrument.md#alternatives",
        "G2", G2_OUTCOME_ANY, G2_TREATMENT, "iv2_poly3",
        G1_BASELINE_CONTROLS, df_g2_iv2p3, G2_VCOV,
        G2_WEIGHT, f"MSAs with iv2_poly3 (N={len(df_g2_iv2p3)})",
        "baseline controls, polynomial IV", "cluster(statefip)",
        G2_DESIGN_AUDIT, G2_INFERENCE,
        axis_block_name="data_construction",
        axis_block={"family": "instrument", "instrument": "iv2_poly3"})

# iv2_log
df_g2_iv2log = df_g2.dropna(subset=['iv2_log']).copy()
if len(df_g2_iv2log) >= 50:
    run_iv(
        "rc/data/instrument/price_rent_ratio",
        "specification_tree/modules/robustness/instrument.md#alternatives",
        "G2", G2_OUTCOME_ANY, G2_TREATMENT, "iv2_log",
        G1_BASELINE_CONTROLS, df_g2_iv2log, G2_VCOV,
        G2_WEIGHT, f"MSAs with iv2_log (N={len(df_g2_iv2log)})",
        "baseline controls, log IV", "cluster(statefip)",
        G2_DESIGN_AUDIT, G2_INFERENCE,
        axis_block_name="data_construction",
        axis_block={"family": "instrument", "instrument": "iv2_log"})


# ============================================================
# G2: WEIGHT VARIANTS
# ============================================================

print("Running G2 weight variants...")

# Unweighted
run_iv(
    "rc/weights/unweighted",
    "specification_tree/modules/robustness/weights.md#unweighted",
    "G2", G2_OUTCOME_ANY, G2_TREATMENT, G2_INSTRUMENT,
    G1_BASELINE_CONTROLS, df_g2, G2_VCOV,
    None, "Full MSA sample (education, unweighted)", "baseline controls", "cluster(statefip)",
    G2_DESIGN_AUDIT, G2_INFERENCE,
    axis_block_name="weights",
    axis_block={"family": "unweighted"})


# ============================================================
# G2: JOINT VARIANTS (controls + instrument)
# ============================================================

print("Running G2 joint variants...")

# No controls + binary IV
run_iv(
    "rc/joint/controls_instrument/no_ctrl_iv_sig",
    "specification_tree/modules/robustness/joint.md",
    "G2", G2_OUTCOME_ANY, G2_TREATMENT, "iv_sig",
    [], df_g2, G2_VCOV,
    G2_WEIGHT, "Full MSA sample (education)", "no controls, binary IV", "cluster(statefip)",
    G2_DESIGN_AUDIT, G2_INFERENCE,
    axis_block_name="joint",
    axis_block={"controls": "none", "instrument": "iv_sig"})

# Full controls + iv2_poly3
if len(df_g2_iv2p3) >= 50 and REGION_DUMMY_COLS_G2:
    run_iv(
        "rc/joint/controls_instrument/full_ctrl_iv2_poly3",
        "specification_tree/modules/robustness/joint.md",
        "G2", G2_OUTCOME_ANY, G2_TREATMENT, "iv2_poly3",
        G1_BASELINE_CONTROLS + REGION_DUMMY_COLS_G2, df_g2_iv2p3, G2_VCOV,
        G2_WEIGHT, f"MSAs with iv2_poly3 (N={len(df_g2_iv2p3)})",
        "baseline+region, polynomial IV", "cluster(statefip)",
        G2_DESIGN_AUDIT, G2_INFERENCE,
        axis_block_name="joint",
        axis_block={"controls": "baseline_plus_region", "instrument": "iv2_poly3"})


# ============================================================
# INFERENCE VARIANTS
# ============================================================

print("\n=== Running Inference Variants ===")

infer_counter = 0

def run_inference_variant(base_run_id, spec_id, spec_tree_path, baseline_group_id,
                          outcome_var, treatment_var, instrument_var,
                          controls, data, vcov_spec, weight_var,
                          focal_var, vcov_desc):
    """Re-run baseline spec with different variance-covariance estimator."""
    global infer_counter
    infer_counter += 1
    infer_run_id = f"{PAPER_ID}_infer_{infer_counter:03d}"

    try:
        ctrl_str = " + ".join(controls) if controls else "1"
        formula = f"{outcome_var} ~ {ctrl_str} | 0 | {treatment_var} ~ {instrument_var}"

        kwargs = dict(data=data, vcov=vcov_spec)
        if weight_var:
            kwargs['weights'] = weight_var

        m = pf.feols(formula, **kwargs)

        coef_val = float(m.coef().get(focal_var, np.nan))
        se_val = float(m.se().get(focal_var, np.nan))
        pval = float(m.pvalue().get(focal_var, np.nan))

        try:
            ci = m.confint()
            ci_lower = float(ci.loc[focal_var, ci.columns[0]])
            ci_upper = float(ci.loc[focal_var, ci.columns[1]])
        except Exception:
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
            design={"instrumental_variables": G1_DESIGN_AUDIT if baseline_group_id == "G1" else G2_DESIGN_AUDIT},
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
        err_det = error_details_from_exception(e, stage="inference")
        payload = make_failure_payload(
            error=err_msg,
            error_details=err_det,
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
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


# G1 inference variants
print("G1 inference variants...")

# HC1 (robust, no clustering)
run_inference_variant(
    baseline_g1_run_id, "infer/se/hc/hc1",
    "specification_tree/modules/inference/standard_errors.md#heteroskedasticity-robust",
    "G1", G1_OUTCOME, G1_TREATMENT, G1_INSTRUMENT,
    G1_BASELINE_CONTROLS, df_g1, "hetero", G1_WEIGHT,
    G1_TREATMENT, "HC1 (robust, no clustering)")

# HC2
run_inference_variant(
    baseline_g1_run_id, "infer/se/hc/hc2",
    "specification_tree/modules/inference/standard_errors.md#hc2",
    "G1", G1_OUTCOME, G1_TREATMENT, G1_INSTRUMENT,
    G1_BASELINE_CONTROLS, df_g1, "HC2", G1_WEIGHT,
    G1_TREATMENT, "HC2 (small-sample correction)")

# G2 inference variants
print("G2 inference variants...")

# HC1
run_inference_variant(
    baseline_g2_run_id, "infer/se/hc/hc1",
    "specification_tree/modules/inference/standard_errors.md#heteroskedasticity-robust",
    "G2", G2_OUTCOME_ANY, G2_TREATMENT, G2_INSTRUMENT,
    G1_BASELINE_CONTROLS, df_g2, "hetero", G2_WEIGHT,
    G2_TREATMENT, "HC1 (robust, no clustering)")


# ============================================================
# WRITE OUTPUTS
# ============================================================

print(f"\n=== Writing outputs ===")
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
    print("\nFailed specs:")
    for _, row in failed.iterrows():
        print(f"  {row['spec_id']}: {row['run_error'][:80]}")

# G1 summary
g1_success = successful[successful['baseline_group_id'] == 'G1']
g2_success = successful[successful['baseline_group_id'] == 'G2']

for grp_name, grp_df in [("G1 (Employment)", g1_success), ("G2 (Education)", g2_success)]:
    if len(grp_df) > 0:
        base_rows = grp_df[grp_df['spec_id'].str.startswith('baseline')]
        if len(base_rows) > 0:
            bc = base_rows.iloc[0]
            print(f"\n{grp_name} Baseline:")
            print(f"  Coef: {bc['coefficient']:.6f}")
            print(f"  SE: {bc['std_error']:.6f}")
            print(f"  p-value: {bc['p_value']:.6f}")
            print(f"  N: {bc['n_obs']:.0f}")

        print(f"\n{grp_name} Coefficient Range:")
        print(f"  Min: {grp_df['coefficient'].min():.6f}")
        print(f"  Max: {grp_df['coefficient'].max():.6f}")
        print(f"  Median: {grp_df['coefficient'].median():.6f}")
        n_sig = (grp_df['p_value'] < 0.05).sum()
        print(f"  Significant at 5%: {n_sig}/{len(grp_df)}")


# ============================================================
# SPECIFICATION_SEARCH.md
# ============================================================

print("\nWriting SPECIFICATION_SEARCH.md...")

md_lines = []
md_lines.append("# Specification Search Report: 113109-V1")
md_lines.append("")
md_lines.append("**Paper:** Charles, Hurst, & Notowidigdo (2018), \"Housing Booms and Busts, Labor Market Outcomes, and College Attendance\", AER")
md_lines.append("")
md_lines.append("## Design")
md_lines.append("")
md_lines.append("- **Method:** Instrumental Variables (2SLS)")
md_lines.append("- **Instrument:** Saiz housing supply elasticity")
md_lines.append("- **Treatment (G1):** hp_growth_real_00_06 (deltaP + units_growth)")
md_lines.append("- **Treatment (G2):** housing_demand_shock (same construction)")
md_lines.append("- **Controls:** college_share_2000, female_employed_share_2000, pop_prev, share_foreign_18_55_2000")
md_lines.append("- **Clustering:** statefip")
md_lines.append("")

# G1 baseline
md_lines.append("## G1: Employment (Table 1)")
md_lines.append("")
if len(g1_success) > 0:
    base_g1 = g1_success[g1_success['spec_id'].str.startswith('baseline')]
    if len(base_g1) > 0:
        bc = base_g1.iloc[0]
        md_lines.append("### Baseline")
        md_lines.append("")
        md_lines.append(f"| Statistic | Value |")
        md_lines.append(f"|-----------|-------|")
        md_lines.append(f"| Outcome | d_emp_18_25_le |")
        md_lines.append(f"| Coefficient | {bc['coefficient']:.6f} |")
        md_lines.append(f"| Std. Error | {bc['std_error']:.6f} |")
        md_lines.append(f"| p-value | {bc['p_value']:.6f} |")
        md_lines.append(f"| 95% CI | [{bc['ci_lower']:.6f}, {bc['ci_upper']:.6f}] |")
        md_lines.append(f"| N | {bc['n_obs']:.0f} |")
        md_lines.append(f"| R-squared | {bc['r_squared']:.4f} |")
        md_lines.append(f"| Weight | msa_total_emp_all_2000 |")
        md_lines.append("")

# G2 baseline
md_lines.append("## G2: College Enrollment (Table 3)")
md_lines.append("")
if len(g2_success) > 0:
    base_g2 = g2_success[g2_success['spec_id'].str.startswith('baseline')]
    if len(base_g2) > 0:
        for _, bc in base_g2.iterrows():
            md_lines.append(f"### {bc['spec_id']}")
            md_lines.append("")
            md_lines.append(f"| Statistic | Value |")
            md_lines.append(f"|-----------|-------|")
            md_lines.append(f"| Outcome | {bc['outcome_var']} |")
            md_lines.append(f"| Coefficient | {bc['coefficient']:.6f} |")
            md_lines.append(f"| Std. Error | {bc['std_error']:.6f} |")
            md_lines.append(f"| p-value | {bc['p_value']:.6f} |")
            md_lines.append(f"| 95% CI | [{bc['ci_lower']:.6f}, {bc['ci_upper']:.6f}] |")
            md_lines.append(f"| N | {bc['n_obs']:.0f} |")
            md_lines.append(f"| Weight | pop (exp(pop_prev)) |")
            md_lines.append("")

md_lines.append("## Specification Counts")
md_lines.append("")
md_lines.append(f"- Total specifications: {len(spec_df)}")
md_lines.append(f"- Successful: {len(successful)}")
md_lines.append(f"- Failed: {len(failed)}")
md_lines.append(f"- Inference variants: {len(infer_df)}")
md_lines.append("")

# Category breakdown by group
for grp_name, grp_df in [("G1", g1_success), ("G2", g2_success)]:
    md_lines.append(f"### {grp_name} Category Breakdown")
    md_lines.append("")
    md_lines.append("| Category | Count | Sig. (p<0.05) | Coef Range |")
    md_lines.append("|----------|-------|---------------|------------|")

    categories = {
        "Baseline": grp_df[grp_df['spec_id'].str.startswith('baseline')],
        "Design (LIML)": grp_df[grp_df['spec_id'].str.startswith('design/')],
        "Controls LOO": grp_df[grp_df['spec_id'].str.startswith('rc/controls/loo/')],
        "Controls Sets": grp_df[grp_df['spec_id'].str.startswith('rc/controls/sets/')],
        "Controls Subset": grp_df[grp_df['spec_id'].str.startswith('rc/controls/subset/')],
        "Sample": grp_df[grp_df['spec_id'].str.startswith('rc/sample/')],
        "Treatment/Outcome": grp_df[grp_df['spec_id'].str.startswith('rc/form/')],
        "Instrument": grp_df[grp_df['spec_id'].str.startswith('rc/data/instrument/')],
        "Weights": grp_df[grp_df['spec_id'].str.startswith('rc/weights/')],
        "Joint": grp_df[grp_df['spec_id'].str.startswith('rc/joint/')],
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
    md_lines.append("| Group | Spec ID | SE | p-value | 95% CI |")
    md_lines.append("|-------|---------|-----|---------|--------|")
    for _, row in infer_df.iterrows():
        if row['run_success'] == 1:
            md_lines.append(f"| {row['baseline_group_id']} | {row['spec_id']} | {row['std_error']:.6f} | {row['p_value']:.6f} | [{row['ci_lower']:.6f}, {row['ci_upper']:.6f}] |")
        else:
            md_lines.append(f"| {row['baseline_group_id']} | {row['spec_id']} | FAILED | - | - |")

md_lines.append("")

# Overall assessment
md_lines.append("## Overall Assessment")
md_lines.append("")

for grp_name, grp_df in [("G1 (Employment)", g1_success), ("G2 (Education)", g2_success)]:
    if len(grp_df) > 0:
        n_sig_total = (grp_df['p_value'] < 0.05).sum()
        pct_sig = n_sig_total / len(grp_df) * 100
        sign_consistent = ((grp_df['coefficient'] > 0).sum() == len(grp_df)) or \
                          ((grp_df['coefficient'] < 0).sum() == len(grp_df))
        median_coef = grp_df['coefficient'].median()
        sign_word = "positive" if median_coef > 0 else "negative"

        if pct_sig >= 80 and sign_consistent:
            strength = "STRONG"
        elif pct_sig >= 50 and sign_consistent:
            strength = "MODERATE"
        elif pct_sig >= 30:
            strength = "WEAK"
        else:
            strength = "FRAGILE"

        md_lines.append(f"### {grp_name}")
        md_lines.append(f"- **Sign consistency:** {'All specifications have the same sign' if sign_consistent else 'Mixed signs across specifications'}")
        md_lines.append(f"- **Significance stability:** {n_sig_total}/{len(grp_df)} ({pct_sig:.1f}%) significant at 5%")
        md_lines.append(f"- **Direction:** Median coefficient is {sign_word} ({median_coef:.6f})")
        md_lines.append(f"- **Robustness assessment:** {strength}")
        md_lines.append("")

md_lines.append(f"Surface hash: `{SURFACE_HASH}`")
md_lines.append("")

with open(f"{OUTPUT_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write("\n".join(md_lines))

print(f"Wrote SPECIFICATION_SEARCH.md")
print("\nDone!")
