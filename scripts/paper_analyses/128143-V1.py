"""
Specification Search Script for Douenne & Fabre (2022)
"Yellow Vests, Pessimistic Beliefs, and Carbon Tax Aversion"
American Economic Journal: Economic Policy

Paper ID: 128143-V1

Surface-driven execution:
  - G1: Self-interest IV (Table 5.2): acceptance ~ non_perdant (instrumented by random dividend eligibility)
  - G2: Environmental effectiveness IV (Table 5.4): approval ~ effectiveness (instrumented by info treatment)
  - Manual 2SLS (LPM first stage, LPM second stage) with survey weights
  - 50+ specifications total

Outputs:
  - specification_results.csv (baseline, rc/* rows)
  - inference_results.csv (infer/* rows)
  - SPECIFICATION_SEARCH.md
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json
import sys
import warnings
import hashlib

warnings.filterwarnings('ignore')

# Paths
BASE_DIR = "data/downloads/extracted/128143-V1"
CODE_DIR = f"{BASE_DIR}/yellow_vests_aej_ep/code"
PAPER_ID = "128143-V1"

sys.path.insert(0, "scripts")
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash, software_block
)

# Load surface
with open(f"{BASE_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

# ============================================================
# DATA LOADING AND PREPARATION
# ============================================================
print("Loading data...")
df = pd.read_csv(f"{CODE_DIR}/survey_prepared.csv", encoding='latin1', sep=';', low_memory=False)
print(f"  Loaded survey: {df.shape[0]} rows, {df.shape[1]} cols")

# Fix weight column: uses comma as decimal separator (French CSV artifact)
if 'weight' in df.columns and df['weight'].dtype == object:
    df['weight'] = df['weight'].astype(str).str.replace(',', '.', regex=False).astype(float)

# Convert string-numeric columns to numeric
# Some columns use dot decimals but stored as strings
numeric_convert_cols = [
    'Simule_gain', 'Simule_gain2', 'hausse_depenses_par_uc',
    'Revenu', 'Revenu2', 'Revenu_conjoint', 'Revenu_conjoint2',
    'revenu', 'revenu_conjoint', 'rev_tot', 'niveau_vie', 'uc',
    'nb_14_et_plus', 'nb_adultes', 'taille_menage',
    'age_18_24', 'age_25_34', 'age_35_49', 'age_50_64', 'age_65_plus',
    'simule_gain'
]
for c in numeric_convert_cols:
    if c in df.columns:
        if df[c].dtype == object:
            # Handle potential comma decimals
            df[c] = df[c].astype(str).str.replace(',', '.', regex=False)
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Load ERFS reference distribution to compute percentile_revenu
rev_ref = pd.read_csv(f"{CODE_DIR}/rev_i_erfs2014.csv")
ref_sorted = np.sort(rev_ref['x'].values)

def percentiles_revenu(x):
    """Empirical CDF: proportion of reference values <= x."""
    return np.searchsorted(ref_sorted, x, side='right') / len(ref_sorted)

# Create percentile_revenu and percentile_revenu_conjoint
df['percentile_revenu'] = 100 * percentiles_revenu(df['revenu'].fillna(0).values * 12)
df['percentile_revenu_conjoint'] = 100 * percentiles_revenu(df['revenu_conjoint'].fillna(0).values * 12)

# Create piecewise linear income terms (knots at 20 and 70)
for v in ['percentile_revenu', 'percentile_revenu_conjoint']:
    df[f'{v}_k20'] = np.minimum(df[v] - 20, 0)
    df[f'{v}_k70'] = np.minimum(df[v] - 70, 0)

# Create binary outcome and treatment variables
df['outcome_cible_notno'] = (df['taxe_cible_approbation'] != 'Non').astype(float)
df['outcome_cible_yes'] = (df['taxe_cible_approbation'] == 'Oui').astype(float)
df['outcome_initial_yes'] = (df['taxe_approbation'] == 'Oui').astype(float)
df['outcome_initial_notno'] = (df['taxe_approbation'] != 'Non').astype(float)

df['endogenous_non_perdant'] = (df['gagnant_cible_categorie'] != 'Perdant').astype(float)
df['endogenous_eff_yes'] = (df['taxe_efficace'] == 'Oui').astype(float)
df['endogenous_eff_notno'] = (df['taxe_efficace'] != 'Non').astype(float)

df['taxe_approbation_nsp'] = (df['taxe_approbation'] == 'NSP').astype(float)
df['taxe_efficace_num'] = (df['taxe_efficace'] == 'Oui').astype(float)

# Handle prog_na: in R code prog_na is already in the data
# Convert categorical vars to numeric where needed
for c in ['prog_na']:
    if c in df.columns and df[c].dtype == 'object':
        # prog_na values: Non, Oui, NSP -> convert to dummies
        df['prog_na_oui'] = (df['prog_na'] == 'Oui').astype(float)
        df['prog_na_nsp'] = (df['prog_na'] == 'NSP').astype(float)

# Create traite interaction
df['traite_interaction'] = df['traite_cible'] * df['traite_cible_conjoint']

# Create gagnant_categorie dummies for G2
df['gagnant_cat_gagnant'] = (df['gagnant_categorie'] == 'Gagnant').astype(float)
df['gagnant_cat_nonaffecte'] = (df['gagnant_categorie'] == 'Non affect\xe9').astype(float)

# Create apres_modifs numeric
df['apres_modifs_num'] = df['apres_modifs'].astype(float) if df['apres_modifs'].dtype == bool else pd.to_numeric(df['apres_modifs'], errors='coerce')
df['info_CC_num'] = pd.to_numeric(df['info_CC'], errors='coerce')

# Handle factor variables that R treats as dummies: sexe, statut_emploi, csp, region, diplome4, taille_agglo, fume, actualite
# In R, these are passed as-is to lm() which auto-dummifies factors
# We need to create dummies manually for OLS in Python
# First, let's check types
factor_vars = ['sexe', 'statut_emploi', 'csp', 'region', 'diplome4', 'taille_agglo', 'fume', 'actualite']
dummy_frames = []
dummy_col_map = {}  # maps conceptual var -> list of dummy columns

for fv in factor_vars:
    if fv in df.columns:
        vals = df[fv].astype(str)
        unique_vals = sorted(vals.unique())
        if len(unique_vals) <= 1:
            continue
        # Drop first category (like R default)
        base_cat = unique_vals[0]
        dummies = pd.get_dummies(vals, prefix=fv, drop_first=True, dtype=float)
        # Clean column names for formula compatibility
        clean_cols = {}
        for col in dummies.columns:
            clean_name = col.replace(' ', '_').replace("'", '').replace(',', '').replace('-', '_').replace('(', '').replace(')', '').replace('/', '_').replace('.', '_')
            clean_cols[col] = f"d_{clean_name}"
        dummies = dummies.rename(columns=clean_cols)
        dummy_col_map[fv] = list(dummies.columns)
        for col in dummies.columns:
            df[col] = dummies[col].values

# Also handle single numeric "factor" vars
for fv in ['single']:
    if fv in df.columns:
        dummy_col_map[fv] = [fv]

# Define variable groups matching the R code
# Demographics (numeric + dummies): 17 vars minus excluded (revenu, rev_tot, age, age_65_plus)
demo_numeric = ['taille_menage', 'nb_14_et_plus', 'nb_adultes', 'uc', 'niveau_vie',
                'age_18_24', 'age_25_34', 'age_35_49', 'age_50_64']
demo_dummies = []
for fv in ['sexe', 'statut_emploi', 'csp', 'region', 'diplome4', 'taille_agglo', 'fume', 'actualite']:
    if fv in dummy_col_map:
        demo_dummies.extend(dummy_col_map[fv])

all_demo_cols = demo_numeric + demo_dummies

# Piecewise income terms
piecewise_income = ['percentile_revenu', 'percentile_revenu_k20', 'percentile_revenu_k70',
                    'percentile_revenu_conjoint', 'percentile_revenu_conjoint_k20', 'percentile_revenu_conjoint_k70']

# G1 controls (self-interest): variables_reg_self_interest
g1_named = ['Simule_gain', 'Simule_gain2', 'taxe_efficace_num', 'single', 'hausse_depenses_par_uc']
g1_prog_na_dummies = ['prog_na_oui', 'prog_na_nsp']
g1_variables_reg_si = g1_prog_na_dummies + g1_named + all_demo_cols + piecewise_income
g1_second_stage_extras = ['taxe_approbation_nsp', 'tax_acceptance']
# 'cible' is numeric with values 20,30,40,50 -- treat as continuous in baseline
g1_cible = ['cible']

# G2 controls (environmental effectiveness): variables_reg_ee
g2_named_numeric = ['Revenu', 'Revenu2', 'Revenu_conjoint', 'Revenu_conjoint2', 'single', 'Simule_gain', 'Simule_gain2']
g2_gagnant_dummies = ['gagnant_cat_gagnant', 'gagnant_cat_nonaffecte']
g2_variables_reg_ee = g2_named_numeric + g2_gagnant_dummies + all_demo_cols

# Subsamples
df['subsample_p10_p60'] = ((df['percentile_revenu'] <= 60) & (df['percentile_revenu'] >= 10)) | \
                          ((df['percentile_revenu_conjoint'] <= 60) & (df['percentile_revenu_conjoint'] >= 10))

# Drop rows with missing values in key variables
# For G1
g1_all_vars_first = ['traite_cible', 'traite_cible_conjoint', 'traite_interaction', 'endogenous_non_perdant'] + g1_variables_reg_si + g1_cible + g1_second_stage_extras
g1_all_vars_second = ['outcome_cible_notno'] + g1_variables_reg_si + g1_cible + g1_second_stage_extras + ['non_perdant_hat']

print(f"  Subsample [10,60]: {df['subsample_p10_p60'].sum()} rows")
print(f"  G1 controls: {len(g1_variables_reg_si)} reg vars + {len(g1_cible)} cible + {len(g1_second_stage_extras)} extras")
print(f"  G2 controls: {len(g2_variables_reg_ee)} reg vars")

# Results containers
results = []
inference_results = []
spec_run_counter = 0


def run_manual_2sls(spec_id, spec_tree_path, baseline_group_id,
                    outcome_var, endogenous_var, instruments, controls,
                    data, weight_col, sample_desc, controls_desc,
                    axis_block_name=None, axis_block=None,
                    func_form_block=None, notes=""):
    """
    Run manual 2SLS: first-stage LPM, second-stage LPM with fitted values.
    This matches the paper's approach using R's lm() with weights.
    """
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        # Prepare data: drop NAs in all relevant columns
        all_vars = [outcome_var, endogenous_var] + instruments + controls
        if weight_col:
            all_vars.append(weight_col)

        df_reg = data.dropna(subset=[v for v in all_vars if v in data.columns]).copy()
        if len(df_reg) == 0:
            raise ValueError("No observations after dropping NAs")

        # Build X matrices
        first_stage_X = instruments + controls
        second_stage_X = controls + ['endogenous_hat']

        # First stage: endogenous_var ~ instruments + controls
        X1 = df_reg[first_stage_X].astype(float)
        X1 = sm.add_constant(X1)
        y1 = df_reg[endogenous_var].astype(float)
        w = df_reg[weight_col].astype(float) if weight_col else None

        m1 = sm.WLS(y1, X1, weights=w).fit() if w is not None else sm.OLS(y1, X1).fit()
        df_reg['endogenous_hat'] = m1.fittedvalues

        # Second stage: outcome_var ~ fitted(endogenous) + controls
        X2 = df_reg[second_stage_X].astype(float)
        X2 = sm.add_constant(X2)
        y2 = df_reg[outcome_var].astype(float)

        m2 = sm.WLS(y2, X2, weights=w).fit() if w is not None else sm.OLS(y2, X2).fit()

        # Extract focal coefficient on endogenous_hat
        coef_val = float(m2.params.get('endogenous_hat', np.nan))
        se_val = float(m2.bse.get('endogenous_hat', np.nan))
        pval = float(m2.pvalues.get('endogenous_hat', np.nan))
        ci = m2.conf_int()
        ci_lower = float(ci.loc['endogenous_hat', 0]) if 'endogenous_hat' in ci.index else np.nan
        ci_upper = float(ci.loc['endogenous_hat', 1]) if 'endogenous_hat' in ci.index else np.nan
        nobs = int(m2.nobs)
        r2 = float(m2.rsquared)

        all_coefs = {k: float(v) for k, v in m2.params.items()}

        # First-stage F-stat on instruments
        first_stage_f = None
        try:
            # Test joint significance of instruments in first stage
            r_matrix = np.zeros((len(instruments), len(m1.params)))
            for i, inst in enumerate(instruments):
                if inst in m1.params.index:
                    idx = list(m1.params.index).index(inst)
                    r_matrix[i, idx] = 1
            f_test = m1.f_test(r_matrix)
            first_stage_f = float(f_test.fvalue)
        except Exception:
            first_stage_f = None

        # Build payload
        blocks = {}
        if axis_block_name and axis_block:
            blocks[axis_block_name] = axis_block
        if func_form_block:
            blocks["functional_form"] = func_form_block
        blocks["bundle"] = {
            "first_stage": {
                "dep_var": endogenous_var,
                "instruments": instruments,
                "nobs": nobs,
                "first_stage_f": first_stage_f,
                "first_stage_r2": float(m1.rsquared),
            },
            "second_stage": {
                "dep_var": outcome_var,
                "endogenous_hat": "endogenous_hat",
                "nobs": nobs,
                "r2": r2,
            }
        }

        design_info = surface_obj["baseline_groups"][0 if baseline_group_id == "G1" else 1]["design_audit"]

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": "infer/se/ols_default", "params": {},
                       "method": "ols_default", "type": "manual_2sls_lpm_se"},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"randomized_experiment": design_info},
            blocks=blocks,
        )

        results.append({
            "paper_id": PAPER_ID,
            "spec_run_id": run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": endogenous_var,
            "coefficient": coef_val,
            "std_error": se_val,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": nobs,
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": "",
            "controls_desc": controls_desc,
            "cluster_var": "",
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
            "treatment_var": endogenous_var,
            "coefficient": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": np.nan,
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": "",
            "controls_desc": controls_desc,
            "cluster_var": "",
            "run_success": 0,
            "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


def run_inference_variant_2sls(base_run_id, spec_id, spec_tree_path, baseline_group_id,
                                outcome_var, endogenous_var, instruments, controls,
                                data, weight_col, vcov_type):
    """Re-run 2SLS under different inference (HC1, HC2)."""
    global spec_run_counter
    spec_run_counter += 1
    infer_run_id = f"{PAPER_ID}_infer_{spec_run_counter:03d}"

    try:
        all_vars = [outcome_var, endogenous_var] + instruments + controls
        if weight_col:
            all_vars.append(weight_col)
        df_reg = data.dropna(subset=[v for v in all_vars if v in data.columns]).copy()

        first_stage_X_cols = instruments + controls
        X1 = sm.add_constant(df_reg[first_stage_X_cols].astype(float))
        y1 = df_reg[endogenous_var].astype(float)
        w = df_reg[weight_col].astype(float) if weight_col else None

        m1 = sm.WLS(y1, X1, weights=w).fit() if w is not None else sm.OLS(y1, X1).fit()
        df_reg['endogenous_hat'] = m1.fittedvalues

        second_stage_X_cols = controls + ['endogenous_hat']
        X2 = sm.add_constant(df_reg[second_stage_X_cols].astype(float))
        y2 = df_reg[outcome_var].astype(float)

        # Fit with robust SEs
        m2 = sm.WLS(y2, X2, weights=w).fit(cov_type=vcov_type) if w is not None else sm.OLS(y2, X2).fit(cov_type=vcov_type)

        coef_val = float(m2.params.get('endogenous_hat', np.nan))
        se_val = float(m2.bse.get('endogenous_hat', np.nan))
        pval = float(m2.pvalues.get('endogenous_hat', np.nan))
        ci = m2.conf_int()
        ci_lower = float(ci.loc['endogenous_hat', 0]) if 'endogenous_hat' in ci.index else np.nan
        ci_upper = float(ci.loc['endogenous_hat', 1]) if 'endogenous_hat' in ci.index else np.nan
        nobs = int(m2.nobs)
        r2 = float(m2.rsquared)

        all_coefs = {k: float(v) for k, v in m2.params.items()}
        design_info = surface_obj["baseline_groups"][0 if baseline_group_id == "G1" else 1]["design_audit"]

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": spec_id, "params": {},
                       "method": vcov_type, "type": f"manual_2sls_{vcov_type}_se"},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"randomized_experiment": design_info},
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
            "run_success": 0,
            "run_error": err_msg
        })


# ============================================================
# G1: SELF-INTEREST IV (Table 5.2)
# ============================================================
print("\n=== G1: Self-Interest IV ===")

# G1 Instruments
g1_instruments = ['traite_cible', 'traite_cible_conjoint', 'traite_interaction']

# G1 Full control set for first stage = variables_reg_self_interest + cible + taxe_approbation_nsp + tax_acceptance
# For second stage = variables_reg_self_interest + cible + taxe_approbation_nsp + tax_acceptance + non_perdant_hat
g1_controls = g1_variables_reg_si + g1_cible + g1_second_stage_extras

# Subsample data
df_g1_sub = df[df['subsample_p10_p60']].copy()

# BASELINE: Table 5.2 Col 1 - IV on subsample [p10,p60]
print("  Running baseline (Table 5.2 Col 1, subsample [p10,p60])...")
base_g1_run, base_g1_coef, base_g1_se, base_g1_p, base_g1_n = run_manual_2sls(
    "baseline", "designs/randomized_experiment.md#baseline", "G1",
    "outcome_cible_notno", "endogenous_non_perdant",
    g1_instruments, g1_controls,
    df_g1_sub, "weight",
    "Subsample: percentile_revenu in [10,60]",
    "variables_reg_self_interest + cible + taxe_approbation_nsp + tax_acceptance"
)
print(f"    Baseline G1: coef={base_g1_coef:.4f}, se={base_g1_se:.4f}, p={base_g1_p:.4f}, N={base_g1_n}")

# ADDITIONAL BASELINE: Table 5.2 Col 2 - IV on full sample
print("  Running additional baseline (Table 5.2 Col 2, full sample)...")
# Full sample has an extra 'single' in second stage (already in controls via g1_named)
g1_controls_full = g1_variables_reg_si + g1_cible + g1_second_stage_extras
run_manual_2sls(
    "baseline__table52_col2_iv_fullsample", "designs/randomized_experiment.md#baseline", "G1",
    "outcome_cible_notno", "endogenous_non_perdant",
    g1_instruments, g1_controls_full,
    df, "weight",
    "Full sample",
    "variables_reg_self_interest + cible + taxe_approbation_nsp + tax_acceptance + single"
)

# RC: OUTCOME FORM - approval Yes instead of acceptance not-No
print("  Running outcome variant: approval Yes...")
run_manual_2sls(
    "rc/form/outcome_approval_yes", "modules/robustness/functional_form.md#outcome-transform", "G1",
    "outcome_cible_yes", "endogenous_non_perdant",
    g1_instruments, g1_controls,
    df_g1_sub, "weight",
    "Subsample [p10,p60]",
    "baseline controls; outcome = taxe_cible_approbation=='Oui'",
    func_form_block={"spec_id": "rc/form/outcome_approval_yes",
                     "transform": "binary_recoding",
                     "interpretation": "Effect on approval (Yes only) vs acceptance (not No)",
                     "original_var": "taxe_cible_approbation!='Non'",
                     "transformed_var": "taxe_cible_approbation=='Oui'"}
)

# RC: LOO controls (by conceptual block)
print("  Running LOO controls...")

# Drop taxe_efficace
g1_loo_taxe_efficace = [c for c in g1_controls if c != 'taxe_efficace_num']
run_manual_2sls(
    "rc/controls/loo/taxe_efficace", "modules/robustness/controls.md#leave-one-out-controls-loo", "G1",
    "outcome_cible_notno", "endogenous_non_perdant",
    g1_instruments, g1_loo_taxe_efficace,
    df_g1_sub, "weight",
    "Subsample [p10,p60]", "baseline minus taxe_efficace",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/taxe_efficace", "family": "loo",
                "dropped": ["taxe_efficace"], "n_controls": len(g1_loo_taxe_efficace)}
)

# Drop tax_acceptance
g1_loo_tax_acceptance = [c for c in g1_controls if c != 'tax_acceptance']
run_manual_2sls(
    "rc/controls/loo/tax_acceptance", "modules/robustness/controls.md#leave-one-out-controls-loo", "G1",
    "outcome_cible_notno", "endogenous_non_perdant",
    g1_instruments, g1_loo_tax_acceptance,
    df_g1_sub, "weight",
    "Subsample [p10,p60]", "baseline minus tax_acceptance",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/tax_acceptance", "family": "loo",
                "dropped": ["tax_acceptance"], "n_controls": len(g1_loo_tax_acceptance)}
)

# Drop piecewise income (all 6 terms)
g1_loo_piecewise = [c for c in g1_controls if c not in piecewise_income]
run_manual_2sls(
    "rc/controls/loo/piecewise_income", "modules/robustness/controls.md#leave-one-out-controls-loo", "G1",
    "outcome_cible_notno", "endogenous_non_perdant",
    g1_instruments, g1_loo_piecewise,
    df_g1_sub, "weight",
    "Subsample [p10,p60]", "baseline minus piecewise income (6 terms)",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/piecewise_income", "family": "loo",
                "dropped": piecewise_income, "n_controls": len(g1_loo_piecewise)}
)

# Drop demographics block (all demo dummies and numerics)
g1_loo_demographics = [c for c in g1_controls if c not in all_demo_cols]
run_manual_2sls(
    "rc/controls/loo/demographics", "modules/robustness/controls.md#leave-one-out-controls-loo", "G1",
    "outcome_cible_notno", "endogenous_non_perdant",
    g1_instruments, g1_loo_demographics,
    df_g1_sub, "weight",
    "Subsample [p10,p60]", "baseline minus all demographics",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/demographics", "family": "loo",
                "dropped": ["demographics_block"], "n_controls": len(g1_loo_demographics)}
)

# Drop hausse_depenses_par_uc
g1_loo_hausse = [c for c in g1_controls if c != 'hausse_depenses_par_uc']
run_manual_2sls(
    "rc/controls/loo/hausse_depenses_par_uc", "modules/robustness/controls.md#leave-one-out-controls-loo", "G1",
    "outcome_cible_notno", "endogenous_non_perdant",
    g1_instruments, g1_loo_hausse,
    df_g1_sub, "weight",
    "Subsample [p10,p60]", "baseline minus hausse_depenses_par_uc",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/hausse_depenses_par_uc", "family": "loo",
                "dropped": ["hausse_depenses_par_uc"], "n_controls": len(g1_loo_hausse)}
)

# Drop simule_gain (Simule_gain + Simule_gain2)
g1_loo_simule = [c for c in g1_controls if c not in ['Simule_gain', 'Simule_gain2']]
run_manual_2sls(
    "rc/controls/loo/simule_gain", "modules/robustness/controls.md#leave-one-out-controls-loo", "G1",
    "outcome_cible_notno", "endogenous_non_perdant",
    g1_instruments, g1_loo_simule,
    df_g1_sub, "weight",
    "Subsample [p10,p60]", "baseline minus Simule_gain + Simule_gain2",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/simule_gain", "family": "loo",
                "dropped": ["Simule_gain", "Simule_gain2"], "n_controls": len(g1_loo_simule)}
)

# Drop prog_na dummies
g1_loo_prog = [c for c in g1_controls if c not in g1_prog_na_dummies]
run_manual_2sls(
    "rc/controls/loo/prog_na", "modules/robustness/controls.md#leave-one-out-controls-loo", "G1",
    "outcome_cible_notno", "endogenous_non_perdant",
    g1_instruments, g1_loo_prog,
    df_g1_sub, "weight",
    "Subsample [p10,p60]", "baseline minus prog_na",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/prog_na", "family": "loo",
                "dropped": ["prog_na"], "n_controls": len(g1_loo_prog)}
)

# RC: UNWEIGHTED
print("  Running unweighted variant...")
run_manual_2sls(
    "rc/weights/unweighted", "modules/robustness/weights.md#unweighted", "G1",
    "outcome_cible_notno", "endogenous_non_perdant",
    g1_instruments, g1_controls,
    df_g1_sub, None,  # no weights
    "Subsample [p10,p60], unweighted",
    "baseline controls, no survey weights",
    axis_block_name="weights",
    axis_block={"spec_id": "rc/weights/unweighted", "weighted": False}
)

# G1 INFERENCE VARIANTS
print("  Running inference variants for G1...")
run_inference_variant_2sls(
    base_g1_run, "infer/se/hc/hc1",
    "modules/inference/standard_errors.md#heteroskedasticity-robust", "G1",
    "outcome_cible_notno", "endogenous_non_perdant",
    g1_instruments, g1_controls,
    df_g1_sub, "weight", "HC1"
)

run_inference_variant_2sls(
    base_g1_run, "infer/se/hc/hc2",
    "modules/inference/standard_errors.md#heteroskedasticity-robust", "G1",
    "outcome_cible_notno", "endogenous_non_perdant",
    g1_instruments, g1_controls,
    df_g1_sub, "weight", "HC2"
)


# ============================================================
# G2: ENVIRONMENTAL EFFECTIVENESS IV (Table 5.4)
# ============================================================
print("\n=== G2: Environmental Effectiveness IV ===")

# G2 Instruments
g2_instruments = ['apres_modifs_num', 'info_CC_num']

# BASELINE: Table 5.4 Col 1 - approval(Yes) ~ effectiveness(Yes)
print("  Running baseline G2 (Table 5.4 Col 1)...")
base_g2_run1, base_g2_coef1, base_g2_se1, base_g2_p1, base_g2_n1 = run_manual_2sls(
    "baseline", "designs/randomized_experiment.md#baseline", "G2",
    "outcome_initial_yes", "endogenous_eff_yes",
    g2_instruments, g2_variables_reg_ee,
    df, "weight",
    "Full sample",
    "variables_reg_ee (income quadratic + single + estimated gains + gagnant + demographics)"
)
print(f"    Baseline G2 Col1: coef={base_g2_coef1:.4f}, se={base_g2_se1:.4f}, p={base_g2_p1:.4f}, N={base_g2_n1}")

# ADDITIONAL BASELINE: Table 5.4 Col 3 - acceptance(not No) ~ effectiveness(Yes)
print("  Running additional baseline G2 (Table 5.4 Col 3)...")
base_g2_run3, base_g2_coef3, base_g2_se3, base_g2_p3, base_g2_n3 = run_manual_2sls(
    "baseline__table54_col3_iv_notnoyes", "designs/randomized_experiment.md#baseline", "G2",
    "outcome_initial_notno", "endogenous_eff_yes",
    g2_instruments, g2_variables_reg_ee,
    df, "weight",
    "Full sample",
    "variables_reg_ee; outcome = taxe_approbation!='Non'"
)
print(f"    Baseline G2 Col3: coef={base_g2_coef3:.4f}, se={base_g2_se3:.4f}, p={base_g2_p3:.4f}, N={base_g2_n3}")

# RC: OUTCOME FORM - approval Yes (this is really Col 1, so the variant is from Col 3's perspective)
print("  Running G2 outcome variant...")
run_manual_2sls(
    "rc/form/outcome_approval_yes", "modules/robustness/functional_form.md#outcome-transform", "G2",
    "outcome_initial_yes", "endogenous_eff_yes",
    g2_instruments, g2_variables_reg_ee,
    df, "weight",
    "Full sample",
    "baseline controls; outcome = taxe_approbation=='Oui' (from col3 baseline perspective)",
    func_form_block={"spec_id": "rc/form/outcome_approval_yes",
                     "transform": "binary_recoding",
                     "interpretation": "Effect on approval (Yes) vs acceptance (not No)",
                     "original_var": "taxe_approbation!='Non'",
                     "transformed_var": "taxe_approbation=='Oui'"}
)

# RC: LOO controls for G2
print("  Running LOO controls for G2...")

# Drop income quadratic (Revenu, Revenu2, Revenu_conjoint, Revenu_conjoint2)
g2_loo_income = [c for c in g2_variables_reg_ee if c not in ['Revenu', 'Revenu2', 'Revenu_conjoint', 'Revenu_conjoint2']]
run_manual_2sls(
    "rc/controls/loo/income_quadratic", "modules/robustness/controls.md#leave-one-out-controls-loo", "G2",
    "outcome_initial_notno", "endogenous_eff_yes",
    g2_instruments, g2_loo_income,
    df, "weight",
    "Full sample", "baseline minus income quadratic",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/income_quadratic", "family": "loo",
                "dropped": ["Revenu", "Revenu2", "Revenu_conjoint", "Revenu_conjoint2"],
                "n_controls": len(g2_loo_income)}
)

# Drop estimated gains (Simule_gain, Simule_gain2)
g2_loo_gains = [c for c in g2_variables_reg_ee if c not in ['Simule_gain', 'Simule_gain2']]
run_manual_2sls(
    "rc/controls/loo/estimated_gains", "modules/robustness/controls.md#leave-one-out-controls-loo", "G2",
    "outcome_initial_notno", "endogenous_eff_yes",
    g2_instruments, g2_loo_gains,
    df, "weight",
    "Full sample", "baseline minus estimated gains",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/estimated_gains", "family": "loo",
                "dropped": ["Simule_gain", "Simule_gain2"],
                "n_controls": len(g2_loo_gains)}
)

# Drop gagnant_categorie dummies
g2_loo_gagnant = [c for c in g2_variables_reg_ee if c not in g2_gagnant_dummies]
run_manual_2sls(
    "rc/controls/loo/gagnant_categorie", "modules/robustness/controls.md#leave-one-out-controls-loo", "G2",
    "outcome_initial_notno", "endogenous_eff_yes",
    g2_instruments, g2_loo_gagnant,
    df, "weight",
    "Full sample", "baseline minus gagnant_categorie",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/gagnant_categorie", "family": "loo",
                "dropped": ["gagnant_categorie"], "n_controls": len(g2_loo_gagnant)}
)

# Drop demographics block
g2_loo_demo = [c for c in g2_variables_reg_ee if c not in all_demo_cols]
run_manual_2sls(
    "rc/controls/loo/demographics", "modules/robustness/controls.md#leave-one-out-controls-loo", "G2",
    "outcome_initial_notno", "endogenous_eff_yes",
    g2_instruments, g2_loo_demo,
    df, "weight",
    "Full sample", "baseline minus demographics",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/demographics", "family": "loo",
                "dropped": ["demographics_block"], "n_controls": len(g2_loo_demo)}
)

# RC: UNWEIGHTED for G2
print("  Running unweighted variant for G2...")
run_manual_2sls(
    "rc/weights/unweighted", "modules/robustness/weights.md#unweighted", "G2",
    "outcome_initial_notno", "endogenous_eff_yes",
    g2_instruments, g2_variables_reg_ee,
    df, None,
    "Full sample, unweighted", "baseline controls, no survey weights",
    axis_block_name="weights",
    axis_block={"spec_id": "rc/weights/unweighted", "weighted": False}
)

# G2 INFERENCE VARIANTS
print("  Running inference variants for G2...")
# Use Col 3 (not No ~ Yes) as base for inference variants
run_inference_variant_2sls(
    base_g2_run3, "infer/se/hc/hc1",
    "modules/inference/standard_errors.md#heteroskedasticity-robust", "G2",
    "outcome_initial_notno", "endogenous_eff_yes",
    g2_instruments, g2_variables_reg_ee,
    df, "weight", "HC1"
)

run_inference_variant_2sls(
    base_g2_run3, "infer/se/hc/hc2",
    "modules/inference/standard_errors.md#heteroskedasticity-robust", "G2",
    "outcome_initial_notno", "endogenous_eff_yes",
    g2_instruments, g2_variables_reg_ee,
    df, "weight", "HC2"
)

# ============================================================
# ADDITIONAL G1 and G2 specs for combinatorial variety
# ============================================================
print("\n=== Running additional combined variants ===")

# G1: Full sample versions of LOO specs
for loo_name, loo_ctrls in [
    ("taxe_efficace", g1_loo_taxe_efficace),
    ("tax_acceptance", g1_loo_tax_acceptance),
    ("piecewise_income", g1_loo_piecewise),
    ("demographics", g1_loo_demographics),
    ("hausse_depenses_par_uc", g1_loo_hausse),
    ("simule_gain", g1_loo_simule),
    ("prog_na", g1_loo_prog),
]:
    run_manual_2sls(
        f"rc/controls/loo/{loo_name}__fullsample",
        "modules/robustness/controls.md#leave-one-out-controls-loo", "G1",
        "outcome_cible_notno", "endogenous_non_perdant",
        g1_instruments, loo_ctrls,
        df, "weight",
        f"Full sample", f"baseline minus {loo_name} (full sample)",
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/controls/loo/{loo_name}", "family": "loo",
                    "dropped": [loo_name], "sample": "full",
                    "n_controls": len(loo_ctrls)}
    )

# G1: Outcome variant on full sample
run_manual_2sls(
    "rc/form/outcome_approval_yes__fullsample",
    "modules/robustness/functional_form.md#outcome-transform", "G1",
    "outcome_cible_yes", "endogenous_non_perdant",
    g1_instruments, g1_controls_full,
    df, "weight",
    "Full sample",
    "baseline controls; outcome = taxe_cible_approbation=='Oui' (full sample)",
    func_form_block={"spec_id": "rc/form/outcome_approval_yes",
                     "transform": "binary_recoding",
                     "interpretation": "Approval (Yes) on full sample",
                     "original_var": "taxe_cible_approbation!='Non'",
                     "transformed_var": "taxe_cible_approbation=='Oui'"}
)

# G1: Unweighted on full sample
run_manual_2sls(
    "rc/weights/unweighted__fullsample",
    "modules/robustness/weights.md#unweighted", "G1",
    "outcome_cible_notno", "endogenous_non_perdant",
    g1_instruments, g1_controls_full,
    df, None,
    "Full sample, unweighted", "baseline controls, no survey weights (full sample)",
    axis_block_name="weights",
    axis_block={"spec_id": "rc/weights/unweighted", "weighted": False, "sample": "full"}
)

# G2: LOO on approval (Yes) outcome
for loo_name, loo_ctrls in [
    ("income_quadratic", g2_loo_income),
    ("estimated_gains", g2_loo_gains),
    ("gagnant_categorie", g2_loo_gagnant),
    ("demographics", g2_loo_demo),
]:
    run_manual_2sls(
        f"rc/controls/loo/{loo_name}__approval_yes",
        "modules/robustness/controls.md#leave-one-out-controls-loo", "G2",
        "outcome_initial_yes", "endogenous_eff_yes",
        g2_instruments, loo_ctrls,
        df, "weight",
        f"Full sample", f"baseline minus {loo_name}; outcome = approval (Yes)",
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/controls/loo/{loo_name}", "family": "loo",
                    "dropped": [loo_name], "outcome": "approval_yes",
                    "n_controls": len(loo_ctrls)}
    )

# G2: Unweighted + approval variant
run_manual_2sls(
    "rc/weights/unweighted__approval_yes",
    "modules/robustness/weights.md#unweighted", "G2",
    "outcome_initial_yes", "endogenous_eff_yes",
    g2_instruments, g2_variables_reg_ee,
    df, None,
    "Full sample, unweighted", "baseline controls, approval (Yes), no weights",
    axis_block_name="weights",
    axis_block={"spec_id": "rc/weights/unweighted", "weighted": False, "outcome": "approval_yes"}
)

# Additional G1 inference on full-sample baseline
run_inference_variant_2sls(
    f"{PAPER_ID}_run_002",  # full-sample baseline
    "infer/se/hc/hc1",
    "modules/inference/standard_errors.md#heteroskedasticity-robust", "G1",
    "outcome_cible_notno", "endogenous_non_perdant",
    g1_instruments, g1_controls_full,
    df, "weight", "HC1"
)

# Additional G2 inference on Col 1 baseline
run_inference_variant_2sls(
    base_g2_run1,
    "infer/se/hc/hc1",
    "modules/inference/standard_errors.md#heteroskedasticity-robust", "G2",
    "outcome_initial_yes", "endogenous_eff_yes",
    g2_instruments, g2_variables_reg_ee,
    df, "weight", "HC1"
)

# ============================================================
# ADDITIONAL SPECS: Individual demo factor LOO (G1 subsample)
# ============================================================
print("\n=== Running individual demographic LOO (G1 subsample) ===")

individual_demo_loo = {
    'sexe': dummy_col_map.get('sexe', []),
    'statut_emploi': dummy_col_map.get('statut_emploi', []),
    'csp': dummy_col_map.get('csp', []),
    'region': dummy_col_map.get('region', []),
    'diplome4': dummy_col_map.get('diplome4', []),
    'taille_agglo': dummy_col_map.get('taille_agglo', []),
    'fume': dummy_col_map.get('fume', []),
    'actualite': dummy_col_map.get('actualite', []),
}

for demo_name, demo_cols_to_drop in individual_demo_loo.items():
    if not demo_cols_to_drop:
        continue
    loo_ctrl = [c for c in g1_controls if c not in demo_cols_to_drop]
    run_manual_2sls(
        f"rc/controls/loo/demo_{demo_name}",
        "modules/robustness/controls.md#leave-one-out-controls-loo", "G1",
        "outcome_cible_notno", "endogenous_non_perdant",
        g1_instruments, loo_ctrl,
        df_g1_sub, "weight",
        f"Subsample [p10,p60]", f"baseline minus demo: {demo_name}",
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/controls/loo/demo_{demo_name}", "family": "loo",
                    "dropped": [demo_name], "n_controls": len(loo_ctrl)}
    )

# ============================================================
# ADDITIONAL SPECS: Individual demo factor LOO (G2 full sample)
# ============================================================
print("\n=== Running individual demographic LOO (G2 full sample) ===")

for demo_name, demo_cols_to_drop in individual_demo_loo.items():
    if not demo_cols_to_drop:
        continue
    loo_ctrl = [c for c in g2_variables_reg_ee if c not in demo_cols_to_drop]
    run_manual_2sls(
        f"rc/controls/loo/demo_{demo_name}",
        "modules/robustness/controls.md#leave-one-out-controls-loo", "G2",
        "outcome_initial_notno", "endogenous_eff_yes",
        g2_instruments, loo_ctrl,
        df, "weight",
        f"Full sample", f"baseline minus demo: {demo_name}",
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/controls/loo/demo_{demo_name}", "family": "loo",
                    "dropped": [demo_name], "n_controls": len(loo_ctrl)}
    )

# ============================================================
# ADDITIONAL SPECS: Drop numeric demo vars individually (G1)
# ============================================================
print("\n=== Running numeric demo LOO (G1) ===")

numeric_demo_loo = ['taille_menage', 'nb_14_et_plus', 'nb_adultes', 'uc', 'niveau_vie']
for var in numeric_demo_loo:
    loo_ctrl = [c for c in g1_controls if c != var]
    run_manual_2sls(
        f"rc/controls/loo/demo_{var}",
        "modules/robustness/controls.md#leave-one-out-controls-loo", "G1",
        "outcome_cible_notno", "endogenous_non_perdant",
        g1_instruments, loo_ctrl,
        df_g1_sub, "weight",
        f"Subsample [p10,p60]", f"baseline minus {var}",
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/controls/loo/demo_{var}", "family": "loo",
                    "dropped": [var], "n_controls": len(loo_ctrl)}
    )

# ============================================================
# ADDITIONAL SPECS: Drop single (G1 and G2)
# ============================================================
print("\n=== Running drop-single variants ===")

# G1: drop single
g1_loo_single = [c for c in g1_controls if c != 'single']
run_manual_2sls(
    "rc/controls/loo/single",
    "modules/robustness/controls.md#leave-one-out-controls-loo", "G1",
    "outcome_cible_notno", "endogenous_non_perdant",
    g1_instruments, g1_loo_single,
    df_g1_sub, "weight",
    "Subsample [p10,p60]", "baseline minus single",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/single", "family": "loo",
                "dropped": ["single"], "n_controls": len(g1_loo_single)}
)

# G2: drop single
g2_loo_single = [c for c in g2_variables_reg_ee if c != 'single']
run_manual_2sls(
    "rc/controls/loo/single",
    "modules/robustness/controls.md#leave-one-out-controls-loo", "G2",
    "outcome_initial_notno", "endogenous_eff_yes",
    g2_instruments, g2_loo_single,
    df, "weight",
    "Full sample", "baseline minus single",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/single", "family": "loo",
                "dropped": ["single"], "n_controls": len(g2_loo_single)}
)

# ============================================================
# ADDITIONAL SPECS: Drop cible (G1 only, it's the policy variant control)
# ============================================================
g1_loo_cible = [c for c in g1_controls if c != 'cible']
run_manual_2sls(
    "rc/controls/loo/cible",
    "modules/robustness/controls.md#leave-one-out-controls-loo", "G1",
    "outcome_cible_notno", "endogenous_non_perdant",
    g1_instruments, g1_loo_cible,
    df_g1_sub, "weight",
    "Subsample [p10,p60]", "baseline minus cible",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/cible", "family": "loo",
                "dropped": ["cible"], "n_controls": len(g1_loo_cible)}
)

# ============================================================
# WRITE OUTPUTS
# ============================================================
print("\n=== Writing outputs ===")

df_results = pd.DataFrame(results)
df_results.to_csv(f"{BASE_DIR}/specification_results.csv", index=False)
print(f"Wrote {len(df_results)} rows to specification_results.csv")

df_infer = pd.DataFrame(inference_results)
df_infer.to_csv(f"{BASE_DIR}/inference_results.csv", index=False)
print(f"Wrote {len(df_infer)} rows to inference_results.csv")

n_success = df_results['run_success'].sum()
n_fail = (df_results['run_success'] == 0).sum()
n_infer_success = df_infer['run_success'].sum() if len(df_infer) > 0 else 0
n_infer_fail = (df_infer['run_success'] == 0).sum() if len(df_infer) > 0 else 0

# Summary by group
g1_specs = df_results[df_results['baseline_group_id'] == 'G1']
g2_specs = df_results[df_results['baseline_group_id'] == 'G2']

md_lines = [
    f"# Specification Search: {PAPER_ID}",
    f"**Paper**: Douenne & Fabre (2022) 'Yellow Vests, Pessimistic Beliefs, and Carbon Tax Aversion'",
    f"**Design**: Randomized experiment (survey experiment with IV/2SLS)",
    "",
    "## Surface Summary",
    f"- Baseline groups: 2 (G1: self-interest IV, G2: environmental effectiveness IV)",
    f"- G1 budget: max 60 core, 30 control subsets",
    f"- G2 budget: max 50 core, 25 control subsets",
    f"- Seed: 128143",
    f"- Surface hash: {SURFACE_HASH}",
    "",
    "## G1: Self-Interest IV (Table 5.2)",
    f"- Outcome: taxe_cible_approbation != 'Non' (carbon tax acceptance)",
    f"- Endogenous: non_perdant (believes does not lose from targeted dividend)",
    f"- Instruments: traite_cible, traite_cible_conjoint, interaction",
    f"- Method: Manual 2SLS (first-stage LPM, second-stage LPM with fitted values)",
    f"- Weights: survey weights",
    f"- Subsample: percentile_revenu in [10,60] (near income threshold)",
    f"- Baseline coefficient: {base_g1_coef:.4f} (SE={base_g1_se:.4f}, p={base_g1_p:.4f}, N={base_g1_n})",
    f"- G1 specs: {len(g1_specs)} rows ({g1_specs['run_success'].sum()} success)",
    "",
    "## G2: Environmental Effectiveness IV (Table 5.4)",
    f"- Outcome: taxe_approbation != 'Non' (tax acceptance) / == 'Oui' (approval)",
    f"- Endogenous: taxe_efficace == 'Oui' (believes tax is environmentally effective)",
    f"- Instruments: apres_modifs (info on EE), info_CC (info on climate change)",
    f"- Method: Manual 2SLS with survey weights",
    f"- Baseline Col1 (Yes~Yes): coef={base_g2_coef1:.4f}, SE={base_g2_se1:.4f}, p={base_g2_p1:.4f}, N={base_g2_n1}",
    f"- Baseline Col3 (notNo~Yes): coef={base_g2_coef3:.4f}, SE={base_g2_se3:.4f}, p={base_g2_p3:.4f}, N={base_g2_n3}",
    f"- G2 specs: {len(g2_specs)} rows ({g2_specs['run_success'].sum()} success)",
    "",
    "## Execution Summary",
    f"- Total specification rows: {len(df_results)} ({n_success} success, {n_fail} failed)",
    f"- Inference rows: {len(df_infer)} ({n_infer_success} success, {n_infer_fail} failed)",
    "",
    "### G1 Breakdown:",
    f"- baseline: 1 (subsample [p10,p60])",
    f"- baseline__fullsample: 1",
    f"- rc/form/outcome: 1 (approval Yes)",
    f"- rc/controls/loo: 7 (taxe_efficace, tax_acceptance, piecewise_income, demographics, hausse, simule_gain, prog_na)",
    f"- rc/weights/unweighted: 1",
    f"- rc/controls/loo (full sample): 7",
    f"- rc/form/outcome (full sample): 1",
    f"- rc/weights/unweighted (full sample): 1",
    "",
    "### G2 Breakdown:",
    f"- baseline: 1 (approval Yes ~ effectiveness Yes)",
    f"- baseline__col3: 1 (acceptance notNo ~ effectiveness Yes)",
    f"- rc/form/outcome: 1 (approval Yes variant from col3 baseline)",
    f"- rc/controls/loo: 4 (income, gains, gagnant, demographics)",
    f"- rc/weights/unweighted: 1",
    f"- rc/controls/loo (approval_yes): 4",
    f"- rc/weights/unweighted (approval_yes): 1",
    "",
    "### Inference variants:",
    f"- G1: HC1 on baseline (subsample), HC2 on baseline (subsample), HC1 on full sample",
    f"- G2: HC1 on Col3 baseline, HC2 on Col3 baseline, HC1 on Col1 baseline",
    "",
    "## Notes",
    "- Manual 2SLS: SEs do not correct for generated-regressor uncertainty (matches paper).",
    "- percentile_revenu computed from ERFS 2014 reference distribution (ecdf).",
    "- Piecewise linear income terms: knots at percentile 20 and 70.",
    "- Factor variables (sexe, statut_emploi, csp, region, diplome4, taille_agglo, fume, actualite)",
    "  converted to dummies with first-category dropped.",
    "- Linked adjustment: control sets vary jointly in first and second stage.",
    "",
    "## Software",
    f"- Python {sys.version.split()[0]}",
    f"- statsmodels {SW_BLOCK['packages'].get('statsmodels', 'N/A')}",
    f"- pandas {SW_BLOCK['packages'].get('pandas', 'N/A')}",
    f"- numpy {SW_BLOCK['packages'].get('numpy', 'N/A')}",
    "",
    "## Deviations",
    "- percentile_revenu reconstructed from ERFS reference distribution rather than loaded from RData.",
    "- Factor variable dummification may differ slightly in category ordering from R's default.",
    "- Some numeric columns required explicit conversion from string format in the CSV.",
]

with open(f"{BASE_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write("\n".join(md_lines) + "\n")
print(f"Wrote SPECIFICATION_SEARCH.md")

print(f"\n=== DONE: {PAPER_ID} ===")
print(f"Total specs: {len(df_results)} + {len(df_infer)} inference")
