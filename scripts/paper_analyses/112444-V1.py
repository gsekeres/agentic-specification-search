"""
Specification Search Script for Reinhart & Rogoff (2011)
"From Financial Crash to Debt Crisis"
American Economic Review, 101(5), 1676-1706.

Paper ID: 112444-V1

Surface-driven execution:
  - G1: Banking crisis ~ contagion variables (center focal)
  - G2: External debt crisis ~ contagion variables (debt_move focal)
  - Pooled OLS with HC2 robust SE (baseline)
  - Three estimation periods: 1824-2009, 1900-2009, 1946-2009
  - With and without public debt ratio
  - Logit/probit functional form alternatives
  - Country FE / year FE additions

Outputs:
  - specification_results.csv
  - inference_results.csv
  - SPECIFICATION_SEARCH.md
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import statsmodels.formula.api as smf
import statsmodels.api as sm
import json
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, "scripts")
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash, software_block
)

PAPER_ID = "112444-V1"
DATA_DIR = "data/downloads/extracted/112444-V1"
EXCEL_DIR = f"{DATA_DIR}/Reinhart_Rogoff_20080344_Data/Varieties"
DEBT_DIR = f"{DATA_DIR}/Reinhart_Rogoff_20080344_Data/Debt_to_GDP"
OUTPUT_DIR = DATA_DIR

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

G1_DESIGN_AUDIT = surface_obj["baseline_groups"][0]["design_audit"]
G2_DESIGN_AUDIT = surface_obj["baseline_groups"][1]["design_audit"]
G1_INFERENCE_CANONICAL = surface_obj["baseline_groups"][0]["inference_plan"]["canonical"]
G2_INFERENCE_CANONICAL = surface_obj["baseline_groups"][1]["inference_plan"]["canonical"]

results = []
inference_results = []
spec_run_counter = 0

# ============================================================
# DATA CONSTRUCTION
# ============================================================
# The TSP code loads data from external text files in wide format (71 country series).
# We reconstruct the panel from the Varieties Excel files (crisis indicators)
# and Debt_to_GDP Excel files (public debt/GDP).

print("Building panel dataset from Excel files...")

# --- Step 1: Extract banking and external debt crisis indicators ---
skip_sheets = ['Contents', 'CrisisDefinitions', 'CrisisDefinition', 'Sheet3', 'Sheet1']
parts = ['I', 'II', 'III', 'IV']

all_crisis_data = []
country_order = []

for part in parts:
    xls = pd.ExcelFile(f"{EXCEL_DIR}/Varieties_Part_{part}.xls")
    for sheet in xls.sheet_names:
        if sheet in skip_sheets:
            continue
        df_sheet = pd.read_excel(xls, sheet_name=sheet, header=None)
        # Find where data starts (first row with year >= 1800)
        start_row = None
        for i in range(len(df_sheet)):
            try:
                val = float(df_sheet.iloc[i, 0])
                if 1800 <= val <= 2010:
                    start_row = i
                    break
            except:
                pass
        if start_row is None:
            print(f"  WARNING: No data start for {sheet}")
            continue

        data = df_sheet.iloc[start_row:].copy()
        data.columns = range(len(data.columns))
        # Columns: 0=year, 1=independence, 2=currency, 3=inflation,
        #          4=stock_crash, 5=domestic_debt, 6=external_debt, 7=banking
        data = data[[0, 6, 7]].copy()
        data.columns = ['year', 'external_debt_crisis', 'banking_crisis']
        data['year'] = pd.to_numeric(data['year'], errors='coerce')
        data['external_debt_crisis'] = pd.to_numeric(data['external_debt_crisis'], errors='coerce')
        data['banking_crisis'] = pd.to_numeric(data['banking_crisis'], errors='coerce')
        data = data.dropna(subset=['year'])
        data['year'] = data['year'].astype(int)
        data = data[(data['year'] >= 1800) & (data['year'] <= 2009)]
        data['country'] = sheet
        country_order.append(sheet)
        all_crisis_data.append(data)

crisis_panel = pd.concat(all_crisis_data, ignore_index=True)
print(f"  Crisis panel: {len(crisis_panel)} obs, {len(country_order)} countries")

# --- Step 2: Create country index matching TSP ordering ---
# TSP uses countries 1-71; we have 70 from sheets. The TSP data uses b66=UK, b67=US.
# Our ordering matches: country_order[65]='UK', country_order[66]='US'
country_to_idx = {c: i+1 for i, c in enumerate(country_order)}

# Verify UK/US positions
assert country_order[65] == 'UK', f"Expected UK at 66, got {country_order[65]}"
assert country_order[66] == 'US', f"Expected US at 67, got {country_order[66]}"

# --- Step 3: Development classification ---
# Based on R&R classification (This Time is Different):
# Advanced economies (development=2 in TSP dummy coding):
advanced_countries = {
    'Australia', 'Austria', 'Belgium', 'Canada', 'Denmark', 'Finland', 'France',
    'Germany', 'Greece', 'Iceland', 'Ireland', 'Italy', 'Japan', 'Korea',
    'Netherlands', 'NewZealand', 'Norway', 'Portugal', 'Singapore', 'Spain',
    'Sweden', 'Switzerland', 'Taiwan', 'UK', 'US'
}
# Everything else is emerging (development=1 in TSP)
# Note: TSP uses DEVELOP1 = (development==1) = emerging,
#              DEVELOP2 = (development==2) = advanced
# No constant in the banking equations (develop1+develop2 span intercept)

crisis_panel['development'] = crisis_panel['country'].apply(
    lambda c: 2 if c in advanced_countries else 1
)

# --- Step 4: Extract public debt/GDP ratio ---
debt_skip = ['Contents', 'CrisisDefinitions', 'CrisisDefinition', 'Sheet3', 'Sheet1']
debt_parts = ['I', 'II', 'III', 'IV']

# Map Varieties country names to Debt_to_GDP sheet names
# They might differ slightly
debt_sheet_lookup = {}
for part in debt_parts:
    try:
        xls = pd.ExcelFile(f"{DEBT_DIR}/Debt_to_GDP_Part_{part}.xls")
        for s in xls.sheet_names:
            if s not in debt_skip:
                debt_sheet_lookup[s] = (part, s)
    except:
        pass

all_debt_data = []
for country in country_order:
    # Try exact match first, then common variants
    candidates = [country]
    if country == 'CentralAfricanRep':
        candidates.append('CentralAfricanRepublic')
        candidates.append('Central African Republic')
        candidates.append('CentralAfricanRep')
    if country == 'CoteDIvoire':
        candidates.append("CoteDIvoire")
        candidates.append("Cote D'Ivoire")
    if country == 'SouthAfrica':
        candidates.append('South Africa')
    if country == 'SriLanka':
        candidates.append('Sri Lanka')
    if country == 'NewZealand':
        candidates.append('New Zealand')
    if country == 'ElSalvador':
        candidates.append('El Salvador')
    if country == 'DominicanRepublic':
        candidates.append('Dominican Republic')
    if country == 'CostaRica':
        candidates.append('Costa Rica')

    found = False
    for cand in candidates:
        if cand in debt_sheet_lookup:
            part, sheet = debt_sheet_lookup[cand]
            try:
                df_debt = pd.read_excel(
                    f"{DEBT_DIR}/Debt_to_GDP_Part_{part}.xls",
                    sheet_name=sheet, header=None
                )
                # Find data start (year column in column 0)
                start_row = None
                for i in range(len(df_debt)):
                    try:
                        val = float(df_debt.iloc[i, 0])
                        if 1800 <= val <= 2010:
                            start_row = i
                            break
                    except:
                        pass
                if start_row is not None:
                    d = df_debt.iloc[start_row:].copy()
                    d.columns = range(len(d.columns))
                    # Column 0 = year, column 2 = total public debt/GDP (domestic+external)
                    d = d[[0, 2]].copy()
                    d.columns = ['year', 'public_debt_gdp']
                    d['year'] = pd.to_numeric(d['year'], errors='coerce')
                    d['public_debt_gdp'] = pd.to_numeric(d['public_debt_gdp'], errors='coerce')
                    d = d.dropna(subset=['year'])
                    d['year'] = d['year'].astype(int)
                    d['country'] = country
                    all_debt_data.append(d)
                    found = True
                    break
            except Exception as e:
                pass
    if not found:
        pass  # Some countries may not have debt data

if all_debt_data:
    debt_panel = pd.concat(all_debt_data, ignore_index=True)
    print(f"  Debt panel: {len(debt_panel)} obs")
else:
    debt_panel = pd.DataFrame(columns=['year', 'country', 'public_debt_gdp'])

# --- Step 5: Merge crisis and debt data ---
panel = crisis_panel.merge(debt_panel[['year', 'country', 'public_debt_gdp']],
                           on=['year', 'country'], how='left')

# --- Step 6: Construct derived variables (matching TSP code) ---
# bank_move = 3-year lagged MA of banking crisis: (bank(t-1) + bank(t-2) + bank(t-3))/3
# debt_move = 3-year lagged MA of ext debt crisis: (debt(t-1) + debt(t-2) + debt(t-3))/3
# center = (UK_bank(t) + UK_bank(t-1) + UK_bank(t-2) + US_bank(t) + US_bank(t-1) + US_bank(t-2))/6
# public = public_debt_gdp(t) - public_debt_gdp(t-2)   [2-year change]

panel = panel.sort_values(['country', 'year']).reset_index(drop=True)

# Create lagged MA for each country
for lag_col, source_col, new_col in [
    ('banking_crisis', 'banking_crisis', 'bank_move'),
    ('external_debt_crisis', 'external_debt_crisis', 'debt_move')
]:
    panel[new_col] = panel.groupby('country')[source_col].transform(
        lambda x: (x.shift(1) + x.shift(2) + x.shift(3)) / 3
    )

# Center variable: uses UK and US banking crises
uk_bank = crisis_panel[crisis_panel['country'] == 'UK'][['year', 'banking_crisis']].copy()
uk_bank.columns = ['year', 'uk_bank']
us_bank = crisis_panel[crisis_panel['country'] == 'US'][['year', 'banking_crisis']].copy()
us_bank.columns = ['year', 'us_bank']

center_df = uk_bank.merge(us_bank, on='year', how='outer')
center_df = center_df.sort_values('year').reset_index(drop=True)
# center = (uk(t) + uk(t-1) + uk(t-2) + us(t) + us(t-1) + us(t-2)) / 6
center_df['center'] = (
    center_df['uk_bank'] + center_df['uk_bank'].shift(1) + center_df['uk_bank'].shift(2) +
    center_df['us_bank'] + center_df['us_bank'].shift(1) + center_df['us_bank'].shift(2)
) / 6

panel = panel.merge(center_df[['year', 'center']], on='year', how='left')

# Public debt change (2-year change)
panel['public'] = panel.groupby('country')['public_debt_gdp'].transform(
    lambda x: x - x.shift(2)
)

# Development dummies (no constant in banking equations)
panel['develop1'] = (panel['development'] == 1).astype(float)
panel['develop2'] = (panel['development'] == 2).astype(float)

# Country numeric id for FE
panel['country_id'] = panel['country'].map(country_to_idx)

# Region variable (approximate -- R&R use: Africa, Asia, Europe, LatAm, NorthAm, Oceania)
region_map = {}
africa = ['Algeria', 'Angola', 'CentralAfricanRep', 'CoteDIvoire', 'Egypt', 'Ghana',
          'Kenya', 'Mauritius', 'Morocco', 'Nigeria', 'SouthAfrica', 'Tunisia', 'Zambia', 'Zimbabwe']
asia = ['China', 'India', 'Indonesia', 'Japan', 'Korea', 'Malaysia', 'Myanmar',
        'Philippines', 'Singapore', 'SriLanka', 'Taiwan', 'Thailand']
europe = ['Austria', 'Belgium', 'Denmark', 'Finland', 'France', 'Germany', 'Greece',
          'Hungary', 'Iceland', 'Ireland', 'Italy', 'Netherlands', 'Norway', 'Poland',
          'Portugal', 'Romania', 'Russia', 'Spain', 'Sweden', 'Switzerland', 'Turkey', 'UK']
latam = ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'CostaRica',
         'DominicanRepublic', 'Ecuador', 'ElSalvador', 'Guatemala', 'Honduras',
         'Mexico', 'Nicaragua', 'Panama', 'Paraguay', 'Peru', 'Uruguay', 'Venezuela']
northam = ['Canada', 'US']
oceania = ['Australia', 'NewZealand']

for c in africa: region_map[c] = 'Africa'
for c in asia: region_map[c] = 'Asia'
for c in europe: region_map[c] = 'Europe'
for c in latam: region_map[c] = 'LatAm'
for c in northam: region_map[c] = 'NorthAm'
for c in oceania: region_map[c] = 'Oceania'
panel['region'] = panel['country'].map(region_map).fillna('Other')

# Rename outcome variables
panel['bank'] = panel['banking_crisis']
panel['debt'] = panel['external_debt_crisis']

# Drop rows with missing constructed variables
panel_full = panel.dropna(subset=['bank', 'debt', 'bank_move', 'debt_move', 'center']).copy()

print(f"  Full panel (with derived vars): {len(panel_full)} obs")

# ============================================================
# SAMPLE DEFINITIONS
# ============================================================
# Three estimation periods matching the TSP code
df_1824 = panel_full[(panel_full['year'] >= 1824) & (panel_full['year'] <= 2009)].copy()
df_1900 = panel_full[(panel_full['year'] >= 1900) & (panel_full['year'] <= 2009)].copy()
df_1946 = panel_full[(panel_full['year'] >= 1947) & (panel_full['year'] <= 2009)].copy()

# Subsamples for public debt regressions (need non-missing public)
df_1824_pub = df_1824.dropna(subset=['public']).copy()
df_1900_pub = df_1900.dropna(subset=['public']).copy()
df_1946_pub = df_1946.dropna(subset=['public']).copy()

print(f"  1824-2009: {len(df_1824)} obs (with public: {len(df_1824_pub)})")
print(f"  1900-2009: {len(df_1900)} obs (with public: {len(df_1900_pub)})")
print(f"  1946-2009: {len(df_1946)} obs (with public: {len(df_1946_pub)})")

# Advanced/emerging subsamples
df_1824_adv = df_1824[df_1824['develop2'] == 1].copy()
df_1824_eme = df_1824[df_1824['develop1'] == 1].copy()

# ============================================================
# HELPER: Run OLS (pooled, no intercept when using develop dummies)
# ============================================================
def run_ols_spec(spec_id, spec_tree_path, baseline_group_id,
                 outcome_var, treatment_var, rhs_vars,
                 data, sample_desc, controls_desc,
                 design_audit, inference_canonical,
                 fe_vars=None, use_constant=False,
                 cluster_var_name="", vcov_spec="hetero",
                 axis_block_name=None, axis_block=None,
                 func_form_block=None, notes=""):
    """Run pooled OLS with specified RHS variables.
    If use_constant=False and develop1/develop2 in rhs_vars, no constant added.
    If use_constant=True, constant is added (for debt equations in some periods).
    """
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        # Build formula
        rhs_str = " + ".join(rhs_vars)

        if fe_vars:
            fe_str = " + ".join(fe_vars)
            if use_constant:
                formula = f"{outcome_var} ~ {rhs_str} | {fe_str}"
            else:
                formula = f"{outcome_var} ~ {rhs_str} - 1 | {fe_str}"
        else:
            if use_constant:
                formula = f"{outcome_var} ~ {rhs_str}"
            else:
                formula = f"{outcome_var} ~ {rhs_str} - 1"

        # Use pyfixest for HC1/HC2 or clustered SE
        # Note: pyfixest "hetero" = HC1. For HC2, use statsmodels.
        # TSP HCTYPE=2 = HC2 robust SE
        # We use statsmodels OLS with HC2 for canonical inference

        if fe_vars:
            # Use pyfixest for FE models
            m = pf.feols(formula, data=data, vcov=vcov_spec)
            coef_val = float(m.coef().get(treatment_var, np.nan))
            se_val = float(m.se().get(treatment_var, np.nan))
            pval = float(m.pvalue().get(treatment_var, np.nan))
            try:
                ci = m.confint()
                ci_lower = float(ci.loc[treatment_var, ci.columns[0]])
                ci_upper = float(ci.loc[treatment_var, ci.columns[1]])
            except:
                ci_lower, ci_upper = np.nan, np.nan
            nobs = int(m._N)
            try:
                r2 = float(m._r2)
            except:
                r2 = np.nan
            all_coefs = {k: float(v) for k, v in m.coef().items()}
        else:
            # Use statsmodels for non-FE models to get HC2
            reg_vars = [outcome_var] + rhs_vars
            df_reg = data[reg_vars].dropna().copy()

            y = df_reg[outcome_var].values
            if use_constant:
                X = sm.add_constant(df_reg[rhs_vars].values)
                x_names = ['Intercept'] + rhs_vars
            else:
                X = df_reg[rhs_vars].values
                x_names = rhs_vars

            model = sm.OLS(y, X)
            res = model.fit(cov_type='HC2')

            tidx = x_names.index(treatment_var)
            coef_val = float(res.params[tidx])
            se_val = float(res.bse[tidx])
            pval = float(res.pvalues[tidx])
            ci = res.conf_int()
            ci_lower = float(ci[tidx, 0])
            ci_upper = float(ci[tidx, 1])
            nobs = int(res.nobs)
            r2 = float(res.rsquared)
            all_coefs = {x_names[i]: float(res.params[i]) for i in range(len(x_names))}

        # Build payload
        payload_kwargs = dict(
            coefficients=all_coefs,
            inference={"spec_id": inference_canonical["spec_id"],
                       "params": inference_canonical.get("params", {})},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"panel_fixed_effects": design_audit},
        )
        if axis_block_name and axis_block:
            payload_kwargs["axis_block_name"] = axis_block_name
            payload_kwargs["axis_block"] = axis_block
        if func_form_block:
            payload_kwargs["axis_block_name"] = "functional_form"
            payload_kwargs["axis_block"] = func_form_block
        if notes:
            payload_kwargs["notes"] = notes

        payload = make_success_payload(**payload_kwargs)

        fe_desc = ",".join(fe_vars) if fe_vars else "none"

        results.append({
            "paper_id": PAPER_ID, "spec_run_id": run_id,
            "spec_id": spec_id, "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var, "treatment_var": treatment_var,
            "coefficient": coef_val, "std_error": se_val, "p_value": pval,
            "ci_lower": ci_lower, "ci_upper": ci_upper,
            "n_obs": nobs, "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc, "fixed_effects": fe_desc,
            "controls_desc": controls_desc, "cluster_var": cluster_var_name,
            "run_success": 1, "run_error": ""
        })
        return run_id, coef_val, se_val, pval, nobs

    except Exception as e:
        err_msg = str(e)[:240]
        payload = make_failure_payload(
            error=err_msg,
            error_details=error_details_from_exception(e, stage="estimation"),
            software=SW_BLOCK, surface_hash=SURFACE_HASH
        )
        results.append({
            "paper_id": PAPER_ID, "spec_run_id": run_id,
            "spec_id": spec_id, "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var, "treatment_var": treatment_var,
            "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "n_obs": np.nan, "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc, "fixed_effects": fe_desc if 'fe_desc' in dir() else "none",
            "controls_desc": controls_desc, "cluster_var": cluster_var_name,
            "run_success": 0, "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


def run_logit_spec(spec_id, spec_tree_path, baseline_group_id,
                   outcome_var, treatment_var, rhs_vars,
                   data, sample_desc, controls_desc,
                   design_audit, inference_canonical,
                   use_constant=True, cluster_var_name="",
                   axis_block_name=None, axis_block=None,
                   func_form_block=None, notes=""):
    """Run logit model and return marginal effects at means."""
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        reg_vars = [outcome_var] + rhs_vars
        df_reg = data[reg_vars].dropna().copy()

        # statsmodels logit
        rhs_str = " + ".join(rhs_vars)
        if use_constant:
            formula = f"{outcome_var} ~ {rhs_str}"
        else:
            formula = f"{outcome_var} ~ {rhs_str} - 1"

        logit_res = smf.logit(formula, data=df_reg).fit(cov_type='HC2', disp=0)

        # Get coefficient for treatment var
        coef_val = float(logit_res.params[treatment_var])
        se_val = float(logit_res.bse[treatment_var])
        pval = float(logit_res.pvalues[treatment_var])
        try:
            ci = logit_res.conf_int()
            ci_lower = float(ci.loc[treatment_var, 0])
            ci_upper = float(ci.loc[treatment_var, 1])
        except:
            ci_lower, ci_upper = np.nan, np.nan
        nobs = int(logit_res.nobs)
        r2 = float(logit_res.prsquared)
        all_coefs = {k: float(v) for k, v in logit_res.params.items()}

        ff_block = func_form_block or {
            "model": "logit",
            "interpretation": "Log-odds coefficients from binary logistic regression. "
                            "Positive coefficient means higher probability of crisis."
        }

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": inference_canonical["spec_id"],
                       "params": inference_canonical.get("params", {})},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"panel_fixed_effects": design_audit},
            axis_block_name="functional_form",
            axis_block=ff_block,
            notes=notes if notes else None,
        )

        results.append({
            "paper_id": PAPER_ID, "spec_run_id": run_id,
            "spec_id": spec_id, "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var, "treatment_var": treatment_var,
            "coefficient": coef_val, "std_error": se_val, "p_value": pval,
            "ci_lower": ci_lower, "ci_upper": ci_upper,
            "n_obs": nobs, "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc, "fixed_effects": "none",
            "controls_desc": controls_desc, "cluster_var": cluster_var_name,
            "run_success": 1, "run_error": ""
        })
        return run_id, coef_val, se_val, pval, nobs

    except Exception as e:
        err_msg = str(e)[:240]
        payload = make_failure_payload(
            error=err_msg,
            error_details=error_details_from_exception(e, stage="logit_estimation"),
            software=SW_BLOCK, surface_hash=SURFACE_HASH
        )
        results.append({
            "paper_id": PAPER_ID, "spec_run_id": run_id,
            "spec_id": spec_id, "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var, "treatment_var": treatment_var,
            "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "n_obs": np.nan, "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc, "fixed_effects": "none",
            "controls_desc": controls_desc, "cluster_var": cluster_var_name,
            "run_success": 0, "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


def run_probit_spec(spec_id, spec_tree_path, baseline_group_id,
                    outcome_var, treatment_var, rhs_vars,
                    data, sample_desc, controls_desc,
                    design_audit, inference_canonical,
                    use_constant=True, cluster_var_name="",
                    notes=""):
    """Run probit model."""
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        reg_vars = [outcome_var] + rhs_vars
        df_reg = data[reg_vars].dropna().copy()

        rhs_str = " + ".join(rhs_vars)
        if use_constant:
            formula = f"{outcome_var} ~ {rhs_str}"
        else:
            formula = f"{outcome_var} ~ {rhs_str} - 1"

        probit_res = smf.probit(formula, data=df_reg).fit(cov_type='HC2', disp=0)

        coef_val = float(probit_res.params[treatment_var])
        se_val = float(probit_res.bse[treatment_var])
        pval = float(probit_res.pvalues[treatment_var])
        try:
            ci = probit_res.conf_int()
            ci_lower = float(ci.loc[treatment_var, 0])
            ci_upper = float(ci.loc[treatment_var, 1])
        except:
            ci_lower, ci_upper = np.nan, np.nan
        nobs = int(probit_res.nobs)
        r2 = float(probit_res.prsquared)
        all_coefs = {k: float(v) for k, v in probit_res.params.items()}

        ff_block = {
            "model": "probit",
            "interpretation": "Probit coefficients from binary probit regression. "
                            "Positive coefficient means higher probability of crisis."
        }

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": inference_canonical["spec_id"],
                       "params": inference_canonical.get("params", {})},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"panel_fixed_effects": design_audit},
            axis_block_name="functional_form",
            axis_block=ff_block,
            notes=notes if notes else None,
        )

        results.append({
            "paper_id": PAPER_ID, "spec_run_id": run_id,
            "spec_id": spec_id, "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var, "treatment_var": treatment_var,
            "coefficient": coef_val, "std_error": se_val, "p_value": pval,
            "ci_lower": ci_lower, "ci_upper": ci_upper,
            "n_obs": nobs, "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc, "fixed_effects": "none",
            "controls_desc": controls_desc, "cluster_var": cluster_var_name,
            "run_success": 1, "run_error": ""
        })
        return run_id, coef_val, se_val, pval, nobs

    except Exception as e:
        err_msg = str(e)[:240]
        payload = make_failure_payload(
            error=err_msg,
            error_details=error_details_from_exception(e, stage="probit_estimation"),
            software=SW_BLOCK, surface_hash=SURFACE_HASH
        )
        results.append({
            "paper_id": PAPER_ID, "spec_run_id": run_id,
            "spec_id": spec_id, "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var, "treatment_var": treatment_var,
            "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "n_obs": np.nan, "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc, "fixed_effects": "none",
            "controls_desc": controls_desc, "cluster_var": cluster_var_name,
            "run_success": 0, "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


# ============================================================
# INFERENCE HELPER
# ============================================================
def add_inference_row(base_run_id, spec_id, spec_tree_path, baseline_group_id,
                      outcome_var, treatment_var, rhs_vars, data,
                      use_constant, infer_vcov, cluster_col=None,
                      fe_vars=None, notes=""):
    """Recompute inference under an alternative variance estimator."""
    global spec_run_counter
    spec_run_counter += 1
    infer_run_id = f"{PAPER_ID}_infer_{spec_run_counter:03d}"

    try:
        if fe_vars:
            rhs_str = " + ".join(rhs_vars)
            fe_str = " + ".join(fe_vars)
            if use_constant:
                formula = f"{outcome_var} ~ {rhs_str} | {fe_str}"
            else:
                formula = f"{outcome_var} ~ {rhs_str} - 1 | {fe_str}"
            m = pf.feols(formula, data=data, vcov=infer_vcov)
            coef_val = float(m.coef().get(treatment_var, np.nan))
            se_val = float(m.se().get(treatment_var, np.nan))
            pval = float(m.pvalue().get(treatment_var, np.nan))
            try:
                ci = m.confint()
                ci_lower = float(ci.loc[treatment_var, ci.columns[0]])
                ci_upper = float(ci.loc[treatment_var, ci.columns[1]])
            except:
                ci_lower, ci_upper = np.nan, np.nan
            nobs = int(m._N)
            try:
                r2 = float(m._r2)
            except:
                r2 = np.nan
            all_coefs = {k: float(v) for k, v in m.coef().items()}
        else:
            reg_vars = [outcome_var] + rhs_vars
            if cluster_col:
                reg_vars_ext = reg_vars + [cluster_col]
            else:
                reg_vars_ext = reg_vars
            df_reg = data[[c for c in reg_vars_ext if c in data.columns]].dropna().copy()

            y = df_reg[outcome_var].values
            if use_constant:
                X = sm.add_constant(df_reg[rhs_vars].values)
                x_names = ['Intercept'] + rhs_vars
            else:
                X = df_reg[rhs_vars].values
                x_names = rhs_vars

            model = sm.OLS(y, X)
            if cluster_col:
                res = model.fit(cov_type='cluster', cov_kwds={'groups': df_reg[cluster_col]})
            elif infer_vcov == 'HC1':
                res = model.fit(cov_type='HC1')
            else:
                res = model.fit(cov_type='HC2')

            tidx = x_names.index(treatment_var)
            coef_val = float(res.params[tidx])
            se_val = float(res.bse[tidx])
            pval = float(res.pvalues[tidx])
            ci = res.conf_int()
            ci_lower = float(ci[tidx, 0])
            ci_upper = float(ci[tidx, 1])
            nobs = int(res.nobs)
            r2 = float(res.rsquared)
            all_coefs = {x_names[i]: float(res.params[i]) for i in range(len(x_names))}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": spec_id, "params": {}},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            notes=notes if notes else None,
        )

        inference_results.append({
            "paper_id": PAPER_ID, "inference_run_id": infer_run_id,
            "spec_run_id": base_run_id, "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "coefficient": coef_val, "std_error": se_val, "p_value": pval,
            "ci_lower": ci_lower, "ci_upper": ci_upper,
            "n_obs": nobs, "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 1, "run_error": ""
        })
    except Exception as e:
        err_msg = str(e)[:240]
        payload = make_failure_payload(
            error=err_msg,
            error_details=error_details_from_exception(e, stage="inference"),
            software=SW_BLOCK, surface_hash=SURFACE_HASH
        )
        inference_results.append({
            "paper_id": PAPER_ID, "inference_run_id": infer_run_id,
            "spec_run_id": base_run_id, "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "n_obs": np.nan, "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 0, "run_error": err_msg
        })


# ============================================================
# G1: BANKING CRISIS CONTAGION
# ============================================================
print("\n=== G1: Banking Crisis Contagion ===")

# --- G1 Baseline: Eq5 with public debt (1824-2009) ---
# bank ~ develop1 + develop2 + bank_move + debt_move + center + public
# HC2 robust SE, no constant, focal = center
g1_base_rhs = ['develop1', 'develop2', 'bank_move', 'debt_move', 'center', 'public']
g1_base_run_id, *_ = run_ols_spec(
    spec_id="baseline__eq5_bank_public",
    spec_tree_path="specification_tree/methods/panel_fixed_effects.md",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=g1_base_rhs,
    data=df_1824_pub, sample_desc="1824-2009, non-missing public debt",
    controls_desc="develop1,develop2,bank_move,debt_move,public",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    use_constant=False,
    notes="Baseline Eq5: bank ~ develop1+develop2+bank_move+debt_move+center+public, HC2 SE"
)
print(f"  Baseline Eq5 (bank, 1824 w/ public): run_id={g1_base_run_id}")

# --- G1 Eq1 without public (1824-2009) ---
g1_eq1_rhs = ['develop1', 'develop2', 'bank_move', 'debt_move', 'center']
g1_eq1_run_id, *_ = run_ols_spec(
    spec_id="rc/controls/loo/drop_public",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=g1_eq1_rhs,
    data=df_1824, sample_desc="1824-2009, full sample",
    controls_desc="develop1,develop2,bank_move,debt_move",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    use_constant=False,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_public", "family": "loo",
                "dropped": ["public"], "n_controls": 4},
    notes="Eq1: bank ~ develop1+develop2+bank_move+debt_move+center (no public)"
)

# --- G1 LOO: drop bank_move ---
g1_loo_rhs = ['develop1', 'develop2', 'debt_move', 'center', 'public']
run_ols_spec(
    spec_id="rc/controls/loo/drop_bank_move",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=g1_loo_rhs,
    data=df_1824_pub, sample_desc="1824-2009, non-missing public",
    controls_desc="develop1,develop2,debt_move,center,public",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    use_constant=False,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_bank_move", "family": "loo",
                "dropped": ["bank_move"], "n_controls": 4},
)

# --- G1 LOO: drop debt_move ---
run_ols_spec(
    spec_id="rc/controls/loo/drop_debt_move",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=['develop1', 'develop2', 'bank_move', 'center', 'public'],
    data=df_1824_pub, sample_desc="1824-2009, non-missing public",
    controls_desc="develop1,develop2,bank_move,center,public",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    use_constant=False,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_debt_move", "family": "loo",
                "dropped": ["debt_move"], "n_controls": 4},
)

# --- G1 LOO: drop develop1 ---
run_ols_spec(
    spec_id="rc/controls/loo/drop_develop1",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=['develop2', 'bank_move', 'debt_move', 'center', 'public'],
    data=df_1824_pub, sample_desc="1824-2009, non-missing public",
    controls_desc="develop2,bank_move,debt_move,center,public",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    use_constant=False,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_develop1", "family": "loo",
                "dropped": ["develop1"], "n_controls": 4},
)

# --- G1 LOO: drop develop2 ---
run_ols_spec(
    spec_id="rc/controls/loo/drop_develop2",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=['develop1', 'bank_move', 'debt_move', 'center', 'public'],
    data=df_1824_pub, sample_desc="1824-2009, non-missing public",
    controls_desc="develop1,bank_move,debt_move,center,public",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    use_constant=False,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_develop2", "family": "loo",
                "dropped": ["develop2"], "n_controls": 4},
)

# --- G1 Sample period: 1900-2009 ---
run_ols_spec(
    spec_id="rc/sample/period/1900_2009",
    spec_tree_path="specification_tree/modules/robustness/sample.md#period",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=g1_base_rhs,
    data=df_1900_pub, sample_desc="1900-2009, non-missing public",
    controls_desc="develop1,develop2,bank_move,debt_move,public",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    use_constant=False,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/period/1900_2009", "period": "1900-2009"},
)

# --- G1 Sample period: 1946-2009 ---
run_ols_spec(
    spec_id="rc/sample/period/1946_2009",
    spec_tree_path="specification_tree/modules/robustness/sample.md#period",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=g1_base_rhs,
    data=df_1946_pub, sample_desc="1946-2009, non-missing public",
    controls_desc="develop1,develop2,bank_move,debt_move,public",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    use_constant=False,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/period/1946_2009", "period": "1946-2009"},
)

# --- G1 Sample period: 1900-2009 without public ---
run_ols_spec(
    spec_id="rc/joint/period_1900_no_public",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=g1_eq1_rhs,
    data=df_1900, sample_desc="1900-2009, full sample",
    controls_desc="develop1,develop2,bank_move,debt_move",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    use_constant=False,
    axis_block_name="joint",
    axis_block={"changes": ["period=1900-2009", "drop_public"]},
    notes="Eq1 1900-2009: bank ~ develop1+develop2+bank_move+debt_move+center"
)

# --- G1 Sample period: 1946-2009 without public ---
run_ols_spec(
    spec_id="rc/joint/period_1946_no_public",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=g1_eq1_rhs,
    data=df_1946, sample_desc="1946-2009, full sample",
    controls_desc="develop1,develop2,bank_move,debt_move",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    use_constant=False,
    axis_block_name="joint",
    axis_block={"changes": ["period=1946-2009", "drop_public"]},
    notes="Eq1 1946-2009: bank ~ develop1+develop2+bank_move+debt_move+center"
)

# --- G1 Advanced only ---
run_ols_spec(
    spec_id="rc/sample/subset/advanced_only",
    spec_tree_path="specification_tree/modules/robustness/sample.md#subset",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=['bank_move', 'debt_move', 'center', 'public'],
    data=df_1824_pub[df_1824_pub['develop2'] == 1],
    sample_desc="1824-2009, advanced economies only",
    controls_desc="bank_move,debt_move,public",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    use_constant=True,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subset/advanced_only", "subset": "advanced economies"},
)

# --- G1 Emerging only ---
run_ols_spec(
    spec_id="rc/sample/subset/emerging_only",
    spec_tree_path="specification_tree/modules/robustness/sample.md#subset",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=['bank_move', 'debt_move', 'center', 'public'],
    data=df_1824_pub[df_1824_pub['develop1'] == 1],
    sample_desc="1824-2009, emerging markets only",
    controls_desc="bank_move,debt_move,public",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    use_constant=True,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subset/emerging_only", "subset": "emerging markets"},
)

# --- G1 Drop financial centers (UK/US) ---
df_no_fc = df_1824_pub[~df_1824_pub['country'].isin(['UK', 'US'])].copy()
run_ols_spec(
    spec_id="rc/sample/subset/drop_financial_centers",
    spec_tree_path="specification_tree/modules/robustness/sample.md#subset",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=g1_base_rhs,
    data=df_no_fc,
    sample_desc="1824-2009, excluding UK and US",
    controls_desc="develop1,develop2,bank_move,debt_move,public",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    use_constant=False,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subset/drop_financial_centers",
                "dropped_countries": ["UK", "US"]},
)

# --- G1 Trim outliers (1st/99th percentile of bank) ---
# For binary outcome, trimming doesn't apply directly.
# Instead, trim on bank_move or public
q01 = df_1824_pub['public'].quantile(0.01)
q99 = df_1824_pub['public'].quantile(0.99)
df_trimmed = df_1824_pub[(df_1824_pub['public'] >= q01) & (df_1824_pub['public'] <= q99)]
run_ols_spec(
    spec_id="rc/sample/outliers/trim_public_1_99",
    spec_tree_path="specification_tree/modules/robustness/sample.md#outliers",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=g1_base_rhs,
    data=df_trimmed,
    sample_desc="1824-2009, public debt trimmed 1st-99th pctile",
    controls_desc="develop1,develop2,bank_move,debt_move,public",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    use_constant=False,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/trim_public_1_99",
                "trimming": "public debt 1st-99th percentile"},
)

# --- G1 Logit ---
run_logit_spec(
    spec_id="rc/form/model/logit",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md#model",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=g1_base_rhs,
    data=df_1824_pub,
    sample_desc="1824-2009, non-missing public",
    controls_desc="develop1,develop2,bank_move,debt_move,public",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    use_constant=False,
    notes="Logit: bank ~ develop1+develop2+bank_move+debt_move+center+public"
)

# --- G1 Probit ---
run_probit_spec(
    spec_id="rc/form/model/probit",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md#model",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=g1_base_rhs,
    data=df_1824_pub,
    sample_desc="1824-2009, non-missing public",
    controls_desc="develop1,develop2,bank_move,debt_move,public",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    use_constant=False,
    notes="Probit: bank ~ develop1+develop2+bank_move+debt_move+center+public"
)

# --- G1 Logit without public ---
run_logit_spec(
    spec_id="rc/joint/logit_no_public",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=g1_eq1_rhs,
    data=df_1824,
    sample_desc="1824-2009, full sample",
    controls_desc="develop1,develop2,bank_move,debt_move",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    use_constant=False,
    notes="Logit Eq3: bank ~ develop1+develop2+bank_move+debt_move+center"
)

# --- G1 Logit 1900-2009 ---
run_logit_spec(
    spec_id="rc/joint/logit_1900",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=g1_base_rhs,
    data=df_1900_pub,
    sample_desc="1900-2009, non-missing public",
    controls_desc="develop1,develop2,bank_move,debt_move,public",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    use_constant=False,
    notes="Logit 1900-2009 with public"
)

# --- G1 Logit 1946-2009 ---
run_logit_spec(
    spec_id="rc/joint/logit_1946",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=g1_base_rhs,
    data=df_1946_pub,
    sample_desc="1946-2009, non-missing public",
    controls_desc="develop1,develop2,bank_move,debt_move,public",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    use_constant=False,
    notes="Logit 1946-2009 with public"
)

# --- G1 Add country FE ---
g1_country_fe_run_id, *_ = run_ols_spec(
    spec_id="rc/fe/add_country",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#add",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=['bank_move', 'debt_move', 'center', 'public'],
    data=df_1824_pub,
    sample_desc="1824-2009, non-missing public, country FE",
    controls_desc="bank_move,debt_move,public",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fe_vars=["country_id"],
    use_constant=True,
    vcov_spec="hetero",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/add_country", "added": ["country"]},
    notes="Country FE absorbs develop dummies"
)

# --- G1 Add year FE ---
run_ols_spec(
    spec_id="rc/fe/add_year",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#add",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=['develop1', 'develop2', 'bank_move', 'debt_move', 'center', 'public'],
    data=df_1824_pub,
    sample_desc="1824-2009, non-missing public, year FE",
    controls_desc="develop1,develop2,bank_move,debt_move,public",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fe_vars=["year"],
    use_constant=False,
    vcov_spec="hetero",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/add_year", "added": ["year"]},
    notes="Year FE added. center variable may be collinear with year FE."
)

# --- G1 Add country + year FE ---
run_ols_spec(
    spec_id="rc/fe/add_country_year",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#add",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=['bank_move', 'debt_move', 'center', 'public'],
    data=df_1824_pub,
    sample_desc="1824-2009, non-missing public, country+year FE",
    controls_desc="bank_move,debt_move,public",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fe_vars=["country_id", "year"],
    use_constant=True,
    vcov_spec="hetero",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/add_country_year", "added": ["country", "year"]},
    notes="Country+Year FE. center is time-varying only, may lose variation."
)

# --- G1 Add region FE ---
run_ols_spec(
    spec_id="rc/fe/add_region",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#add",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=['bank_move', 'debt_move', 'center', 'public'],
    data=df_1824_pub,
    sample_desc="1824-2009, non-missing public, region FE",
    controls_desc="bank_move,debt_move,public",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fe_vars=["region"],
    use_constant=True,
    vcov_spec="hetero",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/add_region", "added": ["region"]},
    notes="Region FE replaces development dummies"
)

# --- G1 Country FE + 1900-2009 ---
run_ols_spec(
    spec_id="rc/joint/fe_country_1900",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=['bank_move', 'debt_move', 'center', 'public'],
    data=df_1900_pub,
    sample_desc="1900-2009, non-missing public, country FE",
    controls_desc="bank_move,debt_move,public",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fe_vars=["country_id"],
    use_constant=True,
    vcov_spec="hetero",
    axis_block_name="joint",
    axis_block={"changes": ["add_country_fe", "period=1900-2009"]},
)

# --- G1 Country FE + 1946-2009 ---
run_ols_spec(
    spec_id="rc/joint/fe_country_1946",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=['bank_move', 'debt_move', 'center', 'public'],
    data=df_1946_pub,
    sample_desc="1946-2009, non-missing public, country FE",
    controls_desc="bank_move,debt_move,public",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fe_vars=["country_id"],
    use_constant=True,
    vcov_spec="hetero",
    axis_block_name="joint",
    axis_block={"changes": ["add_country_fe", "period=1946-2009"]},
)

# --- G1 Country FE, no public, 1824 ---
run_ols_spec(
    spec_id="rc/joint/fe_country_no_public",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=['bank_move', 'debt_move', 'center'],
    data=df_1824,
    sample_desc="1824-2009, full sample, country FE",
    controls_desc="bank_move,debt_move",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fe_vars=["country_id"],
    use_constant=True,
    vcov_spec="hetero",
    axis_block_name="joint",
    axis_block={"changes": ["add_country_fe", "drop_public"]},
)

# --- G1 Probit 1900-2009 ---
run_probit_spec(
    spec_id="rc/joint/probit_1900",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=g1_base_rhs,
    data=df_1900_pub,
    sample_desc="1900-2009, non-missing public",
    controls_desc="develop1,develop2,bank_move,debt_move,public",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    use_constant=False,
    notes="Probit 1900-2009 with public"
)

# --- G1 Advanced, no public ---
run_ols_spec(
    spec_id="rc/joint/advanced_no_public",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=['bank_move', 'debt_move', 'center'],
    data=df_1824[df_1824['develop2'] == 1],
    sample_desc="1824-2009, advanced only, no public",
    controls_desc="bank_move,debt_move",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    use_constant=True,
    axis_block_name="joint",
    axis_block={"changes": ["advanced_only", "drop_public"]},
)

# --- G1 Emerging, no public ---
run_ols_spec(
    spec_id="rc/joint/emerging_no_public",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=['bank_move', 'debt_move', 'center'],
    data=df_1824[df_1824['develop1'] == 1],
    sample_desc="1824-2009, emerging only, no public",
    controls_desc="bank_move,debt_move",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    use_constant=True,
    axis_block_name="joint",
    axis_block={"changes": ["emerging_only", "drop_public"]},
)

# --- G1 1900 Advanced only ---
run_ols_spec(
    spec_id="rc/joint/advanced_1900",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=['bank_move', 'debt_move', 'center', 'public'],
    data=df_1900_pub[df_1900_pub['develop2'] == 1],
    sample_desc="1900-2009, advanced only, with public",
    controls_desc="bank_move,debt_move,public",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    use_constant=True,
    axis_block_name="joint",
    axis_block={"changes": ["advanced_only", "period=1900-2009"]},
)

# --- G1 1900 Emerging only ---
run_ols_spec(
    spec_id="rc/joint/emerging_1900",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=['bank_move', 'debt_move', 'center', 'public'],
    data=df_1900_pub[df_1900_pub['develop1'] == 1],
    sample_desc="1900-2009, emerging only, with public",
    controls_desc="bank_move,debt_move,public",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    use_constant=True,
    axis_block_name="joint",
    axis_block={"changes": ["emerging_only", "period=1900-2009"]},
)

# --- G1 LOO: drop center (treatment var becomes bank_move) ---
# This is a sensitivity check, keeping center in the model but making bank_move focal
# Actually, dropping the treatment var doesn't make sense. Instead, drop center
# from controls and use bank_move as an alternative treatment.
# Let's skip this and instead add more period x controls variants.

# --- G1 Year FE + 1900 ---
run_ols_spec(
    spec_id="rc/joint/fe_year_1900",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=['develop1', 'develop2', 'bank_move', 'debt_move', 'center', 'public'],
    data=df_1900_pub,
    sample_desc="1900-2009, non-missing public, year FE",
    controls_desc="develop1,develop2,bank_move,debt_move,public",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fe_vars=["year"],
    use_constant=False,
    vcov_spec="hetero",
    axis_block_name="joint",
    axis_block={"changes": ["add_year_fe", "period=1900-2009"]},
)

# --- G1 Region FE + 1900 ---
run_ols_spec(
    spec_id="rc/joint/fe_region_1900",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=['bank_move', 'debt_move', 'center', 'public'],
    data=df_1900_pub,
    sample_desc="1900-2009, non-missing public, region FE",
    controls_desc="bank_move,debt_move,public",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fe_vars=["region"],
    use_constant=True,
    vcov_spec="hetero",
    axis_block_name="joint",
    axis_block={"changes": ["add_region_fe", "period=1900-2009"]},
)

# --- G1 Probit no public ---
run_probit_spec(
    spec_id="rc/joint/probit_no_public",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=g1_eq1_rhs,
    data=df_1824,
    sample_desc="1824-2009, full sample",
    controls_desc="develop1,develop2,bank_move,debt_move",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    use_constant=False,
    notes="Probit without public"
)

# --- G1 Country FE + country cluster ---
run_ols_spec(
    spec_id="rc/joint/fe_country_cluster",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=['bank_move', 'debt_move', 'center', 'public'],
    data=df_1824_pub,
    sample_desc="1824-2009, non-missing public, country FE + cluster",
    controls_desc="bank_move,debt_move,public",
    design_audit=G1_DESIGN_AUDIT,
    inference_canonical=G1_INFERENCE_CANONICAL,
    fe_vars=["country_id"],
    use_constant=True,
    vcov_spec={"CRV1": "country_id"},
    cluster_var_name="country_id",
    axis_block_name="joint",
    axis_block={"changes": ["add_country_fe", "cluster_country"]},
)

print(f"  G1 specs so far: {len([r for r in results if r['baseline_group_id'] == 'G1'])}")


# ============================================================
# G1 INFERENCE VARIANTS
# ============================================================
print("\n=== G1 Inference Variants ===")

# HC1 for baseline
add_inference_row(
    base_run_id=g1_base_run_id,
    spec_id="infer/se/hc/hc1",
    spec_tree_path="specification_tree/modules/inference/robust_se.md#hc1",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=g1_base_rhs,
    data=df_1824_pub,
    use_constant=False,
    infer_vcov='HC1',
    notes="HC1 robust SE instead of HC2"
)

# Cluster by country for baseline
add_inference_row(
    base_run_id=g1_base_run_id,
    spec_id="infer/se/cluster/country",
    spec_tree_path="specification_tree/modules/inference/cluster_se.md",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=g1_base_rhs,
    data=df_1824_pub,
    use_constant=False,
    infer_vcov='cluster',
    cluster_col='country_id',
    notes="Cluster SE by country"
)

# HC1 for eq1 (no public)
add_inference_row(
    base_run_id=g1_eq1_run_id,
    spec_id="infer/se/hc/hc1",
    spec_tree_path="specification_tree/modules/inference/robust_se.md#hc1",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=g1_eq1_rhs,
    data=df_1824,
    use_constant=False,
    infer_vcov='HC1',
    notes="HC1 for eq1 without public"
)

# Cluster by country for eq1
add_inference_row(
    base_run_id=g1_eq1_run_id,
    spec_id="infer/se/cluster/country",
    spec_tree_path="specification_tree/modules/inference/cluster_se.md",
    baseline_group_id="G1",
    outcome_var="bank", treatment_var="center",
    rhs_vars=g1_eq1_rhs,
    data=df_1824,
    use_constant=False,
    infer_vcov='cluster',
    cluster_col='country_id',
    notes="Cluster SE by country for eq1"
)


# ============================================================
# G2: EXTERNAL DEBT CRISIS CONTAGION
# ============================================================
print("\n=== G2: External Debt Crisis Contagion ===")

# --- G2 Baseline: Eq6 with public debt (1824-2009) ---
# debt ~ develop1 + develop2 + bank_move + debt_move + center + public
# focal = debt_move
g2_base_rhs = ['develop1', 'develop2', 'bank_move', 'debt_move', 'center', 'public']
g2_base_run_id, *_ = run_ols_spec(
    spec_id="baseline__eq6_debt_public",
    spec_tree_path="specification_tree/methods/panel_fixed_effects.md",
    baseline_group_id="G2",
    outcome_var="debt", treatment_var="debt_move",
    rhs_vars=g2_base_rhs,
    data=df_1824_pub, sample_desc="1824-2009, non-missing public",
    controls_desc="develop1,develop2,bank_move,center,public",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    use_constant=False,
    notes="Baseline Eq6: debt ~ develop1+develop2+bank_move+debt_move+center+public, HC2 SE"
)
print(f"  Baseline Eq6 (debt, 1824 w/ public): run_id={g2_base_run_id}")

# --- G2 Eq2 without develop dummies, without public (1824-2009) ---
# In 1824 period: debt ~ develop1+develop2+bank_move+debt_move+center
g2_eq2_rhs_full = ['develop1', 'develop2', 'bank_move', 'debt_move', 'center']
g2_eq2_run_id, *_ = run_ols_spec(
    spec_id="rc/controls/loo/drop_public_g2",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G2",
    outcome_var="debt", treatment_var="debt_move",
    rhs_vars=g2_eq2_rhs_full,
    data=df_1824, sample_desc="1824-2009, full sample",
    controls_desc="develop1,develop2,bank_move,center",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    use_constant=False,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/single/add_public", "family": "loo",
                "dropped": ["public"], "n_controls": 4},
    notes="Eq2: debt ~ develop1+develop2+bank_move+debt_move+center"
)

# --- G2 LOO: drop bank_move ---
run_ols_spec(
    spec_id="rc/controls/loo/drop_bank_move",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G2",
    outcome_var="debt", treatment_var="debt_move",
    rhs_vars=['develop1', 'develop2', 'debt_move', 'center', 'public'],
    data=df_1824_pub, sample_desc="1824-2009, non-missing public",
    controls_desc="develop1,develop2,center,public",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    use_constant=False,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_bank_move", "family": "loo",
                "dropped": ["bank_move"], "n_controls": 4},
)

# --- G2 LOO: drop center ---
run_ols_spec(
    spec_id="rc/controls/loo/drop_center",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G2",
    outcome_var="debt", treatment_var="debt_move",
    rhs_vars=['develop1', 'develop2', 'bank_move', 'debt_move', 'public'],
    data=df_1824_pub, sample_desc="1824-2009, non-missing public",
    controls_desc="develop1,develop2,bank_move,public",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    use_constant=False,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_center", "family": "loo",
                "dropped": ["center"], "n_controls": 4},
)

# --- G2 LOO: drop develop1 ---
run_ols_spec(
    spec_id="rc/controls/loo/drop_develop1",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G2",
    outcome_var="debt", treatment_var="debt_move",
    rhs_vars=['develop2', 'bank_move', 'debt_move', 'center', 'public'],
    data=df_1824_pub, sample_desc="1824-2009, non-missing public",
    controls_desc="develop2,bank_move,center,public",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    use_constant=False,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_develop1", "family": "loo",
                "dropped": ["develop1"], "n_controls": 4},
)

# --- G2 LOO: drop develop2 ---
run_ols_spec(
    spec_id="rc/controls/loo/drop_develop2",
    spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
    baseline_group_id="G2",
    outcome_var="debt", treatment_var="debt_move",
    rhs_vars=['develop1', 'bank_move', 'debt_move', 'center', 'public'],
    data=df_1824_pub, sample_desc="1824-2009, non-missing public",
    controls_desc="develop1,bank_move,center,public",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    use_constant=False,
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/loo/drop_develop2", "family": "loo",
                "dropped": ["develop2"], "n_controls": 4},
)

# --- G2 Sample: 1900-2009 ---
# Note: TSP 1900 Eq2 uses C instead of develop dummies
run_ols_spec(
    spec_id="rc/sample/period/1900_2009",
    spec_tree_path="specification_tree/modules/robustness/sample.md#period",
    baseline_group_id="G2",
    outcome_var="debt", treatment_var="debt_move",
    rhs_vars=g2_base_rhs,
    data=df_1900_pub, sample_desc="1900-2009, non-missing public",
    controls_desc="develop1,develop2,bank_move,center,public",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    use_constant=False,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/period/1900_2009", "period": "1900-2009"},
)

# --- G2 Sample: 1946-2009 ---
run_ols_spec(
    spec_id="rc/sample/period/1946_2009",
    spec_tree_path="specification_tree/modules/robustness/sample.md#period",
    baseline_group_id="G2",
    outcome_var="debt", treatment_var="debt_move",
    rhs_vars=g2_base_rhs,
    data=df_1946_pub, sample_desc="1946-2009, non-missing public",
    controls_desc="develop1,develop2,bank_move,center,public",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    use_constant=False,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/period/1946_2009", "period": "1946-2009"},
)

# --- G2 1900 no public (matching TSP Eq2 form: C + bank_move + debt_move + center) ---
run_ols_spec(
    spec_id="rc/joint/period_1900_no_public",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G2",
    outcome_var="debt", treatment_var="debt_move",
    rhs_vars=['bank_move', 'debt_move', 'center'],
    data=df_1900, sample_desc="1900-2009, full sample",
    controls_desc="bank_move,center",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    use_constant=True,
    axis_block_name="joint",
    axis_block={"changes": ["period=1900-2009", "drop_public", "constant_instead_of_develop"]},
    notes="Matches TSP 1900 Eq2: debt ~ C + bank_move + debt_move + center"
)

# --- G2 1946 no public ---
run_ols_spec(
    spec_id="rc/joint/period_1946_no_public",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G2",
    outcome_var="debt", treatment_var="debt_move",
    rhs_vars=['bank_move', 'debt_move', 'center'],
    data=df_1946, sample_desc="1946-2009, full sample",
    controls_desc="bank_move,center",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    use_constant=True,
    axis_block_name="joint",
    axis_block={"changes": ["period=1946-2009", "drop_public", "constant_instead_of_develop"]},
    notes="Matches TSP 1947 Eq2: debt ~ C + bank_move + debt_move + center"
)

# --- G2 Advanced only ---
run_ols_spec(
    spec_id="rc/sample/subset/advanced_only",
    spec_tree_path="specification_tree/modules/robustness/sample.md#subset",
    baseline_group_id="G2",
    outcome_var="debt", treatment_var="debt_move",
    rhs_vars=['bank_move', 'debt_move', 'center', 'public'],
    data=df_1824_pub[df_1824_pub['develop2'] == 1],
    sample_desc="1824-2009, advanced only",
    controls_desc="bank_move,center,public",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    use_constant=True,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subset/advanced_only", "subset": "advanced"},
)

# --- G2 Emerging only ---
run_ols_spec(
    spec_id="rc/sample/subset/emerging_only",
    spec_tree_path="specification_tree/modules/robustness/sample.md#subset",
    baseline_group_id="G2",
    outcome_var="debt", treatment_var="debt_move",
    rhs_vars=['bank_move', 'debt_move', 'center', 'public'],
    data=df_1824_pub[df_1824_pub['develop1'] == 1],
    sample_desc="1824-2009, emerging only",
    controls_desc="bank_move,center,public",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    use_constant=True,
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/subset/emerging_only", "subset": "emerging"},
)

# --- G2 Logit ---
run_logit_spec(
    spec_id="rc/form/model/logit",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md#model",
    baseline_group_id="G2",
    outcome_var="debt", treatment_var="debt_move",
    rhs_vars=g2_base_rhs,
    data=df_1824_pub,
    sample_desc="1824-2009, non-missing public",
    controls_desc="develop1,develop2,bank_move,center,public",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    use_constant=False,
    notes="Logit for debt crisis"
)

# --- G2 Probit ---
run_probit_spec(
    spec_id="rc/form/model/probit",
    spec_tree_path="specification_tree/modules/robustness/functional_form.md#model",
    baseline_group_id="G2",
    outcome_var="debt", treatment_var="debt_move",
    rhs_vars=g2_base_rhs,
    data=df_1824_pub,
    sample_desc="1824-2009, non-missing public",
    controls_desc="develop1,develop2,bank_move,center,public",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    use_constant=False,
    notes="Probit for debt crisis"
)

# --- G2 Add country FE ---
g2_country_fe_run_id, *_ = run_ols_spec(
    spec_id="rc/fe/add_country",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#add",
    baseline_group_id="G2",
    outcome_var="debt", treatment_var="debt_move",
    rhs_vars=['bank_move', 'debt_move', 'center', 'public'],
    data=df_1824_pub,
    sample_desc="1824-2009, non-missing public, country FE",
    controls_desc="bank_move,center,public",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    fe_vars=["country_id"],
    use_constant=True,
    vcov_spec="hetero",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/add_country", "added": ["country"]},
)

# --- G2 Add year FE ---
run_ols_spec(
    spec_id="rc/fe/add_year",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#add",
    baseline_group_id="G2",
    outcome_var="debt", treatment_var="debt_move",
    rhs_vars=['develop1', 'develop2', 'bank_move', 'debt_move', 'center', 'public'],
    data=df_1824_pub,
    sample_desc="1824-2009, non-missing public, year FE",
    controls_desc="develop1,develop2,bank_move,center,public",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    fe_vars=["year"],
    use_constant=False,
    vcov_spec="hetero",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/add_year", "added": ["year"]},
)

# --- G2 Country FE + 1900 ---
run_ols_spec(
    spec_id="rc/joint/fe_country_1900",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G2",
    outcome_var="debt", treatment_var="debt_move",
    rhs_vars=['bank_move', 'debt_move', 'center', 'public'],
    data=df_1900_pub,
    sample_desc="1900-2009, non-missing public, country FE",
    controls_desc="bank_move,center,public",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    fe_vars=["country_id"],
    use_constant=True,
    vcov_spec="hetero",
    axis_block_name="joint",
    axis_block={"changes": ["add_country_fe", "period=1900-2009"]},
)

# --- G2 Country FE + 1946 ---
run_ols_spec(
    spec_id="rc/joint/fe_country_1946",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G2",
    outcome_var="debt", treatment_var="debt_move",
    rhs_vars=['bank_move', 'debt_move', 'center', 'public'],
    data=df_1946_pub,
    sample_desc="1946-2009, non-missing public, country FE",
    controls_desc="bank_move,center,public",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    fe_vars=["country_id"],
    use_constant=True,
    vcov_spec="hetero",
    axis_block_name="joint",
    axis_block={"changes": ["add_country_fe", "period=1946-2009"]},
)

# --- G2 Logit 1900 ---
run_logit_spec(
    spec_id="rc/joint/logit_1900",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G2",
    outcome_var="debt", treatment_var="debt_move",
    rhs_vars=g2_base_rhs,
    data=df_1900_pub,
    sample_desc="1900-2009, non-missing public",
    controls_desc="develop1,develop2,bank_move,center,public",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    use_constant=False,
    notes="Logit 1900-2009 for debt"
)

# --- G2 Logit no public ---
run_logit_spec(
    spec_id="rc/joint/logit_no_public",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G2",
    outcome_var="debt", treatment_var="debt_move",
    rhs_vars=g2_eq2_rhs_full,
    data=df_1824,
    sample_desc="1824-2009, full sample",
    controls_desc="develop1,develop2,bank_move,center",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    use_constant=False,
    notes="Logit without public for debt"
)

# --- G2 Country FE + cluster ---
run_ols_spec(
    spec_id="rc/joint/fe_country_cluster",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G2",
    outcome_var="debt", treatment_var="debt_move",
    rhs_vars=['bank_move', 'debt_move', 'center', 'public'],
    data=df_1824_pub,
    sample_desc="1824-2009, non-missing public, country FE + cluster",
    controls_desc="bank_move,center,public",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    fe_vars=["country_id"],
    use_constant=True,
    vcov_spec={"CRV1": "country_id"},
    cluster_var_name="country_id",
    axis_block_name="joint",
    axis_block={"changes": ["add_country_fe", "cluster_country"]},
)

# --- G2 Probit 1900 ---
run_probit_spec(
    spec_id="rc/joint/probit_1900",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G2",
    outcome_var="debt", treatment_var="debt_move",
    rhs_vars=g2_base_rhs,
    data=df_1900_pub,
    sample_desc="1900-2009, non-missing public",
    controls_desc="develop1,develop2,bank_move,center,public",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    use_constant=False,
    notes="Probit 1900 for debt"
)

# --- G2 Region FE ---
run_ols_spec(
    spec_id="rc/fe/add_region",
    spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#add",
    baseline_group_id="G2",
    outcome_var="debt", treatment_var="debt_move",
    rhs_vars=['bank_move', 'debt_move', 'center', 'public'],
    data=df_1824_pub,
    sample_desc="1824-2009, non-missing public, region FE",
    controls_desc="bank_move,center,public",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    fe_vars=["region"],
    use_constant=True,
    vcov_spec="hetero",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/add_region", "added": ["region"]},
)

# --- G2 1900 emerging only ---
run_ols_spec(
    spec_id="rc/joint/emerging_1900",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G2",
    outcome_var="debt", treatment_var="debt_move",
    rhs_vars=['bank_move', 'debt_move', 'center', 'public'],
    data=df_1900_pub[df_1900_pub['develop1'] == 1],
    sample_desc="1900-2009, emerging only, with public",
    controls_desc="bank_move,center,public",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    use_constant=True,
    axis_block_name="joint",
    axis_block={"changes": ["emerging_only", "period=1900-2009"]},
)

# --- G2 Country FE, no public ---
run_ols_spec(
    spec_id="rc/joint/fe_country_no_public",
    spec_tree_path="specification_tree/modules/robustness/joint.md",
    baseline_group_id="G2",
    outcome_var="debt", treatment_var="debt_move",
    rhs_vars=['bank_move', 'debt_move', 'center'],
    data=df_1824,
    sample_desc="1824-2009, full sample, country FE",
    controls_desc="bank_move,center",
    design_audit=G2_DESIGN_AUDIT,
    inference_canonical=G2_INFERENCE_CANONICAL,
    fe_vars=["country_id"],
    use_constant=True,
    vcov_spec="hetero",
    axis_block_name="joint",
    axis_block={"changes": ["add_country_fe", "drop_public"]},
)

print(f"  G2 specs so far: {len([r for r in results if r['baseline_group_id'] == 'G2'])}")


# ============================================================
# G2 INFERENCE VARIANTS
# ============================================================
print("\n=== G2 Inference Variants ===")

# Cluster by country for baseline
add_inference_row(
    base_run_id=g2_base_run_id,
    spec_id="infer/se/cluster/country",
    spec_tree_path="specification_tree/modules/inference/cluster_se.md",
    baseline_group_id="G2",
    outcome_var="debt", treatment_var="debt_move",
    rhs_vars=g2_base_rhs,
    data=df_1824_pub,
    use_constant=False,
    infer_vcov='cluster',
    cluster_col='country_id',
    notes="Cluster SE by country for G2 baseline"
)

# HC1 for G2 Eq2
add_inference_row(
    base_run_id=g2_eq2_run_id,
    spec_id="infer/se/hc/hc1",
    spec_tree_path="specification_tree/modules/inference/robust_se.md#hc1",
    baseline_group_id="G2",
    outcome_var="debt", treatment_var="debt_move",
    rhs_vars=g2_eq2_rhs_full,
    data=df_1824,
    use_constant=False,
    infer_vcov='HC1',
    notes="HC1 for G2 eq2"
)


# ============================================================
# WRITE OUTPUTS
# ============================================================
print(f"\n=== Writing outputs ===")
print(f"  Total specification rows: {len(results)}")
print(f"  Total inference rows: {len(inference_results)}")

g1_count = len([r for r in results if r['baseline_group_id'] == 'G1'])
g2_count = len([r for r in results if r['baseline_group_id'] == 'G2'])
g1_success = len([r for r in results if r['baseline_group_id'] == 'G1' and r['run_success'] == 1])
g2_success = len([r for r in results if r['baseline_group_id'] == 'G2' and r['run_success'] == 1])
print(f"  G1: {g1_count} total ({g1_success} success)")
print(f"  G2: {g2_count} total ({g2_success} success)")

# Write specification_results.csv
spec_df = pd.DataFrame(results)
spec_df.to_csv(f"{OUTPUT_DIR}/specification_results.csv", index=False)

# Write inference_results.csv
if inference_results:
    infer_df = pd.DataFrame(inference_results)
    infer_df.to_csv(f"{OUTPUT_DIR}/inference_results.csv", index=False)

# Write SPECIFICATION_SEARCH.md
total_specs = len(results)
total_success = len([r for r in results if r['run_success'] == 1])
total_failed = total_specs - total_success

md_content = f"""# Specification Search: {PAPER_ID}

## Paper
Reinhart, Carmen M. and Kenneth S. Rogoff (2011). "From Financial Crash to Debt Crisis."
*American Economic Review*, 101(5), 1676-1706.

## Surface Summary
- **Baseline groups**: 2
  - G1: Banking crisis contagion (focal: center variable)
  - G2: External debt crisis contagion (focal: debt_move variable)
- **Design**: Pooled OLS with HC2 robust SE (TSP HCTYPE=2)
- **Budgets**: G1 max 60, G2 max 40
- **Seed**: 112444

## Data Reconstruction
The original TSP code loads panel data from external text files (banking.txt, ext_debt.txt,
development.txt, region.txt, public_gdp.txt) that are not included in the replication package.
The panel was reconstructed from:
- **Varieties_Part_I-IV.xls**: Banking crisis (col 7) and external debt crisis (col 6) indicators
  for 70 countries, 1800-2010
- **Debt_to_GDP_Part_I-IV.xls**: Total public debt/GDP ratios

Derived variables follow TSP code:
- `bank_move`: 3-year lagged MA of banking crisis: (bank(t-1)+bank(t-2)+bank(t-3))/3
- `debt_move`: 3-year lagged MA of external debt crisis: (debt(t-1)+debt(t-2)+debt(t-3))/3
- `center`: UK/US financial center contagion: (UK_bank(t)+UK_bank(t-1)+UK_bank(t-2)+US_bank(t)+US_bank(t-1)+US_bank(t-2))/6
- `public`: 2-year change in public debt/GDP ratio
- `develop1`: Emerging market dummy (development==1)
- `develop2`: Advanced economy dummy (development==2)

## Estimation
- **Canonical inference**: HC2 robust standard errors (statsmodels OLS with cov_type='HC2')
- **FE models**: pyfixest with vcov="hetero" (HC1) for FE specifications
- **Logit/probit**: statsmodels with HC2

## Execution Summary

| Metric | Count |
|--------|-------|
| Total specs planned | {total_specs} |
| Specs executed successfully | {total_success} |
| Specs failed | {total_failed} |
| G1 specs | {g1_count} ({g1_success} success) |
| G2 specs | {g2_count} ({g2_success} success) |
| Inference variants | {len(inference_results)} |

## RC Axes Executed

### G1: Banking Crisis
- Baseline (Eq5 with public, 1824-2009)
- Controls LOO: drop public, bank_move, debt_move, develop1, develop2
- Sample periods: 1900-2009, 1946-2009
- Sample subsets: advanced only, emerging only, drop UK/US, trim public debt
- Functional form: logit, probit
- Fixed effects: country, year, country+year, region
- Joint variations: period x controls, FE x period, subsample x period, FE x cluster

### G2: External Debt Crisis
- Baseline (Eq6 with public, 1824-2009)
- Controls LOO: drop public, bank_move, center, develop1, develop2
- Sample periods: 1900-2009, 1946-2009
- Sample subsets: advanced only, emerging only
- Functional form: logit, probit
- Fixed effects: country, year, region
- Joint variations: period x controls, FE x period, FE x cluster

## Deviations from Surface
- Data reconstructed from Excel files rather than original TSP text files
- Development classification based on standard R&R grouping; exact classification
  may differ slightly from original (no definitive mapping in the replication package)
- N counts may differ slightly from TSP output due to missing value handling
  in the reconstructed dataset
- The TSP code has 71 country series (b1-b71, d1-d71) but only 70 countries
  are found in the Varieties Excel files. One country may be missing.
- center variable in year-FE models may have reduced variation due to collinearity

## Software
- Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}
- pandas, numpy, pyfixest, statsmodels
- Surface hash: {SURFACE_HASH}
"""

with open(f"{OUTPUT_DIR}/SPECIFICATION_SEARCH.md", 'w') as f:
    f.write(md_content)

print(f"\nOutputs written to {OUTPUT_DIR}/")
print("Done.")
