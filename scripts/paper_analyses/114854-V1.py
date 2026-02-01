"""
Specification Search Script for Paper 114854-V1
"Paging Inspector Sands: The Costs of Public Information"
by Sacha Kapoor and Arvind Magesan

This paper examines the effect of pedestrian countdown signals on traffic safety.
The main collision data is not available in the replication package, but we can
replicate Tables 8 and 9 which analyze pedestrian and automobile flow effects.

Method: Cross-sectional OLS with various control sets
Treatment: countdown indicator (whether countdown signal installed before count date)
Outcomes:
  - ped_vol8hr (8-hour pedestrian volume)
  - tot_count (total automobile count)
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import statsmodels.api as sm
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_PATH = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/114854-V1/Data-and-Programs'
OUTPUT_PATH = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/114854-V1'

# Paper metadata
PAPER_ID = '114854-V1'
PAPER_TITLE = 'Paging Inspector Sands: The Costs of Public Information'
JOURNAL = 'AER'

results = []

def extract_results(model, spec_id, spec_tree_path, outcome_var, treatment_var,
                   sample_desc, fixed_effects, controls_desc, cluster_var,
                   model_type, n_obs=None, coef_vector=None):
    """Extract results from pyfixest model and return as dict."""
    try:
        coef = model.coef()[treatment_var]
        se = model.se()[treatment_var]
        t_stat = model.tstat()[treatment_var]
        p_val = model.pvalue()[treatment_var]
        # Calculate CI
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se
        # Get R-squared and N from model
        r2 = model._r2 if hasattr(model, '_r2') else None
        if n_obs is None:
            n_obs = model._N if hasattr(model, '_N') else len(model._data) if hasattr(model, '_data') else None
    except Exception as e:
        print(f"    Error extracting results: {e}")
        return None

    # Build coefficient vector JSON
    if coef_vector is None:
        coef_vector = {
            "treatment": {"var": treatment_var, "coef": float(coef), "se": float(se), "pval": float(p_val)},
            "controls": [],
            "fixed_effects": fixed_effects.split(', ') if fixed_effects else [],
            "diagnostics": {}
        }
        # Add other coefficients
        for var in model.coef().index:
            if var != treatment_var and not var.startswith('C(') and not var.startswith('_'):
                coef_vector["controls"].append({
                    "var": var,
                    "coef": float(model.coef()[var]),
                    "se": float(model.se()[var]),
                    "pval": float(model.pvalue()[var])
                })

    return {
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'outcome_var': outcome_var,
        'treatment_var': treatment_var,
        'coefficient': float(coef),
        'std_error': float(se),
        't_stat': float(t_stat),
        'p_value': float(p_val),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'n_obs': int(n_obs),
        'r_squared': float(r2) if r2 is not None else None,
        'coefficient_vector_json': json.dumps(coef_vector),
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    }

def extract_sm_results(result, spec_id, spec_tree_path, outcome_var, treatment_var,
                       sample_desc, fixed_effects, controls_desc, cluster_var,
                       model_type, n_obs):
    """Extract results from statsmodels result and return as dict."""
    try:
        coef = result.params[treatment_var]
        se = result.bse[treatment_var]
        t_stat = result.tvalues[treatment_var]
        p_val = result.pvalues[treatment_var]
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se
        r2 = result.rsquared if hasattr(result, 'rsquared') else None
    except:
        return None

    coef_vector = {
        "treatment": {"var": treatment_var, "coef": float(coef), "se": float(se), "pval": float(p_val)},
        "controls": [],
        "fixed_effects": fixed_effects.split(', ') if fixed_effects else [],
        "diagnostics": {}
    }
    for var in result.params.index:
        if var != treatment_var and var != 'const' and not var.startswith('C('):
            try:
                coef_vector["controls"].append({
                    "var": var,
                    "coef": float(result.params[var]),
                    "se": float(result.bse[var]),
                    "pval": float(result.pvalues[var])
                })
            except:
                pass

    return {
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'outcome_var': outcome_var,
        'treatment_var': treatment_var,
        'coefficient': float(coef),
        'std_error': float(se),
        't_stat': float(t_stat),
        'p_value': float(p_val),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'n_obs': int(n_obs),
        'r_squared': float(r2) if r2 is not None else None,
        'coefficient_vector_json': json.dumps(coef_vector),
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    }

# ============================================
# LOAD AND PREPARE PEDESTRIAN DATA
# ============================================
print("Loading pedestrian flow data...")
ped = pd.read_stata(f'{DATA_PATH}/pedestrianflow.dta')

# Parse dates
ped['installdate2'] = pd.to_datetime(ped['installdate'], format='%m/%d/%Y', errors='coerce')
ped['activdate2'] = pd.to_datetime(ped['activdate'], format='%m/%d/%Y', errors='coerce')
ped['install_year'] = ped['installdate2'].dt.year
ped['active_year'] = ped['activdate2'].dt.year

# Generate year, month, day of week
ped['yr'] = ped['count_date'].dt.year
ped['mo'] = ped['count_date'].dt.month
ped['day'] = ped['count_date'].dt.dayofweek

# Sample restrictions per do-file:
# a) Traffic light activated after 2004 - drop
# b) Countdown never installed - drop
ped_clean = ped[ped['active_year'] < 2004].copy()
ped_clean = ped_clean[ped_clean['install_year'].notna()].copy()

# Generate east/west and north/south indicators
ped_clean['ew_ind'] = (ped_clean['xcoordinate'] > 313104822.1).astype(int)
ped_clean['ns_ind'] = (ped_clean['ycoordinate'] > 4841254464).astype(int)

# Generate countdown indicator
ped_clean['countdown'] = (ped_clean['installdate2'] <= ped_clean['count_date']).astype(int)

# Drop missing outcome
ped_clean = ped_clean.dropna(subset=['ped_vol8hr', 'countdown', 'day', 'mo', 'yr'])

# Create categorical variables for fixed effects
ped_clean['day_cat'] = ped_clean['day'].astype(str)
ped_clean['mo_cat'] = ped_clean['mo'].astype(str)
ped_clean['yr_cat'] = ped_clean['yr'].astype(str)

print(f"Pedestrian data: {len(ped_clean)} observations after cleaning")

# ============================================
# PEDESTRIAN FLOW SPECIFICATIONS (TABLE 8)
# ============================================
print("\n=== PEDESTRIAN FLOW SPECIFICATIONS ===")

# Baseline: Table 8 Column 1 - bivariate
try:
    model = pf.feols("ped_vol8hr ~ countdown", data=ped_clean, vcov='hetero')
    res = extract_results(model, 'baseline', 'methods/cross_sectional_ols.md',
                         'ped_vol8hr', 'countdown',
                         'Pedestrian flow - bivariate', '', 'None', 'None', 'OLS')
    if res:
        results.append(res)
        print(f"  Baseline (bivariate): coef={res['coefficient']:.3f}, p={res['p_value']:.4f}")
except Exception as e:
    print(f"  Error in baseline: {e}")

# Table 8 Column 2: + day FE
try:
    model = pf.feols("ped_vol8hr ~ countdown | day_cat", data=ped_clean, vcov='hetero')
    res = extract_results(model, 'cross_sectional/fe/day', 'methods/cross_sectional_ols.md#fixed-effects',
                         'ped_vol8hr', 'countdown',
                         'Pedestrian flow - day FE', 'day', 'None', 'None', 'OLS',
                         )
    if res:
        results.append(res)
        print(f"  + Day FE: coef={res['coefficient']:.3f}, p={res['p_value']:.4f}")
except Exception as e:
    print(f"  Error in day FE: {e}")

# Table 8 Column 3: + day + month FE
try:
    model = pf.feols("ped_vol8hr ~ countdown | day_cat + mo_cat", data=ped_clean, vcov='hetero')
    res = extract_results(model, 'cross_sectional/fe/day_month', 'methods/cross_sectional_ols.md#fixed-effects',
                         'ped_vol8hr', 'countdown',
                         'Pedestrian flow - day + month FE', 'day, month', 'None', 'None', 'OLS',
                         )
    if res:
        results.append(res)
        print(f"  + Day + Month FE: coef={res['coefficient']:.3f}, p={res['p_value']:.4f}")
except Exception as e:
    print(f"  Error in day+month FE: {e}")

# Table 8 Column 4: + day + month + year FE
try:
    model = pf.feols("ped_vol8hr ~ countdown | day_cat + mo_cat + yr_cat", data=ped_clean, vcov='hetero')
    res = extract_results(model, 'cross_sectional/fe/day_month_year', 'methods/cross_sectional_ols.md#fixed-effects',
                         'ped_vol8hr', 'countdown',
                         'Pedestrian flow - day + month + year FE', 'day, month, year', 'None', 'None', 'OLS',
                         )
    if res:
        results.append(res)
        print(f"  + Day + Month + Year FE: coef={res['coefficient']:.3f}, p={res['p_value']:.4f}")
except Exception as e:
    print(f"  Error in day+month+year FE: {e}")

# Table 8 Column 5: + day + month + year FE + ns_ind + ew_ind
try:
    model = pf.feols("ped_vol8hr ~ countdown + ns_ind + ew_ind | day_cat + mo_cat + yr_cat",
                    data=ped_clean, vcov='hetero')
    res = extract_results(model, 'cross_sectional/fe/day_month_year_geo', 'methods/cross_sectional_ols.md#fixed-effects',
                         'ped_vol8hr', 'countdown',
                         'Pedestrian flow - time FE + geographic controls', 'day, month, year',
                         'ns_ind, ew_ind', 'None', 'OLS', )
    if res:
        results.append(res)
        print(f"  + Time FE + Geo: coef={res['coefficient']:.3f}, p={res['p_value']:.4f}")
except Exception as e:
    print(f"  Error in time FE + geo: {e}")

# Table 8 Column 6: + main street FE (if enough variation)
try:
    # Create main street category
    ped_clean['main_cat'] = pd.Categorical(ped_clean['main']).codes.astype(str)
    model = pf.feols("ped_vol8hr ~ countdown + ns_ind + ew_ind | day_cat + mo_cat + yr_cat + main_cat",
                    data=ped_clean, vcov='hetero')
    res = extract_results(model, 'cross_sectional/fe/full_model', 'methods/cross_sectional_ols.md#fixed-effects',
                         'ped_vol8hr', 'countdown',
                         'Pedestrian flow - full model with main street FE',
                         'day, month, year, main_street', 'ns_ind, ew_ind', 'None', 'OLS', )
    if res:
        results.append(res)
        print(f"  + Full model (main street FE): coef={res['coefficient']:.3f}, p={res['p_value']:.4f}")
except Exception as e:
    print(f"  Error in full model: {e}")

# Table 8 Column 7: + side route FE
try:
    ped_clean['side1_cat'] = pd.Categorical(ped_clean['side1route']).codes.astype(str)
    model = pf.feols("ped_vol8hr ~ countdown + ns_ind + ew_ind | day_cat + mo_cat + yr_cat + main_cat + side1_cat",
                    data=ped_clean, vcov='hetero')
    res = extract_results(model, 'cross_sectional/fe/full_with_side', 'methods/cross_sectional_ols.md#fixed-effects',
                         'ped_vol8hr', 'countdown',
                         'Pedestrian flow - full model with side route FE',
                         'day, month, year, main_street, side_route', 'ns_ind, ew_ind', 'None', 'OLS', )
    if res:
        results.append(res)
        print(f"  + Side route FE: coef={res['coefficient']:.3f}, p={res['p_value']:.4f}")
except Exception as e:
    print(f"  Error in side route model: {e}")

# ============================================
# LOAD AND PREPARE AUTOMOBILE DATA
# ============================================
print("\nLoading automobile flow data...")
auto = pd.read_stata(f'{DATA_PATH}/automobileflow.dta')

# Parse dates
auto['installdate2'] = pd.to_datetime(auto['installdate'], format='%m/%d/%Y', errors='coerce')
auto['activdate2'] = pd.to_datetime(auto['activdate'], format='%m/%d/%Y', errors='coerce')
auto['install_year'] = auto['installdate2'].dt.year
auto['active_year'] = auto['activdate2'].dt.year

# Parse count_date (DMY format in auto data)
auto['count_date2'] = pd.to_datetime(auto['count_date'], format='%d/%m/%Y', errors='coerce')
auto['yr'] = auto['count_date2'].dt.year
auto['mo'] = auto['count_date2'].dt.month
auto['day'] = auto['count_date2'].dt.dayofweek

# Sample restrictions per do-file
auto_clean = auto[auto['active_year'] < 2004].copy()
auto_clean = auto_clean[auto_clean['install_year'].notna()].copy()

# Generate geographic indicators (note: lat/long names are swapped in original data)
auto_clean['ew_ind'] = (auto_clean['latitude'] > 313104822.1).astype(int)
auto_clean['ns_ind'] = (auto_clean['longtitude'] > 4841254464).astype(int)

# Generate countdown indicator
auto_clean['countdown'] = (auto_clean['installdate2'] <= auto_clean['count_date2']).astype(int)

# Drop if count is missing
auto_clean = auto_clean.dropna(subset=['tot_count', 'countdown', 'day', 'mo', 'yr'])

# Generate count frequency variable (nvar)
auto_clean['nvar'] = auto_clean.groupby('id_pcs')['id_pcs'].transform('count')

# Create categorical variables for fixed effects
auto_clean['day_cat'] = auto_clean['day'].astype(str)
auto_clean['mo_cat'] = auto_clean['mo'].astype(str)
auto_clean['yr_cat'] = auto_clean['yr'].astype(str)
auto_clean['street1_cat'] = pd.Categorical(auto_clean['street1']).codes.astype(str)
auto_clean['street2_cat'] = pd.Categorical(auto_clean['street2']).codes.astype(str)

print(f"Automobile data: {len(auto_clean)} observations after cleaning")

# ============================================
# AUTOMOBILE FLOW SPECIFICATIONS (TABLE 9)
# ============================================
print("\n=== AUTOMOBILE FLOW SPECIFICATIONS ===")

# Table 9 Column 1: bivariate clustered
try:
    model = pf.feols("tot_count ~ countdown", data=auto_clean, vcov={'CRV1': 'id_pcs'})
    res = extract_results(model, 'auto/baseline', 'methods/cross_sectional_ols.md',
                         'tot_count', 'countdown',
                         'Automobile flow - bivariate', '', 'None', 'id_pcs', 'OLS',
                         )
    if res:
        results.append(res)
        print(f"  Auto Baseline: coef={res['coefficient']:.3f}, p={res['p_value']:.4f}")
except Exception as e:
    print(f"  Error in auto baseline: {e}")

# Table 9 Column 2: + day FE
try:
    model = pf.feols("tot_count ~ countdown | day_cat", data=auto_clean, vcov={'CRV1': 'id_pcs'})
    res = extract_results(model, 'auto/fe/day', 'methods/cross_sectional_ols.md#fixed-effects',
                         'tot_count', 'countdown',
                         'Automobile flow - day FE', 'day', 'None', 'id_pcs', 'OLS',
                         )
    if res:
        results.append(res)
        print(f"  Auto + Day FE: coef={res['coefficient']:.3f}, p={res['p_value']:.4f}")
except Exception as e:
    print(f"  Error in auto day FE: {e}")

# Table 9 Column 3: + day + month FE
try:
    model = pf.feols("tot_count ~ countdown | day_cat + mo_cat", data=auto_clean, vcov={'CRV1': 'id_pcs'})
    res = extract_results(model, 'auto/fe/day_month', 'methods/cross_sectional_ols.md#fixed-effects',
                         'tot_count', 'countdown',
                         'Automobile flow - day + month FE', 'day, month', 'None', 'id_pcs', 'OLS',
                         )
    if res:
        results.append(res)
        print(f"  Auto + Day + Month FE: coef={res['coefficient']:.3f}, p={res['p_value']:.4f}")
except Exception as e:
    print(f"  Error in auto day+month FE: {e}")

# Table 9 Column 4: + day + month + year FE
try:
    model = pf.feols("tot_count ~ countdown | day_cat + mo_cat + yr_cat", data=auto_clean, vcov={'CRV1': 'id_pcs'})
    res = extract_results(model, 'auto/fe/day_month_year', 'methods/cross_sectional_ols.md#fixed-effects',
                         'tot_count', 'countdown',
                         'Automobile flow - day + month + year FE', 'day, month, year', 'None', 'id_pcs', 'OLS',
                         )
    if res:
        results.append(res)
        print(f"  Auto + Day + Month + Year FE: coef={res['coefficient']:.3f}, p={res['p_value']:.4f}")
except Exception as e:
    print(f"  Error in auto day+month+year FE: {e}")

# Table 9 Column 5: + geographic + street1 FE
try:
    model = pf.feols("tot_count ~ countdown + ew_ind + ns_ind | day_cat + mo_cat + yr_cat + street1_cat",
                    data=auto_clean, vcov={'CRV1': 'id_pcs'})
    res = extract_results(model, 'auto/fe/time_geo_street1', 'methods/cross_sectional_ols.md#fixed-effects',
                         'tot_count', 'countdown',
                         'Automobile flow - time + geo + street1 FE', 'day, month, year, street1',
                         'ew_ind, ns_ind', 'id_pcs', 'OLS', )
    if res:
        results.append(res)
        print(f"  Auto + Time + Geo + Street1: coef={res['coefficient']:.3f}, p={res['p_value']:.4f}")
except Exception as e:
    print(f"  Error in auto time+geo+street1: {e}")

# Table 9 Column 6: + street2 FE
try:
    model = pf.feols("tot_count ~ countdown + ew_ind + ns_ind | day_cat + mo_cat + yr_cat + street1_cat + street2_cat",
                    data=auto_clean, vcov={'CRV1': 'id_pcs'})
    res = extract_results(model, 'auto/fe/time_geo_street12', 'methods/cross_sectional_ols.md#fixed-effects',
                         'tot_count', 'countdown',
                         'Automobile flow - time + geo + street1 + street2 FE',
                         'day, month, year, street1, street2', 'ew_ind, ns_ind', 'id_pcs', 'OLS', )
    if res:
        results.append(res)
        print(f"  Auto + Street2 FE: coef={res['coefficient']:.3f}, p={res['p_value']:.4f}")
except Exception as e:
    print(f"  Error in auto street2 model: {e}")

# Table 9 Column 7: + nvar control
try:
    model = pf.feols("tot_count ~ countdown + ew_ind + ns_ind + nvar | day_cat + mo_cat + yr_cat + street1_cat + street2_cat",
                    data=auto_clean, vcov={'CRV1': 'id_pcs'})
    res = extract_results(model, 'auto/fe/full_model', 'methods/cross_sectional_ols.md#fixed-effects',
                         'tot_count', 'countdown',
                         'Automobile flow - full model with count frequency',
                         'day, month, year, street1, street2', 'ew_ind, ns_ind, nvar', 'id_pcs', 'OLS', )
    if res:
        results.append(res)
        print(f"  Auto Full Model: coef={res['coefficient']:.3f}, p={res['p_value']:.4f}")
except Exception as e:
    print(f"  Error in auto full model: {e}")

# ============================================
# ROBUSTNESS CHECKS - CLUSTERING VARIATIONS
# ============================================
print("\n=== ROBUSTNESS: CLUSTERING VARIATIONS (Pedestrian) ===")

# Use the full pedestrian model as reference
try:
    # No clustering (robust SE only)
    model = pf.feols("ped_vol8hr ~ countdown + ns_ind + ew_ind | day_cat + mo_cat + yr_cat",
                    data=ped_clean, vcov='hetero')
    res = extract_results(model, 'robust/cluster/none', 'robustness/clustering_variations.md#single-level-clustering',
                         'ped_vol8hr', 'countdown',
                         'Pedestrian flow - robust SE', 'day, month, year',
                         'ns_ind, ew_ind', 'None (robust)', 'OLS', )
    if res:
        results.append(res)
        print(f"  Robust SE: coef={res['coefficient']:.3f}, se={res['std_error']:.3f}")
except Exception as e:
    print(f"  Error in robust SE: {e}")

# Cluster by main street
try:
    model = pf.feols("ped_vol8hr ~ countdown + ns_ind + ew_ind | day_cat + mo_cat + yr_cat",
                    data=ped_clean.dropna(subset=['main']), vcov={'CRV1': 'main'})
    res = extract_results(model, 'robust/cluster/main_street', 'robustness/clustering_variations.md#single-level-clustering',
                         'ped_vol8hr', 'countdown',
                         'Pedestrian flow - clustered by main street', 'day, month, year',
                         'ns_ind, ew_ind', 'main_street', 'OLS', )
    if res:
        results.append(res)
        print(f"  Cluster main street: coef={res['coefficient']:.3f}, se={res['std_error']:.3f}")
except Exception as e:
    print(f"  Error in cluster main street: {e}")

# Cluster by year
try:
    model = pf.feols("ped_vol8hr ~ countdown + ns_ind + ew_ind | day_cat + mo_cat + yr_cat",
                    data=ped_clean, vcov={'CRV1': 'yr'})
    res = extract_results(model, 'robust/cluster/year', 'robustness/clustering_variations.md#single-level-clustering',
                         'ped_vol8hr', 'countdown',
                         'Pedestrian flow - clustered by year', 'day, month, year',
                         'ns_ind, ew_ind', 'year', 'OLS', )
    if res:
        results.append(res)
        print(f"  Cluster year: coef={res['coefficient']:.3f}, se={res['std_error']:.3f}")
except Exception as e:
    print(f"  Error in cluster year: {e}")

# ============================================
# ROBUSTNESS: CLUSTERING (Automobile)
# ============================================
print("\n=== ROBUSTNESS: CLUSTERING VARIATIONS (Automobile) ===")

# No clustering (robust SE only)
try:
    model = pf.feols("tot_count ~ countdown + ew_ind + ns_ind | day_cat + mo_cat + yr_cat",
                    data=auto_clean, vcov='hetero')
    res = extract_results(model, 'robust/cluster/auto_none', 'robustness/clustering_variations.md#single-level-clustering',
                         'tot_count', 'countdown',
                         'Automobile flow - robust SE', 'day, month, year',
                         'ew_ind, ns_ind', 'None (robust)', 'OLS', )
    if res:
        results.append(res)
        print(f"  Auto Robust SE: coef={res['coefficient']:.3f}, se={res['std_error']:.3f}")
except Exception as e:
    print(f"  Error in auto robust SE: {e}")

# Cluster by street1
try:
    model = pf.feols("tot_count ~ countdown + ew_ind + ns_ind | day_cat + mo_cat + yr_cat",
                    data=auto_clean.dropna(subset=['street1']), vcov={'CRV1': 'street1'})
    res = extract_results(model, 'robust/cluster/auto_street1', 'robustness/clustering_variations.md#single-level-clustering',
                         'tot_count', 'countdown',
                         'Automobile flow - clustered by street1', 'day, month, year',
                         'ew_ind, ns_ind', 'street1', 'OLS', )
    if res:
        results.append(res)
        print(f"  Auto Cluster street1: coef={res['coefficient']:.3f}, se={res['std_error']:.3f}")
except Exception as e:
    print(f"  Error in auto cluster street1: {e}")

# ============================================
# ROBUSTNESS: SAMPLE RESTRICTIONS
# ============================================
print("\n=== ROBUSTNESS: SAMPLE RESTRICTIONS ===")

# Pedestrian - early period
try:
    ped_early = ped_clean[ped_clean['yr'] <= ped_clean['yr'].median()]
    if len(ped_early) > 50:
        model = pf.feols("ped_vol8hr ~ countdown + ns_ind + ew_ind | day_cat + mo_cat + yr_cat",
                        data=ped_early, vcov='hetero')
        res = extract_results(model, 'robust/sample/ped_early_period', 'robustness/sample_restrictions.md#time-based-restrictions',
                             'ped_vol8hr', 'countdown',
                             f'Pedestrian flow - early period (yr<={ped_clean["yr"].median()})',
                             'day, month, year', 'ns_ind, ew_ind', 'None', 'OLS', )
        if res:
            results.append(res)
            print(f"  Ped early period: coef={res['coefficient']:.3f}, n={res['n_obs']}")
except Exception as e:
    print(f"  Error in ped early period: {e}")

# Pedestrian - late period
try:
    ped_late = ped_clean[ped_clean['yr'] > ped_clean['yr'].median()]
    if len(ped_late) > 50:
        model = pf.feols("ped_vol8hr ~ countdown + ns_ind + ew_ind | day_cat + mo_cat + yr_cat",
                        data=ped_late, vcov='hetero')
        res = extract_results(model, 'robust/sample/ped_late_period', 'robustness/sample_restrictions.md#time-based-restrictions',
                             'ped_vol8hr', 'countdown',
                             f'Pedestrian flow - late period (yr>{ped_clean["yr"].median()})',
                             'day, month, year', 'ns_ind, ew_ind', 'None', 'OLS', )
        if res:
            results.append(res)
            print(f"  Ped late period: coef={res['coefficient']:.3f}, n={res['n_obs']}")
except Exception as e:
    print(f"  Error in ped late period: {e}")

# Pedestrian - trim outliers (1%)
try:
    q1 = ped_clean['ped_vol8hr'].quantile(0.01)
    q99 = ped_clean['ped_vol8hr'].quantile(0.99)
    ped_trim = ped_clean[(ped_clean['ped_vol8hr'] >= q1) & (ped_clean['ped_vol8hr'] <= q99)]
    model = pf.feols("ped_vol8hr ~ countdown + ns_ind + ew_ind | day_cat + mo_cat + yr_cat",
                    data=ped_trim, vcov='hetero')
    res = extract_results(model, 'robust/sample/ped_trim_1pct', 'robustness/sample_restrictions.md#outlier-handling',
                         'ped_vol8hr', 'countdown',
                         'Pedestrian flow - trimmed 1% outliers',
                         'day, month, year', 'ns_ind, ew_ind', 'None', 'OLS', )
    if res:
        results.append(res)
        print(f"  Ped trim 1%: coef={res['coefficient']:.3f}, n={res['n_obs']}")
except Exception as e:
    print(f"  Error in ped trim: {e}")

# Automobile - early period
try:
    auto_early = auto_clean[auto_clean['yr'] <= auto_clean['yr'].median()]
    if len(auto_early) > 50:
        model = pf.feols("tot_count ~ countdown + ew_ind + ns_ind | day_cat + mo_cat + yr_cat",
                        data=auto_early, vcov={'CRV1': 'id_pcs'})
        res = extract_results(model, 'robust/sample/auto_early_period', 'robustness/sample_restrictions.md#time-based-restrictions',
                             'tot_count', 'countdown',
                             f'Automobile flow - early period',
                             'day, month, year', 'ew_ind, ns_ind', 'id_pcs', 'OLS', )
        if res:
            results.append(res)
            print(f"  Auto early period: coef={res['coefficient']:.3f}, n={res['n_obs']}")
except Exception as e:
    print(f"  Error in auto early period: {e}")

# Automobile - late period
try:
    auto_late = auto_clean[auto_clean['yr'] > auto_clean['yr'].median()]
    if len(auto_late) > 50:
        model = pf.feols("tot_count ~ countdown + ew_ind + ns_ind | day_cat + mo_cat + yr_cat",
                        data=auto_late, vcov={'CRV1': 'id_pcs'})
        res = extract_results(model, 'robust/sample/auto_late_period', 'robustness/sample_restrictions.md#time-based-restrictions',
                             'tot_count', 'countdown',
                             f'Automobile flow - late period',
                             'day, month, year', 'ew_ind, ns_ind', 'id_pcs', 'OLS', )
        if res:
            results.append(res)
            print(f"  Auto late period: coef={res['coefficient']:.3f}, n={res['n_obs']}")
except Exception as e:
    print(f"  Error in auto late period: {e}")

# ============================================
# ROBUSTNESS: LEAVE-ONE-OUT (Controls)
# ============================================
print("\n=== ROBUSTNESS: LEAVE-ONE-OUT ===")

# Pedestrian: drop ns_ind
try:
    model = pf.feols("ped_vol8hr ~ countdown + ew_ind | day_cat + mo_cat + yr_cat",
                    data=ped_clean, vcov='hetero')
    res = extract_results(model, 'robust/loo/drop_ns_ind', 'robustness/leave_one_out.md',
                         'ped_vol8hr', 'countdown',
                         'Pedestrian flow - drop ns_ind', 'day, month, year',
                         'ew_ind', 'None', 'OLS', )
    if res:
        results.append(res)
        print(f"  Ped drop ns_ind: coef={res['coefficient']:.3f}")
except Exception as e:
    print(f"  Error in ped drop ns_ind: {e}")

# Pedestrian: drop ew_ind
try:
    model = pf.feols("ped_vol8hr ~ countdown + ns_ind | day_cat + mo_cat + yr_cat",
                    data=ped_clean, vcov='hetero')
    res = extract_results(model, 'robust/loo/drop_ew_ind', 'robustness/leave_one_out.md',
                         'ped_vol8hr', 'countdown',
                         'Pedestrian flow - drop ew_ind', 'day, month, year',
                         'ns_ind', 'None', 'OLS', )
    if res:
        results.append(res)
        print(f"  Ped drop ew_ind: coef={res['coefficient']:.3f}")
except Exception as e:
    print(f"  Error in ped drop ew_ind: {e}")

# Automobile: drop ns_ind
try:
    model = pf.feols("tot_count ~ countdown + ew_ind | day_cat + mo_cat + yr_cat",
                    data=auto_clean, vcov={'CRV1': 'id_pcs'})
    res = extract_results(model, 'robust/loo/auto_drop_ns_ind', 'robustness/leave_one_out.md',
                         'tot_count', 'countdown',
                         'Automobile flow - drop ns_ind', 'day, month, year',
                         'ew_ind', 'id_pcs', 'OLS', )
    if res:
        results.append(res)
        print(f"  Auto drop ns_ind: coef={res['coefficient']:.3f}")
except Exception as e:
    print(f"  Error in auto drop ns_ind: {e}")

# Automobile: drop ew_ind
try:
    model = pf.feols("tot_count ~ countdown + ns_ind | day_cat + mo_cat + yr_cat",
                    data=auto_clean, vcov={'CRV1': 'id_pcs'})
    res = extract_results(model, 'robust/loo/auto_drop_ew_ind', 'robustness/leave_one_out.md',
                         'tot_count', 'countdown',
                         'Automobile flow - drop ew_ind', 'day, month, year',
                         'ns_ind', 'id_pcs', 'OLS', )
    if res:
        results.append(res)
        print(f"  Auto drop ew_ind: coef={res['coefficient']:.3f}")
except Exception as e:
    print(f"  Error in auto drop ew_ind: {e}")

# Automobile: drop nvar
try:
    model = pf.feols("tot_count ~ countdown + ew_ind + ns_ind | day_cat + mo_cat + yr_cat + street1_cat + street2_cat",
                    data=auto_clean, vcov={'CRV1': 'id_pcs'})
    res = extract_results(model, 'robust/loo/auto_drop_nvar', 'robustness/leave_one_out.md',
                         'tot_count', 'countdown',
                         'Automobile flow - drop nvar (full model)',
                         'day, month, year, street1, street2', 'ew_ind, ns_ind', 'id_pcs', 'OLS', )
    if res:
        results.append(res)
        print(f"  Auto drop nvar: coef={res['coefficient']:.3f}")
except Exception as e:
    print(f"  Error in auto drop nvar: {e}")

# ============================================
# ROBUSTNESS: SINGLE COVARIATE
# ============================================
print("\n=== ROBUSTNESS: SINGLE COVARIATE ===")

# Pedestrian: bivariate
try:
    model = pf.feols("ped_vol8hr ~ countdown", data=ped_clean, vcov='hetero')
    res = extract_results(model, 'robust/single/ped_none', 'robustness/single_covariate.md',
                         'ped_vol8hr', 'countdown',
                         'Pedestrian flow - bivariate only', '',
                         'None', 'None', 'OLS', )
    if res:
        results.append(res)
        print(f"  Ped bivariate: coef={res['coefficient']:.3f}")
except Exception as e:
    print(f"  Error in ped bivariate: {e}")

# Pedestrian: + ns_ind only
try:
    model = pf.feols("ped_vol8hr ~ countdown + ns_ind", data=ped_clean, vcov='hetero')
    res = extract_results(model, 'robust/single/ped_ns_ind', 'robustness/single_covariate.md',
                         'ped_vol8hr', 'countdown',
                         'Pedestrian flow - + ns_ind only', '',
                         'ns_ind', 'None', 'OLS', )
    if res:
        results.append(res)
        print(f"  Ped + ns_ind only: coef={res['coefficient']:.3f}")
except Exception as e:
    print(f"  Error in ped ns_ind only: {e}")

# Pedestrian: + ew_ind only
try:
    model = pf.feols("ped_vol8hr ~ countdown + ew_ind", data=ped_clean, vcov='hetero')
    res = extract_results(model, 'robust/single/ped_ew_ind', 'robustness/single_covariate.md',
                         'ped_vol8hr', 'countdown',
                         'Pedestrian flow - + ew_ind only', '',
                         'ew_ind', 'None', 'OLS', )
    if res:
        results.append(res)
        print(f"  Ped + ew_ind only: coef={res['coefficient']:.3f}")
except Exception as e:
    print(f"  Error in ped ew_ind only: {e}")

# ============================================
# ROBUSTNESS: FUNCTIONAL FORM
# ============================================
print("\n=== ROBUSTNESS: FUNCTIONAL FORM ===")

# Pedestrian: log outcome
try:
    ped_clean['log_ped_vol'] = np.log(ped_clean['ped_vol8hr'] + 1)
    model = pf.feols("log_ped_vol ~ countdown + ns_ind + ew_ind | day_cat + mo_cat + yr_cat",
                    data=ped_clean, vcov='hetero')
    res = extract_results(model, 'robust/form/ped_y_log', 'robustness/functional_form.md#outcome-variable-transformations',
                         'log(ped_vol8hr+1)', 'countdown',
                         'Pedestrian flow - log outcome', 'day, month, year',
                         'ns_ind, ew_ind', 'None', 'OLS', )
    if res:
        results.append(res)
        print(f"  Ped log outcome: coef={res['coefficient']:.3f}")
except Exception as e:
    print(f"  Error in ped log outcome: {e}")

# Pedestrian: asinh outcome
try:
    ped_clean['asinh_ped_vol'] = np.arcsinh(ped_clean['ped_vol8hr'])
    model = pf.feols("asinh_ped_vol ~ countdown + ns_ind + ew_ind | day_cat + mo_cat + yr_cat",
                    data=ped_clean, vcov='hetero')
    res = extract_results(model, 'robust/form/ped_y_asinh', 'robustness/functional_form.md#outcome-variable-transformations',
                         'asinh(ped_vol8hr)', 'countdown',
                         'Pedestrian flow - asinh outcome', 'day, month, year',
                         'ns_ind, ew_ind', 'None', 'OLS', )
    if res:
        results.append(res)
        print(f"  Ped asinh outcome: coef={res['coefficient']:.3f}")
except Exception as e:
    print(f"  Error in ped asinh outcome: {e}")

# Automobile: log outcome
try:
    auto_clean['log_tot_count'] = np.log(auto_clean['tot_count'] + 1)
    model = pf.feols("log_tot_count ~ countdown + ew_ind + ns_ind | day_cat + mo_cat + yr_cat",
                    data=auto_clean, vcov={'CRV1': 'id_pcs'})
    res = extract_results(model, 'robust/form/auto_y_log', 'robustness/functional_form.md#outcome-variable-transformations',
                         'log(tot_count+1)', 'countdown',
                         'Automobile flow - log outcome', 'day, month, year',
                         'ew_ind, ns_ind', 'id_pcs', 'OLS', )
    if res:
        results.append(res)
        print(f"  Auto log outcome: coef={res['coefficient']:.3f}")
except Exception as e:
    print(f"  Error in auto log outcome: {e}")

# Automobile: asinh outcome
try:
    auto_clean['asinh_tot_count'] = np.arcsinh(auto_clean['tot_count'])
    model = pf.feols("asinh_tot_count ~ countdown + ew_ind + ns_ind | day_cat + mo_cat + yr_cat",
                    data=auto_clean, vcov={'CRV1': 'id_pcs'})
    res = extract_results(model, 'robust/form/auto_y_asinh', 'robustness/functional_form.md#outcome-variable-transformations',
                         'asinh(tot_count)', 'countdown',
                         'Automobile flow - asinh outcome', 'day, month, year',
                         'ew_ind, ns_ind', 'id_pcs', 'OLS', )
    if res:
        results.append(res)
        print(f"  Auto asinh outcome: coef={res['coefficient']:.3f}")
except Exception as e:
    print(f"  Error in auto asinh outcome: {e}")

# ============================================
# SAVE RESULTS
# ============================================
print(f"\n=== SAVING RESULTS ===")
print(f"Total specifications: {len(results)}")

# Convert to DataFrame and save
results_df = pd.DataFrame(results)
results_df.to_csv(f'{OUTPUT_PATH}/specification_results.csv', index=False)
print(f"Saved to {OUTPUT_PATH}/specification_results.csv")

# Print summary
print("\n=== SUMMARY STATISTICS ===")
print(f"Total specifications run: {len(results_df)}")
print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")
print(f"Coefficient range: [{results_df['coefficient'].min():.3f}, {results_df['coefficient'].max():.3f}]")
print(f"Median coefficient: {results_df['coefficient'].median():.3f}")
print(f"Mean coefficient: {results_df['coefficient'].mean():.3f}")
