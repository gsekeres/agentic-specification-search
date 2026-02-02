#!/usr/bin/env python3
"""
Specification Search: 114854-V1 - Paging Inspector Sands: The Costs of Public Information
Authors: Sacha Kapoor and Arvind Magesan
Journal: American Economic Journal: Economic Policy

This script performs a systematic specification search following the i4r methodology.

Paper Overview:
- Studies the effect of pedestrian countdown signals on traffic safety
- Main hypothesis: Countdown signals affect traffic behavior (collisions, flow)
- Treatment: Installation of pedestrian countdown signals at intersections
- Identification: Staggered rollout across intersections over time

Data Available:
- Tables 8 & 9 only (collision data requires license agreement)
- pedestrianflow.dta: Pedestrian volume data
- automobileflow.dta: Automobile volume data

Method: Cross-sectional OLS with various fixed effects (for flow regressions)
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# Configuration
PAPER_ID = "114854-V1"
PAPER_TITLE = "Paging Inspector Sands: The Costs of Public Information"
JOURNAL = "AEJ-Economic Policy"
DATA_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/114854-V1/Data-and-Programs"

# Results storage
results = []

def add_result(spec_id, spec_tree_path, outcome_var, treatment_var, model, df_used,
               sample_desc, fixed_effects, controls_desc, cluster_var, model_type,
               coef_var=None):
    """Add a specification result to the results list."""

    if coef_var is None:
        coef_var = treatment_var

    try:
        coef = model.coef()[coef_var]
        se = model.se()[coef_var]
        tstat = model.tstat()[coef_var]
        pval = model.pvalue()[coef_var]
        n_obs = model._N  # pyfixest uses _N for nobs
        r2 = model._r2 if hasattr(model, '_r2') else None

        # Build coefficient vector
        coef_vector = {
            "treatment": {
                "var": coef_var,
                "coef": float(coef),
                "se": float(se),
                "pval": float(pval)
            },
            "controls": [],
            "fixed_effects_absorbed": fixed_effects.split(" + ") if fixed_effects else [],
            "diagnostics": {
                "n_obs": int(n_obs),
                "r_squared": float(r2) if r2 is not None else None
            }
        }

        # Add control coefficients (skip FE dummies)
        for var in model.coef().index:
            if var != coef_var and not var.startswith('C('):
                coef_vector["controls"].append({
                    "var": var,
                    "coef": float(model.coef()[var]),
                    "se": float(model.se()[var]),
                    "pval": float(model.pvalue()[var])
                })

        result = {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': float(coef),
            'std_error': float(se),
            't_stat': float(tstat),
            'p_value': float(pval),
            'ci_lower': float(coef - 1.96 * se),
            'ci_upper': float(coef + 1.96 * se),
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
        results.append(result)
        print(f"  {spec_id}: coef={coef:.4f}, se={se:.4f}, p={pval:.4f}, n={n_obs}")

    except Exception as e:
        print(f"  ERROR in {spec_id}: {str(e)}")


# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

# Load pedestrian flow data
ped = pd.read_stata(f"{DATA_DIR}/pedestrianflow.dta")

# Load automobile flow data
auto = pd.read_stata(f"{DATA_DIR}/automobileflow.dta")

# ============================================================================
# PREPARE PEDESTRIAN FLOW DATA (Table 8)
# ============================================================================

print("\nPreparing Pedestrian Flow Data...")

# Convert dates
ped['installdate2'] = pd.to_datetime(ped['installdate'], format='%m/%d/%Y', errors='coerce')
ped['activdate2'] = pd.to_datetime(ped['activdate'], format='%m/%d/%Y', errors='coerce')
ped['install_year'] = ped['installdate2'].dt.year
ped['active_year'] = ped['activdate2'].dt.year

# count_date is already datetime
ped['yr'] = ped['count_date'].dt.year
ped['mo'] = ped['count_date'].dt.month
ped['day'] = ped['count_date'].dt.dayofweek  # 0=Monday, 6=Sunday

# Drop observations where:
# a) Traffic light activated after 2004
# b) Countdown never installed
ped_clean = ped[
    (ped['active_year'] < 2004) &
    (ped['install_year'].notna())
].copy()

# Generate east/west and north/south indicators
ped_clean['ew_ind'] = (ped_clean['xcoordinate'] > 313104822.1).astype(int)
ped_clean['ns_ind'] = (ped_clean['ycoordinate'] > 4841254464).astype(int)

# Generate countdown indicator (treatment)
ped_clean['countdown'] = (ped_clean['installdate2'] <= ped_clean['count_date']).astype(int)

# Clean outcome variable
ped_clean = ped_clean[ped_clean['ped_vol8hr'].notna()].copy()

# Create categorical variables for fixed effects
ped_clean['day_cat'] = ped_clean['day'].astype(str)
ped_clean['mo_cat'] = ped_clean['mo'].astype(str)
ped_clean['yr_cat'] = ped_clean['yr'].astype(str)
ped_clean['main_cat'] = ped_clean['main'].fillna('Unknown')
ped_clean['side1route_cat'] = ped_clean['side1route'].fillna('Unknown')

print(f"  Pedestrian data: {len(ped_clean)} observations")
print(f"  Treatment rate: {ped_clean['countdown'].mean():.2%}")

# ============================================================================
# PREPARE AUTOMOBILE FLOW DATA (Table 9)
# ============================================================================

print("\nPreparing Automobile Flow Data...")

# Convert dates
auto['installdate2'] = pd.to_datetime(auto['installdate'], format='%m/%d/%Y', errors='coerce')
auto['activdate2'] = pd.to_datetime(auto['activdate'], format='%m/%d/%Y', errors='coerce')
auto['install_year'] = auto['installdate2'].dt.year
auto['active_year'] = auto['activdate2'].dt.year

# Parse count_date (format is DD/MM/YYYY)
auto['count_date2'] = pd.to_datetime(auto['count_date'], format='%d/%m/%Y', errors='coerce')
auto['yr'] = auto['count_date2'].dt.year
auto['mo'] = auto['count_date2'].dt.month
auto['day'] = auto['count_date2'].dt.dayofweek

# Drop observations where:
# a) Traffic light activated after 2004
# b) Countdown never installed
auto_clean = auto[
    (auto['active_year'] < 2004) &
    (auto['install_year'].notna())
].copy()

# Generate east/west and north/south indicators
auto_clean['ew_ind'] = (auto_clean['latitude'] > 313104822.1).astype(int)
auto_clean['ns_ind'] = (auto_clean['longtitude'] > 4841254464).astype(int)

# Generate countdown indicator
auto_clean['countdown'] = (auto_clean['installdate2'] <= auto_clean['count_date2']).astype(int)

# Drop if count is missing
auto_clean = auto_clean[auto_clean['tot_count'].notna()].copy()

# Generate nvar - how often an intersection is counted
nvar_counts = auto_clean.groupby('id_pcs')['id_pcs'].transform('count')
auto_clean['nvar1'] = nvar_counts

# Create categorical variables for fixed effects
auto_clean['day_cat'] = auto_clean['day'].astype(str)
auto_clean['mo_cat'] = auto_clean['mo'].astype(str)
auto_clean['yr_cat'] = auto_clean['yr'].astype(str)
auto_clean['street1_cat'] = auto_clean['street1'].fillna('Unknown')
auto_clean['street2_cat'] = auto_clean['street2'].fillna('Unknown')

# Create id_pcs string for clustering
auto_clean['id_pcs_str'] = auto_clean['id_pcs'].astype(str)

print(f"  Automobile data: {len(auto_clean)} observations")
print(f"  Treatment rate: {auto_clean['countdown'].mean():.2%}")
print(f"  Unique intersections: {auto_clean['id_pcs'].nunique()}")

# ============================================================================
# PEDESTRIAN FLOW SPECIFICATIONS (Table 8)
# ============================================================================

print("\n" + "=" * 70)
print("PEDESTRIAN FLOW SPECIFICATIONS")
print("=" * 70)

# --- BASELINE (Table 8, Col 7 - full specification) ---
print("\n--- Baseline Specification (Table 8 Col 7) ---")
try:
    model = pf.feols("ped_vol8hr ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ns_ind + ew_ind + C(main_cat) + C(side1route_cat)",
                     data=ped_clean, vcov='hetero')
    add_result(
        spec_id='baseline',
        spec_tree_path='methods/cross_sectional_ols.md',
        outcome_var='ped_vol8hr',
        treatment_var='countdown',
        model=model,
        df_used=ped_clean,
        sample_desc='Pedestrian counts at intersections with PCS installed, traffic light activated before 2004',
        fixed_effects='day + month + year + main street + side route',
        controls_desc='ns_ind, ew_ind (location indicators)',
        cluster_var='robust',
        model_type='OLS'
    )
except Exception as e:
    print(f"  Baseline ERROR: {e}")

# --- CONTROL VARIATIONS ---
print("\n--- Control Variations ---")

# No controls
try:
    model = pf.feols("ped_vol8hr ~ countdown", data=ped_clean, vcov='hetero')
    add_result('robust/control/none', 'robustness/control_progression.md',
               'ped_vol8hr', 'countdown', model, ped_clean,
               'Full sample', 'None', 'None', 'robust', 'OLS')
except Exception as e:
    print(f"  ERROR: {e}")

# Day FE only
try:
    model = pf.feols("ped_vol8hr ~ countdown + C(day_cat)", data=ped_clean, vcov='hetero')
    add_result('robust/control/day_only', 'robustness/control_progression.md',
               'ped_vol8hr', 'countdown', model, ped_clean,
               'Full sample', 'day', 'None', 'robust', 'OLS')
except Exception as e:
    print(f"  ERROR: {e}")

# Day + Month FE
try:
    model = pf.feols("ped_vol8hr ~ countdown + C(day_cat) + C(mo_cat)", data=ped_clean, vcov='hetero')
    add_result('robust/control/day_month', 'robustness/control_progression.md',
               'ped_vol8hr', 'countdown', model, ped_clean,
               'Full sample', 'day + month', 'None', 'robust', 'OLS')
except Exception as e:
    print(f"  ERROR: {e}")

# Day + Month + Year FE
try:
    model = pf.feols("ped_vol8hr ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat)", data=ped_clean, vcov='hetero')
    add_result('robust/control/day_month_year', 'robustness/control_progression.md',
               'ped_vol8hr', 'countdown', model, ped_clean,
               'Full sample', 'day + month + year', 'None', 'robust', 'OLS')
except Exception as e:
    print(f"  ERROR: {e}")

# Day + Month + Year + Location
try:
    model = pf.feols("ped_vol8hr ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ns_ind + ew_ind", data=ped_clean, vcov='hetero')
    add_result('robust/control/time_location', 'robustness/control_progression.md',
               'ped_vol8hr', 'countdown', model, ped_clean,
               'Full sample', 'day + month + year', 'ns_ind, ew_ind', 'robust', 'OLS')
except Exception as e:
    print(f"  ERROR: {e}")

# Day + Month + Year + Location + Main Street
try:
    model = pf.feols("ped_vol8hr ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ns_ind + ew_ind + C(main_cat)", data=ped_clean, vcov='hetero')
    add_result('robust/control/time_location_main', 'robustness/control_progression.md',
               'ped_vol8hr', 'countdown', model, ped_clean,
               'Full sample', 'day + month + year + main street', 'ns_ind, ew_ind', 'robust', 'OLS')
except Exception as e:
    print(f"  ERROR: {e}")

# Drop controls one at a time from full spec
print("\n--- Leave-One-Out Controls ---")

# Drop location controls
try:
    model = pf.feols("ped_vol8hr ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + C(main_cat) + C(side1route_cat)",
                     data=ped_clean, vcov='hetero')
    add_result('robust/control/drop_location', 'robustness/leave_one_out.md',
               'ped_vol8hr', 'countdown', model, ped_clean,
               'Full sample', 'day + month + year + main + side route', 'None (dropped location)', 'robust', 'OLS')
except Exception as e:
    print(f"  ERROR: {e}")

# Drop main street FE
try:
    model = pf.feols("ped_vol8hr ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ns_ind + ew_ind + C(side1route_cat)",
                     data=ped_clean, vcov='hetero')
    add_result('robust/control/drop_main', 'robustness/leave_one_out.md',
               'ped_vol8hr', 'countdown', model, ped_clean,
               'Full sample', 'day + month + year + side route', 'ns_ind, ew_ind', 'robust', 'OLS')
except Exception as e:
    print(f"  ERROR: {e}")

# Drop side route FE
try:
    model = pf.feols("ped_vol8hr ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ns_ind + ew_ind + C(main_cat)",
                     data=ped_clean, vcov='hetero')
    add_result('robust/control/drop_sideroute', 'robustness/leave_one_out.md',
               'ped_vol8hr', 'countdown', model, ped_clean,
               'Full sample', 'day + month + year + main', 'ns_ind, ew_ind', 'robust', 'OLS')
except Exception as e:
    print(f"  ERROR: {e}")

# Drop year FE
try:
    model = pf.feols("ped_vol8hr ~ countdown + C(day_cat) + C(mo_cat) + ns_ind + ew_ind + C(main_cat) + C(side1route_cat)",
                     data=ped_clean, vcov='hetero')
    add_result('robust/control/drop_year', 'robustness/leave_one_out.md',
               'ped_vol8hr', 'countdown', model, ped_clean,
               'Full sample', 'day + month + main + side route', 'ns_ind, ew_ind', 'robust', 'OLS')
except Exception as e:
    print(f"  ERROR: {e}")

# Drop month FE
try:
    model = pf.feols("ped_vol8hr ~ countdown + C(day_cat) + C(yr_cat) + ns_ind + ew_ind + C(main_cat) + C(side1route_cat)",
                     data=ped_clean, vcov='hetero')
    add_result('robust/control/drop_month', 'robustness/leave_one_out.md',
               'ped_vol8hr', 'countdown', model, ped_clean,
               'Full sample', 'day + year + main + side route', 'ns_ind, ew_ind', 'robust', 'OLS')
except Exception as e:
    print(f"  ERROR: {e}")

# Drop day FE
try:
    model = pf.feols("ped_vol8hr ~ countdown + C(mo_cat) + C(yr_cat) + ns_ind + ew_ind + C(main_cat) + C(side1route_cat)",
                     data=ped_clean, vcov='hetero')
    add_result('robust/control/drop_day', 'robustness/leave_one_out.md',
               'ped_vol8hr', 'countdown', model, ped_clean,
               'Full sample', 'month + year + main + side route', 'ns_ind, ew_ind', 'robust', 'OLS')
except Exception as e:
    print(f"  ERROR: {e}")

# --- SAMPLE RESTRICTIONS ---
print("\n--- Sample Restrictions ---")

# By year
for year in sorted(ped_clean['yr'].dropna().unique()):
    if pd.notna(year):
        try:
            df_sub = ped_clean[ped_clean['yr'] != year]
            if len(df_sub) > 100 and df_sub['countdown'].nunique() > 1:
                model = pf.feols("ped_vol8hr ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ns_ind + ew_ind + C(main_cat) + C(side1route_cat)",
                                data=df_sub, vcov='hetero')
                add_result(f'robust/sample/drop_year_{int(year)}', 'robustness/sample_restrictions.md',
                          'ped_vol8hr', 'countdown', model, df_sub,
                          f'Excluding year {int(year)}', 'day + month + year + main + side route', 'ns_ind, ew_ind', 'robust', 'OLS')
        except Exception as e:
            print(f"  Year {year} ERROR: {e}")

# By location quadrant
try:
    df_east = ped_clean[ped_clean['ew_ind'] == 1]
    if len(df_east) > 50:
        model = pf.feols("ped_vol8hr ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ns_ind + C(main_cat) + C(side1route_cat)",
                        data=df_east, vcov='hetero')
        add_result('robust/sample/east_only', 'robustness/sample_restrictions.md',
                  'ped_vol8hr', 'countdown', model, df_east,
                  'East side only', 'day + month + year + main + side route', 'ns_ind', 'robust', 'OLS')
except Exception as e:
    print(f"  East ERROR: {e}")

try:
    df_west = ped_clean[ped_clean['ew_ind'] == 0]
    if len(df_west) > 50:
        model = pf.feols("ped_vol8hr ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ns_ind + C(main_cat) + C(side1route_cat)",
                        data=df_west, vcov='hetero')
        add_result('robust/sample/west_only', 'robustness/sample_restrictions.md',
                  'ped_vol8hr', 'countdown', model, df_west,
                  'West side only', 'day + month + year + main + side route', 'ns_ind', 'robust', 'OLS')
except Exception as e:
    print(f"  West ERROR: {e}")

try:
    df_north = ped_clean[ped_clean['ns_ind'] == 1]
    if len(df_north) > 50:
        model = pf.feols("ped_vol8hr ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ew_ind + C(main_cat) + C(side1route_cat)",
                        data=df_north, vcov='hetero')
        add_result('robust/sample/north_only', 'robustness/sample_restrictions.md',
                  'ped_vol8hr', 'countdown', model, df_north,
                  'North side only', 'day + month + year + main + side route', 'ew_ind', 'robust', 'OLS')
except Exception as e:
    print(f"  North ERROR: {e}")

try:
    df_south = ped_clean[ped_clean['ns_ind'] == 0]
    if len(df_south) > 50:
        model = pf.feols("ped_vol8hr ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ew_ind + C(main_cat) + C(side1route_cat)",
                        data=df_south, vcov='hetero')
        add_result('robust/sample/south_only', 'robustness/sample_restrictions.md',
                  'ped_vol8hr', 'countdown', model, df_south,
                  'South side only', 'day + month + year + main + side route', 'ew_ind', 'robust', 'OLS')
except Exception as e:
    print(f"  South ERROR: {e}")

# Outlier treatment
print("\n--- Outlier Treatment ---")

# Winsorize at different percentiles
for pct in [1, 5, 10]:
    try:
        df_wins = ped_clean.copy()
        lower = df_wins['ped_vol8hr'].quantile(pct/100)
        upper = df_wins['ped_vol8hr'].quantile(1-pct/100)
        df_wins['ped_vol8hr_wins'] = df_wins['ped_vol8hr'].clip(lower=lower, upper=upper)

        model = pf.feols("ped_vol8hr_wins ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ns_ind + ew_ind + C(main_cat) + C(side1route_cat)",
                        data=df_wins, vcov='hetero')
        add_result(f'robust/sample/winsorize_{pct}pct', 'robustness/sample_restrictions.md',
                  'ped_vol8hr_wins', 'countdown', model, df_wins,
                  f'Outcome winsorized at {pct}%', 'day + month + year + main + side route', 'ns_ind, ew_ind', 'robust', 'OLS')
    except Exception as e:
        print(f"  Winsor {pct}% ERROR: {e}")

# Trim extreme values
try:
    df_trim = ped_clean[
        (ped_clean['ped_vol8hr'] > ped_clean['ped_vol8hr'].quantile(0.01)) &
        (ped_clean['ped_vol8hr'] < ped_clean['ped_vol8hr'].quantile(0.99))
    ].copy()
    model = pf.feols("ped_vol8hr ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ns_ind + ew_ind + C(main_cat) + C(side1route_cat)",
                    data=df_trim, vcov='hetero')
    add_result('robust/sample/trim_1pct', 'robustness/sample_restrictions.md',
              'ped_vol8hr', 'countdown', model, df_trim,
              'Trimmed top/bottom 1%', 'day + month + year + main + side route', 'ns_ind, ew_ind', 'robust', 'OLS')
except Exception as e:
    print(f"  Trim ERROR: {e}")

# --- FUNCTIONAL FORM ---
print("\n--- Functional Form ---")

# Log outcome
try:
    ped_clean['log_ped_vol'] = np.log(ped_clean['ped_vol8hr'] + 1)
    model = pf.feols("log_ped_vol ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ns_ind + ew_ind + C(main_cat) + C(side1route_cat)",
                    data=ped_clean, vcov='hetero')
    add_result('robust/funcform/log_outcome', 'robustness/functional_form.md',
              'log_ped_vol8hr', 'countdown', model, ped_clean,
              'Full sample', 'day + month + year + main + side route', 'ns_ind, ew_ind', 'robust', 'OLS')
except Exception as e:
    print(f"  Log ERROR: {e}")

# Inverse hyperbolic sine
try:
    ped_clean['ihs_ped_vol'] = np.arcsinh(ped_clean['ped_vol8hr'])
    model = pf.feols("ihs_ped_vol ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ns_ind + ew_ind + C(main_cat) + C(side1route_cat)",
                    data=ped_clean, vcov='hetero')
    add_result('robust/funcform/ihs_outcome', 'robustness/functional_form.md',
              'ihs_ped_vol8hr', 'countdown', model, ped_clean,
              'Full sample', 'day + month + year + main + side route', 'ns_ind, ew_ind', 'robust', 'OLS')
except Exception as e:
    print(f"  IHS ERROR: {e}")

# Square root
try:
    ped_clean['sqrt_ped_vol'] = np.sqrt(ped_clean['ped_vol8hr'])
    model = pf.feols("sqrt_ped_vol ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ns_ind + ew_ind + C(main_cat) + C(side1route_cat)",
                    data=ped_clean, vcov='hetero')
    add_result('robust/funcform/sqrt_outcome', 'robustness/functional_form.md',
              'sqrt_ped_vol8hr', 'countdown', model, ped_clean,
              'Full sample', 'day + month + year + main + side route', 'ns_ind, ew_ind', 'robust', 'OLS')
except Exception as e:
    print(f"  Sqrt ERROR: {e}")

# ============================================================================
# AUTOMOBILE FLOW SPECIFICATIONS (Table 9)
# ============================================================================

print("\n" + "=" * 70)
print("AUTOMOBILE FLOW SPECIFICATIONS")
print("=" * 70)

# --- BASELINE (Table 9, Col 7 - full specification) ---
print("\n--- Baseline Specification (Table 9 Col 7) ---")
try:
    model = pf.feols("tot_count ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ew_ind + ns_ind + C(street1_cat) + C(street2_cat) + nvar1",
                     data=auto_clean, vcov={'CRV1': 'id_pcs_str'})
    add_result(
        spec_id='baseline_auto',
        spec_tree_path='methods/cross_sectional_ols.md',
        outcome_var='tot_count',
        treatment_var='countdown',
        model=model,
        df_used=auto_clean,
        sample_desc='Automobile counts at intersections with PCS installed, traffic light activated before 2004',
        fixed_effects='day + month + year + street1 + street2',
        controls_desc='ew_ind, ns_ind, nvar1',
        cluster_var='id_pcs',
        model_type='OLS'
    )
except Exception as e:
    print(f"  Baseline Auto ERROR: {e}")

# --- CONTROL VARIATIONS ---
print("\n--- Control Variations (Auto) ---")

# No controls
try:
    model = pf.feols("tot_count ~ countdown", data=auto_clean, vcov={'CRV1': 'id_pcs_str'})
    add_result('robust/control/none_auto', 'robustness/control_progression.md',
               'tot_count', 'countdown', model, auto_clean,
               'Full sample auto', 'None', 'None', 'id_pcs', 'OLS')
except Exception as e:
    print(f"  ERROR: {e}")

# Day only
try:
    model = pf.feols("tot_count ~ countdown + C(day_cat)", data=auto_clean, vcov={'CRV1': 'id_pcs_str'})
    add_result('robust/control/day_only_auto', 'robustness/control_progression.md',
               'tot_count', 'countdown', model, auto_clean,
               'Full sample auto', 'day', 'None', 'id_pcs', 'OLS')
except Exception as e:
    print(f"  ERROR: {e}")

# Day + Month
try:
    model = pf.feols("tot_count ~ countdown + C(day_cat) + C(mo_cat)", data=auto_clean, vcov={'CRV1': 'id_pcs_str'})
    add_result('robust/control/day_month_auto', 'robustness/control_progression.md',
               'tot_count', 'countdown', model, auto_clean,
               'Full sample auto', 'day + month', 'None', 'id_pcs', 'OLS')
except Exception as e:
    print(f"  ERROR: {e}")

# Day + Month + Year
try:
    model = pf.feols("tot_count ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat)", data=auto_clean, vcov={'CRV1': 'id_pcs_str'})
    add_result('robust/control/day_month_year_auto', 'robustness/control_progression.md',
               'tot_count', 'countdown', model, auto_clean,
               'Full sample auto', 'day + month + year', 'None', 'id_pcs', 'OLS')
except Exception as e:
    print(f"  ERROR: {e}")

# Day + Month + Year + Location + Street1
try:
    model = pf.feols("tot_count ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ew_ind + ns_ind + C(street1_cat)",
                     data=auto_clean, vcov={'CRV1': 'id_pcs_str'})
    add_result('robust/control/time_loc_street1_auto', 'robustness/control_progression.md',
               'tot_count', 'countdown', model, auto_clean,
               'Full sample auto', 'day + month + year + street1', 'ew_ind, ns_ind', 'id_pcs', 'OLS')
except Exception as e:
    print(f"  ERROR: {e}")

# Day + Month + Year + Location + Both Streets
try:
    model = pf.feols("tot_count ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ew_ind + ns_ind + C(street1_cat) + C(street2_cat)",
                     data=auto_clean, vcov={'CRV1': 'id_pcs_str'})
    add_result('robust/control/time_loc_streets_auto', 'robustness/control_progression.md',
               'tot_count', 'countdown', model, auto_clean,
               'Full sample auto', 'day + month + year + street1 + street2', 'ew_ind, ns_ind', 'id_pcs', 'OLS')
except Exception as e:
    print(f"  ERROR: {e}")

# --- LEAVE-ONE-OUT (Auto) ---
print("\n--- Leave-One-Out Controls (Auto) ---")

# Drop nvar1
try:
    model = pf.feols("tot_count ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ew_ind + ns_ind + C(street1_cat) + C(street2_cat)",
                     data=auto_clean, vcov={'CRV1': 'id_pcs_str'})
    add_result('robust/control/drop_nvar_auto', 'robustness/leave_one_out.md',
               'tot_count', 'countdown', model, auto_clean,
               'Full sample auto', 'day + month + year + street1 + street2', 'ew_ind, ns_ind (dropped nvar)', 'id_pcs', 'OLS')
except Exception as e:
    print(f"  ERROR: {e}")

# Drop street2
try:
    model = pf.feols("tot_count ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ew_ind + ns_ind + C(street1_cat) + nvar1",
                     data=auto_clean, vcov={'CRV1': 'id_pcs_str'})
    add_result('robust/control/drop_street2_auto', 'robustness/leave_one_out.md',
               'tot_count', 'countdown', model, auto_clean,
               'Full sample auto', 'day + month + year + street1', 'ew_ind, ns_ind, nvar1', 'id_pcs', 'OLS')
except Exception as e:
    print(f"  ERROR: {e}")

# Drop street1
try:
    model = pf.feols("tot_count ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ew_ind + ns_ind + C(street2_cat) + nvar1",
                     data=auto_clean, vcov={'CRV1': 'id_pcs_str'})
    add_result('robust/control/drop_street1_auto', 'robustness/leave_one_out.md',
               'tot_count', 'countdown', model, auto_clean,
               'Full sample auto', 'day + month + year + street2', 'ew_ind, ns_ind, nvar1', 'id_pcs', 'OLS')
except Exception as e:
    print(f"  ERROR: {e}")

# Drop location
try:
    model = pf.feols("tot_count ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + C(street1_cat) + C(street2_cat) + nvar1",
                     data=auto_clean, vcov={'CRV1': 'id_pcs_str'})
    add_result('robust/control/drop_location_auto', 'robustness/leave_one_out.md',
               'tot_count', 'countdown', model, auto_clean,
               'Full sample auto', 'day + month + year + street1 + street2', 'nvar1 (dropped location)', 'id_pcs', 'OLS')
except Exception as e:
    print(f"  ERROR: {e}")

# Drop year
try:
    model = pf.feols("tot_count ~ countdown + C(day_cat) + C(mo_cat) + ew_ind + ns_ind + C(street1_cat) + C(street2_cat) + nvar1",
                     data=auto_clean, vcov={'CRV1': 'id_pcs_str'})
    add_result('robust/control/drop_year_auto', 'robustness/leave_one_out.md',
               'tot_count', 'countdown', model, auto_clean,
               'Full sample auto', 'day + month + street1 + street2', 'ew_ind, ns_ind, nvar1', 'id_pcs', 'OLS')
except Exception as e:
    print(f"  ERROR: {e}")

# --- INFERENCE VARIATIONS ---
print("\n--- Inference Variations ---")

# Heteroskedasticity-robust only
try:
    model = pf.feols("tot_count ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ew_ind + ns_ind + C(street1_cat) + C(street2_cat) + nvar1",
                     data=auto_clean, vcov='hetero')
    add_result('robust/inference/robust_se_auto', 'robustness/inference_alternatives.md',
               'tot_count', 'countdown', model, auto_clean,
               'Full sample auto', 'day + month + year + street1 + street2', 'ew_ind, ns_ind, nvar1', 'robust', 'OLS')
except Exception as e:
    print(f"  ERROR: {e}")

# IID SEs
try:
    model = pf.feols("tot_count ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ew_ind + ns_ind + C(street1_cat) + C(street2_cat) + nvar1",
                     data=auto_clean, vcov='iid')
    add_result('robust/inference/iid_se_auto', 'robustness/inference_alternatives.md',
               'tot_count', 'countdown', model, auto_clean,
               'Full sample auto', 'day + month + year + street1 + street2', 'ew_ind, ns_ind, nvar1', 'iid', 'OLS')
except Exception as e:
    print(f"  ERROR: {e}")

# --- SAMPLE RESTRICTIONS (Auto) ---
print("\n--- Sample Restrictions (Auto) ---")

# By year
for year in sorted(auto_clean['yr'].dropna().unique()):
    if pd.notna(year):
        try:
            df_sub = auto_clean[auto_clean['yr'] != year]
            if len(df_sub) > 500 and df_sub['countdown'].nunique() > 1:
                model = pf.feols("tot_count ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ew_ind + ns_ind + C(street1_cat) + C(street2_cat) + nvar1",
                                data=df_sub, vcov={'CRV1': 'id_pcs_str'})
                add_result(f'robust/sample/drop_year_{int(year)}_auto', 'robustness/sample_restrictions.md',
                          'tot_count', 'countdown', model, df_sub,
                          f'Auto, excluding year {int(year)}', 'day + month + year + street1 + street2', 'ew_ind, ns_ind, nvar1', 'id_pcs', 'OLS')
        except Exception as e:
            print(f"  Year {year} auto ERROR: {e}")

# By location
try:
    df_east = auto_clean[auto_clean['ew_ind'] == 1]
    if len(df_east) > 100:
        model = pf.feols("tot_count ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ns_ind + C(street1_cat) + C(street2_cat) + nvar1",
                        data=df_east, vcov={'CRV1': 'id_pcs_str'})
        add_result('robust/sample/east_only_auto', 'robustness/sample_restrictions.md',
                  'tot_count', 'countdown', model, df_east,
                  'Auto, east side only', 'day + month + year + street1 + street2', 'ns_ind, nvar1', 'id_pcs', 'OLS')
except Exception as e:
    print(f"  East auto ERROR: {e}")

try:
    df_west = auto_clean[auto_clean['ew_ind'] == 0]
    if len(df_west) > 100:
        model = pf.feols("tot_count ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ns_ind + C(street1_cat) + C(street2_cat) + nvar1",
                        data=df_west, vcov={'CRV1': 'id_pcs_str'})
        add_result('robust/sample/west_only_auto', 'robustness/sample_restrictions.md',
                  'tot_count', 'countdown', model, df_west,
                  'Auto, west side only', 'day + month + year + street1 + street2', 'ns_ind, nvar1', 'id_pcs', 'OLS')
except Exception as e:
    print(f"  West auto ERROR: {e}")

try:
    df_north = auto_clean[auto_clean['ns_ind'] == 1]
    if len(df_north) > 100:
        model = pf.feols("tot_count ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ew_ind + C(street1_cat) + C(street2_cat) + nvar1",
                        data=df_north, vcov={'CRV1': 'id_pcs_str'})
        add_result('robust/sample/north_only_auto', 'robustness/sample_restrictions.md',
                  'tot_count', 'countdown', model, df_north,
                  'Auto, north side only', 'day + month + year + street1 + street2', 'ew_ind, nvar1', 'id_pcs', 'OLS')
except Exception as e:
    print(f"  North auto ERROR: {e}")

try:
    df_south = auto_clean[auto_clean['ns_ind'] == 0]
    if len(df_south) > 100:
        model = pf.feols("tot_count ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ew_ind + C(street1_cat) + C(street2_cat) + nvar1",
                        data=df_south, vcov={'CRV1': 'id_pcs_str'})
        add_result('robust/sample/south_only_auto', 'robustness/sample_restrictions.md',
                  'tot_count', 'countdown', model, df_south,
                  'Auto, south side only', 'day + month + year + street1 + street2', 'ew_ind, nvar1', 'id_pcs', 'OLS')
except Exception as e:
    print(f"  South auto ERROR: {e}")

# --- OUTLIER TREATMENT (Auto) ---
print("\n--- Outlier Treatment (Auto) ---")

for pct in [1, 5, 10]:
    try:
        df_wins = auto_clean.copy()
        lower = df_wins['tot_count'].quantile(pct/100)
        upper = df_wins['tot_count'].quantile(1-pct/100)
        df_wins['tot_count_wins'] = df_wins['tot_count'].clip(lower=lower, upper=upper)

        model = pf.feols("tot_count_wins ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ew_ind + ns_ind + C(street1_cat) + C(street2_cat) + nvar1",
                        data=df_wins, vcov={'CRV1': 'id_pcs_str'})
        add_result(f'robust/sample/winsorize_{pct}pct_auto', 'robustness/sample_restrictions.md',
                  'tot_count_wins', 'countdown', model, df_wins,
                  f'Auto, outcome winsorized at {pct}%', 'day + month + year + street1 + street2', 'ew_ind, ns_ind, nvar1', 'id_pcs', 'OLS')
    except Exception as e:
        print(f"  Winsor {pct}% auto ERROR: {e}")

# Trim
try:
    df_trim = auto_clean[
        (auto_clean['tot_count'] > auto_clean['tot_count'].quantile(0.01)) &
        (auto_clean['tot_count'] < auto_clean['tot_count'].quantile(0.99))
    ].copy()
    model = pf.feols("tot_count ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ew_ind + ns_ind + C(street1_cat) + C(street2_cat) + nvar1",
                    data=df_trim, vcov={'CRV1': 'id_pcs_str'})
    add_result('robust/sample/trim_1pct_auto', 'robustness/sample_restrictions.md',
              'tot_count', 'countdown', model, df_trim,
              'Auto, trimmed top/bottom 1%', 'day + month + year + street1 + street2', 'ew_ind, ns_ind, nvar1', 'id_pcs', 'OLS')
except Exception as e:
    print(f"  Trim auto ERROR: {e}")

# --- FUNCTIONAL FORM (Auto) ---
print("\n--- Functional Form (Auto) ---")

# Log outcome
try:
    auto_clean['log_tot_count'] = np.log(auto_clean['tot_count'] + 1)
    model = pf.feols("log_tot_count ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ew_ind + ns_ind + C(street1_cat) + C(street2_cat) + nvar1",
                    data=auto_clean, vcov={'CRV1': 'id_pcs_str'})
    add_result('robust/funcform/log_outcome_auto', 'robustness/functional_form.md',
              'log_tot_count', 'countdown', model, auto_clean,
              'Auto, full sample', 'day + month + year + street1 + street2', 'ew_ind, ns_ind, nvar1', 'id_pcs', 'OLS')
except Exception as e:
    print(f"  Log auto ERROR: {e}")

# IHS
try:
    auto_clean['ihs_tot_count'] = np.arcsinh(auto_clean['tot_count'])
    model = pf.feols("ihs_tot_count ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ew_ind + ns_ind + C(street1_cat) + C(street2_cat) + nvar1",
                    data=auto_clean, vcov={'CRV1': 'id_pcs_str'})
    add_result('robust/funcform/ihs_outcome_auto', 'robustness/functional_form.md',
              'ihs_tot_count', 'countdown', model, auto_clean,
              'Auto, full sample', 'day + month + year + street1 + street2', 'ew_ind, ns_ind, nvar1', 'id_pcs', 'OLS')
except Exception as e:
    print(f"  IHS auto ERROR: {e}")

# Sqrt
try:
    auto_clean['sqrt_tot_count'] = np.sqrt(auto_clean['tot_count'])
    model = pf.feols("sqrt_tot_count ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ew_ind + ns_ind + C(street1_cat) + C(street2_cat) + nvar1",
                    data=auto_clean, vcov={'CRV1': 'id_pcs_str'})
    add_result('robust/funcform/sqrt_outcome_auto', 'robustness/functional_form.md',
              'sqrt_tot_count', 'countdown', model, auto_clean,
              'Auto, full sample', 'day + month + year + street1 + street2', 'ew_ind, ns_ind, nvar1', 'id_pcs', 'OLS')
except Exception as e:
    print(f"  Sqrt auto ERROR: {e}")

# --- ALTERNATIVE OUTCOMES ---
print("\n--- Alternative Outcomes (Auto) ---")

# AM peak volume
try:
    df_am = auto_clean[auto_clean['am_pk_vol'].notna()].copy()
    model = pf.feols("am_pk_vol ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ew_ind + ns_ind + C(street1_cat) + C(street2_cat) + nvar1",
                    data=df_am, vcov={'CRV1': 'id_pcs_str'})
    add_result('robust/outcome/am_peak_vol', 'robustness/measurement.md',
              'am_pk_vol', 'countdown', model, df_am,
              'Auto, AM peak volume', 'day + month + year + street1 + street2', 'ew_ind, ns_ind, nvar1', 'id_pcs', 'OLS')
except Exception as e:
    print(f"  AM peak ERROR: {e}")

# PM peak volume
try:
    df_pm = auto_clean[auto_clean['pm_pk_vol'].notna()].copy()
    model = pf.feols("pm_pk_vol ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ew_ind + ns_ind + C(street1_cat) + C(street2_cat) + nvar1",
                    data=df_pm, vcov={'CRV1': 'id_pcs_str'})
    add_result('robust/outcome/pm_peak_vol', 'robustness/measurement.md',
              'pm_pk_vol', 'countdown', model, df_pm,
              'Auto, PM peak volume', 'day + month + year + street1 + street2', 'ew_ind, ns_ind, nvar1', 'id_pcs', 'OLS')
except Exception as e:
    print(f"  PM peak ERROR: {e}")

# Off-peak volume
try:
    df_off = auto_clean[auto_clean['off_pk_vol'].notna()].copy()
    model = pf.feols("off_pk_vol ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ew_ind + ns_ind + C(street1_cat) + C(street2_cat) + nvar1",
                    data=df_off, vcov={'CRV1': 'id_pcs_str'})
    add_result('robust/outcome/off_peak_vol', 'robustness/measurement.md',
              'off_pk_vol', 'countdown', model, df_off,
              'Auto, off-peak volume', 'day + month + year + street1 + street2', 'ew_ind, ns_ind, nvar1', 'id_pcs', 'OLS')
except Exception as e:
    print(f"  Off-peak ERROR: {e}")

# --- HETEROGENEITY ---
print("\n--- Heterogeneity Analysis ---")

# Interaction with location
try:
    model = pf.feols("ped_vol8hr ~ countdown * ew_ind + C(day_cat) + C(mo_cat) + C(yr_cat) + ns_ind + C(main_cat) + C(side1route_cat)",
                    data=ped_clean, vcov='hetero')
    add_result('robust/heterogeneity/ew_interact_ped', 'robustness/heterogeneity.md',
              'ped_vol8hr', 'countdown', model, ped_clean,
              'Pedestrian, heterogeneity by E/W location', 'day + month + year + main + side route', 'ns_ind, countdown*ew_ind', 'robust', 'OLS',
              coef_var='countdown')
except Exception as e:
    print(f"  EW interact ped ERROR: {e}")

try:
    model = pf.feols("ped_vol8hr ~ countdown * ns_ind + C(day_cat) + C(mo_cat) + C(yr_cat) + ew_ind + C(main_cat) + C(side1route_cat)",
                    data=ped_clean, vcov='hetero')
    add_result('robust/heterogeneity/ns_interact_ped', 'robustness/heterogeneity.md',
              'ped_vol8hr', 'countdown', model, ped_clean,
              'Pedestrian, heterogeneity by N/S location', 'day + month + year + main + side route', 'ew_ind, countdown*ns_ind', 'robust', 'OLS',
              coef_var='countdown')
except Exception as e:
    print(f"  NS interact ped ERROR: {e}")

# Auto heterogeneity
try:
    model = pf.feols("tot_count ~ countdown * ew_ind + C(day_cat) + C(mo_cat) + C(yr_cat) + ns_ind + C(street1_cat) + C(street2_cat) + nvar1",
                    data=auto_clean, vcov={'CRV1': 'id_pcs_str'})
    add_result('robust/heterogeneity/ew_interact_auto', 'robustness/heterogeneity.md',
              'tot_count', 'countdown', model, auto_clean,
              'Auto, heterogeneity by E/W location', 'day + month + year + street1 + street2', 'ns_ind, nvar1, countdown*ew_ind', 'id_pcs', 'OLS',
              coef_var='countdown')
except Exception as e:
    print(f"  EW interact auto ERROR: {e}")

try:
    model = pf.feols("tot_count ~ countdown * ns_ind + C(day_cat) + C(mo_cat) + C(yr_cat) + ew_ind + C(street1_cat) + C(street2_cat) + nvar1",
                    data=auto_clean, vcov={'CRV1': 'id_pcs_str'})
    add_result('robust/heterogeneity/ns_interact_auto', 'robustness/heterogeneity.md',
              'tot_count', 'countdown', model, auto_clean,
              'Auto, heterogeneity by N/S location', 'day + month + year + street1 + street2', 'ew_ind, nvar1, countdown*ns_ind', 'id_pcs', 'OLS',
              coef_var='countdown')
except Exception as e:
    print(f"  NS interact auto ERROR: {e}")

# Heterogeneity by count frequency
try:
    auto_clean['high_freq'] = (auto_clean['nvar1'] > auto_clean['nvar1'].median()).astype(int)
    model = pf.feols("tot_count ~ countdown * high_freq + C(day_cat) + C(mo_cat) + C(yr_cat) + ew_ind + ns_ind + C(street1_cat) + C(street2_cat)",
                    data=auto_clean, vcov={'CRV1': 'id_pcs_str'})
    add_result('robust/heterogeneity/high_freq_auto', 'robustness/heterogeneity.md',
              'tot_count', 'countdown', model, auto_clean,
              'Auto, heterogeneity by count frequency', 'day + month + year + street1 + street2', 'ew_ind, ns_ind, countdown*high_freq', 'id_pcs', 'OLS',
              coef_var='countdown')
except Exception as e:
    print(f"  High freq auto ERROR: {e}")

# --- PLACEBO TESTS ---
print("\n--- Placebo Tests ---")

# Future treatment (pseudo-placebo)
try:
    ped_clean['installdate_fake'] = ped_clean['installdate2'] - pd.DateOffset(years=1)
    ped_clean['countdown_fake'] = (ped_clean['installdate_fake'] <= ped_clean['count_date']).astype(int)

    df_pre = ped_clean[ped_clean['countdown'] == 0].copy()
    if len(df_pre) > 100 and df_pre['countdown_fake'].sum() > 10:
        model = pf.feols("ped_vol8hr ~ countdown_fake + C(day_cat) + C(mo_cat) + C(yr_cat) + ns_ind + ew_ind + C(main_cat) + C(side1route_cat)",
                        data=df_pre, vcov='hetero')
        add_result('robust/placebo/fake_timing_ped', 'robustness/placebo_tests.md',
                  'ped_vol8hr', 'countdown_fake', model, df_pre,
                  'Pedestrian, placebo treatment 1 year early', 'day + month + year + main + side route', 'ns_ind, ew_ind', 'robust', 'OLS',
                  coef_var='countdown_fake')
except Exception as e:
    print(f"  Placebo ped ERROR: {e}")

try:
    auto_clean['installdate_fake'] = auto_clean['installdate2'] - pd.DateOffset(years=1)
    auto_clean['countdown_fake'] = (auto_clean['installdate_fake'] <= auto_clean['count_date2']).astype(int)

    df_pre = auto_clean[auto_clean['countdown'] == 0].copy()
    if len(df_pre) > 100 and df_pre['countdown_fake'].sum() > 10:
        model = pf.feols("tot_count ~ countdown_fake + C(day_cat) + C(mo_cat) + C(yr_cat) + ew_ind + ns_ind + C(street1_cat) + C(street2_cat) + nvar1",
                        data=df_pre, vcov={'CRV1': 'id_pcs_str'})
        add_result('robust/placebo/fake_timing_auto', 'robustness/placebo_tests.md',
                  'tot_count', 'countdown_fake', model, df_pre,
                  'Auto, placebo treatment 1 year early', 'day + month + year + street1 + street2', 'ew_ind, ns_ind, nvar1', 'id_pcs', 'OLS',
                  coef_var='countdown_fake')
except Exception as e:
    print(f"  Placebo auto ERROR: {e}")

# --- ADDITIONAL VARIATIONS ---
print("\n--- Additional Variations ---")

# High volume vs low volume intersections (pedestrian)
try:
    median_vol = ped_clean['ped_vol8hr'].median()
    df_high = ped_clean[ped_clean['ped_vol8hr'] >= median_vol].copy()
    model = pf.feols("ped_vol8hr ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ns_ind + ew_ind + C(main_cat) + C(side1route_cat)",
                    data=df_high, vcov='hetero')
    add_result('robust/sample/high_volume_ped', 'robustness/sample_restrictions.md',
              'ped_vol8hr', 'countdown', model, df_high,
              'Pedestrian, above median volume', 'day + month + year + main + side route', 'ns_ind, ew_ind', 'robust', 'OLS')
except Exception as e:
    print(f"  High vol ped ERROR: {e}")

try:
    df_low = ped_clean[ped_clean['ped_vol8hr'] < median_vol].copy()
    model = pf.feols("ped_vol8hr ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ns_ind + ew_ind + C(main_cat) + C(side1route_cat)",
                    data=df_low, vcov='hetero')
    add_result('robust/sample/low_volume_ped', 'robustness/sample_restrictions.md',
              'ped_vol8hr', 'countdown', model, df_low,
              'Pedestrian, below median volume', 'day + month + year + main + side route', 'ns_ind, ew_ind', 'robust', 'OLS')
except Exception as e:
    print(f"  Low vol ped ERROR: {e}")

# High vs low volume (auto)
try:
    median_vol = auto_clean['tot_count'].median()
    df_high = auto_clean[auto_clean['tot_count'] >= median_vol].copy()
    model = pf.feols("tot_count ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ew_ind + ns_ind + C(street1_cat) + C(street2_cat) + nvar1",
                    data=df_high, vcov={'CRV1': 'id_pcs_str'})
    add_result('robust/sample/high_volume_auto', 'robustness/sample_restrictions.md',
              'tot_count', 'countdown', model, df_high,
              'Auto, above median volume', 'day + month + year + street1 + street2', 'ew_ind, ns_ind, nvar1', 'id_pcs', 'OLS')
except Exception as e:
    print(f"  High vol auto ERROR: {e}")

try:
    df_low = auto_clean[auto_clean['tot_count'] < median_vol].copy()
    model = pf.feols("tot_count ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ew_ind + ns_ind + C(street1_cat) + C(street2_cat) + nvar1",
                    data=df_low, vcov={'CRV1': 'id_pcs_str'})
    add_result('robust/sample/low_volume_auto', 'robustness/sample_restrictions.md',
              'tot_count', 'countdown', model, df_low,
              'Auto, below median volume', 'day + month + year + street1 + street2', 'ew_ind, ns_ind, nvar1', 'id_pcs', 'OLS')
except Exception as e:
    print(f"  Low vol auto ERROR: {e}")

# By day of week (weekday vs weekend)
try:
    df_weekday = ped_clean[ped_clean['day'] < 5].copy()
    model = pf.feols("ped_vol8hr ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ns_ind + ew_ind + C(main_cat) + C(side1route_cat)",
                    data=df_weekday, vcov='hetero')
    add_result('robust/sample/weekday_ped', 'robustness/sample_restrictions.md',
              'ped_vol8hr', 'countdown', model, df_weekday,
              'Pedestrian, weekdays only', 'day + month + year + main + side route', 'ns_ind, ew_ind', 'robust', 'OLS')
except Exception as e:
    print(f"  Weekday ped ERROR: {e}")

try:
    df_weekend = ped_clean[ped_clean['day'] >= 5].copy()
    if len(df_weekend) > 50:
        model = pf.feols("ped_vol8hr ~ countdown + C(mo_cat) + C(yr_cat) + ns_ind + ew_ind + C(main_cat) + C(side1route_cat)",
                        data=df_weekend, vcov='hetero')
        add_result('robust/sample/weekend_ped', 'robustness/sample_restrictions.md',
                  'ped_vol8hr', 'countdown', model, df_weekend,
                  'Pedestrian, weekends only', 'month + year + main + side route', 'ns_ind, ew_ind', 'robust', 'OLS')
except Exception as e:
    print(f"  Weekend ped ERROR: {e}")

try:
    df_weekday = auto_clean[auto_clean['day'] < 5].copy()
    model = pf.feols("tot_count ~ countdown + C(day_cat) + C(mo_cat) + C(yr_cat) + ew_ind + ns_ind + C(street1_cat) + C(street2_cat) + nvar1",
                    data=df_weekday, vcov={'CRV1': 'id_pcs_str'})
    add_result('robust/sample/weekday_auto', 'robustness/sample_restrictions.md',
              'tot_count', 'countdown', model, df_weekday,
              'Auto, weekdays only', 'day + month + year + street1 + street2', 'ew_ind, ns_ind, nvar1', 'id_pcs', 'OLS')
except Exception as e:
    print(f"  Weekday auto ERROR: {e}")

try:
    df_weekend = auto_clean[auto_clean['day'] >= 5].copy()
    if len(df_weekend) > 50:
        model = pf.feols("tot_count ~ countdown + C(mo_cat) + C(yr_cat) + ew_ind + ns_ind + C(street1_cat) + C(street2_cat) + nvar1",
                        data=df_weekend, vcov={'CRV1': 'id_pcs_str'})
        add_result('robust/sample/weekend_auto', 'robustness/sample_restrictions.md',
                  'tot_count', 'countdown', model, df_weekend,
                  'Auto, weekends only', 'month + year + street1 + street2', 'ew_ind, ns_ind, nvar1', 'id_pcs', 'OLS')
except Exception as e:
    print(f"  Weekend auto ERROR: {e}")

# Seasonal analysis
try:
    df_summer = ped_clean[ped_clean['mo'].isin([6, 7, 8])].copy()
    if len(df_summer) > 50:
        model = pf.feols("ped_vol8hr ~ countdown + C(day_cat) + C(yr_cat) + ns_ind + ew_ind + C(main_cat) + C(side1route_cat)",
                        data=df_summer, vcov='hetero')
        add_result('robust/sample/summer_ped', 'robustness/sample_restrictions.md',
                  'ped_vol8hr', 'countdown', model, df_summer,
                  'Pedestrian, summer months', 'day + year + main + side route', 'ns_ind, ew_ind', 'robust', 'OLS')
except Exception as e:
    print(f"  Summer ped ERROR: {e}")

try:
    df_winter = ped_clean[ped_clean['mo'].isin([12, 1, 2])].copy()
    if len(df_winter) > 50:
        model = pf.feols("ped_vol8hr ~ countdown + C(day_cat) + C(yr_cat) + ns_ind + ew_ind + C(main_cat) + C(side1route_cat)",
                        data=df_winter, vcov='hetero')
        add_result('robust/sample/winter_ped', 'robustness/sample_restrictions.md',
                  'ped_vol8hr', 'countdown', model, df_winter,
                  'Pedestrian, winter months', 'day + year + main + side route', 'ns_ind, ew_ind', 'robust', 'OLS')
except Exception as e:
    print(f"  Winter ped ERROR: {e}")

try:
    df_summer = auto_clean[auto_clean['mo'].isin([6, 7, 8])].copy()
    if len(df_summer) > 100:
        model = pf.feols("tot_count ~ countdown + C(day_cat) + C(yr_cat) + ew_ind + ns_ind + C(street1_cat) + C(street2_cat) + nvar1",
                        data=df_summer, vcov={'CRV1': 'id_pcs_str'})
        add_result('robust/sample/summer_auto', 'robustness/sample_restrictions.md',
                  'tot_count', 'countdown', model, df_summer,
                  'Auto, summer months', 'day + year + street1 + street2', 'ew_ind, ns_ind, nvar1', 'id_pcs', 'OLS')
except Exception as e:
    print(f"  Summer auto ERROR: {e}")

try:
    df_winter = auto_clean[auto_clean['mo'].isin([12, 1, 2])].copy()
    if len(df_winter) > 100:
        model = pf.feols("tot_count ~ countdown + C(day_cat) + C(yr_cat) + ew_ind + ns_ind + C(street1_cat) + C(street2_cat) + nvar1",
                        data=df_winter, vcov={'CRV1': 'id_pcs_str'})
        add_result('robust/sample/winter_auto', 'robustness/sample_restrictions.md',
                  'tot_count', 'countdown', model, df_winter,
                  'Auto, winter months', 'day + year + street1 + street2', 'ew_ind, ns_ind, nvar1', 'id_pcs', 'OLS')
except Exception as e:
    print(f"  Winter auto ERROR: {e}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

# Convert to DataFrame
results_df = pd.DataFrame(results)
print(f"\nTotal specifications: {len(results_df)}")

# Save to CSV
output_path = f"{DATA_DIR}/../specification_results.csv"
results_df.to_csv(output_path, index=False)
print(f"Saved results to: {output_path}")

# Summary statistics
print("\n--- Summary Statistics ---")
print(f"Total specifications: {len(results_df)}")
print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")
print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")
print(f"Range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")

# Breakdown by category
print("\n--- Breakdown by Category ---")
results_df['category'] = results_df['spec_id'].apply(lambda x: x.split('/')[0] if '/' in x else 'baseline')
for cat in results_df['category'].unique():
    cat_df = results_df[results_df['category'] == cat]
    pct_pos = 100 * (cat_df['coefficient'] > 0).mean()
    pct_sig = 100 * (cat_df['p_value'] < 0.05).mean()
    print(f"  {cat}: N={len(cat_df)}, {pct_pos:.0f}% positive, {pct_sig:.0f}% sig at 5%")

print("\nDone!")
