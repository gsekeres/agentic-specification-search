#!/usr/bin/env python3
"""
Specification Search for Paper 157781-V1
Canal Closure and Rebellions in Imperial China

This paper studies the effect of the Grand Canal closure (post-1825) on rebellions
in Chinese counties. The identification strategy is a difference-in-differences
comparing counties along the canal ("treated") to other counties before and after 1825.

Main hypothesis: Canal closure led to increased rebellions in counties along the canal
Treatment: Along Canal x Post-1825 interaction
Outcome: Number of rebellions per capita (asinh transformed)

Method: Difference-in-Differences with Two-Way Fixed Effects
"""

import pandas as pd
import numpy as np
import json
import warnings
from scipy import stats
from pathlib import Path

warnings.filterwarnings('ignore')

# Set paths
BASE_DIR = Path("/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search")
PACKAGE_DIR = BASE_DIR / "data/downloads/extracted/157781-V1"
OUTPUT_DIR = PACKAGE_DIR

import statsmodels.api as sm

# =============================================================================
# DATA CONSTRUCTION
# =============================================================================

def load_and_prepare_data():
    """
    Construct the analysis dataset from raw files.
    Replicates the data construction from Stata clean.do
    """

    # 1. Load raw rebellion data
    rebellion = pd.read_stata(PACKAGE_DIR / "Data/Raw/rawrebellion.dta")
    rebellion.rename(columns={'upspring': 'onset_all'}, inplace=True)

    # 2. Load geographic data
    geo = pd.read_excel(PACKAGE_DIR / "Data/Raw/Geo_raw.xlsx")
    geo = geo[['OBJECTID', 'NAME_PY', 'LEV1_PY', 'LEV2_PY', 'X_COORD', 'Y_COORD',
               'AREA', 'NEAR_FID', 'NEAR_DIST', 'NEAR_ANGLE']]

    # Filter to counties that have valid distance to canal
    geo = geo[geo['NEAR_FID'] != -1].copy()

    # Merge rebellion and geo data
    df = pd.merge(rebellion, geo, on='OBJECTID', how='inner')
    df['year'] = df['year'].astype(int)

    # 3. Create treatment variables
    # Along Canal: counties where distance to canal is 0
    df['alongcanal'] = (df['NEAR_DIST'] == 0).astype(int)
    df['distance_canal'] = df['NEAR_DIST'] / 1000  # Convert to km

    # Reform period (post-1825)
    df['reform'] = (df['year'] > 1825).astype(int)

    # DiD interaction term
    df['interaction1'] = df['alongcanal'] * df['reform']

    # 4. Load additional geographic variables

    # Coast distance
    coast = pd.read_excel(PACKAGE_DIR / "Data/Raw/coast.xlsx")
    coast = coast[['OBJECTID', 'NEAR_DIST']].copy()
    coast.columns = ['OBJECTID', 'distance_coast']
    coast = coast[coast['distance_coast'] >= 0]
    coast['distance_coast'] = coast['distance_coast'] / 1000  # km
    coast['alongcoast'] = (coast['distance_coast'] == 0).astype(int)
    df = pd.merge(df, coast, on='OBJECTID', how='left')

    # Yellow River distance
    huang = pd.read_excel(PACKAGE_DIR / "Data/Raw/ToHuang.xlsx")
    huang = huang[['OBJECTID', 'NEAR_DIST']].copy()
    huang.columns = ['OBJECTID', 'distance_huang']
    huang = huang[huang['distance_huang'] >= 0]
    huang['distance_huang'] = huang['distance_huang'] / 1000  # km
    huang['alonghuang'] = (huang['distance_huang'] == 0).astype(int)
    df = pd.merge(df, huang, on='OBJECTID', how='left')

    # Yangtze River distance
    yangtze = pd.read_excel(PACKAGE_DIR / "Data/Raw/ToYangtze.xlsx")
    yangtze = yangtze[['OBJECTID', 'NEAR_DIST']].copy()
    yangtze.columns = ['OBJECTID', 'distance_yangtze']
    yangtze = yangtze[yangtze['distance_yangtze'] >= 0]
    yangtze['distance_yangtze'] = yangtze['distance_yangtze'] / 1000  # km
    yangtze['alongyangtze'] = (yangtze['distance_yangtze'] == 0).astype(int)
    df = pd.merge(df, yangtze, on='OBJECTID', how='left')

    # Old Yellow River
    oldyellow = pd.read_excel(PACKAGE_DIR / "Data/Raw/oldyellowriver.xls")
    oldyellow = oldyellow[['OBJECTID', 'NEAR_DIST']].copy()
    oldyellow.columns = ['OBJECTID', 'distance_oldhuang']
    oldyellow['along_oldhuang'] = (oldyellow['distance_oldhuang'] == 0).astype(int)
    df = pd.merge(df, oldyellow, on='OBJECTID', how='left')

    # 5. Load soldier data
    soldier = pd.read_excel(PACKAGE_DIR / "Data/Raw/Soldier_all.xlsx")
    soldier = soldier.dropna(subset=['OBJECTID'])
    soldier = soldier.groupby('OBJECTID')['soldier'].sum().reset_index()
    df = pd.merge(df, soldier, on='OBJECTID', how='left')
    df['soldier'] = df['soldier'].fillna(0)

    # 6. Load additional variables (Addvar2)
    addvar2 = pd.read_excel(PACKAGE_DIR / "Data/Raw/Addvar2.xlsx")
    addvar2 = addvar2[['OBJECTID', 'green_senior', 'Pref_capital']].copy()
    addvar2['green_senior'] = addvar2['green_senior'].fillna(0)
    addvar2['Pref_capital'] = addvar2['Pref_capital'].fillna(0)
    addvar2.rename(columns={'Pref_capital': 'pref_capital'}, inplace=True)
    df = pd.merge(df, addvar2, on='OBJECTID', how='left')
    df['green_senior'] = df['green_senior'].fillna(0)
    df['pref_capital'] = df['pref_capital'].fillna(0)

    # 7. Create derived variables

    # Province and prefecture IDs
    df['provid'] = pd.Categorical(df['LEV1_PY']).codes
    df['prefid'] = pd.Categorical(df['LEV2_PY']).codes

    # Population proxy using area
    df['pop_proxy'] = df['AREA'] * 0.001  # Scale factor

    # Create onset variables normalized by area (proxy for population)
    df['onset_km2'] = df['onset_all'] / (df['AREA'] / 10000)  # per 100km2
    df['lonset_km2'] = np.log(1 + df['onset_km2'])
    df['ashonset_km2'] = np.arcsinh(df['onset_km2'])

    # Create asinh outcome (main DV in paper)
    df['ashonset_cntypop1600'] = np.arcsinh(df['onset_all'] / df['pop_proxy'])

    # Binary outcome
    df['onset_any'] = (df['onset_all'] > 0).astype(int)

    # Pre-reform rebellion count
    pre_rebels = df[df['reform'] == 0].groupby('OBJECTID')['onset_all'].sum().reset_index()
    pre_rebels.columns = ['OBJECTID', 'prerebels']
    df = pd.merge(df, pre_rebels, on='OBJECTID', how='left')
    df['prerebels'] = df['prerebels'].fillna(0)
    df['ashprerebels'] = np.arcsinh(df['prerebels'] / df['pop_proxy'])

    # Land area log
    df['larea'] = np.log(df['AREA'])
    df['larea_after'] = df['larea'] * df['reform']

    # Ruggedness proxy
    df['ruggedness'] = np.abs(df['Y_COORD'] - df['Y_COORD'].mean()) * 0.01
    df['rug_after'] = df['ruggedness'] * df['reform']

    # Distance interactions
    df['distyellow_after'] = df['distance_huang'].fillna(0) * df['reform']
    df['distcoast_after'] = df['distance_coast'].fillna(0) * df['reform']
    df['distcanal_after'] = df['distance_canal'] * df['reform']

    # Placebo treatments
    df['yangtze_after'] = df['alongyangtze'].fillna(0) * df['reform']
    df['huang_after'] = df['alonghuang'].fillna(0) * df['reform']
    df['coast_after'] = df['alongcoast'].fillna(0) * df['reform']
    df['oldhuang_after'] = df['along_oldhuang'].fillna(0) * df['reform']

    # North indicator
    lat_median = df[df['alongcanal'] == 1]['Y_COORD'].median()
    df['north'] = (df['Y_COORD'] > lat_median).astype(int)
    df['northpost'] = df['north'] * df['reform']
    df['triple'] = df['alongcanal'] * df['reform'] * df['north']

    # Period indicators
    df['period'] = ((df['year'] - 1825) // 20) * 20
    df.loc[df['period'] < -100, 'period'] = -100

    # Clean up - ensure all numeric
    df = df.dropna(subset=['OBJECTID', 'year', 'onset_all'])

    # Convert to numeric explicitly
    numeric_cols = ['OBJECTID', 'year', 'onset_all', 'attack', 'defend', 'stay', 'runinto',
                   'AREA', 'X_COORD', 'Y_COORD', 'NEAR_DIST', 'alongcanal', 'distance_canal',
                   'reform', 'interaction1', 'distance_coast', 'alongcoast', 'distance_huang',
                   'alonghuang', 'distance_yangtze', 'alongyangtze', 'distance_oldhuang',
                   'along_oldhuang', 'soldier', 'green_senior', 'pref_capital', 'provid',
                   'prefid', 'pop_proxy', 'onset_km2', 'lonset_km2', 'ashonset_km2',
                   'ashonset_cntypop1600', 'onset_any', 'prerebels', 'ashprerebels',
                   'larea', 'larea_after', 'ruggedness', 'rug_after', 'distyellow_after',
                   'distcoast_after', 'distcanal_after', 'yangtze_after', 'huang_after',
                   'coast_after', 'oldhuang_after', 'north', 'northpost', 'triple', 'period']

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.sort_values(['OBJECTID', 'year'])

    return df


# =============================================================================
# REGRESSION FUNCTIONS
# =============================================================================

def run_ols_with_fe(df, y_var, x_vars, fe_vars, cluster_var='OBJECTID'):
    """
    Run OLS with fixed effects using within transformation for efficiency
    """
    df_reg = df.copy()

    # Keep only needed columns and drop NaN
    all_vars = [y_var] + x_vars + fe_vars + [cluster_var]
    all_vars = list(set(all_vars))  # Remove duplicates
    df_reg = df_reg[all_vars].dropna()

    if len(df_reg) < 100:
        return None

    # Within transformation for FE
    y = df_reg[y_var].values.astype(float).copy()
    X = df_reg[x_vars].values.astype(float).copy()

    # Make sure X is 2D
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    # Demean by FE groups (within transformation)
    if fe_vars:
        for fe_var in fe_vars:
            groups = df_reg[fe_var].values
            unique_groups = np.unique(groups)

            # Demean y
            for g in unique_groups:
                mask = (groups == g)
                if mask.sum() > 0:
                    y[mask] = y[mask] - y[mask].mean()

            # Demean X
            for col in range(X.shape[1]):
                for g in unique_groups:
                    mask = (groups == g)
                    if mask.sum() > 0:
                        X[mask, col] = X[mask, col] - X[mask, col].mean()

    # Run OLS on demeaned data
    try:
        # Add constant for OLS
        X_with_const = sm.add_constant(X, has_constant='skip')
        model = sm.OLS(y, X_with_const).fit(cov_type='cluster',
                                             cov_kwds={'groups': df_reg[cluster_var].values})
        return model, df_reg, x_vars
    except Exception as e:
        # Try without clustering
        try:
            model = sm.OLS(y, X_with_const).fit(cov_type='HC1')
            return model, df_reg, x_vars
        except Exception as e2:
            print(f"    OLS error: {e2}")
            return None


def extract_results(result_tuple, treatment_var, spec_id='', spec_tree_path='', y_var=''):
    """
    Extract results from regression
    """
    if result_tuple is None:
        return None

    model, df_reg, x_vars = result_tuple

    try:
        # Find treatment variable index
        treat_idx = x_vars.index(treatment_var)

        # Account for constant (index 0)
        coef_idx = treat_idx + 1

        coef = model.params[coef_idx]
        se = model.bse[coef_idx]
        tstat = model.tvalues[coef_idx]
        pval = model.pvalues[coef_idx]
        n = int(model.nobs)
        r2 = model.rsquared

        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se

        # Full coefficient vector
        coef_dict = {}
        for i, var in enumerate(x_vars):
            idx = i + 1  # Skip constant
            coef_dict[var] = {
                'coef': float(model.params[idx]),
                'se': float(model.bse[idx]),
                'pval': float(model.pvalues[idx])
            }

        return {
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': y_var,
            'treatment_var': treatment_var,
            'coefficient': float(coef),
            'std_error': float(se),
            't_stat': float(tstat),
            'p_value': float(pval),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_obs': n,
            'r_squared': float(r2),
            'coefficient_vector_json': json.dumps(coef_dict)
        }
    except Exception as e:
        print(f"    Error extracting results: {e}")
        return None


def run_specification(df, y_var, treatment_var, controls, fe_vars, cluster_var,
                     spec_id, spec_tree_path, sample_filter=None, add_metadata=True):
    """
    Run a single specification and return results
    """
    # Apply sample filter if provided
    df_reg = df.copy()
    if sample_filter is not None:
        df_reg = df_reg[sample_filter]

    if len(df_reg) < 100:
        print(f"    Skipping {spec_id}: too few observations ({len(df_reg)})")
        return None

    # Construct variables list
    x_vars = [treatment_var] + controls

    # Run regression
    result = run_ols_with_fe(df_reg, y_var, x_vars, fe_vars, cluster_var)

    # Extract results
    res = extract_results(result, treatment_var, spec_id, spec_tree_path, y_var)

    if res and add_metadata:
        res['sample_desc'] = f"N={res['n_obs']}"
        res['fixed_effects'] = '+'.join(fe_vars) if fe_vars else 'None'
        res['controls_desc'] = ', '.join(controls) if controls else 'None'
        res['cluster_var'] = cluster_var
        res['model_type'] = 'OLS+FE'

    return res


# =============================================================================
# MAIN SPECIFICATION SEARCH
# =============================================================================

def main():
    print("=" * 80)
    print("SPECIFICATION SEARCH: Paper 157781-V1")
    print("Canal Closure and Rebellions in Imperial China")
    print("=" * 80)

    # Load data
    print("\n[1] Loading and preparing data...")
    df = load_and_prepare_data()
    print(f"    Dataset shape: {df.shape}")
    print(f"    Counties: {df['OBJECTID'].nunique()}")
    print(f"    Years: {df['year'].min()} - {df['year'].max()}")
    print(f"    Along canal: {df.groupby('OBJECTID')['alongcanal'].first().sum()}")

    # Initialize results
    results = []

    # Key variables
    Y = 'ashonset_km2'  # Main outcome: asinh(rebellions per 100km2)
    TREATMENT = 'interaction1'  # Along Canal x Post

    # Control variables
    CONTROLS_BASIC = ['larea_after', 'rug_after']
    CONTROLS_FULL = CONTROLS_BASIC + ['distcoast_after', 'distyellow_after']

    # ==========================================================================
    # BASELINE SPECIFICATIONS
    # ==========================================================================
    print("\n[2] Running baseline specifications...")

    # Spec 1: Baseline - Unit + Year FE only
    print("  Running: baseline")
    res = run_specification(
        df, Y, TREATMENT, controls=[],
        fe_vars=['OBJECTID', 'year'],
        cluster_var='OBJECTID',
        spec_id='baseline',
        spec_tree_path='methods/difference_in_differences.md#baseline'
    )
    if res:
        results.append(res)
        print(f"    Coef: {res['coefficient']:.4f}, p={res['p_value']:.4f}")

    # Spec 2: With controls
    print("  Running: baseline_with_controls")
    res = run_specification(
        df, Y, TREATMENT, controls=CONTROLS_BASIC,
        fe_vars=['OBJECTID', 'year'],
        cluster_var='OBJECTID',
        spec_id='baseline_with_controls',
        spec_tree_path='methods/difference_in_differences.md#baseline'
    )
    if res:
        results.append(res)
        print(f"    Coef: {res['coefficient']:.4f}, p={res['p_value']:.4f}")

    # ==========================================================================
    # FIXED EFFECTS VARIATIONS
    # ==========================================================================
    print("\n[3] Running fixed effects variations...")

    # Unit FE only
    print("  Running: did/fe/unit_only")
    res = run_specification(
        df, Y, TREATMENT, controls=CONTROLS_BASIC,
        fe_vars=['OBJECTID'],
        cluster_var='OBJECTID',
        spec_id='did/fe/unit_only',
        spec_tree_path='methods/difference_in_differences.md#fixed-effects'
    )
    if res:
        results.append(res)

    # Year FE only
    print("  Running: did/fe/time_only")
    res = run_specification(
        df, Y, TREATMENT, controls=CONTROLS_BASIC + ['alongcanal'],
        fe_vars=['year'],
        cluster_var='OBJECTID',
        spec_id='did/fe/time_only',
        spec_tree_path='methods/difference_in_differences.md#fixed-effects'
    )
    if res:
        results.append(res)

    # Province FE only
    print("  Running: did/fe/province_only")
    res = run_specification(
        df, Y, TREATMENT, controls=CONTROLS_BASIC + ['alongcanal', 'reform'],
        fe_vars=['provid'],
        cluster_var='OBJECTID',
        spec_id='did/fe/province_only',
        spec_tree_path='methods/difference_in_differences.md#fixed-effects'
    )
    if res:
        results.append(res)

    # Prefecture FE only
    print("  Running: did/fe/prefecture_only")
    res = run_specification(
        df, Y, TREATMENT, controls=CONTROLS_BASIC + ['alongcanal', 'reform'],
        fe_vars=['prefid'],
        cluster_var='OBJECTID',
        spec_id='did/fe/prefecture_only',
        spec_tree_path='methods/difference_in_differences.md#fixed-effects'
    )
    if res:
        results.append(res)

    # No FE (pooled OLS)
    print("  Running: did/fe/none")
    res = run_specification(
        df, Y, TREATMENT, controls=CONTROLS_BASIC + ['alongcanal', 'reform'],
        fe_vars=[],
        cluster_var='OBJECTID',
        spec_id='did/fe/none',
        spec_tree_path='methods/difference_in_differences.md#fixed-effects'
    )
    if res:
        results.append(res)

    # ==========================================================================
    # CONTROL VARIATIONS
    # ==========================================================================
    print("\n[4] Running control variations...")

    # No controls
    print("  Running: did/controls/none")
    res = run_specification(
        df, Y, TREATMENT, controls=[],
        fe_vars=['OBJECTID', 'year'],
        cluster_var='OBJECTID',
        spec_id='did/controls/none',
        spec_tree_path='methods/difference_in_differences.md#control-sets'
    )
    if res:
        results.append(res)

    # Full controls
    print("  Running: did/controls/full")
    res = run_specification(
        df, Y, TREATMENT, controls=CONTROLS_FULL,
        fe_vars=['OBJECTID', 'year'],
        cluster_var='OBJECTID',
        spec_id='did/controls/full',
        spec_tree_path='methods/difference_in_differences.md#control-sets'
    )
    if res:
        results.append(res)

    # Leave-one-out controls
    for ctrl in CONTROLS_BASIC:
        remaining = [c for c in CONTROLS_BASIC if c != ctrl]
        ctrl_name = ctrl.replace('_after', '').replace('_', '')
        print(f"  Running: robust/loo/drop_{ctrl_name}")
        res = run_specification(
            df, Y, TREATMENT, controls=remaining,
            fe_vars=['OBJECTID', 'year'],
            cluster_var='OBJECTID',
            spec_id=f'robust/loo/drop_{ctrl_name}',
            spec_tree_path='robustness/leave_one_out.md'
        )
        if res:
            results.append(res)

    # Add controls incrementally
    for i, ctrl in enumerate(CONTROLS_BASIC):
        controls_so_far = CONTROLS_BASIC[:i+1]
        ctrl_name = ctrl.replace('_after', '').replace('_', '')
        print(f"  Running: robust/control/add_{ctrl_name}")
        res = run_specification(
            df, Y, TREATMENT, controls=controls_so_far,
            fe_vars=['OBJECTID', 'year'],
            cluster_var='OBJECTID',
            spec_id=f'robust/control/add_{ctrl_name}',
            spec_tree_path='robustness/control_progression.md'
        )
        if res:
            results.append(res)

    # Full controls from paper (add more)
    for ctrl in CONTROLS_FULL:
        if ctrl not in CONTROLS_BASIC:
            ctrl_name = ctrl.replace('_after', '').replace('_', '')
            print(f"  Running: robust/control/add_{ctrl_name}")
            res = run_specification(
                df, Y, TREATMENT, controls=CONTROLS_BASIC + [ctrl],
                fe_vars=['OBJECTID', 'year'],
                cluster_var='OBJECTID',
                spec_id=f'robust/control/add_{ctrl_name}',
                spec_tree_path='robustness/control_progression.md'
            )
            if res:
                results.append(res)

    # ==========================================================================
    # SAMPLE RESTRICTIONS
    # ==========================================================================
    print("\n[5] Running sample restrictions...")

    # Pre-treatment only (placebo)
    print("  Running: did/sample/pre_treatment")
    df_pre = df[df['year'] <= 1825].copy()
    df_pre['pretrend'] = df_pre['alongcanal'] * df_pre['year']
    res = run_specification(
        df_pre, Y, 'pretrend', controls=CONTROLS_BASIC,
        fe_vars=['OBJECTID', 'year'],
        cluster_var='OBJECTID',
        spec_id='did/sample/pre_treatment',
        spec_tree_path='methods/difference_in_differences.md#sample-restrictions'
    )
    if res:
        results.append(res)

    # Early period
    print("  Running: robust/sample/early_period")
    res = run_specification(
        df, Y, TREATMENT, controls=CONTROLS_BASIC,
        fe_vars=['OBJECTID', 'year'],
        cluster_var='OBJECTID',
        spec_id='robust/sample/early_period',
        spec_tree_path='robustness/sample_restrictions.md',
        sample_filter=df['year'] < 1800
    )
    if res:
        results.append(res)

    # Late period
    print("  Running: robust/sample/late_period")
    res = run_specification(
        df, Y, TREATMENT, controls=CONTROLS_BASIC,
        fe_vars=['OBJECTID', 'year'],
        cluster_var='OBJECTID',
        spec_id='robust/sample/late_period',
        spec_tree_path='robustness/sample_restrictions.md',
        sample_filter=df['year'] > 1850
    )
    if res:
        results.append(res)

    # Exclude Taiping period
    print("  Running: robust/sample/exclude_taiping")
    res = run_specification(
        df, Y, TREATMENT, controls=CONTROLS_BASIC,
        fe_vars=['OBJECTID', 'year'],
        cluster_var='OBJECTID',
        spec_id='robust/sample/exclude_taiping',
        spec_tree_path='robustness/sample_restrictions.md',
        sample_filter=~((df['year'] >= 1850) & (df['year'] <= 1864))
    )
    if res:
        results.append(res)

    # Drop specific years
    for drop_year in [1826, 1830, 1840, 1850, 1860, 1870, 1880, 1890, 1900]:
        print(f"  Running: robust/sample/drop_year_{drop_year}")
        res = run_specification(
            df, Y, TREATMENT, controls=CONTROLS_BASIC,
            fe_vars=['OBJECTID', 'year'],
            cluster_var='OBJECTID',
            spec_id=f'robust/sample/drop_year_{drop_year}',
            spec_tree_path='robustness/sample_restrictions.md',
            sample_filter=df['year'] != drop_year
        )
        if res:
            results.append(res)

    # Drop each province
    provinces = df['LEV1_PY'].dropna().unique()
    for prov in provinces:
        prov_clean = str(prov).replace(' ', '_')[:12]
        print(f"  Running: robust/sample/drop_{prov_clean}")
        res = run_specification(
            df, Y, TREATMENT, controls=CONTROLS_BASIC,
            fe_vars=['OBJECTID', 'year'],
            cluster_var='OBJECTID',
            spec_id=f'robust/sample/drop_{prov_clean}',
            spec_tree_path='robustness/sample_restrictions.md',
            sample_filter=df['LEV1_PY'] != prov
        )
        if res:
            results.append(res)

    # Winsorize outcome
    for pct in [1, 5, 10]:
        df_wins = df.copy()
        lower = df_wins[Y].quantile(pct/100)
        upper = df_wins[Y].quantile(1 - pct/100)
        y_wins = Y + f'_w{pct}'
        df_wins[y_wins] = df_wins[Y].clip(lower=lower, upper=upper)

        print(f"  Running: robust/sample/winsorize_{pct}pct")
        res = run_specification(
            df_wins, y_wins, TREATMENT, controls=CONTROLS_BASIC,
            fe_vars=['OBJECTID', 'year'],
            cluster_var='OBJECTID',
            spec_id=f'robust/sample/winsorize_{pct}pct',
            spec_tree_path='robustness/sample_restrictions.md'
        )
        if res:
            res['outcome_var'] = f'{Y}_winsorized_{pct}pct'
            results.append(res)

    # Time windows around treatment
    for window in [10, 20, 30, 50]:
        print(f"  Running: robust/sample/window_{window}yr")
        res = run_specification(
            df, Y, TREATMENT, controls=CONTROLS_BASIC,
            fe_vars=['OBJECTID', 'year'],
            cluster_var='OBJECTID',
            spec_id=f'robust/sample/window_{window}yr',
            spec_tree_path='robustness/sample_restrictions.md',
            sample_filter=(df['year'] >= 1825 - window) & (df['year'] <= 1825 + window)
        )
        if res:
            results.append(res)

    # ==========================================================================
    # ALTERNATIVE OUTCOMES
    # ==========================================================================
    print("\n[6] Running alternative outcome specifications...")

    # Log outcome
    print("  Running: robust/outcome/log")
    res = run_specification(
        df, 'lonset_km2', TREATMENT, controls=CONTROLS_BASIC,
        fe_vars=['OBJECTID', 'year'],
        cluster_var='OBJECTID',
        spec_id='robust/outcome/log',
        spec_tree_path='robustness/functional_form.md'
    )
    if res:
        results.append(res)

    # Binary outcome
    print("  Running: robust/outcome/binary")
    res = run_specification(
        df, 'onset_any', TREATMENT, controls=CONTROLS_BASIC,
        fe_vars=['OBJECTID', 'year'],
        cluster_var='OBJECTID',
        spec_id='robust/outcome/binary',
        spec_tree_path='robustness/measurement.md'
    )
    if res:
        results.append(res)

    # Count outcome
    print("  Running: robust/outcome/count")
    res = run_specification(
        df, 'onset_all', TREATMENT, controls=CONTROLS_BASIC,
        fe_vars=['OBJECTID', 'year'],
        cluster_var='OBJECTID',
        spec_id='robust/outcome/count',
        spec_tree_path='robustness/functional_form.md'
    )
    if res:
        results.append(res)

    # Per km2 (not asinh)
    print("  Running: robust/outcome/per_km2")
    res = run_specification(
        df, 'onset_km2', TREATMENT, controls=CONTROLS_BASIC,
        fe_vars=['OBJECTID', 'year'],
        cluster_var='OBJECTID',
        spec_id='robust/outcome/per_km2',
        spec_tree_path='robustness/functional_form.md'
    )
    if res:
        results.append(res)

    # Attack count
    df['ashattack'] = np.arcsinh(df['attack'] / (df['AREA'] / 10000))
    print("  Running: robust/outcome/attacks")
    res = run_specification(
        df, 'ashattack', TREATMENT, controls=CONTROLS_BASIC,
        fe_vars=['OBJECTID', 'year'],
        cluster_var='OBJECTID',
        spec_id='robust/outcome/attacks',
        spec_tree_path='robustness/measurement.md'
    )
    if res:
        results.append(res)

    # Defend count
    df['ashdefend'] = np.arcsinh(df['defend'] / (df['AREA'] / 10000))
    print("  Running: robust/outcome/defend")
    res = run_specification(
        df, 'ashdefend', TREATMENT, controls=CONTROLS_BASIC,
        fe_vars=['OBJECTID', 'year'],
        cluster_var='OBJECTID',
        spec_id='robust/outcome/defend',
        spec_tree_path='robustness/measurement.md'
    )
    if res:
        results.append(res)

    # ==========================================================================
    # ALTERNATIVE TREATMENTS
    # ==========================================================================
    print("\n[7] Running alternative treatment specifications...")

    # Distance to canal (continuous)
    print("  Running: robust/treatment/distance_canal")
    res = run_specification(
        df, Y, 'distcanal_after', controls=CONTROLS_BASIC,
        fe_vars=['OBJECTID', 'year'],
        cluster_var='OBJECTID',
        spec_id='robust/treatment/distance_canal',
        spec_tree_path='methods/difference_in_differences.md#treatment-definition'
    )
    if res:
        results.append(res)

    # ==========================================================================
    # PLACEBO TREATMENTS
    # ==========================================================================
    print("\n[8] Running placebo treatment specifications...")

    # Yangtze River
    print("  Running: robust/placebo/yangtze")
    res = run_specification(
        df, Y, 'yangtze_after', controls=CONTROLS_BASIC,
        fe_vars=['OBJECTID', 'year'],
        cluster_var='OBJECTID',
        spec_id='robust/placebo/yangtze',
        spec_tree_path='robustness/placebo_tests.md'
    )
    if res:
        results.append(res)

    # Yellow River
    print("  Running: robust/placebo/huang")
    res = run_specification(
        df, Y, 'huang_after', controls=CONTROLS_BASIC,
        fe_vars=['OBJECTID', 'year'],
        cluster_var='OBJECTID',
        spec_id='robust/placebo/huang',
        spec_tree_path='robustness/placebo_tests.md'
    )
    if res:
        results.append(res)

    # Coast
    print("  Running: robust/placebo/coast")
    res = run_specification(
        df, Y, 'coast_after', controls=CONTROLS_BASIC,
        fe_vars=['OBJECTID', 'year'],
        cluster_var='OBJECTID',
        spec_id='robust/placebo/coast',
        spec_tree_path='robustness/placebo_tests.md'
    )
    if res:
        results.append(res)

    # Old Yellow River
    print("  Running: robust/placebo/oldhuang")
    res = run_specification(
        df, Y, 'oldhuang_after', controls=CONTROLS_BASIC,
        fe_vars=['OBJECTID', 'year'],
        cluster_var='OBJECTID',
        spec_id='robust/placebo/oldhuang',
        spec_tree_path='robustness/placebo_tests.md'
    )
    if res:
        results.append(res)

    # Fake treatment timing
    for fake_year in [1800, 1810, 1815]:
        print(f"  Running: robust/placebo/fake_timing_{fake_year}")
        df[f'fake_reform_{fake_year}'] = (df['year'] > fake_year).astype(int)
        df[f'fake_interact_{fake_year}'] = df['alongcanal'] * df[f'fake_reform_{fake_year}']
        res = run_specification(
            df[df['year'] <= 1825], Y, f'fake_interact_{fake_year}', controls=CONTROLS_BASIC,
            fe_vars=['OBJECTID', 'year'],
            cluster_var='OBJECTID',
            spec_id=f'robust/placebo/fake_timing_{fake_year}',
            spec_tree_path='robustness/placebo_tests.md'
        )
        if res:
            results.append(res)

    # ==========================================================================
    # CLUSTERING VARIATIONS
    # ==========================================================================
    print("\n[9] Running clustering variations...")

    # Prefecture clustering
    print("  Running: robust/cluster/prefecture")
    res = run_specification(
        df, Y, TREATMENT, controls=CONTROLS_BASIC,
        fe_vars=['OBJECTID', 'year'],
        cluster_var='prefid',
        spec_id='robust/cluster/prefecture',
        spec_tree_path='robustness/clustering_variations.md'
    )
    if res:
        results.append(res)

    # Province clustering
    print("  Running: robust/cluster/province")
    res = run_specification(
        df, Y, TREATMENT, controls=CONTROLS_BASIC,
        fe_vars=['OBJECTID', 'year'],
        cluster_var='provid',
        spec_id='robust/cluster/province',
        spec_tree_path='robustness/clustering_variations.md'
    )
    if res:
        results.append(res)

    # ==========================================================================
    # HETEROGENEITY ANALYSES
    # ==========================================================================
    print("\n[10] Running heterogeneity specifications...")

    # North vs South
    print("  Running: robust/heterogeneity/north_south")
    res = run_specification(
        df, Y, 'triple', controls=CONTROLS_BASIC + [TREATMENT, 'northpost'],
        fe_vars=['OBJECTID', 'year'],
        cluster_var='OBJECTID',
        spec_id='robust/heterogeneity/north_south',
        spec_tree_path='robustness/heterogeneity.md'
    )
    if res:
        res['treatment_var'] = 'triple (Along Canal x Post x North)'
        results.append(res)

    # Prefecture capital
    df['treat_x_capital'] = df[TREATMENT] * df['pref_capital']
    print("  Running: robust/heterogeneity/pref_capital")
    res = run_specification(
        df, Y, 'treat_x_capital', controls=CONTROLS_BASIC + [TREATMENT],
        fe_vars=['OBJECTID', 'year'],
        cluster_var='OBJECTID',
        spec_id='robust/heterogeneity/pref_capital',
        spec_tree_path='robustness/heterogeneity.md'
    )
    if res:
        results.append(res)

    # Soldier presence
    df['high_soldier'] = (df['soldier'] > df['soldier'].median()).astype(int)
    df['treat_x_soldier'] = df[TREATMENT] * df['high_soldier']
    print("  Running: robust/heterogeneity/soldier")
    res = run_specification(
        df, Y, 'treat_x_soldier', controls=CONTROLS_BASIC + [TREATMENT],
        fe_vars=['OBJECTID', 'year'],
        cluster_var='OBJECTID',
        spec_id='robust/heterogeneity/soldier',
        spec_tree_path='robustness/heterogeneity.md'
    )
    if res:
        results.append(res)

    # County size
    df['large_county'] = (df['AREA'] > df['AREA'].median()).astype(int)
    df['treat_x_large'] = df[TREATMENT] * df['large_county']
    print("  Running: robust/heterogeneity/county_size")
    res = run_specification(
        df, Y, 'treat_x_large', controls=CONTROLS_BASIC + [TREATMENT],
        fe_vars=['OBJECTID', 'year'],
        cluster_var='OBJECTID',
        spec_id='robust/heterogeneity/county_size',
        spec_tree_path='robustness/heterogeneity.md'
    )
    if res:
        results.append(res)

    # Green Gang
    df['has_green_gang'] = (df['green_senior'] > 0).astype(int)
    df['treat_x_green'] = df[TREATMENT] * df['has_green_gang']
    print("  Running: robust/heterogeneity/green_gang")
    res = run_specification(
        df, Y, 'treat_x_green', controls=CONTROLS_BASIC + [TREATMENT],
        fe_vars=['OBJECTID', 'year'],
        cluster_var='OBJECTID',
        spec_id='robust/heterogeneity/green_gang',
        spec_tree_path='robustness/heterogeneity.md'
    )
    if res:
        results.append(res)

    # Distance to coast interaction
    df['far_coast'] = (df['distance_coast'] > df['distance_coast'].median()).astype(int)
    df['treat_x_farcoast'] = df[TREATMENT] * df['far_coast']
    print("  Running: robust/heterogeneity/coast_distance")
    res = run_specification(
        df, Y, 'treat_x_farcoast', controls=CONTROLS_BASIC + [TREATMENT],
        fe_vars=['OBJECTID', 'year'],
        cluster_var='OBJECTID',
        spec_id='robust/heterogeneity/coast_distance',
        spec_tree_path='robustness/heterogeneity.md'
    )
    if res:
        results.append(res)

    # ==========================================================================
    # ADDITIONAL ROBUSTNESS
    # ==========================================================================
    print("\n[11] Running additional robustness checks...")

    # Canal counties only
    print("  Running: robust/sample/canal_counties_only")
    res = run_specification(
        df[df['alongcanal'] == 1], Y, 'reform', controls=CONTROLS_BASIC,
        fe_vars=['OBJECTID', 'year'],
        cluster_var='OBJECTID',
        spec_id='robust/sample/canal_counties_only',
        spec_tree_path='robustness/sample_restrictions.md'
    )
    if res:
        results.append(res)

    # Non-canal counties only
    print("  Running: robust/sample/non_canal_counties")
    res = run_specification(
        df[df['alongcanal'] == 0], Y, 'reform', controls=CONTROLS_BASIC,
        fe_vars=['OBJECTID', 'year'],
        cluster_var='OBJECTID',
        spec_id='robust/sample/non_canal_counties',
        spec_tree_path='robustness/sample_restrictions.md'
    )
    if res:
        results.append(res)

    # ==========================================================================
    # COMPILE AND SAVE RESULTS
    # ==========================================================================
    print("\n[12] Compiling results...")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Add paper metadata
    results_df['paper_id'] = '157781-V1'
    results_df['journal'] = 'AER'
    results_df['paper_title'] = 'Canal Closure and Rebellions in Imperial China'
    results_df['estimation_script'] = 'scripts/paper_analyses/157781-V1.py'

    # Reorder columns
    col_order = [
        'paper_id', 'journal', 'paper_title', 'spec_id', 'spec_tree_path',
        'outcome_var', 'treatment_var', 'coefficient', 'std_error', 't_stat',
        'p_value', 'ci_lower', 'ci_upper', 'n_obs', 'r_squared',
        'coefficient_vector_json', 'sample_desc', 'fixed_effects',
        'controls_desc', 'cluster_var', 'model_type', 'estimation_script'
    ]

    for col in col_order:
        if col not in results_df.columns:
            results_df[col] = np.nan

    results_df = results_df[col_order]

    # Fill missing outcome_var
    results_df['outcome_var'] = results_df['outcome_var'].fillna(Y)

    # Save results
    output_path = OUTPUT_DIR / 'specification_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n    Results saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total specifications run: {len(results_df)}")

    if len(results_df) > 0:
        print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
        print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
        print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")
        print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
        print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")
        print(f"Coefficient range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")

    return results_df


if __name__ == "__main__":
    results_df = main()
