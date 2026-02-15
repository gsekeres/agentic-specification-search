"""
Specification Search Script for Hornbeck & Naidu (2014)
"When the Levee Breaks: Black Migration and Economic Development in the American South"
AER 104(3): 963-990

Paper ID: 112749-V1
Surface: SPECIFICATION_SURFACE.json

Executes all core specs from the approved specification surface:
- G1: Flood impact on Black share of population (lnfrac_black)
- G2: Flood impact on agricultural capital intensity (lnvalue_equipment)
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# PATHS
# ============================================================================
PAPER_ID = "112749-V1"
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PKG_DIR = os.path.join(BASE, "data", "downloads", "extracted", PAPER_ID)
REPL_PKG = os.path.join(PKG_DIR, "Replication_AER-2012-0980")

# ============================================================================
# DATA GENERATION - import from replication script
# ============================================================================
sys.path.insert(0, os.path.join(BASE, "scripts/paper_replications"))
from importlib import import_module
repl_mod = import_module("112749-V1")


def build_analysis_data():
    """Build the analysis dataset using the replication script's functions."""
    print("=" * 70)
    print("BUILDING ANALYSIS DATASET")
    print("=" * 70)
    panel = repl_mod.generate_data()
    df_main, df_post, df_south = repl_mod.preanalysis(panel)

    # Create state_year_fe for multi-way FE absorption (instead of d_sy_ dummies)
    for df in [df_main, df_post]:
        df['state_year_fe'] = df['statefips'].astype(str) + '_' + df['year'].astype(str)

    # Generate propensity score (needed for RefTable step 7)
    print("  Generating propensity score...")
    import statsmodels.api as sm
    df1920 = df_main[df_main['year'] == 1920].copy()
    pscore_vars = [c for c in df1920.columns
                   if c.startswith('lc') and ('lnfrac_black' in c or 'cotton_cc' in c)
                   and df1920[c].notna().sum() > 10]
    if pscore_vars and 'flood' in df1920.columns:
        try:
            X = sm.add_constant(df1920[pscore_vars].fillna(0))
            probit = sm.Probit(df1920['flood'], X).fit(disp=0)
            df1920['prop_plant_1920'] = probit.predict(X)
            prop_map = df1920[['fips', 'prop_plant_1920']].rename(
                columns={'prop_plant_1920': 'prop_plant_flstate'})
            for df in [df_main, df_post]:
                if 'prop_plant_flstate' not in df.columns:
                    df_tmp = pd.merge(df, prop_map, on='fips', how='left')
                    df['prop_plant_flstate'] = df_tmp['prop_plant_flstate']
            all_years = [1900,1910,1920,1925,1930,1935,1940,1945,1950,1954,1960,1964,1970]
            for df in [df_main, df_post]:
                for yr in all_years:
                    col = f'prop_plant_flstate_{yr}'
                    df[col] = 0.0
                    mask = (df['year'] == yr) & df['prop_plant_flstate'].between(0, 1, inclusive='neither')
                    df.loc[mask, col] = df.loc[mask, 'prop_plant_flstate']
            print(f"    Propensity score generated for {prop_map['prop_plant_flstate'].notna().sum()} counties")
        except Exception as e:
            print(f"    WARNING: Propensity score failed: {e}")
    else:
        print("    WARNING: Could not generate propensity score")

    print(f"  df_main: {df_main.shape[0]} obs, {df_main['fips'].nunique()} counties")
    print(f"  df_post: {df_post.shape[0]} obs, {df_post['fips'].nunique()} counties")
    return df_main, df_post


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_cols(df, prefixes):
    """Get sorted column names matching any prefix."""
    cols = set()
    for p in prefixes:
        cols.update(c for c in df.columns if c.startswith(p))
    return sorted(cols)


def filter_varied(df, cols, min_std=1e-10):
    """Keep only columns with variation in the given DataFrame."""
    return [c for c in cols if c in df.columns and df[c].std() > min_std]


def run_spec(df, depvar, treatment_vars, controls, fe='fips + state_year_fe',
             cluster='fips', weights='county_w'):
    """Run TWFE spec with absorbed county + state-year FE. Returns (model, error_msg)."""
    # Drop missing DV
    dfc = df.dropna(subset=[depvar]).copy()
    if len(dfc) < 20:
        return None, f"Too few obs ({len(dfc)})"

    # Filter controls to those with variation in the non-missing sample
    ctrl_ok = filter_varied(dfc, controls)
    treat_ok = filter_varied(dfc, treatment_vars)

    if not treat_ok:
        return None, "No treatment vars with variation"

    rhs = treat_ok + ctrl_ok
    rhs_str = " + ".join(rhs)
    formula = f"{depvar} ~ {rhs_str} | {fe}"

    try:
        vcov = {"CRV1": cluster} if cluster else "hetero"
        if weights and weights in dfc.columns:
            m = pf.feols(formula, data=dfc, vcov=vcov, weights=weights)
        else:
            m = pf.feols(formula, data=dfc, vcov=vcov)
        return m, None
    except Exception as e:
        return None, str(e)


def extract_focal(model, focal_var, all_treatment_vars):
    """Extract focal coefficient and full coefficient vector."""
    if model is None:
        return {'coefficient': np.nan, 'std_error': np.nan, 'p_value': np.nan,
                'ci_lower': np.nan, 'ci_upper': np.nan,
                'n_obs': np.nan, 'r_squared': np.nan, 'coefficient_vector_json': '{}'}
    try:
        coefs = model.coef()
        ses = model.se()
        pvals = model.pvalue()
        ci = model.confint()

        if focal_var in coefs.index:
            coef, se, pv = float(coefs[focal_var]), float(ses[focal_var]), float(pvals[focal_var])
            ci_l, ci_u = float(ci.loc[focal_var, '2.5%']), float(ci.loc[focal_var, '97.5%'])
        else:
            coef = se = pv = ci_l = ci_u = np.nan

        coef_dict = {tv: round(float(coefs[tv]), 6) for tv in all_treatment_vars if tv in coefs.index}

        return {
            'coefficient': round(coef, 6) if not np.isnan(coef) else np.nan,
            'std_error': round(se, 6) if not np.isnan(se) else np.nan,
            'p_value': round(pv, 6) if not np.isnan(pv) else np.nan,
            'ci_lower': round(ci_l, 6) if not np.isnan(ci_l) else np.nan,
            'ci_upper': round(ci_u, 6) if not np.isnan(ci_u) else np.nan,
            'n_obs': int(model._N), 'r_squared': round(float(model._r2), 4),
            'coefficient_vector_json': json.dumps(coef_dict)
        }
    except Exception as e:
        return {'coefficient': np.nan, 'std_error': np.nan, 'p_value': np.nan,
                'ci_lower': np.nan, 'ci_upper': np.nan,
                'n_obs': np.nan, 'r_squared': np.nan,
                'coefficient_vector_json': json.dumps({'error': str(e)})}


# ============================================================================
# CONTROL BLOCK DEFINITIONS
# ============================================================================
def get_geography(df):
    return get_cols(df, ['cotton_s_', 'corn_s_', 'ld_', 'dx_', 'dy_', 'rug_'])

def get_crop_suit(df):
    return get_cols(df, ['cotton_s_', 'corn_s_'])

def get_dist_ms(df):
    return get_cols(df, ['ld_'])

def get_coords(df):
    return get_cols(df, ['dx_', 'dy_'])

def get_rugg(df):
    return get_cols(df, ['rug_'])

def get_newdeal(df):
    return get_cols(df, ['lnpcpubwor_', 'lnpcaaa_', 'lnpcrelief_', 'lnpcndloan_', 'lnpcndins_'])

def get_tenancy_mfg(df):
    return get_cols(df, ['lag2_lnfarms_nonwhite_t_', 'lag3_lnfarms_nonwhite_t_',
                         'lag4_lnfarms_nonwhite_t_',
                         'lag2_lnmfgestab_', 'lag4_lnmfgestab_',
                         'lag2_lnmfgavewages_', 'lag4_lnmfgavewages_'])

def get_plantation(df):
    return get_cols(df, ['plantation_'])

def get_propscore(df):
    return get_cols(df, ['prop_plant_flstate_1'])

def get_lags_g1(df, depvar='lnfrac_black'):
    return get_cols(df, [f'lag2_{depvar}_', f'lag3_{depvar}_', f'lag4_{depvar}_'])

def get_lags_g2(df, depvar='lnvalue_equipment'):
    return get_cols(df, [f'lag1_{depvar}_', f'lag2_{depvar}_', f'lag3_{depvar}_', f'lag4_{depvar}_'])


# ============================================================================
# SPEC RUNNER FOR ONE GROUP
# ============================================================================
def run_group_specs(df_post, group_id, depvar, focal_var, treatment_vars,
                    get_lags_fn, results, run_id_start,
                    f2_treatment=None, f3_treatment=None,
                    has_pre1960=False):
    """Run all specs for one baseline group."""
    run_id = run_id_start
    geo = get_geography(df_post)
    nd = get_newdeal(df_post)
    lags = get_lags_fn(df_post, depvar)
    tenancy = get_tenancy_mfg(df_post)
    plant = get_plantation(df_post)
    pscore = get_propscore(df_post)
    crop = get_crop_suit(df_post)
    dist = get_dist_ms(df_post)
    coords_ = get_coords(df_post)
    rugg = get_rugg(df_post)

    f2_focal = focal_var.replace('f_int_', 'f2_int_') if f2_treatment else None
    f3_focal = focal_var.replace('f_int_', 'f3_int_') if f3_treatment else None

    def add_result(model, spec_id, tree_path, sample, controls, cluster='fips',
                   treat_var_label='f_int_{year}', focal=focal_var, treat_list=treatment_vars):
        nonlocal run_id
        run_id += 1
        res = extract_focal(model, focal, treat_list)
        results.append({
            'paper_id': PAPER_ID,
            'spec_run_id': f'{PAPER_ID}_{group_id}_{run_id:04d}',
            'spec_id': spec_id, 'spec_tree_path': tree_path,
            'baseline_group_id': group_id,
            'outcome_var': depvar, 'treatment_var': treat_var_label,
            'sample_desc': sample,
            'fixed_effects': 'county + state-by-year',
            'controls_desc': controls, 'cluster_var': cluster,
            **res
        })
        status = f"coef={res['coefficient']:.4f}" if not np.isnan(res['coefficient']) else "FAILED"
        print(f"  [{group_id}] {spec_id}: {status}, N={res['n_obs']}")

    # === BASELINES ===
    print(f"\n--- {group_id} Baselines ---")
    # Baseline 1: geo + lags
    m, e = run_spec(df_post, depvar, treatment_vars, geo + lags)
    add_result(m, 'baseline', 'design/difference_in_differences/estimator/twfe',
               f'Baseline (geography+lags)', 'geography + lagged DV')
    baseline1_res = extract_focal(m, focal_var, treatment_vars)

    # Baseline 2: geo + lags + ND
    m, e = run_spec(df_post, depvar, treatment_vars, geo + lags + nd)
    add_result(m, 'baseline', 'design/difference_in_differences/estimator/twfe',
               f'Baseline (geography+lags+NewDeal)', 'geography + lagged DV + New Deal')
    baseline2_res = extract_focal(m, focal_var, treatment_vars)

    # === CONTROL PROGRESSIONS ===
    print(f"\n--- {group_id} Control Progressions ---")
    progressions = {
        'none': [],
        'lagged_dv_only': lags,
        'geography_only': geo,
        # geography_and_lags = baseline1, skip
        'geography_lags_tenancy_mfg': geo + lags + tenancy,
        # geography_lags_newdeal = baseline2, skip
        'geography_lags_newdeal_plantation': geo + lags + nd + plant,
        'geography_lags_newdeal_tenancy_mfg': geo + lags + nd + tenancy,
        'geography_lags_newdeal_tenancy_mfg_propscore': geo + lags + nd + tenancy + pscore,
    }
    for name, ctrls in progressions.items():
        sid = 'rc/controls/sets/none' if name == 'none' else f'rc/controls/progression/{name}'
        m, e = run_spec(df_post, depvar, treatment_vars, ctrls)
        add_result(m, sid, sid, f'Progression: {name}', name)

    # === LOO BLOCKS (from baseline2 = geo + lags + ND) ===
    print(f"\n--- {group_id} LOO Blocks ---")
    base_loo = geo + lags + nd
    loo_defs = {
        'drop_geography': [c for c in base_loo if c not in set(geo)],
        'drop_lagged_dv': [c for c in base_loo if c not in set(lags)],
        'drop_new_deal': [c for c in base_loo if c not in set(nd)],
        'drop_crop_suitability': [c for c in base_loo if c not in set(crop)],
        'drop_distance_ms': [c for c in base_loo if c not in set(dist)],
        'drop_coordinates': [c for c in base_loo if c not in set(coords_)],
        'drop_ruggedness': [c for c in base_loo if c not in set(rugg)],
    }
    # drop_tenancy_mfg from step6
    step6 = geo + lags + nd + tenancy
    loo_defs['drop_tenancy_mfg'] = [c for c in step6 if c not in set(tenancy)]

    for name, ctrls in loo_defs.items():
        sid = f'rc/controls/loo_block/{name}'
        m, e = run_spec(df_post, depvar, treatment_vars, ctrls)
        add_result(m, sid, sid, f'LOO: {name}', f'base minus {name.replace("drop_","")}')

    # === UNWEIGHTED ===
    print(f"\n--- {group_id} Unweighted ---")
    for base_label, ctrls in [('geo+lags', geo + lags), ('geo+lags+ND', geo + lags + nd)]:
        m, e = run_spec(df_post, depvar, treatment_vars, ctrls, weights=None)
        add_result(m, 'rc/weights/main/unweighted', 'rc/weights/main/unweighted',
                   f'Unweighted ({base_label})', base_label)

    # === SAMPLE: DROP YEARS ===
    print(f"\n--- {group_id} Sample Restrictions ---")

    # Drop 1970
    df_no70 = df_post[df_post['year'] != 1970].copy()
    treat_no70 = [t for t in treatment_vars if t != 'f_int_1970']
    for base_label, ctrls in [('geo+lags', geo + lags), ('geo+lags+ND', geo + lags + nd)]:
        m, e = run_spec(df_no70, depvar, treat_no70, ctrls)
        add_result(m, 'rc/sample/time/drop_1970', 'rc/sample/time/drop_1970',
                   f'Drop 1970 ({base_label})', base_label, treat_list=treat_no70)

    # Drop 1930
    df_no30 = df_post[df_post['year'] != 1930].copy()
    treat_no30 = [t for t in treatment_vars if t != 'f_int_1930']
    for base_label, ctrls in [('geo+lags', geo + lags), ('geo+lags+ND', geo + lags + nd)]:
        m, e = run_spec(df_no30, depvar, treat_no30, ctrls)
        add_result(m, 'rc/sample/time/drop_1930', 'rc/sample/time/drop_1930',
                   f'Drop 1930 ({base_label})', base_label, treat_list=treat_no30)

    # Pre-1960 only (G2 only)
    if has_pre1960:
        df_pre60 = df_post[~df_post['year'].isin([1960, 1964, 1970])].copy()
        treat_pre60 = [t for t in treatment_vars if not any(t.endswith(str(y)) for y in [1960, 1964, 1970])]
        for base_label, ctrls in [('geo+lags', geo + lags), ('geo+lags+ND', geo + lags + nd)]:
            m, e = run_spec(df_pre60, depvar, treat_pre60, ctrls)
            add_result(m, 'rc/sample/time/pre1960_only', 'rc/sample/time/pre1960_only',
                       f'Pre-1960 only ({base_label})', base_label, treat_list=treat_pre60)

    # Trim treatment P95
    print(f"\n--- {group_id} Trim Treatment ---")
    p95 = df_post.loc[df_post['flood_intensity'] > 0, 'flood_intensity'].quantile(0.95)
    df_trim = df_post[df_post['flood_intensity'] <= p95].copy()
    for base_label, ctrls in [('geo+lags', geo + lags), ('geo+lags+ND', geo + lags + nd)]:
        m, e = run_spec(df_trim, depvar, treatment_vars, ctrls)
        add_result(m, 'rc/sample/outliers/trim_treatment_p95', 'rc/sample/outliers/trim_treatment_p95',
                   f'Trim P95 ({base_label})', base_label)

    # === ALTERNATIVE TREATMENT MEASURES ===
    print(f"\n--- {group_id} Alternative Treatment ---")
    if f2_treatment:
        for base_label, ctrls in [('geo+lags', geo + lags), ('geo+lags+ND', geo + lags + nd)]:
            m, e = run_spec(df_post, depvar, f2_treatment, ctrls)
            add_result(m, 'rc/form/treatment/alt_measure_redcross_acres',
                       'rc/form/treatment/alt_measure_redcross_acres',
                       f'RedCross acres ({base_label})', base_label,
                       treat_var_label='f2_int_{year}', focal=f2_focal, treat_list=f2_treatment)

    if f3_treatment:
        for base_label, ctrls in [('geo+lags', geo + lags), ('geo+lags+ND', geo + lags + nd)]:
            m, e = run_spec(df_post, depvar, f3_treatment, ctrls)
            add_result(m, 'rc/form/treatment/alt_measure_redcross_people',
                       'rc/form/treatment/alt_measure_redcross_people',
                       f'RedCross people ({base_label})', base_label,
                       treat_var_label='f3_int_{year}', focal=f3_focal, treat_list=f3_treatment)

    # === INFERENCE VARIANTS ===
    # Try baseline2 (geo+lags+ND) first; fall back to baseline1 (geo+lags) if singular
    print(f"\n--- {group_id} Inference Variants ---")
    for infer_ctrls, infer_label in [(geo + lags + nd, 'geo+lags+ND'), (geo + lags, 'geo+lags')]:
        # HC1
        m, e = run_spec(df_post, depvar, treatment_vars, infer_ctrls, cluster=None)
        if m:
            add_result(m, 'infer/se/hc/hc1', 'infer/se/hc/hc1',
                       f'HC1 SE ({infer_label})', infer_label, cluster='none (HC1)')
            break

    for infer_ctrls, infer_label in [(geo + lags + nd, 'geo+lags+ND'), (geo + lags, 'geo+lags')]:
        # State cluster
        m, e = run_spec(df_post, depvar, treatment_vars, infer_ctrls, cluster='statefips')
        if m:
            add_result(m, 'infer/se/cluster/state', 'infer/se/cluster/state',
                       f'State-clustered SE ({infer_label})', infer_label, cluster='statefips')
            break

    for infer_ctrls, infer_label in [(geo + lags + nd, 'geo+lags+ND'), (geo + lags, 'geo+lags')]:
        # Unit cluster
        m, e = run_spec(df_post, depvar, treatment_vars, infer_ctrls, cluster='fips')
        if m:
            add_result(m, 'infer/se/cluster/unit', 'infer/se/cluster/unit',
                       f'County-clustered SE ({infer_label})', infer_label, cluster='fips')
            break

    # Conley SEs (coefficient only, SE not available)
    # Use whichever baseline succeeded
    ref_res = baseline2_res if not np.isnan(baseline2_res['coefficient']) else baseline1_res
    for dist_km, dist_label in [(50, 'conley_50km'), (100, 'conley_100km'), (200, 'conley_200km')]:
        run_id += 1
        results.append({
            'paper_id': PAPER_ID,
            'spec_run_id': f'{PAPER_ID}_{group_id}_{run_id:04d}',
            'spec_id': f'infer/se/spatial/{dist_label}',
            'spec_tree_path': f'infer/se/spatial/{dist_label}',
            'baseline_group_id': group_id,
            'outcome_var': depvar, 'treatment_var': 'f_int_{year}',
            'sample_desc': f'Conley SE ({dist_km}km)',
            'fixed_effects': 'county + state-by-year',
            'controls_desc': 'geo+lags',
            'cluster_var': f'spatial_{dist_km}km',
            'coefficient': ref_res['coefficient'],
            'std_error': np.nan, 'p_value': np.nan,
            'ci_lower': np.nan, 'ci_upper': np.nan,
            'n_obs': ref_res['n_obs'],
            'r_squared': ref_res['r_squared'],
            'coefficient_vector_json': json.dumps({
                'note': f'Conley {dist_km}km SE not computable in pyfixest',
                focal_var: ref_res['coefficient']
            })
        })
        print(f"  [{group_id}] infer/se/spatial/{dist_label}: coef={ref_res['coefficient']}, SE=N/A")

    return run_id


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Specification search started: {start_time}")
    print(f"Paper: {PAPER_ID}\n")

    df_main, df_post = build_analysis_data()

    results = []

    # G1: Black population share
    print("\n" + "=" * 70)
    print("G1: FLOOD IMPACT ON BLACK SHARE OF POPULATION")
    print("=" * 70)
    G1_TREAT = ['f_int_1930', 'f_int_1940', 'f_int_1950', 'f_int_1960', 'f_int_1970']
    G1_F2 = ['f2_int_1930', 'f2_int_1940', 'f2_int_1950', 'f2_int_1960', 'f2_int_1970']
    G1_F3 = ['f3_int_1930', 'f3_int_1940', 'f3_int_1950', 'f3_int_1960', 'f3_int_1970']

    g1_end = run_group_specs(
        df_post, 'G1', 'lnfrac_black', 'f_int_1950', G1_TREAT,
        get_lags_g1, results, 0,
        f2_treatment=G1_F2, f3_treatment=G1_F3,
        has_pre1960=False
    )

    # G2: Agricultural capital intensity
    print("\n" + "=" * 70)
    print("G2: FLOOD IMPACT ON AGRICULTURAL CAPITAL INTENSITY")
    print("=" * 70)
    # G2 treatment: all post-flood years (even if DV missing for some)
    # pyfixest will drop obs where DV is missing; treatment vars for those years
    # will be zero in the non-missing sample and automatically excluded
    G2_TREAT = ['f_int_1930', 'f_int_1935', 'f_int_1940', 'f_int_1945',
                'f_int_1950', 'f_int_1954', 'f_int_1960', 'f_int_1964', 'f_int_1970']
    G2_F2 = ['f2_int_1930', 'f2_int_1935', 'f2_int_1940', 'f2_int_1945',
             'f2_int_1950', 'f2_int_1954', 'f2_int_1960', 'f2_int_1964', 'f2_int_1970']
    G2_F3 = ['f3_int_1930', 'f3_int_1935', 'f3_int_1940', 'f3_int_1945',
             'f3_int_1950', 'f3_int_1954', 'f3_int_1960', 'f3_int_1964', 'f3_int_1970']

    g2_end = run_group_specs(
        df_post, 'G2', 'lnvalue_equipment', 'f_int_1940', G2_TREAT,
        get_lags_g2, results, g1_end,
        f2_treatment=G2_F2, f3_treatment=G2_F3,
        has_pre1960=True
    )

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    col_order = [
        'paper_id', 'spec_run_id', 'spec_id', 'spec_tree_path',
        'baseline_group_id', 'outcome_var', 'treatment_var',
        'coefficient', 'std_error', 'p_value', 'ci_lower', 'ci_upper',
        'n_obs', 'r_squared', 'coefficient_vector_json',
        'sample_desc', 'fixed_effects', 'controls_desc', 'cluster_var'
    ]
    for c in col_order:
        if c not in results_df.columns:
            results_df[c] = ''
    results_df = results_df[col_order]

    # Save CSV
    out_csv = os.path.join(PKG_DIR, "specification_results.csv")
    results_df.to_csv(out_csv, index=False)

    # Summary
    n_total = len(results_df)
    n_g1 = (results_df['baseline_group_id'] == 'G1').sum()
    n_g2 = (results_df['baseline_group_id'] == 'G2').sum()
    n_baseline = (results_df['spec_id'] == 'baseline').sum()
    n_rc = results_df['spec_id'].str.startswith('rc/').sum()
    n_infer = results_df['spec_id'].str.startswith('infer/').sum()
    n_failed = results_df['coefficient'].isna().sum()
    n_success = n_total - n_failed
    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total: {n_total} | G1: {n_g1} | G2: {n_g2}")
    print(f"Baselines: {n_baseline} | RC: {n_rc} | Infer: {n_infer}")
    print(f"Succeeded: {n_success} | Failed: {n_failed}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Saved to {out_csv}")

    # Write SPECIFICATION_SEARCH.md
    md = f"""# Specification Search Report: {PAPER_ID}

## Paper
Hornbeck & Naidu (2014), "When the Levee Breaks: Black Migration and Economic Development in the American South", AER 104(3): 963-990.

## Surface Summary
- **Baseline groups**: 2 (G1: Black population share, G2: Agricultural capital intensity)
- **Design**: Continuous-treatment DiD with county FE (absorbed) + state-by-year FE (absorbed)
- **Sampling**: Full enumeration (block-based control progressions, not combinatorial)
- **Seed**: G1=112749001, G2=112749002 (not used; full enumeration)
- **Budget**: G1 max 80, G2 max 85

## Execution Summary
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
- **Total specs**: {n_total}
- **G1**: {n_g1} | **G2**: {n_g2}
- **Baselines**: {n_baseline} | **RC**: {n_rc} | **Infer**: {n_infer}
- **Succeeded**: {n_success} | **Failed**: {n_failed}
- **Elapsed**: {elapsed:.1f}s

## Baseline Results

### G1: lnfrac_black
- Table 2 Col 1 (geo+lags): focal f_int_1950 coef ~ -0.202, N=978
- Table 2 Col 2 (geo+lags+ND): focal f_int_1950 coef ~ -0.240, N=978

### G2: lnvalue_equipment
- Note: Equipment data only available for years 1920, 1925, 1930, 1940, 1970 (5 of 13 panel years)
- Table 4 Col 3 (geo+lags): focal f_int_1940 (peak available post-flood year), N=815
- Table 4 Col 4 (geo+lags+ND): focal f_int_1940, N=815
- G2 focal changed to f_int_1940 because f_int_1950 has zero variation in equipment-available sample

## Spec Categories

### Control Progressions (rc/controls/progression/*)
1. `none` -- FE only (stress test)
2. `lagged_dv_only` -- lagged DV (RefTable step 1)
3. `geography_only` -- geography only (stress test)
4. `geography_and_lags` -- = baseline 1 (RefTable step 2)
5. `geography_lags_tenancy_mfg` -- + tenancy/mfg (RefTable step 3)
6. `geography_lags_newdeal` -- = baseline 2 (RefTable step 4)
7. `geography_lags_newdeal_plantation` -- + plantation (RefTable step 5)
8. `geography_lags_newdeal_tenancy_mfg` -- step 6
9. `geography_lags_newdeal_tenancy_mfg_propscore` -- step 7

### LOO Blocks (rc/controls/loo_block/*)
Relative to baseline 2: drop_geography, drop_lagged_dv, drop_new_deal, drop_crop_suitability, drop_distance_ms, drop_coordinates, drop_ruggedness, drop_tenancy_mfg.

### Weight Variants
- `rc/weights/main/unweighted` -- no analytic weights

### Sample Restrictions
- `rc/sample/time/drop_1970`, `drop_1930`
- `rc/sample/time/pre1960_only` (G2 only)
- `rc/sample/outliers/trim_treatment_p95`

### Treatment Form
- `rc/form/treatment/alt_measure_redcross_acres` (f2_int)
- `rc/form/treatment/alt_measure_redcross_people` (f3_int)

### Inference
- `infer/se/cluster/unit` (county)
- `infer/se/hc/hc1` (heteroskedasticity-robust)
- `infer/se/cluster/state`
- `infer/se/spatial/conley_50km`, `conley_100km`, `conley_200km` (coef only; SE=N/A)

## Deviations

1. **G2 focal parameter**: Changed from f_int_1950 to f_int_1940 because equipment data is only available for years 1920/1925/1930/1940/1970. The 1950 treatment dummy has zero variation in the equipment-available sample.

2. **Conley SE**: Not computable in pyfixest. Point estimates are identical to baseline; SEs recorded as NaN.

3. **State-year FE**: Absorbed via `| fips + state_year_fe` (pyfixest multi-way FE) instead of including d_sy_* dummies on RHS. This avoids collinearity issues, especially for G2 where only 5 time periods have non-missing DV.

4. **Some control progressions may fail** if the control block introduces perfect collinearity in the G2 restricted sample.

## Software
- Python 3.x, pyfixest (TWFE with multi-way absorbed FE, CRV1), pandas, numpy, statsmodels
"""
    md_path = os.path.join(PKG_DIR, "SPECIFICATION_SEARCH.md")
    with open(md_path, 'w') as f:
        f.write(md)
    print(f"Saved {md_path}")
