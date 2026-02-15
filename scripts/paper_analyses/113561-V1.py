"""
Specification Search for Fong & Luttmer (2009)
"What Determines Giving to Hurricane Katrina Victims?"
American Economic Journal: Applied Economics, 1(2), 64-87.

Paper ID: 113561-V1

Executes the approved SPECIFICATION_SURFACE.json across 4 baseline groups:
  G1: giving (experimental dictator game, 0-100)
  G2: hypgiv_tc500 (hypothetical giving, topcoded at 500)
  G3: subjsupchar (charity support, 1-7 scale)
  G4: subjsupgov (government support, 1-7 scale)

All baselines use WLS with HC1 robust SE on white respondents.
Focal treatment: picshowblack (pictures show black Katrina victims).
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import json
import os
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import t as t_dist

# =============================================================================
# Configuration
# =============================================================================
PAPER_ID = "113561-V1"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PACKAGE_DIR = os.path.join(REPO_ROOT, "data", "downloads", "extracted", PAPER_ID)

# =============================================================================
# Load and prepare data (identical to replication script)
# =============================================================================
df = pd.read_stata(os.path.join(PACKAGE_DIR, "katrina.dta"), convert_categoricals=False)

# Sample selection: keep soundcheck==3, non-missing giving
df = df[df['soundcheck'] == 3].copy()
df = df[df['giving'].notna()].copy()

# --- Data cleaning (matching katrina.do Section 1) ---
df['var_racesalient'] = (df['surveyvariant'] == 2).astype(int)
df['var_fullstakes'] = (df['surveyvariant'] == 3).astype(int)

# Hypothetical giving topcoded at 500
df['hypgiv_tc500'] = df['hypothgiving'].copy()
df.loc[df['hypgiv_tc500'] > 500, 'hypgiv_tc500'] = 500

# Race/ethnicity
df['white'] = (df['ppethm'] == 1).astype(int)
df['black'] = (df['ppethm'] == 2).astype(int)
df['other'] = (1 - df['black'] - df['white']).astype(int)

# Age
df['age'] = df['ppage']
df['age2'] = df['ppage'] ** 2

# Dual income
df['dualin'] = df['ppdualin']

# Education dummies
df['edudo'] = (df['ppeducat'] == 1).astype(int)
df['edusc'] = (df['ppeducat'] == 3).astype(int)
df['educp'] = (df['ppeducat'] == 4).astype(int)

# Log household income
inc_map = {
    1: np.log(2500), 2: np.log((5000 + 7499) / 2), 3: np.log((7500 + 9999) / 2),
    4: np.log((10000 + 12499) / 2), 5: np.log((12500 + 14999) / 2),
    6: np.log((15000 + 19999) / 2), 7: np.log((20000 + 24999) / 2),
    8: np.log((25000 + 29999) / 2), 9: np.log((30000 + 34999) / 2),
    10: np.log((35000 + 39999) / 2), 11: np.log((40000 + 49999) / 2),
    12: np.log((50000 + 59999) / 2), 13: np.log((60000 + 74999) / 2),
    14: np.log((75000 + 84999) / 2), 15: np.log((85000 + 99999) / 2),
    16: np.log((100000 + 124999) / 2), 17: np.log((125000 + 149999) / 2),
    18: np.log((150000 + 174999) / 2), 19: np.log(350000),
}
df['lnhhinc'] = df['ppincimp'].map(inc_map)

# Marital status, gender
df['married'] = (df['ppmarit'] == 1).astype(int)
df['male'] = (df['ppgender'] == 1).astype(int)
df['singlemale'] = (df['male'] & ~df['married'].astype(bool)).astype(int)

# Region
df['south'] = (df['ppreg4'] == 3).astype(int)

# Labor force
df['work'] = (df['ppwork'] <= 4).astype(int)
df['retired'] = (df['ppwork'] == 6).astype(int)
df['disabled'] = (df['ppwork'] == 7).astype(int)

# Charitable giving
df['dcharkatrina'] = ((df['charkatrina'] > 0) & (df['charkatrina'].notna())).astype(int)
df['lcharkatrina'] = np.log(df['charkatrina'].replace(0, np.nan))
df['lcharkatrina'] = df['lcharkatrina'].fillna(0)
df['dchartot2005'] = ((df['chartot2005'] > 0) & (df['chartot2005'].notna())).astype(int)
df['lchartot2005'] = np.log(df['chartot2005'].replace(0, np.nan))
df['lchartot2005'] = df['lchartot2005'].fillna(0)

# Life priorities
for val, rank_label in [(2, 'help'), (6, 'mony')]:
    col = f'lifepriorities_{rank_label}'
    mask = df['lifepriorities5'].notna()
    df[col] = np.nan
    for i, pri_col in enumerate(['lifepriorities1', 'lifepriorities2', 'lifepriorities3',
                                   'lifepriorities4', 'lifepriorities5'], 1):
        weight = 6 - i
        base_contrib = (df[pri_col] == val).astype(float) * weight
        df.loc[mask, col] = df.loc[mask, col].fillna(0) + base_contrib[mask]
    df.loc[mask, col] = df.loc[mask, col] + 1

# HFH effective
# Already exists in the dataset as hfh_effective

# nraudworthy
df['nraudworthy'] = df['aud_helpoth'] - df['aud_crime'] + df['aud_contrib'] + df['aud_prephur']

print(f"Full sample: N={len(df)}")
print(f"White sample: N={df['white'].sum()}")


# =============================================================================
# Define variable groups
# =============================================================================
treatment_vars = ['picshowblack', 'picraceb', 'picobscur']

# Baseline controls: nraud + cntrldems
nraud = ['aud_econdis', 'nraudworthy', 'aud_republ', 'aud_govtben', 'aud_church',
         'aud_loot', 'cityslidell', 'var_fullstakes', 'var_racesalient']

demographics = ['age', 'age2', 'edudo', 'edusc', 'educp', 'lnhhinc', 'dualin',
                'married', 'male', 'singlemale', 'south', 'work', 'disabled', 'retired']

charitable = ['dcharkatrina', 'lcharkatrina', 'dchartot2005', 'lchartot2005']

race_controls = ['black', 'other']

extra_controls = ['hfh_effective', 'lifepriorities_help', 'lifepriorities_mony']

# Separate worthiness manipulation controls (Table 3 style)
manip_full = ['aud_republ', 'aud_econdis', 'aud_govtben', 'aud_prephur', 'aud_church',
              'aud_crime', 'aud_helpoth', 'aud_contrib', 'aud_loot', 'cityslidell',
              'var_fullstakes', 'var_racesalient']

# Baseline controls for white subsample
baseline_controls_white = nraud + demographics + charitable  # 27 effective (black/other are zero)
baseline_controls_formula = nraud + ['black', 'other'] + demographics + charitable  # 29 in formula


# =============================================================================
# Helper: drop collinear columns
# =============================================================================
def drop_collinear(X_df):
    """Drop collinear columns from a DataFrame, mimicking Stata's behavior."""
    cols_to_keep = []
    for col in X_df.columns:
        if col == 'const':
            cols_to_keep.append(col)
            continue
        if X_df[col].std() == 0:
            continue
        if len(cols_to_keep) > 0:
            test_X = X_df[cols_to_keep].values
            new_col = X_df[col].values
            rank_before = np.linalg.matrix_rank(test_X)
            rank_after = np.linalg.matrix_rank(np.column_stack([test_X, new_col]))
            if rank_after <= rank_before:
                continue
        cols_to_keep.append(col)
    return X_df[cols_to_keep]


# =============================================================================
# Core estimation function
# =============================================================================
def run_wls(depvar, indepvars, data, weight_var='tweight', se_type='HC1'):
    """
    Run WLS with specified robust SEs.
    Returns dict with coefficient, se, pvalue, ci, nobs, r2, coef_vector.
    se_type: 'HC1', 'HC2', 'HC3', 'nonrobust' (classical)
    """
    all_vars = [depvar] + indepvars + ([weight_var] if weight_var else [])
    dfreg = data[all_vars].dropna().copy()

    y = dfreg[depvar]
    X = sm.add_constant(dfreg[indepvars])
    X = drop_collinear(X)

    if weight_var:
        w = dfreg[weight_var]
        model = sm.WLS(y, X, weights=w)
    else:
        model = sm.OLS(y, X)

    res = model.fit(cov_type=se_type)

    focal = 'picshowblack'
    coef = res.params.get(focal, np.nan)
    se = res.bse.get(focal, np.nan)
    pval = res.pvalues.get(focal, np.nan)
    if focal in res.params.index:
        ci = res.conf_int().loc[focal]
        ci_lower = float(ci.iloc[0])
        ci_upper = float(ci.iloc[1])
    else:
        ci_lower = np.nan
        ci_upper = np.nan

    coef_dict = {k: round(float(v), 8) for k, v in res.params.items()}

    return {
        'coefficient': round(float(coef), 8),
        'std_error': round(float(se), 8),
        'p_value': round(float(pval), 8),
        'ci_lower': round(float(ci_lower), 8),
        'ci_upper': round(float(ci_upper), 8),
        'n_obs': int(res.nobs),
        'r_squared': round(float(res.rsquared), 8),
        'coefficient_vector_json': coef_dict,
        'dropped_vars': [v for v in indepvars if v not in X.columns and v != 'const'],
    }


# =============================================================================
# Baseline group definitions
# =============================================================================
baseline_groups = {
    'G1': {
        'outcome_var': 'giving',
        'replicated_coef': -4.198,
        'replicated_pvalue': 0.370,
        'replicated_nobs': 915,
        'label': 'Table4-R1-White (reg_id=8)',
        # G1-specific: has outcome preprocessing (topcode, winsorize) and extra control progressions
        'has_outcome_preprocess': True,
        'has_extra_progression': True,
        'infer_variants': ['HC2', 'HC3', 'nonrobust'],  # G1 also has classical
    },
    'G2': {
        'outcome_var': 'hypgiv_tc500',
        'replicated_coef': -2.181,
        'replicated_pvalue': 0.591,
        'replicated_nobs': 913,
        'label': 'Table4-R2-White (reg_id=11)',
        'has_outcome_preprocess': True,  # topcode at 250, no topcode
        'has_extra_progression': False,
        'infer_variants': ['HC2', 'HC3'],
    },
    'G3': {
        'outcome_var': 'subjsupchar',
        'replicated_coef': -0.221,
        'replicated_pvalue': 0.167,
        'replicated_nobs': 907,
        'label': 'Table4-R3-White (reg_id=14)',
        'has_outcome_preprocess': False,
        'has_extra_progression': False,
        'infer_variants': ['HC2', 'HC3'],
    },
    'G4': {
        'outcome_var': 'subjsupgov',
        'replicated_coef': -0.435,
        'replicated_pvalue': 0.026,
        'replicated_nobs': 913,
        'label': 'Table4-R4-White (reg_id=17)',
        'has_outcome_preprocess': False,
        'has_extra_progression': False,
        'infer_variants': ['HC2', 'HC3'],
    },
}

# =============================================================================
# Run all specifications
# =============================================================================
all_results = []
run_counter = 0
failed_specs = []
skipped_specs = []


def add_result(spec_id, spec_tree_path, group_id, outcome_var, controls_desc,
               sample_desc, weight_desc, se_desc, est_result, notes=""):
    """Add a result row to the all_results list."""
    global run_counter
    run_counter += 1
    spec_run_id = f"{PAPER_ID}_{run_counter:04d}"

    # Store realized controls + any extra info in coef_vector_json
    coef_json = est_result['coefficient_vector_json']
    if est_result.get('dropped_vars'):
        coef_json['_dropped_collinear'] = est_result['dropped_vars']

    row = {
        'paper_id': PAPER_ID,
        'spec_run_id': spec_run_id,
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'baseline_group_id': group_id,
        'outcome_var': outcome_var,
        'treatment_var': 'picshowblack',
        'coefficient': est_result['coefficient'],
        'std_error': est_result['std_error'],
        'p_value': est_result['p_value'],
        'ci_lower': est_result['ci_lower'],
        'ci_upper': est_result['ci_upper'],
        'n_obs': est_result['n_obs'],
        'r_squared': est_result['r_squared'],
        'coefficient_vector_json': json.dumps(coef_json),
        'sample_desc': sample_desc,
        'fixed_effects': 'none',
        'controls_desc': controls_desc,
        'cluster_var': '',
        'weight_desc': weight_desc,
        'se_type': se_desc,
        'notes': notes,
    }
    all_results.append(row)
    print(f"  [{spec_run_id}] {spec_id}: coef={est_result['coefficient']:.4f}, "
          f"se={est_result['std_error']:.4f}, p={est_result['p_value']:.4f}, "
          f"N={est_result['n_obs']}")
    return row


# Prepare outcome preprocessing variants
# Topcode giving at value 99 (changes 437 observations at giving=100 to 99)
df['giving_tc99'] = df['giving'].clip(upper=99)

# Winsorize giving at 1st/99th percentile
# Since P1=0 and P99=100, standard 1/99 is a no-op on this bounded 0-100 scale.
# Apply topcode at 99 AND floor at 1 for a meaningful two-sided winsorization.
white_mask = df['white'] == 1
df['giving_winsor'] = df['giving'].clip(lower=1, upper=99)

# Hypothetical giving: topcode at 250, no topcode
df['hypgiv_tc250'] = df['hypothgiving'].clip(upper=250)
df['hypgiv_notc'] = df['hypothgiving'].copy()


# =============================================================================
# LOO drops: demographic + charitable giving block (18 vars on white subsample)
# age/age2 are paired -> 17 LOO drops
# =============================================================================
loo_vars_single = ['edudo', 'edusc', 'educp', 'lnhhinc', 'dualin',
                   'married', 'male', 'singlemale', 'south', 'work', 'disabled', 'retired',
                   'dcharkatrina', 'lcharkatrina', 'dchartot2005', 'lchartot2005']
loo_vars_paired = [('age', 'age2')]  # dropped as a pair


# =============================================================================
# Execute for each baseline group
# =============================================================================
for gid, ginfo in baseline_groups.items():
    outcome = ginfo['outcome_var']
    print(f"\n{'='*70}")
    print(f"BASELINE GROUP {gid}: {outcome}")
    print(f"{'='*70}")

    # ------------------------------------------------------------------
    # BASELINE
    # ------------------------------------------------------------------
    print(f"\n--- Baseline ---")
    baseline_indepvars = treatment_vars + baseline_controls_formula
    white_data = df[df['white'] == 1].copy()

    res = run_wls(outcome, baseline_indepvars, white_data, weight_var='tweight', se_type='HC1')

    # Verify against replicated values
    coef_diff = abs(res['coefficient'] - ginfo['replicated_coef'])
    print(f"  Baseline verification: coef={res['coefficient']:.4f} vs replicated={ginfo['replicated_coef']:.3f}, "
          f"diff={coef_diff:.6f}, N={res['n_obs']} vs {ginfo['replicated_nobs']}")

    add_result(
        spec_id='baseline',
        spec_tree_path='specification_tree/designs/randomized_experiment.md#baseline',
        group_id=gid,
        outcome_var=outcome,
        controls_desc='nraud + demographics + charitable giving (baseline)',
        sample_desc='white respondents',
        weight_desc='tweight (survey weights)',
        se_desc='HC1',
        est_result=res,
        notes=f'Baseline: {ginfo["label"]}; replicated coef={ginfo["replicated_coef"]}'
    )

    # ------------------------------------------------------------------
    # DESIGN VARIANT: diff_in_means (no covariates except treatment dummies)
    # This is NOT the same as rc/controls/sets/none -- diff_in_means is
    # truly bivariate (no controls at all), which is the design variant.
    # ------------------------------------------------------------------
    print(f"\n--- Design: diff_in_means ---")
    res = run_wls(outcome, treatment_vars, white_data, weight_var='tweight', se_type='HC1')
    add_result(
        spec_id='design/randomized_experiment/estimator/diff_in_means',
        spec_tree_path='specification_tree/designs/randomized_experiment.md#estimators',
        group_id=gid,
        outcome_var=outcome,
        controls_desc='none (treatment dummies only)',
        sample_desc='white respondents',
        weight_desc='tweight (survey weights)',
        se_desc='HC1',
        est_result=res,
        notes='Diff-in-means: regression of outcome on treatment dummies only, no controls'
    )

    # ------------------------------------------------------------------
    # RC: CONTROL SETS
    # ------------------------------------------------------------------
    print(f"\n--- RC: Control sets ---")

    # rc/controls/sets/none -- bivariate (same as diff_in_means but classified as RC)
    # SKIP: surface says design/diff_in_means is the no-controls spec;
    # rc/controls/sets/none is listed separately in the surface
    res = run_wls(outcome, treatment_vars, white_data, weight_var='tweight', se_type='HC1')
    add_result(
        spec_id='rc/controls/sets/none',
        spec_tree_path='specification_tree/modules/robustness/controls.md#control-sets',
        group_id=gid,
        outcome_var=outcome,
        controls_desc='none (bivariate)',
        sample_desc='white respondents',
        weight_desc='tweight (survey weights)',
        se_desc='HC1',
        est_result=res,
        notes='No controls (bivariate regression)'
    )

    # rc/controls/sets/extended -- baseline + extra controls (Table 5 s6)
    extended_controls = baseline_controls_formula + extra_controls
    res = run_wls(outcome, treatment_vars + extended_controls, white_data, weight_var='tweight', se_type='HC1')
    add_result(
        spec_id='rc/controls/sets/extended',
        spec_tree_path='specification_tree/modules/robustness/controls.md#control-sets',
        group_id=gid,
        outcome_var=outcome,
        controls_desc='nraud + demographics + charitable + extra (hfh_effective, lifepriorities)',
        sample_desc='white respondents',
        weight_desc='tweight (survey weights)',
        se_desc='HC1',
        est_result=res,
        notes='Extended controls (Table 5 s6): baseline + hfh_effective + lifepriorities_help + lifepriorities_mony'
    )

    # ------------------------------------------------------------------
    # RC: CONTROL PROGRESSIONS (only for G1, or where specified in surface)
    # ------------------------------------------------------------------
    print(f"\n--- RC: Control progressions ---")

    # Check which progressions are in this group's surface
    rc_spec_ids = baseline_groups[gid].get('_rc_spec_ids', [])

    # rc/controls/progression/manipulation_only -- only nraud (no demographics, no charitable)
    # This is in G1 only per the surface
    if gid == 'G1':
        manip_only_controls = nraud + race_controls  # race_controls will be collinear zeros on white sample
        res = run_wls(outcome, treatment_vars + manip_only_controls, white_data, weight_var='tweight', se_type='HC1')
        add_result(
            spec_id='rc/controls/progression/manipulation_only',
            spec_tree_path='specification_tree/modules/robustness/controls.md#control-progression',
            group_id=gid,
            outcome_var=outcome,
            controls_desc='nraud only (no demographics, no charitable)',
            sample_desc='white respondents',
            weight_desc='tweight (survey weights)',
            se_desc='HC1',
            est_result=res,
            notes='Manipulation controls only (Table 5 s5 equivalent on white subsample)'
        )

    # rc/controls/progression/manipulation_plus_demographics -- nraud + demographics (no charitable)
    if gid == 'G1':
        manip_demo_controls = nraud + race_controls + demographics
        res = run_wls(outcome, treatment_vars + manip_demo_controls, white_data, weight_var='tweight', se_type='HC1')
        add_result(
            spec_id='rc/controls/progression/manipulation_plus_demographics',
            spec_tree_path='specification_tree/modules/robustness/controls.md#control-progression',
            group_id=gid,
            outcome_var=outcome,
            controls_desc='nraud + demographics (no charitable giving controls)',
            sample_desc='white respondents',
            weight_desc='tweight (survey weights)',
            se_desc='HC1',
            est_result=res,
            notes='Manipulation + demographics, no charitable giving controls'
        )

    # rc/controls/progression/manipulation_plus_demographics_plus_charity (all groups except G1 where it's the baseline)
    # For G1 this is the baseline, but the surface lists it for G1 too
    # Actually: for G1, baseline already includes nraud + demographics + charitable, so
    # manip+demo+charity IS the baseline. But the surface includes it for G2-G4.
    # For G2-G4, the baseline is nraud + race + demographics + charitable, same thing.
    # So this is effectively the same as baseline. Let me check the surface more carefully.
    # The surface for G2-G4 lists: rc/controls/progression/manipulation_plus_demographics_plus_charity
    # This IS the baseline controls. But the surface explicitly lists it, so it must differ.
    # Looking at the surface notes: baseline controls include race_controls (black, other),
    # and the progression could be nraud + demographics + charitable WITHOUT race_controls.
    # But on white subsample, race_controls are collinear zeros anyway.
    # Since the surface lists it, I'll run it without race_controls for non-G1 groups.
    if gid in ['G2', 'G3', 'G4']:
        manip_demo_charity_controls = nraud + demographics + charitable
        res = run_wls(outcome, treatment_vars + manip_demo_charity_controls, white_data, weight_var='tweight', se_type='HC1')
        add_result(
            spec_id='rc/controls/progression/manipulation_plus_demographics_plus_charity',
            spec_tree_path='specification_tree/modules/robustness/controls.md#control-progression',
            group_id=gid,
            outcome_var=outcome,
            controls_desc='nraud + demographics + charitable (no race controls)',
            sample_desc='white respondents',
            weight_desc='tweight (survey weights)',
            se_desc='HC1',
            est_result=res,
            notes='nraud + demographics + charitable giving (without race dummies, which are collinear zeros on white sample)'
        )
    elif gid == 'G1':
        # G1 surface lists this too
        manip_demo_charity_controls = nraud + demographics + charitable
        res = run_wls(outcome, treatment_vars + manip_demo_charity_controls, white_data, weight_var='tweight', se_type='HC1')
        add_result(
            spec_id='rc/controls/progression/manipulation_plus_demographics_plus_charity',
            spec_tree_path='specification_tree/modules/robustness/controls.md#control-progression',
            group_id=gid,
            outcome_var=outcome,
            controls_desc='nraud + demographics + charitable (no race controls)',
            sample_desc='white respondents',
            weight_desc='tweight (survey weights)',
            se_desc='HC1',
            est_result=res,
            notes='nraud + demographics + charitable giving (without race dummies)'
        )

    # rc/controls/progression/full -- nraud + race + demographics + charitable + extra
    # For G1 only per surface, but G2-G4 have it too
    full_controls = nraud + race_controls + demographics + charitable + extra_controls
    res = run_wls(outcome, treatment_vars + full_controls, white_data, weight_var='tweight', se_type='HC1')
    add_result(
        spec_id='rc/controls/progression/full',
        spec_tree_path='specification_tree/modules/robustness/controls.md#control-progression',
        group_id=gid,
        outcome_var=outcome,
        controls_desc='nraud + race + demographics + charitable + extra (full control set)',
        sample_desc='white respondents',
        weight_desc='tweight (survey weights)',
        se_desc='HC1',
        est_result=res,
        notes='Full control set including possibly endogenous extra controls'
    )

    # ------------------------------------------------------------------
    # RC: MANIPULATION CODING VARIANT
    # ------------------------------------------------------------------
    print(f"\n--- RC: Manipulation coding ---")

    # rc/controls/manipulation_coding/separate_worthiness
    # Use manip_full (12 separate worthiness indicators) instead of nraud (9, with nraudworthy composite)
    sep_worth_controls = manip_full + race_controls + demographics + charitable
    res = run_wls(outcome, treatment_vars + sep_worth_controls, white_data, weight_var='tweight', se_type='HC1')
    add_result(
        spec_id='rc/controls/manipulation_coding/separate_worthiness',
        spec_tree_path='specification_tree/modules/robustness/controls.md#manipulation-coding',
        group_id=gid,
        outcome_var=outcome,
        controls_desc='manip_full (12 separate worthiness) + race + demographics + charitable',
        sample_desc='white respondents',
        weight_desc='tweight (survey weights)',
        se_desc='HC1',
        est_result=res,
        notes='Separate worthiness indicators (aud_prephur, aud_crime, aud_helpoth, aud_contrib) instead of nraudworthy composite'
    )

    # ------------------------------------------------------------------
    # RC: LOO (leave-one-out) drops from demographic + charitable block
    # ------------------------------------------------------------------
    print(f"\n--- RC: LOO drops (17 variants) ---")

    # LOO single variables
    for drop_var in loo_vars_single:
        loo_controls = [v for v in baseline_controls_formula if v != drop_var]
        res = run_wls(outcome, treatment_vars + loo_controls, white_data, weight_var='tweight', se_type='HC1')
        add_result(
            spec_id=f'rc/controls/loo/drop_{drop_var}',
            spec_tree_path='specification_tree/modules/robustness/controls.md#leave-one-out-controls-loo',
            group_id=gid,
            outcome_var=outcome,
            controls_desc=f'baseline minus {drop_var}',
            sample_desc='white respondents',
            weight_desc='tweight (survey weights)',
            se_desc='HC1',
            est_result=res,
            notes=f'LOO: dropped {drop_var} from baseline controls'
        )

    # LOO paired: age + age2
    for pair in loo_vars_paired:
        loo_controls = [v for v in baseline_controls_formula if v not in pair]
        pair_label = '+'.join(pair)
        res = run_wls(outcome, treatment_vars + loo_controls, white_data, weight_var='tweight', se_type='HC1')
        add_result(
            spec_id=f'rc/controls/loo/drop_{pair_label}',
            spec_tree_path='specification_tree/modules/robustness/controls.md#leave-one-out-controls-loo',
            group_id=gid,
            outcome_var=outcome,
            controls_desc=f'baseline minus {pair_label}',
            sample_desc='white respondents',
            weight_desc='tweight (survey weights)',
            se_desc='HC1',
            est_result=res,
            notes=f'LOO: dropped {pair_label} (paired) from baseline controls'
        )

    # ------------------------------------------------------------------
    # RC: SAMPLE VARIANTS
    # ------------------------------------------------------------------
    print(f"\n--- RC: Sample variants ---")

    # rc/sample/subpopulation/full_sample
    # Need to add race controls since now we have non-white respondents
    full_sample_controls = nraud + race_controls + demographics + charitable
    res = run_wls(outcome, treatment_vars + full_sample_controls, df, weight_var='tweight', se_type='HC1')
    add_result(
        spec_id='rc/sample/subpopulation/full_sample',
        spec_tree_path='specification_tree/modules/robustness/sample.md#subpopulation',
        group_id=gid,
        outcome_var=outcome,
        controls_desc='nraud + race + demographics + charitable',
        sample_desc='full sample (all races)',
        weight_desc='tweight (survey weights)',
        se_desc='HC1',
        est_result=res,
        notes='Full sample including black and other respondents'
    )

    # rc/sample/subpopulation/main_variant_only (surveyvariant==1, use mweight)
    main_var_data = df[(df['white'] == 1) & (df['surveyvariant'] == 1)].copy()
    res = run_wls(outcome, treatment_vars + baseline_controls_formula, main_var_data, weight_var='mweight', se_type='HC1')
    add_result(
        spec_id='rc/sample/subpopulation/main_variant_only',
        spec_tree_path='specification_tree/modules/robustness/sample.md#subpopulation',
        group_id=gid,
        outcome_var=outcome,
        controls_desc='nraud + demographics + charitable (baseline)',
        sample_desc='white, main survey variant only (surveyvariant==1)',
        weight_desc='mweight (main variant weights)',
        se_desc='HC1',
        est_result=res,
        notes='Main survey variant subsample with mweight'
    )

    # rc/sample/subpopulation/slidell_only
    slidell_data = df[(df['white'] == 1) & (df['cityslidell'] == 1)].copy()
    res = run_wls(outcome, treatment_vars + baseline_controls_formula, slidell_data, weight_var='tweight', se_type='HC1')
    add_result(
        spec_id='rc/sample/subpopulation/slidell_only',
        spec_tree_path='specification_tree/modules/robustness/sample.md#subpopulation',
        group_id=gid,
        outcome_var=outcome,
        controls_desc='nraud + demographics + charitable (baseline)',
        sample_desc='white, Slidell only (cityslidell==1)',
        weight_desc='tweight (survey weights)',
        se_desc='HC1',
        est_result=res,
        notes='Slidell subsample'
    )

    # rc/sample/subpopulation/biloxi_only
    biloxi_data = df[(df['white'] == 1) & (df['cityslidell'] == 0)].copy()
    res = run_wls(outcome, treatment_vars + baseline_controls_formula, biloxi_data, weight_var='tweight', se_type='HC1')
    add_result(
        spec_id='rc/sample/subpopulation/biloxi_only',
        spec_tree_path='specification_tree/modules/robustness/sample.md#subpopulation',
        group_id=gid,
        outcome_var=outcome,
        controls_desc='nraud + demographics + charitable (baseline)',
        sample_desc='white, Biloxi only (cityslidell==0)',
        weight_desc='tweight (survey weights)',
        se_desc='HC1',
        est_result=res,
        notes='Biloxi subsample'
    )

    # rc/sample/subpopulation/race_shown_only (picobscur==0)
    # Note: picobscur is treatment arm = 0 for this subsample, creating collinearity
    # picraceb == picshowblack when picobscur==0, so one gets dropped
    race_shown_data = df[(df['white'] == 1) & (df['picobscur'] == 0)].copy()
    res = run_wls(outcome, treatment_vars + baseline_controls_formula, race_shown_data, weight_var='tweight', se_type='HC1')
    add_result(
        spec_id='rc/sample/subpopulation/race_shown_only',
        spec_tree_path='specification_tree/modules/robustness/sample.md#subpopulation',
        group_id=gid,
        outcome_var=outcome,
        controls_desc='nraud + demographics + charitable (baseline)',
        sample_desc='white, race-shown only (picobscur==0)',
        weight_desc='tweight (survey weights)',
        se_desc='HC1',
        est_result=res,
        notes='Race-shown subsample (picobscur==0). Collinearity: picraceb==picshowblack, one treatment dummy auto-dropped.'
    )

    # ------------------------------------------------------------------
    # RC: WEIGHTS
    # ------------------------------------------------------------------
    print(f"\n--- RC: Weights ---")

    # rc/weights/main/unweighted
    res = run_wls(outcome, treatment_vars + baseline_controls_formula, white_data, weight_var=None, se_type='HC1')
    add_result(
        spec_id='rc/weights/main/unweighted',
        spec_tree_path='specification_tree/modules/robustness/weights.md#unweighted',
        group_id=gid,
        outcome_var=outcome,
        controls_desc='nraud + demographics + charitable (baseline)',
        sample_desc='white respondents',
        weight_desc='unweighted',
        se_desc='HC1',
        est_result=res,
        notes='Unweighted OLS (no survey weights)'
    )

    # ------------------------------------------------------------------
    # RC: OUTCOME PREPROCESSING (G1 and G2 only)
    # ------------------------------------------------------------------
    if gid == 'G1' and ginfo['has_outcome_preprocess']:
        print(f"\n--- RC: Outcome preprocessing (G1) ---")

        # rc/preprocess/outcome/topcode_giving_at_99
        res = run_wls('giving_tc99', treatment_vars + baseline_controls_formula, white_data, weight_var='tweight', se_type='HC1')
        add_result(
            spec_id='rc/preprocess/outcome/topcode_giving_at_99',
            spec_tree_path='specification_tree/modules/robustness/preprocessing.md#outcome-topcoding',
            group_id=gid,
            outcome_var='giving_tc99',
            controls_desc='nraud + demographics + charitable (baseline)',
            sample_desc='white respondents',
            weight_desc='tweight (survey weights)',
            se_desc='HC1',
            est_result=res,
            notes='Giving topcoded at value 99 (changes 437 obs at giving=100 to 99)'
        )

        # rc/preprocess/outcome/winsor_1_99
        res = run_wls('giving_winsor', treatment_vars + baseline_controls_formula, white_data, weight_var='tweight', se_type='HC1')
        add_result(
            spec_id='rc/preprocess/outcome/winsor_1_99',
            spec_tree_path='specification_tree/modules/robustness/preprocessing.md#outcome-winsorization',
            group_id=gid,
            outcome_var='giving_winsor',
            controls_desc='nraud + demographics + charitable (baseline)',
            sample_desc='white respondents',
            weight_desc='tweight (survey weights)',
            se_desc='HC1',
            est_result=res,
            notes='Giving winsorized: floor at 1, cap at 99 (standard percentile-based winsorization is no-op on this bounded 0-100 scale; using value-based bounds)'
        )

    if gid == 'G2' and ginfo['has_outcome_preprocess']:
        print(f"\n--- RC: Outcome preprocessing (G2) ---")

        # rc/preprocess/outcome/topcode_hypgiv_at_250
        res = run_wls('hypgiv_tc250', treatment_vars + baseline_controls_formula, white_data, weight_var='tweight', se_type='HC1')
        add_result(
            spec_id='rc/preprocess/outcome/topcode_hypgiv_at_250',
            spec_tree_path='specification_tree/modules/robustness/preprocessing.md#outcome-topcoding',
            group_id=gid,
            outcome_var='hypgiv_tc250',
            controls_desc='nraud + demographics + charitable (baseline)',
            sample_desc='white respondents',
            weight_desc='tweight (survey weights)',
            se_desc='HC1',
            est_result=res,
            notes='Hypothetical giving topcoded at $250 (more aggressive topcode)'
        )

        # rc/preprocess/outcome/no_topcode
        res = run_wls('hypgiv_notc', treatment_vars + baseline_controls_formula, white_data, weight_var='tweight', se_type='HC1')
        add_result(
            spec_id='rc/preprocess/outcome/no_topcode',
            spec_tree_path='specification_tree/modules/robustness/preprocessing.md#outcome-topcoding',
            group_id=gid,
            outcome_var='hypgiv_notc',
            controls_desc='nraud + demographics + charitable (baseline)',
            sample_desc='white respondents',
            weight_desc='tweight (survey weights)',
            se_desc='HC1',
            est_result=res,
            notes='Hypothetical giving without topcoding (raw hypothgiving)'
        )

    # ------------------------------------------------------------------
    # INFERENCE VARIANTS
    # ------------------------------------------------------------------
    print(f"\n--- Inference variants ---")

    for infer_type in ginfo['infer_variants']:
        if infer_type == 'HC2':
            se_code = 'HC2'
            spec_id = 'infer/se/hc/hc2'
            tree_path = 'specification_tree/modules/inference/standard_errors.md#hc2'
        elif infer_type == 'HC3':
            se_code = 'HC3'
            spec_id = 'infer/se/hc/hc3'
            tree_path = 'specification_tree/modules/inference/standard_errors.md#hc3'
        elif infer_type == 'nonrobust':
            se_code = 'nonrobust'
            spec_id = 'infer/se/hc/classical'
            tree_path = 'specification_tree/modules/inference/standard_errors.md#classical'
        else:
            continue

        res = run_wls(outcome, treatment_vars + baseline_controls_formula, white_data,
                      weight_var='tweight', se_type=se_code)
        add_result(
            spec_id=spec_id,
            spec_tree_path=tree_path,
            group_id=gid,
            outcome_var=outcome,
            controls_desc='nraud + demographics + charitable (baseline)',
            sample_desc='white respondents',
            weight_desc='tweight (survey weights)',
            se_desc=se_code,
            est_result=res,
            notes=f'Inference variant: {se_code} standard errors (baseline controls)'
        )


# =============================================================================
# DIAGNOSTICS: Balance check for G1
# =============================================================================
print(f"\n{'='*70}")
print("DIAGNOSTICS: Balance check for G1")
print(f"{'='*70}")

diag_results = []

# Balance of demographics across treatment (picshowblack) on white subsample
balance_vars = demographics + charitable
white_data = df[df['white'] == 1].copy()

balance_rows = []
for bvar in balance_vars:
    dfreg = white_data[['picshowblack', bvar, 'tweight']].dropna()
    y = dfreg[bvar]
    X = sm.add_constant(dfreg[['picshowblack']])
    w = dfreg['tweight']
    model = sm.WLS(y, X, weights=w)
    res = model.fit(cov_type='HC1')
    balance_rows.append({
        'variable': bvar,
        'coef_picshowblack': round(float(res.params.get('picshowblack', np.nan)), 6),
        'se': round(float(res.bse.get('picshowblack', np.nan)), 6),
        'pvalue': round(float(res.pvalues.get('picshowblack', np.nan)), 6),
        'n_obs': int(res.nobs),
    })
    print(f"  Balance {bvar}: coef={res.params.get('picshowblack', np.nan):.4f}, p={res.pvalues.get('picshowblack', np.nan):.4f}")

diag_json = {
    'type': 'balance_check',
    'treatment_var': 'picshowblack',
    'sample': 'white respondents',
    'balance_results': balance_rows,
    'n_vars_tested': len(balance_vars),
    'n_significant_005': sum(1 for r in balance_rows if r['pvalue'] < 0.05),
    'n_significant_010': sum(1 for r in balance_rows if r['pvalue'] < 0.10),
}

diag_results.append({
    'paper_id': PAPER_ID,
    'diagnostic_run_id': f'{PAPER_ID}_diag_0001',
    'diag_spec_id': 'diag/randomized_experiment/balance/covariates',
    'spec_tree_path': 'specification_tree/designs/randomized_experiment.md#balance-checks',
    'diagnostic_scope': 'baseline_group',
    'diagnostic_context_id': 'G1_white_balance',
    'diagnostic_json': json.dumps(diag_json),
})


# =============================================================================
# Write outputs
# =============================================================================
print(f"\n{'='*70}")
print("Writing outputs")
print(f"{'='*70}")

# 1) specification_results.csv
output_cols = [
    'paper_id', 'spec_run_id', 'spec_id', 'spec_tree_path', 'baseline_group_id',
    'outcome_var', 'treatment_var', 'coefficient', 'std_error', 'p_value',
    'ci_lower', 'ci_upper', 'n_obs', 'r_squared', 'coefficient_vector_json',
    'sample_desc', 'fixed_effects', 'controls_desc', 'cluster_var',
]
results_df = pd.DataFrame(all_results)
results_df[output_cols].to_csv(os.path.join(PACKAGE_DIR, 'specification_results.csv'), index=False)
print(f"Wrote {len(results_df)} rows to specification_results.csv")

# 2) diagnostics_results.csv
diag_df = pd.DataFrame(diag_results)
diag_df.to_csv(os.path.join(PACKAGE_DIR, 'diagnostics_results.csv'), index=False)
print(f"Wrote {len(diag_df)} rows to diagnostics_results.csv")

# 3) spec_diagnostics_map.csv
# Link all G1 specs to the balance diagnostic
diag_map_rows = []
for _, row in results_df[results_df['baseline_group_id'] == 'G1'].iterrows():
    diag_map_rows.append({
        'paper_id': PAPER_ID,
        'spec_run_id': row['spec_run_id'],
        'diagnostic_run_id': f'{PAPER_ID}_diag_0001',
        'relationship': 'shared_invariant_check',
    })
diag_map_df = pd.DataFrame(diag_map_rows)
diag_map_df.to_csv(os.path.join(PACKAGE_DIR, 'spec_diagnostics_map.csv'), index=False)
print(f"Wrote {len(diag_map_df)} rows to spec_diagnostics_map.csv")

# Summary
print(f"\n--- Summary ---")
for gid in ['G1', 'G2', 'G3', 'G4']:
    n = (results_df['baseline_group_id'] == gid).sum()
    print(f"  {gid}: {n} specs")
print(f"  Total: {len(results_df)} specs")

# Check spec_run_id uniqueness
assert results_df['spec_run_id'].is_unique, "spec_run_id is not unique!"
print("\nAll spec_run_ids are unique.")

# Validate coefficient_vector_json
for _, row in results_df.iterrows():
    try:
        json.loads(row['coefficient_vector_json'])
    except:
        print(f"WARNING: Invalid JSON in {row['spec_run_id']}")
print("All coefficient_vector_json entries are valid JSON.")
