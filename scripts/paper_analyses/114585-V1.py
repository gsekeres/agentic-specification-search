"""
Specification Search: 114585-V1
Bachmann, Berg, and Sims (2014)
"Inflation Expectations and Readiness to Spend: Cross-Sectional Evidence"
AER: P&P

Method: Ordered probit (discrete choice)
DV: DUR (readiness to spend on durables, {-1, 0, 1})
Treatment: PX1 (1-year inflation expectations) with ZLB interaction
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from scipy import stats
import json
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================================
# PATHS
# ============================================================
BASE_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search"
PKG_DIR = os.path.join(BASE_DIR, "data/downloads/extracted/114585-V1")
DATA_PATH = os.path.join(PKG_DIR, "replication/SurveysOfConsumers.dta")

# ============================================================
# LOAD DATA
# ============================================================
print("Loading data...")
df_raw = pd.read_stata(DATA_PATH)
print(f"Raw data shape: {df_raw.shape}")

# ============================================================
# VARIABLE DEFINITIONS
# ============================================================
# Treatment variable
TREATMENT_VAR = 'PX1'

# Outcome variable
OUTCOME_VAR = 'DUR'

# Interaction: ZLB * PX1
# The paper's model is: oprobit DUR PX1 ZLB#c.PX1 ZLB [controls] i.mm
# ZLB#c.PX1 means ZLB interacted with continuous PX1

# Idiosyncratic expectations controls
idio_controls = ['PEXP', 'RINC', 'RATEX', 'BUS12', 'BUS5', 'UNEMP', 'PAGO', 'GOVT']

# Aggregate controls
agg_controls = ['BUS12AG', 'PX1DISP', 'VXO', 'FFR', 'UNRATE', 'INFLATION', 'INFLVOLA', 'CPIDURABLES']

# Demographic controls
demo_controls = ['SEX', 'MARRY', 'ECLGRD', 'AFRICAN', 'HISPANIC', 'NATIVE', 'ASIAN',
                 'WEST', 'NORTHEAST', 'SOUTH', 'FAMSIZE', 'AGE', 'INCOME']

# AGE polynomial terms (AGE^2, AGE^3)
# We'll add these manually

# Month fixed effects: i.mm

# Full baseline controls
baseline_controls = idio_controls + agg_controls + demo_controls

PAPER_ID = "114585-V1"
PAPER_TITLE = "Inflation Expectations and Readiness to Spend: Cross-Sectional Evidence"
JOURNAL = "AER: P&P"

results = []

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def prepare_baseline_data(df, extra_vars=None, sample_filter=None):
    """Prepare data with baseline sample restrictions and variable transformations."""
    # Baseline sample: 1984-2012, first interview only
    mask = (df['yyyymm'] >= 198401) & (df['yyyymm'] <= 201212) & (df['idprev'].isna())
    if sample_filter is not None:
        mask = mask & sample_filter

    dfs = df[mask].copy()

    # Add interaction term
    dfs['ZLB_PX1'] = dfs['ZLB'] * dfs['PX1']

    # Add AGE polynomial terms
    dfs['AGE2'] = dfs['AGE'] ** 2
    dfs['AGE3'] = dfs['AGE'] ** 3

    # Month dummies
    dfs['mm_int'] = dfs['mm'].astype(int)

    # Required vars
    req_vars = ['DUR', 'PX1', 'ZLB', 'ZLB_PX1'] + baseline_controls + ['AGE2', 'AGE3', 'mm_int']
    if extra_vars:
        req_vars = req_vars + [v for v in extra_vars if v not in req_vars]

    dfs = dfs.dropna(subset=[v for v in req_vars if v in dfs.columns])

    return dfs


def run_oprobit(dfs, outcome_var, treatment_var, controls, include_month_fe=True,
                include_zlb_interaction=True, include_age_poly=True):
    """Run ordered probit and return results dict."""

    # Build X matrix
    X_vars = [treatment_var]
    if include_zlb_interaction and 'ZLB' in dfs.columns:
        if treatment_var == 'PX1':
            X_vars += ['ZLB_PX1', 'ZLB']
        elif treatment_var == 'PX5':
            dfs = dfs.copy()
            dfs['ZLB_PX5'] = dfs['ZLB'] * dfs['PX5']
            X_vars += ['ZLB_PX5', 'ZLB']
        elif treatment_var == 'DPX1':
            dfs = dfs.copy()
            dfs['ZLB_DPX1'] = dfs['ZLB'] * dfs['DPX1']
            X_vars += ['ZLB_DPX1', 'ZLB']
        else:
            X_vars += ['ZLB']

    X_vars += [c for c in controls if c not in X_vars]

    if include_age_poly and 'AGE' in controls:
        X_vars += ['AGE2', 'AGE3']

    if include_month_fe:
        month_dummies = pd.get_dummies(dfs['mm_int'], prefix='month', drop_first=True, dtype=float)
        X_mat = pd.concat([dfs[X_vars].reset_index(drop=True), month_dummies.reset_index(drop=True)], axis=1)
    else:
        X_mat = dfs[X_vars].copy()

    y = dfs[outcome_var].values

    # Map DUR from {-1, 0, 1} to {0, 1, 2} for OrderedModel
    if outcome_var == 'DUR' or outcome_var == 'DDUR':
        y_mapped = y + 1  # {-1,0,1} -> {0,1,2}
    elif outcome_var == 'CAR' or outcome_var == 'HOM':
        y_mapped = y + 1
    else:
        y_mapped = y
    y_mapped = y_mapped.astype(int)

    try:
        model = OrderedModel(y_mapped, X_mat.values, distr='probit')
        res = model.fit(method='bfgs', maxiter=5000, disp=False)

        # Extract treatment coefficient
        treat_idx = X_vars.index(treatment_var)
        coef = res.params[treat_idx]
        se = res.bse[treat_idx]
        z = coef / se if se > 0 else np.nan
        pval = 2 * (1 - stats.norm.cdf(abs(z)))
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se

        # Build coefficient vector JSON
        coef_vec = {
            "treatment": {"var": treatment_var, "coef": float(coef), "se": float(se), "pval": float(pval)},
            "controls": [],
            "diagnostics": {
                "pseudo_r2_mcfadden": float(res.prsquared) if hasattr(res, 'prsquared') else None,
                "ll_model": float(res.llf),
                "ll_null": float(res.llnull) if hasattr(res, 'llnull') else None,
                "aic": float(res.aic) if hasattr(res, 'aic') else None,
                "bic": float(res.bic) if hasattr(res, 'bic') else None,
            },
            "n_obs": int(len(y)),
        }

        # Add control coefficients (just the named ones, not month dummies)
        for i, var in enumerate(X_vars):
            if var != treatment_var and i < len(res.params):
                coef_vec["controls"].append({
                    "var": var,
                    "coef": float(res.params[i]),
                    "se": float(res.bse[i]) if i < len(res.bse) else None,
                })

        return {
            'coefficient': float(coef),
            'std_error': float(se),
            't_stat': float(z),
            'p_value': float(pval),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_obs': int(len(y)),
            'r_squared': float(res.prsquared) if hasattr(res, 'prsquared') else None,
            'coefficient_vector_json': json.dumps(coef_vec),
            'converged': True,
        }
    except Exception as e:
        return {
            'coefficient': np.nan,
            'std_error': np.nan,
            't_stat': np.nan,
            'p_value': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n_obs': 0,
            'r_squared': np.nan,
            'coefficient_vector_json': json.dumps({"error": str(e)}),
            'converged': False,
        }


def add_result(spec_id, spec_tree_path, outcome_var, treatment_var, res_dict,
               sample_desc="", fixed_effects="", controls_desc="", cluster_var="none",
               model_type="ordered_probit"):
    """Add a result to the results list."""
    results.append({
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'outcome_var': outcome_var,
        'treatment_var': treatment_var,
        'coefficient': res_dict['coefficient'],
        'std_error': res_dict['std_error'],
        't_stat': res_dict['t_stat'],
        'p_value': res_dict['p_value'],
        'ci_lower': res_dict['ci_lower'],
        'ci_upper': res_dict['ci_upper'],
        'n_obs': res_dict['n_obs'],
        'r_squared': res_dict['r_squared'],
        'coefficient_vector_json': res_dict['coefficient_vector_json'],
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
    })


# ============================================================
# SPEC 1: BASELINE - Tables 1 & 2
# ============================================================
print("\n=== BASELINE ===")
df_base = prepare_baseline_data(df_raw)
print(f"Baseline sample: N={len(df_base)}")

res = run_oprobit(df_base, 'DUR', 'PX1', baseline_controls)
add_result('baseline', 'methods/discrete_choice.md#baseline', 'DUR', 'PX1', res,
           sample_desc='1984-2012, first interview', fixed_effects='month_of_year',
           controls_desc='idiosyncratic + aggregate + demographics')
print(f"  PX1 coef={res['coefficient']:.6f}, se={res['std_error']:.6f}, p={res['p_value']:.4f}, N={res['n_obs']}")


# ============================================================
# SPEC 2-3: MODEL TYPE VARIATIONS (Table 3 analogs)
# ============================================================
print("\n=== MODEL TYPE VARIATIONS ===")

# Ordered logit
print("  Ordered logit...")
try:
    y_bl = df_base['DUR'].values + 1
    X_vars_bl = ['PX1', 'ZLB_PX1', 'ZLB'] + baseline_controls + ['AGE2', 'AGE3']
    month_dummies_bl = pd.get_dummies(df_base['mm_int'], prefix='month', drop_first=True, dtype=float)
    X_bl = pd.concat([df_base[X_vars_bl].reset_index(drop=True), month_dummies_bl.reset_index(drop=True)], axis=1)

    model_ologit = OrderedModel(y_bl.astype(int), X_bl.values, distr='logit')
    res_ologit = model_ologit.fit(method='bfgs', maxiter=5000, disp=False)
    coef_ol = res_ologit.params[0]
    se_ol = res_ologit.bse[0]
    z_ol = coef_ol / se_ol
    pval_ol = 2 * (1 - stats.norm.cdf(abs(z_ol)))

    res_dict = {
        'coefficient': float(coef_ol), 'std_error': float(se_ol), 't_stat': float(z_ol),
        'p_value': float(pval_ol), 'ci_lower': float(coef_ol - 1.96*se_ol),
        'ci_upper': float(coef_ol + 1.96*se_ol), 'n_obs': len(y_bl),
        'r_squared': float(res_ologit.prsquared) if hasattr(res_ologit, 'prsquared') else None,
        'coefficient_vector_json': json.dumps({"treatment": {"var": "PX1", "coef": float(coef_ol), "se": float(se_ol), "pval": float(pval_ol)}}),
        'converged': True
    }
    add_result('discrete/multi/ordered_logit', 'methods/discrete_choice.md#model-type-multinomial-outcome',
               'DUR', 'PX1', res_dict, sample_desc='1984-2012, first interview',
               fixed_effects='month_of_year', controls_desc='full baseline', model_type='ordered_logit')
    print(f"    coef={coef_ol:.6f}, p={pval_ol:.4f}")
except Exception as e:
    print(f"    FAILED: {e}")

# Linear probability model (OLS on DUR)
print("  Linear probability model...")
try:
    X_lpm = sm.add_constant(X_bl)
    model_lpm = sm.OLS(df_base['DUR'].values, X_lpm.values).fit()
    coef_lpm = model_lpm.params[1]  # PX1 is first var after constant
    se_lpm = model_lpm.bse[1]
    pval_lpm = model_lpm.pvalues[1]

    res_dict = {
        'coefficient': float(coef_lpm), 'std_error': float(se_lpm),
        't_stat': float(model_lpm.tvalues[1]),
        'p_value': float(pval_lpm), 'ci_lower': float(coef_lpm - 1.96*se_lpm),
        'ci_upper': float(coef_lpm + 1.96*se_lpm), 'n_obs': int(model_lpm.nobs),
        'r_squared': float(model_lpm.rsquared),
        'coefficient_vector_json': json.dumps({"treatment": {"var": "PX1", "coef": float(coef_lpm), "se": float(se_lpm), "pval": float(pval_lpm)}}),
        'converged': True
    }
    add_result('discrete/binary/lpm', 'methods/discrete_choice.md#model-type-binary-outcome',
               'DUR', 'PX1', res_dict, sample_desc='1984-2012, first interview',
               fixed_effects='month_of_year', controls_desc='full baseline', model_type='OLS_LPM')
    print(f"    coef={coef_lpm:.6f}, p={pval_lpm:.4f}")
except Exception as e:
    print(f"    FAILED: {e}")


# ============================================================
# SPEC 4-7: CONTROL VARIATIONS (Table 3 analogs)
# ============================================================
print("\n=== CONTROL VARIATIONS ===")

# (4) Without idiosyncratic expectations (Table 3 col 2)
print("  Without idiosyncratic expectations...")
controls_no_idio = agg_controls + demo_controls
res = run_oprobit(df_base, 'DUR', 'PX1', controls_no_idio)
add_result('robust/control/no_idiosyncratic', 'robustness/control_progression.md',
           'DUR', 'PX1', res, sample_desc='1984-2012, first interview',
           controls_desc='aggregate + demographics only')
print(f"    coef={res['coefficient']:.6f}, p={res['p_value']:.4f}")

# (5) Without aggregate controls
print("  Without aggregate controls...")
controls_no_agg = idio_controls + demo_controls
res = run_oprobit(df_base, 'DUR', 'PX1', controls_no_agg)
add_result('robust/control/no_aggregate', 'robustness/control_progression.md',
           'DUR', 'PX1', res, sample_desc='1984-2012, first interview',
           controls_desc='idiosyncratic + demographics only')
print(f"    coef={res['coefficient']:.6f}, p={res['p_value']:.4f}")

# (6) Without demographic controls
print("  Without demographic controls...")
controls_no_demo = idio_controls + agg_controls
res = run_oprobit(df_base, 'DUR', 'PX1', controls_no_demo, include_age_poly=False)
add_result('robust/control/no_demographics', 'robustness/control_progression.md',
           'DUR', 'PX1', res, sample_desc='1984-2012, first interview',
           controls_desc='idiosyncratic + aggregate only')
print(f"    coef={res['coefficient']:.6f}, p={res['p_value']:.4f}")

# (7) No controls at all (bivariate + ZLB interaction)
print("  No controls (bivariate)...")
res = run_oprobit(df_base, 'DUR', 'PX1', [], include_age_poly=False)
add_result('robust/control/none', 'robustness/control_progression.md',
           'DUR', 'PX1', res, sample_desc='1984-2012, first interview',
           controls_desc='none')
print(f"    coef={res['coefficient']:.6f}, p={res['p_value']:.4f}")


# ============================================================
# SPEC 8-17: LEAVE-ONE-OUT (drop each control group one at a time)
# ============================================================
print("\n=== LEAVE-ONE-OUT ===")

loo_groups = {
    'PEXP': ['PEXP'], 'RINC': ['RINC'], 'RATEX': ['RATEX'],
    'BUS12': ['BUS12'], 'BUS5': ['BUS5'], 'UNEMP': ['UNEMP'],
    'PAGO': ['PAGO'], 'GOVT': ['GOVT'],
    'macro_agg': ['BUS12AG', 'PX1DISP', 'VXO'],
    'policy_agg': ['FFR', 'UNRATE', 'INFLATION', 'INFLVOLA', 'CPIDURABLES'],
}

for group_name, vars_to_drop in loo_groups.items():
    remaining = [c for c in baseline_controls if c not in vars_to_drop]
    res = run_oprobit(df_base, 'DUR', 'PX1', remaining)
    add_result(f'robust/loo/drop_{group_name}', 'robustness/leave_one_out.md',
               'DUR', 'PX1', res, sample_desc='1984-2012, first interview',
               controls_desc=f'baseline minus {group_name}')
    print(f"  drop {group_name}: coef={res['coefficient']:.6f}, p={res['p_value']:.4f}")


# ============================================================
# SPEC 18-22: CONTROL PROGRESSION (add incrementally)
# ============================================================
print("\n=== CONTROL PROGRESSION ===")

control_sets = [
    ('only_idio', idio_controls),
    ('idio_plus_macro', idio_controls + ['BUS12AG', 'PX1DISP', 'VXO']),
    ('idio_plus_all_agg', idio_controls + agg_controls),
    ('idio_agg_plus_basic_demo', idio_controls + agg_controls + ['SEX', 'MARRY', 'ECLGRD', 'INCOME']),
    ('full', baseline_controls),
]

for set_name, ctrls in control_sets:
    inc_age = 'AGE' in ctrls
    res = run_oprobit(df_base, 'DUR', 'PX1', ctrls, include_age_poly=inc_age)
    add_result(f'robust/control/add_{set_name}', 'robustness/control_progression.md',
               'DUR', 'PX1', res, sample_desc='1984-2012, first interview',
               controls_desc=set_name)
    print(f"  {set_name}: coef={res['coefficient']:.6f}, p={res['p_value']:.4f}")


# ============================================================
# SPEC 23-24: WITHOUT ZLB INTERACTION / WITHOUT ZLB
# ============================================================
print("\n=== ZLB INTERACTION VARIATIONS ===")

# Without ZLB interaction
print("  No ZLB interaction...")
res = run_oprobit(df_base, 'DUR', 'PX1', baseline_controls, include_zlb_interaction=False)
add_result('robust/estimation/no_zlb_interaction', 'robustness/model_specification.md',
           'DUR', 'PX1', res, sample_desc='1984-2012, first interview',
           controls_desc='full baseline, no ZLB interaction')
print(f"    coef={res['coefficient']:.6f}, p={res['p_value']:.4f}")

# Without month FE (Table 3 variant)
print("  No month FE...")
res = run_oprobit(df_base, 'DUR', 'PX1', baseline_controls, include_month_fe=False)
add_result('robust/estimation/no_month_fe', 'robustness/model_specification.md',
           'DUR', 'PX1', res, sample_desc='1984-2012, first interview',
           controls_desc='full baseline, no month FE', fixed_effects='none')
print(f"    coef={res['coefficient']:.6f}, p={res['p_value']:.4f}")


# ============================================================
# SPEC 25-26: MONTH FE VARIATIONS (Table 3)
# ============================================================
print("\n=== MONTH FE: YEAR-MONTH FE ===")
# Year-month FE (Table 3, col 4) - use i.yyyymm instead of i.mm
# This absorbs ZLB and all aggregate controls
try:
    dfs_ym = df_base.copy()
    # With year-month FE, we drop aggregate controls (absorbed)
    controls_ym = idio_controls + demo_controls
    X_vars_ym = ['PX1'] + controls_ym + ['AGE2', 'AGE3']
    ym_dummies = pd.get_dummies(dfs_ym['yyyymm'], prefix='ym', drop_first=True, dtype=float)
    X_ym = pd.concat([dfs_ym[X_vars_ym].reset_index(drop=True), ym_dummies.reset_index(drop=True)], axis=1)
    y_ym = (dfs_ym['DUR'].values + 1).astype(int)

    model_ym = OrderedModel(y_ym, X_ym.values, distr='probit')
    res_ym = model_ym.fit(method='bfgs', maxiter=5000, disp=False)
    coef_ym = res_ym.params[0]
    se_ym = res_ym.bse[0]
    z_ym = coef_ym / se_ym
    pval_ym = 2 * (1 - stats.norm.cdf(abs(z_ym)))

    res_dict = {
        'coefficient': float(coef_ym), 'std_error': float(se_ym), 't_stat': float(z_ym),
        'p_value': float(pval_ym), 'ci_lower': float(coef_ym - 1.96*se_ym),
        'ci_upper': float(coef_ym + 1.96*se_ym), 'n_obs': len(y_ym),
        'r_squared': float(res_ym.prsquared) if hasattr(res_ym, 'prsquared') else None,
        'coefficient_vector_json': json.dumps({"treatment": {"var": "PX1", "coef": float(coef_ym), "se": float(se_ym), "pval": float(pval_ym)}}),
        'converged': True
    }
    add_result('robust/estimation/yearmonth_fe', 'robustness/model_specification.md',
               'DUR', 'PX1', res_dict, sample_desc='1984-2012, first interview',
               fixed_effects='year_month', controls_desc='idio + demo (agg absorbed)')
    print(f"  Year-month FE: coef={coef_ym:.6f}, p={pval_ym:.4f}")
except Exception as e:
    print(f"  FAILED: {e}")


# ============================================================
# SPEC 27-28: ALTERNATIVE TREATMENT VARIABLES
# ============================================================
print("\n=== ALTERNATIVE TREATMENTS ===")

# 5-year inflation expectations (Table 3 col 5 / Table 9)
print("  PX5 (5-year inflation expectations)...")
mask_px5 = (df_raw['yyyymm'] >= 199004) & (df_raw['yyyymm'] <= 201212) & (df_raw['idprev'].isna())
df_px5 = df_raw[mask_px5].copy()
df_px5['ZLB_PX1'] = df_px5['ZLB'] * df_px5['PX1']
df_px5['ZLB_PX5'] = df_px5['ZLB'] * df_px5['PX5']
df_px5['AGE2'] = df_px5['AGE'] ** 2
df_px5['AGE3'] = df_px5['AGE'] ** 3
df_px5['mm_int'] = df_px5['mm'].astype(int)
extra_ctrls_px5 = baseline_controls + ['HOMEOWN']
df_px5 = df_px5.dropna(subset=['DUR', 'PX5', 'ZLB', 'ZLB_PX5'] + extra_ctrls_px5 + ['AGE2', 'AGE3', 'mm_int'])

res = run_oprobit(df_px5, 'DUR', 'PX5', extra_ctrls_px5)
add_result('robust/treatment/px5', 'robustness/measurement.md',
           'DUR', 'PX5', res, sample_desc='1990-2012, first interview',
           controls_desc='baseline + HOMEOWN')
print(f"    coef={res['coefficient']:.6f}, p={res['p_value']:.4f}, N={res['n_obs']}")

# Change in 1Y inflation expectations (Table 4)
print("  DPX1 (change in inflation expectations)...")
mask_dpx1 = (df_raw['yyyymm'] >= 198401) & (df_raw['yyyymm'] <= 201212) & (df_raw['second'] == 1)
df_dpx1 = df_raw[mask_dpx1].copy()
df_dpx1['ZLB_DPX1'] = df_dpx1['ZLB'] * df_dpx1['DPX1']
df_dpx1['ZLB_PX1'] = df_dpx1['ZLB'] * df_dpx1['PX1']
df_dpx1['AGE2'] = df_dpx1['AGE'] ** 2
df_dpx1['AGE3'] = df_dpx1['AGE'] ** 3
df_dpx1['mm_int'] = df_dpx1['mm'].astype(int)
df_dpx1 = df_dpx1.dropna(subset=['DUR', 'DPX1', 'ZLB', 'ZLB_DPX1'] + baseline_controls + ['AGE2', 'AGE3', 'mm_int'])

res = run_oprobit(df_dpx1, 'DUR', 'DPX1', baseline_controls)
add_result('robust/treatment/dpx1_level', 'robustness/measurement.md',
           'DUR', 'DPX1', res, sample_desc='1984-2012, second interview',
           controls_desc='full baseline')
print(f"    coef={res['coefficient']:.6f}, p={res['p_value']:.4f}, N={res['n_obs']}")


# ============================================================
# SPEC 29-30: ALTERNATIVE OUTCOMES (Tables 9, 10)
# ============================================================
print("\n=== ALTERNATIVE OUTCOMES ===")

# Change in durables readiness (Table 4 col 2)
print("  DDUR (change in durables readiness)...")
df_ddur = df_dpx1.dropna(subset=['DDUR'])
res = run_oprobit(df_ddur, 'DDUR', 'DPX1', baseline_controls)
add_result('robust/outcome/ddur_change', 'robustness/measurement.md',
           'DDUR', 'DPX1', res, sample_desc='1984-2012, second interview',
           controls_desc='full baseline')
print(f"    coef={res['coefficient']:.6f}, p={res['p_value']:.4f}, N={res['n_obs']}")

# Cars (Table 9)
print("  CAR (readiness to spend on cars)...")
mask_car = (df_raw['yyyymm'] >= 199004) & (df_raw['yyyymm'] <= 201101) & (df_raw['idprev'].isna())
df_car = df_raw[mask_car].copy()
df_car['ZLB_PX5'] = df_car['ZLB'] * df_car['PX5']
df_car['ZLB_PX1'] = df_car['ZLB'] * df_car['PX1']
df_car['AGE2'] = df_car['AGE'] ** 2
df_car['AGE3'] = df_car['AGE'] ** 3
df_car['mm_int'] = df_car['mm'].astype(int)
car_controls = baseline_controls + ['HOMEOWN', 'CARLOAN']
# Use PX5 for car/house (as in original)
df_car = df_car.dropna(subset=['CAR', 'PX5', 'ZLB'] + car_controls + ['AGE2', 'AGE3', 'mm_int', 'GAS5'])
car_controls_full = car_controls + ['GAS5']
res = run_oprobit(df_car, 'CAR', 'PX5', car_controls_full)
add_result('robust/outcome/car_px5', 'robustness/measurement.md',
           'CAR', 'PX5', res, sample_desc='1990-2011, first interview',
           controls_desc='baseline + HOMEOWN + CARLOAN + GAS5')
print(f"    coef={res['coefficient']:.6f}, p={res['p_value']:.4f}, N={res['n_obs']}")

# Houses (Table 10)
print("  HOM (readiness to spend on houses)...")
mask_hom = (df_raw['yyyymm'] >= 198701) & (df_raw['yyyymm'] <= 201212) & (df_raw['idprev'].isna())
df_hom = df_raw[mask_hom].copy()
df_hom['ZLB_PX1'] = df_hom['ZLB'] * df_hom['PX1']
df_hom['AGE2'] = df_hom['AGE'] ** 2
df_hom['AGE3'] = df_hom['AGE'] ** 3
df_hom['mm_int'] = df_hom['mm'].astype(int)
hom_controls = baseline_controls + ['MORTG', 'SHILLER']
df_hom = df_hom.dropna(subset=['HOM', 'PX1', 'ZLB'] + hom_controls + ['AGE2', 'AGE3', 'mm_int'])
res = run_oprobit(df_hom, 'HOM', 'PX1', hom_controls)
add_result('robust/outcome/house_px1', 'robustness/measurement.md',
           'HOM', 'PX1', res, sample_desc='1987-2012, first interview',
           controls_desc='baseline + MORTG + SHILLER')
print(f"    coef={res['coefficient']:.6f}, p={res['p_value']:.4f}, N={res['n_obs']}")


# ============================================================
# SPEC 31-38: SAMPLE RESTRICTIONS - AGE, EDUCATION, INCOME (Table 5)
# ============================================================
print("\n=== SAMPLE RESTRICTIONS: HETEROGENEITY (Table 5) ===")

het_samples = {
    'age_ge48': df_raw['AGE'] >= 48,
    'age_lt48': df_raw['AGE'] < 48,
    'college': df_raw['ECLGRD'] == 1,
    'no_college': df_raw['ECLGRD'] == 0,
    'top20_income': df_raw['ytl5'] == 5,
    'bot20_income': df_raw['ytl5'] == 1,
    'male': df_raw['SEX'] == 1,
    'female': df_raw['SEX'] == 0,
}

for sample_name, sample_mask in het_samples.items():
    dfs = prepare_baseline_data(df_raw, sample_filter=sample_mask)
    if len(dfs) < 100:
        print(f"  {sample_name}: SKIPPED (N={len(dfs)})")
        continue
    # For college/no_college, drop ECLGRD from controls
    ctrls = baseline_controls.copy()
    if 'college' in sample_name:
        ctrls = [c for c in ctrls if c != 'ECLGRD']
    if 'male' in sample_name or 'female' in sample_name:
        ctrls = [c for c in ctrls if c != 'SEX']
    res = run_oprobit(dfs, 'DUR', 'PX1', ctrls)
    add_result(f'robust/sample/{sample_name}', 'robustness/sample_restrictions.md',
               'DUR', 'PX1', res, sample_desc=f'1984-2012, first interview, {sample_name}',
               controls_desc='baseline (adapted)')
    print(f"  {sample_name}: coef={res['coefficient']:.6f}, p={res['p_value']:.4f}, N={res['n_obs']}")


# ============================================================
# SPEC 39-44: BIRTH COHORT SUBSAMPLES (Table 6)
# ============================================================
print("\n=== BIRTH COHORT SUBSAMPLES (Table 6) ===")

cohort_samples = {
    'born_pre1930': df_raw['BIRTHY'] < 1930,
    'born_1930_1949': (df_raw['BIRTHY'] >= 1930) & (df_raw['BIRTHY'] < 1950),
    'born_1950_1959': (df_raw['BIRTHY'] >= 1950) & (df_raw['BIRTHY'] < 1960),
    'born_1960_1969': (df_raw['BIRTHY'] >= 1960) & (df_raw['BIRTHY'] < 1970),
    'born_post1970': df_raw['BIRTHY'] >= 1970,
}

for cohort_name, cohort_mask in cohort_samples.items():
    dfs = prepare_baseline_data(df_raw, sample_filter=cohort_mask)
    if len(dfs) < 100:
        print(f"  {cohort_name}: SKIPPED (N={len(dfs)})")
        continue
    res = run_oprobit(dfs, 'DUR', 'PX1', baseline_controls)
    add_result(f'robust/sample/cohort_{cohort_name}', 'robustness/sample_restrictions.md',
               'DUR', 'PX1', res, sample_desc=f'1984-2012, first interview, {cohort_name}',
               controls_desc='full baseline')
    print(f"  {cohort_name}: coef={res['coefficient']:.6f}, p={res['p_value']:.4f}, N={res['n_obs']}")


# ============================================================
# SPEC 45-48: INFLATION EXPECTATION QUARTILE SUBSAMPLES (Table 7)
# ============================================================
print("\n=== INFLATION QUARTILE SUBSAMPLES (Table 7) ===")

for q in [1, 2, 3, 4]:
    mask_q = df_raw['INFLQRTL'] == q
    dfs = prepare_baseline_data(df_raw, sample_filter=mask_q)
    if len(dfs) < 100:
        print(f"  quartile {q}: SKIPPED (N={len(dfs)})")
        continue
    res = run_oprobit(dfs, 'DUR', 'PX1', baseline_controls)
    add_result(f'robust/sample/inflqrtl_{q}', 'robustness/sample_restrictions.md',
               'DUR', 'PX1', res, sample_desc=f'1984-2012, first interview, inflation quartile {q}',
               controls_desc='full baseline')
    print(f"  quartile {q}: coef={res['coefficient']:.6f}, p={res['p_value']:.4f}, N={res['n_obs']}")


# ============================================================
# SPEC 49-53: TIME PERIOD RESTRICTIONS
# ============================================================
print("\n=== TIME PERIOD RESTRICTIONS ===")

time_samples = {
    'pre_2000': (df_raw['yyyymm'] >= 198401) & (df_raw['yyyymm'] <= 199912),
    'post_2000': (df_raw['yyyymm'] >= 200001) & (df_raw['yyyymm'] <= 201212),
    'pre_zlb': (df_raw['yyyymm'] >= 198401) & (df_raw['yyyymm'] <= 200811),
    'zlb_period': (df_raw['yyyymm'] >= 200812) & (df_raw['yyyymm'] <= 201212),
    'great_mod': (df_raw['yyyymm'] >= 198401) & (df_raw['yyyymm'] <= 200712),
}

for time_name, time_mask in time_samples.items():
    dfs = prepare_baseline_data(df_raw, sample_filter=time_mask)
    if len(dfs) < 100:
        print(f"  {time_name}: SKIPPED (N={len(dfs)})")
        continue
    res = run_oprobit(dfs, 'DUR', 'PX1', baseline_controls)
    add_result(f'robust/sample/time_{time_name}', 'robustness/sample_restrictions.md',
               'DUR', 'PX1', res, sample_desc=f'{time_name}',
               controls_desc='full baseline')
    print(f"  {time_name}: coef={res['coefficient']:.6f}, p={res['p_value']:.4f}, N={res['n_obs']}")


# ============================================================
# SPEC 54-58: REGIONAL SUBSAMPLES
# ============================================================
print("\n=== REGIONAL SUBSAMPLES ===")

region_samples = {
    'west_only': df_raw['WEST'] == 1,
    'northeast_only': df_raw['NORTHEAST'] == 1,
    'south_only': df_raw['SOUTH'] == 1,
    'midwest_only': (df_raw['WEST'] == 0) & (df_raw['NORTHEAST'] == 0) & (df_raw['SOUTH'] == 0),
}

for reg_name, reg_mask in region_samples.items():
    ctrls = [c for c in baseline_controls if c not in ['WEST', 'NORTHEAST', 'SOUTH']]
    dfs = prepare_baseline_data(df_raw, sample_filter=reg_mask)
    if len(dfs) < 100:
        print(f"  {reg_name}: SKIPPED (N={len(dfs)})")
        continue
    res = run_oprobit(dfs, 'DUR', 'PX1', ctrls)
    add_result(f'robust/sample/region_{reg_name}', 'robustness/sample_restrictions.md',
               'DUR', 'PX1', res, sample_desc=f'1984-2012, first interview, {reg_name}',
               controls_desc='baseline minus region dummies')
    print(f"  {reg_name}: coef={res['coefficient']:.6f}, p={res['p_value']:.4f}, N={res['n_obs']}")


# ============================================================
# SPEC 59-61: INFERENCE ALTERNATIVES
# ============================================================
print("\n=== INFERENCE ALTERNATIVES ===")

# Robust (sandwich) standard errors
print("  Robust SEs...")
try:
    y_rb = (df_base['DUR'].values + 1).astype(int)
    X_vars_rb = ['PX1', 'ZLB_PX1', 'ZLB'] + baseline_controls + ['AGE2', 'AGE3']
    month_dummies_rb = pd.get_dummies(df_base['mm_int'], prefix='month', drop_first=True, dtype=float)
    X_rb = pd.concat([df_base[X_vars_rb].reset_index(drop=True), month_dummies_rb.reset_index(drop=True)], axis=1)

    model_rb = OrderedModel(y_rb, X_rb.values, distr='probit')
    res_rb = model_rb.fit(method='bfgs', maxiter=5000, disp=False, cov_type='HC1')

    coef_rb = res_rb.params[0]
    se_rb = res_rb.bse[0]
    z_rb = coef_rb / se_rb
    pval_rb = 2 * (1 - stats.norm.cdf(abs(z_rb)))

    res_dict = {
        'coefficient': float(coef_rb), 'std_error': float(se_rb), 't_stat': float(z_rb),
        'p_value': float(pval_rb), 'ci_lower': float(coef_rb - 1.96*se_rb),
        'ci_upper': float(coef_rb + 1.96*se_rb), 'n_obs': len(y_rb),
        'r_squared': float(res_rb.prsquared) if hasattr(res_rb, 'prsquared') else None,
        'coefficient_vector_json': json.dumps({"treatment": {"var": "PX1", "coef": float(coef_rb), "se": float(se_rb), "pval": float(pval_rb)}}),
        'converged': True
    }
    add_result('robust/cluster/robust_hc1', 'robustness/inference_alternatives.md',
               'DUR', 'PX1', res_dict, sample_desc='1984-2012, first interview',
               controls_desc='full baseline', cluster_var='robust_HC1')
    print(f"    coef={coef_rb:.6f}, se={se_rb:.6f}, p={pval_rb:.4f}")
except Exception as e:
    print(f"    FAILED: {e}")

# Clustered by year-month
print("  Clustered by yyyymm...")
try:
    groups = df_base['yyyymm'].values
    res_cl = model_rb.fit(method='bfgs', maxiter=5000, disp=False,
                          cov_type='cluster', cov_kwds={'groups': groups})
    coef_cl = res_cl.params[0]
    se_cl = res_cl.bse[0]
    z_cl = coef_cl / se_cl
    pval_cl = 2 * (1 - stats.norm.cdf(abs(z_cl)))

    res_dict = {
        'coefficient': float(coef_cl), 'std_error': float(se_cl), 't_stat': float(z_cl),
        'p_value': float(pval_cl), 'ci_lower': float(coef_cl - 1.96*se_cl),
        'ci_upper': float(coef_cl + 1.96*se_cl), 'n_obs': len(y_rb),
        'r_squared': float(res_cl.prsquared) if hasattr(res_cl, 'prsquared') else None,
        'coefficient_vector_json': json.dumps({"treatment": {"var": "PX1", "coef": float(coef_cl), "se": float(se_cl), "pval": float(pval_cl)}}),
        'converged': True
    }
    add_result('robust/cluster/yyyymm', 'robustness/clustering_variations.md',
               'DUR', 'PX1', res_dict, sample_desc='1984-2012, first interview',
               controls_desc='full baseline', cluster_var='yyyymm')
    print(f"    coef={coef_cl:.6f}, se={se_cl:.6f}, p={pval_cl:.4f}")
except Exception as e:
    print(f"    FAILED: {e}")

# Clustered by year
print("  Clustered by year...")
try:
    groups_yr = df_base['yyyy'].values if 'yyyy' in df_base.columns else (df_base['yyyymm'] // 100).values
    res_cly = model_rb.fit(method='bfgs', maxiter=5000, disp=False,
                           cov_type='cluster', cov_kwds={'groups': groups_yr})
    coef_cly = res_cly.params[0]
    se_cly = res_cly.bse[0]
    z_cly = coef_cly / se_cly
    pval_cly = 2 * (1 - stats.norm.cdf(abs(z_cly)))

    res_dict = {
        'coefficient': float(coef_cly), 'std_error': float(se_cly), 't_stat': float(z_cly),
        'p_value': float(pval_cly), 'ci_lower': float(coef_cly - 1.96*se_cly),
        'ci_upper': float(coef_cly + 1.96*se_cly), 'n_obs': len(y_rb),
        'r_squared': float(res_cly.prsquared) if hasattr(res_cly, 'prsquared') else None,
        'coefficient_vector_json': json.dumps({"treatment": {"var": "PX1", "coef": float(coef_cly), "se": float(se_cly), "pval": float(pval_cly)}}),
        'converged': True
    }
    add_result('robust/cluster/year', 'robustness/clustering_variations.md',
               'DUR', 'PX1', res_dict, sample_desc='1984-2012, first interview',
               controls_desc='full baseline', cluster_var='year')
    print(f"    coef={coef_cly:.6f}, se={se_cly:.6f}, p={pval_cly:.4f}")
except Exception as e:
    print(f"    FAILED: {e}")


# ============================================================
# SPEC 62-66: FUNCTIONAL FORM - outcome transformations
# ============================================================
print("\n=== FUNCTIONAL FORM ===")

# Binary: good time to buy (DUR == 1) vs not
print("  Binary: probit (DUR==1 vs not)...")
df_bin = df_base.copy()
df_bin['DUR_binary'] = (df_bin['DUR'] == 1).astype(int)
try:
    X_vars_bin = ['PX1', 'ZLB_PX1', 'ZLB'] + baseline_controls + ['AGE2', 'AGE3']
    month_d = pd.get_dummies(df_bin['mm_int'], prefix='month', drop_first=True, dtype=float)
    X_bin = pd.concat([df_bin[X_vars_bin].reset_index(drop=True), month_d.reset_index(drop=True)], axis=1)
    X_bin = sm.add_constant(X_bin)

    model_probit = sm.Probit(df_bin['DUR_binary'].values, X_bin.values).fit(disp=False, maxiter=5000)
    coef_pr = model_probit.params[1]
    se_pr = model_probit.bse[1]
    pval_pr = model_probit.pvalues[1]

    res_dict = {
        'coefficient': float(coef_pr), 'std_error': float(se_pr),
        't_stat': float(model_probit.tvalues[1]),
        'p_value': float(pval_pr), 'ci_lower': float(coef_pr - 1.96*se_pr),
        'ci_upper': float(coef_pr + 1.96*se_pr), 'n_obs': int(model_probit.nobs),
        'r_squared': float(model_probit.prsquared),
        'coefficient_vector_json': json.dumps({"treatment": {"var": "PX1", "coef": float(coef_pr), "se": float(se_pr), "pval": float(pval_pr)}}),
        'converged': True
    }
    add_result('robust/funcform/binary_probit', 'robustness/functional_form.md',
               'DUR_binary', 'PX1', res_dict, sample_desc='1984-2012, first interview',
               controls_desc='full baseline', model_type='binary_probit')
    print(f"    coef={coef_pr:.6f}, p={pval_pr:.4f}")
except Exception as e:
    print(f"    FAILED: {e}")

# Binary: bad time (DUR == -1) vs not
print("  Binary: probit (DUR==-1 vs not)...")
try:
    df_bin['DUR_bad'] = (df_bin['DUR'] == -1).astype(int)
    model_bad = sm.Probit(df_bin['DUR_bad'].values, X_bin.values).fit(disp=False, maxiter=5000)
    coef_bad = model_bad.params[1]
    se_bad = model_bad.bse[1]
    pval_bad = model_bad.pvalues[1]

    res_dict = {
        'coefficient': float(coef_bad), 'std_error': float(se_bad),
        't_stat': float(model_bad.tvalues[1]),
        'p_value': float(pval_bad), 'ci_lower': float(coef_bad - 1.96*se_bad),
        'ci_upper': float(coef_bad + 1.96*se_bad), 'n_obs': int(model_bad.nobs),
        'r_squared': float(model_bad.prsquared),
        'coefficient_vector_json': json.dumps({"treatment": {"var": "PX1", "coef": float(coef_bad), "se": float(se_bad), "pval": float(pval_bad)}}),
        'converged': True
    }
    add_result('robust/funcform/binary_probit_bad', 'robustness/functional_form.md',
               'DUR_bad', 'PX1', res_dict, sample_desc='1984-2012, first interview',
               controls_desc='full baseline', model_type='binary_probit')
    print(f"    coef={coef_bad:.6f}, p={pval_bad:.4f}")
except Exception as e:
    print(f"    FAILED: {e}")

# Logit binary
print("  Binary: logit (DUR==1 vs not)...")
try:
    model_logit = sm.Logit(df_bin['DUR_binary'].values, X_bin.values).fit(disp=False, maxiter=5000)
    coef_lg = model_logit.params[1]
    se_lg = model_logit.bse[1]
    pval_lg = model_logit.pvalues[1]

    res_dict = {
        'coefficient': float(coef_lg), 'std_error': float(se_lg),
        't_stat': float(model_logit.tvalues[1]),
        'p_value': float(pval_lg), 'ci_lower': float(coef_lg - 1.96*se_lg),
        'ci_upper': float(coef_lg + 1.96*se_lg), 'n_obs': int(model_logit.nobs),
        'r_squared': float(model_logit.prsquared),
        'coefficient_vector_json': json.dumps({"treatment": {"var": "PX1", "coef": float(coef_lg), "se": float(se_lg), "pval": float(pval_lg)}}),
        'converged': True
    }
    add_result('robust/funcform/binary_logit', 'robustness/functional_form.md',
               'DUR_binary', 'PX1', res_dict, sample_desc='1984-2012, first interview',
               controls_desc='full baseline', model_type='binary_logit')
    print(f"    coef={coef_lg:.6f}, p={pval_lg:.4f}")
except Exception as e:
    print(f"    FAILED: {e}")

# OLS on DUR levels
print("  OLS on DUR levels...")
try:
    model_ols = sm.OLS(df_base['DUR'].values, X_bin.values).fit()
    coef_ols = model_ols.params[1]
    se_ols = model_ols.bse[1]
    pval_ols = model_ols.pvalues[1]

    res_dict = {
        'coefficient': float(coef_ols), 'std_error': float(se_ols),
        't_stat': float(model_ols.tvalues[1]),
        'p_value': float(pval_ols), 'ci_lower': float(coef_ols - 1.96*se_ols),
        'ci_upper': float(coef_ols + 1.96*se_ols), 'n_obs': int(model_ols.nobs),
        'r_squared': float(model_ols.rsquared),
        'coefficient_vector_json': json.dumps({"treatment": {"var": "PX1", "coef": float(coef_ols), "se": float(se_ols), "pval": float(pval_ols)}}),
        'converged': True
    }
    add_result('robust/funcform/ols_levels', 'robustness/functional_form.md',
               'DUR', 'PX1', res_dict, sample_desc='1984-2012, first interview',
               controls_desc='full baseline', model_type='OLS')
    print(f"    coef={coef_ols:.6f}, p={pval_ols:.4f}")
except Exception as e:
    print(f"    FAILED: {e}")

# OLS with robust SEs
print("  OLS on DUR levels with robust SEs...")
try:
    model_ols_r = sm.OLS(df_base['DUR'].values, X_bin.values).fit(cov_type='HC1')
    coef_olsr = model_ols_r.params[1]
    se_olsr = model_ols_r.bse[1]
    pval_olsr = model_ols_r.pvalues[1]

    res_dict = {
        'coefficient': float(coef_olsr), 'std_error': float(se_olsr),
        't_stat': float(model_ols_r.tvalues[1]),
        'p_value': float(pval_olsr), 'ci_lower': float(coef_olsr - 1.96*se_olsr),
        'ci_upper': float(coef_olsr + 1.96*se_olsr), 'n_obs': int(model_ols_r.nobs),
        'r_squared': float(model_ols_r.rsquared),
        'coefficient_vector_json': json.dumps({"treatment": {"var": "PX1", "coef": float(coef_olsr), "se": float(se_olsr), "pval": float(pval_olsr)}}),
        'converged': True
    }
    add_result('robust/funcform/ols_levels_robust', 'robustness/functional_form.md',
               'DUR', 'PX1', res_dict, sample_desc='1984-2012, first interview',
               controls_desc='full baseline', model_type='OLS_robust')
    print(f"    coef={coef_olsr:.6f}, p={pval_olsr:.4f}")
except Exception as e:
    print(f"    FAILED: {e}")


# ============================================================
# SPEC 67-69: OUTLIER TREATMENT
# ============================================================
print("\n=== OUTLIER TREATMENT ===")

# Winsorize PX1 at different levels
for pct in [1, 5, 10]:
    dfs_w = df_base.copy()
    lo = dfs_w['PX1'].quantile(pct/100)
    hi = dfs_w['PX1'].quantile(1 - pct/100)
    dfs_w['PX1'] = dfs_w['PX1'].clip(lower=lo, upper=hi)
    dfs_w['ZLB_PX1'] = dfs_w['ZLB'] * dfs_w['PX1']
    res = run_oprobit(dfs_w, 'DUR', 'PX1', baseline_controls)
    add_result(f'robust/sample/winsorize_px1_{pct}pct', 'robustness/sample_restrictions.md',
               'DUR', 'PX1', res, sample_desc=f'PX1 winsorized at {pct}%')
    print(f"  Winsorize PX1 at {pct}%: coef={res['coefficient']:.6f}, p={res['p_value']:.4f}")

# Trim extreme PX1 values
print("  Trim extreme PX1...")
dfs_trim = df_base.copy()
lo_t = dfs_trim['PX1'].quantile(0.01)
hi_t = dfs_trim['PX1'].quantile(0.99)
dfs_trim = dfs_trim[(dfs_trim['PX1'] >= lo_t) & (dfs_trim['PX1'] <= hi_t)]
res = run_oprobit(dfs_trim, 'DUR', 'PX1', baseline_controls)
add_result('robust/sample/trim_px1_1pct', 'robustness/sample_restrictions.md',
           'DUR', 'PX1', res, sample_desc='PX1 trimmed at 1%')
print(f"  Trim PX1 1%: coef={res['coefficient']:.6f}, p={res['p_value']:.4f}")


# ============================================================
# SPEC 70-72: SECOND INTERVIEW SAMPLE
# ============================================================
print("\n=== SECOND INTERVIEW SAMPLE ===")

mask_second = (df_raw['yyyymm'] >= 198401) & (df_raw['yyyymm'] <= 201212) & (df_raw['second'] == 1)
df_second = df_raw[mask_second].copy()
df_second['ZLB_PX1'] = df_second['ZLB'] * df_second['PX1']
df_second['AGE2'] = df_second['AGE'] ** 2
df_second['AGE3'] = df_second['AGE'] ** 3
df_second['mm_int'] = df_second['mm'].astype(int)
df_second = df_second.dropna(subset=['DUR', 'PX1', 'ZLB', 'ZLB_PX1'] + baseline_controls + ['AGE2', 'AGE3', 'mm_int'])

res = run_oprobit(df_second, 'DUR', 'PX1', baseline_controls)
add_result('robust/sample/second_interview', 'robustness/sample_restrictions.md',
           'DUR', 'PX1', res, sample_desc='1984-2012, second interview only')
print(f"  Second interview: coef={res['coefficient']:.6f}, p={res['p_value']:.4f}, N={res['n_obs']}")

# Both interviews pooled
mask_all = (df_raw['yyyymm'] >= 198401) & (df_raw['yyyymm'] <= 201212)
df_all = df_raw[mask_all].copy()
df_all['ZLB_PX1'] = df_all['ZLB'] * df_all['PX1']
df_all['AGE2'] = df_all['AGE'] ** 2
df_all['AGE3'] = df_all['AGE'] ** 3
df_all['mm_int'] = df_all['mm'].astype(int)
df_all = df_all.dropna(subset=['DUR', 'PX1', 'ZLB', 'ZLB_PX1'] + baseline_controls + ['AGE2', 'AGE3', 'mm_int'])

res = run_oprobit(df_all, 'DUR', 'PX1', baseline_controls)
add_result('robust/sample/all_interviews', 'robustness/sample_restrictions.md',
           'DUR', 'PX1', res, sample_desc='1984-2012, all interviews pooled')
print(f"  All interviews: coef={res['coefficient']:.6f}, p={res['p_value']:.4f}, N={res['n_obs']}")

# Single interview only
mask_single = (df_raw['yyyymm'] >= 198401) & (df_raw['yyyymm'] <= 201212) & (df_raw['single'] == 1)
df_single = df_raw[mask_single].copy()
df_single['ZLB_PX1'] = df_single['ZLB'] * df_single['PX1']
df_single['AGE2'] = df_single['AGE'] ** 2
df_single['AGE3'] = df_single['AGE'] ** 3
df_single['mm_int'] = df_single['mm'].astype(int)
df_single = df_single.dropna(subset=['DUR', 'PX1', 'ZLB', 'ZLB_PX1'] + baseline_controls + ['AGE2', 'AGE3', 'mm_int'])

if len(df_single) > 100:
    res = run_oprobit(df_single, 'DUR', 'PX1', baseline_controls)
    add_result('robust/sample/single_interview', 'robustness/sample_restrictions.md',
               'DUR', 'PX1', res, sample_desc='1984-2012, single interview only')
    print(f"  Single interview: coef={res['coefficient']:.6f}, p={res['p_value']:.4f}, N={res['n_obs']}")


# ============================================================
# SPEC 73-76: HETEROGENEITY VIA INTERACTIONS
# ============================================================
print("\n=== HETEROGENEITY (INTERACTIONS) ===")

for het_var in ['SEX', 'ECLGRD', 'MARRY']:
    dfs_het = df_base.copy()
    dfs_het[f'PX1_x_{het_var}'] = dfs_het['PX1'] * dfs_het[het_var]

    het_controls = baseline_controls + [f'PX1_x_{het_var}']

    try:
        X_vars_h = ['PX1', 'ZLB_PX1', 'ZLB'] + het_controls + ['AGE2', 'AGE3']
        month_d_h = pd.get_dummies(dfs_het['mm_int'], prefix='month', drop_first=True, dtype=float)
        X_h = pd.concat([dfs_het[X_vars_h].reset_index(drop=True), month_d_h.reset_index(drop=True)], axis=1)
        y_h = (dfs_het['DUR'].values + 1).astype(int)

        model_h = OrderedModel(y_h, X_h.values, distr='probit')
        res_h = model_h.fit(method='bfgs', maxiter=5000, disp=False)

        # Find the interaction coefficient
        int_idx = X_vars_h.index(f'PX1_x_{het_var}')
        coef_int = res_h.params[int_idx]
        se_int = res_h.bse[int_idx]
        z_int = coef_int / se_int
        pval_int = 2 * (1 - stats.norm.cdf(abs(z_int)))

        # Also record main PX1 effect
        coef_main = res_h.params[0]
        se_main = res_h.bse[0]
        z_main = coef_main / se_main
        pval_main = 2 * (1 - stats.norm.cdf(abs(z_main)))

        # Report main effect
        res_dict = {
            'coefficient': float(coef_main), 'std_error': float(se_main), 't_stat': float(z_main),
            'p_value': float(pval_main), 'ci_lower': float(coef_main - 1.96*se_main),
            'ci_upper': float(coef_main + 1.96*se_main), 'n_obs': len(y_h),
            'r_squared': float(res_h.prsquared) if hasattr(res_h, 'prsquared') else None,
            'coefficient_vector_json': json.dumps({
                "treatment": {"var": "PX1", "coef": float(coef_main), "se": float(se_main), "pval": float(pval_main)},
                "interaction": {"var": f"PX1_x_{het_var}", "coef": float(coef_int), "se": float(se_int), "pval": float(pval_int)},
            }),
            'converged': True
        }
        add_result(f'robust/heterogeneity/px1_x_{het_var.lower()}', 'robustness/heterogeneity.md',
                   'DUR', 'PX1', res_dict, sample_desc='1984-2012, first interview',
                   controls_desc=f'baseline + PX1*{het_var} interaction')
        print(f"  PX1*{het_var}: main={coef_main:.6f}(p={pval_main:.4f}), interaction={coef_int:.6f}(p={pval_int:.4f})")
    except Exception as e:
        print(f"  PX1*{het_var}: FAILED: {e}")

# Interaction with ZLB dummy explicitly (different from baseline ZLB*PX1)
# Already in baseline, but let's try PX1 * AGE interaction
try:
    dfs_het = df_base.copy()
    dfs_het['PX1_x_AGE'] = dfs_het['PX1'] * dfs_het['AGE']
    het_controls_age = baseline_controls + ['PX1_x_AGE']
    X_vars_ha = ['PX1', 'ZLB_PX1', 'ZLB'] + het_controls_age + ['AGE2', 'AGE3']
    month_d_ha = pd.get_dummies(dfs_het['mm_int'], prefix='month', drop_first=True, dtype=float)
    X_ha = pd.concat([dfs_het[X_vars_ha].reset_index(drop=True), month_d_ha.reset_index(drop=True)], axis=1)
    y_ha = (dfs_het['DUR'].values + 1).astype(int)

    model_ha = OrderedModel(y_ha, X_ha.values, distr='probit')
    res_ha = model_ha.fit(method='bfgs', maxiter=5000, disp=False)

    coef_main = res_ha.params[0]
    se_main = res_ha.bse[0]
    pval_main = 2 * (1 - stats.norm.cdf(abs(coef_main / se_main)))
    int_idx = X_vars_ha.index('PX1_x_AGE')
    coef_int = res_ha.params[int_idx]
    se_int = res_ha.bse[int_idx]
    pval_int = 2 * (1 - stats.norm.cdf(abs(coef_int / se_int)))

    res_dict = {
        'coefficient': float(coef_main), 'std_error': float(se_main),
        't_stat': float(coef_main/se_main),
        'p_value': float(pval_main), 'ci_lower': float(coef_main - 1.96*se_main),
        'ci_upper': float(coef_main + 1.96*se_main), 'n_obs': len(y_ha),
        'r_squared': float(res_ha.prsquared) if hasattr(res_ha, 'prsquared') else None,
        'coefficient_vector_json': json.dumps({
            "treatment": {"var": "PX1", "coef": float(coef_main), "se": float(se_main), "pval": float(pval_main)},
            "interaction": {"var": "PX1_x_AGE", "coef": float(coef_int), "se": float(se_int), "pval": float(pval_int)},
        }),
        'converged': True
    }
    add_result('robust/heterogeneity/px1_x_age', 'robustness/heterogeneity.md',
               'DUR', 'PX1', res_dict, sample_desc='1984-2012, first interview',
               controls_desc='baseline + PX1*AGE interaction')
    print(f"  PX1*AGE: main={coef_main:.6f}(p={pval_main:.4f}), interaction={coef_int:.6f}(p={pval_int:.4f})")
except Exception as e:
    print(f"  PX1*AGE: FAILED: {e}")


# ============================================================
# SPEC 77-78: EXTENDED CONTROLS (Table 3 cols 3-4)
# ============================================================
print("\n=== EXTENDED CONTROLS ===")

# With gas price expectations, homeowner, subjective probabilities (Table 3 col 3)
print("  Extended controls (gas, homeown, pjob, pinc)...")
mask_ext = (df_raw['yyyymm'] >= 199801) & (df_raw['yyyymm'] <= 201212) & (df_raw['idprev'].isna())
df_ext = df_raw[mask_ext].copy()
df_ext['ZLB_PX1'] = df_ext['ZLB'] * df_ext['PX1']
df_ext['AGE2'] = df_ext['AGE'] ** 2
df_ext['AGE3'] = df_ext['AGE'] ** 3
df_ext['mm_int'] = df_ext['mm'].astype(int)
ext_controls = baseline_controls + ['PJOB', 'PINC', 'GAS1', 'HOMEOWN']
df_ext = df_ext.dropna(subset=['DUR', 'PX1', 'ZLB', 'ZLB_PX1'] + ext_controls + ['AGE2', 'AGE3', 'mm_int'])

res = run_oprobit(df_ext, 'DUR', 'PX1', ext_controls)
add_result('robust/control/extended_gas_prob', 'robustness/control_progression.md',
           'DUR', 'PX1', res, sample_desc='1998-2012, first interview',
           controls_desc='baseline + PJOB + PINC + GAS1 + HOMEOWN')
print(f"    coef={res['coefficient']:.6f}, p={res['p_value']:.4f}, N={res['n_obs']}")

# Without expected nominal interest rate (RATEX dropped) - Additional results section
print("  Without RATEX...")
controls_no_ratex = [c for c in baseline_controls if c != 'RATEX']
res = run_oprobit(df_base, 'DUR', 'PX1', controls_no_ratex)
add_result('robust/control/no_ratex', 'robustness/leave_one_out.md',
           'DUR', 'PX1', res, sample_desc='1984-2012, first interview',
           controls_desc='baseline minus RATEX')
print(f"    coef={res['coefficient']:.6f}, p={res['p_value']:.4f}")


# ============================================================
# SPEC 79-80: CONTROL FUNCTION (Table 11)
# ============================================================
print("\n=== CONTROL FUNCTION (Table 11 analog) ===")

# First stage: PX1 on PX1LAG + controls (only for second-interview subsample)
try:
    mask_cf = (df_raw['yyyymm'] >= 198401) & (df_raw['yyyymm'] <= 201212) & (df_raw['second'] == 1)
    df_cf = df_raw[mask_cf].copy()
    df_cf['ZLB_PX1'] = df_cf['ZLB'] * df_cf['PX1']
    df_cf['AGE2'] = df_cf['AGE'] ** 2
    df_cf['AGE3'] = df_cf['AGE'] ** 3
    df_cf['mm_int'] = df_cf['mm'].astype(int)
    cf_first_controls = baseline_controls + ['PX1LAG']
    df_cf = df_cf.dropna(subset=['DUR', 'PX1', 'ZLB', 'ZLB_PX1', 'PX1LAG'] + baseline_controls + ['AGE2', 'AGE3', 'mm_int'])

    # First stage OLS
    X_vars_cf = ['ZLB'] + baseline_controls + ['PX1LAG', 'AGE2', 'AGE3']
    month_d_cf = pd.get_dummies(df_cf['mm_int'], prefix='month', drop_first=True, dtype=float)
    X_cf = pd.concat([df_cf[X_vars_cf].reset_index(drop=True), month_d_cf.reset_index(drop=True)], axis=1)
    X_cf = sm.add_constant(X_cf)

    model_fs = sm.OLS(df_cf['PX1'].values, X_cf.values).fit()
    df_cf = df_cf.reset_index(drop=True)
    df_cf['PX1HAT'] = model_fs.fittedvalues
    df_cf['UHAT'] = df_cf['PX1'].values - df_cf['PX1HAT']

    # Second stage: oprobit DUR PX1 ZLB#c.PX1 ... UHAT
    cf_controls = baseline_controls + ['UHAT']
    res = run_oprobit(df_cf, 'DUR', 'PX1', cf_controls)
    add_result('robust/estimation/control_function', 'robustness/model_specification.md',
               'DUR', 'PX1', res, sample_desc='1984-2012, second interview, control function',
               controls_desc='baseline + UHAT (control function residual)')
    print(f"  Control function: coef={res['coefficient']:.6f}, p={res['p_value']:.4f}, N={res['n_obs']}")
except Exception as e:
    print(f"  Control function: FAILED: {e}")


# ============================================================
# SPEC 81-83: DEMEANED EXPECTATIONS
# ============================================================
print("\n=== DEMEANED / ALTERNATIVE MEASUREMENT ===")

# PX1BAR (demeaned 1Y expectations)
mask_bar = (df_raw['yyyymm'] >= 198401) & (df_raw['yyyymm'] <= 201212) & (df_raw['second'] == 1)
df_bar = df_raw[mask_bar].copy()
df_bar['ZLB_PX1BAR'] = df_bar['ZLB'] * df_bar['PX1BAR']
df_bar['ZLB_PX1'] = df_bar['ZLB'] * df_bar['PX1']
df_bar['AGE2'] = df_bar['AGE'] ** 2
df_bar['AGE3'] = df_bar['AGE'] ** 3
df_bar['mm_int'] = df_bar['mm'].astype(int)
df_bar = df_bar.dropna(subset=['DUR', 'PX1BAR', 'ZLB', 'ZLB_PX1BAR'] + baseline_controls + ['AGE2', 'AGE3', 'mm_int'])

if len(df_bar) > 100:
    # Need custom run since treatment is PX1BAR
    df_bar_c = df_bar.copy()
    df_bar_c['ZLB_PX1'] = df_bar_c['ZLB_PX1BAR']  # hack for run_oprobit
    df_bar_c['PX1'] = df_bar_c['PX1BAR']
    res = run_oprobit(df_bar_c, 'DUR', 'PX1', baseline_controls)
    add_result('robust/treatment/px1bar_demeaned', 'robustness/measurement.md',
               'DUR', 'PX1BAR', res, sample_desc='1984-2012, second interview',
               controls_desc='full baseline, demeaned inflation expectations')
    print(f"  PX1BAR: coef={res['coefficient']:.6f}, p={res['p_value']:.4f}, N={res['n_obs']}")


# ============================================================
# SAVE RESULTS
# ============================================================
print(f"\n{'='*60}")
print(f"TOTAL SPECIFICATIONS: {len(results)}")
print(f"{'='*60}")

df_results = pd.DataFrame(results)

# Save to package directory
output_path = os.path.join(PKG_DIR, "specification_results.csv")
df_results.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")

# Summary stats
print(f"\n=== SUMMARY ===")
valid = df_results[df_results['coefficient'].notna()]
print(f"Valid specifications: {len(valid)}/{len(df_results)}")
print(f"Positive coefficients: {(valid['coefficient'] > 0).sum()} ({100*(valid['coefficient'] > 0).mean():.1f}%)")
print(f"Negative coefficients: {(valid['coefficient'] < 0).sum()} ({100*(valid['coefficient'] < 0).mean():.1f}%)")
print(f"Significant at 5%: {(valid['p_value'] < 0.05).sum()} ({100*(valid['p_value'] < 0.05).mean():.1f}%)")
print(f"Significant at 1%: {(valid['p_value'] < 0.01).sum()} ({100*(valid['p_value'] < 0.01).mean():.1f}%)")
print(f"Median coefficient: {valid['coefficient'].median():.6f}")
print(f"Mean coefficient: {valid['coefficient'].mean():.6f}")
print(f"Range: [{valid['coefficient'].min():.6f}, {valid['coefficient'].max():.6f}]")
print(f"Median t-stat: {valid['t_stat'].median():.3f}")
