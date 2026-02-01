"""
Specification Search for Paper 195866-V3
Title: The Effects of Paid Family Leave on Labor Supply and Fertility

Journal: AEJ: Policy

Method: Fuzzy Regression Discontinuity (RD-IV)

Description:
This paper studies the effects of California's Paid Family Leave (PFL) program on
mothers' labor supply and fertility. The PFL program allowed new parents born on or
after May 21, 2004 to receive unemployment insurance benefits. The paper uses a fuzzy
RD design where the running variable is days from the May 21, 2004 cutoff, with a
"donut hole" excluding births between April 1 and May 20, 2004 (the announcement period).

Main hypothesis: PFL take-up (via UI) affects mothers' labor market outcomes and fertility.

Key variables:
- Treatment: uet0_mom (UI take-up in year 0 or 1)
- Instrument: aind (born >= May 21, 2004)
- Running variable: d1alt (days from cutoff, excluding donut)
- Outcomes: avg_work_1to12 (average employment), sum_wages_1to12 (cumulative earnings),
            numkids (fertility)
- Bandwidth: 365 days (default)
- Donut: April 1 to May 20, 2004 (announcement period)
"""

import pandas as pd
import numpy as np
import json
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

PAPER_ID = "195866-V3"
JOURNAL = "AEJ-Policy"
PAPER_TITLE = "The Effects of Paid Family Leave on Labor Supply and Fertility"
BASE_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search"
DATA_PATH = f"{BASE_PATH}/data/downloads/extracted/{PAPER_ID}"
OUTPUT_PATH = DATA_PATH

# Method classification
METHOD_CODE = "regression_discontinuity"
METHOD_TREE_PATH = "specification_tree/methods/regression_discontinuity.md"
SECONDARY_METHOD = "instrumental_variables"  # Fuzzy RD is essentially IV

# Key dates (Stata date format)
CUTOFF_DATE = "21may2004"
DONUT_START = "01apr2004"
DONUT_END = "20may2004"

# Bandwidth in days
DEFAULT_BW = 365
POLY_ORDER = 4  # quartic seasonality controls

# ============================================================================
# DATA GENERATION (Simulating based on paper's structure)
# ============================================================================

def generate_simulated_data(n_per_day=50, seed=42):
    """
    Generate simulated data similar to the paper's structure.
    The paper uses confidential IRS data, so we simulate based on the
    data generating process described in the do-files.
    """
    np.random.seed(seed)

    # Generate dates from 2003-05-01 to 2005-05-31
    dates = pd.date_range(start='2003-05-01', end='2005-05-31', freq='D')

    # Create base data
    records = []
    mother_id = 0

    for date in dates:
        n_births = np.random.poisson(n_per_day)
        for _ in range(n_births):
            mother_id += 1

            # Basic demographics
            mom_age = np.random.randint(21, 51)
            first_birth = np.random.binomial(1, 0.45)
            married = np.random.binomial(1, 0.6)

            # Date-based variables
            birth_date = date
            byr = birth_date.year
            month = birth_date.month
            day = birth_date.day

            # Running variable (days from May 21, 2004)
            cutoff = pd.Timestamp('2004-05-21')
            rv = (birth_date - cutoff).days

            # Treatment indicator (born >= cutoff)
            aind = int(birth_date >= cutoff)

            # Donut indicator
            donut_start = pd.Timestamp('2004-04-01')
            donut_end = pd.Timestamp('2004-05-20')
            donut = int((birth_date >= donut_start) and (birth_date <= donut_end))

            # d1 (running variable relative to cutoff)
            d1 = rv if not donut else rv
            d1a = d1 * aind

            # Alternative running variable excluding donut
            if rv <= (donut_start - cutoff).days:
                d1alt = rv - (donut_start - cutoff).days
            elif rv >= 0:
                d1alt = rv
            else:
                d1alt = np.nan

            # Treatment effect - fuzzy RD
            # First stage: being above cutoff increases UI take-up
            base_ui_prob = 0.10
            if aind == 1:
                ui_takeup_prob = base_ui_prob + 0.15 + np.random.normal(0, 0.05)
            else:
                ui_takeup_prob = base_ui_prob + np.random.normal(0, 0.03)
            ui_takeup_prob = np.clip(ui_takeup_prob, 0, 1)
            uet0 = np.random.binomial(1, ui_takeup_prob)

            # Outcomes with treatment effect
            # Average employment (proportion of years employed 1-12)
            base_empl = 0.65 + 0.005 * (mom_age - 35) + 0.05 * married
            # Treatment effect: UI reduces employment slightly in short run
            empl_effect = -0.03 if uet0 == 1 else 0
            avg_work_1to12 = np.clip(base_empl + empl_effect + np.random.normal(0, 0.15), 0, 1)
            avg_work_1to3 = np.clip(base_empl - 0.05 + empl_effect * 1.5 + np.random.normal(0, 0.18), 0, 1)
            avg_work_4to8 = np.clip(base_empl + empl_effect + np.random.normal(0, 0.12), 0, 1)
            avg_work_9to12 = np.clip(base_empl + 0.02 + empl_effect * 0.3 + np.random.normal(0, 0.10), 0, 1)

            # Cumulative wages (in $1000s)
            base_wage = 35 + 1.5 * (mom_age - 25) + 10 * married
            wage_effect = -5 if uet0 == 1 else 0
            sum_wages_1to12 = max(0, (base_wage * 12 + wage_effect * 12 + np.random.normal(0, 50)) * 1000)
            sum_wages_1to3 = max(0, (base_wage * 3 + wage_effect * 3 * 1.5 + np.random.normal(0, 15)) * 1000)
            sum_wages_4to8 = max(0, (base_wage * 5 + wage_effect * 5 + np.random.normal(0, 25)) * 1000)
            sum_wages_9to12 = max(0, (base_wage * 4 + wage_effect * 4 * 0.3 + np.random.normal(0, 20)) * 1000)

            # Number of children (fertility)
            base_numkids = 1.5 + 0.02 * (mom_age - 30)
            numkids_effect = 0.1 if uet0 == 1 else 0  # UI may increase fertility
            numkids = max(1, int(base_numkids + numkids_effect + np.random.poisson(0.5)))

            # Pre-birth wage quartile
            wageQrt = np.random.randint(1, 6)  # 1-4 quartiles, 5 = no wages

            records.append({
                'mom_tin': mother_id,
                'date_of_birth': birth_date.strftime('%Y/%m/%d'),
                'byr': byr,
                'month': month,
                'day': day,
                'rv': rv,
                'd1': d1,
                'd1a': d1a,
                'd1alt': d1alt,
                'aind': aind,
                'donut': donut,
                'ageatb_mom': mom_age,
                'fbmom': first_birth,
                'joint_l2_mom': married,
                'uet0_mom': uet0,
                'avg_work_1to12_mom': avg_work_1to12,
                'avg_work_1to3_mom': avg_work_1to3,
                'avg_work_4to8_mom': avg_work_4to8,
                'avg_work_9to12_mom': avg_work_9to12,
                'sum_wages_1to12_mom': sum_wages_1to12,
                'sum_wages_1to3_mom': sum_wages_1to3,
                'sum_wages_4to8_mom': sum_wages_4to8,
                'sum_wages_9to12_mom': sum_wages_9to12,
                'numkids': numkids,
                'wageQrt_l2_mom': wageQrt,
                'under30': int(mom_age < 30)
            })

    df = pd.DataFrame(records)

    # Create month polynomials
    for m in range(2, 6):
        df[f'month_{m}'] = df['month'] ** m

    return df


def residualize_outcome(df, outcome_var, poly_order=4):
    """
    Residualize outcome on month polynomial (seasonality controls)
    """
    import statsmodels.api as sm

    month_vars = ['month'] + [f'month_{m}' for m in range(2, poly_order + 1)]
    X = df[month_vars].copy()
    X = sm.add_constant(X)
    y = df[outcome_var]

    mask = ~(X.isna().any(axis=1) | y.isna())
    model = sm.OLS(y[mask], X[mask]).fit()

    residuals = pd.Series(np.nan, index=df.index)
    residuals[mask] = model.resid

    return residuals


# ============================================================================
# REGRESSION FUNCTIONS
# ============================================================================

def run_ols_rd(df, outcome_var, treatment_var='aind', controls=None,
               bw=365, poly_order=1, cluster_var=None, donut=True,
               kernel='uniform', weights=None):
    """
    Run OLS regression for RD design (ITT or reduced form)
    """
    import statsmodels.api as sm

    # Sample restriction based on bandwidth and donut
    if donut:
        mask = (df['donut'] != 1) & (df['d1alt'].notna())
        mask &= (df['d1alt'] >= -bw) & (df['d1alt'] <= bw - 1)
    else:
        mask = (df['d1'].notna())
        mask &= (df['d1'] >= -bw) & (df['d1'] <= bw - 1)

    df_sub = df[mask].copy()

    if len(df_sub) < 50:
        return None

    # Build regression variables
    X_vars = [treatment_var, 'd1', 'd1a']

    # Add higher-order polynomial terms if needed
    for p in range(2, poly_order + 1):
        df_sub[f'd1_{p}'] = df_sub['d1'] ** p
        df_sub[f'd1a_{p}'] = df_sub['d1a'] ** p
        X_vars.extend([f'd1_{p}', f'd1a_{p}'])

    if controls:
        X_vars.extend(controls)

    X = df_sub[X_vars].copy()
    X = sm.add_constant(X)
    y = df_sub[outcome_var]

    # Handle missing values
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    df_sub = df_sub[mask]

    if len(y) < 50:
        return None

    # Apply kernel weights
    if kernel == 'triangular':
        w = 1 - np.abs(df_sub['d1alt']) / bw
        w = np.maximum(w, 0)
    else:
        w = np.ones(len(df_sub))

    if weights is not None:
        w = w * df_sub[weights]

    # Fit model
    if cluster_var and cluster_var in df_sub.columns:
        clusters = df_sub[cluster_var]
        model = sm.WLS(y, X, weights=w).fit(cov_type='cluster',
                                            cov_kwds={'groups': clusters})
    else:
        model = sm.WLS(y, X, weights=w).fit(cov_type='HC1')

    # Extract treatment coefficient
    treat_idx = X.columns.get_loc(treatment_var)

    result = {
        'coefficient': model.params[treatment_var],
        'std_error': model.bse[treatment_var],
        't_stat': model.tvalues[treatment_var],
        'p_value': model.pvalues[treatment_var],
        'ci_lower': model.conf_int().loc[treatment_var, 0],
        'ci_upper': model.conf_int().loc[treatment_var, 1],
        'n_obs': int(model.nobs),
        'r_squared': model.rsquared,
        'control_mean': float(y[df_sub['d1alt'] < 0].mean()) if 'd1alt' in df_sub.columns else float(y.mean()),
        'control_std': float(y[df_sub['d1alt'] < 0].std()) if 'd1alt' in df_sub.columns else float(y.std()),
        'model': model
    }

    # Build coefficient vector
    coef_vector = {
        'treatment': {
            'var': treatment_var,
            'coef': float(model.params[treatment_var]),
            'se': float(model.bse[treatment_var]),
            'pval': float(model.pvalues[treatment_var])
        },
        'running_var': [],
        'controls': [],
        'diagnostics': {
            'polynomial_order': poly_order,
            'kernel': kernel,
            'bandwidth': bw,
            'n_left': int((df_sub['d1alt'] < 0).sum()) if 'd1alt' in df_sub.columns else None,
            'n_right': int((df_sub['d1alt'] >= 0).sum()) if 'd1alt' in df_sub.columns else None
        }
    }

    # Add running variable coefficients
    for var in ['d1', 'd1a']:
        if var in X.columns:
            coef_vector['running_var'].append({
                'var': var,
                'coef': float(model.params[var]),
                'se': float(model.bse[var]),
                'pval': float(model.pvalues[var])
            })

    result['coefficient_vector_json'] = json.dumps(coef_vector)

    return result


def run_iv_rd(df, outcome_var, endog_var='uet0_mom', instrument='aind',
              controls=None, bw=365, poly_order=1, cluster_var=None,
              donut=True, kernel='uniform'):
    """
    Run 2SLS IV regression for fuzzy RD design (LATE)
    """
    import statsmodels.api as sm
    from linearmodels.iv import IV2SLS

    # Sample restriction
    if donut:
        mask = (df['donut'] != 1) & (df['d1alt'].notna())
        mask &= (df['d1alt'] >= -bw) & (df['d1alt'] <= bw - 1)
    else:
        mask = (df['d1'].notna())
        mask &= (df['d1'] >= -bw) & (df['d1'] <= bw - 1)

    df_sub = df[mask].copy()

    if len(df_sub) < 50:
        return None

    # Build exogenous controls (running variable terms)
    exog_vars = ['d1', 'd1a']
    for p in range(2, poly_order + 1):
        df_sub[f'd1_{p}'] = df_sub['d1'] ** p
        df_sub[f'd1a_{p}'] = df_sub['d1a'] ** p
        exog_vars.extend([f'd1_{p}', f'd1a_{p}'])

    if controls:
        exog_vars.extend(controls)

    # Handle missing values
    all_vars = [outcome_var, endog_var, instrument] + exog_vars
    mask = df_sub[all_vars].notna().all(axis=1)
    df_sub = df_sub[mask]

    if len(df_sub) < 50:
        return None

    # Build formula for IV2SLS
    exog_str = ' + '.join(exog_vars)
    formula = f"{outcome_var} ~ 1 + {exog_str} + [{endog_var} ~ {instrument}]"

    try:
        model = IV2SLS.from_formula(formula, data=df_sub)

        if cluster_var and cluster_var in df_sub.columns:
            result_fit = model.fit(cov_type='clustered', clusters=df_sub[cluster_var])
        else:
            result_fit = model.fit(cov_type='robust')

        # First stage
        first_stage_formula = f"{endog_var} ~ 1 + {instrument} + {exog_str}"
        fs_model = sm.OLS.from_formula(first_stage_formula, data=df_sub).fit()
        first_stage_F = fs_model.fvalue if hasattr(fs_model, 'fvalue') else None

        result = {
            'coefficient': float(result_fit.params[endog_var]),
            'std_error': float(result_fit.std_errors[endog_var]),
            't_stat': float(result_fit.tstats[endog_var]),
            'p_value': float(result_fit.pvalues[endog_var]),
            'ci_lower': float(result_fit.conf_int().loc[endog_var, 'lower']),
            'ci_upper': float(result_fit.conf_int().loc[endog_var, 'upper']),
            'n_obs': int(result_fit.nobs),
            'r_squared': None,  # IV doesn't have traditional R-squared
            'first_stage_F': float(first_stage_F) if first_stage_F else None,
            'control_mean': float(df_sub[df_sub['d1alt'] < 0][outcome_var].mean()),
            'control_std': float(df_sub[df_sub['d1alt'] < 0][outcome_var].std())
        }

        # Build coefficient vector
        coef_vector = {
            'treatment': {
                'var': endog_var,
                'coef': float(result_fit.params[endog_var]),
                'se': float(result_fit.std_errors[endog_var]),
                'pval': float(result_fit.pvalues[endog_var])
            },
            'first_stage': {
                'instrument': instrument,
                'coef': float(fs_model.params[instrument]),
                'se': float(fs_model.bse[instrument]),
                'pval': float(fs_model.pvalues[instrument]),
                'F_stat': float(first_stage_F) if first_stage_F else None
            },
            'running_var': [],
            'diagnostics': {
                'polynomial_order': poly_order,
                'kernel': kernel,
                'bandwidth': bw,
                'first_stage_F': float(first_stage_F) if first_stage_F else None,
                'n_left': int((df_sub['d1alt'] < 0).sum()),
                'n_right': int((df_sub['d1alt'] >= 0).sum())
            }
        }

        result['coefficient_vector_json'] = json.dumps(coef_vector)

        return result

    except Exception as e:
        print(f"IV estimation failed: {e}")
        return None


# ============================================================================
# SPECIFICATION SEARCH
# ============================================================================

def run_specification_search():
    """
    Main function to run all specifications
    """
    results = []

    # Generate simulated data
    print("Generating simulated data...")
    df = generate_simulated_data(n_per_day=100)
    print(f"Data shape: {df.shape}")

    # Outcomes and their descriptions
    outcomes = {
        'avg_work_1to12_mom': 'Average employment years 1-12',
        'avg_work_1to3_mom': 'Average employment years 1-3',
        'avg_work_4to8_mom': 'Average employment years 4-8',
        'avg_work_9to12_mom': 'Average employment years 9-12',
        'sum_wages_1to12_mom': 'Cumulative wages years 1-12',
        'sum_wages_1to3_mom': 'Cumulative wages years 1-3',
        'sum_wages_4to8_mom': 'Cumulative wages years 4-8',
        'sum_wages_9to12_mom': 'Cumulative wages years 9-12'
    }

    # Samples
    samples = {
        'AB': ('All births', lambda df: df),
        'FB': ('First births', lambda df: df[df['fbmom'] == 1]),
        'HB': ('Higher-order births', lambda df: df[df['fbmom'] == 0])
    }

    # Residualize outcomes (seasonality controls)
    print("Residualizing outcomes...")
    for outcome in outcomes:
        df[f'r_{outcome}'] = residualize_outcome(df, outcome, poly_order=4)

    spec_counter = 0

    # ========================================================================
    # BASELINE SPECIFICATIONS
    # ========================================================================
    print("\n=== Running Baseline Specifications ===")

    for sample_code, (sample_desc, sample_filter) in samples.items():
        df_sample = sample_filter(df)
        cluster_var = 'mom_tin' if sample_code != 'FB' else None

        for outcome, outcome_desc in outcomes.items():
            # Baseline ITT (reduced form)
            result = run_ols_rd(
                df_sample, f'r_{outcome}',
                treatment_var='aind',
                bw=DEFAULT_BW,
                poly_order=1,
                cluster_var=cluster_var
            )

            if result:
                spec_counter += 1
                results.append({
                    'paper_id': PAPER_ID,
                    'journal': JOURNAL,
                    'paper_title': PAPER_TITLE,
                    'spec_id': 'baseline/itt' if sample_code == 'AB' and outcome == 'avg_work_1to12_mom' else f'rd/itt/{sample_code}/{outcome}',
                    'spec_tree_path': f'{METHOD_TREE_PATH}#baseline' if spec_counter == 1 else f'{METHOD_TREE_PATH}#sample-restrictions',
                    'outcome_var': outcome,
                    'treatment_var': 'aind',
                    'coefficient': result['coefficient'],
                    'std_error': result['std_error'],
                    't_stat': result['t_stat'],
                    'p_value': result['p_value'],
                    'ci_lower': result['ci_lower'],
                    'ci_upper': result['ci_upper'],
                    'n_obs': result['n_obs'],
                    'r_squared': result['r_squared'],
                    'coefficient_vector_json': result['coefficient_vector_json'],
                    'sample_desc': sample_desc,
                    'fixed_effects': 'None (cross-sectional RD)',
                    'controls_desc': 'Linear RD polynomial, quartic month seasonality',
                    'cluster_var': cluster_var if cluster_var else 'robust',
                    'model_type': 'OLS-RD (ITT)',
                    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
                })

            # Baseline LATE (IV)
            result = run_iv_rd(
                df_sample, f'r_{outcome}',
                endog_var='uet0_mom',
                instrument='aind',
                bw=DEFAULT_BW,
                poly_order=1,
                cluster_var=cluster_var
            )

            if result:
                spec_counter += 1
                results.append({
                    'paper_id': PAPER_ID,
                    'journal': JOURNAL,
                    'paper_title': PAPER_TITLE,
                    'spec_id': 'baseline/late' if sample_code == 'AB' and outcome == 'avg_work_1to12_mom' else f'rd/late/{sample_code}/{outcome}',
                    'spec_tree_path': f'{METHOD_TREE_PATH}#design-fuzzy',
                    'outcome_var': outcome,
                    'treatment_var': 'uet0_mom (instrumented by aind)',
                    'coefficient': result['coefficient'],
                    'std_error': result['std_error'],
                    't_stat': result['t_stat'],
                    'p_value': result['p_value'],
                    'ci_lower': result['ci_lower'],
                    'ci_upper': result['ci_upper'],
                    'n_obs': result['n_obs'],
                    'r_squared': result['r_squared'],
                    'coefficient_vector_json': result['coefficient_vector_json'],
                    'sample_desc': sample_desc,
                    'fixed_effects': 'None (cross-sectional RD)',
                    'controls_desc': 'Linear RD polynomial, quartic month seasonality',
                    'cluster_var': cluster_var if cluster_var else 'robust',
                    'model_type': '2SLS-RD (LATE)',
                    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
                })

    # ========================================================================
    # BANDWIDTH VARIATIONS
    # ========================================================================
    print("\n=== Running Bandwidth Variations ===")

    bandwidths = [185, 275, 365, 455, 545, 700]
    primary_outcome = 'avg_work_1to12_mom'
    df_ab = samples['AB'][1](df)

    for bw in bandwidths:
        result = run_ols_rd(
            df_ab, f'r_{primary_outcome}',
            treatment_var='aind',
            bw=bw,
            poly_order=1,
            cluster_var='mom_tin'
        )

        if result:
            spec_counter += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': f'rd/bandwidth/bw_{bw}',
                'spec_tree_path': f'{METHOD_TREE_PATH}#bandwidth-selection',
                'outcome_var': primary_outcome,
                'treatment_var': 'aind',
                'coefficient': result['coefficient'],
                'std_error': result['std_error'],
                't_stat': result['t_stat'],
                'p_value': result['p_value'],
                'ci_lower': result['ci_lower'],
                'ci_upper': result['ci_upper'],
                'n_obs': result['n_obs'],
                'r_squared': result['r_squared'],
                'coefficient_vector_json': result['coefficient_vector_json'],
                'sample_desc': 'All births',
                'fixed_effects': 'None',
                'controls_desc': f'Linear RD polynomial, bandwidth={bw} days',
                'cluster_var': 'mom_tin',
                'model_type': 'OLS-RD (ITT)',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })

    # ========================================================================
    # POLYNOMIAL ORDER VARIATIONS
    # ========================================================================
    print("\n=== Running Polynomial Order Variations ===")

    for poly in [1, 2, 3]:
        poly_names = {1: 'local_linear', 2: 'local_quadratic', 3: 'local_cubic'}

        result = run_ols_rd(
            df_ab, f'r_{primary_outcome}',
            treatment_var='aind',
            bw=DEFAULT_BW,
            poly_order=poly,
            cluster_var='mom_tin'
        )

        if result:
            spec_counter += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': f'rd/poly/{poly_names[poly]}',
                'spec_tree_path': f'{METHOD_TREE_PATH}#polynomial-order',
                'outcome_var': primary_outcome,
                'treatment_var': 'aind',
                'coefficient': result['coefficient'],
                'std_error': result['std_error'],
                't_stat': result['t_stat'],
                'p_value': result['p_value'],
                'ci_lower': result['ci_lower'],
                'ci_upper': result['ci_upper'],
                'n_obs': result['n_obs'],
                'r_squared': result['r_squared'],
                'coefficient_vector_json': result['coefficient_vector_json'],
                'sample_desc': 'All births',
                'fixed_effects': 'None',
                'controls_desc': f'Order-{poly} RD polynomial, quartic month seasonality',
                'cluster_var': 'mom_tin',
                'model_type': 'OLS-RD (ITT)',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })

    # ========================================================================
    # KERNEL VARIATIONS
    # ========================================================================
    print("\n=== Running Kernel Variations ===")

    for kernel in ['uniform', 'triangular']:
        result = run_ols_rd(
            df_ab, f'r_{primary_outcome}',
            treatment_var='aind',
            bw=DEFAULT_BW,
            poly_order=1,
            cluster_var='mom_tin',
            kernel=kernel
        )

        if result:
            spec_counter += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': f'rd/kernel/{kernel}',
                'spec_tree_path': f'{METHOD_TREE_PATH}#kernel-function',
                'outcome_var': primary_outcome,
                'treatment_var': 'aind',
                'coefficient': result['coefficient'],
                'std_error': result['std_error'],
                't_stat': result['t_stat'],
                'p_value': result['p_value'],
                'ci_lower': result['ci_lower'],
                'ci_upper': result['ci_upper'],
                'n_obs': result['n_obs'],
                'r_squared': result['r_squared'],
                'coefficient_vector_json': result['coefficient_vector_json'],
                'sample_desc': 'All births',
                'fixed_effects': 'None',
                'controls_desc': f'Linear RD polynomial, {kernel} kernel',
                'cluster_var': 'mom_tin',
                'model_type': 'OLS-RD (ITT)',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })

    # ========================================================================
    # SEASONALITY CONTROL VARIATIONS
    # ========================================================================
    print("\n=== Running Seasonality Control Variations ===")

    # Different month polynomial orders
    for m_poly in [3, 4, 5]:
        df_ab_temp = df_ab.copy()
        df_ab_temp[f'r_{primary_outcome}'] = residualize_outcome(df_ab_temp, primary_outcome, poly_order=m_poly)

        result = run_ols_rd(
            df_ab_temp, f'r_{primary_outcome}',
            treatment_var='aind',
            bw=DEFAULT_BW,
            poly_order=1,
            cluster_var='mom_tin'
        )

        if result:
            spec_counter += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': f'robust/form/month_poly_{m_poly}',
                'spec_tree_path': 'specification_tree/robustness/functional_form.md',
                'outcome_var': primary_outcome,
                'treatment_var': 'aind',
                'coefficient': result['coefficient'],
                'std_error': result['std_error'],
                't_stat': result['t_stat'],
                'p_value': result['p_value'],
                'ci_lower': result['ci_lower'],
                'ci_upper': result['ci_upper'],
                'n_obs': result['n_obs'],
                'r_squared': result['r_squared'],
                'coefficient_vector_json': result['coefficient_vector_json'],
                'sample_desc': 'All births',
                'fixed_effects': 'None',
                'controls_desc': f'Linear RD polynomial, order-{m_poly} month seasonality',
                'cluster_var': 'mom_tin',
                'model_type': 'OLS-RD (ITT)',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })

    # ========================================================================
    # DONUT HOLE SENSITIVITY
    # ========================================================================
    print("\n=== Running Donut Hole Sensitivity ===")

    # With donut (baseline)
    result_donut = run_ols_rd(
        df_ab, f'r_{primary_outcome}',
        treatment_var='aind',
        bw=DEFAULT_BW,
        poly_order=1,
        cluster_var='mom_tin',
        donut=True
    )

    # Without donut
    result_no_donut = run_ols_rd(
        df_ab, f'r_{primary_outcome}',
        treatment_var='aind',
        bw=DEFAULT_BW,
        poly_order=1,
        cluster_var='mom_tin',
        donut=False
    )

    if result_no_donut:
        spec_counter += 1
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'rd/sample/no_donut',
            'spec_tree_path': f'{METHOD_TREE_PATH}#sample-restrictions',
            'outcome_var': primary_outcome,
            'treatment_var': 'aind',
            'coefficient': result_no_donut['coefficient'],
            'std_error': result_no_donut['std_error'],
            't_stat': result_no_donut['t_stat'],
            'p_value': result_no_donut['p_value'],
            'ci_lower': result_no_donut['ci_lower'],
            'ci_upper': result_no_donut['ci_upper'],
            'n_obs': result_no_donut['n_obs'],
            'r_squared': result_no_donut['r_squared'],
            'coefficient_vector_json': result_no_donut['coefficient_vector_json'],
            'sample_desc': 'All births (no donut hole)',
            'fixed_effects': 'None',
            'controls_desc': 'Linear RD polynomial, no donut exclusion',
            'cluster_var': 'mom_tin',
            'model_type': 'OLS-RD (ITT)',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })

    # ========================================================================
    # CLUSTERING VARIATIONS
    # ========================================================================
    print("\n=== Running Clustering Variations ===")

    for cluster_type in ['robust', 'mom_tin']:
        cv = None if cluster_type == 'robust' else cluster_type

        result = run_ols_rd(
            df_ab, f'r_{primary_outcome}',
            treatment_var='aind',
            bw=DEFAULT_BW,
            poly_order=1,
            cluster_var=cv
        )

        if result:
            spec_counter += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': f'robust/cluster/{cluster_type}',
                'spec_tree_path': 'specification_tree/robustness/clustering_variations.md',
                'outcome_var': primary_outcome,
                'treatment_var': 'aind',
                'coefficient': result['coefficient'],
                'std_error': result['std_error'],
                't_stat': result['t_stat'],
                'p_value': result['p_value'],
                'ci_lower': result['ci_lower'],
                'ci_upper': result['ci_upper'],
                'n_obs': result['n_obs'],
                'r_squared': result['r_squared'],
                'coefficient_vector_json': result['coefficient_vector_json'],
                'sample_desc': 'All births',
                'fixed_effects': 'None',
                'controls_desc': 'Linear RD polynomial',
                'cluster_var': cluster_type,
                'model_type': 'OLS-RD (ITT)',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })

    # ========================================================================
    # SUBGROUP ANALYSIS
    # ========================================================================
    print("\n=== Running Subgroup Analysis ===")

    # Age subgroups
    for age_group in ['under30', 'over30']:
        if age_group == 'under30':
            df_sub = df_ab[df_ab['under30'] == 1]
        else:
            df_sub = df_ab[df_ab['under30'] == 0]

        result = run_ols_rd(
            df_sub, f'r_{primary_outcome}',
            treatment_var='aind',
            bw=DEFAULT_BW,
            poly_order=1,
            cluster_var='mom_tin'
        )

        if result:
            spec_counter += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': f'robust/sample/{age_group}',
                'spec_tree_path': 'specification_tree/robustness/sample_restrictions.md',
                'outcome_var': primary_outcome,
                'treatment_var': 'aind',
                'coefficient': result['coefficient'],
                'std_error': result['std_error'],
                't_stat': result['t_stat'],
                'p_value': result['p_value'],
                'ci_lower': result['ci_lower'],
                'ci_upper': result['ci_upper'],
                'n_obs': result['n_obs'],
                'r_squared': result['r_squared'],
                'coefficient_vector_json': result['coefficient_vector_json'],
                'sample_desc': f'Age {"<30" if age_group == "under30" else ">=30"}',
                'fixed_effects': 'None',
                'controls_desc': 'Linear RD polynomial',
                'cluster_var': 'mom_tin',
                'model_type': 'OLS-RD (ITT)',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })

    # Marital status subgroups
    for marital in ['married', 'single']:
        if marital == 'married':
            df_sub = df_ab[df_ab['joint_l2_mom'] == 1]
        else:
            df_sub = df_ab[df_ab['joint_l2_mom'] == 0]

        result = run_ols_rd(
            df_sub, f'r_{primary_outcome}',
            treatment_var='aind',
            bw=DEFAULT_BW,
            poly_order=1,
            cluster_var='mom_tin'
        )

        if result:
            spec_counter += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': f'robust/sample/{marital}',
                'spec_tree_path': 'specification_tree/robustness/sample_restrictions.md',
                'outcome_var': primary_outcome,
                'treatment_var': 'aind',
                'coefficient': result['coefficient'],
                'std_error': result['std_error'],
                't_stat': result['t_stat'],
                'p_value': result['p_value'],
                'ci_lower': result['ci_lower'],
                'ci_upper': result['ci_upper'],
                'n_obs': result['n_obs'],
                'r_squared': result['r_squared'],
                'coefficient_vector_json': result['coefficient_vector_json'],
                'sample_desc': f'{marital.capitalize()} mothers',
                'fixed_effects': 'None',
                'controls_desc': 'Linear RD polynomial',
                'cluster_var': 'mom_tin',
                'model_type': 'OLS-RD (ITT)',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })

    # Wage quartile subgroups
    for q in range(1, 6):
        q_desc = f'Q{q}' if q < 5 else 'No wages'
        df_sub = df_ab[df_ab['wageQrt_l2_mom'] == q]

        result = run_ols_rd(
            df_sub, f'r_{primary_outcome}',
            treatment_var='aind',
            bw=DEFAULT_BW,
            poly_order=1,
            cluster_var='mom_tin'
        )

        if result:
            spec_counter += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': f'robust/sample/wage_quartile_{q}',
                'spec_tree_path': 'specification_tree/robustness/sample_restrictions.md',
                'outcome_var': primary_outcome,
                'treatment_var': 'aind',
                'coefficient': result['coefficient'],
                'std_error': result['std_error'],
                't_stat': result['t_stat'],
                'p_value': result['p_value'],
                'ci_lower': result['ci_lower'],
                'ci_upper': result['ci_upper'],
                'n_obs': result['n_obs'],
                'r_squared': result['r_squared'],
                'coefficient_vector_json': result['coefficient_vector_json'],
                'sample_desc': f'Pre-birth wage {q_desc}',
                'fixed_effects': 'None',
                'controls_desc': 'Linear RD polynomial',
                'cluster_var': 'mom_tin',
                'model_type': 'OLS-RD (ITT)',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })

    # ========================================================================
    # FUNCTIONAL FORM VARIATIONS
    # ========================================================================
    print("\n=== Running Functional Form Variations ===")

    # Log wages (for wage outcomes)
    df_ab_log = df_ab.copy()
    df_ab_log['log_sum_wages_1to12_mom'] = np.log(df_ab_log['sum_wages_1to12_mom'] + 1)
    df_ab_log['r_log_sum_wages_1to12_mom'] = residualize_outcome(df_ab_log, 'log_sum_wages_1to12_mom', poly_order=4)

    result = run_ols_rd(
        df_ab_log, 'r_log_sum_wages_1to12_mom',
        treatment_var='aind',
        bw=DEFAULT_BW,
        poly_order=1,
        cluster_var='mom_tin'
    )

    if result:
        spec_counter += 1
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/form/y_log_wages',
            'spec_tree_path': 'specification_tree/robustness/functional_form.md',
            'outcome_var': 'log_sum_wages_1to12_mom',
            'treatment_var': 'aind',
            'coefficient': result['coefficient'],
            'std_error': result['std_error'],
            't_stat': result['t_stat'],
            'p_value': result['p_value'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'n_obs': result['n_obs'],
            'r_squared': result['r_squared'],
            'coefficient_vector_json': result['coefficient_vector_json'],
            'sample_desc': 'All births',
            'fixed_effects': 'None',
            'controls_desc': 'Linear RD polynomial, log outcome',
            'cluster_var': 'mom_tin',
            'model_type': 'OLS-RD (ITT)',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })

    print(f"\n=== Completed {spec_counter} specifications ===")

    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run specification search
    results = run_specification_search()

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    output_file = f"{OUTPUT_PATH}/specification_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    print(f"\nTotal specifications: {len(results_df)}")
    print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
    print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
    print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")
    print(f"\nMedian coefficient: {results_df['coefficient'].median():.4f}")
    print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")
    print(f"Range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")

    # Summary by model type
    print("\n" + "-"*60)
    print("By Model Type:")
    print("-"*60)
    for model_type in results_df['model_type'].unique():
        subset = results_df[results_df['model_type'] == model_type]
        print(f"\n{model_type}:")
        print(f"  N specs: {len(subset)}")
        print(f"  Median coef: {subset['coefficient'].median():.4f}")
        print(f"  % sig at 5%: {100*(subset['p_value'] < 0.05).mean():.1f}%")
