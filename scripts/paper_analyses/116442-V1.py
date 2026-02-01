"""
Specification Search for Paper 116442-V1
"Competition and the Use of Foggy Pricing"
Authors: Miravete (AEJ Microeconomics)

This script replicates and extends the main analysis examining how competition
(duopoly vs monopoly) affects "foggy" (confusing) pricing in cellular telephone markets.
"""

import struct
import numpy as np
import pandas as pd
import json
import warnings
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

PAPER_ID = "116442-V1"
PAPER_TITLE = "Competition and the Use of Foggy Pricing"
JOURNAL = "AEJ-Microeconomics"

BASE_PATH = Path("/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search")
DATA_PATH = BASE_PATH / "data/downloads/extracted/116442-V1/20110032_Data/AEJ_Miravete_Data"
OUTPUT_PATH = BASE_PATH / "data/downloads/extracted/116442-V1"

# =============================================================================
# Data Loading Functions
# =============================================================================

def read_gauss_fmt(filepath):
    """Read a GAUSS .fmt file and return as numpy array"""
    with open(filepath, 'rb') as f:
        data = f.read()

    # Parse header - dimensions at offset 0x80, 0x84
    nrows = struct.unpack('<I', data[0x80:0x84])[0]
    ncols = struct.unpack('<I', data[0x84:0x88])[0]

    # Read the data matrix from offset 0x88 (136 bytes)
    data_start = 0x88
    n_elements = nrows * ncols
    matrix_data = struct.unpack(f'<{n_elements}d', data[data_start:data_start + n_elements * 8])
    matrix = np.array(matrix_data).reshape(nrows, ncols)

    return matrix


def get_column_names():
    """Return variable names from the GAUSS code (193 total)"""
    return [
        "SCENARIO", "MARKET", "YEAR", "DUOPOLY", "WIRELINE", "ALPHA_i", "BETA_i",
        "GAMMA_i", "C_i", "LAMBDA_i", "Z0_i", "ALPHA_j", "BETA_j", "GAMMA_j",
        "C_j", "LAMBDA_j", "Z0_j", "AP_PEAK", "AP_OFFP", "TIME", "UNOBSERV",
        "MKT_AGE", "LEAD", "BUSINESS", "COMMUTE", "TCELLS", "GROWTH", "INCOME",
        "EDUCAT", "COVERAGE", "MEDINAGE", "POVERTY", "WAGE", "ENERGY", "OPERATE",
        "RENT", "PRIME", "POPULAT", "DENSITY", "CRIME", "VIOLENT", "PROPERTY",
        "SVCRIMES", "TEMPERAT", "RAIN", "MULTIMKT", "BELL", "REGULAT", "CORRELAT",
        "CONSPLUS", "PROFITS", "WELFARE", "EXPSALE", "EXPTARF", "EXPRATE", "EXPMKUP",
        "SURFACE", "NORTH", "WEST", "BELLBELL", "INDBELL", "BELLIND", "INDIND",
        "LIN", "SNET", "CONTEL", "GTE", "VANG", "MCCAW", "USWEST",
        "CENTEL", "PACTEL", "SWBELL", "ALLTEL", "AMERTECH", "BELLATL", "NYNEX",
        "BELLSTH", "REST", "OTHER", "PREG1", "PREG2", "VARPVRTY", "ECOST",
        "PLANS_i", "PLANS_j", "FOGGY1i", "FOGGY2i", "FOGGY3i", "FOGGY4i", "FOGGY5i",
        "FOGGY6i", "FOGGY1j", "FOGGY2j", "FOGGY3j", "FOGGY4j", "FOGGY5j", "FOGGY6j",
        "FEE_1i", "PEAK_A1i", "OFFP_A1i", "PEAK_P1i", "OFFP_P1i", "FEE_2i", "PEAK_A2i",
        "OFFP_A2i", "PEAK_P2i", "OFFP_P2i", "FEE_3i", "PEAK_A3i", "OFFP_A3i", "PEAK_P3i",
        "OFFP_P3i", "FEE_4i", "PEAK_A4i", "OFFP_A4i", "PEAK_P4i", "OFFP_P4i", "FEE_5i",
        "PEAK_A5i", "OFFP_A5i", "PEAK_P5i", "OFFP_P5i", "FEE_6i", "PEAK_A6i", "OFFP_A6i",
        "PEAK_P6i", "OFFP_P6i", "FEE_1j", "PEAK_A1j", "OFFP_A1j", "PEAK_P1j", "OFFP_P1j",
        "FEE_2j", "PEAK_A2j", "OFFP_A2j", "PEAK_P2j", "OFFP_P2j", "FEE_3j", "PEAK_A3j",
        "OFFP_A3j", "PEAK_P3j", "OFFP_P3j", "FEE_4j", "PEAK_A4j", "OFFP_A4j", "PEAK_P4j",
        "OFFP_P4j", "FEE_5j", "PEAK_A5j", "OFFP_A5j", "PEAK_P5j", "OFFP_P5j", "FEE_6j",
        "PEAK_A6j", "OFFP_A6j", "PEAK_P6j", "OFFP_P6j", "NEAREND", "AP_PKOPK", "PLANit",
        "PLANjt", "PLANit_1", "PLANjt_1", "EFFPLi", "FOGGYi", "SHFOGGYi", "HHFOGGYi",
        "EFFPLj", "FOGGYj", "SHFOGGYj", "HHFOGGYj", "PHS_PL_i", "PHS_FG_i", "PHS_PL_j",
        "PHS_FG_j", "POP90", "FAM90", "HHOLD90", "AGE90", "AGE90d", "HSIZE90",
        "HSIZE90d", "TRAV90", "TRAV90d", "EDU90", "EDU90d", "INC90", "INC90d",
        "MEDINC", "PCINC", "AVGjSHFj", "AVGjHHFj"
    ]


def load_and_prepare_data(sigma_scenario=1):
    """
    Load data and apply transformations as in the GAUSS code.

    sigma_scenario: 0-7 corresponds to different uncertainty levels
        0: No uncertainty (degenerate)
        1: sigma = 0.10*Mean (default)
        2: sigma = 0.25*Mean
        ...
        7: sigma = 3.00*Mean
    """
    # Load the data file for uniform usage distribution (k=1)
    filename = f"ALLX9{sigma_scenario}_1.fmt"
    filepath = DATA_PATH / filename

    matrix = read_gauss_fmt(filepath)
    colnames = get_column_names()

    # Create DataFrame with available columns
    df = pd.DataFrame(matrix[:, :len(colnames)], columns=colnames[:matrix.shape[1]])

    # Apply transformations from GAUSS code
    # Create YEAR92 indicator (TIME >= 30)
    df['YEAR92'] = (df['TIME'] >= 30).astype(int)

    # Create AP_PKOPK as average of peak and off-peak Arrow-Pratt index
    df['AP_PKOPK'] = (df['AP_PEAK'] + df['AP_OFFP']) / 2

    # Exclusions: market age > 0 and plans > 0
    df = df[df['MKT_AGE'] > 0]
    df = df[df['PLANit'] > 0]

    # Select only wireline firms (as in original code)
    df = df[df['WIRELINE'] == 1].copy()

    # Define FOGGYi as PLANit - EFFPLi (actual - effective plans = foggy plans)
    df['FOGGYi'] = df['PLANit'] - df['EFFPLi']
    df['FOGGYi'] = df['FOGGYi'].clip(lower=0)  # Ensure non-negative

    # Calculate share of foggy plans
    df['SHFOGGYi'] = df['FOGGYi'] / df['PLANit']
    df['SHFOGGYi'] = df['SHFOGGYi'].fillna(0)

    # Define TREATMNT variable (quarters since duopoly)
    df['TREATMNT'] = np.nan
    for market in df['MARKET'].unique():
        mask = df['MARKET'] == market
        duopoly_times = df.loc[mask & (df['DUOPOLY'] == 1), 'TIME']
        if len(duopoly_times) > 0:
            treat_start = duopoly_times.min()
            df.loc[mask, 'TREATMNT'] = df.loc[mask, 'TIME'] - treat_start

    # Correct MULTIMKT to be zero in monopoly
    df['MULTIMKT'] = df['MULTIMKT'] * df['DUOPOLY']

    # Create market and time identifiers
    df['MARKET'] = df['MARKET'].astype(int)
    df['TIME'] = df['TIME'].astype(int)

    return df


# =============================================================================
# Regression Functions
# =============================================================================

def create_dummies(df, col, prefix=None, drop_first=True):
    """Create dummy variables for a categorical column"""
    if prefix is None:
        prefix = col
    dummies = pd.get_dummies(df[col], prefix=prefix, drop_first=drop_first)
    return dummies


def demean_by_group(df, var, group_var):
    """Demean a variable within groups"""
    return df[var] - df.groupby(group_var)[var].transform('mean')


def demean_twoway(df, var, group1, group2, max_iter=100, tol=1e-8):
    """
    Demean a variable by two groups iteratively (for two-way FE).
    Uses alternating projection method.
    """
    demeaned = df[var].copy()
    for _ in range(max_iter):
        old = demeaned.copy()
        # Demean by first group
        demeaned = demeaned - df.groupby(group1)[var].transform('mean') + df[var].mean()
        demeaned = demeaned - demeaned.groupby(df[group1]).transform('mean')
        # Demean by second group
        demeaned = demeaned - demeaned.groupby(df[group2]).transform('mean')
        # Check convergence
        if (np.abs(demeaned - old) < tol).all():
            break
    return demeaned


def run_ols_with_fe(df, y_var, x_vars, fe_vars=None, cluster_var=None):
    """
    Run OLS regression with fixed effects absorbed using within transformation.

    Parameters:
    -----------
    df : DataFrame
    y_var : str - dependent variable
    x_vars : list - independent variables
    fe_vars : list - fixed effect variables to absorb
    cluster_var : str - clustering variable for standard errors

    Returns:
    --------
    dict with results
    """
    # Prepare data
    df_reg = df.copy()

    # Remove any observations with missing values
    all_vars = [y_var] + x_vars + (fe_vars if fe_vars else [])
    if cluster_var:
        all_vars.append(cluster_var)
    df_reg = df_reg.dropna(subset=all_vars)

    if len(df_reg) < 10:
        return {
            'coefficient': np.nan,
            'std_error': np.nan,
            't_stat': np.nan,
            'p_value': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n_obs': 0,
            'r_squared': np.nan,
            'coefficient_vector': {},
            'success': False,
            'error': 'Insufficient observations'
        }

    # Apply within transformation for fixed effects
    if fe_vars:
        if len(fe_vars) == 2:
            # Two-way FE: use iterative demeaning
            y_demeaned = demean_twoway(df_reg, y_var, fe_vars[0], fe_vars[1])
            X_demeaned = pd.DataFrame()
            for var in x_vars:
                X_demeaned[var] = demean_twoway(df_reg, var, fe_vars[0], fe_vars[1])
        else:
            # Single FE: simple demeaning
            y_demeaned = demean_by_group(df_reg, y_var, fe_vars[0])
            X_demeaned = pd.DataFrame()
            for var in x_vars:
                X_demeaned[var] = demean_by_group(df_reg, var, fe_vars[0])

        X = X_demeaned
        y = y_demeaned
    else:
        # No FE
        X = df_reg[x_vars].copy()
        X = sm.add_constant(X)
        y = df_reg[y_var]

    # Fit OLS
    try:
        if cluster_var and cluster_var in df_reg.columns:
            # Clustered standard errors
            model = OLS(y, X)
            results = model.fit(cov_type='cluster', cov_kwds={'groups': df_reg[cluster_var]})
        else:
            # Robust standard errors
            model = OLS(y, X)
            results = model.fit(cov_type='HC1')

        # Extract results for treatment variable (first x_var)
        treat_var = x_vars[0]
        if treat_var in results.params.index:
            coef = results.params[treat_var]
            se = results.bse[treat_var]
            tstat = results.tvalues[treat_var]
            pval = results.pvalues[treat_var]
            ci = results.conf_int()
            if treat_var in ci.index:
                ci_lower, ci_upper = ci.loc[treat_var]
            else:
                ci_lower, ci_upper = np.nan, np.nan
        else:
            coef = se = tstat = pval = ci_lower = ci_upper = np.nan

        # Get all coefficients for x_vars
        coef_vector = {}
        for var in x_vars:
            if var in results.params.index:
                coef_vector[var] = {
                    'coef': float(results.params[var]),
                    'se': float(results.bse[var]),
                    'pval': float(results.pvalues[var])
                }

        # Calculate R-squared (within for FE models)
        r_sq = float(results.rsquared) if not fe_vars else 1 - (y - results.fittedvalues).var() / y.var()

        return {
            'coefficient': float(coef),
            'std_error': float(se),
            't_stat': float(tstat),
            'p_value': float(pval),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_obs': int(results.nobs),
            'r_squared': float(r_sq),
            'coefficient_vector': coef_vector,
            'success': True
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
            'coefficient_vector': {},
            'success': False,
            'error': str(e)
        }


def run_poisson(df, y_var, x_vars, cluster_var=None):
    """
    Run Poisson regression for count data.
    """
    from statsmodels.discrete.discrete_model import Poisson

    df_reg = df.copy()
    all_vars = [y_var] + x_vars
    if cluster_var:
        all_vars.append(cluster_var)
    df_reg = df_reg.dropna(subset=all_vars)

    # Ensure y is non-negative integer
    df_reg[y_var] = df_reg[y_var].clip(lower=0).round().astype(int)

    X = df_reg[x_vars]
    X = sm.add_constant(X)
    y = df_reg[y_var]

    try:
        model = Poisson(y, X)
        if cluster_var:
            results = model.fit(cov_type='cluster', cov_kwds={'groups': df_reg[cluster_var]}, disp=0)
        else:
            results = model.fit(cov_type='HC1', disp=0)

        treat_var = x_vars[0]
        coef = results.params[treat_var]
        se = results.bse[treat_var]
        tstat = results.tvalues[treat_var]
        pval = results.pvalues[treat_var]
        ci_lower, ci_upper = results.conf_int().loc[treat_var]

        coef_vector = {}
        for var in x_vars:
            if var in results.params.index:
                coef_vector[var] = {
                    'coef': float(results.params[var]),
                    'se': float(results.bse[var]),
                    'pval': float(results.pvalues[var])
                }

        return {
            'coefficient': float(coef),
            'std_error': float(se),
            't_stat': float(tstat),
            'p_value': float(pval),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_obs': int(results.nobs),
            'r_squared': float(results.prsquared),  # Pseudo R-squared
            'coefficient_vector': coef_vector,
            'success': True
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
            'coefficient_vector': {},
            'success': False,
            'error': str(e)
        }


# =============================================================================
# Specification Search
# =============================================================================

def run_specification_search():
    """Run the full specification search"""

    print("Loading data...")
    df = load_and_prepare_data(sigma_scenario=1)
    print(f"Data loaded: {len(df)} observations")
    print(f"Unique markets: {df['MARKET'].nunique()}")
    print(f"Unique time periods: {df['TIME'].nunique()}")
    print(f"Duopoly observations: {(df['DUOPOLY'] == 1).sum()}")

    results = []

    # Define treatment and outcome variables
    treatment_var = 'DUOPOLY'
    outcome_vars = ['FOGGYi', 'SHFOGGYi', 'HHFOGGYi']

    # Define control variables (from GAUSS code)
    control_vars = ['AP_PEAK', 'AP_OFFP']

    # Market characteristics that could serve as controls
    market_controls = ['MULTIMKT', 'BELL', 'REGULAT']

    # All controls
    all_controls = control_vars + market_controls

    # Fixed effects
    market_fe = 'MARKET'
    time_fe = 'TIME'

    # Define method map
    method_map = {
        "method_code": "difference_in_differences",
        "method_tree_path": "specification_tree/methods/difference_in_differences.md",
        "specs_to_run": [],
        "robustness_specs": []
    }

    spec_count = 0

    # ==========================================================================
    # 1. BASELINE SPECIFICATIONS (for each outcome)
    # ==========================================================================

    print("\nRunning baseline specifications...")

    for outcome in outcome_vars:
        spec_id = f"baseline_{outcome}"

        result = run_ols_with_fe(
            df=df,
            y_var=outcome,
            x_vars=[treatment_var] + control_vars,
            fe_vars=[market_fe, time_fe],
            cluster_var=market_fe
        )

        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': 'methods/difference_in_differences.md#baseline',
            'outcome_var': outcome,
            'treatment_var': treatment_var,
            'coefficient': result['coefficient'],
            'std_error': result['std_error'],
            't_stat': result['t_stat'],
            'p_value': result['p_value'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'n_obs': result['n_obs'],
            'r_squared': result['r_squared'],
            'coefficient_vector_json': json.dumps(result['coefficient_vector']),
            'sample_desc': 'Wireline firms only',
            'fixed_effects': 'Market + Time FE',
            'controls_desc': ', '.join(control_vars),
            'cluster_var': market_fe,
            'model_type': 'OLS with two-way FE',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })
        spec_count += 1
        print(f"  {spec_id}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # ==========================================================================
    # 2. FIXED EFFECTS VARIATIONS
    # ==========================================================================

    print("\nRunning FE variations...")

    fe_specs = [
        ('did/fe/unit_only', [market_fe], 'Market FE only'),
        ('did/fe/time_only', [time_fe], 'Time FE only'),
        ('did/fe/twoway', [market_fe, time_fe], 'Market + Time FE'),
        ('did/fe/none', None, 'No FE (pooled OLS)'),
    ]

    for outcome in outcome_vars:
        for spec_id, fe_vars, fe_desc in fe_specs:
            result = run_ols_with_fe(
                df=df,
                y_var=outcome,
                x_vars=[treatment_var] + control_vars,
                fe_vars=fe_vars,
                cluster_var=market_fe if fe_vars else None
            )

            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': f"{spec_id}_{outcome}",
                'spec_tree_path': 'methods/difference_in_differences.md#fixed-effects',
                'outcome_var': outcome,
                'treatment_var': treatment_var,
                'coefficient': result['coefficient'],
                'std_error': result['std_error'],
                't_stat': result['t_stat'],
                'p_value': result['p_value'],
                'ci_lower': result['ci_lower'],
                'ci_upper': result['ci_upper'],
                'n_obs': result['n_obs'],
                'r_squared': result['r_squared'],
                'coefficient_vector_json': json.dumps(result['coefficient_vector']),
                'sample_desc': 'Wireline firms only',
                'fixed_effects': fe_desc,
                'controls_desc': ', '.join(control_vars),
                'cluster_var': market_fe if fe_vars else 'None',
                'model_type': 'OLS',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })
            spec_count += 1

    # ==========================================================================
    # 3. CONTROL SET VARIATIONS
    # ==========================================================================

    print("\nRunning control variations...")

    control_specs = [
        ('did/controls/none', [], 'No controls'),
        ('did/controls/minimal', control_vars[:1], 'AP_PEAK only'),
        ('did/controls/baseline', control_vars, 'AP_PEAK, AP_OFFP'),
        ('did/controls/full', all_controls, 'All controls'),
    ]

    for outcome in outcome_vars:
        for spec_id, controls, ctrl_desc in control_specs:
            x_vars = [treatment_var] + controls if controls else [treatment_var]

            result = run_ols_with_fe(
                df=df,
                y_var=outcome,
                x_vars=x_vars,
                fe_vars=[market_fe, time_fe],
                cluster_var=market_fe
            )

            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': f"{spec_id}_{outcome}",
                'spec_tree_path': 'methods/difference_in_differences.md#control-sets',
                'outcome_var': outcome,
                'treatment_var': treatment_var,
                'coefficient': result['coefficient'],
                'std_error': result['std_error'],
                't_stat': result['t_stat'],
                'p_value': result['p_value'],
                'ci_lower': result['ci_lower'],
                'ci_upper': result['ci_upper'],
                'n_obs': result['n_obs'],
                'r_squared': result['r_squared'],
                'coefficient_vector_json': json.dumps(result['coefficient_vector']),
                'sample_desc': 'Wireline firms only',
                'fixed_effects': 'Market + Time FE',
                'controls_desc': ctrl_desc,
                'cluster_var': market_fe,
                'model_type': 'OLS with two-way FE',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })
            spec_count += 1

    # ==========================================================================
    # 4. SAMPLE RESTRICTIONS
    # ==========================================================================

    print("\nRunning sample restriction variations...")

    # Early period (first half)
    df_early = df[df['TIME'] <= df['TIME'].median()].copy()
    # Late period (second half)
    df_late = df[df['TIME'] > df['TIME'].median()].copy()
    # Exclude always-treated (markets that start in duopoly)
    always_treated = df.groupby('MARKET')['DUOPOLY'].min()
    never_monopoly = always_treated[always_treated == 1].index
    df_exclude_always = df[~df['MARKET'].isin(never_monopoly)].copy()

    sample_specs = [
        ('did/sample/full', df, 'Full sample'),
        ('did/sample/early_period', df_early, 'First half of sample period'),
        ('did/sample/late_period', df_late, 'Second half of sample period'),
        ('did/sample/exclude_always_treated', df_exclude_always, 'Exclude always-treated markets'),
    ]

    for outcome in outcome_vars:
        for spec_id, df_sample, sample_desc in sample_specs:
            result = run_ols_with_fe(
                df=df_sample,
                y_var=outcome,
                x_vars=[treatment_var] + control_vars,
                fe_vars=[market_fe, time_fe],
                cluster_var=market_fe
            )

            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': f"{spec_id}_{outcome}",
                'spec_tree_path': 'methods/difference_in_differences.md#sample-restrictions',
                'outcome_var': outcome,
                'treatment_var': treatment_var,
                'coefficient': result['coefficient'],
                'std_error': result['std_error'],
                't_stat': result['t_stat'],
                'p_value': result['p_value'],
                'ci_lower': result['ci_lower'],
                'ci_upper': result['ci_upper'],
                'n_obs': result['n_obs'],
                'r_squared': result['r_squared'],
                'coefficient_vector_json': json.dumps(result['coefficient_vector']),
                'sample_desc': sample_desc,
                'fixed_effects': 'Market + Time FE',
                'controls_desc': ', '.join(control_vars),
                'cluster_var': market_fe,
                'model_type': 'OLS with two-way FE',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })
            spec_count += 1

    # ==========================================================================
    # 5. FUNCTIONAL FORM - POISSON (for count outcome FOGGYi)
    # ==========================================================================

    print("\nRunning Poisson regression for count outcome...")

    # Poisson for FOGGYi (count outcome)
    outcome = 'FOGGYi'
    result = run_poisson(
        df=df,
        y_var=outcome,
        x_vars=[treatment_var] + control_vars,
        cluster_var=market_fe
    )

    results.append({
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': f"robust/form/poisson_{outcome}",
        'spec_tree_path': 'robustness/functional_form.md#alternative-estimators',
        'outcome_var': outcome,
        'treatment_var': treatment_var,
        'coefficient': result['coefficient'],
        'std_error': result['std_error'],
        't_stat': result['t_stat'],
        'p_value': result['p_value'],
        'ci_lower': result['ci_lower'],
        'ci_upper': result['ci_upper'],
        'n_obs': result['n_obs'],
        'r_squared': result['r_squared'],
        'coefficient_vector_json': json.dumps(result['coefficient_vector']),
        'sample_desc': 'Wireline firms only',
        'fixed_effects': 'None (Poisson)',
        'controls_desc': ', '.join(control_vars),
        'cluster_var': market_fe,
        'model_type': 'Poisson PML',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })
    spec_count += 1

    # ==========================================================================
    # 6. LOG TRANSFORMATION
    # ==========================================================================

    print("\nRunning log transformation specifications...")

    for outcome in outcome_vars:
        # Create log-transformed outcome
        df[f'{outcome}_log'] = np.log(df[outcome] + 0.1)

        result = run_ols_with_fe(
            df=df,
            y_var=f'{outcome}_log',
            x_vars=[treatment_var] + control_vars,
            fe_vars=[market_fe, time_fe],
            cluster_var=market_fe
        )

        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': f"robust/form/y_log_{outcome}",
            'spec_tree_path': 'robustness/functional_form.md#outcome-variable-transformations',
            'outcome_var': f'{outcome}_log',
            'treatment_var': treatment_var,
            'coefficient': result['coefficient'],
            'std_error': result['std_error'],
            't_stat': result['t_stat'],
            'p_value': result['p_value'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'n_obs': result['n_obs'],
            'r_squared': result['r_squared'],
            'coefficient_vector_json': json.dumps(result['coefficient_vector']),
            'sample_desc': 'Wireline firms only',
            'fixed_effects': 'Market + Time FE',
            'controls_desc': ', '.join(control_vars),
            'cluster_var': market_fe,
            'model_type': 'OLS with two-way FE (log outcome)',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })
        spec_count += 1

    # ==========================================================================
    # 7. LEAVE-ONE-OUT ROBUSTNESS
    # ==========================================================================

    print("\nRunning leave-one-out robustness checks...")

    outcome = 'FOGGYi'  # Primary outcome

    for drop_var in control_vars:
        remaining_controls = [c for c in control_vars if c != drop_var]

        result = run_ols_with_fe(
            df=df,
            y_var=outcome,
            x_vars=[treatment_var] + remaining_controls,
            fe_vars=[market_fe, time_fe],
            cluster_var=market_fe
        )

        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': f"robust/loo/drop_{drop_var}",
            'spec_tree_path': 'robustness/leave_one_out.md',
            'outcome_var': outcome,
            'treatment_var': treatment_var,
            'coefficient': result['coefficient'],
            'std_error': result['std_error'],
            't_stat': result['t_stat'],
            'p_value': result['p_value'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'n_obs': result['n_obs'],
            'r_squared': result['r_squared'],
            'coefficient_vector_json': json.dumps(result['coefficient_vector']),
            'sample_desc': 'Wireline firms only',
            'fixed_effects': 'Market + Time FE',
            'controls_desc': f'Dropped: {drop_var}',
            'cluster_var': market_fe,
            'model_type': 'OLS with two-way FE',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })
        spec_count += 1

    # ==========================================================================
    # 8. CLUSTERING VARIATIONS
    # ==========================================================================

    print("\nRunning clustering variations...")

    outcome = 'FOGGYi'

    cluster_specs = [
        ('robust/cluster/none', None, 'Robust (no clustering)'),
        ('robust/cluster/unit', market_fe, 'Cluster by market'),
        ('robust/cluster/time', time_fe, 'Cluster by time'),
    ]

    for spec_id, cluster, cluster_desc in cluster_specs:
        result = run_ols_with_fe(
            df=df,
            y_var=outcome,
            x_vars=[treatment_var] + control_vars,
            fe_vars=[market_fe, time_fe],
            cluster_var=cluster
        )

        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': 'robustness/clustering_variations.md',
            'outcome_var': outcome,
            'treatment_var': treatment_var,
            'coefficient': result['coefficient'],
            'std_error': result['std_error'],
            't_stat': result['t_stat'],
            'p_value': result['p_value'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'n_obs': result['n_obs'],
            'r_squared': result['r_squared'],
            'coefficient_vector_json': json.dumps(result['coefficient_vector']),
            'sample_desc': 'Wireline firms only',
            'fixed_effects': 'Market + Time FE',
            'controls_desc': ', '.join(control_vars),
            'cluster_var': cluster_desc,
            'model_type': 'OLS with two-way FE',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })
        spec_count += 1

    # ==========================================================================
    # 9. DYNAMIC TREATMENT EFFECTS (Event Study)
    # ==========================================================================

    print("\nRunning event study specifications...")

    outcome = 'FOGGYi'

    # Create treatment timing dummies
    df['TREAT_POST'] = (df['TREATMNT'] >= 0).astype(int) * df['DUOPOLY']

    # Create period dummies for event study
    for lag in range(7):  # 0 to 6 periods after treatment
        df[f'TREAT_{lag}'] = ((df['TREATMNT'] == lag) & (df['DUOPOLY'] == 1)).astype(int)

    # Event study with individual period dummies
    event_vars = [f'TREAT_{i}' for i in range(7)]

    result = run_ols_with_fe(
        df=df,
        y_var=outcome,
        x_vars=event_vars + control_vars,
        fe_vars=[market_fe, time_fe],
        cluster_var=market_fe
    )

    results.append({
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': 'did/dynamic/leads_lags',
        'spec_tree_path': 'methods/difference_in_differences.md#dynamic-effects',
        'outcome_var': outcome,
        'treatment_var': 'Event study dummies (TREAT_0 to TREAT_6)',
        'coefficient': result['coefficient_vector'].get('TREAT_0', {}).get('coef', np.nan),
        'std_error': result['coefficient_vector'].get('TREAT_0', {}).get('se', np.nan),
        't_stat': result['t_stat'],
        'p_value': result['coefficient_vector'].get('TREAT_0', {}).get('pval', np.nan),
        'ci_lower': result['ci_lower'],
        'ci_upper': result['ci_upper'],
        'n_obs': result['n_obs'],
        'r_squared': result['r_squared'],
        'coefficient_vector_json': json.dumps(result['coefficient_vector']),
        'sample_desc': 'Wireline firms only',
        'fixed_effects': 'Market + Time FE',
        'controls_desc': ', '.join(control_vars),
        'cluster_var': market_fe,
        'model_type': 'Event study with two-way FE',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })
    spec_count += 1

    # ==========================================================================
    # 10. DIFFERENT UNCERTAINTY SCENARIOS
    # ==========================================================================

    print("\nRunning different uncertainty scenario specifications...")

    for sigma in [0, 2, 4, 7]:  # Different uncertainty levels
        try:
            df_sigma = load_and_prepare_data(sigma_scenario=sigma)

            result = run_ols_with_fe(
                df=df_sigma,
                y_var='FOGGYi',
                x_vars=[treatment_var] + control_vars,
                fe_vars=[market_fe, time_fe],
                cluster_var=market_fe
            )

            sigma_desc = {0: 'No uncertainty', 2: 'sigma=0.25*Mean',
                         4: 'sigma=1.00*Mean', 7: 'sigma=3.00*Mean'}

            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': f'custom/uncertainty_sigma{sigma}',
                'spec_tree_path': 'custom',
                'outcome_var': 'FOGGYi',
                'treatment_var': treatment_var,
                'coefficient': result['coefficient'],
                'std_error': result['std_error'],
                't_stat': result['t_stat'],
                'p_value': result['p_value'],
                'ci_lower': result['ci_lower'],
                'ci_upper': result['ci_upper'],
                'n_obs': result['n_obs'],
                'r_squared': result['r_squared'],
                'coefficient_vector_json': json.dumps(result['coefficient_vector']),
                'sample_desc': f'Wireline firms, {sigma_desc.get(sigma, f"sigma scenario {sigma}")}',
                'fixed_effects': 'Market + Time FE',
                'controls_desc': ', '.join(control_vars),
                'cluster_var': market_fe,
                'model_type': 'OLS with two-way FE',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })
            spec_count += 1
        except Exception as e:
            print(f"  Warning: Could not load sigma scenario {sigma}: {e}")

    print(f"\nTotal specifications run: {spec_count}")

    return results


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(f"Specification Search for {PAPER_ID}")
    print(f"{PAPER_TITLE}")
    print("=" * 70)

    # Run specification search
    results = run_specification_search()

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    output_file = OUTPUT_PATH / "specification_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    # Filter to valid results
    valid_results = results_df[results_df['coefficient'].notna()].copy()

    print(f"\nTotal specifications: {len(results_df)}")
    print(f"Valid specifications: {len(valid_results)}")

    if len(valid_results) > 0:
        print(f"\nPositive coefficients: {(valid_results['coefficient'] > 0).sum()} ({100*(valid_results['coefficient'] > 0).mean():.1f}%)")
        print(f"Significant at 5%: {(valid_results['p_value'] < 0.05).sum()} ({100*(valid_results['p_value'] < 0.05).mean():.1f}%)")
        print(f"Significant at 1%: {(valid_results['p_value'] < 0.01).sum()} ({100*(valid_results['p_value'] < 0.01).mean():.1f}%)")
        print(f"\nMedian coefficient: {valid_results['coefficient'].median():.4f}")
        print(f"Mean coefficient: {valid_results['coefficient'].mean():.4f}")
        print(f"Range: [{valid_results['coefficient'].min():.4f}, {valid_results['coefficient'].max():.4f}]")

    print("\nDone!")
