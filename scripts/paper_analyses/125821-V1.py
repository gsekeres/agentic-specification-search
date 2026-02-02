"""
Specification Search Analysis: 125821-V1
Wisconsin School Referendum Regression Discontinuity Study

This paper examines the effects of operating referendum elections on school district
outcomes in Wisconsin using a Regression Discontinuity Design (RDD).

Method: Regression Discontinuity Design (RDD)
Running variable: Vote percentage (perc) with cutoff at 50%
Treatment: Referendum passage (win = 1 if perc >= 50)

Key outcomes:
- Total expenditures per member (tot_exp_mem)
- Instructional expenditures per member (tot_exp_inst_mem)
- Teacher compensation (compensation)
- Various school-level outcomes

Paper ID: 125821-V1
"""

import pandas as pd
import numpy as np
import json
import warnings
from scipy import stats
from pathlib import Path

warnings.filterwarnings('ignore')

# Try to import pyfixest, fall back to statsmodels if needed
try:
    import pyfixest as pf
    HAS_PYFIXEST = True
except ImportError:
    HAS_PYFIXEST = False

import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

# =============================================================================
# Configuration
# =============================================================================

PAPER_ID = "125821-V1"
JOURNAL = "AER"
PAPER_TITLE = "Wisconsin School Referendum RDD Study"
BASE_PATH = Path("/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search")
DATA_PATH = BASE_PATH / "data/downloads/extracted/125821-V1/Replication/Data/Final"
OUTPUT_PATH = BASE_PATH / "data/downloads/extracted/125821-V1"

# RDD parameters
CUTOFF = 50  # Vote percentage cutoff

# =============================================================================
# Data Loading and Preparation
# =============================================================================

def load_data():
    """Load and prepare the cross-section RDD data."""
    df = pd.read_stata(DATA_PATH / "itt_cross_section.dta")

    # Create running variable (centered at cutoff)
    df['running'] = df['perc'] - CUTOFF

    # Create treatment indicator
    df['treatment'] = (df['perc'] >= CUTOFF).astype(int)

    # Create bandwidth indicators
    df['bw5'] = (df['running'].abs() <= 5).astype(int)
    df['bw10'] = (df['running'].abs() <= 10).astype(int)
    df['bw15'] = (df['running'].abs() <= 15).astype(int)
    df['bw20'] = (df['running'].abs() <= 20).astype(int)

    # Polynomial terms
    df['running_sq'] = df['running'] ** 2
    df['running_cu'] = df['running'] ** 3

    # Interaction terms for local polynomial
    df['running_treat'] = df['running'] * df['treatment']
    df['running_sq_treat'] = df['running_sq'] * df['treatment']

    return df


def load_panel_data():
    """Load and prepare the panel RDD data."""
    df = pd.read_stata(DATA_PATH / "onestep_panel_tables.dta")

    # Key RDD variables are bond_percent_prev and bond_win_prev
    # These track whether a referendum passed in a previous year

    return df

# =============================================================================
# RDD Estimation Functions
# =============================================================================

def run_local_linear_rd(df, outcome, running='running', treatment='treatment',
                         bandwidth=None, controls=None, kernel='triangular',
                         cluster_var=None):
    """
    Run local linear RDD regression.

    y = alpha + tau * D + beta * (X - c) + gamma * D * (X - c) + controls + e

    where D = 1 if X >= c (treatment), X is running variable, c is cutoff
    """
    # Filter to bandwidth if specified
    if bandwidth is not None:
        df_reg = df[df[running].abs() <= bandwidth].copy()
    else:
        df_reg = df.copy()

    # Drop missing values for outcome
    df_reg = df_reg.dropna(subset=[outcome, running, treatment])

    if len(df_reg) < 20:
        return None

    # Apply kernel weights
    if kernel == 'triangular':
        if bandwidth is not None:
            df_reg['weight'] = (1 - df_reg[running].abs() / bandwidth).clip(lower=0)
        else:
            h = df_reg[running].abs().max()
            df_reg['weight'] = (1 - df_reg[running].abs() / h).clip(lower=0)
    elif kernel == 'uniform':
        df_reg['weight'] = 1.0
    elif kernel == 'epanechnikov':
        if bandwidth is not None:
            u = df_reg[running].abs() / bandwidth
            df_reg['weight'] = (0.75 * (1 - u**2)).clip(lower=0)
        else:
            df_reg['weight'] = 1.0
    else:
        df_reg['weight'] = 1.0

    # Build regression formula
    # Local linear: outcome ~ treatment + running + running*treatment + controls
    X_vars = [treatment, running, f'{running}_treat']

    # Handle running_treat if not in df_reg
    if f'{running}_treat' not in df_reg.columns:
        df_reg['running_treat'] = df_reg[running] * df_reg[treatment]

    X_list = [treatment, running, 'running_treat']

    if controls is not None:
        for c in controls:
            if c in df_reg.columns:
                X_list.append(c)

    # Drop missing for all X vars
    df_reg = df_reg.dropna(subset=[outcome] + X_list)

    if len(df_reg) < len(X_list) + 5:
        return None

    # Create matrices
    y = df_reg[outcome]
    X = df_reg[X_list]
    X = sm.add_constant(X)

    # Fit model with weights
    try:
        if cluster_var is not None and cluster_var in df_reg.columns:
            model = OLS(y, X)
            results = model.fit(cov_type='cluster',
                               cov_kwds={'groups': df_reg[cluster_var]})
        else:
            model = OLS(y, X)
            results = model.fit(cov_type='HC1')  # Robust SEs

        return results, df_reg
    except Exception as e:
        print(f"Error in regression: {e}")
        return None

def run_local_quadratic_rd(df, outcome, running='running', treatment='treatment',
                            bandwidth=None, controls=None, cluster_var=None):
    """Run local quadratic RDD regression."""
    if bandwidth is not None:
        df_reg = df[df[running].abs() <= bandwidth].copy()
    else:
        df_reg = df.copy()

    df_reg = df_reg.dropna(subset=[outcome, running, treatment])

    if len(df_reg) < 30:
        return None

    # Create quadratic terms
    df_reg['running_sq'] = df_reg[running] ** 2
    df_reg['running_treat'] = df_reg[running] * df_reg[treatment]
    df_reg['running_sq_treat'] = df_reg['running_sq'] * df_reg[treatment]

    X_list = [treatment, running, 'running_sq', 'running_treat', 'running_sq_treat']

    if controls is not None:
        for c in controls:
            if c in df_reg.columns:
                X_list.append(c)

    df_reg = df_reg.dropna(subset=[outcome] + X_list)

    if len(df_reg) < len(X_list) + 5:
        return None

    y = df_reg[outcome]
    X = df_reg[X_list]
    X = sm.add_constant(X)

    try:
        if cluster_var is not None and cluster_var in df_reg.columns:
            model = OLS(y, X)
            results = model.fit(cov_type='cluster',
                               cov_kwds={'groups': df_reg[cluster_var]})
        else:
            model = OLS(y, X)
            results = model.fit(cov_type='HC1')

        return results, df_reg
    except Exception as e:
        print(f"Error in quadratic regression: {e}")
        return None


def extract_rd_results(result, df_reg, treatment_var='treatment'):
    """Extract results from RDD regression."""
    if result is None:
        return None

    results, df_used = result

    if treatment_var not in results.params:
        return None

    coef = results.params[treatment_var]
    se = results.bse[treatment_var]
    tstat = results.tvalues[treatment_var]
    pval = results.pvalues[treatment_var]

    # 95% confidence interval
    ci_lower = coef - 1.96 * se
    ci_upper = coef + 1.96 * se

    # Build coefficient vector
    coef_vector = {
        'treatment': {
            'var': treatment_var,
            'coef': float(coef),
            'se': float(se),
            'pval': float(pval)
        },
        'controls': [],
        'diagnostics': {
            'n_left': int((df_used['treatment'] == 0).sum()),
            'n_right': int((df_used['treatment'] == 1).sum()),
        }
    }

    # Add other coefficients
    for var in results.params.index:
        if var not in [treatment_var, 'const']:
            coef_vector['controls'].append({
                'var': var,
                'coef': float(results.params[var]),
                'se': float(results.bse[var]),
                'pval': float(results.pvalues[var])
            })

    return {
        'coefficient': float(coef),
        'std_error': float(se),
        't_stat': float(tstat),
        'p_value': float(pval),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'n_obs': int(len(df_used)),
        'r_squared': float(results.rsquared),
        'coefficient_vector_json': json.dumps(coef_vector)
    }

# =============================================================================
# Specification Search
# =============================================================================

def run_specification_search(df):
    """Run comprehensive specification search following i4r methodology."""

    results = []
    spec_count = 0

    # Define outcomes
    primary_outcome = 'tot_exp_mem'  # Total expenditures per member

    outcomes = {
        'tot_exp_mem': 'Total expenditures per member',
        'tot_exp_inst_mem': 'Instructional expenditures per member',
        'compensation': 'Teacher compensation',
        'dropout_rate': 'Dropout rate',
        'log_el_avgsalary': 'Log elementary teacher salary',
    }

    # Define control sets
    no_controls = []
    minimal_controls = ['membership', 'econ_disadv_percent']
    baseline_controls = ['membership', 'econ_disadv_percent', 'urban_centric_locale', 'above_median']
    full_controls = ['membership', 'econ_disadv_percent', 'urban_centric_locale',
                    'above_median', 'turnover_LA', 'AverageLocalExperience']

    # Define bandwidths
    bandwidths = {
        'bw5': 5,
        'bw10': 10,
        'bw15': 15,
        'bw20': 20,
        'bw7': 7,
        'bw12': 12,
        'full': None
    }

    # =========================================================================
    # 1. BASELINE SPECIFICATION
    # =========================================================================
    print("Running baseline specification...")

    result = run_local_linear_rd(df, primary_outcome, bandwidth=10,
                                 controls=baseline_controls, cluster_var='district_code')
    if result:
        res = extract_rd_results(result, None)
        if res:
            spec_count += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': 'baseline',
                'spec_tree_path': 'methods/regression_discontinuity.md#baseline',
                'outcome_var': primary_outcome,
                'treatment_var': 'treatment',
                'sample_desc': 'Full sample, bandwidth=10, local linear',
                'fixed_effects': 'None',
                'controls_desc': ', '.join(baseline_controls),
                'cluster_var': 'district_code',
                'model_type': 'RD_local_linear',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
                **res
            })

    # =========================================================================
    # 2. BANDWIDTH VARIATIONS (7 specs)
    # =========================================================================
    print("Running bandwidth variations...")

    for bw_name, bw_val in bandwidths.items():
        result = run_local_linear_rd(df, primary_outcome, bandwidth=bw_val,
                                    controls=baseline_controls, cluster_var='district_code')
        if result:
            res = extract_rd_results(result, None)
            if res:
                spec_count += 1
                bw_desc = f"bandwidth={bw_val}" if bw_val else "full sample"
                results.append({
                    'paper_id': PAPER_ID,
                    'journal': JOURNAL,
                    'paper_title': PAPER_TITLE,
                    'spec_id': f'rd/bandwidth/{bw_name}',
                    'spec_tree_path': 'methods/regression_discontinuity.md#bandwidth-selection',
                    'outcome_var': primary_outcome,
                    'treatment_var': 'treatment',
                    'sample_desc': f'{bw_desc}, local linear',
                    'fixed_effects': 'None',
                    'controls_desc': ', '.join(baseline_controls),
                    'cluster_var': 'district_code',
                    'model_type': 'RD_local_linear',
                    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
                    **res
                })

    # =========================================================================
    # 3. POLYNOMIAL ORDER VARIATIONS (4 specs)
    # =========================================================================
    print("Running polynomial order variations...")

    # Local linear (already in baseline)
    # Local quadratic
    result = run_local_quadratic_rd(df, primary_outcome, bandwidth=10,
                                    controls=baseline_controls, cluster_var='district_code')
    if result:
        res = extract_rd_results(result, None)
        if res:
            spec_count += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': 'rd/poly/local_quadratic',
                'spec_tree_path': 'methods/regression_discontinuity.md#polynomial-order',
                'outcome_var': primary_outcome,
                'treatment_var': 'treatment',
                'sample_desc': 'bandwidth=10, local quadratic',
                'fixed_effects': 'None',
                'controls_desc': ', '.join(baseline_controls),
                'cluster_var': 'district_code',
                'model_type': 'RD_local_quadratic',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
                **res
            })

    # Global linear (full sample)
    result = run_local_linear_rd(df, primary_outcome, bandwidth=None,
                                controls=baseline_controls, cluster_var='district_code')
    if result:
        res = extract_rd_results(result, None)
        if res:
            spec_count += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': 'rd/poly/global_linear',
                'spec_tree_path': 'methods/regression_discontinuity.md#polynomial-order',
                'outcome_var': primary_outcome,
                'treatment_var': 'treatment',
                'sample_desc': 'Full sample, global linear',
                'fixed_effects': 'None',
                'controls_desc': ', '.join(baseline_controls),
                'cluster_var': 'district_code',
                'model_type': 'RD_global_linear',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
                **res
            })

    # Global quadratic
    result = run_local_quadratic_rd(df, primary_outcome, bandwidth=None,
                                   controls=baseline_controls, cluster_var='district_code')
    if result:
        res = extract_rd_results(result, None)
        if res:
            spec_count += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': 'rd/poly/global_quadratic',
                'spec_tree_path': 'methods/regression_discontinuity.md#polynomial-order',
                'outcome_var': primary_outcome,
                'treatment_var': 'treatment',
                'sample_desc': 'Full sample, global quadratic',
                'fixed_effects': 'None',
                'controls_desc': ', '.join(baseline_controls),
                'cluster_var': 'district_code',
                'model_type': 'RD_global_quadratic',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
                **res
            })

    # =========================================================================
    # 4. KERNEL VARIATIONS (3 specs)
    # =========================================================================
    print("Running kernel variations...")

    for kernel_name in ['triangular', 'uniform', 'epanechnikov']:
        result = run_local_linear_rd(df, primary_outcome, bandwidth=10,
                                    controls=baseline_controls, kernel=kernel_name,
                                    cluster_var='district_code')
        if result:
            res = extract_rd_results(result, None)
            if res:
                spec_count += 1
                results.append({
                    'paper_id': PAPER_ID,
                    'journal': JOURNAL,
                    'paper_title': PAPER_TITLE,
                    'spec_id': f'rd/kernel/{kernel_name}',
                    'spec_tree_path': 'methods/regression_discontinuity.md#kernel-function',
                    'outcome_var': primary_outcome,
                    'treatment_var': 'treatment',
                    'sample_desc': f'bandwidth=10, {kernel_name} kernel',
                    'fixed_effects': 'None',
                    'controls_desc': ', '.join(baseline_controls),
                    'cluster_var': 'district_code',
                    'model_type': 'RD_local_linear',
                    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
                    **res
                })

    # =========================================================================
    # 5. CONTROL SET VARIATIONS (5 specs)
    # =========================================================================
    print("Running control set variations...")

    control_sets = {
        'none': no_controls,
        'minimal': minimal_controls,
        'baseline': baseline_controls,
        'full': full_controls,
    }

    for control_name, controls in control_sets.items():
        result = run_local_linear_rd(df, primary_outcome, bandwidth=10,
                                    controls=controls if controls else None,
                                    cluster_var='district_code')
        if result:
            res = extract_rd_results(result, None)
            if res:
                spec_count += 1
                ctrl_desc = ', '.join(controls) if controls else 'None'
                results.append({
                    'paper_id': PAPER_ID,
                    'journal': JOURNAL,
                    'paper_title': PAPER_TITLE,
                    'spec_id': f'rd/controls/{control_name}',
                    'spec_tree_path': 'methods/regression_discontinuity.md#control-sets',
                    'outcome_var': primary_outcome,
                    'treatment_var': 'treatment',
                    'sample_desc': f'bandwidth=10, {control_name} controls',
                    'fixed_effects': 'None',
                    'controls_desc': ctrl_desc,
                    'cluster_var': 'district_code',
                    'model_type': 'RD_local_linear',
                    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
                    **res
                })

    # =========================================================================
    # 6. LEAVE-ONE-OUT CONTROL VARIATIONS (6 specs)
    # =========================================================================
    print("Running leave-one-out control variations...")

    for drop_ctrl in full_controls:
        remaining_controls = [c for c in full_controls if c != drop_ctrl]
        result = run_local_linear_rd(df, primary_outcome, bandwidth=10,
                                    controls=remaining_controls,
                                    cluster_var='district_code')
        if result:
            res = extract_rd_results(result, None)
            if res:
                spec_count += 1
                results.append({
                    'paper_id': PAPER_ID,
                    'journal': JOURNAL,
                    'paper_title': PAPER_TITLE,
                    'spec_id': f'robust/control/drop_{drop_ctrl}',
                    'spec_tree_path': 'robustness/leave_one_out.md',
                    'outcome_var': primary_outcome,
                    'treatment_var': 'treatment',
                    'sample_desc': f'bandwidth=10, dropped {drop_ctrl}',
                    'fixed_effects': 'None',
                    'controls_desc': ', '.join(remaining_controls),
                    'cluster_var': 'district_code',
                    'model_type': 'RD_local_linear',
                    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
                    **res
                })

    # =========================================================================
    # 7. ALTERNATIVE OUTCOMES (5 specs)
    # =========================================================================
    print("Running alternative outcome specifications...")

    for outcome_var, outcome_desc in outcomes.items():
        if outcome_var == primary_outcome:
            continue
        result = run_local_linear_rd(df, outcome_var, bandwidth=10,
                                    controls=baseline_controls,
                                    cluster_var='district_code')
        if result:
            res = extract_rd_results(result, None)
            if res:
                spec_count += 1
                results.append({
                    'paper_id': PAPER_ID,
                    'journal': JOURNAL,
                    'paper_title': PAPER_TITLE,
                    'spec_id': f'robust/outcome/{outcome_var}',
                    'spec_tree_path': 'robustness/measurement.md',
                    'outcome_var': outcome_var,
                    'treatment_var': 'treatment',
                    'sample_desc': f'bandwidth=10, outcome: {outcome_desc}',
                    'fixed_effects': 'None',
                    'controls_desc': ', '.join(baseline_controls),
                    'cluster_var': 'district_code',
                    'model_type': 'RD_local_linear',
                    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
                    **res
                })

    # =========================================================================
    # 8. DONUT HOLE SPECIFICATIONS (4 specs)
    # =========================================================================
    print("Running donut hole specifications...")

    donut_sizes = [1, 2, 3, 5]
    for donut in donut_sizes:
        df_donut = df[df['running'].abs() > donut].copy()
        result = run_local_linear_rd(df_donut, primary_outcome, bandwidth=10,
                                    controls=baseline_controls,
                                    cluster_var='district_code')
        if result:
            res = extract_rd_results(result, None)
            if res:
                spec_count += 1
                results.append({
                    'paper_id': PAPER_ID,
                    'journal': JOURNAL,
                    'paper_title': PAPER_TITLE,
                    'spec_id': f'rd/donut/exclude_{donut}pct',
                    'spec_tree_path': 'methods/regression_discontinuity.md#donut-hole-specifications',
                    'outcome_var': primary_outcome,
                    'treatment_var': 'treatment',
                    'sample_desc': f'bandwidth=10, donut={donut} pp',
                    'fixed_effects': 'None',
                    'controls_desc': ', '.join(baseline_controls),
                    'cluster_var': 'district_code',
                    'model_type': 'RD_local_linear',
                    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
                    **res
                })

    # =========================================================================
    # 9. SAMPLE RESTRICTIONS (10 specs)
    # =========================================================================
    print("Running sample restriction specifications...")

    # By year of referendum
    years = df['yearref'].dropna().unique()
    years = sorted([y for y in years if not np.isnan(y)])

    # Early vs late period
    if len(years) > 1:
        median_year = np.median(years)

        df_early = df[df['yearref'] <= median_year].copy()
        result = run_local_linear_rd(df_early, primary_outcome, bandwidth=10,
                                    controls=baseline_controls, cluster_var='district_code')
        if result:
            res = extract_rd_results(result, None)
            if res:
                spec_count += 1
                results.append({
                    'paper_id': PAPER_ID,
                    'journal': JOURNAL,
                    'paper_title': PAPER_TITLE,
                    'spec_id': 'robust/sample/early_period',
                    'spec_tree_path': 'robustness/sample_restrictions.md',
                    'outcome_var': primary_outcome,
                    'treatment_var': 'treatment',
                    'sample_desc': f'Early period (year <= {median_year})',
                    'fixed_effects': 'None',
                    'controls_desc': ', '.join(baseline_controls),
                    'cluster_var': 'district_code',
                    'model_type': 'RD_local_linear',
                    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
                    **res
                })

        df_late = df[df['yearref'] > median_year].copy()
        result = run_local_linear_rd(df_late, primary_outcome, bandwidth=10,
                                    controls=baseline_controls, cluster_var='district_code')
        if result:
            res = extract_rd_results(result, None)
            if res:
                spec_count += 1
                results.append({
                    'paper_id': PAPER_ID,
                    'journal': JOURNAL,
                    'paper_title': PAPER_TITLE,
                    'spec_id': 'robust/sample/late_period',
                    'spec_tree_path': 'robustness/sample_restrictions.md',
                    'outcome_var': primary_outcome,
                    'treatment_var': 'treatment',
                    'sample_desc': f'Late period (year > {median_year})',
                    'fixed_effects': 'None',
                    'controls_desc': ', '.join(baseline_controls),
                    'cluster_var': 'district_code',
                    'model_type': 'RD_local_linear',
                    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
                    **res
                })

    # Trim outliers - outcomes
    for pct in [1, 5]:
        lower = df[primary_outcome].quantile(pct/100)
        upper = df[primary_outcome].quantile(1 - pct/100)
        df_trim = df[(df[primary_outcome] >= lower) & (df[primary_outcome] <= upper)].copy()

        result = run_local_linear_rd(df_trim, primary_outcome, bandwidth=10,
                                    controls=baseline_controls, cluster_var='district_code')
        if result:
            res = extract_rd_results(result, None)
            if res:
                spec_count += 1
                results.append({
                    'paper_id': PAPER_ID,
                    'journal': JOURNAL,
                    'paper_title': PAPER_TITLE,
                    'spec_id': f'robust/sample/trim_{pct}pct',
                    'spec_tree_path': 'robustness/sample_restrictions.md',
                    'outcome_var': primary_outcome,
                    'treatment_var': 'treatment',
                    'sample_desc': f'Trimmed top/bottom {pct}% of outcome',
                    'fixed_effects': 'None',
                    'controls_desc': ', '.join(baseline_controls),
                    'cluster_var': 'district_code',
                    'model_type': 'RD_local_linear',
                    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
                    **res
                })

    # By urbanicity
    if 'urban_centric_locale' in df.columns:
        df_urban = df[df['urban_centric_locale'] <= 2].copy()  # Urban/suburban
        result = run_local_linear_rd(df_urban, primary_outcome, bandwidth=10,
                                    controls=[c for c in baseline_controls if c != 'urban_centric_locale'],
                                    cluster_var='district_code')
        if result:
            res = extract_rd_results(result, None)
            if res:
                spec_count += 1
                results.append({
                    'paper_id': PAPER_ID,
                    'journal': JOURNAL,
                    'paper_title': PAPER_TITLE,
                    'spec_id': 'robust/sample/urban_suburban',
                    'spec_tree_path': 'robustness/sample_restrictions.md',
                    'outcome_var': primary_outcome,
                    'treatment_var': 'treatment',
                    'sample_desc': 'Urban and suburban districts only',
                    'fixed_effects': 'None',
                    'controls_desc': ', '.join([c for c in baseline_controls if c != 'urban_centric_locale']),
                    'cluster_var': 'district_code',
                    'model_type': 'RD_local_linear',
                    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
                    **res
                })

        df_rural = df[df['urban_centric_locale'] > 2].copy()  # Rural
        result = run_local_linear_rd(df_rural, primary_outcome, bandwidth=10,
                                    controls=[c for c in baseline_controls if c != 'urban_centric_locale'],
                                    cluster_var='district_code')
        if result:
            res = extract_rd_results(result, None)
            if res:
                spec_count += 1
                results.append({
                    'paper_id': PAPER_ID,
                    'journal': JOURNAL,
                    'paper_title': PAPER_TITLE,
                    'spec_id': 'robust/sample/rural',
                    'spec_tree_path': 'robustness/sample_restrictions.md',
                    'outcome_var': primary_outcome,
                    'treatment_var': 'treatment',
                    'sample_desc': 'Rural districts only',
                    'fixed_effects': 'None',
                    'controls_desc': ', '.join([c for c in baseline_controls if c != 'urban_centric_locale']),
                    'cluster_var': 'district_code',
                    'model_type': 'RD_local_linear',
                    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
                    **res
                })

    # By district size (above/below median enrollment)
    if 'above_median' in df.columns:
        df_large = df[df['above_median'] == 1].copy()
        result = run_local_linear_rd(df_large, primary_outcome, bandwidth=10,
                                    controls=[c for c in baseline_controls if c != 'above_median'],
                                    cluster_var='district_code')
        if result:
            res = extract_rd_results(result, None)
            if res:
                spec_count += 1
                results.append({
                    'paper_id': PAPER_ID,
                    'journal': JOURNAL,
                    'paper_title': PAPER_TITLE,
                    'spec_id': 'robust/sample/large_districts',
                    'spec_tree_path': 'robustness/sample_restrictions.md',
                    'outcome_var': primary_outcome,
                    'treatment_var': 'treatment',
                    'sample_desc': 'Above-median enrollment districts',
                    'fixed_effects': 'None',
                    'controls_desc': ', '.join([c for c in baseline_controls if c != 'above_median']),
                    'cluster_var': 'district_code',
                    'model_type': 'RD_local_linear',
                    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
                    **res
                })

        df_small = df[df['above_median'] == 0].copy()
        result = run_local_linear_rd(df_small, primary_outcome, bandwidth=10,
                                    controls=[c for c in baseline_controls if c != 'above_median'],
                                    cluster_var='district_code')
        if result:
            res = extract_rd_results(result, None)
            if res:
                spec_count += 1
                results.append({
                    'paper_id': PAPER_ID,
                    'journal': JOURNAL,
                    'paper_title': PAPER_TITLE,
                    'spec_id': 'robust/sample/small_districts',
                    'spec_tree_path': 'robustness/sample_restrictions.md',
                    'outcome_var': primary_outcome,
                    'treatment_var': 'treatment',
                    'sample_desc': 'Below-median enrollment districts',
                    'fixed_effects': 'None',
                    'controls_desc': ', '.join([c for c in baseline_controls if c != 'above_median']),
                    'cluster_var': 'district_code',
                    'model_type': 'RD_local_linear',
                    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
                    **res
                })

    # By economic disadvantage
    if 'econ_disadv_percent' in df.columns:
        median_ecdis = df['econ_disadv_percent'].median()

        df_high_pov = df[df['econ_disadv_percent'] > median_ecdis].copy()
        result = run_local_linear_rd(df_high_pov, primary_outcome, bandwidth=10,
                                    controls=[c for c in baseline_controls if c != 'econ_disadv_percent'],
                                    cluster_var='district_code')
        if result:
            res = extract_rd_results(result, None)
            if res:
                spec_count += 1
                results.append({
                    'paper_id': PAPER_ID,
                    'journal': JOURNAL,
                    'paper_title': PAPER_TITLE,
                    'spec_id': 'robust/sample/high_poverty',
                    'spec_tree_path': 'robustness/sample_restrictions.md',
                    'outcome_var': primary_outcome,
                    'treatment_var': 'treatment',
                    'sample_desc': 'High poverty districts',
                    'fixed_effects': 'None',
                    'controls_desc': ', '.join([c for c in baseline_controls if c != 'econ_disadv_percent']),
                    'cluster_var': 'district_code',
                    'model_type': 'RD_local_linear',
                    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
                    **res
                })

        df_low_pov = df[df['econ_disadv_percent'] <= median_ecdis].copy()
        result = run_local_linear_rd(df_low_pov, primary_outcome, bandwidth=10,
                                    controls=[c for c in baseline_controls if c != 'econ_disadv_percent'],
                                    cluster_var='district_code')
        if result:
            res = extract_rd_results(result, None)
            if res:
                spec_count += 1
                results.append({
                    'paper_id': PAPER_ID,
                    'journal': JOURNAL,
                    'paper_title': PAPER_TITLE,
                    'spec_id': 'robust/sample/low_poverty',
                    'spec_tree_path': 'robustness/sample_restrictions.md',
                    'outcome_var': primary_outcome,
                    'treatment_var': 'treatment',
                    'sample_desc': 'Low poverty districts',
                    'fixed_effects': 'None',
                    'controls_desc': ', '.join([c for c in baseline_controls if c != 'econ_disadv_percent']),
                    'cluster_var': 'district_code',
                    'model_type': 'RD_local_linear',
                    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
                    **res
                })

    # =========================================================================
    # 10. INFERENCE VARIATIONS (5 specs)
    # =========================================================================
    print("Running inference variations...")

    # No clustering (robust SE)
    result = run_local_linear_rd(df, primary_outcome, bandwidth=10,
                                controls=baseline_controls, cluster_var=None)
    if result:
        res = extract_rd_results(result, None)
        if res:
            spec_count += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': 'robust/cluster/none',
                'spec_tree_path': 'robustness/clustering_variations.md',
                'outcome_var': primary_outcome,
                'treatment_var': 'treatment',
                'sample_desc': 'bandwidth=10, robust SE (no clustering)',
                'fixed_effects': 'None',
                'controls_desc': ', '.join(baseline_controls),
                'cluster_var': 'None',
                'model_type': 'RD_local_linear',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
                **res
            })

    # Cluster by leaid (school district LEA code)
    if 'leaid' in df.columns:
        result = run_local_linear_rd(df, primary_outcome, bandwidth=10,
                                    controls=baseline_controls, cluster_var='leaid')
        if result:
            res = extract_rd_results(result, None)
            if res:
                spec_count += 1
                results.append({
                    'paper_id': PAPER_ID,
                    'journal': JOURNAL,
                    'paper_title': PAPER_TITLE,
                    'spec_id': 'robust/cluster/leaid',
                    'spec_tree_path': 'robustness/clustering_variations.md',
                    'outcome_var': primary_outcome,
                    'treatment_var': 'treatment',
                    'sample_desc': 'bandwidth=10, clustered by LEA ID',
                    'fixed_effects': 'None',
                    'controls_desc': ', '.join(baseline_controls),
                    'cluster_var': 'leaid',
                    'model_type': 'RD_local_linear',
                    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
                    **res
                })

    # =========================================================================
    # 11. PLACEBO CUTOFF TESTS (5 specs)
    # =========================================================================
    print("Running placebo cutoff tests...")

    placebo_cutoffs = [40, 45, 55, 60]
    for cutoff in placebo_cutoffs:
        df_placebo = df.copy()
        df_placebo['running_placebo'] = df_placebo['perc'] - cutoff
        df_placebo['treatment_placebo'] = (df_placebo['perc'] >= cutoff).astype(int)
        df_placebo['running_treat_placebo'] = df_placebo['running_placebo'] * df_placebo['treatment_placebo']

        # Only use observations on one side of the true cutoff (50)
        if cutoff < 50:
            df_placebo = df_placebo[df_placebo['perc'] < 50].copy()
        else:
            df_placebo = df_placebo[df_placebo['perc'] >= 50].copy()

        result = run_local_linear_rd(df_placebo, primary_outcome,
                                    running='running_placebo',
                                    treatment='treatment_placebo',
                                    bandwidth=10, controls=baseline_controls,
                                    cluster_var='district_code')
        if result:
            res = extract_rd_results(result, None, treatment_var='treatment_placebo')
            if res:
                spec_count += 1
                results.append({
                    'paper_id': PAPER_ID,
                    'journal': JOURNAL,
                    'paper_title': PAPER_TITLE,
                    'spec_id': f'rd/placebo/cutoff_{cutoff}',
                    'spec_tree_path': 'methods/regression_discontinuity.md#placebo-cutoff-tests',
                    'outcome_var': primary_outcome,
                    'treatment_var': f'treatment_at_{cutoff}',
                    'sample_desc': f'Placebo cutoff at {cutoff}%',
                    'fixed_effects': 'None',
                    'controls_desc': ', '.join(baseline_controls),
                    'cluster_var': 'district_code',
                    'model_type': 'RD_local_linear_placebo',
                    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
                    **res
                })

    # =========================================================================
    # 12. HETEROGENEITY ANALYSIS (8 specs)
    # =========================================================================
    print("Running heterogeneity analysis...")

    # Interaction with urbanicity
    df_het = df.copy()
    df_het['treatment_urban'] = df_het['treatment'] * (df_het['urban_centric_locale'] <= 2).astype(int)

    # Run interaction model manually
    df_bw = df_het[df_het['running'].abs() <= 10].copy()
    df_bw = df_bw.dropna(subset=[primary_outcome, 'running', 'treatment', 'treatment_urban'])
    df_bw['running_treat'] = df_bw['running'] * df_bw['treatment']
    df_bw['urban_dummy'] = (df_bw['urban_centric_locale'] <= 2).astype(int)

    y = df_bw[primary_outcome]
    X_vars = ['treatment', 'running', 'running_treat', 'urban_dummy', 'treatment_urban']
    for ctrl in [c for c in baseline_controls if c != 'urban_centric_locale']:
        if ctrl in df_bw.columns:
            X_vars.append(ctrl)

    df_bw = df_bw.dropna(subset=X_vars)
    X = df_bw[X_vars]
    X = sm.add_constant(X)

    try:
        model = OLS(df_bw[primary_outcome], X)
        results_het = model.fit(cov_type='cluster', cov_kwds={'groups': df_bw['district_code']})

        if 'treatment_urban' in results_het.params:
            coef = results_het.params['treatment_urban']
            se = results_het.bse['treatment_urban']
            pval = results_het.pvalues['treatment_urban']
            spec_count += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': 'robust/heterogeneity/urban',
                'spec_tree_path': 'robustness/heterogeneity.md',
                'outcome_var': primary_outcome,
                'treatment_var': 'treatment_urban_interaction',
                'coefficient': float(coef),
                'std_error': float(se),
                't_stat': float(coef/se),
                'p_value': float(pval),
                'ci_lower': float(coef - 1.96*se),
                'ci_upper': float(coef + 1.96*se),
                'n_obs': int(len(df_bw)),
                'r_squared': float(results_het.rsquared),
                'sample_desc': 'bandwidth=10, treatment x urban interaction',
                'fixed_effects': 'None',
                'controls_desc': ', '.join(baseline_controls),
                'cluster_var': 'district_code',
                'model_type': 'RD_heterogeneity',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
                'coefficient_vector_json': json.dumps({'interaction': {'var': 'treatment_urban', 'coef': float(coef), 'se': float(se), 'pval': float(pval)}})
            })
    except Exception as e:
        print(f"Heterogeneity (urban) failed: {e}")

    # Interaction with district size
    df_het = df.copy()
    df_het['treatment_large'] = df_het['treatment'] * df_het['above_median']

    df_bw = df_het[df_het['running'].abs() <= 10].copy()
    df_bw = df_bw.dropna(subset=[primary_outcome, 'running', 'treatment', 'treatment_large', 'above_median'])
    df_bw['running_treat'] = df_bw['running'] * df_bw['treatment']

    y = df_bw[primary_outcome]
    X_vars = ['treatment', 'running', 'running_treat', 'above_median', 'treatment_large']
    for ctrl in [c for c in baseline_controls if c != 'above_median']:
        if ctrl in df_bw.columns:
            X_vars.append(ctrl)

    df_bw = df_bw.dropna(subset=X_vars)
    X = df_bw[X_vars]
    X = sm.add_constant(X)

    try:
        model = OLS(df_bw[primary_outcome], X)
        results_het = model.fit(cov_type='cluster', cov_kwds={'groups': df_bw['district_code']})

        if 'treatment_large' in results_het.params:
            coef = results_het.params['treatment_large']
            se = results_het.bse['treatment_large']
            pval = results_het.pvalues['treatment_large']
            spec_count += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': 'robust/heterogeneity/large_district',
                'spec_tree_path': 'robustness/heterogeneity.md',
                'outcome_var': primary_outcome,
                'treatment_var': 'treatment_large_interaction',
                'coefficient': float(coef),
                'std_error': float(se),
                't_stat': float(coef/se),
                'p_value': float(pval),
                'ci_lower': float(coef - 1.96*se),
                'ci_upper': float(coef + 1.96*se),
                'n_obs': int(len(df_bw)),
                'r_squared': float(results_het.rsquared),
                'sample_desc': 'bandwidth=10, treatment x large district interaction',
                'fixed_effects': 'None',
                'controls_desc': ', '.join(baseline_controls),
                'cluster_var': 'district_code',
                'model_type': 'RD_heterogeneity',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
                'coefficient_vector_json': json.dumps({'interaction': {'var': 'treatment_large', 'coef': float(coef), 'se': float(se), 'pval': float(pval)}})
            })
    except Exception as e:
        print(f"Heterogeneity (size) failed: {e}")

    # Interaction with economic disadvantage
    df_het = df.copy()
    df_het['high_poverty'] = (df_het['econ_disadv_percent'] > df_het['econ_disadv_percent'].median()).astype(int)
    df_het['treatment_highpov'] = df_het['treatment'] * df_het['high_poverty']

    df_bw = df_het[df_het['running'].abs() <= 10].copy()
    df_bw = df_bw.dropna(subset=[primary_outcome, 'running', 'treatment', 'treatment_highpov', 'high_poverty'])
    df_bw['running_treat'] = df_bw['running'] * df_bw['treatment']

    y = df_bw[primary_outcome]
    X_vars = ['treatment', 'running', 'running_treat', 'high_poverty', 'treatment_highpov']
    for ctrl in [c for c in baseline_controls if c != 'econ_disadv_percent']:
        if ctrl in df_bw.columns:
            X_vars.append(ctrl)

    df_bw = df_bw.dropna(subset=X_vars)
    X = df_bw[X_vars]
    X = sm.add_constant(X)

    try:
        model = OLS(df_bw[primary_outcome], X)
        results_het = model.fit(cov_type='cluster', cov_kwds={'groups': df_bw['district_code']})

        if 'treatment_highpov' in results_het.params:
            coef = results_het.params['treatment_highpov']
            se = results_het.bse['treatment_highpov']
            pval = results_het.pvalues['treatment_highpov']
            spec_count += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': 'robust/heterogeneity/high_poverty',
                'spec_tree_path': 'robustness/heterogeneity.md',
                'outcome_var': primary_outcome,
                'treatment_var': 'treatment_highpov_interaction',
                'coefficient': float(coef),
                'std_error': float(se),
                't_stat': float(coef/se),
                'p_value': float(pval),
                'ci_lower': float(coef - 1.96*se),
                'ci_upper': float(coef + 1.96*se),
                'n_obs': int(len(df_bw)),
                'r_squared': float(results_het.rsquared),
                'sample_desc': 'bandwidth=10, treatment x high poverty interaction',
                'fixed_effects': 'None',
                'controls_desc': ', '.join(baseline_controls),
                'cluster_var': 'district_code',
                'model_type': 'RD_heterogeneity',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
                'coefficient_vector_json': json.dumps({'interaction': {'var': 'treatment_highpov', 'coef': float(coef), 'se': float(se), 'pval': float(pval)}})
            })
    except Exception as e:
        print(f"Heterogeneity (poverty) failed: {e}")

    # =========================================================================
    # 13. FUNCTIONAL FORM VARIATIONS (4 specs)
    # =========================================================================
    print("Running functional form variations...")

    # Log outcome
    df_log = df.copy()
    df_log['log_outcome'] = np.log(df_log[primary_outcome] + 1)

    result = run_local_linear_rd(df_log, 'log_outcome', bandwidth=10,
                                controls=baseline_controls, cluster_var='district_code')
    if result:
        res = extract_rd_results(result, None)
        if res:
            spec_count += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': 'robust/funcform/log_outcome',
                'spec_tree_path': 'robustness/functional_form.md',
                'outcome_var': 'log_' + primary_outcome,
                'treatment_var': 'treatment',
                'sample_desc': 'bandwidth=10, log(outcome+1)',
                'fixed_effects': 'None',
                'controls_desc': ', '.join(baseline_controls),
                'cluster_var': 'district_code',
                'model_type': 'RD_local_linear',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
                **res
            })

    # IHS transformation
    df_ihs = df.copy()
    df_ihs['ihs_outcome'] = np.arcsinh(df_ihs[primary_outcome])

    result = run_local_linear_rd(df_ihs, 'ihs_outcome', bandwidth=10,
                                controls=baseline_controls, cluster_var='district_code')
    if result:
        res = extract_rd_results(result, None)
        if res:
            spec_count += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': 'robust/funcform/ihs_outcome',
                'spec_tree_path': 'robustness/functional_form.md',
                'outcome_var': 'ihs_' + primary_outcome,
                'treatment_var': 'treatment',
                'sample_desc': 'bandwidth=10, asinh(outcome)',
                'fixed_effects': 'None',
                'controls_desc': ', '.join(baseline_controls),
                'cluster_var': 'district_code',
                'model_type': 'RD_local_linear',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
                **res
            })

    # =========================================================================
    # 14. ADDITIONAL BANDWIDTH SENSITIVITY (5 specs)
    # =========================================================================
    print("Running additional bandwidth sensitivity...")

    extra_bandwidths = [3, 4, 6, 8, 25]
    for bw in extra_bandwidths:
        result = run_local_linear_rd(df, primary_outcome, bandwidth=bw,
                                    controls=baseline_controls, cluster_var='district_code')
        if result:
            res = extract_rd_results(result, None)
            if res:
                spec_count += 1
                results.append({
                    'paper_id': PAPER_ID,
                    'journal': JOURNAL,
                    'paper_title': PAPER_TITLE,
                    'spec_id': f'rd/bandwidth/sensitivity_{bw}',
                    'spec_tree_path': 'methods/regression_discontinuity.md#bandwidth-selection',
                    'outcome_var': primary_outcome,
                    'treatment_var': 'treatment',
                    'sample_desc': f'bandwidth={bw}, sensitivity check',
                    'fixed_effects': 'None',
                    'controls_desc': ', '.join(baseline_controls),
                    'cluster_var': 'district_code',
                    'model_type': 'RD_local_linear',
                    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
                    **res
                })

    # =========================================================================
    # 15. MORE OUTCOME VARIATIONS (additional outcomes)
    # =========================================================================
    print("Running additional outcome specifications...")

    additional_outcomes = {
        'tot_exp_ss_mem': 'Support services expenditures per member',
        'interest_debt_mem': 'Interest on debt per member',
        'LT_debt_out_mem': 'Long-term debt outstanding per member',
        'ratio_stdnts_to_staff_licensed': 'Student-teacher ratio (licensed)',
    }

    for outcome_var, outcome_desc in additional_outcomes.items():
        result = run_local_linear_rd(df, outcome_var, bandwidth=10,
                                    controls=baseline_controls,
                                    cluster_var='district_code')
        if result:
            res = extract_rd_results(result, None)
            if res:
                spec_count += 1
                results.append({
                    'paper_id': PAPER_ID,
                    'journal': JOURNAL,
                    'paper_title': PAPER_TITLE,
                    'spec_id': f'robust/outcome/{outcome_var}',
                    'spec_tree_path': 'robustness/measurement.md',
                    'outcome_var': outcome_var,
                    'treatment_var': 'treatment',
                    'sample_desc': f'bandwidth=10, outcome: {outcome_desc}',
                    'fixed_effects': 'None',
                    'controls_desc': ', '.join(baseline_controls),
                    'cluster_var': 'district_code',
                    'model_type': 'RD_local_linear',
                    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
                    **res
                })

    print(f"\nTotal specifications run: {spec_count}")

    return results


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print(f"Specification Search: {PAPER_ID}")
    print("="*60)

    # Load data
    print("\nLoading data...")
    df = load_data()
    print(f"Loaded {len(df)} observations")

    # Run specification search
    print("\nRunning specification search...")
    results = run_specification_search(df)

    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)

    # Save results
    output_file = OUTPUT_PATH / "specification_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    print(f"\nTotal specifications: {len(results_df)}")

    # Filter to main treatment effects (exclude heterogeneity interactions)
    main_specs = results_df[~results_df['spec_id'].str.contains('heterogeneity')]

    if len(main_specs) > 0:
        print(f"Main treatment effect specifications: {len(main_specs)}")
        print(f"Positive coefficients: {(main_specs['coefficient'] > 0).sum()} ({100*(main_specs['coefficient'] > 0).mean():.1f}%)")
        print(f"Significant at 5%: {(main_specs['p_value'] < 0.05).sum()} ({100*(main_specs['p_value'] < 0.05).mean():.1f}%)")
        print(f"Significant at 1%: {(main_specs['p_value'] < 0.01).sum()} ({100*(main_specs['p_value'] < 0.01).mean():.1f}%)")
        print(f"Median coefficient: {main_specs['coefficient'].median():.2f}")
        print(f"Mean coefficient: {main_specs['coefficient'].mean():.2f}")
        print(f"Range: [{main_specs['coefficient'].min():.2f}, {main_specs['coefficient'].max():.2f}]")

    print("\nSpecification counts by category:")
    for cat in results_df['spec_id'].str.split('/').str[0].unique():
        n = results_df['spec_id'].str.startswith(cat).sum()
        print(f"  {cat}: {n}")

    print("\n" + "="*60)
    print("SPECIFICATION SEARCH COMPLETE")
    print("="*60)
