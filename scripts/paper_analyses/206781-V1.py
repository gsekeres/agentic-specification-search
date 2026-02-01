"""
Specification Search: Paper 206781-V1
Title: Targeting Impact versus Deprivation
Authors: Haushofer, Niehaus, Paramo, Miguel, Walker
Journal: AER

Method: Cross-sectional OLS with cluster-randomized experimental identification (RCT)

IMPORTANT DESIGN NOTE: This is a CLUSTER-RANDOMIZED TRIAL where treatment was assigned
at the VILLAGE level (328 treated villages, 325 control villages). Therefore:
- Village fixed effects CANNOT be used (they would absorb all treatment variation)
- Standard errors MUST be clustered at the village level
- The appropriate specification is OLS with robust clustered SEs

The paper uses causal forests to analyze heterogeneous treatment effects, but the
underlying average treatment effects can be estimated with standard regression.
"""

import pandas as pd
import numpy as np
import json
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.quantile_regression import QuantReg

# Configuration
PAPER_ID = "206781-V1"
JOURNAL = "AER"
PAPER_TITLE = "Targeting Impact versus Deprivation"
BASE_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search"
DATA_PATH = f"{BASE_PATH}/data/downloads/extracted/{PAPER_ID}/data"
OUTPUT_PATH = f"{BASE_PATH}/data/downloads/extracted/{PAPER_ID}"

# Define covariates (from R code: load_functions_and_data.R)
COVARIATES = [
    'bl_hhsize', 'bl_widow', 'bl_female_fr', 'bl_hh_has_children', 'bl_children_school',
    'bl_child_3', 'bl_child_6', 'bl_has_elder', 'bl_any_livestock', 'bl_own_land',
    'bl_own_quarter_acre', 'bl_own_TV_radio', 'bl_meals_eaten', 'bl_meals_protein',
    'bl_selfemp', 'bl_emp'
]

# Outcome variables (following paper's PAP)
OUTCOMES = {
    'el_consumption_hh': 'Total household consumption (PPP)',
    'el_income_hh': 'Total household income (PPP)',
    'el_assets_hh': 'Total household assets (PPP)',
    'el_fsi': 'Food security index',
    'el_food_cons_hh': 'Food consumption (PPP)'
}

# Primary outcome for main analysis
PRIMARY_OUTCOME = 'el_consumption_hh'
TREATMENT_VAR = 'treat'


def load_data():
    """Load and prepare analysis data."""
    df = pd.read_stata(f"{DATA_PATH}/PMT_targeting_data.dta")

    # Apply filtering conditions from R code
    df = df[df['eligible'] == 1].copy()
    df = df[df['targeting_sample'] == 1].copy()

    # Drop rows with too many missing values
    missing_per_row = df[COVARIATES].isna().sum(axis=1)
    df = df[missing_per_row <= 7].copy()

    # Fill remaining missing covariates with median (small number of missing)
    for cov in COVARIATES:
        if df[cov].isna().sum() > 0:
            df[cov] = df[cov].fillna(df[cov].median())

    # Verify cluster-randomization
    village_treat = df.groupby('village_code')['treat'].mean()
    n_treat_villages = (village_treat == 1).sum()
    n_control_villages = (village_treat == 0).sum()

    print(f"Analysis sample size: {len(df)}")
    print(f"Treatment: {df['treat'].sum()} treated HH, {int((1-df['treat']).sum())} control HH")
    print(f"Villages: {n_treat_villages} treated, {n_control_villages} control (cluster-randomized)")

    return df


def run_ols_regression(df, outcome, controls, cluster_var='village_code', weights=None, se_type='cluster'):
    """Run OLS regression with cluster-robust standard errors."""

    # Build formula
    if controls:
        controls_str = " + ".join(controls)
        formula = f"{outcome} ~ {TREATMENT_VAR} + {controls_str}"
    else:
        formula = f"{outcome} ~ {TREATMENT_VAR}"

    try:
        if weights is not None:
            model = smf.wls(formula, data=df, weights=df[weights]).fit(cov_type='HC1')
        elif se_type == 'cluster' and cluster_var:
            model = smf.ols(formula, data=df).fit(
                cov_type='cluster',
                cov_kwds={'groups': df[cluster_var]}
            )
        elif se_type == 'hc1':
            model = smf.ols(formula, data=df).fit(cov_type='HC1')
        elif se_type == 'hc3':
            model = smf.ols(formula, data=df).fit(cov_type='HC3')
        else:
            model = smf.ols(formula, data=df).fit(cov_type='HC1')

        coef = model.params[TREATMENT_VAR]
        se = model.bse[TREATMENT_VAR]
        pval = model.pvalues[TREATMENT_VAR]
        nobs = int(model.nobs)
        r2 = model.rsquared

        # Get all coefficients
        all_coefs = {}
        for var in model.params.index:
            if var != 'Intercept':
                all_coefs[var] = {
                    'coef': float(model.params[var]),
                    'se': float(model.bse[var]),
                    'pval': float(model.pvalues[var])
                }

        # Calculate t-stat and CI
        t_stat = coef / se
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se

        return {
            'coefficient': float(coef),
            'std_error': float(se),
            't_stat': float(t_stat),
            'p_value': float(pval),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_obs': int(nobs),
            'r_squared': float(r2),
            'all_coefficients': all_coefs,
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
            'n_obs': np.nan,
            'r_squared': np.nan,
            'all_coefficients': {},
            'success': False,
            'error': str(e)
        }


def run_quantile_regression(df, outcome, controls, quantile=0.5):
    """Run quantile regression."""
    try:
        y = df[outcome]
        if controls:
            X = df[[TREATMENT_VAR] + controls].copy()
        else:
            X = df[[TREATMENT_VAR]].copy()
        X = sm.add_constant(X)

        model = QuantReg(y, X).fit(q=quantile)

        coef = model.params[TREATMENT_VAR]
        se = model.bse[TREATMENT_VAR]
        pval = model.pvalues[TREATMENT_VAR]

        t_stat = coef / se
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se

        all_coefs = {}
        for var in model.params.index:
            if var != 'const':
                all_coefs[var] = {
                    'coef': float(model.params[var]),
                    'se': float(model.bse[var]),
                    'pval': float(model.pvalues[var])
                }

        return {
            'coefficient': float(coef),
            'std_error': float(se),
            't_stat': float(t_stat),
            'p_value': float(pval),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_obs': int(len(df)),
            'r_squared': float(model.prsquared) if hasattr(model, 'prsquared') else np.nan,
            'all_coefficients': all_coefs,
            'success': True
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def format_coefficient_vector(result, controls, fe_vars=None):
    """Format coefficient vector as JSON."""
    if not result['success']:
        return json.dumps({'error': result.get('error', 'Unknown error')})

    coef_vector = {
        'treatment': {
            'var': TREATMENT_VAR,
            'coef': result['coefficient'],
            'se': result['std_error'],
            'pval': result['p_value']
        },
        'controls': [],
        'fixed_effects': fe_vars if fe_vars else [],
        'diagnostics': {
            'n_obs': result['n_obs'],
            'r_squared': result['r_squared']
        }
    }

    for var, vals in result.get('all_coefficients', {}).items():
        if var != TREATMENT_VAR:
            coef_vector['controls'].append({
                'var': var,
                'coef': vals['coef'],
                'se': vals['se'],
                'pval': vals['pval']
            })

    return json.dumps(coef_vector)


def run_all_specifications(df):
    """Run all specifications and return results."""
    results = []

    # ===========================================
    # BASELINE: Main specification with full controls and clustered SEs
    # Note: NO village FE due to cluster-randomized design
    # ===========================================
    print("\n=== Running Baseline Specifications (All Outcomes) ===")

    for outcome, outcome_desc in OUTCOMES.items():
        result = run_ols_regression(df, outcome, COVARIATES, cluster_var='village_code')

        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'baseline' if outcome == PRIMARY_OUTCOME else f'baseline/{outcome}',
            'spec_tree_path': 'methods/cross_sectional_ols.md#baseline',
            'outcome_var': outcome,
            'treatment_var': TREATMENT_VAR,
            'coefficient': result['coefficient'],
            'std_error': result['std_error'],
            't_stat': result['t_stat'],
            'p_value': result['p_value'],
            'ci_lower': result['ci_lower'],
            'ci_upper': result['ci_upper'],
            'n_obs': result['n_obs'],
            'r_squared': result['r_squared'],
            'coefficient_vector_json': format_coefficient_vector(result, COVARIATES),
            'sample_desc': 'Eligible HH, targeting sample, cluster-RCT',
            'fixed_effects': 'None (cluster-randomized)',
            'controls_desc': 'Full PMT covariates (16 vars)',
            'cluster_var': 'village_code',
            'model_type': 'OLS (cluster-randomized)',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })

        print(f"  {outcome}: coef={result['coefficient']:.2f}, SE={result['std_error']:.2f}, p={result['p_value']:.4f}")

    # ===========================================
    # OLS METHOD VARIATIONS
    # ===========================================
    print("\n=== Running Method Variations ===")

    # WLS with household weights
    result = run_ols_regression(df, PRIMARY_OUTCOME, COVARIATES, weights='hhweight_EL')
    results.append({
        'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
        'spec_id': 'ols/method/wls',
        'spec_tree_path': 'methods/cross_sectional_ols.md#estimation-method',
        'outcome_var': PRIMARY_OUTCOME, 'treatment_var': TREATMENT_VAR,
        'coefficient': result['coefficient'], 'std_error': result['std_error'],
        't_stat': result['t_stat'], 'p_value': result['p_value'],
        'ci_lower': result['ci_lower'], 'ci_upper': result['ci_upper'],
        'n_obs': result['n_obs'], 'r_squared': result['r_squared'],
        'coefficient_vector_json': format_coefficient_vector(result, COVARIATES),
        'sample_desc': 'Eligible HH, targeting sample',
        'fixed_effects': 'None',
        'controls_desc': 'Full PMT covariates',
        'cluster_var': 'None (WLS with robust SE)',
        'model_type': 'WLS',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })
    print(f"  WLS: coef={result['coefficient']:.2f}, p={result['p_value']:.4f}")

    # ===========================================
    # QUANTILE REGRESSIONS
    # ===========================================
    print("\n=== Running Quantile Regressions ===")

    for q, q_name in [(0.25, '25'), (0.5, 'median'), (0.75, '75')]:
        result = run_quantile_regression(df, PRIMARY_OUTCOME, COVARIATES, quantile=q)
        if result['success']:
            results.append({
                'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
                'spec_id': f'ols/method/quantile_{q_name}',
                'spec_tree_path': 'methods/cross_sectional_ols.md#estimation-method',
                'outcome_var': PRIMARY_OUTCOME, 'treatment_var': TREATMENT_VAR,
                'coefficient': result['coefficient'], 'std_error': result['std_error'],
                't_stat': result['t_stat'], 'p_value': result['p_value'],
                'ci_lower': result['ci_lower'], 'ci_upper': result['ci_upper'],
                'n_obs': result['n_obs'], 'r_squared': result['r_squared'],
                'coefficient_vector_json': format_coefficient_vector(result, COVARIATES),
                'sample_desc': 'Eligible HH, targeting sample',
                'fixed_effects': 'None',
                'controls_desc': 'Full PMT covariates',
                'cluster_var': 'None (quantile)',
                'model_type': f'Quantile ({q})',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })
            print(f"  Q{q_name}: coef={result['coefficient']:.2f}, p={result['p_value']:.4f}")

    # ===========================================
    # CONTROL SET VARIATIONS
    # ===========================================
    print("\n=== Running Control Set Variations ===")

    # No controls (bivariate)
    result = run_ols_regression(df, PRIMARY_OUTCOME, [], cluster_var='village_code')
    results.append({
        'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
        'spec_id': 'ols/controls/none',
        'spec_tree_path': 'methods/cross_sectional_ols.md#control-sets',
        'outcome_var': PRIMARY_OUTCOME, 'treatment_var': TREATMENT_VAR,
        'coefficient': result['coefficient'], 'std_error': result['std_error'],
        't_stat': result['t_stat'], 'p_value': result['p_value'],
        'ci_lower': result['ci_lower'], 'ci_upper': result['ci_upper'],
        'n_obs': result['n_obs'], 'r_squared': result['r_squared'],
        'coefficient_vector_json': format_coefficient_vector(result, []),
        'sample_desc': 'Eligible HH, targeting sample',
        'fixed_effects': 'None',
        'controls_desc': 'None (bivariate)',
        'cluster_var': 'village_code',
        'model_type': 'OLS',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })
    print(f"  No controls: coef={result['coefficient']:.2f}, p={result['p_value']:.4f}")

    # Demographics only
    demo_controls = ['bl_hhsize', 'bl_widow', 'bl_female_fr', 'bl_hh_has_children', 'bl_has_elder']
    result = run_ols_regression(df, PRIMARY_OUTCOME, demo_controls, cluster_var='village_code')
    results.append({
        'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
        'spec_id': 'ols/controls/demographics',
        'spec_tree_path': 'methods/cross_sectional_ols.md#control-sets',
        'outcome_var': PRIMARY_OUTCOME, 'treatment_var': TREATMENT_VAR,
        'coefficient': result['coefficient'], 'std_error': result['std_error'],
        't_stat': result['t_stat'], 'p_value': result['p_value'],
        'ci_lower': result['ci_lower'], 'ci_upper': result['ci_upper'],
        'n_obs': result['n_obs'], 'r_squared': result['r_squared'],
        'coefficient_vector_json': format_coefficient_vector(result, demo_controls),
        'sample_desc': 'Eligible HH, targeting sample',
        'fixed_effects': 'None',
        'controls_desc': 'Demographics only (5 vars)',
        'cluster_var': 'village_code',
        'model_type': 'OLS',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })
    print(f"  Demographics: coef={result['coefficient']:.2f}, p={result['p_value']:.4f}")

    # Assets only
    asset_controls = ['bl_any_livestock', 'bl_own_land', 'bl_own_quarter_acre', 'bl_own_TV_radio']
    result = run_ols_regression(df, PRIMARY_OUTCOME, asset_controls, cluster_var='village_code')
    results.append({
        'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
        'spec_id': 'ols/controls/assets_only',
        'spec_tree_path': 'methods/cross_sectional_ols.md#control-sets',
        'outcome_var': PRIMARY_OUTCOME, 'treatment_var': TREATMENT_VAR,
        'coefficient': result['coefficient'], 'std_error': result['std_error'],
        't_stat': result['t_stat'], 'p_value': result['p_value'],
        'ci_lower': result['ci_lower'], 'ci_upper': result['ci_upper'],
        'n_obs': result['n_obs'], 'r_squared': result['r_squared'],
        'coefficient_vector_json': format_coefficient_vector(result, asset_controls),
        'sample_desc': 'Eligible HH, targeting sample',
        'fixed_effects': 'None',
        'controls_desc': 'Asset indicators only (4 vars)',
        'cluster_var': 'village_code',
        'model_type': 'OLS',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })
    print(f"  Assets only: coef={result['coefficient']:.2f}, p={result['p_value']:.4f}")

    # Children controls only
    children_controls = ['bl_hh_has_children', 'bl_children_school', 'bl_child_3', 'bl_child_6']
    result = run_ols_regression(df, PRIMARY_OUTCOME, children_controls, cluster_var='village_code')
    results.append({
        'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
        'spec_id': 'ols/controls/children_only',
        'spec_tree_path': 'methods/cross_sectional_ols.md#control-sets',
        'outcome_var': PRIMARY_OUTCOME, 'treatment_var': TREATMENT_VAR,
        'coefficient': result['coefficient'], 'std_error': result['std_error'],
        't_stat': result['t_stat'], 'p_value': result['p_value'],
        'ci_lower': result['ci_lower'], 'ci_upper': result['ci_upper'],
        'n_obs': result['n_obs'], 'r_squared': result['r_squared'],
        'coefficient_vector_json': format_coefficient_vector(result, children_controls),
        'sample_desc': 'Eligible HH, targeting sample',
        'fixed_effects': 'None',
        'controls_desc': 'Children indicators only (4 vars)',
        'cluster_var': 'village_code',
        'model_type': 'OLS',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })
    print(f"  Children only: coef={result['coefficient']:.2f}, p={result['p_value']:.4f}")

    # Employment controls only
    emp_controls = ['bl_selfemp', 'bl_emp']
    result = run_ols_regression(df, PRIMARY_OUTCOME, emp_controls, cluster_var='village_code')
    results.append({
        'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
        'spec_id': 'ols/controls/employment_only',
        'spec_tree_path': 'methods/cross_sectional_ols.md#control-sets',
        'outcome_var': PRIMARY_OUTCOME, 'treatment_var': TREATMENT_VAR,
        'coefficient': result['coefficient'], 'std_error': result['std_error'],
        't_stat': result['t_stat'], 'p_value': result['p_value'],
        'ci_lower': result['ci_lower'], 'ci_upper': result['ci_upper'],
        'n_obs': result['n_obs'], 'r_squared': result['r_squared'],
        'coefficient_vector_json': format_coefficient_vector(result, emp_controls),
        'sample_desc': 'Eligible HH, targeting sample',
        'fixed_effects': 'None',
        'controls_desc': 'Employment indicators only (2 vars)',
        'cluster_var': 'village_code',
        'model_type': 'OLS',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })
    print(f"  Employment only: coef={result['coefficient']:.2f}, p={result['p_value']:.4f}")

    # ===========================================
    # STANDARD ERROR VARIATIONS
    # ===========================================
    print("\n=== Running Standard Error Variations ===")

    # HC1 robust SE (no clustering)
    result = run_ols_regression(df, PRIMARY_OUTCOME, COVARIATES, se_type='hc1')
    results.append({
        'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
        'spec_id': 'robust/se/hc1',
        'spec_tree_path': 'robustness/clustering_variations.md#alternative-se-methods',
        'outcome_var': PRIMARY_OUTCOME, 'treatment_var': TREATMENT_VAR,
        'coefficient': result['coefficient'], 'std_error': result['std_error'],
        't_stat': result['t_stat'], 'p_value': result['p_value'],
        'ci_lower': result['ci_lower'], 'ci_upper': result['ci_upper'],
        'n_obs': result['n_obs'], 'r_squared': result['r_squared'],
        'coefficient_vector_json': format_coefficient_vector(result, COVARIATES),
        'sample_desc': 'Eligible HH, targeting sample',
        'fixed_effects': 'None',
        'controls_desc': 'Full PMT covariates',
        'cluster_var': 'None (HC1)',
        'model_type': 'OLS',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })
    print(f"  HC1: coef={result['coefficient']:.2f}, SE={result['std_error']:.2f}, p={result['p_value']:.4f}")

    # HC3 robust SE
    result = run_ols_regression(df, PRIMARY_OUTCOME, COVARIATES, se_type='hc3')
    results.append({
        'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
        'spec_id': 'robust/se/hc3',
        'spec_tree_path': 'robustness/clustering_variations.md#alternative-se-methods',
        'outcome_var': PRIMARY_OUTCOME, 'treatment_var': TREATMENT_VAR,
        'coefficient': result['coefficient'], 'std_error': result['std_error'],
        't_stat': result['t_stat'], 'p_value': result['p_value'],
        'ci_lower': result['ci_lower'], 'ci_upper': result['ci_upper'],
        'n_obs': result['n_obs'], 'r_squared': result['r_squared'],
        'coefficient_vector_json': format_coefficient_vector(result, COVARIATES),
        'sample_desc': 'Eligible HH, targeting sample',
        'fixed_effects': 'None',
        'controls_desc': 'Full PMT covariates',
        'cluster_var': 'None (HC3)',
        'model_type': 'OLS',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })
    print(f"  HC3: coef={result['coefficient']:.2f}, SE={result['std_error']:.2f}, p={result['p_value']:.4f}")

    # Cluster at sublocation level (higher aggregation)
    result = run_ols_regression(df, PRIMARY_OUTCOME, COVARIATES, cluster_var='s1_q2b_sublocation')
    results.append({
        'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
        'spec_id': 'robust/cluster/sublocation',
        'spec_tree_path': 'robustness/clustering_variations.md#higher-level-clustering',
        'outcome_var': PRIMARY_OUTCOME, 'treatment_var': TREATMENT_VAR,
        'coefficient': result['coefficient'], 'std_error': result['std_error'],
        't_stat': result['t_stat'], 'p_value': result['p_value'],
        'ci_lower': result['ci_lower'], 'ci_upper': result['ci_upper'],
        'n_obs': result['n_obs'], 'r_squared': result['r_squared'],
        'coefficient_vector_json': format_coefficient_vector(result, COVARIATES),
        'sample_desc': 'Eligible HH, targeting sample',
        'fixed_effects': 'None',
        'controls_desc': 'Full PMT covariates',
        'cluster_var': 's1_q2b_sublocation',
        'model_type': 'OLS',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })
    print(f"  Sublocation cluster: coef={result['coefficient']:.2f}, SE={result['std_error']:.2f}, p={result['p_value']:.4f}")

    # ===========================================
    # LEAVE-ONE-OUT ROBUSTNESS
    # ===========================================
    print("\n=== Running Leave-One-Out Specifications ===")

    for drop_var in COVARIATES:
        loo_controls = [c for c in COVARIATES if c != drop_var]
        result = run_ols_regression(df, PRIMARY_OUTCOME, loo_controls, cluster_var='village_code')
        results.append({
            'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
            'spec_id': f'robust/loo/drop_{drop_var}',
            'spec_tree_path': 'robustness/leave_one_out.md',
            'outcome_var': PRIMARY_OUTCOME, 'treatment_var': TREATMENT_VAR,
            'coefficient': result['coefficient'], 'std_error': result['std_error'],
            't_stat': result['t_stat'], 'p_value': result['p_value'],
            'ci_lower': result['ci_lower'], 'ci_upper': result['ci_upper'],
            'n_obs': result['n_obs'], 'r_squared': result['r_squared'],
            'coefficient_vector_json': format_coefficient_vector(result, loo_controls),
            'sample_desc': 'Eligible HH, targeting sample',
            'fixed_effects': 'None',
            'controls_desc': f'Full controls minus {drop_var}',
            'cluster_var': 'village_code',
            'model_type': 'OLS',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })
    print(f"  Completed {len(COVARIATES)} leave-one-out specifications")

    # ===========================================
    # SINGLE COVARIATE REGRESSIONS
    # ===========================================
    print("\n=== Running Single Covariate Specifications ===")

    for single_var in COVARIATES:
        result = run_ols_regression(df, PRIMARY_OUTCOME, [single_var], cluster_var='village_code')
        results.append({
            'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
            'spec_id': f'robust/single/{single_var}',
            'spec_tree_path': 'robustness/single_covariate.md',
            'outcome_var': PRIMARY_OUTCOME, 'treatment_var': TREATMENT_VAR,
            'coefficient': result['coefficient'], 'std_error': result['std_error'],
            't_stat': result['t_stat'], 'p_value': result['p_value'],
            'ci_lower': result['ci_lower'], 'ci_upper': result['ci_upper'],
            'n_obs': result['n_obs'], 'r_squared': result['r_squared'],
            'coefficient_vector_json': format_coefficient_vector(result, [single_var]),
            'sample_desc': 'Eligible HH, targeting sample',
            'fixed_effects': 'None',
            'controls_desc': f'Only {single_var}',
            'cluster_var': 'village_code',
            'model_type': 'OLS',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })
    print(f"  Completed {len(COVARIATES)} single covariate specifications")

    # ===========================================
    # SAMPLE RESTRICTIONS
    # ===========================================
    print("\n=== Running Sample Restriction Specifications ===")

    # Winsorized outcomes
    df_wins = df.copy()
    p1, p99 = df_wins[PRIMARY_OUTCOME].quantile([0.01, 0.99])
    df_wins[f'{PRIMARY_OUTCOME}_wins'] = df_wins[PRIMARY_OUTCOME].clip(p1, p99)

    result = run_ols_regression(df_wins, f'{PRIMARY_OUTCOME}_wins', COVARIATES, cluster_var='village_code')
    results.append({
        'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
        'spec_id': 'ols/sample/winsorized',
        'spec_tree_path': 'methods/cross_sectional_ols.md#sample-restrictions',
        'outcome_var': f'{PRIMARY_OUTCOME}_wins', 'treatment_var': TREATMENT_VAR,
        'coefficient': result['coefficient'], 'std_error': result['std_error'],
        't_stat': result['t_stat'], 'p_value': result['p_value'],
        'ci_lower': result['ci_lower'], 'ci_upper': result['ci_upper'],
        'n_obs': result['n_obs'], 'r_squared': result['r_squared'],
        'coefficient_vector_json': format_coefficient_vector(result, COVARIATES),
        'sample_desc': 'Winsorized at 1%/99%',
        'fixed_effects': 'None',
        'controls_desc': 'Full PMT covariates',
        'cluster_var': 'village_code',
        'model_type': 'OLS',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })
    print(f"  Winsorized: coef={result['coefficient']:.2f}, p={result['p_value']:.4f}")

    # Trimmed (drop extremes)
    df_trim = df[(df[PRIMARY_OUTCOME] >= p1) & (df[PRIMARY_OUTCOME] <= p99)].copy()
    result = run_ols_regression(df_trim, PRIMARY_OUTCOME, COVARIATES, cluster_var='village_code')
    results.append({
        'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
        'spec_id': 'ols/sample/trimmed',
        'spec_tree_path': 'methods/cross_sectional_ols.md#sample-restrictions',
        'outcome_var': PRIMARY_OUTCOME, 'treatment_var': TREATMENT_VAR,
        'coefficient': result['coefficient'], 'std_error': result['std_error'],
        't_stat': result['t_stat'], 'p_value': result['p_value'],
        'ci_lower': result['ci_lower'], 'ci_upper': result['ci_upper'],
        'n_obs': result['n_obs'], 'r_squared': result['r_squared'],
        'coefficient_vector_json': format_coefficient_vector(result, COVARIATES),
        'sample_desc': 'Trimmed 1%/99% tails',
        'fixed_effects': 'None',
        'controls_desc': 'Full PMT covariates',
        'cluster_var': 'village_code',
        'model_type': 'OLS',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })
    print(f"  Trimmed: coef={result['coefficient']:.2f}, p={result['p_value']:.4f}")

    # Subgroup: Female-headed households
    df_female = df[df['bl_female_fr'] == 1].copy()
    result = run_ols_regression(df_female, PRIMARY_OUTCOME, [c for c in COVARIATES if c != 'bl_female_fr'],
                                 cluster_var='village_code')
    results.append({
        'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
        'spec_id': 'ols/sample/subgroup_female_head',
        'spec_tree_path': 'methods/cross_sectional_ols.md#sample-restrictions',
        'outcome_var': PRIMARY_OUTCOME, 'treatment_var': TREATMENT_VAR,
        'coefficient': result['coefficient'], 'std_error': result['std_error'],
        't_stat': result['t_stat'], 'p_value': result['p_value'],
        'ci_lower': result['ci_lower'], 'ci_upper': result['ci_upper'],
        'n_obs': result['n_obs'], 'r_squared': result['r_squared'],
        'coefficient_vector_json': format_coefficient_vector(result, [c for c in COVARIATES if c != 'bl_female_fr']),
        'sample_desc': 'Female-headed households only',
        'fixed_effects': 'None',
        'controls_desc': 'Full controls minus bl_female_fr',
        'cluster_var': 'village_code',
        'model_type': 'OLS',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })
    print(f"  Female-headed: coef={result['coefficient']:.2f}, p={result['p_value']:.4f}, n={result['n_obs']}")

    # Subgroup: Male-headed households
    df_male = df[df['bl_female_fr'] == 0].copy()
    result = run_ols_regression(df_male, PRIMARY_OUTCOME, [c for c in COVARIATES if c != 'bl_female_fr'],
                                 cluster_var='village_code')
    results.append({
        'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
        'spec_id': 'ols/sample/subgroup_male_head',
        'spec_tree_path': 'methods/cross_sectional_ols.md#sample-restrictions',
        'outcome_var': PRIMARY_OUTCOME, 'treatment_var': TREATMENT_VAR,
        'coefficient': result['coefficient'], 'std_error': result['std_error'],
        't_stat': result['t_stat'], 'p_value': result['p_value'],
        'ci_lower': result['ci_lower'], 'ci_upper': result['ci_upper'],
        'n_obs': result['n_obs'], 'r_squared': result['r_squared'],
        'coefficient_vector_json': format_coefficient_vector(result, [c for c in COVARIATES if c != 'bl_female_fr']),
        'sample_desc': 'Male-headed households only',
        'fixed_effects': 'None',
        'controls_desc': 'Full controls minus bl_female_fr',
        'cluster_var': 'village_code',
        'model_type': 'OLS',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })
    print(f"  Male-headed: coef={result['coefficient']:.2f}, p={result['p_value']:.4f}, n={result['n_obs']}")

    # Subgroup: Households with children
    df_children = df[df['bl_hh_has_children'] == 1].copy()
    result = run_ols_regression(df_children, PRIMARY_OUTCOME, [c for c in COVARIATES if c != 'bl_hh_has_children'],
                                 cluster_var='village_code')
    results.append({
        'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
        'spec_id': 'ols/sample/subgroup_with_children',
        'spec_tree_path': 'methods/cross_sectional_ols.md#sample-restrictions',
        'outcome_var': PRIMARY_OUTCOME, 'treatment_var': TREATMENT_VAR,
        'coefficient': result['coefficient'], 'std_error': result['std_error'],
        't_stat': result['t_stat'], 'p_value': result['p_value'],
        'ci_lower': result['ci_lower'], 'ci_upper': result['ci_upper'],
        'n_obs': result['n_obs'], 'r_squared': result['r_squared'],
        'coefficient_vector_json': format_coefficient_vector(result, [c for c in COVARIATES if c != 'bl_hh_has_children']),
        'sample_desc': 'Households with children only',
        'fixed_effects': 'None',
        'controls_desc': 'Full controls minus bl_hh_has_children',
        'cluster_var': 'village_code',
        'model_type': 'OLS',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })
    print(f"  With children: coef={result['coefficient']:.2f}, p={result['p_value']:.4f}, n={result['n_obs']}")

    # Subgroup: Widows
    df_widow = df[df['bl_widow'] == 1].copy()
    result = run_ols_regression(df_widow, PRIMARY_OUTCOME, [c for c in COVARIATES if c != 'bl_widow'],
                                 cluster_var='village_code')
    results.append({
        'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
        'spec_id': 'ols/sample/subgroup_widow',
        'spec_tree_path': 'methods/cross_sectional_ols.md#sample-restrictions',
        'outcome_var': PRIMARY_OUTCOME, 'treatment_var': TREATMENT_VAR,
        'coefficient': result['coefficient'], 'std_error': result['std_error'],
        't_stat': result['t_stat'], 'p_value': result['p_value'],
        'ci_lower': result['ci_lower'], 'ci_upper': result['ci_upper'],
        'n_obs': result['n_obs'], 'r_squared': result['r_squared'],
        'coefficient_vector_json': format_coefficient_vector(result, [c for c in COVARIATES if c != 'bl_widow']),
        'sample_desc': 'Widow households only',
        'fixed_effects': 'None',
        'controls_desc': 'Full controls minus bl_widow',
        'cluster_var': 'village_code',
        'model_type': 'OLS',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })
    print(f"  Widows: coef={result['coefficient']:.2f}, p={result['p_value']:.4f}, n={result['n_obs']}")

    # Subgroup: Large households (above median size)
    median_hhsize = df['bl_hhsize'].median()
    df_large = df[df['bl_hhsize'] > median_hhsize].copy()
    result = run_ols_regression(df_large, PRIMARY_OUTCOME, COVARIATES, cluster_var='village_code')
    results.append({
        'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
        'spec_id': 'ols/sample/subgroup_large_hh',
        'spec_tree_path': 'methods/cross_sectional_ols.md#sample-restrictions',
        'outcome_var': PRIMARY_OUTCOME, 'treatment_var': TREATMENT_VAR,
        'coefficient': result['coefficient'], 'std_error': result['std_error'],
        't_stat': result['t_stat'], 'p_value': result['p_value'],
        'ci_lower': result['ci_lower'], 'ci_upper': result['ci_upper'],
        'n_obs': result['n_obs'], 'r_squared': result['r_squared'],
        'coefficient_vector_json': format_coefficient_vector(result, COVARIATES),
        'sample_desc': f'Large households (HH size > {median_hhsize:.1f})',
        'fixed_effects': 'None',
        'controls_desc': 'Full PMT covariates',
        'cluster_var': 'village_code',
        'model_type': 'OLS',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })
    print(f"  Large HH: coef={result['coefficient']:.2f}, p={result['p_value']:.4f}, n={result['n_obs']}")

    # Subgroup: Small households
    df_small = df[df['bl_hhsize'] <= median_hhsize].copy()
    result = run_ols_regression(df_small, PRIMARY_OUTCOME, COVARIATES, cluster_var='village_code')
    results.append({
        'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
        'spec_id': 'ols/sample/subgroup_small_hh',
        'spec_tree_path': 'methods/cross_sectional_ols.md#sample-restrictions',
        'outcome_var': PRIMARY_OUTCOME, 'treatment_var': TREATMENT_VAR,
        'coefficient': result['coefficient'], 'std_error': result['std_error'],
        't_stat': result['t_stat'], 'p_value': result['p_value'],
        'ci_lower': result['ci_lower'], 'ci_upper': result['ci_upper'],
        'n_obs': result['n_obs'], 'r_squared': result['r_squared'],
        'coefficient_vector_json': format_coefficient_vector(result, COVARIATES),
        'sample_desc': f'Small households (HH size <= {median_hhsize:.1f})',
        'fixed_effects': 'None',
        'controls_desc': 'Full PMT covariates',
        'cluster_var': 'village_code',
        'model_type': 'OLS',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })
    print(f"  Small HH: coef={result['coefficient']:.2f}, p={result['p_value']:.4f}, n={result['n_obs']}")

    # ===========================================
    # FUNCTIONAL FORM VARIATIONS
    # ===========================================
    print("\n=== Running Functional Form Variations ===")

    # Log outcome
    df_log = df.copy()
    df_log['log_consumption'] = np.log(df_log[PRIMARY_OUTCOME] + 1)

    result = run_ols_regression(df_log, 'log_consumption', COVARIATES, cluster_var='village_code')
    results.append({
        'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
        'spec_id': 'ols/form/log_dep',
        'spec_tree_path': 'methods/cross_sectional_ols.md#functional-form',
        'outcome_var': 'log(consumption+1)', 'treatment_var': TREATMENT_VAR,
        'coefficient': result['coefficient'], 'std_error': result['std_error'],
        't_stat': result['t_stat'], 'p_value': result['p_value'],
        'ci_lower': result['ci_lower'], 'ci_upper': result['ci_upper'],
        'n_obs': result['n_obs'], 'r_squared': result['r_squared'],
        'coefficient_vector_json': format_coefficient_vector(result, COVARIATES),
        'sample_desc': 'Eligible HH, targeting sample',
        'fixed_effects': 'None',
        'controls_desc': 'Full PMT covariates',
        'cluster_var': 'village_code',
        'model_type': 'OLS (log outcome)',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })
    print(f"  Log outcome: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # IHS transformation
    df_ihs = df.copy()
    df_ihs['ihs_consumption'] = np.arcsinh(df_ihs[PRIMARY_OUTCOME])

    result = run_ols_regression(df_ihs, 'ihs_consumption', COVARIATES, cluster_var='village_code')
    results.append({
        'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
        'spec_id': 'ols/form/ihs',
        'spec_tree_path': 'methods/cross_sectional_ols.md#functional-form',
        'outcome_var': 'asinh(consumption)', 'treatment_var': TREATMENT_VAR,
        'coefficient': result['coefficient'], 'std_error': result['std_error'],
        't_stat': result['t_stat'], 'p_value': result['p_value'],
        'ci_lower': result['ci_lower'], 'ci_upper': result['ci_upper'],
        'n_obs': result['n_obs'], 'r_squared': result['r_squared'],
        'coefficient_vector_json': format_coefficient_vector(result, COVARIATES),
        'sample_desc': 'Eligible HH, targeting sample',
        'fixed_effects': 'None',
        'controls_desc': 'Full PMT covariates',
        'cluster_var': 'village_code',
        'model_type': 'OLS (IHS outcome)',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })
    print(f"  IHS outcome: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # ===========================================
    # ALTERNATIVE OUTCOME MEASURES
    # ===========================================
    print("\n=== Running Alternative Outcome Specifications ===")

    # Per capita consumption
    df_pc = df.copy()
    df_pc['cons_pc'] = df_pc[PRIMARY_OUTCOME] / df_pc['el_hhsize']

    result = run_ols_regression(df_pc, 'cons_pc', COVARIATES, cluster_var='village_code')
    results.append({
        'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
        'spec_id': 'custom/outcome_per_capita',
        'spec_tree_path': 'custom',
        'outcome_var': 'consumption_per_capita', 'treatment_var': TREATMENT_VAR,
        'coefficient': result['coefficient'], 'std_error': result['std_error'],
        't_stat': result['t_stat'], 'p_value': result['p_value'],
        'ci_lower': result['ci_lower'], 'ci_upper': result['ci_upper'],
        'n_obs': result['n_obs'], 'r_squared': result['r_squared'],
        'coefficient_vector_json': format_coefficient_vector(result, COVARIATES),
        'sample_desc': 'Eligible HH, targeting sample',
        'fixed_effects': 'None',
        'controls_desc': 'Full PMT covariates',
        'cluster_var': 'village_code',
        'model_type': 'OLS',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })
    print(f"  Per capita: coef={result['coefficient']:.2f}, p={result['p_value']:.4f}")

    return results


def main():
    """Main execution."""
    print("=" * 60)
    print(f"Specification Search: {PAPER_ID}")
    print(f"Paper: {PAPER_TITLE}")
    print("=" * 60)

    # Load data
    df = load_data()

    # Run all specifications
    results = run_all_specifications(df)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    output_file = f"{OUTPUT_PATH}/specification_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n\nSaved {len(results_df)} specifications to {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Filter to main outcome (consumption variants)
    main_results = results_df[results_df['outcome_var'].isin([
        PRIMARY_OUTCOME, 'log(consumption+1)', 'asinh(consumption)',
        f'{PRIMARY_OUTCOME}_wins', 'consumption_per_capita'
    ])].copy()

    print(f"Total specifications: {len(results_df)}")
    print(f"Main outcome (consumption) specifications: {len(main_results)}")

    valid = main_results[main_results['coefficient'].notna()]
    print(f"\nValid estimates: {len(valid)}")
    print(f"Positive coefficients: {(valid['coefficient'] > 0).sum()} ({100*(valid['coefficient'] > 0).mean():.1f}%)")
    print(f"Significant at 5%: {(valid['p_value'] < 0.05).sum()} ({100*(valid['p_value'] < 0.05).mean():.1f}%)")
    print(f"Significant at 1%: {(valid['p_value'] < 0.01).sum()} ({100*(valid['p_value'] < 0.01).mean():.1f}%)")

    # Get baseline coefficient for reference
    baseline = main_results[main_results['spec_id'] == 'baseline']
    if len(baseline) > 0:
        baseline_coef = baseline['coefficient'].values[0]
        print(f"\nBaseline coefficient: {baseline_coef:.2f}")

    # Print coefficient range (excluding log/IHS transforms)
    level_specs = valid[~valid['outcome_var'].str.contains('log|asinh', case=False, na=False)]
    if len(level_specs) > 0:
        print(f"\nCoefficient range (level specs): [{level_specs['coefficient'].min():.2f}, {level_specs['coefficient'].max():.2f}]")
        print(f"Median coefficient: {level_specs['coefficient'].median():.2f}")
        print(f"Mean coefficient: {level_specs['coefficient'].mean():.2f}")

    return results_df


if __name__ == "__main__":
    results_df = main()
