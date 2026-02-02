"""
Specification Search: Digital Addiction (163822-V2)
Allcott, Gentzkow, and Song (2022) AER

This script runs a systematic specification search on the paper's main results.

Paper Overview:
- RCT studying effect of screen-time limits and bonus incentives on smartphone usage
- Treatment 1: Bonus - paid for reducing usage
- Treatment 2: Limit - app-imposed usage limits
- Main outcome: Phone usage (minutes/day) on FITSBY apps
- Secondary outcomes: Subjective well-being, addiction indices

Method Classification: Cross-sectional OLS (Randomized Experiment)
- No panel structure in main specifications
- Stratified randomization with controls
- Robust standard errors clustered at individual level
"""

import pandas as pd
import numpy as np
import json
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

# Try to import pyfixest, fall back to statsmodels if not available
try:
    import pyfixest as pf
    USE_PYFIXEST = True
except ImportError:
    USE_PYFIXEST = False

import statsmodels.api as sm
import statsmodels.formula.api as smf

# =============================================================================
# Configuration
# =============================================================================

PAPER_ID = "163822-V2"
PAPER_TITLE = "Digital Addiction"
JOURNAL = "AER"
DATA_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/163822-V2/data/temptation/output/final_data_sample.tab"
OUTPUT_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/163822-V2"

# =============================================================================
# Data Loading and Preparation
# =============================================================================

def load_data():
    """Load and prepare the data"""
    df = pd.read_csv(DATA_PATH, sep='\t', low_memory=False)

    # Create treatment indicators
    # B = Bonus treatment (S3_Bonus == 1)
    # L = Limit treatment (S2_LimitType != 0)
    df['B'] = (df['S3_Bonus'] == 1).astype(int)
    df['L'] = (df['S2_LimitType'] != 0).astype(int)

    # Clean LimitType for clearer coding
    df['LimitType'] = df['S2_LimitType'].fillna(0).astype(int)

    # Create stratification dummies from Stratifier variable
    # Stratifier encodes stratification groups
    if 'Stratifier' in df.columns:
        df['Strat'] = df['Stratifier'].fillna(0).astype(int)
        # Create dummies for stratification
        strat_dummies = pd.get_dummies(df['Strat'], prefix='strat', drop_first=True)
        df = pd.concat([df, strat_dummies], axis=1)

    # Normalize outcome variables (create _N versions if not present)
    # These are standardized versions of indices
    for var in ['PhoneUseChange', 'AddictionIndex', 'SMSIndex', 'SWBIndex', 'LifeBetter']:
        for survey in ['S1', 'S3', 'S4']:
            raw_var = f'{survey}_{var}'
            norm_var = f'{survey}_{var}_N'
            if raw_var in df.columns and norm_var not in df.columns:
                # Standardize to mean 0, sd 1
                df[norm_var] = (df[raw_var] - df[raw_var].mean()) / df[raw_var].std()

    # Create demographic controls
    df['Female'] = (df['S0_Gender'] == 2).astype(float) if 'S0_Gender' in df.columns else np.nan
    df['Age'] = df['S0_Age'] if 'S0_Age' in df.columns else np.nan
    df['Age_sq'] = df['Age'] ** 2 if 'Age' in df.columns else np.nan

    # Education - higher = more education
    df['Education'] = df['S1_Education'] if 'S1_Education' in df.columns else np.nan

    # Income
    df['Income'] = df['S1_Income'] if 'S1_Income' in df.columns else np.nan

    # Baseline usage
    df['Baseline_Usage'] = df['PD_P1_UsageFITSBY'] if 'PD_P1_UsageFITSBY' in df.columns else np.nan
    df['Baseline_Usage_Total'] = df['PD_P1_Usage'] if 'PD_P1_Usage' in df.columns else np.nan

    # Create heterogeneity indicators (median splits)
    for var, newvar in [('Age', 'High_Age'), ('Baseline_Usage', 'High_Usage'),
                        ('S1_Education', 'High_Education'), ('S1_Income', 'High_Income')]:
        if var in df.columns:
            median_val = df[var].median()
            df[newvar] = (df[var] >= median_val).astype(float)

    if 'StratAddictionLifeIndex' in df.columns:
        median_val = df['StratAddictionLifeIndex'].median()
        df['High_Addiction'] = (df['StratAddictionLifeIndex'] >= median_val).astype(float)

    if 'StratWantRestrictionIndex' in df.columns:
        median_val = df['StratWantRestrictionIndex'].median()
        df['High_Restriction'] = (df['StratWantRestrictionIndex'] >= median_val).astype(float)

    return df

# =============================================================================
# Regression Helper Functions
# =============================================================================

def run_ols(df, formula, cluster_var=None, weights=None):
    """
    Run OLS regression with optional clustering and weights.
    Returns dictionary with results.
    """
    try:
        # Clean data for this regression
        df_clean = df.dropna(subset=[v.strip() for v in formula.replace('~', '+').replace('*', '+').replace('|', '+').split('+') if v.strip() and not v.strip().startswith('C(') and not v.strip().startswith('np.')])
    except:
        df_clean = df.dropna()

    if len(df_clean) < 30:
        return None

    try:
        if weights is not None and weights in df_clean.columns:
            model = smf.wls(formula, data=df_clean, weights=df_clean[weights]).fit(cov_type='HC1')
        else:
            model = smf.ols(formula, data=df_clean).fit(cov_type='HC1')

        return {
            'model': model,
            'n_obs': int(model.nobs),
            'r_squared': model.rsquared,
            'params': model.params,
            'bse': model.bse,
            'pvalues': model.pvalues,
            'conf_int': model.conf_int()
        }
    except Exception as e:
        print(f"Error in regression: {e}")
        return None


def extract_treatment_effect(result, treatment_var):
    """Extract treatment effect from regression result"""
    if result is None:
        return None

    params = result['params']
    bse = result['bse']
    pvals = result['pvalues']
    conf_int = result['conf_int']

    # Find the treatment variable
    treat_vars = [v for v in params.index if treatment_var in v and 'Intercept' not in v]
    if not treat_vars:
        return None

    treat_var = treat_vars[0]

    coef = params[treat_var]
    se = bse[treat_var]
    pval = pvals[treat_var]
    ci_lower = conf_int.loc[treat_var, 0]
    ci_upper = conf_int.loc[treat_var, 1]
    t_stat = coef / se if se > 0 else np.nan

    return {
        'coefficient': coef,
        'std_error': se,
        't_stat': t_stat,
        'p_value': pval,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }


def create_coef_vector_json(result, treatment_var, controls, fixed_effects=None):
    """Create JSON representation of full coefficient vector"""
    if result is None:
        return None

    params = result['params']
    bse = result['bse']
    pvals = result['pvalues']

    # Treatment
    treat_vars = [v for v in params.index if treatment_var in v and 'Intercept' not in v]
    treat_var = treat_vars[0] if treat_vars else None

    treatment_dict = None
    if treat_var:
        treatment_dict = {
            'var': treat_var,
            'coef': float(params[treat_var]),
            'se': float(bse[treat_var]),
            'pval': float(pvals[treat_var])
        }

    # Controls
    controls_list = []
    for var in params.index:
        if var != 'Intercept' and var != treat_var:
            controls_list.append({
                'var': var,
                'coef': float(params[var]),
                'se': float(bse[var]),
                'pval': float(pvals[var])
            })

    return json.dumps({
        'treatment': treatment_dict,
        'controls': controls_list,
        'fixed_effects': fixed_effects if fixed_effects else [],
        'diagnostics': {
            'r_squared': float(result['r_squared']),
            'n_obs': int(result['n_obs'])
        }
    })


# =============================================================================
# Specification Definitions
# =============================================================================

def get_stratification_controls(df):
    """Get list of stratification control column names"""
    strat_cols = [c for c in df.columns if c.startswith('strat_')]
    return strat_cols


def run_all_specifications(df):
    """Run all specifications and return results list"""
    results = []

    # Get stratification controls
    strat_controls = get_stratification_controls(df)
    strat_formula = ' + '.join(strat_controls) if strat_controls else ''

    # Main outcome variable
    main_outcome = 'PD_P3_UsageFITSBY'
    baseline_control = 'PD_P1_UsageFITSBY'

    # Alternative outcomes
    alt_outcomes = {
        'usage_total': 'PD_P3_Usage',
        'usage_p4': 'PD_P4_UsageFITSBY',
        'usage_p432': 'PD_P432_UsageFITSBY',
        'usage_p5432': 'PD_P5432_UsageFITSBY',
    }

    # SWB outcomes
    swb_outcomes = {
        'swb_index': 'S3_SWBIndex_N',
        'addiction_index': 'S3_AddictionIndex_N',
        'sms_index': 'S3_SMSIndex_N',
    }

    # Demographic controls
    demo_controls = ['Age', 'Female', 'Education', 'Income']
    demo_formula = ' + '.join([c for c in demo_controls if c in df.columns and df[c].notna().sum() > 100])

    spec_counter = 0

    # =========================================================================
    # BASELINE SPECIFICATIONS (Treatment: Bonus)
    # =========================================================================

    print("Running baseline specifications...")

    # 1. Baseline - Bonus effect on FITSBY usage
    formula = f'{main_outcome} ~ B + {baseline_control}'
    if strat_formula:
        formula += f' + {strat_formula}'

    result = run_ols(df, formula)
    if result:
        treat_effect = extract_treatment_effect(result, 'B')
        if treat_effect:
            spec_counter += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': 'baseline',
                'spec_tree_path': 'methods/cross_sectional_ols.md',
                'outcome_var': main_outcome,
                'treatment_var': 'B (Bonus)',
                'coefficient': treat_effect['coefficient'],
                'std_error': treat_effect['std_error'],
                't_stat': treat_effect['t_stat'],
                'p_value': treat_effect['p_value'],
                'ci_lower': treat_effect['ci_lower'],
                'ci_upper': treat_effect['ci_upper'],
                'n_obs': result['n_obs'],
                'r_squared': result['r_squared'],
                'coefficient_vector_json': create_coef_vector_json(result, 'B', strat_controls),
                'sample_desc': 'Full sample',
                'fixed_effects': 'None',
                'controls_desc': f'Baseline usage + stratification',
                'cluster_var': 'UserID (robust SE)',
                'model_type': 'OLS',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })

    # 2. Baseline - Limit effect on FITSBY usage
    formula = f'{main_outcome} ~ L + {baseline_control}'
    if strat_formula:
        formula += f' + {strat_formula}'

    result = run_ols(df, formula)
    if result:
        treat_effect = extract_treatment_effect(result, 'L')
        if treat_effect:
            spec_counter += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': 'baseline_limit',
                'spec_tree_path': 'methods/cross_sectional_ols.md',
                'outcome_var': main_outcome,
                'treatment_var': 'L (Limit)',
                'coefficient': treat_effect['coefficient'],
                'std_error': treat_effect['std_error'],
                't_stat': treat_effect['t_stat'],
                'p_value': treat_effect['p_value'],
                'ci_lower': treat_effect['ci_lower'],
                'ci_upper': treat_effect['ci_upper'],
                'n_obs': result['n_obs'],
                'r_squared': result['r_squared'],
                'coefficient_vector_json': create_coef_vector_json(result, 'L', strat_controls),
                'sample_desc': 'Full sample',
                'fixed_effects': 'None',
                'controls_desc': f'Baseline usage + stratification',
                'cluster_var': 'UserID (robust SE)',
                'model_type': 'OLS',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })

    # =========================================================================
    # CONTROL VARIATIONS
    # =========================================================================

    print("Running control variations...")

    # 3. No controls (bivariate)
    formula = f'{main_outcome} ~ B'
    result = run_ols(df, formula)
    if result:
        treat_effect = extract_treatment_effect(result, 'B')
        if treat_effect:
            spec_counter += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': 'robust/control/none',
                'spec_tree_path': 'robustness/control_progression.md',
                'outcome_var': main_outcome,
                'treatment_var': 'B (Bonus)',
                'coefficient': treat_effect['coefficient'],
                'std_error': treat_effect['std_error'],
                't_stat': treat_effect['t_stat'],
                'p_value': treat_effect['p_value'],
                'ci_lower': treat_effect['ci_lower'],
                'ci_upper': treat_effect['ci_upper'],
                'n_obs': result['n_obs'],
                'r_squared': result['r_squared'],
                'coefficient_vector_json': create_coef_vector_json(result, 'B', []),
                'sample_desc': 'Full sample',
                'fixed_effects': 'None',
                'controls_desc': 'No controls (bivariate)',
                'cluster_var': 'UserID (robust SE)',
                'model_type': 'OLS',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })

    # 4. Only baseline usage control
    formula = f'{main_outcome} ~ B + {baseline_control}'
    result = run_ols(df, formula)
    if result:
        treat_effect = extract_treatment_effect(result, 'B')
        if treat_effect:
            spec_counter += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': 'robust/control/baseline_only',
                'spec_tree_path': 'robustness/control_progression.md',
                'outcome_var': main_outcome,
                'treatment_var': 'B (Bonus)',
                'coefficient': treat_effect['coefficient'],
                'std_error': treat_effect['std_error'],
                't_stat': treat_effect['t_stat'],
                'p_value': treat_effect['p_value'],
                'ci_lower': treat_effect['ci_lower'],
                'ci_upper': treat_effect['ci_upper'],
                'n_obs': result['n_obs'],
                'r_squared': result['r_squared'],
                'coefficient_vector_json': create_coef_vector_json(result, 'B', [baseline_control]),
                'sample_desc': 'Full sample',
                'fixed_effects': 'None',
                'controls_desc': 'Baseline usage only',
                'cluster_var': 'UserID (robust SE)',
                'model_type': 'OLS',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })

    # 5. Full demographic controls
    if demo_formula:
        formula = f'{main_outcome} ~ B + {baseline_control} + {demo_formula}'
        if strat_formula:
            formula += f' + {strat_formula}'

        result = run_ols(df, formula)
        if result:
            treat_effect = extract_treatment_effect(result, 'B')
            if treat_effect:
                spec_counter += 1
                results.append({
                    'paper_id': PAPER_ID,
                    'journal': JOURNAL,
                    'paper_title': PAPER_TITLE,
                    'spec_id': 'robust/control/full_demographics',
                    'spec_tree_path': 'robustness/control_progression.md',
                    'outcome_var': main_outcome,
                    'treatment_var': 'B (Bonus)',
                    'coefficient': treat_effect['coefficient'],
                    'std_error': treat_effect['std_error'],
                    't_stat': treat_effect['t_stat'],
                    'p_value': treat_effect['p_value'],
                    'ci_lower': treat_effect['ci_lower'],
                    'ci_upper': treat_effect['ci_upper'],
                    'n_obs': result['n_obs'],
                    'r_squared': result['r_squared'],
                    'coefficient_vector_json': create_coef_vector_json(result, 'B', demo_controls + strat_controls),
                    'sample_desc': 'Full sample',
                    'fixed_effects': 'None',
                    'controls_desc': 'Baseline + demographics + stratification',
                    'cluster_var': 'UserID (robust SE)',
                    'model_type': 'OLS',
                    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
                })

    # 6-10. Leave-one-out on stratification controls
    if strat_controls:
        for drop_control in strat_controls[:5]:  # Drop first 5 strat controls
            remaining = [c for c in strat_controls if c != drop_control]
            if remaining:
                formula = f'{main_outcome} ~ B + {baseline_control} + ' + ' + '.join(remaining)
                result = run_ols(df, formula)
                if result:
                    treat_effect = extract_treatment_effect(result, 'B')
                    if treat_effect:
                        spec_counter += 1
                        results.append({
                            'paper_id': PAPER_ID,
                            'journal': JOURNAL,
                            'paper_title': PAPER_TITLE,
                            'spec_id': f'robust/control/drop_{drop_control}',
                            'spec_tree_path': 'robustness/leave_one_out.md',
                            'outcome_var': main_outcome,
                            'treatment_var': 'B (Bonus)',
                            'coefficient': treat_effect['coefficient'],
                            'std_error': treat_effect['std_error'],
                            't_stat': treat_effect['t_stat'],
                            'p_value': treat_effect['p_value'],
                            'ci_lower': treat_effect['ci_lower'],
                            'ci_upper': treat_effect['ci_upper'],
                            'n_obs': result['n_obs'],
                            'r_squared': result['r_squared'],
                            'coefficient_vector_json': create_coef_vector_json(result, 'B', remaining),
                            'sample_desc': 'Full sample',
                            'fixed_effects': 'None',
                            'controls_desc': f'Drop {drop_control}',
                            'cluster_var': 'UserID (robust SE)',
                            'model_type': 'OLS',
                            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
                        })

    # Add controls incrementally
    print("Running incremental control additions...")
    base_controls = [baseline_control] + (strat_controls[:3] if strat_controls else [])
    for i, ctrl in enumerate(base_controls):
        controls_so_far = base_controls[:i+1]
        formula = f'{main_outcome} ~ B + ' + ' + '.join(controls_so_far)
        result = run_ols(df, formula)
        if result:
            treat_effect = extract_treatment_effect(result, 'B')
            if treat_effect:
                spec_counter += 1
                results.append({
                    'paper_id': PAPER_ID,
                    'journal': JOURNAL,
                    'paper_title': PAPER_TITLE,
                    'spec_id': f'robust/control/add_{ctrl}',
                    'spec_tree_path': 'robustness/control_progression.md',
                    'outcome_var': main_outcome,
                    'treatment_var': 'B (Bonus)',
                    'coefficient': treat_effect['coefficient'],
                    'std_error': treat_effect['std_error'],
                    't_stat': treat_effect['t_stat'],
                    'p_value': treat_effect['p_value'],
                    'ci_lower': treat_effect['ci_lower'],
                    'ci_upper': treat_effect['ci_upper'],
                    'n_obs': result['n_obs'],
                    'r_squared': result['r_squared'],
                    'coefficient_vector_json': create_coef_vector_json(result, 'B', controls_so_far),
                    'sample_desc': 'Full sample',
                    'fixed_effects': 'None',
                    'controls_desc': f'Controls up to {ctrl}',
                    'cluster_var': 'UserID (robust SE)',
                    'model_type': 'OLS',
                    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
                })

    # =========================================================================
    # ALTERNATIVE OUTCOMES
    # =========================================================================

    print("Running alternative outcomes...")

    # Usage outcomes
    for outcome_name, outcome_var in alt_outcomes.items():
        if outcome_var in df.columns:
            formula = f'{outcome_var} ~ B + {baseline_control}'
            if strat_formula:
                formula += f' + {strat_formula}'

            result = run_ols(df, formula)
            if result:
                treat_effect = extract_treatment_effect(result, 'B')
                if treat_effect:
                    spec_counter += 1
                    results.append({
                        'paper_id': PAPER_ID,
                        'journal': JOURNAL,
                        'paper_title': PAPER_TITLE,
                        'spec_id': f'robust/outcome/{outcome_name}',
                        'spec_tree_path': 'robustness/measurement.md',
                        'outcome_var': outcome_var,
                        'treatment_var': 'B (Bonus)',
                        'coefficient': treat_effect['coefficient'],
                        'std_error': treat_effect['std_error'],
                        't_stat': treat_effect['t_stat'],
                        'p_value': treat_effect['p_value'],
                        'ci_lower': treat_effect['ci_lower'],
                        'ci_upper': treat_effect['ci_upper'],
                        'n_obs': result['n_obs'],
                        'r_squared': result['r_squared'],
                        'coefficient_vector_json': create_coef_vector_json(result, 'B', strat_controls),
                        'sample_desc': 'Full sample',
                        'fixed_effects': 'None',
                        'controls_desc': 'Baseline usage + stratification',
                        'cluster_var': 'UserID (robust SE)',
                        'model_type': 'OLS',
                        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
                    })

    # SWB outcomes
    for outcome_name, outcome_var in swb_outcomes.items():
        if outcome_var in df.columns:
            baseline_swb = outcome_var.replace('S3_', 'S1_')
            formula = f'{outcome_var} ~ B'
            if baseline_swb in df.columns:
                formula += f' + {baseline_swb}'
            if strat_formula:
                formula += f' + {strat_formula}'

            result = run_ols(df, formula)
            if result:
                treat_effect = extract_treatment_effect(result, 'B')
                if treat_effect:
                    spec_counter += 1
                    results.append({
                        'paper_id': PAPER_ID,
                        'journal': JOURNAL,
                        'paper_title': PAPER_TITLE,
                        'spec_id': f'robust/outcome/{outcome_name}',
                        'spec_tree_path': 'robustness/measurement.md',
                        'outcome_var': outcome_var,
                        'treatment_var': 'B (Bonus)',
                        'coefficient': treat_effect['coefficient'],
                        'std_error': treat_effect['std_error'],
                        't_stat': treat_effect['t_stat'],
                        'p_value': treat_effect['p_value'],
                        'ci_lower': treat_effect['ci_lower'],
                        'ci_upper': treat_effect['ci_upper'],
                        'n_obs': result['n_obs'],
                        'r_squared': result['r_squared'],
                        'coefficient_vector_json': create_coef_vector_json(result, 'B', strat_controls),
                        'sample_desc': 'Full sample',
                        'fixed_effects': 'None',
                        'controls_desc': 'Baseline + stratification',
                        'cluster_var': 'UserID (robust SE)',
                        'model_type': 'OLS',
                        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
                    })

    # App-specific outcomes
    print("Running app-specific outcomes...")
    apps = ['Facebook', 'Instagram', 'Twitter', 'Snapchat', 'Browser', 'YouTube']
    for app in apps:
        outcome_var = f'PD_P3_Usage_{app}'
        baseline_var = f'PD_P1_Usage_{app}'
        if outcome_var in df.columns and baseline_var in df.columns:
            formula = f'{outcome_var} ~ B + {baseline_var}'
            if strat_formula:
                formula += f' + {strat_formula}'

            result = run_ols(df, formula)
            if result:
                treat_effect = extract_treatment_effect(result, 'B')
                if treat_effect:
                    spec_counter += 1
                    results.append({
                        'paper_id': PAPER_ID,
                        'journal': JOURNAL,
                        'paper_title': PAPER_TITLE,
                        'spec_id': f'robust/outcome/app_{app.lower()}',
                        'spec_tree_path': 'robustness/measurement.md',
                        'outcome_var': outcome_var,
                        'treatment_var': 'B (Bonus)',
                        'coefficient': treat_effect['coefficient'],
                        'std_error': treat_effect['std_error'],
                        't_stat': treat_effect['t_stat'],
                        'p_value': treat_effect['p_value'],
                        'ci_lower': treat_effect['ci_lower'],
                        'ci_upper': treat_effect['ci_upper'],
                        'n_obs': result['n_obs'],
                        'r_squared': result['r_squared'],
                        'coefficient_vector_json': create_coef_vector_json(result, 'B', strat_controls),
                        'sample_desc': 'Full sample',
                        'fixed_effects': 'None',
                        'controls_desc': f'Baseline {app} usage + stratification',
                        'cluster_var': 'UserID (robust SE)',
                        'model_type': 'OLS',
                        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
                    })

    # =========================================================================
    # SAMPLE RESTRICTIONS
    # =========================================================================

    print("Running sample restrictions...")

    # By gender
    for gender_val, gender_name in [(1, 'male'), (2, 'female')]:
        if 'S0_Gender' in df.columns:
            df_sub = df[df['S0_Gender'] == gender_val]
            if len(df_sub) > 50:
                formula = f'{main_outcome} ~ B + {baseline_control}'
                if strat_formula:
                    formula += f' + {strat_formula}'

                result = run_ols(df_sub, formula)
                if result:
                    treat_effect = extract_treatment_effect(result, 'B')
                    if treat_effect:
                        spec_counter += 1
                        results.append({
                            'paper_id': PAPER_ID,
                            'journal': JOURNAL,
                            'paper_title': PAPER_TITLE,
                            'spec_id': f'robust/sample/{gender_name}',
                            'spec_tree_path': 'robustness/sample_restrictions.md',
                            'outcome_var': main_outcome,
                            'treatment_var': 'B (Bonus)',
                            'coefficient': treat_effect['coefficient'],
                            'std_error': treat_effect['std_error'],
                            't_stat': treat_effect['t_stat'],
                            'p_value': treat_effect['p_value'],
                            'ci_lower': treat_effect['ci_lower'],
                            'ci_upper': treat_effect['ci_upper'],
                            'n_obs': result['n_obs'],
                            'r_squared': result['r_squared'],
                            'coefficient_vector_json': create_coef_vector_json(result, 'B', strat_controls),
                            'sample_desc': f'{gender_name.capitalize()} only',
                            'fixed_effects': 'None',
                            'controls_desc': 'Baseline usage + stratification',
                            'cluster_var': 'UserID (robust SE)',
                            'model_type': 'OLS',
                            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
                        })

    # By age (median split)
    if 'Age' in df.columns:
        age_median = df['Age'].median()
        for condition, name in [((df['Age'] < age_median), 'young'), ((df['Age'] >= age_median), 'old')]:
            df_sub = df[condition]
            if len(df_sub) > 50:
                formula = f'{main_outcome} ~ B + {baseline_control}'
                if strat_formula:
                    formula += f' + {strat_formula}'

                result = run_ols(df_sub, formula)
                if result:
                    treat_effect = extract_treatment_effect(result, 'B')
                    if treat_effect:
                        spec_counter += 1
                        results.append({
                            'paper_id': PAPER_ID,
                            'journal': JOURNAL,
                            'paper_title': PAPER_TITLE,
                            'spec_id': f'robust/sample/{name}',
                            'spec_tree_path': 'robustness/sample_restrictions.md',
                            'outcome_var': main_outcome,
                            'treatment_var': 'B (Bonus)',
                            'coefficient': treat_effect['coefficient'],
                            'std_error': treat_effect['std_error'],
                            't_stat': treat_effect['t_stat'],
                            'p_value': treat_effect['p_value'],
                            'ci_lower': treat_effect['ci_lower'],
                            'ci_upper': treat_effect['ci_upper'],
                            'n_obs': result['n_obs'],
                            'r_squared': result['r_squared'],
                            'coefficient_vector_json': create_coef_vector_json(result, 'B', strat_controls),
                            'sample_desc': f'{name.capitalize()} (age {"<" if name=="young" else ">="} {age_median:.0f})',
                            'fixed_effects': 'None',
                            'controls_desc': 'Baseline usage + stratification',
                            'cluster_var': 'UserID (robust SE)',
                            'model_type': 'OLS',
                            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
                        })

    # By baseline usage (high/low users)
    if 'Baseline_Usage' in df.columns:
        usage_median = df['Baseline_Usage'].median()
        for condition, name in [((df['Baseline_Usage'] < usage_median), 'low_usage'),
                                ((df['Baseline_Usage'] >= usage_median), 'high_usage')]:
            df_sub = df[condition]
            if len(df_sub) > 50:
                formula = f'{main_outcome} ~ B + {baseline_control}'
                if strat_formula:
                    formula += f' + {strat_formula}'

                result = run_ols(df_sub, formula)
                if result:
                    treat_effect = extract_treatment_effect(result, 'B')
                    if treat_effect:
                        spec_counter += 1
                        results.append({
                            'paper_id': PAPER_ID,
                            'journal': JOURNAL,
                            'paper_title': PAPER_TITLE,
                            'spec_id': f'robust/sample/{name}',
                            'spec_tree_path': 'robustness/sample_restrictions.md',
                            'outcome_var': main_outcome,
                            'treatment_var': 'B (Bonus)',
                            'coefficient': treat_effect['coefficient'],
                            'std_error': treat_effect['std_error'],
                            't_stat': treat_effect['t_stat'],
                            'p_value': treat_effect['p_value'],
                            'ci_lower': treat_effect['ci_lower'],
                            'ci_upper': treat_effect['ci_upper'],
                            'n_obs': result['n_obs'],
                            'r_squared': result['r_squared'],
                            'coefficient_vector_json': create_coef_vector_json(result, 'B', strat_controls),
                            'sample_desc': f'{name.replace("_", " ").title()} (usage {"<" if "low" in name else ">="} {usage_median:.0f})',
                            'fixed_effects': 'None',
                            'controls_desc': 'Baseline usage + stratification',
                            'cluster_var': 'UserID (robust SE)',
                            'model_type': 'OLS',
                            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
                        })

    # By education
    if 'Education' in df.columns:
        edu_median = df['Education'].median()
        for condition, name in [((df['Education'] < edu_median), 'low_education'),
                                ((df['Education'] >= edu_median), 'high_education')]:
            df_sub = df[condition]
            if len(df_sub) > 50:
                formula = f'{main_outcome} ~ B + {baseline_control}'
                if strat_formula:
                    formula += f' + {strat_formula}'

                result = run_ols(df_sub, formula)
                if result:
                    treat_effect = extract_treatment_effect(result, 'B')
                    if treat_effect:
                        spec_counter += 1
                        results.append({
                            'paper_id': PAPER_ID,
                            'journal': JOURNAL,
                            'paper_title': PAPER_TITLE,
                            'spec_id': f'robust/sample/{name}',
                            'spec_tree_path': 'robustness/sample_restrictions.md',
                            'outcome_var': main_outcome,
                            'treatment_var': 'B (Bonus)',
                            'coefficient': treat_effect['coefficient'],
                            'std_error': treat_effect['std_error'],
                            't_stat': treat_effect['t_stat'],
                            'p_value': treat_effect['p_value'],
                            'ci_lower': treat_effect['ci_lower'],
                            'ci_upper': treat_effect['ci_upper'],
                            'n_obs': result['n_obs'],
                            'r_squared': result['r_squared'],
                            'coefficient_vector_json': create_coef_vector_json(result, 'B', strat_controls),
                            'sample_desc': f'{name.replace("_", " ").title()}',
                            'fixed_effects': 'None',
                            'controls_desc': 'Baseline usage + stratification',
                            'cluster_var': 'UserID (robust SE)',
                            'model_type': 'OLS',
                            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
                        })

    # Outlier treatment - winsorize
    print("Running outlier treatments...")
    for pct in [1, 5, 10]:
        df_wins = df.copy()
        lower = df_wins[main_outcome].quantile(pct/100)
        upper = df_wins[main_outcome].quantile(1 - pct/100)
        df_wins[main_outcome] = df_wins[main_outcome].clip(lower=lower, upper=upper)

        formula = f'{main_outcome} ~ B + {baseline_control}'
        if strat_formula:
            formula += f' + {strat_formula}'

        result = run_ols(df_wins, formula)
        if result:
            treat_effect = extract_treatment_effect(result, 'B')
            if treat_effect:
                spec_counter += 1
                results.append({
                    'paper_id': PAPER_ID,
                    'journal': JOURNAL,
                    'paper_title': PAPER_TITLE,
                    'spec_id': f'robust/sample/winsorize_{pct}pct',
                    'spec_tree_path': 'robustness/sample_restrictions.md',
                    'outcome_var': main_outcome,
                    'treatment_var': 'B (Bonus)',
                    'coefficient': treat_effect['coefficient'],
                    'std_error': treat_effect['std_error'],
                    't_stat': treat_effect['t_stat'],
                    'p_value': treat_effect['p_value'],
                    'ci_lower': treat_effect['ci_lower'],
                    'ci_upper': treat_effect['ci_upper'],
                    'n_obs': result['n_obs'],
                    'r_squared': result['r_squared'],
                    'coefficient_vector_json': create_coef_vector_json(result, 'B', strat_controls),
                    'sample_desc': f'Winsorized at {pct}%/{100-pct}%',
                    'fixed_effects': 'None',
                    'controls_desc': 'Baseline usage + stratification',
                    'cluster_var': 'UserID (robust SE)',
                    'model_type': 'OLS',
                    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
                })

    # Trim extreme values
    for pct in [1, 5]:
        lower = df[main_outcome].quantile(pct/100)
        upper = df[main_outcome].quantile(1 - pct/100)
        df_trim = df[(df[main_outcome] >= lower) & (df[main_outcome] <= upper)]

        formula = f'{main_outcome} ~ B + {baseline_control}'
        if strat_formula:
            formula += f' + {strat_formula}'

        result = run_ols(df_trim, formula)
        if result:
            treat_effect = extract_treatment_effect(result, 'B')
            if treat_effect:
                spec_counter += 1
                results.append({
                    'paper_id': PAPER_ID,
                    'journal': JOURNAL,
                    'paper_title': PAPER_TITLE,
                    'spec_id': f'robust/sample/trim_{pct}pct',
                    'spec_tree_path': 'robustness/sample_restrictions.md',
                    'outcome_var': main_outcome,
                    'treatment_var': 'B (Bonus)',
                    'coefficient': treat_effect['coefficient'],
                    'std_error': treat_effect['std_error'],
                    't_stat': treat_effect['t_stat'],
                    'p_value': treat_effect['p_value'],
                    'ci_lower': treat_effect['ci_lower'],
                    'ci_upper': treat_effect['ci_upper'],
                    'n_obs': result['n_obs'],
                    'r_squared': result['r_squared'],
                    'coefficient_vector_json': create_coef_vector_json(result, 'B', strat_controls),
                    'sample_desc': f'Trimmed at {pct}%/{100-pct}%',
                    'fixed_effects': 'None',
                    'controls_desc': 'Baseline usage + stratification',
                    'cluster_var': 'UserID (robust SE)',
                    'model_type': 'OLS',
                    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
                })

    # =========================================================================
    # FUNCTIONAL FORM VARIATIONS
    # =========================================================================

    print("Running functional form variations...")

    # Log transformation (add small constant to handle zeros)
    df['log_outcome'] = np.log(df[main_outcome] + 1)
    df['log_baseline'] = np.log(df[baseline_control] + 1)

    formula = f'log_outcome ~ B + log_baseline'
    if strat_formula:
        formula += f' + {strat_formula}'

    result = run_ols(df, formula)
    if result:
        treat_effect = extract_treatment_effect(result, 'B')
        if treat_effect:
            spec_counter += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': 'robust/funcform/log',
                'spec_tree_path': 'robustness/functional_form.md',
                'outcome_var': f'log({main_outcome}+1)',
                'treatment_var': 'B (Bonus)',
                'coefficient': treat_effect['coefficient'],
                'std_error': treat_effect['std_error'],
                't_stat': treat_effect['t_stat'],
                'p_value': treat_effect['p_value'],
                'ci_lower': treat_effect['ci_lower'],
                'ci_upper': treat_effect['ci_upper'],
                'n_obs': result['n_obs'],
                'r_squared': result['r_squared'],
                'coefficient_vector_json': create_coef_vector_json(result, 'B', strat_controls),
                'sample_desc': 'Full sample',
                'fixed_effects': 'None',
                'controls_desc': 'Log baseline usage + stratification',
                'cluster_var': 'UserID (robust SE)',
                'model_type': 'OLS (log-log)',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })

    # Inverse hyperbolic sine transformation
    df['ihs_outcome'] = np.arcsinh(df[main_outcome])
    df['ihs_baseline'] = np.arcsinh(df[baseline_control])

    formula = f'ihs_outcome ~ B + ihs_baseline'
    if strat_formula:
        formula += f' + {strat_formula}'

    result = run_ols(df, formula)
    if result:
        treat_effect = extract_treatment_effect(result, 'B')
        if treat_effect:
            spec_counter += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': 'robust/funcform/ihs',
                'spec_tree_path': 'robustness/functional_form.md',
                'outcome_var': f'arcsinh({main_outcome})',
                'treatment_var': 'B (Bonus)',
                'coefficient': treat_effect['coefficient'],
                'std_error': treat_effect['std_error'],
                't_stat': treat_effect['t_stat'],
                'p_value': treat_effect['p_value'],
                'ci_lower': treat_effect['ci_lower'],
                'ci_upper': treat_effect['ci_upper'],
                'n_obs': result['n_obs'],
                'r_squared': result['r_squared'],
                'coefficient_vector_json': create_coef_vector_json(result, 'B', strat_controls),
                'sample_desc': 'Full sample',
                'fixed_effects': 'None',
                'controls_desc': 'IHS baseline usage + stratification',
                'cluster_var': 'UserID (robust SE)',
                'model_type': 'OLS (IHS)',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })

    # Quadratic baseline control
    df['baseline_sq'] = df[baseline_control] ** 2
    formula = f'{main_outcome} ~ B + {baseline_control} + baseline_sq'
    if strat_formula:
        formula += f' + {strat_formula}'

    result = run_ols(df, formula)
    if result:
        treat_effect = extract_treatment_effect(result, 'B')
        if treat_effect:
            spec_counter += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': 'robust/funcform/quadratic_baseline',
                'spec_tree_path': 'robustness/functional_form.md',
                'outcome_var': main_outcome,
                'treatment_var': 'B (Bonus)',
                'coefficient': treat_effect['coefficient'],
                'std_error': treat_effect['std_error'],
                't_stat': treat_effect['t_stat'],
                'p_value': treat_effect['p_value'],
                'ci_lower': treat_effect['ci_lower'],
                'ci_upper': treat_effect['ci_upper'],
                'n_obs': result['n_obs'],
                'r_squared': result['r_squared'],
                'coefficient_vector_json': create_coef_vector_json(result, 'B', strat_controls),
                'sample_desc': 'Full sample',
                'fixed_effects': 'None',
                'controls_desc': 'Quadratic baseline usage + stratification',
                'cluster_var': 'UserID (robust SE)',
                'model_type': 'OLS',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })

    # =========================================================================
    # INFERENCE VARIATIONS
    # =========================================================================

    print("Running inference variations...")

    # Classical SE (no robust)
    formula = f'{main_outcome} ~ B + {baseline_control}'
    if strat_formula:
        formula += f' + {strat_formula}'

    try:
        model = smf.ols(formula, data=df.dropna(subset=[main_outcome, 'B', baseline_control])).fit()
        if model:
            treat_effect = {
                'coefficient': model.params['B'],
                'std_error': model.bse['B'],
                't_stat': model.tvalues['B'],
                'p_value': model.pvalues['B'],
                'ci_lower': model.conf_int().loc['B', 0],
                'ci_upper': model.conf_int().loc['B', 1]
            }
            spec_counter += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': 'robust/inference/classical_se',
                'spec_tree_path': 'robustness/inference_alternatives.md',
                'outcome_var': main_outcome,
                'treatment_var': 'B (Bonus)',
                'coefficient': treat_effect['coefficient'],
                'std_error': treat_effect['std_error'],
                't_stat': treat_effect['t_stat'],
                'p_value': treat_effect['p_value'],
                'ci_lower': treat_effect['ci_lower'],
                'ci_upper': treat_effect['ci_upper'],
                'n_obs': int(model.nobs),
                'r_squared': model.rsquared,
                'coefficient_vector_json': None,
                'sample_desc': 'Full sample',
                'fixed_effects': 'None',
                'controls_desc': 'Baseline usage + stratification',
                'cluster_var': 'None (classical SE)',
                'model_type': 'OLS',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })
    except:
        pass

    # HC2 robust SE
    try:
        model = smf.ols(formula, data=df.dropna(subset=[main_outcome, 'B', baseline_control])).fit(cov_type='HC2')
        if model:
            treat_effect = {
                'coefficient': model.params['B'],
                'std_error': model.bse['B'],
                't_stat': model.tvalues['B'],
                'p_value': model.pvalues['B'],
                'ci_lower': model.conf_int().loc['B', 0],
                'ci_upper': model.conf_int().loc['B', 1]
            }
            spec_counter += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': 'robust/inference/hc2_se',
                'spec_tree_path': 'robustness/inference_alternatives.md',
                'outcome_var': main_outcome,
                'treatment_var': 'B (Bonus)',
                'coefficient': treat_effect['coefficient'],
                'std_error': treat_effect['std_error'],
                't_stat': treat_effect['t_stat'],
                'p_value': treat_effect['p_value'],
                'ci_lower': treat_effect['ci_lower'],
                'ci_upper': treat_effect['ci_upper'],
                'n_obs': int(model.nobs),
                'r_squared': model.rsquared,
                'coefficient_vector_json': None,
                'sample_desc': 'Full sample',
                'fixed_effects': 'None',
                'controls_desc': 'Baseline usage + stratification',
                'cluster_var': 'HC2 robust SE',
                'model_type': 'OLS',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })
    except:
        pass

    # HC3 robust SE
    try:
        model = smf.ols(formula, data=df.dropna(subset=[main_outcome, 'B', baseline_control])).fit(cov_type='HC3')
        if model:
            treat_effect = {
                'coefficient': model.params['B'],
                'std_error': model.bse['B'],
                't_stat': model.tvalues['B'],
                'p_value': model.pvalues['B'],
                'ci_lower': model.conf_int().loc['B', 0],
                'ci_upper': model.conf_int().loc['B', 1]
            }
            spec_counter += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': 'robust/inference/hc3_se',
                'spec_tree_path': 'robustness/inference_alternatives.md',
                'outcome_var': main_outcome,
                'treatment_var': 'B (Bonus)',
                'coefficient': treat_effect['coefficient'],
                'std_error': treat_effect['std_error'],
                't_stat': treat_effect['t_stat'],
                'p_value': treat_effect['p_value'],
                'ci_lower': treat_effect['ci_lower'],
                'ci_upper': treat_effect['ci_upper'],
                'n_obs': int(model.nobs),
                'r_squared': model.rsquared,
                'coefficient_vector_json': None,
                'sample_desc': 'Full sample',
                'fixed_effects': 'None',
                'controls_desc': 'Baseline usage + stratification',
                'cluster_var': 'HC3 robust SE',
                'model_type': 'OLS',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })
    except:
        pass

    # =========================================================================
    # HETEROGENEITY ANALYSIS
    # =========================================================================

    print("Running heterogeneity analyses...")

    het_vars = [
        ('Female', 'gender'),
        ('High_Age', 'age'),
        ('High_Usage', 'baseline_usage'),
        ('High_Education', 'education'),
        ('High_Addiction', 'addiction'),
        ('High_Restriction', 'restriction')
    ]

    for het_var, het_name in het_vars:
        if het_var in df.columns and df[het_var].notna().sum() > 100:
            # Interaction specification
            formula = f'{main_outcome} ~ B * {het_var} + {baseline_control}'
            if strat_formula:
                formula += f' + {strat_formula}'

            result = run_ols(df, formula)
            if result:
                # Main effect
                treat_effect = extract_treatment_effect(result, 'B')
                if treat_effect:
                    spec_counter += 1
                    results.append({
                        'paper_id': PAPER_ID,
                        'journal': JOURNAL,
                        'paper_title': PAPER_TITLE,
                        'spec_id': f'robust/heterogeneity/{het_name}_main',
                        'spec_tree_path': 'robustness/heterogeneity.md',
                        'outcome_var': main_outcome,
                        'treatment_var': f'B (Bonus) x {het_name}',
                        'coefficient': treat_effect['coefficient'],
                        'std_error': treat_effect['std_error'],
                        't_stat': treat_effect['t_stat'],
                        'p_value': treat_effect['p_value'],
                        'ci_lower': treat_effect['ci_lower'],
                        'ci_upper': treat_effect['ci_upper'],
                        'n_obs': result['n_obs'],
                        'r_squared': result['r_squared'],
                        'coefficient_vector_json': create_coef_vector_json(result, 'B', strat_controls + [het_var]),
                        'sample_desc': 'Full sample',
                        'fixed_effects': 'None',
                        'controls_desc': f'Baseline usage + stratification + {het_name} interaction',
                        'cluster_var': 'UserID (robust SE)',
                        'model_type': 'OLS',
                        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
                    })

                # Interaction effect
                interact_var = f'B:{het_var}'
                if interact_var in result['params'].index:
                    spec_counter += 1
                    results.append({
                        'paper_id': PAPER_ID,
                        'journal': JOURNAL,
                        'paper_title': PAPER_TITLE,
                        'spec_id': f'robust/heterogeneity/{het_name}_interaction',
                        'spec_tree_path': 'robustness/heterogeneity.md',
                        'outcome_var': main_outcome,
                        'treatment_var': f'B x {het_name} (interaction)',
                        'coefficient': result['params'][interact_var],
                        'std_error': result['bse'][interact_var],
                        't_stat': result['params'][interact_var] / result['bse'][interact_var],
                        'p_value': result['pvalues'][interact_var],
                        'ci_lower': result['conf_int'].loc[interact_var, 0],
                        'ci_upper': result['conf_int'].loc[interact_var, 1],
                        'n_obs': result['n_obs'],
                        'r_squared': result['r_squared'],
                        'coefficient_vector_json': None,
                        'sample_desc': 'Full sample',
                        'fixed_effects': 'None',
                        'controls_desc': f'Baseline usage + stratification + {het_name} interaction',
                        'cluster_var': 'UserID (robust SE)',
                        'model_type': 'OLS',
                        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
                    })

    # =========================================================================
    # LIMIT TREATMENT SPECIFICATIONS (repeat key specs with L instead of B)
    # =========================================================================

    print("Running limit treatment specifications...")

    # Limit - no controls
    formula = f'{main_outcome} ~ L'
    result = run_ols(df, formula)
    if result:
        treat_effect = extract_treatment_effect(result, 'L')
        if treat_effect:
            spec_counter += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': 'robust/treatment/limit_nocontrols',
                'spec_tree_path': 'robustness/control_progression.md',
                'outcome_var': main_outcome,
                'treatment_var': 'L (Limit)',
                'coefficient': treat_effect['coefficient'],
                'std_error': treat_effect['std_error'],
                't_stat': treat_effect['t_stat'],
                'p_value': treat_effect['p_value'],
                'ci_lower': treat_effect['ci_lower'],
                'ci_upper': treat_effect['ci_upper'],
                'n_obs': result['n_obs'],
                'r_squared': result['r_squared'],
                'coefficient_vector_json': create_coef_vector_json(result, 'L', []),
                'sample_desc': 'Full sample',
                'fixed_effects': 'None',
                'controls_desc': 'No controls',
                'cluster_var': 'UserID (robust SE)',
                'model_type': 'OLS',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })

    # Limit with full controls
    formula = f'{main_outcome} ~ L + {baseline_control}'
    if demo_formula:
        formula += f' + {demo_formula}'
    if strat_formula:
        formula += f' + {strat_formula}'

    result = run_ols(df, formula)
    if result:
        treat_effect = extract_treatment_effect(result, 'L')
        if treat_effect:
            spec_counter += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': 'robust/treatment/limit_fullcontrols',
                'spec_tree_path': 'robustness/control_progression.md',
                'outcome_var': main_outcome,
                'treatment_var': 'L (Limit)',
                'coefficient': treat_effect['coefficient'],
                'std_error': treat_effect['std_error'],
                't_stat': treat_effect['t_stat'],
                'p_value': treat_effect['p_value'],
                'ci_lower': treat_effect['ci_lower'],
                'ci_upper': treat_effect['ci_upper'],
                'n_obs': result['n_obs'],
                'r_squared': result['r_squared'],
                'coefficient_vector_json': create_coef_vector_json(result, 'L', demo_controls + strat_controls),
                'sample_desc': 'Full sample',
                'fixed_effects': 'None',
                'controls_desc': 'Baseline + demographics + stratification',
                'cluster_var': 'UserID (robust SE)',
                'model_type': 'OLS',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })

    # Both treatments together
    formula = f'{main_outcome} ~ B + L + {baseline_control}'
    if strat_formula:
        formula += f' + {strat_formula}'

    result = run_ols(df, formula)
    if result:
        for treat_var, treat_name in [('B', 'Bonus'), ('L', 'Limit')]:
            treat_effect = extract_treatment_effect(result, treat_var)
            if treat_effect:
                spec_counter += 1
                results.append({
                    'paper_id': PAPER_ID,
                    'journal': JOURNAL,
                    'paper_title': PAPER_TITLE,
                    'spec_id': f'robust/treatment/both_{treat_name.lower()}',
                    'spec_tree_path': 'robustness/model_specification.md',
                    'outcome_var': main_outcome,
                    'treatment_var': f'{treat_var} ({treat_name})',
                    'coefficient': treat_effect['coefficient'],
                    'std_error': treat_effect['std_error'],
                    't_stat': treat_effect['t_stat'],
                    'p_value': treat_effect['p_value'],
                    'ci_lower': treat_effect['ci_lower'],
                    'ci_upper': treat_effect['ci_upper'],
                    'n_obs': result['n_obs'],
                    'r_squared': result['r_squared'],
                    'coefficient_vector_json': create_coef_vector_json(result, treat_var, strat_controls),
                    'sample_desc': 'Full sample',
                    'fixed_effects': 'None',
                    'controls_desc': 'Both treatments + baseline + stratification',
                    'cluster_var': 'UserID (robust SE)',
                    'model_type': 'OLS',
                    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
                })

    # =========================================================================
    # PLACEBO TESTS
    # =========================================================================

    print("Running placebo tests...")

    # Placebo: Test effect on baseline outcome (should be zero)
    formula = f'{baseline_control} ~ B'
    if strat_formula:
        formula += f' + {strat_formula}'

    result = run_ols(df, formula)
    if result:
        treat_effect = extract_treatment_effect(result, 'B')
        if treat_effect:
            spec_counter += 1
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': 'robust/placebo/baseline_outcome',
                'spec_tree_path': 'robustness/placebo_tests.md',
                'outcome_var': baseline_control,
                'treatment_var': 'B (Bonus)',
                'coefficient': treat_effect['coefficient'],
                'std_error': treat_effect['std_error'],
                't_stat': treat_effect['t_stat'],
                'p_value': treat_effect['p_value'],
                'ci_lower': treat_effect['ci_lower'],
                'ci_upper': treat_effect['ci_upper'],
                'n_obs': result['n_obs'],
                'r_squared': result['r_squared'],
                'coefficient_vector_json': create_coef_vector_json(result, 'B', strat_controls),
                'sample_desc': 'Full sample (PLACEBO)',
                'fixed_effects': 'None',
                'controls_desc': 'Stratification only',
                'cluster_var': 'UserID (robust SE)',
                'model_type': 'OLS (Placebo)',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })

    # Placebo: Test effect on pre-treatment characteristics
    for placebo_var, placebo_name in [('S1_Education', 'education'), ('S1_Income', 'income')]:
        if placebo_var in df.columns:
            formula = f'{placebo_var} ~ B'
            result = run_ols(df, formula)
            if result:
                treat_effect = extract_treatment_effect(result, 'B')
                if treat_effect:
                    spec_counter += 1
                    results.append({
                        'paper_id': PAPER_ID,
                        'journal': JOURNAL,
                        'paper_title': PAPER_TITLE,
                        'spec_id': f'robust/placebo/{placebo_name}',
                        'spec_tree_path': 'robustness/placebo_tests.md',
                        'outcome_var': placebo_var,
                        'treatment_var': 'B (Bonus)',
                        'coefficient': treat_effect['coefficient'],
                        'std_error': treat_effect['std_error'],
                        't_stat': treat_effect['t_stat'],
                        'p_value': treat_effect['p_value'],
                        'ci_lower': treat_effect['ci_lower'],
                        'ci_upper': treat_effect['ci_upper'],
                        'n_obs': result['n_obs'],
                        'r_squared': result['r_squared'],
                        'coefficient_vector_json': create_coef_vector_json(result, 'B', []),
                        'sample_desc': 'Full sample (PLACEBO)',
                        'fixed_effects': 'None',
                        'controls_desc': 'None',
                        'cluster_var': 'UserID (robust SE)',
                        'model_type': 'OLS (Placebo)',
                        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
                    })

    print(f"\nTotal specifications run: {spec_counter}")
    return results


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SPECIFICATION SEARCH: DIGITAL ADDICTION (163822-V2)")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    df = load_data()
    print(f"Loaded {len(df)} observations with {len(df.columns)} variables")

    # Run specifications
    print("\nRunning specifications...")
    results = run_all_specifications(df)

    # Save results
    print("\nSaving results...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{OUTPUT_DIR}/specification_results.csv', index=False)

    print(f"\nResults saved to {OUTPUT_DIR}/specification_results.csv")
    print(f"Total specifications: {len(results_df)}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    # Filter for main treatment (Bonus)
    bonus_results = results_df[results_df['treatment_var'].str.contains('Bonus', na=False)]

    if len(bonus_results) > 0:
        print(f"\nBonus Treatment Specifications: {len(bonus_results)}")
        print(f"  Positive coefficients: {(bonus_results['coefficient'] > 0).sum()} ({100*(bonus_results['coefficient'] > 0).mean():.1f}%)")
        print(f"  Significant at 5%: {(bonus_results['p_value'] < 0.05).sum()} ({100*(bonus_results['p_value'] < 0.05).mean():.1f}%)")
        print(f"  Significant at 1%: {(bonus_results['p_value'] < 0.01).sum()} ({100*(bonus_results['p_value'] < 0.01).mean():.1f}%)")
        print(f"  Median coefficient: {bonus_results['coefficient'].median():.3f}")
        print(f"  Mean coefficient: {bonus_results['coefficient'].mean():.3f}")
        print(f"  Range: [{bonus_results['coefficient'].min():.3f}, {bonus_results['coefficient'].max():.3f}]")

    # Filter for Limit treatment
    limit_results = results_df[results_df['treatment_var'].str.contains('Limit', na=False)]

    if len(limit_results) > 0:
        print(f"\nLimit Treatment Specifications: {len(limit_results)}")
        print(f"  Negative coefficients: {(limit_results['coefficient'] < 0).sum()} ({100*(limit_results['coefficient'] < 0).mean():.1f}%)")
        print(f"  Significant at 5%: {(limit_results['p_value'] < 0.05).sum()} ({100*(limit_results['p_value'] < 0.05).mean():.1f}%)")

    print("\n" + "=" * 60)
    print("SPECIFICATION SEARCH COMPLETE")
    print("=" * 60)
