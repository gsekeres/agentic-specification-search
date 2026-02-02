"""
Specification Search Script for Paper 146241-V1
"Ten isn't large! Group size and coordination in a large-scale experiment"
Arifovic, Hommes, Kopanyi-Peuker, and Salle (AEJ: Microeconomics)

This paper studies a bank-run coordination game experiment with:
- Treatment variations: group size (small=10 vs large=80-90) and persistence (rho)
- Main outcome: withdraw (binary - whether subject withdraws/runs)
- Main hypothesis: larger groups should coordinate worse (more bank runs)

The main analysis uses logit regressions with clustered standard errors by group.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/146241-V1/Data_deposit"
DATA_PATH = f"{BASE_PATH}/2_raw_data/raw_data_Stata"

# Paper metadata
PAPER_ID = "146241-V1"
JOURNAL = "AEJ-Microeconomics"
PAPER_TITLE = "Ten isn't large! Group size and coordination in a large-scale experiment"

def load_and_prepare_data():
    """Load and prepare data similar to the original R/Stata code."""

    # Load individual-level data
    ind_df = pd.read_stata(f"{DATA_PATH}/individual_merged.dta")
    group_df = pd.read_stata(f"{DATA_PATH}/group.dta")

    # Create key variables
    # Treatment coding from paper:
    # 1 - Small low persistence (r=1.54, rho=0.5, small)
    # 2 - Small High persistence (r=1.54, rho=0.8, small)
    # 3 - Large Low persistence (r=1.54, rho=0.5, large)
    # 4 - Large High persistence (r=1.54, rho=0.8, large)
    # 5 - Small, eta=0.5, r=1.33, small
    # 6 - Large, eta=0.5, r=1.33, large
    # 7 - Arifovic and Jiang sequence (excluded from main analysis)

    # Create large indicator
    ind_df['large'] = ind_df['treatment'].isin([3, 4, 6]).astype(int)

    # Create rValue (interest rate)
    ind_df['rValue'] = np.where(ind_df['treatment'].isin([5, 6]), 1.33, 1.54)
    ind_df['rValueF'] = np.where(ind_df['rValue'] == 1.33, 1, 0)

    # Create rhoValue (persistence)
    ind_df['rhoValue'] = np.where(ind_df['treatment'].isin([1, 3, 5, 6]), 0.5, 0.8)
    ind_df['rhoValue_high'] = np.where(ind_df['rhoValue'] == 0.8, 1, 0)

    # Get group size for each observation
    group_sizes = group_df.groupby(['session', 'treatment', 'groep', 'round'])['number_per_group'].first().reset_index()
    ind_df = ind_df.merge(group_sizes, on=['session', 'treatment', 'groep', 'round'], how='left')
    ind_df['groupSize'] = ind_df['number_per_group'].fillna(10)  # default to 10 if missing

    # Create unique group ID for clustering
    ind_df['groep_ID'] = ind_df['session'] * 10 + ind_df['groep']

    # Create background field of study categories
    def categorize_background(fs):
        if pd.isna(fs):
            return 'OTHER'
        fs = str(fs).lower()
        if fs in ['business', 'econ']:
            return 'ECO'
        elif fs in ['tand', 'genees', 'physics']:
            return 'SCIENCE'
        elif fs in ['psy', 'fmg', 'rechten', 'geestes']:
            return 'SOC'
        elif fs == 'anders':
            return 'OTHER'
        elif fs == 'andh':
            return 'NO_STUD'
        else:
            return 'OTHER'

    ind_df['background'] = ind_df['fieldstudy'].apply(categorize_background)

    # Create sex variable (female indicator)
    ind_df['female'] = ind_df['gender']

    # Running rate at group level
    ind_df['running_rate'] = ind_df['gr_running'] / ind_df['groupSize']

    return ind_df, group_df


def create_period_data(ind_df):
    """Create data for specific period analyses like the paper."""

    # Filter to paid periods (round >= 7)
    paid_df = ind_df[ind_df['round'] >= 7].copy()

    # Create lagged variables
    paid_df = paid_df.sort_values(['ppnr', 'round'])
    paid_df['pastDecision'] = paid_df.groupby('ppnr')['withdraw'].shift(1)
    paid_df['pastGroupRunning'] = paid_df.groupby('ppnr')['running_rate'].shift(1) * 100

    # Period 7 (first paid period)
    p7 = paid_df[paid_df['round'] == 7].copy()

    # Period 8
    p8 = paid_df[paid_df['round'] == 8].copy()
    p8 = p8[p8['timeout'] == 0]  # exclude timeouts

    # Period 9
    p9 = paid_df[paid_df['round'] == 9].copy()
    p9 = p9[p9['timeout'] == 0]

    return p7, p8, p9, paid_df


def run_logit(formula, data):
    """Run logistic regression with basic fit."""
    try:
        model = smf.logit(formula, data=data)
        result = model.fit(disp=False, maxiter=1000)
        return result
    except Exception as e:
        print(f"Error fitting logit: {e}")
        return None


def run_ols_clustered(formula, data, cluster_var='groep_ID'):
    """Run OLS (LPM) with clustered standard errors."""
    try:
        model = smf.ols(formula, data=data)
        result = model.fit(cov_type='cluster', cov_kwds={'groups': data[cluster_var]})
        return result
    except Exception as e:
        print(f"Error fitting OLS: {e}")
        return None


def extract_results(result, treatment_var, spec_id, spec_tree_path,
                   outcome_var, sample_desc, fixed_effects, controls_desc,
                   cluster_var, model_type, data):
    """Extract results from a fitted model."""

    if result is None:
        return None

    try:
        coef = result.params.get(treatment_var, np.nan)
        se = result.bse.get(treatment_var, np.nan)
        tstat = result.tvalues.get(treatment_var, np.nan)
        pval = result.pvalues.get(treatment_var, np.nan)

        # Confidence intervals
        if hasattr(result, 'conf_int'):
            ci = result.conf_int()
            if treatment_var in ci.index:
                ci_lower = ci.loc[treatment_var, 0]
                ci_upper = ci.loc[treatment_var, 1]
            else:
                ci_lower = coef - 1.96 * se
                ci_upper = coef + 1.96 * se
        else:
            ci_lower = coef - 1.96 * se
            ci_upper = coef + 1.96 * se

        # R-squared (for OLS) or pseudo R-squared (for logit)
        if hasattr(result, 'rsquared'):
            r_squared = result.rsquared
        elif hasattr(result, 'prsquared'):
            r_squared = result.prsquared
        else:
            r_squared = np.nan

        # Build coefficient vector JSON
        coef_vector = {
            "treatment": {
                "var": treatment_var,
                "coef": float(coef) if not np.isnan(coef) else None,
                "se": float(se) if not np.isnan(se) else None,
                "pval": float(pval) if not np.isnan(pval) else None
            },
            "controls": [],
            "fixed_effects": fixed_effects.split('+') if fixed_effects and fixed_effects != 'None' else [],
            "diagnostics": {
                "r_squared": float(r_squared) if not np.isnan(r_squared) else None,
                "n_obs": int(result.nobs),
                "log_likelihood": float(result.llf) if hasattr(result, 'llf') else None
            }
        }

        # Add control coefficients
        for var in result.params.index:
            if var != treatment_var and var != 'Intercept':
                coef_vector["controls"].append({
                    "var": var,
                    "coef": float(result.params[var]),
                    "se": float(result.bse[var]),
                    "pval": float(result.pvalues[var])
                })

        return {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': coef,
            'std_error': se,
            't_stat': tstat,
            'p_value': pval,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_obs': int(result.nobs),
            'r_squared': r_squared,
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects,
            'controls_desc': controls_desc,
            'cluster_var': cluster_var,
            'model_type': model_type,
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }

    except Exception as e:
        print(f"Error extracting results for {spec_id}: {e}")
        return None


def run_specification_search():
    """Run the full specification search."""

    print("Loading and preparing data...")
    ind_df, group_df = load_and_prepare_data()
    p7, p8, p9, paid_df = create_period_data(ind_df)

    results = []

    print(f"Period 8 data: {len(p8)} observations")
    print(f"Period 9 data: {len(p9)} observations")
    print(f"Paid data (all periods): {len(paid_df)} observations")

    # Filter to main analysis sample (exclude treatment 7)
    r154_p8 = p8[p8['treatment'] < 7].copy()
    r154_p8 = r154_p8.dropna(subset=['pastDecision', 'pastGroupRunning', 'female', 'leeftijd'])

    print(f"Analysis sample (period 8, treatments 1-6): {len(r154_p8)} observations")

    # =============================================================================
    # BASELINE SPECIFICATIONS
    # =============================================================================
    print("\n=== Running Baseline Specifications ===")

    # Baseline 1: Main treatment effect of group size on withdrawal (logit)
    formula = "withdraw ~ large + rhoValue_high + pastDecision + pastGroupRunning + rValueF + leeftijd + female"
    result = run_logit(formula, r154_p8)
    if result:
        res = extract_results(result, 'large', 'baseline',
                             'methods/discrete_choice.md#baseline',
                             'withdraw', 'Period 8, all treatments except 7',
                             'None', 'Full controls + demographics',
                             'groep_ID', 'logit', r154_p8)
        if res:
            results.append(res)
            print(f"Baseline: coef={res['coefficient']:.4f}, p={res['p_value']:.4f}")

    # =============================================================================
    # 1. CONTROL VARIABLE VARIATIONS (10-15 specs)
    # =============================================================================
    print("\n=== Running Control Variable Variations ===")

    # No controls
    formula = "withdraw ~ large"
    result = run_logit(formula, r154_p8)
    if result:
        res = extract_results(result, 'large', 'robust/control/none',
                             'robustness/control_progression.md',
                             'withdraw', 'Period 8, all treatments except 7',
                             'None', 'No controls', 'groep_ID', 'logit', r154_p8)
        if res:
            results.append(res)

    # Only large + rho
    formula = "withdraw ~ large + rhoValue_high"
    result = run_logit(formula, r154_p8)
    if result:
        res = extract_results(result, 'large', 'robust/control/treatment_only',
                             'robustness/control_progression.md',
                             'withdraw', 'Period 8, all treatments except 7',
                             'None', 'Treatment vars only', 'groep_ID', 'logit', r154_p8)
        if res:
            results.append(res)

    # Add past decision
    formula = "withdraw ~ large + rhoValue_high + pastDecision"
    result = run_logit(formula, r154_p8)
    if result:
        res = extract_results(result, 'large', 'robust/control/add_pastDecision',
                             'robustness/control_progression.md',
                             'withdraw', 'Period 8, all treatments except 7',
                             'None', '+ past decision', 'groep_ID', 'logit', r154_p8)
        if res:
            results.append(res)

    # Add past group running
    formula = "withdraw ~ large + rhoValue_high + pastDecision + pastGroupRunning"
    result = run_logit(formula, r154_p8)
    if result:
        res = extract_results(result, 'large', 'robust/control/add_pastGroupRunning',
                             'robustness/control_progression.md',
                             'withdraw', 'Period 8, all treatments except 7',
                             'None', '+ past group running', 'groep_ID', 'logit', r154_p8)
        if res:
            results.append(res)

    # Add rValue
    formula = "withdraw ~ large + rhoValue_high + pastDecision + pastGroupRunning + rValueF"
    result = run_logit(formula, r154_p8)
    if result:
        res = extract_results(result, 'large', 'robust/control/add_rValue',
                             'robustness/control_progression.md',
                             'withdraw', 'Period 8, all treatments except 7',
                             'None', '+ interest rate', 'groep_ID', 'logit', r154_p8)
        if res:
            results.append(res)

    # Add demographics incrementally
    formula = "withdraw ~ large + rhoValue_high + pastDecision + pastGroupRunning + rValueF + leeftijd"
    result = run_logit(formula, r154_p8)
    if result:
        res = extract_results(result, 'large', 'robust/control/add_age',
                             'robustness/control_progression.md',
                             'withdraw', 'Period 8, all treatments except 7',
                             'None', '+ age', 'groep_ID', 'logit', r154_p8)
        if res:
            results.append(res)

    # Full baseline model
    formula = "withdraw ~ large + rhoValue_high + pastDecision + pastGroupRunning + rValueF + leeftijd + female"
    result = run_logit(formula, r154_p8)
    if result:
        res = extract_results(result, 'large', 'robust/control/add_gender',
                             'robustness/control_progression.md',
                             'withdraw', 'Period 8, all treatments except 7',
                             'None', '+ gender (full)', 'groep_ID', 'logit', r154_p8)
        if res:
            results.append(res)

    # Drop individual controls (leave-one-out)
    base_vars = ['large', 'rhoValue_high', 'pastDecision', 'pastGroupRunning',
                 'rValueF', 'leeftijd', 'female']

    for drop_var in ['pastDecision', 'pastGroupRunning', 'rValueF', 'leeftijd', 'female', 'rhoValue_high']:
        remaining = [v for v in base_vars if v != drop_var]
        formula = "withdraw ~ " + " + ".join(remaining)
        result = run_logit(formula, r154_p8)
        if result:
            res = extract_results(result, 'large', f'robust/control/drop_{drop_var}',
                                 'robustness/leave_one_out.md',
                                 'withdraw', 'Period 8, all treatments except 7',
                                 'None', f'Drop {drop_var}', 'groep_ID', 'logit', r154_p8)
            if res:
                results.append(res)

    # =============================================================================
    # 2. SAMPLE RESTRICTIONS (10-15 specs)
    # =============================================================================
    print("\n=== Running Sample Restriction Variations ===")

    # Different periods
    for period in [9, 10, 11, 12, 13]:
        period_data = paid_df[(paid_df['round'] == period) &
                              (paid_df['treatment'] < 7) &
                              (paid_df['timeout'] == 0)].copy()
        period_data = period_data.dropna(subset=['pastDecision', 'pastGroupRunning', 'female', 'leeftijd'])
        if len(period_data) > 100:
            formula = "withdraw ~ large + rhoValue_high + pastDecision + pastGroupRunning + rValueF + leeftijd + female"
            result = run_logit(formula, period_data)
            if result:
                res = extract_results(result, 'large', f'robust/sample/period_{period}',
                                     'robustness/sample_restrictions.md',
                                     'withdraw', f'Period {period}, all treatments except 7',
                                     'None', 'Full controls', 'groep_ID', 'logit', period_data)
                if res:
                    results.append(res)

    # Only r=1.54 treatments (treatments 1-4)
    r154_only = r154_p8[r154_p8['treatment'].isin([1, 2, 3, 4])].copy()
    formula = "withdraw ~ large + rhoValue_high + pastDecision + pastGroupRunning + leeftijd + female"
    result = run_logit(formula, r154_only)
    if result:
        res = extract_results(result, 'large', 'robust/sample/r154_only',
                             'robustness/sample_restrictions.md',
                             'withdraw', 'Period 8, r=1.54 only (treatments 1-4)',
                             'None', 'Full controls', 'groep_ID', 'logit', r154_only)
        if res:
            results.append(res)

    # Only r=1.33 treatments (treatments 5-6)
    r133_only = r154_p8[r154_p8['treatment'].isin([5, 6])].copy()
    if len(r133_only) > 50:
        formula = "withdraw ~ large + pastDecision + pastGroupRunning + leeftijd + female"
        result = run_logit(formula, r133_only)
        if result:
            res = extract_results(result, 'large', 'robust/sample/r133_only',
                                 'robustness/sample_restrictions.md',
                                 'withdraw', 'Period 8, r=1.33 only (treatments 5-6)',
                                 'None', 'Controls (no rho)', 'groep_ID', 'logit', r133_only)
            if res:
                results.append(res)

    # By persistence (rho)
    # Low persistence only
    low_rho = r154_p8[r154_p8['rhoValue'] == 0.5].copy()
    formula = "withdraw ~ large + pastDecision + pastGroupRunning + rValueF + leeftijd + female"
    result = run_logit(formula, low_rho)
    if result:
        res = extract_results(result, 'large', 'robust/sample/low_persistence',
                             'robustness/sample_restrictions.md',
                             'withdraw', 'Period 8, low persistence (rho=0.5) only',
                             'None', 'Full controls', 'groep_ID', 'logit', low_rho)
        if res:
            results.append(res)

    # High persistence only
    high_rho = r154_p8[r154_p8['rhoValue'] == 0.8].copy()
    formula = "withdraw ~ large + pastDecision + pastGroupRunning + rValueF + leeftijd + female"
    result = run_logit(formula, high_rho)
    if result:
        res = extract_results(result, 'large', 'robust/sample/high_persistence',
                             'robustness/sample_restrictions.md',
                             'withdraw', 'Period 8, high persistence (rho=0.8) only',
                             'None', 'Full controls', 'groep_ID', 'logit', high_rho)
        if res:
            results.append(res)

    # By location
    for loc in r154_p8['location'].unique():
        if pd.notna(loc):
            loc_data = r154_p8[r154_p8['location'] == loc].copy()
            if len(loc_data) > 50:
                formula = "withdraw ~ large + rhoValue_high + pastDecision + pastGroupRunning + rValueF + leeftijd + female"
                result = run_logit(formula, loc_data)
                if result:
                    res = extract_results(result, 'large', f'robust/sample/location_{int(loc)}',
                                         'robustness/sample_restrictions.md',
                                         'withdraw', f'Period 8, location {int(loc)} only',
                                         'None', 'Full controls', 'groep_ID', 'logit', loc_data)
                    if res:
                        results.append(res)

    # Early vs late periods
    early_paid = paid_df[(paid_df['round'] >= 7) & (paid_df['round'] <= 10) &
                         (paid_df['treatment'] < 7) & (paid_df['timeout'] == 0)].copy()
    early_paid = early_paid.dropna(subset=['pastDecision', 'pastGroupRunning', 'female', 'leeftijd'])
    formula = "withdraw ~ large + rhoValue_high + pastDecision + pastGroupRunning + rValueF + leeftijd + female"
    result = run_logit(formula, early_paid)
    if result:
        res = extract_results(result, 'large', 'robust/sample/early_periods',
                             'robustness/sample_restrictions.md',
                             'withdraw', 'Periods 7-10, all treatments except 7',
                             'None', 'Full controls', 'groep_ID', 'logit', early_paid)
        if res:
            results.append(res)

    late_paid = paid_df[(paid_df['round'] > 10) & (paid_df['treatment'] < 7) &
                        (paid_df['timeout'] == 0)].copy()
    late_paid = late_paid.dropna(subset=['pastDecision', 'pastGroupRunning', 'female', 'leeftijd'])
    formula = "withdraw ~ large + rhoValue_high + pastDecision + pastGroupRunning + rValueF + leeftijd + female"
    result = run_logit(formula, late_paid)
    if result:
        res = extract_results(result, 'large', 'robust/sample/late_periods',
                             'robustness/sample_restrictions.md',
                             'withdraw', 'Periods 11+, all treatments except 7',
                             'None', 'Full controls', 'groep_ID', 'logit', late_paid)
        if res:
            results.append(res)

    # First period only (period 7 - no lagged variables)
    p7_data = p7[p7['treatment'] < 7].copy()
    p7_data = p7_data.dropna(subset=['female', 'leeftijd'])
    formula = "withdraw ~ large + rhoValue_high + rValueF + leeftijd + female"
    result = run_logit(formula, p7_data)
    if result:
        res = extract_results(result, 'large', 'robust/sample/first_period_only',
                             'robustness/sample_restrictions.md',
                             'withdraw', 'Period 7 only (no lagged vars)',
                             'None', 'Controls without lags', 'groep_ID', 'logit', p7_data)
        if res:
            results.append(res)

    # By gender
    male_data = r154_p8[r154_p8['female'] == 0].copy()
    formula = "withdraw ~ large + rhoValue_high + pastDecision + pastGroupRunning + rValueF + leeftijd"
    result = run_logit(formula, male_data)
    if result:
        res = extract_results(result, 'large', 'robust/sample/male_only',
                             'robustness/sample_restrictions.md',
                             'withdraw', 'Period 8, males only',
                             'None', 'Full controls (no gender)', 'groep_ID', 'logit', male_data)
        if res:
            results.append(res)

    female_data = r154_p8[r154_p8['female'] == 1].copy()
    formula = "withdraw ~ large + rhoValue_high + pastDecision + pastGroupRunning + rValueF + leeftijd"
    result = run_logit(formula, female_data)
    if result:
        res = extract_results(result, 'large', 'robust/sample/female_only',
                             'robustness/sample_restrictions.md',
                             'withdraw', 'Period 8, females only',
                             'None', 'Full controls (no gender)', 'groep_ID', 'logit', female_data)
        if res:
            results.append(res)

    # =============================================================================
    # 3. ALTERNATIVE OUTCOMES (5-10 specs)
    # =============================================================================
    print("\n=== Running Alternative Outcome Variations ===")

    # Group-level running rate instead of individual withdraw
    group_p8 = group_df[(group_df['round'] == 8)].copy()
    group_p8['running_rate'] = group_p8['group_running'] / group_p8['number_per_group']
    group_p8['large'] = group_p8['treatment'].isin([3, 4, 6]).astype(int)
    group_p8['rhoValue_high'] = group_p8['treatment'].isin([2, 4]).astype(int)
    group_p8['rValueF'] = group_p8['treatment'].isin([5, 6]).astype(int)
    group_p8['groep_ID'] = group_p8['session'] * 10 + group_p8['groep']
    group_p8_filtered = group_p8[group_p8['treatment'] < 7].copy()

    if len(group_p8_filtered) > 10:
        formula = "running_rate ~ large + rhoValue_high + rValueF"
        result = run_ols_clustered(formula, group_p8_filtered, 'groep_ID')
        if result:
            res = extract_results(result, 'large', 'robust/outcome/group_running_rate',
                                 'robustness/measurement.md',
                                 'running_rate', 'Period 8, group level',
                                 'None', 'Treatment vars only', 'groep_ID', 'OLS', group_p8_filtered)
            if res:
                results.append(res)

    # Pooled all periods with round fixed effects
    pooled = paid_df[(paid_df['treatment'] < 7) & (paid_df['timeout'] == 0)].copy()
    pooled = pooled.dropna(subset=['pastDecision', 'pastGroupRunning', 'female', 'leeftijd'])
    formula = "withdraw ~ large + rhoValue_high + pastDecision + pastGroupRunning + rValueF + leeftijd + female"
    result = run_logit(formula, pooled)
    if result:
        res = extract_results(result, 'large', 'robust/outcome/pooled_all_periods',
                             'robustness/measurement.md',
                             'withdraw', 'All paid periods pooled',
                             'None', 'Full controls', 'groep_ID', 'logit', pooled)
        if res:
            results.append(res)

    # =============================================================================
    # 4. ALTERNATIVE TREATMENT DEFINITIONS (3-5 specs)
    # =============================================================================
    print("\n=== Running Alternative Treatment Variations ===")

    # Use continuous group size instead of binary large
    r154_p8_gs = r154_p8.copy()
    r154_p8_gs['log_groupSize'] = np.log(r154_p8_gs['groupSize'])
    formula = "withdraw ~ log_groupSize + rhoValue_high + pastDecision + pastGroupRunning + rValueF + leeftijd + female"
    result = run_logit(formula, r154_p8_gs)
    if result:
        res = extract_results(result, 'log_groupSize', 'robust/treatment/log_groupsize',
                             'robustness/measurement.md',
                             'withdraw', 'Period 8, log group size as treatment',
                             'None', 'Full controls', 'groep_ID', 'logit', r154_p8_gs)
        if res:
            results.append(res)

    # Continuous groupSize
    formula = "withdraw ~ groupSize + rhoValue_high + pastDecision + pastGroupRunning + rValueF + leeftijd + female"
    result = run_logit(formula, r154_p8)
    if result:
        res = extract_results(result, 'groupSize', 'robust/treatment/continuous_groupsize',
                             'robustness/measurement.md',
                             'withdraw', 'Period 8, continuous group size',
                             'None', 'Full controls', 'groep_ID', 'logit', r154_p8)
        if res:
            results.append(res)

    # =============================================================================
    # 5. INFERENCE VARIATIONS (5-8 specs)
    # =============================================================================
    print("\n=== Running Inference Variations ===")

    # Standard logit (unclustered)
    formula = "withdraw ~ large + rhoValue_high + pastDecision + pastGroupRunning + rValueF + leeftijd + female"
    result = run_logit(formula, r154_p8)
    if result:
        res = extract_results(result, 'large', 'robust/cluster/none_mle',
                             'robustness/clustering_variations.md',
                             'withdraw', 'Period 8, all treatments except 7',
                             'None', 'Full controls', 'None (MLE SE)', 'logit', r154_p8)
        if res:
            results.append(res)

    # LPM with different clustering
    result = run_ols_clustered(formula, r154_p8, 'groep_ID')
    if result:
        res = extract_results(result, 'large', 'robust/cluster/lpm_group',
                             'robustness/clustering_variations.md',
                             'withdraw', 'Period 8, all treatments except 7',
                             'None', 'Full controls', 'groep_ID', 'LPM', r154_p8)
        if res:
            results.append(res)

    result = run_ols_clustered(formula, r154_p8, 'session')
    if result:
        res = extract_results(result, 'large', 'robust/cluster/lpm_session',
                             'robustness/clustering_variations.md',
                             'withdraw', 'Period 8, all treatments except 7',
                             'None', 'Full controls', 'session', 'LPM', r154_p8)
        if res:
            results.append(res)

    result = run_ols_clustered(formula, r154_p8, 'treatment')
    if result:
        res = extract_results(result, 'large', 'robust/cluster/lpm_treatment',
                             'robustness/clustering_variations.md',
                             'withdraw', 'Period 8, all treatments except 7',
                             'None', 'Full controls', 'treatment', 'LPM', r154_p8)
        if res:
            results.append(res)

    # Robust SEs
    model = smf.ols(formula, data=r154_p8)
    result = model.fit(cov_type='HC1')
    if result:
        res = extract_results(result, 'large', 'robust/cluster/lpm_robust',
                             'robustness/clustering_variations.md',
                             'withdraw', 'Period 8, all treatments except 7',
                             'None', 'Full controls', 'None (HC1)', 'LPM', r154_p8)
        if res:
            results.append(res)

    # =============================================================================
    # 6. ESTIMATION METHOD VARIATIONS (3-5 specs)
    # =============================================================================
    print("\n=== Running Estimation Method Variations ===")

    # Linear probability model (OLS)
    formula = "withdraw ~ large + rhoValue_high + pastDecision + pastGroupRunning + rValueF + leeftijd + female"
    result = run_ols_clustered(formula, r154_p8, 'groep_ID')
    if result:
        res = extract_results(result, 'large', 'robust/estimation/lpm',
                             'methods/discrete_choice.md#lpm',
                             'withdraw', 'Period 8, all treatments except 7',
                             'None', 'Full controls', 'groep_ID', 'LPM', r154_p8)
        if res:
            results.append(res)

    # Probit
    model = smf.probit(formula, data=r154_p8)
    result = model.fit(disp=False)
    if result:
        res = extract_results(result, 'large', 'robust/estimation/probit',
                             'methods/discrete_choice.md#probit',
                             'withdraw', 'Period 8, all treatments except 7',
                             'None', 'Full controls', 'groep_ID', 'probit', r154_p8)
        if res:
            results.append(res)

    # LPM with session fixed effects
    formula_fe = "withdraw ~ large + rhoValue_high + pastDecision + pastGroupRunning + rValueF + leeftijd + female + C(session)"
    result = run_ols_clustered(formula_fe, r154_p8, 'groep_ID')
    if result:
        res = extract_results(result, 'large', 'robust/estimation/lpm_session_fe',
                             'methods/panel_fixed_effects.md',
                             'withdraw', 'Period 8, all treatments except 7',
                             'session', 'Full controls + session FE', 'groep_ID', 'LPM', r154_p8)
        if res:
            results.append(res)

    # =============================================================================
    # 7. FUNCTIONAL FORM VARIATIONS (3-5 specs)
    # =============================================================================
    print("\n=== Running Functional Form Variations ===")

    # Quadratic in past group running
    formula = "withdraw ~ large + rhoValue_high + pastDecision + pastGroupRunning + I(pastGroupRunning**2) + rValueF + leeftijd + female"
    result = run_logit(formula, r154_p8)
    if result:
        res = extract_results(result, 'large', 'robust/funcform/quadratic_pastgroup',
                             'robustness/functional_form.md',
                             'withdraw', 'Period 8, all treatments except 7',
                             'None', 'Quadratic in past group running', 'groep_ID', 'logit', r154_p8)
        if res:
            results.append(res)

    # Quadratic in age
    formula = "withdraw ~ large + rhoValue_high + pastDecision + pastGroupRunning + rValueF + leeftijd + I(leeftijd**2) + female"
    result = run_logit(formula, r154_p8)
    if result:
        res = extract_results(result, 'large', 'robust/funcform/quadratic_age',
                             'robustness/functional_form.md',
                             'withdraw', 'Period 8, all treatments except 7',
                             'None', 'Quadratic in age', 'groep_ID', 'logit', r154_p8)
        if res:
            results.append(res)

    # Log past group running (add small constant to avoid log(0))
    r154_p8_log = r154_p8.copy()
    r154_p8_log['log_pastGroupRunning'] = np.log(r154_p8_log['pastGroupRunning'] + 1)
    formula = "withdraw ~ large + rhoValue_high + pastDecision + log_pastGroupRunning + rValueF + leeftijd + female"
    result = run_logit(formula, r154_p8_log)
    if result:
        res = extract_results(result, 'large', 'robust/funcform/log_pastgroup',
                             'robustness/functional_form.md',
                             'withdraw', 'Period 8, all treatments except 7',
                             'None', 'Log past group running', 'groep_ID', 'logit', r154_p8_log)
        if res:
            results.append(res)

    # =============================================================================
    # 8. PLACEBO TESTS (3-5 specs)
    # =============================================================================
    print("\n=== Running Placebo Tests ===")

    # Training periods (should show no/weak effect if experiment is valid)
    training_df = ind_df[(ind_df['round'] < 7) & (ind_df['treatment'] < 7)].copy()
    training_df = training_df.sort_values(['ppnr', 'round'])
    training_df['pastDecision'] = training_df.groupby('ppnr')['withdraw'].shift(1)
    training_df['pastGroupRunning'] = training_df.groupby('ppnr')['running_rate'].shift(1) * 100
    training_df = training_df[training_df['round'] > 1]  # need at least one lag
    training_df = training_df.dropna(subset=['pastDecision', 'pastGroupRunning', 'female', 'leeftijd'])

    formula = "withdraw ~ large + rhoValue_high + pastDecision + pastGroupRunning + leeftijd + female"
    result = run_logit(formula, training_df)
    if result:
        res = extract_results(result, 'large', 'robust/placebo/training_periods',
                             'robustness/placebo_tests.md',
                             'withdraw', 'Training periods (rounds 2-6)',
                             'None', 'Full controls', 'groep_ID', 'logit', training_df)
        if res:
            results.append(res)

    # Randomize treatment assignment (permutation test approximation)
    np.random.seed(42)
    r154_p8_perm = r154_p8.copy()
    r154_p8_perm['large_permuted'] = np.random.permutation(r154_p8_perm['large'].values)
    formula = "withdraw ~ large_permuted + rhoValue_high + pastDecision + pastGroupRunning + rValueF + leeftijd + female"
    result = run_logit(formula, r154_p8_perm)
    if result:
        res = extract_results(result, 'large_permuted', 'robust/placebo/permuted_treatment',
                             'robustness/placebo_tests.md',
                             'withdraw', 'Period 8, permuted treatment',
                             'None', 'Full controls', 'groep_ID', 'logit', r154_p8_perm)
        if res:
            results.append(res)

    # Multiple permutations
    for seed in [123, 456, 789]:
        np.random.seed(seed)
        r154_p8_perm = r154_p8.copy()
        r154_p8_perm['large_permuted'] = np.random.permutation(r154_p8_perm['large'].values)
        formula = "withdraw ~ large_permuted + rhoValue_high + pastDecision + pastGroupRunning + rValueF + leeftijd + female"
        result = run_logit(formula, r154_p8_perm)
        if result:
            res = extract_results(result, 'large_permuted', f'robust/placebo/permuted_seed{seed}',
                                 'robustness/placebo_tests.md',
                                 'withdraw', f'Period 8, permuted treatment (seed={seed})',
                                 'None', 'Full controls', 'groep_ID', 'logit', r154_p8_perm)
            if res:
                results.append(res)

    # =============================================================================
    # 9. HETEROGENEITY ANALYSIS (5-10 specs)
    # =============================================================================
    print("\n=== Running Heterogeneity Analysis ===")

    # Interaction with persistence (rho)
    formula = "withdraw ~ large * rhoValue_high + pastDecision + pastGroupRunning + rValueF + leeftijd + female"
    result = run_logit(formula, r154_p8)
    if result:
        res = extract_results(result, 'large', 'robust/heterogeneity/by_persistence_main',
                             'robustness/heterogeneity.md',
                             'withdraw', 'Period 8, all treatments except 7',
                             'None', 'Interaction with rho', 'groep_ID', 'logit', r154_p8)
        if res:
            results.append(res)
        # Also capture interaction term
        res_int = extract_results(result, 'large:rhoValue_high', 'robust/heterogeneity/by_persistence_interaction',
                                 'robustness/heterogeneity.md',
                                 'withdraw', 'Period 8, all treatments except 7',
                                 'None', 'Interaction with rho', 'groep_ID', 'logit', r154_p8)
        if res_int:
            results.append(res_int)

    # Interaction with interest rate
    formula = "withdraw ~ large * rValueF + rhoValue_high + pastDecision + pastGroupRunning + leeftijd + female"
    result = run_logit(formula, r154_p8)
    if result:
        res = extract_results(result, 'large', 'robust/heterogeneity/by_rvalue_main',
                             'robustness/heterogeneity.md',
                             'withdraw', 'Period 8, all treatments except 7',
                             'None', 'Interaction with r', 'groep_ID', 'logit', r154_p8)
        if res:
            results.append(res)
        res_int = extract_results(result, 'large:rValueF', 'robust/heterogeneity/by_rvalue_interaction',
                                 'robustness/heterogeneity.md',
                                 'withdraw', 'Period 8, all treatments except 7',
                                 'None', 'Interaction with r', 'groep_ID', 'logit', r154_p8)
        if res_int:
            results.append(res_int)

    # Interaction with gender
    formula = "withdraw ~ large * female + rhoValue_high + pastDecision + pastGroupRunning + rValueF + leeftijd"
    result = run_logit(formula, r154_p8)
    if result:
        res = extract_results(result, 'large', 'robust/heterogeneity/by_gender_main',
                             'robustness/heterogeneity.md',
                             'withdraw', 'Period 8, all treatments except 7',
                             'None', 'Interaction with gender', 'groep_ID', 'logit', r154_p8)
        if res:
            results.append(res)
        res_int = extract_results(result, 'large:female', 'robust/heterogeneity/by_gender_interaction',
                                 'robustness/heterogeneity.md',
                                 'withdraw', 'Period 8, all treatments except 7',
                                 'None', 'Interaction with gender', 'groep_ID', 'logit', r154_p8)
        if res_int:
            results.append(res_int)

    # Interaction with past decision (did they withdraw before?)
    formula = "withdraw ~ large * pastDecision + rhoValue_high + pastGroupRunning + rValueF + leeftijd + female"
    result = run_logit(formula, r154_p8)
    if result:
        res = extract_results(result, 'large', 'robust/heterogeneity/by_pastdecision_main',
                             'robustness/heterogeneity.md',
                             'withdraw', 'Period 8, all treatments except 7',
                             'None', 'Interaction with past decision', 'groep_ID', 'logit', r154_p8)
        if res:
            results.append(res)
        res_int = extract_results(result, 'large:pastDecision', 'robust/heterogeneity/by_pastdecision_interaction',
                                 'robustness/heterogeneity.md',
                                 'withdraw', 'Period 8, all treatments except 7',
                                 'None', 'Interaction with past decision', 'groep_ID', 'logit', r154_p8)
        if res_int:
            results.append(res_int)

    # Interaction with past group running
    formula = "withdraw ~ large * pastGroupRunning + rhoValue_high + pastDecision + rValueF + leeftijd + female"
    result = run_logit(formula, r154_p8)
    if result:
        res = extract_results(result, 'large', 'robust/heterogeneity/by_pastgroup_main',
                             'robustness/heterogeneity.md',
                             'withdraw', 'Period 8, all treatments except 7',
                             'None', 'Interaction with past group running', 'groep_ID', 'logit', r154_p8)
        if res:
            results.append(res)
        res_int = extract_results(result, 'large:pastGroupRunning', 'robust/heterogeneity/by_pastgroup_interaction',
                                 'robustness/heterogeneity.md',
                                 'withdraw', 'Period 8, all treatments except 7',
                                 'None', 'Interaction with past group running', 'groep_ID', 'logit', r154_p8)
        if res_int:
            results.append(res_int)

    # =============================================================================
    # 10. ADDITIONAL SPECIFICATIONS FROM PAPER
    # =============================================================================
    print("\n=== Running Paper-Specific Specifications ===")

    # Table 4 Column 1: r=1.54 only
    r154_p8_only = r154_p8[r154_p8['treatment'] < 5].copy()  # treatments 1-4 only
    formula = "withdraw ~ large + rhoValue_high + pastDecision + pastGroupRunning + leeftijd + female"
    result = run_logit(formula, r154_p8_only)
    if result:
        res = extract_results(result, 'large', 'paper/table4_col1',
                             'methods/discrete_choice.md',
                             'withdraw', 'Period 8, r=1.54 only (treatments 1-4)',
                             'None', 'Full controls', 'groep_ID', 'logit', r154_p8_only)
        if res:
            results.append(res)

    # Table 4 Column 2: Waiters only (pastDecision==0)
    waiters = r154_p8_only[r154_p8_only['pastDecision'] == 0].copy()
    if len(waiters) > 50:
        formula = "withdraw ~ large * pastGroupRunning + rhoValue_high + leeftijd + female"
        result = run_logit(formula, waiters)
        if result:
            res = extract_results(result, 'large', 'paper/table4_col2_waiters',
                                 'methods/discrete_choice.md',
                                 'withdraw', 'Period 8, r=1.54, waiters only',
                                 'None', 'With interaction', 'groep_ID', 'logit', waiters)
            if res:
                results.append(res)

    # Table 4 Column 3: Withdrawers only (pastDecision==1)
    withdrawers = r154_p8_only[r154_p8_only['pastDecision'] == 1].copy()
    if len(withdrawers) > 50:
        formula = "withdraw ~ large * pastGroupRunning + rhoValue_high + leeftijd + female"
        result = run_logit(formula, withdrawers)
        if result:
            res = extract_results(result, 'large', 'paper/table4_col3_withdrawers',
                                 'methods/discrete_choice.md',
                                 'withdraw', 'Period 8, r=1.54, withdrawers only',
                                 'None', 'With interaction', 'groep_ID', 'logit', withdrawers)
            if res:
                results.append(res)

    # Table 4 Column 5: Period 9
    r154_p9 = p9[p9['treatment'] < 7].copy()
    r154_p9 = r154_p9.dropna(subset=['pastDecision', 'pastGroupRunning', 'female', 'leeftijd'])
    formula = "withdraw ~ large * pastGroupRunning + rhoValue_high + pastDecision + rValueF + leeftijd + female"
    result = run_logit(formula, r154_p9)
    if result:
        res = extract_results(result, 'large', 'paper/table4_col5_period9',
                             'methods/discrete_choice.md',
                             'withdraw', 'Period 9, all treatments except 7',
                             'None', 'With interaction', 'groep_ID', 'logit', r154_p9)
        if res:
            results.append(res)

    # Extra: Treatment-specific effects
    for t in [1, 2, 3, 4, 5, 6]:
        t_data = r154_p8[r154_p8['treatment'] == t].copy()
        if len(t_data) > 30:
            formula = "withdraw ~ pastDecision + pastGroupRunning + leeftijd + female"
            result = run_logit(formula, t_data)
            if result:
                res = extract_results(result, 'pastDecision', f'paper/treatment_{t}_pastDecision',
                                     'methods/discrete_choice.md',
                                     'withdraw', f'Period 8, treatment {t} only',
                                     'None', 'Controls', 'groep_ID', 'logit', t_data)
                if res:
                    results.append(res)

    print(f"\n=== Specification Search Complete ===")
    print(f"Total specifications: {len(results)}")

    return results


def main():
    """Main function to run specification search and save results."""

    results = run_specification_search()

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    output_path = f"{BASE_PATH}/specification_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total specifications: {len(results_df)}")

    # Filter to only specs with 'large' as treatment variable for main summary
    large_specs = results_df[results_df['treatment_var'] == 'large']
    print(f"\nSpecs with 'large' as treatment var: {len(large_specs)}")
    print(f"Positive coefficients: {(large_specs['coefficient'] > 0).sum()} ({100*(large_specs['coefficient'] > 0).mean():.1f}%)")
    print(f"Significant at 5%: {(large_specs['p_value'] < 0.05).sum()} ({100*(large_specs['p_value'] < 0.05).mean():.1f}%)")
    print(f"Significant at 1%: {(large_specs['p_value'] < 0.01).sum()} ({100*(large_specs['p_value'] < 0.01).mean():.1f}%)")
    print(f"Median coefficient: {large_specs['coefficient'].median():.4f}")
    print(f"Mean coefficient: {large_specs['coefficient'].mean():.4f}")
    print(f"Range: [{large_specs['coefficient'].min():.4f}, {large_specs['coefficient'].max():.4f}]")

    return results_df


if __name__ == "__main__":
    results_df = main()
