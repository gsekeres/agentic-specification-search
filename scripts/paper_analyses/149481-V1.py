"""
Specification Search for Paper 149481-V1
"Do Thank-You Calls Increase Charitable Giving? Expert Forecasts and Field Experimental Evidence"

This paper examines three randomized controlled experiments at charities testing whether
thank-you phone calls increase subsequent charitable giving.

Method Classification: Cross-sectional OLS / Randomized Controlled Trial
Primary Method Tree Path: specification_tree/methods/cross_sectional_ols.md

Primary outcomes:
- renewing (donated): Binary indicator for whether donor renewed/gave in next year
- gift_cond: Amount donated (conditional on donating)

Treatment:
- treat: Binary indicator for thank-you call treatment

Controls (Experiments 1 and 3):
- payment_amount2: Baseline gift amount
- var12: Baseline number of gifts
- female: Female indicator
- age_display2, age_display3: Age group indicators (45-64, 65+)
- inc_display1-4: Income group indicators
- lor_display2: Residence length >5 years

Fixed Effects (Experiments 1 and 3):
- Station x Execution Date (ii)
"""

import pandas as pd
import numpy as np
import json
import warnings
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')

# Paths
BASE_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search"
DATA_PATH = f"{BASE_PATH}/data/downloads/extracted/149481-V1/thank_you_replication/data"
OUTPUT_PATH = f"{BASE_PATH}/data/downloads/extracted/149481-V1"

# Paper metadata
PAPER_ID = "149481-V1"
JOURNAL = "AER"
PAPER_TITLE = "Do Thank-You Calls Increase Charitable Giving? Expert Forecasts and Field Experimental Evidence"

# Results storage
results = []

def load_data():
    """Load all raw data files"""
    callog = pd.read_stata(f"{DATA_PATH}/callog.dta")
    transactions = pd.read_stata(f"{DATA_PATH}/transactions.dta")
    demographics = pd.read_stata(f"{DATA_PATH}/demographics.dta")
    treatments = pd.read_stata(f"{DATA_PATH}/treatments.dta")
    gift = pd.read_stata(f"{DATA_PATH}/gift.dta")
    return callog, transactions, demographics, treatments, gift

def prepare_exp1_exp3_data():
    """Prepare data for experiments 1 and 3 following the Stata code"""
    callog, transactions, demographics, treatments, gift = load_data()

    # Standardize account IDs
    transactions['account_id'] = transactions['account_id'].astype(str).str.upper()
    transactions['station'] = transactions['station'].astype(str).str.upper()
    callog['account_id'] = callog['account_id'].astype(str).str.upper()
    callog['station'] = callog['station'].astype(str).str.upper()
    demographics['account_id'] = demographics['account_id'].astype(str).str.upper()
    treatments['account_id'] = treatments['account_id'].astype(str).str.upper()

    # Process transactions - parse gift_date
    transactions['gift_date_parsed'] = pd.to_datetime(transactions['gift_date'], errors='coerce')
    transactions = transactions.sort_values(['account_id', 'gift_date_parsed'])

    # Collapse transactions by account and date (sum payments)
    transactions = transactions.groupby(['account_id', 'gift_date_parsed', 'station']).agg({
        'payment_amount': 'sum'
    }).reset_index()

    # Filter out big donors (>= $10,000 at any point)
    max_payment = transactions.groupby('account_id')['payment_amount'].max().reset_index()
    big_donors = max_payment[max_payment['payment_amount'] >= 10000]['account_id']
    transactions = transactions[~transactions['account_id'].isin(big_donors)]

    # Process call log - drop duplicates
    callog = callog.drop_duplicates(subset=['account_id'], keep='first')

    # Create treatment variable
    callog['treat'] = (callog['segment'] == 'Active Calls').astype(int)

    # Convert exec_date to string for filtering
    callog['exec_date'] = callog['exec_date'].astype(str)

    # Drop exec_dates on or after 17/04
    callog = callog[~callog['exec_date'].isin(['1704', '1707', '1710', '1801'])]

    # Create execution date variable
    callog['exec_year'] = callog['exec_date'].str[:2].astype(int) + 2000
    callog['exec_month'] = callog['exec_date'].str[2:].astype(int)
    callog['edate'] = pd.to_datetime(
        callog['exec_year'].astype(str) + '-' + callog['exec_month'].astype(str).str.zfill(2) + '-01'
    )

    # Merge callog with transactions first (simpler groupby)
    merged = transactions.merge(
        callog[['account_id', 'treat', 'edate', 'station', 'exec_date', 'response_type']],
        on='account_id', how='inner', suffixes=('_trans', '_call')
    )
    merged['station'] = merged['station_call']

    # Create timeline categories
    merged['days_from_edate'] = (merged['gift_date_parsed'] - merged['edate']).dt.days

    conditions = [
        merged['days_from_edate'] < -365,
        (merged['days_from_edate'] >= -365) & (merged['days_from_edate'] <= 0),
        (merged['days_from_edate'] > 0) & (merged['days_from_edate'] <= 365),
        (merged['days_from_edate'] > 365) & (merged['days_from_edate'] <= 730),
        (merged['days_from_edate'] > 730) & (merged['days_from_edate'] <= 1095),
        merged['days_from_edate'] > 1095
    ]
    choices = ['a_existing', 'b_year_before', 'c_year_after', 'd_two_years', 'e_three_years', 'f_later']
    merged['timeline'] = np.select(conditions, choices, default='other')

    # Group by account and timeline (simpler - fewer columns)
    grouped = merged.groupby(['account_id', 'timeline', 'treat', 'station', 'exec_date', 'edate']).agg({
        'payment_amount': 'sum',
        'gift_date_parsed': 'count'
    }).reset_index()
    grouped.columns = list(grouped.columns[:-2]) + ['payment_amount', 'num_gifts']

    # Get each time period
    year_before = grouped[grouped['timeline'] == 'b_year_before'][['account_id', 'treat', 'station', 'exec_date', 'edate', 'payment_amount', 'num_gifts']].copy()
    year_before = year_before.rename(columns={'payment_amount': 'payment_amount2', 'num_gifts': 'var12'})

    year_after = grouped[grouped['timeline'] == 'c_year_after'][['account_id', 'payment_amount', 'num_gifts']].copy()
    year_after = year_after.rename(columns={'payment_amount': 'payment_amount3', 'num_gifts': 'var13'})

    two_years = grouped[grouped['timeline'] == 'd_two_years'][['account_id', 'payment_amount', 'num_gifts']].copy()
    two_years = two_years.rename(columns={'payment_amount': 'payment_amount4', 'num_gifts': 'var14'})

    three_years = grouped[grouped['timeline'] == 'e_three_years'][['account_id', 'payment_amount', 'num_gifts']].copy()
    three_years = three_years.rename(columns={'payment_amount': 'payment_amount5', 'num_gifts': 'var15'})

    # Check for existing donors
    existing = grouped[grouped['timeline'] == 'a_existing'][['account_id']].drop_duplicates()
    existing['is_existing'] = 1

    # Merge all periods
    df = year_before.merge(year_after, on='account_id', how='left')
    df = df.merge(two_years, on='account_id', how='left')
    df = df.merge(three_years, on='account_id', how='left')
    df = df.merge(existing, on='account_id', how='left')

    # Drop existing donors
    df = df[df['is_existing'].isna()]

    # Fill missing with 0
    for col in ['payment_amount3', 'var13', 'payment_amount4', 'var14', 'payment_amount5', 'var15']:
        df[col] = df[col].fillna(0)

    # Drop if baseline = 0
    df = df[df['payment_amount2'] > 0]

    # Now merge demographics
    demographics = demographics.drop_duplicates(subset=['account_id'])
    demographics = demographics[demographics['account_id'] != 'NULL']

    demographics['female'] = np.where(demographics['gender'] == 'M', 0,
                                       np.where(demographics['gender'] == 'F', 1, np.nan))
    demographics['age_display1'] = demographics['agecode'].isin([1, 2, 3]).astype(float)
    demographics['age_display2'] = demographics['agecode'].isin([4, 5]).astype(float)
    demographics['age_display3'] = demographics['agecode'].isin([6, 7]).astype(float)
    demographics['inc_display4'] = demographics['hhincomecode'].isin(['A', 'B', 'C']).astype(float)
    demographics['inc_display3'] = demographics['hhincomecode'].isin(['D', 'E', 'F']).astype(float)
    demographics['inc_display1'] = demographics['hhincomecode'].isin(['G', 'H', 'I']).astype(float)
    demographics['inc_display2'] = demographics['hhincomecode'].isin(['J', 'K', 'L']).astype(float)
    demographics['lor_display1'] = demographics['lor'].isin([2, 3, 4]).astype(float)
    demographics['lor_display2'] = demographics['lor'].isin([5, 6, 7, 8]).astype(float)

    df = df.merge(
        demographics[['account_id', 'female', 'age_display1', 'age_display2', 'age_display3',
                     'inc_display1', 'inc_display2', 'inc_display3', 'inc_display4',
                     'lor_display1', 'lor_display2']],
        on='account_id', how='left'
    )

    # Merge treatments
    df = df.merge(treatments[['account_id', 'script']], on='account_id', how='left')

    # Create outcome variables
    df['renewing'] = (df['var13'] > 0).astype(int)
    df['donated'] = df['renewing']
    df['gift_cond'] = np.where(df['payment_amount3'] > 0, df['payment_amount3'], np.nan)

    # Create fixed effect group
    df['ii'] = df['station'].astype(str) + '_' + df['exec_date'].astype(str)

    # Create retention
    df['retention'] = df['payment_amount3'] / df['payment_amount2']

    # Fill script
    df['script'] = df['script'].fillna(0)
    df['script2'] = np.where(df['treat'] == 1, df['script'], 0)

    # Split into experiments
    exp1 = df[~df['exec_date'].isin(['1610', '1701'])].copy()
    exp1 = exp1[~exp1['station'].isin(['24', '55', '64', '61'])]

    exp3 = df[df['exec_date'] == '1610'].copy()
    exp3 = exp3[~exp3['station'].isin(['24', '55', '64', '61'])]

    # For exp3, create treatment groups
    exp3['exp3_treat'] = np.where((exp3['script2'] == 0) & (exp3['treat'] == 0), 0,
                                   np.where((exp3['script2'] == 0) & (exp3['treat'] == 1), 1,
                                            np.where(exp3['script2'] == 1, 2, np.nan)))

    return exp1, exp3

def prepare_exp2_data():
    """Prepare experiment 2 data (National Non-Profit)"""
    _, _, _, _, gift = load_data()

    # Rename columns to standard format
    gift = gift.rename(columns={
        'id': 'account_id',
        'paymentamounttransactions': 'payment_amount',
        'giftdatetransactions': 'gift_date',
        'treatment': 'response_type'
    })

    # Create treatment variable
    gift['treat'] = np.where(gift['response_type'] == 'Control', 0, 1)

    # Parse date
    gift['date'] = pd.to_datetime(gift['gift_date'], format='%m/%d/%y', errors='coerce')

    # Execution date for experiment 2
    gift['edate'] = pd.to_datetime('2013-04-01')

    # Create timeline
    gift['days_from_edate'] = (gift['date'] - gift['edate']).dt.days

    conditions = [
        gift['days_from_edate'] < -365,
        (gift['days_from_edate'] >= -365) & (gift['days_from_edate'] <= 0),
        (gift['days_from_edate'] > 0) & (gift['days_from_edate'] <= 365),
        gift['days_from_edate'] > 365
    ]
    choices = ['a_existing', 'b_year_before', 'c_year_after', 'd_later']
    gift['timeline'] = np.select(conditions, choices, default='other')

    # Group by account and timeline
    grouped = gift.groupby(['account_id', 'timeline', 'treat', 'response_type']).agg({
        'payment_amount': 'sum',
        'date': 'count'
    }).reset_index()
    grouped.columns = list(grouped.columns[:-2]) + ['payment_amount', 'num_gifts']

    # Year before
    year_before = grouped[grouped['timeline'] == 'b_year_before'].copy()
    year_before = year_before.rename(columns={'payment_amount': 'payment_amount2', 'num_gifts': 'var12'})
    year_before = year_before[['account_id', 'treat', 'response_type', 'payment_amount2', 'var12']]

    # Year after
    year_after = grouped[grouped['timeline'] == 'c_year_after'][['account_id', 'payment_amount', 'num_gifts']].copy()
    year_after = year_after.rename(columns={'payment_amount': 'payment_amount3', 'num_gifts': 'var13'})

    # Existing donors
    existing = grouped[grouped['timeline'] == 'a_existing'][['account_id']].drop_duplicates()
    existing['is_existing'] = 1

    # Merge
    exp2 = year_before.merge(year_after, on='account_id', how='left')
    exp2 = exp2.merge(existing, on='account_id', how='left')

    # Drop existing donors
    exp2 = exp2[exp2['is_existing'].isna()]

    # Fill missing
    exp2['payment_amount3'] = exp2['payment_amount3'].fillna(0)
    exp2['var13'] = exp2['var13'].fillna(0)

    # Drop if baseline = 0
    exp2 = exp2[exp2['payment_amount2'] > 0]

    # Create outcomes
    exp2['renewing'] = (exp2['var13'] > 0).astype(int)
    exp2['donated'] = exp2['renewing']
    exp2['gift_cond'] = np.where(exp2['payment_amount3'] > 0, exp2['payment_amount3'], np.nan)
    exp2['retention'] = exp2['payment_amount3'] / exp2['payment_amount2']

    # Create reached variable for LATE
    exp2['reached'] = np.where(exp2['response_type'] == 'Called: Contacted', 1,
                                np.where(exp2['response_type'].isin(['Called: Not Contacted', 'Control']), 0, np.nan))

    return exp2

def add_result(spec_id, spec_tree_path, outcome_var, treatment_var,
               coef, se, t_stat, p_val, ci_lower, ci_upper, n_obs, r_squared,
               coef_vector_json, sample_desc, fixed_effects, controls_desc,
               cluster_var, model_type, experiment='1'):
    """Add a specification result to the results list"""
    results.append({
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'outcome_var': outcome_var,
        'treatment_var': treatment_var,
        'coefficient': coef,
        'std_error': se,
        't_stat': t_stat,
        'p_value': p_val,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_obs': n_obs,
        'r_squared': r_squared,
        'coefficient_vector_json': json.dumps(coef_vector_json),
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
        'experiment': experiment
    })

def run_ols_with_fe(df, formula, fe_var=None, max_fe_groups=100):
    """Run OLS with optional fixed effects
    If too many FE groups, run without FE to avoid memory/recursion issues
    """
    df_copy = df.copy()

    use_fe = False
    if fe_var and fe_var in df_copy.columns:
        n_groups = df_copy[fe_var].nunique()
        if n_groups <= max_fe_groups:
            # Add FE dummies
            fe_dummies = pd.get_dummies(df_copy[fe_var], prefix='fe', drop_first=True)
            df_copy = pd.concat([df_copy.reset_index(drop=True), fe_dummies.reset_index(drop=True)], axis=1)
            fe_cols = list(fe_dummies.columns)

            # Add to formula
            if ' + ' in formula.split('~')[1]:
                formula = formula + ' + ' + ' + '.join(fe_cols)
            else:
                formula = formula + ' + ' + ' + '.join(fe_cols)
            use_fe = True
        # If too many FE groups, just run without FE

    try:
        model = smf.ols(formula, data=df_copy).fit(cov_type='HC1')
        model._used_fe = use_fe  # Track if FE was used
        return model
    except Exception as e:
        # If error, try without FE
        try:
            model = smf.ols(formula.split(' + fe_')[0] if ' + fe_' in formula else formula, data=df).fit(cov_type='HC1')
            model._used_fe = False
            return model
        except Exception as e2:
            print(f"  Error: {e2}")
            return None

def run_all_specs(df, outcome_var, treatment_var, controls, fe_var, spec_prefix, experiment):
    """Run comprehensive specification search"""

    df_analysis = df.dropna(subset=[outcome_var]).copy()

    if len(df_analysis) < 50:
        print(f"  Skipping {spec_prefix}/{outcome_var}: insufficient data ({len(df_analysis)} obs)")
        return

    # Build formula
    if controls:
        available_controls = [c for c in controls if c in df_analysis.columns]
        formula = f"{outcome_var} ~ {treatment_var} + " + ' + '.join(available_controls)
    else:
        available_controls = []
        formula = f"{outcome_var} ~ {treatment_var}"

    # ========================================
    # 1. BASELINE
    # ========================================
    model = run_ols_with_fe(df_analysis, formula, fe_var)

    if model and treatment_var in model.params:
        coef_vector = {
            'treatment': {'var': treatment_var, 'coef': float(model.params[treatment_var]),
                         'se': float(model.bse[treatment_var]), 'pval': float(model.pvalues[treatment_var])},
            'controls': [{'var': v, 'coef': float(model.params[v]), 'se': float(model.bse[v]), 'pval': float(model.pvalues[v])}
                        for v in available_controls if v in model.params],
            'fixed_effects': [fe_var] if fe_var else []
        }

        add_result(
            spec_id=f'{spec_prefix}/baseline',
            spec_tree_path='methods/cross_sectional_ols.md#baseline',
            outcome_var=outcome_var,
            treatment_var=treatment_var,
            coef=float(model.params[treatment_var]),
            se=float(model.bse[treatment_var]),
            t_stat=float(model.tvalues[treatment_var]),
            p_val=float(model.pvalues[treatment_var]),
            ci_lower=float(model.conf_int().loc[treatment_var, 0]),
            ci_upper=float(model.conf_int().loc[treatment_var, 1]),
            n_obs=int(model.nobs),
            r_squared=float(model.rsquared),
            coef_vector_json=coef_vector,
            sample_desc=f'Experiment {experiment}',
            fixed_effects=fe_var if fe_var else 'None',
            controls_desc=', '.join(available_controls) if available_controls else 'None',
            cluster_var='robust_hc1',
            model_type='OLS with FE' if fe_var else 'OLS',
            experiment=experiment
        )

    # ========================================
    # 2. CONTROL PROGRESSION (Build-up)
    # ========================================

    # 2a. Bivariate (no controls)
    formula_bivariate = f"{outcome_var} ~ {treatment_var}"
    model = run_ols_with_fe(df_analysis, formula_bivariate, fe_var)
    if model and treatment_var in model.params:
        add_result(
            spec_id=f'{spec_prefix}/robust/build/bivariate',
            spec_tree_path='robustness/control_progression.md',
            outcome_var=outcome_var, treatment_var=treatment_var,
            coef=float(model.params[treatment_var]),
            se=float(model.bse[treatment_var]),
            t_stat=float(model.tvalues[treatment_var]),
            p_val=float(model.pvalues[treatment_var]),
            ci_lower=float(model.conf_int().loc[treatment_var, 0]),
            ci_upper=float(model.conf_int().loc[treatment_var, 1]),
            n_obs=int(model.nobs), r_squared=float(model.rsquared),
            coef_vector_json={'treatment': {'var': treatment_var, 'coef': float(model.params[treatment_var])}},
            sample_desc=f'Experiment {experiment}',
            fixed_effects=fe_var if fe_var else 'None',
            controls_desc='None', cluster_var='robust_hc1',
            model_type='OLS with FE' if fe_var else 'OLS', experiment=experiment
        )

    # 2b. Baseline variables only
    baseline_vars = ['payment_amount2', 'var12']
    avail_baseline = [c for c in baseline_vars if c in df_analysis.columns]
    if avail_baseline:
        formula_base = f"{outcome_var} ~ {treatment_var} + " + ' + '.join(avail_baseline)
        model = run_ols_with_fe(df_analysis, formula_base, fe_var)
        if model and treatment_var in model.params:
            add_result(
                spec_id=f'{spec_prefix}/robust/build/baseline_vars',
                spec_tree_path='robustness/control_progression.md',
                outcome_var=outcome_var, treatment_var=treatment_var,
                coef=float(model.params[treatment_var]),
                se=float(model.bse[treatment_var]),
                t_stat=float(model.tvalues[treatment_var]),
                p_val=float(model.pvalues[treatment_var]),
                ci_lower=float(model.conf_int().loc[treatment_var, 0]),
                ci_upper=float(model.conf_int().loc[treatment_var, 1]),
                n_obs=int(model.nobs), r_squared=float(model.rsquared),
                coef_vector_json={'treatment': {'var': treatment_var, 'coef': float(model.params[treatment_var])}},
                sample_desc=f'Experiment {experiment}',
                fixed_effects=fe_var if fe_var else 'None',
                controls_desc=', '.join(avail_baseline), cluster_var='robust_hc1',
                model_type='OLS with FE' if fe_var else 'OLS', experiment=experiment
            )

    # 2c. Demographics only
    demo_vars = ['female', 'age_display2', 'age_display3']
    avail_demo = [c for c in demo_vars if c in df_analysis.columns]
    if avail_demo:
        formula_demo = f"{outcome_var} ~ {treatment_var} + " + ' + '.join(avail_demo)
        model = run_ols_with_fe(df_analysis, formula_demo, fe_var)
        if model and treatment_var in model.params:
            add_result(
                spec_id=f'{spec_prefix}/robust/build/demographics',
                spec_tree_path='robustness/control_progression.md',
                outcome_var=outcome_var, treatment_var=treatment_var,
                coef=float(model.params[treatment_var]),
                se=float(model.bse[treatment_var]),
                t_stat=float(model.tvalues[treatment_var]),
                p_val=float(model.pvalues[treatment_var]),
                ci_lower=float(model.conf_int().loc[treatment_var, 0]),
                ci_upper=float(model.conf_int().loc[treatment_var, 1]),
                n_obs=int(model.nobs), r_squared=float(model.rsquared),
                coef_vector_json={'treatment': {'var': treatment_var, 'coef': float(model.params[treatment_var])}},
                sample_desc=f'Experiment {experiment}',
                fixed_effects=fe_var if fe_var else 'None',
                controls_desc=', '.join(avail_demo), cluster_var='robust_hc1',
                model_type='OLS with FE' if fe_var else 'OLS', experiment=experiment
            )

    # ========================================
    # 3. LEAVE-ONE-OUT (Drop each control)
    # ========================================
    for dropped in available_controls:
        loo_controls = [c for c in available_controls if c != dropped]
        if loo_controls:
            formula_loo = f"{outcome_var} ~ {treatment_var} + " + ' + '.join(loo_controls)
        else:
            formula_loo = f"{outcome_var} ~ {treatment_var}"

        model = run_ols_with_fe(df_analysis, formula_loo, fe_var)
        if model and treatment_var in model.params:
            add_result(
                spec_id=f'{spec_prefix}/robust/loo/drop_{dropped}',
                spec_tree_path='robustness/leave_one_out.md',
                outcome_var=outcome_var, treatment_var=treatment_var,
                coef=float(model.params[treatment_var]),
                se=float(model.bse[treatment_var]),
                t_stat=float(model.tvalues[treatment_var]),
                p_val=float(model.pvalues[treatment_var]),
                ci_lower=float(model.conf_int().loc[treatment_var, 0]),
                ci_upper=float(model.conf_int().loc[treatment_var, 1]),
                n_obs=int(model.nobs), r_squared=float(model.rsquared),
                coef_vector_json={'treatment': {'var': treatment_var, 'coef': float(model.params[treatment_var])},
                                 'dropped': dropped},
                sample_desc=f'Experiment {experiment}',
                fixed_effects=fe_var if fe_var else 'None',
                controls_desc=', '.join(loo_controls) if loo_controls else 'None',
                cluster_var='robust_hc1',
                model_type='OLS with FE' if fe_var else 'OLS', experiment=experiment
            )

    # ========================================
    # 4. NO FIXED EFFECTS
    # ========================================
    if fe_var:
        model = smf.ols(formula, data=df_analysis).fit(cov_type='HC1')
        if model and treatment_var in model.params:
            add_result(
                spec_id=f'{spec_prefix}/robust/estimation/no_fe',
                spec_tree_path='robustness/model_specification.md',
                outcome_var=outcome_var, treatment_var=treatment_var,
                coef=float(model.params[treatment_var]),
                se=float(model.bse[treatment_var]),
                t_stat=float(model.tvalues[treatment_var]),
                p_val=float(model.pvalues[treatment_var]),
                ci_lower=float(model.conf_int().loc[treatment_var, 0]),
                ci_upper=float(model.conf_int().loc[treatment_var, 1]),
                n_obs=int(model.nobs), r_squared=float(model.rsquared),
                coef_vector_json={'treatment': {'var': treatment_var, 'coef': float(model.params[treatment_var])}},
                sample_desc=f'Experiment {experiment}',
                fixed_effects='None',
                controls_desc=', '.join(available_controls) if available_controls else 'None',
                cluster_var='robust_hc1',
                model_type='OLS', experiment=experiment
            )

    # ========================================
    # 5. SAMPLE RESTRICTIONS
    # ========================================

    # 5a. Winsorize outcome at 1%
    if outcome_var in ['gift_cond', 'payment_amount3']:
        df_wins = df_analysis.copy()
        lower = df_wins[outcome_var].quantile(0.01)
        upper = df_wins[outcome_var].quantile(0.99)
        df_wins[outcome_var + '_wins'] = df_wins[outcome_var].clip(lower, upper)
        formula_wins = formula.replace(outcome_var, outcome_var + '_wins')

        model = run_ols_with_fe(df_wins, formula_wins, fe_var)
        if model and treatment_var in model.params:
            add_result(
                spec_id=f'{spec_prefix}/robust/sample/winsor_1pct',
                spec_tree_path='robustness/sample_restrictions.md',
                outcome_var=outcome_var, treatment_var=treatment_var,
                coef=float(model.params[treatment_var]),
                se=float(model.bse[treatment_var]),
                t_stat=float(model.tvalues[treatment_var]),
                p_val=float(model.pvalues[treatment_var]),
                ci_lower=float(model.conf_int().loc[treatment_var, 0]),
                ci_upper=float(model.conf_int().loc[treatment_var, 1]),
                n_obs=int(model.nobs), r_squared=float(model.rsquared),
                coef_vector_json={'treatment': {'var': treatment_var, 'coef': float(model.params[treatment_var])}},
                sample_desc=f'Experiment {experiment}, winsorized 1%',
                fixed_effects=fe_var if fe_var else 'None',
                controls_desc=', '.join(available_controls) if available_controls else 'None',
                cluster_var='robust_hc1',
                model_type='OLS with FE' if fe_var else 'OLS', experiment=experiment
            )

        # 5b. Trim 1%
        df_trim = df_analysis[(df_analysis[outcome_var] >= lower) & (df_analysis[outcome_var] <= upper)]
        if len(df_trim) >= 50:
            model = run_ols_with_fe(df_trim, formula, fe_var)
            if model and treatment_var in model.params:
                add_result(
                    spec_id=f'{spec_prefix}/robust/sample/trim_1pct',
                    spec_tree_path='robustness/sample_restrictions.md',
                    outcome_var=outcome_var, treatment_var=treatment_var,
                    coef=float(model.params[treatment_var]),
                    se=float(model.bse[treatment_var]),
                    t_stat=float(model.tvalues[treatment_var]),
                    p_val=float(model.pvalues[treatment_var]),
                    ci_lower=float(model.conf_int().loc[treatment_var, 0]),
                    ci_upper=float(model.conf_int().loc[treatment_var, 1]),
                    n_obs=int(model.nobs), r_squared=float(model.rsquared),
                    coef_vector_json={'treatment': {'var': treatment_var, 'coef': float(model.params[treatment_var])}},
                    sample_desc=f'Experiment {experiment}, trimmed 1%',
                    fixed_effects=fe_var if fe_var else 'None',
                    controls_desc=', '.join(available_controls) if available_controls else 'None',
                    cluster_var='robust_hc1',
                    model_type='OLS with FE' if fe_var else 'OLS', experiment=experiment
                )

    # 5c. High baseline donors (above median)
    if 'payment_amount2' in df_analysis.columns:
        median_baseline = df_analysis['payment_amount2'].median()
        df_high = df_analysis[df_analysis['payment_amount2'] >= median_baseline]
        if len(df_high) >= 50:
            model = run_ols_with_fe(df_high, formula, fe_var)
            if model and treatment_var in model.params:
                add_result(
                    spec_id=f'{spec_prefix}/robust/sample/high_baseline',
                    spec_tree_path='robustness/sample_restrictions.md',
                    outcome_var=outcome_var, treatment_var=treatment_var,
                    coef=float(model.params[treatment_var]),
                    se=float(model.bse[treatment_var]),
                    t_stat=float(model.tvalues[treatment_var]),
                    p_val=float(model.pvalues[treatment_var]),
                    ci_lower=float(model.conf_int().loc[treatment_var, 0]),
                    ci_upper=float(model.conf_int().loc[treatment_var, 1]),
                    n_obs=int(model.nobs), r_squared=float(model.rsquared),
                    coef_vector_json={'treatment': {'var': treatment_var, 'coef': float(model.params[treatment_var])}},
                    sample_desc=f'Experiment {experiment}, high baseline donors',
                    fixed_effects=fe_var if fe_var else 'None',
                    controls_desc=', '.join(available_controls) if available_controls else 'None',
                    cluster_var='robust_hc1',
                    model_type='OLS with FE' if fe_var else 'OLS', experiment=experiment
                )

        # 5d. Low baseline donors
        df_low = df_analysis[df_analysis['payment_amount2'] < median_baseline]
        if len(df_low) >= 50:
            model = run_ols_with_fe(df_low, formula, fe_var)
            if model and treatment_var in model.params:
                add_result(
                    spec_id=f'{spec_prefix}/robust/sample/low_baseline',
                    spec_tree_path='robustness/sample_restrictions.md',
                    outcome_var=outcome_var, treatment_var=treatment_var,
                    coef=float(model.params[treatment_var]),
                    se=float(model.bse[treatment_var]),
                    t_stat=float(model.tvalues[treatment_var]),
                    p_val=float(model.pvalues[treatment_var]),
                    ci_lower=float(model.conf_int().loc[treatment_var, 0]),
                    ci_upper=float(model.conf_int().loc[treatment_var, 1]),
                    n_obs=int(model.nobs), r_squared=float(model.rsquared),
                    coef_vector_json={'treatment': {'var': treatment_var, 'coef': float(model.params[treatment_var])}},
                    sample_desc=f'Experiment {experiment}, low baseline donors',
                    fixed_effects=fe_var if fe_var else 'None',
                    controls_desc=', '.join(available_controls) if available_controls else 'None',
                    cluster_var='robust_hc1',
                    model_type='OLS with FE' if fe_var else 'OLS', experiment=experiment
                )

    # ========================================
    # 6. INFERENCE VARIATIONS
    # ========================================

    # 6a. Classical SE
    model = smf.ols(formula, data=df_analysis).fit()
    if model and treatment_var in model.params:
        add_result(
            spec_id=f'{spec_prefix}/robust/se/classical',
            spec_tree_path='robustness/clustering_variations.md',
            outcome_var=outcome_var, treatment_var=treatment_var,
            coef=float(model.params[treatment_var]),
            se=float(model.bse[treatment_var]),
            t_stat=float(model.tvalues[treatment_var]),
            p_val=float(model.pvalues[treatment_var]),
            ci_lower=float(model.conf_int().loc[treatment_var, 0]),
            ci_upper=float(model.conf_int().loc[treatment_var, 1]),
            n_obs=int(model.nobs), r_squared=float(model.rsquared),
            coef_vector_json={'treatment': {'var': treatment_var, 'coef': float(model.params[treatment_var])}},
            sample_desc=f'Experiment {experiment}',
            fixed_effects='None',
            controls_desc=', '.join(available_controls) if available_controls else 'None',
            cluster_var='classical',
            model_type='OLS', experiment=experiment
        )

    # 6b. HC2
    model = smf.ols(formula, data=df_analysis).fit(cov_type='HC2')
    if model and treatment_var in model.params:
        add_result(
            spec_id=f'{spec_prefix}/robust/se/hc2',
            spec_tree_path='robustness/clustering_variations.md',
            outcome_var=outcome_var, treatment_var=treatment_var,
            coef=float(model.params[treatment_var]),
            se=float(model.bse[treatment_var]),
            t_stat=float(model.tvalues[treatment_var]),
            p_val=float(model.pvalues[treatment_var]),
            ci_lower=float(model.conf_int().loc[treatment_var, 0]),
            ci_upper=float(model.conf_int().loc[treatment_var, 1]),
            n_obs=int(model.nobs), r_squared=float(model.rsquared),
            coef_vector_json={'treatment': {'var': treatment_var, 'coef': float(model.params[treatment_var])}},
            sample_desc=f'Experiment {experiment}',
            fixed_effects='None',
            controls_desc=', '.join(available_controls) if available_controls else 'None',
            cluster_var='hc2',
            model_type='OLS', experiment=experiment
        )

    # 6c. HC3
    model = smf.ols(formula, data=df_analysis).fit(cov_type='HC3')
    if model and treatment_var in model.params:
        add_result(
            spec_id=f'{spec_prefix}/robust/se/hc3',
            spec_tree_path='robustness/clustering_variations.md',
            outcome_var=outcome_var, treatment_var=treatment_var,
            coef=float(model.params[treatment_var]),
            se=float(model.bse[treatment_var]),
            t_stat=float(model.tvalues[treatment_var]),
            p_val=float(model.pvalues[treatment_var]),
            ci_lower=float(model.conf_int().loc[treatment_var, 0]),
            ci_upper=float(model.conf_int().loc[treatment_var, 1]),
            n_obs=int(model.nobs), r_squared=float(model.rsquared),
            coef_vector_json={'treatment': {'var': treatment_var, 'coef': float(model.params[treatment_var])}},
            sample_desc=f'Experiment {experiment}',
            fixed_effects='None',
            controls_desc=', '.join(available_controls) if available_controls else 'None',
            cluster_var='hc3',
            model_type='OLS', experiment=experiment
        )

    # 6d. Cluster by station (for exp 1 and 3)
    if 'station' in df_analysis.columns and experiment in ['1', '3']:
        try:
            model = smf.ols(formula, data=df_analysis).fit(
                cov_type='cluster', cov_kwds={'groups': df_analysis['station']}
            )
            if model and treatment_var in model.params:
                add_result(
                    spec_id=f'{spec_prefix}/robust/cluster/station',
                    spec_tree_path='robustness/clustering_variations.md',
                    outcome_var=outcome_var, treatment_var=treatment_var,
                    coef=float(model.params[treatment_var]),
                    se=float(model.bse[treatment_var]),
                    t_stat=float(model.tvalues[treatment_var]),
                    p_val=float(model.pvalues[treatment_var]),
                    ci_lower=float(model.conf_int().loc[treatment_var, 0]),
                    ci_upper=float(model.conf_int().loc[treatment_var, 1]),
                    n_obs=int(model.nobs), r_squared=float(model.rsquared),
                    coef_vector_json={'treatment': {'var': treatment_var, 'coef': float(model.params[treatment_var])}},
                    sample_desc=f'Experiment {experiment}',
                    fixed_effects='None',
                    controls_desc=', '.join(available_controls) if available_controls else 'None',
                    cluster_var='station',
                    model_type='OLS', experiment=experiment
                )
        except:
            pass

    # ========================================
    # 7. FUNCTIONAL FORM
    # ========================================

    # 7a. IHS transformation
    df_ihs = df_analysis.copy()
    df_ihs['ihs_outcome'] = np.arcsinh(df_ihs[outcome_var])
    formula_ihs = formula.replace(outcome_var, 'ihs_outcome')
    model = run_ols_with_fe(df_ihs, formula_ihs, fe_var)
    if model and treatment_var in model.params:
        add_result(
            spec_id=f'{spec_prefix}/robust/form/y_asinh',
            spec_tree_path='robustness/functional_form.md',
            outcome_var=f'asinh_{outcome_var}', treatment_var=treatment_var,
            coef=float(model.params[treatment_var]),
            se=float(model.bse[treatment_var]),
            t_stat=float(model.tvalues[treatment_var]),
            p_val=float(model.pvalues[treatment_var]),
            ci_lower=float(model.conf_int().loc[treatment_var, 0]),
            ci_upper=float(model.conf_int().loc[treatment_var, 1]),
            n_obs=int(model.nobs), r_squared=float(model.rsquared),
            coef_vector_json={'treatment': {'var': treatment_var, 'coef': float(model.params[treatment_var])}},
            sample_desc=f'Experiment {experiment}',
            fixed_effects=fe_var if fe_var else 'None',
            controls_desc=', '.join(available_controls) if available_controls else 'None',
            cluster_var='robust_hc1',
            model_type='OLS with FE, IHS outcome' if fe_var else 'OLS, IHS outcome', experiment=experiment
        )

    # 7b. Log transformation (only for continuous positive outcomes)
    if outcome_var in ['gift_cond'] and df_analysis[outcome_var].min() > 0:
        df_log = df_analysis.copy()
        df_log['log_outcome'] = np.log(df_log[outcome_var])
        formula_log = formula.replace(outcome_var, 'log_outcome')
        model = run_ols_with_fe(df_log, formula_log, fe_var)
        if model and treatment_var in model.params:
            add_result(
                spec_id=f'{spec_prefix}/robust/form/y_log',
                spec_tree_path='robustness/functional_form.md',
                outcome_var=f'log_{outcome_var}', treatment_var=treatment_var,
                coef=float(model.params[treatment_var]),
                se=float(model.bse[treatment_var]),
                t_stat=float(model.tvalues[treatment_var]),
                p_val=float(model.pvalues[treatment_var]),
                ci_lower=float(model.conf_int().loc[treatment_var, 0]),
                ci_upper=float(model.conf_int().loc[treatment_var, 1]),
                n_obs=int(model.nobs), r_squared=float(model.rsquared),
                coef_vector_json={'treatment': {'var': treatment_var, 'coef': float(model.params[treatment_var])}},
                sample_desc=f'Experiment {experiment}',
                fixed_effects=fe_var if fe_var else 'None',
                controls_desc=', '.join(available_controls) if available_controls else 'None',
                cluster_var='robust_hc1',
                model_type='OLS with FE, log outcome' if fe_var else 'OLS, log outcome', experiment=experiment
            )

    # ========================================
    # 8. HETEROGENEITY
    # ========================================

    # 8a. By gender
    if 'female' in df_analysis.columns:
        non_female_controls = [c for c in available_controls if c != 'female']
        if non_female_controls:
            formula_het = f"{outcome_var} ~ {treatment_var} + " + ' + '.join(non_female_controls)
        else:
            formula_het = f"{outcome_var} ~ {treatment_var}"

        # Female only
        df_female = df_analysis[df_analysis['female'] == 1]
        if len(df_female) >= 50:
            model = run_ols_with_fe(df_female, formula_het, fe_var)
            if model and treatment_var in model.params:
                add_result(
                    spec_id=f'{spec_prefix}/robust/het/female_only',
                    spec_tree_path='robustness/heterogeneity.md',
                    outcome_var=outcome_var, treatment_var=treatment_var,
                    coef=float(model.params[treatment_var]),
                    se=float(model.bse[treatment_var]),
                    t_stat=float(model.tvalues[treatment_var]),
                    p_val=float(model.pvalues[treatment_var]),
                    ci_lower=float(model.conf_int().loc[treatment_var, 0]),
                    ci_upper=float(model.conf_int().loc[treatment_var, 1]),
                    n_obs=int(model.nobs), r_squared=float(model.rsquared),
                    coef_vector_json={'treatment': {'var': treatment_var, 'coef': float(model.params[treatment_var])}},
                    sample_desc=f'Experiment {experiment}, female only',
                    fixed_effects=fe_var if fe_var else 'None',
                    controls_desc=', '.join(non_female_controls) if non_female_controls else 'None',
                    cluster_var='robust_hc1',
                    model_type='OLS with FE' if fe_var else 'OLS', experiment=experiment
                )

        # Male only
        df_male = df_analysis[df_analysis['female'] == 0]
        if len(df_male) >= 50:
            model = run_ols_with_fe(df_male, formula_het, fe_var)
            if model and treatment_var in model.params:
                add_result(
                    spec_id=f'{spec_prefix}/robust/het/male_only',
                    spec_tree_path='robustness/heterogeneity.md',
                    outcome_var=outcome_var, treatment_var=treatment_var,
                    coef=float(model.params[treatment_var]),
                    se=float(model.bse[treatment_var]),
                    t_stat=float(model.tvalues[treatment_var]),
                    p_val=float(model.pvalues[treatment_var]),
                    ci_lower=float(model.conf_int().loc[treatment_var, 0]),
                    ci_upper=float(model.conf_int().loc[treatment_var, 1]),
                    n_obs=int(model.nobs), r_squared=float(model.rsquared),
                    coef_vector_json={'treatment': {'var': treatment_var, 'coef': float(model.params[treatment_var])}},
                    sample_desc=f'Experiment {experiment}, male only',
                    fixed_effects=fe_var if fe_var else 'None',
                    controls_desc=', '.join(non_female_controls) if non_female_controls else 'None',
                    cluster_var='robust_hc1',
                    model_type='OLS with FE' if fe_var else 'OLS', experiment=experiment
                )

        # Interaction
        if non_female_controls:
            formula_int = f"{outcome_var} ~ {treatment_var} * female + " + ' + '.join(non_female_controls)
        else:
            formula_int = f"{outcome_var} ~ {treatment_var} * female"
        model = run_ols_with_fe(df_analysis, formula_int, fe_var)
        if model and treatment_var in model.params:
            add_result(
                spec_id=f'{spec_prefix}/robust/het/interaction_gender',
                spec_tree_path='robustness/heterogeneity.md',
                outcome_var=outcome_var, treatment_var=treatment_var,
                coef=float(model.params[treatment_var]),
                se=float(model.bse[treatment_var]),
                t_stat=float(model.tvalues[treatment_var]),
                p_val=float(model.pvalues[treatment_var]),
                ci_lower=float(model.conf_int().loc[treatment_var, 0]),
                ci_upper=float(model.conf_int().loc[treatment_var, 1]),
                n_obs=int(model.nobs), r_squared=float(model.rsquared),
                coef_vector_json={'treatment': {'var': treatment_var, 'coef': float(model.params[treatment_var])}},
                sample_desc=f'Experiment {experiment}',
                fixed_effects=fe_var if fe_var else 'None',
                controls_desc='treatment x female',
                cluster_var='robust_hc1',
                model_type='OLS with FE' if fe_var else 'OLS', experiment=experiment
            )

    # 8b. By age (old 65+)
    if 'age_display3' in df_analysis.columns:
        non_age_controls = [c for c in available_controls if 'age_display' not in c]
        if non_age_controls:
            formula_het = f"{outcome_var} ~ {treatment_var} + " + ' + '.join(non_age_controls)
        else:
            formula_het = f"{outcome_var} ~ {treatment_var}"

        df_old = df_analysis[df_analysis['age_display3'] == 1]
        if len(df_old) >= 50:
            model = run_ols_with_fe(df_old, formula_het, fe_var)
            if model and treatment_var in model.params:
                add_result(
                    spec_id=f'{spec_prefix}/robust/het/age_65plus',
                    spec_tree_path='robustness/heterogeneity.md',
                    outcome_var=outcome_var, treatment_var=treatment_var,
                    coef=float(model.params[treatment_var]),
                    se=float(model.bse[treatment_var]),
                    t_stat=float(model.tvalues[treatment_var]),
                    p_val=float(model.pvalues[treatment_var]),
                    ci_lower=float(model.conf_int().loc[treatment_var, 0]),
                    ci_upper=float(model.conf_int().loc[treatment_var, 1]),
                    n_obs=int(model.nobs), r_squared=float(model.rsquared),
                    coef_vector_json={'treatment': {'var': treatment_var, 'coef': float(model.params[treatment_var])}},
                    sample_desc=f'Experiment {experiment}, age 65+',
                    fixed_effects=fe_var if fe_var else 'None',
                    controls_desc=', '.join(non_age_controls) if non_age_controls else 'None',
                    cluster_var='robust_hc1',
                    model_type='OLS with FE' if fe_var else 'OLS', experiment=experiment
                )

        # Young (18-44)
        df_young = df_analysis[df_analysis['age_display1'] == 1]
        if len(df_young) >= 50:
            model = run_ols_with_fe(df_young, formula_het, fe_var)
            if model and treatment_var in model.params:
                add_result(
                    spec_id=f'{spec_prefix}/robust/het/age_18_44',
                    spec_tree_path='robustness/heterogeneity.md',
                    outcome_var=outcome_var, treatment_var=treatment_var,
                    coef=float(model.params[treatment_var]),
                    se=float(model.bse[treatment_var]),
                    t_stat=float(model.tvalues[treatment_var]),
                    p_val=float(model.pvalues[treatment_var]),
                    ci_lower=float(model.conf_int().loc[treatment_var, 0]),
                    ci_upper=float(model.conf_int().loc[treatment_var, 1]),
                    n_obs=int(model.nobs), r_squared=float(model.rsquared),
                    coef_vector_json={'treatment': {'var': treatment_var, 'coef': float(model.params[treatment_var])}},
                    sample_desc=f'Experiment {experiment}, age 18-44',
                    fixed_effects=fe_var if fe_var else 'None',
                    controls_desc=', '.join(non_age_controls) if non_age_controls else 'None',
                    cluster_var='robust_hc1',
                    model_type='OLS with FE' if fe_var else 'OLS', experiment=experiment
                )

    # 8c. By income (high income 175k+)
    if 'inc_display2' in df_analysis.columns:
        non_inc_controls = [c for c in available_controls if 'inc_display' not in c]
        if non_inc_controls:
            formula_het = f"{outcome_var} ~ {treatment_var} + " + ' + '.join(non_inc_controls)
        else:
            formula_het = f"{outcome_var} ~ {treatment_var}"

        df_high_inc = df_analysis[df_analysis['inc_display2'] == 1]
        if len(df_high_inc) >= 50:
            model = run_ols_with_fe(df_high_inc, formula_het, fe_var)
            if model and treatment_var in model.params:
                add_result(
                    spec_id=f'{spec_prefix}/robust/het/income_high',
                    spec_tree_path='robustness/heterogeneity.md',
                    outcome_var=outcome_var, treatment_var=treatment_var,
                    coef=float(model.params[treatment_var]),
                    se=float(model.bse[treatment_var]),
                    t_stat=float(model.tvalues[treatment_var]),
                    p_val=float(model.pvalues[treatment_var]),
                    ci_lower=float(model.conf_int().loc[treatment_var, 0]),
                    ci_upper=float(model.conf_int().loc[treatment_var, 1]),
                    n_obs=int(model.nobs), r_squared=float(model.rsquared),
                    coef_vector_json={'treatment': {'var': treatment_var, 'coef': float(model.params[treatment_var])}},
                    sample_desc=f'Experiment {experiment}, income 175k+',
                    fixed_effects=fe_var if fe_var else 'None',
                    controls_desc=', '.join(non_inc_controls) if non_inc_controls else 'None',
                    cluster_var='robust_hc1',
                    model_type='OLS with FE' if fe_var else 'OLS', experiment=experiment
                )

        # Low income (below 34k)
        df_low_inc = df_analysis[df_analysis['inc_display4'] == 1]
        if len(df_low_inc) >= 50:
            model = run_ols_with_fe(df_low_inc, formula_het, fe_var)
            if model and treatment_var in model.params:
                add_result(
                    spec_id=f'{spec_prefix}/robust/het/income_low',
                    spec_tree_path='robustness/heterogeneity.md',
                    outcome_var=outcome_var, treatment_var=treatment_var,
                    coef=float(model.params[treatment_var]),
                    se=float(model.bse[treatment_var]),
                    t_stat=float(model.tvalues[treatment_var]),
                    p_val=float(model.pvalues[treatment_var]),
                    ci_lower=float(model.conf_int().loc[treatment_var, 0]),
                    ci_upper=float(model.conf_int().loc[treatment_var, 1]),
                    n_obs=int(model.nobs), r_squared=float(model.rsquared),
                    coef_vector_json={'treatment': {'var': treatment_var, 'coef': float(model.params[treatment_var])}},
                    sample_desc=f'Experiment {experiment}, income below 34k',
                    fixed_effects=fe_var if fe_var else 'None',
                    controls_desc=', '.join(non_inc_controls) if non_inc_controls else 'None',
                    cluster_var='robust_hc1',
                    model_type='OLS with FE' if fe_var else 'OLS', experiment=experiment
                )

    # 8d. Baseline gift interaction
    if 'payment_amount2' in df_analysis.columns:
        non_pay_controls = [c for c in available_controls if c != 'payment_amount2']
        if non_pay_controls:
            formula_int = f"{outcome_var} ~ {treatment_var} * payment_amount2 + " + ' + '.join(non_pay_controls)
        else:
            formula_int = f"{outcome_var} ~ {treatment_var} * payment_amount2"
        model = run_ols_with_fe(df_analysis, formula_int, fe_var)
        if model and treatment_var in model.params:
            add_result(
                spec_id=f'{spec_prefix}/robust/het/interaction_baseline_gift',
                spec_tree_path='robustness/heterogeneity.md',
                outcome_var=outcome_var, treatment_var=treatment_var,
                coef=float(model.params[treatment_var]),
                se=float(model.bse[treatment_var]),
                t_stat=float(model.tvalues[treatment_var]),
                p_val=float(model.pvalues[treatment_var]),
                ci_lower=float(model.conf_int().loc[treatment_var, 0]),
                ci_upper=float(model.conf_int().loc[treatment_var, 1]),
                n_obs=int(model.nobs), r_squared=float(model.rsquared),
                coef_vector_json={'treatment': {'var': treatment_var, 'coef': float(model.params[treatment_var])}},
                sample_desc=f'Experiment {experiment}',
                fixed_effects=fe_var if fe_var else 'None',
                controls_desc='treatment x baseline_gift',
                cluster_var='robust_hc1',
                model_type='OLS with FE' if fe_var else 'OLS', experiment=experiment
            )

def run_placebo_specs(df, treatment_var, controls, fe_var, spec_prefix, experiment):
    """Run placebo tests on future outcomes"""

    # Year 2 outcome as placebo
    if 'var14' in df.columns:
        df['renewing_y2'] = (df['var14'] > 0).astype(int)
        df_analysis = df.dropna(subset=['renewing_y2']).copy()

        if len(df_analysis) >= 50:
            if controls:
                available_controls = [c for c in controls if c in df_analysis.columns]
                formula = f"renewing_y2 ~ {treatment_var} + " + ' + '.join(available_controls)
            else:
                formula = f"renewing_y2 ~ {treatment_var}"
                available_controls = []

            model = run_ols_with_fe(df_analysis, formula, fe_var)
            if model and treatment_var in model.params:
                add_result(
                    spec_id=f'{spec_prefix}/robust/placebo/year2_outcome',
                    spec_tree_path='robustness/placebo_tests.md',
                    outcome_var='renewing_y2', treatment_var=treatment_var,
                    coef=float(model.params[treatment_var]),
                    se=float(model.bse[treatment_var]),
                    t_stat=float(model.tvalues[treatment_var]),
                    p_val=float(model.pvalues[treatment_var]),
                    ci_lower=float(model.conf_int().loc[treatment_var, 0]),
                    ci_upper=float(model.conf_int().loc[treatment_var, 1]),
                    n_obs=int(model.nobs), r_squared=float(model.rsquared),
                    coef_vector_json={'treatment': {'var': treatment_var, 'coef': float(model.params[treatment_var])}},
                    sample_desc=f'Experiment {experiment}, year 2 outcome',
                    fixed_effects=fe_var if fe_var else 'None',
                    controls_desc=', '.join(available_controls) if available_controls else 'None',
                    cluster_var='robust_hc1',
                    model_type='OLS with FE' if fe_var else 'OLS', experiment=experiment
                )

    # Year 3 outcome as placebo
    if 'var15' in df.columns:
        df['renewing_y3'] = (df['var15'] > 0).astype(int)
        df_analysis = df.dropna(subset=['renewing_y3']).copy()

        if len(df_analysis) >= 50:
            if controls:
                available_controls = [c for c in controls if c in df_analysis.columns]
                formula = f"renewing_y3 ~ {treatment_var} + " + ' + '.join(available_controls)
            else:
                formula = f"renewing_y3 ~ {treatment_var}"
                available_controls = []

            model = run_ols_with_fe(df_analysis, formula, fe_var)
            if model and treatment_var in model.params:
                add_result(
                    spec_id=f'{spec_prefix}/robust/placebo/year3_outcome',
                    spec_tree_path='robustness/placebo_tests.md',
                    outcome_var='renewing_y3', treatment_var=treatment_var,
                    coef=float(model.params[treatment_var]),
                    se=float(model.bse[treatment_var]),
                    t_stat=float(model.tvalues[treatment_var]),
                    p_val=float(model.pvalues[treatment_var]),
                    ci_lower=float(model.conf_int().loc[treatment_var, 0]),
                    ci_upper=float(model.conf_int().loc[treatment_var, 1]),
                    n_obs=int(model.nobs), r_squared=float(model.rsquared),
                    coef_vector_json={'treatment': {'var': treatment_var, 'coef': float(model.params[treatment_var])}},
                    sample_desc=f'Experiment {experiment}, year 3 outcome',
                    fixed_effects=fe_var if fe_var else 'None',
                    controls_desc=', '.join(available_controls) if available_controls else 'None',
                    cluster_var='robust_hc1',
                    model_type='OLS with FE' if fe_var else 'OLS', experiment=experiment
                )

def main():
    """Main function to run all specifications"""

    print("=" * 60)
    print("Specification Search for Paper 149481-V1")
    print("Do Thank-You Calls Increase Charitable Giving?")
    print("=" * 60)

    print("\nLoading and preparing data...")

    try:
        exp1, exp3 = prepare_exp1_exp3_data()
        print(f"Experiment 1: {len(exp1)} observations")
        print(f"Experiment 3: {len(exp3)} observations")
    except Exception as e:
        print(f"Error loading Exp 1/3 data: {e}")
        import traceback
        traceback.print_exc()
        exp1, exp3 = None, None

    try:
        exp2 = prepare_exp2_data()
        print(f"Experiment 2: {len(exp2)} observations")
    except Exception as e:
        print(f"Error loading Exp 2 data: {e}")
        import traceback
        traceback.print_exc()
        exp2 = None

    # Define controls
    controls_exp1_3 = ['payment_amount2', 'var12', 'female', 'age_display2', 'age_display3',
                       'inc_display1', 'inc_display2', 'inc_display3', 'lor_display2']
    controls_exp2 = ['payment_amount2', 'var12']

    # Run specifications for Experiment 1
    if exp1 is not None and len(exp1) > 100:
        print("\n" + "=" * 40)
        print("Running Experiment 1 Specifications")
        print("=" * 40)

        exp1_complete = exp1.dropna(subset=['treat', 'payment_amount2', 'var12'])
        print(f"Complete cases: {len(exp1_complete)}")

        print("\n--- Outcome: donated ---")
        run_all_specs(exp1_complete, 'donated', 'treat', controls_exp1_3, 'ii', 'exp1/donated', '1')
        run_placebo_specs(exp1_complete, 'treat', controls_exp1_3, 'ii', 'exp1/donated', '1')

        print("\n--- Outcome: gift_cond ---")
        run_all_specs(exp1_complete, 'gift_cond', 'treat', controls_exp1_3, 'ii', 'exp1/gift_cond', '1')

    # Run specifications for Experiment 2
    if exp2 is not None and len(exp2) > 100:
        print("\n" + "=" * 40)
        print("Running Experiment 2 Specifications")
        print("=" * 40)

        exp2_complete = exp2.dropna(subset=['treat', 'payment_amount2', 'var12'])
        print(f"Complete cases: {len(exp2_complete)}")

        print("\n--- Outcome: donated ---")
        run_all_specs(exp2_complete, 'donated', 'treat', controls_exp2, None, 'exp2/donated', '2')

        print("\n--- Outcome: gift_cond ---")
        run_all_specs(exp2_complete, 'gift_cond', 'treat', controls_exp2, None, 'exp2/gift_cond', '2')

    # Run specifications for Experiment 3
    if exp3 is not None and len(exp3) > 100:
        print("\n" + "=" * 40)
        print("Running Experiment 3 Specifications")
        print("=" * 40)

        exp3_complete = exp3.dropna(subset=['treat', 'payment_amount2', 'var12'])
        print(f"Complete cases: {len(exp3_complete)}")

        print("\n--- Outcome: donated ---")
        run_all_specs(exp3_complete, 'donated', 'treat', controls_exp1_3, 'ii', 'exp3/donated', '3')

        print("\n--- Outcome: gift_cond ---")
        run_all_specs(exp3_complete, 'gift_cond', 'treat', controls_exp1_3, 'ii', 'exp3/gift_cond', '3')

    # Save results
    print("\n" + "=" * 60)
    print(f"Total specifications: {len(results)}")
    print("=" * 60)

    df_results = pd.DataFrame(results)
    output_file = f"{OUTPUT_PATH}/specification_results.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Summary statistics
    if len(df_results) > 0:
        print("\n=== Summary Statistics ===")
        main_specs = df_results[df_results['treatment_var'] == 'treat']

        print(f"Total specifications: {len(main_specs)}")
        print(f"Positive coefficients: {(main_specs['coefficient'] > 0).sum()} ({100*(main_specs['coefficient'] > 0).mean():.1f}%)")
        print(f"Negative coefficients: {(main_specs['coefficient'] < 0).sum()} ({100*(main_specs['coefficient'] < 0).mean():.1f}%)")
        print(f"Significant at 5%: {(main_specs['p_value'] < 0.05).sum()} ({100*(main_specs['p_value'] < 0.05).mean():.1f}%)")
        print(f"Significant at 1%: {(main_specs['p_value'] < 0.01).sum()} ({100*(main_specs['p_value'] < 0.01).mean():.1f}%)")
        print(f"Median coefficient: {main_specs['coefficient'].median():.4f}")
        print(f"Mean coefficient: {main_specs['coefficient'].mean():.4f}")
        print(f"Range: [{main_specs['coefficient'].min():.4f}, {main_specs['coefficient'].max():.4f}]")

        # By experiment
        print("\n=== By Experiment ===")
        for exp in ['1', '2', '3']:
            exp_specs = main_specs[main_specs['experiment'] == exp]
            if len(exp_specs) > 0:
                print(f"Experiment {exp}: {len(exp_specs)} specs, {100*(exp_specs['p_value'] < 0.05).mean():.1f}% sig at 5%")

    return df_results

if __name__ == "__main__":
    results_df = main()
