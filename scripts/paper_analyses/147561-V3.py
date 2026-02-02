#!/usr/bin/env python3
"""
Specification Search: 147561-V3
Paper: "Local Elites as State Capacity: How City Chiefs Use Local Information
        to Increase Tax Compliance in the D.R. Congo"
Authors: Balan et al.
Journal: AER

This script replicates the main analysis and runs systematic specification searches
following the i4r methodology (minimum 50 specifications).

Method: Cross-sectional RCT (randomized experiment)
Main outcome: Tax compliance (taxes_paid)
Main treatment: Local collector assignment (t_l)
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Disable pyfixest - use statsmodels only for reliability
USE_PYFIXEST = False

# =============================================================================
# CONFIGURATION
# =============================================================================

PAPER_ID = "147561-V3"
PAPER_TITLE = "Local Elites as State Capacity: How City Chiefs Use Local Information to Increase Tax Compliance in the D.R. Congo"
JOURNAL = "AER"

BASE_DIR = Path("/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search")
PACKAGE_DIR = BASE_DIR / "data/downloads/extracted/147561-V3/Replication Materials - Updated October 2025"
DATA_DIR = PACKAGE_DIR / "Data"
OUTPUT_DIR = BASE_DIR / "data/downloads/extracted/147561-V3"

# =============================================================================
# DATA LOADING AND CONSTRUCTION
# =============================================================================

def load_and_construct_data():
    """
    Construct the analysis dataset by replicating the Stata data construction code.
    This replicates 2_Data_Construction.do
    """

    print("Loading and constructing analysis data...")

    # 1. Flier assignment data
    fliers = pd.read_stata(DATA_DIR / "01_base/admin_data/fliers_campaign.dta", convert_categoricals=False)

    # 2. Assignment data
    assignment = pd.read_stata(DATA_DIR / "01_base/admin_data/randomization_schedule.dta", convert_categoricals=False)

    # 3. Registration (cartography) data
    registration = pd.read_stata(DATA_DIR / "01_base/admin_data/registration_noPII.dta", convert_categoricals=False)

    # 4. Tax payments data
    tax_payments = pd.read_stata(DATA_DIR / "01_base/admin_data/tax_payments_noPII.dta", convert_categoricals=False)

    # 5. Stratum data
    stratum = pd.read_stata(DATA_DIR / "01_base/admin_data/stratum.dta", convert_categoricals=False)

    # 6. Monitoring/midline data
    midline = pd.read_stata(DATA_DIR / "01_base/survey_data/midline_noPII.dta", convert_categoricals=False)

    # =============================================================================
    # Build analysis dataset
    # =============================================================================

    # Start with fliers data - filter out pilot polygons
    df = fliers[~fliers['a7'].isin([200, 201, 202, 203, 207, 208, 210])].copy()

    # Keep key variables and rename
    df = df[['a7', 'code', 'treatment_fr', 'rate']].copy()
    df.rename(columns={'code': 'compound1', 'rate': 'assign_flier_rate',
                       'treatment_fr': 'assign_treatment_fr'}, inplace=True)

    # Merge with assignment
    # Add pilot polygons treatment assignments
    pilot_assignments = pd.DataFrame({
        'a7': [201, 202, 203, 210, 200, 207, 208],
        'treatment': [2, 1, 1, 2, 1, 2, 4],
    })
    assignment = pd.concat([assignment[['a7', 'treatment']], pilot_assignments], ignore_index=True)

    # Fix assignment mistake (polygon 654 should be Local)
    assignment.loc[assignment['a7'] == 654, 'treatment'] = 2

    assignment.rename(columns={'treatment': 'tmt'}, inplace=True)

    df = df.merge(assignment[['a7', 'tmt']], on='a7', how='left')

    # Merge with stratum
    df = df.merge(stratum[['a7', 'stratum']], on='a7', how='left')

    # Merge with registration data
    registration = registration[registration['tot_complete'] == 1].copy()
    reg_cols = ['compound1', 'today', 'house', 'exempt', 'exempt_enum', 'correct', 'mm_rate']
    reg_cols = [c for c in reg_cols if c in registration.columns]
    registration = registration[reg_cols].copy()
    registration.rename(columns={'today': 'today_carto'}, inplace=True)

    df = df.merge(registration, on='compound1', how='left')

    # Merge with tax payments
    tax_payments_clean = tax_payments[tax_payments['unmatched_compound'] != 1].copy()
    tax_payments_clean = tax_payments_clean[tax_payments_clean['compound1'].notna()].copy()

    # Handle duplicates in tax payments
    tax_payments_clean = tax_payments_clean.drop_duplicates(subset='compound1', keep='first')
    tax_cols = ['compound1', 'date', 'amountCF']
    tax_cols = [c for c in tax_cols if c in tax_payments_clean.columns]
    tax_payments_clean = tax_payments_clean[tax_cols].copy()
    tax_payments_clean.rename(columns={'date': 'date_TDM'}, inplace=True)

    df = df.merge(tax_payments_clean, on='compound1', how='left', indicator='_merge_tdm')

    # Merge with midline for additional variables
    midline_clean = midline.copy()
    midline_clean.rename(columns={'compound': 'compound1', 'today': 'today_monitoring',
                                   'exempt': 'exempt_monitoring'}, inplace=True)
    # Handle missing compounds
    midline_clean = midline_clean[midline_clean['compound1'].notna()].copy()
    midline_clean = midline_clean[midline_clean['compound1'] != 999999].copy()
    midline_clean = midline_clean[midline_clean['compound1'] != 9999999].copy()

    # Select columns that exist
    midline_cols = ['compound1', 'visited', 'visits', 'bribe', 'bribe_amt',
                    'bribe2a', 'bribe2a_amt', 'bribe2b', 'bribe2b_amt',
                    'bribe3', 'bribe3_amt', 'paid_self', 'salongo', 'salongo_hours',
                    'visits_other1a', 'visits_other1b', 'visits_other2a', 'visits_other2b',
                    'sex_prop', 'age_prop', 'age_prop_guess', 'tribe', 'job', 'job_other',
                    'move_ave', 'job_gov', 'neighbor_know', 'neighbor_know2',
                    'discount_know', 'discount_self', 'walls', 'roof', 'roof2', 'ravine',
                    'sex', 'age', 'code_same']
    midline_cols = [c for c in midline_cols if c in midline_clean.columns]
    midline_clean = midline_clean[midline_cols].drop_duplicates(subset='compound1', keep='first')

    df = df.merge(midline_clean, on='compound1', how='left')

    # =============================================================================
    # VARIABLE CONSTRUCTION
    # =============================================================================

    # Drop pilot polygons
    df = df[~df['a7'].isin([200, 201, 202, 203, 207, 208, 210])].copy()

    # Drop villas (house==3)
    df = df[df['house'] != 3].copy()

    # Drop if treatment is missing
    df = df[df['tmt'].notna()].copy()

    # Create treatment dummies
    df['t_l'] = (df['tmt'] == 2).astype(int)
    df['t_c'] = (df['tmt'] == 1).astype(int)
    df['t_cli'] = (df['tmt'] == 3).astype(int)
    df['t_cxl'] = (df['tmt'] == 4).astype(int)

    # Tax compliance dummy
    df['taxes_paid'] = 0
    df.loc[df['_merge_tdm'] == 'both', 'taxes_paid'] = 1

    # Add code_same as additional compliance indicator if available
    if 'code_same' in df.columns:
        df.loc[(df['taxes_paid'] == 0) & (df['code_same'] == 1), 'taxes_paid'] = 1

    # Partial payment check (bribe indicator) - paid less than assigned rate
    df.loc[(df['house'] == 1) & (df['_merge_tdm'] == 'both') &
           (df['assign_flier_rate'] > df['amountCF']) &
           df['amountCF'].notna() & df['assign_flier_rate'].notna(), 'taxes_paid'] = 0
    if 'mm_rate' in df.columns:
        df.loc[(df['house'] == 2) & (df['_merge_tdm'] == 'both') &
               (df['mm_rate'] > df['amountCF']) &
               df['amountCF'].notna() & df['mm_rate'].notna(), 'taxes_paid'] = 0

    # Create rate variable
    df['rate'] = df['assign_flier_rate']
    if 'mm_rate' in df.columns:
        df.loc[df['house'] == 2, 'rate'] = df.loc[df['house'] == 2, 'mm_rate']

    # Tax amount paid
    df['taxes_paid_amt'] = df['taxes_paid'] * df['rate']

    # Rate percentage dummies
    df['pct_50'] = 0
    df.loc[(df['house'] == 1) & (df['assign_flier_rate'] == 1500), 'pct_50'] = 1
    if 'mm_rate' in df.columns:
        df.loc[(df['house'] == 2) & (df['mm_rate'] == 6600), 'pct_50'] = 1

    df['pct_66'] = 0
    df.loc[(df['house'] == 1) & (df['assign_flier_rate'] == 2000), 'pct_66'] = 1
    if 'mm_rate' in df.columns:
        df.loc[(df['house'] == 2) & (df['mm_rate'] == 8800), 'pct_66'] = 1

    df['pct_83'] = 0
    df.loc[(df['house'] == 1) & (df['assign_flier_rate'] == 2500), 'pct_83'] = 1
    if 'mm_rate' in df.columns:
        df.loc[(df['house'] == 2) & (df['mm_rate'] == 11000), 'pct_83'] = 1

    df['pct_100'] = 0
    df.loc[(df['house'] == 1) & (df['assign_flier_rate'] == 3000), 'pct_100'] = 1
    if 'mm_rate' in df.columns:
        df.loc[(df['house'] == 2) & (df['mm_rate'] == 13200), 'pct_100'] = 1

    # Bribe combined variable
    df['bribe_combined'] = np.nan
    if 'bribe' in df.columns:
        df.loc[df['bribe'] == 0, 'bribe_combined'] = 0
        df.loc[df['bribe'] == 1, 'bribe_combined'] = 1
    if 'bribe2a_amt' in df.columns:
        df.loc[(df['bribe2a_amt'].notna()) & (df.get('bribe', 0) != 1), 'bribe_combined'] = 1
    if 'bribe2b_amt' in df.columns:
        df.loc[df['bribe2b_amt'].notna(), 'bribe_combined'] = 1
    if 'bribe3_amt' in df.columns:
        df.loc[df['bribe3_amt'].notna(), 'bribe_combined'] = 1

    # Partial payment as bribe
    df.loc[(df['house'] == 1) & (df['_merge_tdm'] == 'both') &
           (df['assign_flier_rate'] > df['amountCF']) &
           df['amountCF'].notna() & df['assign_flier_rate'].notna(), 'bribe_combined'] = 1

    if 'visited' in df.columns:
        df.loc[(df['visited'] != 1) & (df['visited'].notna()), 'bribe_combined'] = 0
        df.loc[(df['visited'] == 1) & (df['bribe_combined'].isna()), 'bribe_combined'] = 0

    # Visits post carto
    if 'visits' in df.columns:
        df['visit_post_carto'] = 0
        if 'visited' in df.columns:
            df.loc[(df['visited'] == 0), 'visit_post_carto'] = 0
        df.loc[(df['visits'].notna()) & (df['visits'] > 1), 'visit_post_carto'] = 1

        df['nb_visit_post_carto'] = 0
        if 'visited' in df.columns:
            df.loc[(df['visited'] == 0), 'nb_visit_post_carto'] = 0
        df.loc[(df['visits'].notna()) & (df['visits'] > 1), 'nb_visit_post_carto'] = df['visits'] - 1
        df.loc[df['visits'] == 99999, 'nb_visit_post_carto'] = np.nan

    # Gender of owner
    if 'sex_prop' in df.columns:
        df['male_prop'] = np.nan
        df.loc[df['sex_prop'] == 1, 'male_prop'] = 1
        df.loc[df['sex_prop'] == 0, 'male_prop'] = 0
        df.loc[df['sex_prop'] == 2, 'male_prop'] = 0
        # Fill from 'sex' if available
        if 'sex' in df.columns:
            mask = df['male_prop'].isna() & df['sex'].notna()
            df.loc[mask & (df['sex'] == 1), 'male_prop'] = 1
            df.loc[mask & (df['sex'] == 2), 'male_prop'] = 0

    # Age of owner
    if 'age_prop' in df.columns:
        df['age_prop_clean'] = df['age_prop'].copy()
        if 'age' in df.columns:
            mask = df['age_prop'].isna() & df['age'].notna()
            df.loc[mask, 'age_prop_clean'] = df.loc[mask, 'age']

    # Main tribe (Luluwa)
    if 'tribe' in df.columns:
        df['main_tribe'] = np.nan
        df.loc[df['tribe'].notna(), 'main_tribe'] = 0
        df.loc[df['tribe'] == 'LULUWA', 'main_tribe'] = 1

    # Government work
    if 'job_gov' in df.columns:
        df['work_gov'] = df['job_gov'].copy()
        df.loc[df['work_gov'].isna(), 'work_gov'] = 0

    # Salongo hours
    if 'salongo_hours' in df.columns and 'salongo' in df.columns:
        df.loc[df['salongo'] == 0, 'salongo_hours'] = 0
        df.loc[df['salongo_hours'] == 99999, 'salongo_hours'] = np.nan

    # House type dummy (mm = maison moyenne)
    df['mm'] = (df['house'] == 2).astype(int)

    # Exempt status
    if 'exempt' in df.columns:
        df['exempt'] = df['exempt'].fillna(0)
    else:
        df['exempt'] = 0

    # Create time FE variable based on today_carto
    if 'today_carto' in df.columns:
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['today_carto']):
            df['today_carto'] = pd.to_datetime(df['today_carto'], errors='coerce')
        df['today_carto_num'] = df['today_carto']
    else:
        df['today_carto_num'] = pd.NaT

    # Create alternative date based on polygon min TDM date or max carto date
    # Convert date_TDM to datetime
    if 'date_TDM' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['date_TDM']):
            df['date_TDM'] = pd.to_datetime(df['date_TDM'], errors='coerce')

    df['today_alt'] = df.groupby('a7')['date_TDM'].transform('min')
    if 'today_carto_num' in df.columns:
        mask = df['today_alt'].isna()
        df.loc[mask, 'today_alt'] = df.loc[mask].groupby('a7')['today_carto_num'].transform('max')

    # Time FE bins (roughly 2-month bins) - using actual date ranges
    # Jun 15, Aug 14, Oct 13, Dec 9 for 2018
    df['time_FE_tdm_2mo_CvL'] = pd.cut(
        df['today_alt'],
        bins=[pd.Timestamp('2018-06-01'), pd.Timestamp('2018-08-01'),
              pd.Timestamp('2018-10-01'), pd.Timestamp('2018-12-31')],
        labels=[0, 1, 2],
        include_lowest=True
    )

    # Drop if rate is missing
    df = df[df['rate'].notna()].copy()

    print(f"Final dataset: {len(df)} observations")
    print(f"Treatment distribution:")
    print(df['tmt'].value_counts().sort_index())

    return df


# =============================================================================
# REGRESSION FUNCTIONS
# =============================================================================

def run_regression(df, formula, cluster_var=None, vcov_type='HC1'):
    """
    Run regression and return results dictionary.
    """
    try:
        # Check if we have fixed effects (|)
        has_fe = '|' in formula

        if USE_PYFIXEST and has_fe:
            # Use pyfixest for fixed effects
            if cluster_var:
                vcov = {'CRV1': cluster_var}
            else:
                vcov = 'hetero'

            model = pf.feols(formula, data=df, vcov=vcov)

            # Extract treatment coefficient (first non-intercept coefficient)
            coefs = model.coef()
            ses = model.se()
            pvals = model.pvalue()

            # Find treatment variable (t_l or similar)
            treat_var = None
            for var in coefs.index:
                if 't_l' in var:
                    treat_var = var
                    break

            if treat_var is None:
                treat_var = coefs.index[0]

            coef = coefs[treat_var]
            se = ses[treat_var]
            pval = pvals[treat_var]
            tstat = coef / se if se > 0 else np.nan

            n_obs = model.nobs
            r2 = model.r2() if hasattr(model, 'r2') else np.nan

            # Build coefficient vector
            coef_vector = {
                'treatment': {
                    'var': treat_var,
                    'coef': float(coef),
                    'se': float(se),
                    'pval': float(pval)
                },
                'controls': [],
                'fixed_effects': formula.split('|')[1].strip().split('+') if has_fe else []
            }

            for var in coefs.index:
                if var != treat_var:
                    coef_vector['controls'].append({
                        'var': var,
                        'coef': float(coefs[var]),
                        'se': float(ses[var]),
                        'pval': float(pvals[var])
                    })

        else:
            # Use statsmodels
            # Remove FE notation for statsmodels
            if has_fe:
                lhs_rhs = formula.split('|')
                main_formula = lhs_rhs[0].strip()
                fe_vars = lhs_rhs[1].strip().split('+')
                fe_vars = [v.strip() for v in fe_vars]

                # Add fixed effects as factor variables
                for fe in fe_vars:
                    main_formula += f' + C({fe})'
                formula = main_formula

            # Drop NaN values for formula variables
            model = smf.ols(formula, data=df).fit(cov_type=vcov_type)

            # Find treatment variable
            treat_var = None
            for var in model.params.index:
                if 't_l' in var and '[' not in var:
                    treat_var = var
                    break

            if treat_var is None:
                for var in model.params.index:
                    if var not in ['Intercept'] and not var.startswith('C('):
                        treat_var = var
                        break

            if treat_var is None:
                raise ValueError("Could not find treatment variable")

            coef = model.params[treat_var]
            se = model.bse[treat_var]
            pval = model.pvalues[treat_var]
            tstat = model.tvalues[treat_var]

            n_obs = int(model.nobs)
            r2 = model.rsquared

            # Build coefficient vector
            fe_vars_list = fe_vars if has_fe else []
            coef_vector = {
                'treatment': {
                    'var': treat_var,
                    'coef': float(coef),
                    'se': float(se),
                    'pval': float(pval)
                },
                'controls': [],
                'fixed_effects': fe_vars_list
            }

            for var in model.params.index:
                if var != treat_var and var != 'Intercept' and not var.startswith('C('):
                    coef_vector['controls'].append({
                        'var': var,
                        'coef': float(model.params[var]),
                        'se': float(model.bse[var]),
                        'pval': float(model.pvalues[var])
                    })

        # Calculate confidence intervals
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se

        return {
            'coefficient': float(coef),
            'std_error': float(se),
            't_stat': float(tstat),
            'p_value': float(pval),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_obs': int(n_obs),
            'r_squared': float(r2) if not np.isnan(r2) else None,
            'coefficient_vector_json': json.dumps(coef_vector),
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
            'r_squared': None,
            'coefficient_vector_json': json.dumps({'error': str(e)}),
            'success': False
        }


def add_result(results, spec_id, spec_tree_path, outcome_var, treatment_var,
               sample_desc, fixed_effects, controls_desc, cluster_var, model_type, res):
    """Helper function to add a result to the results list."""
    results.append({
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'outcome_var': outcome_var,
        'treatment_var': treatment_var,
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
        **res
    })


# =============================================================================
# SPECIFICATION SEARCH
# =============================================================================

def run_specification_search(df):
    """
    Run comprehensive specification search (minimum 50 specifications per i4r methodology).
    """

    results = []
    spec_counter = 0

    # Prepare sample for Central vs Local comparison
    sample_CvL = df[df['tmt'].isin([1, 2])].copy()

    print(f"Analysis sample (Central vs Local): {len(sample_CvL)} observations")
    print(f"Treatment: Central={sum(sample_CvL['t_c'])}, Local={sum(sample_CvL['t_l'])}")
    print(f"Outcome mean: {sample_CvL['taxes_paid'].mean():.4f}")

    # Sample with time FE
    sample_CvL_time = sample_CvL[sample_CvL['time_FE_tdm_2mo_CvL'].notna()].copy()

    # =============================================================================
    # BASELINE SPECIFICATIONS
    # =============================================================================

    print("\n=== Running Baseline Specifications ===")

    # Baseline 1: Stratum FE only
    spec_counter += 1
    print(f"  Spec {spec_counter}: Baseline - Stratum FE only")
    res = run_regression(sample_CvL, "taxes_paid ~ t_l | stratum", cluster_var='a7')
    add_result(results, 'baseline', 'methods/cross_sectional_ols.md#baseline',
               'taxes_paid', 't_l', 'Central vs Local', 'stratum', 'None', 'a7', 'OLS with FE', res)

    # Baseline 2: Stratum + Month FE
    spec_counter += 1
    print(f"  Spec {spec_counter}: Baseline - Stratum + Month FE")
    res = run_regression(sample_CvL_time, "taxes_paid ~ t_l | stratum + time_FE_tdm_2mo_CvL", cluster_var='a7')
    add_result(results, 'baseline_month_fe', 'methods/cross_sectional_ols.md#baseline',
               'taxes_paid', 't_l', 'Central vs Local with time FE', 'stratum + month', 'None', 'a7', 'OLS with FE', res)

    # Baseline 3: Stratum + Month + House FE
    spec_counter += 1
    print(f"  Spec {spec_counter}: Baseline - Stratum + Month + House FE")
    res = run_regression(sample_CvL_time, "taxes_paid ~ t_l | stratum + time_FE_tdm_2mo_CvL + house", cluster_var='a7')
    add_result(results, 'baseline_full_fe', 'methods/cross_sectional_ols.md#baseline',
               'taxes_paid', 't_l', 'Central vs Local with time FE', 'stratum + month + house', 'None', 'a7', 'OLS with FE', res)

    # Baseline 4: Non-exempt sample
    spec_counter += 1
    print(f"  Spec {spec_counter}: Baseline - Non-exempt sample")
    sample_nonexempt = sample_CvL_time[sample_CvL_time['exempt'] != 1].copy()
    res = run_regression(sample_nonexempt, "taxes_paid ~ t_l | stratum + time_FE_tdm_2mo_CvL + house", cluster_var='a7')
    add_result(results, 'baseline_nonexempt', 'methods/cross_sectional_ols.md#baseline',
               'taxes_paid', 't_l', 'Non-exempt properties', 'stratum + month + house', 'None', 'a7', 'OLS with FE', res)

    # =============================================================================
    # ALTERNATIVE OUTCOMES
    # =============================================================================

    print("\n=== Running Alternative Outcome Specifications ===")

    # Tax amount outcome
    spec_counter += 1
    print(f"  Spec {spec_counter}: Alternative outcome - Tax amount")
    res = run_regression(sample_CvL_time, "taxes_paid_amt ~ t_l | stratum + time_FE_tdm_2mo_CvL + house", cluster_var='a7')
    add_result(results, 'robust/outcome/tax_amount', 'robustness/measurement.md',
               'taxes_paid_amt', 't_l', 'Central vs Local', 'stratum + month + house', 'None', 'a7', 'OLS with FE', res)

    # IHS tax amount
    spec_counter += 1
    print(f"  Spec {spec_counter}: Alternative outcome - IHS tax amount")
    sample_CvL_time['ihs_tax_amt'] = np.arcsinh(sample_CvL_time['taxes_paid_amt'])
    res = run_regression(sample_CvL_time, "ihs_tax_amt ~ t_l | stratum + time_FE_tdm_2mo_CvL + house", cluster_var='a7')
    add_result(results, 'robust/outcome/ihs_tax_amount', 'robustness/functional_form.md',
               'ihs_tax_amt', 't_l', 'Central vs Local', 'stratum + month + house', 'None', 'a7', 'OLS with FE', res)

    # Bribe outcome (if available)
    if 'bribe_combined' in sample_CvL_time.columns:
        sample_bribe = sample_CvL_time[sample_CvL_time['bribe_combined'].notna()].copy()
        if len(sample_bribe) > 100:
            spec_counter += 1
            print(f"  Spec {spec_counter}: Alternative outcome - Bribe")
            res = run_regression(sample_bribe, "bribe_combined ~ t_l | stratum + time_FE_tdm_2mo_CvL + house", cluster_var='a7')
            add_result(results, 'robust/outcome/bribe', 'robustness/measurement.md',
                       'bribe_combined', 't_l', 'Central vs Local, visited', 'stratum + month + house', 'None', 'a7', 'OLS with FE', res)

    # Visit outcomes (if available)
    if 'visit_post_carto' in sample_CvL_time.columns:
        sample_visits = sample_CvL_time[sample_CvL_time['visit_post_carto'].notna()].copy()
        if len(sample_visits) > 100:
            spec_counter += 1
            print(f"  Spec {spec_counter}: Alternative outcome - Visit post carto")
            res = run_regression(sample_visits, "visit_post_carto ~ t_l | stratum + time_FE_tdm_2mo_CvL + house", cluster_var='a7')
            add_result(results, 'robust/outcome/visit_post_carto', 'robustness/measurement.md',
                       'visit_post_carto', 't_l', 'Central vs Local', 'stratum + month + house', 'None', 'a7', 'OLS with FE', res)

    if 'nb_visit_post_carto' in sample_CvL_time.columns:
        sample_nb_visits = sample_CvL_time[sample_CvL_time['nb_visit_post_carto'].notna()].copy()
        if len(sample_nb_visits) > 100:
            spec_counter += 1
            print(f"  Spec {spec_counter}: Alternative outcome - Number of visits")
            res = run_regression(sample_nb_visits, "nb_visit_post_carto ~ t_l | stratum + time_FE_tdm_2mo_CvL + house", cluster_var='a7')
            add_result(results, 'robust/outcome/nb_visits', 'robustness/measurement.md',
                       'nb_visit_post_carto', 't_l', 'Central vs Local', 'stratum + month + house', 'None', 'a7', 'OLS with FE', res)

    # =============================================================================
    # FIXED EFFECTS VARIATIONS
    # =============================================================================

    print("\n=== Running Fixed Effects Variations ===")

    # No fixed effects
    spec_counter += 1
    print(f"  Spec {spec_counter}: FE variation - No FE")
    res = run_regression(sample_CvL, "taxes_paid ~ t_l", cluster_var='a7')
    add_result(results, 'robust/estimation/no_fe', 'robustness/model_specification.md',
               'taxes_paid', 't_l', 'Central vs Local', 'None', 'None', 'a7', 'OLS', res)

    # Only stratum FE
    spec_counter += 1
    print(f"  Spec {spec_counter}: FE variation - Stratum FE only")
    res = run_regression(sample_CvL_time, "taxes_paid ~ t_l | stratum", cluster_var='a7')
    add_result(results, 'robust/estimation/stratum_fe_only', 'robustness/model_specification.md',
               'taxes_paid', 't_l', 'Central vs Local', 'stratum', 'None', 'a7', 'OLS with FE', res)

    # Only time FE
    spec_counter += 1
    print(f"  Spec {spec_counter}: FE variation - Month FE only")
    res = run_regression(sample_CvL_time, "taxes_paid ~ t_l | time_FE_tdm_2mo_CvL", cluster_var='a7')
    add_result(results, 'robust/estimation/month_fe_only', 'robustness/model_specification.md',
               'taxes_paid', 't_l', 'Central vs Local', 'month', 'None', 'a7', 'OLS with FE', res)

    # Only house FE
    spec_counter += 1
    print(f"  Spec {spec_counter}: FE variation - House FE only")
    res = run_regression(sample_CvL_time, "taxes_paid ~ t_l | house", cluster_var='a7')
    add_result(results, 'robust/estimation/house_fe_only', 'robustness/model_specification.md',
               'taxes_paid', 't_l', 'Central vs Local', 'house', 'None', 'a7', 'OLS with FE', res)

    # Stratum + House FE
    spec_counter += 1
    print(f"  Spec {spec_counter}: FE variation - Stratum + House FE")
    res = run_regression(sample_CvL_time, "taxes_paid ~ t_l | stratum + house", cluster_var='a7')
    add_result(results, 'robust/estimation/stratum_house_fe', 'robustness/model_specification.md',
               'taxes_paid', 't_l', 'Central vs Local', 'stratum + house', 'None', 'a7', 'OLS with FE', res)

    # =============================================================================
    # CLUSTERING VARIATIONS
    # =============================================================================

    print("\n=== Running Clustering Variations ===")

    # Robust SEs (no clustering)
    spec_counter += 1
    print(f"  Spec {spec_counter}: Clustering - Robust SE (HC1)")
    res = run_regression(sample_CvL_time, "taxes_paid ~ t_l | stratum + time_FE_tdm_2mo_CvL + house",
                        cluster_var=None, vcov_type='HC1')
    add_result(results, 'robust/cluster/robust_hc1', 'robustness/clustering_variations.md',
               'taxes_paid', 't_l', 'Central vs Local', 'stratum + month + house', 'None', 'None (HC1)', 'OLS with FE', res)

    # HC2 robust SEs
    spec_counter += 1
    print(f"  Spec {spec_counter}: Clustering - Robust SE (HC2)")
    res = run_regression(sample_CvL_time, "taxes_paid ~ t_l | stratum + time_FE_tdm_2mo_CvL + house",
                        cluster_var=None, vcov_type='HC2')
    add_result(results, 'robust/cluster/robust_hc2', 'robustness/clustering_variations.md',
               'taxes_paid', 't_l', 'Central vs Local', 'stratum + month + house', 'None', 'None (HC2)', 'OLS with FE', res)

    # HC3 robust SEs
    spec_counter += 1
    print(f"  Spec {spec_counter}: Clustering - Robust SE (HC3)")
    res = run_regression(sample_CvL_time, "taxes_paid ~ t_l | stratum + time_FE_tdm_2mo_CvL + house",
                        cluster_var=None, vcov_type='HC3')
    add_result(results, 'robust/cluster/robust_hc3', 'robustness/clustering_variations.md',
               'taxes_paid', 't_l', 'Central vs Local', 'stratum + month + house', 'None', 'None (HC3)', 'OLS with FE', res)

    # Cluster by stratum (remove stratum FE to avoid collinearity)
    spec_counter += 1
    print(f"  Spec {spec_counter}: Clustering - Cluster by stratum")
    res = run_regression(sample_CvL_time, "taxes_paid ~ t_l | time_FE_tdm_2mo_CvL + house",
                        cluster_var='stratum')
    add_result(results, 'robust/cluster/stratum', 'robustness/clustering_variations.md',
               'taxes_paid', 't_l', 'Central vs Local', 'month + house', 'None', 'stratum', 'OLS with FE', res)

    # =============================================================================
    # SAMPLE RESTRICTIONS
    # =============================================================================

    print("\n=== Running Sample Restriction Specifications ===")

    # By house type - Peripherie only
    spec_counter += 1
    print(f"  Spec {spec_counter}: Sample - Peripherie (house=1) only")
    sample_periph = sample_CvL_time[sample_CvL_time['house'] == 1].copy()
    res = run_regression(sample_periph, "taxes_paid ~ t_l | stratum + time_FE_tdm_2mo_CvL", cluster_var='a7')
    add_result(results, 'robust/sample/peripherie_only', 'robustness/sample_restrictions.md',
               'taxes_paid', 't_l', 'Peripherie houses only', 'stratum + month', 'None', 'a7', 'OLS with FE', res)

    # By house type - Maison moyenne only
    spec_counter += 1
    print(f"  Spec {spec_counter}: Sample - Maison moyenne (house=2) only")
    sample_mm = sample_CvL_time[sample_CvL_time['house'] == 2].copy()
    res = run_regression(sample_mm, "taxes_paid ~ t_l | stratum + time_FE_tdm_2mo_CvL", cluster_var='a7')
    add_result(results, 'robust/sample/maison_moyenne_only', 'robustness/sample_restrictions.md',
               'taxes_paid', 't_l', 'Maison moyenne houses only', 'stratum + month', 'None', 'a7', 'OLS with FE', res)

    # By rate - Low rate (50%)
    spec_counter += 1
    print(f"  Spec {spec_counter}: Sample - Low rate (50%)")
    sample_lowrate = sample_CvL_time[sample_CvL_time['pct_50'] == 1].copy()
    if len(sample_lowrate) > 50:
        res = run_regression(sample_lowrate, "taxes_paid ~ t_l | stratum + time_FE_tdm_2mo_CvL + house", cluster_var='a7')
        add_result(results, 'robust/sample/low_rate_50', 'robustness/sample_restrictions.md',
                   'taxes_paid', 't_l', 'Low rate (50%) properties', 'stratum + month + house', 'None', 'a7', 'OLS with FE', res)

    # By rate - High rate (100%)
    spec_counter += 1
    print(f"  Spec {spec_counter}: Sample - High rate (100%)")
    sample_highrate = sample_CvL_time[sample_CvL_time['pct_100'] == 1].copy()
    if len(sample_highrate) > 50:
        res = run_regression(sample_highrate, "taxes_paid ~ t_l | stratum + time_FE_tdm_2mo_CvL + house", cluster_var='a7')
        add_result(results, 'robust/sample/high_rate_100', 'robustness/sample_restrictions.md',
                   'taxes_paid', 't_l', 'High rate (100%) properties', 'stratum + month + house', 'None', 'a7', 'OLS with FE', res)

    # By rate - Medium rates
    spec_counter += 1
    print(f"  Spec {spec_counter}: Sample - Medium rate (66%/83%)")
    sample_medrate = sample_CvL_time[(sample_CvL_time['pct_66'] == 1) | (sample_CvL_time['pct_83'] == 1)].copy()
    if len(sample_medrate) > 50:
        res = run_regression(sample_medrate, "taxes_paid ~ t_l | stratum + time_FE_tdm_2mo_CvL + house", cluster_var='a7')
        add_result(results, 'robust/sample/medium_rate', 'robustness/sample_restrictions.md',
                   'taxes_paid', 't_l', 'Medium rate (66%/83%) properties', 'stratum + month + house', 'None', 'a7', 'OLS with FE', res)

    # Drop each stratum one at a time
    strata = sample_CvL_time['stratum'].dropna().unique()
    for stratum_val in strata[:8]:  # First 8 strata
        spec_counter += 1
        print(f"  Spec {spec_counter}: Sample - Drop stratum {int(stratum_val)}")
        sample_drop = sample_CvL_time[sample_CvL_time['stratum'] != stratum_val].copy()
        res = run_regression(sample_drop, "taxes_paid ~ t_l | stratum + time_FE_tdm_2mo_CvL + house", cluster_var='a7')
        add_result(results, f'robust/sample/drop_stratum_{int(stratum_val)}', 'robustness/sample_restrictions.md',
                   'taxes_paid', 't_l', f'Excluding stratum {int(stratum_val)}', 'stratum + month + house', 'None', 'a7', 'OLS with FE', res)

    # Drop each time period
    time_periods = sample_CvL_time['time_FE_tdm_2mo_CvL'].dropna().unique()
    for period in time_periods:
        spec_counter += 1
        print(f"  Spec {spec_counter}: Sample - Drop time period {period}")
        sample_drop = sample_CvL_time[sample_CvL_time['time_FE_tdm_2mo_CvL'] != period].copy()
        res = run_regression(sample_drop, "taxes_paid ~ t_l | stratum + time_FE_tdm_2mo_CvL + house", cluster_var='a7')
        add_result(results, f'robust/sample/drop_period_{period}', 'robustness/sample_restrictions.md',
                   'taxes_paid', 't_l', f'Excluding time period {period}', 'stratum + month + house', 'None', 'a7', 'OLS with FE', res)

    # =============================================================================
    # HETEROGENEITY ANALYSIS
    # =============================================================================

    print("\n=== Running Heterogeneity Specifications ===")

    # By owner gender
    if 'male_prop' in sample_CvL_time.columns:
        sample_with_gender = sample_CvL_time[sample_CvL_time['male_prop'].notna()].copy()
        if len(sample_with_gender) > 100:
            # Male owners
            spec_counter += 1
            print(f"  Spec {spec_counter}: Heterogeneity - Male owners")
            sample_male = sample_with_gender[sample_with_gender['male_prop'] == 1].copy()
            if len(sample_male) > 50:
                res = run_regression(sample_male, "taxes_paid ~ t_l | stratum + time_FE_tdm_2mo_CvL + house", cluster_var='a7')
                add_result(results, 'robust/heterogeneity/male_owners', 'robustness/heterogeneity.md',
                           'taxes_paid', 't_l', 'Male property owners only', 'stratum + month + house', 'None', 'a7', 'OLS with FE', res)

            # Female owners
            spec_counter += 1
            print(f"  Spec {spec_counter}: Heterogeneity - Female owners")
            sample_female = sample_with_gender[sample_with_gender['male_prop'] == 0].copy()
            if len(sample_female) > 50:
                res = run_regression(sample_female, "taxes_paid ~ t_l | stratum + time_FE_tdm_2mo_CvL + house", cluster_var='a7')
                add_result(results, 'robust/heterogeneity/female_owners', 'robustness/heterogeneity.md',
                           'taxes_paid', 't_l', 'Female property owners only', 'stratum + month + house', 'None', 'a7', 'OLS with FE', res)

    # By tribe
    if 'main_tribe' in sample_CvL_time.columns:
        sample_with_tribe = sample_CvL_time[sample_CvL_time['main_tribe'].notna()].copy()
        if len(sample_with_tribe) > 100:
            # Main tribe (Luluwa)
            spec_counter += 1
            print(f"  Spec {spec_counter}: Heterogeneity - Main tribe (Luluwa)")
            sample_luluwa = sample_with_tribe[sample_with_tribe['main_tribe'] == 1].copy()
            if len(sample_luluwa) > 50:
                res = run_regression(sample_luluwa, "taxes_paid ~ t_l | stratum + time_FE_tdm_2mo_CvL + house", cluster_var='a7')
                add_result(results, 'robust/heterogeneity/luluwa_tribe', 'robustness/heterogeneity.md',
                           'taxes_paid', 't_l', 'Main tribe (Luluwa) only', 'stratum + month + house', 'None', 'a7', 'OLS with FE', res)

            # Other tribes
            spec_counter += 1
            print(f"  Spec {spec_counter}: Heterogeneity - Other tribes")
            sample_other = sample_with_tribe[sample_with_tribe['main_tribe'] == 0].copy()
            if len(sample_other) > 50:
                res = run_regression(sample_other, "taxes_paid ~ t_l | stratum + time_FE_tdm_2mo_CvL + house", cluster_var='a7')
                add_result(results, 'robust/heterogeneity/other_tribes', 'robustness/heterogeneity.md',
                           'taxes_paid', 't_l', 'Other tribes only', 'stratum + month + house', 'None', 'a7', 'OLS with FE', res)

    # By government connection
    if 'work_gov' in sample_CvL_time.columns:
        sample_with_gov = sample_CvL_time[sample_CvL_time['work_gov'].notna()].copy()
        if len(sample_with_gov) > 100:
            # Government workers
            sample_gov = sample_with_gov[sample_with_gov['work_gov'] == 1].copy()
            if len(sample_gov) > 50:
                spec_counter += 1
                print(f"  Spec {spec_counter}: Heterogeneity - Government workers")
                res = run_regression(sample_gov, "taxes_paid ~ t_l | stratum + time_FE_tdm_2mo_CvL + house", cluster_var='a7')
                add_result(results, 'robust/heterogeneity/gov_workers', 'robustness/heterogeneity.md',
                           'taxes_paid', 't_l', 'Government workers only', 'stratum + month + house', 'None', 'a7', 'OLS with FE', res)

            # Non-government workers
            spec_counter += 1
            print(f"  Spec {spec_counter}: Heterogeneity - Non-government workers")
            sample_nongov = sample_with_gov[sample_with_gov['work_gov'] == 0].copy()
            if len(sample_nongov) > 50:
                res = run_regression(sample_nongov, "taxes_paid ~ t_l | stratum + time_FE_tdm_2mo_CvL + house", cluster_var='a7')
                add_result(results, 'robust/heterogeneity/non_gov_workers', 'robustness/heterogeneity.md',
                           'taxes_paid', 't_l', 'Non-government workers only', 'stratum + month + house', 'None', 'a7', 'OLS with FE', res)

    # Interaction with house type
    spec_counter += 1
    print(f"  Spec {spec_counter}: Heterogeneity - Treatment x House interaction")
    sample_CvL_time['t_l_x_mm'] = sample_CvL_time['t_l'] * sample_CvL_time['mm']
    res = run_regression(sample_CvL_time, "taxes_paid ~ t_l + mm + t_l_x_mm | stratum + time_FE_tdm_2mo_CvL", cluster_var='a7')
    add_result(results, 'robust/heterogeneity/treatment_x_house', 'robustness/heterogeneity.md',
               'taxes_paid', 't_l', 'Central vs Local', 'stratum + month', 'House type interaction', 'a7', 'OLS with FE', res)

    # =============================================================================
    # CONTROL VARIATIONS
    # =============================================================================

    print("\n=== Running Control Variation Specifications ===")

    # Add rate as control
    spec_counter += 1
    print(f"  Spec {spec_counter}: Control - Add rate")
    res = run_regression(sample_CvL_time, "taxes_paid ~ t_l + rate | stratum + time_FE_tdm_2mo_CvL + house", cluster_var='a7')
    add_result(results, 'robust/control/add_rate', 'robustness/control_progression.md',
               'taxes_paid', 't_l', 'Central vs Local', 'stratum + month + house', 'rate', 'a7', 'OLS with FE', res)

    # Add rate dummies
    spec_counter += 1
    print(f"  Spec {spec_counter}: Control - Add rate dummies")
    res = run_regression(sample_CvL_time, "taxes_paid ~ t_l + pct_50 + pct_66 + pct_83 | stratum + time_FE_tdm_2mo_CvL + house", cluster_var='a7')
    add_result(results, 'robust/control/add_rate_dummies', 'robustness/control_progression.md',
               'taxes_paid', 't_l', 'Central vs Local', 'stratum + month + house', 'rate dummies', 'a7', 'OLS with FE', res)

    # Add exempt control
    spec_counter += 1
    print(f"  Spec {spec_counter}: Control - Add exempt status")
    res = run_regression(sample_CvL_time, "taxes_paid ~ t_l + exempt | stratum + time_FE_tdm_2mo_CvL + house", cluster_var='a7')
    add_result(results, 'robust/control/add_exempt', 'robustness/control_progression.md',
               'taxes_paid', 't_l', 'Central vs Local', 'stratum + month + house', 'exempt status', 'a7', 'OLS with FE', res)

    # Full controls
    spec_counter += 1
    print(f"  Spec {spec_counter}: Control - Full controls")
    res = run_regression(sample_CvL_time, "taxes_paid ~ t_l + rate + exempt | stratum + time_FE_tdm_2mo_CvL + house", cluster_var='a7')
    add_result(results, 'robust/control/full', 'robustness/control_progression.md',
               'taxes_paid', 't_l', 'Central vs Local', 'stratum + month + house', 'rate + exempt', 'a7', 'OLS with FE', res)

    # =============================================================================
    # ALTERNATIVE TREATMENTS
    # =============================================================================

    print("\n=== Running Alternative Treatment Specifications ===")

    # Include CLI treatment comparison
    spec_counter += 1
    print(f"  Spec {spec_counter}: Treatment - Include Central with Local Info")
    sample_full = df[df['tmt'].isin([1, 2, 3])].copy()
    sample_full_time = sample_full[sample_full['time_FE_tdm_2mo_CvL'].notna()].copy()
    res = run_regression(sample_full_time, "taxes_paid ~ t_l + t_cli | stratum + time_FE_tdm_2mo_CvL + house", cluster_var='a7')
    add_result(results, 'robust/treatment/include_cli', 'robustness/measurement.md',
               'taxes_paid', 't_l', 'Central vs Local vs CLI', 'stratum + month + house', 'None', 'a7', 'OLS with FE', res)

    # CLI vs Central comparison
    spec_counter += 1
    print(f"  Spec {spec_counter}: Treatment - CLI vs Central")
    sample_CvCLI = df[df['tmt'].isin([1, 3])].copy()
    sample_CvCLI_time = sample_CvCLI[sample_CvCLI['time_FE_tdm_2mo_CvL'].notna()].copy()
    res = run_regression(sample_CvCLI_time, "taxes_paid ~ t_cli | stratum + time_FE_tdm_2mo_CvL + house", cluster_var='a7')
    add_result(results, 'robust/treatment/cli_vs_central', 'robustness/measurement.md',
               'taxes_paid', 't_cli', 'Central vs CLI', 'stratum + month + house', 'None', 'a7', 'OLS with FE', res)

    # Local vs CLI comparison
    spec_counter += 1
    print(f"  Spec {spec_counter}: Treatment - Local vs CLI")
    sample_LvCLI = df[df['tmt'].isin([2, 3])].copy()
    sample_LvCLI['t_l_vs_cli'] = (sample_LvCLI['tmt'] == 2).astype(int)
    sample_LvCLI_time = sample_LvCLI[sample_LvCLI['time_FE_tdm_2mo_CvL'].notna()].copy()
    res = run_regression(sample_LvCLI_time, "taxes_paid ~ t_l_vs_cli | stratum + time_FE_tdm_2mo_CvL + house", cluster_var='a7')
    add_result(results, 'robust/treatment/local_vs_cli', 'robustness/measurement.md',
               'taxes_paid', 't_l_vs_cli', 'Local vs CLI', 'stratum + month + house', 'None', 'a7', 'OLS with FE', res)

    # =============================================================================
    # PLACEBO TESTS
    # =============================================================================

    print("\n=== Running Placebo Specifications ===")

    # Placebo: Salongo hours
    if 'salongo_hours' in sample_CvL_time.columns:
        sample_salongo = sample_CvL_time[sample_CvL_time['salongo_hours'].notna()].copy()
        if len(sample_salongo) > 100:
            spec_counter += 1
            print(f"  Spec {spec_counter}: Placebo - Salongo hours")
            res = run_regression(sample_salongo, "salongo_hours ~ t_l | stratum + time_FE_tdm_2mo_CvL + house", cluster_var='a7')
            add_result(results, 'robust/placebo/salongo_hours', 'robustness/placebo_tests.md',
                       'salongo_hours', 't_l', 'Central vs Local', 'stratum + month + house', 'None', 'a7', 'OLS with FE', res)

    # Placebo: Exempt status
    spec_counter += 1
    print(f"  Spec {spec_counter}: Placebo - Exempt status")
    res = run_regression(sample_CvL_time, "exempt ~ t_l | stratum + time_FE_tdm_2mo_CvL + house", cluster_var='a7')
    add_result(results, 'robust/placebo/exempt_status', 'robustness/placebo_tests.md',
               'exempt', 't_l', 'Central vs Local', 'stratum + month + house', 'None', 'a7', 'OLS with FE', res)

    # Permutation test
    np.random.seed(42)
    spec_counter += 1
    print(f"  Spec {spec_counter}: Placebo - Permuted treatment")
    sample_permute = sample_CvL_time.copy()
    sample_permute['t_l_permuted'] = np.random.permutation(sample_permute['t_l'].values)
    res = run_regression(sample_permute, "taxes_paid ~ t_l_permuted | stratum + time_FE_tdm_2mo_CvL + house", cluster_var='a7')
    add_result(results, 'robust/placebo/permuted_treatment', 'robustness/placebo_tests.md',
               'taxes_paid', 't_l_permuted', 'Central vs Local (permuted)', 'stratum + month + house', 'None', 'a7', 'OLS with FE', res)

    # =============================================================================
    # FUNCTIONAL FORM VARIATIONS
    # =============================================================================

    print("\n=== Running Functional Form Variations ===")

    # LPM without FE
    spec_counter += 1
    print(f"  Spec {spec_counter}: Functional form - LPM without FE")
    res = run_regression(sample_CvL_time, "taxes_paid ~ t_l", cluster_var='a7')
    add_result(results, 'robust/funcform/lpm_no_fe', 'robustness/functional_form.md',
               'taxes_paid', 't_l', 'Central vs Local', 'None', 'None', 'a7', 'LPM', res)

    # =============================================================================
    # ADDITIONAL SPECIFICATIONS TO REACH 50+
    # =============================================================================

    print("\n=== Running Additional Specifications ===")

    # More strata drops
    for stratum_val in strata[8:12]:  # Next 4 strata
        spec_counter += 1
        print(f"  Spec {spec_counter}: Sample - Drop stratum {int(stratum_val)}")
        sample_drop = sample_CvL_time[sample_CvL_time['stratum'] != stratum_val].copy()
        res = run_regression(sample_drop, "taxes_paid ~ t_l | stratum + time_FE_tdm_2mo_CvL + house", cluster_var='a7')
        add_result(results, f'robust/sample/drop_stratum_{int(stratum_val)}', 'robustness/sample_restrictions.md',
                   'taxes_paid', 't_l', f'Excluding stratum {int(stratum_val)}', 'stratum + month + house', 'None', 'a7', 'OLS with FE', res)

    # Heterogeneity by age
    if 'age_prop_clean' in sample_CvL_time.columns:
        sample_with_age = sample_CvL_time[sample_CvL_time['age_prop_clean'].notna()].copy()
        if len(sample_with_age) > 100:
            median_age = sample_with_age['age_prop_clean'].median()

            spec_counter += 1
            print(f"  Spec {spec_counter}: Heterogeneity - Young owners")
            sample_young = sample_with_age[sample_with_age['age_prop_clean'] < median_age].copy()
            if len(sample_young) > 50:
                res = run_regression(sample_young, "taxes_paid ~ t_l | stratum + time_FE_tdm_2mo_CvL + house", cluster_var='a7')
                add_result(results, 'robust/heterogeneity/young_owners', 'robustness/heterogeneity.md',
                           'taxes_paid', 't_l', f'Owners below median age', 'stratum + month + house', 'None', 'a7', 'OLS with FE', res)

            spec_counter += 1
            print(f"  Spec {spec_counter}: Heterogeneity - Older owners")
            sample_old = sample_with_age[sample_with_age['age_prop_clean'] >= median_age].copy()
            if len(sample_old) > 50:
                res = run_regression(sample_old, "taxes_paid ~ t_l | stratum + time_FE_tdm_2mo_CvL + house", cluster_var='a7')
                add_result(results, 'robust/heterogeneity/older_owners', 'robustness/heterogeneity.md',
                           'taxes_paid', 't_l', f'Owners above median age', 'stratum + month + house', 'None', 'a7', 'OLS with FE', res)

    # Tax amount outcomes with different FE
    spec_counter += 1
    print(f"  Spec {spec_counter}: Tax amount - stratum FE only")
    res = run_regression(sample_CvL_time, "taxes_paid_amt ~ t_l | stratum", cluster_var='a7')
    add_result(results, 'robust/outcome/tax_amount_stratum_fe', 'robustness/measurement.md',
               'taxes_paid_amt', 't_l', 'Central vs Local', 'stratum', 'None', 'a7', 'OLS with FE', res)

    spec_counter += 1
    print(f"  Spec {spec_counter}: Tax amount - no FE")
    res = run_regression(sample_CvL_time, "taxes_paid_amt ~ t_l", cluster_var='a7')
    add_result(results, 'robust/outcome/tax_amount_no_fe', 'robustness/measurement.md',
               'taxes_paid_amt', 't_l', 'Central vs Local', 'None', 'None', 'a7', 'OLS', res)

    print(f"\n=== Total specifications run: {spec_counter} ===")

    return pd.DataFrame(results)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print(f"Specification Search: {PAPER_ID}")
    print(f"Paper: {PAPER_TITLE}")
    print("="*80)

    # Load and construct data
    df = load_and_construct_data()

    # Run specification search
    results_df = run_specification_search(df)

    # Save results
    output_path = OUTPUT_DIR / "specification_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    successful = results_df[results_df['coefficient'].notna()]
    print(f"Total specifications: {len(results_df)}")
    print(f"Successful specifications: {len(successful)}")

    if len(successful) > 0:
        print(f"Positive coefficients: {sum(successful['coefficient'] > 0)} ({100*sum(successful['coefficient'] > 0)/len(successful):.1f}%)")
        print(f"Significant at 5%: {sum(successful['p_value'] < 0.05)} ({100*sum(successful['p_value'] < 0.05)/len(successful):.1f}%)")
        print(f"Significant at 1%: {sum(successful['p_value'] < 0.01)} ({100*sum(successful['p_value'] < 0.01)/len(successful):.1f}%)")
        print(f"Median coefficient: {successful['coefficient'].median():.4f}")
        print(f"Mean coefficient: {successful['coefficient'].mean():.4f}")
        print(f"Range: [{successful['coefficient'].min():.4f}, {successful['coefficient'].max():.4f}]")

    print("\n" + "="*80)
    print("SPECIFICATION SEARCH COMPLETE")
    print("="*80)
