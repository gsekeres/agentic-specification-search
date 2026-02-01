#!/usr/bin/env python3
"""
Specification Search: 158081-V7
Paper: Work from Home before and after the COVID-19 Outbreak
Authors: Bick, Blandin, and Mertens (2023), AEJ: Macroeconomics
Data: Real-Time Population Survey (RPS)

This script runs a systematic specification search on the RPS data,
analyzing the determinants of work-from-home behavior.
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path('/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/158081-V7')
OUTPUT_DIR = DATA_DIR

def load_and_prepare_data():
    """Load RPS data and prepare variables for analysis."""

    df = pd.read_csv(DATA_DIR / 'rps_data_release_v5.2.csv', low_memory=False)

    # Filter to employed individuals in reference week (not February 2020)
    # February 2020 observations are recalls, not contemporaneous
    df = df[df['emp_simple'] == 'Employed'].copy()
    df = df[~df['month_running'].str.contains('February 2020', na=False)].copy()

    # Convert numeric columns
    df['days_working'] = pd.to_numeric(df['days_working'], errors='coerce')
    df['days_commuting'] = pd.to_numeric(df['days_commuting'], errors='coerce')
    df['uhrs'] = pd.to_numeric(df['uhrs'], errors='coerce')
    df['ahrs'] = pd.to_numeric(df['ahrs'], errors='coerce')
    df['wearn'] = pd.to_numeric(df['wearn'], errors='coerce')
    df['wage'] = pd.to_numeric(df['wage'], errors='coerce')

    # Create WFH indicators
    # Primary outcome: days working from home (days_working - days_commuting)
    df['days_wfh'] = df['days_working'] - df['days_commuting']
    df['days_wfh'] = df['days_wfh'].clip(lower=0)  # Can't be negative

    # WFH share (fraction of work days at home)
    df['wfh_share'] = df['days_wfh'] / df['days_working']
    df['wfh_share'] = df['wfh_share'].replace([np.inf, -np.inf], np.nan)
    df['wfh_share'] = df['wfh_share'].fillna(0)

    # Binary WFH indicator (any WFH)
    df['any_wfh'] = (df['days_wfh'] > 0).astype(float)

    # Full WFH indicator (all days at home)
    df['full_wfh'] = (df['days_commuting'] == 0).astype(float)

    # Create time variables
    df['post_covid'] = (df['year'] >= 2020) & (df['month'] >= 4) | (df['year'] > 2020)
    df['post_covid'] = df['post_covid'].astype(float)

    # Create year-month indicator
    df['year_month'] = df['year'].astype(str) + '_' + df['month'].astype(str).str.zfill(2)

    # Create demographic dummies
    df['female'] = (df['sex'] == 'Female').astype(float)
    df['male'] = (df['sex'] == 'Male').astype(float)

    # Education categories
    df['educ_lths'] = df['educ'].str.contains('Did not complete', na=False).astype(float)
    df['educ_hs'] = df['educ'].str.contains('High school graduate', na=False).astype(float)
    df['educ_somecol'] = df['educ'].str.contains('Some college', na=False).astype(float)
    df['educ_assoc'] = df['educ'].str.contains("Associate", na=False).astype(float)
    df['educ_ba'] = df['educ'].str.contains("Bachelor", na=False).astype(float)
    df['educ_grad'] = df['educ'].str.contains('Graduate degree', na=False).astype(float)
    df['college'] = ((df['educ_ba'] == 1) | (df['educ_grad'] == 1)).astype(float)

    # Race/ethnicity
    df['white'] = (df['race_comb'] == 'Non-hispanic white').astype(float)
    df['black'] = (df['race_comb'] == 'Non-hispanic black').astype(float)
    df['hispanic'] = (df['race_comb'] == 'Hispanic').astype(float)
    df['other_race'] = (df['race_comb'] == 'Other').astype(float)

    # Marital status
    df['married'] = df['marst'].str.contains('Married', na=False).astype(float)

    # Children
    df['has_children'] = (df['nchild'] != 'No children').astype(float)

    # Age groups
    df['age_18_24'] = ((df['age'] >= 18) & (df['age'] <= 24)).astype(float)
    df['age_25_34'] = ((df['age'] >= 25) & (df['age'] <= 34)).astype(float)
    df['age_35_44'] = ((df['age'] >= 35) & (df['age'] <= 44)).astype(float)
    df['age_45_54'] = ((df['age'] >= 45) & (df['age'] <= 54)).astype(float)
    df['age_55_64'] = ((df['age'] >= 55) & (df['age'] <= 64)).astype(float)
    df['age_sq'] = df['age'] ** 2

    # Industry categories
    industry_map = {
        'Information Services': 'teleworkable',
        'Banking, Finance, or Insurance': 'teleworkable',
        'Professional, Technical, or Business Services': 'teleworkable',
        'Education': 'teleworkable',
        'Government, including Military': 'teleworkable',
        'Health Care and Social Assistance': 'essential',
        'Retail Trade': 'essential',
        'Transportation and Warehousing': 'essential',
        'Construction': 'essential',
        'Manufacturing': 'essential',
    }
    df['teleworkable_ind'] = df['ind18'].map(lambda x: 1.0 if 'Information' in str(x) or 'Banking' in str(x) or
                                              'Professional' in str(x) or 'Education' in str(x) else 0.0)

    # Employment type
    df['self_employed'] = (df['emptype'] == 'Self-employed').astype(float)
    df['govt_emp'] = df['emptype'].str.contains('Government', na=False).astype(float)
    df['private_emp'] = df['emptype'].str.contains('Private-sector', na=False).astype(float)

    # State fixed effects - encode as numeric
    df['state_fe'] = pd.Categorical(df['statefip']).codes

    # Wave fixed effects
    df['wave_fe'] = pd.Categorical(df['wave']).codes

    # Filter to observations with valid WFH data
    df = df[df['days_working'].notna() & df['days_commuting'].notna()].copy()

    return df


def run_regression(df, formula, outcome_var, treatment_var, spec_id, spec_tree_path,
                   weights=None, cluster_var=None, sample_desc='Full sample'):
    """Run a single OLS regression and return results dictionary."""

    # Filter to complete cases for the formula
    try:
        y_var = formula.split('~')[0].strip()
        model_df = df.dropna(subset=[y_var]).copy()

        if weights is not None:
            model_df = model_df[model_df[weights].notna()].copy()
            wts = model_df[weights]
        else:
            wts = None

        if len(model_df) < 30:
            return None

        # Fit model
        if wts is not None:
            model = smf.wls(formula, data=model_df, weights=wts).fit(cov_type='HC1')
        else:
            model = smf.ols(formula, data=model_df).fit(cov_type='HC1')

        # Extract treatment coefficient
        if treatment_var in model.params.index:
            coef = model.params[treatment_var]
            se = model.bse[treatment_var]
            tstat = model.tvalues[treatment_var]
            pval = model.pvalues[treatment_var]
        else:
            return None

        # 95% CI
        ci = model.conf_int(alpha=0.05)
        if treatment_var in ci.index:
            ci_lower = ci.loc[treatment_var, 0]
            ci_upper = ci.loc[treatment_var, 1]
        else:
            ci_lower = ci_upper = np.nan

        # Build coefficient vector
        coef_vector = {
            'treatment': {
                'var': treatment_var,
                'coef': float(coef),
                'se': float(se),
                'pval': float(pval)
            },
            'controls': [],
            'n_obs': int(model.nobs),
            'r_squared': float(model.rsquared),
            'adj_r_squared': float(model.rsquared_adj) if hasattr(model, 'rsquared_adj') else None
        }

        # Add control coefficients
        for var in model.params.index:
            if var != treatment_var and var != 'Intercept':
                coef_vector['controls'].append({
                    'var': var,
                    'coef': float(model.params[var]),
                    'se': float(model.bse[var]),
                    'pval': float(model.pvalues[var])
                })

        result = {
            'paper_id': '158081-V7',
            'journal': 'AEJ: Macro',
            'paper_title': 'Work from Home before and after the COVID-19 Outbreak',
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': float(coef),
            'std_error': float(se),
            't_stat': float(tstat),
            'p_value': float(pval),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_obs': int(model.nobs),
            'r_squared': float(model.rsquared),
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc,
            'fixed_effects': 'None' if 'wave' not in formula else 'Wave',
            'controls_desc': formula.split('~')[1].strip(),
            'cluster_var': cluster_var if cluster_var else 'None',
            'model_type': 'WLS' if weights else 'OLS',
            'estimation_script': 'scripts/paper_analyses/158081-V7.py'
        }

        return result

    except Exception as e:
        print(f"Error in {spec_id}: {str(e)}")
        return None


def run_specification_search(df):
    """Run the full specification search."""

    results = []

    # Define treatment variables
    treatment_vars = ['college', 'female', 'teleworkable_ind', 'self_employed']

    # Define outcome variables
    outcome_vars = ['wfh_share', 'any_wfh', 'days_wfh']

    # Define control sets
    demographic_controls = 'female + age + age_sq + married + has_children'
    education_controls = 'educ_hs + educ_somecol + educ_assoc + educ_ba + educ_grad'
    race_controls = 'white + black + hispanic'
    full_controls = f'{demographic_controls} + {education_controls} + {race_controls}'

    # ===================
    # BASELINE SPECIFICATIONS
    # ===================
    print("Running baseline specifications...")

    # Main outcome: WFH share by education (college premium)
    # This captures the key finding that college workers have higher WFH rates

    for outcome in outcome_vars:
        # Baseline: College effect on WFH with demographic controls
        formula = f'{outcome} ~ college + age + age_sq + female + married + has_children + white + black + hispanic'
        res = run_regression(df, formula, outcome, 'college', 'baseline',
                           'methods/cross_sectional_ols.md#baseline',
                           weights='wgt', sample_desc='Employed workers, survey weeks')
        if res:
            results.append(res)

    # ===================
    # OLS CORE VARIATIONS
    # ===================
    print("Running OLS core variations...")

    for outcome in ['wfh_share', 'any_wfh']:
        treatment = 'college'

        # Control set variations
        # No controls (bivariate)
        formula = f'{outcome} ~ {treatment}'
        res = run_regression(df, formula, outcome, treatment,
                           f'ols/controls/none_{outcome}',
                           'methods/cross_sectional_ols.md#control-sets',
                           weights='wgt')
        if res:
            results.append(res)

        # Demographics only
        formula = f'{outcome} ~ {treatment} + age + age_sq + female + married'
        res = run_regression(df, formula, outcome, treatment,
                           f'ols/controls/demographics_{outcome}',
                           'methods/cross_sectional_ols.md#control-sets',
                           weights='wgt')
        if res:
            results.append(res)

        # Full controls
        formula = f'{outcome} ~ {treatment} + {full_controls}'
        res = run_regression(df, formula, outcome, treatment,
                           f'ols/controls/full_{outcome}',
                           'methods/cross_sectional_ols.md#control-sets',
                           weights='wgt')
        if res:
            results.append(res)

        # Unweighted
        formula = f'{outcome} ~ {treatment} + age + age_sq + female + married + has_children'
        res = run_regression(df, formula, outcome, treatment,
                           f'ols/method/unweighted_{outcome}',
                           'methods/cross_sectional_ols.md#estimation-method',
                           weights=None)
        if res:
            results.append(res)

    # ===================
    # SAMPLE RESTRICTIONS
    # ===================
    print("Running sample restrictions...")

    for outcome in ['wfh_share', 'any_wfh']:
        treatment = 'college'
        base_formula = f'{outcome} ~ {treatment} + age + age_sq + female + married + has_children'

        # By gender
        for gender, label in [(1, 'female'), (0, 'male')]:
            df_sub = df[df['female'] == gender].copy()
            formula = f'{outcome} ~ {treatment} + age + age_sq + married + has_children'
            res = run_regression(df_sub, formula, outcome, treatment,
                               f'ols/sample/subgroup_{label}_{outcome}',
                               'methods/cross_sectional_ols.md#sample-restrictions',
                               weights='wgt', sample_desc=f'{label.capitalize()} workers only')
            if res:
                results.append(res)

        # By age group
        for age_group, age_min, age_max in [('young', 18, 35), ('prime', 35, 55), ('older', 55, 65)]:
            df_sub = df[(df['age'] >= age_min) & (df['age'] < age_max)].copy()
            res = run_regression(df_sub, base_formula, outcome, treatment,
                               f'ols/sample/subgroup_{age_group}_{outcome}',
                               'methods/cross_sectional_ols.md#sample-restrictions',
                               weights='wgt', sample_desc=f'Age {age_min}-{age_max}')
            if res:
                results.append(res)

        # By time period
        # 2020 only (COVID shock year)
        df_2020 = df[df['year'] == 2020].copy()
        res = run_regression(df_2020, base_formula, outcome, treatment,
                           f'ols/sample/year_2020_{outcome}',
                           'methods/cross_sectional_ols.md#sample-restrictions',
                           weights='wgt', sample_desc='2020 only')
        if res:
            results.append(res)

        # 2021 only
        df_2021 = df[df['year'] == 2021].copy()
        if len(df_2021) > 100:
            res = run_regression(df_2021, base_formula, outcome, treatment,
                               f'ols/sample/year_2021_{outcome}',
                               'methods/cross_sectional_ols.md#sample-restrictions',
                               weights='wgt', sample_desc='2021 only')
            if res:
                results.append(res)

        # 2022+ (post-acute COVID)
        df_post = df[df['year'] >= 2022].copy()
        if len(df_post) > 100:
            res = run_regression(df_post, base_formula, outcome, treatment,
                               f'ols/sample/year_2022plus_{outcome}',
                               'methods/cross_sectional_ols.md#sample-restrictions',
                               weights='wgt', sample_desc='2022 onwards')
            if res:
                results.append(res)

        # By employment type
        for emp_type, label in [(1, 'self_employed'), (0, 'wage_workers')]:
            df_sub = df[df['self_employed'] == emp_type].copy()
            formula = f'{outcome} ~ {treatment} + age + age_sq + female + married + has_children'
            res = run_regression(df_sub, formula, outcome, treatment,
                               f'ols/sample/subgroup_{label}_{outcome}',
                               'methods/cross_sectional_ols.md#sample-restrictions',
                               weights='wgt', sample_desc=f'{label}')
            if res:
                results.append(res)

    # ===================
    # ALTERNATIVE TREATMENTS
    # ===================
    print("Running alternative treatment specifications...")

    for outcome in ['wfh_share', 'any_wfh']:
        # Gender as treatment
        formula = f'{outcome} ~ female + age + age_sq + college + married + has_children'
        res = run_regression(df, formula, outcome, 'female',
                           f'ols/treatment/female_{outcome}',
                           'methods/cross_sectional_ols.md#baseline',
                           weights='wgt')
        if res:
            results.append(res)

        # Teleworkable industry as treatment
        formula = f'{outcome} ~ teleworkable_ind + age + age_sq + female + college + married'
        res = run_regression(df, formula, outcome, 'teleworkable_ind',
                           f'ols/treatment/teleworkable_{outcome}',
                           'methods/cross_sectional_ols.md#baseline',
                           weights='wgt')
        if res:
            results.append(res)

        # Self-employment as treatment
        formula = f'{outcome} ~ self_employed + age + age_sq + female + college + married'
        res = run_regression(df, formula, outcome, 'self_employed',
                           f'ols/treatment/self_employed_{outcome}',
                           'methods/cross_sectional_ols.md#baseline',
                           weights='wgt')
        if res:
            results.append(res)

        # Age (continuous) as treatment
        formula = f'{outcome} ~ age + age_sq + female + college + married + has_children'
        res = run_regression(df, formula, outcome, 'age',
                           f'ols/treatment/age_{outcome}',
                           'methods/cross_sectional_ols.md#baseline',
                           weights='wgt')
        if res:
            results.append(res)

    # ===================
    # LEAVE-ONE-OUT ROBUSTNESS
    # ===================
    print("Running leave-one-out specifications...")

    controls = ['age', 'age_sq', 'female', 'married', 'has_children', 'white', 'black', 'hispanic']

    for outcome in ['wfh_share', 'any_wfh']:
        treatment = 'college'

        for drop_var in controls:
            remaining = [c for c in controls if c != drop_var]
            formula = f'{outcome} ~ {treatment} + {" + ".join(remaining)}'
            res = run_regression(df, formula, outcome, treatment,
                               f'robust/loo/drop_{drop_var}_{outcome}',
                               'robustness/leave_one_out.md',
                               weights='wgt', sample_desc=f'Dropping {drop_var}')
            if res:
                results.append(res)

    # ===================
    # SINGLE COVARIATE ROBUSTNESS
    # ===================
    print("Running single covariate specifications...")

    for outcome in ['wfh_share', 'any_wfh']:
        treatment = 'college'

        for single_var in controls:
            formula = f'{outcome} ~ {treatment} + {single_var}'
            res = run_regression(df, formula, outcome, treatment,
                               f'robust/single/{single_var}_{outcome}',
                               'robustness/single_covariate.md',
                               weights='wgt', sample_desc=f'Single control: {single_var}')
            if res:
                results.append(res)

    # ===================
    # FUNCTIONAL FORM VARIATIONS
    # ===================
    print("Running functional form specifications...")

    # Log outcome (days_wfh + 1)
    df['log_days_wfh'] = np.log(df['days_wfh'] + 1)
    treatment = 'college'
    formula = f'log_days_wfh ~ {treatment} + age + age_sq + female + married + has_children'
    res = run_regression(df, formula, 'log_days_wfh', treatment,
                       'ols/form/log_dep',
                       'methods/cross_sectional_ols.md#functional-form',
                       weights='wgt')
    if res:
        results.append(res)

    # Quadratic in age (no age_sq in formula, just age)
    for outcome in ['wfh_share', 'any_wfh']:
        formula = f'{outcome} ~ college + age + female + married + has_children'
        res = run_regression(df, formula, outcome, 'college',
                           f'ols/form/linear_age_{outcome}',
                           'methods/cross_sectional_ols.md#functional-form',
                           weights='wgt', sample_desc='Linear age only')
        if res:
            results.append(res)

    # ===================
    # INTERACTION EFFECTS
    # ===================
    print("Running interaction specifications...")

    for outcome in ['wfh_share', 'any_wfh']:
        # College x Female interaction
        df['college_female'] = df['college'] * df['female']
        formula = f'{outcome} ~ college + female + college_female + age + age_sq + married'
        res = run_regression(df, formula, outcome, 'college',
                           f'ols/interact/gender_{outcome}',
                           'methods/cross_sectional_ols.md#interaction-effects',
                           weights='wgt')
        if res:
            results.append(res)

        # Also report the interaction term
        res_int = run_regression(df, formula, outcome, 'college_female',
                               f'ols/interact/college_x_female_{outcome}',
                               'methods/cross_sectional_ols.md#interaction-effects',
                               weights='wgt')
        if res_int:
            results.append(res_int)

    return results


def main():
    """Main execution."""

    print("=" * 60)
    print("Specification Search: 158081-V7")
    print("Work from Home before and after the COVID-19 Outbreak")
    print("=" * 60)
    print()

    # Load data
    print("Loading and preparing data...")
    df = load_and_prepare_data()
    print(f"Analysis sample: {len(df)} observations")
    print()

    # Summary statistics
    print("Sample summary:")
    print(f"  WFH share mean: {df['wfh_share'].mean():.3f}")
    print(f"  Any WFH rate: {df['any_wfh'].mean():.3f}")
    print(f"  Days WFH mean: {df['days_wfh'].mean():.2f}")
    print(f"  College rate: {df['college'].mean():.3f}")
    print(f"  Female rate: {df['female'].mean():.3f}")
    print()

    # Run specification search
    results = run_specification_search(df)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    output_path = OUTPUT_DIR / 'specification_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(results_df)} specifications to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SPECIFICATION SEARCH SUMMARY")
    print("=" * 60)
    print(f"Total specifications: {len(results_df)}")
    print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
    print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
    print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")
    print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
    print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")
    print(f"Coefficient range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")

    # By outcome variable
    print("\nBy outcome variable:")
    for outcome in results_df['outcome_var'].unique():
        sub = results_df[results_df['outcome_var'] == outcome]
        sig_rate = 100 * (sub['p_value'] < 0.05).mean()
        print(f"  {outcome}: n={len(sub)}, median coef={sub['coefficient'].median():.4f}, sig rate={sig_rate:.1f}%")

    # By treatment variable
    print("\nBy treatment variable:")
    for treatment in results_df['treatment_var'].unique():
        sub = results_df[results_df['treatment_var'] == treatment]
        sig_rate = 100 * (sub['p_value'] < 0.05).mean()
        print(f"  {treatment}: n={len(sub)}, median coef={sub['coefficient'].median():.4f}, sig rate={sig_rate:.1f}%")

    return results_df


if __name__ == '__main__':
    results_df = main()
