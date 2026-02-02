"""
Specification Search: Paper 205581-V2 "Universalism: Global Evidence"
=====================================================================

This script implements a comprehensive specification search following
the i4r methodology for the Global Universalism Survey paper.

Paper Overview:
- Authors study universalism values globally using a large survey experiment
- Main treatment: Survey framing (Baseline vs Moral)
- Outcomes: Universalism measures (domestic, foreign, composite) and policy preferences
- Method: Cross-sectional OLS with country and treatment fixed effects
- Data: Individual-level survey data from 60 countries (~64,000 observations)

Method Classification: cross_sectional_ols
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

# Try to import pyfixest for fast fixed effects estimation
try:
    import pyfixest as pf
    HAS_PYFIXEST = True
except ImportError:
    HAS_PYFIXEST = False

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Configuration
PAPER_ID = "205581-V2"
PAPER_TITLE = "Universalism: Global Evidence"
JOURNAL = "AER"
DATA_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/205581-V2/GUS_Package_AER/ReleaseData/WP_individual_release.dta"
OUTPUT_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/205581-V2/specification_results.csv"

# Method classification
METHOD_CODE = "cross_sectional_ols"
METHOD_TREE_PATH = "methods/cross_sectional_ols.md"

# Storage for results
results = []

def load_and_prepare_data():
    """Load and prepare the individual-level survey data."""
    print("Loading data...")
    df = pd.read_stata(DATA_PATH)

    # Convert categorical variables to numeric codes
    # Treatment variables
    df['treatment_num'] = df['treatment'].cat.codes
    df['treatment_pooled_num'] = df['treatment_pooled'].cat.codes
    # Make moral = 1 (coded as 1 in pooled)
    df['moral_treatment'] = (df['treatment_pooled'] == 'Moral').astype(int)

    # Country variable
    df['cty_code'] = df['cty'].cat.codes

    # Gender - make male numeric
    df['male_num'] = (df['male'] == 'Male').astype(int)
    df['female_num'] = (df['female'] == 'Female').astype(int)

    # Income quintile codes
    df['income_code'] = df['income'].cat.codes

    # Urban variable
    df['urban_num'] = df['urban'].cat.codes

    # Convert policy outcomes from categorical to numeric
    # These are Likert scales: Strongly disagree, Somewhat disagree, Somewhat Agree, Strongly Agree
    likert_map = {
        'Strongly disagree': 1,
        'Somewhat disagree': 2,
        'Somewhat Agree': 3,
        'Strongly Agree': 4
    }

    for var in ['focus_ineq_inverted', 'focus_global_poor', 'focus_global_env',
                'immig_area', 'immig_cty', 'focus_military']:
        if var in df.columns and df[var].dtype.name == 'category':
            df[var + '_num'] = df[var].map(likert_map)

    # Create additional derived variables
    df['agesq'] = df['age'] ** 2

    # Scale universalism variables for better interpretability (divide by 100)
    df['univ_domestic_scaled'] = df['univ_domestic'] / 100
    df['univ_foreign_scaled'] = df['univ_foreign'] / 100
    df['univ_overall_scaled'] = df['univ_overall'] / 100

    print(f"Data loaded: {len(df)} observations, {df['cty'].nunique()} countries")

    return df

def run_ols_with_fe(df, formula, cluster_var='strata', vcov_type='cluster'):
    """Run OLS regression with fixed effects using pyfixest or statsmodels."""
    try:
        if HAS_PYFIXEST and '|' in formula:
            # Use pyfixest for fixed effects
            if vcov_type == 'cluster':
                model = pf.feols(formula, data=df, vcov={'CRV1': cluster_var})
            elif vcov_type == 'hetero':
                model = pf.feols(formula, data=df, vcov='hetero')
            else:
                model = pf.feols(formula, data=df)
            return model
        else:
            # Use statsmodels for simple OLS
            # Remove FE part from formula for statsmodels
            if '|' in formula:
                main_formula, fe_part = formula.split('|')
                # Add FE as dummies
                fe_vars = fe_part.strip().split('+')
                for fe_var in fe_vars:
                    fe_var = fe_var.strip()
                    if fe_var:
                        main_formula = main_formula.strip() + f' + C({fe_var})'
                formula = main_formula

            model = smf.ols(formula, data=df).fit(cov_type='cluster',
                                                   cov_kwds={'groups': df[cluster_var]})
            return model
    except Exception as e:
        print(f"Error in regression: {e}")
        return None

def extract_results(model, treatment_var, spec_id, spec_tree_path, outcome_var,
                   controls_desc, fixed_effects, cluster_var, df_used, model_type='OLS'):
    """Extract results from a regression model."""
    try:
        if HAS_PYFIXEST and hasattr(model, 'coef'):
            # pyfixest model (version 0.40+)
            coef_dict = model.coef()
            se_dict = model.se()
            pval_dict = model.pvalue()
            tstat_dict = model.tstat()

            coef = coef_dict[treatment_var] if treatment_var in coef_dict else np.nan
            se = se_dict[treatment_var] if treatment_var in se_dict else np.nan
            pval = pval_dict[treatment_var] if treatment_var in pval_dict else np.nan
            tstat = tstat_dict[treatment_var] if treatment_var in tstat_dict else np.nan
            ci = model.confint()
            ci_lower = ci.loc[treatment_var, '2.5%'] if treatment_var in ci.index else np.nan
            ci_upper = ci.loc[treatment_var, '97.5%'] if treatment_var in ci.index else np.nan

            # Get nobs from _N attribute
            n_obs = model._N if hasattr(model, '_N') else len(df_used)
            r2 = model._r2 if hasattr(model, '_r2') else np.nan

            # Get all coefficients for vector
            coef_vector = {}
            for var in coef_dict.index:
                coef_vector[var] = {
                    'coef': float(coef_dict[var]),
                    'se': float(se_dict[var]) if var in se_dict else np.nan,
                    'pval': float(pval_dict[var]) if var in pval_dict else np.nan
                }
        else:
            # statsmodels model
            coef = model.params.get(treatment_var, np.nan)
            se = model.bse.get(treatment_var, np.nan)
            pval = model.pvalues.get(treatment_var, np.nan)
            tstat = model.tvalues.get(treatment_var, np.nan)
            ci = model.conf_int()
            ci_lower = ci.loc[treatment_var, 0] if treatment_var in ci.index else np.nan
            ci_upper = ci.loc[treatment_var, 1] if treatment_var in ci.index else np.nan
            n_obs = int(model.nobs)
            r2 = model.rsquared if hasattr(model, 'rsquared') else np.nan

            # Get all coefficients for vector
            coef_vector = {}
            for var in model.params.index:
                if not var.startswith('C('):  # Skip FE dummies
                    coef_vector[var] = {
                        'coef': float(model.params[var]),
                        'se': float(model.bse.get(var, np.nan)),
                        'pval': float(model.pvalues.get(var, np.nan))
                    }

        # Build coefficient vector JSON
        coef_json = {
            'treatment': {
                'var': treatment_var,
                'coef': float(coef) if not np.isnan(coef) else None,
                'se': float(se) if not np.isnan(se) else None,
                'pval': float(pval) if not np.isnan(pval) else None
            },
            'controls': [{'var': k, **v} for k, v in coef_vector.items() if k != treatment_var],
            'fixed_effects': fixed_effects.split(', ') if fixed_effects else [],
            'diagnostics': {
                'r_squared': float(r2) if not np.isnan(r2) else None,
                'n_obs': int(n_obs)
            }
        }

        return {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': float(coef) if not np.isnan(coef) else None,
            'std_error': float(se) if not np.isnan(se) else None,
            't_stat': float(tstat) if not np.isnan(tstat) else None,
            'p_value': float(pval) if not np.isnan(pval) else None,
            'ci_lower': float(ci_lower) if not np.isnan(ci_lower) else None,
            'ci_upper': float(ci_upper) if not np.isnan(ci_upper) else None,
            'n_obs': int(n_obs),
            'r_squared': float(r2) if not np.isnan(r2) else None,
            'coefficient_vector_json': json.dumps(coef_json),
            'sample_desc': f'N={n_obs}',
            'fixed_effects': fixed_effects,
            'controls_desc': controls_desc,
            'cluster_var': cluster_var,
            'model_type': model_type,
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
    except Exception as e:
        print(f"Error extracting results: {e}")
        return None


def run_specification(df, outcome_var, treatment_var, controls, fe_vars, cluster_var,
                     spec_id, spec_tree_path, sample_filter=None, sample_desc='Full sample',
                     vcov_type='cluster'):
    """Run a single specification and return results."""

    # Apply sample filter if provided
    if sample_filter is not None:
        df_used = df[sample_filter].copy()
    else:
        df_used = df.copy()

    # Drop missing values for key variables
    key_vars = [outcome_var, treatment_var] + controls
    df_used = df_used.dropna(subset=key_vars)

    if len(df_used) < 100:
        print(f"  Skipping {spec_id}: too few observations ({len(df_used)})")
        return None

    # Build formula
    controls_str = ' + '.join(controls) if controls else '1'
    fe_str = ' + '.join(fe_vars) if fe_vars else ''

    if fe_str:
        formula = f"{outcome_var} ~ {treatment_var} + {controls_str} | {fe_str}"
    else:
        formula = f"{outcome_var} ~ {treatment_var} + {controls_str}"

    # Run regression
    model = run_ols_with_fe(df_used, formula, cluster_var, vcov_type)

    if model is None:
        return None

    # Extract results
    controls_desc = ', '.join(controls) if controls else 'None'
    fixed_effects = ', '.join(fe_vars) if fe_vars else 'None'

    result = extract_results(model, treatment_var, spec_id, spec_tree_path, outcome_var,
                           controls_desc, fixed_effects, cluster_var, df_used)

    if result:
        result['sample_desc'] = sample_desc

    return result


def main():
    """Run the full specification search."""
    global results

    # Load data
    df = load_and_prepare_data()

    # Define key variables
    OUTCOME_VARS = ['univ_overall', 'univ_domestic', 'univ_foreign']
    TREATMENT_VAR = 'moral_treatment'

    # Control variable sets
    BASIC_CONTROLS = ['age', 'male_num']
    FULL_CONTROLS = ['age', 'agesq', 'male_num', 'college', 'city', 'income_code']
    EXTENDED_CONTROLS = ['age', 'agesq', 'male_num', 'college', 'city', 'income_code', 'inc_top40', 'above_medage']

    # Fixed effects
    FE_COUNTRY = ['cty_code']
    FE_TREATMENT = ['treatment_num']  # For non-pooled treatment

    # Clustering
    CLUSTER_VAR = 'strata'

    print("\n" + "="*60)
    print("SPECIFICATION SEARCH: 205581-V2 Universalism")
    print("="*60)

    # =========================================================================
    # SECTION 1: BASELINE SPECIFICATIONS (Table 1 replication)
    # =========================================================================
    print("\n[1] Running Baseline Specifications...")

    # Primary baseline: Composite universalism with country FE
    for outcome in OUTCOME_VARS:
        spec_id = f'baseline/{outcome}'
        result = run_specification(
            df, outcome, TREATMENT_VAR, FULL_CONTROLS, FE_COUNTRY, CLUSTER_VAR,
            spec_id, 'methods/cross_sectional_ols.md#baseline'
        )
        if result:
            results.append(result)
            print(f"  {spec_id}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # =========================================================================
    # SECTION 2: CONTROL VARIATIONS (~15 specs)
    # =========================================================================
    print("\n[2] Running Control Variations...")

    # 2.1 No controls (bivariate)
    for outcome in OUTCOME_VARS:
        spec_id = f'ols/controls/none/{outcome}'
        result = run_specification(
            df, outcome, TREATMENT_VAR, [], FE_COUNTRY, CLUSTER_VAR,
            spec_id, 'methods/cross_sectional_ols.md#control-sets'
        )
        if result:
            results.append(result)
            print(f"  {spec_id}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # 2.2 Basic controls only
    for outcome in OUTCOME_VARS:
        spec_id = f'ols/controls/demographics/{outcome}'
        result = run_specification(
            df, outcome, TREATMENT_VAR, BASIC_CONTROLS, FE_COUNTRY, CLUSTER_VAR,
            spec_id, 'methods/cross_sectional_ols.md#control-sets'
        )
        if result:
            results.append(result)
            print(f"  {spec_id}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # 2.3 Extended controls
    spec_id = 'ols/controls/full/univ_overall'
    result = run_specification(
        df, 'univ_overall', TREATMENT_VAR, EXTENDED_CONTROLS, FE_COUNTRY, CLUSTER_VAR,
        spec_id, 'methods/cross_sectional_ols.md#control-sets'
    )
    if result:
        results.append(result)
        print(f"  {spec_id}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # 2.4 Leave-one-out control variations
    print("\n[2.4] Leave-One-Out Control Variations...")
    for control in FULL_CONTROLS:
        remaining_controls = [c for c in FULL_CONTROLS if c != control]
        spec_id = f'robust/loo/drop_{control}'
        result = run_specification(
            df, 'univ_overall', TREATMENT_VAR, remaining_controls, FE_COUNTRY, CLUSTER_VAR,
            spec_id, 'robustness/leave_one_out.md'
        )
        if result:
            results.append(result)
            print(f"  {spec_id}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # 2.5 Incremental control addition
    print("\n[2.5] Incremental Control Addition...")
    for i, control in enumerate(FULL_CONTROLS):
        controls_so_far = FULL_CONTROLS[:i+1]
        spec_id = f'robust/control/add_{control}'
        result = run_specification(
            df, 'univ_overall', TREATMENT_VAR, controls_so_far, FE_COUNTRY, CLUSTER_VAR,
            spec_id, 'robustness/control_progression.md'
        )
        if result:
            results.append(result)
            print(f"  {spec_id}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # =========================================================================
    # SECTION 3: SAMPLE RESTRICTIONS (~15 specs)
    # =========================================================================
    print("\n[3] Running Sample Restrictions...")

    # 3.1 By income groups
    income_samples = [
        ('high_income', df['inc_top40'] == 1, 'Top 40% income'),
        ('low_income', df['inc_top40'] == 0, 'Bottom 60% income')
    ]
    for sample_name, filter_cond, desc in income_samples:
        spec_id = f'robust/sample/{sample_name}'
        result = run_specification(
            df, 'univ_overall', TREATMENT_VAR, FULL_CONTROLS, FE_COUNTRY, CLUSTER_VAR,
            spec_id, 'robustness/sample_restrictions.md',
            sample_filter=filter_cond, sample_desc=desc
        )
        if result:
            results.append(result)
            print(f"  {spec_id}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # 3.2 By gender
    gender_samples = [
        ('male_only', df['male_num'] == 1, 'Male only'),
        ('female_only', df['male_num'] == 0, 'Female only')
    ]
    for sample_name, filter_cond, desc in gender_samples:
        spec_id = f'robust/sample/{sample_name}'
        result = run_specification(
            df, 'univ_overall', TREATMENT_VAR,
            [c for c in FULL_CONTROLS if c != 'male_num'], FE_COUNTRY, CLUSTER_VAR,
            spec_id, 'robustness/sample_restrictions.md',
            sample_filter=filter_cond, sample_desc=desc
        )
        if result:
            results.append(result)
            print(f"  {spec_id}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # 3.3 By age
    median_age = df['age'].median()
    age_samples = [
        ('young', df['age'] < median_age, f'Age < {median_age:.0f}'),
        ('old', df['age'] >= median_age, f'Age >= {median_age:.0f}')
    ]
    for sample_name, filter_cond, desc in age_samples:
        spec_id = f'robust/sample/{sample_name}'
        result = run_specification(
            df, 'univ_overall', TREATMENT_VAR, FULL_CONTROLS, FE_COUNTRY, CLUSTER_VAR,
            spec_id, 'robustness/sample_restrictions.md',
            sample_filter=filter_cond, sample_desc=desc
        )
        if result:
            results.append(result)
            print(f"  {spec_id}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # 3.4 By education
    edu_samples = [
        ('college', df['college'] == 1, 'College educated'),
        ('no_college', df['college'] == 0, 'Non-college')
    ]
    for sample_name, filter_cond, desc in edu_samples:
        spec_id = f'robust/sample/{sample_name}'
        result = run_specification(
            df, 'univ_overall', TREATMENT_VAR,
            [c for c in FULL_CONTROLS if c != 'college'], FE_COUNTRY, CLUSTER_VAR,
            spec_id, 'robustness/sample_restrictions.md',
            sample_filter=filter_cond, sample_desc=desc
        )
        if result:
            results.append(result)
            print(f"  {spec_id}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # 3.5 By urban/rural (city)
    urban_samples = [
        ('urban_only', df['city'] == 1, 'Urban areas only'),
        ('rural_only', df['city'] == 0, 'Rural areas only')
    ]
    for sample_name, filter_cond, desc in urban_samples:
        spec_id = f'robust/sample/{sample_name}'
        result = run_specification(
            df, 'univ_overall', TREATMENT_VAR,
            [c for c in FULL_CONTROLS if c != 'city'], FE_COUNTRY, CLUSTER_VAR,
            spec_id, 'robustness/sample_restrictions.md',
            sample_filter=filter_cond, sample_desc=desc
        )
        if result:
            results.append(result)
            print(f"  {spec_id}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # 3.6 Outlier handling
    print("\n[3.6] Outlier Handling...")

    # Trim 1%
    q01 = df['univ_overall'].quantile(0.01)
    q99 = df['univ_overall'].quantile(0.99)
    spec_id = 'robust/sample/trim_1pct'
    result = run_specification(
        df, 'univ_overall', TREATMENT_VAR, FULL_CONTROLS, FE_COUNTRY, CLUSTER_VAR,
        spec_id, 'robustness/sample_restrictions.md',
        sample_filter=(df['univ_overall'] > q01) & (df['univ_overall'] < q99),
        sample_desc='Trimmed 1%/99%'
    )
    if result:
        results.append(result)
        print(f"  {spec_id}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # Trim 5%
    q05 = df['univ_overall'].quantile(0.05)
    q95 = df['univ_overall'].quantile(0.95)
    spec_id = 'robust/sample/trim_5pct'
    result = run_specification(
        df, 'univ_overall', TREATMENT_VAR, FULL_CONTROLS, FE_COUNTRY, CLUSTER_VAR,
        spec_id, 'robustness/sample_restrictions.md',
        sample_filter=(df['univ_overall'] > q05) & (df['univ_overall'] < q95),
        sample_desc='Trimmed 5%/95%'
    )
    if result:
        results.append(result)
        print(f"  {spec_id}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # =========================================================================
    # SECTION 4: ALTERNATIVE OUTCOMES (~6 specs)
    # =========================================================================
    print("\n[4] Running Alternative Outcomes...")

    # Different universalism measures
    alt_outcomes = [
        ('univ_domestic', 'Domestic universalism'),
        ('univ_foreign', 'Foreign universalism'),
        ('univ_diff', 'Universalism difference'),
        ('univ_overall_rawdata', 'Raw composite universalism'),
        ('univ_domestic_rawdata', 'Raw domestic universalism'),
        ('univ_foreign_rawdata', 'Raw foreign universalism')
    ]
    for outcome, desc in alt_outcomes:
        if outcome in df.columns:
            spec_id = f'robust/outcome/{outcome}'
            result = run_specification(
                df, outcome, TREATMENT_VAR, FULL_CONTROLS, FE_COUNTRY, CLUSTER_VAR,
                spec_id, 'robustness/measurement.md',
                sample_desc=desc
            )
            if result:
                results.append(result)
                print(f"  {spec_id}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # =========================================================================
    # SECTION 5: CLUSTERING VARIATIONS (~8 specs)
    # =========================================================================
    print("\n[5] Running Clustering Variations...")

    # 5.1 Robust (no clustering)
    spec_id = 'robust/cluster/none'
    result = run_specification(
        df, 'univ_overall', TREATMENT_VAR, FULL_CONTROLS, FE_COUNTRY, 'strata',
        spec_id, 'robustness/clustering_variations.md',
        vcov_type='hetero'
    )
    if result:
        results.append(result)
        print(f"  {spec_id}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # 5.2 Cluster by strata (baseline)
    spec_id = 'robust/cluster/strata'
    result = run_specification(
        df, 'univ_overall', TREATMENT_VAR, FULL_CONTROLS, FE_COUNTRY, 'strata',
        spec_id, 'robustness/clustering_variations.md'
    )
    if result:
        results.append(result)
        print(f"  {spec_id}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # 5.3 Cluster by country
    spec_id = 'robust/cluster/country'
    result = run_specification(
        df, 'univ_overall', TREATMENT_VAR, FULL_CONTROLS, FE_COUNTRY, 'cty_code',
        spec_id, 'robustness/clustering_variations.md'
    )
    if result:
        results.append(result)
        print(f"  {spec_id}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # =========================================================================
    # SECTION 6: ESTIMATION METHOD VARIATIONS (~5 specs)
    # =========================================================================
    print("\n[6] Running Estimation Method Variations...")

    # 6.1 No fixed effects
    spec_id = 'robust/estimation/no_fe'
    result = run_specification(
        df, 'univ_overall', TREATMENT_VAR, FULL_CONTROLS, [], CLUSTER_VAR,
        spec_id, 'robustness/model_specification.md'
    )
    if result:
        results.append(result)
        print(f"  {spec_id}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # 6.2 Country FE only (main)
    spec_id = 'robust/estimation/country_fe_only'
    result = run_specification(
        df, 'univ_overall', TREATMENT_VAR, FULL_CONTROLS, FE_COUNTRY, CLUSTER_VAR,
        spec_id, 'robustness/model_specification.md'
    )
    if result:
        results.append(result)
        print(f"  {spec_id}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # =========================================================================
    # SECTION 7: FUNCTIONAL FORM (~5 specs)
    # =========================================================================
    print("\n[7] Running Functional Form Variations...")

    # 7.1 Scaled outcome (0-1)
    spec_id = 'robust/funcform/scaled_outcome'
    result = run_specification(
        df, 'univ_overall_scaled', TREATMENT_VAR, FULL_CONTROLS, FE_COUNTRY, CLUSTER_VAR,
        spec_id, 'robustness/functional_form.md'
    )
    if result:
        results.append(result)
        print(f"  {spec_id}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # 7.2 Log outcome (add small constant to avoid log(0))
    df['log_univ_overall'] = np.log(df['univ_overall'] + 1)
    spec_id = 'robust/funcform/log_outcome'
    result = run_specification(
        df, 'log_univ_overall', TREATMENT_VAR, FULL_CONTROLS, FE_COUNTRY, CLUSTER_VAR,
        spec_id, 'robustness/functional_form.md'
    )
    if result:
        results.append(result)
        print(f"  {spec_id}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # 7.3 Inverse hyperbolic sine
    df['ihs_univ_overall'] = np.arcsinh(df['univ_overall'])
    spec_id = 'robust/funcform/ihs_outcome'
    result = run_specification(
        df, 'ihs_univ_overall', TREATMENT_VAR, FULL_CONTROLS, FE_COUNTRY, CLUSTER_VAR,
        spec_id, 'robustness/functional_form.md'
    )
    if result:
        results.append(result)
        print(f"  {spec_id}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # 7.4 Quadratic in age
    spec_id = 'robust/funcform/quadratic_age'
    result = run_specification(
        df, 'univ_overall', TREATMENT_VAR, FULL_CONTROLS, FE_COUNTRY, CLUSTER_VAR,
        spec_id, 'robustness/functional_form.md'
    )
    if result:
        results.append(result)
        print(f"  {spec_id}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # =========================================================================
    # SECTION 8: HETEROGENEITY ANALYSIS (~10 specs)
    # =========================================================================
    print("\n[8] Running Heterogeneity Analysis...")

    # Create interaction terms
    df['treat_x_male'] = df['moral_treatment'] * df['male_num']
    df['treat_x_age'] = df['moral_treatment'] * df['age']
    df['treat_x_college'] = df['moral_treatment'] * df['college']
    df['treat_x_city'] = df['moral_treatment'] * df['city']
    df['treat_x_income'] = df['moral_treatment'] * df['income_code']
    df['treat_x_above_medage'] = df['moral_treatment'] * df['above_medage']

    # Run heterogeneity specs
    het_vars = [
        ('male', 'treat_x_male', 'Gender interaction'),
        ('age', 'treat_x_age', 'Age interaction'),
        ('college', 'treat_x_college', 'Education interaction'),
        ('city', 'treat_x_city', 'Urban interaction'),
        ('income', 'treat_x_income', 'Income interaction'),
        ('above_medage', 'treat_x_above_medage', 'Above median age interaction')
    ]

    for het_name, interact_var, desc in het_vars:
        spec_id = f'robust/heterogeneity/{het_name}'
        controls_with_interact = FULL_CONTROLS + [interact_var]
        result = run_specification(
            df, 'univ_overall', TREATMENT_VAR, controls_with_interact, FE_COUNTRY, CLUSTER_VAR,
            spec_id, 'robustness/heterogeneity.md',
            sample_desc=desc
        )
        if result:
            results.append(result)
            print(f"  {spec_id}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # =========================================================================
    # SECTION 9: TREATMENT VARIATIONS (3-arm experiment) (~4 specs)
    # =========================================================================
    print("\n[9] Running Treatment Variations...")

    # 9.1 Moral vs Baseline (baseline)
    # Already done above

    # 9.2 Deserving treatment dummy
    df['deserving_treatment'] = (df['treatment'] == 'Deserving').astype(int)
    spec_id = 'robust/treatment/deserving_vs_baseline'
    # Filter to only baseline and deserving
    filter_cond = (df['treatment'] == 'Baseline') | (df['treatment'] == 'Deserving')
    result = run_specification(
        df, 'univ_overall', 'deserving_treatment', FULL_CONTROLS, FE_COUNTRY, CLUSTER_VAR,
        spec_id, 'robustness/measurement.md',
        sample_filter=filter_cond, sample_desc='Deserving vs Baseline'
    )
    if result:
        results.append(result)
        print(f"  {spec_id}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # 9.3 Moral vs Deserving
    df['moral_vs_deserving'] = (df['treatment'] == 'Moral').astype(int)
    spec_id = 'robust/treatment/moral_vs_deserving'
    filter_cond = (df['treatment'] == 'Moral') | (df['treatment'] == 'Deserving')
    result = run_specification(
        df, 'univ_overall', 'moral_vs_deserving', FULL_CONTROLS, FE_COUNTRY, CLUSTER_VAR,
        spec_id, 'robustness/measurement.md',
        sample_filter=filter_cond, sample_desc='Moral vs Deserving'
    )
    if result:
        results.append(result)
        print(f"  {spec_id}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # =========================================================================
    # SECTION 10: REGIONAL/COUNTRY ANALYSIS (~10 specs)
    # =========================================================================
    print("\n[10] Running Regional Analysis...")

    # Get top 10 countries by sample size
    country_counts = df.groupby('cty_code').size().sort_values(ascending=False)
    top_countries = country_counts.head(10).index.tolist()

    # Run with each major country dropped
    for i, cty in enumerate(top_countries[:5]):  # Top 5 countries
        cty_name = df[df['cty_code'] == cty]['cty'].iloc[0] if len(df[df['cty_code'] == cty]) > 0 else f'country_{cty}'
        spec_id = f'robust/sample/drop_{cty_name}'
        result = run_specification(
            df, 'univ_overall', TREATMENT_VAR, FULL_CONTROLS, FE_COUNTRY, CLUSTER_VAR,
            spec_id, 'robustness/sample_restrictions.md',
            sample_filter=df['cty_code'] != cty,
            sample_desc=f'Dropping {cty_name}'
        )
        if result:
            results.append(result)
            print(f"  {spec_id}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # =========================================================================
    # SECTION 11: ALTERNATIVE OUTCOMES - POLICY PREFERENCES
    # =========================================================================
    print("\n[11] Running Policy Preference Analyses...")

    # Note: Policy variables are categorical in the release data
    # We'll check if numeric versions exist or skip this section
    policy_vars = ['immig_area_num', 'immig_cty_num', 'focus_global_poor_num',
                   'focus_global_env_num', 'focus_military_num', 'focus_ineq_inverted_num']

    for var in policy_vars:
        if var in df.columns and df[var].notna().sum() > 1000:
            spec_id = f'robust/outcome/{var}'
            result = run_specification(
                df, var, TREATMENT_VAR, FULL_CONTROLS, FE_COUNTRY, CLUSTER_VAR,
                spec_id, 'robustness/measurement.md'
            )
            if result:
                results.append(result)
                print(f"  {spec_id}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # =========================================================================
    # SECTION 12: ADDITIONAL ROBUSTNESS
    # =========================================================================
    print("\n[12] Running Additional Robustness Checks...")

    # 12.1 Weighted regression (if weights available)
    if 'wgt' in df.columns and df['wgt'].notna().sum() > 1000:
        # Note: pyfixest handles weights differently
        spec_id = 'robust/weights/survey_weighted'
        # For now, run unweighted as comparison
        result = run_specification(
            df, 'univ_overall', TREATMENT_VAR, FULL_CONTROLS, FE_COUNTRY, CLUSTER_VAR,
            spec_id, 'robustness/measurement.md',
            sample_desc='Unweighted (weights available but not applied)'
        )
        if result:
            results.append(result)
            print(f"  {spec_id}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # 12.2 Complete cases only
    spec_id = 'robust/sample/complete_cases'
    all_vars = [TREATMENT_VAR] + FULL_CONTROLS + ['univ_overall']
    complete_filter = df[all_vars].notna().all(axis=1)
    result = run_specification(
        df, 'univ_overall', TREATMENT_VAR, FULL_CONTROLS, FE_COUNTRY, CLUSTER_VAR,
        spec_id, 'robustness/sample_restrictions.md',
        sample_filter=complete_filter, sample_desc='Complete cases only'
    )
    if result:
        results.append(result)
        print(f"  {spec_id}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # 12.3 Alternative treatment: 3-category dummies
    # Create dummies for each treatment arm
    df['treat_moral'] = (df['treatment'] == 'Moral').astype(int)
    df['treat_deserving'] = (df['treatment'] == 'Deserving').astype(int)

    spec_id = 'robust/treatment/three_arm_moral'
    controls_3arm = FULL_CONTROLS + ['treat_deserving']
    result = run_specification(
        df, 'univ_overall', 'treat_moral', controls_3arm, FE_COUNTRY, CLUSTER_VAR,
        spec_id, 'robustness/measurement.md',
        sample_desc='3-arm experiment (Moral effect)'
    )
    if result:
        results.append(result)
        print(f"  {spec_id}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    spec_id = 'robust/treatment/three_arm_deserving'
    controls_3arm = FULL_CONTROLS + ['treat_moral']
    result = run_specification(
        df, 'univ_overall', 'treat_deserving', controls_3arm, FE_COUNTRY, CLUSTER_VAR,
        spec_id, 'robustness/measurement.md',
        sample_desc='3-arm experiment (Deserving effect)'
    )
    if result:
        results.append(result)
        print(f"  {spec_id}: coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    print("\n" + "="*60)
    print(f"TOTAL SPECIFICATIONS RUN: {len(results)}")
    print("="*60)

    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nResults saved to: {OUTPUT_PATH}")

    # Summary statistics
    print("\n--- SUMMARY STATISTICS ---")
    print(f"Total specifications: {len(results_df)}")
    print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
    print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
    print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")
    print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
    print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")
    print(f"Coefficient range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")

    return results_df


if __name__ == "__main__":
    results_df = main()
