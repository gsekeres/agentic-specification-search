"""
Specification Search: 194886-V3
Paper: "Resisting Social Pressure in the Household Using Mobile Money:
       Experimental Evidence on Microenterprise Investment in Uganda"
Author: Emma Riley
Journal: AER

This script runs systematic specification variations on the main results.
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# Configuration
# ==============================================================================

PAPER_ID = "194886-V3"
JOURNAL = "AER"
PAPER_TITLE = "Resisting Social Pressure in the Household Using Mobile Money: Experimental Evidence on Microenterprise Investment in Uganda"

# Paths
DATA_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/194886-V3/replication/input/survey_data.dta"
OUTPUT_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/194886-V3/replication/specification_results.csv"

# Primary outcomes (from master.do)
PRIMARY_OUTCOMES = ['earn_business', 'much_saved', 'capital']

# Baseline controls - the paper controls for baseline value of outcome + strata FE
# The main specification is: outcome ~ outcome_base + treatment2 + treatment3 | strata_fixed_base

# ==============================================================================
# Load Data
# ==============================================================================

print("Loading data...")
df = pd.read_stata(DATA_PATH)

# Filter to consented sample (drop if consent != 1, equivalent to Stata's drop if consent!=1)
df = df[df['consent'] == 'yes'].copy()
print(f"Sample after consent filter: {len(df)} observations")

# Create treatment indicators from string variable
df['treatment1_num'] = (df['treatment1'] == 1).astype(int)
df['treatment2_num'] = (df['treatment2'] == 1).astype(int)
df['treatment3_num'] = (df['treatment3'] == 1).astype(int)

# Convert strata to numeric for pyfixest
df['strata_num'] = pd.factorize(df['strata_fixed_base'])[0]

# ==============================================================================
# Helper Functions
# ==============================================================================

def extract_results(model, treatment_var, spec_id, spec_tree_path, outcome_var,
                    controls_desc, fixed_effects, cluster_var, model_type,
                    sample_desc="Full consented sample"):
    """Extract results from a pyfixest or statsmodels model."""

    is_pyfixest = hasattr(model, '_N')

    if is_pyfixest:
        # Pyfixest interface
        coef_series = model.coef()
        se_series = model.se()
        pval_series = model.pvalue()

        coef = coef_series[treatment_var]
        se = se_series[treatment_var]
        pval = pval_series[treatment_var]
        n_obs = model._N
        r2 = model._r2

        # Get all coefficients
        all_coefs = coef_series.to_dict()
        all_se = se_series.to_dict()
        all_pval = pval_series.to_dict()

    else:
        # Try statsmodels interface
        coef = model.params[treatment_var]
        se = model.bse[treatment_var]
        pval = model.pvalues[treatment_var]
        n_obs = int(model.nobs)

        try:
            r2 = model.rsquared
        except:
            r2 = None

        all_coefs = model.params.to_dict()
        all_se = model.bse.to_dict()
        all_pval = model.pvalues.to_dict()

    # Calculate t-stat and CI
    t_stat = coef / se
    ci_lower = coef - 1.96 * se
    ci_upper = coef + 1.96 * se

    # Build coefficient vector JSON
    coef_vector = {
        "treatment": {"var": treatment_var, "coef": float(coef), "se": float(se), "pval": float(pval)},
        "controls": [],
        "fixed_effects": fixed_effects.split(', ') if fixed_effects else [],
        "diagnostics": {}
    }

    for var in all_coefs:
        if var != treatment_var and var != 'Intercept':
            coef_vector["controls"].append({
                "var": var,
                "coef": float(all_coefs[var]),
                "se": float(all_se[var]),
                "pval": float(all_pval[var])
            })

    return {
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'outcome_var': outcome_var,
        'treatment_var': treatment_var,
        'coefficient': float(coef),
        'std_error': float(se),
        't_stat': float(t_stat),
        'p_value': float(pval),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'n_obs': int(n_obs),
        'r_squared': float(r2) if r2 is not None else None,
        'coefficient_vector_json': json.dumps(coef_vector),
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': 'scripts/paper_analyses/194886-V3.py'
    }


def run_specification(df, outcome, treatment_vars, controls, fe_var=None,
                      cluster_var=None, vcov_type='hetero'):
    """Run a regression specification and return model."""

    # Build formula
    control_str = " + ".join(controls) if controls else ""
    treatment_str = " + ".join(treatment_vars)

    if control_str:
        rhs = f"{treatment_str} + {control_str}"
    else:
        rhs = treatment_str

    if fe_var:
        formula = f"{outcome} ~ {rhs} | {fe_var}"
    else:
        formula = f"{outcome} ~ {rhs}"

    # Run regression
    if vcov_type == 'hetero':
        model = pf.feols(formula, data=df, vcov='hetero')
    elif vcov_type == 'cluster':
        model = pf.feols(formula, data=df, vcov={'CRV1': cluster_var})
    else:
        model = pf.feols(formula, data=df, vcov=vcov_type)

    return model


# ==============================================================================
# Run Specifications
# ==============================================================================

results = []

print("\n" + "="*80)
print("RUNNING SPECIFICATION SEARCH")
print("="*80)

# ------------------------------------------------------------------------------
# BASELINE SPECIFICATIONS (Table 1 replication)
# ------------------------------------------------------------------------------
print("\n--- Baseline Specifications (Table 1 replication) ---")

for outcome in PRIMARY_OUTCOMES:
    print(f"  Running baseline for {outcome}...")

    baseline_control = f"{outcome}_base"

    # Check if baseline control exists
    if baseline_control not in df.columns:
        print(f"    Warning: {baseline_control} not found, skipping")
        continue

    # Create analysis sample (drop missing)
    df_temp = df[[outcome, baseline_control, 'treatment2_num', 'treatment3_num', 'strata_num']].dropna()

    # Run baseline specification: outcome ~ outcome_base + treatment2 + treatment3 | strata
    try:
        model = pf.feols(f"{outcome} ~ {baseline_control} + treatment2_num + treatment3_num | strata_num",
                         data=df_temp, vcov='hetero')

        # Extract results for treatment2 (Mobile Account)
        results.append(extract_results(
            model, 'treatment2_num',
            spec_id='baseline',
            spec_tree_path='methods/cross_sectional_ols.md#baseline',
            outcome_var=outcome,
            controls_desc=f'{baseline_control}',
            fixed_effects='strata',
            cluster_var='robust',
            model_type='OLS with absorbing strata FE'
        ))

        # Extract results for treatment3 (Mobile Disburse) as separate row
        results.append(extract_results(
            model, 'treatment3_num',
            spec_id='baseline',
            spec_tree_path='methods/cross_sectional_ols.md#baseline',
            outcome_var=outcome,
            controls_desc=f'{baseline_control}',
            fixed_effects='strata',
            cluster_var='robust',
            model_type='OLS with absorbing strata FE'
        ))

        print(f"    {outcome}: treatment2 coef = {model.coef()['treatment2_num']:.2f}, treatment3 coef = {model.coef()['treatment3_num']:.2f}")

    except Exception as e:
        print(f"    Error running baseline for {outcome}: {e}")

# ------------------------------------------------------------------------------
# CONTROL VARIATIONS
# ------------------------------------------------------------------------------
print("\n--- Control Variations ---")

# No controls (bivariate)
print("  Running bivariate specifications...")
for outcome in PRIMARY_OUTCOMES:
    df_temp = df[[outcome, 'treatment2_num', 'treatment3_num', 'strata_num']].dropna()

    try:
        model = pf.feols(f"{outcome} ~ treatment2_num + treatment3_num | strata_num",
                         data=df_temp, vcov='hetero')

        for tvar, tname in [('treatment2_num', 'Mobile Account'), ('treatment3_num', 'Mobile Disburse')]:
            results.append(extract_results(
                model, tvar,
                spec_id='ols/controls/none',
                spec_tree_path='methods/cross_sectional_ols.md#control-sets',
                outcome_var=outcome,
                controls_desc='None (bivariate)',
                fixed_effects='strata',
                cluster_var='robust',
                model_type='OLS with absorbing strata FE'
            ))
    except Exception as e:
        print(f"    Error in bivariate for {outcome}: {e}")

# Add demographic controls
print("  Running with demographic controls...")
demo_controls = ['respondent_age_base', 'married_base', 'hh_size_base',
                 'completed_primary_base', 'completed_secondary_base']

for outcome in PRIMARY_OUTCOMES:
    baseline_control = f"{outcome}_base"

    # Get available controls
    available_controls = [c for c in demo_controls + [baseline_control] if c in df.columns]
    df_temp = df[[outcome, 'treatment2_num', 'treatment3_num', 'strata_num'] + available_controls].dropna()

    try:
        control_str = " + ".join(available_controls)
        model = pf.feols(f"{outcome} ~ treatment2_num + treatment3_num + {control_str} | strata_num",
                         data=df_temp, vcov='hetero')

        for tvar in ['treatment2_num', 'treatment3_num']:
            results.append(extract_results(
                model, tvar,
                spec_id='ols/controls/demographics',
                spec_tree_path='methods/cross_sectional_ols.md#control-sets',
                outcome_var=outcome,
                controls_desc='Demographics + baseline outcome',
                fixed_effects='strata',
                cluster_var='robust',
                model_type='OLS with absorbing strata FE'
            ))
    except Exception as e:
        print(f"    Error with demographics for {outcome}: {e}")

# Full controls (as in Table A12)
print("  Running with full controls...")
full_controls = ['respondent_age_base', 'married_base', 'hh_size_base',
                 'completed_primary_base', 'completed_secondary_base',
                 'mobile_account_base', 'hyperbolic_base', 'work_occupation_base']

for outcome in PRIMARY_OUTCOMES:
    baseline_control = f"{outcome}_base"

    available_controls = [c for c in full_controls + [baseline_control] if c in df.columns]
    df_temp = df[[outcome, 'treatment2_num', 'treatment3_num', 'strata_num'] + available_controls].dropna()

    try:
        control_str = " + ".join(available_controls)
        model = pf.feols(f"{outcome} ~ treatment2_num + treatment3_num + {control_str} | strata_num",
                         data=df_temp, vcov='hetero')

        for tvar in ['treatment2_num', 'treatment3_num']:
            results.append(extract_results(
                model, tvar,
                spec_id='ols/controls/full',
                spec_tree_path='methods/cross_sectional_ols.md#control-sets',
                outcome_var=outcome,
                controls_desc='Full controls (demographics + baseline + behavioral)',
                fixed_effects='strata',
                cluster_var='robust',
                model_type='OLS with absorbing strata FE'
            ))
    except Exception as e:
        print(f"    Error with full controls for {outcome}: {e}")

# ------------------------------------------------------------------------------
# FUNCTIONAL FORM VARIATIONS
# ------------------------------------------------------------------------------
print("\n--- Functional Form Variations ---")

# Log outcome
print("  Running log specifications...")
for outcome in PRIMARY_OUTCOMES:
    baseline_control = f"{outcome}_base"

    # Create log variables
    df_temp = df[[outcome, baseline_control, 'treatment2_num', 'treatment3_num', 'strata_num']].dropna()
    df_temp = df_temp[df_temp[outcome] > 0]  # Need positive values for log
    df_temp['ln_outcome'] = np.log(df_temp[outcome])
    df_temp['ln_baseline'] = np.log(df_temp[baseline_control].clip(lower=0.01))

    try:
        model = pf.feols(f"ln_outcome ~ ln_baseline + treatment2_num + treatment3_num | strata_num",
                         data=df_temp, vcov='hetero')

        for tvar in ['treatment2_num', 'treatment3_num']:
            results.append(extract_results(
                model, tvar,
                spec_id='robust/form/y_log',
                spec_tree_path='robustness/functional_form.md',
                outcome_var=f'ln_{outcome}',
                controls_desc=f'ln_{baseline_control}',
                fixed_effects='strata',
                cluster_var='robust',
                model_type='OLS (log-log)',
                sample_desc='Positive values only'
            ))
    except Exception as e:
        print(f"    Error with log for {outcome}: {e}")

# Inverse hyperbolic sine (handles zeros)
print("  Running asinh specifications...")
for outcome in PRIMARY_OUTCOMES:
    baseline_control = f"{outcome}_base"

    df_temp = df[[outcome, baseline_control, 'treatment2_num', 'treatment3_num', 'strata_num']].dropna()
    df_temp['asinh_outcome'] = np.arcsinh(df_temp[outcome])
    df_temp['asinh_baseline'] = np.arcsinh(df_temp[baseline_control])

    try:
        model = pf.feols(f"asinh_outcome ~ asinh_baseline + treatment2_num + treatment3_num | strata_num",
                         data=df_temp, vcov='hetero')

        for tvar in ['treatment2_num', 'treatment3_num']:
            results.append(extract_results(
                model, tvar,
                spec_id='robust/form/y_asinh',
                spec_tree_path='robustness/functional_form.md',
                outcome_var=f'asinh_{outcome}',
                controls_desc=f'asinh_{baseline_control}',
                fixed_effects='strata',
                cluster_var='robust',
                model_type='OLS (asinh transformation)'
            ))
    except Exception as e:
        print(f"    Error with asinh for {outcome}: {e}")

# ------------------------------------------------------------------------------
# SAMPLE RESTRICTIONS
# ------------------------------------------------------------------------------
print("\n--- Sample Restrictions ---")

# Different winsorization levels (as in Table A11)
winsor_levels = [(100, 'No winsorizing'), (99.5, 'Winsorize 99.5%'),
                 (98, 'Winsorize 98%'), (95, 'Winsorize 95%')]

for pctile, desc in winsor_levels:
    print(f"  Running {desc}...")

    for outcome in PRIMARY_OUTCOMES:
        # The data already has winsorized versions
        if pctile == 100:
            outcome_var = f"{outcome}_100"
            baseline_var = f"{outcome}_100_base"
        elif pctile == 99.5:
            outcome_var = f"{outcome}_995"
            baseline_var = f"{outcome}_995_base"
        elif pctile == 98:
            outcome_var = f"{outcome}_98"
            baseline_var = f"{outcome}_98_base"
        elif pctile == 95:
            outcome_var = f"{outcome}_95"
            baseline_var = f"{outcome}_95_base"

        if outcome_var not in df.columns or baseline_var not in df.columns:
            continue

        df_temp = df[[outcome_var, baseline_var, 'treatment2_num', 'treatment3_num', 'strata_num']].dropna()

        try:
            model = pf.feols(f"{outcome_var} ~ {baseline_var} + treatment2_num + treatment3_num | strata_num",
                             data=df_temp, vcov='hetero')

            spec_id = f"robust/sample/winsor_{int(pctile)}pct" if pctile < 100 else "robust/sample/no_winsor"

            for tvar in ['treatment2_num', 'treatment3_num']:
                results.append(extract_results(
                    model, tvar,
                    spec_id=spec_id,
                    spec_tree_path='robustness/sample_restrictions.md',
                    outcome_var=outcome_var,
                    controls_desc=baseline_var,
                    fixed_effects='strata',
                    cluster_var='robust',
                    model_type='OLS with absorbing strata FE',
                    sample_desc=desc
                ))
        except Exception as e:
            print(f"    Error with {desc} for {outcome}: {e}")

# Trimming outliers (alternative to winsorizing)
print("  Running trimmed specifications...")
for outcome in PRIMARY_OUTCOMES:
    baseline_control = f"{outcome}_base"

    df_temp = df[[outcome, baseline_control, 'treatment2_num', 'treatment3_num', 'strata_num']].dropna()

    # Trim top and bottom 1%
    lower = df_temp[outcome].quantile(0.01)
    upper = df_temp[outcome].quantile(0.99)
    df_trimmed = df_temp[(df_temp[outcome] >= lower) & (df_temp[outcome] <= upper)]

    try:
        model = pf.feols(f"{outcome} ~ {baseline_control} + treatment2_num + treatment3_num | strata_num",
                         data=df_trimmed, vcov='hetero')

        for tvar in ['treatment2_num', 'treatment3_num']:
            results.append(extract_results(
                model, tvar,
                spec_id='robust/sample/trim_1pct',
                spec_tree_path='robustness/sample_restrictions.md',
                outcome_var=outcome,
                controls_desc=baseline_control,
                fixed_effects='strata',
                cluster_var='robust',
                model_type='OLS with absorbing strata FE',
                sample_desc='Trimmed at 1% and 99%'
            ))
    except Exception as e:
        print(f"    Error with trimming for {outcome}: {e}")

# ------------------------------------------------------------------------------
# STANDARD ERROR VARIATIONS
# ------------------------------------------------------------------------------
print("\n--- Standard Error Variations ---")

# HC2 and HC3 standard errors
for se_type in ['HC2', 'HC3']:
    print(f"  Running with {se_type} standard errors...")
    for outcome in PRIMARY_OUTCOMES:
        baseline_control = f"{outcome}_base"

        df_temp = df[[outcome, baseline_control, 'treatment2_num', 'treatment3_num', 'strata_num']].dropna()

        # Ensure numeric types
        for col in [outcome, baseline_control, 'treatment2_num', 'treatment3_num']:
            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')

        df_temp = df_temp.dropna()

        # Create strata dummies for statsmodels
        strata_dummies = pd.get_dummies(df_temp['strata_num'], prefix='strata', drop_first=True).astype(float)
        X = df_temp[['treatment2_num', 'treatment3_num', baseline_control]].astype(float)
        X = pd.concat([X, strata_dummies], axis=1)
        X = sm.add_constant(X)
        y = df_temp[outcome].astype(float)

        try:
            model = sm.OLS(y, X).fit(cov_type=se_type)

            for tvar in ['treatment2_num', 'treatment3_num']:
                results.append(extract_results(
                    model, tvar,
                    spec_id=f'robust/se/{se_type.lower()}',
                    spec_tree_path='robustness/clustering_variations.md',
                    outcome_var=outcome,
                    controls_desc=baseline_control,
                    fixed_effects='strata dummies',
                    cluster_var=se_type,
                    model_type=f'OLS with {se_type} SE'
                ))
        except Exception as e:
            print(f"    Error with {se_type} for {outcome}: {e}")

# Clustered SE by branch
print("  Running with branch-clustered standard errors...")
for outcome in PRIMARY_OUTCOMES:
    baseline_control = f"{outcome}_base"

    df_temp = df[[outcome, baseline_control, 'treatment2_num', 'treatment3_num', 'strata_num', 'branch_name']].dropna()
    df_temp['branch_num'] = pd.factorize(df_temp['branch_name'])[0]

    try:
        model = pf.feols(f"{outcome} ~ {baseline_control} + treatment2_num + treatment3_num | strata_num",
                         data=df_temp, vcov={'CRV1': 'branch_num'})

        for tvar in ['treatment2_num', 'treatment3_num']:
            results.append(extract_results(
                model, tvar,
                spec_id='robust/cluster/branch',
                spec_tree_path='robustness/clustering_variations.md',
                outcome_var=outcome,
                controls_desc=baseline_control,
                fixed_effects='strata',
                cluster_var='branch',
                model_type='OLS with branch-clustered SE'
            ))
    except Exception as e:
        print(f"    Error with branch clustering for {outcome}: {e}")

# ------------------------------------------------------------------------------
# LEAVE-ONE-OUT ROBUSTNESS
# ------------------------------------------------------------------------------
print("\n--- Leave-One-Out Robustness ---")

# The main specification only has baseline outcome as control, so we test
# dropping other potential controls that could be added

additional_controls = ['respondent_age_base', 'married_base', 'hh_size_base',
                       'completed_primary_base', 'mobile_account_base', 'hyperbolic_base']

for outcome in PRIMARY_OUTCOMES:
    baseline_control = f"{outcome}_base"

    for dropped in additional_controls:
        if dropped not in df.columns:
            continue

        # Use all controls except the dropped one
        controls_used = [c for c in additional_controls if c != dropped and c in df.columns] + [baseline_control]

        df_temp = df[[outcome, 'treatment2_num', 'treatment3_num', 'strata_num'] + controls_used].dropna()

        try:
            control_str = " + ".join(controls_used)
            model = pf.feols(f"{outcome} ~ treatment2_num + treatment3_num + {control_str} | strata_num",
                             data=df_temp, vcov='hetero')

            for tvar in ['treatment2_num', 'treatment3_num']:
                results.append(extract_results(
                    model, tvar,
                    spec_id=f'robust/loo/drop_{dropped.replace("_base", "")}',
                    spec_tree_path='robustness/leave_one_out.md',
                    outcome_var=outcome,
                    controls_desc=f'All controls except {dropped}',
                    fixed_effects='strata',
                    cluster_var='robust',
                    model_type='OLS with absorbing strata FE'
                ))
        except Exception as e:
            print(f"    Error dropping {dropped} for {outcome}: {e}")

# ------------------------------------------------------------------------------
# SINGLE COVARIATE SPECIFICATIONS
# ------------------------------------------------------------------------------
print("\n--- Single Covariate Specifications ---")

for outcome in PRIMARY_OUTCOMES:
    baseline_control = f"{outcome}_base"

    # Bivariate (no controls, no baseline)
    df_temp = df[[outcome, 'treatment2_num', 'treatment3_num', 'strata_num']].dropna()

    try:
        model = pf.feols(f"{outcome} ~ treatment2_num + treatment3_num | strata_num",
                         data=df_temp, vcov='hetero')

        for tvar in ['treatment2_num', 'treatment3_num']:
            results.append(extract_results(
                model, tvar,
                spec_id='robust/single/none',
                spec_tree_path='robustness/single_covariate.md',
                outcome_var=outcome,
                controls_desc='None (treatment only)',
                fixed_effects='strata',
                cluster_var='robust',
                model_type='OLS with absorbing strata FE'
            ))
    except Exception as e:
        print(f"    Error with bivariate for {outcome}: {e}")

    # Single covariate tests
    single_controls = ['respondent_age_base', 'married_base', 'hh_size_base',
                       'completed_primary_base', 'mobile_account_base', baseline_control]

    for control in single_controls:
        if control not in df.columns:
            continue

        df_temp = df[[outcome, control, 'treatment2_num', 'treatment3_num', 'strata_num']].dropna()

        try:
            model = pf.feols(f"{outcome} ~ treatment2_num + treatment3_num + {control} | strata_num",
                             data=df_temp, vcov='hetero')

            control_name = control.replace('_base', '')
            for tvar in ['treatment2_num', 'treatment3_num']:
                results.append(extract_results(
                    model, tvar,
                    spec_id=f'robust/single/{control_name}',
                    spec_tree_path='robustness/single_covariate.md',
                    outcome_var=outcome,
                    controls_desc=f'{control} only',
                    fixed_effects='strata',
                    cluster_var='robust',
                    model_type='OLS with absorbing strata FE'
                ))
        except Exception as e:
            print(f"    Error with {control} for {outcome}: {e}")

# ------------------------------------------------------------------------------
# HETEROGENEITY SPECIFICATIONS (from Table 2)
# ------------------------------------------------------------------------------
print("\n--- Heterogeneity Specifications ---")

# Self-control and family pressure heterogeneity
hetero_vars = ['hetero_selfc_median', 'hetero_family_median']

for outcome in PRIMARY_OUTCOMES:
    baseline_control = f"{outcome}_base"

    for hetero_var in hetero_vars:
        if hetero_var not in df.columns:
            continue

        df_temp = df[[outcome, baseline_control, 'treatment2_num', 'treatment3_num',
                      'strata_num', hetero_var]].dropna()

        # Create interaction terms
        df_temp['treat2_x_hetero'] = df_temp['treatment2_num'] * df_temp[hetero_var]
        df_temp['treat3_x_hetero'] = df_temp['treatment3_num'] * df_temp[hetero_var]

        try:
            model = pf.feols(f"{outcome} ~ {baseline_control} + treatment2_num + treatment3_num + "
                             f"treat2_x_hetero + treat3_x_hetero + {hetero_var} | strata_num",
                             data=df_temp, vcov='hetero')

            hetero_name = hetero_var.replace('hetero_', '').replace('_median', '')

            # Main effects
            for tvar in ['treatment2_num', 'treatment3_num']:
                results.append(extract_results(
                    model, tvar,
                    spec_id=f'ols/interact/{hetero_name}',
                    spec_tree_path='methods/cross_sectional_ols.md#interaction-effects',
                    outcome_var=outcome,
                    controls_desc=f'{baseline_control} + {hetero_var} + interactions',
                    fixed_effects='strata',
                    cluster_var='robust',
                    model_type='OLS with heterogeneity interactions'
                ))

            # Interaction effects
            for tvar, tname in [('treat2_x_hetero', f'treatment2_x_{hetero_name}'),
                                ('treat3_x_hetero', f'treatment3_x_{hetero_name}')]:
                results.append(extract_results(
                    model, tvar,
                    spec_id=f'ols/interact/{hetero_name}_interaction',
                    spec_tree_path='methods/cross_sectional_ols.md#interaction-effects',
                    outcome_var=outcome,
                    controls_desc=f'{baseline_control} + {hetero_var} + interactions',
                    fixed_effects='strata',
                    cluster_var='robust',
                    model_type='OLS with heterogeneity interactions'
                ))
        except Exception as e:
            print(f"    Error with {hetero_var} for {outcome}: {e}")

# Married/spouse presence heterogeneity
print("  Running spouse presence heterogeneity...")
for outcome in PRIMARY_OUTCOMES:
    baseline_control = f"{outcome}_base"

    if 'no_spouse_home_base' not in df.columns:
        continue

    df_temp = df[[outcome, baseline_control, 'treatment2_num', 'treatment3_num',
                  'strata_num', 'no_spouse_home_base']].dropna()

    df_temp['treat2_x_nospouse'] = df_temp['treatment2_num'] * df_temp['no_spouse_home_base']
    df_temp['treat3_x_nospouse'] = df_temp['treatment3_num'] * df_temp['no_spouse_home_base']

    try:
        model = pf.feols(f"{outcome} ~ {baseline_control} + treatment2_num + treatment3_num + "
                         f"treat2_x_nospouse + treat3_x_nospouse + no_spouse_home_base | strata_num",
                         data=df_temp, vcov='hetero')

        for tvar in ['treatment2_num', 'treatment3_num']:
            results.append(extract_results(
                model, tvar,
                spec_id='ols/interact/spouse_presence',
                spec_tree_path='methods/cross_sectional_ols.md#interaction-effects',
                outcome_var=outcome,
                controls_desc=f'{baseline_control} + no_spouse_home_base + interactions',
                fixed_effects='strata',
                cluster_var='robust',
                model_type='OLS with spouse presence interactions'
            ))
    except Exception as e:
        print(f"    Error with spouse presence for {outcome}: {e}")

# ------------------------------------------------------------------------------
# ROBUSTNESS TO ADDITIONAL CONTROLS (Table A12)
# ------------------------------------------------------------------------------
print("\n--- Robustness to Additional Controls (Table A12 style) ---")

# Control for imbalanced variables
print("  Controlling for imbalanced variables...")
imbalanced_controls = ['mobile_account_base', 'completed_secondary_base', 'hyperbolic_base']

for outcome in PRIMARY_OUTCOMES:
    baseline_control = f"{outcome}_base"

    available = [c for c in imbalanced_controls + [baseline_control] if c in df.columns]
    df_temp = df[[outcome, 'treatment2_num', 'treatment3_num', 'strata_num'] + available].dropna()

    try:
        control_str = " + ".join(available)
        model = pf.feols(f"{outcome} ~ treatment2_num + treatment3_num + {control_str} | strata_num",
                         data=df_temp, vcov='hetero')

        for tvar in ['treatment2_num', 'treatment3_num']:
            results.append(extract_results(
                model, tvar,
                spec_id='robust/controls/imbalanced',
                spec_tree_path='robustness/leave_one_out.md',
                outcome_var=outcome,
                controls_desc='Baseline + imbalanced variables',
                fixed_effects='strata',
                cluster_var='robust',
                model_type='OLS with absorbing strata FE'
            ))
    except Exception as e:
        print(f"    Error with imbalanced controls for {outcome}: {e}")

# Control for time trend
print("  Controlling for time trend...")
for outcome in PRIMARY_OUTCOMES:
    baseline_control = f"{outcome}_base"

    if 'days' not in df.columns:
        continue

    df_temp = df[[outcome, baseline_control, 'treatment2_num', 'treatment3_num',
                  'strata_num', 'days', 'days2']].dropna()

    try:
        model = pf.feols(f"{outcome} ~ {baseline_control} + treatment2_num + treatment3_num + days + days2 | strata_num",
                         data=df_temp, vcov='hetero')

        for tvar in ['treatment2_num', 'treatment3_num']:
            results.append(extract_results(
                model, tvar,
                spec_id='robust/controls/time_trend',
                spec_tree_path='robustness/leave_one_out.md',
                outcome_var=outcome,
                controls_desc='Baseline + linear and quadratic time trend',
                fixed_effects='strata',
                cluster_var='robust',
                model_type='OLS with absorbing strata FE'
            ))
    except Exception as e:
        print(f"    Error with time trend for {outcome}: {e}")

# Control for takeup correlates
print("  Controlling for takeup correlates...")
takeup_controls = ['married_base', 'own_decision_base']

for outcome in PRIMARY_OUTCOMES:
    baseline_control = f"{outcome}_base"

    available = [c for c in takeup_controls + [baseline_control] if c in df.columns]
    df_temp = df[[outcome, 'treatment2_num', 'treatment3_num', 'strata_num'] + available].dropna()

    try:
        control_str = " + ".join(available)
        model = pf.feols(f"{outcome} ~ treatment2_num + treatment3_num + {control_str} | strata_num",
                         data=df_temp, vcov='hetero')

        for tvar in ['treatment2_num', 'treatment3_num']:
            results.append(extract_results(
                model, tvar,
                spec_id='robust/controls/takeup_correlates',
                spec_tree_path='robustness/leave_one_out.md',
                outcome_var=outcome,
                controls_desc='Baseline + takeup correlates',
                fixed_effects='strata',
                cluster_var='robust',
                model_type='OLS with absorbing strata FE'
            ))
    except Exception as e:
        print(f"    Error with takeup correlates for {outcome}: {e}")

# ------------------------------------------------------------------------------
# SECONDARY OUTCOMES
# ------------------------------------------------------------------------------
print("\n--- Secondary Outcomes ---")

secondary_outcomes = [
    # Profits
    ('t_sales', 'monthly_sales'),
    ('sales', 'weekly_sales'),
    ('monthly_profit', 'monthly_profit'),
    ('weekly_profit', 'weekly_profit'),
    # Household
    ('hh_income', 'household_income'),
    ('consumption_total', 'total_consumption'),
    ('consump_food', 'food_consumption'),
    # Labor
    ('total_hoursbusiness', 'total_hours_business'),
    ('hours_week', 'woman_hours_week'),
]

for outcome, outcome_name in secondary_outcomes:
    if outcome not in df.columns:
        continue

    baseline_control = f"{outcome}_base"

    if baseline_control not in df.columns:
        # Run without baseline control
        df_temp = df[[outcome, 'treatment2_num', 'treatment3_num', 'strata_num']].dropna()
        controls_desc = 'None'
        formula = f"{outcome} ~ treatment2_num + treatment3_num | strata_num"
    else:
        df_temp = df[[outcome, baseline_control, 'treatment2_num', 'treatment3_num', 'strata_num']].dropna()
        controls_desc = baseline_control
        formula = f"{outcome} ~ {baseline_control} + treatment2_num + treatment3_num | strata_num"

    try:
        model = pf.feols(formula, data=df_temp, vcov='hetero')

        for tvar in ['treatment2_num', 'treatment3_num']:
            results.append(extract_results(
                model, tvar,
                spec_id='secondary',
                spec_tree_path='methods/cross_sectional_ols.md#baseline',
                outcome_var=outcome,
                controls_desc=controls_desc,
                fixed_effects='strata',
                cluster_var='robust',
                model_type='OLS with absorbing strata FE'
            ))
    except Exception as e:
        print(f"    Error with {outcome}: {e}")

# ==============================================================================
# Save Results
# ==============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved {len(results_df)} specifications to {OUTPUT_PATH}")

# Summary statistics
print("\n--- Summary Statistics ---")
print(f"Total specifications: {len(results_df)}")
print(f"Unique outcomes: {results_df['outcome_var'].nunique()}")
print(f"Unique spec_ids: {results_df['spec_id'].nunique()}")

# Check significance
sig_05 = (results_df['p_value'] < 0.05).sum()
sig_01 = (results_df['p_value'] < 0.01).sum()
positive = (results_df['coefficient'] > 0).sum()

print(f"\nSignificant at 5%: {sig_05} ({100*sig_05/len(results_df):.1f}%)")
print(f"Significant at 1%: {sig_01} ({100*sig_01/len(results_df):.1f}%)")
print(f"Positive coefficients: {positive} ({100*positive/len(results_df):.1f}%)")

print("\n--- Coefficient Summary by Treatment ---")
for tvar in ['treatment2_num', 'treatment3_num']:
    subset = results_df[results_df['treatment_var'] == tvar]
    print(f"\n{tvar}:")
    print(f"  Mean coefficient: {subset['coefficient'].mean():.2f}")
    print(f"  Median coefficient: {subset['coefficient'].median():.2f}")
    print(f"  Range: [{subset['coefficient'].min():.2f}, {subset['coefficient'].max():.2f}]")
    print(f"  % significant at 5%: {100*(subset['p_value'] < 0.05).mean():.1f}%")

print("\nDone!")
