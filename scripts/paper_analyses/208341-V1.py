#!/usr/bin/env python3
"""
Specification Search for Paper 208341-V1
=========================================

Paper: "Land Rental Subsidies and Agricultural Productivity"
(Acampora, Casaburi, Willis - AER)

Study: RCT evaluating land rental subsidies vs cash transfers on agricultural outcomes
Method: ITT (Intent-to-Treat) with reduced form analysis
        The paper uses IV where randomized assignment instruments for actual receipt,
        but since this is an RCT, reduced form (ITT) effects are the key causal estimates
Main outcomes: Cultivation, input use, labor, output value, value added

This script runs 50+ specifications following the i4r methodology.
"""

import pandas as pd
import numpy as np
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

# ==========================================
# CONFIGURATION
# ==========================================

PAPER_ID = "208341-V1"
JOURNAL = "AER"
PAPER_TITLE = "Land Rental Subsidies and Agricultural Productivity"
PACKAGE_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/208341-V1"
OUTPUT_DIR = PACKAGE_DIR
SCRIPT_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/scripts/paper_analyses/208341-V1.py"

# Method classification
METHOD_CODE = "cross_sectional_ols"  # RCT with ITT analysis
METHOD_TREE_PATH = "specification_tree/methods/cross_sectional_ols.md"

# ==========================================
# DATA LOADING
# ==========================================

print("Loading data...")
df = pd.read_stata(f"{PACKAGE_DIR}/AER_dta_cleaned/merged_data_target_plot_AER.dta")

# Convert all numeric columns to float64 to avoid dtype issues
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df[col] = df[col].astype(np.float64)

# Define key variables
PRIMARY_OUTCOME = 'ETwadj_ag_va1_r6_qaB_1'  # Value Added
PRIMARY_TREATMENT = 'rental_subsidy'

OUTCOME_VARS = {
    'ETd2_1_plot_use_cltvtd_1': 'Plot Cultivated',
    'ETd34_ag_inputs1_B_1': 'Input Value',
    'ETL_val_1': 'Labor Value',
    'ETe1_3_h_value1_qa_1': 'Output Value',
    'ETwadj_ag_va1_r6_qaB_1': 'Value Added'
}

# Baseline controls from the do file
BASELINE_CONTROLS = ['L_target_plot_size_mean', 'Bd2_1_cltvtd_2019long_1', 'Be2_1_cltvtd_18SR_1',
                     'Bd34_ag_inputs1_1', 'Bwadj_vd6_3_L_hhdays_r6_1', 'Bvd6_6_L_hire_days_1',
                     'Be2_8_SR_h_value_1']

# ==========================================
# DATA PREPARATION
# ==========================================

print("Preparing data...")

# Create tags for first observation per farmer
df['tag_fin'] = ~df.duplicated(subset=['fin'], keep='first')

# Filter to existing baseline controls
existing_controls = [c for c in BASELINE_CONTROLS if c in df.columns]

# Fill missing baseline controls and create dummies
for var in existing_controls:
    df[f'missing_{var}'] = df[var].isna().astype(float)
    df[var] = df[var].fillna(df[var].median())  # Use median imputation instead of -999

full_controls = existing_controls.copy()

# Create strata dummies - use a subset to avoid collinearity
strata_unique = df['stratum_reg'].dropna().unique()
# Create dummies for the first N-1 strata
for s in sorted(strata_unique)[:-1]:  # Drop last to avoid collinearity
    df[f'strata_{int(s)}'] = (df['stratum_reg'] == s).astype(float)
strata_cols = [f'strata_{int(s)}' for s in sorted(strata_unique)[:-1]]

# Create round dummies
round_unique = df['endline_round'].dropna().unique()
for r in sorted(round_unique)[:-1]:  # Drop last
    df[f'round_{int(r)}'] = (df['endline_round'] == r).astype(float)
round_cols = [f'round_{int(r)}' for r in sorted(round_unique)[:-1]]

# Create IHS transformations
for outcome in ['ETd34_ag_inputs1_B_1', 'ETL_val_1', 'ETe1_3_h_value1_qa_1', 'ETwadj_ag_va1_r6_qaB_1']:
    if outcome in df.columns:
        df[f'ihs_{outcome}'] = np.arcsinh(df[outcome])

# Create log transformations
for outcome in ['ETd34_ag_inputs1_B_1', 'ETL_val_1', 'ETe1_3_h_value1_qa_1']:
    if outcome in df.columns:
        df[f'log_{outcome}'] = np.log(df[outcome] + 1)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def winsorize(series, lower=0.01, upper=0.99):
    """Winsorize series at given percentiles."""
    lower_val = series.quantile(lower)
    upper_val = series.quantile(upper)
    return series.clip(lower=lower_val, upper=upper_val)

def run_regression(data, outcome, treatment_var, controls_list, fe_cols,
                   spec_id, spec_tree_path, sample_desc, fe_desc, controls_desc,
                   cluster_var='fin'):
    """Run OLS regression with fixed effects."""

    # Build complete variable list
    all_vars = [outcome, treatment_var] + controls_list + fe_cols
    if cluster_var:
        all_vars.append(cluster_var)
    all_vars = [v for v in all_vars if v in data.columns]

    # Clean data
    data_clean = data[all_vars].dropna().copy()

    if len(data_clean) < 50:
        print(f"Skipping {spec_id}: insufficient observations ({len(data_clean)})")
        return None

    try:
        y = data_clean[outcome].astype(float)

        # Build X matrix
        X_vars = [treatment_var] + [c for c in controls_list if c in data_clean.columns] + \
                 [f for f in fe_cols if f in data_clean.columns]
        X = data_clean[X_vars].astype(float)
        X = sm.add_constant(X)

        # Remove columns with zero variance
        X = X.loc[:, X.std() > 0]

        if treatment_var not in X.columns:
            print(f"Skipping {spec_id}: treatment variable dropped")
            return None

        model = OLS(y, X)

        if cluster_var and cluster_var in data_clean.columns:
            result = model.fit(cov_type='cluster', cov_kwds={'groups': data_clean[cluster_var]})
        else:
            result = model.fit(cov_type='HC1')

        coef = result.params[treatment_var]
        se = result.bse[treatment_var]
        t_stat = result.tvalues[treatment_var]
        pval = result.pvalues[treatment_var]
        ci_lower, ci_upper = result.conf_int().loc[treatment_var]
        n_obs = int(result.nobs)
        r_squared = result.rsquared

        coef_dict = {
            'treatment': {
                'var': treatment_var,
                'coef': float(coef),
                'se': float(se),
                'pval': float(pval)
            },
            'controls': [],
            'fixed_effects': fe_desc.split(', ') if fe_desc else [],
            'diagnostics': {}
        }

        # Add other coefficients (limit to non-FE)
        for var in result.params.index:
            if var not in [treatment_var, 'const'] and not var.startswith('strata_') and not var.startswith('round_'):
                coef_dict['controls'].append({
                    'var': var,
                    'coef': float(result.params[var]),
                    'se': float(result.bse[var]),
                    'pval': float(result.pvalues[var])
                })

        return {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': outcome,
            'treatment_var': treatment_var,
            'coefficient': float(coef),
            'std_error': float(se),
            't_stat': float(t_stat),
            'p_value': float(pval),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_obs': n_obs,
            'r_squared': float(r_squared) if not np.isnan(r_squared) else None,
            'coefficient_vector_json': json.dumps(coef_dict),
            'sample_desc': sample_desc,
            'fixed_effects': fe_desc,
            'controls_desc': controls_desc,
            'cluster_var': cluster_var if cluster_var else 'none',
            'model_type': 'OLS',
            'estimation_script': SCRIPT_PATH
        }
    except Exception as e:
        print(f"Error in {spec_id}: {e}")
        return None

# ==========================================
# SPECIFICATION SEARCH
# ==========================================

results = []

print("\n" + "="*60)
print("RUNNING SPECIFICATION SEARCH")
print("="*60)

# ==========================================
# 1. BASELINE SPECIFICATIONS (ITT)
# ==========================================
print("\n1. Baseline Specifications (ITT)...")

# Main ITT effect for all outcomes
for outcome_name, outcome_label in OUTCOME_VARS.items():
    result = run_regression(
        df, outcome_name, 'rental_subsidy',
        full_controls, strata_cols + round_cols,
        f'baseline/{outcome_name}',
        'methods/cross_sectional_ols.md#baseline',
        'Full sample, all rounds',
        'stratum_reg, endline_round',
        'Full baseline controls'
    )
    if result:
        results.append(result)

# ==========================================
# 2. CASH DROP EFFECTS (ITT)
# ==========================================
print("\n2. Cash Drop Effects (ITT)...")

for outcome_name, outcome_label in OUTCOME_VARS.items():
    result = run_regression(
        df, outcome_name, 'cash_drop',
        full_controls, strata_cols + round_cols,
        f'itt/cash_drop/{outcome_name}',
        'methods/cross_sectional_ols.md#baseline',
        'Cash drop ITT effect',
        'stratum_reg, endline_round',
        'Full baseline controls'
    )
    if result:
        results.append(result)

# ==========================================
# 3. CONTROL VARIATIONS
# ==========================================
print("\n3. Control Variations...")

# 3a. No controls (just FE)
for outcome_name in ['ETwadj_ag_va1_r6_qaB_1', 'ETd2_1_plot_use_cltvtd_1']:
    result = run_regression(
        df, outcome_name, 'rental_subsidy',
        [], strata_cols + round_cols,
        f'robust/control/none/{outcome_name}',
        'robustness/control_progression.md',
        'No baseline controls',
        'stratum_reg, endline_round',
        'No controls (FE only)'
    )
    if result:
        results.append(result)

# 3b. Plot size only
result = run_regression(
    df, PRIMARY_OUTCOME, 'rental_subsidy',
    ['L_target_plot_size_mean'], strata_cols + round_cols,
    'robust/control/plot_size_only',
    'robustness/control_progression.md',
    'Plot size control only',
    'stratum_reg, endline_round',
    'Plot size only'
)
if result:
    results.append(result)

# 3c. Leave-one-out controls
for control in existing_controls:
    remaining_controls = [c for c in full_controls if c != control]
    result = run_regression(
        df, PRIMARY_OUTCOME, 'rental_subsidy',
        remaining_controls, strata_cols + round_cols,
        f'robust/control/drop_{control}',
        'robustness/leave_one_out.md',
        f'Dropped {control}',
        'stratum_reg, endline_round',
        f'Dropped {control}'
    )
    if result:
        results.append(result)

# 3d. Add controls incrementally
for i in range(1, len(existing_controls)+1):
    controls_subset = existing_controls[:i]
    result = run_regression(
        df, PRIMARY_OUTCOME, 'rental_subsidy',
        controls_subset, strata_cols + round_cols,
        f'robust/control/add_{i}_controls',
        'robustness/control_progression.md',
        f'First {i} controls',
        'stratum_reg, endline_round',
        f'{i} controls'
    )
    if result:
        results.append(result)

# ==========================================
# 4. SAMPLE RESTRICTIONS
# ==========================================
print("\n4. Sample Restrictions...")

# 4a. By endline round
for round_num in df['endline_round'].dropna().unique():
    df_round = df[df['endline_round'] == round_num].copy()
    result = run_regression(
        df_round, PRIMARY_OUTCOME, 'rental_subsidy',
        full_controls, strata_cols,  # No round FE for single round
        f'robust/sample/round_{int(round_num)}',
        'robustness/sample_restrictions.md',
        f'Endline round {int(round_num)} only',
        'stratum_reg',
        'Full baseline controls'
    )
    if result:
        results.append(result)

# 4b. By stratum
for stratum, stratum_name in [(1, 'stratum_C'), (2, 'stratum_NC')]:
    df_stratum = df[df['stratum_C_NC'] == stratum].copy()
    result = run_regression(
        df_stratum, PRIMARY_OUTCOME, 'rental_subsidy',
        full_controls, strata_cols + round_cols,
        f'robust/sample/{stratum_name}',
        'robustness/sample_restrictions.md',
        f'{stratum_name} only',
        'stratum_reg, endline_round',
        'Full baseline controls'
    )
    if result:
        results.append(result)

# 4c. Early vs late rounds
df_early = df[df['endline_round'] <= 2].copy()
df_late = df[df['endline_round'] >= 3].copy()

result = run_regression(
    df_early, PRIMARY_OUTCOME, 'rental_subsidy',
    full_controls, strata_cols + round_cols,
    'robust/sample/early_rounds',
    'robustness/sample_restrictions.md',
    'Rounds 1-2 only',
    'stratum_reg, endline_round',
    'Full baseline controls'
)
if result:
    results.append(result)

result = run_regression(
    df_late, PRIMARY_OUTCOME, 'rental_subsidy',
    full_controls, strata_cols + round_cols,
    'robust/sample/late_rounds',
    'robustness/sample_restrictions.md',
    'Rounds 3-4 only',
    'stratum_reg, endline_round',
    'Full baseline controls'
)
if result:
    results.append(result)

# 4d. Winsorization
for pct in [1, 5, 10]:
    df_wins = df.copy()
    df_wins[f'{PRIMARY_OUTCOME}_wins'] = winsorize(df_wins[PRIMARY_OUTCOME], pct/100, 1-pct/100)

    result = run_regression(
        df_wins, f'{PRIMARY_OUTCOME}_wins', 'rental_subsidy',
        full_controls, strata_cols + round_cols,
        f'robust/sample/winsorize_{pct}pct',
        'robustness/sample_restrictions.md',
        f'Outcome winsorized at {pct}%',
        'stratum_reg, endline_round',
        'Full baseline controls'
    )
    if result:
        results.append(result)

# 4e. Trim extreme values
df_trimmed = df[(df[PRIMARY_OUTCOME] > df[PRIMARY_OUTCOME].quantile(0.01)) &
                (df[PRIMARY_OUTCOME] < df[PRIMARY_OUTCOME].quantile(0.99))].copy()
result = run_regression(
    df_trimmed, PRIMARY_OUTCOME, 'rental_subsidy',
    full_controls, strata_cols + round_cols,
    'robust/sample/trim_1pct',
    'robustness/sample_restrictions.md',
    'Trimmed top and bottom 1%',
    'stratum_reg, endline_round',
    'Full baseline controls'
)
if result:
    results.append(result)

# ==========================================
# 5. ALTERNATIVE OUTCOMES (Functional Form)
# ==========================================
print("\n5. Alternative Outcomes (Functional Form)...")

# 5a. IHS transformations
for outcome in ['ETd34_ag_inputs1_B_1', 'ETL_val_1', 'ETe1_3_h_value1_qa_1', 'ETwadj_ag_va1_r6_qaB_1']:
    ihs_var = f'ihs_{outcome}'
    if ihs_var in df.columns:
        result = run_regression(
            df, ihs_var, 'rental_subsidy',
            full_controls, strata_cols + round_cols,
            f'robust/funcform/ihs_{outcome}',
            'robustness/functional_form.md',
            f'IHS transformation of {outcome}',
            'stratum_reg, endline_round',
            'Full baseline controls'
        )
        if result:
            results.append(result)

# 5b. Log transformations
for outcome in ['ETd34_ag_inputs1_B_1', 'ETL_val_1', 'ETe1_3_h_value1_qa_1']:
    log_var = f'log_{outcome}'
    if log_var in df.columns:
        result = run_regression(
            df, log_var, 'rental_subsidy',
            full_controls, strata_cols + round_cols,
            f'robust/funcform/log_{outcome}',
            'robustness/functional_form.md',
            f'Log transformation of {outcome}',
            'stratum_reg, endline_round',
            'Full baseline controls'
        )
        if result:
            results.append(result)

# ==========================================
# 6. INFERENCE VARIATIONS
# ==========================================
print("\n6. Inference Variations...")

# 6a. Robust SE (no clustering)
result = run_regression(
    df, PRIMARY_OUTCOME, 'rental_subsidy',
    full_controls, strata_cols + round_cols,
    'robust/cluster/robust_only',
    'robustness/clustering_variations.md',
    'Robust SE, no clustering',
    'stratum_reg, endline_round',
    'Full baseline controls',
    cluster_var=None
)
if result:
    results.append(result)

# 6b. Cluster by stratum
result = run_regression(
    df, PRIMARY_OUTCOME, 'rental_subsidy',
    full_controls, strata_cols + round_cols,
    'robust/cluster/stratum',
    'robustness/clustering_variations.md',
    'Clustered by stratum',
    'stratum_reg, endline_round',
    'Full baseline controls',
    cluster_var='stratum_reg'
)
if result:
    results.append(result)

# ==========================================
# 7. FIXED EFFECTS VARIATIONS
# ==========================================
print("\n7. Fixed Effects Variations...")

# 7a. Strata FE only (no round FE)
result = run_regression(
    df, PRIMARY_OUTCOME, 'rental_subsidy',
    full_controls, strata_cols,
    'robust/fe/strata_only',
    'robustness/model_specification.md',
    'Strata FE only',
    'stratum_reg',
    'Full baseline controls'
)
if result:
    results.append(result)

# 7b. Round FE only (no strata FE)
result = run_regression(
    df, PRIMARY_OUTCOME, 'rental_subsidy',
    full_controls, round_cols,
    'robust/fe/round_only',
    'robustness/model_specification.md',
    'Round FE only',
    'endline_round',
    'Full baseline controls'
)
if result:
    results.append(result)

# 7c. No fixed effects
result = run_regression(
    df, PRIMARY_OUTCOME, 'rental_subsidy',
    full_controls, [],
    'robust/fe/none',
    'robustness/model_specification.md',
    'No fixed effects',
    'None',
    'Full baseline controls'
)
if result:
    results.append(result)

# ==========================================
# 8. HETEROGENEITY ANALYSIS
# ==========================================
print("\n8. Heterogeneity Analysis...")

# Create subgroups
median_plot_size = df['L_target_plot_size_mean'].median()
df['large_plot'] = (df['L_target_plot_size_mean'] > median_plot_size).astype(float)
df['high_baseline_cult'] = (df['Bd2_1_cltvtd_2019long_1'] > 0.5).astype(float)

# By plot size
for outcome in ['ETwadj_ag_va1_r6_qaB_1', 'ETd2_1_plot_use_cltvtd_1']:
    for size, size_name in [(1, 'large_plot'), (0, 'small_plot')]:
        df_size = df[df['large_plot'] == size].copy()
        result = run_regression(
            df_size, outcome, 'rental_subsidy',
            full_controls, strata_cols + round_cols,
            f'robust/heterogeneity/{size_name}/{outcome}',
            'robustness/heterogeneity.md',
            f'{size_name} subsample',
            'stratum_reg, endline_round',
            'Full baseline controls'
        )
        if result:
            results.append(result)

# By baseline cultivation
for cult, cult_name in [(1, 'high_baseline_cult'), (0, 'low_baseline_cult')]:
    df_cult = df[df['high_baseline_cult'] == cult].copy()
    result = run_regression(
        df_cult, PRIMARY_OUTCOME, 'rental_subsidy',
        full_controls, strata_cols + round_cols,
        f'robust/heterogeneity/{cult_name}',
        'robustness/heterogeneity.md',
        f'{cult_name}',
        'stratum_reg, endline_round',
        'Full baseline controls'
    )
    if result:
        results.append(result)

# ==========================================
# 9. FIRST STAGE / MECHANISM
# ==========================================
print("\n9. First Stage / Mechanism...")

# Effect on actual receipt
result = run_regression(
    df, 'any_PUsubsidy_paid', 'rental_subsidy',
    full_controls, strata_cols + round_cols,
    'robust/first_stage/subsidy',
    'methods/instrumental_variables.md#first-stage',
    'First stage: subsidy received',
    'stratum_reg, endline_round',
    'Full baseline controls'
)
if result:
    results.append(result)

result = run_regression(
    df, 'any_PUcash_paid', 'cash_drop',
    full_controls, strata_cols + round_cols,
    'robust/first_stage/cash',
    'methods/instrumental_variables.md#first-stage',
    'First stage: cash received',
    'stratum_reg, endline_round',
    'Full baseline controls'
)
if result:
    results.append(result)

# Effect on plot rented
if 'PUrented' in df.columns:
    result = run_regression(
        df, 'PUrented', 'rental_subsidy',
        full_controls, strata_cols + round_cols,
        'robust/mechanism/plot_rented',
        'robustness/measurement.md',
        'Mechanism: plot rented',
        'stratum_reg, endline_round',
        'Full baseline controls'
    )
    if result:
        results.append(result)

# ==========================================
# 10. PLACEBO TESTS
# ==========================================
print("\n10. Placebo Tests...")

# Using baseline outcome as placebo (should be zero effect)
baseline_outcomes = ['Bd2_1_cltvtd_2019long_1', 'Bd34_ag_inputs1_1', 'Be2_8_SR_h_value_1']
for baseline_out in baseline_outcomes:
    if baseline_out in df.columns:
        df_placebo = df[df['tag_fin'] == True].copy()
        result = run_regression(
            df_placebo, baseline_out, 'rental_subsidy',
            ['L_target_plot_size_mean'], strata_cols,
            f'robust/placebo/baseline_{baseline_out}',
            'robustness/placebo_tests.md',
            'Placebo: baseline outcome',
            'stratum_reg',
            'Plot size only'
        )
        if result:
            results.append(result)

# ==========================================
# 11. ADDITIONAL OUTCOMES
# ==========================================
print("\n11. Additional Outcomes...")

additional_outcomes = {
    'ETd2_2_plot_crops_22_1': 'Maize Cultivation',
    'ETd2_2_plot_crops_C_1': 'Commercial Crops',
    'ETd3_2_seed_value1_B_1': 'Seed Value',
    'ETd3_4_seed_improved_1': 'Improved Seeds'
}

for outcome, label in additional_outcomes.items():
    if outcome in df.columns:
        result = run_regression(
            df, outcome, 'rental_subsidy',
            full_controls, strata_cols + round_cols,
            f'robust/outcome/additional/{outcome}',
            'robustness/measurement.md',
            f'Alternative outcome: {label}',
            'stratum_reg, endline_round',
            'Full baseline controls'
        )
        if result:
            results.append(result)

# ==========================================
# 12. CROSS-TREATMENT COMPARISONS
# ==========================================
print("\n12. Cross-Treatment Comparisons...")

# Test rental vs cash effect difference
# Add both treatments to single regression
for outcome_name in ['ETwadj_ag_va1_r6_qaB_1', 'ETd2_1_plot_use_cltvtd_1']:
    try:
        data_clean = df.dropna(subset=[outcome_name, 'rental_subsidy', 'cash_drop'] + full_controls).copy()
        y = data_clean[outcome_name].astype(float)

        X_vars = ['rental_subsidy', 'cash_drop'] + full_controls + strata_cols + round_cols
        X_vars = [v for v in X_vars if v in data_clean.columns]
        X = data_clean[X_vars].astype(float)
        X = sm.add_constant(X)
        X = X.loc[:, X.std() > 0]

        model = OLS(y, X)
        result_obj = model.fit(cov_type='cluster', cov_kwds={'groups': data_clean['fin']})

        # Test difference: rental - cash
        from scipy.stats import t as t_dist
        rental_coef = result_obj.params['rental_subsidy']
        cash_coef = result_obj.params['cash_drop']
        diff = rental_coef - cash_coef

        # Get variance-covariance matrix
        vcov = result_obj.cov_params()
        var_diff = vcov.loc['rental_subsidy', 'rental_subsidy'] + vcov.loc['cash_drop', 'cash_drop'] - 2 * vcov.loc['rental_subsidy', 'cash_drop']
        se_diff = np.sqrt(var_diff)
        t_stat_diff = diff / se_diff
        pval_diff = 2 * (1 - t_dist.cdf(abs(t_stat_diff), result_obj.df_resid))

        result = {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': f'robust/comparison/rental_vs_cash/{outcome_name}',
            'spec_tree_path': 'robustness/heterogeneity.md',
            'outcome_var': outcome_name,
            'treatment_var': 'rental_minus_cash',
            'coefficient': float(diff),
            'std_error': float(se_diff),
            't_stat': float(t_stat_diff),
            'p_value': float(pval_diff),
            'ci_lower': float(diff - 1.96 * se_diff),
            'ci_upper': float(diff + 1.96 * se_diff),
            'n_obs': int(result_obj.nobs),
            'r_squared': float(result_obj.rsquared),
            'coefficient_vector_json': json.dumps({
                'treatment': {'var': 'rental_minus_cash', 'coef': float(diff), 'se': float(se_diff), 'pval': float(pval_diff)},
                'controls': [],
                'fixed_effects': ['stratum_reg', 'endline_round'],
                'diagnostics': {'rental_coef': float(rental_coef), 'cash_coef': float(cash_coef)}
            }),
            'sample_desc': 'Rental vs Cash comparison',
            'fixed_effects': 'stratum_reg, endline_round',
            'controls_desc': 'Full baseline controls',
            'cluster_var': 'fin',
            'model_type': 'OLS',
            'estimation_script': SCRIPT_PATH
        }
        results.append(result)
    except Exception as e:
        print(f"Error in comparison spec for {outcome_name}: {e}")

# ==========================================
# SAVE RESULTS
# ==========================================

print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

results_df = pd.DataFrame(results)

output_path = f"{OUTPUT_DIR}/specification_results.csv"
results_df.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")
print(f"Total specifications run: {len(results_df)}")

# ==========================================
# SUMMARY STATISTICS
# ==========================================

if len(results_df) > 0:
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    print(f"\nTotal specifications: {len(results_df)}")
    print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({(results_df['coefficient'] > 0).mean()*100:.1f}%)")
    print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({(results_df['p_value'] < 0.05).mean()*100:.1f}%)")
    print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({(results_df['p_value'] < 0.01).mean()*100:.1f}%)")
    print(f"\nCoefficient range: [{results_df['coefficient'].min():.3f}, {results_df['coefficient'].max():.3f}]")
    print(f"Median coefficient: {results_df['coefficient'].median():.3f}")
    print(f"Mean coefficient: {results_df['coefficient'].mean():.3f}")

    # Breakdown by category
    print("\n" + "-"*40)
    print("Breakdown by Specification Category:")
    print("-"*40)

    results_df['category'] = results_df['spec_id'].apply(lambda x: x.split('/')[0])
    category_summary = results_df.groupby('category').agg({
        'coefficient': ['count', 'mean', 'std'],
        'p_value': lambda x: (x < 0.05).mean()
    }).round(3)
    category_summary.columns = ['N', 'Mean Coef', 'Std Coef', '% Sig 5%']
    print(category_summary.to_string())

print("\n" + "="*60)
print("SPECIFICATION SEARCH COMPLETE")
print("="*60)
