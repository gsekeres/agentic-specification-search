"""
Specification Search for Paper 116433-V1
"Referrals: Peer Screening and Enforcement in a Consumer Credit Field Experiment"
By Bryan, Karlan and Zinman

This paper analyzes a 2x2 factorial RCT examining:
- Enforcement: Whether the referrer is responsible for the referred's loan repayment
- Selection: Whether there's an incentive for the referrer conditional on referral quality

Treatment groups:
1. Approved only (control)
2. Approved + Repay (enforcement only)
3. Repay only (selection only)
4. Repay --> Approved (both enforcement and selection)

Main outcomes: interest, repaid, portion, charged_off

Method: Cross-sectional OLS with experimental treatments
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

# Try to import pyfixest, fall back to statsmodels if not available
try:
    import pyfixest as pf
    HAS_PYFIXEST = True
except ImportError:
    HAS_PYFIXEST = False

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Configuration
PAPER_ID = "116433-V1"
PAPER_TITLE = "Referrals: Peer Screening and Enforcement in a Consumer Credit Field Experiment"
JOURNAL = "AER"
DATA_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/116433-V1/20130234_Data/Referrals_data.dta"
OUTPUT_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/116433-V1"

# Load data
df = pd.read_stata(DATA_PATH)

# Create treatment indicators (matching Stata code)
treatment_map = {'Approved': 1, 'Approved + Repay': 2, 'Repay': 3, 'Repay --> Approved': 4}
df['treatment_num'] = df['treatment'].map(treatment_map)

# Create factorial design variables
df['enforcement'] = ((df['treatment_num'] == 2) | (df['treatment_num'] == 3)).astype(int)
df['selection'] = ((df['treatment_num'] == 3) | (df['treatment_num'] == 4)).astype(int)
df['enforcement_x_selection'] = df['enforcement'] * df['selection']

# Create referrer characteristic variables (matching Table 2/A1 do files)
df['referrer_itcmissing'] = ((df['referrer_itcscore'].isna()) | (df['referrer_itcscore'] == 0)).astype(int)
df['referrer_itcscore'] = df['referrer_itcscore'].fillna(0)

# Job type dummies
df['referrer_gov'] = (df['referrer_jobtype'] == 1).astype(int) if 'referrer_jobtype' in df.columns else 0
df['referrer_clean'] = (df['referrer_jobtype'] == 2).astype(int) if 'referrer_jobtype' in df.columns else 0
df['referrer_secure'] = (df['referrer_jobtype'] == 3).astype(int) if 'referrer_jobtype' in df.columns else 0
df['referrer_retail'] = (df['referrer_jobtype'] == 4).astype(int) if 'referrer_jobtype' in df.columns else 0
df['referrer_it'] = (df['referrer_jobtype'] == 5).astype(int) if 'referrer_jobtype' in df.columns else 0
df['referrer_ag'] = (df['referrer_jobtype'] == 6).astype(int) if 'referrer_jobtype' in df.columns else 0

# Education and salary dummies
df['referrer_education'] = (df['referrer_highedu'].cat.codes >= 2).astype(int) if 'referrer_highedu' in df.columns else 0
df['referrer_salaryM'] = (df['referrer_salaryoccurence'] == 2).astype(int) if 'referrer_salaryoccurence' in df.columns else 0

# Scaled variables
df['referrer_disposableincome000'] = df['referrer_disposableincome'] / 1000
df['referrer_requestedamount000'] = df['referrer_requestedamount'] / 1000

# Female indicator (convert categorical to numeric)
if 'referrer_female' in df.columns:
    df['referrer_female_num'] = df['referrer_female'].cat.codes

# Define control variable sets
CONTROLS_BASIC = ['referrer_female_num', 'referrer_age']
CONTROLS_FULL = ['referrer_female_num', 'referrer_age', 'referrer_education',
                 'referrer_salaryM', 'referrer_disposableincome000', 'referrer_applicationscore',
                 'referrer_itcscore', 'referrer_itcmissing', 'referrer_requestedamount000',
                 'referrer_requestedterm', 'referrer_gov', 'referrer_clean',
                 'referrer_secure', 'referrer_retail', 'referrer_it', 'referrer_ag']

# Main outcome variables
OUTCOMES = ['interest', 'repaid', 'portion', 'charged_off']

# Primary treatment variable for main analysis
PRIMARY_TREATMENT = 'enforcement'  # Main result in Table 5/6

# Results storage
results = []

def run_ols_regression(df_sub, outcome, treatment_vars, controls, cluster_var=None, robust=True):
    """
    Run OLS regression and return results dictionary.
    """
    # Build formula
    all_vars = treatment_vars + controls
    formula = f"{outcome} ~ " + " + ".join(all_vars)

    # Drop missing
    cols_needed = [outcome] + all_vars
    if cluster_var:
        cols_needed.append(cluster_var)
    df_reg = df_sub[cols_needed].dropna()

    if len(df_reg) < 10:
        return None

    try:
        model = smf.ols(formula, data=df_reg)
        if cluster_var and cluster_var in df_reg.columns:
            fit = model.fit(cov_type='cluster', cov_kwds={'groups': df_reg[cluster_var]})
        elif robust:
            fit = model.fit(cov_type='HC1')
        else:
            fit = model.fit()

        return fit, df_reg
    except Exception as e:
        print(f"Error in regression: {e}")
        return None, None

def extract_results(fit, df_reg, treatment_var, outcome, spec_id, spec_tree_path,
                    controls_desc, fixed_effects='None', cluster_var='None', model_type='OLS'):
    """
    Extract results from fitted model into standard format.
    """
    if fit is None:
        return None

    try:
        coef = fit.params.get(treatment_var, np.nan)
        se = fit.bse.get(treatment_var, np.nan)
        tstat = fit.tvalues.get(treatment_var, np.nan)
        pval = fit.pvalues.get(treatment_var, np.nan)
        ci = fit.conf_int().loc[treatment_var] if treatment_var in fit.conf_int().index else [np.nan, np.nan]

        # Build coefficient vector
        coef_vector = {
            "treatment": {
                "var": treatment_var,
                "coef": float(coef),
                "se": float(se),
                "pval": float(pval)
            },
            "controls": []
        }
        for var in fit.params.index:
            if var != treatment_var and var != 'Intercept':
                coef_vector["controls"].append({
                    "var": var,
                    "coef": float(fit.params[var]),
                    "se": float(fit.bse[var]),
                    "pval": float(fit.pvalues[var])
                })

        coef_vector["diagnostics"] = {
            "r_squared": float(fit.rsquared),
            "adj_r_squared": float(fit.rsquared_adj),
            "f_stat": float(fit.fvalue) if hasattr(fit, 'fvalue') else None,
            "f_pval": float(fit.f_pvalue) if hasattr(fit, 'f_pvalue') else None
        }

        return {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': outcome,
            'treatment_var': treatment_var,
            'coefficient': coef,
            'std_error': se,
            't_stat': tstat,
            'p_value': pval,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'n_obs': int(fit.nobs),
            'r_squared': fit.rsquared,
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': f'N={int(fit.nobs)}',
            'fixed_effects': fixed_effects,
            'controls_desc': controls_desc,
            'cluster_var': cluster_var,
            'model_type': model_type,
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
    except Exception as e:
        print(f"Error extracting results: {e}")
        return None

print("="*60)
print("SPECIFICATION SEARCH: 116433-V1")
print("="*60)
print()

# ============================================================================
# BASELINE SPECIFICATIONS (Table 5/6 - Main Results)
# ============================================================================
print("Running baseline specifications (Table 5/6 replication)...")

for outcome in OUTCOMES:
    # Table 6 style: Simple OLS with enforcement and selection
    fit, df_reg = run_ols_regression(df, outcome, ['enforcement', 'selection'], [], robust=True)
    if fit is not None:
        result = extract_results(fit, df_reg, 'enforcement', outcome,
                                'baseline', 'methods/cross_sectional_ols.md#baseline',
                                'None (factorial design)', 'None', 'robust')
        if result:
            results.append(result)
            print(f"  {outcome}: enforcement coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

        # Also extract selection effect
        result_sel = extract_results(fit, df_reg, 'selection', outcome,
                                    'baseline_selection', 'methods/cross_sectional_ols.md#baseline',
                                    'None (factorial design)', 'None', 'robust')
        if result_sel:
            results.append(result_sel)

# ============================================================================
# TABLE 5 - WITH INTERACTION
# ============================================================================
print("\nRunning Table 5 specification (with interaction)...")

for outcome in OUTCOMES:
    fit, df_reg = run_ols_regression(df, outcome,
                                     ['enforcement', 'selection', 'enforcement_x_selection'],
                                     [], robust=True)
    if fit is not None:
        result = extract_results(fit, df_reg, 'enforcement', outcome,
                                'ols/factorial/with_interaction',
                                'methods/cross_sectional_ols.md#baseline',
                                'Factorial design with interaction', 'None', 'robust')
        if result:
            results.append(result)

        # Interaction term
        result_int = extract_results(fit, df_reg, 'enforcement_x_selection', outcome,
                                    'ols/factorial/interaction_term',
                                    'methods/cross_sectional_ols.md#interaction',
                                    'Factorial design with interaction', 'None', 'robust')
        if result_int:
            results.append(result_int)

# ============================================================================
# METHOD-SPECIFIC: CONTROL VARIATIONS (OLS Control Sets)
# ============================================================================
print("\nRunning control set variations...")

# No controls (bivariate)
for outcome in OUTCOMES:
    fit, df_reg = run_ols_regression(df, outcome, ['enforcement', 'selection'], [], robust=True)
    if fit is not None:
        result = extract_results(fit, df_reg, 'enforcement', outcome,
                                'ols/controls/none',
                                'methods/cross_sectional_ols.md#control-sets',
                                'No controls', 'None', 'robust')
        if result:
            results.append(result)

# Basic controls
for outcome in OUTCOMES:
    fit, df_reg = run_ols_regression(df, outcome, ['enforcement', 'selection'],
                                     CONTROLS_BASIC, robust=True)
    if fit is not None:
        result = extract_results(fit, df_reg, 'enforcement', outcome,
                                'ols/controls/demographics',
                                'methods/cross_sectional_ols.md#control-sets',
                                'Demographics only (female, age)', 'None', 'robust')
        if result:
            results.append(result)

# Full controls
for outcome in OUTCOMES:
    fit, df_reg = run_ols_regression(df, outcome, ['enforcement', 'selection'],
                                     CONTROLS_FULL, robust=True)
    if fit is not None:
        result = extract_results(fit, df_reg, 'enforcement', outcome,
                                'ols/controls/full',
                                'methods/cross_sectional_ols.md#control-sets',
                                'Full controls (all referrer characteristics)', 'None', 'robust')
        if result:
            results.append(result)
            print(f"  {outcome} (full controls): enforcement coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# ============================================================================
# METHOD-SPECIFIC: STANDARD ERROR VARIATIONS
# ============================================================================
print("\nRunning standard error variations...")

for outcome in OUTCOMES:
    # Classical SE (no robust)
    fit, df_reg = run_ols_regression(df, outcome, ['enforcement', 'selection'], [], robust=False)
    if fit is not None:
        result = extract_results(fit, df_reg, 'enforcement', outcome,
                                'ols/se/classical',
                                'methods/cross_sectional_ols.md#standard-errors',
                                'No controls', 'None', 'None', 'OLS')
        if result:
            results.append(result)

    # Robust SE (HC1)
    fit, df_reg = run_ols_regression(df, outcome, ['enforcement', 'selection'], [], robust=True)
    if fit is not None:
        result = extract_results(fit, df_reg, 'enforcement', outcome,
                                'ols/se/robust',
                                'methods/cross_sectional_ols.md#standard-errors',
                                'No controls', 'None', 'HC1', 'OLS')
        if result:
            results.append(result)

# ============================================================================
# SUBGROUP ANALYSES
# ============================================================================
print("\nRunning subgroup analyses...")

# By treatment subgroups (enforcement channel pairwise comparisons)
# Treatment 1 vs 2 (no enforcement vs enforcement, no selection)
df_t1t2 = df[df['treatment_num'].isin([1, 2])]
for outcome in OUTCOMES:
    fit, df_reg = run_ols_regression(df_t1t2, outcome, ['enforcement'], [], robust=True)
    if fit is not None:
        result = extract_results(fit, df_reg, 'enforcement', outcome,
                                'ols/sample/enforcement_no_selection',
                                'methods/cross_sectional_ols.md#sample-restrictions',
                                'No controls, treatments 1 vs 2', 'None', 'robust')
        if result:
            results.append(result)
            print(f"  {outcome} (T1 vs T2): enforcement coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

# Treatment 3 vs 4 (no enforcement vs enforcement, with selection)
df_t3t4 = df[df['treatment_num'].isin([3, 4])]
for outcome in OUTCOMES:
    fit, df_reg = run_ols_regression(df_t3t4, outcome, ['enforcement'], [], robust=True)
    if fit is not None:
        result = extract_results(fit, df_reg, 'enforcement', outcome,
                                'ols/sample/enforcement_with_selection',
                                'methods/cross_sectional_ols.md#sample-restrictions',
                                'No controls, treatments 3 vs 4', 'None', 'robust')
        if result:
            results.append(result)

# Selection channel
# Treatment 1 vs 4 (comparing selection effect)
df_t1t4 = df[df['treatment_num'].isin([1, 4])]
for outcome in OUTCOMES:
    fit, df_reg = run_ols_regression(df_t1t4, outcome, ['selection'], [], robust=True)
    if fit is not None:
        result = extract_results(fit, df_reg, 'selection', outcome,
                                'ols/sample/selection_no_enforcement',
                                'methods/cross_sectional_ols.md#sample-restrictions',
                                'No controls, treatments 1 vs 4', 'None', 'robust')
        if result:
            results.append(result)

# Treatment 2 vs 3 (selection with enforcement)
df_t2t3 = df[df['treatment_num'].isin([2, 3])]
for outcome in OUTCOMES:
    fit, df_reg = run_ols_regression(df_t2t3, outcome, ['selection'], [], robust=True)
    if fit is not None:
        result = extract_results(fit, df_reg, 'selection', outcome,
                                'ols/sample/selection_with_enforcement',
                                'methods/cross_sectional_ols.md#sample-restrictions',
                                'No controls, treatments 2 vs 3', 'None', 'robust')
        if result:
            results.append(result)

# ============================================================================
# ROBUSTNESS: LEAVE-ONE-OUT
# ============================================================================
print("\nRunning leave-one-out robustness checks...")

for outcome in OUTCOMES:
    for control in CONTROLS_FULL:
        remaining_controls = [c for c in CONTROLS_FULL if c != control]
        fit, df_reg = run_ols_regression(df, outcome, ['enforcement', 'selection'],
                                         remaining_controls, robust=True)
        if fit is not None:
            result = extract_results(fit, df_reg, 'enforcement', outcome,
                                    f'robust/loo/drop_{control}',
                                    'robustness/leave_one_out.md',
                                    f'Full controls minus {control}', 'None', 'robust')
            if result:
                results.append(result)

# ============================================================================
# ROBUSTNESS: SINGLE COVARIATE
# ============================================================================
print("\nRunning single covariate robustness checks...")

for outcome in OUTCOMES:
    # Bivariate
    fit, df_reg = run_ols_regression(df, outcome, ['enforcement', 'selection'], [], robust=True)
    if fit is not None:
        result = extract_results(fit, df_reg, 'enforcement', outcome,
                                'robust/single/none',
                                'robustness/single_covariate.md',
                                'No controls (bivariate)', 'None', 'robust')
        if result:
            results.append(result)

    # Each control alone
    for control in CONTROLS_FULL:
        fit, df_reg = run_ols_regression(df, outcome, ['enforcement', 'selection'],
                                         [control], robust=True)
        if fit is not None:
            result = extract_results(fit, df_reg, 'enforcement', outcome,
                                    f'robust/single/{control}',
                                    'robustness/single_covariate.md',
                                    f'Only {control}', 'None', 'robust')
            if result:
                results.append(result)

# ============================================================================
# ROBUSTNESS: CLUSTERING VARIATIONS
# ============================================================================
print("\nRunning clustering variations...")

# Cluster by referrer branch
df['referrer_branch_code'] = df['referrer_branchname'].cat.codes
for outcome in OUTCOMES:
    fit, df_reg = run_ols_regression(df, outcome, ['enforcement', 'selection'],
                                     CONTROLS_FULL, cluster_var='referrer_branch_code', robust=True)
    if fit is not None:
        result = extract_results(fit, df_reg, 'enforcement', outcome,
                                'robust/cluster/branch',
                                'robustness/clustering_variations.md#single-level-clustering',
                                'Full controls', 'None', 'referrer_branch', 'OLS')
        if result:
            results.append(result)

# ============================================================================
# ROBUSTNESS: FUNCTIONAL FORM
# ============================================================================
print("\nRunning functional form variations...")

# Log outcome (adding small constant for zeros)
for outcome in OUTCOMES:
    df_temp = df.copy()
    # Outcomes are already proportions, so we use asinh for zeros
    df_temp[f'{outcome}_asinh'] = np.arcsinh(df_temp[outcome])

    fit, df_reg = run_ols_regression(df_temp, f'{outcome}_asinh',
                                     ['enforcement', 'selection'], [], robust=True)
    if fit is not None:
        result = extract_results(fit, df_reg, 'enforcement', f'{outcome}_asinh',
                                'robust/form/y_asinh',
                                'robustness/functional_form.md#outcome-variable-transformations',
                                'No controls, asinh outcome', 'None', 'robust')
        if result:
            result['outcome_var'] = outcome  # Keep original name
            results.append(result)

# Quadratic in age
for outcome in OUTCOMES:
    df_temp = df.copy()
    df_temp['referrer_age_sq'] = df_temp['referrer_age'] ** 2
    controls_quad = ['referrer_female_num', 'referrer_age', 'referrer_age_sq']

    fit, df_reg = run_ols_regression(df_temp, outcome, ['enforcement', 'selection'],
                                     controls_quad, robust=True)
    if fit is not None:
        result = extract_results(fit, df_reg, 'enforcement', outcome,
                                'robust/form/controls_quadratic',
                                'robustness/functional_form.md#control-variable-transformations',
                                'Age and age squared', 'None', 'robust')
        if result:
            results.append(result)

# ============================================================================
# ROBUSTNESS: SAMPLE RESTRICTIONS
# ============================================================================
print("\nRunning sample restriction checks...")

# Exclude extreme outcomes (trim 1% tails)
for outcome in OUTCOMES:
    df_temp = df.copy()
    lower = df_temp[outcome].quantile(0.01)
    upper = df_temp[outcome].quantile(0.99)
    df_trimmed = df_temp[(df_temp[outcome] >= lower) & (df_temp[outcome] <= upper)]

    fit, df_reg = run_ols_regression(df_trimmed, outcome, ['enforcement', 'selection'],
                                     [], robust=True)
    if fit is not None:
        result = extract_results(fit, df_reg, 'enforcement', outcome,
                                'robust/sample/trim_1pct',
                                'robustness/sample_restrictions.md#outlier-handling',
                                'No controls, trimmed 1%', 'None', 'robust')
        if result:
            results.append(result)

# Complete cases only
for outcome in OUTCOMES:
    df_complete = df[CONTROLS_FULL + ['enforcement', 'selection', outcome]].dropna()

    fit, df_reg = run_ols_regression(df_complete, outcome, ['enforcement', 'selection'],
                                     CONTROLS_FULL, robust=True)
    if fit is not None:
        result = extract_results(fit, df_reg, 'enforcement', outcome,
                                'robust/sample/complete_cases',
                                'robustness/sample_restrictions.md#data-quality',
                                'Full controls, complete cases', 'None', 'robust')
        if result:
            results.append(result)

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Convert to DataFrame
results_df = pd.DataFrame(results)
print(f"\nTotal specifications run: {len(results_df)}")

# Save to CSV
output_path = f"{OUTPUT_DIR}/specification_results.csv"
results_df.to_csv(output_path, index=False)
print(f"Saved results to: {output_path}")

# Summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

# Filter to unique enforcement coefficient results
enforcement_results = results_df[results_df['treatment_var'] == 'enforcement'].copy()

print(f"\nTotal enforcement specifications: {len(enforcement_results)}")
print(f"Positive coefficients: {(enforcement_results['coefficient'] > 0).sum()} ({100*(enforcement_results['coefficient'] > 0).mean():.1f}%)")
print(f"Significant at 5%: {(enforcement_results['p_value'] < 0.05).sum()} ({100*(enforcement_results['p_value'] < 0.05).mean():.1f}%)")
print(f"Significant at 1%: {(enforcement_results['p_value'] < 0.01).sum()} ({100*(enforcement_results['p_value'] < 0.01).mean():.1f}%)")
print(f"Median coefficient: {enforcement_results['coefficient'].median():.4f}")
print(f"Mean coefficient: {enforcement_results['coefficient'].mean():.4f}")
print(f"Range: [{enforcement_results['coefficient'].min():.4f}, {enforcement_results['coefficient'].max():.4f}]")

# By outcome
print("\nBy outcome variable:")
for outcome in OUTCOMES:
    outcome_res = enforcement_results[enforcement_results['outcome_var'] == outcome]
    if len(outcome_res) > 0:
        sig_rate = 100 * (outcome_res['p_value'] < 0.05).mean()
        print(f"  {outcome}: N={len(outcome_res)}, median coef={outcome_res['coefficient'].median():.4f}, sig at 5%: {sig_rate:.1f}%")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
