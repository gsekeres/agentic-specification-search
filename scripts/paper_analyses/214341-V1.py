"""
Specification Search: Stanton and Thomas "Who Benefits from Online Gig Economy Platforms"
Paper ID: 214341-V1
Journal: AER

This script analyzes the publicly available survey data from the replication package.

Main Hypothesis: Workers on online gig economy platforms earn markups above their
willingness to accept (WTA), with surplus varying by worker characteristics.

Note: Most of the paper's analyses rely on confidential platform data that is not
publicly available. This analysis focuses on Tables 7 and A1 which use the survey data.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
PAPER_ID = "214341-V1"
JOURNAL = "AER"
PAPER_TITLE = "Who Benefits from Online Gig Economy Platforms"
DATA_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/214341-V1/StantonThomas_AER_Replication/PublicData/SurveyData_PublicArchive_DeIdentified.dta"
OUTPUT_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/214341-V1/StantonThomas_AER_Replication"

# =============================================================================
# Load Data
# =============================================================================
df_raw = pd.read_stata(DATA_PATH)

# Define samples
df_full = df_raw[df_raw['numjobs'] > 0].copy()  # Exclude workers with no jobs (as in original)
df_survey = df_raw[(df_raw['InSurvey'] == 1) & (df_raw['numjobs'] > 0)].copy()

# Create additional variables
df_survey['log_numjobs'] = np.log(df_survey['numjobs'] + 1)
df_survey['log_rate'] = np.log(df_survey['scraped_profile_rate'] + 1)
df_survey['experienced'] = (df_survey['numjobs'] >= 5).astype(int)
df_survey['high_rate'] = (df_survey['scraped_profile_rate'] >= 20).astype(int)

# =============================================================================
# Results Container
# =============================================================================
results = []

def create_result_dict(spec_id, spec_tree_path, outcome_var, treatment_var, model,
                       sample_desc, fixed_effects, controls_desc, cluster_var,
                       model_type, n_obs, r_squared=None):
    """Create standardized result dictionary."""

    # Get coefficient info for treatment variable
    if treatment_var in model.params.index:
        coef = model.params[treatment_var]
        se = model.bse[treatment_var]
        tstat = model.tvalues[treatment_var]
        pval = model.pvalues[treatment_var]
        ci = model.conf_int().loc[treatment_var]
        ci_lower, ci_upper = ci[0], ci[1]
    else:
        # For intercept-only models
        coef = model.params['Intercept'] if 'Intercept' in model.params.index else model.params.iloc[0]
        se = model.bse.iloc[0]
        tstat = model.tvalues.iloc[0]
        pval = model.pvalues.iloc[0]
        ci = model.conf_int().iloc[0]
        ci_lower, ci_upper = ci[0], ci[1]

    # Full coefficient vector
    coef_vector = {
        "treatment": {
            "var": treatment_var,
            "coef": float(coef),
            "se": float(se),
            "pval": float(pval)
        },
        "controls": [],
        "fixed_effects": [],
        "diagnostics": {
            "first_stage_F": None,
            "overid_pval": None,
            "hausman_pval": None
        }
    }

    # Add other coefficients as controls
    for var in model.params.index:
        if var != treatment_var and var != 'Intercept':
            coef_vector["controls"].append({
                "var": var,
                "coef": float(model.params[var]),
                "se": float(model.bse[var]),
                "pval": float(model.pvalues[var])
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
        't_stat': float(tstat),
        'p_value': float(pval),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'n_obs': int(n_obs),
        'r_squared': float(r_squared) if r_squared is not None else None,
        'coefficient_vector_json': json.dumps(coef_vector),
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    }

# =============================================================================
# Analysis 1: Descriptive Statistics - Surplus Markups (Table 7 replication)
# =============================================================================
print("=" * 70)
print("Analysis 1: Surplus Markups - Descriptive Statistics")
print("=" * 70)

outcome_vars = {
    'hrly_surp_rel_min': 'Markup Relative to WTA',
    'hrly_surp_rel_expost': 'Markup Relative to Ex-Post WTA',
    'hrly_surp_rel_outside': 'Markup Relative to Outside Wage',
    'fixed_surp_rel_expost': 'Fixed-Price Markup Relative to Ex-Post WTA'
}

for outcome, label in outcome_vars.items():
    temp_df = df_survey[df_survey[outcome].notna()].copy()
    if len(temp_df) > 0:
        # Simple mean test (intercept-only model)
        y = temp_df[outcome]
        model = sm.OLS(y, sm.add_constant(np.ones(len(y)))).fit()

        spec_id = f"baseline/mean/{outcome.replace('hrly_surp_rel_', '').replace('fixed_surp_rel_', 'fixed_')}"
        result = create_result_dict(
            spec_id=spec_id,
            spec_tree_path="methods/cross_sectional_ols.md#baseline",
            outcome_var=outcome,
            treatment_var="const",
            model=model,
            sample_desc=f"Survey participants with valid {outcome}",
            fixed_effects="None",
            controls_desc="None (intercept-only)",
            cluster_var=None,
            model_type="OLS",
            n_obs=len(temp_df),
            r_squared=model.rsquared
        )
        results.append(result)

        print(f"\n{label}:")
        print(f"  N = {len(temp_df)}")
        print(f"  Mean = {y.mean():.4f} (SE = {y.std()/np.sqrt(len(y)):.4f})")
        print(f"  Std Dev = {y.std():.4f}")
        print(f"  One-sample t-test (H0: mean=1): t = {(y.mean()-1)/(y.std()/np.sqrt(len(y))):.3f}, p = {stats.ttest_1samp(y, 1).pvalue:.4f}")

# =============================================================================
# Analysis 2: Survey Balance Table (Table A1 replication)
# =============================================================================
print("\n" + "=" * 70)
print("Analysis 2: Platform Workers vs Survey Participants Balance")
print("=" * 70)

balance_vars = ['numjobs', 'scraped_profile_rate', 'c_US']

for var in balance_vars:
    # Comparison: Platform participants vs Survey participants
    temp_df = df_full[df_full[var].notna()].copy()

    # Run regression of variable on InSurvey indicator
    model = smf.ols(f'{var} ~ InSurvey', data=temp_df).fit(cov_type='HC1')

    result = create_result_dict(
        spec_id=f"balance/{var}",
        spec_tree_path="methods/cross_sectional_ols.md#baseline",
        outcome_var=var,
        treatment_var="InSurvey",
        model=model,
        sample_desc="All platform participants with numjobs > 0",
        fixed_effects="None",
        controls_desc="None",
        cluster_var=None,
        model_type="OLS with robust SE",
        n_obs=len(temp_df),
        r_squared=model.rsquared
    )
    results.append(result)

    print(f"\n{var}:")
    print(f"  Platform mean: {temp_df[temp_df['InSurvey']==0][var].mean():.3f}")
    print(f"  Survey mean: {temp_df[temp_df['InSurvey']==1][var].mean():.3f}")
    print(f"  Difference: {model.params['InSurvey']:.3f} (SE = {model.bse['InSurvey']:.3f})")
    print(f"  p-value: {model.pvalues['InSurvey']:.4f}")

# =============================================================================
# Analysis 3: Determinants of Surplus (Cross-sectional OLS)
# =============================================================================
print("\n" + "=" * 70)
print("Analysis 3: Determinants of Worker Surplus")
print("=" * 70)

# Main outcome: markup relative to minimum WTA
outcome = 'hrly_surp_rel_min'
temp_df = df_survey[df_survey[outcome].notna()].copy()

# 3a. Baseline bivariate: log number of jobs
print("\n3a. Effect of experience (log number of jobs) on markup:")
model_base = smf.ols(f'{outcome} ~ log_numjobs', data=temp_df).fit(cov_type='HC1')
result = create_result_dict(
    spec_id="ols/controls/none",
    spec_tree_path="methods/cross_sectional_ols.md#control-sets",
    outcome_var=outcome,
    treatment_var="log_numjobs",
    model=model_base,
    sample_desc="Survey participants with valid hrly_surp_rel_min",
    fixed_effects="None",
    controls_desc="None",
    cluster_var=None,
    model_type="OLS with robust SE",
    n_obs=len(temp_df),
    r_squared=model_base.rsquared
)
results.append(result)
print(f"  Coef: {model_base.params['log_numjobs']:.4f} (SE = {model_base.bse['log_numjobs']:.4f})")
print(f"  p-value: {model_base.pvalues['log_numjobs']:.4f}")

# 3b. With controls
print("\n3b. Adding controls:")
model_controls = smf.ols(f'{outcome} ~ log_numjobs + log_rate + c_US', data=temp_df).fit(cov_type='HC1')
result = create_result_dict(
    spec_id="ols/controls/baseline",
    spec_tree_path="methods/cross_sectional_ols.md#control-sets",
    outcome_var=outcome,
    treatment_var="log_numjobs",
    model=model_controls,
    sample_desc="Survey participants with valid hrly_surp_rel_min",
    fixed_effects="None",
    controls_desc="log_rate, c_US",
    cluster_var=None,
    model_type="OLS with robust SE",
    n_obs=len(temp_df),
    r_squared=model_controls.rsquared
)
results.append(result)
print(f"  log_numjobs: {model_controls.params['log_numjobs']:.4f} (p = {model_controls.pvalues['log_numjobs']:.4f})")
print(f"  log_rate: {model_controls.params['log_rate']:.4f} (p = {model_controls.pvalues['log_rate']:.4f})")
print(f"  c_US: {model_controls.params['c_US']:.4f} (p = {model_controls.pvalues['c_US']:.4f})")

# 3c. US vs non-US comparison
print("\n3c. US vs non-US workers:")
model_us = smf.ols(f'{outcome} ~ c_US', data=temp_df).fit(cov_type='HC1')
result = create_result_dict(
    spec_id="ols/subgroup/us_vs_nonus",
    spec_tree_path="methods/cross_sectional_ols.md#sample-restrictions",
    outcome_var=outcome,
    treatment_var="c_US",
    model=model_us,
    sample_desc="Survey participants with valid hrly_surp_rel_min",
    fixed_effects="None",
    controls_desc="None",
    cluster_var=None,
    model_type="OLS with robust SE",
    n_obs=len(temp_df),
    r_squared=model_us.rsquared
)
results.append(result)
print(f"  US effect: {model_us.params['c_US']:.4f} (SE = {model_us.bse['c_US']:.4f})")
print(f"  p-value: {model_us.pvalues['c_US']:.4f}")

# =============================================================================
# Analysis 4: Leave-One-Out Robustness
# =============================================================================
print("\n" + "=" * 70)
print("Analysis 4: Leave-One-Out Robustness")
print("=" * 70)

controls = ['log_numjobs', 'log_rate', 'c_US']
baseline_coef = model_controls.params['log_numjobs']

for drop_var in controls:
    remaining = [c for c in controls if c != drop_var]
    formula = f'{outcome} ~ {" + ".join(remaining)}'
    model_loo = smf.ols(formula, data=temp_df).fit(cov_type='HC1')

    # Treatment variable changes depending on what we drop
    if 'log_numjobs' in remaining:
        treatment = 'log_numjobs'
    else:
        treatment = remaining[0]

    result = create_result_dict(
        spec_id=f"robust/loo/drop_{drop_var}",
        spec_tree_path="robustness/leave_one_out.md",
        outcome_var=outcome,
        treatment_var=treatment,
        model=model_loo,
        sample_desc="Survey participants with valid hrly_surp_rel_min",
        fixed_effects="None",
        controls_desc=", ".join(remaining),
        cluster_var=None,
        model_type="OLS with robust SE",
        n_obs=len(temp_df),
        r_squared=model_loo.rsquared
    )
    results.append(result)

    if 'log_numjobs' in remaining:
        coef_change = (model_loo.params['log_numjobs'] - baseline_coef) / baseline_coef * 100
        print(f"\nDrop {drop_var}:")
        print(f"  log_numjobs coef: {model_loo.params['log_numjobs']:.4f}")
        print(f"  Change from baseline: {coef_change:.1f}%")

# =============================================================================
# Analysis 5: Functional Form Robustness
# =============================================================================
print("\n" + "=" * 70)
print("Analysis 5: Functional Form Robustness")
print("=" * 70)

# 5a. Log outcome
print("\n5a. Log outcome (log markup):")
temp_df['log_markup'] = np.log(temp_df[outcome])
model_logY = smf.ols('log_markup ~ log_numjobs + log_rate + c_US', data=temp_df).fit(cov_type='HC1')
result = create_result_dict(
    spec_id="robust/form/y_log",
    spec_tree_path="robustness/functional_form.md",
    outcome_var="log_markup",
    treatment_var="log_numjobs",
    model=model_logY,
    sample_desc="Survey participants with valid hrly_surp_rel_min",
    fixed_effects="None",
    controls_desc="log_rate, c_US",
    cluster_var=None,
    model_type="OLS with robust SE",
    n_obs=len(temp_df),
    r_squared=model_logY.rsquared
)
results.append(result)
print(f"  log_numjobs coef: {model_logY.params['log_numjobs']:.4f} (p = {model_logY.pvalues['log_numjobs']:.4f})")

# 5b. Binary treatment
print("\n5b. Binary treatment (experienced = 5+ jobs):")
model_binary = smf.ols(f'{outcome} ~ experienced + log_rate + c_US', data=temp_df).fit(cov_type='HC1')
result = create_result_dict(
    spec_id="robust/form/x_binary",
    spec_tree_path="robustness/functional_form.md",
    outcome_var=outcome,
    treatment_var="experienced",
    model=model_binary,
    sample_desc="Survey participants with valid hrly_surp_rel_min",
    fixed_effects="None",
    controls_desc="log_rate, c_US",
    cluster_var=None,
    model_type="OLS with robust SE",
    n_obs=len(temp_df),
    r_squared=model_binary.rsquared
)
results.append(result)
print(f"  Experienced effect: {model_binary.params['experienced']:.4f} (p = {model_binary.pvalues['experienced']:.4f})")

# 5c. Quadratic in experience
print("\n5c. Quadratic in experience:")
temp_df['log_numjobs_sq'] = temp_df['log_numjobs'] ** 2
model_quad = smf.ols(f'{outcome} ~ log_numjobs + log_numjobs_sq + log_rate + c_US', data=temp_df).fit(cov_type='HC1')
result = create_result_dict(
    spec_id="robust/form/quadratic",
    spec_tree_path="robustness/functional_form.md",
    outcome_var=outcome,
    treatment_var="log_numjobs",
    model=model_quad,
    sample_desc="Survey participants with valid hrly_surp_rel_min",
    fixed_effects="None",
    controls_desc="log_numjobs_sq, log_rate, c_US",
    cluster_var=None,
    model_type="OLS with robust SE",
    n_obs=len(temp_df),
    r_squared=model_quad.rsquared
)
results.append(result)
print(f"  log_numjobs coef: {model_quad.params['log_numjobs']:.4f} (p = {model_quad.pvalues['log_numjobs']:.4f})")
print(f"  log_numjobs_sq coef: {model_quad.params['log_numjobs_sq']:.4f} (p = {model_quad.pvalues['log_numjobs_sq']:.4f})")

# =============================================================================
# Analysis 6: Weighted Estimates
# =============================================================================
print("\n" + "=" * 70)
print("Analysis 6: Weighted (IPW) Estimates")
print("=" * 70)

# WLS with inverse probability weights
model_weighted = smf.wls(f'{outcome} ~ log_numjobs + log_rate + c_US',
                         data=temp_df, weights=temp_df['weight']).fit(cov_type='HC1')
result = create_result_dict(
    spec_id="ols/method/wls",
    spec_tree_path="methods/cross_sectional_ols.md#estimation-method",
    outcome_var=outcome,
    treatment_var="log_numjobs",
    model=model_weighted,
    sample_desc="Survey participants with IPW weights",
    fixed_effects="None",
    controls_desc="log_rate, c_US",
    cluster_var=None,
    model_type="WLS (IPW) with robust SE",
    n_obs=len(temp_df),
    r_squared=model_weighted.rsquared
)
results.append(result)
print(f"  log_numjobs coef: {model_weighted.params['log_numjobs']:.4f} (p = {model_weighted.pvalues['log_numjobs']:.4f})")

# =============================================================================
# Analysis 7: Alternative Outcomes
# =============================================================================
print("\n" + "=" * 70)
print("Analysis 7: Alternative Outcome Measures")
print("=" * 70)

alt_outcomes = {
    'hrly_surp_rel_expost': 'Ex-post WTA markup',
    'hrly_surp_rel_outside': 'Outside wage markup',
    'fixed_surp_rel_expost': 'Fixed-price ex-post markup'
}

for alt_outcome, label in alt_outcomes.items():
    alt_temp = df_survey[df_survey[alt_outcome].notna()].copy()
    alt_temp['log_numjobs'] = np.log(alt_temp['numjobs'] + 1)
    alt_temp['log_rate'] = np.log(alt_temp['scraped_profile_rate'] + 1)

    if len(alt_temp) >= 10:  # Minimum sample size
        try:
            model_alt = smf.ols(f'{alt_outcome} ~ log_numjobs + log_rate + c_US', data=alt_temp).fit(cov_type='HC1')
            result = create_result_dict(
                spec_id=f"ols/outcome/{alt_outcome.replace('hrly_surp_rel_', '').replace('fixed_surp_rel_', 'fixed_')}",
                spec_tree_path="methods/cross_sectional_ols.md#baseline",
                outcome_var=alt_outcome,
                treatment_var="log_numjobs",
                model=model_alt,
                sample_desc=f"Survey participants with valid {alt_outcome}",
                fixed_effects="None",
                controls_desc="log_rate, c_US",
                cluster_var=None,
                model_type="OLS with robust SE",
                n_obs=len(alt_temp),
                r_squared=model_alt.rsquared
            )
            results.append(result)
            print(f"\n{label}:")
            print(f"  N = {len(alt_temp)}")
            print(f"  log_numjobs coef: {model_alt.params['log_numjobs']:.4f} (p = {model_alt.pvalues['log_numjobs']:.4f})")
        except Exception as e:
            print(f"\n{label}: Could not estimate ({e})")

# =============================================================================
# Analysis 8: Quantile Regression
# =============================================================================
print("\n" + "=" * 70)
print("Analysis 8: Quantile Regression")
print("=" * 70)

from statsmodels.regression.quantile_regression import QuantReg

temp_df_clean = temp_df[['hrly_surp_rel_min', 'log_numjobs', 'log_rate', 'c_US']].dropna()
y = temp_df_clean[outcome]
X = sm.add_constant(temp_df_clean[['log_numjobs', 'log_rate', 'c_US']])

for q in [0.25, 0.50, 0.75]:
    try:
        model_qr = QuantReg(y, X).fit(q=q)

        # Create a simple result dict for quantile regression
        result = {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': f"ols/method/quantile_{int(q*100)}",
            'spec_tree_path': "methods/cross_sectional_ols.md#estimation-method",
            'outcome_var': outcome,
            'treatment_var': "log_numjobs",
            'coefficient': float(model_qr.params['log_numjobs']),
            'std_error': float(model_qr.bse['log_numjobs']),
            't_stat': float(model_qr.tvalues['log_numjobs']),
            'p_value': float(model_qr.pvalues['log_numjobs']),
            'ci_lower': float(model_qr.conf_int().loc['log_numjobs', 0]),
            'ci_upper': float(model_qr.conf_int().loc['log_numjobs', 1]),
            'n_obs': int(len(y)),
            'r_squared': float(model_qr.prsquared),
            'coefficient_vector_json': json.dumps({
                "treatment": {"var": "log_numjobs", "coef": float(model_qr.params['log_numjobs']),
                             "se": float(model_qr.bse['log_numjobs']), "pval": float(model_qr.pvalues['log_numjobs'])},
                "controls": [{"var": "log_rate", "coef": float(model_qr.params['log_rate']),
                             "se": float(model_qr.bse['log_rate']), "pval": float(model_qr.pvalues['log_rate'])},
                            {"var": "c_US", "coef": float(model_qr.params['c_US']),
                             "se": float(model_qr.bse['c_US']), "pval": float(model_qr.pvalues['c_US'])}],
                "fixed_effects": [],
                "diagnostics": {}
            }),
            'sample_desc': "Survey participants with valid hrly_surp_rel_min",
            'fixed_effects': "None",
            'controls_desc': "log_rate, c_US",
            'cluster_var': None,
            'model_type': f"Quantile regression (q={q})",
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
        results.append(result)
        print(f"\nQuantile {int(q*100)}th percentile:")
        print(f"  log_numjobs coef: {model_qr.params['log_numjobs']:.4f} (p = {model_qr.pvalues['log_numjobs']:.4f})")
    except Exception as e:
        print(f"\nQuantile {int(q*100)}th: Error - {e}")

# =============================================================================
# Analysis 9: Sample Restrictions
# =============================================================================
print("\n" + "=" * 70)
print("Analysis 9: Sample Restrictions")
print("=" * 70)

# 9a. Trimmed sample (exclude outliers)
q_low = temp_df[outcome].quantile(0.05)
q_high = temp_df[outcome].quantile(0.95)
trimmed_df = temp_df[(temp_df[outcome] >= q_low) & (temp_df[outcome] <= q_high)].copy()

model_trim = smf.ols(f'{outcome} ~ log_numjobs + log_rate + c_US', data=trimmed_df).fit(cov_type='HC1')
result = create_result_dict(
    spec_id="ols/sample/trimmed",
    spec_tree_path="methods/cross_sectional_ols.md#sample-restrictions",
    outcome_var=outcome,
    treatment_var="log_numjobs",
    model=model_trim,
    sample_desc="5-95 percentile trimmed sample",
    fixed_effects="None",
    controls_desc="log_rate, c_US",
    cluster_var=None,
    model_type="OLS with robust SE",
    n_obs=len(trimmed_df),
    r_squared=model_trim.rsquared
)
results.append(result)
print(f"\nTrimmed sample (5-95 percentile):")
print(f"  N = {len(trimmed_df)}")
print(f"  log_numjobs coef: {model_trim.params['log_numjobs']:.4f} (p = {model_trim.pvalues['log_numjobs']:.4f})")

# 9b. Non-US only
nonus_df = temp_df[temp_df['c_US'] == 0].copy()
if len(nonus_df) >= 10:
    model_nonus = smf.ols(f'{outcome} ~ log_numjobs + log_rate', data=nonus_df).fit(cov_type='HC1')
    result = create_result_dict(
        spec_id="ols/sample/subgroup_nonus",
        spec_tree_path="methods/cross_sectional_ols.md#sample-restrictions",
        outcome_var=outcome,
        treatment_var="log_numjobs",
        model=model_nonus,
        sample_desc="Non-US workers only",
        fixed_effects="None",
        controls_desc="log_rate",
        cluster_var=None,
        model_type="OLS with robust SE",
        n_obs=len(nonus_df),
        r_squared=model_nonus.rsquared
    )
    results.append(result)
    print(f"\nNon-US workers only:")
    print(f"  N = {len(nonus_df)}")
    print(f"  log_numjobs coef: {model_nonus.params['log_numjobs']:.4f} (p = {model_nonus.pvalues['log_numjobs']:.4f})")

# =============================================================================
# Save Results
# =============================================================================
print("\n" + "=" * 70)
print("Saving Results")
print("=" * 70)

results_df = pd.DataFrame(results)
output_file = f"{OUTPUT_PATH}/specification_results.csv"
results_df.to_csv(output_file, index=False)
print(f"\nSaved {len(results)} specifications to: {output_file}")

# =============================================================================
# Summary Statistics
# =============================================================================
print("\n" + "=" * 70)
print("Summary Statistics")
print("=" * 70)

print(f"\nTotal specifications: {len(results)}")
print(f"Positive coefficients: {sum(r['coefficient'] > 0 for r in results)} ({100*sum(r['coefficient'] > 0 for r in results)/len(results):.1f}%)")
print(f"Significant at 5%: {sum(r['p_value'] < 0.05 for r in results)} ({100*sum(r['p_value'] < 0.05 for r in results)/len(results):.1f}%)")
print(f"Significant at 1%: {sum(r['p_value'] < 0.01 for r in results)} ({100*sum(r['p_value'] < 0.01 for r in results)/len(results):.1f}%)")

# Coefficient summary for log_numjobs specifications
log_numjobs_specs = [r for r in results if r['treatment_var'] == 'log_numjobs']
if log_numjobs_specs:
    coefs = [r['coefficient'] for r in log_numjobs_specs]
    print(f"\nlog_numjobs coefficient summary (n={len(coefs)}):")
    print(f"  Median: {np.median(coefs):.4f}")
    print(f"  Mean: {np.mean(coefs):.4f}")
    print(f"  Range: [{min(coefs):.4f}, {max(coefs):.4f}]")

print("\n" + "=" * 70)
print("Analysis Complete")
print("=" * 70)
