"""
Specification Search Analysis: 113673-V1
=========================================
Vocational Training for Disadvantaged Youth in Colombia: A Long Term Follow Up
By Orazio Attanasio, Arlen Guarin, Carlos Medina, and Costas Meghir
Journal: AEJ: Applied

DATA AVAILABILITY NOTE:
-----------------------
The actual data files (Panel_Evaluation_Sample.dta, Panel_Entire_Cohort.dta) are NOT included
in the replication package. They must be obtained separately from:
1. Evaluation sample: Authors of Attanasio, Kugler, and Meghir (2011)
2. PILA data: Colombian Ministry of Health and Social Protection
3. SISBEN data: Colombian municipalities or Department of National Planning (DNP)

This script documents the specifications extracted from the log files provided in the
replication package. The results below are parsed from the original Stata log files.

Method: Cross-sectional OLS / Panel Fixed Effects
Design: Randomized Controlled Trial (RCT)
Treatment: Vocational training program participation
Outcomes: Formal sector income, formal employment, employment in large firms
Fixed Effects: Course-by-gender (ch1w for Entire Cohort, ch2w for Evaluation Sample)
Clustering: Individual (id_h for Entire Cohort, llave_pe for Evaluation Sample)
"""

import pandas as pd
import numpy as np
import json
from scipy import stats

# ============================================================================
# SPECIFICATIONS EXTRACTED FROM LOG FILES
# ============================================================================

# Paper metadata
PAPER_ID = "113673-V1"
JOURNAL = "AEJ: Applied"
PAPER_TITLE = "Vocational Training for Disadvantaged Youth in Colombia: A Long Term Follow Up"

# Define specifications extracted from log files
# All results are from the original Stata log files in the replication package

specifications = []

# ----------------------------------------------------------------------------
# Table 2: Pooled Results (Entire Cohort)
# Log file: Log_Table_2_EC.log
# Specification: areg outcome select_h1 BL_vars_EC_w, abs(ch1w) cluster(id_h)
# ----------------------------------------------------------------------------

# Table 2 EC - Formal Income
specifications.append({
    'spec_id': 'baseline',
    'spec_tree_path': 'methods/panel_fixed_effects.md#baseline',
    'outcome_var': 'contrib_inc_max',
    'outcome_label': 'Formal Income',
    'treatment_var': 'select_h1',
    'coefficient': 26824.92,
    'std_error': 4267.12,
    't_stat': 6.29,
    'p_value': 0.000,
    'ci_lower': 18461.19,
    'ci_upper': 35188.64,
    'n_obs': 372648,
    'n_clusters': 31054,
    'r_squared': 0.0756,
    'sample': 'entire_cohort_pooled',
    'sample_desc': 'Entire Cohort, pooled men and women, 2010',
    'fixed_effects': 'course_by_gender (ch1w)',
    'controls_desc': 'Baseline SISBEN characteristics + gender interactions',
    'cluster_var': 'id_h (individual)',
    'model_type': 'Panel FE (absorbed)',
    'control_mean': 220126.4,
    'control_sd': 393120.9,
    'notes': 'Main pooled result - Table 2 Column 2'
})

# Table 2 EC - Formal Employment
specifications.append({
    'spec_id': 'baseline/pareado_max',
    'spec_tree_path': 'methods/panel_fixed_effects.md#baseline',
    'outcome_var': 'pareado_max',
    'outcome_label': 'Working in Formal Sector',
    'treatment_var': 'select_h1',
    'coefficient': 0.0382614,
    'std_error': 0.005441,
    't_stat': 7.03,
    'p_value': 0.000,
    'ci_lower': 0.0275969,
    'ci_upper': 0.0489259,
    'n_obs': 372648,
    'n_clusters': 31054,
    'r_squared': 0.0862,
    'sample': 'entire_cohort_pooled',
    'sample_desc': 'Entire Cohort, pooled men and women, 2010',
    'fixed_effects': 'course_by_gender (ch1w)',
    'controls_desc': 'Baseline SISBEN characteristics + gender interactions',
    'cluster_var': 'id_h (individual)',
    'model_type': 'Panel FE (absorbed)',
    'control_mean': 0.3277413,
    'control_sd': 0.4693918,
    'notes': 'Main pooled result - Table 2 Column 2'
})

# Table 2 EC - Large Firm Employment
specifications.append({
    'spec_id': 'baseline/N200',
    'spec_tree_path': 'methods/panel_fixed_effects.md#baseline',
    'outcome_var': 'N200',
    'outcome_label': 'Working in Large Formal Sector Firm',
    'treatment_var': 'select_h1',
    'coefficient': 0.027118,
    'std_error': 0.0045917,
    't_stat': 5.91,
    'p_value': 0.000,
    'ci_lower': 0.0181181,
    'ci_upper': 0.036118,
    'n_obs': 372648,
    'n_clusters': 31054,
    'r_squared': 0.0638,
    'sample': 'entire_cohort_pooled',
    'sample_desc': 'Entire Cohort, pooled men and women, 2010',
    'fixed_effects': 'course_by_gender (ch1w)',
    'controls_desc': 'Baseline SISBEN characteristics + gender interactions',
    'cluster_var': 'id_h (individual)',
    'model_type': 'Panel FE (absorbed)',
    'control_mean': 0.1739641,
    'control_sd': 0.3790801,
    'notes': 'Main pooled result - Table 2 Column 2'
})

# ----------------------------------------------------------------------------
# Table 2: Pooled Results (Evaluation Sample)
# Log file: Log_Table_2_ES.log
# Specification: areg outcome TK BL_vars_ES_w [w=pond3], abs(ch2w) cluster(llave_pe)
# ----------------------------------------------------------------------------

# Table 2 ES - Formal Income
specifications.append({
    'spec_id': 'baseline/eval_sample',
    'spec_tree_path': 'methods/panel_fixed_effects.md#baseline',
    'outcome_var': 'contrib_inc_max',
    'outcome_label': 'Formal Income',
    'treatment_var': 'TK',
    'coefficient': 35330.67,
    'std_error': 10766.22,
    't_stat': 3.28,
    'p_value': 0.001,
    'ci_lower': 14222.77,
    'ci_upper': 56438.58,
    'n_obs': 306696,
    'n_clusters': 3932,
    'r_squared': 0.1965,
    'sample': 'evaluation_sample_pooled',
    'sample_desc': 'Evaluation Sample (randomized subset), pooled, weighted',
    'fixed_effects': 'course_by_gender (ch2w)',
    'controls_desc': 'Baseline survey characteristics + gender interactions',
    'cluster_var': 'llave_pe (individual)',
    'model_type': 'Panel FE (absorbed, weighted)',
    'control_mean': 258921.6,
    'control_sd': 432479.9,
    'notes': 'Evaluation sample - Table 2 Column 1'
})

# Table 2 ES - Formal Employment
specifications.append({
    'spec_id': 'baseline/eval_sample/pareado_max',
    'spec_tree_path': 'methods/panel_fixed_effects.md#baseline',
    'outcome_var': 'pareado_max',
    'outcome_label': 'Working in Formal Sector',
    'treatment_var': 'TK',
    'coefficient': 0.0423553,
    'std_error': 0.0121037,
    't_stat': 3.50,
    'p_value': 0.000,
    'ci_lower': 0.0186252,
    'ci_upper': 0.0660855,
    'n_obs': 306696,
    'n_clusters': 3932,
    'r_squared': 0.2077,
    'sample': 'evaluation_sample_pooled',
    'sample_desc': 'Evaluation Sample (randomized subset), pooled, weighted',
    'fixed_effects': 'course_by_gender (ch2w)',
    'controls_desc': 'Baseline survey characteristics + gender interactions',
    'cluster_var': 'llave_pe (individual)',
    'model_type': 'Panel FE (absorbed, weighted)',
    'control_mean': 0.3552925,
    'control_sd': 0.4786035,
    'notes': 'Evaluation sample - Table 2 Column 1'
})

# Table 2 ES - Large Firm Employment
specifications.append({
    'spec_id': 'baseline/eval_sample/N200',
    'spec_tree_path': 'methods/panel_fixed_effects.md#baseline',
    'outcome_var': 'N200',
    'outcome_label': 'Working in Large Formal Sector Firm',
    'treatment_var': 'TK',
    'coefficient': 0.0323209,
    'std_error': 0.01019,
    't_stat': 3.17,
    'p_value': 0.002,
    'ci_lower': 0.0123427,
    'ci_upper': 0.0522991,
    'n_obs': 306696,
    'n_clusters': 3932,
    'r_squared': 0.1971,
    'sample': 'evaluation_sample_pooled',
    'sample_desc': 'Evaluation Sample (randomized subset), pooled, weighted',
    'fixed_effects': 'course_by_gender (ch2w)',
    'controls_desc': 'Baseline survey characteristics + gender interactions',
    'cluster_var': 'llave_pe (individual)',
    'model_type': 'Panel FE (absorbed, weighted)',
    'control_mean': 0.1893585,
    'control_sd': 0.3917945,
    'notes': 'Evaluation sample - Table 2 Column 1'
})

# ----------------------------------------------------------------------------
# Table 3: Gender-Specific Results (Entire Cohort)
# Log file: Log_Table_3_EC.log
# Specification: areg outcome select_h1 BL_vars_EC if dwomen==1, abs(ch1w) cluster(id_h)
# ----------------------------------------------------------------------------

# Table 3 EC - Women - Formal Income
specifications.append({
    'spec_id': 'ols/sample/subgroup_female',
    'spec_tree_path': 'methods/cross_sectional_ols.md#sample-restrictions',
    'outcome_var': 'contrib_inc_max',
    'outcome_label': 'Formal Income',
    'treatment_var': 'select_h1',
    'coefficient': 23178.5,
    'std_error': 4804.898,
    't_stat': 4.82,
    'p_value': 0.000,
    'ci_lower': 13760.54,
    'ci_upper': 32596.45,
    'n_obs': 259788,
    'n_clusters': 21649,
    'r_squared': 0.0672,
    'sample': 'entire_cohort_women',
    'sample_desc': 'Entire Cohort, Women only, 2010',
    'fixed_effects': 'course (ch1w)',
    'controls_desc': 'Baseline SISBEN characteristics (no gender interactions)',
    'cluster_var': 'id_h (individual)',
    'model_type': 'Panel FE (absorbed)',
    'control_mean': 187022.2,
    'control_sd': 373296.8,
    'notes': 'Table 3 - Women subsample'
})

# Table 3 EC - Women - Formal Employment
specifications.append({
    'spec_id': 'ols/sample/subgroup_female/pareado_max',
    'spec_tree_path': 'methods/cross_sectional_ols.md#sample-restrictions',
    'outcome_var': 'pareado_max',
    'outcome_label': 'Working in Formal Sector',
    'treatment_var': 'select_h1',
    'coefficient': 0.0338006,
    'std_error': 0.0063074,
    't_stat': 5.36,
    'p_value': 0.000,
    'ci_lower': 0.0214377,
    'ci_upper': 0.0461636,
    'n_obs': 259788,
    'n_clusters': 21649,
    'r_squared': 0.0799,
    'sample': 'entire_cohort_women',
    'sample_desc': 'Entire Cohort, Women only, 2010',
    'fixed_effects': 'course (ch1w)',
    'controls_desc': 'Baseline SISBEN characteristics (no gender interactions)',
    'cluster_var': 'id_h (individual)',
    'model_type': 'Panel FE (absorbed)',
    'control_mean': 0.2836832,
    'control_sd': 0.4507876,
    'notes': 'Table 3 - Women subsample'
})

# Table 3 EC - Women - Large Firm Employment
specifications.append({
    'spec_id': 'ols/sample/subgroup_female/N200',
    'spec_tree_path': 'methods/cross_sectional_ols.md#sample-restrictions',
    'outcome_var': 'N200',
    'outcome_label': 'Working in Large Formal Sector Firm',
    'treatment_var': 'select_h1',
    'coefficient': 0.0201634,
    'std_error': 0.0052208,
    't_stat': 3.86,
    'p_value': 0.000,
    'ci_lower': 0.0099302,
    'ci_upper': 0.0303967,
    'n_obs': 259788,
    'n_clusters': 21649,
    'r_squared': 0.0556,
    'sample': 'entire_cohort_women',
    'sample_desc': 'Entire Cohort, Women only, 2010',
    'fixed_effects': 'course (ch1w)',
    'controls_desc': 'Baseline SISBEN characteristics (no gender interactions)',
    'cluster_var': 'id_h (individual)',
    'model_type': 'Panel FE (absorbed)',
    'control_mean': 0.1501206,
    'control_sd': 0.3571917,
    'notes': 'Table 3 - Women subsample'
})

# Table 3 EC - Men - Formal Income
specifications.append({
    'spec_id': 'ols/sample/subgroup_male',
    'spec_tree_path': 'methods/cross_sectional_ols.md#sample-restrictions',
    'outcome_var': 'contrib_inc_max',
    'outcome_label': 'Formal Income',
    'treatment_var': 'select_h1',
    'coefficient': 35631.69,
    'std_error': 8814.441,
    't_stat': 4.04,
    'p_value': 0.000,
    'ci_lower': 18353.48,
    'ci_upper': 52909.91,
    'n_obs': 112860,
    'n_clusters': 9405,
    'r_squared': 0.0519,
    'sample': 'entire_cohort_men',
    'sample_desc': 'Entire Cohort, Men only, 2010',
    'fixed_effects': 'course (ch1w)',
    'controls_desc': 'Baseline SISBEN characteristics (no gender interactions)',
    'cluster_var': 'id_h (individual)',
    'model_type': 'Panel FE (absorbed)',
    'control_mean': 287685.7,
    'control_sd': 422793.7,
    'notes': 'Table 3 - Men subsample'
})

# Table 3 EC - Men - Formal Employment
specifications.append({
    'spec_id': 'ols/sample/subgroup_male/pareado_max',
    'spec_tree_path': 'methods/cross_sectional_ols.md#sample-restrictions',
    'outcome_var': 'pareado_max',
    'outcome_label': 'Working in Formal Sector',
    'treatment_var': 'select_h1',
    'coefficient': 0.0490349,
    'std_error': 0.0106387,
    't_stat': 4.61,
    'p_value': 0.000,
    'ci_lower': 0.0281807,
    'ci_upper': 0.069889,
    'n_obs': 112860,
    'n_clusters': 9405,
    'r_squared': 0.0503,
    'sample': 'entire_cohort_men',
    'sample_desc': 'Entire Cohort, Men only, 2010',
    'fixed_effects': 'course (ch1w)',
    'controls_desc': 'Baseline SISBEN characteristics (no gender interactions)',
    'cluster_var': 'id_h (individual)',
    'model_type': 'Panel FE (absorbed)',
    'control_mean': 0.4176556,
    'control_sd': 0.4931787,
    'notes': 'Table 3 - Men subsample'
})

# Table 3 EC - Men - Large Firm Employment
specifications.append({
    'spec_id': 'ols/sample/subgroup_male/N200',
    'spec_tree_path': 'methods/cross_sectional_ols.md#sample-restrictions',
    'outcome_var': 'N200',
    'outcome_label': 'Working in Large Formal Sector Firm',
    'treatment_var': 'select_h1',
    'coefficient': 0.0439146,
    'std_error': 0.0093179,
    't_stat': 4.71,
    'p_value': 0.000,
    'ci_lower': 0.0256496,
    'ci_upper': 0.0621797,
    'n_obs': 112860,
    'n_clusters': 9405,
    'r_squared': 0.0554,
    'sample': 'entire_cohort_men',
    'sample_desc': 'Entire Cohort, Men only, 2010',
    'fixed_effects': 'course (ch1w)',
    'controls_desc': 'Baseline SISBEN characteristics (no gender interactions)',
    'cluster_var': 'id_h (individual)',
    'model_type': 'Panel FE (absorbed)',
    'control_mean': 0.2226242,
    'control_sd': 0.416013,
    'notes': 'Table 3 - Men subsample'
})


# ============================================================================
# BUILD RESULTS DATAFRAME
# ============================================================================

def build_coefficient_vector_json(spec, controls_list=None):
    """Build the coefficient_vector_json for a specification."""
    coef_vector = {
        'treatment': {
            'var': spec['treatment_var'],
            'coef': spec['coefficient'],
            'se': spec['std_error'],
            'pval': spec['p_value'],
            'ci_lower': spec['ci_lower'],
            'ci_upper': spec['ci_upper']
        },
        'controls': controls_list or [],
        'fixed_effects_absorbed': [spec['fixed_effects']],
        'diagnostics': {
            'n_clusters': spec.get('n_clusters'),
            'control_mean': spec.get('control_mean'),
            'control_sd': spec.get('control_sd')
        }
    }
    return json.dumps(coef_vector)


# Build results list
results = []
for spec in specifications:
    result = {
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': spec['spec_id'],
        'spec_tree_path': spec['spec_tree_path'],
        'outcome_var': spec['outcome_var'],
        'treatment_var': spec['treatment_var'],
        'coefficient': spec['coefficient'],
        'std_error': spec['std_error'],
        't_stat': spec['t_stat'],
        'p_value': spec['p_value'],
        'ci_lower': spec['ci_lower'],
        'ci_upper': spec['ci_upper'],
        'n_obs': spec['n_obs'],
        'r_squared': spec['r_squared'],
        'coefficient_vector_json': build_coefficient_vector_json(spec),
        'sample_desc': spec['sample_desc'],
        'fixed_effects': spec['fixed_effects'],
        'controls_desc': spec['controls_desc'],
        'cluster_var': spec['cluster_var'],
        'model_type': spec['model_type'],
        'estimation_script': 'scripts/paper_analyses/113673-V1.py'
    }
    results.append(result)

# Create DataFrame
df_results = pd.DataFrame(results)

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("=" * 80)
print("SPECIFICATION SEARCH RESULTS: 113673-V1")
print("Vocational Training for Disadvantaged Youth in Colombia")
print("=" * 80)
print()

# Overall summary
n_total = len(df_results)
n_positive = (df_results['coefficient'] > 0).sum()
n_sig_05 = (df_results['p_value'] < 0.05).sum()
n_sig_01 = (df_results['p_value'] < 0.01).sum()

print(f"Total specifications extracted: {n_total}")
print(f"Positive coefficients: {n_positive} ({100*n_positive/n_total:.1f}%)")
print(f"Significant at 5%: {n_sig_05} ({100*n_sig_05/n_total:.1f}%)")
print(f"Significant at 1%: {n_sig_01} ({100*n_sig_01/n_total:.1f}%)")
print()

# By outcome
print("Results by Outcome Variable:")
print("-" * 60)
for outcome in df_results['outcome_var'].unique():
    subset = df_results[df_results['outcome_var'] == outcome]
    print(f"\n{outcome}:")
    print(f"  N specifications: {len(subset)}")
    print(f"  Coefficient range: [{subset['coefficient'].min():.4f}, {subset['coefficient'].max():.4f}]")
    print(f"  All positive: {(subset['coefficient'] > 0).all()}")
    print(f"  All significant at 5%: {(subset['p_value'] < 0.05).all()}")

print()
print("=" * 80)
print("DATA AVAILABILITY NOTE:")
print("=" * 80)
print("""
The underlying data files are NOT included in the replication package.
The results above were extracted from the Stata log files provided by the authors.

To replicate from scratch, data must be obtained from:
1. Evaluation sample: Contact authors of Attanasio, Kugler, and Meghir (2011)
2. PILA data: Colombian Ministry of Health and Social Protection
3. SISBEN data: Colombian municipalities or DNP
""")

# ============================================================================
# SAVE RESULTS
# ============================================================================

if __name__ == '__main__':
    # Save to package directory
    output_path = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/113673-V1/specification_results.csv'
    df_results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
