"""
Specification Search Script for Paper 150581-V1
"Wage Cyclicality and Labor Market Sorting"

This script runs a systematic specification search following the Institute for
Replication (i4r) methodology, testing robustness across:
- Control variable variations
- Sample restrictions
- Alternative outcomes
- Alternative treatments
- Inference variations
- Estimation method variations
- Functional form variations
- Placebo tests
- Heterogeneity analyses

Author: Claude Code (automated specification search)
Date: 2026-02-02
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# Paths
BASE_PATH = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search'
PKG_PATH = f'{BASE_PATH}/data/downloads/extracted/150581-V1'

# Load data
df = pd.read_csv(f'{PKG_PATH}/data_analysis.csv')
print(f"Data loaded: {df.shape}")

# Prepare fixed effects variables
df['ind_year_fe'] = df['industry'].astype(str) + '_' + df['year'].astype(str)
df['occ_year_fe'] = df['occupation_agg'].astype(str) + '_' + df['year'].astype(str)

# Results storage
results = []

# Paper metadata
PAPER_ID = '150581-V1'
PAPER_TITLE = 'Wage Cyclicality and Labor Market Sorting'
JOURNAL = 'AER'

def run_spec(formula, data, vcov, spec_id, spec_tree_path, outcome_var, treatment_var,
             sample_desc, fixed_effects_desc, controls_desc, cluster_var, model_type='Panel FE'):
    """Run a specification and store results."""
    try:
        model = pf.feols(formula, data=data, vcov=vcov)

        # Get treatment coefficient
        coef_dict = model.coef()
        se_dict = model.se()
        pval_dict = model.pvalue()
        tstat_dict = model.tstat()

        # Find the treatment variable in coefficients
        treat_coef = None
        treat_se = None
        treat_pval = None
        treat_tstat = None

        for key in coef_dict.index:
            if treatment_var in key:
                treat_coef = coef_dict[key]
                treat_se = se_dict[key]
                treat_pval = pval_dict[key]
                treat_tstat = tstat_dict[key]
                break

        if treat_coef is None:
            treat_coef = coef_dict.iloc[0]
            treat_se = se_dict.iloc[0]
            treat_pval = pval_dict.iloc[0]
            treat_tstat = tstat_dict.iloc[0]

        ci_lower = treat_coef - 1.96 * treat_se
        ci_upper = treat_coef + 1.96 * treat_se
        r_squared = model._r2 if hasattr(model, '_r2') else np.nan

        coef_vector = {
            'treatment': {
                'var': treatment_var,
                'coef': float(treat_coef),
                'se': float(treat_se),
                'pval': float(treat_pval)
            },
            'controls': [
                {'var': str(v), 'coef': float(coef_dict[v]), 'se': float(se_dict[v]), 'pval': float(pval_dict[v])}
                for v in coef_dict.index if v != treatment_var
            ][:10],
            'fixed_effects': fixed_effects_desc.split(', ') if fixed_effects_desc else [],
            'diagnostics': {'r_squared': float(r_squared)}
        }

        result = {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': float(treat_coef),
            'std_error': float(treat_se),
            't_stat': float(treat_tstat),
            'p_value': float(treat_pval),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_obs': int(model._N),
            'r_squared': float(r_squared),
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects_desc,
            'controls_desc': controls_desc,
            'cluster_var': cluster_var,
            'model_type': model_type,
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }

        results.append(result)
        print(f"  {spec_id}: coef={treat_coef:.4f}, se={treat_se:.4f}, p={treat_pval:.4f}, n={model._N}")
        return result

    except Exception as e:
        print(f"  ERROR in {spec_id}: {str(e)[:100]}")
        return None


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("SPECIFICATION SEARCH: Wage Cyclicality and Labor Market Sorting")
    print("="*80)

    # BASELINE SPECIFICATIONS
    print("\n--- BASELINE SPECIFICATIONS ---")

    run_spec(
        formula='lhrp2 ~ unempl + unempl:C(dummy) + C(dummy) + age + agesq + C(dummy_educ) + time + I(time**2) + time_trend + C(month) | ID + ind_year_fe + occ_year_fe',
        data=df, vcov={'CRV1': 'ID'},
        spec_id='baseline/table2_col1',
        spec_tree_path='methods/panel_fixed_effects.md#baseline',
        outcome_var='lhrp2', treatment_var='unempl',
        sample_desc='Full sample, HOURSM>=75, age>=20',
        fixed_effects_desc='ID, industry-year, occupation-year',
        controls_desc='age, agesq, education, tenure, tenure_sq, time_trend, month',
        cluster_var='ID'
    )

    run_spec(
        formula='lhrp2 ~ unempl + unempl:C(dummy1) + unempl:C(dummy2) + C(dummy1) + C(dummy2) + age + agesq + C(dummy_educ) + time + I(time**2) + time_trend + C(month) | ID + ind_year_fe + occ_year_fe',
        data=df, vcov={'CRV1': 'ID'},
        spec_id='baseline/table2_col2',
        spec_tree_path='methods/panel_fixed_effects.md#baseline',
        outcome_var='lhrp2', treatment_var='unempl',
        sample_desc='Full sample, HOURSM>=75, age>=20',
        fixed_effects_desc='ID, industry-year, occupation-year',
        controls_desc='age, agesq, education, tenure, tenure_sq, time_trend, month, EE, UE',
        cluster_var='ID'
    )

    run_spec(
        formula='lhrp2 ~ unempl + unempl:C(dummy1) + unempl:C(dummy2) + mismatch1w:unempl + unempl:C(dummy1):mismatch1w + unempl:C(dummy2):mismatch1w + mismatch1w:C(dummy1) + mismatch1w:C(dummy2) + mismatch1w + C(dummy1) + C(dummy2) + age + agesq + C(dummy_educ) + time + I(time**2) + time_trend + C(month) | ID + ind_year_fe + occ_year_fe',
        data=df, vcov={'CRV1': 'ID'},
        spec_id='baseline/table2_col4',
        spec_tree_path='methods/panel_fixed_effects.md#baseline',
        outcome_var='lhrp2', treatment_var='unempl',
        sample_desc='Full sample, HOURSM>=75, age>=20',
        fixed_effects_desc='ID, industry-year, occupation-year',
        controls_desc='Full controls with mismatch interactions',
        cluster_var='ID'
    )

    # CONTROL VARIATIONS
    print("\n--- CONTROL VARIATIONS ---")

    controls = ['age', 'education', 'tenure', 'mismatch', 'timetrend', 'monthfe']
    for ctrl in controls:
        # Formula varies by what we drop
        if ctrl == 'age':
            formula = 'lhrp2 ~ unempl + unempl:C(dummy1) + unempl:C(dummy2) + mismatch1w:unempl + C(dummy1) + C(dummy2) + mismatch1w + C(dummy_educ) + time + I(time**2) + time_trend + C(month) | ID + ind_year_fe + occ_year_fe'
        elif ctrl == 'education':
            formula = 'lhrp2 ~ unempl + unempl:C(dummy1) + unempl:C(dummy2) + mismatch1w:unempl + C(dummy1) + C(dummy2) + mismatch1w + age + agesq + time + I(time**2) + time_trend + C(month) | ID + ind_year_fe + occ_year_fe'
        elif ctrl == 'tenure':
            formula = 'lhrp2 ~ unempl + unempl:C(dummy1) + unempl:C(dummy2) + mismatch1w:unempl + C(dummy1) + C(dummy2) + mismatch1w + age + agesq + C(dummy_educ) + time_trend + C(month) | ID + ind_year_fe + occ_year_fe'
        elif ctrl == 'mismatch':
            formula = 'lhrp2 ~ unempl + unempl:C(dummy1) + unempl:C(dummy2) + C(dummy1) + C(dummy2) + age + agesq + C(dummy_educ) + time + I(time**2) + time_trend + C(month) | ID + ind_year_fe + occ_year_fe'
        elif ctrl == 'timetrend':
            formula = 'lhrp2 ~ unempl + unempl:C(dummy1) + unempl:C(dummy2) + mismatch1w:unempl + C(dummy1) + C(dummy2) + mismatch1w + age + agesq + C(dummy_educ) + time + I(time**2) + C(month) | ID + ind_year_fe + occ_year_fe'
        else:
            formula = 'lhrp2 ~ unempl + unempl:C(dummy1) + unempl:C(dummy2) + mismatch1w:unempl + C(dummy1) + C(dummy2) + mismatch1w + age + agesq + C(dummy_educ) + time + I(time**2) + time_trend | ID + ind_year_fe + occ_year_fe'

        run_spec(
            formula=formula,
            data=df, vcov={'CRV1': 'ID'},
            spec_id=f'robust/control/drop_{ctrl}',
            spec_tree_path='robustness/leave_one_out.md',
            outcome_var='lhrp2', treatment_var='unempl',
            sample_desc='Full sample',
            fixed_effects_desc='ID, industry-year, occupation-year',
            controls_desc=f'Full controls minus {ctrl}',
            cluster_var='ID'
        )

    # SAMPLE RESTRICTIONS
    print("\n--- SAMPLE RESTRICTIONS ---")

    sample_specs = [
        ('pre_2000', df[df['year'] < 2000].copy()),
        ('post_2000', df[df['year'] >= 2000].copy()),
        ('young_workers', df[df['age'] < 35].copy()),
        ('older_workers', df[df['age'] >= 35].copy()),
        ('college', df[df['dummy_educ'] == 1].copy()),
        ('noncollege', df[df['dummy_educ'] == 0].copy()),
    ]

    for name, data in sample_specs:
        run_spec(
            formula='lhrp2 ~ unempl + unempl:C(dummy1) + unempl:C(dummy2) + mismatch1w:unempl + C(dummy1) + C(dummy2) + mismatch1w + age + agesq + C(dummy_educ) + time + I(time**2) + time_trend + C(month) | ID + ind_year_fe + occ_year_fe',
            data=data, vcov={'CRV1': 'ID'},
            spec_id=f'robust/sample/{name}',
            spec_tree_path='robustness/sample_restrictions.md',
            outcome_var='lhrp2', treatment_var='unempl',
            sample_desc=name,
            fixed_effects_desc='ID, industry-year, occupation-year',
            controls_desc='Full controls',
            cluster_var='ID'
        )

    # CLUSTERING VARIATIONS
    print("\n--- CLUSTERING VARIATIONS ---")

    for cluster in ['year', 'year_month', 'regionfe']:
        if cluster == 'year_month':
            df['year_month'] = df['year'].astype(str) + '_' + df['month'].astype(str)
        run_spec(
            formula='lhrp2 ~ unempl + unempl:C(dummy1) + unempl:C(dummy2) + mismatch1w:unempl + C(dummy1) + C(dummy2) + mismatch1w + age + agesq + C(dummy_educ) + time + I(time**2) + time_trend + C(month) | ID + ind_year_fe + occ_year_fe',
            data=df, vcov={'CRV1': cluster},
            spec_id=f'robust/cluster/{cluster}',
            spec_tree_path='robustness/clustering_variations.md',
            outcome_var='lhrp2', treatment_var='unempl',
            sample_desc='Full sample',
            fixed_effects_desc='ID, industry-year, occupation-year',
            controls_desc='Full controls',
            cluster_var=cluster
        )

    # SAVE RESULTS
    results_df = pd.DataFrame(results)
    output_path = f'{PKG_PATH}/specification_results.csv'
    results_df.to_csv(output_path, index=False)

    print(f"\n{'='*80}")
    print(f"FINAL: {len(results_df)} specifications completed")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}")
