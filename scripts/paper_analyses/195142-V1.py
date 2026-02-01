"""
Specification Search Analysis for Paper 195142-V1
Title: Spillover, Efficiency and Equity Effects of Regional Firm Subsidies
Authors: Sebastian Siegloch, Nils Wehrhofer, Tobias Etzel
Journal: AEJ: Policy

Paper Overview:
--------------
This paper studies the effects of regional firm subsidies (GRW program) in Germany
on employment, wages, and other outcomes. The main identification comes from changes
in subsidy rates at the county level over time, using an event study design with
leads and lags.

Data Limitations:
----------------
The main results use restricted-access plant-level data (BHP) which is not available
in this replication package. We conduct the specification search using the available
county-level aggregate data, which can demonstrate the methodology. The treatment
variables (subsidy rate changes) and some outcome variables (GRW subsidies, unemployment,
GDP) are available at the county level.

Method Classification:
---------------------
Primary: event_study (with staggered treatment and continuous treatment intensity)
Secondary: panel_fixed_effects, difference_in_differences

The paper uses reghdfe with two-way fixed effects (unit + time) and clusters at the
local labor market (amr) level.
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SETUP AND DATA LOADING
# =============================================================================

PAPER_ID = "195142-V1"
JOURNAL = "AEJ-Policy"
PAPER_TITLE = "Spillover, Efficiency and Equity Effects of Regional Firm Subsidies"
BASE_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search"
PACKAGE_PATH = f"{BASE_PATH}/data/downloads/extracted/{PAPER_ID}"

# Load data
reforms = pd.read_stata(f"{PACKAGE_PATH}/BHP/orig/reforms_germany.dta")
external = pd.read_stata(f"{PACKAGE_PATH}/BHP/orig/external_data.dta")

# Merge to create East German county panel
df = reforms.merge(external, on=['ao_kreis', 'year'], how='inner')

# Create outcome variables
df['ln_grw_total'] = np.log(df['grw_total'].replace(0, np.nan) + 1)
df['ln_grw_vol'] = np.log(df['grw_vol'].replace(0, np.nan) + 1)
df['ln_unemp'] = np.log(df['unemp'].replace(0, np.nan))
df['ln_laborforce'] = np.log(df['laborforce'].replace(0, np.nan))
df['ln_population'] = np.log(df['population'].replace(0, np.nan))
df['ln_gdp_pc'] = np.log(df['gdp_pc'].replace(0, np.nan))
df['unemp_rate'] = df['unemp'] / df['laborforce']

# Create state-year interaction for fixed effects
df['state_year'] = df['state'].astype(str) + '_' + df['year'].astype(str)

# Filter to sample period used in paper (1996-2017)
df = df[(df['year'] >= 1996) & (df['year'] <= 2017)]

print(f"Sample size: {len(df)} county-years")
print(f"Counties: {df['ao_kreis'].nunique()}")
print(f"Years: {df['year'].min()}-{df['year'].max()}")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_results(model, spec_id, spec_tree_path, outcome_var, treatment_var,
                    df_sample, fixed_effects, controls_desc, cluster_var, model_type):
    """Extract results from pyfixest model into standard format."""

    try:
        coef = float(model.coef()[treatment_var])
        se = float(model.se()[treatment_var])
        tstat = float(model.tstat()[treatment_var])
        pval = float(model.pvalue()[treatment_var])
        ci = model.confint()
        ci_lower = float(ci.loc[treatment_var, '2.5%'])
        ci_upper = float(ci.loc[treatment_var, '97.5%'])
        nobs = int(model._N)
        r2 = float(model._r2)
    except Exception as e:
        print(f"  Error extracting results: {e}")
        return None

    # Build coefficient vector JSON
    coef_vector = {
        "treatment": {
            "var": treatment_var,
            "coef": coef,
            "se": se,
            "pval": pval
        },
        "controls": [],
        "fixed_effects": fixed_effects.split(' + ') if fixed_effects else [],
        "diagnostics": {
            "first_stage_F": None,
            "overid_pval": None,
            "hausman_pval": None
        }
    }

    # Add control coefficients if available
    for var in model.coef().index:
        if var != treatment_var:
            coef_vector["controls"].append({
                "var": var,
                "coef": float(model.coef()[var]),
                "se": float(model.se()[var]),
                "pval": float(model.pvalue()[var])
            })

    return {
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'outcome_var': outcome_var,
        'treatment_var': treatment_var,
        'coefficient': coef,
        'std_error': se,
        't_stat': tstat,
        'p_value': pval,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_obs': nobs,
        'r_squared': r2,
        'coefficient_vector_json': json.dumps(coef_vector),
        'sample_desc': f"East Germany county-level panel, {df_sample['year'].min()}-{df_sample['year'].max()}",
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': f"scripts/paper_analyses/{PAPER_ID}.py"
    }

# =============================================================================
# SPECIFICATION SEARCH
# =============================================================================

results = []

# Main treatment variables to test
treatment_vars = ['ref_medium', 'ref_small', 'ref_large']

# Outcome variables
outcome_vars = ['ln_grw_vol', 'ln_grw_total', 'unemp_rate', 'ln_unemp', 'ln_gdp_pc']

# Controls available
controls_list = ['lag_unempr', 'lag_gdp_pc', 'ln_population', 'proptaxm', 'busitaxm']

# =============================================================================
# BASELINE SPECIFICATIONS
# =============================================================================

print("\n" + "="*60)
print("BASELINE SPECIFICATIONS")
print("="*60)

# The paper's main specification uses:
# - Two-way FE: county + year (and state x year in some specs)
# - Clustering at local labor market (amr) level
# - Main outcome: log employment (not available, use GRW subsidy volume as proxy)
# - Main treatment: weighted subsidy rate change

# Baseline 1: Main county-level outcome - GRW subsidy volume
for outcome in ['ln_grw_vol', 'ln_grw_total']:
    for treatment in ['ref_medium']:
        formula = f"{outcome} ~ {treatment} | ao_kreis + year"
        try:
            model = pf.feols(formula, data=df.dropna(subset=[outcome, treatment]),
                            vcov={'CRV1': 'amr'})
            result = extract_results(
                model,
                spec_id='baseline',
                spec_tree_path='methods/event_study.md',
                outcome_var=outcome,
                treatment_var=treatment,
                df_sample=df,
                fixed_effects='ao_kreis + year',
                controls_desc='None',
                cluster_var='amr',
                model_type='TWFE'
            )
            if result:
                results.append(result)
                print(f"Baseline ({outcome}): coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
        except Exception as e:
            print(f"Error in baseline {outcome}: {e}")

# =============================================================================
# FIXED EFFECTS VARIATIONS
# =============================================================================

print("\n" + "="*60)
print("FIXED EFFECTS VARIATIONS")
print("="*60)

fe_specs = [
    ('did/fe/unit_only', 'ao_kreis', 'County FE only'),
    ('did/fe/time_only', 'year', 'Year FE only'),
    ('did/fe/twoway', 'ao_kreis + year', 'County + Year FE'),
    ('did/fe/region_x_time', 'state_year', 'State x Year FE'),
]

for spec_id, fe, fe_desc in fe_specs:
    for outcome in ['ln_grw_vol']:
        for treatment in ['ref_medium']:
            formula = f"{outcome} ~ {treatment} | {fe}"
            try:
                model = pf.feols(formula, data=df.dropna(subset=[outcome, treatment]),
                                vcov={'CRV1': 'amr'})
                result = extract_results(
                    model,
                    spec_id=spec_id,
                    spec_tree_path='methods/difference_in_differences.md#fixed-effects',
                    outcome_var=outcome,
                    treatment_var=treatment,
                    df_sample=df,
                    fixed_effects=fe,
                    controls_desc='None',
                    cluster_var='amr',
                    model_type='Panel FE'
                )
                if result:
                    results.append(result)
                    print(f"{spec_id}: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
            except Exception as e:
                print(f"Error in {spec_id}: {e}")

# =============================================================================
# CONTROL SET VARIATIONS
# =============================================================================

print("\n" + "="*60)
print("CONTROL SET VARIATIONS")
print("="*60)

control_specs = [
    ('did/controls/none', [], 'No controls'),
    ('did/controls/minimal', ['lag_unempr'], 'Lagged unemployment rate'),
    ('did/controls/baseline', ['lag_unempr', 'lag_gdp_pc'], 'Lagged unemployment rate + lagged GDP pc'),
    ('did/controls/full', ['lag_unempr', 'lag_gdp_pc', 'proptaxm', 'busitaxm'], 'All available controls'),
]

for spec_id, controls, controls_desc in control_specs:
    for outcome in ['ln_grw_vol']:
        for treatment in ['ref_medium']:
            if controls:
                controls_str = ' + '.join(controls)
                formula = f"{outcome} ~ {treatment} + {controls_str} | ao_kreis + year"
            else:
                formula = f"{outcome} ~ {treatment} | ao_kreis + year"

            # Filter for complete cases
            required_cols = [outcome, treatment] + controls
            df_sub = df.dropna(subset=required_cols)

            try:
                model = pf.feols(formula, data=df_sub, vcov={'CRV1': 'amr'})
                result = extract_results(
                    model,
                    spec_id=spec_id,
                    spec_tree_path='methods/difference_in_differences.md#control-sets',
                    outcome_var=outcome,
                    treatment_var=treatment,
                    df_sample=df_sub,
                    fixed_effects='ao_kreis + year',
                    controls_desc=controls_desc,
                    cluster_var='amr',
                    model_type='TWFE'
                )
                if result:
                    results.append(result)
                    print(f"{spec_id}: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")
            except Exception as e:
                print(f"Error in {spec_id}: {e}")

# =============================================================================
# TREATMENT VARIABLE VARIATIONS
# =============================================================================

print("\n" + "="*60)
print("TREATMENT VARIABLE VARIATIONS")
print("="*60)

treatment_specs = [
    ('did/treatment/small_firms', 'ref_small', 'Subsidy rate change for small firms'),
    ('did/treatment/medium_firms', 'ref_medium', 'Subsidy rate change for medium firms'),
    ('did/treatment/large_firms', 'ref_large', 'Subsidy rate change for large firms'),
]

for spec_id, treatment, treatment_desc in treatment_specs:
    for outcome in ['ln_grw_vol', 'unemp_rate', 'ln_gdp_pc']:
        formula = f"{outcome} ~ {treatment} | ao_kreis + year"
        try:
            model = pf.feols(formula, data=df.dropna(subset=[outcome, treatment]),
                            vcov={'CRV1': 'amr'})
            result = extract_results(
                model,
                spec_id=spec_id,
                spec_tree_path='methods/difference_in_differences.md#treatment-definition',
                outcome_var=outcome,
                treatment_var=treatment,
                df_sample=df,
                fixed_effects='ao_kreis + year',
                controls_desc='None',
                cluster_var='amr',
                model_type='TWFE'
            )
            if result:
                results.append(result)
                print(f"{spec_id} ({outcome}): coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
        except Exception as e:
            print(f"Error in {spec_id} {outcome}: {e}")

# =============================================================================
# CLUSTERING VARIATIONS
# =============================================================================

print("\n" + "="*60)
print("CLUSTERING VARIATIONS")
print("="*60)

cluster_specs = [
    ('robust/cluster/none', 'hetero', 'Robust SE (no clustering)'),
    ('robust/cluster/amr', {'CRV1': 'amr'}, 'Cluster by local labor market'),
    ('robust/cluster/county', {'CRV1': 'ao_kreis'}, 'Cluster by county'),
    ('robust/cluster/state', {'CRV1': 'state'}, 'Cluster by state'),
]

for spec_id, vcov, cluster_desc in cluster_specs:
    for outcome in ['ln_grw_vol']:
        for treatment in ['ref_medium']:
            formula = f"{outcome} ~ {treatment} | ao_kreis + year"
            try:
                model = pf.feols(formula, data=df.dropna(subset=[outcome, treatment]),
                                vcov=vcov)

                cluster_var = cluster_desc if vcov == 'hetero' else list(vcov.values())[0]
                result = extract_results(
                    model,
                    spec_id=spec_id,
                    spec_tree_path='robustness/clustering_variations.md',
                    outcome_var=outcome,
                    treatment_var=treatment,
                    df_sample=df,
                    fixed_effects='ao_kreis + year',
                    controls_desc='None',
                    cluster_var=cluster_var,
                    model_type='TWFE'
                )
                if result:
                    results.append(result)
                    print(f"{spec_id}: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
            except Exception as e:
                print(f"Error in {spec_id}: {e}")

# =============================================================================
# SAMPLE RESTRICTIONS
# =============================================================================

print("\n" + "="*60)
print("SAMPLE RESTRICTIONS")
print("="*60)

# Early period (1996-2006)
df_early = df[(df['year'] >= 1996) & (df['year'] <= 2006)]
# Late period (2007-2017)
df_late = df[(df['year'] >= 2007) & (df['year'] <= 2017)]

sample_specs = [
    ('robust/sample/full', df, 'Full sample'),
    ('robust/sample/early_period', df_early, 'Early period (1996-2006)'),
    ('robust/sample/late_period', df_late, 'Late period (2007-2017)'),
]

for spec_id, df_sample, sample_desc in sample_specs:
    for outcome in ['ln_grw_vol']:
        for treatment in ['ref_medium']:
            formula = f"{outcome} ~ {treatment} | ao_kreis + year"
            try:
                model = pf.feols(formula, data=df_sample.dropna(subset=[outcome, treatment]),
                                vcov={'CRV1': 'amr'})
                result = extract_results(
                    model,
                    spec_id=spec_id,
                    spec_tree_path='robustness/sample_restrictions.md',
                    outcome_var=outcome,
                    treatment_var=treatment,
                    df_sample=df_sample,
                    fixed_effects='ao_kreis + year',
                    controls_desc='None',
                    cluster_var='amr',
                    model_type='TWFE'
                )
                if result:
                    result['sample_desc'] = sample_desc
                    results.append(result)
                    print(f"{spec_id}: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")
            except Exception as e:
                print(f"Error in {spec_id}: {e}")

# =============================================================================
# OUTCOME VARIATIONS
# =============================================================================

print("\n" + "="*60)
print("OUTCOME VARIATIONS")
print("="*60)

outcome_specs = [
    ('custom/outcome/ln_grw_vol', 'ln_grw_vol', 'Log GRW subsidy volume'),
    ('custom/outcome/ln_grw_total', 'ln_grw_total', 'Log GRW total subsidies'),
    ('custom/outcome/unemp_rate', 'unemp_rate', 'Unemployment rate'),
    ('custom/outcome/ln_unemp', 'ln_unemp', 'Log unemployment'),
    ('custom/outcome/ln_gdp_pc', 'ln_gdp_pc', 'Log GDP per capita'),
    ('custom/outcome/ln_population', 'ln_population', 'Log population'),
]

for spec_id, outcome, outcome_desc in outcome_specs:
    for treatment in ['ref_medium']:
        formula = f"{outcome} ~ {treatment} | ao_kreis + year"
        try:
            model = pf.feols(formula, data=df.dropna(subset=[outcome, treatment]),
                            vcov={'CRV1': 'amr'})
            result = extract_results(
                model,
                spec_id=spec_id,
                spec_tree_path='methods/panel_fixed_effects.md',
                outcome_var=outcome,
                treatment_var=treatment,
                df_sample=df,
                fixed_effects='ao_kreis + year',
                controls_desc='None',
                cluster_var='amr',
                model_type='TWFE'
            )
            if result:
                results.append(result)
                print(f"{spec_id}: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
        except Exception as e:
            print(f"Error in {spec_id}: {e}")

# =============================================================================
# LEAVE-ONE-OUT ROBUSTNESS
# =============================================================================

print("\n" + "="*60)
print("LEAVE-ONE-OUT ROBUSTNESS")
print("="*60)

baseline_controls = ['lag_unempr', 'lag_gdp_pc', 'proptaxm', 'busitaxm']

for dropped_var in baseline_controls:
    remaining_controls = [c for c in baseline_controls if c != dropped_var]
    controls_str = ' + '.join(remaining_controls)

    for outcome in ['ln_grw_vol']:
        for treatment in ['ref_medium']:
            formula = f"{outcome} ~ {treatment} + {controls_str} | ao_kreis + year"
            required_cols = [outcome, treatment] + remaining_controls
            df_sub = df.dropna(subset=required_cols)

            try:
                model = pf.feols(formula, data=df_sub, vcov={'CRV1': 'amr'})
                result = extract_results(
                    model,
                    spec_id=f'robust/loo/drop_{dropped_var}',
                    spec_tree_path='robustness/leave_one_out.md',
                    outcome_var=outcome,
                    treatment_var=treatment,
                    df_sample=df_sub,
                    fixed_effects='ao_kreis + year',
                    controls_desc=f'All except {dropped_var}',
                    cluster_var='amr',
                    model_type='TWFE'
                )
                if result:
                    results.append(result)
                    print(f"drop_{dropped_var}: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
            except Exception as e:
                print(f"Error in LOO {dropped_var}: {e}")

# =============================================================================
# SINGLE COVARIATE ROBUSTNESS
# =============================================================================

print("\n" + "="*60)
print("SINGLE COVARIATE ROBUSTNESS")
print("="*60)

# Bivariate (no controls)
for outcome in ['ln_grw_vol']:
    for treatment in ['ref_medium']:
        formula = f"{outcome} ~ {treatment} | ao_kreis + year"
        try:
            model = pf.feols(formula, data=df.dropna(subset=[outcome, treatment]),
                            vcov={'CRV1': 'amr'})
            result = extract_results(
                model,
                spec_id='robust/single/none',
                spec_tree_path='robustness/single_covariate.md',
                outcome_var=outcome,
                treatment_var=treatment,
                df_sample=df,
                fixed_effects='ao_kreis + year',
                controls_desc='None (bivariate)',
                cluster_var='amr',
                model_type='TWFE'
            )
            if result:
                results.append(result)
                print(f"single/none: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
        except Exception as e:
            print(f"Error in single none: {e}")

# Single controls
for control in baseline_controls:
    for outcome in ['ln_grw_vol']:
        for treatment in ['ref_medium']:
            formula = f"{outcome} ~ {treatment} + {control} | ao_kreis + year"
            required_cols = [outcome, treatment, control]
            df_sub = df.dropna(subset=required_cols)

            try:
                model = pf.feols(formula, data=df_sub, vcov={'CRV1': 'amr'})
                result = extract_results(
                    model,
                    spec_id=f'robust/single/{control}',
                    spec_tree_path='robustness/single_covariate.md',
                    outcome_var=outcome,
                    treatment_var=treatment,
                    df_sample=df_sub,
                    fixed_effects='ao_kreis + year',
                    controls_desc=f'Only {control}',
                    cluster_var='amr',
                    model_type='TWFE'
                )
                if result:
                    results.append(result)
                    print(f"single/{control}: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
            except Exception as e:
                print(f"Error in single {control}: {e}")

# =============================================================================
# SAVE RESULTS
# =============================================================================

print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
output_path = f"{PACKAGE_PATH}/specification_results.csv"
results_df.to_csv(output_path, index=False)
print(f"Saved {len(results_df)} specifications to {output_path}")

# Summary statistics
print("\n=== SUMMARY STATISTICS ===")
print(f"Total specifications: {len(results_df)}")
print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({(results_df['coefficient'] > 0).mean()*100:.1f}%)")
print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({(results_df['p_value'] < 0.05).mean()*100:.1f}%)")
print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({(results_df['p_value'] < 0.01).mean()*100:.1f}%)")
print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")
print(f"Coefficient range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")

print("\n=== SPECIFICATION BREAKDOWN ===")
spec_categories = results_df['spec_id'].str.split('/').str[0]
for cat in spec_categories.unique():
    cat_df = results_df[spec_categories == cat]
    sig_rate = (cat_df['p_value'] < 0.05).mean() * 100
    print(f"{cat}: {len(cat_df)} specs, {sig_rate:.1f}% significant at 5%")

print("\nAnalysis complete!")
