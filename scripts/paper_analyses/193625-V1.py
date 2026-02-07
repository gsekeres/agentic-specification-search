"""
Specification Search for Paper 193625-V1
AEJ: Policy 2025

Title: Differential Effects of Recessions on Graduate Income by College Tier

This paper studies how the Great Recession differentially affected income outcomes
for college graduates depending on the selectivity of their undergraduate institution.

Main hypothesis: Elite colleges protected their graduates from recession effects more
than less selective institutions.

Method: Difference-in-differences / Event Study
- Treatment: badreccz (severely affected commuting zone)
- Unit: University (super_opeid)
- Time: Birth cohort (1980-1991)
- Outcome: Log median income (lnk_medpos)
- Key interaction: cohort x tier x recession severity

Fixed Effects: super_opeid + cz*cohort (or simpler variants)
Clustering: super_opeid
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/193625-V1/ReplicationPackage_FinalAccept/ReplicationPackage"
OUTPUT_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/193625-V1"

# Paper metadata
PAPER_ID = "193625-V1"
JOURNAL = "AEJ-Policy"
PAPER_TITLE = "Differential Effects of Recessions on Graduate Income by College Tier"

def load_and_prepare_data():
    """Load and prepare the mobility dataset for analysis."""
    df = pd.read_stata(f"{DATA_DIR}/Mobility_byTier20Aug21.dta", convert_categoricals=False)

    # Create tierbarrU if not exists (collapsed tier groups)
    if 'tierbarrU' not in df.columns:
        df['tierbarrU'] = np.nan
        df.loc[df['tier'] == 1, 'tierbarrU'] = 1
        df.loc[df['tier'] == 2, 'tierbarrU'] = 2
        df.loc[df['tier'].isin([3, 4]), 'tierbarrU'] = 3
        df.loc[df['tier'].isin([5, 6]), 'tierbarrU'] = 4
        df.loc[df['tier'].isin([7, 8]), 'tierbarrU'] = 5
        df.loc[df['tier'] == 9, 'tierbarrU'] = 6

    # Create log count
    df['lncount'] = np.log(df['count'])

    # Create interaction variables for cohort x badreccz
    # The key treatment effect is post-1983 cohorts x badreccz (recession severity)
    # 1983 is the treatment year (cohort that graduated around 2005-2007, entering labor market during recession)
    df['post'] = (df['cohort'] >= 1984).astype(float)
    df['treat'] = df['post'] * df['badreccz']

    # For tier-specific analysis
    for tier in [1, 2, 3, 4, 5, 6]:
        df[f'tier{tier}'] = (df['tierbarrU'] == tier).astype(float)
        df[f'post_tier{tier}'] = df['post'] * df[f'tier{tier}']
        df[f'treat_tier{tier}'] = df['treat'] * df[f'tier{tier}']

    # Sample flag from original do file: tagunivyr
    df = df.sort_values(['super_opeid', 'cohort'])
    df['tagunivyr'] = df.groupby(['super_opeid', 'cohort']).cumcount() == 0

    # Create sample restriction for main analysis
    df['main_sample'] = (
        (df['super_opeid'] != -1) &
        (df['multi'] == 0) &
        (df['tagunivyr'] == True) &
        (df['lnk_medpos'].notna()) &
        (df['badreccz'].notna())
    )

    # Select only necessary columns and drop NaN in key variables
    keep_cols = ['super_opeid', 'cohort', 'cz', 'tierbarrU', 'lnk_medpos', 'lnk_median', 'lnk_mean',
                 'k_q5', 'k_q4', 'k_q3', 'k_q2', 'k_q1', 'k_top10pc', 'k_top5pc', 'k_top1pc', 'k_0inc',
                 'par_q1', 'par_q2', 'par_q3', 'par_q4', 'par_q5', 'par_top10pc', 'par_top5pc',
                 'par_top1pc', 'par_toppt1pc', 'female', 'lncount', 'count', 'badreccz', 'shock',
                 'post', 'treat', 'tier1', 'tier2', 'tier3', 'tier4', 'tier5', 'tier6',
                 'post_tier1', 'post_tier2', 'post_tier3', 'post_tier4', 'post_tier5', 'post_tier6',
                 'treat_tier1', 'treat_tier2', 'treat_tier3', 'treat_tier4', 'treat_tier5', 'treat_tier6',
                 'multi', 'tagunivyr', 'main_sample']

    # Add any other columns that exist
    for col in keep_cols:
        if col not in df.columns:
            print(f"Warning: {col} not in df")

    available_cols = [c for c in keep_cols if c in df.columns]
    df = df[available_cols].copy()

    return df

def run_spec(df, formula, vcov_spec, sample_filter=None, treatment_var='treat',
             spec_id='', spec_tree_path='', outcome_var='lnk_medpos', controls_desc='',
             fixed_effects_desc='', model_type='FE', sample_desc='Full sample'):
    """Run a single specification and return results dictionary."""

    try:
        # Apply sample filter
        data = df[df['main_sample'] == True].copy() if sample_filter is None else df[sample_filter].copy()

        # Drop missing values for this spec
        data = data.dropna(subset=['super_opeid', 'cohort'])

        # Run regression
        model = pf.feols(formula, data=data, vcov=vcov_spec)

        # Extract coefficient for treatment variable
        coefs = model.coef()
        ses = model.se()
        pvals = model.pvalue()

        # Find treatment variable coefficient
        if treatment_var in coefs.index:
            coef = coefs[treatment_var]
            se = ses[treatment_var]
            pval = pvals[treatment_var]
            tstat = coef / se if se > 0 else np.nan
            ci_lower = coef - 1.96 * se
            ci_upper = coef + 1.96 * se
        else:
            # Try to find partial match
            matching_vars = [v for v in coefs.index if treatment_var in v]
            if matching_vars:
                tv = matching_vars[0]
                coef = coefs[tv]
                se = ses[tv]
                pval = pvals[tv]
                tstat = coef / se if se > 0 else np.nan
                ci_lower = coef - 1.96 * se
                ci_upper = coef + 1.96 * se
            else:
                coef = se = pval = tstat = ci_lower = ci_upper = np.nan

        # Build coefficient vector JSON
        coef_vector = {
            "treatment": {"var": treatment_var, "coef": float(coef), "se": float(se), "pval": float(pval)},
            "controls": [],
            "fixed_effects": fixed_effects_desc.split(' + ') if fixed_effects_desc else [],
            "diagnostics": {"first_stage_F": None, "overid_pval": None}
        }
        for var in coefs.index:
            if var != treatment_var and not var.startswith('_'):
                coef_vector["controls"].append({
                    "var": var,
                    "coef": float(coefs[var]),
                    "se": float(ses[var]),
                    "pval": float(pvals[var])
                })

        result = {
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
            'n_obs': int(model._N),
            'r_squared': float(model._r2) if hasattr(model, '_r2') else np.nan,
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects_desc,
            'controls_desc': controls_desc,
            'cluster_var': str(vcov_spec),
            'model_type': model_type,
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }

        return result

    except Exception as e:
        print(f"Error in spec {spec_id}: {str(e)[:100]}")
        return {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': np.nan,
            'std_error': np.nan,
            't_stat': np.nan,
            'p_value': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n_obs': np.nan,
            'r_squared': np.nan,
            'coefficient_vector_json': json.dumps({"error": str(e)[:200]}),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects_desc,
            'controls_desc': controls_desc,
            'cluster_var': str(vcov_spec),
            'model_type': model_type,
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }

def main():
    """Run all specifications and save results."""

    print("Loading and preparing data...")
    df = load_and_prepare_data()
    print(f"Data loaded: {len(df)} observations, {df['main_sample'].sum()} in main sample")

    results = []

    # Define control sets
    controls_basic = ['par_q1', 'par_q2', 'par_q3', 'par_q4', 'par_q5', 'lncount',
                      'par_top10pc', 'par_top5pc', 'par_top1pc', 'par_toppt1pc', 'female']
    controls_minimal = ['female', 'lncount']
    controls_parental = ['par_q1', 'par_q2', 'par_q3', 'par_q4', 'par_q5', 'par_top10pc']

    # Standard clustering
    cluster_spec = {'CRV1': 'super_opeid'}

    # ============================================
    # BASELINE SPECIFICATIONS
    # ============================================
    print("\nRunning baseline specifications...")

    # Baseline: Simple DiD with post*badreccz interaction
    results.append(run_spec(
        df,
        formula="lnk_medpos ~ treat + " + " + ".join(controls_basic) + " | super_opeid + cohort",
        vcov_spec=cluster_spec,
        spec_id='baseline',
        spec_tree_path='methods/difference_in_differences.md#baseline',
        controls_desc='Full controls',
        fixed_effects_desc='super_opeid + cohort'
    ))

    # ============================================
    # FIXED EFFECTS VARIATIONS (did/fe/*)
    # ============================================
    print("Running fixed effects variations...")

    # Unit FE only
    results.append(run_spec(
        df,
        formula="lnk_medpos ~ treat + " + " + ".join(controls_basic) + " | super_opeid",
        vcov_spec=cluster_spec,
        spec_id='did/fe/unit_only',
        spec_tree_path='methods/difference_in_differences.md#fixed-effects',
        controls_desc='Full controls',
        fixed_effects_desc='super_opeid'
    ))

    # Time FE only
    results.append(run_spec(
        df,
        formula="lnk_medpos ~ treat + " + " + ".join(controls_basic) + " | cohort",
        vcov_spec=cluster_spec,
        spec_id='did/fe/time_only',
        spec_tree_path='methods/difference_in_differences.md#fixed-effects',
        controls_desc='Full controls',
        fixed_effects_desc='cohort'
    ))

    # Two-way FE (standard)
    results.append(run_spec(
        df,
        formula="lnk_medpos ~ treat + " + " + ".join(controls_basic) + " | super_opeid + cohort",
        vcov_spec=cluster_spec,
        spec_id='did/fe/twoway',
        spec_tree_path='methods/difference_in_differences.md#fixed-effects',
        controls_desc='Full controls',
        fixed_effects_desc='super_opeid + cohort'
    ))

    # CZ x cohort FE (paper's main specification)
    results.append(run_spec(
        df,
        formula="lnk_medpos ~ treat + " + " + ".join(controls_basic) + " | super_opeid + cz^cohort",
        vcov_spec=cluster_spec,
        spec_id='did/fe/cz_x_cohort',
        spec_tree_path='methods/difference_in_differences.md#fixed-effects',
        controls_desc='Full controls',
        fixed_effects_desc='super_opeid + cz*cohort'
    ))

    # No fixed effects
    results.append(run_spec(
        df,
        formula="lnk_medpos ~ treat + " + " + ".join(controls_basic),
        vcov_spec=cluster_spec,
        spec_id='did/fe/none',
        spec_tree_path='methods/difference_in_differences.md#fixed-effects',
        controls_desc='Full controls',
        fixed_effects_desc='None',
        model_type='OLS'
    ))

    # Tier x cohort FE
    results.append(run_spec(
        df,
        formula="lnk_medpos ~ treat + " + " + ".join(controls_basic) + " | super_opeid + tierbarrU^cohort",
        vcov_spec=cluster_spec,
        spec_id='did/fe/tier_x_cohort',
        spec_tree_path='methods/difference_in_differences.md#fixed-effects',
        controls_desc='Full controls',
        fixed_effects_desc='super_opeid + tier*cohort'
    ))

    # ============================================
    # CONTROL SET VARIATIONS (did/controls/*)
    # ============================================
    print("Running control set variations...")

    # No controls
    results.append(run_spec(
        df,
        formula="lnk_medpos ~ treat | super_opeid + cohort",
        vcov_spec=cluster_spec,
        spec_id='did/controls/none',
        spec_tree_path='methods/difference_in_differences.md#control-sets',
        controls_desc='No controls',
        fixed_effects_desc='super_opeid + cohort'
    ))

    # Minimal controls
    results.append(run_spec(
        df,
        formula="lnk_medpos ~ treat + " + " + ".join(controls_minimal) + " | super_opeid + cohort",
        vcov_spec=cluster_spec,
        spec_id='did/controls/minimal',
        spec_tree_path='methods/difference_in_differences.md#control-sets',
        controls_desc='Minimal (female, lncount)',
        fixed_effects_desc='super_opeid + cohort'
    ))

    # Parental controls only
    results.append(run_spec(
        df,
        formula="lnk_medpos ~ treat + " + " + ".join(controls_parental) + " | super_opeid + cohort",
        vcov_spec=cluster_spec,
        spec_id='did/controls/parental_only',
        spec_tree_path='methods/difference_in_differences.md#control-sets',
        controls_desc='Parental income controls only',
        fixed_effects_desc='super_opeid + cohort'
    ))

    # ============================================
    # LEAVE-ONE-OUT CONTROL VARIATIONS
    # ============================================
    print("Running leave-one-out control variations...")

    for ctrl in controls_basic:
        remaining = [c for c in controls_basic if c != ctrl]
        results.append(run_spec(
            df,
            formula="lnk_medpos ~ treat + " + " + ".join(remaining) + " | super_opeid + cohort",
            vcov_spec=cluster_spec,
            spec_id=f'robust/control/drop_{ctrl}',
            spec_tree_path='robustness/leave_one_out.md',
            controls_desc=f'Full controls minus {ctrl}',
            fixed_effects_desc='super_opeid + cohort'
        ))

    # ============================================
    # ALTERNATIVE OUTCOMES
    # ============================================
    print("Running alternative outcome specifications...")

    outcome_vars = {
        'lnk_median': 'Log median income (including zeros)',
        'lnk_mean': 'Log mean income',
        'k_q5': 'Fraction in top income quintile',
        'k_q4': 'Fraction in 4th income quintile',
        'k_q3': 'Fraction in 3rd income quintile',
        'k_q2': 'Fraction in 2nd income quintile',
        'k_q1': 'Fraction in bottom income quintile',
        'k_top10pc': 'Fraction in top 10% income',
        'k_top5pc': 'Fraction in top 5% income',
        'k_top1pc': 'Fraction in top 1% income',
        'k_0inc': 'Fraction with zero earnings'
    }

    for outcome, desc in outcome_vars.items():
        results.append(run_spec(
            df,
            formula=f"{outcome} ~ treat + " + " + ".join(controls_basic) + " | super_opeid + cohort",
            vcov_spec=cluster_spec,
            spec_id=f'robust/outcome/{outcome}',
            spec_tree_path='robustness/measurement.md',
            outcome_var=outcome,
            controls_desc='Full controls',
            fixed_effects_desc='super_opeid + cohort',
            sample_desc=f'Full sample - {desc}'
        ))

    # ============================================
    # SAMPLE RESTRICTIONS
    # ============================================
    print("Running sample restriction specifications...")

    # By tier
    for tier in [1, 2, 3, 4, 5, 6]:
        tier_filter = (df['main_sample'] == True) & (df['tierbarrU'] == tier)
        results.append(run_spec(
            df,
            formula="lnk_medpos ~ treat + " + " + ".join(controls_basic) + " | super_opeid + cohort",
            vcov_spec=cluster_spec,
            sample_filter=tier_filter,
            spec_id=f'robust/sample/tier{tier}_only',
            spec_tree_path='robustness/sample_restrictions.md',
            controls_desc='Full controls',
            fixed_effects_desc='super_opeid + cohort',
            sample_desc=f'Tier {tier} universities only'
        ))

    # Elite vs non-elite
    elite_filter = (df['main_sample'] == True) & (df['tierbarrU'] <= 2)
    results.append(run_spec(
        df,
        formula="lnk_medpos ~ treat + " + " + ".join(controls_basic) + " | super_opeid + cohort",
        vcov_spec=cluster_spec,
        sample_filter=elite_filter,
        spec_id='robust/sample/elite_only',
        spec_tree_path='robustness/sample_restrictions.md',
        controls_desc='Full controls',
        fixed_effects_desc='super_opeid + cohort',
        sample_desc='Elite universities only (Tier 1-2)'
    ))

    non_elite_filter = (df['main_sample'] == True) & (df['tierbarrU'] >= 3)
    results.append(run_spec(
        df,
        formula="lnk_medpos ~ treat + " + " + ".join(controls_basic) + " | super_opeid + cohort",
        vcov_spec=cluster_spec,
        sample_filter=non_elite_filter,
        spec_id='robust/sample/non_elite_only',
        spec_tree_path='robustness/sample_restrictions.md',
        controls_desc='Full controls',
        fixed_effects_desc='super_opeid + cohort',
        sample_desc='Non-elite universities only (Tier 3-6)'
    ))

    # Drop each cohort
    for cohort_yr in range(1980, 1992):
        cohort_filter = (df['main_sample'] == True) & (df['cohort'] != cohort_yr)
        results.append(run_spec(
            df,
            formula="lnk_medpos ~ treat + " + " + ".join(controls_basic) + " | super_opeid + cohort",
            vcov_spec=cluster_spec,
            sample_filter=cohort_filter,
            spec_id=f'robust/sample/drop_cohort{cohort_yr}',
            spec_tree_path='robustness/sample_restrictions.md',
            controls_desc='Full controls',
            fixed_effects_desc='super_opeid + cohort',
            sample_desc=f'Drop cohort {cohort_yr}'
        ))

    # Pre-recession only (cohorts 1980-1983)
    pre_filter = (df['main_sample'] == True) & (df['cohort'] <= 1983)
    results.append(run_spec(
        df,
        formula="lnk_medpos ~ treat + " + " + ".join(controls_basic) + " | super_opeid + cohort",
        vcov_spec=cluster_spec,
        sample_filter=pre_filter,
        spec_id='robust/sample/pre_recession_cohorts',
        spec_tree_path='robustness/sample_restrictions.md',
        controls_desc='Full controls',
        fixed_effects_desc='super_opeid + cohort',
        sample_desc='Pre-recession cohorts only (1980-1983)'
    ))

    # Post-recession only (cohorts 1984+)
    post_filter = (df['main_sample'] == True) & (df['cohort'] >= 1984)
    results.append(run_spec(
        df,
        formula="lnk_medpos ~ treat + " + " + ".join(controls_basic) + " | super_opeid + cohort",
        vcov_spec=cluster_spec,
        sample_filter=post_filter,
        spec_id='robust/sample/post_recession_cohorts',
        spec_tree_path='robustness/sample_restrictions.md',
        controls_desc='Full controls',
        fixed_effects_desc='super_opeid + cohort',
        sample_desc='Post-recession cohorts only (1984+)'
    ))

    # Severely affected CZs only
    severe_filter = (df['main_sample'] == True) & (df['badreccz'] == 1)
    results.append(run_spec(
        df,
        formula="lnk_medpos ~ post + " + " + ".join(controls_basic) + " | super_opeid + cohort",
        vcov_spec=cluster_spec,
        sample_filter=severe_filter,
        treatment_var='post',
        spec_id='robust/sample/severe_cz_only',
        spec_tree_path='robustness/sample_restrictions.md',
        controls_desc='Full controls',
        fixed_effects_desc='super_opeid + cohort',
        sample_desc='Severely affected CZs only'
    ))

    # Mildly affected CZs only
    mild_filter = (df['main_sample'] == True) & (df['badreccz'] == 0)
    results.append(run_spec(
        df,
        formula="lnk_medpos ~ post + " + " + ".join(controls_basic) + " | super_opeid + cohort",
        vcov_spec=cluster_spec,
        sample_filter=mild_filter,
        treatment_var='post',
        spec_id='robust/sample/mild_cz_only',
        spec_tree_path='robustness/sample_restrictions.md',
        controls_desc='Full controls',
        fixed_effects_desc='super_opeid + cohort',
        sample_desc='Mildly affected CZs only'
    ))

    # ============================================
    # INFERENCE VARIATIONS (CLUSTERING)
    # ============================================
    print("Running inference/clustering variations...")

    # Cluster by CZ
    results.append(run_spec(
        df,
        formula="lnk_medpos ~ treat + " + " + ".join(controls_basic) + " | super_opeid + cohort",
        vcov_spec={'CRV1': 'cz'},
        spec_id='robust/cluster/cz',
        spec_tree_path='robustness/clustering_variations.md',
        controls_desc='Full controls',
        fixed_effects_desc='super_opeid + cohort'
    ))

    # Heteroskedasticity-robust (no clustering)
    results.append(run_spec(
        df,
        formula="lnk_medpos ~ treat + " + " + ".join(controls_basic) + " | super_opeid + cohort",
        vcov_spec='hetero',
        spec_id='robust/cluster/robust_hc',
        spec_tree_path='robustness/clustering_variations.md',
        controls_desc='Full controls',
        fixed_effects_desc='super_opeid + cohort'
    ))

    # ============================================
    # FUNCTIONAL FORM
    # ============================================
    print("Running functional form variations...")

    # Levels instead of logs (k_median_nozero is lnk_medpos in levels - need to compute from data)
    # k_median_nozero is stored separately as the raw median income
    # Since k_median_nozero may not be in our reduced df, we'll compute levels from log
    df['k_median_levels'] = np.exp(df['lnk_medpos'])
    results.append(run_spec(
        df,
        formula="k_median_levels ~ treat + " + " + ".join(controls_basic) + " | super_opeid + cohort",
        vcov_spec=cluster_spec,
        spec_id='robust/funcform/levels_outcome',
        spec_tree_path='robustness/functional_form.md',
        outcome_var='k_median_levels',
        controls_desc='Full controls',
        fixed_effects_desc='super_opeid + cohort',
        sample_desc='Outcome in levels (not log)'
    ))

    # IHS transformation
    df['ihs_median'] = np.arcsinh(df['k_median_levels'])
    results.append(run_spec(
        df,
        formula="ihs_median ~ treat + " + " + ".join(controls_basic) + " | super_opeid + cohort",
        vcov_spec=cluster_spec,
        spec_id='robust/funcform/ihs_outcome',
        spec_tree_path='robustness/functional_form.md',
        outcome_var='ihs_median',
        controls_desc='Full controls',
        fixed_effects_desc='super_opeid + cohort',
        sample_desc='IHS transformation of outcome'
    ))

    # ============================================
    # ALTERNATIVE TREATMENT DEFINITIONS
    # ============================================
    print("Running alternative treatment specifications...")

    # Continuous shock measure
    df['treat_continuous'] = df['post'] * df['shock']
    results.append(run_spec(
        df,
        formula="lnk_medpos ~ treat_continuous + " + " + ".join(controls_basic) + " | super_opeid + cohort",
        vcov_spec=cluster_spec,
        treatment_var='treat_continuous',
        spec_id='robust/treatment/continuous_shock',
        spec_tree_path='methods/difference_in_differences.md#treatment-definition',
        controls_desc='Full controls',
        fixed_effects_desc='super_opeid + cohort',
        sample_desc='Continuous recession shock measure'
    ))

    # Different threshold for badreccz (above/below median shock)
    median_shock = df['shock'].median()
    df['badreccz_median'] = (df['shock'] > median_shock).astype(float)
    df['treat_median'] = df['post'] * df['badreccz_median']
    results.append(run_spec(
        df,
        formula="lnk_medpos ~ treat_median + " + " + ".join(controls_basic) + " | super_opeid + cohort",
        vcov_spec=cluster_spec,
        treatment_var='treat_median',
        spec_id='robust/treatment/median_threshold',
        spec_tree_path='methods/difference_in_differences.md#treatment-definition',
        controls_desc='Full controls',
        fixed_effects_desc='super_opeid + cohort',
        sample_desc='Recession severity at median threshold'
    ))

    # Top quartile shock
    q75_shock = df['shock'].quantile(0.75)
    df['badreccz_q75'] = (df['shock'] > q75_shock).astype(float)
    df['treat_q75'] = df['post'] * df['badreccz_q75']
    results.append(run_spec(
        df,
        formula="lnk_medpos ~ treat_q75 + " + " + ".join(controls_basic) + " | super_opeid + cohort",
        vcov_spec=cluster_spec,
        treatment_var='treat_q75',
        spec_id='robust/treatment/q75_threshold',
        spec_tree_path='methods/difference_in_differences.md#treatment-definition',
        controls_desc='Full controls',
        fixed_effects_desc='super_opeid + cohort',
        sample_desc='Recession severity at 75th percentile threshold'
    ))

    # ============================================
    # PLACEBO TESTS
    # ============================================
    print("Running placebo tests...")

    # Fake treatment year (1981 instead of 1983)
    df['post_fake1981'] = (df['cohort'] >= 1981).astype(float)
    df['treat_fake1981'] = df['post_fake1981'] * df['badreccz']
    pre_1984_filter = (df['main_sample'] == True) & (df['cohort'] < 1984)
    results.append(run_spec(
        df,
        formula="lnk_medpos ~ treat_fake1981 + " + " + ".join(controls_basic) + " | super_opeid + cohort",
        vcov_spec=cluster_spec,
        sample_filter=pre_1984_filter,
        treatment_var='treat_fake1981',
        spec_id='robust/placebo/fake_treatment_1981',
        spec_tree_path='robustness/placebo_tests.md',
        controls_desc='Full controls',
        fixed_effects_desc='super_opeid + cohort',
        sample_desc='Pre-treatment sample only (cohorts 1980-1983), fake treatment year 1981'
    ))

    # Fake treatment year (1982)
    df['post_fake1982'] = (df['cohort'] >= 1982).astype(float)
    df['treat_fake1982'] = df['post_fake1982'] * df['badreccz']
    results.append(run_spec(
        df,
        formula="lnk_medpos ~ treat_fake1982 + " + " + ".join(controls_basic) + " | super_opeid + cohort",
        vcov_spec=cluster_spec,
        sample_filter=pre_1984_filter,
        treatment_var='treat_fake1982',
        spec_id='robust/placebo/fake_treatment_1982',
        spec_tree_path='robustness/placebo_tests.md',
        controls_desc='Full controls',
        fixed_effects_desc='super_opeid + cohort',
        sample_desc='Pre-treatment sample only, fake treatment year 1982'
    ))

    # ============================================
    # HETEROGENEITY ANALYSIS
    # ============================================
    print("Running heterogeneity specifications...")

    # By female proportion
    df['high_female'] = (df['female'] > df['female'].median()).astype(float)
    df['treat_x_high_female'] = df['treat'] * df['high_female']
    results.append(run_spec(
        df,
        formula="lnk_medpos ~ treat + treat_x_high_female + high_female + " + " + ".join([c for c in controls_basic if c != 'female']) + " | super_opeid + cohort",
        vcov_spec=cluster_spec,
        treatment_var='treat_x_high_female',
        spec_id='robust/heterogeneity/by_female',
        spec_tree_path='robustness/heterogeneity.md',
        controls_desc='Full controls',
        fixed_effects_desc='super_opeid + cohort',
        sample_desc='Interaction with high female share'
    ))

    # By parental income
    df['high_par_income'] = (df['par_q5'] > df['par_q5'].median()).astype(float)
    df['treat_x_high_par'] = df['treat'] * df['high_par_income']
    results.append(run_spec(
        df,
        formula="lnk_medpos ~ treat + treat_x_high_par + high_par_income + " + " + ".join([c for c in controls_basic if c not in ['par_q1', 'par_q2', 'par_q3', 'par_q4', 'par_q5']]) + " | super_opeid + cohort",
        vcov_spec=cluster_spec,
        treatment_var='treat_x_high_par',
        spec_id='robust/heterogeneity/by_parental_income',
        spec_tree_path='robustness/heterogeneity.md',
        controls_desc='Full controls',
        fixed_effects_desc='super_opeid + cohort',
        sample_desc='Interaction with high parental income'
    ))

    # By university size
    df['large_univ'] = (df['count'] > df['count'].median()).astype(float)
    df['treat_x_large'] = df['treat'] * df['large_univ']
    results.append(run_spec(
        df,
        formula="lnk_medpos ~ treat + treat_x_large + large_univ + " + " + ".join([c for c in controls_basic if c != 'lncount']) + " | super_opeid + cohort",
        vcov_spec=cluster_spec,
        treatment_var='treat_x_large',
        spec_id='robust/heterogeneity/by_univ_size',
        spec_tree_path='robustness/heterogeneity.md',
        controls_desc='Full controls',
        fixed_effects_desc='super_opeid + cohort',
        sample_desc='Interaction with large university'
    ))

    # ============================================
    # TIER-SPECIFIC TREATMENT EFFECTS (Triple-Difference style)
    # ============================================
    print("Running tier-specific treatment effect specifications...")

    # Triple-difference: treat x tier interactions
    for tier in [1, 2, 3, 4, 5, 6]:
        results.append(run_spec(
            df,
            formula=f"lnk_medpos ~ treat + treat_tier{tier} + tier{tier} + " + " + ".join(controls_basic) + " | super_opeid + cohort",
            vcov_spec=cluster_spec,
            treatment_var=f'treat_tier{tier}',
            spec_id=f'robust/heterogeneity/treat_x_tier{tier}',
            spec_tree_path='robustness/heterogeneity.md',
            controls_desc='Full controls',
            fixed_effects_desc='super_opeid + cohort',
            sample_desc=f'Treatment x Tier {tier} interaction'
        ))

    # Elite vs non-elite interaction
    df['elite'] = (df['tierbarrU'] <= 2).astype(float)
    df['treat_x_elite'] = df['treat'] * df['elite']
    results.append(run_spec(
        df,
        formula="lnk_medpos ~ treat + treat_x_elite + elite + " + " + ".join(controls_basic) + " | super_opeid + cohort",
        vcov_spec=cluster_spec,
        treatment_var='treat_x_elite',
        spec_id='robust/heterogeneity/treat_x_elite',
        spec_tree_path='robustness/heterogeneity.md',
        controls_desc='Full controls',
        fixed_effects_desc='super_opeid + cohort',
        sample_desc='Treatment x Elite (Tier 1-2) interaction'
    ))

    # ============================================
    # PROGRESSIVE CONTROL ADDITION
    # ============================================
    print("Running progressive control addition...")

    progressive_controls = ['female', 'lncount', 'par_q1', 'par_q2', 'par_q3', 'par_q4', 'par_q5', 'par_top10pc']
    for i in range(1, len(progressive_controls) + 1):
        controls_subset = progressive_controls[:i]
        results.append(run_spec(
            df,
            formula="lnk_medpos ~ treat + " + " + ".join(controls_subset) + " | super_opeid + cohort",
            vcov_spec=cluster_spec,
            spec_id=f'robust/control/progressive_{i}',
            spec_tree_path='robustness/control_progression.md',
            controls_desc=f'Progressive controls ({i}): ' + ', '.join(controls_subset),
            fixed_effects_desc='super_opeid + cohort'
        ))

    # ============================================
    # ADDITIONAL ROBUSTNESS
    # ============================================
    print("Running additional robustness checks...")

    # Winsorize outcomes at 1%
    df['lnk_medpos_wins1'] = df['lnk_medpos'].clip(
        lower=df['lnk_medpos'].quantile(0.01),
        upper=df['lnk_medpos'].quantile(0.99)
    )
    results.append(run_spec(
        df,
        formula="lnk_medpos_wins1 ~ treat + " + " + ".join(controls_basic) + " | super_opeid + cohort",
        vcov_spec=cluster_spec,
        spec_id='robust/sample/winsorize_1pct',
        spec_tree_path='robustness/sample_restrictions.md',
        outcome_var='lnk_medpos_wins1',
        controls_desc='Full controls',
        fixed_effects_desc='super_opeid + cohort',
        sample_desc='Outcome winsorized at 1%'
    ))

    # Winsorize at 5%
    df['lnk_medpos_wins5'] = df['lnk_medpos'].clip(
        lower=df['lnk_medpos'].quantile(0.05),
        upper=df['lnk_medpos'].quantile(0.95)
    )
    results.append(run_spec(
        df,
        formula="lnk_medpos_wins5 ~ treat + " + " + ".join(controls_basic) + " | super_opeid + cohort",
        vcov_spec=cluster_spec,
        spec_id='robust/sample/winsorize_5pct',
        spec_tree_path='robustness/sample_restrictions.md',
        outcome_var='lnk_medpos_wins5',
        controls_desc='Full controls',
        fixed_effects_desc='super_opeid + cohort',
        sample_desc='Outcome winsorized at 5%'
    ))

    # Trim extreme values
    trim_filter = (df['main_sample'] == True) & \
                  (df['lnk_medpos'] > df['lnk_medpos'].quantile(0.01)) & \
                  (df['lnk_medpos'] < df['lnk_medpos'].quantile(0.99))
    results.append(run_spec(
        df,
        formula="lnk_medpos ~ treat + " + " + ".join(controls_basic) + " | super_opeid + cohort",
        vcov_spec=cluster_spec,
        sample_filter=trim_filter,
        spec_id='robust/sample/trim_1pct',
        spec_tree_path='robustness/sample_restrictions.md',
        controls_desc='Full controls',
        fixed_effects_desc='super_opeid + cohort',
        sample_desc='Trim top/bottom 1%'
    ))

    # ============================================
    # SAVE RESULTS
    # ============================================
    print(f"\nTotal specifications run: {len(results)}")

    results_df = pd.DataFrame(results)
    output_path = f"{OUTPUT_DIR}/specification_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

    # Summary statistics
    valid_results = results_df[results_df['coefficient'].notna()]
    print(f"\nValid specifications: {len(valid_results)}")
    print(f"Positive coefficients: {(valid_results['coefficient'] > 0).sum()} ({100*(valid_results['coefficient'] > 0).mean():.1f}%)")
    print(f"Significant at 5%: {(valid_results['p_value'] < 0.05).sum()} ({100*(valid_results['p_value'] < 0.05).mean():.1f}%)")
    print(f"Significant at 1%: {(valid_results['p_value'] < 0.01).sum()} ({100*(valid_results['p_value'] < 0.01).mean():.1f}%)")
    print(f"Median coefficient: {valid_results['coefficient'].median():.4f}")
    print(f"Mean coefficient: {valid_results['coefficient'].mean():.4f}")
    print(f"Range: [{valid_results['coefficient'].min():.4f}, {valid_results['coefficient'].max():.4f}]")

    return results_df

if __name__ == "__main__":
    results_df = main()
