"""
Specification Search for Paper 158401-V1
========================================
Paper: Market Access and Quality Upgrading: Evidence from Four Field Experiments

This paper studies the impact of providing market access for quality maize on
farmer investment, productivity, and income in Uganda using a randomized
controlled trial (RCT).

Method Classification:
- method_code: panel_fixed_effects (with RCT design)
- method_tree_path: specification_tree/methods/panel_fixed_effects.md

Key Design Features:
- Treatment: buy_treatment (access to market for quality maize)
- Unit: Household (hhh_id)
- Time: Survey seasons (7 seasons)
- Clustering: Village (ea_code)
- ANCOVA design: Controls for baseline values
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import warnings
warnings.filterwarnings('ignore')

# Set paths
BASE_PATH = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/158401-V1'

# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_panel_data():
    """Prepare panel data replicating the Stata data processing."""

    # Load raw panel data
    panel_g1 = pd.read_stata(f'{BASE_PATH}/data/raw/panel_g1.dta')

    # Convert survey_season to numeric
    season_map = {
        'spri2017': 1, 'fall2017': 2, 'spri2018': 3, 'fall2018': 4,
        'spri2019': 5, 'fall2019': 6, 'spri2020': 7
    }
    panel_g1['survey_season_num'] = panel_g1['survey_season'].map(season_map).astype(int)

    # Create season indicators
    panel_g1['season_ante'] = (panel_g1['survey_season_num'] <= 3).astype(int)
    panel_g1['season_post'] = (panel_g1['survey_season_num'] >= 4).astype(int)

    # Create acreage variable
    plot_cols = [f'plot_size_{i}' for i in range(1, 13) if f'plot_size_{i}' in panel_g1.columns]
    panel_g1['acreage'] = panel_g1[plot_cols].sum(axis=1, min_count=1)

    # Create harvest variable
    harvest_cols = [f'harvest_kg_{i}' for i in range(1, 10) if f'harvest_kg_{i}' in panel_g1.columns]
    panel_g1['harvest_kg_tot'] = panel_g1[harvest_cols].sum(axis=1, min_count=1)
    if 'harvest_kg' in panel_g1.columns:
        panel_g1['harvest_kg_tot'] = panel_g1['harvest_kg_tot'].fillna(panel_g1['harvest_kg'])

    # Calculate yield (tons/hectare)
    panel_g1['yield_ha_ton'] = (panel_g1['harvest_kg_tot'] / panel_g1['acreage']) * 2.47105 / 1000

    # Create sold quantity
    sold_cols = [f'sold_kg_p_{i}' for i in range(1, 8) if f'sold_kg_p_{i}' in panel_g1.columns]
    panel_g1['sold_kg_tot'] = panel_g1[sold_cols].sum(axis=1, min_count=1)

    # Share sold
    panel_g1['share_sold'] = panel_g1['sold_kg_tot'] / panel_g1['harvest_kg_tot']

    # Create total revenue
    rev_cols = [f'tot_rev_combined_{i}' for i in range(1, 8) if f'tot_rev_combined_{i}' in panel_g1.columns]
    panel_g1['rev_tot'] = panel_g1[rev_cols].sum(axis=1, min_count=1)

    # Create price
    panel_g1['price_ugx'] = panel_g1['rev_tot'] / panel_g1['sold_kg_tot']

    # Create village-season average price
    panel_g1['price_vm'] = panel_g1.groupby(['ea_code', 'survey_season_num'])['price_ugx'].transform('mean')

    # Harvest value
    panel_g1['remainder'] = panel_g1['harvest_kg_tot'] - panel_g1['sold_kg_tot']
    panel_g1['harvest_value_ugx'] = np.where(
        panel_g1['remainder'] >= 0,
        panel_g1['rev_tot'].fillna(0) + panel_g1['remainder'].fillna(0) * panel_g1['price_vm'].fillna(0),
        panel_g1['harvest_kg_tot'] * panel_g1['price_ugx']
    )

    # Harvest in tons
    panel_g1['harvest_ton'] = panel_g1['harvest_kg_tot'] / 1000

    # Fill household characteristics
    panel_g1['main_road_min'] = pd.to_numeric(panel_g1['main_road_min'], errors='coerce')
    for var in ['hhr_n', 'main_road_min']:
        if var in panel_g1.columns:
            panel_g1[var] = panel_g1.groupby('hhh_id')[var].transform(lambda x: x.ffill().bfill())

    return panel_g1


def run_regression(df, formula, cluster_var=None, spec_id='baseline'):
    """Run regression and return standardized results dictionary."""
    try:
        if cluster_var and cluster_var in df.columns:
            model = pf.feols(formula, data=df, vcov={'CRV1': cluster_var})
        else:
            model = pf.feols(formula, data=df, vcov='hetero')

        # Get treatment coefficient
        coef_names = model.coef().index.tolist()
        treat_var = None
        for c in coef_names:
            if 'buy_treatment' in c.lower() or c == 'buy_treatment':
                treat_var = c
                break
        if treat_var is None:
            treat_var = coef_names[0]

        coef = model.coef()[treat_var]
        se = model.se()[treat_var]
        tstat = model.tstat()[treat_var]
        pval = model.pvalue()[treat_var]
        ci = model.confint()
        ci_lower = ci.loc[treat_var, '2.5%']
        ci_upper = ci.loc[treat_var, '97.5%']
        r2 = model._r2 if hasattr(model, '_r2') else None
        n_obs = model._N if hasattr(model, '_N') else len(df)

        # Build coefficient vector
        coef_vector = {
            'treatment': {'var': treat_var, 'coef': float(coef), 'se': float(se), 'pval': float(pval)},
            'controls': [{'var': v, 'coef': float(model.coef()[v]), 'se': float(model.se()[v]), 'pval': float(model.pvalue()[v])} for v in coef_names if v != treat_var]
        }

        return {
            'coefficient': float(coef),
            'std_error': float(se),
            't_stat': float(tstat),
            'p_value': float(pval),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_obs': int(n_obs),
            'r_squared': float(r2) if r2 else None,
            'coefficient_vector_json': json.dumps(coef_vector)
        }
    except Exception as e:
        print(f"  Error in {spec_id}: {str(e)}")
        return None


def run_specification_search():
    """Run comprehensive specification search following i4r methodology."""

    print("Loading and preparing data...")
    df = prepare_panel_data()

    results = []
    paper_id = '158401-V1'
    journal = 'AER'
    paper_title = 'Market Access and Quality Upgrading: Evidence from Four Field Experiments'
    treatment_var = 'buy_treatment'
    cluster_var = 'ea_code'
    outcome = 'yield_ha_ton'

    # Filter to post-treatment period
    df_post = df[df['season_post'] == 1].copy()

    # Create ANCOVA control (baseline yield)
    baseline_yield = df[df['survey_season_num'] == 3].groupby('hhh_id')[outcome].mean()
    pre_yield = df[df['season_ante'] == 1].groupby('hhh_id')[outcome].mean()
    df_post = df_post.merge(baseline_yield.rename('yield_baseline').reset_index(), on='hhh_id', how='left')
    df_post = df_post.merge(pre_yield.rename('yield_pre').reset_index(), on='hhh_id', how='left')
    df_post['yield_ancova'] = df_post['yield_baseline'].fillna(df_post['yield_pre'])

    print(f"Post-treatment sample: {len(df_post)} obs, {df_post['hhh_id'].nunique()} households")
    print(f"Seasons: {sorted(df_post['survey_season_num'].unique())}")
    print(f"Villages: {df_post['ea_code'].nunique()}")

    spec_count = 0

    def add_result(result, spec_id, spec_tree_path, outcome_var, sample_desc, fe_desc, controls_desc, cluster_desc='ea_code', model_type='OLS'):
        nonlocal spec_count
        if result:
            spec_count += 1
            results.append({
                'paper_id': paper_id, 'journal': journal, 'paper_title': paper_title,
                'spec_id': spec_id, 'spec_tree_path': spec_tree_path,
                'outcome_var': outcome_var, 'treatment_var': treatment_var,
                'sample_desc': sample_desc, 'fixed_effects': fe_desc,
                'controls_desc': controls_desc, 'cluster_var': cluster_desc,
                'model_type': model_type, 'estimation_script': 'specification_search.py',
                **result
            })
            print(f"  Spec {spec_count}: {spec_id} - coef={result['coefficient']:.4f}, p={result['p_value']:.4f}")

    # ==========================================================================
    # 1. BASELINE SPECIFICATIONS (5 specs)
    # ==========================================================================
    print("\n1. Running baseline specifications...")

    # Main outcome: yield
    result = run_regression(df_post, f'{outcome} ~ {treatment_var} + yield_ancova + C(survey_season_num)', cluster_var, 'baseline_yield')
    add_result(result, 'baseline_yield', 'methods/panel_fixed_effects.md#baseline', outcome, 'Post-treatment, ANCOVA', 'Season FE', 'Baseline yield')

    # Alternative outcomes
    for alt_out in ['harvest_ton', 'price_ugx', 'harvest_value_ugx', 'share_sold']:
        if alt_out in df_post.columns and df_post[alt_out].notna().sum() > 100:
            result = run_regression(df_post, f'{alt_out} ~ {treatment_var} + C(survey_season_num)', cluster_var, f'baseline_{alt_out}')
            add_result(result, f'baseline_{alt_out}', 'methods/panel_fixed_effects.md#baseline', alt_out, 'Post-treatment', 'Season FE', 'None')

    # ==========================================================================
    # 2. FIXED EFFECTS VARIATIONS (4 specs)
    # ==========================================================================
    print("\n2. Running fixed effects variations...")

    # No FE
    result = run_regression(df_post, f'{outcome} ~ {treatment_var} + yield_ancova', cluster_var, 'panel/fe/none')
    add_result(result, 'panel/fe/none', 'methods/panel_fixed_effects.md#fixed-effects', outcome, 'Post-treatment', 'None', 'Baseline yield')

    # Season FE only
    result = run_regression(df_post, f'{outcome} ~ {treatment_var} + yield_ancova + C(survey_season_num)', cluster_var, 'panel/fe/season')
    add_result(result, 'panel/fe/season', 'methods/panel_fixed_effects.md#fixed-effects', outcome, 'Post-treatment', 'Season FE', 'Baseline yield')

    # Village FE only
    result = run_regression(df_post, f'{outcome} ~ {treatment_var} + yield_ancova + C(ea_code)', cluster_var, 'panel/fe/village')
    add_result(result, 'panel/fe/village', 'methods/panel_fixed_effects.md#fixed-effects', outcome, 'Post-treatment', 'Village FE', 'Baseline yield')

    # Two-way FE
    result = run_regression(df_post, f'{outcome} ~ {treatment_var} + yield_ancova + C(survey_season_num) + C(ea_code)', cluster_var, 'panel/fe/twoway')
    add_result(result, 'panel/fe/twoway', 'methods/panel_fixed_effects.md#fixed-effects', outcome, 'Post-treatment', 'Season + Village FE', 'Baseline yield')

    # ==========================================================================
    # 3. CONTROL VARIATIONS (8 specs)
    # ==========================================================================
    print("\n3. Running control variations...")

    # No controls
    result = run_regression(df_post, f'{outcome} ~ {treatment_var} + C(survey_season_num)', cluster_var, 'did/controls/none')
    add_result(result, 'did/controls/none', 'methods/difference_in_differences.md#control-sets', outcome, 'Post-treatment', 'Season FE', 'None')

    # With HH characteristics
    result = run_regression(df_post, f'{outcome} ~ {treatment_var} + yield_ancova + hhr_n + C(survey_season_num)', cluster_var, 'did/controls/hhr_n')
    add_result(result, 'did/controls/hhr_n', 'methods/difference_in_differences.md#control-sets', outcome, 'Post-treatment', 'Season FE', 'Baseline + HH size')

    result = run_regression(df_post, f'{outcome} ~ {treatment_var} + yield_ancova + main_road_min + C(survey_season_num)', cluster_var, 'did/controls/distance')
    add_result(result, 'did/controls/distance', 'methods/difference_in_differences.md#control-sets', outcome, 'Post-treatment', 'Season FE', 'Baseline + distance')

    result = run_regression(df_post, f'{outcome} ~ {treatment_var} + yield_ancova + hhr_n + main_road_min + C(survey_season_num)', cluster_var, 'did/controls/full')
    add_result(result, 'did/controls/full', 'methods/difference_in_differences.md#control-sets', outcome, 'Post-treatment', 'Season FE', 'Baseline + HH chars')

    # Leave-one-out
    result = run_regression(df_post, f'{outcome} ~ {treatment_var} + C(survey_season_num)', cluster_var, 'robust/loo/drop_ancova')
    add_result(result, 'robust/loo/drop_ancova', 'robustness/leave_one_out.md', outcome, 'Post-treatment', 'Season FE', 'No ANCOVA')

    result = run_regression(df_post, f'{outcome} ~ {treatment_var} + yield_ancova + main_road_min + C(survey_season_num)', cluster_var, 'robust/loo/drop_hhr_n')
    add_result(result, 'robust/loo/drop_hhr_n', 'robustness/leave_one_out.md', outcome, 'Post-treatment', 'Season FE', 'Dropped hhr_n')

    result = run_regression(df_post, f'{outcome} ~ {treatment_var} + yield_ancova + hhr_n + C(survey_season_num)', cluster_var, 'robust/loo/drop_distance')
    add_result(result, 'robust/loo/drop_distance', 'robustness/leave_one_out.md', outcome, 'Post-treatment', 'Season FE', 'Dropped distance')

    # ==========================================================================
    # 4. SAMPLE RESTRICTIONS (15 specs)
    # ==========================================================================
    print("\n4. Running sample restrictions...")

    base_formula = f'{outcome} ~ {treatment_var} + yield_ancova + C(survey_season_num)'

    # Drop each season
    for season in [4, 5, 6, 7]:
        df_sub = df_post[df_post['survey_season_num'] != season].copy()
        result = run_regression(df_sub, base_formula, cluster_var, f'robust/sample/drop_season_{season}')
        add_result(result, f'robust/sample/drop_season_{season}', 'robustness/sample_restrictions.md', outcome, f'Dropped season {season}', 'Season FE', 'Baseline yield')

    # Early vs late
    df_early = df_post[df_post['survey_season_num'] <= 5].copy()
    result = run_regression(df_early, base_formula, cluster_var, 'robust/sample/early_period')
    add_result(result, 'robust/sample/early_period', 'robustness/sample_restrictions.md', outcome, 'Early period (4-5)', 'Season FE', 'Baseline yield')

    df_late = df_post[df_post['survey_season_num'] >= 6].copy()
    result = run_regression(df_late, base_formula, cluster_var, 'robust/sample/late_period')
    add_result(result, 'robust/sample/late_period', 'robustness/sample_restrictions.md', outcome, 'Late period (6-7)', 'Season FE', 'Baseline yield')

    # Winsorize
    for pct in [1, 5, 10]:
        df_wins = df_post.copy()
        lower = df_wins[outcome].quantile(pct/100)
        upper = df_wins[outcome].quantile(1 - pct/100)
        df_wins[f'{outcome}_wins'] = df_wins[outcome].clip(lower=lower, upper=upper)
        result = run_regression(df_wins, f'{outcome}_wins ~ {treatment_var} + yield_ancova + C(survey_season_num)', cluster_var, f'robust/sample/winsor_{pct}pct')
        add_result(result, f'robust/sample/winsor_{pct}pct', 'robustness/sample_restrictions.md', f'{outcome}_wins', f'Winsorized {pct}%', 'Season FE', 'Baseline yield')

    # Trim
    df_trim = df_post.copy()
    lower = df_trim[outcome].quantile(0.01)
    upper = df_trim[outcome].quantile(0.99)
    df_trim = df_trim[(df_trim[outcome] >= lower) & (df_trim[outcome] <= upper)]
    result = run_regression(df_trim, base_formula, cluster_var, 'robust/sample/trim_1pct')
    add_result(result, 'robust/sample/trim_1pct', 'robustness/sample_restrictions.md', outcome, 'Trimmed 1%', 'Season FE', 'Baseline yield')

    # Balanced panel
    obs_per_hh = df_post.groupby('hhh_id').size()
    balanced_hhs = obs_per_hh[obs_per_hh == obs_per_hh.max()].index
    df_balanced = df_post[df_post['hhh_id'].isin(balanced_hhs)].copy()
    result = run_regression(df_balanced, base_formula, cluster_var, 'robust/sample/balanced')
    add_result(result, 'robust/sample/balanced', 'robustness/sample_restrictions.md', outcome, 'Balanced panel', 'Season FE', 'Baseline yield')

    # ==========================================================================
    # 5. ALTERNATIVE OUTCOMES (8 specs)
    # ==========================================================================
    print("\n5. Running alternative outcomes...")

    for alt_out in ['harvest_ton', 'price_ugx', 'harvest_value_ugx', 'share_sold', 'acreage']:
        if alt_out in df_post.columns and df_post[alt_out].notna().sum() > 100:
            result = run_regression(df_post, f'{alt_out} ~ {treatment_var} + C(survey_season_num)', cluster_var, f'robust/outcome/{alt_out}')
            add_result(result, f'robust/outcome/{alt_out}', 'robustness/measurement.md', alt_out, 'Post-treatment', 'Season FE', 'None')

    # Log transformations
    df_post['log_harvest_ton'] = np.log(df_post['harvest_ton'].clip(lower=0.001) + 1)
    result = run_regression(df_post, f'log_harvest_ton ~ {treatment_var} + C(survey_season_num)', cluster_var, 'robust/funcform/log_harvest')
    add_result(result, 'robust/funcform/log_harvest', 'robustness/functional_form.md', 'log_harvest_ton', 'Post-treatment', 'Season FE', 'Log transform')

    df_post['log_price'] = np.log(df_post['price_ugx'].clip(lower=1) + 1)
    result = run_regression(df_post, f'log_price ~ {treatment_var} + C(survey_season_num)', cluster_var, 'robust/funcform/log_price')
    add_result(result, 'robust/funcform/log_price', 'robustness/functional_form.md', 'log_price', 'Post-treatment', 'Season FE', 'Log transform')

    # IHS
    df_post['ihs_yield'] = np.arcsinh(df_post[outcome])
    result = run_regression(df_post, f'ihs_yield ~ {treatment_var} + C(survey_season_num)', cluster_var, 'robust/funcform/ihs_yield')
    add_result(result, 'robust/funcform/ihs_yield', 'robustness/functional_form.md', 'ihs_yield', 'Post-treatment', 'Season FE', 'IHS transform')

    # ==========================================================================
    # 6. INFERENCE VARIATIONS (5 specs)
    # ==========================================================================
    print("\n6. Running inference variations...")

    # No clustering
    result = run_regression(df_post, base_formula, cluster_var=None, spec_id='robust/cluster/none')
    add_result(result, 'robust/cluster/none', 'robustness/clustering_variations.md', outcome, 'Post-treatment', 'Season FE', 'Baseline yield', 'None (robust HC)')

    # Cluster by household
    result = run_regression(df_post, base_formula, cluster_var='hhh_id', spec_id='robust/cluster/unit')
    add_result(result, 'robust/cluster/unit', 'robustness/clustering_variations.md', outcome, 'Post-treatment', 'Season FE', 'Baseline yield', 'hhh_id (household)')

    # Two-way clustering
    try:
        model = pf.feols(base_formula, data=df_post, vcov={'CRV1': ['ea_code', 'survey_season_num']})
        coef = model.coef()[treatment_var]
        se = model.se()[treatment_var]
        pval = model.pvalue()[treatment_var]
        ci = model.confint()
        spec_count += 1
        results.append({
            'paper_id': paper_id, 'journal': journal, 'paper_title': paper_title,
            'spec_id': 'robust/cluster/twoway', 'spec_tree_path': 'robustness/clustering_variations.md',
            'outcome_var': outcome, 'treatment_var': treatment_var,
            'coefficient': float(coef), 'std_error': float(se),
            't_stat': float(model.tstat()[treatment_var]), 'p_value': float(pval),
            'ci_lower': float(ci.loc[treatment_var, '2.5%']),
            'ci_upper': float(ci.loc[treatment_var, '97.5%']),
            'n_obs': int(model._N), 'r_squared': float(model._r2) if hasattr(model, '_r2') else None,
            'sample_desc': 'Post-treatment', 'fixed_effects': 'Season FE',
            'controls_desc': 'Baseline yield', 'cluster_var': 'Two-way (village, season)',
            'model_type': 'OLS', 'estimation_script': 'specification_search.py',
            'coefficient_vector_json': json.dumps({'treatment': {'coef': float(coef), 'se': float(se), 'pval': float(pval)}})
        })
        print(f"  Spec {spec_count}: robust/cluster/twoway - coef={coef:.4f}, p={pval:.4f}")
    except Exception as e:
        print(f"  Error in two-way clustering: {e}")

    # ==========================================================================
    # 7. PLACEBO TESTS (2 specs)
    # ==========================================================================
    print("\n7. Running placebo tests...")

    # Pre-treatment placebo
    df_pre = df[df['season_ante'] == 1].copy()
    result = run_regression(df_pre, f'{outcome} ~ {treatment_var} + C(survey_season_num)', cluster_var, 'robust/placebo/pre_treatment')
    add_result(result, 'robust/placebo/pre_treatment', 'robustness/placebo_tests.md', outcome, 'Pre-treatment (placebo)', 'Season FE', 'None')

    # ==========================================================================
    # 8. HETEROGENEITY ANALYSIS (10 specs)
    # ==========================================================================
    print("\n8. Running heterogeneity analysis...")

    # HH size
    median_hhr = df_post['hhr_n'].median()
    df_post['large_hh'] = (df_post['hhr_n'] > median_hhr).astype(int)

    result = run_regression(df_post, f'{outcome} ~ {treatment_var}*large_hh + yield_ancova + C(survey_season_num)', cluster_var, 'robust/heterogeneity/hh_size')
    add_result(result, 'robust/heterogeneity/hh_size', 'robustness/heterogeneity.md', outcome, 'Post-treatment', 'Season FE', 'HH size interaction')

    df_small = df_post[df_post['large_hh'] == 0].copy()
    result = run_regression(df_small, base_formula, cluster_var, 'robust/sample/small_hh')
    add_result(result, 'robust/sample/small_hh', 'robustness/sample_restrictions.md', outcome, 'Small HH only', 'Season FE', 'Baseline yield')

    df_large = df_post[df_post['large_hh'] == 1].copy()
    result = run_regression(df_large, base_formula, cluster_var, 'robust/sample/large_hh')
    add_result(result, 'robust/sample/large_hh', 'robustness/sample_restrictions.md', outcome, 'Large HH only', 'Season FE', 'Baseline yield')

    # Distance
    median_dist = df_post['main_road_min'].median()
    df_post['far_from_road'] = (df_post['main_road_min'] > median_dist).astype(int)

    result = run_regression(df_post, f'{outcome} ~ {treatment_var}*far_from_road + yield_ancova + C(survey_season_num)', cluster_var, 'robust/heterogeneity/distance')
    add_result(result, 'robust/heterogeneity/distance', 'robustness/heterogeneity.md', outcome, 'Post-treatment', 'Season FE', 'Distance interaction')

    df_close = df_post[df_post['far_from_road'] == 0].copy()
    result = run_regression(df_close, base_formula, cluster_var, 'robust/sample/close_to_road')
    add_result(result, 'robust/sample/close_to_road', 'robustness/sample_restrictions.md', outcome, 'Close to road', 'Season FE', 'Baseline yield')

    df_far = df_post[df_post['far_from_road'] == 1].copy()
    result = run_regression(df_far, base_formula, cluster_var, 'robust/sample/far_from_road')
    add_result(result, 'robust/sample/far_from_road', 'robustness/sample_restrictions.md', outcome, 'Far from road', 'Season FE', 'Baseline yield')

    # Baseline yield heterogeneity
    median_yield = df_post['yield_ancova'].median()
    df_post['high_baseline'] = (df_post['yield_ancova'] > median_yield).astype(int)

    result = run_regression(df_post, f'{outcome} ~ {treatment_var}*high_baseline + C(survey_season_num)', cluster_var, 'robust/heterogeneity/baseline_yield')
    add_result(result, 'robust/heterogeneity/baseline_yield', 'robustness/heterogeneity.md', outcome, 'Post-treatment', 'Season FE', 'Baseline yield interaction')

    # ==========================================================================
    # 9. ESTIMATION METHOD VARIATIONS (3 specs)
    # ==========================================================================
    print("\n9. Running estimation method variations...")

    # First differences
    df_fd = df_post.sort_values(['hhh_id', 'survey_season_num']).copy()
    df_fd['yield_fd'] = df_fd.groupby('hhh_id')[outcome].diff()
    result = run_regression(df_fd.dropna(subset=['yield_fd']), f'yield_fd ~ {treatment_var}', cluster_var, 'panel/method/first_diff')
    add_result(result, 'panel/method/first_diff', 'methods/panel_fixed_effects.md#estimation-method', 'yield_fd', 'First differences', 'None', 'None', model_type='First Differences')

    # Pooled OLS
    result = run_regression(df_post, f'{outcome} ~ {treatment_var}', cluster_var, 'panel/method/pooled_ols')
    add_result(result, 'panel/method/pooled_ols', 'methods/panel_fixed_effects.md#estimation-method', outcome, 'Pooled OLS', 'None', 'None', model_type='Pooled OLS')

    # ==========================================================================
    # 10. JACKKNIFE - DROP VILLAGES (5 specs)
    # ==========================================================================
    print("\n10. Running jackknife (drop villages)...")

    villages = df_post['ea_code'].dropna().unique()[:5]
    for village in villages:
        df_drop = df_post[df_post['ea_code'] != village].copy()
        result = run_regression(df_drop, base_formula, cluster_var, f'robust/jackknife/drop_{village}')
        add_result(result, f'robust/jackknife/drop_{village}', 'robustness/sample_restrictions.md#influential-observations', outcome, f'Dropped {village}', 'Season FE', 'Baseline yield')

    # ==========================================================================
    # 11. ADDITIONAL SPECS
    # ==========================================================================
    print("\n11. Running additional specifications...")

    # Treatment x time
    result = run_regression(df_post, f'{outcome} ~ {treatment_var}*survey_season_num + yield_ancova', cluster_var, 'custom/treatment_x_time')
    add_result(result, 'custom/treatment_x_time', 'custom', outcome, 'Treatment effect over time', 'None', 'Baseline + time interaction')

    # Standardized
    df_post['yield_std'] = (df_post[outcome] - df_post[outcome].mean()) / df_post[outcome].std()
    result = run_regression(df_post, f'yield_std ~ {treatment_var} + C(survey_season_num)', cluster_var, 'robust/funcform/standardized')
    add_result(result, 'robust/funcform/standardized', 'robustness/functional_form.md', 'yield_std', 'Post-treatment', 'Season FE', 'Standardized outcome')

    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================

    print(f"\n{'='*60}")
    print(f"TOTAL SPECIFICATIONS RUN: {spec_count}")
    print(f"{'='*60}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{BASE_PATH}/specification_results.csv', index=False)
    print(f"\nResults saved to {BASE_PATH}/specification_results.csv")

    return results_df


if __name__ == '__main__':
    results = run_specification_search()

    if len(results) > 0:
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)

        print(f"\nTotal specifications: {len(results)}")
        print(f"Positive coefficients: {(results['coefficient'] > 0).sum()} ({100*(results['coefficient'] > 0).mean():.1f}%)")
        print(f"Significant at 5%: {(results['p_value'] < 0.05).sum()} ({100*(results['p_value'] < 0.05).mean():.1f}%)")
        print(f"Significant at 1%: {(results['p_value'] < 0.01).sum()} ({100*(results['p_value'] < 0.01).mean():.1f}%)")
        print(f"Median coefficient: {results['coefficient'].median():.4f}")
        print(f"Mean coefficient: {results['coefficient'].mean():.4f}")
        print(f"Range: [{results['coefficient'].min():.4f}, {results['coefficient'].max():.4f}]")
    else:
        print("\nNo specifications were successfully run.")
