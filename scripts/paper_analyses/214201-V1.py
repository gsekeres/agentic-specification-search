"""
Specification Search: 214201-V1
Khan, Khwaja, Olken - Mission vs. Financial Incentives for Community Health Workers
AER 2025

This paper studies how different types of incentives (mission-based vs. financial)
affect the performance of Lady Health Workers (LHWs) in Pakistan using an RCT design.

Main hypothesis: Mission-based incentives can be as effective as or more effective
than financial incentives in improving worker effort/performance.

Primary outcome: lhw_visit (household visited by health worker, binary 0/1)
Treatment variables:
- treat_mission_nobonus: Mission incentive (no bonus)
- treat_bonus_pr: Financial incentive (performance bonus)
- treat5: Mission + Financial incentive combined
- treat_social_all: Social recognition/placebo

Identification: Block-randomized controlled trial with block and wave fixed effects
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import warnings
warnings.filterwarnings('ignore')

# Set paths
BASE_PATH = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search'
DATA_PATH = f'{BASE_PATH}/data/downloads/extracted/214201-V1/replication_khan_mission/data'
OUTPUT_PATH = f'{BASE_PATH}/data/downloads/extracted/214201-V1'

# Paper metadata
PAPER_ID = '214201-V1'
JOURNAL = 'AER'
PAPER_TITLE = 'Mission vs. Financial Incentives for Community Health Workers'

# Method classification
METHOD_CODE = 'panel_fixed_effects'
METHOD_TREE_PATH = 'methods/panel_fixed_effects.md'

# Load data
print("Loading data...")
df = pd.read_stata(f'{DATA_PATH}/master_long.dta')

# Convert block to integer for fixed effects
df['block'] = df['block'].astype(int)
df['wave'] = df['wave'].astype(int)
df['lhw_id'] = df['lhw_id'].astype(int)

# Create pooled sample indicator (waves 2,3,4 - excluding baseline wave 0 and wave 5)
df['pooled_sample'] = ((df['data1'] != 1) & (df['data5'] != 1)).astype(int)

# Results storage
results = []

def run_regression(formula, data, weights=None, vcov_type='CRV1', cluster_var='lhw_id',
                   spec_id='', spec_tree_path='', outcome_var='', treatment_var='',
                   sample_desc='', fixed_effects='', controls_desc=''):
    """Run regression and store results"""
    try:
        if weights:
            model = pf.feols(formula, data=data, weights=weights, vcov={vcov_type: cluster_var})
        else:
            model = pf.feols(formula, data=data, vcov={vcov_type: cluster_var})

        # Get coefficient for treatment variable
        coefs = model.coef()
        ses = model.se()
        pvals = model.pvalue()

        # Build coefficient vector JSON
        coef_vector = {
            "treatment": {},
            "controls": [],
            "fixed_effects": [],
            "diagnostics": {}
        }

        for var in coefs.index:
            if treatment_var in var or var in ['treat_mission_nobonus', 'treat_bonus_pr', 'treat5', 'treat_social_all']:
                coef_vector["treatment"][var] = {
                    "coef": float(coefs[var]),
                    "se": float(ses[var]),
                    "pval": float(pvals[var])
                }
            else:
                coef_vector["controls"].append({
                    "var": var,
                    "coef": float(coefs[var]),
                    "se": float(ses[var]),
                    "pval": float(pvals[var])
                })

        # Get primary treatment coefficient (treat_mission_nobonus is main treatment of interest)
        if treatment_var in coefs.index:
            main_coef = float(coefs[treatment_var])
            main_se = float(ses[treatment_var])
            main_pval = float(pvals[treatment_var])
            main_tstat = main_coef / main_se if main_se > 0 else np.nan
        else:
            # Fallback to first treatment variable
            for t in ['treat_mission_nobonus', 'treat_bonus_pr', 'treat5', 'treat_social_all']:
                if t in coefs.index:
                    main_coef = float(coefs[t])
                    main_se = float(ses[t])
                    main_pval = float(pvals[t])
                    main_tstat = main_coef / main_se if main_se > 0 else np.nan
                    treatment_var = t
                    break

        # Confidence intervals
        ci_lower = main_coef - 1.96 * main_se
        ci_upper = main_coef + 1.96 * main_se

        result = {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': main_coef,
            'std_error': main_se,
            't_stat': main_tstat,
            'p_value': main_pval,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_obs': int(model._N),
            'r_squared': float(model._r2) if hasattr(model, '_r2') else np.nan,
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects,
            'controls_desc': controls_desc,
            'cluster_var': cluster_var,
            'model_type': 'Panel FE (reghdfe)',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
        results.append(result)
        return result
    except Exception as e:
        print(f"Error in {spec_id}: {str(e)}")
        return None

print("\n=== RUNNING SPECIFICATION SEARCH ===\n")

# ============================================
# BASELINE REPLICATION (Table 1, Column 1)
# ============================================
print("1. Running baseline replication...")

# Pooled sample (waves 2,3,4)
df_pooled = df[df['pooled_sample'] == 1].copy()

# Baseline: Main Table 1 specification WITH weights
run_regression(
    formula="lhw_visit ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all | block + wave",
    data=df_pooled,
    weights='pw',
    spec_id='baseline',
    spec_tree_path=f'{METHOD_TREE_PATH}#baseline',
    outcome_var='lhw_visit',
    treatment_var='treat_mission_nobonus',
    sample_desc='Pooled waves 2,3,4 (N=21299)',
    fixed_effects='block + wave',
    controls_desc='None (treatment dummies only)'
)

# ============================================
# FIXED EFFECTS VARIATIONS
# ============================================
print("2. Running fixed effects variations...")

# FE: No fixed effects
run_regression(
    formula="lhw_visit ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all",
    data=df_pooled,
    weights='pw',
    spec_id='panel/fe/none',
    spec_tree_path=f'{METHOD_TREE_PATH}#fixed-effects-structure',
    outcome_var='lhw_visit',
    treatment_var='treat_mission_nobonus',
    sample_desc='Pooled waves 2,3,4',
    fixed_effects='None (pooled OLS)',
    controls_desc='None'
)

# FE: Block only
run_regression(
    formula="lhw_visit ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all | block",
    data=df_pooled,
    weights='pw',
    spec_id='panel/fe/block_only',
    spec_tree_path=f'{METHOD_TREE_PATH}#fixed-effects-structure',
    outcome_var='lhw_visit',
    treatment_var='treat_mission_nobonus',
    sample_desc='Pooled waves 2,3,4',
    fixed_effects='block only',
    controls_desc='None'
)

# FE: Wave only
run_regression(
    formula="lhw_visit ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all | wave",
    data=df_pooled,
    weights='pw',
    spec_id='panel/fe/wave_only',
    spec_tree_path=f'{METHOD_TREE_PATH}#fixed-effects-structure',
    outcome_var='lhw_visit',
    treatment_var='treat_mission_nobonus',
    sample_desc='Pooled waves 2,3,4',
    fixed_effects='wave only',
    controls_desc='None'
)

# ============================================
# SAMPLE RESTRICTIONS
# ============================================
print("3. Running sample restrictions...")

# By wave
for wave_num in [2, 3, 4]:
    df_wave = df[df[f'data{wave_num}'] == 1].copy()
    run_regression(
        formula="lhw_visit ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all | block + wave",
        data=df_wave,
        weights='pw',
        spec_id=f'robust/sample/wave_{wave_num}',
        spec_tree_path='robustness/sample_restrictions.md#time-period-restrictions',
        outcome_var='lhw_visit',
        treatment_var='treat_mission_nobonus',
        sample_desc=f'Wave {wave_num} only',
        fixed_effects='block + wave',
        controls_desc='None'
    )

# Baseline performance splits
df_unique = df_pooled.drop_duplicates('lhw_id')
baseline_median = df_unique['baseline_perf'].median()

# High baseline performers
df_high_base = df_pooled[df_pooled['baseline_perf'] >= baseline_median].copy()
run_regression(
    formula="lhw_visit ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all | block + wave",
    data=df_high_base,
    weights='pw',
    spec_id='robust/sample/high_baseline_perf',
    spec_tree_path='robustness/sample_restrictions.md#demographic-restrictions',
    outcome_var='lhw_visit',
    treatment_var='treat_mission_nobonus',
    sample_desc='High baseline performers only',
    fixed_effects='block + wave',
    controls_desc='None'
)

# Low baseline performers
df_low_base = df_pooled[df_pooled['baseline_perf'] < baseline_median].copy()
run_regression(
    formula="lhw_visit ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all | block + wave",
    data=df_low_base,
    weights='pw',
    spec_id='robust/sample/low_baseline_perf',
    spec_tree_path='robustness/sample_restrictions.md#demographic-restrictions',
    outcome_var='lhw_visit',
    treatment_var='treat_mission_nobonus',
    sample_desc='Low baseline performers only',
    fixed_effects='block + wave',
    controls_desc='None'
)

# Create sample trimming variables (from tabA6.do)
df_unique_trim = df.drop_duplicates('lhw_id').copy()
tot_hh_95 = df_unique_trim['tot_hh'].quantile(0.95)
tot_hh_05 = df_unique_trim['tot_hh'].quantile(0.05)

df['top_hh'] = (df['tot_hh'] > tot_hh_95).astype(int)
df['bot_hh'] = (df['tot_hh'] < tot_hh_05).astype(int)

# Block size calculations
block_sizes = df.drop_duplicates('lhw_id').groupby('block').size()
df['block_size'] = df['block'].map(block_sizes)
block_95 = df['block_size'].quantile(0.95)
block_05 = df['block_size'].quantile(0.05)
df['big_block'] = (df['block_size'] > block_95).astype(int)
df['small_block'] = (df['block_size'] < block_05).astype(int)

# Update pooled sample
df_pooled = df[df['pooled_sample'] == 1].copy()

# Drop top 5% households
df_no_top = df_pooled[df_pooled['top_hh'] == 0].copy()
run_regression(
    formula="lhw_visit ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all | block + wave",
    data=df_no_top,
    weights='pw',
    spec_id='robust/sample/drop_top5pct_hh',
    spec_tree_path='robustness/sample_restrictions.md#outlier-treatment',
    outcome_var='lhw_visit',
    treatment_var='treat_mission_nobonus',
    sample_desc='Excluding top 5% HH size communities',
    fixed_effects='block + wave',
    controls_desc='None'
)

# Drop bottom 5% households
df_no_bot = df_pooled[df_pooled['bot_hh'] == 0].copy()
run_regression(
    formula="lhw_visit ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all | block + wave",
    data=df_no_bot,
    weights='pw',
    spec_id='robust/sample/drop_bot5pct_hh',
    spec_tree_path='robustness/sample_restrictions.md#outlier-treatment',
    outcome_var='lhw_visit',
    treatment_var='treat_mission_nobonus',
    sample_desc='Excluding bottom 5% HH size communities',
    fixed_effects='block + wave',
    controls_desc='None'
)

# Drop big blocks
df_no_big = df_pooled[df_pooled['big_block'] == 0].copy()
run_regression(
    formula="lhw_visit ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all | block + wave",
    data=df_no_big,
    weights='pw',
    spec_id='robust/sample/drop_big_blocks',
    spec_tree_path='robustness/sample_restrictions.md#outlier-treatment',
    outcome_var='lhw_visit',
    treatment_var='treat_mission_nobonus',
    sample_desc='Excluding top 5% block sizes',
    fixed_effects='block + wave',
    controls_desc='None'
)

# Drop small blocks
df_no_small = df_pooled[df_pooled['small_block'] == 0].copy()
run_regression(
    formula="lhw_visit ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all | block + wave",
    data=df_no_small,
    weights='pw',
    spec_id='robust/sample/drop_small_blocks',
    spec_tree_path='robustness/sample_restrictions.md#outlier-treatment',
    outcome_var='lhw_visit',
    treatment_var='treat_mission_nobonus',
    sample_desc='Excluding bottom 5% block sizes',
    fixed_effects='block + wave',
    controls_desc='None'
)

# ============================================
# CLUSTERING VARIATIONS
# ============================================
print("4. Running clustering variations...")

# Robust heteroskedasticity SEs (no clustering) - pyfixest needs cluster for CRV
# Using iid vcov instead
try:
    model = pf.feols("lhw_visit ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all | block + wave",
                     data=df_pooled, weights='pw', vcov='iid')
    coefs = model.coef()
    ses = model.se()
    pvals = model.pvalue()
    main_coef = float(coefs['treat_mission_nobonus'])
    main_se = float(ses['treat_mission_nobonus'])
    main_pval = float(pvals['treat_mission_nobonus'])
    main_tstat = main_coef / main_se if main_se > 0 else np.nan
    ci_lower = main_coef - 1.96 * main_se
    ci_upper = main_coef + 1.96 * main_se

    coef_vector = {"treatment": {}, "controls": [], "fixed_effects": [], "diagnostics": {}}
    for var in coefs.index:
        coef_vector["treatment"][var] = {"coef": float(coefs[var]), "se": float(ses[var]), "pval": float(pvals[var])}

    results.append({
        'paper_id': PAPER_ID, 'journal': JOURNAL, 'paper_title': PAPER_TITLE,
        'spec_id': 'robust/cluster/iid_se',
        'spec_tree_path': 'robustness/clustering_variations.md#robust-se',
        'outcome_var': 'lhw_visit', 'treatment_var': 'treat_mission_nobonus',
        'coefficient': main_coef, 'std_error': main_se, 't_stat': main_tstat,
        'p_value': main_pval, 'ci_lower': ci_lower, 'ci_upper': ci_upper,
        'n_obs': int(model._N), 'r_squared': float(model._r2),
        'coefficient_vector_json': json.dumps(coef_vector),
        'sample_desc': 'Pooled waves 2,3,4 (iid SE)',
        'fixed_effects': 'block + wave', 'controls_desc': 'None',
        'cluster_var': 'None (iid)', 'model_type': 'Panel FE (reghdfe)',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })
except Exception as e:
    print(f"Error in robust/cluster/iid_se: {e}")

# Cluster by block
run_regression(
    formula="lhw_visit ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all | block + wave",
    data=df_pooled,
    weights='pw',
    vcov_type='CRV1',
    cluster_var='block',
    spec_id='robust/cluster/block',
    spec_tree_path='robustness/clustering_variations.md#single-level-clustering',
    outcome_var='lhw_visit',
    treatment_var='treat_mission_nobonus',
    sample_desc='Pooled waves 2,3,4',
    fixed_effects='block + wave',
    controls_desc='None'
)

# ============================================
# WEIGHTS VARIATIONS
# ============================================
print("5. Running weights variations...")

# Unweighted
run_regression(
    formula="lhw_visit ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all | block + wave",
    data=df_pooled,
    weights=None,
    spec_id='robust/weights/unweighted',
    spec_tree_path='robustness/measurement.md#weights',
    outcome_var='lhw_visit',
    treatment_var='treat_mission_nobonus',
    sample_desc='Pooled waves 2,3,4 (unweighted)',
    fixed_effects='block + wave',
    controls_desc='None'
)

# ============================================
# ALTERNATIVE OUTCOME VARIABLES
# ============================================
print("6. Running alternative outcomes...")

# were_preg_served - pregnant women served
df_preg = df_pooled[df_pooled['were_preg_served'].notna()].copy()
if len(df_preg) > 100:
    run_regression(
        formula="were_preg_served ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all | block + wave",
        data=df_preg,
        weights=None,
        spec_id='robust/outcome/pregnant_served',
        spec_tree_path='robustness/measurement.md#alternative-outcomes',
        outcome_var='were_preg_served',
        treatment_var='treat_mission_nobonus',
        sample_desc='HH with pregnant women',
        fixed_effects='block + wave',
        controls_desc='None'
    )

# were_child_served - children served
df_child = df_pooled[df_pooled['were_child_served'].notna()].copy()
if len(df_child) > 100:
    run_regression(
        formula="were_child_served ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all | block + wave",
        data=df_child,
        weights=None,
        spec_id='robust/outcome/child_served',
        spec_tree_path='robustness/measurement.md#alternative-outcomes',
        outcome_var='were_child_served',
        treatment_var='treat_mission_nobonus',
        sample_desc='HH with children',
        fixed_effects='block + wave',
        controls_desc='None'
    )

# tb_check - TB check performed
df_tb = df_pooled[df_pooled['tb_check'].notna()].copy()
if len(df_tb) > 100:
    run_regression(
        formula="tb_check ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all | block + wave",
        data=df_tb,
        weights=None,
        spec_id='robust/outcome/tb_check',
        spec_tree_path='robustness/measurement.md#alternative-outcomes',
        outcome_var='tb_check',
        treatment_var='treat_mission_nobonus',
        sample_desc='Pooled sample',
        fixed_effects='block + wave',
        controls_desc='None'
    )

# Health outcomes from Table 3 (wave 5 only, with children)
df_wave5 = df[(df['data5'] == 1) & (df['child_exist'] == 1)].copy()
df_wave5_clean = df_wave5[df_wave5['diarrhea_incidence'].notna()].copy()
if len(df_wave5_clean) > 100:
    run_regression(
        formula="diarrhea_incidence ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all | block",
        data=df_wave5_clean,
        weights=None,
        spec_id='robust/outcome/diarrhea_incidence',
        spec_tree_path='robustness/measurement.md#alternative-outcomes',
        outcome_var='diarrhea_incidence',
        treatment_var='treat_mission_nobonus',
        sample_desc='Wave 5, HH with children',
        fixed_effects='block',
        controls_desc='None'
    )

# Vaccination schedule
df_vac = df_wave5[df_wave5['vac_sch'].notna()].copy()
if len(df_vac) > 100:
    run_regression(
        formula="vac_sch ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all | block",
        data=df_vac,
        weights=None,
        spec_id='robust/outcome/vac_schedule',
        spec_tree_path='robustness/measurement.md#alternative-outcomes',
        outcome_var='vac_sch',
        treatment_var='treat_mission_nobonus',
        sample_desc='Wave 5, HH with children',
        fixed_effects='block',
        controls_desc='None'
    )

# ============================================
# ALTERNATIVE TREATMENT DEFINITIONS
# ============================================
print("7. Running alternative treatment definitions...")

# Focus on financial incentive coefficient
run_regression(
    formula="lhw_visit ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all | block + wave",
    data=df_pooled,
    weights='pw',
    spec_id='robust/treatment/financial_incentive',
    spec_tree_path='robustness/measurement.md#alternative-treatments',
    outcome_var='lhw_visit',
    treatment_var='treat_bonus_pr',
    sample_desc='Pooled waves 2,3,4',
    fixed_effects='block + wave',
    controls_desc='None'
)

# Focus on mission+bonus combined (treat5) coefficient
run_regression(
    formula="lhw_visit ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all | block + wave",
    data=df_pooled,
    weights='pw',
    spec_id='robust/treatment/mission_plus_bonus',
    spec_tree_path='robustness/measurement.md#alternative-treatments',
    outcome_var='lhw_visit',
    treatment_var='treat5',
    sample_desc='Pooled waves 2,3,4',
    fixed_effects='block + wave',
    controls_desc='None'
)

# Focus on placebo (social recognition) coefficient
run_regression(
    formula="lhw_visit ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all | block + wave",
    data=df_pooled,
    weights='pw',
    spec_id='robust/treatment/placebo_social',
    spec_tree_path='robustness/measurement.md#alternative-treatments',
    outcome_var='lhw_visit',
    treatment_var='treat_social_all',
    sample_desc='Pooled waves 2,3,4',
    fixed_effects='block + wave',
    controls_desc='None'
)

# Mission (all versions combined)
run_regression(
    formula="lhw_visit ~ treat_mission_all + treat_bonus_pr + treat_social_all | block + wave",
    data=df_pooled,
    weights='pw',
    spec_id='robust/treatment/mission_all',
    spec_tree_path='robustness/measurement.md#alternative-treatments',
    outcome_var='lhw_visit',
    treatment_var='treat_mission_all',
    sample_desc='Pooled waves 2,3,4',
    fixed_effects='block + wave',
    controls_desc='None'
)

# ============================================
# HETEROGENEITY ANALYSES (from tabA7.do)
# ============================================
print("8. Running heterogeneity analyses...")

# Health diploma interaction
df_het = df_pooled.copy()
df_het['health_diploma_std'] = df_het['health_diploma']
df_het['inter_mission_diploma'] = df_het['treat_mission_nobonus'] * df_het['health_diploma_std']
df_het_clean = df_het[df_het['health_diploma_std'].notna()].copy()

if len(df_het_clean) > 100:
    run_regression(
        formula="lhw_visit ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all + health_diploma_std + inter_mission_diploma | block + wave",
        data=df_het_clean,
        weights='pw',
        spec_id='robust/heterogeneity/health_diploma',
        spec_tree_path='robustness/heterogeneity.md#education',
        outcome_var='lhw_visit',
        treatment_var='treat_mission_nobonus',
        sample_desc='Pooled, with health diploma interaction',
        fixed_effects='block + wave',
        controls_desc='health_diploma interaction'
    )

# Years of schooling interaction
df_het = df_pooled.copy()
df_het['years_school_std'] = (df_het['years_school'] - df_het['years_school'].mean()) / df_het['years_school'].std()
df_het['inter_mission_school'] = df_het['treat_mission_nobonus'] * df_het['years_school_std']
df_het_clean = df_het[df_het['years_school_std'].notna()].copy()

if len(df_het_clean) > 100:
    run_regression(
        formula="lhw_visit ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all + years_school_std + inter_mission_school | block + wave",
        data=df_het_clean,
        weights='pw',
        spec_id='robust/heterogeneity/years_school',
        spec_tree_path='robustness/heterogeneity.md#education',
        outcome_var='lhw_visit',
        treatment_var='treat_mission_nobonus',
        sample_desc='Pooled, with years school interaction',
        fixed_effects='block + wave',
        controls_desc='years_school interaction'
    )

# Tenure interaction
df_het = df_pooled.copy()
df_het['tenure_std'] = (df_het['tenure'] - df_het['tenure'].mean()) / df_het['tenure'].std()
df_het['inter_mission_tenure'] = df_het['treat_mission_nobonus'] * df_het['tenure_std']
df_het_clean = df_het[df_het['tenure_std'].notna()].copy()

if len(df_het_clean) > 100:
    run_regression(
        formula="lhw_visit ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all + tenure_std + inter_mission_tenure | block + wave",
        data=df_het_clean,
        weights='pw',
        spec_id='robust/heterogeneity/tenure',
        spec_tree_path='robustness/heterogeneity.md#experience',
        outcome_var='lhw_visit',
        treatment_var='treat_mission_nobonus',
        sample_desc='Pooled, with tenure interaction',
        fixed_effects='block + wave',
        controls_desc='tenure interaction'
    )

# PSM (public service motivation) interaction
df_het = df_pooled.copy()
df_het['psm_std'] = (df_het['psm'] - df_het['psm'].mean()) / df_het['psm'].std()
df_het['inter_mission_psm'] = df_het['treat_mission_nobonus'] * df_het['psm_std']
df_het_clean = df_het[df_het['psm_std'].notna()].copy()

if len(df_het_clean) > 100:
    run_regression(
        formula="lhw_visit ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all + psm_std + inter_mission_psm | block + wave",
        data=df_het_clean,
        weights='pw',
        spec_id='robust/heterogeneity/psm',
        spec_tree_path='robustness/heterogeneity.md#motivation',
        outcome_var='lhw_visit',
        treatment_var='treat_mission_nobonus',
        sample_desc='Pooled, with PSM interaction',
        fixed_effects='block + wave',
        controls_desc='psm interaction'
    )

# IQ score interaction
df_het = df_pooled.copy()
df_het['iq_std'] = (df_het['iq_score'] - df_het['iq_score'].mean()) / df_het['iq_score'].std()
df_het['inter_mission_iq'] = df_het['treat_mission_nobonus'] * df_het['iq_std']
df_het_clean = df_het[df_het['iq_std'].notna()].copy()

if len(df_het_clean) > 100:
    run_regression(
        formula="lhw_visit ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all + iq_std + inter_mission_iq | block + wave",
        data=df_het_clean,
        weights='pw',
        spec_id='robust/heterogeneity/iq_score',
        spec_tree_path='robustness/heterogeneity.md#ability',
        outcome_var='lhw_visit',
        treatment_var='treat_mission_nobonus',
        sample_desc='Pooled, with IQ score interaction',
        fixed_effects='block + wave',
        controls_desc='iq_score interaction'
    )

# Baseline performance interaction
df_het = df_pooled.copy()
df_het['baseline_std'] = (df_het['baseline_perf'] - df_het['baseline_perf'].mean()) / df_het['baseline_perf'].std()
df_het['inter_mission_baseline'] = df_het['treat_mission_nobonus'] * df_het['baseline_std']
run_regression(
    formula="lhw_visit ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all + baseline_std + inter_mission_baseline | block + wave",
    data=df_het,
    weights='pw',
    spec_id='robust/heterogeneity/baseline_perf',
    spec_tree_path='robustness/heterogeneity.md#baseline-performance',
    outcome_var='lhw_visit',
    treatment_var='treat_mission_nobonus',
    sample_desc='Pooled, with baseline perf interaction',
    fixed_effects='block + wave',
    controls_desc='baseline_perf interaction'
)

# ============================================
# CONTROL VARIABLE ADDITIONS
# ============================================
print("9. Running control variations...")

# Add baseline performance as control
run_regression(
    formula="lhw_visit ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all + baseline_perf | block + wave",
    data=df_pooled,
    weights='pw',
    spec_id='robust/control/add_baseline_perf',
    spec_tree_path='robustness/control_progression.md#add-controls',
    outcome_var='lhw_visit',
    treatment_var='treat_mission_nobonus',
    sample_desc='Pooled waves 2,3,4',
    fixed_effects='block + wave',
    controls_desc='baseline_perf'
)

# Add IQ score
df_iq = df_pooled[df_pooled['iq_score'].notna()].copy()
run_regression(
    formula="lhw_visit ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all + iq_score | block + wave",
    data=df_iq,
    weights='pw',
    spec_id='robust/control/add_iq',
    spec_tree_path='robustness/control_progression.md#add-controls',
    outcome_var='lhw_visit',
    treatment_var='treat_mission_nobonus',
    sample_desc='Pooled, IQ available',
    fixed_effects='block + wave',
    controls_desc='iq_score'
)

# Add PSM
df_psm = df_pooled[df_pooled['psm'].notna()].copy()
run_regression(
    formula="lhw_visit ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all + psm | block + wave",
    data=df_psm,
    weights='pw',
    spec_id='robust/control/add_psm',
    spec_tree_path='robustness/control_progression.md#add-controls',
    outcome_var='lhw_visit',
    treatment_var='treat_mission_nobonus',
    sample_desc='Pooled, PSM available',
    fixed_effects='block + wave',
    controls_desc='psm'
)

# Full controls
df_full = df_pooled.dropna(subset=['iq_score', 'psm', 'years_school']).copy()
if len(df_full) > 500:
    run_regression(
        formula="lhw_visit ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all + baseline_perf + iq_score + psm + years_school | block + wave",
        data=df_full,
        weights='pw',
        spec_id='robust/control/full_controls',
        spec_tree_path='robustness/control_progression.md#full-controls',
        outcome_var='lhw_visit',
        treatment_var='treat_mission_nobonus',
        sample_desc='Pooled, all controls available',
        fixed_effects='block + wave',
        controls_desc='baseline_perf + iq_score + psm + years_school'
    )

# ============================================
# PLACEBO TESTS
# ============================================
print("10. Running placebo tests...")

# Placebo: Baseline period (wave 0) - should show no effect
df_baseline = df[df['data1'] == 1].copy()
run_regression(
    formula="lhw_visit ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all | block",
    data=df_baseline,
    weights='pw',
    spec_id='robust/placebo/baseline_period',
    spec_tree_path='robustness/placebo_tests.md#pre-treatment',
    outcome_var='lhw_visit',
    treatment_var='treat_mission_nobonus',
    sample_desc='Baseline period (wave 0, pre-treatment)',
    fixed_effects='block',
    controls_desc='None'
)

# ============================================
# FUNCTIONAL FORM VARIATIONS
# ============================================
print("11. Running functional form variations...")

# Create transformed outcomes
df_pooled['lhw_visit_2x'] = df_pooled['lhw_visit'] * 2  # Linear scaling

# Linear probability model is baseline, nothing changes structurally but we can look at different aggregation

# Winsorize outcome at different levels (but since binary, this is less relevant)
# Instead, let's try log transformation of continuous controls

# ============================================
# ADDITIONAL ROBUSTNESS - DROP SINGLE BLOCKS
# ============================================
print("12. Running leave-one-out block analysis...")

unique_blocks = df_pooled['block'].unique()[:10]  # First 10 blocks to keep manageable
for block_id in unique_blocks:
    df_drop = df_pooled[df_pooled['block'] != block_id].copy()
    run_regression(
        formula="lhw_visit ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all | block + wave",
        data=df_drop,
        weights='pw',
        spec_id=f'robust/loo/drop_block_{int(block_id)}',
        spec_tree_path='robustness/leave_one_out.md#geographic-units',
        outcome_var='lhw_visit',
        treatment_var='treat_mission_nobonus',
        sample_desc=f'Excluding block {int(block_id)}',
        fixed_effects='block + wave',
        controls_desc='None'
    )

# ============================================
# ALTERNATIVE SPECIFICATIONS FROM TABLE A4 (No weights)
# ============================================
print("13. Running Table A4 variations (unweighted)...")

run_regression(
    formula="lhw_visit ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all | block + wave",
    data=df_pooled,
    weights=None,
    spec_id='robust/method/tabA4_unweighted',
    spec_tree_path='robustness/model_specification.md#estimation-alternatives',
    outcome_var='lhw_visit',
    treatment_var='treat_mission_nobonus',
    sample_desc='Pooled waves 2,3,4 (unweighted, Table A4 style)',
    fixed_effects='block + wave',
    controls_desc='None'
)

# ============================================
# PEER EFFECTS SPECIFICATION (Table 5)
# ============================================
print("14. Running peer effects specifications...")

# Public vs Private mission treatment
run_regression(
    formula="lhw_visit ~ treat_mission_pri + treat_mission_nobonuspub + treat_bonus_pr + treat5 + treat_social_all | block + wave",
    data=df_pooled,
    weights='pw',
    spec_id='robust/treatment/public_vs_private',
    spec_tree_path='robustness/measurement.md#alternative-treatments',
    outcome_var='lhw_visit',
    treatment_var='treat_mission_nobonuspub',
    sample_desc='Pooled waves 2,3,4',
    fixed_effects='block + wave',
    controls_desc='Public vs private treatment distinction'
)

# ============================================
# ADDITIONAL SAMPLE SPLITS
# ============================================
print("15. Running additional sample splits...")

# By community size terciles
df_pooled['tot_hh_tercile'] = pd.qcut(df_pooled['tot_hh'], 3, labels=['small', 'medium', 'large'])

for tercile in ['small', 'medium', 'large']:
    df_tercile = df_pooled[df_pooled['tot_hh_tercile'] == tercile].copy()
    run_regression(
        formula="lhw_visit ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all | block + wave",
        data=df_tercile,
        weights='pw',
        spec_id=f'robust/sample/hh_tercile_{tercile}',
        spec_tree_path='robustness/sample_restrictions.md#demographic-restrictions',
        outcome_var='lhw_visit',
        treatment_var='treat_mission_nobonus',
        sample_desc=f'Community size tercile: {tercile}',
        fixed_effects='block + wave',
        controls_desc='None'
    )

# ============================================
# OUTCOME: CONDITIONAL ON VISIT
# ============================================
print("16. Running conditional outcome specifications...")

# Services conditional on visit (from Table 2)
df_visited = df_pooled[df_pooled['lhw_visit'] == 1].copy()

# Pregnant served conditional on visit
df_preg_cond = df_visited[(df_visited['preg_exist'] == 1) & (df_visited['were_preg_served'].notna())].copy()
if len(df_preg_cond) > 100:
    run_regression(
        formula="were_preg_served ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all | block + wave",
        data=df_preg_cond,
        weights=None,
        spec_id='robust/outcome/pregnant_served_cond',
        spec_tree_path='robustness/measurement.md#conditional-outcomes',
        outcome_var='were_preg_served',
        treatment_var='treat_mission_nobonus',
        sample_desc='Conditional on visit, pregnant exists',
        fixed_effects='block + wave',
        controls_desc='None'
    )

# Child served conditional on visit
df_child_cond = df_visited[(df_visited['child_exist'] == 1) & (df_visited['were_child_served'].notna())].copy()
if len(df_child_cond) > 100:
    run_regression(
        formula="were_child_served ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all | block + wave",
        data=df_child_cond,
        weights=None,
        spec_id='robust/outcome/child_served_cond',
        spec_tree_path='robustness/measurement.md#conditional-outcomes',
        outcome_var='were_child_served',
        treatment_var='treat_mission_nobonus',
        sample_desc='Conditional on visit, child exists',
        fixed_effects='block + wave',
        controls_desc='None'
    )

# TB check conditional on visit
df_tb_cond = df_visited[df_visited['tb_check'].notna()].copy()
if len(df_tb_cond) > 100:
    run_regression(
        formula="tb_check ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all | block + wave",
        data=df_tb_cond,
        weights=None,
        spec_id='robust/outcome/tb_check_cond',
        spec_tree_path='robustness/measurement.md#conditional-outcomes',
        outcome_var='tb_check',
        treatment_var='treat_mission_nobonus',
        sample_desc='Conditional on visit',
        fixed_effects='block + wave',
        controls_desc='None'
    )

# ============================================
# INFERENCE ALTERNATIVES
# ============================================
print("17. Running inference alternatives...")

# CRV3 (small sample correction)
try:
    run_regression(
        formula="lhw_visit ~ treat_mission_nobonus + treat_bonus_pr + treat5 + treat_social_all | block + wave",
        data=df_pooled,
        weights='pw',
        vcov_type='CRV3',
        cluster_var='lhw_id',
        spec_id='robust/inference/crv3',
        spec_tree_path='robustness/inference_alternatives.md#small-sample-correction',
        outcome_var='lhw_visit',
        treatment_var='treat_mission_nobonus',
        sample_desc='Pooled waves 2,3,4',
        fixed_effects='block + wave',
        controls_desc='None'
    )
except:
    pass

# ============================================
# FINAL SAVE
# ============================================
print("\n=== SAVING RESULTS ===\n")

# Create results DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
results_df.to_csv(f'{OUTPUT_PATH}/specification_results.csv', index=False)
print(f"Total specifications: {len(results_df)}")
print(f"Results saved to: {OUTPUT_PATH}/specification_results.csv")

# Summary statistics
print("\n=== SUMMARY STATISTICS ===\n")
print(f"Total specifications run: {len(results_df)}")
print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({(results_df['coefficient'] > 0).mean()*100:.1f}%)")
print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({(results_df['p_value'] < 0.05).mean()*100:.1f}%)")
print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({(results_df['p_value'] < 0.01).mean()*100:.1f}%)")
print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")
print(f"Coefficient range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")
