"""
Specification Search: 195301-V1
Paper: Toews & Vezina - "Enemies of the People" (AEJ: Applied Economics)

Topic: The long-run effects of political persecution in Soviet Gulag camps
       on modern Russian firm-level outcomes (wages, profits, value added)

Hypothesis: Higher share of "enemies" (political prisoners) in Gulag camps
            leads to HIGHER wages/productivity in modern firms nearby
            (human capital channel)

Method: Cross-sectional OLS with regional fixed effects
Treatment: share_enemies_1952_100 (share of enemies in camps within 100km)
Outcomes: lnwage, lnnetprofitpc_rawemp, lnvalueaddedpc_rawemp

Data: SPARK (Interfax) firm-level data merged with Gulag camp data
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/195301-V1/toews_vezina_replication/CodeandData/data"
OUTPUT_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/195301-V1"

# Paper metadata
PAPER_ID = "195301-V1"
JOURNAL = "AEJ-Applied"
PAPER_TITLE = "Enemies of the People"

# Load the main analysis dataset
print("Loading data...")
df = pd.read_stata(f"{DATA_DIR}/spark_enemies_temp.dta")
firms_region = pd.read_stata(f"{DATA_DIR}/firms_region_all.dta")

print(f"Initial observations: {len(df)}")

# Merge region info
df = df.merge(firms_region, on='sparkid', how='left')

# Clean data - remove missing sparkid
df = df[df['sparkid'].notna()]

# Create key variables as in original Stata code
# Treatment variable
if 'share_enemies_1952_100' not in df.columns:
    df['share_enemies_1952_100'] = df['enemies100Gulag1952'] / df['people_100_1952']

# Log of prisoners within 100km
if 'ln100people1952' not in df.columns and 'people_100_1952' in df.columns:
    df['ln100people1952'] = np.log(df['people_100_1952'])

# Outcome: lnwage
if 'lnwage' not in df.columns and 'medicinsurance2018' in df.columns and 'employees2018' in df.columns:
    df['lnwage'] = np.log(df['medicinsurance2018'] / df['employees2018'])

# Distance to Moscow
if 'dist_to_Moscow' not in df.columns:
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    moscow_lat = 55.7558
    moscow_lon = 37.6173
    df['dist_to_Moscow'] = np.sqrt((df['latitude'] - moscow_lat)**2 +
                                    (df['longitude'] - moscow_lon)**2) * 111

print(f"Observations after cleaning: {len(df)}")

# Results storage
results = []

def extract_results(model, spec_id, spec_tree_path, outcome_var, treatment_var,
                   sample_desc, fixed_effects, controls_desc, cluster_var, model_type):
    """Extract results from pyfixest model"""
    try:
        coef = model.coef()[treatment_var]
        se = model.se()[treatment_var]
        t_stat = model.tstat()[treatment_var]
        pval = model.pvalue()[treatment_var]
        ci = model.confint()
        ci_lower = ci.loc[treatment_var, '2.5%'] if '2.5%' in ci.columns else ci.loc[treatment_var].iloc[0]
        ci_upper = ci.loc[treatment_var, '97.5%'] if '97.5%' in ci.columns else ci.loc[treatment_var].iloc[1]
        n_obs = model._N
        r2 = model._r2 if hasattr(model, '_r2') else None

        # Build coefficient vector
        coef_vector = {
            "treatment": {
                "var": treatment_var,
                "coef": float(coef),
                "se": float(se),
                "pval": float(pval)
            },
            "controls": [],
            "fixed_effects": fixed_effects.split(' + ') if fixed_effects else [],
            "diagnostics": {}
        }

        # Add all coefficients
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
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
    except Exception as e:
        print(f"  Error extracting results for {spec_id}: {e}")
        return None

def run_spec(formula, data, vcov, spec_id, spec_tree_path, outcome_var, treatment_var,
             sample_desc, fixed_effects, controls_desc, cluster_var, model_type,
             weight_col=None):
    """Run a single specification and add to results"""
    try:
        # Filter to valid observations for the outcome and treatment
        vars_needed = [outcome_var, treatment_var]
        data_clean = data.dropna(subset=vars_needed).copy()

        if weight_col and weight_col in data_clean.columns:
            data_clean = data_clean[data_clean[weight_col] > 0]
            data_clean = data_clean[data_clean[weight_col].notna()]
            model = pf.feols(formula, data=data_clean, vcov=vcov, weights=weight_col)
        else:
            model = pf.feols(formula, data=data_clean, vcov=vcov)

        result = extract_results(model, spec_id, spec_tree_path, outcome_var, treatment_var,
                                sample_desc, fixed_effects, controls_desc, cluster_var, model_type)
        if result:
            results.append(result)
            print(f"  {spec_id}: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}, n={result['n_obs']}")
    except Exception as e:
        print(f"  ERROR {spec_id}: {e}")

# ========================================================================
# PREPARE ANALYSIS SAMPLE
# ========================================================================
print("\n" + "="*70)
print("PREPARING ANALYSIS SAMPLE")
print("="*70)

# Main analysis sample: firms with valid treatment and outcome
analysis_df = df[df['share_enemies_1952_100'].notna() &
                 df['lnwage'].notna() &
                 df['employees2018'].notna() &
                 (df['employees2018'] > 0) &
                 df['Oblast'].notna()].copy()

# Need to handle Oblast as categorical for FE
analysis_df['Oblast'] = analysis_df['Oblast'].astype(int).astype(str).astype('category')

# Prepare gulag_cluster for clustering
if 'gulag_cluster' in analysis_df.columns:
    analysis_df['gulag_cluster'] = analysis_df['gulag_cluster'].fillna('missing')
    analysis_df = analysis_df[analysis_df['gulag_cluster'] != '']

# Create additional variables needed
# Log enemies
if 'lnenemies100Gulag1952' not in analysis_df.columns and 'enemies100Gulag1952' in analysis_df.columns:
    analysis_df['lnenemies100Gulag1952'] = np.log(1 + analysis_df['enemies100Gulag1952'])

# Alternative denominator - using 1926 population
if 'pop_within_100km_1926' in analysis_df.columns and 'enemies100Gulag1952' in analysis_df.columns and 'people_100_1952' in analysis_df.columns:
    analysis_df['enemy_share_pop1926_100'] = analysis_df['enemies100Gulag1952'] / (analysis_df['pop_within_100km_1926'] + analysis_df['people_100_1952'])

# 1939 camps
if 'share_enemies_1939_100' not in analysis_df.columns:
    if 'enemies100Gulag1939' in analysis_df.columns and 'people_100_1939' in analysis_df.columns:
        analysis_df['share_enemies_1939_100'] = analysis_df['enemies100Gulag1939'] / analysis_df['people_100_1939']
        analysis_df['ln100people1939'] = np.log(analysis_df['people_100_1939'])

# Create spatial treatment variables for different radii
for radius in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
    treat_var = f'share_enemies_1952_{radius}'
    if treat_var not in analysis_df.columns:
        enemies_var = f'enemies{radius}Gulag1952'
        people_var = f'people_{radius}_1952'
        if enemies_var in analysis_df.columns and people_var in analysis_df.columns:
            analysis_df[treat_var] = analysis_df[enemies_var] / analysis_df[people_var]

print(f"Analysis sample size: {len(analysis_df)}")

# ========================================================================
# BASELINE SPECIFICATIONS (Table 5 replication - wages)
# ========================================================================
print("\n" + "="*70)
print("BASELINE SPECIFICATIONS")
print("="*70)

print("\n1. Baseline specifications (Table 5 replication):")

# Baseline 1: Table 5 Column 1 - Treatment only, Oblast FE
run_spec("lnwage ~ share_enemies_1952_100 | Oblast",
         analysis_df,
         {'CRV1': 'gulag_cluster'},
         "baseline",
         "methods/cross_sectional_ols.md#baseline",
         "lnwage", "share_enemies_1952_100",
         "Full sample, weighted by employees",
         "Oblast", "Treatment only",
         "gulag_cluster", "WLS with FE",
         weight_col='employees2018')

# Baseline 2: With prisoners control (Column 2)
run_spec("lnwage ~ share_enemies_1952_100 + ln100people1952 + latitude + longitude | Oblast",
         analysis_df,
         {'CRV1': 'gulag_cluster'},
         "baseline_w_controls_1",
         "methods/cross_sectional_ols.md#baseline",
         "lnwage", "share_enemies_1952_100",
         "Full sample, weighted by employees",
         "Oblast", "+ ln prisoners, lat, lon",
         "gulag_cluster", "WLS with FE",
         weight_col='employees2018')

# Baseline 3: Full controls (Column 3)
run_spec("lnwage ~ share_enemies_1952_100 + ln100people1952 + latitude + longitude + lnpop_within_100km_1926 + lnkm_to_1937_tracks | Oblast",
         analysis_df,
         {'CRV1': 'gulag_cluster'},
         "baseline_full_controls",
         "methods/cross_sectional_ols.md#baseline",
         "lnwage", "share_enemies_1952_100",
         "Full sample, weighted by employees",
         "Oblast", "+ pop 1926, railway dist",
         "gulag_cluster", "WLS with FE",
         weight_col='employees2018')

# Baseline 4-6: Excluding Moscow
moscow_df = analysis_df[analysis_df['dist_to_Moscow'] > 100].copy()

run_spec("lnwage ~ share_enemies_1952_100 + ln100people1952 | Oblast",
         moscow_df,
         {'CRV1': 'gulag_cluster'},
         "baseline_no_moscow",
         "methods/cross_sectional_ols.md#baseline",
         "lnwage", "share_enemies_1952_100",
         "Excluding Moscow (>100km)",
         "Oblast", "+ ln prisoners",
         "gulag_cluster", "WLS with FE",
         weight_col='employees2018')

run_spec("lnwage ~ share_enemies_1952_100 + ln100people1952 + latitude + longitude + lnpop_within_100km_1926 + lnkm_to_1937_tracks | Oblast",
         moscow_df,
         {'CRV1': 'gulag_cluster'},
         "baseline_no_moscow_full",
         "methods/cross_sectional_ols.md#baseline",
         "lnwage", "share_enemies_1952_100",
         "Excluding Moscow (>100km), full controls",
         "Oblast", "+ pop 1926, railway dist",
         "gulag_cluster", "WLS with FE",
         weight_col='employees2018')

# ========================================================================
# ALTERNATIVE OUTCOMES (Table 7, Table A10)
# ========================================================================
print("\n" + "="*70)
print("ALTERNATIVE OUTCOMES")
print("="*70)

# Profits
run_spec("lnnetprofitpc_rawemp ~ share_enemies_1952_100 + ln100people1952 | Oblast",
         analysis_df,
         {'CRV1': 'gulag_cluster'},
         "robust/outcome/profits",
         "robustness/alternative_outcomes.md",
         "lnnetprofitpc_rawemp", "share_enemies_1952_100",
         "Full sample",
         "Oblast", "+ ln prisoners",
         "gulag_cluster", "WLS with FE",
         weight_col='employees2018')

run_spec("lnnetprofitpc_rawemp ~ share_enemies_1952_100 + ln100people1952 + latitude + longitude + lnpop_within_100km_1926 + lnkm_to_1937_tracks | Oblast",
         analysis_df,
         {'CRV1': 'gulag_cluster'},
         "robust/outcome/profits_full",
         "robustness/alternative_outcomes.md",
         "lnnetprofitpc_rawemp", "share_enemies_1952_100",
         "Full sample, full controls",
         "Oblast", "Full controls",
         "gulag_cluster", "WLS with FE",
         weight_col='employees2018')

# Value added
run_spec("lnvalueaddedpc_rawemp ~ share_enemies_1952_100 + ln100people1952 | Oblast",
         analysis_df,
         {'CRV1': 'gulag_cluster'},
         "robust/outcome/value_added",
         "robustness/alternative_outcomes.md",
         "lnvalueaddedpc_rawemp", "share_enemies_1952_100",
         "Full sample",
         "Oblast", "+ ln prisoners",
         "gulag_cluster", "WLS with FE",
         weight_col='employees2018')

run_spec("lnvalueaddedpc_rawemp ~ share_enemies_1952_100 + ln100people1952 + latitude + longitude + lnpop_within_100km_1926 + lnkm_to_1937_tracks | Oblast",
         analysis_df,
         {'CRV1': 'gulag_cluster'},
         "robust/outcome/value_added_full",
         "robustness/alternative_outcomes.md",
         "lnvalueaddedpc_rawemp", "share_enemies_1952_100",
         "Full sample, full controls",
         "Oblast", "Full controls",
         "gulag_cluster", "WLS with FE",
         weight_col='employees2018')

# Revenue
run_spec("lnrevenuepc_rawemp ~ share_enemies_1952_100 + ln100people1952 | Oblast",
         analysis_df,
         {'CRV1': 'gulag_cluster'},
         "robust/outcome/revenue",
         "robustness/alternative_outcomes.md",
         "lnrevenuepc_rawemp", "share_enemies_1952_100",
         "Full sample",
         "Oblast", "+ ln prisoners",
         "gulag_cluster", "WLS with FE",
         weight_col='employees2018')

# ========================================================================
# ALTERNATIVE TREATMENT DEFINITIONS (Table A8, A9)
# ========================================================================
print("\n" + "="*70)
print("ALTERNATIVE TREATMENT DEFINITIONS")
print("="*70)

# Log enemies (Table A8)
run_spec("lnwage ~ lnenemies100Gulag1952 + ln100people1952 | Oblast",
         analysis_df,
         {'CRV1': 'gulag_cluster'},
         "robust/treatment/ln_enemies",
         "robustness/alternative_treatment.md",
         "lnwage", "lnenemies100Gulag1952",
         "Full sample",
         "Oblast", "+ ln prisoners",
         "gulag_cluster", "WLS with FE",
         weight_col='employees2018')

run_spec("lnwage ~ lnenemies100Gulag1952 + ln100people1952 + latitude + longitude + lnpop_within_100km_1926 + lnkm_to_1937_tracks | Oblast",
         analysis_df,
         {'CRV1': 'gulag_cluster'},
         "robust/treatment/ln_enemies_full",
         "robustness/alternative_treatment.md",
         "lnwage", "lnenemies100Gulag1952",
         "Full sample, full controls",
         "Oblast", "Full controls",
         "gulag_cluster", "WLS with FE",
         weight_col='employees2018')

# Alternative denominator (Table A9)
run_spec("lnwage ~ enemy_share_pop1926_100 + ln100people1952 | Oblast",
         analysis_df,
         {'CRV1': 'gulag_cluster'},
         "robust/treatment/pop1926_denom",
         "robustness/alternative_treatment.md",
         "lnwage", "enemy_share_pop1926_100",
         "Full sample, 1926 pop denominator",
         "Oblast", "+ ln prisoners",
         "gulag_cluster", "WLS with FE",
         weight_col='employees2018')

# 1939 camps instead of 1952
if 'share_enemies_1939_100' in analysis_df.columns:
    run_spec("lnwage ~ share_enemies_1939_100 + ln100people1952 | Oblast",
             analysis_df,
             {'CRV1': 'gulag_cluster'},
             "robust/treatment/1939_camps",
             "robustness/alternative_treatment.md",
             "lnwage", "share_enemies_1939_100",
             "Using 1939 camp data",
             "Oblast", "+ ln prisoners",
             "gulag_cluster", "WLS with FE",
             weight_col='employees2018')

# ========================================================================
# SPATIAL VARIATIONS (Figure A11 - different radii)
# ========================================================================
print("\n" + "="*70)
print("SPATIAL VARIATIONS (Different radii)")
print("="*70)

for radius in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
    treat_var = f'share_enemies_1952_{radius}'
    if treat_var in analysis_df.columns:
        run_spec(f"lnwage ~ {treat_var} | Oblast",
                 analysis_df,
                 {'CRV1': 'gulag_cluster'},
                 f"robust/spatial/radius_{radius}km",
                 "robustness/spatial_variations.md",
                 "lnwage", treat_var,
                 f"Treatment within {radius}km",
                 "Oblast", "Treatment only",
                 "gulag_cluster", "WLS with FE",
                 weight_col='employees2018')

# ========================================================================
# CONTROL VARIATIONS
# ========================================================================
print("\n" + "="*70)
print("CONTROL VARIATIONS")
print("="*70)

# No controls (bivariate)
run_spec("lnwage ~ share_enemies_1952_100 | Oblast",
         analysis_df,
         {'CRV1': 'gulag_cluster'},
         "robust/control/none",
         "robustness/control_variations.md",
         "lnwage", "share_enemies_1952_100",
         "No controls beyond FE",
         "Oblast", "None",
         "gulag_cluster", "WLS with FE",
         weight_col='employees2018')

# Only prisoners
run_spec("lnwage ~ share_enemies_1952_100 + ln100people1952 | Oblast",
         analysis_df,
         {'CRV1': 'gulag_cluster'},
         "robust/control/prisoners_only",
         "robustness/control_variations.md",
         "lnwage", "share_enemies_1952_100",
         "Prisoners control only",
         "Oblast", "+ ln prisoners",
         "gulag_cluster", "WLS with FE",
         weight_col='employees2018')

# Only geography
run_spec("lnwage ~ share_enemies_1952_100 + latitude + longitude | Oblast",
         analysis_df,
         {'CRV1': 'gulag_cluster'},
         "robust/control/geography_only",
         "robustness/control_variations.md",
         "lnwage", "share_enemies_1952_100",
         "Geography controls only",
         "Oblast", "+ lat, lon",
         "gulag_cluster", "WLS with FE",
         weight_col='employees2018')

# Drop lat/lon from full
run_spec("lnwage ~ share_enemies_1952_100 + ln100people1952 + lnpop_within_100km_1926 + lnkm_to_1937_tracks | Oblast",
         analysis_df,
         {'CRV1': 'gulag_cluster'},
         "robust/control/drop_geography",
         "robustness/control_variations.md",
         "lnwage", "share_enemies_1952_100",
         "Full controls without lat/lon",
         "Oblast", "No geography",
         "gulag_cluster", "WLS with FE",
         weight_col='employees2018')

# Drop 1926 population
run_spec("lnwage ~ share_enemies_1952_100 + ln100people1952 + latitude + longitude + lnkm_to_1937_tracks | Oblast",
         analysis_df,
         {'CRV1': 'gulag_cluster'},
         "robust/control/drop_pop1926",
         "robustness/control_variations.md",
         "lnwage", "share_enemies_1952_100",
         "Without 1926 population",
         "Oblast", "No pop 1926",
         "gulag_cluster", "WLS with FE",
         weight_col='employees2018')

# Drop railway distance
run_spec("lnwage ~ share_enemies_1952_100 + ln100people1952 + latitude + longitude + lnpop_within_100km_1926 | Oblast",
         analysis_df,
         {'CRV1': 'gulag_cluster'},
         "robust/control/drop_railway",
         "robustness/control_variations.md",
         "lnwage", "share_enemies_1952_100",
         "Without railway distance",
         "Oblast", "No railway",
         "gulag_cluster", "WLS with FE",
         weight_col='employees2018')

# ========================================================================
# SAMPLE RESTRICTIONS
# ========================================================================
print("\n" + "="*70)
print("SAMPLE RESTRICTIONS")
print("="*70)

# Exclude mining firms
no_mining = analysis_df[analysis_df['Description_Parent'] != 'MINING AND QUARRYING'].copy()
run_spec("lnwage ~ share_enemies_1952_100 + ln100people1952 | Oblast",
         no_mining,
         {'CRV1': 'gulag_cluster'},
         "robust/sample/no_mining",
         "robustness/sample_restrictions.md",
         "lnwage", "share_enemies_1952_100",
         "Excluding mining firms",
         "Oblast", "+ ln prisoners",
         "gulag_cluster", "WLS with FE",
         weight_col='employees2018')

# Firm size restrictions
for min_emp in [10, 50, 100, 200]:
    size_df = analysis_df[analysis_df['employees2018'] >= min_emp].copy()
    run_spec("lnwage ~ share_enemies_1952_100 + ln100people1952 | Oblast",
             size_df,
             {'CRV1': 'gulag_cluster'},
             f"robust/sample/min_{min_emp}_employees",
             "robustness/sample_restrictions.md",
             "lnwage", "share_enemies_1952_100",
             f"Firms with >= {min_emp} employees",
             "Oblast", "+ ln prisoners",
             "gulag_cluster", "WLS with FE",
             weight_col='employees2018')

# Enemy share restrictions
high_enemy = analysis_df[analysis_df['share_enemies_1952_100'] < 0.9].copy()
run_spec("lnwage ~ share_enemies_1952_100 + ln100people1952 | Oblast",
         high_enemy,
         {'CRV1': 'gulag_cluster'},
         "robust/sample/no_high_enemy",
         "robustness/sample_restrictions.md",
         "lnwage", "share_enemies_1952_100",
         "Excluding >90% enemy camps",
         "Oblast", "+ ln prisoners",
         "gulag_cluster", "WLS with FE",
         weight_col='employees2018')

# Only camps with some enemies
some_enemy = analysis_df[(analysis_df['share_enemies_1952_100'] > 0) &
                         (analysis_df['share_enemies_1952_100'].notna())].copy()
run_spec("lnwage ~ share_enemies_1952_100 + ln100people1952 | Oblast",
         some_enemy,
         {'CRV1': 'gulag_cluster'},
         "robust/sample/positive_enemy",
         "robustness/sample_restrictions.md",
         "lnwage", "share_enemies_1952_100",
         "Only camps with enemies >0%",
         "Oblast", "+ ln prisoners",
         "gulag_cluster", "WLS with FE",
         weight_col='employees2018')

# Winsorize outcome at 1%
analysis_df['lnwage_wins1'] = analysis_df['lnwage'].clip(
    lower=analysis_df['lnwage'].quantile(0.01),
    upper=analysis_df['lnwage'].quantile(0.99)
)
run_spec("lnwage_wins1 ~ share_enemies_1952_100 + ln100people1952 | Oblast",
         analysis_df,
         {'CRV1': 'gulag_cluster'},
         "robust/sample/winsorize_1pct",
         "robustness/sample_restrictions.md",
         "lnwage_wins1", "share_enemies_1952_100",
         "Winsorized wages at 1%/99%",
         "Oblast", "+ ln prisoners",
         "gulag_cluster", "WLS with FE",
         weight_col='employees2018')

# Winsorize at 5%
analysis_df['lnwage_wins5'] = analysis_df['lnwage'].clip(
    lower=analysis_df['lnwage'].quantile(0.05),
    upper=analysis_df['lnwage'].quantile(0.95)
)
run_spec("lnwage_wins5 ~ share_enemies_1952_100 + ln100people1952 | Oblast",
         analysis_df,
         {'CRV1': 'gulag_cluster'},
         "robust/sample/winsorize_5pct",
         "robustness/sample_restrictions.md",
         "lnwage_wins5", "share_enemies_1952_100",
         "Winsorized wages at 5%/95%",
         "Oblast", "+ ln prisoners",
         "gulag_cluster", "WLS with FE",
         weight_col='employees2018')

# Trim outliers
trim_df = analysis_df[(analysis_df['lnwage'] > analysis_df['lnwage'].quantile(0.01)) &
                      (analysis_df['lnwage'] < analysis_df['lnwage'].quantile(0.99))].copy()
run_spec("lnwage ~ share_enemies_1952_100 + ln100people1952 | Oblast",
         trim_df,
         {'CRV1': 'gulag_cluster'},
         "robust/sample/trim_1pct",
         "robustness/sample_restrictions.md",
         "lnwage", "share_enemies_1952_100",
         "Trimmed 1%/99% of wages",
         "Oblast", "+ ln prisoners",
         "gulag_cluster", "WLS with FE",
         weight_col='employees2018')

# ========================================================================
# CLUSTERING VARIATIONS
# ========================================================================
print("\n" + "="*70)
print("CLUSTERING VARIATIONS")
print("="*70)

# Robust SE (no clustering)
run_spec("lnwage ~ share_enemies_1952_100 + ln100people1952 | Oblast",
         analysis_df,
         'hetero',
         "robust/cluster/robust_se",
         "robustness/clustering_variations.md",
         "lnwage", "share_enemies_1952_100",
         "Full sample",
         "Oblast", "+ ln prisoners",
         "robust", "WLS with FE",
         weight_col='employees2018')

# Cluster by Oblast (region)
run_spec("lnwage ~ share_enemies_1952_100 + ln100people1952 | Oblast",
         analysis_df,
         {'CRV1': 'Oblast'},
         "robust/cluster/region",
         "robustness/clustering_variations.md",
         "lnwage", "share_enemies_1952_100",
         "Full sample",
         "Oblast", "+ ln prisoners",
         "Oblast", "WLS with FE",
         weight_col='employees2018')

# Cluster by region_camp (if available)
if 'region_camp' in analysis_df.columns:
    region_df = analysis_df[analysis_df['region_camp'].notna() & (analysis_df['region_camp'] != '')].copy()
    if len(region_df) > 100:
        run_spec("lnwage ~ share_enemies_1952_100 + ln100people1952 | Oblast",
                 region_df,
                 {'CRV1': 'region_camp'},
                 "robust/cluster/camp_region",
                 "robustness/clustering_variations.md",
                 "lnwage", "share_enemies_1952_100",
                 "Full sample",
                 "Oblast", "+ ln prisoners",
                 "region_camp", "WLS with FE",
                 weight_col='employees2018')

# ========================================================================
# ESTIMATION METHOD VARIATIONS
# ========================================================================
print("\n" + "="*70)
print("ESTIMATION METHOD VARIATIONS")
print("="*70)

# No fixed effects
run_spec("lnwage ~ share_enemies_1952_100 + ln100people1952",
         analysis_df,
         {'CRV1': 'gulag_cluster'},
         "robust/estimation/no_fe",
         "robustness/estimation_variations.md",
         "lnwage", "share_enemies_1952_100",
         "Full sample",
         "None", "+ ln prisoners",
         "gulag_cluster", "WLS no FE",
         weight_col='employees2018')

# Unweighted (OLS instead of WLS)
run_spec("lnwage ~ share_enemies_1952_100 + ln100people1952 | Oblast",
         analysis_df,
         {'CRV1': 'gulag_cluster'},
         "robust/weights/unweighted",
         "robustness/weight_variations.md",
         "lnwage", "share_enemies_1952_100",
         "Full sample, unweighted",
         "Oblast", "+ ln prisoners",
         "gulag_cluster", "OLS with FE")

# Add sector FE
sector_df = analysis_df[analysis_df['Description_Parent'].notna()].copy()
sector_df['sector'] = sector_df['Description_Parent'].astype('category')
run_spec("lnwage ~ share_enemies_1952_100 + ln100people1952 | Oblast + sector",
         sector_df,
         {'CRV1': 'gulag_cluster'},
         "robust/estimation/sector_fe",
         "robustness/estimation_variations.md",
         "lnwage", "share_enemies_1952_100",
         "With sector FE",
         "Oblast + sector", "+ ln prisoners",
         "gulag_cluster", "WLS with FE",
         weight_col='employees2018')

# ========================================================================
# HETEROGENEITY ANALYSIS
# ========================================================================
print("\n" + "="*70)
print("HETEROGENEITY ANALYSIS")
print("="*70)

# Create Gulag category dummies (as in Table 6)
analysis_df['gulag_low'] = ((analysis_df['share_enemies_1952_100'] < 0.01) &
                            (analysis_df['share_enemies_1952_100'].notna())).astype(int)
analysis_df['gulag_high'] = ((analysis_df['share_enemies_1952_100'] >= 0.2) &
                             (analysis_df['share_enemies_1952_100'].notna())).astype(int)
analysis_df['gulag_med'] = ((analysis_df['share_enemies_1952_100'] >= 0.01) &
                            (analysis_df['share_enemies_1952_100'] < 0.2) &
                            (analysis_df['share_enemies_1952_100'].notna())).astype(int)

# Table 6 - categorical treatment
run_spec("lnwage ~ gulag_med + gulag_high | Oblast",
         analysis_df,
         {'CRV1': 'Oblast'},
         "robust/heterogeneity/gulag_categories",
         "robustness/heterogeneity.md",
         "lnwage", "gulag_high",
         "Categorical treatment (high enemy share)",
         "Oblast", "Category dummies",
         "Oblast", "WLS with FE",
         weight_col='employees2018')

run_spec("lnwage ~ gulag_med + gulag_high + latitude + longitude + lnpop_within_100km_1926 + lnkm_to_1937_tracks | Oblast",
         analysis_df,
         {'CRV1': 'Oblast'},
         "robust/heterogeneity/gulag_categories_full",
         "robustness/heterogeneity.md",
         "lnwage", "gulag_high",
         "Categorical treatment with full controls",
         "Oblast", "Category dummies + controls",
         "Oblast", "WLS with FE",
         weight_col='employees2018')

# By sector - run for major sectors
major_sectors = analysis_df['Description_Parent'].value_counts().head(10).index.tolist()
for sector in major_sectors:
    if pd.notna(sector) and sector != '':
        sector_sub = analysis_df[analysis_df['Description_Parent'] == sector].copy()
        if len(sector_sub) > 100:
            safe_sector = sector.replace(' ', '_').replace(',', '').replace('/', '_').replace(';', '')[:30]
            run_spec("lnwage ~ share_enemies_1952_100 | Oblast",
                     sector_sub,
                     {'CRV1': 'gulag_cluster'},
                     f"robust/heterogeneity/sector_{safe_sector}",
                     "robustness/heterogeneity.md",
                     "lnwage", "share_enemies_1952_100",
                     f"Sector: {sector[:40]}",
                     "Oblast", "Treatment only",
                     "gulag_cluster", "WLS with FE",
                     weight_col='employees2018')

# ========================================================================
# PLACEBO TESTS
# ========================================================================
print("\n" + "="*70)
print("PLACEBO TESTS")
print("="*70)

# Pre-1952 data - using 1939 enemy share
if 'share_enemies_1939_100' in analysis_df.columns:
    df_1939 = analysis_df[analysis_df['share_enemies_1939_100'].notna()].copy()
    run_spec("lnwage ~ share_enemies_1939_100 + share_enemies_1952_100 + ln100people1952 | Oblast",
             df_1939,
             {'CRV1': 'gulag_cluster'},
             "robust/placebo/1939_controlling_1952",
             "robustness/placebo_tests.md",
             "lnwage", "share_enemies_1939_100",
             "1939 treatment controlling for 1952",
             "Oblast", "+ 1952 treatment, ln prisoners",
             "gulag_cluster", "WLS with FE",
             weight_col='employees2018')

# ========================================================================
# ADDITIONAL ROBUSTNESS FROM PAPER
# ========================================================================
print("\n" + "="*70)
print("ADDITIONAL ROBUSTNESS")
print("="*70)

# Mining town restriction
analysis_df['is_mining'] = (analysis_df['Description_Parent'] == 'MINING AND QUARRYING').astype(int)
mining_share = analysis_df.groupby('gulag_cluster')['is_mining'].transform('mean')
analysis_df['mining_share'] = mining_share

no_mining_towns = analysis_df[analysis_df['mining_share'] < 0.02].copy()
run_spec("lnwage ~ share_enemies_1952_100 + ln100people1952 | Oblast",
         no_mining_towns,
         {'CRV1': 'gulag_cluster'},
         "robust/sample/no_mining_towns",
         "robustness/sample_restrictions.md",
         "lnwage", "share_enemies_1952_100",
         "Excluding mining towns (<2% mining)",
         "Oblast", "+ ln prisoners",
         "gulag_cluster", "WLS with FE",
         weight_col='employees2018')

# Combined restrictions - excluding Moscow + no mining
combined_df = analysis_df[(analysis_df['dist_to_Moscow'] > 100) &
                          (analysis_df['Description_Parent'] != 'MINING AND QUARRYING')].copy()
run_spec("lnwage ~ share_enemies_1952_100 + ln100people1952 + latitude + longitude + lnpop_within_100km_1926 + lnkm_to_1937_tracks | Oblast",
         combined_df,
         {'CRV1': 'gulag_cluster'},
         "robust/sample/no_moscow_no_mining",
         "robustness/sample_restrictions.md",
         "lnwage", "share_enemies_1952_100",
         "No Moscow, no mining, full controls",
         "Oblast", "Full controls",
         "gulag_cluster", "WLS with FE",
         weight_col='employees2018')

# ========================================================================
# INTERACTION EFFECTS
# ========================================================================
print("\n" + "="*70)
print("INTERACTION EFFECTS")
print("="*70)

# Interaction with firm size
analysis_df['large_firm'] = (analysis_df['employees2018'] >= 100).astype(int)
run_spec("lnwage ~ share_enemies_1952_100 * large_firm + ln100people1952 | Oblast",
         analysis_df,
         {'CRV1': 'gulag_cluster'},
         "robust/heterogeneity/large_firm_interact",
         "robustness/heterogeneity.md",
         "lnwage", "share_enemies_1952_100",
         "Interaction with large firm dummy",
         "Oblast", "+ ln prisoners, interaction",
         "gulag_cluster", "WLS with FE",
         weight_col='employees2018')

# Interaction with distance to Moscow
analysis_df['far_from_moscow'] = (analysis_df['dist_to_Moscow'] > 500).astype(int)
run_spec("lnwage ~ share_enemies_1952_100 * far_from_moscow + ln100people1952 | Oblast",
         analysis_df,
         {'CRV1': 'gulag_cluster'},
         "robust/heterogeneity/moscow_distance_interact",
         "robustness/heterogeneity.md",
         "lnwage", "share_enemies_1952_100",
         "Interaction with far from Moscow",
         "Oblast", "+ ln prisoners, interaction",
         "gulag_cluster", "WLS with FE",
         weight_col='employees2018')

# ========================================================================
# FUNCTIONAL FORM VARIATIONS
# ========================================================================
print("\n" + "="*70)
print("FUNCTIONAL FORM VARIATIONS")
print("="*70)

# Quadratic treatment
analysis_df['share_enemies_sq'] = analysis_df['share_enemies_1952_100'] ** 2
run_spec("lnwage ~ share_enemies_1952_100 + share_enemies_sq + ln100people1952 | Oblast",
         analysis_df,
         {'CRV1': 'gulag_cluster'},
         "robust/funcform/quadratic_treatment",
         "robustness/functional_form.md",
         "lnwage", "share_enemies_1952_100",
         "Quadratic treatment specification",
         "Oblast", "+ squared treatment, ln prisoners",
         "gulag_cluster", "WLS with FE",
         weight_col='employees2018')

# Level of wage instead of log
analysis_df['wage_level'] = analysis_df['medicinsurance2018'] / analysis_df['employees2018']
run_spec("wage_level ~ share_enemies_1952_100 + ln100people1952 | Oblast",
         analysis_df,
         {'CRV1': 'gulag_cluster'},
         "robust/funcform/wage_level",
         "robustness/functional_form.md",
         "wage_level", "share_enemies_1952_100",
         "Wage in levels (not log)",
         "Oblast", "+ ln prisoners",
         "gulag_cluster", "WLS with FE",
         weight_col='employees2018')

# ========================================================================
# SAVE RESULTS
# ========================================================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Convert to DataFrame
results_df = pd.DataFrame(results)
print(f"Total specifications run: {len(results_df)}")

# Save to CSV
output_path = f"{OUTPUT_DIR}/specification_results.csv"
results_df.to_csv(output_path, index=False)
print(f"Results saved to: {output_path}")

# Summary statistics
if len(results_df) > 0:
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    # Filter to main treatment variable specs only for summary
    main_specs = results_df[results_df['treatment_var'] == 'share_enemies_1952_100']
    print(f"\nMain treatment variable specifications: {len(main_specs)}")
    print(f"Positive coefficients: {(main_specs['coefficient'] > 0).sum()} ({100*(main_specs['coefficient'] > 0).mean():.1f}%)")
    print(f"Significant at 5%: {(main_specs['p_value'] < 0.05).sum()} ({100*(main_specs['p_value'] < 0.05).mean():.1f}%)")
    print(f"Significant at 1%: {(main_specs['p_value'] < 0.01).sum()} ({100*(main_specs['p_value'] < 0.01).mean():.1f}%)")
    print(f"Median coefficient: {main_specs['coefficient'].median():.4f}")
    print(f"Mean coefficient: {main_specs['coefficient'].mean():.4f}")
    print(f"Range: [{main_specs['coefficient'].min():.4f}, {main_specs['coefficient'].max():.4f}]")

    # All specs summary
    print(f"\nAll specifications: {len(results_df)}")
    print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
    print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
    print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")

    # Category breakdown
    print("\n" + "="*70)
    print("SPECIFICATION BREAKDOWN BY CATEGORY")
    print("="*70)

    def categorize_spec(spec_id):
        if spec_id.startswith('baseline'):
            return 'Baseline'
        elif 'outcome' in spec_id:
            return 'Alternative outcomes'
        elif 'treatment' in spec_id:
            return 'Alternative treatments'
        elif 'spatial' in spec_id:
            return 'Spatial variations'
        elif 'control' in spec_id:
            return 'Control variations'
        elif 'sample' in spec_id:
            return 'Sample restrictions'
        elif 'cluster' in spec_id:
            return 'Inference variations'
        elif 'estimation' in spec_id or 'weights' in spec_id:
            return 'Estimation method'
        elif 'heterogeneity' in spec_id:
            return 'Heterogeneity'
        elif 'placebo' in spec_id:
            return 'Placebo tests'
        elif 'funcform' in spec_id:
            return 'Functional form'
        else:
            return 'Other'

    results_df['category'] = results_df['spec_id'].apply(categorize_spec)
    category_summary = results_df.groupby('category').agg({
        'spec_id': 'count',
        'coefficient': lambda x: (x > 0).sum(),
        'p_value': lambda x: (x < 0.05).sum()
    }).rename(columns={'spec_id': 'N', 'coefficient': 'N_positive', 'p_value': 'N_sig_5pct'})

    category_summary['pct_positive'] = 100 * category_summary['N_positive'] / category_summary['N']
    category_summary['pct_sig_5pct'] = 100 * category_summary['N_sig_5pct'] / category_summary['N']

    print(category_summary.round(1))
else:
    print("No results to summarize - all specifications failed")

print("\nDone!")
