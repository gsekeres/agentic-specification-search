"""
Specification Search for Paper 125321-V1
"Can Technology Solve the Principal-Agent Problem? Evidence from China's War on Air Pollution"
Authors: Greenstone, He, Jia, Liu

Method: Regression Discontinuity Design with Event Study
Treatment: Automation of air pollution monitoring stations
Running variable: Days/months relative to automation date
Cutoff: T = 0 (automation date)
"""

import pandas as pd
import numpy as np
import json
import warnings
from datetime import datetime, timedelta
import pyfixest as pf

warnings.filterwarnings('ignore')

# Paths
DATA_PATH = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/125321-V1/China_Pollution_Monitoring/Data/'
OUTPUT_PATH = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/125321-V1/'

# Paper metadata
PAPER_ID = '125321-V1'
JOURNAL = 'AER'
PAPER_TITLE = 'Can Technology Solve the Principal-Agent Problem? Evidence from China\'s War on Air Pollution'

# Results storage
results = []

def safe_float(x):
    """Safely convert to float"""
    try:
        return float(x)
    except:
        return np.nan

def get_coefficient_vector(model, treatment_var, controls, fixed_effects=None):
    """Extract coefficient vector from model"""
    try:
        coef_vec = {
            'treatment': {
                'var': treatment_var,
                'coef': safe_float(model.coef().get(treatment_var, np.nan)),
                'se': safe_float(model.se().get(treatment_var, np.nan)),
                'pval': safe_float(model.pvalue().get(treatment_var, np.nan))
            },
            'controls': [],
            'fixed_effects': fixed_effects if fixed_effects else [],
            'diagnostics': {}
        }

        for c in controls:
            if c in model.coef().index:
                coef_vec['controls'].append({
                    'var': c,
                    'coef': safe_float(model.coef().get(c, np.nan)),
                    'se': safe_float(model.se().get(c, np.nan)),
                    'pval': safe_float(model.pvalue().get(c, np.nan))
                })
    except:
        coef_vec = {'treatment': {'var': treatment_var}, 'controls': [], 'fixed_effects': [], 'diagnostics': {}}

    return coef_vec

def add_result(spec_id, spec_tree_path, outcome_var, treatment_var,
               coefficient, std_error, t_stat, p_value,
               ci_lower, ci_upper, n_obs, r_squared,
               coefficient_vector_json, sample_desc, fixed_effects,
               controls_desc, cluster_var, model_type):
    """Add a result to the results list"""
    results.append({
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'outcome_var': outcome_var,
        'treatment_var': treatment_var,
        'coefficient': coefficient,
        'std_error': std_error,
        't_stat': t_stat,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_obs': n_obs,
        'r_squared': r_squared,
        'coefficient_vector_json': json.dumps(coefficient_vector_json),
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })

def run_rd_pyfixest(df, outcome, running_var, cutoff, controls, fe_vars, cluster_var,
                    spec_id, spec_tree_path, sample_desc, polynomial=1):
    """Run parametric RD regression using pyfixest"""
    try:
        # Prepare data
        keep_vars = [outcome, running_var, cluster_var] + controls + fe_vars
        keep_vars = list(set([v for v in keep_vars if v in df.columns]))
        df_clean = df[keep_vars].dropna().copy()

        if len(df_clean) < 100:
            return False

        # Create treatment indicator and centered running variable
        df_clean['T_centered'] = df_clean[running_var] - cutoff
        df_clean['after'] = (df_clean['T_centered'] >= 0).astype(int)
        df_clean['after_T'] = df_clean['after'] * df_clean['T_centered']

        # Add polynomial terms
        poly_terms = ['T_centered', 'after_T']
        for p in range(2, polynomial + 1):
            df_clean[f'T{p}'] = df_clean['T_centered'] ** p
            df_clean[f'after_T{p}'] = df_clean['after'] * df_clean[f'T{p}']
            poly_terms.extend([f'T{p}', f'after_T{p}'])

        # Build formula
        rhs = ['after'] + poly_terms + controls
        rhs_str = ' + '.join(rhs)

        if fe_vars:
            fe_str = ' + '.join(fe_vars)
            formula = f'{outcome} ~ {rhs_str} | {fe_str}'
        else:
            formula = f'{outcome} ~ {rhs_str}'

        # Run regression
        model = pf.feols(formula, data=df_clean, vcov={'CRV1': cluster_var})

        coef = model.coef()['after']
        se = model.se()['after']
        t = model.tstat()['after']
        pval = model.pvalue()['after']
        ci = model.confint().loc['after']

        coef_vec = get_coefficient_vector(model, 'after', controls, fe_vars)

        add_result(
            spec_id=spec_id,
            spec_tree_path=spec_tree_path,
            outcome_var=outcome,
            treatment_var='after',
            coefficient=float(coef),
            std_error=float(se),
            t_stat=float(t),
            p_value=float(pval),
            ci_lower=float(ci['2.5%']),
            ci_upper=float(ci['97.5%']),
            n_obs=int(model._N),
            r_squared=float(model._r2) if hasattr(model, '_r2') else np.nan,
            coefficient_vector_json=coef_vec,
            sample_desc=sample_desc,
            fixed_effects=', '.join(fe_vars) if fe_vars else 'None',
            controls_desc=', '.join(controls) if controls else 'None',
            cluster_var=cluster_var,
            model_type=f'OLS_poly{polynomial}'
        )

        return True

    except Exception as e:
        print(f"Error in {spec_id}: {e}")
        return False

def run_rd_local(df, outcome, running_var, cutoff, controls, fe_vars, cluster_var,
                 spec_id, spec_tree_path, sample_desc, bandwidth=90, kernel='triangular'):
    """Run local polynomial RD with bandwidth restriction"""
    try:
        # Prepare data
        keep_vars = [outcome, running_var, cluster_var] + controls + fe_vars
        keep_vars = list(set([v for v in keep_vars if v in df.columns]))
        df_clean = df[keep_vars].dropna().copy()

        # Create running variable and restrict to bandwidth
        df_clean['T_centered'] = df_clean[running_var] - cutoff
        df_clean = df_clean[df_clean['T_centered'].abs() <= bandwidth].copy()

        if len(df_clean) < 100:
            return False

        df_clean['after'] = (df_clean['T_centered'] >= 0).astype(int)
        df_clean['after_T'] = df_clean['after'] * df_clean['T_centered']

        # Create kernel weights
        if kernel == 'triangular':
            df_clean['weight'] = (1 - df_clean['T_centered'].abs() / bandwidth).clip(lower=0.001)
        elif kernel == 'uniform':
            df_clean['weight'] = 1.0
        elif kernel == 'epanechnikov':
            df_clean['weight'] = (0.75 * (1 - (df_clean['T_centered'] / bandwidth) ** 2)).clip(lower=0.001)

        # Build formula
        rhs = ['after', 'T_centered', 'after_T'] + controls
        rhs_str = ' + '.join(rhs)

        if fe_vars:
            fe_str = ' + '.join(fe_vars)
            formula = f'{outcome} ~ {rhs_str} | {fe_str}'
        else:
            formula = f'{outcome} ~ {rhs_str}'

        # Run weighted regression
        model = pf.feols(formula, data=df_clean, vcov={'CRV1': cluster_var}, weights='weight')

        coef = model.coef()['after']
        se = model.se()['after']
        t = model.tstat()['after']
        pval = model.pvalue()['after']
        ci = model.confint().loc['after']

        coef_vec = get_coefficient_vector(model, 'after', controls, fe_vars)
        coef_vec['diagnostics']['bandwidth'] = bandwidth
        coef_vec['diagnostics']['kernel'] = kernel

        add_result(
            spec_id=spec_id,
            spec_tree_path=spec_tree_path,
            outcome_var=outcome,
            treatment_var='after',
            coefficient=float(coef),
            std_error=float(se),
            t_stat=float(t),
            p_value=float(pval),
            ci_lower=float(ci['2.5%']),
            ci_upper=float(ci['97.5%']),
            n_obs=int(model._N),
            r_squared=float(model._r2) if hasattr(model, '_r2') else np.nan,
            coefficient_vector_json=coef_vec,
            sample_desc=f'{sample_desc}, bw={bandwidth}',
            fixed_effects=', '.join(fe_vars) if fe_vars else 'None',
            controls_desc=', '.join(controls) if controls else 'None',
            cluster_var=cluster_var,
            model_type=f'RD_local_{kernel}'
        )

        return True

    except Exception as e:
        print(f"Error in {spec_id}: {e}")
        return False

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

print("Loading data...")

# Load datasets
df_pollution = pd.read_stata(DATA_PATH + 'pollution_1116.dta')
df_weather = pd.read_stata(DATA_PATH + 'weather_1116.dta')
df_stations = pd.read_stata(DATA_PATH + 'station_list.dta')

# Merge
df_pollution = df_pollution.rename(columns={'station_n': 'pm10_n'})
df_weather = df_weather.rename(columns={'station_n': 'pm10_n'})

df = df_pollution.merge(df_weather, on=['pm10_n', 'date'], how='left')
df = df.merge(df_stations, on=['pm10_n', 'code_city'], how='left')

# Convert dates
base_date = datetime(1960, 1, 1)
df['date_dt'] = df['date'].apply(lambda x: base_date + timedelta(days=int(x)) if pd.notna(x) else pd.NaT)

# Create variables
df['T'] = df['date'] - df['auto_date']
df['year'] = df['date_dt'].dt.year
df['month'] = df['date_dt'].dt.month
df['after'] = (df['T'] >= 0).astype(int)
df['l_pm10'] = np.log(df['pm10'].clip(lower=1))
df['ihs_pm10'] = np.arcsinh(df['pm10'])

print(f"Dataset shape: {df.shape}")

# Weather controls
weather_controls = ['wind_speed', 'rain', 'temp', 'rh']

# ============================================================================
# SPECIFICATION SEARCH
# ============================================================================

print("\nRunning specification search...")

# ----------------------------------------------------------------------------
# 1. BASELINE SPECIFICATIONS
# ----------------------------------------------------------------------------
print("\n1. Baseline specifications...")

run_rd_pyfixest(df, 'pm10', 'T', 0, weather_controls, ['pm10_n', 'month'], 'code_city',
                'baseline', 'methods/regression_discontinuity.md#baseline',
                'All stations, daily data', polynomial=1)

run_rd_pyfixest(df, 'pm10', 'T', 0, [], ['pm10_n', 'month'], 'code_city',
                'rd/controls/none', 'methods/regression_discontinuity.md#control-sets',
                'No weather controls', polynomial=1)

# ----------------------------------------------------------------------------
# 2. POLYNOMIAL VARIATIONS
# ----------------------------------------------------------------------------
print("\n2. Polynomial variations...")

for poly in [2, 3, 4]:
    run_rd_pyfixest(df, 'pm10', 'T', 0, weather_controls, ['pm10_n', 'month'], 'code_city',
                    f'rd/poly/order_{poly}', 'methods/regression_discontinuity.md#polynomial-order',
                    f'Polynomial order {poly}', polynomial=poly)

# ----------------------------------------------------------------------------
# 3. BANDWIDTH VARIATIONS
# ----------------------------------------------------------------------------
print("\n3. Bandwidth variations...")

for bw in [30, 60, 90, 120, 180, 365]:
    run_rd_local(df, 'pm10', 'T', 0, weather_controls, ['pm10_n', 'month'], 'code_city',
                 f'rd/bandwidth/fixed_{bw}d', 'methods/regression_discontinuity.md#bandwidth-selection',
                 f'Fixed bandwidth = {bw} days', bandwidth=bw)

# ----------------------------------------------------------------------------
# 4. KERNEL VARIATIONS
# ----------------------------------------------------------------------------
print("\n4. Kernel variations...")

for kernel in ['triangular', 'uniform', 'epanechnikov']:
    run_rd_local(df, 'pm10', 'T', 0, weather_controls, ['pm10_n', 'month'], 'code_city',
                 f'rd/kernel/{kernel}', 'methods/regression_discontinuity.md#kernel-function',
                 f'{kernel.capitalize()} kernel', bandwidth=90, kernel=kernel)

# ----------------------------------------------------------------------------
# 5. SAMPLE RESTRICTIONS - By Wave
# ----------------------------------------------------------------------------
print("\n5. Sample by wave...")

for phase in [1, 2]:
    df_phase = df[df['phase'] == phase]
    run_rd_pyfixest(df_phase, 'pm10', 'T', 0, weather_controls, ['pm10_n', 'month'], 'code_city',
                    f'robust/sample/wave_{phase}', 'robustness/sample_restrictions.md',
                    f'Wave {phase} only')

# Deadline cities
df_deadline = df[(df['auto_date'] == 19359) | (df['auto_date'] == 19724)]
run_rd_pyfixest(df_deadline, 'pm10', 'T', 0, weather_controls, ['pm10_n', 'month'], 'code_city',
                'robust/sample/deadline_cities', 'robustness/sample_restrictions.md',
                'Deadline cities only')

# Non-deadline cities
df_non_deadline = df[(df['auto_date'] != 19359) & (df['auto_date'] != 19724)]
run_rd_pyfixest(df_non_deadline, 'pm10', 'T', 0, weather_controls, ['pm10_n', 'month'], 'code_city',
                'robust/sample/non_deadline_cities', 'robustness/sample_restrictions.md',
                'Non-deadline cities')

# ----------------------------------------------------------------------------
# 6. TIME PERIOD RESTRICTIONS
# ----------------------------------------------------------------------------
print("\n6. Time period restrictions...")

for year in [2012, 2013, 2014, 2015, 2016]:
    df_year = df[df['year'] == year]
    if len(df_year.dropna(subset=['pm10'])) > 1000:
        run_rd_pyfixest(df_year, 'pm10', 'T', 0, weather_controls, ['pm10_n', 'month'], 'code_city',
                        f'robust/sample/year_{year}', 'robustness/sample_restrictions.md',
                        f'Year {year} only')

for drop_year in [2012, 2013, 2014, 2015, 2016]:
    df_drop = df[df['year'] != drop_year]
    run_rd_pyfixest(df_drop, 'pm10', 'T', 0, weather_controls, ['pm10_n', 'month'], 'code_city',
                    f'robust/sample/drop_year_{drop_year}', 'robustness/sample_restrictions.md',
                    f'Excluding year {drop_year}')

# ----------------------------------------------------------------------------
# 7. CONTROL VARIABLE VARIATIONS
# ----------------------------------------------------------------------------
print("\n7. Control variable variations...")

# Leave one out
for control in weather_controls:
    remaining = [c for c in weather_controls if c != control]
    run_rd_pyfixest(df, 'pm10', 'T', 0, remaining, ['pm10_n', 'month'], 'code_city',
                    f'robust/control/drop_{control}', 'robustness/leave_one_out.md',
                    f'Dropping {control}')

# Single covariate
for control in weather_controls:
    run_rd_pyfixest(df, 'pm10', 'T', 0, [control], ['pm10_n', 'month'], 'code_city',
                    f'robust/control/only_{control}', 'robustness/single_covariate.md',
                    f'Only {control}')

# Progressive addition
controls_progressive = []
for i, control in enumerate(weather_controls):
    controls_progressive.append(control)
    run_rd_pyfixest(df, 'pm10', 'T', 0, controls_progressive.copy(), ['pm10_n', 'month'], 'code_city',
                    f'robust/control/add_{i+1}_{control}', 'robustness/control_progression.md',
                    f'Adding {control}')

# ----------------------------------------------------------------------------
# 8. ALTERNATIVE OUTCOMES
# ----------------------------------------------------------------------------
print("\n8. Alternative outcomes...")

run_rd_pyfixest(df, 'l_pm10', 'T', 0, weather_controls, ['pm10_n', 'month'], 'code_city',
                'robust/outcome/log_pm10', 'robustness/functional_form.md', 'Log PM10')

run_rd_pyfixest(df, 'ihs_pm10', 'T', 0, weather_controls, ['pm10_n', 'month'], 'code_city',
                'robust/funcform/ihs_pm10', 'robustness/functional_form.md', 'IHS PM10')

if df['so2'].notna().sum() > 10000:
    run_rd_pyfixest(df, 'so2', 'T', 0, weather_controls, ['pm10_n', 'month'], 'code_city',
                    'robust/outcome/so2', 'robustness/measurement.md', 'SO2 as outcome')

if df['no2'].notna().sum() > 10000:
    run_rd_pyfixest(df, 'no2', 'T', 0, weather_controls, ['pm10_n', 'month'], 'code_city',
                    'robust/outcome/no2', 'robustness/measurement.md', 'NO2 as outcome')

# ----------------------------------------------------------------------------
# 9. FIXED EFFECTS VARIATIONS
# ----------------------------------------------------------------------------
print("\n9. Fixed effects variations...")

run_rd_pyfixest(df, 'pm10', 'T', 0, weather_controls, ['pm10_n'], 'code_city',
                'robust/estimation/station_fe_only', 'robustness/model_specification.md',
                'Station FE only')

run_rd_pyfixest(df, 'pm10', 'T', 0, weather_controls, ['month'], 'code_city',
                'robust/estimation/month_fe_only', 'robustness/model_specification.md',
                'Month FE only')

run_rd_pyfixest(df, 'pm10', 'T', 0, weather_controls, [], 'code_city',
                'robust/estimation/no_fe', 'robustness/model_specification.md',
                'No fixed effects')

run_rd_pyfixest(df, 'pm10', 'T', 0, weather_controls, ['code_city', 'month'], 'code_city',
                'robust/estimation/city_month_fe', 'robustness/model_specification.md',
                'City and month FE')

# Year-month FE
df['year_month'] = df['year'].astype(str) + '_' + df['month'].astype(str)
run_rd_pyfixest(df, 'pm10', 'T', 0, weather_controls, ['pm10_n', 'year_month'], 'code_city',
                'robust/estimation/station_yearmonth_fe', 'robustness/model_specification.md',
                'Station and year-month FE')

# ----------------------------------------------------------------------------
# 10. CLUSTERING VARIATIONS
# ----------------------------------------------------------------------------
print("\n10. Clustering variations...")

run_rd_pyfixest(df, 'pm10', 'T', 0, weather_controls, ['pm10_n', 'month'], 'pm10_n',
                'robust/cluster/station', 'robustness/clustering_variations.md',
                'Clustered at station level')

# ----------------------------------------------------------------------------
# 11. DONUT HOLE SPECIFICATIONS
# ----------------------------------------------------------------------------
print("\n11. Donut hole specifications...")

for donut in [7, 14, 30]:
    df_donut = df[df['T'].abs() > donut]
    run_rd_pyfixest(df_donut, 'pm10', 'T', 0, weather_controls, ['pm10_n', 'month'], 'code_city',
                    f'rd/donut/exclude_{donut}d', 'methods/regression_discontinuity.md#donut-hole-specifications',
                    f'Excluding {donut} days around cutoff')

# ----------------------------------------------------------------------------
# 12. PLACEBO TESTS
# ----------------------------------------------------------------------------
print("\n12. Placebo tests...")

for placebo_cutoff in [-180, -90, -60, 90, 180]:
    run_rd_pyfixest(df, 'pm10', 'T', placebo_cutoff, weather_controls, ['pm10_n', 'month'], 'code_city',
                    f'rd/placebo/cutoff_{placebo_cutoff}', 'methods/regression_discontinuity.md#placebo-cutoff-tests',
                    f'Placebo cutoff at T={placebo_cutoff}')

# Weather as placebo
for weather_var in weather_controls:
    other_controls = [c for c in weather_controls if c != weather_var]
    run_rd_pyfixest(df, weather_var, 'T', 0, other_controls, ['pm10_n', 'month'], 'code_city',
                    f'rd/placebo/outcome_{weather_var}', 'robustness/placebo_tests.md',
                    f'Placebo outcome: {weather_var}')

# ----------------------------------------------------------------------------
# 13. OUTLIER TREATMENT
# ----------------------------------------------------------------------------
print("\n13. Outlier treatment...")

for pct in [1, 5, 10]:
    df_wins = df.copy()
    lower = df_wins['pm10'].quantile(pct/100)
    upper = df_wins['pm10'].quantile(1 - pct/100)
    df_wins['pm10'] = df_wins['pm10'].clip(lower=lower, upper=upper)
    run_rd_pyfixest(df_wins, 'pm10', 'T', 0, weather_controls, ['pm10_n', 'month'], 'code_city',
                    f'robust/sample/winsorize_{pct}pct', 'robustness/sample_restrictions.md',
                    f'PM10 winsorized at {pct}%')

df_trim = df[(df['pm10'] > df['pm10'].quantile(0.01)) & (df['pm10'] < df['pm10'].quantile(0.99))]
run_rd_pyfixest(df_trim, 'pm10', 'T', 0, weather_controls, ['pm10_n', 'month'], 'code_city',
                'robust/sample/trim_1pct', 'robustness/sample_restrictions.md', 'Trimmed 1% tails')

# ----------------------------------------------------------------------------
# 14. WINDOW RESTRICTIONS
# ----------------------------------------------------------------------------
print("\n14. Window restrictions...")

for window in [30, 60, 90, 120, 180, 365]:
    df_window = df[df['T'].abs() <= window]
    run_rd_pyfixest(df_window, 'pm10', 'T', 0, weather_controls, ['pm10_n', 'month'], 'code_city',
                    f'rd/sample/window_{window}d', 'robustness/sample_restrictions.md',
                    f'Window = +/- {window} days')

# ----------------------------------------------------------------------------
# 15. HETEROGENEITY ANALYSIS
# ----------------------------------------------------------------------------
print("\n15. Heterogeneity analysis...")

for phase in [1, 2]:
    df_phase = df[df['phase'] == phase]
    run_rd_local(df_phase, 'pm10', 'T', 0, weather_controls, ['pm10_n', 'month'], 'code_city',
                 f'robust/heterogeneity/wave_{phase}_local', 'robustness/heterogeneity.md',
                 f'Wave {phase} local RD', bandwidth=90)

# ============================================================================
# SAVE RESULTS
# ============================================================================

print(f"\nTotal specifications run: {len(results)}")

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
output_file = OUTPUT_PATH + 'specification_results.csv'
results_df.to_csv(output_file, index=False)
print(f"Results saved to: {output_file}")

# Print summary
print("\n" + "="*60)
print("SPECIFICATION SEARCH SUMMARY")
print("="*60)
print(f"Total specifications: {len(results_df)}")

# Handle NaN values in coefficient
valid_coef = results_df['coefficient'].dropna()
valid_pval = results_df['p_value'].dropna()

print(f"Valid coefficients: {len(valid_coef)}")
if len(valid_coef) > 0:
    print(f"Positive coefficients: {(valid_coef > 0).sum()} ({100*(valid_coef > 0).mean():.1f}%)")
    print(f"Negative coefficients: {(valid_coef < 0).sum()} ({100*(valid_coef < 0).mean():.1f}%)")
    print(f"Median coefficient: {valid_coef.median():.2f}")
    print(f"Mean coefficient: {valid_coef.mean():.2f}")
    print(f"Range: [{valid_coef.min():.2f}, {valid_coef.max():.2f}]")

if len(valid_pval) > 0:
    print(f"Significant at 5%: {(valid_pval < 0.05).sum()} ({100*(valid_pval < 0.05).mean():.1f}%)")
    print(f"Significant at 1%: {(valid_pval < 0.01).sum()} ({100*(valid_pval < 0.01).mean():.1f}%)")
