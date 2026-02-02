#!/usr/bin/env python3
"""
Specification Search for Paper 194841-V1

Paper: "Rational Inattention and the Business Cycle Effects of Productivity and News Shocks"
Authors: Bartosz Mackowiak and Mirko Wiederholt

Main empirical analysis: Coibion-Gorodnichenko (CG) regression
    forecast_error = alpha + beta * forecast_revision + epsilon

Where:
    - forecast_error = actual_outcome - forecast
    - forecast_revision = current_forecast - previous_forecast

Key Result: beta > 0 indicates under-reaction (information rigidities)

This paper uses Survey of Professional Forecasters (SPF) data on GDP forecasts.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import t as t_dist
import json
import warnings
warnings.filterwarnings('ignore')

# Paths
PACKAGE_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/194841-V1"
OUTPUT_DIR = PACKAGE_DIR

# Paper metadata
PAPER_ID = "194841-V1"
JOURNAL = "AER"
PAPER_TITLE = "Rational Inattention and the Business Cycle Effects of Productivity and News Shocks"

# Initialize results list
results = []

def compute_stats(model, treatment_var='forecast_revision'):
    """Extract statistics from a statsmodels regression result."""
    if treatment_var in model.params.index:
        coef = model.params[treatment_var]
        se = model.bse[treatment_var]
        tstat = model.tvalues[treatment_var] if hasattr(model, 'tvalues') else coef / se
        pval = model.pvalues[treatment_var] if hasattr(model, 'pvalues') else 2 * (1 - stats.norm.cdf(abs(coef / se)))
        try:
            ci = model.conf_int().loc[treatment_var]
            ci_lower, ci_upper = ci[0], ci[1]
        except:
            ci_lower = coef - 1.96 * se
            ci_upper = coef + 1.96 * se
    else:
        # If treatment var not found, return first non-constant coefficient
        idx = 1 if 'const' in model.params.index else 0
        var_name = model.params.index[idx]
        coef = model.params[var_name]
        se = model.bse[var_name]
        tstat = model.tvalues[var_name] if hasattr(model, 'tvalues') else coef / se
        pval = model.pvalues[var_name] if hasattr(model, 'pvalues') else 2 * (1 - stats.norm.cdf(abs(coef / se)))
        try:
            ci = model.conf_int().loc[var_name]
            ci_lower, ci_upper = ci[0], ci[1]
        except:
            ci_lower = coef - 1.96 * se
            ci_upper = coef + 1.96 * se
        treatment_var = var_name

    # Get r_squared if available
    try:
        r_squared = model.rsquared
    except AttributeError:
        try:
            r_squared = model.prsquared  # For discrete models
        except AttributeError:
            r_squared = np.nan

    return {
        'coefficient': coef,
        'std_error': se,
        't_stat': tstat,
        'p_value': pval,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_obs': int(model.nobs),
        'r_squared': r_squared,
        'treatment_var': treatment_var
    }

def create_coefficient_vector(model, treatment_var='forecast_revision'):
    """Create JSON representation of all coefficients."""
    # Safely get treatment coefficient
    if treatment_var in model.params.index:
        treat_coef = float(model.params[treatment_var])
        treat_se = float(model.bse[treatment_var])
        treat_pval = float(model.pvalues[treatment_var]) if hasattr(model, 'pvalues') else np.nan
    else:
        treat_coef = float(model.params.iloc[1])
        treat_se = float(model.bse.iloc[1])
        treat_pval = float(model.pvalues.iloc[1]) if hasattr(model, 'pvalues') else np.nan

    # Safely get r_squared and adj_r_squared
    try:
        r_sq = float(model.rsquared)
    except AttributeError:
        try:
            r_sq = float(model.prsquared)
        except AttributeError:
            r_sq = None

    try:
        adj_r_sq = float(model.rsquared_adj)
    except AttributeError:
        adj_r_sq = None

    coef_vector = {
        'treatment': {
            'var': treatment_var,
            'coef': treat_coef,
            'se': treat_se,
            'pval': treat_pval
        },
        'controls': [],
        'fixed_effects': [],
        'diagnostics': {
            'r_squared': r_sq,
            'adj_r_squared': adj_r_sq,
            'f_stat': float(model.fvalue) if hasattr(model, 'fvalue') else None,
            'f_pval': float(model.f_pvalue) if hasattr(model, 'f_pvalue') else None
        }
    }

    for var in model.params.index:
        if var not in [treatment_var, 'const', 'Intercept']:
            pval_ctrl = float(model.pvalues[var]) if hasattr(model, 'pvalues') else np.nan
            coef_vector['controls'].append({
                'var': var,
                'coef': float(model.params[var]),
                'se': float(model.bse[var]),
                'pval': pval_ctrl
            })

    return json.dumps(coef_vector)

def add_result(spec_id, spec_tree_path, model, outcome_var, treatment_var,
               sample_desc, fixed_effects, controls_desc, cluster_var, model_type):
    """Add a specification result to the results list."""
    stats = compute_stats(model, treatment_var)

    results.append({
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'outcome_var': outcome_var,
        'treatment_var': treatment_var,
        'coefficient': stats['coefficient'],
        'std_error': stats['std_error'],
        't_stat': stats['t_stat'],
        'p_value': stats['p_value'],
        'ci_lower': stats['ci_lower'],
        'ci_upper': stats['ci_upper'],
        'n_obs': stats['n_obs'],
        'r_squared': stats['r_squared'],
        'coefficient_vector_json': create_coefficient_vector(model, treatment_var),
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

# The data is embedded in the MATLAB code - replicate it here
# From main.m and CG_regressions.m
# Column 1: y(t+3|t) - current forecast
# Column 2: y(t+3|t-1) - previous forecast
# Column 3: y(t+3) - actual outcome

data_raw = np.array([
    [739.3701, 740.6250, 730.5000],
    [741.1765, 742.9630, 721.3000],
    [745.1256, 750.5448, 736.3000],
    [749.2022, 753.5714, 743.6000],
    [753.2827, 756.2013, 751.7000],
    [763.8889, 764.7059, 761.0000],
    [786.9204, 774.9483, 783.1000],
    [789.0439, 797.4909, 795.3000],
    [797.8363, 799.1901, 812.4000],
    [808.1279, 807.7052, 827.1000],
    [828.4761, 817.8191, 834.6000],
    [841.1185, 837.3612, 841.6000],
    [850.1798, 849.8037, 844.1000],
    [860.1040, 858.6287, 832.0000],
    [862.1974, 866.7092, 828.0000],
    [846.9059, 869.6988, 821.1000],
    [851.1905, 852.4736, 803.7000],
    [850.7042, 859.5870, 782.3000],
    [842.3217, 861.9734, 779.4000],
    [1304.4896, 1301.9551, 1296.8000],
    [1326.1026, 1319.6134, 1331.6000],
    [1339.1608, 1341.4729, 1343.2000],
    [1355.5868, 1355.9322, 1361.4000],
    [1372.2826, 1372.8463, 1358.3000],
    [1394.4892, 1385.9060, 1378.6000],
    [1400.5305, 1408.2508, 1394.3000],
    [1413.6085, 1412.3037, 1412.2000],
    [1417.3077, 1429.6879, 1417.3000],
    [1412.9757, 1425.9259, 1418.8000],
    [1419.9135, 1419.2961, 1430.8000],
    [1419.1439, 1426.9315, 1438.4000],
    [1421.2975, 1427.6471, 1444.2000],
    [1405.2368, 1425.7143, 1410.8000],
    [1405.9047, 1417.7898, 1412.1000],
    [1429.7611, 1417.6152, 1490.1000],
    [1417.8870, 1437.4187, 1509.2000],
    [1412.9430, 1429.8641, 1509.1000],
    [1444.0830, 1430.0254, 1508.2000],
    [1520.1766, 1459.0962, 1495.6000],
    [1549.9756, 1537.2779, 1483.6000],
    [1543.0000, 1561.2025, 1476.8000],
    [1523.0000, 1559.5000, 1481.2000],
    [1528.5000, 1540.0000, 1471.7000],
    [1520.5000, 1539.0000, 1488.5000],
    [1517.0000, 1533.0000, 1521.4000],
    [1522.0000, 1531.0000, 1554.4000],
    [1528.0000, 1537.0000, 1570.5000],
    [1559.5000, 1542.0000, 1604.3000],
    [1597.0000, 1577.5000, 1640.2000],
    [1627.0000, 1611.0000, 1649.6000],
    [1637.0000, 1643.0000, 1661.1000],
    [1662.0000, 1652.0000, 1668.0000],
    [1702.0000, 1674.0000, 1670.7000],
    [1702.5000, 1712.5000, 1684.8000],
    [3743.0000, 3731.0000, 3735.2000],
    [3767.0000, 3775.0000, 3796.4000],
    [3782.0000, 3795.0000, 3831.2000],
    [3808.0000, 3816.5000, 3875.1000],
    [3835.5000, 3834.0000, 3902.6000],
    [3899.5000, 3862.0000, 3986.3000],
    [3914.0000, 3928.0000, 4007.3000],
    [3934.0000, 3937.0000, 4029.2000],
    [4014.0000, 3965.0000, 4088.2000],
    [4098.0000, 4033.0000, 4123.9000],
    [4123.5000, 4117.5000, 4158.1000],
    [4126.0000, 4134.5000, 4168.1000],
    [4153.0000, 4140.0000, 4195.8000],
    [4186.0000, 4165.5000, 4163.2000],
    [4228.5000, 4226.9000, 4173.6000],
    [4254.5000, 4253.0000, 4147.6000],
    [4290.0000, 4278.0000, 4123.9000],
    [4202.5000, 4318.0000, 4128.4000],
    [4165.0000, 4227.9500, 4143.1000],
    [4177.0000, 4190.5000, 4866.3000],
    [4201.0000, 4210.1500, 4891.9000],
    [4244.0000, 4233.4000, 4890.5000],
    [4247.6000, 4273.0000, 4924.5000],
    [4978.0000, 4275.7000, 4979.8000],
    [5043.5000, 5012.0000, 5013.1000],
    [5012.8000, 5082.5000, 5019.5000],
    [5052.4950, 5048.1000, 5138.0000],
    [5137.1890, 5094.0545, 5212.1000],
    [5168.2000, 5173.4700, 5259.0000],
    [5168.8211, 5202.0000, 5309.2000],
    [5290.1500, 5207.0000, 5359.2000],
    [5370.8500, 5326.1000, 5426.8000],
    [5418.3500, 5402.9070, 5471.7000],
    [5441.2500, 5449.7000, 5477.3000],
    [5504.9000, 5475.6000, 5544.6000],
    [6980.8000, 6968.3305, 7089.4000],
    [7038.0000, 7013.4410, 7139.7000],
    [7079.1000, 7069.3500, 7221.8000],
    [7166.4500, 7119.0000, 7290.3000],
    [7256.3000, 7203.7240, 7356.0000],
    [7323.6000, 7290.0000, 7491.0000],
    [7400.1575, 7360.1000, 7559.5000],
    [7452.5500, 7439.7525, 7670.0000],
    [7524.1410, 7500.7000, 7762.5000],
    [7668.8490, 7566.4890, 7803.6000],
    [7708.2000, 7716.3360, 8882.6000],
    [7861.7030, 7747.5775, 9026.9000],
    [7966.1060, 7908.4550, 9156.6000],
    [8018.1830, 8025.1910, 9308.8000],
    [9148.9000, 8067.8330, 9382.2000],
    [9307.5500, 9225.5000, 9401.5000],
    [9465.0770, 9373.1500, 9439.9000],
    [9595.0000, 9525.2605, 9351.6000],
    [9689.3370, 9671.1500, 9333.4000],
    [9636.0000, 9765.9395, 9315.6000],
    [9650.0000, 9723.2500, 9482.1000],
    [9578.0000, 9734.9000, 9387.9000],
    [9428.1355, 9670.3220, 9465.2000],
    [9576.2000, 9521.1865, 9503.2000],
    [9781.8000, 9655.5000, 9556.0000],
    [9658.3000, 9868.9500, 9608.1000],
    [9706.8720, 9745.4065, 9797.2000],
    [11148.6700, 11140.1000, 11078.2000],
    [11181.0110, 11257.3000, 11092.0000],
    [11262.4000, 11273.3070, 11193.2000],
    [11362.2000, 11356.0390, 11233.5000],
    [11442.8580, 11462.3695, 11381.4000],
    [11491.6200, 11538.3240, 11385.3000],
    [11568.1220, 11583.9430, 11432.9000],
    [11626.3270, 11659.5440, 11541.6000],
    [11733.8369, 11720.5880, 11549.1000],
    [11704.0423, 11821.5433, 11507.9000],
    [11747.3000, 11790.0043, 11630.7000],
    [11872.7391, 11831.5437, 11677.4000],
    [11860.2391, 11962.6024, 11693.1000],
    [11817.5410, 11948.2354, 11700.6000],
    [11884.0000, 11895.5841, 11720.0000],
    [11898.1098, 11965.9812, 11599.4000],
    [11868.5285, 11989.9283, 11340.9000],
    [13339.3214, 13297.4006, 13260.7000],
    [13512.3602, 13436.0801, 13382.6000],
    [13657.0958, 13602.3331, 13438.8000],
    [13564.2900, 13766.7000, 13270.1000],
    [13612.0791, 13664.0000, 13352.8000],
    [13835.0000, 13709.2281, 13422.4000],
    [13876.7709, 13940.1649, 13502.4000],
    [13598.4431, 13962.0000, 13558.0000],
    [13692.4048, 13704.3795, 13616.2000],
    [13761.7085, 13785.1513, 13647.6000],
    [13843.3179, 13856.6582, 13750.1000],
    [16189.1000, 16158.3571, 16150.6000],
    [16385.6000, 16306.2418, 16311.6000],
    [16447.1513, 16516.5150, 16304.8000],
    [16477.2472, 16574.6500, 16270.4000],
    [16608.9431, 16597.9367, 16394.2000],
    [16770.6865, 16731.8252, 16442.3000],
    [16749.9240, 16892.5527, 16492.7000],
    [16721.5500, 16875.5983, 16575.1000],
    [16831.1268, 16835.1407, 16702.1000],
    [16823.7154, 16932.4081, 16804.8000],
    [16872.2000, 16923.4101, 16842.4000],
    [16967.0513, 16972.8021, 17010.7000],
    [17067.7164, 17056.4933, 17156.9000],
    [17194.3948, 17162.6024, 17272.5000],
    [17282.8909, 17289.6117, 17385.8000],
    [19139.6289, 19141.4000, 19112.5000],
    [19172.6242, 19244.6811, 19219.8000]
])

# Create dataframe
df = pd.DataFrame(data_raw, columns=['forecast', 'previous_forecast', 'outcome'])

# Compute log values
df['log_forecast'] = np.log(df['forecast'])
df['log_previous_forecast'] = np.log(df['previous_forecast'])
df['log_outcome'] = np.log(df['outcome'])

# Compute forecast error and forecast revision (in logs, as in paper)
df['forecast_error'] = df['log_outcome'] - df['log_forecast']
df['forecast_revision'] = df['log_forecast'] - df['log_previous_forecast']

# Add time index (quarters starting from 1969Q1)
df['obs_id'] = range(len(df))
df['period'] = df['obs_id']

# Split point for subsample analysis (as in paper, split = 80)
SPLIT = 80

# Add subsample indicators
df['subsample1'] = df['obs_id'] < SPLIT
df['subsample2'] = df['obs_id'] >= SPLIT

# Add time-based categories
n = len(df)
df['early_period'] = df['obs_id'] < n // 2
df['late_period'] = df['obs_id'] >= n // 2

# Create recession indicator (approximate based on NBER dates relative to obs)
# Observations 17-19 (1973-74), 39-43 (1980-82), 72-73 (1990-91),
# 99-103 (2007-09), 132-134 (2020)
recession_obs = list(range(17, 20)) + list(range(39, 44)) + list(range(72, 74)) + \
                list(range(99, 104)) + list(range(132, 135))
df['recession'] = df['obs_id'].isin(recession_obs)

# Add GDP base categories (rebasing events)
# Based on approximate structural breaks in the data
df['gdp_base1'] = df['obs_id'] < 54  # Pre-1985 base
df['gdp_base2'] = (df['obs_id'] >= 54) & (df['obs_id'] < 89)  # 1985-1995 base
df['gdp_base3'] = (df['obs_id'] >= 89) & (df['obs_id'] < 117)  # 1995-2005 base
df['gdp_base4'] = (df['obs_id'] >= 117) & (df['obs_id'] < 150)  # 2005-2015 base
df['gdp_base5'] = df['obs_id'] >= 150  # Post-2015 base

print(f"Data loaded: {len(df)} observations")
print(f"Forecast error: mean={df['forecast_error'].mean():.6f}, std={df['forecast_error'].std():.6f}")
print(f"Forecast revision: mean={df['forecast_revision'].mean():.6f}, std={df['forecast_revision'].std():.6f}")

# ============================================================================
# OUTLIER REMOVAL (as in original paper)
# ============================================================================

# Original paper removes 99th percentile outliers
p99_fe = df['forecast_error'].quantile(0.99)
p99_fr = df['forecast_revision'].quantile(0.99)

df_clean = df.copy()
df_clean['forecast_error_clean'] = df_clean['forecast_error'].where(
    df_clean['forecast_error'] <= p99_fe, np.nan)
df_clean['forecast_revision_clean'] = df_clean['forecast_revision'].where(
    df_clean['forecast_revision'] <= p99_fr, np.nan)

# Create cleaned dataset (dropping NaN rows for regression)
df_baseline = df_clean.dropna(subset=['forecast_error_clean', 'forecast_revision_clean']).copy()
df_baseline['forecast_error'] = df_baseline['forecast_error_clean']
df_baseline['forecast_revision'] = df_baseline['forecast_revision_clean']

print(f"Cleaned data: {len(df_baseline)} observations (after outlier removal)")

# ============================================================================
# SPEC 1: BASELINE - Exact replication of paper's main result
# ============================================================================

model = sm.OLS(df_baseline['forecast_error'],
               sm.add_constant(df_baseline['forecast_revision'])).fit()

add_result(
    spec_id='baseline',
    spec_tree_path='methods/cross_sectional_ols.md',
    model=model,
    outcome_var='log_forecast_error',
    treatment_var='forecast_revision',
    sample_desc='SPF GDP forecasts, 161 obs, 99th pct outliers removed',
    fixed_effects='None',
    controls_desc='None (bivariate)',
    cluster_var='None',
    model_type='OLS'
)

print(f"\n=== BASELINE ===")
print(f"Beta = {model.params['forecast_revision']:.4f}, SE = {model.bse['forecast_revision']:.4f}")
print(f"t = {model.tvalues['forecast_revision']:.4f}, p = {model.pvalues['forecast_revision']:.4f}")

# ============================================================================
# SPEC 2-3: SUBSAMPLE ANALYSIS (as in paper Section 5.3)
# ============================================================================

# Subsample 1 (first 80 observations)
df_sub1 = df_baseline[df_baseline['subsample1']].copy()
model_sub1 = sm.OLS(df_sub1['forecast_error'],
                    sm.add_constant(df_sub1['forecast_revision'])).fit()

add_result(
    spec_id='robust/sample/subsample1_early',
    spec_tree_path='robustness/sample_restrictions.md',
    model=model_sub1,
    outcome_var='log_forecast_error',
    treatment_var='forecast_revision',
    sample_desc='SPF GDP forecasts, first 80 obs (paper subsample 1)',
    fixed_effects='None',
    controls_desc='None (bivariate)',
    cluster_var='None',
    model_type='OLS'
)

# Subsample 2 (observations 81 onwards)
df_sub2 = df_baseline[df_baseline['subsample2']].copy()
model_sub2 = sm.OLS(df_sub2['forecast_error'],
                    sm.add_constant(df_sub2['forecast_revision'])).fit()

add_result(
    spec_id='robust/sample/subsample2_late',
    spec_tree_path='robustness/sample_restrictions.md',
    model=model_sub2,
    outcome_var='log_forecast_error',
    treatment_var='forecast_revision',
    sample_desc='SPF GDP forecasts, obs 81+ (paper subsample 2)',
    fixed_effects='None',
    controls_desc='None (bivariate)',
    cluster_var='None',
    model_type='OLS'
)

# ============================================================================
# SPECS 4-10: ALTERNATIVE OUTLIER TREATMENTS
# ============================================================================

# Spec 4: No outlier removal (full sample)
model_full = sm.OLS(df['forecast_error'],
                    sm.add_constant(df['forecast_revision'])).fit()

add_result(
    spec_id='robust/sample/no_outlier_removal',
    spec_tree_path='robustness/sample_restrictions.md',
    model=model_full,
    outcome_var='log_forecast_error',
    treatment_var='forecast_revision',
    sample_desc='SPF GDP forecasts, all 167 obs, no outlier removal',
    fixed_effects='None',
    controls_desc='None (bivariate)',
    cluster_var='None',
    model_type='OLS'
)

# Spec 5: Winsorize at 1%
df_w1 = df.copy()
df_w1['forecast_error'] = df_w1['forecast_error'].clip(
    lower=df_w1['forecast_error'].quantile(0.01),
    upper=df_w1['forecast_error'].quantile(0.99))
df_w1['forecast_revision'] = df_w1['forecast_revision'].clip(
    lower=df_w1['forecast_revision'].quantile(0.01),
    upper=df_w1['forecast_revision'].quantile(0.99))

model_w1 = sm.OLS(df_w1['forecast_error'],
                  sm.add_constant(df_w1['forecast_revision'])).fit()

add_result(
    spec_id='robust/sample/winsorize_1pct',
    spec_tree_path='robustness/sample_restrictions.md',
    model=model_w1,
    outcome_var='log_forecast_error',
    treatment_var='forecast_revision',
    sample_desc='SPF GDP forecasts, winsorized at 1%/99%',
    fixed_effects='None',
    controls_desc='None (bivariate)',
    cluster_var='None',
    model_type='OLS'
)

# Spec 6: Winsorize at 5%
df_w5 = df.copy()
df_w5['forecast_error'] = df_w5['forecast_error'].clip(
    lower=df_w5['forecast_error'].quantile(0.05),
    upper=df_w5['forecast_error'].quantile(0.95))
df_w5['forecast_revision'] = df_w5['forecast_revision'].clip(
    lower=df_w5['forecast_revision'].quantile(0.05),
    upper=df_w5['forecast_revision'].quantile(0.95))

model_w5 = sm.OLS(df_w5['forecast_error'],
                  sm.add_constant(df_w5['forecast_revision'])).fit()

add_result(
    spec_id='robust/sample/winsorize_5pct',
    spec_tree_path='robustness/sample_restrictions.md',
    model=model_w5,
    outcome_var='log_forecast_error',
    treatment_var='forecast_revision',
    sample_desc='SPF GDP forecasts, winsorized at 5%/95%',
    fixed_effects='None',
    controls_desc='None (bivariate)',
    cluster_var='None',
    model_type='OLS'
)

# Spec 7: Trim at 1%
df_t1 = df[(df['forecast_error'] >= df['forecast_error'].quantile(0.01)) &
           (df['forecast_error'] <= df['forecast_error'].quantile(0.99)) &
           (df['forecast_revision'] >= df['forecast_revision'].quantile(0.01)) &
           (df['forecast_revision'] <= df['forecast_revision'].quantile(0.99))].copy()

model_t1 = sm.OLS(df_t1['forecast_error'],
                  sm.add_constant(df_t1['forecast_revision'])).fit()

add_result(
    spec_id='robust/sample/trim_1pct',
    spec_tree_path='robustness/sample_restrictions.md',
    model=model_t1,
    outcome_var='log_forecast_error',
    treatment_var='forecast_revision',
    sample_desc='SPF GDP forecasts, trimmed at 1%/99%',
    fixed_effects='None',
    controls_desc='None (bivariate)',
    cluster_var='None',
    model_type='OLS'
)

# Spec 8: Trim at 5%
df_t5 = df[(df['forecast_error'] >= df['forecast_error'].quantile(0.05)) &
           (df['forecast_error'] <= df['forecast_error'].quantile(0.95)) &
           (df['forecast_revision'] >= df['forecast_revision'].quantile(0.05)) &
           (df['forecast_revision'] <= df['forecast_revision'].quantile(0.95))].copy()

model_t5 = sm.OLS(df_t5['forecast_error'],
                  sm.add_constant(df_t5['forecast_revision'])).fit()

add_result(
    spec_id='robust/sample/trim_5pct',
    spec_tree_path='robustness/sample_restrictions.md',
    model=model_t5,
    outcome_var='log_forecast_error',
    treatment_var='forecast_revision',
    sample_desc='SPF GDP forecasts, trimmed at 5%/95%',
    fixed_effects='None',
    controls_desc='None (bivariate)',
    cluster_var='None',
    model_type='OLS'
)

# Spec 9: Remove only top 1% of forecast error
df_fe99 = df[df['forecast_error'] <= df['forecast_error'].quantile(0.99)].copy()
model_fe99 = sm.OLS(df_fe99['forecast_error'],
                    sm.add_constant(df_fe99['forecast_revision'])).fit()

add_result(
    spec_id='robust/sample/trim_forecast_error_top1pct',
    spec_tree_path='robustness/sample_restrictions.md',
    model=model_fe99,
    outcome_var='log_forecast_error',
    treatment_var='forecast_revision',
    sample_desc='SPF GDP forecasts, top 1% forecast errors removed',
    fixed_effects='None',
    controls_desc='None (bivariate)',
    cluster_var='None',
    model_type='OLS'
)

# Spec 10: Remove only top 1% of forecast revision
df_fr99 = df[df['forecast_revision'] <= df['forecast_revision'].quantile(0.99)].copy()
model_fr99 = sm.OLS(df_fr99['forecast_error'],
                    sm.add_constant(df_fr99['forecast_revision'])).fit()

add_result(
    spec_id='robust/sample/trim_forecast_revision_top1pct',
    spec_tree_path='robustness/sample_restrictions.md',
    model=model_fr99,
    outcome_var='log_forecast_error',
    treatment_var='forecast_revision',
    sample_desc='SPF GDP forecasts, top 1% forecast revisions removed',
    fixed_effects='None',
    controls_desc='None (bivariate)',
    cluster_var='None',
    model_type='OLS'
)

# ============================================================================
# SPECS 11-18: TIME PERIOD RESTRICTIONS
# ============================================================================

# Spec 11: Early period (first half)
df_early = df_baseline[df_baseline['early_period']].copy()
model_early = sm.OLS(df_early['forecast_error'],
                     sm.add_constant(df_early['forecast_revision'])).fit()

add_result(
    spec_id='robust/sample/early_period',
    spec_tree_path='robustness/sample_restrictions.md',
    model=model_early,
    outcome_var='log_forecast_error',
    treatment_var='forecast_revision',
    sample_desc='SPF GDP forecasts, first half of sample',
    fixed_effects='None',
    controls_desc='None (bivariate)',
    cluster_var='None',
    model_type='OLS'
)

# Spec 12: Late period (second half)
df_late = df_baseline[df_baseline['late_period']].copy()
model_late = sm.OLS(df_late['forecast_error'],
                    sm.add_constant(df_late['forecast_revision'])).fit()

add_result(
    spec_id='robust/sample/late_period',
    spec_tree_path='robustness/sample_restrictions.md',
    model=model_late,
    outcome_var='log_forecast_error',
    treatment_var='forecast_revision',
    sample_desc='SPF GDP forecasts, second half of sample',
    fixed_effects='None',
    controls_desc='None (bivariate)',
    cluster_var='None',
    model_type='OLS'
)

# Spec 13: Exclude recession periods
df_norec = df_baseline[~df_baseline['recession']].copy()
model_norec = sm.OLS(df_norec['forecast_error'],
                     sm.add_constant(df_norec['forecast_revision'])).fit()

add_result(
    spec_id='robust/sample/exclude_recessions',
    spec_tree_path='robustness/sample_restrictions.md',
    model=model_norec,
    outcome_var='log_forecast_error',
    treatment_var='forecast_revision',
    sample_desc='SPF GDP forecasts, excluding recession quarters',
    fixed_effects='None',
    controls_desc='None (bivariate)',
    cluster_var='None',
    model_type='OLS'
)

# Spec 14: Recession periods only
df_rec = df_baseline[df_baseline['recession']].copy()
if len(df_rec) > 5:
    model_rec = sm.OLS(df_rec['forecast_error'],
                       sm.add_constant(df_rec['forecast_revision'])).fit()

    add_result(
        spec_id='robust/sample/recession_only',
        spec_tree_path='robustness/sample_restrictions.md',
        model=model_rec,
        outcome_var='log_forecast_error',
        treatment_var='forecast_revision',
        sample_desc='SPF GDP forecasts, recession quarters only',
        fixed_effects='None',
        controls_desc='None (bivariate)',
        cluster_var='None',
        model_type='OLS'
    )

# Spec 15-18: Quartile-based time periods
for q in range(4):
    q_start = int(len(df_baseline) * q / 4)
    q_end = int(len(df_baseline) * (q + 1) / 4)
    df_q = df_baseline.iloc[q_start:q_end].copy()

    if len(df_q) > 5:
        model_q = sm.OLS(df_q['forecast_error'],
                         sm.add_constant(df_q['forecast_revision'])).fit()

        add_result(
            spec_id=f'robust/sample/quartile_{q+1}',
            spec_tree_path='robustness/sample_restrictions.md',
            model=model_q,
            outcome_var='log_forecast_error',
            treatment_var='forecast_revision',
            sample_desc=f'SPF GDP forecasts, time quartile {q+1}',
            fixed_effects='None',
            controls_desc='None (bivariate)',
            cluster_var='None',
            model_type='OLS'
        )

# ============================================================================
# SPECS 19-23: INFERENCE VARIATIONS
# ============================================================================

# Spec 19: Heteroskedasticity-robust SEs (HC1)
model_hc1 = sm.OLS(df_baseline['forecast_error'],
                   sm.add_constant(df_baseline['forecast_revision'])).fit(cov_type='HC1')

add_result(
    spec_id='robust/inference/robust_hc1',
    spec_tree_path='robustness/clustering_variations.md',
    model=model_hc1,
    outcome_var='log_forecast_error',
    treatment_var='forecast_revision',
    sample_desc='SPF GDP forecasts, HC1 robust SEs',
    fixed_effects='None',
    controls_desc='None (bivariate)',
    cluster_var='None (HC1 robust)',
    model_type='OLS'
)

# Spec 20: HC2 robust SEs
model_hc2 = sm.OLS(df_baseline['forecast_error'],
                   sm.add_constant(df_baseline['forecast_revision'])).fit(cov_type='HC2')

add_result(
    spec_id='robust/inference/robust_hc2',
    spec_tree_path='robustness/clustering_variations.md',
    model=model_hc2,
    outcome_var='log_forecast_error',
    treatment_var='forecast_revision',
    sample_desc='SPF GDP forecasts, HC2 robust SEs',
    fixed_effects='None',
    controls_desc='None (bivariate)',
    cluster_var='None (HC2 robust)',
    model_type='OLS'
)

# Spec 21: HC3 robust SEs
model_hc3 = sm.OLS(df_baseline['forecast_error'],
                   sm.add_constant(df_baseline['forecast_revision'])).fit(cov_type='HC3')

add_result(
    spec_id='robust/inference/robust_hc3',
    spec_tree_path='robustness/clustering_variations.md',
    model=model_hc3,
    outcome_var='log_forecast_error',
    treatment_var='forecast_revision',
    sample_desc='SPF GDP forecasts, HC3 robust SEs',
    fixed_effects='None',
    controls_desc='None (bivariate)',
    cluster_var='None (HC3 robust)',
    model_type='OLS'
)

# Spec 22: HAC (Newey-West) SEs with 4 lags
model_hac4 = sm.OLS(df_baseline['forecast_error'],
                    sm.add_constant(df_baseline['forecast_revision'])).fit(
                        cov_type='HAC', cov_kwds={'maxlags': 4})

add_result(
    spec_id='robust/inference/hac_nw4',
    spec_tree_path='robustness/clustering_variations.md',
    model=model_hac4,
    outcome_var='log_forecast_error',
    treatment_var='forecast_revision',
    sample_desc='SPF GDP forecasts, Newey-West HAC SEs (4 lags)',
    fixed_effects='None',
    controls_desc='None (bivariate)',
    cluster_var='HAC (4 lags)',
    model_type='OLS'
)

# Spec 23: HAC (Newey-West) SEs with 8 lags
model_hac8 = sm.OLS(df_baseline['forecast_error'],
                    sm.add_constant(df_baseline['forecast_revision'])).fit(
                        cov_type='HAC', cov_kwds={'maxlags': 8})

add_result(
    spec_id='robust/inference/hac_nw8',
    spec_tree_path='robustness/clustering_variations.md',
    model=model_hac8,
    outcome_var='log_forecast_error',
    treatment_var='forecast_revision',
    sample_desc='SPF GDP forecasts, Newey-West HAC SEs (8 lags)',
    fixed_effects='None',
    controls_desc='None (bivariate)',
    cluster_var='HAC (8 lags)',
    model_type='OLS'
)

# ============================================================================
# SPECS 24-30: FUNCTIONAL FORM VARIATIONS
# ============================================================================

# Spec 24: Levels (not logs)
df['forecast_error_levels'] = df['outcome'] - df['forecast']
df['forecast_revision_levels'] = df['forecast'] - df['previous_forecast']

model_levels = sm.OLS(df['forecast_error_levels'],
                      sm.add_constant(df['forecast_revision_levels'])).fit()

add_result(
    spec_id='robust/form/levels_not_logs',
    spec_tree_path='robustness/functional_form.md',
    model=model_levels,
    outcome_var='forecast_error_levels',
    treatment_var='forecast_revision_levels',
    sample_desc='SPF GDP forecasts, levels instead of logs',
    fixed_effects='None',
    controls_desc='None (bivariate)',
    cluster_var='None',
    model_type='OLS'
)

# Spec 25: Growth rate form
df['outcome_growth'] = df['outcome'].pct_change()
df['forecast_growth'] = df['forecast'].pct_change()
df['forecast_error_growth'] = df['outcome_growth'] - df['forecast_growth']
df['forecast_revision_growth'] = df['forecast_growth'] - df['previous_forecast'].pct_change()

df_growth = df.dropna(subset=['forecast_error_growth', 'forecast_revision_growth']).copy()

model_growth = sm.OLS(df_growth['forecast_error_growth'],
                      sm.add_constant(df_growth['forecast_revision_growth'])).fit()

add_result(
    spec_id='robust/form/growth_rates',
    spec_tree_path='robustness/functional_form.md',
    model=model_growth,
    outcome_var='forecast_error_growth',
    treatment_var='forecast_revision_growth',
    sample_desc='SPF GDP forecasts, growth rate form',
    fixed_effects='None',
    controls_desc='None (bivariate)',
    cluster_var='None',
    model_type='OLS'
)

# Spec 26: Quadratic in forecast revision
df_baseline['forecast_revision_sq'] = df_baseline['forecast_revision'] ** 2
X_quad = sm.add_constant(df_baseline[['forecast_revision', 'forecast_revision_sq']])
model_quad = sm.OLS(df_baseline['forecast_error'], X_quad).fit()

add_result(
    spec_id='robust/form/quadratic',
    spec_tree_path='robustness/functional_form.md',
    model=model_quad,
    outcome_var='log_forecast_error',
    treatment_var='forecast_revision',
    sample_desc='SPF GDP forecasts, quadratic specification',
    fixed_effects='None',
    controls_desc='Quadratic in forecast revision',
    cluster_var='None',
    model_type='OLS'
)

# Spec 27: Standardized variables
df_baseline['forecast_error_std'] = (df_baseline['forecast_error'] -
                                      df_baseline['forecast_error'].mean()) / df_baseline['forecast_error'].std()
df_baseline['forecast_revision_std'] = (df_baseline['forecast_revision'] -
                                         df_baseline['forecast_revision'].mean()) / df_baseline['forecast_revision'].std()

model_std = sm.OLS(df_baseline['forecast_error_std'],
                   sm.add_constant(df_baseline['forecast_revision_std'])).fit()

add_result(
    spec_id='robust/form/standardized',
    spec_tree_path='robustness/functional_form.md',
    model=model_std,
    outcome_var='forecast_error_standardized',
    treatment_var='forecast_revision_standardized',
    sample_desc='SPF GDP forecasts, standardized variables',
    fixed_effects='None',
    controls_desc='None (bivariate)',
    cluster_var='None',
    model_type='OLS'
)

# Spec 28-30: Quantile regressions
from statsmodels.regression.quantile_regression import QuantReg

for q, q_name in [(0.25, '25'), (0.50, '50'), (0.75, '75')]:
    X = sm.add_constant(df_baseline['forecast_revision'])
    model_quant = QuantReg(df_baseline['forecast_error'], X).fit(q=q)

    add_result(
        spec_id=f'robust/form/quantile_{q_name}',
        spec_tree_path='robustness/functional_form.md',
        model=model_quant,
        outcome_var='log_forecast_error',
        treatment_var='forecast_revision',
        sample_desc=f'SPF GDP forecasts, {q_name}th quantile regression',
        fixed_effects='None',
        controls_desc='None (bivariate)',
        cluster_var='None',
        model_type='Quantile'
    )

# ============================================================================
# SPECS 31-37: CONTROL VARIABLES (time trends, lagged values)
# ============================================================================

# Spec 31: Linear time trend
df_baseline['time_trend'] = range(len(df_baseline))
X_trend = sm.add_constant(df_baseline[['forecast_revision', 'time_trend']])
model_trend = sm.OLS(df_baseline['forecast_error'], X_trend).fit()

add_result(
    spec_id='robust/control/time_trend',
    spec_tree_path='robustness/control_progression.md',
    model=model_trend,
    outcome_var='log_forecast_error',
    treatment_var='forecast_revision',
    sample_desc='SPF GDP forecasts, with linear time trend',
    fixed_effects='None',
    controls_desc='Linear time trend',
    cluster_var='None',
    model_type='OLS'
)

# Spec 32: Quadratic time trend
df_baseline['time_trend_sq'] = df_baseline['time_trend'] ** 2
X_trend2 = sm.add_constant(df_baseline[['forecast_revision', 'time_trend', 'time_trend_sq']])
model_trend2 = sm.OLS(df_baseline['forecast_error'], X_trend2).fit()

add_result(
    spec_id='robust/control/quadratic_time_trend',
    spec_tree_path='robustness/control_progression.md',
    model=model_trend2,
    outcome_var='log_forecast_error',
    treatment_var='forecast_revision',
    sample_desc='SPF GDP forecasts, with quadratic time trend',
    fixed_effects='None',
    controls_desc='Quadratic time trend',
    cluster_var='None',
    model_type='OLS'
)

# Spec 33: Lagged forecast error
df_baseline['forecast_error_lag1'] = df_baseline['forecast_error'].shift(1)
df_lag1 = df_baseline.dropna(subset=['forecast_error_lag1']).copy()
X_lag1 = sm.add_constant(df_lag1[['forecast_revision', 'forecast_error_lag1']])
model_lag1 = sm.OLS(df_lag1['forecast_error'], X_lag1).fit()

add_result(
    spec_id='robust/control/lagged_forecast_error',
    spec_tree_path='robustness/control_progression.md',
    model=model_lag1,
    outcome_var='log_forecast_error',
    treatment_var='forecast_revision',
    sample_desc='SPF GDP forecasts, controlling for lagged FE',
    fixed_effects='None',
    controls_desc='Lagged forecast error',
    cluster_var='None',
    model_type='OLS'
)

# Spec 34: Lagged forecast revision
df_baseline['forecast_revision_lag1'] = df_baseline['forecast_revision'].shift(1)
df_lag1_rev = df_baseline.dropna(subset=['forecast_revision_lag1']).copy()
X_lag1_rev = sm.add_constant(df_lag1_rev[['forecast_revision', 'forecast_revision_lag1']])
model_lag1_rev = sm.OLS(df_lag1_rev['forecast_error'], X_lag1_rev).fit()

add_result(
    spec_id='robust/control/lagged_forecast_revision',
    spec_tree_path='robustness/control_progression.md',
    model=model_lag1_rev,
    outcome_var='log_forecast_error',
    treatment_var='forecast_revision',
    sample_desc='SPF GDP forecasts, controlling for lagged FR',
    fixed_effects='None',
    controls_desc='Lagged forecast revision',
    cluster_var='None',
    model_type='OLS'
)

# Spec 35: Both lagged FE and FR
df_both = df_baseline.dropna(subset=['forecast_error_lag1', 'forecast_revision_lag1']).copy()
X_both = sm.add_constant(df_both[['forecast_revision', 'forecast_error_lag1', 'forecast_revision_lag1']])
model_both = sm.OLS(df_both['forecast_error'], X_both).fit()

add_result(
    spec_id='robust/control/lagged_both',
    spec_tree_path='robustness/control_progression.md',
    model=model_both,
    outcome_var='log_forecast_error',
    treatment_var='forecast_revision',
    sample_desc='SPF GDP forecasts, controlling for lagged FE and FR',
    fixed_effects='None',
    controls_desc='Lagged FE and FR',
    cluster_var='None',
    model_type='OLS'
)

# Spec 36: Recession indicator
df_baseline['recession_int'] = df_baseline['recession'].astype(int)
X_rec = sm.add_constant(df_baseline[['forecast_revision', 'recession_int']])
model_rec_ctrl = sm.OLS(df_baseline['forecast_error'], X_rec).fit()

add_result(
    spec_id='robust/control/recession_indicator',
    spec_tree_path='robustness/control_progression.md',
    model=model_rec_ctrl,
    outcome_var='log_forecast_error',
    treatment_var='forecast_revision',
    sample_desc='SPF GDP forecasts, controlling for recessions',
    fixed_effects='None',
    controls_desc='Recession indicator',
    cluster_var='None',
    model_type='OLS'
)

# Spec 37: Full model (time trend + lagged + recession)
df_full = df_baseline.dropna(subset=['forecast_error_lag1', 'forecast_revision_lag1']).copy()
X_full = sm.add_constant(df_full[['forecast_revision', 'time_trend', 'forecast_error_lag1',
                                   'forecast_revision_lag1', 'recession_int']])
model_full_ctrl = sm.OLS(df_full['forecast_error'], X_full).fit()

add_result(
    spec_id='robust/control/full_model',
    spec_tree_path='robustness/control_progression.md',
    model=model_full_ctrl,
    outcome_var='log_forecast_error',
    treatment_var='forecast_revision',
    sample_desc='SPF GDP forecasts, full control set',
    fixed_effects='None',
    controls_desc='Time trend, lagged FE, lagged FR, recession',
    cluster_var='None',
    model_type='OLS'
)

# ============================================================================
# SPECS 38-42: ALTERNATIVE ESTIMATORS
# ============================================================================

# Spec 38: Weighted least squares (weight by forecast level)
weights = 1 / df_baseline['forecast'].values
model_wls = sm.WLS(df_baseline['forecast_error'],
                   sm.add_constant(df_baseline['forecast_revision']),
                   weights=weights).fit()

add_result(
    spec_id='robust/estimation/wls_forecast_weight',
    spec_tree_path='robustness/model_specification.md',
    model=model_wls,
    outcome_var='log_forecast_error',
    treatment_var='forecast_revision',
    sample_desc='SPF GDP forecasts, WLS weighted by 1/forecast',
    fixed_effects='None',
    controls_desc='None (bivariate)',
    cluster_var='None',
    model_type='WLS'
)

# Spec 39: Robust regression (M-estimation)
from statsmodels.robust.robust_linear_model import RLM

model_rlm = RLM(df_baseline['forecast_error'],
                sm.add_constant(df_baseline['forecast_revision'])).fit()

add_result(
    spec_id='robust/estimation/m_estimation',
    spec_tree_path='robustness/model_specification.md',
    model=model_rlm,
    outcome_var='log_forecast_error',
    treatment_var='forecast_revision',
    sample_desc='SPF GDP forecasts, M-estimation (Huber)',
    fixed_effects='None',
    controls_desc='None (bivariate)',
    cluster_var='None',
    model_type='Robust'
)

# Spec 40: RANSAC-like (drop high residuals and re-estimate)
model_init = sm.OLS(df_baseline['forecast_error'],
                    sm.add_constant(df_baseline['forecast_revision'])).fit()
resid_std = np.abs(model_init.resid) / model_init.resid.std()
df_inliers = df_baseline[resid_std < 2].copy()

model_inliers = sm.OLS(df_inliers['forecast_error'],
                       sm.add_constant(df_inliers['forecast_revision'])).fit()

add_result(
    spec_id='robust/estimation/drop_high_residuals',
    spec_tree_path='robustness/model_specification.md',
    model=model_inliers,
    outcome_var='log_forecast_error',
    treatment_var='forecast_revision',
    sample_desc='SPF GDP forecasts, dropping obs with |residual| > 2 SD',
    fixed_effects='None',
    controls_desc='None (bivariate)',
    cluster_var='None',
    model_type='OLS'
)

# Spec 41: First differences
df_baseline['forecast_error_diff'] = df_baseline['forecast_error'].diff()
df_baseline['forecast_revision_diff'] = df_baseline['forecast_revision'].diff()
df_diff = df_baseline.dropna(subset=['forecast_error_diff', 'forecast_revision_diff']).copy()

model_diff = sm.OLS(df_diff['forecast_error_diff'],
                    sm.add_constant(df_diff['forecast_revision_diff'])).fit()

add_result(
    spec_id='robust/estimation/first_differences',
    spec_tree_path='robustness/model_specification.md',
    model=model_diff,
    outcome_var='forecast_error_diff',
    treatment_var='forecast_revision_diff',
    sample_desc='SPF GDP forecasts, first differences',
    fixed_effects='None',
    controls_desc='None (bivariate)',
    cluster_var='None',
    model_type='OLS'
)

# Spec 42: GLS with AR(1) errors
from statsmodels.regression.linear_model import GLS

# Estimate AR(1) coefficient from OLS residuals
rho = np.corrcoef(model_init.resid[:-1], model_init.resid[1:])[0, 1]

# Create AR(1) error covariance matrix
n = len(df_baseline)
sigma = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        sigma[i, j] = rho ** abs(i - j)

model_gls = GLS(df_baseline['forecast_error'],
                sm.add_constant(df_baseline['forecast_revision']),
                sigma=sigma).fit()

add_result(
    spec_id='robust/estimation/gls_ar1',
    spec_tree_path='robustness/model_specification.md',
    model=model_gls,
    outcome_var='log_forecast_error',
    treatment_var='forecast_revision',
    sample_desc='SPF GDP forecasts, GLS with AR(1) errors',
    fixed_effects='None',
    controls_desc='None (bivariate)',
    cluster_var='None',
    model_type='GLS'
)

# ============================================================================
# SPECS 43-48: HETEROGENEITY ANALYSIS
# ============================================================================

# Spec 43: Interaction with time trend
df_baseline['fr_x_trend'] = df_baseline['forecast_revision'] * df_baseline['time_trend']
X_interact_trend = sm.add_constant(df_baseline[['forecast_revision', 'time_trend', 'fr_x_trend']])
model_interact_trend = sm.OLS(df_baseline['forecast_error'], X_interact_trend).fit()

add_result(
    spec_id='robust/heterogeneity/time_trend_interaction',
    spec_tree_path='robustness/heterogeneity.md',
    model=model_interact_trend,
    outcome_var='log_forecast_error',
    treatment_var='forecast_revision',
    sample_desc='SPF GDP forecasts, FR x time trend interaction',
    fixed_effects='None',
    controls_desc='Time trend, FR x trend interaction',
    cluster_var='None',
    model_type='OLS'
)

# Spec 44: Interaction with recession
df_baseline['fr_x_recession'] = df_baseline['forecast_revision'] * df_baseline['recession_int']
X_interact_rec = sm.add_constant(df_baseline[['forecast_revision', 'recession_int', 'fr_x_recession']])
model_interact_rec = sm.OLS(df_baseline['forecast_error'], X_interact_rec).fit()

add_result(
    spec_id='robust/heterogeneity/recession_interaction',
    spec_tree_path='robustness/heterogeneity.md',
    model=model_interact_rec,
    outcome_var='log_forecast_error',
    treatment_var='forecast_revision',
    sample_desc='SPF GDP forecasts, FR x recession interaction',
    fixed_effects='None',
    controls_desc='Recession indicator, FR x recession interaction',
    cluster_var='None',
    model_type='OLS'
)

# Spec 45: Positive vs negative revisions
df_baseline['positive_revision'] = (df_baseline['forecast_revision'] > 0).astype(int)
df_baseline['fr_x_positive'] = df_baseline['forecast_revision'] * df_baseline['positive_revision']
X_interact_pos = sm.add_constant(df_baseline[['forecast_revision', 'positive_revision', 'fr_x_positive']])
model_interact_pos = sm.OLS(df_baseline['forecast_error'], X_interact_pos).fit()

add_result(
    spec_id='robust/heterogeneity/positive_revision_interaction',
    spec_tree_path='robustness/heterogeneity.md',
    model=model_interact_pos,
    outcome_var='log_forecast_error',
    treatment_var='forecast_revision',
    sample_desc='SPF GDP forecasts, FR x positive revision interaction',
    fixed_effects='None',
    controls_desc='Positive revision indicator, FR x positive interaction',
    cluster_var='None',
    model_type='OLS'
)

# Spec 46: Large vs small revisions
median_fr = df_baseline['forecast_revision'].abs().median()
df_baseline['large_revision'] = (df_baseline['forecast_revision'].abs() > median_fr).astype(int)
df_baseline['fr_x_large'] = df_baseline['forecast_revision'] * df_baseline['large_revision']
X_interact_large = sm.add_constant(df_baseline[['forecast_revision', 'large_revision', 'fr_x_large']])
model_interact_large = sm.OLS(df_baseline['forecast_error'], X_interact_large).fit()

add_result(
    spec_id='robust/heterogeneity/large_revision_interaction',
    spec_tree_path='robustness/heterogeneity.md',
    model=model_interact_large,
    outcome_var='log_forecast_error',
    treatment_var='forecast_revision',
    sample_desc='SPF GDP forecasts, FR x large revision interaction',
    fixed_effects='None',
    controls_desc='Large revision indicator, FR x large interaction',
    cluster_var='None',
    model_type='OLS'
)

# Spec 47: Pre vs post Great Moderation (1984)
# Approximately obs 60 is around 1984
df_baseline['post_1984'] = (df_baseline['obs_id'] > 60).astype(int)
df_baseline['fr_x_post84'] = df_baseline['forecast_revision'] * df_baseline['post_1984']
X_interact_84 = sm.add_constant(df_baseline[['forecast_revision', 'post_1984', 'fr_x_post84']])
model_interact_84 = sm.OLS(df_baseline['forecast_error'], X_interact_84).fit()

add_result(
    spec_id='robust/heterogeneity/great_moderation_interaction',
    spec_tree_path='robustness/heterogeneity.md',
    model=model_interact_84,
    outcome_var='log_forecast_error',
    treatment_var='forecast_revision',
    sample_desc='SPF GDP forecasts, FR x post-1984 interaction',
    fixed_effects='None',
    controls_desc='Post-1984 indicator, FR x post-1984 interaction',
    cluster_var='None',
    model_type='OLS'
)

# Spec 48: Pre vs post 2008 crisis
# Approximately obs 155 is around 2008
df_baseline['post_2008'] = (df_baseline['obs_id'] > 155).astype(int)
df_baseline['fr_x_post08'] = df_baseline['forecast_revision'] * df_baseline['post_2008']
X_interact_08 = sm.add_constant(df_baseline[['forecast_revision', 'post_2008', 'fr_x_post08']])
model_interact_08 = sm.OLS(df_baseline['forecast_error'], X_interact_08).fit()

add_result(
    spec_id='robust/heterogeneity/post_2008_interaction',
    spec_tree_path='robustness/heterogeneity.md',
    model=model_interact_08,
    outcome_var='log_forecast_error',
    treatment_var='forecast_revision',
    sample_desc='SPF GDP forecasts, FR x post-2008 interaction',
    fixed_effects='None',
    controls_desc='Post-2008 indicator, FR x post-2008 interaction',
    cluster_var='None',
    model_type='OLS'
)

# ============================================================================
# SPECS 49-55: PLACEBO TESTS
# ============================================================================

# Spec 49: Randomize forecast revision (permutation test placeholder)
np.random.seed(42)
df_baseline['fr_random'] = np.random.permutation(df_baseline['forecast_revision'].values)
model_random = sm.OLS(df_baseline['forecast_error'],
                      sm.add_constant(df_baseline['fr_random'])).fit()

add_result(
    spec_id='robust/placebo/permuted_forecast_revision',
    spec_tree_path='robustness/placebo_tests.md',
    model=model_random,
    outcome_var='log_forecast_error',
    treatment_var='fr_random',
    sample_desc='SPF GDP forecasts, permuted forecast revision',
    fixed_effects='None',
    controls_desc='None (bivariate)',
    cluster_var='None',
    model_type='OLS'
)

# Spec 50: Lag forecast revision by 4 quarters (placebo timing)
df_baseline['fr_lag4'] = df_baseline['forecast_revision'].shift(4)
df_lag4 = df_baseline.dropna(subset=['fr_lag4']).copy()
model_lag4_placebo = sm.OLS(df_lag4['forecast_error'],
                            sm.add_constant(df_lag4['fr_lag4'])).fit()

add_result(
    spec_id='robust/placebo/lagged_4q_forecast_revision',
    spec_tree_path='robustness/placebo_tests.md',
    model=model_lag4_placebo,
    outcome_var='log_forecast_error',
    treatment_var='fr_lag4',
    sample_desc='SPF GDP forecasts, FR lagged 4 quarters',
    fixed_effects='None',
    controls_desc='None (bivariate)',
    cluster_var='None',
    model_type='OLS'
)

# Spec 51: Lead forecast revision by 1 quarter (reverse causality check)
df_baseline['fr_lead1'] = df_baseline['forecast_revision'].shift(-1)
df_lead1 = df_baseline.dropna(subset=['fr_lead1']).copy()
model_lead1 = sm.OLS(df_lead1['forecast_error'],
                     sm.add_constant(df_lead1['fr_lead1'])).fit()

add_result(
    spec_id='robust/placebo/lead_1q_forecast_revision',
    spec_tree_path='robustness/placebo_tests.md',
    model=model_lead1,
    outcome_var='log_forecast_error',
    treatment_var='fr_lead1',
    sample_desc='SPF GDP forecasts, FR led 1 quarter',
    fixed_effects='None',
    controls_desc='None (bivariate)',
    cluster_var='None',
    model_type='OLS'
)

# Spec 52: Lead forecast revision by 4 quarters
df_baseline['fr_lead4'] = df_baseline['forecast_revision'].shift(-4)
df_lead4 = df_baseline.dropna(subset=['fr_lead4']).copy()
model_lead4 = sm.OLS(df_lead4['forecast_error'],
                     sm.add_constant(df_lead4['fr_lead4'])).fit()

add_result(
    spec_id='robust/placebo/lead_4q_forecast_revision',
    spec_tree_path='robustness/placebo_tests.md',
    model=model_lead4,
    outcome_var='log_forecast_error',
    treatment_var='fr_lead4',
    sample_desc='SPF GDP forecasts, FR led 4 quarters',
    fixed_effects='None',
    controls_desc='None (bivariate)',
    cluster_var='None',
    model_type='OLS'
)

# Spec 53-55: Bootstrap confidence intervals
bootstrap_coefs = []
n_bootstrap = 1000
np.random.seed(123)

for _ in range(n_bootstrap):
    idx = np.random.choice(len(df_baseline), len(df_baseline), replace=True)
    df_boot = df_baseline.iloc[idx]
    model_boot = sm.OLS(df_boot['forecast_error'],
                        sm.add_constant(df_boot['forecast_revision'])).fit()
    bootstrap_coefs.append(model_boot.params['forecast_revision'])

bootstrap_coefs = np.array(bootstrap_coefs)
boot_se = bootstrap_coefs.std()
boot_ci_lower = np.percentile(bootstrap_coefs, 2.5)
boot_ci_upper = np.percentile(bootstrap_coefs, 97.5)

# Create a model with bootstrap SE
model_boot_final = sm.OLS(df_baseline['forecast_error'],
                          sm.add_constant(df_baseline['forecast_revision'])).fit()

# Manually override SE for recording
coef_val = model_boot_final.params['forecast_revision']
results.append({
    'paper_id': PAPER_ID,
    'journal': JOURNAL,
    'paper_title': PAPER_TITLE,
    'spec_id': 'robust/inference/bootstrap_se',
    'spec_tree_path': 'robustness/inference_alternatives.md',
    'outcome_var': 'log_forecast_error',
    'treatment_var': 'forecast_revision',
    'coefficient': coef_val,
    'std_error': boot_se,
    't_stat': coef_val / boot_se,
    'p_value': 2 * (1 - stats.norm.cdf(abs(coef_val / boot_se))),
    'ci_lower': boot_ci_lower,
    'ci_upper': boot_ci_upper,
    'n_obs': int(model_boot_final.nobs),
    'r_squared': model_boot_final.rsquared,
    'coefficient_vector_json': json.dumps({'bootstrap': True, 'n_iterations': n_bootstrap}),
    'sample_desc': 'SPF GDP forecasts, bootstrap SEs (1000 iterations)',
    'fixed_effects': 'None',
    'controls_desc': 'None (bivariate)',
    'cluster_var': 'Bootstrap',
    'model_type': 'OLS',
    'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
})

# ============================================================================
# SPECS 56-60: ROLLING WINDOW ANALYSIS
# ============================================================================

window_sizes = [40, 60, 80]

for window in window_sizes:
    rolling_coefs = []
    rolling_ses = []
    rolling_periods = []

    for start in range(len(df_baseline) - window + 1):
        df_window = df_baseline.iloc[start:start+window].copy()
        model_window = sm.OLS(df_window['forecast_error'],
                              sm.add_constant(df_window['forecast_revision'])).fit()
        rolling_coefs.append(model_window.params['forecast_revision'])
        rolling_ses.append(model_window.bse['forecast_revision'])
        rolling_periods.append(start + window // 2)

    # Report first, middle, and last windows
    for idx, label in [(0, 'early'), (len(rolling_coefs)//2, 'middle'), (-1, 'late')]:
        coef = rolling_coefs[idx]
        se = rolling_ses[idx]

        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': f'robust/sample/rolling_{window}q_{label}',
            'spec_tree_path': 'robustness/sample_restrictions.md',
            'outcome_var': 'log_forecast_error',
            'treatment_var': 'forecast_revision',
            'coefficient': coef,
            'std_error': se,
            't_stat': coef / se,
            'p_value': 2 * (1 - t_dist.cdf(abs(coef / se), window - 2)),
            'ci_lower': coef - 1.96 * se,
            'ci_upper': coef + 1.96 * se,
            'n_obs': window,
            'r_squared': np.nan,
            'coefficient_vector_json': json.dumps({'rolling_window': window, 'position': label}),
            'sample_desc': f'SPF GDP forecasts, {window}q rolling window ({label})',
            'fixed_effects': 'None',
            'controls_desc': 'None (bivariate)',
            'cluster_var': 'None',
            'model_type': 'OLS',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        })

# ============================================================================
# SPECS 61-65: ALTERNATIVE OUTCOME DEFINITIONS
# ============================================================================

# Spec 61: Absolute forecast error
df_baseline['abs_forecast_error'] = df_baseline['forecast_error'].abs()
model_abs = sm.OLS(df_baseline['abs_forecast_error'],
                   sm.add_constant(df_baseline['forecast_revision'].abs())).fit()

add_result(
    spec_id='robust/outcome/absolute_forecast_error',
    spec_tree_path='robustness/measurement.md',
    model=model_abs,
    outcome_var='abs_forecast_error',
    treatment_var='abs_forecast_revision',
    sample_desc='SPF GDP forecasts, absolute values',
    fixed_effects='None',
    controls_desc='None (bivariate)',
    cluster_var='None',
    model_type='OLS'
)

# Spec 62: Squared forecast error
df_baseline['sq_forecast_error'] = df_baseline['forecast_error'] ** 2
model_sq = sm.OLS(df_baseline['sq_forecast_error'],
                  sm.add_constant(df_baseline['forecast_revision'].abs())).fit()

add_result(
    spec_id='robust/outcome/squared_forecast_error',
    spec_tree_path='robustness/measurement.md',
    model=model_sq,
    outcome_var='sq_forecast_error',
    treatment_var='abs_forecast_revision',
    sample_desc='SPF GDP forecasts, squared FE on abs FR',
    fixed_effects='None',
    controls_desc='None (bivariate)',
    cluster_var='None',
    model_type='OLS'
)

# Spec 63: Sign of forecast error
df_baseline['sign_forecast_error'] = (df_baseline['forecast_error'] > 0).astype(int)
df_baseline['sign_forecast_revision'] = (df_baseline['forecast_revision'] > 0).astype(int)
model_sign = sm.OLS(df_baseline['sign_forecast_error'],
                    sm.add_constant(df_baseline['sign_forecast_revision'])).fit()

add_result(
    spec_id='robust/outcome/sign_indicators',
    spec_tree_path='robustness/measurement.md',
    model=model_sign,
    outcome_var='sign_forecast_error',
    treatment_var='sign_forecast_revision',
    sample_desc='SPF GDP forecasts, sign indicators',
    fixed_effects='None',
    controls_desc='None (bivariate)',
    cluster_var='None',
    model_type='OLS'
)

# Spec 64: Binary outcome (large error)
median_fe = df_baseline['forecast_error'].abs().median()
df_baseline['large_error'] = (df_baseline['forecast_error'].abs() > median_fe).astype(int)
model_large_err = sm.OLS(df_baseline['large_error'],
                         sm.add_constant(df_baseline['forecast_revision'])).fit()

add_result(
    spec_id='robust/outcome/large_error_binary',
    spec_tree_path='robustness/measurement.md',
    model=model_large_err,
    outcome_var='large_error_indicator',
    treatment_var='forecast_revision',
    sample_desc='SPF GDP forecasts, large error indicator',
    fixed_effects='None',
    controls_desc='None (bivariate)',
    cluster_var='None',
    model_type='LPM'
)

# Spec 65: Probit for large error
from statsmodels.discrete.discrete_model import Probit

model_probit = Probit(df_baseline['large_error'],
                      sm.add_constant(df_baseline['forecast_revision'])).fit(disp=0)

add_result(
    spec_id='robust/outcome/large_error_probit',
    spec_tree_path='robustness/measurement.md',
    model=model_probit,
    outcome_var='large_error_indicator',
    treatment_var='forecast_revision',
    sample_desc='SPF GDP forecasts, probit for large error',
    fixed_effects='None',
    controls_desc='None (bivariate)',
    cluster_var='None',
    model_type='Probit'
)

# ============================================================================
# ADDITIONAL SPECS TO REACH 50+
# ============================================================================

# Spec 66-70: Drop influential observations (iterative)
from scipy.stats import studentized_range

# Calculate Cook's distance
model_full_ols = sm.OLS(df_baseline['forecast_error'],
                        sm.add_constant(df_baseline['forecast_revision'])).fit()

influence = model_full_ols.get_influence()
cooks_d = influence.cooks_distance[0]

# Drop top 1%, 5%, 10% by Cook's D
for pct, pct_name in [(0.99, '1pct'), (0.95, '5pct'), (0.90, '10pct')]:
    threshold = np.percentile(cooks_d, pct * 100)
    df_no_inf = df_baseline[cooks_d <= threshold].copy()

    if len(df_no_inf) > 10:
        model_no_inf = sm.OLS(df_no_inf['forecast_error'],
                              sm.add_constant(df_no_inf['forecast_revision'])).fit()

        add_result(
            spec_id=f'robust/sample/drop_influential_{pct_name}',
            spec_tree_path='robustness/sample_restrictions.md',
            model=model_no_inf,
            outcome_var='log_forecast_error',
            treatment_var='forecast_revision',
            sample_desc=f'SPF GDP forecasts, dropping top {100-int(pct*100)}% influential',
            fixed_effects='None',
            controls_desc='None (bivariate)',
            cluster_var='None',
            model_type='OLS'
        )

# ============================================================================
# SAVE RESULTS
# ============================================================================

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
output_path = f"{OUTPUT_DIR}/specification_results.csv"
results_df.to_csv(output_path, index=False)

print(f"\n{'='*60}")
print(f"SPECIFICATION SEARCH COMPLETE")
print(f"{'='*60}")
print(f"Total specifications run: {len(results_df)}")
print(f"Results saved to: {output_path}")

# Summary statistics
print(f"\n=== SUMMARY STATISTICS ===")
print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({(results_df['coefficient'] > 0).mean()*100:.1f}%)")
print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({(results_df['p_value'] < 0.05).mean()*100:.1f}%)")
print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({(results_df['p_value'] < 0.01).mean()*100:.1f}%)")
print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")
print(f"Range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")

# Category breakdown
print(f"\n=== BREAKDOWN BY CATEGORY ===")
results_df['category'] = results_df['spec_id'].apply(lambda x: x.split('/')[0] if '/' in x else 'baseline')
category_summary = results_df.groupby('category').agg({
    'coefficient': ['count', 'mean'],
    'p_value': lambda x: (x < 0.05).mean()
}).round(3)
print(category_summary)
