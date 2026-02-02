"""
Specification Search Script for Paper 113829-V1
"Sweetening the Deal? Political Connections and Sugar Mills in India"
Author: Sandip Sukhtankar

This paper studies whether sugar mills with politically connected chairmen
pay farmers more during election years ("buying votes").

Main hypothesis: Mills with politician chairmen pay higher prices during election years
Outcome variable: rprice (real price paid per ton of cane)
Treatment variable: interall (polcon * election year)
Method: Panel fixed effects with two-way clustering

Method Classification:
- method_code: panel_fixed_effects
- method_tree_path: specification_tree/methods/panel_fixed_effects.md
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SETUP AND DATA LOADING
# ============================================================================

BASE_PATH = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search'
DATA_PATH = f'{BASE_PATH}/data/downloads/extracted/113829-V1/AEJ-App-2011-0020_data-and-replication-info/entirepanelrec.dta'

PAPER_ID = '113829-V1'
PAPER_TITLE = 'Sweetening the Deal? Political Connections and Sugar Mills in India'
JOURNAL = 'AEJ-Applied'

# Load data
df = pd.read_stata(DATA_PATH)

# Apply sample restrictions from original do-file
# Drop trial years
df = df[df['checktrial1'] != 1]
# Drop mills with no data on political connections
df = df[df['evercheck'].notna()]

# Ensure year is integer for FE
df['year'] = df['year'].astype(int)
df['tabfinal'] = df['tabfinal'].astype(int)

# Define control variable groups
RAIN_VARS = ['r1_', 'r2_', 'r3_', 'r4_', 'r5_', 'r6_', 'r7_', 'r8_', 'r9_', 'r10_', 'r11_', 'r12_',
             'r1_dev', 'r2_dev', 'r3_dev', 'r4_dev', 'r5_dev', 'r6_dev', 'r7_dev', 'r8_dev',
             'r9_dev', 'r10_dev', 'r11_dev', 'r12_dev']
MILL_CONTROLS = ['propbreakdown', 'propshortage']

# Create additional variables for robustness
df['lnrprice'] = np.log(df['rprice'])
df['rprice_ihs'] = np.arcsinh(df['rprice'])

# Create zone_year categorical - use existing if available
if 'zone_year' in df.columns:
    df['zone_year_cat'] = df['zone_year'].astype('Int64').astype(str)
else:
    # If not available, we'll skip two-way clustering at zone-year level
    df['zone_year_cat'] = None

# Create variables for heterogeneity analysis
df['high_capacity'] = (df['capacity'] > df['capacity'].median()).astype(int)
df['year_early'] = (df['year'] <= 1999).astype(int)
df['year_late'] = (df['year'] > 1999).astype(int)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def run_regression(formula, data, vcov_type='CRV1', cluster_var='tabfinal', spec_id='',
                   spec_tree_path='', outcome_var='rprice', treatment_var='interall',
                   sample_desc='full sample', fixed_effects='mill + year', controls_desc=''):
    """
    Run a regression and return results in standardized format.
    """
    try:
        # Handle clustering
        if vcov_type == 'hetero':
            model = pf.feols(formula, data=data, vcov='hetero')
        elif vcov_type == 'CRV1':
            model = pf.feols(formula, data=data, vcov={'CRV1': cluster_var})
        elif isinstance(vcov_type, dict):
            # For two-way clustering, need to format differently
            # pyfixest expects: vcov={'CRV1': 'var1+var2'}
            if 'CRV1' in vcov_type and isinstance(vcov_type['CRV1'], list):
                cluster_vars = '+'.join(vcov_type['CRV1'])
                model = pf.feols(formula, data=data, vcov={'CRV1': cluster_vars})
            else:
                model = pf.feols(formula, data=data, vcov=vcov_type)
        else:
            model = pf.feols(formula, data=data, vcov=vcov_type)

        # Extract results - pyfixest returns pandas Series
        coef_series = model.coef()
        se_series = model.se()
        pval_series = model.pvalue()
        tstat_series = model.tstat()

        # Get treatment coefficient
        if treatment_var in coef_series.index:
            treat_coef = float(coef_series[treatment_var])
            treat_se = float(se_series[treatment_var])
            treat_pval = float(pval_series[treatment_var])
            treat_tstat = float(tstat_series[treatment_var])
        else:
            # Treatment variable not found (might be absorbed or missing)
            treat_coef = np.nan
            treat_se = np.nan
            treat_pval = np.nan
            treat_tstat = np.nan

        # Calculate CI
        ci_lower = treat_coef - 1.96 * treat_se if not np.isnan(treat_se) else np.nan
        ci_upper = treat_coef + 1.96 * treat_se if not np.isnan(treat_se) else np.nan

        # Get number of observations and R-squared from private attributes
        n_obs = int(model._N)
        r_squared = float(model._r2)

        # Build coefficient vector JSON
        coef_vector = {
            'treatment': {
                'var': treatment_var,
                'coef': float(treat_coef) if not np.isnan(treat_coef) else None,
                'se': float(treat_se) if not np.isnan(treat_se) else None,
                'pval': float(treat_pval) if not np.isnan(treat_pval) else None
            },
            'controls': [],
            'fixed_effects': fixed_effects.split(' + ') if fixed_effects else [],
            'n_obs': n_obs,
            'r_squared': r_squared
        }

        # Add other coefficients
        for var in coef_series.index:
            if var != treatment_var:
                coef_vector['controls'].append({
                    'var': var,
                    'coef': float(coef_series[var]) if not np.isnan(coef_series[var]) else None,
                    'se': float(se_series[var]) if not np.isnan(se_series[var]) else None,
                    'pval': float(pval_series[var]) if not np.isnan(pval_series[var]) else None
                })

        return {
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec_id,
            'spec_tree_path': spec_tree_path,
            'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': treat_coef,
            'std_error': treat_se,
            't_stat': treat_tstat,
            'p_value': treat_pval,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_obs': n_obs,
            'r_squared': r_squared,
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects,
            'controls_desc': controls_desc,
            'cluster_var': cluster_var if vcov_type == 'CRV1' else str(vcov_type),
            'model_type': 'FE-OLS',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
    except Exception as e:
        print(f"Error in {spec_id}: {e}")
        return None

# ============================================================================
# SPECIFICATION SEARCH
# ============================================================================

results = []

# ----------------------------------------------------------------------------
# 1. BASELINE SPECIFICATION (Table 2, Column 2)
# ----------------------------------------------------------------------------
print("Running baseline specification...")

# The main specification from the paper: Table 2, Col 2
# rprice ~ polcon + interall + capacity + rain controls | tabfinal + year
# Clustered at tabfinal and zone_year (two-way)

# Build formula with rain controls
rain_formula = ' + '.join(RAIN_VARS)
baseline_formula = f'rprice ~ polcon + interall + capacity + {rain_formula} | tabfinal + year'

result = run_regression(
    formula=baseline_formula,
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='baseline',
    spec_tree_path='methods/panel_fixed_effects.md#baseline',
    controls_desc='capacity, monthly rainfall, rainfall deviations',
    sample_desc='full sample (drop trial years, missing political connection data)'
)
if result:
    results.append(result)
    baseline_coef = result['coefficient']
    print(f"  Baseline coefficient: {baseline_coef:.4f}, p-value: {result['p_value']:.4f}")

# ----------------------------------------------------------------------------
# 2. CORE PANEL FE SPECIFICATIONS
# ----------------------------------------------------------------------------
print("Running core panel FE specifications...")

# 2.1 No fixed effects (pooled OLS)
result = run_regression(
    formula=f'rprice ~ polcon + interall + capacity + {rain_formula}',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='panel/fe/none',
    spec_tree_path='methods/panel_fixed_effects.md#fixed-effects-structure',
    fixed_effects='none',
    controls_desc='capacity, rainfall controls'
)
if result:
    results.append(result)

# 2.2 Unit FE only
result = run_regression(
    formula=f'rprice ~ polcon + interall + capacity + {rain_formula} | tabfinal',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='panel/fe/unit',
    spec_tree_path='methods/panel_fixed_effects.md#fixed-effects-structure',
    fixed_effects='mill',
    controls_desc='capacity, rainfall controls'
)
if result:
    results.append(result)

# 2.3 Time FE only
result = run_regression(
    formula=f'rprice ~ polcon + interall + capacity + {rain_formula} | year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='panel/fe/time',
    spec_tree_path='methods/panel_fixed_effects.md#fixed-effects-structure',
    fixed_effects='year',
    controls_desc='capacity, rainfall controls'
)
if result:
    results.append(result)

# 2.4 Two-way FE (baseline)
result = run_regression(
    formula=f'rprice ~ polcon + interall + capacity + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='panel/fe/twoway',
    spec_tree_path='methods/panel_fixed_effects.md#fixed-effects-structure',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall controls'
)
if result:
    results.append(result)

# ----------------------------------------------------------------------------
# 3. CONTROL VARIABLE VARIATIONS
# ----------------------------------------------------------------------------
print("Running control variable variations...")

# 3.1 No controls (just treatment + FE)
result = run_regression(
    formula='rprice ~ polcon + interall | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/control/none',
    spec_tree_path='robustness/leave_one_out.md',
    fixed_effects='mill + year',
    controls_desc='none'
)
if result:
    results.append(result)

# 3.2 Only capacity
result = run_regression(
    formula='rprice ~ polcon + interall + capacity | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/control/capacity_only',
    spec_tree_path='robustness/single_covariate.md',
    fixed_effects='mill + year',
    controls_desc='capacity only'
)
if result:
    results.append(result)

# 3.3 Capacity + rainfall levels only (no deviations)
rain_levels = ' + '.join([f'r{i}_' for i in range(1, 13)])
result = run_regression(
    formula=f'rprice ~ polcon + interall + capacity + {rain_levels} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/control/rain_levels_only',
    spec_tree_path='robustness/leave_one_out.md',
    fixed_effects='mill + year',
    controls_desc='capacity, monthly rainfall (no deviations)'
)
if result:
    results.append(result)

# 3.4 Capacity + rainfall deviations only (no levels)
rain_devs = ' + '.join([f'r{i}_dev' for i in range(1, 13)])
result = run_regression(
    formula=f'rprice ~ polcon + interall + capacity + {rain_devs} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/control/rain_devs_only',
    spec_tree_path='robustness/leave_one_out.md',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall deviations (no levels)'
)
if result:
    results.append(result)

# 3.5 Drop capacity
result = run_regression(
    formula=f'rprice ~ polcon + interall + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/loo/drop_capacity',
    spec_tree_path='robustness/leave_one_out.md',
    fixed_effects='mill + year',
    controls_desc='rainfall controls (no capacity)'
)
if result:
    results.append(result)

# 3.6 Full controls (add mill operation variables)
full_formula = f'rprice ~ polcon + interall + capacity + recovery + propbreakdown + propshortage + {rain_formula} | tabfinal + year'
result = run_regression(
    formula=full_formula,
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/control/full',
    spec_tree_path='robustness/control_progression.md',
    fixed_effects='mill + year',
    controls_desc='capacity, recovery, propbreakdown, propshortage, rainfall'
)
if result:
    results.append(result)

# 3.7 Add recovery only
result = run_regression(
    formula=f'rprice ~ polcon + interall + capacity + recovery + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/control/add_recovery',
    spec_tree_path='robustness/control_progression.md',
    fixed_effects='mill + year',
    controls_desc='capacity, recovery, rainfall'
)
if result:
    results.append(result)

# 3.8 Minimal controls (capacity only, no rainfall)
result = run_regression(
    formula='rprice ~ polcon + interall + capacity | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/control/minimal',
    spec_tree_path='robustness/control_progression.md',
    fixed_effects='mill + year',
    controls_desc='capacity only'
)
if result:
    results.append(result)

# 3.9 Leave out polcon (just interall)
result = run_regression(
    formula=f'rprice ~ interall + capacity + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/loo/drop_polcon',
    spec_tree_path='robustness/leave_one_out.md',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall (no polcon main effect)'
)
if result:
    results.append(result)

# ----------------------------------------------------------------------------
# 4. ALTERNATIVE TREATMENT DEFINITIONS
# ----------------------------------------------------------------------------
print("Running alternative treatment definitions...")

# 4.1 pcconnected (national election connection)
result = run_regression(
    formula=f'rprice ~ polcon + pcconnected + capacity + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/treatment/pcconnected',
    spec_tree_path='robustness/measurement.md',
    treatment_var='pcconnected',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# 4.2 acconnected (state election connection)
result = run_regression(
    formula=f'rprice ~ polcon + acconnected + capacity + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/treatment/acconnected',
    spec_tree_path='robustness/measurement.md',
    treatment_var='acconnected',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# 4.3 pcinter (national election interaction only)
result = run_regression(
    formula=f'rprice ~ polcon + pcinter + capacity + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/treatment/pcinter',
    spec_tree_path='robustness/measurement.md',
    treatment_var='pcinter',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# 4.4 acinter (state election interaction only)
result = run_regression(
    formula=f'rprice ~ polcon + acinter + capacity + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/treatment/acinter',
    spec_tree_path='robustness/measurement.md',
    treatment_var='acinter',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# 4.5 polcon (politician chairman - no election interaction)
result = run_regression(
    formula=f'rprice ~ polcon + capacity + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/treatment/polcon_only',
    spec_tree_path='robustness/measurement.md',
    treatment_var='polcon',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# 4.6 interall_9497 (excluding 1994 and 1997 elections)
result = run_regression(
    formula=f'rprice ~ polcon + interall_9497 + capacity + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/treatment/interall_9497',
    spec_tree_path='robustness/measurement.md',
    treatment_var='interall_9497',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# ----------------------------------------------------------------------------
# 5. ALTERNATIVE OUTCOMES
# ----------------------------------------------------------------------------
print("Running alternative outcomes...")

# 5.1 Log price
result = run_regression(
    formula=f'lnrprice ~ polcon + interall + capacity + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/outcome/lnrprice',
    spec_tree_path='robustness/functional_form.md#outcome-variable-transformations',
    outcome_var='lnrprice',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# 5.2 Recovery (sugar produced per unit cane)
result = run_regression(
    formula=f'recovery ~ polcon + interall + capacity + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/outcome/recovery',
    spec_tree_path='robustness/functional_form.md',
    outcome_var='recovery',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# 5.3 Not in operation
result = run_regression(
    formula=f'notinoperation ~ polcon + interall + capacity + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/outcome/notinoperation',
    spec_tree_path='robustness/functional_form.md',
    outcome_var='notinoperation',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# 5.4 Actual hours worked
result = run_regression(
    formula=f'actualhours ~ polcon + interall + capacity + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/outcome/actualhours',
    spec_tree_path='robustness/functional_form.md',
    outcome_var='actualhours',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# 5.5 Cane crushed
result = run_regression(
    formula=f'canecrushed ~ polcon + interall + capacity + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/outcome/canecrushed',
    spec_tree_path='robustness/functional_form.md',
    outcome_var='canecrushed',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# 5.6 Proportion breakdown
result = run_regression(
    formula=f'propbreakdown ~ polcon + interall + capacity + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/outcome/propbreakdown',
    spec_tree_path='robustness/functional_form.md',
    outcome_var='propbreakdown',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# 5.7 Proportion shortage
result = run_regression(
    formula=f'propshortage ~ polcon + interall + capacity + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/outcome/propshortage',
    spec_tree_path='robustness/functional_form.md',
    outcome_var='propshortage',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# 5.8 Lime (as percent of cane crushed)
result = run_regression(
    formula=f'lime ~ polcon + interall + capacity + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/outcome/lime',
    spec_tree_path='robustness/functional_form.md',
    outcome_var='lime',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# 5.9 Sulphur (as percent of cane crushed)
result = run_regression(
    formula=f'sulphur ~ polcon + interall + capacity + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/outcome/sulphur',
    spec_tree_path='robustness/functional_form.md',
    outcome_var='sulphur',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# 5.10 Cane planted
result = run_regression(
    formula=f'caneplant ~ polcon + interall + capacity + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/outcome/caneplant',
    spec_tree_path='robustness/functional_form.md',
    outcome_var='caneplant',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# ----------------------------------------------------------------------------
# 6. CLUSTERING/INFERENCE VARIATIONS
# ----------------------------------------------------------------------------
print("Running clustering variations...")

# 6.1 Robust (heteroskedasticity-consistent) SEs
result = run_regression(
    formula=f'rprice ~ polcon + interall + capacity + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='hetero',
    cluster_var=None,
    spec_id='robust/cluster/robust_hc',
    spec_tree_path='robustness/clustering_variations.md#alternative-se-methods',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# 6.2 Cluster by year
result = run_regression(
    formula=f'rprice ~ polcon + interall + capacity + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='year',
    spec_id='robust/cluster/year',
    spec_tree_path='robustness/clustering_variations.md#single-level-clustering',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# 6.3 Two-way clustering (mill and year)
result = run_regression(
    formula=f'rprice ~ polcon + interall + capacity + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type={'CRV1': ['tabfinal', 'year']},
    cluster_var='tabfinal + year',
    spec_id='robust/cluster/twoway_mill_year',
    spec_tree_path='robustness/clustering_variations.md#two-way-clustering',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# ----------------------------------------------------------------------------
# 7. SAMPLE RESTRICTIONS
# ----------------------------------------------------------------------------
print("Running sample restrictions...")

# 7.1 Early period (1993-1999)
df_early = df[df['year'] <= 1999]
result = run_regression(
    formula=f'rprice ~ polcon + interall + capacity + {rain_formula} | tabfinal + year',
    data=df_early,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/sample/early_period',
    spec_tree_path='robustness/sample_restrictions.md#time-based-restrictions',
    sample_desc='1993-1999',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# 7.2 Late period (2000-2005)
df_late = df[df['year'] >= 2000]
result = run_regression(
    formula=f'rprice ~ polcon + interall + capacity + {rain_formula} | tabfinal + year',
    data=df_late,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/sample/late_period',
    spec_tree_path='robustness/sample_restrictions.md#time-based-restrictions',
    sample_desc='2000-2005',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# 7.3 Exclude first year (1993)
df_no_first = df[df['year'] > 1993]
result = run_regression(
    formula=f'rprice ~ polcon + interall + capacity + {rain_formula} | tabfinal + year',
    data=df_no_first,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/sample/exclude_first_year',
    spec_tree_path='robustness/sample_restrictions.md#time-based-restrictions',
    sample_desc='exclude 1993',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# 7.4 Exclude last year (2005)
df_no_last = df[df['year'] < 2005]
result = run_regression(
    formula=f'rprice ~ polcon + interall + capacity + {rain_formula} | tabfinal + year',
    data=df_no_last,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/sample/exclude_last_year',
    spec_tree_path='robustness/sample_restrictions.md#time-based-restrictions',
    sample_desc='exclude 2005',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# 7.5 Winsorize outcome at 1%
df_wins = df.copy()
lower = df_wins['rprice'].quantile(0.01)
upper = df_wins['rprice'].quantile(0.99)
df_wins['rprice'] = df_wins['rprice'].clip(lower=lower, upper=upper)
result = run_regression(
    formula=f'rprice ~ polcon + interall + capacity + {rain_formula} | tabfinal + year',
    data=df_wins,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/sample/winsor_1pct',
    spec_tree_path='robustness/sample_restrictions.md#outlier-handling',
    sample_desc='winsorized 1/99 pct',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# 7.6 Winsorize outcome at 5%
df_wins5 = df.copy()
lower5 = df_wins5['rprice'].quantile(0.05)
upper5 = df_wins5['rprice'].quantile(0.95)
df_wins5['rprice'] = df_wins5['rprice'].clip(lower=lower5, upper=upper5)
result = run_regression(
    formula=f'rprice ~ polcon + interall + capacity + {rain_formula} | tabfinal + year',
    data=df_wins5,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/sample/winsor_5pct',
    spec_tree_path='robustness/sample_restrictions.md#outlier-handling',
    sample_desc='winsorized 5/95 pct',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# 7.7 Trim outliers (drop extreme 1%)
df_trim = df[(df['rprice'] > df['rprice'].quantile(0.01)) &
             (df['rprice'] < df['rprice'].quantile(0.99))]
result = run_regression(
    formula=f'rprice ~ polcon + interall + capacity + {rain_formula} | tabfinal + year',
    data=df_trim,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/sample/trim_1pct',
    spec_tree_path='robustness/sample_restrictions.md#outlier-handling',
    sample_desc='trim 1/99 pct',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# 7.8 Only mills with politician chairman (ever)
df_polcheck = df[df['polcheck'] == 1]
result = run_regression(
    formula=f'rprice ~ polcon + interall + capacity + {rain_formula} | tabfinal + year',
    data=df_polcheck,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/sample/polcheck_mills',
    spec_tree_path='robustness/sample_restrictions.md#treatment-based-restrictions',
    sample_desc='mills with any political connection ever',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# 7.9 Only mills without political connections (placebo-like)
df_no_polcheck = df[df['polcheck'] == 0]
result = run_regression(
    formula=f'rprice ~ capacity + {rain_formula} | tabfinal + year',
    data=df_no_polcheck,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/sample/no_polcheck_mills',
    spec_tree_path='robustness/sample_restrictions.md#treatment-based-restrictions',
    treatment_var='capacity',  # placeholder since no political variables
    sample_desc='mills without political connection ever',
    fixed_effects='mill + year',
    controls_desc='rainfall'
)
if result:
    results.append(result)

# 7.10 Large capacity mills only
df_large = df[df['capacity'] >= df['capacity'].median()]
result = run_regression(
    formula=f'rprice ~ polcon + interall + capacity + {rain_formula} | tabfinal + year',
    data=df_large,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/sample/large_mills',
    spec_tree_path='robustness/sample_restrictions.md#geographic/unit-restrictions',
    sample_desc='above median capacity',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# 7.11 Small capacity mills only
df_small = df[df['capacity'] < df['capacity'].median()]
result = run_regression(
    formula=f'rprice ~ polcon + interall + capacity + {rain_formula} | tabfinal + year',
    data=df_small,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/sample/small_mills',
    spec_tree_path='robustness/sample_restrictions.md#geographic/unit-restrictions',
    sample_desc='below median capacity',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# 7.12 Mills with at least 5 observations
obs_per_mill = df.groupby('tabfinal')['rprice'].count()
mills_with_5plus = obs_per_mill[obs_per_mill >= 5].index
df_5plus = df[df['tabfinal'].isin(mills_with_5plus)]
result = run_regression(
    formula=f'rprice ~ polcon + interall + capacity + {rain_formula} | tabfinal + year',
    data=df_5plus,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/sample/min_obs_5',
    spec_tree_path='robustness/sample_restrictions.md#panel-specific',
    sample_desc='mills with 5+ observations',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# 7.13 Drop each year one at a time
for year_to_drop in df['year'].unique():
    df_drop_year = df[df['year'] != year_to_drop]
    result = run_regression(
        formula=f'rprice ~ polcon + interall + capacity + {rain_formula} | tabfinal + year',
        data=df_drop_year,
        vcov_type='CRV1',
        cluster_var='tabfinal',
        spec_id=f'robust/sample/drop_year_{int(year_to_drop)}',
        spec_tree_path='robustness/sample_restrictions.md#time-based-restrictions',
        sample_desc=f'exclude year {int(year_to_drop)}',
        fixed_effects='mill + year',
        controls_desc='capacity, rainfall'
    )
    if result:
        results.append(result)

# ----------------------------------------------------------------------------
# 8. FUNCTIONAL FORM VARIATIONS
# ----------------------------------------------------------------------------
print("Running functional form variations...")

# 8.1 Log-level (already done as outcome variation)

# 8.2 IHS transformation
result = run_regression(
    formula=f'rprice_ihs ~ polcon + interall + capacity + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/form/y_asinh',
    spec_tree_path='robustness/functional_form.md#outcome-variable-transformations',
    outcome_var='rprice_ihs',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# 8.3 Add quadratic capacity
result = run_regression(
    formula=f'rprice ~ polcon + interall + capacity + I(capacity**2) + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/form/quadratic_capacity',
    spec_tree_path='robustness/functional_form.md#nonlinear-specifications',
    fixed_effects='mill + year',
    controls_desc='capacity, capacity squared, rainfall'
)
if result:
    results.append(result)

# 8.4 Log capacity
df['log_capacity'] = np.log(df['capacity'])
result = run_regression(
    formula=f'rprice ~ polcon + interall + log_capacity + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/form/log_capacity',
    spec_tree_path='robustness/functional_form.md#control-variable-transformations',
    fixed_effects='mill + year',
    controls_desc='log capacity, rainfall'
)
if result:
    results.append(result)

# 8.5 Standardized outcome
df['rprice_std'] = (df['rprice'] - df['rprice'].mean()) / df['rprice'].std()
result = run_regression(
    formula=f'rprice_std ~ polcon + interall + capacity + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/form/y_standardized',
    spec_tree_path='robustness/functional_form.md#outcome-variable-transformations',
    outcome_var='rprice_std',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# ----------------------------------------------------------------------------
# 9. PLACEBO TESTS
# ----------------------------------------------------------------------------
print("Running placebo tests...")

# 9.1 Party in state as placebo (from Table 6)
result = run_regression(
    formula=f'rprice ~ partyinstate + partyincenter + capacity + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/placebo/party_power',
    spec_tree_path='robustness/placebo_tests.md#outcome-placebos',
    treatment_var='partyinstate',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# 9.2 Test on capacity (should not be affected by election timing)
result = run_regression(
    formula=f'capacity ~ polcon + interall | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/placebo/outcome_capacity',
    spec_tree_path='robustness/placebo_tests.md#outcome-placebos',
    outcome_var='capacity',
    fixed_effects='mill + year',
    controls_desc='none'
)
if result:
    results.append(result)

# 9.3 Election timing in/out of season (from Appendix 3)
result = run_regression(
    formula=f'rprice ~ polcon + allout_int + allin_int + capacity + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/placebo/election_timing',
    spec_tree_path='robustness/placebo_tests.md',
    treatment_var='allin_int',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# 9.4 After election effect (should be smaller)
result = run_regression(
    formula=f'rprice ~ pcinter + pcinterafter + capacity + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/placebo/post_election',
    spec_tree_path='robustness/placebo_tests.md#temporal-placebos',
    treatment_var='pcinterafter',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# ----------------------------------------------------------------------------
# 10. HETEROGENEITY ANALYSIS
# ----------------------------------------------------------------------------
print("Running heterogeneity analysis...")

# 10.1 Close election margin interaction (Table 3)
result = run_regression(
    formula=f'rprice ~ polcon + close_all*interall + capacity + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/het/close_election',
    spec_tree_path='robustness/heterogeneity.md#treatment-related-heterogeneity',
    treatment_var='close_all:interall',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# 10.2 By mill capacity
result = run_regression(
    formula=f'rprice ~ polcon + interall*high_capacity + capacity + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/het/by_capacity',
    spec_tree_path='robustness/heterogeneity.md#baseline-characteristics',
    treatment_var='interall:high_capacity',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# 10.3 Early vs late period
result = run_regression(
    formula=f'rprice ~ polcon + interall*year_late + capacity + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/het/by_period',
    spec_tree_path='robustness/heterogeneity.md#time-based-heterogeneity',
    treatment_var='interall:year_late',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# 10.4 National vs state elections
result = run_regression(
    formula=f'rprice ~ polcon + pcinter + acinter + capacity + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/het/national_vs_state',
    spec_tree_path='robustness/heterogeneity.md',
    treatment_var='pcinter',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# 10.5 Win vs lose elections (Table 5 style)
result = run_regression(
    formula=f'rprice ~ pcwin + pclose + pcwinafter + pcloseafter + capacity + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/het/win_vs_lose_pc',
    spec_tree_path='robustness/heterogeneity.md#treatment-related-heterogeneity',
    treatment_var='pcwin',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# 10.6 State election win vs lose
result = run_regression(
    formula=f'rprice ~ acwin + aclose + acwinafter + acloseafter + capacity + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/het/win_vs_lose_ac',
    spec_tree_path='robustness/heterogeneity.md#treatment-related-heterogeneity',
    treatment_var='acwin',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# 10.7 Split sample: high capacity
result = run_regression(
    formula=f'rprice ~ polcon + interall + capacity + {rain_formula} | tabfinal + year',
    data=df[df['high_capacity'] == 1],
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/het/subsample_high_capacity',
    spec_tree_path='robustness/heterogeneity.md#baseline-characteristics',
    sample_desc='high capacity mills',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# 10.8 Split sample: low capacity
result = run_regression(
    formula=f'rprice ~ polcon + interall + capacity + {rain_formula} | tabfinal + year',
    data=df[df['high_capacity'] == 0],
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='robust/het/subsample_low_capacity',
    spec_tree_path='robustness/heterogeneity.md#baseline-characteristics',
    sample_desc='low capacity mills',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# ----------------------------------------------------------------------------
# 11. ADDITIONAL SPECIFICATIONS FROM PAPER
# ----------------------------------------------------------------------------
print("Running additional specifications from paper...")

# 11.1 Appendix 2 style: separate pcall and acall
result = run_regression(
    formula=f'rprice ~ pcall + pcinter + capacity + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='custom/pcall_pcinter',
    spec_tree_path='custom',
    treatment_var='pcinter',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

result = run_regression(
    formula=f'rprice ~ acall + acinter + capacity + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='custom/acall_acinter',
    spec_tree_path='custom',
    treatment_var='acinter',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# 11.2 Election outside vs during crushing season
result = run_regression(
    formula=f'rprice ~ polcon + pcoutlse + pcinlse + capacity + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='custom/election_season_pc',
    spec_tree_path='custom',
    treatment_var='pcinlse',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

result = run_regression(
    formula=f'rprice ~ polcon + acoute + acine + capacity + {rain_formula} | tabfinal + year',
    data=df,
    vcov_type='CRV1',
    cluster_var='tabfinal',
    spec_id='custom/election_season_ac',
    spec_tree_path='custom',
    treatment_var='acine',
    fixed_effects='mill + year',
    controls_desc='capacity, rainfall'
)
if result:
    results.append(result)

# ============================================================================
# SAVE RESULTS
# ============================================================================

print(f"\nTotal specifications run: {len(results)}")

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Save to package directory
output_path = f'{BASE_PATH}/data/downloads/extracted/113829-V1/specification_results.csv'
results_df.to_csv(output_path, index=False)
print(f"Saved results to {output_path}")

# Also save to scripts directory for backup
backup_path = f'{BASE_PATH}/scripts/paper_analyses/113829-V1_results.csv'
results_df.to_csv(backup_path, index=False)

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

# Focus on main treatment variable (interall)
main_results = results_df[results_df['treatment_var'] == 'interall'].copy()

if len(main_results) > 0:
    print(f"\nSpecifications with interall as treatment: {len(main_results)}")
    print(f"Positive coefficients: {(main_results['coefficient'] > 0).sum()} ({(main_results['coefficient'] > 0).mean()*100:.1f}%)")
    print(f"Significant at 5%: {(main_results['p_value'] < 0.05).sum()} ({(main_results['p_value'] < 0.05).mean()*100:.1f}%)")
    print(f"Significant at 1%: {(main_results['p_value'] < 0.01).sum()} ({(main_results['p_value'] < 0.01).mean()*100:.1f}%)")
    print(f"Median coefficient: {main_results['coefficient'].median():.4f}")
    print(f"Mean coefficient: {main_results['coefficient'].mean():.4f}")
    print(f"Range: [{main_results['coefficient'].min():.4f}, {main_results['coefficient'].max():.4f}]")

print(f"\nAll specifications: {len(results_df)}")
print(f"With positive treatment coefficient: {(results_df['coefficient'] > 0).sum()}")
print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()}")
