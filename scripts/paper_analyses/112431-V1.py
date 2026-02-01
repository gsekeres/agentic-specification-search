"""
Specification Search: 112431-V1
Paper: "Electoral Accountability and Corruption: Evidence from the Audits of Local Government"
Authors: Ferraz and Finan (AER)

Hypothesis: First-term mayors (with reelection incentives) are less corrupt than
            second-term mayors (lame ducks without reelection incentives).

Treatment: first (1 = first-term mayor, 0 = second-term/reeleito)
Outcome: pcorrupt (share of audited resources found to involve corruption)
Method: Cross-sectional OLS with state fixed effects
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

import pyfixest as pf
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Paths
PACKAGE_DIR = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/112431-V1'
OUTPUT_DIR = PACKAGE_DIR

# Paper metadata
PAPER_ID = '112431-V1'
JOURNAL = 'AER'
PAPER_TITLE = 'Electoral Accountability and Corruption: Evidence from the Audits of Local Government'

# Load data
df = pd.read_stata(f'{PACKAGE_DIR}/corruptiondata_aer.dta')

# Apply sample filter (esample2 == 1)
df = df[df['esample2'] == 1].copy()
print(f"Sample size after filter: {len(df)}")

# Define control variable sets (from Stata do file)
# prefchar2 = pref_masc pref_idade_tse pref_escola party_d1 party_d3-party_d18
# munichar2 = lpop purb p_secundario mun_novo lpib02 gini_ipea
# Additional: lrec_trans, p_cad_pref, vereador_eleit, ENLP2000, comarca, lfunc_ativ, lrec_fisc

# Mayor characteristics
prefchar = ['pref_masc', 'pref_idade_tse', 'pref_escola']
party_dummies = ['party_d1', 'party_d3', 'party_d4', 'party_d5', 'party_d6', 'party_d7',
                 'party_d8', 'party_d9', 'party_d10', 'party_d11', 'party_d12', 'party_d13',
                 'party_d14', 'party_d15', 'party_d16', 'party_d17', 'party_d18']

# Municipality characteristics
munichar = ['lpop', 'purb', 'p_secundario', 'mun_novo', 'lpib02', 'gini_ipea']

# Political/institutional controls
political_controls = ['p_cad_pref', 'vereador_eleit', 'ENLP2000', 'comarca']

# Audit controls
audit_controls = ['lrec_trans', 'lfunc_ativ', 'lrec_fisc']

# Lottery dummies
sorteio_dummies = ['sorteio2', 'sorteio3', 'sorteio4', 'sorteio5', 'sorteio6',
                   'sorteio7', 'sorteio8', 'sorteio9', 'sorteio10']

# State dummies (for FE)
uf_dummies = [c for c in df.columns if c.startswith('uf_d')]

# Full control set (Table 4, col 6)
full_controls = prefchar + party_dummies + munichar + ['lrec_trans'] + political_controls + sorteio_dummies

# Create variable lists for different control sets
all_controls = prefchar + party_dummies + munichar + ['lrec_trans', 'lfunc_ativ'] + political_controls + sorteio_dummies

# Handle missing values - create analysis sample
analysis_vars = ['pcorrupt', 'first'] + full_controls + uf_dummies
df_analysis = df.dropna(subset=['pcorrupt', 'first'] + prefchar + munichar + ['lrec_trans'] + political_controls).copy()
print(f"Analysis sample after dropping missing: {len(df_analysis)}")

# Results storage
results = []

def extract_results(model, spec_id, spec_tree_path, outcome_var, treatment_var,
                   controls_desc, fixed_effects, cluster_var, model_type,
                   sample_desc, n_obs, coef_vector_dict=None):
    """Extract results from pyfixest or statsmodels model"""

    if hasattr(model, 'coef'):  # pyfixest
        coef = model.coef()[treatment_var]
        se = model.se()[treatment_var]
        tstat = model.tstat()[treatment_var]
        pval = model.pvalue()[treatment_var]
        ci = model.confint()
        ci_lower = ci.loc[treatment_var, '2.5%']
        ci_upper = ci.loc[treatment_var, '97.5%']
        r2 = model.r2 if hasattr(model, 'r2') else None
    else:  # statsmodels
        coef = model.params[treatment_var]
        se = model.bse[treatment_var]
        tstat = model.tvalues[treatment_var]
        pval = model.pvalues[treatment_var]
        ci_lower = model.conf_int().loc[treatment_var, 0]
        ci_upper = model.conf_int().loc[treatment_var, 1]
        r2 = model.rsquared if hasattr(model, 'rsquared') else None

    # Build coefficient vector JSON
    if coef_vector_dict is None:
        coef_vector_dict = {}

    coef_vector_dict['treatment'] = {
        'var': treatment_var,
        'coef': float(coef),
        'se': float(se),
        'pval': float(pval)
    }

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
        't_stat': float(tstat),
        'p_value': float(pval),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'n_obs': int(n_obs),
        'r_squared': float(r2) if r2 is not None else None,
        'coefficient_vector_json': json.dumps(coef_vector_dict),
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    }

# ============================================================================
# BASELINE SPECIFICATIONS (Table 4)
# ============================================================================

print("\n=== Running Baseline Specifications ===")

# Baseline 1: No controls (Table 4, Col 1)
formula = 'pcorrupt ~ first'
model = smf.ols(formula, data=df_analysis).fit(cov_type='HC1')
results.append(extract_results(
    model, 'baseline', 'methods/cross_sectional_ols.md',
    'pcorrupt', 'first', 'None', 'None', 'robust', 'OLS',
    'Full sample esample2==1', len(df_analysis)
))
print(f"Baseline (no controls): coef={model.params['first']:.6f}, se={model.bse['first']:.6f}, p={model.pvalues['first']:.4f}")

# Baseline 2: Mayor characteristics (Table 4, Col 2)
formula = 'pcorrupt ~ first + ' + ' + '.join(prefchar + party_dummies)
model = smf.ols(formula, data=df_analysis).fit(cov_type='HC1')
results.append(extract_results(
    model, 'ols/controls/mayor_chars', 'methods/cross_sectional_ols.md#control-sets',
    'pcorrupt', 'first', 'Mayor characteristics + party dummies', 'None', 'robust', 'OLS',
    'Full sample esample2==1', len(df_analysis)
))
print(f"Mayor chars: coef={model.params['first']:.6f}, se={model.bse['first']:.6f}, p={model.pvalues['first']:.4f}")

# Baseline 3: Mayor + Municipality characteristics (Table 4, Col 3)
controls_3 = prefchar + party_dummies + munichar + ['lrec_trans']
formula = 'pcorrupt ~ first + ' + ' + '.join(controls_3)
df_temp = df_analysis.dropna(subset=controls_3)
model = smf.ols(formula, data=df_temp).fit(cov_type='HC1')
results.append(extract_results(
    model, 'ols/controls/mayor_munic', 'methods/cross_sectional_ols.md#control-sets',
    'pcorrupt', 'first', 'Mayor + municipality characteristics', 'None', 'robust', 'OLS',
    'Full sample esample2==1', len(df_temp)
))
print(f"Mayor+munic: coef={model.params['first']:.6f}, se={model.bse['first']:.6f}, p={model.pvalues['first']:.4f}")

# Baseline 4: Full controls without FE (Table 4, Col 4)
controls_4 = prefchar + party_dummies + munichar + ['lrec_trans'] + political_controls
formula = 'pcorrupt ~ first + ' + ' + '.join(controls_4)
df_temp = df_analysis.dropna(subset=controls_4)
model = smf.ols(formula, data=df_temp).fit(cov_type='HC1')
results.append(extract_results(
    model, 'ols/controls/full_no_fe', 'methods/cross_sectional_ols.md#control-sets',
    'pcorrupt', 'first', 'Full controls without lottery/state FE', 'None', 'robust', 'OLS',
    'Full sample esample2==1', len(df_temp)
))
print(f"Full no FE: coef={model.params['first']:.6f}, se={model.bse['first']:.6f}, p={model.pvalues['first']:.4f}")

# Baseline 5: With lottery dummies (Table 4, Col 5)
controls_5 = prefchar + party_dummies + munichar + ['lrec_trans'] + political_controls + sorteio_dummies
formula = 'pcorrupt ~ first + ' + ' + '.join(controls_5)
df_temp = df_analysis.dropna(subset=controls_5)
model = smf.ols(formula, data=df_temp).fit(cov_type='HC1')
results.append(extract_results(
    model, 'ols/controls/full_lottery', 'methods/cross_sectional_ols.md#control-sets',
    'pcorrupt', 'first', 'Full controls + lottery FE', 'Lottery', 'robust', 'OLS',
    'Full sample esample2==1', len(df_temp)
))
print(f"Full+lottery: coef={model.params['first']:.6f}, se={model.bse['first']:.6f}, p={model.pvalues['first']:.4f}")

# Baseline 6: With state FE (Table 4, Col 6) - MAIN SPECIFICATION
controls_6 = prefchar + party_dummies + munichar + ['lrec_trans'] + political_controls + sorteio_dummies + uf_dummies[:-1]  # drop one for collinearity
formula = 'pcorrupt ~ first + ' + ' + '.join(controls_6)
df_temp = df_analysis.dropna(subset=controls_6)
model = smf.ols(formula, data=df_temp).fit(cov_type='HC1')
baseline_coef = model.params['first']
baseline_se = model.bse['first']
baseline_pval = model.pvalues['first']
results.append(extract_results(
    model, 'ols/fe/state', 'methods/cross_sectional_ols.md#fixed-effects',
    'pcorrupt', 'first', 'Full controls + lottery + state FE', 'State + Lottery', 'robust', 'OLS',
    'Full sample esample2==1', len(df_temp)
))
print(f"Full+state FE (MAIN): coef={baseline_coef:.6f}, se={baseline_se:.6f}, p={baseline_pval:.4f}")

# ============================================================================
# METHOD-SPECIFIC SPECIFICATIONS
# ============================================================================

print("\n=== Running Method-Specific Specifications ===")

# OLS with different SE types
controls_base = prefchar + party_dummies + munichar + ['lrec_trans'] + political_controls + sorteio_dummies + uf_dummies[:-1]
formula = 'pcorrupt ~ first + ' + ' + '.join(controls_base)
df_temp = df_analysis.dropna(subset=controls_base)

# Classical SE
model = smf.ols(formula, data=df_temp).fit()
results.append(extract_results(
    model, 'ols/se/classical', 'methods/cross_sectional_ols.md#standard-errors',
    'pcorrupt', 'first', 'Full controls + FE', 'State + Lottery', 'classical', 'OLS',
    'Full sample esample2==1', len(df_temp)
))

# HC2
model = smf.ols(formula, data=df_temp).fit(cov_type='HC2')
results.append(extract_results(
    model, 'ols/se/hc2', 'methods/cross_sectional_ols.md#standard-errors',
    'pcorrupt', 'first', 'Full controls + FE', 'State + Lottery', 'HC2', 'OLS',
    'Full sample esample2==1', len(df_temp)
))

# HC3
model = smf.ols(formula, data=df_temp).fit(cov_type='HC3')
results.append(extract_results(
    model, 'ols/se/hc3', 'methods/cross_sectional_ols.md#standard-errors',
    'pcorrupt', 'first', 'Full controls + FE', 'State + Lottery', 'HC3', 'OLS',
    'Full sample esample2==1', len(df_temp)
))

print("SE variations completed")

# ============================================================================
# ALTERNATIVE OUTCOMES (Table 5)
# ============================================================================

print("\n=== Running Alternative Outcomes ===")

# ncorrupt - number of violations
if 'ncorrupt' in df_analysis.columns and df_analysis['ncorrupt'].notna().sum() > 100:
    formula = 'ncorrupt ~ first + ' + ' + '.join(controls_base)
    df_temp2 = df_analysis.dropna(subset=controls_base + ['ncorrupt'])
    model = smf.ols(formula, data=df_temp2).fit(cov_type='HC1')
    results.append(extract_results(
        model, 'ols/outcome/ncorrupt', 'methods/cross_sectional_ols.md',
        'ncorrupt', 'first', 'Full controls + FE', 'State + Lottery', 'robust', 'OLS',
        'Full sample esample2==1', len(df_temp2)
    ))
    print(f"ncorrupt: coef={model.params['first']:.6f}, p={model.pvalues['first']:.4f}")

# ncorrupt_os - share of service items with corruption
if 'ncorrupt_os' in df_analysis.columns and df_analysis['ncorrupt_os'].notna().sum() > 100:
    formula = 'ncorrupt_os ~ first + ' + ' + '.join(controls_base)
    df_temp2 = df_analysis.dropna(subset=controls_base + ['ncorrupt_os'])
    model = smf.ols(formula, data=df_temp2).fit(cov_type='HC1')
    results.append(extract_results(
        model, 'ols/outcome/ncorrupt_os', 'methods/cross_sectional_ols.md',
        'ncorrupt_os', 'first', 'Full controls + FE', 'State + Lottery', 'robust', 'OLS',
        'Full sample esample2==1', len(df_temp2)
    ))
    print(f"ncorrupt_os: coef={model.params['first']:.6f}, p={model.pvalues['first']:.4f}")

# pmismanagement (Table 8)
if 'pmismanagement' in df_analysis.columns and df_analysis['pmismanagement'].notna().sum() > 100:
    formula = 'pmismanagement ~ first + ' + ' + '.join(controls_base)
    df_temp2 = df_analysis.dropna(subset=controls_base + ['pmismanagement'])
    if len(df_temp2) > 50:
        model = smf.ols(formula, data=df_temp2).fit(cov_type='HC1')
        results.append(extract_results(
            model, 'ols/outcome/pmismanagement', 'methods/cross_sectional_ols.md',
            'pmismanagement', 'first', 'Full controls + FE', 'State + Lottery', 'robust', 'OLS',
            'Full sample esample2==1', len(df_temp2)
        ))
        print(f"pmismanagement: coef={model.params['first']:.6f}, p={model.pvalues['first']:.4f}")

# ============================================================================
# LEAVE-ONE-OUT ROBUSTNESS
# ============================================================================

print("\n=== Running Leave-One-Out Robustness ===")

# Core controls for LOO (excluding dummies which would be too many)
core_controls_loo = prefchar + munichar + ['lrec_trans'] + political_controls

for drop_var in core_controls_loo:
    remaining = [c for c in core_controls_loo if c != drop_var]
    full_remaining = remaining + party_dummies + sorteio_dummies + uf_dummies[:-1]
    formula = 'pcorrupt ~ first + ' + ' + '.join(full_remaining)
    try:
        df_temp = df_analysis.dropna(subset=full_remaining)
        model = smf.ols(formula, data=df_temp).fit(cov_type='HC1')
        results.append(extract_results(
            model, f'robust/loo/drop_{drop_var}', 'robustness/leave_one_out.md',
            'pcorrupt', 'first', f'Full controls minus {drop_var}', 'State + Lottery', 'robust', 'OLS',
            'Full sample esample2==1', len(df_temp)
        ))
    except Exception as e:
        print(f"LOO {drop_var} failed: {e}")

print("Leave-one-out completed")

# ============================================================================
# SINGLE COVARIATE ROBUSTNESS
# ============================================================================

print("\n=== Running Single Covariate Robustness ===")

# Bivariate
formula = 'pcorrupt ~ first'
model = smf.ols(formula, data=df_analysis).fit(cov_type='HC1')
results.append(extract_results(
    model, 'robust/single/none', 'robustness/single_covariate.md',
    'pcorrupt', 'first', 'None (bivariate)', 'None', 'robust', 'OLS',
    'Full sample esample2==1', len(df_analysis)
))

# Single covariate for key controls
key_single_controls = ['lpop', 'purb', 'lpib02', 'gini_ipea', 'pref_masc', 'pref_idade_tse',
                       'pref_escola', 'p_cad_pref', 'comarca', 'ENLP2000']

for control in key_single_controls:
    formula = f'pcorrupt ~ first + {control}'
    try:
        df_temp = df_analysis.dropna(subset=[control])
        model = smf.ols(formula, data=df_temp).fit(cov_type='HC1')
        results.append(extract_results(
            model, f'robust/single/{control}', 'robustness/single_covariate.md',
            'pcorrupt', 'first', f'Only {control}', 'None', 'robust', 'OLS',
            'Full sample esample2==1', len(df_temp)
        ))
    except Exception as e:
        print(f"Single {control} failed: {e}")

print("Single covariate completed")

# ============================================================================
# CLUSTERING VARIATIONS
# ============================================================================

print("\n=== Running Clustering Variations ===")

# Note: The original paper uses robust SE (not clustered).
# We test different clustering levels.

# State-level clustering
formula = 'pcorrupt ~ first + ' + ' + '.join(controls_base)
df_temp = df_analysis.dropna(subset=controls_base)
df_temp['state'] = df_temp['uf']

try:
    model = smf.ols(formula, data=df_temp).fit(cov_type='cluster', cov_kwds={'groups': df_temp['state']})
    results.append(extract_results(
        model, 'robust/cluster/state', 'robustness/clustering_variations.md#single-level-clustering',
        'pcorrupt', 'first', 'Full controls + FE', 'State + Lottery', 'cluster_state', 'OLS',
        'Full sample esample2==1', len(df_temp)
    ))
    print(f"Cluster state: coef={model.params['first']:.6f}, se={model.bse['first']:.6f}")
except Exception as e:
    print(f"State clustering failed: {e}")

# Lottery clustering
try:
    model = smf.ols(formula, data=df_temp).fit(cov_type='cluster', cov_kwds={'groups': df_temp['nsorteio']})
    results.append(extract_results(
        model, 'robust/cluster/lottery', 'robustness/clustering_variations.md#single-level-clustering',
        'pcorrupt', 'first', 'Full controls + FE', 'State + Lottery', 'cluster_lottery', 'OLS',
        'Full sample esample2==1', len(df_temp)
    ))
    print(f"Cluster lottery: coef={model.params['first']:.6f}, se={model.bse['first']:.6f}")
except Exception as e:
    print(f"Lottery clustering failed: {e}")

print("Clustering variations completed")

# ============================================================================
# FUNCTIONAL FORM ROBUSTNESS
# ============================================================================

print("\n=== Running Functional Form Robustness ===")

# Log transformation (adding small constant for zeros)
df_temp = df_analysis.dropna(subset=controls_base)
df_temp['log_pcorrupt'] = np.log(df_temp['pcorrupt'] + 0.001)

formula = 'log_pcorrupt ~ first + ' + ' + '.join(controls_base)
model = smf.ols(formula, data=df_temp).fit(cov_type='HC1')
results.append(extract_results(
    model, 'robust/form/y_log', 'robustness/functional_form.md#outcome-variable-transformations',
    'log(pcorrupt+0.001)', 'first', 'Full controls + FE', 'State + Lottery', 'robust', 'OLS',
    'Full sample esample2==1', len(df_temp)
))
print(f"Log outcome: coef={model.params['first']:.6f}, p={model.pvalues['first']:.4f}")

# Asinh transformation
df_temp['asinh_pcorrupt'] = np.arcsinh(df_temp['pcorrupt'] * 100)  # Scale for interpretability
formula = 'asinh_pcorrupt ~ first + ' + ' + '.join(controls_base)
model = smf.ols(formula, data=df_temp).fit(cov_type='HC1')
results.append(extract_results(
    model, 'robust/form/y_asinh', 'robustness/functional_form.md#outcome-variable-transformations',
    'asinh(pcorrupt*100)', 'first', 'Full controls + FE', 'State + Lottery', 'robust', 'OLS',
    'Full sample esample2==1', len(df_temp)
))
print(f"Asinh outcome: coef={model.params['first']:.6f}, p={model.pvalues['first']:.4f}")

# Tobit model (left-censored at 0)
try:
    from statsmodels.regression.linear_model import OLS
    from statsmodels.discrete.truncreg import TruncatedLLS

    # Simple Tobit approximation using OLS on non-censored
    df_positive = df_temp[df_temp['pcorrupt'] > 0].copy()
    formula = 'pcorrupt ~ first + ' + ' + '.join(controls_base)
    model = smf.ols(formula, data=df_positive).fit(cov_type='HC1')
    results.append(extract_results(
        model, 'robust/form/positive_only', 'robustness/functional_form.md#alternative-estimators',
        'pcorrupt', 'first', 'Full controls + FE', 'State + Lottery', 'robust', 'OLS (positive only)',
        'Positive corruption only', len(df_positive)
    ))
    print(f"Positive only: coef={model.params['first']:.6f}, p={model.pvalues['first']:.4f}")
except Exception as e:
    print(f"Tobit-style failed: {e}")

print("Functional form completed")

# ============================================================================
# SAMPLE RESTRICTIONS
# ============================================================================

print("\n=== Running Sample Restrictions ===")

df_temp = df_analysis.dropna(subset=controls_base)
formula = 'pcorrupt ~ first + ' + ' + '.join(controls_base)

# Trim outliers (1% tails of pcorrupt)
lower = df_temp['pcorrupt'].quantile(0.01)
upper = df_temp['pcorrupt'].quantile(0.99)
df_trimmed = df_temp[(df_temp['pcorrupt'] >= lower) & (df_temp['pcorrupt'] <= upper)]
model = smf.ols(formula, data=df_trimmed).fit(cov_type='HC1')
results.append(extract_results(
    model, 'robust/sample/trim_1pct', 'robustness/sample_restrictions.md#outlier-handling',
    'pcorrupt', 'first', 'Full controls + FE', 'State + Lottery', 'robust', 'OLS',
    'Trimmed 1% tails', len(df_trimmed)
))
print(f"Trim 1%: coef={model.params['first']:.6f}, p={model.pvalues['first']:.4f}, n={len(df_trimmed)}")

# Winsorize at 5%
df_winsor = df_temp.copy()
lower5 = df_temp['pcorrupt'].quantile(0.05)
upper5 = df_temp['pcorrupt'].quantile(0.95)
df_winsor['pcorrupt'] = df_winsor['pcorrupt'].clip(lower=lower5, upper=upper5)
model = smf.ols(formula, data=df_winsor).fit(cov_type='HC1')
results.append(extract_results(
    model, 'robust/sample/winsor_5pct', 'robustness/sample_restrictions.md#outlier-handling',
    'pcorrupt', 'first', 'Full controls + FE', 'State + Lottery', 'robust', 'OLS',
    'Winsorized 5%/95%', len(df_winsor)
))
print(f"Winsor 5%: coef={model.params['first']:.6f}, p={model.pvalues['first']:.4f}")

# By population size subgroups
median_pop = df_temp['lpop'].median()

# Small municipalities
df_small = df_temp[df_temp['lpop'] <= median_pop]
model = smf.ols(formula, data=df_small).fit(cov_type='HC1')
results.append(extract_results(
    model, 'robust/sample/small_munic', 'robustness/sample_restrictions.md#geographic-unit-restrictions',
    'pcorrupt', 'first', 'Full controls + FE', 'State + Lottery', 'robust', 'OLS',
    'Small municipalities (below median pop)', len(df_small)
))
print(f"Small munic: coef={model.params['first']:.6f}, p={model.pvalues['first']:.4f}, n={len(df_small)}")

# Large municipalities
df_large = df_temp[df_temp['lpop'] > median_pop]
model = smf.ols(formula, data=df_large).fit(cov_type='HC1')
results.append(extract_results(
    model, 'robust/sample/large_munic', 'robustness/sample_restrictions.md#geographic-unit-restrictions',
    'pcorrupt', 'first', 'Full controls + FE', 'State + Lottery', 'robust', 'OLS',
    'Large municipalities (above median pop)', len(df_large)
))
print(f"Large munic: coef={model.params['first']:.6f}, p={model.pvalues['first']:.4f}, n={len(df_large)}")

# Urban vs Rural (by urbanization rate)
median_urban = df_temp['purb'].median()

df_urban = df_temp[df_temp['purb'] > median_urban]
model = smf.ols(formula, data=df_urban).fit(cov_type='HC1')
results.append(extract_results(
    model, 'robust/sample/high_urban', 'robustness/sample_restrictions.md#geographic-unit-restrictions',
    'pcorrupt', 'first', 'Full controls + FE', 'State + Lottery', 'robust', 'OLS',
    'High urbanization municipalities', len(df_urban)
))
print(f"High urban: coef={model.params['first']:.6f}, p={model.pvalues['first']:.4f}, n={len(df_urban)}")

df_rural = df_temp[df_temp['purb'] <= median_urban]
model = smf.ols(formula, data=df_rural).fit(cov_type='HC1')
results.append(extract_results(
    model, 'robust/sample/low_urban', 'robustness/sample_restrictions.md#geographic-unit-restrictions',
    'pcorrupt', 'first', 'Full controls + FE', 'State + Lottery', 'robust', 'OLS',
    'Low urbanization municipalities', len(df_rural)
))
print(f"Low urban: coef={model.params['first']:.6f}, p={model.pvalues['first']:.4f}, n={len(df_rural)}")

# By income level
median_income = df_temp['lpib02'].median()

df_poor = df_temp[df_temp['lpib02'] <= median_income]
model = smf.ols(formula, data=df_poor).fit(cov_type='HC1')
results.append(extract_results(
    model, 'robust/sample/low_income', 'robustness/sample_restrictions.md#demographic-subgroups',
    'pcorrupt', 'first', 'Full controls + FE', 'State + Lottery', 'robust', 'OLS',
    'Low income municipalities', len(df_poor)
))
print(f"Low income: coef={model.params['first']:.6f}, p={model.pvalues['first']:.4f}, n={len(df_poor)}")

df_rich = df_temp[df_temp['lpib02'] > median_income]
model = smf.ols(formula, data=df_rich).fit(cov_type='HC1')
results.append(extract_results(
    model, 'robust/sample/high_income', 'robustness/sample_restrictions.md#demographic-subgroups',
    'pcorrupt', 'first', 'Full controls + FE', 'State + Lottery', 'robust', 'OLS',
    'High income municipalities', len(df_rich)
))
print(f"High income: coef={model.params['first']:.6f}, p={model.pvalues['first']:.4f}, n={len(df_rich)}")

# By judicial presence
df_comarca = df_temp[df_temp['comarca'] == 1]
if len(df_comarca) > 50:
    model = smf.ols(formula, data=df_comarca).fit(cov_type='HC1')
    results.append(extract_results(
        model, 'robust/sample/with_judiciary', 'robustness/sample_restrictions.md',
        'pcorrupt', 'first', 'Full controls + FE', 'State + Lottery', 'robust', 'OLS',
        'Municipalities with judiciary district', len(df_comarca)
    ))
    print(f"With judiciary: coef={model.params['first']:.6f}, p={model.pvalues['first']:.4f}, n={len(df_comarca)}")

df_no_comarca = df_temp[df_temp['comarca'] == 0]
if len(df_no_comarca) > 50:
    model = smf.ols(formula, data=df_no_comarca).fit(cov_type='HC1')
    results.append(extract_results(
        model, 'robust/sample/without_judiciary', 'robustness/sample_restrictions.md',
        'pcorrupt', 'first', 'Full controls + FE', 'State + Lottery', 'robust', 'OLS',
        'Municipalities without judiciary district', len(df_no_comarca)
    ))
    print(f"Without judiciary: coef={model.params['first']:.6f}, p={model.pvalues['first']:.4f}, n={len(df_no_comarca)}")

print("Sample restrictions completed")

# ============================================================================
# INTERACTION EFFECTS (Table 10)
# ============================================================================

print("\n=== Running Interaction Specifications ===")

df_temp = df_analysis.dropna(subset=controls_base)

# Interaction with comarca (judicial district)
df_temp['first_comarca'] = df_temp['first'] * df_temp['comarca']
formula = 'pcorrupt ~ first + first_comarca + ' + ' + '.join(controls_base)
model = smf.ols(formula, data=df_temp).fit(cov_type='HC1')
results.append(extract_results(
    model, 'ols/interact/comarca', 'methods/cross_sectional_ols.md#interaction-effects',
    'pcorrupt', 'first', 'Full controls + first*comarca interaction', 'State + Lottery', 'robust', 'OLS',
    'Full sample esample2==1', len(df_temp)
))
print(f"Interact comarca: first={model.params['first']:.6f}, first_comarca={model.params['first_comarca']:.6f}")

# Interaction with media presence
if 'media2' in df_temp.columns and df_temp['media2'].notna().sum() > 100:
    df_temp['first_media'] = df_temp['first'] * df_temp['media2']
    formula = 'pcorrupt ~ first + first_media + media2 + ' + ' + '.join(controls_base)
    model = smf.ols(formula, data=df_temp).fit(cov_type='HC1')
    results.append(extract_results(
        model, 'ols/interact/media', 'methods/cross_sectional_ols.md#interaction-effects',
        'pcorrupt', 'first', 'Full controls + first*media interaction', 'State + Lottery', 'robust', 'OLS',
        'Full sample esample2==1', len(df_temp)
    ))
    print(f"Interact media: first={model.params['first']:.6f}, first_media={model.params['first_media']:.6f}")

# Interaction with political competition (ENEP)
df_temp['h_ENEP2000'] = 1 / df_temp['ENEP2000']
df_temp['first_h_ENEP'] = df_temp['first'] * df_temp['h_ENEP2000']
formula = 'pcorrupt ~ first + first_h_ENEP + h_ENEP2000 + ' + ' + '.join(controls_base)
model = smf.ols(formula, data=df_temp).fit(cov_type='HC1')
results.append(extract_results(
    model, 'ols/interact/political_competition', 'methods/cross_sectional_ols.md#interaction-effects',
    'pcorrupt', 'first', 'Full controls + first*1/ENEP interaction', 'State + Lottery', 'robust', 'OLS',
    'Full sample esample2==1', len(df_temp)
))
print(f"Interact ENEP: first={model.params['first']:.6f}, first_h_ENEP={model.params['first_h_ENEP']:.6f}")

print("Interaction specifications completed")

# ============================================================================
# REGRESSION DISCONTINUITY SPECIFICATIONS (Table 6)
# ============================================================================

print("\n=== Running RD-Style Specifications ===")

# Create running variable (margin of victory)
df_rd = df_analysis.copy()

# Running variable construction from Stata code
df_rd['wm'] = np.where(df_rd['reeleito']==1, df_rd['winmargin2000'], np.nan)
df_rd['wm'] = np.where(df_rd['incumbent']==1, df_rd['winmargin2000_inclost'], df_rd['wm'])
df_rd['running'] = df_rd['wm']
df_rd['running'] = np.where(df_rd['incumbent']==1, -df_rd['wm'], df_rd['running'])

# Filter to valid running variable
df_rd = df_rd.dropna(subset=['running'])
print(f"RD sample size: {len(df_rd)}")

if len(df_rd) > 100:
    df_rd['running2'] = df_rd['running']**2
    df_rd['running3'] = df_rd['running']**3

    controls_rd = prefchar + party_dummies + munichar + ['lrec_trans'] + political_controls + sorteio_dummies + uf_dummies[:-1]
    df_rd = df_rd.dropna(subset=controls_rd)

    # Linear running variable
    formula = 'pcorrupt ~ first + running + ' + ' + '.join(controls_rd)
    model = smf.ols(formula, data=df_rd).fit(cov_type='HC1')
    results.append(extract_results(
        model, 'rd/poly/global_linear', 'methods/regression_discontinuity.md#polynomial-order',
        'pcorrupt', 'first', 'Full controls + linear running', 'State + Lottery', 'robust', 'OLS-RD style',
        'RD sample (valid margin)', len(df_rd)
    ))
    print(f"RD linear: coef={model.params['first']:.6f}, p={model.pvalues['first']:.4f}")

    # Quadratic running variable
    formula = 'pcorrupt ~ first + running + running2 + ' + ' + '.join(controls_rd)
    model = smf.ols(formula, data=df_rd).fit(cov_type='HC1')
    results.append(extract_results(
        model, 'rd/poly/global_quadratic', 'methods/regression_discontinuity.md#polynomial-order',
        'pcorrupt', 'first', 'Full controls + quadratic running', 'State + Lottery', 'robust', 'OLS-RD style',
        'RD sample (valid margin)', len(df_rd)
    ))
    print(f"RD quadratic: coef={model.params['first']:.6f}, p={model.pvalues['first']:.4f}")

    # Cubic running variable
    formula = 'pcorrupt ~ first + running + running2 + running3 + ' + ' + '.join(controls_rd)
    model = smf.ols(formula, data=df_rd).fit(cov_type='HC1')
    results.append(extract_results(
        model, 'rd/poly/global_cubic', 'methods/regression_discontinuity.md#polynomial-order',
        'pcorrupt', 'first', 'Full controls + cubic running', 'State + Lottery', 'robust', 'OLS-RD style',
        'RD sample (valid margin)', len(df_rd)
    ))
    print(f"RD cubic: coef={model.params['first']:.6f}, p={model.pvalues['first']:.4f}")

    # Different slope on each side (spline)
    df_rd['spline1'] = df_rd['first'] * df_rd['running']
    formula = 'pcorrupt ~ first + running + spline1 + ' + ' + '.join(controls_rd)
    model = smf.ols(formula, data=df_rd).fit(cov_type='HC1')
    results.append(extract_results(
        model, 'rd/poly/spline_linear', 'methods/regression_discontinuity.md#polynomial-order',
        'pcorrupt', 'first', 'Full controls + linear spline', 'State + Lottery', 'robust', 'OLS-RD style',
        'RD sample (valid margin)', len(df_rd)
    ))
    print(f"RD spline: coef={model.params['first']:.6f}, p={model.pvalues['first']:.4f}")

print("RD specifications completed")

# ============================================================================
# EXPERIENCE CONTROLS (Table 7)
# ============================================================================

print("\n=== Running Experience Specifications ===")

df_temp = df_analysis.dropna(subset=controls_base)

# With experience control
if 'exp_prefeito' in df_temp.columns:
    formula = 'pcorrupt ~ first + exp_prefeito + ' + ' + '.join(controls_base)
    model = smf.ols(formula, data=df_temp.dropna(subset=['exp_prefeito'])).fit(cov_type='HC1')
    results.append(extract_results(
        model, 'ols/controls/with_experience', 'methods/cross_sectional_ols.md#control-sets',
        'pcorrupt', 'first', 'Full controls + experience', 'State + Lottery', 'robust', 'OLS',
        'Full sample esample2==1', model.nobs
    ))
    print(f"With experience: coef={model.params['first']:.6f}, p={model.pvalues['first']:.4f}")

# Subsample: Those with experience (Table 7 style)
if 'reeleito_2004' in df_temp.columns:
    df_exp = df_temp[(df_temp['reeleito']==1) | (df_temp['reeleito_2004']==1)]
    if len(df_exp) > 50:
        formula = 'pcorrupt ~ first + ' + ' + '.join(controls_base)
        model = smf.ols(formula, data=df_exp).fit(cov_type='HC1')
        results.append(extract_results(
            model, 'robust/sample/experienced_mayors', 'robustness/sample_restrictions.md',
            'pcorrupt', 'first', 'Full controls + FE', 'State + Lottery', 'robust', 'OLS',
            'Experienced mayors only (reeleito or reeleito_2004)', len(df_exp)
        ))
        print(f"Experienced only: coef={model.params['first']:.6f}, p={model.pvalues['first']:.4f}, n={len(df_exp)}")

print("Experience specifications completed")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n=== Saving Results ===")

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
results_df.to_csv(f'{OUTPUT_DIR}/specification_results.csv', index=False)
print(f"Saved {len(results_df)} specifications to specification_results.csv")

# Summary statistics
print("\n=== Summary Statistics ===")
print(f"Total specifications: {len(results_df)}")
print(f"Positive coefficients: {(results_df['coefficient'] < 0).sum()} ({100*(results_df['coefficient'] < 0).mean():.1f}%)")  # Note: negative = less corruption for first-term
print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")
print(f"Coefficient range: [{results_df['coefficient'].min():.6f}, {results_df['coefficient'].max():.6f}]")
print(f"Median coefficient: {results_df['coefficient'].median():.6f}")
print(f"Mean coefficient: {results_df['coefficient'].mean():.6f}")

# Baseline comparison
baseline_row = results_df[results_df['spec_id'] == 'ols/fe/state'].iloc[0]
print(f"\nBaseline (state FE): coef={baseline_row['coefficient']:.6f}, p={baseline_row['p_value']:.4f}")

print("\nDone!")
