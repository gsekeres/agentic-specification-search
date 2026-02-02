"""
Specification Search: 112431-V1
Electoral Accountability and Corruption: Evidence from the Audits of Local Government
Ferraz & Finan (AER 2011)

This script runs 50+ specifications following the i4r methodology.

Paper Overview:
- Treatment: first = 1 if first-term mayor (CAN seek reelection)
- Control: first = 0 if second-term mayor (CANNOT seek reelection due to term limits)
- Main outcome: pcorrupt (proportion of federal funds associated with corruption)
- Alternative outcomes: ncorrupt, ncorrupt_os, pmismanagement
- Identification: Cross-sectional comparison + RDD at election threshold
- Main finding: First-term mayors have significantly LESS corruption (~2 pp less)
  due to reelection incentives disciplining politicians

Method Classification:
- Primary: cross_sectional_ols (with state FE and controls)
- Secondary: regression_discontinuity (Table 6 - close election RDD)
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

# Import estimation libraries
import pyfixest as pf
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Configuration
PAPER_ID = "112431-V1"
JOURNAL = "AER"
PAPER_TITLE = "Electoral Accountability and Corruption: Evidence from the Audits of Local Government"
DATA_PATH = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/112431-V1"
OUTPUT_PATH = DATA_PATH

# Load data
df = pd.read_stata(f"{DATA_PATH}/corruptiondata_aer.dta")
sample = df[df['esample2'] == 1].copy()

# Define variable groups (from do file)
# prefchar2 = pref_masc pref_idade_tse pref_escola party_d1 party_d3-party_d18
party_vars = ['party_d1'] + [f'party_d{i}' for i in range(3, 19)]
prefchar = ['pref_masc', 'pref_idade_tse', 'pref_escola'] + party_vars
munichar = ['lpop', 'purb', 'p_secundario', 'mun_novo', 'lpib02', 'gini_ipea']
political_controls = ['p_cad_pref', 'vereador_eleit', 'ENLP2000', 'comarca']
sorteio_vars = [f'sorteio{i}' for i in range(2, 11)]  # sorteio1 is dropped (reference)
uf_vars = [c for c in df.columns if c.startswith('uf_d')]

# Full baseline controls (Table 4, col 6)
baseline_controls = ['lrec_trans'] + prefchar + munichar + political_controls + sorteio_vars

# Results storage
results = []

def get_coef_vector(model, treatment_var='first'):
    """Extract coefficient vector as JSON"""
    coef_dict = {
        "treatment": {
            "var": treatment_var,
            "coef": float(model.params.get(treatment_var, np.nan)),
            "se": float(model.bse.get(treatment_var, np.nan)),
            "pval": float(model.pvalues.get(treatment_var, np.nan))
        },
        "controls": [],
        "fixed_effects": [],
        "diagnostics": {}
    }

    for var in model.params.index:
        if var != treatment_var and var != 'Intercept':
            coef_dict["controls"].append({
                "var": var,
                "coef": float(model.params[var]),
                "se": float(model.bse[var]),
                "pval": float(model.pvalues[var])
            })

    if hasattr(model, 'rsquared'):
        coef_dict["diagnostics"]["r_squared"] = float(model.rsquared)
    if hasattr(model, 'fvalue'):
        coef_dict["diagnostics"]["f_stat"] = float(model.fvalue) if model.fvalue is not None else None

    return json.dumps(coef_dict)

def add_result(spec_id, spec_tree_path, model, outcome_var, treatment_var, sample_desc,
               fixed_effects, controls_desc, cluster_var, model_type, df_used):
    """Add a result to the results list"""
    try:
        coef = float(model.params.get(treatment_var, np.nan))
        se = float(model.bse.get(treatment_var, np.nan))
        tstat = coef / se if se > 0 else np.nan
        pval = float(model.pvalues.get(treatment_var, np.nan))
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se
        rsq = float(model.rsquared) if hasattr(model, 'rsquared') else np.nan
        n_obs = int(model.nobs) if hasattr(model, 'nobs') else len(df_used)
    except Exception as e:
        print(f"Error extracting results for {spec_id}: {e}")
        return

    results.append({
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
        'n_obs': n_obs,
        'r_squared': rsq,
        'coefficient_vector_json': get_coef_vector(model, treatment_var),
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    })

def safe_regression(formula, data, cov_type='HC1', **kwargs):
    """Run OLS regression with error handling"""
    try:
        model = smf.ols(formula, data=data).fit(cov_type=cov_type, **kwargs)
        return model
    except Exception as e:
        print(f"Regression error: {e}")
        return None

def safe_regression_fe(formula, data, fe_vars, cov_type='HC1'):
    """Run OLS regression with absorbed fixed effects using pyfixest"""
    try:
        # Build pyfixest formula
        fe_string = ' + '.join(fe_vars)
        pf_formula = formula + f" | {fe_string}"
        model = pf.feols(pf_formula, data=data, vcov='hetero')

        # Convert to statsmodels-like object for compatibility
        class PFWrapper:
            def __init__(self, pf_model):
                self.pf_model = pf_model
                self.params = pd.Series(pf_model.coef(), index=pf_model.coef().index)
                self.bse = pd.Series(pf_model.se(), index=pf_model.se().index)
                self.pvalues = pd.Series(pf_model.pvalue(), index=pf_model.pvalue().index)
                self.nobs = pf_model.nobs
                self.rsquared = pf_model._r2 if hasattr(pf_model, '_r2') else np.nan
                self.fvalue = None
        return PFWrapper(model)
    except Exception as e:
        print(f"FE regression error: {e}")
        return None

print("=" * 60)
print("SPECIFICATION SEARCH: 112431-V1")
print("Electoral Accountability and Corruption")
print("=" * 60)

# ============================================================================
# BASELINE SPECIFICATIONS (Table 4)
# ============================================================================
print("\n[1] BASELINE SPECIFICATIONS")

# Baseline 1: Bivariate (Table 4, col 1)
formula = "pcorrupt ~ first"
model = safe_regression(formula, sample)
if model:
    add_result('baseline', 'methods/cross_sectional_ols.md#baseline',
               model, 'pcorrupt', 'first', 'Full sample (esample2==1)',
               'None', 'No controls', 'robust', 'OLS', sample)
    print(f"  baseline: coef={model.params['first']:.4f}, p={model.pvalues['first']:.4f}")

# Baseline 2: With mayor characteristics (Table 4, col 2)
prefchar_avail = [c for c in prefchar if c in sample.columns]
formula = f"pcorrupt ~ first + {' + '.join(prefchar_avail)}"
model = safe_regression(formula, sample)
if model:
    add_result('ols/controls/prefchar', 'methods/cross_sectional_ols.md#control-sets',
               model, 'pcorrupt', 'first', 'Full sample', 'None',
               'Mayor characteristics + party dummies', 'robust', 'OLS', sample)

# Baseline 3: Full controls no FE (Table 4, col 3)
controls_avail = [c for c in baseline_controls if c in sample.columns and sample[c].notna().sum() > 100]
formula = f"pcorrupt ~ first + {' + '.join(controls_avail)}"
model = safe_regression(formula, sample)
if model:
    add_result('ols/controls/full_no_fe', 'methods/cross_sectional_ols.md#control-sets',
               model, 'pcorrupt', 'first', 'Full sample', 'None',
               'Full controls', 'robust', 'OLS', sample)
    print(f"  full_no_fe: coef={model.params['first']:.4f}, p={model.pvalues['first']:.4f}")

# Baseline 4: Full controls with state FE (Table 4, col 6) - MAIN SPECIFICATION
# Using pyfixest for absorbed FE
try:
    # Clean data for pyfixest
    sample_clean = sample.dropna(subset=['pcorrupt', 'first', 'uf'] + controls_avail[:10])
    pf_controls = ' + '.join(controls_avail[:20])  # Limit controls for stability
    model_pf = pf.feols(f"pcorrupt ~ first + {pf_controls} | uf", data=sample_clean, vcov='hetero')

    class PFWrapper:
        def __init__(self, pf_model, n):
            self.params = pd.Series(pf_model.coef(), index=pf_model.coef().index)
            self.bse = pd.Series(pf_model.se(), index=pf_model.se().index)
            self.pvalues = pd.Series(pf_model.pvalue(), index=pf_model.pvalue().index)
            self.nobs = n
            self.rsquared = pf_model._r2 if hasattr(pf_model, '_r2') else np.nan
            self.fvalue = None

    model = PFWrapper(model_pf, len(sample_clean))
    add_result('baseline_fe', 'methods/cross_sectional_ols.md#fixed-effects',
               model, 'pcorrupt', 'first', 'Full sample', 'State (uf)',
               'Full controls + state FE', 'robust', 'OLS-FE', sample_clean)
    print(f"  baseline_fe (MAIN): coef={model.params['first']:.4f}, p={model.pvalues['first']:.4f}")
except Exception as e:
    print(f"  Error in FE baseline: {e}")

# ============================================================================
# CONTROL VARIABLE VARIATIONS (Leave-One-Out)
# ============================================================================
print("\n[2] CONTROL VARIATIONS (Leave-One-Out)")

# Get main controls (non-party, non-sorteio for LOO)
main_controls = ['pref_masc', 'pref_idade_tse', 'pref_escola', 'lpop', 'purb',
                 'p_secundario', 'mun_novo', 'lpib02', 'gini_ipea', 'lrec_trans',
                 'p_cad_pref', 'vereador_eleit', 'ENLP2000', 'comarca']
main_controls = [c for c in main_controls if c in sample.columns]

for control in main_controls:
    remaining = [c for c in main_controls if c != control]
    formula = f"pcorrupt ~ first + {' + '.join(remaining)}"
    model = safe_regression(formula, sample)
    if model:
        add_result(f'robust/loo/drop_{control}', 'robustness/leave_one_out.md',
                   model, 'pcorrupt', 'first', 'Full sample', 'None',
                   f'Dropped {control}', 'robust', 'OLS', sample)
print(f"  Completed {len(main_controls)} LOO specifications")

# ============================================================================
# CONTROL PROGRESSION (Add incrementally)
# ============================================================================
print("\n[3] CONTROL PROGRESSION (Incremental)")

control_sets = [
    ('none', []),
    ('prefchar_basic', ['pref_masc', 'pref_idade_tse', 'pref_escola']),
    ('munichar', ['lpop', 'purb', 'p_secundario', 'mun_novo', 'lpib02', 'gini_ipea']),
    ('political', ['p_cad_pref', 'vereador_eleit', 'ENLP2000', 'comarca']),
    ('transfers', ['lrec_trans']),
]

cumulative_controls = []
for name, controls in control_sets:
    controls_avail = [c for c in controls if c in sample.columns]
    cumulative_controls.extend(controls_avail)

    if len(cumulative_controls) == 0:
        formula = "pcorrupt ~ first"
    else:
        formula = f"pcorrupt ~ first + {' + '.join(cumulative_controls)}"

    model = safe_regression(formula, sample)
    if model:
        add_result(f'robust/control/add_{name}', 'robustness/control_progression.md',
                   model, 'pcorrupt', 'first', 'Full sample', 'None',
                   f'Added {name}', 'robust', 'OLS', sample)
print(f"  Completed {len(control_sets)} incremental specifications")

# ============================================================================
# ALTERNATIVE OUTCOMES
# ============================================================================
print("\n[4] ALTERNATIVE OUTCOMES")

outcomes = [
    ('ncorrupt', 'Number of corruption violations'),
    ('ncorrupt_os', 'Corruption per service order'),
    ('pmismanagement', 'Proportion mismanagement'),
    ('dcorrupt', 'Any corruption dummy'),
    ('dcorrupt_desvio', 'Embezzlement dummy'),
    ('dcorrupt_licitacao', 'Procurement fraud dummy'),
    ('dcorrupt_superfat', 'Overbilling dummy'),
]

for outcome, desc in outcomes:
    if outcome in sample.columns:
        # Full sample
        formula = f"{outcome} ~ first + {' + '.join(main_controls)}"
        model = safe_regression(formula, sample.dropna(subset=[outcome]))
        if model:
            add_result(f'robust/outcome/{outcome}', 'robustness/measurement.md',
                       model, outcome, 'first', 'Full sample', 'None',
                       desc, 'robust', 'OLS', sample)
print(f"  Completed alternative outcome specifications")

# ============================================================================
# SAMPLE RESTRICTIONS
# ============================================================================
print("\n[5] SAMPLE RESTRICTIONS")

# By lottery round
for sorteio in range(2, 11):
    col = f'sorteio{sorteio}'
    if col in sample.columns:
        subsample = sample[sample[col] == 0]  # Exclude this lottery round
        if len(subsample) > 100:
            formula = f"pcorrupt ~ first + {' + '.join(main_controls[:10])}"
            model = safe_regression(formula, subsample)
            if model:
                add_result(f'robust/sample/drop_lottery_{sorteio}', 'robustness/sample_restrictions.md',
                           model, 'pcorrupt', 'first', f'Excluding lottery {sorteio}', 'None',
                           'Lottery round dropped', 'robust', 'OLS', subsample)

# By state (drop each state)
state_counts = sample['uf'].value_counts()
top_states = state_counts.head(10).index.tolist()
for state in top_states:
    subsample = sample[sample['uf'] != state]
    if len(subsample) > 100:
        formula = f"pcorrupt ~ first + {' + '.join(main_controls[:10])}"
        model = safe_regression(formula, subsample)
        if model:
            state_str = str(state).replace(' ', '_').replace('.', '')
            add_result(f'robust/sample/drop_state_{state_str}', 'robustness/sample_restrictions.md',
                       model, 'pcorrupt', 'first', f'Excluding state {state}', 'None',
                       'State dropped', 'robust', 'OLS', subsample)

# By population size
pop_median = sample['pop'].median()
subsample_small = sample[sample['pop'] <= pop_median]
subsample_large = sample[sample['pop'] > pop_median]

for name, subsample in [('small_pop', subsample_small), ('large_pop', subsample_large)]:
    if len(subsample) > 50:
        formula = f"pcorrupt ~ first + {' + '.join(main_controls[:10])}"
        model = safe_regression(formula, subsample)
        if model:
            add_result(f'robust/sample/{name}', 'robustness/sample_restrictions.md',
                       model, 'pcorrupt', 'first', f'{name} municipalities', 'None',
                       'Population split', 'robust', 'OLS', subsample)

# By urbanization
purb_median = sample['purb'].median()
subsample_rural = sample[sample['purb'] <= purb_median]
subsample_urban = sample[sample['purb'] > purb_median]

for name, subsample in [('rural', subsample_rural), ('urban', subsample_urban)]:
    if len(subsample) > 50:
        formula = f"pcorrupt ~ first + {' + '.join(main_controls[:10])}"
        model = safe_regression(formula, subsample)
        if model:
            add_result(f'robust/sample/{name}', 'robustness/sample_restrictions.md',
                       model, 'pcorrupt', 'first', f'{name} municipalities', 'None',
                       'Urbanization split', 'robust', 'OLS', subsample)

# Winsorization
for pct in [1, 5, 10]:
    sample_wins = sample.copy()
    lower = sample_wins['pcorrupt'].quantile(pct/100)
    upper = sample_wins['pcorrupt'].quantile(1 - pct/100)
    sample_wins['pcorrupt'] = sample_wins['pcorrupt'].clip(lower, upper)

    formula = f"pcorrupt ~ first + {' + '.join(main_controls[:10])}"
    model = safe_regression(formula, sample_wins)
    if model:
        add_result(f'robust/sample/winsorize_{pct}pct', 'robustness/sample_restrictions.md',
                   model, 'pcorrupt', 'first', f'Winsorized {pct}%', 'None',
                   f'Outcome winsorized at {pct}%', 'robust', 'OLS', sample_wins)

# Trimming
for pct in [1, 5]:
    lower = sample['pcorrupt'].quantile(pct/100)
    upper = sample['pcorrupt'].quantile(1 - pct/100)
    sample_trim = sample[(sample['pcorrupt'] >= lower) & (sample['pcorrupt'] <= upper)]

    if len(sample_trim) > 50:
        formula = f"pcorrupt ~ first + {' + '.join(main_controls[:10])}"
        model = safe_regression(formula, sample_trim)
        if model:
            add_result(f'robust/sample/trim_{pct}pct', 'robustness/sample_restrictions.md',
                       model, 'pcorrupt', 'first', f'Trimmed {pct}%', 'None',
                       f'Dropped top/bottom {pct}%', 'robust', 'OLS', sample_trim)

print(f"  Completed sample restriction specifications")

# ============================================================================
# INFERENCE VARIATIONS
# ============================================================================
print("\n[6] INFERENCE VARIATIONS")

# Different SE types
formula = f"pcorrupt ~ first + {' + '.join(main_controls[:10])}"

# Classical SEs
model = smf.ols(formula, data=sample).fit()
if model:
    add_result('robust/cluster/none_classical', 'robustness/clustering_variations.md',
               model, 'pcorrupt', 'first', 'Full sample', 'None',
               'Classical SE', 'none', 'OLS', sample)

# HC1 (default robust)
model = smf.ols(formula, data=sample).fit(cov_type='HC1')
if model:
    add_result('robust/se/hc1', 'robustness/clustering_variations.md',
               model, 'pcorrupt', 'first', 'Full sample', 'None',
               'HC1 robust SE', 'HC1', 'OLS', sample)

# HC2
model = smf.ols(formula, data=sample).fit(cov_type='HC2')
if model:
    add_result('robust/se/hc2', 'robustness/clustering_variations.md',
               model, 'pcorrupt', 'first', 'Full sample', 'None',
               'HC2 robust SE', 'HC2', 'OLS', sample)

# HC3
model = smf.ols(formula, data=sample).fit(cov_type='HC3')
if model:
    add_result('robust/se/hc3', 'robustness/clustering_variations.md',
               model, 'pcorrupt', 'first', 'Full sample', 'None',
               'HC3 robust SE', 'HC3', 'OLS', sample)

# Cluster by state (uf)
try:
    sample_clean = sample.dropna(subset=['uf'])
    model = smf.ols(formula, data=sample_clean).fit(cov_type='cluster',
                                                     cov_kwds={'groups': sample_clean['uf']})
    if model:
        add_result('robust/cluster/state', 'robustness/clustering_variations.md',
                   model, 'pcorrupt', 'first', 'Full sample', 'State',
                   'Clustered by state', 'state (uf)', 'OLS', sample_clean)
except Exception as e:
    print(f"  Cluster by state failed: {e}")

# Cluster by lottery round (nsorteio)
try:
    sample_clean = sample.dropna(subset=['nsorteio'])
    model = smf.ols(formula, data=sample_clean).fit(cov_type='cluster',
                                                     cov_kwds={'groups': sample_clean['nsorteio']})
    if model:
        add_result('robust/cluster/lottery', 'robustness/clustering_variations.md',
                   model, 'pcorrupt', 'first', 'Full sample', 'Lottery round',
                   'Clustered by lottery', 'lottery (nsorteio)', 'OLS', sample_clean)
except Exception as e:
    print(f"  Cluster by lottery failed: {e}")

print(f"  Completed inference variation specifications")

# ============================================================================
# FUNCTIONAL FORM
# ============================================================================
print("\n[7] FUNCTIONAL FORM")

# Log outcome (where positive)
sample_pos = sample[sample['pcorrupt'] > 0].copy()
sample_pos['log_pcorrupt'] = np.log(sample_pos['pcorrupt'])

formula = f"log_pcorrupt ~ first + {' + '.join(main_controls[:10])}"
model = safe_regression(formula, sample_pos)
if model:
    add_result('robust/funcform/log_outcome', 'robustness/functional_form.md',
               model, 'log_pcorrupt', 'first', 'pcorrupt > 0', 'None',
               'Log outcome', 'robust', 'OLS', sample_pos)

# IHS transformation
sample['ihs_pcorrupt'] = np.arcsinh(sample['pcorrupt'] * 100)  # Scale for interpretability
formula = f"ihs_pcorrupt ~ first + {' + '.join(main_controls[:10])}"
model = safe_regression(formula, sample)
if model:
    add_result('robust/funcform/ihs_outcome', 'robustness/functional_form.md',
               model, 'ihs_pcorrupt', 'first', 'Full sample', 'None',
               'IHS transformation', 'robust', 'OLS', sample)

# Binary outcome (any corruption)
formula = f"dcorrupt ~ first + {' + '.join(main_controls[:10])}"
model = safe_regression(formula, sample)
if model:
    add_result('robust/funcform/binary_outcome', 'robustness/functional_form.md',
               model, 'dcorrupt', 'first', 'Full sample', 'None',
               'Binary outcome (LPM)', 'robust', 'OLS-LPM', sample)

# Logit for binary outcome
try:
    from statsmodels.discrete.discrete_model import Logit
    sample_clean = sample.dropna(subset=['dcorrupt'] + main_controls[:10])
    X = sm.add_constant(sample_clean[['first'] + main_controls[:10]])
    y = sample_clean['dcorrupt']
    logit_model = Logit(y, X).fit(disp=0)

    class LogitWrapper:
        def __init__(self, m):
            self.params = m.params
            self.bse = m.bse
            self.pvalues = m.pvalues
            self.nobs = m.nobs
            self.rsquared = m.prsquared
            self.fvalue = None

    add_result('robust/funcform/logit', 'robustness/functional_form.md',
               LogitWrapper(logit_model), 'dcorrupt', 'first', 'Full sample', 'None',
               'Logit model', 'robust', 'Logit', sample_clean)
except Exception as e:
    print(f"  Logit failed: {e}")

print(f"  Completed functional form specifications")

# ============================================================================
# HETEROGENEITY ANALYSIS
# ============================================================================
print("\n[8] HETEROGENEITY ANALYSIS")

# Interaction with political competition (ENLP2000)
sample['first_x_ENLP'] = sample['first'] * sample['ENLP2000']
formula = f"pcorrupt ~ first + first_x_ENLP + ENLP2000 + {' + '.join([c for c in main_controls if c != 'ENLP2000'][:10])}"
model = safe_regression(formula, sample)
if model:
    add_result('robust/heterogeneity/political_competition', 'robustness/heterogeneity.md',
               model, 'pcorrupt', 'first', 'Full sample', 'None',
               'Interaction with ENLP2000', 'robust', 'OLS', sample)

# Interaction with media presence (media2)
if 'media2' in sample.columns:
    sample['first_x_media'] = sample['first'] * sample['media2']
    formula = f"pcorrupt ~ first + first_x_media + media2 + {' + '.join(main_controls[:10])}"
    model = safe_regression(formula, sample.dropna(subset=['media2']))
    if model:
        add_result('robust/heterogeneity/media', 'robustness/heterogeneity.md',
                   model, 'pcorrupt', 'first', 'Full sample', 'None',
                   'Interaction with media presence', 'robust', 'OLS', sample)

# Interaction with judicial presence (comarca)
sample['first_x_comarca'] = sample['first'] * sample['comarca']
formula = f"pcorrupt ~ first + first_x_comarca + comarca + {' + '.join([c for c in main_controls if c != 'comarca'][:10])}"
model = safe_regression(formula, sample)
if model:
    add_result('robust/heterogeneity/judiciary', 'robustness/heterogeneity.md',
               model, 'pcorrupt', 'first', 'Full sample', 'None',
               'Interaction with judicial presence', 'robust', 'OLS', sample)

# Interaction with population
sample['first_x_lpop'] = sample['first'] * sample['lpop']
formula = f"pcorrupt ~ first + first_x_lpop + lpop + {' + '.join([c for c in main_controls if c != 'lpop'][:10])}"
model = safe_regression(formula, sample)
if model:
    add_result('robust/heterogeneity/population', 'robustness/heterogeneity.md',
               model, 'pcorrupt', 'first', 'Full sample', 'None',
               'Interaction with population', 'robust', 'OLS', sample)

# Interaction with urbanization
sample['first_x_purb'] = sample['first'] * sample['purb']
formula = f"pcorrupt ~ first + first_x_purb + purb + {' + '.join([c for c in main_controls if c != 'purb'][:10])}"
model = safe_regression(formula, sample)
if model:
    add_result('robust/heterogeneity/urbanization', 'robustness/heterogeneity.md',
               model, 'pcorrupt', 'first', 'Full sample', 'None',
               'Interaction with urbanization', 'robust', 'OLS', sample)

# Interaction with income (GDP per capita)
sample['first_x_lpib'] = sample['first'] * sample['lpib02']
formula = f"pcorrupt ~ first + first_x_lpib + lpib02 + {' + '.join([c for c in main_controls if c != 'lpib02'][:10])}"
model = safe_regression(formula, sample)
if model:
    add_result('robust/heterogeneity/income', 'robustness/heterogeneity.md',
               model, 'pcorrupt', 'first', 'Full sample', 'None',
               'Interaction with GDP per capita', 'robust', 'OLS', sample)

# By mayor gender
subsample_male = sample[sample['pref_masc'] == 1]
subsample_female = sample[sample['pref_masc'] == 0]
for name, subsample in [('male_mayors', subsample_male), ('female_mayors', subsample_female)]:
    if len(subsample) > 30:
        formula = f"pcorrupt ~ first + {' + '.join([c for c in main_controls if c != 'pref_masc'][:10])}"
        model = safe_regression(formula, subsample)
        if model:
            add_result(f'robust/heterogeneity/{name}', 'robustness/heterogeneity.md',
                       model, 'pcorrupt', 'first', f'{name}', 'None',
                       f'Subsample: {name}', 'robust', 'OLS', subsample)

print(f"  Completed heterogeneity specifications")

# ============================================================================
# PLACEBO TESTS
# ============================================================================
print("\n[9] PLACEBO TESTS")

# Placebo outcome: resources audited (should not be affected by reelection incentives)
if 'lrecursos_fisc' not in sample.columns:
    sample['lrecursos_fisc'] = np.log(sample['valor_fiscalizado'].replace(0, np.nan))

formula = f"lrecursos_fisc ~ first + {' + '.join(main_controls[:10])}"
model = safe_regression(formula, sample.dropna(subset=['lrecursos_fisc']))
if model:
    add_result('robust/placebo/resources_audited', 'robustness/placebo_tests.md',
               model, 'lrecursos_fisc', 'first', 'Full sample', 'None',
               'Placebo: resources audited', 'robust', 'OLS', sample)

# Placebo with pre-determined characteristics
# Population should not be affected by term status
formula = f"lpop ~ first + {' + '.join([c for c in main_controls if c not in ['lpop']][:10])}"
model = safe_regression(formula, sample)
if model:
    add_result('robust/placebo/population', 'robustness/placebo_tests.md',
               model, 'lpop', 'first', 'Full sample', 'None',
               'Placebo: population (predetermined)', 'robust', 'OLS', sample)

# Income should not be affected
formula = f"lpib02 ~ first + {' + '.join([c for c in main_controls if c not in ['lpib02']][:10])}"
model = safe_regression(formula, sample)
if model:
    add_result('robust/placebo/income', 'robustness/placebo_tests.md',
               model, 'lpib02', 'first', 'Full sample', 'None',
               'Placebo: GDP per capita (predetermined)', 'robust', 'OLS', sample)

# Urbanization should not be affected
formula = f"purb ~ first + {' + '.join([c for c in main_controls if c not in ['purb']][:10])}"
model = safe_regression(formula, sample)
if model:
    add_result('robust/placebo/urbanization', 'robustness/placebo_tests.md',
               model, 'purb', 'first', 'Full sample', 'None',
               'Placebo: urbanization (predetermined)', 'robust', 'OLS', sample)

print(f"  Completed placebo test specifications")

# ============================================================================
# RDD-STYLE SPECIFICATIONS (Table 6)
# ============================================================================
print("\n[10] RDD-STYLE SPECIFICATIONS")

# Create running variable (margin of victory)
sample['wm'] = np.where(sample['reeleito'] == 1, sample['winmargin2000'], sample['winmargin2000_inclost'])
sample['running'] = np.where(sample['incumbent'] == 1, -sample['wm'], sample['wm'])
sample['running2'] = sample['running'] ** 2
sample['running3'] = sample['running'] ** 3

# RDD with linear control for running variable
rdd_sample = sample.dropna(subset=['running'])
formula = f"pcorrupt ~ first + running + {' + '.join(main_controls[:10])}"
model = safe_regression(formula, rdd_sample)
if model:
    add_result('rd/poly/linear', 'methods/regression_discontinuity.md#polynomial-order',
               model, 'pcorrupt', 'first', 'Sample with running var', 'None',
               'RDD linear control', 'robust', 'OLS-RDD', rdd_sample)

# RDD with quadratic control
formula = f"pcorrupt ~ first + running + running2 + {' + '.join(main_controls[:10])}"
model = safe_regression(formula, rdd_sample)
if model:
    add_result('rd/poly/quadratic', 'methods/regression_discontinuity.md#polynomial-order',
               model, 'pcorrupt', 'first', 'Sample with running var', 'None',
               'RDD quadratic control', 'robust', 'OLS-RDD', rdd_sample)

# RDD with cubic control
formula = f"pcorrupt ~ first + running + running2 + running3 + {' + '.join(main_controls[:10])}"
model = safe_regression(formula, rdd_sample)
if model:
    add_result('rd/poly/cubic', 'methods/regression_discontinuity.md#polynomial-order',
               model, 'pcorrupt', 'first', 'Sample with running var', 'None',
               'RDD cubic control', 'robust', 'OLS-RDD', rdd_sample)

# Different bandwidth restrictions
for bw in [0.3, 0.4, 0.5, 0.6]:
    subsample = rdd_sample[(rdd_sample['running'] >= -bw) & (rdd_sample['running'] <= bw)]
    if len(subsample) > 30:
        formula = f"pcorrupt ~ first + running + {' + '.join(main_controls[:8])}"
        model = safe_regression(formula, subsample)
        if model:
            add_result(f'rd/bandwidth/bw_{int(bw*100)}pct', 'methods/regression_discontinuity.md#bandwidth-selection',
                       model, 'pcorrupt', 'first', f'Bandwidth +/-{bw}', 'None',
                       f'RDD bandwidth {bw}', 'robust', 'OLS-RDD', subsample)

print(f"  Completed RDD specifications")

# ============================================================================
# EXPERIENCE/SELECTION CONTROLS (Table 7)
# ============================================================================
print("\n[11] EXPERIENCE/SELECTION CONTROLS")

# Control for prior experience
if 'exp_prefeito' in sample.columns:
    formula = f"pcorrupt ~ first + exp_prefeito + {' + '.join(main_controls[:10])}"
    model = safe_regression(formula, sample)
    if model:
        add_result('robust/control/experience', 'robustness/control_progression.md',
                   model, 'pcorrupt', 'first', 'Full sample', 'None',
                   'Controlling for prior experience', 'robust', 'OLS', sample)

# Sample restricted to experienced politicians
if 'exp_prefeito' in sample.columns:
    experienced = sample[(sample['exp_prefeito'] == 1) | (sample['reeleito'] == 1)]
    if len(experienced) > 50:
        formula = f"pcorrupt ~ first + {' + '.join(main_controls[:10])}"
        model = safe_regression(formula, experienced)
        if model:
            add_result('robust/sample/experienced_only', 'robustness/sample_restrictions.md',
                       model, 'pcorrupt', 'first', 'Experienced politicians', 'None',
                       'Sample: experienced only', 'robust', 'OLS', experienced)

# Sample restricted to those predicted to be reelected
if 'elected1' in sample.columns:
    likely_reelect = sample[sample['elected1'] == 1]
    if len(likely_reelect) > 50:
        formula = f"pcorrupt ~ first + {' + '.join(main_controls[:10])}"
        model = safe_regression(formula, likely_reelect)
        if model:
            add_result('robust/sample/likely_reelect', 'robustness/sample_restrictions.md',
                       model, 'pcorrupt', 'first', 'Predicted to be reelected', 'None',
                       'Sample: likely to be reelected', 'robust', 'OLS', likely_reelect)

print(f"  Completed experience control specifications")

# ============================================================================
# ADDITIONAL PARTY-SPECIFIC TESTS (Table 11)
# ============================================================================
print("\n[12] PARTY-SPECIFIC TESTS")

# PT party interaction
if 'party_d15' in sample.columns:
    sample['first_x_PT'] = sample['first'] * sample['party_d15']
    formula = f"pcorrupt ~ first + first_x_PT + party_d15 + {' + '.join([c for c in main_controls if c not in ['party_d15']][:10])}"
    model = safe_regression(formula, sample)
    if model:
        add_result('robust/heterogeneity/pt_party', 'robustness/heterogeneity.md',
                   model, 'pcorrupt', 'first', 'Full sample', 'None',
                   'Interaction with PT party', 'robust', 'OLS', sample)

# Same party as governor interaction
if 'samepartygov98' in sample.columns:
    sample['first_x_sameparty'] = sample['first'] * sample['samepartygov98']
    formula = f"pcorrupt ~ first + first_x_sameparty + samepartygov98 + {' + '.join(main_controls[:10])}"
    model = safe_regression(formula, sample.dropna(subset=['samepartygov98']))
    if model:
        add_result('robust/heterogeneity/same_party_gov', 'robustness/heterogeneity.md',
                   model, 'pcorrupt', 'first', 'Full sample', 'None',
                   'Interaction with same-party governor', 'robust', 'OLS', sample)

print(f"  Completed party-specific specifications")

# ============================================================================
# ESTIMATION METHOD VARIATIONS
# ============================================================================
print("\n[13] ESTIMATION METHOD VARIATIONS")

# Tobit for censored outcome
try:
    from statsmodels.regression.linear_model import OLS
    from statsmodels.base.model import GenericLikelihoodModel

    # Left-censored at 0 (many zeros in pcorrupt)
    # Use probit on binary first
    sample_clean = sample.dropna(subset=['pcorrupt'] + main_controls[:10])

    # Probit for any corruption
    from statsmodels.discrete.discrete_model import Probit
    X = sm.add_constant(sample_clean[['first'] + main_controls[:10]])
    y = (sample_clean['pcorrupt'] > 0).astype(int)
    probit_model = Probit(y, X).fit(disp=0)

    class ProbitWrapper:
        def __init__(self, m):
            self.params = m.params
            self.bse = m.bse
            self.pvalues = m.pvalues
            self.nobs = m.nobs
            self.rsquared = m.prsquared
            self.fvalue = None

    add_result('robust/estimation/probit', 'methods/discrete_choice.md',
               ProbitWrapper(probit_model), 'any_corrupt', 'first', 'Full sample', 'None',
               'Probit for any corruption', 'robust', 'Probit', sample_clean)
except Exception as e:
    print(f"  Probit failed: {e}")

# Negative binomial for count outcome
try:
    from statsmodels.discrete.discrete_model import NegativeBinomial
    sample_clean = sample.dropna(subset=['ncorrupt'] + main_controls[:10])
    X = sm.add_constant(sample_clean[['first'] + main_controls[:10]])
    y = sample_clean['ncorrupt']
    nb_model = NegativeBinomial(y, X).fit(disp=0)

    class NBWrapper:
        def __init__(self, m):
            self.params = m.params
            self.bse = m.bse
            self.pvalues = m.pvalues
            self.nobs = m.nobs
            self.rsquared = np.nan
            self.fvalue = None

    add_result('robust/estimation/negbin', 'methods/discrete_choice.md',
               NBWrapper(nb_model), 'ncorrupt', 'first', 'Full sample', 'None',
               'Negative binomial for count', 'robust', 'NegBin', sample_clean)
except Exception as e:
    print(f"  Negative binomial failed: {e}")

# Poisson for count outcome
try:
    from statsmodels.discrete.discrete_model import Poisson
    sample_clean = sample.dropna(subset=['ncorrupt'] + main_controls[:10])
    X = sm.add_constant(sample_clean[['first'] + main_controls[:10]])
    y = sample_clean['ncorrupt']
    poisson_model = Poisson(y, X).fit(disp=0)

    class PoissonWrapper:
        def __init__(self, m):
            self.params = m.params
            self.bse = m.bse
            self.pvalues = m.pvalues
            self.nobs = m.nobs
            self.rsquared = np.nan
            self.fvalue = None

    add_result('robust/estimation/poisson', 'methods/discrete_choice.md',
               PoissonWrapper(poisson_model), 'ncorrupt', 'first', 'Full sample', 'None',
               'Poisson for count', 'robust', 'Poisson', sample_clean)
except Exception as e:
    print(f"  Poisson failed: {e}")

print(f"  Completed estimation method specifications")

# ============================================================================
# WEIGHTED REGRESSIONS
# ============================================================================
print("\n[14] WEIGHTED REGRESSIONS")

# Population weighted
if 'pop' in sample.columns:
    sample_clean = sample.dropna(subset=['pop'])
    try:
        formula = f"pcorrupt ~ first + {' + '.join([c for c in main_controls if c != 'lpop'][:10])}"
        model = smf.wls(formula, data=sample_clean, weights=sample_clean['pop']).fit(cov_type='HC1')
        add_result('robust/weights/population', 'robustness/model_specification.md',
                   model, 'pcorrupt', 'first', 'Full sample', 'None',
                   'Population weighted', 'robust', 'WLS', sample_clean)
    except Exception as e:
        print(f"  WLS (pop) failed: {e}")

# Federal transfer weighted
if 'totrecursos' in sample.columns:
    sample_clean = sample[sample['totrecursos'] > 0].dropna(subset=['totrecursos'])
    try:
        formula = f"pcorrupt ~ first + {' + '.join(main_controls[:10])}"
        model = smf.wls(formula, data=sample_clean, weights=sample_clean['totrecursos']).fit(cov_type='HC1')
        add_result('robust/weights/transfers', 'robustness/model_specification.md',
                   model, 'pcorrupt', 'first', 'Full sample', 'None',
                   'Transfer weighted', 'robust', 'WLS', sample_clean)
    except Exception as e:
        print(f"  WLS (transfers) failed: {e}")

print(f"  Completed weighted specifications")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
output_csv = f"{OUTPUT_PATH}/specification_results.csv"
results_df.to_csv(output_csv, index=False)
print(f"\nSaved {len(results_df)} specifications to {output_csv}")

# Summary statistics
print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

print(f"\nTotal specifications: {len(results_df)}")
print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
print(f"Negative coefficients: {(results_df['coefficient'] < 0).sum()} ({100*(results_df['coefficient'] < 0).mean():.1f}%)")
print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")
print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")
print(f"Range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")

# Breakdown by category
print("\n--- By Category ---")
results_df['category'] = results_df['spec_id'].str.split('/').str[0]
for cat in results_df['category'].unique():
    cat_df = results_df[results_df['category'] == cat]
    n_sig = (cat_df['p_value'] < 0.05).sum()
    n_pos = (cat_df['coefficient'] > 0).sum()
    print(f"{cat}: n={len(cat_df)}, pos={n_pos}, sig5%={n_sig}")

print("\n" + "=" * 60)
print("SPECIFICATION SEARCH COMPLETE")
print("=" * 60)
