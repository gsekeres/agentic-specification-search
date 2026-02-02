"""
Specification Search for Paper 116442-V1

Paper: "Competition and the Use of Foggy Pricing"
Authors: Miravete et al.
Journal: AEJ: Microeconomics

Main hypothesis: Competition (duopoly) leads to increased "fogginess" (tariff complexity)
Method: Difference-in-differences with market and time fixed effects
Treatment: DUOPOLY (0/1 indicator for market becoming duopoly)
Outcomes: FOGGYi (count of foggy plans), SHFOGGYi (share of foggy plans), HHFOGGYi
"""

import struct
import numpy as np
import pandas as pd
import pyfixest as pf
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA LOADING
# ============================================================================

def load_gauss_fmt(filepath, header_size=136, ncols=191, nrows=1801):
    """Load GAUSS .fmt binary matrix file"""
    with open(filepath, 'rb') as f:
        f.seek(header_size)
        data = np.frombuffer(f.read(), dtype='<f8')
    return data[:nrows * ncols].reshape(nrows, ncols)

# Variable names from GAUSS code
KNAMES = ['SCENARIO', 'MARKET', 'YEAR', 'DUOPOLY', 'WIRELINE', 'ALPHA_i', 'BETA_i',
         'GAMMA_i', 'C_i', 'LAMBDA_i', 'Z0_i', 'ALPHA_j', 'BETA_j', 'GAMMA_j',
         'C_j', 'LAMBDA_j', 'Z0_j', 'AP_PEAK', 'AP_OFFP', 'TIME', 'UNOBSERV',
         'MKT_AGE', 'LEAD', 'BUSINESS', 'COMMUTE', 'TCELLS', 'GROWTH', 'INCOME',
         'EDUCAT', 'COVERAGE', 'MEDINAGE', 'POVERTY', 'WAGE', 'ENERGY', 'OPERATE',
         'RENT', 'PRIME', 'POPULAT', 'DENSITY', 'CRIME', 'VIOLENT', 'PROPERTY',
         'SVCRIMES', 'TEMPERAT', 'RAIN', 'MULTIMKT', 'BELL', 'REGULAT', 'CORRELAT',
         'CONSPLUS', 'PROFITS', 'WELFARE', 'EXPSALE', 'EXPTARF', 'EXPRATE', 'EXPMKUP',
         'SURFACE', 'NORTH', 'WEST', 'BELLBELL', 'INDBELL', 'BELLIND', 'INDIND',
         'LIN', 'SNET', 'CONTEL', 'GTE', 'VANG', 'MCCAW', 'USWEST', 'CENTEL',
         'PACTEL', 'SWBELL', 'ALLTEL', 'AMERTECH', 'BELLATL', 'NYNEX', 'BELLSTH',
         'REST', 'OTHER', 'PREG1', 'PREG2', 'VARPVRTY', 'ECOST', 'PLANS_i',
         'PLANS_j', 'FOGGY1i', 'FOGGY2i', 'FOGGY3i', 'FOGGY4i', 'FOGGY5i', 'FOGGY6i',
         'FOGGY1j', 'FOGGY2j', 'FOGGY3j', 'FOGGY4j', 'FOGGY5j', 'FOGGY6j', 'FEE_1i',
         'PEAK_A1i', 'OFFP_A1i', 'PEAK_P1i', 'OFFP_P1i', 'FEE_2i', 'PEAK_A2i',
         'OFFP_A2i', 'PEAK_P2i', 'OFFP_P2i', 'FEE_3i', 'PEAK_A3i', 'OFFP_A3i',
         'PEAK_P3i', 'OFFP_P3i', 'FEE_4i', 'PEAK_A4i', 'OFFP_A4i', 'PEAK_P4i',
         'OFFP_P4i', 'FEE_5i', 'PEAK_A5i', 'OFFP_A5i', 'PEAK_P5i', 'OFFP_P5i',
         'FEE_6i', 'PEAK_A6i', 'OFFP_A6i', 'PEAK_P6i', 'OFFP_P6i', 'FEE_1j',
         'PEAK_A1j', 'OFFP_A1j', 'PEAK_P1j', 'OFFP_P1j', 'FEE_2j', 'PEAK_A2j',
         'OFFP_A2j', 'PEAK_P2j', 'OFFP_P2j', 'FEE_3j', 'PEAK_A3j', 'OFFP_A3j',
         'PEAK_P3j', 'OFFP_P3j', 'FEE_4j', 'PEAK_A4j', 'OFFP_A4j', 'PEAK_P4j',
         'OFFP_P4j', 'FEE_5j', 'PEAK_A5j', 'OFFP_A5j', 'PEAK_P5j', 'OFFP_P5j',
         'FEE_6j', 'PEAK_A6j', 'OFFP_A6j', 'PEAK_P6j', 'OFFP_P6j', 'NEAREND',
         'AP_PKOPK', 'PLANit', 'PLANjt', 'PLANit_1', 'PLANjt_1', 'EFFPLi',
         'FOGGYi', 'SHFOGGYi', 'HHFOGGYi', 'EFFPLj', 'FOGGYj', 'SHFOGGYj',
         'HHFOGGYj', 'PHS_PL_i', 'PHS_FG_i', 'PHS_PL_j', 'PHS_FG_j', 'POP90',
         'FAM90', 'HHOLD90', 'AGE90', 'AGE90d', 'HSIZE90', 'HSIZE90d', 'TRAV90',
         'TRAV90d', 'EDU90', 'EDU90d', 'INC90', 'INC90d', 'MEDINC', 'PCINC'][:191]

# Load data
DATA_PATH = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/116442-V1/20110032_Data/AEJ_Miravete_Data/ALLX90_0.fmt'

data = load_gauss_fmt(DATA_PATH)
df = pd.DataFrame(data, columns=KNAMES)

# Data transformations following GAUSS code
# Apply same transformations as in GAUSS code

# Select only wireline firms (WIRELINE == 1) as in the paper
# Note: The GAUSS code filters to wireline firms for main analysis
df_wire = df[df['WIRELINE'] == 1].copy()

# Keep observations with positive MKT_AGE and positive PLANS (as per GAUSS code)
df_wire = df_wire[df_wire['MKT_AGE'] > 0].copy()
df_wire = df_wire[df_wire['PLANit'] > 0].copy()

# Create additional variables
df_wire['MARKET_int'] = df_wire['MARKET'].astype(int)
df_wire['TIME_int'] = df_wire['TIME'].astype(int)

# Create FOGGYi_count = PLANit - EFFPLi (number of foggy plans)
df_wire['FOGGYi_count'] = df_wire['PLANit'] - df_wire['EFFPLi']
df_wire['FOGGYi_count'] = df_wire['FOGGYi_count'].clip(lower=0)

# Log transformations for outcome variables
df_wire['log_FOGGYi'] = np.log(df_wire['FOGGYi_count'] + 0.1)
df_wire['log_PLANS_i'] = np.log(df_wire['PLANS_i'] + 0.1)

# Create treatment timing variable
# TREATMNT is time since duopoly entry
df_wire['treat_time'] = df_wire.groupby('MARKET_int')['TIME'].transform('min') - 1
df_wire['TREATMNT'] = df_wire['TIME'] - df_wire['treat_time']
df_wire.loc[df_wire['DUOPOLY'] == 0, 'TREATMNT'] = 0

print(f"Dataset shape after filtering: {df_wire.shape}")
print(f"Number of markets: {df_wire['MARKET_int'].nunique()}")
print(f"Number of time periods: {df_wire['TIME_int'].nunique()}")
print(f"Duopoly observations: {(df_wire['DUOPOLY'] == 1).sum()}")
print(f"Monopoly observations: {(df_wire['DUOPOLY'] == 0).sum()}")

# ============================================================================
# RESULTS STORAGE
# ============================================================================

results = []

PAPER_ID = '116442-V1'
JOURNAL = 'AEJ-Microeconomics'
PAPER_TITLE = 'Competition and the Use of Foggy Pricing'

def extract_results(model, spec_id, spec_tree_path, outcome_var, treatment_var,
                   sample_desc, fixed_effects, controls_desc, cluster_var, model_type, df_used):
    """Extract results from pyfixest model"""
    try:
        coef = model.coef()[treatment_var]
        se = model.se()[treatment_var]
        tstat = model.tstat()[treatment_var]
        pval = model.pvalue()[treatment_var]
        ci = model.confint()
        ci_lower = ci.loc[treatment_var, '2.5%']
        ci_upper = ci.loc[treatment_var, '97.5%']

        # Get all coefficients for JSON
        all_coefs = model.coef()
        all_se = model.se()
        all_pval = model.pvalue()

        coef_vector = {
            'treatment': {
                'var': treatment_var,
                'coef': float(coef),
                'se': float(se),
                'pval': float(pval)
            },
            'controls': [
                {'var': v, 'coef': float(all_coefs[v]), 'se': float(all_se[v]), 'pval': float(all_pval[v])}
                for v in all_coefs.index if v != treatment_var
            ],
            'fixed_effects': fixed_effects.split(' + ') if fixed_effects else [],
            'diagnostics': {}
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
            'n_obs': int(model._N),
            'r_squared': float(model._r2) if hasattr(model, '_r2') else np.nan,
            'coefficient_vector_json': json.dumps(coef_vector),
            'sample_desc': sample_desc,
            'fixed_effects': fixed_effects,
            'controls_desc': controls_desc,
            'cluster_var': cluster_var,
            'model_type': model_type,
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
        }
    except Exception as e:
        print(f"Error extracting results for {spec_id}: {e}")
        return None

# ============================================================================
# BASELINE SPECIFICATION
# ============================================================================

# Main outcome is FOGGYi (number of foggy plans) as count data
# Paper uses Poisson PML with market and time FE, clustered SE

# Control variables from GAUSS code
CONTROL_VARS = ['AP_PEAK', 'AP_OFFP']  # Arrow-Pratt indices

print("\n" + "="*60)
print("BASELINE SPECIFICATION")
print("="*60)

# Baseline: OLS with log outcome, market + time FE, clustered by market
try:
    baseline = pf.feols(
        'log_FOGGYi ~ DUOPOLY + AP_PEAK + AP_OFFP | MARKET_int + TIME_int',
        data=df_wire,
        vcov={'CRV1': 'MARKET_int'}
    )
    result = extract_results(
        baseline, 'baseline', 'methods/difference_in_differences.md',
        'log_FOGGYi', 'DUOPOLY', 'Wireline firms, positive MKT_AGE and PLANS',
        'MARKET_int + TIME_int', 'AP_PEAK + AP_OFFP', 'MARKET_int', 'OLS-FE', df_wire
    )
    if result:
        results.append(result)
        print(f"Baseline: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"Baseline failed: {e}")

# ============================================================================
# FIXED EFFECTS VARIATIONS
# ============================================================================

print("\n" + "="*60)
print("FIXED EFFECTS VARIATIONS")
print("="*60)

fe_specs = [
    ('did/fe/unit_only', 'MARKET_int', 'Market FE only'),
    ('did/fe/time_only', 'TIME_int', 'Time FE only'),
    ('did/fe/twoway', 'MARKET_int + TIME_int', 'Two-way FE'),
    ('did/fe/none', None, 'No fixed effects (pooled OLS)')
]

for spec_id, fe, fe_desc in fe_specs:
    try:
        if fe:
            formula = f'log_FOGGYi ~ DUOPOLY + AP_PEAK + AP_OFFP | {fe}'
        else:
            formula = 'log_FOGGYi ~ DUOPOLY + AP_PEAK + AP_OFFP'

        model = pf.feols(formula, data=df_wire, vcov={'CRV1': 'MARKET_int'})
        result = extract_results(
            model, spec_id, 'methods/difference_in_differences.md#fixed-effects',
            'log_FOGGYi', 'DUOPOLY', 'Wireline firms', fe if fe else 'None',
            'AP_PEAK + AP_OFFP', 'MARKET_int', 'OLS-FE', df_wire
        )
        if result:
            results.append(result)
            print(f"{spec_id}: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
    except Exception as e:
        print(f"{spec_id} failed: {e}")

# ============================================================================
# CONTROL VARIABLE VARIATIONS
# ============================================================================

print("\n" + "="*60)
print("CONTROL VARIABLE VARIATIONS")
print("="*60)

# No controls
try:
    model = pf.feols('log_FOGGYi ~ DUOPOLY | MARKET_int + TIME_int',
                     data=df_wire, vcov={'CRV1': 'MARKET_int'})
    result = extract_results(
        model, 'did/controls/none', 'methods/difference_in_differences.md#control-sets',
        'log_FOGGYi', 'DUOPOLY', 'Wireline firms', 'MARKET_int + TIME_int',
        'None', 'MARKET_int', 'OLS-FE', df_wire
    )
    if result:
        results.append(result)
        print(f"No controls: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"No controls failed: {e}")

# Drop AP_PEAK
try:
    model = pf.feols('log_FOGGYi ~ DUOPOLY + AP_OFFP | MARKET_int + TIME_int',
                     data=df_wire, vcov={'CRV1': 'MARKET_int'})
    result = extract_results(
        model, 'robust/control/drop_AP_PEAK', 'robustness/leave_one_out.md',
        'log_FOGGYi', 'DUOPOLY', 'Wireline firms', 'MARKET_int + TIME_int',
        'AP_OFFP', 'MARKET_int', 'OLS-FE', df_wire
    )
    if result:
        results.append(result)
        print(f"Drop AP_PEAK: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"Drop AP_PEAK failed: {e}")

# Drop AP_OFFP
try:
    model = pf.feols('log_FOGGYi ~ DUOPOLY + AP_PEAK | MARKET_int + TIME_int',
                     data=df_wire, vcov={'CRV1': 'MARKET_int'})
    result = extract_results(
        model, 'robust/control/drop_AP_OFFP', 'robustness/leave_one_out.md',
        'log_FOGGYi', 'DUOPOLY', 'Wireline firms', 'MARKET_int + TIME_int',
        'AP_PEAK', 'MARKET_int', 'OLS-FE', df_wire
    )
    if result:
        results.append(result)
        print(f"Drop AP_OFFP: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"Drop AP_OFFP failed: {e}")

# Add more controls: MULTIMKT (multimarket contact)
try:
    model = pf.feols('log_FOGGYi ~ DUOPOLY + AP_PEAK + AP_OFFP + MULTIMKT | MARKET_int + TIME_int',
                     data=df_wire, vcov={'CRV1': 'MARKET_int'})
    result = extract_results(
        model, 'robust/control/add_MULTIMKT', 'robustness/leave_one_out.md',
        'log_FOGGYi', 'DUOPOLY', 'Wireline firms', 'MARKET_int + TIME_int',
        'AP_PEAK + AP_OFFP + MULTIMKT', 'MARKET_int', 'OLS-FE', df_wire
    )
    if result:
        results.append(result)
        print(f"Add MULTIMKT: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"Add MULTIMKT failed: {e}")

# Add MKT_AGE (market experience)
try:
    model = pf.feols('log_FOGGYi ~ DUOPOLY + AP_PEAK + AP_OFFP + MKT_AGE | MARKET_int + TIME_int',
                     data=df_wire, vcov={'CRV1': 'MARKET_int'})
    result = extract_results(
        model, 'robust/control/add_MKT_AGE', 'robustness/leave_one_out.md',
        'log_FOGGYi', 'DUOPOLY', 'Wireline firms', 'MARKET_int + TIME_int',
        'AP_PEAK + AP_OFFP + MKT_AGE', 'MARKET_int', 'OLS-FE', df_wire
    )
    if result:
        results.append(result)
        print(f"Add MKT_AGE: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"Add MKT_AGE failed: {e}")

# Full controls
FULL_CONTROLS = ['AP_PEAK', 'AP_OFFP', 'MULTIMKT', 'MKT_AGE', 'LEAD']
try:
    ctrl_str = ' + '.join(FULL_CONTROLS)
    model = pf.feols(f'log_FOGGYi ~ DUOPOLY + {ctrl_str} | MARKET_int + TIME_int',
                     data=df_wire, vcov={'CRV1': 'MARKET_int'})
    result = extract_results(
        model, 'did/controls/full', 'methods/difference_in_differences.md#control-sets',
        'log_FOGGYi', 'DUOPOLY', 'Wireline firms', 'MARKET_int + TIME_int',
        ctrl_str, 'MARKET_int', 'OLS-FE', df_wire
    )
    if result:
        results.append(result)
        print(f"Full controls: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"Full controls failed: {e}")

# ============================================================================
# ALTERNATIVE OUTCOME VARIABLES
# ============================================================================

print("\n" + "="*60)
print("ALTERNATIVE OUTCOME VARIABLES")
print("="*60)

# SHFOGGYi (share of foggy plans)
try:
    model = pf.feols('SHFOGGYi ~ DUOPOLY + AP_PEAK + AP_OFFP | MARKET_int + TIME_int',
                     data=df_wire, vcov={'CRV1': 'MARKET_int'})
    result = extract_results(
        model, 'robust/outcome/SHFOGGYi', 'robustness/functional_form.md',
        'SHFOGGYi', 'DUOPOLY', 'Wireline firms', 'MARKET_int + TIME_int',
        'AP_PEAK + AP_OFFP', 'MARKET_int', 'OLS-FE', df_wire
    )
    if result:
        results.append(result)
        print(f"SHFOGGYi: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"SHFOGGYi failed: {e}")

# HHFOGGYi (HHI of foggy plans)
try:
    model = pf.feols('HHFOGGYi ~ DUOPOLY + AP_PEAK + AP_OFFP | MARKET_int + TIME_int',
                     data=df_wire, vcov={'CRV1': 'MARKET_int'})
    result = extract_results(
        model, 'robust/outcome/HHFOGGYi', 'robustness/functional_form.md',
        'HHFOGGYi', 'DUOPOLY', 'Wireline firms', 'MARKET_int + TIME_int',
        'AP_PEAK + AP_OFFP', 'MARKET_int', 'OLS-FE', df_wire
    )
    if result:
        results.append(result)
        print(f"HHFOGGYi: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"HHFOGGYi failed: {e}")

# FOGGYi_count in levels
try:
    model = pf.feols('FOGGYi_count ~ DUOPOLY + AP_PEAK + AP_OFFP | MARKET_int + TIME_int',
                     data=df_wire, vcov={'CRV1': 'MARKET_int'})
    result = extract_results(
        model, 'robust/outcome/FOGGYi_levels', 'robustness/functional_form.md',
        'FOGGYi_count', 'DUOPOLY', 'Wireline firms', 'MARKET_int + TIME_int',
        'AP_PEAK + AP_OFFP', 'MARKET_int', 'OLS-FE', df_wire
    )
    if result:
        results.append(result)
        print(f"FOGGYi levels: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"FOGGYi levels failed: {e}")

# PLANS_i (total plans - placebo-like)
try:
    model = pf.feols('log_PLANS_i ~ DUOPOLY + AP_PEAK + AP_OFFP | MARKET_int + TIME_int',
                     data=df_wire, vcov={'CRV1': 'MARKET_int'})
    result = extract_results(
        model, 'robust/outcome/log_PLANS_i', 'robustness/functional_form.md',
        'log_PLANS_i', 'DUOPOLY', 'Wireline firms', 'MARKET_int + TIME_int',
        'AP_PEAK + AP_OFFP', 'MARKET_int', 'OLS-FE', df_wire
    )
    if result:
        results.append(result)
        print(f"log_PLANS_i: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"log_PLANS_i failed: {e}")

# EFFPLi (effective/non-dominated plans)
try:
    df_wire['log_EFFPLi'] = np.log(df_wire['EFFPLi'] + 0.1)
    model = pf.feols('log_EFFPLi ~ DUOPOLY + AP_PEAK + AP_OFFP | MARKET_int + TIME_int',
                     data=df_wire, vcov={'CRV1': 'MARKET_int'})
    result = extract_results(
        model, 'robust/outcome/log_EFFPLi', 'robustness/functional_form.md',
        'log_EFFPLi', 'DUOPOLY', 'Wireline firms', 'MARKET_int + TIME_int',
        'AP_PEAK + AP_OFFP', 'MARKET_int', 'OLS-FE', df_wire
    )
    if result:
        results.append(result)
        print(f"log_EFFPLi: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"log_EFFPLi failed: {e}")

# ============================================================================
# CLUSTERING VARIATIONS
# ============================================================================

print("\n" + "="*60)
print("CLUSTERING VARIATIONS")
print("="*60)

# Robust SE (no clustering)
try:
    model = pf.feols('log_FOGGYi ~ DUOPOLY + AP_PEAK + AP_OFFP | MARKET_int + TIME_int',
                     data=df_wire, vcov='hetero')
    result = extract_results(
        model, 'robust/cluster/none', 'robustness/clustering_variations.md',
        'log_FOGGYi', 'DUOPOLY', 'Wireline firms', 'MARKET_int + TIME_int',
        'AP_PEAK + AP_OFFP', 'None (robust)', 'OLS-FE', df_wire
    )
    if result:
        results.append(result)
        print(f"Robust SE: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"Robust SE failed: {e}")

# Cluster by time
try:
    model = pf.feols('log_FOGGYi ~ DUOPOLY + AP_PEAK + AP_OFFP | MARKET_int + TIME_int',
                     data=df_wire, vcov={'CRV1': 'TIME_int'})
    result = extract_results(
        model, 'robust/cluster/time', 'robustness/clustering_variations.md',
        'log_FOGGYi', 'DUOPOLY', 'Wireline firms', 'MARKET_int + TIME_int',
        'AP_PEAK + AP_OFFP', 'TIME_int', 'OLS-FE', df_wire
    )
    if result:
        results.append(result)
        print(f"Cluster time: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"Cluster time failed: {e}")

# Two-way clustering - pyfixest requires special syntax
try:
    model = pf.feols('log_FOGGYi ~ DUOPOLY + AP_PEAK + AP_OFFP | MARKET_int + TIME_int',
                     data=df_wire, vcov='twoway')
    result = extract_results(
        model, 'robust/cluster/unit_time', 'robustness/clustering_variations.md',
        'log_FOGGYi', 'DUOPOLY', 'Wireline firms', 'MARKET_int + TIME_int',
        'AP_PEAK + AP_OFFP', 'MARKET_int + TIME_int', 'OLS-FE', df_wire
    )
    if result:
        results.append(result)
        print(f"Two-way cluster: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"Two-way cluster failed: {e}")

# ============================================================================
# SAMPLE RESTRICTIONS
# ============================================================================

print("\n" + "="*60)
print("SAMPLE RESTRICTIONS")
print("="*60)

# Early period (TIME <= 15)
try:
    df_early = df_wire[df_wire['TIME'] <= 15]
    if len(df_early) > 50:
        model = pf.feols('log_FOGGYi ~ DUOPOLY + AP_PEAK + AP_OFFP | MARKET_int + TIME_int',
                         data=df_early, vcov={'CRV1': 'MARKET_int'})
        result = extract_results(
            model, 'robust/sample/early_period', 'robustness/sample_restrictions.md',
            'log_FOGGYi', 'DUOPOLY', 'Wireline firms, TIME <= 15', 'MARKET_int + TIME_int',
            'AP_PEAK + AP_OFFP', 'MARKET_int', 'OLS-FE', df_early
        )
        if result:
            results.append(result)
            print(f"Early period: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"Early period failed: {e}")

# Late period (TIME > 15)
try:
    df_late = df_wire[df_wire['TIME'] > 15]
    if len(df_late) > 50:
        model = pf.feols('log_FOGGYi ~ DUOPOLY + AP_PEAK + AP_OFFP | MARKET_int + TIME_int',
                         data=df_late, vcov={'CRV1': 'MARKET_int'})
        result = extract_results(
            model, 'robust/sample/late_period', 'robustness/sample_restrictions.md',
            'log_FOGGYi', 'DUOPOLY', 'Wireline firms, TIME > 15', 'MARKET_int + TIME_int',
            'AP_PEAK + AP_OFFP', 'MARKET_int', 'OLS-FE', df_late
        )
        if result:
            results.append(result)
            print(f"Late period: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"Late period failed: {e}")

# Trim outliers - top/bottom 5% of outcome
try:
    p5, p95 = df_wire['log_FOGGYi'].quantile([0.05, 0.95])
    df_trim = df_wire[(df_wire['log_FOGGYi'] >= p5) & (df_wire['log_FOGGYi'] <= p95)]
    if len(df_trim) > 50:
        model = pf.feols('log_FOGGYi ~ DUOPOLY + AP_PEAK + AP_OFFP | MARKET_int + TIME_int',
                         data=df_trim, vcov={'CRV1': 'MARKET_int'})
        result = extract_results(
            model, 'robust/sample/trim_5pct', 'robustness/sample_restrictions.md',
            'log_FOGGYi', 'DUOPOLY', 'Wireline firms, trimmed 5%', 'MARKET_int + TIME_int',
            'AP_PEAK + AP_OFFP', 'MARKET_int', 'OLS-FE', df_trim
        )
        if result:
            results.append(result)
            print(f"Trim 5%: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"Trim 5% failed: {e}")

# Winsorize outcome at 1%
try:
    df_wins = df_wire.copy()
    p1, p99 = df_wins['log_FOGGYi'].quantile([0.01, 0.99])
    df_wins['log_FOGGYi_wins'] = df_wins['log_FOGGYi'].clip(lower=p1, upper=p99)
    model = pf.feols('log_FOGGYi_wins ~ DUOPOLY + AP_PEAK + AP_OFFP | MARKET_int + TIME_int',
                     data=df_wins, vcov={'CRV1': 'MARKET_int'})
    result = extract_results(
        model, 'robust/sample/winsor_1pct', 'robustness/sample_restrictions.md',
        'log_FOGGYi_wins', 'DUOPOLY', 'Wireline firms, winsorized 1%', 'MARKET_int + TIME_int',
        'AP_PEAK + AP_OFFP', 'MARKET_int', 'OLS-FE', df_wins
    )
    if result:
        results.append(result)
        print(f"Winsorize 1%: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"Winsorize 1% failed: {e}")

# Drop first year
try:
    min_time = df_wire['TIME'].min()
    df_no_first = df_wire[df_wire['TIME'] > min_time]
    if len(df_no_first) > 50:
        model = pf.feols('log_FOGGYi ~ DUOPOLY + AP_PEAK + AP_OFFP | MARKET_int + TIME_int',
                         data=df_no_first, vcov={'CRV1': 'MARKET_int'})
        result = extract_results(
            model, 'robust/sample/exclude_first_year', 'robustness/sample_restrictions.md',
            'log_FOGGYi', 'DUOPOLY', 'Wireline firms, excl first year', 'MARKET_int + TIME_int',
            'AP_PEAK + AP_OFFP', 'MARKET_int', 'OLS-FE', df_no_first
        )
        if result:
            results.append(result)
            print(f"Excl first year: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"Excl first year failed: {e}")

# Drop last year
try:
    max_time = df_wire['TIME'].max()
    df_no_last = df_wire[df_wire['TIME'] < max_time]
    if len(df_no_last) > 50:
        model = pf.feols('log_FOGGYi ~ DUOPOLY + AP_PEAK + AP_OFFP | MARKET_int + TIME_int',
                         data=df_no_last, vcov={'CRV1': 'MARKET_int'})
        result = extract_results(
            model, 'robust/sample/exclude_last_year', 'robustness/sample_restrictions.md',
            'log_FOGGYi', 'DUOPOLY', 'Wireline firms, excl last year', 'MARKET_int + TIME_int',
            'AP_PEAK + AP_OFFP', 'MARKET_int', 'OLS-FE', df_no_last
        )
        if result:
            results.append(result)
            print(f"Excl last year: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"Excl last year failed: {e}")

# Drop specific time periods
for drop_time in [5, 10, 15, 20, 25]:
    try:
        df_drop = df_wire[df_wire['TIME'] != drop_time]
        if len(df_drop) > 50:
            model = pf.feols('log_FOGGYi ~ DUOPOLY + AP_PEAK + AP_OFFP | MARKET_int + TIME_int',
                             data=df_drop, vcov={'CRV1': 'MARKET_int'})
            result = extract_results(
                model, f'robust/sample/drop_time_{drop_time}', 'robustness/sample_restrictions.md',
                'log_FOGGYi', 'DUOPOLY', f'Wireline firms, excl TIME={drop_time}', 'MARKET_int + TIME_int',
                'AP_PEAK + AP_OFFP', 'MARKET_int', 'OLS-FE', df_drop
            )
            if result:
                results.append(result)
                print(f"Drop TIME={drop_time}: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
    except Exception as e:
        print(f"Drop TIME={drop_time} failed: {e}")

# Markets with min observations
for min_obs in [3, 5, 10]:
    try:
        market_counts = df_wire.groupby('MARKET_int').size()
        valid_markets = market_counts[market_counts >= min_obs].index
        df_sub = df_wire[df_wire['MARKET_int'].isin(valid_markets)]
        if len(df_sub) > 50 and df_sub['MARKET_int'].nunique() > 10:
            model = pf.feols('log_FOGGYi ~ DUOPOLY + AP_PEAK + AP_OFFP | MARKET_int + TIME_int',
                             data=df_sub, vcov={'CRV1': 'MARKET_int'})
            result = extract_results(
                model, f'robust/sample/min_obs_{min_obs}', 'robustness/sample_restrictions.md',
                'log_FOGGYi', 'DUOPOLY', f'Wireline firms, markets with >={min_obs} obs', 'MARKET_int + TIME_int',
                'AP_PEAK + AP_OFFP', 'MARKET_int', 'OLS-FE', df_sub
            )
            if result:
                results.append(result)
                print(f"Min obs {min_obs}: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
    except Exception as e:
        print(f"Min obs {min_obs} failed: {e}")

# ============================================================================
# FUNCTIONAL FORM VARIATIONS
# ============================================================================

print("\n" + "="*60)
print("FUNCTIONAL FORM VARIATIONS")
print("="*60)

# IHS transformation
try:
    df_wire['ihs_FOGGYi'] = np.arcsinh(df_wire['FOGGYi_count'])
    model = pf.feols('ihs_FOGGYi ~ DUOPOLY + AP_PEAK + AP_OFFP | MARKET_int + TIME_int',
                     data=df_wire, vcov={'CRV1': 'MARKET_int'})
    result = extract_results(
        model, 'robust/funcform/ihs_outcome', 'robustness/functional_form.md',
        'ihs_FOGGYi', 'DUOPOLY', 'Wireline firms', 'MARKET_int + TIME_int',
        'AP_PEAK + AP_OFFP', 'MARKET_int', 'OLS-FE', df_wire
    )
    if result:
        results.append(result)
        print(f"IHS: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"IHS failed: {e}")

# Levels (no transformation)
try:
    model = pf.feols('FOGGYi_count ~ DUOPOLY + AP_PEAK + AP_OFFP | MARKET_int + TIME_int',
                     data=df_wire, vcov={'CRV1': 'MARKET_int'})
    result = extract_results(
        model, 'robust/funcform/levels', 'robustness/functional_form.md',
        'FOGGYi_count', 'DUOPOLY', 'Wireline firms', 'MARKET_int + TIME_int',
        'AP_PEAK + AP_OFFP', 'MARKET_int', 'OLS-FE', df_wire
    )
    if result:
        results.append(result)
        print(f"Levels: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"Levels failed: {e}")

# Squared treatment (nonlinear effect)
try:
    df_wire['DUOPOLY_sq'] = df_wire['DUOPOLY'] ** 2
    model = pf.feols('log_FOGGYi ~ DUOPOLY + DUOPOLY_sq + AP_PEAK + AP_OFFP | MARKET_int + TIME_int',
                     data=df_wire, vcov={'CRV1': 'MARKET_int'})
    result = extract_results(
        model, 'robust/funcform/quadratic_treatment', 'robustness/functional_form.md',
        'log_FOGGYi', 'DUOPOLY', 'Wireline firms', 'MARKET_int + TIME_int',
        'AP_PEAK + AP_OFFP + DUOPOLY_sq', 'MARKET_int', 'OLS-FE', df_wire
    )
    if result:
        results.append(result)
        print(f"Quadratic: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"Quadratic failed: {e}")

# ============================================================================
# HETEROGENEITY ANALYSES
# ============================================================================

print("\n" + "="*60)
print("HETEROGENEITY ANALYSES")
print("="*60)

# By BELL status (Bell company vs non-Bell)
try:
    df_wire['DUOPOLY_x_BELL'] = df_wire['DUOPOLY'] * df_wire['BELL']
    model = pf.feols('log_FOGGYi ~ DUOPOLY + DUOPOLY_x_BELL + BELL + AP_PEAK + AP_OFFP | MARKET_int + TIME_int',
                     data=df_wire, vcov={'CRV1': 'MARKET_int'})
    result = extract_results(
        model, 'robust/heterogeneity/BELL', 'robustness/heterogeneity.md',
        'log_FOGGYi', 'DUOPOLY', 'Wireline firms', 'MARKET_int + TIME_int',
        'AP_PEAK + AP_OFFP + BELL + DUOPOLY_x_BELL', 'MARKET_int', 'OLS-FE', df_wire
    )
    if result:
        results.append(result)
        print(f"Heterog BELL: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"Heterog BELL failed: {e}")

# By REGULAT (regulation indicator)
try:
    df_wire['DUOPOLY_x_REGULAT'] = df_wire['DUOPOLY'] * df_wire['REGULAT']
    model = pf.feols('log_FOGGYi ~ DUOPOLY + DUOPOLY_x_REGULAT + REGULAT + AP_PEAK + AP_OFFP | MARKET_int + TIME_int',
                     data=df_wire, vcov={'CRV1': 'MARKET_int'})
    result = extract_results(
        model, 'robust/heterogeneity/REGULAT', 'robustness/heterogeneity.md',
        'log_FOGGYi', 'DUOPOLY', 'Wireline firms', 'MARKET_int + TIME_int',
        'AP_PEAK + AP_OFFP + REGULAT + DUOPOLY_x_REGULAT', 'MARKET_int', 'OLS-FE', df_wire
    )
    if result:
        results.append(result)
        print(f"Heterog REGULAT: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"Heterog REGULAT failed: {e}")

# By MULTIMKT (multimarket contact)
try:
    df_wire['MULTIMKT_high'] = (df_wire['MULTIMKT'] > df_wire['MULTIMKT'].median()).astype(int)
    df_wire['DUOPOLY_x_MULTIMKT_high'] = df_wire['DUOPOLY'] * df_wire['MULTIMKT_high']
    model = pf.feols('log_FOGGYi ~ DUOPOLY + DUOPOLY_x_MULTIMKT_high + MULTIMKT_high + AP_PEAK + AP_OFFP | MARKET_int + TIME_int',
                     data=df_wire, vcov={'CRV1': 'MARKET_int'})
    result = extract_results(
        model, 'robust/heterogeneity/MULTIMKT', 'robustness/heterogeneity.md',
        'log_FOGGYi', 'DUOPOLY', 'Wireline firms', 'MARKET_int + TIME_int',
        'AP_PEAK + AP_OFFP + MULTIMKT_high + DUOPOLY_x_MULTIMKT_high', 'MARKET_int', 'OLS-FE', df_wire
    )
    if result:
        results.append(result)
        print(f"Heterog MULTIMKT: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"Heterog MULTIMKT failed: {e}")

# By market size (POPULAT)
try:
    df_wire['POPULAT_high'] = (df_wire['POPULAT'] > df_wire['POPULAT'].median()).astype(int)
    df_wire['DUOPOLY_x_POPULAT_high'] = df_wire['DUOPOLY'] * df_wire['POPULAT_high']
    model = pf.feols('log_FOGGYi ~ DUOPOLY + DUOPOLY_x_POPULAT_high + POPULAT_high + AP_PEAK + AP_OFFP | MARKET_int + TIME_int',
                     data=df_wire, vcov={'CRV1': 'MARKET_int'})
    result = extract_results(
        model, 'robust/heterogeneity/POPULAT', 'robustness/heterogeneity.md',
        'log_FOGGYi', 'DUOPOLY', 'Wireline firms', 'MARKET_int + TIME_int',
        'AP_PEAK + AP_OFFP + POPULAT_high + DUOPOLY_x_POPULAT_high', 'MARKET_int', 'OLS-FE', df_wire
    )
    if result:
        results.append(result)
        print(f"Heterog POPULAT: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"Heterog POPULAT failed: {e}")

# By income (INCOME)
try:
    df_wire['INCOME_high'] = (df_wire['INCOME'] > df_wire['INCOME'].median()).astype(int)
    df_wire['DUOPOLY_x_INCOME_high'] = df_wire['DUOPOLY'] * df_wire['INCOME_high']
    model = pf.feols('log_FOGGYi ~ DUOPOLY + DUOPOLY_x_INCOME_high + INCOME_high + AP_PEAK + AP_OFFP | MARKET_int + TIME_int',
                     data=df_wire, vcov={'CRV1': 'MARKET_int'})
    result = extract_results(
        model, 'robust/heterogeneity/INCOME', 'robustness/heterogeneity.md',
        'log_FOGGYi', 'DUOPOLY', 'Wireline firms', 'MARKET_int + TIME_int',
        'AP_PEAK + AP_OFFP + INCOME_high + DUOPOLY_x_INCOME_high', 'MARKET_int', 'OLS-FE', df_wire
    )
    if result:
        results.append(result)
        print(f"Heterog INCOME: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"Heterog INCOME failed: {e}")

# By MKT_AGE (market experience)
try:
    df_wire['MKT_AGE_high'] = (df_wire['MKT_AGE'] > df_wire['MKT_AGE'].median()).astype(int)
    df_wire['DUOPOLY_x_MKT_AGE_high'] = df_wire['DUOPOLY'] * df_wire['MKT_AGE_high']
    model = pf.feols('log_FOGGYi ~ DUOPOLY + DUOPOLY_x_MKT_AGE_high + MKT_AGE_high + AP_PEAK + AP_OFFP | MARKET_int + TIME_int',
                     data=df_wire, vcov={'CRV1': 'MARKET_int'})
    result = extract_results(
        model, 'robust/heterogeneity/MKT_AGE', 'robustness/heterogeneity.md',
        'log_FOGGYi', 'DUOPOLY', 'Wireline firms', 'MARKET_int + TIME_int',
        'AP_PEAK + AP_OFFP + MKT_AGE_high + DUOPOLY_x_MKT_AGE_high', 'MARKET_int', 'OLS-FE', df_wire
    )
    if result:
        results.append(result)
        print(f"Heterog MKT_AGE: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"Heterog MKT_AGE failed: {e}")

# By geographic regions
try:
    df_wire['DUOPOLY_x_NORTH'] = df_wire['DUOPOLY'] * df_wire['NORTH']
    model = pf.feols('log_FOGGYi ~ DUOPOLY + DUOPOLY_x_NORTH + NORTH + AP_PEAK + AP_OFFP | MARKET_int + TIME_int',
                     data=df_wire, vcov={'CRV1': 'MARKET_int'})
    result = extract_results(
        model, 'robust/heterogeneity/NORTH', 'robustness/heterogeneity.md',
        'log_FOGGYi', 'DUOPOLY', 'Wireline firms', 'MARKET_int + TIME_int',
        'AP_PEAK + AP_OFFP + NORTH + DUOPOLY_x_NORTH', 'MARKET_int', 'OLS-FE', df_wire
    )
    if result:
        results.append(result)
        print(f"Heterog NORTH: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"Heterog NORTH failed: {e}")

try:
    df_wire['DUOPOLY_x_WEST'] = df_wire['DUOPOLY'] * df_wire['WEST']
    model = pf.feols('log_FOGGYi ~ DUOPOLY + DUOPOLY_x_WEST + WEST + AP_PEAK + AP_OFFP | MARKET_int + TIME_int',
                     data=df_wire, vcov={'CRV1': 'MARKET_int'})
    result = extract_results(
        model, 'robust/heterogeneity/WEST', 'robustness/heterogeneity.md',
        'log_FOGGYi', 'DUOPOLY', 'Wireline firms', 'MARKET_int + TIME_int',
        'AP_PEAK + AP_OFFP + WEST + DUOPOLY_x_WEST', 'MARKET_int', 'OLS-FE', df_wire
    )
    if result:
        results.append(result)
        print(f"Heterog WEST: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"Heterog WEST failed: {e}")

# ============================================================================
# PLACEBO TESTS
# ============================================================================

print("\n" + "="*60)
print("PLACEBO TESTS")
print("="*60)

# Pre-treatment period only (DUOPOLY == 0)
try:
    df_pre = df_wire[df_wire['DUOPOLY'] == 0]
    # Create fake treatment (early vs late monopoly)
    df_pre['FAKE_TREAT'] = (df_pre['TIME'] > df_pre['TIME'].median()).astype(int)
    if len(df_pre) > 50:
        model = pf.feols('log_FOGGYi ~ FAKE_TREAT + AP_PEAK + AP_OFFP | MARKET_int + TIME_int',
                         data=df_pre, vcov={'CRV1': 'MARKET_int'})
        result = extract_results(
            model, 'robust/placebo/pre_treatment', 'robustness/placebo_tests.md',
            'log_FOGGYi', 'FAKE_TREAT', 'Monopoly period only', 'MARKET_int + TIME_int',
            'AP_PEAK + AP_OFFP', 'MARKET_int', 'OLS-FE', df_pre
        )
        if result:
            results.append(result)
            print(f"Pre-treatment placebo: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"Pre-treatment placebo failed: {e}")

# Fake timing - shift treatment timing by 2 periods
try:
    df_fake = df_wire.copy()
    df_fake['DUOPOLY_FAKE'] = df_fake.groupby('MARKET_int')['DUOPOLY'].shift(2).fillna(0)
    model = pf.feols('log_FOGGYi ~ DUOPOLY_FAKE + AP_PEAK + AP_OFFP | MARKET_int + TIME_int',
                     data=df_fake, vcov={'CRV1': 'MARKET_int'})
    result = extract_results(
        model, 'robust/placebo/fake_timing_lead2', 'robustness/placebo_tests.md',
        'log_FOGGYi', 'DUOPOLY_FAKE', 'Wireline firms, treatment shifted +2', 'MARKET_int + TIME_int',
        'AP_PEAK + AP_OFFP', 'MARKET_int', 'OLS-FE', df_fake
    )
    if result:
        results.append(result)
        print(f"Fake timing +2: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"Fake timing +2 failed: {e}")

# Fake treatment - random assignment
try:
    np.random.seed(42)
    df_rand = df_wire.copy()
    markets = df_rand['MARKET_int'].unique()
    np.random.shuffle(markets)
    fake_treated = markets[:len(markets)//2]
    df_rand['DUOPOLY_RANDOM'] = df_rand['MARKET_int'].isin(fake_treated).astype(int)
    model = pf.feols('log_FOGGYi ~ DUOPOLY_RANDOM + AP_PEAK + AP_OFFP | MARKET_int + TIME_int',
                     data=df_rand, vcov={'CRV1': 'MARKET_int'})
    result = extract_results(
        model, 'robust/placebo/random_treatment', 'robustness/placebo_tests.md',
        'log_FOGGYi', 'DUOPOLY_RANDOM', 'Wireline firms, random treatment', 'MARKET_int + TIME_int',
        'AP_PEAK + AP_OFFP', 'MARKET_int', 'OLS-FE', df_rand
    )
    if result:
        results.append(result)
        print(f"Random treatment: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"Random treatment failed: {e}")

# ============================================================================
# DYNAMIC TREATMENT EFFECTS (EVENT STUDY)
# ============================================================================

print("\n" + "="*60)
print("DYNAMIC TREATMENT EFFECTS")
print("="*60)

# Create event time indicators
try:
    # Event time relative to duopoly entry
    df_event = df_wire.copy()

    # Find first duopoly period for each market
    first_duo = df_event[df_event['DUOPOLY'] == 1].groupby('MARKET_int')['TIME'].min()
    df_event['first_duopoly'] = df_event['MARKET_int'].map(first_duo)
    df_event['event_time'] = df_event['TIME'] - df_event['first_duopoly']

    # For never-treated, set event_time to large negative
    df_event.loc[df_event['first_duopoly'].isna(), 'event_time'] = -99
    df_event['first_duopoly'] = df_event['first_duopoly'].fillna(999)

    # Create indicators for event time (binned)
    for lag in [1, 2, 3, 4, 5, 6]:
        df_event[f'post_{lag}'] = (df_event['event_time'] == lag).astype(int)
    df_event['post_7plus'] = (df_event['event_time'] >= 7).astype(int)

    # Event study regression
    post_vars = ' + '.join([f'post_{i}' for i in range(1, 7)] + ['post_7plus'])
    model = pf.feols(f'log_FOGGYi ~ {post_vars} + AP_PEAK + AP_OFFP | MARKET_int + TIME_int',
                     data=df_event, vcov={'CRV1': 'MARKET_int'})

    # Extract each lag coefficient
    for lag in [1, 2, 3, 4, 5, 6]:
        try:
            coef = model.coef()[f'post_{lag}']
            se = model.se()[f'post_{lag}']
            pval = model.pvalue()[f'post_{lag}']
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': f'did/dynamic/lag_{lag}',
                'spec_tree_path': 'methods/difference_in_differences.md#dynamic-effects',
                'outcome_var': 'log_FOGGYi',
                'treatment_var': f'post_{lag}',
                'coefficient': float(coef),
                'std_error': float(se),
                't_stat': float(coef/se),
                'p_value': float(pval),
                'ci_lower': float(coef - 1.96*se),
                'ci_upper': float(coef + 1.96*se),
                'n_obs': int(model._N),
                'r_squared': float(model._r2) if hasattr(model, '_r2') else np.nan,
                'coefficient_vector_json': json.dumps({'treatment': {'var': f'post_{lag}', 'coef': float(coef), 'se': float(se), 'pval': float(pval)}}),
                'sample_desc': 'Wireline firms',
                'fixed_effects': 'MARKET_int + TIME_int',
                'controls_desc': 'AP_PEAK + AP_OFFP',
                'cluster_var': 'MARKET_int',
                'model_type': 'OLS-FE',
                'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
            })
            print(f"Lag {lag}: coef={coef:.4f}, se={se:.4f}, p={pval:.4f}")
        except Exception as e:
            print(f"Lag {lag} extraction failed: {e}")
except Exception as e:
    print(f"Event study failed: {e}")

# ============================================================================
# ESTIMATION METHOD VARIATIONS
# ============================================================================

print("\n" + "="*60)
print("ESTIMATION METHOD VARIATIONS")
print("="*60)

# First differences
try:
    df_fd = df_wire.sort_values(['MARKET_int', 'TIME']).copy()
    for var in ['log_FOGGYi', 'DUOPOLY', 'AP_PEAK', 'AP_OFFP']:
        df_fd[f'd_{var}'] = df_fd.groupby('MARKET_int')[var].diff()
    df_fd = df_fd.dropna(subset=['d_log_FOGGYi', 'd_DUOPOLY', 'd_AP_PEAK', 'd_AP_OFFP'])

    if len(df_fd) > 50:
        model = pf.feols('d_log_FOGGYi ~ d_DUOPOLY + d_AP_PEAK + d_AP_OFFP | TIME_int',
                         data=df_fd, vcov={'CRV1': 'MARKET_int'})
        result = extract_results(
            model, 'did/method/first_diff', 'methods/panel_fixed_effects.md#estimation-method',
            'd_log_FOGGYi', 'd_DUOPOLY', 'Wireline firms, first differences', 'TIME_int',
            'd_AP_PEAK + d_AP_OFFP', 'MARKET_int', 'First Differences', df_fd
        )
        if result:
            results.append(result)
            print(f"First diff: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
except Exception as e:
    print(f"First diff failed: {e}")

# ============================================================================
# ADDITIONAL CONTROL COMBINATIONS
# ============================================================================

print("\n" + "="*60)
print("ADDITIONAL CONTROL COMBINATIONS")
print("="*60)

# Incrementally add controls
control_sets = [
    ('robust/control/add_step1', ['AP_PEAK']),
    ('robust/control/add_step2', ['AP_PEAK', 'AP_OFFP']),
    ('robust/control/add_step3', ['AP_PEAK', 'AP_OFFP', 'MULTIMKT']),
    ('robust/control/add_step4', ['AP_PEAK', 'AP_OFFP', 'MULTIMKT', 'MKT_AGE']),
    ('robust/control/add_step5', ['AP_PEAK', 'AP_OFFP', 'MULTIMKT', 'MKT_AGE', 'LEAD']),
    ('robust/control/add_step6', ['AP_PEAK', 'AP_OFFP', 'MULTIMKT', 'MKT_AGE', 'LEAD', 'BELL']),
]

for spec_id, controls in control_sets:
    try:
        ctrl_str = ' + '.join(controls)
        model = pf.feols(f'log_FOGGYi ~ DUOPOLY + {ctrl_str} | MARKET_int + TIME_int',
                         data=df_wire, vcov={'CRV1': 'MARKET_int'})
        result = extract_results(
            model, spec_id, 'robustness/control_progression.md',
            'log_FOGGYi', 'DUOPOLY', 'Wireline firms', 'MARKET_int + TIME_int',
            ctrl_str, 'MARKET_int', 'OLS-FE', df_wire
        )
        if result:
            results.append(result)
            print(f"{spec_id}: coef={result['coefficient']:.4f}, se={result['std_error']:.4f}, p={result['p_value']:.4f}")
    except Exception as e:
        print(f"{spec_id} failed: {e}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Save to package directory
OUTPUT_PATH = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/116442-V1/specification_results.csv'
results_df.to_csv(OUTPUT_PATH, index=False)

print(f"\nTotal specifications: {len(results_df)}")
print(f"Saved to: {OUTPUT_PATH}")

# Summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Total specifications: {len(results_df)}")
print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({(results_df['coefficient'] > 0).mean()*100:.1f}%)")
print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({(results_df['p_value'] < 0.05).mean()*100:.1f}%)")
print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({(results_df['p_value'] < 0.01).mean()*100:.1f}%)")
print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")
print(f"Range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")
