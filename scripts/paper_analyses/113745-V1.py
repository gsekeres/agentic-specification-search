#!/usr/bin/env python3
"""
Specification Search: Paper 113745-V1
"Disclosure by Politicians" - Djankov, La Porta, Lopez-de-Silanes, Shleifer (2010)
AEJ: Applied Economics

This script runs a systematic specification search following the specification tree.
Method: Cross-sectional OLS

Main hypothesis: Public disclosure of politicians' financial information is associated
with better government quality and lower corruption.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_and_prepare_data():
    """Load and merge all data from the Excel file."""
    xlsx_file = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/113745-V1/AEJAPP-2009-0187_data/AEJAPP-2009-0187_AppendixC.xlsx'

    # Load AppendixC1 - Main disclosure indices
    df1 = pd.read_excel(xlsx_file, sheet_name='AppendixC1_Indices_in_Paper', skiprows=8)
    df1.columns = ['country', 'disc_req', 'ft_pubprac_all', 'v_mc_all', 's_mc_all',
                   'v_cit_mp_prac_all', 's_cit_mp_prac_all', 'registrar_bylaw', 'checker_bylaw']
    df1 = df1[df1['country'].notna() & (df1['country'] != 'cname_wb_law')].reset_index(drop=True)

    # Load AppendixC8 - Other variables (controls and outcomes)
    df8 = pd.read_excel(xlsx_file, sheet_name='AppendixC8_OtherVariables', skiprows=4)
    df8.columns = ['country', 'flopsobs', 'icrg_d03_07', 'goveff_96_07', 'lncost_07', 'exp_risk',
                  'democ_part_van', 'democ_5006', 'lngni06', 'negpres06', 'fuel00', 'protmg80',
                  'catho80', 'muslim80', 'contnt_fdum1', 'contnt_fdum2', 'contnt_fdum3',
                  'contnt_fdum4', 'contnt_fdum5', 'kaufmann2003_2005', 'ti_03_07', 'her_03_08',
                  'gcr2003_2008', 'per_firms_bribe_octsample', 'gni_ihigh', 'gni_iupmi',
                  'gni_ilomi', 'gni_ilow', 'high_dem5006', 'legor07_uk', 'legor07_fr',
                  'legor07_ger', 'legor07_scan', 'high_freepress']
    df8 = df8[df8['country'].notna() & (df8['country'] != 'cname_wb_law')].reset_index(drop=True)

    # Merge datasets
    df = pd.merge(df1, df8, on='country', how='outer')

    # Convert string columns to numeric
    for col in ['disc_req', 'ft_pubprac_all', 'v_mc_all', 's_mc_all',
                'v_cit_mp_prac_all', 's_cit_mp_prac_all', 'registrar_bylaw', 'checker_bylaw']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Create additional variables
    # Combined public disclosure index (average of values and sources publicly available)
    df['public_disclosure'] = (df['v_cit_mp_prac_all'].fillna(0) + df['s_cit_mp_prac_all'].fillna(0)) / 2

    # Internal disclosure index (values and sources to congress only)
    df['internal_disclosure'] = (df['v_mc_all'].fillna(0) + df['s_mc_all'].fillna(0)) / 2

    # Enforcement index
    df['enforcement'] = (df['registrar_bylaw'].fillna(0) + df['checker_bylaw'].fillna(0)) / 2

    # Democracy score (scaled)
    df['democracy'] = df['democ_5006']

    # Press freedom (note: negative values = freer press in original coding)
    df['press_freedom'] = -df['negpres06']  # Flip sign so higher = freer

    # Income controls
    df['log_gni'] = df['lngni06']

    return df


# =============================================================================
# REGRESSION FUNCTIONS
# =============================================================================

def run_ols_regression(df, outcome_var, treatment_var, controls=None, robust=True):
    """Run OLS regression and return results dictionary."""

    # Build formula
    if controls:
        formula = f"{outcome_var} ~ {treatment_var} + {' + '.join(controls)}"
    else:
        formula = f"{outcome_var} ~ {treatment_var}"

    # Drop missing values for relevant variables
    vars_needed = [outcome_var, treatment_var] + (controls if controls else [])
    df_clean = df.dropna(subset=vars_needed)

    if len(df_clean) < 10:
        return None

    try:
        if robust:
            model = smf.ols(formula, data=df_clean).fit(cov_type='HC1')
        else:
            model = smf.ols(formula, data=df_clean).fit()

        # Extract treatment coefficient
        coef = model.params[treatment_var]
        se = model.bse[treatment_var]
        tstat = model.tvalues[treatment_var]
        pval = model.pvalues[treatment_var]
        ci = model.conf_int().loc[treatment_var]

        # Build coefficient vector
        coef_vector = {
            "treatment": {
                "var": treatment_var,
                "coef": float(coef),
                "se": float(se),
                "pval": float(pval),
                "ci_lower": float(ci[0]),
                "ci_upper": float(ci[1])
            },
            "controls": [],
            "diagnostics": {
                "r_squared": float(model.rsquared),
                "adj_r_squared": float(model.rsquared_adj),
                "f_stat": float(model.fvalue) if model.fvalue else None,
                "f_pval": float(model.f_pvalue) if model.f_pvalue else None
            }
        }

        # Add control coefficients
        for var in model.params.index:
            if var not in ['Intercept', treatment_var]:
                coef_vector["controls"].append({
                    "var": var,
                    "coef": float(model.params[var]),
                    "se": float(model.bse[var]),
                    "pval": float(model.pvalues[var])
                })

        return {
            'coefficient': coef,
            'std_error': se,
            't_stat': tstat,
            'p_value': pval,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'n_obs': int(model.nobs),
            'r_squared': model.rsquared,
            'coefficient_vector_json': json.dumps(coef_vector)
        }

    except Exception as e:
        print(f"Error in regression: {e}")
        return None


def run_quantile_regression(df, outcome_var, treatment_var, controls=None, quantile=0.5):
    """Run quantile regression."""
    from statsmodels.regression.quantile_regression import QuantReg

    vars_needed = [outcome_var, treatment_var] + (controls if controls else [])
    df_clean = df.dropna(subset=vars_needed)

    if len(df_clean) < 10:
        return None

    try:
        # Build design matrix
        X_vars = [treatment_var] + (controls if controls else [])
        X = sm.add_constant(df_clean[X_vars])
        y = df_clean[outcome_var]

        model = QuantReg(y, X).fit(q=quantile)

        coef = model.params[treatment_var]
        se = model.bse[treatment_var]
        tstat = model.tvalues[treatment_var]
        pval = model.pvalues[treatment_var]
        ci = model.conf_int().loc[treatment_var]

        coef_vector = {
            "treatment": {
                "var": treatment_var,
                "coef": float(coef),
                "se": float(se),
                "pval": float(pval)
            },
            "controls": [],
            "diagnostics": {"pseudo_r_squared": float(model.prsquared)}
        }

        return {
            'coefficient': coef,
            'std_error': se,
            't_stat': tstat,
            'p_value': pval,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'n_obs': int(model.nobs),
            'r_squared': model.prsquared,
            'coefficient_vector_json': json.dumps(coef_vector)
        }
    except Exception as e:
        print(f"Quantile regression error: {e}")
        return None


# =============================================================================
# SPECIFICATION DEFINITIONS
# =============================================================================

# Paper metadata
PAPER_ID = "113745-V1"
JOURNAL = "AEJ-Applied"
PAPER_TITLE = "Disclosure by Politicians"

# Main outcome variables
OUTCOME_VARS = {
    'goveff_96_07': 'Government Effectiveness',
    'kaufmann2003_2005': 'Kaufmann Corruption Index',
    'ti_03_07': 'Transparency International Corruption',
    'icrg_d03_07': 'ICRG Corruption Index'
}

# Treatment variable (main focus: public disclosure)
TREATMENT_VAR = 'public_disclosure'
TREATMENT_VAR_ALT = 'ft_pubprac_all'  # Binary publicly available

# Control variables used in the paper
BASELINE_CONTROLS = ['log_gni', 'democracy', 'press_freedom']
EXTENDED_CONTROLS = ['log_gni', 'democracy', 'press_freedom', 'legor07_uk', 'legor07_fr', 'legor07_ger', 'legor07_scan']
CONTINENT_CONTROLS = ['contnt_fdum1', 'contnt_fdum2', 'contnt_fdum3', 'contnt_fdum4', 'contnt_fdum5']


# =============================================================================
# RUN SPECIFICATIONS
# =============================================================================

def run_all_specifications():
    """Run all specifications and return results."""

    df = load_and_prepare_data()
    results = []

    # Primary outcome for most specifications
    PRIMARY_OUTCOME = 'goveff_96_07'

    # =========================================================================
    # BASELINE SPECIFICATIONS
    # =========================================================================

    print("Running baseline specifications...")

    # Baseline: Main specification from paper (public disclosure -> government effectiveness)
    res = run_ols_regression(df, PRIMARY_OUTCOME, TREATMENT_VAR, BASELINE_CONTROLS)
    if res:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'baseline',
            'spec_tree_path': 'methods/cross_sectional_ols.md',
            'outcome_var': PRIMARY_OUTCOME,
            'treatment_var': TREATMENT_VAR,
            'fixed_effects': 'None',
            'controls_desc': 'log_gni, democracy, press_freedom',
            'cluster_var': 'None',
            'model_type': 'OLS',
            'sample_desc': 'All countries with complete data',
            'estimation_script': 'scripts/paper_analyses/113745-V1.py',
            **res
        })

    # =========================================================================
    # METHOD-SPECIFIC: DIFFERENT OUTCOMES
    # =========================================================================

    print("Running outcome variations...")

    for outcome_var, outcome_desc in OUTCOME_VARS.items():
        res = run_ols_regression(df, outcome_var, TREATMENT_VAR, BASELINE_CONTROLS)
        if res:
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': f'ols/outcome/{outcome_var}',
                'spec_tree_path': 'methods/cross_sectional_ols.md#outcome-variations',
                'outcome_var': outcome_var,
                'treatment_var': TREATMENT_VAR,
                'fixed_effects': 'None',
                'controls_desc': 'log_gni, democracy, press_freedom',
                'cluster_var': 'None',
                'model_type': 'OLS',
                'sample_desc': f'All countries - {outcome_desc}',
                'estimation_script': 'scripts/paper_analyses/113745-V1.py',
                **res
            })

    # =========================================================================
    # METHOD-SPECIFIC: DIFFERENT TREATMENT MEASURES
    # =========================================================================

    print("Running treatment variations...")

    treatment_vars = {
        'ft_pubprac_all': 'Publicly Available (Binary)',
        'v_cit_mp_prac_all': 'Values Publicly Available',
        's_cit_mp_prac_all': 'Sources Publicly Available',
        'internal_disclosure': 'Internal Disclosure (to Congress)',
        'disc_req': 'Disclosure Required (Binary)',
        'enforcement': 'Enforcement Index'
    }

    for treat_var, treat_desc in treatment_vars.items():
        res = run_ols_regression(df, PRIMARY_OUTCOME, treat_var, BASELINE_CONTROLS)
        if res:
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': f'ols/treatment/{treat_var}',
                'spec_tree_path': 'methods/cross_sectional_ols.md#treatment-variations',
                'outcome_var': PRIMARY_OUTCOME,
                'treatment_var': treat_var,
                'fixed_effects': 'None',
                'controls_desc': 'log_gni, democracy, press_freedom',
                'cluster_var': 'None',
                'model_type': 'OLS',
                'sample_desc': f'All countries - {treat_desc}',
                'estimation_script': 'scripts/paper_analyses/113745-V1.py',
                **res
            })

    # =========================================================================
    # OLS CONTROL SET VARIATIONS
    # =========================================================================

    print("Running control set variations...")

    # No controls (bivariate)
    res = run_ols_regression(df, PRIMARY_OUTCOME, TREATMENT_VAR, controls=None)
    if res:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'ols/controls/none',
            'spec_tree_path': 'methods/cross_sectional_ols.md#control-sets',
            'outcome_var': PRIMARY_OUTCOME,
            'treatment_var': TREATMENT_VAR,
            'fixed_effects': 'None',
            'controls_desc': 'None (bivariate)',
            'cluster_var': 'None',
            'model_type': 'OLS',
            'sample_desc': 'All countries',
            'estimation_script': 'scripts/paper_analyses/113745-V1.py',
            **res
        })

    # Extended controls with legal origins
    res = run_ols_regression(df, PRIMARY_OUTCOME, TREATMENT_VAR, EXTENDED_CONTROLS)
    if res:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'ols/controls/extended',
            'spec_tree_path': 'methods/cross_sectional_ols.md#control-sets',
            'outcome_var': PRIMARY_OUTCOME,
            'treatment_var': TREATMENT_VAR,
            'fixed_effects': 'None',
            'controls_desc': 'Baseline + legal origins',
            'cluster_var': 'None',
            'model_type': 'OLS',
            'sample_desc': 'All countries',
            'estimation_script': 'scripts/paper_analyses/113745-V1.py',
            **res
        })

    # Full controls with continents
    full_controls = BASELINE_CONTROLS + CONTINENT_CONTROLS
    res = run_ols_regression(df, PRIMARY_OUTCOME, TREATMENT_VAR, full_controls)
    if res:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'ols/controls/full',
            'spec_tree_path': 'methods/cross_sectional_ols.md#control-sets',
            'outcome_var': PRIMARY_OUTCOME,
            'treatment_var': TREATMENT_VAR,
            'fixed_effects': 'None',
            'controls_desc': 'Baseline + continent dummies',
            'cluster_var': 'None',
            'model_type': 'OLS',
            'sample_desc': 'All countries',
            'estimation_script': 'scripts/paper_analyses/113745-V1.py',
            **res
        })

    # Kitchen sink
    kitchen_sink = EXTENDED_CONTROLS + CONTINENT_CONTROLS
    res = run_ols_regression(df, PRIMARY_OUTCOME, TREATMENT_VAR, kitchen_sink)
    if res:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'ols/controls/kitchen_sink',
            'spec_tree_path': 'methods/cross_sectional_ols.md#control-sets',
            'outcome_var': PRIMARY_OUTCOME,
            'treatment_var': TREATMENT_VAR,
            'fixed_effects': 'None',
            'controls_desc': 'All controls (legal origins + continents)',
            'cluster_var': 'None',
            'model_type': 'OLS',
            'sample_desc': 'All countries',
            'estimation_script': 'scripts/paper_analyses/113745-V1.py',
            **res
        })

    # =========================================================================
    # OLS STANDARD ERROR VARIATIONS
    # =========================================================================

    print("Running SE variations...")

    # Classical (homoskedastic) SE
    res = run_ols_regression(df, PRIMARY_OUTCOME, TREATMENT_VAR, BASELINE_CONTROLS, robust=False)
    if res:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'ols/se/classical',
            'spec_tree_path': 'methods/cross_sectional_ols.md#standard-errors',
            'outcome_var': PRIMARY_OUTCOME,
            'treatment_var': TREATMENT_VAR,
            'fixed_effects': 'None',
            'controls_desc': 'log_gni, democracy, press_freedom',
            'cluster_var': 'None (classical SE)',
            'model_type': 'OLS',
            'sample_desc': 'All countries - Classical SE',
            'estimation_script': 'scripts/paper_analyses/113745-V1.py',
            **res
        })

    # HC2 robust SE
    df_clean = df.dropna(subset=[PRIMARY_OUTCOME, TREATMENT_VAR] + BASELINE_CONTROLS)
    try:
        formula = f"{PRIMARY_OUTCOME} ~ {TREATMENT_VAR} + {' + '.join(BASELINE_CONTROLS)}"
        model = smf.ols(formula, data=df_clean).fit(cov_type='HC2')
        coef = model.params[TREATMENT_VAR]
        se = model.bse[TREATMENT_VAR]
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'ols/se/hc2',
            'spec_tree_path': 'methods/cross_sectional_ols.md#standard-errors',
            'outcome_var': PRIMARY_OUTCOME,
            'treatment_var': TREATMENT_VAR,
            'coefficient': coef,
            'std_error': se,
            't_stat': model.tvalues[TREATMENT_VAR],
            'p_value': model.pvalues[TREATMENT_VAR],
            'ci_lower': model.conf_int().loc[TREATMENT_VAR][0],
            'ci_upper': model.conf_int().loc[TREATMENT_VAR][1],
            'n_obs': int(model.nobs),
            'r_squared': model.rsquared,
            'fixed_effects': 'None',
            'controls_desc': 'log_gni, democracy, press_freedom',
            'cluster_var': 'None (HC2 SE)',
            'model_type': 'OLS',
            'sample_desc': 'All countries - HC2 SE',
            'estimation_script': 'scripts/paper_analyses/113745-V1.py',
            'coefficient_vector_json': json.dumps({"treatment": {"coef": float(coef), "se": float(se)}})
        })
    except:
        pass

    # HC3 robust SE
    try:
        model = smf.ols(formula, data=df_clean).fit(cov_type='HC3')
        coef = model.params[TREATMENT_VAR]
        se = model.bse[TREATMENT_VAR]
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'ols/se/hc3',
            'spec_tree_path': 'methods/cross_sectional_ols.md#standard-errors',
            'outcome_var': PRIMARY_OUTCOME,
            'treatment_var': TREATMENT_VAR,
            'coefficient': coef,
            'std_error': se,
            't_stat': model.tvalues[TREATMENT_VAR],
            'p_value': model.pvalues[TREATMENT_VAR],
            'ci_lower': model.conf_int().loc[TREATMENT_VAR][0],
            'ci_upper': model.conf_int().loc[TREATMENT_VAR][1],
            'n_obs': int(model.nobs),
            'r_squared': model.rsquared,
            'fixed_effects': 'None',
            'controls_desc': 'log_gni, democracy, press_freedom',
            'cluster_var': 'None (HC3 SE)',
            'model_type': 'OLS',
            'sample_desc': 'All countries - HC3 SE',
            'estimation_script': 'scripts/paper_analyses/113745-V1.py',
            'coefficient_vector_json': json.dumps({"treatment": {"coef": float(coef), "se": float(se)}})
        })
    except:
        pass

    # =========================================================================
    # ROBUSTNESS: LEAVE-ONE-OUT
    # =========================================================================

    print("Running leave-one-out specifications...")

    for control in BASELINE_CONTROLS:
        remaining_controls = [c for c in BASELINE_CONTROLS if c != control]
        res = run_ols_regression(df, PRIMARY_OUTCOME, TREATMENT_VAR, remaining_controls)
        if res:
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': f'robust/loo/drop_{control}',
                'spec_tree_path': 'robustness/leave_one_out.md',
                'outcome_var': PRIMARY_OUTCOME,
                'treatment_var': TREATMENT_VAR,
                'fixed_effects': 'None',
                'controls_desc': f'Baseline minus {control}',
                'cluster_var': 'None',
                'model_type': 'OLS',
                'sample_desc': f'All countries - dropped {control}',
                'estimation_script': 'scripts/paper_analyses/113745-V1.py',
                **res
            })

    # =========================================================================
    # ROBUSTNESS: SINGLE COVARIATE
    # =========================================================================

    print("Running single covariate specifications...")

    for control in BASELINE_CONTROLS:
        res = run_ols_regression(df, PRIMARY_OUTCOME, TREATMENT_VAR, [control])
        if res:
            results.append({
                'paper_id': PAPER_ID,
                'journal': JOURNAL,
                'paper_title': PAPER_TITLE,
                'spec_id': f'robust/single/{control}',
                'spec_tree_path': 'robustness/single_covariate.md',
                'outcome_var': PRIMARY_OUTCOME,
                'treatment_var': TREATMENT_VAR,
                'fixed_effects': 'None',
                'controls_desc': f'{control} only',
                'cluster_var': 'None',
                'model_type': 'OLS',
                'sample_desc': f'All countries - single control {control}',
                'estimation_script': 'scripts/paper_analyses/113745-V1.py',
                **res
            })

    # =========================================================================
    # ROBUSTNESS: SAMPLE RESTRICTIONS
    # =========================================================================

    print("Running sample restriction specifications...")

    # High income countries only
    df_high = df[df['gni_ihigh'] == 1]
    res = run_ols_regression(df_high, PRIMARY_OUTCOME, TREATMENT_VAR, BASELINE_CONTROLS)
    if res:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/sample/high_income',
            'spec_tree_path': 'robustness/sample_restrictions.md',
            'outcome_var': PRIMARY_OUTCOME,
            'treatment_var': TREATMENT_VAR,
            'fixed_effects': 'None',
            'controls_desc': 'log_gni, democracy, press_freedom',
            'cluster_var': 'None',
            'model_type': 'OLS',
            'sample_desc': 'High income countries only',
            'estimation_script': 'scripts/paper_analyses/113745-V1.py',
            **res
        })

    # Non-high income countries
    df_nonhigh = df[df['gni_ihigh'] != 1]
    res = run_ols_regression(df_nonhigh, PRIMARY_OUTCOME, TREATMENT_VAR, BASELINE_CONTROLS)
    if res:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/sample/non_high_income',
            'spec_tree_path': 'robustness/sample_restrictions.md',
            'outcome_var': PRIMARY_OUTCOME,
            'treatment_var': TREATMENT_VAR,
            'fixed_effects': 'None',
            'controls_desc': 'log_gni, democracy, press_freedom',
            'cluster_var': 'None',
            'model_type': 'OLS',
            'sample_desc': 'Non-high income countries',
            'estimation_script': 'scripts/paper_analyses/113745-V1.py',
            **res
        })

    # High democracy countries only
    df_highdem = df[df['high_dem5006'] == 1]
    res = run_ols_regression(df_highdem, PRIMARY_OUTCOME, TREATMENT_VAR, BASELINE_CONTROLS)
    if res:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/sample/high_democracy',
            'spec_tree_path': 'robustness/sample_restrictions.md',
            'outcome_var': PRIMARY_OUTCOME,
            'treatment_var': TREATMENT_VAR,
            'fixed_effects': 'None',
            'controls_desc': 'log_gni, democracy, press_freedom',
            'cluster_var': 'None',
            'model_type': 'OLS',
            'sample_desc': 'High democracy countries only',
            'estimation_script': 'scripts/paper_analyses/113745-V1.py',
            **res
        })

    # Low democracy countries only
    df_lowdem = df[df['high_dem5006'] != 1]
    res = run_ols_regression(df_lowdem, PRIMARY_OUTCOME, TREATMENT_VAR, BASELINE_CONTROLS)
    if res:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/sample/low_democracy',
            'spec_tree_path': 'robustness/sample_restrictions.md',
            'outcome_var': PRIMARY_OUTCOME,
            'treatment_var': TREATMENT_VAR,
            'fixed_effects': 'None',
            'controls_desc': 'log_gni, democracy, press_freedom',
            'cluster_var': 'None',
            'model_type': 'OLS',
            'sample_desc': 'Low democracy countries',
            'estimation_script': 'scripts/paper_analyses/113745-V1.py',
            **res
        })

    # Exclude Africa
    df_noafrica = df[df['contnt_fdum1'] != 1]
    res = run_ols_regression(df_noafrica, PRIMARY_OUTCOME, TREATMENT_VAR, BASELINE_CONTROLS)
    if res:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/sample/exclude_africa',
            'spec_tree_path': 'robustness/sample_restrictions.md',
            'outcome_var': PRIMARY_OUTCOME,
            'treatment_var': TREATMENT_VAR,
            'fixed_effects': 'None',
            'controls_desc': 'log_gni, democracy, press_freedom',
            'cluster_var': 'None',
            'model_type': 'OLS',
            'sample_desc': 'Excluding Africa',
            'estimation_script': 'scripts/paper_analyses/113745-V1.py',
            **res
        })

    # Common law countries only
    df_common = df[df['legor07_uk'] == 1]
    res = run_ols_regression(df_common, PRIMARY_OUTCOME, TREATMENT_VAR, BASELINE_CONTROLS)
    if res:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/sample/common_law',
            'spec_tree_path': 'robustness/sample_restrictions.md',
            'outcome_var': PRIMARY_OUTCOME,
            'treatment_var': TREATMENT_VAR,
            'fixed_effects': 'None',
            'controls_desc': 'log_gni, democracy, press_freedom',
            'cluster_var': 'None',
            'model_type': 'OLS',
            'sample_desc': 'Common law countries only',
            'estimation_script': 'scripts/paper_analyses/113745-V1.py',
            **res
        })

    # Civil law countries only
    df_civil = df[(df['legor07_fr'] == 1) | (df['legor07_ger'] == 1) | (df['legor07_scan'] == 1)]
    res = run_ols_regression(df_civil, PRIMARY_OUTCOME, TREATMENT_VAR, BASELINE_CONTROLS)
    if res:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/sample/civil_law',
            'spec_tree_path': 'robustness/sample_restrictions.md',
            'outcome_var': PRIMARY_OUTCOME,
            'treatment_var': TREATMENT_VAR,
            'fixed_effects': 'None',
            'controls_desc': 'log_gni, democracy, press_freedom',
            'cluster_var': 'None',
            'model_type': 'OLS',
            'sample_desc': 'Civil law countries only',
            'estimation_script': 'scripts/paper_analyses/113745-V1.py',
            **res
        })

    # Trimmed sample (drop extreme outcomes)
    q01 = df[PRIMARY_OUTCOME].quantile(0.01)
    q99 = df[PRIMARY_OUTCOME].quantile(0.99)
    df_trimmed = df[(df[PRIMARY_OUTCOME] >= q01) & (df[PRIMARY_OUTCOME] <= q99)]
    res = run_ols_regression(df_trimmed, PRIMARY_OUTCOME, TREATMENT_VAR, BASELINE_CONTROLS)
    if res:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/sample/trim_1pct',
            'spec_tree_path': 'robustness/sample_restrictions.md',
            'outcome_var': PRIMARY_OUTCOME,
            'treatment_var': TREATMENT_VAR,
            'fixed_effects': 'None',
            'controls_desc': 'log_gni, democracy, press_freedom',
            'cluster_var': 'None',
            'model_type': 'OLS',
            'sample_desc': 'Trimmed at 1st/99th percentile',
            'estimation_script': 'scripts/paper_analyses/113745-V1.py',
            **res
        })

    # =========================================================================
    # ROBUSTNESS: FUNCTIONAL FORM
    # =========================================================================

    print("Running functional form variations...")

    # Standardized outcome
    df['goveff_std'] = (df[PRIMARY_OUTCOME] - df[PRIMARY_OUTCOME].mean()) / df[PRIMARY_OUTCOME].std()
    res = run_ols_regression(df, 'goveff_std', TREATMENT_VAR, BASELINE_CONTROLS)
    if res:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/form/y_standardized',
            'spec_tree_path': 'robustness/functional_form.md',
            'outcome_var': 'goveff_std',
            'treatment_var': TREATMENT_VAR,
            'fixed_effects': 'None',
            'controls_desc': 'log_gni, democracy, press_freedom',
            'cluster_var': 'None',
            'model_type': 'OLS',
            'sample_desc': 'All countries - standardized outcome',
            'estimation_script': 'scripts/paper_analyses/113745-V1.py',
            **res
        })

    # Rank outcome
    df['goveff_rank'] = df[PRIMARY_OUTCOME].rank()
    res = run_ols_regression(df, 'goveff_rank', TREATMENT_VAR, BASELINE_CONTROLS)
    if res:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/form/y_rank',
            'spec_tree_path': 'robustness/functional_form.md',
            'outcome_var': 'goveff_rank',
            'treatment_var': TREATMENT_VAR,
            'fixed_effects': 'None',
            'controls_desc': 'log_gni, democracy, press_freedom',
            'cluster_var': 'None',
            'model_type': 'OLS',
            'sample_desc': 'All countries - rank outcome',
            'estimation_script': 'scripts/paper_analyses/113745-V1.py',
            **res
        })

    # Quadratic treatment
    df['public_disclosure_sq'] = df[TREATMENT_VAR] ** 2
    quadratic_controls = BASELINE_CONTROLS + ['public_disclosure_sq']
    res = run_ols_regression(df, PRIMARY_OUTCOME, TREATMENT_VAR, quadratic_controls)
    if res:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/form/quadratic',
            'spec_tree_path': 'robustness/functional_form.md',
            'outcome_var': PRIMARY_OUTCOME,
            'treatment_var': TREATMENT_VAR,
            'fixed_effects': 'None',
            'controls_desc': 'Baseline + treatment squared',
            'cluster_var': 'None',
            'model_type': 'OLS',
            'sample_desc': 'All countries - quadratic specification',
            'estimation_script': 'scripts/paper_analyses/113745-V1.py',
            **res
        })

    # Median regression (quantile 0.5)
    res = run_quantile_regression(df, PRIMARY_OUTCOME, TREATMENT_VAR, BASELINE_CONTROLS, quantile=0.5)
    if res:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/form/quantile_50',
            'spec_tree_path': 'robustness/functional_form.md',
            'outcome_var': PRIMARY_OUTCOME,
            'treatment_var': TREATMENT_VAR,
            'fixed_effects': 'None',
            'controls_desc': 'log_gni, democracy, press_freedom',
            'cluster_var': 'None',
            'model_type': 'Quantile (Median)',
            'sample_desc': 'All countries - median regression',
            'estimation_script': 'scripts/paper_analyses/113745-V1.py',
            **res
        })

    # 25th percentile regression
    res = run_quantile_regression(df, PRIMARY_OUTCOME, TREATMENT_VAR, BASELINE_CONTROLS, quantile=0.25)
    if res:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/form/quantile_25',
            'spec_tree_path': 'robustness/functional_form.md',
            'outcome_var': PRIMARY_OUTCOME,
            'treatment_var': TREATMENT_VAR,
            'fixed_effects': 'None',
            'controls_desc': 'log_gni, democracy, press_freedom',
            'cluster_var': 'None',
            'model_type': 'Quantile (25th)',
            'sample_desc': 'All countries - 25th percentile regression',
            'estimation_script': 'scripts/paper_analyses/113745-V1.py',
            **res
        })

    # 75th percentile regression
    res = run_quantile_regression(df, PRIMARY_OUTCOME, TREATMENT_VAR, BASELINE_CONTROLS, quantile=0.75)
    if res:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'robust/form/quantile_75',
            'spec_tree_path': 'robustness/functional_form.md',
            'outcome_var': PRIMARY_OUTCOME,
            'treatment_var': TREATMENT_VAR,
            'fixed_effects': 'None',
            'controls_desc': 'log_gni, democracy, press_freedom',
            'cluster_var': 'None',
            'model_type': 'Quantile (75th)',
            'sample_desc': 'All countries - 75th percentile regression',
            'estimation_script': 'scripts/paper_analyses/113745-V1.py',
            **res
        })

    # =========================================================================
    # INTERACTIONS
    # =========================================================================

    print("Running interaction specifications...")

    # Public disclosure x Democracy interaction
    df['disclosure_x_democracy'] = df[TREATMENT_VAR] * df['democracy']
    interact_controls = BASELINE_CONTROLS + ['disclosure_x_democracy']
    res = run_ols_regression(df, PRIMARY_OUTCOME, TREATMENT_VAR, interact_controls)
    if res:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'ols/interact/democracy',
            'spec_tree_path': 'methods/cross_sectional_ols.md#interaction-effects',
            'outcome_var': PRIMARY_OUTCOME,
            'treatment_var': TREATMENT_VAR,
            'fixed_effects': 'None',
            'controls_desc': 'Baseline + disclosure x democracy',
            'cluster_var': 'None',
            'model_type': 'OLS',
            'sample_desc': 'All countries - democracy interaction',
            'estimation_script': 'scripts/paper_analyses/113745-V1.py',
            **res
        })

    # Public disclosure x Press freedom interaction
    df['disclosure_x_pressfree'] = df[TREATMENT_VAR] * df['press_freedom']
    interact_controls2 = BASELINE_CONTROLS + ['disclosure_x_pressfree']
    res = run_ols_regression(df, PRIMARY_OUTCOME, TREATMENT_VAR, interact_controls2)
    if res:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': 'ols/interact/press_freedom',
            'spec_tree_path': 'methods/cross_sectional_ols.md#interaction-effects',
            'outcome_var': PRIMARY_OUTCOME,
            'treatment_var': TREATMENT_VAR,
            'fixed_effects': 'None',
            'controls_desc': 'Baseline + disclosure x press freedom',
            'cluster_var': 'None',
            'model_type': 'OLS',
            'sample_desc': 'All countries - press freedom interaction',
            'estimation_script': 'scripts/paper_analyses/113745-V1.py',
            **res
        })

    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Specification Search: 113745-V1")
    print("Disclosure by Politicians")
    print("=" * 60)

    results = run_all_specifications()

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    output_path = '/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/113745-V1/specification_results.csv'
    results_df.to_csv(output_path, index=False)

    print(f"\nTotal specifications run: {len(results_df)}")
    print(f"Results saved to: {output_path}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    print(f"\nTotal specifications: {len(results_df)}")
    print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({100*(results_df['coefficient'] > 0).mean():.1f}%)")
    print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({100*(results_df['p_value'] < 0.05).mean():.1f}%)")
    print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({100*(results_df['p_value'] < 0.01).mean():.1f}%)")
    print(f"Coefficient range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")
    print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
    print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")
