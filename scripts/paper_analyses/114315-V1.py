#!/usr/bin/env python3
"""
Specification Search: 114315-V1
Paper: "The Geography of Trade in Online Transactions: Evidence from eBay and MercadoLibre"
Authors: Hortacsu, Martinez-Jerez & Douglas (2008)
Journal: AEJ: Microeconomics

IMPORTANT DATA LIMITATION:
The main transaction data files (all_US_items.dta for eBay, datareg_pais and datareg_prov
for MercadoLibre) are confidential and not included in the replication package due to
confidentiality agreements. Only auxiliary data files with state-level characteristics
are available. This script documents the intended specifications and provides a template
for replication if data becomes available.

Method: Panel Fixed Effects (Gravity Model)
- Outcome: Log transaction counts (lntcount_bs) or Log transaction volume (lntvol_bs)
- Treatment: Log distance (lndist) and same-state/same-country indicator (samestate/samepais)
- Fixed Effects: Buyer state FE + Seller state FE (absorbed)
- Standard Errors: Robust

Key hypothesis: Geographic distance and political borders affect online trade volumes,
even when the online marketplace theoretically eliminates geographic frictions.
"""

import pandas as pd
import numpy as np
import json
import os
from scipy import stats

# Attempt to import estimation packages
try:
    import pyfixest as pf
    HAS_PYFIXEST = True
except ImportError:
    HAS_PYFIXEST = False

try:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# Paths
PACKAGE_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/114315-V1/programfiles_AEJMicro2007_0011"
OUTPUT_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/114315-V1"

# Paper metadata
PAPER_ID = "114315-V1"
PAPER_TITLE = "The Geography of Trade in Online Transactions: Evidence from eBay and MercadoLibre"
JOURNAL = "AEJ-Microeconomics"

# ============================================================================
# SPECIFICATION MAPPING
# ============================================================================

METHOD_MAP = {
    "method_code": "panel_fixed_effects",
    "method_tree_path": "specification_tree/methods/panel_fixed_effects.md",
    "specs_to_run": [
        {"spec_id": "baseline", "spec_tree_path": "methods/panel_fixed_effects.md#baseline"},
        {"spec_id": "panel/fe/none", "spec_tree_path": "methods/panel_fixed_effects.md#fixed-effects-structure"},
        {"spec_id": "panel/fe/unit", "spec_tree_path": "methods/panel_fixed_effects.md#fixed-effects-structure"},
        {"spec_id": "panel/fe/twoway", "spec_tree_path": "methods/panel_fixed_effects.md#fixed-effects-structure"},
        {"spec_id": "panel/controls/none", "spec_tree_path": "methods/panel_fixed_effects.md#control-sets"},
        {"spec_id": "panel/controls/baseline", "spec_tree_path": "methods/panel_fixed_effects.md#control-sets"},
        {"spec_id": "panel/controls/full", "spec_tree_path": "methods/panel_fixed_effects.md#control-sets"},
        {"spec_id": "panel/sample/category_fe", "spec_tree_path": "methods/panel_fixed_effects.md#sample-restrictions"},
        {"spec_id": "panel/sample/video_games", "spec_tree_path": "methods/panel_fixed_effects.md#sample-restrictions"},
        {"spec_id": "panel/sample/jewelry", "spec_tree_path": "methods/panel_fixed_effects.md#sample-restrictions"},
        {"spec_id": "panel/sample/new_buyers", "spec_tree_path": "methods/panel_fixed_effects.md#sample-restrictions"},
        {"spec_id": "panel/sample/sophisticated_buyers", "spec_tree_path": "methods/panel_fixed_effects.md#sample-restrictions"},
        {"spec_id": "panel/sample/good_sellers", "spec_tree_path": "methods/panel_fixed_effects.md#sample-restrictions"},
        {"spec_id": "panel/sample/bad_sellers", "spec_tree_path": "methods/panel_fixed_effects.md#sample-restrictions"},
    ],
    "robustness_specs": [
        {"spec_id": "robust/cluster/none", "spec_tree_path": "robustness/clustering_variations.md#single-level-clustering"},
        {"spec_id": "robust/cluster/sstate", "spec_tree_path": "robustness/clustering_variations.md#single-level-clustering"},
        {"spec_id": "robust/cluster/bstate", "spec_tree_path": "robustness/clustering_variations.md#single-level-clustering"},
        {"spec_id": "robust/loo/drop_medshipfrac", "spec_tree_path": "robustness/leave_one_out.md"},
        {"spec_id": "robust/loo/drop_sametimezone", "spec_tree_path": "robustness/leave_one_out.md"},
        {"spec_id": "robust/single/none", "spec_tree_path": "robustness/single_covariate.md"},
        {"spec_id": "robust/single/medshipfrac", "spec_tree_path": "robustness/single_covariate.md"},
        {"spec_id": "robust/single/sametimezone", "spec_tree_path": "robustness/single_covariate.md"},
        {"spec_id": "robust/form/y_level", "spec_tree_path": "robustness/functional_form.md"},
        {"spec_id": "robust/form/y_log", "spec_tree_path": "robustness/functional_form.md"},
        {"spec_id": "robust/form/volume_outcome", "spec_tree_path": "robustness/functional_form.md"},
    ]
}


def create_coefficient_vector_json(coef, se, pval, controls=None, fixed_effects=None, diagnostics=None):
    """Create standardized coefficient vector JSON."""
    cv = {
        "treatment": {
            "var": "samestate",
            "coef": float(coef) if not pd.isna(coef) else None,
            "se": float(se) if not pd.isna(se) else None,
            "pval": float(pval) if not pd.isna(pval) else None,
        },
        "controls": controls or [],
        "fixed_effects_absorbed": fixed_effects or [],
        "diagnostics": diagnostics or {}
    }
    return json.dumps(cv)


def load_auxiliary_data():
    """Load the available auxiliary data files."""
    data = {}

    # Load state pair info
    try:
        df_stateinfo = pd.read_stata(os.path.join(PACKAGE_DIR, 'buyersellerstateinfo.dta'))
        data['stateinfo'] = df_stateinfo
        print(f"Loaded buyersellerstateinfo.dta: {df_stateinfo.shape}")
    except Exception as e:
        print(f"Could not load buyersellerstateinfo.dta: {e}")

    # Load sales tax data
    try:
        df_bsales = pd.read_stata(os.path.join(PACKAGE_DIR, 'bsales_taxes.dta'))
        df_ssales = pd.read_stata(os.path.join(PACKAGE_DIR, 'ssales_taxes.dta'))
        data['bsales_taxes'] = df_bsales
        data['ssales_taxes'] = df_ssales
        print(f"Loaded sales tax data")
    except Exception as e:
        print(f"Could not load sales tax data: {e}")

    # Load timezone data
    try:
        df_btimezones = pd.read_stata(os.path.join(PACKAGE_DIR, 'btimezones.dta'))
        df_stimezones = pd.read_stata(os.path.join(PACKAGE_DIR, 'stimezones.dta'))
        data['btimezones'] = df_btimezones
        data['stimezones'] = df_stimezones
        print(f"Loaded timezone data")
    except Exception as e:
        print(f"Could not load timezone data: {e}")

    # Load category classifications
    try:
        df_categories = pd.read_stata(os.path.join(PACKAGE_DIR, 'categories_classified.dta'))
        data['categories'] = df_categories
        print(f"Loaded category classifications: {df_categories.shape}")
    except Exception as e:
        print(f"Could not load category data: {e}")

    return data


def extract_results_from_log():
    """
    Extract regression results from the log file for MercadoLibre analysis.
    This is the only actual regression output available due to data confidentiality.
    """
    results = []

    # Results from table_countries_aej.log - Country level analysis
    # Table 3, Panel A: Country-level regression
    results.append({
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': 'baseline_mercadolibre_country',
        'spec_tree_path': 'methods/panel_fixed_effects.md#baseline',
        'outcome_var': 'logqty',
        'treatment_var': 'samepais',
        'coefficient': 10.81357,
        'std_error': 1.75745,
        't_stat': 6.15,
        'p_value': 0.000,
        'ci_lower': 7.295651,
        'ci_upper': 14.33149,
        'n_obs': 79,
        'r_squared': 0.5534,
        'r_squared_within': 0.6857,
        'r_squared_between': 0.1604,
        'coefficient_vector_json': create_coefficient_vector_json(
            coef=10.81357, se=1.75745, pval=0.000,
            controls=[
                {"var": "logdist", "coef": -0.545574, "se": 0.5176705, "pval": 0.296}
            ],
            fixed_effects=["seller_country"],
            diagnostics={"n_groups": 10, "avg_obs_per_group": 7.9}
        ),
        'sample_desc': 'MercadoLibre country-level transactions, pooled',
        'fixed_effects': 'Seller country FE',
        'controls_desc': 'log distance, buyer country dummies',
        'cluster_var': 'none (robust SE)',
        'model_type': 'Panel FE (within)',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
        'data_available': True,
        'notes': 'Results extracted from table_countries_aej.log'
    })

    # Results from table_countries_aej.log - Province level analysis
    # Table 3, Panel B: Province-level regression
    results.append({
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': 'baseline_mercadolibre_province',
        'spec_tree_path': 'methods/panel_fixed_effects.md#baseline',
        'outcome_var': 'logqty',
        'treatment_var': 'sameprov',
        'coefficient': 1.010718,
        'std_error': 0.1561668,
        't_stat': 6.47,
        'p_value': 0.000,
        'ci_lower': 0.7045815,
        'ci_upper': 1.316855,
        'n_obs': 7175,
        'r_squared': 0.3754,
        'r_squared_within': 0.6940,
        'r_squared_between': 0.0006,
        'coefficient_vector_json': create_coefficient_vector_json(
            coef=1.010718, se=0.1561668, pval=0.000,
            controls=[
                {"var": "logdist", "coef": -0.3816296, "se": 0.0302367, "pval": 0.000},
                {"var": "samepais", "coef": 6.068261, "se": 0.0803655, "pval": 0.000}
            ],
            fixed_effects=["seller_province"],
            diagnostics={"n_groups": 248, "avg_obs_per_group": 28.9}
        ),
        'sample_desc': 'MercadoLibre province-level transactions, pooled',
        'fixed_effects': 'Seller province FE',
        'controls_desc': 'log distance, same country, buyer province dummies',
        'cluster_var': 'none (robust SE)',
        'model_type': 'Panel FE (within)',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
        'data_available': True,
        'notes': 'Results extracted from table_countries_aej.log'
    })

    # Additional result: samepais effect in province regression
    results.append({
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': 'mercadolibre_province_samepais',
        'spec_tree_path': 'methods/panel_fixed_effects.md#baseline',
        'outcome_var': 'logqty',
        'treatment_var': 'samepais',
        'coefficient': 6.068261,
        'std_error': 0.0803655,
        't_stat': 75.51,
        'p_value': 0.000,
        'ci_lower': 5.910719,
        'ci_upper': 6.225803,
        'n_obs': 7175,
        'r_squared': 0.3754,
        'r_squared_within': 0.6940,
        'r_squared_between': 0.0006,
        'coefficient_vector_json': create_coefficient_vector_json(
            coef=6.068261, se=0.0803655, pval=0.000,
            controls=[
                {"var": "logdist", "coef": -0.3816296, "se": 0.0302367, "pval": 0.000},
                {"var": "sameprov", "coef": 1.010718, "se": 0.1561668, "pval": 0.000}
            ],
            fixed_effects=["seller_province"],
            diagnostics={"n_groups": 248, "avg_obs_per_group": 28.9}
        ),
        'sample_desc': 'MercadoLibre province-level, same country effect',
        'fixed_effects': 'Seller province FE',
        'controls_desc': 'log distance, same province, buyer province dummies',
        'cluster_var': 'none (robust SE)',
        'model_type': 'Panel FE (within)',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
        'data_available': True,
        'notes': 'Results extracted from table_countries_aej.log'
    })

    # Distance effect in province regression
    results.append({
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': 'mercadolibre_province_distance',
        'spec_tree_path': 'methods/panel_fixed_effects.md#baseline',
        'outcome_var': 'logqty',
        'treatment_var': 'logdist',
        'coefficient': -0.3816296,
        'std_error': 0.0302367,
        't_stat': -12.62,
        'p_value': 0.000,
        'ci_lower': -0.4409032,
        'ci_upper': -0.3223561,
        'n_obs': 7175,
        'r_squared': 0.3754,
        'r_squared_within': 0.6940,
        'r_squared_between': 0.0006,
        'coefficient_vector_json': create_coefficient_vector_json(
            coef=-0.3816296, se=0.0302367, pval=0.000,
            controls=[
                {"var": "sameprov", "coef": 1.010718, "se": 0.1561668, "pval": 0.000},
                {"var": "samepais", "coef": 6.068261, "se": 0.0803655, "pval": 0.000}
            ],
            fixed_effects=["seller_province"],
            diagnostics={"n_groups": 248, "avg_obs_per_group": 28.9}
        ),
        'sample_desc': 'MercadoLibre province-level, distance elasticity',
        'fixed_effects': 'Seller province FE',
        'controls_desc': 'same province, same country, buyer province dummies',
        'cluster_var': 'none (robust SE)',
        'model_type': 'Panel FE (within)',
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
        'data_available': True,
        'notes': 'Results extracted from table_countries_aej.log'
    })

    return results


def document_ebay_specifications():
    """
    Document the eBay specifications from the Stata do files.
    Data is not available, so we document the intended specifications.
    """
    results = []

    # Table 2 specifications (from tables1_2_5_6.do)
    ebay_specs = [
        # Table 2, Model I: Basic gravity
        {
            'spec_id': 'ebay_table2_model1',
            'spec_tree_path': 'methods/panel_fixed_effects.md#fixed-effects-structure',
            'formula': 'lntcount_bs ~ lndist + lnsstate_count + lnbstate_count',
            'controls_desc': 'log seller state count, log buyer state count',
            'fixed_effects': 'None (pooled OLS)',
            'model_type': 'OLS',
            'notes': 'Table 2, Model I - Basic gravity, no FE'
        },
        # Table 2, Model II: Add same state
        {
            'spec_id': 'ebay_table2_model2',
            'spec_tree_path': 'methods/panel_fixed_effects.md#fixed-effects-structure',
            'formula': 'lntcount_bs ~ lndist + samestate + lnsstate_count + lnbstate_count',
            'controls_desc': 'log seller state count, log buyer state count',
            'fixed_effects': 'None (pooled OLS)',
            'model_type': 'OLS',
            'notes': 'Table 2, Model II - Add same state indicator'
        },
        # Table 2, Model III: Buyer and seller state FE (BASELINE)
        {
            'spec_id': 'baseline',
            'spec_tree_path': 'methods/panel_fixed_effects.md#baseline',
            'formula': 'lntcount_bs ~ lndist + samestate | bstate_abb + sstate_abb',
            'controls_desc': 'none (absorbed by FE)',
            'fixed_effects': 'Buyer state FE + Seller state FE (absorbed)',
            'model_type': 'Panel FE (two-way)',
            'notes': 'Table 2, Model III - Main specification with two-way FE'
        },
        # Table 2, Model IV: Volume as outcome
        {
            'spec_id': 'robust/form/volume_outcome',
            'spec_tree_path': 'robustness/functional_form.md',
            'formula': 'lntvol_bs ~ lndist + samestate | bstate_abb + sstate_abb',
            'controls_desc': 'none (absorbed by FE)',
            'fixed_effects': 'Buyer state FE + Seller state FE (absorbed)',
            'model_type': 'Panel FE (two-way)',
            'notes': 'Table 2, Model IV - Transaction volume as outcome'
        },
        # Table 5 specifications
        {
            'spec_id': 'panel/controls/baseline',
            'spec_tree_path': 'methods/panel_fixed_effects.md#control-sets',
            'formula': 'lntcount_bs ~ lndist + samestate + medshipfrac_bs | bstate_abb + sstate_abb',
            'controls_desc': 'median shipping fraction',
            'fixed_effects': 'Buyer state FE + Seller state FE (absorbed)',
            'model_type': 'Panel FE (two-way)',
            'notes': 'Table 5, Model 2 - Add shipping fraction control'
        },
        {
            'spec_id': 'panel/controls/full',
            'spec_tree_path': 'methods/panel_fixed_effects.md#control-sets',
            'formula': 'lntcount_bs ~ lndist + samestate + medshipfrac_bs + sametimezone | bstate_abb + sstate_abb',
            'controls_desc': 'median shipping fraction, same timezone',
            'fixed_effects': 'Buyer state FE + Seller state FE (absorbed)',
            'model_type': 'Panel FE (two-way)',
            'notes': 'Table 5, Model 3 - Add timezone control'
        },
        # Table 4 specifications - Subsamples
        {
            'spec_id': 'panel/sample/category_fe',
            'spec_tree_path': 'methods/panel_fixed_effects.md#sample-restrictions',
            'formula': 'lntcount_bs ~ lndist + samestate | categorytop + bstate_abb + sstate_abb',
            'controls_desc': 'none (absorbed by FE)',
            'fixed_effects': 'Category FE + Buyer state FE + Seller state FE',
            'model_type': 'Panel FE (high-dimensional)',
            'notes': 'Table 4 - Category fixed effects'
        },
        {
            'spec_id': 'panel/sample/video_games',
            'spec_tree_path': 'methods/panel_fixed_effects.md#sample-restrictions',
            'formula': 'lntcount_bs ~ lndist + samestate | bstate_abb + sstate_abb',
            'controls_desc': 'none (absorbed by FE)',
            'fixed_effects': 'Buyer state FE + Seller state FE',
            'model_type': 'Panel FE (two-way)',
            'notes': 'Table 4 - Video Games subsample (categoryexact==62053)'
        },
        {
            'spec_id': 'panel/sample/jewelry',
            'spec_tree_path': 'methods/panel_fixed_effects.md#sample-restrictions',
            'formula': 'lntcount_bs ~ lndist + samestate | bstate_abb + sstate_abb',
            'controls_desc': 'none (absorbed by FE)',
            'fixed_effects': 'Buyer state FE + Seller state FE',
            'model_type': 'Panel FE (two-way)',
            'notes': 'Table 4 - Jewelry subsample (categorytop==281)'
        },
        {
            'spec_id': 'panel/sample/new_buyers',
            'spec_tree_path': 'methods/panel_fixed_effects.md#sample-restrictions',
            'formula': 'lntcount_bs ~ lndist + samestate | bstate_abb + sstate_abb',
            'controls_desc': 'none (absorbed by FE)',
            'fixed_effects': 'Buyer state FE + Seller state FE',
            'model_type': 'Panel FE (two-way)',
            'notes': 'Table 4 - New buyers subsample (buyerfeedback<157)'
        },
        {
            'spec_id': 'panel/sample/sophisticated_buyers',
            'spec_tree_path': 'methods/panel_fixed_effects.md#sample-restrictions',
            'formula': 'lntcount_bs ~ lndist + samestate | bstate_abb + sstate_abb',
            'controls_desc': 'none (absorbed by FE)',
            'fixed_effects': 'Buyer state FE + Seller state FE',
            'model_type': 'Panel FE (two-way)',
            'notes': 'Table 4 - Sophisticated buyers subsample (buyerfeedback>=157)'
        },
        {
            'spec_id': 'panel/sample/good_sellers',
            'spec_tree_path': 'methods/panel_fixed_effects.md#sample-restrictions',
            'formula': 'lntcount_bs ~ lndist + samestate | bstate_abb + sstate_abb',
            'controls_desc': 'none (absorbed by FE)',
            'fixed_effects': 'Buyer state FE + Seller state FE',
            'model_type': 'Panel FE (two-way)',
            'notes': 'Table 4 - Good sellers subsample (sellerfeedbackrating>99.8)'
        },
        {
            'spec_id': 'panel/sample/bad_sellers',
            'spec_tree_path': 'methods/panel_fixed_effects.md#sample-restrictions',
            'formula': 'lntcount_bs ~ lndist + samestate | bstate_abb + sstate_abb',
            'controls_desc': 'none (absorbed by FE)',
            'fixed_effects': 'Buyer state FE + Seller state FE',
            'model_type': 'Panel FE (two-way)',
            'notes': 'Table 4 - Bad sellers subsample (sellerfeedbackrating<99.8)'
        },
        # Table 6 - Heterogeneity by seller reputation and sales taxes
        {
            'spec_id': 'panel/interact/seller_reputation',
            'spec_tree_path': 'methods/panel_fixed_effects.md',
            'formula': 'lntcount_bs ~ lndist + samestate + medshipfrac_bs + lndist*badseller + lndist*verybadseller + samestate*badseller + samestate*verybadseller | bstate_abb + sstate_abb',
            'controls_desc': 'median shipping fraction, seller reputation interactions',
            'fixed_effects': 'Buyer state FE + Seller state FE',
            'model_type': 'Panel FE (two-way)',
            'notes': 'Table 6, Model 1 - Seller reputation interactions'
        },
        {
            'spec_id': 'panel/interact/sales_tax',
            'spec_tree_path': 'methods/panel_fixed_effects.md',
            'formula': 'lntcount_bs ~ lndist + samestate + medshipfrac_bs + lndist*sales_tax_bins + samestate*sales_tax_bins | bstate_abb + sstate_abb',
            'controls_desc': 'median shipping fraction, sales tax interactions',
            'fixed_effects': 'Buyer state FE + Seller state FE',
            'model_type': 'Panel FE (two-way)',
            'notes': 'Table 6, Model 2 - Sales tax interactions'
        },
    ]

    for spec in ebay_specs:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec['spec_id'],
            'spec_tree_path': spec['spec_tree_path'],
            'outcome_var': 'lntcount_bs',
            'treatment_var': 'samestate',
            'coefficient': None,
            'std_error': None,
            't_stat': None,
            'p_value': None,
            'ci_lower': None,
            'ci_upper': None,
            'n_obs': None,
            'r_squared': None,
            'coefficient_vector_json': json.dumps({
                "treatment": {"var": "samestate", "coef": None, "se": None, "pval": None},
                "controls": [],
                "fixed_effects_absorbed": [],
                "diagnostics": {},
                "formula": spec['formula']
            }),
            'sample_desc': 'eBay US transactions (data unavailable)',
            'fixed_effects': spec['fixed_effects'],
            'controls_desc': spec['controls_desc'],
            'cluster_var': 'robust',
            'model_type': spec['model_type'],
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
            'data_available': False,
            'notes': f"{spec['notes']} - DATA UNAVAILABLE due to confidentiality"
        })

    return results


def generate_robustness_specs():
    """Generate robustness specification templates (data not available)."""
    results = []

    robustness_specs = [
        # Clustering variations
        {'spec_id': 'robust/cluster/none', 'spec_tree_path': 'robustness/clustering_variations.md#single-level-clustering',
         'notes': 'Robust (heteroskedasticity-robust) SE, no clustering'},
        {'spec_id': 'robust/cluster/sstate', 'spec_tree_path': 'robustness/clustering_variations.md#single-level-clustering',
         'notes': 'Cluster by seller state'},
        {'spec_id': 'robust/cluster/bstate', 'spec_tree_path': 'robustness/clustering_variations.md#single-level-clustering',
         'notes': 'Cluster by buyer state'},
        {'spec_id': 'robust/cluster/twoway', 'spec_tree_path': 'robustness/clustering_variations.md#two-way-clustering',
         'notes': 'Two-way clustering by buyer and seller state'},

        # Leave-one-out
        {'spec_id': 'robust/loo/drop_medshipfrac', 'spec_tree_path': 'robustness/leave_one_out.md',
         'notes': 'Drop median shipping fraction control'},
        {'spec_id': 'robust/loo/drop_sametimezone', 'spec_tree_path': 'robustness/leave_one_out.md',
         'notes': 'Drop same timezone control'},

        # Single covariate
        {'spec_id': 'robust/single/none', 'spec_tree_path': 'robustness/single_covariate.md',
         'notes': 'Bivariate: samestate only (no distance, no controls)'},
        {'spec_id': 'robust/single/distance', 'spec_tree_path': 'robustness/single_covariate.md',
         'notes': 'samestate + log distance only'},
        {'spec_id': 'robust/single/medshipfrac', 'spec_tree_path': 'robustness/single_covariate.md',
         'notes': 'samestate + median shipping fraction only'},

        # Functional form
        {'spec_id': 'robust/form/y_level', 'spec_tree_path': 'robustness/functional_form.md',
         'notes': 'Transaction counts in levels (not log)'},
        {'spec_id': 'robust/form/poisson', 'spec_tree_path': 'robustness/functional_form.md',
         'notes': 'Poisson regression for count outcome'},
    ]

    for spec in robustness_specs:
        results.append({
            'paper_id': PAPER_ID,
            'journal': JOURNAL,
            'paper_title': PAPER_TITLE,
            'spec_id': spec['spec_id'],
            'spec_tree_path': spec['spec_tree_path'],
            'outcome_var': 'lntcount_bs',
            'treatment_var': 'samestate',
            'coefficient': None,
            'std_error': None,
            't_stat': None,
            'p_value': None,
            'ci_lower': None,
            'ci_upper': None,
            'n_obs': None,
            'r_squared': None,
            'coefficient_vector_json': json.dumps({
                "treatment": {"var": "samestate", "coef": None, "se": None, "pval": None},
                "controls": [],
                "fixed_effects_absorbed": ["bstate_abb", "sstate_abb"],
                "diagnostics": {}
            }),
            'sample_desc': 'eBay US transactions (data unavailable)',
            'fixed_effects': 'Buyer state FE + Seller state FE',
            'controls_desc': 'varies by specification',
            'cluster_var': 'varies by specification',
            'model_type': 'Panel FE',
            'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py',
            'data_available': False,
            'notes': f"{spec['notes']} - DATA UNAVAILABLE due to confidentiality"
        })

    return results


def main():
    """Main function to run specification search."""
    print("=" * 70)
    print(f"Specification Search: {PAPER_ID}")
    print(f"Paper: {PAPER_TITLE}")
    print("=" * 70)
    print()

    # Load auxiliary data (for documentation purposes)
    print("Loading auxiliary data files...")
    aux_data = load_auxiliary_data()
    print()

    # Collect all results
    all_results = []

    # 1. Extract results from available log files (MercadoLibre)
    print("Extracting results from log files...")
    log_results = extract_results_from_log()
    all_results.extend(log_results)
    print(f"  Extracted {len(log_results)} specifications from log files")

    # 2. Document eBay specifications (data not available)
    print("Documenting eBay specifications (data unavailable)...")
    ebay_results = document_ebay_specifications()
    all_results.extend(ebay_results)
    print(f"  Documented {len(ebay_results)} eBay specifications")

    # 3. Generate robustness specifications (data not available)
    print("Generating robustness specifications (data unavailable)...")
    robustness_results = generate_robustness_specs()
    all_results.extend(robustness_results)
    print(f"  Generated {len(robustness_results)} robustness specifications")

    # Convert to DataFrame
    df_results = pd.DataFrame(all_results)

    # Save results
    output_path = os.path.join(OUTPUT_DIR, 'specification_results.csv')
    df_results.to_csv(output_path, index=False)
    print(f"\nSaved {len(df_results)} specifications to {output_path}")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    # Only for results with actual data
    df_with_data = df_results[df_results['data_available'] == True].copy()
    if len(df_with_data) > 0:
        df_with_data['significant_05'] = df_with_data['p_value'] < 0.05
        df_with_data['significant_01'] = df_with_data['p_value'] < 0.01
        df_with_data['positive'] = df_with_data['coefficient'] > 0

        print(f"\nSpecifications with actual data: {len(df_with_data)}")
        print(f"  Positive coefficients: {df_with_data['positive'].sum()} ({100*df_with_data['positive'].mean():.1f}%)")
        print(f"  Significant at 5%: {df_with_data['significant_05'].sum()} ({100*df_with_data['significant_05'].mean():.1f}%)")
        print(f"  Significant at 1%: {df_with_data['significant_01'].sum()} ({100*df_with_data['significant_01'].mean():.1f}%)")

        # For MercadoLibre results
        print("\n  MercadoLibre Results (from log files):")
        for _, row in df_with_data.iterrows():
            print(f"    {row['spec_id']}: coef={row['coefficient']:.4f}, se={row['std_error']:.4f}, p={row['p_value']:.3f}")

    print(f"\nSpecifications documented (data unavailable): {len(df_results[df_results['data_available'] == False])}")

    print("\n" + "=" * 70)
    print("DATA LIMITATIONS")
    print("=" * 70)
    print("""
The main transaction data files are confidential and not available:
- all_US_items.dta (eBay US transactions)
- datareg_pais (MercadoLibre country-level)
- datareg_prov (MercadoLibre province-level)

Only auxiliary files with state/country characteristics are available:
- buyersellerstateinfo.dta (US state pair distances, GDP, population)
- bsales_taxes.dta, ssales_taxes.dta (state sales tax rates)
- btimezones.dta, stimezones.dta (state time zones)
- categories_classified.dta (product category classifications)

Results from MercadoLibre analysis are available in table_countries_aej.log
and have been extracted for this specification search.
""")

    return df_results


if __name__ == "__main__":
    results = main()
