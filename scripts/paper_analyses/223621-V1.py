"""
Specification Search Script for Paper 223621-V1
"Micro vs Macro Labor Supply Elasticities: The Role of Dynamic Returns to Effort"
Kleven, Kreiner, Larsen, Sogaard - AER

NOTE: This script is a TEMPLATE ONLY.
The underlying data is CONFIDENTIAL and only accessible via Statistics Denmark secure servers.
This script documents the specifications that WOULD be run if data were available.

Method: Instrumental Variables (2SLS)
- Endogenous: Change in log net-of-tax rate (dMTR_mover)
- Instrument: Simulated tax change based on reform (dMTR_sim_mover_btw)
- Outcome: Change in log wage income (dln_wi)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Would use these if data were available:
# import pyfixest as pf
# from linearmodels.iv import IV2SLS, IVLIML
# from linearmodels.panel import PanelOLS

PAPER_ID = "223621-V1"
JOURNAL = "AER"
PAPER_TITLE = "Micro vs Macro Labor Supply Elasticities: The Role of Dynamic Returns to Effort"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "downloads" / "extracted" / PAPER_ID

# =============================================================================
# DATA LOADING (TEMPLATE - DATA NOT AVAILABLE)
# =============================================================================

def load_and_prepare_data(reform_year: int, lead: int):
    """
    Template for data loading and preparation.

    In actual implementation, would:
    1. Load "TempData/Reform {reform_year}.dta"
    2. Keep years: base_year, base_year+2, base_year+lead
    3. Create lagged variables
    4. Define switchers (firm and occupation)
    5. Apply sample restrictions

    Parameters:
    -----------
    reform_year : int
        Either 1987 or 2009
    lead : int
        Horizon for outcome (2 or 4 years)

    Returns:
    --------
    pd.DataFrame
    """
    raise NotImplementedError("Data is confidential - cannot load outside Statistics Denmark")


def define_variables(df: pd.DataFrame):
    """
    Template for variable definitions based on original Stata code.
    """
    # Outcome: change in log wage income
    # df['dln_wi'] = np.log(df['wi_l']) - np.log(df['wi'])

    # Treatment (endogenous): change in log net-of-tax rate for movers
    # df['dMTR'] = np.log(1 - df['MTR_l']) - np.log(1 - df['MTR'])
    # df['dMTR_mover'] = df['dMTR'] * df['switch']

    # Instrument: simulated tax change (within bin mean)
    # df['dMTR_sim_mover_btw'] = df.groupby(['year', 'bin'])['dMTR_sim_2y'].transform('mean') * df['switch']

    # Fixed effects
    # df['job'] = df['cvrnr'] + '_' + df['occupation']

    # Controls
    # df['married'] = (df['civst'] == 3).astype(int)
    # df['male'] = 1 - df['female']
    # df['college'] = df['audd_hoved_l5'].isin([40, 50, 60, 70]).astype(int)

    pass


def apply_sample_restrictions(df: pd.DataFrame):
    """
    Apply sample restrictions from original code (v1b.do).

    Restrictions:
    - Age 20-60
    - Wage income > 12 (in thousands DKK)
    - Valid firm identifier
    - Personal income 250-1000 (thousands DKK)
    - Non-missing MTR
    - Leave-one-out share < 50%
    """
    pass


# =============================================================================
# SPECIFICATION DEFINITIONS
# =============================================================================

SPECIFICATIONS = {
    # Main results (Table 1)
    "baseline": {
        "spec_id": "baseline",
        "spec_tree_path": "methods/instrumental_variables.md#baseline",
        "description": "2SLS with job FE, 2-year horizon, 2009 reform",
        "model_type": "IV2SLS",
        "outcome": "dln_wi",
        "endogenous": "dMTR_mover",
        "instrument": "dMTR_sim_mover_btw",
        "fixed_effects": ["job", "bin_switch", "year_switch"],
        "controls": ["dMTR_sim_btw"],
        "cluster_var": None,
        "vcov": "robust",
        "sample": "full",
        "reform_year": 2009,
        "lead": 2
    },

    # First stage
    "iv/first_stage/baseline": {
        "spec_id": "iv/first_stage/baseline",
        "spec_tree_path": "methods/instrumental_variables.md#first-stage",
        "description": "First stage: dMTR_mover on dMTR_sim_mover_btw",
        "model_type": "OLS",
        "outcome": "dMTR_mover",
        "treatment": "dMTR_sim_mover_btw",
        "fixed_effects": ["job", "bin_switch", "year_switch"],
        "controls": ["dMTR_sim_btw"],
        "vcov": "robust"
    },

    # Reduced form
    "iv/first_stage/reduced_form": {
        "spec_id": "iv/first_stage/reduced_form",
        "spec_tree_path": "methods/instrumental_variables.md#first-stage",
        "description": "Reduced form: dln_wi on dMTR_sim_mover_btw",
        "model_type": "OLS",
        "outcome": "dln_wi",
        "treatment": "dMTR_sim_mover_btw",
        "fixed_effects": ["job", "bin_switch", "year_switch"],
        "controls": ["dMTR_sim_btw"],
        "vcov": "robust"
    },

    # OLS (ignoring endogeneity)
    "iv/method/ols": {
        "spec_id": "iv/method/ols",
        "spec_tree_path": "methods/instrumental_variables.md#estimation-method",
        "description": "OLS (ignoring endogeneity)",
        "model_type": "OLS",
        "outcome": "dln_wi",
        "treatment": "dMTR_mover",
        "fixed_effects": ["job", "bin_switch", "year_switch"],
        "vcov": "robust"
    },

    # No controls
    "iv/controls/none": {
        "spec_id": "iv/controls/none",
        "spec_tree_path": "methods/instrumental_variables.md#control-sets",
        "description": "2SLS without job FE",
        "model_type": "IV2SLS",
        "outcome": "dln_wi",
        "endogenous": "dMTR_mover",
        "instrument": "dMTR_sim_mover_btw",
        "fixed_effects": ["bin_switch", "year_switch"],
        "controls": ["dMTR_sim_btw"],
        "vcov": "robust"
    },

    # Demographics only
    "iv/controls/minimal": {
        "spec_id": "iv/controls/minimal",
        "spec_tree_path": "methods/instrumental_variables.md#control-sets",
        "description": "2SLS with demographic controls only",
        "model_type": "IV2SLS",
        "outcome": "dln_wi",
        "endogenous": "dMTR_mover",
        "instrument": "dMTR_sim_mover_btw",
        "fixed_effects": ["bin_switch", "year_switch"],
        "controls": ["dMTR_sim_btw", "age_fe", "male", "kids", "married", "log_firm_size", "occupation_rank"],
        "vcov": "robust"
    },

    # Full controls with job FE
    "iv/controls/full": {
        "spec_id": "iv/controls/full",
        "spec_tree_path": "methods/instrumental_variables.md#control-sets",
        "description": "2SLS with job FE and demographic controls",
        "model_type": "IV2SLS",
        "outcome": "dln_wi",
        "endogenous": "dMTR_mover",
        "instrument": "dMTR_sim_mover_btw",
        "fixed_effects": ["job", "bin_switch", "year_switch"],
        "controls": ["dMTR_sim_btw", "age_fe", "male", "kids", "married"],
        "vcov": "robust"
    },

    # Donut specifications
    "robust/sample/donut_small": {
        "spec_id": "robust/sample/donut_small",
        "spec_tree_path": "robustness/sample_restrictions.md",
        "description": "Exclude 2.5% around kink threshold",
        "model_type": "IV2SLS",
        "outcome": "dln_wi",
        "endogenous": "dMTR_mover",
        "instrument": "dMTR_sim_mover_btw",
        "fixed_effects": ["job", "bin_switch", "year_switch"],
        "sample_restriction": "donut1 == 0"
    },

    "robust/sample/donut_large": {
        "spec_id": "robust/sample/donut_large",
        "spec_tree_path": "robustness/sample_restrictions.md",
        "description": "Exclude 5% around kink threshold",
        "model_type": "IV2SLS",
        "outcome": "dln_wi",
        "endogenous": "dMTR_mover",
        "instrument": "dMTR_sim_mover_btw",
        "fixed_effects": ["job", "bin_switch", "year_switch"],
        "sample_restriction": "donut2 == 0"
    },

    # Top earner subsamples
    "robust/sample/top10": {
        "spec_id": "robust/sample/top10",
        "spec_tree_path": "robustness/sample_restrictions.md",
        "description": "Top 10% earners only",
        "model_type": "IV2SLS",
        "sample_restriction": "top10 == 1"
    },

    "robust/sample/top5": {
        "spec_id": "robust/sample/top5",
        "spec_tree_path": "robustness/sample_restrictions.md",
        "description": "Top 5% earners only",
        "model_type": "IV2SLS",
        "sample_restriction": "top5 == 1"
    },

    # Switcher type variations
    "robust/switcher_type/firm": {
        "spec_id": "robust/switcher_type/firm",
        "spec_tree_path": "robustness/sample_restrictions.md",
        "description": "Firm switchers only",
        "model_type": "IV2SLS",
        "sample_restriction": "switch == switch_firm"
    },

    "robust/switcher_type/occupation": {
        "spec_id": "robust/switcher_type/occupation",
        "spec_tree_path": "robustness/sample_restrictions.md",
        "description": "Occupation switchers only (within firm)",
        "model_type": "IV2SLS",
        "sample_restriction": "switch == switch_disco_all"
    },

    "robust/switcher_type/masslayoff": {
        "spec_id": "robust/switcher_type/masslayoff",
        "spec_tree_path": "robustness/sample_restrictions.md",
        "description": "Mass layoff switchers only (exogenous)",
        "model_type": "IV2SLS",
        "sample_restriction": "switch == masslayoff"
    },

    # Horizon variations (would need separate data prep)
    "robust/horizon/4year": {
        "spec_id": "robust/horizon/4year",
        "spec_tree_path": "custom/horizons.md",
        "description": "4-year outcome horizon",
        "model_type": "IV2SLS",
        "lead": 4
    },

    # Standard approach (all workers)
    "standard_approach": {
        "spec_id": "standard_approach",
        "spec_tree_path": "custom/comparison.md",
        "description": "Standard micro approach including all workers",
        "model_type": "IV2SLS",
        "outcome": "dln_wi",
        "endogenous": "standard_dMTR_mover",  # Scaled by switcher probability
        "instrument": "dMTR_sim_mover_btw",
        "note": "Comparison specification - standard micro elasticity approach"
    }
}

# Also add specs for 1987 reform
for spec_id in list(SPECIFICATIONS.keys()):
    if "reform" not in spec_id:
        new_id = f"{spec_id}_1987"
        new_spec = SPECIFICATIONS[spec_id].copy()
        new_spec["spec_id"] = new_id
        new_spec["reform_year"] = 1987
        new_spec["description"] = new_spec.get("description", "") + " (1987 reform)"
        SPECIFICATIONS[new_id] = new_spec


# =============================================================================
# ESTIMATION FUNCTIONS (TEMPLATES)
# =============================================================================

def run_iv_specification(df: pd.DataFrame, spec: dict) -> dict:
    """
    Run a single IV specification.

    Template showing what would be run with pyfixest or linearmodels.
    """
    # Would use pyfixest syntax like:
    # model = pf.feols(
    #     "dln_wi ~ dMTR_sim_btw | job + bin_switch + year_switch | dMTR_mover ~ dMTR_sim_mover_btw",
    #     data=df,
    #     vcov="hetero"
    # )

    # Or linearmodels IV2SLS:
    # from linearmodels.iv import IV2SLS
    # model = IV2SLS.from_formula(
    #     "dln_wi ~ 1 + dMTR_sim_btw + EntityEffects + TimeEffects + [dMTR_mover ~ dMTR_sim_mover_btw]",
    #     data=df.set_index(['pnr', 'year'])
    # )
    # result = model.fit(cov_type='robust')

    raise NotImplementedError("Cannot run - data not available")


def run_ols_specification(df: pd.DataFrame, spec: dict) -> dict:
    """
    Run OLS specification (first stage, reduced form, or naive OLS).
    """
    # Would use:
    # model = pf.feols(
    #     "y ~ x | fe1 + fe2",
    #     data=df,
    #     vcov="hetero"
    # )
    raise NotImplementedError("Cannot run - data not available")


# =============================================================================
# RESULTS FORMATTING
# =============================================================================

def format_result(spec: dict, model_result) -> dict:
    """
    Format estimation results into standard output format.
    """
    # Extract key coefficient (treatment/endogenous variable)
    treatment_var = spec.get("endogenous", spec.get("treatment", "dMTR_mover"))

    result = {
        "paper_id": PAPER_ID,
        "journal": JOURNAL,
        "paper_title": PAPER_TITLE,
        "spec_id": spec["spec_id"],
        "spec_tree_path": spec["spec_tree_path"],
        "outcome_var": spec.get("outcome", "dln_wi"),
        "treatment_var": treatment_var,
        # These would be filled from model_result:
        "coefficient": None,  # model_result.params[treatment_var]
        "std_error": None,    # model_result.std_errors[treatment_var]
        "t_stat": None,       # model_result.tstats[treatment_var]
        "p_value": None,      # model_result.pvalues[treatment_var]
        "ci_lower": None,
        "ci_upper": None,
        "n_obs": None,        # model_result.nobs
        "r_squared": None,    # model_result.rsquared if OLS
        "coefficient_vector_json": None,  # Full coefficient vector
        "sample_desc": spec.get("description", ""),
        "fixed_effects": str(spec.get("fixed_effects", [])),
        "controls_desc": str(spec.get("controls", [])),
        "cluster_var": spec.get("cluster_var", "robust"),
        "model_type": spec["model_type"],
        "estimation_script": f"scripts/paper_analyses/{PAPER_ID}.py"
    }

    return result


def build_coefficient_vector_json(model_result, spec: dict) -> str:
    """
    Build the full coefficient vector JSON.
    """
    coef_dict = {
        "treatment": {
            "var": spec.get("endogenous", spec.get("treatment")),
            "coef": None,
            "se": None,
            "pval": None
        },
        "controls": [],
        "fixed_effects": spec.get("fixed_effects", []),
        "diagnostics": {
            "first_stage_F": None,
            "overid_pval": None,
            "hausman_pval": None
        }
    }

    return json.dumps(coef_dict)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function.

    This would:
    1. Load data
    2. Run all specifications
    3. Save results to CSV
    """

    print(f"=" * 60)
    print(f"Specification Search: {PAPER_ID}")
    print(f"=" * 60)
    print()
    print("STATUS: CANNOT EXECUTE")
    print("REASON: Data is confidential administrative data from Statistics Denmark")
    print()
    print(f"Number of specifications defined: {len(SPECIFICATIONS)}")
    print()
    print("Specifications that would be run:")
    print("-" * 40)

    for spec_id, spec in SPECIFICATIONS.items():
        print(f"  - {spec_id}: {spec.get('description', 'No description')}")

    print()
    print("-" * 40)
    print("To replicate this analysis:")
    print("1. Obtain access to Statistics Denmark research servers")
    print("2. Request project #707853")
    print("3. Run original Stata code or translate to Python/R")
    print("-" * 40)

    # Create empty results file to indicate attempted but blocked
    results_df = pd.DataFrame({
        "paper_id": [PAPER_ID],
        "journal": [JOURNAL],
        "paper_title": [PAPER_TITLE],
        "spec_id": ["NO_DATA_AVAILABLE"],
        "spec_tree_path": ["N/A"],
        "outcome_var": ["dln_wi"],
        "treatment_var": ["dMTR_mover"],
        "coefficient": [np.nan],
        "std_error": [np.nan],
        "t_stat": [np.nan],
        "p_value": [np.nan],
        "ci_lower": [np.nan],
        "ci_upper": [np.nan],
        "n_obs": [np.nan],
        "r_squared": [np.nan],
        "coefficient_vector_json": [None],
        "sample_desc": ["Data unavailable - confidential administrative data"],
        "fixed_effects": ["N/A"],
        "controls_desc": ["N/A"],
        "cluster_var": ["N/A"],
        "model_type": ["IV2SLS"],
        "estimation_script": [f"scripts/paper_analyses/{PAPER_ID}.py"],
        "status": ["BLOCKED_NO_DATA"]
    })

    output_path = OUTPUT_DIR / "specification_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nPlaceholder results saved to: {output_path}")

    return results_df


if __name__ == "__main__":
    main()
