"""
Specification Search Script for Paper 195428-V1
"Fueling Alternatives: Gas Station Choice and the Implications for Electric Charging"
by Jackson Dorsey, Ashley Langer, and Shaun McRae

STATUS: CANNOT EXECUTE - DATA NOT AVAILABLE

The core analysis data is confidential/proprietary and not included in the
public replication package:
- IVBSS driving behavior data (confidential)
- OPIS gasoline station and price data (proprietary)
- Mechanical Turk fuel gauge classifications (personal identifiers)

This script documents what specifications WOULD have been run if data were available.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Paper metadata
PAPER_ID = "195428-V1"
PAPER_TITLE = "Fueling Alternatives: Gas Station Choice and the Implications for Electric Charging"
AUTHORS = "Jackson Dorsey, Ashley Langer, and Shaun McRae"
JOURNAL = "AEJ: Policy"
METHOD = "discrete_choice"
METHOD_TREE_PATH = "methods/discrete_choice.md"

# Output paths
PKG_DIR = Path("/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/195428-V1")
RESULTS_FILE = PKG_DIR / "specification_results.csv"

# =============================================================================
# DATA REQUIREMENTS (NOT MET)
# =============================================================================
"""
Required data files that are NOT available:
1. cache/estimation_data.rda - Main estimation dataset
2. cache/simulation_data.rda - EV charging simulation data
3. raw/ivbss/*.csv - Driving behavior (GPS, fuel consumption)
4. raw/opis/*.dta, *.xlsx - Gas station locations and prices
5. raw/mturk/*.csv - Fuel gauge classifications
"""

# =============================================================================
# MODEL STRUCTURE (from code inspection)
# =============================================================================
"""
The paper estimates a nested logit model of gas station choice.

Key Variables:
- Outcome: chosen (binary - whether station/option was chosen)
- Choice sets: choice_set (trip-level identifier)
- Inside option: inside_option (1 if gas station, 0 if "no stop")
- Outside option: outside_option (1 if "no stop" option)

Model Parameters:
- lambda: Nesting parameter (0-1), measures correlation within inside options
- alpha: Price coefficient (on expected expenditure)
- theta: Weight on current vs. average price (imperfect info models only)
- beta_excess_time: Disutility of extra travel time to station
- beta_inside_option*tank_level: Utility of stopping varies with fuel level

Utility Specification:
U_ij = alpha * E[expenditure] + beta_excess_time * excess_time
       + sum(beta_k * X_k) + brand_FE + month_year_FE + epsilon

Where for imperfect info models:
E[price] = theta * current_price + (1-theta) * avg_price
"""

# =============================================================================
# PLANNED SPECIFICATIONS (50+)
# =============================================================================

planned_specs = []

# -----------------------------------------------------------------------------
# BASELINE: Table 5 (Models 1-4)
# -----------------------------------------------------------------------------
baseline_models = [
    {"spec_id": "baseline/model1",
     "description": "Perfect info, all stations, with brand FE",
     "info_structure": "perfect",
     "choice_set": "all",
     "brand_fe": True},
    {"spec_id": "baseline/model2",
     "description": "Perfect info, passed stations only, with brand FE",
     "info_structure": "perfect",
     "choice_set": "passed",
     "brand_fe": True},
    {"spec_id": "baseline/model3",
     "description": "Imperfect info (theta estimated), all stations, with brand FE",
     "info_structure": "imperfect",
     "choice_set": "all",
     "brand_fe": True},
    {"spec_id": "baseline/model4",
     "description": "Imperfect info, passed stations only, with brand FE",
     "info_structure": "imperfect",
     "choice_set": "passed",
     "brand_fe": True},
]
planned_specs.extend(baseline_models)

# -----------------------------------------------------------------------------
# FIXED EFFECTS VARIATIONS: Appendix Table A3
# -----------------------------------------------------------------------------
fe_variations = [
    {"spec_id": "robust/fe/no_brand_model1",
     "description": "Perfect info, all stations, NO brand FE",
     "brand_fe": False,
     "month_year_fe": True},
    {"spec_id": "robust/fe/no_brand_model2",
     "description": "Perfect info, passed only, NO brand FE",
     "brand_fe": False,
     "month_year_fe": True},
    {"spec_id": "robust/fe/no_brand_model3",
     "description": "Imperfect info, all stations, NO brand FE",
     "brand_fe": False,
     "month_year_fe": True},
    {"spec_id": "robust/fe/no_brand_model4",
     "description": "Imperfect info, passed only, NO brand FE",
     "brand_fe": False,
     "month_year_fe": True},
    {"spec_id": "robust/fe/no_monthyear_model1",
     "description": "Perfect info, all stations, no month-year FE",
     "brand_fe": True,
     "month_year_fe": False},
    {"spec_id": "robust/fe/no_monthyear_model3",
     "description": "Imperfect info, all stations, no month-year FE",
     "brand_fe": True,
     "month_year_fe": False},
    {"spec_id": "robust/fe/no_fe_model1",
     "description": "Perfect info, all stations, no FE",
     "brand_fe": False,
     "month_year_fe": False},
    {"spec_id": "robust/fe/no_fe_model3",
     "description": "Imperfect info, all stations, no FE",
     "brand_fe": False,
     "month_year_fe": False},
]
planned_specs.extend(fe_variations)

# -----------------------------------------------------------------------------
# CONTROL VARIATIONS
# -----------------------------------------------------------------------------
control_variations = [
    {"spec_id": "robust/control/no_tank_level",
     "description": "Drop tank level controls from outside option utility"},
    {"spec_id": "robust/control/linear_tank_only",
     "description": "Only linear tank level (drop squared term)"},
    {"spec_id": "robust/control/tank_level_cubic",
     "description": "Add cubic tank level term"},
    {"spec_id": "robust/control/no_excess_time",
     "description": "Drop excess travel time from station utility"},
    {"spec_id": "robust/control/excess_time_squared",
     "description": "Add quadratic excess time term"},
]
planned_specs.extend(control_variations)

# -----------------------------------------------------------------------------
# ESTIMATION METHOD VARIATIONS
# -----------------------------------------------------------------------------
estimation_variations = [
    {"spec_id": "robust/estimation/standard_logit",
     "description": "Standard logit (lambda=1, no nesting)",
     "nested_logit": False},
    {"spec_id": "robust/estimation/nested_logit_all",
     "description": "Nested logit with all stations (baseline)",
     "nested_logit": True,
     "choice_set": "all"},
    {"spec_id": "robust/estimation/nested_logit_passed",
     "description": "Nested logit with passed stations only",
     "nested_logit": True,
     "choice_set": "passed"},
    {"spec_id": "robust/estimation/mixed_logit",
     "description": "Mixed logit with random coefficients on price/time"},
    {"spec_id": "robust/estimation/conditional_logit",
     "description": "Conditional logit with driver FE"},
]
planned_specs.extend(estimation_variations)

# -----------------------------------------------------------------------------
# SAMPLE RESTRICTIONS
# -----------------------------------------------------------------------------
sample_restrictions = [
    {"spec_id": "robust/sample/weekday_only",
     "description": "Weekday trips only"},
    {"spec_id": "robust/sample/weekend_only",
     "description": "Weekend trips only"},
    {"spec_id": "robust/sample/spring_2009",
     "description": "April-June 2009 only"},
    {"spec_id": "robust/sample/fall_2009",
     "description": "October-December 2009 only"},
    {"spec_id": "robust/sample/winter_2010",
     "description": "January-March 2010 only"},
    {"spec_id": "robust/sample/drop_first_week",
     "description": "Drop first week of each driver"},
    {"spec_id": "robust/sample/drop_short_trips",
     "description": "Drop trips < 5 miles"},
    {"spec_id": "robust/sample/drop_long_trips",
     "description": "Drop trips > 50 miles"},
    {"spec_id": "robust/sample/low_tank_only",
     "description": "Only trips with tank < 1/4 full"},
    {"spec_id": "robust/sample/high_tank_only",
     "description": "Only trips with tank > 1/2 full"},
]
planned_specs.extend(sample_restrictions)

# -----------------------------------------------------------------------------
# HETEROGENEITY: Appendix Table D1
# -----------------------------------------------------------------------------
heterogeneity_specs = [
    {"spec_id": "robust/heterogeneity/age_young",
     "description": "Interaction with age 20-30 dummy",
     "interaction_var": "age_cat_age_20to30"},
    {"spec_id": "robust/heterogeneity/age_middle",
     "description": "Interaction with age 40-50 dummy",
     "interaction_var": "age_cat_age_40to50"},
    {"spec_id": "robust/heterogeneity/age_senior",
     "description": "Interaction with age 60-70 dummy",
     "interaction_var": "age_cat_age_60to70"},
    {"spec_id": "robust/heterogeneity/gender",
     "description": "Interaction with female dummy",
     "interaction_var": "female"},
    {"spec_id": "robust/heterogeneity/income_high",
     "description": "Interaction with high income dummy",
     "interaction_var": "high_income"},
    {"spec_id": "robust/heterogeneity/income_low",
     "description": "Interaction with low income dummy",
     "interaction_var": "low_income"},
    {"spec_id": "robust/heterogeneity/weekday",
     "description": "Interaction with weekday dummy",
     "interaction_var": "weekday"},
    {"spec_id": "robust/heterogeneity/garage",
     "description": "Interaction with home has garage dummy",
     "interaction_var": "garage"},
    {"spec_id": "robust/heterogeneity/urban",
     "description": "Interaction with urban home location",
     "interaction_var": "urban"},
    {"spec_id": "robust/heterogeneity/commute_trip",
     "description": "Interaction with work commute indicator",
     "interaction_var": "commute_trip"},
]
planned_specs.extend(heterogeneity_specs)

# -----------------------------------------------------------------------------
# INFERENCE VARIATIONS
# -----------------------------------------------------------------------------
inference_variations = [
    {"spec_id": "robust/inference/se_robust",
     "description": "Robust (sandwich) standard errors"},
    {"spec_id": "robust/inference/se_cluster_driver",
     "description": "Cluster SE at driver level"},
    {"spec_id": "robust/inference/se_cluster_station",
     "description": "Cluster SE at station level"},
    {"spec_id": "robust/inference/se_cluster_day",
     "description": "Cluster SE at day level"},
    {"spec_id": "robust/inference/bootstrap_100",
     "description": "Bootstrap SE with 100 replications"},
    {"spec_id": "robust/inference/bootstrap_500",
     "description": "Bootstrap SE with 500 replications (paper uses this)"},
]
planned_specs.extend(inference_variations)

# -----------------------------------------------------------------------------
# FUNCTIONAL FORM VARIATIONS
# -----------------------------------------------------------------------------
functional_form = [
    {"spec_id": "robust/funcform/log_price",
     "description": "Log of expected expenditure instead of levels"},
    {"spec_id": "robust/funcform/price_per_gallon",
     "description": "Price per gallon instead of expected expenditure"},
    {"spec_id": "robust/funcform/log_excess_time",
     "description": "Log of excess time"},
    {"spec_id": "robust/funcform/piecewise_time",
     "description": "Piecewise linear excess time (0-5, 5-10, 10+ minutes)"},
    {"spec_id": "robust/funcform/interaction_price_time",
     "description": "Interaction between price and excess time"},
]
planned_specs.extend(functional_form)

# -----------------------------------------------------------------------------
# ALTERNATIVE TREATMENT DEFINITIONS
# -----------------------------------------------------------------------------
treatment_variations = [
    {"spec_id": "robust/treatment/current_price_only",
     "description": "Use only current observed price (no averaging)"},
    {"spec_id": "robust/treatment/avg_price_only",
     "description": "Use only long-run average price"},
    {"spec_id": "robust/treatment/rolling_avg_7day",
     "description": "Use 7-day rolling average price"},
    {"spec_id": "robust/treatment/rolling_avg_30day",
     "description": "Use 30-day rolling average price"},
]
planned_specs.extend(treatment_variations)

# =============================================================================
# OUTPUT SUMMARY
# =============================================================================

print(f"Paper: {PAPER_TITLE}")
print(f"Paper ID: {PAPER_ID}")
print(f"Journal: {JOURNAL}")
print(f"Method: {METHOD}")
print(f"\nTotal planned specifications: {len(planned_specs)}")
print(f"\nBreakdown by category:")

categories = {}
for spec in planned_specs:
    cat = spec['spec_id'].split('/')[0]
    if cat not in categories:
        categories[cat] = 0
    categories[cat] += 1

for cat, count in categories.items():
    print(f"  {cat}: {count}")

print("\n" + "="*60)
print("STATUS: DATA NOT AVAILABLE - SPECIFICATIONS CANNOT BE RUN")
print("="*60)
print("""
To replicate this paper, you would need to:
1. Contact Jim Sayer at UMTRI (jimsayer@umich.edu) for IVBSS data access
2. Purchase OPIS gasoline price data (energysales@opisnet.com)
3. Potentially recreate MTurk fuel gauge classifications

The paper's structural model estimation requires all of these data sources
to be combined into the estimation_data.rda file used by the nested logit
likelihood function.
""")
