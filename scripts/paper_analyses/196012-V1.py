"""
Specification Search Analysis: 196012-V1
"Regulating Conglomerates: Evidence from an Energy Conservation Program in China"

Authors: Chen, Chen, Liu, Suarez Serrato, Xu
Journal: American Economic Review

STATUS: DATA NOT AVAILABLE

This script documents the specification search that WOULD have been conducted
if the proprietary data were available. The data is not included in the
replication package due to confidentiality restrictions.

Data Sources Required (all proprietary):
- NBS: National Bureau of Statistics firm survey (2001-2009, 2011)
- CMEP: Chinese Ministry of Environmental Protection pollution data (1998-2010)
- ATS: Annual Tax Survey (2008-2011)
- SAIC: State Administration for Industry and Commerce registration (1949-2018)

Contact for Data Access:
- Fudan University Data Center: econdata@fudan.edu.cn
- Lead author: Qiaoyi Chen (joychen@fudan.edu.cn)
"""

import pandas as pd
import numpy as np
import json

# Configuration
PAPER_ID = "196012-V1"
PACKAGE_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/196012-V1/"

# Method classification
METHOD_CODE = "difference_in_differences"
METHOD_TREE_PATH = "specification_tree/methods/difference_in_differences.md"

# Main hypothesis
HYPOTHESIS = """
China's Top 1,000 Energy-Consuming Enterprises program (launched 2006) led
regulated firms to reduce energy consumption while maintaining output through
within-conglomerate production reallocation.
"""

# Key variables (from the do files)
OUTCOMES = ["lenergy", "loutput", "lefficiency"]
TREATMENT = "treat_post"  # Treatment indicator for Top 1,000 firms after 2007
CONTROLS = ["soe", "roa", "age", "exp"]  # State-owned, ROA, age, exporter

# Fixed effect structures used in paper
FE_STRUCTURES = {
    "basic": "firmid year",
    "industry_time": "firmid ind2_year",
    "full": "firmid ind2_year province_year"
}

# Planned specifications
PLANNED_SPECS = {
    "method_code": METHOD_CODE,
    "method_tree_path": METHOD_TREE_PATH,
    "specs_to_run": [
        {"spec_id": "baseline", "spec_tree_path": "methods/difference_in_differences.md#baseline"},
        {"spec_id": "did/fe/twoway", "spec_tree_path": "methods/difference_in_differences.md#fixed-effects"},
        {"spec_id": "did/fe/industry_x_time", "spec_tree_path": "methods/difference_in_differences.md#fixed-effects"},
        {"spec_id": "did/fe/region_x_time", "spec_tree_path": "methods/difference_in_differences.md#fixed-effects"},
        {"spec_id": "did/controls/none", "spec_tree_path": "methods/difference_in_differences.md#control-sets"},
        {"spec_id": "did/controls/full", "spec_tree_path": "methods/difference_in_differences.md#control-sets"},
    ],
    "robustness_specs": [
        {"spec_id": "robust/loo/drop_soe", "spec_tree_path": "robustness/leave_one_out.md"},
        {"spec_id": "robust/loo/drop_roa", "spec_tree_path": "robustness/leave_one_out.md"},
        {"spec_id": "robust/loo/drop_age", "spec_tree_path": "robustness/leave_one_out.md"},
        {"spec_id": "robust/loo/drop_exp", "spec_tree_path": "robustness/leave_one_out.md"},
        {"spec_id": "robust/cluster/year", "spec_tree_path": "robustness/clustering_variations.md"},
        {"spec_id": "robust/cluster/industry", "spec_tree_path": "robustness/clustering_variations.md"},
        {"spec_id": "robust/single/soe", "spec_tree_path": "robustness/single_covariate.md"},
        {"spec_id": "robust/single/roa", "spec_tree_path": "robustness/single_covariate.md"},
    ],
    "sample_restriction_specs": [
        {"spec_id": "did/sample/rank_lt_8000", "description": "Top 10,000 rank < 8,000"},
        {"spec_id": "did/sample/rank_lt_5000", "description": "Top 10,000 rank < 5,000"},
        {"spec_id": "did/sample/rank_lt_3000", "description": "Top 10,000 rank < 3,000"},
        {"spec_id": "did/sample/rank_gt_300", "description": "Top 10,000 rank > 300"},
        {"spec_id": "did/sample/rank_gt_500", "description": "Top 10,000 rank > 500"},
        {"spec_id": "did/sample/rank_gt_800", "description": "Top 10,000 rank > 800"},
    ]
}


def main():
    """
    Main analysis - CANNOT RUN DUE TO MISSING DATA

    This function documents what the analysis would look like if data were available.
    """

    print("=" * 60)
    print(f"Specification Search: {PAPER_ID}")
    print("=" * 60)
    print()
    print("STATUS: DATA NOT AVAILABLE")
    print()
    print("The following data files are required but not provided:")
    required_files = [
        "Data/Table1.dta",
        "Data/Table2_1.dta",
        "Data/Table2_2.dta",
        "Data/Table3_1.dta",
        "Data/Table3_2.dta",
        "Data/TableA7_1.dta",
        "Data/TableA7_2.dta",
    ]
    for f in required_files:
        print(f"  - {f}")

    print()
    print("The following specifications WOULD have been run:")
    print()

    total_specs = (len(PLANNED_SPECS["specs_to_run"]) +
                   len(PLANNED_SPECS["robustness_specs"]) +
                   len(PLANNED_SPECS["sample_restriction_specs"]))

    # Account for 3 outcomes
    total_specs = total_specs * len(OUTCOMES)

    print(f"Total planned specifications: {total_specs}")
    print(f"  - Main specs x {len(OUTCOMES)} outcomes: {len(PLANNED_SPECS['specs_to_run']) * len(OUTCOMES)}")
    print(f"  - Robustness specs x {len(OUTCOMES)} outcomes: {len(PLANNED_SPECS['robustness_specs']) * len(OUTCOMES)}")
    print(f"  - Sample restrictions x {len(OUTCOMES)} outcomes: {len(PLANNED_SPECS['sample_restriction_specs']) * len(OUTCOMES)}")

    print()
    print("Method Map:")
    print(json.dumps(PLANNED_SPECS, indent=2))

    print()
    print("=" * 60)
    print("REPLICATION NOT POSSIBLE - PROPRIETARY DATA REQUIRED")
    print("=" * 60)


# Document the regression specifications from Table 1
TABLE1_SPECS = """
Table 1: Effects of the Program on Regulated Firms

Panel A: Energy Use (lenergy)
Panel B: Output (loutput)
Panel C: Energy Efficiency (lefficiency)

All panels use:
- Column 1: firm FE + year FE, cluster(firmid)
- Column 2: firm FE + industry*year FE, cluster(firmid)
- Column 3: firm FE + industry*year FE + province*year FE, cluster(firmid)
- Column 4: Same as Col 3 + controls (soe roa age exp)

Stata command:
reghdfe {outcome} treat_post {controls}, absorb({fe}) cluster(firmid)
"""


# Document the regression specifications from Table 2
TABLE2_SPECS = """
Table 2: Spillover Effects on Related Firms

Uses distance matching to identify related (treated) vs control firms.
Then runs same DiD specification as Table 1.

Treatment: related_post (related firm indicator x post-2007)
Matching: Pre-treatment output trajectories (Mahalanobis distance)
Weights: Frequency weights based on matching counts
"""


# Document the regression specifications from Table 3
TABLE3_SPECS = """
Table 3: Industry-level Spillovers and Within-Conglomerate Effects

Panel A: Industry-Level Spillovers
- Controls for industry-level output and energy use
- Treatment: spillover_post

Panel B: Within-Conglomerate Difference-in-Differences
- Conglomerate x year FE absorbs all conglomerate-level variation
- Compares Top 1,000 firms to other firms in same conglomerate
- Treatment: treat_post + top1000 indicator
"""


if __name__ == "__main__":
    main()
