"""
Replication script for 112870-V2
"Optimal Life Cycle Unemployment Insurance" by Michelacci & Ruffo (AER 2015)

STATUS: NOT POSSIBLE
Reason: PSID data (data.dta) required by PSID_consumption_regressions.do
is not included in the replication package. The README states it was moved
to https://doi.org/10.3886/E228166V1.

Per replication protocol, when any regressions require unavailable data,
the entire paper is classified as "not possible."

The package contains 38 regression commands across 4 Stata do-files:
  - elasticity_SIPP.do: 28 Cox PH regressions (Tables 1, 2; Figure 1)
  - sevpay_M.do: 3 Cox PH regressions (Table 3)
  - unemploymentCPS.do: 3 OLS/IV regressions (Figure 1b, Figure A2)
  - PSID_consumption_regressions.do: 4 panel regressions (Figures 2, 3) -- DATA MISSING

Additionally, the quantitative model is implemented in Matlab (.m files)
using structural calibration/simulation methods, not regressions.
"""

import os
import csv

PAPER_ID = "112870-V2"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PACKAGE_DIR = os.path.join(REPO_ROOT, "data", "downloads", "extracted", PAPER_ID)

def main():
    print(f"Replication for {PAPER_ID}: NOT POSSIBLE")
    print("=" * 60)
    print()
    print("Reason: PSID data (data.dta) is not included in the package.")
    print("It was moved to a separate deposit: https://doi.org/10.3886/E228166V1")
    print()
    print("Total regressions in original package: 38")
    print("  - elasticity_SIPP.do: 28 (stcox)")
    print("  - sevpay_M.do: 3 (stcox with tvc)")
    print("  - unemploymentCPS.do: 3 (reg, ivreg)")
    print("  - PSID_consumption_regressions.do: 4 (xtreg) -- DATA MISSING")
    print()
    print("Writing empty replication.csv (header only)...")

    output_path = os.path.join(PACKAGE_DIR, "replication.csv")
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "paper_id", "reg_id", "outcome_var", "treatment_var",
            "coefficient", "std_error", "p_value", "ci_lower", "ci_upper",
            "n_obs", "r_squared", "original_coefficient", "original_std_error",
            "match_status", "coefficient_vector_json", "fixed_effects",
            "controls_desc", "cluster_var", "estimator", "sample_desc", "notes"
        ])

    print(f"Written to {output_path}")

if __name__ == "__main__":
    main()
