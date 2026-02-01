"""
Specification Search Script: 113577-V1
Paper: Teacher Peer Effects on Student Achievement (AEJ: Applied)

Authors: [From NCERDC Data]
Topic: Effect of peer teachers on student test score growth in North Carolina

DATA AVAILABILITY: RESTRICTED
================================================================================
This script documents the specifications that WOULD be run if data were available.
The main analysis dataset requires access to the North Carolina Education Research
Data Center (NCERDC) at Duke University, which requires:
1. Signed Data Use Agreement
2. IRB approval
3. Research proposal
4. Compensation to Data Center

Contact: http://www.childandfamilypolicy.duke.edu/ep/nceddatacenter/
================================================================================
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================
PAPER_ID = "113577-V1"
PAPER_TITLE = "Teacher Peer Effects on Student Achievement"
JOURNAL = "AEJ: Applied"

BASE_DIR = Path("/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search")
DATA_DIR = BASE_DIR / "data" / "downloads" / "extracted" / PAPER_ID / "Code_NC_AEJ"
OUTPUT_DIR = BASE_DIR / "data" / "downloads" / "extracted" / PAPER_ID

# =============================================================================
# Check Data Availability
# =============================================================================

def check_data_availability():
    """Check if required data files exist."""

    required_files = [
        "Final_file_JAN09.dta",  # Main analysis file
        "basic_Jan09.dta",       # Intermediate file
        "FX_Jan09.dta",          # Teacher fixed effects
        "reg_file_Jan09.dta",    # Regression file
    ]

    available_files = list(DATA_DIR.glob("*.dta"))
    print(f"Available .dta files: {[f.name for f in available_files]}")

    missing = []
    for f in required_files:
        if not (DATA_DIR / f).exists() and not (DATA_DIR / "temp" / f).exists():
            missing.append(f)

    if missing:
        print(f"\nMISSING REQUIRED FILES:")
        for f in missing:
            print(f"  - {f}")
        return False
    return True


def document_specifications():
    """
    Document the specifications that would be run if data were available.
    Based on analysis of the Stata do files.
    """

    specs = []

    # ==========================================================================
    # From 3_peer_value_added_results.do - Main Results
    # ==========================================================================

    # Math outcomes
    math_specs = [
        {
            "spec_id": "baseline/math/pooled",
            "spec_tree_path": "methods/panel_fixed_effects.md#baseline",
            "outcome_var": "m_growth",
            "treatment_var": "peer_tfx_m",
            "fixed_effects": "None (pooled OLS)",
            "controls_desc": "year*grade, sex, ethnic, pared, r_same, s_same, clsize, exp dummies, teacher FE, peer FE missing dummy",
            "cluster_var": "t_s (teacher-school)",
            "model_type": "OLS",
            "notes": "Pooled OLS regression"
        },
        {
            "spec_id": "baseline/math/school_year_fe",
            "spec_tree_path": "methods/panel_fixed_effects.md#fixed-effects",
            "outcome_var": "m_growth",
            "treatment_var": "peer_tfx_m",
            "fixed_effects": "school-year (s_s)",
            "controls_desc": "lagged math, year*grade, sex, ethnic, pared, r_same, s_same, clsize, exp dummies, teacher FE, peer FE missing",
            "cluster_var": "t_s",
            "model_type": "FE",
            "notes": "School-year absorbing regression (areg)"
        },
        {
            "spec_id": "baseline/math/student_fe",
            "spec_tree_path": "methods/panel_fixed_effects.md#fixed-effects",
            "outcome_var": "m_growth",
            "treatment_var": "peer_tfx_m",
            "fixed_effects": "student (mastid)",
            "controls_desc": "year*grade, r_same, s_same, clsize, exp dummies, teacher FE, peer FE missing",
            "cluster_var": "t_s",
            "model_type": "FE",
            "notes": "Student FE - controls for time-invariant student characteristics"
        },
        {
            "spec_id": "baseline/math/teacher_school_fe",
            "spec_tree_path": "methods/panel_fixed_effects.md#fixed-effects",
            "outcome_var": "m_growth",
            "treatment_var": "peer_tfx_m",
            "fixed_effects": "teacher-school (t_s)",
            "controls_desc": "lagged math, year*grade, sex, ethnic, pared, r_same, s_same, clsize, exp dummies, peer FE missing",
            "cluster_var": "t_s",
            "model_type": "FE",
            "notes": "Teacher-school FE - controls for teacher-school specific effects"
        },
        {
            "spec_id": "baseline/math/twoway_fe",
            "spec_tree_path": "methods/panel_fixed_effects.md#fixed-effects",
            "outcome_var": "m_growth",
            "treatment_var": "peer_tfx_m",
            "fixed_effects": "teacher-school + school-year (FELSDVREG)",
            "controls_desc": "lagged math, year*grade, sex, ethnic, pared, r_same, s_same, clsize, exp dummies, peer FE missing",
            "cluster_var": "t_s",
            "model_type": "Two-way FE",
            "notes": "Main specification using FELSDVREG for high-dimensional FE"
        },
    ]

    # Reading outcomes (parallel to math)
    reading_specs = [
        {
            "spec_id": "baseline/reading/pooled",
            "spec_tree_path": "methods/panel_fixed_effects.md#baseline",
            "outcome_var": "r_growth",
            "treatment_var": "peer_tfx_r",
            "fixed_effects": "None (pooled OLS)",
            "controls_desc": "year*grade, sex, ethnic, pared, r_same, s_same, clsize, exp dummies, teacher FE, peer FE missing",
            "cluster_var": "t_s",
            "model_type": "OLS",
            "notes": "Pooled OLS - reading"
        },
        {
            "spec_id": "baseline/reading/school_year_fe",
            "spec_tree_path": "methods/panel_fixed_effects.md#fixed-effects",
            "outcome_var": "r_growth",
            "treatment_var": "peer_tfx_r",
            "fixed_effects": "school-year (s_s)",
            "controls_desc": "lagged read, year*grade, sex, ethnic, pared, r_same, s_same, clsize, exp dummies, teacher FE, peer FE missing",
            "cluster_var": "t_s",
            "model_type": "FE",
            "notes": "School-year FE - reading"
        },
        {
            "spec_id": "baseline/reading/student_fe",
            "spec_tree_path": "methods/panel_fixed_effects.md#fixed-effects",
            "outcome_var": "r_growth",
            "treatment_var": "peer_tfx_r",
            "fixed_effects": "student (mastid)",
            "controls_desc": "year*grade, r_same, s_same, clsize, exp dummies, teacher FE, peer FE missing",
            "cluster_var": "t_s",
            "model_type": "FE",
            "notes": "Student FE - reading"
        },
        {
            "spec_id": "baseline/reading/teacher_school_fe",
            "spec_tree_path": "methods/panel_fixed_effects.md#fixed-effects",
            "outcome_var": "r_growth",
            "treatment_var": "peer_tfx_r",
            "fixed_effects": "teacher-school (t_s)",
            "controls_desc": "lagged read, year*grade, sex, ethnic, pared, r_same, s_same, clsize, exp dummies, peer FE missing",
            "cluster_var": "t_s",
            "model_type": "FE",
            "notes": "Teacher-school FE - reading"
        },
        {
            "spec_id": "baseline/reading/twoway_fe",
            "spec_tree_path": "methods/panel_fixed_effects.md#fixed-effects",
            "outcome_var": "r_growth",
            "treatment_var": "peer_tfx_r",
            "fixed_effects": "teacher-school + school-year (FELSDVREG)",
            "controls_desc": "lagged read, year*grade, sex, ethnic, pared, r_same, s_same, clsize, exp dummies, peer FE missing",
            "cluster_var": "t_s",
            "model_type": "Two-way FE",
            "notes": "Main specification - reading"
        },
    ]

    # ==========================================================================
    # From 2_Part_1.do - Teacher Characteristics Results
    # ==========================================================================

    teacher_char_specs = [
        {
            "spec_id": "panel/fe/teacher_char/math",
            "spec_tree_path": "methods/panel_fixed_effects.md#fixed-effects",
            "outcome_var": "m_growth",
            "treatment_var": "peer_lic_score, peer_adv_deg, peer_reg_lic, peer_cert",
            "fixed_effects": "school-year (s_s)",
            "controls_desc": "lagged math, year*grade, sex, ethnic, pared, r_same, s_same, clsize, own teacher chars, experience",
            "cluster_var": "t_s",
            "model_type": "FE",
            "notes": "Peer observable characteristics instead of peer FE"
        },
        {
            "spec_id": "panel/fe/teacher_char/reading",
            "spec_tree_path": "methods/panel_fixed_effects.md#fixed-effects",
            "outcome_var": "r_growth",
            "treatment_var": "peer_lic_score, peer_adv_deg, peer_reg_lic, peer_cert",
            "fixed_effects": "school-year (s_s)",
            "controls_desc": "lagged read, year*grade, sex, ethnic, pared, r_same, s_same, clsize, own teacher chars, experience",
            "cluster_var": "t_s",
            "model_type": "FE",
            "notes": "Peer observable characteristics - reading"
        },
    ]

    # ==========================================================================
    # From 4_Future_teachers_and_peers.do - Placebo Tests
    # ==========================================================================

    placebo_specs = [
        {
            "spec_id": "panel/placebo/future_teacher/math",
            "spec_tree_path": "methods/panel_fixed_effects.md#placebo",
            "outcome_var": "m_growth",
            "treatment_var": "fut_tfx_m (future teacher quality)",
            "fixed_effects": "school-year",
            "controls_desc": "lagged math, sex, pared, ethnic, r_same, s_same, clsize, exp, peer_tfx_m, teach_fx_m",
            "cluster_var": "teachid",
            "model_type": "FE",
            "notes": "Future teacher as placebo - should be zero if no selection"
        },
        {
            "spec_id": "panel/placebo/future_teacher/reading",
            "spec_tree_path": "methods/panel_fixed_effects.md#placebo",
            "outcome_var": "r_growth",
            "treatment_var": "fut_tfx_r",
            "fixed_effects": "school-year",
            "controls_desc": "lagged read, sex, pared, ethnic, r_same, s_same, clsize, exp, peer_tfx_r, teach_fx_r",
            "cluster_var": "teachid",
            "model_type": "FE",
            "notes": "Future teacher placebo - reading"
        },
        {
            "spec_id": "panel/placebo/future_peers/math",
            "spec_tree_path": "methods/panel_fixed_effects.md#placebo",
            "outcome_var": "m_growth",
            "treatment_var": "fut_m_peers",
            "fixed_effects": "teacher-school + school-year (FELSDVREG)",
            "controls_desc": "lagged math, sex, pared, ethnic, r_same, s_same, clsize, exp, peer_tfx_m, lag_m_peers, lag2_m_peers",
            "cluster_var": "t_s",
            "model_type": "Two-way FE",
            "notes": "Future peer quality as placebo"
        },
        {
            "spec_id": "panel/placebo/future_peers/reading",
            "spec_tree_path": "methods/panel_fixed_effects.md#placebo",
            "outcome_var": "r_growth",
            "treatment_var": "fut_r_peers",
            "fixed_effects": "teacher-school + school-year (FELSDVREG)",
            "controls_desc": "lagged read, sex, pared, ethnic, r_same, s_same, clsize, exp, peer_tfx_r, lag_r_peers, lag2_r_peers",
            "cluster_var": "t_s",
            "model_type": "Two-way FE",
            "notes": "Future peer quality placebo - reading"
        },
    ]

    # ==========================================================================
    # Robustness Specifications
    # ==========================================================================

    # Student FE robustness (from 3_peer_value_added_results.do)
    student_fe_specs = [
        {
            "spec_id": "panel/robustness/student_fe/math",
            "spec_tree_path": "methods/panel_fixed_effects.md#robustness",
            "outcome_var": "m_growth",
            "treatment_var": "peer_tfx_m",
            "fixed_effects": "student + teacher-school (FELSDVREG)",
            "controls_desc": "year*grade, sex, ethnic, pared, r_same, s_same, clsize, exp dummies, peer FE missing",
            "cluster_var": "t_s",
            "model_type": "Two-way FE",
            "notes": "Student FE instead of school-year FE"
        },
        {
            "spec_id": "panel/robustness/student_fe/reading",
            "spec_tree_path": "methods/panel_fixed_effects.md#robustness",
            "outcome_var": "r_growth",
            "treatment_var": "peer_tfx_r",
            "fixed_effects": "student + teacher-school (FELSDVREG)",
            "controls_desc": "year*grade, sex, ethnic, pared, r_same, s_same, clsize, exp dummies, peer FE missing",
            "cluster_var": "t_s",
            "model_type": "Two-way FE",
            "notes": "Student FE robustness - reading"
        },
    ]

    # Grade-switching interaction (from 4_Future_teachers_and_peers.do)
    grade_switch_specs = [
        {
            "spec_id": "panel/heterogeneity/grade_switch/math",
            "spec_tree_path": "methods/panel_fixed_effects.md#heterogeneity",
            "outcome_var": "m_growth",
            "treatment_var": "t_m_grade * peer_tfx_m, t_m_grade * lag_m_peers",
            "fixed_effects": "teacher-school + school-year",
            "controls_desc": "lagged math, sex, pared, ethnic, r_same, s_same, clsize, exp, peer FE missing, grade-year",
            "cluster_var": "t_s",
            "model_type": "Two-way FE",
            "notes": "Heterogeneity by whether teacher switched grades"
        },
    ]

    # Combine all specs
    all_specs = (math_specs + reading_specs + teacher_char_specs +
                 placebo_specs + student_fe_specs + grade_switch_specs)

    # Add common fields
    for spec in all_specs:
        spec["paper_id"] = PAPER_ID
        spec["journal"] = JOURNAL
        spec["paper_title"] = PAPER_TITLE
        spec["coefficient"] = np.nan
        spec["std_error"] = np.nan
        spec["t_stat"] = np.nan
        spec["p_value"] = np.nan
        spec["ci_lower"] = np.nan
        spec["ci_upper"] = np.nan
        spec["n_obs"] = np.nan
        spec["r_squared"] = np.nan
        spec["coefficient_vector_json"] = json.dumps({"status": "data_unavailable"})
        spec["sample_desc"] = "NC students grades 3-5, years 2001-2006"
        spec["estimation_script"] = f"scripts/paper_analyses/{PAPER_ID}.py"

    return all_specs


def main():
    """Main execution function."""

    print(f"=" * 80)
    print(f"Specification Search: {PAPER_ID}")
    print(f"=" * 80)

    # Check data availability
    print("\nChecking data availability...")
    data_available = check_data_availability()

    if not data_available:
        print("\n" + "=" * 80)
        print("DATA NOT AVAILABLE - Cannot run specification search")
        print("=" * 80)
        print("\nThe main analysis dataset requires restricted data access from NCERDC.")
        print("See: http://www.childandfamilypolicy.duke.edu/ep/nceddatacenter/")

        # Document what WOULD be run
        print("\nDocumenting specifications that would be run...")
        specs = document_specifications()

        # Save documentation
        df = pd.DataFrame(specs)

        # Reorder columns
        cols = [
            "paper_id", "journal", "paper_title", "spec_id", "spec_tree_path",
            "outcome_var", "treatment_var", "coefficient", "std_error", "t_stat",
            "p_value", "ci_lower", "ci_upper", "n_obs", "r_squared",
            "coefficient_vector_json", "sample_desc", "fixed_effects",
            "controls_desc", "cluster_var", "model_type", "estimation_script", "notes"
        ]
        df = df[cols]

        output_path = OUTPUT_DIR / "specification_results.csv"
        df.to_csv(output_path, index=False)
        print(f"\nDocumented {len(specs)} specifications to: {output_path}")

        return

    # If data IS available (which it won't be), run the actual analysis
    print("\nData available - would run specifications here...")


if __name__ == "__main__":
    main()
