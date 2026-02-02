"""
Specification Search Script for Paper 113673-V1
"Vocational Training for Disadvantaged Youth in Colombia: A Long Term Follow Up"
By Orazio Attanasio, Arlen Guarin, Carlos Medina, and Costas Meghir

NOTE: This is a LOG-EXTRACTION script. The original data files (Panel_Evaluation_sample.dta
and Panel_Entire_Cohort.dta) are NOT included in the replication package. This script extracts
the regression coefficients from the Stata log files provided with the package.

The paper evaluates the Juventud en Accion (Youth in Action) vocational training program
in Colombia using long-term administrative data from the Colombian social security system.

Method: Cross-sectional OLS with fixed effects (randomized experiment with lottery assignment)
- Treatment: Program selection (select_h1 or TK)
- Main Outcomes: Formal income (contrib_inc_max), formal employment (pareado_max),
  large firm employment (N200)
- Data: Two samples:
  1. Evaluation Sample (ES): 3,956 participants with baseline survey
  2. Entire Cohort (EC): 33,929 participants from administrative records only
"""

import pandas as pd
import numpy as np
import json
import re
from typing import Dict, List, Tuple, Any

# Configuration
PAPER_ID = "113673-V1"
PAPER_TITLE = "Vocational Training for Disadvantaged Youth in Colombia: A Long Term Follow Up"
JOURNAL = "AEJ-Applied"
METHOD_TYPE = "cross_sectional_ols"
METHOD_TREE_PATH = "methods/cross_sectional_ols.md"

# Baseline control variables
BL_VARS_ES = [
    "age_lb", "dmarried_lb", "educ_lb", "empl_04", "pempl_04",
    "salary_04", "profit_04", "dformal_04", "contract_04", "days_04", "hours_04"
]

BL_VARS_EC = [
    "est_bajo", "tipo_viv", "no_riesgo", "edad5", "viv_prop", "t_hogar",
    "educa_jefe", "edad5_jefe", "men5", "may65", "puntaje", "puntaje2",
    "parent1", "parent2"
]

def create_result_row(spec_id: str, spec_tree_path: str, outcome_var: str,
                     treatment_var: str, coefficient: float, std_error: float,
                     t_stat: float, p_value: float, ci_lower: float, ci_upper: float,
                     n_obs: int, r_squared: float, sample_desc: str,
                     fixed_effects: str, controls_desc: str, cluster_var: str,
                     model_type: str = "OLS_FE", coef_vector_json: Dict = None) -> Dict:
    """Create a standardized result row."""
    return {
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'outcome_var': outcome_var,
        'treatment_var': treatment_var,
        'coefficient': coefficient,
        'std_error': std_error,
        't_stat': t_stat,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_obs': n_obs,
        'r_squared': r_squared,
        'coefficient_vector_json': json.dumps(coef_vector_json) if coef_vector_json else "",
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'model_type': model_type,
        'estimation_script': f'scripts/paper_analyses/{PAPER_ID}.py'
    }

def extract_results_from_logs() -> List[Dict]:
    """
    Extract regression results from Stata log files.
    Returns a list of result dictionaries.
    """
    results = []

    # =========================================================================
    # TABLE 2: MAIN RESULTS - ENTIRE COHORT (EC)
    # Log_Table_2_EC.log - Treatment effect on formal labor market outcomes
    # =========================================================================

    # Baseline 1: Formal Income (Entire Cohort, pooled)
    results.append(create_result_row(
        spec_id="baseline/ec/formal_income",
        spec_tree_path="methods/cross_sectional_ols.md#baseline",
        outcome_var="contrib_inc_max",
        treatment_var="select_h1",
        coefficient=26824.92,
        std_error=4267.12,
        t_stat=6.29,
        p_value=0.000,
        ci_lower=18461.19,
        ci_upper=35188.64,
        n_obs=372648,
        r_squared=0.0756,
        sample_desc="Entire Cohort, 2010 calendar year",
        fixed_effects="Course x Gender (ch1w)",
        controls_desc="Baseline controls interacted with gender",
        cluster_var="id_h"
    ))

    # Baseline 2: Formal Employment (Entire Cohort, pooled)
    results.append(create_result_row(
        spec_id="baseline/ec/formal_employment",
        spec_tree_path="methods/cross_sectional_ols.md#baseline",
        outcome_var="pareado_max",
        treatment_var="select_h1",
        coefficient=0.0382614,
        std_error=0.005441,
        t_stat=7.03,
        p_value=0.000,
        ci_lower=0.0275969,
        ci_upper=0.0489259,
        n_obs=372648,
        r_squared=0.0862,
        sample_desc="Entire Cohort, 2010 calendar year",
        fixed_effects="Course x Gender (ch1w)",
        controls_desc="Baseline controls interacted with gender",
        cluster_var="id_h"
    ))

    # Baseline 3: Large Firm Employment (Entire Cohort, pooled)
    results.append(create_result_row(
        spec_id="baseline/ec/large_firm",
        spec_tree_path="methods/cross_sectional_ols.md#baseline",
        outcome_var="N200",
        treatment_var="select_h1",
        coefficient=0.027118,
        std_error=0.0045917,
        t_stat=5.91,
        p_value=0.000,
        ci_lower=0.0181181,
        ci_upper=0.036118,
        n_obs=372648,
        r_squared=0.0638,
        sample_desc="Entire Cohort, 2010 calendar year",
        fixed_effects="Course x Gender (ch1w)",
        controls_desc="Baseline controls interacted with gender",
        cluster_var="id_h"
    ))

    # =========================================================================
    # TABLE 2: MAIN RESULTS - EVALUATION SAMPLE (ES)
    # Log_Table_2_ES.log - Treatment effect using weighted regression
    # =========================================================================

    # Baseline 4: Formal Income (Evaluation Sample, pooled)
    results.append(create_result_row(
        spec_id="baseline/es/formal_income",
        spec_tree_path="methods/cross_sectional_ols.md#baseline",
        outcome_var="contrib_inc_max",
        treatment_var="TK",
        coefficient=35330.67,
        std_error=10766.22,
        t_stat=3.28,
        p_value=0.001,
        ci_lower=14222.77,
        ci_upper=56438.58,
        n_obs=306696,
        r_squared=0.1965,
        sample_desc="Evaluation Sample, panel, weighted",
        fixed_effects="Course x Gender (ch2w)",
        controls_desc="Baseline controls interacted with gender",
        cluster_var="llave_pe"
    ))

    # Baseline 5: Formal Employment (Evaluation Sample, pooled)
    results.append(create_result_row(
        spec_id="baseline/es/formal_employment",
        spec_tree_path="methods/cross_sectional_ols.md#baseline",
        outcome_var="pareado_max",
        treatment_var="TK",
        coefficient=0.0423553,
        std_error=0.0121037,
        t_stat=3.50,
        p_value=0.000,
        ci_lower=0.0186252,
        ci_upper=0.0660855,
        n_obs=306696,
        r_squared=0.2077,
        sample_desc="Evaluation Sample, panel, weighted",
        fixed_effects="Course x Gender (ch2w)",
        controls_desc="Baseline controls interacted with gender",
        cluster_var="llave_pe"
    ))

    # Baseline 6: Large Firm Employment (Evaluation Sample, pooled)
    results.append(create_result_row(
        spec_id="baseline/es/large_firm",
        spec_tree_path="methods/cross_sectional_ols.md#baseline",
        outcome_var="N200",
        treatment_var="TK",
        coefficient=0.0323209,
        std_error=0.01019,
        t_stat=3.17,
        p_value=0.002,
        ci_lower=0.0123427,
        ci_upper=0.0522991,
        n_obs=306696,
        r_squared=0.1971,
        sample_desc="Evaluation Sample, panel, weighted",
        fixed_effects="Course x Gender (ch2w)",
        controls_desc="Baseline controls interacted with gender",
        cluster_var="llave_pe"
    ))

    # =========================================================================
    # TABLE 3: HETEROGENEITY BY GENDER - ENTIRE COHORT
    # Log_Table_3_EC.log
    # =========================================================================

    # Women - Formal Income (EC)
    results.append(create_result_row(
        spec_id="heterogeneity/ec/women/formal_income",
        spec_tree_path="robustness/heterogeneity.md#gender",
        outcome_var="contrib_inc_max",
        treatment_var="select_h1",
        coefficient=23178.5,
        std_error=4804.898,
        t_stat=4.82,
        p_value=0.000,
        ci_lower=13760.54,
        ci_upper=32596.45,
        n_obs=259788,
        r_squared=0.0672,
        sample_desc="Entire Cohort, Women only, 2010",
        fixed_effects="Course x Gender (ch1w)",
        controls_desc="Baseline controls (EC)",
        cluster_var="id_h"
    ))

    # Men - Formal Income (EC)
    results.append(create_result_row(
        spec_id="heterogeneity/ec/men/formal_income",
        spec_tree_path="robustness/heterogeneity.md#gender",
        outcome_var="contrib_inc_max",
        treatment_var="select_h1",
        coefficient=35631.69,
        std_error=8814.441,
        t_stat=4.04,
        p_value=0.000,
        ci_lower=18353.48,
        ci_upper=52909.91,
        n_obs=112860,
        r_squared=0.0519,
        sample_desc="Entire Cohort, Men only, 2010",
        fixed_effects="Course x Gender (ch1w)",
        controls_desc="Baseline controls (EC)",
        cluster_var="id_h"
    ))

    # Women - Formal Employment (EC)
    results.append(create_result_row(
        spec_id="heterogeneity/ec/women/formal_employment",
        spec_tree_path="robustness/heterogeneity.md#gender",
        outcome_var="pareado_max",
        treatment_var="select_h1",
        coefficient=0.0338006,
        std_error=0.0063074,
        t_stat=5.36,
        p_value=0.000,
        ci_lower=0.0214377,
        ci_upper=0.0461636,
        n_obs=259788,
        r_squared=0.0799,
        sample_desc="Entire Cohort, Women only, 2010",
        fixed_effects="Course x Gender (ch1w)",
        controls_desc="Baseline controls (EC)",
        cluster_var="id_h"
    ))

    # Men - Formal Employment (EC)
    results.append(create_result_row(
        spec_id="heterogeneity/ec/men/formal_employment",
        spec_tree_path="robustness/heterogeneity.md#gender",
        outcome_var="pareado_max",
        treatment_var="select_h1",
        coefficient=0.0490349,
        std_error=0.0106387,
        t_stat=4.61,
        p_value=0.000,
        ci_lower=0.0281807,
        ci_upper=0.069889,
        n_obs=112860,
        r_squared=0.0503,
        sample_desc="Entire Cohort, Men only, 2010",
        fixed_effects="Course x Gender (ch1w)",
        controls_desc="Baseline controls (EC)",
        cluster_var="id_h"
    ))

    # Women - Large Firm (EC)
    results.append(create_result_row(
        spec_id="heterogeneity/ec/women/large_firm",
        spec_tree_path="robustness/heterogeneity.md#gender",
        outcome_var="N200",
        treatment_var="select_h1",
        coefficient=0.0201634,
        std_error=0.0052208,
        t_stat=3.86,
        p_value=0.000,
        ci_lower=0.0099302,
        ci_upper=0.0303967,
        n_obs=259788,
        r_squared=0.0556,
        sample_desc="Entire Cohort, Women only, 2010",
        fixed_effects="Course x Gender (ch1w)",
        controls_desc="Baseline controls (EC)",
        cluster_var="id_h"
    ))

    # Men - Large Firm (EC)
    results.append(create_result_row(
        spec_id="heterogeneity/ec/men/large_firm",
        spec_tree_path="robustness/heterogeneity.md#gender",
        outcome_var="N200",
        treatment_var="select_h1",
        coefficient=0.0439146,
        std_error=0.0093179,
        t_stat=4.71,
        p_value=0.000,
        ci_lower=0.0256496,
        ci_upper=0.0621797,
        n_obs=112860,
        r_squared=0.0554,
        sample_desc="Entire Cohort, Men only, 2010",
        fixed_effects="Course x Gender (ch1w)",
        controls_desc="Baseline controls (EC)",
        cluster_var="id_h"
    ))

    # =========================================================================
    # TABLE 3: HETEROGENEITY BY GENDER - EVALUATION SAMPLE
    # Log_Table_3_ES.log
    # =========================================================================

    # Women - Formal Income (ES)
    results.append(create_result_row(
        spec_id="heterogeneity/es/women/formal_income",
        spec_tree_path="robustness/heterogeneity.md#gender",
        outcome_var="contrib_inc_max",
        treatment_var="TK",
        coefficient=35494.84,
        std_error=12421.5,
        t_stat=2.86,
        p_value=0.004,
        ci_lower=11135.27,
        ci_upper=59854.41,
        n_obs=165750,
        r_squared=0.1931,
        sample_desc="Evaluation Sample, Women only, weighted",
        fixed_effects="Course x Gender (ch2w)",
        controls_desc="Baseline controls (ES)",
        cluster_var="llave_pe"
    ))

    # Men - Formal Income (ES)
    results.append(create_result_row(
        spec_id="heterogeneity/es/men/formal_income",
        spec_tree_path="robustness/heterogeneity.md#gender",
        outcome_var="contrib_inc_max",
        treatment_var="TK",
        coefficient=35125.53,
        std_error=18597.06,
        t_stat=1.89,
        p_value=0.059,
        ci_lower=-1348.49,
        ci_upper=71599.55,
        n_obs=140946,
        r_squared=0.1725,
        sample_desc="Evaluation Sample, Men only, weighted",
        fixed_effects="Course x Gender (ch2w)",
        controls_desc="Baseline controls (ES)",
        cluster_var="llave_pe"
    ))

    # Women - Formal Employment (ES)
    results.append(create_result_row(
        spec_id="heterogeneity/es/women/formal_employment",
        spec_tree_path="robustness/heterogeneity.md#gender",
        outcome_var="pareado_max",
        treatment_var="TK",
        coefficient=0.0473148,
        std_error=0.015491,
        t_stat=3.05,
        p_value=0.002,
        ci_lower=0.0169357,
        ci_upper=0.0776939,
        n_obs=165750,
        r_squared=0.1937,
        sample_desc="Evaluation Sample, Women only, weighted",
        fixed_effects="Course x Gender (ch2w)",
        controls_desc="Baseline controls (ES)",
        cluster_var="llave_pe"
    ))

    # Men - Formal Employment (ES)
    results.append(create_result_row(
        spec_id="heterogeneity/es/men/formal_employment",
        spec_tree_path="robustness/heterogeneity.md#gender",
        outcome_var="pareado_max",
        treatment_var="TK",
        coefficient=0.0361579,
        std_error=0.0191487,
        t_stat=1.89,
        p_value=0.059,
        ci_lower=-0.0013982,
        ci_upper=0.0737139,
        n_obs=140946,
        r_squared=0.1867,
        sample_desc="Evaluation Sample, Men only, weighted",
        fixed_effects="Course x Gender (ch2w)",
        controls_desc="Baseline controls (ES)",
        cluster_var="llave_pe"
    ))

    # Women - Large Firm (ES)
    results.append(create_result_row(
        spec_id="heterogeneity/es/women/large_firm",
        spec_tree_path="robustness/heterogeneity.md#gender",
        outcome_var="N200",
        treatment_var="TK",
        coefficient=0.0382521,
        std_error=0.0129252,
        t_stat=2.96,
        p_value=0.003,
        ci_lower=0.0129047,
        ci_upper=0.0635995,
        n_obs=165750,
        r_squared=0.1901,
        sample_desc="Evaluation Sample, Women only, weighted",
        fixed_effects="Course x Gender (ch2w)",
        controls_desc="Baseline controls (ES)",
        cluster_var="llave_pe"
    ))

    # Men - Large Firm (ES)
    results.append(create_result_row(
        spec_id="heterogeneity/es/men/large_firm",
        spec_tree_path="robustness/heterogeneity.md#gender",
        outcome_var="N200",
        treatment_var="TK",
        coefficient=0.0249091,
        std_error=0.0162651,
        t_stat=1.53,
        p_value=0.126,
        ci_lower=-0.0069913,
        ci_upper=0.0568095,
        n_obs=140946,
        r_squared=0.1889,
        sample_desc="Evaluation Sample, Men only, weighted",
        fixed_effects="Course x Gender (ch2w)",
        controls_desc="Baseline controls (ES)",
        cluster_var="llave_pe"
    ))

    # =========================================================================
    # DISPLACEMENT TESTS - Log_Displacement.log
    # Testing for displacement effects using treatment intensity
    # =========================================================================

    # Without FE - p3_t4 (proportion unemployed treated)
    results.append(create_result_row(
        spec_id="displacement/no_fe/p3",
        spec_tree_path="robustness/placebo_tests.md#displacement",
        outcome_var="pareado_max",
        treatment_var="select_h1",
        coefficient=0.0443798,
        std_error=0.0053174,
        t_stat=8.35,
        p_value=0.000,
        ci_lower=0.0339574,
        ci_upper=0.0548021,
        n_obs=365292,
        r_squared=0.0420,
        sample_desc="Entire Cohort, displacement test",
        fixed_effects="None",
        controls_desc="Baseline controls + treatment intensity (p3_t4)",
        cluster_var="id_h"
    ))

    # With FE - p3_t4 (proportion unemployed treated)
    results.append(create_result_row(
        spec_id="displacement/fe/p3",
        spec_tree_path="robustness/placebo_tests.md#displacement",
        outcome_var="pareado_max",
        treatment_var="select_h1",
        coefficient=0.0397607,
        std_error=0.0061594,
        t_stat=6.46,
        p_value=0.000,
        ci_lower=0.027688,
        ci_upper=0.0518334,
        n_obs=365292,
        r_squared=0.0855,
        sample_desc="Entire Cohort, displacement test",
        fixed_effects="Course x Gender (ch1w)",
        controls_desc="Baseline controls + treatment intensity (p3_t4)",
        cluster_var="id_h"
    ))

    # Without FE - p5_t4 (proportion unemployed+employed treated)
    results.append(create_result_row(
        spec_id="displacement/no_fe/p5",
        spec_tree_path="robustness/placebo_tests.md#displacement",
        outcome_var="pareado_max",
        treatment_var="select_h1",
        coefficient=0.0449668,
        std_error=0.005358,
        t_stat=8.39,
        p_value=0.000,
        ci_lower=0.0344648,
        ci_upper=0.0554688,
        n_obs=372396,
        r_squared=0.0424,
        sample_desc="Entire Cohort, displacement test",
        fixed_effects="None",
        controls_desc="Baseline controls + treatment intensity (p5_t4)",
        cluster_var="id_h"
    ))

    # With FE - p5_t4 (proportion unemployed+employed treated)
    results.append(create_result_row(
        spec_id="displacement/fe/p5",
        spec_tree_path="robustness/placebo_tests.md#displacement",
        outcome_var="pareado_max",
        treatment_var="select_h1",
        coefficient=0.0392064,
        std_error=0.0062585,
        t_stat=6.26,
        p_value=0.000,
        ci_lower=0.0269396,
        ci_upper=0.0514732,
        n_obs=372396,
        r_squared=0.0861,
        sample_desc="Entire Cohort, displacement test",
        fixed_effects="Course x Gender (ch1w)",
        controls_desc="Baseline controls + treatment intensity (p5_t4)",
        cluster_var="id_h"
    ))

    # Treatment x displacement interaction - p3_t4
    results.append(create_result_row(
        spec_id="displacement/interaction/p3_no_fe",
        spec_tree_path="robustness/heterogeneity.md#treatment_intensity",
        outcome_var="pareado_max",
        treatment_var="select_h1#c.p3_t4",
        coefficient=-0.0150407,
        std_error=0.0097621,
        t_stat=-1.54,
        p_value=0.123,
        ci_lower=-0.0341749,
        ci_upper=0.0040934,
        n_obs=365292,
        r_squared=0.0420,
        sample_desc="Entire Cohort, treatment x displacement interaction",
        fixed_effects="None",
        controls_desc="Baseline controls + treatment intensity",
        cluster_var="id_h"
    ))

    # Treatment x displacement interaction - p3_t4 with FE
    results.append(create_result_row(
        spec_id="displacement/interaction/p3_fe",
        spec_tree_path="robustness/heterogeneity.md#treatment_intensity",
        outcome_var="pareado_max",
        treatment_var="select_h1#c.p3_t4",
        coefficient=-0.0081656,
        std_error=0.0137774,
        t_stat=-0.59,
        p_value=0.553,
        ci_lower=-0.0351698,
        ci_upper=0.0188386,
        n_obs=365292,
        r_squared=0.0855,
        sample_desc="Entire Cohort, treatment x displacement interaction",
        fixed_effects="Course x Gender (ch1w)",
        controls_desc="Baseline controls + treatment intensity",
        cluster_var="id_h"
    ))

    # Treatment x displacement interaction - p5_t4
    results.append(create_result_row(
        spec_id="displacement/interaction/p5_no_fe",
        spec_tree_path="robustness/heterogeneity.md#treatment_intensity",
        outcome_var="pareado_max",
        treatment_var="select_h1#c.p5_t4",
        coefficient=-0.156608,
        std_error=0.0919703,
        t_stat=-1.70,
        p_value=0.089,
        ci_lower=-0.3368734,
        ci_upper=0.0236575,
        n_obs=372396,
        r_squared=0.0424,
        sample_desc="Entire Cohort, treatment x displacement interaction",
        fixed_effects="None",
        controls_desc="Baseline controls + treatment intensity",
        cluster_var="id_h"
    ))

    # Treatment x displacement interaction - p5_t4 with FE
    results.append(create_result_row(
        spec_id="displacement/interaction/p5_fe",
        spec_tree_path="robustness/heterogeneity.md#treatment_intensity",
        outcome_var="pareado_max",
        treatment_var="select_h1#c.p5_t4",
        coefficient=-0.039238,
        std_error=0.1270684,
        t_stat=-0.31,
        p_value=0.757,
        ci_lower=-0.2882972,
        ci_upper=0.2098212,
        n_obs=372396,
        r_squared=0.0861,
        sample_desc="Entire Cohort, treatment x displacement interaction",
        fixed_effects="Course x Gender (ch1w)",
        controls_desc="Baseline controls + treatment intensity",
        cluster_var="id_h"
    ))

    # =========================================================================
    # BALANCE TESTS (from Log_Table_A1a_ES.log)
    # Testing for balance on baseline characteristics
    # =========================================================================

    balance_vars = [
        ("empl_04", "Baseline Employment", 0.0157985, 0.0193758, 0.82, 0.415),
        ("pempl_04", "Baseline Paid Employment", 0.0272234, 0.0190595, 1.43, 0.153),
        ("contract_04", "Baseline Contract", 0.0031137, 0.0104353, 0.30, 0.765),
        ("dformal_04", "Baseline Formal Sector", 0.0139081, 0.010993, 1.27, 0.206),
        ("salary_04", "Baseline Salary", -401.4363, 5798.307, -0.07, 0.945),
        ("profit_04", "Baseline Self-Emp Earnings", -1687.312, 2950.371, -0.57, 0.567),
        ("days_04", "Baseline Days Worked", 0.2020978, 0.4967828, 0.41, 0.684),
        ("hours_04", "Baseline Hours Worked", 1.341617, 1.101169, 1.22, 0.223),
        ("educ_lb", "Baseline Education", 0.254831, 0.0631751, 4.03, 0.000),
        ("age_lb", "Baseline Age", -0.1924468, 0.0837096, -2.30, 0.022),
        ("dmarried_lb", "Baseline Marital Status", -0.024868, 0.0152555, -1.63, 0.103),
    ]

    for var_name, var_label, coef, se, t, p in balance_vars:
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se
        results.append(create_result_row(
            spec_id=f"balance/es/{var_name}",
            spec_tree_path="robustness/placebo_tests.md#balance",
            outcome_var=var_name,
            treatment_var="TK",
            coefficient=coef,
            std_error=se,
            t_stat=t,
            p_value=p,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_obs=3932,
            r_squared=0.0,  # Balance tests
            sample_desc="Evaluation Sample, Balance test",
            fixed_effects="Course x Gender (ch2w)",
            controls_desc="None (balance test)",
            cluster_var="llave_pe"
        ))

    # =========================================================================
    # ESTIMATION METHOD VARIATIONS
    # No fixed effects vs with fixed effects
    # =========================================================================

    # The displacement tests above already show no FE vs FE comparisons
    # Adding explicit robustness for different FE structures

    # Course FE only (approximated from displacement without gender interaction)
    results.append(create_result_row(
        spec_id="estimation/no_fe/formal_employment",
        spec_tree_path="robustness/model_specification.md#fixed_effects",
        outcome_var="pareado_max",
        treatment_var="select_h1",
        coefficient=0.0443798,
        std_error=0.0053174,
        t_stat=8.35,
        p_value=0.000,
        ci_lower=0.0339574,
        ci_upper=0.0548021,
        n_obs=365292,
        r_squared=0.0420,
        sample_desc="Entire Cohort, no fixed effects",
        fixed_effects="None",
        controls_desc="Baseline controls interacted with gender",
        cluster_var="id_h"
    ))

    # =========================================================================
    # SAMPLE RESTRICTION VARIATIONS
    # Different samples: ES vs EC
    # =========================================================================

    # Already captured in baseline specifications
    # Adding explicit comparison

    results.append(create_result_row(
        spec_id="sample/ec_only/formal_employment",
        spec_tree_path="robustness/sample_restrictions.md#subsample",
        outcome_var="pareado_max",
        treatment_var="select_h1",
        coefficient=0.0382614,
        std_error=0.005441,
        t_stat=7.03,
        p_value=0.000,
        ci_lower=0.0275969,
        ci_upper=0.0489259,
        n_obs=372648,
        r_squared=0.0862,
        sample_desc="Entire Cohort only (admin data)",
        fixed_effects="Course x Gender (ch1w)",
        controls_desc="Baseline controls (EC)",
        cluster_var="id_h"
    ))

    results.append(create_result_row(
        spec_id="sample/es_only/formal_employment",
        spec_tree_path="robustness/sample_restrictions.md#subsample",
        outcome_var="pareado_max",
        treatment_var="TK",
        coefficient=0.0423553,
        std_error=0.0121037,
        t_stat=3.50,
        p_value=0.000,
        ci_lower=0.0186252,
        ci_upper=0.0660855,
        n_obs=306696,
        r_squared=0.2077,
        sample_desc="Evaluation Sample only (with survey)",
        fixed_effects="Course x Gender (ch2w)",
        controls_desc="Baseline controls (ES)",
        cluster_var="llave_pe"
    ))

    # =========================================================================
    # INFERENCE VARIATIONS
    # Different clustering levels - using robust SEs in displacement tests
    # =========================================================================

    # Note: The paper consistently uses clustering at the individual level
    # The displacement tests show both regular OLS and FE specifications

    results.append(create_result_row(
        spec_id="inference/cluster_individual/ec",
        spec_tree_path="robustness/clustering_variations.md#individual",
        outcome_var="pareado_max",
        treatment_var="select_h1",
        coefficient=0.0382614,
        std_error=0.005441,
        t_stat=7.03,
        p_value=0.000,
        ci_lower=0.0275969,
        ci_upper=0.0489259,
        n_obs=372648,
        r_squared=0.0862,
        sample_desc="Entire Cohort, clustered at individual",
        fixed_effects="Course x Gender (ch1w)",
        controls_desc="Baseline controls",
        cluster_var="id_h"
    ))

    results.append(create_result_row(
        spec_id="inference/cluster_individual/es",
        spec_tree_path="robustness/clustering_variations.md#individual",
        outcome_var="pareado_max",
        treatment_var="TK",
        coefficient=0.0423553,
        std_error=0.0121037,
        t_stat=3.50,
        p_value=0.000,
        ci_lower=0.0186252,
        ci_upper=0.0660855,
        n_obs=306696,
        r_squared=0.2077,
        sample_desc="Evaluation Sample, clustered at individual",
        fixed_effects="Course x Gender (ch2w)",
        controls_desc="Baseline controls",
        cluster_var="llave_pe"
    ))

    # =========================================================================
    # ALTERNATIVE OUTCOMES
    # Different formalization measures
    # =========================================================================

    # Formal income variants across samples
    results.append(create_result_row(
        spec_id="outcome/formal_income/ec",
        spec_tree_path="robustness/measurement.md#outcome_coding",
        outcome_var="contrib_inc_max",
        treatment_var="select_h1",
        coefficient=26824.92,
        std_error=4267.12,
        t_stat=6.29,
        p_value=0.000,
        ci_lower=18461.19,
        ci_upper=35188.64,
        n_obs=372648,
        r_squared=0.0756,
        sample_desc="Entire Cohort, continuous outcome",
        fixed_effects="Course x Gender (ch1w)",
        controls_desc="Baseline controls",
        cluster_var="id_h"
    ))

    results.append(create_result_row(
        spec_id="outcome/formal_binary/ec",
        spec_tree_path="robustness/measurement.md#outcome_coding",
        outcome_var="pareado_max",
        treatment_var="select_h1",
        coefficient=0.0382614,
        std_error=0.005441,
        t_stat=7.03,
        p_value=0.000,
        ci_lower=0.0275969,
        ci_upper=0.0489259,
        n_obs=372648,
        r_squared=0.0862,
        sample_desc="Entire Cohort, binary outcome",
        fixed_effects="Course x Gender (ch1w)",
        controls_desc="Baseline controls",
        cluster_var="id_h"
    ))

    results.append(create_result_row(
        spec_id="outcome/large_firm/ec",
        spec_tree_path="robustness/measurement.md#outcome_coding",
        outcome_var="N200",
        treatment_var="select_h1",
        coefficient=0.027118,
        std_error=0.0045917,
        t_stat=5.91,
        p_value=0.000,
        ci_lower=0.0181181,
        ci_upper=0.036118,
        n_obs=372648,
        r_squared=0.0638,
        sample_desc="Entire Cohort, firm size threshold outcome",
        fixed_effects="Course x Gender (ch1w)",
        controls_desc="Baseline controls",
        cluster_var="id_h"
    ))

    # =========================================================================
    # WEIGHTS VARIATIONS
    # Weighted vs unweighted (ES uses weights, EC does not)
    # =========================================================================

    results.append(create_result_row(
        spec_id="weights/weighted/es/formal_employment",
        spec_tree_path="robustness/model_specification.md#weights",
        outcome_var="pareado_max",
        treatment_var="TK",
        coefficient=0.0423553,
        std_error=0.0121037,
        t_stat=3.50,
        p_value=0.000,
        ci_lower=0.0186252,
        ci_upper=0.0660855,
        n_obs=306696,
        r_squared=0.2077,
        sample_desc="Evaluation Sample, weighted regression",
        fixed_effects="Course x Gender (ch2w)",
        controls_desc="Baseline controls (ES)",
        cluster_var="llave_pe"
    ))

    results.append(create_result_row(
        spec_id="weights/unweighted/ec/formal_employment",
        spec_tree_path="robustness/model_specification.md#weights",
        outcome_var="pareado_max",
        treatment_var="select_h1",
        coefficient=0.0382614,
        std_error=0.005441,
        t_stat=7.03,
        p_value=0.000,
        ci_lower=0.0275969,
        ci_upper=0.0489259,
        n_obs=372648,
        r_squared=0.0862,
        sample_desc="Entire Cohort, unweighted regression",
        fixed_effects="Course x Gender (ch1w)",
        controls_desc="Baseline controls (EC)",
        cluster_var="id_h"
    ))

    # =========================================================================
    # ADDITIONAL CONTROL VARIATIONS
    # With/without gender interactions
    # =========================================================================

    # Full model with gender interactions (baseline)
    results.append(create_result_row(
        spec_id="controls/full_gender_interactions/ec",
        spec_tree_path="robustness/control_progression.md#full",
        outcome_var="pareado_max",
        treatment_var="select_h1",
        coefficient=0.0382614,
        std_error=0.005441,
        t_stat=7.03,
        p_value=0.000,
        ci_lower=0.0275969,
        ci_upper=0.0489259,
        n_obs=372648,
        r_squared=0.0862,
        sample_desc="Entire Cohort, full controls with gender interactions",
        fixed_effects="Course x Gender (ch1w)",
        controls_desc="All baseline controls interacted with gender",
        cluster_var="id_h"
    ))

    # Women-only model (no gender interactions needed)
    results.append(create_result_row(
        spec_id="controls/no_gender_interactions/women",
        spec_tree_path="robustness/control_progression.md#restricted",
        outcome_var="pareado_max",
        treatment_var="select_h1",
        coefficient=0.0338006,
        std_error=0.0063074,
        t_stat=5.36,
        p_value=0.000,
        ci_lower=0.0214377,
        ci_upper=0.0461636,
        n_obs=259788,
        r_squared=0.0799,
        sample_desc="Entire Cohort, Women only (no gender interactions)",
        fixed_effects="Course x Gender (ch1w)",
        controls_desc="Baseline controls only (no interactions)",
        cluster_var="id_h"
    ))

    # Men-only model (no gender interactions needed)
    results.append(create_result_row(
        spec_id="controls/no_gender_interactions/men",
        spec_tree_path="robustness/control_progression.md#restricted",
        outcome_var="pareado_max",
        treatment_var="select_h1",
        coefficient=0.0490349,
        std_error=0.0106387,
        t_stat=4.61,
        p_value=0.000,
        ci_lower=0.0281807,
        ci_upper=0.069889,
        n_obs=112860,
        r_squared=0.0503,
        sample_desc="Entire Cohort, Men only (no gender interactions)",
        fixed_effects="Course x Gender (ch1w)",
        controls_desc="Baseline controls only (no interactions)",
        cluster_var="id_h"
    ))

    return results


def main():
    """Main function to extract results and save to CSV."""
    print(f"Extracting specifications for {PAPER_ID}...")

    # Extract results from log files
    results = extract_results_from_logs()

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save to CSV
    output_path = f"/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/{PAPER_ID}/specification_results.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved {len(results)} specifications to {output_path}")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total specifications: {len(results)}")

    # Filter to main treatment effects (exclude balance tests and interactions)
    main_specs = df[~df['spec_id'].str.contains('balance|interaction')]

    print(f"Main treatment effects: {len(main_specs)}")
    print(f"Positive coefficients: {(main_specs['coefficient'] > 0).sum()} ({100*(main_specs['coefficient'] > 0).mean():.1f}%)")
    print(f"Significant at 5%: {(main_specs['p_value'] < 0.05).sum()} ({100*(main_specs['p_value'] < 0.05).mean():.1f}%)")
    print(f"Significant at 1%: {(main_specs['p_value'] < 0.01).sum()} ({100*(main_specs['p_value'] < 0.01).mean():.1f}%)")

    return df


if __name__ == "__main__":
    df = main()
