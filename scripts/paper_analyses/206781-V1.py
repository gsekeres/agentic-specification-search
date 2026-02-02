#!/usr/bin/env python3
"""
Paper Analysis: 206781-V1
====================================

Paper: "Who Should Get Social Insurance?" - Cash Transfer Targeting Study
Authors: Walker et al.
Journal: AER

This paper uses an RCT design to study the effects of cash transfers on
household outcomes (consumption, assets, income) in Kenya. The study compares
different targeting rules: most deprived (D), socially optimal (SO), and
most impacted (I).

Method: Cross-sectional OLS on RCT data
Treatment: Randomized cash transfer (treat)
Outcomes: Consumption, assets, income (time-demeaned versions)
Controls: PMT covariates (household characteristics)
Clustering: Village level

Usage:
    python scripts/paper_analyses/206781-V1.py
"""

import os
import json
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

warnings.filterwarnings('ignore')

# Try importing pyfixest, fall back to statsmodels
try:
    import pyfixest as pf
    USE_PYFIXEST = True
except ImportError:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    USE_PYFIXEST = False

# =============================================================================
# CONFIGURATION
# =============================================================================

PAPER_ID = "206781-V1"
JOURNAL = "AER"
PAPER_TITLE = "Who Should Get Social Insurance? A Machine Learning Approach to Cash Transfer Targeting"
METHOD_TYPE = "cross_sectional_ols"

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
PACKAGE_DIR = BASE_DIR / "data" / "downloads" / "extracted" / PAPER_ID
OUTPUT_FILE = PACKAGE_DIR / "specification_results.csv"

# Variables
# Primary outcomes (time-demeaned versions as in the paper)
OUTCOME_VARS = ['el_cons_T_hh', 'el_assets_T_hh', 'el_income_T_hh']
# Alternative outcomes
ALT_OUTCOME_VARS = ['el_consumption_hh', 'el_assets_hh', 'el_income_hh', 'el_food_cons_T_hh']

TREATMENT_VAR = "treat"

# PMT covariates from the paper
CONTROL_VARS = [
    'bl_hhsize', 'bl_widow', 'bl_female_fr', 'bl_hh_has_children',
    'bl_children_school', 'bl_child_3', 'bl_child_6', 'bl_has_elder',
    'bl_any_livestock', 'bl_own_land', 'bl_own_quarter_acre',
    'bl_own_TV_radio', 'bl_meals_eaten', 'bl_meals_protein',
    'bl_selfemp', 'bl_emp'
]

# Demographic controls
DEMO_CONTROLS = ['bl_hhsize', 'bl_widow', 'bl_female_fr', 'bl_hh_has_children']
# Economic controls
ECON_CONTROLS = ['bl_any_livestock', 'bl_own_land', 'bl_own_quarter_acre', 'bl_own_TV_radio']
# Employment controls
EMP_CONTROLS = ['bl_selfemp', 'bl_emp']
# Food security controls
FOOD_CONTROLS = ['bl_meals_eaten', 'bl_meals_protein']

CLUSTER_VAR = "village_code"
WEIGHT_VAR = "hhweight_EL"

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load and prepare the data for analysis."""
    data_file = PACKAGE_DIR / "data" / "PMT_targeting_data.dta"

    df = pd.read_stata(data_file)

    # Extract numeric merge code
    df['merge_c_num'] = pd.to_numeric(
        df['merge_c'].astype(str).str.extract(r'(\d+)', expand=False),
        errors='coerce'
    )

    # Filter as in original R code:
    # 1. Eligible households only
    # 2. Matched baseline-endline (merge_c == 3)
    # 3. Targeting sample (clean data)
    df = df[df['eligible'] == 1].copy()
    df = df[df['merge_c_num'] == 3].copy()
    df = df[df['targeting_sample'] == 1].copy()

    # Drop rows with too many missing X values (matching R code logic)
    missing_counts = df[CONTROL_VARS].isna().sum(axis=1)
    df = df[missing_counts <= 7].copy()

    # Drop observations with missing values in key outcomes and treatment
    key_vars = ['el_consumption_hh', 'el_assets_hh', 'el_income_hh', 'treat']
    df = df.dropna(subset=key_vars).copy()

    # Create additional variables for specifications
    # Log transformations (add small constant to avoid log(0))
    for outcome in ['el_consumption_hh', 'el_assets_hh', 'el_income_hh']:
        df[f'log_{outcome}'] = np.log(df[outcome] + 100)
        df[f'ihs_{outcome}'] = np.arcsinh(df[outcome])

    # Create binary indicators for heterogeneity analysis
    df['large_hh'] = (df['bl_hhsize'] > df['bl_hhsize'].median()).astype(int)
    df['has_children'] = df['bl_hh_has_children'].fillna(0).astype(int)
    df['owns_land'] = df['bl_own_land'].fillna(0).astype(int)
    df['widow'] = df['bl_widow'].fillna(0).astype(int)
    df['female_resp'] = df['bl_female_fr'].fillna(0).astype(int)
    df['employed'] = ((df['bl_selfemp'].fillna(0) == 1) | (df['bl_emp'].fillna(0) == 1)).astype(int)

    # Per capita versions
    df['hh_size_safe'] = df['bl_hhsize'].replace(0, 1)
    df['el_cons_T_pc'] = df['el_cons_T_hh'] / df['hh_size_safe']
    df['el_assets_T_pc'] = df['el_assets_T_hh'] / df['hh_size_safe']
    df['el_income_T_pc'] = df['el_income_T_hh'] / df['hh_size_safe']

    # Sublocation for alternative clustering
    df['sublocation'] = df['s1_q2b_sublocation']

    return df


# =============================================================================
# SPECIFICATION RUNNER
# =============================================================================

def run_specification(df, outcome, treatment, controls, fe, cluster,
                      spec_id, spec_tree_path, sample_desc="Full sample",
                      weights=None, interaction_var=None):
    """
    Run a single specification and return results dictionary.
    """
    # Handle missing data
    all_vars = [outcome, treatment] + controls
    if cluster:
        all_vars.append(cluster)
    if weights:
        all_vars.append(weights)
    if interaction_var and interaction_var not in all_vars:
        all_vars.append(interaction_var)
    if fe:
        all_vars.extend(fe)

    df_clean = df[all_vars].dropna()

    if len(df_clean) < 50:
        print(f"  Skipping {spec_id}: insufficient observations ({len(df_clean)})")
        return None

    # Build formula
    control_str = " + ".join(controls) if controls else ""
    fe_str = " + ".join(fe) if fe else ""

    # Handle interaction
    if interaction_var:
        formula_treat = f"{treatment} * {interaction_var}"
    else:
        formula_treat = treatment

    if USE_PYFIXEST:
        formula = f"{outcome} ~ {formula_treat}"
        if control_str:
            formula += f" + {control_str}"
        if fe_str:
            formula += f" | {fe_str}"

        try:
            if weights and weights in df_clean.columns:
                model = pf.feols(formula, data=df_clean,
                               vcov={'CRV1': cluster} if cluster else 'hetero',
                               weights=weights)
            else:
                model = pf.feols(formula, data=df_clean,
                               vcov={'CRV1': cluster} if cluster else 'hetero')

            # Get treatment coefficient
            coef = model.coef()[treatment]
            se = model.se()[treatment]
            pval = model.pvalue()[treatment]
            n_obs = model._N
            r2 = model._r2
            tstat = coef / se
            ci_lower = coef - 1.96 * se
            ci_upper = coef + 1.96 * se

            # Build coefficient vector
            coef_vector = {
                "treatment": {
                    "var": treatment,
                    "coef": float(coef),
                    "se": float(se),
                    "pval": float(pval)
                },
                "controls": [],
                "fixed_effects": fe if fe else [],
                "diagnostics": {
                    "n_clusters": int(df_clean[cluster].nunique()) if cluster else None
                }
            }

            # Add interaction coefficient if applicable
            if interaction_var:
                interact_term = f"{treatment}:{interaction_var}"
                if interact_term in model.coef().index:
                    coef_vector["interaction"] = {
                        "var": interact_term,
                        "coef": float(model.coef()[interact_term]),
                        "se": float(model.se()[interact_term]),
                        "pval": float(model.pvalue()[interact_term])
                    }

            for ctrl in controls:
                if ctrl in model.coef().index:
                    coef_vector["controls"].append({
                        "var": ctrl,
                        "coef": float(model.coef()[ctrl]),
                        "se": float(model.se()[ctrl]),
                        "pval": float(model.pvalue()[ctrl])
                    })

        except Exception as e:
            print(f"  Error running {spec_id}: {e}")
            return None

    else:
        # Fallback to statsmodels
        formula = f"{outcome} ~ {formula_treat}"
        if control_str:
            formula += f" + {control_str}"
        if fe_str:
            for f in fe:
                formula += f" + C({f})"

        try:
            if weights and weights in df_clean.columns:
                model = smf.wls(formula, data=df_clean, weights=df_clean[weights]).fit(
                    cov_type='cluster',
                    cov_kwds={'groups': df_clean[cluster]} if cluster else None
                )
            else:
                model = smf.ols(formula, data=df_clean).fit(
                    cov_type='cluster',
                    cov_kwds={'groups': df_clean[cluster]} if cluster else None
                )

            coef = model.params[treatment]
            se = model.bse[treatment]
            pval = model.pvalues[treatment]
            n_obs = int(model.nobs)
            r2 = model.rsquared
            tstat = model.tvalues[treatment]
            ci_lower, ci_upper = model.conf_int().loc[treatment]

            coef_vector = {
                "treatment": {"var": treatment, "coef": float(coef),
                              "se": float(se), "pval": float(pval)},
                "controls": [],
                "fixed_effects": fe if fe else [],
                "diagnostics": {}
            }

        except Exception as e:
            print(f"  Error running {spec_id}: {e}")
            return None

    return {
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'outcome_var': outcome,
        'treatment_var': treatment,
        'coefficient': float(coef),
        'std_error': float(se),
        't_stat': float(tstat),
        'p_value': float(pval),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'n_obs': int(n_obs),
        'r_squared': float(r2),
        'coefficient_vector_json': json.dumps(coef_vector),
        'sample_desc': sample_desc,
        'fixed_effects': ", ".join(fe) if fe else "None",
        'controls_desc': ", ".join(controls) if controls else "None",
        'cluster_var': cluster if cluster else "None",
        'model_type': "WLS" if weights else "OLS",
        'estimation_script': f"scripts/paper_analyses/{PAPER_ID}.py"
    }


# =============================================================================
# SPECIFICATION SEARCH
# =============================================================================

def run_specification_search(df):
    """Run all specifications for this paper. Target: 50+ specifications."""
    results = []

    # Primary outcome for most tests
    PRIMARY_OUTCOME = 'el_cons_T_hh'

    # =========================================================================
    # 1. BASELINE SPECIFICATIONS (3 specs - one per outcome)
    # =========================================================================
    print("\n[1] Running baseline specifications...")
    for outcome in OUTCOME_VARS:
        result = run_specification(
            df, outcome, TREATMENT_VAR, CONTROL_VARS, [], CLUSTER_VAR,
            spec_id=f"baseline/{outcome}",
            spec_tree_path=f"methods/{METHOD_TYPE}.md",
            weights=WEIGHT_VAR
        )
        if result:
            results.append(result)

    # =========================================================================
    # 2. CONTROL VARIATIONS (16+ specs)
    # =========================================================================
    print("\n[2] Running control variations...")

    # 2a. No controls (bivariate)
    result = run_specification(
        df, PRIMARY_OUTCOME, TREATMENT_VAR, [], [], CLUSTER_VAR,
        spec_id="ols/controls/none",
        spec_tree_path="robustness/control_progression.md",
        weights=WEIGHT_VAR
    )
    if result:
        results.append(result)

    # 2b. Control sets
    control_sets = [
        ("ols/controls/demographics", DEMO_CONTROLS),
        ("ols/controls/economic", ECON_CONTROLS),
        ("ols/controls/employment", EMP_CONTROLS),
        ("ols/controls/food", FOOD_CONTROLS),
        ("ols/controls/demo_econ", DEMO_CONTROLS + ECON_CONTROLS),
        ("ols/controls/full", CONTROL_VARS),
    ]

    for spec_id, controls in control_sets:
        result = run_specification(
            df, PRIMARY_OUTCOME, TREATMENT_VAR, controls, [], CLUSTER_VAR,
            spec_id=spec_id,
            spec_tree_path="robustness/control_progression.md",
            weights=WEIGHT_VAR
        )
        if result:
            results.append(result)

    # 2c. Leave-one-out (drop each control)
    print("  Running leave-one-out...")
    for control in CONTROL_VARS:
        remaining = [c for c in CONTROL_VARS if c != control]
        result = run_specification(
            df, PRIMARY_OUTCOME, TREATMENT_VAR, remaining, [], CLUSTER_VAR,
            spec_id=f"robust/loo/drop_{control}",
            spec_tree_path="robustness/leave_one_out.md",
            weights=WEIGHT_VAR
        )
        if result:
            results.append(result)

    # =========================================================================
    # 3. INFERENCE/CLUSTERING VARIATIONS (6+ specs)
    # =========================================================================
    print("\n[3] Running inference variations...")

    # 3a. Robust SE (no clustering)
    result = run_specification(
        df, PRIMARY_OUTCOME, TREATMENT_VAR, CONTROL_VARS, [], None,
        spec_id="robust/cluster/none",
        spec_tree_path="robustness/clustering_variations.md",
        weights=WEIGHT_VAR
    )
    if result:
        results.append(result)

    # 3b. Different clustering levels
    cluster_vars = ['village_code', 'sublocation']
    for cluster in cluster_vars:
        if cluster in df.columns and df[cluster].notna().sum() > 0:
            result = run_specification(
                df, PRIMARY_OUTCOME, TREATMENT_VAR, CONTROL_VARS, [], cluster,
                spec_id=f"robust/cluster/{cluster}",
                spec_tree_path="robustness/clustering_variations.md",
                weights=WEIGHT_VAR
            )
            if result:
                results.append(result)

    # 3c. Unweighted
    result = run_specification(
        df, PRIMARY_OUTCOME, TREATMENT_VAR, CONTROL_VARS, [], CLUSTER_VAR,
        spec_id="robust/weights/unweighted",
        spec_tree_path="robustness/clustering_variations.md",
        weights=None
    )
    if result:
        results.append(result)

    # =========================================================================
    # 4. SAMPLE RESTRICTIONS (10+ specs)
    # =========================================================================
    print("\n[4] Running sample restrictions...")

    # 4a. Drop outliers (winsorize/trim)
    for pct in [1, 5, 10]:
        df_trim = df.copy()
        lower = df_trim[PRIMARY_OUTCOME].quantile(pct/100)
        upper = df_trim[PRIMARY_OUTCOME].quantile(1 - pct/100)
        df_trim = df_trim[(df_trim[PRIMARY_OUTCOME] >= lower) & (df_trim[PRIMARY_OUTCOME] <= upper)]

        result = run_specification(
            df_trim, PRIMARY_OUTCOME, TREATMENT_VAR, CONTROL_VARS, [], CLUSTER_VAR,
            spec_id=f"robust/sample/trim_{pct}pct",
            spec_tree_path="robustness/sample_restrictions.md",
            sample_desc=f"Trimmed {pct}% tails",
            weights=WEIGHT_VAR
        )
        if result:
            results.append(result)

    # 4b. By household size
    df_small = df[df['bl_hhsize'] <= df['bl_hhsize'].median()]
    df_large = df[df['bl_hhsize'] > df['bl_hhsize'].median()]

    result = run_specification(
        df_small, PRIMARY_OUTCOME, TREATMENT_VAR, CONTROL_VARS, [], CLUSTER_VAR,
        spec_id="robust/sample/small_hh",
        spec_tree_path="robustness/sample_restrictions.md",
        sample_desc="Small households (below median)",
        weights=WEIGHT_VAR
    )
    if result:
        results.append(result)

    result = run_specification(
        df_large, PRIMARY_OUTCOME, TREATMENT_VAR, CONTROL_VARS, [], CLUSTER_VAR,
        spec_id="robust/sample/large_hh",
        spec_tree_path="robustness/sample_restrictions.md",
        sample_desc="Large households (above median)",
        weights=WEIGHT_VAR
    )
    if result:
        results.append(result)

    # 4c. By land ownership
    df_land = df[df['owns_land'] == 1]
    df_noland = df[df['owns_land'] == 0]

    result = run_specification(
        df_land, PRIMARY_OUTCOME, TREATMENT_VAR, CONTROL_VARS, [], CLUSTER_VAR,
        spec_id="robust/sample/owns_land",
        spec_tree_path="robustness/sample_restrictions.md",
        sample_desc="Households that own land",
        weights=WEIGHT_VAR
    )
    if result:
        results.append(result)

    result = run_specification(
        df_noland, PRIMARY_OUTCOME, TREATMENT_VAR, CONTROL_VARS, [], CLUSTER_VAR,
        spec_id="robust/sample/no_land",
        spec_tree_path="robustness/sample_restrictions.md",
        sample_desc="Households without land",
        weights=WEIGHT_VAR
    )
    if result:
        results.append(result)

    # 4d. By widow status
    df_widow = df[df['widow'] == 1]
    df_notwidow = df[df['widow'] == 0]

    result = run_specification(
        df_widow, PRIMARY_OUTCOME, TREATMENT_VAR, CONTROL_VARS, [], CLUSTER_VAR,
        spec_id="robust/sample/widow",
        spec_tree_path="robustness/sample_restrictions.md",
        sample_desc="Widow-headed households",
        weights=WEIGHT_VAR
    )
    if result:
        results.append(result)

    result = run_specification(
        df_notwidow, PRIMARY_OUTCOME, TREATMENT_VAR, CONTROL_VARS, [], CLUSTER_VAR,
        spec_id="robust/sample/not_widow",
        spec_tree_path="robustness/sample_restrictions.md",
        sample_desc="Non-widow households",
        weights=WEIGHT_VAR
    )
    if result:
        results.append(result)

    # =========================================================================
    # 5. ALTERNATIVE OUTCOMES (8+ specs)
    # =========================================================================
    print("\n[5] Running alternative outcomes...")

    # 5a. Non-demeaned outcomes
    for outcome in ['el_consumption_hh', 'el_assets_hh', 'el_income_hh']:
        result = run_specification(
            df, outcome, TREATMENT_VAR, CONTROL_VARS, [], CLUSTER_VAR,
            spec_id=f"robust/outcome/{outcome}",
            spec_tree_path="robustness/measurement.md",
            weights=WEIGHT_VAR
        )
        if result:
            results.append(result)

    # 5b. Food consumption
    result = run_specification(
        df, 'el_food_cons_T_hh', TREATMENT_VAR, CONTROL_VARS, [], CLUSTER_VAR,
        spec_id="robust/outcome/food_consumption",
        spec_tree_path="robustness/measurement.md",
        weights=WEIGHT_VAR
    )
    if result:
        results.append(result)

    # 5c. Per capita outcomes
    for outcome in ['el_cons_T_pc', 'el_assets_T_pc', 'el_income_T_pc']:
        result = run_specification(
            df, outcome, TREATMENT_VAR, CONTROL_VARS, [], CLUSTER_VAR,
            spec_id=f"robust/outcome/{outcome}",
            spec_tree_path="robustness/measurement.md",
            weights=WEIGHT_VAR
        )
        if result:
            results.append(result)

    # =========================================================================
    # 6. FUNCTIONAL FORM VARIATIONS (5+ specs)
    # =========================================================================
    print("\n[6] Running functional form variations...")

    # 6a. Log outcome (levels, not demeaned)
    result = run_specification(
        df, 'log_el_consumption_hh', TREATMENT_VAR, CONTROL_VARS, [], CLUSTER_VAR,
        spec_id="robust/funcform/log_consumption",
        spec_tree_path="robustness/functional_form.md",
        weights=WEIGHT_VAR
    )
    if result:
        results.append(result)

    # 6b. IHS transformation
    result = run_specification(
        df, 'ihs_el_consumption_hh', TREATMENT_VAR, CONTROL_VARS, [], CLUSTER_VAR,
        spec_id="robust/funcform/ihs_consumption",
        spec_tree_path="robustness/functional_form.md",
        weights=WEIGHT_VAR
    )
    if result:
        results.append(result)

    result = run_specification(
        df, 'log_el_assets_hh', TREATMENT_VAR, CONTROL_VARS, [], CLUSTER_VAR,
        spec_id="robust/funcform/log_assets",
        spec_tree_path="robustness/functional_form.md",
        weights=WEIGHT_VAR
    )
    if result:
        results.append(result)

    result = run_specification(
        df, 'log_el_income_hh', TREATMENT_VAR, CONTROL_VARS, [], CLUSTER_VAR,
        spec_id="robust/funcform/log_income",
        spec_tree_path="robustness/functional_form.md",
        weights=WEIGHT_VAR
    )
    if result:
        results.append(result)

    # =========================================================================
    # 7. HETEROGENEITY ANALYSIS (12+ specs)
    # =========================================================================
    print("\n[7] Running heterogeneity analysis...")

    # 7a. Interaction specifications
    het_vars = [
        ('large_hh', 'Household size'),
        ('has_children', 'Has children'),
        ('owns_land', 'Owns land'),
        ('widow', 'Widow status'),
        ('female_resp', 'Female respondent'),
        ('employed', 'Employed')
    ]

    for het_var, het_desc in het_vars:
        if het_var in df.columns:
            result = run_specification(
                df, PRIMARY_OUTCOME, TREATMENT_VAR, CONTROL_VARS, [], CLUSTER_VAR,
                spec_id=f"robust/het/interaction_{het_var}",
                spec_tree_path="robustness/heterogeneity.md",
                weights=WEIGHT_VAR,
                interaction_var=het_var
            )
            if result:
                results.append(result)

    # 7b. Subgroup analysis (by children)
    df_children = df[df['has_children'] == 1]
    df_nochildren = df[df['has_children'] == 0]

    result = run_specification(
        df_children, PRIMARY_OUTCOME, TREATMENT_VAR, CONTROL_VARS, [], CLUSTER_VAR,
        spec_id="robust/het/has_children",
        spec_tree_path="robustness/heterogeneity.md",
        sample_desc="Households with children",
        weights=WEIGHT_VAR
    )
    if result:
        results.append(result)

    result = run_specification(
        df_nochildren, PRIMARY_OUTCOME, TREATMENT_VAR, CONTROL_VARS, [], CLUSTER_VAR,
        spec_id="robust/het/no_children",
        spec_tree_path="robustness/heterogeneity.md",
        sample_desc="Households without children",
        weights=WEIGHT_VAR
    )
    if result:
        results.append(result)

    # 7c. By employment status
    df_employed = df[df['employed'] == 1]
    df_notemployed = df[df['employed'] == 0]

    result = run_specification(
        df_employed, PRIMARY_OUTCOME, TREATMENT_VAR, CONTROL_VARS, [], CLUSTER_VAR,
        spec_id="robust/het/employed",
        spec_tree_path="robustness/heterogeneity.md",
        sample_desc="Employed households",
        weights=WEIGHT_VAR
    )
    if result:
        results.append(result)

    result = run_specification(
        df_notemployed, PRIMARY_OUTCOME, TREATMENT_VAR, CONTROL_VARS, [], CLUSTER_VAR,
        spec_id="robust/het/not_employed",
        spec_tree_path="robustness/heterogeneity.md",
        sample_desc="Not employed households",
        weights=WEIGHT_VAR
    )
    if result:
        results.append(result)

    # 7d. By female respondent
    df_female = df[df['female_resp'] == 1]
    df_male = df[df['female_resp'] == 0]

    result = run_specification(
        df_female, PRIMARY_OUTCOME, TREATMENT_VAR, CONTROL_VARS, [], CLUSTER_VAR,
        spec_id="robust/het/female_respondent",
        spec_tree_path="robustness/heterogeneity.md",
        sample_desc="Female respondent",
        weights=WEIGHT_VAR
    )
    if result:
        results.append(result)

    result = run_specification(
        df_male, PRIMARY_OUTCOME, TREATMENT_VAR, CONTROL_VARS, [], CLUSTER_VAR,
        spec_id="robust/het/male_respondent",
        spec_tree_path="robustness/heterogeneity.md",
        sample_desc="Male respondent",
        weights=WEIGHT_VAR
    )
    if result:
        results.append(result)

    # =========================================================================
    # 8. PLACEBO TESTS (3+ specs)
    # =========================================================================
    print("\n[8] Running placebo tests...")

    # 8a. Placebo outcome (baseline consumption should not be affected by treatment)
    # Use pre-determined controls only
    minimal_controls = ['bl_hhsize', 'bl_widow']

    # Create a synthetic placebo - randomized fake treatment
    np.random.seed(42)
    df['placebo_treat'] = np.random.binomial(1, 0.5, len(df))

    result = run_specification(
        df, PRIMARY_OUTCOME, 'placebo_treat', CONTROL_VARS, [], CLUSTER_VAR,
        spec_id="robust/placebo/random_treatment",
        spec_tree_path="robustness/placebo_tests.md",
        weights=WEIGHT_VAR
    )
    if result:
        results.append(result)

    # 8b. Balanced covariate test - treatment should not predict baseline characteristics
    for bl_var in ['bl_hhsize', 'bl_meals_eaten']:
        result = run_specification(
            df, bl_var, TREATMENT_VAR, [], [], CLUSTER_VAR,
            spec_id=f"robust/placebo/balance_{bl_var}",
            spec_tree_path="robustness/placebo_tests.md",
            weights=WEIGHT_VAR
        )
        if result:
            results.append(result)

    # =========================================================================
    # 9. ADDITIONAL ROBUSTNESS ACROSS ALL OUTCOMES (9+ specs)
    # =========================================================================
    print("\n[9] Running cross-outcome robustness...")

    # Run key specifications across other outcomes
    for outcome in ['el_assets_T_hh', 'el_income_T_hh']:
        # No controls
        result = run_specification(
            df, outcome, TREATMENT_VAR, [], [], CLUSTER_VAR,
            spec_id=f"robust/outcome/{outcome}_no_controls",
            spec_tree_path="robustness/control_progression.md",
            weights=WEIGHT_VAR
        )
        if result:
            results.append(result)

        # Trimmed sample
        df_trim = df.copy()
        lower = df_trim[outcome].quantile(0.01)
        upper = df_trim[outcome].quantile(0.99)
        df_trim = df_trim[(df_trim[outcome] >= lower) & (df_trim[outcome] <= upper)]

        result = run_specification(
            df_trim, outcome, TREATMENT_VAR, CONTROL_VARS, [], CLUSTER_VAR,
            spec_id=f"robust/sample/{outcome}_trim_1pct",
            spec_tree_path="robustness/sample_restrictions.md",
            sample_desc=f"Trimmed 1% tails for {outcome}",
            weights=WEIGHT_VAR
        )
        if result:
            results.append(result)

        # Unweighted
        result = run_specification(
            df, outcome, TREATMENT_VAR, CONTROL_VARS, [], CLUSTER_VAR,
            spec_id=f"robust/weights/{outcome}_unweighted",
            spec_tree_path="robustness/clustering_variations.md",
            weights=None
        )
        if result:
            results.append(result)

    return results


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def generate_summary_report(results_df, df):
    """Generate SPECIFICATION_SEARCH.md summary report."""

    # Calculate statistics
    n_specs = len(results_df)
    n_positive = (results_df['coefficient'] > 0).sum()
    n_sig_05 = (results_df['p_value'] < 0.05).sum()
    n_sig_01 = (results_df['p_value'] < 0.01).sum()
    median_coef = results_df['coefficient'].median()
    mean_coef = results_df['coefficient'].mean()
    min_coef = results_df['coefficient'].min()
    max_coef = results_df['coefficient'].max()

    # Categorize specifications
    categories = {
        'Baseline': results_df['spec_id'].str.startswith('baseline').sum(),
        'Control variations': results_df['spec_id'].str.contains('controls|loo').sum(),
        'Sample restrictions': results_df['spec_id'].str.contains('sample').sum(),
        'Alternative outcomes': results_df['spec_id'].str.contains('outcome').sum(),
        'Inference variations': results_df['spec_id'].str.contains('cluster|weights').sum(),
        'Functional form': results_df['spec_id'].str.contains('funcform').sum(),
        'Heterogeneity': results_df['spec_id'].str.contains('het').sum(),
        'Placebo tests': results_df['spec_id'].str.contains('placebo').sum(),
    }

    # Robustness assessment
    pct_positive = n_positive / n_specs * 100
    pct_sig = n_sig_05 / n_specs * 100

    if pct_positive > 80 and pct_sig > 70:
        assessment = "STRONG"
        assessment_desc = "Results are highly robust across specifications."
    elif pct_positive > 60 and pct_sig > 50:
        assessment = "MODERATE"
        assessment_desc = "Results are generally robust but some sensitivity exists."
    else:
        assessment = "WEAK"
        assessment_desc = "Results show substantial sensitivity to specification choices."

    report = f"""# Specification Search: {PAPER_TITLE}

## Paper Overview
- **Paper ID**: {PAPER_ID}
- **Journal**: {JOURNAL}
- **Topic**: Cash transfer targeting and household welfare effects
- **Hypothesis**: Cash transfers improve household consumption, assets, and income
- **Method**: Cross-sectional OLS on RCT data
- **Data**: GiveDirectly Kenya experiment (n={len(df)} eligible households)

## Classification
- **Method Type**: cross_sectional_ols (RCT analysis)
- **Spec Tree Path**: methods/cross_sectional_ols.md

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total specifications | {n_specs} |
| Positive coefficients | {n_positive} ({pct_positive:.1f}%) |
| Significant at 5% | {n_sig_05} ({pct_sig:.1f}%) |
| Significant at 1% | {n_sig_01} ({n_sig_01/n_specs*100:.1f}%) |
| Median coefficient | {median_coef:.2f} |
| Mean coefficient | {mean_coef:.2f} |
| Range | [{min_coef:.2f}, {max_coef:.2f}] |

## Robustness Assessment

**{assessment}** support for the main hypothesis.

{assessment_desc}

## Specification Breakdown by Category (i4r format)

| Category | N | % Positive | % Sig 5% |
|----------|---|------------|----------|
"""

    for cat, n in categories.items():
        if n > 0:
            cat_df = results_df[results_df['spec_id'].str.contains(cat.lower().split()[0])]
            if len(cat_df) == 0:
                # Try alternative matching
                if cat == 'Control variations':
                    cat_df = results_df[results_df['spec_id'].str.contains('controls|loo')]
                elif cat == 'Sample restrictions':
                    cat_df = results_df[results_df['spec_id'].str.contains('sample')]
                elif cat == 'Alternative outcomes':
                    cat_df = results_df[results_df['spec_id'].str.contains('outcome')]
                elif cat == 'Inference variations':
                    cat_df = results_df[results_df['spec_id'].str.contains('cluster|weights')]
                elif cat == 'Functional form':
                    cat_df = results_df[results_df['spec_id'].str.contains('funcform')]
                elif cat == 'Heterogeneity':
                    cat_df = results_df[results_df['spec_id'].str.contains('het')]
                elif cat == 'Placebo tests':
                    cat_df = results_df[results_df['spec_id'].str.contains('placebo')]

            if len(cat_df) > 0:
                pct_pos = (cat_df['coefficient'] > 0).mean() * 100
                pct_sig = (cat_df['p_value'] < 0.05).mean() * 100
                report += f"| {cat} | {len(cat_df)} | {pct_pos:.0f}% | {pct_sig:.0f}% |\n"

    report += f"| **TOTAL** | **{n_specs}** | **{pct_positive:.0f}%** | **{n_sig_05/n_specs*100:.0f}%** |\n"

    report += f"""
## Key Findings

1. **Main effect**: The baseline specification shows {'positive' if results_df[results_df['spec_id'].str.startswith('baseline')]['coefficient'].mean() > 0 else 'negative'} treatment effects on household welfare outcomes.

2. **Robustness to controls**: Results {'remain stable' if (results_df[results_df['spec_id'].str.contains('control|loo')]['p_value'] < 0.05).mean() > 0.5 else 'show some sensitivity'} when varying control sets.

3. **Heterogeneity**: Treatment effects {'vary significantly' if (results_df[results_df['spec_id'].str.contains('het')]['p_value'] < 0.05).mean() > 0.3 else 'are relatively homogeneous'} across subgroups.

## Critical Caveats

1. This is an RCT, so internal validity is high, but external validity depends on sample characteristics.
2. The paper's main analysis uses causal forests for heterogeneous treatment effects, which we approximate with simple subgroup analysis.
3. Time-demeaned outcomes (the "_T_" versions) account for time trends in the survey.

## Files Generated

- `specification_results.csv`
- `SPECIFICATION_SEARCH.md`
- `scripts/paper_analyses/{PAPER_ID}.py`
"""

    return report


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print(f"Specification Search: {PAPER_ID}")
    print(f"Paper: {PAPER_TITLE}")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df = load_data()
    print(f"Loaded {len(df)} observations")
    print(f"Treatment: {df[TREATMENT_VAR].sum()} treated, {(df[TREATMENT_VAR]==0).sum()} control")

    # Run specification search
    print("\nRunning specifications...")
    results = run_specification_search(df)

    # Save results
    print(f"\n{'='*70}")
    print(f"Saving {len(results)} specifications to {OUTPUT_FILE}")
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)

    # Generate and save summary report
    report = generate_summary_report(results_df, df)
    report_file = PACKAGE_DIR / "SPECIFICATION_SEARCH.md"
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"Summary report saved to {report_file}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    print(f"Total specifications: {len(results)}")
    print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()} ({(results_df['coefficient'] > 0).mean()*100:.1f}%)")
    print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()} ({(results_df['p_value'] < 0.05).mean()*100:.1f}%)")
    print(f"Significant at 1%: {(results_df['p_value'] < 0.01).sum()} ({(results_df['p_value'] < 0.01).mean()*100:.1f}%)")
    print(f"Median coefficient: {results_df['coefficient'].median():.4f}")
    print(f"Mean coefficient: {results_df['coefficient'].mean():.4f}")
    print(f"Coefficient range: [{results_df['coefficient'].min():.4f}, {results_df['coefficient'].max():.4f}]")

    print("\nDone!")
    return results_df


if __name__ == "__main__":
    main()
