#!/usr/bin/env python3
"""
Paper Analysis Template: {PAPER_ID}
====================================

This template provides the structure for running systematic specification
searches on AEA replication packages.

Usage:
    python scripts/paper_analyses/{PAPER_ID}.py

Replace {PAPER_ID} with the actual paper identifier (e.g., 223561-V1).
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Try importing pyfixest, fall back to statsmodels
try:
    import pyfixest as pf
    USE_PYFIXEST = True
except ImportError:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    USE_PYFIXEST = False

# =============================================================================
# CONFIGURATION - Edit these for each paper
# =============================================================================

PAPER_ID = "{PAPER_ID}"  # e.g., "223561-V1"
JOURNAL = "{JOURNAL}"  # e.g., "AER", "AEJ-Applied"
PAPER_TITLE = "{PAPER_TITLE}"
METHOD_TYPE = "{METHOD_TYPE}"  # e.g., "difference_in_differences"

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
PACKAGE_DIR = BASE_DIR / "data" / "downloads" / "extracted" / PAPER_ID
OUTPUT_FILE = PACKAGE_DIR / "specification_results.csv"

# Variables - customize for this paper
OUTCOME_VAR = "y"
TREATMENT_VAR = "treat"
CONTROL_VARS = ["control1", "control2", "control3"]
FIXED_EFFECTS = ["unit", "time"]
CLUSTER_VAR = "unit"

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load and prepare the data for analysis."""
    # Adjust path and format for your data
    data_file = PACKAGE_DIR / "data" / "main_data.dta"

    if data_file.suffix == '.dta':
        df = pd.read_stata(data_file)
    elif data_file.suffix == '.csv':
        df = pd.read_csv(data_file)
    elif data_file.suffix == '.parquet':
        df = pd.read_parquet(data_file)
    else:
        raise ValueError(f"Unknown data format: {data_file.suffix}")

    # Add any necessary variable transformations here
    # df['log_y'] = np.log(df['y'] + 1)

    return df


# =============================================================================
# SPECIFICATION RUNNER
# =============================================================================

def run_specification(df, outcome, treatment, controls, fe, cluster,
                      spec_id, spec_tree_path, sample_desc="Full sample"):
    """
    Run a single specification and return results dictionary.
    """
    # Build formula
    control_str = " + ".join(controls) if controls else ""
    fe_str = " + ".join(fe) if fe else ""

    if USE_PYFIXEST:
        formula = f"{outcome} ~ {treatment}"
        if control_str:
            formula += f" + {control_str}"
        if fe_str:
            formula += f" | {fe_str}"

        try:
            model = pf.feols(formula, data=df, vcov={'CRV1': cluster})
            coef = model.coef()[treatment]
            se = model.se()[treatment]
            pval = model.pvalue()[treatment]
            n_obs = model.nobs
            r2 = model.r2
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
                "fixed_effects": fe,
                "diagnostics": {}
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
            print(f"Error running {spec_id}: {e}")
            return None

    else:
        # Fallback to statsmodels
        formula = f"{outcome} ~ {treatment}"
        if control_str:
            formula += f" + {control_str}"
        if fe_str:
            formula += f" + C({fe[0]})"
            for f in fe[1:]:
                formula += f" + C({f})"

        try:
            model = smf.ols(formula, data=df).fit(
                cov_type='cluster',
                cov_kwds={'groups': df[cluster]}
            )
            coef = model.params[treatment]
            se = model.bse[treatment]
            pval = model.pvalues[treatment]
            n_obs = model.nobs
            r2 = model.rsquared
            tstat = model.tvalues[treatment]
            ci_lower, ci_upper = model.conf_int().loc[treatment]

            coef_vector = {
                "treatment": {"var": treatment, "coef": float(coef),
                              "se": float(se), "pval": float(pval)},
                "controls": [],
                "fixed_effects": fe,
                "diagnostics": {}
            }

        except Exception as e:
            print(f"Error running {spec_id}: {e}")
            return None

    return {
        'paper_id': PAPER_ID,
        'journal': JOURNAL,
        'paper_title': PAPER_TITLE,
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'outcome_var': outcome,
        'treatment_var': treatment,
        'coefficient': coef,
        'std_error': se,
        't_stat': tstat,
        'p_value': pval,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_obs': n_obs,
        'r_squared': r2,
        'coefficient_vector_json': json.dumps(coef_vector),
        'sample_desc': sample_desc,
        'fixed_effects': ", ".join(fe) if fe else "None",
        'controls_desc': ", ".join(controls) if controls else "None",
        'cluster_var': cluster,
        'model_type': "FE" if fe else "OLS",
        'estimation_script': f"scripts/paper_analyses/{PAPER_ID}.py"
    }


# =============================================================================
# SPECIFICATION SEARCH
# =============================================================================

def run_specification_search(df):
    """Run all specifications for this paper."""
    results = []

    # -------------------------------------------------------------------------
    # BASELINE
    # -------------------------------------------------------------------------
    print("Running baseline...")
    result = run_specification(
        df, OUTCOME_VAR, TREATMENT_VAR, CONTROL_VARS, FIXED_EFFECTS, CLUSTER_VAR,
        spec_id="baseline",
        spec_tree_path=f"methods/{METHOD_TYPE}.md"
    )
    if result:
        results.append(result)

    # -------------------------------------------------------------------------
    # FIXED EFFECTS VARIATIONS
    # -------------------------------------------------------------------------
    print("Running FE variations...")
    fe_variations = [
        ("did/fe/none", []),
        ("did/fe/unit_only", [FIXED_EFFECTS[0]]),
        ("did/fe/time_only", [FIXED_EFFECTS[1]] if len(FIXED_EFFECTS) > 1 else []),
        ("did/fe/twoway", FIXED_EFFECTS),
    ]

    for spec_id, fe in fe_variations:
        result = run_specification(
            df, OUTCOME_VAR, TREATMENT_VAR, CONTROL_VARS, fe, CLUSTER_VAR,
            spec_id=spec_id,
            spec_tree_path=f"methods/{METHOD_TYPE}.md#fixed-effects"
        )
        if result:
            results.append(result)

    # -------------------------------------------------------------------------
    # CONTROL SET VARIATIONS
    # -------------------------------------------------------------------------
    print("Running control variations...")
    control_variations = [
        ("did/controls/none", []),
        ("did/controls/full", CONTROL_VARS),
    ]

    for spec_id, controls in control_variations:
        result = run_specification(
            df, OUTCOME_VAR, TREATMENT_VAR, controls, FIXED_EFFECTS, CLUSTER_VAR,
            spec_id=spec_id,
            spec_tree_path=f"methods/{METHOD_TYPE}.md#control-sets"
        )
        if result:
            results.append(result)

    # -------------------------------------------------------------------------
    # LEAVE-ONE-OUT
    # -------------------------------------------------------------------------
    print("Running leave-one-out...")
    for control in CONTROL_VARS:
        remaining = [c for c in CONTROL_VARS if c != control]
        result = run_specification(
            df, OUTCOME_VAR, TREATMENT_VAR, remaining, FIXED_EFFECTS, CLUSTER_VAR,
            spec_id=f"robust/loo/drop_{control}",
            spec_tree_path="robustness/leave_one_out.md"
        )
        if result:
            results.append(result)

    # -------------------------------------------------------------------------
    # SINGLE COVARIATE
    # -------------------------------------------------------------------------
    print("Running single covariate...")
    # Bivariate
    result = run_specification(
        df, OUTCOME_VAR, TREATMENT_VAR, [], FIXED_EFFECTS, CLUSTER_VAR,
        spec_id="robust/single/none",
        spec_tree_path="robustness/single_covariate.md"
    )
    if result:
        results.append(result)

    for control in CONTROL_VARS:
        result = run_specification(
            df, OUTCOME_VAR, TREATMENT_VAR, [control], FIXED_EFFECTS, CLUSTER_VAR,
            spec_id=f"robust/single/{control}",
            spec_tree_path="robustness/single_covariate.md"
        )
        if result:
            results.append(result)

    # -------------------------------------------------------------------------
    # CLUSTERING VARIATIONS
    # -------------------------------------------------------------------------
    print("Running clustering variations...")
    cluster_vars = [CLUSTER_VAR]  # Add more cluster variables as available
    # cluster_vars = [CLUSTER_VAR, 'region', 'time']

    for cluster in cluster_vars:
        if cluster in df.columns:
            result = run_specification(
                df, OUTCOME_VAR, TREATMENT_VAR, CONTROL_VARS, FIXED_EFFECTS, cluster,
                spec_id=f"robust/cluster/{cluster}",
                spec_tree_path="robustness/clustering_variations.md"
            )
            if result:
                results.append(result)

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print(f"Specification Search: {PAPER_ID}")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    df = load_data()
    print(f"Loaded {len(df)} observations")

    # Run specification search
    print("\nRunning specifications...")
    results = run_specification_search(df)

    # Save results
    print(f"\nSaving {len(results)} specifications to {OUTPUT_FILE}")
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total specifications: {len(results)}")
    if 'coefficient' in results_df.columns:
        print(f"Positive coefficients: {(results_df['coefficient'] > 0).sum()}")
        print(f"Significant at 5%: {(results_df['p_value'] < 0.05).sum()}")
        print(f"Median coefficient: {results_df['coefficient'].median():.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
