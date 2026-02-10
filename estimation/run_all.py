#!/usr/bin/env python3
"""
run_all.py
==========

Master script to run all estimation and figure generation.

Usage:
    python estimation/run_all.py                # Run data + estimation + figures
    python estimation/run_all.py --data         # Data construction only
    python estimation/run_all.py --est          # Estimation only
    python estimation/run_all.py --figs         # Figures only
    python estimation/run_all.py --extensions   # Extension analyses only
    python estimation/run_all.py --all          # Everything including extensions
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
SCRIPTS_DIR = BASE_DIR / "estimation" / "scripts"


def run_python(script_name):
    """Run a Python script."""
    script_path = SCRIPTS_DIR / script_name
    print(f"\n{'=' * 60}")
    print(f"Running: {script_name}")
    print("=" * 60)
    result = subprocess.run([sys.executable, str(script_path)], cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"Warning: {script_name} exited with code {result.returncode}")
    return result.returncode


def run_julia(script_name):
    """Run a Julia script."""
    script_path = SCRIPTS_DIR / script_name
    print(f"\n{'=' * 60}")
    print(f"Running: {script_name}")
    print("=" * 60)
    result = subprocess.run(["julia", str(script_path)], cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"Warning: {script_name} exited with code {result.returncode}")
    return result.returncode


def main():
    args = sys.argv[1:]

    run_all_flag = '--all' in args
    run_data = '--data' in args or len(args) == 0 or run_all_flag
    run_est = '--est' in args or len(args) == 0 or run_all_flag
    run_figs = '--figs' in args or len(args) == 0 or run_all_flag
    run_ext = '--extensions' in args or run_all_flag

    print("=" * 60)
    print("Empirical Analysis Pipeline")
    print("=" * 60)

    # Step 1: Data construction
    if run_data:
        print("\n" + "#" * 60)
        print("STEP 1: DATA CONSTRUCTION")
        print("#" * 60)

        run_python("00_summarize_verification.py")
        run_python("01a_build_i4r_claim_map.py")
        run_python("01b_build_i4r_oracle_claim_map.py")
        run_python("01_build_claim_level.py")
        run_python("02_build_spec_level.py")
        run_python("03_extract_i4r_baseline.py")
        run_python("08_i4r_paper_audit.py")
        run_python("10_inference_audit.py")
        run_python("07_i4r_discrepancies.py")

    # Step 2: Estimation
    if run_est:
        print("\n" + "#" * 60)
        print("STEP 2: ESTIMATION")
        print("#" * 60)

        run_python("04_fit_mixture.py")
        run_python("05_estimate_dependence.py")
        run_python("06_counterfactual.py")
        # Appendix tables (requires outputs from both data + estimation stages)
        if (BASE_DIR / "estimation" / "results" / "inference_audit_i4r.csv").exists():
            run_python("11_write_overleaf_tables.py")
        else:
            print("Skipping 11_write_overleaf_tables.py (missing inference_audit_i4r.csv; run --data first).")

    # Step 3: Figures (Julia)
    if run_figs:
        print("\n" + "#" * 60)
        print("STEP 3: FIGURES")
        print("#" * 60)

        run_julia("make_figures.jl")

    # Step 4: Extension analyses (appendix robustness)
    if run_ext:
        print("\n" + "#" * 60)
        print("STEP 4: EXTENSION ANALYSES")
        print("#" * 60)

        # Mixture extensions
        run_python("12_bootstrap_mixture_ci.py")
        run_python("14_leave_one_out_cv.py")
        run_python("15_journal_subgroup.py")
        run_python("16_posterior_assignment.py")

        # Dependence extensions
        run_python("17_dependence_heterogeneity.py")
        run_python("18_sign_consistency.py")
        run_python("19_funnel_plot.py")

        # Counterfactual extensions
        run_python("20_counterfactual_montecarlo.py")
        run_python("21_effective_sample_size.py")
        run_python("22_window_surface.py")
        run_python("23_within_paper_dispersion.py")

        # Summary tables
        run_python("24_summary_statistics.py")
        run_python("25_variance_decomposition.py")
        run_python("26_build_paper_catalog.py")
        run_python("27_mixture_comparison_table.py")
        run_python("27_sensitivity_tables.py")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)

    print("\nOutputs:")
    print("  Data:      estimation/data/{claim_level,spec_level,i4r_comparison}.csv")
    print("  Results:   estimation/results/{mixture_params,dependence,counterfactual}*.json")
    print("  Figures:   estimation/figures/*.pdf")
    print("  Overleaf:  overleaf/tex/v8_figures/*.pdf, overleaf/tex/v8_tables/*.tex")


if __name__ == "__main__":
    main()
