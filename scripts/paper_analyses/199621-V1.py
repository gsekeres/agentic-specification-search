"""
Specification Search Script: 199621-V1
Paper: "The Decline of Too Big To Fail"
Authors: Antje Berndt, Darrell Duffie, and Yichao Zhu
Journal: American Economic Review

STATUS: DATA NOT AVAILABLE

This paper requires proprietary data that is NOT included in the replication package.
The main analysis sample (sample_daily.dta) must be constructed from:
1. IHS Markit CDS data (via WRDS)
2. CRSP Daily Stock File (via WRDS)
3. Compustat Quarterly/Annual (via WRDS)
4. OptionMetrics (via WRDS)
5. Moody's Default and Recovery Database (via Moody's SFTP)

Method Classification: Structural Calibration (primary) + Panel Fixed Effects (auxiliary)

If data becomes available, this script would:
1. Load sample_daily.dta
2. Replicate Table A.3 reduced-form panel FE regression
3. Run specification variations per panel_fixed_effects.md
4. Run robustness checks per the robustness templates
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path

# Configuration
PAPER_ID = "199621-V1"
JOURNAL = "AER"
PAPER_TITLE = "The Decline of Too Big To Fail"
DATA_PATH = Path("/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/199621-V1/Replication/01_Data/")
OUTPUT_PATH = DATA_PATH

# Check for required data
required_file = DATA_PATH / "sample_daily.dta"

def check_data_availability():
    """Check if required data files exist."""
    if not required_file.exists():
        print(f"ERROR: Required data file not found: {required_file}")
        print("\nThis paper requires data from proprietary sources:")
        print("1. IHS Markit CDS data (via WRDS)")
        print("2. CRSP Daily Stock File (via WRDS)")
        print("3. Compustat (via WRDS)")
        print("4. OptionMetrics (via WRDS)")
        print("5. Moody's DRD (via Moody's SFTP)")
        print("\nSee the replication package README for data construction instructions.")
        return False
    return True


def run_table_a3_replication():
    """
    Replicate Table A.3: Reduced-form estimates of no-bailout probabilities for GSIBs

    Specification:
    - Outcome: log(CDS_5yr)
    - Treatment: GSIB indicator interacted with time FE
    - Controls: Tangible equity ratio (bvetan/bvatan)
    - Fixed Effects: Firm (permco) + Time (date)
    - Standard Errors: Robust heteroskedasticity
    """
    try:
        import pyfixest as pf
    except ImportError:
        print("pyfixest not installed. Run: pip install pyfixest")
        return None

    # Load data
    df = pd.read_stata(DATA_PATH / "sample_daily.dta")

    # Create variables following TableA3.do
    # ratio_tan = max(0, bvetan/bvatan) filtered for ratio_tan < 1
    df['ratio_tan'] = np.maximum(0, df['bvetan'] / df['bvatan'])
    df = df[(df['ratio_tan'] < 1) & (df['ratio_tan'].notna())]

    # logcds = ln(cds5)
    df['logcds'] = np.log(df['cds5'])

    # Panel A: Non-banks only
    df_nonbank = df[(df['gsib'] == 0) & (df['olb'] == 0) & (df['bank_small'] == 0)]

    # Panel B: GSIBs + Non-banks (excluding OLBs and small banks)
    df_gsib_nonbank = df[(df['olb'] == 0) & (df['bank_small'] == 0)]

    results = []

    # Panel A: Non-banks only
    try:
        model_a = pf.feols(
            "logcds ~ ratio_tan | permco + date",
            data=df_nonbank,
            vcov='HC1'  # Robust SE
        )
        results.append({
            'spec_id': 'baseline_panel_a',
            'spec_tree_path': 'methods/panel_fixed_effects.md#fixed-effects-structure',
            'outcome_var': 'logcds',
            'treatment_var': 'ratio_tan',
            'coefficient': model_a.coef()['ratio_tan'],
            'std_error': model_a.se()['ratio_tan'],
            't_stat': model_a.tstat()['ratio_tan'],
            'p_value': model_a.pvalue()['ratio_tan'],
            'n_obs': model_a.nobs,
            'r_squared': model_a.r2,
            'sample_desc': 'Non-banks only',
            'fixed_effects': 'Firm + Time',
            'cluster_var': 'None (robust)',
            'model_type': 'Panel FE'
        })
    except Exception as e:
        print(f"Panel A regression failed: {e}")

    # Panel B: GSIBs + Non-banks with GSIB-time interactions
    # This requires creating gsib_mo1-gsib_mo264 interaction terms
    # ... (implementation would follow TableA3.do logic)

    return results


def main():
    """Main execution function."""
    print(f"=" * 60)
    print(f"Specification Search: {PAPER_ID}")
    print(f"Paper: {PAPER_TITLE}")
    print(f"Journal: {JOURNAL}")
    print(f"=" * 60)
    print()

    # Check data availability
    if not check_data_availability():
        print("\nExiting: Cannot run specification search without data.")
        sys.exit(1)

    # If data exists, run analysis
    results = run_table_a3_replication()

    if results:
        # Save results
        results_df = pd.DataFrame(results)
        output_file = OUTPUT_PATH / "specification_results.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    else:
        print("\nNo results generated.")


if __name__ == "__main__":
    main()
