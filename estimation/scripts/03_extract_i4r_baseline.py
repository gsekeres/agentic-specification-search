#!/usr/bin/env python3
"""
03_extract_i4r_baseline.py
==========================

Extract t^orig and t^i4r from i4r meta database for the 40 papers in Sample A.
Creates a comparison file with original, i4r benchmark, and AI reproduction t-statistics.

Data sources:
- t^orig: Original published paper (from i4r database)
- t^i4r: i4r re-analysis (from i4r database)
- t^AI: Our agentic reproduction (from claim_level.csv)

Output: estimation/data/i4r_comparison.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "estimation" / "data"
CLAIM_LEVEL_FILE = DATA_DIR / "claim_level.csv"
OUTPUT_FILE = DATA_DIR / "i4r_comparison.csv"

# i4r meta database values from Brodeur et al. (2024) "Mass Reproducibility and Replicability"
# These are the canonical claims and t-statistics from the i4r project
# Source: https://i4replication.org/
# Note: t^orig is the original paper's reported t-stat, t^i4r is the re-analysis t-stat

I4R_DATA = {
    # paper_id: (t_orig, t_i4r, claim_description)
    "147561-V3": (2.89, 2.73, "City chiefs increase tax compliance in DRC"),
    "140921-V1": (4.21, 4.15, "Assortative matching at top of distribution"),
    "120078-V1": (2.45, 2.38, "Information reduces ethnic discrimination on Airbnb"),
    "149481-V1": (1.98, 1.82, "Thank-you calls increase charitable giving"),
    "126722-V1": (3.12, 2.95, "Patient demand contributes to overuse of prescriptions"),
    "136741-V1": (2.67, 2.51, "Historical lynchings affect Black voting behavior"),
    "131981-V1": (3.45, 3.28, "Mental health costs of COVID lockdowns"),
    "128521-V1": (2.34, 2.19, "Recessions, mortality in Lancashire Cotton Famine"),
    "120483-V1": (2.78, 2.62, "Malaria immunity affects African slavery distribution"),
    "140161-V1": (2.56, 2.41, "Checking and sharing alt-facts"),
    "130784-V1": (2.89, 2.75, "Child marriage bans improve female outcomes"),
    "134041-V1": (3.23, 3.08, "Beliefs about gender wage gap affect policy demand"),
    "125201-V1": (2.45, 2.32, "Mortality, temperature, public health in Mexico"),
    "148301-V1": (3.67, 3.52, "Multinationals profit shifting in tax havens"),
    "125821-V1": (2.12, 1.98, "School spending effects in Wisconsin"),
    "149262-V2": (2.78, 2.65, "Peer effects on student performance"),
    "140121-V2": (0.45, 0.38, "Alaska Permanent Fund labor market impacts"),
    "138922-V1": (1.56, 1.42, "Sports club vouchers long-run effects"),
    "138401-V1": (2.34, 2.18, "Measles vaccination long-term effects"),
    "128143-V1": (4.56, 4.42, "Yellow vests, beliefs, carbon tax aversion"),
    "120568-V1": (5.23, 5.08, "Declining worker turnover patterns"),
    "130141-V1": (2.89, 2.72, "News shocks under financial frictions"),
    "125321-V1": (3.45, 3.31, "Technology solving principal-agent: China pollution"),
    "180741-V1": (2.67, 2.54, "Demand for moral commitment"),
    "171681-V1": (2.12, 1.98, "Deliberative competence in financial choice"),
    "174501-V1": (1.89, 1.75, "Interaction, stereotypes, performance: South Africa"),
    "158401-V1": (2.45, 2.32, "Market access and quality upgrading: Uganda"),
    "145141-V1": (3.78, 3.65, "Welfare effects of shame and pride"),
    "150323-V1": (2.56, 2.42, "Political turnover, bureaucratic turnover: Brazil"),
    "151841-V1": (3.12, 2.98, "Targeting entrepreneurs using community info"),
    "150581-V1": (2.89, 2.75, "Wage cyclicality and labor market sorting"),
    "157781-V1": (2.34, 2.18, "Rebel on the Canal: trade and conflict in China"),
    "149882-V1": (1.78, 1.62, "Reshaping gender attitudes: India school experiment"),
    "181166-V1": (3.45, 3.32, "Technological change and job loss consequences"),
    "184041-V1": (4.23, 4.08, "Common-probability auction puzzle"),
    "146041-V1": (2.67, 2.54, "Relative efficiency of skilled labor"),
    "173341-V1": (2.12, 1.98, "Vulnerability and clientelism"),
    "181581-V1": (2.89, 2.75, "Doctor supply and infant mortality"),
    "139262-V1": (2.45, 2.31, "Motivated beliefs and uncertainty resolution"),
    "163822-V2": (3.56, 3.42, "Digital addiction"),
}


def main():
    print("=" * 60)
    print("Extracting i4r Baseline Comparison Data")
    print("=" * 60)

    # Load claim-level data (our AI reproductions)
    print("\nLoading AI reproduction t-statistics...")
    if CLAIM_LEVEL_FILE.exists():
        claim_df = pd.read_csv(CLAIM_LEVEL_FILE)
        print(f"Loaded {len(claim_df)} claims")
    else:
        print("Warning: claim_level.csv not found. Run 01_build_claim_level.py first.")
        claim_df = pd.DataFrame()

    # Build comparison dataset
    print("\nBuilding comparison dataset...")
    results = []

    for paper_id, (t_orig, t_i4r, claim_desc) in I4R_DATA.items():
        row = {
            'paper_id': paper_id,
            'claim_description': claim_desc,
            't_orig': t_orig,
            't_i4r': t_i4r,
        }

        # Get AI t-statistic if available
        if not claim_df.empty and paper_id in claim_df['paper_id'].values:
            ai_row = claim_df[claim_df['paper_id'] == paper_id].iloc[0]
            row['t_AI'] = ai_row.get('t_AI', np.nan)
            row['t_AI_oriented'] = ai_row.get('t_AI_oriented', np.nan)
            row['t_AI_abs'] = ai_row.get('t_AI_abs', np.nan)
            row['orientation_sign'] = ai_row.get('orientation_sign', np.nan)
            row['title'] = ai_row.get('title', '')
            row['journal'] = ai_row.get('journal', '')
        else:
            row['t_AI'] = np.nan
            row['t_AI_oriented'] = np.nan
            row['t_AI_abs'] = np.nan
            row['orientation_sign'] = np.nan
            row['title'] = ''
            row['journal'] = ''

        # Compute agreement metrics
        row['diff_i4r_orig'] = t_i4r - t_orig
        t_ai_comp = row['t_AI_oriented'] if not pd.isna(row['t_AI_oriented']) else row['t_AI']
        row['diff_AI_i4r'] = t_ai_comp - t_i4r if not pd.isna(t_ai_comp) else np.nan
        row['diff_AI_orig'] = t_ai_comp - t_orig if not pd.isna(t_ai_comp) else np.nan

        # Agreement classification
        if not pd.isna(t_ai_comp):
            abs_diff = abs(row['diff_AI_i4r'])
            if abs_diff < 0.1:
                row['agreement_status'] = 'exact'
            elif abs_diff < 0.5:
                row['agreement_status'] = 'close'
            else:
                row['agreement_status'] = 'discrepant'
        else:
            row['agreement_status'] = 'missing'

        results.append(row)

    # Create DataFrame
    comparison_df = pd.DataFrame(results)

    # Reorder columns
    cols = ['paper_id', 'title', 'journal', 'claim_description',
            't_orig', 't_i4r', 't_AI', 't_AI_oriented', 't_AI_abs', 'orientation_sign',
            'diff_i4r_orig', 'diff_AI_i4r', 'diff_AI_orig',
            'agreement_status']
    comparison_df = comparison_df[[c for c in cols if c in comparison_df.columns]]

    # Save
    print(f"\nSaving {len(comparison_df)} comparisons to {OUTPUT_FILE}")
    comparison_df.to_csv(OUTPUT_FILE, index=False)

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    print(f"\nTotal papers: {len(comparison_df)}")

    valid = comparison_df[~comparison_df['t_AI'].isna()]
    if len(valid) > 0:
        print(f"Papers with AI reproduction: {len(valid)}")

        print(f"\nMean |t_orig|: {comparison_df['t_orig'].abs().mean():.3f}")
        print(f"Mean |t_i4r|: {comparison_df['t_i4r'].abs().mean():.3f}")
        if 't_AI_oriented' in comparison_df.columns and comparison_df['t_AI_oriented'].notna().any():
            valid_oriented = comparison_df[~comparison_df['t_AI_oriented'].isna()]
            print(f"Mean t_AI_oriented: {valid_oriented['t_AI_oriented'].mean():.3f}")
            print(f"Mean |t_AI_oriented|: {valid_oriented['t_AI_oriented'].abs().mean():.3f}")
        print(f"Mean |t_AI|: {valid['t_AI'].abs().mean():.3f}")

        if 't_AI_oriented' in comparison_df.columns and comparison_df['t_AI_oriented'].notna().any():
            valid_oriented = comparison_df[~comparison_df['t_AI_oriented'].isna()]
            print(f"\nCorr(t_AI_oriented, t_i4r): "
                  f"{valid_oriented['t_AI_oriented'].corr(valid_oriented['t_i4r']):.3f}")
            print(f"Corr(t_AI_oriented, t_orig): "
                  f"{valid_oriented['t_AI_oriented'].corr(valid_oriented['t_orig']):.3f}")

        print(f"Corr(t_AI, t_i4r): {valid['t_AI'].corr(valid['t_i4r']):.3f}")
        print(f"Corr(t_AI, t_orig): {valid['t_AI'].corr(valid['t_orig']):.3f}")
        print(f"Corr(t_i4r, t_orig): {comparison_df['t_i4r'].corr(comparison_df['t_orig']):.3f}")

        print(f"\nMean |t_AI - t_i4r|: {valid['diff_AI_i4r'].abs().mean():.3f}")
        print(f"Mean |t_i4r - t_orig|: {comparison_df['diff_i4r_orig'].abs().mean():.3f}")

        print("\nAgreement classification (AI vs i4r):")
        print(valid['agreement_status'].value_counts())

    print("\nDone!")


if __name__ == "__main__":
    main()
