#!/usr/bin/env python3
"""
Specification Search: 231591-V1
Paper: "The Rise of NGO Activism" by Daubanes and Rochet
Journal: AEJ: Papers & Proceedings

IMPORTANT: This script documents the analysis that WOULD be run if data were available.
The critical NGO negative reports data is proprietary (Covalence Ethical Quote) and
not included in the replication package.

Original Stata code runs:
  Model I:  xtreg neg_reports d.lobby L.d.lobby i.year, fe vce(cluster _ID)
  Model II: xtreg lobby neg_reports L.neg_reports i.year, fe vce(cluster _ID)

Method: Panel Fixed Effects with dynamic elements
Method Tree Path: specification_tree/methods/panel_fixed_effects.md
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

# Try to import estimation packages (for documentation purposes)
try:
    import pyfixest as pf
    PYFIXEST_AVAILABLE = True
except ImportError:
    PYFIXEST_AVAILABLE = False

try:
    from linearmodels.panel import PanelOLS, RandomEffects
    LINEARMODELS_AVAILABLE = True
except ImportError:
    LINEARMODELS_AVAILABLE = False

# =============================================================================
# Configuration
# =============================================================================
PAPER_ID = "231591-V1"
PAPER_TITLE = "The Rise of NGO Activism"
JOURNAL = "AEJ: P&P"
AUTHORS = "Julien Daubanes and Jean-Charles Rochet"

# Paths
BASE_DIR = Path("/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search")
DATA_DIR = BASE_DIR / "data/downloads/extracted/231591-V1/data/Data"
OUTPUT_DIR = BASE_DIR / "data/downloads/extracted/231591-V1"
SCRIPT_PATH = "scripts/paper_analyses/231591-V1.py"

# =============================================================================
# Data Loading (Available Data Only)
# =============================================================================

def load_lobbying_data():
    """
    Load lobbying data from Lobbying.xlsx.
    This is the ONLY data available in the replication package.
    """
    xlsx_path = DATA_DIR / "Lobbying.xlsx"

    # Read raw data (irregular Excel format)
    df_raw = pd.read_excel(xlsx_path, header=None)

    # Extract structure
    years = [2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014]
    sectors = df_raw.iloc[6:16, 0].values
    lobby_data = df_raw.iloc[6:16, 1:14].values

    # Create long-format panel
    panel_data = []
    for i, sector in enumerate(sectors):
        for j, year in enumerate(years):
            panel_data.append({
                'sector': sector,
                'year': year,
                'lobby': lobby_data[i, j]
            })

    df = pd.DataFrame(panel_data)

    # Convert lobby to numeric and to millions
    df['lobby'] = pd.to_numeric(df['lobby'], errors='coerce') / 1_000_000

    # Create sector ID
    df['_ID'] = df.groupby('sector').ngroup() + 1

    # Sort and set index
    df = df.sort_values(['_ID', 'year']).reset_index(drop=True)

    return df


def prepare_panel_data(df):
    """
    Prepare panel structure for estimation.
    Creates first differences and lags as in original Stata code.
    """
    df = df.copy()

    # Sort for proper lag/diff calculation
    df = df.sort_values(['_ID', 'year'])

    # First difference of lobby (d.lobby in Stata)
    df['d_lobby'] = df.groupby('_ID')['lobby'].diff()

    # Lag of first difference (L.d.lobby in Stata)
    df['L_d_lobby'] = df.groupby('_ID')['d_lobby'].shift(1)

    return df


# =============================================================================
# Specification Functions (Would be run if data available)
# =============================================================================

def run_baseline_model_1(df):
    """
    Model I: neg_reports ~ d.lobby + L.d.lobby | sector + year, cluster(sector)

    CANNOT RUN: neg_reports variable not available
    """
    if 'neg_reports' not in df.columns:
        return {
            'spec_id': 'baseline_model1',
            'spec_tree_path': 'methods/panel_fixed_effects.md#baseline',
            'status': 'data_missing',
            'error': 'neg_reports variable not in data (proprietary CEQ data required)'
        }

    # This code would run if data were available:
    # model = pf.feols("neg_reports ~ d_lobby + L_d_lobby | _ID + year",
    #                  data=df, vcov={'CRV1': '_ID'})
    # return extract_results(model, 'baseline_model1', 'd_lobby')


def run_baseline_model_2(df):
    """
    Model II: lobby ~ neg_reports + L.neg_reports | sector + year, cluster(sector)

    CANNOT RUN: neg_reports variable not available
    """
    if 'neg_reports' not in df.columns:
        return {
            'spec_id': 'baseline_model2',
            'spec_tree_path': 'methods/panel_fixed_effects.md#baseline',
            'status': 'data_missing',
            'error': 'neg_reports variable not in data (proprietary CEQ data required)'
        }


def run_fe_variations(df):
    """
    Fixed effects variations from panel_fixed_effects.md:
    - panel/fe/none: Pooled OLS
    - panel/fe/unit: Sector FE only
    - panel/fe/time: Year FE only
    - panel/fe/twoway: Sector + Year FE

    All CANNOT RUN due to missing outcome/treatment variables.
    """
    specs = [
        ('panel/fe/none', 'Pooled OLS'),
        ('panel/fe/unit', 'Sector FE only'),
        ('panel/fe/time', 'Year FE only'),
        ('panel/fe/twoway', 'Sector + Year FE'),
    ]

    results = []
    for spec_id, desc in specs:
        results.append({
            'spec_id': spec_id,
            'spec_tree_path': 'methods/panel_fixed_effects.md#fixed-effects-structure',
            'status': 'data_missing',
            'description': desc,
            'error': 'neg_reports variable not in data'
        })
    return results


def run_clustering_variations(df):
    """
    Clustering variations from robustness/clustering_variations.md:
    - robust/cluster/none: No clustering
    - robust/cluster/unit: By sector (baseline)

    All CANNOT RUN due to missing data.
    """
    specs = [
        ('robust/cluster/none', 'Robust SE, no clustering'),
        ('robust/cluster/unit', 'Cluster by sector'),
    ]

    results = []
    for spec_id, desc in specs:
        results.append({
            'spec_id': spec_id,
            'spec_tree_path': 'robustness/clustering_variations.md',
            'status': 'data_missing',
            'description': desc,
            'error': 'neg_reports variable not in data'
        })
    return results


def run_sample_restrictions(df):
    """
    Sample restriction robustness from robustness/sample_restrictions.md:
    - robust/sample/early: 2002-2008
    - robust/sample/late: 2009-2014

    All CANNOT RUN due to missing data.
    """
    specs = [
        ('robust/sample/early', '2002-2008 subsample'),
        ('robust/sample/late', '2009-2014 subsample'),
    ]

    results = []
    for spec_id, desc in specs:
        results.append({
            'spec_id': spec_id,
            'spec_tree_path': 'robustness/sample_restrictions.md',
            'status': 'data_missing',
            'description': desc,
            'error': 'neg_reports variable not in data'
        })
    return results


# =============================================================================
# Main Execution
# =============================================================================

def main():
    print("=" * 70)
    print(f"Specification Search: {PAPER_ID}")
    print(f"Paper: {PAPER_TITLE}")
    print(f"Authors: {AUTHORS}")
    print("=" * 70)

    # Load available data
    print("\n[1] Loading available data...")
    df = load_lobbying_data()
    print(f"    Loaded lobbying data: {len(df)} observations")
    print(f"    Sectors: {df['sector'].nunique()}")
    print(f"    Years: {df['year'].min()} - {df['year'].max()}")

    # Prepare panel
    print("\n[2] Preparing panel structure...")
    df = prepare_panel_data(df)

    # Check for required variables
    print("\n[3] Checking data availability...")
    required_vars = ['neg_reports', 'ngo', 'events']
    for var in required_vars:
        if var not in df.columns:
            print(f"    MISSING: {var} (proprietary CEQ data)")
        else:
            print(f"    FOUND: {var}")

    # Document available data summary
    print("\n[4] Available data summary:")
    print(f"    Lobby (millions USD):")
    print(f"      Mean: ${df['lobby'].mean():.1f}M")
    print(f"      Min:  ${df['lobby'].min():.1f}M")
    print(f"      Max:  ${df['lobby'].max():.1f}M")

    # Document what would be run
    print("\n[5] Specifications that WOULD be run:")

    all_specs = []

    # Baseline
    result = run_baseline_model_1(df)
    all_specs.append(result)
    print(f"    - baseline_model1: {result['status']}")

    result = run_baseline_model_2(df)
    all_specs.append(result)
    print(f"    - baseline_model2: {result['status']}")

    # FE variations
    for result in run_fe_variations(df):
        all_specs.append(result)
        print(f"    - {result['spec_id']}: {result['status']}")

    # Clustering
    for result in run_clustering_variations(df):
        all_specs.append(result)
        print(f"    - {result['spec_id']}: {result['status']}")

    # Sample restrictions
    for result in run_sample_restrictions(df):
        all_specs.append(result)
        print(f"    - {result['spec_id']}: {result['status']}")

    print(f"\n[6] Total planned specifications: {len(all_specs)}")
    print(f"    Specifications run: 0")
    print(f"    Data missing: {len(all_specs)}")

    print("\n" + "=" * 70)
    print("RESULT: CANNOT COMPLETE SPECIFICATION SEARCH")
    print("REASON: Proprietary CEQ data not included in replication package")
    print("ACTION: Purchase CEQ data from https://www.covalence.ch")
    print("=" * 70)

    return all_specs


if __name__ == "__main__":
    main()
