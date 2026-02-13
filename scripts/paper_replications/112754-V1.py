"""
Replication script for 112754-V1:
"Search, Liquidity, and the Dynamics of House Prices and Construction"
by Allen Head, Huw Lloyd-Ellis, and Hongfei Sun
American Economic Review, 2014, 104(4), 1172-1210

STATUS: NOT POSSIBLE

This paper cannot be replicated in Python because its main results depend on:

1. Panel VAR estimation using Inessa Love's `pvar` Stata command (system-GMM),
   which has no Python equivalent. The external package is available from:
   http://go.worldbank.org/E96NEWM7L0

2. Dynare DSGE model simulations (requires Matlab + Dynare), which generate
   the theoretical moments reported in the paper's main tables.

The Stata do-files estimate panel VARs with 5-6 variables (income, prices,
sales, construction rate, population growth, plus one of: construction wages,
construction labor, or rents) using system-GMM with Helmert (forward
orthogonal deviations) transformation for fixed effects removal.

The Dynare .mod files define and simulate a structural search model of the
housing market, and the Matlab .m files compute simulated moments from the
linearized quarterly model.

This script only loads and prepares the data as a reference for the
data-preparation steps that precede the panel VAR estimation.
"""

import os
import sys
import pandas as pd
import numpy as np
import csv
import json

# ============================================================
# Configuration
# ============================================================
PAPER_ID = "112754-V1"
PACKAGE_DIR = "/Users/gabesekeres/Dropbox/Papers/competition_science/agentic_specification_search/data/downloads/extracted/112754-V1"
DATA_DIR = os.path.join(PACKAGE_DIR, "data", "Original2014_data", "AER-2011-1278_Estimation")
OUTPUT_CSV = os.path.join(PACKAGE_DIR, "replication.csv")

# ============================================================
# Load data (for reference only)
# ============================================================
print(f"Loading data from {DATA_DIR}/msadata.dta ...")
try:
    df = pd.read_stata(os.path.join(DATA_DIR, "msadata.dta"))
    print(f"Data loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"MSA codes: {df['msacode'].nunique()} unique cities")
    print(f"Years: {int(df['year'].min())} to {int(df['year'].max())}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nSample of key variables:")
    key_vars = [c for c in ['msacode', 'year', 'income', 'average4q', 'permits',
                            'population', 'sales', 'h_census', 'cpi', 'price2000',
                            'cons_earning', 'cons_employment', 'rent', 'consdat'] if c in df.columns]
    print(df[key_vars].describe())
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

# ============================================================
# Demonstrate data preparation (matching varest1.do)
# ============================================================
print("\n" + "=" * 60)
print("Data preparation steps (replicating varest1.do variable construction)")
print("=" * 60)

df = df.sort_values(['msacode', 'year']).reset_index(drop=True)

# Compute housing stock (matching Stata code)
df['psum'] = df.groupby('msacode').apply(
    lambda g: (0.96 * 0.975 * g['permits']).cumsum()
).reset_index(level=0, drop=True)

# Year 2000 indicator (y21 = year==2000 dummy, since year dummies start from 1980)
df['y2000'] = (df['year'] == 2000).astype(float)

df['totp99'] = df.groupby('msacode').apply(
    lambda g: (g['psum'] * g['y2000']).sum()
).reindex(df['msacode']).values

if 'h_census' in df.columns:
    df['h2000'] = df.groupby('msacode')['h_census'].transform('sum')
    df['hstock'] = df['h2000'] - df['totp99'] + df['psum']

# Price normalization
if 'price2000' in df.columns:
    df['p2000'] = df.groupby('msacode').apply(
        lambda g: (g['price2000'] * 100000).sum()
    ).reindex(df['msacode']).values

if 'average4q' in df.columns:
    df['pi2000'] = df.groupby('msacode').apply(
        lambda g: (g['average4q'] * g['y2000']).sum()
    ).reindex(df['msacode']).values

    df['pop2000'] = df.groupby('msacode').apply(
        lambda g: (g['population'] * g['y2000']).sum()
    ).reindex(df['msacode']).values

# Variable construction
df['popgrow'] = df.groupby('msacode')['population'].shift(-1)
df['popgrow'] = np.log(df['popgrow'] / df['population'])

if 'sales' in df.columns:
    df['salesgrow'] = np.log(df.groupby('msacode')['sales'].shift(1)) - np.log(df.groupby('msacode')['sales'].shift(2))

if 'permits' in df.columns and 'hstock' in df.columns:
    df['conrate'] = df['permits'] / df['hstock']

df['wage'] = np.log(df['income'] / df['cpi'])
if 'average4q' in df.columns and 'p2000' in df.columns and 'pi2000' in df.columns:
    df['price'] = np.log(df['average4q'] * df['p2000'] / (df['pi2000'] * df['cpi']))

print(f"\nConstructed variables available: wage, price, conrate, popgrow, salesgrow")
print(f"Sample after dropping year >= 2008: {(df['year'] < 2008).sum()} observations")

# ============================================================
# NOTE: The actual estimation cannot proceed without:
# 1. Stata's `pvar` command (system-GMM panel VAR)
# 2. Dynare + Matlab for DSGE model simulation
# ============================================================
print("\n" + "=" * 60)
print("REPLICATION NOT POSSIBLE")
print("=" * 60)
print("""
The paper's main results require:

1. Panel VAR estimation (system-GMM) using Inessa Love's `pvar` command
   - 4 panel VARs in the original 2014 code (main, cwage, clabour, rent)
   - 2 additional panel VARs in the 2016 update (growth rates, LSDV)
   - No Python package implements this exact procedure

2. Dynare DSGE model simulations
   - 17 .mod files across Estimation and Simulation folders
   - Requires Matlab + Dynare toolbox
   - Generates theoretical moments for Tables 1-2, 7-10

3. Matlab simulation scripts
   - 4 .m files that call Dynare and compute moments
   - Require Matlab runtime
""")

# ============================================================
# Write empty replication.csv (header only)
# ============================================================
header = [
    'paper_id', 'reg_id', 'outcome_var', 'treatment_var', 'coefficient',
    'std_error', 'p_value', 'ci_lower', 'ci_upper', 'n_obs', 'r_squared',
    'original_coefficient', 'original_std_error', 'match_status',
    'coefficient_vector_json', 'fixed_effects', 'controls_desc',
    'cluster_var', 'estimator', 'sample_desc', 'notes'
]

with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

print(f"\nWrote empty replication.csv to {OUTPUT_CSV}")
print("Done.")
