"""
Preprocessing script for 113517-V1
"The Relative Power of Employment-to-Employment Reallocation
and Unemployment Exits in Predicting Wage Growth"
Moscarini & Postel-Vinay, AER P&P 2017

Translates the data-preparation section of Table_1_regressions.do
from Stata to Python, then saves preprocessed.parquet for the
replication script.

Input:  ee_wage_pan_all_pans.dta  (~3 GB Stata file, ~6M rows)
Output: preprocessed.parquet
"""

import pandas as pd
import numpy as np
import time
import os

DATA_DIR = 'data/downloads/extracted/113517-V1/Codes-and-data'
DTA_PATH = os.path.join(DATA_DIR, 'ee_wage_pan_all_pans.dta')
OUT_PATH = os.path.join(DATA_DIR, 'preprocessed.parquet')

# ============================================================
# Columns needed by the replication script
# ============================================================
# Only read the columns actually used, to reduce memory.
KEEP_COLS = [
    'panel_id', 'year_month',
    # Dependent variables (log earnings/wages)
    'logern_nom', 'logern', 'loghwr_nom', 'loghwr',
    # Transition indicators
    'eetrans_i', 'uetrans_i', 'netrans_i', 'eutrans_i', 'entrans_i',
    # Variables that get lagged
    'clw', 'siz', 'ind', 'occ', 'emp', 'unm', 'uni', 'married', 'state', 'phr',
    # Other needed variables
    'hrs',        # for loghrs
    'wgt',        # weights
    'sex', 'race', 'education', 'age',  # for mkt / agegroup
    'dicey',      # for filtering
]

t0 = time.time()

# ============================================================
# 1. Load .dta (only needed columns)
# ============================================================
print(f"Loading {DTA_PATH} ...")
print(f"  Reading only {len(KEEP_COLS)} columns to save memory.")
df = pd.read_stata(DTA_PATH, columns=KEEP_COLS, convert_categoricals=False)
print(f"  Loaded in {time.time()-t0:.1f}s, shape: {df.shape}")
print(f"  Memory: {df.memory_usage(deep=True).sum()/1e9:.2f} GB")

# ============================================================
# 2. Create loghrs  (Stata: gen loghrs = log(hrs))
# ============================================================
df['loghrs'] = np.log(df['hrs'])

# ============================================================
# 3. Create lagged variables
#    Stata: bys panel_id (year_month): gen lag`var' = `var'[_n-1]
#    For each variable in log* clw siz ind occ emp unm uni married state phr
# ============================================================
print("Creating lagged variables...")
t1 = time.time()

# Sort by panel_id and year_month (matches Stata bys panel_id (year_month))
df.sort_values(['panel_id', 'year_month'], inplace=True)

# Variables to lag: all log* columns present + clw siz ind occ emp unm uni married state phr
log_cols = [c for c in df.columns if c.startswith('log')]
other_lag_cols = ['clw', 'siz', 'ind', 'occ', 'emp', 'unm', 'uni', 'married', 'state', 'phr']
lag_vars = log_cols + other_lag_cols

# Group by panel_id and shift
# NOTE: Stata assigns var[_n-1] even across gaps (just the previous row
# within the same panel_id after sorting by year_month). pandas groupby
# shift(1) does exactly this.
grouped = df.groupby('panel_id', sort=False)
for var in lag_vars:
    df[f'lag{var}'] = grouped[var].shift(1)

print(f"  Created {len(lag_vars)} lagged variables in {time.time()-t1:.1f}s")

# ============================================================
# 4. Recode laguni: 2 -> 0   (Stata: recode laguni (2=0))
# ============================================================
df['laguni'] = df['laguni'].replace(2, 0)

# ============================================================
# 5. Create lagpub  (Stata: gen lagpub = inrange(lagclw, 3, 5))
#    lagclw in {3, 4, 5} => public sector worker
# ============================================================
df['lagpub'] = ((df['lagclw'] >= 3) & (df['lagclw'] <= 5)).astype('int8')

# ============================================================
# 6. Create agegroup
#    Stata:
#      gen agegroup = (age<=25) + 2*inrange(age,26,35) + 3*inrange(age,36,45)
#                   + 4*inrange(age,46,60) + 5*(age>60)
# ============================================================
age = df['age']
df['agegroup'] = (
    1 * (age <= 25) +
    2 * ((age >= 26) & (age <= 35)) +
    3 * ((age >= 36) & (age <= 45)) +
    4 * ((age >= 46) & (age <= 60)) +
    5 * (age > 60)
).astype('int8')

# ============================================================
# 7. Keep only dicey == 0   (Stata: keep if dicey==0)
# ============================================================
n_before = len(df)
df = df[df['dicey'] == 0].copy()
print(f"  Dropped dicey!=0: {n_before:,} -> {len(df):,} rows")
df.drop(columns=['dicey'], inplace=True)

# ============================================================
# 8. Create mkt = group(sex, race, agegroup, education)
#    and   mkt_t = group(mkt, year_month)
#    Stata: egen mkt = group($mktvars)
#           egen mkt_t = group(mkt year_month)
# ============================================================
print("Creating market (mkt) and market-time (mkt_t) indicators...")
t1 = time.time()

# egen group() in Stata assigns consecutive integers starting from 1
# Use pandas factorize on tuples or ngroup
df['mkt'] = df.groupby(['sex', 'race', 'agegroup', 'education'], sort=True).ngroup() + 1
df['mkt_t'] = df.groupby(['mkt', 'year_month'], sort=True).ngroup() + 1

print(f"  {df['mkt'].nunique()} markets, {df['mkt_t'].nunique()} market-time cells "
      f"in {time.time()-t1:.1f}s")

# ============================================================
# 9. Eligibility variables
#    Stata:
#      gen EZeligible = lagemp > 0
#      gen UZeligible = lagunm > 0
#      gen NZeligible = lagemp==0 & lagunm==0
#      gen UReligible = lagemp>0 | lagunm>0
#      gen DWeligible = lagemp>0 & emp>0
# ============================================================
df['EZeligible'] = (df['lagemp'] > 0).astype('int8')
df['UZeligible'] = (df['lagunm'] > 0).astype('int8')
df['NZeligible'] = ((df['lagemp'] == 0) & (df['lagunm'] == 0)).astype('int8')
df['UReligible'] = ((df['lagemp'] > 0) | (df['lagunm'] > 0)).astype('int8')
df['DWeligible'] = ((df['lagemp'] > 0) & (df['emp'] > 0)).astype('int8')

# ============================================================
# 10. Create year_month_num (numeric for regressions)
#     Stata uses year_month as a numeric (monthly date).
#     We convert the datetime to a numeric period index.
# ============================================================
# Convert datetime to a monthly period number (months since some epoch)
ym_min = df['year_month'].min()
df['year_month_num'] = (
    (df['year_month'].dt.year - ym_min.year) * 12 +
    (df['year_month'].dt.month - ym_min.month) + 1
)

# ============================================================
# 11. Drop intermediate columns not needed by replication
# ============================================================
# Keep only what the replication script actually uses
drop_cols = ['hrs', 'age', 'lagclw', 'lagmarried']
# Also drop raw lag columns not used in the replication
# The replication uses: lagstate, laguni, lagsiz, lagocc, lagind, lagpub,
#   lagphr, laglogern_nom, laglogern, lagloghwr_nom, lagloghwr,
#   lagemp, lagunm, loghrs (created but not used — keep anyway)
# Not used: lagloghrs, laglogswr, laglogswr_nom, laglogwr, laglogwr_nom,
#   lagsiz (actually IS used), lagmarried, lagclw (replaced by lagpub)
unused_lag = ['lagloghrs', 'laglogswr', 'laglogswr_nom', 'laglogwr', 'laglogwr_nom']
drop_cols += unused_lag

# Only drop columns that actually exist
drop_cols = [c for c in drop_cols if c in df.columns]
df.drop(columns=drop_cols, inplace=True)

# ============================================================
# 12. Save to parquet
# ============================================================
print(f"\nFinal shape: {df.shape}")
print(f"  Memory: {df.memory_usage(deep=True).sum()/1e9:.2f} GB")
print(f"Saving to {OUT_PATH} ...")
df.to_parquet(OUT_PATH, index=False, engine='pyarrow')
print(f"  Saved. File size: {os.path.getsize(OUT_PATH)/1e6:.1f} MB")

print(f"\nTotal time: {time.time()-t0:.1f}s")
print("Done.")
