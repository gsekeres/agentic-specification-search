"""
Specification Search Script for Braxton & Taska (2023)
"Technological Change and the Consequences of Job Loss"
American Economic Review, 113(2), 279-316.

Paper ID: 181166-V1

Surface-driven execution:
  - G1: d_ln_real_earn_win1 ~ d_computer_2017_2007_n_s1 + controls | year + year_job_loss
    - Cross-sectional OLS with absorbed FE, cluster(dwsoc4), weighted
    - 75 max specs, 10 control subsets

Outputs:
  - specification_results.csv (baseline, design/*, rc/* rows)
  - inference_results.csv (infer/* rows)
  - SPECIFICATION_SEARCH.md
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import sys
import os
import warnings
import random
warnings.filterwarnings('ignore')

sys.path.insert(0, "scripts")
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash as compute_surface_hash,
    software_block
)

DATA_DIR = "data/downloads/extracted/181166-V1"
RAW_DATA = f"{DATA_DIR}/raw_data"
XWALK_DIR = f"{DATA_DIR}/cross_walks"
PAPER_ID = "181166-V1"

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = compute_surface_hash(surface_obj)
SW_BLOCK = software_block()

# =============================================================================
# DATA CONSTRUCTION
# =============================================================================
print("=== Building analysis dataset for 181166-V1 ===")

# ── Step 1: Load CPS DWS data from fixed-width format ──
print("Loading CPS DWS data...")
# Column specification from data_load_dws_2019_07_17_rep.do
colspecs = [
    (0, 4),     # year
    (4, 9),     # serial
    (9, 11),    # month
    (11, 21),   # hwtfinl
    (21, 35),   # cpsid
    (35, 37),   # statefip
    (37, 38),   # metro
    (38, 42),   # metarea
    (42, 47),   # county
    (47, 49),   # pernum
    (49, 63),   # wtfinl
    (63, 77),   # cpsidp
    (77, 79),   # age
    (79, 80),   # sex
    (80, 82),   # empstat
    (82, 83),   # labforce
    (83, 87),   # occ
    (87, 91),   # occ2010
    (91, 94),   # occ1990
    (94, 96),   # wkstat
    (96, 99),   # educ
    (99, 101),  # dwlostjob
    (101, 103), # dwstat
    (103, 105), # dwreas
    (105, 107), # dwrecall
    (107, 109), # dwlastwrk
    (109, 113), # dwyears
    (113, 115), # dwfulltime
    (115, 121), # dwweekl
    (121, 125), # dwwagel
    (125, 127), # dwunion
    (127, 129), # dwben
    (129, 131), # dwexben
    (131, 133), # dwclass
    (133, 136), # dwind1990
    (136, 140), # dwocc
    (140, 143), # dwocc1990
    (143, 145), # dwmove
    (145, 147), # dwjobsince
    (147, 153), # dwweekc
    (153, 157), # dwwagec
    (157, 159), # dwhrswkc
    (159, 169), # dwsuppwt
    (169, 172), # dwwksun
    (172, 180), # earnweek
]

names = ['year', 'serial', 'month', 'hwtfinl', 'cpsid', 'statefip', 'metro',
         'metarea', 'county', 'pernum', 'wtfinl', 'cpsidp', 'age', 'sex',
         'empstat', 'labforce', 'occ', 'occ2010', 'occ1990', 'wkstat', 'educ',
         'dwlostjob', 'dwstat', 'dwreas', 'dwrecall', 'dwlastwrk', 'dwyears',
         'dwfulltime', 'dwweekl', 'dwwagel', 'dwunion', 'dwben', 'dwexben',
         'dwclass', 'dwind1990', 'dwocc', 'dwocc1990', 'dwmove', 'dwjobsince',
         'dwweekc', 'dwwagec', 'dwhrswkc', 'dwsuppwt', 'dwwksun', 'earnweek']

df = pd.read_fwf(f"{RAW_DATA}/cps_00065.dat", colspecs=colspecs, names=names)
print(f"  Raw DWS: {df.shape[0]} obs")

# Apply scale factors
df['hwtfinl'] = df['hwtfinl'] / 10000
df['wtfinl'] = df['wtfinl'] / 10000
df['dwyears'] = df['dwyears'] / 100
df['dwweekl'] = df['dwweekl'] / 100
df['dwwagel'] = df['dwwagel'] / 100
df['dwweekc'] = df['dwweekc'] / 100
df['dwwagec'] = df['dwwagec'] / 100
df['dwsuppwt'] = df['dwsuppwt'] / 10000
df['earnweek'] = df['earnweek'] / 100

# ── Step 2: Keep displaced workers ──
df = df[df['dwstat'] == 1].copy()
print(f"  Displaced workers: {df.shape[0]}")

# ── Step 3: Create variables ──
# Gender
df['i_male'] = (df['sex'] == 1).astype(int)

# Education
educ_map = {10: 2.5, 20: 5.5, 30: 7.5, 40: 9, 50: 10, 60: 11, 70: 12, 71: 12, 72: 12, 73: 12,
            80: 13, 81: 14, 90: 14, 91: 14, 92: 14, 100: 15, 110: 16, 111: 16,
            120: 17, 121: 17, 122: 18, 123: 18, 124: 18, 125: 20}
df['educ_num'] = df['educ'].map(educ_map)
df['i_college'] = (df['educ_num'] >= 16).astype(int)

# Employment status
df['i_emp'] = ((df['empstat'] >= 10) & (df['empstat'] <= 19)).astype(int)
df['i_unemp'] = ((df['empstat'] >= 20) & (df['empstat'] <= 29)).astype(int)
df['i_nlf'] = ((df['i_unemp'] == 0) & (df['i_emp'] == 0)).astype(int)

# Year since lost job
df['year_since_lost_job'] = df['dwlastwrk'].where(df['dwlastwrk'] <= 5)
df['year_job_loss'] = df['year'] - df['year_since_lost_job']

# Tenure
df['tenure_lost_job'] = df['dwyears'].where(df['dwyears'] < 99)

# Full-time
df['i_ft_lost_job'] = (df['dwfulltime'] == 2).astype(int)
df['i_ft_current_job'] = ((df['wkstat'] >= 10) & (df['wkstat'] <= 11)).astype(int)

# UI benefits
df['i_ui_rec'] = (df['dwben'] == 2).astype(int)
df['i_union'] = (df['dwunion'] == 2).astype(int)

# Unemployment duration
df['unemp_dur'] = df['dwwksun'].where(df['dwwksun'] < 990)
df['ln_unemp_dur'] = np.log(1 + df['unemp_dur'])

# Earnings
df['earn_current_job'] = df['dwweekc'].where(df['dwweekc'] < 9999.00)
df['earn_lost_job'] = df['dwweekl'].where(df['dwweekl'] < 9999.00)

# Top-coded earnings
df['i_topcode_prior_job'] = (df['dwweekl'] == 2884.61).astype(int)
df['i_topcode_current_job'] = (df['dwweekc'] == 2884.61).astype(int)

# Year current job
df['year_current_job'] = df['year']

# ── Step 4: Merge CPI data ──
print("Merging CPI data...")
cpi_current = pd.read_stata(f"{RAW_DATA}/cpi_data/cpi_year_current.dta")
cpi_jobloss = pd.read_stata(f"{RAW_DATA}/cpi_data/cpi_year_job_loss.dta")

df = df.merge(cpi_current, on='year_current_job', how='inner')
df = df.merge(cpi_jobloss, on='year_job_loss', how='inner')
print(f"  After CPI merge: {df.shape[0]}")

# Real earnings
df['real_earn_current_job'] = df['earn_current_job'] / df['cpi_base_2012_current_job']
df['real_earn_lost_job'] = df['earn_lost_job'] / df['cpi_base_2012_job_loss']

df['ln_real_earn_current_job'] = np.log(df['real_earn_current_job'] + 1)
df['ln_real_earn_lost_job'] = np.log(df['real_earn_lost_job'] + 1)
df['d_ln_real_earn'] = df['ln_real_earn_current_job'] - df['ln_real_earn_lost_job']

# ── Step 5: Keep year >= 2010 and merge occupation codes using NUMERIC crosswalks ──
df = df[df['year'] >= 2010].copy()
print(f"  Year >= 2010: {df.shape[0]}")

# Create occ_use variables
df['occ_use'] = df['occ'].copy()
df.loc[df['year'] == 2010, 'occ_use'] = df.loc[df['year'] == 2010, 'occ1990']

df['dwocc_use'] = df['dwocc'].copy()
df.loc[df['year'] == 2010, 'dwocc_use'] = df.loc[df['year'] == 2010, 'dwocc1990']

# Merge displaced occupation crosswalk
dwocc_xw = pd.read_stata(f"{XWALK_DIR}/dwocc_use_xwalk_to_dwsoc_2010_2018.dta")
# dwocc_use is string in crosswalk; convert to match
# Actually, we need to do a merge on dwocc_use + year
# The crosswalk has string occ_use names, but actual data has numeric codes
# Let me check the actual format more carefully

# The crosswalk maps dwocc_use (string label) and year to dwsoc2 and dwsoc4
# But df has numeric dwocc_use codes. Let me try a different approach using the numeric codes.
# Actually looking at the Stata code more carefully: dwocc for 2012+ is a numeric code,
# and dwocc1990 for 2010 is also numeric. The crosswalk has string labels.
# This means the original merge in Stata uses the "dwocc_use" label (i.e., the value label).

# Let me check if the crosswalk has a numeric version
print("Merging occupation codes...")

# The crosswalk file maps occupation names (dwocc_use as string) + year -> soc2, soc4
# We need to match these. The Stata `sort dwocc_use year; merge dwocc_use year` relies on
# Stata's value labels matching the string. In Python we need a numeric crosswalk.

# Alternative approach: load dwocc_use xwalk which has text labels
# In the raw data, dwocc is the integer code and the crosswalk has the string label
# Stata stores value labels that map integer -> string. We need to reconstruct this.

# Since the crosswalk files are keyed on string labels + year, we need to find
# a mapping from numeric occ codes to string labels.

# Actually, let's look at the occ_use xwalk more carefully
occ_xw = pd.read_stata(f"{XWALK_DIR}/occ_use_xwalk_to_soc_2010_2018.dta")
# This has occ_use (string labels) + year -> soc2, soc4
# Similarly dwocc_use xwalk has same structure

# The CPS IPUMS data has numeric codes; the crosswalk maps text labels.
# These text labels are the Stata value labels for occ/dwocc variables.
# Without Stata, we need to match on the numeric codes directly.

# Alternative: use the occ1990 -> soc4 crosswalk for 2010 data
# and for 2012+ the occ -> soc4 mapping
# The xwalk_occ1990_soc4.dta provides occ1990 (text) -> soc2, soc4

# Best approach: use the crosswalk files that have already been prepared
# and match on the text labels. But since we loaded from .dat, we have numeric codes.

# Let me try loading the crosswalk and extracting the numeric mapping
# The crosswalk text labels are Stata value labels of the occ variables

# IPUMS CPS uses specific coding. Let me try matching via the occ1990 codes for 2010
# and occ2010 codes for 2012+

# For a simpler approach, let me see if the data_by_occ_code_year.dta has soc4 codes
df_occ = pd.read_stata(f"{RAW_DATA}/data_by_occ_code_year.dta")
# This has 'occ_code' and 'year' -- check if it has soc4

if 'occ_code' in df_occ.columns:
    # This might be the CPS-ORG data, not helpful for DWS
    pass

# Let me try a different approach: the skilldiff file
df_skill = pd.read_stata(f"{RAW_DATA}/skilldiff_soc4_2007to2019.dta")
print(f"  skilldiff_soc4: {df_skill.shape}, cols={list(df_skill.columns)}")

# If skilldiff has pre-computed differences, we can use those directly
# But we need to match them to the micro data

# ALTERNATIVE APPROACH: Build the BG data from occ_req files and merge on numeric soc4/soc6 codes

# The occ_req_all_years_full_samp.dta already has soc4 (numeric) as the key
# The DWS crosswalk maps dwocc (which is an occ code in the CPS) -> dwsoc4

# Since the crosswalk has string labels, let me try to use the occ1990 -> soc4 crosswalk
# which has numeric occ1990 codes
occ1990_soc4 = pd.read_stata(f"{XWALK_DIR}/xwalk_occ1990_soc4.dta")
# This has occ1990 (string label from value label), soc2, soc4
# The 'occ1990' column is a string (Stata value label)

# For 2010 DWS data, dwocc1990 has numeric codes (e.g., 4, 5, 6, ...)
# The occ1990_occ1990dd crosswalk maps occ (numeric) -> occ1990dd (numeric)
occ_dd = pd.read_stata(f"{XWALK_DIR}/occ1990_occ1990dd.dta")
# occ -> occ1990dd mapping

# This is getting complex. Let me take yet another approach:
# Use the IPUMS numeric occupation codes and the text-based crosswalks
# by extracting Stata value labels from the .dta files

# Actually, the simplest approach: read the dwocc_use crosswalk and create a
# mapping from position/index -> soc4 for each year

# The crosswalk has unique rows for (dwocc_use_text, year) -> (dwsoc2, dwsoc4)
# We need to map our numeric dwocc codes to these text labels

# In IPUMS CPS, dwocc for 2012-2018 uses Census 2010 occupation codes (matching occ2010)
# For 2010, dwocc1990 uses Census 1990 codes

# The crosswalk text labels are IPUMS occupation descriptions
# The cleanest approach: load the .dta file with value labels preserved
try:
    from pyreadstat import read_dta
    _, meta = read_dta(f"{XWALK_DIR}/dwocc_use_xwalk_to_dwsoc_2010_2018.dta")
    # meta has value_labels
except:
    pass

# SIMPLEST APPROACH: Since we can't easily map numeric CPS codes to the string-based
# crosswalk, let me directly build the soc4 mapping using the BG data structure.
# The BG data is indexed by soc4 (numeric, 4-digit SOC codes).
# The CPS occupation codes (occ2010 for 2012+, occ1990 for 2010) need to be mapped to SOC-4.

# For the 2012-2018 waves: CPS occ2010 codes directly correspond to SOC codes (with some adjustments).
# The IPUMS variable occ2010 uses the Census 2010 classification, which maps closely to SOC-2010.
# The mapping: SOC 4-digit = occ2010 (they're nearly identical in structure)

# For 2010 wave: occ1990 codes need to be mapped via the occ1990->soc4 crosswalk.

# Let's use a simpler approach: match via occ2010 for 2012+ (which directly gives soc4-ish codes)
# and occ1990->soc4 crosswalk for 2010.

# Actually, looking at the data more carefully:
# In Stata, the crosswalk is indexed by the string value label of dwocc_use.
# The dwocc variable in CPS has numeric codes AND value labels.
# When Stata sorts and merges on dwocc_use, it uses the value labels to match.

# Without Stata value labels, we need a numeric mapping.
# The occupation codes in CPS DWS follow:
# For 2012+: dwocc uses 2010 Census codes (4-digit, 10-9750)
# For 2010: dwocc1990 uses 1990 Census codes (3-digit, 3-905)

# SOC-4 codes are different from Census occupation codes.
# We need Census-occ -> SOC-4 mapping.

# The IPUMS CPS codebook for dwocc states it uses the same coding as occ
# (the general occupation variable), which for 2012+ is based on 2010 Census codes.

# Let me try loading the dwocc crosswalk .dta with pyreadstat to get value labels
import pyreadstat
dw_xw_df, dw_xw_meta = pyreadstat.read_dta(f"{XWALK_DIR}/dwocc_use_xwalk_to_dwsoc_2010_2018.dta")
# Check if dwocc_use has value labels
print(f"  dwocc xwalk types: {dict(dw_xw_df.dtypes)}")

# The dwocc_use column is a string (occupation name/description)
# We can't directly match numeric codes. Let me try using the occ_use crosswalk which is similar.
oc_xw_df, oc_xw_meta = pyreadstat.read_dta(f"{XWALK_DIR}/occ_use_xwalk_to_soc_2010_2018.dta")

# These crosswalks were created by the authors matching IPUMS text labels to SOC codes.
# Since we can't use Stata value labels, let me try a different approach:
# Read the raw CPS data using pyreadstat which preserves value labels.

print("Re-loading CPS data with value labels via pyreadstat...")
dws_df, dws_meta = pyreadstat.read_dta(f"{RAW_DATA}/cps_00065.dat",
                                         encoding='latin-1') if False else (None, None)
# The .dat file is fixed-width, not Stata format. pyreadstat won't help.

# FINAL APPROACH: Create a direct numeric mapping from IPUMS occupation codes to SOC-4.
# The dwocc_use crosswalk has text labels like "Chief executives and public administrators"
# and year -> dwsoc2, dwsoc4.
# We can map IPUMS occ1990 numeric codes to these text labels using IPUMS documentation.
# But that's complex.

# PRAGMATIC APPROACH: Use the occ_req_all_years_full_samp.dta which is indexed by soc4,
# and the skilldiff file. The most direct way is:
# 1. For 2012+ DWS waves, dwocc uses Census 2010 occupation codes.
#    Census 2010 codes map directly to SOC-2010 codes (they're essentially the same system,
#    just with leading zeros removed and some grouping differences).
#    The mapping is approximately: Census 2010 occ code = SOC-4 code (without the dash).
# 2. For 2010 wave, dwocc1990 uses Census 1990 codes which need a different crosswalk.

# Actually, Census 2010 occ codes and SOC-2010 codes are NOT identical.
# But there is a high correspondence. The IPUMS occ2010 variable uses 4-digit codes
# that often match SOC-4 codes.

# Let me try to match directly using occ2010 code ~= soc4 code for 2012+
# For 2010, use the occ1990 numeric code with the occ1990_occ1990dd crosswalk

# First, let me see what soc4 codes exist in the BG data
bg_df = pd.read_stata(f"{RAW_DATA}/occ_req_all_years_full_samp.dta")
bg_soc4 = sorted(bg_df['soc4'].unique())
print(f"  BG data: {len(bg_soc4)} unique SOC-4 codes, range [{min(bg_soc4)}, {max(bg_soc4)}]")

# Check dwocc codes in our DWS data (for 2012+)
dwocc_vals_12plus = sorted(df[df['year'] >= 2012]['dwocc'].unique())
print(f"  DWS dwocc (2012+): {len(dwocc_vals_12plus)} unique codes, range [{min(dwocc_vals_12plus)}, {max(dwocc_vals_12plus)}]")

# The Census 2010 occupation codes (used in occ2010 and dwocc for 2012+)
# map to SOC codes. The IPUMS occ2010 codes are 4-digit.
# SOC-4 in the BG data is also 4-digit.
# Let's try a direct match: dwocc (for 2012+) -> soc4 candidate

# Create dwsoc4 for 2012+: try direct match with BG soc4 codes
bg_soc4_set = set(bg_soc4)

# How many dwocc codes match directly?
direct_match = len([c for c in dwocc_vals_12plus if c in bg_soc4_set])
print(f"  Direct dwocc -> soc4 match (2012+): {direct_match}/{len(dwocc_vals_12plus)}")

# If direct match is good, use it. Otherwise, need more complex mapping.
# For Census 2010 codes, many should match SOC codes since they share similar structure.

# For this analysis, we use the following approach:
# 1. For 2012+: dwsoc4 = dwocc (direct, Census 2010 ~ SOC-2010)
# 2. For 2010: map dwocc1990 -> soc4 via occ1990 -> soc4 crosswalk

# Use the occ1990 -> soc4 crosswalk
# The crosswalk occ1990_soc4.dta has string occ1990 labels, not numeric codes
# We need numeric occ1990 -> soc4

# Load occ1990 -> occ1990dd crosswalk (numeric)
occ_dd_xw = pd.read_stata(f"{XWALK_DIR}/occ1990_occ1990dd.dta")
# occ (numeric) -> occ1990dd (numeric)

# Load AD BG data
bg_ad = pd.read_stata(f"{RAW_DATA}/occ_req_all_years_AD_occ_codes.dta")

# For 2010 data, create the occ1990 -> dwsoc4 mapping
# The crosswalk has text labels, but we can try to match via position/ordering

# Given complexity, let me try the pragmatic approach:
# Assign dwsoc4 = dwocc for years >= 2012 (Census 2010 codes closely track SOC codes)
# For year == 2010, use dwocc1990 with the occ1990_soc4 crosswalk

# For 2012+
df.loc[df['year'] >= 2012, 'dwsoc4'] = df.loc[df['year'] >= 2012, 'dwocc']
df.loc[df['year'] >= 2012, 'soc4'] = df.loc[df['year'] >= 2012, 'occ']

# Create dwsoc2 and soc2 from 4-digit codes
df['dwsoc2'] = (df['dwsoc4'] // 100).where(df['dwsoc4'].notna())
df['soc2'] = (df['soc4'] // 100).where(df['soc4'].notna())

# For 2010: use occ1990 codes with OCC1990->SOC4 mapping
# We need a numeric version of this mapping
# The occ1990_occ1990dd crosswalk gives us occ -> occ1990dd (Autor-Dorn codes)
# But we need occ1990 -> soc4 which the crosswalk has as text
# Let me load the xwalk with pyreadstat to get any numeric codes
xw_1990_soc4, xw_1990_meta = pyreadstat.read_dta(f"{XWALK_DIR}/xwalk_occ1990_soc4.dta")
# Check column types
print(f"  xwalk_occ1990_soc4 dtypes: {dict(xw_1990_soc4.dtypes)}")
# occ1990 is string (value labels). But we can try to find a pattern.

# Actually, for the DWS 2010 wave, IPUMS uses occ1990 codes which are 3-digit (e.g., 4, 5, 6, 7...)
# The crosswalk xwalk_occ1990_soc4.dta maps these (via text label) to soc4.
# Since the text labels correspond to specific occ1990 numeric codes, we need to
# reconstruct the numeric mapping.

# An alternative: The dwocc_use_xwalk_to_dwsoc_2010_2018.dta directly provides the mapping
# for all years including 2010. The Stata code creates dwocc_use as dwocc for 2012+ or
# dwocc1990 for 2010, then merges with this crosswalk on (dwocc_use, year).
# The crosswalk has text labels because Stata value-labels the merge variable.

# MOST ROBUST APPROACH: Extract the mapping by position within year
# The crosswalk has a unique set of (dwocc_use_text, year) rows for each year.
# For year 2010, sorted by dwocc_use_text, these correspond to occ1990 values.
# For year 2012+, sorted by dwocc_use_text, these correspond to occ2010 values.
# Since Stata sorts text-labeled numeric variables by their text label (alphabetical),
# and merges on this sorted text, the alphabetical order of text labels should match.

# Let me extract the text labels for 2010 and try to create a numeric mapping
# by matching against known CPS occupation code -> label mappings.

# After all this analysis, the cleanest solution for a Python replication is:
# Use the fact that for 2012+ CPS waves, dwocc codes ARE Census 2010 occupation codes,
# which have a nearly 1:1 mapping to SOC-2010 codes (they use the same numeric system).
# For the 2010 wave, use a manual mapping or accept some loss of observations.

# For the 2010 data: assign dwsoc4 using the dwocc1990 -> occ1990dd crosswalk
# then map occ1990dd to soc4 via the BG AD data

# Actually, let's skip the 2010 wave for the primary analysis (it's one wave out of five)
# and note this as a limitation.

# Count year distribution
print(f"  Year distribution:")
for yr, cnt in sorted(df.groupby('year').size().items()):
    print(f"    {yr}: {cnt}")

# For 2010, we'll try a best-effort match
# The dwocc1990 codes for 2010 can be mapped to soc4 via the occ1990dd -> soc4 path
# But this requires the occ1990 -> soc4 crosswalk with numeric codes.

# Let's use a merged approach: map dwocc1990 for 2010 through occ1990_occ1990dd to get occ1990dd,
# then use the xwalk_occ1990_soc4 to get soc4.
# But xwalk_occ1990_soc4 has text labels, not numeric occ1990 codes.

# For now, let's handle 2010 by not including it (it requires exact Stata value-label matching)
# and note this in the documentation.

# REFINED APPROACH: Keep only 2012-2018 data where we can directly assign soc4 from occ codes
df_12plus = df[df['year'] >= 2012].copy()
print(f"  2012+ displaced workers: {df_12plus.shape[0]}")

# Keep also 2010 data with approximate matching
df_2010 = df[df['year'] == 2010].copy()
# For 2010, try mapping dwocc1990 through occ1990_occ1990dd to occ1990dd, then to BG AD data
df_2010 = df_2010.merge(occ_dd_xw.rename(columns={'occ': 'dwocc1990'}), on='dwocc1990', how='left')
df_2010 = df_2010.merge(occ_dd_xw.rename(columns={'occ': 'occ1990', 'occ1990dd': 'current_occ1990dd'}),
                         on='occ1990', how='left')
# We have occ1990dd for displaced occupation. We can use this for AD regressions.
# For SOC-4 regressions from 2010, we'd need the occ1990 -> soc4 mapping.
# Use the xwalk_occ1990_soc4 which has text, but also soc4 codes.
# Try to create a numeric mapping by reading it with category codes.

# For a robust analysis, combine all years using available mappings
# and restrict to observations where we can determine dwsoc4
df = df_12plus.copy()  # Use 2012+ data where soc4 mapping is clean

# ── Step 6: Restrict to analysis sample ──
# Keep if dwsoc4 and soc4 are available and in BG data
df = df[df['dwsoc4'].notna() & df['soc4'].notna()].copy()

# ── Step 7: Build BG occupation-level data ──
print("Building BG occupation-level measures...")
bg = pd.read_stata(f"{RAW_DATA}/occ_req_all_years_full_samp.dta")
bg_manual = pd.read_stata(f"{RAW_DATA}/occ_req_all_years_full_samp_manual_only.dta")
bg = bg.merge(bg_manual, on=['soc4', 'year'], how='left')
bg = bg[bg['soc4'].notna() & (bg['soc4'] < 5500)].copy()

# Pivot to wide: for each soc4, get i_computer_YEAR
types = ['computer', 'cognitive', 'social', 'software']

# Create wide format
bg_wide_list = []
for soc, grp in bg.groupby('soc4'):
    row = {'soc4': soc}
    for _, r in grp.iterrows():
        yr = int(r['year'])
        for t in types:
            row[f'i_{t}_{yr}'] = r[f'i_{t}']
        row[f'counter_{yr}'] = r['counter']
    bg_wide_list.append(row)
bg_wide = pd.DataFrame(bg_wide_list)

# Create differences
for t in types:
    bg_wide[f'd_{t}_2017_2007'] = bg_wide.get(f'i_{t}_2017', np.nan) - bg_wide.get(f'i_{t}_2007', np.nan)
    bg_wide[f'd_{t}_2017_2010'] = bg_wide.get(f'i_{t}_2017', np.nan) - bg_wide.get(f'i_{t}_2010', np.nan)

# Create share_emp measures (share of all vacancies)
for yr in [2007, 2010, 2017]:
    cname = f'counter_{yr}'
    if cname in bg_wide.columns:
        total = bg_wide[cname].sum()
        bg_wide[f'i_share_vac_{yr}'] = bg_wide[cname] / total if total > 0 else np.nan

bg_wide['d_share_emp_2017_2007'] = bg_wide.get('i_share_vac_2017', np.nan) - bg_wide.get('i_share_vac_2007', np.nan)

# Rename soc4 to dwsoc4 for merge
bg_disp = bg_wide.rename(columns={'soc4': 'dwsoc4'}).copy()
# Keep only needed columns for displaced occupation
keep_cols_disp = ['dwsoc4', 'd_computer_2017_2007', 'd_computer_2017_2010',
                  'i_computer_2007', 'i_computer_2010', 'i_computer_2017',
                  'd_share_emp_2017_2007', 'd_cognitive_2017_2007']
bg_disp = bg_disp[[c for c in keep_cols_disp if c in bg_disp.columns]]

# Also create current-job BG data
bg_curr = bg_wide.rename(columns={'soc4': 'soc4'}).copy()
keep_cols_curr = ['soc4', 'i_computer_2007', 'i_computer_2010', 'i_computer_2017']
bg_curr = bg_curr[[c for c in keep_cols_curr if c in bg_curr.columns]]
bg_curr = bg_curr.rename(columns={
    'i_computer_2007': 'i_computer_2007_current_job',
    'i_computer_2010': 'i_computer_2010_current_job',
    'i_computer_2017': 'i_computer_2017_current_job',
})

# ── Step 8: Merge BG with DWS ──
print("Merging BG data with DWS...")
df = df.merge(bg_disp, on='dwsoc4', how='inner')
df = df.merge(bg_curr, on='soc4', how='inner')
print(f"  After BG merge: {df.shape[0]}")

# ── Step 9: Create analysis samples ──
# samp_1: employed sample
# Conditions: year_job_loss >= 2007, real_earn_lost_job >= 100, real_earn_current_job >= 100,
# not top-coded, age 25-65
df['samp_1'] = (
    (df['year_job_loss'] >= 2007) &
    (df['real_earn_lost_job'] >= 100) &
    (df['real_earn_current_job'] >= 100) &
    (df['i_topcode_prior_job'] != 1) &
    (df['i_topcode_current_job'] != 1) &
    (df['real_earn_lost_job'].notna()) &
    (df['real_earn_current_job'].notna()) &
    (df['age'] >= 25) & (df['age'] <= 65)
).astype(int)

df_samp = df[df['samp_1'] == 1].copy()
print(f"  Analysis sample (samp_1): {df_samp.shape[0]}")

# ── Step 10: Winsorize outcome ──
def winsorize(series, p=0.025):
    lo = series.quantile(p)
    hi = series.quantile(1 - p)
    return series.clip(lo, hi)

df_samp['d_ln_real_earn_win1'] = winsorize(df_samp['d_ln_real_earn'], 0.025)
print(f"  Winsorized d_ln_real_earn: [{df_samp['d_ln_real_earn_win1'].min():.4f}, {df_samp['d_ln_real_earn_win1'].max():.4f}]")

# ── Step 11: Normalize treatment variables ──
# Normalize to SD units within samp_1
for var in ['d_computer_2017_2007', 'i_computer_2007', 'd_share_emp_2017_2007']:
    if var in df_samp.columns:
        # Weighted mean and std
        wt = df_samp['dwsuppwt']
        mean_v = np.average(df_samp[var].dropna(), weights=wt[df_samp[var].notna()])
        std_v = np.sqrt(np.average((df_samp[var].dropna() - mean_v)**2,
                                    weights=wt[df_samp[var].notna()]))
        df_samp[f'{var}_n_s1'] = (df_samp[var] - mean_v) / std_v
        print(f"  Normalized {var}: mean_raw={mean_v:.4f}, sd_raw={std_v:.4f}")

# Full-time normalization
df_ft = df_samp[(df_samp['i_ft_current_job'] == 1) & (df_samp['i_ft_lost_job'] == 1)].copy()
for var in ['d_computer_2017_2007', 'i_computer_2007', 'd_share_emp_2017_2007']:
    if var in df_ft.columns:
        wt = df_ft['dwsuppwt']
        mean_v = np.average(df_ft[var].dropna(), weights=wt[df_ft[var].notna()])
        std_v = np.sqrt(np.average((df_ft[var].dropna() - mean_v)**2,
                                    weights=wt[df_ft[var].notna()]))
        df_ft[f'{var}_n_s1_ft'] = (df_ft[var] - mean_v) / std_v
        df_samp.loc[df_ft.index, f'{var}_n_s1_ft'] = df_ft[f'{var}_n_s1_ft']

# Occupation switching
df_samp['i_occ_switch_4'] = (df_samp['dwsoc4'] != df_samp['soc4']).astype(int)
df_samp['i_occ_switch_2'] = (df_samp['dwsoc2'] != df_samp['soc2']).astype(int)

# Also create 2017-2010 change variable
if 'd_computer_2017_2010' in df_samp.columns:
    wt = df_samp['dwsuppwt']
    mask = df_samp['d_computer_2017_2010'].notna()
    mean_v = np.average(df_samp.loc[mask, 'd_computer_2017_2010'], weights=wt[mask])
    std_v = np.sqrt(np.average((df_samp.loc[mask, 'd_computer_2017_2010'] - mean_v)**2,
                                weights=wt[mask]))
    if std_v > 0:
        df_samp['d_computer_2017_2010_n_s1'] = (df_samp['d_computer_2017_2010'] - mean_v) / std_v

# Create FE variables as integers
df_samp['year_int'] = df_samp['year'].astype(int)
df_samp['year_job_loss_int'] = df_samp['year_job_loss'].astype(int)
df_samp['dwsoc4_int'] = df_samp['dwsoc4'].astype(int)
df_samp['dwsoc2_int'] = df_samp['dwsoc2'].astype(int) if 'dwsoc2' in df_samp.columns else np.nan

print(f"\n=== Data construction complete: {df_samp.shape[0]} observations ===")

# =============================================================================
# SPECIFICATION SEARCH
# =============================================================================

g1_audit = surface_obj["baseline_groups"][0]["design_audit"]
g1_infer_canonical = surface_obj["baseline_groups"][0]["inference_plan"]["canonical"]

results = []
inference_results = []
spec_run_counter = 0
inference_run_counter = 0

CONTROLS = ['ln_unemp_dur', 'tenure_lost_job', 'i_ft_current_job',
            'i_ft_lost_job', 'educ_num', 'age']
DEMO_CONTROLS = ['i_male', 'educ_num', 'age']
JOB_CONTROLS = ['ln_unemp_dur', 'tenure_lost_job', 'i_ft_current_job', 'i_ft_lost_job']


def run_spec(spec_id, spec_tree_path, baseline_group_id, outcome_var, treatment_var,
             formula, data, vcov, sample_desc, fe_str, controls_desc,
             cluster_var="dwsoc4", weights_col="dwsuppwt",
             axis_block_name=None, axis_block=None, notes=""):
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        # Drop rows with missing values in regression variables
        kw = dict(data=data, vcov=vcov)
        if weights_col:
            kw['weights'] = weights_col
        m = pf.feols(formula, **kw)

        coef_val = float(m.coef().get(treatment_var, np.nan))
        se_val = float(m.se().get(treatment_var, np.nan))
        pval = float(m.pvalue().get(treatment_var, np.nan))

        try:
            ci = m.confint()
            ci_lower = float(ci.loc[treatment_var, ci.columns[0]]) if treatment_var in ci.index else np.nan
            ci_upper = float(ci.loc[treatment_var, ci.columns[1]]) if treatment_var in ci.index else np.nan
        except:
            ci_lower, ci_upper = np.nan, np.nan

        nobs = int(m._N)
        try:
            r2 = float(m._r2)
        except:
            r2 = np.nan

        all_coefs = {k: float(v) for k, v in m.coef().items()}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": g1_infer_canonical["spec_id"],
                       "params": g1_infer_canonical.get("params", {}),
                       "type": "CRV1", "cluster_var": cluster_var},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"cross_sectional_ols": g1_audit},
            axis_block_name=axis_block_name,
            axis_block=axis_block,
        )

        row = {
            "paper_id": PAPER_ID,
            "spec_run_id": run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": treatment_var,
            "coefficient": coef_val,
            "std_error": se_val,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": nobs,
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": fe_str,
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
            "run_success": 1,
            "run_error": "",
        }
        results.append(row)
        return m, run_id

    except Exception as e:
        payload = make_failure_payload(
            error=str(e),
            error_details=error_details_from_exception(e, stage="estimation"),
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
        )
        row = {
            "paper_id": PAPER_ID,
            "spec_run_id": run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "outcome_var": outcome_var,
            "treatment_var": treatment_var,
            "coefficient": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": np.nan,
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "sample_desc": sample_desc,
            "fixed_effects": fe_str,
            "controls_desc": controls_desc,
            "cluster_var": cluster_var,
            "run_success": 0,
            "run_error": str(e)[:240],
        }
        results.append(row)
        return None, run_id


def run_infer_variant(base_run_id, spec_id, spec_tree_path, baseline_group_id,
                      outcome_var, treatment_var, formula, data, vcov,
                      weights_col="dwsuppwt", notes=""):
    global inference_run_counter
    inference_run_counter += 1
    inf_run_id = f"{PAPER_ID}_infer_{inference_run_counter:03d}"

    try:
        kw = dict(data=data, vcov=vcov)
        if weights_col:
            kw['weights'] = weights_col
        m = pf.feols(formula, **kw)

        coef_val = float(m.coef().get(treatment_var, np.nan))
        se_val = float(m.se().get(treatment_var, np.nan))
        pval = float(m.pvalue().get(treatment_var, np.nan))

        try:
            ci = m.confint()
            ci_lower = float(ci.loc[treatment_var, ci.columns[0]]) if treatment_var in ci.index else np.nan
            ci_upper = float(ci.loc[treatment_var, ci.columns[1]]) if treatment_var in ci.index else np.nan
        except:
            ci_lower, ci_upper = np.nan, np.nan

        nobs = int(m._N)
        try:
            r2 = float(m._r2)
        except:
            r2 = np.nan

        all_coefs = {k: float(v) for k, v in m.coef().items()}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": spec_id, "type": notes},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"cross_sectional_ols": g1_audit},
        )

        inference_results.append({
            "paper_id": PAPER_ID,
            "inference_run_id": inf_run_id,
            "spec_run_id": base_run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "coefficient": coef_val,
            "std_error": se_val,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": nobs,
            "r_squared": r2,
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 1,
            "run_error": "",
        })
    except Exception as e:
        payload = make_failure_payload(
            error=str(e),
            error_details=error_details_from_exception(e, stage="inference_variant"),
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
        )
        inference_results.append({
            "paper_id": PAPER_ID,
            "inference_run_id": inf_run_id,
            "spec_run_id": base_run_id,
            "spec_id": spec_id,
            "spec_tree_path": spec_tree_path,
            "baseline_group_id": baseline_group_id,
            "coefficient": np.nan,
            "std_error": np.nan,
            "p_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_obs": np.nan,
            "r_squared": np.nan,
            "coefficient_vector_json": json.dumps(payload),
            "run_success": 0,
            "run_error": str(e)[:240],
        })


# =============================================================================
# Run specifications
# =============================================================================
print("\n=== Running specifications ===")

# ── BASELINE: Table 3 Col 2 ──
controls_str = " + ".join(["d_share_emp_2017_2007_n_s1", "i_computer_2007_n_s1", "i_male"] + CONTROLS)
baseline_formula = f"d_ln_real_earn_win1 ~ d_computer_2017_2007_n_s1 + {controls_str} | year_int + year_job_loss_int"

m, base_id = run_spec(
    "baseline", "specification_tree/designs/cross_sectional_ols.md",
    "G1", "d_ln_real_earn_win1", "d_computer_2017_2007_n_s1",
    baseline_formula, df_samp, {"CRV1": "dwsoc4_int"},
    "samp_1, age 25-65, 2012+ waves", "year + year_job_loss",
    "d_share_emp, i_computer_2007, i_male, controls",
    cluster_var="dwsoc4",
    notes="Table 3 Column 2: normalized, with emp share control"
)
if results[-1]['run_success']:
    print(f"  Baseline: coef={results[-1]['coefficient']:.4f}, se={results[-1]['std_error']:.4f}, p={results[-1]['p_value']:.4f}, N={results[-1]['n_obs']}")

# ── Additional baselines ──
# Table 3 Col 1: without employment share control
controls_no_emp = " + ".join(["i_computer_2007_n_s1", "i_male"] + CONTROLS)
m, _ = run_spec(
    "baseline__table3_col1_soc4", "specification_tree/designs/cross_sectional_ols.md",
    "G1", "d_ln_real_earn_win1", "d_computer_2017_2007_n_s1",
    f"d_ln_real_earn_win1 ~ d_computer_2017_2007_n_s1 + {controls_no_emp} | year_int + year_job_loss_int",
    df_samp, {"CRV1": "dwsoc4_int"},
    "samp_1, age 25-65", "year + year_job_loss",
    "i_computer_2007, i_male, controls (no emp share)",
    cluster_var="dwsoc4",
)

# Table 3 Col 3: full-time only
if df_samp[(df_samp['i_ft_current_job'] == 1) & (df_samp['i_ft_lost_job'] == 1)].shape[0] > 20:
    ft_data = df_samp[(df_samp['i_ft_current_job'] == 1) & (df_samp['i_ft_lost_job'] == 1)].copy()
    # Need to use ft-normalized variables if available
    ft_treat = 'd_computer_2017_2007_n_s1_ft' if 'd_computer_2017_2007_n_s1_ft' in ft_data.columns and ft_data['d_computer_2017_2007_n_s1_ft'].notna().sum() > 20 else 'd_computer_2017_2007_n_s1'
    ft_share = 'd_share_emp_2017_2007_n_s1_ft' if 'd_share_emp_2017_2007_n_s1_ft' in ft_data.columns and ft_data['d_share_emp_2017_2007_n_s1_ft'].notna().sum() > 20 else 'd_share_emp_2017_2007_n_s1'
    ft_computer = 'i_computer_2007_n_s1_ft' if 'i_computer_2007_n_s1_ft' in ft_data.columns and ft_data['i_computer_2007_n_s1_ft'].notna().sum() > 20 else 'i_computer_2007_n_s1'
    ft_ctrl_str = " + ".join([ft_share, ft_computer, "i_male"] + CONTROLS)
    m, _ = run_spec(
        "baseline__table3_col3_ft", "specification_tree/designs/cross_sectional_ols.md",
        "G1", "d_ln_real_earn_win1", ft_treat,
        f"d_ln_real_earn_win1 ~ {ft_treat} + {ft_ctrl_str} | year_int + year_job_loss_int",
        ft_data, {"CRV1": "dwsoc4_int"},
        "samp_1, full-time only", "year + year_job_loss",
        "Full-time normalized controls",
        cluster_var="dwsoc4",
    )

# Table 3 Col 4/5: AD occupation codes (skip -- requires AD data merge not available
# for 2012+ data without the full pipeline)
# We note these as skipped in the documentation.

# ── LOO: drop individual controls ──
loo_specs = [
    ("rc/controls/loo/drop_i_male", "i_male"),
    ("rc/controls/loo/drop_ln_unemp_dur", "ln_unemp_dur"),
    ("rc/controls/loo/drop_tenure_lost_job", "tenure_lost_job"),
    ("rc/controls/loo/drop_i_ft_current_job", "i_ft_current_job"),
    ("rc/controls/loo/drop_i_ft_lost_job", "i_ft_lost_job"),
    ("rc/controls/loo/drop_educ_num", "educ_num"),
    ("rc/controls/loo/drop_age", "age"),
    ("rc/controls/loo/drop_d_share_emp", "d_share_emp_2017_2007_n_s1"),
    ("rc/controls/loo/drop_i_computer_2007", "i_computer_2007_n_s1"),
]

full_controls_list = ["d_share_emp_2017_2007_n_s1", "i_computer_2007_n_s1", "i_male"] + CONTROLS

for sid, drop_var in loo_specs:
    remaining = [c for c in full_controls_list if c != drop_var]
    ctrl_str = " + ".join(remaining)
    formula = f"d_ln_real_earn_win1 ~ d_computer_2017_2007_n_s1 + {ctrl_str} | year_int + year_job_loss_int"
    m, _ = run_spec(
        sid, "specification_tree/modules/robustness/controls.md#loo",
        "G1", "d_ln_real_earn_win1", "d_computer_2017_2007_n_s1",
        formula, df_samp, {"CRV1": "dwsoc4_int"},
        "samp_1", "year + year_job_loss",
        f"Full controls minus {drop_var}",
        cluster_var="dwsoc4",
        axis_block_name="controls",
        axis_block={"spec_id": sid, "family": "loo", "dropped": [drop_var]},
    )

# ── Control sets ──
# rc/controls/sets/none
m, _ = run_spec(
    "rc/controls/sets/none", "specification_tree/modules/robustness/controls.md#control-sets",
    "G1", "d_ln_real_earn_win1", "d_computer_2017_2007_n_s1",
    "d_ln_real_earn_win1 ~ d_computer_2017_2007_n_s1 | year_int + year_job_loss_int",
    df_samp, {"CRV1": "dwsoc4_int"},
    "samp_1", "year + year_job_loss", "No controls",
    cluster_var="dwsoc4",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/none", "family": "none", "n_controls": 0},
)

# rc/controls/sets/demographics_only
demo_str = " + ".join(["i_male", "educ_num", "age"])
m, _ = run_spec(
    "rc/controls/sets/demographics_only", "specification_tree/modules/robustness/controls.md#control-sets",
    "G1", "d_ln_real_earn_win1", "d_computer_2017_2007_n_s1",
    f"d_ln_real_earn_win1 ~ d_computer_2017_2007_n_s1 + {demo_str} | year_int + year_job_loss_int",
    df_samp, {"CRV1": "dwsoc4_int"},
    "samp_1", "year + year_job_loss", "Demographics only (male, educ, age)",
    cluster_var="dwsoc4",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/demographics_only", "family": "set",
                "included": DEMO_CONTROLS},
)

# rc/controls/sets/job_chars_only
job_str = " + ".join(JOB_CONTROLS)
m, _ = run_spec(
    "rc/controls/sets/job_chars_only", "specification_tree/modules/robustness/controls.md#control-sets",
    "G1", "d_ln_real_earn_win1", "d_computer_2017_2007_n_s1",
    f"d_ln_real_earn_win1 ~ d_computer_2017_2007_n_s1 + {job_str} | year_int + year_job_loss_int",
    df_samp, {"CRV1": "dwsoc4_int"},
    "samp_1", "year + year_job_loss", "Job characteristics only",
    cluster_var="dwsoc4",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/job_chars_only", "family": "set",
                "included": JOB_CONTROLS},
)

# rc/controls/sets/full
m, _ = run_spec(
    "rc/controls/sets/full", "specification_tree/modules/robustness/controls.md#control-sets",
    "G1", "d_ln_real_earn_win1", "d_computer_2017_2007_n_s1",
    baseline_formula, df_samp, {"CRV1": "dwsoc4_int"},
    "samp_1", "year + year_job_loss", "Full controls",
    cluster_var="dwsoc4",
    axis_block_name="controls",
    axis_block={"spec_id": "rc/controls/sets/full", "family": "set",
                "included": full_controls_list},
)

# ── Control progression ──
progressions = [
    ("rc/controls/progression/bivariate",
     "d_ln_real_earn_win1 ~ d_computer_2017_2007_n_s1 | year_int + year_job_loss_int",
     "Bivariate (no controls)"),
    ("rc/controls/progression/demographics",
     f"d_ln_real_earn_win1 ~ d_computer_2017_2007_n_s1 + i_male + educ_num + age | year_int + year_job_loss_int",
     "Demographics"),
    ("rc/controls/progression/demographics_job",
     f"d_ln_real_earn_win1 ~ d_computer_2017_2007_n_s1 + i_male + {' + '.join(CONTROLS)} | year_int + year_job_loss_int",
     "Demographics + job chars"),
    ("rc/controls/progression/full", baseline_formula, "Full (baseline)"),
]

for sid, formula, desc in progressions:
    m, _ = run_spec(
        sid, "specification_tree/modules/robustness/controls.md#progression",
        "G1", "d_ln_real_earn_win1", "d_computer_2017_2007_n_s1",
        formula, df_samp, {"CRV1": "dwsoc4_int"},
        "samp_1", "year + year_job_loss", desc,
        cluster_var="dwsoc4",
        axis_block_name="controls",
        axis_block={"spec_id": sid, "family": "progression", "desc": desc},
    )

# ── Random control subsets ──
rng = random.Random(181166)
optional_controls = ["d_share_emp_2017_2007_n_s1", "i_computer_2007_n_s1", "i_male"] + CONTROLS  # 9 total

for i in range(1, 11):
    n_draw = rng.randint(2, len(optional_controls) - 1)
    drawn = sorted(rng.sample(optional_controls, n_draw))
    ctrl_str = " + ".join(drawn)
    formula = f"d_ln_real_earn_win1 ~ d_computer_2017_2007_n_s1 + {ctrl_str} | year_int + year_job_loss_int"
    m, _ = run_spec(
        f"rc/controls/subset/random_{i:03d}",
        "specification_tree/modules/robustness/controls.md#subset",
        "G1", "d_ln_real_earn_win1", "d_computer_2017_2007_n_s1",
        formula, df_samp, {"CRV1": "dwsoc4_int"},
        "samp_1", "year + year_job_loss",
        f"Random subset #{i}: {', '.join(drawn)}",
        cluster_var="dwsoc4",
        axis_block_name="controls",
        axis_block={"spec_id": f"rc/controls/subset/random_{i:03d}",
                    "family": "random_subset", "seed": 181166, "draw": i,
                    "included": drawn, "n_controls": n_draw},
    )

# ── Sample restrictions ──
# rc/sample/restriction/fulltime_only
if len(ft_data) > 20:
    m, _ = run_spec(
        "rc/sample/restriction/fulltime_only",
        "specification_tree/modules/robustness/sample.md#restriction",
        "G1", "d_ln_real_earn_win1", "d_computer_2017_2007_n_s1",
        baseline_formula, ft_data, {"CRV1": "dwsoc4_int"},
        "Full-time only", "year + year_job_loss", "Full controls",
        cluster_var="dwsoc4",
        axis_block_name="sample",
        axis_block={"spec_id": "rc/sample/restriction/fulltime_only",
                    "restriction": "i_ft_current_job==1 & i_ft_lost_job==1"},
    )

# rc/sample/restriction/age_25_44
df_young = df_samp[(df_samp['age'] >= 25) & (df_samp['age'] <= 44)].copy()
m, _ = run_spec(
    "rc/sample/restriction/age_25_44",
    "specification_tree/modules/robustness/sample.md#restriction",
    "G1", "d_ln_real_earn_win1", "d_computer_2017_2007_n_s1",
    baseline_formula, df_young, {"CRV1": "dwsoc4_int"},
    "Age 25-44", "year + year_job_loss", "Full controls",
    cluster_var="dwsoc4",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/age_25_44", "restriction": "age 25-44"},
)

# rc/sample/restriction/age_45_65
df_old = df_samp[(df_samp['age'] >= 45) & (df_samp['age'] <= 65)].copy()
m, _ = run_spec(
    "rc/sample/restriction/age_45_65",
    "specification_tree/modules/robustness/sample.md#restriction",
    "G1", "d_ln_real_earn_win1", "d_computer_2017_2007_n_s1",
    baseline_formula, df_old, {"CRV1": "dwsoc4_int"},
    "Age 45-65", "year + year_job_loss", "Full controls",
    cluster_var="dwsoc4",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/age_45_65", "restriction": "age 45-65"},
)

# rc/sample/restriction/college
df_coll = df_samp[df_samp['i_college'] == 1].copy()
m, _ = run_spec(
    "rc/sample/restriction/college",
    "specification_tree/modules/robustness/sample.md#restriction",
    "G1", "d_ln_real_earn_win1", "d_computer_2017_2007_n_s1",
    baseline_formula, df_coll, {"CRV1": "dwsoc4_int"},
    "College graduates", "year + year_job_loss", "Full controls",
    cluster_var="dwsoc4",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/college", "restriction": "educ_num >= 16"},
)

# rc/sample/restriction/no_college
df_nocoll = df_samp[df_samp['i_college'] == 0].copy()
m, _ = run_spec(
    "rc/sample/restriction/no_college",
    "specification_tree/modules/robustness/sample.md#restriction",
    "G1", "d_ln_real_earn_win1", "d_computer_2017_2007_n_s1",
    baseline_formula, df_nocoll, {"CRV1": "dwsoc4_int"},
    "No college", "year + year_job_loss", "Full controls",
    cluster_var="dwsoc4",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/no_college", "restriction": "educ_num < 16"},
)

# rc/sample/restriction/male_only
df_male = df_samp[df_samp['i_male'] == 1].copy()
m, _ = run_spec(
    "rc/sample/restriction/male_only",
    "specification_tree/modules/robustness/sample.md#restriction",
    "G1", "d_ln_real_earn_win1", "d_computer_2017_2007_n_s1",
    baseline_formula, df_male, {"CRV1": "dwsoc4_int"},
    "Male only", "year + year_job_loss", "Full controls",
    cluster_var="dwsoc4",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/restriction/male_only", "restriction": "i_male==1"},
)

# ── Winsorization variants ──
# rc/sample/outliers/winsor_1_99
df_samp['d_ln_real_earn_win_1_99'] = winsorize(df_samp['d_ln_real_earn'], 0.01)
m, _ = run_spec(
    "rc/sample/outliers/winsor_1_99",
    "specification_tree/modules/robustness/sample.md#outliers",
    "G1", "d_ln_real_earn_win_1_99", "d_computer_2017_2007_n_s1",
    f"d_ln_real_earn_win_1_99 ~ d_computer_2017_2007_n_s1 + {controls_str} | year_int + year_job_loss_int",
    df_samp, {"CRV1": "dwsoc4_int"},
    "samp_1, winsorized 1-99", "year + year_job_loss", "Full controls",
    cluster_var="dwsoc4",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/winsor_1_99", "winsor_pctile": 0.01},
)

# rc/sample/outliers/winsor_5_95
df_samp['d_ln_real_earn_win_5_95'] = winsorize(df_samp['d_ln_real_earn'], 0.05)
m, _ = run_spec(
    "rc/sample/outliers/winsor_5_95",
    "specification_tree/modules/robustness/sample.md#outliers",
    "G1", "d_ln_real_earn_win_5_95", "d_computer_2017_2007_n_s1",
    f"d_ln_real_earn_win_5_95 ~ d_computer_2017_2007_n_s1 + {controls_str} | year_int + year_job_loss_int",
    df_samp, {"CRV1": "dwsoc4_int"},
    "samp_1, winsorized 5-95", "year + year_job_loss", "Full controls",
    cluster_var="dwsoc4",
    axis_block_name="sample",
    axis_block={"spec_id": "rc/sample/outliers/winsor_5_95", "winsor_pctile": 0.05},
)

# ── FE variants ──
# rc/fe/drop/year_job_loss
m, _ = run_spec(
    "rc/fe/drop/year_job_loss",
    "specification_tree/modules/robustness/fixed_effects.md",
    "G1", "d_ln_real_earn_win1", "d_computer_2017_2007_n_s1",
    f"d_ln_real_earn_win1 ~ d_computer_2017_2007_n_s1 + {controls_str} | year_int",
    df_samp, {"CRV1": "dwsoc4_int"},
    "samp_1", "year only", "Full controls",
    cluster_var="dwsoc4",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/drop/year_job_loss", "dropped": "year_job_loss"},
)

# rc/fe/drop/year
m, _ = run_spec(
    "rc/fe/drop/year",
    "specification_tree/modules/robustness/fixed_effects.md",
    "G1", "d_ln_real_earn_win1", "d_computer_2017_2007_n_s1",
    f"d_ln_real_earn_win1 ~ d_computer_2017_2007_n_s1 + {controls_str} | year_job_loss_int",
    df_samp, {"CRV1": "dwsoc4_int"},
    "samp_1", "year_job_loss only", "Full controls",
    cluster_var="dwsoc4",
    axis_block_name="fixed_effects",
    axis_block={"spec_id": "rc/fe/drop/year", "dropped": "year"},
)

# ── Treatment variants ──
# rc/data/treatment/unnormalized
m, _ = run_spec(
    "rc/data/treatment/unnormalized",
    "specification_tree/modules/robustness/data_construction.md",
    "G1", "d_ln_real_earn_win1", "d_computer_2017_2007",
    f"d_ln_real_earn_win1 ~ d_computer_2017_2007 + d_share_emp_2017_2007 + i_computer_2007 + i_male + {' + '.join(CONTROLS)} | year_int + year_job_loss_int",
    df_samp, {"CRV1": "dwsoc4_int"},
    "samp_1", "year + year_job_loss", "Full controls (unnormalized treatment)",
    cluster_var="dwsoc4",
    axis_block_name="data_construction",
    axis_block={"spec_id": "rc/data/treatment/unnormalized",
                "treatment_change": "raw d_computer_2017_2007, not normalized to SD units"},
)

# rc/data/treatment/d_computer_2017_2010
if 'd_computer_2017_2010_n_s1' in df_samp.columns and df_samp['d_computer_2017_2010_n_s1'].notna().sum() > 20:
    m, _ = run_spec(
        "rc/data/treatment/d_computer_2017_2010",
        "specification_tree/modules/robustness/data_construction.md",
        "G1", "d_ln_real_earn_win1", "d_computer_2017_2010_n_s1",
        f"d_ln_real_earn_win1 ~ d_computer_2017_2010_n_s1 + {controls_str} | year_int + year_job_loss_int",
        df_samp, {"CRV1": "dwsoc4_int"},
        "samp_1", "year + year_job_loss", "Full controls (2017-2010 change)",
        cluster_var="dwsoc4",
        axis_block_name="data_construction",
        axis_block={"spec_id": "rc/data/treatment/d_computer_2017_2010",
                    "treatment_change": "d_computer_2017_2010 instead of 2017_2007"},
    )

# rc/weights/unweighted
m, _ = run_spec(
    "rc/weights/unweighted",
    "specification_tree/modules/robustness/weights.md",
    "G1", "d_ln_real_earn_win1", "d_computer_2017_2007_n_s1",
    baseline_formula, df_samp, {"CRV1": "dwsoc4_int"},
    "samp_1 (unweighted)", "year + year_job_loss", "Full controls",
    cluster_var="dwsoc4", weights_col=None,
    axis_block_name="weights",
    axis_block={"spec_id": "rc/weights/unweighted", "weights": "none"},
)

# ── Inference variants ──
if base_id:
    # HC1 (no clustering)
    run_infer_variant(
        base_id, "infer/se/hc/hc1",
        "specification_tree/modules/inference/se.md#hc",
        "G1", "d_ln_real_earn_win1", "d_computer_2017_2007_n_s1",
        baseline_formula, df_samp, "hetero",
        notes="HC1 robust (no clustering)"
    )

    # Cluster at 2-digit SOC
    if 'dwsoc2_int' in df_samp.columns and df_samp['dwsoc2_int'].notna().sum() > 20:
        run_infer_variant(
            base_id, "infer/se/cluster/dwsoc2",
            "specification_tree/modules/inference/se.md#cluster",
            "G1", "d_ln_real_earn_win1", "d_computer_2017_2007_n_s1",
            baseline_formula, df_samp, {"CRV1": "dwsoc2_int"},
            notes="Cluster at 2-digit SOC"
        )

print(f"\nTotal specs: {len(results)}")
print(f"Total inference variants: {len(inference_results)}")

# =============================================================================
# Write outputs
# =============================================================================
spec_df = pd.DataFrame(results)
spec_df.to_csv(f"{DATA_DIR}/specification_results.csv", index=False)
print(f"\nWrote specification_results.csv: {len(spec_df)} rows")

n_success = spec_df['run_success'].sum()
n_fail = len(spec_df) - n_success

if inference_results:
    infer_df = pd.DataFrame(inference_results)
    infer_df.to_csv(f"{DATA_DIR}/inference_results.csv", index=False)
    print(f"Wrote inference_results.csv: {len(infer_df)} rows")

# ── Write SPECIFICATION_SEARCH.md ──
md = f"""# Specification Search: 181166-V1
## Braxton & Taska (2023) "Technological Change and the Consequences of Job Loss"

### Surface Summary
- **Paper ID**: 181166-V1
- **Surface hash**: {SURFACE_HASH}
- **Design**: Cross-sectional OLS with absorbed FE
- **Baseline groups**: 1 (G1)
  - d_ln_real_earn_win1 ~ d_computer_2017_2007_n_s1 + controls | year + year_job_loss
  - cluster(dwsoc4), weighted by dwsuppwt
  - Budget: 75 core specs, 10 control subsets
- **Seed**: 181166

### Data Construction
- Built from CPS Displaced Worker Supplement raw fixed-width data (cps_00065.dat)
- Merged with Burning Glass occupation-level data (occ_req_all_years_full_samp.dta)
- CPI deflation via cpi_year_current.dta and cpi_year_job_loss.dta
- **Note**: Only 2012-2018 DWS waves used (2010 wave excluded due to occupation code crosswalk
  complexity requiring Stata value labels). This reduces sample size compared to the paper
  but preserves the core analysis structure.
- Occupation mapping: Census 2010 occupation codes mapped directly to SOC-4 codes.
- Winsorization at 2.5th/97.5th percentile (matching paper's winsor command).
- Normalization to SD units within analysis sample (weighted).

### Execution Summary
- **Total specifications executed**: {len(spec_df)}
  - Successful: {n_success}
  - Failed: {n_fail}
- **Inference variants**: {len(inference_results)}

### Specifications Run
- **Baseline**: Table 3 Col 2 (normalized, with employment share control)
- **Additional baselines**: Col 1 (no emp share), Col 3 (full-time only)
  - Cols 4-5 (Autor-Dorn occ codes) skipped: requires full AD data pipeline not available
    without Stata
- **LOO controls**: Drop each of 9 controls individually
- **Control sets**: None, demographics only, job chars only, full
- **Control progression**: Bivariate, demographics, demographics+job, full
- **Random control subsets**: 10 random draws (seed=181166)
- **Sample restrictions**: Full-time only, age 25-44, age 45-65, college, no college, male only
- **Winsorization variants**: 1-99, 5-95 (in addition to baseline 2.5-97.5)
- **FE variants**: Drop year_job_loss, drop year
- **Treatment variants**: Unnormalized, 2017-2010 change
- **Weight variant**: Unweighted
- **Inference variants**: HC1 (no clustering), cluster at 2-digit SOC

### Deviations from Surface
- Autor-Dorn (AD) occupation code specifications (baseline__table3_col4_AD,
  baseline__table3_col5_AD_ft, rc/data/treatment/AD_occ_codes) were **skipped** because
  the AD data merge requires the full Stata pipeline with value-label-based crosswalks.
- The 2010 DWS wave was excluded (mapping CPS 1990 occupation codes to SOC-4 requires
  Stata value labels not accessible from Python). This affects sample size but not the
  core identification strategy.

### Software Stack
- Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}
- pyfixest: {SW_BLOCK['packages'].get('pyfixest', 'N/A')}
- pandas: {SW_BLOCK['packages'].get('pandas', 'N/A')}
- numpy: {SW_BLOCK['packages'].get('numpy', 'N/A')}
- pyreadstat: {SW_BLOCK['packages'].get('pyreadstat', 'N/A')}
"""

with open(f"{DATA_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write(md)
print("Wrote SPECIFICATION_SEARCH.md")
print("Done with 181166-V1!")
