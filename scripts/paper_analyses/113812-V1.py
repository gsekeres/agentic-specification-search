"""
Specification Search Script for Stange (2012)
"An Empirical Investigation of the Option Value of College Enrollment"
American Economic Journal: Applied Economics, 4(1), 49-84.

Paper ID: 113812-V1

Surface-driven execution:
  - G1: pvlifeincd05 ~ schooling dummies + demographics + ability | robust SEs
  - Reduced-form Mincer-style regression from NLSY79 data used to project
    lifetime income. The paper's structural model uses these reduced-form
    estimates as inputs.
  - Key parameter: m_s16 (male 4-year college premium on lifetime income)
  - 50+ specifications across controls LOO, controls progression,
    alternative DVs (discount rates, log, wage-based), sample restrictions,
    heterogeneous returns, gender subgroups

Outputs:
  - specification_results.csv
  - inference_results.csv
  - SPECIFICATION_SEARCH.md
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import json
import sys
import traceback
import warnings
warnings.filterwarnings('ignore')

# Add scripts dir for utilities
sys.path.insert(0, "scripts")
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash, software_block
)

PAPER_ID = "113812-V1"
DATA_DIR = "data/downloads/extracted/113812-V1"
OUTPUT_DIR = DATA_DIR
DATA_PATH = f"{DATA_DIR}/APP2010_0313_DataAnalysisPrograms/datasetfiles/ExternalData/nlsydata.dta"
SAMPLE_PATH = f"{DATA_DIR}/APP2010_0313_DataAnalysisPrograms/datasetfiles/ExternalData/nlsysample.csv"

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

bg = surface_obj["baseline_groups"][0]
design_audit = bg["design_audit"]
inference_canonical = bg["inference_plan"]["canonical"]

# ============================================================
# Data Loading and Preparation
# Replicate nlsylifeincome.do in Python
# ============================================================

print("Loading NLSY79 raw data...")
df_raw = pd.read_stata(DATA_PATH)
print(f"  Raw data: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

# --- Recode background variables ---
df = df_raw.copy()
df.rename(columns={'ID': 'id'}, inplace=True)

# Education category mapping (used for multiple variables)
educ_map = {
    'NONE': 0, '1ST GRADE': 1, '2ND GRADE': 2, '3RD GRADE': 3,
    '4TH GRADE': 4, '5TH GRADE': 5, '6TH GRADE': 6, '7TH GRADE': 7,
    '8TH GRADE': 8, '9TH GRADE': 9, '10TH GRADE': 10, '11TH GRADE': 11,
    '12TH GRADE': 12, '1ST YR COL': 13, '2ND YR COL': 14, '3RD YR COL': 15,
    '4TH YR COL': 16, '5TH YR COL': 17, '6TH YR COL': 18,
    '7TH YR COL': 19, '8TH YR COL OR MORE': 20,
    '1ST YEAR COLLEGE': 13, '2ND YEAR COLLEGE': 14, '3RD YEAR COLLEGE': 15,
    '4TH YEAR COLLEGE': 16, '5TH YEAR COLLEGE': 17, '6TH YEAR COLLEGE': 18,
    '7TH YEAR COLLEGE': 19, '8TH YEAR COLLEGE OR MORE': 20,
}

# Race: category -> numeric
race_map = {'HISPANIC': 1, 'BLACK': 2, 'NON-BLACK, NON-HISPANIC': 3}
df['race_num'] = df['race'].map(race_map)
df['latino'] = (df['race_num'] == 1).astype(float)
df['black'] = (df['race_num'] == 2).astype(float)
df['other'] = (df['race_num'] == 3).astype(float)

# Sex
sex_map = {'MALE': 1, 'FEMALE': 2}
df['sex_num'] = df['sex'].map(sex_map)
df['male'] = (df['sex_num'] == 1).astype(float)
df['female'] = (df['sex_num'] == 2).astype(float)

# Age in 1979
df['age79'] = pd.to_numeric(df['age79'], errors='coerce')

# Mother/Father education -- these are categorical like educ vars
df['mothereduc79'] = df['mothereduc79'].astype(str).str.strip().map(educ_map)
df['fathereduc79'] = df['fathereduc79'].astype(str).str.strip().map(educ_map)

# AFQT
df['afqt89'] = pd.to_numeric(df['afqt89'], errors='coerce')
df['afqt89'] = df['afqt89'].where(df['afqt89'] >= 0)

# Urban
df['urban14_raw'] = pd.to_numeric(df['urban14'], errors='coerce')
# Recode: 1=urban, 0=rural (combining not-farm and farm), missing if <0
# In the original: 1="IN TOWN OR CITY" -> 1, 2="IN COUNTRY-NOT FARM"->0, 3="ON FARM"->0
urban_map = {'IN TOWN OR CITY': 1, 'IN COUNTRY-NOT FARM': 0, 'ON FARM OR RANCH': 0}
df['urban14'] = df['urban14'].map(urban_map)

# Sample indicator
df['sample_num'] = pd.to_numeric(df['sample_NLSY'].cat.codes, errors='coerce')
# We need to identify the cross-section sample: sample <= 8 in Stata
# The original sample_NLSY categories: need to map to 1-20
# NLSY79 sample IDs: 1-8 are cross-section, 9-14 are supplemental, 15-20 are military
sample_str_to_num = {
    'CROSS MALE WHITE': 1, 'CROSS FEMALE WHITE': 2,
    'CROSS MALE BLACK': 3, 'CROSS FEMALE BLACK': 4,
    'CROSS MALE HISPANIC': 5, 'CROSS FEMALE HISPANIC': 6,
    'SUP MALE WH POOR': 7, 'SUP FEMALE WH POOR': 8, 'SUP FEM WH POOR': 8,
    'SUP MALE BLACK': 9, 'SUP FEMALE BLACK': 10,
    'SUP MALE HISPANIC': 11, 'SUP FEMALE HISPANIC': 12,
    'MIL MALE WHITE': 13, 'MIL FEMALE WHITE': 14,
    'MIL MALE BLACK': 15, 'MIL FEMALE BLACK': 16,
    'MIL MALE HISPANIC': 17, 'MIL FEMALE HISPANIC': 18,
    'SUP MALE WH POOR ': 7, 'SUP FEM WH POOR ': 8,
}
df['sample'] = df['sample_NLSY'].astype(str).map(sample_str_to_num)
# If unmapped, try cleaning
unmapped = df['sample'].isna()
if unmapped.sum() > 0:
    # Try matching more flexibly
    for idx in df[unmapped].index:
        s = str(df.loc[idx, 'sample_NLSY']).strip().upper()
        for k, v in sample_str_to_num.items():
            if k.strip().upper() in s or s in k.strip().upper():
                df.loc[idx, 'sample'] = v
                break

# Weights
df['w79'] = pd.to_numeric(df['w79'], errors='coerce')

# --- Recode education data ---
# Convert education categories to numeric years (using educ_map defined above)

for yr_suffix in list(range(79, 95)):
    col = f'educ{yr_suffix}'
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().map(educ_map)

# Also handle educ96, educ98, educ100, educ102, educ104
for yr_suffix in [96, 98, 100, 102, 104]:
    col = f'educ{yr_suffix}'
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().map(educ_map)

# For years 95-127, set to educ94 (as in do-file)
for yr in range(95, 128):
    df[f'educ{yr}'] = df['educ94']

# Interpolate education: if attainment unchanged, fill missing years
for x0 in range(79, 93):
    x1 = x0 + 1
    col0 = f'educ{x0}'
    col1 = f'educ{x1}'
    if col0 in df.columns and col1 in df.columns:
        for xfill in range(x0+2, min(x0+13, 128)):
            colfill = f'educ{xfill}'
            if colfill in df.columns:
                mask = (df[col0] == df[colfill]) & df[col1].isna()
                df.loc[mask, col1] = df.loc[mask, colfill]

# --- Recode income data ---
for col in df.columns:
    if col.startswith('totinc'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df.loc[df[col] < 0, col] = np.nan

# Missing years
for yr in [94, 96, 98, 100, 102, 104, 105, 106]:
    if f'totinc{yr}' not in df.columns:
        df[f'totinc{yr}'] = np.nan

# CPI-U deflators to 1992 dollars
cpiu = {
    78: 0.465, 79: 0.517, 80: 0.587, 81: 0.648, 82: 0.688,
    83: 0.71, 84: 0.741, 85: 0.767, 86: 0.781, 87: 0.81,
    88: 0.843, 89: 0.884, 90: 0.932, 91: 0.971, 92: 1.0,
    93: 1.03, 94: 1.056, 95: 1.086, 96: 1.118, 97: 1.144,
    98: 1.162, 99: 1.187, 100: 1.227, 101: 1.262, 102: 1.282,
    103: 1.311, 104: 1.346
}

for yr, c in cpiu.items():
    col = f'totinc{yr}'
    if col in df.columns:
        df[col] = df[col] / c

# Interpolate between survey years
if 'totinc94' in df.columns and 'totinc93' in df.columns and 'totinc95' in df.columns:
    df['totinc94'] = (df['totinc93'] + df['totinc95']) / 2
if 'totinc96' in df.columns:
    df['totinc96'] = (df['totinc95'] + df['totinc97']) / 2
if 'totinc98' in df.columns:
    df['totinc98'] = (df['totinc97'] + df['totinc99']) / 2
if 'totinc100' in df.columns:
    df['totinc100'] = (df['totinc99'] + df['totinc101']) / 2
if 'totinc102' in df.columns:
    df['totinc102'] = (df['totinc101'] + df['totinc103']) / 2

# Linear interpolation for missing within-survey-year income
for x0 in range(78, 102):
    x1 = x0 + 1
    x2 = x0 + 2
    col1 = f'totinc{x1}'
    col0 = f'totinc{x0}'
    col2 = f'totinc{x2}'
    if col1 in df.columns and col0 in df.columns and col2 in df.columns:
        mask = df[col1].isna() & df[col0].notna() & df[col2].notna()
        df.loc[mask, col1] = (df.loc[mask, col0] + df.loc[mask, col2]) / 2

# --- Generate age-specific income ---
# Map year-specific income to age-specific
for age_at_start in range(14, 23):
    for age in range(age_at_start, min(age_at_start + 26, 46)):
        yr = age + 79 - age_at_start
        inc_col = f'inc{age}'
        totinc_col = f'totinc{yr}'
        if inc_col not in df.columns:
            df[inc_col] = np.nan
        if totinc_col in df.columns:
            mask = df['age79'] == age_at_start
            df.loc[mask, inc_col] = df.loc[mask, totinc_col]

# Assume no income growth after age 38
for age in range(39, 63):
    if f'inc{age}' not in df.columns:
        df[f'inc{age}'] = np.nan
    df[f'inc{age}'] = df[f'inc{age}'].fillna(df.get('inc38', np.nan) if 'inc38' in df.columns else np.nan)
    mask = df[f'inc{age}'].isna() & df.get('inc38', pd.Series(dtype=float)).notna()
    if 'inc38' in df.columns:
        df.loc[mask, f'inc{age}'] = df.loc[mask, 'inc38']

# inc25 for later use
df['inc25'] = df.get('inc25', pd.Series(np.nan, index=df.index))

# --- HS graduation and continuous enrollment ---
# hsged mapping: NLSY codes are 0=not grad, 1=diploma, 2=GED, 3=both
hsged_map = {'HIGH SCHOOL DIPLOMA': 1, 'GED': 2, 'BOTH': 3}

for yr in [79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 96, 98, 100, 102, 104]:
    hsged_col = f'hsged{yr}'
    if hsged_col in df.columns:
        # Try categorical mapping
        mapped = df[hsged_col].astype(str).str.strip().map(hsged_map)
        # For values that didn't map (like -4, -5, -3, NO), try numeric
        still_nan = mapped.isna()
        numeric_vals = pd.to_numeric(df[hsged_col], errors='coerce')
        # Negative values or "NO" -> keep as NaN
        mapped.loc[still_nan & (numeric_vals >= 0)] = numeric_vals.loc[still_nan & (numeric_vals >= 0)]
        df[hsged_col] = mapped

    for prefix in ['hsgrad', 'hsgradmo', 'hsgradyr']:
        col = f'{prefix}{yr}'
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df.loc[df[col] < 0, col] = np.nan

# HS diploma year for conteduc3
for yr in [94, 96, 98, 100, 102, 104]:
    col = f'hsgradyr{yr}'
    if col in df.columns:
        df.loc[df[col].notna() & (df[col] > 1900), col] = df.loc[df[col].notna() & (df[col] > 1900), col] - 1900

# Construct hsdiploma, hsdiplomayr
df['hsdiploma'] = np.nan
df['hsdiplomayr'] = np.nan

for yr in range(79, 105):
    hsged_col = f'hsged{yr}'
    hsgradyr_col = f'hsgradyr{yr}'
    if hsged_col in df.columns:
        hsged_vals = df[hsged_col]
        # hsdiploma = 0 if hsged==0 or hsged==2
        mask0 = (df['hsdiploma'].isna()) & (hsged_vals.isin([0, 2]))
        df.loc[mask0, 'hsdiploma'] = 0
        # hsdiploma = 1 if hsged==1 or hsged==3
        mask1 = (df['hsdiploma'].isna() | (df['hsdiploma'] == 0)) & (hsged_vals.isin([1, 3]))
        if hsgradyr_col in df.columns:
            df.loc[mask1, 'hsdiplomayr'] = df.loc[mask1, hsgradyr_col]
        df.loc[mask1, 'hsdiploma'] = 1

# --- Generate age-specific education from year-specific ---
# The do-file creates educ14, educ15, ..., educ46 from educ{year} based on age79
# educ{age} = educ{year} where year = age + 79 - age79
for age_at_79 in range(14, 23):
    for target_age in range(age_at_79, min(age_at_79 + 26, 47)):
        yr = target_age + 79 - age_at_79
        yr_col = f'educ{yr}'
        age_col = f'educ_age{target_age}'
        if age_col not in df.columns:
            df[age_col] = np.nan
        if yr_col in df.columns:
            mask = df['age79'] == age_at_79
            df.loc[mask, age_col] = df.loc[mask, yr_col]

# Continuous enrollment (conteduc)
# Measure 1: crude measure based on educational attainment by age
df['conteduc1'] = np.nan
# People with educ_age18>=12 (HS completed by age 18)
for start_age in [18, 19]:
    base_educ = 12
    ecol_start = f'educ_age{start_age}'
    if ecol_start not in df.columns:
        continue
    mask_start = df[ecol_start].notna() & (df[ecol_start] >= base_educ)
    df.loc[mask_start & df['conteduc1'].isna(), 'conteduc1'] = 12
    for yrs_after in range(1, 9):
        target_age = start_age + yrs_after
        target_educ = base_educ + yrs_after
        ecol = f'educ_age{target_age}'
        if ecol in df.columns:
            mask = df[ecol].notna() & (df[ecol] >= target_educ)
            df.loc[mask, 'conteduc1'] = target_educ

# People with only 12th grade by older ages
for check_age in [20, 21, 22]:
    ecol = f'educ_age{check_age}'
    if ecol in df.columns:
        mask = df['conteduc1'].isna() & (df[ecol] == 12) & (df['age79'] == check_age)
        df.loc[mask, 'conteduc1'] = 12

# Measure 3: using stated year of HS graduation (uses year-based educ)
df['conteduc3'] = np.nan
for hs_yr in range(74, 90):
    t1 = hs_yr + 1
    for yrs_after in range(0, 9):
        target_yr = t1 + yrs_after
        target_educ = 12 + yrs_after
        ecol = f'educ{target_yr}'
        if ecol in df.columns:
            mask = (df['hsdiplomayr'] == hs_yr) & df[ecol].notna() & (df[ecol] >= target_educ)
            df.loc[mask, 'conteduc3'] = target_educ

# Combine measures
df['conteduc'] = df['conteduc3']
df.loc[df['conteduc'].isna(), 'conteduc'] = df.loc[df['conteduc'].isna(), 'conteduc1']

# --- Region variables ---
# Region coding: NORTHEAST=1, NORTH CENTRAL=2, SOUTH=3, WEST=4
region_map = {'NORTHEAST': 1, 'NORTH CENTRAL': 2, 'SOUTH': 3, 'WEST': 4}

# Convert all region columns to numeric
for yr in range(79, 105):
    rcol = f'region{yr}'
    if rcol in df.columns:
        df[f'{rcol}_num'] = df[rcol].astype(str).str.strip().map(region_map)

# Region at age 18
df['region18'] = np.nan
for check_age, yr in [(14, 83), (15, 82), (16, 81), (17, 80)]:
    rcol_num = f'region{yr}_num'
    if rcol_num in df.columns:
        mask = df['age79'] == check_age
        df.loc[mask, 'region18'] = df.loc[mask, rcol_num]
for check_age in range(18, 23):
    rcol_num = 'region79_num'
    if rcol_num in df.columns:
        mask = (df['age79'] == check_age) & df[rcol_num].notna()
        df.loc[mask, 'region18'] = df.loc[mask, rcol_num]

df['regionne'] = (df['region18'] == 1).astype(float).where(df['region18'].notna())
df['regionnc'] = (df['region18'] == 2).astype(float).where(df['region18'].notna())
df['regionso'] = (df['region18'] == 3).astype(float).where(df['region18'].notna())
df['regionwe'] = (df['region18'] == 4).astype(float).where(df['region18'].notna())

# --- HS GPA ---
# Course grades are letter grades in the Stata file. In Stata, these get loaded as
# numeric codes: A=1, B=2, C=3, D=4, E/F=5, PASS=6 (6 excluded as incomplete).
# GPA = sum(grade_points) / num_courses where A=4, B=3, C=2, D=1, F=0.
# The NLSY coding: 1=A, 2=B, 3=C, 4=D, 5=E/F(fail), 6=PASS(excluded)
grade_letter_to_points = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'E/F (NON-PASS)': 0, 'PASS': np.nan}
# Also handle numeric codes from Stata: 1->A=4, 2->B=3, 3->C=2, 4->D=1, 5->F=0
grade_num_to_points = {1: 4, 2: 3, 3: 2, 4: 1, 5: 0}

df['courseshs'] = 0
df['gradepointshs'] = 0.0
for i in range(1, 65):
    gcol = f'course{i}grade'
    if gcol in df.columns:
        # Try mapping from letter grades first
        grade_pts = df[gcol].astype(str).str.strip().map(grade_letter_to_points)
        # If that yields all NaN, try numeric
        if grade_pts.notna().sum() == 0:
            grade_pts = pd.to_numeric(df[gcol], errors='coerce')
            grade_pts = grade_pts.map(grade_num_to_points)

        valid = grade_pts.notna()
        df.loc[valid, 'courseshs'] += 1
        df.loc[valid, 'gradepointshs'] += grade_pts[valid]

df.loc[df['courseshs'] == 0, 'courseshs'] = np.nan
df['gpahs'] = df['gradepointshs'] / df['courseshs']

# --- Compute present value of lifetime income ---
# For each starting age (18-25), compute PV of income from that age to 62
discount_05 = 0.05
discount_10 = 0.10

for start_age in range(18, 26):
    # Count income years
    income_yrs_col = f'incomeyrs{start_age}'
    df[income_yrs_col] = 0
    for age in range(start_age, 63):
        icol = f'inc{age}'
        if icol in df.columns:
            df.loc[df[icol].notna(), income_yrs_col] += 1

    expected_yrs = 63 - start_age  # ages start_age to 62 inclusive

    # PV at 5% discount
    pv05_col = f'pvlifeinc{start_age}62d05'
    df[pv05_col] = np.nan
    complete = df[income_yrs_col] == expected_yrs
    df.loc[complete, pv05_col] = 0.0
    for age in range(start_age, 63):
        icol = f'inc{age}'
        if icol in df.columns:
            df.loc[complete, pv05_col] += ((1 / (1 + discount_05)) ** (age - start_age)) * df.loc[complete, icol].fillna(0)

    # PV at 10% discount
    pv10_col = f'pvlifeinc{start_age}62d10'
    df[pv10_col] = np.nan
    df.loc[complete, pv10_col] = 0.0
    for age in range(start_age, 63):
        icol = f'inc{age}'
        if icol in df.columns:
            df.loc[complete, pv10_col] += ((1 / (1 + discount_10)) ** (age - start_age)) * df.loc[complete, icol].fillna(0)

# Assign PV based on continuous education level
df['pvlifeincd05'] = np.nan
df.loc[df['conteduc'] == 12, 'pvlifeincd05'] = df.loc[df['conteduc'] == 12, 'pvlifeinc1962d05']
df.loc[df['conteduc'] == 13, 'pvlifeincd05'] = df.loc[df['conteduc'] == 13, 'pvlifeinc2062d05']
df.loc[df['conteduc'] == 14, 'pvlifeincd05'] = df.loc[df['conteduc'] == 14, 'pvlifeinc2162d05']
df.loc[df['conteduc'] == 15, 'pvlifeincd05'] = df.loc[df['conteduc'] == 15, 'pvlifeinc2262d05']
df.loc[df['conteduc'] >= 16, 'pvlifeincd05'] = df.loc[df['conteduc'] >= 16, 'pvlifeinc2362d05']

df['pvlifeincd10'] = np.nan
df.loc[df['conteduc'] == 12, 'pvlifeincd10'] = df.loc[df['conteduc'] == 12, 'pvlifeinc1962d10']
df.loc[df['conteduc'] == 13, 'pvlifeincd10'] = df.loc[df['conteduc'] == 13, 'pvlifeinc2062d10']
df.loc[df['conteduc'] == 14, 'pvlifeincd10'] = df.loc[df['conteduc'] == 14, 'pvlifeinc2162d10']
df.loc[df['conteduc'] == 15, 'pvlifeincd10'] = df.loc[df['conteduc'] == 15, 'pvlifeinc2262d10']
df.loc[df['conteduc'] >= 16, 'pvlifeincd10'] = df.loc[df['conteduc'] >= 16, 'pvlifeinc2362d10']

# Scale to thousands
df['pvlifeincd05'] = df['pvlifeincd05'] / 1000
df['pvlifeincd10'] = df['pvlifeincd10'] / 1000

# Log versions
df['lpvlifeincd05'] = np.log(df['pvlifeincd05'].clip(lower=0.001))
df['lpvlifeincd10'] = np.log(df['pvlifeincd10'].clip(lower=0.001))

# Income at age 25 for secondary analysis
df['inc25_clean'] = df['inc25'].copy()
df.loc[df['inc25_clean'] > 100000, 'inc25_clean'] = 100000

# --- Drop income outliers as in original ---
# drop if totinc88 > 500000
df = df[~((df['totinc88'] > 500000) & df['totinc88'].notna())].copy()
df = df[~((df['totinc91'] > 1000000) & df['totinc91'].notna())].copy()

# --- Generate analysis variables ---
df['parentba'] = np.nan
has_mother = df['mothereduc79'].notna()
has_father = df['fathereduc79'].notna()
df.loc[has_mother | has_father, 'parentba'] = 0
df.loc[(has_mother & (df['mothereduc79'] >= 16)) | (has_father & (df['fathereduc79'] >= 16)), 'parentba'] = 1

df['parented'] = np.nan
both = has_mother & has_father
df.loc[both, 'parented'] = np.maximum(df.loc[both, 'fathereduc79'], df.loc[both, 'mothereduc79'])
df.loc[has_father & ~has_mother, 'parented'] = df.loc[has_father & ~has_mother, 'fathereduc79']
df.loc[has_mother & ~has_father, 'parented'] = df.loc[has_mother & ~has_father, 'mothereduc79']

# Schooling dummies
df['s12'] = (df['conteduc'] == 12).astype(float).where(df['conteduc'].notna())
df['s13'] = (df['conteduc'] == 13).astype(float).where(df['conteduc'].notna())
df['s14'] = (df['conteduc'] == 14).astype(float).where(df['conteduc'].notna())
df['s15'] = (df['conteduc'] == 15).astype(float).where(df['conteduc'].notna())
df['s16'] = (df['conteduc'] >= 16).astype(float).where(df['conteduc'].notna())

# Ability measures
df['afqthigh'] = (df['afqt89'] >= 50).astype(float).where(df['afqt89'].notna())
df['gpahigh'] = (df['gpahs'] >= 2.5).astype(float).where(df['gpahs'].notna())

# Interactions
df['gpahs_afqt89'] = df['gpahs'] * df['afqt89']
df['gpahs_parented'] = df['gpahs'] * df['parented']
df['afqt89_parented'] = df['afqt89'] * df['parented']

# Schooling x ability interactions
df['s1315_afqt89'] = (df['s13'] + df['s14'] + df['s15']) * df['afqt89']
df['s16_afqt89'] = df['s16'] * df['afqt89']
df['s1315_gpahs'] = (df['s13'] + df['s14'] + df['s15']) * df['gpahs']
df['s16_gpahs'] = df['s16'] * df['gpahs']
df['s1315_parented'] = (df['s13'] + df['s14'] + df['s15']) * df['parented']
df['s16_parented'] = df['s16'] * df['parented']

# Gender interactions
base_vars = ['s12', 's13', 's14', 's15', 's16', 'parented', 'black', 'latino',
             'regionnc', 'regionso', 'regionwe', 'urban14', 'gpahs', 'afqt89',
             'gpahs_afqt89', 'gpahs_parented', 'afqt89_parented',
             's1315_afqt89', 's16_afqt89', 's1315_gpahs', 's16_gpahs',
             's1315_parented', 's16_parented', 'inc25_clean']

for var in base_vars:
    if var in df.columns:
        df[f'm_{var}'] = df['male'] * df[var]
        df[f'f_{var}'] = df['female'] * df[var]

print(f"\nData after processing: {len(df)} rows")
print(f"  conteduc non-missing: {df['conteduc'].notna().sum()}")
print(f"  pvlifeincd05 non-missing: {df['pvlifeincd05'].notna().sum()}")
print(f"  gpahs non-missing: {df['gpahs'].notna().sum()}")
print(f"  afqt89 non-missing: {df['afqt89'].notna().sum()}")
print(f"  parented non-missing: {df['parented'].notna().sum()}")

# --- Define estimation sample ---
# The baseline regression uses all NLSY79 sample members with complete data
# (the original code does NOT restrict to sample<=4 in the baseline; that restriction
# is commented out with *drop if sample > 8)
analysis_vars = ['pvlifeincd05', 'male', 'female',
                 'm_s13', 'm_s14', 'm_s15', 'm_s16',
                 'm_parented', 'm_black', 'm_latino',
                 'm_regionnc', 'm_regionso', 'm_regionwe',
                 'm_urban14', 'm_gpahs', 'm_afqt89',
                 'm_gpahs_afqt89', 'm_gpahs_parented', 'm_afqt89_parented',
                 'f_s13', 'f_s14', 'f_s15', 'f_s16',
                 'f_parented', 'f_black', 'f_latino',
                 'f_regionnc', 'f_regionso', 'f_regionwe',
                 'f_urban14', 'f_gpahs', 'f_afqt89',
                 'f_gpahs_afqt89', 'f_gpahs_parented', 'f_afqt89_parented']

df_est = df.dropna(subset=analysis_vars).copy()
print(f"\nEstimation sample: {len(df_est)} rows")
print(f"  Males: {(df_est['male']==1).sum()}, Females: {(df_est['female']==1).sum()}")
print(f"  Mean pvlifeincd05: {df_est['pvlifeincd05'].mean():.1f}")
print(f"  Schooling distribution:")
for s in [12, 13, 14, 15]:
    n = (df_est['conteduc'] == s).sum()
    print(f"    s={s}: {n} ({n/len(df_est)*100:.1f}%)")
n16 = (df_est['conteduc'] >= 16).sum()
print(f"    s>=16: {n16} ({n16/len(df_est)*100:.1f}%)")

# ============================================================
# Accumulators
# ============================================================

results = []
inference_results = []
spec_run_counter = 0

# ============================================================
# Helper: run_spec (OLS with robust SEs via pyfixest)
# ============================================================

def run_spec(spec_id, spec_tree_path, baseline_group_id,
             outcome_var, treatment_var, rhs_vars, data,
             vcov_type, sample_desc, controls_desc,
             no_intercept=True,
             axis_block_name=None, axis_block=None, notes=""):
    """Run a single OLS specification and record results."""
    global spec_run_counter
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"

    try:
        rhs_str = " + ".join(rhs_vars)
        if no_intercept:
            # pyfixest doesn't directly support noconstant for feols,
            # but we can use statsmodels or include the constant-absorbing dummies
            # Since male + female = 1 absorbs the constant, the OLS will
            # effectively be no-intercept if both are included.
            # pyfixest feols always includes an intercept, so we include
            # male and female as regressors (which together span the constant)
            formula = f"{outcome_var} ~ {rhs_str}"
        else:
            formula = f"{outcome_var} ~ {rhs_str}"

        if vcov_type == "iid":
            vcov_arg = "iid"
        elif vcov_type == "HC3":
            vcov_arg = {"CRV1": "id"}  # fallback; use HC3 via statsmodels if needed
            # pyfixest supports HC1 via hetero, HC3 not directly
            vcov_arg = "hetero"  # HC1
        else:
            vcov_arg = "hetero"  # HC1 = robust

        m = pf.feols(formula, data=data, vcov=vcov_arg)

        coef_val = float(m.coef().get(treatment_var, np.nan))
        se_val = float(m.se().get(treatment_var, np.nan))
        pval = float(m.pvalue().get(treatment_var, np.nan))

        try:
            ci = m.confint()
            ci_lower = float(ci.loc[treatment_var, ci.columns[0]]) if treatment_var in ci.index else np.nan
            ci_upper = float(ci.loc[treatment_var, ci.columns[1]]) if treatment_var in ci.index else np.nan
        except:
            ci_lower = np.nan
            ci_upper = np.nan

        nobs = int(m._N)
        try:
            r2 = float(m._r2)
        except:
            r2 = np.nan

        all_coefs = {k: float(v) for k, v in m.coef().items()}

        payload = make_success_payload(
            coefficients=all_coefs,
            inference={"spec_id": inference_canonical["spec_id"],
                       "method": "robust", "cluster_vars": []},
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH,
            design={"cross_sectional_ols": design_audit},
            axis_block_name=axis_block_name,
            axis_block=axis_block,
        )

        results.append({
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
            "fixed_effects": "none",
            "controls_desc": controls_desc,
            "cluster_var": "",
            "run_success": 1,
            "run_error": ""
        })
        return run_id, coef_val, se_val, pval, nobs

    except Exception as e:
        err_msg = str(e)[:240]
        err_details = error_details_from_exception(e, stage="estimation")
        payload = make_failure_payload(
            error=err_msg,
            error_details=err_details,
            software=SW_BLOCK,
            surface_hash=SURFACE_HASH
        )

        results.append({
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
            "fixed_effects": "none",
            "controls_desc": controls_desc,
            "cluster_var": "",
            "run_success": 0,
            "run_error": err_msg
        })
        return run_id, np.nan, np.nan, np.nan, np.nan


# ============================================================
# Define variable groups
# ============================================================

# Male RHS for baseline (Spec A): no-constant with male/female intercepts
MALE_SCHOOLING = ['m_s13', 'm_s14', 'm_s15', 'm_s16']
MALE_DEMOGRAPHICS = ['m_parented', 'm_black', 'm_latino',
                     'm_regionnc', 'm_regionso', 'm_regionwe', 'm_urban14']
MALE_ABILITY = ['m_gpahs', 'm_afqt89']
MALE_INTERACTIONS = ['m_gpahs_afqt89', 'm_gpahs_parented', 'm_afqt89_parented']

FEMALE_SCHOOLING = ['f_s13', 'f_s14', 'f_s15', 'f_s16']
FEMALE_DEMOGRAPHICS = ['f_parented', 'f_black', 'f_latino',
                       'f_regionnc', 'f_regionso', 'f_regionwe', 'f_urban14']
FEMALE_ABILITY = ['f_gpahs', 'f_afqt89']
FEMALE_INTERACTIONS = ['f_gpahs_afqt89', 'f_gpahs_parented', 'f_afqt89_parented']

BASELINE_RHS = ['male'] + MALE_SCHOOLING + MALE_DEMOGRAPHICS + MALE_ABILITY + MALE_INTERACTIONS + \
               ['female'] + FEMALE_SCHOOLING + FEMALE_DEMOGRAPHICS + FEMALE_ABILITY + FEMALE_INTERACTIONS

MALE_CONTROLS = MALE_DEMOGRAPHICS + MALE_ABILITY + MALE_INTERACTIONS
FEMALE_CONTROLS = FEMALE_DEMOGRAPHICS + FEMALE_ABILITY + FEMALE_INTERACTIONS
ALL_CONTROLS = MALE_CONTROLS + FEMALE_CONTROLS

# Heterogeneous returns variables
HETERO_AFQT = ['m_s1315_afqt89', 'm_s16_afqt89', 'f_s1315_afqt89', 'f_s16_afqt89']
HETERO_GPAHS = ['m_s1315_gpahs', 'm_s16_gpahs', 'f_s1315_gpahs', 'f_s16_gpahs']
HETERO_PARENTED = ['m_s1315_parented', 'm_s16_parented', 'f_s1315_parented', 'f_s16_parented']


# ============================================================
# 1. BASELINE SPECIFICATION (Spec A: 5% discount)
# ============================================================
print("\n=== Running Baseline Specification ===")
run_spec(
    spec_id="baseline__specA_5pct",
    spec_tree_path="cross_sectional_ols/baseline",
    baseline_group_id="G1",
    outcome_var="pvlifeincd05",
    treatment_var="m_s16",
    rhs_vars=BASELINE_RHS,
    data=df_est,
    vcov_type="HC1",
    sample_desc="Full NLSY79 sample with complete data",
    controls_desc="Gender-interacted: schooling dummies + demographics + ability + interactions"
)

# ============================================================
# 2. CONTROLS LEAVE-ONE-OUT
# ============================================================
print("\n=== Running Controls LOO ===")

loo_groups = {
    'drop_m_parented': (['m_parented', 'f_parented'], "Drop parented"),
    'drop_m_black': (['m_black', 'f_black'], "Drop black"),
    'drop_m_latino': (['m_latino', 'f_latino'], "Drop latino"),
    'drop_m_region': (['m_regionnc', 'm_regionso', 'm_regionwe',
                       'f_regionnc', 'f_regionso', 'f_regionwe'], "Drop region"),
    'drop_m_urban14': (['m_urban14', 'f_urban14'], "Drop urban"),
    'drop_m_gpahs': (['m_gpahs', 'f_gpahs', 'm_gpahs_afqt89', 'f_gpahs_afqt89',
                       'm_gpahs_parented', 'f_gpahs_parented'], "Drop GPA + interactions"),
    'drop_m_afqt89': (['m_afqt89', 'f_afqt89', 'm_gpahs_afqt89', 'f_gpahs_afqt89',
                        'm_afqt89_parented', 'f_afqt89_parented'], "Drop AFQT + interactions"),
    'drop_m_interactions': (['m_gpahs_afqt89', 'f_gpahs_afqt89',
                              'm_gpahs_parented', 'f_gpahs_parented',
                              'm_afqt89_parented', 'f_afqt89_parented'], "Drop all interactions"),
}

for loo_name, (drop_vars, desc) in loo_groups.items():
    rhs = [v for v in BASELINE_RHS if v not in drop_vars]
    run_spec(
        spec_id=f"rc/controls/loo/{loo_name}",
        spec_tree_path="cross_sectional_ols/rc/controls/loo",
        baseline_group_id="G1",
        outcome_var="pvlifeincd05",
        treatment_var="m_s16",
        rhs_vars=rhs,
        data=df_est,
        vcov_type="HC1",
        sample_desc="Full NLSY79 sample",
        controls_desc=desc
    )

# ============================================================
# 3. CONTROLS PROGRESSION
# ============================================================
print("\n=== Running Controls Progression ===")

progressions = {
    'schooling_only': ['male'] + MALE_SCHOOLING + ['female'] + FEMALE_SCHOOLING,
    'schooling_demographics': ['male'] + MALE_SCHOOLING + MALE_DEMOGRAPHICS + \
                              ['female'] + FEMALE_SCHOOLING + FEMALE_DEMOGRAPHICS,
    'schooling_demo_ability': ['male'] + MALE_SCHOOLING + MALE_DEMOGRAPHICS + MALE_ABILITY + \
                              ['female'] + FEMALE_SCHOOLING + FEMALE_DEMOGRAPHICS + FEMALE_ABILITY,
    'full': BASELINE_RHS,
}

for prog_name, rhs in progressions.items():
    run_spec(
        spec_id=f"rc/controls/progression/{prog_name}",
        spec_tree_path="cross_sectional_ols/rc/controls/progression",
        baseline_group_id="G1",
        outcome_var="pvlifeincd05",
        treatment_var="m_s16",
        rhs_vars=rhs,
        data=df_est,
        vcov_type="HC1",
        sample_desc="Full NLSY79 sample",
        controls_desc=f"Progression: {prog_name}"
    )

# ============================================================
# 4. GENDER SUBGROUPS (men only, women only)
# ============================================================
print("\n=== Running Gender Subgroups ===")

# Men only
df_men = df_est[df_est['male'] == 1].copy()
men_rhs = ['m_s13', 'm_s14', 'm_s15', 'm_s16',
            'm_parented', 'm_black', 'm_latino',
            'm_regionnc', 'm_regionso', 'm_regionwe', 'm_urban14',
            'm_gpahs', 'm_afqt89',
            'm_gpahs_afqt89', 'm_gpahs_parented', 'm_afqt89_parented']
run_spec(
    spec_id="rc/controls/sets/men_only",
    spec_tree_path="cross_sectional_ols/rc/controls/sets",
    baseline_group_id="G1",
    outcome_var="pvlifeincd05",
    treatment_var="m_s16",
    rhs_vars=men_rhs,
    data=df_men,
    vcov_type="HC1",
    sample_desc="Men only",
    controls_desc="Men only: schooling + demographics + ability + interactions",
    no_intercept=False
)

# Women only
df_women = df_est[df_est['female'] == 1].copy()
women_rhs = ['f_s13', 'f_s14', 'f_s15', 'f_s16',
             'f_parented', 'f_black', 'f_latino',
             'f_regionnc', 'f_regionso', 'f_regionwe', 'f_urban14',
             'f_gpahs', 'f_afqt89',
             'f_gpahs_afqt89', 'f_gpahs_parented', 'f_afqt89_parented']
run_spec(
    spec_id="rc/controls/sets/women_only",
    spec_tree_path="cross_sectional_ols/rc/controls/sets",
    baseline_group_id="G1",
    outcome_var="pvlifeincd05",
    treatment_var="f_s16",
    rhs_vars=women_rhs,
    data=df_women,
    vcov_type="HC1",
    sample_desc="Women only",
    controls_desc="Women only: schooling + demographics + ability + interactions",
    no_intercept=False
)

# ============================================================
# 5. ALTERNATIVE DEPENDENT VARIABLES
# ============================================================
print("\n=== Running Alternative DVs ===")

# Need to ensure alt DVs have complete data
alt_dvs = {
    'pvlifeincd10': ('pvlifeincd10', 'PV lifetime income, 10% discount'),
    'lpvlifeincd05': ('lpvlifeincd05', 'Log PV lifetime income, 5% discount'),
    'lpvlifeincd10': ('lpvlifeincd10', 'Log PV lifetime income, 10% discount'),
}

for dv_name, (dv_col, dv_desc) in alt_dvs.items():
    if dv_col in df_est.columns:
        df_dv = df_est.dropna(subset=[dv_col]).copy()
        if len(df_dv) > 100:
            run_spec(
                spec_id=f"rc/outcome/alt_dv/{dv_name}",
                spec_tree_path="cross_sectional_ols/rc/outcome",
                baseline_group_id="G1",
                outcome_var=dv_col,
                treatment_var="m_s16",
                rhs_vars=BASELINE_RHS,
                data=df_dv,
                vcov_type="HC1",
                sample_desc="Full NLSY79 sample",
                controls_desc=f"Alt DV: {dv_desc}"
            )

# ============================================================
# 6. HETEROGENEOUS RETURNS SPECIFICATIONS
# ============================================================
print("\n=== Running Heterogeneous Returns ===")

# Spec C from earnings_projection.do: add schooling x ability interactions
hetero_specs = {
    'add_s1315_afqt89_interactions': (HETERO_AFQT, "Add schooling x AFQT interactions"),
    'add_s1315_gpahs_interactions': (HETERO_GPAHS, "Add schooling x GPA interactions"),
    'add_s1315_parented_interactions': (HETERO_PARENTED, "Add schooling x parent ed interactions"),
}

for het_name, (extra_vars, het_desc) in hetero_specs.items():
    het_rhs = BASELINE_RHS + [v for v in extra_vars if v not in BASELINE_RHS]
    df_het = df_est.dropna(subset=[v for v in extra_vars if v in df_est.columns]).copy()
    if len(df_het) > 100:
        run_spec(
            spec_id=f"rc/controls/heterog/{het_name}",
            spec_tree_path="cross_sectional_ols/rc/controls/heterog",
            baseline_group_id="G1",
            outcome_var="pvlifeincd05",
            treatment_var="m_s16",
            rhs_vars=het_rhs,
            data=df_het,
            vcov_type="HC1",
            sample_desc="Full NLSY79 sample",
            controls_desc=het_desc
        )

# Full heterogeneous spec (Spec C in the paper)
full_hetero = BASELINE_RHS + HETERO_AFQT + HETERO_GPAHS + HETERO_PARENTED
full_hetero = list(dict.fromkeys(full_hetero))  # deduplicate preserving order
df_hetf = df_est.dropna(subset=[v for v in (HETERO_AFQT + HETERO_GPAHS + HETERO_PARENTED) if v in df_est.columns]).copy()
if len(df_hetf) > 100:
    run_spec(
        spec_id="rc/controls/heterog/full_heterogeneous",
        spec_tree_path="cross_sectional_ols/rc/controls/heterog",
        baseline_group_id="G1",
        outcome_var="pvlifeincd05",
        treatment_var="m_s16",
        rhs_vars=full_hetero,
        data=df_hetf,
        vcov_type="HC1",
        sample_desc="Full NLSY79 sample",
        controls_desc="Full heterogeneous returns: add all schooling x ability/GPA/parented interactions"
    )

# ============================================================
# 7. SAMPLE RESTRICTIONS
# ============================================================
print("\n=== Running Sample Restrictions ===")

# Cross-section only (sample <= 8)
df_xsec = df_est[df_est['sample'] <= 8].copy()
if len(df_xsec) > 100:
    run_spec(
        spec_id="rc/sample/restrict/cross_section_only",
        spec_tree_path="cross_sectional_ols/rc/sample/restrict",
        baseline_group_id="G1",
        outcome_var="pvlifeincd05",
        treatment_var="m_s16",
        rhs_vars=BASELINE_RHS,
        data=df_xsec,
        vcov_type="HC1",
        sample_desc="Cross-section sample only (sample<=8)",
        controls_desc="Baseline controls, cross-section sample"
    )

# Cross-section weighted (as in Spec E from earnings_projection.do)
if len(df_xsec) > 100 and 'w79' in df_xsec.columns:
    df_xsec_w = df_xsec.dropna(subset=['w79']).copy()
    if len(df_xsec_w) > 100:
        try:
            rhs_str = " + ".join(BASELINE_RHS)
            formula = f"pvlifeincd05 ~ {rhs_str}"
            m_w = pf.feols(formula, data=df_xsec_w, vcov="hetero", weights="w79")
            spec_run_counter += 1
            run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"
            coef_val = float(m_w.coef().get("m_s16", np.nan))
            se_val = float(m_w.se().get("m_s16", np.nan))
            pval = float(m_w.pvalue().get("m_s16", np.nan))
            ci = m_w.confint()
            ci_lower = float(ci.loc["m_s16", ci.columns[0]]) if "m_s16" in ci.index else np.nan
            ci_upper = float(ci.loc["m_s16", ci.columns[1]]) if "m_s16" in ci.index else np.nan
            nobs = int(m_w._N)
            r2 = float(m_w._r2) if hasattr(m_w, '_r2') else np.nan
            all_coefs = {k: float(v) for k, v in m_w.coef().items()}
            payload = make_success_payload(
                coefficients=all_coefs,
                inference={"spec_id": inference_canonical["spec_id"],
                           "method": "robust", "cluster_vars": []},
                software=SW_BLOCK, surface_hash=SURFACE_HASH,
                design={"cross_sectional_ols": design_audit},
            )
            results.append({
                "paper_id": PAPER_ID, "spec_run_id": run_id,
                "spec_id": "rc/sample/restrict/cross_section_weighted",
                "spec_tree_path": "cross_sectional_ols/rc/sample/restrict",
                "baseline_group_id": "G1",
                "outcome_var": "pvlifeincd05", "treatment_var": "m_s16",
                "coefficient": coef_val, "std_error": se_val, "p_value": pval,
                "ci_lower": ci_lower, "ci_upper": ci_upper,
                "n_obs": nobs, "r_squared": r2,
                "coefficient_vector_json": json.dumps(payload),
                "sample_desc": "Cross-section, weighted (w79)",
                "fixed_effects": "none",
                "controls_desc": "Baseline, cross-section weighted",
                "cluster_var": "", "run_success": 1, "run_error": ""
            })
            print(f"  cross_section_weighted: coef={coef_val:.2f}, N={nobs}")
        except Exception as e:
            print(f"  cross_section_weighted FAILED: {e}")
            spec_run_counter += 1
            run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"
            payload = make_failure_payload(error=str(e)[:240],
                error_details=error_details_from_exception(e, stage="estimation"),
                software=SW_BLOCK, surface_hash=SURFACE_HASH)
            results.append({
                "paper_id": PAPER_ID, "spec_run_id": run_id,
                "spec_id": "rc/sample/restrict/cross_section_weighted",
                "spec_tree_path": "cross_sectional_ols/rc/sample/restrict",
                "baseline_group_id": "G1",
                "outcome_var": "pvlifeincd05", "treatment_var": "m_s16",
                "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
                "ci_lower": np.nan, "ci_upper": np.nan,
                "n_obs": np.nan, "r_squared": np.nan,
                "coefficient_vector_json": json.dumps(payload),
                "sample_desc": "Cross-section, weighted (w79)",
                "fixed_effects": "none",
                "controls_desc": "Baseline, cross-section weighted",
                "cluster_var": "", "run_success": 0, "run_error": str(e)[:240]
            })

# Trim income outliers
pv05 = df_est['pvlifeincd05']
p01, p99 = pv05.quantile(0.01), pv05.quantile(0.99)

df_trim = df_est[(pv05 >= p01) & (pv05 <= p99)].copy()
run_spec(
    spec_id="rc/sample/trim/trim_1_99",
    spec_tree_path="cross_sectional_ols/rc/sample/trim",
    baseline_group_id="G1",
    outcome_var="pvlifeincd05",
    treatment_var="m_s16",
    rhs_vars=BASELINE_RHS,
    data=df_trim,
    vcov_type="HC1",
    sample_desc=f"Trimmed 1-99 percentile of pvlifeincd05 [{p01:.1f}, {p99:.1f}]",
    controls_desc="Baseline controls"
)

df_trim_top = df_est[pv05 <= p99].copy()
run_spec(
    spec_id="rc/sample/trim/drop_top1pct_income",
    spec_tree_path="cross_sectional_ols/rc/sample/trim",
    baseline_group_id="G1",
    outcome_var="pvlifeincd05",
    treatment_var="m_s16",
    rhs_vars=BASELINE_RHS,
    data=df_trim_top,
    vcov_type="HC1",
    sample_desc=f"Drop top 1% of pvlifeincd05 (>{p99:.1f})",
    controls_desc="Baseline controls"
)

df_trim_bot = df_est[pv05 >= p01].copy()
run_spec(
    spec_id="rc/sample/trim/drop_bottom1pct_income",
    spec_tree_path="cross_sectional_ols/rc/sample/trim",
    baseline_group_id="G1",
    outcome_var="pvlifeincd05",
    treatment_var="m_s16",
    rhs_vars=BASELINE_RHS,
    data=df_trim_bot,
    vcov_type="HC1",
    sample_desc=f"Drop bottom 1% of pvlifeincd05 (<{p01:.1f})",
    controls_desc="Baseline controls"
)

# Age subgroups
median_age = df_est['age79'].median()
df_young = df_est[df_est['age79'] <= median_age].copy()
df_old = df_est[df_est['age79'] > median_age].copy()

if len(df_young) > 100:
    run_spec(
        spec_id="rc/sample/subgroup/age79_young",
        spec_tree_path="cross_sectional_ols/rc/sample/subgroup",
        baseline_group_id="G1",
        outcome_var="pvlifeincd05",
        treatment_var="m_s16",
        rhs_vars=BASELINE_RHS,
        data=df_young,
        vcov_type="HC1",
        sample_desc=f"Young cohort (age79<={median_age:.0f})",
        controls_desc="Baseline controls"
    )

if len(df_old) > 100:
    run_spec(
        spec_id="rc/sample/subgroup/age79_old",
        spec_tree_path="cross_sectional_ols/rc/sample/subgroup",
        baseline_group_id="G1",
        outcome_var="pvlifeincd05",
        treatment_var="m_s16",
        rhs_vars=BASELINE_RHS,
        data=df_old,
        vcov_type="HC1",
        sample_desc=f"Old cohort (age79>{median_age:.0f})",
        controls_desc="Baseline controls"
    )

# ============================================================
# 8. RANDOM CONTROL SUBSETS
# ============================================================
print("\n=== Running Random Control Subsets ===")

np.random.seed(42)
for i in range(1, 16):
    # Randomly select ~60% of controls (maintaining symmetry: pick male controls, mirror female)
    male_ctrl_groups = [
        ['m_parented'],
        ['m_black'],
        ['m_latino'],
        ['m_regionnc', 'm_regionso', 'm_regionwe'],
        ['m_urban14'],
        ['m_gpahs'],
        ['m_afqt89'],
        ['m_gpahs_afqt89'],
        ['m_gpahs_parented'],
        ['m_afqt89_parented'],
    ]
    n_keep = np.random.randint(4, 9)
    chosen_idx = np.random.choice(len(male_ctrl_groups), n_keep, replace=False)
    male_chosen = []
    female_chosen = []
    for idx in chosen_idx:
        male_chosen.extend(male_ctrl_groups[idx])
        female_chosen.extend([v.replace('m_', 'f_') for v in male_ctrl_groups[idx]])

    rhs = ['male'] + MALE_SCHOOLING + male_chosen + ['female'] + FEMALE_SCHOOLING + female_chosen
    run_spec(
        spec_id=f"rc/controls/subset/random_{i:03d}",
        spec_tree_path="cross_sectional_ols/rc/controls/subset",
        baseline_group_id="G1",
        outcome_var="pvlifeincd05",
        treatment_var="m_s16",
        rhs_vars=rhs,
        data=df_est,
        vcov_type="HC1",
        sample_desc="Full NLSY79 sample",
        controls_desc=f"Random subset {i}: {n_keep}/10 control groups"
    )

# ============================================================
# 9. ADDITIONAL ROBUSTNESS: alternative treatment variable focus
# ============================================================
print("\n=== Running Alternative Treatment Focus ===")

# Focus on m_s13 (some college 1 year) as treatment
run_spec(
    spec_id="rc/treatment/alt/m_s13_focus",
    spec_tree_path="cross_sectional_ols/rc/treatment",
    baseline_group_id="G1",
    outcome_var="pvlifeincd05",
    treatment_var="m_s13",
    rhs_vars=BASELINE_RHS,
    data=df_est,
    vcov_type="HC1",
    sample_desc="Full NLSY79 sample",
    controls_desc="Baseline, focus on 1yr college premium"
)

# Focus on m_s14 (2 years)
run_spec(
    spec_id="rc/treatment/alt/m_s14_focus",
    spec_tree_path="cross_sectional_ols/rc/treatment",
    baseline_group_id="G1",
    outcome_var="pvlifeincd05",
    treatment_var="m_s14",
    rhs_vars=BASELINE_RHS,
    data=df_est,
    vcov_type="HC1",
    sample_desc="Full NLSY79 sample",
    controls_desc="Baseline, focus on 2yr college premium"
)

# Focus on m_s15 (3 years)
run_spec(
    spec_id="rc/treatment/alt/m_s15_focus",
    spec_tree_path="cross_sectional_ols/rc/treatment",
    baseline_group_id="G1",
    outcome_var="pvlifeincd05",
    treatment_var="m_s15",
    rhs_vars=BASELINE_RHS,
    data=df_est,
    vcov_type="HC1",
    sample_desc="Full NLSY79 sample",
    controls_desc="Baseline, focus on 3yr college premium"
)

# ============================================================
# 10. ADDITIONAL TRIMMING / FUNCTIONAL FORM VARIANTS
# ============================================================
print("\n=== Running Additional Trimming/Functional Form ===")

# Trim at 5/95
p05, p95 = pv05.quantile(0.05), pv05.quantile(0.95)
df_trim_5_95 = df_est[(pv05 >= p05) & (pv05 <= p95)].copy()
run_spec(
    spec_id="rc/sample/trim/trim_5_95",
    spec_tree_path="cross_sectional_ols/rc/sample/trim",
    baseline_group_id="G1",
    outcome_var="pvlifeincd05",
    treatment_var="m_s16",
    rhs_vars=BASELINE_RHS,
    data=df_trim_5_95,
    vcov_type="HC1",
    sample_desc=f"Trimmed 5-95 percentile [{p05:.1f}, {p95:.1f}]",
    controls_desc="Baseline controls"
)

# Drop s13 from RHS (collapse 1yr college into baseline)
rhs_no_s13 = [v for v in BASELINE_RHS if v not in ['m_s13', 'f_s13']]
run_spec(
    spec_id="rc/controls/loo/drop_s13",
    spec_tree_path="cross_sectional_ols/rc/controls/loo",
    baseline_group_id="G1",
    outcome_var="pvlifeincd05",
    treatment_var="m_s16",
    rhs_vars=rhs_no_s13,
    data=df_est,
    vcov_type="HC1",
    sample_desc="Full NLSY79 sample",
    controls_desc="Drop s13 dummies (1yr college absorbed into base)"
)

# Baseline on 10% discount DV with heterogeneous returns
if len(df_hetf) > 100:
    run_spec(
        spec_id="rc/outcome/alt_dv/pvlifeincd10_hetero",
        spec_tree_path="cross_sectional_ols/rc/outcome",
        baseline_group_id="G1",
        outcome_var="pvlifeincd10",
        treatment_var="m_s16",
        rhs_vars=full_hetero,
        data=df_hetf.dropna(subset=['pvlifeincd10']),
        vcov_type="HC1",
        sample_desc="Full NLSY79 sample",
        controls_desc="Alt DV: 10% discount, full heterogeneous returns"
    )

# ============================================================
# 10b. ADDITIONAL: Pooled (non-interacted) specification
# ============================================================
print("\n=== Running Pooled (non-interacted) Specification ===")

# Pooled specification without gender interactions
pooled_rhs = ['s13', 's14', 's15', 's16', 'female', 'parented', 'black', 'latino',
              'regionnc', 'regionso', 'regionwe', 'urban14', 'gpahs', 'afqt89',
              'gpahs_afqt89', 'gpahs_parented', 'afqt89_parented']
df_pooled = df_est.dropna(subset=pooled_rhs).copy()
if len(df_pooled) > 100:
    run_spec(
        spec_id="rc/controls/sets/pooled_no_interactions",
        spec_tree_path="cross_sectional_ols/rc/controls/sets",
        baseline_group_id="G1",
        outcome_var="pvlifeincd05",
        treatment_var="s16",
        rhs_vars=pooled_rhs,
        data=df_pooled,
        vcov_type="HC1",
        sample_desc="Full NLSY79 sample",
        controls_desc="Pooled: no gender interactions, s16 is 4yr college dummy",
        no_intercept=False
    )

# ============================================================
# 11. INFERENCE VARIANTS (on baseline spec)
# ============================================================
print("\n=== Running Inference Variants ===")

# Homoskedastic SEs
rhs_str = " + ".join(BASELINE_RHS)
formula = f"pvlifeincd05 ~ {rhs_str}"

try:
    m_iid = pf.feols(formula, data=df_est, vcov="iid")
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"
    coef_val = float(m_iid.coef().get("m_s16", np.nan))
    se_val = float(m_iid.se().get("m_s16", np.nan))
    pval = float(m_iid.pvalue().get("m_s16", np.nan))
    ci = m_iid.confint()
    ci_lower = float(ci.loc["m_s16", ci.columns[0]]) if "m_s16" in ci.index else np.nan
    ci_upper = float(ci.loc["m_s16", ci.columns[1]]) if "m_s16" in ci.index else np.nan
    nobs = int(m_iid._N)
    r2 = float(m_iid._r2)

    inference_results.append({
        "paper_id": PAPER_ID, "spec_run_id": run_id,
        "spec_id": "infer/homoskedastic",
        "parent_spec_id": "baseline__specA_5pct",
        "baseline_group_id": "G1",
        "outcome_var": "pvlifeincd05", "treatment_var": "m_s16",
        "coefficient": coef_val, "std_error": se_val, "p_value": pval,
        "ci_lower": ci_lower, "ci_upper": ci_upper,
        "n_obs": nobs, "r_squared": r2,
        "se_type": "iid", "cluster_var": "",
        "run_success": 1, "run_error": ""
    })
    print(f"  Homoskedastic: se={se_val:.2f}, p={pval:.4f}")
except Exception as e:
    print(f"  Homoskedastic FAILED: {e}")
    spec_run_counter += 1
    inference_results.append({
        "paper_id": PAPER_ID, "spec_run_id": f"{PAPER_ID}_run_{spec_run_counter:03d}",
        "spec_id": "infer/homoskedastic",
        "parent_spec_id": "baseline__specA_5pct",
        "baseline_group_id": "G1",
        "outcome_var": "pvlifeincd05", "treatment_var": "m_s16",
        "coefficient": np.nan, "std_error": np.nan, "p_value": np.nan,
        "ci_lower": np.nan, "ci_upper": np.nan,
        "n_obs": np.nan, "r_squared": np.nan,
        "se_type": "iid", "cluster_var": "",
        "run_success": 0, "run_error": str(e)[:240]
    })

# HC1 (baseline -- just record for inference comparison)
try:
    m_hc1 = pf.feols(formula, data=df_est, vcov="hetero")
    spec_run_counter += 1
    run_id = f"{PAPER_ID}_run_{spec_run_counter:03d}"
    coef_val = float(m_hc1.coef().get("m_s16", np.nan))
    se_val = float(m_hc1.se().get("m_s16", np.nan))
    pval = float(m_hc1.pvalue().get("m_s16", np.nan))
    ci = m_hc1.confint()
    ci_lower = float(ci.loc["m_s16", ci.columns[0]]) if "m_s16" in ci.index else np.nan
    ci_upper = float(ci.loc["m_s16", ci.columns[1]]) if "m_s16" in ci.index else np.nan
    nobs = int(m_hc1._N)
    r2 = float(m_hc1._r2)

    inference_results.append({
        "paper_id": PAPER_ID, "spec_run_id": run_id,
        "spec_id": "infer/HC1",
        "parent_spec_id": "baseline__specA_5pct",
        "baseline_group_id": "G1",
        "outcome_var": "pvlifeincd05", "treatment_var": "m_s16",
        "coefficient": coef_val, "std_error": se_val, "p_value": pval,
        "ci_lower": ci_lower, "ci_upper": ci_upper,
        "n_obs": nobs, "r_squared": r2,
        "se_type": "HC1", "cluster_var": "",
        "run_success": 1, "run_error": ""
    })
    print(f"  HC1: se={se_val:.2f}, p={pval:.4f}")
except Exception as e:
    print(f"  HC1 FAILED: {e}")

# ============================================================
# Save outputs
# ============================================================
print("\n=== Saving Outputs ===")

spec_df = pd.DataFrame(results)
spec_df.to_csv(f"{OUTPUT_DIR}/specification_results.csv", index=False)
print(f"Wrote specification_results.csv: {len(spec_df)} rows")

infer_df = pd.DataFrame(inference_results) if inference_results else pd.DataFrame()
infer_df.to_csv(f"{OUTPUT_DIR}/inference_results.csv", index=False)
print(f"Wrote inference_results.csv: {len(infer_df)} rows")

# Summary statistics
successful = spec_df[spec_df['run_success'] == 1]
failed = spec_df[spec_df['run_success'] == 0]
print(f"\nSuccessful: {len(successful)}/{len(spec_df)}")
print(f"Failed: {len(failed)}/{len(spec_df)}")

if len(successful) > 0:
    base_row = successful[successful['spec_id'] == 'baseline__specA_5pct']
    if len(base_row) > 0:
        b = base_row.iloc[0]
        print(f"\nBaseline: coef={b['coefficient']:.4f}, se={b['std_error']:.4f}, "
              f"p={b['p_value']:.4f}, N={b['n_obs']:.0f}, R2={b['r_squared']:.4f}")

    print(f"\nCoefficient range (m_s16): [{successful['coefficient'].min():.4f}, {successful['coefficient'].max():.4f}]")
    print(f"p-value range: [{successful['p_value'].min():.4f}, {successful['p_value'].max():.4f}]")
    n_sig = (successful['p_value'] < 0.05).sum()
    print(f"Significant at 5%: {n_sig}/{len(successful)}")

# ============================================================
# Write SPECIFICATION_SEARCH.md
# ============================================================

md_lines = []
md_lines.append("# Specification Search Report")
md_lines.append(f"**Paper:** 113812-V1 (Stange 2012, AEJ: Applied)")
md_lines.append(f"**Title:** An Empirical Investigation of the Option Value of College Enrollment")
md_lines.append("")
md_lines.append("## Design")
md_lines.append("- **Method:** Cross-sectional OLS (Mincer-style lifetime income regression)")
md_lines.append("- **Data:** NLSY79 (National Longitudinal Survey of Youth 1979)")
md_lines.append("- **Outcome:** pvlifeincd05 (PV lifetime income, 5% discount, thousands of 1992$)")
md_lines.append("- **Treatment:** m_s16 (male 4-year college completion dummy)")
md_lines.append(f"- **Controls:** {len(BASELINE_RHS)} gender-interacted regressors (schooling, demographics, ability)")
md_lines.append("- **Standard errors:** HC1 (robust)")
md_lines.append("")

if len(successful) > 0 and len(base_row) > 0:
    bc = base_row.iloc[0]
    md_lines.append("## Baseline Result")
    md_lines.append("")
    md_lines.append(f"| Statistic | Value |")
    md_lines.append(f"|-----------|-------|")
    md_lines.append(f"| Coefficient | {bc['coefficient']:.4f} |")
    md_lines.append(f"| Std. Error | {bc['std_error']:.4f} |")
    md_lines.append(f"| p-value | {bc['p_value']:.6f} |")
    md_lines.append(f"| 95% CI | [{bc['ci_lower']:.4f}, {bc['ci_upper']:.4f}] |")
    md_lines.append(f"| N | {bc['n_obs']:.0f} |")
    md_lines.append(f"| R-squared | {bc['r_squared']:.4f} |")
    md_lines.append("")

md_lines.append("## Specification Counts")
md_lines.append("")
md_lines.append(f"- Total specifications: {len(spec_df)}")
md_lines.append(f"- Successful: {len(successful)}")
md_lines.append(f"- Failed: {len(failed)}")
md_lines.append(f"- Inference variants: {len(infer_df)}")
md_lines.append("")

# Category breakdown
md_lines.append("## Category Breakdown")
md_lines.append("")
md_lines.append("| Category | Count | Sig. (p<0.05) | Coef Range |")
md_lines.append("|----------|-------|---------------|------------|")

categories = {
    "Baseline": successful[successful['spec_id'].str.startswith('baseline')],
    "Controls LOO": successful[successful['spec_id'].str.startswith('rc/controls/loo/')],
    "Controls Progression": successful[successful['spec_id'].str.startswith('rc/controls/progression/')],
    "Controls Sets": successful[successful['spec_id'].str.startswith('rc/controls/sets/')],
    "Alt DVs": successful[successful['spec_id'].str.startswith('rc/outcome/')],
    "Heterog Returns": successful[successful['spec_id'].str.startswith('rc/controls/heterog/')],
    "Sample Restrictions": successful[successful['spec_id'].str.startswith('rc/sample/')],
    "Control Subsets": successful[successful['spec_id'].str.startswith('rc/controls/subset/')],
    "Alt Treatment": successful[successful['spec_id'].str.startswith('rc/treatment/')],
}

for cat_name, cat_df in categories.items():
    if len(cat_df) > 0:
        n_sig_cat = (cat_df['p_value'] < 0.05).sum()
        coef_range = f"[{cat_df['coefficient'].min():.4f}, {cat_df['coefficient'].max():.4f}]"
        md_lines.append(f"| {cat_name} | {len(cat_df)} | {n_sig_cat}/{len(cat_df)} | {coef_range} |")

md_lines.append("")

# Inference variants
md_lines.append("## Inference Variants")
md_lines.append("")
if len(infer_df) > 0:
    md_lines.append("| Spec ID | SE | p-value | 95% CI |")
    md_lines.append("|---------|-----|---------|--------|")
    for _, row in infer_df.iterrows():
        if row['run_success'] == 1:
            md_lines.append(f"| {row['spec_id']} | {row['std_error']:.4f} | {row['p_value']:.6f} | [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}] |")
        else:
            md_lines.append(f"| {row['spec_id']} | FAILED | - | - |")

md_lines.append("")

# Overall assessment
md_lines.append("## Overall Assessment")
md_lines.append("")
if len(successful) > 0:
    # Filter to only m_s16-focused specs for sign/significance assessment
    s16_specs = successful[successful['treatment_var'] == 'm_s16']
    if len(s16_specs) > 0:
        n_sig_total = (s16_specs['p_value'] < 0.05).sum()
        pct_sig = n_sig_total / len(s16_specs) * 100
        sign_consistent = ((s16_specs['coefficient'] > 0).sum() == len(s16_specs)) or \
                          ((s16_specs['coefficient'] < 0).sum() == len(s16_specs))
        median_coef = s16_specs['coefficient'].median()
        sign_word = "positive" if median_coef > 0 else "negative"

        md_lines.append(f"- **Sign consistency (m_s16 specs):** {'All specifications have the same sign' if sign_consistent else 'Mixed signs across specifications'}")
        md_lines.append(f"- **Significance stability:** {n_sig_total}/{len(s16_specs)} ({pct_sig:.1f}%) specifications significant at 5%")
        md_lines.append(f"- **Direction:** Median coefficient is {sign_word} ({median_coef:.4f})")

        if pct_sig >= 80 and sign_consistent:
            strength = "STRONG"
        elif pct_sig >= 50 and sign_consistent:
            strength = "MODERATE"
        elif pct_sig >= 30:
            strength = "WEAK"
        else:
            strength = "FRAGILE"

        md_lines.append(f"- **Robustness assessment:** {strength}")

md_lines.append("")
md_lines.append(f"Surface hash: `{SURFACE_HASH}`")
md_lines.append("")

with open(f"{OUTPUT_DIR}/SPECIFICATION_SEARCH.md", "w") as f:
    f.write("\n".join(md_lines))

print(f"\nWrote SPECIFICATION_SEARCH.md")
print(f"\n=== Pipeline Complete for {PAPER_ID} ===")
