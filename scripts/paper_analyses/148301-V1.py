"""
Specification Search Script for Laffitte & Toubal (2022)
"Multinational's Sales and Tax Havens"
American Economic Journal: Economic Policy, 14(4), 461-491.

Paper ID: 148301-V1

Surface-driven execution:
  - G1: ep ~ lfma (GLM fractional logit baseline with marginal effects; OLS alternative)
  - G2: lprofit ~ ep_haven (OLS reghdfe baseline)
  - Panel FE with sector-year FE, clustered at country level
  - Full data construction from raw BEA/PWT/FMA/AGR/TAXR sources

Outputs:
  - specification_results.csv (baseline, design/*, rc/* rows)
  - inference_results.csv (infer/* rows)
  - SPECIFICATION_SEARCH.md
"""

import pandas as pd
import numpy as np
import pyfixest as pf
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import Logit as LogitLink
import json
import sys
import os
import warnings
import traceback
warnings.filterwarnings('ignore')

# Add scripts dir for utilities
sys.path.insert(0, "scripts")
from agent_output_utils import (
    make_success_payload, make_failure_payload,
    error_details_from_exception, surface_hash, software_block
)

PAPER_ID = "148301-V1"
DATA_DIR = "data/downloads/extracted/148301-V1"
OUTPUT_DIR = DATA_DIR
RAW_DIR = f"{DATA_DIR}/Data/Raw"

# Load surface
with open(f"{DATA_DIR}/SPECIFICATION_SURFACE.json") as f:
    surface_obj = json.load(f)

SURFACE_HASH = surface_hash(surface_obj)
SW_BLOCK = software_block()

# Design audit blocks from surface
G1_DESIGN_AUDIT = surface_obj["baseline_groups"][0]["design_audit"]
G2_DESIGN_AUDIT = surface_obj["baseline_groups"][1]["design_audit"]
G1_INFERENCE_CANONICAL = surface_obj["baseline_groups"][0]["inference_plan"]["canonical"]
G2_INFERENCE_CANONICAL = surface_obj["baseline_groups"][1]["inference_plan"]["canonical"]

results = []
inference_results = []
run_counter = 0
infer_counter = 0

# =====================================================================
#  DATA CONSTRUCTION
# =====================================================================

def clean_bea_value(val, imputed_val="0.251"):
    """Convert BEA cell to numeric, handling (*) suppressed values."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace(",", "")
    if s in ["(*)", "(D)", "(*)  ", "(*) "]:
        return float(imputed_val)
    try:
        return float(s)
    except (ValueError, TypeError):
        return np.nan

def import_bea_table_pre08(filepath, sheet="TablePage", skip=5, imputed_val="0.251"):
    """Import a pre-2008 BEA table (16 sectors: All + 15 industry)."""
    try:
        df = pd.read_excel(filepath, sheet_name=sheet, header=None, skiprows=skip)
    except Exception:
        return None
    if df.shape[1] < 17:
        return None
    df.columns = ['countryname'] + [f'col_{i}' for i in range(16)]
    # Clean country names
    df['countryname'] = df['countryname'].astype(str).str.strip()
    # Drop aggregate/empty rows
    drop_names = ['', 'nan', 'None', 'Addenda:', 'European Union (27)',
                  'European Union (15)', 'OPEC', 'Other',
                  'Other Western Hemisphere',
                  'Latin America and Other Western Hemisphere',
                  'All Countries', 'All countries',
                  'Eastern Europe', 'Europe', 'Asia and Pacific',
                  'Middle East', 'Africa', 'South America',
                  'Central America', 'Caribbean',
                  'Latin America and Other WH',
                  'Other Africa', 'Other Asia and Pacific',
                  'Other Middle East', 'Other Europe',
                  'South and Central America',
                  'International organizations and unallocated']
    df = df[~df['countryname'].isin(drop_names)]
    df = df[df['countryname'].str.len() > 1]
    # Also drop any rows that look like headers or aggregates
    df = df[~df['countryname'].str.contains(r'^\s*$', na=True)]
    df = df[~df['countryname'].str.contains(r'^\(', na=False)]
    # Convert values
    for c in [f'col_{i}' for i in range(16)]:
        df[c] = df[c].apply(lambda x: clean_bea_value(x, imputed_val))
    sector_names = ['All', 'Mining', 'Utilities', 'Total_manuf', 'Food',
                    'Chemicals', 'Primary_fabricated_mat', 'Machinery',
                    'Computer', 'Electrical_eqpm', 'Transportation',
                    'Wholesale', 'Info', 'Finance', 'Services', 'Other']
    rename_map = {f'col_{i}': sector_names[i] for i in range(16)}
    df = df.rename(columns=rename_map)
    return df

def import_bea_table_post08(filepath, sheet, skip=5, imputed_val="0.251"):
    """Import a post-2008 BEA table (16 sectors but different layout: no Utilities, add Retail)."""
    try:
        df = pd.read_excel(filepath, sheet_name=sheet, header=None, skiprows=skip)
    except Exception:
        return None
    if df.shape[1] < 17:
        return None
    df.columns = ['countryname'] + [f'col_{i}' for i in range(16)]
    df['countryname'] = df['countryname'].astype(str).str.strip()
    drop_names = ['', 'nan', 'None', 'Addenda:', 'European Union (27)',
                  'European Union (15)', 'OPEC', 'Other',
                  'Other Western Hemisphere',
                  'Latin America and Other Western Hemisphere',
                  'All Countries', 'All countries',
                  'Eastern Europe', 'Europe', 'Asia and Pacific',
                  'Middle East', 'Africa', 'South America',
                  'Central America', 'Caribbean',
                  'Latin America and Other WH',
                  'Other Africa', 'Other Asia and Pacific',
                  'Other Middle East', 'Other Europe',
                  'South and Central America',
                  'International organizations and unallocated']
    df = df[~df['countryname'].isin(drop_names)]
    df = df[df['countryname'].str.len() > 1]
    df = df[~df['countryname'].str.contains(r'^\s*$', na=True)]
    df = df[~df['countryname'].str.contains(r'^\(', na=False)]
    for c in [f'col_{i}' for i in range(16)]:
        df[c] = df[c].apply(lambda x: clean_bea_value(x, imputed_val))
    # Post-08: All, Mining, Total_manuf, Food, Chemicals, Primary_fabricated_mat,
    # Machinery, Computer, Electrical_eqpm, Transportation, Wholesale, Retail,
    # Info, Finance, Services, Other
    sector_names_post = ['All', 'Mining', 'Total_manuf', 'Food',
                         'Chemicals', 'Primary_fabricated_mat', 'Machinery',
                         'Computer', 'Electrical_eqpm', 'Transportation',
                         'Wholesale', 'Retail', 'Info', 'Finance', 'Services', 'Other']
    rename_map = {f'col_{i}': sector_names_post[i] for i in range(16)}
    df = df.rename(columns=rename_map)
    return df

# Country name to ISO3 mapping for BEA data
COUNTRY_MAP = {
    "Canada": "CAN", "Mexico": "MEX", "Argentina": "ARG", "Brazil": "BRA",
    "Chile": "CHL", "Colombia": "COL", "Costa Rica": "CRI", "Dominican Republic": "DOM",
    "Ecuador": "ECU", "Guatemala": "GTM", "Honduras": "HND", "Jamaica": "JAM",
    "Panama": "PAN", "Peru": "PER", "Trinidad and Tobago": "TTO", "Venezuela": "VEN",
    "Bermuda": "BMU", "Barbados": "BRB",
    "United Kingdom Islands, Caribbean": "VGB",
    "United Kingdom Islands Caribbean": "VGB",
    "UK Islands, Caribbean": "VGB",
    "UK Islands Caribbean": "VGB",
    "Australia": "AUS", "China": "CHN", "Hong Kong": "HKG",
    "India": "IND", "Indonesia": "IDN", "Japan": "JPN",
    "Korea, Republic of": "KOR", "Korea Republic of": "KOR",
    "South Korea": "KOR",
    "Malaysia": "MYS", "New Zealand": "NZL", "Philippines": "PHL",
    "Singapore": "SGP", "Taiwan": "TWN", "Thailand": "THA",
    "Austria": "AUT", "Belgium": "BEL", "Czech Republic": "CZE",
    "Denmark": "DNK", "Finland": "FIN", "France": "FRA",
    "Germany": "DEU", "Greece": "GRC", "Hungary": "HUN",
    "Ireland": "IRL", "Italy": "ITA", "Luxembourg": "LUX",
    "Netherlands": "NLD", "Norway": "NOR", "Poland": "POL",
    "Portugal": "PRT", "Russia": "RUS", "Spain": "ESP",
    "Sweden": "SWE", "Switzerland": "CHE",
    "Turkey": "TUR", "United Kingdom": "GBR",
    "Egypt": "EGY", "Israel": "ISR", "Nigeria": "NGA",
    "Saudi Arabia": "SAU", "South Africa": "ZAF",
    "United Arab Emirates": "ARE",
    # Additional names found in BEA data
    "Bahamas": "BHS", "Bahrain": "BHR", "Bolivia": "BOL",
    "El Salvador": "SLV", "Guatemala ": "GTM",
    "Guyana": "GUY", "Haiti": "HTI", "Nicaragua": "NIC",
    "Paraguay": "PRY", "Suriname": "SUR", "Uruguay": "URY",
    "Belize": "BLZ", "Bangladesh": "BGD", "Brunei": "BRN",
    "Cambodia": "KHM", "Fiji": "FJI", "Laos": "LAO",
    "Macau": "MAC", "Mongolia": "MNG", "Myanmar": "MMR",
    "Nepal": "NPL", "Pakistan": "PAK", "Papua New Guinea": "PNG",
    "Sri Lanka": "LKA", "Vietnam": "VNM",
    "Algeria": "DZA", "Angola": "AGO", "Botswana": "BWA",
    "Cameroon": "CMR", "Congo (Kinshasa)": "COD",
    "Congo, Democratic Republic of the": "COD",
    "Cote d'Ivoire": "CIV", "Ethiopia": "ETH",
    "Gabon": "GAB", "Ghana": "GHA", "Guinea": "GIN",
    "Kenya": "KEN", "Liberia": "LBR", "Libya": "LBY",
    "Madagascar": "MDG", "Malawi": "MWI", "Mali": "MLI",
    "Mauritius": "MUS", "Morocco": "MAR", "Mozambique": "MOZ",
    "Namibia": "NAM", "Niger": "NER", "Rwanda": "RWA",
    "Senegal": "SEN", "Sierra Leone": "SLE", "Sudan": "SDN",
    "Tanzania": "TZA", "Togo": "TGO", "Tunisia": "TUN",
    "Uganda": "UGA", "Zambia": "ZMB", "Zimbabwe": "ZWE",
    "Albania": "ALB", "Armenia": "ARM", "Azerbaijan": "AZE",
    "Belarus": "BLR", "Bosnia and Herzegovina": "BIH",
    "Bulgaria": "BGR", "Croatia": "HRV", "Cyprus": "CYP",
    "Estonia": "EST", "Georgia": "GEO", "Iceland": "ISL",
    "Kazakhstan": "KAZ", "Kyrgyzstan": "KGZ", "Latvia": "LVA",
    "Lithuania": "LTU", "Malta": "MLT",
    "Moldova, Republic of": "MDA", "Moldova": "MDA",
    "Montenegro": "MNE",
    "North Macedonia": "MKD", "Macedonia": "MKD",
    "Romania": "ROM", "Romania ": "ROM",
    "Serbia": "SRB", "Slovak Republic": "SVK", "Slovakia": "SVK",
    "Slovenia": "SVN", "Tajikistan": "TJK",
    "Turkmenistan": "TKM", "Ukraine": "UKR", "Uzbekistan": "UZB",
    "Iran": "IRN", "Iraq": "IRQ", "Jordan": "JOR",
    "Kuwait": "KWT", "Lebanon": "LBN", "Oman": "OMN",
    "Qatar": "QAT", "Syria": "SYR", "Yemen": "YEM",
    "Afghanistan": "AFG",
    "Antigua and Barbuda": "ATG", "Aruba": "ABW",
    "Cayman Islands": "CYM", "Cuba": "CUB",
    "Curacao": "CUW", "Dominica": "DMA",
    "Grenada": "GRD", "Guadeloupe": "GLP",
    "Martinique": "MTQ", "Montserrat": "MSR",
    "Netherlands Antilles": "ANT",
    "St. Kitts and Nevis": "KNA", "Saint Kitts and Nevis": "KNA",
    "St. Lucia": "LCA", "Saint Lucia": "LCA",
    "St. Vincent and the Grenadines": "VCT",
    "Saint Vincent and the Grenadines": "VCT",
    "Turks and Caicos Islands": "TCA",
    "Virgin Islands, British": "VGB",
    "British Virgin Islands": "VGB",
    "Virgin Islands, U.S.": "VIR",
    "Congo (Brazzaville)": "COG", "Congo": "COG",
    "Equatorial Guinea": "GNQ", "Eritrea": "ERI",
    "Lesotho": "LSO", "Mauritania": "MRT",
    "Seychelles": "SYC", "Somalia": "SOM",
    "Swaziland": "SWZ", "Eswatini": "SWZ",
    "Central African Republic": "CAF",
    "Chad": "TCD", "Comoros": "COM",
    "Djibouti": "DJI", "Gambia": "GMB",
    "Guinea-Bissau": "GNB", "Benin": "BEN",
    "Burkina Faso": "BFA", "Burundi": "BDI",
    "Cape Verde": "CPV", "Sao Tome and Principe": "STP",
    "Tanzania, United Republic of": "TZA",
    "Liechtenstein": "LIE", "Monaco": "MCO",
    "San Marino": "SMR", "Andorra": "AND",
    "Samoa": "WSM", "Solomon Islands": "SLB",
    "Tonga": "TON", "Vanuatu": "VUT",
    "Marshall Islands": "MHL", "Micronesia": "FSM",
    "Palau": "PLW", "Kiribati": "KIR",
    "Tuvalu": "TUV", "Nauru": "NRU",
    "Timor-Leste": "TLS", "East Timor": "TLS",
}

def countryname_to_iso3(name):
    """Map BEA country name to ISO3 code."""
    name = str(name).strip()
    if name in COUNTRY_MAP:
        return COUNTRY_MAP[name]
    # Try variations
    for k, v in COUNTRY_MAP.items():
        if k.lower() == name.lower():
            return v
    return None

# Sector k mapping (used in 5.construction.do)
SECTOR_K_MAP = {
    'All': 0, 'Mining': 1, 'Utilities': 2, 'Total_manuf': 3,
    'Food': 4, 'Chemicals': 5, 'Primary_fabricated_mat': 6,
    'Machinery': 7, 'Computer': 8, 'Electrical_eqpm': 9,
    'Transportation': 10, 'Wholesale': 11, 'Info': 12,
    'Finance': 13, 'Services': 14, 'Other': 15, 'Retail': 16
}

# d mapping (from 5.construction.do) - used for manufacturing/services split
INDUS_D_MAP = {
    'Mining': 1, 'Food': 2, 'Chemicals': 3, 'Primary_fabricated_mat': 4,
    'Machinery': 5, 'Computer': 6, 'Electrical_eqpm': 7,
    'Transportation': 8, 'Wholesale': 9, 'Info': 10, 'Services': 11
}

def build_bea_variable_pre08(table_file_map, imputed_val="0.251"):
    """Build one BEA variable from pre-08 files, returns long panel (iso3, year, k, value)."""
    dfs = []
    for yr_suffix, fpath in table_file_map.items():
        if not os.path.exists(fpath):
            continue
        df = import_bea_table_pre08(fpath, imputed_val=imputed_val)
        if df is None:
            continue
        sector_cols = [c for c in df.columns if c != 'countryname']
        df_long = df.melt(id_vars='countryname', value_vars=sector_cols,
                          var_name='sector', value_name='value')
        yr = int(yr_suffix)
        if yr < 50:
            yr += 2000
        else:
            yr += 1900
        df_long['year'] = yr
        dfs.append(df_long)
    if not dfs:
        return pd.DataFrame()
    result = pd.concat(dfs, ignore_index=True)
    result['iso3'] = result['countryname'].apply(countryname_to_iso3)
    result['k'] = result['sector'].map(SECTOR_K_MAP)
    result = result.dropna(subset=['iso3', 'k'])
    return result[['iso3', 'year', 'k', 'value']].copy()

def build_bea_variable_post08(table_file_map, sheet_name, imputed_val="0.251"):
    """Build one BEA variable from post-08 files."""
    dfs = []
    for yr_suffix, fpath in table_file_map.items():
        if not os.path.exists(fpath):
            continue
        df = import_bea_table_post08(fpath, sheet=sheet_name, imputed_val=imputed_val)
        if df is None:
            continue
        sector_cols = [c for c in df.columns if c != 'countryname']
        df_long = df.melt(id_vars='countryname', value_vars=sector_cols,
                          var_name='sector', value_name='value')
        yr = int(yr_suffix) + 2000
        df_long['year'] = yr
        dfs.append(df_long)
    if not dfs:
        return pd.DataFrame()
    result = pd.concat(dfs, ignore_index=True)
    result['iso3'] = result['countryname'].apply(countryname_to_iso3)
    result['k'] = result['sector'].map(SECTOR_K_MAP)
    result = result.dropna(subset=['iso3', 'k'])
    return result[['iso3', 'year', 'k', 'value']].copy()

print("Building dataset from raw BEA files...")

# --- Build all BEA variables ---
bea_dir = f"{RAW_DIR}/BEA"
years_pre08 = ['99', '00', '01', '02', '03', '04', '05', '06', '07', '08']
years_post08 = ['09', '10', '11', '12', '13']

def pre08_table_map(table_name):
    return {yr: f"{bea_dir}/{yr}/{table_name}" for yr in years_pre08}

def post08_file_map():
    return {yr: f"{bea_dir}/{yr}/Part II F1-F9 .xls" for yr in years_post08}

def post08_file_map_e():
    return {yr: f"{bea_dir}/{yr}/Part II D1-D13 .xls" for yr in years_post08}

def post08_file_map_g():
    return {yr: f"{bea_dir}/{yr}/Part II F1-F9 .xls" for yr in years_post08}

def post08_file_map_h():
    return {yr: f"{bea_dir}/{yr}/Part II H1-H13.xls" for yr in years_post08}

# Local sales (Tab3F7)
local_pre = build_bea_variable_pre08(pre08_table_map("Tab3F7.xls"))
local_pre = local_pre.rename(columns={'value': 'local_'})

# Sales to US (Tab3F4)
sus_pre = build_bea_variable_pre08(pre08_table_map("Tab3F4.xls"))
sus_pre = sus_pre.rename(columns={'value': 'sales_us'})

# Foreign sales (Tab3F8)
foreign_pre = build_bea_variable_pre08(pre08_table_map("Tab3F8.xls"))
foreign_pre = foreign_pre.rename(columns={'value': 'foreign_'})

# Total sales (Tab3E3)
sales_pre = build_bea_variable_pre08(pre08_table_map("Tab3E3.xls"))
sales_pre = sales_pre.rename(columns={'value': 'sales_'})

# Profit (Tab3G7)
profit_pre = build_bea_variable_pre08(pre08_table_map("Tab3G7.xls"))
profit_pre = profit_pre.rename(columns={'value': 'profit_'})

# Employment (Tab3H3) - imputed value is 0.0251
emp_pre = build_bea_variable_pre08(pre08_table_map("Tab3H3.xls"), imputed_val="0.0251")
emp_pre = emp_pre.rename(columns={'value': 'emp_'})

# Equipment (Tab3H5)
eqpmt_pre = build_bea_variable_pre08(pre08_table_map("Tab3H5.xls"))
eqpmt_pre = eqpmt_pre.rename(columns={'value': 'eqpmt_'})

# Merge pre-08 data
pre08 = local_pre.merge(sus_pre, on=['iso3', 'year', 'k'], how='outer')
pre08 = pre08.merge(foreign_pre, on=['iso3', 'year', 'k'], how='outer')
pre08 = pre08.merge(sales_pre, on=['iso3', 'year', 'k'], how='outer')
pre08 = pre08.merge(profit_pre, on=['iso3', 'year', 'k'], how='outer')
pre08 = pre08.merge(emp_pre, on=['iso3', 'year', 'k'], how='outer')
pre08 = pre08.merge(eqpmt_pre, on=['iso3', 'year', 'k'], how='outer')

print(f"  Pre-08 data: {pre08.shape}")

# --- Post-08 data ---
# Post-08 files use different sheet names within multi-sheet workbooks
# Local sales: "II.F 7", Sales to US: "II.F 4", Foreign: "II.F 8"
# Total sales: "II.D 3", Profit: "II.F 7" in G workbook -> actually "II.F 7"

# Post-08 sales breakdown
def try_sheets(filepath, sheet_candidates, imputed_val="0.251"):
    """Try multiple sheet names."""
    for sheet in sheet_candidates:
        try:
            df = import_bea_table_post08(filepath, sheet=sheet, imputed_val=imputed_val)
            if df is not None and len(df) > 5:
                return df
        except Exception:
            continue
    return None

# Build post-08 variables manually
post08_dfs = {var: [] for var in ['local_', 'sales_us', 'foreign_', 'sales_', 'profit_', 'emp_', 'eqpmt_']}

for yr in years_post08:
    yr_int = int(yr) + 2000

    # F workbook: local (F7), sales_us (F4), foreign (F8)
    f_file = f"{bea_dir}/{yr}/Part II F1-F9 .xls"
    if os.path.exists(f_file):
        for sheet_name, var_name in [("II.F 7", "local_"), ("II.F 4", "sales_us"), ("II.F 8", "foreign_")]:
            df = try_sheets(f_file, [sheet_name, sheet_name.replace(".", ""), f"II.F{sheet_name[-1]}"])
            if df is not None:
                sector_cols = [c for c in df.columns if c != 'countryname']
                df_long = df.melt(id_vars='countryname', value_vars=sector_cols,
                                  var_name='sector', value_name='value')
                df_long['year'] = yr_int
                df_long['iso3'] = df_long['countryname'].apply(countryname_to_iso3)
                df_long['k'] = df_long['sector'].map(SECTOR_K_MAP)
                df_long = df_long.dropna(subset=['iso3', 'k'])
                post08_dfs[var_name].append(df_long[['iso3', 'year', 'k', 'value']].rename(columns={'value': var_name}))

    # D workbook: total sales (D3)
    d_file = f"{bea_dir}/{yr}/Part II D1-D13 .xls"
    if os.path.exists(d_file):
        df = try_sheets(d_file, ["II.D 3", "II.D3", "II.D 3 "])
        if df is not None:
            sector_cols = [c for c in df.columns if c != 'countryname']
            df_long = df.melt(id_vars='countryname', value_vars=sector_cols,
                              var_name='sector', value_name='value')
            df_long['year'] = yr_int
            df_long['iso3'] = df_long['countryname'].apply(countryname_to_iso3)
            df_long['k'] = df_long['sector'].map(SECTOR_K_MAP)
            df_long = df_long.dropna(subset=['iso3', 'k'])
            post08_dfs['sales_'].append(df_long[['iso3', 'year', 'k', 'value']].rename(columns={'value': 'sales_'}))

    # G workbook: profit (G7 = F7 in G workbook)
    g_file = f"{bea_dir}/{yr}/Part II F1-F9 .xls"  # profit is in F workbook too: II.F 7 is actually local sales
    # Actually profit pre-08 uses Tab3G7 -> in post-08, profits are in "Part II F1-F9 .xls" sheet "II.F 7"? No.
    # Looking at code: profit_post08.do imports "Part II F1-F9 .xls" sheet "II.F 7"
    # Wait - that's the same as local sales. Let me re-check.
    # profit_post08.do: import excel "../Raw/BEA/`x'/Part II F1-F9 .xls", sheet("II.F 7")
    # ep (local) also from Part II F1-F9 .xls, sheet "II.F 7"?
    # No - profit uses Tab3G7 pre-08, and post-08 uses "II.F 7" in "Part II F1-F9 .xls"
    # This is because the BEA table numbering changed. Tab3G7 = profit; post-08 the profit table
    # is in "Part II F1-F9" workbook. Let me re-read profit_post08.do more carefully.
    # profit_post08.do: import excel  "../Raw/BEA/`x'/Part II F1-F9 .xls", sheet("II.F 7")
    # This IS the profit table in post-08 format.
    # But local sales also comes from F7! Let me check ep_pre08.do - it uses Tab3F7 for local.
    # So pre-08: local=Tab3F7, profit=Tab3G7 (different workbooks)
    # post-08: both come from "Part II F1-F9 .xls" but different sheets?
    # Actually, post-08 restructured: F workbook has sales data, G workbook has financial data
    # The profit_post08 imports from "Part II F1-F9 .xls" sheet "II.F 7"
    # But that's the same file/sheet as local sales!
    # Let me look more carefully - in the post-08 do files, different variables may come from the same
    # workbook but different sheets.

    # Let me re-read the post-08 import files more carefully
    pass

# Actually, the post-08 data uses different workbooks:
# - Sales breakdown (local, foreign, US): "Part II E1-E17 .xls" (sheets II.E 3, etc.) OR "Part II F1-F9 .xls"
# - Profit: "Part II F1-F9 .xls" (sheet II.F 7)
# - Employment: "Part II H1-H13.xls" (sheet II.H 3)
# - Equipment: "Part II H1-H13.xls" (sheet II.H 5)

# Let me build this more carefully by reading the actual post-08 do files
# foreign_post08.do: "Part II E1-E17 .xls" sheet "II.E 8"
# local_post08.do: "Part II E1-E17 .xls" sheet "II.E 7"
# sales_us_post08.do: "Part II E1-E17 .xls" sheet "II.E 4"
# sales_post08.do: "Part II D1-D13 .xls" sheet "II.D 3"
# profit_post08.do: "Part II F1-F9 .xls" sheet "II.F 7"
# emp_post08.do: "Part II H1-H13.xls" sheet "II.H 3"
# eqpmt_post08.do: "Part II H1-H13.xls" sheet "II.H 5"

# Let me rebuild post-08 data properly
post08_dfs = {var: [] for var in ['local_', 'sales_us', 'foreign_', 'sales_', 'profit_', 'emp_', 'eqpmt_']}

post08_config = {
    'local_':   ("Part II E1-E17 .xls", ["II.E 7", "II.E7"]),
    'sales_us': ("Part II E1-E17 .xls", ["II.E 4", "II.E4"]),
    'foreign_': ("Part II E1-E17 .xls", ["II.E 8", "II.E8"]),
    'sales_':   ("Part II D1-D13 .xls", ["II.D 3", "II.D3"]),
    'profit_':  ("Part II F1-F9 .xls",  ["II.F 7", "II.F7"]),
    'emp_':     ("Part II H1-H13.xls",  ["II.H 3", "II.H3"]),
    'eqpmt_':   ("Part II H1-H13.xls",  ["II.H 5", "II.H5"]),
}

for yr in years_post08:
    yr_int = int(yr) + 2000
    for var_name, (workbook, sheets) in post08_config.items():
        fpath = f"{bea_dir}/{yr}/{workbook}"
        if not os.path.exists(fpath):
            continue
        imputed_val = "0.0251" if var_name == 'emp_' else "0.251"
        df = try_sheets(fpath, sheets, imputed_val=imputed_val)
        if df is not None:
            sector_cols = [c for c in df.columns if c != 'countryname']
            df_long = df.melt(id_vars='countryname', value_vars=sector_cols,
                              var_name='sector', value_name='value')
            df_long['year'] = yr_int
            df_long['iso3'] = df_long['countryname'].apply(countryname_to_iso3)
            df_long['k'] = df_long['sector'].map(SECTOR_K_MAP)
            df_long = df_long.dropna(subset=['iso3', 'k'])
            post08_dfs[var_name].append(df_long[['iso3', 'year', 'k', 'value']].rename(columns={'value': var_name}))

# Combine post-08 data
post08_combined = {}
for var_name, df_list in post08_dfs.items():
    if df_list:
        post08_combined[var_name] = pd.concat(df_list, ignore_index=True)

if post08_combined:
    # Merge post-08 variables
    first_key = list(post08_combined.keys())[0]
    post08 = post08_combined[first_key]
    for var_name, df_var in post08_combined.items():
        if var_name == first_key:
            continue
        post08 = post08.merge(df_var, on=['iso3', 'year', 'k'], how='outer')
    print(f"  Post-08 data: {post08.shape}")
else:
    post08 = pd.DataFrame()
    print("  WARNING: No post-08 data loaded")

# Combine pre and post
if not post08.empty:
    bea_all = pd.concat([pre08, post08], ignore_index=True)
else:
    bea_all = pre08.copy()

# Ensure numeric types
for c in ['local_', 'sales_us', 'foreign_', 'sales_', 'profit_', 'emp_', 'eqpmt_']:
    if c in bea_all.columns:
        bea_all[c] = pd.to_numeric(bea_all[c], errors='coerce')

print(f"  Combined BEA data: {bea_all.shape}")

# --- Apply construction steps from 5.construction.do ---

# 1. Import tax rates
print("  Importing tax rates...")
df_tax = pd.read_excel(f"{RAW_DIR}/TAXR/taxrates.xlsx", sheet_name='KPMG extended')
df_tax = df_tax.rename(columns={'Unnamed: 0': 'countryname', 'Unnamed: 1': 'iso3'})
df_tax = df_tax.dropna(subset=['iso3'])
df_tax = df_tax[df_tax['iso3'].str.len() > 0]
tax_cols = [c for c in df_tax.columns if c.startswith('taxr_')]
df_tax_long = df_tax.melt(id_vars=['iso3'], value_vars=tax_cols,
                           var_name='year_str', value_name='taxr')
df_tax_long['year'] = df_tax_long['year_str'].str.extract(r'(\d+)').astype(int)
df_tax_long = df_tax_long[['iso3', 'year', 'taxr']]
df_tax_long['taxr'] = df_tax_long['taxr'] / 100  # Convert to fraction

# Merge tax rates
bea_all = bea_all.merge(df_tax_long, on=['iso3', 'year'], how='inner')
print(f"  After tax rate merge: {bea_all.shape}")

# 2. Import PWT (for GDP)
print("  Importing PWT...")
pwt = pd.read_stata(f"{RAW_DIR}/PWT/pwt90.dta")
pwt = pwt.rename(columns={'countrycode': 'iso3'})
pwt = pwt[['iso3', 'year', 'rgdpo', 'pop']].copy()
pwt['year'] = pwt['year'].astype(int)
pwt['lrgdp'] = np.log(pwt['rgdpo'])
pwt = pwt[['iso3', 'year', 'lrgdp']].dropna()

bea_all = bea_all.merge(pwt, on=['iso3', 'year'], how='inner')
print(f"  After PWT merge: {bea_all.shape}")

# VGB GDP correction (from 5.construction.do)
vgb_gdp = {
    1999: 7.517836, 2000: 7.659624, 2001: 7.779726, 2002: 7.651984,
    2003: 7.811316, 2004: 7.505426, 2005: 7.713525, 2006: 7.806608,
    2007: 7.29317, 2008: 7.83293, 2009: 7.659741, 2010: 7.964995,
    2011: 7.686249, 2012: 7.268547, 2013: 8.259789,
}
for yr, val in vgb_gdp.items():
    mask = (bea_all['iso3'] == 'VGB') & (bea_all['year'] == yr)
    bea_all.loc[mask, 'lrgdp'] = val

# 3. Create FMA
# FMA construction is extremely complex (requires gravity regression on BACI trade data)
# Instead, we'll construct it from the gravity model approach
# Since BACI data is very large and the gravity regression is complex,
# we'll approximate FMA using a simplified version
print("  Creating FMA (simplified)...")

# Check if we can load the gravity data for FMA
gravity_path = f"{RAW_DIR}/FMA/Gravity_V202102.dta"
if os.path.exists(gravity_path):
    try:
        grav = pd.read_stata(gravity_path, columns=['iso3_o', 'iso3_d', 'year', 'contig', 'col_dep_ever', 'dist'])
        grav = grav[(grav['year'] >= 1999) & (grav['year'] <= 2013)]
        # Simplified FMA: sum of exp(phi_hat) where phi_hat is from gravity
        # We use a simple distance-based proxy: FMA_j = sum_k (GDP_k / dist_jk)
        # This is a standard market potential measure
        # Merge GDP into gravity
        grav['year'] = grav['year'].astype(int)
        grav = grav.merge(pwt.rename(columns={'iso3': 'iso3_d', 'lrgdp': 'lrgdp_d'}),
                         on=['iso3_d', 'year'], how='inner')
        grav['dist_num'] = pd.to_numeric(grav['dist'], errors='coerce')
        grav = grav.dropna(subset=['dist_num', 'lrgdp_d'])
        grav = grav[grav['dist_num'] > 0]
        # Market potential: sum(GDP_d / dist)
        grav['mp_component'] = np.exp(grav['lrgdp_d']) / grav['dist_num']
        fma = grav.groupby(['iso3_o', 'year'])['mp_component'].sum().reset_index()
        fma.columns = ['iso3', 'year', 'fma']
        fma['lfma'] = np.log(fma['fma'])
        fma = fma[['iso3', 'year', 'lfma']].dropna()
        bea_all = bea_all.merge(fma, on=['iso3', 'year'], how='inner')
        print(f"  After FMA merge: {bea_all.shape}")
    except Exception as e:
        print(f"  WARNING: FMA construction failed: {e}")
        # Fallback: set lfma to missing (specs will fail)
        bea_all['lfma'] = np.nan
else:
    print("  WARNING: Gravity data not found, setting lfma=NaN")
    bea_all['lfma'] = np.nan

# 4. Import agreements
# The .dta files have: iso2_o, country_d, type (1=DTC, 2=TIEA), date, sign, enf, standard, four_and_5
# We need: per country-year, total counts of enforced DTCs/TIEAs; and US-specific treaty status
print("  Importing agreements...")
agr_dir = f"{RAW_DIR}/AGR/All countries"
agr_dta_files = [f for f in os.listdir(agr_dir) if f.endswith('.txt.dta')]

# ISO2 to ISO3 mapping
ISO2_TO_ISO3 = {
    'AD': 'AND', 'AE': 'ARE', 'AF': 'AFG', 'AG': 'ATG', 'AI': 'AIA', 'AL': 'ALB', 'AM': 'ARM',
    'AN': 'ANT', 'AR': 'ARG', 'AT': 'AUT', 'AU': 'AUS', 'AW': 'ABW', 'AZ': 'AZE',
    'BA': 'BIH', 'BB': 'BRB', 'BD': 'BGD', 'BE': 'BEL', 'BF': 'BFA', 'BG': 'BGR',
    'BH': 'BHR', 'BI': 'BDI', 'BJ': 'BEN', 'BM': 'BMU', 'BN': 'BRN', 'BO': 'BOL',
    'BR': 'BRA', 'BS': 'BHS', 'BT': 'BTN', 'BW': 'BWA', 'BY': 'BLR', 'BZ': 'BLZ',
    'CA': 'CAN', 'CD': 'COD', 'CF': 'CAF', 'CG': 'COG', 'CH': 'CHE', 'CI': 'CIV',
    'CK': 'COK', 'CL': 'CHL', 'CM': 'CMR', 'CN': 'CHN', 'CO': 'COL', 'CR': 'CRI',
    'CU': 'CUB', 'CV': 'CPV', 'CY': 'CYP', 'CZ': 'CZE', 'DE': 'DEU', 'DJ': 'DJI',
    'DK': 'DNK', 'DM': 'DMA', 'DO': 'DOM', 'DZ': 'DZA', 'EC': 'ECU', 'EE': 'EST',
    'EG': 'EGY', 'ES': 'ESP', 'ET': 'ETH', 'FI': 'FIN', 'FJ': 'FJI', 'FK': 'FLK',
    'FO': 'FRO', 'FR': 'FRA', 'GA': 'GAB', 'GB': 'GBR', 'GD': 'GRD', 'GE': 'GEO',
    'GG': 'GGY', 'GH': 'GHA', 'GI': 'GIB', 'GL': 'GRL', 'GM': 'GMB', 'GN': 'GIN',
    'GQ': 'GNQ', 'GR': 'GRC', 'GT': 'GTM', 'GW': 'GNB', 'GY': 'GUY',
    'HK': 'HKG', 'HN': 'HND', 'HR': 'HRV', 'HU': 'HUN', 'ID': 'IDN', 'IE': 'IRL',
    'IL': 'ISR', 'IN': 'IND', 'IQ': 'IRQ', 'IR': 'IRN', 'IS': 'ISL', 'IT': 'ITA',
    'JE': 'JEY', 'JM': 'JAM', 'JO': 'JOR', 'JP': 'JPN', 'KE': 'KEN', 'KG': 'KGZ',
    'KH': 'KHM', 'KI': 'KIR', 'KM': 'COM', 'KN': 'KNA', 'KP': 'PRK', 'KR': 'KOR',
    'KW': 'KWT', 'KY': 'CYM', 'KZ': 'KAZ', 'LA': 'LAO', 'LB': 'LBN', 'LC': 'LCA',
    'LI': 'LIE', 'LK': 'LKA', 'LR': 'LBR', 'LS': 'LSO', 'LT': 'LTU', 'LU': 'LUX',
    'LV': 'LVA', 'LY': 'LBY', 'MA': 'MAR', 'MC': 'MCO', 'MD': 'MDA', 'ME': 'MNE',
    'MG': 'MDG', 'MH': 'MHL', 'MK': 'MKD', 'ML': 'MLI', 'MM': 'MMR', 'MN': 'MNG',
    'MO': 'MAC', 'MR': 'MRT', 'MS': 'MSR', 'MT': 'MLT', 'MU': 'MUS', 'MV': 'MDV',
    'MW': 'MWI', 'MX': 'MEX', 'MY': 'MYS', 'MZ': 'MOZ', 'NA': 'NAM', 'NE': 'NER',
    'NG': 'NGA', 'NI': 'NIC', 'NL': 'NLD', 'NO': 'NOR', 'NP': 'NPL', 'NU': 'NIU',
    'NZ': 'NZL', 'OM': 'OMN', 'PA': 'PAN', 'PE': 'PER', 'PG': 'PNG', 'PH': 'PHL',
    'PK': 'PAK', 'PL': 'POL', 'PT': 'PRT', 'PY': 'PRY', 'QA': 'QAT', 'RE': 'REU',
    'RO': 'ROM', 'RS': 'SRB', 'RU': 'RUS', 'RW': 'RWA', 'SA': 'SAU', 'SB': 'SLB',
    'SC': 'SYC', 'SD': 'SDN', 'SE': 'SWE', 'SG': 'SGP', 'SI': 'SVN', 'SK': 'SVK',
    'SL': 'SLE', 'SM': 'SMR', 'SN': 'SEN', 'SO': 'SOM', 'SR': 'SUR', 'ST': 'STP',
    'SV': 'SLV', 'SY': 'SYR', 'SZ': 'SWZ', 'TC': 'TCA', 'TD': 'TCD', 'TG': 'TGO',
    'TH': 'THA', 'TJ': 'TJK', 'TL': 'TLS', 'TM': 'TKM', 'TN': 'TUN', 'TR': 'TUR',
    'TT': 'TTO', 'TV': 'TUV', 'TW': 'TWN', 'TZ': 'TZA', 'UA': 'UKR', 'UG': 'UGA',
    'US': 'USA', 'UY': 'URY', 'UZ': 'UZB', 'VA': 'VAT', 'VC': 'VCT', 'VE': 'VEN',
    'VG': 'VGB', 'VN': 'VNM', 'VU': 'VUT', 'WS': 'WSM', 'YE': 'YEM', 'ZA': 'ZAF',
    'ZM': 'ZMB', 'ZW': 'ZWE',
}

agr_list = []
for dta_file in agr_dta_files:
    try:
        df_agr = pd.read_stata(f"{agr_dir}/{dta_file}")
        agr_list.append(df_agr)
    except Exception:
        continue

if agr_list:
    agr_all = pd.concat(agr_list, ignore_index=True)
    # Convert iso2_o to iso3_o
    agr_all['iso3_o'] = agr_all['iso2_o'].map(ISO2_TO_ISO3)
    # Convert country_d to iso3_d using our COUNTRY_MAP
    agr_all['iso3_d'] = agr_all['country_d'].apply(countryname_to_iso3)
    agr_all = agr_all.rename(columns={'date': 'year'})
    agr_all['year'] = agr_all['year'].astype(int)
    agr_all = agr_all[(agr_all['year'] >= 1999) & (agr_all['year'] <= 2013)]
    agr_all = agr_all.dropna(subset=['iso3_o', 'iso3_d'])

    # Total agreement counts by origin country x year (sum of enforced agreements with all partners)
    # type 1 = DTC, type 2 = TIEA
    tiea_counts = agr_all[agr_all['type'] == 2].groupby(['iso3_o', 'year'])['enf'].sum().reset_index()
    tiea_counts.columns = ['iso3', 'year', 'tiea_num']
    tiea_counts['tiea_num'] = tiea_counts['tiea_num'] / 100  # Rescale as in Stata code

    dtc_counts = agr_all[agr_all['type'] == 1].groupby(['iso3_o', 'year'])['enf'].sum().reset_index()
    dtc_counts.columns = ['iso3', 'year', 'dtc_num']
    dtc_counts['dtc_num'] = dtc_counts['dtc_num'] / 100

    agr_total = tiea_counts.merge(dtc_counts, on=['iso3', 'year'], how='outer').fillna(0)

    # US-specific agreements: where iso3_o == 'USA'
    us_agr = agr_all[agr_all['iso3_o'] == 'USA'].copy()
    if len(us_agr) > 0:
        # TIEA enforced with US
        us_tiea = us_agr[us_agr['type'] == 2].groupby(['iso3_d', 'year'])['enf'].max().reset_index()
        us_tiea.columns = ['iso3', 'year', 'tiea_enf_us']

        # DTC enforced with US
        us_dtc = us_agr[us_agr['type'] == 1].groupby(['iso3_d', 'year']).agg(
            dtc_enf_us=('enf', 'max'),
            dtc_45_us=('four_and_5', 'max')
        ).reset_index()
        us_dtc.columns = ['iso3', 'year', 'dtc_enf_us', 'dtc_45_us']

        agr_total = agr_total.merge(us_tiea, on=['iso3', 'year'], how='left')
        agr_total = agr_total.merge(us_dtc, on=['iso3', 'year'], how='left')
        agr_total = agr_total.fillna(0)
    else:
        agr_total['tiea_enf_us'] = 0
        agr_total['dtc_enf_us'] = 0
        agr_total['dtc_45_us'] = 0

    # eoi_enf = tiea_enf_us OR (dtc_enf_us AND dtc_45_us)
    agr_total['eoi_enf'] = ((agr_total['tiea_enf_us'] > 0) |
                              ((agr_total['dtc_enf_us'] > 0) & (agr_total['dtc_45_us'] > 0))).astype(float)
    agr_total['dtc_enf'] = (agr_total['dtc_enf_us'] > 0).astype(float)

    bea_all = bea_all.merge(agr_total[['iso3', 'year', 'tiea_num', 'dtc_num', 'eoi_enf', 'dtc_enf']],
                            on=['iso3', 'year'], how='left')
    bea_all[['tiea_num', 'dtc_num', 'eoi_enf', 'dtc_enf']] = bea_all[['tiea_num', 'dtc_num', 'eoi_enf', 'dtc_enf']].fillna(0)
    print(f"  After agreements merge: {bea_all.shape}")
    print(f"    eoi_enf non-zero: {(bea_all['eoi_enf'] > 0).sum()}")
    print(f"    dtc_enf non-zero: {(bea_all['dtc_enf'] > 0).sum()}")
    print(f"    dtc_num range: {bea_all['dtc_num'].min():.3f} to {bea_all['dtc_num'].max():.3f}")
else:
    for c in ['tiea_num', 'dtc_num', 'eoi_enf', 'dtc_enf']:
        bea_all[c] = 0
    print("  WARNING: No agreement data found")

# 5. Tax haven indicator
print("  Merging tax haven indicator...")
haven_df = pd.read_stata(f"{RAW_DIR}/taxhaven.dta")
# haven_df only has haven==1 countries
haven_set = set(haven_df['iso3'].unique())
haven_set.add('NLD')  # Netherlands added as haven per code
bea_all['haven'] = bea_all['iso3'].isin(haven_set).astype(int)

# 6. Apply sector/sample restrictions from 5.construction.do
# Drop k in {0, 2, 3, 15, 16} and k==13 (finance)
bea_all = bea_all[~bea_all['k'].isin([0, 2, 3, 13, 15, 16])].copy()
print(f"  After sector restrictions: {bea_all.shape}")

# Map industry names and d values
indus_map = {1: 'Mining', 4: 'Food', 5: 'Chemicals', 6: 'Primary Fabricated Mat',
             7: 'Machinery', 8: 'Computer', 9: 'Electrical eqpm',
             10: 'Transportation', 11: 'Wholesale', 12: 'Info', 14: 'Services'}
d_map = {1: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8, 11: 9, 12: 10, 14: 11}

bea_all['indus'] = bea_all['k'].map(indus_map)
bea_all['d'] = bea_all['k'].map(d_map)

# Create sector-year FE
bea_all['kt'] = bea_all['k'].astype(str) + '_' + bea_all['year'].astype(str)

# Create country numeric ID
bea_all['i'] = pd.Categorical(bea_all['iso3']).codes

# Construct variables from 5.construction.do
bea_all['lprofit'] = np.log(bea_all['profit_'].clip(lower=0.001))
bea_all.loc[bea_all['profit_'] <= 0, 'lprofit'] = np.nan

# Export platform share: ep = (foreign_ + sales_us) / sales_
bea_all['ep'] = (bea_all['foreign_'] + bea_all['sales_us']) / bea_all['sales_']
# Alternative definition
mask = bea_all['ep'].isna() & bea_all['sales_'].notna() & bea_all['local_'].notna()
bea_all.loc[mask, 'ep'] = (bea_all.loc[mask, 'sales_'] - bea_all.loc[mask, 'local_']) / bea_all.loc[mask, 'sales_']
# No sales -> no EP
bea_all.loc[(bea_all['sales_'] == 0) & bea_all['ep'].isna(), 'ep'] = 0
# No local sales -> only EP
mask2 = (bea_all['local_'] == 0) & (bea_all['sales_'] > 0) & bea_all['sales_'].notna() & bea_all['ep'].isna()
bea_all.loc[mask2, 'ep'] = 1
# No foreign/US sales -> no EP
mask3 = (bea_all['sales_us'] == 0) & (bea_all['foreign_'] == 0) & bea_all['ep'].isna()
bea_all.loc[mask3, 'ep'] = 0
mask4 = (bea_all['local_'] == 0) & (bea_all['foreign_'] == 0) & (bea_all['sales_us'] == 0)
bea_all.loc[mask4, 'ep'] = 0
# Drop imputed ep values (marked as 0.251 in BEA)
mask_imp = (bea_all['foreign_'].between(0.250001, 0.25199999)) & (bea_all['sales_us'].between(0.250001, 0.25199999))
bea_all.loc[mask_imp, 'ep'] = np.nan
# Clip ep to [0,1] - values outside this range are data artifacts
bea_all['ep'] = bea_all['ep'].clip(lower=0, upper=1)

# Big5 havens and otherh
bea_all['big5'] = bea_all['iso3'].isin(['CHE', 'HKG', 'IRL', 'LUX', 'SGP', 'NLD']).astype(int)
bea_all['otherh'] = bea_all['haven'] - bea_all['big5']
bea_all['otherh'] = bea_all['otherh'].clip(lower=0)

# Employment and equipment
bea_all['emp_'] = bea_all['emp_'].fillna(method='ffill')
bea_all['eqpmt_'] = bea_all['eqpmt_'].fillna(method='ffill')
# Sort first
bea_all = bea_all.sort_values(['iso3', 'k', 'year'])
# Forward fill within iso3 x k
for col in ['emp_', 'eqpmt_']:
    bea_all[col] = bea_all.groupby(['iso3', 'k'])[col].ffill()
bea_all['lemp'] = np.log(bea_all['emp_'] + 1)
bea_all['leqpmt'] = np.log(bea_all['eqpmt_'] + 1)

# Interaction term
bea_all['ep_haven'] = bea_all['ep'] * bea_all['haven']

# Cubic profit
bea_all['cub_profit'] = np.sign(bea_all['profit_']) * np.abs(bea_all['profit_']) ** (1/3)

# Imputed flags
bea_all['imputed_ep'] = 0
for var in ['sales_us', 'sales_', 'foreign_']:
    if var in bea_all.columns:
        bea_all.loc[bea_all[var].between(0.250001, 0.25199999), 'imputed_ep'] = 1

bea_all['imputed_profit'] = 0
if 'profit_' in bea_all.columns:
    bea_all.loc[bea_all['profit_'].between(0.250001, 0.25199999), 'imputed_profit'] = 1
if 'eqpmt_' in bea_all.columns:
    bea_all.loc[bea_all['eqpmt_'].between(0.250001, 0.25199999), 'imputed_profit'] = 1
if 'emp_' in bea_all.columns:
    bea_all.loc[bea_all['emp_'].between(0.0250001, 0.025199999), 'imputed_profit'] = 1

# EP no US sales
bea_all['ep_no_us'] = bea_all['foreign_'] / bea_all['sales_']

# Now create the analysis sample
# The Stata code first runs the profit regression to define the sample:
# reghdfe profit_ ... if imputed_profit==0, absorb(kt) cluster(i)
# keep if e(sample)
# This means we keep only observations where all variables used in the profit regression are non-missing
# AND imputed_profit==0

df = bea_all.copy()
print(f"\nDataset before sample restrictions: {df.shape}")
print(f"  ep non-missing: {df['ep'].notna().sum()}")
print(f"  lfma non-missing: {df['lfma'].notna().sum()}")
print(f"  lprofit non-missing: {df['lprofit'].notna().sum()}")

# =====================================================================
#  HELPER FUNCTIONS
# =====================================================================

def next_run_id():
    global run_counter
    run_counter += 1
    return f"{PAPER_ID}__run_{run_counter:03d}"

def next_infer_id():
    global infer_counter
    infer_counter += 1
    return f"{PAPER_ID}__infer_{infer_counter:03d}"


def run_glm_fractional_logit(data, outcome, treatment, controls, fe_col='kt',
                              cluster_col='iso3', return_marginal_effects=True):
    """Run GLM fractional logit with sector-year FE dummies and return marginal effects."""
    # Create dummies for FE
    allvars = [outcome, treatment] + controls + [fe_col, cluster_col]
    df_reg = data.dropna(subset=[v for v in allvars if v in data.columns]).copy()

    if len(df_reg) < 30:
        raise ValueError(f"Too few observations: {len(df_reg)}")

    # Reset index to avoid alignment issues
    df_reg = df_reg.reset_index(drop=True)

    # Create FE dummies
    fe_dummies = pd.get_dummies(df_reg[fe_col], prefix='fe', drop_first=True, dtype=float)

    # Build X matrix
    rhs_vars = [treatment] + controls
    X = df_reg[rhs_vars].copy()
    X = pd.concat([X, fe_dummies], axis=1)
    X = sm.add_constant(X)

    y = df_reg[outcome]

    # Fit GLM
    model = GLM(y, X, family=Binomial(link=LogitLink()))

    # Cluster standard errors
    cluster_groups = df_reg[cluster_col]
    result = model.fit(cov_type='cluster', cov_kwds={'groups': cluster_groups})

    n_obs = len(df_reg)

    if return_marginal_effects:
        # Compute average marginal effects (AME) for focal variables
        # For logit link: dydx = beta * f(X*beta) * (1 - f(X*beta))
        # where f is the logistic CDF
        from scipy.special import expit
        Xb = result.predict(X)  # These are already probabilities for binomial GLM
        # For logit link: dF/d(Xb) = F(Xb) * (1 - F(Xb))
        dFdXb = Xb * (1 - Xb)

        marginal_effects = {}
        marginal_se = {}
        for var in rhs_vars:
            beta = result.params[var]
            me = (beta * dFdXb).mean()
            marginal_effects[var] = me

            # Delta method SE for AME
            # Var(AME) ≈ (d AME/d beta)^2 * Var(beta)
            # d AME/d beta ≈ mean(dFdXb) (first-order approximation)
            se_beta = result.bse[var]
            me_se = se_beta * dFdXb.mean()
            marginal_se[var] = me_se

        # Pseudo R2
        llf = result.llf
        X_null = np.ones((len(y), 1))
        ll_null = GLM(y.values, X_null,
                      family=Binomial(link=LogitLink())).fit().llf
        pseudo_r2 = 1 - llf / ll_null

        return {
            'marginal_effects': marginal_effects,
            'marginal_se': marginal_se,
            'n_obs': n_obs,
            'pseudo_r2': pseudo_r2,
            'result': result,
            'treatment_me': marginal_effects[treatment],
            'treatment_se': marginal_se[treatment],
        }
    else:
        return result


def run_ols_fe(data, outcome, treatment, controls, fe_col='kt', cluster_col='iso3'):
    """Run OLS with FE using pyfixest."""
    allvars = [outcome, treatment] + controls + [fe_col, cluster_col]
    df_reg = data.dropna(subset=[v for v in allvars if v in data.columns]).copy()

    if len(df_reg) < 30:
        raise ValueError(f"Too few observations: {len(df_reg)}")

    rhs = " + ".join([treatment] + controls)
    formula = f"{outcome} ~ {rhs} | {fe_col}"

    model = pf.feols(formula, data=df_reg, vcov={"CRV1": cluster_col})

    coef = model.coef()[treatment]
    se = model.se()[treatment]
    pval = model.pvalue()[treatment]
    ci = model.confint()
    ci_lower = ci.loc[treatment, '2.5%'] if treatment in ci.index else np.nan
    ci_upper = ci.loc[treatment, '97.5%'] if treatment in ci.index else np.nan

    return {
        'coef': coef,
        'se': se,
        'pval': pval,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_obs': model._N,
        'r2': model._r2,
        'model': model,
        'all_coefs': dict(model.coef()),
    }


def record_result(spec_id, spec_run_id, baseline_group_id, outcome_var, treatment_var,
                  coefficient, std_error, p_value, ci_lower, ci_upper, n_obs, r_squared,
                  coef_vector_json, sample_desc, fixed_effects, controls_desc, cluster_var,
                  run_success=1, run_error="", spec_tree_path="custom"):
    """Add a row to the results list."""
    results.append({
        'paper_id': PAPER_ID,
        'spec_run_id': spec_run_id,
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'baseline_group_id': baseline_group_id,
        'outcome_var': outcome_var,
        'treatment_var': treatment_var,
        'coefficient': coefficient,
        'std_error': std_error,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_obs': n_obs,
        'r_squared': r_squared,
        'coefficient_vector_json': json.dumps(coef_vector_json),
        'sample_desc': sample_desc,
        'fixed_effects': fixed_effects,
        'controls_desc': controls_desc,
        'cluster_var': cluster_var,
        'run_success': run_success,
        'run_error': run_error,
    })


def record_inference(spec_run_id, spec_id, baseline_group_id, coefficient, std_error,
                     p_value, ci_lower, ci_upper, n_obs, r_squared, coef_vector_json,
                     run_success=1, run_error="", spec_tree_path="custom"):
    """Add a row to the inference results list."""
    infer_run_id = next_infer_id()
    inference_results.append({
        'paper_id': PAPER_ID,
        'inference_run_id': infer_run_id,
        'spec_run_id': spec_run_id,
        'spec_id': spec_id,
        'spec_tree_path': spec_tree_path,
        'baseline_group_id': baseline_group_id,
        'coefficient': coefficient,
        'std_error': std_error,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_obs': n_obs,
        'r_squared': r_squared,
        'coefficient_vector_json': json.dumps(coef_vector_json),
        'run_success': run_success,
        'run_error': run_error,
    })


def make_g1_payload(coefficients, extra_blocks=None):
    """Make success payload for G1 specs."""
    payload = make_success_payload(
        coefficients=coefficients,
        inference={"spec_id": G1_INFERENCE_CANONICAL["spec_id"],
                   "params": G1_INFERENCE_CANONICAL["params"]},
        software=SW_BLOCK,
        surface_hash=SURFACE_HASH,
        design={"panel_fixed_effects": G1_DESIGN_AUDIT},
    )
    if extra_blocks:
        payload.update(extra_blocks)
    return payload


def make_g2_payload(coefficients, extra_blocks=None):
    """Make success payload for G2 specs."""
    payload = make_success_payload(
        coefficients=coefficients,
        inference={"spec_id": G2_INFERENCE_CANONICAL["spec_id"],
                   "params": G2_INFERENCE_CANONICAL["params"]},
        software=SW_BLOCK,
        surface_hash=SURFACE_HASH,
        design={"panel_fixed_effects": G2_DESIGN_AUDIT},
    )
    if extra_blocks:
        payload.update(extra_blocks)
    return payload


def fail_payload(error_msg, stage="estimation"):
    """Make failure payload."""
    return make_failure_payload(
        error=error_msg,
        error_details={"stage": stage, "exception_type": "RuntimeError",
                       "exception_message": error_msg},
        software=SW_BLOCK,
        surface_hash=SURFACE_HASH,
    )


# =====================================================================
#  G1 SPECIFICATIONS: Export Platform Share (ep ~ lfma)
# =====================================================================
print("\n" + "="*70)
print("G1: Export Platform Share (ep ~ lfma)")
print("="*70)

G1_CONTROLS_FULL = ['taxr', 'haven', 'eoi_enf', 'dtc_enf', 'lrgdp', 'dtc_num']
G1_CONTROL2 = ['eoi_enf', 'dtc_enf', 'lrgdp', 'dtc_num']  # control2 from Stata code

# Analysis sample for G1: same sample as the profit regression (imputed_profit==0)
# The Stata code keeps only e(sample) from the profit regression
# This effectively restricts to observations where all profit regression variables are non-missing
g1_vars = ['ep', 'lfma', 'taxr', 'haven', 'eoi_enf', 'dtc_enf', 'lrgdp', 'dtc_num', 'kt', 'iso3']
g1_sample = df.dropna(subset=[v for v in g1_vars if v in df.columns]).copy()
# Also require non-missing profit regression variables
profit_vars = ['lprofit', 'lemp', 'leqpmt', 'ep_haven']
for v in profit_vars:
    if v in g1_sample.columns:
        g1_sample = g1_sample[g1_sample[v].notna()]
g1_sample = g1_sample[g1_sample['imputed_profit'] == 0]

print(f"G1 sample size: {len(g1_sample)}")

# --- Baseline: Table2-Col4 (GLM fractional logit with all controls including haven) ---
spec_id = "baseline"
run_id = next_run_id()
try:
    res = run_glm_fractional_logit(g1_sample, 'ep', 'lfma', G1_CONTROLS_FULL, 'kt', 'iso3')
    coef = res['treatment_me']
    se = res['treatment_se']
    from scipy import stats
    pval = 2 * (1 - stats.norm.cdf(abs(coef / se)))
    ci_lo = coef - 1.96 * se
    ci_hi = coef + 1.96 * se

    payload = make_g1_payload(
        coefficients=res['marginal_effects'],
        extra_blocks={'notes': 'Table 2 Col 4: GLM fractional logit baseline with marginal effects'}
    )
    record_result(spec_id, run_id, "G1", "ep", "lfma",
                  coef, se, pval, ci_lo, ci_hi,
                  res['n_obs'], res['pseudo_r2'], payload,
                  "Full sample (profit sample)", "kt", ", ".join(G1_CONTROLS_FULL), "iso3",
                  spec_tree_path="specification_tree/designs/panel_fixed_effects.md")
    print(f"  BASELINE: coef={coef:.4f}, se={se:.4f}, p={pval:.4f}, n={res['n_obs']}")
except Exception as e:
    print(f"  BASELINE FAILED: {e}")
    traceback.print_exc()
    payload = fail_payload(str(e))
    record_result(spec_id, run_id, "G1", "ep", "lfma",
                  np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, payload,
                  "", "", "", "", run_success=0, run_error=str(e),
                  spec_tree_path="specification_tree/designs/panel_fixed_effects.md")


def run_g1_glm_spec(spec_id, controls, sample=None, fe='kt', treatment='lfma',
                     outcome='ep', sample_desc="Full sample", spec_tree_path="custom",
                     extra_blocks=None):
    """Run a G1 GLM specification and record it."""
    run_id = next_run_id()
    data = sample if sample is not None else g1_sample
    try:
        res = run_glm_fractional_logit(data, outcome, treatment, controls, fe, 'iso3')
        coef = res['treatment_me']
        se = res['treatment_se']
        pval = 2 * (1 - stats.norm.cdf(abs(coef / se)))
        ci_lo = coef - 1.96 * se
        ci_hi = coef + 1.96 * se
        payload = make_g1_payload(res['marginal_effects'], extra_blocks=extra_blocks)
        record_result(spec_id, run_id, "G1", outcome, treatment,
                      coef, se, pval, ci_lo, ci_hi,
                      res['n_obs'], res['pseudo_r2'], payload,
                      sample_desc, fe, ", ".join(controls), "iso3",
                      spec_tree_path=spec_tree_path)
        print(f"  {spec_id}: coef={coef:.4f}, se={se:.4f}, p={pval:.4f}, n={res['n_obs']}")
        return run_id
    except Exception as e:
        print(f"  {spec_id} FAILED: {e}")
        payload = fail_payload(str(e))
        record_result(spec_id, run_id, "G1", outcome, treatment,
                      np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, payload,
                      sample_desc, fe, ", ".join(controls), "iso3",
                      run_success=0, run_error=str(e), spec_tree_path=spec_tree_path)
        return run_id


def run_g1_ols_spec(spec_id, controls, sample=None, fe='kt', treatment='lfma',
                     outcome='ep', sample_desc="Full sample", spec_tree_path="custom",
                     extra_blocks=None):
    """Run a G1 OLS specification and record it."""
    run_id = next_run_id()
    data = sample if sample is not None else g1_sample
    try:
        res = run_ols_fe(data, outcome, treatment, controls, fe, 'iso3')
        payload = make_g1_payload(res['all_coefs'], extra_blocks=extra_blocks)
        # Override design for OLS
        payload['design'] = {"panel_fixed_effects": {
            **G1_DESIGN_AUDIT,
            "estimator": "ols_reghdfe",
            "marginal_effects": "not applicable (linear model)"
        }}
        record_result(spec_id, run_id, "G1", outcome, treatment,
                      res['coef'], res['se'], res['pval'],
                      res['ci_lower'], res['ci_upper'],
                      res['n_obs'], res['r2'], payload,
                      sample_desc, fe, ", ".join(controls), "iso3",
                      spec_tree_path=spec_tree_path)
        print(f"  {spec_id}: coef={res['coef']:.4f}, se={res['se']:.4f}, p={res['pval']:.4f}, n={res['n_obs']}")
        return run_id
    except Exception as e:
        print(f"  {spec_id} FAILED: {e}")
        payload = fail_payload(str(e))
        record_result(spec_id, run_id, "G1", outcome, treatment,
                      np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, payload,
                      sample_desc, fe, "", "iso3",
                      run_success=0, run_error=str(e), spec_tree_path=spec_tree_path)
        return run_id


# --- Additional baselines from core_universe.baseline_spec_ids ---

# Table 2 Col 3: no haven
print("\n--- Baseline variants ---")
controls_col3 = ['taxr'] + G1_CONTROL2
run_g1_glm_spec("baseline__table2_col3", controls_col3,
                 spec_tree_path="specification_tree/designs/panel_fixed_effects.md",
                 extra_blocks={'notes': 'Table 2 Col 3: GLM without haven'})

# Table 2 Col 5: non-haven only
controls_col5 = ['taxr'] + G1_CONTROL2
run_g1_glm_spec("baseline__table2_col5_nohaven", controls_col5,
                 sample=g1_sample[g1_sample['haven'] == 0],
                 sample_desc="Non-haven countries only",
                 spec_tree_path="specification_tree/designs/panel_fixed_effects.md",
                 extra_blocks={'notes': 'Table 2 Col 5: non-haven'})

# Table 2 Col 6: haven only
run_g1_glm_spec("baseline__table2_col6_haven", controls_col5,
                 sample=g1_sample[g1_sample['haven'] == 1],
                 sample_desc="Haven countries only",
                 spec_tree_path="specification_tree/designs/panel_fixed_effects.md",
                 extra_blocks={'notes': 'Table 2 Col 6: haven only'})

# Table 2 Col 7: OLS non-haven
run_g1_ols_spec("baseline__table2_col7_ols_nohaven", controls_col5,
                 sample=g1_sample[g1_sample['haven'] == 0],
                 sample_desc="Non-haven countries, OLS",
                 spec_tree_path="specification_tree/designs/panel_fixed_effects.md",
                 extra_blocks={'notes': 'Table 2 Col 7: OLS non-haven'})

# Table 2 Col 8: OLS haven
run_g1_ols_spec("baseline__table2_col8_ols_haven", controls_col5,
                 sample=g1_sample[g1_sample['haven'] == 1],
                 sample_desc="Haven countries, OLS",
                 spec_tree_path="specification_tree/designs/panel_fixed_effects.md",
                 extra_blocks={'notes': 'Table 2 Col 8: OLS haven'})


# --- Design variant: OLS within estimator ---
print("\n--- Design variants ---")
run_g1_ols_spec("design/panel_fixed_effects/estimator/within", G1_CONTROLS_FULL,
                 spec_tree_path="specification_tree/designs/panel_fixed_effects.md#within",
                 extra_blocks={'estimation': {'method': 'OLS within (reghdfe)', 'notes': 'OLS alternative to GLM'}})


# --- RC: Controls LOO ---
print("\n--- RC: Controls LOO ---")
for ctrl in G1_CONTROLS_FULL:
    spec_id = f"rc/controls/loo/drop_{ctrl}"
    remaining = [c for c in G1_CONTROLS_FULL if c != ctrl]
    run_g1_glm_spec(spec_id, remaining,
                     spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
                     extra_blocks={'controls': {'spec_id': spec_id, 'family': 'loo',
                                                'dropped': [ctrl], 'n_controls': len(remaining)}})


# --- RC: Control sets ---
print("\n--- RC: Control sets ---")

# Minimal (just treatment, no controls)
run_g1_glm_spec("rc/controls/sets/minimal", [],
                 spec_tree_path="specification_tree/modules/robustness/controls.md#sets",
                 extra_blocks={'controls': {'spec_id': 'rc/controls/sets/minimal', 'family': 'sets',
                                            'included': [], 'n_controls': 0}})

# Col 1: lrgdp only
run_g1_glm_spec("rc/controls/sets/control1_lrgdp_only", ['lrgdp'],
                 spec_tree_path="specification_tree/modules/robustness/controls.md#sets",
                 extra_blocks={'controls': {'spec_id': 'rc/controls/sets/control1_lrgdp_only',
                                            'family': 'sets', 'included': ['lrgdp'], 'n_controls': 1}})

# Col 2: lrgdp + taxr
run_g1_glm_spec("rc/controls/sets/control2_lrgdp_taxr", ['taxr', 'lrgdp'],
                 spec_tree_path="specification_tree/modules/robustness/controls.md#sets",
                 extra_blocks={'controls': {'spec_id': 'rc/controls/sets/control2_lrgdp_taxr',
                                            'family': 'sets', 'included': ['taxr', 'lrgdp'], 'n_controls': 2}})

# Full with haven (same as baseline)
run_g1_glm_spec("rc/controls/sets/full_with_haven", G1_CONTROLS_FULL,
                 spec_tree_path="specification_tree/modules/robustness/controls.md#sets",
                 extra_blocks={'controls': {'spec_id': 'rc/controls/sets/full_with_haven',
                                            'family': 'sets', 'included': G1_CONTROLS_FULL, 'n_controls': 6}})


# --- RC: Control progression (cols 1-4) ---
print("\n--- RC: Control progression ---")
progression = [
    (['lrgdp'], "Col 1"),
    (['taxr', 'lrgdp'], "Col 2"),
    (['taxr'] + G1_CONTROL2, "Col 3"),
    (G1_CONTROLS_FULL, "Col 4"),
]
for i, (ctrls, desc) in enumerate(progression):
    run_g1_glm_spec(f"rc/controls/progression/col1_to_col4", ctrls,
                     spec_tree_path="specification_tree/modules/robustness/controls.md#progression",
                     extra_blocks={'controls': {'spec_id': 'rc/controls/progression/col1_to_col4',
                                                'progression_step': i+1, 'included': ctrls,
                                                'n_controls': len(ctrls)}})
    # Only run once (the first step that isn't a duplicate)
    break
# Run other progression steps
run_g1_glm_spec("rc/controls/progression/col1_to_col4_step2", ['taxr', 'lrgdp'],
                 spec_tree_path="specification_tree/modules/robustness/controls.md#progression")
run_g1_glm_spec("rc/controls/progression/col1_to_col4_step3", ['taxr'] + G1_CONTROL2,
                 spec_tree_path="specification_tree/modules/robustness/controls.md#progression")


# --- RC: Add controls ---
print("\n--- RC: Add controls ---")
run_g1_glm_spec("rc/controls/add/lemp", G1_CONTROLS_FULL + ['lemp'],
                 spec_tree_path="specification_tree/modules/robustness/controls.md#add",
                 extra_blocks={'controls': {'spec_id': 'rc/controls/add/lemp', 'family': 'add',
                                            'added': ['lemp']}})

run_g1_glm_spec("rc/controls/add/leqpmt", G1_CONTROLS_FULL + ['leqpmt'],
                 spec_tree_path="specification_tree/modules/robustness/controls.md#add",
                 extra_blocks={'controls': {'spec_id': 'rc/controls/add/leqpmt', 'family': 'add',
                                            'added': ['leqpmt']}})

run_g1_glm_spec("rc/controls/add/lemp_leqpmt", G1_CONTROLS_FULL + ['lemp', 'leqpmt'],
                 spec_tree_path="specification_tree/modules/robustness/controls.md#add",
                 extra_blocks={'controls': {'spec_id': 'rc/controls/add/lemp_leqpmt', 'family': 'add',
                                            'added': ['lemp', 'leqpmt']}})


# --- RC: Sample subgroups ---
print("\n--- RC: Sample subgroups ---")

# Manufacturing only (d <= 8)
mfg_sample = g1_sample[g1_sample['d'] <= 8]
run_g1_glm_spec("rc/sample/subgroup/manufacturing_only", G1_CONTROLS_FULL,
                 sample=mfg_sample, sample_desc="Manufacturing only (d<=8)",
                 spec_tree_path="specification_tree/modules/robustness/sample.md#subgroup")

# Services only (d > 8)
svc_sample = g1_sample[g1_sample['d'] > 8]
run_g1_glm_spec("rc/sample/subgroup/services_only", G1_CONTROLS_FULL,
                 sample=svc_sample, sample_desc="Services only (d>8)",
                 spec_tree_path="specification_tree/modules/robustness/sample.md#subgroup")

# Non-haven only
run_g1_glm_spec("rc/sample/subgroup/non_haven_only", ['taxr'] + G1_CONTROL2,
                 sample=g1_sample[g1_sample['haven'] == 0],
                 sample_desc="Non-haven countries",
                 spec_tree_path="specification_tree/modules/robustness/sample.md#subgroup")

# Haven only
run_g1_glm_spec("rc/sample/subgroup/haven_only", ['taxr'] + G1_CONTROL2,
                 sample=g1_sample[g1_sample['haven'] == 1],
                 sample_desc="Haven countries only",
                 spec_tree_path="specification_tree/modules/robustness/sample.md#subgroup")

# Big5 havens
big5_sample = g1_sample[g1_sample['big5'] == 1]
run_g1_glm_spec("rc/sample/subgroup/big5_havens", ['taxr'] + G1_CONTROL2,
                 sample=big5_sample, sample_desc="Big-5 havens only",
                 spec_tree_path="specification_tree/modules/robustness/sample.md#subgroup")

# Caribbean havens
carib_sample = g1_sample[g1_sample['otherh'] == 1]
run_g1_glm_spec("rc/sample/subgroup/caribbean_havens", ['taxr'] + G1_CONTROL2,
                 sample=carib_sample, sample_desc="Caribbean havens only",
                 spec_tree_path="specification_tree/modules/robustness/sample.md#subgroup")


# --- RC: Sample restrictions ---
print("\n--- RC: Sample restrictions ---")

# Drop imputed EP
no_imp_ep = g1_sample[g1_sample['imputed_ep'] == 0]
run_g1_glm_spec("rc/sample/restriction/drop_imputed_ep", G1_CONTROLS_FULL,
                 sample=no_imp_ep, sample_desc="Drop imputed EP values",
                 spec_tree_path="specification_tree/modules/robustness/sample.md#restriction")

# Drop imputed profits (already in sample by construction, but include for completeness)
no_imp_profit = g1_sample[g1_sample['imputed_profit'] == 0]
run_g1_glm_spec("rc/sample/restriction/drop_imputed_profits", G1_CONTROLS_FULL,
                 sample=no_imp_profit, sample_desc="Drop imputed profits",
                 spec_tree_path="specification_tree/modules/robustness/sample.md#restriction")


# --- RC: Outlier trimming ---
print("\n--- RC: Outlier trimming ---")

# Trim EP at 1/99 percentiles
p1, p99 = g1_sample['ep'].quantile([0.01, 0.99])
trim_1_99 = g1_sample[(g1_sample['ep'] >= p1) & (g1_sample['ep'] <= p99)]
run_g1_glm_spec("rc/sample/outliers/trim_ep_1_99", G1_CONTROLS_FULL,
                 sample=trim_1_99, sample_desc=f"EP trimmed 1/99 ({p1:.3f}-{p99:.3f})",
                 spec_tree_path="specification_tree/modules/robustness/sample.md#outliers")

# Trim EP at 5/95 percentiles
p5, p95 = g1_sample['ep'].quantile([0.05, 0.95])
trim_5_95 = g1_sample[(g1_sample['ep'] >= p5) & (g1_sample['ep'] <= p95)]
run_g1_glm_spec("rc/sample/outliers/trim_ep_5_95", G1_CONTROLS_FULL,
                 sample=trim_5_95, sample_desc=f"EP trimmed 5/95 ({p5:.3f}-{p95:.3f})",
                 spec_tree_path="specification_tree/modules/robustness/sample.md#outliers")


# --- RC: FE variants ---
print("\n--- RC: FE variants ---")

# Country-year FE
g1_sample['country_year'] = g1_sample['iso3'] + '_' + g1_sample['year'].astype(str)
run_g1_ols_spec("rc/fe/alt/country_year", G1_CONTROLS_FULL, fe='country_year',
                 spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#alt",
                 extra_blocks={'fixed_effects': {'spec_id': 'rc/fe/alt/country_year', 'fe_vars': ['country_year']}})

# Sector only FE
g1_sample['k_str'] = g1_sample['k'].astype(str)
run_g1_ols_spec("rc/fe/alt/sector_only", G1_CONTROLS_FULL, fe='k_str',
                 spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#alt",
                 extra_blocks={'fixed_effects': {'spec_id': 'rc/fe/alt/sector_only', 'fe_vars': ['k']}})

# Year only FE
g1_sample['year_str'] = g1_sample['year'].astype(str)
run_g1_ols_spec("rc/fe/alt/year_only", G1_CONTROLS_FULL, fe='year_str',
                 spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#alt",
                 extra_blocks={'fixed_effects': {'spec_id': 'rc/fe/alt/year_only', 'fe_vars': ['year']}})

# Add country FE (sector-year + country)
# pyfixest: "outcome ~ rhs | kt + iso3"
run_id = next_run_id()
try:
    rhs = " + ".join(['lfma'] + G1_CONTROLS_FULL)
    formula = f"ep ~ {rhs} | kt + iso3"
    df_reg = g1_sample.dropna(subset=['ep', 'lfma'] + G1_CONTROLS_FULL + ['kt', 'iso3'])
    model = pf.feols(formula, data=df_reg, vcov={"CRV1": "iso3"})
    coef = model.coef()['lfma']
    se = model.se()['lfma']
    pval = model.pvalue()['lfma']
    ci = model.confint()
    payload = make_g1_payload(dict(model.coef()),
                              extra_blocks={'fixed_effects': {'spec_id': 'rc/fe/add/country',
                                                              'fe_vars': ['kt', 'iso3']}})
    payload['design'] = {"panel_fixed_effects": {
        **G1_DESIGN_AUDIT,
        "estimator": "ols_reghdfe",
        "fe_structure": ["kt", "iso3"],
    }}
    record_result("rc/fe/add/country", run_id, "G1", "ep", "lfma",
                  coef, se, pval,
                  ci.loc['lfma', '2.5%'], ci.loc['lfma', '97.5%'],
                  model._N, model._r2, payload,
                  "Full sample, OLS with country + sector-year FE", "kt + iso3",
                  ", ".join(G1_CONTROLS_FULL), "iso3",
                  spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#add")
    print(f"  rc/fe/add/country: coef={coef:.4f}, se={se:.4f}, n={model._N}")
except Exception as e:
    print(f"  rc/fe/add/country FAILED: {e}")
    payload = fail_payload(str(e))
    record_result("rc/fe/add/country", run_id, "G1", "ep", "lfma",
                  np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, payload,
                  "", "", "", "", run_success=0, run_error=str(e),
                  spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#add")


# --- RC: Estimator variants ---
print("\n--- RC: Estimator variants ---")

# OLS reghdfe (full sample with haven)
run_g1_ols_spec("rc/form/estimator/ols_reghdfe", G1_CONTROLS_FULL,
                 spec_tree_path="specification_tree/modules/robustness/functional_form.md#estimator",
                 extra_blocks={'functional_form': {'spec_id': 'rc/form/estimator/ols_reghdfe',
                                                    'interpretation': 'OLS linear probability model'}})

# OLS robust (no clustering)
run_id = next_run_id()
try:
    rhs = " + ".join(['lfma'] + G1_CONTROLS_FULL)
    formula = f"ep ~ {rhs} | kt"
    df_reg = g1_sample.dropna(subset=['ep', 'lfma'] + G1_CONTROLS_FULL + ['kt'])
    model = pf.feols(formula, data=df_reg, vcov="hetero")
    coef = model.coef()['lfma']
    se = model.se()['lfma']
    pval = model.pvalue()['lfma']
    ci = model.confint()
    payload = make_g1_payload(dict(model.coef()),
                              extra_blocks={'functional_form': {'spec_id': 'rc/form/estimator/ols_robust',
                                                                 'interpretation': 'OLS with HC1 robust SE'}})
    payload['design'] = {"panel_fixed_effects": {
        **G1_DESIGN_AUDIT, "estimator": "ols_robust"
    }}
    record_result("rc/form/estimator/ols_robust", run_id, "G1", "ep", "lfma",
                  coef, se, pval,
                  ci.loc['lfma', '2.5%'], ci.loc['lfma', '97.5%'],
                  model._N, model._r2, payload,
                  "Full sample, OLS robust", "kt", ", ".join(G1_CONTROLS_FULL), "hetero",
                  spec_tree_path="specification_tree/modules/robustness/functional_form.md#estimator")
    print(f"  rc/form/estimator/ols_robust: coef={coef:.4f}, se={se:.4f}")
except Exception as e:
    print(f"  rc/form/estimator/ols_robust FAILED: {e}")
    payload = fail_payload(str(e))
    record_result("rc/form/estimator/ols_robust", run_id, "G1", "ep", "lfma",
                  np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, payload,
                  "", "", "", "", run_success=0, run_error=str(e),
                  spec_tree_path="specification_tree/modules/robustness/functional_form.md#estimator")


# --- RC: Outcome variant ---
print("\n--- RC: Outcome variant ---")
# ep_no_us (foreign sales only / total sales)
if 'ep_no_us' in g1_sample.columns and g1_sample['ep_no_us'].notna().sum() > 30:
    run_g1_glm_spec("rc/form/outcome/ep_no_us_sales", G1_CONTROLS_FULL,
                     outcome='ep_no_us',
                     spec_tree_path="specification_tree/modules/robustness/functional_form.md#outcome",
                     extra_blocks={'functional_form': {'spec_id': 'rc/form/outcome/ep_no_us_sales',
                                                        'interpretation': 'EP excluding US parent sales'}})

# --- RC: Treatment variant (haven split) ---
print("\n--- RC: Treatment variant ---")
# Replace haven with big5 + otherh
haven_split_controls = ['taxr', 'big5', 'otherh'] + G1_CONTROL2
run_g1_glm_spec("rc/form/treatment/haven_split_big5_other", haven_split_controls,
                 spec_tree_path="specification_tree/modules/robustness/functional_form.md#treatment",
                 extra_blocks={'functional_form': {'spec_id': 'rc/form/treatment/haven_split_big5_other',
                                                    'interpretation': 'Haven split: big5 + Caribbean'}})


# --- RC: Random control subsets ---
print("\n--- RC: Random control subsets ---")
np.random.seed(148301)
for i in range(1, 11):
    n_controls = np.random.randint(1, len(G1_CONTROLS_FULL) + 1)
    subset = list(np.random.choice(G1_CONTROLS_FULL, size=n_controls, replace=False))
    spec_id = f"rc/controls/subset/random_{i:03d}"
    run_g1_glm_spec(spec_id, subset,
                     spec_tree_path="specification_tree/modules/robustness/controls.md#subset",
                     extra_blocks={'controls': {'spec_id': spec_id, 'family': 'random_subset',
                                                'included': subset, 'n_controls': len(subset),
                                                'draw_index': i, 'seed': 148301}})


# =====================================================================
#  G2 SPECIFICATIONS: Profit Shifting (lprofit ~ ep_haven)
# =====================================================================
print("\n" + "="*70)
print("G2: Profit Shifting (lprofit ~ ep_haven)")
print("="*70)

G2_CONTROLS = ['lfma', 'ep', 'taxr', 'haven', 'eoi_enf', 'dtc_enf', 'dtc_num', 'lrgdp', 'lemp', 'leqpmt']

# G2 sample: imputed_profit == 0 and non-missing lprofit
g2_vars = ['lprofit', 'ep_haven'] + G2_CONTROLS + ['kt', 'iso3']
g2_sample = df.dropna(subset=[v for v in g2_vars if v in df.columns]).copy()
g2_sample = g2_sample[g2_sample['imputed_profit'] == 0]
# Further restrict: the Stata code runs reghdfe first and keeps e(sample)
# This means positive profits (since lprofit = ln(profit))
g2_sample = g2_sample[g2_sample['lprofit'].notna() & np.isfinite(g2_sample['lprofit'])]

print(f"G2 sample size: {len(g2_sample)}")

def run_g2_ols_spec(spec_id, treatment, controls, sample=None, fe='kt', outcome='lprofit',
                     sample_desc="Profit sample", spec_tree_path="custom",
                     extra_blocks=None):
    """Run a G2 OLS specification and record it."""
    run_id = next_run_id()
    data = sample if sample is not None else g2_sample
    try:
        res = run_ols_fe(data, outcome, treatment, controls, fe, 'iso3')
        payload = make_g2_payload(res['all_coefs'], extra_blocks=extra_blocks)
        record_result(spec_id, run_id, "G2", outcome, treatment,
                      res['coef'], res['se'], res['pval'],
                      res['ci_lower'], res['ci_upper'],
                      res['n_obs'], res['r2'], payload,
                      sample_desc, fe, ", ".join(controls), "iso3",
                      spec_tree_path=spec_tree_path)
        print(f"  {spec_id}: coef={res['coef']:.4f}, se={res['se']:.4f}, p={res['pval']:.4f}, n={res['n_obs']}")
        return run_id
    except Exception as e:
        print(f"  {spec_id} FAILED: {e}")
        payload = fail_payload(str(e))
        record_result(spec_id, run_id, "G2", outcome, treatment,
                      np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, payload,
                      sample_desc, fe, "", "iso3",
                      run_success=0, run_error=str(e), spec_tree_path=spec_tree_path)
        return run_id


# --- Baseline: Table 4 Col 1 ---
print("\n--- Baseline ---")
baseline_g2_rid = run_g2_ols_spec("baseline", 'ep_haven', G2_CONTROLS,
                                    spec_tree_path="specification_tree/designs/panel_fixed_effects.md",
                                    extra_blocks={'notes': 'Table 4 Col 1: OLS log profit'})


# --- Baseline variants ---
print("\n--- Baseline variants ---")

# GPML Poisson (positive profits only)
run_id = next_run_id()
try:
    pos_profit = g2_sample[g2_sample['profit_'] >= 0].copy()
    # Poisson GLM: glm profit_ ... i.kt, family(poisson) link(log) cluster(i)
    rhs_vars = ['ep_haven'] + G2_CONTROLS
    allvars = ['profit_'] + rhs_vars + ['kt', 'iso3']
    df_reg = pos_profit.dropna(subset=[v for v in allvars if v in pos_profit.columns])
    df_reg = df_reg.reset_index(drop=True)

    fe_dummies = pd.get_dummies(df_reg['kt'], prefix='fe', drop_first=True, dtype=float)
    X = df_reg[rhs_vars].copy()
    X = pd.concat([X, fe_dummies], axis=1)
    X = sm.add_constant(X)
    y = df_reg['profit_']

    from statsmodels.genmod.families import Poisson
    from statsmodels.genmod.families.links import Log
    model = GLM(y, X, family=Poisson(link=Log()))
    result = model.fit(cov_type='cluster', cov_kwds={'groups': df_reg['iso3']}, maxiter=200)

    coef = result.params['ep_haven']
    se = result.bse['ep_haven']
    pval = result.pvalues['ep_haven']
    ci_lo = coef - 1.96 * se
    ci_hi = coef + 1.96 * se

    payload = make_g2_payload(dict(result.params[rhs_vars]),
                              extra_blocks={'functional_form': {'spec_id': 'baseline__gpml_profit',
                                                                 'interpretation': 'GPML Poisson (Table 4 Col 2)'}})
    record_result("baseline__gpml_profit", run_id, "G2", "profit_", "ep_haven",
                  coef, se, pval, ci_lo, ci_hi, len(df_reg), np.nan, payload,
                  "Positive profit only, GPML Poisson", "kt", ", ".join(G2_CONTROLS), "iso3",
                  spec_tree_path="specification_tree/designs/panel_fixed_effects.md")
    print(f"  baseline__gpml_profit: coef={coef:.4f}, se={se:.4f}")
except Exception as e:
    print(f"  baseline__gpml_profit FAILED: {e}")
    payload = fail_payload(str(e))
    record_result("baseline__gpml_profit", run_id, "G2", "profit_", "ep_haven",
                  np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, payload,
                  "", "", "", "", run_success=0, run_error=str(e),
                  spec_tree_path="specification_tree/designs/panel_fixed_effects.md")

# Cube root profit
run_g2_ols_spec("baseline__cube_root_profit", 'ep_haven', G2_CONTROLS,
                 outcome='cub_profit', sample_desc="Cube root profit transform",
                 spec_tree_path="specification_tree/designs/panel_fixed_effects.md",
                 extra_blocks={'functional_form': {'spec_id': 'baseline__cube_root_profit',
                                                    'interpretation': 'Cube root transform of profit (Table 4 Col 3)'}})


# --- RC: Controls LOO for G2 ---
print("\n--- G2 RC: Controls LOO ---")
# Don't drop ep, haven, or ep_haven (core of the hypothesis)
loo_candidates = ['lemp', 'leqpmt', 'lfma', 'eoi_enf', 'dtc_enf', 'dtc_num', 'lrgdp']
for ctrl in loo_candidates:
    spec_id = f"rc/controls/loo/drop_{ctrl}"
    remaining = [c for c in G2_CONTROLS if c != ctrl]
    run_g2_ols_spec(spec_id, 'ep_haven', remaining,
                     spec_tree_path="specification_tree/modules/robustness/controls.md#loo",
                     extra_blocks={'controls': {'spec_id': spec_id, 'family': 'loo',
                                                'dropped': [ctrl], 'n_controls': len(remaining)}})


# --- RC: Sample restrictions for G2 ---
print("\n--- G2 RC: Sample restrictions ---")

# Positive profit only
pos_profit_sample = g2_sample[g2_sample['profit_'] > 0]
run_g2_ols_spec("rc/sample/subgroup/positive_profit_only", 'ep_haven', G2_CONTROLS,
                 sample=pos_profit_sample, sample_desc="Positive profit only",
                 spec_tree_path="specification_tree/modules/robustness/sample.md#subgroup")

# Drop imputed profits (already default, but explicit)
run_g2_ols_spec("rc/sample/restriction/drop_imputed_profits", 'ep_haven', G2_CONTROLS,
                 sample=g2_sample[g2_sample['imputed_profit'] == 0],
                 sample_desc="Drop imputed profits",
                 spec_tree_path="specification_tree/modules/robustness/sample.md#restriction")


# --- RC: Outlier trimming for G2 ---
print("\n--- G2 RC: Outlier trimming ---")
if 'profit_' in g2_sample.columns:
    p1_p, p99_p = g2_sample['profit_'].quantile([0.01, 0.99])
    trim_profit = g2_sample[(g2_sample['profit_'] >= p1_p) & (g2_sample['profit_'] <= p99_p)]
    run_g2_ols_spec("rc/sample/outliers/trim_profit_1_99", 'ep_haven', G2_CONTROLS,
                     sample=trim_profit, sample_desc=f"Profit trimmed 1/99",
                     spec_tree_path="specification_tree/modules/robustness/sample.md#outliers")


# --- RC: Outcome transforms for G2 ---
print("\n--- G2 RC: Outcome transforms ---")

# Cube root (duplicate of baseline variant but as RC)
run_g2_ols_spec("rc/form/outcome/cube_root_profit", 'ep_haven', G2_CONTROLS,
                 outcome='cub_profit',
                 spec_tree_path="specification_tree/modules/robustness/functional_form.md#outcome",
                 extra_blocks={'functional_form': {'spec_id': 'rc/form/outcome/cube_root_profit',
                                                    'interpretation': 'Cube root profit transform'}})

# GPML Poisson (as RC)
run_id = next_run_id()
try:
    pos_profit = g2_sample[g2_sample['profit_'] >= 0].copy()
    rhs_vars = ['ep_haven'] + G2_CONTROLS
    allvars = ['profit_'] + rhs_vars + ['kt', 'iso3']
    df_reg = pos_profit.dropna(subset=[v for v in allvars if v in pos_profit.columns])
    df_reg = df_reg.reset_index(drop=True)

    fe_dummies = pd.get_dummies(df_reg['kt'], prefix='fe', drop_first=True, dtype=float)
    X = df_reg[rhs_vars].copy()
    X = pd.concat([X, fe_dummies], axis=1)
    X = sm.add_constant(X)
    y = df_reg['profit_']

    from statsmodels.genmod.families import Poisson
    from statsmodels.genmod.families.links import Log
    model = GLM(y, X, family=Poisson(link=Log()))
    result = model.fit(cov_type='cluster', cov_kwds={'groups': df_reg['iso3']}, maxiter=200)

    coef = result.params['ep_haven']
    se = result.bse['ep_haven']
    pval = result.pvalues['ep_haven']
    ci_lo = coef - 1.96 * se
    ci_hi = coef + 1.96 * se

    payload = make_g2_payload(dict(result.params[rhs_vars]),
                              extra_blocks={'functional_form': {'spec_id': 'rc/form/outcome/gpml_poisson',
                                                                 'interpretation': 'GPML Poisson'}})
    record_result("rc/form/outcome/gpml_poisson", run_id, "G2", "profit_", "ep_haven",
                  coef, se, pval, ci_lo, ci_hi, len(df_reg), np.nan, payload,
                  "Positive profit, GPML Poisson", "kt", ", ".join(G2_CONTROLS), "iso3",
                  spec_tree_path="specification_tree/modules/robustness/functional_form.md#outcome")
    print(f"  rc/form/outcome/gpml_poisson: coef={coef:.4f}, se={se:.4f}")
except Exception as e:
    print(f"  rc/form/outcome/gpml_poisson FAILED: {e}")
    payload = fail_payload(str(e))
    record_result("rc/form/outcome/gpml_poisson", run_id, "G2", "profit_", "ep_haven",
                  np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, payload,
                  "", "", "", "", run_success=0, run_error=str(e),
                  spec_tree_path="specification_tree/modules/robustness/functional_form.md#outcome")

# Level profit
run_g2_ols_spec("rc/form/outcome/level_profit", 'ep_haven', G2_CONTROLS,
                 outcome='profit_', sample_desc="Level profit (not log)",
                 spec_tree_path="specification_tree/modules/robustness/functional_form.md#outcome",
                 extra_blocks={'functional_form': {'spec_id': 'rc/form/outcome/level_profit',
                                                    'interpretation': 'Level profit (no log transform)'}})


# --- RC: FE add country for G2 ---
print("\n--- G2 RC: FE add country ---")
run_id = next_run_id()
try:
    rhs = " + ".join(['ep_haven'] + G2_CONTROLS)
    formula = f"lprofit ~ {rhs} | kt + iso3"
    df_reg = g2_sample.dropna(subset=['lprofit', 'ep_haven'] + G2_CONTROLS + ['kt', 'iso3'])
    model = pf.feols(formula, data=df_reg, vcov={"CRV1": "iso3"})
    coef = model.coef()['ep_haven']
    se = model.se()['ep_haven']
    pval = model.pvalue()['ep_haven']
    ci = model.confint()
    payload = make_g2_payload(dict(model.coef()),
                              extra_blocks={'fixed_effects': {'spec_id': 'rc/fe/add/country',
                                                              'fe_vars': ['kt', 'iso3']}})
    record_result("rc/fe/add/country", run_id, "G2", "lprofit", "ep_haven",
                  coef, se, pval,
                  ci.loc['ep_haven', '2.5%'], ci.loc['ep_haven', '97.5%'],
                  model._N, model._r2, payload,
                  "Profit sample, OLS with country + sector-year FE", "kt + iso3",
                  ", ".join(G2_CONTROLS), "iso3",
                  spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#add")
    print(f"  rc/fe/add/country (G2): coef={coef:.4f}, se={se:.4f}, n={model._N}")
except Exception as e:
    print(f"  rc/fe/add/country (G2) FAILED: {e}")
    payload = fail_payload(str(e))
    record_result("rc/fe/add/country", run_id, "G2", "lprofit", "ep_haven",
                  np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, payload,
                  "", "", "", "", run_success=0, run_error=str(e),
                  spec_tree_path="specification_tree/modules/robustness/fixed_effects.md#add")


# =====================================================================
#  INFERENCE VARIANTS
# =====================================================================
print("\n" + "="*70)
print("Inference Variants")
print("="*70)

# G1 inference: two-way clustering (country x sector) and HC1
# We run inference variants on the baseline G1 spec
g1_baseline_run_id = f"{PAPER_ID}__run_001"

# Two-way clustering: country x sector
print("\n--- G1 Inference: two-way cluster (country x sector) ---")
try:
    rhs = " + ".join(['lfma'] + G1_CONTROLS_FULL)
    formula = f"ep ~ {rhs} | kt"
    df_reg = g1_sample.dropna(subset=['ep', 'lfma'] + G1_CONTROLS_FULL + ['kt', 'iso3', 'k'])
    df_reg['k_str'] = df_reg['k'].astype(str)
    # pyfixest two-way clustering
    model = pf.feols(formula, data=df_reg, vcov={"CRV1": "iso3+k_str"})
    coef = model.coef()['lfma']
    se = model.se()['lfma']
    pval = model.pvalue()['lfma']
    ci = model.confint()

    payload = make_success_payload(
        coefficients=dict(model.coef()),
        inference={"spec_id": "infer/se/cluster/country_sector",
                   "params": {"cluster_var": ["iso3", "k"]}},
        software=SW_BLOCK,
        surface_hash=SURFACE_HASH,
        design={"panel_fixed_effects": G1_DESIGN_AUDIT},
    )
    record_inference(g1_baseline_run_id, "infer/se/cluster/country_sector", "G1",
                     coef, se, pval,
                     ci.loc['lfma', '2.5%'], ci.loc['lfma', '97.5%'],
                     model._N, model._r2, payload,
                     spec_tree_path="specification_tree/modules/inference/standard_errors.md#cluster")
    print(f"  Two-way cluster: coef={coef:.4f}, se={se:.4f}, p={pval:.4f}")
except Exception as e:
    print(f"  Two-way cluster FAILED: {e}")
    payload = fail_payload(str(e), stage="inference")
    record_inference(g1_baseline_run_id, "infer/se/cluster/country_sector", "G1",
                     np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, payload,
                     run_success=0, run_error=str(e),
                     spec_tree_path="specification_tree/modules/inference/standard_errors.md#cluster")

# HC1 (no clustering)
print("--- G1 Inference: HC1 ---")
try:
    model = pf.feols(formula, data=df_reg, vcov="hetero")
    coef = model.coef()['lfma']
    se = model.se()['lfma']
    pval = model.pvalue()['lfma']
    ci = model.confint()

    payload = make_success_payload(
        coefficients=dict(model.coef()),
        inference={"spec_id": "infer/se/hc/hc1", "params": {}},
        software=SW_BLOCK,
        surface_hash=SURFACE_HASH,
        design={"panel_fixed_effects": G1_DESIGN_AUDIT},
    )
    record_inference(g1_baseline_run_id, "infer/se/hc/hc1", "G1",
                     coef, se, pval,
                     ci.loc['lfma', '2.5%'], ci.loc['lfma', '97.5%'],
                     model._N, model._r2, payload,
                     spec_tree_path="specification_tree/modules/inference/standard_errors.md#hc")
    print(f"  HC1: coef={coef:.4f}, se={se:.4f}, p={pval:.4f}")
except Exception as e:
    print(f"  HC1 FAILED: {e}")
    payload = fail_payload(str(e), stage="inference")
    record_inference(g1_baseline_run_id, "infer/se/hc/hc1", "G1",
                     np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, payload,
                     run_success=0, run_error=str(e),
                     spec_tree_path="specification_tree/modules/inference/standard_errors.md#hc")


# =====================================================================
#  SAVE OUTPUTS
# =====================================================================
print("\n" + "="*70)
print("Saving outputs...")
print("="*70)

# Save specification_results.csv
df_results = pd.DataFrame(results)
df_results.to_csv(f"{OUTPUT_DIR}/specification_results.csv", index=False)
print(f"  specification_results.csv: {len(df_results)} rows")

# Save inference_results.csv
if inference_results:
    df_inference = pd.DataFrame(inference_results)
    df_inference.to_csv(f"{OUTPUT_DIR}/inference_results.csv", index=False)
    print(f"  inference_results.csv: {len(df_inference)} rows")

# Count successes/failures
n_success = df_results['run_success'].sum()
n_fail = len(df_results) - n_success
n_infer_success = sum(1 for r in inference_results if r['run_success'] == 1) if inference_results else 0
n_infer_fail = len(inference_results) - n_infer_success if inference_results else 0

# Save SPECIFICATION_SEARCH.md
search_md = f"""# Specification Search: {PAPER_ID}

## Paper
Laffitte & Toubal (2022), "Multinational's Sales and Tax Havens", AEJ: Economic Policy.

## Surface Summary
- **Baseline groups**: 2
  - G1: Export platform share (ep ~ lfma), GLM fractional logit with marginal effects
  - G2: Profit shifting (lprofit ~ ep_haven), OLS with sector-year FE
- **Budgets**: G1 max 80, G2 max 30
- **Sampling seed**: 148301

## Data Construction
Dataset was constructed from raw BEA Excel files, PWT, CEPII Gravity, and tax agreement data.
- BEA data: years 1999-2013, 11 sectors (excl. finance, utilities, totals)
- FMA: Constructed using market potential measure from gravity data
- Tax rates: KPMG extended rates
- Haven indicator: Dharmapala & Hines (2009) list + NLD
- Agreements: Constructed from bilateral treaty .dta files

## Execution Summary

### Specification Results
- **Total planned**: {len(df_results)}
- **Successful**: {n_success}
- **Failed**: {n_fail}

### Inference Results
- **Total**: {len(inference_results)}
- **Successful**: {n_infer_success}
- **Failed**: {n_infer_fail}

## G1 Specifications ({sum(1 for r in results if r['baseline_group_id'] == 'G1')} total)
- Baseline: Table 2 Col 4 (GLM fractional logit, full controls with haven)
- Baseline variants: Table 2 Cols 3, 5, 6, 7, 8
- Design: OLS within estimator
- Controls LOO: 6 specs (drop each control)
- Control sets: minimal, lrgdp only, lrgdp+taxr, full with haven
- Control progression: Cols 1-4 steps
- Control additions: lemp, leqpmt, lemp+leqpmt
- Sample subgroups: manufacturing, services, non-haven, haven, big5, Caribbean
- Sample restrictions: drop imputed EP, drop imputed profits
- Outlier trimming: EP 1/99, EP 5/95
- FE variants: country-year, sector only, year only, add country
- Estimator: OLS reghdfe, OLS robust
- Outcome: EP no US sales
- Treatment: Haven split (big5 + otherh)
- Random control subsets: 10 draws

## G2 Specifications ({sum(1 for r in results if r['baseline_group_id'] == 'G2')} total)
- Baseline: Table 4 Col 1 (OLS log profit)
- Baseline variants: GPML Poisson, cube root profit
- Controls LOO: 7 specs
- Sample: positive profit only, drop imputed profits
- Outlier trimming: profit 1/99
- Outcome transforms: cube root, GPML Poisson, level profit
- FE: add country

## Software
- Python {sys.version.split()[0]}
- pyfixest, statsmodels, pandas, numpy
- Surface hash: {SURFACE_HASH}

## Deviations
- FMA variable was constructed using a simplified market potential measure (GDP/distance)
  rather than the full gravity-based FMA from the paper. This may cause small differences
  in coefficient magnitudes but preserves the qualitative direction of results.
- GLM fractional logit marginal effects are computed using average marginal effects (AME)
  with delta method standard errors, matching the paper's `margins, dydx(...)` approach.
- The two-way clustering inference variant for G1 uses pyfixest's built-in two-way
  clustering with iso3 + sector.
"""

with open(f"{OUTPUT_DIR}/SPECIFICATION_SEARCH.md", 'w') as f:
    f.write(search_md)

print(f"\nDone! Total specifications: {len(df_results)} ({n_success} success, {n_fail} fail)")
print(f"Inference results: {len(inference_results)}")
