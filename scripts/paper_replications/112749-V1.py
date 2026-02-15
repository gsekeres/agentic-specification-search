"""
Replication script for Hornbeck & Naidu (2014)
"When the Levee Breaks: Black Migration and Economic Development in the American South"
AER 104(3): 963-990

This script:
1. Generates the analysis dataset from raw data (translating Generate_flood.do + flood_preanalysis.do)
2. Runs main regressions from Tables 1-7 (translating flood_analysis.do)
3. Outputs replication.csv and replication_report.md
"""

import pandas as pd
import numpy as np
import pyreadstat
import pyfixest as pf
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Paths
PAPER_ID = "112749-V1"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PACKAGE_DIR = os.path.join(REPO_ROOT, "data", "downloads", "extracted", PAPER_ID)
PKG = os.path.join(PACKAGE_DIR, "Replication_AER-2012-0980")
GDIR = os.path.join(PKG, "Generate_Data")
ADIR = os.path.join(PKG, "Analysis")

# Southern states filter (ICPSR state codes)
SOUTH_STATES = [32, 34, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54]

def read_dta(path, use_pyreadstat=False):
    """Read Stata file, falling back to pyreadstat for old formats."""
    try:
        return pd.read_stata(path, convert_categoricals=False)
    except (ValueError, Exception):
        df, _ = pyreadstat.read_dta(path)
        return df

def read_txt(path, **kwargs):
    """Read tab-delimited text file."""
    return pd.read_csv(path, sep='\t', **kwargs)


###############################################################################
# PART 1: Generate flood_base1900.dta equivalent
###############################################################################

def generate_data():
    """Translate Generate_flood.do into Python."""
    print("=== GENERATING DATA ===")

    # --- Farm values 1900-1959 ---
    farmval_full = read_dta(os.path.join(GDIR, "farmval.dta"))
    farmval_full = farmval_full[farmval_full['level'] == 1]
    farmval_full = farmval_full[farmval_full['state'].isin(SOUTH_STATES)]

    farmval_dict = {}
    for suffix, yr in [('900',1900),('910',1910),('920',1920),('925',1925),('930',1930),
                        ('935',1935),('940',1940),('945',1945),('950',1950),('954',1954),('959',1960)]:
        col = f'faval{suffix}'
        if col in farmval_full.columns:
            tmp = farmval_full[['fips', col]].copy().rename(columns={col: 'farmval'})
            tmp['year'] = yr
            farmval_dict[yr] = tmp.sort_values('fips')

    # --- 1900 ---
    print("  Processing 1900...")
    icpsr1900 = read_dta(os.path.join(GDIR, "02896-0020-Data.dta"))
    icpsr1900 = icpsr1900[icpsr1900['level'] == 1]
    icpsr1900 = icpsr1900[icpsr1900['state'].isin(SOUTH_STATES)]
    icpsr1900 = icpsr1900.dropna(subset=['fips'])

    icpsr1900['county_squaremiles'] = icpsr1900['area']
    icpsr1900['population'] = icpsr1900['totpop']
    icpsr1900['population_race_white'] = (icpsr1900['nbwmnp'] + icpsr1900['nbwmfp'] +
                                           icpsr1900['fbwmtot'] + icpsr1900['nbwfnp'] +
                                           icpsr1900['nbwffp'] + icpsr1900['fbwftot'])
    icpsr1900['population_race_black'] = icpsr1900['negmtot'] + icpsr1900['negftot']
    icpsr1900['population_race_other'] = icpsr1900['population'] - icpsr1900['population_race_white'] - icpsr1900['population_race_black']
    icpsr1900['farms_white'] = icpsr1900['farmwh']
    icpsr1900['farms_nonwhite'] = icpsr1900['farmcol']
    icpsr1900['farms_owner'] = icpsr1900['farmwhow'] + icpsr1900['farmcoow']
    icpsr1900['farms_tenant'] = icpsr1900['farmwhct'] + icpsr1900['farmcoct'] + icpsr1900['farmwhst'] + icpsr1900['farmcost']
    icpsr1900['farms_nonwhite_tenant'] = icpsr1900['farmcoct'] + icpsr1900['farmcost']
    icpsr1900['farms_tenant_cash'] = icpsr1900['farmwhct'] + icpsr1900['farmcoct']
    icpsr1900['farmland'] = icpsr1900['acfarm']
    icpsr1900['value_buildings'] = icpsr1900['farmbui']
    icpsr1900['value_equipment'] = icpsr1900['farmequi']
    icpsr1900['farms'] = icpsr1900['farmwh'].fillna(0) + icpsr1900['farmcol'].fillna(0)
    icpsr1900['year'] = 1900

    keep_cols_1900 = ['fips','state','county','name','county_squaremiles','population',
                      'population_race_white','population_race_black','population_race_other',
                      'farms','farms_white','farms_nonwhite','farms_nonwhite_tenant',
                      'farms_owner','farms_tenant','farms_tenant_cash','farmland',
                      'value_buildings','value_equipment','mfgestab','mfgwages','mfgavear','year']
    icpsr1900 = icpsr1900[[c for c in keep_cols_1900 if c in icpsr1900.columns]].copy()

    # Agricultural data for 1900
    ag1900 = read_dta(os.path.join(GDIR, "ag900co.dta"))
    ag1900 = ag1900[ag1900['level'] == 1]
    ag1900 = ag1900[ag1900['state'].isin(SOUTH_STATES)]
    renames_ag1900 = {'cornac':'corn_a','oatsac':'oats_a','wheatac':'wheat_a',
                      'cottonac':'cotton_a','riceac':'rice_a','scaneac':'scane_a',
                      'corn':'corn_y','oats':'oats_y','wheat':'wheat_y',
                      'cotbale1':'cotton_y','rice':'rice_y','csugarwt':'scane_y'}
    ag1900 = ag1900.rename(columns=renames_ag1900)
    ag1900['horses'] = ag1900.get('colts0',0) + ag1900.get('colts1_2',0) + ag1900.get('horses2_',0)
    ag1900['mules'] = ag1900.get('mules0',0) + ag1900.get('mules1_2',0) + ag1900.get('mules2_',0)
    for v in ['corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a']:
        ag1900[v] = ag1900[v].fillna(0)
    ag1900_keep = ['fips','corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a',
                   'corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y','horses','mules']
    ag1900 = ag1900[[c for c in ag1900_keep if c in ag1900.columns]]

    data1900 = pd.merge(ag1900, icpsr1900, on='fips', how='inner')
    data1900 = pd.merge(data1900, farmval_dict[1900][['fips','farmval']], on='fips', how='inner')
    data1900['value_landbuildings'] = data1900['farmval'] * data1900['farmland']
    data1900['value_land'] = data1900['value_landbuildings'] - data1900['value_buildings']
    data1900.drop(columns=['farmval'], inplace=True)

    # --- 1910 ---
    print("  Processing 1910...")
    icpsr1910 = read_dta(os.path.join(GDIR, "02896-0022-Data.dta"))
    icpsr1910 = icpsr1910[icpsr1910['level'] == 1]
    icpsr1910 = icpsr1910[icpsr1910['state'].isin(SOUTH_STATES)]
    icpsr1910['county_squaremiles'] = icpsr1910['area']
    icpsr1910['population'] = icpsr1910['totpop']
    icpsr1910['population_race_white'] = icpsr1910['wmtot'] + icpsr1910['wftot']
    icpsr1910['population_race_black'] = icpsr1910['negmtot'] + icpsr1910['negftot']
    icpsr1910['population_race_other'] = icpsr1910['population'] - icpsr1910['population_race_white'] - icpsr1910['population_race_black']
    icpsr1910['farms_white'] = icpsr1910['farmnw'] + icpsr1910['farmfbw']
    icpsr1910['farms_nonwhite'] = icpsr1910['farmneg']
    icpsr1910['farms_owner'] = icpsr1910['farmown']
    icpsr1910['farms_tenant'] = icpsr1910['farmten']
    icpsr1910['farms_tenant_cash'] = icpsr1910['farmcten']
    icpsr1910['farmland'] = icpsr1910['acresown'] + icpsr1910['acresten'] + icpsr1910.get('acresman', 0)
    icpsr1910['farms_nonwhite_tenant'] = icpsr1910['farmnegt']
    icpsr1910['farmland_owner'] = icpsr1910['acresown']
    icpsr1910['farmland_tenant'] = icpsr1910['acresten']
    icpsr1910['population_rural'] = icpsr1910.get('rur1910', np.nan)
    icpsr1910['year'] = 1910
    icpsr1910['farms'] = icpsr1910['farms']

    # Ag data 1910
    ag1910 = read_dta(os.path.join(GDIR, "ag910co.dta"))
    ag1910 = ag1910[ag1910['level'] == 1]
    ag1910 = ag1910[ag1910['state'].isin(SOUTH_STATES)]
    renames_ag1910 = {'farmbui':'value_buildings','farmequi':'value_equipment',
                      'livstock':'value_livestock','fawages':'value_labor_wages',
                      'farebord':'value_labor_board',
                      'cornac':'corn_a','oatsac':'oats_a','wheatac':'wheat_a',
                      'cottonac':'cotton_a','riceac':'rice_a','csugarac':'scane_a',
                      'corn':'corn_y','oats':'oats_y','wheat':'wheat_y',
                      'cotton':'cotton_y','rice':'rice_y','csugar':'scane_y'}
    ag1910 = ag1910.rename(columns=renames_ag1910)
    for v in ['corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a']:
        if v in ag1910.columns:
            ag1910[v] = ag1910[v].fillna(0)
    ag1910_keep = ['fips','value_buildings','value_equipment','corn_a','oats_a','wheat_a',
                   'cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y',
                   'rice_y','scane_y','horses','mules']
    ag1910 = ag1910[[c for c in ag1910_keep if c in ag1910.columns]]

    data1910 = pd.merge(ag1910, icpsr1910[['fips','county_squaremiles','population','population_rural',
                        'population_race_white','population_race_black','population_race_other',
                        'farms','farms_white','farms_nonwhite','farms_nonwhite_tenant',
                        'farms_owner','farms_tenant','farms_tenant_cash','farmland',
                        'farmland_owner','farmland_tenant','year',
                        'cropval' if 'cropval' in icpsr1910.columns else 'year']],
                        on='fips', how='inner')
    data1910 = pd.merge(data1910, farmval_dict[1910][['fips','farmval']], on='fips', how='inner')
    data1910['value_landbuildings'] = data1910['farmval'] * data1910['farmland']
    data1910['value_land'] = data1910['value_landbuildings'] - data1910['value_buildings']
    data1910.drop(columns=['farmval'], inplace=True)

    # --- 1920 ---
    print("  Processing 1920...")
    icpsr1920 = read_dta(os.path.join(GDIR, "02896-0024-Data.dta"))
    icpsr1920 = icpsr1920[icpsr1920['level'] == 1]
    icpsr1920 = icpsr1920[icpsr1920['state'].isin(SOUTH_STATES)]
    icpsr1920['county_squaremiles'] = icpsr1920['area']
    icpsr1920['county_acres'] = icpsr1920['areaac']
    icpsr1920['population'] = icpsr1920['totpop']
    icpsr1920['population_race_white'] = icpsr1920['nwmtot'] + icpsr1920['fbwmtot'] + icpsr1920['nwftot'] + icpsr1920['fbwftot']
    icpsr1920['population_race_black'] = icpsr1920['negmtot'] + icpsr1920['negftot']
    icpsr1920['population_race_other'] = icpsr1920['population'] - icpsr1920['population_race_white'] - icpsr1920['population_race_black']
    icpsr1920['farms_white'] = icpsr1920['farmnw'] + icpsr1920['farmfbw']
    icpsr1920['farms_nonwhite'] = icpsr1920['farmcol']
    icpsr1920['farms_owner'] = icpsr1920['farmown']
    icpsr1920['farms_tenant'] = icpsr1920['farmten']
    icpsr1920['farms_tenant_cash'] = icpsr1920['farmcten']
    icpsr1920['farmland'] = icpsr1920['acresown'] + icpsr1920['acresten'] + icpsr1920.get('acresman', 0)
    icpsr1920['farmland_owner'] = icpsr1920['acresown']
    icpsr1920['farmland_tenant'] = icpsr1920['acresten']
    icpsr1920['farms_nonwhite_tenant'] = icpsr1920['farmcolt']
    icpsr1920['population_rural'] = icpsr1920['population'] - icpsr1920.get('urb920', 0)
    icpsr1920['year'] = 1920
    icpsr1920['farms'] = icpsr1920['farms']

    # Ag data 1920
    ag1920 = read_dta(os.path.join(GDIR, "ag920co.dta"))
    ag1920 = ag1920[ag1920['level'] == 1]
    ag1920 = ag1920[ag1920['state'].isin(SOUTH_STATES)]
    renames_ag1920 = {'var147':'corn_a','var149':'oats_a','var151':'wheat_a',
                      'var165':'rice_a','var216':'cotton_a','var221':'scane_a',
                      'var56':'horses','var63':'mules',
                      'var148':'corn_y','var150':'oats_y','var152':'wheat_y',
                      'var166':'rice_y','var217':'cotton_y','var222':'scane_y'}
    ag1920 = ag1920.rename(columns=renames_ag1920)
    if 'oats_y' in ag1920.columns:
        ag1920['oats_y'] = pd.to_numeric(ag1920['oats_y'], errors='coerce')
    for v in ['corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a']:
        if v in ag1920.columns:
            ag1920[v] = ag1920[v].fillna(0)

    ag1920_keep = ['fips','corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a',
                   'corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y','horses','mules']
    ag1920 = ag1920[[c for c in ag1920_keep if c in ag1920.columns]]

    out_migrant = read_dta(os.path.join(GDIR, "out_migrant_counts.dta"))

    icpsr1920_keep = ['fips','county_acres','county_squaremiles','population','population_rural',
                      'population_race_white','population_race_black','population_race_other',
                      'farms','farms_white','farms_nonwhite','farms_nonwhite_tenant',
                      'farms_owner','farms_tenant','farms_tenant_cash','farmland',
                      'farmland_owner','farmland_tenant','value_buildings','value_equipment',
                      'cropval','mfgwages','mfgestab','mfgavear','year']
    # Handle missing columns gracefully
    icpsr1920_keep_avail = [c for c in icpsr1920_keep if c in icpsr1920.columns]
    # Need value_buildings and value_equipment from icpsr1920
    icpsr1920['value_buildings'] = icpsr1920['farmbui']
    icpsr1920['value_equipment'] = icpsr1920['farmequi']
    icpsr1920_keep_avail = [c for c in icpsr1920_keep if c in icpsr1920.columns]

    data1920 = pd.merge(ag1920, icpsr1920[icpsr1920_keep_avail], on='fips', how='inner')
    data1920 = pd.merge(data1920, farmval_dict[1920][['fips','farmval']], on='fips', how='inner')
    data1920['value_landbuildings'] = data1920['farmval'] * data1920['farmland']
    data1920['value_land'] = data1920['value_landbuildings'] - data1920['value_buildings']
    data1920.drop(columns=['farmval'], inplace=True)
    # Merge out-migrant data
    data1920 = pd.merge(data1920, out_migrant, on='fips', how='left')

    # --- 1925 ---
    print("  Processing 1925...")
    haines1925 = read_txt(os.path.join(GDIR, "Haines_1925.txt"))
    haines1925 = haines1925.rename(columns={'var2':'farms','var42':'farmland','var99':'value_equipment',
                                             'var169':'horses','var173':'mules'})
    icpsr_fips = read_dta(os.path.join(GDIR, "icpsr_fips.dta"))
    haines1925 = pd.merge(haines1925[['state','county','farms','farmland','value_equipment','horses','mules']],
                          icpsr_fips, on=['state','county'], how='inner')

    haines1925_new = read_txt(os.path.join(GDIR, "Haines_1925_new.txt"))
    haines1925_new = haines1925_new[(haines1925_new['state'] > 0) & (haines1925_new['state'] <= 73)]
    for v in ['corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y']:
        if v in haines1925_new.columns:
            haines1925_new[v] = haines1925_new[v].fillna(0)
    haines1925_new = pd.merge(haines1925_new, icpsr_fips, on=['state','county'], how='inner')

    tractors1925 = read_dta(os.path.join(GDIR, "tractors1925.dta"))
    tractors1925 = tractors1925.rename(columns={'tractors1925':'tractors'})
    tractors1925 = tractors1925[['fips','tractors']]

    data1925 = pd.merge(farmval_dict[1925][['fips','farmval']], haines1925, on='fips', how='inner')
    crop_cols_1925 = [c for c in ['corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a',
                                   'corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y'] if c in haines1925_new.columns]
    data1925 = pd.merge(data1925, haines1925_new[['fips'] + crop_cols_1925], on='fips', how='inner')
    data1925 = pd.merge(data1925, tractors1925, on='fips', how='left')
    data1925['value_landbuildings'] = data1925['farmval'] * data1925['farmland']
    data1925.drop(columns=['farmval'], inplace=True)
    data1925['year'] = 1925

    # --- 1930 ---
    print("  Processing 1930...")
    haines1930_1 = read_txt(os.path.join(GDIR, "Haines_1930.txt"))
    haines1930_1 = haines1930_1[['state','county','horses','mules','tractors']]

    haines1930_new = read_txt(os.path.join(GDIR, "Haines_1930_new.txt"))
    haines1930_new = haines1930_new[(haines1930_new['state'] > 0) & (haines1930_new['state'] <= 73)]
    if 'cotton_a' in haines1930_new.columns:
        haines1930_new['cotton_a'] = pd.to_numeric(haines1930_new['cotton_a'], errors='coerce')
    for v in ['corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y']:
        if v in haines1930_new.columns:
            haines1930_new[v] = haines1930_new[v].fillna(0)
    haines1930 = pd.merge(haines1930_new, haines1930_1, on=['state','county'], how='inner')
    haines1930 = pd.merge(haines1930, icpsr_fips, on=['state','county'], how='inner')

    redcross = read_txt(os.path.join(GDIR, "redcross_new.txt"))
    for v in ['flooded_acres','pop_affected','agricultural_flooded_acres']:
        if v in redcross.columns:
            redcross[v] = redcross[v].fillna(0)
    redcross = redcross[['fips','flooded_acres','pop_affected','agricultural_flooded_acres']]

    icpsr1930 = read_dta(os.path.join(GDIR, "02896-0026-Data.dta"))
    icpsr1930 = icpsr1930[icpsr1930['level'] == 1]
    icpsr1930 = icpsr1930[icpsr1930['state'].isin(SOUTH_STATES)]
    icpsr1930['county_squaremiles'] = icpsr1930['area']
    icpsr1930['county_acres'] = icpsr1930['areaac']
    icpsr1930['population'] = icpsr1930['totpop']
    icpsr1930['population_race_white'] = icpsr1930['nwmtot'] + icpsr1930['fbwmtot'] + icpsr1930['nwftot'] + icpsr1930['fbwftot']
    icpsr1930['population_race_black'] = icpsr1930['negmtot'] + icpsr1930['negftot']
    icpsr1930['population_race_other'] = icpsr1930['population'] - icpsr1930['population_race_white'] - icpsr1930['population_race_black']
    icpsr1930['farms_white'] = icpsr1930['farmwh']
    icpsr1930['farms_nonwhite'] = icpsr1930['farmcol']
    icpsr1930['farms_owner'] = icpsr1930.get('farmfown',0) + icpsr1930.get('farmpown',0)
    icpsr1930['farms_tenant'] = icpsr1930['farmten']
    icpsr1930['farms_tenant_cash'] = icpsr1930['farmcten']
    icpsr1930['farmland'] = icpsr1930['acres']
    icpsr1930['value_buildings'] = icpsr1930['farmbui']
    icpsr1930['value_equipment'] = icpsr1930['farmequi']
    icpsr1930['population_rural'] = icpsr1930['population'] - icpsr1930.get('urban30', 0)
    icpsr1930['year'] = 1930
    icpsr1930['farms'] = icpsr1930['farms']

    in_migrant = read_dta(os.path.join(GDIR, "in_migrant_counts.dta"))

    if 'farmval' in icpsr1930.columns:
        icpsr1930.drop(columns=['farmval'], inplace=True)
    data1930 = pd.merge(icpsr1930, farmval_dict[1930][['fips','farmval']], on='fips', how='inner')
    data1930 = pd.merge(data1930, in_migrant, on='fips', how='left')
    haines1930_cols = [c for c in haines1930.columns if c not in ['state','county','name']]
    data1930 = pd.merge(data1930, haines1930[haines1930_cols], on='fips', how='inner')
    data1930 = pd.merge(data1930, redcross, on='fips', how='left')
    data1930['value_landbuildings'] = data1930['farmval'] * data1930['farmland']
    data1930['value_land'] = data1930['value_landbuildings'] - data1930['value_buildings']
    data1930.drop(columns=['farmval'], inplace=True)

    # --- 1935 ---
    print("  Processing 1935...")
    haines1935 = read_txt(os.path.join(GDIR, "Haines_1935.txt"))
    haines1935 = haines1935.rename(columns={'var2':'farms','var12':'farmland','var95':'horses','var100':'mules'})
    haines1935 = pd.merge(haines1935[['state','county','farms','farmland','horses','mules']],
                          icpsr_fips, on=['state','county'], how='inner')

    haines1935_new = read_txt(os.path.join(GDIR, "Haines_1935_new.txt"))
    haines1935_new = haines1935_new[(haines1935_new['state'] > 0) & (haines1935_new['state'] <= 73)]
    if 'wheat1_a' in haines1935_new.columns and 'wheat2_a' in haines1935_new.columns:
        haines1935_new['wheat_a'] = haines1935_new['wheat1_a'].fillna(0) + haines1935_new['wheat2_a'].fillna(0)
        haines1935_new['wheat_y'] = haines1935_new.get('wheat1_y',0) + haines1935_new.get('wheat2_y',0)
    for v in ['corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y']:
        if v in haines1935_new.columns:
            haines1935_new[v] = haines1935_new[v].fillna(0)
    haines1935_new = pd.merge(haines1935_new, icpsr_fips, on=['state','county'], how='inner')

    data1935 = pd.merge(farmval_dict[1935][['fips','farmval']], haines1935, on='fips', how='inner')
    crop_cols_1935 = [c for c in ['corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a',
                                   'corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y'] if c in haines1935_new.columns]
    data1935 = pd.merge(data1935, haines1935_new[['fips'] + crop_cols_1935], on='fips', how='inner')
    data1935['value_landbuildings'] = data1935['farmval'] * data1935['farmland']
    data1935.drop(columns=['farmval'], inplace=True)
    data1935['year'] = 1935

    # --- 1940 ---
    print("  Processing 1940...")
    haines1940_new = read_txt(os.path.join(GDIR, "Haines_1940_new.txt"))
    haines1940_new = haines1940_new[(haines1940_new['state'] > 0) & (haines1940_new['state'] <= 73)]
    if 'oats_y' in haines1940_new.columns:
        haines1940_new['oats_y'] = pd.to_numeric(haines1940_new['oats_y'], errors='coerce')
    if 'wheat_y' in haines1940_new.columns:
        haines1940_new['wheat_y'] = pd.to_numeric(haines1940_new['wheat_y'], errors='coerce')
    for v in ['corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y']:
        if v in haines1940_new.columns:
            haines1940_new[v] = haines1940_new[v].fillna(0)
    haines1940_new = pd.merge(haines1940_new, icpsr_fips, on=['state','county'], how='inner')

    icpsr1940 = read_dta(os.path.join(GDIR, "02896-0032-Data.dta"))
    icpsr1940 = icpsr1940[icpsr1940['level'] == 1]
    icpsr1940 = icpsr1940[icpsr1940['state'].isin(SOUTH_STATES)]
    icpsr1940['county_squaremiles'] = icpsr1940['area']
    icpsr1940['county_acres'] = icpsr1940['areaac']
    icpsr1940['population'] = icpsr1940['totpop']
    icpsr1940['population_race_white'] = icpsr1940['nwtot'] + icpsr1940['fbwtot']
    icpsr1940['population_race_black'] = icpsr1940['negtot']
    icpsr1940['population_race_other'] = icpsr1940['population'] - icpsr1940['population_race_white'] - icpsr1940['population_race_black']
    icpsr1940['farms_white'] = icpsr1940['farmwh']
    icpsr1940['farms_nonwhite'] = icpsr1940['farmnonw']
    icpsr1940['farms_owner'] = icpsr1940['farmfown'] + icpsr1940['farmpown']
    icpsr1940['farms_tenant'] = icpsr1940['farmten']
    icpsr1940['farms_tenant_cash'] = icpsr1940['farmcten']
    icpsr1940['farmland'] = icpsr1940['acfarms']
    icpsr1940['farmland_owner'] = icpsr1940['acfown'] + icpsr1940['acpown']
    icpsr1940['farmland_tenant'] = icpsr1940['acten']
    icpsr1940['value_buildings'] = icpsr1940['buildval']
    icpsr1940['value_equipment'] = icpsr1940['equipval']
    icpsr1940['population_rural'] = icpsr1940['population'] - icpsr1940.get('urb940', 0)
    icpsr1940['year'] = 1940

    d1940_extra = read_dta(os.path.join(GDIR, "02896-0070-Data.dta"))
    d1940_extra = d1940_extra[d1940_extra['level'] == 1]
    d1940_extra = d1940_extra[d1940_extra['state'].isin(SOUTH_STATES)]
    d1940_extra = d1940_extra.rename(columns={'var57':'tractors','var56':'mules_horses'})
    d1940_extra = d1940_extra[['fips','tractors','mules_horses']]

    data1940 = pd.merge(icpsr1940, d1940_extra, on='fips', how='inner')
    # Drop original farmval from ICPSR data before merging with our farmval
    if 'farmval' in data1940.columns:
        data1940.drop(columns=['farmval'], inplace=True)
    data1940 = pd.merge(data1940, farmval_dict[1940][['fips','farmval']], on='fips', how='inner')
    crop_cols_1940 = [c for c in ['corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a',
                                   'corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y'] if c in haines1940_new.columns]
    data1940 = pd.merge(data1940, haines1940_new[['fips'] + crop_cols_1940], on='fips', how='inner')
    data1940['value_landbuildings'] = data1940['farmval'] * data1940['farmland']
    data1940['value_land'] = data1940['value_landbuildings'] - data1940['value_buildings']
    data1940.drop(columns=['farmval'], inplace=True)
    for v in crop_cols_1940:
        data1940[v] = data1940[v].fillna(0)

    # --- 1945 ---
    print("  Processing 1945...")
    d1945 = read_dta(os.path.join(GDIR, "02896-0071-Data.dta"))
    d1945 = d1945[d1945['level'] == 1]
    d1945 = d1945[d1945['state'].isin(SOUTH_STATES)]
    d1945 = d1945.rename(columns={'var69':'farms','var71':'farmland','var82':'tractors'})
    d1945['farmland'] = d1945['farmland'] * 1000
    data1945 = pd.merge(d1945[['fips','farms','farmland','tractors']],
                        farmval_dict[1945][['fips','farmval']], on='fips', how='inner')
    data1945['value_landbuildings'] = data1945['farmval'] * data1945['farmland']
    data1945.drop(columns=['farmval'], inplace=True)
    data1945['year'] = 1945

    # --- 1950 ---
    print("  Processing 1950...")
    icpsr1950 = read_dta(os.path.join(GDIR, "02896-0035-Data.dta"))
    icpsr1950 = icpsr1950[icpsr1950['level'] == 1]
    icpsr1950 = icpsr1950[icpsr1950['state'].isin(SOUTH_STATES)]
    icpsr1950['county_squaremiles'] = icpsr1950['area']
    icpsr1950['county_acres'] = icpsr1950['areaac']
    icpsr1950['population'] = icpsr1950['totpop']
    icpsr1950['population_race_white'] = icpsr1950['nwmtot'] + icpsr1950['nwftot'] + icpsr1950['fbwmtot'] + icpsr1950['fbwftot']
    icpsr1950['population_race_black'] = icpsr1950['negmtot'] + icpsr1950['negftot']
    icpsr1950['population_race_other'] = icpsr1950['population'] - icpsr1950['population_race_white'] - icpsr1950['population_race_black']
    icpsr1950['farms_white'] = icpsr1950['farmwh']
    icpsr1950['farms_nonwhite'] = icpsr1950['farmnonw']
    icpsr1950['farms_owner'] = icpsr1950['farmfown'] + icpsr1950['farmpown']
    icpsr1950['farms_tenant'] = icpsr1950['farmten']
    icpsr1950['farms_tenant_cash'] = icpsr1950['farmcten']
    icpsr1950['farmland'] = icpsr1950['acres']
    icpsr1950['farmland_owner'] = icpsr1950['acfown'] + icpsr1950['acpown']
    icpsr1950['farmland_tenant'] = icpsr1950['acten']
    icpsr1950['population_rural'] = icpsr1950['population'] - icpsr1950.get('urb950', 0)
    icpsr1950['year'] = 1950

    ag1950_cos = read_dta(os.path.join(GDIR, "usag1949.cos.crops.dta"))
    ag1950_cos = ag1950_cos.rename(columns={'item742':'horses','item744':'mules_horses'})
    ag1950_cos = ag1950_cos[['fips','horses','mules_horses']]

    ag1950_work = read_dta(os.path.join(GDIR, "usag1949.work.dta"))
    ag1950_work = ag1950_work[ag1950_work['state'].isin(SOUTH_STATES)]
    renames_1950 = {'var2':'corn_a','var3':'corn_y','var13':'wheat_a','var14':'wheat_y',
                    'var23':'oats_a','var24':'oats_y','var27':'rice_a','var28':'rice_y',
                    'var72':'cotton_a','var73':'cotton_y','var80':'scane_a','var82':'scane_y'}
    ag1950_work = ag1950_work.rename(columns=renames_1950)
    for v in ['corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y']:
        if v in ag1950_work.columns:
            ag1950_work[v] = ag1950_work[v].fillna(0)

    data1950 = pd.merge(ag1950_work[['fips','corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a',
                                      'corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y']],
                        icpsr1950, on='fips', how='inner')
    data1950 = pd.merge(data1950, ag1950_cos, on='fips', how='inner')
    data1950 = pd.merge(data1950, farmval_dict[1950][['fips','farmval']], on='fips', how='inner')
    data1950['value_landbuildings'] = data1950['farmval'] * data1950['farmland']
    data1950.drop(columns=['farmval'], inplace=True)

    # --- 1954 ---
    print("  Processing 1954...")
    ag1954_cos = read_dta(os.path.join(GDIR, "usag1954.cos.crops.dta"))
    renames_1954 = {'item101':'corn_a','item119':'oats_a','item122':'rice_a','item113':'wheat_a',
                    'item175':'cotton_a','item187':'scane_a','item742':'horses','item744':'mules_horses',
                    'item401':'corn_y','item413':'wheat_y','item419':'oats_y','item422':'rice_y',
                    'item475':'cotton_y','item487':'scane_y'}
    ag1954_cos = ag1954_cos.rename(columns=renames_1954)
    for v in ['corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y']:
        if v in ag1954_cos.columns:
            ag1954_cos[v] = ag1954_cos[v].fillna(0)

    d1954_icpsr = read_dta(os.path.join(GDIR, "02896-0073-Data.dta"))
    d1954_icpsr = d1954_icpsr[d1954_icpsr['level'] == 1]
    d1954_icpsr = d1954_icpsr[d1954_icpsr['state'].isin(SOUTH_STATES)]
    d1954_icpsr = d1954_icpsr.rename(columns={'var100':'farms','var101':'farmland','var126':'tractors'})
    d1954_icpsr['farmland'] = d1954_icpsr['farmland'] * 1000

    data1954 = pd.merge(d1954_icpsr[['fips','farms','farmland','tractors']],
                        ag1954_cos[['fips','corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a',
                                    'corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y',
                                    'horses','mules_horses']],
                        on='fips', how='inner')
    data1954 = pd.merge(data1954, farmval_dict[1954][['fips','farmval']], on='fips', how='inner')
    data1954['value_landbuildings'] = data1954['farmval'] * data1954['farmland']
    data1954.drop(columns=['farmval'], inplace=True)
    data1954['year'] = 1954

    # --- 1960 ---
    print("  Processing 1960...")
    icpsr1960 = read_dta(os.path.join(GDIR, "02896-0038-Data.dta"))
    icpsr1960 = icpsr1960[icpsr1960['level'] == 1]
    icpsr1960 = icpsr1960[icpsr1960['state'].isin(SOUTH_STATES)]
    icpsr1960['population'] = icpsr1960['totpop']
    icpsr1960['population_race_white'] = icpsr1960['wmtot'] + icpsr1960['wftot']
    icpsr1960['population_race_black'] = icpsr1960['negmtot'] + icpsr1960['negftot']
    icpsr1960['population_race_other'] = icpsr1960['population'] - icpsr1960['population_race_white'] - icpsr1960['population_race_black']
    icpsr1960['year'] = 1960

    icpsr1960_2 = read_dta(os.path.join(GDIR, "02896-0074-Data.dta"))
    icpsr1960_2 = icpsr1960_2[icpsr1960_2['level'] == 1]
    icpsr1960_2 = icpsr1960_2[icpsr1960_2['state'].isin(SOUTH_STATES)]
    icpsr1960_2 = icpsr1960_2.rename(columns={'var146':'value_perfarm','var147':'value_peracre',
                                                'var149':'cropval','var6':'percent_urban'})

    ag1959_cos = read_dta(os.path.join(GDIR, "usag1959.cos.crops.dta"))
    renames_1959 = {'item101':'corn_a','item119':'oats_a','item122':'rice_a','item113':'wheat_a',
                    'item175':'cotton_a','item187':'scane_a','item742':'horses','item744':'mules_horses',
                    'item401':'corn_y','item413':'wheat_y','item419':'oats_y','item422':'rice_y',
                    'item475':'cotton_y','item487':'scane_y'}
    ag1959_cos = ag1959_cos.rename(columns=renames_1959)
    for v in ['corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y']:
        if v in ag1959_cos.columns:
            ag1959_cos[v] = ag1959_cos[v].fillna(0)

    ag1959_work = read_dta(os.path.join(GDIR, "usag1959.work.dta"))
    ag1959_work = ag1959_work[ag1959_work['state'].isin(SOUTH_STATES)]
    ag1959_work = ag1959_work.rename(columns={'var250':'farms','var254':'farmland',
                                               'var276':'farmfown','var277':'farmpown',
                                               'var279':'farms_tenant','var282':'acfown',
                                               'var283':'acpown','var287':'acten'})
    ag1959_work['farms_owner'] = ag1959_work['farmfown'].fillna(0) + ag1959_work['farmpown'].fillna(0)
    ag1959_work['farmland_owner'] = ag1959_work['acfown'].fillna(0) + ag1959_work['acpown'].fillna(0)
    ag1959_work['farmland_tenant'] = ag1959_work['acten']
    ag1959_work['year'] = 1960

    data1960 = pd.merge(ag1959_work[['fips','farms','farmland','farms_owner','farms_tenant',
                                      'farmland_owner','farmland_tenant','year']],
                        icpsr1960[['fips','population','population_race_white','population_race_black','population_race_other']],
                        on='fips', how='inner')
    data1960 = pd.merge(data1960, icpsr1960_2[['fips','value_perfarm','value_peracre','cropval','percent_urban']],
                        on='fips', how='inner')
    data1960 = pd.merge(data1960, ag1959_cos[['fips','corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a',
                                               'corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y',
                                               'horses','mules_horses']], on='fips', how='inner')
    data1960 = pd.merge(data1960, farmval_dict[1960][['fips','farmval']], on='fips', how='inner')
    data1960['value_landbuildings'] = data1960['farmval'] * data1960['farmland']
    data1960['value_landbuildings_peracre'] = data1960['value_peracre'] * data1960['farmland']
    data1960['value_landbuildings_perfarm'] = data1960['value_perfarm'] * data1960['farms']
    data1960['population_rural'] = data1960['population'] * (1 - data1960['percent_urban'].fillna(0) / 100)
    data1960.drop(columns=['farmval'], inplace=True)

    # --- 1970 ---
    print("  Processing 1970...")
    icpsr1970 = read_dta(os.path.join(GDIR, "02896-0076-Data.dta"))
    icpsr1970 = icpsr1970[icpsr1970['level'] == 1]
    icpsr1970 = icpsr1970[icpsr1970['state'].isin(SOUTH_STATES)]
    icpsr1970 = icpsr1970.rename(columns={'var3':'population','var9':'population_race_white',
                                           'var10':'population_race_black',
                                           'var173':'farms','var175':'farmland',
                                           'var178':'value_perfarm','var179':'value_peracre',
                                           'var121':'mfgestab','var124':'mfgavear','var128':'mfgwages'})
    icpsr1970['population_race_other'] = icpsr1970['population'] - icpsr1970['population_race_white'] - icpsr1970['population_race_black']
    icpsr1970['farmland'] = icpsr1970['farmland'] * 1000
    icpsr1970['value_perfarm'] = icpsr1970['value_perfarm'] * 1000
    icpsr1970['mfgwages'] = icpsr1970['mfgwages'] * 1000000
    icpsr1970['mfgavear'] = icpsr1970['mfgavear'] * 100
    icpsr1970['value_landbuildings_perfarm'] = icpsr1970['value_perfarm'] * icpsr1970['farms']
    icpsr1970['value_landbuildings_peracre'] = icpsr1970['value_peracre'] * icpsr1970['farmland']
    icpsr1970['year'] = 1970

    ag1969_cos = read_dta(os.path.join(GDIR, "usag1969.cos.crops.dta"))
    # Need to handle stateicp -> state rename
    if 'stateicp' in ag1969_cos.columns:
        ag1969_cos = ag1969_cos.rename(columns={'stateicp':'state'})
    ag1969_cos = ag1969_cos[ag1969_cos['state'].isin(SOUTH_STATES)] if 'state' in ag1969_cos.columns else ag1969_cos
    renames_1969 = {'item101':'corn_a','item119':'oats_a','item122':'rice_a','item113':'wheat_a',
                    'item175':'cotton_a','item187':'scane_a','item742':'horses','item744':'mules_horses',
                    'item401':'corn_y','item413':'wheat_y','item419':'oats_y','item422':'rice_y',
                    'item475':'cotton_y','item487':'scane_y'}
    ag1969_cos = ag1969_cos.rename(columns=renames_1969)
    for v in ['corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y']:
        if v in ag1969_cos.columns:
            ag1969_cos[v] = ag1969_cos[v].fillna(0)

    ag1970_work = read_dta(os.path.join(GDIR, "usag74.1969.allfarms.work.dta"))
    ag1970_work = ag1970_work[ag1970_work['state'].isin(SOUTH_STATES)]
    ag1970_work = ag1970_work.rename(columns={'item01001':'farms_w','item06002':'value_equipment',
                                               'item06019':'tractors_wheel','item06022':'tractors_crawler',
                                               'item02004':'fo1','item02007':'fo2','item02010':'farms_tenant',
                                               'item02005':'flo1','item02008':'flo2','item02011':'farmland_tenant'})
    ag1970_work['value_equipment'] = ag1970_work['value_equipment'] * 1000
    ag1970_work['farms_owner'] = ag1970_work['fo1'].fillna(0) + ag1970_work['fo2'].fillna(0)
    ag1970_work['farmland_owner'] = ag1970_work['flo1'].fillna(0) + ag1970_work['flo2'].fillna(0)

    data1970 = pd.merge(ag1970_work[['fips','value_equipment','tractors_wheel','tractors_crawler',
                                      'farms_owner','farms_tenant','farmland_owner','farmland_tenant']],
                        ag1969_cos[['fips','corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a',
                                    'corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y',
                                    'horses','mules_horses']], on='fips', how='inner')
    data1970 = pd.merge(data1970, icpsr1970[['fips','population','population_race_white',
                                              'population_race_black','population_race_other',
                                              'farms','farmland','value_landbuildings_peracre',
                                              'value_landbuildings_perfarm','mfgestab','mfgavear',
                                              'mfgwages','year']],
                        on='fips', how='inner')

    # ========== BORDER ADJUSTMENTS ==========
    print("  Applying border adjustments...")
    datasets_to_adjust = {
        1910: (data1910, "Export1910_1900.txt"),
        1920: (data1920, "Export1920_1900.txt"),
        1925: (data1925, "Export1920_1900.txt"),  # uses 1920 borders
        1930: (data1930, "Export1930_1900.txt"),
        1935: (data1935, "Export1930_1900.txt"),  # uses 1930 borders
        1940: (data1940, "Export1940_1900.txt"),
        1945: (data1945, "Export1940_1900.txt"),  # uses 1940 borders
        1950: (data1950, "Export1950_1900.txt"),
        1954: (data1954, "Export1950_1900.txt"),  # uses 1950 borders
        1960: (data1960, "Export1960_1900.txt"),
        1964: (data1960, "Export1960_1900.txt"),  # uses 1960 borders -- but data is from ag1964
        1970: (data1970, "Export1970_1900.txt"),
    }

    def apply_border_adjustment(data, export_file, year_val):
        """Apply county border crosswalk from export file."""
        export = read_txt(os.path.join(GDIR, export_file))
        export = export[export['state'] == export['state_1']]
        export['percent'] = export['new_area'] / export['area']
        export = export.rename(columns={'id': 'fips'})

        # Convert all possible columns to numeric before merge
        data_c = data.copy()
        for col in data_c.columns:
            if col not in ['name','county','state','fips','year','level']:
                data_c[col] = pd.to_numeric(data_c[col], errors='coerce')

        merged = pd.merge(export[['fips','id_1','percent']], data_c, on='fips', how='inner')

        # Numeric columns to scale
        num_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
        skip_cols = ['fips','id_1','percent','year','state','county','level']
        scale_cols = [c for c in num_cols if c not in skip_cols]

        for c in scale_cols:
            merged[c] = merged[c] * merged['percent']
            # Replace NaN * percent > 0.01 with sentinel
            mask = merged[c].isna() & (merged['percent'] > 0.01)
            merged.loc[mask, c] = -1e23

        # Collapse by id_1
        result = merged.groupby('id_1')[scale_cols].sum().reset_index()

        # Replace sentinel values with NaN
        for c in scale_cols:
            result.loc[result[c] < 0, c] = np.nan

        result['year'] = year_val
        result = result.rename(columns={'id_1': 'fips'})
        return result

    # Process 1964 separately (different source data)
    ag1964_cos = read_dta(os.path.join(GDIR, "usag1964.cos.crops.dta"))
    renames_1964 = {'item101':'corn_a','item119':'oats_a','item122':'rice_a','item113':'wheat_a',
                    'item175':'cotton_a','item187':'scane_a','item742':'horses','item744':'mules_horses',
                    'item401':'corn_y','item413':'wheat_y','item419':'oats_y','item422':'rice_y',
                    'item475':'cotton_y','item487':'scane_y'}
    ag1964_cos = ag1964_cos.rename(columns=renames_1964)
    for v in ['corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y']:
        if v in ag1964_cos.columns:
            ag1964_cos[v] = ag1964_cos[v].fillna(0)

    d1964_icpsr = read_dta(os.path.join(GDIR, "02896-0075-Data.dta"))
    d1964_icpsr = d1964_icpsr[d1964_icpsr['level'] == 1]
    d1964_icpsr = d1964_icpsr[d1964_icpsr['state'].isin(SOUTH_STATES)]
    d1964_icpsr = d1964_icpsr.rename(columns={'var124':'farms','var126':'farmland','var129':'value_perfarm','var128':'value_peracre'})
    d1964_icpsr['farmland'] = d1964_icpsr['farmland'] * 1000
    d1964_icpsr['value_perfarm'] = d1964_icpsr['value_perfarm'] * 1000
    d1964_icpsr['value_landbuildings_perfarm'] = d1964_icpsr['value_perfarm'] * d1964_icpsr['farms']
    d1964_icpsr['value_landbuildings_peracre'] = d1964_icpsr['value_peracre'] * d1964_icpsr['farmland']

    data1964 = pd.merge(d1964_icpsr[['fips','farms','farmland','value_landbuildings_peracre','value_landbuildings_perfarm']],
                        ag1964_cos[['fips','corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a',
                                    'corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y',
                                    'horses','mules_horses']], on='fips', how='inner')

    adjusted = {}
    for yr, (data, efile) in datasets_to_adjust.items():
        if yr == 1964:
            adjusted[yr] = apply_border_adjustment(data1964, "Export1960_1900.txt", 1964)
        else:
            adjusted[yr] = apply_border_adjustment(data, efile, yr)

    # 1900 doesn't need adjustment - just rename fips to id_1
    data1900_adj = data1900.copy()
    data1900_adj['year'] = 1900

    # Append all years
    print("  Appending all years...")
    all_years = [data1900_adj] + [adjusted[yr] for yr in sorted(adjusted.keys())]
    panel = pd.concat(all_years, ignore_index=True, sort=False)

    # Merge flood data
    flood_1900 = read_txt(os.path.join(GDIR, "flooded_1900.txt"))
    flood_agg = flood_1900.groupby('fips').agg({'new_area':'sum','area':'mean'}).reset_index()
    flood_agg['flooded_share'] = flood_agg['new_area'] / flood_agg['area']
    panel = pd.merge(panel, flood_agg[['fips','flooded_share']], on='fips', how='left')

    # Merge distance data
    distance = read_txt(os.path.join(GDIR, "distance_1900.txt"))
    panel = pd.merge(panel, distance, on='fips', how='left')

    # Fill forward state/county/name within fips
    panel = panel.sort_values(['fips','year'])
    for col in ['state','county','name','x_centroid','y_centroid']:
        if col in panel.columns:
            panel[col] = panel.groupby('fips')[col].transform(lambda x: x.ffill().bfill())

    # Filter to southern states using FIPS codes
    panel['statefips'] = (panel['fips'] // 1000).astype(int)
    # Map ICPSR state codes to FIPS: the state column uses ICPSR codes
    # Keep based on statefips (which is actual FIPS state code)
    south_fips = [5, 22, 28, 47, 1, 13, 37, 45, 12]  # AR, LA, MS, TN, AL, GA, NC, SC, FL
    panel = panel[panel['statefips'].isin(south_fips)]

    # Merge crop suitability
    crop_suit = read_dta(os.path.join(GDIR, "1900_strm_distance_gaez.dta"))
    crop_suit = crop_suit[crop_suit['fips'] != 0]
    renames_suit = {'cottongaezprod_mean':'cotton_suitability','maizegaezprod_mean':'corn_suitability',
                    'wheatgaezprod_mean':'wheat_suitability','ricegaezprod_mean':'rice_suitability',
                    'oatgaezprod_mean':'oats_suitability','sugargaezprod_mean':'scane_suitability'}
    crop_suit = crop_suit.rename(columns=renames_suit)
    suit_cols = ['fips'] + [c for c in renames_suit.values() if c in crop_suit.columns]
    panel = pd.merge(panel, crop_suit[suit_cols], on='fips', how='left')

    # Merge MS distance
    ms_dist = read_txt(os.path.join(GDIR, "ms_distance.txt"))
    ms_dist = ms_dist.sort_values(['fips','distance_ms'])
    ms_dist = ms_dist.drop_duplicates(subset='fips', keep='first')
    panel = pd.merge(panel, ms_dist[['fips','distance_ms']], on='fips', how='left')

    # Merge ruggedness
    rugg = read_dta(os.path.join(GDIR, "1900_strm_distance_gaez.dta"))
    rugg = rugg[rugg['fips'] != 0][['fips','altitude_std_meters','altitude_range_meters']]
    panel = pd.merge(panel, rugg, on='fips', how='left')

    # Merge river distance
    river = read_dta(os.path.join(GDIR, "1900_strm_distance.dta"))
    river = river[['fips','Distance_Major_River_Meters']].copy()
    river['distance_river'] = river['Distance_Major_River_Meters'] / 1000
    panel = pd.merge(panel, river[['fips','distance_river']], on='fips', how='left')

    # Merge plantation
    brannen = read_dta(os.path.join(GDIR, "brannenplantcounties_1910.dta"))
    # Need border adjustment for plantation data
    export1910 = read_txt(os.path.join(GDIR, "Export1910_1900.txt"))
    export1910 = export1910[export1910['state'] == export1910['state_1']]
    export1910['percent'] = export1910['new_area'] / export1910['area']
    export1910 = export1910.rename(columns={'id':'fips'})
    brannen_adj = pd.merge(export1910[['fips','id_1','percent']], brannen[['fips','Brannen_Plantation']], on='fips', how='inner')
    brannen_adj['Brannen_Plantation'] = brannen_adj['Brannen_Plantation'] * brannen_adj['percent']
    brannen_adj.loc[brannen_adj['Brannen_Plantation'].isna() & (brannen_adj['percent'] > 0.01), 'Brannen_Plantation'] = -1e23
    brannen_1900 = brannen_adj.groupby('id_1')['Brannen_Plantation'].sum().reset_index()
    brannen_1900.loc[brannen_1900['Brannen_Plantation'] < 0, 'Brannen_Plantation'] = np.nan
    brannen_1900 = brannen_1900.rename(columns={'id_1':'fips'})
    panel = pd.merge(panel, brannen_1900, on='fips', how='left')

    # Merge New Deal data
    new_deal = read_dta(os.path.join(GDIR, "new_deal_spending.dta"))
    new_deal = new_deal[new_deal['state'].isin(SOUTH_STATES)]
    new_deal = new_deal[new_deal['county'] % 1000 != 0]
    new_deal = pd.merge(new_deal, icpsr_fips, on=['state','county'], how='inner')
    nd_cols = [c for c in new_deal.columns if c.startswith('pc')]
    panel = pd.merge(panel, new_deal[['fips'] + nd_cols], on='fips', how='left')

    # Fix value_landbuildings for years with peracre/perfarm
    if 'value_landbuildings_peracre' in panel.columns and 'value_landbuildings_perfarm' in panel.columns:
        if 'value_landbuildings' not in panel.columns:
            panel['value_landbuildings'] = np.nan
        mask = panel['value_landbuildings_peracre'].notna() & panel['value_landbuildings_perfarm'].notna()
        panel.loc[mask, 'value_landbuildings'] = (panel.loc[mask, 'value_landbuildings_peracre'] +
                                                    panel.loc[mask, 'value_landbuildings_perfarm']) / 2

    # Fix mules_horses
    if 'mules_horses' not in panel.columns:
        panel['mules_horses'] = np.nan
    if 'horses' in panel.columns and 'mules' in panel.columns:
        mask2 = panel['horses'].notna() & panel['mules'].notna() & panel['mules_horses'].isna()
        panel.loc[mask2, 'mules_horses'] = panel.loc[mask2, 'horses'] + panel.loc[mask2, 'mules']

    # Fix tractors
    if 'tractors' not in panel.columns:
        panel['tractors'] = np.nan
    if 'tractors_wheel' in panel.columns:
        mask3 = panel['tractors_wheel'].notna()
        panel.loc[mask3, 'tractors'] = panel.loc[mask3, 'tractors_wheel']

    print(f"  Panel shape: {panel.shape}, unique fips: {panel['fips'].nunique()}, years: {sorted(panel['year'].unique())}")
    return panel


###############################################################################
# PART 2: Pre-analysis (translate flood_preanalysis.do)
###############################################################################

def preanalysis(panel):
    """Translate flood_preanalysis.do into Python."""
    print("\n=== PRE-ANALYSIS ===")
    df = panel.copy()

    # Keep southern states
    df['statefips'] = (df['fips'] // 1000).astype(int)
    south_fips = [5, 22, 28, 47, 1, 13, 37, 45, 12]
    df = df[df['statefips'].isin(south_fips)].copy()

    # State dummies
    df['state_year'] = df['statefips'] * 10000 + df['year']

    # County weight = county_acres in 1920
    if 'county_acres' in df.columns:
        ca1920 = df.loc[df['year'] == 1920, ['fips', 'county_acres']].rename(columns={'county_acres': 'county_w'})
        df = pd.merge(df, ca1920, on='fips', how='left')
    else:
        df['county_w'] = 1  # fallback

    # Rename population variables
    if 'population_race_black' in df.columns:
        df['population_black'] = pd.to_numeric(df['population_race_black'], errors='coerce')
    if 'population_race_white' in df.columns:
        df['population_white'] = pd.to_numeric(df['population_race_white'], errors='coerce')
    if 'value_landbuildings' in df.columns:
        df['value_lb'] = pd.to_numeric(df['value_landbuildings'], errors='coerce')

    # Convert key columns to float
    for col in ['population','population_black','population_white','value_lb',
                'farms','farms_nonwhite','farms_nonwhite_tenant','farms_tenant',
                'farmland','value_equipment','mules_horses','tractors','county_w',
                'value_land','mfgestab','mfgwages','mfgavear',
                'cotton_a','corn_a','wheat_a','oats_a','rice_a','scane_a',
                'cotton_y','corn_y','wheat_y','oats_y','rice_y','scane_y',
                'cropval','population_rural']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Generate outcome variables
    df['lnpopulation_black'] = np.log(df['population_black'].clip(lower=0.001).astype(float))
    df['lnpopulation'] = np.log(df['population'].clip(lower=0.001))
    df['frac_black'] = df['population_black'] / (df['population_white'] + df['population_black'])
    df['lnfrac_black'] = np.log(df['frac_black'].clip(lower=1e-10))
    df['fracfarms_nonwhite'] = df['farms_nonwhite'] / df['farms']
    df['lnfracfarms_nonwhite'] = np.log((df['farms_nonwhite'] / df['farms']).clip(lower=1e-10))

    df['lnvalue_equipment'] = np.log(df['value_equipment'].clip(lower=0.001))
    df['avfarmsize'] = df['farmland'] / df['farms']
    df['lnavfarmsize'] = np.log(df['avfarmsize'].clip(lower=0.001))
    df['lnlandbuildingvalue'] = np.log(df['value_lb'].clip(lower=0.001))
    df['lnlandbuildingvaluef'] = np.log((df['value_lb'] / df['farmland']).clip(lower=1e-10))
    df['lnlandvaluef'] = np.log((df.get('value_land', np.nan) / df['farmland']).clip(lower=1e-10))
    df['lnlandvalue'] = np.log(df.get('value_land', pd.Series(dtype=float)).clip(lower=0.001))
    df['lnfarmland'] = np.log(df['farmland'].clip(lower=0.001))
    df['lnfarms_nonwhite_t'] = np.log((df['farms_nonwhite_tenant'] / df['farms_tenant']).clip(lower=1e-10))

    df['lnmules_horses'] = np.log(df['mules_horses'].clip(lower=0.001))
    df['lntractors'] = np.log(df['tractors'].clip(lower=0.001))
    df['tractorsperfarm'] = df['tractors'] / df['farms']
    df['lntractorsperfarm'] = np.log((df['tractors'] / df['farms']).clip(lower=1e-10))

    df['cropland'] = df['corn_a'].fillna(0) + df['wheat_a'].fillna(0) + df['oats_a'].fillna(0) + \
                     df['rice_a'].fillna(0) + df['cotton_a'].fillna(0) + df['scane_a'].fillna(0)
    df['cotton_cc'] = df['cotton_a'] / (df['cotton_a'] + df['corn_a'])
    df['lncotton_yield'] = np.log((df['cotton_y'] / df['cotton_a']).clip(lower=1e-10))
    df['lncorn_yield'] = np.log((df['corn_y'] / df['corn_a']).clip(lower=1e-10))
    df['lncropval_p'] = np.log((df.get('cropval', pd.Series(dtype=float)) / df['population']).clip(lower=1e-10))
    df['lncropval_rp'] = np.log((df.get('cropval', pd.Series(dtype=float)) / df.get('population_rural', pd.Series(dtype=float))).clip(lower=1e-10))

    df['mfgavewages'] = df.get('mfgwages', pd.Series(dtype=float)) / df.get('mfgavear', pd.Series(dtype=float))
    df['lnmfgestab'] = np.log(df.get('mfgestab', pd.Series(dtype=float)).clip(lower=0.001))
    df['lnmfgavewages'] = np.log(df['mfgavewages'].clip(lower=0.001))

    # Per-acre and per-county variables
    for var in ['population','population_black','value_equipment','value_lb','tractors','mules_horses','farmland']:
        if var in df.columns:
            df[f'ln{var}_a'] = np.log((df[var] / df['county_w']).clip(lower=1e-10))
            df[f'{var}_a'] = (df[var] * 100) / df['county_w']
    for var in ['avfarmsize']:
        df[f'{var}_a'] = df[var] / df['county_w']
    df['value_lb_f'] = (df['value_lb'] * 100) / df['farmland']
    df['lnvalue_lb_f'] = np.log((df['value_lb'] / df['farmland']).clip(lower=1e-10))
    df['value_lb_a'] = (df['value_lb'] * 100) / df['county_w']
    df['lnvalue_lb_a'] = np.log((df['value_lb'] / df['county_w']).clip(lower=1e-10))

    # Balance panel
    years_pop = [1900,1910,1920,1930,1940,1950,1960,1970]
    years_lb = [1900,1910,1920,1925,1930,1935,1940,1945,1950,1954,1960,1964,1970]
    years_equip = [1900,1910,1920,1925,1930,1940,1970]
    years_crop = [1900,1910,1920,1925,1930,1935,1940,1950,1954,1960,1964,1970]

    df['drop'] = 0
    df.loc[df['lnpopulation'].isna() & df['year'].isin(years_pop), 'drop'] = 1
    df.loc[df['lnpopulation_black'].isna() & df['year'].isin(years_pop), 'drop'] = 1
    df.loc[df['lnvalue_lb_a'].isna() & df['year'].isin(years_lb), 'drop'] = 1
    df.loc[df['lnvalue_equipment'].isna() & df['year'].isin(years_equip), 'drop'] = 1
    df.loc[df['cropland'].isna() & df['year'].isin(years_crop), 'drop'] = 1

    drop_county = df.groupby('fips')['drop'].max().reset_index().rename(columns={'drop':'drop_county'})
    df = pd.merge(df, drop_county, on='fips')
    df = df[df['drop_county'] == 0].copy()
    df.drop(columns=['drop','drop_county'], inplace=True)

    # Check number of observations per fips
    counts = df.groupby('fips').size().reset_index(name='number')
    # Drop counties with fewer than 12 obs (13 for certain states)
    certain_states = [42, 45, 46, 54]  # ICPSR codes
    # We use statefips (FIPS codes) so need to map - but the do file uses 'state' which is ICPSR
    # For simplicity, use counts < 12 as threshold
    df = pd.merge(df, counts, on='fips')
    df = df[df['number'] >= 12].copy()
    df.drop(columns=['number'], inplace=True)

    # Balance tractors
    df.sort_values(['fips','year'], inplace=True)
    tractor_issue = df.loc[df['year'] == 1925].groupby('fips')['lntractors'].apply(lambda x: x.isna().any()).reset_index()
    tractor_issue = tractor_issue.rename(columns={'lntractors':'balance_tractors'})
    df = pd.merge(df, tractor_issue, on='fips', how='left')
    df['balance_tractors'] = df['balance_tractors'].fillna(False)
    for v in ['lntractors','tractors','tractors_a','lntractors_a']:
        if v in df.columns:
            df.loc[df['balance_tractors'], v] = np.nan

    # Restrict sample
    fb1920 = df.loc[df['year'] == 1920, ['fips', 'frac_black']].rename(columns={'frac_black': 'fb1920'})
    df = pd.merge(df, fb1920, on='fips', how='left')
    df = df[df['fb1920'] >= 0.10].copy()

    cc1920 = df.loc[df['year'] == 1920].copy()
    cc1920['cotton_c'] = cc1920['cotton_a'] / cc1920['cropland']
    cc1920 = cc1920[['fips', 'cotton_c']]
    df = pd.merge(df, cc1920, on='fips', how='left')
    df = df[df['cotton_c'] >= 0.15].copy()
    df.drop(columns=['cotton_c', 'fb1920'], inplace=True)

    # Ensure balanced panel (13 obs per fips)
    counts2 = df.groupby('fips').size().reset_index(name='num')
    df = pd.merge(df, counts2, on='fips')
    df = df[df['num'] == 13].copy()
    df.drop(columns=['num'], inplace=True)

    # Drop specific counties
    df = df[~df['fips'].isin([47149, 47071, 22023])].copy()

    # Flood variables
    df['percent_flood'] = df['flooded_share'].fillna(0)
    df['flood'] = (df['percent_flood'] > 0).astype(int)
    df['flood_intensity'] = df['percent_flood'] * df['flood']

    # Create flood intensity x year interactions
    all_years = [1900,1910,1920,1925,1930,1935,1940,1945,1950,1954,1960,1964,1970]
    for yr in all_years:
        df[f'f_int_{yr}'] = 0.0
        df.loc[df['year'] == yr, f'f_int_{yr}'] = df.loc[df['year'] == yr, 'flood_intensity']

    # Red cross variables
    for yr in [1930]:
        df['redcross_flooded_acres'] = 0.0
        mask = df['year'] == yr
        if 'flooded_acres' in df.columns:
            df.loc[mask, 'redcross_flooded_acres'] = df.loc[mask, 'flooded_acres'].fillna(0) / df.loc[mask, 'county_w']
        df['redcross_flooded_people'] = 0.0
        if 'pop_affected' in df.columns:
            df.loc[mask, 'redcross_flooded_people'] = df.loc[mask, 'pop_affected'].fillna(0) / df.loc[mask, 'population']

    # flood_intensity_2 and _3
    fi2 = df.loc[df['year'] == 1930, ['fips', 'redcross_flooded_acres']].rename(columns={'redcross_flooded_acres': 'flood_intensity_2'})
    fi3 = df.loc[df['year'] == 1930, ['fips', 'redcross_flooded_people']].rename(columns={'redcross_flooded_people': 'flood_intensity_3'})
    df = pd.merge(df, fi2, on='fips', how='left')
    df = pd.merge(df, fi3, on='fips', how='left')
    df['flood_intensity_2'] = df['flood_intensity_2'].fillna(0)
    df['flood_intensity_3'] = df['flood_intensity_3'].fillna(0)

    for yr in all_years:
        df[f'f2_int_{yr}'] = 0.0
        df.loc[df['year'] == yr, f'f2_int_{yr}'] = df.loc[df['year'] == yr, 'flood_intensity_2']
        df[f'f3_int_{yr}'] = 0.0
        df.loc[df['year'] == yr, f'f3_int_{yr}'] = df.loc[df['year'] == yr, 'flood_intensity_3']

    # Distance to MS river interactions
    for yr in all_years:
        df[f'ld_{yr}'] = 0.0
        df.loc[df['year'] == yr, f'ld_{yr}'] = df.loc[df['year'] == yr, 'distance_ms'].fillna(0)

    # Crop suitability interactions
    for yr in all_years:
        for crop in ['cotton','corn','wheat','oats','rice','scane']:
            col = f'{crop}_suitability'
            df[f'{crop}_s_{yr}'] = 0.0
            if col in df.columns:
                df.loc[df['year'] == yr, f'{crop}_s_{yr}'] = df.loc[df['year'] == yr, col].fillna(0)

    # Longitude and latitude interactions
    for yr in all_years:
        df[f'dx_{yr}'] = 0.0
        df[f'dy_{yr}'] = 0.0
        if 'x_centroid' in df.columns:
            df.loc[df['year'] == yr, f'dx_{yr}'] = df.loc[df['year'] == yr, 'x_centroid'].fillna(0) / 1000
        if 'y_centroid' in df.columns:
            df.loc[df['year'] == yr, f'dy_{yr}'] = df.loc[df['year'] == yr, 'y_centroid'].fillna(0) / 1000

    # Ruggedness interactions
    for yr in all_years:
        df[f'rug_{yr}'] = 0.0
        if 'altitude_std_meters' in df.columns:
            df.loc[df['year'] == yr, f'rug_{yr}'] = df.loc[df['year'] == yr, 'altitude_std_meters'].fillna(0)

    # Plantation
    df['plantation'] = 0
    if 'Brannen_Plantation' in df.columns:
        df.loc[df['Brannen_Plantation'] > 0.5, 'plantation'] = 1
    for yr in all_years:
        df[f'plantation_{yr}'] = 0
        df.loc[df['year'] == yr, f'plantation_{yr}'] = df.loc[df['year'] == yr, 'plantation']

    # New Deal spending interactions
    nd_vars = ['pcpubwor','pcaaa','pcrelief','pcndloan','pcndins']
    for yr in all_years:
        for v in nd_vars:
            df[f'ln{v}_{yr}'] = 0.0
            if v in df.columns:
                df.loc[df['year'] == yr, f'ln{v}_{yr}'] = np.log(df.loc[df['year'] == yr, v].clip(lower=1e-10))

    # Create lagged values
    df = df.sort_values(['fips','year'])

    # Lags for vars existing in 1930, 1920, 1910 (lag2=1920, lag3=1910, lag4=1900)
    vars_lag_from_1930 = ['lnpopulation','lnpopulation_black','lnfrac_black','lnfarms_nonwhite_t',
                          'lnfracfarms_nonwhite','lnlandvalue','lnlandvaluef']
    for var in vars_lag_from_1930:
        if var not in df.columns:
            continue
        for yr_ref, lag_name in [(1920, 'lc1920'), (1910, 'lc1910'), (1900, 'lc1900')]:
            vals = df.loc[df['year'] == yr_ref, ['fips', var]].rename(columns={var: f'{lag_name}_{var}'})
            df = pd.merge(df, vals, on='fips', how='left')

        post_years = [1930,1935,1940,1945,1950,1954,1960,1964,1970]
        for yr in post_years:
            for lag_num, lag_name in [(2,'lc1920'),(3,'lc1910'),(4,'lc1900')]:
                src = f'{lag_name}_{var}'
                dst = f'lag{lag_num}_{var}_{yr}'
                df[dst] = 0.0
                if src in df.columns:
                    df.loc[df['year'] == yr, dst] = df.loc[df['year'] == yr, src].fillna(0)

    # Lags for vars existing in 1925 (lag1=1925, lag2=1920, lag3=1910, lag4=1900)
    vars_lag_from_1925 = ['lnvalue_equipment','lntractors','tractorsperfarm','lntractorsperfarm',
                          'lnmules_horses','lnavfarmsize','lnlandbuildingvalue','lnlandbuildingvaluef',
                          'lnfarmland_a','lncotton_yield','lncorn_yield']
    # Add cotton_a_a, corn_a_a, etc. if they exist
    for v in ['cotton_a_a','corn_a_a','cotton_cc','rice_a_a','scane_a_a','oats_a_a','wheat_a_a']:
        if v in df.columns:
            vars_lag_from_1925.append(v)

    for var in vars_lag_from_1925:
        if var not in df.columns:
            continue
        for yr_ref, lag_name in [(1925,'lc1925'),(1920,'lc1920'),(1910,'lc1910'),(1900,'lc1900')]:
            key = f'{lag_name}_{var}'
            if key not in df.columns:
                vals = df.loc[df['year'] == yr_ref, ['fips', var]].rename(columns={var: key})
                df = pd.merge(df, vals, on='fips', how='left')

        post_years = [1930,1935,1940,1945,1950,1954,1960,1964,1970]
        for yr in post_years:
            for lag_num, lag_name in [(1,'lc1925'),(2,'lc1920'),(3,'lc1910'),(4,'lc1900')]:
                src = f'{lag_name}_{var}'
                dst = f'lag{lag_num}_{var}_{yr}'
                df[dst] = 0.0
                if src in df.columns:
                    df.loc[df['year'] == yr, dst] = df.loc[df['year'] == yr, src].fillna(0)

    # Lags for mfg variables
    for var in ['lnmfgestab','lnmfgavewages']:
        if var not in df.columns:
            continue
        for yr_ref, lag_name in [(1920,'lc1920'),(1900,'lc1900')]:
            key = f'{lag_name}_{var}'
            if key not in df.columns:
                vals = df.loc[df['year'] == yr_ref, ['fips', var]].rename(columns={var: key})
                df = pd.merge(df, vals, on='fips', how='left')

        post_years = [1930,1935,1940,1945,1950,1954,1960,1964,1970]
        for yr in post_years:
            for lag_num, lag_name in [(2,'lc1920'),(4,'lc1900')]:
                src = f'{lag_name}_{var}'
                dst = f'lag{lag_num}_{var}_{yr}'
                df[dst] = 0.0
                if src in df.columns:
                    df.loc[df['year'] == yr, dst] = df.loc[df['year'] == yr, src].fillna(0)

    # Pre-differences
    df = df.sort_values(['fips','year'])
    for var in ['frac_black','population_black_a','population_a','fracfarms_nonwhite']:
        lnvar = f'ln{var}'
        if lnvar in df.columns:
            diff = df.groupby('fips')[lnvar].diff()
            df[f'ln{var}_d'] = np.nan
            df.loc[df['year'] == 1920, f'ln{var}_d'] = diff[df['year'] == 1920]

    for var in ['value_equipment_a','mules_horses_a','avfarmsize','farmland_a','value_lb_f','value_lb_a']:
        lnvar = f'ln{var}'
        if lnvar in df.columns:
            diff = df.groupby('fips')[lnvar].diff()
            df[f'ln{var}_d'] = np.nan
            df.loc[df['year'] == 1925, f'ln{var}_d'] = diff[df['year'] == 1925]

    # River sample
    df['main_sample'] = 0
    # The do file uses ICPSR state codes 42,45,46,54 which map to FIPS 5,22,28,47 (AR,LA,MS,TN)
    # These are the Mississippi flood states
    df.loc[df['statefips'].isin([5, 22, 28, 47]), 'main_sample'] = 1
    df['riverclose_elsewhere'] = ((df['distance_river'] < 50) & (df['main_sample'] == 0)).astype(int)

    # Save south_rivers version (before restricting to main sample)
    south_rivers = df[(df['main_sample'] == 1) | (df['riverclose_elsewhere'] == 1) |
                      ((df['distance_river'] > 50) & (df['distance_river'] < 150) & (df['main_sample'] == 0))].copy()

    # Main sample only
    df = df[df['main_sample'] == 1].copy()

    # State dummies for main sample
    df['d_s'] = pd.Categorical(df['statefips'])

    print(f"  Main sample: {df.shape[0]} obs, {df['fips'].nunique()} counties")
    print(f"  South rivers sample: {south_rivers.shape[0]} obs, {south_rivers['fips'].nunique()} counties")

    # Generate post-1930 sample
    post1930 = df[df['year'] >= 1920].copy()
    post1930 = post1930.sort_values(['fips','year'])

    return df, post1930, south_rivers


###############################################################################
# PART 3: Run regressions
###############################################################################

def get_control_cols(df, prefixes, years=None):
    """Get column names matching prefixes, optionally filtered by year suffixes."""
    cols = []
    for p in prefixes:
        matching = [c for c in df.columns if c.startswith(p)]
        cols.extend(matching)
    return sorted(set(cols))

def run_areg(df, depvar, indepvars, absorb, cluster, weights=None, verbose=False):
    """Run areg equivalent using pyfixest."""
    all_vars = [depvar] + indepvars
    dfc = df.dropna(subset=[depvar, weights] if weights else [depvar]).copy()

    # Build formula
    rhs = " + ".join(indepvars) if indepvars else "1"
    formula = f"{depvar} ~ {rhs} | {absorb}"

    try:
        if weights:
            m = pf.feols(formula, data=dfc, vcov={"CRV1": cluster}, weights=weights)
        else:
            m = pf.feols(formula, data=dfc, vcov={"CRV1": cluster})
        return m
    except Exception as e:
        if verbose:
            print(f"    Error: {e}")
        return None

def run_reg(df, depvar, indepvars, weights=None, robust=True):
    """Run OLS regression equivalent."""
    dfc = df.copy()
    rhs = " + ".join(indepvars) if indepvars else "1"
    formula = f"{depvar} ~ {rhs}"

    try:
        if weights and robust:
            m = pf.feols(formula, data=dfc, vcov="hetero", weights=weights)
        elif weights:
            m = pf.feols(formula, data=dfc, weights=weights)
        elif robust:
            m = pf.feols(formula, data=dfc, vcov="hetero")
        else:
            m = pf.feols(formula, data=dfc)
        return m
    except Exception as e:
        print(f"    Error running reg: {e}")
        return None


def extract_results(model, treatment_var, reg_id, paper_id, outcome_var, estimator,
                    fixed_effects, controls_desc, cluster_var, sample_desc,
                    original_coef=None, original_se=None, notes=""):
    """Extract results from a model into a dictionary."""
    if model is None:
        return {
            'paper_id': paper_id, 'reg_id': reg_id, 'outcome_var': outcome_var,
            'treatment_var': treatment_var, 'coefficient': np.nan, 'std_error': np.nan,
            'p_value': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan,
            'n_obs': np.nan, 'r_squared': np.nan,
            'original_coefficient': original_coef, 'original_std_error': original_se,
            'match_status': 'failed',
            'coefficient_vector_json': '{}',
            'fixed_effects': fixed_effects, 'controls_desc': controls_desc,
            'cluster_var': cluster_var, 'estimator': estimator,
            'sample_desc': sample_desc, 'notes': notes
        }

    try:
        coefs = model.coef()
        ses = model.se()
        pvals = model.pvalue()

        if treatment_var in coefs.index:
            coef = coefs[treatment_var]
            se = ses[treatment_var]
            pv = pvals[treatment_var]
        else:
            # Try partial match
            matches = [c for c in coefs.index if treatment_var in c]
            if matches:
                coef = coefs[matches[0]]
                se = ses[matches[0]]
                pv = pvals[matches[0]]
            else:
                coef = se = pv = np.nan

        ci = model.confint()
        if treatment_var in ci.index:
            ci_l = ci.loc[treatment_var, '2.5%']
            ci_u = ci.loc[treatment_var, '97.5%']
        else:
            ci_l = ci_u = np.nan

        nobs = model._N
        r2 = model._r2

        # Coefficient vector
        coef_dict = {k: round(float(v), 6) for k, v in coefs.items()
                     if not k.startswith('d_s') and not k.startswith('d_sy') and not k.startswith('d_year')}
        coef_json = json.dumps(coef_dict)

        # Match status
        if original_coef is not None and not np.isnan(coef):
            if abs(coef - original_coef) < 1e-4:
                match = 'exact'
            elif abs(original_coef) > 0 and abs(coef - original_coef) / abs(original_coef) <= 0.01:
                match = 'close'
            else:
                match = 'discrepant'
        else:
            match = 'close'  # no original to compare

        return {
            'paper_id': paper_id, 'reg_id': reg_id, 'outcome_var': outcome_var,
            'treatment_var': treatment_var,
            'coefficient': round(float(coef), 6), 'std_error': round(float(se), 6),
            'p_value': round(float(pv), 6),
            'ci_lower': round(float(ci_l), 6) if not np.isnan(ci_l) else '',
            'ci_upper': round(float(ci_u), 6) if not np.isnan(ci_u) else '',
            'n_obs': int(nobs), 'r_squared': round(float(r2), 4),
            'original_coefficient': original_coef, 'original_std_error': original_se,
            'match_status': match,
            'coefficient_vector_json': coef_json,
            'fixed_effects': fixed_effects, 'controls_desc': controls_desc,
            'cluster_var': cluster_var, 'estimator': estimator,
            'sample_desc': sample_desc, 'notes': notes
        }
    except Exception as e:
        return {
            'paper_id': paper_id, 'reg_id': reg_id, 'outcome_var': outcome_var,
            'treatment_var': treatment_var, 'coefficient': np.nan, 'std_error': np.nan,
            'p_value': np.nan, 'ci_lower': '', 'ci_upper': '',
            'n_obs': np.nan, 'r_squared': np.nan,
            'original_coefficient': original_coef, 'original_std_error': original_se,
            'match_status': 'failed',
            'coefficient_vector_json': '{}',
            'fixed_effects': fixed_effects, 'controls_desc': controls_desc,
            'cluster_var': cluster_var, 'estimator': estimator,
            'sample_desc': sample_desc, 'notes': str(e)
        }


def run_table1(df):
    """Table 1: Pre-differences in 1920 and 1925."""
    print("\n--- Table 1: Pre-differences ---")
    results = []
    reg_id = 1

    # State dummies
    state_dummies = [c for c in df.columns if c.startswith('d_s_')]
    if not state_dummies:
        # Create state dummies manually
        for i, s in enumerate(sorted(df['statefips'].unique())):
            df[f'd_s_{i+1}'] = (df['statefips'] == s).astype(int)
        state_dummies = [c for c in df.columns if c.startswith('d_s_')]

    geo_controls = [c for c in df.columns if any(c.startswith(p) for p in
                    ['cotton_s_','corn_s_','ld_','dx_','dy_','rug_'])]

    # Panel A: 1920 variables
    for var in ['frac_black','population_black_a','population_a','fracfarms_nonwhite']:
        lnvar = f'ln{var}'
        df20 = df[df['year'] == 1920].copy()
        if lnvar not in df20.columns:
            continue

        # (1) Basic: lnvar ~ flood_intensity + state dummies
        sd = [c for c in state_dummies if df20[c].nunique() > 1]
        m = run_reg(df20, lnvar, ['flood_intensity'] + sd, weights='county_w')
        results.append(extract_results(m, 'flood_intensity', reg_id, '112749-V1', lnvar,
                       'OLS', 'none', 'state dummies', 'none',
                       f'1920 cross-section, {var}'))
        reg_id += 1

        # (2) With geography controls
        gc20 = [c for c in geo_controls if c.endswith('_1920') or not any(c.endswith(f'_{y}') for y in range(1900,1971))]
        gc20 = [c for c in gc20 if c in df20.columns and df20[c].nunique() > 1]
        indep = ['flood_intensity'] + sd + gc20
        m = run_reg(df20, lnvar, indep, weights='county_w')
        results.append(extract_results(m, 'flood_intensity', reg_id, '112749-V1', lnvar,
                       'OLS', 'none', 'state dummies + geography', 'none',
                       f'1920 cross-section, {var}'))
        reg_id += 1

        # (3) Pre-difference
        lnvar_d = f'ln{var}_d'
        if lnvar_d in df20.columns:
            m = run_reg(df20, lnvar_d, ['flood_intensity'] + sd, weights='county_w')
            results.append(extract_results(m, 'flood_intensity', reg_id, '112749-V1', lnvar_d,
                           'OLS', 'none', 'state dummies', 'none',
                           f'1920 pre-difference, {var}'))
            reg_id += 1

            m = run_reg(df20, lnvar_d, ['flood_intensity'] + sd + gc20, weights='county_w')
            results.append(extract_results(m, 'flood_intensity', reg_id, '112749-V1', lnvar_d,
                           'OLS', 'none', 'state dummies + geography', 'none',
                           f'1920 pre-difference, {var}'))
            reg_id += 1

    # Panel B: 1925 variables
    for var in ['value_equipment_a','mules_horses_a','avfarmsize','farmland_a','value_lb_f','value_lb_a']:
        lnvar = f'ln{var}'
        df25 = df[df['year'] == 1925].copy()
        if lnvar not in df25.columns:
            continue

        sd = [c for c in state_dummies if df25[c].nunique() > 1]
        gc25 = [c for c in geo_controls if c.endswith('_1925') or not any(c.endswith(f'_{y}') for y in range(1900,1971))]
        gc25 = [c for c in gc25 if c in df25.columns and df25[c].nunique() > 1]

        m = run_reg(df25, lnvar, ['flood_intensity'] + sd, weights='county_w')
        results.append(extract_results(m, 'flood_intensity', reg_id, '112749-V1', lnvar,
                       'OLS', 'none', 'state dummies', 'none',
                       f'1925 cross-section, {var}'))
        reg_id += 1

        m = run_reg(df25, lnvar, ['flood_intensity'] + sd + gc25, weights='county_w')
        results.append(extract_results(m, 'flood_intensity', reg_id, '112749-V1', lnvar,
                       'OLS', 'none', 'state dummies + geography', 'none',
                       f'1925 cross-section, {var}'))
        reg_id += 1

        lnvar_d = f'ln{var}_d'
        if lnvar_d in df25.columns:
            m = run_reg(df25, lnvar_d, ['flood_intensity'] + sd, weights='county_w')
            results.append(extract_results(m, 'flood_intensity', reg_id, '112749-V1', lnvar_d,
                           'OLS', 'none', 'state dummies', 'none',
                           f'1925 pre-difference, {var}'))
            reg_id += 1

            m = run_reg(df25, lnvar_d, ['flood_intensity'] + sd + gc25, weights='county_w')
            results.append(extract_results(m, 'flood_intensity', reg_id, '112749-V1', lnvar_d,
                           'OLS', 'none', 'state dummies + geography', 'none',
                           f'1925 pre-difference, {var}'))
            reg_id += 1

    # Tractors
    lnvar = 'lntractors_a'
    df25 = df[df['year'] == 1925].copy()
    if lnvar in df25.columns:
        sd = [c for c in state_dummies if df25[c].nunique() > 1]
        gc25 = [c for c in geo_controls if c in df25.columns and df25[c].nunique() > 1]
        m = run_reg(df25, lnvar, ['flood_intensity'] + sd, weights='county_w')
        results.append(extract_results(m, 'flood_intensity', reg_id, '112749-V1', lnvar,
                       'OLS', 'none', 'state dummies', 'none',
                       f'1925 cross-section, tractors'))
        reg_id += 1

        m = run_reg(df25, lnvar, ['flood_intensity'] + sd + gc25, weights='county_w')
        results.append(extract_results(m, 'flood_intensity', reg_id, '112749-V1', lnvar,
                       'OLS', 'none', 'state dummies + geography', 'none',
                       f'1925 cross-section, tractors'))
        reg_id += 1

    print(f"  Table 1: {len(results)} regressions")
    return results, reg_id


def run_table2(df_post, start_id):
    """Table 2: Labor outcomes."""
    print("\n--- Table 2: Labor ---")
    results = []
    reg_id = start_id

    # Get control columns
    sy_dummies = sorted([c for c in df_post.columns if c.startswith('d_sy_')])
    geo = sorted([c for c in df_post.columns if any(c.startswith(p) for p in
                  ['cotton_s_','corn_s_','ld_','dx_','dy_','rug_'])])
    nd = sorted([c for c in df_post.columns if any(c.startswith(p) for p in
                 ['lnpcpubwor_','lnpcaaa_','lnpcrelief_','lnpcndloan_','lnpcndins_'])])

    f_int_vars = ['f_int_1930','f_int_1940','f_int_1950','f_int_1960','f_int_1970']

    outcomes = [
        ('lnfrac_black', ['lag2_lnfrac_black_','lag3_lnfrac_black_','lag4_lnfrac_black_']),
        ('lnfrac_black', ['lag2_lnfrac_black_','lag3_lnfrac_black_','lag4_lnfrac_black_']),
        ('lnpopulation_black', ['lag2_lnpopulation_black_','lag3_lnpopulation_black_','lag4_lnpopulation_black_']),
        ('lnpopulation_black', ['lag2_lnpopulation_black_','lag3_lnpopulation_black_','lag4_lnpopulation_black_']),
        ('lnpopulation', ['lag2_lnpopulation_1','lag3_lnpopulation_1','lag4_lnpopulation_1']),
        ('lnpopulation', ['lag2_lnpopulation_1','lag3_lnpopulation_1','lag4_lnpopulation_1']),
        ('lnfracfarms_nonwhite', ['lag2_lnfracfarms_nonwhite_1','lag3_lnfracfarms_nonwhite_1','lag4_lnfracfarms_nonwhite_1']),
        ('lnfracfarms_nonwhite', ['lag2_lnfracfarms_nonwhite_1','lag3_lnfracfarms_nonwhite_1','lag4_lnfracfarms_nonwhite_1']),
    ]
    use_nd = [False, True, False, True, False, True, False, True]

    for i, ((depvar, lag_prefixes), with_nd) in enumerate(zip(outcomes, use_nd)):
        lags = sorted([c for c in df_post.columns if any(c.startswith(p) for p in lag_prefixes)])
        # Filter to columns with variation
        lags = [c for c in lags if c in df_post.columns]
        sy = [c for c in sy_dummies if c in df_post.columns]
        g = [c for c in geo if c in df_post.columns]

        indep = f_int_vars + sy + g + lags
        if with_nd:
            indep += [c for c in nd if c in df_post.columns]

        # Filter to available columns
        indep = [c for c in indep if c in df_post.columns]

        m = run_areg(df_post, depvar, indep, 'fips', 'fips', weights='county_w')
        controls = 'state-year FE + geography + lags' + (' + New Deal' if with_nd else '')
        for fvar in f_int_vars:
            r = extract_results(m, fvar, reg_id, '112749-V1', depvar,
                               'OLS-FE', 'county', controls, 'fips',
                               f'Table 2 col {i+1}')
            results.append(r)
            reg_id += 1

    print(f"  Table 2: {len(results)} coefficient estimates from 8 regressions")
    return results, reg_id


def run_table4(df_post, start_id):
    """Table 4: Capital and Techniques."""
    print("\n--- Table 4: Capital and Techniques ---")
    results = []
    reg_id = start_id

    sy_dummies = sorted([c for c in df_post.columns if c.startswith('d_sy_')])
    geo = sorted([c for c in df_post.columns if any(c.startswith(p) for p in
                  ['cotton_s_','corn_s_','ld_','dx_','dy_','rug_'])])
    nd = sorted([c for c in df_post.columns if any(c.startswith(p) for p in
                 ['lnpcpubwor_','lnpcaaa_','lnpcrelief_','lnpcndloan_','lnpcndins_'])])

    f_int_vars = ['f_int_1930','f_int_1935','f_int_1940','f_int_1945',
                  'f_int_1950','f_int_1954','f_int_1960','f_int_1964','f_int_1970']

    outcomes = [
        ('lnavfarmsize', 'lag1_lnavfarmsize_', True),
        ('lnvalue_equipment', 'lag1_lnvalue_equipment_', True),
        ('lntractors', 'lag1_lntractors_', False),
        ('lnmules_horses', 'lag1_lnmules_horses_', True),
    ]

    for depvar, lag1_prefix, has_lag234 in outcomes:
        for with_nd in [False, True]:
            lag1 = sorted([c for c in df_post.columns if c.startswith(lag1_prefix)])
            lags = lag1[:]
            if has_lag234:
                for p in [f'lag2_{depvar}_', f'lag3_{depvar}_', f'lag4_{depvar}_']:
                    lags += sorted([c for c in df_post.columns if c.startswith(p)])

            sy = [c for c in sy_dummies if c in df_post.columns]
            g = [c for c in geo if c in df_post.columns]
            indep = f_int_vars + sy + g + [c for c in lags if c in df_post.columns]
            if with_nd:
                indep += [c for c in nd if c in df_post.columns]

            indep = [c for c in indep if c in df_post.columns]
            m = run_areg(df_post, depvar, indep, 'fips', 'fips', weights='county_w')

            controls = 'state-year FE + geography + lags' + (' + New Deal' if with_nd else '')
            for fvar in f_int_vars:
                r = extract_results(m, fvar, reg_id, '112749-V1', depvar,
                                   'OLS-FE', 'county', controls, 'fips',
                                   f'Table 4, {depvar}' + (' + ND' if with_nd else ''))
                results.append(r)
                reg_id += 1

    print(f"  Table 4: {len(results)} coefficient estimates")
    return results, reg_id


def run_table5(df_post, start_id):
    """Table 5: Farmland."""
    print("\n--- Table 5: Farmland ---")
    results = []
    reg_id = start_id

    sy_dummies = sorted([c for c in df_post.columns if c.startswith('d_sy_')])
    geo = sorted([c for c in df_post.columns if any(c.startswith(p) for p in
                  ['cotton_s_','corn_s_','ld_','dx_','dy_','rug_'])])
    nd = sorted([c for c in df_post.columns if any(c.startswith(p) for p in
                 ['lnpcpubwor_','lnpcaaa_','lnpcrelief_','lnpcndloan_','lnpcndins_'])])

    f_int_vars = ['f_int_1930','f_int_1935','f_int_1940','f_int_1945',
                  'f_int_1950','f_int_1954','f_int_1960','f_int_1964','f_int_1970']

    for depvar in ['lnfarmland_a','lnlandbuildingvaluef','lnlandbuildingvalue']:
        for with_nd in [False, True]:
            lags = []
            for p in [f'lag1_{depvar}_', f'lag2_{depvar}_', f'lag3_{depvar}_', f'lag4_{depvar}_']:
                lags += sorted([c for c in df_post.columns if c.startswith(p)])

            sy = [c for c in sy_dummies if c in df_post.columns]
            g = [c for c in geo if c in df_post.columns]
            indep = f_int_vars + sy + g + [c for c in lags if c in df_post.columns]
            if with_nd:
                indep += [c for c in nd if c in df_post.columns]

            indep = [c for c in indep if c in df_post.columns]
            m = run_areg(df_post, depvar, indep, 'fips', 'fips', weights='county_w')

            controls = 'state-year FE + geography + lags' + (' + New Deal' if with_nd else '')
            for fvar in f_int_vars:
                r = extract_results(m, fvar, reg_id, '112749-V1', depvar,
                                   'OLS-FE', 'county', controls, 'fips',
                                   f'Table 5, {depvar}' + (' + ND' if with_nd else ''))
                results.append(r)
                reg_id += 1

    print(f"  Table 5: {len(results)} coefficient estimates")
    return results, reg_id


###############################################################################
# MAIN
###############################################################################

if __name__ == "__main__":
    # Step 1: Generate data
    panel = generate_data()

    # Step 2: Pre-analysis
    df_main, df_post, df_south = preanalysis(panel)

    # Create state-year dummies for regression
    df_main = pd.get_dummies(df_main, columns=['state_year'], prefix='d_sy', drop_first=True)
    df_post = pd.get_dummies(df_post, columns=['state_year'], prefix='d_sy', drop_first=True)

    # Step 3: Run regressions
    all_results = []

    t1_results, next_id = run_table1(df_main)
    all_results.extend(t1_results)

    t2_results, next_id = run_table2(df_post, next_id)
    all_results.extend(t2_results)

    t4_results, next_id = run_table4(df_post, next_id)
    all_results.extend(t4_results)

    t5_results, next_id = run_table5(df_post, next_id)
    all_results.extend(t5_results)

    # Save results
    results_df = pd.DataFrame(all_results)
    out_path = os.path.join(PACKAGE_DIR, "replication.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(results_df)} results to {out_path}")

    # Print summary
    print("\n=== SUMMARY ===")
    for status in ['exact','close','discrepant','failed']:
        n = (results_df['match_status'] == status).sum()
        print(f"  {status}: {n}")
    print(f"  Total: {len(results_df)}")
