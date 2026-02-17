"""
Specification Search Script for Hornbeck & Naidu (2014)
"When the Levee Breaks: Black Migration and Economic Development in the American South"
AER, 104(3): 963-990

Paper ID: 112749-V1

This script executes the approved specification surface for two baseline groups:
  G1: Black population share (lnfrac_black, Table 2)
  G2: Farm equipment value (lnvalue_equipment, Table 4)

It reuses the data assembly from the replication script and runs
all baseline + design + RC specifications defined in SPECIFICATION_SURFACE.json.
"""

import pandas as pd
import numpy as np
import pyreadstat
import pyfixest as pf
import json
import os
import warnings
import traceback
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================
PAPER_ID = "112749-V1"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PKG_DIR = os.path.join(REPO_ROOT, "data", "downloads", "extracted", PAPER_ID)
DATA_DIR = os.path.join(PKG_DIR, "Replication_AER-2012-0980", "Generate_Data")
OUT_DIR = PKG_DIR

# =============================================================================
# HELPER FUNCTIONS (from replication script)
# =============================================================================
def read_dta_robust(path):
    try:
        return pd.read_stata(path, convert_categoricals=False)
    except (ValueError, Exception):
        df, _ = pyreadstat.read_dta(path)
        return df

SOUTHERN_STATES = [32, 34, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54]
FIPS_STATES = [5, 22, 28, 47, 1, 13, 37, 45, 12]

def filter_southern(df, col='state'):
    return df[df[col].isin(SOUTHERN_STATES)].copy()

def filter_level1(df):
    return df[df['level'] == 1].copy()

def border_adjust(year_data, crosswalk_file, year_vars):
    cw = pd.read_csv(crosswalk_file, sep='\t')
    cw = cw[cw['state'] == cw['state_1']].copy()
    cw['percent'] = cw['new_area'] / cw['area']
    cw = cw.rename(columns={'id': 'fips'})
    merged = pd.merge(cw[['fips', 'id_1', 'percent']], year_data, on='fips', how='inner')
    for var in year_vars:
        if var in merged.columns:
            mask_missing = merged[var].isna() & (merged['percent'] > 0.01)
            merged.loc[~merged[var].isna(), var] = merged.loc[~merged[var].isna(), var] * merged.loc[~merged[var].isna(), 'percent']
            merged.loc[mask_missing, var] = -1e23
    agg_vars = [v for v in year_vars if v in merged.columns]
    result = merged.groupby('id_1')[agg_vars].sum().reset_index()
    for var in agg_vars:
        result.loc[result[var] < 0, var] = np.nan
    return result

def get_control_cols(df, patterns):
    cols = []
    for pat in patterns:
        if pat.endswith('*'):
            prefix = pat[:-1]
            cols.extend([c for c in df.columns if c.startswith(prefix)])
        else:
            if pat in df.columns:
                cols.append(pat)
    return sorted(set(cols))

# =============================================================================
# DATA ASSEMBLY (identical to replication script)
# =============================================================================
print("=" * 70)
print("DATA ASSEMBLY")
print("=" * 70)

farmval = read_dta_robust(f'{DATA_DIR}/farmval.dta')
farmval = filter_level1(farmval); farmval = filter_southern(farmval)

# 1900
df_1900_icpsr = read_dta_robust(f'{DATA_DIR}/02896-0020-Data.dta')
df_1900_icpsr = filter_level1(df_1900_icpsr); df_1900_icpsr = filter_southern(df_1900_icpsr)
df_1900_icpsr = df_1900_icpsr.dropna(subset=['fips'])
df_1900_icpsr['population'] = df_1900_icpsr['totpop']
df_1900_icpsr['population_race_white'] = (df_1900_icpsr['nbwmnp'] + df_1900_icpsr['nbwmfp'] + df_1900_icpsr['fbwmtot'] + df_1900_icpsr['nbwfnp'] + df_1900_icpsr['nbwffp'] + df_1900_icpsr['fbwftot'])
df_1900_icpsr['population_race_black'] = df_1900_icpsr['negmtot'] + df_1900_icpsr['negftot']
df_1900_icpsr['population_race_other'] = df_1900_icpsr['population'] - df_1900_icpsr['population_race_white'] - df_1900_icpsr['population_race_black']
df_1900_icpsr.rename(columns={'farmwh': 'farms_white', 'farmcol': 'farms_nonwhite', 'acfarm': 'farmland', 'farmbui': 'value_buildings', 'farmequi': 'value_equipment', 'area': 'county_squaremiles'}, inplace=True)
df_1900_icpsr['farms_owner'] = df_1900_icpsr['farmwhow'] + df_1900_icpsr['farmcoow']
df_1900_icpsr['farms_tenant'] = df_1900_icpsr['farmwhct'] + df_1900_icpsr['farmcoct'] + df_1900_icpsr['farmwhst'] + df_1900_icpsr['farmcost']
df_1900_icpsr['farms_nonwhite_tenant'] = df_1900_icpsr['farmcoct'] + df_1900_icpsr['farmcost']
df_1900_icpsr['farms_tenant_cash'] = df_1900_icpsr['farmwhct'] + df_1900_icpsr['farmcoct']
df_1900_icpsr['year'] = 1900
keep_1900 = ['fips','year','state','county','county_squaremiles','population','population_race_white','population_race_black','population_race_other','farms','farms_white','farms_nonwhite','farms_nonwhite_tenant','farms_owner','farms_tenant','farms_tenant_cash','farmland','value_buildings','value_equipment','mfgestab','mfgwages','mfgavear']
df_1900_icpsr = df_1900_icpsr[[c for c in keep_1900 if c in df_1900_icpsr.columns]]
df_1900_ag = read_dta_robust(f'{DATA_DIR}/ag900co.dta')
df_1900_ag = filter_level1(df_1900_ag); df_1900_ag = filter_southern(df_1900_ag)
df_1900_ag.rename(columns={'cornac':'corn_a','oatsac':'oats_a','wheatac':'wheat_a','cottonac':'cotton_a','riceac':'rice_a','scaneac':'scane_a','corn':'corn_y','oats':'oats_y','wheat':'wheat_y','cotbale1':'cotton_y','rice':'rice_y','csugarwt':'scane_y'}, inplace=True)
df_1900_ag['horses'] = df_1900_ag[['colts0','colts1_2','horses2_']].fillna(0).sum(axis=1)
df_1900_ag['mules'] = df_1900_ag[['mules0','mules1_2','mules2_']].fillna(0).sum(axis=1)
for v in ['corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a']: df_1900_ag[v] = pd.to_numeric(df_1900_ag[v], errors='coerce').fillna(0)
df_1900_ag = df_1900_ag[[c for c in ['fips','corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y','horses','mules'] if c in df_1900_ag.columns]]
df_1900 = pd.merge(df_1900_ag, df_1900_icpsr, on='fips', how='inner')
df_1900 = pd.merge(df_1900, farmval[['fips','faval900']].rename(columns={'faval900':'farmval'}), on='fips', how='inner')
df_1900['value_landbuildings'] = df_1900['farmval'] * df_1900['farmland']
df_1900['value_land'] = df_1900['value_landbuildings'] - df_1900['value_buildings']
df_1900.drop(columns=['farmval'], inplace=True)
df_1900.rename(columns={'fips':'id_1'}, inplace=True)

# 1910
df_1910_icpsr = read_dta_robust(f'{DATA_DIR}/02896-0022-Data.dta')
df_1910_icpsr = filter_level1(df_1910_icpsr); df_1910_icpsr = filter_southern(df_1910_icpsr)
df_1910_icpsr['population'] = df_1910_icpsr['totpop']
df_1910_icpsr['population_race_white'] = df_1910_icpsr['wmtot'] + df_1910_icpsr['wftot']
df_1910_icpsr['population_race_black'] = df_1910_icpsr['negmtot'] + df_1910_icpsr['negftot']
df_1910_icpsr['population_race_other'] = df_1910_icpsr['population'] - df_1910_icpsr['population_race_white'] - df_1910_icpsr['population_race_black']
df_1910_icpsr['farms_white'] = df_1910_icpsr['farmnw'] + df_1910_icpsr['farmfbw']
df_1910_icpsr.rename(columns={'farmneg':'farms_nonwhite','farmown':'farms_owner','farmten':'farms_tenant','farmcten':'farms_tenant_cash','area':'county_squaremiles','rur1910':'population_rural'}, inplace=True)
df_1910_icpsr['farmland'] = df_1910_icpsr['acresown'].fillna(0) + df_1910_icpsr['acresten'].fillna(0) + df_1910_icpsr.get('acresman', pd.Series(0, index=df_1910_icpsr.index)).fillna(0)
df_1910_icpsr['farms_nonwhite_tenant'] = df_1910_icpsr['farmnegt']
df_1910_icpsr['year'] = 1910
df_1910_ag = read_dta_robust(f'{DATA_DIR}/ag910co.dta')
df_1910_ag = filter_level1(df_1910_ag); df_1910_ag = filter_southern(df_1910_ag)
df_1910_ag.rename(columns={'farmbui':'value_buildings','farmequi':'value_equipment','cornac':'corn_a','oatsac':'oats_a','wheatac':'wheat_a','cottonac':'cotton_a','riceac':'rice_a','csugarac':'scane_a','corn':'corn_y','oats':'oats_y','wheat':'wheat_y','cotton':'cotton_y','rice':'rice_y','csugar':'scane_y'}, inplace=True)
for v in ['corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a']:
    if v in df_1910_ag.columns: df_1910_ag[v] = pd.to_numeric(df_1910_ag[v], errors='coerce').fillna(0)
df_1910_ag = df_1910_ag[[c for c in ['fips','value_buildings','value_equipment','corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y','horses','mules'] if c in df_1910_ag.columns]]
df_1910 = pd.merge(df_1910_ag, df_1910_icpsr[['fips','population','population_rural','population_race_white','population_race_black','population_race_other','farms','farms_white','farms_nonwhite','farms_nonwhite_tenant','farms_owner','farms_tenant','farms_tenant_cash','farmland','county_squaremiles','cropval']], on='fips', how='inner')
df_1910 = pd.merge(df_1910, farmval[['fips','faval910']].rename(columns={'faval910':'farmval'}), on='fips', how='inner')
df_1910['value_landbuildings'] = df_1910['farmval'] * df_1910['farmland']
df_1910['value_land'] = df_1910['value_landbuildings'] - df_1910['value_buildings']
df_1910.drop(columns=['farmval'], inplace=True)
adj_vars_1910 = ['county_squaremiles','population','population_rural','population_race_white','population_race_black','population_race_other','farms','farms_white','farms_nonwhite','farms_nonwhite_tenant','farms_owner','farms_tenant','farms_tenant_cash','farmland','value_landbuildings','value_land','value_buildings','value_equipment','horses','mules','corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y','cropval']
df_1910_adj = border_adjust(df_1910, f'{DATA_DIR}/Export1910_1900.txt', adj_vars_1910); df_1910_adj['year'] = 1910

# 1920
df_1920_icpsr = read_dta_robust(f'{DATA_DIR}/02896-0024-Data.dta')
df_1920_icpsr = filter_level1(df_1920_icpsr); df_1920_icpsr = filter_southern(df_1920_icpsr)
df_1920_icpsr['population'] = df_1920_icpsr['totpop']
df_1920_icpsr['population_race_white'] = df_1920_icpsr['nwmtot'] + df_1920_icpsr['fbwmtot'] + df_1920_icpsr['nwftot'] + df_1920_icpsr['fbwftot']
df_1920_icpsr['population_race_black'] = df_1920_icpsr['negmtot'] + df_1920_icpsr['negftot']
df_1920_icpsr['population_race_other'] = df_1920_icpsr['population'] - df_1920_icpsr['population_race_white'] - df_1920_icpsr['population_race_black']
df_1920_icpsr['farms_white'] = df_1920_icpsr['farmnw'] + df_1920_icpsr['farmfbw']
df_1920_icpsr.rename(columns={'farmcol':'farms_nonwhite','farmown':'farms_owner','farmten':'farms_tenant','farmcten':'farms_tenant_cash','areaac':'county_acres','area':'county_squaremiles','farmbui':'value_buildings','farmequi':'value_equipment','farmcolt':'farms_nonwhite_tenant'}, inplace=True)
df_1920_icpsr['farmland'] = df_1920_icpsr['acresown'].fillna(0) + df_1920_icpsr['acresten'].fillna(0) + df_1920_icpsr.get('acresman', pd.Series(0, index=df_1920_icpsr.index)).fillna(0)
df_1920_icpsr['population_rural'] = df_1920_icpsr['population'] - df_1920_icpsr['urb920'].fillna(0)
df_1920_icpsr['year'] = 1920
df_1920_ag = read_dta_robust(f'{DATA_DIR}/ag920co.dta')
df_1920_ag = filter_level1(df_1920_ag); df_1920_ag = filter_southern(df_1920_ag)
df_1920_ag.rename(columns={'var147':'corn_a','var149':'oats_a','var151':'wheat_a','var165':'rice_a','var216':'cotton_a','var221':'scane_a','var56':'horses','var63':'mules','var148':'corn_y','var150':'oats_y','var152':'wheat_y','var166':'rice_y','var217':'cotton_y','var222':'scane_y'}, inplace=True)
for v in ['oats_y','corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a']:
    if v in df_1920_ag.columns: df_1920_ag[v] = pd.to_numeric(df_1920_ag[v], errors='coerce').fillna(0)
df_1920_ag = df_1920_ag[[c for c in ['fips','corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y','horses','mules'] if c in df_1920_ag.columns]]
out_migrant = read_dta_robust(f'{DATA_DIR}/out_migrant_counts.dta')
df_1920 = pd.merge(df_1920_ag, df_1920_icpsr[[c for c in ['fips','county_acres','county_squaremiles','population','population_rural','population_race_white','population_race_black','population_race_other','farms','farms_white','farms_nonwhite','farms_nonwhite_tenant','farms_owner','farms_tenant','farms_tenant_cash','farmland','value_buildings','value_equipment','cropval','mfgwages','mfgestab','mfgavear'] if c in df_1920_icpsr.columns]], on='fips', how='inner')
df_1920 = pd.merge(df_1920, farmval[['fips','faval920']].rename(columns={'faval920':'farmval'}), on='fips', how='inner')
df_1920['value_landbuildings'] = df_1920['farmval'] * df_1920['farmland']
df_1920['value_land'] = df_1920['value_landbuildings'] - df_1920['value_buildings']
df_1920.drop(columns=['farmval'], inplace=True)
df_1920 = pd.merge(df_1920, out_migrant, on='fips', how='left')
adj_vars_1920 = ['county_acres','county_squaremiles','population','population_rural','population_race_white','population_race_black','population_race_other','farms','farms_white','farms_nonwhite','farms_nonwhite_tenant','farms_owner','farms_tenant','farms_tenant_cash','farmland','value_landbuildings','value_land','value_buildings','value_equipment','horses','mules','corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y','cropval','people','samecounty','samestate','samearea','sameregion','people_white','samecounty_white','samestate_white','samearea_white','sameregion_white','people_black','samecounty_black','samestate_black','samearea_black','sameregion_black','mfgwages','mfgestab','mfgavear']
df_1920_adj = border_adjust(df_1920, f'{DATA_DIR}/Export1920_1900.txt', adj_vars_1920); df_1920_adj['year'] = 1920

# 1925-1970 (condensed)
icpsr_fips = read_dta_robust(f'{DATA_DIR}/icpsr_fips.dta')
h25 = pd.read_csv(f'{DATA_DIR}/Haines_1925.txt', sep='\t'); h25.rename(columns={'var2':'farms','var42':'farmland','var99':'value_equipment','var169':'horses','var173':'mules'}, inplace=True)
h25 = pd.merge(h25[['state','county','farms','farmland','value_equipment','horses','mules']], icpsr_fips, on=['state','county'], how='inner')
h25n = pd.read_csv(f'{DATA_DIR}/Haines_1925_new.txt', sep='\t'); h25n = h25n[(h25n['state']>0)&(h25n['state']<=73)].copy()
for v in ['corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y']:
    if v in h25n.columns: h25n[v] = pd.to_numeric(h25n[v], errors='coerce').fillna(0)
h25n = pd.merge(h25n, icpsr_fips, on=['state','county'], how='inner')
t25 = read_dta_robust(f'{DATA_DIR}/tractors1925.dta'); t25.rename(columns={'tractors1925':'tractors'}, inplace=True); t25 = t25[['fips','tractors']]
df_1925 = pd.merge(farmval[['fips','faval925']].rename(columns={'faval925':'farmval'}), h25[['fips','farms','farmland','value_equipment','horses','mules']], on='fips', how='inner')
cc25 = ['fips'] + [c for c in ['corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y'] if c in h25n.columns]
df_1925 = pd.merge(df_1925, h25n[cc25], on='fips', how='inner')
df_1925 = pd.merge(df_1925, t25, on='fips', how='left')
df_1925['value_landbuildings'] = df_1925['farmval'] * df_1925['farmland']; df_1925.drop(columns=['farmval'], inplace=True)
df_1925_adj = border_adjust(df_1925, f'{DATA_DIR}/Export1920_1900.txt', ['farms','farmland','value_landbuildings','value_equipment','horses','mules','corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y','tractors']); df_1925_adj['year'] = 1925

# 1930
h30_1 = pd.read_csv(f'{DATA_DIR}/Haines_1930.txt', sep='\t')[['state','county','horses','mules','tractors']]
h30n = pd.read_csv(f'{DATA_DIR}/Haines_1930_new.txt', sep='\t'); h30n = h30n[(h30n['state']>0)&(h30n['state']<=73)].copy()
for v in ['cotton_a','corn_a','oats_a','wheat_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y']:
    if v in h30n.columns: h30n[v] = pd.to_numeric(h30n[v], errors='coerce').fillna(0)
h30 = pd.merge(h30n, h30_1, on=['state','county'], how='inner'); h30 = pd.merge(h30, icpsr_fips, on=['state','county'], how='inner')
redcross = pd.read_csv(f'{DATA_DIR}/redcross_new.txt', sep='\t')
for v in ['flooded_acres','pop_affected','agricultural_flooded_acres']:
    if v in redcross.columns: redcross[v] = redcross[v].fillna(0)
df_1930_icpsr = read_dta_robust(f'{DATA_DIR}/02896-0026-Data.dta')
df_1930_icpsr = filter_level1(df_1930_icpsr); df_1930_icpsr = filter_southern(df_1930_icpsr)
df_1930_icpsr['population'] = df_1930_icpsr['totpop']
df_1930_icpsr['population_race_white'] = df_1930_icpsr['nwmtot'] + df_1930_icpsr['fbwmtot'] + df_1930_icpsr['nwftot'] + df_1930_icpsr['fbwftot']
df_1930_icpsr['population_race_black'] = df_1930_icpsr['negmtot'] + df_1930_icpsr['negftot']
df_1930_icpsr['population_race_other'] = df_1930_icpsr['population'] - df_1930_icpsr['population_race_white'] - df_1930_icpsr['population_race_black']
df_1930_icpsr['farms_white'] = df_1930_icpsr['farmwh']
df_1930_icpsr.rename(columns={'farmcol':'farms_nonwhite','farmten':'farms_tenant','farmcten':'farms_tenant_cash','acres':'farmland','areaac':'county_acres','area':'county_squaremiles','farmbui':'value_buildings','farmequi':'value_equipment'}, inplace=True)
df_1930_icpsr['farms_owner'] = df_1930_icpsr['farmfown'] + df_1930_icpsr['farmpown']
df_1930_icpsr['population_rural'] = df_1930_icpsr['population'] - df_1930_icpsr['urban30'].fillna(0)
in_migrant = read_dta_robust(f'{DATA_DIR}/in_migrant_counts.dta')
df_1930 = df_1930_icpsr[[c for c in ['fips','county_acres','county_squaremiles','population','population_rural','population_race_white','population_race_black','population_race_other','farms','farms_white','farms_nonwhite','farms_owner','farms_tenant','farms_tenant_cash','farmland','value_buildings','value_equipment','cropval','mfgwages','mfgestab','mfgavear'] if c in df_1930_icpsr.columns]].copy()
df_1930 = pd.merge(df_1930, farmval[['fips','faval930']].rename(columns={'faval930':'farmval'}), on='fips', how='inner')
df_1930 = pd.merge(df_1930, in_migrant, on='fips', how='left')
hk30 = [c for c in ['fips','horses','mules','tractors','corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y'] if c in h30.columns]
df_1930 = pd.merge(df_1930, h30[hk30], on='fips', how='inner')
df_1930 = pd.merge(df_1930, redcross[['fips','flooded_acres','pop_affected','agricultural_flooded_acres']], on='fips', how='left')
df_1930['value_landbuildings'] = df_1930['farmval'] * df_1930['farmland']
df_1930['value_land'] = df_1930['value_landbuildings'] - df_1930['value_buildings']; df_1930.drop(columns=['farmval'], inplace=True)
adj_vars_1930 = ['county_acres','county_squaremiles','population','population_rural','population_race_white','population_race_black','population_race_other','farms','farms_white','farms_nonwhite','farms_owner','farms_tenant','farms_tenant_cash','farmland','value_landbuildings','value_land','value_buildings','value_equipment','corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y','horses','mules','tractors','flooded_acres','pop_affected','agricultural_flooded_acres','in_people','in_samecounty','in_samestate','in_samearea','in_sameregion','in_people_white','in_samecounty_white','in_samestate_white','in_samearea_white','in_sameregion_white','in_people_black','in_samecounty_black','in_samestate_black','in_samearea_black','in_sameregion_black','cropval','mfgwages','mfgestab','mfgavear']
df_1930_adj = border_adjust(df_1930, f'{DATA_DIR}/Export1930_1900.txt', adj_vars_1930); df_1930_adj['year'] = 1930

# 1935
h35 = pd.read_csv(f'{DATA_DIR}/Haines_1935.txt', sep='\t'); h35.rename(columns={'var2':'farms','var12':'farmland','var95':'horses','var100':'mules'}, inplace=True)
h35 = pd.merge(h35[['state','county','farms','farmland','horses','mules']], icpsr_fips, on=['state','county'], how='inner')
h35n = pd.read_csv(f'{DATA_DIR}/Haines_1935_new.txt', sep='\t'); h35n = h35n[(h35n['state']>0)&(h35n['state']<=73)].copy()
if 'wheat1_a' in h35n.columns:
    h35n['wheat_a'] = pd.to_numeric(h35n.get('wheat1_a',0), errors='coerce').fillna(0) + pd.to_numeric(h35n.get('wheat2_a',0), errors='coerce').fillna(0)
    h35n['wheat_y'] = pd.to_numeric(h35n.get('wheat1_y',0), errors='coerce').fillna(0) + pd.to_numeric(h35n.get('wheat2_y',0), errors='coerce').fillna(0)
for v in ['corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y']:
    if v in h35n.columns: h35n[v] = pd.to_numeric(h35n[v], errors='coerce').fillna(0)
h35n = pd.merge(h35n, icpsr_fips, on=['state','county'], how='inner')
df_1935 = pd.merge(farmval[['fips','faval935']].rename(columns={'faval935':'farmval'}), h35[['fips','farms','farmland','horses','mules']], on='fips', how='inner')
cc35 = ['fips'] + [c for c in ['corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y'] if c in h35n.columns]
df_1935 = pd.merge(df_1935, h35n[cc35], on='fips', how='inner')
df_1935['value_landbuildings'] = df_1935['farmval'] * df_1935['farmland']; df_1935.drop(columns=['farmval'], inplace=True)
df_1935_adj = border_adjust(df_1935, f'{DATA_DIR}/Export1930_1900.txt', ['farms','farmland','value_landbuildings','horses','mules','corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y']); df_1935_adj['year'] = 1935

# 1940
h40 = pd.read_csv(f'{DATA_DIR}/Haines_1940_new.txt', sep='\t'); h40 = h40[(h40['state']>0)&(h40['state']<=73)].copy()
for v in ['oats_y','wheat_y','corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','cotton_y','rice_y','scane_y']:
    if v in h40.columns: h40[v] = pd.to_numeric(h40[v], errors='coerce').fillna(0)
h40 = pd.merge(h40, icpsr_fips, on=['state','county'], how='inner')
df_1940_i = read_dta_robust(f'{DATA_DIR}/02896-0032-Data.dta'); df_1940_i = filter_level1(df_1940_i); df_1940_i = filter_southern(df_1940_i)
df_1940_i['population'] = df_1940_i['totpop']; df_1940_i['population_race_white'] = df_1940_i['nwtot'] + df_1940_i['fbwtot']; df_1940_i['population_race_black'] = df_1940_i['negtot']; df_1940_i['population_race_other'] = df_1940_i['population'] - df_1940_i['population_race_white'] - df_1940_i['population_race_black']
df_1940_i.rename(columns={'farmnonw':'farms_nonwhite','farmten':'farms_tenant','farmcten':'farms_tenant_cash','acfarms':'farmland','buildval':'value_buildings','equipval':'value_equipment','areaac':'county_acres','area':'county_squaremiles'}, inplace=True)
df_1940_i['farms_owner'] = df_1940_i['farmfown'] + df_1940_i['farmpown']; df_1940_i['population_rural'] = df_1940_i['population'] - df_1940_i['urb940'].fillna(0)
df_1940_e = read_dta_robust(f'{DATA_DIR}/02896-0070-Data.dta'); df_1940_e = filter_level1(df_1940_e); df_1940_e = filter_southern(df_1940_e)
df_1940_e.rename(columns={'var57':'tractors','var56':'mules_horses'}, inplace=True)
df_1940 = df_1940_i[[c for c in ['fips','county_acres','county_squaremiles','population','population_rural','population_race_white','population_race_black','population_race_other','farms','farms_nonwhite','farms_owner','farms_tenant','farms_tenant_cash','farmland','value_buildings','value_equipment','cropval','mfgwages','mfgestab','mfgavear'] if c in df_1940_i.columns]].copy()
df_1940 = pd.merge(df_1940, df_1940_e[['fips','tractors','mules_horses']], on='fips', how='inner')
df_1940 = pd.merge(df_1940, farmval[['fips','faval940']].rename(columns={'faval940':'farmval'}), on='fips', how='inner')
hk40 = ['fips'] + [c for c in ['corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y'] if c in h40.columns]
df_1940 = pd.merge(df_1940, h40[hk40], on='fips', how='inner')
df_1940['value_landbuildings'] = df_1940['farmval'] * df_1940['farmland']; df_1940['value_land'] = df_1940['value_landbuildings'] - df_1940['value_buildings']; df_1940.drop(columns=['farmval'], inplace=True)
df_1940_adj = border_adjust(df_1940, f'{DATA_DIR}/Export1940_1900.txt', ['county_acres','county_squaremiles','population','population_rural','population_race_white','population_race_black','population_race_other','farms','farms_nonwhite','farms_owner','farms_tenant','farms_tenant_cash','farmland','value_landbuildings','value_land','value_buildings','value_equipment','mules_horses','tractors','corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y','cropval']); df_1940_adj['year'] = 1940

# 1945
df_1945_r = read_dta_robust(f'{DATA_DIR}/02896-0071-Data.dta'); df_1945_r = filter_level1(df_1945_r); df_1945_r = filter_southern(df_1945_r)
df_1945_r.rename(columns={'var69':'farms','var71':'farmland','var82':'tractors'}, inplace=True); df_1945_r['farmland'] = df_1945_r['farmland'] * 1000
df_1945 = pd.merge(df_1945_r[['fips','farms','farmland','tractors']], farmval[['fips','faval945']].rename(columns={'faval945':'farmval'}), on='fips', how='inner')
df_1945['value_landbuildings'] = df_1945['farmval'] * df_1945['farmland']; df_1945.drop(columns=['farmval'], inplace=True)
df_1945_adj = border_adjust(df_1945, f'{DATA_DIR}/Export1940_1900.txt', ['farms','farmland','value_landbuildings','tractors']); df_1945_adj['year'] = 1945

# 1950
df_1950_i = read_dta_robust(f'{DATA_DIR}/02896-0035-Data.dta'); df_1950_i = filter_level1(df_1950_i); df_1950_i = filter_southern(df_1950_i)
df_1950_i['population'] = df_1950_i['totpop']; df_1950_i['population_race_white'] = df_1950_i['nwmtot'] + df_1950_i['nwftot'] + df_1950_i['fbwmtot'] + df_1950_i['fbwftot']
df_1950_i['population_race_black'] = df_1950_i['negmtot'] + df_1950_i['negftot']; df_1950_i['population_race_other'] = df_1950_i['population'] - df_1950_i['population_race_white'] - df_1950_i['population_race_black']
df_1950_i.rename(columns={'farmnonw':'farms_nonwhite','farmten':'farms_tenant','farmcten':'farms_tenant_cash','acres':'farmland','areaac':'county_acres','area':'county_squaremiles'}, inplace=True)
df_1950_i['farms_owner'] = df_1950_i['farmfown'] + df_1950_i['farmpown']; df_1950_i['population_rural'] = df_1950_i['population'] - df_1950_i['urb950'].fillna(0)
aw50 = read_dta_robust(f'{DATA_DIR}/usag1949.work.dta'); aw50 = filter_southern(aw50)
aw50.rename(columns={'var2':'corn_a','var3':'corn_y','var13':'wheat_a','var14':'wheat_y','var23':'oats_a','var24':'oats_y','var27':'rice_a','var28':'rice_y','var72':'cotton_a','var73':'cotton_y','var80':'scane_a','var82':'scane_y'}, inplace=True)
for v in ['corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y']:
    if v in aw50.columns: aw50[v] = pd.to_numeric(aw50[v], errors='coerce').fillna(0)
ac50 = read_dta_robust(f'{DATA_DIR}/usag1949.cos.crops.dta'); ac50.rename(columns={'item742':'horses','item744':'mules_horses'}, inplace=True)
df_1950 = df_1950_i[[c for c in ['fips','county_acres','county_squaremiles','population','population_rural','population_race_white','population_race_black','population_race_other','farms','farms_nonwhite','farms_owner','farms_tenant','farms_tenant_cash','farmland','cropval'] if c in df_1950_i.columns]].copy()
cc50 = [c for c in ['fips','corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y'] if c in aw50.columns]
df_1950 = pd.merge(df_1950, aw50[cc50], on='fips', how='inner')
df_1950 = pd.merge(df_1950, ac50[['fips','horses','mules_horses']], on='fips', how='inner')
df_1950 = pd.merge(df_1950, farmval[['fips','faval950']].rename(columns={'faval950':'farmval'}), on='fips', how='inner')
df_1950['value_landbuildings'] = df_1950['farmval'] * df_1950['farmland']; df_1950.drop(columns=['farmval'], inplace=True)
df_1950_adj = border_adjust(df_1950, f'{DATA_DIR}/Export1950_1900.txt', ['county_acres','county_squaremiles','population','population_rural','population_race_white','population_race_black','population_race_other','farms','farms_nonwhite','farms_owner','farms_tenant','farms_tenant_cash','farmland','value_landbuildings','horses','mules_horses','corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y','cropval']); df_1950_adj['year'] = 1950

# 1954-1970 (condensed)
ir59 = {'item101':'corn_a','item119':'oats_a','item122':'rice_a','item113':'wheat_a','item175':'cotton_a','item187':'scane_a','item742':'horses','item744':'mules_horses','item401':'corn_y','item413':'wheat_y','item419':'oats_y','item422':'rice_y','item475':'cotton_y','item487':'scane_y'}
a54 = read_dta_robust(f'{DATA_DIR}/usag1954.cos.crops.dta'); a54.rename(columns=ir59, inplace=True)
for v in ['corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y']:
    if v in a54.columns: a54[v] = pd.to_numeric(a54[v], errors='coerce').fillna(0)
e54 = read_dta_robust(f'{DATA_DIR}/02896-0073-Data.dta'); e54 = filter_level1(e54); e54 = filter_southern(e54)
e54.rename(columns={'var100':'farms','var101':'farmland','var126':'tractors'}, inplace=True); e54['farmland'] = e54['farmland'] * 1000
df_1954 = pd.merge(e54[['fips','farms','farmland','tractors']], a54[['fips','corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y','horses','mules_horses']], on='fips', how='inner')
df_1954 = pd.merge(df_1954, farmval[['fips','faval954']].rename(columns={'faval954':'farmval'}), on='fips', how='inner')
df_1954['value_landbuildings'] = df_1954['farmval'] * df_1954['farmland']; df_1954.drop(columns=['farmval'], inplace=True)
df_1954_adj = border_adjust(df_1954, f'{DATA_DIR}/Export1950_1900.txt', ['farms','farmland','value_landbuildings','horses','mules_horses','tractors','corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y']); df_1954_adj['year'] = 1954

df_1960_i = read_dta_robust(f'{DATA_DIR}/02896-0038-Data.dta'); df_1960_i = filter_level1(df_1960_i); df_1960_i = filter_southern(df_1960_i)
df_1960_i['population'] = df_1960_i['totpop']; df_1960_i['population_race_white'] = df_1960_i['wmtot'] + df_1960_i['wftot']; df_1960_i['population_race_black'] = df_1960_i['negmtot'] + df_1960_i['negftot']; df_1960_i['population_race_other'] = df_1960_i['population'] - df_1960_i['population_race_white'] - df_1960_i['population_race_black']
e60 = read_dta_robust(f'{DATA_DIR}/02896-0074-Data.dta'); e60 = filter_level1(e60); e60 = filter_southern(e60); e60.rename(columns={'var146':'value_perfarm','var147':'value_peracre','var149':'cropval','var6':'percent_urban'}, inplace=True)
a59 = read_dta_robust(f'{DATA_DIR}/usag1959.cos.crops.dta'); a59.rename(columns=ir59, inplace=True)
for v in ['corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y']:
    if v in a59.columns: a59[v] = pd.to_numeric(a59[v], errors='coerce').fillna(0)
aw59 = read_dta_robust(f'{DATA_DIR}/usag1959.work.dta'); aw59 = filter_southern(aw59); aw59.rename(columns={'var250':'farms','var254':'farmland'}, inplace=True)
df_1960 = pd.merge(aw59[['fips','farms','farmland']], df_1960_i[['fips','population','population_race_white','population_race_black','population_race_other']], on='fips', how='inner')
df_1960 = pd.merge(df_1960, e60[['fips','value_perfarm','value_peracre','cropval','percent_urban']], on='fips', how='inner')
df_1960 = pd.merge(df_1960, a59[['fips','corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y','horses','mules_horses']], on='fips', how='inner')
df_1960 = pd.merge(df_1960, farmval[['fips','faval959']].rename(columns={'faval959':'farmval'}), on='fips', how='inner')
df_1960['value_landbuildings'] = df_1960['farmval'] * df_1960['farmland']; df_1960['value_landbuildings_peracre'] = df_1960['value_peracre'] * df_1960['farmland']; df_1960['value_landbuildings_perfarm'] = df_1960['value_perfarm'] * df_1960['farms']
df_1960['population_rural'] = df_1960['population'] * (1 - df_1960['percent_urban'].fillna(0)/100); df_1960.drop(columns=['farmval'], inplace=True)
df_1960_adj = border_adjust(df_1960, f'{DATA_DIR}/Export1960_1900.txt', ['population','population_rural','population_race_white','population_race_black','population_race_other','farms','farmland','value_landbuildings','value_landbuildings_peracre','value_landbuildings_perfarm','horses','mules_horses','corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y','cropval']); df_1960_adj['year'] = 1960

a64 = read_dta_robust(f'{DATA_DIR}/usag1964.cos.crops.dta'); a64.rename(columns=ir59, inplace=True)
for v in ['corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y']:
    if v in a64.columns: a64[v] = pd.to_numeric(a64[v], errors='coerce').fillna(0)
e64 = read_dta_robust(f'{DATA_DIR}/02896-0075-Data.dta'); e64 = filter_level1(e64); e64 = filter_southern(e64)
e64.rename(columns={'var124':'farms','var126':'farmland','var129':'value_perfarm','var128':'value_peracre'}, inplace=True); e64['farmland'] = e64['farmland']*1000; e64['value_perfarm'] = e64['value_perfarm']*1000
df_1964 = pd.merge(e64[['fips','farms','farmland','value_perfarm','value_peracre']], a64[['fips','corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y','horses','mules_horses']], on='fips', how='inner')
df_1964['value_landbuildings_perfarm'] = df_1964['value_perfarm'] * df_1964['farms']; df_1964['value_landbuildings_peracre'] = df_1964['value_peracre'] * df_1964['farmland']
df_1964_adj = border_adjust(df_1964, f'{DATA_DIR}/Export1960_1900.txt', ['farms','farmland','value_landbuildings_peracre','value_landbuildings_perfarm','horses','mules_horses','corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y']); df_1964_adj['year'] = 1964

df_1970_i = read_dta_robust(f'{DATA_DIR}/02896-0076-Data.dta'); df_1970_i = filter_level1(df_1970_i); df_1970_i = filter_southern(df_1970_i)
df_1970_i.rename(columns={'var3':'population','var9':'population_race_white','var10':'population_race_black','var173':'farms','var175':'farmland','var178':'value_perfarm','var179':'value_peracre','var121':'mfgestab','var124':'mfgavear','var128':'mfgwages'}, inplace=True)
df_1970_i['population_race_other'] = df_1970_i['population'] - df_1970_i['population_race_white'] - df_1970_i['population_race_black']
df_1970_i['farmland'] = df_1970_i['farmland']*1000; df_1970_i['value_perfarm'] = df_1970_i['value_perfarm']*1000
df_1970_i['value_landbuildings_perfarm'] = df_1970_i['value_perfarm'] * df_1970_i['farms']; df_1970_i['value_landbuildings_peracre'] = df_1970_i['value_peracre'] * df_1970_i['farmland']
a69 = read_dta_robust(f'{DATA_DIR}/usag1969.cos.crops.dta')
if 'stateicp' in a69.columns: a69.rename(columns={'stateicp':'state'}, inplace=True)
a69 = filter_southern(a69); a69.rename(columns=ir59, inplace=True)
for v in ['corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y']:
    if v in a69.columns: a69[v] = pd.to_numeric(a69[v], errors='coerce').fillna(0)
aw69 = read_dta_robust(f'{DATA_DIR}/usag74.1969.allfarms.work.dta'); aw69 = filter_southern(aw69); aw69.rename(columns={'item06002':'value_equipment'}, inplace=True)
if 'value_equipment' in aw69.columns: aw69['value_equipment'] = aw69['value_equipment']*1000
df_1970 = pd.merge(df_1970_i[['fips','population','population_race_white','population_race_black','population_race_other','farms','farmland','value_landbuildings_peracre','value_landbuildings_perfarm']], a69[['fips','corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y','horses','mules_horses']], on='fips', how='inner')
if 'value_equipment' in aw69.columns: df_1970 = pd.merge(df_1970, aw69[['fips','value_equipment']], on='fips', how='left')
df_1970_adj = border_adjust(df_1970, f'{DATA_DIR}/Export1970_1900.txt', ['population','population_race_white','population_race_black','population_race_other','farms','farmland','value_landbuildings_peracre','value_landbuildings_perfarm','value_equipment','horses','mules_horses','corn_a','oats_a','wheat_a','cotton_a','rice_a','scane_a','corn_y','oats_y','wheat_y','cotton_y','rice_y','scane_y']); df_1970_adj['year'] = 1970

# APPEND
all_yrs = [df_1900, df_1910_adj, df_1920_adj, df_1925_adj, df_1930_adj, df_1935_adj, df_1940_adj, df_1945_adj, df_1950_adj, df_1954_adj, df_1960_adj, df_1964_adj, df_1970_adj]
for d in all_yrs:
    if 'id_1' not in d.columns and 'fips' in d.columns: d.rename(columns={'fips':'id_1'}, inplace=True)
panel = pd.concat(all_yrs, ignore_index=True, sort=False); panel.rename(columns={'id_1':'fips'}, inplace=True)
m = panel['value_landbuildings_peracre'].notna() & panel['value_landbuildings_perfarm'].notna()
if m.any(): panel.loc[m, 'value_landbuildings'] = (panel.loc[m, 'value_landbuildings_peracre'] + panel.loc[m, 'value_landbuildings_perfarm'])/2
panel.drop(columns=['value_landbuildings_peracre','value_landbuildings_perfarm'], errors='ignore', inplace=True)
mhm = panel['horses'].notna() & panel['mules'].notna() & panel['mules_horses'].isna()
if mhm.any(): panel.loc[mhm, 'mules_horses'] = panel.loc[mhm, 'horses'] + panel.loc[mhm, 'mules']

# Auxiliary
fl = pd.read_csv(f'{DATA_DIR}/flooded_1900.txt', sep='\t'); fl = fl.groupby('fips').agg({'new_area':'sum','area':'mean'}).reset_index(); fl['flooded_share'] = fl['new_area'] / fl['area']
panel = pd.merge(panel, fl[['fips','flooded_share']], on='fips', how='left')
panel = pd.merge(panel, pd.read_csv(f'{DATA_DIR}/distance_1900.txt', sep='\t'), on='fips', how='left')
panel.sort_values(['fips','year'], inplace=True)
for col in ['state','county','name','x_centroid','y_centroid']:
    if col in panel.columns: panel[col] = panel.groupby('fips')[col].transform(lambda x: x.ffill().bfill())
panel['statefips'] = np.floor(panel['fips']/1000).astype(int); panel = panel[panel['statefips'].isin(FIPS_STATES)].copy()
cs = read_dta_robust(f'{DATA_DIR}/1900_strm_distance_gaez.dta'); cs = cs[cs['fips']!=0]
cs.rename(columns={'cottongaezprod_mean':'cotton_suitability','maizegaezprod_mean':'corn_suitability','wheatgaezprod_mean':'wheat_suitability','ricegaezprod_mean':'rice_suitability','oatgaezprod_mean':'oats_suitability','sugargaezprod_mean':'scane_suitability'}, inplace=True)
panel = pd.merge(panel, cs[['fips','cotton_suitability','corn_suitability','wheat_suitability','rice_suitability','oats_suitability','scane_suitability']], on='fips', how='left')
msd = pd.read_csv(f'{DATA_DIR}/ms_distance.txt', sep='\t'); msd = msd.sort_values(['fips','distance_ms']).drop_duplicates(subset='fips', keep='first')
panel = pd.merge(panel, msd[['fips','distance_ms']], on='fips', how='left')
rg = read_dta_robust(f'{DATA_DIR}/1900_strm_distance_gaez.dta'); rg = rg[rg['fips']!=0][['fips','altitude_std_meters','altitude_range_meters']]
for c in ['altitude_std_meters','altitude_range_meters']:
    if c in panel.columns: panel.drop(columns=[c], inplace=True)
panel = pd.merge(panel, rg, on='fips', how='left')
rv = read_dta_robust(f'{DATA_DIR}/1900_strm_distance.dta'); rv['distance_river'] = rv['Distance_Major_River_Meters']/1000
panel = pd.merge(panel, rv[['fips','distance_river']].drop_duplicates('fips'), on='fips', how='left')
nd = read_dta_robust(f'{DATA_DIR}/new_deal_spending.dta'); nd = filter_southern(nd); nd = nd[nd['county']%1000!=0].copy()
nd = pd.merge(nd, icpsr_fips, on=['state','county'], how='inner')
panel = pd.merge(panel, nd[['fips','pcpubwor','pcaaa','pcrelief','pcndloan','pcndins']], on='fips', how='left')

# PREANALYSIS
panel.rename(columns={'population_race_black':'population_black','population_race_white':'population_white'}, inplace=True)
if 'value_landbuildings' in panel.columns: panel.rename(columns={'value_landbuildings':'value_lb'}, inplace=True)
panel['county_1920'] = np.where(panel['year']==1920, panel['county_acres'], np.nan)
panel['county_w'] = panel.groupby('fips')['county_1920'].transform('max'); panel.drop(columns=['county_1920'], inplace=True)
for c in ['population_black','population','population_white','farms_nonwhite','farms','value_equipment','farmland','value_lb','mules_horses','tractors','county_w']:
    if c in panel.columns: panel[c] = panel[c].astype(float)
panel['lnpopulation_black'] = np.log(panel['population_black'].clip(lower=0.001))
panel['lnpopulation'] = np.log(panel['population'].clip(lower=0.001))
panel['frac_black'] = panel['population_black'] / (panel['population_white'] + panel['population_black'])
panel['lnfrac_black'] = np.log(panel['frac_black'].clip(lower=1e-10))
panel['fracfarms_nonwhite'] = panel['farms_nonwhite'] / panel['farms']
panel['lnfracfarms_nonwhite'] = np.log((panel['farms_nonwhite']/panel['farms']).clip(lower=1e-10))
panel['lnvalue_equipment'] = np.log(panel['value_equipment'].clip(lower=0.001))
panel['avfarmsize'] = panel['farmland'] / panel['farms']
panel['lnavfarmsize'] = np.log(panel['avfarmsize'].clip(lower=0.001))
panel['lnlandbuildingvalue'] = np.log(panel['value_lb'].clip(lower=0.001))
panel['lnlandbuildingvaluef'] = np.log((panel['value_lb']/panel['farmland']).clip(lower=1e-10))
panel['lnmules_horses'] = np.log(panel['mules_horses'].clip(lower=0.001))
panel['lntractors'] = np.log(panel['tractors'].clip(lower=0.001))
panel['lnfarmland_a'] = np.log((panel['farmland']*100/panel['county_w']).clip(lower=1e-10))
panel['frac_black_level'] = panel['frac_black']
panel['value_equipment_level'] = panel['value_equipment']

# Balance
panel['drop'] = 0
for y in [1900,1910,1920,1930,1940,1950,1960,1970]:
    panel.loc[(panel['year']==y) & panel['lnpopulation'].isna(), 'drop'] = 1
    panel.loc[(panel['year']==y) & panel['lnpopulation_black'].isna(), 'drop'] = 1
panel['drop_county'] = panel.groupby('fips')['drop'].transform('max')
panel = panel[panel['drop_county']==0].copy(); panel.drop(columns=['drop','drop_county'], inplace=True)
panel.sort_values(['fips','year'], inplace=True)
panel['number'] = panel.groupby('fips')['year'].transform('count'); panel = panel[panel['number']>=12].copy(); panel.drop(columns=['number'], inplace=True)
panel['cropland'] = panel['corn_a'].fillna(0)+panel['wheat_a'].fillna(0)+panel['oats_a'].fillna(0)+panel['rice_a'].fillna(0)+panel['cotton_a'].fillna(0)+panel['scane_a'].fillna(0)
panel['cotton_c'] = np.where(panel['year']==1920, panel['cotton_a']/panel['cropland'].replace(0,np.nan), np.nan)
fr20 = panel.loc[panel['year']==1920, ['fips','frac_black','cotton_c']].copy()
kf = fr20[(fr20['frac_black']>=0.10)&(fr20['cotton_c']>=0.15)]['fips'].values
panel = panel[panel['fips'].isin(kf)].copy()
panel['number'] = panel.groupby('fips')['year'].transform('count'); panel = panel[panel['number']==13].copy()
panel.drop(columns=['number','cotton_c','cropland'], inplace=True)
panel = panel[~panel['fips'].isin([47149,47071,22023])].copy()

# Population weight
pop20 = panel.loc[panel['year']==1920, ['fips','population']].rename(columns={'population':'pop_w_1920'})
panel = pd.merge(panel, pop20, on='fips', how='left')

# Flood variables
panel['flooded_share'] = panel['flooded_share'].fillna(0)
panel['flood'] = (panel['flooded_share']>0).astype(int)
panel['flood_intensity'] = panel['flooded_share'] * panel['flood']
all_years_list = [1900,1910,1920,1925,1930,1935,1940,1945,1950,1954,1960,1964,1970]
for y in all_years_list:
    panel[f'f_int_{y}'] = np.where(panel['year']==y, panel['flood_intensity'], 0)
    panel[f'f_bin_{y}'] = np.where(panel['year']==y, panel['flood'], 0)

# State-year dummies
panel['state_year'] = panel['statefips']*10000 + panel['year']
dsy = pd.get_dummies(panel['state_year'], prefix='d_sy').astype(float)
panel = pd.concat([panel.reset_index(drop=True), dsy.reset_index(drop=True)], axis=1)
ds = pd.get_dummies(panel['statefips'], prefix='d_s').astype(float)
panel = pd.concat([panel.reset_index(drop=True), ds.reset_index(drop=True)], axis=1)

# Time-interacted controls
for y in all_years_list:
    panel[f'ld_{y}'] = np.where(panel['year']==y, panel['distance_ms'].fillna(0), 0)
    panel[f'dx_{y}'] = np.where(panel['year']==y, panel['x_centroid'].fillna(0)/1000, 0)
    panel[f'dy_{y}'] = np.where(panel['year']==y, panel['y_centroid'].fillna(0)/1000, 0)
    panel[f'rug_{y}'] = np.where(panel['year']==y, panel['altitude_std_meters'].fillna(0), 0)
    for crop in ['cotton','corn','wheat','oats','rice','scane']:
        sc = f'{crop}_suitability'
        if sc in panel.columns: panel[f'{crop}_s_{y}'] = np.where(panel['year']==y, panel[sc].fillna(0), 0)
for y in all_years_list:
    for var in ['pcpubwor','pcaaa','pcrelief','pcndloan','pcndins']:
        panel[f'ln{var}_{y}'] = np.where(panel['year']==y, np.log(panel[var].clip(lower=1e-10)), 0)

# Lagged values
panel.sort_values(['fips','year'], inplace=True)
for var in ['lnpopulation','lnpopulation_black','lnfrac_black','lnfracfarms_nonwhite']:
    for yv in [1920,1910,1900]:
        ln = f'lc{yv}_{var}'; vay = panel.loc[panel['year']==yv, ['fips',var]].rename(columns={var:ln})
        panel = pd.merge(panel, vay, on='fips', how='left')
    for y in [1930,1935,1940,1945,1950,1954,1960,1964,1970]:
        for yv,lg in {1920:2,1910:3,1900:4}.items():
            panel[f'lag{lg}_{var}_{y}'] = np.where(panel['year']==y, panel[f'lc{yv}_{var}'].fillna(0), 0)
for var in ['lnvalue_equipment','lntractors','lnmules_horses','lnavfarmsize','lnlandbuildingvalue','lnlandbuildingvaluef','lnfarmland_a']:
    for yv in [1925,1920,1910,1900]:
        ln = f'lc{yv}_{var}'; vay = panel.loc[panel['year']==yv, ['fips',var]].rename(columns={var:ln})
        panel = pd.merge(panel, vay, on='fips', how='left')
    for y in [1930,1935,1940,1945,1950,1954,1960,1964,1970]:
        for yv,lg in {1925:1,1920:2,1910:3,1900:4}.items():
            panel[f'lag{lg}_{var}_{y}'] = np.where(panel['year']==y, panel[f'lc{yv}_{var}'].fillna(0), 0)

preanalysis_post1930 = panel[panel['year']>=1920].copy()
print(f"Panel ready: {panel.fips.nunique()} counties, {len(panel)} obs")

# =============================================================================
# SPECIFICATION SEARCH ENGINE
# =============================================================================
print("\n" + "="*70 + "\nSPECIFICATION SEARCH\n" + "="*70)

spec_results = []; inference_results = []; run_counter = 0
GEO = ['cotton_s_*','corn_s_*','ld_*','dx_*','dy_*','rug_*']
SY = ['d_sy_*']
ND = ['lnpcpubwor_*','lnpcaaa_*','lnpcrelief_*','lnpcndloan_*','lnpcndins_*']
G1L = ['lag2_lnfrac_black_*','lag3_lnfrac_black_*','lag4_lnfrac_black_*']
G2L = ['lag1_lnvalue_equipment_*','lag2_lnvalue_equipment_*','lag3_lnvalue_equipment_*','lag4_lnvalue_equipment_*']
G1T = ['f_int_1930','f_int_1940','f_int_1950','f_int_1960','f_int_1970']
G2T = ['f_int_1930','f_int_1935','f_int_1940','f_int_1945','f_int_1950','f_int_1954','f_int_1960','f_int_1964','f_int_1970']

def run_spec(df, dv, tvs, cps, ab, cl, wt, sid, stp, bgid, ft, sd='post1930', cd='', fed='fips', cv_extra=None):
    global run_counter; run_counter += 1; srid = f'{PAPER_ID}__{bgid}__{run_counter:03d}'
    ctrls = get_control_cols(df, cps) if cps else []
    avs = list(set([dv]+tvs+ctrls+([ab] if ab else [])+([cl] if cl else [])+([wt] if wt else [])))
    rd = df.dropna(subset=[v for v in avs if v in df.columns]).copy()
    if len(rd) < 10:
        spec_results.append({'paper_id':PAPER_ID,'spec_run_id':srid,'spec_id':sid,'spec_tree_path':stp,'baseline_group_id':bgid,'outcome_var':dv,'treatment_var':ft,'coefficient':np.nan,'std_error':np.nan,'p_value':np.nan,'ci_lower':np.nan,'ci_upper':np.nan,'n_obs':len(rd),'r_squared':np.nan,'coefficient_vector_json':json.dumps({'error': f'Only {len(rd)} obs'}),'sample_desc':sd,'fixed_effects':fed,'controls_desc':cd,'cluster_var':cl or '','run_success':0,'run_error':f'Only {len(rd)} obs'})
        return None
    rhs = ' + '.join(tvs+ctrls); fm = f'{dv} ~ {rhs} | {ab}' if ab else f'{dv} ~ {rhs}'
    try:
        vc = {'CRV1':cl} if cl else 'hetero'
        mdl = pf.feols(fm, data=rd, vcov=vc, weights=wt)
        co,se,pv,ci = mdl.coef(),mdl.se(),mdl.pvalue(),mdl.confint()
        cd_j = {tv:round(float(co[tv]),8) for tv in tvs if tv in co.index}
        if cv_extra:
            cd_j.update(cv_extra)
        fc = float(co[ft]) if ft in co.index else np.nan; fse = float(se[ft]) if ft in se.index else np.nan
        fp = float(pv[ft]) if ft in pv.index else np.nan
        fcl = float(ci.loc[ft].iloc[0]) if ft in ci.index else np.nan; fcu = float(ci.loc[ft].iloc[1]) if ft in ci.index else np.nan
        spec_results.append({'paper_id':PAPER_ID,'spec_run_id':srid,'spec_id':sid,'spec_tree_path':stp,'baseline_group_id':bgid,'outcome_var':dv,'treatment_var':ft,'coefficient':round(fc,6),'std_error':round(fse,6),'p_value':round(fp,6),'ci_lower':round(fcl,6),'ci_upper':round(fcu,6),'n_obs':int(mdl._N),'r_squared':round(float(mdl._r2),6) if mdl._r2 is not None else np.nan,'coefficient_vector_json':json.dumps(cd_j),'sample_desc':sd,'fixed_effects':fed,'controls_desc':cd,'cluster_var':cl or '','run_success':1,'run_error':''})
        print(f"  [{run_counter:03d}] {sid}: coef({ft})={fc:.4f}, se={fse:.4f}, N={mdl._N}")
        return mdl
    except Exception as e:
        em = str(e)[:200]; print(f"  [{run_counter:03d}] {sid}: FAILED - {em[:80]}")
        spec_results.append({'paper_id':PAPER_ID,'spec_run_id':srid,'spec_id':sid,'spec_tree_path':stp,'baseline_group_id':bgid,'outcome_var':dv,'treatment_var':ft,'coefficient':np.nan,'std_error':np.nan,'p_value':np.nan,'ci_lower':np.nan,'ci_upper':np.nan,'n_obs':np.nan,'r_squared':np.nan,'coefficient_vector_json':json.dumps({'error': em}),'sample_desc':sd,'fixed_effects':fed,'controls_desc':cd,'cluster_var':cl or '','run_success':0,'run_error':em})
        return None

def run_infer(df, dv, tvs, cps, ab, wt, isid, istp, bgid, ft, bsrid, vcspec):
    global run_counter; run_counter += 1; irid = f'{PAPER_ID}__infer__{run_counter:03d}'
    ctrls = get_control_cols(df, cps) if cps else []
    avs = list(set([dv]+tvs+ctrls+([ab] if ab else [])+([wt] if wt else [])))
    rd = df.dropna(subset=[v for v in avs if v in df.columns]).copy()
    if len(rd) < 10:
        inference_results.append({'paper_id':PAPER_ID,'inference_run_id':irid,'spec_run_id':bsrid,'spec_id':isid,'spec_tree_path':istp,'baseline_group_id':bgid,'coefficient':np.nan,'std_error':np.nan,'p_value':np.nan,'ci_lower':np.nan,'ci_upper':np.nan,'n_obs':len(rd),'r_squared':np.nan,'coefficient_vector_json':json.dumps({'error': f'Only {len(rd)} obs'}),'run_success':0,'run_error':f'Only {len(rd)} obs'})
        return
    rhs = ' + '.join(tvs+ctrls); fm = f'{dv} ~ {rhs} | {ab}' if ab else f'{dv} ~ {rhs}'
    try:
        mdl = pf.feols(fm, data=rd, vcov=vcspec, weights=wt)
        co,se,pv,ci = mdl.coef(),mdl.se(),mdl.pvalue(),mdl.confint()
        cd_j = {tv:round(float(co[tv]),8) for tv in tvs if tv in co.index}
        fc = float(co[ft]) if ft in co.index else np.nan; fse = float(se[ft]) if ft in se.index else np.nan
        fp = float(pv[ft]) if ft in pv.index else np.nan
        fcl = float(ci.loc[ft].iloc[0]) if ft in ci.index else np.nan; fcu = float(ci.loc[ft].iloc[1]) if ft in ci.index else np.nan
        inference_results.append({'paper_id':PAPER_ID,'inference_run_id':irid,'spec_run_id':bsrid,'spec_id':isid,'spec_tree_path':istp,'baseline_group_id':bgid,'coefficient':round(fc,6),'std_error':round(fse,6),'p_value':round(fp,6),'ci_lower':round(fcl,6),'ci_upper':round(fcu,6),'n_obs':int(mdl._N),'r_squared':round(float(mdl._r2),6) if mdl._r2 is not None else np.nan,'coefficient_vector_json':json.dumps(cd_j),'run_success':1,'run_error':''})
    except Exception as e:
        inference_results.append({'paper_id':PAPER_ID,'inference_run_id':irid,'spec_run_id':bsrid,'spec_id':isid,'spec_tree_path':istp,'baseline_group_id':bgid,'coefficient':np.nan,'std_error':np.nan,'p_value':np.nan,'ci_lower':np.nan,'ci_upper':np.nan,'n_obs':np.nan,'r_squared':np.nan,'coefficient_vector_json':json.dumps({'error': str(e)[:200]}),'run_success':0,'run_error':str(e)[:200]})

# --- G1: Black Population Share ---
print("\n--- G1: Black Population Share ---")
G1C = SY+GEO+G1L; G1F = 'f_int_1930'
run_spec(preanalysis_post1930, 'lnfrac_black', G1T, G1C, 'fips','fips','county_w', 'baseline','designs/difference_in_differences.md#baseline','G1',G1F, cd='state_year_FE+geo+lags')
run_spec(preanalysis_post1930, 'lnfrac_black', G1T, G1C, 'fips','fips','county_w', 'design/difference_in_differences/estimator/twfe','designs/difference_in_differences.md#estimators','G1',G1F, cd='state_year_FE+geo+lags')
run_spec(preanalysis_post1930, 'lnfrac_black', G1T, [], 'fips','fips','county_w', 'rc/controls/sets/none','modules/robustness/controls.md#standard-control-sets','G1',G1F, cd='none')
run_spec(preanalysis_post1930, 'lnfrac_black', G1T, SY, 'fips','fips','county_w', 'rc/controls/sets/minimal','modules/robustness/controls.md#standard-control-sets','G1',G1F, cd='state_year_FE')
run_spec(preanalysis_post1930, 'lnfrac_black', G1T, G1C+ND, 'fips','fips','county_w', 'rc/controls/sets/extended','modules/robustness/controls.md#standard-control-sets','G1',G1F, cd='state_year_FE+geo+lags+ND')
run_spec(preanalysis_post1930, 'lnfrac_black', G1T, SY+GEO, 'fips','fips','county_w', 'rc/controls/progression/geography','modules/robustness/controls.md#control-progression','G1',G1F, cd='state_year_FE+geo')
run_spec(preanalysis_post1930, 'lnfrac_black', G1T, SY+G1L, 'fips','fips','county_w', 'rc/controls/progression/lags','modules/robustness/controls.md#control-progression','G1',G1F, cd='state_year_FE+lags')
run_spec(preanalysis_post1930, 'lnfrac_black', G1T, SY+ND, 'fips','fips','county_w', 'rc/controls/progression/new_deal','modules/robustness/controls.md#control-progression','G1',G1F, cd='state_year_FE+ND')
run_spec(preanalysis_post1930, 'lnfrac_black', G1T, SY+GEO+G1L+ND, 'fips','fips','county_w', 'rc/controls/progression/full','modules/robustness/controls.md#control-progression','G1',G1F, cd='state_year_FE+geo+lags+ND')
run_spec(preanalysis_post1930, 'lnfrac_black', G1T, GEO+G1L, 'fips','fips','county_w', 'rc/fe/drop/state_year','modules/robustness/fixed_effects.md#dropping-fe','G1',G1F, cd='geo+lags', fed='fips_only')
dn30 = preanalysis_post1930[preanalysis_post1930['year']!=1930].copy()
run_spec(dn30, 'lnfrac_black', ['f_int_1940','f_int_1950','f_int_1960','f_int_1970'], G1C, 'fips','fips','county_w', 'rc/sample/time/drop_1930','modules/robustness/sample.md#time-restrictions','G1','f_int_1940', sd='post1930_no1930', cd='state_year_FE+geo+lags')
dn70 = preanalysis_post1930[preanalysis_post1930['year']!=1970].copy()
run_spec(dn70, 'lnfrac_black', ['f_int_1930','f_int_1940','f_int_1950','f_int_1960'], G1C, 'fips','fips','county_w', 'rc/sample/time/drop_1970','modules/robustness/sample.md#time-restrictions','G1',G1F, sd='post1930_no1970', cd='state_year_FE+geo+lags')
dsh = preanalysis_post1930[preanalysis_post1930['year']<=1950].copy()
run_spec(dsh, 'lnfrac_black', ['f_int_1930','f_int_1940','f_int_1950'], G1C, 'fips','fips','county_w', 'rc/sample/time/short_window_1930_1950','modules/robustness/sample.md#time-restrictions','G1',G1F, sd='1920-1950', cd='state_year_FE+geo+lags')
q01=preanalysis_post1930['lnfrac_black'].quantile(0.01); q99=preanalysis_post1930['lnfrac_black'].quantile(0.99)
dtr = preanalysis_post1930[(preanalysis_post1930['lnfrac_black']>=q01)&(preanalysis_post1930['lnfrac_black']<=q99)].copy()
run_spec(dtr, 'lnfrac_black', G1T, G1C, 'fips','fips','county_w', 'rc/sample/outliers/trim_y_1_99','modules/robustness/sample.md#outliers','G1',G1F, sd='post1930_trimmed', cd='state_year_FE+geo+lags')
run_spec(preanalysis_post1930, 'frac_black_level', G1T, G1C, 'fips','fips','county_w', 'rc/form/outcome/level','modules/robustness/functional_form.md#outcome-transformations','G1',G1F, cd='state_year_FE+geo+lags', cv_extra={'functional_form': {'target':'outcome','operation':'transform','outcome_transform':'level','baseline_outcome_var':'lnfrac_black','new_outcome_var':'frac_black_level','interpretation':'Outcome in levels instead of log; coefficient interpreted in level units.'}})
run_spec(preanalysis_post1930, 'lnfrac_black', ['f_bin_1930','f_bin_1940','f_bin_1950','f_bin_1960','f_bin_1970'], G1C, 'fips','fips','county_w', 'rc/form/treatment/binary_flood','modules/robustness/functional_form.md#treatment-transformations','G1','f_bin_1930', cd='state_year_FE+geo+lags', cv_extra={'functional_form': {'target':'treatment','operation':'binarize','source_var':'flooded_share','baseline_treatment_family':'f_int_{year}','new_var_family':'f_bin_{year}','recode_rule':'flood = 1[flooded_share > 0]; f_bin_{y} = flood * 1[year==y]','threshold':0,'direction':'>','units':'share (0-1)','interpretation':'Extensive-margin dynamic effects (flooded vs not flooded) rather than per-unit intensity.'}})
run_spec(preanalysis_post1930, 'lnfrac_black', G1T, G1C, 'fips','fips',None, 'rc/weights/main/unweighted','modules/robustness/weights.md#main-weight-choices','G1',G1F, cd='state_year_FE+geo+lags')
run_spec(preanalysis_post1930, 'lnfrac_black', G1T, G1C, 'fips','fips','pop_w_1920', 'rc/weights/main/population_1920','modules/robustness/weights.md#main-weight-choices','G1',G1F, cd='state_year_FE+geo+lags')
run_spec(preanalysis_post1930, 'lnfrac_black', G1T, G1C, 'fips','fips','county_w', 'baseline__f_int_1950','designs/difference_in_differences.md#baseline','G1','f_int_1950', cd='state_year_FE+geo+lags')
run_spec(preanalysis_post1930, 'lnfrac_black', G1T, G1C, 'fips','fips',None, 'rc/weights/main/unweighted__f_int_1950','modules/robustness/weights.md#main-weight-choices','G1','f_int_1950', cd='state_year_FE+geo+lags')
run_spec(preanalysis_post1930, 'lnfrac_black', G1T, G1C+ND, 'fips','fips','county_w', 'rc/controls/sets/extended__f_int_1950','modules/robustness/controls.md#standard-control-sets','G1','f_int_1950', cd='state_year_FE+geo+lags+ND')
run_spec(preanalysis_post1930, 'lnfrac_black', G1T, [], 'fips','fips','county_w', 'rc/controls/sets/none__f_int_1950','modules/robustness/controls.md#standard-control-sets','G1','f_int_1950', cd='none')
run_spec(preanalysis_post1930, 'lnpopulation_black', G1T, SY+GEO+['lag2_lnpopulation_black_*','lag3_lnpopulation_black_*','lag4_lnpopulation_black_*'], 'fips','fips','county_w', 'rc/form/outcome/alt_pop_black','modules/robustness/functional_form.md#outcome-transformations','G1',G1F, cd='state_year_FE+geo+pop_black_lags', cv_extra={'functional_form': {'target':'outcome','operation':'operationalization_swap','baseline_outcome_var':'lnfrac_black','new_outcome_var':'lnpopulation_black','interpretation':'Alternative outcome operationalization (log Black population level) used to probe denominator sensitivity vs log share.'}})

# --- G2: Farm Equipment Value ---
print("\n--- G2: Farm Equipment Value ---")
G2C = SY+GEO+G2L; G2F = 'f_int_1940'
run_spec(preanalysis_post1930, 'lnvalue_equipment', G2T, G2C, 'fips','fips','county_w', 'baseline','designs/difference_in_differences.md#baseline','G2',G2F, cd='state_year_FE+geo+equip_lags')
run_spec(preanalysis_post1930, 'lnvalue_equipment', G2T, G2C, 'fips','fips','county_w', 'design/difference_in_differences/estimator/twfe','designs/difference_in_differences.md#estimators','G2',G2F, cd='state_year_FE+geo+equip_lags')
run_spec(preanalysis_post1930, 'lnvalue_equipment', G2T, [], 'fips','fips','county_w', 'rc/controls/sets/none','modules/robustness/controls.md#standard-control-sets','G2',G2F, cd='none')
run_spec(preanalysis_post1930, 'lnvalue_equipment', G2T, SY, 'fips','fips','county_w', 'rc/controls/sets/minimal','modules/robustness/controls.md#standard-control-sets','G2',G2F, cd='state_year_FE')
run_spec(preanalysis_post1930, 'lnvalue_equipment', G2T, G2C+ND, 'fips','fips','county_w', 'rc/controls/sets/extended','modules/robustness/controls.md#standard-control-sets','G2',G2F, cd='state_year_FE+geo+equip_lags+ND')
run_spec(preanalysis_post1930, 'lnvalue_equipment', G2T, SY+GEO, 'fips','fips','county_w', 'rc/controls/progression/geography','modules/robustness/controls.md#control-progression','G2',G2F, cd='state_year_FE+geo')
run_spec(preanalysis_post1930, 'lnvalue_equipment', G2T, SY+G2L, 'fips','fips','county_w', 'rc/controls/progression/lags','modules/robustness/controls.md#control-progression','G2',G2F, cd='state_year_FE+equip_lags')
run_spec(preanalysis_post1930, 'lnvalue_equipment', G2T, SY+ND, 'fips','fips','county_w', 'rc/controls/progression/new_deal','modules/robustness/controls.md#control-progression','G2',G2F, cd='state_year_FE+ND')
run_spec(preanalysis_post1930, 'lnvalue_equipment', G2T, SY+GEO+G2L+ND, 'fips','fips','county_w', 'rc/controls/progression/full','modules/robustness/controls.md#control-progression','G2',G2F, cd='state_year_FE+geo+equip_lags+ND')
run_spec(preanalysis_post1930, 'lnvalue_equipment', G2T, GEO+G2L, 'fips','fips','county_w', 'rc/fe/drop/state_year','modules/robustness/fixed_effects.md#dropping-fe','G2',G2F, cd='geo+equip_lags', fed='fips_only')
dn30g2 = preanalysis_post1930[preanalysis_post1930['year']!=1930].copy()
run_spec(dn30g2, 'lnvalue_equipment', [t for t in G2T if t!='f_int_1930'], G2C, 'fips','fips','county_w', 'rc/sample/time/drop_first_post','modules/robustness/sample.md#time-restrictions','G2',G2F, sd='post1930_no1930', cd='state_year_FE+geo+equip_lags')
dshg2 = preanalysis_post1930[preanalysis_post1930['year']<=1950].copy()
run_spec(dshg2, 'lnvalue_equipment', [t for t in G2T if int(t.split('_')[-1])<=1950], G2C, 'fips','fips','county_w', 'rc/sample/time/short_window','modules/robustness/sample.md#time-restrictions','G2',G2F, sd='1920-1950', cd='state_year_FE+geo+equip_lags')
q01e=preanalysis_post1930['lnvalue_equipment'].quantile(0.01); q99e=preanalysis_post1930['lnvalue_equipment'].quantile(0.99)
dtre = preanalysis_post1930[(preanalysis_post1930['lnvalue_equipment']>=q01e)&(preanalysis_post1930['lnvalue_equipment']<=q99e)].copy()
run_spec(dtre, 'lnvalue_equipment', G2T, G2C, 'fips','fips','county_w', 'rc/sample/outliers/trim_y_1_99','modules/robustness/sample.md#outliers','G2',G2F, sd='post1930_trimmed', cd='state_year_FE+geo+equip_lags')
run_spec(preanalysis_post1930, 'value_equipment_level', G2T, G2C, 'fips','fips','county_w', 'rc/form/outcome/level','modules/robustness/functional_form.md#outcome-transformations','G2',G2F, cd='state_year_FE+geo+equip_lags', cv_extra={'functional_form': {'target':'outcome','operation':'transform','outcome_transform':'level','baseline_outcome_var':'lnvalue_equipment','new_outcome_var':'value_equipment_level','interpretation':'Outcome in levels instead of log; coefficient interpreted in level units.'}})
run_spec(preanalysis_post1930, 'lnvalue_equipment', ['f_bin_1930','f_bin_1935','f_bin_1940','f_bin_1945','f_bin_1950','f_bin_1954','f_bin_1960','f_bin_1964','f_bin_1970'], G2C, 'fips','fips','county_w', 'rc/form/treatment/binary_flood','modules/robustness/functional_form.md#treatment-transformations','G2','f_bin_1940', cd='state_year_FE+geo+equip_lags', cv_extra={'functional_form': {'target':'treatment','operation':'binarize','source_var':'flooded_share','baseline_treatment_family':'f_int_{year}','new_var_family':'f_bin_{year}','recode_rule':'flood = 1[flooded_share > 0]; f_bin_{y} = flood * 1[year==y]','threshold':0,'direction':'>','units':'share (0-1)','interpretation':'Extensive-margin dynamic effects (flooded vs not flooded) rather than per-unit intensity.'}})
run_spec(preanalysis_post1930, 'lnvalue_equipment', G2T, G2C, 'fips','fips',None, 'rc/weights/main/unweighted','modules/robustness/weights.md#main-weight-choices','G2',G2F, cd='state_year_FE+geo+equip_lags')
run_spec(preanalysis_post1930, 'lnvalue_equipment', G2T, G2C, 'fips','fips','pop_w_1920', 'rc/weights/main/population_1920','modules/robustness/weights.md#main-weight-choices','G2',G2F, cd='state_year_FE+geo+equip_lags')
run_spec(preanalysis_post1930, 'lnvalue_equipment', G2T, G2C, 'fips','fips','county_w', 'baseline__f_int_1930','designs/difference_in_differences.md#baseline','G2','f_int_1930', cd='state_year_FE+geo+equip_lags')
run_spec(preanalysis_post1930, 'lnvalue_equipment', G2T, G2C, 'fips','fips','county_w', 'baseline__f_int_1970','designs/difference_in_differences.md#baseline','G2','f_int_1970', cd='state_year_FE+geo+equip_lags')
run_spec(preanalysis_post1930, 'lnvalue_equipment', G2T, G2C, 'fips','fips',None, 'rc/weights/main/unweighted__f_int_1970','modules/robustness/weights.md#main-weight-choices','G2','f_int_1970', cd='state_year_FE+geo+equip_lags')
run_spec(preanalysis_post1930, 'lnvalue_equipment', G2T, G2C+ND, 'fips','fips','county_w', 'rc/controls/sets/extended__f_int_1970','modules/robustness/controls.md#standard-control-sets','G2','f_int_1970', cd='state_year_FE+geo+equip_lags+ND')
run_spec(preanalysis_post1930, 'lnvalue_equipment', G2T, [], 'fips','fips','county_w', 'rc/controls/sets/none__f_int_1970','modules/robustness/controls.md#standard-control-sets','G2','f_int_1970', cd='none')
run_spec(preanalysis_post1930, 'lntractors', G2T, SY+GEO+['lag1_lntractors_*'], 'fips','fips','county_w', 'rc/form/outcome/alt_tractors','modules/robustness/functional_form.md#outcome-transformations','G2',G2F, cd='state_year_FE+geo+tractor_lags', cv_extra={'functional_form': {'target':'outcome','operation':'operationalization_swap','baseline_outcome_var':'lnvalue_equipment','new_outcome_var':'lntractors','interpretation':'Alternative mechanization proxy (log tractors) used as a robustness check vs equipment value.'}})

# INFERENCE VARIANTS
print("\n--- Inference Variants ---")
bg1id = [r['spec_run_id'] for r in spec_results if r['baseline_group_id']=='G1' and r['spec_id']=='baseline'][0]
bg2id = [r['spec_run_id'] for r in spec_results if r['baseline_group_id']=='G2' and r['spec_id']=='baseline'][0]
run_infer(preanalysis_post1930,'lnfrac_black',G1T,G1C,'fips','county_w','infer/se/hc/hc1','modules/inference/standard_errors.md#robust-se','G1',G1F,bg1id,'hetero')
run_infer(preanalysis_post1930,'lnfrac_black',G1T,G1C,'fips','county_w','infer/se/cluster/state','modules/inference/standard_errors.md#clustering','G1',G1F,bg1id,{'CRV1':'statefips'})
run_infer(preanalysis_post1930,'lnvalue_equipment',G2T,G2C,'fips','county_w','infer/se/hc/hc1','modules/inference/standard_errors.md#robust-se','G2',G2F,bg2id,'hetero')
run_infer(preanalysis_post1930,'lnvalue_equipment',G2T,G2C,'fips','county_w','infer/se/cluster/state','modules/inference/standard_errors.md#clustering','G2',G2F,bg2id,{'CRV1':'statefips'})

# OUTPUT
print("\n" + "="*70 + "\nOUTPUT\n" + "="*70)
df_spec = pd.DataFrame(spec_results); df_spec.to_csv(f'{OUT_DIR}/specification_results.csv', index=False)
print(f"Wrote {len(df_spec)} rows to specification_results.csv")
print(f"  Successes: {df_spec['run_success'].sum()}, Failures: {(df_spec['run_success']==0).sum()}")
print(f"  G1: {len(df_spec[df_spec['baseline_group_id']=='G1'])}, G2: {len(df_spec[df_spec['baseline_group_id']=='G2'])}")
if inference_results:
    df_inf = pd.DataFrame(inference_results); df_inf.to_csv(f'{OUT_DIR}/inference_results.csv', index=False)
    print(f"Wrote {len(df_inf)} rows to inference_results.csv")
print("Done.")
