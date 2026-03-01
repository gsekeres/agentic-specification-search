# Specification Surface: 112749-V1

## Paper: "When the Levee Breaks: Black Migration and Economic Development in the American South"
**Authors**: Hornbeck & Naidu, AER 2014

## Design Classification
**Primary**: `panel_fixed_effects` (county panel 1900-1970 with county FE, state-year FE)

The paper exploits the Great Mississippi Flood of 1927 as a natural experiment. Counties with more area flooded experienced larger Black out-migration, which in turn affected agricultural technology adoption. The main specification is a panel FE regression with flood intensity x year dummies as treatment variables.

## Baseline Groups

### G1: Black Labor / Population Outcomes (Table 2)
- **Claim**: Flooded counties experienced larger declines in Black population share and Black population after the flood
- **Outcome variables**: `lnfrac_black`, `lnpopulation_black`, `lnpopulation`, `lnfracfarms_nonwhite`
- **Treatment**: `f_int_YEAR` (flood_intensity x year dummies for 1930, 1940, 1950, 1960, 1970)
- **Key controls**: State-year FE (d_sy_*), crop suitability x year, distance to MS x year, lat/lon x year, ruggedness x year, lagged DV values (1920, 1910, 1900)
- **FE**: County (fips), absorbed via `areg`
- **Clustering**: County (fips)
- **Weights**: County area (county_w = county_acres in 1920)

### G2: Agricultural Capital Outcomes (Table 4)
- **Claim**: Flooded counties experienced shifts in agricultural technology -- larger farms, more tractors/equipment, fewer mules/horses
- **Outcome variables**: `lnavfarmsize`, `lnvalue_equipment`, `lntractors`, `lnmules_horses`
- **Treatment**: Same as G1 but for ag census years (1930, 1935, 1940, 1945, 1950, 1954, 1960, 1964, 1970)

## Specification Axes

### Controls variation
1. **Baseline**: State-year FE + geography (crop suitability, distance, lat/lon, ruggedness) + lagged DVs
2. **Add New Deal spending**: +lnpcpubwor_*, lnpcaaa_*, lnpcrelief_*, lnpcndloan_*, lnpcndins_*
3. **Add tenancy/manufacturing**: +lag*_lnfarms_nonwhite_t_*, lag*_lnmfgestab_*, lag*_lnmfgavewages_*
4. **Add plantation interaction**: +plantation_*
5. **Add propensity score**: +prop_plant_flstate_*
6. **Drop geography**: Remove cotton_s_*, corn_s_*, ld_*, dx_*, dy_*, rug_*
7. **Drop lagged DVs**: Remove lag2_*, lag3_*, lag4_*
8. **Lagged DVs only (no geography)**: State-year FE + lagged DVs only

### Treatment variation
1. **Baseline**: flood_intensity = flooded_share x flood indicator
2. **Red Cross acreage**: Alternative flood measure based on Red Cross flooded acres/county area
3. **Red Cross population**: Alternative based on pop_affected/population
4. **Red Cross agricultural**: Alternative based on agricultural_flooded_acres/farmland

### Sample variation
1. **Baseline**: Main sample (MS River counties, frac_black >= 0.10, cotton >= 0.15 in 1920)
2. **Flood counties only**: Keep only counties with percent_flood > 0
3. **Drop outliers**: Trim extreme flood_intensity values

### Functional form
1. **Baseline**: Log outcomes
2. **Level outcomes**: Use level instead of log

### Weights
1. **Baseline**: county_w (area-weighted)
2. **Unweighted**: No weights

### Inference
1. **Canonical**: Cluster at county (fips)
2. **Robust (HC1)**: Heteroskedasticity-robust without clustering
3. **State cluster**: Cluster at state level

## Budget
- Target: 50-60 specifications across both baseline groups
- G1 (labor): ~35 specs
- G2 (capital): ~20 specs
- Full enumeration of the revealed control/treatment/sample/weight axes

## What is excluded
- Table 1 (pre-differences) -- balance checks, not main estimates
- Table 3 (migration) -- uses different individual-level data
- Tables 6-7 (other Southern rivers, non-flooded distance) -- different samples/treatment variables
- Robustness Tables (RefTable 1-4) -- supplementary, covered by our specification axes
- Conley SE section -- spatial SE as inference variant only
- Plantation interaction regressions -- covered as an rc axis
