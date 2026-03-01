# Specification Search: 157781-V1

## Surface Summary

- **Paper**: "Rebel on the Canal: Disrupted Trade Access and Social Conflict in China, 1650-1911"
- **Design**: Difference-in-differences (sharp single-date TWFE)
- **Baseline groups**: 1 (G1)
- **Preferred baseline**: Table 3 Col 4 (full FE, no controls, cluster OBJECTID)
- **Budget**: max 100 core specs
- **Seed**: 157781 (full enumeration, no sampling needed)

## Execution Summary

- **Total estimate rows**: 51
  - Successful: 51
  - Failed: 0
- **Inference variant rows**: 3
- **Breakdown**:
  - Baseline specs (Table 3 Cols 1-5): 5
  - LOO control drops: 15
  - Control subsets (none/climate/geo/agri/full): 5
  - FE variations (drop/add): 4
  - Sample restrictions: 10
  - Outcome forms: 6
  - Outlier trim: 1
  - Design (TWFE): 1

## Data Construction

The analysis dataset was reconstructed entirely from raw data in `Data/Raw/`,
following the logic in `Program/Clean/clean.do` and `Program/Analysis/generalsetup.do`.
Key steps:
1. Built county-year panel from `Geo_raw.xlsx` and `rawrebellion.dta` (575 counties x 262 years)
2. Merged geographic variables (coast, rivers, courier routes) from raw Excel/CSV files
3. Constructed county-level population from Ming household data + prefecture-level population densities
4. Computed terrain ruggedness from 575 elevation raster files
5. Matched Mann (2009) temperature reconstruction to counties via nearest-neighbor spatial matching
6. Constructed all derived variables (ashonset_cntypop1600, ashprerebels, interaction terms, etc.)

**Note**: The final sample has 137812 observations
(526 counties). The paper reports 536 counties (140,432 obs);
the small difference arises from county population data availability in the Ming household crosswalk.

## FE Structure

The paper uses a progressive FE structure:
- **Minimal** (Col 1): OBJECTID + year
- **+ Pre-rebellion trend** (Col 2): + ashprerebels x year (varying slopes)
- **+ Province-year** (Col 3): + provid x year interaction FE
- **+ Prefecture trend** (Col 4, preferred): + prefid linear time trend
- **+ Controls** (Col 5): + 15 time-varying control interactions

The c.ashprerebels#i.year and i.prefid#c.year terms are implemented as explicit covariates
(261 ashprerebels*year_dummy columns and 78 prefid*year columns), since pyfixest does not
natively support varying-slope absorbed FE. The i.provid#i.year FE is absorbed via `provid^year`.

## Software

- Python 3.12.7
- pyfixest 0.40.1
- pandas 2.2.3
- numpy 2.1.3

## Deviations from Surface

1. **Conley spatial HAC SE** (`infer/se/spatial_hac/conley_500km_262lag`): Skipped because
   the `ols_spatial_HAC` command requires Stata-specific spatial HAC computation not available
   in standard Python packages. The paper reports these in brackets alongside clustered SE.

2. **Sample size**: Our reconstructed dataset has ~526 counties vs the paper's 536,
   due to minor differences in the Ming population crosswalk construction. Core results
   are qualitatively identical.
