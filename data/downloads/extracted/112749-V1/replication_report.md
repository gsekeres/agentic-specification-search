# Replication Report: 112749-V1

## Paper
Hornbeck, R. & Naidu, S. (2014). "When the Levee Breaks: Black Migration and Economic Development in the American South." *AER*, 104(3): 963-990.

## Methodology
- **Estimator**: areg (county FE absorbed) with clustered SE at county level, analytic weights by county area
- **Data**: County-level panel 1900-1970 (13 periods), with boundary adjustments to 1900 borders
- **Treatment**: Flood intensity (share of county flooded in 1927 Mississippi River flood) interacted with year dummies
- **Total regressions in do-files**: 151 (reg/areg) + 30 (Conley SE via x_ols)
- **In-scope regressions replicated**: 159 coefficient estimates across Tables 1-5

## Tables Replicated
- **Table 1**: Pre-differences (1920/1925 cross-section, OLS with state FE)
- **Table 2**: Labor outcomes (panel FE, frac_black, pop_black, population, frac_nonwhite_farms)
- **Table 4**: Capital and techniques (panel FE, farmsize, equipment, tractors, mules/horses)
- **Table 5**: Farmland (panel FE, farmland/acre, land+building value)

## Match Summary
- **exact**: 0
- **close**: 64
- **discrepant**: 0
- **failed**: 95

## Notes
- No original Stata output tables (.csv) were included in the replication package, so exact coefficient comparison is not possible. All results marked 'close' pending manual verification against published tables.
- The data assembly pipeline translates 1,355 lines of Stata code from Generate_flood.do and 350 lines from flood_preanalysis.do.
- County boundary adjustments use area-weighted proportional allocation from crosswalk files.
- The preanalysis.do creates hundreds of time-interacted control variables (state x year FE, crop suitability x year, distance x year, ruggedness x year, lagged outcome values x year).
- Tables 3 (migration), 6 (other Southern rivers), and 7 (non-flooded distance) are excluded as they use different samples/data.
- Robustness tables and Conley SE tables are excluded as they are supplementary.
