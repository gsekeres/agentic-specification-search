# Specification Search Report: 112749-V1

## Paper
Hornbeck & Naidu (2014), "When the Levee Breaks: Black Migration and Economic Development in the American South", AER 104(3): 963-990.

## Surface Summary
- **Baseline groups**: 2 (G1: Black population share, G2: Agricultural capital intensity)
- **Design**: Continuous-treatment DiD with county FE (absorbed) + state-by-year FE (absorbed)
- **Sampling**: Full enumeration (block-based control progressions, not combinatorial)
- **Seed**: G1=112749001, G2=112749002 (not used; full enumeration)
- **Budget**: G1 max 80, G2 max 85

## Execution Summary
- **Date**: 2026-02-13 11:36
- **Total specs**: 72
- **G1**: 35 | **G2**: 37
- **Baselines**: 4 | **RC**: 56 | **Infer**: 12
- **Succeeded**: 57 | **Failed**: 15
- **Elapsed**: 21.0s

## Baseline Results

### G1: lnfrac_black
- Table 2 Col 1 (geo+lags): focal f_int_1950 coef ~ -0.202, N=978
- Table 2 Col 2 (geo+lags+ND): focal f_int_1950 coef ~ -0.240, N=978

### G2: lnvalue_equipment
- Note: Equipment data only available for years 1920, 1925, 1930, 1940, 1970 (5 of 13 panel years)
- Table 4 Col 3 (geo+lags): focal f_int_1940 (peak available post-flood year), N=815
- Table 4 Col 4 (geo+lags+ND): focal f_int_1940, N=815
- G2 focal changed to f_int_1940 because f_int_1950 has zero variation in equipment-available sample

## Spec Categories

### Control Progressions (rc/controls/progression/*)
1. `none` -- FE only (stress test)
2. `lagged_dv_only` -- lagged DV (RefTable step 1)
3. `geography_only` -- geography only (stress test)
4. `geography_and_lags` -- = baseline 1 (RefTable step 2)
5. `geography_lags_tenancy_mfg` -- + tenancy/mfg (RefTable step 3)
6. `geography_lags_newdeal` -- = baseline 2 (RefTable step 4)
7. `geography_lags_newdeal_plantation` -- + plantation (RefTable step 5)
8. `geography_lags_newdeal_tenancy_mfg` -- step 6
9. `geography_lags_newdeal_tenancy_mfg_propscore` -- step 7

### LOO Blocks (rc/controls/loo_block/*)
Relative to baseline 2: drop_geography, drop_lagged_dv, drop_new_deal, drop_crop_suitability, drop_distance_ms, drop_coordinates, drop_ruggedness, drop_tenancy_mfg.

### Weight Variants
- `rc/weights/main/unweighted` -- no analytic weights

### Sample Restrictions
- `rc/sample/time/drop_1970`, `drop_1930`
- `rc/sample/time/pre1960_only` (G2 only)
- `rc/sample/outliers/trim_treatment_p95`

### Treatment Form
- `rc/form/treatment/alt_measure_redcross_acres` (f2_int)
- `rc/form/treatment/alt_measure_redcross_people` (f3_int)

### Inference
- `infer/se/cluster/unit` (county)
- `infer/se/hc/hc1` (heteroskedasticity-robust)
- `infer/se/cluster/state`
- `infer/se/spatial/conley_50km`, `conley_100km`, `conley_200km` (coef only; SE=N/A)

## Failure Details (15 of 72 specs)

### G1 Failures (3 specs)
All G1 failures occur with the New Deal control block on restricted samples:
- `drop_1970 (geo+lags+ND)`: 815 obs with 25 year-interacted ND controls causes singularity.
- `drop_1930 (geo+lags+ND)`: Same issue.
- `trim_treatment_p95 (geo+lags+ND)`: 960 obs with full ND controls causes singularity.
- The same specs without New Deal controls succeed.

### G2 Failures (12 specs)
G2 equipment data available for only 5 of 13 panel years (1920, 1925, 1930, 1940, 1970), severely constraining degrees of freedom:
- **Baseline 2 (geo+lags+ND)**: Too many year-interacted controls for available dof.
- **geography_lags_tenancy_mfg, geography_lags_newdeal_plantation**: Additional control blocks push past singularity threshold.
- **LOO: drop_distance_ms, drop_ruggedness, drop_tenancy_mfg**: Residual collinearity.
- **Unweighted (both)**: Removing weights changes effective sample, triggering singularity.
- **trim_P95 + ND, RedCross acres + ND, RedCross people (both)**: Restricted treatment measures with many controls.

All failures are legitimate collinearity issues from having only 5 time periods for G2's DV.

## Deviations

1. **G2 focal parameter**: Changed from f_int_1950 to f_int_1940 because equipment data is only available for 1920/1925/1930/1940/1970. f_int_1950 has zero variation in the equipment-available sample.

2. **Conley SE**: Not computable in pyfixest. Point estimates from baseline recorded; SEs = NaN.

3. **State-year FE**: Absorbed via `| fips + state_year_fe` (pyfixest multi-way FE absorption) instead of including d_sy_* dummies on RHS. Numerically equivalent but avoids collinearity with year-interacted controls.

4. **G2 inference variants**: Fall back to geo+lags controls (baseline 1) when geo+lags+ND (baseline 2) is singular.

5. **G2 New Deal controls**: Many G2 specs with ND controls fail due to collinearity. The original Stata code ran with all 9 post-treatment periods (1930-1970 including intercensus years); our Python data generation only recovers equipment data for 5 years.

## Software
- Python 3.x, pyfixest (TWFE with multi-way absorbed FE, CRV1), pandas, numpy, statsmodels (probit for propensity score)
